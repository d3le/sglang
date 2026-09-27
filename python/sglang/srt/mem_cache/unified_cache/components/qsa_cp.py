"""The QSA compressed page component of the unified radix tree.

Stage 2 of ``docs/qsa-logical-compressed-mapping.md``: every radix node owns
the compressed pages covering its own token range ("node boundary == page
ownership boundary"; node keys are page aligned by the tree, and the QSA
page is exactly one tree page).  A matching request reads the node chain's
pages (immutable, shared) followed by its own writable tail; insert,
split, recovery, and eviction move the ownership at those boundaries only.

The component is passive with respect to eviction: compressed pages are 1:1
with full-KV pages, so a node's compressed pages are freed exactly when the
FULL component's value is evicted (cascade with equal priority), and request
tails are protected by the same path lock that protects their full-KV slots.
It therefore takes no component lock of its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.qsa_compressed_cache import (
    QSA_VALUE_PATH_VERSION,
    attach_qsa_compressed_cache,
)
from sglang.srt.mem_cache.unified_cache.cache_action import (
    FreeComponentDeviceSlot,
)
from sglang.srt.mem_cache.unified_cache.components.base import (
    ComponentType,
    EvictLayer,
    TreeComponent,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_radix_cache import (
        NodeId,
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


class QSACompressedComponent(TreeComponent):
    component_type = ComponentType.QSA_CP

    def __init__(self, cache: "UnifiedRadixCache", params) -> None:
        super().__init__(cache, params)
        kv_pool = cache.token_to_kv_pool_allocator.get_kvcache()
        self.page_cache = attach_qsa_compressed_cache(
            cache.req_to_token_pool, kv_pool
        )
        self.page_cache.claim_radix()
        self.page_size = self.page_cache.page_size
        if params.page_size != self.page_size:
            raise RuntimeError(
                "QSA compressed pages and the radix tree must share the page "
                f"size: tree={params.page_size}, compressed={self.page_size}"
            )

    @property
    def value_path_version(self) -> int:
        return QSA_VALUE_PATH_VERSION

    # ---- match ----

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        ct = self.component_type
        version = self.value_path_version

        def valid(node: UnifiedTreeNode) -> bool:
            cd = node.component_data[ct]
            if cd.value is None:
                return False
            return cd.metadata.get("value_path_version") == version

        return valid

    def refresh_lru(self, phase, node, root_node) -> None:
        # The LRU is maintained by set_component_device_value for the tree
        # sanity check; eviction never walks it (passive component).
        return

    def _dec_session_coverage(self, session_id: str, leaf) -> None:
        raise NotImplementedError(
            "QSA compressed pages do not support session radix caches"
        )

    def _advance_session_coverage(self, session_id: str, leaf, old_ancestor) -> None:
        raise NotImplementedError(
            "QSA compressed pages do not support session radix caches"
        )

    def _recede_session_coverage(self, session_id: str, leaf, fallback) -> None:
        raise NotImplementedError(
            "QSA compressed pages do not support session radix caches"
        )

    def finalize_match_result_in_cache(
        self, params: MatchPrefixParams, result: MatchResult
    ) -> MatchResult:
        req = params.req
        if req is None:
            return result
        ct = self.component_type
        root = self.tree_core.root_node
        node = self.tree_core.node_by_id(result.best_match_node)
        chunks: list[torch.Tensor] = []
        cur = node
        while cur is not None and cur is not root:
            value = cur.component_data[ct].value
            if value is None:
                raise RuntimeError(
                    "QSA compressed node chain broke at node "
                    f"{cur.id}: matched node without compressed pages"
                )
            chunks.append(value)
            cur = cur.parent
        chunks.reverse()
        device = self.page_cache.req_pages.device
        if chunks:
            pages = torch.cat(chunks).to(dtype=torch.int32)
        else:
            pages = torch.empty(0, dtype=torch.int32, device=device)
        shared_tokens = int(result.device_indices.numel())
        if pages.numel() * self.page_size != shared_tokens:
            raise RuntimeError(
                "QSA compressed match does not cover the full-KV prefix: "
                f"pages={pages.numel()} * page_size={self.page_size} != "
                f"full_kv_hit={shared_tokens}"
            )
        self.page_cache.stage_shared(req, pages, shared_tokens)
        return result

    # ---- insert ----

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        row = req.kv.req_pool_idx
        if row is None:
            return None
        page_aligned = (int(token_ids_len) // self.page_size) * self.page_size
        if page_aligned <= 0:
            return None
        insert_params.qsa_row = int(row)
        insert_params.qsa_value_path_version = self.value_path_version
        return page_aligned

    def update_component_on_insert_overlap(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
        result: InsertResult,
        cache_actions: list,
    ) -> int:
        # The node already owns this range: the request's own pages there are
        # duplicates and go back to the allocator.
        pages = self._detach(params, total_prefix_len, total_prefix_len + prefix_len)
        if pages is not None and pages.numel():
            cache_actions.append(
                FreeComponentDeviceSlot([pages], component_type=self.component_type)
            )
        return prefix_len

    def recover_after_unevict(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        params: InsertParams,
        result: InsertResult,
        cache_actions: list,
    ) -> None:
        # The evicted node is restored from the request's fresh KV; its
        # compressed pages come from the request's own pages for the same span.
        pages = self._detach(params, total_prefix_len, total_prefix_len + prefix_len)
        if pages is None or pages.numel() == 0:
            raise RuntimeError(
                "QSA compressed recovery found no request-owned pages for "
                f"node {node.id} span ({total_prefix_len}, "
                f"{total_prefix_len + prefix_len})"
            )
        self._set_pages(node, pages, params.qsa_value_path_version)

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
        cache_actions: list,
    ) -> None:
        if not is_new_leaf:
            return
        key_len = len(node.key)
        total = 0 if params.key is None else len(params.key)
        start = total - key_len
        pages = self._detach(params, start, total)
        if pages is None or pages.numel() * self.page_size != key_len:
            raise RuntimeError(
                "QSA compressed insert cannot cover the new leaf: "
                f"node={node.id}, key_len={key_len}, "
                f"pages={0 if pages is None else pages.numel()}"
            )
        self._set_pages(node, pages, params.qsa_value_path_version)

    def _detach(
        self, params: InsertParams, token_start: int, token_end: int
    ) -> Optional[torch.Tensor]:
        row = params.qsa_row
        if row is None or row < 0:
            return None
        return self.page_cache.detach_own(row, token_start, token_end)

    def _set_pages(
        self, node: UnifiedTreeNode, pages: torch.Tensor, version: int
    ) -> None:
        ct = self.component_type
        cd = node.component_data[ct]
        cd.metadata["value_path_version"] = int(version)
        # set_component_device_value maintains both the ledger and the device
        # LRU membership the tree sanity check requires of aux components.
        self.tree_core.set_component_device_value(
            node.id, ct, pages.to(dtype=torch.int32).clone()
        )

    # ---- lifecycle ----

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ) -> None:
        ct = self.component_type
        child_cd = child.component_data[ct]
        parent_cd = new_parent.component_data[ct]
        parent_cd.metadata = dict(child_cd.metadata)
        if child_cd.value is None:
            return
        split_pages = len(new_parent.key) // self.page_size
        if split_pages <= 0 or split_pages > int(child_cd.value.numel()):
            raise RuntimeError(
                "QSA compressed split outside the child's pages: "
                f"parent key={len(new_parent.key)}, child pages="
                f"{int(child_cd.value.numel())}, page_size={self.page_size}"
            )
        parent_cd.value = child_cd.value[:split_pages].clone()
        child_cd.value = child_cd.value[split_pages:].clone()

    def evict_component(
        self,
        node: UnifiedTreeNode,
        device_frees: dict,
        host_frees: dict,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        ct = self.component_type
        cd = node.component_data[ct]
        freed = 0
        if EvictLayer.DEVICE in target and cd.value is not None:
            device_frees[ct].append(cd.value)
            freed = int(cd.value.numel())
            self.tree_core.component_evictable_size_[ct] -= freed
            cd.value = None
        if cd.host_value is not None:
            # Host tiers are not supported for the compressed pages.
            raise NotImplementedError(
                "QSA compressed pages have no host tier; evicting host state "
                f"on node {node.id} is unsupported"
            )
        return freed, 0

    def eviction_priority(self, is_leaf: bool) -> int:
        # Compressed pages die with their full-KV pages: same tier as FULL.
        return 0 if is_leaf else 2

    def _evict_device_start(self, request_cnt: int) -> None:
        if request_cnt > 0:
            raise RuntimeError(
                "QSA compressed pages evict only through their FULL pages; "
                f"got an eviction request of {request_cnt}"
            )

    def _evict_device_next_node(self, tracker, device_frees, host_frees):
        return None

    def _evict_device_end(self) -> None:
        return

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        # The request's shared pages live on FULL-locked path nodes.
        return result

    def release_component_lock(
        self, node: UnifiedTreeNode, params, lock_host: bool = False
    ) -> None:
        return

    def apply_component_action(self, action: ComponentAction) -> None:
        if isinstance(action, FreeComponentDeviceSlot):
            for pages in action.indices:
                self.page_cache.release_pages(pages)
            return
        raise AssertionError(
            f"QSACompressedComponent: unhandled ComponentAction {type(action).__name__}"
        )


__all__ = ["QSACompressedComponent"]
