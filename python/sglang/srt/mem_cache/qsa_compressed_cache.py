"""Ownership of the QSA compressed-K pages (logical mapping).

Stage 2 of ``docs/qsa-logical-compressed-mapping.md``: the compressed cache
stops being a shadow of the physical full-KV slot space
(``compressed_slot = full_slot // ratio``).  Pages live in their own page
space and are owned by exactly one owner at a time:

- a radix node owns the pages covering its own token range (the immutable
  shared prefix of every request that matches the node), or
- a request owns the tail pages it wrote itself.

The request-visible read table is the concatenation

    read_table = concat(matched_node_chain_pages, request_tail_pages)

so the write side of the attention backend can assert that every write lands
in the request-owned tail.  Page ownership moves at the radix lifecycle
boundaries only (insert donation, node split, node eviction, request free);
``full_slot // ratio`` is never used to derive a compressed identity.

Per request row the table is laid out as

    [0, shared)      node-owned, read-only   (shared prefix)
    [own_lo, own_hi) request-owned, writable (tail)
    outside          unallocated (-1) or stale detached entries

``own_lo == shared`` holds right after a match; an insert detaches the head of
the tail (``own_lo`` advances as those pages move to the new nodes) until the
next match rebinds the row.
"""

from __future__ import annotations

import os
import time
from typing import Any, List, Optional

import torch

_TIMING = os.environ.get("SGLANG_QSA_CP_TIMING") == "1"

ENV = "SGLANG_QSA_LOGICAL_COMPRESSED"

# Version of the value path that produces a compressed entry.  Two compressed
# segments may be shared only when their versions match.  Bump this on any
# change that alters the produced values for the same token sequence: kernel
# selection, factorization, dtype, backend.  Purely performance-neutral knobs
# (e.g. a Triton ``num_stages`` proven to keep the values bitwise identical)
# do not need a bump.  The row-intrinsic prefill config (README 2.4) and the
# exact mamba checkpoint (README 2.5) are part of version 1.
QSA_VALUE_PATH_VERSION = 1


def logical_compressed_enabled() -> bool:
    return os.environ.get(ENV) == "1"


class QSACompressedPageCache:
    """Compressed page allocator plus the per-request logical read table.

    Ownership invariants (``check_balance`` asserts the page ledger):

    - a page is either free or handed out (request tail, detached, node);
    - ``len(free_pages) + in_use == total_pages - 1`` (page 0 is the inert
      dump, never handed out);
    - a request writes only into ``[own_lo, own_hi)`` of its own row;
    - ``shared <= own_lo <= own_hi``.
    """

    def __init__(self, kv_pool: Any, req_rows: int) -> None:
        self.kv_pool = kv_pool
        self.compress_ratio = int(kv_pool.qsa_compress_ratio)
        self.cps = int(kv_pool.qsa_compressed_page_size)
        self.page_size = self.cps * self.compress_ratio
        if self.cps <= 0 or self.compress_ratio <= 1:
            raise ValueError(
                "QSA logical compressed mapping needs compress_ratio >= 2 and "
                f"compressed_page_size >= 1, got cps={self.cps}, ratio={self.compress_ratio}"
            )
        self.total_pages = int(kv_pool.qsa_compressed_capacity) // self.cps
        if self.total_pages <= 1:
            raise ValueError("QSA compressed cache has no allocatable page")
        self.req_rows = int(req_rows)
        if self.req_rows <= 0:
            raise ValueError("QSA compressed cache needs request slots")
        # One extra page absorbs the ceil of a context that ends mid-page.
        self.max_pages_per_req = (
            int(kv_pool.size) + self.page_size - 1
        ) // self.page_size + 1
        # Logical read table: one row per request slot.
        self.req_pages = torch.full(
            (self.req_rows, self.max_pages_per_req),
            -1,
            dtype=torch.int32,
            device=kv_pool.device,
        )
        # Host-side ownership state, one entry per request slot.
        self.shared_pages: List[int] = [0] * self.req_rows
        # The request's own page ids, host-authoritative: freeing a row never
        # reads the device table back (that syncs the host on the stream).
        self.own_ids: List[List[int]] = [[] for _ in range(self.req_rows)]
        self.own_lo: List[int] = [0] * self.req_rows
        self.own_hi: List[int] = [0] * self.req_rows
        self.shared_tokens: List[int] = [0] * self.req_rows
        # Device mirrors for the prefill write-target assertion.
        self.shared_tokens_dev = torch.zeros(
            self.req_rows, dtype=torch.int32, device=kv_pool.device
        )
        self.tail_lo_dev = torch.zeros(
            self.req_rows, dtype=torch.int32, device=kv_pool.device
        )
        self.tail_hi_dev = torch.zeros(
            self.req_rows, dtype=torch.int32, device=kv_pool.device
        )
        # Page 0 stays the inert dump; the rest is the free list.
        self.free_pages: List[int] = list(range(1, self.total_pages))
        # Mirrors free_pages for O(1) double-free detection.
        self.free_set: set[int] = set(self.free_pages)
        self.in_use = 0
        self.dense_capacity = self.total_pages - 1
        self.coverable_tokens = self.dense_capacity * self.page_size
        # The composite this cache belongs to (set on attach) and whether the
        # radix component claimed this page space for shared ownership.
        self.composite: Optional[QSACompressedAuxCaches] = None
        self.radix_owned = False
        # The request pool's clear()/reset paths call these two attribute
        # names; keep them inert for interface compatibility.
        self.reserved_slots = torch.empty(0, dtype=torch.int64, device=kv_pool.device)
        self.free_slots = self.reserved_slots

    # ---- allocator ----

    def claim_radix(self) -> None:
        """Claim this page space for node-owned sharing (single-pool only)."""
        if self.composite is None or len(self.composite.caches) != 1:
            count = None if self.composite is None else len(self.composite.caches)
            raise NotImplementedError(
                "QSA logical compressed radix sharing requires exactly one "
                f"compressed page space; attached caches={count}"
            )
        self.composite.claim_radix()
        self.radix_owned = True

    def check_balance(self) -> None:
        if len(self.free_pages) != len(self.free_set):
            raise RuntimeError(
                "QSA compressed free list and set disagree: "
                f"list={len(self.free_pages)}, set={len(self.free_set)}"
            )
        if len(self.free_pages) + self.in_use != self.total_pages - 1:
            raise RuntimeError(
                "QSA compressed page accounting is unbalanced: "
                f"free={len(self.free_pages)}, in_use={self.in_use}, "
                f"total={self.total_pages}"
            )

    def _return_pages(self, values: List[int]) -> None:
        for value in values:
            if value <= 0 or value >= self.total_pages:
                raise RuntimeError(
                    f"QSA compressed page id out of range: {value} "
                    f"(total={self.total_pages})"
                )
            if value in self.free_set:
                raise RuntimeError(
                    f"QSA compressed page {value} freed twice"
                )
        self.free_pages.extend(values)
        self.free_set.update(values)
        self.in_use -= len(values)

    def _take_pages(self, count: int) -> List[int]:
        if count > len(self.free_pages):
            raise RuntimeError(
                "QSA logical compressed cache is out of pages: "
                f"requested={count}, available={len(self.free_pages)}"
            )
        # Hand pages out like the full-KV allocator does (LIFO from the tail,
        # ascending within one request) so a request's compressed slots live
        # in the same address region under both mappings.
        pages = sorted(self.free_pages[-count:])
        del self.free_pages[-count:]
        self.free_set.difference_update(pages)
        self.in_use += count
        return pages

    def release_pages(self, pages: Any) -> None:
        """Return pages to the free list (drained evictions / detached tails)."""
        if torch.is_tensor(pages):
            values = [int(value) for value in pages.reshape(-1).cpu().tolist()]
        else:
            values = [int(value) for value in pages]
        self._return_pages(values)
        self.check_balance()

    def reset_allocator(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.req_pages.fill_(-1)
        self.shared_pages = [0] * self.req_rows
        self.own_ids = [[] for _ in range(self.req_rows)]
        self.own_lo = [0] * self.req_rows
        self.own_hi = [0] * self.req_rows
        self.shared_tokens = [0] * self.req_rows
        self.shared_tokens_dev.zero_()
        self.tail_lo_dev.zero_()
        self.tail_hi_dev.zero_()
        self.free_pages = list(range(1, self.total_pages))
        self.free_set = set(self.free_pages)
        self.in_use = 0

    # ---- request read table ----

    def stage_shared(
        self, req: Any, pages: torch.Tensor, shared_tokens: int
    ) -> None:
        """Record the matched node chain's pages for the request's next alloc."""
        req.kv.qsa_shared_pages = pages.to(dtype=torch.int32)
        req.kv.qsa_shared_tokens = int(shared_tokens)

    def bind_match(self, row: int, pages: torch.Tensor, shared_tokens: int) -> None:
        """Point the row's shared prefix at the matched node chain's pages.

        The tail pages the matched chain made redundant were detached by the
        insert hooks (donated to new nodes or freed as duplicates), so the
        tail's head is already at or beyond the new shared boundary.
        """
        if row < 0 or row >= self.req_rows:
            raise RuntimeError(f"QSA compressed row out of range: {row}")
        if shared_tokens % self.page_size != 0:
            raise RuntimeError(
                "QSA shared prefix must be page aligned: "
                f"tokens={shared_tokens}, page_size={self.page_size}"
            )
        count = int(shared_tokens) // self.page_size
        if pages.numel() != count:
            raise RuntimeError(
                "QSA shared page count does not cover the matched prefix: "
                f"pages={pages.numel()}, tokens={shared_tokens}"
            )
        shared_old = self.shared_pages[row]
        own_lo = self.own_lo[row]
        own_hi = self.own_hi[row]
        if count < shared_old:
            raise RuntimeError(
                "QSA matched prefix receded (node evicted under a live request?): "
                f"row={row}, shared {shared_old}->{count}"
            )
        if count < own_lo:
            raise RuntimeError(
                "QSA match does not reach the detached tail head: "
                f"row={row}, shared={count}, own_lo={own_lo}"
            )
        if count > own_hi and not (own_lo == 0 and own_hi == 0):
            raise RuntimeError(
                "QSA match grew past the request-owned tail: "
                f"row={row}, shared {shared_old}->{count}, tail=[{own_lo}, {own_hi})"
            )
        if count > shared_old:
            self.req_pages[row, :count] = pages.to(self.req_pages.dtype)
        # The tail window is empty right after a fresh bind; the allocation
        # that always follows reopens it (target length >= matched prefix).
        self.shared_pages[row] = count
        self.own_lo[row] = count
        self.own_hi[row] = max(own_hi, count)
        self.shared_tokens[row] = int(shared_tokens)
        self.shared_tokens_dev[row] = int(shared_tokens)
        self.tail_lo_dev[row] = count
        self.tail_hi_dev[row] = self.own_hi[row]

    def _bind_req(self, req: Any) -> None:
        pages = getattr(req.kv, "qsa_shared_pages", None)
        if pages is None:
            return
        self.bind_match(
            int(req.kv.req_pool_idx),
            pages,
            int(req.kv.qsa_shared_tokens),
        )

    def alloc_to_lengths(
        self,
        *,
        req_pool_indices_cpu: Any,
        target_seq_lens_cpu: Any,
        reqs: Optional[Any] = None,
    ) -> None:
        """Bind staged shared prefixes, then cover each row's tail with pages."""

        _t0 = time.perf_counter() if _TIMING else 0.0
        _took = 0
        if reqs is not None:
            for req in reqs:
                self._bind_req(req)
        if not torch.is_tensor(req_pool_indices_cpu):
            req_pool_indices_cpu = torch.tensor(req_pool_indices_cpu, dtype=torch.long)
        if not torch.is_tensor(target_seq_lens_cpu):
            target_seq_lens_cpu = torch.tensor(target_seq_lens_cpu, dtype=torch.long)
        indices = [int(value) for value in req_pool_indices_cpu.long().reshape(-1).tolist()]
        lengths = [int(value) for value in target_seq_lens_cpu.long().reshape(-1).tolist()]
        for req, length in zip(indices, lengths):
            if req < 0 or req >= self.req_rows:
                continue
            need = (length + self.page_size - 1) // self.page_size
            need = min(need, self.max_pages_per_req)
            own_hi = self.own_hi[req]
            if need <= own_hi:
                continue
            take = need - own_hi
            pages = self._take_pages(take)
            # Pinned staging + a non-blocking copy: a pageable host-to-device
            # copy here would block the host until the in-flight step drains,
            # idling the GPU for the rest of the step.
            staged = torch.tensor(pages, dtype=self.req_pages.dtype, pin_memory=True)
            self.req_pages[req, own_hi:need].copy_(staged, non_blocking=True)
            self.own_ids[req].extend(pages)
            self.own_hi[req] = need
            self.tail_hi_dev[req] = need
            _took += take
        self.check_balance()
        if _TIMING:
            _elapsed = time.perf_counter() - _t0
            if _elapsed > 0.002:
                print(
                    "QSA_CP_TIMING alloc_to_lengths %.1fms rows=%s lens=%s took=%d"
                    % (
                        _elapsed * 1000.0,
                        [int(v) for v in req_pool_indices_cpu.long().reshape(-1).tolist()],
                        [int(v) for v in target_seq_lens_cpu.long().reshape(-1).tolist()],
                        _took,
                    ),
                    flush=True,
                )

    def free(self, req_pool_idx: int) -> None:
        """Release a request row: only the request-owned tail pages."""
        req = int(req_pool_idx)
        if req < 0 or req >= self.req_rows:
            return
        own_lo = self.own_lo[req]
        own_hi = self.own_hi[req]
        if own_hi > own_lo:
            self._return_pages(self.own_ids[req][: own_hi - own_lo])
        self.own_ids[req] = []
        self.req_pages[req].fill_(-1)
        self.shared_pages[req] = 0
        self.own_lo[req] = 0
        self.own_hi[req] = 0
        self.shared_tokens[req] = 0
        self.shared_tokens_dev[req] = 0
        self.tail_lo_dev[req] = 0
        self.tail_hi_dev[req] = 0
        self.check_balance()

    # ---- node ownership ----

    def detach_own(
        self, row: int, token_start: int, token_end: int
    ) -> Optional[torch.Tensor]:
        """Detach the request-owned pages of a token range (head of the tail).

        The pages leave the request tail but stay allocated; the caller either
        donates them to a radix node (insert) or frees them (duplicate).  The
        range must be page aligned and lie inside the current tail.
        """
        if token_end <= token_start:
            return None
        if token_start % self.page_size or token_end % self.page_size:
            raise RuntimeError(
                "QSA compressed ownership moves on page boundaries only: "
                f"range=({token_start}, {token_end}), page_size={self.page_size}"
            )
        own_lo = self.own_lo[row]
        own_hi = self.own_hi[row]
        start = max(token_start // self.page_size, own_lo)
        end = min(token_end // self.page_size, own_hi)
        if end <= start:
            return None
        if start != own_lo:
            raise RuntimeError(
                "QSA detach must start at the tail head (non-monotonic walk): "
                f"row={row}, start={start}, own_lo={own_lo}, own_hi={own_hi}"
            )
        pages = self.req_pages[row, start:end].clone()
        del self.own_ids[row][: end - start]
        self.own_lo[row] = end
        self.tail_lo_dev[row] = end
        self.req_pages[row, start:end] = -1
        return pages

    def row_token_range(self, row: int) -> tuple[int, int]:
        """Token range currently covered by the row (shared + tail)."""
        return (0, self.own_hi[row] * self.page_size)

    # ---- read side ----

    def page_table(self, req_pool_indices) -> torch.Tensor:
        return self.req_pages.index_select(0, req_pool_indices.long())

    def shared_tokens_table(self, req_pool_indices) -> torch.Tensor:
        return self.shared_tokens_dev.index_select(0, req_pool_indices.long())

    def tail_lo_pages_table(self, req_pool_indices) -> torch.Tensor:
        return self.tail_lo_dev.index_select(0, req_pool_indices.long())

    def tail_hi_pages_table(self, req_pool_indices) -> torch.Tensor:
        return self.tail_hi_dev.index_select(0, req_pool_indices.long())

    def pages_needed(self, length: int) -> int:
        return min(
            (int(length) + self.page_size - 1) // self.page_size,
            self.max_pages_per_req,
        )


class QSACompressedAuxCaches:
    """Composite aux cache: one page cache per QSA KV pool.

    The request pool exposes a single aux slot; the target and draft workers
    may each own a compressed buffer, so the slot fans out to every attached
    page cache.  The KV lifecycle hooks (alloc_to_lengths / free / clear) are
    forwarded to all of them.  Radix-owned sharing needs a single page space
    (the frozen-KV draft reads the target's pool), so a second attach after
    the component claimed ownership is rejected.
    """

    # The request's staged shared prefix rides in on alloc_to_lengths.
    wants_reqs = True

    def __init__(self) -> None:
        self.caches: dict[int, QSACompressedPageCache] = {}
        self.claimed = False

    def claim_radix(self) -> None:
        self.claimed = True

    def attach(self, kv_pool: Any) -> QSACompressedPageCache:
        key = id(kv_pool)
        cache = self.caches.get(key)
        if cache is None:
            if self.claimed or any(
                existing.radix_owned for existing in self.caches.values()
            ):
                raise NotImplementedError(
                    "QSA logical compressed radix sharing requires a single "
                    "compressed page space; a second KV pool attached after "
                    "the radix component took ownership."
                )
            cache = QSACompressedPageCache(
                kv_pool, int(kv_pool.qsa_num_request_slots)
            )
            cache.composite = self
            self.caches[key] = cache
            kv_pool.qsa_compressed_page_cache = cache
        return cache

    @property
    def dense_capacity(self) -> int:
        return min((cache.dense_capacity for cache in self.caches.values()), default=0)

    def schedulable_token_capacity(self, physical_capacity: int) -> int:
        """KV tokens still schedulable; the compressed pages mirror the KV pages 1:1."""

        coverable = max(
            (cache.coverable_tokens for cache in self.caches.values()), default=0
        )
        return min(int(physical_capacity), coverable)

    def alloc_to_lengths(self, **kwargs) -> None:
        reqs = kwargs.get("reqs")
        for cache in self.caches.values():
            cache.alloc_to_lengths(**kwargs)
        if reqs is not None:
            for req in reqs:
                req.kv.qsa_shared_pages = None
                req.kv.qsa_shared_tokens = 0

    def free(self, req_pool_idx: int) -> None:
        for cache in self.caches.values():
            cache.free(req_pool_idx)

    def clear(self) -> None:
        for cache in self.caches.values():
            cache.clear()

    def reset_allocator(self) -> None:
        for cache in self.caches.values():
            cache.clear()


def attach_qsa_compressed_cache(pool: Any, kv_pool: Any) -> QSACompressedPageCache:
    """Attach (or reuse) the logical compressed cache for one KV pool."""

    existing = getattr(pool, "_aux_cache", None)
    if isinstance(existing, QSACompressedAuxCaches):
        return existing.attach(kv_pool)
    if existing is not None:
        raise RuntimeError(
            "a non-QSA aux cache is already attached to the request pool; "
            "the QSA logical compressed mapping needs the aux slot"
        )
    composite = QSACompressedAuxCaches()
    pool.attach_aux_cache(composite)
    return composite.attach(kv_pool)


def get_qsa_compressed_page_cache(kv_pool: Any) -> Optional[QSACompressedPageCache]:
    return getattr(kv_pool, "qsa_compressed_page_cache", None)
