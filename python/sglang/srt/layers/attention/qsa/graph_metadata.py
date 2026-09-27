"""GPU builders for QSA CUDA-graph replay metadata.

Compressed addressing is pure arithmetic over the page-aligned full-KV cache:
a group's compressed slot is any of its raw slots floor-divided by the compress ratio,
so per-row graph buffers are rebuilt from request lengths and ``req_to_token`` alone;
accept-dependent speculative lengths never need the host.

Both kernels run once eagerly at capture warmup,
then are recorded into the main CUDA graph through ``init_forward_metadata_in_graph``;
all inputs are stable-address runner buffers.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_graph_layout_kernel(
    # Request-level inputs.
    seq_lens_ptr,  # [bs] base lengths
    req_pool_ptr,  # [bs] request pool slots
    extend_lens_ptr,  # [bs] per-request extend lengths (draft extend)
    # Row layout buffers (graph persistent state).
    row_seq_lens_ptr,
    row_prefix_lens_ptr,
    row_req_pool_ptr,
    bs,
    num_tokens,
    num_padding,
    extend_len,  # uniform extend length (target verify); 0 -> extend_lens_ptr
    MODE: tl.constexpr,  # 0 = decode, 1 = target verify, 2 = draft extend
):
    pid = tl.program_id(0)

    if MODE == 0:
        if pid < bs:
            real = pid < bs - num_padding
            seq_len = tl.load(seq_lens_ptr + pid).to(tl.int32)
            req = tl.load(req_pool_ptr + pid).to(tl.int64)
            # Padding rows alias request slot 0: it is never allocated, so
            # its pending-ring rows are the inert dump for their state
            # stores, and its req_to_token row reads stay in-bounds.
            req = tl.where(real, req, 0)
            seq_len = tl.where(real, seq_len, 1)
            prefix = tl.maximum(seq_len - 1, 0)
            tl.store(row_seq_lens_ptr + pid, seq_len)
            tl.store(row_req_pool_ptr + pid, req.to(tl.int32))
            tl.store(row_prefix_lens_ptr + pid, prefix)
        return

    real_reqs = bs - num_padding
    if pid == bs:
        # Tail program: dummy rows for the static capacity past the real layout.
        if MODE == 1:
            row_start = real_reqs * extend_len
        else:
            row_start = 0
            for j in range(real_reqs):
                row_start += tl.load(extend_lens_ptr + j)
        for row in range(row_start, num_tokens):
            tl.store(row_seq_lens_ptr + row, 1)
            tl.store(row_prefix_lens_ptr + row, 0)
            # Request slot 0 is never allocated: inert for ring stores and
            # in-bounds for every row read.
            tl.store(row_req_pool_ptr + row, 0)
        return

    if MODE == 1:
        eff = tl.where(pid < real_reqs, extend_len, 0)
        offset = tl.minimum(pid, real_reqs) * extend_len
    else:
        eff = 0
        offset = 0
        for j in range(bs):
            e_j = tl.where(j < real_reqs, tl.load(extend_lens_ptr + j), 0)
            offset += tl.where(j < pid, e_j, 0)
            eff = tl.where(j == pid, e_j, eff)
    base = tl.load(seq_lens_ptr + pid).to(tl.int32)
    req = tl.load(req_pool_ptr + pid).to(tl.int64)
    if MODE == 1:
        prefix = base
        limit = base + eff
    else:
        prefix = tl.maximum(base - eff, 0)
        limit = base
    for j in range(eff):
        row = offset + j
        seq_len = tl.minimum(prefix + 1 + j, limit)
        tl.store(row_seq_lens_ptr + row, seq_len)
        tl.store(row_prefix_lens_ptr + row, prefix)
        tl.store(row_req_pool_ptr + row, req.to(tl.int32))


@triton.jit
def _qsa_graph_row_metadata_kernel(
    # Row layout buffers (filled by the layout kernel).
    row_seq_lens_ptr,
    row_req_pool_ptr,
    # Graph output buffers.
    compressed_lens_ptr,
    write_locs_ptr,
    page_table_ptr,
    logical_positions_ptr,
    state_slots_ptr,
    ring_locs_ptr,
    # Pool state.
    req_to_token_ptr,
    req_to_token_row_stride,
    max_pages,
    # Logical compressed page table: one row per request pool slot.
    req_pages_ptr,
    req_pages_row_stride,
    RATIO: tl.constexpr,
    RING: tl.constexpr,  # pending-ring slots per request
    FULL_PAGE: tl.constexpr,  # full-KV tokens per page
    CPS: tl.constexpr,  # compressed slots per page
    PAGE_BLOCK: tl.constexpr,
    LOGICAL: tl.constexpr,
):
    row = tl.program_id(0)
    seq_len = tl.load(row_seq_lens_ptr + row).to(tl.int32)
    req = tl.load(row_req_pool_ptr + row).to(tl.int64)
    token_row = req * req_to_token_row_stride
    current = tl.maximum(seq_len - 1, 0)

    compressed = seq_len // RATIO
    tl.store(compressed_lens_ptr + row, compressed)

    # A boundary row completes its group.  The physical path derives the
    # compressed slot from the request's full-KV slot (full_slot // ratio);
    # the logical path reads the request's own page table instead, so a write
    # can never land in a node-owned shared page.
    boundary = (seq_len > 0) & (seq_len % RATIO == 0)
    if LOGICAL:
        logical_row = req_pages_ptr + req * req_pages_row_stride
        block = tl.maximum(compressed - 1, 0)
        page_index = tl.minimum(block // CPS, max_pages - 1)
        page_id = tl.load(logical_row + page_index).to(tl.int32)
        write_loc = tl.where(boundary, page_id * CPS + (block % CPS), 0)
    else:
        last_loc = tl.load(req_to_token_ptr + token_row + current).to(tl.int32)
        write_loc = tl.where(boundary, last_loc // RATIO, 0)
    tl.store(write_locs_ptr + row, write_loc)

    tl.store(logical_positions_ptr + row, current)
    tl.store(state_slots_ptr + row, req * RING + (current % RING).to(tl.int64))
    ring_base = row.to(tl.int64) * RATIO
    for k in tl.static_range(RATIO):
        member = tl.maximum(current - (RATIO - 1 - k), 0)
        slot = req * RING + (member % RING).to(tl.int64)
        tl.store(ring_locs_ptr + ring_base + k, slot.to(tl.int32))

    # Page-table entries are compressed page ids.  Physical: the request's
    # FULL-KV page ids from the page-aligned req_to_token row.  Logical: the
    # request's own page table (shared node pages followed by the tail).
    table_row = page_table_ptr + row.to(tl.int64) * max_pages
    offs = tl.arange(0, PAGE_BLOCK)
    if LOGICAL:
        logical_row = req_pages_ptr + req * req_pages_row_stride
        for p0 in range(0, max_pages, PAGE_BLOCK):
            idx = p0 + offs
            valid = idx < max_pages
            page = tl.load(logical_row + idx, mask=valid, other=0)
            tl.store(table_row + idx, tl.maximum(page, 0), mask=valid)
    else:
        row_width_pages = req_to_token_row_stride // FULL_PAGE
        for p0 in range(0, max_pages, PAGE_BLOCK):
            idx = p0 + offs
            valid = idx < tl.minimum(max_pages, row_width_pages)
            loc = tl.load(
                req_to_token_ptr + token_row + idx * FULL_PAGE, mask=valid, other=0
            )
            tl.store(table_row + idx, tl.maximum(loc // FULL_PAGE, 0), mask=valid)


def supports_graph_metadata_kernels(pool, device, *, logical: bool = False) -> bool:
    """Whether the CUDA fast path can serve this pool/device pair.

    ``logical`` selects the compressed-page-table variant of the row kernel,
    which the same launch path provides.
    """

    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    if torch.device(device).type != "cuda" or not isinstance(pool, QSATokenToKVPool):
        return False
    if logical:
        return getattr(pool, "qsa_compressed_page_cache", None) is not None
    return True


def launch_graph_metadata(
    *,
    mode,
    bs,
    num_rows,
    seq_lens,
    req_pool_indices,
    extend_lens,
    extend_len,
    num_padding,
    metadata,
    req_to_token,
    pool,
    logical_page_cache=None,
) -> None:

    indexer = metadata.indexer_metadata
    max_pages = indexer.graph_compressed_page_table.shape[1]
    row_seq_lens = metadata.sequence_lengths
    row_req_pool = metadata.row_req_pool_indices
    row_prefix_lens = indexer.graph_prefix_lengths

    _qsa_graph_layout_kernel[(bs + 1,)](
        seq_lens,
        req_pool_indices,
        (
            extend_lens
            if extend_lens is not None
            else row_seq_lens  # unused dummy pointer
        ),
        row_seq_lens,
        row_prefix_lens,
        row_req_pool,
        bs,
        num_rows,
        num_padding,
        extend_len,
        MODE=mode,
        num_warps=1,
    )
    logical = logical_page_cache is not None
    if logical:
        req_pages = logical_page_cache.req_pages
        req_pages_stride = req_pages.stride(0)
    else:
        # Unused pointers for the physical variant.
        req_pages = req_to_token
        req_pages_stride = 0
    _qsa_graph_row_metadata_kernel[(num_rows,)](
        row_seq_lens,
        row_req_pool,
        indexer.graph_compressed_lengths,
        indexer.graph_write_locs,
        indexer.graph_compressed_page_table,
        indexer.decode_logical_positions,
        indexer.pending_ring_slots,
        indexer.graph_ring_group_locs,
        req_to_token,
        req_to_token.stride(0),
        max_pages,
        req_pages,
        req_pages_stride,
        RATIO=indexer.compress_ratio,
        RING=pool.qsa_ring_size,
        FULL_PAGE=pool.qsa_compressed_page_size * indexer.compress_ratio,
        CPS=pool.qsa_compressed_page_size,
        PAGE_BLOCK=128,
        LOGICAL=logical,
        num_warps=1,
    )
