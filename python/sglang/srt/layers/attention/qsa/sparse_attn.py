"""Validated sparse GQA operators migrated from the QSA reference branch."""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

# Diagnostic: pin one Triton (BLOCK_N, num_warps, num_stages) configuration for
# every total_q so the prefill reduction order does not depend on the batch
# shape.  The default per-shape table changes BLOCK_N at 128/256 queries
# (64 -> 32), which makes the same row produce different values in a solo vs
# batched prefill.
_QSA_SPARSE_CANONICAL_CONFIG = os.environ.get(
    "SGLANG_QSA_SPARSE_CANONICAL_CONFIG", ""
).strip()


def _read_diag_config() -> str:
    if os.path.exists("/diag/qsa_sparse_config"):
        try:
            with open("/diag/qsa_sparse_config", encoding="utf-8") as handle:
                return handle.read().strip()
        except OSError:
            return ""
    return ""


# Canonical per-row Triton configuration for the prefill kernels.  A row's
# value must not depend on how many queries happened to share its prefill
# batch, so the configuration is picked from the row's own visible KV count
# (the kernel's ``row_topk``) instead of the batch shape.  Rows are grouped by
# that key and every group is launched with its own (BLOCK_N, num_warps,
# num_stages); ``num_stages`` never changes values, so it stays a pure
# performance knob.  The bucket configs are the validated device table tiers.
_ROW_BUCKETS = (
    (64, (64, 8, 2)),
    (512, (32, 4, 2)),
    (None, (16, 1, 2)),
)
_ROW_BUCKET_LIMITS = tuple(limit for limit, _ in _ROW_BUCKETS if limit is not None)


def _row_config(visible: int, topk: int):
    key = min(visible, topk)
    for limit, config in _ROW_BUCKETS:
        if limit is None or key <= limit:
            return config
    return _ROW_BUCKETS[-1][1]


def _pinned_row_config():
    pinned = _read_diag_config() or _QSA_SPARSE_CANONICAL_CONFIG
    if not pinned:
        return None
    block_n, warps, stages = (int(value) for value in pinned.split(","))
    return (block_n, warps, stages)


def _split_row_ranges(starts, ends, visible_bases, topk, tags=None):
    """Split request row ranges on the per-row configuration boundaries."""
    ranges = []
    for index, (start, end, base) in enumerate(zip(starts, ends, visible_bases)):
        cuts = [start]
        for limit in _ROW_BUCKET_LIMITS:
            if limit >= topk:
                continue
            cut = start + (limit - base) + 1
            if start < cut < end:
                cuts.append(cut)
        cuts.append(end)
        for begin, stop in zip(cuts, cuts[1:]):
            if stop <= begin:
                continue
            visible = base + (begin - start)
            tag = None if tags is None else tags[index]
            ranges.append((_row_config(visible, topk), begin, stop, visible, tag))
    return ranges


def _group_row_ranges(ranges):
    order = []
    grouped = {}
    for entry in ranges:
        config = entry[0]
        if config not in grouped:
            grouped[config] = []
            order.append(config)
        grouped[config].append(entry)
    return [(config, grouped[config]) for config in order]


@triton.jit
def _sparse_gqa_prefill(
    q,
    k,
    v,
    out,
    indices,
    seq_starts,
    seq_ends,
    kv_starts,
    visible_bases,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    seq_start = tl.load(seq_starts + batch).to(tl.int64)
    seq_end = tl.load(seq_ends + batch).to(tl.int64)
    query_relative = tl.program_id(0).to(tl.int64)
    query = seq_start + query_relative
    if query >= seq_end:
        return

    # The KV base is the owning request's start row: the row's token indices
    # are request-relative even when the launch only covers a row sub-range.
    kv_start = tl.load(kv_starts + batch).to(tl.int64)
    visible = tl.load(visible_bases + batch).to(tl.int64) + query_relative
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    head_start = group * GROUP_SIZE
    q_values = tl.load(
        q
        + query * sq_m
        + (head_start + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + kv_start * sk_n + group * sk_h
    v_base = v + kv_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (head_start + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton(q, k, v, max_seqlen_k, indices, cu_seqlens, scale):
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    topk = indices.shape[-1]
    out = torch.empty_like(q)
    starts = [int(value) for value in cu_seqlens[:-1].tolist()]
    ends = [int(value) for value in cu_seqlens[1:].tolist()]
    # No cached prefix in this path: the first row of a request sees one token.
    visible_bases = [1] * len(starts)
    pinned = _pinned_row_config()
    if pinned is None:
        ranges = _split_row_ranges(starts, ends, visible_bases, topk, tags=starts)
    else:
        ranges = [
            (pinned, begin, stop, visible, start)
            for begin, stop, visible, start in zip(starts, ends, visible_bases, starts)
            if stop > begin
        ]
    for (block_n, warps, stages), group in _group_row_ranges(ranges):
        range_starts = torch.tensor(
            [entry[1] for entry in group], dtype=torch.int32, device=q.device
        )
        range_ends = torch.tensor(
            [entry[2] for entry in group], dtype=torch.int32, device=q.device
        )
        range_k = torch.tensor(
            [entry[4] for entry in group], dtype=torch.int32, device=q.device
        )
        range_visible = torch.tensor(
            [entry[3] for entry in group], dtype=torch.int32, device=q.device
        )
        max_rows = max(entry[2] - entry[1] for entry in group)
        _sparse_gqa_prefill[(max_rows, len(group) * num_kv_heads)](
            q,
            k,
            v,
            out,
            indices,
            range_starts,
            range_ends,
            range_k,
            range_visible,
            scale,
            topk,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            indices.stride(0),
            indices.stride(1) if indices.ndim == 3 else 0,
            indices.stride(2) if indices.ndim == 3 else indices.stride(1),
            NUM_KV_HEADS=num_kv_heads,
            GROUP_SIZE=group_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=head_dim,
            num_warps=warps,
            num_stages=stages,
        )
    return out


@triton.jit
def _sparse_gqa_chunk_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_q,
    cu_q_end,
    cu_k,
    visible_bases,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q_end + batch)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    k_start = tl.load(cu_k + batch).to(tl.int64)
    visible = tl.load(visible_bases + batch).to(tl.int64) + query_relative
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q
        + query * sq_m
        + (group * GROUP_SIZE + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + k_start * sk_n + group * sk_h
    v_base = v + k_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        # The chunk-prefill K/V tensors are gathered from the KV pool and can
        # therefore carry the FP8 storage dtype, which Triton's dot rejects
        # (`Unsupported rhs dtype fp8e4nv`). Convert to Q's dtype; the QSA
        # backend writes the pool without per-tensor k/v scales, so this is a
        # plain cast (no-op for BF16 pools).
        keys = keys.to(q_values.dtype)
        values = values.to(q_values.dtype)
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (group * GROUP_SIZE + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton_ck(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    topk = indices.shape[-1]
    out = torch.empty_like(q)
    q_starts = [int(value) for value in cu_q[:-1].tolist()]
    q_ends = [int(value) for value in cu_q[1:].tolist()]
    k_starts = [int(value) for value in cu_k[:-1].tolist()]
    # Visible KV count of each request's first row (its cached prefix + one).
    visible_bases = [
        int(kv_lens[index]) - (q_ends[index] - q_starts[index]) + 1
        for index in range(len(q_starts))
    ]
    pinned = _pinned_row_config()
    if pinned is None:
        ranges = _split_row_ranges(q_starts, q_ends, visible_bases, topk, tags=k_starts)
    else:
        ranges = [
            (pinned, q_starts[index], q_ends[index], visible_bases[index], k_starts[index])
            for index in range(len(q_starts))
            if q_ends[index] > q_starts[index]
        ]
    for (block_n, warps, stages), group in _group_row_ranges(ranges):
        range_starts = torch.tensor(
            [entry[1] for entry in group], dtype=torch.int32, device=q.device
        )
        range_ends = torch.tensor(
            [entry[2] for entry in group], dtype=torch.int32, device=q.device
        )
        range_k = torch.tensor(
            [entry[4] for entry in group], dtype=torch.int32, device=q.device
        )
        range_visible = torch.tensor(
            [entry[3] for entry in group], dtype=torch.int32, device=q.device
        )
        max_rows = max(entry[2] - entry[1] for entry in group)
        _sparse_gqa_chunk_prefill[(max_rows, len(group) * num_kv_heads)](
            q,
            k,
            v,
            out,
            indices,
            range_starts,
            range_ends,
            range_k,
            range_visible,
            scale,
            topk,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            indices.stride(0),
            indices.stride(1) if indices.ndim == 3 else 0,
            indices.stride(2) if indices.ndim == 3 else indices.stride(1),
            NUM_KV_HEADS=num_kv_heads,
            GROUP_SIZE=group_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=head_dim,
            num_warps=warps,
            num_stages=stages,
        )
    return out


@triton.jit
def _fa2_valid_counts(
    seq_lens,
    indices,
    counts,
    topk: tl.constexpr,
    stride_i: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_TOPK)
    length = tl.load(seq_lens + row)
    positions = tl.load(
        indices + row * stride_i + cols,
        mask=cols < topk,
        other=-1,
    )
    valid = (positions >= 0) & (positions < length)
    tl.store(counts + row, tl.sum(valid.to(tl.int32), axis=0))


@triton.jit
def _fa2_prefix_sum(counts, cu_k, batch, BLOCK_B: tl.constexpr):
    rows = tl.arange(0, BLOCK_B)
    valid_rows = rows < batch
    row_counts = tl.load(counts + rows, mask=valid_rows, other=0)
    tl.store(cu_k, 0)
    tl.store(cu_k + rows + 1, tl.cumsum(row_counts, 0), mask=valid_rows)


def qwen_sparse_fa2_cu_seqlens_triton(
    seq_lens, indices, counts, cu_k, batch, topk, block_b: Optional[int] = None
):
    block_b = block_b or triton.next_power_of_2(batch)
    # One request per program: Triton caps a tile at 1M elements,
    # which [next_pow2(topk), next_pow2(batch)] exceeds at topk=2051, batch=512.
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )
    # Prefix sum is only over the batch dimension and remains a small 1-D
    # tensor, including during CUDA graph capture.
    _fa2_prefix_sum[(1,)](
        counts,
        cu_k,
        batch,
        BLOCK_B=block_b,
        num_warps=8,
    )


@triton.jit
def _compact_kv(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    pad_cols,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ZERO_FILL: tl.constexpr,
):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    )
    # 64-bit element offsets: slot * heads * dim exceeds int32 once the pool holds
    # more than 2^31 / (heads * dim) tokens (~4.2M for 2 x 256), which an FP8 pool
    # on one GPU does reach.
    src = slots.to(tl.int64)[:, None] * heads * dim + head * dim + dims[None, :]
    dst = (
        (pack_start + cols).to(tl.int64)[:, None] * heads * dim
        + head * dim
        + dims[None, :]
    )
    load_mask = valid[:, None] & (dims[None, :] < dim)
    if ZERO_FILL:
        # Strided (page-aligned) packing: the paged decode kernel reads whole pages,
        # so every slot in [valid_count, pad_cols) must hold zeros, never stale bytes.
        # `valid_count` here is the row's page-aligned stride, not its valid count, so
        # the store covers the full region while the load stays limited to valid rows.
        store_mask = (cols < pad_cols)[:, None] & (dims[None, :] < dim)
    else:
        store_mask = load_mask
    # Dequantize while gathering: the scratch is allocated in the query dtype, so an
    # FP8 pool is read as fp8 and stored as bf16. The QSA backend writes the pool
    # without per-tensor k/v scales (see set_kv_buffer calls in
    # qwen_sparse_attn_backend.py), so no scale is applied here either.
    out_dtype = out_k.dtype.element_ty
    tl.store(
        out_k + dst,
        tl.load(k + src, mask=load_mask, other=0.0).to(out_dtype),
        mask=store_mask,
    )
    tl.store(
        out_v + dst,
        tl.load(v + src, mask=load_mask, other=0.0).to(out_dtype),
        mask=store_mask,
    )


def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
    """Valid-count pass alone, without the packed cu_seqlens prefix sum."""
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )


def qwen_sparse_kv_extraction_compact_triton(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    batch,
    topk,
    zero_fill_cols: int = 0,
):
    """Gather the selected K/V rows into ``out_k``/``out_v``.

    ``zero_fill_cols`` > 0 selects the strided (page-aligned) layout used by the paged
    decode kernel: row ``b`` owns ``[cu_k[b], cu_k[b] + zero_fill_cols)`` and every slot
    past its valid rows is zero-filled. Paged kernels read whole pages and multiply the
    masked probabilities into V, so stale or uninitialized bytes there (NaN/Inf bit
    patterns) would otherwise leak into the output. ``0`` keeps the compact layout for
    the varlen fallback, whose rows are packed back-to-back.

    ``out_k``/``out_v`` may use a wider dtype than the pool (bf16 scratch for an FP8
    pool); rows are converted while gathering.

    Both layouts assume the valid entries of each ``indices`` row are contiguous at
    the front (``expand_qsa_block_indices`` sorts them that way): ``valid_count`` is a
    count, not a mask, so a ``-1`` in the middle of a row would shift the packing.
    """
    _, heads, dim = k.shape
    block_topk = 16
    zero_fill = zero_fill_cols > 0
    num_cols = zero_fill_cols if zero_fill else topk
    _compact_kv[(batch, heads, triton.cdiv(num_cols, block_topk))](
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        topk,
        heads,
        dim,
        req_to_token.stride(0),
        indices.stride(0),
        num_cols,
        BLOCK_TOPK=block_topk,
        BLOCK_D=triton.next_power_of_2(dim),
        ZERO_FILL=zero_fill,
        num_warps=8,
    )


__all__ = [
    "qwen_sparse_fa2_cu_seqlens_triton",
    "qwen_sparse_valid_counts_triton",
    "qwen_sparse_kv_extraction_compact_triton",
    "sparse_gqa_fwd_interface_triton",
    "sparse_gqa_fwd_interface_triton_ck",
]
