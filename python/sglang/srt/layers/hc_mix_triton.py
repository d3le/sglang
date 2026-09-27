"""Deterministic fused HC low-rank mix for decode-size batches.

One persistent kernel replaces the five-kernel `GatedResidual._mix_compute` chain.
The down projection uses a fixed-order split-K workspace reduction rather than
floating-point atomics, so the result is independent of CTA scheduling. One CTA
per SM keeps every CTA resident, so the software grid barriers cannot deadlock;
the last CTA to finish resets the barrier counters and a captured CUDA graph
replays with the buffers back in their initial state.

Row counts beyond ``_FUSED_MIX_MAX_ROWS`` stay on the torch.compile path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_FUSED_MIX_MAX_ROWS = 16
# The Qwen3.8 shape has K=10240 and therefore 40 BLOCK_K tiles. Four fixed
# partitions are the fastest GB10 choice in the promotion microbenchmark while
# making each partial output unique to one CTA. This remains a compile-time
# knob for reproducing the split-K comparison.
_FUSED_MIX_SPLIT_K = 4


@triton.jit
def _grid_barrier(counter_ptr, num_ctas):
    # A CTA must publish completion only after every warp has finished the
    # preceding writes.  An acq_rel atomic by the CTA leader does not order
    # stores from the other warps without this intra-CTA barrier.
    tl.debug_barrier()
    tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    while tl.atomic_add(counter_ptr, 0, sem="acq_rel", scope="gpu") < num_ctas:
        pass


@triton.jit
def _hc_mix_persistent_kernel(
    x_ptr,
    w_down_ptr,
    w_up_ptr,
    partial_ptr,
    t_raw_ptr,
    out_ptr,
    counters_ptr,
    K,
    LOWRANK,
    HS,
    num_rows,
    num_ctas,
    inv_hc,
    ROWS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    n_blocks = tl.cdiv(LOWRANK, BLOCK_N)
    k_chunks = tl.cdiv(K, BLOCK_K)
    # Each (n-block, split) task owns a disjoint partial tile. The K chunks
    # within a split are accumulated in increasing order inside one CTA.
    for tile in range(pid, n_blocks * SPLIT_K, num_ctas):
        nb = tile % n_blocks
        split = tile // n_blocks
        n = nb * BLOCK_N + offs_n
        mask_n = n < LOWRANK
        acc = tl.zeros((ROWS, BLOCK_N), dtype=tl.float32)
        for kc in range(split, k_chunks, SPLIT_K):
            k = kc * BLOCK_K + offs_k
            xt = tl.load(
                x_ptr + offs_m[:, None] * K + k[None, :],
                mask=mask_m[:, None],
                other=0.0,
            )
            w = tl.load(
                w_down_ptr + n[:, None] * K + k[None, :],
                mask=mask_n[:, None],
                other=0.0,
            )
            acc = tl.dot(xt, tl.trans(w), acc)
        partial_base = split * ROWS * LOWRANK
        tl.store(
            partial_ptr
            + partial_base
            + offs_m[:, None] * LOWRANK
            + n[None, :],
            acc,
            mask=mask_n[None, :],
        )
    _grid_barrier(counters_ptr + 1, num_ctas)

    # Reduce partials in a fixed split order. Each output tile has one owner,
    # so this phase has no floating-point atomics.
    for tile in range(pid, n_blocks, num_ctas):
        n = (tile % n_blocks) * BLOCK_N + offs_n
        mask_n = n < LOWRANK
        acc = tl.zeros((ROWS, BLOCK_N), dtype=tl.float32)
        for split in range(SPLIT_K):
            partial_base = split * ROWS * LOWRANK
            part = tl.load(
                partial_ptr
                + partial_base
                + offs_m[:, None] * LOWRANK
                + n[None, :],
                mask=mask_n[None, :],
                other=0.0,
            )
            acc += part
        tl.store(
            t_raw_ptr + offs_m[:, None] * LOWRANK + n[None, :],
            acc,
            mask=mask_n[None, :],
        )
    _grid_barrier(counters_ptr + 2, num_ctas)

    offs_j = tl.arange(0, BLOCK_J)
    offs_r = tl.arange(0, BLOCK_R)
    offs_g = tl.arange(0, HC)
    j_blocks = tl.cdiv(HS, BLOCK_J)
    for jb in range(pid, j_blocks, num_ctas):
        j = jb * BLOCK_J + offs_j
        mask_j = j < HS
        gj = offs_g[:, None] * HS + j[None, :]
        gj_flat = tl.reshape(gj, (HC * BLOCK_J,))
        mask_gj = tl.reshape(
            tl.broadcast_to(mask_j[None, :], (HC, BLOCK_J)), (HC * BLOCK_J,)
        )
        acc = tl.zeros((ROWS, HC * BLOCK_J), dtype=tl.float32)
        for r0 in range(0, LOWRANK, BLOCK_R):
            r = r0 + offs_r
            mask_r = r < LOWRANK
            a = tl.load(
                t_raw_ptr + offs_m[:, None] * LOWRANK + r[None, :],
                mask=mask_r[None, :],
                other=0.0,
            )
            a = a * inv_hc
            t = (a * tl.sigmoid(a)).to(x_ptr.dtype.element_ty)
            w = tl.load(
                w_up_ptr + gj_flat[:, None] * LOWRANK + r[None, :],
                mask=mask_gj[:, None] & mask_r[None, :],
                other=0.0,
            )
            acc = tl.dot(t, tl.trans(w), acc)
        gate = tl.sigmoid(tl.reshape(acc, (ROWS, HC, BLOCK_J)))
        xg = tl.load(
            x_ptr
            + offs_m[:, None, None] * (HC * HS)
            + offs_g[None, :, None] * HS
            + j[None, None, :],
            mask=mask_m[:, None, None] & mask_j[None, None, :],
            other=0.0,
        ).to(tl.float32)
        out = tl.sum(gate * xg, axis=1) * inv_hc
        tl.store(
            out_ptr + offs_m[:, None] * HS + j[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_j[None, :],
        )

    # Publish output completion only after all CTA threads have finished their
    # stores. The counter reset is safe for the next same-stream invocation.
    tl.debug_barrier()
    ticket = tl.atomic_add(counters_ptr + 0, 1, sem="acq_rel", scope="gpu")
    if ticket == num_ctas - 1:
        tl.store(counters_ptr + 1, 0)
        tl.store(counters_ptr + 2, 0)
        tl.store(counters_ptr + 0, 0)


_counters_cache = {}


def _get_counters(device: torch.device) -> torch.Tensor:
    # Graph capture must receive an invocation-owned counter: target and draft
    # graphs may be captured from the same Python stream and replay separately.
    # Outside capture, reuse one counter per stream to avoid a tiny allocator
    # call on every eager HC mix; same-stream launches are serialized, while
    # different streams get distinct barrier state.
    if torch.cuda.is_current_stream_capturing():
        return torch.zeros(3, dtype=torch.int32, device=device)
    stream = torch.cuda.current_stream(device)
    key = (device, stream.cuda_stream)
    buf = _counters_cache.get(key)
    if buf is None:
        buf = torch.zeros(3, dtype=torch.int32, device=device)
        _counters_cache[key] = buf
    return buf


def fused_hc_mix_supported(
    hyper_input_normed: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor
) -> bool:
    return (
        hyper_input_normed.is_cuda
        and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
        and w_down.dtype == hyper_input_normed.dtype
        and w_up.dtype == hyper_input_normed.dtype
        and hyper_input_normed.shape[0] <= _FUSED_MIX_MAX_ROWS
        and hyper_input_normed.dim() == 2
        and hyper_input_normed.shape[1] % 2048 == 0
        and hyper_input_normed.is_contiguous()
        and w_down.is_contiguous()
        and w_up.is_contiguous()
    )


def fused_hc_mix(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
) -> torch.Tensor:
    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    rows_pad = 16
    split_k = _FUSED_MIX_SPLIT_K
    device = hyper_input_normed.device
    num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    partial = torch.empty(
        (split_k, rows_pad, lowrank), dtype=torch.float32, device=device
    )
    t_raw = torch.empty((rows_pad, lowrank), dtype=torch.float32, device=device)
    out = torch.empty((rows, hs), dtype=hyper_input_normed.dtype, device=device)
    if rows == 0:
        return out
    _hc_mix_persistent_kernel[(num_ctas,)](
        hyper_input_normed,
        w_down,
        w_up,
        partial,
        t_raw,
        out,
        _get_counters(device),
        k,
        lowrank,
        hs,
        rows,
        num_ctas,
        1.0 / hc,
        ROWS=rows_pad,
        HC=hc,
        BLOCK_N=32,
        BLOCK_K=256,
        BLOCK_J=32,
        BLOCK_R=64,
        SPLIT_K=split_k,
        num_warps=8,
    )
    return out
