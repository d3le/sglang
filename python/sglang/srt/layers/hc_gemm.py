"""Deterministic hyperconnection GEMM: Triton small-M + CuTe large-M.

Both backends use mma.sync m16n8k16 with fp32 accumulation and an ascending K
loop, so they are bitwise identical for the same logical rows (verified on
GB10/sm_121).  Rows up to ``HC_GEMM_THRESHOLD`` use the Triton kernel (best at
small M); larger row counts use the CuTe DSL kernel (cuBLAS-level at prefill
sizes).  ``prepare`` compiles/warms both backends during model load so that no
JIT happens on the request path.
"""

from __future__ import annotations

import os
from typing import Iterable

import torch
import triton
import triton.language as tl

os.environ.setdefault("CUTE_DSL_CACHE_DIR", "/root/.cache/sglang/cute-dsl")

HC_GEMM_THRESHOLD = 256
_SMALL_TILE = (16, 32, 64)
# Small-M shapes measured better with narrower tiles on GB10 (probe_smallm):
_SMALL_TILES = {
    (320, 10240): (8, 16, 512),
    (10240, 320): (8, 128, 64),
    (4, 10240): (8, 16, 64),
    # Decode shapes of the deterministic bf16 linears: a deeper K pipeline reaches 227-250
    # GB/s; every tile was bitwise equal to the old (16, 32, 64) at M=4/16.
    (96, 2560): (8, 16, 256),
    (512, 2560): (8, 32, 128),
    (640, 2560): (8, 32, 256),
    (1280, 2560): (8, 32, 256),
    (2560, 2560): (8, 16, 512),
    (2560, 6144): (8, 16, 256),
    (10240, 2560): (8, 16, 512),
    (13312, 2560): (8, 16, 512),
}
# (num_warps, num_stages) per shape; the default stays 4/3.  The decode down
# GEMM (narrow N, long K) needs BK=512 with 4 pipeline stages to reach ~217 GB/s
# on GB10 (63.5 -> 41.5-44 us cold, bitwise vs the old tile at
# M=1/4/16/255/256/257/8192).
_SMALL_TILE_WS = {
    (320, 10240): (4, 4),
    (96, 2560): (4, 4),
    (512, 2560): (4, 6),
    (640, 2560): (4, 4),
    (1280, 2560): (4, 4),
    (2560, 2560): (4, 3),
    (2560, 6144): (4, 6),
    (10240, 2560): (4, 4),
    (13312, 2560): (4, 4),
}
_CUTE_TILES = {
    # hyperconnection
    (320, 10240): (64, 128, 64),
    (10240, 320): (128, 128, 32),
    # bf16_linear sites (shared expert, router, protected GDN projections
    # plus the shapes observed at prefill by the diagnostic probe)
    (1280, 2560): (128, 128, 32),
    (2560, 640): (128, 128, 32),
    (512, 2560): (128, 128, 32),
    (96, 2560): (128, 128, 32),
    (2560, 6144): (128, 128, 32),
    (10240, 2560): (128, 128, 32),
    (13312, 2560): (128, 128, 32),
    (2560, 2560): (128, 128, 32),
    (640, 2560): (128, 128, 32),
}
BF16_LINEAR_SHAPES = (
    (1280, 2560),
    (2560, 640),
    (512, 2560),
    (96, 2560),
    (2560, 6144),
    (10240, 2560),
    (13312, 2560),
    (2560, 2560),
    (640, 2560),
)
_COMPILED: dict[tuple, object] = {}
_PREPARED: set[tuple] = set()


@triton.jit
def _hc_gemm_small_kernel(
    a_ptr, w_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_wn, stride_wk, stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    mask_m = offs_m < M
    mask_n = offs_n < N
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)
        acc = tl.dot(a, w, acc)
        a_ptrs += BK * stride_ak
        w_ptrs += BK * stride_wk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def _launch_small(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    m, k = x.shape
    n = weight.shape[0]
    bm, bn, bk = _SMALL_TILES.get((n, k), _SMALL_TILE)
    warps, stages = _SMALL_TILE_WS.get((n, k), (4, 3))
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    _hc_gemm_small_kernel[grid](
        x, weight, out,
        m, n, k,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BM=bm, BN=bn, BK=bk,
        num_warps=warps, num_stages=stages,
    )


_GATE_SPLITS = 32
_GATE_TILE_BM = 16
_COMBINE_BN = 512
_GATE_SCRATCH: dict[int, "torch.Tensor"] = {}
# Captured CUDA graphs hold the scratch pointer; a later prefill call grows the
# buffer, so the old tensor must stay alive instead of being freed.
_GATE_SCRATCH_RETIRED: list["torch.Tensor"] = []


@triton.jit
def _round_bf16(x):
    """Force a round-to-nearest-even f32 -> bf16 conversion.

    ``x.to(tl.bfloat16).to(tl.float32)`` can be folded back to the f32 value by
    the compiler (the intermediate rounding is lost), so the eager rounding
    chain cannot be reproduced with a plain cast.  The inline PTX keeps it.
    """

    return tl.inline_asm_elementwise(
        "cvt.rn.bf16.f32 $0, $1;", "=h,r", [x], dtype=tl.bfloat16, is_pure=True, pack=1
    )


@triton.jit
def _hc_gate_partial_kernel(
    normed_ptr, weight_ptr, partial_ptr,
    M, K, HC,
    stride_nm, stride_nk, stride_wn, stride_wk,
    stride_ps, stride_pm, stride_pc,
    SPLITS: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, NP: tl.constexpr,
):
    s = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < M
    offs_n = tl.arange(0, NP)
    offs_k = tl.arange(0, BK)
    chunk = tl.cdiv(K, SPLITS)
    start = s * chunk
    stop = tl.minimum(start + chunk, K)
    acc = tl.zeros((BM, NP), dtype=tl.float32)
    for k0 in range(start, stop, BK):
        offs = k0 + offs_k
        mask_k = offs < K
        a = tl.load(
            normed_ptr + offs_m[:, None] * stride_nm + offs[None, :] * stride_nk,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        w = tl.load(
            weight_ptr + offs_n[None, :] * stride_wn + offs[:, None] * stride_wk,
            mask=(offs_n[None, :] < HC) & mask_k[:, None], other=0.0,
        )
        acc = tl.dot(a, w, acc)
    tl.store(
        partial_ptr + s * stride_ps + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pc,
        acc, mask=mask_m[:, None] & (offs_n[None, :] < HC),
    )


@triton.jit
def _hc_combine_apply_kernel(
    block_ptr, residual_ptr, partial_ptr, out_ptr,
    M, HS,
    stride_bm, stride_bh,
    stride_rm, stride_rk,
    stride_ps, stride_pm, stride_pc,
    stride_om, stride_ok,
    SPLITS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, HC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < M
    offs_hc = tl.arange(0, HC)
    gate = tl.zeros((BM, HC), dtype=tl.float32)
    for s in range(SPLITS):
        gate += tl.load(
            partial_ptr + s * stride_ps + offs_m[:, None] * stride_pm
            + offs_hc[None, :] * stride_pc,
            mask=mask_m[:, None], other=0.0,
        )
    # Reproduce the eager rounding chain: bf16 GEMM out, /hc -> bf16,
    # sigmoid -> bf16, *2 -> bf16.
    gate = _round_bf16(gate)
    scaled = _round_bf16(gate.to(tl.float32) / HC)
    sig = _round_bf16(tl.sigmoid(scaled.to(tl.float32)))
    weight_out = _round_bf16(2.0 * sig.to(tl.float32))

    offs_hs = pid_n * BN + tl.arange(0, BN)
    mask_hs = offs_hs < HS
    mask = mask_m[:, None] & mask_hs[None, :]
    block = tl.load(
        block_ptr + offs_m[:, None] * stride_bm + offs_hs[None, :] * stride_bh,
        mask=mask, other=0.0,
    )
    for c in tl.static_range(HC):
        residual = tl.load(
            residual_ptr + offs_m[:, None] * stride_rm
            + (c * HS + offs_hs)[None, :] * stride_rk,
            mask=mask, other=0.0,
        )
        gate_c = tl.sum(
            tl.where(offs_hc[None, :] == c, weight_out, 0.0), axis=1
        )
        injection = _round_bf16(
            block.to(tl.float32) * gate_c[:, None].to(tl.float32)
        )
        combined = _round_bf16(residual.to(tl.float32) + injection.to(tl.float32))
        tl.store(
            out_ptr + offs_m[:, None] * stride_om
            + (c * HS + offs_hs)[None, :] * stride_ok,
            combined, mask=mask,
        )


def hc_combine_small(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc: int,
    hs: int,
) -> torch.Tensor:
    """Deterministic fused combine gate + apply for any row count."""

    m = block_output.shape[0]
    n = inject_weight.shape[0]
    if hc > 16 or n != hc:
        raise ValueError("hc_combine_small only supports hc == N <= 16")
    device = block_output.device
    normed = normed_residual.detach().contiguous()
    block = block_output.contiguous()
    scratch = _GATE_SCRATCH.get(device.index)
    rows_cap = max(m, HC_GEMM_THRESHOLD)
    if scratch is None or scratch.shape[0] != _GATE_SPLITS or scratch.shape[1] < rows_cap:
        if scratch is not None:
            _GATE_SCRATCH_RETIRED.append(scratch)
        scratch = torch.empty(
            (_GATE_SPLITS, rows_cap, n), device=device, dtype=torch.float32
        )
        _GATE_SCRATCH[device.index] = scratch
    _hc_gate_partial_kernel[(_GATE_SPLITS, triton.cdiv(m, _GATE_TILE_BM))](
        normed, inject_weight.detach().contiguous(), scratch,
        m, normed.shape[1], n,
        normed.stride(0), normed.stride(1),
        inject_weight.stride(0), inject_weight.stride(1),
        scratch.stride(0), scratch.stride(1), scratch.stride(2),
        SPLITS=_GATE_SPLITS, BM=_GATE_TILE_BM, BK=64, NP=16,
        num_warps=4, num_stages=3,
    )
    out = torch.empty_like(residual)
    _hc_combine_apply_kernel[
        (triton.cdiv(m, _GATE_TILE_BM), triton.cdiv(hs, _COMBINE_BN))
    ](
        block, residual, scratch, out,
        m, hs,
        block.stride(0), block.stride(1),
        residual.stride(0), residual.stride(1),
        scratch.stride(0), scratch.stride(1), scratch.stride(2),
        out.stride(0), out.stride(1),
        SPLITS=_GATE_SPLITS, BM=_GATE_TILE_BM, BN=_COMBINE_BN, HC=hc,
        num_warps=4, num_stages=3,
    )
    return out


_MIX_TAIL_BM = 8
_MIX_TAIL_BN = 512


@triton.jit
def _hc_mix_tail_kernel(
    mix_ptr, norm_ptr, out_ptr,
    M, HS,
    stride_mm, stride_nm, stride_om,
    HC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_j = pid_n * BN + tl.arange(0, BN)
    mask = (offs_m < M)[:, None] & (offs_j < HS)[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for c in tl.static_range(HC):
        offs = c * HS + offs_j
        w = tl.load(
            mix_ptr + offs_m[:, None] * stride_mm + offs[None, :],
            mask=mask, other=0.0,
        )
        n = tl.load(
            norm_ptr + offs_m[:, None] * stride_nm + offs[None, :],
            mask=mask, other=0.0,
        )
        sig = _round_bf16(tl.sigmoid(w.to(tl.float32)))
        prod = _round_bf16(sig.to(tl.float32) * n.to(tl.float32))
        acc += prod.to(tl.float32)
    out = _round_bf16(acc / HC)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_j[None, :], out, mask=mask
    )


def hc_mix_tail(
    mix_weight: torch.Tensor, normed: torch.Tensor, hc: int, hs: int
) -> torch.Tensor:
    """Reproduce the eager mix tail (sigmoid -> mul -> mean) bitwise at any M."""

    m = mix_weight.shape[0]
    out = torch.empty((m, hs), device=mix_weight.device, dtype=torch.bfloat16)
    if m == 0:
        return out
    mix = mix_weight.detach().contiguous()
    norm = normed.detach().contiguous()
    grid = (triton.cdiv(m, _MIX_TAIL_BM), triton.cdiv(hs, _MIX_TAIL_BN))
    _hc_mix_tail_kernel[grid](
        mix, norm, out,
        m, hs,
        mix.stride(0), norm.stride(0), out.stride(0),
        HC=hc, BM=_MIX_TAIL_BM, BN=_MIX_TAIL_BN,
        num_warps=4, num_stages=2,
    )
    return out


def _self_check(compiled, n: int, k: int) -> None:
    """Reject a broken CuTe kernel at load time instead of during a request."""

    m = 320
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    from sglang.srt.layers import hc_gemm_cute

    mA, mB, mC = hc_gemm_cute._cute_tensors(x, w, out)
    import cuda.bindings.driver as cuda_driver

    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(mA, mB, mC, stream)
    torch.cuda.synchronize()
    ref = (x.float() @ w.float().t()).to(torch.bfloat16)
    if not torch.allclose(out.float(), ref.float(), atol=0.5, rtol=0.05):
        raise RuntimeError("CuTe self-check mismatch")


def _cute_compile(n: int, k: int):
    tile = _CUTE_TILES[(n, k)]
    from sglang.srt.layers import hc_gemm_cute

    return hc_gemm_cute.compile_hc_gemm(tile, 4)


def prepare(device: torch.device, shapes: Iterable[tuple[int, int]]) -> None:
    """Compile and warm both backends; call during model load, never per request."""

    if not torch.cuda.is_available():
        return
    device = torch.device(device)
    for n, k in shapes:
        key = (device.index, n, k)
        if key in _PREPARED:
            continue
        os.makedirs(os.environ["CUTE_DSL_CACHE_DIR"], exist_ok=True)
        x = torch.zeros(4, k, device=device, dtype=torch.bfloat16)
        w = torch.zeros(n, k, device=device, dtype=torch.bfloat16)
        out = torch.empty(4, n, device=device, dtype=torch.bfloat16)
        _launch_small(x, w, out)
        import fcntl

        if n % 8 != 0 or k % 8 != 0:
            _COMPILED[key] = None
            print(f"[hc-gemm] {(n, k)} stays on Triton (alignment)", flush=True)
            _PREPARED.add(key)
            continue
        lock_path = os.path.join(os.environ["CUTE_DSL_CACHE_DIR"], ".hc_gemm.lock")
        try:
            with open(lock_path, "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    compiled = _cute_compile(n, k)
                    _self_check(compiled, n, k)
                    _COMPILED[key] = compiled
                    print(
                        f"[hc-gemm] CuTe compiled for {(n, k)} "
                        f"tile={_CUTE_TILES[(n, k)]}",
                        flush=True,
                    )
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
        except Exception as error:  # noqa: BLE001 - CuTe is optional (non-GB10 images)
            _COMPILED[key] = None
            print(f"[hc-gemm] CuTe compile FAILED for {(n, k)}: {error!r}", flush=True)
        _PREPARED.add(key)


def hc_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return ``x @ weight.T`` in bf16 with a deterministic kernel choice."""

    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.numel() == 0
        or x.shape[-1] != weight.shape[1]
    ):
        return torch.nn.functional.linear(x, weight)
    if x.dim() != 2:
        out = hc_gemm(x.reshape(-1, x.shape[-1]), weight)
        return out.view(*x.shape[:-1], weight.shape[0])
    x = x.detach().contiguous()
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    if m == 0:
        return out
    key = (x.device.index, n, k)
    compiled = _COMPILED.get(key)
    if (
        m <= HC_GEMM_THRESHOLD
        or compiled is None
        or n % 8 != 0
        or k % 8 != 0
    ):
        # The CuTe epilogue needs 16-byte alignment on the C row stride and on
        # the K vector; narrow shapes stay on the Triton kernel.
        _launch_small(x, weight, out)
        return out
    from sglang.srt.layers import hc_gemm_cute

    mA, mB, mC = hc_gemm_cute._cute_tensors(x, weight, out)
    import cuda.bindings.driver as cuda_driver

    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(mA, mB, mC, stream)
    return out
