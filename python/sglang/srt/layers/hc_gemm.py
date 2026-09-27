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
    bm, bn, bk = _SMALL_TILE
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    _hc_gemm_small_kernel[grid](
        x, weight, out,
        m, n, k,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BM=bm, BN=bn, BK=bk,
        num_warps=4, num_stages=3,
    )


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
