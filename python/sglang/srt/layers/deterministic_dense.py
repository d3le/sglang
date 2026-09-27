"""Batch- and chunk-invariant routing for dense GEMMs.

Two opt-in switches make the dense GEMMs of a forward independent of how the
rows were batched or chunked, which is what bitwise-reproducible serving with
prefix caching needs (the same token must get the same value in a solo run, a
batched run, a chunked prefill and after a partial prefix hit):

``SGLANG_DETERMINISTIC_BF16_LINEAR``
    Every unquantized BF16 dense GEMM (``UnquantizedLinearMethod`` layers and
    ``F.linear`` calls, e.g. ``nn.Linear`` modules) goes through the
    deterministic ``hc_gemm`` dispatcher: Triton tiles for small M and CuTe
    kernels for large M, each with a fixed reduction order.  Calls with a bias,
    or at a large M for a shape the dispatcher has not compiled, use
    ``matmul_persistent`` (also a fixed reduction order).

``SGLANG_FP8_POSITIONAL_PREFILL``
    ModelOpt FP8 linears on SM12x normally use the flashinfer bmm (cuBLASLt)
    path, whose tile choice depends on M.  With this switch a prefill token row
    uses that stock path when its absolute position is at least
    ``SGLANG_FP8_POSITIONAL_MIN_POS`` (default 256) and the channelwise path
    otherwise; stock calls with fewer than ``SGLANG_FP8_POSITIONAL_PAD_ROWS``
    (default 64) rows are zero-padded so the kernel stays in its row-invariant
    regime.  A token's path then depends only on its own position, so chunking
    and prefix hits cannot change it.  Every other forward (decode, target
    verify, draft steps) uses the channelwise path.
"""

from __future__ import annotations

import contextlib
import logging

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_installed = False
_bf16_active = False
_fp8_active = False
_weight_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

# Prefill row layout of the running forward (fp8 positional routing).
_is_prefill = False
_segments: list[int] | None = None
_prefixes: list[int] | None = None


def install() -> None:
    """Install the enabled routings once per process (idempotent)."""

    global _installed
    if _installed:
        return
    _installed = True
    if envs.SGLANG_DETERMINISTIC_BF16_LINEAR.get():
        _install_bf16()
    if envs.SGLANG_FP8_POSITIONAL_PREFILL.get():
        _install_fp8_positional()
    if _bf16_active or _fp8_active:
        logger.info(
            "Deterministic dense routing: bf16 dispatcher=%s, fp8 positional prefill=%s",
            _bf16_active,
            _fp8_active,
        )


def forward_context(forward_batch):
    """Record the prefill row layout for the duration of one model forward."""

    if not _fp8_active:
        return contextlib.nullcontext()
    return _prefill_rows(forward_batch)


@contextlib.contextmanager
def _prefill_rows(forward_batch):
    global _is_prefill, _segments, _prefixes
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    previous = (_is_prefill, _segments, _prefixes)
    mode = forward_batch.forward_mode
    segments = prefixes = None
    if mode == ForwardMode.EXTEND:
        plens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        if plens is None:
            tensor = getattr(forward_batch, "extend_prefix_lens", None)
            if tensor is not None:
                plens = tensor.tolist()
        if plens is not None:
            prefixes = [int(value) for value in plens]
        lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if lens is None:
            tensor = getattr(forward_batch, "extend_seq_lens", None)
            if tensor is not None:
                lens = tensor.tolist()
        if lens:
            segments = [int(value) for value in lens]
    _is_prefill = mode in (ForwardMode.EXTEND, ForwardMode.MIXED)
    _segments, _prefixes = segments, prefixes
    try:
        yield
    finally:
        _is_prefill, _segments, _prefixes = previous


# ---------------------------------------------------------------------------
# BF16: deterministic dispatcher


def _weight_transposed(weight: torch.Tensor) -> torch.Tensor:
    cached = _weight_cache.get(id(weight))
    if cached is None or cached[0] is not weight:
        cached = (weight, weight.t().contiguous())
        _weight_cache[id(weight)] = cached
    return cached[1]


def _eligible(x, weight, bias) -> bool:
    if not torch.is_tensor(x) or not torch.is_tensor(weight):
        return False
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.numel() == 0:
        return False
    if weight.dtype != torch.bfloat16 or x.shape[-1] != weight.shape[1]:
        return False
    if bias is not None and not (bias.is_cuda and bias.dtype == torch.bfloat16):
        return False
    return True


def _persistent_linear(x, weight, bias=None):
    from sglang.srt.batch_invariant_ops import matmul_persistent

    out = matmul_persistent(
        x.reshape(-1, x.shape[-1]), _weight_transposed(weight), bias
    )
    return out.view(*x.shape[:-1], weight.shape[0])


def deterministic_linear(x, weight, bias=None):
    from sglang.srt.layers import hc_gemm

    if bias is not None:
        return _persistent_linear(x, weight, bias)
    rows = x.numel() // x.shape[-1] if x.shape[-1] else 0
    if rows > hc_gemm.HC_GEMM_THRESHOLD:
        key = (x.device.index, weight.shape[0], weight.shape[1])
        if hc_gemm._COMPILED.get(key) is None:
            # Triton is far from cuBLAS at large M for an uncompiled shape.
            return _persistent_linear(x, weight, bias)
    return hc_gemm.hc_gemm(x, weight)


def _install_bf16() -> None:
    global _bf16_active
    import torch.nn.functional as F

    from sglang.srt.layers.hc_gemm import BF16_LINEAR_SHAPES, prepare
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    original_apply = UnquantizedLinearMethod.apply
    original_linear = F.linear

    def apply(self, layer, x, bias=None):
        if _eligible(x, layer.weight, bias):
            return deterministic_linear(x, layer.weight, bias)
        return original_apply(self, layer, x, bias)

    def linear(x, weight, bias=None):
        if _eligible(x, weight, bias):
            return deterministic_linear(x, weight, bias)
        return original_linear(x, weight, bias)

    UnquantizedLinearMethod.apply = apply
    F.linear = linear
    prepare(torch.device("cuda", torch.cuda.current_device()), BF16_LINEAR_SHAPES)
    _bf16_active = True


# ---------------------------------------------------------------------------
# FP8: position-intrinsic prefill routing


def _install_fp8_positional() -> None:
    global _fp8_active
    from torch import nn

    from sglang.srt.layers.quantization.fp8_utils import (
        apply_fp8_linear_bmm_flashinfer,
    )
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod
    from sglang.srt.layers.quantization.utils import convert_to_channelwise

    original_apply = ModelOptFp8LinearMethod.apply
    min_pos = envs.SGLANG_FP8_POSITIONAL_MIN_POS.get()
    pad_rows = envs.SGLANG_FP8_POSITIONAL_PAD_ROWS.get()
    converted: set[int] = set()

    def stock_scale(layer):
        scale = getattr(layer, "_positional_tensor_scale", None)
        if scale is None:
            scale = layer.weight_scale.detach().clone()
            layer._positional_tensor_scale = scale
        return scale

    def ensure_channelwise(layer):
        # Keep the per-tensor scale for the stock path, then move the layer to
        # the channelwise path (the default for every non-prefill forward).
        if getattr(layer, "use_flashinfer_bmm", False):
            stock_scale(layer)
            layer.use_flashinfer_bmm = False
            if id(layer) not in converted:
                converted.add(id(layer))
                try:
                    layer.weight_scale = nn.Parameter(
                        convert_to_channelwise(layer.weight_scale, layer.logical_widths),
                        requires_grad=False,
                    )
                except Exception:
                    pass

    def stock(x, layer, bias):
        return apply_fp8_linear_bmm_flashinfer(
            x, layer.weight, stock_scale(layer), layer.input_scale, bias
        )

    def channelwise(self, layer, x, bias):
        ensure_channelwise(layer)
        return original_apply(self, layer, x, bias)

    def stock_padded(x2d, layer, bias):
        rows = x2d.shape[0]
        if rows >= pad_rows:
            return stock(x2d, layer, bias)
        padded = x2d.new_zeros((pad_rows, x2d.shape[-1]))
        padded[:rows] = x2d
        return stock(padded, layer, bias)[:rows]

    def positional(self, layer, x, bias, rows):
        """Split prefill rows by absolute position; None when not applicable."""
        segments, prefixes = _segments, _prefixes
        if (
            not segments
            or prefixes is None
            or len(prefixes) != len(segments)
            or sum(segments) != rows
        ):
            return None
        flat = x.reshape(-1, x.shape[-1])
        channel_parts, stock_parts = [], []
        start = 0
        for size, prefix in zip(segments, prefixes):
            low = max(0, min(size, min_pos - prefix))
            if low:
                channel_parts.append((start, start + low))
            if size - low:
                stock_parts.append((start + low, start + size))
            start += size
        out = None
        for parts, run in (
            (channel_parts, lambda t: channelwise(self, layer, t, bias)),
            (stock_parts, lambda t: stock_padded(t, layer, bias)),
        ):
            if not parts:
                continue
            if len(parts) == 1 and parts[0] == (0, rows):
                return run(flat).view(*x.shape[:-1], -1)
            piece = (
                torch.cat([flat[a:b] for a, b in parts])
                if len(parts) > 1
                else flat[parts[0][0] : parts[0][1]]
            )
            result = run(piece)
            if out is None:
                out = result.new_empty((rows, result.shape[-1]))
            offset = 0
            for a, b in parts:
                out[a:b] = result[offset : offset + (b - a)]
                offset += b - a
        return out.view(*x.shape[:-1], out.shape[-1])

    def apply(self, layer, x, bias=None):
        if _is_prefill:
            rows = x.numel() // x.shape[-1] if x.shape[-1] else 0
            result = positional(self, layer, x, bias, rows)
            if result is not None:
                return result
            # Rows that are not the packed prompt tokens (e.g. the last-token
            # logits of each request) keep the channelwise path.
        return channelwise(self, layer, x, bias)

    ModelOptFp8LinearMethod.apply = apply
    _fp8_active = True
