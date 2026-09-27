import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.hc_mix_triton import (
    _FUSED_MIX_MAX_ROWS,
    fused_hc_mix,
    fused_hc_mix_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HC_COUNT = 4
HIDDEN_SIZE = 2560
LOWRANK = 320


def _reference_mix(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
    compute_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Mirrors GatedResidual._mix_compute in hyperconnection.py."""
    x = hyper_input_normed.to(compute_dtype)
    t = F.silu(F.linear(x, w_down.to(compute_dtype)) / hc)
    u = torch.sigmoid(F.linear(t, w_up.to(compute_dtype)))
    return (u.unflatten(-1, (hc, hs)) * x.unflatten(-1, (hc, hs))).mean(dim=-2)


def _make_inputs(num_tokens: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x = torch.randn(num_tokens, HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda")
    w_down = (
        torch.randn(LOWRANK, HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda") * 0.02
    )
    w_up = (
        torch.randn(HC_COUNT * HIDDEN_SIZE, LOWRANK, dtype=dtype, device="cuda") * 0.02
    )
    return x, w_down, w_up


_TOLERANCES = {
    torch.bfloat16: dict(rtol=1e-2, atol=5e-3),
    torch.float16: dict(rtol=2e-3, atol=1e-3),
}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 4, 7, _FUSED_MIX_MAX_ROWS])
def test_fused_hc_mix_matches_reference(dtype, num_tokens):
    x, w_down, w_up = _make_inputs(num_tokens, dtype)
    assert fused_hc_mix_supported(x, w_down, w_up)
    out = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    ref = _reference_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    torch.testing.assert_close(out.to(torch.float64), ref, **_TOLERANCES[dtype])


def test_fused_hc_mix_no_less_accurate_than_eager():
    """The fused kernel (fp32 accumulation throughout) must not be farther
    from the fp64 reference than the eager bf16 chain it replaces."""
    x, w_down, w_up = _make_inputs(8, torch.bfloat16)
    ref = _reference_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    fused = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    eager = _reference_mix(
        x, w_down, w_up, HC_COUNT, HIDDEN_SIZE, compute_dtype=torch.bfloat16
    )
    fused_err = (fused.to(torch.float64) - ref).abs().max()
    eager_err = (eager.to(torch.float64) - ref).abs().max()
    assert fused_err <= eager_err * 1.5 + 1e-6


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", range(1, _FUSED_MIX_MAX_ROWS + 1))
def test_fused_hc_mix_is_bitwise_stable_in_eager_and_graph(dtype, num_tokens):
    """The normal kernel must not depend on CTA order or graph replay state."""
    x, w_down, w_up = _make_inputs(num_tokens, dtype)
    call = lambda: fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    eager = [call().clone() for _ in range(100)]
    torch.cuda.synchronize()
    assert all(torch.equal(eager[0], value) for value in eager[1:])

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = call()
    graph_outputs = []
    for _ in range(100):
        graph.replay()
        graph_outputs.append(graph_out.clone())
    torch.cuda.synchronize()
    assert all(torch.equal(graph_outputs[0], value) for value in graph_outputs[1:])
    assert torch.equal(graph_outputs[0], eager[0])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_hc_mix_has_no_stale_workspace(dtype):
    """Changing packed row counts must not expose a previous invocation."""
    inputs = {rows: _make_inputs(rows, dtype) for rows in (1, 8, 16)}
    clean = fused_hc_mix(*inputs[8], HC_COUNT, HIDDEN_SIZE).clone()
    fused_hc_mix(*inputs[16], HC_COUNT, HIDDEN_SIZE)
    fused_hc_mix(*inputs[1], HC_COUNT, HIDDEN_SIZE)
    after = fused_hc_mix(*inputs[8], HC_COUNT, HIDDEN_SIZE).clone()
    torch.cuda.synchronize()
    assert torch.equal(clean, after)


def test_fused_hc_mix_supports_four_concurrent_streams():
    """Separate requests must not share barrier state or deadlock."""
    x, w_down, w_up = _make_inputs(_FUSED_MIX_MAX_ROWS, torch.bfloat16)
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream() for _ in range(4)]
    all_outputs = []
    for stream in streams:
        with torch.cuda.stream(stream):
            stream_outputs = []
            for _ in range(10):
                stream_outputs.append(
                    fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
                )
            all_outputs.append(stream_outputs)
    for stream in streams:
        stream.synchronize()
    torch.cuda.synchronize()
    reference = all_outputs[0][0]
    assert all(
        torch.equal(reference, value)
        for stream_outputs in all_outputs
        for value in stream_outputs
    )


def test_fused_hc_mix_gate_rejects_prefill_rows():
    x, w_down, w_up = _make_inputs(_FUSED_MIX_MAX_ROWS + 1, torch.bfloat16)
    assert not fused_hc_mix_supported(x, w_down, w_up)


@pytest.mark.parametrize("num_tokens", [17, 32, 128])
def test_fused_hc_mix_leaves_large_batches_on_fallback(num_tokens):
    x, w_down, w_up = _make_inputs(num_tokens, torch.bfloat16)
    assert not fused_hc_mix_supported(x, w_down, w_up)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
