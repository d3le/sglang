"""Stable compaction of sparse token indices, including MTP's appended tail."""

import torch
import triton
import triton.language as tl


@triton.jit
def _compact_indices(
    indices, output, width: tl.constexpr, stride: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    values = tl.load(indices + row * stride + col, col < width, other=-1)
    valid = (col < width) & (values >= 0)
    rank = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.sum(valid.to(tl.int32))
    # The scatter writes [0, count); padding writes [count, width). These
    # disjoint ranges avoid both uninitialized output and inter-thread races.
    tl.store(output + row * width + col, -1, (col >= count) & (col < width))
    tl.store(output + row * width + rank, values, valid)


def compact_sparse_indices(indices: torch.Tensor) -> torch.Tensor:
    """Move nonnegative entries to the front, preserving order and duplicates.

    Unlike token sorting/selection this only removes holes; the selected KV
    sequence is unchanged. No host readback is needed during CUDA graph replay.
    """
    if indices.ndim != 2 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("sparse indices must be a 2D int32/int64 tensor")
    rows, width = indices.shape
    if not rows or not width:
        return indices.clone()
    if not indices.is_cuda or width > 65536:
        order = torch.argsort((indices < 0).to(torch.int32), dim=1, stable=True)
        return indices.gather(1, order).clamp_min(-1)
    indices = indices.contiguous()
    output = torch.empty_like(indices)
    _compact_indices[(rows,)](
        indices, output, width, indices.stride(0), triton.next_power_of_2(width)
    )
    return output
