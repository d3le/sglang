"""Regression: MTP tail must survive compact KV gathering, eager and graphed."""

import ast
from pathlib import Path
import unittest

import torch

from sglang.srt.layers.attention.qsa.mtp_indices import compact_sparse_indices
from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_fa2_cu_seqlens_triton,
    qwen_sparse_kv_extraction_compact_triton,
)


def shared_class():
    # Exercise the actual class without importing the serving backend's model
    # and distributed-runtime dependencies.
    import sglang.srt.layers.attention.qsa.mtp_indices as module

    path = Path(module.__file__).parent.parent / "qwen_sparse_attn_backend.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "QSAMTPSharedSparseIndices"
    )
    namespace = {"torch": torch, "compact_sparse_indices": compact_sparse_indices}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[cls.name]


class MTPIndexCompactionTest(unittest.TestCase):
    def test_shared_layout_cpu(self):
        state = shared_class()(
            layer_ids=[0], num_requests=1, token_topk=2051, tail_width=4, device="cpu"
        )
        seed = torch.full((1, 2051), -1, dtype=torch.int32)
        seed[0, :96] = torch.arange(96)
        state.capture(seed, torch.tensor([0]), torch.tensor([96]), 0)
        before = state.indices.clone()
        for step in range(4):
            actual = state.lookup(torch.tensor([0]), torch.tensor([96 + step]), 0)
            self.assertEqual(actual[0, : 97 + step].tolist(), list(range(97 + step)))
            self.assertTrue(bool((actual[0, 97 + step :] == -1).all()))
        self.assertTrue(torch.equal(before, state.indices))

    def test_cpu_stable_order_and_empty(self):
        x = torch.tensor([[7, -1, 3, 3, -2, 9], [-1] * 6], dtype=torch.int32)
        self.assertEqual(
            compact_sparse_indices(x).tolist(), [[7, 3, 3, 9, -1, -1], [-1] * 6]
        )
        self.assertEqual(compact_sparse_indices(x[:0]).shape, (0, 6))
        self.assertEqual(compact_sparse_indices(x[:, :0]).shape, (2, 0))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gpu_matches_reference(self):
        for width in (1, 17, 2055, 8195):
            torch.manual_seed(123)
            x = torch.randint(-7, 200, (5, width), dtype=torch.int32)
            x[0] = -1
            x[1] = 0
            expected = compact_sparse_indices(x)
            for dtype in (torch.int32, torch.int64):
                result = compact_sparse_indices(x.to(device="cuda", dtype=dtype)).cpu()
                self.assertTrue(torch.equal(result, expected.to(dtype)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_shared_tail_kv_gather_and_graph(self):
        cls = shared_class()
        # Include compression remainders, exact seed-width saturation, long
        # context, different request rows, and all draft steps.
        lengths = [96, 97, 98, 99, 2048, 2051, 262140]
        width, tail_width = 2051, 4
        state = cls(
            layer_ids=[0],
            num_requests=len(lengths),
            token_topk=width,
            tail_width=tail_width,
            device="cuda",
        )
        requests = torch.arange(len(lengths), device="cuda")
        lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        seeds = torch.full((len(lengths), width), -1, dtype=torch.int32)
        for i, length in enumerate(lengths):
            count = min(length, width)
            if length > width:
                count = 2048
            seeds[i, :count] = torch.arange(count)
        seeds = seeds.cuda()
        state.capture(seeds, requests, lens, 0)
        frozen = state.indices.clone()
        positions = lens.clone()
        # Share a deterministic logical->physical map to keep the test small.
        max_len = max(lengths) + tail_width + 1
        mapping = torch.arange(max_len, device="cuda", dtype=torch.int32).repeat(
            len(lengths), 1
        )
        kv = (torch.arange(max_len, device="cuda") % 127).to(torch.bfloat16)
        kv = kv[:, None, None].expand(-1, 1, 16).contiguous()
        for step in range(tail_width):
            positions.copy_(lens + step)
            eager = state.lookup(requests, positions, 0)
            expected_rows = []
            for i, length in enumerate(lengths):
                seed = seeds[i][seeds[i] >= 0]
                row = torch.cat(
                    (seed, torch.arange(length, length + step + 1, device="cuda"))
                )
                expected_rows.append(row)
                self.assertTrue(torch.equal(eager[i, : row.numel()], row))
                self.assertTrue(bool((eager[i, row.numel() :] == -1).all()))
            seq_lens = positions + 1
            counts = torch.empty_like(lens)
            cu_k = torch.empty(len(lengths) + 1, dtype=torch.int32, device="cuda")
            qwen_sparse_fa2_cu_seqlens_triton(
                seq_lens, eager, counts, cu_k, len(lengths), width + tail_width
            )
            capacity = len(lengths) * (width + tail_width)
            packed_k = torch.full(
                (capacity, 1, 16), float("nan"), device="cuda", dtype=kv.dtype
            )
            packed_v = torch.full_like(packed_k, float("nan"))
            qwen_sparse_kv_extraction_compact_triton(
                kv,
                kv,
                mapping,
                requests,
                eager,
                seq_lens,
                cu_k,
                packed_k,
                packed_v,
                len(lengths),
                width + tail_width,
            )
            count = int(cu_k[-1])
            expected_kv = kv[torch.cat(expected_rows).long()]
            self.assertTrue(torch.equal(packed_k[:count], expected_kv))
            self.assertTrue(torch.equal(packed_v[:count], expected_kv))
            self.assertTrue(bool(torch.isfinite(packed_k[:count]).all()))
        self.assertTrue(torch.equal(frozen, state.indices))
        # Capture lookup and the real gather, then vary positions at replay.
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            graphed = state.lookup(requests, positions, 0)
            graph_lens = positions + 1
            qwen_sparse_fa2_cu_seqlens_triton(
                graph_lens, graphed, counts, cu_k, len(lengths), width + tail_width
            )
            qwen_sparse_kv_extraction_compact_triton(
                kv,
                kv,
                mapping,
                requests,
                graphed,
                graph_lens,
                cu_k,
                packed_k,
                packed_v,
                len(lengths),
                width + tail_width,
            )
        for step in (0, 3, 1, 2, 0):
            positions.copy_(lens + step)
            packed_k.fill_(float("nan"))
            packed_v.fill_(float("nan"))
            graph.replay()
            self.assertTrue(torch.equal(graphed, state.lookup(requests, positions, 0)))
            expected = torch.cat([kv[r[r >= 0].long()] for r in graphed])
            self.assertTrue(torch.equal(packed_k[: expected.shape[0]], expected))
            self.assertTrue(torch.equal(packed_v[: expected.shape[0]], expected))


if __name__ == "__main__":
    unittest.main()
