"""Validated reuse of the file-backed Qwen4-Exp PLE table (CPU part).

Covers the manifest contract: what is recorded, when a table is reused, what
invalidates it, that a crash before finalization leaves no manifest, and that
the advisory lock keeps a second process from rebuilding a table that is in
use. All tests are CPU-only; no CUDA or model weights are needed.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

import torch

from sglang.srt.model_loader import weight_utils
from sglang.srt.models.qwen4_exp_ple_table import (
    PLE_REUSE_MANIFEST_SUFFIX,
    PLE_REUSE_MANIFEST_VERSION,
    _PleTableLock,
    _reset_reuse_state_for_tests,
    allocate_ple_host_table,
    collect_ple_source_manifest,
    finalize_ple_table_reuse,
    ple_table_file_name,
    reuse_skippable_weight,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SHARD_ROWS = 8
SHARD_DIM = 4
SHARDS_PER_FILE = 2
NUM_FILES = 2
TOTAL_ROWS = SHARD_ROWS * SHARDS_PER_FILE * NUM_FILES


def _shard_name(index: int) -> str:
    return (
        "model.language_model.layers.1.ple.ple_embedding"
        f".ngram_embedding.shard_{index}.weight"
    )


def _write_checkpoint(root: str, *, shard_rows: int = SHARD_ROWS) -> dict:
    import safetensors.torch

    weight_map = {}
    index = 0
    for file_index in range(NUM_FILES):
        tensors = {}
        for _ in range(SHARDS_PER_FILE):
            name = _shard_name(index)
            tensors[name] = torch.randn(shard_rows, SHARD_DIM).to(torch.float8_e4m3fn)
            weight_map[name] = f"model-plefp8-{file_index:05d}.safetensors"
            index += 1
        safetensors.torch.save_file(
            tensors, os.path.join(root, f"model-plefp8-{file_index:05d}.safetensors")
        )
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as handle:
        json.dump({"metadata": {}, "weight_map": weight_map}, handle)
    return weight_map


class PleTableReuseTestBase(CustomTestCase):
    def setUp(self):
        _reset_reuse_state_for_tests()
        self.addCleanup(_reset_reuse_state_for_tests)
        self._checkpoint = tempfile.TemporaryDirectory()
        self.addCleanup(self._checkpoint.cleanup)
        self._tables = tempfile.TemporaryDirectory()
        self.addCleanup(self._tables.cleanup)
        self.weight_map = _write_checkpoint(self._checkpoint.name)
        self.reuse_env = mock.patch.dict(
            os.environ, {"SGLANG_QWEN4_PLE_FILE_REUSE": "1"}
        )
        self.reuse_env.start()
        self.addCleanup(self.reuse_env.stop)

    def table_path(self, source_dir=None):
        directory = source_dir or self._tables.name
        return os.path.join(
            directory,
            ple_table_file_name((TOTAL_ROWS, SHARD_DIM), torch.float8_e4m3fn, "rows0-16"),
        )

    def allocate(self, source_dir=None):
        return allocate_ple_host_table(
            (TOTAL_ROWS, SHARD_DIM),
            torch.float8_e4m3fn,
            "file",
            self._tables.name,
            tag="rows0-16",
            source_dir=source_dir or self._checkpoint.name,
        )

    def build_and_publish(self):
        table = self.allocate()
        self.assertFalse(table._sglang_ple_reuse_valid)
        rows = torch.arange(TOTAL_ROWS * SHARD_DIM, dtype=torch.float32).reshape(
            TOTAL_ROWS, SHARD_DIM
        )
        table.copy_(rows.to(torch.float8_e4m3fn))
        finalize_ple_table_reuse()
        return table

    def manifest(self, source_dir=None):
        with open(self.table_path(source_dir) + PLE_REUSE_MANIFEST_SUFFIX) as handle:
            return json.load(handle)


class TestManifestCollection(PleTableReuseTestBase):
    def test_records_source_files_and_shard_headers(self):
        manifest = collect_ple_source_manifest(self._checkpoint.name)
        self.assertEqual(len(manifest["sources"]), NUM_FILES)
        for source in manifest["sources"]:
            self.assertEqual(len(source["keys"]), SHARDS_PER_FILE)
            for key in source["keys"]:
                self.assertEqual(key["dtype"], "F8_E4M3")
                self.assertEqual(key["shape"], [SHARD_ROWS, SHARD_DIM])
        names = [key["name"] for source in manifest["sources"] for key in source["keys"]]
        self.assertEqual(sorted(names), sorted(self.weight_map))

    def test_missing_index_returns_none(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertIsNone(collect_ple_source_manifest(empty))
        self.assertIsNone(collect_ple_source_manifest(None))

    def test_missing_source_file_returns_none(self):
        os.unlink(os.path.join(self._checkpoint.name, "model-plefp8-00000.safetensors"))
        self.assertIsNone(collect_ple_source_manifest(self._checkpoint.name))


class TestManifestLifecycle(PleTableReuseTestBase):
    def test_build_publishes_manifest_atomically(self):
        self.build_and_publish()
        path = self.table_path()
        manifest = self.manifest()
        self.assertEqual(manifest["version"], PLE_REUSE_MANIFEST_VERSION)
        self.assertEqual(manifest["table"]["shape"], [TOTAL_ROWS, SHARD_DIM])
        self.assertEqual(manifest["table"]["dtype"], "float8_e4m3fn")
        self.assertEqual(manifest["table"]["tag"], "rows0-16")
        leftovers = [n for n in os.listdir(self._tables.name) if ".tmp" in n]
        self.assertEqual(leftovers, [])
        self.assertTrue(os.path.exists(path))

    def test_valid_manifest_is_reused(self):
        built = self.build_and_publish()
        rows = built.clone()
        table = self.allocate()
        self.assertTrue(table._sglang_ple_reuse_valid)
        self.assertTrue(torch.equal(table, rows))
        self.assertTrue(reuse_skippable_weight(_shard_name(0)))
        self.assertFalse(reuse_skippable_weight("model.layers.0.self_attn.q_proj.weight"))
        self.assertFalse(reuse_skippable_weight(_shard_name(0).replace(".ple.", ".other.")))

    def test_manifest_records_sample_digest(self):
        self.build_and_publish()
        self.assertEqual(len(self.manifest()["sample_digest"]), 64)

    def test_sample_rows_are_exact_for_large_tables(self):
        from sglang.srt.models.qwen4_exp_ple_table import _sample_row_index

        for rows in (1, 2, 17, 4096, 320001536):
            index = _sample_row_index(rows)
            self.assertEqual(index.dtype, torch.int64)
            self.assertEqual(int(index[0]), 0)
            self.assertEqual(int(index[-1]), rows - 1)
            self.assertEqual(int(index.numel()), min(4096, rows))

    def test_digest_ignores_the_default_device(self):
        # Model init allocates the table under a CUDA default-device context;
        # the sample index must still be built next to the CPU table.
        self.build_and_publish()
        _reset_reuse_state_for_tests()
        with torch.device("meta"):
            table = self.allocate()
        self.assertTrue(table._sglang_ple_reuse_valid)

    def test_changed_table_content_invalidates(self):
        built = self.build_and_publish()
        # Row 0 is always sampled; change it after the manifest was published.
        built.view(torch.uint8)[0].fill_(0x55)
        _reset_reuse_state_for_tests()
        table = self.allocate()
        self.assertFalse(table._sglang_ple_reuse_valid)
        self.assertFalse(reuse_skippable_weight(_shard_name(0)))
        self.assertFalse(
            os.path.exists(self.table_path() + PLE_REUSE_MANIFEST_SUFFIX)
        )

    def test_legacy_manifest_without_digest_gets_one(self):
        self.build_and_publish()
        path = self.table_path() + PLE_REUSE_MANIFEST_SUFFIX
        legacy = self.manifest()
        digest = legacy.pop("sample_digest")
        with open(path, "w") as handle:
            json.dump(legacy, handle)
        _reset_reuse_state_for_tests()
        table = self.allocate()
        self.assertTrue(table._sglang_ple_reuse_valid)
        self.assertEqual(self.manifest()["sample_digest"], digest)

    def test_skip_hook_is_registered_with_the_iterator(self):
        self.build_and_publish()
        self.allocate()
        self.assertIn(reuse_skippable_weight, weight_utils.WEIGHT_KEY_SKIP_HOOKS)
        self.assertTrue(weight_utils._skip_weight_key(_shard_name(1)))
        _reset_reuse_state_for_tests()
        self.assertFalse(weight_utils._skip_weight_key(_shard_name(1)))

    def test_without_manifest_nothing_is_reused(self):
        table = self.allocate()
        self.assertFalse(table._sglang_ple_reuse_valid)
        self.assertFalse(reuse_skippable_weight(_shard_name(0)))

    def test_source_mtime_change_invalidates(self):
        self.build_and_publish()
        _reset_reuse_state_for_tests()
        source = os.path.join(self._checkpoint.name, "model-plefp8-00000.safetensors")
        stat = os.stat(source)
        os.utime(source, ns=(stat.st_atime_ns + 10**9, stat.st_mtime_ns + 10**9))
        table = self.allocate()
        self.assertFalse(table._sglang_ple_reuse_valid)

    def test_source_shard_shape_change_invalidates(self):
        self.build_and_publish()
        _reset_reuse_state_for_tests()
        # Rebuild one source file with a different shard shape; the header, not
        # the file size, is what the manifest records.
        import safetensors.torch

        rewritten = {
            name: torch.randn(SHARD_ROWS + 1, SHARD_DIM).to(torch.float8_e4m3fn)
            for name in self.weight_map
            if self.weight_map[name] == "model-plefp8-00000.safetensors"
        }
        safetensors.torch.save_file(
            rewritten, os.path.join(self._checkpoint.name, "model-plefp8-00000.safetensors")
        )
        table = self.allocate()
        self.assertFalse(table._sglang_ple_reuse_valid)

    def test_corrupt_manifest_invalidates(self):
        self.build_and_publish()
        _reset_reuse_state_for_tests()
        with open(self.table_path() + PLE_REUSE_MANIFEST_SUFFIX, "w") as handle:
            handle.write("{not json")
        table = self.allocate()
        self.assertFalse(table._sglang_ple_reuse_valid)

    def test_manifest_version_change_invalidates(self):
        self.build_and_publish()
        _reset_reuse_state_for_tests()
        manifest = self.manifest()
        manifest["version"] = PLE_REUSE_MANIFEST_VERSION + 1
        with open(self.table_path() + PLE_REUSE_MANIFEST_SUFFIX, "w") as handle:
            json.dump(manifest, handle)
        table = self.allocate()
        self.assertFalse(table._sglang_ple_reuse_valid)

    def test_forced_rebuild_invalidates_before_rewrite(self):
        self.build_and_publish()
        _reset_reuse_state_for_tests()
        with mock.patch.dict(
            os.environ, {"SGLANG_QWEN4_PLE_FILE_REUSE_FORCE_REBUILD": "1"}
        ):
            table = self.allocate()
            self.assertFalse(table._sglang_ple_reuse_valid)
            # Invalidated before the rewrite so a crash cannot leave a manifest
            # that still matches the sources.
            self.assertFalse(
                os.path.exists(self.table_path() + PLE_REUSE_MANIFEST_SUFFIX)
            )
            finalize_ple_table_reuse()
        self.assertTrue(os.path.exists(self.table_path() + PLE_REUSE_MANIFEST_SUFFIX))

    def test_crash_before_finalize_leaves_no_manifest(self):
        self.build_and_publish()
        _reset_reuse_state_for_tests()
        with mock.patch.dict(
            os.environ, {"SGLANG_QWEN4_PLE_FILE_REUSE_FORCE_REBUILD": "1"}
        ):
            table = self.allocate()
            self.assertFalse(table._sglang_ple_reuse_valid)
            # Simulate a crash before finalize: the filesystem keeps the
            # partially rewritten table and no manifest, while the process
            # state (and its flock) is gone.
            _reset_reuse_state_for_tests()
        table = self.allocate()
        self.assertFalse(table._sglang_ple_reuse_valid)

    def test_reuse_disabled_by_default(self):
        with mock.patch.dict(os.environ, {"SGLANG_QWEN4_PLE_FILE_REUSE": "0"}):
            table = self.allocate()
            self.assertFalse(table._sglang_ple_reuse_valid)
            self.assertFalse(
                os.path.exists(self.table_path() + PLE_REUSE_MANIFEST_SUFFIX)
            )


class TestReuseLock(PleTableReuseTestBase):
    def test_reader_blocks_writer(self):
        self.build_and_publish()
        _reset_reuse_state_for_tests()
        reader = _PleTableLock(self.table_path() + ".reuse.lock", 0.0)
        self.assertTrue(reader.acquire(exclusive=False))
        self.addCleanup(reader.release)
        with mock.patch.dict(
            os.environ, {"SGLANG_QWEN4_PLE_FILE_REUSE_FORCE_REBUILD": "1"}
        ):
            with self.assertRaises(RuntimeError):
                self.allocate()

    def test_writer_blocks_reader(self):
        self.build_and_publish()
        _reset_reuse_state_for_tests()
        writer = _PleTableLock(self.table_path() + ".reuse.lock", 0.0)
        self.assertTrue(writer.acquire(exclusive=True))
        self.addCleanup(writer.release)
        with self.assertRaises(RuntimeError):
            self.allocate()

    def test_lock_timeout_can_wait(self):
        self.build_and_publish()
        _reset_reuse_state_for_tests()
        writer = _PleTableLock(self.table_path() + ".reuse.lock", 0.0)
        self.assertTrue(writer.acquire(exclusive=True))
        self.addCleanup(writer.release)
        with mock.patch.dict(
            os.environ, {"SGLANG_QWEN4_PLE_FILE_REUSE_LOCK_TIMEOUT_S": "0.1"}
        ):
            with self.assertRaises(RuntimeError):
                self.allocate()


if __name__ == "__main__":
    unittest.main()
