"""Host-side storage for the offloaded Qwen4-Exp PLE n-gram table.

``--ple-offload-embedding`` keeps the PLE table (47.7 GiB in fp8 for
Qwen3.8-Flash-Next) out of device memory and lets the Triton gather kernel read
rows straight from a host pointer. Two backends provide that pointer:

``pinned`` (default)
    ``torch.empty(..., pin_memory=True)``. On a discrete GPU this frees VRAM.

``file``
    A file-backed, shared ``mmap`` of a sparse file under
    ``--ple-offload-dir``. Meant for unified-memory parts (GB10 / DGX Spark and
    similar), where pinned host memory comes out of the *same* pool as the
    model weights and ``pinned`` therefore frees nothing: Qwen3.8-Flash-Next is
    126.0 GiB of weights on a 121.63 GiB box and does not boot with ``pinned``.
    The kernel dereferences the pageable pointer directly, which only works on
    devices that report ``cudaDevAttrPageableMemoryAccessUsesHostPageTables``;
    rows are paged in from storage on demand, the file is sparse, deterministic
    in name and reused across restarts, and gathers of prefill size hint the
    page cache (``posix_fadvise(WILLNEED)``) so page faults are served
    concurrently instead of one at a time. A background trimmer keeps the
    mapping's resident set under a budget, because faulting rows in maps whole
    page-cache folios and the table would otherwise creep towards full
    residency (see ``PleFileRssTrimmer``).

This module has no Triton or CUDA-kernel imports so that its allocator and
prefetcher can be unit-tested on CPU.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import logging
import os
import re
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_LIBC: Optional[ctypes.CDLL] = None
_SMAPS_HEADER = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) ")
_SMAPS_RSS = re.compile(r"^Rss:\s+(\d+) kB")

PLE_OFFLOAD_BACKENDS = ("pinned", "file")

# cudaDeviceAttr enum values (cuda_runtime_api.h).
_CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100
_MADV_RANDOM = 1
_MADV_DONTNEED = 4
_PAGE_SHIFT = 12
# One MADV_DONTNEED call takes mmap_lock for its whole range; over the full
# 47.7 GiB table that is ~3.5 s during which every fault in the process --
# including the ones the gather kernel takes -- stalls. Trim in slices.
PLE_FILE_RSS_TRIM_CHUNK_BYTES = 1 << 30
# Below this many rows a gather is decode-sized (16 rows per token): the page
# faults are cheap and the host-side hint would cost more than it saves.
PLE_FILE_PREFETCH_MIN_ROWS = 2048

# Validated reuse of the file-backed table (opt-in via
# SGLANG_QWEN4_PLE_FILE_REUSE). The manifest records the table metadata and the
# source shard metadata the table was built from; a boot that finds them
# unchanged skips rewriting the table and lets the weight iterator skip reading
# the PLE shard tensors. The manifest is only published after the table is
# fsynced, and it is removed before a rebuild, so a crash cannot leave a
# manifest describing a partially rewritten table.
PLE_REUSE_MANIFEST_VERSION = 1
PLE_REUSE_MANIFEST_SUFFIX = ".reuse.json"
PLE_REUSE_LOCK_SUFFIX = ".reuse.lock"
PLE_REUSE_SHARD_RE = re.compile(
    r"^(?P<prefix>.+)\.ngram_embedding\.shard_(?P<index>\d+)\.weight$"
)

_REUSE_SKIP_PREFIXES: set = set()
_REUSE_PENDING_TABLES: list = []
_REUSE_LOCK_HANDLES: dict = {}
_REUSE_HOOK_INSTALLED = False


def reuse_skippable_weight(name: str) -> bool:
    """True when a validated table already holds the rows for this checkpoint key."""
    if not _REUSE_SKIP_PREFIXES:
        return False
    match = PLE_REUSE_SHARD_RE.match(name)
    return bool(match) and match.group("prefix") in _REUSE_SKIP_PREFIXES


def _register_reuse_skip(prefix: str) -> None:
    global _REUSE_HOOK_INSTALLED
    _REUSE_SKIP_PREFIXES.add(prefix)
    if _REUSE_HOOK_INSTALLED:
        return
    from sglang.srt.model_loader import weight_utils

    hooks = weight_utils.WEIGHT_KEY_SKIP_HOOKS
    if reuse_skippable_weight not in hooks:
        hooks.append(reuse_skippable_weight)
    _REUSE_HOOK_INSTALLED = True


def _read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"not a safetensors file: {path}")
        (header_len,) = struct.unpack("<Q", raw)
        header = json.loads(handle.read(header_len))
    header.pop("__metadata__", None)
    return header


def _stat_source(path: str) -> dict:
    stat = os.stat(path)
    return {
        "file": os.path.basename(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "inode": int(stat.st_ino),
    }


def collect_ple_source_manifest(model_dir: Optional[str]) -> Optional[dict]:
    """Source-shard metadata for the file-backed PLE table, or None.

    Reads the weight index and the safetensors headers only; tensor data is
    never touched. Returns None when the checkpoint has no PLE n-gram shards, in
    which case validated reuse is unavailable and the normal path is used.
    """
    if not model_dir:
        return None
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return None
    try:
        with open(index_path, "r") as handle:
            weight_map = json.load(handle)["weight_map"]
    except (OSError, KeyError, ValueError):
        return None

    by_file: dict = {}
    for name, filename in weight_map.items():
        if PLE_REUSE_SHARD_RE.match(name):
            by_file.setdefault(filename, []).append(name)
    if not by_file:
        return None

    sources = []
    for filename in sorted(by_file):
        source_path = os.path.join(model_dir, filename)
        try:
            entry = _stat_source(source_path)
            header = _read_safetensors_header(source_path)
        except (OSError, ValueError):
            return None
        keys = []
        for name in sorted(by_file[filename]):
            meta = header.get(name)
            if not isinstance(meta, dict):
                return None
            keys.append(
                {
                    "name": name,
                    "dtype": meta.get("dtype"),
                    "shape": [int(d) for d in meta.get("shape", [])],
                }
            )
        entry["keys"] = keys
        sources.append(entry)
    return {"sources": sources}


PLE_REUSE_SAMPLE_ROWS = 4096


def _sample_row_index(rows: int) -> torch.Tensor:
    """Evenly spaced row ids, first and last included, in exact int64.

    Built on the CPU explicitly (model init runs under a CUDA default-device
    context) and with integer arithmetic: a float32 linspace cannot represent
    row ids of a 320M-row table and would round the last one out of range.
    """
    count = min(PLE_REUSE_SAMPLE_ROWS, rows)
    if count <= 1:
        return torch.zeros(max(count, 0), dtype=torch.int64, device="cpu")
    steps = torch.arange(count, dtype=torch.int64, device="cpu")
    return torch.unique(steps * (rows - 1) // (count - 1))


def _table_sample_digest(table: torch.Tensor) -> str:
    """Digest of evenly spaced table rows (first and last included).

    The manifest otherwise only describes shapes and source stats, so a table
    changed after publication would still validate.  Sampling keeps the check
    to a few MB of reads on a multi-GB file-backed table.
    """
    import hashlib

    rows = int(table.shape[0])
    index = _sample_row_index(rows)
    data = table.view(torch.uint8).index_select(0, index)
    digest = hashlib.sha256()
    digest.update(f"{rows}:{int(index.numel())}:".encode())
    digest.update(data.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _manifest_path(table_path: str) -> str:
    return table_path + PLE_REUSE_MANIFEST_SUFFIX


def _load_manifest(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _write_manifest_atomic(path: str, manifest: dict) -> None:
    tmp_path = f"{path}.tmp{os.getpid()}"
    with open(tmp_path, "w") as handle:
        json.dump(manifest, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    dir_fd = None
    try:
        dir_fd = os.open(os.path.dirname(path), os.O_RDONLY)
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        if dir_fd is not None:
            os.close(dir_fd)


def _manifest_reason(
    manifest: Optional[dict], expected_table: dict, expected_sources: dict
) -> str:
    if manifest is None:
        return "no reuse manifest"
    if manifest.get("version") != PLE_REUSE_MANIFEST_VERSION:
        return "manifest version mismatch"
    if manifest.get("table") != expected_table:
        return "table metadata mismatch"
    if manifest.get("sources") != expected_sources["sources"]:
        return "source shard metadata mismatch"
    return ""


class _PleTableLock:
    """Advisory lock serializing builders and readers of one table file.

    A reader holds a shared lock for its lifetime; a builder takes an exclusive
    lock while it rewrites the table. The lock file is separate from the table,
    so it survives a crash without touching table content.
    """

    def __init__(self, path: str, timeout_s: float) -> None:
        self._path = path
        self._timeout_s = max(float(timeout_s), 0.0)
        self._fd: Optional[int] = None

    def acquire(self, *, exclusive: bool) -> bool:
        import fcntl
        import time

        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + self._timeout_s
        while True:
            try:
                fcntl.flock(fd, mode | fcntl.LOCK_NB)
                self._fd = fd
                return True
            except OSError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    return False
                time.sleep(0.2)

    def downgrade_to_shared(self) -> None:
        if self._fd is None:
            return
        import fcntl

        try:
            fcntl.flock(self._fd, fcntl.LOCK_SH)
        except OSError:
            pass

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            os.close(self._fd)
        finally:
            self._fd = None


def _store_lock(path: str, lock: _PleTableLock) -> None:
    previous = _REUSE_LOCK_HANDLES.pop(path, None)
    if previous is not None:
        previous.release()
    _REUSE_LOCK_HANDLES[path] = lock


def _prepare_file_reuse(
    table: torch.Tensor,
    path: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    tag: Optional[str],
    source_dir: Optional[str],
) -> None:
    nbytes = int(table.numel() * table.element_size())
    expected_table = {
        "size": nbytes,
        "shape": [int(d) for d in shape],
        "dtype": str(dtype).replace("torch.", ""),
        "tag": tag or "",
    }
    expected_sources = collect_ple_source_manifest(source_dir)
    if expected_sources is None:
        logger.warning(
            "PLE table: reuse requested but no PLE shard index under %r; "
            "using the normal rewrite path",
            source_dir,
        )
        return

    timeout_s = float(envs.SGLANG_QWEN4_PLE_FILE_REUSE_LOCK_TIMEOUT_S.get())
    lock = _PleTableLock(path + PLE_REUSE_LOCK_SUFFIX, timeout_s)

    forced = bool(envs.SGLANG_QWEN4_PLE_FILE_REUSE_FORCE_REBUILD.get())
    manifest = None
    if forced:
        reason = "forced rebuild"
    else:
        manifest = _load_manifest(_manifest_path(path))
        reason = _manifest_reason(manifest, expected_table, expected_sources)

    if not reason:
        if not lock.acquire(exclusive=False):
            raise RuntimeError(
                "PLE table reuse lock is held by another process; concurrent "
                f"users of {path!r} are not supported"
            )
        digest = _table_sample_digest(table)
        stored = manifest.get("sample_digest")
        if stored is not None and stored != digest:
            reason = "table content digest mismatch"
            lock.release()
            lock = _PleTableLock(path + PLE_REUSE_LOCK_SUFFIX, timeout_s)
        elif stored is None:
            # Manifests published before the digest existed: record it now
            # instead of forcing a full rewrite of a table that validated.
            manifest["sample_digest"] = digest
            try:
                _write_manifest_atomic(_manifest_path(path), manifest)
                logger.info("PLE table: sample digest added to the reuse manifest")
            except OSError as exc:
                logger.warning("PLE table: could not add the sample digest (%s)", exc)

    if not reason:
        _store_lock(path, lock)
        table._sglang_ple_reuse_valid = True
        for source in expected_sources["sources"]:
            for key in source["keys"]:
                prefix = key["name"][: key["name"].rindex(".ngram_embedding.shard_")]
                _register_reuse_skip(prefix)
        logger.info(
            "PLE table: reuse validated, skipping the checkpoint rewrite (%s)",
            path,
        )
        return

    if not lock.acquire(exclusive=True):
        raise RuntimeError(
            "PLE table rebuild lock is held by another process; stop the other "
            f"server using this directory before restarting: {path}"
        )
    _store_lock(path, lock)
    manifest_path = _manifest_path(path)
    if os.path.exists(manifest_path):
        # Invalidate first: a crash while rewriting must not leave a manifest
        # that still matches the source shards.
        os.unlink(manifest_path)
    table._sglang_ple_pending_manifest = {
        "version": PLE_REUSE_MANIFEST_VERSION,
        "table": expected_table,
        "sources": expected_sources["sources"],
    }
    _REUSE_PENDING_TABLES.append(table)
    logger.info(
        "PLE table: reuse unavailable (%s); rebuilding %s and publishing a manifest",
        reason,
        path,
    )


def finalize_ple_table_reuse() -> None:
    """Publish reuse manifests for tables rewritten in this process.

    Called after the weight loader has copied every PLE shard. The table is
    fsynced before the manifest becomes visible, and the exclusive lock is
    downgraded to shared so the serving process still keeps out builders.
    """
    for table in list(_REUSE_PENDING_TABLES):
        path = getattr(table, "_sglang_ple_file_path", None)
        manifest = getattr(table, "_sglang_ple_pending_manifest", None)
        if path is None or manifest is None:
            continue
        try:
            fd = os.open(path, os.O_RDWR)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as exc:
            logger.warning("PLE table: fsync before manifest publish failed (%s)", exc)
        manifest = dict(manifest, sample_digest=_table_sample_digest(table))
        try:
            _write_manifest_atomic(_manifest_path(path), manifest)
        except OSError as exc:
            logger.warning("PLE table: manifest publish failed (%s)", exc)
            continue
        lock = _REUSE_LOCK_HANDLES.get(path)
        if lock is not None:
            lock.downgrade_to_shared()
        logger.info("PLE table: reuse manifest published for %s", path)
    _REUSE_PENDING_TABLES.clear()


def _reset_reuse_state_for_tests() -> None:
    for lock in list(_REUSE_LOCK_HANDLES.values()):
        lock.release()
    _REUSE_LOCK_HANDLES.clear()
    _REUSE_PENDING_TABLES.clear()
    _REUSE_SKIP_PREFIXES.clear()



class PleFilePrefetcher:
    """Hint the page cache about the rows a prefill-sized gather is about to read.

    With the table on storage, a cold prefill chunk faults tens of thousands of
    4 KiB pages one at a time from inside the gather kernel. Advising them
    first (``posix_fadvise(WILLNEED)`` per distinct page, on one background
    thread) lets the block layer serve them concurrently. Measured on a GB10 /
    NVMe: cold prefill 650-750 tok/s -> 1,000-2,100 tok/s (warm: ~2,200-2,600).
    Decode-sized gathers are skipped; nothing runs during CUDA-graph capture.
    """

    def __init__(
        self,
        path: str,
        row_bytes: int,
        min_rows: int = PLE_FILE_PREFETCH_MIN_ROWS,
    ) -> None:
        self._fd = os.open(path, os.O_RDONLY)
        self._row_bytes = int(row_bytes)
        self._min_rows = int(min_rows)
        self._pool = ThreadPoolExecutor(max_workers=1)

    @staticmethod
    def pages_for_rows(row_ids: torch.Tensor, row_bytes: int) -> list[int]:
        start = row_ids.to(torch.int64) * row_bytes
        end = start + (row_bytes - 1)
        return (
            torch.cat([start >> _PAGE_SHIFT, end >> _PAGE_SHIFT])
            .unique(sorted=True)
            .tolist()
        )

    def _advise(self, pages: list[int]) -> None:
        for p in pages:
            try:
                os.posix_fadvise(
                    self._fd, p << _PAGE_SHIFT, 1 << _PAGE_SHIFT, os.POSIX_FADV_WILLNEED
                )
            except OSError:
                return

    def enqueue(
        self,
        flat_ids: torch.Tensor,
        *,
        vocab_start: int = 0,
        vocab_end: Optional[int] = None,
    ) -> bool:
        """Queue the hint for ``flat_ids``. Returns whether anything was queued."""
        if flat_ids.numel() < self._min_rows:
            return False
        if flat_ids.is_cuda and torch.cuda.is_current_stream_capturing():
            return False
        # The .cpu() syncs the stream; acceptable for prefill chunks (~1 s) and
        # it is what lets the page set be computed without touching the kernel.
        row_ids = flat_ids.detach().cpu()
        if vocab_end is not None:
            # The file contains only this rank's vocabulary shard.
            row_ids = row_ids[(row_ids >= vocab_start) & (row_ids < vocab_end)]
        row_ids = row_ids - vocab_start
        if row_ids.numel() == 0:
            return False
        pages = self.pages_for_rows(row_ids, self._row_bytes)
        self._pool.submit(self._advise, pages)
        return True

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        try:
            os.close(self._fd)
        except OSError:
            pass


class PleFileRssTrimmer:
    """Keep the mapped table's resident set under a budget.

    Every random row fault maps in a whole page-cache folio, so with large
    folios (Linux 6.x) the mapping's Rss climbs towards the table's full size
    while a generated token only reads a few KB of it: measured ~45 KB of Rss
    growth per token on a GB10. On a unified-memory part that is not a slow
    leak, it is a countdown -- the free-memory readings that size the KV pool
    come from the same pool the folios are accumulating in.

    ``MADV_RANDOM`` does not prevent it (it limits readahead I/O, not the
    mapping-in of folios already in cache) and ``posix_fadvise(DONTNEED)`` does
    not release them either. ``MADV_DONTNEED`` over the mapping does: the page
    table entries go, the pages stay in the page cache, and hot rows come back
    at minor-fault cost.

    Dropping entries under a running gather is the state this backend already
    handles: the file starts out entirely unfaulted and every cold row is
    faulted in from inside the kernel through the same host page tables. What
    must not happen is one ``madvise`` call over the whole table, so the trim
    is chunked (see ``PLE_FILE_RSS_TRIM_CHUNK_BYTES``) and runs on its own
    daemon thread -- decode replays a CUDA graph and executes no Python, so a
    hook in the gather would never fire in the phase that grows the table.
    """

    def __init__(
        self,
        addr: int,
        nbytes: int,
        budget_bytes: int,
        interval_s: float,
        chunk_bytes: int = PLE_FILE_RSS_TRIM_CHUNK_BYTES,
    ) -> None:
        self._addr = int(addr)
        self._nbytes = int(nbytes)
        self._budget = int(budget_bytes)
        self._interval = float(interval_s)
        self._chunk = int(chunk_bytes)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="ple-file-rss-trim", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def mapping_rss_bytes(self) -> Optional[int]:
        """Resident bytes of the VMAs backing the table, or None off Linux."""
        return _mapping_rss_bytes(self._addr, self._nbytes)

    def trim_once(self) -> int:
        """Drop the mapping's resident pages if over budget. Returns bytes freed."""
        before = self.mapping_rss_bytes()
        if before is None or before <= self._budget:
            return 0
        for offset in range(0, self._nbytes, self._chunk):
            if self._stop.is_set():
                break
            length = min(self._chunk, self._nbytes - offset)
            if not _madvise(self._addr + offset, length, _MADV_DONTNEED):
                return 0
            # Let the faults that queued behind mmap_lock through.
            self._stop.wait(0.005)
        after = self.mapping_rss_bytes()
        freed = before - after if after is not None else 0
        logger.info(
            "PLE table: trimmed resident set %.1f -> %.1f GiB (budget %.1f GiB)",
            before / 2**30,
            (after if after is not None else 0) / 2**30,
            self._budget / 2**30,
        )
        return max(freed, 0)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.trim_once()
            except Exception as exc:  # advisory only; never fail a request
                logger.warning("PLE table: resident-set trim skipped (%s)", exc)

    def close(self) -> None:
        self._stop.set()


def allocate_ple_host_table(
    shape: Sequence[int],
    dtype: torch.dtype,
    backend: str = "pinned",
    table_dir: Optional[str] = None,
    tag: Optional[str] = None,
    source_dir: Optional[str] = None,
) -> torch.Tensor:
    """Return a host tensor of ``shape``/``dtype`` for the PLE table.

    For the file backend, ``table_dir`` should be private to one checkpoint
    (the server defaults it to ``$SGLANG_CACHE_DIR/ple/<model path>``): the
    file name only encodes shape, dtype and ``tag``, and every boot rewrites
    the whole table through the weight loader. With
    ``SGLANG_QWEN4_PLE_FILE_REUSE`` the rewrite is skipped when a previously
    published manifest still matches ``source_dir``'s PLE shard metadata; a
    manifest is then (re)published by ``finalize_ple_table_reuse`` after the
    loader finishes.
    """
    if backend not in PLE_OFFLOAD_BACKENDS:
        raise ValueError(
            f"unknown PLE offload backend {backend!r}; choose from {PLE_OFFLOAD_BACKENDS}"
        )
    if backend == "pinned":
        return torch.empty(tuple(shape), dtype=dtype, device="cpu", pin_memory=True)

    numel = 1
    for d in shape:
        numel *= int(d)
    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    table_dir = os.path.expanduser(table_dir or envs.SGLANG_QWEN4_PLE_FILE_DIR.get())
    os.makedirs(table_dir, exist_ok=True)
    path = os.path.join(table_dir, ple_table_file_name(shape, dtype, tag))
    if not os.path.exists(path) or os.path.getsize(path) != nbytes:
        # Sparse: only pages that get written take disk space.
        with open(path, "wb") as f:
            f.truncate(nbytes)
    logger.info(
        "PLE table: file-backed mmap %s (%.1f GiB, %s)", path, nbytes / 2**30, dtype
    )
    storage = torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)
    _madvise_random(storage, nbytes)
    table = storage.view(dtype).view(*[int(d) for d in shape])
    table._sglang_ple_file_path = path  # consumed by PleFilePrefetcher
    table._sglang_ple_reuse_valid = False
    if envs.SGLANG_QWEN4_PLE_FILE_REUSE.get():
        _prepare_file_reuse(table, path, shape, dtype, tag, source_dir)
    return table


def make_ple_file_prefetcher(table: torch.Tensor) -> Optional[PleFilePrefetcher]:
    """A prefetcher for a table returned by ``allocate_ple_host_table(..., "file")``."""
    path = getattr(table, "_sglang_ple_file_path", None)
    if path is None or not envs.SGLANG_QWEN4_PLE_FILE_PREFETCH.get():
        return None
    row_bytes = (
        int(table.shape[-1]) * table.element_size()
        if table.dim() >= 2
        else table.element_size()
    )
    prefetcher = PleFilePrefetcher(path=path, row_bytes=row_bytes)
    logger.info(
        "PLE table: WILLNEED prefetch on for gathers of >= %d rows (row = %d B)",
        PLE_FILE_PREFETCH_MIN_ROWS,
        row_bytes,
    )
    return prefetcher


def make_ple_file_rss_trimmer(table: torch.Tensor) -> Optional[PleFileRssTrimmer]:
    """A started trimmer for a table from ``allocate_ple_host_table(..., "file")``.

    ``SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=0`` turns it off; it is also absent
    where the resident set cannot be read (no ``/proc/self/smaps``).
    """
    path = getattr(table, "_sglang_ple_file_path", None)
    if path is None:
        return None
    budget_gb = float(envs.SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB.get())
    if budget_gb <= 0:
        return None
    nbytes = table.numel() * table.element_size()
    if _mapping_rss_bytes(table.data_ptr(), nbytes) is None:
        logger.warning(
            "PLE table: resident-set trim off, /proc/self/smaps is not readable; "
            "the mapping will creep towards %.1f GiB resident",
            nbytes / 2**30,
        )
        return None
    trimmer = PleFileRssTrimmer(
        addr=table.data_ptr(),
        nbytes=nbytes,
        budget_bytes=int(budget_gb * 2**30),
        interval_s=float(envs.SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S.get()),
    )
    trimmer.start()
    logger.info(
        "PLE table: resident set capped at %.1f GiB, checked every %.0f s",
        budget_gb,
        float(envs.SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S.get()),
    )
    return trimmer


def check_file_backend_supported(device_index: int = 0) -> None:
    """Fail fast at load time instead of silently reading garbage in the kernel."""
    if envs.SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK.get():
        logger.warning(
            "PLE table: file backend device check skipped by "
            "SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK"
        )
        return
    supported = device_uses_host_page_tables(device_index)
    if supported is None:
        raise RuntimeError(
            "--ple-offload-backend file: could not query "
            "cudaDevAttrPageableMemoryAccessUsesHostPageTables. Set "
            "SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1 only if you know the "
            "device reads pageable host memory through the host page tables."
        )
    if not supported:
        raise ValueError(
            "--ple-offload-backend file needs a device whose pageable host "
            "memory accesses go through the host page tables (unified-memory "
            "parts such as GB10). This device reports it does not; use "
            "--ple-offload-backend pinned."
        )


def default_ple_table_dir(model_path: str) -> str:
    """``$SGLANG_QWEN4_PLE_FILE_DIR/<model path>``, one directory per checkpoint."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_path).rstrip("/")).strip("_")
    return os.path.join(envs.SGLANG_QWEN4_PLE_FILE_DIR.get(), safe or "model")


def ple_table_file_name(
    shape: Sequence[int], dtype: torch.dtype, tag: Optional[str] = None
) -> str:
    """Deterministic file name so the sparse table is reused across restarts.

    ``tag`` distinguishes tables of the same shape that must not share a file,
    e.g. the vocabulary shards of different tensor-parallel ranks.
    """
    numel = 1
    for d in shape:
        numel *= int(d)
    elem = torch.empty(0, dtype=dtype).element_size()
    dims = "x".join(str(int(d)) for d in shape)
    suffix = f"_{tag}" if tag else ""
    return f"ple_table_{dims}_{str(dtype).replace('torch.', '')}_{numel * elem}B{suffix}.bin"


def device_uses_host_page_tables(device_index: int = 0) -> Optional[bool]:
    """Whether pageable host memory is directly addressable by the GPU.

    Returns None when the CUDA runtime library cannot be queried.
    """
    candidates = [ctypes.util.find_library("cudart")]
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(torch_lib):
        candidates += sorted(
            os.path.join(torch_lib, f)
            for f in os.listdir(torch_lib)
            if f.startswith("libcudart.so")
        )
    try:
        import nvidia.cuda_runtime  # type: ignore

        nv_lib = os.path.join(os.path.dirname(nvidia.cuda_runtime.__file__), "lib")
        if os.path.isdir(nv_lib):
            candidates += sorted(
                os.path.join(nv_lib, f)
                for f in os.listdir(nv_lib)
                if f.startswith("libcudart.so")
            )
    except Exception:
        pass
    for name in [c for c in candidates if c]:
        try:
            cudart = ctypes.CDLL(name)
            value = ctypes.c_int()
            rc = cudart.cudaDeviceGetAttribute(
                ctypes.byref(value),
                ctypes.c_int(
                    _CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
                ),
                ctypes.c_int(device_index),
            )
            if rc == 0:
                return bool(value.value)
        except OSError:
            continue
    return None


def _madvise_random(storage: torch.Tensor, nbytes: int) -> None:
    """The table is pure random access (16 rows of 160 B per token). Without
    this the kernel's readahead pulls its whole window: measured 1.4 MB of disk
    per token, ~560x the bytes actually used.

    It bounds readahead I/O only. Folios that are already in the page cache are
    still mapped in whole on a fault, which is what ``PleFileRssTrimmer``
    exists for."""
    if not _madvise(storage.data_ptr(), nbytes, _MADV_RANDOM):
        logger.warning("PLE table: madvise(MADV_RANDOM) not applied")


def _libc() -> Optional[ctypes.CDLL]:
    global _LIBC
    if _LIBC is None:
        try:
            _LIBC = ctypes.CDLL(
                ctypes.util.find_library("c") or "libc.so.6", use_errno=True
            )
        except OSError:
            return None
    return _LIBC


def _madvise(addr: int, length: int, advice: int) -> bool:
    """``madvise(2)`` on our own mapping. Advisory: never affects correctness."""
    libc = _libc()
    if libc is None:
        return False
    try:
        rc = libc.madvise(
            ctypes.c_void_p(addr), ctypes.c_size_t(length), ctypes.c_int(advice)
        )
    except Exception:
        return False
    if rc != 0:
        logger.warning(
            "PLE table: madvise(advice=%d) failed (errno %d)",
            advice,
            ctypes.get_errno(),
        )
        return False
    return True


def _mapping_rss_bytes(
    addr: int, nbytes: int, smaps_path: str = "/proc/self/smaps"
) -> Optional[int]:
    """Resident bytes of the VMAs overlapping ``[addr, addr + nbytes)``.

    Summed per mapping rather than taken from ``statm``/``smaps_rollup``: only
    the table's own residency should drive the trim, and on a unified-memory
    box the process RSS is dominated by everything else.
    """
    lo, hi = int(addr), int(addr) + int(nbytes)
    total = 0
    overlapping = False
    try:
        with open(smaps_path, "r") as f:
            for line in f:
                header = _SMAPS_HEADER.match(line)
                if header is not None:
                    start = int(header.group(1), 16)
                    end = int(header.group(2), 16)
                    overlapping = start < hi and end > lo
                elif overlapping:
                    rss = _SMAPS_RSS.match(line)
                    if rss is not None:
                        total += int(rss.group(1)) * 1024
    except OSError:
        return None
    return total
