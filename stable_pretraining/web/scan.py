"""Background scanner for the web viewer.

Discovers runs by walking the root directory for ``sidecar.json`` files
and tracks per-run mtime/size of ``sidecar.json`` and ``metrics.csv``.
A polling thread re-scans every ``poll_interval`` seconds and pushes
deltas to subscribed SSE queues.

NFS-safe by design: no inotify, only ``stat`` polling.
"""

from __future__ import annotations

import csv
import json
import os
import queue
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class _Run:
    run_id: str
    run_dir: Path
    sidecar_mtime: float
    metrics_mtime: float
    metrics_size: int
    media_mtime: float = 0.0
    media_size: int = 0
    sidecar: dict = field(default_factory=dict)


class RunScanner:
    """Polls a directory tree for sidecar+metrics changes and fans out events."""

    def __init__(self, root: Path, poll_interval: float = 1.0) -> None:
        self.root = Path(root).expanduser().resolve()
        self.poll_interval = poll_interval
        self._runs: dict[str, _Run] = {}
        self._lock = threading.Lock()
        self._subs: set[queue.Queue] = set()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Scan progress (used to render a loading bar in the UI while the
        # initial NFS walk is in progress; thousands of sidecars over NFS can
        # take tens of seconds).
        # phase: "starting" | "discovering" | "loading" | "idle"
        self._progress = {
            "phase": "starting",
            "total": 0,
            "done": 0,
            "initial_done": False,
        }
        # Per-run cache of metrics, in two forms:
        #   - parsed_dict: structured (used by Python callers)
        #   - json_bytes: pre-serialised, NaN-sanitised JSON (sent verbatim
        #                 by the HTTP layer to skip re-serialisation cost)
        # Invalidated when the metrics.csv mtime or size changes.
        # run_id -> (mtime, size, parsed_dict, json_bytes_or_None)
        self._metrics_cache: dict[str, tuple[float, int, dict, Optional[bytes]]] = {}

    # ---- lifecycle ----

    def start(self) -> None:
        """Start the background scanner thread.

        Returns immediately. The first scan runs in the background so that
        callers (e.g. the HTTP server) become responsive before a potentially
        slow NFS walk over thousands of sidecars completes. Clients learn
        about discovered runs via the SSE /api/stream channel.
        """
        self._thread = threading.Thread(
            target=self._initial_then_loop, daemon=True, name="spt-web-scanner"
        )
        self._thread.start()

    def _initial_then_loop(self) -> None:
        """Run the first scan, then enter the steady-state poll loop."""
        try:
            changed, removed = self._scan(initial=True)
            if changed or removed:
                self._publish("update", {"changed": changed, "removed": removed})
        except Exception:
            pass
        with self._lock:
            self._progress["phase"] = "idle"
            self._progress["initial_done"] = True
        self._publish("progress", self.progress_json())
        self._loop()

    # ---- progress --------------------------------------------------------

    def progress_json(self) -> dict:
        """Snapshot of the current scan progress (used by the UI loading bar)."""
        with self._lock:
            return dict(self._progress)

    def _set_progress(self, **kwargs) -> None:
        with self._lock:
            self._progress.update(kwargs)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ---- pub/sub ----

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=128)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def _publish(self, event_type: str, data: Any) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait({"type": event_type, "data": data})
            except queue.Full:
                pass

    # ---- scanning ----

    @staticmethod
    def _list_dir(d: Path) -> tuple[list[Path], list[Path]]:
        """One ``readdir`` call. Returns (subdirs, sidecar_paths_in_dir)."""
        subdirs: list[Path] = []
        sidecars: list[Path] = []
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            subdirs.append(Path(entry.path))
                        elif entry.name == "sidecar.json":
                            sidecars.append(Path(entry.path))
                    except OSError:
                        continue
        except OSError:
            pass
        return subdirs, sidecars

    def _parallel_walk(
        self, root: Path, report_progress: bool = False, max_workers: int = 32
    ) -> list[Path]:
        """Find every ``sidecar.json`` under ``root`` with a parallel BFS.

        On NFS each ``readdir`` is a network round-trip, so a single-threaded
        walk is round-trip-bound. By running many ``readdir`` calls
        concurrently through a thread pool we overlap the latency and
        approach link-bandwidth limits.

        On a 2 200-run tree this drops discovery from ~3.7 s to <1 s.

        ``report_progress=True`` publishes a ``progress`` SSE event at most
        ~5 Hz so the UI loading bar can show a live counter.
        """
        results: list[Path] = []
        last_pub = time.monotonic()
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="spt-walk"
        ) as exe:
            in_flight = {exe.submit(self._list_dir, root)}
            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    try:
                        subdirs, sidecars = fut.result()
                    except Exception:
                        continue
                    results.extend(sidecars)
                    for sd in subdirs:
                        in_flight.add(exe.submit(self._list_dir, sd))
                if report_progress:
                    now = time.monotonic()
                    if now - last_pub >= 0.2:
                        self._set_progress(
                            phase="discovering", total=len(results), done=0
                        )
                        self._publish("progress", self.progress_json())
                        last_pub = now
        return results

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_interval):
            try:
                changed, removed = self._scan()
                if changed or removed:
                    self._publish("update", {"changed": changed, "removed": removed})
            except Exception:
                # Scan errors are recoverable; the next tick will retry.
                pass

    def _scan(self, initial: bool = False) -> tuple[list[str], list[str]]:
        """Walk for sidecar.json files and refresh tracked runs.

        On NFS-mounted run trees the bottleneck is per-file ``stat`` /
        ``read`` syscalls (each is a network round-trip), not CPU. The
        per-sidecar work is therefore dispatched to a small thread pool;
        this empirically takes a 2 000-run cold scan from ~16 s to ~3 s.

        ``initial=True`` enables progress reporting via the ``progress`` SSE
        event so the UI can render a loading bar during a slow first walk.
        """
        if initial:
            self._set_progress(phase="discovering", total=0, done=0)
            self._publish("progress", self.progress_json())
            sidecar_paths = self._parallel_walk(self.root, report_progress=True)
            self._set_progress(phase="loading", total=len(sidecar_paths), done=0)
            self._publish("progress", self.progress_json())
        else:
            sidecar_paths = self._parallel_walk(self.root, report_progress=False)
        seen: set[str] = set()
        changed: list[str] = []

        # Thread fan-out for the per-file work. cap at 16 to avoid drowning
        # the NFS server with concurrent metadata requests.
        max_workers = min(16, max(1, (os.cpu_count() or 4) * 2))

        def _process_one(sidecar_path: Path):
            try:
                st = sidecar_path.stat()
            except OSError:
                return None
            run_dir = sidecar_path.parent
            run_id = self._run_id_for(run_dir)

            metrics_path = run_dir / "metrics.csv"
            try:
                mst = metrics_path.stat()
                m_mtime, m_size = mst.st_mtime, mst.st_size
            except OSError:
                m_mtime, m_size = 0.0, 0

            media_path = run_dir / "media.jsonl"
            try:
                medst = media_path.stat()
                med_mtime, med_size = medst.st_mtime, medst.st_size
            except OSError:
                med_mtime, med_size = 0.0, 0

            with self._lock:
                existing = self._runs.get(run_id)

            need_update = (
                existing is None
                or existing.sidecar_mtime != st.st_mtime
                or existing.metrics_mtime != m_mtime
                or existing.metrics_size != m_size
                or existing.media_mtime != med_mtime
                or existing.media_size != med_size
            )
            sidecar_data: Optional[dict] = None
            if need_update:
                try:
                    sidecar_data = json.loads(sidecar_path.read_text())
                except (json.JSONDecodeError, OSError):
                    return run_id  # mark as seen so it isn't pruned, but no update
            return (
                run_id,
                run_dir,
                st.st_mtime,
                m_mtime,
                m_size,
                med_mtime,
                med_size,
                sidecar_data,
                need_update,
            )

        # Stream results back as workers finish so the UI sees progress.
        # ``executor.map`` returns in submit order, which is fine here — each
        # NFS round-trip is roughly the same cost so we don't lose much by
        # not using ``as_completed``, and ordered results play well with the
        # rest of this function.
        results: list = []
        if initial and sidecar_paths:
            # Tunable: report every ~2% or 50 files, whichever is bigger.
            tick = max(50, len(sidecar_paths) // 50)
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                for i, r in enumerate(exe.map(_process_one, sidecar_paths), start=1):
                    results.append(r)
                    if i % tick == 0 or i == len(sidecar_paths):
                        self._set_progress(done=i)
                        self._publish("progress", self.progress_json())
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                results = list(exe.map(_process_one, sidecar_paths))

        for r in results:
            if r is None:
                continue
            if isinstance(r, str):  # seen-but-unparseable sidecar
                seen.add(r)
                continue
            (
                run_id,
                run_dir,
                s_mtime,
                m_mtime,
                m_size,
                med_mtime,
                med_size,
                sidecar_data,
                need_update,
            ) = r
            seen.add(run_id)
            if need_update and sidecar_data is not None:
                with self._lock:
                    self._runs[run_id] = _Run(
                        run_id=run_id,
                        run_dir=run_dir,
                        sidecar_mtime=s_mtime,
                        metrics_mtime=m_mtime,
                        metrics_size=m_size,
                        media_mtime=med_mtime,
                        media_size=med_size,
                        sidecar=sidecar_data,
                    )
                changed.append(run_id)

        with self._lock:
            removed = [rid for rid in self._runs if rid not in seen]
            for rid in removed:
                del self._runs[rid]
                self._metrics_cache.pop(rid, None)

        return changed, removed

    def _run_id_for(self, run_dir: Path) -> str:
        try:
            rel = run_dir.relative_to(self.root)
        except ValueError:
            return run_dir.name
        s = str(rel)
        return run_dir.name if s in (".", "") else s

    # ---- queries ----

    def runs_json(self) -> list[dict]:
        with self._lock:
            runs = list(self._runs.values())
        return [self._serialize(r) for r in runs]

    @staticmethod
    def _serialize(run: _Run) -> dict:
        s = run.sidecar
        return {
            "run_id": run.run_id,
            "run_dir": str(run.run_dir),
            "status": s.get("status"),
            "created_at": s.get("created_at"),
            "tags": s.get("tags") or [],
            "notes": s.get("notes") or "",
            "hparams": s.get("hparams") or {},
            "summary": s.get("summary") or {},
            "checkpoint_path": s.get("checkpoint_path"),
            "metrics_size": run.metrics_size,
            "has_media": run.media_size > 0,
        }

    def metrics_json(self, run_id: str) -> Optional[dict]:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            return None
        mpath = run.run_dir / "metrics.csv"
        if not mpath.is_file():
            return {"metrics": {}}

        # Cache key = (mtime, size). If the file hasn't changed we serve the
        # parsed result directly, which is orders of magnitude faster than
        # re-reading and re-parsing a multi-MiB CSV from NFS.
        try:
            st = mpath.stat()
            cache_key = (st.st_mtime, st.st_size)
        except OSError:
            cache_key = None
        if cache_key is not None:
            cached = self._metrics_cache.get(run_id)
            if cached is not None and (cached[0], cached[1]) == cache_key:
                return cached[2]  # parsed dict (HTTP layer prefers metrics_json_bytes)

        with mpath.open("r", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return {"metrics": {}}

            step_idx = header.index("step") if "step" in header else None
            epoch_idx = header.index("epoch") if "epoch" in header else None
            metric_cols = [
                (i, name)
                for i, name in enumerate(header)
                if name and name not in ("step", "epoch")
            ]

            metrics: dict[str, dict[str, list]] = {
                name: {"step": [], "epoch": [], "y": []} for _, name in metric_cols
            }

            for row_idx, row in enumerate(reader):
                step = _maybe_float(row, step_idx)
                epoch = _maybe_float(row, epoch_idx)
                # Fallback x if neither column populated.
                if step is None and epoch is None:
                    step = float(row_idx)

                for i, name in metric_cols:
                    if i >= len(row) or row[i] == "":
                        continue
                    try:
                        y = float(row[i])
                    except ValueError:
                        continue
                    m = metrics[name]
                    m["step"].append(step)
                    m["epoch"].append(epoch)
                    m["y"].append(y)

        result = {"metrics": {k: v for k, v in metrics.items() if v["y"]}}
        if cache_key is not None:
            self._metrics_cache[run_id] = (
                cache_key[0],
                cache_key[1],
                result,
                None,  # JSON bytes filled lazily by metrics_json_bytes
            )
        return result

    def metrics_stream(self, run_id: str):
        """Yield metrics for ``run_id`` in chunks as the CSV is parsed.

        Each yielded value is a ``dict`` with shape::

            {"chunk": int, "metrics": {<name>: {"step":[...], "epoch":[...], "y":[...]}}}

        and the final yielded value is ``{"done": true}``. If the metrics are
        already cached we emit them as a single chunk followed by ``done`` —
        callers don't need a separate fast-path. If the run is unknown we
        yield ``None`` (HTTP layer turns this into a 404).

        While streaming we accumulate the parsed structure in memory and
        publish it to ``self._metrics_cache`` on completion, so successive
        reads of the same run hit the warm cache.
        """
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            yield None
            return
        mpath = run.run_dir / "metrics.csv"
        if not mpath.is_file():
            yield {"chunk": 0, "metrics": {}}
            yield {"done": True}
            return

        # Warm-cache fast path — emit the full payload as one chunk.
        try:
            st = mpath.stat()
            cache_key = (st.st_mtime, st.st_size)
        except OSError:
            cache_key = None
        if cache_key is not None:
            cached = self._metrics_cache.get(run_id)
            if cached is not None and (cached[0], cached[1]) == cache_key:
                # Slice the cached payload into smaller chunks so the client
                # still gets a progressive paint even on a warm cache hit.
                # The server is fast here — slicing a few thousand floats is
                # cheap — and the visual feel ("filling in") is what the
                # user notices vs one big drop.
                CHUNK_POINTS = 5000
                full_metrics = cached[2]["metrics"]
                # Build per-metric slices in lockstep so each chunk carries
                # roughly CHUNK_POINTS total points across all metrics.
                names = list(full_metrics.keys())
                offsets = {n: 0 for n in names}
                lengths = {n: len(full_metrics[n]["y"]) for n in names}
                chunk_id = 0
                while any(offsets[n] < lengths[n] for n in names):
                    payload: dict[str, dict[str, list]] = {}
                    remaining = CHUNK_POINTS
                    for n in names:
                        if remaining <= 0:
                            break
                        a = offsets[n]
                        b = min(lengths[n], a + remaining)
                        if b <= a:
                            continue
                        payload[n] = {
                            "step": full_metrics[n]["step"][a:b],
                            "epoch": full_metrics[n]["epoch"][a:b],
                            "y": full_metrics[n]["y"][a:b],
                        }
                        offsets[n] = b
                        remaining -= b - a
                    if payload:
                        yield {"chunk": chunk_id, "metrics": payload}
                        chunk_id += 1
                yield {"done": True}
                return

        # Cold path: parse + emit incrementally so the browser can paint
        # while the file is still being read.
        with mpath.open("r", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                yield {"chunk": 0, "metrics": {}}
                yield {"done": True}
                return

            step_idx = header.index("step") if "step" in header else None
            epoch_idx = header.index("epoch") if "epoch" in header else None
            metric_cols = [
                (i, name)
                for i, name in enumerate(header)
                if name and name not in ("step", "epoch")
            ]

            # Full structure (kept for caching at end). The per-chunk dicts
            # only carry NEW points since the previous flush so the client
            # can append cheaply.
            full: dict[str, dict[str, list]] = {
                name: {"step": [], "epoch": [], "y": []} for _, name in metric_cols
            }
            chunk_buf: dict[str, dict[str, list]] = {
                name: {"step": [], "epoch": [], "y": []} for _, name in metric_cols
            }

            CHUNK_ROWS = 5000  # flush after this many rows...
            CHUNK_INTERVAL = 0.2  # ...or this many seconds, whichever first.
            chunk_id = 0
            last_flush = time.monotonic()
            row_count = 0

            def _flush_chunk():
                nonlocal chunk_id, chunk_buf, last_flush
                # Drop empty metric columns from the chunk so transmission
                # stays sparse.
                payload = {k: v for k, v in chunk_buf.items() if v["y"]}
                if payload:
                    out = {"chunk": chunk_id, "metrics": payload}
                    chunk_id += 1
                    chunk_buf = {
                        name: {"step": [], "epoch": [], "y": []}
                        for _, name in metric_cols
                    }
                    last_flush = time.monotonic()
                    return out
                last_flush = time.monotonic()
                return None

            for row_idx, row in enumerate(reader):
                row_count += 1
                step = _maybe_float(row, step_idx)
                epoch = _maybe_float(row, epoch_idx)
                if step is None and epoch is None:
                    step = float(row_idx)
                for i, name in metric_cols:
                    if i >= len(row) or row[i] == "":
                        continue
                    try:
                        y = float(row[i])
                    except ValueError:
                        continue
                    full[name]["step"].append(step)
                    full[name]["epoch"].append(epoch)
                    full[name]["y"].append(y)
                    chunk_buf[name]["step"].append(step)
                    chunk_buf[name]["epoch"].append(epoch)
                    chunk_buf[name]["y"].append(y)

                if row_count % 1024 == 0 and (
                    row_count >= CHUNK_ROWS
                    or time.monotonic() - last_flush >= CHUNK_INTERVAL
                ):
                    out = _flush_chunk()
                    if out is not None:
                        yield out

            # Final partial chunk
            out = _flush_chunk()
            if out is not None:
                yield out

        # Populate cache from the in-memory full structure.
        result = {"metrics": {k: v for k, v in full.items() if v["y"]}}
        if cache_key is not None:
            self._metrics_cache[run_id] = (
                cache_key[0],
                cache_key[1],
                result,
                None,  # JSON bytes lazily populated by metrics_json_bytes
            )
        yield {"done": True}

    def metrics_json_bytes(self, run_id: str) -> Optional[bytes]:
        """Return the metrics response as pre-serialised, NaN-safe JSON bytes.

        First call materialises the dict (reusing the structured cache via
        :meth:`metrics_json`) and serialises once; subsequent calls return
        the cached bytes directly so the HTTP layer is reduced to a memcpy.
        """
        # Check bytes cache directly.
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            return None
        mpath = run.run_dir / "metrics.csv"
        try:
            st = mpath.stat()
            cache_key = (st.st_mtime, st.st_size)
        except OSError:
            cache_key = None
        if cache_key is not None:
            cached = self._metrics_cache.get(run_id)
            if (
                cached is not None
                and (cached[0], cached[1]) == cache_key
                and cached[3] is not None
            ):
                return cached[3]

        data = self.metrics_json(run_id)
        if data is None:
            return None
        # Sanitize once (replace NaN/Inf with None) and serialise.
        from .server import _safe_dumps  # local import to avoid cycle

        body = _safe_dumps(data).encode("utf-8")
        if cache_key is not None:
            cached = self._metrics_cache.get(run_id)
            if cached is not None and (cached[0], cached[1]) == cache_key:
                self._metrics_cache[run_id] = (
                    cached[0],
                    cached[1],
                    cached[2],
                    body,
                )
        return body

    def media_json(self, run_id: str) -> Optional[dict]:
        """Return the media events for a run by parsing ``media.jsonl``.

        Returns ``None`` if the run is unknown.  Returns
        ``{"events": []}`` if there is no media yet (empty/missing file).
        Each event has at least ``step``, ``tag``, ``type``, ``path``;
        videos may also have ``fps`` and ``format``.
        """
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            return None
        mpath = run.run_dir / "media.jsonl"
        if not mpath.is_file():
            return {"events": []}
        events: list[dict] = []
        try:
            with mpath.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip a partially-written line; the next pass picks
                        # it up after the writer fsyncs.
                        continue
        except OSError:
            pass
        return {"events": events}

    def media_file_path(self, run_id: str, rel_path: str) -> Optional[Path]:
        """Resolve a media file path safely, with ``..`` traversal blocked.

        The resolved file must live under ``{run_dir}/media/`` and must
        actually exist as a file.  Returns ``None`` otherwise — the
        caller should respond with 404.
        """
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            return None
        # Reject anything obviously malicious before doing path math.
        if not rel_path or rel_path.startswith("/") or ".." in rel_path.split("/"):
            return None
        base = run.run_dir.resolve()
        media_root = (base / "media").resolve()
        target = (base / rel_path).resolve()
        # Must be inside {run_dir}/media/ — guards against absolute /etc/...
        # symlinks and any escape via canonicalisation.
        try:
            target.relative_to(media_root)
        except ValueError:
            return None
        if not target.is_file():
            return None
        return target


def _maybe_float(row: list[str], idx: Optional[int]) -> Optional[float]:
    if idx is None or idx >= len(row) or row[idx] == "":
        return None
    try:
        return float(row[idx])
    except ValueError:
        return None
