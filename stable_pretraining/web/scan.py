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
import queue
import threading
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

    # ---- lifecycle ----

    def start(self) -> None:
        self._scan()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="spt-web-scanner"
        )
        self._thread.start()

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

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_interval):
            try:
                changed, removed = self._scan()
                if changed or removed:
                    self._publish("update", {"changed": changed, "removed": removed})
            except Exception:
                # Scan errors are recoverable; the next tick will retry.
                pass

    def _scan(self) -> tuple[list[str], list[str]]:
        seen: set[str] = set()
        changed: list[str] = []

        for sidecar_path in self.root.rglob("sidecar.json"):
            try:
                st = sidecar_path.stat()
            except OSError:
                continue
            run_dir = sidecar_path.parent
            run_id = self._run_id_for(run_dir)
            seen.add(run_id)

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

            if (
                existing is None
                or existing.sidecar_mtime != st.st_mtime
                or existing.metrics_mtime != m_mtime
                or existing.metrics_size != m_size
                or existing.media_mtime != med_mtime
                or existing.media_size != med_size
            ):
                try:
                    sidecar = json.loads(sidecar_path.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                with self._lock:
                    self._runs[run_id] = _Run(
                        run_id=run_id,
                        run_dir=run_dir,
                        sidecar_mtime=st.st_mtime,
                        metrics_mtime=m_mtime,
                        metrics_size=m_size,
                        media_mtime=med_mtime,
                        media_size=med_size,
                        sidecar=sidecar,
                    )
                changed.append(run_id)

        with self._lock:
            removed = [rid for rid in self._runs if rid not in seen]
            for rid in removed:
                del self._runs[rid]

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

        out = {k: v for k, v in metrics.items() if v["y"]}
        return {"metrics": out}

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
