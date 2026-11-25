"""
Lightweight Main-Thread Freeze Watchdog for diagnosing UI stalls in production builds.

A QTimer on the Qt main GUI thread updates a monotonic heartbeat timestamp every
`heartbeat_ms` (default 200 ms). A daemon thread checks that timestamp periodically;
whenever the main GUI thread fails to respond for >= `threshold_ms` (default 600 ms),
the watchdog dumps the exact Python stack traces of the main thread and every active
background thread to `%LOCALAPPDATA%/PhotoGallery/logs/freeze_traces.log`.

Overhead during normal operation is negligible (one float assignment every 200 ms).
"""
from __future__ import annotations

import datetime
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer

from src.utils.paths import get_app_data_dir


MAX_LOG_BYTES = 512 * 1024  # 512 KB cap (trimmed in-place to keep latest ~256 KB)
TRIM_KEEP_BYTES = 256 * 1024


def get_freeze_log_path(base_dir: Path | str | None = None) -> Path:
    root = Path(base_dir) if base_dir is not None else get_app_data_dir()
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / "freeze_traces.log"


def _format_all_thread_stacks(main_tid: int, watchdog_tid: int) -> str:
    frames = sys._current_frames()
    thread_names: dict[int, str] = {
        t.ident: t.name for t in threading.enumerate() if t.ident is not None
    }
    lines: list[str] = []

    # 1. Main GUI thread first
    main_frame = frames.get(main_tid)
    lines.append(f"--- MAIN GUI THREAD (tid={main_tid}) ---")
    if main_frame is not None:
        lines.extend(traceback.format_stack(main_frame))
    else:
        lines.append("  <main thread frame unavailable>\n")

    # 2. All other non-watchdog threads
    other_tids = [tid for tid in frames if tid not in (main_tid, watchdog_tid)]
    for tid in other_tids:
        tname = thread_names.get(tid, "UnknownThread")
        lines.append(f"--- WORKER THREAD '{tname}' (tid={tid}) ---")
        lines.extend(traceback.format_stack(frames[tid]))

    return "".join(lines)


class MainThreadFreezeWatchdog(QObject):
    """Detects main-thread UI stalls and logs full multi-thread stack traces."""

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        heartbeat_ms: int = 200,
        threshold_ms: int = 600,
        log_path: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._heartbeat_ms = max(50, int(heartbeat_ms))
        self._threshold_s = max(0.15, float(threshold_ms) / 1000.0)
        self._log_path = log_path or get_freeze_log_path()
        self._main_tid = threading.main_thread().ident or threading.get_ident()
        self._start_mono = time.monotonic()
        self._last_beat_mono = time.monotonic()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._dump_count = 0

        self._timer = QTimer(self)
        self._timer.setInterval(self._heartbeat_ms)
        self._timer.timeout.connect(self._on_heartbeat)

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def dump_count(self) -> int:
        return self._dump_count

    def _on_heartbeat(self) -> None:
        self._last_beat_mono = time.monotonic()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        now = time.monotonic()
        self._start_mono = now
        self._last_beat_mono = now
        self._timer.start()
        self._thread = threading.Thread(
            target=self._watch_loop,
            name="FreezeWatchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        try:
            self._timer.stop()
        except Exception:
            pass
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _append_log(self, text: str) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            if self._log_path.exists() and self._log_path.stat().st_size > MAX_LOG_BYTES:
                try:
                    raw = self._log_path.read_bytes()
                    tail = raw[-TRIM_KEEP_BYTES:]
                    nl = tail.find(b"\n")
                    if nl != -1:
                        tail = tail[nl + 1 :]
                    self._log_path.write_bytes(tail)
                except OSError:
                    pass
            with self._log_path.open("a", encoding="utf-8", errors="replace") as fh:
                fh.write(text)
                fh.flush()
        except Exception:
            pass

    def _watch_loop(self) -> None:
        watchdog_tid = threading.get_ident()
        poll_s = min(0.15, self._threshold_s / 3.0)
        # Capture up to 3 snapshots per freeze episode (e.g. at 0.6s, 2.5s, 6.0s)
        snapshot_multipliers = (1.0, 4.0, 10.0)
        in_freeze = False
        freeze_started_beat = 0.0
        snapshots_taken = 0

        while not self._stop_event.wait(poll_s):
            now = time.monotonic()
            beat = self._last_beat_mono
            stall_s = now - beat

            if stall_s >= self._threshold_s:
                if not in_freeze or beat != freeze_started_beat:
                    in_freeze = True
                    freeze_started_beat = beat
                    snapshots_taken = 0

                if snapshots_taken < len(snapshot_multipliers):
                    next_trigger = self._threshold_s * snapshot_multipliers[snapshots_taken]
                    if stall_s >= next_trigger:
                        snapshots_taken += 1
                        self._dump_count += 1
                        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        uptime_s = now - self._start_mono
                        header = (
                            f"\n======================================================================\n"
                            f"[FREEZE DETECTED #{self._dump_count} snap {snapshots_taken}/3] "
                            f"{ts} | pid={os.getpid()} | uptime={uptime_s:.1f}s | stall={stall_s*1000:.0f}ms\n"
                            f"======================================================================\n"
                        )
                        body = _format_all_thread_stacks(self._main_tid, watchdog_tid)
                        self._append_log(header + body + "\n")
            elif in_freeze:
                total_stall = max(0.0, beat - freeze_started_beat)
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self._append_log(
                    f"[FREEZE RECOVERED] {ts} | total_block~={total_stall:.2f}s "
                    f"(snapshots={snapshots_taken})\n"
                )
                in_freeze = False
                snapshots_taken = 0


def install_freeze_watchdog(app: Any) -> MainThreadFreezeWatchdog | None:
    """Install and start the MainThreadFreezeWatchdog on `app` (disabled under pytest)."""
    if "PYTEST_CURRENT_TEST" in os.environ:
        return None
    if os.environ.get("GALLERY_DISABLE_FREEZE_WATCHDOG") == "1":
        return None
    try:
        threshold_ms = int(os.environ.get("GALLERY_FREEZE_MS", "600"))
    except ValueError:
        threshold_ms = 600
    wd = MainThreadFreezeWatchdog(parent=app, heartbeat_ms=200, threshold_ms=threshold_ms)
    wd.start()
    try:
        app.aboutToQuit.connect(wd.stop)
    except Exception:
        pass
    try:
        app._freeze_watchdog = wd
    except Exception:
        pass
    return wd
