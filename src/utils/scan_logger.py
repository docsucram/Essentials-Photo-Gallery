import datetime
import faulthandler
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from src.utils.paths import get_app_data_dir


_CRASH_TRACE_LOCK = threading.Lock()
_FAULT_LOG_HANDLE: Any = None
_ACTIVE_TRACE_PATHS: dict[int, tuple[str, str]] = {}  # thread_id -> (stage, file_path)
_STARTUP_T0 = time.perf_counter()
_STARTUP_STAGES: dict[str, float] = {}


def get_crash_trace_path(base_dir: Path | str | None = None) -> Path:
    root = Path(base_dir) if base_dir is not None else get_app_data_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root / "active_scan_trace.json"


def record_active_decode(stage: str, file_path: str | Path, base_dir: Path | str | None = None) -> None:
    """
    Writes a lightweight crash-recovery breadcrumb before risky native C-extension decodes
    (rawpy, OpenCV, pillow-heif, ONNX). If the process hard-crashes (SIGSEGV / Access Violation),
    the next startup can report the exact culprit file and offer to skip it.
    """
    if "PYTEST_CURRENT_TEST" in os.environ and base_dir is None:
        return
    tid = threading.get_ident()
    path_str = str(file_path)
    with _CRASH_TRACE_LOCK:
        _ACTIVE_TRACE_PATHS[tid] = (stage, path_str)
        try:
            trace_file = get_crash_trace_path(base_dir)
            payload = {
                "pid": os.getpid(),
                "stage": stage,
                "file_path": path_str,
                "timestamp": int(time.time()),
            }
            trace_file.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass


def clear_active_decode(base_dir: Path | str | None = None) -> None:
    """Clears the current thread's active decode breadcrumb once native decoding finishes safely."""
    if "PYTEST_CURRENT_TEST" in os.environ and base_dir is None:
        return
    tid = threading.get_ident()
    with _CRASH_TRACE_LOCK:
        _ACTIVE_TRACE_PATHS.pop(tid, None)
        if _ACTIVE_TRACE_PATHS:
            # Another worker thread still has an active decode; leave its entry recorded
            stage, path_str = next(iter(_ACTIVE_TRACE_PATHS.values()))
            try:
                trace_file = get_crash_trace_path(base_dir)
                trace_file.write_text(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "stage": stage,
                            "file_path": path_str,
                            "timestamp": int(time.time()),
                        }
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass
            return
        try:
            trace_file = get_crash_trace_path(base_dir)
            if trace_file.exists():
                trace_file.unlink(missing_ok=True)
        except Exception:
            pass


def read_stale_crash_trace(base_dir: Path | str | None = None) -> dict[str, Any] | None:
    """
    Returns crash-trace metadata if a previous run terminated unexpectedly mid-decode
    (different PID than the current process), or None if clean.
    """
    try:
        trace_file = get_crash_trace_path(base_dir)
        if not trace_file.is_file():
            return None
        data = json.loads(trace_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data.get("file_path"):
            return None
        if int(data.get("pid") or -1) == os.getpid():
            return None
        return data
    except Exception:
        return None


def clear_stale_crash_trace(base_dir: Path | str | None = None) -> None:
    """Removes any stale crash-recovery trace file after prompting or recovery."""
    try:
        trace_file = get_crash_trace_path(base_dir)
        if trace_file.exists():
            trace_file.unlink(missing_ok=True)
    except Exception:
        pass


def mark_startup_stage(stage: str) -> float:
    """Records and returns elapsed milliseconds since process start for a named startup stage."""
    elapsed_ms = round((time.perf_counter() - _STARTUP_T0) * 1000.0, 2)
    if stage not in _STARTUP_STAGES:
        _STARTUP_STAGES[stage] = elapsed_ms
    return _STARTUP_STAGES[stage]


def get_startup_timings() -> dict[str, float]:
    """Returns a copy of recorded startup stage timestamps (in ms)."""
    return dict(_STARTUP_STAGES)


def _get_log_dir() -> Path:
    """Returns a writable directory for application logs."""
    local_logs = Path("logs")
    try:
        local_logs.mkdir(parents=True, exist_ok=True)
        t = local_logs / ".w"
        t.touch()
        t.unlink()
        return local_logs
    except (PermissionError, OSError):
        p = get_app_data_dir() / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p


def log_scan_error(worker_name: str, file_path: str | Path, error: Exception | str) -> None:
    """
    Appends broken or corrupted media file details to logs/scan_errors.log.
    Safe against I/O exceptions and creates the logs directory if it does not exist.
    """
    try:
        log_dir = _get_log_dir()
        log_file = log_dir / "scan_errors.log"
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] [{worker_name}] Error processing '{file_path}': {error}\n")
    except Exception:
        pass


def install_global_crash_logger(log_dir: Path | str | None = None) -> None:
    """
    Installs a sys.excepthook crash handler that logs uncaught exceptions
    and full tracebacks to logs/crash.log without terminating the GUI thread.
    """
    target_dir = Path(log_dir) if log_dir else _get_log_dir()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    crash_file = target_dir / "crash.log"

    if "PYTEST_CURRENT_TEST" not in os.environ:
        global _FAULT_LOG_HANDLE
        try:
            _FAULT_LOG_HANDLE = open(crash_file, "a", encoding="utf-8")
            faulthandler.enable(file=_FAULT_LOG_HANDLE, all_threads=True)
        except Exception:
            pass

    def _crash_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        entry = f"[{ts}] UNCAUGHT EXCEPTION:\n{tb_text}\n{'=' * 70}\n"
        try:
            with open(crash_file, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception:
            pass
        sys.stderr.write(entry)
        sys.stderr.flush()

    sys.excepthook = _crash_handler


class UniversalToolTipFilter:
    """Placeholder to be initialized after PySide6 imports."""
    pass


def install_universal_tooltip_filter(app: Any = None) -> Any:
    """
    Installs a universal event filter intercepting all QTipLabel / ToolTip widgets across the app.
    Guarantees tooltips never render with system dark mode / black backgrounds regardless
    of local widget stylesheets or child hierarchy overrides.
    """
    from PySide6.QtCore import QEvent, QObject, Qt
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QApplication, QWidget

    class _ToolTipFilter(QObject):
        def eventFilter(self, obj: QObject, event: QEvent) -> bool:
            if getattr(obj, "objectName", lambda: "")() == "qtooltip_label" or (
                isinstance(obj, QWidget) and obj.windowType() == Qt.WindowType.ToolTip
            ):
                if not getattr(obj, "_styled_universal", False):
                    obj._styled_universal = True
                    try:
                        pal = obj.palette()
                        pal.setColor(QPalette.ColorRole.Window, QColor("#FFFFFF"))
                        pal.setColor(QPalette.ColorRole.WindowText, QColor("#1E395B"))
                        pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#FFFFFF"))
                        pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#1E395B"))
                        obj.setPalette(pal)
                        obj.setStyleSheet(
                            "background-color: #FFFFFF !important; "
                            "color: #1E395B !important; "
                            "border: 1px solid #7DA2CE !important; "
                            "border-radius: 2px !important; "
                            "padding: 3px 6px !important; "
                            "font-family: 'Segoe UI', sans-serif !important; "
                            "font-size: 9pt !important;"
                        )
                    except Exception:
                        pass
            return False

    if app is None:
        app = QApplication.instance()
    if app is None:
        return None
    if not hasattr(app, "_universal_tooltip_filter"):
        filt = _ToolTipFilter(app)
        app.installEventFilter(filt)
        app._universal_tooltip_filter = filt
    return getattr(app, "_universal_tooltip_filter", None)

