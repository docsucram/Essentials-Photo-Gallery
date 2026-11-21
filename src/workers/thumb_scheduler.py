"""
Two-tier thumbnail & face-crop scheduler for Essentials Photo Gallery:
1. BatchThumbCacheWorker: Single batched SQLite lookup from `thumbnails.db` on a dedicated
   fast-read pool so already-cached thumbnails populate in <5 ms without waiting behind slow
   RAW / video disk decodes.
2. BatchFaceCropWorker: Off-thread batched face-crop prefetch eliminating synchronous SQLite
   reads inside `MediaCardDelegate.paint()`.
3. ThumbnailWriteCoalescer: Coalesces decoded WebP blobs into batched SQLite transactions
   (`put_webp_blobs_batch`) instead of 1-by-1 single-row commits.
"""
from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QTimer, Signal, Slot
from PySide6.QtGui import QImage

from src.db.connection import DualDatabaseEngine
from src.db.repository import ThumbnailRepository


class _BatchThumbCacheSignals(QObject):
    # token, decoded_items: list[(media_id, QImage)], missing_ids: list[int]
    batch_ready = Signal(int, list, list)


class BatchThumbCacheWorker(QRunnable):
    """
    Reads a viewport batch of WebP thumbnails from `thumbnails.db` in ONE SQLite query,
    decodes them to QImages on the fast-read worker thread, and reports cache misses
    for the priority disk-decode pool.
    """

    def __init__(
        self,
        engine: DualDatabaseEngine,
        token: int,
        media_ids: Sequence[int],
        expected_mtimes: dict[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.token = int(token)
        self.media_ids = [int(m) for m in media_ids]
        self.expected_mtimes = dict(expected_mtimes) if expected_mtimes else None
        self.signals = _BatchThumbCacheSignals()

    @Slot()
    def run(self) -> None:
        decoded: list[tuple[int, QImage]] = []
        missing: list[int] = []
        try:
            thumb_repo = ThumbnailRepository(self.engine)
            blobs = thumb_repo.get_webp_blobs_batch(
                self.media_ids, expected_mtimes=self.expected_mtimes
            )
            for mid in self.media_ids:
                blob = blobs.get(mid)
                if blob:
                    qimg = QImage.fromData(blob, "WEBP")
                    if not qimg.isNull():
                        decoded.append((mid, qimg))
                        continue
                missing.append(mid)
        except Exception:
            missing = list(self.media_ids)
        finally:
            try:
                if threading.current_thread() is not threading.main_thread():
                    self.engine.close_thread_connections()
            except Exception:
                pass
        self.signals.batch_ready.emit(self.token, decoded, missing)


class _BatchFaceCropSignals(QObject):
    # token, person_id, decoded_items: list[(media_id, QImage)]
    crops_ready = Signal(int, int, list)


class BatchFaceCropWorker(QRunnable):
    """Prefetches confirmed face-crop WebP blobs off the GUI thread in one batched query."""

    def __init__(
        self,
        engine: DualDatabaseEngine,
        token: int,
        person_id: int,
        media_ids: Sequence[int],
    ) -> None:
        super().__init__()
        self.engine = engine
        self.token = int(token)
        self.person_id = int(person_id)
        self.media_ids = [int(m) for m in media_ids]
        self.signals = _BatchFaceCropSignals()

    @Slot()
    def run(self) -> None:
        decoded: list[tuple[int, QImage]] = []
        try:
            thumb_repo = ThumbnailRepository(self.engine)
            blobs = thumb_repo.get_face_thumbnails_for_media_person_batch(
                self.media_ids, self.person_id
            )
            for mid, blob in blobs.items():
                if not blob:
                    continue
                qimg = QImage.fromData(blob, "WEBP")
                if not qimg.isNull():
                    decoded.append((int(mid), qimg))
        except Exception:
            pass
        finally:
            try:
                if threading.current_thread() is not threading.main_thread():
                    self.engine.close_thread_connections()
            except Exception:
                pass
        self.signals.crops_ready.emit(self.token, self.person_id, decoded)


class ThumbnailWriteCoalescer(QObject):
    """
    Thread-safe write buffer coalescing newly generated WebP thumbnails into batched
    `thumbnails.db` commits (`put_webp_blobs_batch`) every `batch_size` items or `flush_ms`.
    """

    def __init__(
        self,
        engine: DualDatabaseEngine,
        *,
        batch_size: int = 16,
        flush_ms: int = 150,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.batch_size = max(1, int(batch_size))
        self._lock = threading.Lock()
        self._pending: dict[int, tuple[int, bytes, int, int, int]] = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(max(20, int(flush_ms)))
        self._timer.timeout.connect(self.flush)

    def enqueue(
        self,
        media_id: int,
        webp_bytes: bytes,
        mtime: int,
        width: int = 256,
        height: int = 256,
    ) -> None:
        """Queues a WebP blob from any worker thread; flushes automatically when full."""
        should_flush_now = False
        with self._lock:
            self._pending[int(media_id)] = (
                int(media_id),
                webp_bytes,
                int(mtime),
                int(width),
                int(height),
            )
            if len(self._pending) >= self.batch_size:
                should_flush_now = True
        if should_flush_now:
            self.flush()
        else:
            # Arm single-shot flush timer safely from any thread
            QTimer.singleShot(0, self._arm_timer)

    @Slot()
    def _arm_timer(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    @Slot()
    def flush(self) -> int:
        """Commits all buffered WebP thumbnails in one batched SQLite transaction."""
        with self._lock:
            if not self._pending:
                return 0
            batch = list(self._pending.values())
            self._pending.clear()
        try:
            thumb_repo = ThumbnailRepository(self.engine)
            thumb_repo.put_webp_blobs_batch(batch)
            return len(batch)
        except Exception:
            return 0


def order_ids_by_viewport_center(media_ids: Sequence[int]) -> list[int]:
    """
    Orders visible + buffer media IDs starting from the center of the viewport outward
    so tiles directly in front of the user decode before off-screen prefetch buffer items.
    """
    n = len(media_ids)
    if n <= 2:
        return list(media_ids)
    mid = n // 2
    ordered: list[int] = [media_ids[mid]]
    offset = 1
    while mid - offset >= 0 or mid + offset < n:
        if mid - offset >= 0:
            ordered.append(media_ids[mid - offset])
        if mid + offset < n:
            ordered.append(media_ids[mid + offset])
        offset += 1
    return ordered
