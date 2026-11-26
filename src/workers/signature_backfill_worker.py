"""
Very-low-priority background worker that backfills `media_signatures`
(pHash, dHash, Laplacian sharpness, 24-byte LAB Haar) from already-cached
WebP thumbnails in `thumbnails.db`.

Never opens original photo files or triggers cloud/OneDrive downloads.
Resumes cleanly across restarts and exits permanently once caught up.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from src.core.duplicate_finder import (
    compute_image_dhash,
    compute_image_phash,
    compute_lab_haar_signature,
    compute_sharpness_score,
)
from src.db.connection import DualDatabaseEngine
from src.db.repository import ThumbnailRepository

_PG64_MAGIC = b"PG64THMB"


class SignatureBackfillSignals(QObject):
    progress = Signal(int, int)       # (done, total)
    finished_stats = Signal(int)      # (newly_computed_count)
    error = Signal(str)


class SignatureBackfillWorker(QRunnable):
    """Gentle background backfill of `media_signatures` from existing `thumbnails.db` WebP blobs."""

    def __init__(
        self,
        engine: DualDatabaseEngine,
        cancel_flag: Any = None,
        is_paused_cb: Callable[[], bool] | None = None,
        chunk_size: int = 50,
        chunk_sleep_sec: float = 0.06,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.cancel_flag = cancel_flag
        self.is_paused_cb = is_paused_cb
        self.chunk_size = max(10, int(chunk_size))
        self.chunk_sleep_sec = max(0.0, float(chunk_sleep_sec))
        self.signals = SignatureBackfillSignals()

    def _should_stop(self) -> bool:
        if self.cancel_flag is not None and self.cancel_flag.is_set():
            return True
        if callable(self.is_paused_cb):
            try:
                if self.is_paused_cb():
                    return True
            except Exception:
                pass
        return False

    def run(self) -> None:
        processed = 0
        try:
            tconn = self.engine.thumbnails_conn()
            trepo = ThumbnailRepository(self.engine)
            # Quick count of cached thumbnails missing a current media_signatures entry
            row_tot = tconn.execute(
                """
                SELECT COUNT(*) FROM thumbnails t
                LEFT JOIN media_signatures ms ON ms.media_id = t.media_id
                WHERE t.webp_blob IS NOT NULL
                  AND (ms.media_id IS NULL OR ms.mtime != t.mtime OR ms.lab_haar IS NULL OR length(ms.lab_haar) != 24);
                """
            ).fetchone()
            total_missing = int(row_tot[0]) if row_tot else 0
            if total_missing <= 0:
                self.signals.finished_stats.emit(0)
                return

            last_mid = 0
            while not self._should_stop():
                rows = tconn.execute(
                    """
                    SELECT t.media_id, t.mtime, t.webp_blob
                    FROM thumbnails t
                    LEFT JOIN media_signatures ms ON ms.media_id = t.media_id
                    WHERE t.media_id > ?
                      AND t.webp_blob IS NOT NULL
                      AND (ms.media_id IS NULL OR ms.mtime != t.mtime OR ms.lab_haar IS NULL OR length(ms.lab_haar) != 24)
                    ORDER BY t.media_id ASC
                    LIMIT ?;
                    """,
                    (last_mid, self.chunk_size),
                ).fetchall()
                if not rows:
                    break

                batch_to_save: list[tuple[int, int, int, int, float, bytes]] = []
                for mid_raw, mtime_raw, blob_raw in rows:
                    mid = int(mid_raw)
                    last_mid = mid
                    if self._should_stop():
                        break
                    if not blob_raw:
                        continue
                    raw_bytes = bytes(blob_raw)
                    if raw_bytes.startswith(_PG64_MAGIC):
                        raw_bytes = raw_bytes[len(_PG64_MAGIC):]
                    if not raw_bytes:
                        continue
                    arr_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
                    if arr_bgr is None or arr_bgr.size == 0:
                        continue
                    arr_gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)
                    ph = compute_image_phash(arr_gray)
                    dh = compute_image_dhash(arr_gray)
                    sharp = compute_sharpness_score(arr_gray)
                    haar = compute_lab_haar_signature(arr_bgr)
                    batch_to_save.append((mid, int(mtime_raw or 0), ph, dh, sharp, haar))

                if batch_to_save:
                    trepo.upsert_signatures_batch(batch_to_save)
                    processed += len(batch_to_save)
                    self.signals.progress.emit(min(processed, total_missing), total_missing)

                if self.chunk_sleep_sec > 0.0 and not self._should_stop():
                    time.sleep(self.chunk_sleep_sec)

            self.signals.finished_stats.emit(processed)
        except Exception as exc:
            self.signals.error.emit(str(exc))
