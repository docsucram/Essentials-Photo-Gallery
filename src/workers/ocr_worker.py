"""
Background OCR Scan Worker:
Queries non-video media where ``ocr_scanned = 0 AND f.scan_ocr = 1 AND f.monitor_mode != 'excluded'``,
decodes a fast <=1024px RGB proxy (with thumbnails.db fallback), extracts text via ``OcrEngine``,
and stores normalized text in ``media.ocr_text`` + ``media_ocr_fts``.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from src.core.ocr_engine import OcrEngine
from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository

logger = logging.getLogger(__name__)


class OcrWorkerSignals(QObject):
    started = Signal()
    progressed = Signal(int, int)  # (done, total)
    status_message = Signal(str)
    finished = Signal(int)  # processed_count
    error = Signal(str)


class OcrScanWorker(QRunnable):
    """Background QRunnable that indexes Text-in-Photo (OCR) into catalog.db + FTS5."""

    def __init__(
        self,
        engine: DualDatabaseEngine,
        *,
        folder_id: int | None = None,
        batch_size: int = 24,
        ocr_engine: OcrEngine | None = None,
    ) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.engine = engine
        self.repo = CatalogRepository(engine)
        self.thumb_repo = ThumbnailRepository(engine)
        self.folder_id = folder_id
        self.batch_size = max(1, int(batch_size))
        self.ocr_engine = ocr_engine or OcrEngine()
        self.signals = OcrWorkerSignals()
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def _is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _load_rgb_proxy(self, media_id: int, file_path: Path) -> np.ndarray | None:
        """Load up to 1024px RGB proxy from disk, falling back to cached WebP thumbnail."""
        if file_path.is_file():
            try:
                raw = np.fromfile(str(file_path), dtype=np.uint8)
                bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if bgr is not None and bgr.size > 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    if max(h, w) > 1024:
                        scale = 1024.0 / float(max(h, w))
                        rgb = cv2.resize(
                            rgb,
                            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                            interpolation=cv2.INTER_AREA,
                        )
                    return rgb
            except Exception as exc:
                logger.debug("Direct OCR decode failed for %s: %s", file_path, exc)

        # Fallback to cached thumbnail blob if original file is offline/slow
        try:
            blob = self.thumb_repo.get_webp_blob(media_id)
            if blob:
                arr = np.frombuffer(blob, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None and bgr.size > 0:
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        return None

    def run(self) -> None:
        processed = 0
        try:
            self.signals.started.emit()
            conn = self.engine.catalog_conn()

            folder_filter = ""
            params: list[Any] = []
            if self.folder_id is not None:
                folder_filter = """
                    AND m.folder_id IN (
                        WITH RECURSIVE sub(id) AS (
                            SELECT ?
                            UNION ALL
                            SELECT f2.id FROM folders f2 JOIN sub s ON f2.parent_id = s.id
                        )
                        SELECT id FROM sub
                    )
                """
                params.append(int(self.folder_id))

            count_row = conn.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM media m
                JOIN folders f ON f.id = m.folder_id
                WHERE COALESCE(m.is_missing, 0) = 0
                  AND COALESCE(m.is_video, 0) = 0
                  AND COALESCE(m.ocr_scanned, 0) = 0
                  AND COALESCE(f.scan_ocr, 1) = 1
                  AND COALESCE(f.monitor_mode, 'always') != 'excluded'
                  {folder_filter};
                """,
                params,
            ).fetchone()
            total = int(count_row["cnt"] or 0) if count_row else 0
            if total == 0:
                self.signals.finished.emit(0)
                return

            self.signals.status_message.emit(f"Indexing text in {total:,} photo(s) (OCR)...")
            self.signals.progressed.emit(0, total)

            while not self._is_cancelled():
                batch_rows = list(conn.execute(
                    f"""
                    SELECT m.id, m.filename, f.path AS folder_path
                    FROM media m
                    JOIN folders f ON f.id = m.folder_id
                    WHERE COALESCE(m.is_missing, 0) = 0
                      AND COALESCE(m.is_video, 0) = 0
                      AND COALESCE(m.ocr_scanned, 0) = 0
                      AND COALESCE(f.scan_ocr, 1) = 1
                      AND COALESCE(f.monitor_mode, 'always') != 'excluded'
                      {folder_filter}
                    ORDER BY m.id ASC
                    LIMIT ?;
                    """,
                    params + [self.batch_size],
                ).fetchall() or ())
                if not batch_rows:
                    break

                for row in batch_rows:
                    if self._is_cancelled():
                        break
                    mid = int(row["id"])
                    fpath = Path(str(row["folder_path"])) / str(row["filename"])
                    rgb = self._load_rgb_proxy(mid, fpath)
                    text = self.ocr_engine.extract_text(rgb) if rgb is not None else ""
                    self.repo.update_media_ocr(mid, text)
                    processed += 1
                    self.signals.progressed.emit(min(processed, total), total)
                    # Yield briefly so background OCR never starves UI navigation or thumbnail workers
                    time.sleep(0.02)

            self.signals.finished.emit(processed)
        except Exception as exc:
            logger.exception("OcrScanWorker failed: %s", exc)
            self.signals.error.emit(str(exc))
            self.signals.finished.emit(processed)
        finally:
            try:
                self.engine.close_thread_connections()
            except Exception:
                pass
