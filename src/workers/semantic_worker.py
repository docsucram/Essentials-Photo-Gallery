"""
Background Semantic Embedding Index Worker for MobileCLIP-S1 ONNX Add-on.

Strictly reads cached WebP thumbnails from `thumbnails.db` (never reads original
full-res files on disk or triggers cloud OneDrive placeholder downloads).
"""
from __future__ import annotations

import threading
import time
from typing import Any, Sequence

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from legacy_reference.imgBrowserv13 import CancellationFlag
from src.core.semantic_engine import MODEL_VERSION_ID, SemanticEngine, SemanticVectorRepository
from src.db.connection import DualDatabaseEngine
from src.db.repository import ThumbnailRepository


class SemanticIndexWorkerSignals(QObject):
    progress = Signal(int, int, str)        # done, total, message
    finished_stats = Signal(int, int)       # indexed_now, total_indexed
    error = Signal(str)


class SemanticIndexWorker(QRunnable):
    """Low-priority worker computing float16 MobileCLIP embeddings from cached thumbnails."""

    def __init__(
        self,
        db_engine: DualDatabaseEngine,
        semantic_repo: SemanticVectorRepository,
        semantic_engine: SemanticEngine,
        candidates: Sequence[tuple[int, int]] | None = None,
        *,
        cancel_flag: CancellationFlag | None = None,
        is_paused_cb: Any | None = None,
        batch_size: int = 32,
        model_ver: str = MODEL_VERSION_ID,
    ) -> None:
        super().__init__()
        self.db_engine = db_engine
        self.semantic_repo = semantic_repo
        self.semantic_engine = semantic_engine
        self.candidates: list[tuple[int, int]] | None = (
            None if candidates is None else [(int(m), int(mt)) for m, mt in candidates]
        )
        self.cancel_flag = cancel_flag or CancellationFlag()
        self.is_paused_cb = is_paused_cb
        self.batch_size = max(4, int(batch_size))
        self.model_ver = str(model_ver)
        self.signals = SemanticIndexWorkerSignals()

    @Slot()
    def run(self) -> None:
        if self.cancel_flag.is_set or not self.semantic_engine.is_available():
            try:
                self.signals.finished_stats.emit(
                    0, self.semantic_repo.count_indexed(model_ver=self.model_ver)
                )
            except Exception:
                pass
            return

        thumb_repo = ThumbnailRepository(self.db_engine)
        try:
            # Precompute SUBJECT_PRESETS text vectors on this background thread so the 64 MB
            # text ONNX encoder is never loaded synchronously on the main GUI thread.
            if hasattr(self.semantic_engine, "ensure_subject_vectors"):
                self.semantic_engine.ensure_subject_vectors(
                    self.semantic_repo, model_ver=self.model_ver
                )
            if self.candidates is None:
                rows = self.db_engine.catalog_conn().execute(
                    """
                    SELECT id, mtime
                    FROM media
                    WHERE is_video = 0 AND is_missing = 0
                    ORDER BY date_taken DESC, id DESC;
                    """
                ).fetchall()
                candidates = [(int(r[0]), int(r[1])) for r in rows]
            else:
                candidates = self.candidates

            existing = self.semantic_repo.get_indexed_mtimes(
                [m for m, _ in candidates], model_ver=self.model_ver
            )
            todo = [(m, mt) for m, mt in candidates if existing.get(m) != mt]
            total = len(todo)
            indexed_now = 0

            for start in range(0, total, self.batch_size):
                if self.cancel_flag.is_set:
                    break
                if self.is_paused_cb is not None and self.is_paused_cb():
                    break

                chunk = todo[start : start + self.batch_size]
                blobs = thumb_repo.get_webp_blobs_batch([m for m, _ in chunk])
                to_upsert: list[tuple[int, int, str, Any]] = []
                for mid, mtime in chunk:
                    if self.cancel_flag.is_set:
                        break
                    blob = blobs.get(mid)
                    if not blob:
                        continue
                    vec = self.semantic_engine.encode_image_bytes(blob)
                    if vec is not None:
                        to_upsert.append((mid, mtime, self.model_ver, vec))
                        indexed_now += 1
                    time.sleep(0.012)
                if to_upsert:
                    self.semantic_repo.upsert_vectors_batch(to_upsert)
                done = min(total, start + len(chunk))
                self.signals.progress.emit(done, total, f"Indexing semantic subjects ({done}/{total})...")

            total_cnt = self.semantic_repo.count_indexed(model_ver=self.model_ver)
            self.signals.finished_stats.emit(indexed_now, total_cnt)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            if threading.current_thread() is not threading.main_thread():
                self.db_engine.close_thread_connections()
