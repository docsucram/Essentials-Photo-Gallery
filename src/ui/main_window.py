"""
PhotoGalleryMainWindow (QMainWindow) — Central Coordinator for PhotoGallery64.
Connects DualDatabaseEngine, CatalogRepository, ThumbnailRepository, FolderSyncWorker,
ThumbnailDecodeWorker, MetadataSyncDaemon, CatalogListModel, ThumbnailGridView,
LeftNavigationDock, RibbonToolbar, TagAndCaptionDock, SingleMediaViewer, WorldMapView,
and the WLPG bottom status bar.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from PySide6.QtCore import (
    QBuffer,
    QByteArray,
    QEvent,
    QIODevice,
    QLoggingCategory,
    QMimeData,
    QObject,
    QRunnable,
    QPoint,
    QSettings,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
)

QLoggingCategory.setFilterRules(
    "qt.gui.imageio.jpeg.warning=false\n"
    "qt.gui.imageio.warning=false\n"
)
from PySide6.QtGui import (
    QColor,
    QCursor,
    QImage,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPalette,
    QPixmap,
    QShortcut,
    QShowEvent,
    QTransform,
    QUndoCommand,
    QUndoStack,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QTextEdit,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from legacy_reference.imgBrowserv13 import CancellationFlag, LruCache
from src.db.connection import DualDatabaseEngine
from src.db.repository import (
    ArrangeMode,
    CatalogRepository,
    FilterQuerySpec,
    MediaItemRecord,
    ThumbnailRepository,
)
from src.ui.batch_people_view import BatchPeopleView
from src.ui.dialogs import AppSettingsDialog, ChangeTimeTakenDialog, ResizePhotoDialog
from src.ui.grid_model import CatalogListModel
from src.ui.grid_view import MediaCardDelegate, ThumbnailGridView
from src.ui.left_nav_dock import LeftNavigationDock
from src.ui.map_view import WorldMapView
from src.ui.smart_album_dialog import SmartAlbumDialog
from src.ui.win32_frame import (
    apply_win32_frameless_styles,
    handle_win32_native_event,
    is_win32_frameless_supported,
)
from src.ui.albums_view import AlbumsView
from src.ui.compare_view import CompareMediaView
from src.ui.media_viewer import SingleMediaViewer
from src.ui.people_directory_view import PeopleDirectoryView
from src.utils.paths import get_icons_dir, get_themes_dir
from src.ui.person_header_bar import PersonHeaderBar
from src.ui.ribbon_toolbar import RibbonToolbar, make_wlpg_icon
from src.ui.right_tag_dock import TagAndCaptionDock
from src.ui.search_scope_bar import SearchScopeBar
from src.ui.slideshow_view import FullscreenSlideshowDialog
from src.ui.tag_someone_dialog import TagSomeoneDialog
import src.ui.icon_provider as icon_provider
from src.workers.scanner_thumb_worker import (
    FolderSyncWorker,
    ThumbnailDecodeWorker,
    extract_media_metadata,
)
from src.workers.face_worker import FaceDetectionWorker
from src.workers.xmp_sync_worker import MetadataSyncDaemon
from src.workers.folder_watcher import FolderWatcherDaemon
from src.utils.geotag_resolver import resolve_location_query


class _RatingUndoCommand(QUndoCommand):
    """Reversible rating modification command for QUndoStack."""

    def __init__(
        self,
        main_win: PhotoGalleryMainWindow,
        target_ids: Sequence[int],
        new_rating: int,
    ) -> None:
        super().__init__(f"Set rating to {new_rating} star{'s' if new_rating != 1 else ''}")
        self.main_win = main_win
        self.target_ids = list(target_ids)
        self.new_rating = new_rating
        field_map = self.main_win.catalog_repo.get_field_map_for_ids(self.target_ids, "rating")
        self.old_ratings = {mid: int(field_map.get(mid) or 0) for mid in self.target_ids}

    def redo(self) -> None:
        self.main_win.catalog_repo.set_rating(self.target_ids, self.new_rating)
        self.main_win.catalog_model.update_items_metadata(self.target_ids, rating=self.new_rating)
        active_ids = self.main_win.get_active_target_ids() if hasattr(self.main_win, "get_active_target_ids") else getattr(self.main_win, "_selected_ids", [])
        if any(mid in active_ids for mid in self.target_ids):
            summary = self.main_win.catalog_repo.get_selection_summary(active_ids)
            self.main_win.right_tag_dock.bind_selection(active_ids, summary)

    def undo(self) -> None:
        self.main_win.catalog_repo.restore_field_map(self.old_ratings, "rating")
        val_to_mids: dict[int, list[int]] = {}
        for mid, old_r in self.old_ratings.items():
            val_to_mids.setdefault(old_r, []).append(mid)
        for old_r, mids in val_to_mids.items():
            self.main_win.catalog_model.update_items_metadata(mids, rating=old_r)
        active_ids = self.main_win.get_active_target_ids() if hasattr(self.main_win, "get_active_target_ids") else getattr(self.main_win, "_selected_ids", [])
        if any(mid in active_ids for mid in self.target_ids):
            summary = self.main_win.catalog_repo.get_selection_summary(active_ids)
            self.main_win.right_tag_dock.bind_selection(active_ids, summary)


class _FlagUndoCommand(QUndoCommand):
    """Reversible flag modification command for QUndoStack."""

    def __init__(
        self,
        main_win: PhotoGalleryMainWindow,
        target_ids: Sequence[int],
        new_flag: int,
    ) -> None:
        super().__init__("Flag" if new_flag else "Unflag")
        self.main_win = main_win
        self.target_ids = list(target_ids)
        self.new_flag = new_flag
        field_map = self.main_win.catalog_repo.get_field_map_for_ids(self.target_ids, "flag")
        self.old_flags = {mid: int(field_map.get(mid) or 0) for mid in self.target_ids}

    def redo(self) -> None:
        self.main_win.catalog_repo.set_flag(self.target_ids, self.new_flag)
        self.main_win.catalog_model.update_items_metadata(self.target_ids, flag=self.new_flag)
        active_ids = self.main_win.get_active_target_ids() if hasattr(self.main_win, "get_active_target_ids") else getattr(self.main_win, "_selected_ids", [])
        if any(mid in active_ids for mid in self.target_ids):
            summary = self.main_win.catalog_repo.get_selection_summary(active_ids)
            self.main_win.right_tag_dock.bind_selection(active_ids, summary)
            if hasattr(self.main_win.ribbon, "set_flag_state"):
                self.main_win.ribbon.set_flag_state(bool(self.new_flag))

    def undo(self) -> None:
        self.main_win.catalog_repo.restore_field_map(self.old_flags, "flag")
        val_to_mids: dict[int, list[int]] = {}
        for mid, old_f in self.old_flags.items():
            val_to_mids.setdefault(old_f, []).append(mid)
        for old_f, mids in val_to_mids.items():
            self.main_win.catalog_model.update_items_metadata(mids, flag=old_f)
        active_ids = self.main_win.get_active_target_ids() if hasattr(self.main_win, "get_active_target_ids") else getattr(self.main_win, "_selected_ids", [])
        if any(mid in active_ids for mid in self.target_ids):
            summary = self.main_win.catalog_repo.get_selection_summary(active_ids)
            self.main_win.right_tag_dock.bind_selection(active_ids, summary)
            if hasattr(self.main_win.ribbon, "set_flag_state"):
                self.main_win.ribbon.set_flag_state(bool(summary.get("flag", 0)))


class _CaptionUndoCommand(QUndoCommand):
    """Reversible caption modification command for QUndoStack."""

    def __init__(
        self,
        main_win: PhotoGalleryMainWindow,
        target_ids: Sequence[int],
        new_caption: str,
    ) -> None:
        super().__init__("Edit caption")
        self.main_win = main_win
        self.target_ids = list(target_ids)
        self.new_caption = new_caption
        field_map = self.main_win.catalog_repo.get_field_map_for_ids(self.target_ids, "caption")
        self.old_captions = {mid: str(field_map.get(mid) or "") for mid in self.target_ids}

    def redo(self) -> None:
        self.main_win.catalog_repo.set_caption(self.target_ids, self.new_caption)
        self.main_win.catalog_model.update_items_metadata(self.target_ids, caption=self.new_caption)
        active_ids = self.main_win.get_active_target_ids() if hasattr(self.main_win, "get_active_target_ids") else getattr(self.main_win, "_selected_ids", [])
        if any(mid in active_ids for mid in self.target_ids):
            summary = self.main_win.catalog_repo.get_selection_summary(active_ids)
            self.main_win.right_tag_dock.bind_selection(active_ids, summary)

    def undo(self) -> None:
        self.main_win.catalog_repo.restore_field_map(self.old_captions, "caption")
        val_to_mids: dict[str, list[int]] = {}
        for mid, old_c in self.old_captions.items():
            val_to_mids.setdefault(old_c, []).append(mid)
        for old_c, mids in val_to_mids.items():
            self.main_win.catalog_model.update_items_metadata(mids, caption=old_c)
        active_ids = self.main_win.get_active_target_ids() if hasattr(self.main_win, "get_active_target_ids") else getattr(self.main_win, "_selected_ids", [])
        if any(mid in active_ids for mid in self.target_ids):
            summary = self.main_win.catalog_repo.get_selection_summary(active_ids)
            self.main_win.right_tag_dock.bind_selection(active_ids, summary)


class _GeotagUndoCommand(QUndoCommand):
    """Reversible geotag modification command for QUndoStack."""

    def __init__(
        self,
        main_win: PhotoGalleryMainWindow,
        target_ids: Sequence[int],
        new_lat: float | None,
        new_lon: float | None,
        new_location_name: str | None,
    ) -> None:
        super().__init__("Edit geotag")
        self.main_win = main_win
        self.target_ids = list(target_ids)
        self.new_lat = new_lat
        self.new_lon = new_lon
        self.new_location_name = new_location_name
        self.old_geotags = self.main_win.catalog_repo.get_geotag_map(self.target_ids)

    def redo(self) -> None:
        self.main_win.catalog_repo.set_geotag(
            self.target_ids, self.new_lat, self.new_lon, self.new_location_name
        )
        active_ids = self.main_win.get_active_target_ids() if hasattr(self.main_win, "get_active_target_ids") else getattr(self.main_win, "_selected_ids", [])
        if any(mid in active_ids for mid in self.target_ids):
            summary = self.main_win.catalog_repo.get_selection_summary(active_ids)
            self.main_win.right_tag_dock.bind_selection(active_ids, summary)
        if self.main_win.map_view is not None and self.main_win.central_stack.currentWidget() == self.main_win.map_view:
            self.main_win.map_view.set_viewport_bbox(
                self.main_win.map_view._min_lat,
                self.main_win.map_view._max_lat,
                self.main_win.map_view._min_lon,
                self.main_win.map_view._max_lon,
                self.main_win.catalog_repo,
            )

    def undo(self) -> None:
        self.main_win.catalog_repo.restore_geotag_map(self.old_geotags)
        active_ids = self.main_win.get_active_target_ids() if hasattr(self.main_win, "get_active_target_ids") else getattr(self.main_win, "_selected_ids", [])
        if any(mid in active_ids for mid in self.target_ids):
            summary = self.main_win.catalog_repo.get_selection_summary(active_ids)
            self.main_win.right_tag_dock.bind_selection(active_ids, summary)
        if self.main_win.map_view is not None and self.main_win.central_stack.currentWidget() == self.main_win.map_view:
            self.main_win.map_view.set_viewport_bbox(
                self.main_win.map_view._min_lat,
                self.main_win.map_view._max_lat,
                self.main_win.map_view._min_lon,
                self.main_win.map_view._max_lon,
                self.main_win.catalog_repo,
            )



class _RotateUndoCommand(QUndoCommand):
    """Reversible lossless rotation command for QUndoStack."""

    def __init__(
        self,
        main_win: PhotoGalleryMainWindow,
        target_ids: Sequence[int],
        clockwise: bool,
    ) -> None:
        super().__init__("Rotate clockwise" if clockwise else "Rotate counter-clockwise")
        self.main_win = main_win
        self.target_ids = list(target_ids)
        self.clockwise = clockwise

    def redo(self) -> None:
        self.main_win._rotate_internal(self.target_ids, clockwise=self.clockwise)

    def undo(self) -> None:
        self.main_win._rotate_internal(self.target_ids, clockwise=not self.clockwise)


class _MoveFilesUndoCommand(QUndoCommand):
    """Reversible file move command for QUndoStack (moves on disk and updates catalog.db)."""

    def __init__(
        self,
        main_win: PhotoGalleryMainWindow,
        moves: list[dict[str, Any]],
        target_folder_name: str,
    ) -> None:
        count = len(moves)
        s = "s" if count != 1 else ""
        super().__init__(f"Move {count} item{s} to {target_folder_name}")
        self.main_win = main_win
        self.moves = moves

    @staticmethod
    def _safe_move(src: str, dst: str) -> None:
        import gc
        import time
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for attempt in range(6):
            try:
                shutil.move(src, dst)
                return
            except PermissionError:
                gc.collect()
                if attempt == 5:
                    raise
                time.sleep(0.04 * (attempt + 1))

    def redo(self) -> None:
        if not self.moves:
            return
        media_ids = [int(m["media_id"]) for m in self.moves]
        backup_map: dict[int, str | None] = {}
        try:
            conn = self.main_win.engine.catalog_conn()
            q_marks = ",".join("?" for _ in media_ids)
            rows = conn.execute(
                f"SELECT id, original_backup_path FROM media WHERE id IN ({q_marks});",
                media_ids,
            ).fetchall()
            for r in rows:
                backup_map[int(r["id"])] = r["original_backup_path"]
        except Exception:
            pass

        db_batch_moves: list[tuple[int, int, str | None, str | None]] = []
        touched_dirs: set[str] = set()

        for m in self.moves:
            mid = int(m["media_id"])
            flag = self.main_win._active_thumb_flags.pop(mid, None)
            if flag is not None:
                flag.set()
            src = m["src_path"]
            dst = m["dst_path"]
            touched_dirs.add(os.path.dirname(src))
            touched_dirs.add(os.path.dirname(dst))
            if os.path.exists(src):
                self._safe_move(src, dst)

            backup_path = backup_map.get(mid)
            if not backup_path or not os.path.exists(backup_path):
                disk_b = Path(src).parent / ".OriginalFiles" / Path(src).name
                if disk_b.exists():
                    backup_path = str(disk_b)

            new_backup_str: str | None = None
            if backup_path and os.path.exists(backup_path):
                dst_backup_dir = Path(os.path.dirname(dst)) / ".OriginalFiles"
                dst_backup = dst_backup_dir / Path(backup_path).name
                try:
                    self._safe_move(backup_path, str(dst_backup))
                    m["prev_backup_path"] = backup_path
                    m["curr_backup_path"] = str(dst_backup)
                    new_backup_str = str(dst_backup)
                except OSError:
                    pass

            # Move .xmp sidecar if present
            src_p = Path(src)
            for xmp_cand in (src_p.with_name(src_p.name + ".xmp"), src_p.with_suffix(".xmp")):
                if xmp_cand.exists() and xmp_cand != src_p:
                    dst_xmp = Path(os.path.dirname(dst)) / (Path(dst).name + (xmp_cand.suffix if xmp_cand.suffix == ".xmp" else ".xmp"))
                    try:
                        self._safe_move(str(xmp_cand), str(dst_xmp))
                        m["prev_xmp_path"] = str(xmp_cand)
                        m["curr_xmp_path"] = str(dst_xmp)
                    except OSError:
                        pass
                    break

            db_batch_moves.append((mid, int(m["dst_folder_id"]), str(m["dst_filename"]), new_backup_str))
            self.main_win._media_meta_lookup.pop(mid, None)
            self.main_win.catalog_model.update_items_metadata(
                [mid], filename=m["dst_filename"]
            )

        self.main_win.catalog_repo.move_media_folder_batch(db_batch_moves)
        self._sync_moved_folder_mtimes(touched_dirs)
        self.main_win._folder_path_cache.clear()
        self.main_win.rebuild_and_run_query()

    def undo(self) -> None:
        if not self.moves:
            return
        db_batch_moves: list[tuple[int, int, str | None, str | None]] = []
        touched_dirs: set[str] = set()

        for m in self.moves:
            mid = int(m["media_id"])
            flag = self.main_win._active_thumb_flags.pop(mid, None)
            if flag is not None:
                flag.set()
            src = m["dst_path"]
            dst = m["src_path"]
            touched_dirs.add(os.path.dirname(src))
            touched_dirs.add(os.path.dirname(dst))
            if os.path.exists(src):
                self._safe_move(src, dst)

            curr_b = m.get("curr_backup_path")
            prev_b = m.get("prev_backup_path")
            restored_backup_str: str | None = None
            if curr_b and prev_b and os.path.exists(curr_b):
                try:
                    self._safe_move(curr_b, prev_b)
                    restored_backup_str = prev_b
                except OSError:
                    pass

            curr_xmp = m.get("curr_xmp_path")
            prev_xmp = m.get("prev_xmp_path")
            if curr_xmp and prev_xmp and os.path.exists(curr_xmp):
                try:
                    self._safe_move(curr_xmp, prev_xmp)
                except OSError:
                    pass

            db_batch_moves.append((mid, int(m["src_folder_id"]), str(m["src_filename"]), restored_backup_str))
            self.main_win._media_meta_lookup.pop(mid, None)
            self.main_win.catalog_model.update_items_metadata(
                [mid], filename=m["src_filename"]
            )

        self.main_win.catalog_repo.move_media_folder_batch(db_batch_moves)
        self._sync_moved_folder_mtimes(touched_dirs)
        self.main_win._folder_path_cache.clear()
        self.main_win.rebuild_and_run_query()

    def _sync_moved_folder_mtimes(self, touched_dirs: set[str]) -> None:
        """Updates folder mtimes in catalog.db and clears pending FolderWatcher triggers so internal moves don't trigger redundant disk rescans."""
        try:
            watcher = getattr(self.main_win, "folder_watcher", None)
            with self.main_win.engine.write_lock:
                conn = self.main_win.engine.catalog_conn()
                for dpath in touched_dirs:
                    if not dpath:
                        continue
                    try:
                        p = Path(dpath).resolve()
                        if p.exists():
                            st = p.stat()
                            posix_p = p.as_posix().rstrip("/")
                            conn.execute(
                                "UPDATE folders SET mtime = ? WHERE RTRIM(REPLACE(LOWER(path), '\\', '/'), '/') = LOWER(?);",
                                (float(st.st_mtime), posix_p),
                            )
                        if watcher is not None and hasattr(watcher, "_pending_changed_paths"):
                            watcher._pending_changed_paths.discard(str(p))
                            watcher._pending_changed_paths.discard(dpath)
                    except OSError:
                        pass
                conn.commit()
        except Exception:
            pass


class _AvatarBackfillSignals(QObject):
    finished = Signal(int)


class _AvatarBackfillWorker(QRunnable):
    """Background worker that generates missing avatar crops in people table."""

    def __init__(self, engine: DualDatabaseEngine, signals: _AvatarBackfillSignals) -> None:
        super().__init__()
        self.engine = engine
        self.signals = signals

    def run(self) -> None:
        try:
            repo = CatalogRepository(self.engine)
            updated = repo.backfill_missing_people_thumbnails()
            if updated > 0:
                self.signals.finished.emit(updated)
        except Exception:
            pass
        finally:
            if threading.current_thread() is not threading.main_thread():
                self.engine.close_thread_connections()


class _FaceCropPreloadSignals(QObject):
    crop_ready = Signal(int, int, bytes)  # media_id, person_id, webp_blob
    progress = Signal(int, int)          # current, total
    finished = Signal()


class _FaceCropPreloadWorker(QRunnable):
    """
    Fast background worker that extracts and persists missing face thumbnails for a person.
    Uses lightweight Pillow region cropping (~2ms per image) without running heavy ONNX models.
    """

    def __init__(
        self,
        engine: DualDatabaseEngine,
        person_id: int,
        signals: _FaceCropPreloadSignals,
        cancel_flag: CancellationFlag,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.person_id = person_id
        self.signals = signals
        self.cancel_flag = cancel_flag

    def run(self) -> None:
        try:
            cat_conn = self.engine.catalog_conn()
            rows = cat_conn.execute(
                """
                SELECT mp.id AS face_id, mp.media_id, mp.is_manual,
                       mp.rect_x, mp.rect_y, mp.rect_w, mp.rect_h,
                       m.filename, f.path AS folder_path
                FROM media_people mp
                JOIN media m ON m.id = mp.media_id
                JOIN folders f ON f.id = m.folder_id
                WHERE mp.person_id = ? AND mp.state = 1
                ORDER BY m.date_taken DESC, mp.id ASC;
                """,
                (self.person_id,),
            ).fetchall()
            if not rows or self.cancel_flag.is_set():
                self.signals.finished.emit()
                return

            thumb_repo = ThumbnailRepository(self.engine)
            existing_face_ids: set[int] = set()
            try:
                t_conn = self.engine.thumb_conn()
                f_rows = t_conn.execute("SELECT face_id FROM face_thumbnails;").fetchall()
                existing_face_ids = {int(r[0]) for r in f_rows}
            except Exception:
                pass

            missing = [
                r for r in rows
                if int(r["face_id"]) not in existing_face_ids or bool(r["is_manual"])
            ]
            total = len(missing)
            if total == 0:
                self.signals.finished.emit()
                return

            for idx, r in enumerate(missing, start=1):
                if self.cancel_flag.is_set():
                    break
                fid = int(r["face_id"])
                mid = int(r["media_id"])
                rect = (
                    float(r["rect_x"] or 0.0),
                    float(r["rect_y"] or 0.0),
                    float(r["rect_w"] or 0.0),
                    float(r["rect_h"] or 0.0),
                )
                full_path = os.path.join(str(r["folder_path"]), str(r["filename"]))
                blob = CatalogRepository.extract_face_crop_blob(full_path, rect, target_size=128)
                if blob:
                    thumb_repo.put_face_thumbnail(fid, blob)
                    self.signals.crop_ready.emit(mid, self.person_id, blob)
                self.signals.progress.emit(idx, total)
        except Exception:
            pass
        finally:
            self.signals.finished.emit()
            if threading.current_thread() is not threading.main_thread():
                self.engine.close_thread_connections()


class _FramelessResizeFilter(QObject):
    """
    Handles native edge margin resizing on Windows / desktop OS for frameless window.
    """

    def __init__(self, window: QMainWindow, margin: int = 6) -> None:
        super().__init__(window)
        self.window = window
        self.margin = margin
        self._cursor_set: bool = False

    def _get_edge(self, pos: QPoint) -> Qt.Edge | None:
        if self.window.isMaximized():
            return None
        rect = self.window.rect()
        x, y = pos.x(), pos.y()
        w, h = rect.width(), rect.height()
        m = self.margin

        if x < 0 or y < 0 or x > w or y > h:
            return None

        on_left = x <= m
        on_right = x >= w - m
        on_top = y <= m
        on_bottom = y >= h - m

        edges = Qt.Edge(0)
        if on_top:
            edges |= Qt.Edge.TopEdge
        elif on_bottom:
            edges |= Qt.Edge.BottomEdge
        if on_left:
            edges |= Qt.Edge.LeftEdge
        elif on_right:
            edges |= Qt.Edge.RightEdge

        return edges if edges else None

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        t = event.type()
        if t in (QEvent.MouseMove, QEvent.MouseButtonPress):
            global_pos = getattr(event, "globalPosition", None)
            if global_pos is not None:
                g_pt = global_pos().toPoint()
            else:
                g_pt = QCursor.pos()

            local_pos = self.window.mapFromGlobal(g_pt)
            edge = self._get_edge(local_pos)

            if t == QEvent.MouseMove:
                if edge is not None:
                    if edge in (Qt.TopEdge | Qt.LeftEdge, Qt.BottomEdge | Qt.RightEdge):
                        self.window.setCursor(Qt.SizeFDiagCursor)
                    elif edge in (Qt.TopEdge | Qt.RightEdge, Qt.BottomEdge | Qt.LeftEdge):
                        self.window.setCursor(Qt.SizeBDiagCursor)
                    elif edge in (Qt.LeftEdge, Qt.RightEdge):
                        self.window.setCursor(Qt.SizeHorCursor)
                    elif edge in (Qt.TopEdge, Qt.BottomEdge):
                        self.window.setCursor(Qt.SizeVerCursor)
                    self._cursor_set = True
                elif self._cursor_set:
                    self.window.unsetCursor()
                    self._cursor_set = False

            elif t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                if edge is not None:
                    handle = self.window.windowHandle()
                    if handle is not None:
                        handle.startSystemResize(edge)
                        return True

        return super().eventFilter(obj, event)


class PhotoGalleryMainWindow(QMainWindow):
    """
    Windows Live Photo Gallery 64-Bit Clone Main Window (`docs/reference_ui/main_ui.png`).
    Enforces strict thread-pool boundaries: QPixmap is instantiated exclusively on the GUI thread.
    """

    def __init__(
        self,
        engine: DualDatabaseEngine,
        start_xmp_daemon: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setWindowTitle("Essentials Photo Gallery")
        ico_path = Path(__file__).resolve().parent.parent.parent / "resources" / "icons" / "essentials_photo_gallery.ico"
        if not ico_path.exists():
            ico_path = Path(__file__).resolve().parent.parent.parent / "resources" / "icons" / "photogallery_live.ico"
        if ico_path.exists():
            from PySide6.QtGui import QIcon
            self.setWindowIcon(QIcon(str(ico_path)))
        self.resize(1360, 840)
        self.setMinimumSize(500, 400)

        self._resize_filter = None
        if not is_win32_frameless_supported():
            self._resize_filter = _FramelessResizeFilter(self)
            app = QApplication.instance()
            if app:
                app.installEventFilter(self._resize_filter)

        self.engine = engine
        self.catalog_repo = CatalogRepository(engine)
        self.repo = self.catalog_repo
        self.thumb_repo = ThumbnailRepository(engine)
        try:
            self.thumb_repo.purge_mismatched_orientation_thumbnails()
        except Exception:
            pass
        self.pixmap_cache = LruCache(2500)

        self.retention_days: int = 30
        self.find_people_enabled: bool = True
        self.current_theme: Literal["wlpg_aero", "modern", "dark"] = "wlpg_aero"
        self._arrange_by: ArrangeMode = "date"
        self._reverse_sort: bool = False
        self._group_by: str = "month"
        self._group_reverse: bool = False
        self._sort_by: str = "date"
        self._sort_reverse: bool = False
        self._search_text: str = ""
        self._active_bbox: tuple[float, float, float, float] | None = None
        self._active_map_media_ids: list[int] | None = None
        self._active_map_location_label: str | None = None
        self._active_special_people_filter: str | None = None
        self._active_special_people_label: str | None = None
        self._search_scope_bar_explicitly_opened: bool = False
        self._selected_ids: list[int] = []
        self._lead_selected_id: int | None = None
        self._mru_descriptive_tags: list[str] = self._load_saved_mru_tags()
        self._is_shutting_down: bool = False
        self.write_back_paused: bool = True
        self._workers_paused: bool = False
        self.current_icon_theme: str = "wlpg_aero"
        self._scanner_status_text: str = "Scanning: Idle"
        self._face_status_text: str = "Faces: Idle"
        self._face_phase: str = "idle"
        self._face_crop_status_text: str = ""
        self._status_msg_seq: int = 0
        self._face_crop_cancel = CancellationFlag()

        # Undo / Redo Command Stack
        self.undo_stack = QUndoStack(self)

        # Navigation History Stack
        self._nav_history: list[dict[str, Any]] = []
        self._nav_index: int = -1
        self._is_navigating_history: bool = False

        # Dedicated Thread Pools & Cancellation Flags
        self.scanner_pool = QThreadPool(self)
        self.scanner_pool.setMaxThreadCount(1)
        self._scanner_cancel = CancellationFlag()

        self.thumbnail_pool = QThreadPool(self)
        self.thumbnail_pool.setMaxThreadCount(min(8, os.cpu_count() or 4))
        self._active_thumb_flags: dict[int, CancellationFlag] = {}
        self._media_meta_lookup: dict[int, tuple[str, int, bool, int]] = {}
        self._failed_thumb_ids: set[int] = set()
        self._pending_visible_media_ids: list[int] = []

        self.face_pool = QThreadPool(self)
        self.face_pool.setMaxThreadCount(1)
        self.priority_face_pool = QThreadPool(self)
        self.priority_face_pool.setMaxThreadCount(1)
        self._face_cancel = CancellationFlag()
        self._face_worker: FaceDetectionWorker | None = None

        # Background XMP / EXIF Write-Back Daemon
        self.xmp_daemon = MetadataSyncDaemon(engine, poll_interval_ms=1500, parent=self)
        self.xmp_daemon.set_paused(self.write_back_paused)
        self.xmp_daemon.batch_synced.connect(self.on_metadata_batch_synced)
        self.xmp_daemon.sync_error.connect(
            lambda mid, err: self.on_worker_error(f"XMP sync error ({mid}): {err}")
        )
        if start_xmp_daemon:
            self.xmp_daemon.start()
        self.metadata_sync_daemon = self.xmp_daemon

        # Real-Time Continuous Directory Watcher
        self.folder_watcher = FolderWatcherDaemon(engine, debounce_interval_ms=750, parent=self)
        self.folder_watcher.delta_completed.connect(self._on_folder_watcher_delta)
        self.folder_watcher.reconciliation_completed.connect(self._on_startup_reconciliation_completed)
        self.folder_watcher.update_watched_paths()

        # ---------------------------------------------------------------------
        # Build UI Hierarchy
        # ---------------------------------------------------------------------
        # Pre-initialize Qt Quick RHI backing surface before window is shown so
        # creating QWebEngineView later in WorldMapView never recreates the native HWND
        if "PYTEST_CURRENT_TEST" not in os.environ and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            try:
                from PySide6.QtQuickWidgets import QQuickWidget
                self._rhi_surface_anchor = QQuickWidget(self)
                self._rhi_surface_anchor.setFixedSize(0, 0)
            except Exception:
                pass

        # 1. Top Ribbon Toolbar spanning the full top width of the window
        self.ribbon = RibbonToolbar(self)
        self.setMenuWidget(self.ribbon)
        self.undo_stack.canUndoChanged.connect(self.ribbon.set_undo_enabled)
        self.undo_stack.canRedoChanged.connect(self.ribbon.set_redo_enabled)

        # 2. Central QStackedWidget (0: Grid, 1: Single Viewer, 2: World Map)
        self.central_stack = QStackedWidget(self)
        self.central_stack.setFrameShape(QFrame.NoFrame)

        self.catalog_model = CatalogListModel(self.pixmap_cache, self)
        self.card_delegate = MediaCardDelegate(self.pixmap_cache, self)
        self.card_delegate.thumb_repo = self.thumb_repo
        self.grid_view = ThumbnailGridView(
            self.catalog_model, self.card_delegate, self.central_stack
        )
        self.grid_view.main_window = self
        self.grid_view.catalog_repo = self.catalog_repo
        self.grid_view.thumb_repo = self.thumb_repo

        # Person Header Bar (compact Google Photos / Immich style header with view toggle & suggestions banner)
        self.person_header_bar = PersonHeaderBar(self.engine, self.grid_view)
        self.person_header_bar.hide()
        self.grid_view.set_person_header_bar(self.person_header_bar)
        self.person_header_bar.back_to_directory_requested.connect(self.open_people_directory_view)
        self.person_header_bar.review_suggestions_requested.connect(lambda pid: self.open_batch_people_view(person_id=pid))
        self.person_header_bar.open_media_requested.connect(self.open_single_viewer)
        self.person_header_bar.view_mode_changed.connect(self._on_person_view_mode_changed)
        self.person_header_bar.person_renamed.connect(
            lambda pid, name: (self.refresh_navigation_and_autocomplete(), self.left_nav_dock.select_person(pid))
        )

        self.scope_bar = SearchScopeBar(self.grid_view)
        self.scope_bar.setFixedHeight(62)
        self.scope_bar.hide()
        self.grid_view.set_search_scope_bar(self.scope_bar)
        self.grid_view.try_search_all_requested.connect(self._on_search_all_requested)
        self.ribbon.remove_embedded_scope_bar()
        self.ribbon.scope_bar = self.scope_bar
        self.ribbon.txt_search = self.scope_bar.txt_search
        self.central_stack.addWidget(self.grid_view)

        self.media_viewer = SingleMediaViewer(self.central_stack)
        self.media_viewer.main_window = self
        self.media_viewer.catalog_repo = self.catalog_repo
        self.media_viewer.repo = self.catalog_repo
        self.central_stack.addWidget(self.media_viewer)

        self._map_view: WorldMapView | None = None

        self.batch_people_view = BatchPeopleView(self.engine, self.central_stack)
        self.batch_people_view.faces_confirmed.connect(self._on_batch_faces_confirmed)
        self.batch_people_view.faces_rejected.connect(self._on_batch_faces_rejected)
        self.batch_people_view.faces_ignored.connect(self._on_batch_faces_ignored)
        self.batch_people_view.person_renamed.connect(self._on_batch_person_renamed)
        self.batch_people_view.back_to_gallery_requested.connect(self.close_batch_people_view)
        self.batch_people_view.open_media_requested.connect(self.open_single_viewer)
        self.batch_people_view.card_selection_changed.connect(self._on_batch_card_selection_changed)
        QShortcut(QKeySequence(Qt.Key_Escape), self.batch_people_view, self.close_batch_people_view)
        self.central_stack.addWidget(self.batch_people_view)

        # Immich / Google Photos Inspired People Directory View
        self.people_directory_view = PeopleDirectoryView(self.engine, self.central_stack)
        self.people_directory_view.person_selected.connect(self._on_directory_person_selected)
        self.people_directory_view.unnamed_selected.connect(self._on_directory_unnamed_selected)
        self.people_directory_view.review_suggestions_requested.connect(self.open_batch_people_view)
        self.people_directory_view.directory_updated.connect(self.refresh_people_navigation_and_autocomplete)
        self.person_header_bar.unnamed_person_named.connect(self._on_unnamed_person_named)
        self.central_stack.addWidget(self.people_directory_view)

        # Visual Albums & Smart Albums Directory View
        self.albums_view = AlbumsView(self.engine, self.central_stack)
        self.albums_view.album_selected.connect(self._on_albums_view_album_selected)
        self.albums_view.smart_album_selected.connect(self._on_albums_view_smart_album_selected)
        self.albums_view.new_album_requested.connect(self._on_new_album_requested)
        self.albums_view.rename_album_requested.connect(self._on_rename_album_requested)
        self.albums_view.delete_album_requested.connect(self._on_delete_album_requested)
        self.albums_view.new_smart_album_requested.connect(self._on_new_smart_album_requested)
        self.albums_view.edit_smart_album_requested.connect(self._on_edit_smart_album_requested)
        self.albums_view.rename_smart_album_requested.connect(self._on_rename_smart_album_requested)
        self.albums_view.delete_smart_album_requested.connect(self._on_delete_smart_album_requested)
        self.central_stack.addWidget(self.albums_view)

        # Integrated Duplicate & Burst Finder View (like World Map & Batch People Tag)
        from src.ui.duplicate_finder_dialog import DuplicateFinderView
        self.duplicate_finder_view = DuplicateFinderView(
            engine=self.engine,
            parent=self.central_stack,
            is_dark=getattr(self, "_dark_mode", False),
        )
        self.duplicate_finder_view.back_to_gallery_requested.connect(self.show_grid_view)
        self.duplicate_finder_view.duplicates_cleaned.connect(self._on_duplicates_cleaned)
        self.central_stack.addWidget(self.duplicate_finder_view)

        # Integrated Multi-Photo Compare / Light Table View
        self.compare_view = CompareMediaView(self.central_stack)
        self.compare_view.bind_repositories(self.catalog_repo, self.thumb_repo)
        self.compare_view.close_requested.connect(self._on_compare_closed)
        self.compare_view.rating_changed.connect(lambda _mid, _r: self.rebuild_and_run_query())
        self.compare_view.flag_toggled.connect(lambda _mid, _f: self.rebuild_and_run_query())
        self.compare_view.media_deleted.connect(lambda _mid: self.rebuild_and_run_query())
        self.compare_view.open_in_editor_requested.connect(self._on_compare_open_editor)
        self.central_stack.addWidget(self.compare_view)
        self.grid_view.compare_requested.connect(self.open_compare_view)

        self.setCentralWidget(self.central_stack)

        # 3. Left Navigation Dock & Right Tag/Caption Dock
        self.left_nav_dock = LeftNavigationDock(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_nav_dock)
        self.resizeDocks([self.left_nav_dock], [230], Qt.Horizontal)

        self.right_tag_dock = TagAndCaptionDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_tag_dock)

        # 4. WLPG Bottom Status Bar
        self._build_status_bar()

        # 5. Wire Qt Signals & Slots
        self._connect_signals()

        # 6. Apply WLPG Classic Aero Theme & Initial Refresh
        self.apply_theme("wlpg_aero")
        self._start_async_avatar_backfill()
        self.refresh_navigation_and_autocomplete()
        self.rebuild_and_run_query()

        # Seed initial navigation state into history stack
        self._nav_history = [self.left_nav_dock.current_filter_state()]
        self._nav_index = 0
        self._update_nav_buttons()

        # 7. Restore persisted application state (folder, view, docks, geometry)
        self._restore_app_state()

        # 8. Launch low-priority background startup reconciliation for offline filesystem changes
        QTimer.singleShot(150, self.folder_watcher.start_startup_reconciliation)

        # 9. Start live status bar worker monitor timer (production safeguard 2)
        self._monitor_timer = QTimer(self)
        self._monitor_timer.setInterval(1200)
        self._monitor_timer.timeout.connect(self._update_worker_monitor)
        self._monitor_timer.start()

        # 10. Install global event filter for mouse Back / Forward thumb buttons
        app = QApplication.instance()
        if app:
            app.installEventFilter(self)

    def _build_status_bar(self) -> None:
        sb = self.statusBar()
        sb.setFixedHeight(28)
        sb.setStyleSheet(
            """
            QStatusBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F0F4FA, stop:1 #D9E4F2);
                border-top: 1px solid #B8CDE3;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
            }
            QStatusBar QPushButton {
                border: 1px solid transparent;
                background: transparent;
                border-radius: 2px;
                min-width: 22px;
                max-width: 26px;
                height: 20px;
            }
            QStatusBar QPushButton:hover {
                border: 1px solid #7DA2CE;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FFFFFF, stop:0.5 #E7F1FB, stop:1 #D0E4F9);
            }
            QStatusBar QPushButton:pressed {
                border: 1px solid #5A85B8;
                background: #BFDCF7;
            }
            QStatusBar QSlider::groove:horizontal {
                height: 3px;
                background: #B6C8DA;
                border-radius: 1px;
            }
            QStatusBar QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F0F6FA, stop:1 #D8E5F0);
                border: 1px solid #7592B5;
                width: 6px;
                height: 12px;
                margin: -5px 0;
                border-radius: 2px;
            }
            QStatusBar QSlider::handle:horizontal:hover {
                border: 1px solid #3380CC;
                background: #E5F1FC;
            }
            """
        )

        self.lbl_status_counts = QLabel("0 items", sb)
        self.lbl_status_counts.setStyleSheet("padding-left: 6px; font-weight: bold;")
        sb.addWidget(self.lbl_status_counts)

        # Worker Monitor consolidated widget right next to item count
        self.worker_monitor_widget = QWidget(sb)
        mon_hbox = QHBoxLayout(self.worker_monitor_widget)
        mon_hbox.setContentsMargins(4, 0, 4, 0)
        mon_hbox.setSpacing(4)

        # Subtle vertical separator between item count and worker status
        sep_w = QFrame(self.worker_monitor_widget)
        sep_w.setFrameShape(QFrame.VLine)
        sep_w.setStyleSheet("color: #AFC3D8; margin: 3px 2px;")
        mon_hbox.addWidget(sep_w)

        self.btn_pause_workers = QPushButton("⏸", self.worker_monitor_widget)
        self.btn_pause_workers.setCursor(Qt.PointingHandCursor)
        self.btn_pause_workers.setToolTip("Pause background indexing, face detection, and file write-back")
        self.btn_pause_workers.setStyleSheet(
            """
            QPushButton {
                font-family: 'Segoe UI Symbol', 'Segoe UI', sans-serif;
                font-size: 11pt;
                border: none;
                background: transparent;
                color: #1E395B;
                padding: 0;
                min-width: 18px;
                max-width: 20px;
                height: 18px;
            }
            QPushButton:hover {
                color: #0078D7;
            }
            """
        )
        self.btn_pause_workers.clicked.connect(self.toggle_pause_all_workers)
        mon_hbox.addWidget(self.btn_pause_workers)

        self.lbl_worker_monitor = QLabel("Idle", self.worker_monitor_widget)
        self.lbl_worker_monitor.setCursor(Qt.PointingHandCursor)
        self.lbl_worker_monitor.setToolTip("Background tasks status\nClick for details")
        self.lbl_worker_monitor.setStyleSheet(
            """
            QLabel {
                color: #5A738E;
                font-size: 8.5pt;
                padding: 1px 4px;
                border-radius: 3px;
            }
            QLabel:hover {
                color: #0078D7;
                background: #E5F1FC;
            }
            """
        )
        self.lbl_worker_monitor.mousePressEvent = lambda e: self._show_worker_details_menu(e)
        mon_hbox.addWidget(self.lbl_worker_monitor)

        sb.addWidget(self.worker_monitor_widget)

        # Subtle vertical separator before transient status message
        self.sep_status_msg = QFrame(sb)
        self.sep_status_msg.setFrameShape(QFrame.VLine)
        self.sep_status_msg.setStyleSheet("color: #AFC3D8; margin: 3px 6px;")
        self.sep_status_msg.setVisible(False)
        sb.addWidget(self.sep_status_msg)

        self.lbl_sync_status = QLabel("", sb)
        self.lbl_sync_status.setStyleSheet("color: #0066CC; padding-left: 4px;")
        sb.addWidget(self.lbl_sync_status, 1)

        # Override statusBar().showMessage and clearMessage so temporary notifications
        # display in lbl_sync_status without ever hiding item counts or worker monitor controls
        sb.showMessage = self.show_status_message
        sb.clearMessage = lambda: self.show_status_message("")

        # Right-aligned WLPG Bottom Bar Controls matching media_1792165701201_517fc7f1.png
        right_ctrl = QWidget(sb)
        right_ctrl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        r_hbox = QHBoxLayout(right_ctrl)
        r_hbox.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        r_hbox.setContentsMargins(0, 0, 6, 0)
        r_hbox.setSpacing(2)

        self.btn_status_prev = QPushButton(right_ctrl)
        self.btn_status_prev.setIcon(make_wlpg_icon("prev_media", 18))
        self.btn_status_prev.setIconSize(QSize(18, 18))
        self.btn_status_prev.setFixedSize(22, 22)
        self.btn_status_prev.setToolTip("Previous (Left Arrow)")
        self.btn_status_prev.clicked.connect(lambda: self.navigate_single_viewer(-1))
        self.btn_status_prev.setVisible(False)
        r_hbox.addWidget(self.btn_status_prev)

        self.btn_status_next = QPushButton(right_ctrl)
        self.btn_status_next.setIcon(make_wlpg_icon("next_media", 18))
        self.btn_status_next.setIconSize(QSize(18, 18))
        self.btn_status_next.setFixedSize(22, 22)
        self.btn_status_next.setToolTip("Next (Right Arrow)")
        self.btn_status_next.clicked.connect(lambda: self.navigate_single_viewer(1))
        self.btn_status_next.setVisible(False)
        r_hbox.addWidget(self.btn_status_next)

        self.btn_status_rot_ccw = QPushButton(right_ctrl)
        self.btn_status_rot_ccw.setIcon(make_wlpg_icon("rotate_ccw", 18))
        self.btn_status_rot_ccw.setIconSize(QSize(18, 18))
        self.btn_status_rot_ccw.setFixedSize(22, 22)
        self.btn_status_rot_ccw.setToolTip("Rotate left (Ctrl+,)")
        self.btn_status_rot_ccw.clicked.connect(lambda: self.rotate_selection(clockwise=False))
        r_hbox.addWidget(self.btn_status_rot_ccw)

        self.btn_status_rot_cw = QPushButton(right_ctrl)
        self.btn_status_rot_cw.setIcon(make_wlpg_icon("rotate_cw", 18))
        self.btn_status_rot_cw.setIconSize(QSize(18, 18))
        self.btn_status_rot_cw.setFixedSize(22, 22)
        self.btn_status_rot_cw.setToolTip("Rotate right (Ctrl+.)")
        self.btn_status_rot_cw.clicked.connect(lambda: self.rotate_selection(clockwise=True))
        r_hbox.addWidget(self.btn_status_rot_cw)

        self.btn_status_delete = QPushButton(right_ctrl)
        self.btn_status_delete.setIcon(make_wlpg_icon("delete", 16))
        self.btn_status_delete.setIconSize(QSize(16, 16))
        self.btn_status_delete.setFixedSize(22, 22)
        self.btn_status_delete.setToolTip("Delete selected items to Recycle Bin (Del)")
        self.btn_status_delete.clicked.connect(self.delete_selection_to_recycle_bin)
        r_hbox.addWidget(self.btn_status_delete)

        self.btn_status_slideshow = QPushButton(right_ctrl)
        self.btn_status_slideshow.setIcon(make_wlpg_icon("slideshow", 18))
        self.btn_status_slideshow.setIconSize(QSize(18, 18))
        self.btn_status_slideshow.setFixedSize(22, 22)
        self.btn_status_slideshow.setToolTip("Slide show (F12)")
        self.btn_status_slideshow.clicked.connect(lambda: self.dispatch_ribbon_action("slide_show"))
        r_hbox.addWidget(self.btn_status_slideshow)

        # Subtle vertical separator
        sep = QFrame(right_ctrl)
        sep.setFrameShape(QFrame.VLine)
        sep.setFixedSize(1, 16)
        sep.setStyleSheet("color: #AFC3D8; margin: 3px 2px;")
        r_hbox.addWidget(sep)

        self.btn_status_grid = QPushButton(right_ctrl)
        self.btn_status_grid.setIcon(make_wlpg_icon("view_details", 18))
        self.btn_status_grid.setIconSize(QSize(18, 18))
        self.btn_status_grid.setFixedSize(22, 22)
        self.btn_status_grid.setToolTip("View Details")
        self.btn_status_grid.clicked.connect(self._on_status_grid_or_details_clicked)
        r_hbox.addWidget(self.btn_status_grid)

        self.btn_status_fit = QPushButton(right_ctrl)
        self.btn_status_fit.setIcon(make_wlpg_icon("actual_size", 18))
        self.btn_status_fit.setIconSize(QSize(18, 18))
        self.btn_status_fit.setFixedSize(22, 22)
        self.btn_status_fit.setToolTip("Actual size (Ctrl+1)")
        self.btn_status_fit.clicked.connect(self._on_status_zoom_toggle_clicked)
        self.btn_status_fit.setVisible(False)
        r_hbox.addWidget(self.btn_status_fit)

        # Alias for backward compatibility with existing tests
        self.btn_status_actual = self.btn_status_fit

        # Legacy alias for test compatibility
        self.btn_status_viewer = QPushButton(right_ctrl)
        self.btn_status_viewer.setVisible(False)
        self.btn_status_viewer.clicked.connect(self._on_status_viewer_clicked)

        self.btn_status_zoom_out = QPushButton(right_ctrl)
        self.btn_status_zoom_out.setIcon(make_wlpg_icon("zoom_out", 18))
        self.btn_status_zoom_out.setIconSize(QSize(18, 18))
        self.btn_status_zoom_out.setFixedSize(22, 22)
        self.btn_status_zoom_out.setToolTip("Zoom out (-)")
        self.btn_status_zoom_out.clicked.connect(lambda: self.dispatch_ribbon_action("zoom_out"))
        r_hbox.addWidget(self.btn_status_zoom_out)

        self.slider_status_zoom = QSlider(Qt.Horizontal, right_ctrl)
        self.slider_status_zoom.setRange(72, 320)
        self.slider_status_zoom.setValue(150)
        self.slider_status_zoom.setFixedWidth(95)
        self.slider_status_zoom.setToolTip("Thumbnail / Viewer zoom")
        self.slider_status_zoom.valueChanged.connect(self._on_status_zoom_changed)
        r_hbox.addWidget(self.slider_status_zoom)

        self.btn_status_zoom_in = QPushButton(right_ctrl)
        self.btn_status_zoom_in.setIcon(make_wlpg_icon("zoom_in", 18))
        self.btn_status_zoom_in.setIconSize(QSize(18, 18))
        self.btn_status_zoom_in.setFixedSize(22, 22)
        self.btn_status_zoom_in.setToolTip("Zoom in (+)")
        self.btn_status_zoom_in.clicked.connect(lambda: self.dispatch_ribbon_action("zoom_in"))
        r_hbox.addWidget(self.btn_status_zoom_in)

        for sb_btn in (
            self.btn_status_prev,
            self.btn_status_next,
            self.btn_status_rot_ccw,
            self.btn_status_rot_cw,
            self.btn_status_delete,
            self.btn_status_slideshow,
            self.btn_status_grid,
            self.btn_status_fit,
            self.btn_status_zoom_out,
            self.slider_status_zoom,
            self.btn_status_zoom_in,
        ):
            sb_btn.setFocusPolicy(Qt.NoFocus)

        sb.addPermanentWidget(right_ctrl)

    def _connect_signals(self) -> None:
        # Left Navigation Dock
        self.left_nav_dock.filter_changed.connect(self._on_nav_filter_changed)
        self.left_nav_dock.people_directory_requested.connect(self.open_people_directory_view)
        self.left_nav_dock.albums_view_requested.connect(self.open_albums_view)
        self.left_nav_dock.section_order_changed.connect(self._on_nav_section_order_changed)
        self.left_nav_dock.folder_add_requested.connect(self._prompt_and_add_folder)
        self.left_nav_dock.folder_remove_requested.connect(self._on_remove_folder_from_library)
        self.left_nav_dock.folder_renamed.connect(self._on_folder_renamed)
        self.left_nav_dock.folder_created.connect(self._on_folder_created)
        self.left_nav_dock.folder_deleted.connect(self._on_folder_deleted)
        self.left_nav_dock.favourite_path_selected.connect(
            lambda path: self.scan_folder(path, include_subfolders=True)
        )
        self.left_nav_dock.files_dropped_on_folder.connect(self._on_files_dropped_on_folder)
        self.left_nav_dock.tag_hierarchy_changed.connect(self.refresh_navigation_and_autocomplete)
        self.left_nav_dock.place_selected.connect(self._on_place_selected)
        self.left_nav_dock.new_album_requested.connect(self._on_new_album_requested)
        self.left_nav_dock.rename_album_requested.connect(self._on_rename_album_requested)
        self.left_nav_dock.delete_album_requested.connect(self._on_delete_album_requested)
        self.left_nav_dock.new_smart_album_requested.connect(self._on_new_smart_album_requested)
        self.left_nav_dock.edit_smart_album_requested.connect(self._on_edit_smart_album_requested)
        self.left_nav_dock.rename_smart_album_requested.connect(self._on_rename_smart_album_requested)
        self.left_nav_dock.delete_smart_album_requested.connect(self._on_delete_smart_album_requested)
        self.left_nav_dock.save_filter_as_smart_album_requested.connect(
            self._on_save_filter_as_smart_album_requested
        )

        # Top Ribbon Toolbar
        self.ribbon.arrange_changed.connect(self.on_arrange_changed)
        self.ribbon.details_overlay_changed.connect(self._on_details_overlay_changed)
        self.ribbon.person_quick_find_selected.connect(self.on_quick_find_person)
        self.ribbon.search_text_changed.connect(self._on_search_text_changed)
        self.ribbon.search_all_requested.connect(self.left_nav_dock.clear_all_filters)
        self.ribbon.action_triggered.connect(self.dispatch_ribbon_action)
        self.ribbon.layout_mode_changed.connect(self._on_layout_mode_changed)
        self.ribbon.zoom_size_changed.connect(self._on_ribbon_zoom_changed)
        if hasattr(self.ribbon, "tab_widget"):
            self.ribbon.tab_widget.currentChanged.connect(self._on_ribbon_tab_changed)
        self.central_stack.currentChanged.connect(lambda _: self._update_status_grid_button())
        icon_provider.signals.theme_changed.connect(self._on_icon_theme_changed)

        # Authentic WLPG Find & Home Tab Filter Signal Connections
        self.ribbon.date_filter_selected.connect(self._on_date_filter_selected)
        self.ribbon.rating_filter_selected.connect(self._on_rating_filter_selected)
        self.ribbon.tag_filter_selected.connect(self._on_ribbon_tag_filter)
        self.ribbon.untagged_filter_selected.connect(
            self.left_nav_dock.set_untagged_only
        )
        self.ribbon.flagged_filter_selected.connect(
            self.left_nav_dock.set_flagged_only
        )
        self.ribbon.flag_filter_selected.connect(
            self.left_nav_dock.set_flag_filter
        )
        self.ribbon.edited_filter_selected.connect(
            self.left_nav_dock.set_edited_only
        )
        self.ribbon.edited_filter_mode_selected.connect(
            self.left_nav_dock.set_edited_filter
        )
        self.ribbon.orientation_filter_selected.connect(
            self.left_nav_dock.set_orientation_filter
        )
        self.ribbon.media_type_filter_selected.connect(
            self.left_nav_dock.set_media_type
        )
        self.ribbon.person_boolean_mode_changed.connect(
            self.left_nav_dock.set_person_boolean_mode
        )
        self.ribbon.people_special_filter_selected.connect(
            self._on_people_special_filter
        )
        self.ribbon.text_search_toggle_requested.connect(
            lambda: self.toggle_search_scope_bar(force_visible=True)
        )
        self.ribbon.find_duplicates_requested.connect(self.open_duplicate_finder)

        # Central Search Scope Bar Signals
        self.scope_bar.search_text_changed.connect(self._on_search_text_changed)
        self.scope_bar.search_all_requested.connect(self._on_search_all_requested)
        self.scope_bar.location_clear_requested.connect(self._on_clear_location_filter)
        self.scope_bar.pin_toggled.connect(self._on_scope_bar_pin_toggled)
        self.scope_bar.close_requested.connect(self.hide_search_scope_bar)
        self.scope_bar.person_remove_requested.connect(self.left_nav_dock.unselect_person)
        self.scope_bar.special_people_clear_requested.connect(self._on_clear_special_people_filter)
        self.scope_bar.tag_remove_requested.connect(
            lambda tid: self.left_nav_dock.set_tag_filter_state(tid, "none")
        )
        self.scope_bar.rating_clear_requested.connect(self.left_nav_dock.clear_rating_filters)
        self.scope_bar.flagged_clear_requested.connect(
            lambda: self.left_nav_dock.set_flagged_only(False)
        )
        self.scope_bar.date_clear_requested.connect(
            lambda: self.left_nav_dock.select_date(None, None, None)
        )
        self.scope_bar.media_type_clear_requested.connect(
            lambda: self.left_nav_dock.set_media_type(None)
        )
        self.scope_bar.orientation_clear_requested.connect(
            lambda: self.left_nav_dock.set_orientation_filter("all")
        )

        # Quick Access Toolbar & Navigation History Signals
        self.ribbon.back_requested.connect(self.navigate_history_back)
        self.ribbon.forward_requested.connect(self.navigate_history_forward)
        self.ribbon.undo_requested.connect(self._on_undo)
        self.ribbon.redo_requested.connect(self._on_redo)
        self.ribbon.qat_thumbnails_clicked.connect(self.trigger_thumbnail_view)
        self.ribbon.qat_map_clicked.connect(self.show_map_view)
        self.ribbon.qat_people_clicked.connect(self.open_people_directory_view)
        if hasattr(self.ribbon, "map_location_chosen"):
            self.ribbon.map_location_chosen.connect(self._on_ribbon_map_location_chosen)
        if hasattr(self.ribbon, "map_location_search_requested"):
            self.ribbon.map_location_search_requested.connect(self._on_ribbon_map_location_search)
        if hasattr(self.ribbon, "map_filters_changed"):
            self.ribbon.map_filters_changed.connect(self._on_ribbon_map_filters_changed)
        self.ribbon.filter_month_selected.connect(
            lambda m: self._on_date_filter_selected(None, m if m > 0 else None, None)
        )

        # Unified Undo / Redo stack listeners
        self.undo_stack.canUndoChanged.connect(lambda _: self._update_undo_redo_enabled())
        self.undo_stack.canRedoChanged.connect(lambda _: self._update_undo_redo_enabled())
        self.media_viewer.undo_stack.canUndoChanged.connect(lambda _: self._update_undo_redo_enabled())
        self.media_viewer.undo_stack.canRedoChanged.connect(lambda _: self._update_undo_redo_enabled())

        # ThumbnailGridView & MediaCardDelegate
        self.card_delegate.rating_clicked.connect(self.on_inline_rating_changed)
        self.card_delegate.flag_toggled.connect(self.on_inline_flag_changed)
        self.card_delegate.caption_changed.connect(self.on_inline_caption_changed)
        self.grid_view.viewport_range_changed.connect(self.enqueue_visible_thumbnails)
        self.grid_view.selection_ids_changed.connect(self.on_grid_selection_changed)
        self.grid_view.media_activated.connect(self.open_single_viewer)
        self.grid_view.thumbnail_size_changed.connect(self._on_grid_thumbnail_size_changed)

        # Right Tag & Caption / Fine-Tune Dock
        self.right_tag_dock.rating_edited.connect(self.apply_rating_to_selection)
        self.right_tag_dock.flag_edited.connect(self.apply_flag_to_selection)
        self.right_tag_dock.caption_committed.connect(self.apply_caption_to_selection)
        self.right_tag_dock.geotag_edited.connect(self.apply_geotag_to_selection)
        self.right_tag_dock.tag_added.connect(self.on_selection_tag_added)
        self.right_tag_dock.tag_removed.connect(self.on_selection_tag_removed)
        self.right_tag_dock.tag_assign_to_all_requested.connect(self.on_selection_tag_assign_to_all)
        self.right_tag_dock.tag_filter_requested.connect(self._on_right_dock_tag_filter_requested)
        self.right_tag_dock.person_added.connect(self.on_selection_person_added)
        self.right_tag_dock.person_removed.connect(self.on_selection_person_removed)
        self.right_tag_dock.person_hovered.connect(self._on_dock_person_hovered)
        self.right_tag_dock.person_search_requested.connect(self._on_dock_person_search_requested)
        self.right_tag_dock.person_view_suggestions_requested.connect(self._on_dock_person_view_suggestions_requested)
        self.right_tag_dock.person_tag_as_other_requested.connect(self._on_dock_person_tag_as_other_requested)
        self.right_tag_dock.person_set_display_pic_requested.connect(self._on_dock_person_set_display_pic_requested)
        self.right_tag_dock.person_merge_requested.connect(self._on_dock_person_merge_requested)
        self.right_tag_dock.person_rename_requested.connect(self._on_dock_person_rename_requested)
        self.right_tag_dock.person_confirm_requested.connect(self._on_dock_person_confirm_requested)
        self.right_tag_dock.person_reject_suggestion_requested.connect(self._on_dock_person_reject_suggestion_requested)
        self.right_tag_dock.person_tag_mode_requested.connect(self._on_dock_person_tag_mode_requested)
        self.right_tag_dock.face_region_hovered.connect(self._on_dock_face_region_hovered)
        self.right_tag_dock.assign_face_requested.connect(self._on_assign_face_requested)
        self.right_tag_dock.dismiss_face_requested.connect(self._on_dismiss_face_requested)
        self.right_tag_dock.remove_face_requested.connect(self._on_remove_face_requested)
        self.right_tag_dock.person_locate_face_requested.connect(self.on_locate_person_face_requested)
        self.right_tag_dock.person_link_face_requested.connect(self.on_link_person_face_requested)
        self.media_viewer.face_region_named.connect(self._on_assign_face_requested)
        self.media_viewer.face_suggestion_confirmed.connect(self._on_dock_person_confirm_requested)
        self.media_viewer.face_suggestion_rejected.connect(self._on_dock_person_reject_suggestion_requested)
        self.media_viewer.status_message_requested.connect(self.show_status_message)
        self.right_tag_dock.change_time_requested.connect(self.open_change_time_taken_dialog)
        self.right_tag_dock.filename_committed.connect(self.on_selection_filename_committed)
        self.right_tag_dock.date_taken_edited.connect(self.on_selection_date_taken_edited)
        self.right_tag_dock.author_committed.connect(self.on_selection_author_committed)
        self.left_nav_dock.visibilityChanged.connect(
            lambda vis: self._sync_nav_pane_buttons(vis)
        )
        self.right_tag_dock.visibilityChanged.connect(
            lambda vis: self._sync_tag_pane_buttons(vis)
        )

        # Fine-Tune Sliders -> SingleMediaViewer
        self.right_tag_dock.fine_tune_exposure_changed.connect(
            self.media_viewer.set_exposure_parameters
        )
        self.right_tag_dock.fine_tune_color_changed.connect(
            self.media_viewer.set_color_parameters
        )
        self.right_tag_dock.fine_tune_optics_changed.connect(
            self.media_viewer.set_optics_parameters
        )
        self.right_tag_dock.fine_tune_levels_changed.connect(
            self.media_viewer.set_levels
        )
        self.right_tag_dock.fine_tune_reset_requested.connect(
            self.media_viewer.reset_recipe
        )
        self.right_tag_dock.fine_tune_straighten_changed.connect(
            self.media_viewer.set_straighten
        )
        self.right_tag_dock.fine_tune_detail_changed.connect(
            self.media_viewer.set_detail
        )
        self.right_tag_dock.compare_started.connect(
            lambda: self.media_viewer.set_compare_mode(True)
        )
        self.right_tag_dock.compare_ended.connect(
            lambda: self.media_viewer.set_compare_mode(False)
        )
        self.right_tag_dock.analyze_noise_requested.connect(
            self._on_analyze_noise_requested
        )
        self.right_tag_dock.fine_tune_slider_pressed.connect(
            self.media_viewer.begin_slider_adjustment
        )
        self.right_tag_dock.fine_tune_slider_released.connect(
            self.media_viewer.commit_slider_adjustment
        )
        self.media_viewer.rotate_requested.connect(self.rotate_selection)
        self.right_tag_dock.auto_straighten_requested.connect(
            self.media_viewer.apply_auto_straighten
        )
        self.right_tag_dock.fine_tune_auto_adjust.connect(
            self.media_viewer.apply_auto_adjust
        )
        self.right_tag_dock.depth_bokeh_changed.connect(
            self.media_viewer.set_depth_bokeh_parameters
        )
        self.right_tag_dock.depth_relight_changed.connect(
            self.media_viewer.set_depth_relight_parameters
        )
        self.right_tag_dock.depth_haze_changed.connect(
            self.media_viewer.set_depth_haze_parameters
        )
        self.right_tag_dock.depth_reset_requested.connect(
            self.media_viewer.reset_depth_light
        )
        self.right_tag_dock.copy_subject_requested.connect(
            self.media_viewer.copy_subject_to_clipboard
        )
        self.right_tag_dock.remove_background_requested.connect(
            self.media_viewer.remove_background_cutout
        )
        self.right_tag_dock.mask_preview_toggled.connect(
            self.media_viewer.set_mask_overlay_preview
        )

        # Keyboard shortcuts: Ctrl+I, Ctrl+F, Ctrl+N, Ctrl+T, Undo/Redo, Navigation, Select All, Cut/Copy/Paste
        self.sc_info = QShortcut(QKeySequence("Ctrl+I"), self)
        self.sc_info.activated.connect(lambda: self.toggle_right_pane("info"))

        self.sc_nav_pane = QShortcut(QKeySequence("Ctrl+N"), self)
        self.sc_nav_pane.activated.connect(self.toggle_navigation_pane)

        self.sc_add_tag = QShortcut(QKeySequence("Ctrl+T"), self)
        self.sc_add_tag.activated.connect(self.focus_descriptive_tag_box)

        self.sc_fine_tune = QShortcut(QKeySequence("Ctrl+F"), self)
        self.sc_fine_tune.activated.connect(self._on_ctrl_f)

        self.sc_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.sc_undo.activated.connect(self._on_undo)

        self.sc_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.sc_redo.activated.connect(self._on_redo)

        self.sc_back = QShortcut(QKeySequence("Alt+Left"), self)
        self.sc_back.activated.connect(self.navigate_history_back)

        self.sc_forward = QShortcut(QKeySequence("Alt+Right"), self)
        self.sc_forward.activated.connect(self.navigate_history_forward)

        self.sc_browser_back = QShortcut(QKeySequence(QKeySequence.StandardKey.Back), self)
        self.sc_browser_back.activated.connect(self.navigate_history_back)

        self.sc_browser_forward = QShortcut(QKeySequence(QKeySequence.StandardKey.Forward), self)
        self.sc_browser_forward.activated.connect(self.navigate_history_forward)

        self.sc_select_all = QShortcut(QKeySequence("Ctrl+A"), self)
        self.sc_select_all.activated.connect(self.select_all_items)

        self.sc_cut = QShortcut(QKeySequence("Ctrl+X"), self)
        self.sc_cut.activated.connect(self.cut_selection_to_clipboard)

        self.sc_copy = QShortcut(QKeySequence("Ctrl+C"), self)
        self.sc_copy.activated.connect(self.copy_selection_to_clipboard)

        self.sc_paste = QShortcut(QKeySequence("Ctrl+V"), self)
        self.sc_paste.activated.connect(self.paste_from_clipboard)

        self.sc_rot_left = QShortcut(QKeySequence("Ctrl+,"), self)
        self.sc_rot_left.activated.connect(lambda: self.rotate_selection(clockwise=False))

        self.sc_rot_right = QShortcut(QKeySequence("Ctrl+."), self)
        self.sc_rot_right.activated.connect(lambda: self.rotate_selection(clockwise=True))

        self.sc_slideshow = QShortcut(QKeySequence(Qt.Key_F12), self)
        self.sc_slideshow.activated.connect(lambda: self.dispatch_ribbon_action("slide_show"))

        self.sc_thumbnail_view = QShortcut(QKeySequence("Ctrl+0"), self)
        self.sc_thumbnail_view.activated.connect(self.trigger_thumbnail_view)

        self.sc_actual_size = QShortcut(QKeySequence("Ctrl+1"), self)
        self.sc_actual_size.activated.connect(lambda: self.media_viewer.set_zoom_mode("100"))

        # F4: Immersive Viewer Mode (hides side panes, status bar, QAT, and collapses ribbon)
        self.sc_immersive = QShortcut(QKeySequence("F4"), self)
        self.sc_immersive.setContext(Qt.ApplicationShortcut)
        self.sc_immersive.activated.connect(self.toggle_immersive_viewer_mode)

        # Star rating shortcuts (0 through 5)
        self._rating_shortcuts = []
        for r in range(6):
            sc = QShortcut(QKeySequence(str(r)), self)
            sc.activated.connect(lambda val=r: self._on_rating_shortcut(val))
            self._rating_shortcuts.append(sc)

        # Zoom in / out shortcuts (+, =, -, _) and Print shortcut (Ctrl+P)
        self._zoom_shortcuts = []
        for seq in ("+", "="):
            sc = QShortcut(QKeySequence(seq), self)
            sc.activated.connect(self._on_zoom_in_shortcut)
            self._zoom_shortcuts.append(sc)
        for seq in ("-", "_"):
            sc = QShortcut(QKeySequence(seq), self)
            sc.activated.connect(self._on_zoom_out_shortcut)
            self._zoom_shortcuts.append(sc)

        sc_print = QShortcut(QKeySequence("Ctrl+P"), self)
        sc_print.activated.connect(self.open_print_dialog)
        self._zoom_shortcuts.append(sc_print)

        self.sc_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self.sc_save.activated.connect(lambda: self.dispatch_ribbon_action("save_photo"))

        self.sc_save_copy = QShortcut(QKeySequence("Ctrl+Shift+S"), self)
        self.sc_save_copy.activated.connect(lambda: self.dispatch_ribbon_action("save_as_copy"))

        self.sc_properties_ret = QShortcut(QKeySequence("Alt+Return"), self)
        self.sc_properties_ret.activated.connect(self.open_file_properties)
        self.sc_properties_enter = QShortcut(QKeySequence("Alt+Enter"), self)
        self.sc_properties_enter.activated.connect(self.open_file_properties)

        # SingleMediaViewer
        self.media_viewer.closed.connect(self.show_grid_view)
        self.media_viewer.navigate_requested.connect(self.navigate_single_viewer)
        self.media_viewer.pixel_edit_saved.connect(self._on_pixel_edit_saved)
        self.media_viewer.pixel_edit_saved_as_copy.connect(self._on_pixel_edit_saved_as_copy)
        self.media_viewer.pixel_edit_reverted.connect(self._on_pixel_edit_reverted)
        self.media_viewer.face_region_tagged.connect(self._on_viewer_face_region_tagged)
        self.media_viewer.image_changed.connect(
            self.right_tag_dock.histogram_widget.set_histogram_data
        )
        self.media_viewer.image_changed.connect(
            lambda _: self._refresh_effects_gallery_thumbnails()
        )
        self.media_viewer.recipe_changed.connect(
            self.right_tag_dock.sync_recipe
        )
        self.media_viewer.recipe_changed.connect(
            lambda rec: self.ribbon.effects_gallery.set_active_effect(rec.effect)
            if hasattr(self.ribbon, "effects_gallery")
            else None
        )
        self.media_viewer.zoom_percent_changed.connect(
            lambda _: self._update_status_zoom_toggle_button()
        )
        self.media_viewer.zoom_percent_changed.connect(self._sync_status_slider_from_viewer)

        # Ribbon Exposure & Color popup real-time preview connections
        self.ribbon.exposure_preview_requested.connect(self._on_exposure_preview)
        self.ribbon.auto_exposure_preview_requested.connect(self._on_auto_exposure_preview)
        self.ribbon.exposure_preview_cancelled.connect(self._on_preview_cancelled)
        self.ribbon.exposure_adjustment_selected.connect(self._on_exposure_selected)
        self.ribbon.auto_exposure_selected.connect(self._on_auto_exposure_selected)
        self.ribbon.exposure_menu_about_to_show.connect(self._on_exposure_menu_about_to_show)

        self.ribbon.color_preview_requested.connect(self._on_color_preview)
        self.ribbon.auto_color_preview_requested.connect(self._on_auto_color_preview)
        self.ribbon.color_preview_cancelled.connect(self._on_preview_cancelled)
        self.ribbon.color_adjustment_selected.connect(self._on_color_selected)
        self.ribbon.auto_color_selected.connect(self._on_auto_color_selected)
        self.ribbon.color_menu_about_to_show.connect(self._on_color_menu_about_to_show)

        # Ribbon Effects gallery hover real-time preview connections
        self.ribbon.effects_gallery.effect_hovered.connect(self._on_effect_hovered)
        self.ribbon.effects_gallery.effect_unhovered.connect(self._on_preview_cancelled)

        # WorldMapView signals connected lazily when view is created in show_map_view()

    # -------------------------------------------------------------------------
    # Folder Scanning & Progressive Sync
    # -------------------------------------------------------------------------
    def _prompt_and_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add Folder to Gallery")
        if folder:
            self.scan_folder(folder, include_subfolders=True)

    def open_import_wizard(self, source_path: Path | None = None) -> None:
        """Opens the WLPG Camera, Memory Card & Folder Import Wizard."""
        from src.ui.import_dialog import ImportMediaDialog
        dlg = ImportMediaDialog(self.catalog_repo, initial_source=source_path, parent=self)
        dlg.import_completed.connect(self._on_import_wizard_completed)
        dlg.exec()

    def _on_import_wizard_completed(self, created_folders: list[Path]) -> None:
        """Triggered after wizard copies files and creates destination folders."""
        for folder_path in created_folders:
            self.scan_folder(folder_path, include_subfolders=False)
        self.refresh_navigation_and_autocomplete()
        self.rebuild_and_run_query()
        self.lbl_sync_status.setText(f"✓ Imported to {len(created_folders)} folder(s)")

    def _on_remove_folder_from_library(
        self, folder_id: int, folder_path: str, folder_name: str
    ) -> None:
        display_name = folder_name or Path(folder_path).name or f"Folder {folder_id}"
        reply = QMessageBox.question(
            self,
            "Remove Folder from Gallery",
            f"Are you sure you want to remove '{display_name}' and its subfolders from the gallery?\n\n"
            "This will remove the items from the Photo Gallery database and thumbnail cache.\n"
            "(Your original files and folders on disk will NOT be deleted).",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        removed_ids = self.catalog_repo.remove_folder_tree(folder_id)
        for mid in removed_ids:
            self.pixmap_cache.remove(mid)

        # Clear left nav selection if removed folder was selected
        if self.left_nav_dock._folder_id == folder_id:
            self.left_nav_dock._folder_id = None
            self.left_nav_dock._folder_name = "All folders"

        self.refresh_navigation_and_autocomplete()
        self.rebuild_and_run_query()
        self.folder_watcher.update_watched_paths()
        self.statusBar().showMessage(
            f"Removed '{display_name}' and its subfolders from gallery.", 5000
        )

    def _on_folder_renamed(self, old_path: str, new_path: str, new_name: str) -> None:
        """Handles inline folder renaming: updates catalog.db, watcher, and UI state."""
        self.catalog_repo.rename_folder(old_path, new_path, new_name)
        self._media_meta_lookup.clear()
        if hasattr(self, "folder_watcher") and self.folder_watcher is not None:
            self.folder_watcher.update_watched_paths()
        if getattr(self.left_nav_dock, "_folder_name", "") == Path(old_path).name:
            self.left_nav_dock._folder_name = new_name
        self.refresh_navigation_and_autocomplete()
        self.rebuild_and_run_query()
        self.statusBar().showMessage(f"Renamed folder to '{new_name}'.", 4000)

    def _on_folder_created(self, new_dir_str: str) -> None:
        """Handles newly created folders."""
        if hasattr(self, "folder_watcher") and self.folder_watcher is not None:
            self.folder_watcher.update_watched_paths()
        self.refresh_navigation_and_autocomplete()
        self.statusBar().showMessage(f"Created new folder '{Path(new_dir_str).name}'.", 4000)

    def _on_folder_deleted(self, folder_id: int, folder_path: str) -> None:
        """Handles folder deletion moved to Recycle Bin."""
        if folder_id:
            removed_ids = self.catalog_repo.remove_folder_tree(folder_id)
            for mid in removed_ids:
                self.pixmap_cache.remove(mid)
        self._media_meta_lookup.clear()
        if hasattr(self, "folder_watcher") and self.folder_watcher is not None:
            self.folder_watcher.update_watched_paths()
        self.refresh_navigation_and_autocomplete()
        self.rebuild_and_run_query()
        self.statusBar().showMessage("Moved folder to Recycle Bin.", 4000)

    def _on_files_dropped_on_folder(
        self,
        target_folder_id: int,
        target_folder_path: str,
        file_paths: list[str],
        is_copy: bool = False,
    ) -> None:
        if not file_paths:
            return

        if not target_folder_path and target_folder_id:
            frow = self.catalog_repo.get_folder_by_id(target_folder_id)
            if frow:
                target_folder_path = str(frow["path"])
        if not target_folder_path:
            return

        target_dir = os.path.normpath(target_folder_path)
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir, exist_ok=True)
            except Exception:
                pass

        target_folder_name = os.path.basename(target_dir) or "folder"

        if is_copy:
            copied_count = 0
            upsert_batch: list[dict[str, Any]] = []
            for fpath in file_paths:
                norm_fpath = os.path.normpath(fpath)
                if not os.path.exists(norm_fpath):
                    continue
                src_fname = os.path.basename(norm_fpath)
                dst_fname = src_fname
                dst_path = os.path.join(target_dir, dst_fname)
                counter = 1
                stem, ext = os.path.splitext(src_fname)
                while os.path.exists(dst_path):
                    dst_fname = f"{stem} ({counter}){ext}"
                    dst_path = os.path.join(target_dir, dst_fname)
                    counter += 1
                try:
                    shutil.copy2(norm_fpath, dst_path)
                    st = os.stat(dst_path)
                    upsert_batch.append(
                        {
                            "folder_id": target_folder_id,
                            "filename": dst_fname,
                            "ext": ext.lower().lstrip("."),
                            "size": int(st.st_size),
                            "mtime": int(st.st_mtime),
                            "date_taken": None,
                        }
                    )
                    copied_count += 1
                except Exception:
                    pass
            if copied_count > 0:
                if upsert_batch:
                    try:
                        self.catalog_repo.bulk_upsert_media(upsert_batch)
                    except Exception:
                        pass
                self._media_meta_lookup.clear()
                self.refresh_navigation_and_autocomplete()
                self.rebuild_and_run_query()
                self.scan_folder(target_dir, include_subfolders=False)
                item_word = "item" if copied_count == 1 else "items"
                self.statusBar().showMessage(
                    f"Copied {copied_count} {item_word} to {target_folder_name}", 4000
                )
            return

        moves: list[dict[str, Any]] = []
        external_count = 0

        # Build fast O(1) normalized folder path -> (folder_id, raw_path) map
        # so looking up dropped files uses the SQLite (folder_id, filename) index instead of a full-table scan.
        folder_lookup: dict[str, tuple[int, str]] = {}
        for f in self.catalog_repo.get_folders():
            fid = int(f["id"])
            f_raw = str(f["path"])
            norm_key = f_raw.replace("\\", "/").rstrip("/").lower()
            folder_lookup[norm_key] = (fid, f_raw)

        conn = self.catalog_repo.engine.catalog_conn()
        for fpath in file_paths:
            p = Path(fpath)
            if not p.exists():
                continue
            parent_posix = p.parent.as_posix().rstrip("/").lower()
            fname = p.name

            row = None
            folder_match = folder_lookup.get(parent_posix)
            if folder_match is not None:
                matched_fid, matched_fpath = folder_match
                row_m = conn.execute(
                    """
                    SELECT id, folder_id, filename
                    FROM media
                    WHERE folder_id = ? AND filename = ? COLLATE NOCASE;
                    """,
                    (matched_fid, fname),
                ).fetchone()
                if row_m is not None:
                    row = {
                        "id": row_m["id"],
                        "folder_id": row_m["folder_id"],
                        "filename": row_m["filename"],
                        "folder_path": matched_fpath,
                    }
            else:
                row = conn.execute(
                    """
                    SELECT m.id, m.folder_id, m.filename, f.path AS folder_path
                    FROM media m
                    JOIN folders f ON f.id = m.folder_id
                    WHERE RTRIM(REPLACE(LOWER(f.path), '\\', '/'), '/') = ? AND LOWER(m.filename) = LOWER(?);
                    """,
                    (parent_posix, fname),
                ).fetchone()

            if row is None:
                # File dragged from external Explorer or other folder not yet indexed!
                dst_path = os.path.join(target_dir, fname)
                counter = 1
                stem, ext = os.path.splitext(fname)
                while os.path.exists(dst_path) and dst_path.lower() != str(p).lower():
                    dst_path = os.path.join(target_dir, f"{stem} ({counter}){ext}")
                    counter += 1
                try:
                    shutil.move(str(p), dst_path)
                    external_count += 1
                except Exception:
                    pass
                continue

            mid = int(row["id"])
            src_fid = int(row["folder_id"])
            src_fname = str(row["filename"])
            src_path = os.path.normpath(os.path.join(str(row["folder_path"]), src_fname))

            # If already in the target folder, skip
            if src_fid == target_folder_id or os.path.dirname(src_path).lower() == target_dir.lower():
                continue

            # Check if destination filename collides
            dst_fname = src_fname
            dst_path = os.path.join(target_dir, dst_fname)
            counter = 1
            stem, ext = os.path.splitext(src_fname)
            while os.path.exists(dst_path) and dst_path.lower() != src_path.lower():
                dst_fname = f"{stem} ({counter}){ext}"
                dst_path = os.path.join(target_dir, dst_fname)
                counter += 1

            moves.append({
                "media_id": mid,
                "src_path": src_path,
                "dst_path": dst_path,
                "src_folder_id": src_fid,
                "dst_folder_id": target_folder_id,
                "src_filename": src_fname,
                "dst_filename": dst_fname,
            })

        if moves:
            cmd = _MoveFilesUndoCommand(self, moves, target_folder_name)
            self.undo_stack.push(cmd)
            count = len(moves)
            item_word = "item" if count == 1 else "items"
            self.statusBar().showMessage(
                f"Moved {count} {item_word} to {target_folder_name}", 4000
            )

        if external_count > 0:
            self._media_meta_lookup.clear()
            self.scan_folder(target_dir, include_subfolders=False)
            item_word = "item" if external_count == 1 else "items"
            self.statusBar().showMessage(
                f"Moved {external_count} external {item_word} to {target_folder_name}", 4000
            )

    def scan_folder(
        self,
        folder_path: str | Path,
        include_subfolders: bool = True,
        blocking: bool = False,
    ) -> None:
        root_p = Path(folder_path).resolve()
        # Immediately register the root folder (and immediate subfolders) so it appears in the Navigation Pane with 0ms delay
        if root_p.exists() and root_p.is_dir():
            try:
                st = root_p.stat()
                root_fid = self.catalog_repo.upsert_folder(
                    path=str(root_p),
                    name=root_p.name or str(root_p),
                    parent_id=None,
                    mtime=int(st.st_mtime),
                )
                if include_subfolders:
                    from src.workers.scanner_thumb_worker import IGNORED_DIR_NAMES
                    with os.scandir(root_p) as it:
                        for entry in it:
                            if entry.is_dir(follow_symlinks=False) and entry.name.lower() not in IGNORED_DIR_NAMES:
                                try:
                                    sub_st = entry.stat(follow_symlinks=False)
                                    self.catalog_repo.upsert_folder(
                                        path=str(Path(entry.path)),
                                        name=entry.name,
                                        parent_id=root_fid,
                                        mtime=int(sub_st.st_mtime),
                                    )
                                except OSError:
                                    pass
                self.refresh_navigation_and_autocomplete()
            except Exception:
                pass

        self._scanner_cancel.set()
        self._scanner_cancel = CancellationFlag()
        self._scanner_status_text = "Scanning: Active"
        self._update_worker_monitor()
        worker = FolderSyncWorker(
            self.engine,
            root_p,
            include_subfolders=include_subfolders,
            cancel_flag=self._scanner_cancel,
        )
        worker.signals.progress.connect(self.on_sync_progress)
        worker.signals.stage1_completed.connect(self._on_sync_stage1_completed)
        worker.signals.finished.connect(self.on_sync_finished)
        worker.signals.error.connect(self.on_worker_error)
        if blocking:
            worker.run()
        else:
            self.scanner_pool.start(worker)

    def _on_sync_stage1_completed(self, discovered_count: int) -> None:
        """Refreshes navigation tree and grid immediately after fast Stage 1 os.scandir walk completes."""
        self._media_meta_lookup.clear()
        self.refresh_navigation_and_autocomplete()
        self.rebuild_and_run_query()
        if hasattr(self, "folder_watcher"):
            self.folder_watcher.update_watched_paths()

    def on_sync_progress(self, count: int, folder_name: str) -> None:
        self._scanner_status_text = f"Scanning: {count:,} files"
        self._update_worker_monitor()

    def on_sync_finished(self, upserted_count: int, was_completed: bool) -> None:
        self._scanner_status_text = "Scanning: Idle"
        self._update_worker_monitor()
        if upserted_count > 0:
            self.statusBar().showMessage(f"✓ Synced {upserted_count:,} items", 4000)
        elif not was_completed:
            self.statusBar().showMessage("Scan cancelled", 3000)
        if upserted_count > 0 or was_completed:
            self._media_meta_lookup.clear()
        if upserted_count > 1000 and was_completed:
            try:
                self.engine.analyze_catalog()
            except Exception:
                pass
        self.refresh_navigation_and_autocomplete()
        self.rebuild_and_run_query()
        if hasattr(self, "folder_watcher"):
            self.folder_watcher.update_watched_paths()
        if was_completed and upserted_count > 0 and getattr(self, "find_people_enabled", False):
            self.start_face_detection()

    def _on_folder_watcher_delta(
        self, folder_path: str, added: int, modified: int, deleted: int, folders_changed: int = 0
    ) -> None:
        if added > 0 or modified > 0 or deleted > 0 or folders_changed > 0:
            self._media_meta_lookup.clear()
            self.refresh_navigation_and_autocomplete()
            self.rebuild_and_run_query()
            fname = os.path.basename(folder_path) or folder_path
            parts: list[str] = []
            if added > 0:
                parts.append(f"+{added}")
            if modified > 0:
                parts.append(f"~{modified}")
            if deleted > 0:
                parts.append(f"-{deleted}")
            if folders_changed > 0:
                parts.append(f"{folders_changed} folder(s)")
            summary = ", ".join(parts) if parts else "updated"
            self.statusBar().showMessage(f"✓ Synced '{fname}': {summary}", 4000)
            if added > 0 and getattr(self, "find_people_enabled", False):
                self.start_face_detection()

    def _on_startup_reconciliation_completed(
        self, added: int, modified: int, deleted: int, folders_changed: int
    ) -> None:
        if added > 0 or modified > 0 or deleted > 0 or folders_changed > 0:
            self._media_meta_lookup.clear()
            self.refresh_navigation_and_autocomplete()
            self.rebuild_and_run_query()
            parts: list[str] = []
            if added > 0:
                parts.append(f"+{added} item(s)")
            if modified > 0:
                parts.append(f"~{modified} item(s)")
            if deleted > 0:
                parts.append(f"-{deleted} item(s)")
            if folders_changed > 0:
                parts.append(f"{folders_changed} folder(s)")
            self.statusBar().showMessage(f"✓ Startup sync: {', '.join(parts)}", 4000)
            if added > 0 and getattr(self, "find_people_enabled", False):
                self.start_face_detection()

    def on_worker_error(self, err_msg: str) -> None:
        self.lbl_sync_status.setText(f"⚠ {err_msg}")

    def on_metadata_batch_synced(self, media_ids: list[int]) -> None:
        self._update_worker_monitor()

    # -------------------------------------------------------------------------
    # Background Face Detection & Recognition
    # -------------------------------------------------------------------------
    def start_face_detection(self, force_rescan: bool = False) -> None:
        """Launches background FaceDetectionWorker on self.face_pool if not already running."""
        if self._is_shutting_down:
            return
        if self._face_worker is not None:
            return
        self._face_cancel.clear()
        worker = FaceDetectionWorker(
            engine=self.engine,
            cancel_flag=self._face_cancel,
            batch_size=50,
            throttle_ms=5,
            force_rescan=force_rescan,
        )
        worker.signals.status_message.connect(self.on_face_status_message)
        worker.signals.batch_progressed.connect(self.on_face_batch_progressed)
        worker.signals.suggestions_updated.connect(self.on_face_suggestions_updated)
        worker.signals.worker_finished.connect(self.on_face_worker_finished)
        self._face_worker = worker
        self._face_phase = "init"
        self._face_status_text = "Faces: Starting face scan..."
        self._update_worker_monitor()
        self.face_pool.start(worker)

    def stop_face_detection(self) -> None:
        """Requests cancellation of background FaceDetectionWorker."""
        self._face_cancel.set()

    def on_face_status_message(self, message: str) -> None:
        """Updates status text when FaceDetectionWorker transitions between stages."""
        if self._face_worker is not None:
            clean_msg = message
            if "Stage 1" in message:
                self._face_phase = "verify"
                clean_msg = "Verifying face tags (1/2)..."
            elif "Stage 2" in message or "Scanning" in message:
                self._face_phase = "scan"
                clean_msg = "Finding people (2/2)..."
            elif "Stage 3:" in message:
                self._face_phase = "match"
                clean_msg = "Computing face clusters..."
            elif "Stage 4:" in message:
                self._face_phase = "match"
                clean_msg = "Grouping face suggestions..."
            self._face_status_text = f"Faces: {clean_msg}"
            self._update_worker_monitor()

    def on_face_batch_progressed(self, current: int, total: int) -> None:
        """Updates status bar and BatchPeopleView when face scanning makes progress."""
        phase = getattr(self, "_face_phase", "scan")
        phase_label = "Verifying face tags (1/2)" if phase == "verify" else "Finding people (2/2)"
        if total > 0 and current <= total:
            pct = int((current / total) * 100)
            self._face_status_text = f"Faces: {phase_label}: {current:,}/{total:,} ({pct}%)"
        elif total > 0:
            self._face_status_text = f"Faces: {phase_label}: {current:,}/{total:,}"
        elif self._face_worker is not None:
            self._face_status_text = f"Faces: {phase_label}..."
        else:
            self._face_status_text = "Faces: Idle"
        if hasattr(self, "batch_people_view"):
            self.batch_people_view.set_scan_progress(current, total)
        self._update_worker_monitor()

    def on_face_suggestions_updated(self) -> None:
        """Refreshes People tree, autocomplete, and BatchPeopleView when new face suggestions or centroids appear."""
        if hasattr(self.left_nav_dock, "refresh_people_only"):
            people = self.left_nav_dock.refresh_people_only(self.catalog_repo)
            self.ribbon.populate_face_carousel(people)
        else:
            self.refresh_navigation_and_autocomplete()
        if hasattr(self, "batch_people_view") and self.central_stack.currentWidget() == self.batch_people_view:
            self.batch_people_view.refresh_suggestions(debounce=True)

    def on_face_worker_finished(self, total_faces: int) -> None:
        """Cleans up active worker reference and updates status bar and BatchPeopleView."""
        self._face_worker = None
        self._face_phase = "idle"
        self._face_status_text = "Faces: Idle"
        self._update_worker_monitor()
        if hasattr(self, "batch_people_view"):
            self.batch_people_view.set_scan_progress(0, 0)
            if self.central_stack.currentWidget() == self.batch_people_view:
                self.batch_people_view.refresh_suggestions(debounce=True)
        if total_faces > 0:
            self.statusBar().showMessage(f"✓ Face scan complete: {total_faces:,} face(s) found", 4000)
            self.refresh_navigation_and_autocomplete()

    # -------------------------------------------------------------------------
    # Sub-10ms Query Execution & Viewport Thumbnail Prefetching
    # -------------------------------------------------------------------------
    def _load_saved_mru_tags(self) -> list[str]:
        """Loads persisted most-recently-used descriptive tags from QSettings / settings.json."""
        try:
            settings = QSettings("PhotoGallery64", "PhotoGallery64")
            raw = settings.value("recent_descriptive_tags")
            if isinstance(raw, str) and raw.strip():
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()][:15]
            elif isinstance(raw, list):
                return [str(x).strip() for x in raw if str(x).strip()][:15]
        except Exception:
            pass
        try:
            from src.utils.paths import get_app_data_dir
            s_path = get_app_data_dir() / "settings.json"
            if s_path.exists():
                data = json.loads(s_path.read_text(encoding="utf-8"))
                parsed = data.get("recent_descriptive_tags")
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()][:15]
        except Exception:
            pass
        return []

    def _record_mru_descriptive_tag(self, tag_path: str) -> None:
        """Promotes tag_path to the front of the MRU list and persists it."""
        clean = str(tag_path).strip()
        if not clean:
            return
        cur = getattr(self, "_mru_descriptive_tags", [])
        self._mru_descriptive_tags = [
            clean,
            *[t for t in cur if t.lower() != clean.lower()],
        ][:15]
        try:
            settings = QSettings("PhotoGallery64", "PhotoGallery64")
            settings.setValue("recent_descriptive_tags", json.dumps(self._mru_descriptive_tags))
        except Exception:
            pass
        try:
            from src.utils.paths import get_app_data_dir
            s_path = get_app_data_dir() / "settings.json"
            pref_data: dict[str, Any] = {}
            if s_path.exists():
                try:
                    pref_data = json.loads(s_path.read_text(encoding="utf-8"))
                except Exception:
                    pref_data = {}
            pref_data["recent_descriptive_tags"] = list(self._mru_descriptive_tags)
            s_path.write_text(json.dumps(pref_data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def refresh_navigation_and_autocomplete(self) -> None:
        people = self.catalog_repo.get_people_list()
        tags = self.catalog_repo.get_tag_tree()
        self.left_nav_dock.refresh_trees(
            self.catalog_repo, preloaded_people=people, preloaded_tags=tags
        )
        all_tag_paths = [str(t["full_path"]) for t in tags]
        canon_by_lower = {p.lower(): p for p in all_tag_paths}
        mru_valid: list[str] = []
        seen_low: set[str] = set()
        for t in getattr(self, "_mru_descriptive_tags", []):
            low = t.lower()
            if low in canon_by_lower and low not in seen_low:
                mru_valid.append(canon_by_lower[low])
                seen_low.add(low)
        db_recent = self.catalog_repo.get_recent_tags(12)
        for t in db_recent:
            low = t.lower()
            if low not in seen_low:
                mru_valid.append(t)
                seen_low.add(low)
            if len(mru_valid) >= 12:
                break
        self.ribbon.populate_face_carousel(people)
        self.ribbon.populate_tags_menu(tags)
        self.right_tag_dock.set_autocomplete_lists(
            all_tag_paths,
            [str(p["name"]) for p in people],
            recent_tags=mru_valid[:12],
        )

    def refresh_people_navigation_and_autocomplete(self) -> None:
        """Fast people-only navigation and autocomplete refresh (avoids full tree rebuild)."""
        if hasattr(self.left_nav_dock, "refresh_people_only"):
            people = self.left_nav_dock.refresh_people_only(self.catalog_repo)
        else:
            people = self.catalog_repo.get_people_list()
            self.left_nav_dock.refresh_trees(self.catalog_repo, preloaded_people=people)
        self.ribbon.populate_face_carousel(people)
        if hasattr(self.right_tag_dock, "_people_model"):
            self.right_tag_dock._people_model.setStringList([str(p["name"]) for p in people])

    def _on_ribbon_tag_filter(self, tag_id: int | None, mode: str) -> None:
        if tag_id is None:
            if mode in ("AND", "OR"):
                self.left_nav_dock.set_tag_boolean_mode(mode)  # type: ignore[arg-type]
            else:
                self.left_nav_dock.clear_tag_filters()
        else:
            self.left_nav_dock.set_tag_filter_state(tag_id, "include")

    def _on_right_dock_tag_filter_requested(self, tag_id: int) -> None:
        if self.central_stack.currentWidget() != self.grid_view:
            self.show_grid_view()
        self.left_nav_dock.clear_tag_filters()
        self.left_nav_dock.set_tag_filter_state(tag_id, "include")

    def _has_active_filter_facets(self) -> bool:
        state = self.left_nav_dock.current_filter_state()
        m_type = str(state.get("media_type") or "").lower().strip()
        has_media_filter = bool(m_type and m_type not in ("all", "none"))
        o_filt = str(state.get("orientation_filter") or "").lower().strip()
        has_orientation_filter = bool(o_filt and o_filt not in ("all", "none"))
        return bool(
            state.get("album_id") is not None
            or state.get("smart_album_id") is not None
            or state.get("person_ids")
            or state.get("include_tag_ids")
            or state.get("ratings")
            or state.get("flagged_only")
            or has_media_filter
            or has_orientation_filter
            or state.get("date_year") is not None
            or state.get("date_month") is not None
            or self._search_text
            or self._active_bbox is not None
            or self._active_map_media_ids is not None
            or self._active_map_location_label is not None
            or getattr(self, "_active_special_people_filter", None) is not None
            or self._search_scope_bar_explicitly_opened
        )

    def _active_scope_label(self, state: dict[str, Any]) -> str | None:
        if state.get("album_id") is not None:
            return f"Album: {state.get('album_name') or state.get('album_id')}"
        if state.get("smart_album_id") is not None:
            return f"Smart Album: {state.get('smart_album_name') or state.get('smart_album_id')}"
        folder_name = self.left_nav_dock.selected_folder_name()
        if folder_name:
            return f"Folder: {folder_name}"
        return None

    def _update_scope_bar_visibility(self) -> None:
        state = self.left_nav_dock.current_filter_state()
        self.scope_bar.set_scope_label(self._active_scope_label(state))

        self.scope_bar.update_active_filter_chips(
            state,
            self.catalog_repo,
            location_label=self._active_map_location_label,
            special_people_label=getattr(self, "_active_special_people_label", None),
        )
        if self.scope_bar.is_pinned() or self._has_active_filter_facets():
            self.scope_bar.show()
        else:
            self.scope_bar.hide()

    def _on_search_all_requested(self) -> None:
        """Removes only the folder filter, expanding search to all photos and videos while keeping bar open and filters intact."""
        self._search_scope_bar_explicitly_opened = True
        self.left_nav_dock.clear_folder_filter()
        self.scope_bar.show()
        self.scope_bar.focus_search_input()

    def _on_clear_location_filter(self) -> None:
        """Removes the location/map filter chip and re-runs the catalog query."""
        self._active_map_media_ids = None
        self._active_bbox = None
        self._active_map_location_label = None
        self._update_scope_bar_visibility()
        self.rebuild_and_run_query()

    def toggle_search_scope_bar(self, force_visible: bool | None = None) -> None:
        if force_visible is True:
            self._search_scope_bar_explicitly_opened = True
            self.scope_bar.show()
            self.scope_bar.focus_search_input()
        elif force_visible is False:
            self.hide_search_scope_bar()
        else:
            if self.scope_bar.isVisible():
                self.hide_search_scope_bar()
            else:
                self._search_scope_bar_explicitly_opened = True
                self.scope_bar.show()
                self.scope_bar.focus_search_input()

    def hide_search_scope_bar(self) -> None:
        self._search_scope_bar_explicitly_opened = False
        self.scope_bar.set_pinned(False)
        self.scope_bar.clear_search_input()
        self.scope_bar.hide()
        if self._search_text:
            self._search_text = ""
            self.rebuild_and_run_query()

    def _on_scope_bar_pin_toggled(self, pinned: bool) -> None:
        if not pinned and not self._has_active_filter_facets():
            self.hide_search_scope_bar()

    def _on_ctrl_f(self) -> None:
        if self.central_stack.currentWidget() == self.media_viewer:
            self.toggle_right_pane("fine_tune")
        else:
            self.toggle_search_scope_bar(force_visible=True)

    def focus_descriptive_tag_box(self) -> None:
        """Opens the Tag & Caption pane, focuses Add descriptive tags, and brings up suggestions (Ctrl+T)."""
        self.right_tag_dock.focus_descriptive_tag_input()
        if hasattr(self.ribbon, "btn_toggle_tag_pane"):
            self.ribbon.btn_toggle_tag_pane.setChecked(True)

    def _on_undo(self) -> None:
        if self.central_stack.currentWidget() == self.media_viewer:
            if getattr(self.media_viewer, "is_markup_mode", False):
                self.media_viewer._undo_markup()
            elif self.media_viewer.undo_stack.canUndo():
                self.media_viewer.undo_stack.undo()
            elif self.undo_stack.canUndo():
                self.undo_stack.undo()
        else:
            self.undo_stack.undo()

    def _on_redo(self) -> None:
        if self.central_stack.currentWidget() == self.media_viewer:
            if self.media_viewer.undo_stack.canRedo():
                self.media_viewer.undo_stack.redo()
            elif self.undo_stack.canRedo():
                self.undo_stack.redo()
        else:
            self.undo_stack.redo()

    def _update_undo_redo_enabled(self) -> None:
        try:
            if self.central_stack.currentWidget() == self.media_viewer:
                can_u = self.media_viewer.undo_stack.canUndo() or self.undo_stack.canUndo()
                can_r = self.media_viewer.undo_stack.canRedo() or self.undo_stack.canRedo()
                self.ribbon.set_undo_enabled(can_u)
                self.ribbon.set_redo_enabled(can_r)
            else:
                self.ribbon.set_undo_enabled(self.undo_stack.canUndo())
                self.ribbon.set_redo_enabled(self.undo_stack.canRedo())
        except (RuntimeError, AttributeError):
            pass

    def _update_nav_buttons(self) -> None:
        can_back = self._nav_index > 0
        can_forward = 0 <= self._nav_index < len(self._nav_history) - 1
        if hasattr(self, "ribbon"):
            self.ribbon.set_back_enabled(can_back)
            self.ribbon.set_forward_enabled(can_forward)

    def _push_nav_state(self, state: dict[str, Any]) -> None:
        """Pushes a view/filter navigation state onto the history stack."""
        if self._is_navigating_history:
            return
        if 0 <= self._nav_index < len(self._nav_history):
            if self._nav_history[self._nav_index] == state:
                return
        self._nav_history = self._nav_history[: self._nav_index + 1]
        self._nav_history.append(state)
        self._nav_index = len(self._nav_history) - 1
        self._update_nav_buttons()

    def navigate_history_back(self) -> None:
        """Navigates to the previous state in history (browser/explorer style)."""
        if self._nav_index > 0:
            self._nav_index -= 1
            state = self._nav_history[self._nav_index]
            self._apply_nav_state(state)

    def navigate_history_forward(self) -> None:
        """Navigates to the next state in history (browser/explorer style)."""
        if 0 <= self._nav_index < len(self._nav_history) - 1:
            self._nav_index += 1
            state = self._nav_history[self._nav_index]
            self._apply_nav_state(state)

    def _apply_nav_state(self, state: dict[str, Any]) -> None:
        """Applies a view/filter navigation state restored from the history stack."""
        self._is_navigating_history = True
        try:
            view = state.get("view", "grid")
            if view == "people_directory":
                if hasattr(self, "media_viewer") and self.central_stack.currentWidget() == self.media_viewer:
                    self.show_grid_view()
                self._active_map_media_ids = None
                self.grid_view.set_person_header(None)
                self.card_delegate.set_face_crop_person_id(None)
                if hasattr(self, "people_directory_view"):
                    self.central_stack.setCurrentWidget(self.people_directory_view)
                    self.people_directory_view.refresh_directory()
                    self.scope_bar.set_scope_label("People directory")
                    self.ribbon.set_scope_label("People")
                    self.lbl_sync_status.setText("People directory")
            elif view == "albums_view":
                if hasattr(self, "media_viewer") and self.central_stack.currentWidget() == self.media_viewer:
                    self.show_grid_view()
                self._active_map_media_ids = None
                self.grid_view.set_person_header(None)
                self.card_delegate.set_face_crop_person_id(None)
                if hasattr(self, "albums_view"):
                    self.central_stack.setCurrentWidget(self.albums_view)
                    self.ribbon.set_active_mode("albums")
                    self.albums_view.refresh()
                    self.scope_bar.set_scope_label("Albums")
                    self.ribbon.set_scope_label("Albums")
                    self.lbl_sync_status.setText("Albums")
            elif view == "unnamed_person":
                if hasattr(self, "media_viewer") and self.central_stack.currentWidget() == self.media_viewer:
                    self.show_grid_view()
                face_ids = state.get("face_ids", [])
                thumb_blob = state.get("thumb_blob")
                mids = state.get("media_ids", [])
                self._active_map_media_ids = set(mids) if mids else None
                self.central_stack.setCurrentWidget(self.grid_view)
                self.ribbon.set_active_mode("grid")
                self.person_header_bar.load_unnamed_person(face_ids, thumb_blob, photo_count=len(mids))
                self.grid_view.person_header_bar.show()
                self.grid_view._update_top_bars_geometry()
                self.scope_bar.set_scope_label("People to Name")
                self.ribbon.set_scope_label("People")
                self.rebuild_and_run_query()
            elif view == "viewer":
                mid = state.get("media_id")
                if mid is not None:
                    self.open_single_viewer(mid)
            elif view == "grid_special_people":
                if hasattr(self, "media_viewer") and self.central_stack.currentWidget() == self.media_viewer:
                    self.show_grid_view()
                self._active_map_media_ids = set(state.get("media_ids", []))
                if self.central_stack.currentWidget() != self.grid_view:
                    self.central_stack.setCurrentWidget(self.grid_view)
                    self.ribbon.set_active_mode("grid")
                self.grid_view.set_person_header(None)
                self.card_delegate.set_face_crop_person_id(None)
                kind = state.get("special_filter", "")
                lbl = {
                    "all_people": "All tagged or detected",
                    "detected_untagged": "All detected but not tagged",
                    "untagged": "Not tagged or detected",
                }.get(kind, "People")
                self._active_special_people_filter = kind
                self._active_special_people_label = lbl
                folder_name = self.left_nav_dock.selected_folder_name()
                self.scope_bar.set_scope_label(f"Folder: {folder_name}" if folder_name else None)
                self.ribbon.set_scope_label(lbl)
                self._update_scope_bar_visibility()
                self.rebuild_and_run_query()
            elif view == "batch_people":
                if hasattr(self, "media_viewer") and self.central_stack.currentWidget() == self.media_viewer:
                    if self.media_viewer.has_unsaved_adjustments():
                        self.media_viewer.save_current_image_with_backup()
                        self.rebuild_and_run_query()
                    self.media_viewer.media_player.stop()
                self.open_batch_people_view(person_id=state.get("person_id"))
            elif view == "map":
                if hasattr(self, "media_viewer") and self.central_stack.currentWidget() == self.media_viewer:
                    self.show_grid_view()
                self.show_map_view()
            elif view == "grid_map_selection":
                if hasattr(self, "media_viewer") and self.central_stack.currentWidget() == self.media_viewer:
                    self.show_grid_view()
                mids = state.get("media_ids", [])
                loc_label = state.get("location_label", "")
                self._active_map_media_ids = list(mids)
                self._active_map_location_label = loc_label
                filt = state.get("filter")
                if isinstance(filt, dict):
                    self.left_nav_dock.restore_filter_state(filt)
                    selected_pids = filt.get("person_ids", ())
                    self.grid_view.set_person_header(selected_pids[0] if len(selected_pids) == 1 else None)
                    if len(selected_pids) != 1:
                        self.card_delegate.set_face_crop_person_id(None)
                if self.central_stack.currentWidget() != self.grid_view:
                    self.central_stack.setCurrentWidget(self.grid_view)
                    self.ribbon.set_active_mode("grid")
                self._update_scope_bar_visibility()
                self.rebuild_and_run_query()
            elif view == "grid_map_bbox":
                if hasattr(self, "media_viewer") and self.central_stack.currentWidget() == self.media_viewer:
                    self.show_grid_view()
                bbox = state.get("bbox")
                loc_label = state.get("location_label", "")
                self._active_bbox = bbox
                self._active_map_media_ids = None
                self._active_map_location_label = loc_label
                filt = state.get("filter")
                if isinstance(filt, dict):
                    self.left_nav_dock.restore_filter_state(filt)
                    selected_pids = filt.get("person_ids", ())
                    self.grid_view.set_person_header(selected_pids[0] if len(selected_pids) == 1 else None)
                    if len(selected_pids) != 1:
                        self.card_delegate.set_face_crop_person_id(None)
                if self.central_stack.currentWidget() != self.grid_view:
                    self.central_stack.setCurrentWidget(self.grid_view)
                    self.ribbon.set_active_mode("grid")
                self._update_scope_bar_visibility()
                self.rebuild_and_run_query()
            else:
                # Standard grid view (view == "grid" or legacy filter dict)
                if hasattr(self, "media_viewer") and self.central_stack.currentWidget() == self.media_viewer:
                    self.show_grid_view()
                self._active_map_media_ids = None
                self._active_special_people_filter = None
                self._active_special_people_label = None
                filt = state.get("filter", state)
                self.left_nav_dock.restore_filter_state(filt)
                if self.central_stack.currentWidget() != self.grid_view:
                    self.central_stack.setCurrentWidget(self.grid_view)
                    self.ribbon.set_active_mode("grid")
                selected_pids = filt.get("person_ids", ())
                if len(selected_pids) == 1:
                    self.grid_view.set_person_header(selected_pids[0])
                else:
                    self.grid_view.set_person_header(None)
                    self.card_delegate.set_face_crop_person_id(None)
                folder_name = self.left_nav_dock.selected_folder_name()
                scope_desc = f"Folder: {folder_name}" if folder_name else ""
                self.scope_bar.set_scope_label(scope_desc)
                self.ribbon.set_scope_label(scope_desc or "All photos and videos")
                self._update_scope_bar_visibility()
                self.rebuild_and_run_query()
        finally:
            self._is_navigating_history = False
            self._update_nav_buttons()

    def _on_nav_filter_changed(self) -> None:
        self._active_bbox = None
        state = self.left_nav_dock.current_filter_state()
        scope_desc = self._active_scope_label(state) or ""
        self.scope_bar.set_scope_label(scope_desc)
        self.ribbon.set_scope_label(scope_desc or "All photos and videos")
        self._update_scope_bar_visibility()
        if hasattr(self.ribbon, "btn_find_edited"):
            e_filt = self.left_nav_dock.current_filter_state().get("edited_filter", None)
            self.ribbon.btn_find_edited.blockSignals(True)
            self.ribbon.btn_find_edited.setChecked(e_filt is not None)
            self.ribbon._current_edited_filter = e_filt
            if e_filt == 1:
                self.ribbon.btn_find_edited.setToolTip("Filtering: Edited items (click to clear)")
            elif e_filt == 0:
                self.ribbon.btn_find_edited.setToolTip("Filtering: Not edited items (click to clear)")
            else:
                self.ribbon.btn_find_edited.setToolTip("Filter by edited items")
            self.ribbon.btn_find_edited.blockSignals(False)
        if hasattr(self.ribbon, "set_orientation_filter_ui"):
            o_filt = self.left_nav_dock.current_filter_state().get("orientation_filter", "all")
            self.ribbon.set_orientation_filter_ui(o_filt)
        if not self._is_navigating_history:
            filt = self.left_nav_dock.current_filter_state()
            state = dict(filt)
            state["view"] = "grid"
            state["filter"] = filt
            self._push_nav_state(state)

        # If navigating away while in map view, media viewer, people directory, or batch view, return central stack to grid view
        if hasattr(self, "_map_view") and self._map_view is not None and self.central_stack.currentWidget() == self._map_view:
            self.show_grid_view()
        elif hasattr(self, "media_viewer") and self.central_stack.currentWidget() == self.media_viewer:
            self.show_grid_view()
        elif hasattr(self, "people_directory_view") and self.central_stack.currentWidget() == self.people_directory_view:
            self.show_grid_view()
        elif hasattr(self, "albums_view") and self.central_stack.currentWidget() == self.albums_view:
            self.show_grid_view()
        elif hasattr(self, "batch_people_view") and self.central_stack.currentWidget() == self.batch_people_view:
            self.close_batch_people_view()

        # Check if single person filter is active -> show PersonHeaderBar
        selected_pids = self.left_nav_dock.current_filter_state().get("person_ids", ())
        if len(selected_pids) == 1:
            self.grid_view.set_person_header(selected_pids[0])
        else:
            self.grid_view.set_person_header(None)
            self.card_delegate.set_face_crop_person_id(None)

        self.rebuild_and_run_query()

    def _restore_docks_after_batch_people(self) -> None:
        """Restores left navigation and right tag docks when leaving BatchPeopleView."""
        if hasattr(self, "batch_people_view"):
            self.batch_people_view.set_use_ribbon_mode(False)
        if hasattr(self, "_prev_ribbon_tab"):
            delattr(self, "_prev_ribbon_tab")
        is_nav_vis = getattr(self, "_nav_dock_was_visible", True)
        self.left_nav_dock.setVisible(is_nav_vis)
        if is_nav_vis:
            self.left_nav_dock.show()
            self.left_nav_dock.setMinimumWidth(190)
            self.resizeDocks([self.left_nav_dock], [230], Qt.Horizontal)
        if hasattr(self, "ribbon") and hasattr(self.ribbon, "btn_toggle_nav_pane"):
            self.ribbon.btn_toggle_nav_pane.setChecked(is_nav_vis)
        self.right_tag_dock.show()
        self.right_tag_dock.setVisible(True)
        self.right_tag_dock.set_view_mode("grid")

    def open_people_directory_view(self) -> None:
        """Opens Immich / Google Photos inspired People Directory grid."""
        if hasattr(self, "people_directory_view"):
            self._restore_docks_after_batch_people()
            self._active_map_media_ids = None
            self.grid_view.set_person_header(None)
            self.card_delegate.set_face_crop_person_id(None)
            self.central_stack.setCurrentWidget(self.people_directory_view)
            self.ribbon.set_active_mode("people")
            self.people_directory_view.refresh_directory()
            self.scope_bar.set_scope_label("People directory")
            self.ribbon.set_scope_label("People")
            self.lbl_sync_status.setText("People directory")
            self._push_nav_state({"view": "people_directory"})

    def open_albums_view(self) -> None:
        """Opens the visual Albums & Smart Albums directory view."""
        if hasattr(self, "albums_view"):
            self._restore_docks_after_batch_people()
            self._active_map_media_ids = None
            self.grid_view.set_person_header(None)
            self.card_delegate.set_face_crop_person_id(None)
            self.central_stack.setCurrentWidget(self.albums_view)
            self.ribbon.set_active_mode("albums")
            self.albums_view.refresh()
            self.scope_bar.set_scope_label("Albums")
            self.ribbon.set_scope_label("Albums")
            self.lbl_sync_status.setText("Albums")
            if not self._is_navigating_history:
                self._push_nav_state({"view": "albums_view"})

    def _on_albums_view_album_selected(self, album_id: int, album_name: str) -> None:
        self._active_map_media_ids = None
        self.left_nav_dock._select_album_scope(album_id=int(album_id), album_name=album_name)

    def _on_albums_view_smart_album_selected(self, smart_album_id: int, smart_album_name: str) -> None:
        self._active_map_media_ids = None
        self.left_nav_dock._select_smart_album_scope(
            smart_album_id=int(smart_album_id), smart_album_name=smart_album_name
        )

    def open_compare_view(self, media_ids: list[int] | None = None) -> None:
        """Open the Multi-Photo Compare / Light Table view for the selected or active grid photos."""
        if not hasattr(self, "compare_view"):
            return
        ids = list(media_ids) if media_ids is not None else self.grid_view.selected_media_ids()
        if not ids:
            return
        all_grid_ids = [
            int(r.id) for r in getattr(self.catalog_model, "_records", []) if getattr(r, "id", None) is not None
        ]
        self.left_nav_dock.setVisible(False)
        self.right_tag_dock.setVisible(False)
        self.compare_view.bind_repositories(self.catalog_repo, self.thumb_repo)
        self.compare_view.set_dark_mode(getattr(self, "_dark_mode", False))
        self.compare_view.open_media_ids(ids, all_grid_ids=all_grid_ids, repo=self.catalog_repo, thumb_repo=self.thumb_repo)
        self.central_stack.setCurrentWidget(self.compare_view)
        self.ribbon.set_active_mode("compare")
        if hasattr(self.ribbon, "btn_cmp_lock_sync"):
            self.ribbon.btn_cmp_lock_sync.setChecked(self.compare_view.lock_sync)
        if hasattr(self.ribbon, "btn_cmp_peaking"):
            self.ribbon.btn_cmp_peaking.setChecked(self.compare_view.focus_peaking)
        if hasattr(self.ribbon, "btn_cmp_clipping"):
            self.ribbon.btn_cmp_clipping.setChecked(self.compare_view.clipping_warnings)
        if hasattr(self.ribbon, "btn_cmp_auto_advance"):
            self.ribbon.btn_cmp_auto_advance.setChecked(self.compare_view.auto_advance)
        if hasattr(self.ribbon, "btn_cmp_dark_bg"):
            self.ribbon.btn_cmp_dark_bg.setChecked(getattr(self, "_dark_mode", False))
        self.compare_view.setFocus()

    def _on_compare_closed(self, winner_id: Any = None) -> None:
        self.left_nav_dock.setVisible(True)
        self.right_tag_dock.setVisible(True)
        self.show_grid_view()
        if winner_id is not None and hasattr(self.grid_view, "select_media_id"):
            try:
                self.grid_view.select_media_id(int(winner_id))
            except Exception:
                pass

    def _on_compare_open_editor(self, media_id: int) -> None:
        self.left_nav_dock.setVisible(True)
        self.right_tag_dock.setVisible(True)
        self.open_single_viewer(int(media_id))

    def _on_nav_section_order_changed(self, order: list[str]) -> None:
        try:
            settings = QSettings("PhotoGallery64", "PhotoGallery64")
            settings.setValue("nav_section_order", json.dumps(order))
        except Exception:
            pass

    def _on_directory_person_selected(self, person_id: int) -> None:
        """Navigates from People Directory into a single person's photo timeline."""
        self._active_map_media_ids = None
        if hasattr(self, "left_nav_dock"):
            self.left_nav_dock.select_person(person_id)

    def _on_directory_unnamed_selected(self, face_ids: list[int], thumb_blob: bytes | None) -> None:
        """Navigates from People Directory to view photos containing this unnamed person."""
        if not face_ids:
            return

        # Clear any prior person/folder/tag/rating/search filters so the unnamed person's photos are not filtered out
        if hasattr(self, "left_nav_dock") and self.left_nav_dock is not None:
            self.left_nav_dock.clear_all_filters()
        self._search_text = ""
        self._active_bbox = None
        if hasattr(self, "scope_bar") and hasattr(self.scope_bar, "search_input"):
            self.scope_bar.search_input.clear()

        conn = self.engine.catalog_conn()
        placeholders = ",".join("?" for _ in face_ids)
        rows = conn.execute(
            f"SELECT DISTINCT media_id FROM media_people WHERE id IN ({placeholders});",
            face_ids,
        ).fetchall()
        mids = [r[0] for r in rows]

        self._active_map_media_ids = set(mids)
        self.central_stack.setCurrentWidget(self.grid_view)
        self.ribbon.set_active_mode("grid")
        self.person_header_bar.load_unnamed_person(face_ids, thumb_blob, photo_count=len(mids))
        self.grid_view.person_header_bar.show()
        self.grid_view._update_top_bars_geometry()
        self.rebuild_and_run_query()
        self.scope_bar.set_scope_label("People to Name")
        self.ribbon.set_scope_label("People")
        self._push_nav_state({
            "view": "unnamed_person",
            "face_ids": list(face_ids),
            "thumb_blob": thumb_blob,
            "media_ids": mids,
        })

    def _on_unnamed_person_named(self, name: str, face_ids: list[int]) -> None:
        """Assigns a name to an unnamed person group directly from the Person Header Bar."""
        self.people_directory_view._on_name_cluster(name, face_ids)
        conn = self.engine.catalog_conn()
        p = conn.execute("SELECT id FROM people WHERE name = ? COLLATE NOCASE;", (name.strip(),)).fetchone()
        if p:
            self._active_map_media_ids = None
            self.left_nav_dock.select_person(int(p[0]))

    def _on_person_view_mode_changed(self, mode: str) -> None:
        """Toggles between full photos and face crops in the grid view."""
        self._face_crop_cancel.set()
        if mode == "crops":
            selected_pids = self.left_nav_dock.current_filter_state().get("person_ids", ())
            pid = selected_pids[0] if selected_pids else None
            self.card_delegate.set_face_crop_person_id(pid)
            if pid is not None:
                self._face_crop_cancel = CancellationFlag()
                signals = _FaceCropPreloadSignals()
                signals.crop_ready.connect(self._on_face_crop_ready)
                signals.progress.connect(self._on_face_crop_progress)
                signals.finished.connect(self._on_face_crop_finished)
                worker = _FaceCropPreloadWorker(self.engine, pid, signals, self._face_crop_cancel)
                self.thumbnail_pool.start(worker)
        else:
            self._face_crop_status_text = ""
            self.card_delegate.set_face_crop_person_id(None)
            self._update_worker_monitor()
        self.grid_view._sync_metrics_and_scrollbar()

    def _on_face_crop_ready(self, media_id: int, person_id: int, webp_blob: bytes) -> None:
        """Stores preloaded face crop pixmap into delegate cache and refreshes view."""
        if webp_blob and getattr(self.card_delegate, "face_crop_person_id", None) == person_id:
            pm = QPixmap()
            if pm.loadFromData(webp_blob):
                self.card_delegate._face_crop_cache[(media_id, person_id)] = pm
                self.grid_view.viewport().update()

    def _on_face_crop_progress(self, current: int, total: int) -> None:
        """Updates worker monitor status while face crops are generating."""
        if total > 0 and current <= total:
            self._face_crop_status_text = f"Generating face crops: {current}/{total}"
        else:
            self._face_crop_status_text = ""
        self._update_worker_monitor()

    def _on_face_crop_finished(self) -> None:
        """Clears face crop status text when preload worker completes."""
        self._face_crop_status_text = ""
        self._update_worker_monitor()
        self.grid_view.viewport().update()

    def invalidate_face_crop_ui(
        self, media_id: int | None = None, person_ids: Sequence[int] = ()
    ) -> None:
        """
        Invalidates in-memory face crop caches, refreshes active person header avatar/counts,
        and marks grid view dirty when face regions are modified in Single Image View or Grid View.
        """
        if self.central_stack.currentWidget() == self.media_viewer:
            self._viewer_faces_dirty = True

        if hasattr(self, "card_delegate"):
            if media_id is not None:
                self.card_delegate.invalidate_face_crops(media_id=int(media_id))
            for pid in person_ids:
                if pid and int(pid) > 0:
                    self.card_delegate.invalidate_face_crops(person_id=int(pid))
            if media_id is None and not person_ids:
                self.card_delegate.invalidate_face_crops()

        if hasattr(self, "person_header_bar") and self.person_header_bar.isVisible():
            cur_pid = getattr(self.person_header_bar, "current_person_id", None)
            if cur_pid is not None and int(cur_pid) > 0:
                if not person_ids or int(cur_pid) in {int(p) for p in person_ids if p}:
                    self.person_header_bar.load_person(int(cur_pid))

        active_crop_pid = getattr(self.card_delegate, "face_crop_person_id", None)
        if active_crop_pid is not None and self.central_stack.currentWidget() == self.grid_view:
            self._face_crop_cancel.set()
            self._face_crop_cancel = CancellationFlag()
            signals = _FaceCropPreloadSignals()
            signals.crop_ready.connect(self._on_face_crop_ready)
            signals.progress.connect(self._on_face_crop_progress)
            signals.finished.connect(self._on_face_crop_finished)
            worker = _FaceCropPreloadWorker(
                self.engine, int(active_crop_pid), signals, self._face_crop_cancel
            )
            self.thumbnail_pool.start(worker)
        if hasattr(self, "grid_view"):
            self.grid_view.viewport().update()

    def _on_people_special_filter(self, filter_kind: str) -> None:
        """Filters catalog media by WLPG special people criteria: all_people, detected_untagged, untagged."""
        conn = self.engine.catalog_conn()
        if filter_kind == "all_people":
            rows = conn.execute(
                "SELECT DISTINCT media_id FROM media_people WHERE state IN (0, 1);"
            ).fetchall()
            label = "All tagged or detected"
        elif filter_kind == "detected_untagged":
            rows = conn.execute(
                "SELECT DISTINCT media_id FROM media_people WHERE state = 0;"
            ).fetchall()
            label = "All detected but not tagged"
        elif filter_kind == "untagged":
            rows = conn.execute(
                "SELECT id FROM media WHERE id NOT IN (SELECT DISTINCT media_id FROM media_people WHERE state IN (0, 1));"
            ).fetchall()
            label = "Not tagged or detected"
        else:
            return

        mids = {int(r[0]) for r in rows}
        self._active_special_people_filter = filter_kind
        self._active_special_people_label = label
        self._active_map_media_ids = mids
        if self.central_stack.currentWidget() != self.grid_view:
            self.central_stack.setCurrentWidget(self.grid_view)
            self.ribbon.set_active_mode("grid")
        self.grid_view.set_person_header(None)
        self.card_delegate.set_face_crop_person_id(None)
        folder_name = self.left_nav_dock.selected_folder_name()
        self.scope_bar.set_scope_label(f"Folder: {folder_name}" if folder_name else None)
        self.ribbon.set_scope_label(label)
        self._update_scope_bar_visibility()
        self._push_nav_state({
            "view": "grid_special_people",
            "special_filter": filter_kind,
            "media_ids": list(mids),
        })
        self.rebuild_and_run_query()

    def _on_clear_special_people_filter(self) -> None:
        """Removes the special people filter chip and restores the normal folder/global view."""
        self._active_special_people_filter = None
        self._active_special_people_label = None
        self._active_map_media_ids = None
        folder_name = self.left_nav_dock.selected_folder_name()
        self.scope_bar.set_scope_label(f"Folder: {folder_name}" if folder_name else None)
        self.ribbon.set_scope_label(folder_name or "All photos and videos")
        self._update_scope_bar_visibility()
        self.rebuild_and_run_query()

    def _on_date_filter_selected(self, y: int | None, m: int | None, d: int | None) -> None:
        """Filters media by date taken, ensuring grid view is shown and clearing special filters."""
        if self.central_stack.currentWidget() != self.grid_view:
            self.show_grid_view()
        self._active_special_people_filter = None
        self._active_special_people_label = None
        self._active_map_media_ids = None
        self.left_nav_dock.select_date(y, m, d)

    def _on_rating_filter_selected(self, ratings: Sequence[int]) -> None:
        """Filters media by star rating, ensuring grid view is shown and clearing special filters."""
        if self.central_stack.currentWidget() != self.grid_view:
            self.show_grid_view()
        self._active_special_people_filter = None
        self._active_special_people_label = None
        self._active_map_media_ids = None
        self.left_nav_dock.select_ratings(ratings)

    def on_arrange_changed(self, mode: str, reverse: bool) -> None:
        self._arrange_by = mode  # type: ignore[assignment]
        self._reverse_sort = bool(reverse)
        self._group_by = mode
        self._group_reverse = bool(reverse)
        self.rebuild_and_run_query()

    def set_group_by(self, mode: str, reverse: bool | None = None) -> None:
        self._group_by = str(mode).lower()
        if reverse is not None:
            self._group_reverse = bool(reverse)
        if hasattr(self, "ribbon"):
            self.ribbon._arrange_by = self._group_by
            if reverse is not None:
                self.ribbon._reverse_sort = self._group_reverse
                if hasattr(self.ribbon, "btn_reverse_sort"):
                    self.ribbon.btn_reverse_sort.setChecked(self._group_reverse)
            if hasattr(self.ribbon, "_arrange_buttons"):
                if self._group_by in self.ribbon._arrange_buttons:
                    self.ribbon._arrange_buttons[self._group_by].setChecked(True)
                elif hasattr(self.ribbon, "_arrange_btn_group"):
                    cb = self.ribbon._arrange_btn_group.checkedButton()
                    if cb:
                        self.ribbon._arrange_btn_group.setExclusive(False)
                        cb.setChecked(False)
                        self.ribbon._arrange_btn_group.setExclusive(True)
        self.rebuild_and_run_query()

    def set_sort_by(self, mode: str, reverse: bool | None = None) -> None:
        self._sort_by = str(mode).lower()
        if reverse is not None:
            self._sort_reverse = bool(reverse)
        self.rebuild_and_run_query()

    def _on_details_overlay_changed(self, details: set[str]) -> None:
        self.card_delegate.set_visible_details(details)
        if hasattr(self, "grid_view"):
            self.grid_view._sync_metrics_and_scrollbar()
            self.grid_view.viewport().update()
        self._update_status_grid_button()

    def on_quick_find_person(self, person_id: int) -> None:
        if hasattr(self, "_map_view") and self._map_view is not None and self.central_stack.currentWidget() == self._map_view:
            self.show_grid_view()
        elif hasattr(self, "batch_people_view") and self.central_stack.currentWidget() == self.batch_people_view:
            self.close_batch_people_view()
        elif hasattr(self, "media_viewer") and self.central_stack.currentWidget() == self.media_viewer:
            self.show_grid_view()
        toggle = self.ribbon.is_people_and_mode()
        self.left_nav_dock.select_person(person_id, toggle=toggle)

    def _on_search_text_changed(self, text: str) -> None:
        self._search_text = text
        if hasattr(self, "scope_bar") and self.scope_bar.search_text() != text:
            self.scope_bar.txt_search.blockSignals(True)
            self.scope_bar.txt_search.setText(text)
            self.scope_bar.txt_search.blockSignals(False)
        self.rebuild_and_run_query()

    def rebuild_and_run_query(self, preserve_selection: bool = True) -> list[MediaItemRecord]:
        """Builds FilterQuerySpec, queries catalog.db in <10ms, and updates CatalogListModel."""
        prev_ids = list(self._selected_ids)
        headers_enabled = (self._group_by != "none")
        spec: FilterQuerySpec = self.left_nav_dock.build_filter_spec(
            arrange_by=self._arrange_by,
            reverse_sort=self._reverse_sort,
            search_text=self._search_text,
            bbox=self._active_bbox,
            media_ids=tuple(self._active_map_media_ids) if self._active_map_media_ids else None,
            group_by=self._group_by,
            sort_by=self._sort_by,
            sort_reverse=self._sort_reverse,
            group_reverse=self._group_reverse,
        )
        records = self.catalog_repo.query_media_records(spec)
        self.catalog_model.set_records(records, group_headers_enabled=headers_enabled)

        # Update empty state overlay
        is_empty = (len(records) == 0)
        is_search = bool(self._search_text and self._search_text.strip())
        is_scoped = bool(
            spec.folder_id
            or (hasattr(self, "left_nav_dock") and self.left_nav_dock.selected_folder_name() is not None)
            or spec.include_tag_ids
            or spec.person_ids
            or spec.date_year
            or spec.ratings
            or spec.flagged_only
            or (spec.flag_filter is not None)
            or (spec.media_type != "all")
            or spec.bbox
        )
        self.grid_view.set_empty_state(is_empty=is_empty, is_search=is_search, is_scoped=is_scoped)

        if preserve_selection and prev_ids:
            self.grid_view.set_selected_media_ids(prev_ids)
        else:
            self._update_status_counts()
        self.enqueue_visible_thumbnails(self.grid_view.visible_media_ids(buffer_rows=2))
        return records

    def _ensure_media_path_lookup(self, media_ids: list[int]) -> None:
        missing = [mid for mid in media_ids if mid not in self._media_meta_lookup]
        if not missing:
            return

        # Fast O(1) in-memory resolution using CatalogListModel records + cached folder paths
        folder_cache: dict[int, str] | None = getattr(self, "_folder_path_cache", None)
        if not folder_cache or not self._media_meta_lookup:
            try:
                folder_cache = {
                    int(f["id"]): str(f["path"]).replace("\\", "/")
                    for f in self.catalog_repo.get_folders()
                }
                self._folder_path_cache = folder_cache
            except Exception:
                folder_cache = {}

        still_missing: list[int] = []
        records = self.catalog_model._records
        for mid in missing:
            rec_idx = self.catalog_model.record_index_for_id(mid)
            if rec_idx is not None and 0 <= rec_idx < len(records):
                rec = records[rec_idx]
                fdir = folder_cache.get(rec.folder_id)
                if fdir is not None:
                    self._media_meta_lookup[mid] = (
                        f"{fdir}/{rec.filename}",
                        int(rec.mtime),
                        bool(rec.is_video),
                        int(rec.orientation or 1),
                        int(getattr(rec, "metadata_dirty", 0) or 0),
                    )
                    continue
            still_missing.append(mid)

        if not still_missing:
            return

        conn = self.engine.catalog_conn()
        placeholders = ",".join("?" for _ in still_missing)
        rows = conn.execute(
            f"""
            SELECT m.id, f.path || '/' || m.filename AS full_path, m.mtime, m.is_video, m.orientation, m.metadata_dirty
            FROM media m
            JOIN folders f ON f.id = m.folder_id
            WHERE m.id IN ({placeholders});
            """,
            still_missing,
        ).fetchall()
        for r in rows:
            self._media_meta_lookup[int(r["id"])] = (
                str(r["full_path"]),
                int(r["mtime"]),
                bool(r["is_video"]),
                int(r["orientation"] or 1),
                int(r["metadata_dirty"] or 0),
            )

    def enqueue_visible_thumbnails(self, media_ids: list[int]) -> None:
        """
        Cancels out-of-view pending thumbnail workers and schedules ThumbnailDecodeWorker
        tasks for all visible media_ids not yet in LruCache.
        """
        if self._is_shutting_down or not media_ids:
            return

        self._pending_visible_media_ids = list(media_ids)
        visible_set = set(media_ids)
        for mid, flag in list(self._active_thumb_flags.items()):
            if mid not in visible_set:
                flag.set()
                self._active_thumb_flags.pop(mid, None)

        self._pump_visible_thumbnail_queue()

    def _pump_visible_thumbnail_queue(self) -> None:
        """
        Dispatches pending uncached visible thumbnails into thumbnail_pool,
        maintaining up to 512 concurrent tasks so large viewports and small
        thumbnails are never capped at an arbitrary batch limit with blank cards at the end.
        """
        if self._is_shutting_down or not getattr(self, "_pending_visible_media_ids", None):
            return

        failed = getattr(self, "_failed_thumb_ids", set())
        uncached = [
            mid for mid in self._pending_visible_media_ids
            if mid not in self.pixmap_cache and mid not in self._active_thumb_flags and mid not in failed
        ]
        if not uncached:
            return

        available_slots = max(0, 512 - len(self._active_thumb_flags))
        if available_slots <= 0:
            return

        batch_ids = uncached[:available_slots]
        self._ensure_media_path_lookup(batch_ids)
        for mid in batch_ids:
            if mid in self._active_thumb_flags:
                continue
            meta = self._media_meta_lookup.get(mid)
            if meta is None:
                self._failed_thumb_ids.add(mid)
                continue
            fpath, mtime, is_video, orientation, meta_dirty = meta
            cflag = CancellationFlag()
            self._active_thumb_flags[mid] = cflag
            worker = ThumbnailDecodeWorker(
                self.engine,
                media_id=mid,
                file_path=fpath,
                mtime=mtime,
                is_video=is_video,
                orientation=orientation,
                cancel_flag=cflag,
                metadata_dirty=meta_dirty,
            )
            worker.signals.thumbnail_ready.connect(self.on_thumbnail_decoded)
            worker.signals.error.connect(self.on_thumbnail_decode_error)
            self.thumbnail_pool.start(worker)

    def on_thumbnail_decode_error(self, media_id: int, error_msg: str) -> None:
        """Handles thumbnail decode failures cleanly without wedging active flags or stalling the queue."""
        self._active_thumb_flags.pop(media_id, None)
        self._failed_thumb_ids.add(media_id)
        self._pump_visible_thumbnail_queue()

    def on_thumbnail_decoded(self, media_id: int, qimage: QImage) -> None:
        """
        Slot executed strictly on the GUI thread: converts QImage -> QPixmap,
        stores in LruCache, notifies CatalogListModel to repaint the cell,
        and pumps the next batch of uncached visible items.
        """
        self._active_thumb_flags.pop(media_id, None)
        if qimage.isNull():
            self._failed_thumb_ids.add(media_id)
        else:
            pixmap = QPixmap.fromImage(qimage)
            self.pixmap_cache.put(media_id, pixmap)
            self.catalog_model.notify_pixmap_ready(media_id)

        # Continually pump next batch so no visible thumbnails at the bottom stay blank
        self._pump_visible_thumbnail_queue()

    def _start_async_avatar_backfill(self) -> None:
        """Launches background QRunnable to generate avatar crops for people without blocking startup."""
        if self._is_shutting_down:
            return
        signals = _AvatarBackfillSignals(self)
        signals.finished.connect(self._on_avatar_backfill_completed)
        worker = _AvatarBackfillWorker(self.engine, signals)
        self.scanner_pool.start(worker)

    def _on_avatar_backfill_completed(self, updated_count: int) -> None:
        """Called on GUI thread when avatar backfill produces new thumbnails."""
        if updated_count > 0:
            try:
                self.left_nav_dock.reload_people_tree()
                people = self.catalog_repo.get_people_list()
                self.ribbon.populate_face_carousel(people)
                if hasattr(self, "person_header_bar") and self.person_header_bar.isVisible():
                    cur_pid = getattr(self.person_header_bar, "current_person_id", None)
                    if cur_pid is not None and int(cur_pid) > 0:
                        self.person_header_bar.load_person(int(cur_pid))
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Selection & Instant Metadata Edits
    # -------------------------------------------------------------------------
    def on_grid_selection_changed(
        self, selected_ids: list[int], lead_id: int
    ) -> None:
        self._selected_ids = list(selected_ids)
        self._lead_selected_id = lead_id if lead_id > 0 else (selected_ids[0] if selected_ids else None)
        self._update_status_counts()
        self._update_revert_button_state()

        has_sel = len(self._selected_ids) > 0
        for btn in (
            getattr(self.ribbon, "btn_edit_crop", None),
            getattr(self.ribbon, "btn_edit_red_eye", None),
            getattr(self.ribbon, "btn_edit_retouch", None),
            getattr(self.ribbon, "btn_edit_straighten", None),
            getattr(self.ribbon, "btn_edit_fine_tune", None),
        ):
            if btn is not None:
                btn.setEnabled(has_sel)

        summary = self.catalog_repo.get_selection_summary(self._selected_ids)
        self.right_tag_dock.bind_selection(self._selected_ids, summary)

    def _update_status_counts(self) -> None:
        total = self.catalog_model.total_media_count
        self.right_tag_dock.set_total_view_count(total)
        sel = len(self._selected_ids)
        item_word = "item" if total == 1 else "items"
        if sel > 0:
            self.lbl_status_counts.setText(f"{total} {item_word}, {sel} selected")
        else:
            self.lbl_status_counts.setText(f"{total} {item_word}")

    def on_inline_rating_changed(self, media_id: int, rating: int) -> None:
        cmd = _RatingUndoCommand(self, [media_id], rating)
        self.undo_stack.push(cmd)
        lbl = f"{rating} star{'s' if rating != 1 else ''}" if rating > 0 else "Unrated"
        self.show_status_message(f"✓ Rated item: {lbl}", 3000)

    def on_inline_flag_changed(self, media_id: int, new_flag: int) -> None:
        cmd = _FlagUndoCommand(self, [media_id], new_flag)
        self.undo_stack.push(cmd)
        self.show_status_message("✓ Flagged item" if new_flag else "✓ Unflagged item", 3000)

    def on_inline_caption_changed(self, media_id: int, new_caption: str) -> None:
        cmd = _CaptionUndoCommand(self, [media_id], new_caption)
        self.undo_stack.push(cmd)
        self.show_status_message("✓ Updated caption", 3000)

    def _on_rating_shortcut(self, rating: int) -> None:
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit)):
            return
        if hasattr(self, "compare_view") and self.central_stack.currentWidget() is self.compare_view:
            self.compare_view.rate_focused_pane(rating)
            return
        self.apply_rating_to_selection(rating)

    def _on_zoom_in_shortcut(self) -> None:
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit)):
            return
        self.dispatch_ribbon_action("zoom_in")

    def _on_zoom_out_shortcut(self) -> None:
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit)):
            return
        self.dispatch_ribbon_action("zoom_out")

    def get_active_target_ids(self) -> list[int]:
        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
        ):
            return [self.media_viewer.current_media_id]
        return list(self._selected_ids)

    def apply_rating_to_selection(self, rating: int) -> None:
        target_ids = self.get_active_target_ids()
        if not target_ids:
            return
        cmd = _RatingUndoCommand(self, target_ids, rating)
        self.undo_stack.push(cmd)
        lbl = f"{rating} star{'s' if rating != 1 else ''}" if rating > 0 else "Unrated"
        self.show_status_message(f"✓ Rated {len(target_ids)} item(s): {lbl}", 3000)

    def apply_flag_to_selection(self, flag: int) -> None:
        target_ids = self.get_active_target_ids()
        if not target_ids:
            return
        cmd = _FlagUndoCommand(self, target_ids, flag)
        self.undo_stack.push(cmd)
        self.show_status_message(f"✓ {'Flagged' if flag else 'Unflagged'} {len(target_ids)} item(s)", 3000)

    def apply_caption_to_selection(self, caption: str) -> None:
        target_ids = self.get_active_target_ids()
        if not target_ids:
            return
        cmd = _CaptionUndoCommand(self, target_ids, caption)
        self.undo_stack.push(cmd)
        self.show_status_message(f"✓ Updated caption on {len(target_ids)} item(s)", 3000)

    def apply_geotag_to_selection(self, location_name: str) -> None:
        target_ids = self.get_active_target_ids()
        if not target_ids:
            return
        lat, lon, loc_label = resolve_location_query(location_name)
        cmd = _GeotagUndoCommand(self, target_ids, lat, lon, loc_label)
        self.undo_stack.push(cmd)
        self.show_status_message(f"✓ Tagged location '{loc_label}' on {len(target_ids)} item(s)", 3500)

    def on_selection_tag_added(self, tag_path: str) -> None:
        target_ids = self.get_active_target_ids()
        clean_path = tag_path.strip()
        if not target_ids or not clean_path:
            return
        tid: int | None = None
        if "/" not in clean_path and "|" not in clean_path:
            existing = self.catalog_repo.find_tag_by_leaf_name(clean_path)
            if existing is not None:
                tid = int(existing["id"])
        if tid is None:
            tid = self.catalog_repo.create_tag(clean_path)
        self.catalog_repo.assign_tag(target_ids, tid)
        canon_path = self.catalog_repo.get_tag_full_path(tid) or clean_path
        self._record_mru_descriptive_tag(canon_path)
        for mid in target_ids:
            self._media_meta_lookup.pop(mid, None)
        if getattr(self, "metadata_sync_daemon", None) is not None:
            self.metadata_sync_daemon.notify_dirty()
        self.refresh_navigation_and_autocomplete()
        if self.central_stack.currentWidget() == self.grid_view:
            self.rebuild_and_run_query(preserve_selection=True)
        summary = self.catalog_repo.get_selection_summary(target_ids)
        self.right_tag_dock.bind_selection(target_ids, summary)
        self.show_status_message(f"✓ Added tag '{canon_path}' to {len(target_ids)} item(s)", 3000)

    def on_selection_tag_removed(self, tag_id: int) -> None:
        target_ids = self.get_active_target_ids()
        if not target_ids:
            return
        self.catalog_repo.remove_tag(target_ids, tag_id)
        for mid in target_ids:
            self._media_meta_lookup.pop(mid, None)
        if getattr(self, "metadata_sync_daemon", None) is not None:
            self.metadata_sync_daemon.notify_dirty()
        self.refresh_navigation_and_autocomplete()
        if self.central_stack.currentWidget() == self.grid_view:
            self.rebuild_and_run_query(preserve_selection=True)
        summary = self.catalog_repo.get_selection_summary(target_ids)
        self.right_tag_dock.bind_selection(target_ids, summary)
        self.show_status_message(f"✓ Removed tag from {len(target_ids)} item(s)", 3000)

    def on_selection_tag_assign_to_all(self, tag_id: int) -> None:
        target_ids = self.get_active_target_ids()
        if not target_ids:
            return
        self.catalog_repo.assign_tag(target_ids, tag_id)
        canon_path = self.catalog_repo.get_tag_full_path(tag_id)
        if canon_path:
            self._record_mru_descriptive_tag(canon_path)
        for mid in target_ids:
            self._media_meta_lookup.pop(mid, None)
        if getattr(self, "metadata_sync_daemon", None) is not None:
            self.metadata_sync_daemon.notify_dirty()
        self.refresh_navigation_and_autocomplete()
        if self.central_stack.currentWidget() == self.grid_view:
            self.rebuild_and_run_query(preserve_selection=True)
        summary = self.catalog_repo.get_selection_summary(target_ids)
        self.right_tag_dock.bind_selection(target_ids, summary)
        tag_lbl = f"'{canon_path}'" if canon_path else "tag"
        self.show_status_message(f"✓ Applied {tag_lbl} to {len(target_ids)} item(s)", 3000)

    def on_selection_person_added(self, person_name: str) -> None:
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else list(self._selected_ids)
        )
        if not target_ids or not person_name.strip():
            return
        pids_added: list[int] = []
        for mid in target_ids:
            pid = self.catalog_repo.assign_person(mid, person_name.strip())
            pids_added.append(int(pid))
            self.invalidate_face_crop_ui(media_id=int(mid), person_ids=[int(pid)])
        self.refresh_navigation_and_autocomplete()
        summary = self.catalog_repo.get_selection_summary(target_ids)
        self.right_tag_dock.bind_selection(target_ids, summary)
        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
        ):
            detail = self.catalog_repo.get_media_detail(self.media_viewer.current_media_id)
            self.media_viewer.render_face_regions(detail.get("people", []))
        if getattr(self, "find_people_enabled", True):
            self.start_face_detection()
        self.show_status_message(f"✓ Tagged '{person_name.strip()}' in {len(target_ids)} item(s)", 3000)

    def on_selection_person_removed(self, person_id: int) -> None:
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else list(self._selected_ids)
        )
        if not target_ids:
            return
        for mid in target_ids:
            self.catalog_repo.remove_person_tag(mid, person_id)
            self.invalidate_face_crop_ui(media_id=int(mid), person_ids=[int(person_id)])
        self.refresh_navigation_and_autocomplete()
        summary = self.catalog_repo.get_selection_summary(target_ids)
        self.right_tag_dock.bind_selection(target_ids, summary)
        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
        ):
            detail = self.catalog_repo.get_media_detail(self.media_viewer.current_media_id)
            self.media_viewer.render_face_regions(detail.get("people", []))

    def on_locate_person_face_requested(self, person_id: int, person_name: str) -> None:
        """Enters interactive face positioning on the open photo for the specified person with auto-snap."""
        if self.central_stack.currentWidget() != self.media_viewer:
            sel = self.get_active_target_ids()
            if not sel:
                return
            self.open_single_viewer(sel[0])
        if hasattr(self, "media_viewer") and self.media_viewer is not None:
            mid = self.media_viewer.current_media_id
            best_rect = None
            if mid is not None:
                untagged = self.catalog_repo.get_untagged_faces_for_media(int(mid))
                if untagged:
                    best_u = untagged[0]
                    conn = self.engine.catalog_conn()
                    p_row = conn.execute("SELECT centroid_embedding FROM people WHERE id = ?;", (int(person_id),)).fetchone()
                    if p_row and p_row["centroid_embedding"]:
                        import numpy as np
                        from src.core.face_matcher import deserialize_embedding
                        p_vec = deserialize_embedding(bytes(p_row["centroid_embedding"]))
                        if p_vec.ndim > 1:
                            p_vec = p_vec[0]
                        best_sim = -1.0
                        for u in untagged:
                            if u.get("embedding"):
                                u_vec = deserialize_embedding(bytes(u["embedding"]))
                                if u_vec.ndim > 1:
                                    u_vec = u_vec[0]
                                sim = float(np.dot(p_vec, u_vec))
                                if sim > best_sim:
                                    best_sim = sim
                                    best_u = u
                    best_rect = (
                        float(best_u.get("rect_x") or 0.38),
                        float(best_u.get("rect_y") or 0.30),
                        float(best_u.get("rect_w") or 0.22),
                        float(best_u.get("rect_h") or 0.22),
                    )
            self.media_viewer.start_face_tag_mode_for_person(int(person_id), str(person_name), initial_rect=best_rect)

    def on_link_person_face_requested(self, person_id: int, face_id: int) -> None:
        """Links a boxless person tag directly to a detected face box."""
        self.catalog_repo.assign_face_to_person(int(face_id), int(person_id))
        self.invalidate_face_crop_ui(person_ids=[int(person_id)])
        self.refresh_navigation_and_autocomplete()
        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
        ):
            self.media_viewer.reload_faces(highlight_new=True)
        curr_sel = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else (self.grid_view.selected_media_ids() or list(self._selected_ids))
        )
        if curr_sel and hasattr(self, "right_tag_dock") and self.right_tag_dock is not None:
            summary = self.catalog_repo.get_selection_summary(curr_sel)
            self.right_tag_dock.bind_selection(curr_sel, summary)
        self.statusBar().showMessage("✓ Linked person to face box", 4000)
        self.lbl_sync_status.setText("✓ Linked person to face box")
        if getattr(self, "find_people_enabled", True):
            self.start_face_detection()
        self.show_status_message(f"✓ Tagged '{person_name.strip()}' in {len(target_ids)} item(s)", 3000)

    def _on_dock_person_hovered(self, person_id: int, is_hovered: bool) -> None:
        if self.central_stack.currentWidget() == self.media_viewer:
            self.media_viewer.highlight_person_face(person_id if is_hovered else None)

    def _on_dock_person_search_requested(self, person_id: int, person_name: str = "") -> None:
        if self.central_stack.currentWidget() != self.grid_view:
            self.show_grid_view()
        self.left_nav_dock.show()
        self.left_nav_dock.select_person(person_id)
        self.rebuild_and_run_query()

    def _on_dock_person_view_suggestions_requested(self, person_id: int, *args: Any) -> None:
        self.open_batch_people_view(person_id=person_id)

    def _on_dock_person_tag_as_other_requested(self, person_id: int, old_name: str = "") -> None:
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else list(self._selected_ids)
        )
        if not target_ids:
            return
        people_rows = self.catalog_repo.get_people_list()
        new_name, new_pid = TagSomeoneDialog.prompt(people_rows, self, title="Tag someone")
        if not new_name or not new_name.strip():
            return
        new_name_clean = new_name.strip()
        if new_pid is None or new_pid < 0:
            new_pid = self.catalog_repo.upsert_person(new_name_clean)
        for mid in target_ids:
            detail = self.catalog_repo.get_media_detail(mid)
            existing_rect = None
            for p in detail.get("people", []):
                if p.get("person_id") == person_id or (old_name and p.get("name") == old_name):
                    if p.get("rect_x") is not None:
                        existing_rect = (
                            float(p["rect_x"]),
                            float(p["rect_y"]),
                            float(p["rect_w"]),
                            float(p["rect_h"]),
                        )
                    break
            self.catalog_repo.remove_person_tag(mid, person_id)
            assigned_pid = self.catalog_repo.assign_person(mid, new_name_clean, existing_rect)
            self.invalidate_face_crop_ui(
                media_id=int(mid),
                person_ids=[p for p in (int(person_id), int(assigned_pid)) if p > 0],
            )
        self.refresh_navigation_and_autocomplete()
        summary = self.catalog_repo.get_selection_summary(target_ids)
        self.right_tag_dock.bind_selection(target_ids, summary)
        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
        ):
            detail = self.catalog_repo.get_media_detail(self.media_viewer.current_media_id)
            self.media_viewer.render_face_regions(detail.get("people", []))
        if getattr(self, "find_people_enabled", True):
            self.start_face_detection()
        self.show_status_message(f"✓ Tagged '{person_name.strip()}' in {len(target_ids)} item(s)", 3000)

    def _on_dock_person_confirm_requested(self, person_id: int, face_id: int) -> None:
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else list(self._selected_ids)
        )
        if face_id <= 0 and target_ids and person_id > 0:
            placeholders = ",".join("?" for _ in target_ids)
            row = self.catalog_repo.engine.catalog_conn().execute(
                f"""
                SELECT id FROM media_people
                WHERE media_id IN ({placeholders}) AND (person_id = ? OR suggested_person_id = ?) AND state = 0
                LIMIT 1;
                """,
                [*target_ids, person_id, person_id],
            ).fetchone()
            if row:
                face_id = int(row["id"])
        if face_id > 0 and person_id > 0:
            self.catalog_repo.confirm_faces(face_id, person_id)
        self.refresh_navigation_and_autocomplete()
        if target_ids:
            summary = self.catalog_repo.get_selection_summary(target_ids)
            self.right_tag_dock.bind_selection(target_ids, summary)
        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
        ):
            detail = self.catalog_repo.get_media_detail(self.media_viewer.current_media_id)
            self.media_viewer.render_face_regions(detail.get("people", []))
        if hasattr(self, "person_header_bar") and self.person_header_bar.isVisible():
            if getattr(self.person_header_bar, "current_person_id", None) == person_id:
                self.person_header_bar.load_person(person_id)
        self.grid_view.viewport().update()
        self.lbl_sync_status.setText("✓ Confirmed face suggestion")
        if getattr(self, "find_people_enabled", True):
            self.start_face_detection()
        self.show_status_message(f"✓ Tagged '{person_name.strip()}' in {len(target_ids)} item(s)", 3000)

    def _on_dock_person_reject_suggestion_requested(self, person_id: int, face_id: int) -> None:
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else list(self._selected_ids)
        )
        if face_id <= 0 and target_ids and person_id > 0:
            placeholders = ",".join("?" for _ in target_ids)
            row = self.catalog_repo.engine.catalog_conn().execute(
                f"""
                SELECT id FROM media_people
                WHERE media_id IN ({placeholders}) AND (person_id = ? OR suggested_person_id = ?) AND state = 0
                LIMIT 1;
                """,
                [*target_ids, person_id, person_id],
            ).fetchone()
            if row:
                face_id = int(row["id"])
        if face_id > 0:
            self.catalog_repo.reject_face_suggestion(face_id, person_id if person_id > 0 else None)
        if target_ids:
            summary = self.catalog_repo.get_selection_summary(target_ids)
            self.right_tag_dock.bind_selection(target_ids, summary)
        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
        ):
            detail = self.catalog_repo.get_media_detail(self.media_viewer.current_media_id)
            self.media_viewer.render_face_regions(detail.get("people", []))
        if hasattr(self, "person_header_bar") and self.person_header_bar.isVisible():
            if getattr(self.person_header_bar, "current_person_id", None) == person_id:
                self.person_header_bar.load_person(person_id)
        self.grid_view.viewport().update()
        self.lbl_sync_status.setText("Face suggestion dismissed")

    def _on_dock_face_region_hovered(self, region_id: int | None) -> None:
        if self.central_stack.currentWidget() == self.media_viewer:
            self.media_viewer.highlight_face_region(region_id)

    def _on_assign_face_requested(self, face_id: int, person_name: str) -> None:
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else list(self._selected_ids)
        )
        self.catalog_repo.assign_face_to_person(int(face_id), person_name)
        self.refresh_navigation_and_autocomplete()
        if target_ids:
            summary = self.catalog_repo.get_selection_summary(target_ids)
            self.right_tag_dock.bind_selection(target_ids, summary)
        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
        ):
            detail = self.catalog_repo.get_media_detail(self.media_viewer.current_media_id)
            self.media_viewer.render_face_regions(detail.get("face_regions") or detail.get("people", []))
        self.grid_view.viewport().update()
        self.lbl_sync_status.setText(f"✓ Assigned face to '{person_name}'")
        if getattr(self, "find_people_enabled", True):
            self.start_face_detection()
        self.show_status_message(f"✓ Tagged '{person_name.strip()}' in {len(target_ids)} item(s)", 3000)

    def _on_dismiss_face_requested(self, face_id: int) -> None:
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else list(self._selected_ids)
        )
        self.catalog_repo.dismiss_face_region(int(face_id))
        if target_ids:
            summary = self.catalog_repo.get_selection_summary(target_ids)
            self.right_tag_dock.bind_selection(target_ids, summary)
        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
        ):
            detail = self.catalog_repo.get_media_detail(self.media_viewer.current_media_id)
            self.media_viewer.render_face_regions(detail.get("face_regions") or detail.get("people", []))
        self.grid_view.viewport().update()
        self.lbl_sync_status.setText("Dismissed face box")

    def _on_remove_face_requested(self, face_id: int) -> None:
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else list(self._selected_ids)
        )
        self.catalog_repo.remove_face_assignment(int(face_id))
        self.refresh_navigation_and_autocomplete()
        if target_ids:
            summary = self.catalog_repo.get_selection_summary(target_ids)
            self.right_tag_dock.bind_selection(target_ids, summary)
        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
        ):
            detail = self.catalog_repo.get_media_detail(self.media_viewer.current_media_id)
            self.media_viewer.render_face_regions(detail.get("face_regions") or detail.get("people", []))
        self.grid_view.viewport().update()
        self.lbl_sync_status.setText("Removed person from face")

    def _on_face_crop_set_display_pic(
        self, media_id: int, person_id: int, face_id: int | None = None
    ) -> None:
        """Sets cover face / display picture for person_id."""
        if face_id is not None and face_id > 0:
            ok = self.catalog_repo.set_person_display_picture_from_face(person_id, face_id)
        else:
            ok = self.catalog_repo.set_person_display_picture(person_id, media_id)
        if ok:
            if hasattr(self, "person_header_bar") and self.person_header_bar.isVisible():
                if getattr(self.person_header_bar, "current_person_id", None) == person_id:
                    self.person_header_bar.load_person(person_id)
            self.refresh_navigation_and_autocomplete()
            target_ids = [media_id]
            summary = self.catalog_repo.get_selection_summary(target_ids)
            self.right_tag_dock.bind_selection(target_ids, summary)
            self.lbl_sync_status.setText("✓ Updated display picture")

    def _on_face_crop_not_this_person(self, media_ids: list[int], person_id: int) -> None:
        """Removes person tag from specified media and records negative rejection memory."""
        if not media_ids or person_id <= 0:
            return
        conn = self.catalog_repo.engine.catalog_conn()
        placeholders = ",".join("?" for _ in media_ids)
        rows = conn.execute(
            f"""
            SELECT id, media_id FROM media_people
            WHERE media_id IN ({placeholders}) AND person_id = ?;
            """,
            [*media_ids, person_id],
        ).fetchall()
        for r in rows:
            mp_id = int(r["id"])
            mid = int(r["media_id"])
            self.catalog_repo.add_face_rejection(mp_id, person_id)
            self.catalog_repo.remove_person_tag(mid, person_id)
        for mid in media_ids:
            self.invalidate_face_crop_ui(media_id=int(mid), person_ids=[int(person_id)])
        self.refresh_navigation_and_autocomplete()
        self.rebuild_and_run_query(preserve_selection=False)
        if hasattr(self, "person_header_bar") and self.person_header_bar.isVisible():
            if getattr(self.person_header_bar, "current_person_id", None) == person_id:
                self.person_header_bar.load_person(person_id)
        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
        ):
            detail = self.catalog_repo.get_media_detail(self.media_viewer.current_media_id)
            self.media_viewer.render_face_regions(detail.get("people", []))
        self.lbl_sync_status.setText(f"✓ Removed person tag from {len(media_ids)} item(s)")

    def _on_face_crop_not_a_face(
        self,
        media_ids: list[int],
        person_id: int | None = None,
        face_ids: list[int] | None = None,
    ) -> None:
        """Marks face regions as 'Not a face / Delete region' (state = -1)."""
        target_face_ids = list(face_ids or [])
        if not target_face_ids and media_ids:
            conn = self.catalog_repo.engine.catalog_conn()
            placeholders = ",".join("?" for _ in media_ids)
            if person_id is not None and person_id > 0:
                rows = conn.execute(
                    f"SELECT id FROM media_people WHERE media_id IN ({placeholders}) AND person_id = ?;",
                    [*media_ids, person_id],
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT id FROM media_people WHERE media_id IN ({placeholders});",
                    list(media_ids),
                ).fetchall()
            target_face_ids = [int(r["id"]) for r in rows]
        if target_face_ids:
            self.catalog_repo.mark_not_a_face(target_face_ids)
            if person_id is not None and person_id > 0:
                with self.catalog_repo.engine.write_lock:
                    conn = self.catalog_repo.engine.catalog_conn()
                    conn.execute(
                        """
                        UPDATE people
                        SET confirmed_count = (
                            SELECT COUNT(*) FROM media_people WHERE person_id = people.id AND state = 1
                        )
                        WHERE id = ?;
                        """,
                        (person_id,),
                    )
                    conn.commit()
        for mid in media_ids:
            self.invalidate_face_crop_ui(
                media_id=int(mid),
                person_ids=[int(person_id)] if person_id and person_id > 0 else [],
            )
        self.refresh_navigation_and_autocomplete()
        self.rebuild_and_run_query(preserve_selection=False)
        if hasattr(self, "person_header_bar") and self.person_header_bar.isVisible():
            if person_id and getattr(self.person_header_bar, "current_person_id", None) == person_id:
                self.person_header_bar.load_person(person_id)
        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
        ):
            detail = self.catalog_repo.get_media_detail(self.media_viewer.current_media_id)
            self.media_viewer.render_face_regions(detail.get("people", []))
        self.lbl_sync_status.setText(f"✓ Marked {len(target_face_ids)} face region(s) as 'Not a face'")

    def _on_face_crop_tag_as_someone_else(self, media_ids: list[int], person_id: int) -> None:
        """Prompts to tag selected faces to another person contact."""
        if not media_ids:
            return
        people_rows = self.catalog_repo.get_people_list()
        new_name, new_pid = TagSomeoneDialog.prompt(people_rows, self, title="Tag someone")
        if not new_name or not new_name.strip():
            return
        new_name_clean = new_name.strip()
        if new_pid is None or new_pid < 0:
            new_pid = self.catalog_repo.upsert_person(new_name_clean)
        for mid in media_ids:
            detail = self.catalog_repo.get_media_detail(mid)
            existing_rect = None
            for p in detail.get("people", []):
                if p.get("person_id") == person_id:
                    if p.get("rect_x") is not None:
                        existing_rect = (
                            float(p["rect_x"]),
                            float(p["rect_y"]),
                            float(p["rect_w"]),
                            float(p["rect_h"]),
                        )
                    break
            self.catalog_repo.remove_person_tag(mid, person_id)
            assigned_pid = self.catalog_repo.assign_person(mid, new_name_clean, existing_rect)
            self.invalidate_face_crop_ui(
                media_id=int(mid),
                person_ids=[p for p in (int(person_id), int(assigned_pid)) if p > 0],
            )
        self.refresh_navigation_and_autocomplete()
        self.rebuild_and_run_query(preserve_selection=False)
        if hasattr(self, "person_header_bar") and self.person_header_bar.isVisible():
            if getattr(self.person_header_bar, "current_person_id", None) == person_id:
                self.person_header_bar.load_person(person_id)
        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
        ):
            detail = self.catalog_repo.get_media_detail(self.media_viewer.current_media_id)
            self.media_viewer.render_face_regions(detail.get("people", []))
        self.lbl_sync_status.setText(f"✓ Retagged {len(media_ids)} item(s) to '{new_name_clean}'")

    def _on_dock_person_set_display_pic_requested(self, person_id: int) -> None:
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else list(self._selected_ids)
        )
        if not target_ids:
            return
        mid = target_ids[0]
        self._on_face_crop_set_display_pic(mid, person_id)

    def _on_dock_person_merge_requested(self, source_person_id: int, source_name: str) -> None:
        people_rows = self.catalog_repo.get_people_list()
        candidates = [
            dict(r)
            for r in people_rows
            if int(r["id"]) != int(source_person_id)
        ]
        if not candidates:
            QMessageBox.information(
                self,
                "Merge People",
                "There are no other people contacts in the catalog to merge with.",
            )
            return

        src_row = next((r for r in people_rows if int(r["id"]) == int(source_person_id)), None)
        src_thumb = src_row.get("thumbnail_blob") if src_row else None
        src_cnt = src_row.get("confirmed_count", 0) if src_row else 0

        from src.ui.people_directory_view import MergePeopleDialog
        dlg = MergePeopleDialog(
            source_person_name=source_name,
            candidates=candidates,
            source_person_id=source_person_id,
            source_photo_count=src_cnt,
            source_thumb_blob=src_thumb,
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return

        if dlg.is_rename():
            target_name = dlg.target_person_name()
            target_id = self.catalog_repo.rename_person(source_person_id, target_name)
        else:
            target_id = dlg.target_person_id()
            target_name = dlg.target_person_name()
            if target_id is None:
                return
            self.catalog_repo.merge_people(source_person_id, target_id)

        self.refresh_navigation_and_autocomplete()
        if hasattr(self, "people_directory_view"):
            self.people_directory_view.refresh_directory()
        if hasattr(self, "left_nav_dock"):
            curr_pids = self.left_nav_dock.current_filter_state().get("person_ids", ())
            if source_person_id in curr_pids:
                self.left_nav_dock.select_person(target_id)
        if hasattr(self, "person_header_bar") and self.person_header_bar.current_person_id == source_person_id:
            self.person_header_bar.load_person(target_id)
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else list(self._selected_ids)
        )
        if target_ids:
            summary = self.catalog_repo.get_selection_summary(target_ids)
            self.right_tag_dock.bind_selection(target_ids, summary)
        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
        ):
            detail = self.catalog_repo.get_media_detail(self.media_viewer.current_media_id)
            self.media_viewer.render_face_regions(detail.get("people", []))
        self.lbl_sync_status.setText(f"✓ Merged '{source_name}' into '{target_name}'")

    def _on_dock_person_rename_requested(self, person_id: int, old_name: str) -> None:
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Contact",
            "Enter new name for contact:",
            text=old_name,
        )
        if not ok or not new_name.strip() or new_name.strip() == old_name:
            return
        actual_id = self.catalog_repo.rename_person(person_id, new_name.strip())
        self.refresh_navigation_and_autocomplete()
        if actual_id != person_id:
            if hasattr(self, "left_nav_dock"):
                curr_pids = self.left_nav_dock.current_filter_state().get("person_ids", ())
                if person_id in curr_pids:
                    self.left_nav_dock.select_person(actual_id)
            if hasattr(self, "person_header_bar") and self.person_header_bar.current_person_id == person_id:
                self.person_header_bar.load_person(actual_id)
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else list(self._selected_ids)
        )
        if target_ids:
            summary = self.catalog_repo.get_selection_summary(target_ids)
            self.right_tag_dock.bind_selection(target_ids, summary)
        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
        ):
            detail = self.catalog_repo.get_media_detail(self.media_viewer.current_media_id)
            self.media_viewer.render_face_regions(detail.get("people", []))
        self.lbl_sync_status.setText(f"✓ Renamed '{old_name}' to '{new_name.strip()}'")

    def _on_dock_person_tag_mode_requested(self) -> None:
        """Triggered when user clicks 'Add people tags' in Right Dock or 'People tag' on Ribbon."""
        if self.central_stack.currentWidget() != self.media_viewer:
            sel = self.grid_view.selected_media_ids() or list(self._selected_ids)
            if not sel:
                return
            if not self.open_single_viewer(sel[0]):
                return
        self.media_viewer.start_face_tagging_mode()

    def on_selection_filename_committed(self, new_fname: str) -> None:
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else list(self._selected_ids)
        )
        if len(target_ids) != 1:
            return
        mid = target_ids[0]
        new_fname = new_fname.strip()
        if not new_fname:
            return
        try:
            detail = self.catalog_repo.get_media_detail(mid)
            old_fname = str(detail["filename"])
            if old_fname == new_fname:
                return
            folder_p = Path(detail["folder_path"])
            old_path = folder_p / old_fname
            new_path = folder_p / new_fname
            if new_path.exists():
                QMessageBox.warning(self, "File Exists", f"A file named '{new_fname}' already exists.")
                summary = self.catalog_repo.get_selection_summary(target_ids)
                self.right_tag_dock.bind_selection(target_ids, summary)
                return
            if old_path.exists():
                old_path.rename(new_path)
            self.catalog_repo.rename_media(mid, new_fname)
            self._media_meta_lookup.pop(mid, None)
            self.rebuild_and_run_query()
            summary = self.catalog_repo.get_selection_summary(target_ids)
            self.right_tag_dock.bind_selection(target_ids, summary)
            self.lbl_sync_status.setText(f"✓ Renamed file to '{new_fname}'")
        except Exception as e:
            QMessageBox.warning(self, "Rename Error", f"Could not rename file: {e}")

    def on_selection_date_taken_edited(self, new_ts: int) -> None:
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else list(self._selected_ids)
        )
        if not target_ids:
            return
        if len(target_ids) == 1:
            self.catalog_repo.set_date_taken(target_ids, new_ts)
        else:
            lead_id = self._lead_selected_id or target_ids[0]
            self.catalog_repo.apply_relative_time_shift(target_ids, lead_id, new_ts)
        self.refresh_navigation_and_autocomplete()
        self.rebuild_and_run_query()
        summary = self.catalog_repo.get_selection_summary(target_ids)
        self.right_tag_dock.bind_selection(target_ids, summary)
        self.lbl_sync_status.setText(f"✓ Updated Date taken for {len(target_ids)} file(s)")

    def on_selection_author_committed(self, new_author: str) -> None:
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else list(self._selected_ids)
        )
        if not target_ids:
            return
        self.catalog_repo.set_author(target_ids, new_author.strip())
        self.rebuild_and_run_query()
        summary = self.catalog_repo.get_selection_summary(target_ids)
        self.right_tag_dock.bind_selection(target_ids, summary)
        self.lbl_sync_status.setText(f"✓ Updated Author for {len(target_ids)} file(s)")

    def _on_viewer_face_region_tagged(
        self,
        media_id: int,
        person_name: str,
        rx: float,
        ry: float,
        rw: float,
        rh: float,
    ) -> None:
        pid = self.catalog_repo.assign_person(media_id, person_name, (rx, ry, rw, rh))
        self.invalidate_face_crop_ui(media_id=int(media_id), person_ids=[int(pid)])
        self.refresh_navigation_and_autocomplete()
        detail = self.catalog_repo.get_media_detail(media_id)
        self.media_viewer.render_face_regions(detail.get("people", []))
        target_ids = (
            [media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            else list(self._selected_ids)
        )
        if media_id in target_ids:
            summary = self.catalog_repo.get_selection_summary(target_ids)
            self.right_tag_dock.bind_selection(target_ids, summary)
        if getattr(self, "find_people_enabled", True):
            self.start_face_detection()
        self.show_status_message(f"✓ Tagged '{person_name.strip()}' in {len(target_ids)} item(s)", 3000)

    # -------------------------------------------------------------------------
    # Relative 'Change Time Taken' Dialog Integration
    # -------------------------------------------------------------------------
    def open_change_time_taken_dialog(
        self, auto_accept_timestamp: int | None = None
    ) -> ChangeTimeTakenDialog | None:
        if not self._selected_ids:
            return None
        lead_id = self._lead_selected_id or self._selected_ids[0]
        lead_detail = self.catalog_repo.get_media_detail(lead_id)
        lead_dt = int(
            lead_detail["date_taken"]
            if lead_detail["date_taken"] is not None
            else lead_detail["mtime"]
        )
        dlg = ChangeTimeTakenDialog(
            lead_filename=str(lead_detail["filename"]),
            lead_date_taken=lead_dt,
            selected_count=len(self._selected_ids),
            parent=self,
        )
        if auto_accept_timestamp is not None:
            self.catalog_repo.apply_relative_time_shift(
                self._selected_ids, lead_id, int(auto_accept_timestamp)
            )
            self.refresh_navigation_and_autocomplete()
            self.rebuild_and_run_query()
            summary = self.catalog_repo.get_selection_summary(self._selected_ids)
            self.right_tag_dock.bind_selection(self._selected_ids, summary)
            return dlg

        if dlg.exec() == ChangeTimeTakenDialog.Accepted:
            self.catalog_repo.apply_relative_time_shift(
                self._selected_ids, lead_id, dlg.new_lead_timestamp()
            )
            self.refresh_navigation_and_autocomplete()
            self.rebuild_and_run_query()
            summary = self.catalog_repo.get_selection_summary(self._selected_ids)
            self.right_tag_dock.bind_selection(self._selected_ids, summary)
        return dlg

    # -------------------------------------------------------------------------
    # View Switching (Grid <-> Single Media Viewer <-> World Map)
    # -------------------------------------------------------------------------
    def open_single_viewer(self, media_id: int) -> bool:
        try:
            detail = self.catalog_repo.get_media_detail(media_id)
        except KeyError:
            return False
        full_path = Path(detail["folder_path"]) / str(detail["filename"])
        ok = self.media_viewer.load_media(
            media_id=media_id,
            file_path=full_path,
            is_video=bool(detail["is_video"]),
            people_regions=detail.get("people", []),
            orientation=int(detail.get("orientation") or 1),
            metadata_dirty=int(detail.get("metadata_dirty") or 0),
        )
        if ok:
            if self.central_stack.currentWidget() != self.media_viewer:
                self._prev_central_widget = self.central_stack.currentWidget()
                curr_tab = self.ribbon.tab_widget.currentIndex()
                batch_tab_idx = getattr(self.ribbon, "tab_batch_people_index", 5)
                self._prev_ribbon_tab = 0 if curr_tab == batch_tab_idx else curr_tab
                self._viewer_faces_dirty = False
                if self.isVisible() and self.central_stack.currentWidget() == self.grid_view:
                    self._nav_dock_was_visible = not self.left_nav_dock.isHidden()
                else:
                    self._nav_dock_was_visible = True
                self.left_nav_dock.hide()
            self.central_stack.setCurrentWidget(self.media_viewer)
            if self.centralWidget() is not None and self.centralWidget().layout() is not None:
                self.centralWidget().layout().activate()
            if not bool(detail["is_video"]):
                self.media_viewer.set_zoom_mode("fit")
            self.media_viewer.set_dark_mode(getattr(self, "current_theme", "") == "dark")
            self.ribbon.set_active_mode("viewer")
            self.ribbon.tab_widget.setCurrentWidget(self.ribbon.tab_edit)
            self.right_tag_dock.show()
            self.right_tag_dock.setVisible(True)
            self.right_tag_dock.set_view_mode("viewer")
            self._update_undo_redo_enabled()
            if hasattr(self, "scope_bar"):
                self.scope_bar.hide()
            self.btn_status_prev.setVisible(True)
            self.btn_status_next.setVisible(True)
            self.btn_status_fit.setVisible(True)
            self._update_status_zoom_toggle_button()
            self._update_status_grid_button()

            fname = str(detail["filename"])
            self.setWindowTitle(f"{fname} - Essentials Photo Gallery")
            if hasattr(self.ribbon, "lbl_app_title"):
                self.ribbon.lbl_app_title.setText(f"{fname} - Essentials Photo Gallery")

            self._selected_ids = [media_id]
            self._lead_selected_id = media_id
            summary = self.catalog_repo.get_selection_summary([media_id])
            self.right_tag_dock.bind_selection([media_id], summary)

            if hasattr(self.ribbon, "set_flag_state"):
                self.ribbon.set_flag_state(bool(detail.get("flag", 0)))

            if hasattr(self.ribbon, "btn_toggle_tag_pane"):
                self.ribbon.btn_toggle_tag_pane.setChecked(self.right_tag_dock.isVisible())

            records = self.catalog_model._records
            pos = self.catalog_model.record_index_for_id(media_id)
            if pos is not None and records:
                self.lbl_status_counts.setText(f"{pos + 1} of {len(records)}")

            if (
                self.right_tag_dock.isVisible()
                and self.right_tag_dock.current_mode() == "fine_tune"
            ):
                self.right_tag_dock.sync_recipe(self.media_viewer.current_recipe)
                self.right_tag_dock.histogram_widget.set_histogram_data(
                    self.media_viewer.current_pil_image
                )

            self._refresh_effects_gallery_thumbnails()
            self._update_revert_button_state()
            self.media_viewer.graphics_view.setFocus()
            if not self._is_navigating_history:
                if 0 <= self._nav_index < len(self._nav_history) and self._nav_history[self._nav_index].get("view") == "viewer":
                    self._nav_history[self._nav_index]["media_id"] = media_id
                else:
                    self._push_nav_state({"view": "viewer", "media_id": media_id})
        return ok

    def _refresh_effects_gallery_thumbnails(self) -> None:
        if hasattr(self.ribbon, "effects_gallery") and self.central_stack.currentWidget() == self.media_viewer:
            pixmaps = self.media_viewer.get_effect_preview_pixmaps(53, 64)
            if pixmaps:
                self.ribbon.effects_gallery.update_live_thumbnails(pixmaps)

    def open_file_on_startup(self, file_path: Path | str) -> bool:
        """
        Handles Windows 'Open with...' command-line invocation:
        1. Resolves path and parent directory.
        2. Fast-ingests or retrieves media_id from catalog.db with zero lag.
        3. Opens SingleMediaViewer directly displaying the photo or video.
        4. Queues background scan on parent folder so Next/Previous arrow navigation works.
        """
        target_file = Path(file_path).resolve()
        if not target_file.is_file():
            return False

        parent_dir = target_file.parent
        mid = self.catalog_repo.get_media_id_by_file_path(target_file)

        if mid is None:
            try:
                from src.workers.scanner_thumb_worker import extract_media_metadata
                is_vid = target_file.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v"}
                meta = extract_media_metadata(target_file, is_video=is_vid)
                folder_id = self.catalog_repo.upsert_folder(
                    str(parent_dir).replace("\\", "/"),
                    parent_dir.name,
                    None,
                    int(parent_dir.stat().st_mtime) if parent_dir.exists() else 0,
                )
                meta["folder_id"] = folder_id
                meta["filename"] = target_file.name
                meta["ext"] = target_file.suffix.lstrip(".").lower()
                meta["size"] = target_file.stat().st_size
                meta["mtime"] = int(target_file.stat().st_mtime)
                mids = self.catalog_repo.bulk_upsert_media([meta])
                if mids:
                    mid = mids[0]
            except Exception as e:
                logger.warning("Error fast-ingesting startup file '%s': %s", target_file, e)

        # Trigger folder scan in background so neighboring photos load for next/prev navigation
        if parent_dir.exists():
            self.scan_folder(parent_dir, include_subfolders=False)

        if mid is not None:
            return self.open_single_viewer(mid)
        return False

    def show_grid_view(self) -> None:
        if hasattr(self, "duplicate_finder_view") and self.central_stack.currentWidget() == self.duplicate_finder_view:
            self.duplicate_finder_view.stop_scan()

        if self.central_stack.currentWidget() == self.media_viewer and self.media_viewer.has_unsaved_adjustments():
            if not self.media_viewer.prompt_save_changes_if_needed("close"):
                return
            self.rebuild_and_run_query()
        self.media_viewer.media_player.stop()
        self.media_viewer._stop_animated_image()
        self.media_viewer.set_markup_mode(False)
        self.media_viewer.set_retouch_mode(False)
        self.media_viewer.set_red_eye_mode(False)
        self.media_viewer.set_crop_mode(False)

        prev_widget = getattr(self, "_prev_central_widget", None)
        self._prev_central_widget = None
        if hasattr(self, "batch_people_view") and prev_widget == self.batch_people_view:
            self.central_stack.setCurrentWidget(self.batch_people_view)
            self.ribbon.set_active_mode("batch_people")
            self._update_undo_redo_enabled()
            self.left_nav_dock.hide()
            self.right_tag_dock.hide()
            self.btn_status_prev.setVisible(False)
            self.btn_status_next.setVisible(False)
            self.btn_status_fit.setVisible(False)
            self.btn_status_actual.setVisible(False)
            self.setWindowTitle("Batch People Tagging - Essentials Photo Gallery")
            if hasattr(self.ribbon, "lbl_app_title"):
                self.ribbon.lbl_app_title.setText("Essentials Photo Gallery")
            self.lbl_sync_status.setText("Batch people tag view")
            if not self._is_navigating_history:
                pid = getattr(self.batch_people_view, "_filter_person_id", None)
                self._push_nav_state({"view": "batch_people", "person_id": pid})
            return

        if hasattr(self, "_map_view") and self._map_view is not None and prev_widget == self._map_view:
            self.show_map_view()
            return

        if hasattr(self, "batch_people_view") and self.central_stack.currentWidget() == self.batch_people_view:
            self.close_batch_people_view()
            return

        self.central_stack.setCurrentWidget(self.grid_view)
        if hasattr(self, "batch_people_view"):
            self.batch_people_view.set_use_ribbon_mode(False)
        self.ribbon.set_active_mode("grid")
        self.right_tag_dock.set_view_mode("grid")
        self._update_undo_redo_enabled()
        if hasattr(self, "_prev_ribbon_tab"):
            delattr(self, "_prev_ribbon_tab")
        self.ribbon.tab_widget.setCurrentWidget(self.ribbon.tab_home)
        is_nav_vis = getattr(self, "_nav_dock_was_visible", True)
        self.left_nav_dock.setVisible(is_nav_vis)
        if is_nav_vis:
            self.left_nav_dock.show()
            self.left_nav_dock.setMinimumWidth(190)
            self.resizeDocks([self.left_nav_dock], [230], Qt.Horizontal)
        if hasattr(self.ribbon, "btn_toggle_nav_pane"):
            self.ribbon.btn_toggle_nav_pane.setChecked(is_nav_vis)
        self.right_tag_dock.show()
        self.right_tag_dock.setVisible(True)
        self.btn_status_prev.setVisible(False)
        self.btn_status_next.setVisible(False)
        self.btn_status_fit.setVisible(False)
        self.btn_status_actual.setVisible(False)
        self._update_status_grid_button()

        if getattr(self, "_viewer_faces_dirty", False):
            self._viewer_faces_dirty = False
            self.rebuild_and_run_query()
            if hasattr(self, "person_header_bar") and self.person_header_bar.isVisible():
                cur_pid = getattr(self.person_header_bar, "current_person_id", None)
                if cur_pid is not None and int(cur_pid) > 0:
                    self.person_header_bar.load_person(int(cur_pid))
            active_crop_pid = getattr(self.card_delegate, "face_crop_person_id", None)
            if active_crop_pid is not None:
                self.card_delegate.invalidate_face_crops(person_id=int(active_crop_pid))
                self._on_person_view_mode_changed("crops")

        self.setWindowTitle("Essentials Photo Gallery")
        if hasattr(self.ribbon, "lbl_app_title"):
            self.ribbon.lbl_app_title.setText("Essentials Photo Gallery")

        if self.right_tag_dock.current_mode() in ("fine_tune", "depth_light"):
            self.right_tag_dock.set_mode("info")

        if hasattr(self.ribbon, "btn_toggle_tag_pane"):
            self.ribbon.btn_toggle_tag_pane.setChecked(self.right_tag_dock.isVisible())

        self._update_status_counts()
        self._update_scope_bar_visibility()
        self._update_revert_button_state()

        # Rebind selection from grid to right dock
        sel = self.grid_view.selected_media_ids() or list(self._selected_ids)
        summary = self.catalog_repo.get_selection_summary(sel)
        self.right_tag_dock.bind_selection(sel, summary)
        self.grid_view.setFocus()

        # Sync status zoom slider back to current grid thumbnail size
        thumb_sz = getattr(self.grid_view, "_thumb_size", 150)
        self.slider_status_zoom.blockSignals(True)
        self.slider_status_zoom.setValue(thumb_sz)
        self.slider_status_zoom.blockSignals(False)

    @property
    def map_view(self) -> WorldMapView | None:
        return self._map_view

    def _ensure_map_view(self) -> WorldMapView:
        if self._map_view is None:
            self._map_view = WorldMapView(self.central_stack)
            self._map_view.bind_repository(self.catalog_repo, self.thumb_repo)
            self._map_view.close_requested.connect(self.show_grid_view)
            self._map_view.filter_to_bbox_requested.connect(self.on_map_bbox_filter_requested)
            self._map_view.cluster_selected.connect(self._on_map_cluster_selected)
            self._map_view.cluster_open_requested.connect(self._on_map_cluster_clicked_data)
            self._map_view.map_background_clicked.connect(self._on_map_bg_clicked)
            self._map_view.bbox_changed.connect(self._on_map_bbox_changed)
            self._map_view.map_stats_Changed.connect(self._on_map_stats_changed)
            self._map_view.open_media_requested.connect(self.open_single_viewer)
            self.right_tag_dock.map_location_selected.connect(self._on_dock_map_location_selected)
            self.central_stack.addWidget(self._map_view)
        return self._map_view

    def _sync_nav_pane_buttons(self, visible: bool) -> None:
        if hasattr(self.ribbon, "btn_toggle_nav_pane"):
            self.ribbon.btn_toggle_nav_pane.setChecked(bool(visible))
        if hasattr(self.ribbon, "btn_map_nav_pane"):
            self.ribbon.btn_map_nav_pane.setChecked(bool(visible))

    def _sync_tag_pane_buttons(self, visible: bool) -> None:
        if hasattr(self.ribbon, "btn_toggle_tag_pane"):
            self.ribbon.btn_toggle_tag_pane.setChecked(bool(visible))
        if hasattr(self.ribbon, "btn_map_tag_pane"):
            self.ribbon.btn_map_tag_pane.setChecked(bool(visible))

    def _on_map_stats_changed(
        self, geotagged_count: int, cluster_count: int, _total_items: int
    ) -> None:
        if (
            self._map_view is not None
            and self.central_stack.currentWidget() == self._map_view
            and hasattr(self, "lbl_status_counts")
        ):
            suffix = " (filtered)" if self._map_view.has_active_map_filters() else ""
            self.lbl_status_counts.setText(
                f"{geotagged_count:,} geotagged items in {cluster_count:,} clusters{suffix}"
            )

    def _on_ribbon_map_filters_changed(
        self, year: int | None, person_id: int | None, min_rating: int | None
    ) -> None:
        mv = self._ensure_map_view()
        mv.set_map_filters(year=year, person_id=person_id, min_rating=min_rating, fit=False)
        total = sum(c.get("count", 0) for c in mv.current_clusters)
        self._on_map_stats_changed(total, len(mv.current_clusters), total)
        self.right_tag_dock.set_total_view_count(total)
        self.right_tag_dock.set_map_clusters(mv.current_clusters)

    def _on_ribbon_map_location_chosen(self, lat: float, lon: float, _label: str) -> None:
        mv = self._ensure_map_view()
        mv.fly_to(lat, lon, zoom=13)

    def _on_ribbon_map_location_search(self, query: str) -> None:
        mv = self._ensure_map_view()
        mv.fly_to_place(query, zoom=13)

    def show_map_view(self) -> None:
        mv = self._ensure_map_view()
        self.central_stack.setCurrentWidget(mv)
        self.ribbon.set_active_mode("map")
        self.left_nav_dock.hide()
        self.right_tag_dock.hide()
        if hasattr(self.ribbon, "sync_map_pane_buttons"):
            self.ribbon.sync_map_pane_buttons(
                self.left_nav_dock.isVisible(), self.right_tag_dock.isVisible()
            )
        if hasattr(self.ribbon, "sync_map_style_button"):
            self.ribbon.sync_map_style_button(mv.map_style)
        if hasattr(self.ribbon, "populate_map_filters") and hasattr(self, "catalog_repo"):
            try:
                years = self.catalog_repo.get_geotagged_years()
            except Exception:
                years = []
            try:
                people = self.catalog_repo.get_people_list()
            except Exception:
                people = []
            self.ribbon.populate_map_filters(years, people)
        if hasattr(self.ribbon, "set_map_filter_ui"):
            self.ribbon.set_map_filter_ui(*mv.active_map_filters)
        if not getattr(self, "_map_view_initialized", False):
            self._map_view_initialized = True
            mv.fit_to_photos()
        else:
            mv.refresh_current_viewport()
        if getattr(mv, "_web_loaded", False) and getattr(mv, "web_view", None) is not None:
            QTimer.singleShot(60, lambda: mv.web_view.page().runJavaScript("if (typeof map !== 'undefined' && map) { map.invalidateSize(); }"))
        total = sum(c.get("count", 0) for c in mv.current_clusters)
        self._on_map_stats_changed(total, len(mv.current_clusters), total)
        self.right_tag_dock.set_total_view_count(total)
        self.right_tag_dock.set_map_clusters(mv.current_clusters)
        if not self._is_navigating_history:
            self._push_nav_state({"view": "map"})

    def open_batch_people_view(self, person_id: int | None = None) -> None:
        """Toggles between standard grid view and BatchPeopleView."""
        if self.central_stack.currentWidget() == self.batch_people_view and person_id is None:
            self.close_batch_people_view()
            return
        self._batch_people_modified = False
        self.central_stack.setCurrentWidget(self.batch_people_view)
        if hasattr(self, "ribbon"):
            self.ribbon.set_active_mode("batch_people")
        if hasattr(self, "batch_people_view"):
            self.batch_people_view.set_use_ribbon_mode(True)
        # Ensure face detection worker runs if enabled or if no unassigned faces exist yet
        if getattr(self, "find_people_enabled", True) and "PYTEST_CURRENT_TEST" not in os.environ:
            has_pending = bool(self.engine.catalog_conn().execute(
                "SELECT 1 FROM media_people WHERE state = 0 LIMIT 1;"
            ).fetchone())
            force_rescan = (
                not has_pending
                and not getattr(self, "_auto_face_rescan_triggered", False)
            )
            if force_rescan:
                self._auto_face_rescan_triggered = True
            self.start_face_detection(force_rescan=force_rescan)
        if person_id is not None:
            self.batch_people_view.filter_to_person(person_id)
        else:
            self.batch_people_view.refresh_suggestions()
        if hasattr(self, "ribbon") and hasattr(self.ribbon, "update_batch_action_labels") and hasattr(self, "batch_people_view"):
            self.ribbon.update_batch_action_labels(getattr(self.batch_people_view, "_current_view_mode", "suggestions"))
        self.left_nav_dock.hide()
        self.right_tag_dock.hide()
        self.lbl_sync_status.setText("Batch people tag view")
        if not self._is_navigating_history:
            self._push_nav_state({"view": "batch_people", "person_id": person_id})

    def close_batch_people_view(self) -> None:
        """Returns to standard grid view from BatchPeopleView."""
        self.central_stack.setCurrentWidget(self.grid_view)
        if hasattr(self, "ribbon"):
            self.ribbon.set_active_mode("grid")
        self._restore_docks_after_batch_people()
        if getattr(self, "_batch_people_modified", False):
            self._batch_people_modified = False
            self.refresh_people_navigation_and_autocomplete()
            self.rebuild_and_run_query()
        self.lbl_sync_status.setText("Returned to gallery")
        if not self._is_navigating_history:
            filt = self.current_filter.to_dict() if hasattr(self, "current_filter") and self.current_filter else {}
            self._push_nav_state({"view": "grid", "filter": filt})

    def _on_batch_faces_confirmed(self, face_ids: list[int], person_id: int) -> None:
        """Slot called when faces are confirmed in BatchPeopleView."""
        self._batch_people_modified = True
        self.refresh_people_navigation_and_autocomplete()
        if hasattr(self, "people_directory_view") and self.central_stack.currentWidget() == self.people_directory_view:
            self.people_directory_view.refresh_directory()
        if hasattr(self, "grid_view") and getattr(self.grid_view, "person_header_bar", None) is not None:
            if self.grid_view.person_header_bar.isVisible() and self.grid_view.person_header_bar.current_person_id == person_id:
                self.grid_view.person_header_bar.load_person(person_id)
        self.lbl_sync_status.setText(f"✓ Confirmed {len(face_ids)} face(s)")
        if getattr(self, "find_people_enabled", True):
            self.start_face_detection()

    def _on_batch_faces_rejected(self, face_ids: list[int]) -> None:
        """Slot called when faces are rejected in BatchPeopleView."""
        self._batch_people_modified = True
        self.lbl_sync_status.setText(f"Rejected {len(face_ids)} face suggestion(s)")

    def _on_batch_faces_ignored(self, face_ids: list[int]) -> None:
        """Slot called when faces are hidden/ignored in BatchPeopleView."""
        self._batch_people_modified = True

    def _on_batch_person_renamed(self, person_id: int, new_name: str) -> None:
        """Slot called when a person is renamed or merged in BatchPeopleView."""
        self._batch_people_modified = True
        self.refresh_people_navigation_and_autocomplete()

    def _on_batch_card_selection_changed(self, mode: str, swap_name: str | None, person_name: str | None) -> None:
        """Updates ribbon batch actions dynamically when card selection or mode changes."""
        if hasattr(self, "ribbon") and hasattr(self.ribbon, "update_batch_action_labels"):
            self.ribbon.update_batch_action_labels(mode, swap_name, person_name)

    def navigate_single_viewer(self, delta: int) -> bool:
        curr_id = self.media_viewer.current_media_id
        if curr_id is None:
            return False
        if self.central_stack.currentWidget() == self.media_viewer and self.media_viewer.has_unsaved_adjustments():
            if not self.media_viewer.prompt_save_changes_if_needed("navigate"):
                return False
            self.rebuild_and_run_query()
        records = self.catalog_model._records
        pos = self.catalog_model.record_index_for_id(curr_id) if records else None
        if pos is None:
            # Current media is not in the active filtered record list (e.g. startup Open With or stale sub-filter);
            # switch scope to its parent folder (or All Media) so Next/Previous navigation works immediately.
            try:
                det = self.catalog_repo.get_media_detail(curr_id)
                fid = det.get("folder_id")
                if fid is not None and hasattr(self.left_nav_dock, "select_folder"):
                    self.left_nav_dock.select_folder(int(fid))
                else:
                    self.left_nav_dock.clear_all_filters()
                records = self.rebuild_and_run_query()
                pos = self.catalog_model.record_index_for_id(curr_id)
            except Exception:
                pos = None
        if not records or pos is None:
            return False
        next_pos = (pos + delta) % len(records)
        next_id = records[next_pos].id
        self._selected_ids = [next_id]
        self._lead_selected_id = next_id
        summary = self.catalog_repo.get_selection_summary([next_id])
        self.right_tag_dock.bind_selection([next_id], summary)
        ok = self.open_single_viewer(next_id)
        if (
            ok
            and self.right_tag_dock.isVisible()
            and self.right_tag_dock.current_mode() == "fine_tune"
        ):
            self.right_tag_dock.histogram_widget.set_histogram_data(
                self.media_viewer.current_pil_image
            )
        return ok

    def _apply_active_map_filters_to_grid(self) -> bool:
        """Carry active Map Ribbon filters (year, person, rating) into Grid and SearchScopeBar."""
        mv = self._map_view
        if mv is None or not mv.has_active_map_filters():
            return False
        year, person_id, min_rating = mv.active_map_filters
        ratings: tuple[int, ...] = ()
        if min_rating == 0:
            ratings = (0,)
        elif min_rating is not None:
            ratings = tuple(range(min_rating, 6))
        if hasattr(self, "left_nav_dock") and self.left_nav_dock is not None:
            self.left_nav_dock.restore_filter_state(
                {
                    "date_year": year,
                    "person_ids": (person_id,) if person_id is not None else (),
                    "ratings": ratings,
                }
            )
        if hasattr(self, "grid_view") and self.grid_view is not None:
            self.grid_view.set_person_header(person_id)
        if person_id is None and hasattr(self, "card_delegate") and self.card_delegate is not None:
            self.card_delegate.set_face_crop_person_id(None)
        return True

    def on_map_bbox_filter_requested(
        self, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> None:
        self._active_bbox = (min_lat, max_lat, min_lon, max_lon)
        self._active_map_media_ids = None
        self._active_map_location_label = f"Map Area ({min_lat:.1f}°, {min_lon:.1f}° to {max_lat:.1f}°, {max_lon:.1f}°)"
        self.ribbon.set_scope_label(
            f"Map Area ({min_lat:.1f}°..{max_lat:.1f}°, {min_lon:.1f}°..{max_lon:.1f}°)"
        )
        has_map_filters = self._apply_active_map_filters_to_grid()
        self.show_grid_view()
        if has_map_filters:
            self._update_scope_bar_visibility()
        self.rebuild_and_run_query()
        if not self._is_navigating_history:
            self._push_nav_state({
                "view": "grid_map_bbox",
                "bbox": self._active_bbox,
                "location_label": self._active_map_location_label,
                "filter": self.left_nav_dock.current_filter_state(),
            })

    def _on_map_cluster_selected(self, media_ids: list[int], label: str = "") -> None:
        if not media_ids:
            return
        # If huge cluster selection, avoid heavy synchronous full tag dock extraction on UI thread
        if len(media_ids) > 1000:
            self._selected_ids = [int(x) for x in media_ids[:1000]]
            self._lead_selected_id = self._selected_ids[0]
            lead_detail = self.catalog_repo.get_media_detail(self._lead_selected_id)
            summary = {
                "count": len(media_ids),
                "lead": lead_detail,
                "total_size": 0,
                "common_rating": None,
                "common_flag": None,
                "tags": [],
                "people": [],
            }
        else:
            self._selected_ids = [int(x) for x in media_ids]
            self._lead_selected_id = self._selected_ids[0]
            summary = self.catalog_repo.get_selection_summary(self._selected_ids)
        self.right_tag_dock.bind_selection(self._selected_ids, summary)

    def _on_map_bg_clicked(self) -> None:
        self._selected_ids = []
        self._lead_selected_id = None
        self.right_tag_dock.bind_selection([], {"count": 0})
        if self._map_view is not None:
            total = sum(c.get("count", 0) for c in self._map_view.current_clusters)
            self.right_tag_dock.set_total_view_count(total)

    def _on_map_bbox_changed(
        self, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> None:
        if self._map_view is not None:
            clusters = self._map_view.current_clusters
            total = sum(c.get("count", 0) for c in clusters)
            self.right_tag_dock.set_total_view_count(total)
            self.right_tag_dock.set_map_clusters(clusters)

    def _on_dock_map_location_selected(
        self, media_ids: list[int], label: str, lat: float, lon: float
    ) -> None:
        self._on_map_cluster_selected(media_ids, label)
        if self._map_view is not None:
            self._map_view.fly_to(lat, lon, zoom=13)

    def _on_map_cluster_clicked(self, media_ids: list[int]) -> None:
        if not media_ids:
            return
        if len(media_ids) == 1:
            self.open_single_viewer(int(media_ids[0]))
        else:
            self.filter_gallery_to_map_selection(media_ids)

    def _on_map_cluster_clicked_data(self, media_ids: list[int], label: str = "") -> None:
        if not media_ids:
            return
        if len(media_ids) == 1:
            self.open_single_viewer(int(media_ids[0]))
        else:
            loc_label = label or f"{len(media_ids)} items"
            self.filter_gallery_to_map_selection(media_ids, loc_label)

    def _on_place_selected(self, country: str, state: str, city: str) -> None:
        """Flies to place on map or filters gallery to location."""
        target_name = city or state or country
        if not target_name:
            return
        if (
            self._map_view is not None
            and self.central_stack.currentWidget() == self._map_view
        ):
            from src.core.geotag_resolver import search_places_offline

            places = search_places_offline(target_name, limit=3)
            if places:
                p = places[0]
                self._map_view.fly_to(
                    float(p["lat"]), float(p["lon"]), zoom=13 if city else 7
                )
            else:
                if hasattr(self.ribbon, "map_search_box"):
                    self.ribbon.map_search_box.setText(target_name)
        else:
            self._search_text = target_name
            if hasattr(self, "search_box") and self.search_box is not None:
                self.search_box.setText(target_name)
            self.rebuild_and_run_query()

    def filter_gallery_to_map_selection(
        self, media_ids: list[int], location_label: str = ""
    ) -> None:
        if hasattr(self, "left_nav_dock") and self.left_nav_dock is not None:
            self.left_nav_dock.clear_all_filters()
        self._search_text = ""
        if hasattr(self, "search_box") and self.search_box is not None:
            self.search_box.clear()
        if hasattr(self, "ribbon") and hasattr(self.ribbon, "search_input") and self.ribbon.search_input is not None:
            self.ribbon.search_input.clear()
        self._active_tag_ids = []
        self._active_person_ids = []
        self._active_date_filter = None
        self._active_rating_filter = 0
        self._active_flag_filter = False
        has_map_filters = self._apply_active_map_filters_to_grid()

        self._active_map_media_ids = list(media_ids)
        self._active_bbox = None
        count = len(media_ids)
        display_label = (
            f"📍 {location_label} ({count} items)"
            if location_label and f"{count}" not in location_label
            else f"📍 {location_label or f'{count} items'}"
        )
        self._active_map_location_label = display_label
        self.ribbon.set_scope_label(display_label)
        self.show_grid_view()
        if has_map_filters:
            self._update_scope_bar_visibility()
        self.rebuild_and_run_query()
        if not self._is_navigating_history:
            self._push_nav_state({
                "view": "grid_map_selection",
                "media_ids": list(media_ids),
                "location_label": display_label,
                "filter": self.left_nav_dock.current_filter_state(),
            })

    def _on_analyze_noise_requested(self) -> None:
        """Runs the MAD noise profile analysis on the active photo and updates the denoise slider."""
        if self.media_viewer.master_np_rgb is not None:
            from src.imaging.opencv_engine import analyze_noise_profile
            rec_denoise = analyze_noise_profile(self.media_viewer.master_np_rgb)
            self.right_tag_dock.set_denoise_analyzed_value(rec_denoise)
            self.media_viewer.set_detail(
                int(self.media_viewer.current_recipe.sharpen),
                int(round(rec_denoise)),
            )

    def _batch_apply_image_adjustment(
        self, media_ids: list[int], action_name: str, adjust_fn: Any
    ) -> None:
        """
        Applies a non-destructive image adjustment to 1 or N photos in batch:
        1. Backs up original to .OriginalFiles/<filename> if not already backed up
        2. Adjusts pixels via OpenCV/PIL pipeline
        3. Preserves metadata (EXIF/XMP)
        4. Invalidates thumbnails and refreshes grid
        """
        if not media_ids:
            return
        valid_ids = []
        for mid in media_ids:
            detail = self.catalog_repo.get_media_detail(mid)
            if detail and not detail.get("is_video"):
                valid_ids.append(mid)
        if not valid_ids:
            return

        total = len(valid_ids)
        self.show_status_message(f"Applying {action_name} to {total} photo(s)...", 3000)
        import shutil
        import numpy as np
        from PIL import Image

        for idx, mid in enumerate(valid_ids):
            detail = self.catalog_repo.get_media_detail(mid)
            if not detail:
                continue
            folder_p = detail.get("folder_path")
            fname = detail.get("filename")
            if not folder_p or not fname:
                continue
            fpath = Path(folder_p) / fname
            if not fpath.exists():
                continue

            # Ensure .OriginalFiles backup exists
            orig_dir = fpath.parent / ".OriginalFiles"
            orig_dir.mkdir(parents=True, exist_ok=True)
            backup_path = orig_dir / fpath.name
            if not backup_path.exists():
                shutil.copy2(fpath, backup_path)

            try:
                with Image.open(fpath) as pil_img:
                    exif_bytes = pil_img.info.get("exif")
                    icc_profile = pil_img.info.get("icc_profile")
                    rgb_arr = np.array(pil_img.convert("RGB"))

                adjusted_arr = adjust_fn(rgb_arr)
                out_img = Image.fromarray(adjusted_arr)
                save_kwargs: dict[str, Any] = {"quality": 95}
                if exif_bytes:
                    save_kwargs["exif"] = exif_bytes
                if icc_profile:
                    save_kwargs["icc_profile"] = icc_profile
                out_img.save(fpath, **save_kwargs)

                self._on_pixel_edit_saved(mid, str(backup_path))
            except Exception as e:
                import logging
                logging.getLogger(__name__).error("Failed to batch adjust %s: %s", fpath, e)

        self.rebuild_and_run_query()
        self.show_status_message(f"✓ {action_name} applied to {total} photo(s)", 4000)

    def _batch_auto_adjust(self, media_ids: list[int]) -> None:
        from src.imaging.opencv_engine import analyze_auto_adjust, render_adjusted_image

        def _adjust(rgb: np.ndarray) -> np.ndarray:
            rec = analyze_auto_adjust(rgb)
            return render_adjusted_image(rgb, rec)

        self._batch_apply_image_adjustment(media_ids, "Auto Adjust", _adjust)

    def _batch_apply_effect(self, media_ids: list[int], effect_name: str) -> None:
        from src.imaging.opencv_engine import _apply_effect_grade

        def _adjust(rgb: np.ndarray) -> np.ndarray:
            return _apply_effect_grade(rgb, effect_name)

        label = effect_name.replace("_", " ").title()
        self._batch_apply_image_adjustment(media_ids, f"Effect ({label})", _adjust)

    def _refresh_media_after_pixel_change(self, media_id: int) -> None:
        """Invalidate stale thumbnail caches and sync catalog_model with updated disk/DB orientation & dimensions."""
        mid = int(media_id)
        flag = self._active_thumb_flags.pop(mid, None)
        if flag is not None:
            flag.set()
        self.thumb_repo.invalidate([mid])
        if mid in self.pixmap_cache:
            del self.pixmap_cache[mid]
        self._media_meta_lookup.pop(mid, None)
        if hasattr(self, "_failed_thumb_ids"):
            self._failed_thumb_ids.discard(mid)
        if hasattr(self, "catalog_model") and self.catalog_model is not None:
            try:
                detail = self.catalog_repo.get_media_detail(mid)
                if detail:
                    self.catalog_model.update_items_metadata(
                        [mid],
                        orientation=int(detail.get("orientation") or 1),
                        width=int(detail.get("width") or 0),
                        height=int(detail.get("height") or 0),
                        mtime=int(detail.get("mtime") or 0),
                        size=int(detail.get("size") or 0),
                    )
            except Exception:
                pass
        self._update_revert_button_state()

    def _on_pixel_edit_saved(self, media_id: int, backup_path: str) -> None:
        self.catalog_repo.record_pixel_edit_backup(media_id, backup_path)
        self._refresh_media_after_pixel_change(media_id)
        fname = Path(backup_path).name if backup_path else "file"
        self.show_status_message(f"✓ Saved changes to '{fname}'", 4000)

    def _on_pixel_edit_saved_as_copy(self, original_media_id: int, new_path: str) -> None:
        p = Path(new_path)
        if p.exists() and self._is_path_in_library(p.parent):
            try:
                self.scan_folder(p.parent, include_subfolders=False)
                self.rebuild_and_run_query()
            except Exception as e:
                logger.debug("Failed scanning newly saved copy: %s", e)

        settings = QSettings("PhotoGallery64", "PhotoGallery64")
        switch_view = settings.value("save_copy_switch_view", True, type=bool)

        # Seamlessly switch viewer state to the new copy file ONLY if switch_view is True
        if self.central_stack.currentWidget() == self.media_viewer:
            if switch_view:
                self.setWindowTitle(f"{p.name} - Essentials Photo Gallery")
                new_id = self.catalog_repo.get_media_id_by_file_path(p) or getattr(self.media_viewer, "current_media_id", None)
                if new_id:
                    self.media_viewer.current_media_id = new_id
                    self._selected_ids = {new_id}
                    if hasattr(self, "right_tag_dock") and self.right_tag_dock is not None:
                        try:
                            summary = self.catalog_repo.get_selection_summary([new_id])
                            if not summary.get("filename"):
                                summary["filename"] = p.name
                            self.right_tag_dock.bind_selection([new_id], summary)
                        except Exception as ex:
                            logger.debug("Failed binding selection to dock: %s", ex)
            else:
                # Kept on original photo: ensure window title stays original
                if self.media_viewer.current_file_path:
                    self.setWindowTitle(f"{self.media_viewer.current_file_path.name} - Essentials Photo Gallery")

        self._update_revert_button_state()
        self.show_status_message(f"✓ Saved copy '{p.name}'", 4000)

    def _on_pixel_edit_reverted(self, media_id: int) -> None:
        self.catalog_repo.clear_pixel_edit_backup(media_id)
        self._refresh_media_after_pixel_change(media_id)

        # Restore original face boxes from the restored original file
        reverted_name = "file"
        try:
            detail = self.catalog_repo.get_media_detail(media_id)
            if detail and detail.get("filename"):
                reverted_name = str(detail["filename"])
            fpath = Path(detail.get("file_path", ""))
            if fpath.exists():
                reverted_name = fpath.name
                meta = extract_media_metadata(fpath, is_video=bool(detail.get("is_video")))
                self.catalog_repo.restore_original_faces(media_id, meta.get("people", []))
        except Exception:
            pass

        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id == media_id
        ):
            try:
                updated_detail = self.catalog_repo.get_media_detail(media_id)
                self.media_viewer.render_face_regions(updated_detail.get("people", []))
            except Exception:
                pass
        self.show_status_message(f"✓ Reverted '{reverted_name}' to original", 4000)

    # -------------------------------------------------------------------------
    # Ribbon & Status Bar Actions (Rotate, Delete, Fix, Options, Face Rescan)
    # -------------------------------------------------------------------------
    def rescan_faces_for_selection(self, media_ids: Sequence[int] | None = None) -> None:
        """Rescans face detection for selected media files with high-speed targeted worker."""
        if media_ids is None:
            media_ids = (
                [self.media_viewer.current_media_id]
                if self.central_stack.currentWidget() == self.media_viewer
                and self.media_viewer.current_media_id is not None
                else (self.grid_view.selected_media_ids() or list(self._selected_ids))
            )
        if not media_ids:
            return
        mids = [int(m) for m in media_ids if m is not None]
        if not mids:
            return
        self._targeted_face_scan_active = True
        self.catalog_repo.reset_face_scan_for_media(mids)
        self.statusBar().showMessage(f"Rescanning {len(mids)} photo(s) for faces...", 10000)
        self.lbl_sync_status.setText(f"Rescanning {len(mids)} photo(s) for faces...")

        worker = FaceDetectionWorker(
            engine=self.engine,
            target_media_ids=mids,
            batch_size=len(mids),
        )
        worker.signals.media_faces_ready.connect(self._on_targeted_face_scan_ready)
        worker.signals.worker_finished.connect(
            lambda total_faces: self._on_targeted_face_scan_finished(len(mids), total_faces)
        )
        self.priority_face_pool.start(worker)

    def _on_targeted_face_scan_ready(self, mid: int) -> None:
        """Called immediately as each targeted photo completes face scanning."""
        if self.central_stack.currentWidget() == self.media_viewer and self.media_viewer.current_media_id == mid:
            self.media_viewer.reload_faces(highlight_new=True)
        if hasattr(self, "right_tag_dock") and self.right_tag_dock is not None:
            curr_sel = (
                [self.media_viewer.current_media_id]
                if self.central_stack.currentWidget() == self.media_viewer and self.media_viewer.current_media_id is not None
                else (self.grid_view.selected_media_ids() or list(self._selected_ids))
            )
            if mid in curr_sel:
                try:
                    summary = self.catalog_repo.get_selection_summary(curr_sel)
                    self.right_tag_dock.bind_selection(curr_sel, summary)
                except Exception:
                    pass
        if hasattr(self, "grid_view") and self.grid_view is not None:
            self.grid_view.viewport().update()

    def _on_targeted_face_scan_finished(self, count: int, total_faces: int) -> None:
        """Called when all targeted photos complete face scanning."""
        self._targeted_face_scan_active = False
        face_txt = f"{total_faces} face(s) found" if total_faces > 0 else "no faces found"
        self.statusBar().showMessage(f"✓ Face rescan complete for {count} photo(s) ({face_txt}).", 5000)
        self.lbl_sync_status.setText(f"✓ Face rescan complete ({face_txt})")
        if self.central_stack.currentWidget() == self.media_viewer and self.media_viewer.current_media_id is not None:
            self.media_viewer.reload_faces(highlight_new=True)
            if hasattr(self, "right_tag_dock") and self.right_tag_dock is not None:
                try:
                    summary = self.catalog_repo.get_selection_summary([self.media_viewer.current_media_id])
                    self.right_tag_dock.bind_selection([self.media_viewer.current_media_id], summary)
                except Exception:
                    pass
        self.refresh_navigation_and_autocomplete()

    def rotate_selection(self, clockwise: bool = True) -> None:
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else (self.grid_view.selected_media_ids() or list(self._selected_ids))
        )
        if not target_ids:
            return
        cmd = _RotateUndoCommand(self, target_ids, clockwise)
        self.undo_stack.push(cmd)

    def _rotate_internal(self, target_ids: list[int], clockwise: bool) -> None:
        if not target_ids:
            return
        rot_map = self.catalog_repo.rotate_lossless(target_ids, clockwise=clockwise)
        rot_angle = 90 if clockwise else -90
        rot_transform = QTransform().rotate(rot_angle)

        for mid in target_ids:
            # 1. Update in-memory thumbnail cache immediately
            old_pm = self.pixmap_cache.get(mid)
            if old_pm is not None and not old_pm.isNull():
                self.pixmap_cache.put(mid, old_pm.transformed(rot_transform, Qt.SmoothTransformation))

            if (
                self.grid_view.catalog_model
                and self.grid_view.catalog_model.pixmap_cache is not self.pixmap_cache
            ):
                m_old_pm = self.grid_view.catalog_model.pixmap_cache.get(mid)
                if m_old_pm is not None and not m_old_pm.isNull():
                    self.grid_view.catalog_model.pixmap_cache.put(
                        mid, m_old_pm.transformed(rot_transform, Qt.SmoothTransformation)
                    )

            # 2. Rotate WebP thumbnail in thumbnails.db directly
            old_blob = self.thumb_repo.get_webp_blob(mid)
            if old_blob:
                qimg = QImage.fromData(old_blob, "WEBP")
                if not qimg.isNull():
                    rot_qimg = qimg.transformed(rot_transform, Qt.SmoothTransformation)
                    ba = QByteArray()
                    buf = QBuffer(ba)
                    buf.open(QIODevice.WriteOnly)
                    rot_qimg.save(buf, "WEBP", 80)
                    buf.close()
                    m_detail = self.catalog_repo.get_media_detail(mid)
                    mtime = int(m_detail.get("mtime") or 0)
                    self.thumb_repo.put_webp_blob(
                        mid, bytes(ba.data()), mtime, width=rot_qimg.width(), height=rot_qimg.height()
                    )
            else:
                self.thumb_repo.invalidate([mid])

            # 2b. Rotate face crop thumbnails and person display pictures on rotated photo
            try:
                face_rows = self.catalog_repo.engine.catalog_conn().execute(
                    "SELECT id, person_id FROM media_people WHERE media_id = ?;", (mid,)
                ).fetchall()
                for fr in face_rows:
                    fid = int(fr["id"])
                    f_blob = self.thumb_repo.get_face_thumbnail(fid)
                    if f_blob:
                        fqimg = QImage.fromData(f_blob)
                        if not fqimg.isNull():
                            rot_fqimg = fqimg.transformed(rot_transform, Qt.SmoothTransformation)
                            fba = QByteArray()
                            fbuf = QBuffer(fba)
                            fbuf.open(QIODevice.WriteOnly)
                            rot_fqimg.save(fbuf, "WEBP", 88)
                            fbuf.close()
                            new_face_bytes = bytes(fba.data())
                            self.thumb_repo.put_face_thumbnail(fid, new_face_bytes)
                            pid = int(fr["person_id"]) if fr["person_id"] is not None else None
                            if pid:
                                p_row = self.catalog_repo.engine.catalog_conn().execute(
                                    "SELECT avatar_media_people_id FROM people WHERE id = ?;", (pid,)
                                ).fetchone()
                                if p_row and p_row["avatar_media_people_id"] == fid:
                                    with self.catalog_repo.engine.write_lock:
                                        self.catalog_repo.engine.catalog_conn().execute(
                                            "UPDATE people SET thumbnail_blob = ? WHERE id = ?;",
                                            (new_face_bytes, pid),
                                        )
                                        self.catalog_repo.engine.catalog_conn().commit()
            except Exception:
                pass

            self._media_meta_lookup.pop(mid, None)

        # 3. Update catalog model records immediately with new orientation & swapped dimensions
        if self.grid_view.catalog_model:
            for mid, new_orient in rot_map.items():
                self.grid_view.catalog_model.update_items_metadata([mid], orientation=new_orient)

        # 4. If single viewer is currently viewing one of the rotated photos, rotate in memory directly!
        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id in target_ids
        ):
            self.media_viewer.apply_in_memory_rotation(clockwise=clockwise)
            if self.media_viewer.current_media_id in rot_map:
                self.media_viewer.current_orientation = int(rot_map[self.media_viewer.current_media_id])
            self.media_viewer.setFocus()
            self.media_viewer.graphics_view.setFocus()

        self.rebuild_and_run_query()
        self.grid_view.viewport().update()
        rot_dir = "clockwise" if clockwise else "counter-clockwise"
        self.show_status_message(f"✓ Rotated {len(target_ids)} item(s) {rot_dir}", 3000)

    def select_all_items(self) -> None:
        """Selects all items currently displayed in the thumbnail grid."""
        if self.central_stack.currentWidget() == self.grid_view:
            self.grid_view.selectAll()

    def cut_selection_to_clipboard(self) -> None:
        """Cuts file paths of currently selected media to the system clipboard (files & text)."""
        self._copy_or_cut_selection(is_cut=True)

    def copy_selection_to_clipboard(self) -> None:
        """Copies file paths of currently selected media to the system clipboard (files & text)."""
        self._copy_or_cut_selection(is_cut=False)

    def _copy_or_cut_selection(self, is_cut: bool = False) -> None:
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else (self.grid_view.selected_media_ids() or list(self._selected_ids))
        )
        if not target_ids:
            return
        urls: list[QUrl] = []
        paths: list[str] = []
        for mid in target_ids:
            try:
                detail = self.catalog_repo.get_media_detail(mid)
                fpath = Path(detail["folder_path"]) / str(detail["filename"])
                if fpath.exists():
                    urls.append(QUrl.fromLocalFile(str(fpath.resolve())))
                    paths.append(str(fpath.resolve()))
            except Exception:
                pass
        if urls:
            self._copied_urls = list(urls)
            effect = QByteArray(b"\x02\x00\x00\x00" if is_cut else b"\x05\x00\x00\x00")

            def _make_mime() -> QMimeData:
                m = QMimeData()
                m.setUrls(urls)
                m.setText("\r\n".join(paths))
                # Windows Explorer Preferred DropEffect: 0x02 = Move/Cut, 0x05 = Copy
                m.setData("Preferred DropEffect", effect)
                m.setData('application/x-qt-windows-mime;value="Preferred DropEffect"', effect)
                return m

            self._clipboard_mime = _make_mime()
            try:
                QApplication.clipboard().setMimeData(_make_mime())
            except Exception:
                pass
            action_name = "Cut" if is_cut else "Copied"
            item_w = "file" if len(urls) == 1 else "files"
            self.lbl_sync_status.setText(f"✓ {action_name} {len(urls)} {item_w} to clipboard")
            self.statusBar().showMessage(f"{action_name} {len(urls)} {item_w} to clipboard", 3000)

    def paste_from_clipboard(self) -> None:
        """Pastes files from clipboard into the currently active or selected folder."""
        clipboard = QApplication.clipboard()
        mime = clipboard.mimeData()
        if not mime or not mime.hasUrls():
            mime = getattr(self, "_clipboard_mime", None)
        if not mime or not mime.hasUrls():
            self.statusBar().showMessage("Clipboard is empty or does not contain files.", 3000)
            return

        file_paths: list[str] = []
        for u in mime.urls():
            if u.isLocalFile():
                p = u.toLocalFile()
                if os.path.exists(p):
                    file_paths.append(p)
        if not file_paths:
            self.statusBar().showMessage("No valid local files in clipboard.", 3000)
            return

        # Determine target folder
        target_folder_id = getattr(self.left_nav_dock, "_folder_id", None)
        target_folder_path: str | None = None

        if target_folder_id is not None:
            folder_row = self.catalog_repo.get_folder_by_id(target_folder_id)
            if folder_row:
                target_folder_path = folder_row["path"]

        # If no folder selected in left dock, check lead or selected media's folder
        if not target_folder_path:
            sel = self.grid_view.selected_media_ids() or list(self._selected_ids)
            if sel:
                detail = self.catalog_repo.get_media_detail(sel[0])
                target_folder_id = detail.get("folder_id")
                target_folder_path = detail.get("folder_path")

        # If still none, fallback to first folder in catalog
        if not target_folder_path:
            all_f = self.catalog_repo.get_folders()
            if all_f:
                target_folder_id = all_f[0]["id"]
                target_folder_path = all_f[0]["path"]

        if not target_folder_path or target_folder_id is None:
            self.statusBar().showMessage("Please select a target folder to paste into.", 3000)
            return

        # Check Preferred DropEffect: 0x02 = Move/Cut, 0x05 = Copy
        effect = bytes(mime.data("Preferred DropEffect"))
        if not effect:
            effect = bytes(mime.data('application/x-qt-windows-mime;value="Preferred DropEffect"'))
        is_copy = True
        if effect.startswith(b"\x02"):
            is_copy = False

        self._on_files_dropped_on_folder(
            target_folder_id=target_folder_id,
            target_folder_path=target_folder_path,
            file_paths=file_paths,
            is_copy=is_copy,
        )

        # If cut/move, reset clipboard drop effect to copy so subsequent pastes copy
        if not is_copy:
            def _make_copy_mime() -> QMimeData:
                nm = QMimeData()
                nm.setUrls([QUrl.fromLocalFile(p) for p in file_paths if os.path.exists(p)])
                nm.setData("Preferred DropEffect", QByteArray(b"\x05\x00\x00\x00"))
                nm.setData(
                    'application/x-qt-windows-mime;value="Preferred DropEffect"',
                    QByteArray(b"\x05\x00\x00\x00"),
                )
                return nm

            self._clipboard_mime = _make_copy_mime()
            try:
                clipboard.setMimeData(_make_copy_mime())
            except Exception:
                pass

    def delete_selection_to_recycle_bin(self) -> int:
        in_viewer = (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
        )
        current_viewer_idx = None
        if in_viewer:
            current_viewer_idx = self.catalog_model.record_index_for_id(self.media_viewer.current_media_id)

        target_ids = (
            [self.media_viewer.current_media_id]
            if in_viewer
            else (self.grid_view.selected_media_ids() or list(self._selected_ids))
        )
        if not target_ids:
            return 0
        try:
            from send2trash import send2trash
        except ImportError:
            send2trash = None

        deleted_count = 0
        last_deleted_name = ""
        conn = self.engine.catalog_conn()
        for mid in list(target_ids):
            try:
                detail = self.catalog_repo.get_media_detail(mid)
                fpath = Path(detail["folder_path"]) / str(detail["filename"])
                # 1. Delete original file
                if fpath.exists():
                    if send2trash is not None:
                        send2trash(str(fpath))
                    else:
                        fpath.unlink()

                # 2. Delete .OriginalFiles backup if present
                backup_path = detail.get("original_backup_path")
                b_cands = []
                if backup_path:
                    b_cands.append(Path(backup_path))
                b_cands.append(fpath.parent / ".OriginalFiles" / fpath.name)
                for b_cand in b_cands:
                    if b_cand.exists():
                        try:
                            if send2trash is not None:
                                send2trash(str(b_cand))
                            else:
                                b_cand.unlink()
                        except OSError:
                            pass

                # 3. Delete .xmp sidecar if present
                for xmp_cand in (fpath.with_name(fpath.name + ".xmp"), fpath.with_suffix(".xmp")):
                    if xmp_cand.exists() and xmp_cand != fpath:
                        try:
                            if send2trash is not None:
                                send2trash(str(xmp_cand))
                            else:
                                xmp_cand.unlink()
                        except OSError:
                            pass
                with self.engine.write_lock:
                    conn.execute("DELETE FROM media WHERE id = ?;", (mid,))
                    conn.commit()
                deleted_count += 1
                last_deleted_name = str(detail["filename"])
            except Exception:
                pass

        self.thumb_repo.invalidate(target_ids)
        self._selected_ids = [m for m in self._selected_ids if m not in target_ids]
        if self._lead_selected_id in target_ids:
            self._lead_selected_id = self._selected_ids[0] if self._selected_ids else None
        self.refresh_navigation_and_autocomplete()
        self.rebuild_and_run_query()
        if in_viewer:
            records = self.catalog_model._records
            if records:
                target_idx = current_viewer_idx if current_viewer_idx is not None else 0
                if target_idx >= len(records):
                    target_idx = len(records) - 1
                next_id = records[target_idx].id
                self._selected_ids = [next_id]
                self._lead_selected_id = next_id
                summary = self.catalog_repo.get_selection_summary([next_id])
                self.right_tag_dock.bind_selection([next_id], summary)
                self.open_single_viewer(next_id)
            else:
                self.show_grid_view()
        if deleted_count == 1 and last_deleted_name:
            self.show_status_message(f"✓ Moved '{last_deleted_name}' to Recycle Bin", 4000)
        elif deleted_count > 1:
            self.show_status_message(f"✓ Moved {deleted_count:,} items to Recycle Bin", 4000)
        elif target_ids:
            self.show_status_message("⚠ Could not delete selected item(s)", 4000)
        return deleted_count

    def set_desktop_background_for_media(self, media_id: int) -> bool:
        """Sets the selected media file as the Windows desktop wallpaper."""
        try:
            detail = self.catalog_repo.get_media_detail(media_id)
            fpath = Path(detail["folder_path"]) / str(detail["filename"])
            if not fpath.exists():
                return False
            import sys
            if sys.platform == "win32":
                import ctypes
                SPI_SETDESKWALLPAPER = 20
                SPIF_UPDATEINIFILE = 1
                SPIF_SENDWININICHANGE = 2
                ctypes.windll.user32.SystemParametersInfoW(
                    SPI_SETDESKWALLPAPER,
                    0,
                    str(fpath.resolve()),
                    SPIF_UPDATEINIFILE | SPIF_SENDWININICHANGE,
                )
                self.lbl_sync_status.setText(f"✓ Set '{fpath.name}' as desktop background")
                return True
        except Exception as e:
            self.lbl_sync_status.setText(f"⚠ Could not set desktop background: {e}")
        return False

    def open_file_location_for_media(self, media_id: int) -> bool:
        """Reveals the media file in Windows Explorer with the item selected."""
        try:
            detail = self.catalog_repo.get_media_detail(media_id)
            fpath = Path(detail["folder_path"]) / str(detail["filename"])
            if not fpath.exists():
                return False
            import os, subprocess, sys
            norm = os.path.normpath(str(fpath.resolve()))
            if sys.platform == "win32":
                subprocess.Popen(f'explorer.exe /select,"{norm}"')
            else:
                subprocess.Popen(["xdg-open", str(fpath.parent.resolve())])
            return True
        except Exception as e:
            self.lbl_sync_status.setText(f"⚠ Could not open file location: {e}")
        return False

    def open_file_properties(self, media_id: int | None = None) -> bool:
        """Opens native Windows File Properties dialog on the 'Details' tab for the specified or selected media item."""
        try:
            if media_id is None or isinstance(media_id, bool):
                if getattr(self, "is_viewer_mode", False) and hasattr(self, "media_viewer"):
                    media_id = getattr(self.media_viewer, "current_media_id", None)
                elif hasattr(self, "grid_view"):
                    sel = self.grid_view.selected_media_ids()
                    media_id = sel[0] if sel else None
                else:
                    media_id = None

            if media_id is None:
                return False

            detail = self.catalog_repo.get_media_detail(int(media_id))
            if not detail or not detail.get("folder_path") or not detail.get("filename"):
                return False

            fpath = Path(detail["folder_path"]) / str(detail["filename"])
            if not fpath.exists():
                return False

            import os, sys
            norm = os.path.normpath(str(fpath.resolve()))
            if sys.platform == "win32":
                import ctypes
                hwnd = int(self.winId()) if hasattr(self, "winId") else 0
                shell32 = ctypes.windll.shell32
                SHOP_FILEPATH = 0x00000002
                # Request 'Details' tab directly (matches Windows Live Photo Gallery)
                res = shell32.SHObjectProperties(hwnd, SHOP_FILEPATH, norm, "Details")
                if not res and hwnd:
                    res = shell32.SHObjectProperties(0, SHOP_FILEPATH, norm, "Details")
                if not res:
                    res = shell32.SHObjectProperties(hwnd, SHOP_FILEPATH, norm, None)
                return bool(res)
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(fpath.parent.resolve())])
                return True
        except Exception as e:
            logger.error("Could not open file properties: %s", e)
            return False

    def open_with_app_for_media(self, media_id: int, app_choice: str = "default") -> bool:
        """Opens the media file with the chosen app, system default viewer, or Windows Open With dialog."""
        try:
            detail = self.catalog_repo.get_media_detail(media_id)
            fpath = Path(detail["folder_path"]) / str(detail["filename"])
            if not fpath.exists():
                return False
            import subprocess
            import sys
            import ctypes
            if app_choice == "choose":
                if sys.platform == "win32":
                    try:
                        subprocess.Popen(["rundll32.exe", "shell32.dll,OpenAs_RunDLL", str(fpath.resolve())])
                        return True
                    except Exception:
                        pass
                    hwnd = int(self.winId()) if hasattr(self, "winId") else 0
                    ctypes.windll.shell32.OpenAs_RunDLLW(hwnd, 0, f'"{str(fpath.resolve())}"', 1)
                    return True
                else:
                    subprocess.Popen(["xdg-open", str(fpath.resolve())])
                    return True
            elif app_choice == "default":
                if sys.platform == "win32":
                    os.startfile(str(fpath.resolve()))
                else:
                    subprocess.Popen(["xdg-open", str(fpath.resolve())])
                return True
            else:
                from src.ui.grid_view import resolve_windows_app_path
                resolved = resolve_windows_app_path(app_choice)
                if resolved:
                    subprocess.Popen([resolved, str(fpath.resolve())])
                    return True
                elif "photos" in app_choice.lower() and sys.platform == "win32":
                    ps_cmd = (
                        f"$sh = New-Object -ComObject Shell.Application; "
                        f"$folder = $sh.Namespace('{str(fpath.parent.resolve())}'); "
                        f"$item = $folder.ParseName('{fpath.name}'); "
                        f"$verb = $item.Verbs() | Where-Object {{ $_.Name -like '*Photos*' }} | Select-Object -First 1; "
                        f"if ($verb) {{ $verb.DoIt() }}"
                    )
                    subprocess.Popen(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_cmd])
                    return True
                elif sys.platform == "win32":
                    res = ctypes.windll.shell32.ShellExecuteW(None, "open", app_choice, f'"{str(fpath.resolve())}"', None, 1)
                    if res > 32:
                        return True

            self.lbl_sync_status.setText(f"⚠ Could not launch application: {app_choice}")
            return False
        except Exception as e:
            self.lbl_sync_status.setText(f"⚠ Open with error: {e}")
            return False
        return False

    def rename_selection(self) -> None:
        """Selects the filename field in the right Information panel for inline renaming."""
        target_ids = self.grid_view.selected_media_ids() or list(self._selected_ids)
        if not target_ids:
            return
        if len(target_ids) == 1:
            if hasattr(self, "right_tag_dock") and self.right_tag_dock is not None:
                self.right_tag_dock.show()
                self.right_tag_dock.raise_()
                self.right_tag_dock.focus_filename_rename()
                return
            mid = target_ids[0]
            try:
                detail = self.catalog_repo.get_media_detail(mid)
                old_fname = str(detail["filename"])
                folder_p = Path(detail["folder_path"])
                old_path = folder_p / old_fname
                stem = Path(old_fname).stem
                ext = Path(old_fname).suffix
                new_stem, ok = QInputDialog.getText(
                    self, "Rename", "Enter a new name for the file:", text=stem
                )
                if not ok or not new_stem.strip() or new_stem.strip() == stem:
                    return
                new_fname = f"{new_stem.strip()}{ext}"
                new_path = folder_p / new_fname
                if new_path.exists():
                    QMessageBox.warning(self, "File Exists", f"A file named '{new_fname}' already exists.")
                    return
                if old_path.exists():
                    old_path.rename(new_path)
                self.catalog_repo.rename_media(mid, new_fname)
                self._media_meta_lookup.pop(mid, None)
                self.rebuild_and_run_query()
                self.show_status_message(f"✓ Renamed file to '{new_fname}'", 4000)
            except Exception as e:
                QMessageBox.warning(self, "Rename Error", f"Could not rename file: {e}")
        else:
            base_name, ok = QInputDialog.getText(
                self, "Batch Rename", "Enter a base name for the selected files:"
            )
            if not ok or not base_name.strip():
                return
            base_name = base_name.strip()
            renamed = 0
            for i, mid in enumerate(target_ids, start=1):
                try:
                    detail = self.catalog_repo.get_media_detail(mid)
                    old_fname = str(detail["filename"])
                    folder_p = Path(detail["folder_path"])
                    old_path = folder_p / old_fname
                    ext = Path(old_fname).suffix
                    new_fname = f"{base_name} ({i}){ext}"
                    new_path = folder_p / new_fname
                    if old_path.exists() and not new_path.exists():
                        old_path.rename(new_path)
                        self.catalog_repo.rename_media(mid, new_fname)
                        self._media_meta_lookup.pop(mid, None)
                        renamed += 1
                except Exception:
                    pass
            self.rebuild_and_run_query()
            self.show_status_message(f"✓ Renamed {renamed} file(s)", 4000)

    def open_resize_dialog(self) -> ResizePhotoDialog | None:
        """Opens authentic WLPG Resize dialog for selected image photos."""
        target_ids = self.grid_view.selected_media_ids() or list(self._selected_ids)
        if not target_ids:
            return None
        file_paths: list[Path] = []
        for mid in target_ids:
            try:
                detail = self.catalog_repo.get_media_detail(mid)
                fpath = Path(detail["folder_path"]) / str(detail["filename"])
                if fpath.exists() and not bool(detail.get("is_video")):
                    file_paths.append(fpath)
            except Exception:
                pass
        if not file_paths:
            return None

        dlg = ResizePhotoDialog(file_paths, parent=self)
        if dlg.exec() == ResizePhotoDialog.Accepted:
            out_dirs = {p.parent for p in dlg.generated_files}
            for d in out_dirs:
                self.scan_folder(d, include_subfolders=False)
            self.show_status_message(f"✓ Resized and saved {len(dlg.generated_files)} photo(s)", 4000)
        return dlg

    def make_copy_of_selection(self, prompt_dialog: bool = True) -> list[Path]:
        """Creates a duplicate copy of selected media (or current image in viewer) matching WLPG 'Make a copy'."""
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else (self.grid_view.selected_media_ids() or list(self._selected_ids))
        )
        if not target_ids:
            return []

        mid = target_ids[0]
        try:
            detail = self.catalog_repo.get_media_detail(mid)
            folder_p = Path(detail["folder_path"])
            old_fname = str(detail["filename"])
            src_file = folder_p / old_fname
            if not src_file.exists():
                return []

            stem = src_file.stem
            ext = src_file.suffix.lower()

            # Generate default candidate: e.g. "DSC06914 (2).jpg" matching WLPG
            import re
            m = re.match(r"^(.*?)\s*\((\d+)\)$", stem)
            if m:
                base_stem = m.group(1).rstrip()
                start_n = int(m.group(2)) + 1
            else:
                base_stem = stem
                start_n = 2

            n = start_n
            while True:
                candidate_name = f"{base_stem} ({n}){src_file.suffix}"
                if not (folder_p / candidate_name).exists():
                    break
                n += 1

            default_save_path = folder_p / candidate_name

            # Construct filter string matching WLPG and Windows Photo Gallery
            if ext in (".jpg", ".jpeg", ".jpe", ".jfif"):
                filter_str = (
                    "JPG (*.jpg;*.jpe;*.exif;*.jfif;*.jpeg);;"
                    "PNG (*.png);;"
                    "TIFF (*.tif;*.tiff);;"
                    "BMP (*.bmp);;"
                    "All Files (*.*)"
                )
                selected_filter = "JPG (*.jpg;*.jpe;*.exif;*.jfif;*.jpeg)"
            elif ext == ".png":
                filter_str = (
                    "PNG (*.png);;"
                    "JPG (*.jpg;*.jpe;*.exif;*.jfif;*.jpeg);;"
                    "All Files (*.*)"
                )
                selected_filter = "PNG (*.png)"
            elif ext in (".tif", ".tiff"):
                filter_str = (
                    "TIFF (*.tif;*.tiff);;"
                    "JPG (*.jpg;*.jpe;*.exif;*.jfif;*.jpeg);;"
                    "All Files (*.*)"
                )
                selected_filter = "TIFF (*.tif;*.tiff)"
            elif ext == ".bmp":
                filter_str = (
                    "BMP (*.bmp);;"
                    "JPG (*.jpg;*.jpe;*.exif;*.jfif;*.jpeg);;"
                    "All Files (*.*)"
                )
                selected_filter = "BMP (*.bmp)"
            else:
                filter_str = f"{ext.upper().lstrip('.')} (*{ext});;All Files (*.*)"
                selected_filter = f"{ext.upper().lstrip('.')} (*{ext})"

            is_test_env = "pytest" in sys.modules or bool(os.environ.get("PYTEST_CURRENT_TEST"))
            if prompt_dialog and not is_test_env:
                dest_path_str, _ = QFileDialog.getSaveFileName(
                    self,
                    "Make a copy",
                    str(default_save_path),
                    filter_str,
                    selected_filter,
                )
                if not dest_path_str:
                    return []
                new_file = Path(dest_path_str)
            else:
                new_file = default_save_path

            # If in single media viewer and active edits exist, save current edited pixels
            in_viewer = (
                self.central_stack.currentWidget() == self.media_viewer
                and self.media_viewer.current_media_id == mid
            )
            if in_viewer and self.media_viewer.current_pil_image is not None:
                self.media_viewer.ensure_master_rendered()
                save_fmt = "JPEG" if new_file.suffix.lower() in (".jpg", ".jpeg") else "PNG"
                self.media_viewer.current_pil_image.save(new_file, format=save_fmt, quality=95)
            else:
                shutil.copy2(src_file, new_file)

            # Rescan destination directory so the copy shows in gallery
            self.scan_folder(new_file.parent, include_subfolders=False)
            self.rebuild_and_run_query()

            # Find the new media ID in database and clone tags / star rating / caption / flag
            new_mid = None
            try:
                conn = self.engine.catalog_conn()
                f_row = conn.execute("SELECT id FROM folders WHERE path = ?;", (str(new_file.parent),)).fetchone()
                if f_row:
                    m_row = conn.execute(
                        "SELECT id FROM media WHERE folder_id = ? AND filename = ?;",
                        (f_row[0], new_file.name),
                    ).fetchone()
                    if m_row:
                        new_mid = int(m_row[0])
                        conn.execute(
                            "UPDATE media SET rating = (SELECT rating FROM media WHERE id = ?), "
                            "caption = (SELECT caption FROM media WHERE id = ?), "
                            "flagged = (SELECT flagged FROM media WHERE id = ?) "
                            "WHERE id = ?;",
                            (mid, mid, mid, new_mid),
                        )
                        conn.execute(
                            "INSERT OR IGNORE INTO media_tags (media_id, tag_id) "
                            "SELECT ?, tag_id FROM media_tags WHERE media_id = ?;",
                            (new_mid, mid),
                        )
                        conn.commit()
            except Exception:
                pass

            if in_viewer and new_mid:
                # Switch viewer to the newly created copy so user seamlessly continues editing it
                self.open_single_viewer(new_mid)
            else:
                self._select_file_in_grid(new_file)

            self.show_status_message(f"✓ Created copy: {new_file.name}", 4000)
            return [new_file]

        except Exception as e:
            self.show_status_message(f"⚠ Copy error: {e}", 4000)
            return []

    def toggle_right_pane(
        self, mode: Literal["info", "fine_tune"] = "info"
    ) -> None:
        """
        Mutually exclusive toggle for the Right Inspector:
        - mode 'info' (Ctrl+I): Tag & caption / metadata information
        - mode 'fine_tune' (Ctrl+F): Exposure, color, straighten, and detail adjustments
        """
        is_open = not self.right_tag_dock.isHidden()
        if is_open and self.right_tag_dock.current_mode() == mode:
            self.right_tag_dock.hide()
        else:
            self.right_tag_dock.set_mode(mode)
            self.right_tag_dock.show()
            self.right_tag_dock.raise_()
            try:
                target_w = 300 if mode in ("fine_tune", "depth_light") else 280
                self.resizeDocks([self.right_tag_dock], [target_w], Qt.Horizontal)
            except Exception:
                pass
            if mode == "fine_tune":
                self.right_tag_dock.histogram_widget.set_histogram_data(
                    self.media_viewer.current_pil_image
                )
            elif mode == "depth_light":
                proxy = getattr(self.media_viewer, "display_proxy_rgb", None)
                mid = getattr(self.media_viewer, "current_media_id", None)
                if proxy is not None and mid is not None:
                    from src.imaging.creative_effects import get_depth_and_matte_engine
                    QTimer.singleShot(
                        0,
                        lambda: get_depth_and_matte_engine().get_depth_and_matte(proxy, media_id=mid),
                    )
        is_open = not self.right_tag_dock.isHidden()
        self._sync_tag_pane_buttons(is_open)
        if hasattr(self.ribbon, "btn_edit_tag_caption"):
            self.ribbon.btn_edit_tag_caption.setChecked(is_open)

    def toggle_navigation_pane(self) -> None:
        """Toggles the left navigation pane visibility (Ctrl+N)."""
        self.dispatch_ribbon_action("toggle_nav_pane")

    def toggle_immersive_viewer_mode(self) -> None:
        """Toggles Immersive Viewer Mode (F4): hides/restores all side panes, status bar, QAT and collapses ribbon."""
        if not getattr(self, "_is_immersive_mode", False):
            # Enter immersive mode: save current visibility states
            self._immersive_saved_state = {
                "left_nav": self.left_nav_dock.isVisible() if hasattr(self, "left_nav_dock") else False,
                "right_tag": self.right_tag_dock.isVisible() if hasattr(self, "right_tag_dock") else False,
                "fix_pane": (
                    self.media_viewer.fix_pane.isVisible()
                    if hasattr(self, "media_viewer") and hasattr(self.media_viewer, "fix_pane")
                    else False
                ),
                "status_bar": self.statusBar().isVisible() if self.statusBar() else True,
                "top_header": self.ribbon.top_header.isVisible() if hasattr(self.ribbon, "top_header") else True,
                "qat": self.ribbon.qat_container.isVisible() if hasattr(self.ribbon, "qat_container") else True,
                "scope_bar": self.scope_bar.isVisible() if hasattr(self, "scope_bar") else False,
                "ribbon_collapsed": (
                    self.ribbon._ribbon_collapsed if hasattr(self.ribbon, "_ribbon_collapsed") else False
                ),
            }
            if hasattr(self, "left_nav_dock"):
                self.left_nav_dock.hide()
            if hasattr(self, "right_tag_dock"):
                self.right_tag_dock.hide()
            if hasattr(self, "media_viewer") and hasattr(self.media_viewer, "fix_pane"):
                self.media_viewer.fix_pane.hide()
            if self.statusBar():
                self.statusBar().hide()
            if hasattr(self.ribbon, "top_header"):
                self.ribbon.top_header.hide()
            if hasattr(self.ribbon, "qat_container"):
                self.ribbon.qat_container.hide()
            if hasattr(self, "scope_bar"):
                self.scope_bar.hide()
            if hasattr(self.ribbon, "set_ribbon_collapsed"):
                self.ribbon.set_ribbon_collapsed(True)
            if hasattr(self.ribbon, "btn_exit_immersive"):
                self.ribbon.btn_exit_immersive.show()
            self._is_immersive_mode = True
        else:
            # Restore previous visibility states
            saved = getattr(self, "_immersive_saved_state", {})
            if saved.get("left_nav") and hasattr(self, "left_nav_dock"):
                self.left_nav_dock.show()
                try:
                    self.resizeDocks([self.left_nav_dock], [230], Qt.Horizontal)
                except Exception:
                    pass
            if saved.get("right_tag") and hasattr(self, "right_tag_dock"):
                self.right_tag_dock.show()
            if saved.get("fix_pane") and hasattr(self, "media_viewer") and hasattr(self.media_viewer, "fix_pane"):
                self.media_viewer.fix_pane.show()
            if saved.get("status_bar", True) and self.statusBar():
                self.statusBar().show()
            if saved.get("top_header", True) and hasattr(self.ribbon, "top_header"):
                self.ribbon.top_header.show()
            if saved.get("qat", True) and hasattr(self.ribbon, "qat_container"):
                self.ribbon.qat_container.show()
            if saved.get("scope_bar") and hasattr(self, "scope_bar"):
                self.scope_bar.show()
            if not saved.get("ribbon_collapsed", False) and hasattr(self.ribbon, "set_ribbon_collapsed"):
                self.ribbon.set_ribbon_collapsed(False)
            if hasattr(self.ribbon, "btn_exit_immersive"):
                self.ribbon.btn_exit_immersive.hide()
            self._is_immersive_mode = False

    def dispatch_ribbon_action(self, action_id: str) -> None:
        if action_id in ("toggle_immersive", "exit_immersive"):
            self.toggle_immersive_viewer_mode()
        elif action_id == "add_folder":
            self._prompt_and_add_folder()
        elif action_id in ("import_media", "import_photos"):
            self.open_import_wizard()
        elif action_id == "select_all":
            self.select_all_items()
        elif action_id == "cut":
            self.cut_selection_to_clipboard()
        elif action_id == "copy":
            self.copy_selection_to_clipboard()
        elif action_id == "paste":
            self.paste_from_clipboard()
        elif action_id == "rotate_left":
            self.rotate_selection(clockwise=False)
        elif action_id == "rotate_right":
            self.rotate_selection(clockwise=True)
        elif action_id == "edit_caption":
            self.right_tag_dock.set_mode("info")
            self.right_tag_dock.show()
            self.right_tag_dock.txt_caption.setFocus()
            self.right_tag_dock.txt_caption.selectAll()
        elif action_id == "batch_people_tag":
            self.open_batch_people_view()
        elif action_id == "batch_select_all":
            if hasattr(self, "batch_people_view"):
                self.batch_people_view.select_all()
        elif action_id == "batch_confirm":
            if hasattr(self, "batch_people_view"):
                self.batch_people_view.confirm_selected()
        elif action_id == "batch_switch":
            if hasattr(self, "batch_people_view"):
                self.batch_people_view.switch_selected()
        elif action_id == "batch_reject":
            if hasattr(self, "batch_people_view"):
                self.batch_people_view.reject_selected()
        elif action_id == "batch_tag_as":
            if hasattr(self, "batch_people_view"):
                self.batch_people_view._on_tag_as_clicked()
        elif action_id == "batch_ignore":
            if hasattr(self, "batch_people_view"):
                self.batch_people_view.ignore_selected()
        elif action_id == "batch_mode_suggestions":
            if hasattr(self, "batch_people_view"):
                self.batch_people_view.set_view_mode("suggestions")
        elif action_id == "batch_mode_outliers":
            if hasattr(self, "batch_people_view"):
                self.batch_people_view.set_view_mode("outliers")
        elif action_id == "batch_mode_boxless":
            if hasattr(self, "batch_people_view"):
                self.batch_people_view.set_view_mode("boxless")
        elif action_id == "batch_face_options":
            if hasattr(self, "batch_people_view"):
                self.batch_people_view._open_face_options_dialog()
        elif action_id == "batch_close":
            self.close_batch_people_view()
        elif action_id == "people_tag":
            self._on_dock_person_tag_mode_requested()
        elif action_id == "toggle_flag":
            target_ids = (
                [self.media_viewer.current_media_id]
                if self.central_stack.currentWidget() == self.media_viewer
                and self.media_viewer.current_media_id is not None
                else list(self._selected_ids)
            )
            if target_ids:
                flags_map = self.catalog_repo.get_field_map_for_ids(target_ids, "flag")
                has_unflagged = any(flags_map.get(mid, 0) == 0 for mid in target_ids)
                self.apply_flag_to_selection(1 if has_unflagged else 0)
        elif action_id.startswith("rate_"):
            try:
                r_val = int(action_id.split("_")[1])
                self.apply_rating_to_selection(r_val)
            except ValueError:
                pass
        elif action_id == "toggle_scope_bar":
            self.toggle_search_scope_bar()
        elif action_id == "help":
            self._open_about_dialog()
        elif action_id == "exit_app":
            self.close()
        elif action_id == "change_time_taken":
            self.open_change_time_taken_dialog()
        elif action_id == "delete":
            self.delete_selection_to_recycle_bin()
        elif action_id == "toggle_tag_pane":
            self.toggle_right_pane("info")
        elif action_id in ("focus_tag_box", "tag_desc"):
            self.focus_descriptive_tag_box()
        elif action_id == "delete_unused_tags":
            if hasattr(self, "left_nav_dock") and self.left_nav_dock is not None:
                self.left_nav_dock._on_delete_unused_tags()
        elif action_id in ("merge_orphan_tags", "clean_duplicate_tags"):
            self.clean_up_orphan_tags()
        elif action_id in ("restore_orphan_tags", "restore_duplicate_tags"):
            self.restore_last_tag_cleanup()
        elif action_id == "fine_tune":
            if self.central_stack.currentWidget() != self.media_viewer:
                sel = self.grid_view.selected_media_ids() or list(self._selected_ids)
                if sel:
                    self.open_single_viewer(sel[0])
            self.toggle_right_pane("fine_tune")
        elif action_id == "depth_light_pane":
            if self.central_stack.currentWidget() != self.media_viewer:
                sel = self.grid_view.selected_media_ids() or list(self._selected_ids)
                if sel:
                    self.open_single_viewer(sel[0])
            self.toggle_right_pane("depth_light")
        elif action_id == "save_photo":
            if self.central_stack.currentWidget() == self.media_viewer:
                self.media_viewer.save_current_image_with_backup()
        elif action_id == "save_as_copy":
            if self.central_stack.currentWidget() == self.media_viewer:
                self.media_viewer.save_as_copy()
        elif action_id == "save_as_copy_original":
            if self.central_stack.currentWidget() == self.media_viewer:
                self.media_viewer.save_as_copy(forced_target="original")
        elif action_id == "save_as_copy_choose_location":
            if self.central_stack.currentWidget() == self.media_viewer:
                self.media_viewer.save_as_copy(forced_target="choose")
        elif action_id == "open_options_edit":
            from src.ui.options_dialog import OptionsDialog
            dlg = OptionsDialog(engine=self.engine, initial_tab="edit", parent=self)
            dlg.exec()
        elif action_id == "toggle_nav_pane":
            new_vis = not self.left_nav_dock.isVisible()
            self.left_nav_dock.setVisible(new_vis)
            if new_vis:
                self.left_nav_dock.show()
                self.resizeDocks([self.left_nav_dock], [230], Qt.Horizontal)
            self._sync_nav_pane_buttons(new_vis)
        elif action_id == "toggle_immersive":
            self.toggle_immersive_viewer_mode()
        elif action_id == "show_tag_hierarchy":
            self.open_tag_hierarchy_dialog()
        elif action_id in ("show_thumbnails_view", "show_grid_view"):
            self.trigger_thumbnail_view()
        elif action_id == "toggle_albums_view":
            if hasattr(self, "albums_view") and self.central_stack.currentWidget() == self.albums_view:
                self.show_grid_view()
            else:
                self.open_albums_view()
        elif action_id == "open_compare_view":
            self.open_compare_view()
        elif action_id == "compare_layout_side":
            self.compare_view.set_layout_mode("side_by_side")
        elif action_id == "compare_layout_top_bottom":
            self.compare_view.set_layout_mode("top_bottom")
        elif action_id == "compare_layout_3up":
            self.compare_view.set_layout_mode("three_up")
        elif action_id == "compare_layout_quad":
            self.compare_view.set_layout_mode("quad")
        elif action_id == "compare_layout_wipe":
            self.compare_view.set_layout_mode("wipe")
        elif action_id == "compare_swap_panes":
            self.compare_view.swap_panes()
        elif action_id.startswith("compare_lock_sync:"):
            self.compare_view.set_lock_sync(action_id.endswith(":1"))
        elif action_id == "compare_zoom_face":
            self.compare_view.zoom_to_faces()
        elif action_id == "compare_zoom_1to1":
            self.compare_view.zoom_actual_1_to_1()
        elif action_id == "compare_zoom_fit":
            self.compare_view.zoom_fit()
        elif action_id.startswith("compare_focus_peaking:"):
            self.compare_view.set_focus_peaking(action_id.endswith(":1"))
        elif action_id.startswith("compare_clipping_warnings:"):
            self.compare_view.set_clipping_warnings(action_id.endswith(":1"))
        elif action_id.startswith("compare_auto_advance:"):
            self.compare_view.set_auto_advance(action_id.endswith(":1"))
        elif action_id.startswith("compare_dark_backdrop:"):
            self.compare_view.set_dark_mode(action_id.endswith(":1"))
        elif action_id == "compare_pin_active":
            self.compare_view.toggle_pin_focused_pane()
        elif action_id == "compare_dismiss_active":
            self.compare_view.dismiss_focused_pane()
        elif action_id == "compare_keep_winner":
            self.compare_view.keep_winner_only()
        elif action_id == "compare_open_editor":
            self.compare_view.open_winner_in_editor()
        elif action_id == "compare_close":
            self.compare_view.close_compare()
        elif action_id == "toggle_people_directory":
            if hasattr(self, "people_directory_view") and self.central_stack.currentWidget() == self.people_directory_view:
                self.show_grid_view()
            else:
                self.open_people_directory_view()
        elif action_id == "toggle_map_view":
            if hasattr(self, "duplicate_finder_view") and self.central_stack.currentWidget() == self.duplicate_finder_view:
                self.show_grid_view()
                return
            if self.central_stack.currentWidget() == self.map_view:
                self.show_grid_view()
            else:
                self.show_map_view()
        elif action_id == "map_style_light":
            mv = self._ensure_map_view()
            mv.set_map_style("clean")
        elif action_id == "map_style_street":
            mv = self._ensure_map_view()
            mv.set_map_style("street")
        elif action_id == "map_style_satellite":
            mv = self._ensure_map_view()
            mv.set_map_style("satellite")
        elif action_id == "map_fit_photos":
            mv = self._ensure_map_view()
            mv.fit_to_photos()
        elif action_id == "map_view_area_in_grid":
            mv = self._ensure_map_view()
            mv.view_area_in_grid()
        elif action_id == "close_map":
            self.show_grid_view()
        elif action_id in ("geotag", "open_map"):
            self.toggle_right_pane("info")
            if hasattr(self.right_tag_dock, "txt_add_geotag"):
                self.right_tag_dock.txt_add_geotag.setFocus()
                self.right_tag_dock.txt_add_geotag.selectAll()
        elif action_id == "close_viewer":
            self.show_grid_view()
        elif action_id == "prev_media":
            self.navigate_single_viewer(-1)
        elif action_id == "next_media":
            self.navigate_single_viewer(1)
        elif action_id == "auto_adjust":
            if self.central_stack.currentWidget() == self.media_viewer:
                self.media_viewer.apply_auto_adjust()
            else:
                sel = self.grid_view.selected_media_ids() or list(self._selected_ids)
                if len(sel) > 1:
                    self._batch_auto_adjust(sel)
                elif len(sel) == 1:
                    self.open_single_viewer(sel[0])
                    self.media_viewer.apply_auto_adjust()
        elif action_id == "crop_photo":
            if self.central_stack.currentWidget() != self.media_viewer:
                sel = self.grid_view.selected_media_ids() or list(self._selected_ids)
                if sel:
                    self.open_single_viewer(sel[0])
                    self.media_viewer.set_crop_mode(True)
            else:
                self.media_viewer.set_crop_mode(not self.media_viewer.is_crop_mode)
        elif action_id == "revert_original":
            if self.central_stack.currentWidget() == self.media_viewer:
                self.media_viewer.revert_to_original()
            else:
                sel = self.grid_view.selected_media_ids()
                if sel:
                    for mid in sel:
                        detail = self.catalog_repo.get_media_detail(mid)
                        if not detail:
                            continue
                        folder_p = detail.get("folder_path")
                        fname = detail.get("filename")
                        fpath_str = detail.get("file_path")
                        if not fpath_str and folder_p and fname:
                            fpath_str = str(Path(folder_p) / fname)
                        if not fpath_str:
                            continue
                        fpath = Path(fpath_str)
                        bpath = detail.get("original_backup_path") or detail.get("edit_backup_path")
                        if not bpath or not Path(bpath).exists():
                            cand = fpath.parent / ".OriginalFiles" / fpath.name
                            if cand.exists():
                                bpath = str(cand)
                        if bpath and Path(bpath).exists():
                            shutil.copy2(bpath, fpath)
                            try:
                                Path(bpath).unlink()
                            except OSError:
                                pass
                            self._on_pixel_edit_reverted(mid)
                    self.rebuild_and_run_query()
                    self._update_revert_button_state()
        elif action_id.startswith("effect_"):
            eff_name = action_id.split("_", 1)[1]
            if self.central_stack.currentWidget() == self.media_viewer:
                self.media_viewer.apply_preset_effect(eff_name)
            else:
                sel = self.grid_view.selected_media_ids() or list(self._selected_ids)
                if len(sel) > 1:
                    self._batch_apply_effect(sel, eff_name)
                elif len(sel) == 1:
                    self.open_single_viewer(sel[0])
                    self.media_viewer.apply_preset_effect(eff_name)
        elif action_id == "make_copy":
            self.make_copy_of_selection()
        elif action_id == "view_file":
            target = self._lead_selected_id or (self._selected_ids[0] if self._selected_ids else None)
            if target is not None:
                self.open_single_viewer(target)
        elif action_id == "rename_file":
            self.rename_selection()
        elif action_id == "resize_photo":
            self.open_resize_dialog()
        elif action_id == "straighten":
            if self.central_stack.currentWidget() != self.media_viewer:
                sel = self.grid_view.selected_media_ids() or list(self._selected_ids)
                if sel:
                    self.open_single_viewer(sel[0])
            self.media_viewer.apply_auto_straighten()
        elif action_id == "noise_reduction":
            if self.central_stack.currentWidget() != self.media_viewer:
                sel = self.grid_view.selected_media_ids() or list(self._selected_ids)
                if sel:
                    self.open_single_viewer(sel[0])
            self._on_analyze_noise_requested()
        elif action_id == "retouch":
            if self.central_stack.currentWidget() != self.media_viewer:
                sel = self.grid_view.selected_media_ids() or list(self._selected_ids)
                if sel:
                    self.open_single_viewer(sel[0])
                    self.media_viewer.set_retouch_mode(True)
            else:
                self.media_viewer.set_retouch_mode(not self.media_viewer.is_retouch_mode)
        elif action_id == "red_eye":
            if self.central_stack.currentWidget() != self.media_viewer:
                sel = self.grid_view.selected_media_ids() or list(self._selected_ids)
                if sel:
                    self.open_single_viewer(sel[0])
                    self.media_viewer.set_red_eye_mode(True)
            else:
                self.media_viewer.set_red_eye_mode(not self.media_viewer.is_red_eye_mode)
        elif action_id == "color_warm":
            if self.central_stack.currentWidget() != self.media_viewer:
                sel = self.grid_view.selected_media_ids()
                if sel:
                    self.open_single_viewer(sel[0])
            self.media_viewer.set_color_parameters(
                temperature=min(100.0, self.media_viewer.current_recipe.temperature + 25.0)
            )
        elif action_id == "color_cool":
            if self.central_stack.currentWidget() != self.media_viewer:
                sel = self.grid_view.selected_media_ids()
                if sel:
                    self.open_single_viewer(sel[0])
            self.media_viewer.set_color_parameters(
                temperature=max(-100.0, self.media_viewer.current_recipe.temperature - 25.0)
            )
        elif action_id == "color_sat_up":
            if self.central_stack.currentWidget() != self.media_viewer:
                sel = self.grid_view.selected_media_ids()
                if sel:
                    self.open_single_viewer(sel[0])
            self.media_viewer.set_color_parameters(
                saturation=min(100.0, self.media_viewer.current_recipe.saturation + 25.0)
            )
        elif action_id == "color_sat_down":
            if self.central_stack.currentWidget() != self.media_viewer:
                sel = self.grid_view.selected_media_ids()
                if sel:
                    self.open_single_viewer(sel[0])
            self.media_viewer.set_color_parameters(
                saturation=max(-100.0, self.media_viewer.current_recipe.saturation - 25.0)
            )
        elif action_id == "exposure_brighten":
            if self.central_stack.currentWidget() != self.media_viewer:
                sel = self.grid_view.selected_media_ids()
                if sel:
                    self.open_single_viewer(sel[0])
            self.media_viewer.set_exposure_parameters(
                brightness=min(100.0, self.media_viewer.current_recipe.brightness + 15.0)
            )
        elif action_id == "exposure_darken":
            if self.central_stack.currentWidget() != self.media_viewer:
                sel = self.grid_view.selected_media_ids()
                if sel:
                    self.open_single_viewer(sel[0])
            self.media_viewer.set_exposure_parameters(
                brightness=max(-100.0, self.media_viewer.current_recipe.brightness - 15.0)
            )
        elif action_id == "exposure_contrast_up":
            if self.central_stack.currentWidget() != self.media_viewer:
                sel = self.grid_view.selected_media_ids()
                if sel:
                    self.open_single_viewer(sel[0])
            self.media_viewer.set_exposure_parameters(
                contrast=min(100.0, self.media_viewer.current_recipe.contrast + 20.0)
            )
        elif action_id == "exposure_contrast_down":
            if self.central_stack.currentWidget() != self.media_viewer:
                sel = self.grid_view.selected_media_ids()
                if sel:
                    self.open_single_viewer(sel[0])
            self.media_viewer.set_exposure_parameters(
                contrast=max(-100.0, self.media_viewer.current_recipe.contrast - 20.0)
            )
        elif action_id == "tag_face":
            self.media_viewer.tag_selected_crop_as_face()
        elif action_id == "filter_flagged":
            self.left_nav_dock.set_flagged_only(True)
        elif action_id == "filter_photos":
            self.left_nav_dock.set_media_type("image")
        elif action_id == "filter_videos":
            self.left_nav_dock.set_media_type("video")
        elif action_id == "filter_clear":
            self.left_nav_dock.clear_all_filters()
        elif action_id == "slide_show":
            records = self.catalog_model._records
            if not records:
                return
            start_idx = 0
            curr_id = (
                self.media_viewer.current_media_id
                if self.central_stack.currentWidget() == self.media_viewer
                else (self._lead_selected_id or (self._selected_ids[0] if self._selected_ids else None))
            )
            if curr_id is not None:
                start_idx = self.catalog_model.record_index_for_id(curr_id) or 0
            theme = getattr(self.ribbon, "current_slideshow_theme", "static")
            dlg = FullscreenSlideshowDialog(
                records=records,
                start_index=start_idx,
                catalog_repo=self.catalog_repo,
                theme=theme,
                parent=self,
            )
            dlg.exec()
            if hasattr(dlg, "current_theme") and hasattr(self.ribbon, "set_slideshow_theme"):
                self.ribbon.set_slideshow_theme(dlg.current_theme)
            last_id = dlg.current_media_id()
            if last_id is not None:
                self._selected_ids = [last_id]
                self._lead_selected_id = last_id
                summary = self.catalog_repo.get_selection_summary([last_id])
                self.right_tag_dock.bind_selection([last_id], summary)
                if self.central_stack.currentWidget() == self.media_viewer:
                    self.open_single_viewer(last_id)
                else:
                    self.grid_view.select_media_id(last_id)
        elif action_id == "zoom_in":
            if self.central_stack.currentWidget() == self.media_viewer:
                self.media_viewer.zoom_in()
            elif hasattr(self, "duplicate_finder_view") and self.central_stack.currentWidget() == self.duplicate_finder_view:
                self.duplicate_finder_view._zoom_in()
            else:
                self.slider_status_zoom.setValue(
                    min(self.slider_status_zoom.maximum(), self.slider_status_zoom.value() + 25)
                )
        elif action_id == "zoom_out":
            if self.central_stack.currentWidget() == self.media_viewer:
                self.media_viewer.zoom_out()
            elif hasattr(self, "duplicate_finder_view") and self.central_stack.currentWidget() == self.duplicate_finder_view:
                self.duplicate_finder_view._zoom_out()
            else:
                self.slider_status_zoom.setValue(
                    max(self.slider_status_zoom.minimum(), self.slider_status_zoom.value() - 25)
                )
        elif action_id in ("viewer_zoom_actual", "zoom_actual"):
            if self.central_stack.currentWidget() == self.media_viewer:
                self.media_viewer.set_zoom_mode("100")
        elif action_id in ("viewer_zoom_fit", "zoom_fit"):
            if self.central_stack.currentWidget() == self.media_viewer:
                if self._is_viewer_image_bigger_than_window():
                    self.media_viewer.set_zoom_mode("fit")
                else:
                    self.media_viewer.set_zoom_mode("fit_window")
                self._update_status_zoom_toggle_button()
        elif action_id == "toggle_pause_write_back":
            self.toggle_pause_write_back()
        elif action_id == "print":
            self.open_print_dialog()
        elif action_id in ("panorama", "panoramic"):
            self.open_panorama_dialog()
        elif action_id in ("collage", "auto_collage"):
            self.open_collage_dialog()
        elif action_id in ("photo_fuse", "fuse"):
            self.open_photo_fuse_dialog()
        elif action_id in ("markup_redact", "markup"):
            self.toggle_markup_mode()
        elif action_id == "color_pop":
            self.open_color_pop_dialog()
        elif action_id == "portrait_bokeh":
            self.open_portrait_bokeh_dialog()
        elif action_id == "cinematic_motion":
            self.open_cinematic_motion_dialog()
        elif action_id == "backup_catalog":
            self.backup_catalog_dialog()
        elif action_id == "open_settings":
            self._open_settings_dialog()
        elif action_id in ("about_photo_gallery", "about_app"):
            self._open_about_dialog()

    def open_print_dialog(self) -> None:
        """Opens the fully-featured Print Pictures dialog for active/selected photos."""
        from src.ui.print_dialog import PrintPicturesDialog

        items_to_print: list[Any] = []

        if self.central_stack.currentWidget() == self.media_viewer:
            if getattr(self.media_viewer, "current_is_video", False):
                QMessageBox.information(self, "Print", "Videos cannot be printed. Please select a photo to print.")
                return
            if self.media_viewer.master_np_rgb is not None:
                # If adjustments exist, render adjusted image
                try:
                    if hasattr(self.media_viewer, "get_current_display_pixmap"):
                        pm = self.media_viewer.get_current_display_pixmap()
                        qimg = pm.toImage()
                    else:
                        rendered_np = self.media_viewer.display_proxy_rgb or self.media_viewer.master_np_rgb
                        h, w = rendered_np.shape[:2]
                        qimg = QImage(rendered_np.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
                except Exception:
                    rendered_np = self.media_viewer.master_np_rgb
                    h, w = rendered_np.shape[:2]
                    qimg = QImage(rendered_np.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

                fname = "Photo"
                if self.media_viewer.current_file_path:
                    fname = self.media_viewer.current_file_path.name
                items_to_print.append((fname, qimg))
        else:
            # Grid view selection
            target_ids = list(self._selected_ids)
            if not target_ids:
                cur_id = getattr(self.grid_view, "_last_selected_id", None)
                if cur_id:
                    target_ids = [cur_id]

            for mid in target_ids:
                try:
                    detail = self.catalog_repo.get_media_detail(mid)
                    if detail.get("is_video"):
                        continue
                    full_path = Path(detail["folder_path"]) / str(detail["filename"])
                    if full_path.is_file():
                        items_to_print.append(full_path)
                except Exception:
                    continue

        if not items_to_print:
            QMessageBox.information(self, "Print", "No pictures are selected to print.")
            return

        dlg = PrintPicturesDialog(items_to_print, self)
        dlg.exec()

    def _get_selected_photo_paths_and_details(self) -> tuple[list[Path], list[dict[str, Any]]]:
        """Retrieves file paths and catalog metadata records for the active or selected photos."""
        paths: list[Path] = []
        details: list[dict[str, Any]] = []

        if self.central_stack.currentWidget() == self.media_viewer:
            if getattr(self.media_viewer, "current_is_video", False):
                return [], []
            p = Path(self.media_viewer.current_file_path) if self.media_viewer.current_file_path else None
            if p is None and self.media_viewer.current_media_id is not None:
                try:
                    det = self.catalog_repo.get_media_detail(self.media_viewer.current_media_id)
                    p = Path(det["folder_path"]) / str(det["filename"])
                except Exception:
                    p = None

            bgr = None
            if getattr(self.media_viewer, "master_np_rgb", None) is not None:
                import cv2
                bgr = cv2.cvtColor(self.media_viewer.master_np_rgb, cv2.COLOR_RGB2BGR)
            elif getattr(self.media_viewer, "current_pil_image", None) is not None:
                import cv2
                rgb = np.array(self.media_viewer.current_pil_image.convert("RGB"))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            elif p and p.is_file():
                import cv2
                bgr = cv2.imread(str(p))

            if p is not None:
                paths.append(p)
            elif bgr is not None:
                paths.append(Path("photo.jpg"))

            d: dict[str, Any] = {
                "file_path": str(p) if p else "",
                "filename": p.name if p else "photo.jpg",
                "folder_path": str(p.parent) if p else "",
            }
            if bgr is not None:
                d["image_bgr"] = bgr
            details.append(d)
        else:
            target_ids = list(self._selected_ids)
            if not target_ids:
                cur_id = getattr(self.grid_view, "_last_selected_id", None)
                if cur_id:
                    target_ids = [cur_id]

            for mid in target_ids:
                try:
                    det = self.catalog_repo.get_media_detail(mid)
                    if det.get("is_video"):
                        continue
                    folder_p = det.get("folder_path")
                    fname = det.get("filename")
                    full_p = Path(det.get("file_path") or (Path(folder_p) / str(fname)))
                    if full_p.is_file():
                        paths.append(full_p)
                        det["file_path"] = str(full_p)
                        details.append(det)
                except Exception:
                    continue

        return paths, details

    def _is_path_in_library(self, folder_path: Path) -> bool:
        """Checks whether folder_path is an existing library folder or within one."""
        try:
            target_norm = folder_path.resolve()
            folders = self.catalog_repo.get_folders()
            for f in folders:
                f_path = Path(f["path"]).resolve()
                if target_norm == f_path or target_norm.is_relative_to(f_path):
                    return True
        except Exception:
            pass
        return False

    def _on_creation_saved(self, out_file: Path, tool_name: str) -> None:
        """
        Processes a newly created media file.
        If saved within a gallery library folder, scans it so it appears in the gallery and selects it.
        If saved to an external location (e.g. Desktop, Downloads), does NOT add the external folder to library.
        """
        if self._is_path_in_library(out_file.parent):
            self.scan_folder(out_file.parent, include_subfolders=False)
            self.rebuild_and_run_query()
            self._select_file_in_grid(out_file)
            self.show_status_message(f"✓ Created {tool_name}: {out_file.name}", 4000)
        else:
            self.show_status_message(f"✓ Saved {tool_name} to: {out_file.name}", 4000)

    def open_panorama_dialog(self) -> None:
        """Launches the Microsoft ICE-Style Panorama Stitcher."""
        from src.ui.create_studio_dialogs import PanoramaDialog

        paths, _ = self._get_selected_photo_paths_and_details()
        if len(paths) < 2:
            QMessageBox.information(
                self,
                "Panorama",
                "Please select 2 or more overlapping photos to stitch a panorama.",
            )
            return

        dlg = PanoramaDialog(paths, self)
        if dlg.exec() == QDialog.Accepted and dlg.saved_file_path:
            self._on_creation_saved(dlg.saved_file_path, "panorama")

    def open_collage_dialog(self) -> None:
        """Launches the Interactive Collage & 'Then & Now' Studio."""
        from src.ui.create_studio_dialogs import CollageDialog

        _, details = self._get_selected_photo_paths_and_details()
        if not details:
            QMessageBox.information(
                self,
                "Collage",
                "Please select 1 to 20 photos to generate a collage.",
            )
            return

        dlg = CollageDialog(details, self)
        if dlg.exec() == QDialog.Accepted and dlg.saved_file_path:
            self._on_creation_saved(dlg.saved_file_path, "collage")

    def open_photo_fuse_dialog(self) -> None:
        """Launches the Photo Fuse Best-Face Group Shot Combiner."""
        from src.ui.create_studio_dialogs import PhotoFuseDialog

        paths, _ = self._get_selected_photo_paths_and_details()
        if len(paths) < 2:
            QMessageBox.information(
                self,
                "Photo Fuse",
                "Please select 2 to 6 burst group photos to fuse faces.",
            )
            return

        dlg = PhotoFuseDialog(paths, self)
        if dlg.exec() == QDialog.Accepted and dlg.saved_file_path:
            self._on_creation_saved(dlg.saved_file_path, "fused photo")

    def toggle_markup_mode(self) -> None:
        """Toggles the Quick Markup, Callouts & Privacy Redaction tool in the viewer."""
        if self.central_stack.currentWidget() != self.media_viewer:
            sel = self.grid_view.selected_media_ids() or list(self._selected_ids)
            if sel:
                self.open_single_viewer(sel[0])
            else:
                return

        new_state = not getattr(self.media_viewer, "is_markup_mode", False)
        self.media_viewer.set_markup_mode(new_state)

    def _get_single_selected_photo_and_bgr(self) -> tuple[Path | None, Any | None]:
        """Helper returning (file_path, image_bgr) for the active photo."""
        import cv2
        paths, details = self._get_selected_photo_paths_and_details()
        if not paths and not details:
            return None, None

        p = paths[0] if paths else None
        bgr = None
        if details and "image_bgr" in details[0] and details[0]["image_bgr"] is not None:
            bgr = details[0]["image_bgr"]
        elif p and p.is_file():
            bgr = cv2.imread(str(p))

        return p, bgr

    def open_color_pop_dialog(self) -> None:
        """Launches the Google Photos-Style Colour Pop Studio."""
        from src.ui.create_studio_dialogs import ColourPopDialog

        p, bgr = self._get_single_selected_photo_and_bgr()
        if bgr is None:
            QMessageBox.information(self, "Colour Pop", "Please select a photo to apply Colour Pop.")
            return

        dlg = ColourPopDialog(bgr, p, self)
        if dlg.exec() == QDialog.Accepted and dlg.saved_file_path:
            self._on_creation_saved(dlg.saved_file_path, "Colour Pop photo")

    def open_portrait_bokeh_dialog(self) -> None:
        """Opens photo in single image viewer with Depth & light panel active."""
        if self.central_stack.currentWidget() != self.media_viewer:
            sel = self.grid_view.selected_media_ids() or list(self._selected_ids)
            if sel:
                self.open_single_viewer(sel[0])
            else:
                QMessageBox.information(self, "Depth & light", "Please select a photo to adjust Depth & light.")
                return
        self.toggle_right_pane("depth_light")

    def open_cinematic_motion_dialog(self) -> None:
        """Launches the 3D Cinematic Motion Video Generator."""
        from src.ui.create_studio_dialogs import CinematicMotionDialog

        p, bgr = self._get_single_selected_photo_and_bgr()
        if bgr is None:
            QMessageBox.information(self, "3D Cinematic", "Please select a photo to generate 3D Cinematic Motion video.")
            return

        dlg = CinematicMotionDialog(bgr, p, self)
        if dlg.exec() == QDialog.Accepted and dlg.saved_file_path:
            self._on_creation_saved(dlg.saved_file_path, "3D Cinematic video")

    def _select_file_in_grid(self, file_path: Path) -> None:
        try:
            conn = self.catalog_repo.engine.catalog_conn()
            row = conn.execute(
                "SELECT m.id FROM media m JOIN folders f ON f.id = m.folder_id WHERE f.path = ? AND m.filename = ?",
                (str(file_path.parent), file_path.name),
            ).fetchone()
            if row:
                self.grid_view.select_media_id(row[0])
        except Exception:
            pass


    def _open_settings_dialog(self, initial_tab: str | None = None) -> None:
        dlg = AppSettingsDialog(
            retention_days=self.retention_days,
            current_theme=self.current_theme,
            show_dates=getattr(self.left_nav_dock, "_show_dates", True),
            show_tags=getattr(self.left_nav_dock, "_show_tags", True),
            show_people=getattr(self.left_nav_dock, "_show_people", True),
            show_ratings=getattr(self.left_nav_dock, "_show_ratings", True),
            show_places=getattr(self.left_nav_dock, "_show_places", True),
            show_albums=getattr(self.left_nav_dock, "_show_albums", True),
            find_people_enabled=getattr(self, "find_people_enabled", True),
            tooltip_previews=getattr(self, "tooltip_previews", True),
            jpeg_quality=getattr(self, "jpeg_quality", 83),
            auto_straighten=getattr(self, "auto_straighten", True),
            auto_noise=getattr(self, "auto_noise", False),
            auto_color=getattr(self, "auto_color", True),
            auto_exposure=getattr(self, "auto_exposure", True),
            originals_dir=str(Path.home() / "Pictures" / ".OriginalFiles"),
            pause_write_back=self.write_back_paused,
            icon_theme=getattr(self, "current_icon_theme", icon_provider.get_icon_theme()),
            engine=self.engine,
            initial_tab=initial_tab,
            parent=self,
        )
        def _apply_dlg_settings() -> None:
            self.retention_days = dlg.retention_days()
            new_theme = dlg.selected_theme()
            new_icon_theme = dlg.selected_icon_theme()
            theme_or_icons_changed = (new_theme != self.current_theme)
            if new_theme != self.current_theme:
                self.apply_theme(new_theme)

            if self.current_icon_theme != new_icon_theme or icon_provider.get_icon_theme() != new_icon_theme:
                self.current_icon_theme = new_icon_theme
                icon_provider.set_icon_theme(new_icon_theme)
                theme_or_icons_changed = True

            self._on_icon_theme_changed(new_icon_theme)
            if theme_or_icons_changed:
                self.refresh_navigation_and_autocomplete()
            elif hasattr(self, "left_nav_dock") and self.left_nav_dock is not None:
                self.left_nav_dock.refresh_icons()

            self.toggle_pause_write_back(dlg.pause_write_back_enabled())

            # Update Navigation Pane Sections
            self.nav_show_dates = dlg.show_dates()
            self.nav_show_tags = dlg.show_tags()
            self.nav_show_people = dlg.show_people()
            self.nav_show_ratings = dlg.show_ratings()
            self.nav_show_places = dlg.show_places()
            self.nav_show_albums = dlg.show_albums()
            self.left_nav_dock.set_navigation_sections_visibility(
                show_dates=self.nav_show_dates,
                show_tags=self.nav_show_tags,
                show_people=self.nav_show_people,
                show_ratings=self.nav_show_ratings,
                show_places=self.nav_show_places,
                show_albums=self.nav_show_albums,
            )

            old_find_people = getattr(self, "find_people_enabled", True)
            self.find_people_enabled = dlg.find_people_enabled()
            if self.find_people_enabled and not old_find_people:
                self.start_face_detection()
            elif not self.find_people_enabled and old_find_people:
                self.stop_face_detection()

            self.tooltip_previews = dlg.tooltip_previews_enabled()
            self.jpeg_quality = dlg.jpeg_quality()
            self.auto_straighten = dlg.auto_straighten()
            self.auto_noise = dlg.auto_noise()
            self.auto_color = dlg.auto_color()
            self.auto_exposure = dlg.auto_exposure()

            # Save to QSettings
            settings = QSettings("PhotoGallery64", "PhotoGallery64")
            settings.setValue("theme", self.current_theme)
            settings.setValue("icon_theme", getattr(self, "current_icon_theme", "wlpg_aero"))
            settings.setValue("nav_show_dates", self.nav_show_dates)
            settings.setValue("nav_show_tags", self.nav_show_tags)
            settings.setValue("nav_show_people", self.nav_show_people)
            settings.setValue("nav_show_ratings", self.nav_show_ratings)
            settings.setValue("nav_show_places", self.nav_show_places)
            settings.setValue("nav_show_albums", self.nav_show_albums)
            settings.setValue("find_people_enabled", self.find_people_enabled)
            settings.setValue("tooltip_previews", self.tooltip_previews)
            settings.setValue("jpeg_quality", self.jpeg_quality)
            settings.setValue("retention_days", self.retention_days)
            settings.setValue("auto_straighten", self.auto_straighten)
            settings.setValue("auto_noise", self.auto_noise)
            settings.setValue("auto_color", self.auto_color)
            settings.setValue("auto_exposure", self.auto_exposure)
            settings.setValue("save_copy_location", dlg.save_copy_location())
            settings.setValue("save_copy_custom_folder", dlg.save_copy_custom_folder())
            settings.setValue("save_copy_naming_pattern", dlg.save_copy_naming_pattern())
            settings.setValue("save_copy_switch_view", dlg.save_copy_switch_view())
            settings.setValue("unsaved_edits_action", dlg.unsaved_edits_action())

        dlg.settings_applied.connect(_apply_dlg_settings)
        if dlg.exec() == AppSettingsDialog.Accepted:
            _apply_dlg_settings()

            if dlg.compact_requested:
                self.catalog_repo.compact_database()
                self.thumbnail_repo.compact_database()

            if dlg.prune_backups_requested:
                expired = self.catalog_repo.prune_expired_backups(self.retention_days)
                for p in expired:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except OSError:
                        pass

            if dlg.backup_requested:
                QTimer.singleShot(100, self.backup_catalog_dialog)

            if dlg.heal_faces_requested:
                count = self.catalog_repo.reparse_and_heal_face_regions_from_files()
                self.statusBar().showMessage(
                    f"Re-parsed and queued face healing for {count} photos.", 5000
                )
                self.start_face_detection()

        if getattr(dlg, "purge_rescan_requested", False) or getattr(dlg, "rescan_unscanned_requested", False):
            if hasattr(self, "batch_people_view") and self.batch_people_view:
                self.batch_people_view.refresh_suggestions()
            if hasattr(self, "people_directory_view") and self.people_directory_view:
                self.people_directory_view.reload_directory()
            self.start_face_detection()
        if getattr(dlg, "fast_rematch_requested", False):
            if hasattr(self, "batch_people_view") and self.batch_people_view:
                self.batch_people_view.refresh_suggestions()
            if hasattr(self, "people_directory_view") and self.people_directory_view:
                self.people_directory_view.reload_directory()

    def _open_about_dialog(self) -> None:
        from src.ui.dialogs import AboutDialog
        dlg = AboutDialog(self)
        dlg.exec()

    def toggle_pause_write_back(self, paused: bool | None = None) -> None:
        """Toggles or sets the Pause Background File Write-Back (Catalog-Only Mode) setting."""
        if paused is None:
            self.write_back_paused = not self.write_back_paused
        else:
            self.write_back_paused = bool(paused)

        self.xmp_daemon.set_paused(self.write_back_paused)
        self.ribbon.set_write_back_paused(self.write_back_paused)

        settings = QSettings("PhotoGallery64", "PhotoGallery64")
        settings.setValue("write_back_paused", self.write_back_paused)

        try:
            from src.utils.paths import get_app_data_dir
            settings_json_path = get_app_data_dir() / "settings.json"
            pref_data = {}
            if settings_json_path.exists():
                try:
                    with open(settings_json_path, "r", encoding="utf-8") as f:
                        pref_data = json.load(f)
                except Exception:
                    pref_data = {}
            pref_data["write_back_paused"] = bool(self.write_back_paused)
            with open(settings_json_path, "w", encoding="utf-8") as f:
                json.dump(pref_data, f, indent=2)
        except Exception:
            pass

        self._update_worker_monitor()

        if not self.write_back_paused:
            self.statusBar().showMessage("✓ Write to files turned on (Writing changes to files)", 3000)
            self.xmp_daemon.flush_now()
        else:
            dirty = self.catalog_repo.count_dirty_media()
            pending_txt = f" ({dirty:,} pending)" if dirty > 0 else ""
            self.statusBar().showMessage(f"Write to files turned off (Catalog-only mode{pending_txt})", 3000)

    def open_tag_hierarchy_dialog(self) -> None:
        """Opens the 'All of your tags' dialog matching WLPG with drag-and-drop hierarchy organization."""
        from src.ui.tag_hierarchy_dialog import TagHierarchyDialog
        dlg = TagHierarchyDialog(self, repo=self.catalog_repo)
        tags = self.catalog_repo.get_tag_tree()
        dlg.populate(tags)
        dlg.tag_hierarchy_changed.connect(self.refresh_navigation_and_autocomplete)
        dlg.tags_added.connect(self._on_hierarchy_tags_selected)
        dlg.exec()

    def open_duplicate_finder(self) -> None:
        """Opens the modern integrated duplicate & burst finder view."""
        self.show_duplicate_finder_view()

    def show_duplicate_finder_view(self) -> None:
        """Switches central stack to integrated duplicate & burst finder view."""
        if hasattr(self, "duplicate_finder_view") and self.central_stack.currentWidget() == self.duplicate_finder_view:
            return

        # Save previous docks visibility
        self._nav_dock_was_visible = self.left_nav_dock.isVisible()
        self._tag_dock_was_visible = self.right_tag_dock.isVisible()
        self.left_nav_dock.hide()
        self.right_tag_dock.hide()

        # Switch view immediately so the user always sees the Duplicate Finder
        if hasattr(self, "duplicate_finder_view"):
            self.central_stack.setCurrentWidget(self.duplicate_finder_view)
            self.setWindowTitle("Find Duplicates & Bursts - Essentials Photo Gallery")

        records = getattr(self.catalog_model, "_records", None) if hasattr(self, "catalog_model") else None
        sel_ids = set(self.grid_view.selected_media_ids() or ())
        sel_records = [r for r in (records or []) if r.get("id") in sel_ids] if (sel_ids and records) else None

        active_folder_path: str | None = None
        fid = getattr(self.left_nav_dock, "_folder_id", None)
        if fid is not None:
            try:
                folder_id_val = int(fid[0]) if isinstance(fid, (tuple, list)) else int(fid)
                folder_info = self.catalog_repo.get_folder_by_id(folder_id_val)
                if folder_info:
                    active_folder_path = folder_info.get("path")
            except Exception as e:
                logger.debug(f"Could not resolve active folder path for duplicate finder: {e}")

        if hasattr(self, "duplicate_finder_view"):
            try:
                self.duplicate_finder_view.set_media_records(
                    records=records,
                    selected_records=sel_records,
                    current_folder=active_folder_path,
                )
                self.duplicate_finder_view.set_dark_mode(getattr(self, "_dark_mode", False))
                self.duplicate_finder_view.start_scan()
            except Exception as e:
                logger.error(f"Error starting duplicate finder scan: {e}")

    def _on_duplicates_cleaned(self, count: int, freed_bytes: int) -> None:
        self.rebuild_and_run_query()
        self.refresh_navigation_and_autocomplete()
        s = float(freed_bytes)
        sz_str = f"{s / (1024 ** 2):.1f} MB" if s >= 1024 ** 2 else f"{s / 1024:.0f} KB"
        self.statusBar().showMessage(f"Cleaned {count} duplicate photos (freed {sz_str})", 6000)

    def clean_up_orphan_tags(self) -> None:
        """Finds any orphan root tags that duplicate existing hierarchical tags, prompts the user, and merges them."""
        conn = self.catalog_repo.engine.catalog_conn()
        rows = conn.execute(
            """
            SELECT root_t.id AS root_id, hier_t.id AS hier_id, root_t.name AS root_name, hier_t.full_path AS hier_path
            FROM tags root_t
            JOIN tags hier_t ON LOWER(hier_t.name) = LOWER(root_t.name) AND hier_t.parent_id IS NOT NULL
            WHERE root_t.parent_id IS NULL;
            """
        ).fetchall()
        if not rows:
            QMessageBox.information(
                self,
                "Clean Up Duplicate Tags",
                "No duplicate root tags found. All tags are properly organized in the hierarchy.",
            )
            return

        distinct_pairs: dict[int, tuple[int, str, str]] = {}
        for r in rows:
            rid = int(r["root_id"])
            if rid not in distinct_pairs:
                distinct_pairs[rid] = (int(r["hier_id"]), str(r["root_name"]), str(r["hier_path"]))

        lines = [f"  • {name} ➔ {path}" for _, (_, name, path) in distinct_pairs.items()]
        msg = (
            "Clean up duplicate root tags consolidates loose top-level tags into your existing category hierarchy.\n\n"
            "What this does:\n"
            "• Photos tagged with the loose root tag will be reassigned to the matching hierarchical tag.\n"
            "• The redundant top-level tag will be removed from your catalog.\n"
            "• You can restore these tags anytime via the Descriptive tag menu.\n\n"
            f"Found {len(distinct_pairs)} duplicate root tag(s) to merge:\n\n"
            + "\n".join(lines[:12])
            + ("\n  ..." if len(lines) > 12 else "")
            + "\n\nWould you like to proceed with merging these tags?"
        )
        reply = QMessageBox.question(
            self,
            "Clean Up Duplicate Root Tags",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            merged_count = 0
            self._last_tag_cleanup_backup = []
            for rid, (hid, rname, _) in distinct_pairs.items():
                m_rows = conn.execute("SELECT media_id FROM media_tags WHERE tag_id = ?;", (rid,)).fetchall()
                self._last_tag_cleanup_backup.append((rname, hid, [int(mr["media_id"]) for mr in m_rows]))
                self.catalog_repo.merge_tags(rid, hid)
                merged_count += 1
            self.refresh_navigation_and_autocomplete()
            self.statusBar().showMessage(
                f"✓ Merged {merged_count} duplicate root tag(s) into hierarchy. (Restore available from Descriptive tag menu)",
                6000,
            )

    def restore_last_tag_cleanup(self) -> None:
        """Restores the previous duplicate root tags merge operation."""
        backup = getattr(self, "_last_tag_cleanup_backup", None)
        if not backup:
            QMessageBox.information(
                self,
                "Restore Tags",
                "No recent tag cleanup history available to restore in this session.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Restore Cleaned Tags",
            f"Would you like to restore {len(backup)} previously merged root tag(s) and reassign their photos?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        restored = 0
        for root_name, hier_id, media_ids in backup:
            new_root_id = self.catalog_repo.create_tag(root_name)
            if media_ids:
                self.catalog_repo.assign_tag(media_ids, new_root_id)
            restored += 1

        self._last_tag_cleanup_backup = []
        self.refresh_navigation_and_autocomplete()
        self.statusBar().showMessage(f"✓ Restored {restored} root tag(s) and photo assignments", 5000)
        QMessageBox.information(
            self,
            "Restore Tags",
            f"Successfully restored {restored} root tag(s) and re-assigned their photos.",
        )

    def _on_hierarchy_tags_selected(self, tag_ids: list[int]) -> None:
        if hasattr(self, "left_nav_dock") and self.left_nav_dock is not None:
            for tid in tag_ids:
                self.left_nav_dock.set_tag_filter_state(tid, "include")
        else:
            self._active_tag_ids = list(set(self._active_tag_ids) | set(tag_ids))
            self.rebuild_and_run_query()

    def backup_catalog_dialog(self) -> None:
        """Prompts the user to save an online backup snapshot of catalog.db."""
        import datetime
        default_name = f"catalog_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        dest_path, _ = QFileDialog.getSaveFileName(
            self,
            "Backup Catalog Database",
            str(Path.home() / "Documents" / default_name),
            "SQLite Database (*.db);;All Files (*)",
        )
        if not dest_path:
            return
        try:
            self.engine.backup_catalog(dest_path)
            self.statusBar().showMessage(f"✓ Catalog backed up successfully to {Path(dest_path).name}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Backup Error", f"Failed to create database backup:\n{e}")

    def toggle_pause_all_workers(self) -> None:
        """Toggles pause/resume state across all background tasks (scanner, faces, and write-back)."""
        self._workers_paused = not self._workers_paused
        if self._workers_paused:
            self._scanner_cancel.set()
            self.stop_face_detection()
            self.xmp_daemon.set_paused(True)
            self.btn_pause_workers.setText("▶")
            self.btn_pause_workers.setToolTip("Resume background indexing, face detection, and file write-back")
            self.btn_pause_workers.setStyleSheet(
                """
                QPushButton {
                    font-family: 'Segoe UI Symbol', 'Segoe UI', sans-serif;
                    font-size: 11pt;
                    border: none;
                    background: transparent;
                    color: #D9822B;
                    padding: 0;
                    min-width: 18px;
                    max-width: 20px;
                    height: 18px;
                }
                QPushButton:hover {
                    color: #B56214;
                }
                """
            )
            self.statusBar().showMessage("Background workers paused", 3000)
        else:
            self._scanner_cancel.clear()
            self._face_cancel.clear()
            self.xmp_daemon.set_paused(self.write_back_paused)
            if getattr(self, "find_people_enabled", False):
                self.start_face_detection()
            self.btn_pause_workers.setText("⏸")
            self.btn_pause_workers.setToolTip("Pause background indexing, face detection, and file write-back")
            self.btn_pause_workers.setStyleSheet(
                """
                QPushButton {
                    font-family: 'Segoe UI Symbol', 'Segoe UI', sans-serif;
                    font-size: 11pt;
                    border: none;
                    background: transparent;
                    color: #1E395B;
                    padding: 0;
                    min-width: 18px;
                    max-width: 20px;
                    height: 18px;
                }
                QPushButton:hover {
                    color: #0078D7;
                }
                """
            )
            self.statusBar().showMessage("✓ Background workers resumed", 3000)
        self._update_worker_monitor()

    def _update_worker_monitor(self) -> None:
        """Updates the status bar live monitor for active background workers and safe-mode state."""
        try:
            dirty_count = self.catalog_repo.count_dirty_media()
        except Exception:
            dirty_count = 0

        is_write_paused = self.xmp_daemon.is_paused()
        if is_write_paused:
            sync_detail = f"💾 Write to files: Off ({dirty_count:,} pending)" if dirty_count > 0 else "💾 Write to files: Off"
        elif dirty_count > 0:
            sync_detail = f"💾 Write to files: Saving {dirty_count:,} pending"
        else:
            sync_detail = "💾 Write to files: Up to date"

        active_thumbs = len(getattr(self, "_active_thumb_flags", {}))
        thumb_detail = f"🖼️ Thumbnails: {active_thumbs} active" if active_thumbs > 0 else "🖼️ Thumbnails: Idle"
        is_crops = bool(getattr(self, "_face_crop_status_text", ""))
        crop_detail = f"👤 Face Crops: {self._face_crop_status_text}" if is_crops else "👤 Face Crops: Idle"

        scan_summary = self._scanner_status_text.removeprefix("Scanning: ")
        face_summary = self._face_status_text.removeprefix("Faces: ")
        detailed_tooltip = (
            f"📁 Indexing: {scan_summary}\n"
            f"👤 Face Detection: {face_summary}\n"
            f"{crop_detail}\n"
            f"{thumb_detail}\n"
            f"{sync_detail}\n\n"
            "Click for details / background options"
        )

        is_scanning = "Idle" not in self._scanner_status_text
        is_faces = (getattr(self, "_face_worker", None) is not None) or ("Idle" not in self._face_status_text)

        if getattr(self, "_targeted_face_scan_active", False):
            # Targeted rescan in progress; keep targeted message visible
            return

        if self._workers_paused:
            summary_text = "Workers Paused"
            style = "color: #D9822B; font-weight: bold; font-size: 8.5pt;"
        elif is_scanning:
            summary_text = f"Indexing: {scan_summary}"
            style = "color: #0066CC; font-size: 8.5pt;"
        elif is_faces:
            if "Idle" not in self._face_status_text:
                summary_text = face_summary
            else:
                summary_text = "Finding people: Starting..."
            style = "color: #0066CC; font-size: 8.5pt;"
        elif is_crops:
            summary_text = self._face_crop_status_text
            style = "color: #0066CC; font-size: 8.5pt;"
        elif active_thumbs > 0:
            summary_text = f"Loading thumbnails ({active_thumbs})"
            style = "color: #0066CC; font-size: 8.5pt;"
        elif not is_write_paused and dirty_count > 0:
            summary_text = f"Saving metadata: {dirty_count:,}"
            style = "color: #0066CC; font-size: 8.5pt;"
        else:
            summary_text = "No background tasks"
            style = "color: #5A738E; font-size: 8.5pt;"

        if hasattr(self, "lbl_worker_monitor"):
            self.lbl_worker_monitor.setText(summary_text)
            self.lbl_worker_monitor.setToolTip(detailed_tooltip)
            self.lbl_worker_monitor.setStyleSheet(
                f"QLabel {{ {style} padding: 1px 4px; border-radius: 3px; }} QLabel:hover {{ background: #E5F1FC; }}"
            )

    def _build_worker_details_menu(self) -> tuple[QMenu, dict[str, QAction]]:
        """Builds the Background Tasks popup menu and returns `(menu, actions)`."""
        menu = QMenu(self)
        menu.setStyleSheet(
            """
            QMenu {
                background: #FFFFFF;
                border: 1px solid #7592B5;
                padding: 4px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 9pt;
            }
            QMenu::item {
                padding: 4px 20px 4px 10px;
            }
            QMenu::item:selected {
                background-color: #E5F1FC;
                color: #0F6CBD;
                border-radius: 3px;
            }
            QMenu::item:disabled {
                color: #2B3E50;
                font-weight: 500;
            }
            QMenu::separator {
                height: 1px;
                background: #D8E5F0;
                margin: 4px 0;
            }
            """
        )

        header = menu.addAction("Background Tasks")
        header.setEnabled(False)
        menu.addSeparator()

        scan_state_label = self._scanner_status_text.removeprefix("Scanning: ")
        act_scan = menu.addAction(f"📁 Indexing: {scan_state_label}")
        act_scan.setEnabled(False)

        face_state_label = self._face_status_text.removeprefix("Faces: ")
        if not getattr(self, "find_people_enabled", True) and face_state_label == "Idle":
            face_state_label = "Off"
        act_face = menu.addAction(f"👤 Face Detection: {face_state_label} (Options...)")
        act_face.triggered.connect(lambda: self._open_settings_dialog(initial_tab="general"))

        try:
            dirty_count = self.catalog_repo.count_dirty_media()
        except Exception:
            dirty_count = 0
        is_write_paused = self.xmp_daemon.is_paused()
        if is_write_paused:
            sync_info = f"Off (Catalog-only mode - {dirty_count:,} pending)" if dirty_count > 0 else "Off (Catalog-only mode)"
        elif dirty_count > 0:
            sync_info = f"Writing changes ({dirty_count:,} pending)"
        else:
            sync_info = "Up to date"
        act_sync = menu.addAction(f"💾 Write to files: {sync_info} (Options...)")
        act_sync.triggered.connect(lambda: self._open_settings_dialog(initial_tab="general"))

        menu.addSeparator()

        act_toggle_wb = menu.addAction(
            "Turn On 'Write to files'" if is_write_paused else "Turn Off 'Write to files' (Catalog-only mode)"
        )
        act_toggle_wb.triggered.connect(self.toggle_pause_write_back)

        if self._workers_paused:
            act_toggle = menu.addAction("▶ Resume All Background Tasks")
            act_toggle.triggered.connect(self.toggle_pause_all_workers)
        else:
            act_toggle = menu.addAction("⏸ Pause All Background Tasks")
            act_toggle.triggered.connect(self.toggle_pause_all_workers)

        actions = {
            "scan": act_scan,
            "face": act_face,
            "write": act_sync,
            "toggle_write": act_toggle_wb,
            "toggle_all": act_toggle,
        }
        self._last_worker_menu = menu
        self._last_worker_menu_actions = actions
        return menu, actions

    def _show_worker_details_menu(self, event: Any = None) -> None:
        """Presents a clean detailed popup menu of background queues and worker statuses."""
        menu, _ = self._build_worker_details_menu()
        pos = self.lbl_worker_monitor.mapToGlobal(QPoint(0, -menu.sizeHint().height() - 4))
        menu.exec(pos)

    def show_status_message(self, message: str = "", timeout_ms: int = 4000) -> None:
        """Displays transient notifications in lbl_sync_status without hiding item counts or worker controls."""
        if hasattr(self, "lbl_sync_status"):
            seq = getattr(self, "_status_msg_seq", 0) + 1
            self._status_msg_seq = seq
            self.lbl_sync_status.setText(message)
            if hasattr(self, "sep_status_msg"):
                self.sep_status_msg.setVisible(bool(message and message.strip()))
            if timeout_ms > 0 and message:
                QTimer.singleShot(
                    timeout_ms,
                    lambda s=seq: self.show_status_message("", 0) if getattr(self, "_status_msg_seq", 0) == s else None,
                )

    def _update_revert_button_state(self) -> None:
        """Enables or disables 'Revert to original' ribbon button based on active viewer or selected grid items."""
        if not hasattr(self, "ribbon"):
            return
        if self.central_stack.currentWidget() == self.media_viewer:
            can_revert = False
            if self.media_viewer.current_file_path is not None:
                bpath = (
                    self.media_viewer.current_file_path.parent
                    / ".OriginalFiles"
                    / self.media_viewer.current_file_path.name
                )
                if bpath.exists():
                    can_revert = True
            rec = getattr(self.media_viewer, "current_recipe", None)
            has_recipe_edits = (rec is not None and not rec.is_default())
            has_unsaved = self.media_viewer.has_unsaved_adjustments()
            if not can_revert:
                can_revert = has_unsaved
            self.ribbon.set_revert_enabled(can_revert)
            self.ribbon.set_save_enabled(has_unsaved)
            self.ribbon.set_save_copy_enabled(self.media_viewer.current_file_path is not None)
        else:
            self.ribbon.set_save_enabled(False)
            self.ribbon.set_save_copy_enabled(False)
            sel = self.grid_view.selected_media_ids() if hasattr(self, "grid_view") else []
            can_revert = False
            if sel:
                for mid in sel[:25]:
                    detail = self.catalog_repo.get_media_detail(mid)
                    if detail:
                        bpath = detail.get("original_backup_path") or detail.get("edit_backup_path")
                        if bpath and Path(bpath).exists():
                            can_revert = True
                            break
                        fpath_str = detail.get("file_path")
                        if fpath_str:
                            p = Path(fpath_str)
                            if (p.parent / ".OriginalFiles" / p.name).exists():
                                can_revert = True
                                break
            self.ribbon.set_revert_enabled(can_revert)

    def _on_status_viewer_clicked(self) -> None:
        target = self._lead_selected_id or (self._selected_ids[0] if self._selected_ids else None)
        if target is not None:
            self.open_single_viewer(target)

    def _on_ribbon_tab_changed(self, index: int) -> None:
        if not hasattr(self.ribbon, "tab_widget") or not hasattr(self, "right_tag_dock"):
            return
        cur_tab = self.ribbon.tab_widget.widget(index)
        if cur_tab != getattr(self.ribbon, "tab_edit", None):
            if self.right_tag_dock.current_mode() in ("depth_light", "fine_tune"):
                self.right_tag_dock.set_mode("info")

    def _on_layout_mode_changed(self, mode: str) -> None:
        self.grid_view.set_layout_mode(mode)
        if self.central_stack.currentWidget() != self.grid_view:
            self.show_grid_view()

    def _on_grid_thumbnail_size_changed(self, size_px: int) -> None:
        if hasattr(self, "slider_status_zoom"):
            self.slider_status_zoom.blockSignals(True)
            self.slider_status_zoom.setValue(size_px)
            self.slider_status_zoom.blockSignals(False)
        if hasattr(self, "ribbon") and hasattr(self.ribbon, "slider_zoom"):
            self.ribbon.slider_zoom.blockSignals(True)
            self.ribbon.slider_zoom.setValue(size_px)
            self.ribbon.slider_zoom.blockSignals(False)

    def _on_ribbon_zoom_changed(self, size_px: int) -> None:
        self.slider_status_zoom.blockSignals(True)
        self.slider_status_zoom.setValue(size_px)
        self.slider_status_zoom.blockSignals(False)
        self.grid_view.set_thumbnail_size(size_px)

    def _on_status_zoom_changed(self, size_px: int) -> None:
        if self.central_stack.currentWidget() == self.media_viewer:
            # Map status slider (72..320) to viewer zoom (0.2x..4.0x, 150 -> 1.0x)
            factor = (size_px / 150.0) ** 2.0
            zoom_lvl = max(0.2, min(factor, 4.0))
            self.media_viewer.current_zoom_mode = "custom"
            self.media_viewer.graphics_view.set_zoom(zoom_lvl)
            return

        if hasattr(self, "duplicate_finder_view") and self.central_stack.currentWidget() == self.duplicate_finder_view:
            # Map status slider (72..320) to duplicate card width (220..650)
            t = (size_px - 72) / (320 - 72)
            card_w = int(220 + t * (650 - 220))
            self.duplicate_finder_view.slider_zoom.setValue(card_w)
            return

        self.ribbon.slider_zoom.blockSignals(True)
        self.ribbon.slider_zoom.setValue(size_px)
        self.ribbon.slider_zoom.blockSignals(False)
        self.grid_view.set_thumbnail_size(size_px)

    def _sync_status_slider_from_viewer(self, percent: int) -> None:
        if self.central_stack.currentWidget() != self.media_viewer:
            return
        # Inverse mapping: factor = percent / 100.0, factor = (size_px / 150)^2 => size_px = 150 * sqrt(factor)
        import math
        zoom_lvl = max(0.2, min(percent / 100.0, 4.0))
        size_px = int(round(150.0 * math.sqrt(zoom_lvl)))
        clamped = max(72, min(size_px, 320))
        self.slider_status_zoom.blockSignals(True)
        self.slider_status_zoom.setValue(clamped)
        self.slider_status_zoom.blockSignals(False)

    def _is_viewer_image_bigger_than_window(self) -> bool:
        if self.central_stack.currentWidget() != self.media_viewer:
            return False
        if self.media_viewer.master_np_rgb is None:
            return False
        gv = self.media_viewer.graphics_view
        vp = gv.viewport().size()
        sr = self.media_viewer.scene.sceneRect()
        if sr.isEmpty() or vp.isEmpty():
            return False
        cur_w = sr.width() * gv.transform().m11()
        cur_h = sr.height() * gv.transform().m22()
        return (cur_w > vp.width() + 2) or (cur_h > vp.height() + 2)

    def _on_status_zoom_toggle_clicked(self) -> None:
        if self._is_viewer_image_bigger_than_window():
            self.media_viewer.set_zoom_mode("fit")
        else:
            if getattr(self.media_viewer, "current_zoom_mode", "") == "100":
                self.media_viewer.set_zoom_mode("fit")
            else:
                self.media_viewer.set_zoom_mode("100")
        self._update_status_zoom_toggle_button()

    def _on_status_grid_or_details_clicked(self) -> None:
        """
        In Grid View: toggles 'All details' (equivalent to View -> All details).
        In Single Image View or other views: returns to Thumbnail Grid view.
        """
        if self.central_stack.currentWidget() == self.grid_view:
            if hasattr(self, "ribbon") and hasattr(self.ribbon, "chk_all_details"):
                self.ribbon.chk_all_details.setChecked(
                    not self.ribbon.chk_all_details.isChecked()
                )
            self._update_status_grid_button()
        else:
            self.trigger_thumbnail_view()

    def trigger_thumbnail_view(self) -> None:
        """Switches to Thumbnail View (Ctrl+0) from Single Image View, All Details mode, or auxiliary views."""
        self._prev_central_widget = None
        if hasattr(self, "_prev_ribbon_tab"):
            delattr(self, "_prev_ribbon_tab")
        if hasattr(self, "batch_people_view") and self.central_stack.currentWidget() == self.batch_people_view:
            self.close_batch_people_view()
        else:
            self.show_grid_view()
        if (
            self.central_stack.currentWidget() == self.grid_view
            and hasattr(self, "ribbon")
            and hasattr(self.ribbon, "chk_all_details")
            and self.ribbon.chk_all_details.isChecked()
        ):
            self.ribbon.chk_all_details.setChecked(False)
        self._update_status_grid_button()

    def _update_status_grid_button(self) -> None:
        if not hasattr(self, "btn_status_grid"):
            return
        in_grid = (
            hasattr(self, "central_stack")
            and hasattr(self, "grid_view")
            and self.central_stack.currentWidget() == self.grid_view
        )
        in_all_details = bool(
            hasattr(self, "ribbon")
            and hasattr(self.ribbon, "chk_all_details")
            and self.ribbon.chk_all_details.isChecked()
        )
        if in_grid and not in_all_details:
            self.btn_status_grid.setIcon(make_wlpg_icon("view_details", 18))
            self.btn_status_grid.setToolTip("View Details")
        else:
            self.btn_status_grid.setIcon(make_wlpg_icon("thumbnail_view", 18))
            self.btn_status_grid.setToolTip("Gallery view (Ctrl+0)")

    def _update_status_zoom_toggle_button(self) -> None:
        if not hasattr(self, "btn_status_fit"):
            return
        if self._is_viewer_image_bigger_than_window():
            self.btn_status_fit.setIcon(make_wlpg_icon("fit_window", 18))
            self.btn_status_fit.setToolTip("Fit to window")
        else:
            self.btn_status_fit.setIcon(make_wlpg_icon("actual_size", 18))
            self.btn_status_fit.setToolTip("Actual size (Ctrl+1)")

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._update_status_zoom_toggle_button()

    # -------------------------------------------------------------------------
    # Ribbon Exposure, Color & Effects Real-Time Previews & Commitments
    # -------------------------------------------------------------------------
    def _on_exposure_preview(self, brightness: float, contrast: float) -> None:
        if self.central_stack.currentWidget() == self.media_viewer:
            rec = self.media_viewer.current_recipe.copy()
            rec.brightness = brightness
            rec.contrast = contrast
            self.media_viewer.preview_recipe(rec)

    def _on_auto_exposure_preview(self) -> None:
        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.master_np_rgb is not None
        ):
            from src.imaging.opencv_engine import analyze_auto_adjust
            auto_rec = analyze_auto_adjust(self.media_viewer.master_np_rgb)
            rec = self.media_viewer.current_recipe.copy()
            rec.brightness = auto_rec.brightness
            rec.contrast = auto_rec.contrast
            rec.highlights = auto_rec.highlights
            rec.shadows = auto_rec.shadows
            self.media_viewer.preview_recipe(rec)

    def _on_preview_cancelled(self) -> None:
        if self.central_stack.currentWidget() == self.media_viewer:
            self.media_viewer.preview_recipe(None)

    def _on_exposure_selected(self, brightness: float, contrast: float) -> None:
        if hasattr(self.ribbon, "menu_exposure"):
            self.ribbon.menu_exposure.close()
        if self.central_stack.currentWidget() == self.media_viewer:
            new_rec = self.media_viewer.current_recipe.copy()
            new_rec.brightness = brightness
            new_rec.contrast = contrast
            self.media_viewer.apply_recipe_adjustment(new_rec, "Adjust Exposure")
        else:
            sel = self.grid_view.selected_media_ids()
            if sel:
                from src.imaging.opencv_engine import render_adjusted_image, PhotoAdjustmentRecipe

                def _adjust(rgb: np.ndarray) -> np.ndarray:
                    return render_adjusted_image(
                        rgb, PhotoAdjustmentRecipe(brightness=brightness, contrast=contrast)
                    )

                self._batch_apply_image_adjustment(sel, "Adjust Exposure", _adjust)

    def _on_auto_exposure_selected(self) -> None:
        if hasattr(self.ribbon, "menu_exposure"):
            self.ribbon.menu_exposure.close()
        if self.central_stack.currentWidget() == self.media_viewer:
            self.media_viewer.apply_auto_exposure()
        else:
            sel = self.grid_view.selected_media_ids()
            if sel:
                from src.imaging.opencv_engine import analyze_auto_adjust, render_adjusted_image, PhotoAdjustmentRecipe

                def _adjust(rgb: np.ndarray) -> np.ndarray:
                    auto_rec = analyze_auto_adjust(rgb)
                    exp_rec = PhotoAdjustmentRecipe(
                        brightness=auto_rec.brightness,
                        contrast=auto_rec.contrast,
                        highlights=auto_rec.highlights,
                        shadows=auto_rec.shadows,
                    )
                    return render_adjusted_image(rgb, exp_rec)

                self._batch_apply_image_adjustment(sel, "Auto Adjust Exposure", _adjust)

    def _on_exposure_menu_about_to_show(self) -> None:
        base_img = None
        if self.central_stack.currentWidget() == self.media_viewer:
            base_img = self.media_viewer.get_preview_thumbnail(54, 38)
        elif self.central_stack.currentWidget() == self.grid_view:
            sel = self.grid_view.selected_media_ids()
            if sel and sel[0] in self.pixmap_cache:
                pm = self.pixmap_cache[sel[0]]
                qimg = pm.toImage().convertToFormat(QImage.Format_RGB888)
                ptr = qimg.bits()
                ptr.setsize(qimg.sizeInBytes())
                base_img = np.array(ptr, copy=True).reshape((qimg.height(), qimg.width(), 3))
        self.ribbon.exposure_menu_widget.refresh_thumbnails(base_img)

    def _on_color_preview(self, temp: float, tint: float) -> None:
        if self.central_stack.currentWidget() == self.media_viewer:
            rec = self.media_viewer.current_recipe.copy()
            rec.temperature = temp
            rec.tint = tint
            self.media_viewer.preview_recipe(rec)

    def _on_auto_color_preview(self) -> None:
        if (
            self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.master_np_rgb is not None
        ):
            from src.imaging.opencv_engine import analyze_auto_adjust
            auto_rec = analyze_auto_adjust(self.media_viewer.master_np_rgb)
            rec = self.media_viewer.current_recipe.copy()
            rec.temperature = auto_rec.temperature
            rec.tint = auto_rec.tint
            rec.saturation = auto_rec.saturation
            self.media_viewer.preview_recipe(rec)

    def _on_color_selected(self, temp: float, tint: float) -> None:
        if hasattr(self.ribbon, "menu_color"):
            self.ribbon.menu_color.close()
        if self.central_stack.currentWidget() == self.media_viewer:
            new_rec = self.media_viewer.current_recipe.copy()
            new_rec.temperature = temp
            new_rec.tint = tint
            self.media_viewer.apply_recipe_adjustment(new_rec, "Adjust Color")
        else:
            sel = self.grid_view.selected_media_ids()
            if sel:
                from src.imaging.opencv_engine import render_adjusted_image, PhotoAdjustmentRecipe

                def _adjust(rgb: np.ndarray) -> np.ndarray:
                    return render_adjusted_image(
                        rgb, PhotoAdjustmentRecipe(temperature=temp, tint=tint)
                    )

                self._batch_apply_image_adjustment(sel, "Adjust Color", _adjust)

    def _on_auto_color_selected(self) -> None:
        if hasattr(self.ribbon, "menu_color"):
            self.ribbon.menu_color.close()
        if self.central_stack.currentWidget() == self.media_viewer:
            self.media_viewer.apply_auto_color()
        else:
            sel = self.grid_view.selected_media_ids()
            if sel:
                from src.imaging.opencv_engine import analyze_auto_adjust, render_adjusted_image, PhotoAdjustmentRecipe

                def _adjust(rgb: np.ndarray) -> np.ndarray:
                    auto_rec = analyze_auto_adjust(rgb)
                    col_rec = PhotoAdjustmentRecipe(
                        temperature=auto_rec.temperature,
                        tint=auto_rec.tint,
                        saturation=auto_rec.saturation,
                    )
                    return render_adjusted_image(rgb, col_rec)

                self._batch_apply_image_adjustment(sel, "Auto Adjust Color", _adjust)

    def _on_color_menu_about_to_show(self) -> None:
        base_img = None
        if self.central_stack.currentWidget() == self.media_viewer:
            base_img = self.media_viewer.get_preview_thumbnail(54, 38)
        elif self.central_stack.currentWidget() == self.grid_view:
            sel = self.grid_view.selected_media_ids()
            if sel and sel[0] in self.pixmap_cache:
                pm = self.pixmap_cache[sel[0]]
                qimg = pm.toImage().convertToFormat(QImage.Format_RGB888)
                ptr = qimg.bits()
                ptr.setsize(qimg.sizeInBytes())
                base_img = np.array(ptr, copy=True).reshape((qimg.height(), qimg.width(), 3))
        self.ribbon.color_menu_widget.refresh_thumbnails(base_img)

    def _on_effect_hovered(self, eff_id: str) -> None:
        if self.central_stack.currentWidget() == self.media_viewer:
            rec = self.media_viewer.current_recipe.copy()
            rec.effect = eff_id
            self.media_viewer.preview_recipe(rec)

    def _on_icon_theme_changed(self, _theme_name: str = "") -> None:
        """Refreshes status bar and right dock icons when the icon theme changes in Options."""
        if hasattr(self, "btn_status_prev"):
            self.btn_status_prev.setIcon(make_wlpg_icon("prev_media", 18))
        if hasattr(self, "btn_status_next"):
            self.btn_status_next.setIcon(make_wlpg_icon("next_media", 18))
        if hasattr(self, "btn_status_rot_ccw"):
            self.btn_status_rot_ccw.setIcon(make_wlpg_icon("rotate_ccw", 18))
        if hasattr(self, "btn_status_rot_cw"):
            self.btn_status_rot_cw.setIcon(make_wlpg_icon("rotate_cw", 18))
        if hasattr(self, "btn_status_delete"):
            self.btn_status_delete.setIcon(make_wlpg_icon("delete", 16))
        if hasattr(self, "btn_status_slideshow"):
            self.btn_status_slideshow.setIcon(make_wlpg_icon("slideshow", 18))
        if hasattr(self, "btn_status_zoom_out"):
            self.btn_status_zoom_out.setIcon(make_wlpg_icon("zoom_out", 18))
        if hasattr(self, "btn_status_zoom_in"):
            self.btn_status_zoom_in.setIcon(make_wlpg_icon("zoom_in", 18))
        self._update_status_grid_button()
        self._update_status_zoom_toggle_button()
        if hasattr(self, "right_tag_dock") and self.right_tag_dock is not None:
            if hasattr(self.right_tag_dock, "btn_copy_subject"):
                self.right_tag_dock.btn_copy_subject.setIcon(make_wlpg_icon("copy_subject", 16))
            if hasattr(self.right_tag_dock, "btn_remove_bg"):
                self.right_tag_dock.btn_remove_bg.setIcon(make_wlpg_icon("remove_background", 16))

    # -------------------------------------------------------------------------
    # Theme Application, Keyboard Shortcuts & Clean Shutdown
    # -------------------------------------------------------------------------
    def apply_theme(self, theme_name: str) -> None:
        self.setUpdatesEnabled(False)
        try:
            self.current_theme = theme_name  # type: ignore[assignment]
            is_dark = (theme_name == "dark")
            if theme_name == "modern":
                icon_provider.set_icon_theme("fluent")
                self.current_icon_theme = "fluent"
            elif theme_name == "wlpg_aero":
                icon_provider.set_icon_theme("wlpg_aero")
                self.current_icon_theme = "wlpg_aero"
            theme_file = get_themes_dir() / f"{theme_name}.qss"
            if theme_file.exists():
                qss = theme_file.read_text(encoding="utf-8")
                icons_dir = get_icons_dir().as_posix()
                qss = qss.replace("{ICONS_PATH}", icons_dir)
                self.setStyleSheet(qss)
                app = QApplication.instance()
                if app:
                    pal = app.palette()
                    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#FFFFFF"))
                    pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#1E395B"))
                    app.setPalette(pal)
                    QToolTip.setPalette(pal)

            # Propagate theme mode to ribbon and panels
            for w in (
                getattr(self, "ribbon", None),
                getattr(self, "left_nav_dock", None),
                getattr(self, "grid_view", None),
                getattr(self, "media_viewer", None),
                getattr(self, "right_tag_dock", None),
                getattr(self, "person_header_bar", None),
                getattr(self, "scope_bar", None),
                getattr(self, "people_directory_view", None),
                getattr(self, "albums_view", None),
                getattr(self, "batch_people_view", None),
                getattr(self, "duplicate_finder_view", None),
                getattr(self, "compare_view", None),
            ):
                if w is not None:
                    if hasattr(w, "apply_theme_mode"):
                        w.apply_theme_mode(theme_name)
                    elif hasattr(w, "set_dark_mode"):
                        w.set_dark_mode(is_dark)

            # Update status bar styling for dark mode vs light themes
            sb = self.statusBar()
            if sb is not None:
                if is_dark:
                    sb.setStyleSheet(
                        """
                        QStatusBar {
                            background: #171A20;
                            border-top: 1px solid #2A2E37;
                            color: #CBD5E1;
                            font-family: 'Segoe UI';
                            font-size: 8.5pt;
                        }
                        QStatusBar QLabel {
                            color: #CBD5E1;
                            background-color: transparent;
                        }
                        QStatusBar QPushButton {
                            border: 1px solid transparent;
                            background: transparent;
                            color: #CBD5E1;
                            border-radius: 2px;
                            min-width: 22px;
                            max-width: 26px;
                            height: 20px;
                        }
                        QStatusBar QPushButton:hover {
                            border: 1px solid #3B82F6;
                            background: #242A36;
                        }
                        QStatusBar QSlider::groove:horizontal {
                            height: 3px;
                            background: #333A45;
                            border-radius: 1px;
                        }
                        QStatusBar QSlider::handle:horizontal {
                            background: #CBD5E1;
                            border: 1px solid #3B82F6;
                            width: 6px;
                            height: 12px;
                            margin: -5px 0;
                            border-radius: 2px;
                        }
                        """
                    )
                    if hasattr(self, "lbl_status_counts"):
                        self.lbl_status_counts.setStyleSheet("color: #E2E8F0; padding-left: 6px; font-weight: bold; background: transparent;")
                    if hasattr(self, "btn_pause_workers"):
                        self.btn_pause_workers.setStyleSheet(
                            """
                            QPushButton {
                                font-family: 'Segoe UI Symbol', 'Segoe UI', sans-serif;
                                font-size: 11pt;
                                border: none;
                                background: transparent;
                                color: #CBD5E1;
                                padding: 0;
                                min-width: 18px;
                                max-width: 20px;
                                height: 18px;
                            }
                            QPushButton:hover {
                                color: #60A5FA;
                            }
                            """
                        )
                    if hasattr(self, "lbl_worker_monitor"):
                        self.lbl_worker_monitor.setStyleSheet(
                            """
                            QLabel {
                                color: #94A3B8;
                                font-size: 8.5pt;
                                padding: 1px 4px;
                                border-radius: 3px;
                                background: transparent;
                            }
                            QLabel:hover {
                                color: #60A5FA;
                                background: #242A36;
                            }
                            """
                        )
                    if hasattr(self, "lbl_sync_status"):
                        self.lbl_sync_status.setStyleSheet("color: #60A5FA; padding-left: 4px; background: transparent;")
                    if hasattr(self, "sep_status_msg"):
                        self.sep_status_msg.setStyleSheet("color: #333A45; margin: 3px 6px;")
                elif theme_name == "modern":
                    sb.setStyleSheet(
                        """
                        QStatusBar {
                            background: #F3F4F6;
                            border-top: 1px solid #E5E7EB;
                            color: #374151;
                            font-family: 'Segoe UI';
                            font-size: 8.5pt;
                        }
                        QStatusBar QLabel {
                            color: #374151;
                            background-color: transparent;
                        }
                        QStatusBar QPushButton {
                            border: 1px solid transparent;
                            background: transparent;
                            color: #374151;
                            border-radius: 2px;
                            min-width: 22px;
                            max-width: 26px;
                            height: 20px;
                        }
                        QStatusBar QPushButton:hover {
                            border: 1px solid #E5E7EB;
                            background: #E5E7EB;
                        }
                        QStatusBar QSlider::groove:horizontal {
                            height: 3px;
                            background: #E5E7EB;
                            border-radius: 1px;
                        }
                        QStatusBar QSlider::handle:horizontal {
                            background: #0F6CBD;
                            border: 1px solid #0F6CBD;
                            width: 6px;
                            height: 12px;
                            margin: -5px 0;
                            border-radius: 2px;
                        }
                        """
                    )
                    if hasattr(self, "lbl_status_counts"):
                        self.lbl_status_counts.setStyleSheet("color: #111827; padding-left: 6px; font-weight: bold; background: transparent;")
                    if hasattr(self, "btn_pause_workers"):
                        self.btn_pause_workers.setStyleSheet(
                            """
                            QPushButton {
                                font-family: 'Segoe UI Symbol', 'Segoe UI', sans-serif;
                                font-size: 11pt;
                                border: none;
                                background: transparent;
                                color: #374151;
                                padding: 0;
                                min-width: 18px;
                                max-width: 20px;
                                height: 18px;
                            }
                            QPushButton:hover {
                                color: #0F6CBD;
                            }
                            """
                        )
                    if hasattr(self, "lbl_worker_monitor"):
                        self.lbl_worker_monitor.setStyleSheet(
                            """
                            QLabel {
                                color: #6B7280;
                                font-size: 8.5pt;
                                padding: 1px 4px;
                                border-radius: 3px;
                                background: transparent;
                            }
                            QLabel:hover {
                                color: #0F6CBD;
                                background: #E5E7EB;
                            }
                            """
                        )
                    if hasattr(self, "lbl_sync_status"):
                        self.lbl_sync_status.setStyleSheet("color: #0F6CBD; padding-left: 4px; background: transparent;")
                    if hasattr(self, "sep_status_msg"):
                        self.sep_status_msg.setStyleSheet("color: #E5E7EB; margin: 3px 6px;")
                else:
                    sb.setStyleSheet(
                        """
                        QStatusBar {
                            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F0F4FA, stop:1 #D9E4F2);
                            border-top: 1px solid #B8CDE3;
                            color: #1E395B;
                            font-family: 'Segoe UI';
                            font-size: 8.5pt;
                        }
                        QStatusBar QLabel {
                            color: #1E395B;
                            background-color: transparent;
                        }
                        QStatusBar QPushButton {
                            border: 1px solid transparent;
                            background: transparent;
                            border-radius: 2px;
                            min-width: 22px;
                            max-width: 26px;
                            height: 20px;
                        }
                        QStatusBar QPushButton:hover {
                            border: 1px solid #7DA2CE;
                            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FFFFFF, stop:0.5 #E7F1FB, stop:1 #D0E4F9);
                        }
                        QStatusBar QPushButton:pressed {
                            border: 1px solid #5A85B8;
                            background: #BFDCF7;
                        }
                        QStatusBar QSlider::groove:horizontal {
                            height: 3px;
                            background: #B6C8DA;
                            border-radius: 1px;
                        }
                        QStatusBar QSlider::handle:horizontal {
                            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F0F6FA, stop:1 #D8E5F0);
                            border: 1px solid #7592B5;
                            width: 6px;
                            height: 12px;
                            margin: -5px 0;
                            border-radius: 2px;
                        }
                        QStatusBar QSlider::handle:horizontal:hover {
                            border: 1px solid #3380CC;
                            background: #E5F1FC;
                        }
                        """
                    )
                    if hasattr(self, "lbl_status_counts"):
                        self.lbl_status_counts.setStyleSheet("color: #1E395B; padding-left: 6px; font-weight: bold; background: transparent;")
                    if hasattr(self, "btn_pause_workers"):
                        self.btn_pause_workers.setStyleSheet(
                            """
                            QPushButton {
                                font-family: 'Segoe UI Symbol', 'Segoe UI', sans-serif;
                                font-size: 11pt;
                                border: none;
                                background: transparent;
                                color: #1E395B;
                                padding: 0;
                                min-width: 18px;
                                max-width: 20px;
                                height: 18px;
                            }
                            QPushButton:hover {
                                color: #0078D7;
                            }
                            """
                        )
                    if hasattr(self, "lbl_worker_monitor"):
                        self.lbl_worker_monitor.setStyleSheet(
                            """
                            QLabel {
                                color: #5A738E;
                                font-size: 8.5pt;
                                padding: 1px 4px;
                                border-radius: 3px;
                                background: transparent;
                            }
                            QLabel:hover {
                                color: #0078D7;
                                background: #E5F1FC;
                            }
                            """
                        )
                    if hasattr(self, "lbl_sync_status"):
                        self.lbl_sync_status.setStyleSheet("color: #0066CC; padding-left: 4px; background: transparent;")
                    if hasattr(self, "sep_status_msg"):
                        self.sep_status_msg.setStyleSheet("color: #AFC3D8; margin: 3px 6px;")
        finally:
            self.setUpdatesEnabled(True)
            self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        focus_w = QApplication.focusWidget() or self.focusWidget()
        if not focus_w and hasattr(self, "right_tag_dock") and self.right_tag_dock is not None:
            focus_w = self.right_tag_dock.focusWidget()
        is_text_entry = isinstance(focus_w, (QLineEdit, QTextEdit, QPlainTextEdit))

        if (
            not is_text_entry
            and hasattr(self, "compare_view")
            and self.central_stack.currentWidget() is self.compare_view
            and not (event.modifiers() & (Qt.ControlModifier | Qt.AltModifier))
        ):
            self.compare_view.keyPressEvent(event)
            if event.isAccepted():
                return

        # Star ratings: 0, 1, 2, 3, 4, 5 (when not focused on a text entry and no Ctrl/Alt modifier)
        if not is_text_entry and event.modifiers() in (Qt.NoModifier, Qt.KeypadModifier):
            if Qt.Key_0 <= event.key() <= Qt.Key_5:
                rating_val = event.key() - Qt.Key_0
                self.apply_rating_to_selection(rating_val)
                event.accept()
                return

        # Single Media Viewer: Left and Right arrow navigation
        if self.central_stack.currentWidget() == self.media_viewer:
            if not is_text_entry and not self.media_viewer.is_crop_mode:
                if event.key() == Qt.Key_Left and event.modifiers() in (Qt.NoModifier, Qt.KeypadModifier):
                    self.navigate_single_viewer(-1)
                    event.accept()
                    return
                elif event.key() == Qt.Key_Right and event.modifiers() in (Qt.NoModifier, Qt.KeypadModifier):
                    self.navigate_single_viewer(1)
                    event.accept()
                    return

        # Thumbnail Grid View: Arrow / Home / End / Page navigation (when not focused on a text entry)
        if self.central_stack.currentWidget() == self.grid_view:
            if not is_text_entry and not (event.modifiers() & (Qt.ControlModifier | Qt.AltModifier)):
                if event.key() in (
                    Qt.Key_Left,
                    Qt.Key_Right,
                    Qt.Key_Up,
                    Qt.Key_Down,
                    Qt.Key_Home,
                    Qt.Key_End,
                    Qt.Key_PageUp,
                    Qt.Key_PageDown,
                ):
                    self.grid_view.keyPressEvent(event)
                    event.accept()
                    return

        if event.key() in (Qt.Key_F11, Qt.Key_F12):
            self.dispatch_ribbon_action("slide_show")
            event.accept()
            return
        if event.key() == Qt.Key_Escape:
            if hasattr(self, "batch_people_view") and self.central_stack.currentWidget() == self.batch_people_view:
                self.close_batch_people_view()
                event.accept()
                return
            if self.central_stack.currentWidget() == self.media_viewer:
                if self.media_viewer.is_crop_mode:
                    self.media_viewer.set_crop_mode(False)
                elif getattr(self.media_viewer, "is_markup_mode", False):
                    # In markup mode, let media_viewer handle Esc to cancel pending elements,
                    # text notes, or magnifier placement rather than kicking the user out
                    return
                elif getattr(self.media_viewer, "is_retouch_mode", False) or getattr(self.media_viewer, "is_red_eye_mode", False):
                    self.media_viewer._close_spot_fix_toolbar()
                else:
                    self.show_grid_view()
                event.accept()
                return
            if hasattr(self, "duplicate_finder_view") and self.central_stack.currentWidget() == self.duplicate_finder_view:
                self.show_grid_view()
                event.accept()
                return
            if self.central_stack.currentWidget() == self.grid_view:
                if self._active_map_media_ids is not None or self._active_bbox is not None:
                    self._active_map_media_ids = None
                    self._active_bbox = None
                    self._active_map_location_label = None
                    self.show_map_view()
                    event.accept()
                    return
            if self.central_stack.currentWidget() == self.map_view:
                self.show_grid_view()
                event.accept()
                return
            if hasattr(self, "person_header_bar") and self.person_header_bar.isVisible():
                if self._nav_index > 0:
                    self.navigate_history_back()
                else:
                    self.open_people_directory_view()
                event.accept()
                return
            if hasattr(self, "people_directory_view") and self.central_stack.currentWidget() == self.people_directory_view:
                if self._nav_index > 0:
                    self.navigate_history_back()
                else:
                    self.show_grid_view()
                event.accept()
                return
            if hasattr(self, "scope_bar") and self.scope_bar.isVisible() and not self.scope_bar.is_pinned():
                self.hide_search_scope_bar()
                event.accept()
                return
            if self.isFullScreen():
                self.showNormal()
                event.accept()
                return

        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_T:
            self.focus_descriptive_tag_box()
            event.accept()
            return

        if not is_text_entry:
            # Enter / Return: Grid -> Open Single Viewer; Viewer -> Return to Grid
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if self.central_stack.currentWidget() == self.grid_view:
                    target_id = self._lead_selected_id or (self._selected_ids[0] if self._selected_ids else None)
                    if target_id is not None:
                        self.open_single_viewer(target_id)
                        event.accept()
                        return
                elif self.central_stack.currentWidget() == self.media_viewer:
                    self.show_grid_view()
                    event.accept()
                    return

            # Space: Grid -> Open Single Viewer; Viewer -> Next slide
            if event.key() == Qt.Key_Space:
                if self.central_stack.currentWidget() == self.grid_view:
                    target_id = self._lead_selected_id or (self._selected_ids[0] if self._selected_ids else None)
                    if target_id is not None:
                        self.open_single_viewer(target_id)
                        event.accept()
                        return
                elif self.central_stack.currentWidget() == self.media_viewer:
                    if not self.media_viewer.is_crop_mode:
                        self.navigate_single_viewer(1)
                        event.accept()
                        return

            # Clipboard shortcuts: Ctrl+X (Cut), Ctrl+C (Copy), Ctrl+V (Paste)
            if event.modifiers() == Qt.ControlModifier:
                if event.key() == Qt.Key_X:
                    self.cut_selection_to_clipboard()
                    event.accept()
                    return
                elif event.key() == Qt.Key_C:
                    self.copy_selection_to_clipboard()
                    event.accept()
                    return
                elif event.key() == Qt.Key_V:
                    self.paste_from_clipboard()
                    event.accept()
                    return

            # Left / Right Arrow: Viewer navigation
            if event.key() in (Qt.Key_Left, Qt.Key_Right):
                if self.central_stack.currentWidget() == self.media_viewer and not getattr(self.media_viewer, "is_crop_mode", False):
                    delta = -1 if event.key() == Qt.Key_Left else 1
                    self.navigate_single_viewer(delta)
                    event.accept()
                    return

            # Delete: Grid and Viewer modes
            if event.key() == Qt.Key_Delete:
                if self.central_stack.currentWidget() == self.media_viewer:
                    curr_id = self.media_viewer.current_media_id
                    if curr_id is not None:
                        self._selected_ids = [curr_id]
                        self.delete_selection_to_recycle_bin()
                    event.accept()
                    return
                elif self.central_stack.currentWidget() == self.grid_view:
                    self.delete_selection_to_recycle_bin()
                    event.accept()
                    return
        super().keyPressEvent(event)

    def _save_app_state(self, force: bool = False) -> None:
        """Persists window geometry, filter query, selected folder, docks, and view mode to QSettings."""
        if not force and "PYTEST_CURRENT_TEST" in os.environ:
            return
        try:
            settings = QSettings("PhotoGallery64", "PhotoGallery64")
            settings.setValue("geometry", self.saveGeometry())
            settings.setValue("windowState", self.saveState())
            filter_state = self.left_nav_dock.current_filter_state()
            settings.setValue("filter_state", json.dumps(filter_state))
            settings.setValue("tag_dock_visible", not self.right_tag_dock.isHidden())
            if self.central_stack.currentWidget() == self.media_viewer:
                settings.setValue("nav_dock_visible", getattr(self, "_nav_dock_was_visible", True))
            else:
                settings.setValue("nav_dock_visible", not self.left_nav_dock.isHidden())
            settings.setValue("arrange_by", self._arrange_by)
            settings.setValue("reverse_sort", self._reverse_sort)
            settings.setValue("group_by", self._group_by)
            settings.setValue("sort_by", self._sort_by)
            settings.setValue("zoom_val", self.slider_status_zoom.value())

            current_view = "grid"
            if self.central_stack.currentWidget() == self.media_viewer:
                current_view = "viewer"
            elif self.central_stack.currentWidget() == self.map_view:
                current_view = "map"
            settings.setValue("view_mode", current_view)

            lead_id = (
                self.media_viewer.current_media_id
                if current_view == "viewer"
                else (self._lead_selected_id or (self._selected_ids[0] if self._selected_ids else None))
            )
            if lead_id is not None:
                settings.setValue("last_selected_id", lead_id)

            settings.setValue("nav_show_dates", getattr(self.left_nav_dock, "_show_dates", True))
            settings.setValue("nav_show_tags", getattr(self.left_nav_dock, "_show_tags", True))
            settings.setValue("nav_show_people", getattr(self.left_nav_dock, "_show_people", True))
            settings.setValue("nav_show_ratings", getattr(self.left_nav_dock, "_show_ratings", True))
            settings.setValue("nav_show_places", getattr(self.left_nav_dock, "_show_places", True))
            settings.setValue("nav_show_albums", getattr(self.left_nav_dock, "_show_albums", True))
            settings.setValue("nav_section_order", json.dumps(self.left_nav_dock.get_section_order()))
            settings.setValue("find_people_enabled", getattr(self, "find_people_enabled", False))
            settings.setValue("tooltip_previews", getattr(self, "tooltip_previews", True))
            settings.setValue("jpeg_quality", getattr(self, "jpeg_quality", 83))
            settings.setValue("retention_days", getattr(self, "retention_days", 180))
            settings.setValue("auto_straighten", getattr(self, "auto_straighten", True))
            settings.setValue("auto_noise", getattr(self, "auto_noise", False))
            settings.setValue("auto_color", getattr(self, "auto_color", True))
            settings.setValue("auto_exposure", getattr(self, "auto_exposure", True))
            settings.setValue("write_back_paused", getattr(self, "write_back_paused", True))
            settings.setValue("theme", self.current_theme)
            settings.setValue("icon_theme", getattr(self, "current_icon_theme", "wlpg_aero"))

            # Also persist core user preferences to %LOCALAPPDATA%/PhotoGallery/settings.json
            # ensuring they survive application reinstalls, updates, and registry resets.
            try:
                from src.utils.paths import get_app_data_dir
                settings_json_path = get_app_data_dir() / "settings.json"
                pref_data = {
                    "write_back_paused": bool(getattr(self, "write_back_paused", True)),
                    "theme": getattr(self, "current_theme", "wlpg_aero"),
                    "icon_theme": getattr(self, "current_icon_theme", "wlpg_aero"),
                    "nav_show_dates": getattr(self.left_nav_dock, "_show_dates", True),
                    "nav_show_tags": getattr(self.left_nav_dock, "_show_tags", True),
                    "nav_show_people": getattr(self.left_nav_dock, "_show_people", True),
                    "nav_show_ratings": getattr(self.left_nav_dock, "_show_ratings", True),
                    "nav_show_places": getattr(self.left_nav_dock, "_show_places", True),
                    "find_people_enabled": getattr(self, "find_people_enabled", False),
                    "tooltip_previews": getattr(self, "tooltip_previews", True),
                    "jpeg_quality": getattr(self, "jpeg_quality", 83),
                    "retention_days": getattr(self, "retention_days", 180),
                    "auto_straighten": getattr(self, "auto_straighten", True),
                    "auto_noise": getattr(self, "auto_noise", False),
                    "auto_color": getattr(self, "auto_color", True),
                    "auto_exposure": getattr(self, "auto_exposure", True),
                    "zoom_val": self.slider_status_zoom.value(),
                    "recent_descriptive_tags": list(getattr(self, "_mru_descriptive_tags", [])),
                }
                with open(settings_json_path, "w", encoding="utf-8") as f:
                    json.dump(pref_data, f, indent=2)
            except Exception:
                pass
        except Exception:
            pass

    def _restore_app_state(self, force: bool = False) -> None:
        """Restores window geometry, filter query, selected folder, docks, and view mode from QSettings."""
        if not force and "PYTEST_CURRENT_TEST" in os.environ:
            return
        try:
            settings = QSettings("PhotoGallery64", "PhotoGallery64")
            # Load backup preferences from %LOCALAPPDATA%/PhotoGallery/settings.json if present
            backup_prefs: dict[str, Any] = {}
            try:
                from src.utils.paths import get_app_data_dir
                settings_json_path = get_app_data_dir() / "settings.json"
                if settings_json_path.exists():
                    with open(settings_json_path, "r", encoding="utf-8") as f:
                        backup_prefs = json.load(f)
            except Exception:
                backup_prefs = {}
            geom = settings.value("geometry")
            if geom is not None:
                self.restoreGeometry(geom)
            win_state = settings.value("windowState")
            if win_state is not None:
                self.restoreState(win_state)

            f_json = settings.value("filter_state")
            if f_json and isinstance(f_json, str):
                try:
                    f_dict = json.loads(f_json)
                    self.left_nav_dock.restore_filter_state(f_dict)
                except Exception:
                    pass

            zoom_val = settings.value("zoom_val", type=int)
            if zoom_val and 72 <= zoom_val <= 320:
                self.slider_status_zoom.setValue(zoom_val)

            arr_by = settings.value("arrange_by", type=str)
            rev_sort = settings.value("reverse_sort", type=bool)
            if arr_by:
                self._arrange_by = arr_by
                self._reverse_sort = bool(rev_sort)
                self.ribbon.set_arrange_by(self._arrange_by, self._reverse_sort)

            grp_by = settings.value("group_by", type=str)
            if grp_by:
                self._group_by = grp_by
            srt_by = settings.value("sort_by", type=str)
            if srt_by:
                self._sort_by = srt_by

            tag_visible = settings.value("tag_dock_visible")
            if tag_visible is not None:
                is_vis = str(tag_visible).lower() in ("true", "1")
                self.right_tag_dock.setVisible(is_vis)
                if hasattr(self.ribbon, "btn_toggle_tag_pane"):
                    self.ribbon.btn_toggle_tag_pane.setChecked(is_vis)

            view_mode = settings.value("view_mode", type=str)
            nav_visible = settings.value("nav_dock_visible")
            if nav_visible is not None:
                is_nav_vis = str(nav_visible).lower() in ("true", "1")
            else:
                is_nav_vis = True
            if view_mode != "viewer":
                is_nav_vis = True
            self.left_nav_dock.setVisible(is_nav_vis)
            if is_nav_vis:
                self.left_nav_dock.show()
                if hasattr(self.ribbon, "btn_toggle_nav_pane"):
                    self.ribbon.btn_toggle_nav_pane.setChecked(True)
            self.resizeDocks([self.left_nav_dock], [230], Qt.Horizontal)

            def _to_bool(val, default):
                if val is None:
                    return default
                return str(val).lower() in ("true", "1")

            def _get_val(key, default):
                v = settings.value(key)
                if v is not None:
                    return v
                return backup_prefs.get(key, default)

            self.nav_show_dates = _to_bool(_get_val("nav_show_dates", True), True)
            self.nav_show_tags = _to_bool(_get_val("nav_show_tags", True), True)
            self.nav_show_people = _to_bool(_get_val("nav_show_people", True), True)
            self.nav_show_ratings = _to_bool(_get_val("nav_show_ratings", True), True)
            self.nav_show_places = _to_bool(_get_val("nav_show_places", True), True)
            self.nav_show_albums = _to_bool(_get_val("nav_show_albums", True), True)
            raw_order = _get_val("nav_section_order", None)
            if raw_order:
                try:
                    parsed_order = json.loads(str(raw_order)) if isinstance(raw_order, str) else list(raw_order)
                    if isinstance(parsed_order, list):
                        self.left_nav_dock.set_section_order(parsed_order)
                except Exception:
                    pass
            self.find_people_enabled = _to_bool(_get_val("find_people_enabled", False), False)
            self.tooltip_previews = _to_bool(_get_val("tooltip_previews", True), True)
            self.jpeg_quality = int(_get_val("jpeg_quality", 83))
            self.retention_days = int(_get_val("retention_days", 180))
            self.auto_straighten = _to_bool(_get_val("auto_straighten", True), True)
            self.auto_noise = _to_bool(_get_val("auto_noise", False), False)
            self.auto_color = _to_bool(_get_val("auto_color", True), True)
            self.auto_exposure = _to_bool(_get_val("auto_exposure", True), True)
            self.write_back_paused = _to_bool(_get_val("write_back_paused", True), True)
            self.xmp_daemon.set_paused(self.write_back_paused)
            self.ribbon.set_write_back_paused(self.write_back_paused)

            self.left_nav_dock.set_navigation_sections_visibility(
                show_dates=self.nav_show_dates,
                show_tags=self.nav_show_tags,
                show_people=self.nav_show_people,
                show_ratings=self.nav_show_ratings,
                show_places=self.nav_show_places,
                show_albums=self.nav_show_albums,
            )

            saved_theme = _get_val("theme", "wlpg_aero")
            if saved_theme and saved_theme in ("wlpg_aero", "modern", "dark"):
                self.apply_theme(saved_theme)

            saved_icon_theme = _get_val("icon_theme", "wlpg_aero")
            if saved_icon_theme and saved_icon_theme in ("wlpg_aero", "fluent", "minimal"):
                self.current_icon_theme = saved_icon_theme
                icon_provider.set_icon_theme(saved_icon_theme)

            # Clean up any phantom/duplicate person tags on startup
            try:
                self.catalog_repo.deduplicate_media_people()
            except Exception:
                pass

            # Re-run query with restored filter
            records = self.rebuild_and_run_query()
            if len(records) == 0 and self.catalog_repo.get_total_media_count() > 0:
                # Restored filter produced 0 results (e.g. stale/conflicting sub-filter from prior session).
                # Reset to All Media so the user never starts with an empty/blank gallery.
                self.left_nav_dock.clear_all_filters()
                records = self.rebuild_and_run_query()

            last_id = settings.value("last_selected_id", type=int)
            view_mode = settings.value("view_mode", type=str)
            if last_id and any(r.id == last_id for r in records):
                if view_mode == "viewer":
                    self.open_single_viewer(last_id)
                else:
                    self._selected_ids = [last_id]
                    self._lead_selected_id = last_id
                    summary = self.catalog_repo.get_selection_summary([last_id])
                    self.right_tag_dock.bind_selection([last_id], summary)
                    self.grid_view.select_media_id(last_id)
        except Exception:
            pass

    def shutdown(self) -> None:
        """Cleanly stops all background workers, flushes XMP queue, prunes backups, and closes SQLite."""
        if self._is_shutting_down:
            return
        self._is_shutting_down = True

        self._scanner_cancel.set()
        for flag in self._active_thumb_flags.values():
            flag.set()
        self._active_thumb_flags.clear()

        self.scanner_pool.waitForDone(3000)
        self.thumbnail_pool.waitForDone(3000)
        self.stop_face_detection()
        self.face_pool.waitForDone(3000)
        self.priority_face_pool.waitForDone(1500)
        self.xmp_daemon.stop()
        if hasattr(self, "_monitor_timer") and self._monitor_timer.isActive():
            self._monitor_timer.stop()
        if hasattr(self, "folder_watcher"):
            self.folder_watcher.stop()

        # Prune expired '.OriginalFiles' backups older than self.retention_days
        try:
            expired_paths = self.catalog_repo.prune_expired_backups(
                retention_days=self.retention_days
            )
            for p in expired_paths:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass
        except Exception:
            pass

        app = QApplication.instance()
        if app:
            try:
                app.removeEventFilter(self)
            except Exception:
                pass

        self.engine.close_all()

    def eventFilter(self, watched: Any, event: Any) -> bool:
        if event.type() == QEvent.Type.KeyPress:
            from PySide6.QtWidgets import QAbstractItemView, QAbstractSpinBox, QComboBox
            if (
                QApplication.activeModalWidget() is None
                and QApplication.activePopupWidget() is None
            ):
                focus_w = QApplication.focusWidget() or self.focusWidget()
                target_watched = focus_w if focus_w is not None else self
                is_input_w = isinstance(
                    focus_w, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox)
                )
                if watched is target_watched and not is_input_w:
                    if (
                        self.central_stack.currentWidget() == self.media_viewer
                        and not getattr(self.media_viewer, "is_crop_mode", False)
                    ):
                        if event.key() in (Qt.Key_Left, Qt.Key_Right) and event.modifiers() in (
                            Qt.NoModifier,
                            Qt.KeypadModifier,
                        ):
                            self.navigate_single_viewer(-1 if event.key() == Qt.Key_Left else 1)
                            event.accept()
                            return True
                    elif self.central_stack.currentWidget() == self.grid_view:
                        # Allow left_nav_dock trees to keep their own arrow navigation when focused;
                        # otherwise (ribbon buttons, status bar buttons, grid, or main window) navigate grid!
                        in_left_dock = (
                            focus_w is not None
                            and hasattr(self, "left_nav_dock")
                            and self.left_nav_dock is not None
                            and self.left_nav_dock.isAncestorOf(focus_w)
                            and isinstance(focus_w, QAbstractItemView)
                        )
                        if (
                            not in_left_dock
                            and focus_w is not self.grid_view
                            and not (event.modifiers() & (Qt.ControlModifier | Qt.AltModifier))
                            and event.key() in (
                                Qt.Key_Left,
                                Qt.Key_Right,
                                Qt.Key_Up,
                                Qt.Key_Down,
                                Qt.Key_Home,
                                Qt.Key_End,
                                Qt.Key_PageUp,
                                Qt.Key_PageDown,
                            )
                        ):
                            if self.grid_view._navigate_by_key(event.key(), event.modifiers()):
                                self.grid_view.setFocus()
                                event.accept()
                                return True
        if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
            from PySide6.QtGui import QMouseEvent
            if isinstance(event, QMouseEvent):
                if event.button() == Qt.MouseButton.BackButton:
                    if event.type() == QEvent.Type.MouseButtonPress:
                        self.navigate_history_back()
                    return True
                elif event.button() == Qt.MouseButton.ForwardButton:
                    if event.type() == QEvent.Type.MouseButtonPress:
                        self.navigate_history_forward()
                    return True
        return super().eventFilter(watched, event)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        apply_win32_frameless_styles(int(self.winId()))

    def nativeEvent(self, eventType, message):
        res = handle_win32_native_event(self, eventType, message)
        if res is not None:
            return res
        return super().nativeEvent(eventType, message)
        if not getattr(self, "_initial_show_done", False):
            self._initial_show_done = True
            if self.central_stack.currentWidget() == self.media_viewer:
                self.media_viewer.graphics_view.setFocus()
                QTimer.singleShot(0, self.media_viewer.graphics_view.setFocus)
            else:
                self.left_nav_dock.setVisible(True)
                self.left_nav_dock.show()
                self.resizeDocks([self.left_nav_dock], [230], Qt.Horizontal)
                if hasattr(self.ribbon, "btn_toggle_nav_pane"):
                    self.ribbon.btn_toggle_nav_pane.setChecked(True)
                if self.central_stack.currentWidget() == self.grid_view:
                    self.grid_view.setFocus()
                    QTimer.singleShot(0, self.grid_view.setFocus)
            if (
                getattr(self, "find_people_enabled", True)
                and "PYTEST_CURRENT_TEST" not in os.environ
                and self.catalog_model.rowCount() > 0
            ):
                QTimer.singleShot(400, self.start_face_detection)

    # -------------------------------------------------------------------------
    # Static Albums & Smart Albums
    # -------------------------------------------------------------------------
    def _refresh_albums_ui(self) -> None:
        if hasattr(self, "left_nav_dock"):
            self.left_nav_dock.refresh_albums_only(self.catalog_repo)
        if hasattr(self, "albums_view"):
            self.albums_view.refresh()

    def add_media_to_album(self, album_id: int, media_ids: Sequence[int]) -> int:
        """Add ``media_ids`` into Static Album ``album_id`` and refresh UI."""
        mids = [int(m) for m in media_ids]
        if not mids:
            return 0
        added = self.catalog_repo.add_media_to_album(int(album_id), mids)
        self._refresh_albums_ui()
        if hasattr(self, "show_status_message"):
            self.show_status_message(f"Added {len(mids)} item(s) to album.")
        return added

    def create_album_from_media(self, media_ids: Sequence[int]) -> int | None:
        """Prompt for an Album name, create it, and add ``media_ids`` to it."""
        mids = [int(m) for m in media_ids]
        name, ok = QInputDialog.getText(self, "New Album", "Album name:")
        if not ok or not name.strip():
            return None
        try:
            aid = self.catalog_repo.create_album(name.strip())
        except Exception as exc:
            QMessageBox.warning(self, "New Album", f"Could not create album: {exc}")
            return None
        if mids:
            self.catalog_repo.add_media_to_album(aid, mids)
        self._refresh_albums_ui()
        if hasattr(self, "show_status_message"):
            self.show_status_message(f"Created album '{name.strip()}' with {len(mids)} item(s).")
        return aid

    def remove_media_from_album(self, album_id: int, media_ids: Sequence[int]) -> int:
        """Remove ``media_ids`` from Static Album ``album_id`` (non-destructive)."""
        mids = [int(m) for m in media_ids]
        if not mids:
            return 0
        removed = self.catalog_repo.remove_media_from_album(int(album_id), mids)
        self._refresh_albums_ui()
        if self.left_nav_dock.active_album_id() == int(album_id):
            self.rebuild_and_run_query()
        if hasattr(self, "show_status_message"):
            self.show_status_message(f"Removed {removed} item(s) from album.")
        return removed

    def _on_new_album_requested(self) -> None:
        selected = self.grid_view.selected_media_ids() if hasattr(self, "grid_view") else []
        self.create_album_from_media(selected)

    def _on_rename_album_requested(self, album_id: int, current_name: str) -> None:
        new_name, ok = QInputDialog.getText(
            self, "Rename Album", "Album name:", text=current_name
        )
        if not ok or not new_name.strip() or new_name.strip() == current_name:
            return
        try:
            self.catalog_repo.rename_album(int(album_id), new_name.strip())
        except Exception as exc:
            QMessageBox.warning(self, "Rename Album", f"Could not rename album: {exc}")
            return
        if self.left_nav_dock.active_album_id() == int(album_id):
            self.left_nav_dock._album_name = new_name.strip()
            self._update_scope_bar_visibility()
        self._refresh_albums_ui()

    def _on_delete_album_requested(self, album_id: int, current_name: str) -> None:
        resp = QMessageBox.question(
            self,
            "Delete Album",
            f"Delete album '{current_name}'?\n\nPhotos inside will remain in your gallery.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        was_active = self.left_nav_dock.active_album_id() == int(album_id)
        self.catalog_repo.delete_album(int(album_id))
        if was_active:
            self.left_nav_dock.clear_all_filters()
        self._refresh_albums_ui()

    def _on_new_smart_album_requested(
        self, initial_name: str = "", initial_rules: dict[str, Any] | None = None
    ) -> None:
        dlg = SmartAlbumDialog(
            self.catalog_repo,
            initial_name=initial_name,
            initial_rules=initial_rules,
            title="New Smart Album",
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            sid = self.catalog_repo.create_smart_album(dlg.get_name(), dlg.get_rules_dict())
        except Exception as exc:
            QMessageBox.warning(self, "New Smart Album", f"Could not create Smart Album: {exc}")
            return
        self._refresh_albums_ui()
        self.left_nav_dock._select_smart_album_scope(sid, dlg.get_name())

    def _on_edit_smart_album_requested(self, smart_album_id: int) -> None:
        row = self.catalog_repo.get_smart_album(int(smart_album_id))
        if not row:
            return
        try:
            parsed_rules = json.loads(row.get("rules_json") or "{}")
        except Exception:
            parsed_rules = {"match_mode": "all", "rules": []}
        dlg = SmartAlbumDialog(
            self.catalog_repo,
            initial_name=str(row.get("name") or ""),
            initial_rules=parsed_rules,
            title="Edit Smart Album Rules",
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.catalog_repo.update_smart_album(
                int(smart_album_id), name=dlg.get_name(), rules_dict=dlg.get_rules_dict()
            )
        except Exception as exc:
            QMessageBox.warning(self, "Edit Smart Album", f"Could not update Smart Album: {exc}")
            return
        if self.left_nav_dock.current_filter_state().get("smart_album_id") == int(smart_album_id):
            self.left_nav_dock._smart_album_name = dlg.get_name()
            self._update_scope_bar_visibility()
            self.rebuild_and_run_query()
        self._refresh_albums_ui()

    def _on_rename_smart_album_requested(self, smart_album_id: int, current_name: str) -> None:
        new_name, ok = QInputDialog.getText(
            self, "Rename Smart Album", "Smart Album name:", text=current_name
        )
        if not ok or not new_name.strip() or new_name.strip() == current_name:
            return
        try:
            self.catalog_repo.update_smart_album(int(smart_album_id), name=new_name.strip())
        except Exception as exc:
            QMessageBox.warning(self, "Rename Smart Album", f"Could not rename Smart Album: {exc}")
            return
        if self.left_nav_dock.current_filter_state().get("smart_album_id") == int(smart_album_id):
            self.left_nav_dock._smart_album_name = new_name.strip()
            self._update_scope_bar_visibility()
        self._refresh_albums_ui()

    def _on_delete_smart_album_requested(self, smart_album_id: int, current_name: str) -> None:
        resp = QMessageBox.question(
            self,
            "Delete Smart Album",
            f"Delete Smart Album '{current_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        was_active = (
            self.left_nav_dock.current_filter_state().get("smart_album_id") == int(smart_album_id)
        )
        self.catalog_repo.delete_smart_album(int(smart_album_id))
        if was_active:
            self.left_nav_dock.clear_all_filters()
        self._refresh_albums_ui()

    def _build_rules_dict_from_current_filter(self) -> dict[str, Any]:
        """Convert active LeftNavDock + search bar facets into a Smart Album rule tree."""
        state = self.left_nav_dock.current_filter_state()
        rules: list[dict[str, Any]] = []

        if state.get("folder_id") is not None:
            rules.append({"field": "folder", "op": "in_folder", "value": int(state["folder_id"])})
        if state.get("date_year") is not None:
            yr = int(state["date_year"])
            mo = state.get("date_month")
            dy = state.get("date_day")
            if mo is not None and dy is not None:
                rules.append(
                    {"field": "date_taken", "op": "eq", "value": f"{yr:04d}-{int(mo):02d}-{int(dy):02d}"}
                )
            else:
                rules.append({"field": "year", "op": "eq", "value": str(yr)})
                if mo is not None:
                    rules.append({"field": "month", "op": "eq", "value": str(int(mo))})
        elif state.get("date_month") is not None:
            rules.append({"field": "month", "op": "eq", "value": str(int(state["date_month"]))})

        people_lookup = {
            int(p["id"]): str(p.get("name") or "")
            for p in self.catalog_repo.get_people_list()
        }
        for pid in state.get("person_ids") or ():
            pname = people_lookup.get(int(pid), "").strip()
            rules.append({"field": "people", "op": "includes", "value": pname or int(pid)})

        tag_lookup = {
            int(t["id"]): str(t.get("full_path") or t.get("name") or "")
            for t in self.catalog_repo.get_tag_tree()
        }
        for tid in state.get("include_tag_ids") or ():
            tname = tag_lookup.get(int(tid))
            if tname:
                rules.append({"field": "tag", "op": "includes", "value": tname})
        for tid in state.get("exclude_tag_ids") or ():
            tname = tag_lookup.get(int(tid))
            if tname:
                rules.append({"field": "tag", "op": "excludes", "value": tname})
        if state.get("untagged_only"):
            rules.append({"field": "tag", "op": "is_empty", "value": ""})

        ratings = sorted(state.get("ratings") or ())
        if len(ratings) == 1:
            rules.append({"field": "rating", "op": "eq", "value": int(ratings[0])})
        elif ratings:
            rules.append({"field": "rating", "op": "gte", "value": int(min(ratings))})

        if state.get("flagged_only"):
            rules.append({"field": "flag", "op": "is_flagged", "value": ""})

        m_type = str(state.get("media_type") or "").lower()
        if m_type == "image":
            rules.append({"field": "media_type", "op": "is", "value": "photo"})
        elif m_type in ("video", "raw"):
            rules.append({"field": "media_type", "op": "is", "value": m_type})

        o_filt = str(state.get("orientation_filter") or "").lower()
        if o_filt in ("landscape", "portrait", "square"):
            rules.append({"field": "orientation", "op": "is", "value": o_filt})

        if getattr(self, "_search_text", "").strip():
            rules.append(
                {"field": "caption_filename", "op": "contains", "value": self._search_text.strip()}
            )
        if getattr(self, "_active_map_location_label", None):
            rules.append(
                {
                    "field": "place",
                    "op": "contains",
                    "value": str(self._active_map_location_label),
                }
            )
        return {"match_mode": "all", "rules": rules}

    def _on_save_filter_as_smart_album_requested(self) -> None:
        rules_dict = self._build_rules_dict_from_current_filter()
        self._on_new_smart_album_requested(initial_name="", initial_rules=rules_dict)

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            if hasattr(self, "ribbon") and hasattr(self.ribbon, "update_window_controls"):
                self.ribbon.update_window_controls()

    def setWindowTitle(self, title: str) -> None:
        super().setWindowTitle(title)
        if hasattr(self, "ribbon") and hasattr(self.ribbon, "lbl_app_title"):
            self.ribbon.lbl_app_title.setText(title)

    def closeEvent(self, event) -> None:
        self._save_app_state()
        self.shutdown()
        super().closeEvent(event)
