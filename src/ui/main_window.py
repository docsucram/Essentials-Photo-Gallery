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
    Q_ARG,
    QBuffer,
    QByteArray,
    QEvent,
    QIODevice,
    QLoggingCategory,
    QMetaObject,
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
    Slot,
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
from src.ui.bottom_filmstrip_dock import BottomFilmstripDock
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
from src.core.geotag_resolver import resolve_location_query, save_place_to_db


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


class _StackMutationUndoCommand(QUndoCommand):
    """Reversible stack grouping command (auto-stack bursts / RAW+JPEG pairs) for QUndoStack."""

    def __init__(
        self,
        main_win: PhotoGalleryMainWindow,
        before_map: dict[int, int | None],
        after_map: dict[int, int | None],
        description: str,
    ) -> None:
        super().__init__(description)
        self.main_win = main_win
        self.before_map = dict(before_map)
        self.after_map = dict(after_map)
        self._first_redo = True

    def redo(self) -> None:
        if self._first_redo:
            self._first_redo = False
            return
        self.main_win.catalog_repo.restore_stack_parent_map(self.after_map)
        self.main_win._expanded_stack_ids.clear()
        self.main_win.rebuild_and_run_query()

    def undo(self) -> None:
        self.main_win.catalog_repo.restore_stack_parent_map(self.before_map)
        self.main_win._expanded_stack_ids.clear()
        self.main_win.rebuild_and_run_query()


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


class _FavouriteUndoCommand(QUndoCommand):
    """Reversible favourite (is_favourite) modification command for QUndoStack."""

    def __init__(
        self,
        main_win: PhotoGalleryMainWindow,
        target_ids: Sequence[int],
        new_fav: int,
    ) -> None:
        super().__init__("Favourite" if new_fav else "Unfavourite")
        self.main_win = main_win
        self.target_ids = list(target_ids)
        self.new_fav = 1 if new_fav else 0
        field_map = self.main_win.catalog_repo.get_field_map_for_ids(self.target_ids, "is_favourite")
        self.old_favs = {mid: int(field_map.get(mid) or 0) for mid in self.target_ids}

    def _apply_lookup_and_refresh(self, fav_map: dict[int, int]) -> None:
        if hasattr(self.main_win, "catalog_model"):
            self.main_win.catalog_model.update_items_metadata(list(fav_map.keys()), is_favourite=self.new_fav)
        if hasattr(self.main_win, "grid_view") and self.main_win.grid_view.viewport():
            self.main_win.grid_view.viewport().update()
        if hasattr(self.main_win, "media_viewer") and getattr(self.main_win.media_viewer, "_update_cull_hud", None):
            self.main_win.media_viewer._update_cull_hud()
        if hasattr(self.main_win, "left_nav_dock") and hasattr(self.main_win, "catalog_repo"):
            try:
                self.main_win.left_nav_dock.refresh_albums_only(self.main_win.catalog_repo)
            except Exception:
                pass

    def redo(self) -> None:
        self.main_win.catalog_repo.batch_update_favourite(self.target_ids, bool(self.new_fav))
        self._apply_lookup_and_refresh({mid: self.new_fav for mid in self.target_ids})

    def undo(self) -> None:
        self.main_win.catalog_repo.restore_field_map(self.old_favs, "is_favourite")
        for old_v, mids in ((0, [m for m, v in self.old_favs.items() if not v]), (1, [m for m, v in self.old_favs.items() if v])):
            if mids:
                self.main_win.catalog_model.update_items_metadata(mids, is_favourite=old_v)
        if hasattr(self.main_win, "grid_view") and self.main_win.grid_view.viewport():
            self.main_win.grid_view.viewport().update()
        if hasattr(self.main_win, "media_viewer") and getattr(self.main_win.media_viewer, "_update_cull_hud", None):
            self.main_win.media_viewer._update_cull_hud()
        if hasattr(self.main_win, "left_nav_dock") and hasattr(self.main_win, "catalog_repo"):
            try:
                self.main_win.left_nav_dock.refresh_albums_only(self.main_win.catalog_repo)
            except Exception:
                pass


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
        if hasattr(self.main_win, "refresh_navigation_and_autocomplete"):
            self.main_win.refresh_navigation_and_autocomplete({"places", "albums"})


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
        if hasattr(self.main_win, "refresh_navigation_and_autocomplete"):
            self.main_win.refresh_navigation_and_autocomplete({"places", "albums"})


class _BatchGpsMatchUndoCommand(QUndoCommand):
    """Single-step undoable command for Companion Phone GPS Matching."""

    def __init__(
        self,
        main_win: Any,
        matches: list[Any],
        *,
        clock_shift_sec: int = 0,
    ) -> None:
        super().__init__(f"Match Locations ({len(matches)} photos)")
        self.main_win = main_win
        self.matches = list(matches)
        self.clock_shift_sec = int(clock_shift_sec)
        self.target_ids = [int(m.media_id) for m in self.matches]
        self.old_geotags = self.main_win.catalog_repo.get_geotag_map(self.target_ids)

    def _refresh_views(self) -> None:
        active_ids = (
            self.main_win.get_active_target_ids()
            if hasattr(self.main_win, "get_active_target_ids")
            else getattr(self.main_win, "_selected_ids", [])
        )
        if any(mid in active_ids for mid in self.target_ids):
            summary = self.main_win.catalog_repo.get_selection_summary(active_ids)
            self.main_win.right_tag_dock.bind_selection(active_ids, summary)
        if (
            self.main_win._map_view is not None
            and self.main_win.central_stack.currentWidget() == self.main_win._map_view
        ):
            self.main_win._map_view.set_viewport_bbox(
                self.main_win._map_view._min_lat,
                self.main_win._map_view._max_lat,
                self.main_win._map_view._min_lon,
                self.main_win._map_view._max_lon,
                self.main_win.catalog_repo,
            )
        if hasattr(self.main_win, "refresh_navigation_and_autocomplete"):
            self.main_win.refresh_navigation_and_autocomplete({"places", "albums"})

    def redo(self) -> None:
        for m in self.matches:
            self.main_win.catalog_repo.set_geotag(
                [int(m.media_id)],
                float(m.lat),
                float(m.lon),
                m.location_name or None,
            )
        if self.clock_shift_sec != 0 and hasattr(
            self.main_win.catalog_repo, "apply_relative_time_shift"
        ):
            try:
                self.main_win.catalog_repo.apply_relative_time_shift(
                    self.target_ids, self.clock_shift_sec
                )
            except Exception:
                pass
        self._refresh_views()

    def undo(self) -> None:
        self.main_win.catalog_repo.restore_geotag_map(self.old_geotags)
        if self.clock_shift_sec != 0 and hasattr(
            self.main_win.catalog_repo, "apply_relative_time_shift"
        ):
            try:
                self.main_win.catalog_repo.apply_relative_time_shift(
                    self.target_ids, -self.clock_shift_sec
                )
            except Exception:
                pass
        self._refresh_views()



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


class _MotionPhotoBackfillSignals(QObject):
    finished = Signal(int)


class _MotionPhotoBackfillWorker(QRunnable):
    """Lightweight background backfill detecting Motion/Live photos for older catalog rows."""

    def __init__(
        self,
        engine: DualDatabaseEngine,
        signals: _MotionPhotoBackfillSignals,
        *,
        folder_ids: Sequence[int] | None = None,
        batch_limit: int = 1500,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.signals = signals
        self.folder_ids = tuple(int(f) for f in folder_ids) if folder_ids else None
        self.batch_limit = max(50, int(batch_limit))

    def run(self) -> None:
        found = 0
        try:
            from src.imaging.live_photo import detect_motion_photo_offset, find_paired_live_video

            repo = CatalogRepository(self.engine)
            conn = self.engine.catalog_conn()
            if self.folder_ids:
                placeholders = ",".join("?" for _ in self.folder_ids)
                rows = conn.execute(
                    f"""
                    SELECT m.id, f.path AS folder_path, m.filename
                    FROM media m
                    JOIN folders f ON f.id = m.folder_id
                    WHERE m.is_video = 0
                      AND COALESCE(m.is_missing, 0) = 0
                      AND COALESCE(m.is_live_photo, 0) = 0
                      AND COALESCE(m.motion_offset, 0) = 0
                      AND LOWER(COALESCE(m.ext, '')) IN ('jpg', 'jpeg', 'heic', 'heif')
                      AND m.folder_id IN ({placeholders})
                    ORDER BY m.date_taken DESC, m.id DESC
                    LIMIT ?;
                    """,
                    (*self.folder_ids, self.batch_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT m.id, f.path AS folder_path, m.filename
                    FROM media m
                    JOIN folders f ON f.id = m.folder_id
                    WHERE m.is_video = 0
                      AND COALESCE(m.is_missing, 0) = 0
                      AND COALESCE(m.is_live_photo, 0) = 0
                      AND COALESCE(m.motion_offset, 0) = 0
                      AND LOWER(COALESCE(m.ext, '')) IN ('jpg', 'jpeg', 'heic', 'heif')
                    ORDER BY m.date_taken DESC, m.id DESC
                    LIMIT ?;
                    """,
                    (self.batch_limit,),
                ).fetchall()

            live_updates: list[tuple[int, int, str, int]] = []
            mov_pairs: list[tuple[int, int]] = []
            checked_still_ids: list[int] = []
            for r in rows:
                mid = int(r["id"])
                fpath = Path(str(r["folder_path"])) / str(r["filename"])
                paired = find_paired_live_video(fpath)
                if paired is not None:
                    live_updates.append((1, 0, str(paired), mid))
                    mov_mid = repo.get_media_id_by_file_path(paired)
                    if mov_mid is not None:
                        mov_pairs.append((int(mov_mid), mid))
                    continue
                off = detect_motion_photo_offset(fpath)
                if off is not None:
                    live_updates.append((1, int(off), "", mid))
                else:
                    checked_still_ids.append(mid)

            if live_updates:
                repo.update_live_photo_metadata(live_updates)
                found = len(live_updates)
            if mov_pairs:
                repo.mark_live_companion_videos(mov_pairs)
            if checked_still_ids:
                with self.engine.write_lock:
                    wconn = self.engine.catalog_conn()
                    for i in range(0, len(checked_still_ids), 900):
                        chunk = checked_still_ids[i : i + 900]
                        ph = ",".join("?" for _ in chunk)
                        wconn.execute(
                            f"UPDATE media SET motion_offset = -1 WHERE id IN ({ph}) AND COALESCE(is_live_photo, 0) = 0;",
                            chunk,
                        )
                    wconn.commit()
        except Exception:
            found = 0
        finally:
            if threading.current_thread() is not threading.main_thread():
                self.engine.close_thread_connections()
            try:
                self.signals.finished.emit(found)
            except RuntimeError:
                pass


class _AvatarBackfillWorker(QRunnable):
    """Background worker that generates missing avatar crops in people table."""

    def __init__(self, engine: DualDatabaseEngine, signals: _AvatarBackfillSignals) -> None:
        super().__init__()
        self.engine = engine
        self.signals = signals

    def run(self) -> None:
        updated = 0
        try:
            repo = CatalogRepository(self.engine)
            updated = repo.backfill_missing_people_thumbnails()
        except Exception:
            updated = 0
        finally:
            if threading.current_thread() is not threading.main_thread():
                self.engine.close_thread_connections()
            try:
                self.signals.finished.emit(updated)
            except RuntimeError:
                pass


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


class _SubjectsCountSignals(QObject):
    counts_ready = Signal(object)
    finished = Signal()


class _SubjectsCountWorker(QRunnable):
    """Computes semantic Subject preset counts off the GUI thread."""

    def __init__(
        self,
        engine: DualDatabaseEngine,
        semantic_repo: Any,
        semantic_engine: Any,
        subj_vecs: Any,
        *,
        include_private: bool = False,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.semantic_repo = semantic_repo
        self.semantic_engine = semantic_engine
        self.subj_vecs = subj_vecs
        self.include_private = include_private
        self.signals = _SubjectsCountSignals()

    @Slot()
    def run(self) -> None:
        counts: dict[str, int] | None = None
        try:
            scope_ids: set[int] | None = None
            if not self.include_private:
                has_priv = self.engine.catalog_conn().execute(
                    "SELECT 1 FROM folders WHERE is_private = 1 LIMIT 1;"
                ).fetchone()
                if has_priv is not None:
                    rows = self.engine.catalog_conn().execute(
                        """
                        SELECT m.id
                        FROM media m
                        JOIN folders f ON f.id = m.folder_id
                        WHERE m.is_missing = 0 AND f.is_private = 0;
                        """
                    ).fetchall()
                    scope_ids = {int(r[0]) for r in rows}
            counts = self.semantic_engine.compute_subject_counts(
                self.semantic_repo, self.subj_vecs, scope_ids=scope_ids
            )
        except Exception:
            counts = None
        finally:
            if threading.current_thread() is not threading.main_thread():
                self.engine.close_thread_connections()
            try:
                if counts is not None:
                    self.signals.counts_ready.emit(counts)
                self.signals.finished.emit()
            except RuntimeError:
                pass


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
        self._active_similar_order: list[int] | None = None
        self._active_similar_scored: list[tuple[int, float]] | None = None
        self._active_similar_label: str | None = None
        self._active_similar_seed_id: int | None = None
        self._active_similar_seed_name: str = ""
        self._active_similar_pct: int = 81
        self._preserving_similar_on_nav_change: bool = False
        self._active_map_location_label: str | None = None
        self._active_special_people_filter: str | None = None
        self._active_special_people_label: str | None = None
        self._active_date_range: tuple[int, int] | None = None
        self._active_date_label: str | None = None
        self._setting_date_preset: bool = False
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
        self._ocr_status_text: str = "OCR: Idle"
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
        self.thumbnail_pool.setMaxThreadCount(min(4, max(2, (os.cpu_count() or 4) // 2)))
        self.thumb_cache_pool = QThreadPool(self)
        self.thumb_cache_pool.setMaxThreadCount(1)
        from src.workers.thumb_scheduler import ThumbnailWriteCoalescer
        self.thumb_write_coalescer = ThumbnailWriteCoalescer(
            self.engine, batch_size=16, flush_ms=150, parent=self
        )
        self._active_thumb_flags: dict[int, CancellationFlag] = {}
        self._inflight_workers: set[QRunnable] = set()
        self._media_meta_lookup: dict[int, tuple[str, int, bool, int]] = {}
        self._failed_thumb_ids: set[int] = set()
        self._pending_visible_media_ids: list[int] = []
        self._pending_disk_decode_ids: list[int] = []
        self._thumb_cache_token: int = 0
        self._face_crop_token: int = 0

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
        self._pending_watcher_delta: dict[str, Any] = {
            "added": 0,
            "modified": 0,
            "deleted": 0,
            "folders_changed": 0,
            "folders": [],
        }
        self._watcher_delta_timer = QTimer(self)
        self._watcher_delta_timer.setSingleShot(True)
        self._watcher_delta_timer.setInterval(750)
        self._watcher_delta_timer.timeout.connect(self._flush_folder_watcher_delta)
        self.folder_watcher = FolderWatcherDaemon(engine, debounce_interval_ms=750, parent=self)
        self.folder_watcher.delta_completed.connect(self._on_folder_watcher_delta)
        self.folder_watcher.reconciliation_completed.connect(self._on_startup_reconciliation_completed)
        self.folder_watcher.missing_roots_detected.connect(self._on_missing_root_folders_detected)
        self._missing_roots_prompted: set[int] = set()
        self.folder_watcher.update_watched_paths()
        try:
            QApplication.clipboard().dataChanged.connect(self._on_clipboard_data_changed)
        except Exception:
            pass

        # ---------------------------------------------------------------------
        # Build UI Hierarchy
        # ---------------------------------------------------------------------
        # Pre-initialize Qt Quick RHI backing surface before window is shown so
        # creating QWebEngineView later in WorldMapView never recreates the native HWND
        if "PYTEST_CURRENT_TEST" not in os.environ and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            try:
                from PySide6.QtQuickWidgets import QQuickWidget
                self._rhi_surface_anchor = QQuickWidget(self)
                self._rhi_surface_anchor.setClearColor(Qt.transparent)
                self._rhi_surface_anchor.setAttribute(Qt.WA_TransparentForMouseEvents, True)
                self._rhi_surface_anchor.setFixedSize(1, 1)
                self._rhi_surface_anchor.move(-1000, -1000)
                self._rhi_surface_anchor.lower()
                QTimer.singleShot(250, self._rhi_surface_anchor.hide)
            except Exception:
                pass

        # 1. Top Ribbon Toolbar spanning the full top width of the window
        self.ribbon = RibbonToolbar(self)
        self.setMenuWidget(self.ribbon)
        self.undo_stack.canUndoChanged.connect(self.ribbon.set_undo_enabled)
        self.undo_stack.canRedoChanged.connect(self.ribbon.set_redo_enabled)

        # 2. Central QStackedWidget (0: Grid, 1: Single Viewer, 2: World Map)
        class _CentralStackedWidget(QStackedWidget):
            def minimumSizeHint(self) -> QSize:
                return QSize(200, 150)

        self.central_stack = _CentralStackedWidget(self)
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

        from src.ui.welcome_overlay import WelcomeEmptyOverlay

        self.welcome_overlay = WelcomeEmptyOverlay(self.central_stack)
        self.welcome_overlay.add_folder_requested.connect(lambda: self.dispatch_ribbon_action("add_folder"))
        self.welcome_overlay.open_settings_requested.connect(lambda: self.dispatch_ribbon_action("open_settings"))
        self.central_stack.addWidget(self.welcome_overlay)

        self.media_viewer = SingleMediaViewer(self.central_stack)
        self.media_viewer.main_window = self
        self.media_viewer.catalog_repo = self.catalog_repo
        self.media_viewer.repo = self.catalog_repo
        self.central_stack.addWidget(self.media_viewer)

        self._map_view: WorldMapView | None = None

        self.batch_people_view = BatchPeopleView(self.engine, self.central_stack, repo=self.catalog_repo)
        self.batch_people_view.faces_confirmed.connect(self._on_batch_faces_confirmed)
        self.batch_people_view.faces_rejected.connect(self._on_batch_faces_rejected)
        self.batch_people_view.faces_ignored.connect(self._on_batch_faces_ignored)
        self.batch_people_view.person_renamed.connect(self._on_batch_person_renamed)
        self.batch_people_view.view_person_photos_requested.connect(self._on_batch_view_person_photos)
        self.batch_people_view.person_filter_changed.connect(self._on_batch_person_filter_changed)
        self.batch_people_view.back_to_gallery_requested.connect(self.close_batch_people_view)
        self.batch_people_view.open_media_requested.connect(self.open_single_viewer)
        self.batch_people_view.card_selection_changed.connect(self._on_batch_card_selection_changed)
        QShortcut(
            QKeySequence(Qt.Key_Escape),
            self.batch_people_view,
            self.close_batch_people_view,
            context=Qt.WidgetWithChildrenShortcut,
        )
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
        self.albums_view.set_album_cover_requested.connect(self._on_set_album_cover_requested)
        self.albums_view.rename_album_requested.connect(self._on_rename_album_requested)
        self.albums_view.delete_album_requested.connect(self._on_delete_album_requested)
        self.albums_view.new_smart_album_requested.connect(self._on_new_smart_album_requested)
        self.albums_view.edit_smart_album_requested.connect(self._on_edit_smart_album_requested)
        self.albums_view.set_smart_album_cover_requested.connect(self._on_set_smart_album_cover_requested)
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
        self.duplicate_finder_view.scan_state_changed.connect(self._on_duplicates_scan_state_changed)
        self.duplicate_finder_view.summary_state_changed.connect(self._on_duplicates_summary_state_changed)
        self.duplicate_finder_view.find_similar_requested.connect(self.open_find_similar_for_media)
        self.duplicate_finder_view.compare_group_requested.connect(self._on_duplicates_compare_group_requested)
        self.duplicate_finder_view.zoom_changed.connect(self._on_duplicates_zoom_changed)
        self.duplicate_finder_view.media_selected.connect(self._on_duplicates_media_selected)
        self.central_stack.addWidget(self.duplicate_finder_view)

        # Integrated Multi-Photo Compare / Light Table View
        self.compare_view = CompareMediaView(self.central_stack)
        self.compare_view.bind_repositories(self.catalog_repo, self.thumb_repo)
        self.compare_view.close_requested.connect(self._on_compare_closed)
        self.compare_view.rating_changed.connect(lambda _mid, _r: self.rebuild_and_run_query())
        self.compare_view.flag_toggled.connect(lambda _mid, _f: self.rebuild_and_run_query())
        self.compare_view.media_deleted.connect(lambda _mid: self.rebuild_and_run_query())
        self.compare_view.open_in_editor_requested.connect(self._on_compare_open_editor)
        self.compare_view.focused_media_changed.connect(self._on_compare_focused_media_changed)
        self.central_stack.addWidget(self.compare_view)
        self._expanded_stack_ids: set[int] = set()
        self._favourites_only: bool = False
        self.favourites_enabled: bool = True
        self.grid_view.compare_requested.connect(self.open_compare_view)
        self.grid_view.stack_toggle_requested.connect(self.toggle_stack_expanded)
        self.grid_view.favourite_toggled.connect(self._on_card_favourite_toggled)
        self.ribbon.favourite_filter_selected.connect(self._on_favourite_filter_selected)
        self.grid_view.find_similar_requested.connect(self.open_find_similar_for_media)

        self.setCentralWidget(self.central_stack)

        # 3. Left Navigation Dock & Right Tag/Caption Dock
        self.left_nav_dock = LeftNavigationDock(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_nav_dock)
        self.resizeDocks([self.left_nav_dock], [230], Qt.Horizontal)

        self.right_tag_dock = TagAndCaptionDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_tag_dock)

        self.bottom_dock = BottomFilmstripDock(self)
        self.bottom_dock.bind_sources(self.catalog_model, self.pixmap_cache, self.thumb_repo)
        self.bottom_dock.height_anim_tick.connect(self.media_viewer._update_filmstrip_toggle)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.bottom_dock)
        self.bottom_dock.hide()
        self._filmstrip_viewer_enabled: bool = False
        self._active_create_workspace: Any = None
        self._create_ws_snapshot: dict[str, Any] | None = None
        self._active_subject_ids: list[int] | None = None
        self._active_subject_label: str = ""
        self._active_semantic_worker: Any = None
        self._semantic_cancel: Any = None
        self._semantic_status_text: str = "Semantic AI: Idle"
        self.nav_show_cameras: bool = False
        self.nav_show_subjects: bool = False
        self.auto_group_bursts: bool = False
        self.auto_group_raw_jpeg: bool = False
        self._badge_show_flags: bool = True
        self._badge_show_favourites: bool = True
        self._badge_show_groups: bool = True
        self._saved_badge_prefs: tuple[bool, bool, bool] = (True, True, True)
        try:
            from src.core.semantic_engine import SemanticEngine, SemanticVectorRepository

            self.semantic_repo: Any = SemanticVectorRepository(
                self.engine.catalog_path.parent / "semantic_vectors.db"
            )
            self.semantic_engine: Any = SemanticEngine()
            from src.db.repository import set_smart_album_semantic_provider

            set_smart_album_semantic_provider(self.semantic_repo, self.semantic_engine)
        except Exception:
            self.semantic_repo = None
            self.semantic_engine = None

        # 4. WLPG Bottom Status Bar
        self._build_status_bar()

        # 5. Wire Qt Signals & Slots
        self._connect_signals()

        # 6. Apply saved (or default WLPG Aero) Theme ONCE & Initial Nav Refresh
        initial_theme = "wlpg_aero"
        if "PYTEST_CURRENT_TEST" not in os.environ:
            try:
                _s = QSettings("PhotoGallery64", "PhotoGallery64")
                _t = _s.value("theme", type=str)
                if _t in ("wlpg_aero", "modern", "dark"):
                    initial_theme = _t
                _it = _s.value("icon_theme", type=str)
                if _it in ("wlpg_aero", "fluent", "minimal"):
                    self.current_icon_theme = _it
                    icon_provider.set_icon_theme(_it)
            except Exception:
                pass
        self.apply_theme(initial_theme)
        self.refresh_navigation_and_autocomplete()

        # Seed initial navigation state into history stack
        self._nav_history = [self.left_nav_dock.current_filter_state()]
        self._nav_index = 0
        self._update_nav_buttons()

        # 7. Restore persisted application state (runs single initial query with restored filter)
        self._did_initial_query = False
        self._restore_app_state()
        if not self._did_initial_query:
            self.rebuild_and_run_query()
            self._did_initial_query = True

        # 8. Stagger background startup tasks so initial UI, thumbnails, and scrolling stay responsive
        QTimer.singleShot(400, self._check_stale_scan_crash_trace)
        QTimer.singleShot(1000, self._start_async_avatar_backfill)
        QTimer.singleShot(1200, self.folder_watcher.start_startup_reconciliation)

        from src.utils.scan_logger import mark_startup_stage
        mark_startup_stage("main-window-ready")

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

        self.btn_status_filmstrip = QPushButton(right_ctrl)
        self.btn_status_filmstrip.setCheckable(True)
        self.btn_status_filmstrip.setChecked(False)
        self.btn_status_filmstrip.setIcon(make_wlpg_icon("filmstrip", 18))
        self.btn_status_filmstrip.setIconSize(QSize(18, 18))
        self.btn_status_filmstrip.setFixedSize(22, 22)
        self.btn_status_filmstrip.setToolTip("Show / hide filmstrip (Ctrl+Shift+F)")
        self.btn_status_filmstrip.clicked.connect(lambda: self.dispatch_ribbon_action("toggle_filmstrip_pane"))
        self.btn_status_filmstrip.setVisible(False)
        self.btn_status_filmstrip.hide()

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
        self.slider_status_zoom.setCursor(Qt.PointingHandCursor)
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
            self.btn_status_filmstrip,
            self.btn_status_fit,
            self.btn_status_zoom_out,
            self.slider_status_zoom,
            self.btn_status_zoom_in,
        ):
            sb_btn.setFocusPolicy(Qt.NoFocus)
            sb_btn.setCursor(Qt.PointingHandCursor)

        sb.addPermanentWidget(right_ctrl)

    def _connect_signals(self) -> None:
        # Left Navigation Dock
        self.left_nav_dock.filter_changed.connect(self._on_nav_filter_changed)
        self.left_nav_dock.favourite_filter_requested.connect(self._on_favourite_filter_selected)
        self.left_nav_dock.flag_filter_requested.connect(self._on_nav_smart_flag_selected)
        self.left_nav_dock.builtin_smart_albums_changed.connect(self._save_builtin_smart_albums_prefs)
        self.left_nav_dock.people_directory_requested.connect(self.open_people_directory_view)
        self.left_nav_dock.albums_section_requested.connect(self.open_albums_view)
        self.left_nav_dock.dates_section_requested.connect(self._on_dates_section_requested)
        self.left_nav_dock.section_order_changed.connect(self._on_nav_section_order_changed)
        self.left_nav_dock.folder_add_requested.connect(self._prompt_and_add_folder)
        self.left_nav_dock.folder_remove_requested.connect(self._on_remove_folder_from_library)
        self.left_nav_dock.folder_about_to_rename.connect(self._on_folder_about_to_rename)
        self.left_nav_dock.folder_renamed.connect(self._on_folder_renamed)
        self.left_nav_dock.folder_created.connect(self._on_folder_created)
        self.left_nav_dock.folder_deleted.connect(self._on_folder_deleted)
        self.left_nav_dock.favourite_path_selected.connect(
            lambda path: self.scan_folder(path, include_subfolders=True)
        )
        self.left_nav_dock.files_dropped_on_folder.connect(self._on_files_dropped_on_folder)
        self.left_nav_dock.tag_hierarchy_changed.connect(self.refresh_navigation_and_autocomplete)
        self.left_nav_dock.manage_tags_requested.connect(self.open_tag_hierarchy_dialog)
        self.left_nav_dock.place_selected.connect(self._on_place_selected)
        self.left_nav_dock.subjects_selected.connect(self._on_subjects_selected)
        self.left_nav_dock.subjects_section_clicked.connect(self._on_subjects_section_clicked)
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
        self.left_nav_dock.folder_options_requested.connect(self.open_folder_manager)
        self.left_nav_dock.private_mode_toggled.connect(self.set_show_private_folders)
        if hasattr(self.grid_view, "add_folder_requested"):
            self.grid_view.add_folder_requested.connect(self._prompt_and_add_folder)

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
        self.central_stack.currentChanged.connect(self._on_central_stack_changed)
        icon_provider.signals.theme_changed.connect(self._on_icon_theme_changed)

        # Authentic WLPG Find & Home Tab Filter Signal Connections
        self.ribbon.date_filter_selected.connect(self._on_date_filter_selected)
        self.ribbon.date_preset_selected.connect(self._on_date_preset_selected)
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
        self.scope_bar.similarity_threshold_changed.connect(self._on_similar_threshold_changed)
        self.scope_bar.pin_toggled.connect(self._on_scope_bar_pin_toggled)
        self.scope_bar.close_requested.connect(self.hide_search_scope_bar)
        self.scope_bar.person_remove_requested.connect(self.left_nav_dock.unselect_person)
        self.scope_bar.special_people_clear_requested.connect(self._on_clear_special_people_filter)
        self.scope_bar.tag_remove_requested.connect(
            lambda tid: self.left_nav_dock.set_tag_filter_state(tid, "none")
        )
        self.scope_bar.rating_clear_requested.connect(self.left_nav_dock.clear_rating_filters)
        self.scope_bar.flagged_clear_requested.connect(
            lambda: (self.left_nav_dock.set_flag_filter(None), self.left_nav_dock.set_flagged_only(False))
        )
        self.scope_bar.favourites_clear_requested.connect(
            lambda: self._on_favourite_filter_selected(False)
        )
        self.scope_bar.date_clear_requested.connect(self._clear_date_filter)
        self.scope_bar.media_type_clear_requested.connect(
            lambda: self.left_nav_dock.set_media_type(None)
        )
        self.scope_bar.orientation_clear_requested.connect(
            lambda: self.left_nav_dock.set_orientation_filter("all")
        )
        self.scope_bar.edited_clear_requested.connect(
            lambda: self.left_nav_dock.set_edited_filter(None)
        )
        self.scope_bar.untagged_clear_requested.connect(
            lambda: self.left_nav_dock.set_untagged_only(False)
        )
        self.scope_bar.exclude_tag_remove_requested.connect(
            lambda tid: self.left_nav_dock.set_tag_filter_state(tid, "neutral")
        )
        self.scope_bar.camera_clear_requested.connect(
            lambda: self.left_nav_dock.clear_camera_filter()
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
        if hasattr(self.ribbon, "map_scope_changed"):
            self.ribbon.map_scope_changed.connect(self._on_ribbon_map_scope_changed)
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
        self.right_tag_dock.favourite_edited.connect(self.apply_favourite_to_selection)
        self.right_tag_dock.caption_committed.connect(self.apply_caption_to_selection)
        self.right_tag_dock.geotag_edited.connect(self.apply_geotag_to_selection)
        self.right_tag_dock.adjust_geotag_requested.connect(self.open_adjust_location_dialog)
        self.right_tag_dock.tag_added.connect(self.on_selection_tag_added)
        self.right_tag_dock.tag_removed.connect(self.on_selection_tag_removed)
        self.right_tag_dock.tag_assign_to_all_requested.connect(self.on_selection_tag_assign_to_all)
        self.right_tag_dock.tag_filter_requested.connect(self._on_right_dock_tag_filter_requested)
        self.right_tag_dock.manage_tags_requested.connect(self.open_tag_hierarchy_dialog)
        self.right_tag_dock.person_added.connect(self.on_selection_person_added)
        self.right_tag_dock.person_removed.connect(self.on_selection_person_removed)
        self.right_tag_dock.person_hovered.connect(self._on_dock_person_hovered)
        self.right_tag_dock.person_search_requested.connect(self._on_dock_person_search_requested)
        self.right_tag_dock.person_view_suggestions_requested.connect(self._on_dock_person_view_suggestions_requested)
        self.right_tag_dock.person_tag_as_other_requested.connect(self._on_dock_person_tag_as_other_requested)
        self.right_tag_dock.person_set_display_pic_requested.connect(self._on_dock_person_set_display_pic_requested)
        self.right_tag_dock.person_merge_requested.connect(self._on_dock_person_merge_requested)
        self.right_tag_dock.person_rename_requested.connect(self._on_dock_person_rename_requested)
        self.right_tag_dock.ocr_text_edited.connect(self._on_ocr_text_edited)
        self.right_tag_dock.ocr_rescan_requested.connect(self._on_ocr_rescan_requested)
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
        self.right_tag_dock.folder_open_requested.connect(self.open_folder_in_explorer)
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
        self.right_tag_dock.fine_tune_exposure_ext_changed.connect(
            self.media_viewer.set_exposure_ext_parameters
        )
        self.right_tag_dock.fine_tune_smart_enhance_changed.connect(
            self.media_viewer.set_smart_enhance_parameters
        )
        self.right_tag_dock.fine_tune_structure_changed.connect(
            self.media_viewer.set_structure_parameters
        )
        self.right_tag_dock.fine_tune_color_changed.connect(
            self.media_viewer.set_color_parameters
        )
        self.right_tag_dock.fine_tune_optics_changed.connect(
            self.media_viewer.set_optics_parameters
        )
        self.right_tag_dock.fine_tune_optics_ext_changed.connect(
            self.media_viewer.set_optics_ext_parameters
        )
        self.right_tag_dock.fine_tune_look_profile_changed.connect(
            self.media_viewer.set_look_profile_parameters
        )
        self.right_tag_dock.fine_tune_region_changed.connect(
            self.media_viewer.set_region_parameters
        )
        self.right_tag_dock.region_mask_preview_toggled.connect(
            self.media_viewer.set_region_mask_preview
        )
        self.right_tag_dock.region_mask_fill_toggled.connect(
            self.media_viewer.set_region_mask_fill
        )
        self.right_tag_dock.region_brush_mode_changed.connect(
            self.media_viewer.set_region_brush_mode
        )
        self.right_tag_dock.region_brush_cleared.connect(
            self.media_viewer.clear_region_mask_strokes
        )
        self.right_tag_dock.region_mask_invert_requested.connect(
            self.media_viewer.invert_region_mask
        )
        self.media_viewer.region_brush_size_shortcut_changed.connect(
            self.right_tag_dock.slider_region_brush_size.setValue
        )
        self.right_tag_dock.dock_mode_changed.connect(
            lambda m: (
                self.media_viewer.set_suppress_face_overlays(m in ("fine_tune", "depth_light")),
                self.media_viewer.set_depth_light_active(m == "depth_light"),
            )
        )
        self.right_tag_dock.pane_interacted.connect(self._on_right_pane_interacted)
        self.right_tag_dock.region_scope_changed.connect(
            self._update_region_histogram_mask
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
        self.right_tag_dock.fine_tune_detail_ext_changed.connect(
            self.media_viewer.set_detail_ext
        )
        self.right_tag_dock.fine_tune_curve_changed.connect(
            self.media_viewer.set_tone_curves
        )
        self.right_tag_dock.fine_tune_hsl_changed.connect(
            self.media_viewer.set_hsl_parameters
        )
        self.right_tag_dock.wb_eyedropper_toggled.connect(
            self.media_viewer.set_wb_eyedropper_mode
        )
        self.media_viewer.wb_eyedropper_finished.connect(
            lambda: self.right_tag_dock.btn_wb_eyedropper.setChecked(False)
            if hasattr(self.right_tag_dock, "btn_wb_eyedropper")
            else None
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
        self.right_tag_dock.draw_line_straighten_requested.connect(
            lambda: self.media_viewer.set_crop_mode(True, focus_straighten=True)
            if not getattr(self.media_viewer, "is_crop_mode", False)
            else self.media_viewer.btn_crop_straighten_line.setChecked(True)
        )
        self.media_viewer.straighten_preview_changed.connect(
            self.right_tag_dock.sync_straighten_angle
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
        self.right_tag_dock.depth_virtual_lights_changed.connect(
            self.media_viewer.set_virtual_lights_parameters
        )
        self.right_tag_dock.depth_light_gizmos_toggled.connect(
            self.media_viewer.set_virtual_light_gizmos_visible
        )
        self.right_tag_dock.depth_selected_light_changed.connect(
            self.media_viewer.set_selected_virtual_light
        )
        self.right_tag_dock.depth_relight_clear_requested.connect(
            self.media_viewer.clear_relight_only
        )
        self.media_viewer.virtual_lights_canvas_changed.connect(
            self.right_tag_dock.sync_virtual_lights_from_canvas
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
        self.bottom_dock.media_selected.connect(self._on_filmstrip_media_selected)
        self.bottom_dock.close_requested.connect(self._on_bottom_dock_close_requested)
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

        self.sc_filmstrip = QShortcut(QKeySequence("Ctrl+Shift+F"), self)
        self.sc_filmstrip.activated.connect(
            lambda: self.dispatch_ribbon_action("toggle_filmstrip_pane")
            if self.central_stack.currentWidget() == self.media_viewer
            and getattr(self, "_active_create_workspace", None) is None
            else None
        )

        self.sc_properties_ret = QShortcut(QKeySequence("Alt+Return"), self)
        self.sc_properties_ret.activated.connect(self.open_file_properties)
        self.sc_properties_enter = QShortcut(QKeySequence("Alt+Enter"), self)
        self.sc_properties_enter.activated.connect(self.open_file_properties)

        # F1: Keyboard Shortcuts Cheat-Sheet & Quick-Tag Hotkeys (Alt+1..Alt+9)
        self.sc_shortcuts = QShortcut(QKeySequence(Qt.Key_F1), self)
        self.sc_shortcuts.activated.connect(lambda: self.open_keyboard_shortcuts_dialog(0))
        self._quick_tag_shortcuts: list[QShortcut] = []
        for slot in range(1, 10):
            sc_qt = QShortcut(QKeySequence(f"Alt+{slot}"), self)
            sc_qt.activated.connect(lambda s=slot: self._on_quick_tag_hotkey(s))
            self._quick_tag_shortcuts.append(sc_qt)

        # SingleMediaViewer
        self.media_viewer.closed.connect(self.show_grid_view)
        self.media_viewer.navigate_requested.connect(self.navigate_single_viewer)
        self.media_viewer.pixel_edit_saved.connect(self._on_pixel_edit_saved)
        self.media_viewer.pixel_edit_saved_as_copy.connect(self._on_pixel_edit_saved_as_copy)
        self.media_viewer.pixel_edit_reverted.connect(self._on_pixel_edit_reverted)
        self.media_viewer.face_region_tagged.connect(self._on_viewer_face_region_tagged)
        self.media_viewer.image_changed.connect(
            lambda _img: self._update_region_histogram_mask(from_image_changed=True)
        )
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
        self.ribbon.tab_widget.currentChanged.connect(self._on_ribbon_tab_changed)

        # WorldMapView signals connected lazily when view is created in show_map_view()

    # -------------------------------------------------------------------------
    # Folder Scanning & Progressive Sync
    # -------------------------------------------------------------------------
    def _default_pictures_dir(self) -> str:
        from PySide6.QtCore import QStandardPaths
        pic = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation)
        if pic and Path(pic).is_dir():
            return pic
        fallback = Path.home() / "Pictures"
        return str(fallback if fallback.is_dir() else Path.home())

    def _prompt_and_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Add Folder to Gallery", self._default_pictures_dir()
        )
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
        self.show_status_message(f"✓ Imported to {len(created_folders)} folder(s)", 4000)

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

    def _on_folder_about_to_rename(self, old_path: str) -> None:
        """Releases QFileSystemWatcher directory handles before os.rename() on Windows."""
        if hasattr(self, "folder_watcher") and self.folder_watcher is not None:
            self.folder_watcher.release_watched_subtree(old_path)

    def _on_folder_renamed(self, old_path: str, new_path: str, new_name: str) -> None:
        """Handles inline folder renaming: updates catalog.db, watcher, and UI state."""
        self.catalog_repo.rename_folder(old_path, new_path, new_name)
        self._media_meta_lookup.clear()
        if hasattr(self, "folder_watcher") and self.folder_watcher is not None:
            self.folder_watcher.update_watched_paths()
        if getattr(self.left_nav_dock, "_folder_name", "") == Path(old_path).name:
            self.left_nav_dock._folder_name = new_name
        self.refresh_navigation_and_autocomplete({"folders"})
        self.rebuild_and_run_query()
        self.statusBar().showMessage(f"Renamed folder to '{new_name}'.", 4000)

    def _on_folder_created(self, new_dir_str: str) -> None:
        """Handles newly created folders."""
        if hasattr(self, "folder_watcher") and self.folder_watcher is not None:
            self.folder_watcher.update_watched_paths()
        self.refresh_navigation_and_autocomplete({"folders"})
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
                if include_subfolders:
                    self.refresh_navigation_and_autocomplete()
            except Exception:
                pass

        self._scanner_cancel.set()
        self._scanner_cancel = CancellationFlag()
        self._scanner_status_text = "Scanning: Active"
        self._last_scanned_root = root_p
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
            self._retain_pool_worker(worker)
            worker.signals.finished.connect(
                lambda *_args, w=worker: self._release_pool_worker(w)
            )
            self.scanner_pool.start(worker, -1)

    def _on_sync_stage1_completed(self, discovered_count: int) -> None:
        """Refreshes navigation tree and grid immediately after fast Stage 1 os.scandir walk completes."""
        self._media_meta_lookup.clear()
        self.refresh_navigation_and_autocomplete()
        self.rebuild_and_run_query()
        if hasattr(self, "folder_watcher"):
            self.folder_watcher.update_watched_paths()

    def open_folder_manager(self, folder_id: int | None = None) -> None:
        """Open the Picasa-style Folder Manager Dialog."""
        from src.ui.folder_manager_dialog import FolderManagerDialog

        dlg = FolderManagerDialog(
            self.repo,
            self,
            initial_folder_id=folder_id,
            is_dark=(self.current_theme == "dark"),
        )

        policies_dirty = False

        def _apply_policies_changed() -> None:
            if hasattr(self, "folder_watcher") and self.folder_watcher is not None:
                self.folder_watcher.update_watched_paths()
            self._media_meta_lookup.clear()
            self.refresh_navigation_and_autocomplete()
            self.rebuild_and_run_query()

        def _on_policies_changed() -> None:
            nonlocal policies_dirty
            policies_dirty = True

        def _on_open_ocr_options() -> None:
            self._open_settings_dialog(initial_tab="Add-ons")
            dlg._refresh_ocr_backend_label()

        def _on_add_folder_from_manager(folder_path: str) -> None:
            self.scan_folder(folder_path, include_subfolders=True)
            if hasattr(self, "folder_watcher") and self.folder_watcher is not None:
                self.folder_watcher.update_watched_paths()

        def _on_folder_removed_from_manager(removed_ids: list[int]) -> None:
            for mid in removed_ids:
                self.pixmap_cache.pop(mid, None)
                self._media_meta_lookup.pop(mid, None)

        dlg.policies_changed.connect(_on_policies_changed)
        dlg.scan_ocr_requested.connect(self._start_ocr_worker)
        dlg.open_ocr_options_requested.connect(_on_open_ocr_options)
        dlg.folder_add_requested.connect(_on_add_folder_from_manager)
        dlg.folder_removed.connect(_on_folder_removed_from_manager)
        dlg.exec()
        if policies_dirty:
            _apply_policies_changed()

    def _update_private_menu_visibility(self) -> None:
        """Show the File menu Unlock/Hide Private Folders item only when Private Folders are in use."""
        if not hasattr(self.ribbon, "act_show_private_folders"):
            return
        from src.ui.folder_manager_dialog import get_private_auth_mode, has_private_pin

        has_private_folder = False
        try:
            has_private_folder = bool(self.repo.has_private_folders())
        except Exception:
            has_private_folder = False
        in_use = (
            has_private_folder
            or bool(getattr(self.repo, "include_private", False))
            or has_private_pin()
            or get_private_auth_mode() != "none"
        )
        self.ribbon.act_show_private_folders.setVisible(in_use)

    def set_show_private_folders(self, enabled: bool) -> None:
        """Toggle Private Mode session isolation across catalog queries, dock trees, and grid."""
        from PySide6.QtWidgets import QInputDialog, QLineEdit, QMessageBox
        from src.ui.folder_manager_dialog import (
            get_private_auth_mode,
            get_private_auto_lock_minutes,
            has_private_pin,
            verify_private_pin,
            verify_windows_login,
        )

        enabled = bool(enabled)
        auth_mode = get_private_auth_mode()
        if enabled:
            if auth_mode == "windows":
                if not verify_windows_login():
                    enabled = False
            elif auth_mode == "pin" or has_private_pin():
                pin, ok = QInputDialog.getText(
                    self,
                    "Unlock Private Folders",
                    "Enter Private Mode PIN / Passcode:",
                    QLineEdit.Password,
                )
                if not ok or not verify_private_pin(pin):
                    if ok:
                        QMessageBox.warning(self, "Incorrect PIN", "The PIN / Passcode you entered is incorrect.")
                    enabled = False

        self.repo.set_include_private(enabled)
        if hasattr(self, "batch_people_view") and self.batch_people_view is not None:
            self.batch_people_view.repo.set_include_private(enabled)
        if hasattr(self, "left_nav_dock"):
            self.left_nav_dock.set_private_mode_state(enabled)
        if hasattr(self.ribbon, "act_show_private_folders"):
            act = self.ribbon.act_show_private_folders
            act.blockSignals(True)
            act.setChecked(enabled)
            act.setText(
                "Hide Private Folders (🔓 Unlocked)" if enabled else "Show Private Folders (🔒 Locked)"
            )
            act.blockSignals(False)

        if getattr(self, "_private_auto_lock_timer", None) is None:
            self._private_auto_lock_timer = QTimer(self)
            self._private_auto_lock_timer.setSingleShot(True)
            self._private_auto_lock_timer.timeout.connect(lambda: self.set_show_private_folders(False))

        auto_mins = get_private_auto_lock_minutes()
        if enabled and auto_mins > 0:
            self._private_auto_lock_timer.start(auto_mins * 60_000)
        else:
            self._private_auto_lock_timer.stop()

        if self.catalog_repo.has_private_folders():
            self._media_meta_lookup.clear()
            self.refresh_navigation_and_autocomplete()
            self.rebuild_and_run_query()
            self._refresh_subjects_nav_counts()
            if hasattr(self, "batch_people_view") and self.batch_people_view is not None:
                if self.central_stack.currentWidget() == self.batch_people_view:
                    self.batch_people_view.refresh_suggestions()
                else:
                    self.batch_people_view._needs_external_refresh = True
        try:
            persist_priv = bool(enabled) if (auth_mode == "none" and not has_private_pin()) else False
            QSettings("PhotoGallery64", "PhotoGallery64").setValue("ui/show_private_folders", persist_priv)
        except Exception:
            pass
        self.statusBar().showMessage(
            "Private Folders unlocked (🔒) for this session." if enabled else "Private Folders locked and hidden.",
            4000,
        )

    def scan_ocr_for_selection(self, media_ids: Sequence[int] | None = None) -> None:
        """On-demand OCR scan for selected/active photos, even if folder scan_ocr is disabled."""
        target_ids = [int(m) for m in media_ids] if media_ids else list(self.get_active_target_ids())
        if not target_ids:
            return
        from src.core.ocr_engine import has_configured_ocr_engine
        from src.ui.dialogs import prompt_and_download_rapidocr_addon

        if not has_configured_ocr_engine():
            if not prompt_and_download_rapidocr_addon(self, engine=self.engine):
                return
        self._start_ocr_worker(media_ids=target_ids)

    def _start_ocr_worker(
        self,
        folder_id: int | None = None,
        media_ids: Sequence[int] | None = None,
    ) -> None:
        """Launch background OCR indexing for unread photos (or forced on-demand media_ids)."""
        if getattr(self, "_workers_paused", False) and not media_ids:
            return
        if not media_ids:
            from src.core.ocr_engine import OcrEngine

            if OcrEngine().backend_name in ("stub", "windows_media_ocr"):
                return
        from src.workers.ocr_worker import OcrScanWorker

        if getattr(self, "_active_ocr_worker", None) is not None:
            self._active_ocr_worker.cancel()
        worker = OcrScanWorker(self.engine, folder_id=folder_id, media_ids=media_ids)
        self._ocr_status_text = "OCR: Starting..."
        self._update_worker_monitor()

        def _on_ocr_progressed(cur: int, total: int) -> None:
            if getattr(self, "_active_ocr_worker", None) is worker:
                self._ocr_status_text = f"OCR: {cur:,} / {total:,}"
                self._update_worker_monitor()

        def _on_ocr_media_updated(mid: int) -> None:
            self._media_meta_lookup.pop(int(mid), None)
            active_ids = self.get_active_target_ids()
            if active_ids and int(active_ids[0]) == int(mid) and hasattr(self, "right_tag_dock"):
                summary = self.catalog_repo.get_selection_summary([int(mid)])
                self.right_tag_dock.bind_selection([int(mid)], summary)

        def _on_ocr_finished(count: int) -> None:
            if getattr(self, "_active_ocr_worker", None) is worker:
                self._active_ocr_worker = None
                self._ocr_status_text = "OCR: Idle"
                self._update_worker_monitor()
            if count > 0:
                self.statusBar().showMessage(
                    f"✓ OCR text indexing finished ({count:,} photo(s) indexed).", 4000
                )

        def _on_ocr_error(_err: str) -> None:
            if getattr(self, "_active_ocr_worker", None) is worker:
                self._active_ocr_worker = None
                self._ocr_status_text = "OCR: Idle"
                self._update_worker_monitor()

        worker.signals.status_message.connect(
            lambda msg: self.statusBar().showMessage(msg, 4000)
        )
        worker.signals.progressed.connect(_on_ocr_progressed)
        worker.signals.media_updated.connect(_on_ocr_media_updated)
        worker.signals.finished.connect(_on_ocr_finished)
        worker.signals.error.connect(_on_ocr_error)
        self._active_ocr_worker = worker
        self._retain_pool_worker(worker)
        worker.signals.finished.connect(
            lambda *_args, w=worker: self._release_pool_worker(w)
        )
        QThreadPool.globalInstance().start(worker)

    def _visible_non_private_media_ids(self) -> set[int] | None:
        """
        Return all non-missing, non-private media IDs when Private Mode is locked and private
        folders exist; return None when no privacy restriction is active.
        """
        try:
            if bool(getattr(self.catalog_repo, "include_private", False)):
                return None
            if not self.catalog_repo.has_private_folders():
                return None
            rows = self.engine.catalog_conn().execute(
                """
                SELECT m.id
                FROM media m
                JOIN folders f ON f.id = m.folder_id
                WHERE m.is_missing = 0 AND f.is_private = 0;
                """
            ).fetchall()
            return {int(r[0]) for r in rows}
        except Exception:
            return None

    def _start_semantic_worker(self, *, force: bool = False) -> None:
        """Launch background MobileCLIP-S1 semantic indexing from cached thumbnails.db BLOBs."""
        if getattr(self, "_is_shutting_down", False):
            return
        if getattr(self, "_workers_paused", False) and not force:
            return
        if (
            getattr(self, "semantic_repo", None) is None
            or getattr(self, "semantic_engine", None) is None
            or not self.semantic_engine.is_available()
        ):
            return
        from legacy_reference.imgBrowserv13 import CancellationFlag
        from src.workers.semantic_worker import SemanticIndexWorker

        if getattr(self, "_semantic_cancel", None) is not None:
            try:
                self._semantic_cancel.set()
            except Exception:
                pass

        flag = CancellationFlag()
        self._semantic_cancel = flag
        worker = SemanticIndexWorker(
            self.engine,
            self.semantic_repo,
            self.semantic_engine,
            candidates=None,
            cancel_flag=flag,
            is_paused_cb=lambda: bool(getattr(self, "_workers_paused", False)),
        )
        self._semantic_status_text = "Semantic AI: Starting..."
        self._update_worker_monitor()

        def _on_sem_prog(done: int, total: int, _msg: str) -> None:
            if getattr(self, "_active_semantic_worker", None) is worker:
                if total > 0:
                    self._semantic_status_text = f"Semantic AI: {done:,} / {total:,}"
                else:
                    self._semantic_status_text = "Semantic AI: Idle"
                self._update_worker_monitor()

        def _on_sem_finished(indexed_now: int, total_indexed: int) -> None:
            if getattr(self, "_active_semantic_worker", None) is worker:
                self._active_semantic_worker = None
                self._semantic_status_text = "Semantic AI: Idle"
                self._update_worker_monitor()
            self._refresh_subjects_nav_counts()
            if indexed_now > 0:
                self.statusBar().showMessage(
                    f"✓ Semantic AI indexed {indexed_now:,} photo(s) ({total_indexed:,} total).",
                    4000,
                )

        def _on_sem_error(_err: str) -> None:
            if getattr(self, "_active_semantic_worker", None) is worker:
                self._active_semantic_worker = None
                self._semantic_status_text = "Semantic AI: Idle"
                self._update_worker_monitor()

        worker.signals.progress.connect(_on_sem_prog)
        worker.signals.finished_stats.connect(_on_sem_finished)
        worker.signals.error.connect(_on_sem_error)
        self._active_semantic_worker = worker
        self._retain_pool_worker(worker)
        worker.signals.finished_stats.connect(
            lambda *_args, w=worker: self._release_pool_worker(w)
        )
        worker.signals.error.connect(
            lambda *_args, w=worker: self._release_pool_worker(w)
        )
        QThreadPool.globalInstance().start(worker)

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
        if was_completed and upserted_count > 0:
            scope_fids = self._current_stack_scope_folder_ids()
            if getattr(self, "auto_group_raw_jpeg", False):
                try:
                    self.catalog_repo.auto_stack_raw_jpeg_pairs(folder_ids=scope_fids)
                except Exception:
                    pass
            if getattr(self, "auto_group_bursts", False):
                try:
                    self.catalog_repo.auto_stack_bursts(
                        max_gap_sec=2.0, folder_ids=scope_fids, thumb_repo=self.thumb_repo
                    )
                except Exception:
                    pass
        self.refresh_navigation_and_autocomplete()
        self.rebuild_and_run_query()
        if hasattr(self, "folder_watcher"):
            self.folder_watcher.update_watched_paths()
        if was_completed and getattr(self, "_last_scanned_root", None) is not None:
            _rp = self._last_scanned_root
            self._last_scanned_root = None
            QTimer.singleShot(250, lambda p=_rp: self._maybe_prompt_picasa_import(p))
        if (
            was_completed
            and upserted_count > 0
            and getattr(self, "find_people_enabled", True)
            and self._are_face_models_installed()
        ):
            self.start_face_detection()
        if was_completed and upserted_count > 0:
            self._start_ocr_worker()
            self._start_semantic_worker()

    def _on_folder_watcher_delta(
        self, folder_path: str, added: int, modified: int, deleted: int, folders_changed: int = 0
    ) -> None:
        if added > 0 or modified > 0 or deleted > 0 or folders_changed > 0:
            self._media_meta_lookup.clear()
            pend = self._pending_watcher_delta
            pend["added"] += int(added)
            pend["modified"] += int(modified)
            pend["deleted"] += int(deleted)
            pend["folders_changed"] += int(folders_changed)
            if folder_path and folder_path not in pend["folders"]:
                pend["folders"].append(folder_path)
            self._watcher_delta_timer.start(1800)

    def _flush_folder_watcher_delta(self) -> None:
        if getattr(self, "_is_shutting_down", False):
            return
        # Never block an open modal dialog (e.g. Tag Hierarchy / Options) with a full nav+grid refresh
        if QApplication.activeModalWidget() is not None:
            self._watcher_delta_timer.start(1500)
            return
        # Coalesce watcher refreshes while XMP write-back daemon is actively draining batches to disk
        if getattr(getattr(self, "xmp_daemon", None), "_is_flushing", False):
            self._watcher_delta_timer.start(3000)
            return
        # Rate-limit heavy full nav+grid refreshes to at most once every 5s during sustained bursts
        import time
        now = time.monotonic()
        last_flush = getattr(self, "_last_watcher_delta_flush_ts", 0.0)
        if last_flush > 0.0 and (now - last_flush) < 5.0 and "PYTEST_CURRENT_TEST" not in os.environ:
            remaining_ms = max(600, int((5.0 - (now - last_flush)) * 1000))
            self._watcher_delta_timer.start(remaining_ms)
            return

        pend = self._pending_watcher_delta
        added = int(pend["added"])
        modified = int(pend["modified"])
        deleted = int(pend["deleted"])
        folders_changed = int(pend["folders_changed"])
        folders = list(pend["folders"])
        pend["added"] = 0
        pend["modified"] = 0
        pend["deleted"] = 0
        pend["folders_changed"] = 0
        pend["folders"].clear()

        if added <= 0 and modified <= 0 and deleted <= 0 and folders_changed <= 0:
            return

        self._last_watcher_delta_flush_ts = now
        self._media_meta_lookup.clear()
        if added > 0 or deleted > 0 or folders_changed > 0:
            self.refresh_navigation_and_autocomplete()
            self.rebuild_and_run_query()
        else:
            # Modify-only delta (e.g. metadata/XMP touch): skip 8-tree full-catalog nav rebuild
            self.rebuild_and_run_query()
        if len(folders) == 1:
            fname = os.path.basename(folders[0]) or folders[0]
            target_label = f"'{fname}'"
        elif folders:
            target_label = f"{len(folders)} folders"
        else:
            target_label = "library"
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
        self.statusBar().showMessage(f"✓ Synced {target_label}: {summary}", 4000)
        if added > 0 or folders_changed > 0:
            for p in folders[:4]:
                QTimer.singleShot(250, lambda path=p: self._maybe_prompt_picasa_import(path))
        if (
            added > 0
            and getattr(self, "find_people_enabled", True)
            and self._are_face_models_installed()
        ):
            self.start_face_detection()
        if added > 0:
            self._start_ocr_worker()
            self._start_semantic_worker()

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
            if (
                added > 0
                and getattr(self, "find_people_enabled", True)
                and self._are_face_models_installed()
            ):
                self.start_face_detection()
        QTimer.singleShot(3500, self._start_ocr_worker)
        QTimer.singleShot(4500, self._start_semantic_worker)
        QTimer.singleShot(5000, self._maybe_start_startup_face_scan)

    def _on_missing_root_folders_detected(self, missing_roots: list[dict[str, object]]) -> None:
        """Prompts the user when a monitored root folder is missing on disk (moved or disconnected)."""
        if not missing_roots or not self.isVisible():
            return
        changed = False
        for item in missing_roots:
            fid = int(item.get("id", 0) or 0)
            f_path = str(item.get("path", "") or "")
            if not fid or fid in self._missing_roots_prompted:
                continue
            self._missing_roots_prompted.add(fid)

            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Missing Library Folder")
            box.setText(f"The monitored folder could not be found:\n\n{f_path}")
            box.setInformativeText(
                "It may have been moved, renamed, or stored on a disconnected drive. "
                "Would you like to locate its new folder, remove it from the catalog, "
                "or keep it offline for now?"
            )
            btn_locate = box.addButton("Locate Folder...", QMessageBox.AcceptRole)
            btn_remove = box.addButton("Remove from Catalog", QMessageBox.DestructiveRole)
            box.addButton("Keep Offline", QMessageBox.RejectRole)
            box.exec()

            clicked = box.clickedButton()
            if clicked == btn_locate:
                chosen = QFileDialog.getExistingDirectory(
                    self,
                    f"Locate Missing Folder ({os.path.basename(f_path) or f_path})",
                    os.path.dirname(f_path) if os.path.dirname(f_path) else "",
                )
                if chosen and os.path.isdir(chosen):
                    self.catalog_repo.relocate_folder_tree(fid, chosen)
                    self.folder_watcher.update_watched_paths()
                    self.scan_folder(chosen, include_subfolders=True)
                    self.statusBar().showMessage(f"Relocated library folder to {chosen}", 4000)
                    changed = True
            elif clicked == btn_remove:
                removed_mids = self.catalog_repo.remove_folder_tree(fid)
                if removed_mids:
                    self.thumb_repo.invalidate(removed_mids)
                self.folder_watcher.update_watched_paths()
                self.statusBar().showMessage(f"Removed missing folder from catalog: {f_path}", 4000)
                changed = True

        if changed:
            self._media_meta_lookup.clear()
            self.refresh_navigation_and_autocomplete()
            self.rebuild_and_run_query()

    def on_worker_error(self, err_msg: str) -> None:
        self.show_status_message(f"⚠ {err_msg}", 4500)

    def on_metadata_batch_synced(self, media_ids: list[int]) -> None:
        self._update_worker_monitor()

    # -------------------------------------------------------------------------
    # Background Face Detection & Recognition
    # -------------------------------------------------------------------------
    def _are_face_models_installed(self) -> bool:
        """Returns True when InsightFace and its ONNX model files are already cached locally."""
        try:
            from src.core.face_engine import FaceEngine

            return bool(FaceEngine.probe_status().get("available", False))
        except Exception:
            return False

    def _maybe_start_startup_face_scan(self) -> None:
        """Starts background face scanning shortly after startup only when models are already installed."""
        if self._is_shutting_down or getattr(self, "_workers_paused", False):
            return
        if not getattr(self, "find_people_enabled", True):
            return
        if "PYTEST_CURRENT_TEST" in os.environ:
            return
        if self._are_face_models_installed():
            self.start_face_detection()

    def start_face_detection(
        self, force_rescan: bool = False, prompt_if_missing: bool = False
    ) -> None:
        """Launches background FaceDetectionWorker on self.face_pool if not already running."""
        if self._is_shutting_down:
            return
        if self._face_worker is not None:
            return
        allow_download = False
        if not self._are_face_models_installed():
            if not prompt_if_missing:
                return
            try:
                from src.core.face_engine import FaceEngine

                probe = FaceEngine.probe_status()
                if not probe.get("insightface_installed"):
                    self.show_status_message(
                        "Face recognition backend (InsightFace) is not installed in this build.",
                        5000,
                    )
                    return
            except Exception:
                return
            reply = QMessageBox.question(
                self,
                "Download Face Recognition Models",
                "AI face recognition models (~280 MB) are not downloaded yet.\n\n"
                "Download them now to scan and group people?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return
            allow_download = True
        self._face_cancel.clear()
        cpu_count = os.cpu_count() or 4
        batch_size = 25 if cpu_count <= 4 else 50
        throttle_ms = 65 if cpu_count <= 4 else 25
        worker = FaceDetectionWorker(
            engine=self.engine,
            cancel_flag=self._face_cancel,
            batch_size=batch_size,
            throttle_ms=throttle_ms,
            force_rescan=force_rescan,
            allow_model_download=allow_download,
        )
        worker.signals.status_message.connect(self.on_face_status_message)
        worker.signals.batch_progressed.connect(self.on_face_batch_progressed)
        worker.signals.suggestions_updated.connect(self.on_face_suggestions_updated)
        worker.signals.worker_finished.connect(self.on_face_worker_finished)
        self._face_worker = worker
        self._face_phase = "init"
        self._face_status_text = "Faces: Starting face scan..."
        self._update_worker_monitor()
        self._retain_pool_worker(worker)
        worker.signals.worker_finished.connect(
            lambda *_args, w=worker: self._release_pool_worker(w)
        )
        self.face_pool.start(worker)

    def stop_face_detection(self) -> None:
        """Requests cancellation of background FaceDetectionWorker."""
        self._face_cancel.set()

    def on_face_status_message(self, message: str) -> None:
        """Updates status text when FaceDetectionWorker transitions between stages."""
        if self._face_worker is not None:
            clean_msg = message
            if "Downloading" in message:
                self._face_phase = "download"
                clean_msg = message
                self.show_status_message(message, 6000)
            elif "Loading" in message:
                self._face_phase = "load"
                clean_msg = message
                self.show_status_message(message, 3500)
            elif "Stage 1" in message:
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
        """Debounced refresh of People tree, carousel, and BatchPeopleView when new face suggestions arrive."""
        timer = getattr(self, "_face_suggestions_debounce_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._flush_face_suggestions_updated)
            self._face_suggestions_debounce_timer = timer
        timer.start(600)

    def _flush_face_suggestions_updated(self) -> None:
        if getattr(self, "_is_shutting_down", False):
            return
        if hasattr(self.left_nav_dock, "refresh_people_only"):
            people = self.left_nav_dock.refresh_people_only(self.catalog_repo)
            self.ribbon.populate_face_carousel(people)
        else:
            self.refresh_navigation_and_autocomplete({"people", "albums"})
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
            self.refresh_people_navigation_and_autocomplete()

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

    def refresh_navigation_and_autocomplete(
        self,
        sections: set[str] | frozenset[str] | Sequence[str] | None = None,
    ) -> None:
        sec_set: set[str] | None = (
            {str(s).strip().lower() for s in sections if str(s).strip()}
            if sections is not None
            else None
        )
        if sec_set is None or bool(sec_set & {"places", "folders"}):
            if getattr(self, "_map_view", None) is not None and hasattr(self._map_view, "mark_data_dirty"):
                self._map_view.mark_data_dirty()
        if sec_set is None or "folders" in sec_set:
            self._update_private_menu_visibility()

        need_people = sec_set is None or "people" in sec_set
        need_tags = sec_set is None or "tags" in sec_set
        people = self.catalog_repo.get_people_list() if need_people else None
        tags = self.catalog_repo.get_tag_tree() if need_tags else None
        self.left_nav_dock.refresh_trees(
            self.catalog_repo,
            preloaded_people=people,
            preloaded_tags=tags,
            sections=sec_set,
        )

        if need_people and people is not None:
            self.ribbon.populate_face_carousel(people)
            if hasattr(self.right_tag_dock, "_people_model"):
                self.right_tag_dock._people_model.setStringList([str(p["name"]) for p in people])

        if need_tags and tags is not None:
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
            self.ribbon.populate_tags_menu(tags)
            if hasattr(self.right_tag_dock, "txt_add_tag"):
                self.right_tag_dock.txt_add_tag.set_tag_lists(all_tag_paths, mru_valid[:12])

        if sec_set is None or "subjects" in sec_set:
            self._refresh_subjects_nav_counts()

    def refresh_people_navigation_and_autocomplete(self) -> None:
        """Fast people-only navigation and autocomplete refresh (avoids full tree rebuild)."""
        if hasattr(self.left_nav_dock, "refresh_people_only"):
            people = self.left_nav_dock.refresh_people_only(self.catalog_repo)
        else:
            people = self.catalog_repo.get_people_list()
            self.left_nav_dock.refresh_trees(self.catalog_repo, preloaded_people=people)
        self.ribbon.populate_face_carousel(people)
        if hasattr(self, "people_directory_view"):
            self.people_directory_view.invalidate_cache()
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
            state.get("person_ids")
            or state.get("include_tag_ids")
            or state.get("exclude_tag_ids")
            or state.get("untagged_only")
            or state.get("ratings")
            or state.get("flagged_only")
            or state.get("flag_filter") is not None
            or state.get("edited_only")
            or state.get("edited_filter") is not None
            or getattr(self, "_favourites_only", False)
            or has_media_filter
            or has_orientation_filter
            or bool(state.get("dates"))
            or state.get("date_year") is not None
            or state.get("date_month") is not None
            or state.get("date_day") is not None
            or self._active_date_range is not None
            or self._search_text
            or self._active_bbox is not None
            or self._active_map_media_ids is not None
            or self._active_map_location_label is not None
            or getattr(self, "_active_similar_seed_id", None) is not None
            or getattr(self, "_active_similar_order", None) is not None
            or bool(state.get("locations"))
            or state.get("location_country") is not None
            or state.get("location_state") is not None
            or state.get("location_city") is not None
            or bool(state.get("camera_models"))
            or state.get("camera_model") is not None
            or bool(state.get("active_subject_keys"))
            or bool(getattr(self, "_active_subject_label", ""))
            or (getattr(self, "_active_subject_ids", None) is not None)
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
        cams = list(state.get("camera_models") or ())
        if cams:
            if len(cams) == 1:
                c = cams[0]
                return f"Camera: {'No Camera Recorded' if c == '__NO_CAMERA__' else c}"
            return f"Cameras: {len(cams)} selected"
        cam = state.get("camera_model")
        if cam:
            return f"Camera: {'No Camera Recorded' if cam == '__NO_CAMERA__' else cam}"
        subj_lbl = getattr(self, "_active_subject_label", "")
        if subj_lbl:
            return f"Subject: {subj_lbl}"
        return None

    def _active_place_chip_label(self, state: dict[str, Any]) -> str | None:
        if self._active_map_location_label:
            return self._active_map_location_label
        locs = list(state.get("locations") or ())
        if len(locs) > 1:
            labels = [", ".join(p for p in (ci, s, c) if p) for c, s, ci in locs]
            labels = [lbl for lbl in labels if lbl]
            if labels:
                return f"📍 {' + '.join(labels)}"
        parts = [
            p
            for p in (
                state.get("location_city"),
                state.get("location_state"),
                state.get("location_country"),
            )
            if p
        ]
        if parts:
            return f"📍 {', '.join(parts)}"
        return None

    def _update_scope_bar_visibility(self) -> None:
        state = dict(self.left_nav_dock.current_filter_state())
        state["favourites_only"] = bool(getattr(self, "_favourites_only", False))
        self.scope_bar.set_scope_label(self._active_scope_label(state))

        self.scope_bar.update_active_filter_chips(
            state,
            self.catalog_repo,
            location_label=getattr(self, "_active_similar_label", None) or self._active_place_chip_label(state),
            special_people_label=getattr(self, "_active_special_people_label", None),
            date_label=getattr(self, "_active_date_label", None),
        )
        if (
            self.scope_bar.is_pinned()
            or self._has_active_filter_facets()
            or getattr(self, "_active_similar_label", None)
            or bool(getattr(self, "_active_subject_label", ""))
            or bool(state.get("camera_models") or state.get("camera_model"))
        ):
            self.scope_bar.show()
        else:
            self.scope_bar.hide()

    def _on_search_all_requested(self) -> None:
        """Removes only the folder filter, expanding search to all photos and videos while keeping bar open and filters intact."""
        self._search_scope_bar_explicitly_opened = True
        had_similar = getattr(self, "_active_similar_seed_id", None) is not None
        self._preserving_similar_on_nav_change = had_similar
        try:
            self.left_nav_dock.clear_folder_filter()
        finally:
            self._preserving_similar_on_nav_change = False
        if had_similar and getattr(self, "_active_similar_seed_id", None) is not None:
            self._on_similar_threshold_changed(getattr(self, "_active_similar_pct", 81))
        self.scope_bar.show()
        if not had_similar:
            self.scope_bar.focus_search_input()

    def _on_clear_location_filter(self) -> None:
        """Removes the location/map/similar/subject filter chip and re-runs the catalog query."""
        focus_mid: int | None = None
        if getattr(self, "_active_similar_seed_id", None) is not None:
            sel = self.grid_view.selected_media_ids() if hasattr(self, "grid_view") else []
            focus_mid = int(sel[0]) if sel else int(self._active_similar_seed_id)

        self._active_map_media_ids = None
        self._active_similar_order = None
        self._active_similar_scored = None
        self._active_similar_label = None
        self._active_similar_seed_id = None
        self._active_similar_seed_name = ""
        self._active_subject_ids = None
        self._active_subject_label = ""
        self.scope_bar.set_similarity_controls(False)
        self._active_bbox = None
        self._active_map_location_label = None
        if hasattr(self, "left_nav_dock") and self.left_nav_dock is not None:
            self.left_nav_dock.clear_place_filter(emit_signal=False)
        self._update_scope_bar_visibility()
        self.rebuild_and_run_query()
        if focus_mid is not None and hasattr(self, "grid_view"):
            self.grid_view.select_media_id(focus_mid)
            QTimer.singleShot(0, lambda m=focus_mid: self.grid_view.select_media_id(m))

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
        was_explicit = self._search_scope_bar_explicitly_opened
        self._search_scope_bar_explicitly_opened = False
        # Determine whether any search text or filter facet was active before clearing
        # so dismissing the Search by bar never leaves an invisible filter narrowing results.
        state = self.left_nav_dock.current_filter_state()
        m_type = str(state.get("media_type") or "").lower().strip()
        o_filt = str(state.get("orientation_filter") or "").lower().strip()
        had_active_filters = bool(
            self._search_text
            or getattr(self, "_favourites_only", False)
            or (m_type and m_type not in ("all", "none"))
            or (o_filt and o_filt not in ("all", "none"))
            or state.get("person_ids")
            or state.get("include_tag_ids")
            or state.get("exclude_tag_ids")
            or state.get("untagged_only")
            or state.get("ratings")
            or state.get("flagged_only")
            or state.get("flag_filter") is not None
            or state.get("edited_only")
            or state.get("edited_filter") is not None
            or bool(state.get("dates"))
            or state.get("date_year") is not None
            or state.get("date_month") is not None
            or state.get("date_day") is not None
            or self._active_date_range is not None
            or self._active_bbox is not None
            or self._active_map_media_ids is not None
            or self._active_map_location_label is not None
            or getattr(self, "_active_similar_seed_id", None) is not None
            or getattr(self, "_active_similar_order", None) is not None
            or bool(state.get("locations"))
            or state.get("location_country") is not None
            or state.get("location_state") is not None
            or state.get("location_city") is not None
            or bool(state.get("camera_models"))
            or state.get("camera_model") is not None
            or bool(state.get("active_subject_keys"))
            or bool(getattr(self, "_active_subject_label", ""))
            or (getattr(self, "_active_subject_ids", None) is not None)
            or getattr(self, "_active_special_people_filter", None) is not None
        )
        self._search_text = ""
        self._favourites_only = False
        self._active_date_range = None
        self._active_date_label = None
        self._active_bbox = None
        self._active_map_media_ids = None
        self._active_map_location_label = None
        self._active_similar_order = None
        self._active_similar_scored = None
        self._active_similar_label = None
        self._active_similar_seed_id = None
        self._active_similar_seed_name = ""
        self._active_subject_ids = None
        self._active_subject_label = ""
        self._active_special_people_filter = None
        self._active_special_people_label = None

        self.scope_bar.set_pinned(False)
        self.scope_bar.set_similarity_controls(False)
        self.scope_bar.txt_search.blockSignals(True)
        self.scope_bar.clear_search_input()
        self.scope_bar.txt_search.blockSignals(False)
        if had_active_filters:
            self.left_nav_dock.clear_search_filter_facets(emit_signal=False)
            if hasattr(self.ribbon, "btn_find_edited"):
                self.ribbon.btn_find_edited.blockSignals(True)
                self.ribbon.btn_find_edited.setChecked(False)
                self.ribbon._current_edited_filter = None
                self.ribbon.btn_find_edited.setToolTip("Filter by edited items")
                self.ribbon.btn_find_edited.blockSignals(False)
            if hasattr(self.ribbon, "set_orientation_filter_ui"):
                self.ribbon.set_orientation_filter_ui("all")
            self._update_scope_bar_visibility()
            self.rebuild_and_run_query()
        else:
            if was_explicit:
                self._update_scope_bar_visibility()
            self.scope_bar.hide()

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
        ws = getattr(self, "_active_create_workspace", None)
        if ws is not None:
            ws.undo()
            self._update_undo_redo_enabled()
            self._sync_create_ws_ribbon_states()
            return
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
        ws = getattr(self, "_active_create_workspace", None)
        if ws is not None:
            ws.redo()
            self._update_undo_redo_enabled()
            self._sync_create_ws_ribbon_states()
            return
        if self.central_stack.currentWidget() == self.media_viewer:
            if self.media_viewer.undo_stack.canRedo():
                self.media_viewer.undo_stack.redo()
            elif self.undo_stack.canRedo():
                self.undo_stack.redo()
        else:
            self.undo_stack.redo()

    def _update_undo_redo_enabled(self) -> None:
        try:
            ws = getattr(self, "_active_create_workspace", None)
            if ws is not None:
                self.ribbon.set_undo_enabled(bool(ws.can_undo()))
                self.ribbon.set_redo_enabled(bool(ws.can_redo()))
                return
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
        if getattr(self, "_active_create_workspace", None) is not None:
            self.close_create_workspace()
            return
        if self._nav_index > 0:
            self._nav_index -= 1
            state = self._nav_history[self._nav_index]
            self._apply_nav_state(state)

    def navigate_history_forward(self) -> None:
        """Navigates to the next state in history (browser/explorer style)."""
        if getattr(self, "_active_create_workspace", None) is not None:
            if not self.close_create_workspace():
                return
        if 0 <= self._nav_index < len(self._nav_history) - 1:
            self._nav_index += 1
            state = self._nav_history[self._nav_index]
            self._apply_nav_state(state)

    def _apply_nav_state(self, state: dict[str, Any]) -> None:
        """Applies a view/filter navigation state restored from the history stack."""
        self._is_navigating_history = True
        try:
            view = state.get("view", "grid")
            if (
                hasattr(self, "batch_people_view")
                and self.central_stack.currentWidget() == self.batch_people_view
                and view not in ("batch_people", "viewer")
            ):
                self._restore_docks_after_batch_people()
                if getattr(self, "_batch_people_modified", False):
                    self._batch_people_modified = False
                    self.refresh_people_navigation_and_autocomplete()
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
            elif view == "albums_view":
                if hasattr(self, "media_viewer") and self.central_stack.currentWidget() == self.media_viewer:
                    self.show_grid_view()
                self._active_map_media_ids = None
                self.grid_view.set_person_header(None)
                self.card_delegate.set_face_crop_person_id(None)
                if hasattr(self, "albums_view"):
                    sec = state.get("section", "all")
                    label = "My Albums" if sec == "my" else ("Smart Albums" if sec == "smart" else "Albums")
                    self.central_stack.setCurrentWidget(self.albums_view)
                    self.ribbon.set_active_mode("albums")
                    self.albums_view.set_section_filter(sec)
                    self.albums_view.refresh()
                    self.scope_bar.set_scope_label(label)
                    self.ribbon.set_scope_label(label)
            elif view == "unnamed_person":
                if hasattr(self, "media_viewer") and self.central_stack.currentWidget() == self.media_viewer:
                    self.show_grid_view()
                face_ids = state.get("face_ids", [])
                thumb_blob = state.get("thumb_blob")
                mids = state.get("media_ids", [])
                self._active_map_media_ids = set(mids) if mids else None
                self._active_map_location_label = None
                self._active_special_people_filter = "unnamed_person"
                self._active_special_people_label = "Unnamed person"
                self.central_stack.setCurrentWidget(self.grid_view)
                self.ribbon.set_active_mode("grid")
                self.person_header_bar.load_unnamed_person(face_ids, thumb_blob, photo_count=len(mids))
                self.grid_view.person_header_bar.show()
                self.grid_view._update_top_bars_geometry()
                self.scope_bar.set_scope_label("People to Name")
                self.ribbon.set_scope_label("People")
                self._update_scope_bar_visibility()
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

    def _on_dates_section_requested(self) -> None:
        """Shows the main grid arranged by Date taken when 'Date taken' is clicked in the Nav bar."""
        self.show_grid_view()
        if self._group_by not in ("year", "month", "day", "date"):
            self.set_group_by("month")
        elif self._sort_by != "date":
            self.set_sort_by("date")

    def _on_nav_filter_changed(self) -> None:
        self._active_bbox = None
        if not getattr(self, "_preserving_subject_on_nav_change", False):
            self._active_subject_ids = None
            self._active_subject_label = ""
        if not getattr(self, "_preserving_favourites_on_nav_change", False):
            self._favourites_only = False
        if not self._is_navigating_history:
            if not getattr(self, "_preserving_similar_on_nav_change", False):
                self._active_map_media_ids = None
                self._active_similar_order = None
                self._active_similar_scored = None
                self._active_similar_label = None
                self._active_similar_seed_id = None
                self._active_similar_seed_name = ""
                if hasattr(self, "scope_bar"):
                    self.scope_bar.set_similarity_controls(False)
            self._active_map_location_label = None
            self._active_special_people_filter = None
            self._active_special_people_label = None
            if not getattr(self, "_setting_date_preset", False):
                self._active_date_range = None
                self._active_date_label = None
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
            self._update_status_counts()
            self._push_nav_state({"view": "people_directory"})

    def open_albums_view(self, section: str = "all") -> None:
        """Opens the visual Albums & Smart Albums directory view (all, my, or smart)."""
        if hasattr(self, "albums_view"):
            sec = section if section in ("all", "my", "smart") else "all"
            label = "My Albums" if sec == "my" else ("Smart Albums" if sec == "smart" else "Albums")
            self._restore_docks_after_batch_people()
            self._active_map_media_ids = None
            self.grid_view.set_person_header(None)
            self.card_delegate.set_face_crop_person_id(None)
            self.central_stack.setCurrentWidget(self.albums_view)
            self.ribbon.set_active_mode("albums")
            self.albums_view.set_section_filter(sec)
            self.albums_view.refresh()
            self.scope_bar.set_scope_label(label)
            self.ribbon.set_scope_label(label)
            self._update_status_counts()
            if not self._is_navigating_history:
                self._push_nav_state({"view": "albums_view", "section": sec})

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
        if len(ids) == 1:
            members = self.catalog_repo.get_stack_member_ids(int(ids[0]))
            if len(members) > 1:
                ids = members
        all_grid_ids = [
            int(r.id) for r in getattr(self.catalog_model, "_records", []) if getattr(r, "id", None) is not None
        ]
        self.left_nav_dock.setVisible(False)
        self.right_tag_dock.setVisible(False)
        if hasattr(self, "bottom_dock") and getattr(self, "_active_create_workspace", None) is None:
            self.bottom_dock.hide()
        is_dark_theme = (getattr(self, "current_theme", "wlpg_aero") == "dark")
        self.compare_view.bind_repositories(self.catalog_repo, self.thumb_repo)
        self.compare_view.set_dark_mode(is_dark_theme)
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
            self.ribbon.btn_cmp_dark_bg.blockSignals(True)
            self.ribbon.btn_cmp_dark_bg.setChecked(is_dark_theme)
            self.ribbon.btn_cmp_dark_bg.blockSignals(False)
        if hasattr(self.ribbon, "btn_cmp_tag_pane"):
            self.ribbon.btn_cmp_tag_pane.blockSignals(True)
            self.ribbon.btn_cmp_tag_pane.setChecked(False)
            self.ribbon.btn_cmp_tag_pane.blockSignals(False)
        self._on_compare_focused_media_changed(self.compare_view.focused_media_id())
        self.compare_view.setFocus()

    def _on_compare_closed(self, winner_id: Any = None) -> None:
        if getattr(self, "_compare_return_to_duplicates", False) and hasattr(self, "duplicate_finder_view"):
            self._compare_return_to_duplicates = False
            self.left_nav_dock.setVisible(False)
            self.right_tag_dock.setVisible(False)
            self.central_stack.setCurrentWidget(self.duplicate_finder_view)
            self.ribbon.set_active_mode("duplicates")
            self.setWindowTitle("Find Duplicates & Bursts - Essentials Photo Gallery")
            return
        self.left_nav_dock.setVisible(True)
        self.right_tag_dock.setVisible(True)
        self.show_grid_view()
        if winner_id is not None and hasattr(self.grid_view, "select_media_id"):
            try:
                self.grid_view.select_media_id(int(winner_id))
            except Exception:
                pass

    def _on_duplicates_zoom_changed(self, card_width: int) -> None:
        if not hasattr(self, "slider_status_zoom"):
            return
        t = max(0.0, min(1.0, (float(card_width) - 220.0) / (650.0 - 220.0)))
        status_val = int(round(72 + t * (320 - 72)))
        self.slider_status_zoom.blockSignals(True)
        self.slider_status_zoom.setValue(status_val)
        self.slider_status_zoom.blockSignals(False)

    def _on_duplicates_media_selected(self, media_id: int) -> None:
        if not hasattr(self, "right_tag_dock") or self.central_stack.currentWidget() != self.duplicate_finder_view:
            return
        try:
            mid = int(media_id)
            summary = self.catalog_repo.get_selection_summary([mid])
            self.right_tag_dock.bind_selection([mid], summary)
        except Exception:
            pass

    def _on_compare_focused_media_changed(self, media_id: Any) -> None:
        if media_id is None or not hasattr(self, "right_tag_dock"):
            return
        if self.central_stack.currentWidget() != self.compare_view:
            return
        try:
            mid = int(media_id)
            summary = self.catalog_repo.get_selection_summary([mid])
            self.right_tag_dock.bind_selection([mid], summary)
        except Exception:
            pass

    def _on_duplicates_compare_group_requested(self, media_ids: list[int]) -> None:
        if not media_ids:
            return
        self._compare_return_to_duplicates = True
        self.open_compare_view(list(media_ids))

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
        if hasattr(self, "scope_bar"):
            self.scope_bar.clear_search_input()

        conn = self.engine.catalog_conn()
        placeholders = ",".join("?" for _ in face_ids)
        rows = conn.execute(
            f"SELECT DISTINCT media_id FROM media_people WHERE id IN ({placeholders});",
            face_ids,
        ).fetchall()
        mids = [r[0] for r in rows]

        self._active_map_media_ids = set(mids)
        self._active_map_location_label = None
        self._active_special_people_filter = "unnamed_person"
        self._active_special_people_label = "Unnamed person"
        self.central_stack.setCurrentWidget(self.grid_view)
        self.ribbon.set_active_mode("grid")
        self.person_header_bar.load_unnamed_person(face_ids, thumb_blob, photo_count=len(mids))
        self.grid_view.person_header_bar.show()
        self.grid_view._update_top_bars_geometry()
        self._update_scope_bar_visibility()
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
                signals = _FaceCropPreloadSignals(self)
                signals.crop_ready.connect(self._on_face_crop_ready)
                signals.progress.connect(self._on_face_crop_progress)
                signals.finished.connect(self._on_face_crop_finished)
                worker = _FaceCropPreloadWorker(self.engine, pid, signals, self._face_crop_cancel)
                self._retain_pool_worker(worker, signals)
                signals.finished.connect(
                    lambda *_args, w=worker, s=signals: self._release_pool_worker(w, s)
                )
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
            signals = _FaceCropPreloadSignals(self)
            signals.crop_ready.connect(self._on_face_crop_ready)
            signals.progress.connect(self._on_face_crop_progress)
            signals.finished.connect(self._on_face_crop_finished)
            worker = _FaceCropPreloadWorker(
                self.engine, int(active_crop_pid), signals, self._face_crop_cancel
            )
            self._retain_pool_worker(worker, signals)
            signals.finished.connect(
                lambda *_args, w=worker, s=signals: self._release_pool_worker(w, s)
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
        self.grid_view.set_person_header(None)
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
        self._active_date_range = None
        self._active_date_label = None
        self.left_nav_dock.select_date(y, m, d)

    def _clear_date_filter(self) -> None:
        self._active_date_range = None
        self._active_date_label = None
        self.left_nav_dock.select_date(None, None, None)
        self._update_scope_bar_visibility()
        self.rebuild_and_run_query()

    def _on_date_preset_selected(self, code: str) -> None:
        """Filters media by a relative date preset ('today', 'yesterday', 'this_week', etc.) with a clear chip label."""
        import calendar
        import datetime

        if self.central_stack.currentWidget() != self.grid_view:
            self.show_grid_view()

        now = datetime.datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if code == "today":
            start_dt = today_start
            end_dt = today_start + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)
            label = "Today"
        elif code == "yesterday":
            start_dt = today_start - datetime.timedelta(days=1)
            end_dt = today_start - datetime.timedelta(seconds=1)
            label = "Yesterday"
        elif code == "this_week":
            start_dt = today_start - datetime.timedelta(days=today_start.weekday())
            end_dt = start_dt + datetime.timedelta(days=7) - datetime.timedelta(seconds=1)
            label = "This week"
        elif code == "last_week":
            start_dt = today_start - datetime.timedelta(days=today_start.weekday() + 7)
            end_dt = start_dt + datetime.timedelta(days=7) - datetime.timedelta(seconds=1)
            label = "Last week"
        elif code == "this_month":
            last_day = calendar.monthrange(now.year, now.month)[1]
            start_dt = today_start.replace(day=1)
            end_dt = today_start.replace(day=last_day, hour=23, minute=59, second=59)
            label = "This month"
        elif code == "last_month":
            month = now.month - 1 if now.month > 1 else 12
            year = now.year if now.month > 1 else now.year - 1
            last_day = calendar.monthrange(year, month)[1]
            start_dt = datetime.datetime(year, month, 1, 0, 0, 0)
            end_dt = datetime.datetime(year, month, last_day, 23, 59, 59)
            label = "Last month"
        elif code == "this_year":
            start_dt = datetime.datetime(now.year, 1, 1, 0, 0, 0)
            end_dt = datetime.datetime(now.year, 12, 31, 23, 59, 59)
            label = "This year"
        elif code == "last_year":
            start_dt = datetime.datetime(now.year - 1, 1, 1, 0, 0, 0)
            end_dt = datetime.datetime(now.year - 1, 12, 31, 23, 59, 59)
            label = "Last year"
        else:
            return

        self._active_special_people_filter = None
        self._active_special_people_label = None
        self._active_map_media_ids = None
        self._setting_date_preset = True
        try:
            self.left_nav_dock.select_date(None, None, None)
            self._active_date_range = (int(start_dt.timestamp()), int(end_dt.timestamp()))
            self._active_date_label = label
        finally:
            self._setting_date_preset = False
        self._update_scope_bar_visibility()
        self.rebuild_and_run_query()

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
        self._active_bbox = None
        self._active_map_media_ids = None
        self._active_map_location_label = None
        self._active_special_people_filter = None
        self._active_special_people_label = None
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
        if self._active_date_range is not None:
            from dataclasses import replace as _dc_replace

            spec = _dc_replace(
                spec,
                date_from=self._active_date_range[0],
                date_to=self._active_date_range[1],
            )
        sim_order = getattr(self, "_active_similar_order", None)
        subj_order = getattr(self, "_active_subject_ids", None)
        if sim_order:
            # Find Similar searches catalog-wide; don't intersect with the current folder filter
            spec = FilterQuerySpec(
                media_ids=tuple(sim_order),
                search_text=spec.search_text,
            )
        elif subj_order is not None:
            spec = FilterQuerySpec(
                media_ids=tuple(subj_order) if subj_order else (-1,),
                arrange_by=spec.arrange_by,
                reverse_sort=spec.reverse_sort,
                group_by="none",
            )
        exp_stack = tuple(sorted(getattr(self, "_expanded_stack_ids", set())))
        fav_only = bool(getattr(self, "_favourites_only", False))
        if exp_stack or fav_only:
            from dataclasses import replace as _dc_replace

            spec = _dc_replace(spec, expanded_stack_ids=exp_stack, favourites_only=fav_only)
        records = self.catalog_repo.query_media_records(spec)
        if spec.media_type == "motion" and "PYTEST_CURRENT_TEST" not in os.environ:
            self._maybe_start_motion_photo_backfill(folder_ids=spec.folder_ids)

        # Fuse semantic text matches with FTS5 keyword matches when MobileCLIP-S1 is installed
        if (
            not sim_order
            and subj_order is None
            and self._search_text
            and self._search_text.strip()
            and getattr(self, "semantic_engine", None) is not None
            and getattr(self, "semantic_repo", None) is not None
            and self.semantic_engine.is_available()
        ):
            try:
                from dataclasses import replace as _dc_replace
                from src.core.semantic_engine import fuse_semantic_and_fts_scores

                qvec = self.semantic_engine.encode_text(self._search_text.strip())
                if qvec is not None:
                    sem_hits = self.semantic_engine.query_by_vector(
                        self.semantic_repo,
                        qvec,
                        threshold=0.22,
                        limit=400,
                        scope_ids=self._visible_non_private_media_ids(),
                    )
                    if sem_hits:
                        fts_ids = [int(r.id) for r in records]
                        fused_ids = fuse_semantic_and_fts_scores(fts_ids, sem_hits)
                        fts_set = set(fts_ids)
                        missing_ids = [mid for mid in fused_ids if mid not in fts_set]
                        if spec.media_ids is not None:
                            allowed_ids = set(spec.media_ids)
                            missing_ids = [mid for mid in missing_ids if mid in allowed_ids]
                        if missing_ids:
                            scoped_extra = _dc_replace(
                                spec,
                                search_text=None,
                                media_ids=tuple(missing_ids),
                                group_by="none",
                            )
                            extra_recs = self.catalog_repo.query_media_records(scoped_extra)
                            rec_by_id = {int(r.id): r for r in (records + extra_recs)}
                            records = [rec_by_id[mid] for mid in fused_ids if mid in rec_by_id]
            except Exception:
                pass

        if sim_order:
            rank_map = {int(mid): idx for idx, mid in enumerate(sim_order)}
            records.sort(key=lambda r: rank_map.get(int(r.id), len(rank_map)))
            headers_enabled = False
        elif subj_order:
            rank_map = {int(mid): idx for idx, mid in enumerate(subj_order)}
            records.sort(key=lambda r: rank_map.get(int(r.id), len(rank_map)))
            headers_enabled = False
        self.catalog_model.set_records(records, group_headers_enabled=headers_enabled)
        if hasattr(self, "bottom_dock") and self.bottom_dock is not None:
            self.bottom_dock.refresh_items()

        if hasattr(self, "scope_bar") and self.scope_bar is not None:
            self._update_scope_bar_visibility()

        # Update empty state overlay
        is_empty = (len(records) == 0)
        is_search = bool(self._search_text and self._search_text.strip())
        is_scoped = bool(
            spec.folder_id
            or spec.folder_ids
            or (hasattr(self, "left_nav_dock") and self.left_nav_dock.selected_folder_name() is not None)
            or spec.include_tag_ids
            or spec.exclude_tag_ids
            or spec.untagged_only
            or spec.person_ids
            or spec.date_year
            or spec.dates
            or spec.ratings
            or spec.flagged_only
            or (spec.flag_filter is not None)
            or spec.edited_only
            or (spec.edited_filter is not None)
            or spec.favourites_only
            or (spec.media_type != "all")
            or (spec.orientation_filter != "all")
            or spec.bbox
            or spec.location_country
            or spec.location_state
            or spec.location_city
            or spec.locations
            or spec.camera_model
            or spec.camera_models
            or (subj_order is not None)
        )
        is_library_empty = False
        if is_empty and not is_search and not is_scoped:
            try:
                is_library_empty = self.catalog_repo.get_total_media_count() == 0
            except Exception:
                is_library_empty = False
        self.grid_view.set_empty_state(
            is_empty=is_empty,
            is_search=is_search,
            is_scoped=is_scoped,
            is_library_empty=is_library_empty,
        )
        if hasattr(self, "welcome_overlay") and self.welcome_overlay is not None:
            if is_library_empty and self.central_stack.currentWidget() in (self.grid_view, self.welcome_overlay):
                self.central_stack.setCurrentWidget(self.welcome_overlay)
            elif not is_library_empty and self.central_stack.currentWidget() == self.welcome_overlay:
                self.central_stack.setCurrentWidget(self.grid_view)

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
        Two-tier viewport thumbnail scheduler:
        1. Cancels out-of-view pending disk-decode workers.
        2. Prefetches visible cached WebP blobs in ONE batched SQLite query on `thumb_cache_pool`
           (zero head-of-line blocking behind slow RAW/video decodes).
        3. Prefetches visible face-crop thumbnails off-thread when in Person Face-Crop mode.
        4. Dispatches only confirmed cache misses into `thumbnail_pool` ordered from viewport center.
        """
        if self._is_shutting_down or not media_ids:
            return

        from src.workers.thumb_scheduler import (
            BatchFaceCropWorker,
            BatchThumbCacheWorker,
            order_ids_by_viewport_center,
        )

        ordered = order_ids_by_viewport_center(media_ids)
        self._pending_visible_media_ids = ordered
        visible_set = set(ordered)
        for mid, flag in list(self._active_thumb_flags.items()):
            if mid not in visible_set:
                flag.set()
                self._active_thumb_flags.pop(mid, None)

        # Async batched face-crop prefetch (eliminates synchronous SQLite reads in MediaCardDelegate.paint)
        if hasattr(self, "card_delegate") and self.card_delegate.face_crop_person_id is not None:
            pid = int(self.card_delegate.face_crop_person_id)
            missing_crops = [
                mid for mid in ordered
                if (int(mid), pid) not in self.card_delegate._face_crop_cache
            ]
            if missing_crops:
                self._face_crop_token += 1
                fc_worker = BatchFaceCropWorker(
                    self.engine, self._face_crop_token, pid, missing_crops[:256]
                )
                self._retain_pool_worker(fc_worker)
                fc_worker.signals.crops_ready.connect(self._on_face_crop_batch_ready)
                fc_worker.signals.crops_ready.connect(
                    lambda *_args, w=fc_worker: self._release_pool_worker(w)
                )
                self._active_face_crop_worker = fc_worker
                self.thumb_cache_pool.start(fc_worker)

        failed = getattr(self, "_failed_thumb_ids", set())
        uncached = [
            mid for mid in ordered
            if mid not in self.pixmap_cache and mid not in self._active_thumb_flags and mid not in failed
        ]
        if not uncached:
            self._pending_disk_decode_ids = []
            return

        check_batch = uncached[:256]
        self._ensure_media_path_lookup(check_batch)
        expected_mtimes: dict[int, int] = {}
        for mid in check_batch:
            meta = self._media_meta_lookup.get(mid)
            if meta is not None:
                expected_mtimes[mid] = int(meta[1])

        self._thumb_cache_token += 1
        cache_worker = BatchThumbCacheWorker(
            self.engine,
            self._thumb_cache_token,
            check_batch,
            expected_mtimes=expected_mtimes,
        )
        self._retain_pool_worker(cache_worker)
        cache_worker.signals.batch_ready.connect(self._on_thumb_cache_batch_ready)
        cache_worker.signals.batch_ready.connect(
            lambda *_args, w=cache_worker: self._release_pool_worker(w)
        )
        self._active_thumb_cache_worker = cache_worker
        self.thumb_cache_pool.start(cache_worker)

    @Slot(int, int, list)
    def _on_face_crop_batch_ready(
        self, token: int, person_id: int, decoded_items: list[tuple[int, QImage]]
    ) -> None:
        if self._is_shutting_down or token != self._face_crop_token:
            return
        if not hasattr(self, "card_delegate") or self.card_delegate.face_crop_person_id != person_id:
            return
        for mid, qimg in decoded_items:
            if not qimg.isNull():
                self.card_delegate._face_crop_cache[(int(mid), int(person_id))] = QPixmap.fromImage(qimg)
                self.catalog_model.notify_pixmap_ready(int(mid))

    @Slot(int, list, list)
    def _on_thumb_cache_batch_ready(
        self,
        token: int,
        decoded_items: list[tuple[int, QImage]],
        missing_ids: list[int],
    ) -> None:
        if self._is_shutting_down:
            return
        for mid, qimg in decoded_items:
            if not qimg.isNull():
                self.pixmap_cache.put(int(mid), QPixmap.fromImage(qimg))
                self.catalog_model.notify_pixmap_ready(int(mid))
        if token != self._thumb_cache_token:
            return
        failed = getattr(self, "_failed_thumb_ids", set())
        self._pending_disk_decode_ids = [
            int(mid) for mid in missing_ids
            if mid not in self.pixmap_cache and mid not in failed
        ]
        self._pump_visible_thumbnail_queue()

    def _pump_visible_thumbnail_queue(self) -> None:
        """
        Dispatches confirmed cache-miss visible thumbnails into `thumbnail_pool` using a
        compact viewport-center priority window (max 12 in-flight decodes) so scrolling
        re-prioritizes instantly without flooding QThreadPool with hundreds of queued runnables.
        """
        if self._is_shutting_down:
            return
        source_queue = (
            self._pending_disk_decode_ids
            if getattr(self, "_pending_disk_decode_ids", None)
            else getattr(self, "_pending_visible_media_ids", [])
        )
        if not source_queue:
            return

        failed = getattr(self, "_failed_thumb_ids", set())
        uncached = [
            mid for mid in source_queue
            if mid not in self.pixmap_cache and mid not in self._active_thumb_flags and mid not in failed
        ]
        if not uncached:
            return

        max_in_flight = 12
        available_slots = max(0, max_in_flight - len(self._active_thumb_flags))
        if available_slots <= 0:
            return

        batch_ids = uncached[:available_slots]
        self._ensure_media_path_lookup(batch_ids)
        coalescer = getattr(self, "thumb_write_coalescer", None)
        using_miss_queue = bool(getattr(self, "_pending_disk_decode_ids", None))
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
                write_coalescer=coalescer,
                skip_cache_lookup=using_miss_queue,
            )
            self._retain_pool_worker(worker)
            worker.signals.thumbnail_ready.connect(self.on_thumbnail_decoded)
            worker.signals.error.connect(self.on_thumbnail_decode_error)
            worker.signals.finished.connect(
                lambda mid_done, w=worker: self._on_thumb_decode_finished(mid_done, w)
            )
            self.thumbnail_pool.start(worker)

    def _retain_pool_worker(self, worker: QRunnable, signals: QObject | None = None) -> None:
        """Prevents QThreadPool from deleting a worker/signals on a worker thread before slot delivery."""
        worker.setAutoDelete(False)
        sig = signals if signals is not None else getattr(worker, "signals", None)
        if sig is not None and sig.parent() is None:
            sig.setParent(self)
        self._inflight_workers.add(worker)

    def _release_pool_worker(self, worker: QRunnable, signals: QObject | None = None) -> None:
        """Drops the main-thread worker reference and schedules signal QObject deletion."""
        if hasattr(self, "_inflight_workers"):
            self._inflight_workers.discard(worker)
        sig = signals if signals is not None else getattr(worker, "signals", None)
        if sig is not None:
            try:
                sig.deleteLater()
            except RuntimeError:
                pass

    def _on_thumb_decode_finished(self, media_id: int, worker: QRunnable) -> None:
        """Retires a finished ThumbnailDecodeWorker safely on the GUI thread."""
        if self._active_thumb_flags.get(media_id) is getattr(worker, "cancel_flag", None):
            self._active_thumb_flags.pop(media_id, None)
        self._release_pool_worker(worker)

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

    def _check_stale_scan_crash_trace(self) -> None:
        """
        Detects if a previous run died mid-decode inside a native C/C++ codec
        (recorded in active_scan_trace.json) and offers to reveal or skip the culprit file.
        """
        if "PYTEST_CURRENT_TEST" in os.environ:
            return
        from src.utils.scan_logger import clear_stale_crash_trace, read_stale_crash_trace

        trace = read_stale_crash_trace()
        if not trace:
            return
        culprit = str(trace.get("file_path") or "")
        stage = str(trace.get("stage") or "Scanner")
        clear_stale_crash_trace()
        if not culprit:
            return

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Scan Recovery")
        box.setText(f"The previous session ended unexpectedly while decoding:\n\n{culprit}")
        box.setInformativeText(f"Stage: {stage}\n\nYou can reveal this file in Explorer or skip it during future scans.")
        btn_skip = box.addButton("Skip This File", QMessageBox.AcceptRole)
        btn_reveal = box.addButton("Reveal in Explorer", QMessageBox.ActionRole)
        box.addButton("Dismiss", QMessageBox.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked == btn_reveal:
            try:
                import subprocess
                subprocess.Popen(["explorer", "/select,", str(Path(culprit))])
            except Exception:
                pass
        elif clicked == btn_skip:
            try:
                mid = self.catalog_repo.get_media_id_by_file_path(culprit)
                if mid is not None:
                    self.catalog_repo.mark_face_scan_failed([mid])
                    self._failed_thumb_ids.add(int(mid))
            except Exception:
                pass

    def _maybe_start_motion_photo_backfill(
        self, folder_ids: Sequence[int] | None = None
    ) -> None:
        """Runs a throttled background pass to detect Motion/Live photos on pre-existing catalog rows."""
        if getattr(self, "_is_shutting_down", False) or getattr(self, "_motion_backfill_active", False):
            return
        self._motion_backfill_active = True
        signals = _MotionPhotoBackfillSignals(self)
        worker = _MotionPhotoBackfillWorker(
            self.engine,
            signals,
            folder_ids=folder_ids,
            batch_limit=1500,
        )
        self._retain_pool_worker(worker, signals)

        def _on_motion_backfill_done(found_count: int) -> None:
            self._motion_backfill_active = False
            if found_count > 0 and not getattr(self, "_is_shutting_down", False):
                st = self.left_nav_dock.current_filter_state()
                if st.get("media_type") == "motion":
                    self.rebuild_and_run_query()
                    self.show_status_message(
                        f"✓ Detected {found_count:,} motion photo(s)", 3500
                    )

        signals.finished.connect(_on_motion_backfill_done)
        signals.finished.connect(
            lambda *_args, w=worker, s=signals: self._release_pool_worker(w, s)
        )
        self.scanner_pool.start(worker)

    def _start_async_avatar_backfill(self) -> None:
        """Launches background QRunnable to generate avatar crops for people without blocking startup."""
        if self._is_shutting_down:
            return
        signals = _AvatarBackfillSignals(self)
        signals.finished.connect(self._on_avatar_backfill_completed)
        worker = _AvatarBackfillWorker(self.engine, signals)
        self._retain_pool_worker(worker, signals)
        signals.finished.connect(
            lambda *_args, w=worker, s=signals: self._release_pool_worker(w, s)
        )
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

        # Debounced Effects ribbon preview update when resting on a photo in Thumbnail View
        if getattr(self, "_grid_effects_preview_timer", None) is None:
            self._grid_effects_preview_timer = QTimer(self)
            self._grid_effects_preview_timer.setSingleShot(True)
            self._grid_effects_preview_timer.setInterval(350)
            self._grid_effects_preview_timer.timeout.connect(
                self._refresh_effects_gallery_from_grid_selection
            )
        if self._lead_selected_id is not None and self.central_stack.currentWidget() == self.grid_view:
            self._grid_effects_preview_timer.start()
        else:
            self._grid_effects_preview_timer.stop()

    def _on_ribbon_tab_changed(self, _index: int) -> None:
        if (
            self.ribbon.tab_widget.currentWidget() == self.ribbon.tab_edit
            and self.central_stack.currentWidget() == self.grid_view
            and getattr(self, "_lead_selected_id", None) is not None
            and getattr(self, "_grid_effects_preview_timer", None) is not None
        ):
            self._grid_effects_preview_timer.start()

    def _refresh_effects_gallery_from_grid_selection(self) -> None:
        if (
            not hasattr(self.ribbon, "effects_gallery")
            or self.central_stack.currentWidget() != self.grid_view
            or self.ribbon.tab_widget.currentWidget() != self.ribbon.tab_edit
        ):
            return
        mid = getattr(self, "_lead_selected_id", None)
        if mid is None:
            return
        pix = self.pixmap_cache.get(int(mid))
        if pix is None or pix.isNull():
            try:
                blob = self.thumb_repo.get_webp_blob(int(mid))
                if blob:
                    p = QPixmap()
                    if p.loadFromData(blob):
                        pix = p
            except Exception:
                pix = None
        if pix is None or pix.isNull():
            return
        qimg = pix.toImage().convertToFormat(QImage.Format_RGB888)
        w, h = qimg.width(), qimg.height()
        if w <= 0 or h <= 0:
            return
        import numpy as np
        from src.ui.media_viewer import SingleMediaViewer

        ptr = qimg.constBits()
        arr = np.frombuffer(ptr, dtype=np.uint8, count=h * qimg.bytesPerLine()).reshape((h, qimg.bytesPerLine()))
        rgb = arr[:, : w * 3].reshape((h, w, 3)).copy()
        pixmaps = SingleMediaViewer.build_effect_preview_pixmaps_from_rgb(rgb, 53, 64)
        if pixmaps:
            self.ribbon.effects_gallery.update_live_thumbnails(pixmaps)

    def _update_status_counts(self) -> None:
        current = self.central_stack.currentWidget() if hasattr(self, "central_stack") else None
        if hasattr(self, "albums_view") and current == self.albums_view:
            sec = self.albums_view.section_filter()
            if sec == "my":
                cnt = len(getattr(self.albums_view, "_static_albums", []))
                word = "album" if cnt == 1 else "albums"
            elif sec == "smart":
                cnt = len(getattr(self.albums_view, "_smart_albums", []))
                word = "smart album" if cnt == 1 else "smart albums"
            else:
                cnt = len(getattr(self.albums_view, "_static_albums", [])) + len(
                    getattr(self.albums_view, "_smart_albums", [])
                )
                word = "album" if cnt == 1 else "albums"
            self.lbl_status_counts.setText(f"{cnt:,} {word}")
            self.right_tag_dock.set_total_view_count(cnt)
            return
        if hasattr(self, "people_directory_view") and current == self.people_directory_view:
            cnt = len(getattr(self.people_directory_view, "_all_named_people", []))
            word = "person" if cnt == 1 else "people"
            self.lbl_status_counts.setText(f"{cnt:,} {word}")
            self.right_tag_dock.set_total_view_count(cnt)
            return

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
        if getattr(self, "_active_create_workspace", None) is not None:
            return
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
        if hasattr(self, "compare_view") and self.central_stack.currentWidget() == self.compare_view:
            mid = self.compare_view.focused_media_id()
            if mid is not None:
                return [int(mid)]
        if hasattr(self, "duplicate_finder_view") and self.central_stack.currentWidget() == self.duplicate_finder_view:
            mid = self.duplicate_finder_view.selected_media_id()
            if mid is not None:
                return [int(mid)]
        return list(self._selected_ids)

    def apply_rating_to_selection(self, rating: int, *, advance: bool = False) -> None:
        target_ids = self.get_active_target_ids()
        if not target_ids:
            return
        cmd = _RatingUndoCommand(self, target_ids, rating)
        self.undo_stack.push(cmd)
        lbl = f"{rating} star{'s' if rating != 1 else ''}" if rating > 0 else "Unrated"
        self.show_status_message(f"✓ Rated {len(target_ids)} item(s): {lbl}", 3000)
        if advance and self.central_stack.currentWidget() == self.media_viewer and hasattr(self.media_viewer, "flash_cull_hud"):
            self.media_viewer.flash_cull_hud("rating", int(rating))
        if advance:
            self._advance_after_cull()

    def apply_flag_to_selection(self, flag: int, *, advance: bool = False) -> None:
        target_ids = self.get_active_target_ids()
        if not target_ids:
            return
        cmd = _FlagUndoCommand(self, target_ids, flag)
        self.undo_stack.push(cmd)
        if hasattr(self, "media_viewer") and getattr(self.media_viewer, "_update_cull_hud", None):
            self.media_viewer._update_cull_hud()
        if hasattr(self, "left_nav_dock") and hasattr(self, "catalog_repo"):
            try:
                self.left_nav_dock.refresh_albums_only(self.catalog_repo)
            except Exception:
                pass
        if hasattr(self, "right_tag_dock") and self.right_tag_dock is not None:
            try:
                self.right_tag_dock._update_flag_button(int(flag))
            except Exception:
                pass
        if flag == 1:
            lbl = "Picked"
        elif flag == -1:
            lbl = "Rejected"
        else:
            lbl = "Unflagged"
        self.show_status_message(f"✓ {lbl} {len(target_ids)} item(s)", 3000)
        if advance and self.central_stack.currentWidget() == self.media_viewer and hasattr(self.media_viewer, "flash_cull_hud"):
            self.media_viewer.flash_cull_hud("flag", int(flag))
        if advance:
            self._advance_after_cull()

    def apply_favourite_to_selection(
        self, is_fav: bool, *, target_ids: Sequence[int] | None = None, advance: bool = False
    ) -> None:
        ids = list(target_ids) if target_ids is not None else self.get_active_target_ids()
        if not ids:
            return
        cmd = _FavouriteUndoCommand(self, ids, 1 if is_fav else 0)
        self.undo_stack.push(cmd)
        if hasattr(self, "right_tag_dock") and self.right_tag_dock is not None:
            try:
                self.right_tag_dock._update_flag_button(
                    self.right_tag_dock._current_flag, bool(is_fav)
                )
            except Exception:
                pass
        self.show_status_message(
            f"{'♥ Added' if is_fav else '♡ Removed'} {len(ids)} item(s) {'to' if is_fav else 'from'} Favourites",
            3000,
        )
        if advance and self.central_stack.currentWidget() == self.media_viewer and hasattr(self.media_viewer, "flash_cull_hud"):
            self.media_viewer.flash_cull_hud("favourite", bool(is_fav))
        if advance:
            self._advance_after_cull()

    def toggle_favourite_selection(self, *, advance: bool = False) -> None:
        if not getattr(self, "favourites_enabled", True):
            return
        ids = self.get_active_target_ids()
        if not ids:
            return
        fav_map = self.catalog_repo.get_field_map_for_ids(ids, "is_favourite")
        any_unfav = any(int(fav_map.get(mid) or 0) == 0 for mid in ids)
        self.apply_favourite_to_selection(any_unfav, target_ids=ids, advance=advance)

    def _on_card_favourite_toggled(self, media_id: int, is_fav: bool) -> None:
        if not getattr(self, "favourites_enabled", True):
            return
        self.apply_favourite_to_selection(is_fav, target_ids=[int(media_id)])

    def _on_favourite_filter_selected(self, enabled: bool) -> None:
        self._favourites_only = bool(enabled) and bool(getattr(self, "favourites_enabled", True))
        if self._favourites_only:
            # Clear folder/album/flag scope so the permanent Favourites smart album shows all favourites
            self.left_nav_dock._folder_id = None
            self.left_nav_dock._album_id = None
            self.left_nav_dock._smart_album_id = None
            self.left_nav_dock._smart_album_rules = None
            self.left_nav_dock._flag_filter = None
            self.left_nav_dock._flagged_only = False
        self._update_scope_bar_visibility()
        self.rebuild_and_run_query()

    def _on_nav_smart_flag_selected(self, flag_val: object) -> None:
        if flag_val is not None:
            self._favourites_only = False
            self.left_nav_dock._folder_id = None
            self.left_nav_dock._album_id = None
            self.left_nav_dock._smart_album_id = None
            self.left_nav_dock._smart_album_rules = None
        self.left_nav_dock.set_flag_filter(flag_val)

    def _save_builtin_smart_albums_prefs(self) -> None:
        settings = QSettings("PhotoGallery64", "PhotoGallery64")
        settings.setValue("smart_show_favourites", getattr(self.left_nav_dock, "_show_smart_favourites", True))
        settings.setValue("smart_show_picked", getattr(self.left_nav_dock, "_show_smart_picked", True))
        settings.setValue("smart_show_rejected", getattr(self.left_nav_dock, "_show_smart_rejected", True))
        settings.setValue("smart_show_unflagged", getattr(self.left_nav_dock, "_show_smart_unflagged", True))

    def set_favourites_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self.favourites_enabled = enabled
        if hasattr(self, "grid_view") and self.grid_view is not None and self.grid_view.card_delegate is not None:
            self.grid_view.card_delegate.favourites_enabled = enabled
            self.grid_view.viewport().update()
        if hasattr(self, "media_viewer") and self.media_viewer is not None:
            self.media_viewer._favourites_enabled = enabled
            try:
                self.media_viewer._update_cull_hud()
            except Exception:
                pass
        if hasattr(self, "filmstrip_dock") and self.filmstrip_dock is not None:
            dlg = getattr(self.filmstrip_dock, "_delegate", None)
            if dlg is not None:
                dlg.favourites_enabled = enabled
            lv = getattr(self.filmstrip_dock, "list_view", None)
            if lv is not None:
                lv.viewport().update()
        if hasattr(self, "ribbon") and self.ribbon is not None:
            self.ribbon.set_favourites_enabled(enabled)
        if hasattr(self, "left_nav_dock") and self.left_nav_dock is not None:
            self.left_nav_dock.set_favourites_enabled(enabled)
        if hasattr(self, "right_tag_dock") and self.right_tag_dock is not None:
            self.right_tag_dock.set_favourites_enabled(enabled)
        if not enabled:
            needs_requery = False
            if getattr(self, "_favourites_only", False):
                self._favourites_only = False
                needs_requery = True
            if getattr(self, "_group_by", "month") == "favourites":
                self.set_group_by("month")
                needs_requery = False  # set_group_by already triggers rebuild
            if needs_requery:
                self.rebuild_and_run_query()

    def select_all_rejected(self) -> list[int]:
        rejected = self.catalog_repo.get_rejected_media_ids()
        if not rejected:
            self.show_status_message("No rejected items in catalog.", 2500)
            return []
        self.grid_view.set_selected_media_ids(rejected)
        self.show_status_message(f"Selected {len(rejected)} rejected item(s).", 3000)
        return rejected

    def delete_all_rejected(self) -> int:
        rejected = self.catalog_repo.get_rejected_media_ids()
        if not rejected:
            self.show_status_message("No rejected items to delete.", 2500)
            return 0
        reply = QMessageBox.question(
            self,
            "Delete Rejected Photos",
            f"Move {len(rejected)} rejected item(s) to the Recycle Bin?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return 0
        self.grid_view.set_selected_media_ids(rejected)
        return self.delete_selection_to_recycle_bin()

    def _maybe_prompt_picasa_import(self, root_path: str | Path) -> None:
        """Auto-detect .picasa.ini files in newly scanned/added folders and offer 1-click import."""
        if "PYTEST_CURRENT_TEST" in os.environ or getattr(self, "_is_shutting_down", False):
            return
        try:
            rp = Path(root_path)
            if not rp.is_dir():
                return
            norm = os.path.normcase(os.path.normpath(str(rp)))
            settings = QSettings("PhotoGallery64", "PhotoGallery64")
            raw_seen = settings.value("picasa_prompted_roots", [])
            seen: list[str] = list(raw_seen) if isinstance(raw_seen, (list, tuple)) else ([str(raw_seen)] if raw_seen else [])
            if norm in seen:
                return

            from src.core.picasa_importer import apply_picasa_import, scan_picasa_tree
            from src.ui.theme_tokens import get_theme_tokens

            plan = scan_picasa_tree(rp)
            if plan.ini_files == 0 or not plan.entries:
                return

            seen.append(norm)
            settings.setValue("picasa_prompted_roots", seen)

            parts: list[str] = []
            if plan.stars_count:
                parts.append(f"{plan.stars_count} starred")
            if plan.captions_count:
                parts.append(f"{plan.captions_count} captions")
            if plan.tags_count:
                parts.append(f"{plan.tags_count} tags")
            if plan.faces_count:
                parts.append(f"{plan.faces_count} face regions")
            if plan.albums:
                parts.append(f"{len(plan.albums)} album(s)")
            detail_line = ", ".join(parts) if parts else "metadata entries"

            tok = get_theme_tokens(getattr(self, "current_theme", "wlpg_aero"))
            msg = QMessageBox(self)
            msg.setWindowTitle("Google Picasa Metadata Found")
            msg.setIcon(QMessageBox.Information)
            msg.setText(f"Found Google Picasa metadata for {len(plan.entries)} photo(s) in folder '{rp.name}'.")
            msg.setInformativeText(f"Includes: {detail_line}.\n\nImport into your library now?")
            msg.setStyleSheet(
                f"QMessageBox {{ background-color: {tok.bg_surface}; color: {tok.text_primary}; }} "
                f"QMessageBox QLabel {{ color: {tok.text_primary}; background: transparent; }} "
                f"QMessageBox QPushButton {{ background-color: {tok.bg_elevated}; color: {tok.text_primary}; "
                f"border: 1px solid {tok.border_subtle}; border-radius: 4px; padding: 5px 14px; min-width: 76px; }} "
                f"QMessageBox QPushButton:hover {{ background-color: {tok.bg_hover}; border-color: {tok.accent}; }}"
            )
            btn_import = msg.addButton("Import Now", QMessageBox.AcceptRole)
            btn_custom = msg.addButton("Customize...", QMessageBox.ActionRole)
            msg.addButton("Not Now", QMessageBox.RejectRole)
            msg.exec()

            clicked = msg.clickedButton()
            if clicked == btn_import:
                from src.ui.picasa_import_dialog import mark_picasa_folders_imported

                stats = apply_picasa_import(self.catalog_repo, plan)
                mark_picasa_folders_imported({e.folder_path for e in plan.entries} | {rp})
                self.refresh_navigation_and_autocomplete()
                self.rebuild_and_run_query()
                self.show_status_message(
                    f"✓ Picasa import: {stats['matched']} matched "
                    f"({stats['stars']} ♥, {stats['tags']} tags, {stats['faces']} faces, {stats['albums']} albums)",
                    6000,
                )
            elif clicked == btn_custom:
                self.open_picasa_import_dialog(initial_dir=str(rp))
        except Exception:
            logger.debug("Picasa auto-detect check failed for %s", root_path, exc_info=True)

    def open_picasa_import_dialog(self, initial_dir: str = "") -> None:
        from src.ui.picasa_import_dialog import PicasaImportDialog

        initial = initial_dir
        if not initial:
            folders = self.catalog_repo.get_folders()
            if folders:
                initial = str(folders[0]["path"])
        dlg = PicasaImportDialog(self.catalog_repo, initial_dir=initial, parent=self)
        if dlg.exec() == QDialog.Accepted and dlg.import_stats:
            st = dlg.import_stats
            self.refresh_navigation_and_autocomplete()
            self.rebuild_and_run_query()
            self.show_status_message(
                f"✓ Picasa import: {st['matched']} matched ({st['stars']} ♥, {st['tags']} tags, {st['faces']} faces, {st['albums']} albums)",
                5000,
            )

    def open_picasa_folders_dialog(self) -> None:
        from src.ui.picasa_import_dialog import PicasaFoldersDialog

        dlg = PicasaFoldersDialog(self.catalog_repo, parent=self)
        dlg.exec()
        if dlg.any_imported:
            self.refresh_navigation_and_autocomplete()
            self.rebuild_and_run_query()
            self.show_status_message("✓ Picasa metadata imported", 4000)

    def _advance_after_cull(self) -> None:
        """Advance to the next photo after a Shift+Rating/Flag culling action."""
        if self.central_stack.currentWidget() == self.media_viewer:
            self.navigate_single_viewer(1)
        elif self.central_stack.currentWidget() == self.grid_view and hasattr(self.grid_view, "advance_selection"):
            self.grid_view.advance_selection(1)

    def show_quick_preview(self, media_id: int | None = None) -> None:
        """Open or update the Spacebar Quick Preview dialog for the selected media item."""
        if media_id is None:
            target_ids = self.get_active_target_ids()
            if not target_ids:
                return
            media_id = int(target_ids[-1])
        try:
            detail = self.catalog_repo.get_media_detail(int(media_id))
        except Exception:
            detail = None
        if not detail:
            return
        folder_p = Path(str(detail.get("folder_path") or ""))
        filename = str(detail.get("filename") or "")
        file_path = folder_p / filename
        rating = int(detail.get("rating") or 0)
        flag = -1 if int(detail.get("is_rejected") or 0) else int(detail.get("flag") or 0)
        is_fav = bool(detail.get("is_favourite") or 0)

        dlg = getattr(self, "_quick_preview_dialog", None)
        if dlg is None:
            from src.ui.quick_preview_dialog import QuickPreviewDialog

            dlg = QuickPreviewDialog(parent=self)
            dlg.open_full_requested.connect(self._on_quick_preview_open_full)
            dlg.rate_requested.connect(self._on_quick_preview_rate)
            dlg.flag_set_requested.connect(self._on_quick_preview_flag_set)
            dlg.favourite_requested.connect(self._on_quick_preview_favourite)
            dlg.navigate_requested.connect(self._on_quick_preview_navigate)
            self._quick_preview_dialog = dlg

        dlg.show_media(
            int(media_id),
            file_path,
            filename,
            rating=rating,
            flag=flag,
            is_favourite=is_fav,
            favourites_enabled=bool(getattr(self, "favourites_enabled", True)),
        )
        if not dlg.isVisible():
            dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        dlg.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    def _on_quick_preview_open_full(self, media_id: int) -> None:
        dlg = getattr(self, "_quick_preview_dialog", None)
        if dlg is not None and dlg.isVisible():
            dlg.close()
        self.open_single_viewer(int(media_id))

    def _on_quick_preview_rate(self, media_id: int, rating: int) -> None:
        cmd = _RatingUndoCommand(self, [int(media_id)], int(rating))
        self.undo_stack.push(cmd)

    def _on_quick_preview_flag(self, media_id: int) -> None:
        detail = self.catalog_repo.get_media_detail(int(media_id))
        cur_flag = int(detail.get("flag") or 0) if detail else 0
        new_flag = 0 if cur_flag else 1
        self._on_quick_preview_flag_set(media_id, new_flag)

    def _on_quick_preview_flag_set(self, media_id: int, new_flag: int) -> None:
        cmd = _FlagUndoCommand(self, [int(media_id)], int(new_flag))
        self.undo_stack.push(cmd)
        if hasattr(self, "left_nav_dock") and hasattr(self, "catalog_repo"):
            try:
                self.left_nav_dock.refresh_albums_only(self.catalog_repo)
            except Exception:
                pass
        if hasattr(self, "right_tag_dock") and self.right_tag_dock is not None:
            try:
                self.right_tag_dock._update_flag_button(int(new_flag))
            except Exception:
                pass

    def _on_quick_preview_favourite(self, media_id: int) -> None:
        if not getattr(self, "favourites_enabled", True):
            return
        fav_map = self.catalog_repo.get_field_map_for_ids([int(media_id)], "is_favourite")
        cur_fav = bool(fav_map.get(int(media_id), 0))
        self.apply_favourite_to_selection(not cur_fav, target_ids=[int(media_id)])

    def _on_quick_preview_navigate(self, delta: int) -> None:
        if hasattr(self.grid_view, "advance_selection") and self.grid_view.advance_selection(delta):
            sel_ids = self.grid_view.selected_media_ids()
            if sel_ids:
                self.show_quick_preview(int(sel_ids[-1]))

    # -------------------------------------------------------------------------
    # Image Stacking & RAW+JPEG Pairs
    # -------------------------------------------------------------------------
    def toggle_stack_expanded(self, head_id: int) -> None:
        members = self.catalog_repo.get_stack_member_ids(int(head_id))
        hid = int(members[0]) if len(members) > 1 else int(head_id)
        if hid in self._expanded_stack_ids:
            self._expanded_stack_ids.discard(hid)
        else:
            self._expanded_stack_ids.add(hid)
        self.rebuild_and_run_query()

    def toggle_selected_stack_expanded(self) -> None:
        target_ids = list(self.get_active_target_ids())
        if not target_ids:
            return
        members = self.catalog_repo.get_stack_member_ids(int(target_ids[0]))
        if len(members) > 1:
            self.toggle_stack_expanded(members[0])

    def expand_all_stacks(self) -> None:
        records = getattr(self.catalog_model, "_records", []) or []
        heads = [
            int(r.id)
            for r in records
            if int(getattr(r, "stack_count", 1) or 1) > 1
        ]
        if not heads:
            self.show_status_message("No photo groups in current view.", 2500)
            return
        for hid in heads:
            members = self.catalog_repo.get_stack_member_ids(hid)
            if len(members) > 1:
                self._expanded_stack_ids.add(int(members[0]))
        self.rebuild_and_run_query()
        self.show_status_message("✓ Expanded all photo groups in view", 2500)

    def collapse_all_stacks(self) -> None:
        if not self._expanded_stack_ids:
            return
        self._expanded_stack_ids.clear()
        self.rebuild_and_run_query()
        self.show_status_message("✓ Collapsed all photo groups", 2500)

    def set_auto_group_bursts(self, enabled: bool) -> None:
        self.auto_group_bursts = bool(enabled)
        settings = QSettings("PhotoGallery64", "PhotoGallery64")
        settings.setValue("auto_group_bursts", self.auto_group_bursts)
        if hasattr(self, "ribbon") and hasattr(self.ribbon, "set_photo_grouping_state"):
            self.ribbon.set_photo_grouping_state(
                auto_bursts=self.auto_group_bursts,
                auto_raw_jpeg=self.auto_group_raw_jpeg,
            )
        if self.auto_group_bursts:
            count = self.catalog_repo.auto_stack_bursts(
                folder_ids=None,
                max_gap_sec=2.0,
                thumb_repo=self.thumb_repo,
            )
            self._expanded_stack_ids.clear()
            self.rebuild_and_run_query()
            self.show_status_message(f"✓ Auto-group bursts enabled ({count} group(s))", 3500)
        else:
            removed = self.catalog_repo.unstack_by_source("burst")
            self._expanded_stack_ids.clear()
            self.rebuild_and_run_query()
            self.show_status_message(f"✓ Auto-group bursts disabled ({removed} photo(s) ungrouped)", 3500)

    def set_auto_group_raw_jpeg(self, enabled: bool) -> None:
        self.auto_group_raw_jpeg = bool(enabled)
        settings = QSettings("PhotoGallery64", "PhotoGallery64")
        settings.setValue("auto_group_raw_jpeg", self.auto_group_raw_jpeg)
        if hasattr(self, "ribbon") and hasattr(self.ribbon, "set_photo_grouping_state"):
            self.ribbon.set_photo_grouping_state(
                auto_bursts=self.auto_group_bursts,
                auto_raw_jpeg=self.auto_group_raw_jpeg,
            )
        if self.auto_group_raw_jpeg:
            count = self.catalog_repo.auto_stack_raw_jpeg_pairs(folder_ids=None)
            self._expanded_stack_ids.clear()
            self.rebuild_and_run_query()
            self.show_status_message(f"✓ Auto-group RAW + JPEG enabled ({count} pair(s))", 3500)
        else:
            removed = self.catalog_repo.unstack_by_source("raw_jpeg")
            self._expanded_stack_ids.clear()
            self.rebuild_and_run_query()
            self.show_status_message(f"✓ Auto-group RAW + JPEG disabled ({removed} photo(s) ungrouped)", 3500)

    def toggle_all_badges(self) -> None:
        any_on = bool(
            self._badge_show_flags
            or self._badge_show_favourites
            or self._badge_show_groups
        )
        if any_on:
            self._saved_badge_prefs = (
                bool(self._badge_show_flags),
                bool(self._badge_show_favourites),
                bool(self._badge_show_groups),
            )
            self.set_badge_visibility(flags=False, favourites=False, groups=False)
            self.show_status_message("Thumbnail badges hidden", 2000)
        else:
            saved = getattr(self, "_saved_badge_prefs", (True, True, True))
            if not any(saved):
                saved = (True, True, True)
            self.set_badge_visibility(flags=saved[0], favourites=saved[1], groups=saved[2])
            self.show_status_message("Thumbnail badges shown", 2000)

    def set_badge_visibility(
        self,
        *,
        flags: bool | None = None,
        favourites: bool | None = None,
        groups: bool | None = None,
        save: bool = True,
    ) -> None:
        if flags is not None:
            self._badge_show_flags = bool(flags)
        if favourites is not None:
            self._badge_show_favourites = bool(favourites)
        if groups is not None:
            self._badge_show_groups = bool(groups)
        if hasattr(self, "card_delegate") and self.card_delegate is not None:
            self.card_delegate.show_flag_badges = self._badge_show_flags
            self.card_delegate.show_favourite_badges = self._badge_show_favourites
            self.card_delegate.show_group_badges = self._badge_show_groups
        if hasattr(self, "ribbon") and hasattr(self.ribbon, "set_badges_state"):
            self.ribbon.set_badges_state(
                show_flags=self._badge_show_flags,
                show_favourites=self._badge_show_favourites,
                show_groups=self._badge_show_groups,
            )
        if save:
            settings = QSettings("PhotoGallery64", "PhotoGallery64")
            settings.setValue("badge_show_flags", self._badge_show_flags)
            settings.setValue("badge_show_favourites", self._badge_show_favourites)
            settings.setValue("badge_show_groups", self._badge_show_groups)
        if hasattr(self, "grid_view") and self.grid_view is not None:
            self.grid_view.viewport().update()

    def _current_stack_scope_folder_ids(self) -> list[int] | None:
        """Return folder IDs for the active view/folder scope, or None for entire library."""
        if not hasattr(self, "left_nav_dock") or self.left_nav_dock is None:
            return None
        spec = self.left_nav_dock.build_filter_spec()
        records = getattr(self.catalog_model, "_records", []) or []
        if spec.folder_id is not None:
            fids = {int(spec.folder_id)}
            for r in records:
                fid = getattr(r, "folder_id", None)
                if fid:
                    fids.add(int(fid))
            return sorted(fids)
        has_scope = any(
            getattr(spec, attr, None) is not None
            for attr in (
                "album_id",
                "smart_album_id",
                "tag_id",
                "person_id",
                "camera_model",
                "location_country",
                "location_state",
                "location_city",
                "date_from",
            )
        ) or bool(getattr(self, "_active_subject_ids", None))
        if has_scope and records:
            fids = sorted({int(r.folder_id) for r in records if getattr(r, "folder_id", None)})
            if fids:
                return fids
        return None

    def auto_stack_raw_jpeg_pairs(self, *, entire_library: bool = False) -> None:
        scope_fids = None if entire_library else self._current_stack_scope_folder_ids()
        before_map: dict[int, int | None] = {}
        count = self.catalog_repo.auto_stack_raw_jpeg_pairs(
            folder_ids=scope_fids,
            capture_undo_out=before_map,
        )
        if before_map:
            after_map = self.catalog_repo.get_stack_parent_map(list(before_map.keys()))
            self.undo_stack.push(
                _StackMutationUndoCommand(self, before_map, after_map, "Auto-Group RAW + JPEG")
            )
        self._expanded_stack_ids.clear()
        self.rebuild_and_run_query()
        scope_lbl = "in current folder" if scope_fids else "across library"
        self.show_status_message(f"✓ Paired {count} RAW + JPEG group(s) {scope_lbl}", 3500)

    def auto_stack_bursts(self, *, entire_library: bool = False) -> None:
        scope_fids = None if entire_library else self._current_stack_scope_folder_ids()
        before_map: dict[int, int | None] = {}
        count = self.catalog_repo.auto_stack_bursts(
            folder_ids=scope_fids,
            max_gap_sec=2.0,
            thumb_repo=self.thumb_repo,
            capture_undo_out=before_map,
        )
        if before_map:
            after_map = self.catalog_repo.get_stack_parent_map(list(before_map.keys()))
            self.undo_stack.push(
                _StackMutationUndoCommand(self, before_map, after_map, "Auto-Group Bursts")
            )
        self._expanded_stack_ids.clear()
        self.rebuild_and_run_query()
        scope_lbl = "in current folder" if scope_fids else "across library"
        self.show_status_message(f"✓ Grouped {count} burst group(s) {scope_lbl}", 3500)

    def expand_stack_targets_if_collapsed(
        self, target_ids: Sequence[int], *, action_label: str = "modify"
    ) -> list[int]:
        """If any selected ID is a collapsed stack head, include all stack members (with prompt)."""
        out: list[int] = []
        seen: set[int] = set()
        extra_count = 0
        expanded_groups: list[tuple[int, list[int]]] = []
        for mid in target_ids:
            imid = int(mid)
            members = self.catalog_repo.get_stack_member_ids(imid)
            if len(members) > 1 and imid not in self._expanded_stack_ids and members[0] == imid:
                extra_count += len(members) - 1
                expanded_groups.append((imid, members))
            else:
                expanded_groups.append((imid, [imid]))

        include_children = False
        if extra_count > 0:
            reply = QMessageBox.question(
                self,
                "Collapsed Stack Scope",
                f"Your selection includes collapsed stack(s) with {extra_count} hidden item(s).\n\n"
                f"Include all items in the stack(s) when you {action_label}?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            include_children = reply == QMessageBox.Yes

        for imid, members in expanded_groups:
            use_list = members if include_children else [imid]
            for m in use_list:
                if m not in seen:
                    seen.add(m)
                    out.append(m)
        return out

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
        clean_query = str(location_name or "").strip()
        if not clean_query:
            cmd = _GeotagUndoCommand(self, target_ids, None, None, "")
            self.undo_stack.push(cmd)
            self.show_status_message(f"✓ Cleared geotag on {len(target_ids)} item(s)", 3000)
            return

        lat, lon, loc_label = resolve_location_query(
            clean_query, allow_network=False, db_engine=self.engine
        )
        if lat is not None and lon is not None:
            cmd = _GeotagUndoCommand(self, target_ids, lat, lon, loc_label)
            self.undo_stack.push(cmd)
            self.show_status_message(f"✓ Tagged location '{loc_label}' on {len(target_ids)} item(s)", 3500)
            return

        import threading

        def _bg_resolve() -> None:
            try:
                rlat, rlon, rlabel = resolve_location_query(
                    clean_query, allow_network=True, db_engine=self.engine
                )
            except Exception:
                rlat, rlon, rlabel = None, None, clean_query
            self._pending_async_geotag_result = (
                list(target_ids),
                clean_query,
                rlat,
                rlon,
                str(rlabel or ""),
            )
            QMetaObject.invokeMethod(self, "_finish_async_geotag", Qt.QueuedConnection)

        threading.Thread(target=_bg_resolve, daemon=True).start()

    @Slot()
    def _finish_async_geotag(self) -> None:
        res = getattr(self, "_pending_async_geotag_result", None)
        if not res:
            return
        self._pending_async_geotag_result = None
        target_ids, clean_query, rlat, rlon, rlabel = res
        if rlat is not None and rlon is not None:
            final_label = rlabel or clean_query
            cmd = _GeotagUndoCommand(self, target_ids, rlat, rlon, final_label)
            self.undo_stack.push(cmd)
            if self.right_tag_dock.lbl_geotag_status is not None:
                self.right_tag_dock.lbl_geotag_status.setVisible(False)
            self.show_status_message(
                f"✓ Tagged location '{final_label}' on {len(target_ids)} item(s)", 3500
            )
        else:
            self.right_tag_dock.revert_invalid_geotag(clean_query)
            self.show_status_message(
                f"Invalid location: '{clean_query}' was not found — click the pin icon to set on map",
                4500,
            )

    def open_gps_match_dialog(self) -> None:
        """Opens Companion Phone GPS Matching dialog for Grid selection or World Map View."""
        from src.core.gps_matcher import DonorPoint, TargetPhoto
        from src.ui.gps_match_dialog import GpsMatchDialog

        in_map = (
            self._map_view is not None
            and self.central_stack.currentWidget() == self._map_view
        )
        bbox = None
        selected_ids: list[int] = []
        view_ids: list[int] = [int(r.id) for r in getattr(self.catalog_model, "_records", [])][:2500]
        if in_map and self._map_view is not None:
            bbox = (
                float(self._map_view._min_lat),
                float(self._map_view._max_lat),
                float(self._map_view._min_lon),
                float(self._map_view._max_lon),
            )
            cluster_ids = getattr(self, "_selected_ids", [])
            if cluster_ids:
                selected_ids = list(cluster_ids)[:2500]
        else:
            selected_ids = list(self.get_active_target_ids())[:2500]

        def _convert_rows(t_raw: list[dict], d_raw: list[dict]) -> tuple[list[TargetPhoto], list[DonorPoint]]:
            t_list = [
                TargetPhoto(
                    media_id=r["id"],
                    path=r["path"],
                    filename=r["filename"],
                    timestamp=r["date_taken"],
                    camera_model=r["camera_model"],
                )
                for r in t_raw
            ]
            d_list = [
                DonorPoint(
                    media_id=r["id"],
                    timestamp=r["date_taken"],
                    lat=float(r["lat"]),
                    lon=float(r["lon"]),
                    location_name=r["location_name"],
                    camera_model=r["camera_model"],
                    filename=r["filename"],
                    path=r.get("path", ""),
                    source="catalog",
                )
                for r in d_raw
                if r["lat"] is not None and r["lon"] is not None
            ]
            return t_list, d_list

        def _load_scope(scope_key: str) -> tuple[list[TargetPhoto], list[DonorPoint], str]:
            if scope_key == "map" and bbox is not None:
                t_raw, d_raw, lbl = self.catalog_repo.get_gps_matching_candidates(
                    selected_ids=selected_ids,
                    bbox=bbox,
                )
            elif scope_key == "selected" and selected_ids:
                t_raw, d_raw, lbl = self.catalog_repo.get_gps_matching_candidates(
                    selected_ids=selected_ids,
                )
            elif scope_key == "view" and view_ids:
                t_raw, d_raw, lbl = self.catalog_repo.get_gps_matching_candidates(
                    selected_ids=view_ids,
                )
            else:
                return [], [], "Current Scope"
            t_list, d_list = _convert_rows(t_raw, d_raw)
            return t_list, d_list, lbl

        scope_options: list[tuple[str, str]] = []
        if in_map and bbox is not None:
            scope_options.append(("map", "Map Area / Selected Cluster"))
        if len(selected_ids) >= 2:
            scope_options.append(("selected", f"Selected Photos ({len(selected_ids)})"))
        elif len(selected_ids) == 1:
            scope_options.append(("selected", "Selected Photo (1)"))
        if view_ids:
            scope_options.append(("view", f"Current Folder / View ({len(view_ids)})"))

        if not scope_options:
            self.show_status_message(
                "Open a folder or select photos in the Gallery (or zoom into a Map area) to match locations.",
                4000,
            )
            return

        if in_map and bbox is not None:
            initial_scope = "map"
        elif len(selected_ids) >= 2:
            initial_scope = "selected"
        else:
            initial_scope = "view"

        targets, donors, scope_label = _load_scope(initial_scope)

        dlg = GpsMatchDialog(
            targets,
            donors,
            scope_label=scope_label,
            db_engine=self.engine,
            thumb_repo=self.thumb_repo,
            scope_options=scope_options,
            initial_scope=initial_scope,
            scope_loader=_load_scope,
            parent=self,
        )
        if dlg.exec() == QDialog.Accepted:
            chosen = dlg.selected_matches()
            if not chosen:
                return
            shift_sec = dlg.clock_offset_seconds() if dlg.should_shift_timestamps() else 0
            cmd = _BatchGpsMatchUndoCommand(self, chosen, clock_shift_sec=shift_sec)
            if hasattr(self, "undo_stack") and self.undo_stack is not None:
                self.undo_stack.push(cmd)
            else:
                cmd.redo()
            self.show_status_message(
                f"✓ Applied matched GPS locations to {len(chosen)} photo(s) (Ctrl+Z to undo)",
                4500,
            )

    def open_adjust_location_dialog(self) -> None:
        """Opens interactive map pin adjustment dialog for the active single or group selection."""
        target_ids = self.get_active_target_ids()
        if not target_ids:
            self.show_status_message("Select one or more photos to adjust location", 3000)
            return
        from src.ui.map_view import AdjustLocationDialog

        before_map = self.catalog_repo.get_geotag_map(target_ids)
        init_lat: float | None = None
        init_lon: float | None = None
        init_name = ""
        for mid in target_ids:
            info = before_map.get(mid)
            if not info:
                continue
            lat_val, lon_val, loc_val = info
            if init_lat is None and lat_val is not None and lon_val is not None:
                init_lat = float(lat_val)
                init_lon = float(lon_val)
            loc_txt = str(loc_val or "").strip()
            if loc_txt and not init_name:
                init_name = loc_txt

        dlg = AdjustLocationDialog(
            self,
            db_engine=self.engine,
            initial_lat=init_lat,
            initial_lon=init_lon,
            initial_name=init_name,
            item_count=len(target_ids),
        )
        if dlg.exec() == QDialog.Accepted:
            new_lat, new_lon, new_name = dlg.result_geotag()
            self.apply_adjusted_geotag(target_ids, new_lat, new_lon, new_name)

    def apply_adjusted_geotag(
        self,
        target_ids: Sequence[int],
        lat: float,
        lon: float,
        location_name: str,
    ) -> None:
        """Applies pin-adjusted coordinates and custom label to single or group selection with undo."""
        ids = [int(mid) for mid in target_ids]
        if not ids:
            return
        final_label = str(location_name or "").strip() or f"{lat:.4f}°, {lon:.4f}°"
        leaf_name = final_label.split(",")[0].strip() or final_label
        save_place_to_db(
            self.engine,
            {
                "name": leaf_name,
                "formatted_address": final_label,
                "lat": float(lat),
                "lon": float(lon),
            },
        )
        cmd = _GeotagUndoCommand(self, ids, float(lat), float(lon), final_label)
        self.undo_stack.push(cmd)
        self._schedule_tag_nav_refresh({"places", "albums"})
        if self.right_tag_dock.lbl_geotag_status is not None:
            self.right_tag_dock.lbl_geotag_status.setVisible(False)
        self.show_status_message(
            f"✓ Updated location '{final_label}' on {len(ids)} item(s)", 3500
        )

    def _schedule_tag_nav_refresh(self, sections: set[str] | None = None) -> None:
        """Debounces left-nav tree & grid rebuilds after rapid tag/geotag edits."""
        pending: set[str] | None = getattr(self, "_pending_nav_refresh_sections", set())
        if sections is None:
            self._pending_nav_refresh_sections = None
        elif pending is not None:
            pending.update(sections)
            self._pending_nav_refresh_sections = pending

        timer = getattr(self, "_tag_nav_refresh_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(120)

            def _flush() -> None:
                sec = getattr(self, "_pending_nav_refresh_sections", None)
                self._pending_nav_refresh_sections = set()
                self.refresh_navigation_and_autocomplete(
                    sections=sec if sec else None
                )
                if self.central_stack.currentWidget() == self.grid_view:
                    self.rebuild_and_run_query(preserve_selection=True)

            timer.timeout.connect(_flush)
            self._tag_nav_refresh_timer = timer
        timer.start()

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
        summary = self.catalog_repo.get_selection_summary(target_ids)
        self.right_tag_dock.bind_selection(target_ids, summary)
        self._schedule_tag_nav_refresh({"tags", "albums"})
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
        summary = self.catalog_repo.get_selection_summary(target_ids)
        self.right_tag_dock.bind_selection(target_ids, summary)
        self._schedule_tag_nav_refresh({"tags", "albums"})
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
        summary = self.catalog_repo.get_selection_summary(target_ids)
        self.right_tag_dock.bind_selection(target_ids, summary)
        self._schedule_tag_nav_refresh({"tags", "albums"})
        tag_lbl = f"'{canon_path}'" if canon_path else "tag"
        self.show_status_message(f"✓ Applied {tag_lbl} to {len(target_ids)} item(s)", 3000)

    def _on_ocr_text_edited(self, new_text: str) -> None:
        target_ids = self.get_active_target_ids()
        if not target_ids:
            return
        mid = int(target_ids[0])
        clean = str(new_text or "").strip()
        self.catalog_repo.update_media_ocr(mid, clean)
        self._media_meta_lookup.pop(mid, None)
        summary = self.catalog_repo.get_selection_summary([mid])
        self.right_tag_dock.bind_selection([mid], summary)
        if clean:
            self.show_status_message("✓ Updated OCR text", 3000)
        else:
            self.show_status_message("✓ Removed OCR text", 3000)

    def _on_ocr_rescan_requested(self, backend: str) -> None:
        target_ids = self.get_active_target_ids()
        if not target_ids:
            return
        mid = int(target_ids[0])
        detail = self.catalog_repo.get_media_detail(mid)
        if not detail:
            return
        fpath = detail.get("file_path") or (
            str(Path(detail["folder_path"]) / detail["filename"])
            if detail.get("folder_path") and detail.get("filename")
            else None
        )
        if not fpath or not Path(fpath).exists():
            return
        self.show_status_message(f"Rescanning OCR ({backend})...", 4000)

        import threading

        def _bg_ocr() -> None:
            text = ""
            err_msg = ""
            actual_backend = backend
            try:
                import numpy as np
                from PIL import Image, ImageOps
                from src.core.ocr_engine import OcrEngine

                engine = OcrEngine(backend=backend)
                actual_backend = engine.backend_name
                if backend != "auto" and actual_backend == "stub":
                    err_msg = f"OCR backend '{backend}' is not available on this system"
                else:
                    with Image.open(fpath) as im:
                        im_fixed = ImageOps.exif_transpose(im).convert("RGB")
                        rgb = np.ascontiguousarray(np.array(im_fixed, dtype=np.uint8))
                    text = str(engine.extract_text(rgb) or "").strip()
            except Exception as exc:
                err_msg = str(exc)

            self._pending_async_ocr_rescan = (mid, actual_backend, text, err_msg)
            QMetaObject.invokeMethod(self, "_finish_async_ocr_rescan", Qt.QueuedConnection)

        threading.Thread(target=_bg_ocr, daemon=True).start()

    @Slot()
    def _finish_async_ocr_rescan(self) -> None:
        res = getattr(self, "_pending_async_ocr_rescan", None)
        if not res:
            return
        self._pending_async_ocr_rescan = None
        mid, backend, text, err_msg = res
        if err_msg:
            self.show_status_message(f"OCR rescan failed: {err_msg}", 4500)
            return
        if not text:
            existing = str((self.catalog_repo.get_media_detail(mid) or {}).get("ocr_text") or "").strip()
            if existing:
                self.show_status_message(
                    f"OCR rescan ({backend}): no text found — kept existing text",
                    4000,
                )
                return
        self.catalog_repo.update_media_ocr(mid, text)
        self._media_meta_lookup.pop(mid, None)
        active_ids = self.get_active_target_ids()
        if active_ids and int(active_ids[0]) == mid:
            summary = self.catalog_repo.get_selection_summary([mid])
            self.right_tag_dock.bind_selection([mid], summary)
        self.show_status_message(
            f"✓ OCR rescanned ({backend})" if text else f"OCR rescan ({backend}): no text found",
            3500,
        )

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
        self.refresh_navigation_and_autocomplete({"people", "albums"})
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
        self.refresh_navigation_and_autocomplete({"people", "albums"})
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
        self.refresh_navigation_and_autocomplete({"people", "albums"})
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
        if getattr(self, "find_people_enabled", True):
            self.start_face_detection()
        self.show_status_message(f"✓ Tagged '{person_name.strip()}' in {len(target_ids)} item(s)", 3500)

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
        self.refresh_navigation_and_autocomplete({"people", "albums"})
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
            self._notify_batch_people_face_changed([face_id])
        person_name = "person"
        if person_id > 0:
            p_row = self.catalog_repo.engine.catalog_conn().execute(
                "SELECT name FROM people WHERE id = ?;", (person_id,)
            ).fetchone()
            if p_row and p_row["name"]:
                person_name = str(p_row["name"])
        self.refresh_people_navigation_and_autocomplete()
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
        if getattr(self, "find_people_enabled", True):
            self.start_face_detection()
        self.show_status_message(f"✓ Tagged '{person_name.strip()}' in {len(target_ids)} item(s)", 3500)

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
            self._notify_batch_people_face_changed([face_id])
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
        self.show_status_message("Face suggestion dismissed", 3000)

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
        self._notify_batch_people_face_changed([int(face_id)])
        self.refresh_navigation_and_autocomplete({"people", "albums"})
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
        if getattr(self, "find_people_enabled", True):
            self.start_face_detection()
        self.show_status_message(f"✓ Tagged '{person_name.strip()}' in {len(target_ids)} item(s)", 3500)

    def _on_dismiss_face_requested(self, face_id: int) -> None:
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else list(self._selected_ids)
        )
        self.catalog_repo.dismiss_face_region(int(face_id))
        self._notify_batch_people_face_changed([int(face_id)])
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
        self.show_status_message("Dismissed face box", 3000)

    def _on_remove_face_requested(self, face_id: int) -> None:
        target_ids = (
            [self.media_viewer.current_media_id]
            if self.central_stack.currentWidget() == self.media_viewer
            and self.media_viewer.current_media_id is not None
            else list(self._selected_ids)
        )
        self.catalog_repo.remove_face_assignment(int(face_id))
        self.refresh_navigation_and_autocomplete({"people", "albums"})
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
        self.show_status_message("Removed person from face", 3000)

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
            self.refresh_navigation_and_autocomplete({"people", "albums"})
            target_ids = [media_id]
            summary = self.catalog_repo.get_selection_summary(target_ids)
            self.right_tag_dock.bind_selection(target_ids, summary)
            self.show_status_message("✓ Updated display picture", 3500)

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
        self.refresh_navigation_and_autocomplete({"people", "albums"})
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
        self.show_status_message(f"✓ Removed person tag from {len(media_ids)} item(s)", 3500)

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
        self.refresh_navigation_and_autocomplete({"people", "albums"})
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
        self.show_status_message(f"✓ Marked {len(target_face_ids)} face region(s) as 'Not a face'", 3500)

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
        self.refresh_navigation_and_autocomplete({"people", "albums"})
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
        self.show_status_message(f"✓ Retagged {len(media_ids)} item(s) to '{new_name_clean}'", 3500)

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

        self.refresh_navigation_and_autocomplete({"people", "albums"})
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
        self.show_status_message(f"✓ Merged '{source_name}' into '{target_name}'", 3500)

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
        self.refresh_navigation_and_autocomplete({"people", "albums"})
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
        self.show_status_message(f"✓ Renamed '{old_name}' to '{new_name.strip()}'", 3500)

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
            if getattr(self.media_viewer, "current_media_id", None) == mid:
                self.media_viewer.current_file_path = new_path
                if self.central_stack.currentWidget() == self.media_viewer:
                    self.setWindowTitle(f"{new_fname} - Essentials Photo Gallery")
            self.rebuild_and_run_query()
            summary = self.catalog_repo.get_selection_summary(target_ids)
            self.right_tag_dock.bind_selection(target_ids, summary)
            self.show_status_message(f"✓ Renamed file to '{new_fname}'", 3500)
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
        self.refresh_navigation_and_autocomplete({"dates", "albums"})
        self.rebuild_and_run_query()
        summary = self.catalog_repo.get_selection_summary(target_ids)
        self.right_tag_dock.bind_selection(target_ids, summary)
        self.show_status_message(f"✓ Updated Date taken for {len(target_ids)} file(s)", 3500)

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
        self.show_status_message(f"✓ Updated Author for {len(target_ids)} file(s)", 3500)

    def _update_region_histogram_mask(
        self, scope: str | None = None, *, from_image_changed: bool = False
    ) -> None:
        if not hasattr(self, "right_tag_dock") or not hasattr(self.right_tag_dock, "histogram_widget"):
            return
        hist = self.right_tag_dock.histogram_widget
        reg = scope if scope is not None else getattr(self.right_tag_dock, "_active_region_scope", "whole")
        if reg == "whole":
            if from_image_changed:
                hist._hist_mask = None
            else:
                hist.set_histogram_mask(None)
            return
        proxy = getattr(self.media_viewer, "display_proxy_rgb", None)
        if proxy is None:
            proxy = getattr(self.media_viewer, "master_np_rgb", None)
        mask = None
        if proxy is not None:
            try:
                from src.imaging.creative_effects import get_depth_and_matte_engine

                engine = get_depth_and_matte_engine()
                mid = getattr(self.media_viewer, "current_media_id", None)
                mask = engine.get_region_mask(proxy, reg, media_id=mid)
            except Exception:
                mask = None
        if from_image_changed:
            hist._hist_mask = mask
        else:
            hist.set_histogram_mask(mask)

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
        self._notify_batch_people_face_changed()
        self.invalidate_face_crop_ui(media_id=int(media_id), person_ids=[int(pid)])
        self.refresh_navigation_and_autocomplete({"people", "albums"})
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
            self.refresh_navigation_and_autocomplete({"dates", "albums"})
            self.rebuild_and_run_query()
            summary = self.catalog_repo.get_selection_summary(self._selected_ids)
            self.right_tag_dock.bind_selection(self._selected_ids, summary)
            return dlg

        if dlg.exec() == ChangeTimeTakenDialog.Accepted:
            self.catalog_repo.apply_relative_time_shift(
                self._selected_ids, lead_id, dlg.new_lead_timestamp()
            )
            self.refresh_navigation_and_autocomplete({"dates", "albums"})
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
            if hasattr(self.media_viewer, "_update_filmstrip_toggle"):
                self.media_viewer._update_filmstrip_toggle()
            self.media_viewer.setFocus()
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

            if hasattr(self, "bottom_dock") and self.bottom_dock is not None and getattr(self, "_active_create_workspace", None) is None:
                self.bottom_dock.refresh_items()
                self.bottom_dock.set_filmstrip_mode()
                self.bottom_dock.set_active_media_id(media_id)
                show_fs = bool(getattr(self, "_filmstrip_viewer_enabled", False))
                self.bottom_dock.setVisible(show_fs)
                if hasattr(self.ribbon, "btn_toggle_filmstrip"):
                    self.ribbon.btn_toggle_filmstrip.blockSignals(True)
                    self.ribbon.btn_toggle_filmstrip.setChecked(show_fs)
                    self.ribbon.btn_toggle_filmstrip.blockSignals(False)
                if hasattr(self, "btn_status_filmstrip"):
                    self.btn_status_filmstrip.blockSignals(True)
                    self.btn_status_filmstrip.setChecked(show_fs)
                    self.btn_status_filmstrip.blockSignals(False)

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
        if getattr(self, "_viewer_effects_preview_timer", None) is None:
            self._viewer_effects_preview_timer = QTimer(self)
            self._viewer_effects_preview_timer.setSingleShot(True)
            self._viewer_effects_preview_timer.setInterval(150)
            self._viewer_effects_preview_timer.timeout.connect(
                self._do_refresh_effects_gallery_thumbnails
            )
        if hasattr(self.ribbon, "effects_gallery") and self.central_stack.currentWidget() == self.media_viewer:
            self._viewer_effects_preview_timer.start()
        else:
            self._viewer_effects_preview_timer.stop()

    def _do_refresh_effects_gallery_thumbnails(self) -> None:
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
        if getattr(self, "_active_create_workspace", None) is not None:
            if not self.close_create_workspace():
                return
        if hasattr(self, "duplicate_finder_view") and self.central_stack.currentWidget() == self.duplicate_finder_view:
            self.duplicate_finder_view.stop_scan()

        returning_viewer_mid: int | None = None
        if self.central_stack.currentWidget() == self.media_viewer:
            returning_viewer_mid = getattr(self.media_viewer, "current_media_id", None)
            if self.media_viewer.has_unsaved_adjustments():
                if not self.media_viewer.prompt_save_changes_if_needed("close"):
                    return
                self.rebuild_and_run_query()
        self.media_viewer.media_player.stop()
        self.media_viewer._stop_animated_image()
        self.media_viewer.set_markup_mode(False)
        self.media_viewer.set_retouch_mode(False)
        self.media_viewer.set_red_eye_mode(False)
        self.media_viewer.set_crop_mode(False)
        if hasattr(self, "bottom_dock") and self.bottom_dock is not None and getattr(self, "_active_create_workspace", None) is None:
            self.bottom_dock.hide()
        if hasattr(self, "right_tag_dock") and hasattr(self.right_tag_dock, "clear_mask_editing_session"):
            self.right_tag_dock.clear_mask_editing_session()

        prev_widget = getattr(self, "_prev_central_widget", None)
        self._prev_central_widget = None
        if hasattr(self, "batch_people_view") and prev_widget == self.batch_people_view:
            self.central_stack.setCurrentWidget(self.batch_people_view)
            self.ribbon.set_active_mode("batch_people")
            if getattr(self.batch_people_view, "_needs_external_refresh", False):
                self.batch_people_view._needs_external_refresh = False
                self.batch_people_view.refresh_suggestions()
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
        self._update_status_counts()
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
        self.btn_status_filmstrip.setVisible(False)
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

        # Sync viewer's current photo back to gallery selection & focus
        if returning_viewer_mid is not None:
            self.grid_view.select_media_id(int(returning_viewer_mid))
            self._selected_ids = [int(returning_viewer_mid)]
            self._lead_selected_id = int(returning_viewer_mid)

        # Rebind selection from grid to right dock
        sel = self.grid_view.selected_media_ids() or list(self._selected_ids)
        if sel and self._lead_selected_id not in sel:
            self._lead_selected_id = int(sel[-1])
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
            self._map_view.bind_thumbnail_pool(
                self.thumbnail_pool,
                is_paused_fn=lambda: bool(getattr(self, "_workers_paused", False)),
            )
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

    def _prewarm_map_view_idle(self) -> None:
        """No-op: WorldMapView initializes on demand when World Map is clicked (avoids 3s post-launch main-thread QWebEngine freeze)."""
        return

    def _has_active_grid_scope_filter(self) -> bool:
        """Returns True if the current Grid view is narrowed by a Folder, Album, Tag, Place, Date, Person, Rating, or Search."""
        if getattr(self, "_active_bbox", None) is not None:
            return False
        if getattr(self, "_active_similar_order", None) is not None:
            return True
        if getattr(self, "_active_date_range", None) is not None:
            return True
        if hasattr(self, "scope_bar") and (self.scope_bar.search_text() or "").strip():
            return True
        if hasattr(self, "left_nav_dock"):
            st = self.left_nav_dock.current_filter_state()
            if any(
                (
                    st.get("folder_id") is not None,
                    st.get("album_id") is not None,
                    st.get("smart_album_id") is not None,
                    st.get("include_tags"),
                    st.get("exclude_tags"),
                    st.get("person_ids"),
                    st.get("special_people_mode"),
                    st.get("place_filter"),
                    st.get("rating_filters"),
                    st.get("flagged_only"),
                    st.get("media_type"),
                    st.get("orientation_filter") not in (None, "all"),
                )
            ):
                return True
        return False

    def _get_current_view_scope_ids(self) -> list[int]:
        return [int(r.id) for r in getattr(self.catalog_model, "_records", [])]

    def _on_ribbon_map_scope_changed(self, use_current_view: bool) -> None:
        self._map_scope_override = bool(use_current_view)
        mv = self._ensure_map_view()
        if use_current_view:
            ids = self._get_current_view_scope_ids()
            if hasattr(self.ribbon, "set_map_scope_ui"):
                self.ribbon.set_map_scope_ui(True, len(ids))
            mv.set_scope_media_ids(ids, fit=True)
        else:
            if hasattr(self.ribbon, "set_map_scope_ui"):
                self.ribbon.set_map_scope_ui(False, 0)
            mv.set_scope_media_ids(None, fit=True)

    def show_map_view(self) -> None:
        mv = self._ensure_map_view()
        self.central_stack.setCurrentWidget(mv)
        self.ribbon.set_active_mode("map")
        self.left_nav_dock.hide()
        self.right_tag_dock.hide()
        if hasattr(self, "bottom_dock") and self.bottom_dock is not None:
            self.bottom_dock.hide()
        if hasattr(self.ribbon, "sync_map_pane_buttons"):
            self.ribbon.sync_map_pane_buttons(
                self.left_nav_dock.isVisible(), self.right_tag_dock.isVisible()
            )
        if hasattr(self.ribbon, "sync_map_style_button"):
            self.ribbon.sync_map_style_button(mv.map_style)
        if hasattr(self.ribbon, "populate_map_filters") and hasattr(self, "catalog_repo"):
            if mv.is_data_dirty() or not getattr(self, "_map_filters_populated", False):
                try:
                    years = mv.get_cached_geotagged_years() if hasattr(mv, "get_cached_geotagged_years") else self.catalog_repo.get_geotagged_years()
                except Exception:
                    years = []
                try:
                    people = self.catalog_repo.get_people_list()
                except Exception:
                    people = []
                self.ribbon.populate_map_filters(years, people)
                self._map_filters_populated = True
        if hasattr(self.ribbon, "set_map_filter_ui"):
            self.ribbon.set_map_filter_ui(*mv.active_map_filters)

        # Scope World Map automatically to active Folder / Album / Smart Album / Search when filtered,
        # unless user explicitly toggled Entire Library via the Map Scope button.
        override = getattr(self, "_map_scope_override", None)
        use_current_view = self._has_active_grid_scope_filter() if override is None else bool(override)
        scope_ids = self._get_current_view_scope_ids() if use_current_view else None
        if hasattr(self.ribbon, "set_map_scope_ui"):
            self.ribbon.set_map_scope_ui(use_current_view, len(scope_ids or []))
        if scope_ids is not None or mv._scope_media_ids is not None:
            mv.set_scope_media_ids(scope_ids, fit=True)

        if not getattr(self, "_map_view_initialized", False):
            self._map_view_initialized = True
            if scope_ids is None:
                if getattr(mv, "web_view", None) is not None and hasattr(mv, "refresh_overview_async"):
                    mv.refresh_overview_async(fit=True)
                else:
                    mv.fit_to_photos()
        elif mv.is_data_dirty() and scope_ids is None:
            if getattr(mv, "web_view", None) is not None and hasattr(mv, "refresh_overview_async"):
                mv.refresh_overview_async(fit=False)
            else:
                mv.refresh_current_viewport()
        if getattr(mv, "_web_loaded", False) and getattr(mv, "web_view", None) is not None:
            QTimer.singleShot(60, mv.safe_invalidate_map_size)
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
        if self.central_stack.currentWidget() not in (self.batch_people_view, self.media_viewer):
            self._nav_dock_was_visible = (not self.left_nav_dock.isHidden()) if self.isVisible() else True
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
            self.start_face_detection(force_rescan=force_rescan, prompt_if_missing=True)
        if person_id is not None:
            self.batch_people_view.filter_to_person(person_id)
        else:
            self.batch_people_view.refresh_suggestions()
        if hasattr(self, "ribbon") and hasattr(self.ribbon, "update_batch_action_labels") and hasattr(self, "batch_people_view"):
            self.ribbon.update_batch_action_labels(getattr(self.batch_people_view, "_current_view_mode", "suggestions"))
        self.left_nav_dock.hide()
        self.right_tag_dock.hide()
        if hasattr(self, "bottom_dock") and self.bottom_dock is not None:
            self.bottom_dock.hide()
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
        if not self._is_navigating_history:
            filt = self.current_filter.to_dict() if hasattr(self, "current_filter") and self.current_filter else {}
            self._push_nav_state({"view": "grid", "filter": filt})

    def _notify_batch_people_face_changed(self, face_ids: list[int] | None = None) -> None:
        """Immediately syncs external face confirmations/dismissals into BatchPeopleView."""
        if not hasattr(self, "batch_people_view"):
            return
        if hasattr(self.batch_people_view, "sync_external_face_change"):
            self.batch_people_view.sync_external_face_change(face_ids)
        if self.central_stack.currentWidget() == self.batch_people_view:
            self.batch_people_view._needs_external_refresh = False
            self.batch_people_view.refresh_suggestions(debounce=True)

    def _on_batch_faces_confirmed(self, face_ids: list[int], person_id: int) -> None:
        """Slot called when faces are confirmed in BatchPeopleView."""
        self._batch_people_modified = True
        self.refresh_people_navigation_and_autocomplete()
        if hasattr(self, "people_directory_view") and self.central_stack.currentWidget() == self.people_directory_view:
            self.people_directory_view.refresh_directory()
        if hasattr(self, "grid_view") and getattr(self.grid_view, "person_header_bar", None) is not None:
            if self.grid_view.person_header_bar.isVisible() and self.grid_view.person_header_bar.current_person_id == person_id:
                self.grid_view.person_header_bar.load_person(person_id)
        self.show_status_message(f"✓ Confirmed {len(face_ids)} face(s)", 3500)
        if getattr(self, "find_people_enabled", True):
            self.start_face_detection()

    def _on_batch_faces_rejected(self, face_ids: list[int]) -> None:
        """Slot called when faces are rejected in BatchPeopleView."""
        self._batch_people_modified = True
        self.show_status_message(f"Rejected {len(face_ids)} face suggestion(s)", 3500)

    def _on_batch_faces_ignored(self, face_ids: list[int]) -> None:
        """Slot called when faces are hidden/ignored in BatchPeopleView."""
        self._batch_people_modified = True

    def _on_batch_person_renamed(self, person_id: int, new_name: str) -> None:
        """Slot called when a person is renamed or merged in BatchPeopleView."""
        self._batch_people_modified = True
        self.refresh_people_navigation_and_autocomplete()

    def _on_batch_person_filter_changed(self, person_id: int | None) -> None:
        """Keeps the current batch_people history entry synced when the active person filter changes."""
        if getattr(self, "_is_navigating_history", False):
            return
        if (
            0 <= getattr(self, "_nav_index", -1) < len(getattr(self, "_nav_history", []))
            and self._nav_history[self._nav_index].get("view") == "batch_people"
        ):
            self._nav_history[self._nav_index]["person_id"] = person_id

    def _on_batch_view_person_photos(self, person_id: int) -> None:
        """Exits BatchPeopleView and opens the filtered gallery view for the given person_id."""
        self.central_stack.setCurrentWidget(self.grid_view)
        if hasattr(self, "ribbon"):
            self.ribbon.set_active_mode("grid")
        self._restore_docks_after_batch_people()
        if getattr(self, "_batch_people_modified", False):
            self._batch_people_modified = False
            self.refresh_people_navigation_and_autocomplete()
        self.left_nav_dock.select_person(int(person_id))

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
            if self.central_stack.currentWidget() != self.grid_view:
                self.show_grid_view()
            self._active_map_media_ids = None
            self._active_bbox = None
            self._active_map_location_label = None
            cur_locs = getattr(self.left_nav_dock, "_locations", [])
            req_tup = ((country or "").strip(), (state or "").strip(), (city or "").strip())
            if cur_locs == [req_tup] and self.left_nav_dock._folder_id is None and self.left_nav_dock._album_id is None:
                return
            self.left_nav_dock.select_place(country, state, city, clear_scope=True)

    def filter_gallery_to_map_selection(
        self, media_ids: list[int], location_label: str = ""
    ) -> None:
        if hasattr(self, "left_nav_dock") and self.left_nav_dock is not None:
            self.left_nav_dock.clear_all_filters()
        self._search_text = ""
        if hasattr(self, "scope_bar"):
            self.scope_bar.clear_search_input()
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
        if not backup_path:
            # In-memory stroke (not yet saved to disk): avoid costly DB/thumbnail invalidation
            self._update_revert_button_state()
            return
        self.catalog_repo.record_pixel_edit_backup(media_id, backup_path)
        self._refresh_media_after_pixel_change(media_id)
        fname = Path(backup_path).name if backup_path else "file"
        self.show_status_message(f"✓ Saved changes to '{fname}'", 4000)

    def _on_pixel_edit_saved_as_copy(self, original_media_id: int, new_path: str) -> None:
        p = Path(new_path)
        if p.exists() and self._is_path_in_library(p.parent):
            try:
                self.scan_folder(p.parent, include_subfolders=False)
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
        self.show_status_message(f"Rescanning {len(mids)} photo(s) for faces...", 0)

        worker = FaceDetectionWorker(
            engine=self.engine,
            target_media_ids=mids,
            batch_size=len(mids),
        )
        self._retain_pool_worker(worker)
        worker.signals.media_faces_ready.connect(self._on_targeted_face_scan_ready)
        worker.signals.worker_finished.connect(
            lambda total_faces: self._on_targeted_face_scan_finished(len(mids), total_faces)
        )
        worker.signals.worker_finished.connect(
            lambda *_args, w=worker: self._release_pool_worker(w)
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
        self.show_status_message(f"✓ Face rescan complete ({face_txt})", 4500)
        if self.central_stack.currentWidget() == self.media_viewer and self.media_viewer.current_media_id is not None:
            self.media_viewer.reload_faces(highlight_new=True)
            if hasattr(self, "right_tag_dock") and self.right_tag_dock is not None:
                try:
                    summary = self.catalog_repo.get_selection_summary([self.media_viewer.current_media_id])
                    self.right_tag_dock.bind_selection([self.media_viewer.current_media_id], summary)
                except Exception:
                    pass
        self.refresh_navigation_and_autocomplete({"people", "albums"})

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

    def _on_clipboard_data_changed(self) -> None:
        if getattr(self, "_updating_cut_clipboard", False):
            return
        if hasattr(self, "grid_view") and self.grid_view is not None:
            self.grid_view.set_cut_media_ids(set())

    def cut_selection_to_clipboard(self) -> None:
        """Cuts file paths of currently selected media to the system clipboard (files & text)."""
        self._copy_or_cut_selection(is_cut=True)

    def copy_selection_to_clipboard(self) -> None:
        """Copies selected text if a text widget has focus, otherwise copies selected media files."""
        focus_w = QApplication.focusWidget()
        if isinstance(focus_w, (QLineEdit, QTextEdit, QPlainTextEdit)):
            focus_w.copy()
            return
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
            self._updating_cut_clipboard = True
            try:
                QApplication.clipboard().setMimeData(_make_mime())
            except Exception:
                pass
            finally:
                self._updating_cut_clipboard = False
            if is_cut:
                self.grid_view.set_cut_media_ids(target_ids)
            else:
                self.grid_view.set_cut_media_ids(set())
            action_name = "Cut" if is_cut else "Copied"
            item_w = "file" if len(urls) == 1 else "files"
            self.show_status_message(f"✓ {action_name} {len(urls)} {item_w} to clipboard", 3500)

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
            self.grid_view.set_cut_media_ids(set())
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
        if not in_viewer:
            target_ids = self.expand_stack_targets_if_collapsed(target_ids, action_label="delete")
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
                self.catalog_repo.reconcile_stack_after_delete(mid)
                if getattr(self, "semantic_repo", None) is not None:
                    try:
                        self.semantic_repo.delete_vectors([mid])
                    except Exception:
                        pass
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
                self.show_status_message(f"✓ Set '{fpath.name}' as desktop background", 3500)
                return True
        except Exception as e:
            self.show_status_message(f"⚠ Could not set desktop background: {e}", 4500)
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
            self.show_status_message(f"⚠ Could not open file location: {e}", 4500)
        return False

    def open_folder_in_explorer(self, folder_path: str) -> bool:
        """Opens the given directory in Windows File Explorer (or OS file manager)."""
        try:
            raw = (folder_path or "").strip()
            if not raw:
                return False
            target = Path(raw)
            if not target.is_dir():
                self.show_status_message("⚠ Folder no longer exists on disk.", 4500)
                return False
            import os, subprocess, sys
            norm = os.path.normpath(str(target.resolve()))
            if sys.platform == "win32":
                subprocess.Popen(["explorer.exe", norm])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", norm])
            else:
                subprocess.Popen(["xdg-open", norm])
            return True
        except Exception as e:
            self.show_status_message(f"⚠ Could not open folder: {e}", 4500)
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
                        class _OPENASINFO(ctypes.Structure):
                            _fields_ = [
                                ("pcszFile", ctypes.c_wchar_p),
                                ("pcszClass", ctypes.c_wchar_p),
                                ("oaifInFlags", ctypes.c_int),
                            ]

                        # OAIF_ALLOW_REGISTRATION (0x1) | OAIF_EXEC (0x4) = 0x5
                        info = _OPENASINFO(str(fpath.resolve()), None, 0x5)
                        hwnd = int(self.winId()) if hasattr(self, "winId") else 0
                        ctypes.windll.shell32.SHOpenWithDialog(hwnd, ctypes.byref(info))
                        return True
                    except Exception:
                        pass
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
                if app_choice.startswith("progid:") and sys.platform == "win32":
                    progid = app_choice[len("progid:"):]
                    try:
                        class _SHELLEXECUTEINFO(ctypes.Structure):
                            _fields_ = [
                                ("cbSize", ctypes.c_uint32),
                                ("fMask", ctypes.c_uint32),
                                ("hwnd", ctypes.c_void_p),
                                ("lpVerb", ctypes.c_wchar_p),
                                ("lpFile", ctypes.c_wchar_p),
                                ("lpParameters", ctypes.c_wchar_p),
                                ("lpDirectory", ctypes.c_wchar_p),
                                ("nShow", ctypes.c_int),
                                ("hInstApp", ctypes.c_void_p),
                                ("lpIDList", ctypes.c_void_p),
                                ("lpClass", ctypes.c_wchar_p),
                                ("hkeyClass", ctypes.c_void_p),
                                ("dwHotKey", ctypes.c_uint32),
                                ("hIcon", ctypes.c_void_p),
                                ("hProcess", ctypes.c_void_p),
                            ]

                        sei = _SHELLEXECUTEINFO()
                        sei.cbSize = ctypes.sizeof(_SHELLEXECUTEINFO)
                        sei.fMask = 0x00000001  # SEE_MASK_CLASSNAME
                        sei.hwnd = int(self.winId()) if hasattr(self, "winId") else None
                        sei.lpVerb = "open"
                        sei.lpFile = str(fpath.resolve())
                        sei.lpClass = progid
                        sei.nShow = 1  # SW_SHOWNORMAL
                        if ctypes.windll.shell32.ShellExecuteExW(ctypes.byref(sei)):
                            return True
                    except Exception:
                        pass
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

            self.show_status_message(f"⚠ Could not launch application: {app_choice}", 4500)
            return False
        except Exception as e:
            self.show_status_message(f"⚠ Open with error: {e}", 4500)
            return False
        return False

    def rename_selection(self) -> None:
        """Selects the filename field in the right Information panel (or prompts) for renaming."""
        target_ids = list(self.get_active_target_ids())
        if not target_ids:
            return
        if len(target_ids) == 1:
            if hasattr(self, "right_tag_dock") and self.right_tag_dock is not None and self.right_tag_dock.isVisible():
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
                if getattr(self.media_viewer, "current_media_id", None) == mid:
                    self.media_viewer.current_file_path = new_path
                    if self.central_stack.currentWidget() == self.media_viewer:
                        self.setWindowTitle(f"{new_fname} - Essentials Photo Gallery")
                if hasattr(self, "right_tag_dock") and self.right_tag_dock is not None:
                    summary = self.catalog_repo.get_selection_summary([mid])
                    self.right_tag_dock.bind_selection([mid], summary)
                self.rebuild_and_run_query()
                self.show_status_message(f"✓ Renamed file to '{new_fname}'", 4000)
            except Exception as e:
                QMessageBox.warning(self, "Rename Error", f"Could not rename file: {e}")
        else:
            from src.ui.batch_rename_dialog import (
                BatchRenameDialog,
                execute_batch_rename_plan,
            )

            items: list[dict] = []
            for mid in target_ids:
                try:
                    detail = self.catalog_repo.get_media_detail(mid)
                except Exception:
                    detail = None
                if not detail:
                    continue
                items.append(
                    {
                        "id": int(mid),
                        "folder_path": str(detail.get("folder_path") or ""),
                        "filename": str(detail.get("filename") or ""),
                        "date_taken": detail.get("date_taken"),
                        "mtime": detail.get("mtime"),
                        "camera_model": detail.get("camera_model") or detail.get("model") or "",
                    }
                )
            if not items:
                return
            dlg = BatchRenameDialog(items, parent=self)
            if dlg.exec() != QDialog.Accepted:
                return
            plan = dlg.rename_plan()
            if not plan:
                return
            renamed, errors = execute_batch_rename_plan(
                plan,
                self.catalog_repo,
                media_meta_lookup=self._media_meta_lookup,
            )
            if getattr(self.media_viewer, "current_media_id", None) is not None:
                cur_id = int(self.media_viewer.current_media_id)
                for row in plan:
                    if int(row["media_id"]) == cur_id and row["new_path"].exists():
                        self.media_viewer.current_file_path = row["new_path"]
                        if self.central_stack.currentWidget() == self.media_viewer:
                            self.setWindowTitle(f"{row['new_name']} - Essentials Photo Gallery")
                        break
            self.rebuild_and_run_query()
            if errors:
                QMessageBox.warning(
                    self,
                    "Batch Rename Warning",
                    f"Renamed {renamed} file(s), but encountered errors:\n" + "\n".join(errors[:8]),
                )
            else:
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
                target_w = 310 if mode in ("fine_tune", "depth_light") else 280
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
                "bottom_dock": self.bottom_dock.isVisible() if hasattr(self, "bottom_dock") else False,
                "fix_pane": (
                    self.media_viewer.fix_pane.isVisible()
                    if hasattr(self, "media_viewer") and hasattr(self.media_viewer, "fix_pane")
                    else False
                ),
                "status_bar": self.statusBar().isVisible() if self.statusBar() else True,
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
            if hasattr(self, "bottom_dock"):
                self.bottom_dock.hide()
            if hasattr(self, "media_viewer") and hasattr(self.media_viewer, "fix_pane"):
                self.media_viewer.fix_pane.hide()
            if self.statusBar():
                self.statusBar().hide()
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
            if saved.get("bottom_dock") and hasattr(self, "bottom_dock"):
                self.bottom_dock.show()
            if saved.get("fix_pane") and hasattr(self, "media_viewer") and hasattr(self.media_viewer, "fix_pane"):
                self.media_viewer.fix_pane.show()
            if saved.get("status_bar", True) and self.statusBar():
                self.statusBar().show()
            if saved.get("qat", True) and hasattr(self.ribbon, "qat_container"):
                self.ribbon.qat_container.show()
            if saved.get("scope_bar") and hasattr(self, "scope_bar"):
                self.scope_bar.show()
            if not saved.get("ribbon_collapsed", False) and hasattr(self.ribbon, "set_ribbon_collapsed"):
                self.ribbon.set_ribbon_collapsed(False)
            if hasattr(self.ribbon, "btn_exit_immersive"):
                self.ribbon.btn_exit_immersive.hide()
            self._is_immersive_mode = False

    def toggle_filmstrip_dock(self) -> None:
        """Toggle visibility of the bottom filmstrip dock in Single Image View."""
        self.dispatch_ribbon_action("toggle_filmstrip_pane")

    def dispatch_ribbon_action(self, action_id: str) -> None:
        if action_id in ("toggle_immersive", "exit_immersive"):
            self.toggle_immersive_viewer_mode()
        elif action_id == "add_folder":
            self._prompt_and_add_folder()
        elif action_id == "open_folder_manager":
            self.open_folder_manager()
        elif action_id == "toggle_private_folders":
            checked = bool(getattr(self.ribbon, "act_show_private_folders", None) and self.ribbon.act_show_private_folders.isChecked())
            self.set_show_private_folders(checked)
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
        elif action_id == "photo_grouping_primary":
            if self._expanded_stack_ids:
                self.collapse_all_stacks()
            else:
                records = getattr(self.catalog_model, "_records", []) or []
                has_groups = any(int(getattr(r, "stack_count", 1) or 1) > 1 for r in records)
                if has_groups:
                    self.expand_all_stacks()
                else:
                    self.show_status_message(
                        "Click 'Photo grouping ▾' to enable Auto-group bursts or RAW + JPEG.",
                        3200,
                    )
        elif action_id in ("toggle_auto_group_bursts:1", "toggle_auto_group_bursts:0"):
            self.set_auto_group_bursts(action_id.endswith(":1"))
        elif action_id in ("toggle_auto_group_raw_jpeg:1", "toggle_auto_group_raw_jpeg:0"):
            self.set_auto_group_raw_jpeg(action_id.endswith(":1"))
        elif action_id == "stack_toggle":
            self.toggle_selected_stack_expanded()
        elif action_id == "stack_compare":
            self.open_compare_view()
        elif action_id == "expand_all_stacks":
            self.expand_all_stacks()
        elif action_id == "collapse_all_stacks":
            self.collapse_all_stacks()
        elif action_id == "toggle_all_badges":
            self.toggle_all_badges()
        elif action_id in ("toggle_badge_flags:1", "toggle_badge_flags:0"):
            self.set_badge_visibility(flags=action_id.endswith(":1"))
        elif action_id in ("toggle_badge_favourites:1", "toggle_badge_favourites:0"):
            self.set_badge_visibility(favourites=action_id.endswith(":1"))
        elif action_id in ("toggle_badge_groups:1", "toggle_badge_groups:0"):
            self.set_badge_visibility(groups=action_id.endswith(":1"))
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
        elif action_id == "dup_scan":
            if hasattr(self, "duplicate_finder_view"):
                self.duplicate_finder_view.start_scan()
        elif action_id == "dup_stop":
            if hasattr(self, "duplicate_finder_view"):
                self.duplicate_finder_view.stop_scan()
        elif action_id == "dup_choose_folder":
            if hasattr(self, "duplicate_finder_view"):
                anchor = getattr(self.ribbon, "btn_dup_folder", None)
                self.duplicate_finder_view.open_scope_menu(anchor)
        elif action_id.startswith("dup_toggle_exact:"):
            if hasattr(self, "duplicate_finder_view"):
                self.duplicate_finder_view.chk_exact.setChecked(action_id.endswith(":1"))
        elif action_id.startswith("dup_toggle_visual:"):
            if hasattr(self, "duplicate_finder_view"):
                self.duplicate_finder_view.chk_visual.setChecked(action_id.endswith(":1"))
        elif action_id.startswith("dup_toggle_bursts:"):
            if hasattr(self, "duplicate_finder_view"):
                self.duplicate_finder_view.chk_bursts.setChecked(action_id.endswith(":1"))
        elif action_id.startswith("dup_similarity:"):
            if hasattr(self, "duplicate_finder_view"):
                try:
                    self.duplicate_finder_view.set_similarity_threshold(int(action_id.split(":", 1)[1]))
                except ValueError:
                    pass
        elif action_id == "dup_compare_selected":
            if hasattr(self, "duplicate_finder_view"):
                self.duplicate_finder_view.compare_active_group()
        elif action_id == "dup_zoom_in":
            if hasattr(self, "duplicate_finder_view"):
                self.duplicate_finder_view._zoom_in()
        elif action_id == "dup_zoom_out":
            if hasattr(self, "duplicate_finder_view"):
                self.duplicate_finder_view._zoom_out()
        elif action_id.startswith("dup_rule:"):
            if hasattr(self, "duplicate_finder_view"):
                self.duplicate_finder_view.set_rule_by_name(action_id.split(":", 1)[1])
                self.duplicate_finder_view._apply_rule_to_all_groups()
        elif action_id.startswith("dup_sort:"):
            if hasattr(self, "duplicate_finder_view"):
                self.duplicate_finder_view.set_sort_by_key(action_id.split(":", 1)[1])
        elif action_id.startswith("dup_group_by:"):
            if hasattr(self, "duplicate_finder_view"):
                self.duplicate_finder_view.set_group_by_key(action_id.split(":", 1)[1])
        elif action_id.startswith("dup_meta_mode:"):
            if hasattr(self, "duplicate_finder_view"):
                self.duplicate_finder_view.set_metadata_mode(action_id.split(":", 1)[1])
        elif action_id.startswith("dup_toggle_tag_pane:"):
            show_pane = action_id.endswith(":1")
            self.right_tag_dock.setVisible(show_pane)
            if show_pane:
                self.right_tag_dock.show()
                self.resizeDocks([self.right_tag_dock], [270], Qt.Horizontal)
                if hasattr(self, "duplicate_finder_view"):
                    sel_mid = self.duplicate_finder_view.selected_media_id()
                    if sel_mid is not None:
                        self._on_duplicates_media_selected(sel_mid)
        elif action_id == "dup_apply_all":
            if hasattr(self, "duplicate_finder_view"):
                self.duplicate_finder_view._apply_rule_to_all_groups()
        elif action_id.startswith("dup_merge_meta:"):
            if hasattr(self, "duplicate_finder_view"):
                self.duplicate_finder_view.set_merge_metadata_enabled(action_id.endswith(":1"))
        elif action_id == "dup_backup":
            if hasattr(self, "duplicate_finder_view"):
                self.duplicate_finder_view._move_marked_to_backup()
        elif action_id == "dup_recycle":
            if hasattr(self, "duplicate_finder_view"):
                self.duplicate_finder_view._move_marked_to_recycle_bin()
        elif action_id == "dup_close":
            self.show_grid_view()
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
                has_unflagged = any(flags_map.get(mid, 0) != 1 for mid in target_ids)
                self.apply_flag_to_selection(1 if has_unflagged else 0)
        elif action_id == "flag_pick":
            self.apply_flag_to_selection(1)
        elif action_id == "flag_reject":
            self.apply_flag_to_selection(-1)
        elif action_id == "flag_unflag":
            self.apply_flag_to_selection(0)
        elif action_id == "toggle_favourite":
            self.toggle_favourite_selection()
        elif action_id == "select_rejected":
            self.select_all_rejected()
        elif action_id == "delete_rejected":
            self.delete_all_rejected()
        elif action_id == "import_picasa":
            self.open_picasa_import_dialog()
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
            ws = getattr(self, "_active_create_workspace", None)
            if ws is not None:
                out_p = ws.save_result()
                if out_p is not None:
                    self.close_create_workspace(force=True, saved_path=out_p)
            elif self.central_stack.currentWidget() == self.media_viewer:
                self.media_viewer.save_current_image_with_backup()
        elif action_id == "toggle_filmstrip_pane":
            new_fs = not bool(getattr(self, "_filmstrip_viewer_enabled", False))
            self._filmstrip_viewer_enabled = new_fs
            if hasattr(self.ribbon, "btn_toggle_filmstrip"):
                self.ribbon.btn_toggle_filmstrip.blockSignals(True)
                self.ribbon.btn_toggle_filmstrip.setChecked(new_fs)
                self.ribbon.btn_toggle_filmstrip.blockSignals(False)
            if hasattr(self, "btn_status_filmstrip"):
                self.btn_status_filmstrip.blockSignals(True)
                self.btn_status_filmstrip.setChecked(new_fs)
                self.btn_status_filmstrip.blockSignals(False)
            if self.central_stack.currentWidget() == self.media_viewer and hasattr(self, "bottom_dock"):
                self.bottom_dock.refresh_items()
                self.bottom_dock.set_filmstrip_mode()
                self.bottom_dock.set_active_media_id(getattr(self.media_viewer, "current_media_id", None))
                if hasattr(self.bottom_dock, "animate_visible"):
                    self.bottom_dock.animate_visible(new_fs)
                else:
                    self.bottom_dock.setVisible(new_fs)
                if hasattr(self.media_viewer, "_update_filmstrip_toggle"):
                    self.media_viewer._update_filmstrip_toggle()
                self.media_viewer.setFocus()
        elif action_id == "create_ws_save":
            ws = getattr(self, "_active_create_workspace", None)
            if ws is not None:
                out_p = ws.save_result()
                if out_p is not None:
                    self.close_create_workspace(force=True, saved_path=out_p)
        elif action_id == "create_ws_cancel":
            self.close_create_workspace()
        elif action_id.startswith("create_ws:"):
            ws = getattr(self, "_active_create_workspace", None)
            if ws is not None:
                ws.handle_ribbon_action(action_id.split(":", 1)[1])
                self._sync_create_ws_ribbon_states()
                self._update_undo_redo_enabled()
        elif action_id == "save_as_copy":
            if self.central_stack.currentWidget() == self.media_viewer:
                saved = self.media_viewer.save_as_copy()
                if saved:
                    self.show_status_message(f"✓ Saved copy: {Path(saved).name}", 4000)
        elif action_id == "save_as_copy_original":
            if self.central_stack.currentWidget() == self.media_viewer:
                saved = self.media_viewer.save_as_copy(forced_target="original")
                if saved:
                    self.show_status_message(f"✓ Quick saved copy in original location: {Path(saved).name}", 4000)
        elif action_id == "save_as_copy_choose_location":
            if self.central_stack.currentWidget() == self.media_viewer:
                saved = self.media_viewer.save_as_copy(forced_target="choose")
                if saved:
                    self.show_status_message(f"✓ Saved copy: {Path(saved).name}", 4000)
        elif action_id == "save_as_resized_copy":
            if self.central_stack.currentWidget() == self.media_viewer:
                generated = self.media_viewer.save_as_resized_copy()
                if generated:
                    for d in {p.parent for p in generated}:
                        self.scan_folder(d, include_subfolders=False)
                    self.show_status_message(f"✓ Resized and saved {len(generated)} photo(s)", 4000)
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
        elif action_id.startswith("compare_af_points:"):
            enabled = action_id.endswith(":1")
            found = self.compare_view.set_af_overlay(enabled)
            if enabled and found == 0:
                self.show_status_message("No camera AF focus point metadata in compared photos", 3000)
        elif action_id.startswith("compare_clipping_warnings:"):
            self.compare_view.set_clipping_warnings(action_id.endswith(":1"))
        elif action_id.startswith("compare_auto_advance:"):
            self.compare_view.set_auto_advance(action_id.endswith(":1"))
        elif action_id.startswith("compare_dark_backdrop:"):
            self.compare_view.set_dark_mode(action_id.endswith(":1"))
        elif action_id.startswith("compare_toggle_tag_pane:"):
            show_pane = action_id.endswith(":1")
            self.right_tag_dock.setVisible(show_pane)
            if show_pane:
                self.right_tag_dock.show()
                self.resizeDocks([self.right_tag_dock], [270], Qt.Horizontal)
                self._on_compare_focused_media_changed(self.compare_view.focused_media_id())
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
            self.show_map_view()
        elif action_id == "focus_geotag_box":
            self.right_tag_dock.set_mode("info")
            self.right_tag_dock.show()
            self.right_tag_dock.raise_()
            try:
                self.resizeDocks([self.right_tag_dock], [280], Qt.Horizontal)
            except Exception:
                pass
            self.right_tag_dock.focus_geotag_input()
        elif action_id == "adjust_geotag_pin":
            self.open_adjust_location_dialog()
        elif action_id == "match_gps_locations":
            self.open_gps_match_dialog()
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
            angle = float(self.media_viewer.apply_auto_straighten() or 0.0)
            if abs(angle) >= 0.05:
                self.show_status_message(f"✓ Auto straightened ({angle:+.1f}°)", 3000)
            else:
                self.show_status_message("Photo is already level (0.0°)", 3000)
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
            ws = getattr(self, "_active_create_workspace", None)
            if ws is not None:
                ws.zoom_in()
            elif self.central_stack.currentWidget() == self.media_viewer:
                self.media_viewer.zoom_in()
            elif hasattr(self, "duplicate_finder_view") and self.central_stack.currentWidget() == self.duplicate_finder_view:
                self.duplicate_finder_view._zoom_in()
            else:
                self.slider_status_zoom.setValue(
                    min(self.slider_status_zoom.maximum(), self.slider_status_zoom.value() + 25)
                )
        elif action_id == "zoom_out":
            ws = getattr(self, "_active_create_workspace", None)
            if ws is not None:
                ws.zoom_out()
            elif self.central_stack.currentWidget() == self.media_viewer:
                self.media_viewer.zoom_out()
            elif hasattr(self, "duplicate_finder_view") and self.central_stack.currentWidget() == self.duplicate_finder_view:
                self.duplicate_finder_view._zoom_out()
            else:
                self.slider_status_zoom.setValue(
                    max(self.slider_status_zoom.minimum(), self.slider_status_zoom.value() - 25)
                )
        elif action_id in ("viewer_zoom_actual", "zoom_actual"):
            ws = getattr(self, "_active_create_workspace", None)
            if ws is not None:
                ws.zoom_actual()
            elif self.central_stack.currentWidget() == self.media_viewer:
                self.media_viewer.set_zoom_mode("100")
        elif action_id in ("viewer_zoom_fit", "zoom_fit"):
            ws = getattr(self, "_active_create_workspace", None)
            if ws is not None:
                ws.zoom_fit()
            elif self.central_stack.currentWidget() == self.media_viewer:
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
        elif action_id == "show_shortcuts":
            self.open_keyboard_shortcuts_dialog(0)
        elif action_id in ("about_photo_gallery", "about_app"):
            self._open_about_dialog()

    def open_keyboard_shortcuts_dialog(self, initial_tab: int = 0) -> None:
        from src.ui.shortcuts_dialog import KeyboardShortcutsDialog

        try:
            tags = [str(t["full_path"]) for t in self.catalog_repo.get_tag_tree()]
        except Exception:
            tags = []
        dlg = KeyboardShortcutsDialog(
            available_tags=tags,
            is_dark=(getattr(self, "current_theme", "wlpg_aero") == "dark"),
            initial_tab=initial_tab,
            parent=self,
        )
        dlg.exec()

    def _on_quick_tag_hotkey(self, slot: int) -> None:
        if getattr(self, "_active_create_workspace", None) is not None:
            return
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit)):
            return
        from src.ui.shortcuts_dialog import load_quick_tag_hotkeys

        tag_name = load_quick_tag_hotkeys().get(int(slot))
        if not tag_name:
            self.show_status_message(
                f"Alt+{slot} is not assigned — press F1 to configure Quick-Tag hotkeys.",
                3500,
            )
            return
        qp = getattr(self, "_quick_preview_dialog", None)
        if qp is not None and qp.isVisible() and getattr(qp, "_current_media_id", None) is not None:
            mid = int(qp._current_media_id)
            tid = None
            existing = self.catalog_repo.find_tag_by_leaf_name(tag_name)
            if existing is not None:
                tid = int(existing["id"])
            if tid is None:
                tid = self.catalog_repo.create_tag(tag_name)
            self.catalog_repo.assign_tag([mid], tid)
            self.show_status_message(f"✓ Tagged photo with '{tag_name}' (Alt+{slot})", 3000)
            self._schedule_tag_nav_refresh({"tags", "albums"})
            return
        target_ids = self.get_active_target_ids()
        if target_ids:
            self.on_selection_tag_added(tag_name)

    def _prewarm_semantic_matrix_async(self) -> None:
        """Pre-loads the semantic vector matrix on a background thread so first search is sub-5ms."""
        if (
            getattr(self, "_is_shutting_down", False)
            or getattr(self, "semantic_repo", None) is None
            or getattr(self, "_semantic_matrix_prewarmed", False)
        ):
            return
        self._semantic_matrix_prewarmed = True
        repo = self.semantic_repo

        def _worker() -> None:
            try:
                repo.load_matrix()
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True, name="SemanticMatrixWarmup").start()

    def _refresh_subjects_nav_counts(self) -> None:
        if (
            getattr(self, "_is_shutting_down", False)
            or getattr(self, "semantic_repo", None) is None
            or getattr(self, "semantic_engine", None) is None
            or not hasattr(self, "left_nav_dock")
        ):
            return
        try:
            show_subj = bool(
                getattr(self, "nav_show_subjects", False)
                or getattr(self.left_nav_dock, "_show_subjects", False)
            )
            if not show_subj:
                return
            if getattr(self, "_active_semantic_worker", None) is not None:
                now_ts = time.monotonic()
                if (now_ts - getattr(self, "_last_subjects_count_ts", 0.0)) < 5.0 and getattr(
                    self.left_nav_dock, "_subject_counts", None
                ):
                    return
                self._last_subjects_count_ts = now_ts
            subj_vecs = self.semantic_repo.get_subject_vectors()
            if not subj_vecs:
                if self.semantic_engine.is_available():
                    subj_vecs = self.semantic_engine.ensure_subject_vectors(self.semantic_repo)
                if not subj_vecs:
                    if self.semantic_engine.is_available() and getattr(self, "_active_semantic_worker", None) is None:
                        self._start_semantic_worker()
                    return

            engine = getattr(self, "engine", None)
            if engine is not None and "PYTEST_CURRENT_TEST" not in os.environ:
                if getattr(self, "_subjects_count_inflight", False):
                    self._subjects_count_dirty = True
                    return
                self._subjects_count_inflight = True
                self._subjects_count_dirty = False
                include_private = bool(
                    getattr(getattr(self, "catalog_repo", None), "include_private", False)
                )
                worker = _SubjectsCountWorker(
                    engine,
                    self.semantic_repo,
                    self.semantic_engine,
                    subj_vecs,
                    include_private=include_private,
                )
                self._active_subjects_worker = worker

                def _on_counts(counts: dict[str, int]) -> None:
                    if getattr(self, "_is_shutting_down", False):
                        return
                    try:
                        if hasattr(self, "left_nav_dock"):
                            self.left_nav_dock.refresh_subjects(counts)
                    except Exception:
                        pass

                def _on_done() -> None:
                    self._subjects_count_inflight = False
                    if getattr(self, "_active_subjects_worker", None) is worker:
                        self._active_subjects_worker = None
                    if getattr(self, "_subjects_count_dirty", False) and not getattr(
                        self, "_is_shutting_down", False
                    ):
                        self._subjects_count_dirty = False
                        QTimer.singleShot(0, self._refresh_subjects_nav_counts)

                self._retain_pool_worker(worker)
                worker.signals.counts_ready.connect(_on_counts)
                worker.signals.finished.connect(_on_done)
                worker.signals.finished.connect(
                    lambda *_args, w=worker: self._release_pool_worker(w)
                )
                pool = getattr(self, "_subjects_count_pool", None)
                if pool is None:
                    pool = QThreadPool(self)
                    pool.setMaxThreadCount(1)
                    self._subjects_count_pool = pool
                pool.start(worker)
                return

            scope_ids = self._visible_non_private_media_ids()
            counts = self.semantic_engine.compute_subject_counts(
                self.semantic_repo, subj_vecs, scope_ids=scope_ids
            )
            self.left_nav_dock.refresh_subjects(counts)
        except Exception:
            pass

    def _ensure_semantic_addon_with_prompt(self) -> bool:
        """
        Ensure MobileCLIP-S1 ONNX add-on is installed; if missing, prompt the user
        to download it (~85 MB) or open Options -> Add-ons.
        """
        if getattr(self, "semantic_engine", None) is not None and self.semantic_engine.is_available():
            return True
        from src.ui.dialogs import prompt_and_download_semantic_addon

        ok = prompt_and_download_semantic_addon(
            self,
            on_more_options=lambda: self._open_settings_dialog(initial_tab="addons"),
        )
        if ok:
            self._start_semantic_worker(force=True)
            self._refresh_subjects_nav_counts()
            self.show_status_message(
                "✓ MobileCLIP-S1 installed — indexing photos for AI Subjects...",
                4500,
            )
            return True
        return False

    def _on_subjects_section_clicked(self) -> None:
        if getattr(self, "semantic_engine", None) is not None and not self.semantic_engine.is_available():
            self._ensure_semantic_addon_with_prompt()
        else:
            self._refresh_subjects_nav_counts()

    def _on_subjects_selected(self, subjects: Sequence[tuple[str, str]]) -> None:
        """Filters gallery by one or more selected AI Subject presets (union ranked by best match)."""
        subj_list = [(str(k), str(l)) for k, l in subjects if k]
        if not subj_list:
            return
        if len(subj_list) == 1:
            self._on_subject_selected(subj_list[0][0], subj_list[0][1])
            return
        if getattr(self, "semantic_repo", None) is None or getattr(self, "semantic_engine", None) is None:
            return
        from src.core.semantic_engine import SUBJECT_PRESETS

        if not self.semantic_engine.is_available():
            if self._ensure_semantic_addon_with_prompt():
                joined = " + ".join(l for _, l in subj_list)
                self.show_status_message(
                    f"✓ MobileCLIP-S1 installed — indexing photos for '{joined}'...",
                    4500,
                )
            return

        joined_label = " + ".join(l for _, l in subj_list)
        if self.semantic_repo.count_indexed() == 0:
            self._start_semantic_worker(force=True)
            self._active_subject_ids = []
            self._active_subject_label = joined_label
            self._update_scope_bar_visibility()
            self.rebuild_and_run_query()
            self.show_status_message(
                f"Indexing photos for AI Subjects '{joined_label}' in the background...",
                4500,
            )
            return

        subj_vecs = self.semantic_repo.get_subject_vectors()
        scope_ids = self._visible_non_private_media_ids()
        best_scores: dict[int, float] = {}
        for subject_key, _lbl in subj_list:
            qvec = subj_vecs.get(subject_key)
            if qvec is None:
                for key, _plbl, prompts, _thr in SUBJECT_PRESETS:
                    if key == subject_key and self.semantic_engine.is_available():
                        vecs = [self.semantic_engine.encode_text(p) for p in prompts]
                        valid_v = [v for v in vecs if v is not None]
                        if valid_v:
                            qvec = sum(valid_v) / float(len(valid_v))
                            self.semantic_repo.put_subject_vectors({subject_key: qvec})
                        break
            if qvec is None:
                continue
            thr = next((t for k, _l, _p, t in SUBJECT_PRESETS if k == subject_key), 0.24)
            hits = self.semantic_engine.query_by_vector(
                self.semantic_repo, qvec, threshold=thr, limit=2000, scope_ids=scope_ids
            )
            for mid, sc in hits:
                m_int = int(mid)
                if m_int not in best_scores or float(sc) > best_scores[m_int]:
                    best_scores[m_int] = float(sc)

        ordered = [mid for mid, _sc in sorted(best_scores.items(), key=lambda kv: -kv[1])]
        self._active_subject_ids = ordered
        self._active_subject_label = joined_label
        if self.central_stack.currentWidget() != self.grid_view:
            self.show_grid_view()
        self._update_scope_bar_visibility()
        self.rebuild_and_run_query()
        if ordered and not getattr(self.left_nav_dock, "_subject_counts", None):
            self._refresh_subjects_nav_counts()
        self.show_status_message(f"✓ Subjects '{joined_label}': {len(ordered):,} match(es)", 3500)

    def _on_subject_selected(self, subject_key: str, label: str) -> None:
        if getattr(self, "semantic_repo", None) is None or getattr(self, "semantic_engine", None) is None:
            return
        from src.core.semantic_engine import SUBJECT_PRESETS

        if not self.semantic_engine.is_available():
            if self._ensure_semantic_addon_with_prompt():
                self.show_status_message(
                    f"✓ MobileCLIP-S1 installed — indexing photos for '{label}'...",
                    4500,
                )
            return

        if self.semantic_repo.count_indexed() == 0:
            self._start_semantic_worker(force=True)
            self._active_subject_ids = []
            self._active_subject_label = label
            self._update_scope_bar_visibility()
            self.rebuild_and_run_query()
            self.show_status_message(
                f"Indexing photos for AI Subject '{label}' in the background...",
                4500,
            )
            return

        subj_vecs = self.semantic_repo.get_subject_vectors()
        qvec = subj_vecs.get(subject_key)
        if qvec is None:
            for key, _lbl, prompts, _thr in SUBJECT_PRESETS:
                if key == subject_key and self.semantic_engine.is_available():
                    vecs = [self.semantic_engine.encode_text(p) for p in prompts]
                    valid_v = [v for v in vecs if v is not None]
                    if valid_v:
                        qvec = sum(valid_v) / float(len(valid_v))
                        self.semantic_repo.put_subject_vectors({subject_key: qvec})
                    break
        if qvec is None:
            self.show_status_message(f"Subject preset '{label}' vector unavailable.", 3500)
            return
        thr = next((t for k, _l, _p, t in SUBJECT_PRESETS if k == subject_key), 0.24)
        scope_ids = self._visible_non_private_media_ids()
        hits = self.semantic_engine.query_by_vector(
            self.semantic_repo, qvec, threshold=thr, limit=2000, scope_ids=scope_ids
        )
        self._active_subject_ids = [mid for mid, _sc in hits]
        self._active_subject_label = label
        if self.central_stack.currentWidget() != self.grid_view:
            self.show_grid_view()
        self._update_scope_bar_visibility()
        self.rebuild_and_run_query()
        if self._active_subject_ids and int(
            getattr(self.left_nav_dock, "_subject_counts", {}).get(subject_key, 0)
        ) == 0:
            self._refresh_subjects_nav_counts()
        self.show_status_message(f"✓ Subject '{label}': {len(self._active_subject_ids):,} match(es)", 3500)

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
            if self.media_viewer.current_media_id is not None:
                try:
                    det = self.catalog_repo.get_media_detail(self.media_viewer.current_media_id)
                    for k in ("caption", "date_taken", "mtime", "confirmed_people", "people", "face_regions"):
                        if k in det:
                            d[k] = det[k]
                except Exception:
                    pass
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
        """Checks whether folder_path is an existing library folder or within one (pure string check, zero disk I/O)."""
        try:
            target_norm = os.path.normcase(os.path.normpath(str(folder_path))).rstrip("\\/")
            sep = os.sep
            for f in self.catalog_repo.get_folders():
                f_norm = os.path.normcase(os.path.normpath(str(f["path"]))).rstrip("\\/")
                if target_norm == f_norm or target_norm.startswith(f_norm + sep):
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

    def _on_filmstrip_media_selected(self, media_id: int) -> None:
        if getattr(self, "_active_create_workspace", None) is not None:
            return
        if self.central_stack.currentWidget() == self.media_viewer:
            cur_id = getattr(self.media_viewer, "current_media_id", None)
            if cur_id is not None and int(cur_id) == int(media_id):
                return
            if self.media_viewer.has_unsaved_adjustments():
                if not self.media_viewer.prompt_save_changes_if_needed("navigate"):
                    self.bottom_dock.set_active_media_id(cur_id)
                    return
            self.open_single_viewer(int(media_id))

    def _on_bottom_dock_close_requested(self) -> None:
        if getattr(self, "_active_create_workspace", None) is not None:
            self.bottom_dock.hide()
            return
        self._filmstrip_viewer_enabled = False
        self.bottom_dock.hide()
        if hasattr(self.ribbon, "btn_toggle_filmstrip"):
            self.ribbon.btn_toggle_filmstrip.blockSignals(True)
            self.ribbon.btn_toggle_filmstrip.setChecked(False)
            self.ribbon.btn_toggle_filmstrip.blockSignals(False)
        if hasattr(self, "btn_status_filmstrip"):
            self.btn_status_filmstrip.blockSignals(True)
            self.btn_status_filmstrip.setChecked(False)
            self.btn_status_filmstrip.blockSignals(False)

    def _sync_create_ws_ribbon_states(self) -> None:
        ws = getattr(self, "_active_create_workspace", None)
        if ws is None:
            return
        if hasattr(self.ribbon, "sync_create_studio_action_states"):
            self.ribbon.sync_create_studio_action_states(ws.ribbon_action_states())

    def open_create_workspace(self, workspace: Any) -> None:
        """Mount a Create Studio workspace inside MainWindow with contextual ribbon and panes."""
        if getattr(self, "_active_create_workspace", None) is not None:
            if not self.close_create_workspace():
                workspace.deleteLater()
                return

        cur_widget = self.central_stack.currentWidget()
        self._create_ws_snapshot = {
            "central_widget": cur_widget,
            "ribbon_tab": self.ribbon.tab_widget.currentWidget(),
            "nav_visible": not self.left_nav_dock.isHidden(),
            "tag_visible": not self.right_tag_dock.isHidden(),
            "tag_mode": self.right_tag_dock.current_mode(),
            "bottom_visible": not self.bottom_dock.isHidden() if hasattr(self, "bottom_dock") else False,
            "window_title": self.windowTitle(),
            "was_viewer": cur_widget == self.media_viewer,
            "zoom_slider_val": self.slider_status_zoom.value() if hasattr(self, "slider_status_zoom") else 150,
        }

        self._active_create_workspace = workspace
        workspace.apply_theme_mode(getattr(self, "current_theme", "wlpg_aero"))

        canvas_w = workspace.canvas_widget()
        if self.central_stack.indexOf(canvas_w) < 0:
            self.central_stack.addWidget(canvas_w)
        self.central_stack.setCurrentWidget(canvas_w)

        self.left_nav_dock.hide()
        if hasattr(self, "scope_bar") and self.scope_bar is not None:
            self.scope_bar.hide()
        for btn in (
            getattr(self, "btn_status_prev", None),
            getattr(self, "btn_status_next", None),
            getattr(self, "btn_status_filmstrip", None),
        ):
            if btn is not None:
                btn.setVisible(False)
        has_canvas_zoom = getattr(workspace, "workspace_id", "") != "cinematic"
        if hasattr(self, "btn_status_fit") and self.btn_status_fit is not None:
            self.btn_status_fit.setVisible(has_canvas_zoom)
        if has_canvas_zoom and hasattr(workspace, "zoom_state_changed"):
            workspace.zoom_state_changed.connect(self._sync_status_zoom_from_create_workspace)
            self._sync_status_zoom_from_create_workspace(1.0, getattr(workspace, "_is_fit_mode", True))

        inspector_w = workspace.inspector_widget()
        if inspector_w is not None:
            self.right_tag_dock.set_custom_inspector(inspector_w, workspace.display_title)
            self.right_tag_dock.show()
            try:
                self.resizeDocks([self.right_tag_dock], [310], Qt.Horizontal)
            except Exception:
                pass
        else:
            self.right_tag_dock.clear_custom_inspector()
            self.right_tag_dock.hide()

        tray_w = workspace.bottom_tray_widget()
        if hasattr(self, "bottom_dock") and self.bottom_dock is not None:
            if tray_w is not None:
                self.bottom_dock.set_tray_widget(tray_w, workspace.tray_title())
                self.bottom_dock.show()
            else:
                self.bottom_dock.clear_tray_widget()
                self.bottom_dock.hide()

        self.ribbon.configure_create_studio_tab(workspace.display_title, workspace.ribbon_actions())
        self.ribbon.set_active_mode("create_studio")
        self._sync_create_ws_ribbon_states()

        workspace.saved.connect(lambda p: self.close_create_workspace(force=True, saved_path=p))
        workspace.cancelled.connect(self.close_create_workspace)
        workspace.status_message.connect(self.show_status_message)
        workspace.ribbon_state_changed.connect(self._sync_create_ws_ribbon_states)
        workspace.undo_redo_state_changed.connect(lambda _u, _r: self._update_undo_redo_enabled())

        title = f"{workspace.display_title} - Essentials Photo Gallery"
        self.setWindowTitle(title)
        self._update_undo_redo_enabled()
        canvas_w.setFocus()

    def close_create_workspace(self, *, force: bool = False, saved_path: Path | None = None) -> bool:
        """Close the active Create Studio workspace, prompting if unsaved and restoring prior state."""
        ws = getattr(self, "_active_create_workspace", None)
        if ws is None:
            return True

        effective_saved = saved_path if saved_path is not None else getattr(ws, "saved_file_path", None)
        if not force and effective_saved is None and ws.has_unsaved_changes():
            res = QMessageBox.question(
                self,
                ws.display_title,
                f"Save changes before leaving {ws.display_title}?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if res == QMessageBox.StandardButton.Cancel:
                return False
            if res == QMessageBox.StandardButton.Save:
                out = ws.save_result()
                if self._active_create_workspace is not ws:
                    return True
                if out is None:
                    return False
                effective_saved = out

        tool_label = getattr(ws, "tool_save_label", "creation")
        snap = self._create_ws_snapshot or {}

        try:
            ws.cleanup()
        except Exception:
            pass

        self._active_create_workspace = None
        self._create_ws_snapshot = None

        self.right_tag_dock.clear_custom_inspector()
        if hasattr(self, "bottom_dock") and self.bottom_dock is not None:
            self.bottom_dock.clear_tray_widget()
            self.bottom_dock.hide()

        canvas_w = ws.canvas_widget()
        if self.central_stack.indexOf(canvas_w) >= 0:
            self.central_stack.removeWidget(canvas_w)
        ws.deleteLater()

        was_viewer = bool(snap.get("was_viewer", False))
        target_central = snap.get("central_widget")
        if target_central is not None and self.central_stack.indexOf(target_central) >= 0:
            self.central_stack.setCurrentWidget(target_central)
        else:
            self.central_stack.setCurrentWidget(self.grid_view)
            was_viewer = False

        if was_viewer:
            self.ribbon.set_active_mode("viewer")
            self.right_tag_dock.set_view_mode("viewer")
            for btn in (
                getattr(self, "btn_status_prev", None),
                getattr(self, "btn_status_next", None),
                getattr(self, "btn_status_filmstrip", None),
                getattr(self, "btn_status_fit", None),
            ):
                if btn is not None:
                    btn.setVisible(True)
            self._update_status_zoom_toggle_button()
            if hasattr(self, "bottom_dock") and self.bottom_dock is not None:
                self.bottom_dock.refresh_items()
                self.bottom_dock.set_filmstrip_mode()
                self.bottom_dock.set_active_media_id(getattr(self.media_viewer, "current_media_id", None))
                show_fs = bool(getattr(self, "_filmstrip_viewer_enabled", False) and snap.get("bottom_visible", False))
                self.bottom_dock.setVisible(show_fs)
                if hasattr(self.ribbon, "btn_toggle_filmstrip"):
                    self.ribbon.btn_toggle_filmstrip.blockSignals(True)
                    self.ribbon.btn_toggle_filmstrip.setChecked(show_fs)
                    self.ribbon.btn_toggle_filmstrip.blockSignals(False)
                if hasattr(self, "btn_status_filmstrip"):
                    self.btn_status_filmstrip.blockSignals(True)
                    self.btn_status_filmstrip.setChecked(show_fs)
                    self.btn_status_filmstrip.blockSignals(False)
        else:
            self.ribbon.set_active_mode("grid")
            self.right_tag_dock.set_view_mode("grid")
            if hasattr(self, "btn_status_fit") and self.btn_status_fit is not None:
                self.btn_status_fit.setVisible(False)
            prev_z = snap.get("zoom_slider_val")
            if prev_z is not None and hasattr(self, "slider_status_zoom"):
                self.slider_status_zoom.blockSignals(True)
                self.slider_status_zoom.setValue(int(prev_z))
                self.slider_status_zoom.blockSignals(False)
            try:
                self._update_scope_bar_visibility()
            except Exception:
                pass

        prev_tab = snap.get("ribbon_tab")
        if prev_tab is not None and prev_tab is not getattr(self.ribbon, "tab_create_studio", None):
            idx = self.ribbon.tab_widget.indexOf(prev_tab)
            if idx >= 0 and self.ribbon.tab_widget.tabBar().isTabVisible(idx):
                self.ribbon.tab_widget.setCurrentWidget(prev_tab)
        if self.ribbon.tab_widget.currentWidget() is getattr(self.ribbon, "tab_create_studio", None):
            fallback_tab = self.ribbon.tab_edit if was_viewer else self.ribbon.tab_home
            self.ribbon.tab_widget.setCurrentWidget(fallback_tab)
        self.ribbon.tab_widget.tabBar().updateGeometry()
        self.ribbon.tab_widget.updateGeometry()
        self.ribbon.tab_widget.tabBar().update()

        nav_vis = bool(snap.get("nav_visible", not was_viewer))
        self.left_nav_dock.setVisible(nav_vis)
        if nav_vis:
            self.left_nav_dock.show()
            try:
                self.resizeDocks([self.left_nav_dock], [230], Qt.Horizontal)
            except Exception:
                pass
        if hasattr(self.ribbon, "btn_toggle_nav_pane"):
            self.ribbon.btn_toggle_nav_pane.blockSignals(True)
            self.ribbon.btn_toggle_nav_pane.setChecked(nav_vis)
            self.ribbon.btn_toggle_nav_pane.blockSignals(False)

        prev_mode = str(snap.get("tag_mode", "info") or "info")
        if prev_mode not in ("info", "fine_tune", "depth_light"):
            prev_mode = "info"
        self.right_tag_dock.set_mode(prev_mode)
        tag_vis = bool(snap.get("tag_visible", True))
        self.right_tag_dock.setVisible(tag_vis)
        if tag_vis:
            self.right_tag_dock.show()
            try:
                self.resizeDocks([self.right_tag_dock], [310], Qt.Horizontal)
            except Exception:
                pass
        if hasattr(self.ribbon, "btn_toggle_tag_pane"):
            self.ribbon.btn_toggle_tag_pane.blockSignals(True)
            self.ribbon.btn_toggle_tag_pane.setChecked(tag_vis)
            self.ribbon.btn_toggle_tag_pane.blockSignals(False)

        restored_title = str(snap.get("window_title") or "Essentials Photo Gallery")
        self.setWindowTitle(restored_title)
        self._update_undo_redo_enabled()
        self._update_status_grid_button()

        if effective_saved is not None:
            self._on_creation_saved(Path(effective_saved), tool_label)
        return True

    def open_panorama_dialog(self) -> None:
        """Launches the integrated Microsoft ICE-Style Panorama Stitcher Workspace."""
        from src.ui.create_studio.panorama_workspace import PanoramaWorkspace

        paths, _ = self._get_selected_photo_paths_and_details()
        if len(paths) < 2:
            QMessageBox.information(
                self,
                "Panorama",
                "Please select 2 or more overlapping photos to stitch a panorama.",
            )
            return

        ws = PanoramaWorkspace(paths, self)
        self.open_create_workspace(ws)

    def open_collage_dialog(self) -> None:
        """Launches the integrated Interactive Collage & 'Then & Now' Workspace."""
        from src.ui.create_studio.collage_workspace import CollageWorkspace

        _, details = self._get_selected_photo_paths_and_details()
        if not details:
            QMessageBox.information(
                self,
                "Collage",
                "Please select 1 to 30 photos to generate a collage.",
            )
            return

        if len(details) > 30:
            total_sel = len(details)
            details = details[:30]
            self.show_status_message(
                f"Collage capped at 30 photos (using first 30 of {total_sel} selected).",
                4500,
            )

        ws = CollageWorkspace(details, self)
        self.open_create_workspace(ws)

    def open_photo_fuse_dialog(self) -> None:
        """Launches the integrated Photo Fuse Best-Face Group Shot Workspace."""
        from src.ui.create_studio.photo_fuse_workspace import PhotoFuseWorkspace

        paths, _ = self._get_selected_photo_paths_and_details()
        if len(paths) < 2:
            QMessageBox.information(
                self,
                "Photo Fuse",
                "Please select 2 to 6 burst group photos to fuse faces.",
            )
            return

        ws = PhotoFuseWorkspace(paths, self)
        self.open_create_workspace(ws)

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
        """Launches the integrated Colour Pop Workspace."""
        from src.ui.create_studio.colour_pop_workspace import ColourPopWorkspace

        p, bgr = self._get_single_selected_photo_and_bgr()
        if bgr is None:
            QMessageBox.information(self, "Colour Pop", "Please select a photo to apply Colour Pop.")
            return

        ws = ColourPopWorkspace(bgr, p, self)
        self.open_create_workspace(ws)

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
        """Launches the integrated 3D Cinematic Motion Workspace."""
        from src.ui.create_studio.cinematic_workspace import CinematicWorkspace

        paths, details = self._get_selected_photo_paths_and_details()
        bgrs: list[Any] = []
        valid_paths: list[Path] = []
        if len(paths) > 1 or len(details) > 1:
            import cv2
            for idx, d in enumerate(details):
                img = d.get("image_bgr")
                fp = Path(d["file_path"]) if d.get("file_path") else (paths[idx] if idx < len(paths) else None)
                if img is None and fp is not None and fp.is_file():
                    img = cv2.imread(str(fp))
                if img is not None:
                    bgrs.append(img)
                    if fp is not None:
                        valid_paths.append(fp)
        if len(bgrs) > 1:
            ws = CinematicWorkspace(bgrs, valid_paths, self)
            self.open_create_workspace(ws)
            return

        p, bgr = self._get_single_selected_photo_and_bgr()
        if bgr is None:
            QMessageBox.information(self, "3D Cinematic", "Please select a photo to generate 3D Cinematic Motion video.")
            return

        ws = CinematicWorkspace(bgr, p, self)
        self.open_create_workspace(ws)

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
            show_cameras=getattr(self.left_nav_dock, "_show_cameras", False),
            show_subjects=getattr(self.left_nav_dock, "_show_subjects", False),
            favourites_enabled=getattr(self, "favourites_enabled", True),
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
            self.nav_show_cameras = dlg.show_cameras()
            old_show_subjects = bool(getattr(self, "nav_show_subjects", False))
            self.nav_show_subjects = dlg.show_subjects()
            self.left_nav_dock.set_navigation_sections_visibility(
                show_dates=self.nav_show_dates,
                show_tags=self.nav_show_tags,
                show_people=self.nav_show_people,
                show_ratings=self.nav_show_ratings,
                show_places=self.nav_show_places,
                show_albums=self.nav_show_albums,
                show_cameras=self.nav_show_cameras,
                show_subjects=self.nav_show_subjects,
            )
            if self.nav_show_subjects:
                if (
                    not old_show_subjects
                    and not getattr(dlg, "_semantic_prompted_in_dialog", False)
                    and getattr(self, "semantic_engine", None) is not None
                    and not self.semantic_engine.is_available()
                ):
                    dlg._semantic_prompted_in_dialog = True
                    self._ensure_semantic_addon_with_prompt()
                self._refresh_subjects_nav_counts()
                if (
                    getattr(self, "semantic_engine", None) is not None
                    and getattr(self, "semantic_repo", None) is not None
                    and self.semantic_engine.is_available()
                ):
                    self._start_semantic_worker()

            self.set_favourites_enabled(dlg.favourites_enabled())

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
            settings.setValue("nav_show_cameras", self.nav_show_cameras)
            settings.setValue("nav_show_subjects", self.nav_show_subjects)
            settings.setValue("favourites_enabled", self.favourites_enabled)
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
        dlg.picasa_import_requested.connect(self.open_picasa_import_dialog)
        dlg.picasa_folders_requested.connect(self.open_picasa_folders_dialog)
        dlg.semantic_addon_downloaded.connect(lambda: self._start_semantic_worker(force=True))
        if dlg.exec() == AppSettingsDialog.Accepted:
            _apply_dlg_settings()

            if dlg.compact_requested:
                self.catalog_repo.compact_database()
                self.thumbnail_repo.compact_database()

            if getattr(dlg, "rebuild_thumbs_requested", False):
                self.rebuild_thumbnail_cache()

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
            self.start_face_detection(prompt_if_missing=True)
        if getattr(dlg, "fast_rematch_requested", False):
            if hasattr(self, "batch_people_view") and self.batch_people_view:
                self.batch_people_view.refresh_suggestions()
            if hasattr(self, "people_directory_view") and self.people_directory_view:
                self.people_directory_view.reload_directory()
        if getattr(dlg, "rescan_ocr_requested", False):
            queued = self.catalog_repo.reset_ocr_scan_flags()
            self._media_meta_lookup.clear()
            self.statusBar().showMessage(
                f"Queued {queued:,} photo(s) for background OCR re-scan.", 5000
            )
            self._start_ocr_worker()
        if getattr(dlg, "fresh_start_requested", False):
            self.perform_fresh_start_reset()

    def perform_fresh_start_reset(self) -> None:
        """Stops background workers, wipes catalog.db & thumbnails.db, and resets the app to a Fresh Start state."""
        if hasattr(self, "_scanner_cancel"):
            self._scanner_cancel.set()
        for flag in list(getattr(self, "_active_thumb_flags", {}).values()):
            try:
                flag.set()
            except Exception:
                pass
        if hasattr(self, "_active_thumb_flags"):
            self._active_thumb_flags.clear()
        if getattr(self, "_map_view", None) is not None and hasattr(self._map_view, "cancel_pending_thumbnail_work"):
            self._map_view.cancel_pending_thumbnail_work()
        if hasattr(self, "stop_face_detection"):
            try:
                self.stop_face_detection()
            except Exception:
                pass
        if getattr(self, "_active_ocr_worker", None) is not None:
            try:
                self._active_ocr_worker.cancel()
            except Exception:
                pass
            self._active_ocr_worker = None
        if getattr(self, "_semantic_cancel", None) is not None:
            try:
                self._semantic_cancel.set()
            except Exception:
                pass
        self._active_semantic_worker = None
        self._semantic_status_text = "Semantic AI: Idle"
        if hasattr(self, "xmp_daemon"):
            try:
                self.xmp_daemon.set_paused(True)
            except Exception:
                pass
        if hasattr(self, "thumb_write_coalescer"):
            try:
                self.thumb_write_coalescer.flush()
            except Exception:
                pass
        for pool_name in ("scanner_pool", "thumb_cache_pool", "thumbnail_pool", "face_pool", "priority_face_pool", "ocr_pool"):
            pool = getattr(self, pool_name, None)
            if pool is not None:
                try:
                    pool.clear()
                    pool.waitForDone(2000)
                except Exception:
                    pass

        self.pixmap_cache.clear()
        self._media_meta_lookup.clear()
        if hasattr(self, "_failed_thumb_ids"):
            self._failed_thumb_ids.clear()
        if hasattr(self, "_pending_visible_media_ids"):
            self._pending_visible_media_ids.clear()
        if hasattr(self, "_pending_disk_decode_ids"):
            self._pending_disk_decode_ids.clear()
        self._selected_ids = []
        self._lead_selected_id = None
        self._active_map_media_ids = None
        self._active_similar_order = None
        self._active_similar_scored = None
        self._active_similar_label = None
        self._active_similar_seed_id = None
        self._active_subject_ids = None
        self._active_subject_label = ""
        self._active_bbox = None
        self._active_date_range = None
        self._active_date_label = None

        self.engine.reset_all_databases()
        if getattr(self, "semantic_repo", None) is not None:
            try:
                self.semantic_repo.clear_all()
            except Exception:
                pass
        if hasattr(self, "_scanner_cancel") and not getattr(self, "_workers_paused", False):
            self._scanner_cancel.clear()
        if hasattr(self, "xmp_daemon"):
            try:
                self.xmp_daemon.set_paused(
                    bool(getattr(self, "write_back_paused", True))
                    or bool(getattr(self, "_workers_paused", False))
                )
            except Exception:
                pass

        settings = QSettings("PhotoGallery64", "PhotoGallery64")
        for key in ("last_selected_id", "ui/show_private_folders"):
            settings.remove(key)

        if hasattr(self, "left_nav_dock"):
            self.left_nav_dock.clear_all_filters()
        if hasattr(self, "scope_bar"):
            self.scope_bar.set_similarity_controls(False)
            self.scope_bar.hide()
        if self.central_stack.currentWidget() != self.grid_view:
            self.show_grid_view()

        self.refresh_navigation_and_autocomplete()
        self.rebuild_and_run_query(preserve_selection=False)
        if hasattr(self, "folder_watcher"):
            self.folder_watcher.update_watched_paths()
        self.statusBar().showMessage("Catalog reset. Welcome to a Fresh Start!", 5000)

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

        self.xmp_daemon.set_paused(self.write_back_paused or getattr(self, "_workers_paused", False))
        self.ribbon.set_write_back_paused(self.write_back_paused)
        if paused is None:
            self.show_status_message(
                "Background file write-back paused (Catalog-only mode)"
                if self.write_back_paused
                else "Background file write-back resumed",
                2800,
            )

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
            self.xmp_daemon.notify_dirty()
        else:
            dirty = self.catalog_repo.count_dirty_media()
            pending_txt = f" ({dirty:,} pending)" if dirty > 0 else ""
            self.statusBar().showMessage(f"Write to files turned off (Catalog-only mode{pending_txt})", 3000)

    def copy_subject_for_media(self, media_id: int) -> None:
        """Open single viewer if needed and copy the foreground subject cutout to clipboard."""
        if getattr(self.media_viewer, "current_media_id", None) != int(media_id):
            self.open_single_viewer(int(media_id))
        self.media_viewer.copy_subject_to_clipboard()

    def remove_background_for_media(self, media_id: int) -> None:
        """Open single viewer if needed and save a background-removed cutout copy."""
        if getattr(self.media_viewer, "current_media_id", None) != int(media_id):
            self.open_single_viewer(int(media_id))
        self.media_viewer.remove_background_cutout()

    def open_tag_hierarchy_dialog(self) -> None:
        """Opens the 'All of your tags' dialog matching WLPG with drag-and-drop hierarchy organization."""
        from src.ui.tag_hierarchy_dialog import TagHierarchyDialog
        dlg = TagHierarchyDialog(self, repo=self.catalog_repo)
        tags = self.catalog_repo.get_tag_tree()
        dlg.populate(tags)
        dlg.tag_hierarchy_changed.connect(self.refresh_navigation_and_autocomplete)
        dlg.tags_added.connect(self._on_hierarchy_tags_selected)
        dlg.exec()

    def _on_right_pane_interacted(self) -> None:
        """Closes any open face-tag popup/box on the viewer when the user clicks in the right pane."""
        mv = getattr(self, "media_viewer", None)
        if mv is None:
            return
        if getattr(mv, "is_face_tag_mode", False) or (
            hasattr(mv, "_tag_someone_popup") and mv._tag_someone_popup.isVisible()
        ):
            mv.stop_face_tagging_mode()
        if hasattr(mv, "_inline_name_popup") and mv._inline_name_popup.isVisible():
            mv._inline_name_popup.hide()

    def open_duplicate_finder(self) -> None:
        """Opens the modern integrated duplicate & burst finder view."""
        self.show_duplicate_finder_view()

    def open_find_similar_for_media(self, media_id: int, similarity_threshold: float = 0.81) -> None:
        """Finds visually similar photos in the background with a progress bar and filters the Gallery Grid."""
        from PySide6.QtCore import QThread, Signal
        from PySide6.QtWidgets import QProgressDialog
        from src.core.duplicate_finder import DuplicateDetectionEngine

        target_mid = int(media_id)
        thresh = max(0.55, min(0.99, float(similarity_threshold)))

        if self.central_stack.currentWidget() != self.grid_view:
            self.show_grid_view()

        detail = self.catalog_repo.get_media_detail(target_mid)
        seed_name = str(detail["filename"]) if detail and detail.get("filename") else f"Photo #{target_mid}"

        engine = DuplicateDetectionEngine(
            self.catalog_repo,
            self.thumb_repo,
            semantic_repo=getattr(self, "semantic_repo", None),
            semantic_engine=getattr(self, "semantic_engine", None),
        )

        class _SimilarWorker(QThread):
            progress_sig = Signal(int, int, str)
            done_sig = Signal(object, str)

            def run(self) -> None:
                try:
                    # Pre-compute down to the slider minimum (70%) once so live slider scrubbing is instant
                    grp = engine.find_similar_to_photo(
                        seed_media_id=target_mid,
                        similarity_threshold=min(0.70, thresh),
                        media_records=None,
                        max_results=200,
                        progress_cb=lambda c, t, m: self.progress_sig.emit(int(c), int(t), str(m)),
                    )
                    self.done_sig.emit(grp, "")
                except Exception as exc:
                    logger.exception("Find Similar failed for media_id=%s", target_mid)
                    self.done_sig.emit(None, str(exc))

        dlg = QProgressDialog(f"Finding photos similar to '{seed_name}'...", "Cancel", 0, 100, self)
        dlg.setWindowTitle("Find Similar Photos")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumWidth(420)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(5)
        if getattr(self, "_dark_mode", False):
            dlg.setStyleSheet(
                "QProgressDialog { background-color: #1E242C; color: #E2E8F0; }"
                "QLabel { color: #E2E8F0; font-size: 9.5pt; }"
                "QProgressBar { border: 1px solid #3B4252; border-radius: 4px; background: #14171C; text-align: center; color: #E2E8F0; min-height: 18px; }"
                "QProgressBar::chunk { background-color: #2563EB; border-radius: 3px; }"
                "QPushButton { background: #262B34; color: #E2E8F0; border: 1px solid #3B4252; border-radius: 3px; padding: 4px 12px; }"
            )
        else:
            dlg.setStyleSheet(
                "QProgressDialog { background-color: #FBFDFF; color: #1E395B; }"
                "QLabel { color: #1E395B; font-size: 9.5pt; }"
                "QProgressBar { border: 1px solid #B5CBE2; border-radius: 4px; background: #E9EFF6; text-align: center; color: #1E395B; min-height: 18px; }"
                "QProgressBar::chunk { background-color: #0078D7; border-radius: 3px; }"
                "QPushButton { background: #F1F5F9; color: #1E395B; border: 1px solid #94A3B8; border-radius: 3px; padding: 4px 12px; }"
            )

        worker = _SimilarWorker(self)
        self._similar_search_worker = worker
        self._similar_search_dlg = dlg

        def _on_prog(cur: int, tot: int, msg: str) -> None:
            if tot > 0:
                dlg.setMaximum(tot)
                dlg.setValue(cur)
            if msg:
                dlg.setLabelText(msg)

        def _on_cancel() -> None:
            engine.cancel()

        def _on_done(grp: Any, err_msg: str = "") -> None:
            dlg.blockSignals(True)
            dlg.accept()
            if engine._is_cancelled:
                return
            if err_msg:
                QMessageBox.warning(self, "Find Similar Photos", f"Could not search for similar photos:\n{err_msg}")
                return
            if grp is None or len(getattr(grp, "items", [])) <= 1:
                scored_pairs: list[tuple[int, float]] = [(target_mid, 1.0)]
                ordered_ids = [target_mid]
                match_cnt = 0
            else:
                scored_pairs = [
                    (int(it.media_id), float(getattr(it, "similarity_score", 1.0 if int(it.media_id) == target_mid else thresh)))
                    for it in grp.items
                ]
                ordered_ids = [
                    mid for mid, sc in scored_pairs
                    if mid == target_mid or sc >= thresh
                ][:60]
                if not ordered_ids:
                    ordered_ids = [target_mid]
                match_cnt = max(0, len(ordered_ids) - 1)
            pct = int(round(thresh * 100))
            self._active_similar_seed_id = target_mid
            self._active_similar_seed_name = seed_name
            self._active_similar_pct = pct
            self._active_similar_scored = scored_pairs
            self._active_similar_order = ordered_ids
            self._active_similar_label = (
                f"Similar to '{seed_name}' ({match_cnt} matches ≥ {pct}%)"
            )
            self._active_map_media_ids = ordered_ids
            self.scope_bar.set_similarity_controls(True, pct)
            self._update_scope_bar_visibility()
            recs = self.rebuild_and_run_query(preserve_selection=False)
            actual_matches = max(0, len(recs) - 1)
            if actual_matches != match_cnt:
                self._active_similar_label = (
                    f"Similar to '{seed_name}' ({actual_matches} matches ≥ {pct}%)"
                )
                self._update_scope_bar_visibility()
            self.grid_view.select_media_id(target_mid)
            self._update_status_counts()

        worker.progress_sig.connect(_on_prog)
        dlg.canceled.connect(_on_cancel)
        worker.done_sig.connect(_on_done)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        dlg.exec()

    def _on_similar_threshold_changed(self, pct: int) -> None:
        """Live-updates the Find Similar Photos gallery filter when the Scope Bar slider changes."""
        seed_id = getattr(self, "_active_similar_seed_id", None)
        if seed_id is None:
            return
        pct_i = max(70, min(98, int(pct)))
        thresh = pct_i / 100.0
        self._active_similar_pct = pct_i
        seed_name = getattr(self, "_active_similar_seed_name", f"Photo #{seed_id}")

        cached_scored = getattr(self, "_active_similar_scored", None)
        if cached_scored is not None:
            ordered_ids = [
                mid for mid, sc in cached_scored
                if mid == int(seed_id) or sc >= thresh
            ][:60]
            if not ordered_ids:
                ordered_ids = [int(seed_id)]
            match_cnt = max(0, len(ordered_ids) - 1)
        else:
            from src.core.duplicate_finder import DuplicateDetectionEngine

            engine = DuplicateDetectionEngine(
                self.catalog_repo,
                self.thumb_repo,
                semantic_repo=getattr(self, "semantic_repo", None),
                semantic_engine=getattr(self, "semantic_engine", None),
            )
            grp = engine.find_similar_to_photo(
                seed_media_id=int(seed_id),
                similarity_threshold=thresh,
                media_records=None,
                max_results=60,
            )
            if grp is None or len(getattr(grp, "items", [])) <= 1:
                ordered_ids = [int(seed_id)]
                match_cnt = 0
            else:
                ordered_ids = [int(it.media_id) for it in grp.items]
                match_cnt = len(ordered_ids) - 1
        self._active_similar_order = ordered_ids
        self._active_similar_label = f"Similar to '{seed_name}' ({match_cnt} matches ≥ {pct_i}%)"
        self._active_map_media_ids = ordered_ids
        self._update_scope_bar_visibility()
        recs = self.rebuild_and_run_query(preserve_selection=False)
        actual_matches = max(0, len(recs) - 1)
        if actual_matches != match_cnt:
            self._active_similar_label = f"Similar to '{seed_name}' ({actual_matches} matches ≥ {pct_i}%)"
            self._update_scope_bar_visibility()
        self.grid_view.select_media_id(int(seed_id))
        self._update_status_counts()

    def show_duplicate_finder_view(self) -> None:
        """Switches central stack to integrated duplicate & burst finder view."""
        if hasattr(self, "duplicate_finder_view") and self.central_stack.currentWidget() == self.duplicate_finder_view:
            return

        # Save previous docks visibility
        self._nav_dock_was_visible = self.left_nav_dock.isVisible()
        self._tag_dock_was_visible = self.right_tag_dock.isVisible()
        self.left_nav_dock.hide()
        self.right_tag_dock.hide()
        if hasattr(self, "bottom_dock") and self.bottom_dock is not None:
            self.bottom_dock.hide()

        # Switch view immediately so the user always sees the Duplicate Finder
        if hasattr(self, "duplicate_finder_view"):
            self.duplicate_finder_view.set_use_ribbon_mode(True)
            self.central_stack.setCurrentWidget(self.duplicate_finder_view)
            self.ribbon.set_active_mode("duplicates")
            if hasattr(self.ribbon, "btn_dup_tag_pane"):
                self.ribbon.btn_dup_tag_pane.blockSignals(True)
                self.ribbon.btn_dup_tag_pane.setChecked(False)
                self.ribbon.btn_dup_tag_pane.blockSignals(False)
            self.setWindowTitle("Find Duplicates & Bursts - Essentials Photo Gallery")

        records = getattr(self.catalog_model, "_records", None) if hasattr(self, "catalog_model") else None
        sel_ids = set(self.grid_view.selected_media_ids() or ())
        sel_records = (
            [r for r in (records or []) if (r.get("id") if isinstance(r, dict) else getattr(r, "id", None)) in sel_ids]
            if (sel_ids and records)
            else None
        )

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
                dark_now = bool(getattr(self, "_dark_mode", False))
                if self.duplicate_finder_view.is_dark != dark_now:
                    self.duplicate_finder_view.set_dark_mode(dark_now)
                self.duplicate_finder_view.ensure_workspace_visible()
            except Exception as e:
                logger.error(f"Error initializing duplicate finder view: {e}")

    def _on_duplicates_scan_state_changed(self, scanning: bool) -> None:
        self._dup_scanning = bool(scanning)
        marked = getattr(self, "_dup_marked_count", 0)
        if hasattr(self, "ribbon") and hasattr(self.ribbon, "update_duplicates_state"):
            self.ribbon.update_duplicates_state(scanning=self._dup_scanning, marked_count=marked)

    def _on_duplicates_summary_state_changed(self, marked_count: int) -> None:
        self._dup_marked_count = int(marked_count)
        scanning = getattr(self, "_dup_scanning", False)
        if hasattr(self, "ribbon") and hasattr(self.ribbon, "update_duplicates_state"):
            self.ribbon.update_duplicates_state(scanning=scanning, marked_count=self._dup_marked_count)

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
            self.refresh_navigation_and_autocomplete({"tags", "albums"})
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
        self.refresh_navigation_and_autocomplete({"tags", "albums"})
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
            if getattr(self, "_map_view", None) is not None and hasattr(self._map_view, "cancel_pending_thumbnail_work"):
                self._map_view.cancel_pending_thumbnail_work()
            self.stop_face_detection()
            if getattr(self, "_active_ocr_worker", None) is not None:
                self._active_ocr_worker.cancel()
                self._active_ocr_worker = None
            self._ocr_status_text = "OCR: Idle"
            if getattr(self, "_semantic_cancel", None) is not None:
                try:
                    self._semantic_cancel.set()
                except Exception:
                    pass
            self._active_semantic_worker = None
            self._semantic_status_text = "Semantic AI: Idle"
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
            if getattr(self, "find_people_enabled", True) and self._are_face_models_installed():
                self.start_face_detection()
            self._start_ocr_worker()
            self._start_semantic_worker()
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
        ocr_summary = getattr(self, "_ocr_status_text", "OCR: Idle").removeprefix("OCR: ")
        sem_summary = getattr(self, "_semantic_status_text", "Semantic AI: Idle").removeprefix("Semantic AI: ")
        detailed_tooltip = (
            f"📁 Indexing: {scan_summary}\n"
            f"👤 Face Detection: {face_summary}\n"
            f"📝 OCR Indexing: {ocr_summary}\n"
            f"🧠 Semantic AI: {sem_summary}\n"
            f"{crop_detail}\n"
            f"{thumb_detail}\n"
            f"{sync_detail}\n\n"
            "Click for details / background options"
        )

        is_scanning = "Idle" not in self._scanner_status_text
        is_faces = (getattr(self, "_face_worker", None) is not None) or ("Idle" not in self._face_status_text)
        is_ocr = (getattr(self, "_active_ocr_worker", None) is not None) or ("Idle" not in getattr(self, "_ocr_status_text", "OCR: Idle"))
        is_semantic = (getattr(self, "_active_semantic_worker", None) is not None) or ("Idle" not in getattr(self, "_semantic_status_text", "Semantic AI: Idle"))

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
        elif is_ocr:
            summary_text = f"OCR Indexing: {ocr_summary}"
            style = "color: #0066CC; font-size: 8.5pt;"
        elif is_semantic:
            summary_text = f"Semantic AI: {sem_summary}"
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

        ocr_state_label = getattr(self, "_ocr_status_text", "OCR: Idle").removeprefix("OCR: ")
        act_ocr = menu.addAction(f"📝 OCR Indexing: {ocr_state_label} (Options...)")
        act_ocr.triggered.connect(lambda: self._open_settings_dialog(initial_tab="addons"))

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
            "ocr": act_ocr,
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
        ws = getattr(self, "_active_create_workspace", None)
        if ws is not None:
            factor = (size_px / 150.0) ** 2.0
            pct = int(round(max(20.0, min(400.0, factor * 100.0))))
            ws.set_zoom_percent(pct)
            return

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

    def _sync_status_zoom_from_create_workspace(self, scale_val: float, is_fit: bool) -> None:
        if getattr(self, "_active_create_workspace", None) is None:
            return
        import math

        zoom_lvl = max(0.2, min(float(scale_val), 4.0))
        size_px = int(round(150.0 * math.sqrt(zoom_lvl)))
        clamped = max(72, min(size_px, 320))
        if hasattr(self, "slider_status_zoom"):
            self.slider_status_zoom.blockSignals(True)
            self.slider_status_zoom.setValue(clamped)
            self.slider_status_zoom.blockSignals(False)
        if hasattr(self, "btn_status_fit") and self.btn_status_fit is not None:
            if is_fit:
                self.btn_status_fit.setIcon(make_wlpg_icon("actual_size", 18))
                self.btn_status_fit.setToolTip("Actual size (1)")
            else:
                self.btn_status_fit.setIcon(make_wlpg_icon("fit_window", 18))
                self.btn_status_fit.setToolTip("Fit to window (F)")

    def _on_status_zoom_toggle_clicked(self) -> None:
        ws = getattr(self, "_active_create_workspace", None)
        if ws is not None:
            if getattr(ws, "_is_fit_mode", True):
                ws.zoom_actual()
            else:
                ws.zoom_fit()
            return
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
        if getattr(self, "_active_create_workspace", None) is not None:
            if not self.close_create_workspace():
                return
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
    def rebuild_thumbnail_cache(self) -> None:
        """Clear cached thumbnail blobs and regenerate visible thumbnails at HiDPI/panorama quality."""
        try:
            with self.engine.thumb_transaction() as conn:
                conn.execute("DELETE FROM thumbnails;")
        except Exception:
            try:
                conn = self.engine.thumb_conn()
                conn.execute("DELETE FROM thumbnails;")
                conn.commit()
            except Exception:
                pass
        try:
            self.pixmap_cache.clear()
        except Exception:
            pass
        if hasattr(self, "card_delegate") and hasattr(self.card_delegate, "_scaled_pixmap_cache"):
            self.card_delegate._scaled_pixmap_cache.clear()
        if hasattr(self, "grid_view") and self.grid_view is not None:
            self.grid_view.viewport().update()
            self.enqueue_visible_thumbnails(self.grid_view.visible_media_ids(buffer_rows=2))
        self.show_status_message("✓ Thumbnail cache cleared — rebuilding visible thumbnails at HiDPI quality...", 4000)

    def apply_theme(self, theme_name: str) -> None:
        self.setUpdatesEnabled(False)
        try:
            self.current_theme = theme_name  # type: ignore[assignment]
            is_dark = (theme_name == "dark")
            if hasattr(self, "welcome_overlay") and self.welcome_overlay is not None:
                self.welcome_overlay.apply_theme(theme_name)
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
                getattr(self, "bottom_dock", None),
                getattr(self, "_active_create_workspace", None),
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
        qp = getattr(self, "_quick_preview_dialog", None)
        if qp is not None and qp.isVisible():
            qp.keyPressEvent(event)
            if event.isAccepted():
                return
        ws = getattr(self, "_active_create_workspace", None)
        if ws is not None:
            if event.key() == Qt.Key_Escape:
                self.close_create_workspace()
                event.accept()
                return
            if ws.handle_key_press(event):
                event.accept()
                return
            if event.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
                event.accept()
                return
            super().keyPressEvent(event)
            return
        focus_w = QApplication.focusWidget() or self.focusWidget()
        if not focus_w and hasattr(self, "right_tag_dock") and self.right_tag_dock is not None:
            focus_w = self.right_tag_dock.focusWidget()
        from PySide6.QtWidgets import QAbstractSpinBox
        is_text_entry = isinstance(focus_w, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox))

        if (
            not is_text_entry
            and hasattr(self, "compare_view")
            and self.central_stack.currentWidget() is self.compare_view
            and not (event.modifiers() & (Qt.ControlModifier | Qt.AltModifier))
        ):
            self.compare_view.keyPressEvent(event)
            if event.isAccepted():
                return

        # Star ratings: 0..5 (or Shift+0..5 to auto-advance) and Flag: P / Shift+P (when not focused on a text entry)
        if not is_text_entry and not (event.modifiers() & (Qt.ControlModifier | Qt.AltModifier)):
            shift_held = bool(event.modifiers() & Qt.ShiftModifier)
            _SHIFT_DIGIT_KEYS = {
                Qt.Key_ParenRight: 0,
                Qt.Key_Exclam: 1,
                Qt.Key_At: 2,
                Qt.Key_QuoteDbl: 2,
                Qt.Key_NumberSign: 3,
                Qt.Key_sterling: 3,
                Qt.Key_Dollar: 4,
                Qt.Key_Percent: 5,
            }
            rating_val: int | None = None
            if Qt.Key_0 <= event.key() <= Qt.Key_5:
                rating_val = event.key() - Qt.Key_0
            elif shift_held and event.key() in _SHIFT_DIGIT_KEYS:
                rating_val = _SHIFT_DIGIT_KEYS[event.key()]
            elif shift_held and 0x30 <= int(event.nativeVirtualKey() or 0) <= 0x35:
                rating_val = int(event.nativeVirtualKey()) - 0x30
            if rating_val is not None:
                self.apply_rating_to_selection(rating_val, advance=shift_held)
                event.accept()
                return
            if event.key() == Qt.Key_P:
                target_ids = self.get_active_target_ids()
                if target_ids:
                    has_non_picked = any(
                        int(getattr(self.catalog_model.record_by_id(mid), "flag", 0) or 0) != 1
                        for mid in target_ids
                    )
                    self.apply_flag_to_selection(1 if has_non_picked else 0, advance=shift_held)
                    event.accept()
                    return
            if event.key() == Qt.Key_X:
                target_ids = self.get_active_target_ids()
                if target_ids:
                    has_non_rejected = any(
                        int(getattr(self.catalog_model.record_by_id(mid), "flag", 0) or 0) != -1
                        for mid in target_ids
                    )
                    self.apply_flag_to_selection(-1 if has_non_rejected else 0, advance=shift_held)
                    event.accept()
                    return
            if event.key() == Qt.Key_U:
                if self.get_active_target_ids():
                    self.apply_flag_to_selection(0, advance=shift_held)
                    event.accept()
                    return
            if event.key() == Qt.Key_F and getattr(self, "favourites_enabled", True):
                if self.get_active_target_ids():
                    self.toggle_favourite_selection(advance=shift_held)
                    event.accept()
                    return
            if event.key() == Qt.Key_F2 and not shift_held:
                if self.central_stack.currentWidget() in (self.grid_view, self.media_viewer):
                    self.rename_selection()
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
                if getattr(self, "_active_similar_order", None) is not None:
                    self._on_clear_location_filter()
                    event.accept()
                    return
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
                    grid_sel = self.grid_view.selected_media_ids()
                    target_id = (
                        self._lead_selected_id
                        if self._lead_selected_id in grid_sel
                        else (grid_sel[-1] if grid_sel else (self._selected_ids[0] if self._selected_ids else None))
                    )
                    if target_id is not None:
                        self.open_single_viewer(target_id)
                        event.accept()
                        return
                elif self.central_stack.currentWidget() == self.media_viewer:
                    self.show_grid_view()
                    event.accept()
                    return

            # Space: Grid -> Quick Preview; Viewer -> Next slide
            if event.key() == Qt.Key_Space:
                if event.isAutoRepeat():
                    event.accept()
                    return
                if self.central_stack.currentWidget() == self.grid_view:
                    grid_sel = self.grid_view.selected_media_ids()
                    target_id = (
                        self._lead_selected_id
                        if self._lead_selected_id in grid_sel
                        else (grid_sel[-1] if grid_sel else (self._selected_ids[0] if self._selected_ids else None))
                    )
                    if target_id is not None:
                        self.show_quick_preview(target_id)
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
            ws_snap = getattr(self, "_create_ws_snapshot", None)
            if getattr(self, "_active_create_workspace", None) is not None and ws_snap is not None:
                settings.setValue("tag_dock_visible", bool(ws_snap.get("tag_visible", True)))
                settings.setValue("nav_dock_visible", bool(ws_snap.get("nav_visible", True)))
                current_view = "viewer" if ws_snap.get("was_viewer") else "grid"
            else:
                settings.setValue("tag_dock_visible", not self.right_tag_dock.isHidden())
                if self.central_stack.currentWidget() == self.media_viewer:
                    settings.setValue("nav_dock_visible", getattr(self, "_nav_dock_was_visible", True))
                else:
                    settings.setValue("nav_dock_visible", not self.left_nav_dock.isHidden())
                current_view = "grid"
                if self.central_stack.currentWidget() == self.media_viewer:
                    current_view = "viewer"
                elif self.central_stack.currentWidget() == self.map_view:
                    current_view = "map"
            settings.setValue("filmstrip_visible", bool(getattr(self, "_filmstrip_viewer_enabled", False)))
            if hasattr(self, "bottom_dock") and self.bottom_dock is not None:
                settings.setValue("filmstrip_height", int(self.bottom_dock.saved_filmstrip_height()))
            settings.setValue("arrange_by", self._arrange_by)
            settings.setValue("reverse_sort", self._reverse_sort)
            settings.setValue("group_by", self._group_by)
            settings.setValue("sort_by", self._sort_by)
            settings.setValue("zoom_val", self.slider_status_zoom.value())
            if hasattr(self, "grid_view") and hasattr(self.grid_view, "catalog_model"):
                settings.setValue("layout_mode", getattr(self.grid_view.catalog_model, "_layout_mode", "wlpg_grid"))
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
            settings.setValue("nav_show_cameras", getattr(self.left_nav_dock, "_show_cameras", False))
            settings.setValue("nav_show_subjects", getattr(self.left_nav_dock, "_show_subjects", False))
            settings.setValue("favourites_enabled", getattr(self, "favourites_enabled", True))
            settings.setValue("smart_show_favourites", getattr(self.left_nav_dock, "_show_smart_favourites", True))
            settings.setValue("smart_show_picked", getattr(self.left_nav_dock, "_show_smart_picked", True))
            settings.setValue("smart_show_rejected", getattr(self.left_nav_dock, "_show_smart_rejected", True))
            settings.setValue("smart_show_unflagged", getattr(self.left_nav_dock, "_show_smart_unflagged", True))
            settings.setValue("nav_section_order", json.dumps(self.left_nav_dock.get_section_order()))
            settings.setValue("find_people_enabled", getattr(self, "find_people_enabled", True))
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
                    "nav_show_cameras": getattr(self.left_nav_dock, "_show_cameras", False),
                    "nav_show_subjects": getattr(self.left_nav_dock, "_show_subjects", False),
                    "find_people_enabled": getattr(self, "find_people_enabled", True),
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

    def _safe_deduplicate_media_people_async(self) -> None:
        """Runs phantom/duplicate person-tag cleanup after startup without blocking launch."""
        try:
            self.catalog_repo.deduplicate_media_people()
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
            self._filmstrip_viewer_enabled = str(settings.value("filmstrip_visible", "false")).lower() in ("true", "1")
            if hasattr(self, "bottom_dock") and self.bottom_dock is not None:
                fs_h = settings.value("filmstrip_height", type=int)
                if fs_h and fs_h >= 92:
                    self.bottom_dock.set_saved_filmstrip_height(fs_h)
                self.bottom_dock.hide()

            f_json = settings.value("filter_state")
            if f_json and isinstance(f_json, str):
                try:
                    f_dict = json.loads(f_json)
                    self.left_nav_dock.restore_filter_state(f_dict)
                except Exception:
                    pass

            zoom_val = settings.value("zoom_val", type=int)
            if not zoom_val and isinstance(backup_prefs.get("zoom_val"), int):
                zoom_val = int(backup_prefs["zoom_val"])
            if not (zoom_val and 72 <= zoom_val <= 320):
                zoom_val = 150
            self.slider_status_zoom.setValue(zoom_val)

            saved_layout = settings.value("layout_mode", type=str)
            if saved_layout in ("wlpg_grid", "tiles", "mosaic", "masonry"):
                self.ribbon.set_layout_mode(saved_layout)
                self.grid_view.set_layout_mode(saved_layout)

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
            self.nav_show_cameras = _to_bool(_get_val("nav_show_cameras", False), False)
            self.nav_show_subjects = _to_bool(_get_val("nav_show_subjects", False), False)
            raw_order = _get_val("nav_section_order", None)
            if raw_order:
                try:
                    parsed_order = json.loads(str(raw_order)) if isinstance(raw_order, str) else list(raw_order)
                    if isinstance(parsed_order, list):
                        self.left_nav_dock.set_section_order(parsed_order)
                except Exception:
                    pass
            self.find_people_enabled = _to_bool(_get_val("find_people_enabled", True), True)
            self.auto_group_bursts = _to_bool(_get_val("auto_group_bursts", False), False)
            self.auto_group_raw_jpeg = _to_bool(_get_val("auto_group_raw_jpeg", False), False)
            if hasattr(self, "ribbon") and hasattr(self.ribbon, "set_photo_grouping_state"):
                self.ribbon.set_photo_grouping_state(
                    auto_bursts=self.auto_group_bursts,
                    auto_raw_jpeg=self.auto_group_raw_jpeg,
                )
            self.set_badge_visibility(
                flags=_to_bool(_get_val("badge_show_flags", True), True),
                favourites=_to_bool(_get_val("badge_show_favourites", True), True),
                groups=_to_bool(_get_val("badge_show_groups", True), True),
                save=False,
            )
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
                show_cameras=self.nav_show_cameras,
                show_subjects=self.nav_show_subjects,
            )
            if self.nav_show_subjects:
                self._refresh_subjects_nav_counts()
            elif (
                "PYTEST_CURRENT_TEST" not in os.environ
                and getattr(self, "semantic_repo", None) is not None
                and getattr(self, "semantic_engine", None) is not None
                and self.semantic_engine.is_available()
            ):
                QTimer.singleShot(2500, self._prewarm_semantic_matrix_async)
            self.left_nav_dock.set_builtin_smart_albums_visibility(
                favourites=_to_bool(_get_val("smart_show_favourites", True), True),
                picked=_to_bool(_get_val("smart_show_picked", True), True),
                rejected=_to_bool(_get_val("smart_show_rejected", True), True),
                unflagged=_to_bool(_get_val("smart_show_unflagged", True), True),
                emit_signal=False,
            )
            self.set_favourites_enabled(_to_bool(_get_val("favourites_enabled", True), True))

            saved_theme = _get_val("theme", "wlpg_aero")
            if saved_theme and saved_theme in ("wlpg_aero", "modern", "dark"):
                if getattr(self, "current_theme", None) != saved_theme:
                    self.apply_theme(saved_theme)

            saved_icon_theme = _get_val("icon_theme", "wlpg_aero")
            if saved_icon_theme and saved_icon_theme in ("wlpg_aero", "fluent", "minimal"):
                if getattr(self, "current_icon_theme", None) != saved_icon_theme:
                    self.current_icon_theme = saved_icon_theme
                    icon_provider.set_icon_theme(saved_icon_theme)

            # Defer phantom/duplicate person-tag cleanup so it never blocks app launch
            QTimer.singleShot(
                2200,
                lambda: self._safe_deduplicate_media_people_async(),
            )

            # Restore un-pinned private folders visibility across restarts
            try:
                from src.ui.folder_manager_dialog import get_private_auth_mode, has_private_pin

                if get_private_auth_mode() == "none" and not has_private_pin():
                    if bool(settings.value("ui/show_private_folders", False, type=bool)):
                        self.set_show_private_folders(True)
            except Exception:
                pass

            # Re-run query with restored filter
            records = self.rebuild_and_run_query()
            self._did_initial_query = True
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
        ws = getattr(self, "_active_create_workspace", None)
        if ws is not None:
            try:
                ws.cleanup()
            except Exception:
                pass

        # 1. Signal cancellation to all workers simultaneously before waiting
        self._scanner_cancel.set()
        for flag in self._active_thumb_flags.values():
            flag.set()
        self._active_thumb_flags.clear()
        if getattr(self, "_map_view", None) is not None and hasattr(self._map_view, "cancel_pending_thumbnail_work"):
            self._map_view.cancel_pending_thumbnail_work()
        self.stop_face_detection()
        if hasattr(self, "_monitor_timer") and self._monitor_timer.isActive():
            self._monitor_timer.stop()
        if getattr(self, "_active_ocr_worker", None) is not None:
            self._active_ocr_worker.cancel()
        if getattr(self, "_semantic_cancel", None) is not None:
            try:
                self._semantic_cancel.set()
            except Exception:
                pass
        if hasattr(self, "folder_watcher"):
            self.folder_watcher.stop()
        if hasattr(self, "xmp_daemon"):
            self.xmp_daemon.stop_requested.set()
            if getattr(self.xmp_daemon, "_timer", None) is not None and self.xmp_daemon._timer.isActive():
                self.xmp_daemon._timer.stop()

        # 2. Drop any queued (not-yet-started) runnables from all pools
        self.scanner_pool.clear()
        if hasattr(self, "thumb_cache_pool"):
            self.thumb_cache_pool.clear()
        self.thumbnail_pool.clear()
        self.face_pool.clear()
        self.priority_face_pool.clear()
        QThreadPool.globalInstance().clear()

        # 3. Wait briefly for active tasks to unwind and flush buffered WebP writes
        self.scanner_pool.waitForDone(800)
        if hasattr(self, "thumb_cache_pool"):
            self.thumb_cache_pool.waitForDone(500)
        self.thumbnail_pool.waitForDone(800)
        if hasattr(self, "thumb_write_coalescer"):
            try:
                self.thumb_write_coalescer.flush()
            except Exception:
                pass
        self.face_pool.waitForDone(1000)
        self.priority_face_pool.waitForDone(500)
        if hasattr(self, "xmp_daemon"):
            self.xmp_daemon.stop()
        QThreadPool.globalInstance().waitForDone(1000)
        if hasattr(self, "_inflight_workers"):
            self._inflight_workers.clear()
        if getattr(self, "semantic_repo", None) is not None:
            try:
                self.semantic_repo.close()
            except Exception:
                pass

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

        # Tear down QMediaPlayer, LaMa warmup thread, and QWebEngineView/Page so no QtWebEngineProcess or DLL locks linger
        try:
            if hasattr(self, "media_viewer"):
                self.media_viewer._is_shutting_down = True
                wt = getattr(self.media_viewer, "_lama_warmup_thread", None)
                if wt is not None and wt.is_alive():
                    wt.join(timeout=3.0)
            if hasattr(self, "media_viewer") and hasattr(self.media_viewer, "media_player"):
                self.media_viewer.media_player.stop()
                self.media_viewer.media_player.setSource(QUrl())
        except Exception:
            pass
        try:
            mv = getattr(self, "_map_view", None)
            if mv is not None:
                wv = getattr(mv, "web_view", None)
                if wv is not None:
                    page = wv.page()
                    wv.stop()
                    wv.setParent(None)
                    wv.deleteLater()
                    if page is not None:
                        page.deleteLater()
                mv.close()
                mv.deleteLater()
                self._map_view = None
        except Exception:
            pass

        app = QApplication.instance()
        if app:
            try:
                app.removeEventFilter(self)
                app.processEvents()
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
                app_focus = QApplication.focusWidget()
                focus_w = app_focus or self.focusWidget()
                target_watched = app_focus if app_focus is not None else self
                is_input_w = isinstance(
                    focus_w, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox)
                )
                if watched is target_watched and not is_input_w:
                    if self.central_stack.currentWidget() == self.media_viewer:
                        if event.key() == Qt.Key_Escape and event.modifiers() == Qt.NoModifier:
                            if hasattr(self.media_viewer, "handle_escape_key"):
                                self.media_viewer.handle_escape_key()
                            else:
                                self.media_viewer.request_close()
                            event.accept()
                            return True
                        if (
                            not getattr(self.media_viewer, "is_crop_mode", False)
                            and event.key() in (Qt.Key_Left, Qt.Key_Right)
                            and event.modifiers() in (Qt.NoModifier, Qt.KeypadModifier)
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
                QTimer.singleShot(5000, self._maybe_start_startup_face_scan)

    def nativeEvent(self, eventType, message):
        res = handle_win32_native_event(self, eventType, message)
        if res is not None:
            return res
        return super().nativeEvent(eventType, message)

    # -------------------------------------------------------------------------
    # Static Albums & Smart Albums
    # -------------------------------------------------------------------------
    def _refresh_albums_ui(self) -> None:
        if hasattr(self, "left_nav_dock"):
            self.left_nav_dock.refresh_albums_only(self.catalog_repo)
        if hasattr(self, "albums_view"):
            self.albums_view.refresh()
            self._update_status_counts()

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

    def set_album_cover(self, album_id: int, media_id: int) -> None:
        """Sets ``media_id`` as the cover photo for Static Album ``album_id`` and refreshes UI."""
        self.catalog_repo.set_album_cover(int(album_id), int(media_id))
        self._refresh_albums_ui()
        if hasattr(self, "show_status_message"):
            self.show_status_message("Updated album cover.")

    def set_smart_album_cover(self, smart_album_id: int, media_id: int) -> None:
        """Sets ``media_id`` as the cover photo for Smart Album ``smart_album_id`` and refreshes UI."""
        self.catalog_repo.set_smart_album_cover(int(smart_album_id), int(media_id))
        self._refresh_albums_ui()
        if hasattr(self, "show_status_message"):
            self.show_status_message("Updated smart album cover.")

    def _open_cover_picker_dialog(
        self, title: str, candidates: list[dict[str, Any]], on_picked: Any
    ) -> None:
        from PySide6.QtWidgets import QDialogButtonBox, QListWidget, QListWidgetItem
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(560, 420)
        vbox = QVBoxLayout(dlg)
        lbl = QLabel("Double-click a photo or select one and click OK to set as cover:", dlg)
        vbox.addWidget(lbl)

        list_w = QListWidget(dlg)
        list_w.setViewMode(QListWidget.ViewMode.IconMode)
        list_w.setResizeMode(QListWidget.ResizeMode.Adjust)
        list_w.setIconSize(QSize(112, 112))
        list_w.setGridSize(QSize(130, 144))
        list_w.setSpacing(8)
        list_w.setMovement(QListWidget.Movement.Static)

        from src.db.repository import ThumbnailRepository
        thumb_repo = ThumbnailRepository(self.engine)
        for cand in candidates[:120]:
            mid = int(cand["id"])
            item = QListWidgetItem(cand["filename"])
            item.setData(Qt.ItemDataRole.UserRole, mid)
            blob = thumb_repo.get_thumbnail_blob(mid)
            if blob:
                pm = QPixmap()
                if pm.loadFromData(blob):
                    item.setIcon(QIcon(pm.scaled(112, 112, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)))
            if cand.get("is_cover"):
                item.setToolTip("Current album cover")
                list_w.setCurrentItem(item)
            list_w.addItem(item)

        def _pick(it: QListWidgetItem) -> None:
            mid = it.data(Qt.ItemDataRole.UserRole)
            if mid is not None:
                on_picked(int(mid))
                dlg.accept()

        list_w.itemDoubleClicked.connect(_pick)
        vbox.addWidget(list_w, 1)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg)
        btns.accepted.connect(lambda: _pick(list_w.currentItem()) if list_w.currentItem() else dlg.reject())
        btns.rejected.connect(dlg.reject)
        vbox.addWidget(btns)
        dlg.exec()

    def _on_set_album_cover_requested(self, album_id: int, album_name: str) -> None:
        """Opens a thumbnail picker dialog to choose a cover photo for an album."""
        candidates = self.catalog_repo.get_album_cover_candidates(int(album_id))
        if not candidates:
            QMessageBox.information(self, "Set Album Cover", f"Album '{album_name}' has no photos.")
            return
        self._open_cover_picker_dialog(
            f"Set Album Cover — {album_name}",
            candidates,
            lambda mid: self.set_album_cover(int(album_id), mid),
        )

    def _on_set_smart_album_cover_requested(self, smart_album_id: int, album_name: str) -> None:
        """Opens a thumbnail picker dialog to choose a cover photo for a Smart Album."""
        candidates = self.catalog_repo.get_smart_album_cover_candidates(int(smart_album_id))
        if not candidates:
            QMessageBox.information(self, "Set Album Cover", f"Smart Album '{album_name}' has no photos.")
            return
        self._open_cover_picker_dialog(
            f"Set Album Cover — {album_name}",
            candidates,
            lambda mid: self.set_smart_album_cover(int(smart_album_id), mid),
        )

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

    def _on_central_stack_changed(self, _index: int = -1) -> None:
        self._update_status_grid_button()
        self._sync_window_title_for_current_view()

    def _sync_window_title_for_current_view(self) -> None:
        ws = getattr(self, "_active_create_workspace", None)
        if ws is not None:
            self.setWindowTitle(f"{ws.display_title} - Essentials Photo Gallery")
            return
        w = self.central_stack.currentWidget() if hasattr(self, "central_stack") else None
        if w is None:
            return
        if w == getattr(self, "media_viewer", None):
            cur_p = getattr(self.media_viewer, "current_file_path", None)
            if cur_p:
                self.setWindowTitle(f"{cur_p.name} - Essentials Photo Gallery")
            else:
                self.setWindowTitle("Essentials Photo Gallery")
        elif w == getattr(self, "map_view", None):
            self.setWindowTitle("Map - Essentials Photo Gallery")
        elif w == getattr(self, "people_directory_view", None):
            self.setWindowTitle("People - Essentials Photo Gallery")
        elif w == getattr(self, "albums_view", None):
            self.setWindowTitle("Albums - Essentials Photo Gallery")
        elif w == getattr(self, "compare_view", None):
            self.setWindowTitle("Compare - Essentials Photo Gallery")
        elif w == getattr(self, "batch_people_view", None):
            self.setWindowTitle("Batch People Tagging - Essentials Photo Gallery")
        elif w == getattr(self, "duplicate_finder_view", None):
            self.setWindowTitle("Find Duplicates & Bursts - Essentials Photo Gallery")
        else:
            self.setWindowTitle("Essentials Photo Gallery")

    def setWindowTitle(self, title: str) -> None:
        super().setWindowTitle(title)
        if hasattr(self, "ribbon") and hasattr(self.ribbon, "lbl_app_title"):
            self.ribbon.lbl_app_title.setText(title)

    def closeEvent(self, event) -> None:
        if getattr(self, "_active_create_workspace", None) is not None:
            if not self.close_create_workspace():
                event.ignore()
                return
        self._save_app_state()
        self.hide()
        self.shutdown()
        super().closeEvent(event)
