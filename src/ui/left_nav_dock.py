"""
LeftNavigationDock (QDockWidget) implementing Windows Live Photo Gallery's Unified Filter Tree:
- Folders (nested hierarchy with FavouritesMenu & 'Include subfolders' toggle)
- Date taken (Year -> Month -> 'Day, Weekday' with item counts, including pre-1970 dates)
- Descriptive tags (multi-level nested hierarchy with Boolean AND/OR and NOT exclusion)
- People (face/person filter nodes with media counts and face avatars)
- Ratings (5 stars .. 1 star, Unrated with counts)
"""

from __future__ import annotations

import calendar
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from PySide6.QtCore import QByteArray, QMimeData, QPoint, QSettings, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QCursor,
    QDrag,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QIcon,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from legacy_reference.imgBrowserv13 import FavouritesMenu
from src.db.repository import ArrangeMode, CatalogRepository, FilterQuerySpec
from src.ui import icon_provider
from src.ui.dialogs import TagMergeConfirmDialog
from src.ui.ribbon_toolbar import make_person_avatar_pixmap, make_wlpg_icon

_NODE_TYPE_ROLE = Qt.UserRole + 10
_NODE_PAYLOAD_ROLE = Qt.UserRole + 11
_TAG_FILTER_STATE_ROLE = Qt.UserRole + 12  # "neutral" | "include" | "exclude"
_OLD_NAME_ROLE = Qt.UserRole + 13
_OLD_PATH_ROLE = Qt.UserRole + 14
_HAS_AVATAR_ROLE = Qt.UserRole + 15


class _LeftNavTree(QTreeWidget):
    """Custom tree widget supporting Left and Right click drag-and-drop file moves/copies onto folder nodes,
    and hierarchical drag-and-drop of descriptive tags onto other tags or root."""

    files_dropped = Signal(int, str, list, bool)  # target_folder_id, target_folder_path, [file_path_str], is_copy
    tag_dropped = Signal(int, object)  # source_tag_id, target_tag_id (int | None)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        self._drop_hover_item: QTreeWidgetItem | None = None
        self._last_drag_buttons: Qt.MouseButtons = Qt.NoButton
        self._auto_right_drop_action: str | None = None  # For headless testing ("move" | "copy" | "cancel")
        self._drag_start_pos: QPoint | None = None
        self._drag_start_item: QTreeWidgetItem | None = None

        # Non-interfering child badge inside viewport (never steals WindowFromPoint during Windows OLE DoDragDrop)
        self._drag_badge = QLabel(self.viewport())
        self._drag_badge.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._apply_drag_badge_style(False)
        self._drag_badge.hide()

        self._drag_expand_timer = QTimer(self)
        self._drag_expand_timer.setSingleShot(True)
        self._drag_expand_timer.setInterval(450)
        self._drag_expand_timer.timeout.connect(self._on_drag_expand_timeout)

    def _apply_drag_badge_style(self, is_dark: bool = False) -> None:
        if is_dark:
            self._drag_badge.setStyleSheet(
                """
                QLabel {
                    background-color: #1E293B;
                    border: 1px solid #3B82F6;
                    border-radius: 3px;
                    color: #F8FAFC;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    font-weight: 600;
                    padding: 2px 8px;
                }
                """
            )
        else:
            self._drag_badge.setStyleSheet(
                """
                QLabel {
                    background-color: #FFFFFF;
                    border: 1px solid #0078D7;
                    border-radius: 3px;
                    color: #003399;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    font-weight: 600;
                    padding: 2px 8px;
                }
                """
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            pos = event.position().toPoint()
            item = self.itemAt(pos)
            if item is None:
                item = self.itemAt(QPoint(max(24, self.viewport().width() // 2), pos.y()))
            if item is not None and item.data(0, _NODE_TYPE_ROLE) == "tag":
                self._drag_start_pos = pos
                self._drag_start_item = item
            else:
                self._drag_start_pos = None
                self._drag_start_item = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start_item is not None and self._drag_start_pos is not None:
            dist = (event.position().toPoint() - self._drag_start_pos).manhattanLength()
            if dist >= QApplication.startDragDistance():
                item = self._drag_start_item
                self._drag_start_item = None
                self._drag_start_pos = None
                self._start_tag_drag(item)
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_start_item = None
        self._drag_start_pos = None
        super().mouseReleaseEvent(event)

    def _start_tag_drag(self, item: QTreeWidgetItem) -> None:
        tid = item.data(0, _NODE_PAYLOAD_ROLE)
        if tid is None:
            return
        tname = item.data(0, _OLD_NAME_ROLE) or item.text(0).split(" (")[0]

        drag = QDrag(self)
        mime = QMimeData()
        mime.setData("application/x-photogallery-tag-id", QByteArray(str(tid).encode()))
        mime.setText(tname)
        drag.setMimeData(mime)

        pix = self._create_tag_drag_pixmap(tname)
        drag.setPixmap(pix)
        drag.setHotSpot(QPoint(12, 12))

        drag.exec(Qt.MoveAction, Qt.MoveAction)

    def _create_tag_drag_pixmap(self, tag_name: str) -> QPixmap:
        font = QFont("Segoe UI", 9)
        fm = QFontMetrics(font)
        text_w = fm.horizontalAdvance(tag_name)
        h = 24
        w = 26 + text_w + 10
        pix = QPixmap(w, h)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setBrush(QColor(255, 255, 255, 235))
        p.setPen(QPen(QColor("#A0B8D0"), 1))
        p.drawRoundedRect(0, 0, w - 1, h - 1, 3, 3)

        tag_icon = make_wlpg_icon("tag", 14)
        if not tag_icon.isNull():
            p.drawPixmap(5, 5, tag_icon.pixmap(14, 14))
        p.setFont(font)
        p.setPen(QColor("#000000"))
        p.drawText(24, 16, tag_name)
        p.end()
        return pix

    def _is_valid_tag_drop_target(self, target_item: QTreeWidgetItem, source_tag_id: int) -> bool:
        ntype = target_item.data(0, _NODE_TYPE_ROLE)
        if ntype not in ("tag", "section_tags"):
            return False

        if ntype == "tag":
            target_id = target_item.data(0, _NODE_PAYLOAD_ROLE)
            # Cannot drop onto itself
            if target_id == source_tag_id:
                return False

            # Cannot drop onto a descendant of source tag
            cur = target_item.parent()
            while cur is not None:
                if cur.data(0, _NODE_TYPE_ROLE) == "tag" and cur.data(0, _NODE_PAYLOAD_ROLE) == source_tag_id:
                    return False
                cur = cur.parent()

        return True

    def _on_drag_expand_timeout(self) -> None:
        if (
            self._drop_hover_item is not None
            and self._drop_hover_item.childCount() > 0
            and not self._drop_hover_item.isExpanded()
        ):
            self._drop_hover_item.setExpanded(True)

    def _folder_item_at(self, pos: QPoint) -> QTreeWidgetItem | None:
        item = self.itemAt(pos)
        if item is None:
            item = self.itemAt(QPoint(max(24, self.viewport().width() // 2), pos.y()))
        if item is not None and item.data(0, _NODE_TYPE_ROLE) == "folder":
            return item
        return None

    def _clear_drag_hover(self) -> None:
        self._drag_expand_timer.stop()
        self._drag_badge.hide()
        if self._drop_hover_item is not None:
            self._drop_hover_item = None
            self.viewport().update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._drop_hover_item is not None:
            try:
                rect = self.visualItemRect(self._drop_hover_item)
                if rect.isValid():
                    p = QPainter(self.viewport())
                    p.setRenderHint(QPainter.Antialiasing, False)
                    box = rect.adjusted(1, 0, -2, -1)
                    p.setBrush(QColor(204, 232, 255, 95))
                    p.setPen(QPen(QColor("#0078D7"), 1))
                    p.drawRect(box)
                    p.end()
            except (RuntimeError, Exception):
                self._drop_hover_item = None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasFormat("application/x-photogallery-tag-id"):
            event.setDropAction(Qt.MoveAction)
            event.acceptProposedAction()
            return
        if event.mimeData().hasUrls():
            if event.buttons() != Qt.NoButton:
                self._last_drag_buttons = event.buttons()
            is_copy = bool(event.modifiers() & Qt.ControlModifier)
            action = Qt.CopyAction if is_copy else Qt.MoveAction
            if action in event.possibleActions():
                event.setDropAction(action)
                event.accept()
            else:
                event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasFormat("application/x-photogallery-tag-id"):
            pos = event.position().toPoint()
            target_item = self.itemAt(pos)
            if target_item is None:
                target_item = self.itemAt(QPoint(max(24, self.viewport().width() // 2), pos.y()))

            source_id_bytes = event.mimeData().data("application/x-photogallery-tag-id")
            source_tag_id = int(source_id_bytes.data().decode()) if source_id_bytes else -1

            if target_item is not None and self._is_valid_tag_drop_target(target_item, source_tag_id):
                if self._drop_hover_item is not target_item:
                    self._drop_hover_item = target_item
                    self.viewport().update()
                    if target_item.childCount() > 0 and not target_item.isExpanded():
                        self._drag_expand_timer.start()
                    else:
                        self._drag_expand_timer.stop()

                ntype = target_item.data(0, _NODE_TYPE_ROLE)
                if ntype == "section_tags":
                    target_name = "Descriptive tags"
                else:
                    target_name = target_item.data(0, _OLD_NAME_ROLE) or target_item.text(0).split(" (")[0]

                self._drag_badge.setText(f"➔ Move to {target_name}")
                self._drag_badge.adjustSize()
                bx = min(
                    max(6, pos.x() + 10),
                    max(6, self.viewport().width() - self._drag_badge.width() - 4),
                )
                by = max(2, pos.y() + 18)
                if by + self._drag_badge.height() > self.viewport().height() - 2:
                    by = max(2, pos.y() - self._drag_badge.height() - 22)
                self._drag_badge.move(bx, by)
                self._drag_badge.show()
                self._drag_badge.raise_()

                event.setDropAction(Qt.MoveAction)
                event.acceptProposedAction()
                return
            else:
                self._clear_drag_hover()
                event.ignore()
                return

        if event.mimeData().hasUrls():
            if event.buttons() != Qt.NoButton:
                self._last_drag_buttons = event.buttons()
            pos = event.position().toPoint()
            item = self._folder_item_at(pos)
            if item is not None:
                if self._drop_hover_item is not item:
                    self._drop_hover_item = item
                    self.viewport().update()
                    if item.childCount() > 0 and not item.isExpanded():
                        self._drag_expand_timer.start()
                    else:
                        self._drag_expand_timer.stop()

                folder_name = item.text(0)
                is_copy = bool(event.modifiers() & Qt.ControlModifier)
                verb = "Copy" if is_copy else "Move"
                self._drag_badge.setText(f"➔ {verb} to {folder_name}")
                self._drag_badge.adjustSize()
                bx = min(
                    max(6, pos.x() + 10),
                    max(6, self.viewport().width() - self._drag_badge.width() - 4),
                )
                by = max(2, pos.y() + 18)
                if by + self._drag_badge.height() > self.viewport().height() - 2:
                    by = max(2, pos.y() - self._drag_badge.height() - 80)
                self._drag_badge.move(bx, by)
                self._drag_badge.show()
                self._drag_badge.raise_()

                action = Qt.CopyAction if is_copy else Qt.MoveAction
                if action in event.possibleActions():
                    event.setDropAction(action)
                    event.accept()
                else:
                    event.acceptProposedAction()
                return
            else:
                self._clear_drag_hover()
                event.ignore()
                return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self._clear_drag_hover()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        pos = event.position().toPoint()
        item = self.itemAt(pos)
        if item is None:
            item = self.itemAt(QPoint(max(24, self.viewport().width() // 2), pos.y()))
        target_item = item or self._drop_hover_item
        self._clear_drag_hover()

        if event.mimeData().hasFormat("application/x-photogallery-tag-id") and target_item is not None:
            source_id_bytes = event.mimeData().data("application/x-photogallery-tag-id")
            source_tag_id = int(source_id_bytes.data().decode()) if source_id_bytes else -1
            if self._is_valid_tag_drop_target(target_item, source_tag_id):
                ntype = target_item.data(0, _NODE_TYPE_ROLE)
                target_tag_id = (
                    int(target_item.data(0, _NODE_PAYLOAD_ROLE))
                    if ntype == "tag"
                    else None
                )
                self.tag_dropped.emit(source_tag_id, target_tag_id)
                event.setDropAction(Qt.MoveAction)
                event.acceptProposedAction()
                return

        if event.mimeData().hasUrls() and target_item is not None and target_item.data(0, _NODE_TYPE_ROLE) == "folder":
            fid = target_item.data(0, _NODE_PAYLOAD_ROLE)
            fpath = target_item.data(0, _OLD_PATH_ROLE) or target_item.toolTip(0)
            file_paths = [
                url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()
            ]
            if fid is not None and file_paths:
                from src.ui.grid_view import ThumbnailGridView
                is_right = (
                    event.mimeData().data("application/x-photogallery64-drag-button") == b"right"
                    or getattr(ThumbnailGridView, "_active_drag_button", Qt.LeftButton) == Qt.RightButton
                    or bool(self._last_drag_buttons & Qt.RightButton)
                    or bool(event.buttons() & Qt.RightButton)
                )
                self._last_drag_buttons = Qt.NoButton

                if is_right:
                    menu = QMenu(self)
                    menu.setStyleSheet("""
                        QMenu {
                            background-color: #FFFFFF;
                            border: 1px solid #7192BF;
                            font-family: 'Segoe UI';
                            font-size: 9pt;
                            padding: 2px;
                        }
                        QMenu::item {
                            padding: 4px 22px 4px 18px;
                        }
                        QMenu::item:selected {
                            background-color: #CCE8FF;
                            color: #000000;
                        }
                    """)
                    act_move = menu.addAction("Move here")
                    f = act_move.font()
                    f.setBold(True)
                    act_move.setFont(f)
                    act_copy = menu.addAction("Copy here")
                    menu.addSeparator()
                    act_cancel = menu.addAction("Cancel")

                    if self._auto_right_drop_action == "move":
                        chosen = act_move
                    elif self._auto_right_drop_action == "copy":
                        chosen = act_copy
                    elif self._auto_right_drop_action == "cancel":
                        chosen = act_cancel
                    else:
                        chosen = menu.exec(QCursor.pos())

                    if chosen == act_move:
                        event.setDropAction(Qt.MoveAction)
                        event.accept()
                        self.files_dropped.emit(int(fid), str(fpath), file_paths, False)
                    elif chosen == act_copy:
                        event.setDropAction(Qt.CopyAction)
                        event.accept()
                        self.files_dropped.emit(int(fid), str(fpath), file_paths, True)
                    else:
                        event.ignore()
                    return
                else:
                    is_copy = bool(event.modifiers() & Qt.ControlModifier)
                    event.setDropAction(Qt.CopyAction if is_copy else Qt.MoveAction)
                    event.accept()
                    self.files_dropped.emit(int(fid), str(fpath), file_paths, is_copy)
                    return
        self._last_drag_buttons = Qt.NoButton
        event.ignore()


class LeftNavigationDock(QDockWidget):
    """
    Unified collapsible Left Navigation Pane matching WLPG's main_ui.png and find_ui.png.
    Produces an immutable FilterQuerySpec in <1ms whenever any filter node is clicked or toggled.
    """

    filter_changed = Signal()
    folder_add_requested = Signal()
    folder_remove_requested = Signal(int, str, str)  # folder_id, folder_path, folder_name
    folder_renamed = Signal(str, str, str)  # old_path, new_path, new_name
    folder_created = Signal(str)  # new_folder_path
    folder_deleted = Signal(int, str)  # folder_id, folder_path
    favourite_path_selected = Signal(str)
    files_dropped_on_folder = Signal(int, str, list, bool)
    people_directory_requested = Signal()
    albums_view_requested = Signal()
    section_order_changed = Signal(list)
    tag_hierarchy_changed = Signal()
    place_selected = Signal(str, str, str)  # country, state, city
    new_album_requested = Signal()
    rename_album_requested = Signal(int, str)
    delete_album_requested = Signal(int, str)
    new_smart_album_requested = Signal()
    edit_smart_album_requested = Signal(int)
    rename_smart_album_requested = Signal(int, str)
    delete_smart_album_requested = Signal(int, str)
    save_filter_as_smart_album_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Navigation", parent)
        self.setObjectName("LeftNavigationDock")
        self.setFeatures(
            QDockWidget.DockWidgetClosable
        )
        # Hide the dock titlebar so the left pane starts cleanly with the tree like WLPG
        self.setTitleBarWidget(QWidget(self))
        self.setMinimumWidth(200)

        # Internal filter state
        self._folder_id: int | None = None
        self._folder_name: str = "All folders"
        self._include_subfolders: bool = True
        self._date_year: int | None = None
        self._date_month: int | None = None
        self._date_day: int | None = None
        self._include_tag_ids: set[int] = set()
        self._exclude_tag_ids: set[int] = set()
        self._tag_boolean_mode: Literal["AND", "OR"] = "AND"
        self._untagged_only: bool = False
        self._person_ids: set[int] = set()
        self._person_boolean_mode: Literal["AND", "OR"] = "OR"
        self._ratings: set[int] = set()
        self._flagged_only: bool = False
        self._edited_only: bool = False
        self._edited_filter: int | None = None
        self._media_type: Literal["all", "image", "video", "raw"] = "all"
        self._orientation_filter: Literal["all", "landscape", "portrait"] = "all"
        self._favourite_paths: list[str] = []
        self._location_country: str | None = None
        self._location_state: str | None = None
        self._location_city: str | None = None
        self._album_id: int | None = None
        self._album_name: str = ""
        self._smart_album_id: int | None = None
        self._smart_album_name: str = ""
        self._smart_album_rules: str | None = None

        # Section visibility state (Options dialog)
        self._show_albums: bool = True
        self._show_dates: bool = True
        self._show_tags: bool = True
        self._show_people: bool = True
        self._show_ratings: bool = True
        self._show_places: bool = True
        self._default_section_order: tuple[str, ...] = (
            "all_media",
            "section_albums",
            "section_dates",
            "section_people",
            "section_tags",
            "section_ratings",
            "section_places",
        )
        self._section_order: list[str] = list(self._default_section_order)

        # Fast lookup maps for tree items
        self._album_items: dict[int, QTreeWidgetItem] = {}
        self._smart_album_items: dict[int, QTreeWidgetItem] = {}
        self._folder_items: dict[int, QTreeWidgetItem] = {}
        self._tag_items: dict[int, QTreeWidgetItem] = {}
        self._person_items: dict[int, QTreeWidgetItem] = {}
        self._rating_items: dict[int, QTreeWidgetItem] = {}
        self._person_avatar_icon_cache: dict[tuple[int, int], QIcon] = {}
        self._current_theme_mode: str = "wlpg_aero"
        self._updating_tree: bool = False
        self._checkbox_just_changed: bool = False

        # Build UI
        container = QFrame(self)
        container.setObjectName("LeftNavContainer")
        container.setMinimumWidth(200)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Unified QTreeWidget occupying the primary vertical area (matches main_ui.png)
        self.tree = _LeftNavTree(container)
        self.tree.setObjectName("LeftNavTree")
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(16)
        self.tree.setIconSize(QSize(16, 16))
        self.tree.setAnimated(True)
        self.tree.setFrameShape(QFrame.NoFrame)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.files_dropped.connect(self.files_dropped_on_folder)
        self.tree.tag_dropped.connect(self._on_tag_dropped)
        layout.addWidget(self.tree, 1)

        # Clean bottom footer with only the Subfolders toggle (matching WLPG style)
        footer = QFrame(container)
        footer.setObjectName("LeftNavFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(10, 4, 10, 5)
        footer_layout.setSpacing(4)

        self.chk_subfolders = QCheckBox("Subfolders", footer)
        self.chk_subfolders.setToolTip("Include subfolders when a folder is selected")
        self.chk_subfolders.setChecked(True)
        self.chk_subfolders.toggled.connect(self._on_subfolders_toggled)
        footer_layout.addWidget(self.chk_subfolders)
        footer_layout.addStretch(1)

        layout.addWidget(footer, 0)

        self.setWidget(container)
        self._apply_stylesheet()
        self._init_root_sections()
        icon_provider.signals.theme_changed.connect(self._on_icon_theme_changed)

    def sizeHint(self) -> QSize:
        return QSize(230, 600)

    def _on_icon_theme_changed(self, _theme_name: str = "") -> None:
        self.refresh_icons()

    def apply_theme_mode(self, theme_name: str) -> None:
        self._current_theme_mode = str(theme_name)
        is_dark = theme_name == "dark"
        self._dark_mode = is_dark
        self._is_dark = is_dark
        if is_dark:
            self.set_dark_mode(True)
        elif theme_name == "modern":
            self.setStyleSheet(
                """
                QFrame#LeftNavContainer {
                    background: #FAFAFA;
                    border-right: 1px solid #E5E7EB;
                }
                QTreeWidget#LeftNavTree {
                    background: #FAFAFA;
                    border: none;
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                    color: #1F2937;
                    outline: none;
                }
                QTreeWidget#LeftNavTree::item {
                    padding: 2px 4px;
                    border: 1px solid transparent;
                    border-radius: 3px;
                    color: #1F2937;
                }
                QTreeWidget#LeftNavTree::item:hover {
                    background: #EFF6FF;
                    border: 1px solid #BFDBFE;
                }
                QTreeWidget#LeftNavTree::item:selected {
                    background: #DBEAFE;
                    border: 1px solid #60A5FA;
                    color: #1E3A8A;
                }
                QFrame#LeftNavFooter {
                    background: #F3F4F6;
                    border-top: 1px solid #E5E7EB;
                }
                """
            )
            self.tree._apply_drag_badge_style(False)
            self._sync_tree_item_colors()
        else:
            self.set_dark_mode(False)
        self.refresh_icons()

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet(
            """
            QFrame#LeftNavContainer {
                background: #FFFFFF;
                border-right: 1px solid #C5D5E6;
            }
            QTreeWidget#LeftNavTree {
                background: #FFFFFF;
                border: none;
                font-family: 'Segoe UI';
                font-size: 9pt;
                color: #1E395B;
                outline: none;
            }
            QTreeWidget#LeftNavTree::item {
                padding: 2px 4px;
                border: 1px solid transparent;
                border-radius: 2px;
            }
            QTreeWidget#LeftNavTree::item:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F4F9FE, stop:1 #E4F0FC);
                border: 1px solid #B8D6F5;
            }
            QTreeWidget#LeftNavTree::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #DCEBFC, stop:1 #C1DCFA);
                border: 1px solid #84ACDD;
                color: #1E395B;
            }
            QFrame#LeftNavFooter {
                background: #F3F7FC;
                border-top: 1px solid #D5E1F0;
            }
            """
        )

    def set_dark_mode(self, is_dark: bool) -> None:
        self._dark_mode = bool(is_dark)
        self._is_dark = self._dark_mode
        if not self._dark_mode:
            self._apply_stylesheet()
        else:
            self.setStyleSheet(
                """
                QFrame#LeftNavContainer {
                    background: #1B1E24;
                    border-right: 1px solid #2E3440;
                }
                QTreeWidget#LeftNavTree {
                    background: #1B1E24;
                    border: none;
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                    color: #E2E8F0;
                    outline: none;
                }
                QTreeWidget#LeftNavTree::item {
                    padding: 2px 4px;
                    border: 1px solid transparent;
                    border-radius: 2px;
                    color: #E2E8F0;
                }
                QTreeWidget#LeftNavTree::item:hover {
                    background: #262C36;
                    border: 1px solid #3E4756;
                }
                QTreeWidget#LeftNavTree::item:selected {
                    background: #1E3A5F;
                    border: 1px solid #3B82F6;
                    color: #FFFFFF;
                }
                QFrame#LeftNavFooter {
                    background: #16191E;
                    border-top: 1px solid #2E3440;
                }
                QFrame#LeftNavFooter QCheckBox {
                    color: #D8E0EA;
                }
                """
            )
        if hasattr(self, "tree") and hasattr(self.tree, "_apply_drag_badge_style"):
            self.tree._apply_drag_badge_style(self._dark_mode)
        self._sync_tree_item_colors()

    def _sync_tree_item_colors(self) -> None:
        is_dark = getattr(self, "_dark_mode", False)
        sec_brush = QBrush(QColor("#93C5FD" if is_dark else "#1E395B"))
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            if top is not None:
                top.setForeground(0, sec_brush)

    def refresh_icons(self) -> None:
        """Updates all tree item icons in-place when the icon theme changes in Options."""
        try:
            if not hasattr(self, "tree") or self.tree is None:
                return
            self.tree.blockSignals(True)
        except RuntimeError:
            return
        try:
            folder_icon = make_wlpg_icon("folder", 16)
            pic_lib_icon = make_wlpg_icon("pictures_library", 16)
            vid_lib_icon = make_wlpg_icon("videos_library", 16)
            cal_icon = make_wlpg_icon("calendar", 16)
            tag_icon = make_wlpg_icon("tag", 16)
            person_icon = make_wlpg_icon("person", 16)
            star_icon = make_wlpg_icon("star_filled", 16)
            empty_star_icon = make_wlpg_icon("star_empty", 16)
            places_icon = make_wlpg_icon("geotag", 16)

            if hasattr(self, "root_all") and self.root_all is not None:
                self.root_all.setIcon(0, make_wlpg_icon("all_media", 16))
            if hasattr(self, "root_dates") and self.root_dates is not None:
                self.root_dates.setIcon(0, cal_icon)
            if hasattr(self, "root_people") and self.root_people is not None:
                self.root_people.setIcon(0, person_icon)
            if hasattr(self, "root_tags") and self.root_tags is not None:
                self.root_tags.setIcon(0, tag_icon)
            if hasattr(self, "root_ratings") and self.root_ratings is not None:
                self.root_ratings.setIcon(0, star_icon)
            if hasattr(self, "root_places") and self.root_places is not None:
                self.root_places.setIcon(0, places_icon)

            for item in self._folder_items.values():
                fname_lower = (item.data(0, _OLD_NAME_ROLE) or item.text(0) or "").lower()
                if "picture" in fname_lower:
                    item.setIcon(0, pic_lib_icon)
                elif "video" in fname_lower:
                    item.setIcon(0, vid_lib_icon)
                else:
                    item.setIcon(0, folder_icon)

            def _update_recursive_icons(parent_item: QTreeWidgetItem, icn: QIcon) -> None:
                for idx in range(parent_item.childCount()):
                    ch = parent_item.child(idx)
                    if ch is not None:
                        ch.setIcon(0, icn)
                        _update_recursive_icons(ch, icn)

            if hasattr(self, "root_dates") and self.root_dates is not None:
                _update_recursive_icons(self.root_dates, cal_icon)

            if hasattr(self, "item_untagged") and self.item_untagged is not None:
                self.item_untagged.setIcon(0, tag_icon)
            for item in self._tag_items.values():
                item.setIcon(0, tag_icon)

            for item in self._person_items.values():
                if not item.data(0, _HAS_AVATAR_ROLE):
                    item.setIcon(0, person_icon)

            for r_val, item in self._rating_items.items():
                item.setIcon(0, star_icon if r_val > 0 else empty_star_icon)

            if hasattr(self, "root_places") and self.root_places is not None:
                _update_recursive_icons(self.root_places, places_icon)
        except RuntimeError:
            return
        finally:
            try:
                self.tree.blockSignals(False)
            except RuntimeError:
                return
        try:
            self.tree.viewport().update()
        except RuntimeError:
            pass

    def _init_root_sections(self) -> None:
        self.tree.blockSignals(True)
        try:
            self.tree.clear()
            self._album_items.clear()
            self._smart_album_items.clear()
            self._folder_items.clear()
            self._tag_items.clear()
            self._person_items.clear()
            self._rating_items.clear()
            self.item_untagged = None

            sec_font = QFont("Segoe UI", 9, QFont.Bold)
            section_color = QBrush(QColor("#1E395B"))

            def _build_albums() -> None:
                self.root_albums = QTreeWidgetItem(self.tree, ["Albums"])
                self.root_albums.setData(0, _NODE_TYPE_ROLE, "section_albums")
                self.root_albums.setIcon(0, make_wlpg_icon("pictures_library", 16))
                self.root_albums.setFont(0, sec_font)
                self.root_albums.setForeground(0, section_color)
                self.root_albums.setExpanded(True)

                self.root_my_albums = QTreeWidgetItem(self.root_albums, ["My Albums"])
                self.root_my_albums.setData(0, _NODE_TYPE_ROLE, "section_my_albums")
                self.root_my_albums.setIcon(0, make_wlpg_icon("folder", 16))
                self.root_my_albums.setFont(0, sec_font)
                self.root_my_albums.setForeground(0, section_color)
                self.root_my_albums.setExpanded(True)

                self.root_smart_albums = QTreeWidgetItem(self.root_albums, ["Smart Albums"])
                self.root_smart_albums.setData(0, _NODE_TYPE_ROLE, "section_smart_albums")
                self.root_smart_albums.setIcon(0, make_wlpg_icon("star_filled", 16))
                self.root_smart_albums.setFont(0, sec_font)
                self.root_smart_albums.setForeground(0, section_color)
                self.root_smart_albums.setExpanded(True)

            def _build_all() -> None:
                self.root_all = QTreeWidgetItem(self.tree, ["All photos and videos"])
                self.root_all.setData(0, _NODE_TYPE_ROLE, "all_media")
                self.root_all.setIcon(0, make_wlpg_icon("all_media", 16))
                self.root_all.setFont(0, sec_font)
                self.root_all.setForeground(0, section_color)
                self.root_all.setExpanded(True)
                self.root_folders = self.root_all

            def _build_dates() -> None:
                self.root_dates = QTreeWidgetItem(self.tree, ["Date taken"])
                self.root_dates.setData(0, _NODE_TYPE_ROLE, "section_dates")
                self.root_dates.setIcon(0, make_wlpg_icon("calendar", 16))
                self.root_dates.setFont(0, sec_font)
                self.root_dates.setForeground(0, section_color)
                self.root_dates.setExpanded(True)

            def _build_people() -> None:
                self.root_people = QTreeWidgetItem(self.tree, ["People"])
                self.root_people.setData(0, _NODE_TYPE_ROLE, "section_people")
                self.root_people.setIcon(0, make_wlpg_icon("person", 16))
                self.root_people.setFont(0, sec_font)
                self.root_people.setForeground(0, section_color)
                self.root_people.setExpanded(False)

            def _build_tags() -> None:
                self.root_tags = QTreeWidgetItem(self.tree, ["Descriptive tags"])
                self.root_tags.setData(0, _NODE_TYPE_ROLE, "section_tags")
                self.root_tags.setIcon(0, make_wlpg_icon("tag", 16))
                self.root_tags.setFont(0, sec_font)
                self.root_tags.setForeground(0, section_color)
                self.root_tags.setExpanded(True)

            def _build_ratings() -> None:
                self.root_ratings = QTreeWidgetItem(self.tree, ["Ratings"])
                self.root_ratings.setData(0, _NODE_TYPE_ROLE, "section_ratings")
                self.root_ratings.setIcon(0, make_wlpg_icon("star_filled", 16))
                self.root_ratings.setFont(0, sec_font)
                self.root_ratings.setForeground(0, section_color)
                self.root_ratings.setExpanded(False)

            def _build_places() -> None:
                self.root_places = QTreeWidgetItem(self.tree, ["Places"])
                self.root_places.setData(0, _NODE_TYPE_ROLE, "section_places")
                self.root_places.setIcon(0, make_wlpg_icon("geotag", 16))
                self.root_places.setFont(0, sec_font)
                self.root_places.setForeground(0, section_color)
                self.root_places.setExpanded(False)

            builders = {
                "all_media": _build_all,
                "section_albums": _build_albums,
                "section_dates": _build_dates,
                "section_people": _build_people,
                "section_tags": _build_tags,
                "section_ratings": _build_ratings,
                "section_places": _build_places,
            }
            for key in self._section_order:
                if key in builders:
                    builders[key]()
            for key, fn in builders.items():
                if key not in self._section_order:
                    fn()

            # Apply user-configured navigation section visibilities
            if hasattr(self, "_show_albums"):
                self.root_albums.setHidden(not self._show_albums)
            if hasattr(self, "_show_dates"):
                self.root_dates.setHidden(not self._show_dates)
            if hasattr(self, "_show_people"):
                self.root_people.setHidden(not self._show_people)
            if hasattr(self, "_show_tags"):
                self.root_tags.setHidden(not self._show_tags)
            if hasattr(self, "_show_ratings"):
                self.root_ratings.setHidden(not self._show_ratings)
            if hasattr(self, "_show_places"):
                self.root_places.setHidden(not self._show_places)
        finally:
            self.tree.blockSignals(False)

    def set_navigation_sections_visibility(
        self,
        show_dates: bool = True,
        show_tags: bool = True,
        show_people: bool = True,
        show_ratings: bool = True,
        show_places: bool = True,
        show_albums: bool = True,
    ) -> None:
        """Shows or hides top-level navigation sections per Photo Gallery Options dialog."""
        self._show_albums = bool(show_albums)
        self._show_dates = bool(show_dates)
        self._show_tags = bool(show_tags)
        self._show_people = bool(show_people)
        self._show_ratings = bool(show_ratings)
        self._show_places = bool(show_places)
        if hasattr(self, "root_albums"):
            self.root_albums.setHidden(not self._show_albums)
        if hasattr(self, "root_dates"):
            self.root_dates.setHidden(not self._show_dates)
        if hasattr(self, "root_tags"):
            self.root_tags.setHidden(not self._show_tags)
        if hasattr(self, "root_people"):
            self.root_people.setHidden(not self._show_people)
        if hasattr(self, "root_ratings"):
            self.root_ratings.setHidden(not self._show_ratings)
        if hasattr(self, "root_places"):
            self.root_places.setHidden(not self._show_places)

    def set_section_visible(self, section_name: str, visible: bool) -> None:
        """Toggles visibility of an individual section by name ('albums', 'dates', 'tags', 'people', 'ratings', 'places')."""
        sec = section_name.lower()
        if "album" in sec:
            self._show_albums = bool(visible)
            if hasattr(self, "root_albums"):
                self.root_albums.setHidden(not self._show_albums)
        elif "date" in sec:
            self._show_dates = bool(visible)
            if hasattr(self, "root_dates"):
                self.root_dates.setHidden(not self._show_dates)
        elif "tag" in sec:
            self._show_tags = bool(visible)
            if hasattr(self, "root_tags"):
                self.root_tags.setHidden(not self._show_tags)
        elif "people" in sec or "person" in sec:
            self._show_people = bool(visible)
            if hasattr(self, "root_people"):
                self.root_people.setHidden(not self._show_people)
        elif "rating" in sec:
            self._show_ratings = bool(visible)
            if hasattr(self, "root_ratings"):
                self.root_ratings.setHidden(not self._show_ratings)
        elif "place" in sec:
            self._show_places = bool(visible)
            if hasattr(self, "root_places"):
                self.root_places.setHidden(not self._show_places)

    def get_section_order(self) -> list[str]:
        """Return the current top-level navigation section order."""
        return list(self._section_order)

    def set_section_order(self, order: Sequence[str]) -> None:
        """Apply a saved top-level section order and reorder existing tree roots in place."""
        valid = set(self._default_section_order)
        cleaned = [s for s in order if s in valid]
        for s in self._default_section_order:
            if s not in cleaned:
                cleaned.append(s)
        if cleaned == self._section_order:
            return
        self._section_order = cleaned
        self._apply_section_order_to_tree()

    def move_section(self, section_type: str, delta: int) -> None:
        """Move a top-level section up (-1) or down (+1) in the navigation tree."""
        if section_type not in self._section_order:
            return
        idx = self._section_order.index(section_type)
        new_idx = idx + delta
        if not (0 <= new_idx < len(self._section_order)):
            return
        self._section_order[idx], self._section_order[new_idx] = (
            self._section_order[new_idx],
            self._section_order[idx],
        )
        self._apply_section_order_to_tree()
        self.section_order_changed.emit(list(self._section_order))

    def reset_section_order(self) -> None:
        """Restore the default navigation section order (Albums directly under All photos and videos)."""
        self._section_order = list(self._default_section_order)
        self._apply_section_order_to_tree()
        self.section_order_changed.emit(list(self._section_order))

    def _apply_section_order_to_tree(self) -> None:
        """Reorder existing top-level QTreeWidgetItems in place without losing children or expanded state."""
        root_map = {
            "all_media": getattr(self, "root_all", None),
            "section_albums": getattr(self, "root_albums", None),
            "section_dates": getattr(self, "root_dates", None),
            "section_people": getattr(self, "root_people", None),
            "section_tags": getattr(self, "root_tags", None),
            "section_ratings": getattr(self, "root_ratings", None),
            "section_places": getattr(self, "root_places", None),
        }
        self.tree.blockSignals(True)
        try:
            target_pos = 0
            for key in self._section_order:
                item = root_map.get(key)
                if item is None:
                    continue
                cur_idx = self.tree.indexOfTopLevelItem(item)
                if cur_idx < 0:
                    continue
                if cur_idx != target_pos:
                    was_exp = item.isExpanded()
                    taken = self.tree.takeTopLevelItem(cur_idx)
                    if taken is not None:
                        self.tree.insertTopLevelItem(target_pos, taken)
                        taken.setExpanded(was_exp)
                target_pos += 1
        finally:
            self.tree.blockSignals(False)

    # -------------------------------------------------------------------------
    # Populate / Refresh from CatalogRepository
    # -------------------------------------------------------------------------
    def refresh_trees(
        self,
        repo: CatalogRepository,
        preloaded_people: list[dict[str, Any]] | None = None,
        preloaded_tags: list[dict[str, Any]] | None = None,
    ) -> None:
        """Populates Folders, Date Taken, Descriptive Tags, People, and Ratings from catalog.db."""
        self._repo = repo
        self._updating_tree = True
        self.tree.blockSignals(True)
        self.tree.setUpdatesEnabled(False)
        try:
            had_existing_folders = bool(self._folder_items)
            was_people_expanded = self.root_people.isExpanded() if hasattr(self, "root_people") else False
            was_albums_expanded = self.root_albums.isExpanded() if hasattr(self, "root_albums") else True
            was_my_albums_expanded = self.root_my_albums.isExpanded() if hasattr(self, "root_my_albums") else True
            was_smart_albums_expanded = (
                self.root_smart_albums.isExpanded() if hasattr(self, "root_smart_albums") else True
            )
            expanded_folder_ids = {
                fid for fid, itm in self._folder_items.items() if itm.isExpanded()
            }
            self._init_root_sections()
            self.tree.blockSignals(True)

            self._populate_albums_section(repo)
            self.root_albums.setExpanded(was_albums_expanded)
            self.root_my_albums.setExpanded(was_my_albums_expanded)
            self.root_smart_albums.setExpanded(was_smart_albums_expanded)

            folder_icon = make_wlpg_icon("folder", 16)
            cal_icon = make_wlpg_icon("calendar", 16)
            tag_icon = make_wlpg_icon("tag", 16)
            person_icon = make_wlpg_icon("person", 16)
            star_icon = make_wlpg_icon("star_filled", 16)
            empty_star_icon = make_wlpg_icon("star_empty", 16)

            # 1. Populate Folders hierarchy under All photos and videos
            folders = repo.get_folders()
            for f in folders:
                fid = int(f["id"])
                fname = str(f["name"])
                fpath_str = str(f["path"])
                fname_lower = fname.lower()
                if "picture" in fname_lower:
                    f_icon = make_wlpg_icon("pictures_library", 16)
                elif "video" in fname_lower:
                    f_icon = make_wlpg_icon("videos_library", 16)
                else:
                    f_icon = folder_icon

                item = QTreeWidgetItem([fname])
                item.setData(0, _NODE_TYPE_ROLE, "folder")
                item.setData(0, _NODE_PAYLOAD_ROLE, fid)
                item.setData(0, _OLD_NAME_ROLE, fname)
                item.setData(0, _OLD_PATH_ROLE, fpath_str)
                item.setIcon(0, f_icon)
                item.setToolTip(0, fpath_str)
                self._folder_items[fid] = item

            for f in folders:
                fid = int(f["id"])
                pid = f["parent_id"]
                item = self._folder_items[fid]
                if pid is not None and int(pid) in self._folder_items:
                    self._folder_items[int(pid)].addChild(item)
                else:
                    self.root_all.addChild(item)

            for fid, item in self._folder_items.items():
                if fid in expanded_folder_ids:
                    item.setExpanded(True)

            # 2. Populate Date Taken hierarchy (Year -> Month -> Day, Weekday)
            date_rows = repo.get_date_tree_counts()
            # Strict nested aggregation: year -> month -> list of (day, day_label, cnt)
            years_data: dict[int, dict[int, list[tuple[int, str, int]]]] = {}
            for d in date_rows:
                y = int(d["year"])
                m = int(d["month"])
                day = int(d["day"])
                day_label = str(d["day_label"])
                cnt = int(d["count"])
                years_data.setdefault(y, {}).setdefault(m, []).append((day, day_label, cnt))

            for y in sorted(years_data.keys(), reverse=True):
                months_data = years_data[y]
                year_total = sum(cnt for m_days in months_data.values() for (_, _, cnt) in m_days)
                y_item = QTreeWidgetItem([f"{y} ({year_total})"])
                y_item.setData(0, _NODE_TYPE_ROLE, "date")
                y_item.setData(0, _NODE_PAYLOAD_ROLE, (y, None, None))
                y_item.setIcon(0, cal_icon)
                self.root_dates.addChild(y_item)

                for m in sorted(months_data.keys(), reverse=True):
                    days_list = months_data[m]
                    month_total = sum(cnt for (_, _, cnt) in days_list)
                    m_name = calendar.month_name[m] if 1 <= m <= 12 else f"Month {m}"
                    m_item = QTreeWidgetItem([f"{m_name} ({month_total})"])
                    m_item.setData(0, _NODE_TYPE_ROLE, "date")
                    m_item.setData(0, _NODE_PAYLOAD_ROLE, (y, m, None))
                    m_item.setIcon(0, cal_icon)
                    y_item.addChild(m_item)

                    for day, day_label, cnt in sorted(days_list, key=lambda x: x[0], reverse=True):
                        d_item = QTreeWidgetItem([f"{day_label} ({cnt})"])
                        d_item.setData(0, _NODE_TYPE_ROLE, "date")
                        d_item.setData(0, _NODE_PAYLOAD_ROLE, (y, m, day))
                        d_item.setIcon(0, cal_icon)
                        m_item.addChild(d_item)

            # 3. Populate Descriptive Tags hierarchy + "Not tagged" node
            self.item_untagged = QTreeWidgetItem(["Not tagged"])
            self.item_untagged.setData(0, _NODE_TYPE_ROLE, "untagged")
            self.item_untagged.setIcon(0, tag_icon)
            if self._untagged_only:
                self.item_untagged.setSelected(True)
            self.root_tags.addChild(self.item_untagged)

            tag_rows = preloaded_tags if preloaded_tags is not None else repo.get_tag_tree()
            for t in tag_rows:
                tid = int(t["id"])
                tname = str(t["name"])
                cnt = int(t["direct_count"])
                label = f"{tname} ({cnt})" if cnt > 0 else tname
                t_item = QTreeWidgetItem([label])
                t_item.setData(0, _NODE_TYPE_ROLE, "tag")
                t_item.setData(0, _NODE_PAYLOAD_ROLE, tid)
                t_item.setData(0, _TAG_FILTER_STATE_ROLE, "neutral")
                t_item.setData(0, _OLD_NAME_ROLE, tname)
                t_item.setData(0, _OLD_PATH_ROLE, str(t["full_path"]))
                t_item.setIcon(0, tag_icon)
                t_item.setToolTip(0, f"{t['full_path']} (Click to select, Ctrl+click to multi-select)")
                self._tag_items[tid] = t_item

            for t in tag_rows:
                tid = int(t["id"])
                pid = t["parent_id"]
                t_item = self._tag_items[tid]
                if pid is not None and int(pid) in self._tag_items:
                    self._tag_items[int(pid)].addChild(t_item)
                else:
                    self.root_tags.addChild(t_item)
                if tid in self._include_tag_ids:
                    t_item.setSelected(True)
                    self._apply_tag_item_visual_state(t_item, "include")
                elif tid in self._exclude_tag_ids:
                    self._apply_tag_item_visual_state(t_item, "exclude")

            # 4. Populate People nodes (with real face avatar icon when available)
            people_rows = preloaded_people if preloaded_people is not None else repo.get_people_list()
            for p in people_rows:
                pid = int(p["id"])
                pname = str(p["name"])
                cnt = int(p["media_count"])
                blob = p.get("thumbnail_blob")
                p_item = QTreeWidgetItem([f"{pname} ({cnt})" if cnt > 0 else pname])
                p_item.setData(0, _NODE_TYPE_ROLE, "person")
                p_item.setData(0, _NODE_PAYLOAD_ROLE, pid)
                p_item.setData(0, _HAS_AVATAR_ROLE, bool(blob))
                if blob:
                    cache_key = (pid, len(blob))
                    av_icon = self._person_avatar_icon_cache.get(cache_key)
                    if av_icon is None:
                        av_icon = QIcon(make_person_avatar_pixmap(pname, blob, 18))
                        self._person_avatar_icon_cache[cache_key] = av_icon
                    p_item.setIcon(0, av_icon)
                else:
                    p_item.setIcon(0, person_icon)
                if pid in self._person_ids:
                    p_item.setSelected(True)
                self.root_people.addChild(p_item)
                self._person_items[pid] = p_item

            # 5. Populate Ratings nodes (5 stars .. 1 star, Unrated)
            rating_counts = repo.get_rating_counts()
            rating_labels = [
                (5, "★★★★★ (5 stars)"),
                (4, "★★★★☆ (4 stars)"),
                (3, "★★★☆☆ (3 stars)"),
                (2, "★★☆☆☆ (2 stars)"),
                (1, "★☆☆☆☆ (1 star)"),
                (0, "Unrated"),
            ]
            for r_val, r_label in rating_labels:
                cnt = rating_counts.get(r_val, 0)
                r_item = QTreeWidgetItem([f"{r_label} ({cnt})"])
                r_item.setData(0, _NODE_TYPE_ROLE, "rating")
                r_item.setData(0, _NODE_PAYLOAD_ROLE, r_val)
                r_item.setIcon(0, star_icon if r_val > 0 else empty_star_icon)
                r_item.setFlags(r_item.flags() | Qt.ItemIsUserCheckable)
                r_item.setCheckState(
                    0, Qt.Checked if r_val in self._ratings else Qt.Unchecked
                )
                self.root_ratings.addChild(r_item)
                self._rating_items[r_val] = r_item

            # 6. Populate Places hierarchy (Country -> State/Region -> City)
            places_icon = make_wlpg_icon("geotag", 16)
            if hasattr(repo, "get_places_hierarchy"):
                places_data = repo.get_places_hierarchy()
                countries: dict[str, dict[str, dict[str, int]]] = {}
                for p in places_data:
                    c = (p.get("country") or "").strip()
                    s = (p.get("state") or "").strip()
                    city = (p.get("city") or "").strip()
                    cnt = int(p.get("count", 0))

                    # Normalize UK subdivision codes (ENG -> England)
                    if c in ("United Kingdom", "GB", "UK"):
                        if s.upper() in ("ENG", "ENGLAND"):
                            s = "England"
                        elif s.upper() in ("WLS", "WALES"):
                            s = "Wales"
                        elif s.upper() in ("SCT", "SCOTLAND"):
                            s = "Scotland"
                        elif s.upper() in ("NIR", "NORTHERN IRELAND"):
                            s = "Northern Ireland"

                    # If state is numeric code (e.g. '01', '07', '39'), treat as empty so city attaches directly to country
                    if s.isdigit():
                        s = ""

                    countries.setdefault(c, {}).setdefault(s, {})[city] = (
                        countries.setdefault(c, {}).setdefault(s, {}).get(city, 0) + cnt
                    )

                for c_name in sorted(countries.keys()):
                    states = countries[c_name]
                    c_total = sum(cnt for s_cities in states.values() for cnt in s_cities.values())
                    c_item = QTreeWidgetItem([f"{c_name} ({c_total})"])
                    c_item.setData(0, _NODE_TYPE_ROLE, "place")
                    c_item.setData(0, _NODE_PAYLOAD_ROLE, (c_name, "", ""))
                    c_item.setIcon(0, places_icon)
                    self.root_places.addChild(c_item)

                    for s_name in sorted(states.keys()):
                        cities = states[s_name]
                        if s_name:
                            s_total = sum(cities.values())
                            s_item = QTreeWidgetItem([f"{s_name} ({s_total})"])
                            s_item.setData(0, _NODE_TYPE_ROLE, "place")
                            s_item.setData(0, _NODE_PAYLOAD_ROLE, (c_name, s_name, ""))
                            s_item.setIcon(0, places_icon)
                            c_item.addChild(s_item)
                            parent_for_city = s_item
                        else:
                            parent_for_city = c_item

                        for city_name, city_cnt in sorted(cities.items()):
                            if city_name:
                                ct_item = QTreeWidgetItem([f"{city_name} ({city_cnt})"])
                                ct_item.setData(0, _NODE_TYPE_ROLE, "place")
                                ct_item.setData(0, _NODE_PAYLOAD_ROLE, (c_name, s_name, city_name))
                                ct_item.setIcon(0, places_icon)
                                parent_for_city.addChild(ct_item)

            self.root_folders.setExpanded(True)
            self.root_tags.setExpanded(True)
            self.root_people.setExpanded(was_people_expanded)
            if getattr(self, "_dark_mode", False):
                self._sync_tree_item_colors()
            self._sync_tree_selection()
        finally:
            self.tree.blockSignals(False)
            self.tree.setUpdatesEnabled(True)
            self._updating_tree = False
        self._update_summary_label()

    def refresh_people_only(self, repo: CatalogRepository) -> list[dict[str, Any]]:
        """Refreshes only the People section without rebuilding Folders, Dates, Tags, or Ratings."""
        self._repo = repo
        people_rows = repo.get_people_list()
        if not hasattr(self, "root_people") or self.root_people is None:
            self.refresh_trees(repo, preloaded_people=people_rows)
            return people_rows
        self._updating_tree = True
        self.tree.blockSignals(True)
        self.tree.setUpdatesEnabled(False)
        try:
            was_expanded = self.root_people.isExpanded()
            self.root_people.takeChildren()
            self._person_items.clear()
            person_icon = make_wlpg_icon("person", 16)
            for p in people_rows:
                pid = int(p["id"])
                pname = str(p["name"])
                cnt = int(p["media_count"])
                blob = p.get("thumbnail_blob")
                p_item = QTreeWidgetItem([f"{pname} ({cnt})" if cnt > 0 else pname])
                p_item.setData(0, _NODE_TYPE_ROLE, "person")
                p_item.setData(0, _NODE_PAYLOAD_ROLE, pid)
                p_item.setData(0, _HAS_AVATAR_ROLE, bool(blob))
                if blob:
                    cache_key = (pid, len(blob))
                    av_icon = self._person_avatar_icon_cache.get(cache_key)
                    if av_icon is None:
                        av_icon = QIcon(make_person_avatar_pixmap(pname, blob, 18))
                        self._person_avatar_icon_cache[cache_key] = av_icon
                    p_item.setIcon(0, av_icon)
                else:
                    p_item.setIcon(0, person_icon)
                if pid in self._person_ids:
                    p_item.setSelected(True)
                self.root_people.addChild(p_item)
                self._person_items[pid] = p_item
            self.root_people.setExpanded(was_expanded)
        finally:
            self.tree.blockSignals(False)
            self.tree.setUpdatesEnabled(True)
            self._updating_tree = False
        return people_rows

    def _populate_albums_section(self, repo: CatalogRepository) -> None:
        """Populate My Albums and Smart Albums child items under ``root_albums``."""
        if not hasattr(self, "root_my_albums") or not hasattr(self, "root_smart_albums"):
            return
        while self.root_my_albums.childCount():
            self.root_my_albums.takeChild(0)
        while self.root_smart_albums.childCount():
            self.root_smart_albums.takeChild(0)
        self._album_items.clear()
        self._smart_album_items.clear()

        album_icon = make_wlpg_icon("pictures_library", 16)
        smart_icon = make_wlpg_icon("star_filled", 16)

        try:
            static_albums = repo.get_albums_with_counts()
        except Exception:
            static_albums = []
        for alb in static_albums:
            aid = int(alb["id"])
            name = str(alb["name"])
            cnt = int(alb.get("item_count", 0))
            item = QTreeWidgetItem([f"{name} ({cnt})"])
            item.setData(0, _NODE_TYPE_ROLE, "album")
            item.setData(0, _NODE_PAYLOAD_ROLE, (aid, name))
            item.setIcon(0, album_icon)
            if alb.get("description"):
                item.setToolTip(0, str(alb["description"]))
            self.root_my_albums.addChild(item)
            self._album_items[aid] = item

        try:
            smart_albums = repo.get_smart_albums_with_counts()
        except Exception:
            smart_albums = []
        for salb in smart_albums:
            sid = int(salb["id"])
            name = str(salb["name"])
            cnt = int(salb.get("item_count", 0))
            item = QTreeWidgetItem([f"{name} ({cnt})"])
            item.setData(0, _NODE_TYPE_ROLE, "smart_album")
            item.setData(0, _NODE_PAYLOAD_ROLE, (sid, name))
            item.setIcon(0, smart_icon)
            self.root_smart_albums.addChild(item)
            self._smart_album_items[sid] = item

    def refresh_albums_only(self, repo: CatalogRepository) -> None:
        """Refresh only the Static and Smart Albums counts without rebuilding the full tree."""
        self._repo = repo
        self._updating_tree = True
        self.tree.blockSignals(True)
        try:
            self._populate_albums_section(repo)
            if getattr(self, "_dark_mode", False):
                self._sync_tree_item_colors()
            self._sync_tree_selection()
        finally:
            self.tree.blockSignals(False)
            self._updating_tree = False
        self._update_summary_label()

    def active_album_id(self) -> int | None:
        """Return currently selected Static Album id, or None."""
        return self._album_id

    def _apply_tag_item_visual_state(
        self, item: QTreeWidgetItem, state: Literal["neutral", "include", "exclude"]
    ) -> None:
        base_text = item.text(0)
        if base_text.startswith("[NOT] "):
            base_text = base_text[6:]
        item.setData(0, _TAG_FILTER_STATE_ROLE, state)
        font = item.font(0)
        if state == "include":
            font.setStrikeOut(False)
            font.setBold(True)
            item.setFont(0, font)
            item.setForeground(0, QBrush(QColor("#005A9E")))
            item.setText(0, base_text)
            item.setSelected(True)
        elif state == "exclude":
            font.setStrikeOut(True)
            font.setBold(True)
            item.setFont(0, font)
            item.setForeground(0, QBrush(QColor("#C50F1F")))
            item.setText(0, f"[NOT] {base_text}")
        else:
            font.setStrikeOut(False)
            font.setBold(False)
            item.setFont(0, font)
            item.setForeground(0, QBrush(QColor("#1E395B")))
            item.setText(0, base_text)

    # -------------------------------------------------------------------------
    # Visual Selection Synchronization
    # -------------------------------------------------------------------------
    def _sync_tree_selection(self) -> None:
        """Synchronizes tree node visual selection (.setSelected) with internal filter state."""
        prev_block = self.tree.blockSignals(True)
        try:
            self.tree.clearSelection()

            # 0. Albums & Smart Albums
            has_album = False
            if self._album_id is not None and self._album_id in self._album_items:
                a_item = self._album_items[self._album_id]
                a_item.setSelected(True)
                self.tree.setCurrentItem(a_item)
                p = a_item.parent()
                while p:
                    p.setExpanded(True)
                    p = p.parent()
                self.tree.scrollToItem(a_item)
                has_album = True
            elif self._smart_album_id is not None and self._smart_album_id in self._smart_album_items:
                s_item = self._smart_album_items[self._smart_album_id]
                s_item.setSelected(True)
                self.tree.setCurrentItem(s_item)
                p = s_item.parent()
                while p:
                    p.setExpanded(True)
                    p = p.parent()
                self.tree.scrollToItem(s_item)
                has_album = True

            # 1. Folders
            has_folder = False
            if not has_album and self._folder_id is not None and self._folder_id in self._folder_items:
                f_item = self._folder_items[self._folder_id]
                self._folder_name = f_item.text(0)
                f_item.setSelected(True)
                self.tree.setCurrentItem(f_item)
                p = f_item.parent()
                while p:
                    p.setExpanded(True)
                    p = p.parent()
                self.tree.scrollToItem(f_item)
                has_folder = True

            # 2. Dates
            has_date = False
            if not has_folder and self._date_year is not None and hasattr(self, "root_dates") and self.root_dates is not None:
                target_payload = (self._date_year, self._date_month, self._date_day)

                def find_date_item(parent_item: QTreeWidgetItem) -> QTreeWidgetItem | None:
                    for i in range(parent_item.childCount()):
                        child = parent_item.child(i)
                        if child is not None:
                            if child.data(0, _NODE_PAYLOAD_ROLE) == target_payload:
                                return child
                            found = find_date_item(child)
                            if found is not None:
                                return found
                    return None

                matched = find_date_item(self.root_dates)
                if matched is not None:
                    matched.setSelected(True)
                    self.tree.setCurrentItem(matched)
                    p = matched.parent()
                    while p:
                        p.setExpanded(True)
                        p = p.parent()
                    self.tree.scrollToItem(matched)
                    has_date = True

            # 3. Tags
            for tid, t_item in self._tag_items.items():
                if tid in self._include_tag_ids:
                    t_item.setSelected(True)
                    self._apply_tag_item_visual_state(t_item, "include")
                elif tid in self._exclude_tag_ids:
                    self._apply_tag_item_visual_state(t_item, "exclude")
                else:
                    self._apply_tag_item_visual_state(t_item, "neutral")

            # 4. Untagged
            if hasattr(self, "item_untagged") and self.item_untagged is not None:
                self.item_untagged.setSelected(self._untagged_only)

            # 5. People
            for pid, p_item in self._person_items.items():
                p_item.setSelected(pid in self._person_ids)

            # 6. Ratings
            for r_val, r_item in self._rating_items.items():
                r_item.setCheckState(0, Qt.Checked if r_val in self._ratings else Qt.Unchecked)

            # 7. All photos and videos (highlighted whenever no specific facet filter is active)
            is_all_media = (
                not has_album
                and not has_folder
                and not has_date
                and not self._include_tag_ids
                and not self._exclude_tag_ids
                and not self._untagged_only
                and not self._person_ids
                and not self._ratings
            )
            if hasattr(self, "root_all") and self.root_all is not None:
                self.root_all.setSelected(is_all_media)
                if is_all_media:
                    self.tree.setCurrentItem(self.root_all)
        finally:
            self.tree.blockSignals(prev_block)

    # -------------------------------------------------------------------------
    # User Interaction Slots
    # -------------------------------------------------------------------------
    def _on_subfolders_toggled(self, checked: bool) -> None:
        self._include_subfolders = bool(checked)
        self._update_summary_label()
        self.filter_changed.emit()

    def _on_tag_boolean_changed(self, _index: int) -> None:
        if hasattr(self, "cmb_tag_boolean"):
            mode = self.cmb_tag_boolean.currentData()
            self._tag_boolean_mode = "OR" if mode == "OR" else "AND"
        self._update_summary_label()
        self.filter_changed.emit()

    def _on_tree_selection_changed(self) -> None:
        if self._updating_tree:
            return
        selected = self.tree.selectedItems()
        if not selected:
            self._sync_tree_selection()
            return

        tag_ids: set[int] = set()
        person_ids: set[int] = set()
        untagged_sel = False
        folder_tuple: tuple[int, str] | None = None
        date_tuple: tuple | None = None
        album_tuple: tuple[int, str] | None = None
        smart_album_tuple: tuple[int, str] | None = None
        all_media = False

        for it in selected:
            ntype = it.data(0, _NODE_TYPE_ROLE)
            payload = it.data(0, _NODE_PAYLOAD_ROLE)
            if ntype == "tag" and payload is not None:
                tag_ids.add(int(payload))
            elif ntype == "untagged":
                untagged_sel = True
            elif ntype == "person" and payload is not None:
                person_ids.add(int(payload))
            elif ntype == "folder" and payload is not None:
                folder_tuple = (int(payload), it.text(0))
            elif ntype == "date" and isinstance(payload, tuple):
                date_tuple = payload
            elif ntype == "album" and isinstance(payload, tuple):
                album_tuple = (int(payload[0]), str(payload[1]))
            elif ntype == "smart_album" and isinstance(payload, tuple):
                smart_album_tuple = (int(payload[0]), str(payload[1]))
            elif ntype == "all_media":
                all_media = True

        if all_media and (
            album_tuple is not None
            or smart_album_tuple is not None
            or folder_tuple is not None
            or date_tuple is not None
            or tag_ids
            or person_ids
            or untagged_sel
        ):
            all_media = False
            if hasattr(self, "root_all") and self.root_all is not None:
                self.root_all.setSelected(False)

        if all_media:
            self.clear_all_filters()
            return

        if album_tuple is not None:
            aid, aname = album_tuple
            self._select_album_scope(album_id=aid, album_name=aname)
            return
        if smart_album_tuple is not None:
            sid, sname = smart_album_tuple
            self._select_smart_album_scope(smart_album_id=sid, smart_album_name=sname)
            return

        changed = False
        if self._album_id is not None or self._smart_album_id is not None or self._smart_album_rules is not None:
            self._album_id = None
            self._album_name = ""
            self._smart_album_id = None
            self._smart_album_name = ""
            self._smart_album_rules = None
            changed = True
        ctrl_held = bool(QGuiApplication.keyboardModifiers() & Qt.ControlModifier)
        if folder_tuple is not None:
            if hasattr(self, "root_all") and self.root_all is not None and self.root_all.isSelected():
                self.root_all.setSelected(False)
            if self._folder_id != folder_tuple[0]:
                self._folder_id, self._folder_name = folder_tuple
                changed = True
            if not ctrl_held and (self._date_year is not None or self._date_month is not None or self._date_day is not None):
                self._date_year = None
                self._date_month = None
                self._date_day = None
                changed = True
        elif not ctrl_held and date_tuple is not None:
            if hasattr(self, "root_all") and self.root_all is not None and self.root_all.isSelected():
                self.root_all.setSelected(False)
            if self._folder_id is not None:
                self._folder_id = None
                self._folder_name = "All folders"
                changed = True

        if date_tuple is not None:
            if hasattr(self, "root_all") and self.root_all is not None and self.root_all.isSelected():
                self.root_all.setSelected(False)
            y, m, d = date_tuple
            if (self._date_year, self._date_month, self._date_day) != (y, m, d):
                self._date_year, self._date_month, self._date_day = y, m, d
                changed = True

        if self._include_tag_ids != tag_ids:
            self._include_tag_ids = tag_ids
            changed = True
        if self._untagged_only != untagged_sel:
            self._untagged_only = untagged_sel
            changed = True
        if self._person_ids != person_ids:
            self._person_ids = person_ids
            changed = True

        if tag_ids or person_ids or untagged_sel:
            if hasattr(self, "root_all") and self.root_all is not None and self.root_all.isSelected():
                self.root_all.setSelected(False)

        if changed:
            self._update_summary_label()
            self.filter_changed.emit()

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        if self._updating_tree:
            return
        if self._checkbox_just_changed:
            self._checkbox_just_changed = False
            return

        ntype = item.data(0, _NODE_TYPE_ROLE)
        payload = item.data(0, _NODE_PAYLOAD_ROLE)
        ctrl_held = bool(QGuiApplication.keyboardModifiers() & Qt.ControlModifier)

        if ntype == "all_media":
            self.clear_all_filters()
            return
        if ntype == "section_albums":
            self.albums_view_requested.emit()
            return
        if ntype in ("section_my_albums", "section_smart_albums"):
            item.setExpanded(not item.isExpanded())
            return
        if ntype == "album" and isinstance(payload, tuple):
            self._select_album_scope(album_id=int(payload[0]), album_name=str(payload[1]))
            return
        if ntype == "smart_album" and isinstance(payload, tuple):
            self._select_smart_album_scope(smart_album_id=int(payload[0]), smart_album_name=str(payload[1]))
            return
        if ntype == "section_folders":
            self._folder_id = None
            self._folder_name = "All folders"
            item.setExpanded(True)
            self._sync_tree_selection()
            self._update_summary_label()
            self.filter_changed.emit()
            return
        if ntype == "section_dates":
            self._date_year = None
            self._date_month = None
            self._date_day = None
            self._sync_tree_selection()
            self._update_summary_label()
            self.filter_changed.emit()
            return
        if ntype == "section_tags":
            self._updating_tree = True
            try:
                self._include_tag_ids.clear()
                self._exclude_tag_ids.clear()
                self._untagged_only = False
                for t_item in self._tag_items.values():
                    t_item.setSelected(False)
                    self._apply_tag_item_visual_state(t_item, "neutral")
            finally:
                self._updating_tree = False
            self._update_summary_label()
            self.filter_changed.emit()
            return
        if ntype == "section_people":
            self.select_person(None)
            self.people_directory_requested.emit()
            return
        if ntype == "section_ratings":
            self.select_ratings([])
            return
        if ntype == "section_places":
            item.setExpanded(not item.isExpanded())
            return
        if ntype == "place":
            country, state, city = payload if isinstance(payload, tuple) else (str(payload), "", "")
            self.place_selected.emit(country, state, city)
            return
        if ntype == "folder":
            self._folder_id = int(payload)
            self._folder_name = item.text(0)
            if hasattr(self, "root_all") and self.root_all is not None:
                self.root_all.setSelected(False)
            if item.childCount() > 0:
                item.setExpanded(True)
            self._update_summary_label()
            self.filter_changed.emit()
            return
        if ntype == "date" and isinstance(payload, tuple):
            self._date_year, self._date_month, self._date_day = payload
            if hasattr(self, "root_all") and self.root_all is not None:
                self.root_all.setSelected(False)
            self._update_summary_label()
            self.filter_changed.emit()
            return
        if ntype in ("tag", "untagged", "person"):
            # Handled reactively by itemSelectionChanged
            return
        if ntype == "rating":
            r_val = int(payload)
            if ctrl_held:
                new_set = set(self._ratings)
                if r_val in new_set:
                    new_set.discard(r_val)
                else:
                    new_set.add(r_val)
                self.select_ratings(sorted(new_set))
            else:
                self.select_ratings([r_val])
            return

    def _on_item_changed(self, item: QTreeWidgetItem, _column: int) -> None:
        if self._updating_tree:
            return
        ntype = item.data(0, _NODE_TYPE_ROLE)
        payload = item.data(0, _NODE_PAYLOAD_ROLE)

        # Handle inline folder rename completion
        if ntype == "folder" and getattr(self, "_editing_folder_item", None) is item:
            new_name = item.text(0).strip()
            old_name = self._editing_folder_old_name
            if new_name == old_name:
                return  # Initial setFlags trigger; editing still active

            # User committed a new name: finish editing
            old_path = self._editing_folder_old_path
            self._editing_folder_item = None
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)

            if not new_name:
                self._updating_tree = True
                try:
                    item.setText(0, old_name)
                finally:
                    self._updating_tree = False
                return

            invalid_chars = set(r'\/:*?"<>|')
            if any(c in invalid_chars for c in new_name):
                self._updating_tree = True
                try:
                    item.setText(0, old_name)
                finally:
                    self._updating_tree = False
                QToolTip.showText(
                    QCursor.pos(),
                    "A folder name cannot contain any of the following characters:\n\\ / : * ? \" < > |",
                    self.tree,
                )
                return

            old_p = Path(old_path)
            new_p = old_p.parent / new_name

            if new_p.exists():
                self._updating_tree = True
                try:
                    item.setText(0, old_name)
                finally:
                    self._updating_tree = False
                QToolTip.showText(
                    QCursor.pos(),
                    f"A folder named '{new_name}' already exists in this location.",
                    self.tree,
                )
                return

            try:
                os.rename(str(old_p), str(new_p))
            except Exception as exc:
                self._updating_tree = True
                try:
                    item.setText(0, old_name)
                finally:
                    self._updating_tree = False
                QToolTip.showText(
                    QCursor.pos(),
                    f"Could not rename folder: {exc}",
                    self.tree,
                )
                return

            self._updating_tree = True
            try:
                item.setText(0, new_name)
                item.setToolTip(0, str(new_p))
            finally:
                self._updating_tree = False

            if old_path in self._favourite_paths:
                idx = self._favourite_paths.index(old_path)
                self._favourite_paths[idx] = str(new_p)
                self._rebuild_favourites_menu()

            self.folder_renamed.emit(str(old_p), str(new_p), new_name)
            return

        self._checkbox_just_changed = True
        checked = item.checkState(0) == Qt.Checked

        self._updating_tree = True
        try:
            if ntype == "tag":
                tid = int(payload)
                if checked:
                    self._include_tag_ids.add(tid)
                    self._exclude_tag_ids.discard(tid)
                    self._apply_tag_item_visual_state(item, "include")
                else:
                    self._include_tag_ids.discard(tid)
                    self._exclude_tag_ids.discard(tid)
                    self._apply_tag_item_visual_state(item, "neutral")
            elif ntype == "untagged":
                self._untagged_only = checked
            elif ntype == "person":
                pid = int(payload)
                if checked:
                    self._person_ids.add(pid)
                else:
                    self._person_ids.discard(pid)
            elif ntype == "rating":
                r_val = int(payload)
                if checked:
                    self._ratings.add(r_val)
                else:
                    self._ratings.discard(r_val)
        finally:
            self._updating_tree = False

        self._update_summary_label()
        self.filter_changed.emit()

    def _build_context_menu(self, item: QTreeWidgetItem) -> QMenu:
        """Constructs the QMenu corresponding to the given navigation item."""
        ntype = item.data(0, _NODE_TYPE_ROLE)
        payload = item.data(0, _NODE_PAYLOAD_ROLE)

        menu = QMenu(self)
        menu.setStyleSheet(
            """
            QMenu {
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 4px;
                color: #0F172A;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
            }
            QMenu::item {
                padding: 6px 20px 6px 12px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #F1F5F9;
                color: #0F172A;
            }
            QMenu::separator {
                height: 1px;
                background-color: #E2E8F0;
                margin: 4px 6px;
            }
            """
        )
        if ntype == "section_tags":
            act_clean = QAction("Delete unused tags", menu)
            act_clean.triggered.connect(self._on_delete_unused_tags)
            menu.addAction(act_clean)
        elif ntype == "tag":
            tid = int(payload)
            act_inc = QAction("Include tag", menu)
            act_exc = QAction("Exclude tag (NOT)", menu)
            act_clr = QAction("Clear tag filter", menu)
            act_inc.triggered.connect(lambda: self.set_tag_filter_state(tid, "include"))
            act_exc.triggered.connect(lambda: self.set_tag_filter_state(tid, "exclude"))
            act_clr.triggered.connect(lambda: self.set_tag_filter_state(tid, "neutral"))
            menu.addAction(act_inc)
            menu.addAction(act_exc)
            menu.addAction(act_clr)
            menu.addSeparator()

            act_del_this = QAction("Delete tag", menu)
            act_del_this.triggered.connect(lambda: self._on_delete_tag(tid, item.text(0)))
            menu.addAction(act_del_this)

            act_clean = QAction("Delete unused tags", menu)
            act_clean.triggered.connect(self._on_delete_unused_tags)
            menu.addAction(act_clean)
        elif ntype == "folder":
            folder_path = item.toolTip(0)
            folder_name = item.text(0)
            fid = int(payload) if payload is not None else 0

            # 1. Expand / Collapse (matches WLPG folder context menu)
            is_expanded = item.isExpanded()
            act_expand = QAction("Collapse" if is_expanded else "Expand", menu)
            act_expand.setEnabled(item.childCount() > 0)
            act_expand.triggered.connect(lambda: item.setExpanded(not is_expanded))
            menu.addAction(act_expand)

            if folder_path:
                act_open_loc = QAction("Open folder location", menu)
                act_open_loc.triggered.connect(lambda _c=False, fp=folder_path: self._open_folder_location(fp))
                menu.addAction(act_open_loc)

            menu.addSeparator()

            # 2. Cut & Copy
            act_cut = QAction("Cut", menu)
            act_cut.triggered.connect(lambda: self._clipboard_copy_folder(folder_path, is_cut=True))
            menu.addAction(act_cut)

            act_copy = QAction("Copy", menu)
            act_copy.triggered.connect(lambda: self._clipboard_copy_folder(folder_path, is_cut=False))
            menu.addAction(act_copy)

            menu.addSeparator()

            # 3. Delete & Rename
            act_delete = QAction("Delete", menu)
            act_delete.triggered.connect(lambda: self._delete_folder_action(item, fid, folder_path, folder_name))
            menu.addAction(act_delete)

            act_rename = QAction("Rename", menu)
            act_rename.triggered.connect(lambda: self._start_inline_rename(item))
            menu.addAction(act_rename)

            menu.addSeparator()

            # 4. Create new folder
            act_new_folder = QAction("Create new folder", menu)
            act_new_folder.triggered.connect(lambda: self._create_new_folder_action(item, folder_path))
            menu.addAction(act_new_folder)

            menu.addSeparator()

            # Favourites & Remove from Library
            if folder_path:
                act_fav = QAction("Add to Favourites", menu)
                act_fav.triggered.connect(lambda: self.add_favourite(folder_path))
                menu.addAction(act_fav)

            act_rem = QAction("Remove from library", menu)
            act_rem.triggered.connect(
                lambda _c=False, fid_val=fid, fp=folder_path, fn=folder_name: self.folder_remove_requested.emit(
                    fid_val, fp, fn
                )
            )
            menu.addAction(act_rem)
        elif ntype in ("section_folders", "all_media"):
            act_add = QAction(make_wlpg_icon("folder_add", 16), "Add folder to gallery...", menu)
            act_add.triggered.connect(self.folder_add_requested.emit)
            menu.addAction(act_add)
            menu.addSeparator()
            act_save_smart = QAction(
                make_wlpg_icon("star_filled", 16),
                "Save Current Filter as Smart Album...",
                menu,
            )
            act_save_smart.triggered.connect(self.save_filter_as_smart_album_requested.emit)
            menu.addAction(act_save_smart)
        elif ntype in ("section_albums", "section_my_albums"):
            act_new_alb = QAction(make_wlpg_icon("pictures_library", 16), "New Album...", menu)
            act_new_alb.triggered.connect(self.new_album_requested.emit)
            menu.addAction(act_new_alb)
            if ntype == "section_albums":
                act_new_smart = QAction(
                    make_wlpg_icon("star_filled", 16), "New Smart Album...", menu
                )
                act_new_smart.triggered.connect(self.new_smart_album_requested.emit)
                menu.addAction(act_new_smart)
        elif ntype == "section_smart_albums":
            act_new_smart = QAction(
                make_wlpg_icon("star_filled", 16), "New Smart Album...", menu
            )
            act_new_smart.triggered.connect(self.new_smart_album_requested.emit)
            menu.addAction(act_new_smart)
            act_save_smart = QAction(
                "Save Current Filter as Smart Album...",
                menu,
            )
            act_save_smart.triggered.connect(self.save_filter_as_smart_album_requested.emit)
            menu.addAction(act_save_smart)
        elif ntype == "album" and isinstance(payload, tuple):
            aid, aname = int(payload[0]), str(payload[1])
            act_ren = QAction("Rename Album...", menu)
            act_ren.triggered.connect(lambda _c=False, i=aid, n=aname: self.rename_album_requested.emit(i, n))
            menu.addAction(act_ren)
            act_del = QAction("Delete Album", menu)
            act_del.triggered.connect(lambda _c=False, i=aid, n=aname: self.delete_album_requested.emit(i, n))
            menu.addAction(act_del)
        elif ntype == "smart_album" and isinstance(payload, tuple):
            sid, sname = int(payload[0]), str(payload[1])
            act_edit = QAction("Edit Rules...", menu)
            act_edit.triggered.connect(lambda _c=False, i=sid: self.edit_smart_album_requested.emit(i))
            menu.addAction(act_edit)
            act_ren = QAction("Rename Smart Album...", menu)
            act_ren.triggered.connect(lambda _c=False, i=sid, n=sname: self.rename_smart_album_requested.emit(i, n))
            menu.addAction(act_ren)
            menu.addSeparator()
            act_del = QAction("Delete Smart Album", menu)
            act_del.triggered.connect(lambda _c=False, i=sid, n=sname: self.delete_smart_album_requested.emit(i, n))
            menu.addAction(act_del)

        if ntype in self._section_order:
            if not menu.isEmpty():
                menu.addSeparator()
            idx = self._section_order.index(ntype)
            act_up = QAction("Move Section Up", menu)
            act_up.setEnabled(idx > 0)
            act_up.triggered.connect(lambda _c=False, st=ntype: self.move_section(st, -1))
            menu.addAction(act_up)

            act_down = QAction("Move Section Down", menu)
            act_down.setEnabled(idx < len(self._section_order) - 1)
            act_down.triggered.connect(lambda _c=False, st=ntype: self.move_section(st, 1))
            menu.addAction(act_down)

            if self._section_order != list(self._default_section_order):
                act_reset = QAction("Reset Default Section Order", menu)
                act_reset.triggered.connect(self.reset_section_order)
                menu.addAction(act_reset)

        return menu

    def _on_context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        menu = self._build_context_menu(item)
        if not menu.isEmpty():
            menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _on_delete_unused_tags(self) -> None:
        if not hasattr(self, "_repo") or self._repo is None:
            return
        deleted = self._repo.delete_unused_tags()
        self.refresh_trees(self._repo)
        self.filter_changed.emit()
        QMessageBox.information(
            self,
            "Clean Up Tags",
            f"Removed {deleted} unused tag{'s' if deleted != 1 else ''}.\n\n"
            "This only removed tags with zero associated photos. If a photo with these tags is imported later, "
            "the tags will be automatically recreated.",
        )

    def _on_delete_tag(self, tag_id: int, tag_label: str) -> None:
        if not hasattr(self, "_repo") or self._repo is None:
            return
        clean_name = tag_label.split(" (")[0] if " (" in tag_label else tag_label
        reply = QMessageBox.question(
            self,
            "Delete Tag",
            f"Are you sure you want to delete the tag '{clean_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._repo.delete_tag(tag_id)
            self.refresh_trees(self._repo)
            self.filter_changed.emit()

    def _on_tag_dropped(self, source_tag_id: int, target_tag_id: int | None) -> None:
        """Handles moving or merging a dragged descriptive tag onto another tag or Descriptive tags root."""
        if not hasattr(self, "_repo") or self._repo is None:
            return
        if source_tag_id == target_tag_id:
            return

        conn = self._repo.engine.catalog_conn()
        source_row = conn.execute(
            "SELECT id, parent_id, name, full_path FROM tags WHERE id = ?;", (source_tag_id,)
        ).fetchone()
        if source_row is None:
            return

        current_parent_id = source_row["parent_id"]
        # If dropping onto the current parent, nothing needs to be done
        if (target_tag_id is None and current_parent_id is None) or (target_tag_id == current_parent_id):
            return

        source_name = str(source_row["name"])

        # Check if a sibling tag with the same name already exists under the target parent
        existing = self._repo.find_child_tag_by_name(target_tag_id, source_name)
        if existing is not None and int(existing["id"]) != source_tag_id:
            existing_tag_id = int(existing["id"])
            settings = QSettings("PhotoGallery64", "PhotoGallery64")
            suppress = settings.value("suppress_tag_merge_warning", False, type=bool)

            if not suppress:
                dlg = TagMergeConfirmDialog(self)
                if getattr(self, "_auto_tag_merge_decision", None) is not None:
                    res = QDialog.Accepted if self._auto_tag_merge_decision else QDialog.Rejected
                else:
                    res = dlg.exec()

                if res != QDialog.Accepted:
                    return

                if dlg.dont_show_again:
                    settings.setValue("suppress_tag_merge_warning", True)

            # Perform tag merge
            self._repo.merge_tags(source_tag_id, existing_tag_id)
        else:
            # Perform tag move
            self._repo.move_tag(source_tag_id, target_tag_id)

        # Refresh navigation tree and notify application
        self.refresh_trees(self._repo)
        self.tag_hierarchy_changed.emit()

        # Keep target tag (and ancestors) expanded so the moved/merged tag is visible
        if target_tag_id is not None and target_tag_id in self._tag_items:
            t_item = self._tag_items[target_tag_id]
            t_item.setExpanded(True)
            p = t_item.parent()
            while p is not None:
                p.setExpanded(True)
                p = p.parent()
        elif target_tag_id is None and hasattr(self, "root_tags"):
            self.root_tags.setExpanded(True)

    def _open_folder_location(self, folder_path: str) -> None:
        """Reveals the folder in Windows File Explorer."""
        if not folder_path:
            return
        norm = os.path.normpath(folder_path)
        if os.path.exists(norm):
            try:
                import subprocess
                subprocess.Popen(f'explorer.exe "{norm}"')
            except Exception as e:
                logger.warning("Could not open folder location in File Explorer: %s", e)

    def _start_inline_rename(self, item: QTreeWidgetItem) -> None:
        """Starts in-place inline renaming on the tree item, exactly like Windows File Explorer."""
        self._editing_folder_item = item
        self._editing_folder_old_name = item.text(0)
        self._editing_folder_old_path = item.toolTip(0)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.tree.setCurrentItem(item)
        self.tree.editItem(item, 0)

    def _create_new_folder_action(self, parent_item: QTreeWidgetItem, parent_path_str: str) -> None:
        """Creates a new folder on disk and enters inline rename mode in the tree."""
        parent_p = Path(parent_path_str)
        if not parent_p.is_dir():
            return

        candidate = "New folder"
        i = 2
        while (parent_p / candidate).exists():
            candidate = f"New folder ({i})"
            i += 1

        new_dir = parent_p / candidate
        try:
            new_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QToolTip.showText(QCursor.pos(), f"Could not create folder: {exc}", self.tree)
            return

        child = QTreeWidgetItem(parent_item, [candidate])
        child.setData(0, _NODE_TYPE_ROLE, "folder")
        child.setData(0, _NODE_PAYLOAD_ROLE, None)
        child.setToolTip(0, str(new_dir))
        child.setIcon(0, make_wlpg_icon("folder", 16))
        parent_item.setExpanded(True)
        self.tree.setCurrentItem(child)

        self._editing_folder_item = child
        self._editing_folder_old_name = candidate
        self._editing_folder_old_path = str(new_dir)
        child.setFlags(child.flags() | Qt.ItemIsEditable)
        self.tree.editItem(child, 0)

        self.folder_created.emit(str(new_dir))

    def _delete_folder_action(
        self,
        item: QTreeWidgetItem,
        folder_id: int,
        folder_path: str,
        folder_name: str,
    ) -> None:
        """Confirms and moves the folder to the Windows Recycle Bin."""
        reply = QMessageBox.question(
            self,
            "Delete Folder",
            f"Are you sure you want to move '{folder_name}' and all its contents to the Recycle Bin?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        p = Path(folder_path)
        if p.exists():
            try:
                import send2trash
                send2trash.send2trash(str(p))
            except Exception:
                pass

        parent_item = item.parent()
        if parent_item:
            parent_item.removeChild(item)
        else:
            idx = self.tree.indexOfTopLevelItem(item)
            if idx >= 0:
                self.tree.takeTopLevelItem(idx)

        self.folder_deleted.emit(folder_id, folder_path)

    def _clipboard_copy_folder(self, folder_path: str, is_cut: bool = False) -> None:
        """Copies or cuts a folder to the clipboard with Windows Explorer Preferred DropEffect."""
        if not folder_path:
            return
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(folder_path)])
        effect = QByteArray(b"\x02\x00\x00\x00" if is_cut else b"\x05\x00\x00\x00")
        mime.setData("Preferred DropEffect", effect)
        QGuiApplication.clipboard().setMimeData(mime)

    def add_favourite(self, folder_path: str) -> None:
        if folder_path and folder_path not in self._favourite_paths:
            self._favourite_paths.append(folder_path)
            self._rebuild_favourites_menu()

    def _remove_folder_from_favourites(self, folder_path: str) -> None:
        if folder_path in self._favourite_paths:
            self._favourite_paths.remove(folder_path)
            self._rebuild_favourites_menu()

    def _rebuild_favourites_menu(self) -> None:
        if hasattr(self, "favourites_menu"):
            self.favourites_menu.clear()
            for p in self._favourite_paths:
                act = QAction(p, self.favourites_menu)
                act.setData(p)
                act.triggered.connect(
                    lambda _checked=False, path=p: self.favourite_path_selected.emit(path)
                )
                self.favourites_menu.addAction(act)

    # -------------------------------------------------------------------------
    # Programmatic Filter Control & Inspection API
    # -------------------------------------------------------------------------
    def select_folder(
        self, folder_id: int | None, include_subfolders: bool | None = None
    ) -> None:
        self._folder_id = folder_id
        if folder_id is not None and folder_id in self._folder_items:
            self._folder_name = self._folder_items[folder_id].text(0)
        else:
            self._folder_name = "All folders"
        if include_subfolders is not None:
            self._include_subfolders = bool(include_subfolders)
            self.chk_subfolders.blockSignals(True)
            self.chk_subfolders.setChecked(self._include_subfolders)
            self.chk_subfolders.blockSignals(False)
        self._sync_tree_selection()
        self._update_summary_label()
        self.filter_changed.emit()

    def clear_folder_filter(self) -> None:
        """Removes the folder filter, searching all folders while keeping all other filters intact."""
        self._folder_id = None
        self._folder_name = "All folders"
        self._sync_tree_selection()
        self._update_summary_label()
        self.filter_changed.emit()

    def selected_folder_name(self) -> str | None:
        """Returns the name of the currently filtered folder, or None if searching all folders."""
        if self._folder_id is not None:
            return self._folder_name
        return None

    def select_date(
        self,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> None:
        self._date_year = year
        self._date_month = month
        self._date_day = day
        self._sync_tree_selection()
        self._update_summary_label()
        self.filter_changed.emit()

    def set_tag_filter_state(
        self, tag_id: int, state: Literal["neutral", "include", "exclude"]
    ) -> None:
        self._updating_tree = True
        try:
            self._include_tag_ids.discard(tag_id)
            self._exclude_tag_ids.discard(tag_id)
            if state == "include":
                self._include_tag_ids.add(tag_id)
            elif state == "exclude":
                self._exclude_tag_ids.add(tag_id)

            item = self._tag_items.get(tag_id)
            if item is not None:
                self._apply_tag_item_visual_state(item, state)
        finally:
            self._updating_tree = False
        self._update_summary_label()
        self.filter_changed.emit()

    def clear_tag_filters(self) -> None:
        self._updating_tree = True
        try:
            self._include_tag_ids.clear()
            self._exclude_tag_ids.clear()
            self._untagged_only = False
            if hasattr(self, "item_untagged"):
                self.item_untagged.setSelected(False)
            for t_item in self._tag_items.values():
                self._apply_tag_item_visual_state(t_item, "neutral")
        finally:
            self._updating_tree = False
        self._update_summary_label()
        self.filter_changed.emit()

    def set_tag_boolean_mode(self, mode: Literal["AND", "OR"]) -> None:
        self._tag_boolean_mode = mode
        if hasattr(self, "cmb_tag_boolean"):
            idx = 1 if mode == "OR" else 0
            self.cmb_tag_boolean.blockSignals(True)
            self.cmb_tag_boolean.setCurrentIndex(idx)
            self.cmb_tag_boolean.blockSignals(False)
        self._update_summary_label()
        self.filter_changed.emit()

    def set_person_boolean_mode(self, mode: Literal["AND", "OR"]) -> None:
        self._person_boolean_mode = mode
        self._update_summary_label()
        self.filter_changed.emit()

    def set_untagged_only(self, untagged: bool) -> None:
        self._untagged_only = bool(untagged)
        if hasattr(self, "item_untagged"):
            self._updating_tree = True
            self.item_untagged.setSelected(self._untagged_only)
            self._updating_tree = False
        self._update_summary_label()
        self.filter_changed.emit()

    def select_person(self, person_id: int | None, toggle: bool = False) -> None:
        self._updating_tree = True
        try:
            if person_id is None:
                self._person_ids.clear()
            elif toggle:
                if person_id in self._person_ids:
                    self._person_ids.discard(person_id)
                else:
                    self._person_ids.add(person_id)
            else:
                self._person_ids = {person_id}

            for pid, p_item in self._person_items.items():
                p_item.setSelected(pid in self._person_ids)
        finally:
            self._updating_tree = False
        self._update_summary_label()
        self.filter_changed.emit()

    def unselect_person(self, person_id: int) -> None:
        """Removes a single person from active people filters."""
        if person_id in self._person_ids:
            self.select_person(person_id, toggle=True)

    def select_ratings(self, ratings: Sequence[int]) -> None:
        self._updating_tree = True
        try:
            self._ratings = {int(r) for r in ratings}
            for r_val, r_item in self._rating_items.items():
                r_item.setCheckState(
                    0, Qt.Checked if r_val in self._ratings else Qt.Unchecked
                )
        finally:
            self._updating_tree = False
        self._update_summary_label()
        self.filter_changed.emit()

    def clear_rating_filters(self) -> None:
        """Clears all active star rating filters."""
        self.select_ratings(())

    def set_flagged_only(self, flagged_only: bool) -> None:
        self._flagged_only = bool(flagged_only)
        self._flag_filter = 1 if flagged_only else None
        self._update_summary_label()
        self.filter_changed.emit()

    def set_flag_filter(self, flag_filter: int | None) -> None:
        self._flag_filter = flag_filter
        self._flagged_only = (flag_filter == 1)
        self._update_summary_label()
        self.filter_changed.emit()

    def set_edited_only(self, edited_only: bool) -> None:
        self._edited_only = bool(edited_only)
        self._edited_filter = 1 if edited_only else None
        self._update_summary_label()
        self.filter_changed.emit()

    def set_edited_filter(self, edited_filter: int | None) -> None:
        self._edited_filter = edited_filter
        self._edited_only = (edited_filter == 1)
        self._update_summary_label()
        self.filter_changed.emit()

    def set_media_type(self, media_type: Literal["all", "image", "video", "raw"]) -> None:
        self._media_type = media_type
        self._update_summary_label()
        self.filter_changed.emit()

    def set_orientation_filter(self, orientation: Literal["all", "landscape", "portrait"]) -> None:
        self._orientation_filter = orientation
        self._update_summary_label()
        self.filter_changed.emit()

    def _reset_non_album_facets(self) -> None:
        self._folder_id = None
        self._folder_name = "All folders"
        self._date_year = None
        self._date_month = None
        self._date_day = None
        self._include_tag_ids.clear()
        self._exclude_tag_ids.clear()
        self._untagged_only = False
        self._person_ids.clear()
        self._person_boolean_mode = "OR"
        self._ratings.clear()
        self._flagged_only = False
        self._edited_only = False
        self._edited_filter = None
        self._media_type = "all"
        self._orientation_filter = "all"

    def _select_album_scope(self, album_id: int, album_name: str = "") -> None:
        """Select a Static Album as the active gallery scope."""
        if (
            self._album_id == album_id
            and self._smart_album_id is None
            and not self._folder_id
            and not self._include_tag_ids
            and not self._person_ids
        ):
            return
        self._updating_tree = True
        try:
            self._reset_non_album_facets()
            self._smart_album_id = None
            self._smart_album_name = ""
            self._smart_album_rules = None
            self._album_id = int(album_id)
            self._album_name = album_name
            self._sync_tree_selection()
        finally:
            self._updating_tree = False
        self._update_summary_label()
        self.filter_changed.emit()

    def _select_smart_album_scope(self, smart_album_id: int, smart_album_name: str = "") -> None:
        """Select a Smart Album as the active gallery scope."""
        if (
            self._smart_album_id == smart_album_id
            and self._album_id is None
            and not self._folder_id
            and not self._include_tag_ids
            and not self._person_ids
        ):
            return
        self._updating_tree = True
        try:
            self._reset_non_album_facets()
            self._album_id = None
            self._album_name = ""
            self._smart_album_id = int(smart_album_id)
            self._smart_album_name = smart_album_name
            self._smart_album_rules = None
            self._sync_tree_selection()
        finally:
            self._updating_tree = False
        self._update_summary_label()
        self.filter_changed.emit()

    def clear_all_filters(self) -> None:
        self._updating_tree = True
        try:
            self._album_id = None
            self._album_name = ""
            self._smart_album_id = None
            self._smart_album_name = ""
            self._smart_album_rules = None
            self._reset_non_album_facets()

            self._sync_tree_selection()
        finally:
            self._updating_tree = False
        self._update_summary_label()
        self.filter_changed.emit()

    def selected_scope_description(self) -> str:
        parts: list[str] = []
        if self._album_id is not None:
            parts.append(f"Album: {self._album_name or self._album_id}")
        if self._smart_album_id is not None:
            parts.append(f"Smart Album: {self._smart_album_name or self._smart_album_id}")
        if self._folder_id is not None:
            parts.append(f"Folder: {self._folder_name}")
        if self._date_year is not None:
            if self._date_day is not None and self._date_month is not None:
                parts.append(f"Date: {self._date_day}/{self._date_month:02d}/{self._date_year}")
            elif self._date_month is not None:
                parts.append(f"Date: {self._date_month:02d}/{self._date_year}")
            else:
                parts.append(f"Year: {self._date_year}")
        if self._include_tag_ids:
            parts.append(f"Tags ({self._tag_boolean_mode}): {len(self._include_tag_ids)}")
        if self._exclude_tag_ids:
            parts.append(f"Excluded Tags: {len(self._exclude_tag_ids)}")
        if self._untagged_only:
            parts.append("Not tagged")
        if self._person_ids:
            parts.append(f"People: {len(self._person_ids)}")
        if self._ratings:
            parts.append(f"Ratings: {sorted(self._ratings, reverse=True)}")
        if self._flagged_only:
            parts.append("Flagged")
        if getattr(self, "_edited_filter", None) == 1 or self._edited_only:
            parts.append("Edited")
        elif getattr(self, "_edited_filter", None) == 0:
            parts.append("Not edited")
        if self._media_type == "image":
            parts.append("Photos only")
        elif self._media_type == "video":
            parts.append("Videos only")
        elif self._media_type == "raw":
            parts.append("RAW photos only")
        if getattr(self, "_orientation_filter", "all") == "landscape":
            parts.append("Landscape only")
        elif getattr(self, "_orientation_filter", "all") == "portrait":
            parts.append("Portrait only")
        return " | ".join(parts) if parts else "All photos and videos"

    def _update_summary_label(self) -> None:
        if hasattr(self, "lbl_active_summary"):
            self.lbl_active_summary.setText(self.selected_scope_description())

    def current_filter_state(self) -> dict[str, Any]:
        return {
            "album_id": self._album_id,
            "album_name": self._album_name,
            "smart_album_id": self._smart_album_id,
            "smart_album_name": self._smart_album_name,
            "smart_album_rules": self._smart_album_rules,
            "folder_id": self._folder_id,
            "include_subfolders": self._include_subfolders,
            "date_year": self._date_year,
            "date_month": self._date_month,
            "date_day": self._date_day,
            "include_tag_ids": tuple(sorted(self._include_tag_ids)),
            "tag_boolean_mode": self._tag_boolean_mode,
            "exclude_tag_ids": tuple(sorted(self._exclude_tag_ids)),
            "untagged_only": self._untagged_only,
            "person_ids": tuple(sorted(self._person_ids)),
            "person_boolean_mode": self._person_boolean_mode,
            "ratings": tuple(sorted(self._ratings)),
            "flagged_only": self._flagged_only,
            "edited_only": self._edited_only,
            "edited_filter": getattr(self, "_edited_filter", (1 if self._edited_only else None)),
            "media_type": self._media_type,
            "orientation_filter": getattr(self, "_orientation_filter", "all"),
        }

    def restore_filter_state(self, state: dict[str, Any]) -> None:
        """Restores filter state from a dictionary (used for back/forward navigation history and startup restore)."""
        self._album_id = state.get("album_id")
        self._album_name = str(state.get("album_name") or "")
        self._smart_album_id = state.get("smart_album_id")
        self._smart_album_name = str(state.get("smart_album_name") or "")
        self._smart_album_rules = state.get("smart_album_rules")
        self._folder_id = state.get("folder_id")
        self._include_subfolders = state.get("include_subfolders", True)
        self._date_year = state.get("date_year")
        self._date_month = state.get("date_month")
        self._date_day = state.get("date_day")
        self._include_tag_ids = set(state.get("include_tag_ids", ()))
        self._tag_boolean_mode = state.get("tag_boolean_mode", "AND")
        self._exclude_tag_ids = set(state.get("exclude_tag_ids", ()))
        self._untagged_only = state.get("untagged_only", False)
        self._person_ids = set(state.get("person_ids", ()))
        self._person_boolean_mode = state.get("person_boolean_mode", "OR")
        self._ratings = set(state.get("ratings", ()))
        self._flagged_only = state.get("flagged_only", False)
        self._edited_filter = state.get("edited_filter", (1 if state.get("edited_only", False) else None))
        self._edited_only = (self._edited_filter == 1)
        self._media_type = state.get("media_type", "all")
        self._orientation_filter = state.get("orientation_filter", "all")

        self._updating_tree = True
        try:
            self._sync_tree_selection()

            # Update subfolders checkbox
            if hasattr(self, "chk_subfolders"):
                self.chk_subfolders.blockSignals(True)
                self.chk_subfolders.setChecked(self._include_subfolders)
                self.chk_subfolders.blockSignals(False)
        finally:
            self._updating_tree = False

        self._update_summary_label()

    def select_month(self, month: int) -> None:
        """Filters to photos taken in the specified month (1..12) or all dates (0)."""
        if month <= 0:
            self._date_month = None
        else:
            self._date_month = month
        self._update_summary_label()
        self.filter_changed.emit()

    def build_filter_spec(
        self,
        arrange_by: ArrangeMode = "date",
        reverse_sort: bool = False,
        search_text: str = "",
        bbox: tuple[float, float, float, float] | None = None,
        media_ids: tuple[int, ...] | None = None,
        group_by: str = "month",
        sort_by: str = "date",
        sort_reverse: bool = False,
        group_reverse: bool = False,
    ) -> FilterQuerySpec:
        """Constructs an immutable FilterQuerySpec ready for CatalogRepository.query_media_records."""
        return FilterQuerySpec(
            album_id=self._album_id,
            smart_album_id=self._smart_album_id,
            smart_album_rules=self._smart_album_rules,
            folder_id=self._folder_id,
            include_subfolders=self._include_subfolders,
            date_year=self._date_year,
            date_month=self._date_month,
            date_day=self._date_day,
            include_tag_ids=tuple(sorted(self._include_tag_ids)),
            tag_boolean_mode=self._tag_boolean_mode,
            exclude_tag_ids=tuple(sorted(self._exclude_tag_ids)),
            untagged_only=self._untagged_only,
            person_ids=tuple(sorted(self._person_ids)),
            person_boolean_mode=self._person_boolean_mode,
            ratings=tuple(sorted(self._ratings)),
            flagged_only=self._flagged_only,
            flag_filter=getattr(self, "_flag_filter", (1 if self._flagged_only else None)),
            edited_only=self._edited_only,
            edited_filter=getattr(self, "_edited_filter", (1 if self._edited_only else None)),
            media_type=self._media_type,
            orientation_filter=getattr(self, "_orientation_filter", "all"),
            search_text=search_text,
            bbox=bbox,
            media_ids=media_ids,
            arrange_by=arrange_by,
            reverse_sort=reverse_sort,
            group_by=group_by,
            sort_by=sort_by,
            sort_reverse=sort_reverse,
            group_reverse=group_reverse,
        )
