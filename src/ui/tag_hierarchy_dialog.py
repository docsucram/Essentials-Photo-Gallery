"""
Tag Hierarchy Dialog
====================
WLPG-authentic 'All of your tags' dialog matching media_1792290433097_6cb8c319.png.
Displays:
- Title bar: 'Photo Gallery'
- Header: 'All of your tags' (11pt Segoe UI blue)
- Subtitle: 'Select tags to add to your search.'
- Clean white tree view containing 'Descriptive tags' root and hierarchical tags with tag icons
- Drag-and-drop tag organization (nesting tags as subtags, moving to root, merge warnings)
- Multi-selection support (Ctrl+click, Shift+click)
- 'Clean up duplicate tags' tool
- 'Add' and 'Cancel' buttons at bottom-right
"""

from __future__ import annotations

from typing import Any, Sequence

from PySide6.QtCore import QByteArray, QMimeData, QPoint, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QDrag,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.ui.dialogs import TagMergeConfirmDialog
from src.ui.ribbon_toolbar import make_wlpg_icon

_NODE_TYPE_ROLE = 256
_NODE_PAYLOAD_ROLE = 257
_OLD_NAME_ROLE = 259


class _TagHierarchyTree(QTreeWidget):
    """Custom tree widget supporting drag-and-drop re-parenting and merging of tags."""

    tag_dropped = Signal(int, object)  # (source_tag_id: int, target_tag_id: int | None)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        self._drop_hover_item: QTreeWidgetItem | None = None
        self._drag_start_pos: QPoint | None = None
        self._drag_start_item: QTreeWidgetItem | None = None

        # Floating action badge inside viewport
        self._drag_badge = QLabel(self.viewport())
        self._drag_badge.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._drag_badge.setStyleSheet(
            """
            QLabel {
                background-color: #FFFFFF;
                border: 1px solid #A0B8D0;
                border-radius: 2px;
                color: #003399;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                padding: 1px 6px;
            }
            """
        )
        self._drag_badge.hide()

        self._drag_expand_timer = QTimer(self)
        self._drag_expand_timer.setSingleShot(True)
        self._drag_expand_timer.setInterval(450)
        self._drag_expand_timer.timeout.connect(self._on_drag_expand_timeout)

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
            if target_id == source_tag_id:
                return False

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
                    box = rect.adjusted(1, 1, -2, -1)
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
                    prev_block = self.blockSignals(True)
                    try:
                        self.setCurrentItem(target_item)
                    finally:
                        self.blockSignals(prev_block)
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
                    max(4, pos.x() + 14),
                    max(4, self.viewport().width() - self._drag_badge.width() - 4),
                )
                by = max(2, pos.y() + 16)
                if by + self._drag_badge.height() > self.viewport().height() - 2:
                    by = max(2, pos.y() - self._drag_badge.height() - 6)
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

        event.ignore()


class TagHierarchyDialog(QDialog):
    """Dialog allowing the user to browse, organize (drag-and-drop), and add tags to search filter."""

    tags_added = Signal(list)  # list[int]
    tag_hierarchy_changed = Signal()

    def __init__(self, parent: QWidget | None = None, repo: Any | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self.setWindowTitle("Photo Gallery")
        self.resize(620, 800)
        self.setMinimumSize(540, 650)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #FFFFFF;
            }
            """
        )
        self._tag_items: dict[int, QTreeWidgetItem] = {}
        self._auto_tag_merge_decision: bool | None = None  # For headless testing

        main_vbox = QVBoxLayout(self)
        main_vbox.setContentsMargins(14, 14, 14, 12)
        main_vbox.setSpacing(6)

        # 1. Header: "All of your tags"
        lbl_header = QLabel("All of your tags", self)
        lbl_header.setStyleSheet(
            "color: #1A5296; font-family: 'Segoe UI'; font-size: 11pt; font-weight: normal;"
        )
        main_vbox.addWidget(lbl_header)

        # 2. Subtitle: "Select tags to add to your search, or drag tags to organize them into categories."
        lbl_sub = QLabel("Select tags to add to your search, or drag tags to organize them.", self)
        lbl_sub.setStyleSheet(
            "color: #555555; font-family: 'Segoe UI'; font-size: 9pt; margin-bottom: 4px;"
        )
        main_vbox.addWidget(lbl_sub)

        # 3. Tree view of tags
        self.tree = _TagHierarchyTree(self)
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(16)
        self.tree.setIconSize(QSize(16, 16))
        self.tree.setAnimated(True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setStyleSheet(
            """
            QTreeWidget {
                background: #FFFFFF;
                border: 1px solid #C4D5E6;
                font-family: 'Segoe UI';
                font-size: 9pt;
                color: #1E395B;
                outline: none;
            }
            QTreeWidget::item {
                padding: 2px 4px;
                border: 1px solid transparent;
                border-radius: 2px;
            }
            QTreeWidget::item:hover {
                background: #F4F9FE;
                border: 1px solid #B8D6F5;
            }
            QTreeWidget::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #DCEBFC, stop:1 #C1DCFA);
                border: 1px solid #84ACDD;
                color: #1E395B;
            }
            """
        )
        self.tree.tag_dropped.connect(self._on_tag_dropped)
        main_vbox.addWidget(self.tree, 1)

        # 4. Bottom Button Row: [Clean up duplicates] [Add] [Cancel]
        btn_box = QHBoxLayout()
        btn_box.setContentsMargins(0, 6, 0, 0)
        btn_box.setSpacing(8)

        btn_style = """
            QPushButton {
                font-family: 'Segoe UI';
                font-size: 9pt;
                min-width: 75px;
                height: 23px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F2F4F7, stop:1 #E1E6EB);
                border: 1px solid #8E9BA8;
                border-radius: 2px;
                color: #111111;
                padding: 2px 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #EAF2FB, stop:1 #D2E4F7);
                border: 1px solid #3C7FB1;
            }
            QPushButton:pressed {
                background: #CBE0F5;
                border: 1px solid #2C628B;
            }
            QPushButton:default {
                border: 1px solid #3380CC;
            }
        """

        self.btn_clean_orphans = QPushButton("Clean up duplicate tags", self)
        self.btn_clean_orphans.setStyleSheet(btn_style)
        self.btn_clean_orphans.setMinimumWidth(185)
        self.btn_clean_orphans.setToolTip("Finds orphan root tags that duplicate existing hierarchical tags and consolidates them")
        self.btn_clean_orphans.clicked.connect(self._on_clean_duplicate_tags_clicked)
        btn_box.addWidget(self.btn_clean_orphans)

        btn_box.addStretch(1)

        self.btn_add = QPushButton("Add", self)
        self.btn_add.setDefault(True)
        self.btn_add.setStyleSheet(btn_style)
        self.btn_add.clicked.connect(self._on_add_clicked)
        btn_box.addWidget(self.btn_add)

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setStyleSheet(btn_style)
        self.btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(self.btn_cancel)

        main_vbox.addLayout(btn_box)

    def populate(self, tag_rows: Sequence[dict[str, Any]]) -> None:
        """Populates the tree with tags hierarchy under 'Descriptive tags' root."""
        self.tree.clear()
        self._tag_items.clear()

        tag_icon = make_wlpg_icon("tag", 16)
        italic_font = QFont("Segoe UI", 9)
        italic_font.setItalic(True)

        root = QTreeWidgetItem(self.tree, ["Descriptive tags"])
        root.setFont(0, italic_font)
        root.setIcon(0, tag_icon)
        root.setData(0, _NODE_TYPE_ROLE, "section_tags")
        root.setExpanded(True)

        # First pass: instantiate nodes
        for t in tag_rows:
            tid = int(t["id"])
            tname = str(t["name"])
            cnt = int(t.get("direct_count", 0) or t.get("media_count", 0))
            label = f"{tname} ({cnt})" if cnt > 0 else tname
            item = QTreeWidgetItem([label])
            item.setIcon(0, tag_icon)
            item.setData(0, Qt.UserRole, tid)
            item.setData(0, _NODE_TYPE_ROLE, "tag")
            item.setData(0, _NODE_PAYLOAD_ROLE, tid)
            item.setData(0, _OLD_NAME_ROLE, tname)
            self._tag_items[tid] = item

        # Second pass: attach hierarchy
        for t in tag_rows:
            tid = int(t["id"])
            pid = t.get("parent_id")
            item = self._tag_items[tid]
            if pid is not None and int(pid) in self._tag_items:
                self._tag_items[int(pid)].addChild(item)
            else:
                root.addChild(item)

        root.setExpanded(True)
        for i in range(root.childCount()):
            root.child(i).setExpanded(True)

    def _on_tag_dropped(self, source_tag_id: int, target_tag_id: int | None) -> None:
        """Handles dragging a tag onto another tag or Descriptive tags root."""
        if self.repo is None:
            return
        if source_tag_id == target_tag_id:
            return

        conn = self.repo.engine.catalog_conn()
        source_row = conn.execute(
            "SELECT id, parent_id, name, full_path FROM tags WHERE id = ?;", (source_tag_id,)
        ).fetchone()
        if source_row is None:
            return

        current_parent_id = source_row["parent_id"]
        if (target_tag_id is None and current_parent_id is None) or (target_tag_id == current_parent_id):
            return

        source_name = str(source_row["name"])

        # Check if sibling with same name already exists under target
        existing = self.repo.find_child_tag_by_name(target_tag_id, source_name)
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

            self.repo.merge_tags(source_tag_id, existing_tag_id)
        else:
            self.repo.move_tag(source_tag_id, target_tag_id)

        # Refresh dialog tree
        self.populate(self.repo.get_tag_tree())
        self.tag_hierarchy_changed.emit()

    def _on_clean_duplicate_tags_clicked(self) -> None:
        if self.repo is None:
            return
        conn = self.repo.engine.catalog_conn()
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
            "• Photos tagged with the flat root tag will be reassigned to the matching hierarchical tag.\n"
            "• The redundant top-level tag will be removed from your catalog.\n"
            "• You can restore these tags anytime via the Descriptive tag menu.\n\n"
            f"Found {len(distinct_pairs)} duplicate root tag(s) to merge:\n\n"
            + "\n".join(lines[:12])
            + ("\n  ..." if len(lines) > 12 else "")
            + "\n\nWould you like to proceed with merging these tags?"
        )
        reply = QMessageBox.question(
            self,
            "Clean Up Duplicate Tags",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            backup: list[tuple[str, int, list[int]]] = []
            for rid, (hid, rname, _) in distinct_pairs.items():
                m_rows = conn.execute("SELECT media_id FROM media_tags WHERE tag_id = ?;", (rid,)).fetchall()
                backup.append((rname, hid, [int(mr["media_id"]) for mr in m_rows]))
                self.repo.merge_tags(rid, hid)
            parent_win = self.parent() or self.window()
            if hasattr(parent_win, "_last_tag_cleanup_backup"):
                parent_win._last_tag_cleanup_backup = backup
            self.populate(self.repo.get_tag_tree())
            self.tag_hierarchy_changed.emit()

    def selected_tag_ids(self) -> list[int]:
        """Returns list of selected tag IDs."""
        ids: list[int] = []
        for item in self.tree.selectedItems():
            tid = item.data(0, Qt.UserRole)
            if tid is not None:
                ids.append(int(tid))
        return ids

    def _on_add_clicked(self) -> None:
        sel = self.selected_tag_ids()
        if sel:
            self.tags_added.emit(sel)
        self.accept()
