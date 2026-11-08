"""
Batch People Tag Virtualized Confirmation View for PhotoGallery64.
Modeled on Windows Live Photo Gallery ('SPEC.md' Section 5).

Provides Picasa/WLPG-grade instant face confirmation:
- Grouped sections by suggested person with collapsible headers and '[Confirm All]' button
- Top horizontal People Avatar Picker Strip for 1-click filtering
- 96x96 WebP avatar tiles with hover '[✓]' (Confirm) and '[✗]' (Reject) buttons
- Dismissible guidance banner
- Full-photo context preview popover ('Space' key) centered on face bounding box
- Rapid keyboard shortcuts: Enter (Confirm), Delete/Backspace (Not this person),
  I (Don't show again), Space (Quick preview), Esc (Back to gallery), Ctrl+A (Select all)
- Incremental centroid updating and <10ms database commits
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
from PySide6.QtCore import (
    QAbstractListModel,
    QEvent,
    QItemSelectionModel,
    QModelIndex,
    QPoint,
    QRect,
    QRectF,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QResizeEvent,
    QKeySequence,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QCompleter,
    QDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLayout,
    QLineEdit,
    QListView,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from legacy_reference.imgBrowserv13 import LruCache
from src.core.face_matcher import (
    FaceVectorMatcher,
    deserialize_embedding,
    normalize_vector,
    serialize_embedding,
)
from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.ui.person_header_bar import make_circular_avatar
from src.ui.tag_someone_dialog import TagSomeoneDialog

logger = logging.getLogger(__name__)

TILE_WIDTH = 112
TILE_HEIGHT = 118
THUMB_SIZE = 96


@dataclass(slots=True)
class FaceCardRecord:
    """Lightweight in-memory record for a single detected/suggested face."""

    face_id: int
    media_id: int
    suggested_person_id: int | None
    suggested_name: str | None
    confidence: float
    rect_x: float
    rect_y: float
    rect_w: float
    rect_h: float
    filename: str
    folder_path: str
    is_confirmed: bool = False
    outlier_reason: str | None = None
    boxless_id: int | None = None
    is_boxless_match: bool = False
    swap_person_id: int | None = None
    swap_person_name: str | None = None


class _FaceCardModel(QAbstractListModel):
    """List model holding FaceCardRecord items for a single person's group."""

    def __init__(self, records: list[FaceCardRecord], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._records = records

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._records)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._records)):
            return None
        rec = self._records[index.row()]
        if role == Qt.UserRole:
            return rec
        return None

    def remove_face_ids(self, face_ids: set[int]) -> None:
        """Removes the given face_ids from the model and emits appropriate rowsRemoved signals."""
        i = 0
        while i < len(self._records):
            if self._records[i].face_id in face_ids:
                self.beginRemoveRows(QModelIndex(), i, i)
                self._records.pop(i)
                self.endRemoveRows()
            else:
                i += 1


class _FaceCardDelegate(QStyledItemDelegate):
    """
    Renders a 96x96 polaroid face card tile:
    - 96x96 WebP face crop with subtle rounded border
    - Hover checkmark [✓] (Confirm) and cross [✕] (Not this person) buttons
    - Match confidence badge (e.g. 94%)
    - Blue Aero accent border on selection
    """

    confirm_clicked = Signal(int, int)  # (face_id, person_id)
    reject_clicked = Signal(int, int)   # (face_id, person_id)
    switch_clicked = Signal(int, int, str)  # (face_id, target_person_id, target_person_name)

    def __init__(
        self,
        thumb_repo: ThumbnailRepository,
        pixmap_cache: LruCache,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.thumb_repo = thumb_repo
        self.pixmap_cache = pixmap_cache
        self.hovered_row: int = -1
        self.hover_btn: str | None = None  # "confirm" or "reject"

    def sizeHint(self, option, index) -> QSize:
        return QSize(TILE_WIDTH, TILE_HEIGHT)

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        rec: FaceCardRecord | None = index.data(Qt.UserRole)
        if rec is None:
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        rect = option.rect
        is_selected = bool(option.state & QStyle.State_Selected) if hasattr(option, "state") else False
        is_hovered = (index.row() == self.hovered_row)

        card_rect = rect.adjusted(4, 4, -4, -4)
        is_conf = getattr(rec, "is_confirmed", False)

        is_outlier = bool(getattr(rec, "outlier_reason", None))
        is_boxless = bool(getattr(rec, "is_boxless_match", False))

        # 1. Background card fill & border (Distinct attention border for outliers, blue for boxless, green for confirmed)
        if is_outlier:
            bg_color = QColor("#FEF2F2") if is_selected else (QColor("#FFFBEB") if is_hovered else QColor("#FFFFFF"))
            border_color = QColor("#DC2626") if is_selected else (QColor("#F59E0B") if is_hovered else QColor("#FCA5A5"))
        elif is_boxless:
            bg_color = QColor("#F0F9FF") if is_selected else (QColor("#E0F2FE") if is_hovered else QColor("#FFFFFF"))
            border_color = QColor("#0284C7") if is_selected else (QColor("#38BDF8") if is_hovered else QColor("#BAE6FD"))
        elif is_conf:
            bg_color = QColor("#DDF3E4") if is_selected else (QColor("#EAF7EE") if is_hovered else QColor("#F3FAF5"))
            border_color = QColor("#2E7D32") if is_selected else (QColor("#4CAF50") if is_hovered else QColor("#81C784"))
        else:
            bg_color = QColor("#E5F1FB") if is_selected else (QColor("#F0F8FF") if is_hovered else QColor("#FFFFFF"))
            border_color = QColor("#70C0E7") if is_selected else (QColor("#B8D6FB") if is_hovered else QColor("#D9D9D9"))

        path = QPainterPath()
        path.addRoundedRect(QRectF(card_rect), 4.0, 4.0)
        painter.fillPath(path, QBrush(bg_color))
        painter.setPen(QPen(border_color, 1.8 if (is_selected or is_outlier or is_conf) else 1.0))
        painter.drawPath(path)

        # 2. Draw 96x96 Face Thumbnail
        thumb_x = card_rect.x() + (card_rect.width() - THUMB_SIZE) // 2
        thumb_y = card_rect.y() + 4
        thumb_rect = QRect(thumb_x, thumb_y, THUMB_SIZE, THUMB_SIZE)

        pix = self.pixmap_cache.get(rec.face_id)
        if pix is None:
            blob = self.thumb_repo.get_face_thumbnail(rec.face_id)
            if blob:
                qimg = QImage.fromData(blob, "WEBP")
                if not qimg.isNull():
                    pix = QPixmap.fromImage(qimg)
                    self.pixmap_cache.put(rec.face_id, pix)
                else:
                    pix = QPixmap()
                    self.pixmap_cache.put(rec.face_id, pix)
            else:
                pix = QPixmap()
                self.pixmap_cache.put(rec.face_id, pix)

        if pix and not pix.isNull():
            # Clip avatar to rounded rect
            avatar_path = QPainterPath()
            avatar_path.addRoundedRect(QRectF(thumb_rect), 3.0, 3.0)
            painter.save()
            painter.setClipPath(avatar_path)
            painter.drawPixmap(thumb_rect, pix)
            painter.restore()
        else:
            # Fallback neutral face silhouette
            painter.fillRect(thumb_rect, QColor("#EAEAEA"))
            painter.setPen(QColor("#888888"))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(thumb_rect, Qt.AlignCenter, "Face")

        # 3. Status Badge: "✓ Tagged" (green) vs "? Suggestion" / "? 94%" (amber) / Outlier / Boxless
        badge_font = QFont("Segoe UI", 7, QFont.Bold)
        painter.setFont(badge_font)
        if is_outlier:
            full_reason = str(rec.outlier_reason)
            if "Looks like" in full_reason:
                # e.g. "Looks like Dolores Corsi (72%)" -> "⇄ Dolores?"
                inner = full_reason.replace("Looks like", "").strip()
                tname = inner.split("(")[0].strip()
                t_first = tname.split()[0] if tname else "someone"
                pct_str = ""
                if "(" in inner and ")" in inner:
                    pct_str = inner[inner.find("("):inner.find(")")+1]
                badge_text = f"⇄ {t_first}? {pct_str}".strip()
                badge_bg = QColor("#DC2626")
            elif "No face" in full_reason:
                badge_text = "No Face"
                badge_bg = QColor("#B91C1C")
            else:
                pct = int(round(rec.confidence * 100)) if rec.confidence else 0
                badge_text = f"Low match ({pct}%)" if pct > 0 else "Low match"
                badge_bg = QColor("#D97706")

            fm = painter.fontMetrics()
            elided = fm.elidedText(badge_text, Qt.TextElideMode.ElideRight, THUMB_SIZE - 8)
            bw = min(THUMB_SIZE - 6, fm.horizontalAdvance(elided) + 8)
            bh = 15
            badge_rect = QRect(thumb_x + 3, thumb_y + 3, bw, bh)
            badge_path = QPainterPath()
            badge_path.addRoundedRect(QRectF(badge_rect), 3.0, 3.0)
            painter.fillPath(badge_path, QBrush(badge_bg))
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(badge_rect, Qt.AlignCenter, elided)
        elif is_boxless:
            badge_text = "🔗 Missing Box"
            badge_bg = QColor("#0284C7")
            bw = 68
            bh = 15
            badge_rect = QRect(thumb_x + 3, thumb_y + 3, bw, bh)
            badge_path = QPainterPath()
            badge_path.addRoundedRect(QRectF(badge_rect), 3.0, 3.0)
            painter.fillPath(badge_path, QBrush(badge_bg))
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(badge_rect, Qt.AlignCenter, badge_text)
        elif is_conf:
            badge_text = "✓ Tagged"
            badge_bg = QColor("#2E7D32")
            bw = 50
            bh = 15
            badge_rect = QRect(thumb_x + 3, thumb_y + 3, bw, bh)
            badge_path = QPainterPath()
            badge_path.addRoundedRect(QRectF(badge_rect), 3.0, 3.0)
            painter.fillPath(badge_path, QBrush(badge_bg))
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(badge_rect, Qt.AlignCenter, badge_text)
        else:
            pct_val = int(round(rec.confidence * 100)) if rec.confidence > 0.05 else None
            badge_text = f"? {pct_val}%" if pct_val is not None else "? Suggestion"
            badge_bg = QColor("#D97706") if pct_val is not None else QColor("#0284C7")
            bw = 44 if pct_val is not None else 64
            bh = 15
            badge_rect = QRect(thumb_x + 3, thumb_y + 3, bw, bh)
            badge_path = QPainterPath()
            badge_path.addRoundedRect(QRectF(badge_rect), 3.0, 3.0)
            painter.fillPath(badge_path, QBrush(badge_bg))
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(badge_rect, Qt.AlignCenter, badge_text)

        # 4. Hover Action Overlay Buttons: [⇄] (Switch if swap outlier), [✓] (Confirm/Keep), and [✕] (Reject/Remove)
        show_action_buttons = (not getattr(rec, "is_confirmed", False)) or bool(getattr(rec, "outlier_reason", None)) or bool(getattr(rec, "is_boxless_match", False))
        if (is_hovered or is_selected) and show_action_buttons:
            has_swap = bool(getattr(rec, "swap_person_name", None)) or ("Looks like" in getattr(rec, "outlier_reason", ""))
            if has_swap:
                # Switch person button (top right - 58px)
                s_rect = QRect(thumb_x + THUMB_SIZE - 58, thumb_y + 3, 16, 16)
                s_hover = (is_hovered and self.hover_btn == "switch")
                s_bg = QColor("#0284C7") if s_hover else QColor(0, 0, 0, 160)
                painter.setBrush(QBrush(s_bg))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(s_rect)
                painter.setPen(QColor("#FFFFFF"))
                painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
                painter.drawText(s_rect, Qt.AlignCenter, "⇄")

            # Checkmark button (top right - 38px)
            c_rect = QRect(thumb_x + THUMB_SIZE - 38, thumb_y + 3, 16, 16)
            c_hover = (is_hovered and self.hover_btn == "confirm")
            c_bg = QColor("#2E8B57") if c_hover else QColor(0, 0, 0, 150)
            painter.setBrush(QBrush(c_bg))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(c_rect)
            painter.setPen(QPen(QColor("#FFFFFF"), 1.8))
            # Draw checkmark
            cx = c_rect.x()
            cy = c_rect.y()
            painter.drawLine(cx + 4, cy + 8, cx + 7, cy + 12)
            painter.drawLine(cx + 7, cy + 12, cx + 12, cy + 5)

            # Reject cross button (top right - 18px)
            r_rect = QRect(thumb_x + THUMB_SIZE - 18, thumb_y + 3, 16, 16)
            r_hover = (is_hovered and self.hover_btn == "reject")
            r_bg = QColor("#D32F2F") if r_hover else QColor(0, 0, 0, 150)
            painter.setBrush(QBrush(r_bg))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(r_rect)
            painter.setPen(QPen(QColor("#FFFFFF"), 1.8))
            # Draw cross
            rx = r_rect.x()
            ry = r_rect.y()
            painter.drawLine(rx + 5, ry + 5, rx + 11, ry + 11)
            painter.drawLine(rx + 11, ry + 5, rx + 5, ry + 11)

        painter.restore()

    def editorEvent(self, event: QEvent, model: QAbstractListModel, option, index: QModelIndex) -> bool:
        if not index.isValid():
            return False

        rec: FaceCardRecord | None = index.data(Qt.UserRole)
        if rec is None:
            return False

        has_swap = bool(getattr(rec, "swap_person_name", None)) or ("Looks like" in getattr(rec, "outlier_reason", ""))
        thumb_x = option.rect.x() + 4 + (option.rect.width() - 8 - THUMB_SIZE) // 2
        thumb_y = option.rect.y() + 8
        s_rect = QRect(thumb_x + THUMB_SIZE - 58, thumb_y + 3, 16, 16) if has_swap else QRect()
        c_rect = QRect(thumb_x + THUMB_SIZE - 38, thumb_y + 3, 16, 16)
        r_rect = QRect(thumb_x + THUMB_SIZE - 18, thumb_y + 3, 16, 16)

        if event.type() == QEvent.MouseMove:
            pos = event.pos()
            self.hovered_row = index.row()
            if has_swap and s_rect.contains(pos):
                self.hover_btn = "switch"
            elif c_rect.contains(pos):
                self.hover_btn = "confirm"
            elif r_rect.contains(pos):
                self.hover_btn = "reject"
            else:
                self.hover_btn = None
            if hasattr(option, "widget") and option.widget:
                if hasattr(option.widget, "viewport") and option.widget.viewport():
                    option.widget.viewport().update()
                else:
                    option.widget.update(index)
            return False

        elif event.type() == QEvent.MouseButtonRelease:
            pos = event.pos()
            pid = rec.suggested_person_id or 0
            if has_swap and s_rect.contains(pos):
                t_pid = getattr(rec, "swap_person_id", None) or 0
                t_name = getattr(rec, "swap_person_name", None) or ""
                if not t_name and "Looks like" in getattr(rec, "outlier_reason", ""):
                    inner = rec.outlier_reason.replace("Looks like", "").strip()
                    t_name = inner.split("(")[0].strip()
                self.switch_clicked.emit(rec.face_id, t_pid, t_name)
                return True
            elif c_rect.contains(pos):
                self.confirm_clicked.emit(rec.face_id, pid)
                return True
            elif r_rect.contains(pos):
                self.reject_clicked.emit(rec.face_id, pid)
                return True

        return False


class _FaceQuickPreviewDialog(QDialog):
    """
    Instant non-modal full-photo quick-preview popover centered on the face box
    ('SPEC.md' Section 5.4, Space key trigger).
    """

    def __init__(self, record: FaceCardRecord, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Context Preview — {record.filename}")
        self.resize(800, 640)
        self.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header info
        h_info = QHBoxLayout()
        person_str = f"Suggested: {record.suggested_name}" if record.suggested_name else "Unnamed face"
        conf_str = f"({int(round(record.confidence * 100))}% match)" if record.confidence > 0 else ""
        lbl_info = QLabel(f"<b>{record.filename}</b> — {person_str} {conf_str}", self)
        lbl_info.setStyleSheet("font-size: 13px; color: #E0E0E0;")
        h_info.addWidget(lbl_info)
        h_info.addStretch(1)

        btn_close = QPushButton("✕ Close", self)
        btn_close.setFixedWidth(75)
        btn_close.setStyleSheet("background-color: #333333; color: white; border: 1px solid #555555; padding: 4px;")
        btn_close.clicked.connect(self.accept)
        h_info.addWidget(btn_close)
        layout.addLayout(h_info)

        # Image canvas
        self.lbl_image = QLabel(self)
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.lbl_image, 1)

        self._load_and_highlight(record)

    def _load_and_highlight(self, record: FaceCardRecord) -> None:
        fpath = Path(record.folder_path) / record.filename
        if not fpath.exists():
            self.lbl_image.setText("Image not found on disk.")
            return

        try:
            with Image.open(fpath) as pil_img:
                rgb = pil_img.convert("RGB")
                w, h = rgb.size
                qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
                base_pix = QPixmap.fromImage(qimg)

            # Draw highlight rectangle over face
            highlighted = QPixmap(base_pix.size())
            highlighted.fill(Qt.transparent)
            painter = QPainter(highlighted)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.drawPixmap(0, 0, base_pix)

            fx = int(round(record.rect_x * w))
            fy = int(round(record.rect_y * h))
            fw = int(round(record.rect_w * w))
            fh = int(round(record.rect_h * h))
            face_box = QRect(fx, fy, fw, fh)

            # Gold / cyan glowing rectangle
            pen = QPen(QColor("#00E5FF"), max(2, int(w / 350.0)))
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 229, 255, 30)))
            painter.drawRoundedRect(face_box, 4, 4)

            # Name badge
            tag_text = record.suggested_name or "Face"
            badge_font = QFont("Segoe UI", max(10, int(w / 70.0)), QFont.Bold)
            painter.setFont(badge_font)
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(tag_text) + 12
            th = fm.height() + 6
            t_rect = QRect(fx, max(0, fy - th - 4), tw, th)
            painter.fillRect(t_rect, QColor(0, 0, 0, 180))
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(t_rect, Qt.AlignCenter, tag_text)
            painter.end()

            # Scale to fit label
            scaled = highlighted.scaled(760, 560, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_image.setPixmap(scaled)
        except Exception as exc:
            self.lbl_image.setText(f"Error loading preview: {exc}")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Space, Qt.Key_Escape):
            self.accept()
        else:
            super().keyPressEvent(event)


class _AutoHeightListView(QListView):
    """
    QListView that dynamically adjusts its height to fit all wrapped rows of face cards.
    Eliminates internal vertical clipping so all rows wrap naturally inside outer QScrollArea.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def sizeHint(self) -> QSize:
        m = self.model()
        count = m.rowCount() if m is not None else 0
        if count == 0:
            return QSize(0, 0)
        vw = self.viewport().width()
        if vw <= 50:
            vw = self.width()
        if vw <= 50:
            vw = 1000
        cols = max(1, vw // TILE_WIDTH)
        rows = (count + cols - 1) // cols
        return QSize(vw, rows * TILE_HEIGHT + 12)

    def update_height(self) -> None:
        sh = self.sizeHint()
        if self.height() != sh.height():
            self.setFixedHeight(sh.height())
            self.updateGeometry()

    def update(self, *args) -> None:
        if args:
            super().update(*args)
        elif self.viewport():
            self.viewport().update()
        else:
            QWidget.update(self)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.update_height()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            p = self.parent()
            while p is not None:
                if hasattr(p, "back_to_gallery_requested"):
                    p.back_to_gallery_requested.emit()
                    event.accept()
                    return
                p = p.parent()
        super().keyPressEvent(event)


class _PersonGroupSectionWidget(QWidget):
    """
    Collapsible group section representing suggestions for a single person.
    Contains header chevron, counts, [Confirm All (N)] button or inline naming box, and child QListView.
    """

    confirm_all_requested = Signal(int, list)  # (person_id, list[int] face_ids)
    confirm_faces_requested = Signal(int, list)  # (person_id, list[int] face_ids)
    reject_faces_requested = Signal(int, list)   # (person_id, list[int] face_ids)
    not_a_face_requested = Signal(list)          # (list[int] face_ids)
    ignore_faces_requested = Signal(list)        # (list[int] face_ids)
    tag_as_requested = Signal(list)              # (list[int] face_ids)
    name_cluster_requested = Signal(str, list)   # (person_name, list[int] face_ids)
    open_media_requested = Signal(int)           # (media_id)
    preview_requested = Signal(object)           # (FaceCardRecord)
    rename_merge_requested = Signal(int, str)    # (person_id, person_name)

    def __init__(
        self,
        person_id: int | None,
        person_name: str,
        confirmed_count: int,
        records: list[FaceCardRecord],
        thumb_repo: ThumbnailRepository,
        pixmap_cache: LruCache,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.person_id = person_id
        self.person_name = person_name
        self.confirmed_count = confirmed_count
        self.thumb_repo = thumb_repo
        self.pixmap_cache = pixmap_cache
        self.is_collapsed: bool = False
        self.is_unknown_cluster: bool = bool(not person_id and str(person_name).startswith("Unknown Person"))
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self._build_ui(records)

    def _build_ui(self, records: list[FaceCardRecord]) -> None:
        v_layout = QVBoxLayout(self)
        v_layout.setContentsMargins(0, 4, 0, 8)
        v_layout.setSpacing(4)

        # Header Frame
        self.header_frame = QFrame(self)
        self.header_frame.setObjectName("PersonHeaderFrame")
        self.header_frame.setStyleSheet(
            "QFrame#PersonHeaderFrame { background-color: #F4F7FB; border: 1px solid #D0DBE5; border-radius: 4px; padding: 2px; }"
        )
        h_layout = QHBoxLayout(self.header_frame)
        h_layout.setContentsMargins(8, 4, 8, 4)
        h_layout.setSpacing(8)

        self.btn_toggle = QToolButton(self.header_frame)
        self.btn_toggle.setText("▼")
        self.btn_toggle.setStyleSheet("border: none; font-size: 10px; color: #55687C; background: transparent;")
        self.btn_toggle.clicked.connect(self._toggle_collapse)
        h_layout.addWidget(self.btn_toggle)

        cnt = len(records)
        if self.person_id is not None:
            # Interactive clickable name linking to rename/merge with suggestions
            self.lbl_name = QLabel(self.person_name, self.header_frame)
            self.lbl_name.setCursor(Qt.CursorShape.PointingHandCursor)
            self.lbl_name.setToolTip(f"Click to rename or merge '{self.person_name}' with existing contacts...")
            self.lbl_name.setStyleSheet(
                "QLabel { font-weight: bold; font-size: 13px; color: #005A9E; background: transparent; border: none; padding: 1px 4px; border-radius: 3px; }"
                "QLabel:hover { background-color: #E2E8F0; text-decoration: underline; }"
            )
            self.lbl_name.mousePressEvent = lambda ev: (
                self.rename_merge_requested.emit(self.person_id, self.person_name)
                if ev.button() == Qt.MouseButton.LeftButton else None
            )
            h_layout.addWidget(self.lbl_name)

            self.lbl_counts = QLabel("", self.header_frame)
            self.lbl_counts.setStyleSheet("font-size: 11px; color: #55687C; background: transparent; border: none;")
            h_layout.addWidget(self.lbl_counts)
            self.lbl_title = self.lbl_counts
        elif self.is_unknown_cluster:
            title_text = f"{self.person_name} ({cnt} face{'s' if cnt != 1 else ''})"
            self.lbl_title = QLabel(title_text, self.header_frame)
            self.lbl_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #1E395B;")
            h_layout.addWidget(self.lbl_title)
        else:
            title_text = f"{self.person_name} ({cnt} face{'s' if cnt != 1 else ''})"
            self.lbl_title = QLabel(title_text, self.header_frame)
            self.lbl_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #1E395B;")
            h_layout.addWidget(self.lbl_title)
        h_layout.addStretch(1)

        if self.person_id is not None:
            self.btn_confirm_all = QPushButton("✓ Confirm All", self.header_frame)
            self.btn_confirm_all.setStyleSheet(
                "QPushButton { background-color: #2E8B57; color: white; font-weight: bold; padding: 4px 10px; border-radius: 3px; border: 1px solid #236B42; }"
                "QPushButton:hover { background-color: #3CB371; }"
            )
            self.btn_confirm_all.clicked.connect(self._on_confirm_all_clicked)
            h_layout.addWidget(self.btn_confirm_all)
            self.txt_name = None
            self.btn_save_name = None
        elif self.is_unknown_cluster:
            self.btn_confirm_all = None
            self.txt_name = QLineEdit(self.header_frame)
            self.txt_name.setPlaceholderText("Name this person...")
            self.txt_name.setFixedWidth(160)
            self.txt_name.setStyleSheet(
                "QLineEdit { background-color: #FFFFFF; border: 1px solid #B0C4DE; border-radius: 3px; padding: 3px 6px; font-size: 11px; }"
                "QLineEdit:focus { border: 1px solid #2B579A; }"
            )
            self.btn_save_name = QPushButton("Save", self.header_frame)
            self.btn_save_name.setStyleSheet(
                "QPushButton { background-color: #2E8B57; color: white; font-weight: bold; padding: 3px 10px; border-radius: 3px; border: 1px solid #236B42; font-size: 11px; }"
                "QPushButton:hover { background-color: #3CB371; }"
            )
            self.txt_name.returnPressed.connect(self._on_save_name_clicked)
            self.btn_save_name.clicked.connect(self._on_save_name_clicked)
            h_layout.addWidget(self.txt_name)
            h_layout.addWidget(self.btn_save_name)
        else:
            self.btn_confirm_all = None
            self.txt_name = None
            self.btn_save_name = None

        v_layout.addWidget(self.header_frame)

        # Child List View
        self.model = _FaceCardModel(records, self)
        self.delegate = _FaceCardDelegate(self.thumb_repo, self.pixmap_cache, self)
        self.delegate.confirm_clicked.connect(self._on_delegate_confirm)
        self.delegate.reject_clicked.connect(self._on_delegate_reject)
        self.delegate.switch_clicked.connect(self._on_delegate_switch)

        self.list_view = _AutoHeightListView(self)
        self.list_view.setModel(self.model)
        self.list_view.setItemDelegate(self.delegate)
        self.list_view.setViewMode(QListView.IconMode)
        self.list_view.setFlow(QListView.LeftToRight)
        self.list_view.setWrapping(True)
        self.list_view.setResizeMode(QListView.Adjust)
        self.list_view.setUniformItemSizes(True)
        self.list_view.setGridSize(QSize(TILE_WIDTH, TILE_HEIGHT))
        self.list_view.setSelectionMode(QListView.ExtendedSelection)
        self.list_view.setStyleSheet("QListView { border: none; background: transparent; }")
        self.list_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_view.customContextMenuRequested.connect(self._show_context_menu)
        self.list_view.doubleClicked.connect(self._on_double_clicked)

        self.list_view.selectionModel().selectionChanged.connect(lambda *args: self._update_header_text())
        self.model.modelReset.connect(self.list_view.update_height)
        self.model.rowsInserted.connect(lambda *args: self.list_view.update_height())
        self.model.rowsRemoved.connect(lambda *args: self.list_view.update_height())
        self.list_view.update_height()
        v_layout.addWidget(self.list_view)
        self._update_header_text()

    def _update_header_text(self) -> None:
        records = self.model._records
        outliers = [r for r in records if getattr(r, "outlier_reason", None)]
        boxless = [r for r in records if getattr(r, "is_boxless_match", False)]
        unconf = [r for r in records if not getattr(r, "is_confirmed", False) and not getattr(r, "outlier_reason", None) and not getattr(r, "is_boxless_match", False)]

        sel_cnt = len(self.list_view.selectionModel().selectedIndexes())
        sel_text = f", {sel_cnt} selected" if sel_cnt > 0 else ""

        if self.person_id is not None:
            if outliers:
                o_cnt = len(outliers)
                title_name = self.person_name
                confirmed_info = f", {self.confirmed_count} confirmed" if self.confirmed_count > 0 else ""
                counts_text = f"({o_cnt} flagged for review{sel_text}{confirmed_info})"
                short_p = self.person_name if len(self.person_name) <= 14 else (self.person_name[:12] + "…")
                btn_text = f"✓ Keep All as {short_p} ({o_cnt})"
                btn_enabled = o_cnt > 0
            elif boxless:
                b_cnt = len(boxless)
                title_name = self.person_name
                confirmed_info = f", {self.confirmed_count} confirmed" if self.confirmed_count > 0 else ""
                counts_text = f"({b_cnt} missing box{'' if b_cnt == 1 else 'es'}{sel_text}{confirmed_info})"
                btn_text = f"🔗 Link All ({b_cnt})"
                btn_enabled = b_cnt > 0
            else:
                u_cnt = len(unconf)
                title_name = f"{self.person_name}?" if u_cnt > 0 else self.person_name
                confirmed_info = f", {self.confirmed_count} confirmed" if self.confirmed_count > 0 else ""
                counts_text = f"({u_cnt} suggestion{'s' if u_cnt != 1 else ''}{sel_text}{confirmed_info})" if u_cnt > 0 else f"({self.confirmed_count} confirmed{sel_text})"
                btn_text = f"✓ Confirm All ({u_cnt})"
                btn_enabled = u_cnt > 0

            if hasattr(self, "lbl_name"):
                self.lbl_name.setText(title_name)
            if hasattr(self, "lbl_counts"):
                self.lbl_counts.setText(counts_text)
            elif hasattr(self, "lbl_title"):
                self.lbl_title.setText(f"{self.person_name} {counts_text}")
            if self.btn_confirm_all:
                self.btn_confirm_all.setText(btn_text)
                self.btn_confirm_all.setEnabled(btn_enabled)
        elif self.is_unknown_cluster:
            total_cnt = len(self.model._records)
            title_text = f"{self.person_name} ({total_cnt} face{'s' if total_cnt != 1 else ''}{sel_text})"
            if hasattr(self, "lbl_title"):
                self.lbl_title.setText(title_text)
        else:
            total_cnt = len(self.model._records)
            title_text = f"{self.person_name} ({total_cnt} face{'s' if total_cnt != 1 else ''}{sel_text})"
            if hasattr(self, "lbl_title"):
                self.lbl_title.setText(title_text)

    def _on_save_name_clicked(self) -> None:
        if not self.txt_name:
            return
        name = self.txt_name.text().strip()
        if not name:
            return
        face_ids = [r.face_id for r in self.model._records]
        if face_ids:
            self.name_cluster_requested.emit(name, face_ids)

    def _toggle_collapse(self) -> None:
        self.is_collapsed = not self.is_collapsed
        self.list_view.setVisible(not self.is_collapsed)
        self.btn_toggle.setText("▶" if self.is_collapsed else "▼")
        if not self.is_collapsed:
            self.list_view.update_height()

    def _on_confirm_all_clicked(self) -> None:
        face_ids = [r.face_id for r in self.model._records]
        if face_ids and self.person_id is not None:
            self.confirm_all_requested.emit(self.person_id, face_ids)

    def _on_delegate_confirm(self, face_id: int, person_id: int) -> None:
        rec = next((r for r in self.model._records if r.face_id == face_id), None)
        if rec and getattr(rec, "outlier_reason", None):
            try:
                self.parent().repo.verify_face_tag(face_id)
                self.model._records = [r for r in self.model._records if r.face_id != face_id]
                self.model.layoutChanged.emit()
                self._update_header_text()
                if hasattr(self.parent(), "refresh_suggestions"):
                    self.parent().refresh_suggestions()
                return
            except Exception as exc:
                logger.warning("Failed to verify outlier tag: %s", exc)

        if rec and getattr(rec, "is_boxless_match", False) and rec.suggested_person_id:
            try:
                self.parent().repo.assign_face_to_person(rec.face_id, rec.suggested_person_id)
                self.model._records = [r for r in self.model._records if r.face_id != face_id]
                self.model.layoutChanged.emit()
                self._update_header_text()
                if hasattr(self.parent(), "faces_confirmed"):
                    self.parent().faces_confirmed.emit([face_id], rec.suggested_person_id)
                if hasattr(self.parent(), "refresh_suggestions"):
                    self.parent().refresh_suggestions()
                return
            except Exception as exc:
                logger.warning("Failed to link boxless face: %s", exc)

        target_pid = person_id or self.person_id or 0
        if target_pid > 0:
            self.confirm_faces_requested.emit(target_pid, [face_id])
        else:
            self.tag_as_requested.emit([face_id])

    def _on_delegate_reject(self, face_id: int, person_id: int) -> None:
        rec = next((r for r in self.model._records if r.face_id == face_id), None)
        if rec and getattr(rec, "outlier_reason", None):
            try:
                target_pid = person_id or self.person_id
                self.parent().repo.unassign_face_person(face_id, target_pid)
                self.model._records = [r for r in self.model._records if r.face_id != face_id]
                self.model.layoutChanged.emit()
                self._update_header_text()
                if hasattr(self.parent(), "faces_rejected"):
                    self.parent().faces_rejected.emit([face_id])
                if hasattr(self.parent(), "refresh_suggestions"):
                    self.parent().refresh_suggestions()
                return
            except Exception as exc:
                logger.warning("Failed to unassign outlier face: %s", exc)

        target_pid = person_id or self.person_id or 0
        self.reject_faces_requested.emit(target_pid, [face_id])

    def _on_delegate_switch(self, face_id: int, target_pid: int, target_pname: str) -> None:
        rec = next((r for r in self.model._records if r.face_id == face_id), None)
        t_pid = target_pid or (getattr(rec, "swap_person_id", None) if rec else None)
        t_name = target_pname or (getattr(rec, "swap_person_name", None) if rec else None)
        if not t_pid and not t_name and rec and getattr(rec, "outlier_reason", None):
            reason = str(rec.outlier_reason)
            if "Looks like" in reason:
                inner = reason.replace("Looks like", "").strip()
                t_name = inner.split("(")[0].strip()

        parent_view = self.parent()
        repo = getattr(parent_view, "repo", None)
        dest = t_pid if (t_pid and t_pid > 0) else t_name
        if repo and dest:
            try:
                new_pid = repo.assign_face_to_person(face_id, dest)
                repo.verify_face_tag(face_id)
                self.model._records = [r for r in self.model._records if r.face_id != face_id]
                self.model.layoutChanged.emit()
                self._update_header_text()
                if hasattr(parent_view, "faces_confirmed"):
                    parent_view.faces_confirmed.emit([face_id], new_pid)
                if hasattr(parent_view, "refresh_suggestions"):
                    parent_view.refresh_suggestions()
            except Exception as exc:
                logger.warning("Failed to switch face %d to %s: %s", face_id, dest, exc)

    def _on_double_clicked(self, index: QModelIndex) -> None:
        rec = index.data(Qt.UserRole)
        if rec:
            self.open_media_requested.emit(rec.media_id)

    def get_selected_records(self) -> list[FaceCardRecord]:
        indexes = self.list_view.selectionModel().selectedIndexes()
        records: list[FaceCardRecord] = []
        for idx in indexes:
            r = idx.data(Qt.UserRole)
            if r:
                records.append(r)
        return records

    def select_all(self) -> None:
        self.list_view.selectAll()

    def set_dark_mode(self, is_dark: bool) -> None:
        """Applies dark or light theme to section header."""
        self._is_dark = bool(is_dark)
        if is_dark:
            self.header_frame.setStyleSheet(
                "QFrame#PersonHeaderFrame { background-color: #1E232B; border: 1px solid #2D333F; border-radius: 4px; padding: 2px; }"
            )
            if hasattr(self, "lbl_name"):
                self.lbl_name.setStyleSheet(
                    "QLabel { font-weight: bold; font-size: 13px; color: #60A5FA; background: transparent; border: none; padding: 1px 4px; border-radius: 3px; }"
                    "QLabel:hover { background-color: #2D3748; text-decoration: underline; }"
                )
            if hasattr(self, "lbl_counts"):
                self.lbl_counts.setStyleSheet("font-size: 11px; color: #94A3B8; background: transparent; border: none;")
            self.btn_toggle.setStyleSheet("border: none; font-size: 10px; color: #94A3B8; background: transparent;")
        else:
            self.header_frame.setStyleSheet(
                "QFrame#PersonHeaderFrame { background-color: #F4F7FB; border: 1px solid #D0DBE5; border-radius: 4px; padding: 2px; }"
            )
            if hasattr(self, "lbl_name"):
                self.lbl_name.setStyleSheet(
                    "QLabel { font-weight: bold; font-size: 13px; color: #005A9E; background: transparent; border: none; padding: 1px 4px; border-radius: 3px; }"
                    "QLabel:hover { background-color: #E2E8F0; text-decoration: underline; }"
                )
            if hasattr(self, "lbl_counts"):
                self.lbl_counts.setStyleSheet("font-size: 11px; color: #55687C; background: transparent; border: none;")
            self.btn_toggle.setStyleSheet("border: none; font-size: 10px; color: #55687C; background: transparent;")

    def remove_face_ids(self, face_ids: set[int]) -> int:
        """Removes face IDs from this section. Returns remaining count."""
        self.model.remove_face_ids(face_ids)
        self._update_header_text()
        return len(self.model._records)

    def _show_context_menu(self, pos: QPoint) -> None:
        sel = self.get_selected_records()
        if not sel:
            idx = self.list_view.indexAt(pos)
            if idx.isValid():
                self.list_view.selectionModel().select(idx, QItemSelectionModel.Select)
                sel = self.get_selected_records()

        if not sel:
            return

        face_ids = [r.face_id for r in sel]
        menu = QMenu(self)
        is_outlier = any(getattr(r, "outlier_reason", None) for r in sel)

        act_switch = None
        act_confirm = None
        act_reject = None
        act_not_a_face = None
        act_tag_as = None
        act_ignore = None

        if is_outlier:
            # Check for swap suggestion
            swap_target_name = None
            swap_target_pid = None
            for r in sel:
                s_name = getattr(r, "swap_person_name", None)
                if not s_name and "Looks like" in getattr(r, "outlier_reason", ""):
                    inner = r.outlier_reason.replace("Looks like", "").strip()
                    s_name = inner.split("(")[0].strip()
                if s_name:
                    swap_target_name = s_name
                    swap_target_pid = getattr(r, "swap_person_id", None)
                    break

            if swap_target_name:
                act_switch = menu.addAction(f"⇄ Change to {swap_target_name}")
                bold_font = QFont()
                bold_font.setBold(True)
                act_switch.setFont(bold_font)

            act_confirm = menu.addAction(f"✓ Keep as {self.person_name} (Mark verified)")
            act_reject = menu.addAction("✕ Remove tag from photo")
            act_not_a_face = menu.addAction("Not a face (Delete box)")
            act_tag_as = menu.addAction("Tag as somebody else...")
        else:
            act_confirm = menu.addAction(f"✓ Confirm ({len(face_ids)})")
            act_confirm.setEnabled(bool(self.person_id))
            act_reject = menu.addAction(f"✕ Not this person ({len(face_ids)})")
            act_not_a_face = menu.addAction("Not a face (Delete region)")
            act_tag_as = menu.addAction("Tag as...")
            act_ignore = menu.addAction("Don't show again")

        menu.addSeparator()
        act_rename = None
        if self.person_id is not None:
            act_rename = menu.addAction(f"✏ Rename or merge '{self.person_name}'...")
            menu.addSeparator()
        act_open_single = menu.addAction("View photo in single image view")
        act_view_full = menu.addAction("View full photo (Space)")

        chosen = None
        if not getattr(self, "_suppress_menu_exec", False):
            chosen = menu.exec(self.list_view.viewport().mapToGlobal(pos))

        parent_view = self.parent()
        repo = getattr(parent_view, "repo", None)

        if chosen == act_switch and swap_target_name and repo:
            for r in sel:
                dest = getattr(r, "swap_person_id", None) or getattr(r, "swap_person_name", None) or swap_target_pid or swap_target_name
                if dest:
                    try:
                        new_pid = repo.assign_face_to_person(r.face_id, dest)
                        repo.verify_face_tag(r.face_id)
                        if hasattr(parent_view, "faces_confirmed"):
                            parent_view.faces_confirmed.emit([r.face_id], new_pid)
                    except Exception as exc:
                        logger.warning("Failed to switch outlier face: %s", exc)
            self.model._records = [r for r in self.model._records if r.face_id not in set(face_ids)]
            self.model.layoutChanged.emit()
            self._update_header_text()
            if hasattr(parent_view, "refresh_suggestions"):
                parent_view.refresh_suggestions()
        elif chosen == act_confirm:
            if is_outlier and repo:
                for fid in face_ids:
                    try:
                        repo.verify_face_tag(fid)
                    except Exception as exc:
                        logger.warning("Failed to verify outlier tag: %s", exc)
                self.model._records = [r for r in self.model._records if r.face_id not in set(face_ids)]
                self.model.layoutChanged.emit()
                self._update_header_text()
                if hasattr(parent_view, "refresh_suggestions"):
                    parent_view.refresh_suggestions()
            elif self.person_id is not None:
                self.confirm_faces_requested.emit(self.person_id, face_ids)
        elif chosen == act_reject:
            if is_outlier and repo:
                for fid in face_ids:
                    try:
                        repo.unassign_face_person(fid, self.person_id)
                    except Exception as exc:
                        logger.warning("Failed to unassign outlier tag: %s", exc)
                self.model._records = [r for r in self.model._records if r.face_id not in set(face_ids)]
                self.model.layoutChanged.emit()
                self._update_header_text()
                if hasattr(parent_view, "faces_rejected"):
                    parent_view.faces_rejected.emit(face_ids)
                if hasattr(parent_view, "refresh_suggestions"):
                    parent_view.refresh_suggestions()
            else:
                self.reject_faces_requested.emit(self.person_id or 0, face_ids)
        elif chosen == act_not_a_face:
            self.not_a_face_requested.emit(face_ids)
        elif chosen == act_tag_as:
            self.tag_as_requested.emit(face_ids)
        elif chosen == act_ignore:
            self.ignore_faces_requested.emit(face_ids)
        elif chosen == act_rename and self.person_id is not None:
            self.rename_merge_requested.emit(self.person_id, self.person_name)
        elif chosen == act_open_single and sel:
            self.open_media_requested.emit(sel[0].media_id)
        elif chosen == act_view_full and sel:
            self.preview_requested.emit(sel[0])
        return menu


class _PeopleAvatarPickerStrip(QWidget):
    """
    Horizontal top strip displaying circular avatars of people with suggestions.
    Allows 1-click filtering of the suggestion grid ('SPEC.md' Section 5.1).
    Scrollable with smooth mouse-wheel navigation and overflow chevrons so
    names never overlap, crowd, or get crushed even with dozens of people.
    """

    person_selected = Signal(object)  # person_id (int) or None for All
    rename_merge_requested = Signal(int, str)  # (person_id, person_name)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(52)
        self._is_dark: bool = False
        self.__active_person_id: int | None = None
        self._buttons: dict[int | None, QPushButton] = {}

        # Main layout holds left chevron, scroll area, right chevron
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 2, 6, 2)
        main_layout.setSpacing(4)

        # Left scroll button (‹)
        self.btn_left = QToolButton(self)
        self.btn_left.setText("‹")
        self.btn_left.setFixedSize(26, 32)
        self.btn_left.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_left.setToolTip("Scroll left")
        self.btn_left.clicked.connect(lambda: self._scroll_by(-280))
        self.btn_left.setVisible(False)
        main_layout.addWidget(self.btn_left)

        # Horizontal Scroll Area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.viewport().installEventFilter(self)

        # Scroll Content Container
        self.scroll_content = QWidget(self.scroll_area)
        self.scroll_content.installEventFilter(self)
        self._strip_layout = QHBoxLayout(self.scroll_content)
        self._strip_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self._strip_layout.setContentsMargins(4, 6, 4, 6)
        self._strip_layout.setSpacing(6)
        self._strip_layout.addStretch(1)

        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area, 1)

        # Right scroll button (›)
        self.btn_right = QToolButton(self)
        self.btn_right.setText("›")
        self.btn_right.setFixedSize(26, 32)
        self.btn_right.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_right.setToolTip("Scroll right")
        self.btn_right.clicked.connect(lambda: self._scroll_by(280))
        self.btn_right.setVisible(False)
        main_layout.addWidget(self.btn_right)

        # Listen for scroll updates to show/hide chevrons
        sb = self.scroll_area.horizontalScrollBar()
        sb.valueChanged.connect(self._update_scroll_buttons)
        sb.rangeChanged.connect(self._update_scroll_buttons)

        self._apply_theme()

    @property
    def _active_person_id(self) -> int | None:
        return self.__active_person_id

    @_active_person_id.setter
    def _active_person_id(self, person_id: int | None) -> None:
        self.set_active_person_id(person_id)

    def set_active_person_id(self, person_id: int | None) -> None:
        self.__active_person_id = person_id
        for pid, b in self._buttons.items():
            b.setChecked(pid == person_id)
        btn = self._buttons.get(person_id)
        if btn is not None:
            self.scroll_area.ensureWidgetVisible(btn, 50, 0)

    def set_dark_mode(self, is_dark: bool) -> None:
        self._is_dark = bool(is_dark)
        self._apply_theme()

    def _apply_theme(self) -> None:
        if self._is_dark:
            self.setStyleSheet("background-color: #171A20; border-bottom: 1px solid #2D323B;")
            self.scroll_area.setStyleSheet("background: transparent; border: none;")
            self.scroll_content.setStyleSheet("background: transparent;")
            chev_style = (
                "QToolButton { border: 1px solid #374151; border-radius: 13px; background-color: #1F2937; "
                "color: #D1D5DB; font-size: 16px; font-weight: bold; padding-bottom: 2px; } "
                "QToolButton:hover { background-color: #2D333E; border-color: #4B5563; color: #FFFFFF; } "
                "QToolButton:pressed { background-color: #374151; } "
                "QToolButton:disabled { border-color: #262B34; color: #4B5563; background-color: #1A1D23; }"
            )
        else:
            self.setStyleSheet("background-color: #F8FAFC; border-bottom: 1px solid #D9E2EC;")
            self.scroll_area.setStyleSheet("background: transparent; border: none;")
            self.scroll_content.setStyleSheet("background: transparent;")
            chev_style = (
                "QToolButton { border: 1px solid #CBD5E1; border-radius: 13px; background-color: #FFFFFF; "
                "color: #475569; font-size: 16px; font-weight: bold; padding-bottom: 2px; } "
                "QToolButton:hover { background-color: #F1F5F9; border-color: #94A3B8; color: #0F172A; } "
                "QToolButton:pressed { background-color: #E2E8F0; } "
                "QToolButton:disabled { border-color: #E2E8F0; color: #CBD5E1; background-color: #F8FAFC; }"
            )
        self.btn_left.setStyleSheet(chev_style)
        self.btn_right.setStyleSheet(chev_style)
        for btn_key, btn in self._buttons.items():
            btn.setStyleSheet(self._button_style(has_icon=(btn_key is not None)))
        self._update_scroll_buttons()

    def _button_style(self, has_icon: bool) -> str:
        padding = "padding: 0px 10px 0px 6px;" if has_icon else "padding: 0px 12px;"
        if self._is_dark:
            return (
                f"QPushButton {{ border: 1px solid #374151; border-radius: 16px; {padding} "
                "background-color: #1F2937; color: #E5E7EB; font-size: 11px; font-weight: 500; text-align: left; } "
                "QPushButton:hover { background-color: #2D333E; border-color: #4B5563; } "
                "QPushButton:checked { background-color: #2563EB; color: #FFFFFF; border-color: #2563EB; font-weight: bold; } "
                "QPushButton:checked:hover { background-color: #1D4ED8; border-color: #1D4ED8; }"
            )
        else:
            return (
                f"QPushButton {{ border: 1px solid #CBD5E1; border-radius: 16px; {padding} "
                "background-color: #FFFFFF; color: #1E293B; font-size: 11px; font-weight: 500; text-align: left; } "
                "QPushButton:hover { background-color: #F1F5F9; border-color: #94A3B8; } "
                "QPushButton:checked { background-color: #0078D4; color: #FFFFFF; border-color: #0078D4; font-weight: bold; } "
                "QPushButton:checked:hover { background-color: #0067B8; border-color: #0067B8; }"
            )

    def eventFilter(self, watched: Any, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Wheel:
            if isinstance(event, QWheelEvent):
                delta = event.pixelDelta().x() or event.pixelDelta().y()
                if delta == 0:
                    delta = event.angleDelta().x() or event.angleDelta().y()
                if delta != 0:
                    sb = self.scroll_area.horizontalScrollBar()
                    step = int(delta * 1.5) if abs(delta) > 30 else (120 if delta < 0 else -120)
                    sb.setValue(sb.value() - step)
                    return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_scroll_buttons()

    def _scroll_by(self, delta: int) -> None:
        sb = self.scroll_area.horizontalScrollBar()
        sb.setValue(sb.value() + delta)

    def _update_scroll_buttons(self) -> None:
        sb = self.scroll_area.horizontalScrollBar()
        max_val = sb.maximum()
        if max_val <= 0:
            self.btn_left.setVisible(False)
            self.btn_right.setVisible(False)
        else:
            self.btn_left.setVisible(True)
            self.btn_right.setVisible(True)
            self.btn_left.setEnabled(sb.value() > 0)
            self.btn_right.setEnabled(sb.value() < max_val)

    def populate(
        self,
        people_counts: list[tuple[int | None, str, int, bytes | None]],
        mode: str = "suggestions",
    ) -> None:
        """
        Populates avatar buttons: list of (person_id, name, count, thumb_blob).
        person_id=None represents 'All Suggestions' / 'All Flagged'.
        """
        while self._strip_layout.count() > 0:
            item = self._strip_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._buttons.clear()

        # All button
        total_count = sum(cnt for _pid, _name, cnt, _blob in people_counts)
        if mode == "outliers":
            all_text = f"All Flagged ({total_count})"
        elif mode == "boxless":
            all_text = f"All Missing Boxes ({total_count})"
        else:
            all_text = f"All Suggestions ({total_count})"
        btn_all = QPushButton(all_text, self.scroll_content)
        btn_all.setFixedHeight(32)
        btn_all.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_all.setCheckable(True)
        btn_all.setChecked(self.__active_person_id is None)
        btn_all.setToolTip(f"Show all face suggestions ({total_count} total)")

        fm_all = btn_all.fontMetrics()
        all_w = max(115, fm_all.horizontalAdvance(all_text) + 26)
        btn_all.setFixedWidth(all_w)
        btn_all.setStyleSheet(self._button_style(has_icon=False))
        btn_all.clicked.connect(lambda: self._on_clicked(None))
        self._strip_layout.addWidget(btn_all)
        self._buttons[None] = btn_all

        for pid, name, count, thumb_blob in people_counts:
            if count <= 0:
                continue
            btn_key = -1 if pid is None else pid
            name_clean = (str(name) if name is not None else "").strip() or "Unnamed"
            display_name = name_clean if len(name_clean) <= 26 else name_clean[:24] + "…"
            btn_text = f"{display_name} ({count})"
            btn = QPushButton(btn_text, self.scroll_content)
            btn.setFixedHeight(32)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setCheckable(True)
            btn.setChecked(self.__active_person_id == btn_key)
            btn.setToolTip(f"{name_clean} ({count} suggestion{'s' if count != 1 else ''})")

            # High-res circular avatar or color-coded initials fallback
            pm = make_circular_avatar(thumb_blob, name_clean, size=24)
            has_avatar = False
            if pm and not pm.isNull():
                btn.setIcon(QIcon(pm))
                btn.setIconSize(QSize(22, 22))
                has_avatar = True

            fm = btn.fontMetrics()
            text_w = fm.horizontalAdvance(btn_text)
            needed_w = max(70, text_w + (36 if has_avatar else 24))
            btn.setFixedWidth(needed_w)

            btn.setStyleSheet(self._button_style(has_icon=has_avatar))
            btn.clicked.connect(lambda checked=False, p=btn_key: self._on_clicked(p))
            if pid is not None:
                btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                btn.customContextMenuRequested.connect(
                    lambda pos, p=pid, n=name_clean, b=btn: self._show_btn_context_menu(pos, p, n, b)
                )
            self._strip_layout.addWidget(btn)
            self._buttons[btn_key] = btn

        self._strip_layout.addStretch(1)

        # Enforce minimum width on scroll_content so QScrollArea never crushes buttons
        spacing = self._strip_layout.spacing()
        total_w = sum(b.width() for b in self._buttons.values()) + max(0, len(self._buttons) - 1) * spacing + 32
        self.scroll_content.setMinimumWidth(total_w)
        self.scroll_content.adjustSize()

        active_btn = self._buttons.get(self.__active_person_id)
        if active_btn is not None:
            active_btn.setChecked(True)
            QTimer.singleShot(0, lambda: self.scroll_area.ensureWidgetVisible(active_btn, 50, 0))

        QTimer.singleShot(0, self._update_scroll_buttons)

    def _show_btn_context_menu(self, pos: QPoint, person_id: int, person_name: str, btn: QWidget) -> None:
        menu = QMenu(self)
        act_rename = menu.addAction(f"✏ Rename or merge '{person_name}'...")
        act = menu.exec(btn.mapToGlobal(pos))
        if act == act_rename:
            self.rename_merge_requested.emit(person_id, person_name)

    def _on_clicked(self, person_id: int | None) -> None:
        self.set_active_person_id(person_id)
        self.person_selected.emit(person_id)


class BatchPeopleView(QWidget):
    """
    Virtualised 'Batch people tag' Confirmation View ('SPEC.md' Section 5).
    Provides instant multi-selection face confirmation, rejection, naming,
    quick context preview, and background centroid updating.
    """

    faces_confirmed = Signal(list, int)        # (list[int] face_ids, int person_id)
    faces_rejected = Signal(list)              # (list[int] face_ids)
    faces_ignored = Signal(list)               # (list[int] face_ids)
    person_renamed = Signal(int, str)          # (int person_id, str new_name)
    back_to_gallery_requested = Signal()
    open_media_requested = Signal(int)         # (media_id)
    card_selection_changed = Signal(str, object, str)  # (mode, swap_suggested_name, person_name)

    def __init__(self, engine: DualDatabaseEngine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.repo = CatalogRepository(engine)
        self.thumb_repo = ThumbnailRepository(engine)
        self.pixmap_cache = LruCache(5000)

        from PySide6.QtCore import QSettings
        settings = QSettings("PhotoGallery64", "PhotoGallery64")
        self._min_confidence_threshold: float = float(settings.value("face_match_threshold", 0.52))

        self._sections: dict[int | None, _PersonGroupSectionWidget] = {}
        self._all_records: list[FaceCardRecord] = []
        self._filter_person_id: int | None = None
        self._current_view_mode: str = "suggestions"
        self._scan_current: int = 0
        self._scan_total: int = 0

        # Debounce timer to prevent multiple rapid re-clustering passes
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(250)
        self._refresh_timer.timeout.connect(self._do_refresh_suggestions)

        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---------------------------------------------------------------------
        # 1. Top Action Ribbon Bar
        # ---------------------------------------------------------------------
        self.action_bar = QFrame(self)
        self.action_bar.setFixedHeight(48)
        self.action_bar.setStyleSheet("background-color: #F0F4F8; border-bottom: 1px solid #D0DBE5;")
        h_bar = QHBoxLayout(self.action_bar)
        h_bar.setContentsMargins(12, 6, 12, 6)
        h_bar.setSpacing(8)

        self.btn_select_all = QPushButton("Select all (Ctrl+A)", self.action_bar)
        self.btn_select_all.setStyleSheet("padding: 5px 10px; font-size: 11px;")
        self.btn_select_all.clicked.connect(self.select_all)
        h_bar.addWidget(self.btn_select_all)

        self.btn_confirm = QPushButton("✓ Confirm (Enter)", self.action_bar)
        self.btn_confirm.setStyleSheet(
            "QPushButton { background-color: #2E8B57; color: white; font-weight: bold; padding: 5px 12px; border-radius: 3px; border: 1px solid #236B42; }"
            "QPushButton:hover { background-color: #3CB371; }"
        )
        self.btn_confirm.clicked.connect(self.confirm_selected)
        h_bar.addWidget(self.btn_confirm)

        self.btn_reject = QPushButton("✕ Not this person (Del)", self.action_bar)
        self.btn_reject.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; color: #D32F2F; border: 1px solid #D32F2F; padding: 5px 10px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #FFEBEE; }"
        )
        self.btn_reject.clicked.connect(self.reject_selected)
        h_bar.addWidget(self.btn_reject)

        self.btn_tag_as = QPushButton("Tag as...", self.action_bar)
        self.btn_tag_as.setStyleSheet("padding: 5px 10px; font-size: 11px;")
        self.btn_tag_as.clicked.connect(self._on_tag_as_clicked)
        h_bar.addWidget(self.btn_tag_as)

        self.btn_ignore = QPushButton("Don't show again (I)", self.action_bar)
        self.btn_ignore.setStyleSheet("padding: 5px 10px; font-size: 11px;")
        self.btn_ignore.clicked.connect(self.ignore_selected)
        h_bar.addWidget(self.btn_ignore)

        self.btn_toggle_tagged = QPushButton("Show tagged faces", self.action_bar)
        self.btn_toggle_tagged.setCheckable(True)
        self.btn_toggle_tagged.setChecked(False)
        self.btn_toggle_tagged.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; border: 1px solid #B0C4DE; padding: 5px 10px; border-radius: 3px; font-size: 11px; }"
            "QPushButton:checked { background-color: #1E395B; color: white; border-color: #1E395B; font-weight: bold; }"
        )
        self.btn_toggle_tagged.clicked.connect(self._on_toggle_tagged_clicked)
        h_bar.addWidget(self.btn_toggle_tagged)

        self.btn_face_options = QPushButton("⚙ Face Options / Rebuild...", self.action_bar)
        self.btn_face_options.setStyleSheet("padding: 5px 10px; font-size: 11px;")
        self.btn_face_options.clicked.connect(self._open_face_options_dialog)
        h_bar.addWidget(self.btn_face_options)

        h_bar.addStretch(1)

        self.btn_back = QPushButton("✕ Back to Gallery (Esc)", self.action_bar)
        self.btn_back.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; font-weight: bold; border: 1px solid #B0C4DE; padding: 5px 12px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #E2E8F0; }"
        )
        self.btn_back.clicked.connect(self.back_to_gallery_requested.emit)
        h_bar.addWidget(self.btn_back)

        main_layout.addWidget(self.action_bar)

        # ---------------------------------------------------------------------
        # 2. Dismissible Information Banner ('SPEC.md' Section 5.2)
        # ---------------------------------------------------------------------
        self.banner = QFrame(self)
        self.banner.setStyleSheet(
            "QFrame { background-color: #FFF9DB; border-bottom: 1px solid #F5E79E; padding: 4px; }"
        )
        h_banner = QHBoxLayout(self.banner)
        h_banner.setContentsMargins(14, 6, 14, 6)
        h_banner.setSpacing(12)
        self.lbl_banner_text = QLabel(
            "ℹ  As you act on these suggestions, the faces will be removed from the list below, and the confirmed faces will be tagged in your photos.",
            self.banner,
        )
        self.lbl_banner_text.setStyleSheet("color: #665200; font-size: 11px;")
        h_banner.addWidget(self.lbl_banner_text)

        self.chk_include_no_face = QCheckBox("Show tags without detected face (side profiles, occluded)", self.banner)
        self.chk_include_no_face.setStyleSheet("color: #665200; font-size: 11px;")
        self.chk_include_no_face.setChecked(False)
        self.chk_include_no_face.setVisible(False)
        self.chk_include_no_face.toggled.connect(lambda: self.refresh_suggestions(debounce=False))
        h_banner.addWidget(self.chk_include_no_face)

        h_banner.addStretch(1)

        btn_dismiss_banner = QToolButton(self.banner)
        btn_dismiss_banner.setText("✕")
        btn_dismiss_banner.setStyleSheet("border: none; color: #887000; font-size: 11px;")
        btn_dismiss_banner.clicked.connect(lambda: self.banner.setVisible(False))
        h_banner.addWidget(btn_dismiss_banner)

        main_layout.addWidget(self.banner)

        # ---------------------------------------------------------------------
        # 3. People Avatar Picker Strip ('SPEC.md' Section 5.1)
        # ---------------------------------------------------------------------
        self.avatar_strip = _PeopleAvatarPickerStrip(self)
        self.avatar_strip.person_selected.connect(self._on_avatar_strip_filtered)
        self.avatar_strip.rename_merge_requested.connect(self._on_rename_merge_person)
        main_layout.addWidget(self.avatar_strip)

        # ---------------------------------------------------------------------
        # 4. Scrollable Container for Group Sections
        # ---------------------------------------------------------------------
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet(
            """
            QScrollArea {
                border: none;
                background-color: #FFFFFF;
            }
            QScrollBar:vertical {
                border: none;
                background: transparent;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #CBD5E1;
                min-height: 28px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #94A3B8;
            }
            QScrollBar::handle:vertical:pressed {
                background: #64748B;
            }
            QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {
                border: none;
                background: transparent;
                height: 0px;
                width: 0px;
            }
            QScrollBar::sub-page:vertical, QScrollBar::add-page:vertical {
                border: none;
                background: transparent;
            }
            QScrollBar:horizontal {
                border: none;
                background: transparent;
                height: 0px;
                width: 0px;
            }
            QAbstractScrollArea::corner {
                background: transparent;
                border: none;
            }
            """
        )

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(16, 12, 16, 16)
        self.scroll_layout.setSpacing(12)

        self.lbl_empty_state = QLabel(self.scroll_content)
        self.lbl_empty_state.setAlignment(Qt.AlignCenter)
        self.lbl_empty_state.setStyleSheet(
            "color: #4A6585; font-family: 'Segoe UI'; font-size: 11pt; padding: 48px 24px;"
        )
        self.lbl_empty_state.setVisible(False)
        self.scroll_layout.addWidget(self.lbl_empty_state)
        self.scroll_layout.addStretch(1)

        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area, 1)

    def set_dark_mode(self, is_dark: bool) -> None:
        """Applies Dark Studio or standard light styling to BatchPeopleView."""
        self._is_dark = bool(is_dark)
        if is_dark:
            self.action_bar.setStyleSheet("background-color: #1B1E24; border-bottom: 1px solid #2D323B;")
            self.avatar_strip.set_dark_mode(True)
            self.scroll_area.setStyleSheet(
                """
                QScrollArea {
                    border: none;
                    background-color: #121417;
                }
                QScrollBar:vertical {
                    border: none;
                    background: transparent;
                    width: 8px;
                    margin: 0px;
                }
                QScrollBar::handle:vertical {
                    background: #3E4756;
                    min-height: 28px;
                    border-radius: 4px;
                }
                """
            )
            self.scroll_content.setStyleSheet("background-color: #121417;")
            self.lbl_empty_state.setStyleSheet(
                "color: #94A3B8; font-family: 'Segoe UI'; font-size: 11pt; padding: 48px 24px;"
            )
        else:
            self.action_bar.setStyleSheet("background-color: #F0F4F8; border-bottom: 1px solid #D0DBE5;")
            self.avatar_strip.set_dark_mode(False)
            self.scroll_area.setStyleSheet(
                """
                QScrollArea {
                    border: none;
                    background-color: #FFFFFF;
                }
                QScrollBar:vertical {
                    border: none;
                    background: transparent;
                    width: 8px;
                    margin: 0px;
                }
                QScrollBar::handle:vertical {
                    background: #CBD5E1;
                    min-height: 28px;
                    border-radius: 4px;
                }
                """
            )
            self.scroll_content.setStyleSheet("")
            self.lbl_empty_state.setStyleSheet(
                "color: #4A6585; font-family: 'Segoe UI'; font-size: 11pt; padding: 48px 24px;"
            )

    def set_scan_progress(self, current: int, total: int) -> None:
        """Updates live face detection progress indicator in BatchPeopleView."""
        self._scan_current = current
        self._scan_total = total
        self._update_empty_state_label()

    def filter_to_person(self, person_id: int | None) -> None:
        """Filters BatchPeopleView to a specific person_id (or None for All)."""
        self.avatar_strip._active_person_id = person_id
        self._filter_person_id = person_id
        self.refresh_suggestions()

    def set_use_ribbon_mode(self, use_ribbon: bool) -> None:
        """Shows or hides the internal action bar when ribbon has a dedicated contextual tab."""
        self._use_ribbon_mode = bool(use_ribbon)
        if hasattr(self, "action_bar"):
            self.action_bar.setVisible(not use_ribbon)

    def set_view_mode(self, mode: str) -> None:
        """Sets active view mode: 'suggestions', 'outliers', or 'boxless'."""
        if mode not in ("suggestions", "outliers", "boxless"):
            mode = "suggestions"
        if getattr(self, "_current_view_mode", "suggestions") == mode:
            return
        self._current_view_mode = mode
        if hasattr(self, "lbl_banner_text") and self.lbl_banner_text is not None:
            if mode == "outliers":
                self.lbl_banner_text.setText(
                    "⚠  Review Potential Outliers: These faces were tagged with a person but have low visual similarity to that person's other photos or closely resemble someone else. Confirm to keep or reject to unassign."
                )
            elif mode == "boxless":
                self.lbl_banner_text.setText(
                    "🔗  Missing Face Boxes: These photos have person tags that lack bounding boxes, and an unassigned face box was detected in the photo. Confirm to link the box to the person."
                )
            else:
                self.lbl_banner_text.setText(
                    "ℹ  As you act on these suggestions, the faces will be removed from the list below, and the confirmed faces will be tagged in your photos."
                )
        if hasattr(self, "chk_include_no_face") and self.chk_include_no_face is not None:
            self.chk_include_no_face.setVisible(mode == "outliers")
        if hasattr(self, "btn_confirm"):
            if mode == "outliers":
                self.btn_confirm.setText("✓ Keep (Enter)")
                self.btn_reject.setText("✕ Remove Tag (Del)")
            elif mode == "boxless":
                self.btn_confirm.setText("✓ Link Box (Enter)")
                self.btn_reject.setText("✕ Skip (Del)")
            else:
                self.btn_confirm.setText("✓ Confirm (Enter)")
                self.btn_reject.setText("✕ Not this person (Del)")
        self.refresh_suggestions()
        self._on_selection_changed()

    def _update_empty_state_label(self) -> None:
        has_visible_sections = any(not sec.isHidden() for sec in self._sections.values())
        if has_visible_sections:
            self.lbl_empty_state.setVisible(False)
            return
        self.lbl_empty_state.setVisible(True)
        if self._current_view_mode == "outliers":
            self.lbl_empty_state.setText(
                "✓ No face tag outliers found!\n\nAll confirmed face tags match their person models closely."
            )
        elif self._current_view_mode == "boxless":
            self.lbl_empty_state.setText(
                "✓ No missing face boxes found!\n\nAll person tags either have bounding boxes or have no unassigned detected faces."
            )
        elif self._scan_total > 0 and self._scan_current < self._scan_total:
            self.lbl_empty_state.setText(
                f"Finding people in your photos... ({self._scan_current:,} of {self._scan_total:,} photos scanned)\n\n"
                "Detected face suggestions will appear here automatically as scanning progresses."
            )
        else:
            self.lbl_empty_state.setText(
                "No pending face suggestions right now.\n\n"
                "Tip: Tag a few faces manually in your photos using 'People tag', or make sure 'Find people in your photos' is enabled in Options."
            )

    def _on_toggle_tagged_clicked(self) -> None:
        self._show_tagged_faces = self.btn_toggle_tagged.isChecked()
        self.btn_toggle_tagged.setText("Hide tagged faces" if self._show_tagged_faces else "Show tagged faces")
        self.refresh_suggestions()

    # -------------------------------------------------------------------------
    # Data Loading & Section Population
    # -------------------------------------------------------------------------
    def refresh_suggestions(self, debounce: bool = False) -> None:
        """Reloads unassigned face suggestions from catalog.db and updates UI (optionally debounced)."""
        if debounce:
            self._refresh_timer.start()
        else:
            self._refresh_timer.stop()
            self._do_refresh_suggestions()

    def _do_refresh_suggestions(self) -> None:
        """Internal execution of suggestion reload."""
        mode = getattr(self, "_current_view_mode", "suggestions")
        if mode == "outliers":
            target_pid = None if (self._filter_person_id is None or self._filter_person_id == -1) else self._filter_person_id
            inc_no_face = getattr(self, "chk_include_no_face", None) is not None and self.chk_include_no_face.isChecked()
            raw_rows = self.repo.get_face_tag_outliers(
                min_similarity_threshold=0.38,
                limit=1000,
                include_no_face_boxes=inc_no_face,
            )
            if target_pid is not None:
                raw_rows = [r for r in raw_rows if r.get("person_id") == target_pid]
            self._all_records = [
                FaceCardRecord(
                    face_id=int(r["id"]),
                    media_id=int(r["media_id"]),
                    suggested_person_id=int(r.get("person_id")) if r.get("person_id") is not None else None,
                    suggested_name=str(r.get("person_name") or ""),
                    confidence=float(r.get("confidence") or 0.0),
                    rect_x=float(r["rect_x"]),
                    rect_y=float(r["rect_y"]),
                    rect_w=float(r["rect_w"]),
                    rect_h=float(r["rect_h"]),
                    filename=str(r["filename"]),
                    folder_path=str(r["folder_path"]),
                    is_confirmed=True,
                    outlier_reason=str(r.get("outlier_reason") or "Outlier"),
                    swap_person_id=int(r["swap_person_id"]) if r.get("swap_person_id") else None,
                    swap_person_name=str(r["swap_person_name"]) if r.get("swap_person_name") else None,
                )
                for r in raw_rows
            ]
        elif mode == "boxless":
            raw_rows = self.repo.get_boxless_people_matches(limit=1000)
            if self._filter_person_id is not None and self._filter_person_id != -1:
                raw_rows = [r for r in raw_rows if r.get("person_id") == self._filter_person_id]
            self._all_records = [
                FaceCardRecord(
                    face_id=int(r["face_id"]),
                    media_id=int(r["media_id"]),
                    suggested_person_id=int(r["person_id"]),
                    suggested_name=str(r["person_name"]),
                    confidence=float(r.get("confidence") or 0.88),
                    rect_x=float(r["rect_x"]),
                    rect_y=float(r["rect_y"]),
                    rect_w=float(r["rect_w"]),
                    rect_h=float(r["rect_h"]),
                    filename=str(r["filename"]),
                    folder_path=str(r["folder_path"]),
                    is_confirmed=False,
                    boxless_id=int(r["boxless_id"]),
                    is_boxless_match=True,
                )
                for r in raw_rows
            ]
        else:
            conf_thresh = getattr(self, "_min_confidence_threshold", 0.52)
            raw_rows = self.repo.get_pending_face_suggestions(limit=5000, min_confidence=conf_thresh, include_unnamed=True)
            if getattr(self, "_show_tagged_faces", False):
                target_pid = None if (self._filter_person_id is None or self._filter_person_id == -1) else self._filter_person_id
                confirmed_rows = self.repo.get_confirmed_faces(limit=5000, person_id=target_pid)
            else:
                confirmed_rows = []

            all_raw = raw_rows + confirmed_rows

            self._all_records = [
                FaceCardRecord(
                    face_id=int(r["id"]),
                    media_id=int(r["media_id"]),
                    suggested_person_id=int(r["suggested_person_id"]) if r.get("suggested_person_id") is not None else None,
                    suggested_name=str(r["suggested_name"]) if r.get("suggested_name") else None,
                    confidence=float(r.get("confidence") or 0.0),
                    rect_x=float(r["rect_x"]),
                    rect_y=float(r["rect_y"]),
                    rect_w=float(r["rect_w"]),
                    rect_h=float(r["rect_h"]),
                    filename=str(r["filename"]),
                    folder_path=str(r["folder_path"]),
                    is_confirmed=bool(r.get("is_confirmed", False)),
                )
                for r in all_raw
            ]

        # Group records by suggested_person_id
        grouped: dict[int | None, list[FaceCardRecord]] = {}
        person_names: dict[int, str] = {}
        person_thumbs: dict[int, bytes | None] = {}

        for rec in self._all_records:
            pid = rec.suggested_person_id
            if pid not in grouped:
                grouped[pid] = []
            grouped[pid].append(rec)
            if pid is not None and rec.suggested_name:
                person_names[pid] = rec.suggested_name

        # Query confirmed counts and avatar blobs for each person
        conn = self.engine.catalog_conn()
        confirmed_counts: dict[int, int] = {}
        if person_names:
            p_rows = conn.execute(
                f"SELECT id, name, confirmed_count, thumbnail_blob FROM people WHERE id IN ({','.join('?' for _ in person_names)});"
                , list(person_names.keys())
            ).fetchall()
            for r in p_rows:
                confirmed_counts[int(r["id"])] = int(r["confirmed_count"] or 0)
                person_thumbs[int(r["id"])] = bytes(r["thumbnail_blob"]) if r["thumbnail_blob"] else None

        # Populate Avatar Picker Strip
        picker_data: list[tuple[int | None, str, int, bytes | None]] = []
        for pid, p_name in sorted(person_names.items(), key=lambda x: x[1].lower()):
            if mode == "outliers":
                cnt = len([r for r in grouped.get(pid, []) if getattr(r, "outlier_reason", None)])
            elif mode == "boxless":
                cnt = len([r for r in grouped.get(pid, []) if getattr(r, "is_boxless_match", False)])
            else:
                cnt = len([r for r in grouped.get(pid, []) if not r.is_confirmed])
            picker_data.append((pid, p_name, cnt, person_thumbs.get(pid)))

        if None in grouped and grouped[None]:
            unnamed_cnt = len([r for r in grouped[None] if not r.is_confirmed])
            picker_data.append((None, "Unnamed", unnamed_cnt, None))

        self.avatar_strip.populate(picker_data, mode=mode)

        # Clear existing section widgets
        for sec in self._sections.values():
            sec.deleteLater()
        self._sections.clear()

        # 1. Populate Named Person Sections
        named_grouped = {pid: recs for pid, recs in grouped.items() if pid is not None}
        for pid, records in named_grouped.items():
            if self._filter_person_id is not None:
                target_pid = None if self._filter_person_id == -1 else self._filter_person_id
                if pid != target_pid:
                    continue

            p_name = person_names.get(pid, "Unnamed")
            conf_cnt = confirmed_counts.get(pid, 0)

            sec = _PersonGroupSectionWidget(
                person_id=pid,
                person_name=p_name,
                confirmed_count=conf_cnt,
                records=records,
                thumb_repo=self.thumb_repo,
                pixmap_cache=self.pixmap_cache,
                parent=self.scroll_content,
            )
            sec.confirm_all_requested.connect(self._on_section_confirm_all)
            sec.confirm_faces_requested.connect(self._on_section_confirm_faces)
            sec.reject_faces_requested.connect(self._on_section_reject_faces)
            sec.not_a_face_requested.connect(self._on_section_not_a_face)
            sec.ignore_faces_requested.connect(self._on_section_ignore_faces)
            sec.tag_as_requested.connect(self._on_section_tag_as)
            sec.rename_merge_requested.connect(self._on_rename_merge_person)
            sec.open_media_requested.connect(self.open_media_requested.emit)
            sec.preview_requested.connect(self._open_quick_preview)
            sec.list_view.selectionModel().selectionChanged.connect(lambda *args: self._on_selection_changed())

            idx = max(0, self.scroll_layout.count() - 1)
            self.scroll_layout.insertWidget(idx, sec)
            sec.show()
            self._sections[pid] = sec

        # 2. Cluster and Populate Unnamed Face Sections
        unnamed_records = grouped.get(None, [])
        if unnamed_records and (self._filter_person_id is None or self._filter_person_id == -1):
            unnamed_fids = [r.face_id for r in unnamed_records]
            fid_to_rec = {r.face_id: r for r in unnamed_records}
            valid_fids: list[int] = []
            emb_list: list[np.ndarray] = []

            for chunk_start in range(0, len(unnamed_fids), 500):
                chunk = unnamed_fids[chunk_start : chunk_start + 500]
                rows = conn.execute(
                    f"SELECT id, embedding FROM media_people WHERE id IN ({','.join('?' for _ in chunk)}) AND embedding IS NOT NULL;",
                    chunk,
                ).fetchall()
                for row in rows:
                    raw_emb = bytes(row["embedding"]) if row["embedding"] else None
                    if raw_emb and len(raw_emb) >= 2048:
                        valid_fids.append(int(row["id"]))
                        emb_list.append(deserialize_embedding(raw_emb))

            unnamed_sections: list[tuple[str, list[FaceCardRecord]]] = []
            if len(valid_fids) >= 3 and len(emb_list) >= 3:
                clusters = FaceVectorMatcher.cluster_unnamed_faces(
                    valid_fids, np.stack(emb_list, axis=0), sim_threshold=0.58, min_cluster_size=4
                )
                assigned_unnamed_fids = set()
                # Sort clusters by face count (largest first) and keep top 20 significant clusters
                sorted_clusters = sorted(
                    [(k, v) for k, v in clusters.items() if k != "Other Unnamed Faces"],
                    key=lambda x: len(x[1]),
                    reverse=True,
                )
                for idx, (_, c_fids) in enumerate(sorted_clusters[:20], 1):
                    c_recs = [fid_to_rec[fid] for fid in c_fids if fid in fid_to_rec]
                    assigned_unnamed_fids.update(c_fids)
                    if c_recs:
                        unnamed_sections.append((f"Unknown Person #{idx}", c_recs))

                remaining_fids = [fid for fid in unnamed_fids if fid not in assigned_unnamed_fids]
                if remaining_fids:
                    rem_recs = [fid_to_rec[fid] for fid in remaining_fids]
                    unnamed_sections.append(("Other Unnamed Faces", rem_recs))
            else:
                unnamed_sections.append(("Other Unnamed Faces", unnamed_records))

            for cluster_name, records in unnamed_sections:
                sec_key = None if cluster_name in ("Unnamed Faces", "Other Unnamed Faces") else cluster_name
                sec = _PersonGroupSectionWidget(
                    person_id=None,
                    person_name=cluster_name,
                    confirmed_count=0,
                    records=records,
                    thumb_repo=self.thumb_repo,
                    pixmap_cache=self.pixmap_cache,
                    parent=self.scroll_content,
                )
                sec.name_cluster_requested.connect(self._on_section_name_cluster)
                sec.confirm_all_requested.connect(self._on_section_confirm_all)
                sec.confirm_faces_requested.connect(self._on_section_confirm_faces)
                sec.reject_faces_requested.connect(self._on_section_reject_faces)
                sec.not_a_face_requested.connect(self._on_section_not_a_face)
                sec.ignore_faces_requested.connect(self._on_section_ignore_faces)
                sec.tag_as_requested.connect(self._on_section_tag_as)
                sec.open_media_requested.connect(self.open_media_requested.emit)
                sec.preview_requested.connect(self._open_quick_preview)
                sec.list_view.selectionModel().selectionChanged.connect(lambda *args: self._on_selection_changed())

                idx = max(0, self.scroll_layout.count() - 1)
                self.scroll_layout.insertWidget(idx, sec)
                sec.show()
                self._sections[sec_key] = sec

        self._update_empty_state_label()

    def _on_avatar_strip_filtered(self, person_id: int | None) -> None:
        """Filters section visibility when a chip is clicked on the avatar strip."""
        self._filter_person_id = person_id
        for sec in self._sections.values():
            if person_id is None:
                sec.setVisible(True)
            elif person_id == -1:
                sec.setVisible(sec.person_id is None)
            else:
                sec.setVisible(sec.person_id == person_id)
        self._update_empty_state_label()

    def _on_rename_merge_person(self, person_id: int, current_name: str) -> None:
        """Opens MergePeopleDialog to rename a contact with suggestions or merge into another."""
        from src.ui.people_directory_view import MergePeopleDialog

        try:
            all_named = self.repo.get_people_list()
        except Exception:
            all_named = []

        candidates = [p for p in all_named if int(p["id"]) != int(person_id)]
        src_info = next((p for p in all_named if int(p["id"]) == int(person_id)), None)
        src_thumb = src_info.get("thumbnail_blob") if src_info else None
        src_cnt = src_info.get("photo_count", 0) if src_info else 0

        dlg = MergePeopleDialog(
            source_person_name=current_name,
            candidates=candidates,
            source_person_id=person_id,
            source_photo_count=src_cnt,
            source_thumb_blob=src_thumb,
            parent=self,
        )
        if dlg.exec() == QDialog.Accepted:
            if dlg.is_rename():
                target_name = dlg.target_person_name()
                try:
                    actual_id = self.repo.rename_person(person_id, target_name)
                    self.refresh_suggestions()
                    self.person_renamed.emit(actual_id, target_name)
                except Exception as exc:
                    logger.warning("Failed to rename person: %s", exc)
            else:
                dst_id = dlg.target_person_id()
                if dst_id is not None:
                    try:
                        self.repo.merge_people(person_id, dst_id)
                        self.refresh_suggestions()
                        self.person_renamed.emit(dst_id, "")
                    except Exception as exc:
                        logger.warning("Failed to merge people: %s", exc)

    # -------------------------------------------------------------------------
    # Batch Actions: Confirm, Reject, Ignore, Tag As
    # -------------------------------------------------------------------------
    def get_all_selected_records(self) -> list[FaceCardRecord]:
        """Gathers selected FaceCardRecords across all visible sections."""
        selected: list[FaceCardRecord] = []
        for sec in self._sections.values():
            if not sec.isHidden():
                selected.extend(sec.get_selected_records())
        return selected

    def select_all(self) -> None:
        """Selects all face cards across all visible sections (Ctrl+A)."""
        for sec in self._sections.values():
            if not sec.isHidden():
                sec.select_all()

    def confirm_selected(self) -> None:
        """Confirms all currently selected face suggestions (Enter key)."""
        selected = self.get_all_selected_records()
        if not selected:
            # Fallback: confirm first visible section's suggestions
            for sec in self._sections.values():
                if not sec.isHidden() and sec.person_id is not None:
                    sec._on_confirm_all_clicked()
                    return
            return

        # Check if any cards are outliers (Review Outliers mode)
        outlier_cards = [r for r in selected if getattr(r, "outlier_reason", None)]
        for r in outlier_cards:
            try:
                self.repo.verify_face_tag(r.face_id)
            except Exception as exc:
                logger.warning("Failed to verify outlier tag: %s", exc)

        if outlier_cards:
            outlier_fids = {r.face_id for r in outlier_cards}
            self._remove_faces_from_view(outlier_fids)
            self.refresh_suggestions()
            return

        # Check if any cards are boxless matches
        boxless_cards = [r for r in selected if getattr(r, "is_boxless_match", False)]
        for r in boxless_cards:
            if r.suggested_person_id:
                try:
                    self.repo.assign_face_to_person(r.face_id, r.suggested_person_id)
                except Exception as exc:
                    logger.warning("Failed to link boxless face: %s", exc)

        normal_cards = [r for r in selected if not getattr(r, "is_boxless_match", False)]
        # Group by suggested person
        by_person: dict[int, list[int]] = {}
        unnamed_ids: list[int] = []
        for r in normal_cards:
            if r.suggested_person_id:
                by_person.setdefault(r.suggested_person_id, []).append(r.face_id)
            else:
                unnamed_ids.append(r.face_id)

        for pid, fids in by_person.items():
            self._confirm_and_update_centroid(pid, fids)

        if unnamed_ids:
            self._prompt_tag_as(unnamed_ids)

        if boxless_cards:
            boxless_fids = {r.face_id for r in boxless_cards}
            self._remove_faces_from_view(boxless_fids)
            self.faces_confirmed.emit(list(boxless_fids), 0)
            self.refresh_suggestions()

    def reject_selected(self) -> None:
        """Rejects all currently selected face suggestions or unassigns outliers (Delete / Backspace)."""
        selected = self.get_all_selected_records()
        if not selected:
            return

        outlier_cards = [r for r in selected if getattr(r, "outlier_reason", None)]
        for r in outlier_cards:
            try:
                self.repo.unassign_face_person(r.face_id, r.suggested_person_id)
            except Exception:
                self.repo.reject_face_suggestion(r.face_id, r.suggested_person_id)

        normal_cards = [r for r in selected if not getattr(r, "outlier_reason", None)]
        for r in normal_cards:
            pid = r.suggested_person_id
            self.repo.reject_face_suggestion(r.face_id, pid)

        rejected_fids = {r.face_id for r in selected}
        self._remove_faces_from_view(rejected_fids)
        self.faces_rejected.emit(list(rejected_fids))
        try:
            self.repo.recalculate_signatures_and_rematch(min_match_score=getattr(self, "_min_confidence_threshold", 0.52))
            self.refresh_suggestions()
        except Exception as e:
            logger.debug("Fast rematch after reject_selected failed: %s", e)

    def switch_selected(self) -> None:
        """Switches all currently selected outlier cards to their suggested/swap person."""
        selected = self.get_all_selected_records()
        if not selected:
            return

        switched_fids: set[int] = set()
        for r in selected:
            t_pid = getattr(r, "swap_person_id", None)
            t_name = getattr(r, "swap_person_name", None)
            if not t_pid and not t_name and getattr(r, "outlier_reason", None):
                reason = str(r.outlier_reason)
                if "Looks like" in reason:
                    inner = reason.replace("Looks like", "").strip()
                    t_name = inner.split("(")[0].strip()

            dest = t_pid if (t_pid and t_pid > 0) else t_name
            if dest:
                try:
                    new_pid = self.repo.assign_face_to_person(r.face_id, dest)
                    self.repo.verify_face_tag(r.face_id)
                    switched_fids.add(r.face_id)
                    self.faces_confirmed.emit([r.face_id], new_pid)
                except Exception as exc:
                    logger.warning("Failed to switch selected face %d to %s: %s", r.face_id, dest, exc)

        if switched_fids:
            self._remove_faces_from_view(switched_fids)
            self.refresh_suggestions()
            self._on_selection_changed()

    def _on_selection_changed(self) -> None:
        mode = getattr(self, "_current_view_mode", "suggestions")
        selected = self.get_all_selected_records()
        swap_name = None
        person_name = None
        if selected:
            for r in selected:
                s_name = getattr(r, "swap_person_name", None)
                if not s_name and "Looks like" in getattr(r, "outlier_reason", ""):
                    inner = r.outlier_reason.replace("Looks like", "").strip()
                    s_name = inner.split("(")[0].strip()
                if s_name:
                    swap_name = s_name
                    break
            person_name = selected[0].suggested_name
        self.card_selection_changed.emit(mode, swap_name, person_name)

    def ignore_selected(self) -> None:
        """Sets selected faces to state = -2 (Don't show again, I key)."""
        selected = self.get_all_selected_records()
        if not selected:
            return

        fids = [r.face_id for r in selected]
        self.repo.ignore_face(fids)
        self._remove_faces_from_view(set(fids))
        self.faces_ignored.emit(fids)

    def _on_tag_as_clicked(self) -> None:
        selected = self.get_all_selected_records()
        if not selected:
            return
        self._prompt_tag_as([r.face_id for r in selected])

    def _prompt_tag_as(self, face_ids: list[int]) -> None:
        """Prompts the user for a person's name with autocomplete and assigns the faces."""
        if not face_ids:
            return
        all_people = self.repo.get_people_list()
        name, pid = TagSomeoneDialog.prompt(all_people, self, title="Tag someone")
        if name and name.strip():
            clean_name = name.strip()
            if pid is None or pid < 0:
                pid = self.repo.upsert_person(clean_name)

            # Immediately remove from view so thumbnails disappear with zero lag
            fids_set = set(face_ids)
            self._remove_faces_from_view(fids_set)

            is_outliers = (getattr(self, "_current_view_mode", "suggestions") == "outliers")
            for fid in face_ids:
                try:
                    self.repo.assign_face_to_person(fid, pid)
                    if is_outliers:
                        self.repo.verify_face_tag(fid)
                except Exception as exc:
                    logger.warning("Failed to assign face %d to %s: %s", fid, clean_name, exc)

            self.faces_confirmed.emit(face_ids, pid)
            self.refresh_suggestions()
            self._on_selection_changed()

    # -------------------------------------------------------------------------
    # Internal Section Action Handlers
    # -------------------------------------------------------------------------
    def _on_section_confirm_all(self, person_id: int, face_ids: list[int]) -> None:
        if getattr(self, "_current_view_mode", "suggestions") == "outliers":
            for fid in face_ids:
                try:
                    self.repo.verify_face_tag(fid)
                except Exception as exc:
                    logger.warning("Failed to verify outlier tag: %s", exc)
            self._remove_faces_from_view(set(face_ids))
            self.refresh_suggestions()
            return

        if getattr(self, "_current_view_mode", "suggestions") == "boxless":
            for fid in face_ids:
                try:
                    self.repo.assign_face_to_person(fid, person_id)
                except Exception as exc:
                    logger.warning("Failed to link boxless face: %s", exc)
            self._remove_faces_from_view(set(face_ids))
            self.faces_confirmed.emit(face_ids, person_id)
            self.refresh_suggestions()
            return

        self._confirm_and_update_centroid(person_id, face_ids)

    def _on_section_name_cluster(self, name: str, face_ids: list[int]) -> None:
        clean_name = name.strip()
        if not clean_name or not face_ids:
            return
        pid = self.repo.upsert_person(clean_name)
        self._confirm_and_update_centroid(pid, face_ids)

    def _on_section_confirm_faces(self, person_id: int, face_ids: list[int]) -> None:
        if getattr(self, "_current_view_mode", "suggestions") == "outliers":
            for fid in face_ids:
                try:
                    self.repo.verify_face_tag(fid)
                except Exception as exc:
                    logger.warning("Failed to verify outlier tag: %s", exc)
            self._remove_faces_from_view(set(face_ids))
            self.refresh_suggestions()
            self._on_selection_changed()
            return
        self._confirm_and_update_centroid(person_id, face_ids)

    def _on_section_reject_faces(self, person_id: int, face_ids: list[int]) -> None:
        for fid in face_ids:
            self.repo.reject_face_suggestion(fid, person_id if person_id > 0 else None)
        self._remove_faces_from_view(set(face_ids))
        self.faces_rejected.emit(face_ids)
        try:
            self.repo.recalculate_signatures_and_rematch(min_match_score=getattr(self, "_min_confidence_threshold", 0.52))
            self.refresh_suggestions()
        except Exception as e:
            logger.debug("Fast rematch after reject failed: %s", e)

    def _on_section_not_a_face(self, face_ids: list[int]) -> None:
        self.repo.mark_not_a_face(face_ids)
        self._remove_faces_from_view(set(face_ids))
        self.faces_ignored.emit(face_ids)

    def _on_section_ignore_faces(self, face_ids: list[int]) -> None:
        self.repo.ignore_face(face_ids)
        self._remove_faces_from_view(set(face_ids))
        self.faces_ignored.emit(face_ids)

    def _on_section_tag_as(self, face_ids: list[int]) -> None:
        self._prompt_tag_as(face_ids)

    def _open_face_options_dialog(self) -> None:
        from src.ui.options_dialog import OptionsDialog
        dlg = OptionsDialog(engine=self.engine, initial_tab="face_recognition", parent=self)
        dlg.face_tab.match_threshold_changed.connect(self._on_match_threshold_slider_changed)
        dlg.face_tab.purge_rescan_requested.connect(self.refresh_suggestions)
        dlg.face_tab.fast_rematch_requested.connect(self.refresh_suggestions)
        dlg.exec()
        self.refresh_suggestions()

    def _on_match_threshold_slider_changed(self, threshold: float) -> None:
        self._min_confidence_threshold = threshold
        self.refresh_suggestions()

    def _confirm_and_update_centroid(self, person_id: int, face_ids: list[int]) -> None:
        """
        Executes atomic database confirmation (state = 1) and incremental
        centroid / multi-era sub-centroid updating for person_id.
        """
        if not face_ids or person_id <= 0:
            return

        # 1. Update SQLite state = 1, metadata_dirty = 1, and people.confirmed_count
        self.repo.confirm_faces(face_ids, person_id)

        # 2. Centroid / Sub-Centroid Updating
        conn = self.engine.catalog_conn()
        all_emb_rows = conn.execute(
            "SELECT embedding FROM media_people WHERE person_id = ? AND state = 1 AND embedding IS NOT NULL;",
            (person_id,),
        ).fetchall()

        if all_emb_rows:
            all_vectors = [deserialize_embedding(bytes(r["embedding"])) for r in all_emb_rows if r["embedding"]]
            if all_vectors:
                if len(all_vectors) >= 10:
                    sub_c = FaceVectorMatcher.compute_sub_centroids(np.stack(all_vectors, axis=0), max_k=5)
                    self.repo.update_person_centroid(person_id, serialize_embedding(sub_c))
                else:
                    mean_c = normalize_vector(np.mean(np.stack(all_vectors, axis=0), axis=0))
                    self.repo.update_person_centroid(person_id, serialize_embedding(mean_c))

        # 3. Update first avatar thumbnail if person lacks an avatar
        p_row = conn.execute("SELECT thumbnail_blob FROM people WHERE id = ?;", (person_id,)).fetchone()
        if p_row and not p_row["thumbnail_blob"]:
            thumb = self.thumb_repo.get_face_thumbnail(face_ids[0])
            if thumb:
                with self.engine.write_lock:
                    conn.execute("UPDATE people SET thumbnail_blob = ? WHERE id = ?;", (thumb, person_id))
                    conn.commit()

        # 4. Instant re-match remaining unassigned faces against updated centroids (<50ms)
        try:
            self.repo.recalculate_signatures_and_rematch(min_match_score=getattr(self, "_min_confidence_threshold", 0.52))
        except Exception as e:
            logger.debug("Fast rematch after confirm failed: %s", e)

        # 5. Remove confirmed face cards from UI
        self._remove_faces_from_view(set(face_ids))
        self.faces_confirmed.emit(face_ids, person_id)
        self.refresh_suggestions()

    def _remove_faces_from_view(self, face_ids: set[int]) -> None:
        """Removes the given face_ids from all sections and prunes empty sections."""
        for pid in list(self._sections.keys()):
            sec = self._sections[pid]
            remaining = sec.remove_face_ids(face_ids)
            if remaining <= 0:
                sec.deleteLater()
                del self._sections[pid]
        self._update_empty_state_label()

    def _open_quick_preview(self, record: FaceCardRecord) -> None:
        """Opens full photo quick-preview popover centered on face box (Space key)."""
        dlg = _FaceQuickPreviewDialog(record, self)
        dlg.exec()

    # -------------------------------------------------------------------------
    # Keyboard Navigation & Shortcuts ('SPEC.md' Section 5.4)
    # -------------------------------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()

        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.confirm_selected()
            event.accept()
            return

        elif key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.reject_selected()
            event.accept()
            return

        elif key == Qt.Key_I:
            self.ignore_selected()
            event.accept()
            return

        elif key == Qt.Key_Space:
            sel = self.get_all_selected_records()
            if sel:
                self._open_quick_preview(sel[0])
            event.accept()
            return

        elif key == Qt.Key_Escape:
            self.back_to_gallery_requested.emit()
            event.accept()
            return

        elif event.matches(QKeySequence.SelectAll):
            self.select_all()
            event.accept()
            return

        super().keyPressEvent(event)
