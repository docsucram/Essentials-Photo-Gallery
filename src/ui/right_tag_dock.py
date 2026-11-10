"""
TagAndCaptionDock (QDockWidget) implementing Windows Live Photo Gallery's Right Inspector:
- Modes (mutually exclusive, toggled by Ctrl+I / Ctrl+F):
  1. Information & Tag Mode (Ctrl+I):
     - People tags, Geotag, Caption, Descriptive tags
     - Subtle divider
     - Information: Filename, Date taken, Size, Rating, Flag, Dimensions,
       Published, Camera, Author, Exposure, Aperture, Focal length, ISO, Latitude, Longitude
  2. Fine-Tune Mode (Ctrl+F, matches docs/reference_ui/fullsize and edit_ui.png & media_1792165772919_087b03d2.png):
     - Adjust exposure: Brightness, Contrast, Shadows, Highlights + Live Luminance Histogram
     - Adjust color: Color temperature, Tint, Saturation
     - Straighten photo: angle slider with zero detent
     - Adjust detail: Sharpen, Reduce noise + Analyze button
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import (
    QDate,
    QDateTime,
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRectF,
    QSize,
    QStringListModel,
    QTime,
    QTimeZone,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QCompleter,
    QDateEdit,
    QDockWidget,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QSplitterHandle,
    QStackedWidget,
    QStyle,
    QTextEdit,
    QTimeEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from legacy_reference.imgBrowserv13 import (
    _format_exif_orientation,
    format_file_size,
)
from src.core.geotag_resolver import (
    resolve_location_query,
    reverse_geocode,
    search_places_offline,
    search_places_online,
)
from src.db.repository import _epoch_to_datetime_utc
from src.ui.ribbon_toolbar import make_person_avatar_pixmap, make_symbol_icon, make_wlpg_icon


class _AeroFineTuneSlider(QSlider):
    """High-fidelity Aero horizontal slider with double-click to reset."""

    def __init__(self, min_val: int, max_val: int, default_val: int, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Horizontal, parent)
        self.default_val = default_val
        self.setRange(min_val, max_val)
        self.setValue(default_val)
        self.setFixedHeight(18)
        self.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 4px;
                background: #D8DFE6;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0078D7;
                border: 1px solid #005A9E;
                width: 8px;
                margin: -6px 0;
                border-radius: 2px;
            }
            QSlider::handle:horizontal:hover {
                background: #2B88D8;
            }
            """
        )

    def mousePressEvent(self, event: Any) -> None:
        self.sliderPressed.emit()
        if event.button() == Qt.LeftButton:
            opt_min, opt_max = self.minimum(), self.maximum()
            if self.orientation() == Qt.Horizontal and self.width() > 0:
                pos = event.position().x() if hasattr(event, "position") else event.x()
                val = opt_min + (opt_max - opt_min) * (pos / float(self.width()))
                self.setValue(int(round(val)))
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        super().mouseReleaseEvent(event)
        self.sliderReleased.emit()

    def mouseDoubleClickEvent(self, event: Any) -> None:
        self.setValue(self.default_val)
        event.accept()


class HistogramWidget(QWidget):
    """Renders the authentic WLPG dark luminance histogram with interactive Black/White point markers."""

    points_changed = Signal(int, int)  # (black_point, white_point)
    slider_pressed = Signal()
    slider_released = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(94)
        self._counts: list[int] = [0] * 256
        self.black_point: int = 0      # 0 .. 255
        self.white_point: int = 255    # 0 .. 255
        self._drag_handle: str | None = None
        self._hover_handle: str | None = None
        self.setMouseTracking(True)

    def set_histogram_data(self, image_data: Any) -> None:
        if image_data is None:
            self._counts = [0] * 256
            self.update()
            return
        try:
            if isinstance(image_data, np.ndarray):
                gray = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY) if image_data.ndim == 3 else image_data
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
                max_val = float(hist.max()) if hist.max() > 0 else 1.0
                self._counts = [int((val / max_val) * 1000) for val in hist]
            elif isinstance(image_data, Image.Image):
                thumb = image_data.resize((160, 120), Image.Resampling.BILINEAR).convert("L")
                hist = thumb.histogram()
                max_val = max(hist) if hist and max(hist) > 0 else 1
                self._counts = [int((val / max_val) * 1000) for val in hist[:256]]
            else:
                self._counts = [0] * 256
        except Exception:
            self._counts = [0] * 256
        self.update()

    def reset_points(self) -> None:
        self.black_point = 0
        self.white_point = 255
        self.update()

    def _get_handle_x(self, val: int) -> float:
        x_min = 6.0
        x_max = max(7.0, float(self.width() - 6.0))
        return x_min + (val / 255.0) * (x_max - x_min)

    def _get_val_from_x(self, x: float) -> int:
        x_min = 6.0
        x_max = max(7.0, float(self.width() - 6.0))
        norm = (x - x_min) / max(1.0, x_max - x_min)
        return max(0, min(255, int(round(norm * 255.0))))

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w = self.width()
        h = self.height()
        graph_bottom = 68.0

        # Black background with subtle rounded border
        p.setPen(QPen(QColor("#7B8B9B"), 1))
        p.setBrush(QColor("#000000"))
        p.drawRoundedRect(QRectF(1, 1, w - 2, graph_bottom), 2, 2)

        # Subtle dark grey grid lines (matching WLPG screenshot)
        p.setPen(QPen(QColor("#2C2C2C"), 1))
        p.drawLine(QPointF(1, graph_bottom / 2.0), QPointF(w - 2, graph_bottom / 2.0))
        p.drawLine(QPointF(w * 0.33, 1), QPointF(w * 0.33, graph_bottom))
        p.drawLine(QPointF(w * 0.66, 1), QPointF(w * 0.66, graph_bottom))

        # Histogram curve polygon
        if any(self._counts):
            path = QPainterPath()
            path.moveTo(6, graph_bottom - 2)
            usable_w = max(1.0, float(w - 12))
            usable_h = graph_bottom - 4
            step = usable_w / 255.0

            for i in range(256):
                x = 6 + i * step
                y = (graph_bottom - 2) - (self._counts[i] / 1000.0) * usable_h
                path.lineTo(x, y)
            path.lineTo(w - 6, graph_bottom - 2)
            path.closeSubpath()

            grad = QLinearGradient(0, 0, 0, graph_bottom)
            grad.setColorAt(0.0, QColor(220, 225, 230, 220))
            grad.setColorAt(1.0, QColor(140, 150, 160, 180))
            p.setPen(QPen(QColor("#D0D5DA"), 1))
            p.setBrush(grad)
            p.drawPath(path)

        # Draw interactive upward-pointing arrow handles at black and white points
        bx = self._get_handle_x(self.black_point)
        wx = self._get_handle_x(self.white_point)

        def draw_marker(cx: float, is_active: bool, is_black: bool) -> None:
            # Upward arrow / pentagon shape pointing directly at the level axis
            poly = [
                QPointF(cx, graph_bottom + 1.0),
                QPointF(cx - 5.5, graph_bottom + 6.5),
                QPointF(cx - 5.5, graph_bottom + 17.5),
                QPointF(cx + 5.5, graph_bottom + 17.5),
                QPointF(cx + 5.5, graph_bottom + 6.5),
            ]
            if is_black:
                fill_col = QColor("#222222") if not is_active else QColor("#444444")
                border_col = QColor("#111111") if not is_active else QColor("#0078D7")
            else:
                fill_col = QColor("#E0E0E0") if not is_active else QColor("#FFFFFF")
                border_col = QColor("#888888") if not is_active else QColor("#0078D7")

            p.setPen(QPen(border_col, 1.2))
            p.setBrush(fill_col)
            p.drawPolygon(poly)

            # Center accent line
            accent_col = QColor(255, 255, 255, 180) if is_black else QColor(40, 40, 40, 180)
            p.setPen(QPen(accent_col, 1))
            p.drawLine(QPointF(cx, graph_bottom + 8.0), QPointF(cx, graph_bottom + 15.0))

        draw_marker(bx, self._drag_handle == "black" or self._hover_handle == "black", True)
        draw_marker(wx, self._drag_handle == "white" or self._hover_handle == "white", False)
        p.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            bx = self._get_handle_x(self.black_point)
            wx = self._get_handle_x(self.white_point)
            pos_x = float(event.pos().x())
            pos_y = float(event.pos().y())

            dist_b = abs(pos_x - bx)
            dist_w = abs(pos_x - wx)

            if dist_b <= 14.0 and dist_b <= dist_w:
                self._drag_handle = "black"
                self.slider_pressed.emit()
                self.update()
                event.accept()
                return
            elif dist_w <= 14.0:
                self._drag_handle = "white"
                self.slider_pressed.emit()
                self.update()
                event.accept()
                return
            elif pos_y >= 60.0:
                if dist_b < dist_w:
                    self._drag_handle = "black"
                    val = self._get_val_from_x(pos_x)
                    self.black_point = max(0, min(val, self.white_point - 4))
                    self.points_changed.emit(self.black_point, self.white_point)
                else:
                    self._drag_handle = "white"
                    val = self._get_val_from_x(pos_x)
                    self.white_point = max(self.black_point + 4, min(val, 255))
                    self.points_changed.emit(self.black_point, self.white_point)
                self.slider_pressed.emit()
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        pos_x = float(event.pos().x())
        pos_y = float(event.pos().y())

        if self._drag_handle == "black":
            val = self._get_val_from_x(pos_x)
            self.black_point = max(0, min(val, self.white_point - 4))
            self.points_changed.emit(self.black_point, self.white_point)
            self.update()
            event.accept()
            return
        elif self._drag_handle == "white":
            val = self._get_val_from_x(pos_x)
            self.white_point = max(self.black_point + 4, min(val, 255))
            self.points_changed.emit(self.black_point, self.white_point)
            self.update()
            event.accept()
            return

        bx = self._get_handle_x(self.black_point)
        wx = self._get_handle_x(self.white_point)
        dist_b = abs(pos_x - bx)
        dist_w = abs(pos_x - wx)
        prev_hover = self._hover_handle

        if (pos_y >= 55.0 and dist_b <= 14.0) or dist_b <= 10.0:
            self._hover_handle = "black"
            self.setCursor(Qt.PointingHandCursor)
            self.setToolTip(f"Black point (Shadows clip): {self.black_point}")
        elif (pos_y >= 55.0 and dist_w <= 14.0) or dist_w <= 10.0:
            self._hover_handle = "white"
            self.setCursor(Qt.PointingHandCursor)
            self.setToolTip(f"White point (Highlights clip): {self.white_point}")
        else:
            self._hover_handle = None
            self.unsetCursor()
            self.setToolTip("")

        if self._hover_handle != prev_hover:
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_handle is not None:
            self._drag_handle = None
            self.slider_released.emit("Histogram Levels")
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.reset_points()
        self.points_changed.emit(0, 255)
        self.slider_released.emit("Reset Levels")
        event.accept()


class _RemovableChipWidget(QFrame):
    """Compact WLPG blue pill chip with label, optional avatar icon, and '✕' remove button."""

    remove_clicked = Signal(int)

    def __init__(
        self,
        item_id: int,
        label_text: str,
        tooltip: str = "",
        icon_pixmap: QPixmap | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.item_id = item_id
        self.setStyleSheet(
            """
            QFrame {
                background-color: #EBF3FB;
                border: 1px solid #B5D2ED;
                border-radius: 3px;
            }
            QLabel {
                border: none;
                background: transparent;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                padding-left: 2px;
            }
            QPushButton {
                border: none;
                background: transparent;
                color: #5A7A9E;
                font-weight: bold;
                font-size: 9pt;
                padding: 0px 4px;
            }
            QPushButton:hover {
                color: #C50F1F;
            }
            """
        )
        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(3, 1, 3, 1)
        hbox.setSpacing(3)

        if icon_pixmap is not None and not icon_pixmap.isNull():
            icon_lbl = QLabel(self)
            icon_lbl.setPixmap(icon_pixmap)
            icon_lbl.setFixedSize(18, 18)
            hbox.addWidget(icon_lbl)

        lbl = QLabel(label_text, self)
        if tooltip:
            lbl.setToolTip(tooltip)
        hbox.addWidget(lbl, 1)

        btn_del = QPushButton("✕", self)
        btn_del.setFixedWidth(16)
        btn_del.setToolTip(f"Remove '{label_text}'")
        btn_del.clicked.connect(lambda: self.remove_clicked.emit(self.item_id))
        hbox.addWidget(btn_del)


class _AddPersonLineEdit(QLineEdit):
    """LineEdit for 'Add people tags' that emits `clicked` when clicked so SingleMediaViewer can start face tagging mode."""

    clicked = Signal()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.clicked.emit()


class _UnknownFaceCardWidget(QFrame):
    """
    Card for an unassigned detected face (state = 0, person_id IS NULL) in RightTagDock.
    - 30x30 square face avatar thumbnail on the left
    - Clickable 'Who is this?' line edit with QCompleter of known people
    - '✕' button to dismiss false-positive box (state = -1)
    """

    assign_requested = Signal(int, str)       # (face_id, name)
    dismiss_requested = Signal(int)          # (face_id)
    hovered_changed = Signal(int, bool)       # (face_id, is_hovered)

    def __init__(
        self,
        face_id: int,
        icon_pixmap: QPixmap | None = None,
        completer: QCompleter | None = None,
        unlinked_candidates: list[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.face_id = int(face_id)
        self.setObjectName("UnknownFaceCard")
        self.setMouseTracking(True)
        self.setFixedHeight(40)
        self.setProperty("hovered", "false")
        self.setStyleSheet(
            """
            QFrame#UnknownFaceCard {
                background: transparent;
                border: 1px dashed #A4C2E8;
                border-radius: 3px;
            }
            QFrame#UnknownFaceCard[hovered="true"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F6FAFE, stop:1 #E5F1FB);
                border: 1px solid #2979CC;
            }
            """
        )
        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(3, 2, 2, 2)
        hbox.setSpacing(4)

        self.icon_lbl = QLabel(self)
        self.icon_lbl.setFixedSize(34, 34)
        self.icon_lbl.setStyleSheet("border: none; background: transparent; border-radius: 2px;")
        if icon_pixmap is not None and not icon_pixmap.isNull():
            self.icon_lbl.setPixmap(
                icon_pixmap.scaled(34, 34, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            )
        hbox.addWidget(self.icon_lbl)

        self.txt_who = QLineEdit(self)
        self.txt_who.setPlaceholderText("Who is this?")
        cands = [c.strip() for c in (unlinked_candidates or []) if c and c.strip()]
        if completer is not None and hasattr(completer, "model") and completer.model() is not None:
            existing_names: list[str] = []
            m = completer.model()
            if hasattr(m, "stringList"):
                existing_names = list(m.stringList())
            elif hasattr(m, "rowCount"):
                existing_names = [m.data(m.index(i, 0)) for i in range(m.rowCount())]
            merged_names = list(cands) + [n for n in existing_names if n not in cands]
            card_model = QStringListModel(merged_names, self.txt_who)
            card_comp = QCompleter(card_model, self.txt_who)
            card_comp.setCaseSensitivity(Qt.CaseInsensitive)
            card_comp.setFilterMode(Qt.MatchContains)
            card_comp.activated[str].connect(self._on_completer_activated)
            self.txt_who.setCompleter(card_comp)
        self.txt_who.setStyleSheet(
            """
            QLineEdit {
                background: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 2px;
                padding: 1px 4px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                color: #1E395B;
            }
            QLineEdit:hover, QLineEdit:focus {
                border-color: #2979CC;
            }
            """
        )
        self.txt_who.returnPressed.connect(self._on_return)
        self.txt_who.setContextMenuPolicy(Qt.CustomContextMenu)
        self.txt_who.customContextMenuRequested.connect(self._show_txt_context_menu)
        hbox.addWidget(self.txt_who, 1)

        # Quick-link buttons for boxless person tags on this photo
        if cands:
            for cand in cands[:2]:
                btn_link = QPushButton(f"Link {cand}", self)
                btn_link.setFixedHeight(22)
                btn_link.setCursor(Qt.PointingHandCursor)
                btn_link.setToolTip(f"Link detected face to tagged person '{cand}'")
                btn_link.setStyleSheet(
                    """
                    QPushButton {
                        background: #E8F4FD;
                        border: 1px solid #70A6E8;
                        border-radius: 3px;
                        color: #104C8A;
                        font-family: 'Segoe UI';
                        font-size: 8pt;
                        font-weight: 600;
                        padding: 0 5px;
                    }
                    QPushButton:hover {
                        background: #D0E8FC;
                        border-color: #2979CC;
                    }
                    QPushButton:pressed {
                        background: #B9DCFB;
                    }
                    """
                )
                btn_link.clicked.connect(lambda _, c=cand: self.assign_requested.emit(self.face_id, c))
                hbox.addWidget(btn_link)

        self.btn_del = QPushButton(self)
        self.btn_del.setIcon(make_symbol_icon("cross", 14))
        self.btn_del.setFixedSize(18, 18)
        self.btn_del.setCursor(Qt.PointingHandCursor)
        self.btn_del.setToolTip("Dismiss face")
        self.btn_del.setStyleSheet(
            """
            QPushButton {
                border: none;
                background: transparent;
                border-radius: 2px;
            }
            QPushButton:hover {
                background: #F8D7DA;
            }
            """
        )
        self.btn_del.clicked.connect(lambda: self.dismiss_requested.emit(self.face_id))
        hbox.addWidget(self.btn_del)

    def contextMenuEvent(self, event: Any) -> None:
        self._show_context_menu(event.globalPos())
        event.accept()

    def _show_txt_context_menu(self, pos: QPoint) -> None:
        self._show_context_menu(self.txt_who.mapToGlobal(pos))

    def _show_context_menu(self, global_pos: QPoint) -> None:
        menu = QMenu(self)
        menu.setFont(QFont("Segoe UI", 9))

        act_tag = menu.addAction("Tag person...")
        act_tag.triggered.connect(self._focus_name_input)

        menu.addSeparator()

        act_ignore = menu.addAction("Ignore face")
        act_ignore.triggered.connect(lambda: self.dismiss_requested.emit(self.face_id))

        act_not_face = menu.addAction("Not a face")
        act_not_face.triggered.connect(lambda: self.dismiss_requested.emit(self.face_id))

        menu.exec(global_pos)

    def _focus_name_input(self) -> None:
        self.txt_who.setFocus()
        self.txt_who.selectAll()

    def _on_completer_activated(self, text: str) -> None:
        name = text.strip()
        if name:
            self.txt_who.setText(name)
            self.assign_requested.emit(self.face_id, name)

    def _on_return(self) -> None:
        text = self.txt_who.text().strip()
        if text:
            self.assign_requested.emit(self.face_id, text)

    def enterEvent(self, event: Any) -> None:
        super().enterEvent(event)
        self.setProperty("hovered", "true")
        self.style().unpolish(self)
        self.style().polish(self)
        self.hovered_changed.emit(self.face_id, True)

    def leaveEvent(self, event: Any) -> None:
        super().leaveEvent(event)
        self.setProperty("hovered", "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.hovered_changed.emit(self.face_id, False)

    def set_dark_mode(self, is_dark: bool) -> None:
        if is_dark:
            self.setStyleSheet(
                """
                QFrame#UnknownFaceCard {
                    background: transparent;
                    border: 1px dashed #3E4756;
                    border-radius: 3px;
                }
                QFrame#UnknownFaceCard[hovered="true"] {
                    background: #242A36;
                    border: 1px solid #3B82F6;
                }
                """
            )
            self.txt_who.setStyleSheet(
                """
                QLineEdit {
                    background: #14171B;
                    border: 1px solid #333A45;
                    border-radius: 2px;
                    padding: 1px 4px;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    color: #E2E8F0;
                }
                QLineEdit:hover, QLineEdit:focus {
                    border-color: #3B82F6;
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                QFrame#UnknownFaceCard {
                    background: transparent;
                    border: 1px dashed #A4C2E8;
                    border-radius: 3px;
                }
                QFrame#UnknownFaceCard[hovered="true"] {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F6FAFE, stop:1 #E5F1FB);
                    border: 1px solid #2979CC;
                }
                """
            )
            self.txt_who.setStyleSheet(
                """
                QLineEdit {
                    background: #FFFFFF;
                    border: 1px solid #CBD5E1;
                    border-radius: 2px;
                    padding: 1px 4px;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    color: #1E395B;
                }
                QLineEdit:hover, QLineEdit:focus {
                    border-color: #2979CC;
                }
                """
            )


class _PersonTagCardWidget(QFrame):
    """
    WLPG-style person tag split-button card in the right pane (media_1792269887974_67460900.png):
    - 30x30 square face avatar thumbnail on the left
    - Person name label in the middle (clicking searches for photos of this person)
    - If unconfirmed suggestion: inline [✓] Confirm and [✗] Reject buttons
    - If confirmed: [✕] Delete button
    - Split dropdown arrow '▾' on the right opening the WLPG context menu:
      * Search for photos of this person
      * View all suggestions
      * ---
      * Tag as someone else...
      * Remove tag
      * Set as display picture
      * ---
      * Merge...
      * Rename contact...
    """

    remove_clicked = Signal(int)
    search_clicked = Signal(int, str)
    view_suggestions_clicked = Signal(int)
    tag_as_other_clicked = Signal(int)
    set_display_pic_clicked = Signal(int)
    confirm_clicked = Signal(int, int)  # (person_id, face_id)
    reject_suggestion_clicked = Signal(int, int)  # (person_id, face_id)
    merge_clicked = Signal(int, str)
    rename_clicked = Signal(int, str)
    hovered_changed = Signal(int, bool)
    locate_face_clicked = Signal(int, str)  # (person_id, person_name)
    link_face_clicked = Signal(int, int)  # (person_id, face_id)

    def __init__(
        self,
        person_id: int,
        person_name: str,
        icon_pixmap: QPixmap | None = None,
        is_unconfirmed: bool = False,
        face_id: int | None = None,
        is_boxless: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.item_id = int(person_id)
        self.person_id = int(person_id)
        self.person_name = str(person_name)
        self.is_unconfirmed = bool(is_unconfirmed)
        self.face_id = face_id
        self.is_boxless = bool(is_boxless)
        self.setObjectName("PersonTagCard")
        self.setMouseTracking(True)
        self.setFixedHeight(40)
        self.setProperty("hovered", "false")

        self.setStyleSheet(
            """
            QFrame#PersonTagCard {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
            }
            QFrame#PersonTagCard[hovered="true"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F6FAFE, stop:1 #E5F1FB);
                border: 1px solid #84ACDD;
            }
            QLabel#PersonNameLabel {
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 9pt;
                background: transparent;
                border: none;
                padding-left: 4px;
            }
            QToolButton#PersonMenuBtn {
                border: none;
                border-left: 1px solid transparent;
                background: transparent;
                color: #1E395B;
                font-size: 8pt;
                padding: 0px 4px;
            }
            QFrame#PersonTagCard[hovered="true"] QToolButton#PersonMenuBtn {
                border-left: 1px solid #A4C2E8;
            }
            QToolButton#PersonMenuBtn:hover {
                background: rgba(132, 172, 221, 55);
            }
            """
        )

        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(3, 2, 2, 2)
        hbox.setSpacing(4)

        self.icon_lbl = QLabel(self)
        self.icon_lbl.setFixedSize(34, 34)
        self.icon_lbl.setStyleSheet("border: none; background: transparent; border-radius: 2px;")
        if icon_pixmap is not None and not icon_pixmap.isNull():
            self.icon_lbl.setPixmap(
                icon_pixmap.scaled(34, 34, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            )
        hbox.addWidget(self.icon_lbl)

        display_text = f"{self.person_name}?" if self.is_unconfirmed else self.person_name
        self.lbl_name = QLabel(display_text, self)
        self.lbl_name.setObjectName("PersonNameLabel")
        self.lbl_name.setCursor(Qt.PointingHandCursor)
        self.lbl_name.setToolTip(f"Search for photos of {self.person_name}")
        hbox.addWidget(self.lbl_name, 1)

        if self.is_boxless:
            self.btn_locate_box = QPushButton("(no box - add)", self)
            self.btn_locate_box.setObjectName("LocateBoxBtn")
            self.btn_locate_box.setCursor(Qt.PointingHandCursor)
            self.btn_locate_box.setToolTip(f"Click to place or link a face box for {self.person_name} on this photo")
            self.btn_locate_box.setStyleSheet(
                """
                QPushButton#LocateBoxBtn {
                    color: #0066CC;
                    font-family: 'Segoe UI';
                    font-size: 8pt;
                    font-style: italic;
                    background: transparent;
                    border: 1px dashed #A4C2E8;
                    border-radius: 2px;
                    padding: 1px 4px;
                }
                QPushButton#LocateBoxBtn:hover {
                    color: #003366;
                    background: #EBF3FB;
                    border: 1px solid #0066CC;
                }
                """
            )
            self.btn_locate_box.clicked.connect(lambda: self.locate_face_clicked.emit(self.person_id, self.person_name))
            hbox.addWidget(self.btn_locate_box)

        if self.is_unconfirmed:
            self.btn_confirm = QPushButton(self)
            self.btn_confirm.setIcon(make_symbol_icon("check", 14))
            self.btn_confirm.setFixedSize(18, 18)
            self.btn_confirm.setCursor(Qt.PointingHandCursor)
            self.btn_confirm.setToolTip("Confirm this suggestion")
            self.btn_confirm.setStyleSheet(
                "QPushButton { background: transparent; border: 1px solid transparent; border-radius: 2px; }"
                "QPushButton:hover { background: #E8F5E9; border: 1px solid #A5D6A7; }"
                "QPushButton:pressed { background: #C8E6C9; }"
            )
            self.btn_confirm.clicked.connect(lambda: self.confirm_clicked.emit(self.person_id, self.face_id or 0))
            hbox.addWidget(self.btn_confirm)

            self.btn_reject = QPushButton(self)
            self.btn_reject.setIcon(make_symbol_icon("cross", 14))
            self.btn_reject.setFixedSize(18, 18)
            self.btn_reject.setCursor(Qt.PointingHandCursor)
            self.btn_reject.setToolTip("Not this person")
            self.btn_reject.setStyleSheet(
                "QPushButton { background: transparent; border: 1px solid transparent; border-radius: 2px; }"
                "QPushButton:hover { background: #FFEBEE; border: 1px solid #FFCDD2; }"
                "QPushButton:pressed { background: #FFCDD2; }"
            )
            self.btn_reject.clicked.connect(lambda: self.reject_suggestion_clicked.emit(self.person_id, self.face_id or 0))
            hbox.addWidget(self.btn_reject)

        self.btn_menu = QToolButton(self)
        self.btn_menu.setObjectName("PersonMenuBtn")
        self.btn_menu.setText("▾")
        self.btn_menu.setFixedSize(16, 26)
        self.btn_menu.setCursor(Qt.PointingHandCursor)
        self.btn_menu.setToolTip("Options")
        self.btn_menu.clicked.connect(self._show_dropdown_menu)
        hbox.addWidget(self.btn_menu)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.search_clicked.emit(self.person_id, self.person_name)
            event.accept()
            return
        super().mousePressEvent(event)

    def contextMenuEvent(self, event: Any) -> None:
        self._exec_menu(event.globalPos())
        event.accept()

    def _show_dropdown_menu(self) -> None:
        pos = self.btn_menu.mapToGlobal(QPoint(0, self.btn_menu.height()))
        self._exec_menu(pos)

    def _exec_menu(self, global_pos: QPoint) -> QMenu:
        menu = QMenu(self)
        menu.setFont(QFont("Segoe UI", 9))

        if self.is_boxless:
            act_locate = menu.addAction(f"Locate {self.person_name} on photo...")
            act_locate.setIcon(make_symbol_icon("crosshair", 14) if callable(globals().get("make_symbol_icon")) else QIcon())
            act_locate.triggered.connect(lambda: self.locate_face_clicked.emit(self.person_id, self.person_name))

            detected_faces = getattr(self, "detected_unassigned_faces", [])
            if len(detected_faces) == 1:
                df = detected_faces[0]
                df_id = int(df.get("face_id") or df.get("id") or 0)
                act_link = menu.addAction("Link to detected face box")
                act_link.triggered.connect(lambda checked=False, f_id=df_id: self.link_face_clicked.emit(self.person_id, f_id))
            elif len(detected_faces) > 1:
                link_menu = menu.addMenu("Link to detected face box")
                for idx, df in enumerate(detected_faces):
                    df_id = int(df.get("face_id") or df.get("id") or 0)
                    blob = df.get("thumbnail_blob")
                    pix = make_person_avatar_pixmap(f"Face {idx + 1}", blob, size=20, circular=False) if blob else None
                    act_df = link_menu.addAction(f"Detected Face #{idx + 1}")
                    if pix and not pix.isNull():
                        act_df.setIcon(QIcon(pix))
                    act_df.triggered.connect(lambda checked=False, f_id=df_id: self.link_face_clicked.emit(self.person_id, f_id))

            menu.addSeparator()
        elif not self.is_unconfirmed:
            act_adjust = menu.addAction("Adjust face box on photo...")
            act_adjust.triggered.connect(lambda: self.locate_face_clicked.emit(self.person_id, self.person_name))
            menu.addSeparator()

        if self.is_unconfirmed:
            act_confirm = menu.addAction("Confirm")
            act_confirm.triggered.connect(
                lambda: self.confirm_clicked.emit(self.person_id, self.face_id or 0)
            )

            act_tag_other = menu.addAction("Tag as someone else...")
            act_tag_other.triggered.connect(lambda: self.tag_as_other_clicked.emit(self.person_id))

            act_display_pic = menu.addAction("Set as display picture")
            act_display_pic.triggered.connect(lambda: self.set_display_pic_clicked.emit(self.person_id))

            menu.addSeparator()

            act_search = menu.addAction("Search for photos of this person")
            act_search.triggered.connect(lambda: self.search_clicked.emit(self.person_id, self.person_name))

            act_suggest = menu.addAction("View all suggestions")
            act_suggest.triggered.connect(lambda: self.view_suggestions_clicked.emit(self.person_id))

            menu.addSeparator()

            act_reject = menu.addAction("Don't show suggestion")
            act_reject.triggered.connect(
                lambda: self.reject_suggestion_clicked.emit(self.person_id, self.face_id or 0)
            )
        else:
            act_search = menu.addAction("Search for photos of this person")
            act_search.triggered.connect(lambda: self.search_clicked.emit(self.person_id, self.person_name))

            act_suggest = menu.addAction("View all suggestions")
            act_suggest.triggered.connect(lambda: self.view_suggestions_clicked.emit(self.person_id))

            menu.addSeparator()

            act_tag_other = menu.addAction("Tag as someone else...")
            act_tag_other.triggered.connect(lambda: self.tag_as_other_clicked.emit(self.person_id))

            act_remove = menu.addAction("Remove tag")
            act_remove.triggered.connect(lambda: self.remove_clicked.emit(self.person_id))

            act_display_pic = menu.addAction("Set as display picture")
            act_display_pic.triggered.connect(lambda: self.set_display_pic_clicked.emit(self.person_id))

            menu.addSeparator()

            act_merge = menu.addAction("Merge...")
            act_merge.triggered.connect(lambda: self.merge_clicked.emit(self.person_id, self.person_name))

            act_rename = menu.addAction("Rename contact...")
            act_rename.triggered.connect(lambda: self.rename_clicked.emit(self.person_id, self.person_name))

        if not getattr(self, "_suppress_menu_exec", False):
            menu.exec(global_pos)
        return menu

    def enterEvent(self, event: Any) -> None:
        super().enterEvent(event)
        self.setProperty("hovered", "true")
        self.style().unpolish(self)
        self.style().polish(self)
        self.hovered_changed.emit(self.person_id, True)

    def leaveEvent(self, event: Any) -> None:
        super().leaveEvent(event)
        self.setProperty("hovered", "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.hovered_changed.emit(self.person_id, False)

    def set_dark_mode(self, is_dark: bool) -> None:
        if is_dark:
            self.setStyleSheet(
                """
                QFrame#PersonTagCard {
                    background: transparent;
                    border: 1px solid transparent;
                    border-radius: 3px;
                }
                QFrame#PersonTagCard[hovered="true"] {
                    background: #242A36;
                    border: 1px solid #3B82F6;
                }
                QLabel#PersonNameLabel {
                    color: #E2E8F0;
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                    background: transparent;
                    border: none;
                    padding-left: 4px;
                }
                QToolButton#PersonMenuBtn {
                    border: none;
                    border-left: 1px solid transparent;
                    background: transparent;
                    color: #94A3B8;
                    font-size: 8pt;
                    padding: 0px 4px;
                }
                QFrame#PersonTagCard[hovered="true"] QToolButton#PersonMenuBtn {
                    border-left: 1px solid #3E4756;
                }
                QToolButton#PersonMenuBtn:hover {
                    background: rgba(59, 130, 246, 55);
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                QFrame#PersonTagCard {
                    background: transparent;
                    border: 1px solid transparent;
                    border-radius: 3px;
                }
                QFrame#PersonTagCard[hovered="true"] {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #F6FAFE, stop:1 #E5F1FB);
                    border: 1px solid #84ACDD;
                }
                QLabel#PersonNameLabel {
                    color: #1E395B;
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                    background: transparent;
                    border: none;
                    padding-left: 4px;
                }
                QToolButton#PersonMenuBtn {
                    border: none;
                    border-left: 1px solid transparent;
                    background: transparent;
                    color: #1E395B;
                    font-size: 8pt;
                    padding: 0px 4px;
                }
                QFrame#PersonTagCard[hovered="true"] QToolButton#PersonMenuBtn {
                    border-left: 1px solid #A4C2E8;
                }
                QToolButton#PersonMenuBtn:hover {
                    background: rgba(132, 172, 221, 55);
                }
                """
            )


class _DescriptiveTagRowWidget(QFrame):
    """
    WLPG-style hoverable descriptive tag row:
    - Default state: clean hyperlink blue text, no border, no cross button.
    - Hover state: subtle light-blue border box and cross (✕) button appears.
    - Hovering specifically over the text underlines it.
    - If multiple items selected: shows tag count in parentheses, e.g. "Tag (23)".
    - Clicking the text emits tag_clicked(tag_id, tag_name).
    - Clicking the cross emits remove_clicked(tag_id).
    - Right-click opens context menu: "Remove tag" and "Assign to all".
    """

    remove_clicked = Signal(int)
    tag_clicked = Signal(int, str)
    assign_to_all_clicked = Signal(int)

    def __init__(
        self,
        item_id: int,
        label_text: str,
        tag_count: int = 1,
        total_selection_count: int = 1,
        tooltip: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.item_id = item_id
        self.label_text = label_text
        self.tag_count = tag_count
        self.total_selection_count = total_selection_count
        self.setObjectName("DescriptiveTagRow")
        self.setMouseTracking(True)

        self.setStyleSheet(
            """
            QFrame#DescriptiveTagRow {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 2px;
            }
            QFrame#DescriptiveTagRow[hovered="true"] {
                background: #F0F6FC;
                border: 1px solid #7DA2CE;
            }
            QLabel#TagLabel {
                color: #0066CC;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                padding-left: 2px;
                background: transparent;
                border: none;
            }
            QLabel#TagLabel:hover {
                text-decoration: underline;
                color: #004499;
            }
            QLabel#TagCountLabel {
                color: #555555;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                background: transparent;
                border: none;
            }
            QToolButton#DeleteBtn {
                border: none;
                background: transparent;
                color: #4C4C4C;
                font-family: 'Segoe UI';
                font-size: 8pt;
                padding: 0px 2px;
            }
            QToolButton#DeleteBtn:hover {
                color: #C50F1F;
                background: rgba(0, 0, 0, 15);
                border-radius: 2px;
            }
            """
        )

        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(3, 1, 3, 1)
        hbox.setSpacing(3)

        self.lbl = QLabel(label_text, self)
        self.lbl.setObjectName("TagLabel")
        self.lbl.setCursor(Qt.PointingHandCursor)
        if tooltip:
            self.lbl.setToolTip(str(tooltip))
        self.lbl.mousePressEvent = self._on_label_mouse_press
        hbox.addWidget(self.lbl)

        if total_selection_count > 1:
            self.lbl_count = QLabel(f"({tag_count})", self)
            self.lbl_count.setObjectName("TagCountLabel")
            hbox.addWidget(self.lbl_count)

        hbox.addStretch(1)

        self.btn_del = QToolButton(self)
        self.btn_del.setObjectName("DeleteBtn")
        self.btn_del.setText("✕")
        self.btn_del.setFixedSize(16, 16)
        self.btn_del.setCursor(Qt.PointingHandCursor)
        self.btn_del.setToolTip(f"Remove '{label_text}'")
        self.btn_del.clicked.connect(lambda: self.remove_clicked.emit(self.item_id))
        self.btn_del.hide()
        hbox.addWidget(self.btn_del)

    def _on_label_mouse_press(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.tag_clicked.emit(self.item_id, self.label_text)
            event.accept()
            return
        elif event.button() == Qt.RightButton:
            event.ignore()
            return
        super().mousePressEvent(event)

    def contextMenuEvent(self, event: Any) -> None:
        menu = QMenu(self)
        act_remove = menu.addAction("Remove tag")
        act_remove.triggered.connect(lambda: self.remove_clicked.emit(self.item_id))
        if self.total_selection_count > 1:
            act_assign = menu.addAction("Assign to all")
            if self.tag_count >= self.total_selection_count:
                act_assign.setEnabled(False)
            else:
                act_assign.triggered.connect(lambda: self.assign_to_all_clicked.emit(self.item_id))
        menu.exec(event.globalPos())
        event.accept()

    def enterEvent(self, event: Any) -> None:
        try:
            super().enterEvent(event)
        except TypeError:
            pass
        self.setProperty("hovered", "true")
        self.style().unpolish(self)
        self.style().polish(self)
        self.btn_del.show()

    def leaveEvent(self, event: Any) -> None:
        try:
            super().leaveEvent(event)
        except TypeError:
            pass
        self.setProperty("hovered", "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.btn_del.hide()

    def set_dark_mode(self, is_dark: bool) -> None:
        if is_dark:
            self.setStyleSheet(
                """
                QFrame#DescriptiveTagRow {
                    background: transparent;
                    border: 1px solid transparent;
                    border-radius: 2px;
                }
                QFrame#DescriptiveTagRow[hovered="true"] {
                    background: #242A36;
                    border: 1px solid #3B82F6;
                }
                QLabel#TagLabel {
                    color: #60A5FA;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    padding-left: 2px;
                    background: transparent;
                    border: none;
                }
                QLabel#TagLabel:hover {
                    text-decoration: underline;
                    color: #93C5FD;
                }
                QLabel#TagCountLabel {
                    color: #94A3B8;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    background: transparent;
                    border: none;
                }
                QToolButton#DeleteBtn {
                    border: none;
                    background: transparent;
                    color: #94A3B8;
                    font-family: 'Segoe UI';
                    font-size: 8pt;
                    padding: 0px 2px;
                }
                QToolButton#DeleteBtn:hover {
                    color: #EF4444;
                    background: rgba(255, 255, 255, 20);
                    border-radius: 2px;
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                QFrame#DescriptiveTagRow {
                    background: transparent;
                    border: 1px solid transparent;
                    border-radius: 2px;
                }
                QFrame#DescriptiveTagRow[hovered="true"] {
                    background: #F0F6FC;
                    border: 1px solid #7DA2CE;
                }
                QLabel#TagLabel {
                    color: #0066CC;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    padding-left: 2px;
                    background: transparent;
                    border: none;
                }
                QLabel#TagLabel:hover {
                    text-decoration: underline;
                    color: #004499;
                }
                QLabel#TagCountLabel {
                    color: #555555;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    background: transparent;
                    border: none;
                }
                QToolButton#DeleteBtn {
                    border: none;
                    background: transparent;
                    color: #4C4C4C;
                    font-family: 'Segoe UI';
                    font-size: 8pt;
                    padding: 0px 2px;
                }
                QToolButton#DeleteBtn:hover {
                    color: #C50F1F;
                    background: rgba(0, 0, 0, 15);
                    border-radius: 2px;
                }
                """
            )


class _ClickToEditLabel(QLabel):
    """Clean plain text label that emits clicked when pressed."""

    clicked = Signal()

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setWordWrap(True)
        self.setStyleSheet(
            """
            QLabel {
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                padding: 1px 4px;
                border: 1px solid transparent;
                border-radius: 2px;
            }
            QLabel:hover {
                background: #F4F8FC;
                border: 1px solid #C5D9EC;
            }
            """
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class _ClickToEditWidget(QStackedWidget):
    """
    Displays a clean plain text label (no edit borders, spin buttons, or arrows)
    until clicked on, seamlessly switching to the editor widget, and returning
    to the plain text label when editing completes or focus is lost.
    """

    def __init__(
        self,
        editor_widget: QWidget,
        parent: QWidget | None = None,
        editable: bool = True,
        empty_placeholder: str = "-",
    ) -> None:
        super().__init__(parent)
        self.editor_widget = editor_widget
        self.empty_placeholder = empty_placeholder
        self._editable = editable

        self.view_label = _ClickToEditLabel(empty_placeholder, self)
        self.view_label.clicked.connect(self.start_edit)

        self.addWidget(self.view_label)     # index 0
        self.addWidget(self.editor_widget)  # index 1
        self.setCurrentIndex(0)

        # Monitor focus lost across the editor widget and all its children
        self._install_focus_filter(self.editor_widget)

    def _install_focus_filter(self, w: QWidget) -> None:
        w.installEventFilter(self)
        for child in w.findChildren(QWidget):
            child.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if self.currentIndex() == 1 and event.type() == QEvent.FocusOut:
            QTimer.singleShot(120, self._check_focus_loss)
        return super().eventFilter(watched, event)

    def _check_focus_loss(self) -> None:
        if self.currentIndex() == 1:
            fw = QApplication.focusWidget()
            if fw is not None and (fw == self.editor_widget or self.editor_widget.isAncestorOf(fw)):
                return
            self.finish_edit()

    def set_editable(self, editable: bool) -> None:
        self._editable = editable
        is_dark = getattr(self, "_is_dark", False)
        if not editable:
            self.view_label.setCursor(Qt.ArrowCursor)
            self.view_label.setStyleSheet(
                f"""
                QLabel {{
                    color: {'#94A3B8' if is_dark else '#55687C'};
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    padding: 1px 4px;
                    border: 1px solid transparent;
                }}
                """
            )
        else:
            self.view_label.setCursor(Qt.PointingHandCursor)
            self.view_label.setStyleSheet(
                f"""
                QLabel {{
                    color: {'#E2E8F0' if is_dark else '#1E395B'};
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    padding: 1px 4px;
                    border: 1px solid transparent;
                    border-radius: 2px;
                }}
                QLabel:hover {{
                    background: {'#242A36' if is_dark else '#F4F8FC'};
                    border: 1px solid {'#3B82F6' if is_dark else '#C5D9EC'};
                }}
                """
            )

    def set_dark_mode(self, is_dark: bool) -> None:
        self._is_dark = bool(is_dark)
        self.set_editable(self._editable)

    def set_display_text(self, text: str) -> None:
        val = text.strip() if text else ""
        if not val:
            self.view_label.setText(self.empty_placeholder)
        else:
            self.view_label.setText(val)

    def start_edit(self) -> None:
        if not self._editable:
            return
        self.setCurrentIndex(1)
        self.editor_widget.setFocus()
        if isinstance(self.editor_widget, QLineEdit):
            self.editor_widget.selectAll()

    def finish_edit(self, updated_text: str | None = None) -> None:
        if updated_text is not None:
            self.set_display_text(updated_text)
        self.setCurrentIndex(0)


class _StarButton(QPushButton):
    """Interactive star button with hover tracking and WLPG gold / blue outline."""

    def __init__(self, star_index: int, parent: _StarRatingHoverWidget) -> None:
        super().__init__(parent)
        self.star_index = star_index
        self._rating_widget = parent
        self.setFixedSize(18, 18)
        self.setCursor(Qt.PointingHandCursor)
        self.setFlat(True)
        self.setToolTip("Click a star to assign a rating. (0 - 5)")
        self.clicked.connect(lambda: self._rating_widget.on_star_clicked(self.star_index))

    def enterEvent(self, event: Any) -> None:
        super().enterEvent(event)
        self._rating_widget.set_hover_star(self.star_index)

    def leaveEvent(self, event: Any) -> None:
        super().leaveEvent(event)
        self._rating_widget.set_hover_star(0)

    def paintEvent(self, _event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        cx = self.width() / 2.0
        cy = self.height() / 2.0
        outer_r = 7.5
        inner_r = 3.2

        poly = QPolygonF([
            QPointF(
                cx + (outer_r if i % 2 == 0 else inner_r) * math.cos(math.radians(-90.0 + i * 36.0)),
                cy + (outer_r if i % 2 == 0 else inner_r) * math.sin(math.radians(-90.0 + i * 36.0)),
            )
            for i in range(10)
        ])

        hover_star = self._rating_widget.hover_star
        curr_rating = self._rating_widget.rating

        if hover_star > 0:
            if self.star_index <= hover_star and self.star_index <= curr_rating:
                # Stays active rating: solid gold
                p.setPen(QPen(QColor("#D69E00"), 1.0))
                p.setBrush(QBrush(QColor("#F5B800")))
            elif hover_star < self.star_index <= curr_rating:
                # Would be removed: faded translucent yellow to indicate removal
                p.setPen(QPen(QColor(214, 158, 0, 110), 1.0))
                p.setBrush(QBrush(QColor(245, 184, 0, 65)))
            elif curr_rating < self.star_index <= hover_star:
                # Would be added: crisp golden hover outline
                p.setPen(QPen(QColor("#E5A800"), 1.4))
                p.setBrush(Qt.NoBrush)
            else:
                # Inactive rating
                p.setPen(QPen(QColor("#D0DCE8"), 1.0))
                p.setBrush(Qt.NoBrush)
        else:
            if self.star_index <= curr_rating:
                # Active rating: solid golden fill
                p.setPen(QPen(QColor("#D69E00"), 1.0))
                p.setBrush(QBrush(QColor("#F5B800")))
            else:
                # Inactive rating: hollow slate outline
                p.setPen(QPen(QColor("#B2C7DE"), 1.0))
                p.setBrush(Qt.NoBrush)

        p.drawPolygon(poly)
        p.end()


class _StarRatingHoverWidget(QWidget):
    """5-star interactive rating widget with hover outline preview and tooltip."""

    rating_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.rating = 0
        self.hover_star = 0
        self.setToolTip("Click a star to assign a rating. (0 - 5)")

        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(1)

        self.buttons: list[_StarButton] = []
        for i in range(1, 6):
            btn = _StarButton(i, self)
            hbox.addWidget(btn)
            self.buttons.append(btn)
        hbox.addStretch(1)

    def set_hover_star(self, star_idx: int) -> None:
        if self.hover_star != star_idx:
            self.hover_star = star_idx
            for btn in self.buttons:
                btn.update()

    def set_rating(self, rating: int) -> None:
        self.rating = max(0, min(5, int(rating)))
        for btn in self.buttons:
            btn.update()

    def on_star_clicked(self, star_idx: int) -> None:
        new_rating = 0 if self.rating == star_idx else star_idx
        self.set_rating(new_rating)
        self.rating_changed.emit(new_rating)

    def leaveEvent(self, event: Any) -> None:
        super().leaveEvent(event)
        self.set_hover_star(0)


class _DescriptiveTagInput(QLineEdit):
    """
    Descriptive tag line edit that lets the user type immediately while showing
    most recently used tags on click/Ctrl+T, and filtering all tags as they type.
    """

    tag_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._recent_tags: list[str] = []
        self._all_tags: list[str] = []
        self._model = QStringListModel(self)
        self._completer = QCompleter(self._model, self)
        self._completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchContains)
        self._completer.setCompletionMode(QCompleter.PopupCompletion)
        self.setCompleter(self._completer)
        self.textEdited.connect(self._on_text_edited)

    def set_tag_lists(
        self, all_tags: Sequence[str], recent_tags: Sequence[str] | None = None
    ) -> None:
        self._all_tags = list(dict.fromkeys(str(t) for t in all_tags if str(t).strip()))
        if recent_tags is not None:
            self._recent_tags = list(
                dict.fromkeys(str(t) for t in recent_tags if str(t).strip())
            )[:12]
        elif not self._recent_tags and self._all_tags:
            self._recent_tags = list(self._all_tags[:10])
        self._sync_model_for_text(self.text())

    def set_recent_tags(self, tags: Sequence[str]) -> None:
        self._recent_tags = list(
            dict.fromkeys(str(t) for t in tags if str(t).strip())
        )[:12]
        self._sync_model_for_text(self.text())

    def record_used_tag(self, tag: str) -> None:
        clean = str(tag).strip()
        if not clean:
            return
        self._recent_tags = [clean] + [
            t for t in self._recent_tags if t.lower() != clean.lower()
        ]
        self._recent_tags = self._recent_tags[:12]
        if all(t.lower() != clean.lower() for t in self._all_tags):
            self._all_tags.insert(0, clean)

    def _sync_model_for_text(self, text: str) -> None:
        val = text.strip()
        if not val:
            items = self._recent_tags if self._recent_tags else self._all_tags[:10]
            self._model.setStringList(items)
            self._completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
            self._completer.setCompletionPrefix("")
        else:
            # Ordered filtered list: leaf-prefix matches first, then path-prefix, then substring matches (each alphabetical)
            q = val.lower()
            all_unique = list(
                dict.fromkeys(
                    list(self._all_tags)
                    + [t for t in self._recent_tags if t.lower() not in {a.lower() for a in self._all_tags}]
                )
            )
            matched = [t for t in all_unique if q in t.lower()]

            def _sort_key(tag_str: str) -> tuple[int, str, str]:
                low = tag_str.lower()
                leaf = low.rsplit("/", 1)[-1]
                if leaf.startswith(q):
                    rank = 0
                elif low.startswith(q):
                    rank = 1
                else:
                    rank = 2
                return (rank, leaf, low)

            matched.sort(key=_sort_key)
            self._model.setStringList(matched)
            self._completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
            self._completer.setCompletionPrefix("")

    def _on_text_edited(self, text: str) -> None:
        self._sync_model_for_text(text)
        if self._model.rowCount() > 0:
            self._completer.complete()
        elif self._completer.popup() is not None:
            self._completer.popup().hide()

    def show_suggestions_popup(self) -> None:
        self._sync_model_for_text(self.text())
        if self._model.rowCount() > 0:
            self._completer.complete()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            QTimer.singleShot(0, self.show_suggestions_popup)

    def keyPressEvent(self, event: Any) -> None:
        if event.key() == Qt.Key_Down:
            popup = self._completer.popup()
            if popup is None or not popup.isVisible():
                self.show_suggestions_popup()
                return
        super().keyPressEvent(event)


def _make_cross_icon(size: int = 16) -> QIcon:
    """Creates a clean subtle '✕' icon matching DescriptiveTagRowWidget."""
    icon = QIcon()
    for mode, col in [(QIcon.Normal, "#4C4C4C"), (QIcon.Active, "#C50F1F"), (QIcon.Selected, "#C50F1F")]:
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(QColor(col), 1.25, Qt.SolidLine, Qt.RoundCap))
        m = size * 0.30
        p.drawLine(QPointF(m, m), QPointF(size - m, size - m))
        p.drawLine(QPointF(size - m, m), QPointF(m, size - m))
        p.end()
        icon.addPixmap(pm, mode)
    return icon


class _GeotagInput(QLineEdit):
    """
    WLPG-style Geotag input field featuring sub-2ms gazetteer autocomplete,
    a 'Searching...' status indicator, background Nominatim lookup, and [✕] clear action.
    """

    place_selected = Signal(str, float, float)  # formatted_name, lat, lon
    geotag_cleared = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("InlineEdit")
        self.setPlaceholderText("Add geotag")
        self.lbl_status: QLabel | None = None
        self._db_engine: Any = None
        self._current_places: list[dict[str, Any]] = []

        self._model = QStringListModel(self)
        self._completer = QCompleter(self._model, self)
        self._completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchContains)
        self._completer.setCompletionMode(QCompleter.PopupCompletion)
        self._completer.activated[str].connect(self._on_completer_activated)
        self.setCompleter(self._completer)

        popup = self._completer.popup()
        if popup is not None:
            popup.setStyleSheet("""
                QListView {
                    border: 1px solid #0066CC;
                    background-color: #FFFFFF;
                    color: #1E395B;
                    font-size: 11px;
                    padding: 2px;
                    outline: none;
                }
                QListView::item {
                    padding: 5px 8px;
                    border-radius: 2px;
                }
                QListView::item:hover {
                    background-color: #EBF3FB;
                    color: #0066CC;
                }
                QListView::item:selected {
                    background-color: #0066CC;
                    color: #FFFFFF;
                }
            """)

        self.act_clear = self.addAction(
            _make_cross_icon(16),
            QLineEdit.TrailingPosition,
        )
        self.act_clear.setToolTip("Remove geotag")
        self.act_clear.setVisible(False)
        self.act_clear.triggered.connect(self.clear_geotag)

        self.textEdited.connect(self._on_text_edited)
        self.textChanged.connect(self._on_text_changed)

        self._online_timer = QTimer(self)
        self._online_timer.setSingleShot(True)
        self._online_timer.setInterval(350)
        self._online_timer.timeout.connect(self._trigger_online_search)

    def setText(self, text: str) -> None:
        super().setText(text)
        self.setCursorPosition(0)
        self.setToolTip(text if text.strip() else "")

    def focusOutEvent(self, event: Any) -> None:
        super().focusOutEvent(event)
        self.setCursorPosition(0)

    def enterEvent(self, event: Any) -> None:
        super().enterEvent(event)
        if bool(self.text().strip()):
            self.act_clear.setVisible(True)

    def leaveEvent(self, event: Any) -> None:
        super().leaveEvent(event)
        self.act_clear.setVisible(False)

    def _on_text_changed(self, text: str) -> None:
        if not text.strip():
            self.act_clear.setVisible(False)
        elif self.underMouse():
            self.act_clear.setVisible(True)

    def _on_text_edited(self, text: str) -> None:
        clean = text.strip()
        if not clean:
            if self.lbl_status is not None:
                self.lbl_status.setVisible(False)
            self._model.setStringList([])
            if self._completer.popup() is not None:
                self._completer.popup().hide()
            return

        if self.lbl_status is not None:
            self.lbl_status.setText("Searching...")
            self.lbl_status.setVisible(True)

        places = search_places_offline(clean, limit=12, db_engine=self._db_engine)
        self._current_places = places
        items = [p["formatted_address"] for p in places]
        self._model.setStringList(items)

        if items:
            self._completer.complete()

        # If few local results and query >= 3 chars, trigger background Nominatim
        if len(places) < 3 and len(clean) >= 3:
            self._online_timer.start(350)
        else:
            if self.lbl_status is not None:
                self.lbl_status.setVisible(False)

    def _trigger_online_search(self) -> None:
        query = self.text().strip()
        if len(query) < 3:
            if self.lbl_status is not None:
                self.lbl_status.setVisible(False)
            return

        import threading
        def _fetch():
            try:
                results = search_places_online(query, db_engine=self._db_engine, timeout=2.0, limit=6)
                if results and self.text().strip().lower() == query.lower():
                    QTimer.singleShot(0, lambda: self._apply_online_results(results))
                else:
                    QTimer.singleShot(0, lambda: self.lbl_status.setVisible(False) if self.lbl_status else None)
            except Exception:
                if self.lbl_status is not None:
                    QTimer.singleShot(0, lambda: self.lbl_status.setVisible(False))

        threading.Thread(target=_fetch, daemon=True).start()

    def _apply_online_results(self, new_places: list[dict[str, Any]]) -> None:
        if self.lbl_status is not None:
            self.lbl_status.setVisible(False)
        existing_fmts = {p["formatted_address"] for p in self._current_places}
        added = False
        for p in new_places:
            fmt = p["formatted_address"]
            if fmt not in existing_fmts:
                self._current_places.append(p)
                existing_fmts.add(fmt)
                added = True
        if added:
            items = [p["formatted_address"] for p in self._current_places]
            self._model.setStringList(items)
            self._completer.complete()

    def _on_completer_activated(self, text: str) -> None:
        val = text.strip()
        if self.lbl_status is not None:
            self.lbl_status.setVisible(False)
        matched = next((p for p in self._current_places if p["formatted_address"].lower() == val.lower()), None)
        if matched is None:
            lat, lon, fmt = resolve_location_query(val, allow_network=False, db_engine=self._db_engine)
        else:
            lat, lon, fmt = matched["lat"], matched["lon"], matched["formatted_address"]

        self.setText(fmt or val)
        if lat is not None and lon is not None:
            self.place_selected.emit(fmt or val, float(lat), float(lon))

    def clear_geotag(self) -> None:
        self.clear()
        self.act_clear.setVisible(False)
        if self.lbl_status is not None:
            self.lbl_status.setVisible(False)
        self.geotag_cleared.emit()

    def contextMenuEvent(self, event: Any) -> None:
        menu = self.createStandardContextMenu()
        if self.text().strip():
            menu.addSeparator()
            act_rm = menu.addAction("Remove geotag")
            act_rm.triggered.connect(self.clear_geotag)
        menu.exec(event.globalPos())


class _FineTuneGroup(QFrame):
    """Rounded card group box for Fine-Tune controls matching media_1792165772919_087b03d2.png."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("FineTuneGroup")
        self.setStyleSheet(
            """
            QFrame#FineTuneGroup {
                background: #FFFFFF;
                border: 1px solid #D2DFEE;
                border-radius: 4px;
            }
            """
        )
        self.vbox = QVBoxLayout(self)
        self.vbox.setContentsMargins(8, 6, 8, 8)
        self.vbox.setSpacing(4)

        lbl = QLabel(title, self)
        lbl.setStyleSheet(
            "color: #29588C; font-family: 'Segoe UI'; font-size: 9.5pt; font-weight: 600; border: none;"
        )
        self.vbox.addWidget(lbl)


class _NoHorizontalScrollArea(QScrollArea):
    """QScrollArea that strictly suppresses horizontal scrolling and horizontal arrow key events."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.horizontalScrollBar().setEnabled(False)

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        # Strictly suppress horizontal scrolling
        super().scrollContentsBy(0, dy)

    def keyPressEvent(self, event: Any) -> None:
        if event.key() in (Qt.Key_Left, Qt.Key_Right):
            event.ignore()
            return
        super().keyPressEvent(event)


class WLPGSplitterHandle(QSplitterHandle):
    """
    Splitter handle styled like Windows Live Photo Gallery:
    - Subtle horizontal gradient line
    - Centered collapse/expand glyph (▽ / △)
    - Clicking the glyph or double-clicking toggles collapse of the bottom Information pane
    """

    def __init__(self, orientation: Qt.Orientation, parent: QSplitter) -> None:
        super().__init__(orientation, parent)
        self.setFixedHeight(12)
        self.setCursor(Qt.SplitVCursor)
        self._collapsed = False
        self._last_bottom_size = 220
        self._is_dark = False

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w = float(self.width())
        h = float(self.height())
        mid_y = h / 2.0

        is_dark = getattr(self, "_is_dark", False)
        # Background matches pane
        p.fillRect(0, 0, int(w), int(h), QColor("#1B1E24" if is_dark else "#FFFFFF"))

        # Gradient separator line across width
        grad = QLinearGradient(0, mid_y, w, mid_y)
        if is_dark:
            grad.setColorAt(0.0, QColor("#1B1E24"))
            grad.setColorAt(0.2, QColor("#2D3542"))
            grad.setColorAt(0.5, QColor("#475569"))
            grad.setColorAt(0.8, QColor("#2D3542"))
            grad.setColorAt(1.0, QColor("#1B1E24"))
        else:
            grad.setColorAt(0.0, QColor("#E2ECF7"))
            grad.setColorAt(0.2, QColor("#B7D1EE"))
            grad.setColorAt(0.5, QColor("#7CAEDB"))
            grad.setColorAt(0.8, QColor("#B7D1EE"))
            grad.setColorAt(1.0, QColor("#E2ECF7"))
        p.setPen(QPen(grad, 1))
        p.drawLine(QPointF(4, mid_y), QPointF(w - 4, mid_y))

        # Center collapse glyph button background (small oval/pill)
        center_x = w / 2.0
        p.setPen(QPen(QColor("#3E4756" if is_dark else "#B7D1EE"), 1))
        p.setBrush(QColor("#22262E" if is_dark else "#FFFFFF"))
        p.drawRoundedRect(QRectF(center_x - 12, mid_y - 5, 24, 10), 5, 5)

        # Center triangle arrow: ▽ when expanded, △ when collapsed
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#94A3B8" if is_dark else "#4E759B"))
        if not self._collapsed:
            poly = QPolygonF([
                QPointF(center_x - 4, mid_y - 2),
                QPointF(center_x + 4, mid_y - 2),
                QPointF(center_x, mid_y + 3),
            ])
        else:
            poly = QPolygonF([
                QPointF(center_x - 4, mid_y + 2),
                QPointF(center_x + 4, mid_y + 2),
                QPointF(center_x, mid_y - 3),
            ])
        p.drawPolygon(poly)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        self.toggle_collapse()

    def mousePressEvent(self, event: Any) -> None:
        if abs(event.position().x() - self.width() / 2.0) <= 16:
            self.toggle_collapse()
        else:
            super().mousePressEvent(event)

    def _on_splitter_moved(self) -> None:
        splitter = self.splitter()
        if splitter:
            sizes = splitter.sizes()
            if len(sizes) == 2:
                self._collapsed = (sizes[1] <= 35)
                self.update()

    def toggle_collapse(self) -> None:
        splitter = self.splitter()
        if splitter:
            sizes = splitter.sizes()
            if len(sizes) == 2:
                if sizes[1] > 35:
                    self._last_bottom_size = sizes[1]
                    splitter.setSizes([sizes[0] + sizes[1] - 28, 28])
                    self._collapsed = True
                else:
                    restored = max(140, self._last_bottom_size)
                    splitter.setSizes([max(100, sizes[0] + sizes[1] - restored), restored])
                    self._collapsed = False
                self.update()


class WLPGSplitter(QSplitter):
    """Vertical QSplitter using authentic WLPG handle with gradient line and collapse glyph."""

    def createHandle(self) -> QSplitterHandle:
        handle = WLPGSplitterHandle(self.orientation(), self)
        self.splitterMoved.connect(lambda pos, index: handle._on_splitter_moved())
        return handle


class TagAndCaptionDock(QDockWidget):
    """
    Unified Right Sidebar for PhotoGallery64:
    - Mode 'info': Tag and Caption / Information (Ctrl+I)
    - Mode 'fine_tune': Fine-Tune & Adjustments with Live Histogram (Ctrl+F)
    """

    # Signals for information editing
    rating_edited = Signal(int)             # 0..5
    flag_edited = Signal(int)               # 0 or 1
    caption_committed = Signal(str)         # new caption string
    tag_added = Signal(str)                 # full_path or tag name
    tag_removed = Signal(int)               # tag_id
    person_added = Signal(str)              # person name
    person_removed = Signal(int)            # person_id
    geotag_edited = Signal(str)             # location_name
    change_time_requested = Signal()        # user clicked Date Taken link
    filename_committed = Signal(str)        # renamed filename
    date_taken_edited = Signal(int)         # new timestamp (epoch seconds)
    author_committed = Signal(str)          # author string
    tag_filter_requested = Signal(int)      # tag_id clicked to filter gallery
    tag_assign_to_all_requested = Signal(int) # tag_id to assign to all selected items
    person_hovered = Signal(int, bool)      # (person_id, hovered)
    person_search_requested = Signal(int, str)  # (person_id, person_name)
    person_view_suggestions_requested = Signal(int)  # person_id
    person_tag_as_other_requested = Signal(int)  # person_id
    person_set_display_pic_requested = Signal(int)  # person_id
    person_merge_requested = Signal(int, str)  # (person_id, person_name)
    person_rename_requested = Signal(int, str)  # (person_id, person_name)
    person_confirm_requested = Signal(int, int)  # (person_id, face_id)
    person_reject_suggestion_requested = Signal(int, int)  # (person_id, face_id)
    person_tag_mode_requested = Signal()    # user clicked 'Add people tags'
    face_region_hovered = Signal(object)    # face_id / region_id: int | None
    assign_face_requested = Signal(int, str) # (face_id, person_name)
    dismiss_face_requested = Signal(int)     # (face_id)
    remove_face_requested = Signal(int)      # (face_id)
    person_locate_face_requested = Signal(int, str)  # (person_id, person_name)
    person_link_face_requested = Signal(int, int)    # (person_id, face_id)

    # Signals for Fine-Tune adjustments
    fine_tune_exposure_changed = Signal(int, int, int, int, int)   # brightness, contrast, shadows, highlights, clarity
    fine_tune_color_changed = Signal(int, int, int, int)           # temp, tint, saturation, dehaze
    fine_tune_straighten_changed = Signal(float)                   # angle
    fine_tune_detail_changed = Signal(int, int)                    # sharpen, denoise
    fine_tune_optics_changed = Signal(int)                         # vignette
    fine_tune_levels_changed = Signal(int, int)                    # black_point, white_point
    fine_tune_reset_requested = Signal()                           # reset all adjustments to default
    fine_tune_auto_adjust = Signal()
    auto_straighten_requested = Signal()
    analyze_noise_requested = Signal()
    compare_started = Signal()
    compare_ended = Signal()
    fine_tune_slider_pressed = Signal()
    fine_tune_slider_released = Signal(str)

    # Signals for Depth & Light adjustments
    depth_bokeh_changed = Signal(int, int, int, int, int)    # blur, focal_plane, glow, boundary_shift, boundary_feather
    depth_relight_changed = Signal(int, int, int, int)       # near, far, warmth_near, warmth_far
    depth_haze_changed = Signal(int, int)                    # haze, haze_warmth
    depth_reset_requested = Signal()
    copy_subject_requested = Signal()
    remove_background_requested = Signal()
    mask_preview_toggled = Signal(bool)                      # live ruby mask overlay
    map_location_selected = Signal(list, str, float, float)  # media_ids, label, lat, lon

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Tag and caption", parent)
        self.setObjectName("TagAndCaptionDock")
        self.setFeatures(
            QDockWidget.DockWidgetClosable
        )
        self.setTitleBarWidget(QWidget(self))  # Hide default ugly dock titlebar
        self.setMinimumWidth(220)
        self.setMaximumWidth(600)

        self._current_mode: Literal["info", "fine_tune", "depth_light"] = "info"
        self._selected_ids: list[int] = []
        self._current_rating: int = 0
        self._current_flag: int = 0
        self._total_view_count: int = 0
        self._updating_ui: bool = False
        self._original_filename: str = ""
        self._current_date_taken_ts: int = 0

        # Real-time event throttler/coalescer to maintain responsive 60 FPS interactions
        self._dl_timer = QTimer(self)
        self._dl_timer.setSingleShot(True)
        self._dl_timer.setInterval(16)
        self._dl_timer.timeout.connect(self._flush_depth_light_emits)
        self._pending_dl_bokeh = False
        self._pending_dl_relight = False
        self._pending_dl_haze = False

        self._ft_timer = QTimer(self)
        self._ft_timer.setSingleShot(True)
        self._ft_timer.setInterval(16)
        self._ft_timer.timeout.connect(self._flush_fine_tune_emits)
        self._pending_ft_exposure = False
        self._pending_ft_color = False
        self._pending_ft_optics = False
        self._pending_ft_straighten = False
        self._pending_ft_detail = False

        # Outer Container & Header Row with close '✕' button
        self.container = QFrame(self)
        self.container.setObjectName("RightSidebarContainer")
        self.container.setStyleSheet(
            """
            QFrame#RightSidebarContainer {
                background-color: #FFFFFF;
                border-left: 1px solid #D2DFEE;
            }
            """
        )
        outer_vbox = QVBoxLayout(self.container)
        outer_vbox.setContentsMargins(0, 0, 0, 0)
        outer_vbox.setSpacing(0)

        # Subtle Header Row with small '✕' button in top right (matches WLPG screenshots)
        header_row = QWidget(self.container)
        header_row.setFixedHeight(24)
        h_layout = QHBoxLayout(header_row)
        h_layout.setContentsMargins(8, 4, 6, 0)
        h_layout.addStretch(1)

        self.btn_close_dock = QPushButton("✕", header_row)
        self.btn_close_dock.setFixedSize(18, 18)
        self.btn_close_dock.setCursor(Qt.PointingHandCursor)
        self.btn_close_dock.setToolTip("Close pane (Esc)")
        self.btn_close_dock.setStyleSheet(
            """
            QPushButton {
                border: none;
                background: transparent;
                color: #7B8B9B;
                font-family: 'Segoe UI';
                font-size: 9pt;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #C50F1F;
                background: #F8D7DA;
                border-radius: 2px;
            }
            """
        )
        self.btn_close_dock.clicked.connect(self.hide)
        h_layout.addWidget(self.btn_close_dock)
        outer_vbox.addWidget(header_row)

        # Mode Stack: Page 0 = Information & Tags, Page 1 = Fine-Tune, Page 2 = Depth & Light
        self.mode_stack = QStackedWidget(self.container)

        self.page_info = self._build_info_page()
        self.page_fine_tune = self._build_fine_tune_page()
        self.page_depth_light = self._build_depth_light_page()

        self.mode_stack.addWidget(self.page_info)
        self.mode_stack.addWidget(self.page_fine_tune)
        self.mode_stack.addWidget(self.page_depth_light)
        outer_vbox.addWidget(self.mode_stack, 1)

        self.setWidget(self.container)
        self._apply_stylesheet()

    def sizeHint(self) -> QSize:
        return QSize(295, 600) if getattr(self, "_current_mode", "info") in ("fine_tune", "depth_light") else QSize(280, 600)

    def minimumSizeHint(self) -> QSize:
        return QSize(220, 300)

    def keyPressEvent(self, event: Any) -> None:
        if event.key() in (Qt.Key_Left, Qt.Key_Right):
            focus_w = self.focusWidget() or QApplication.focusWidget()
            if isinstance(focus_w, (QLineEdit, QTextEdit, QPlainTextEdit)):
                super().keyPressEvent(event)
                return
            event.ignore()
            return
        super().keyPressEvent(event)

    def set_mode(self, mode: Literal["info", "fine_tune", "depth_light"]) -> None:
        """Switches between Tag & Caption (Ctrl+I), Fine-Tune (Ctrl+F), and Depth & Light."""
        self._current_mode = mode
        if mode == "depth_light":
            self.mode_stack.setCurrentIndex(2)
        elif mode == "fine_tune":
            self.mode_stack.setCurrentIndex(1)
        else:
            self.mode_stack.setCurrentIndex(0)
        self.show()

    def focus_filename_rename(self) -> None:
        """Switches to Information mode, reveals dock, and focuses filename input with select-all."""
        self.set_mode("info")
        self.show()
        self.raise_()
        if hasattr(self, "stack_filename") and hasattr(self, "txt_filename"):
            self.stack_filename.start_edit()
            self.txt_filename.setFocus()
            self.txt_filename.selectAll()

    def current_mode(self) -> str:
        return self._current_mode

    def bind_database_engine(self, engine: Any) -> None:
        self._db_engine = engine
        if hasattr(self, "txt_geotag") and isinstance(self.txt_geotag, _GeotagInput):
            self.txt_geotag._db_engine = engine

    # -------------------------------------------------------------------------
    # Page 1: Information & Tag Pane (docs/reference_ui/main_ui.png & media_1792165701201_517fc7f1.png)
    # -------------------------------------------------------------------------
    def _build_info_page(self) -> QWidget:
        page_widget = QWidget()
        page_widget.setObjectName("InfoPage")
        page_widget.setStyleSheet("background-color: #FFFFFF;")
        page_layout = QVBoxLayout(page_widget)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        self.info_splitter = WLPGSplitter(Qt.Vertical, page_widget)
        self.info_splitter.setObjectName("InfoSplitter")
        self.info_splitter.setStyleSheet("background-color: #FFFFFF;")
        self.info_splitter.setChildrenCollapsible(True)

        # 1. Top Section: Tagging scroll area (People, Geotag, Caption, Descriptive Tags)
        top_scroll = _NoHorizontalScrollArea(self.info_splitter)
        top_scroll.setObjectName("TopScroll")
        top_scroll.setWidgetResizable(True)
        top_scroll.setFrameShape(QFrame.NoFrame)
        top_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        top_scroll.setStyleSheet("background-color: #FFFFFF; border: none;")

        top_content = QWidget(top_scroll)
        top_content.setStyleSheet("background-color: #FFFFFF;")
        top_vbox = QVBoxLayout(top_content)
        top_vbox.setContentsMargins(10, 2, 10, 8)
        top_vbox.setSpacing(6)

        # 0. Locations in View (shown in World Map View)
        self.locations_in_view_container = QWidget(top_content)
        loc_vbox = QVBoxLayout(self.locations_in_view_container)
        loc_vbox.setContentsMargins(0, 0, 0, 4)
        loc_vbox.setSpacing(4)

        lbl_locations = QLabel("Locations in this view", self.locations_in_view_container)
        lbl_locations.setObjectName("SectionTitle")
        loc_vbox.addWidget(lbl_locations)

        self.list_locations_in_view = QListWidget(self.locations_in_view_container)
        self.list_locations_in_view.setFixedHeight(140)
        self.list_locations_in_view.setStyleSheet(
            "QListWidget { border: 1px solid #C8D2E0; border-radius: 2px; background: #FFFFFF; font-family: 'Segoe UI'; font-size: 9pt; } "
            "QListWidget::item { padding: 3px 6px; } "
            "QListWidget::item:hover { background-color: #E5F3FF; } "
            "QListWidget::item:selected { background-color: #CCE8FF; color: #000000; font-weight: 600; }"
        )
        self.list_locations_in_view.itemClicked.connect(self._on_location_in_view_clicked)
        loc_vbox.addWidget(self.list_locations_in_view)

        sep = QFrame(self.locations_in_view_container)
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #E2E8F0;")
        loc_vbox.addWidget(sep)

        self.locations_in_view_container.hide()
        top_vbox.addWidget(self.locations_in_view_container)

        # 1. People tags (Add people tags box above the tagged people list, matching media_1792269887974_67460900.png)
        lbl_people = QLabel("People tags", top_content)
        lbl_people.setObjectName("SectionTitle")
        top_vbox.addWidget(lbl_people)

        self.txt_add_person = _AddPersonLineEdit(top_content)
        self.txt_add_person.setObjectName("InlineEdit")
        self.txt_add_person.setPlaceholderText("Add people tags")
        self.txt_add_person.returnPressed.connect(self._on_add_person_return)
        self.txt_add_person.clicked.connect(self.person_tag_mode_requested.emit)
        self._people_model = QStringListModel(self)
        self._people_completer = QCompleter(self._people_model, self)
        self._people_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._people_completer.setFilterMode(Qt.MatchContains)
        self._people_completer.activated[str].connect(self._on_person_completer_activated)
        self.txt_add_person.setCompleter(self._people_completer)
        top_vbox.addWidget(self.txt_add_person)

        self.people_chips_container = QWidget(top_content)
        self.people_chips_layout = QVBoxLayout(self.people_chips_container)
        self.people_chips_layout.setContentsMargins(0, 0, 0, 0)
        self.people_chips_layout.setSpacing(3)
        top_vbox.addWidget(self.people_chips_container)

        # 2. Geotag (with Searching... status label and autocomplete)
        geo_hdr_row = QWidget(top_content)
        geo_hdr_layout = QHBoxLayout(geo_hdr_row)
        geo_hdr_layout.setContentsMargins(0, 0, 0, 0)
        geo_hdr_layout.setSpacing(4)

        lbl_geo = QLabel("Geotag", geo_hdr_row)
        lbl_geo.setObjectName("SectionTitle")
        geo_hdr_layout.addWidget(lbl_geo)

        self.lbl_geotag_status = QLabel("", geo_hdr_row)
        self.lbl_geotag_status.setStyleSheet("color: #718096; font-size: 11px; font-style: italic;")
        self.lbl_geotag_status.setVisible(False)
        geo_hdr_layout.addStretch(1)
        geo_hdr_layout.addWidget(self.lbl_geotag_status)
        top_vbox.addWidget(geo_hdr_row)

        self.txt_geotag = _GeotagInput(top_content)
        self.txt_geotag.setObjectName("InlineEdit")
        self.txt_geotag.lbl_status = self.lbl_geotag_status
        self.txt_geotag.returnPressed.connect(self._on_geotag_committed)
        self.txt_geotag.place_selected.connect(self._on_place_selected)
        self.txt_geotag.geotag_cleared.connect(self._on_geotag_cleared)
        top_vbox.addWidget(self.txt_geotag)

        # 3. Caption
        lbl_cap = QLabel("Caption", top_content)
        lbl_cap.setObjectName("SectionTitle")
        top_vbox.addWidget(lbl_cap)

        self.txt_caption = QLineEdit(top_content)
        self.txt_caption.setObjectName("InlineEdit")
        self.txt_caption.setPlaceholderText("Add caption")
        self.txt_caption.editingFinished.connect(self._on_caption_editing_finished)
        top_vbox.addWidget(self.txt_caption)

        # 4. Descriptive tags
        lbl_tags = QLabel("Descriptive tags", top_content)
        lbl_tags.setObjectName("SectionTitle")
        top_vbox.addWidget(lbl_tags)

        self.txt_add_tag = _DescriptiveTagInput(top_content)
        self.txt_add_tag.setObjectName("InlineEdit")
        self.txt_add_tag.setPlaceholderText("Add descriptive tags")
        self.txt_add_tag.setToolTip(
            "Add descriptive words to help find and organize photos and videos (Ctrl+T)"
        )
        self.txt_add_tag.returnPressed.connect(self._on_add_tag_return)
        self.txt_add_tag.tag_selected.connect(self.tag_added.emit)
        self._tags_model = self.txt_add_tag._model
        self._tags_completer = self.txt_add_tag._completer
        self._tags_completer.activated[str].connect(self._on_tag_completer_activated)
        top_vbox.addWidget(self.txt_add_tag)

        self.tags_chips_container = QWidget(top_content)
        self.tags_chips_layout = QVBoxLayout(self.tags_chips_container)
        self.tags_chips_layout.setContentsMargins(0, 0, 0, 0)
        self.tags_chips_layout.setSpacing(3)
        top_vbox.addWidget(self.tags_chips_container)
        top_vbox.addStretch(1)
        top_scroll.setWidget(top_content)

        # 2. Bottom Section: Information scroll area
        bottom_scroll = _NoHorizontalScrollArea(self.info_splitter)
        bottom_scroll.setObjectName("BottomScroll")
        bottom_scroll.setWidgetResizable(True)
        bottom_scroll.setFrameShape(QFrame.NoFrame)
        bottom_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        bottom_scroll.setMinimumHeight(28)
        bottom_scroll.setStyleSheet("background-color: #FFFFFF; border: none;")

        bottom_content = QWidget(bottom_scroll)
        bottom_content.setStyleSheet("background-color: #FFFFFF;")
        bottom_vbox = QVBoxLayout(bottom_content)
        bottom_vbox.setContentsMargins(10, 4, 10, 10)
        bottom_vbox.setSpacing(4)

        lbl_info = QLabel("Information", bottom_content)
        lbl_info.setObjectName("SectionTitle")
        bottom_vbox.addWidget(lbl_info)

        # Empty state label: "X items in this view" (media_1792167382404_9d7d42ec.png)
        self.lbl_view_items_count = QLabel("0 items in this view", bottom_content)
        self.lbl_view_items_count.setStyleSheet(
            "color: #55687C; font-family: 'Segoe UI'; font-size: 8.5pt; padding-top: 2px;"
        )
        bottom_vbox.addWidget(self.lbl_view_items_count)

        # Form frame containing metadata when items ARE selected
        self.info_form_frame = QFrame(bottom_content)
        self.info_form_frame.setObjectName("InfoFormFrame")
        form_vbox = QVBoxLayout(self.info_form_frame)
        form_vbox.setContentsMargins(0, 0, 0, 0)
        form_vbox.setSpacing(4)

        self.lbl_selection_header = QLabel("No items selected", self.info_form_frame)
        self.lbl_selection_header.setStyleSheet(
            "color: #7B8B9B; font-family: 'Segoe UI'; font-size: 8.5pt;"
        )
        form_vbox.addWidget(self.lbl_selection_header)

        form = QFormLayout()
        form.setContentsMargins(0, 2, 0, 2)
        form.setSpacing(3)
        form.setLabelAlignment(Qt.AlignLeft)

        def make_val_lbl() -> QLabel:
            lbl = QLabel("-", self.info_form_frame)
            lbl.setStyleSheet("color: #1E395B; font-family: 'Segoe UI'; font-size: 8.5pt;")
            lbl.setWordWrap(True)
            return lbl

        def make_field_lbl(text: str) -> QLabel:
            lbl = QLabel(text, self.info_form_frame)
            lbl.setStyleSheet("color: #55687C; font-family: 'Segoe UI'; font-size: 8.5pt;")
            return lbl

        self.txt_filename = QLineEdit(self.info_form_frame)
        self.txt_filename.setStyleSheet(
            """
            QLineEdit {
                background: #FFFFFF;
                border: 1px solid #7BA7C9;
                border-radius: 2px;
                padding: 1px 4px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                color: #1E395B;
            }
            QLineEdit:focus {
                border: 1px solid #000000;
            }
            """
        )
        self.txt_filename.returnPressed.connect(self._on_filename_return)
        self.txt_filename.editingFinished.connect(self._on_filename_return)
        self.lbl_filename = self.txt_filename  # Alias for backward compatibility
        self.stack_filename = _ClickToEditWidget(
            self.txt_filename, self.info_form_frame, empty_placeholder="-"
        )
        form.addRow(make_field_lbl("Filename:"), self.stack_filename)

        self.date_time_widget = QWidget(self.info_form_frame)
        dt_hbox = QHBoxLayout(self.date_time_widget)
        dt_hbox.setContentsMargins(0, 0, 0, 0)
        dt_hbox.setSpacing(4)

        self.date_edit = QDateEdit(self.date_time_widget)
        self.date_edit.setDisplayFormat("dd/MM/yyyy")
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setStyleSheet(
            """
            QDateEdit {
                background: #FFFFFF;
                border: 1px solid #7BA7C9;
                border-radius: 2px;
                padding: 1px 2px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                color: #1E395B;
            }
            """
        )
        self.date_edit.setMinimumWidth(102)
        self.date_edit.editingFinished.connect(self._on_datetime_committed)

        self.time_edit = QTimeEdit(self.date_time_widget)
        self.time_edit.setDisplayFormat("HH:mm:ss")
        self.time_edit.setMinimumWidth(76)
        self.time_edit.setStyleSheet(
            """
            QTimeEdit {
                background: #FFFFFF;
                border: 1px solid #7BA7C9;
                border-radius: 2px;
                padding: 1px 2px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                color: #1E395B;
            }
            """
        )
        self.time_edit.editingFinished.connect(self._on_datetime_committed)

        dt_hbox.addWidget(self.date_edit)
        dt_hbox.addWidget(self.time_edit)
        dt_hbox.addStretch(1)

        self.btn_date_taken = QPushButton("-", self.info_form_frame)
        self.btn_date_taken.setVisible(False)
        self.btn_date_taken.clicked.connect(self.change_time_requested.emit)

        self.stack_date_time = _ClickToEditWidget(
            self.date_time_widget, self.info_form_frame, empty_placeholder="-"
        )
        form.addRow(make_field_lbl("Date taken:"), self.stack_date_time)

        self.lbl_size = make_val_lbl()
        form.addRow(make_field_lbl("Size:"), self.lbl_size)

        # 5-Star Rating row with interactive hover outline
        self.star_rating_widget = _StarRatingHoverWidget(self.info_form_frame)
        self.star_buttons = self.star_rating_widget.buttons
        self.star_rating_widget.rating_changed.connect(self._on_hover_rating_changed)
        form.addRow(make_field_lbl("Rating:"), self.star_rating_widget)

        # Flag icon row (matching WLPG hollow/filled flag)
        flag_widget = QWidget(self.info_form_frame)
        flag_hbox = QHBoxLayout(flag_widget)
        flag_hbox.setContentsMargins(0, 0, 0, 0)
        self.btn_flag = QPushButton(flag_widget)
        self.btn_flag.setFixedSize(22, 18)
        self.btn_flag.setFlat(True)
        self.btn_flag.setCursor(Qt.PointingHandCursor)
        self.btn_flag.setIcon(make_wlpg_icon("flag_off", 15))
        self.btn_flag.setIconSize(QSize(15, 15))
        self.btn_flag.clicked.connect(self._on_flag_button_clicked)
        flag_hbox.addWidget(self.btn_flag)
        flag_hbox.addStretch(1)
        form.addRow(make_field_lbl("Flag:"), flag_widget)

        self.lbl_dimensions = make_val_lbl()
        form.addRow(make_field_lbl("Dimensions:"), self.lbl_dimensions)

        self.lbl_published = make_val_lbl()
        form.addRow(make_field_lbl("Published:"), self.lbl_published)

        # Video-only rows (shown instead of Camera/Author/Exposure/... for videos)
        self._info_form = form
        self.lbl_duration = make_val_lbl()
        form.addRow(make_field_lbl("Duration:"), self.lbl_duration)
        self.lbl_bitrate = make_val_lbl()
        form.addRow(make_field_lbl("Bitrate:"), self.lbl_bitrate)
        self.lbl_frame_rate = make_val_lbl()
        form.addRow(make_field_lbl("Frame rate:"), self.lbl_frame_rate)

        self.lbl_camera = make_val_lbl()
        form.addRow(make_field_lbl("Camera:"), self.lbl_camera)

        self.txt_author = QLineEdit(self.info_form_frame)
        self.txt_author.setPlaceholderText("Add an author")
        self.txt_author.setStyleSheet(
            """
            QLineEdit {
                background: #FFFFFF;
                border: 1px solid #D5E2F0;
                border-radius: 2px;
                padding: 1px 4px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                color: #1E395B;
            }
            QLineEdit:hover, QLineEdit:focus {
                border: 1px solid #7BA7C9;
            }
            """
        )
        self.txt_author.returnPressed.connect(self._on_author_return)
        self.txt_author.editingFinished.connect(self._on_author_return)
        self.lbl_author = self.txt_author  # Alias for backward compatibility
        self.stack_author = _ClickToEditWidget(
            self.txt_author, self.info_form_frame, empty_placeholder="Add an author"
        )
        form.addRow(make_field_lbl("Author:"), self.stack_author)

        self.lbl_exposure = make_val_lbl()
        form.addRow(make_field_lbl("Exposure:"), self.lbl_exposure)

        self.lbl_aperture = make_val_lbl()
        form.addRow(make_field_lbl("Aperture:"), self.lbl_aperture)

        self.lbl_focal_length = make_val_lbl()
        form.addRow(make_field_lbl("Focal length:"), self.lbl_focal_length)

        self.lbl_iso = make_val_lbl()
        form.addRow(make_field_lbl("ISO:"), self.lbl_iso)

        self.lbl_lat = make_val_lbl()
        form.addRow(make_field_lbl("Latitude:"), self.lbl_lat)

        self.lbl_lon = make_val_lbl()
        form.addRow(make_field_lbl("Longitude:"), self.lbl_lon)

        self._apply_video_info_rows(None)

        # Legacy aliases for test compatibility
        self.lbl_orientation = self.lbl_published
        self.lbl_gps = self.lbl_lat

        form_vbox.addLayout(form)
        bottom_vbox.addWidget(self.info_form_frame)
        bottom_vbox.addStretch(1)

        bottom_scroll.setWidget(bottom_content)

        self.info_splitter.addWidget(top_scroll)
        self.info_splitter.addWidget(bottom_scroll)
        self.info_splitter.setCollapsible(0, False)
        self.info_splitter.setCollapsible(1, False)
        self.info_splitter.setSizes([320, 240])

        page_layout.addWidget(self.info_splitter)

        # Initially nothing selected
        self.info_form_frame.setVisible(False)
        self.lbl_view_items_count.setVisible(True)

        return page_widget

    # -------------------------------------------------------------------------
    # Page 2: Fine-Tune Adjustments Pane (media_1792165772919_087b03d2.png)
    # -------------------------------------------------------------------------
    def _build_fine_tune_page(self) -> QWidget:
        scroll = _NoHorizontalScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget(scroll)
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(8, 2, 8, 8)
        vbox.setSpacing(8)

        # Group 1: Adjust exposure
        grp_exp = _FineTuneGroup("Adjust exposure", content)
        self.ft_slider_brightness = self._make_aero_slider(-100, 100, 0)
        grp_exp.vbox.addWidget(self._make_slider_header("Brightness", self.ft_slider_brightness))
        grp_exp.vbox.addWidget(self.ft_slider_brightness)

        self.ft_slider_contrast = self._make_aero_slider(-100, 100, 0)
        grp_exp.vbox.addWidget(self._make_slider_header("Contrast", self.ft_slider_contrast))
        grp_exp.vbox.addWidget(self.ft_slider_contrast)

        self.ft_slider_shadows = self._make_aero_slider(-100, 100, 0)
        grp_exp.vbox.addWidget(self._make_slider_header("Shadows", self.ft_slider_shadows))
        grp_exp.vbox.addWidget(self.ft_slider_shadows)

        self.ft_slider_highlights = self._make_aero_slider(-100, 100, 0)
        grp_exp.vbox.addWidget(self._make_slider_header("Highlights", self.ft_slider_highlights))
        grp_exp.vbox.addWidget(self.ft_slider_highlights)

        self.ft_slider_clarity = self._make_aero_slider(-100, 100, 0)
        grp_exp.vbox.addWidget(self._make_slider_header("Clarity", self.ft_slider_clarity))
        grp_exp.vbox.addWidget(self.ft_slider_clarity)

        hist_header = self._make_slider_header("Histogram")
        grp_exp.vbox.addWidget(hist_header)
        self.lbl_hist_levels = hist_header.findChild(QLabel, "val_lbl")
        if self.lbl_hist_levels:
            self.lbl_hist_levels.setText("0 - 255")
        self.histogram_widget = HistogramWidget(grp_exp)
        self.histogram_widget.points_changed.connect(self._on_histogram_points_changed)
        self.histogram_widget.slider_pressed.connect(self.fine_tune_slider_pressed.emit)
        self.histogram_widget.slider_released.connect(self.fine_tune_slider_released.emit)
        grp_exp.vbox.addWidget(self.histogram_widget)

        for s in (
            self.ft_slider_brightness,
            self.ft_slider_contrast,
            self.ft_slider_shadows,
            self.ft_slider_highlights,
            self.ft_slider_clarity,
        ):
            s.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
            s.valueChanged.connect(self._emit_fine_tune_exposure)
            s.sliderReleased.connect(
                lambda: self._on_fine_tune_slider_released("Adjust Exposure")
            )

        vbox.addWidget(grp_exp)

        # Group 2: Adjust color
        grp_color = _FineTuneGroup("Adjust color", content)
        self.ft_slider_temp = self._make_aero_slider(-100, 100, 0)
        grp_color.vbox.addWidget(self._make_slider_header("Color temperature", self.ft_slider_temp))
        grp_color.vbox.addWidget(self.ft_slider_temp)

        self.ft_slider_tint = self._make_aero_slider(-100, 100, 0)
        grp_color.vbox.addWidget(self._make_slider_header("Tint", self.ft_slider_tint))
        grp_color.vbox.addWidget(self.ft_slider_tint)

        self.ft_slider_sat = self._make_aero_slider(-100, 100, 0)
        grp_color.vbox.addWidget(self._make_slider_header("Saturation", self.ft_slider_sat))
        grp_color.vbox.addWidget(self.ft_slider_sat)

        self.ft_slider_dehaze = self._make_aero_slider(-100, 100, 0)
        grp_color.vbox.addWidget(self._make_slider_header("Dehaze", self.ft_slider_dehaze))
        grp_color.vbox.addWidget(self.ft_slider_dehaze)

        for s in (self.ft_slider_temp, self.ft_slider_tint, self.ft_slider_sat, self.ft_slider_dehaze):
            s.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
            s.valueChanged.connect(self._emit_fine_tune_color)
            s.sliderReleased.connect(
                lambda: self._on_fine_tune_slider_released("Adjust Color")
            )

        vbox.addWidget(grp_color)

        # Group 3: Straighten photo
        grp_str = _FineTuneGroup("Straighten photo", content)
        str_row = QHBoxLayout()
        self.btn_auto_straighten = QPushButton("Auto straighten", grp_str)
        self.btn_auto_straighten.setFixedHeight(22)
        self.btn_auto_straighten.setStyleSheet(
            """
            QPushButton {
                background: #F7FAFD;
                border: 1px solid #B5D2ED;
                border-radius: 3px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
                padding: 2px 8px;
            }
            QPushButton:hover {
                background: #E4F0FC;
            }
            """
        )
        self.btn_auto_straighten.clicked.connect(self.auto_straighten_requested.emit)
        str_row.addWidget(self.btn_auto_straighten)
        str_row.addStretch(1)
        grp_str.vbox.addLayout(str_row)

        self.ft_slider_straighten = self._make_aero_slider(-15, 15, 0)
        self.ft_slider_straighten.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
        self.ft_slider_straighten.valueChanged.connect(self._emit_fine_tune_straighten)
        self.ft_slider_straighten.sliderReleased.connect(
            lambda: self._on_fine_tune_slider_released("Straighten Photo")
        )
        grp_str.vbox.addWidget(self._make_slider_header("Angle", self.ft_slider_straighten, unit="°"))
        grp_str.vbox.addWidget(self.ft_slider_straighten)
        vbox.addWidget(grp_str)

        # Group 4: Adjust detail
        grp_det = _FineTuneGroup("Adjust detail", content)
        self.ft_slider_sharpen = self._make_aero_slider(0, 100, 0)
        grp_det.vbox.addWidget(self._make_slider_header("Sharpen", self.ft_slider_sharpen))
        grp_det.vbox.addWidget(self.ft_slider_sharpen)

        self.ft_slider_denoise = self._make_aero_slider(0, 100, 0)
        grp_det.vbox.addWidget(self._make_slider_header("Reduce noise", self.ft_slider_denoise))
        grp_det.vbox.addWidget(self.ft_slider_denoise)

        self.ft_slider_sharpen.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
        self.ft_slider_sharpen.valueChanged.connect(self._emit_fine_tune_detail)
        self.ft_slider_sharpen.sliderReleased.connect(
            lambda: self._on_fine_tune_slider_released("Adjust Detail")
        )

        self.ft_slider_denoise.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
        self.ft_slider_denoise.valueChanged.connect(self._emit_fine_tune_detail)
        self.ft_slider_denoise.sliderReleased.connect(
            lambda: self._on_fine_tune_slider_released("Adjust Detail")
        )

        vbox.addWidget(grp_det)

        # Group 5: Vignette & Optics
        grp_optics = _FineTuneGroup("Vignette & optics", content)
        self.ft_slider_vignette = self._make_aero_slider(-100, 100, 0)
        grp_optics.vbox.addWidget(self._make_slider_header("Vignette", self.ft_slider_vignette))
        grp_optics.vbox.addWidget(self.ft_slider_vignette)

        self.ft_slider_vignette.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
        self.ft_slider_vignette.valueChanged.connect(self._emit_fine_tune_optics)
        self.ft_slider_vignette.sliderReleased.connect(
            lambda: self._on_fine_tune_slider_released("Adjust Vignette")
        )

        vbox.addWidget(grp_optics)

        # Bottom Actions: Hold to Compare (Original) and Reset to default
        action_bar = QHBoxLayout()
        action_bar.setSpacing(6)
        action_bar.setContentsMargins(0, 6, 0, 4)

        self.btn_hold_compare = QPushButton("Hold to Compare (Original)", content)
        self.btn_hold_compare.setFixedHeight(26)
        self.btn_hold_compare.setCursor(Qt.PointingHandCursor)
        self.btn_hold_compare.setToolTip("Click and hold to preview original unedited photo (\\)")
        self.btn_hold_compare.setStyleSheet(
            """
            QPushButton {
                background: #F7FAFD;
                border: 1px solid #B5D2ED;
                border-radius: 3px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #E4F0FC;
            }
            QPushButton:pressed {
                background: #CCE4F7;
                border-color: #3399FF;
            }
            """
        )
        self.btn_hold_compare.pressed.connect(self.compare_started.emit)
        self.btn_hold_compare.released.connect(self.compare_ended.emit)
        action_bar.addWidget(self.btn_hold_compare, 1)

        self.btn_reset_default = QPushButton("Reset to default", content)
        self.btn_reset_default.setFixedHeight(26)
        self.btn_reset_default.setCursor(Qt.PointingHandCursor)
        self.btn_reset_default.setToolTip("Reset all adjustments and levels back to default")
        self.btn_reset_default.setStyleSheet(
            """
            QPushButton {
                background: #F7FAFD;
                border: 1px solid #B5D2ED;
                border-radius: 3px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #FFF1F0;
                border-color: #FFA39E;
                color: #CF1322;
            }
            QPushButton:pressed {
                background: #FFCCC7;
            }
            """
        )
        self.btn_reset_default.clicked.connect(self.reset_all_adjustments)
        action_bar.addWidget(self.btn_reset_default, 1)

        vbox.addLayout(action_bar)
        vbox.addStretch(1)

        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _make_slider_header(text: str, slider: _AeroFineTuneSlider | None = None, unit: str = "") -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent; border: none;")
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(4)

        lbl = QLabel(text, container)
        lbl.setStyleSheet(
            "color: #1E395B; font-family: 'Segoe UI'; font-size: 8pt; border: none; background: transparent;"
        )
        h_layout.addWidget(lbl)
        h_layout.addStretch(1)

        val_lbl = QLabel("0" if not unit else f"0{unit}", container)
        val_lbl.setObjectName("val_lbl")
        val_lbl.setMinimumWidth(32)
        val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        val_lbl.setStyleSheet(
            "color: #5A7A9E; font-family: 'Segoe UI'; font-size: 8pt; font-weight: 600; border: none; background: transparent; padding-right: 2px;"
        )
        h_layout.addWidget(val_lbl)

        if slider is not None:
            lbl.setToolTip(f"Double-click to reset {text} to default")
            val_lbl.setToolTip(f"Double-click to reset {text} to default")

            def on_double_click(event):
                slider.setValue(slider.default_val)
                event.accept()

            lbl.mouseDoubleClickEvent = on_double_click
            val_lbl.mouseDoubleClickEvent = on_double_click

            def on_val_changed(val: int):
                prefix = "+" if val > 0 else ""
                val_lbl.setText(f"{prefix}{val}{unit}")

            slider.valueChanged.connect(on_val_changed)
        return container

    @staticmethod
    def _make_slider_label(text: str, slider: _AeroFineTuneSlider | None = None) -> QWidget:
        return TagAndCaptionDock._make_slider_header(text, slider)

    def _on_histogram_points_changed(self, bp: int, wp: int) -> None:
        if hasattr(self, "lbl_hist_levels") and self.lbl_hist_levels:
            self.lbl_hist_levels.setText(f"{bp} - {wp}")
        if self._updating_ui:
            return
        self.fine_tune_levels_changed.emit(bp, wp)

    def reset_all_adjustments(self) -> None:
        self._updating_ui = True
        try:
            if hasattr(self, "lbl_hist_levels") and self.lbl_hist_levels:
                self.lbl_hist_levels.setText("0 - 255")
            for s in (
                self.ft_slider_brightness,
                self.ft_slider_contrast,
                self.ft_slider_shadows,
                self.ft_slider_highlights,
                self.ft_slider_clarity,
                self.ft_slider_temp,
                self.ft_slider_tint,
                self.ft_slider_sat,
                self.ft_slider_dehaze,
                self.ft_slider_straighten,
                self.ft_slider_sharpen,
                self.ft_slider_denoise,
                self.ft_slider_vignette,
            ):
                s.setValue(s.default_val)
            self.histogram_widget.reset_points()
        finally:
            self._updating_ui = False
        self.fine_tune_reset_requested.emit()

    def _build_depth_light_page(self) -> QWidget:
        scroll = _NoHorizontalScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget(scroll)
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(6, 2, 6, 8)
        vbox.setSpacing(6)

        # -------------------------------------------------------------
        # Group 1: Subject Mask & Cutout (Foundational mask for Cutout, Bokeh, Light & Haze)
        # -------------------------------------------------------------
        grp_cutout = _FineTuneGroup("Subject Mask & Cutout", content)

        # Top row: Primary 1-Click Action Buttons with authentic icons
        cutout_row = QHBoxLayout()
        cutout_row.setSpacing(6)
        cutout_row.setContentsMargins(0, 2, 0, 4)

        btn_style_cutout = """
            QPushButton {
                background: #F7FAFD;
                border: 1px solid #B5D2ED;
                border-radius: 3px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
                font-weight: 600;
                padding: 3px 6px;
                text-align: left;
            }
            QPushButton:hover {
                background: #E4F0FC;
                border-color: #7BA7C9;
            }
            QPushButton:pressed {
                background: #CCE4F7;
                border-color: #3399FF;
            }
        """

        self.btn_copy_subject = QPushButton("Copy Subject", grp_cutout)
        self.btn_copy_subject.setIcon(make_wlpg_icon("copy_subject", 16))
        self.btn_copy_subject.setIconSize(QSize(16, 16))
        self.btn_copy_subject.setFixedHeight(28)
        self.btn_copy_subject.setCursor(Qt.PointingHandCursor)
        self.btn_copy_subject.setToolTip(
            "Extract the foreground subject with a transparent background (RGBA) and copy to clipboard"
        )
        self.btn_copy_subject.setStyleSheet(btn_style_cutout)
        self.btn_copy_subject.clicked.connect(self.copy_subject_requested.emit)
        cutout_row.addWidget(self.btn_copy_subject, 1)

        self.btn_remove_background = QPushButton("Remove Background", grp_cutout)
        self.btn_remove_background.setIcon(make_wlpg_icon("remove_background", 16))
        self.btn_remove_background.setIconSize(QSize(16, 16))
        self.btn_remove_background.setFixedHeight(28)
        self.btn_remove_background.setCursor(Qt.PointingHandCursor)
        self.btn_remove_background.setToolTip(
            "Remove the background from this photo (save as PNG, WebP, AVIF, or JXL to keep transparency)"
        )
        self.btn_remove_background.setStyleSheet(btn_style_cutout)
        self.btn_remove_background.clicked.connect(self.remove_background_requested.emit)
        cutout_row.addWidget(self.btn_remove_background, 1)

        grp_cutout.vbox.addLayout(cutout_row)

        # Boundary Shift & Feather Controls (affects subject cutout, bokeh, light & haze)
        self.dl_slider_boundary_shift = self._make_aero_slider(-20, 20, 0)
        grp_cutout.vbox.addWidget(
            self._make_slider_header("Boundary shift (shrink / grow)", self.dl_slider_boundary_shift)
        )
        grp_cutout.vbox.addWidget(self.dl_slider_boundary_shift)

        self.dl_slider_boundary_feather = self._make_aero_slider(0, 20, 0)
        grp_cutout.vbox.addWidget(
            self._make_slider_header("Boundary feather", self.dl_slider_boundary_feather)
        )
        grp_cutout.vbox.addWidget(self.dl_slider_boundary_feather)

        # Live Mask Preview checkbox button
        mask_row = QHBoxLayout()
        mask_row.setContentsMargins(0, 2, 0, 0)
        self.btn_toggle_mask_preview = QPushButton("Preview Subject Mask", grp_cutout)
        self.btn_toggle_mask_preview.setCheckable(True)
        self.btn_toggle_mask_preview.setFixedHeight(22)
        self.btn_toggle_mask_preview.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_mask_preview.setToolTip(
            "Show semi-transparent ruby mask overlay to verify edge alignment"
        )
        self.btn_toggle_mask_preview.setStyleSheet(
            """
            QPushButton {
                background: #F7FAFD;
                border: 1px solid #C8D6E5;
                border-radius: 2px;
                color: #5A7A9E;
                font-family: 'Segoe UI';
                font-size: 7.5pt;
                font-weight: 600;
                padding: 1px 6px;
            }
            QPushButton:hover {
                background: #EBF3FB;
                color: #0066CC;
            }
            QPushButton:checked {
                background: #FFE8E8;
                border-color: #D9534F;
                color: #C9302C;
            }
            """
        )
        self.btn_toggle_mask_preview.toggled.connect(self.mask_preview_toggled.emit)
        mask_row.addWidget(self.btn_toggle_mask_preview)
        mask_row.addStretch(1)
        grp_cutout.vbox.addLayout(mask_row)

        for s in (self.dl_slider_boundary_shift, self.dl_slider_boundary_feather):
            s.sliderPressed.connect(lambda: (
                self.fine_tune_slider_pressed.emit(),
                self.mask_preview_toggled.emit(True) if not self.btn_toggle_mask_preview.isChecked() else None
            ))
            s.valueChanged.connect(self._emit_depth_bokeh)
            s.sliderReleased.connect(lambda: (
                self._on_depth_slider_released("Adjust Boundary"),
                self.mask_preview_toggled.emit(False) if not self.btn_toggle_mask_preview.isChecked() else None
            ))

        vbox.addWidget(grp_cutout)

        # -------------------------------------------------------------
        # Group 2: Portrait Bokeh & Focus
        # -------------------------------------------------------------
        grp_bokeh = _FineTuneGroup("Portrait Bokeh & Focus", content)
        self.dl_slider_blur = self._make_aero_slider(0, 100, 0)
        grp_bokeh.vbox.addWidget(self._make_slider_header("Bokeh blur strength", self.dl_slider_blur))
        grp_bokeh.vbox.addWidget(self.dl_slider_blur)

        self.dl_slider_focal = self._make_aero_slider(0, 100, 0)
        grp_bokeh.vbox.addWidget(self._make_slider_header("Change focal depth", self.dl_slider_focal))
        grp_bokeh.vbox.addWidget(self.dl_slider_focal)

        self.dl_slider_glow = self._make_aero_slider(0, 100, 0)
        grp_bokeh.vbox.addWidget(self._make_slider_header("Specular highlight bloom", self.dl_slider_glow))
        grp_bokeh.vbox.addWidget(self.dl_slider_glow)

        for s in (self.dl_slider_blur, self.dl_slider_focal, self.dl_slider_glow):
            s.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
            s.valueChanged.connect(self._emit_depth_bokeh)
            s.sliderReleased.connect(lambda: self._on_depth_slider_released("Adjust Bokeh & Focus"))

        vbox.addWidget(grp_bokeh)

        # -------------------------------------------------------------
        # Group 3: 3D Depth Relighting
        # -------------------------------------------------------------
        grp_relight = _FineTuneGroup("3D Depth Relighting", content)
        self.dl_slider_near = self._make_aero_slider(-100, 100, 0)
        grp_relight.vbox.addWidget(self._make_slider_header("Near subject light", self.dl_slider_near))
        grp_relight.vbox.addWidget(self.dl_slider_near)

        self.dl_slider_far = self._make_aero_slider(-100, 100, 0)
        grp_relight.vbox.addWidget(self._make_slider_header("Background light", self.dl_slider_far))
        grp_relight.vbox.addWidget(self.dl_slider_far)

        self.dl_slider_warmth_near = self._make_aero_slider(-100, 100, 0)
        grp_relight.vbox.addWidget(self._make_slider_header("Near subject warmth", self.dl_slider_warmth_near))
        grp_relight.vbox.addWidget(self.dl_slider_warmth_near)

        self.dl_slider_warmth_far = self._make_aero_slider(-100, 100, 0)
        grp_relight.vbox.addWidget(self._make_slider_header("Background warmth", self.dl_slider_warmth_far))
        grp_relight.vbox.addWidget(self.dl_slider_warmth_far)

        for s in (self.dl_slider_near, self.dl_slider_far, self.dl_slider_warmth_near, self.dl_slider_warmth_far):
            s.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
            s.valueChanged.connect(self._emit_depth_relight)
            s.sliderReleased.connect(lambda: self._on_depth_slider_released("3D Depth Relighting"))

        vbox.addWidget(grp_relight)

        # -------------------------------------------------------------
        # Group 4: Atmospheric Depth Haze
        # -------------------------------------------------------------
        grp_haze = _FineTuneGroup("Atmospheric Depth Haze", content)
        self.dl_slider_haze = self._make_aero_slider(0, 100, 0)
        grp_haze.vbox.addWidget(self._make_slider_header("Atmospheric haze / mist", self.dl_slider_haze))
        grp_haze.vbox.addWidget(self.dl_slider_haze)

        self.dl_slider_haze_warmth = self._make_aero_slider(-100, 100, 0)
        grp_haze.vbox.addWidget(self._make_slider_header("Haze warmth (Blue <-> Amber)", self.dl_slider_haze_warmth))
        grp_haze.vbox.addWidget(self.dl_slider_haze_warmth)

        for s in (self.dl_slider_haze, self.dl_slider_haze_warmth):
            s.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
            s.valueChanged.connect(self._emit_depth_haze)
            s.sliderReleased.connect(lambda: self._on_depth_slider_released("Atmospheric Depth Haze"))

        vbox.addWidget(grp_haze)

        # Bottom Actions: Hold to Compare & Reset stacked cleanly to prevent horizontal overflow
        action_bar = QVBoxLayout()
        action_bar.setSpacing(5)
        action_bar.setContentsMargins(0, 4, 0, 4)

        self.btn_dl_hold_compare = QPushButton("Hold to Compare (Original)", content)
        self.btn_dl_hold_compare.setFixedHeight(26)
        self.btn_dl_hold_compare.setCursor(Qt.PointingHandCursor)
        self.btn_dl_hold_compare.setToolTip("Click and hold to preview original unedited photo (\\)")
        self.btn_dl_hold_compare.setStyleSheet(
            """
            QPushButton {
                background: #F7FAFD;
                border: 1px solid #B5D2ED;
                border-radius: 3px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #E4F0FC;
            }
            QPushButton:pressed {
                background: #CCE4F7;
                border-color: #3399FF;
            }
            """
        )
        self.btn_dl_hold_compare.pressed.connect(self.compare_started.emit)
        self.btn_dl_hold_compare.released.connect(self.compare_ended.emit)
        action_bar.addWidget(self.btn_dl_hold_compare)

        self.btn_dl_reset = QPushButton("Reset to Default", content)
        self.btn_dl_reset.setFixedHeight(26)
        self.btn_dl_reset.setCursor(Qt.PointingHandCursor)
        self.btn_dl_reset.setToolTip("Reset all depth, blur, relight, and haze adjustments to default")
        self.btn_dl_reset.setStyleSheet(
            """
            QPushButton {
                background: #F7FAFD;
                border: 1px solid #B5D2ED;
                border-radius: 3px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #FFF1F0;
                border-color: #FFA39E;
                color: #CF1322;
            }
            QPushButton:pressed {
                background: #FFCCC7;
                border-color: #F5222D;
            }
            """
        )
        self.btn_dl_reset.clicked.connect(self.depth_reset_requested.emit)
        action_bar.addWidget(self.btn_dl_reset)

        vbox.addLayout(action_bar)
        vbox.addStretch(1)

        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _make_aero_slider(min_val: int, max_val: int, default_val: int) -> _AeroFineTuneSlider:
        return _AeroFineTuneSlider(min_val, max_val, default_val)

    def _emit_fine_tune_exposure(self) -> None:
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_ft", False):
            self._is_sliding_ft = True
            self.fine_tune_slider_pressed.emit()
        self._pending_ft_exposure = True
        if not self._ft_timer.isActive():
            self._ft_timer.start()

    def _emit_fine_tune_color(self) -> None:
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_ft", False):
            self._is_sliding_ft = True
            self.fine_tune_slider_pressed.emit()
        self._pending_ft_color = True
        if not self._ft_timer.isActive():
            self._ft_timer.start()

    def _emit_fine_tune_optics(self) -> None:
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_ft", False):
            self._is_sliding_ft = True
            self.fine_tune_slider_pressed.emit()
        self._pending_ft_optics = True
        if not self._ft_timer.isActive():
            self._ft_timer.start()

    def _emit_fine_tune_straighten(self) -> None:
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_ft", False):
            self._is_sliding_ft = True
            self.fine_tune_slider_pressed.emit()
        self._pending_ft_straighten = True
        if not self._ft_timer.isActive():
            self._ft_timer.start()

    def _emit_fine_tune_detail(self) -> None:
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_ft", False):
            self._is_sliding_ft = True
            self.fine_tune_slider_pressed.emit()
        self._pending_ft_detail = True
        if not self._ft_timer.isActive():
            self._ft_timer.start()

    def _flush_fine_tune_emits(self) -> None:
        if self._updating_ui:
            return
        if self._pending_ft_exposure:
            self._pending_ft_exposure = False
            self.fine_tune_exposure_changed.emit(
                self.ft_slider_brightness.value(),
                self.ft_slider_contrast.value(),
                self.ft_slider_shadows.value(),
                self.ft_slider_highlights.value(),
                self.ft_slider_clarity.value(),
            )
        if self._pending_ft_color:
            self._pending_ft_color = False
            self.fine_tune_color_changed.emit(
                self.ft_slider_temp.value(),
                self.ft_slider_tint.value(),
                self.ft_slider_sat.value(),
                self.ft_slider_dehaze.value(),
            )
        if self._pending_ft_optics:
            self._pending_ft_optics = False
            self.fine_tune_optics_changed.emit(self.ft_slider_vignette.value())
        if self._pending_ft_straighten:
            self._pending_ft_straighten = False
            self.fine_tune_straighten_changed.emit(float(self.ft_slider_straighten.value()))
        if self._pending_ft_detail:
            self._pending_ft_detail = False
            self.fine_tune_detail_changed.emit(
                self.ft_slider_sharpen.value(),
                self.ft_slider_denoise.value(),
            )

    def _on_fine_tune_slider_released(self, action_name: str) -> None:
        self._is_sliding_ft = False
        if self._ft_timer.isActive():
            self._ft_timer.stop()
        self._flush_fine_tune_emits()
        self.fine_tune_slider_released.emit(action_name)

    def _emit_depth_bokeh(self) -> None:
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_depth", False):
            self._is_sliding_depth = True
            self.fine_tune_slider_pressed.emit()
        self._pending_dl_bokeh = True
        if not self._dl_timer.isActive():
            self._dl_timer.start()

    def _emit_depth_relight(self) -> None:
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_depth", False):
            self._is_sliding_depth = True
            self.fine_tune_slider_pressed.emit()
        self._pending_dl_relight = True
        if not self._dl_timer.isActive():
            self._dl_timer.start()

    def _emit_depth_haze(self) -> None:
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_depth", False):
            self._is_sliding_depth = True
            self.fine_tune_slider_pressed.emit()
        self._pending_dl_haze = True
        if not self._dl_timer.isActive():
            self._dl_timer.start()

    def _flush_depth_light_emits(self) -> None:
        if self._updating_ui:
            return
        if self._pending_dl_bokeh:
            self._pending_dl_bokeh = False
            self.depth_bokeh_changed.emit(
                self.dl_slider_blur.value(),
                self.dl_slider_focal.value(),
                self.dl_slider_glow.value(),
                self.dl_slider_boundary_shift.value(),
                self.dl_slider_boundary_feather.value(),
            )
        if self._pending_dl_relight:
            self._pending_dl_relight = False
            self.depth_relight_changed.emit(
                self.dl_slider_near.value(),
                self.dl_slider_far.value(),
                self.dl_slider_warmth_near.value(),
                self.dl_slider_warmth_far.value(),
            )
        if self._pending_dl_haze:
            self._pending_dl_haze = False
            self.depth_haze_changed.emit(
                self.dl_slider_haze.value(),
                self.dl_slider_haze_warmth.value(),
            )

    def _on_depth_slider_released(self, action_name: str) -> None:
        self._is_sliding_depth = False
        if self._dl_timer.isActive():
            self._dl_timer.stop()
        self._flush_depth_light_emits()
        self.fine_tune_slider_released.emit(action_name)

    def set_denoise_analyzed_value(self, val: float) -> None:
        """Sets the noise reduction slider to the calculated noise analysis value."""
        self.ft_slider_denoise.setValue(int(round(val)))

    def sync_recipe(self, recipe: Any) -> None:
        """Synchronizes fine-tune and depth & light sliders with current PhotoAdjustmentRecipe without emitting signals."""
        self._updating_ui = True
        try:
            if recipe is None:
                b = c = s = h = cl = t = ti = sat = dh = str_deg = sh = dn = vig = 0
                d_blur = d_focal = d_glow = d_bshift = d_bfeather = d_near = d_far = d_wn = d_wf = d_haze = d_hw = 0
            else:
                b = int(round(getattr(recipe, "brightness", 0.0)))
                c = int(round(getattr(recipe, "contrast", 0.0)))
                s = int(round(getattr(recipe, "shadows", 0.0)))
                h = int(round(getattr(recipe, "highlights", 0.0)))
                cl = int(round(getattr(recipe, "clarity", 0.0)))
                t = int(round(getattr(recipe, "temperature", 0.0)))
                ti = int(round(getattr(recipe, "tint", 0.0)))
                sat = int(round(getattr(recipe, "saturation", 0.0)))
                dh = int(round(getattr(recipe, "dehaze", 0.0)))
                str_deg = int(round(getattr(recipe, "straighten_deg", 0.0)))
                sh = int(round(getattr(recipe, "sharpen", 0.0)))
                dn = int(round(getattr(recipe, "denoise", 0.0)))
                vig = int(round(getattr(recipe, "vignette", 0.0)))

                d_blur = int(round(getattr(recipe, "depth_blur", 0.0)))
                d_focal = int(round(getattr(recipe, "focal_plane", 0.0)))
                d_glow = int(round(getattr(recipe, "bokeh_glow", 0.0)))
                d_bshift = int(round(getattr(recipe, "boundary_shift", 0.0)))
                d_bfeather = int(round(getattr(recipe, "boundary_feather", 0.0)))
                d_near = int(round(getattr(recipe, "relight_near", 0.0)))
                d_far = int(round(getattr(recipe, "relight_far", 0.0)))
                d_wn = int(round(getattr(recipe, "warmth_near", 0.0)))
                d_wf = int(round(getattr(recipe, "warmth_far", 0.0)))
                d_haze = int(round(getattr(recipe, "depth_haze", 0.0)))
                d_hw = int(round(getattr(recipe, "depth_haze_warmth", 0.0)))

            if hasattr(self, "ft_slider_brightness"):
                self.ft_slider_brightness.setValue(b)
                self.ft_slider_contrast.setValue(c)
                self.ft_slider_shadows.setValue(s)
                self.ft_slider_highlights.setValue(h)
                self.ft_slider_clarity.setValue(cl)
                self.ft_slider_temp.setValue(t)
                self.ft_slider_tint.setValue(ti)
                self.ft_slider_sat.setValue(sat)
                self.ft_slider_dehaze.setValue(dh)
                self.ft_slider_straighten.setValue(str_deg)
                self.ft_slider_sharpen.setValue(sh)
                self.ft_slider_denoise.setValue(dn)
                self.ft_slider_vignette.setValue(vig)

            if hasattr(self, "dl_slider_blur"):
                self.dl_slider_blur.setValue(d_blur)
                self.dl_slider_focal.setValue(d_focal)
                self.dl_slider_glow.setValue(d_glow)
                if hasattr(self, "dl_slider_boundary_shift"):
                    self.dl_slider_boundary_shift.setValue(d_bshift)
                if hasattr(self, "dl_slider_boundary_feather"):
                    self.dl_slider_boundary_feather.setValue(d_bfeather)
                self.dl_slider_near.setValue(d_near)
                self.dl_slider_far.setValue(d_far)
                self.dl_slider_warmth_near.setValue(d_wn)
                self.dl_slider_warmth_far.setValue(d_wf)
                self.dl_slider_haze.setValue(d_haze)
                self.dl_slider_haze_warmth.setValue(d_hw)
        finally:
            self._updating_ui = False

    # -------------------------------------------------------------------------
    # Public Binding & Selection API
    # -------------------------------------------------------------------------
    def set_autocomplete_lists(
        self,
        tag_paths: Sequence[str],
        people_names: Sequence[str],
        recent_tags: Sequence[str] | None = None,
    ) -> None:
        self._people_model.setStringList(list(people_names))
        self.txt_add_tag.set_tag_lists(tag_paths, recent_tags)

    def focus_descriptive_tag_input(self) -> None:
        """Opens the Tag & Caption pane, focuses Add descriptive tags, and shows suggestions (Ctrl+T)."""
        self.set_mode("info")
        self.show()
        self.raise_()
        self.txt_add_tag.setFocus()
        QTimer.singleShot(0, self.txt_add_tag.show_suggestions_popup)

    @staticmethod
    def _clear_layout(layout: QVBoxLayout) -> None:
        while layout.count() > 0:
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

    def _on_location_in_view_clicked(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.UserRole)
        if data and len(data) >= 4:
            mids, label, lat, lon = data[0], data[1], data[2], data[3]
            self.map_location_selected.emit(mids, label, lat, lon)

    def set_map_clusters(self, clusters: Sequence[dict[str, Any]]) -> None:
        """Populates the Locations in View list for Map View."""
        self.list_locations_in_view.clear()
        for c in clusters:
            label = str(c.get("label", ""))
            cnt = int(c.get("count", len(c.get("media_ids", []))))
            item = QListWidgetItem(f"📍 {label} ({cnt})")
            item.setData(
                Qt.UserRole,
                (
                    c.get("media_ids", []),
                    label,
                    float(c.get("lat", 0.0)),
                    float(c.get("lon", 0.0)),
                ),
            )
            self.list_locations_in_view.addItem(item)

    def set_view_mode(self, mode: str) -> None:
        """Toggles between standard 'grid' and 'map' mode in RightTagDock."""
        self._view_mode = mode
        if mode == "map":
            self.locations_in_view_container.show()
        else:
            self.locations_in_view_container.hide()

    @property
    def selected_count(self) -> int:
        """Returns the number of currently selected media items."""
        return len(getattr(self, "_selected_ids", []))

    def set_total_view_count(self, count: int) -> None:
        """Updates total view items count for the Information pane empty state."""
        self._total_view_count = max(0, int(count))
        if not self._selected_ids:
            item_w = "item" if self._total_view_count == 1 else "items"
            self.lbl_view_items_count.setText(f"{self._total_view_count} {item_w} in this view")

    def bind_selection(
        self, media_ids: Sequence[int], summary: dict[str, Any]
    ) -> None:
        self._updating_ui = True
        try:
            self._selected_ids = [int(mid) for mid in media_ids]
            count = int(summary.get("count", len(self._selected_ids)))

            self._clear_layout(self.people_chips_layout)
            self._clear_layout(self.tags_chips_layout)

            if count == 0:
                self.info_form_frame.setVisible(False)
                self.lbl_view_items_count.setVisible(True)
                item_w = "item" if self._total_view_count == 1 else "items"
                self.lbl_view_items_count.setText(f"{self._total_view_count} {item_w} in this view")
                self.lbl_selection_header.setText("No items selected")
                self.txt_filename.setText("-")
                self.txt_filename.setEnabled(False)
                self.stack_filename.finish_edit("-")
                self.stack_filename.set_editable(False)
                self.date_edit.setEnabled(False)
                self.time_edit.setEnabled(False)
                self.stack_date_time.finish_edit("-")
                self.stack_date_time.set_editable(False)
                self.btn_date_taken.setText("-")
                self.lbl_size.setText("-")
                self.lbl_size.setToolTip("")
                self.star_rating_widget.set_rating(0)
                self._update_flag_button(0)
                self.lbl_dimensions.setText("-")
                self.lbl_published.setText("-")
                self.lbl_camera.setText("-")
                self.txt_author.setText("")
                self.txt_author.setEnabled(False)
                self.stack_author.finish_edit("Add an author")
                self.stack_author.set_editable(False)
                self.lbl_exposure.setText("-")
                self.lbl_aperture.setText("-")
                self.lbl_focal_length.setText("-")
                self.lbl_iso.setText("-")
                self.lbl_lat.setText("Unavailable")
                self.lbl_lon.setText("Unavailable")
                self.txt_caption.clear()
                self.txt_geotag.clear()
                return

            self.lbl_view_items_count.setVisible(False)
            self.info_form_frame.setVisible(True)

            # Face regions & people tags (confirmed, unconfirmed suggestions, unknown detected faces)
            face_regions = summary.get("face_regions") or summary.get("people", [])
            boxless_names = [
                str(p.get("person_name") or p.get("name") or "").strip()
                for p in face_regions
                if int(p.get("state", 1)) == 1
                and (float(p.get("rect_w") or 0.0) <= 0.005 or float(p.get("rect_h") or 0.0) <= 0.005)
                and (p.get("person_name") or p.get("name"))
            ]
            for p in face_regions:
                pname = str(p.get("person_name") or p.get("name") or "").strip()
                fid = int(p.get("face_id") or p.get("region_id") or p.get("id") or 0)
                pid = int(p.get("person_id") or p.get("id") or 0) if (p.get("person_id") or (p.get("id") and pname)) else None
                st = int(p.get("state", 1 if (pid and not p.get("is_unconfirmed")) else 0))
                blob = p.get("thumbnail_blob")
                pix = make_person_avatar_pixmap(pname or "Unknown", blob, size=34, circular=False)

                if st == 0 and not pname:
                    # Detected unknown face ("Who is this?")
                    unknown_chip = _UnknownFaceCardWidget(
                        face_id=fid,
                        icon_pixmap=pix,
                        completer=self._people_completer,
                        unlinked_candidates=boxless_names,
                        parent=self.people_chips_container,
                    )
                    if getattr(self, "_is_dark", False):
                        unknown_chip.set_dark_mode(True)
                    unknown_chip.assign_requested.connect(self.assign_face_requested.emit)
                    unknown_chip.dismiss_requested.connect(self.dismiss_face_requested.emit)
                    unknown_chip.hovered_changed.connect(
                        lambda fid_val, hov: self.face_region_hovered.emit(fid_val if hov else None)
                    )
                    unknown_chip.show()
                    self.people_chips_layout.addWidget(unknown_chip)
                else:
                    # Confirmed person (state == 1) or unconfirmed suggestion (state == 0 with person_name)
                    is_boxless_val = bool(st == 1 and (float(p.get("rect_w") or 0.0) <= 0.005 or float(p.get("rect_h") or 0.0) <= 0.005))
                    chip = _PersonTagCardWidget(
                        pid or fid,
                        pname,
                        icon_pixmap=pix,
                        is_unconfirmed=bool(st == 0 or p.get("is_unconfirmed")),
                        face_id=fid,
                        is_boxless=is_boxless_val,
                        parent=self.people_chips_container,
                    )
                    detected_unassigned = [
                        reg for reg in face_regions
                        if int(reg.get("state", 1)) == 0 and not (reg.get("person_name") or reg.get("name"))
                    ]
                    chip.detected_unassigned_faces = detected_unassigned
                    if getattr(self, "_is_dark", False):
                        chip.set_dark_mode(True)
                    chip.locate_face_clicked.connect(self.person_locate_face_requested.emit)
                    chip.link_face_clicked.connect(self.person_link_face_requested.emit)
                    chip.remove_clicked.connect(lambda pid_val, fid_val=fid: (
                        self.remove_face_requested.emit(fid_val) if fid_val else self.person_removed.emit(pid_val)
                    ))
                    chip.search_clicked.connect(self.person_search_requested.emit)
                    chip.view_suggestions_clicked.connect(self.person_view_suggestions_clicked_slot if hasattr(self, "person_view_suggestions_clicked_slot") else self.person_view_suggestions_requested.emit)
                    chip.tag_as_other_clicked.connect(self.person_tag_as_other_requested.emit)
                    chip.set_display_pic_clicked.connect(self.person_set_display_pic_requested.emit)
                    chip.merge_clicked.connect(self.person_merge_requested.emit)
                    chip.rename_clicked.connect(self.person_rename_requested.emit)
                    chip.confirm_clicked.connect(self.person_confirm_requested.emit)
                    chip.reject_suggestion_clicked.connect(self.person_reject_suggestion_requested.emit)
                    chip.hovered_changed.connect(
                        lambda fid_val, hov: self.face_region_hovered.emit(fid_val if hov else None)
                    )
                    chip.hovered_changed.connect(
                        lambda pid_val, hov: self.person_hovered.emit(pid_val, hov)
                    )
                    chip.show()
                    self.people_chips_layout.addWidget(chip)

            # Display-level tag deduplication: if both 'Subject/Occasions/Christmas' and 'Christmas' exist, keep only the hierarchical one
            raw_tags = summary.get("tags", [])
            seen_leaf_hier: set[str] = {
                str(t.get("name") or "").lower()
                for t in raw_tags
                if "/" in str(t.get("full_path") or "")
            }
            deduped_display_tags: list[dict[str, Any]] = []
            seen_display_names: set[str] = set()
            for t in raw_tags:
                t_name = str(t.get("name") or "").strip()
                t_path = str(t.get("full_path") or t_name)
                # If flat tag is already represented as leaf of a hierarchical tag, skip
                if "/" not in t_path and t_name.lower() in seen_leaf_hier:
                    continue
                # Also prevent identical names if multiple tags somehow have identical names
                if t_name.lower() in seen_display_names:
                    continue
                seen_display_names.add(t_name.lower())
                deduped_display_tags.append(t)

            for t in deduped_display_tags:
                t_count = int(t.get("count", 1))
                chip = _DescriptiveTagRowWidget(
                    int(t["id"]),
                    str(t["name"]),
                    tag_count=t_count,
                    total_selection_count=count,
                    tooltip=str(t.get("full_path", t["name"])),
                    parent=self.tags_chips_container,
                )
                if getattr(self, "_is_dark", False):
                    chip.set_dark_mode(True)
                chip.remove_clicked.connect(self.tag_removed.emit)
                chip.tag_clicked.connect(lambda tid, _name: self.tag_filter_requested.emit(tid))
                chip.assign_to_all_clicked.connect(self.tag_assign_to_all_requested.emit)
                chip.show()
                self.tags_chips_layout.addWidget(chip)

            if count == 1:
                self.lbl_selection_header.setText("1 item selected")
                fname = str(summary.get("filename", "-"))
                self._original_filename = fname
                self.txt_filename.setText(fname)
                self.txt_filename.setEnabled(True)
                self.stack_filename.set_display_text(fname)
                self.stack_filename.finish_edit()
                self.stack_filename.set_editable(True)

                dt_val = summary.get("date_taken") or summary.get("mtime")
                if dt_val is not None:
                    self._current_date_taken_ts = int(dt_val)
                    qdt = QDateTime.fromSecsSinceEpoch(int(dt_val), QTimeZone.utc())
                    self.date_edit.setDate(qdt.date())
                    self.time_edit.setTime(qdt.time())
                    self.time_edit.setToolTip(qdt.time().toString("HH:mm"))
                    dt_str = qdt.toString("dd/MM/yyyy HH:mm")
                    self.btn_date_taken.setText(dt_str)
                    self.stack_date_time.set_display_text(dt_str)
                else:
                    self.btn_date_taken.setText("Unavailable")
                    self.stack_date_time.set_display_text("Unavailable")
                self.date_edit.setEnabled(True)
                self.time_edit.setEnabled(True)
                self.stack_date_time.finish_edit()
                self.stack_date_time.set_editable(True)

                sz_str = format_file_size(float(summary.get("size") or 0))
                self.lbl_size.setText(sz_str)
                self.lbl_size.setToolTip(sz_str)

                rating_val = int(summary.get("rating") or 0)
                self.star_rating_widget.set_rating(rating_val)
                self._current_rating = rating_val
                self._update_flag_button(int(summary.get("flag") or 0))

                w = summary.get("width")
                h = summary.get("height")
                self.lbl_dimensions.setText(f"{w} × {h}" if (w and h) else "-")
                self.lbl_published.setText("No")

                make = str(summary.get("camera_make") or "").strip()
                model = str(summary.get("camera_model") or "").strip()
                cam_str = f"{make} {model}".strip() or "-"
                self.lbl_camera.setText(cam_str)

                author_val = str(summary.get("author") or "").strip()
                self.txt_author.setText(author_val)
                self.txt_author.setEnabled(True)
                self.stack_author.set_display_text(author_val or "Add an author")
                self.stack_author.finish_edit()
                self.stack_author.set_editable(True)

                self.lbl_exposure.setText(f"{summary.get('exposure_time') or '-'}")
                f_num = summary.get("f_number")
                self.lbl_aperture.setText(f"f/{f_num:.1f}" if f_num else "-")
                fl = summary.get("focal_length")
                self.lbl_focal_length.setText(f"{fl:.0f} mm" if fl else "-")
                self.lbl_iso.setText(str(summary.get("iso") or "-"))
                self._apply_video_info_rows(summary)

                lat = summary.get("lat")
                lon = summary.get("lon")
                self.lbl_lat.setText(f"{float(lat):.5f}" if lat is not None else "Unavailable")
                self.lbl_lon.setText(f"{float(lon):.5f}" if lon is not None else "Unavailable")

                self.txt_caption.setText(str(summary.get("caption") or ""))
                loc_name = str(summary.get("location_name") or "").strip()
                if not loc_name and lat is not None and lon is not None:
                    try:
                        rev = reverse_geocode(float(lat), float(lon))
                        if rev and rev.get("formatted_address"):
                            loc_name = rev["formatted_address"]
                    except Exception:
                        pass
                self.txt_geotag.setText(loc_name)
            else:
                lead = summary.get("lead", {})
                self.lbl_selection_header.setText(f"{count} items selected")
                self.txt_filename.setText(f"Multiple files ({count})")
                self.txt_filename.setEnabled(False)
                self.stack_filename.set_display_text(f"Multiple files ({count})")
                self.stack_filename.finish_edit()
                self.stack_filename.set_editable(False)

                lead_dt = lead.get("date_taken") or lead.get("mtime")
                if lead_dt is not None:
                    self._current_date_taken_ts = int(lead_dt)
                    qdt = QDateTime.fromSecsSinceEpoch(int(lead_dt), QTimeZone.utc())
                    self.date_edit.setDate(qdt.date())
                    self.time_edit.setTime(qdt.time())
                    self.time_edit.setToolTip(qdt.time().toString("HH:mm"))
                    dt_str = qdt.toString("dd/MM/yyyy HH:mm")
                    self.btn_date_taken.setText(dt_str)
                    self.stack_date_time.set_display_text(dt_str)
                else:
                    self.btn_date_taken.setText("Multiple")
                    self.stack_date_time.set_display_text("Multiple")
                self.date_edit.setEnabled(True)
                self.time_edit.setEnabled(True)
                self.stack_date_time.finish_edit()
                self.stack_date_time.set_editable(True)

                total_sz = f"{format_file_size(float(summary.get('total_size') or 0))} total"
                self.lbl_size.setText(total_sz)
                self.lbl_size.setToolTip(total_sz)

                common_rating = summary.get("common_rating")
                r_val = int(common_rating) if common_rating is not None else 0
                self.star_rating_widget.set_rating(r_val)
                self._current_rating = r_val

                common_flag = summary.get("common_flag")
                self._update_flag_button(int(common_flag) if common_flag is not None else 0)

                self.lbl_dimensions.setText("-")
                self.lbl_published.setText("-")
                self.lbl_camera.setText("-")
                self.txt_author.setText("-")
                self.txt_author.setEnabled(False)
                self.stack_author.set_display_text("-")
                self.stack_author.finish_edit()
                self.stack_author.set_editable(False)
                self.lbl_exposure.setText("-")
                self.lbl_aperture.setText("-")
                self.lbl_focal_length.setText("-")
                self.lbl_iso.setText("-")
                self._apply_video_info_rows(None)
                lead_lat = lead.get("lat")
                lead_lon = lead.get("lon")
                self.lbl_lat.setText(f"{float(lead_lat):.5f}" if lead_lat is not None else "Unavailable")
                self.lbl_lon.setText(f"{float(lead_lon):.5f}" if lead_lon is not None else "Unavailable")

                self.txt_caption.setText(str(lead.get("caption") or ""))
                common_loc = str(summary.get("common_location_name") or lead.get("location_name") or "").strip()
                if not common_loc and lead_lat is not None and lead_lon is not None:
                    try:
                        rev = reverse_geocode(float(lead_lat), float(lead_lon))
                        if rev and rev.get("formatted_address"):
                            common_loc = rev["formatted_address"]
                    except Exception:
                        pass
                self.txt_geotag.setText(common_loc)
        finally:
            self._updating_ui = False

    def _apply_video_info_rows(self, summary: dict[str, Any] | None) -> None:
        """Shows Duration/Bitrate/Frame rate for a single video; Camera/Author/Exposure/... otherwise."""
        is_video = bool(summary and summary.get("is_video"))
        form = getattr(self, "_info_form", None)
        if form is None:
            return
        if is_video and summary is not None:
            total = int(round(float(summary.get("duration") or 0.0)))
            hh, rem = divmod(total, 3600)
            mm, ss = divmod(rem, 60)
            self.lbl_duration.setText(f"{hh:02d}:{mm:02d}:{ss:02d}")
            self.lbl_bitrate.setText(f"{int(summary.get('bitrate_kbps') or 0)}kbps")
            self.lbl_frame_rate.setText(f"{float(summary.get('fps') or 0.0):g} frames/sec")
        photo_rows = (
            self.lbl_camera,
            self.stack_author,
            self.lbl_exposure,
            self.lbl_aperture,
            self.lbl_focal_length,
            self.lbl_iso,
        )
        video_rows = (self.lbl_duration, self.lbl_bitrate, self.lbl_frame_rate)
        for w in photo_rows:
            form.setRowVisible(w, not is_video)
        for w in video_rows:
            form.setRowVisible(w, is_video)

    def _update_star_buttons(self, rating: int) -> None:
        self._current_rating = max(0, min(5, int(rating)))
        self.star_rating_widget.set_rating(self._current_rating)

    def _update_flag_button(self, flag: int) -> None:
        self._current_flag = 1 if flag else 0
        self.btn_flag.setIcon(
            make_wlpg_icon("flag_on" if self._current_flag == 1 else "flag_off", 15)
        )

    # -------------------------------------------------------------------------
    # Editing Slots
    # -------------------------------------------------------------------------
    def _on_filename_return(self) -> None:
        if self._updating_ui or not self._selected_ids or len(self._selected_ids) != 1:
            self.stack_filename.finish_edit()
            return
        new_name = self.txt_filename.text().strip()
        self.stack_filename.finish_edit(new_name or self._original_filename or "-")
        if new_name and new_name != self._original_filename:
            self._original_filename = new_name
            self.filename_committed.emit(new_name)

    def _on_datetime_committed(self) -> None:
        if self._updating_ui or not self._selected_ids:
            self.stack_date_time.finish_edit()
            return
        qdate = self.date_edit.date()
        qtime = self.time_edit.time()
        qdt = QDateTime(qdate, qtime, QTimeZone.utc())
        ts = int(qdt.toSecsSinceEpoch())
        self.time_edit.setToolTip(qtime.toString("HH:mm"))
        self.stack_date_time.finish_edit(qdt.toString("dd/MM/yyyy HH:mm"))
        if ts != self._current_date_taken_ts:
            self._current_date_taken_ts = ts
            self.date_taken_edited.emit(ts)

    def _on_author_return(self) -> None:
        if self._updating_ui or not self._selected_ids:
            self.stack_author.finish_edit()
            return
        new_author = self.txt_author.text().strip()
        self.stack_author.finish_edit(new_author or "Add an author")
        self.author_committed.emit(new_author)

    def _on_hover_rating_changed(self, new_rating: int) -> None:
        if self._updating_ui or not self._selected_ids:
            return
        self._current_rating = new_rating
        self.rating_edited.emit(new_rating)

    def _on_star_button_clicked(self, star_idx: int) -> None:
        if self._updating_ui or not self._selected_ids:
            return
        self.star_rating_widget.on_star_clicked(star_idx)

    def _on_flag_button_clicked(self) -> None:
        if self._updating_ui or not self._selected_ids:
            return
        new_flag = 0 if self._current_flag == 1 else 1
        self._update_flag_button(new_flag)
        self.flag_edited.emit(new_flag)

    def _on_caption_editing_finished(self) -> None:
        if self._updating_ui or not self._selected_ids:
            return
        self.caption_committed.emit(self.txt_caption.text().strip())

    def commit_caption(self, caption_text: str | None = None) -> None:
        if caption_text is not None:
            self.txt_caption.setText(caption_text)
        self.caption_committed.emit(self.txt_caption.text().strip())

    def _on_geotag_committed(self) -> None:
        if self._updating_ui or not self._selected_ids:
            return
        self.txt_geotag.setCursorPosition(0)
        self.geotag_edited.emit(self.txt_geotag.text().strip())

    def _on_place_selected(self, place_name: str, lat: float, lon: float) -> None:
        if self._updating_ui or not self._selected_ids:
            return
        self.txt_geotag.setCursorPosition(0)
        self.lbl_lat.setText(f"{lat:.5f}")
        self.lbl_lon.setText(f"{lon:.5f}")
        self.geotag_edited.emit(place_name)

    def _on_geotag_cleared(self) -> None:
        if self._updating_ui or not self._selected_ids:
            return
        self.lbl_lat.setText("Unavailable")
        self.lbl_lon.setText("Unavailable")
        self.geotag_edited.emit("")

    def _on_add_tag_return(self) -> None:
        text = self.txt_add_tag.text().strip()
        if text:
            self.txt_add_tag.record_used_tag(text)
            self.txt_add_tag.clear()
            self.tag_added.emit(text)

    def _on_tag_completer_activated(self, text: str) -> None:
        val = text.strip()
        if val:
            self.txt_add_tag.record_used_tag(val)
            self.txt_add_tag.clear()
            QTimer.singleShot(0, self.txt_add_tag.clear)
            self.tag_added.emit(val)

    def add_descriptive_tag_text(self, tag_path: str) -> None:
        self.txt_add_tag.setText(tag_path)
        self._on_add_tag_return()

    def _on_add_person_return(self) -> None:
        text = self.txt_add_person.text().strip()
        if text:
            self.txt_add_person.clear()
            self.person_added.emit(text)

    def _on_person_completer_activated(self, text: str) -> None:
        val = text.strip()
        if val:
            self.txt_add_person.clear()
            self.person_added.emit(val)

    def add_person_tag_text(self, person_name: str) -> None:
        self.txt_add_person.setText(person_name)
        self._on_add_person_return()

    def set_dark_mode(self, is_dark: bool) -> None:
        """Applies Dark Studio or standard light styling to the Right Tag & Caption / Fine-Tune Dock."""
        self._is_dark = bool(is_dark)
        self._apply_stylesheet()

        # Update splitter handle
        if hasattr(self, "info_splitter"):
            for i in range(self.info_splitter.count()):
                h = self.info_splitter.handle(i)
                if h is not None:
                    h._is_dark = self._is_dark
                    h.update()

        # Update click-to-edit stacks and inputs
        for st in (
            getattr(self, "stack_filename", None),
            getattr(self, "stack_date_time", None),
            getattr(self, "stack_author", None),
        ):
            if st is not None:
                st.set_dark_mode(self._is_dark)

        if hasattr(self, "lbl_view_items_count"):
            self.lbl_view_items_count.setStyleSheet(
                f"color: {'#94A3B8' if is_dark else '#55687C'}; font-family: 'Segoe UI'; font-size: 8.5pt; padding-top: 2px;"
            )
        if hasattr(self, "lbl_selection_header"):
            self.lbl_selection_header.setStyleSheet(
                f"color: {'#94A3B8' if is_dark else '#7B8B9B'}; font-family: 'Segoe UI'; font-size: 8.5pt;"
            )

        # Update form labels inside info_form_frame only if visible
        if hasattr(self, "info_form_frame") and self.info_form_frame.isVisible() and hasattr(self, "_info_form"):
            form = self._info_form
            for row in range(form.rowCount()):
                lbl_item = form.itemAt(row, QFormLayout.LabelRole)
                if lbl_item and isinstance(lbl_item.widget(), QLabel):
                    lbl_item.widget().setStyleSheet(
                        f"color: {'#94A3B8' if is_dark else '#55687C'}; font-family: 'Segoe UI'; font-size: 8.5pt;"
                    )
                fld_item = form.itemAt(row, QFormLayout.FieldRole)
                if fld_item and isinstance(fld_item.widget(), QLabel):
                    fld_item.widget().setStyleSheet(
                        f"color: {'#E2E8F0' if is_dark else '#1E395B'}; font-family: 'Segoe UI'; font-size: 8.5pt;"
                    )

        # Update existing people/tag chips only if visible
        if hasattr(self, "people_chips_layout") and self.isVisible():
            for i in range(self.people_chips_layout.count()):
                w = self.people_chips_layout.itemAt(i).widget()
                if w and hasattr(w, "set_dark_mode"):
                    w.set_dark_mode(self._is_dark)
        if hasattr(self, "tags_chips_layout") and self.isVisible():
            for i in range(self.tags_chips_layout.count()):
                w = self.tags_chips_layout.itemAt(i).widget()
                if w and hasattr(w, "set_dark_mode"):
                    w.set_dark_mode(self._is_dark)

    def _apply_stylesheet(self) -> None:
        is_dark = getattr(self, "_is_dark", False)
        if is_dark:
            self.container.setStyleSheet(
                """
                QFrame#RightSidebarContainer {
                    background-color: #1B1E24;
                    border-left: 1px solid #2D323B;
                }
                """
            )
            if hasattr(self, "page_info"):
                self.page_info.setStyleSheet("background-color: #1B1E24;")
            if hasattr(self, "info_splitter"):
                self.info_splitter.setStyleSheet("background-color: #1B1E24;")
                for i in range(self.info_splitter.count()):
                    w = self.info_splitter.widget(i)
                    if isinstance(w, QScrollArea):
                        w.setStyleSheet("background-color: #1B1E24; border: none;")
                        if w.widget():
                            w.widget().setStyleSheet("background-color: #1B1E24;")
            self.setStyleSheet(
                """
                QDockWidget#TagAndCaptionDock {
                    background: #1B1E24;
                    background-color: #1B1E24;
                }
                QFrame#RightSidebarContainer {
                    background-color: #1B1E24;
                    border-left: 1px solid #2D323B;
                }
                QWidget#InfoPage {
                    background-color: #1B1E24;
                }
                QSplitter#InfoSplitter {
                    background-color: #1B1E24;
                }
                QScrollArea {
                    background-color: #1B1E24;
                    border: none;
                }
                QScrollArea > QWidget > QWidget {
                    background-color: #1B1E24;
                }
                QFrame#FineTuneGroup {
                    background: #22262E;
                    border: 1px solid #2D323B;
                    border-radius: 4px;
                }
                QLabel#SectionTitle {
                    color: #60A5FA;
                    font-family: 'Segoe UI';
                    font-size: 9.5pt;
                    font-weight: 600;
                    border: none;
                    background: transparent;
                    margin-top: 6px;
                }
                QLineEdit#InlineEdit {
                    background: #14171B;
                    border: 1px solid #333A45;
                    border-radius: 2px;
                    padding: 2px 6px;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    color: #E2E8F0;
                }
                QLineEdit#InlineEdit:focus {
                    border: 1px solid #3B82F6;
                }
                """
            )
            return

        self.container.setStyleSheet(
            """
            QFrame#RightSidebarContainer {
                background-color: #FFFFFF;
                border-left: 1px solid #D2DFEE;
            }
            """
        )
        if hasattr(self, "page_info"):
            self.page_info.setStyleSheet("background-color: #FFFFFF;")
        if hasattr(self, "info_splitter"):
            self.info_splitter.setStyleSheet("background-color: #FFFFFF;")
            for i in range(self.info_splitter.count()):
                w = self.info_splitter.widget(i)
                if isinstance(w, QScrollArea):
                    w.setStyleSheet("background-color: #FFFFFF; border: none;")
                    if w.widget():
                        w.widget().setStyleSheet("background-color: #FFFFFF;")
        self.setStyleSheet(
            """
            QDockWidget#TagAndCaptionDock {
                background: #FFFFFF;
                background-color: #FFFFFF;
            }
            QFrame#RightSidebarContainer {
                background-color: #FFFFFF;
                border-left: 1px solid #D2DFEE;
            }
            QWidget#InfoPage {
                background-color: #FFFFFF;
            }
            QSplitter#InfoSplitter {
                background-color: #FFFFFF;
            }
            QScrollArea {
                background-color: #FFFFFF;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #FFFFFF;
            }
            QLabel#SectionTitle {
                color: #1B64B0;
                font-family: 'Segoe UI';
                font-size: 9.5pt;
                font-weight: 600;
                border: none;
                background: transparent;
                margin-top: 6px;
            }
            QLineEdit#InlineEdit {
                background: #FFFFFF;
                border: 1px solid #D5E2F0;
                border-radius: 2px;
                padding: 2px 6px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                color: #1E395B;
            }
            QLineEdit#InlineEdit:focus {
                border: 1px solid #3380CC;
            }
            """
        )


RightTagDock = TagAndCaptionDock
