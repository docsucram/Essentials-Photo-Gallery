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
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import (
    QDate,
    QDateTime,
    QEvent,
    QMetaObject,
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
    Slot,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QKeySequence,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPixmap,
    QPolygonF,
    QRadialGradient,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QCompleter,
    QDateEdit,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
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
from src.ui.theme_tokens import set_ui_role


class _AeroFineTuneSlider(QSlider):
    """High-fidelity Aero horizontal slider with double-click to reset."""

    def __init__(self, min_val: int, max_val: int, default_val: int, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Horizontal, parent)
        self.default_val = default_val
        self._header_widget: QWidget | None = None
        self.setRange(min_val, max_val)
        self.setValue(default_val)
        self.setFixedHeight(18)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.SizeHorCursor)
        self.set_track_gradient(None)

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(bool(enabled))
        if self._header_widget is not None:
            self._header_widget.setEnabled(bool(enabled))

    def set_track_gradient(self, stops: list[tuple[float, str]] | None) -> None:
        """Apply a neutral or color-coded linear gradient groove to the slider."""
        if stops:
            stops_css = ", ".join(f"stop:{pos:.2f} {col}" for pos, col in stops)
            groove_bg = f"qlineargradient(x1:0, y1:0, x2:1, y2:0, {stops_css})"
            groove_h = "5px"
        else:
            groove_bg = "#D8DFE6"
            groove_h = "4px"
        self.setStyleSheet(
            f"""
            QSlider::groove:horizontal {{
                height: {groove_h};
                background: {groove_bg};
                border-radius: 2px;
            }}
            QSlider::groove:horizontal:disabled {{
                background: rgba(148, 163, 184, 0.25);
            }}
            QSlider::handle:horizontal {{
                background: #0078D7;
                border: 1px solid #005A9E;
                width: 8px;
                margin: -6px 0;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #2B88D8;
            }}
            QSlider::handle:horizontal:disabled {{
                background: #94A3B8;
                border: 1px solid #64748B;
            }}
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

    def wheelEvent(self, event: Any) -> None:
        """Ignore wheel events when unfocused so sidebar scrolling never nudges sliders."""
        if not (self.hasFocus() or (event.modifiers() & Qt.AltModifier)):
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        step = 5 if (event.modifiers() & Qt.ShiftModifier) else 1
        self.setValue(self.value() + (step if delta > 0 else -step))
        event.accept()


class HistogramWidget(QWidget):
    """Renders the authentic WLPG dark luminance histogram with interactive Black/White point markers."""

    points_changed = Signal(int, int)  # (black_point, white_point)
    slider_pressed = Signal()
    slider_released = Signal(str)
    histogram_updated = Signal(object, object, object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(94)
        self._counts: list[int] = [0] * 256
        self._r_counts: list[int] = [0] * 256
        self._g_counts: list[int] = [0] * 256
        self._b_counts: list[int] = [0] * 256
        self._show_rgb_channels: bool = False
        self._handles_enabled: bool = True
        self._hist_mask: np.ndarray | None = None
        self._last_image_data: Any = None
        self.black_point: int = 0      # 0 .. 255
        self.white_point: int = 255    # 0 .. 255
        self._drag_handle: str | None = None
        self._hover_handle: str | None = None
        self.setMouseTracking(True)

    def set_handles_enabled(self, enabled: bool) -> None:
        self._handles_enabled = bool(enabled)
        if not self._handles_enabled:
            self._drag_handle = None
            self._hover_handle = None
            self.unsetCursor()
        self.update()

    def set_histogram_mask(self, mask: np.ndarray | None) -> None:
        self._hist_mask = mask
        if self._last_image_data is not None:
            self.set_histogram_data(self._last_image_data)

    def set_rgb_mode(self, enabled: bool) -> None:
        enabled_b = bool(enabled)
        target_h = 128 if enabled_b else 94
        if self.height() != target_h:
            self.setFixedHeight(target_h)
        if self._show_rgb_channels != enabled_b:
            self._show_rgb_channels = enabled_b
        self.update()

    def set_histogram_data(self, image_data: Any) -> None:
        self._last_image_data = image_data
        empty = [0] * 256
        if image_data is None:
            self._counts = list(empty)
            self._r_counts = list(empty)
            self._g_counts = list(empty)
            self._b_counts = list(empty)
            self.update()
            return
        try:
            if isinstance(image_data, np.ndarray):
                h_i, w_i = image_data.shape[:2]
                cv_mask = None
                if self._hist_mask is not None and self._hist_mask.size > 0:
                    m_res = self._hist_mask
                    if m_res.shape[:2] != (h_i, w_i):
                        m_res = cv2.resize(m_res.astype(np.float32), (w_i, h_i), interpolation=cv2.INTER_LINEAR)
                    cand_m = (m_res > 0.25).astype(np.uint8) * 255
                    if np.any(cand_m > 0):
                        cv_mask = cand_m
                if image_data.ndim == 3 and image_data.shape[2] >= 3:
                    rgb = image_data[:, :, :3]
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    r_h = cv2.calcHist([rgb], [0], cv_mask, [256], [0, 256]).flatten()
                    g_h = cv2.calcHist([rgb], [1], cv_mask, [256], [0, 256]).flatten()
                    b_h = cv2.calcHist([rgb], [2], cv_mask, [256], [0, 256]).flatten()
                    max_rgb = max(float(r_h.max()), float(g_h.max()), float(b_h.max()), 1.0)
                    self._r_counts = [int((val / max_rgb) * 1000) for val in r_h]
                    self._g_counts = [int((val / max_rgb) * 1000) for val in g_h]
                    self._b_counts = [int((val / max_rgb) * 1000) for val in b_h]
                else:
                    gray = image_data if image_data.ndim <= 2 else image_data[:, :, 0]
                    self._r_counts = list(empty)
                    self._g_counts = list(empty)
                    self._b_counts = list(empty)
                hist = cv2.calcHist([gray], [0], cv_mask, [256], [0, 256]).flatten()
                max_val = float(hist.max()) if hist.max() > 0 else 1.0
                self._counts = [int((val / max_val) * 1000) for val in hist]
            elif isinstance(image_data, Image.Image):
                thumb_rgb = image_data.resize((160, 120), Image.Resampling.BILINEAR).convert("RGB")
                rgb_arr = np.asarray(thumb_rgb)
                r_h = cv2.calcHist([rgb_arr], [0], None, [256], [0, 256]).flatten()
                g_h = cv2.calcHist([rgb_arr], [1], None, [256], [0, 256]).flatten()
                b_h = cv2.calcHist([rgb_arr], [2], None, [256], [0, 256]).flatten()
                max_rgb = max(float(r_h.max()), float(g_h.max()), float(b_h.max()), 1.0)
                self._r_counts = [int((val / max_rgb) * 1000) for val in r_h]
                self._g_counts = [int((val / max_rgb) * 1000) for val in g_h]
                self._b_counts = [int((val / max_rgb) * 1000) for val in b_h]
                hist = thumb_rgb.convert("L").histogram()
                max_val = max(hist) if hist and max(hist) > 0 else 1
                self._counts = [int((val / max_val) * 1000) for val in hist[:256]]
            else:
                self._counts = list(empty)
                self._r_counts = list(empty)
                self._g_counts = list(empty)
                self._b_counts = list(empty)
        except Exception:
            self._counts = list(empty)
            self._r_counts = list(empty)
            self._g_counts = list(empty)
            self._b_counts = list(empty)
        self.histogram_updated.emit(self._counts, self._r_counts, self._g_counts, self._b_counts)
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
        graph_bottom = max(40.0, float(h) - 26.0)

        # Black background with subtle rounded border
        p.setPen(QPen(QColor("#7B8B9B"), 1))
        p.setBrush(QColor("#000000"))
        p.drawRoundedRect(QRectF(1, 1, w - 2, graph_bottom), 2, 2)

        # Subtle dark grey grid lines (matching WLPG screenshot)
        p.setPen(QPen(QColor("#2C2C2C"), 1))
        p.drawLine(QPointF(1, graph_bottom / 2.0), QPointF(w - 2, graph_bottom / 2.0))
        p.drawLine(QPointF(w * 0.33, 1), QPointF(w * 0.33, graph_bottom))
        p.drawLine(QPointF(w * 0.66, 1), QPointF(w * 0.66, graph_bottom))

        # Histogram curve polygons (RGB channels behind Luminance)
        usable_w = max(1.0, float(w - 12))
        usable_h = graph_bottom - 4
        step = usable_w / 255.0

        def _build_hist_path(counts: list[int]) -> QPainterPath:
            path = QPainterPath()
            path.moveTo(6, graph_bottom - 2)
            for i in range(256):
                x = 6 + i * step
                y = (graph_bottom - 2) - (counts[i] / 1000.0) * usable_h
                path.lineTo(x, y)
            path.lineTo(w - 6, graph_bottom - 2)
            path.closeSubpath()
            return path

        has_rgb = self._show_rgb_channels and (
            any(self._r_counts) or any(self._g_counts) or any(self._b_counts)
        )
        if has_rgb:
            for counts, fill_col, stroke_col in (
                (self._r_counts, QColor(235, 75, 75, 70), QColor(255, 95, 95, 165)),
                (self._g_counts, QColor(75, 210, 95, 70), QColor(95, 235, 115, 165)),
                (self._b_counts, QColor(75, 145, 255, 70), QColor(100, 170, 255, 165)),
            ):
                if any(counts):
                    p.setPen(QPen(stroke_col, 1))
                    p.setBrush(fill_col)
                    p.drawPath(_build_hist_path(counts))

        if any(self._counts):
            grad = QLinearGradient(0, 0, 0, graph_bottom)
            grad.setColorAt(0.0, QColor(220, 225, 230, 150 if has_rgb else 220))
            grad.setColorAt(1.0, QColor(140, 150, 160, 100 if has_rgb else 180))
            p.setPen(QPen(QColor("#D0D5DA"), 1))
            p.setBrush(grad)
            p.drawPath(_build_hist_path(self._counts))

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
        if event.button() == Qt.LeftButton and self._handles_enabled:
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
            elif pos_y >= max(40.0, float(self.height()) - 26.0) - 8.0:
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
        if not self._handles_enabled:
            super().mouseMoveEvent(event)
            return
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

        handle_y_thresh = max(40.0, float(self.height()) - 26.0) - 13.0
        if (pos_y >= handle_y_thresh and dist_b <= 14.0) or dist_b <= 10.0:
            self._hover_handle = "black"
            self.setCursor(Qt.PointingHandCursor)
            self.setToolTip(f"Black point (Shadows clip): {self.black_point}")
        elif (pos_y >= handle_y_thresh and dist_w <= 14.0) or dist_w <= 10.0:
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


class _ToneCurveCanvas(QWidget):
    """Interactive spline Tone Curve editor for RGB / Red / Green / Blue channels."""

    curve_changed = Signal()
    curve_pressed = Signal()
    curve_released = Signal()

    _CHANNEL_COLORS = {
        "rgb": "#E6ECF2",
        "r": "#F06464",
        "g": "#58C870",
        "b": "#5EA0F5",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(192)
        self.setMouseTracking(True)
        self._active_channel: str = "rgb"
        self._curves: dict[str, list[tuple[float, float]]] = {
            "rgb": [(0.0, 0.0), (1.0, 1.0)],
            "r": [(0.0, 0.0), (1.0, 1.0)],
            "g": [(0.0, 0.0), (1.0, 1.0)],
            "b": [(0.0, 0.0), (1.0, 1.0)],
        }
        self._hist_counts: dict[str, list[int]] = {
            "rgb": [],
            "r": [],
            "g": [],
            "b": [],
        }
        self._drag_idx: int | None = None
        self._hover_idx: int | None = None

    def set_histogram_counts(
        self,
        luma: list[int] | None,
        r: list[int] | None,
        g: list[int] | None,
        b: list[int] | None,
    ) -> None:
        self._hist_counts = {
            "rgb": list(luma or []),
            "r": list(r or []),
            "g": list(g or []),
            "b": list(b or []),
        }
        self.update()

    def set_active_channel(self, channel: str) -> None:
        if channel in self._curves and self._active_channel != channel:
            self._active_channel = channel
            self._drag_idx = None
            self._hover_idx = None
            self.update()

    def get_curves(self) -> dict[str, list[tuple[float, float]]]:
        return {k: [tuple(p) for p in v] for k, v in self._curves.items()}

    def set_curves(self, curves: dict[str, list[tuple[float, float]]]) -> None:
        for ch in ("rgb", "r", "g", "b"):
            pts = curves.get(ch)
            if pts and len(pts) >= 2:
                self._curves[ch] = sorted((float(x), float(y)) for x, y in pts)
            else:
                self._curves[ch] = [(0.0, 0.0), (1.0, 1.0)]
        self.update()

    def apply_s_curve_preset(self) -> None:
        self._curves[self._active_channel] = [
            (0.0, 0.0),
            (0.25, 0.20),
            (0.50, 0.50),
            (0.75, 0.80),
            (1.0, 1.0),
        ]
        self.update()
        self.curve_changed.emit()
        self.curve_released.emit()

    def reset_active_channel(self) -> None:
        self._curves[self._active_channel] = [(0.0, 0.0), (1.0, 1.0)]
        self.update()
        self.curve_changed.emit()
        self.curve_released.emit()

    def _plot_rect(self) -> QRectF:
        return QRectF(8.0, 8.0, max(20.0, float(self.width() - 16)), max(20.0, float(self.height() - 16)))

    def _pt_to_px(self, pt: tuple[float, float]) -> QPointF:
        r = self._plot_rect()
        return QPointF(r.left() + pt[0] * r.width(), r.bottom() - pt[1] * r.height())

    def _px_to_pt(self, pos: QPointF) -> tuple[float, float]:
        r = self._plot_rect()
        nx = max(0.0, min(1.0, (pos.x() - r.left()) / max(1.0, r.width())))
        ny = max(0.0, min(1.0, (r.bottom() - pos.y()) / max(1.0, r.height())))
        return (round(nx, 4), round(ny, 4))

    def paintEvent(self, _event) -> None:
        from src.imaging.opencv_engine import _build_curve_lut

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = self._plot_rect()

        # Clean dark studio background + color-dependent histogram + 4x4 grid
        p.setPen(QPen(QColor("#363D49"), 1))
        p.setBrush(QColor("#1A1D22"))
        p.drawRoundedRect(r, 3, 3)

        hist = self._hist_counts.get(self._active_channel, [])
        if hist and any(hist):
            h_col = QColor(self._CHANNEL_COLORS[self._active_channel])
            h_col.setAlpha(68 if self._active_channel == "rgb" else 88)
            h_stroke = QColor(self._CHANNEL_COLORS[self._active_channel])
            h_stroke.setAlpha(115)
            h_path = QPainterPath()
            h_path.moveTo(r.left() + 1, r.bottom() - 1)
            step_x = (r.width() - 2) / max(1, len(hist) - 1)
            for idx_h, val_h in enumerate(hist):
                hx = r.left() + 1 + idx_h * step_x
                hy = (r.bottom() - 1) - (val_h / 1000.0) * (r.height() - 4)
                h_path.lineTo(hx, hy)
            h_path.lineTo(r.right() - 1, r.bottom() - 1)
            h_path.closeSubpath()
            p.setPen(QPen(h_stroke, 0.9))
            p.setBrush(h_col)
            p.drawPath(h_path)

        p.setPen(QPen(QColor("#282E38"), 1))
        for frac in (0.25, 0.5, 0.75):
            gx = r.left() + frac * r.width()
            gy = r.top() + frac * r.height()
            p.drawLine(QPointF(gx, r.top() + 1), QPointF(gx, r.bottom() - 1))
            p.drawLine(QPointF(r.left() + 1, gy), QPointF(r.right() - 1, gy))

        # Identity diagonal
        p.setPen(QPen(QColor("#38424D"), 1, Qt.DashLine))
        p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.top()))

        # Inactive modified channel curves subtly underneath
        for ch, pts in self._curves.items():
            if ch == self._active_channel:
                continue
            if len(pts) == 2 and pts[0] == (0.0, 0.0) and pts[1] == (1.0, 1.0):
                continue
            lut = _build_curve_lut(pts)
            col = QColor(self._CHANNEL_COLORS[ch])
            col.setAlpha(105)
            p.setPen(QPen(col, 1.3))
            path = QPainterPath()
            for i in range(0, 256, 3):
                px = r.left() + (i / 255.0) * r.width()
                py = r.bottom() - (float(lut[i]) / 255.0) * r.height()
                if i == 0:
                    path.moveTo(px, py)
                else:
                    path.lineTo(px, py)
            px_end = r.right()
            py_end = r.bottom() - (float(lut[255]) / 255.0) * r.height()
            path.lineTo(px_end, py_end)
            p.drawPath(path)

        # Active channel curve
        active_pts = self._curves[self._active_channel]
        lut = _build_curve_lut(active_pts)
        active_col = QColor(self._CHANNEL_COLORS[self._active_channel])
        p.setPen(QPen(active_col, 2.0))
        path = QPainterPath()
        for i in range(256):
            px = r.left() + (i / 255.0) * r.width()
            py = r.bottom() - (float(lut[i]) / 255.0) * r.height()
            if i == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)
        p.drawPath(path)

        # Control point handles
        for idx, pt in enumerate(active_pts):
            c = self._pt_to_px(pt)
            is_hi = idx == self._drag_idx or idx == self._hover_idx
            rad = 5.0 if is_hi else 4.0
            p.setPen(QPen(QColor("#0F1419"), 1.5))
            p.setBrush(QColor("#38BDF8") if is_hi else active_col)
            p.drawEllipse(c, rad, rad)
        p.end()

    def _hit_test(self, pos: QPointF) -> int | None:
        pts = self._curves[self._active_channel]
        best_idx = None
        best_d = 10.0
        for idx, pt in enumerate(pts):
            c = self._pt_to_px(pt)
            d = ((c.x() - pos.x()) ** 2 + (c.y() - pos.y()) ** 2) ** 0.5
            if d <= best_d:
                best_d = d
                best_idx = idx
        return best_idx

    def mousePressEvent(self, event) -> None:
        pos = QPointF(event.pos())
        pts = self._curves[self._active_channel]
        hit = self._hit_test(pos)
        if event.button() == Qt.RightButton:
            if hit is not None and 0 < hit < len(pts) - 1:
                pts.pop(hit)
                self._hover_idx = None
                self.update()
                self.curve_changed.emit()
                self.curve_released.emit()
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            self.curve_pressed.emit()
            if hit is not None:
                self._drag_idx = hit
            else:
                nx, ny = self._px_to_pt(pos)
                nx = max(0.02, min(0.98, nx))
                pts.append((nx, ny))
                pts.sort(key=lambda item: item[0])
                self._drag_idx = pts.index((nx, ny))
                self.curve_changed.emit()
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        pos = QPointF(event.pos())
        pts = self._curves[self._active_channel]
        if self._drag_idx is not None:
            nx, ny = self._px_to_pt(pos)
            idx = self._drag_idx
            if idx == 0:
                nx = 0.0
            elif idx == len(pts) - 1:
                nx = 1.0
            else:
                min_x = pts[idx - 1][0] + 0.02
                max_x = pts[idx + 1][0] - 0.02
                nx = max(min_x, min(max_x, nx))
            pts[idx] = (nx, ny)
            self.update()
            self.curve_changed.emit()
            event.accept()
            return
        prev_h = self._hover_idx
        self._hover_idx = self._hit_test(pos)
        self.setCursor(Qt.PointingHandCursor if self._hover_idx is not None else Qt.CrossCursor)
        if prev_h != self._hover_idx:
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_idx is not None:
            self._drag_idx = None
            self.curve_released.emit()
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            hit = self._hit_test(QPointF(event.pos()))
            pts = self._curves[self._active_channel]
            if hit is not None and 0 < hit < len(pts) - 1:
                pts.pop(hit)
                self._drag_idx = None
                self._hover_idx = None
                self.update()
                self.curve_changed.emit()
                self.curve_released.emit()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)


class _CollapsibleDrawer(QWidget):
    """Compact Lightroom-style collapsible drawer for Advanced Fine Tune controls."""

    def __init__(self, title: str, parent: QWidget | None = None, expanded: bool = False) -> None:
        super().__init__(parent)
        self._title = title
        self._expanded = bool(expanded)
        self._has_active = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 2)
        outer.setSpacing(4)

        self.btn_toggle = QPushButton(self)
        self.btn_toggle.setObjectName("FineTuneDrawerBtn")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(self._expanded)
        self.btn_toggle.setFixedHeight(24)
        self.btn_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_toggle.setStyleSheet(
            """
            QPushButton#FineTuneDrawerBtn {
                text-align: left;
                padding: 2px 8px;
                background: rgba(128, 145, 165, 0.12);
                border: 1px solid rgba(128, 145, 165, 0.28);
                border-radius: 4px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                font-weight: 600;
            }
            QPushButton#FineTuneDrawerBtn:hover {
                background: rgba(56, 152, 236, 0.14);
                border-color: rgba(56, 152, 236, 0.45);
            }
            """
        )
        self.btn_toggle.clicked.connect(self._on_toggled)
        outer.addWidget(self.btn_toggle)

        self.body = QWidget(self)
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(4, 2, 2, 4)
        self.body_layout.setSpacing(5)
        self.body.setVisible(self._expanded)
        outer.addWidget(self.body)
        self._refresh_label()

    def _refresh_label(self) -> None:
        arrow = "▾" if self._expanded else "▸"
        dot = " ●" if self._has_active else ""
        safe_title = self._title.replace("&&", "&").replace("&", "&&")
        self.btn_toggle.setText(f"{arrow}  {safe_title}{dot}")

    def _on_toggled(self, checked: bool) -> None:
        self._expanded = bool(checked)
        self.body.setVisible(self._expanded)
        self.body.updateGeometry()
        self.updateGeometry()
        if self.parentWidget() is not None:
            self.parentWidget().updateGeometry()
        self._refresh_label()

    def set_active_indicator(self, active: bool) -> None:
        active_b = bool(active)
        if self._has_active != active_b:
            self._has_active = active_b
            self._refresh_label()


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
            cnt_text = "(All)" if tag_count >= total_selection_count else f"({tag_count})"
            self.lbl_count = QLabel(cnt_text, self)
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


def _make_pin_icon(size: int = 16, color: str = "#0066CC") -> QIcon:
    """Creates a subtle map pin icon shown while editing a geotag."""
    icon = QIcon()
    for mode, col in [(QIcon.Normal, color), (QIcon.Active, "#004D99"), (QIcon.Selected, "#004D99")]:
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        cx = size * 0.5
        top_y = size * 0.14
        r = size * 0.27
        tip_y = size * 0.88
        path = QPainterPath()
        path.moveTo(cx, tip_y)
        path.quadTo(cx - r * 1.15, size * 0.54, cx - r, top_y + r)
        path.arcTo(QRectF(cx - r, top_y, r * 2.0, r * 2.0), 180.0, -180.0)
        path.quadTo(cx + r * 1.15, size * 0.54, cx, tip_y)
        path.closeSubpath()
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(col)))
        p.drawPath(path)
        p.setBrush(QBrush(QColor("#FFFFFF")))
        p.drawEllipse(QPointF(cx, top_y + r), r * 0.42, r * 0.42)
        p.end()
        icon.addPixmap(pm, mode)
    return icon


class _GeotagInput(QLineEdit):
    """
    WLPG-style Geotag input field featuring sub-2ms gazetteer autocomplete,
    a 'Searching...' status indicator, background Nominatim lookup,
    [✕] clear action on hover, and a focus-only map pin action for adjusting coordinates.
    """

    place_selected = Signal(str, float, float)  # formatted_name, lat, lon
    geotag_cleared = Signal()
    adjust_pin_requested = Signal()

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

        self.act_pin = self.addAction(
            _make_pin_icon(16),
            QLineEdit.TrailingPosition,
        )
        self.act_pin.setToolTip("Adjust location on map...")
        self.act_pin.setVisible(False)
        self.act_pin.triggered.connect(self.adjust_pin_requested.emit)

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

    def focusInEvent(self, event: Any) -> None:
        super().focusInEvent(event)
        self.act_pin.setVisible(True)

    def focusOutEvent(self, event: Any) -> None:
        super().focusOutEvent(event)
        self.act_pin.setVisible(False)
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

        def _fetch() -> None:
            results: list[dict[str, Any]] = []
            try:
                results = search_places_online(query, db_engine=self._db_engine, timeout=4.0, limit=6)
            except Exception:
                results = []
            self._pending_online_search = (query, results)
            QMetaObject.invokeMethod(self, "_finish_online_search", Qt.QueuedConnection)

        threading.Thread(target=_fetch, daemon=True).start()

    @Slot()
    def _finish_online_search(self) -> None:
        payload = getattr(self, "_pending_online_search", None)
        if not payload:
            if self.lbl_status is not None:
                self.lbl_status.setVisible(False)
            return
        self._pending_online_search = None
        query, results = payload
        if results and self.text().strip().lower() == query.lower():
            self._apply_online_results(results)
        elif self.lbl_status is not None:
            self.lbl_status.setVisible(False)

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
    """Rounded card group box for Fine-Tune controls with optional collapsible header."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("FineTuneGroup")
        self._title = title
        self._collapsible = False
        self._expanded = True
        self._has_active = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 8)
        outer.setSpacing(4)

        self.btn_header = QPushButton(title, self)
        self.btn_header.setObjectName("FineTuneGroupTitle")
        self.btn_header.setFlat(True)
        self.btn_header.setStyleSheet(
            """
            QPushButton#FineTuneGroupTitle {
                text-align: left;
                border: none;
                background: transparent;
                padding: 0;
            }
            """
        )
        self.btn_header.clicked.connect(self._on_header_clicked)
        outer.addWidget(self.btn_header)

        self.body = QWidget(self)
        self.body.setObjectName("FineTuneGroupBody")
        self.body.setStyleSheet("QWidget#FineTuneGroupBody { background: transparent; border: none; }")
        self.vbox = QVBoxLayout(self.body)
        self.vbox.setContentsMargins(0, 0, 0, 0)
        self.vbox.setSpacing(4)
        outer.addWidget(self.body)
        self._refresh_header()

    def _refresh_header(self) -> None:
        dot = " ●" if self._has_active else ""
        safe_title = self._title.replace("&&", "&").replace("&", "&&")
        if self._collapsible:
            arrow = "▾ " if self._expanded else "▸ "
            self.btn_header.setText(f"{arrow}{safe_title}{dot}")
            self.btn_header.setCursor(Qt.PointingHandCursor)
        else:
            self.btn_header.setText(f"{safe_title}{dot}")
            self.btn_header.setCursor(Qt.ArrowCursor)

    def _on_header_clicked(self) -> None:
        if self._collapsible:
            self.set_expanded(not self._expanded)

    def set_title(self, title: str) -> None:
        self._title = title
        self._refresh_header()

    def set_collapsible(self, collapsible: bool) -> None:
        self._collapsible = bool(collapsible)
        if not self._collapsible:
            self.set_expanded(True)
        self._refresh_header()

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = bool(expanded)
        self.body.setVisible(self._expanded)
        self._refresh_header()

    def set_active_indicator(self, active: bool) -> None:
        active_b = bool(active)
        if self._has_active != active_b:
            self._has_active = active_b
            self._refresh_header()


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


class _RelightSphereButton(QToolButton):
    """1-click '3D Relight' look preset button with custom-shaded 3D sphere preview."""

    def __init__(self, preset: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.preset = preset
        c1, c2 = preset.get("preview_colors", ("#ffe8d5", "#23252a"))
        style = str(preset.get("preview_style", preset.get("id", "")))
        self._pix = self._render_sphere_pixmap(style, c1, c2)
        self.setIcon(QIcon(self._pix))
        self.setIconSize(QSize(32, 32))
        self.setText(str(preset.get("name", "Look")))
        self.setCheckable(True)
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(62)
        set_ui_role(self, "card")
        self.setStyleSheet(
            """
            QToolButton {
                font-size: 8pt;
                padding: 3px 2px;
                border-radius: 5px;
                border: 1px solid rgba(128, 145, 165, 0.25);
            }
            QToolButton:hover {
                border-color: rgba(59, 130, 246, 0.65);
                background: rgba(59, 130, 246, 0.08);
            }
            QToolButton:checked {
                border: 1.5px solid #2563EB;
                background: rgba(37, 99, 235, 0.16);
                font-weight: 600;
            }
            """
        )

    @staticmethod
    def _render_sphere_pixmap(style: str, hi_hex: str, lo_hex: str) -> QPixmap:
        pix = QPixmap(34, 34)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing, True)
        sphere_rect = QRectF(2.0, 2.0, 30.0, 30.0)
        clip = QPainterPath()
        clip.addEllipse(sphere_rect)
        p.setClipPath(clip)

        if style == "dual_neon":
            g_l = QRadialGradient(QPointF(8.0, 17.0), 22.0, QPointF(7.0, 17.0))
            g_l.setColorAt(0.0, QColor(hi_hex))
            g_l.setColorAt(0.75, QColor("#181b24"))
            g_l.setColorAt(1.0, QColor("#12141c"))
            p.fillRect(sphere_rect, QBrush(g_l))
            g_r = QRadialGradient(QPointF(26.0, 17.0), 20.0, QPointF(27.0, 17.0))
            rc = QColor(lo_hex)
            rc.setAlpha(210)
            g_r.setColorAt(0.0, rc)
            rc2 = QColor(lo_hex)
            rc2.setAlpha(0)
            g_r.setColorAt(1.0, rc2)
            p.fillRect(sphere_rect, QBrush(g_r))
        elif style in ("golden_rim", "moonlit_backlight"):
            grad = QRadialGradient(QPointF(15.0, 17.0), 17.0, QPointF(14.0, 17.0))
            grad.setColorAt(0.0, QColor(lo_hex))
            grad.setColorAt(0.7, QColor("#12151f"))
            grad.setColorAt(1.0, QColor(hi_hex))
            p.fillRect(sphere_rect, QBrush(grad))
            rim_pen = QPen(QColor(hi_hex), 2.8)
            p.setPen(rim_pen)
            p.setBrush(Qt.NoBrush)
            if style == "golden_rim":
                p.drawArc(sphere_rect.adjusted(1.5, 1.5, -1.5, -1.5), -50 * 16, 120 * 16)
            else:
                p.drawArc(sphere_rect.adjusted(1.5, 1.5, -1.5, -1.5), 35 * 16, 110 * 16)
        elif style == "window_gobo":
            grad = QRadialGradient(QPointF(11.0, 11.0), 20.0, QPointF(10.0, 10.0))
            grad.setColorAt(0.0, QColor("#ffffff"))
            grad.setColorAt(0.35, QColor(hi_hex))
            grad.setColorAt(1.0, QColor(lo_hex))
            p.fillRect(sphere_rect, QBrush(grad))
            p.setPen(QPen(QColor(25, 22, 20, 200), 2.6))
            p.drawLine(QPointF(17.0, 2.0), QPointF(17.0, 32.0))
            p.drawLine(QPointF(2.0, 17.0), QPointF(32.0, 17.0))
        else:
            fx = 10.0 if style == "warm_streetlamp" else 12.0
            fy = 9.0 if style == "warm_streetlamp" else 11.0
            grad = QRadialGradient(QPointF(fx, fy), 20.0, QPointF(fx - 1.0, fy - 1.0))
            grad.setColorAt(0.0, QColor("#ffffff"))
            grad.setColorAt(0.30, QColor(hi_hex))
            grad.setColorAt(1.0, QColor(lo_hex))
            p.fillRect(sphere_rect, QBrush(grad))

        p.setClipping(False)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(130, 145, 165, 170), 1.1))
        p.drawEllipse(sphere_rect)
        p.end()
        return pix


class _ClickableFolderLabel(QLabel):
    """Information-pane folder link label that emits `clicked` on left click when enabled."""

    clicked = Signal()

    def __init__(self, text: str = "-", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._clickable: bool = False
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.setWordWrap(True)

    def set_clickable(self, clickable: bool) -> None:
        val = bool(clickable)
        self._clickable = val
        self.setCursor(Qt.CursorShape.PointingHandCursor if val else Qt.CursorShape.ArrowCursor)

    def is_clickable(self) -> bool:
        return self._clickable

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (
            self._clickable
            and event.button() == Qt.MouseButton.LeftButton
            and self.rect().contains(event.position().toPoint())
        ):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class TagAndCaptionDock(QDockWidget):
    """
    Unified Right Sidebar for PhotoGallery64:
    - Mode 'info': Tag and Caption / Information (Ctrl+I)
    - Mode 'fine_tune': Fine-Tune & Adjustments with Live Histogram (Ctrl+F)
    """

    # Signals for information editing
    rating_edited = Signal(int)             # 0..5
    flag_edited = Signal(int)               # -1, 0, or 1
    favourite_edited = Signal(bool)         # True or False
    caption_committed = Signal(str)         # new caption string
    tag_added = Signal(str)                 # full_path or tag name
    tag_removed = Signal(int)               # tag_id
    person_added = Signal(str)              # person name
    person_removed = Signal(int)            # person_id
    geotag_edited = Signal(str)             # location_name
    change_time_requested = Signal()        # user clicked Date Taken link
    filename_committed = Signal(str)        # renamed filename
    folder_open_requested = Signal(str)     # user clicked Folder link to open in Explorer
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
    fine_tune_exposure_ext_changed = Signal(int, int)              # whites, blacks
    fine_tune_smart_enhance_changed = Signal(int)                  # smart_enhance (0..100)
    fine_tune_structure_changed = Signal(int)                      # structure (-100..100)
    fine_tune_color_changed = Signal(int, int, int, int)           # temp, tint, saturation, dehaze
    fine_tune_straighten_changed = Signal(float)                   # angle
    fine_tune_detail_changed = Signal(int, int)                    # sharpen, denoise
    fine_tune_detail_ext_changed = Signal(int, int, int, int, int, int) # denoise_chroma, grain_amount, grain_size, sharpen_masking, sharpen_radius, denoise_detail
    fine_tune_curve_changed = Signal(dict)                         # {"rgb": [...], "r": [...], "g": [...], "b": [...]}
    fine_tune_hsl_changed = Signal(dict, dict, dict)               # hue, sat, lum dicts
    fine_tune_region_changed = Signal(str, dict)                   # region_name, region_dict
    region_scope_changed = Signal(str)                             # "whole"|"subject"|"background"|"sky"
    region_mask_preview_toggled = Signal(str)                      # "subject"|"background"|"sky"|""
    region_mask_fill_toggled = Signal(bool)                        # False=outline only, True=red fill+outline
    region_brush_mode_changed = Signal(str, str, int, bool)        # region, mode ("paint"|"erase"|""), brush_size, smart_select
    region_brush_cleared = Signal(str)                             # region
    region_mask_invert_requested = Signal(str)                     # region
    wb_eyedropper_toggled = Signal(bool)
    fine_tune_look_profile_changed = Signal(str, int)              # effect_key_or_cube, intensity (0..200)
    fine_tune_optics_changed = Signal(int)                         # vignette
    fine_tune_optics_ext_changed = Signal(int, int, int, int)      # vignette, vignette_feather, orton_glow, golden_hour
    fine_tune_levels_changed = Signal(int, int)                    # black_point, white_point
    fine_tune_reset_requested = Signal()                           # reset all adjustments to default
    fine_tune_auto_adjust = Signal()
    auto_straighten_requested = Signal()
    draw_line_straighten_requested = Signal()
    analyze_noise_requested = Signal()
    compare_started = Signal()
    compare_ended = Signal()
    fine_tune_slider_pressed = Signal()
    fine_tune_slider_released = Signal(str)
    dock_mode_changed = Signal(str)  # "info"|"fine_tune"|"depth_light"|"" (when hidden)
    pane_interacted = Signal()       # emitted when user clicks inside the right pane (excluding Add people input)

    # Signals for Depth & Light adjustments
    depth_bokeh_changed = Signal(int, int, int, int, int)    # blur, focal_plane, glow, boundary_shift, boundary_feather
    depth_relight_changed = Signal(int, int, int, int)       # near, far, warmth_near, warmth_far
    depth_virtual_lights_changed = Signal(int, list, bool)   # ambient_dim, virtual_lights, commit
    depth_light_gizmos_toggled = Signal(bool)
    depth_selected_light_changed = Signal(int)
    depth_relight_clear_requested = Signal()
    depth_haze_changed = Signal(int, int)                    # haze, haze_warmth
    depth_reset_requested = Signal()
    copy_subject_requested = Signal()
    remove_background_requested = Signal()
    mask_preview_toggled = Signal(bool)                      # live ruby mask overlay
    map_location_selected = Signal(list, str, float, float)  # media_ids, label, lat, lon
    adjust_geotag_requested = Signal()
    ocr_rescan_requested = Signal(str)                       # preferred backend key
    ocr_text_edited = Signal(str)                            # edited or cleared OCR text

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Tag and caption", parent)
        self.setObjectName("TagAndCaptionDock")
        self.setFeatures(
            QDockWidget.DockWidgetClosable
        )
        self.setTitleBarWidget(QWidget(self))  # Hide default ugly dock titlebar
        self.setMinimumWidth(240)
        self.setMaximumWidth(700)

        self._current_mode: Literal["info", "fine_tune", "depth_light"] = "info"
        self._selected_ids: list[int] = []
        self._current_rating: int = 0
        self._current_flag: int = 0
        self._current_favourite: bool = False
        self._favourites_enabled: bool = True
        self._total_view_count: int = 0
        self._updating_ui: bool = False
        self._original_filename: str = ""
        self._current_date_taken_ts: int = 0
        self._last_caption_text: str = ""
        self._last_valid_geotag: str = ""

        # Real-time event throttler/coalescer to maintain responsive 60 FPS interactions
        self._dl_timer = QTimer(self)
        self._dl_timer.setSingleShot(True)
        self._dl_timer.setInterval(16)
        self._dl_timer.timeout.connect(self._flush_depth_light_emits)
        self._pending_dl_bokeh = False
        self._pending_dl_relight = False
        self._pending_dl_virtual_lights = False
        self._pending_dl_haze = False
        self._dl_virtual_lights: list[dict[str, Any]] = []
        self._dl_selected_light_idx: int = 0

        self._ft_timer = QTimer(self)
        self._ft_timer.setSingleShot(True)
        self._ft_timer.setInterval(16)
        self._ft_timer.timeout.connect(self._flush_fine_tune_emits)
        self._pending_ft_exposure = False
        self._pending_ft_exposure_ext = False
        self._pending_ft_smart_enhance = False
        self._pending_ft_structure = False
        self._pending_ft_color = False
        self._pending_ft_optics = False
        self._pending_ft_straighten = False
        self._pending_ft_detail = False
        self._pending_ft_curve = False
        self._pending_ft_hsl = False
        self._pending_ft_detail_ext = False
        self._pending_ft_region = False
        self._pending_ft_look = False
        self._fine_tune_advanced_mode = False
        self._active_region_scope: str = "whole"
        self._last_preview_region: str = "subject"
        self._region_edits: dict[str, dict[str, Any]] = {}
        self._global_region_mirror: dict[str, Any] = {
            "brightness": 0,
            "contrast": 0,
            "shadows": 0,
            "highlights": 0,
            "whites": 0,
            "blacks": 0,
            "clarity": 0,
            "temperature": 0,
            "tint": 0,
            "saturation": 0,
            "dehaze": 0,
            "curve_rgb": [(0.0, 0.0), (1.0, 1.0)],
            "curve_r": [(0.0, 0.0), (1.0, 1.0)],
            "curve_g": [(0.0, 0.0), (1.0, 1.0)],
            "curve_b": [(0.0, 0.0), (1.0, 1.0)],
        }

        # Outer Container & Header Row with close '✕' button
        self.container = QFrame(self)
        self.container.setObjectName("RightSidebarContainer")
        self.container.setStyleSheet(
            """
            QFrame#RightSidebarContainer {
                background-color: #FFFFFF;
                border-left: 1px solid qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #DCE7F5, stop:0.08 #B7D1EE, stop:0.92 #B7D1EE, stop:1 #DCE7F5);
            }
            """
        )
        outer_vbox = QVBoxLayout(self.container)
        outer_vbox.setContentsMargins(1, 0, 0, 0)
        outer_vbox.setSpacing(0)

        # Subtle Header Row with small '✕' button in top right (matches WLPG screenshots)
        header_row = QWidget(self.container)
        header_row.setFixedHeight(18)
        h_layout = QHBoxLayout(header_row)
        h_layout.setContentsMargins(6, 1, 6, 0)
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

        # Mode Stack: Page 0 = Information & Tags, Page 1 = Fine-Tune, Page 2 = Depth & Light, Page 3 = Custom Inspector
        self.mode_stack = QStackedWidget(self.container)

        self.page_info = self._build_info_page()
        self.page_fine_tune = self._build_fine_tune_page()
        self.page_depth_light = self._build_depth_light_page()
        self.page_custom_inspector = self._build_custom_inspector_page()

        self.mode_stack.addWidget(self.page_info)
        self.mode_stack.addWidget(self.page_fine_tune)
        self.mode_stack.addWidget(self.page_depth_light)
        self.mode_stack.addWidget(self.page_custom_inspector)
        outer_vbox.addWidget(self.mode_stack, 1)

        self.setWidget(self.container)
        self._apply_stylesheet()
        self.set_dark_mode(False)
        self.container.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.MouseButtonPress and watched is self.container:
            if not (hasattr(self, "txt_add_person") and self.txt_add_person.underMouse()):
                self.pane_interacted.emit()
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event: Any) -> None:
        if not (hasattr(self, "txt_add_person") and self.txt_add_person.underMouse()):
            self.pane_interacted.emit()
        super().mousePressEvent(event)

    def sizeHint(self) -> QSize:
        return QSize(310, 600) if getattr(self, "_current_mode", "info") in ("fine_tune", "depth_light", "custom_inspector") else QSize(280, 600)

    def minimumSizeHint(self) -> QSize:
        min_w = 295 if getattr(self, "_current_mode", "info") == "custom_inspector" else (
            285 if getattr(self, "_current_mode", "info") in ("fine_tune", "depth_light") else 240
        )
        return QSize(min_w, 300)

    def keyPressEvent(self, event: Any) -> None:
        if event.key() in (Qt.Key_Left, Qt.Key_Right):
            focus_w = self.focusWidget() or QApplication.focusWidget()
            if isinstance(focus_w, (QLineEdit, QTextEdit, QPlainTextEdit)):
                super().keyPressEvent(event)
                return
            event.ignore()
            return
        super().keyPressEvent(event)

    def _build_custom_inspector_page(self) -> QWidget:
        page = QWidget(self.mode_stack)
        page.setObjectName("CustomInspectorPage")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 4, 8, 8)
        lay.setSpacing(6)

        self.lbl_custom_inspector_title = QLabel("", page)
        self.lbl_custom_inspector_title.setFont(QFont("Segoe UI", 12, QFont.DemiBold))
        self.lbl_custom_inspector_title.setProperty("uiRole", "heading")
        self.lbl_custom_inspector_title.hide()
        lay.addWidget(self.lbl_custom_inspector_title)

        self.custom_inspector_scroll = QScrollArea(page)
        self.custom_inspector_scroll.setObjectName("StudioInspectorScroll")
        self.custom_inspector_scroll.setWidgetResizable(True)
        self.custom_inspector_scroll.setFrameShape(QFrame.NoFrame)
        lay.addWidget(self.custom_inspector_scroll, 1)
        return page

    def set_custom_inspector(self, widget: QWidget | None, title: str = "") -> None:
        """Host an external Create Studio workspace inspector panel inside RightTagDock."""
        self.clear_custom_inspector()
        if title:
            self.lbl_custom_inspector_title.setText(title)
            self.lbl_custom_inspector_title.show()
        else:
            self.lbl_custom_inspector_title.hide()
        if widget is not None:
            self.custom_inspector_scroll.setWidget(widget)
            theme_name = getattr(self, "_theme_name", "dark" if getattr(self, "_is_dark", False) else "wlpg_aero")
            if hasattr(widget, "apply_theme_mode"):
                widget.apply_theme_mode(theme_name)
        self.set_mode("custom_inspector")

    def clear_custom_inspector(self) -> None:
        """Detach hosted inspector widget cleanly without deleting workspace-owned objects."""
        if hasattr(self, "custom_inspector_scroll"):
            old = self.custom_inspector_scroll.takeWidget()
            if old is not None:
                old.setParent(None)
        if hasattr(self, "lbl_custom_inspector_title"):
            self.lbl_custom_inspector_title.hide()

    def set_mode(self, mode: Literal["info", "fine_tune", "depth_light", "custom_inspector"]) -> None:
        """Switches between Tag & Caption (Ctrl+I), Fine-Tune (Ctrl+F), Depth & Light, and Custom Inspector."""
        prev_mode = getattr(self, "_current_mode", "info")
        self._current_mode = mode
        if prev_mode == "fine_tune" and mode != "fine_tune":
            self.clear_mask_editing_session()
        if mode == "custom_inspector":
            self.setMinimumWidth(295)
            self.mode_stack.setCurrentIndex(3)
        elif mode == "depth_light":
            self.setMinimumWidth(285)
            self.mode_stack.setCurrentIndex(2)
        elif mode == "fine_tune":
            self.setMinimumWidth(285)
            self.mode_stack.setCurrentIndex(1)
        else:
            self.setMinimumWidth(240)
            self.mode_stack.setCurrentIndex(0)
        self.updateGeometry()
        self.show()
        self.dock_mode_changed.emit(mode)

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self.dock_mode_changed.emit(getattr(self, "_current_mode", "info"))

    def hideEvent(self, event: Any) -> None:
        if getattr(self, "_current_mode", "info") == "fine_tune":
            self.clear_mask_editing_session()
        self.dock_mode_changed.emit("")
        super().hideEvent(event)

    def clear_mask_editing_session(self) -> None:
        """Clears active mask overlay & brush editing when Fine Tune (Advanced) is closed."""
        self._clear_region_brush_buttons()
        self._set_mask_preview_visible(False)
        self.region_mask_preview_toggled.emit("")
        self.region_brush_mode_changed.emit("", "", self.slider_region_brush_size.value() if hasattr(self, "slider_region_brush_size") else 40, True)
        if getattr(self, "_active_region_scope", "whole") != "whole":
            self.set_active_region_scope("whole")

    def focus_filename_rename(self) -> None:
        """Switches to Information mode, reveals dock, and focuses filename input with select-all."""
        self.set_mode("info")
        self.show()
        self.raise_()
        if hasattr(self, "stack_filename") and hasattr(self, "txt_filename"):
            self.stack_filename.start_edit()
            self.txt_filename.setFocus()
            self.txt_filename.selectAll()

    def focus_geotag_input(self) -> None:
        """Switches to Information mode, reveals dock, and focuses geotag input so edit actions appear."""
        self.set_mode("info")
        self.show()
        self.raise_()
        if hasattr(self, "txt_geotag"):
            self.txt_geotag.setFocus()
            self.txt_geotag.selectAll()

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
        self.txt_geotag.adjust_pin_requested.connect(self.adjust_geotag_requested.emit)
        top_vbox.addWidget(self.txt_geotag)

        # 3. Caption
        lbl_cap = QLabel("Caption", top_content)
        lbl_cap.setObjectName("SectionTitle")
        top_vbox.addWidget(lbl_cap)

        self.txt_caption = QLineEdit(top_content)
        self.txt_caption.setObjectName("InlineEdit")
        self.txt_caption.setPlaceholderText("Add caption")
        self.txt_caption.returnPressed.connect(self._on_caption_return_pressed)
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
            "color: #475569; font-family: 'Segoe UI'; font-size: 9pt; padding-top: 2px;"
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
            "color: #64748B; font-family: 'Segoe UI'; font-size: 9pt;"
        )
        form_vbox.addWidget(self.lbl_selection_header)

        form = QFormLayout()
        form.setContentsMargins(0, 2, 0, 2)
        form.setSpacing(4)
        form.setLabelAlignment(Qt.AlignLeft)

        def make_val_lbl() -> QLabel:
            lbl = QLabel("-", self.info_form_frame)
            lbl.setStyleSheet("color: #1E293B; font-family: 'Segoe UI'; font-size: 9pt;")
            lbl.setWordWrap(True)
            return lbl

        def make_field_lbl(text: str) -> QLabel:
            lbl = QLabel(text, self.info_form_frame)
            lbl.setStyleSheet("color: #475569; font-family: 'Segoe UI'; font-size: 9pt;")
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

        self._current_folder_path: str = ""
        self._folder_is_multiple: bool = False
        self.lbl_folder = _ClickableFolderLabel("-", self.info_form_frame)
        self.lbl_folder.setObjectName("InfoFolderLinkLabel")
        self.lbl_folder.clicked.connect(self._on_folder_label_clicked)
        form.addRow(make_field_lbl("Folder:"), self.lbl_folder)
        self._update_folder_link_style()

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

        # Flag & Favourite row (Picked / Rejected / Favourite)
        flag_widget = QWidget(self.info_form_frame)
        flag_hbox = QHBoxLayout(flag_widget)
        flag_hbox.setContentsMargins(0, 0, 0, 0)
        flag_hbox.setSpacing(4)

        self.btn_flag = QPushButton(flag_widget)
        self.btn_flag.setFixedSize(24, 22)
        self.btn_flag.setFlat(True)
        self.btn_flag.setCursor(Qt.PointingHandCursor)
        self.btn_flag.setIconSize(QSize(15, 15))
        self.btn_flag.setToolTip("Cycle Flag: Pick → Reject → Unflagged (P / X / U)")
        self.btn_flag.clicked.connect(self._on_flag_button_clicked)
        flag_hbox.addWidget(self.btn_flag)

        self.btn_info_fav = QPushButton(flag_widget)
        self.btn_info_fav.setFixedSize(24, 22)
        self.btn_info_fav.setFlat(True)
        self.btn_info_fav.setCursor(Qt.PointingHandCursor)
        self.btn_info_fav.setIconSize(QSize(15, 15))
        self.btn_info_fav.setToolTip("Favourite (F)")
        self.btn_info_fav.clicked.connect(self._on_favourite_button_clicked)
        flag_hbox.addWidget(self.btn_info_fav)

        flag_hbox.addStretch(1)
        form.addRow(make_field_lbl("Flag:"), flag_widget)
        self._update_flag_button(0, False)

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

        # Collapsible OCR "Text in Photo" box (shown ONLY when single selected photo has non-empty ocr_text)
        self.ocr_section_widget = QFrame(bottom_content)
        self.ocr_section_widget.setObjectName("OcrSectionWidget")
        ocr_vbox = QVBoxLayout(self.ocr_section_widget)
        ocr_vbox.setContentsMargins(0, 6, 0, 0)
        ocr_vbox.setSpacing(4)

        ocr_hdr = QHBoxLayout()
        ocr_hdr.setContentsMargins(0, 0, 0, 0)
        self.btn_ocr_toggle = QToolButton(self.ocr_section_widget)
        self.btn_ocr_toggle.setObjectName("OcrToggleBtn")
        self.btn_ocr_toggle.setText("▾ Text in Photo (OCR)")
        self.btn_ocr_toggle.setCheckable(True)
        self.btn_ocr_toggle.setChecked(True)
        self.btn_copy_ocr = QToolButton(self.ocr_section_widget)
        self.btn_copy_ocr.setObjectName("OcrCopyBtn")
        self.btn_copy_ocr.setText("Copy Text")
        self.btn_copy_ocr.setPopupMode(QToolButton.MenuButtonPopup)
        self.btn_copy_ocr.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.btn_copy_ocr.setFixedHeight(22)

        self._ocr_menu = QMenu(self.btn_copy_ocr)
        act_copy_ocr = self._ocr_menu.addAction("Copy Text")
        act_edit_ocr = self._ocr_menu.addAction("Edit OCR Text")
        rescan_menu = self._ocr_menu.addMenu("Rescan with...")
        for lbl, key in (
            ("Auto (Best Available)", "auto"),
            ("RapidOCR (ONNX)", "rapidocr"),
            ("Windows OCR", "windows_media_ocr"),
            ("Tesseract", "pytesseract"),
        ):
            act_r = rescan_menu.addAction(lbl)
            act_r.triggered.connect(lambda _c=False, b=key: self.ocr_rescan_requested.emit(b))
        self._ocr_menu.addSeparator()
        act_remove_ocr = self._ocr_menu.addAction("Remove OCR Text")
        self.btn_copy_ocr.setMenu(self._ocr_menu)

        ocr_hdr.addWidget(self.btn_ocr_toggle)
        ocr_hdr.addStretch(1)
        ocr_hdr.addWidget(self.btn_copy_ocr)
        ocr_vbox.addLayout(ocr_hdr)

        self.ocr_edit_bar = QWidget(self.ocr_section_widget)
        ocr_edit_row = QHBoxLayout(self.ocr_edit_bar)
        ocr_edit_row.setContentsMargins(0, 0, 0, 0)
        ocr_edit_row.setSpacing(4)
        ocr_edit_row.addStretch(1)
        self.btn_ocr_save_edit = QPushButton("Save", self.ocr_edit_bar)
        self.btn_ocr_save_edit.setObjectName("FineTuneActionBtn")
        self.btn_ocr_save_edit.setFixedHeight(20)
        self.btn_ocr_save_edit.setToolTip("Save OCR text (Ctrl+Enter)")
        self.btn_ocr_cancel_edit = QPushButton("Cancel", self.ocr_edit_bar)
        self.btn_ocr_cancel_edit.setObjectName("FineTuneResetBtn")
        self.btn_ocr_cancel_edit.setFixedHeight(20)
        self.btn_ocr_cancel_edit.setToolTip("Cancel editing (Esc)")
        ocr_edit_row.addWidget(self.btn_ocr_save_edit)
        ocr_edit_row.addWidget(self.btn_ocr_cancel_edit)
        self.ocr_edit_bar.setVisible(False)
        ocr_vbox.addWidget(self.ocr_edit_bar)

        self.txt_ocr_preview = QPlainTextEdit(self.ocr_section_widget)
        self.txt_ocr_preview.setObjectName("OcrPreviewEdit")
        self.txt_ocr_preview.setReadOnly(True)
        self.txt_ocr_preview.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        self.txt_ocr_preview.setMaximumHeight(92)
        ocr_vbox.addWidget(self.txt_ocr_preview)
        self._ocr_pre_edit_text: str = ""

        def _refresh_ocr_edit_style() -> None:
            st = self.txt_ocr_preview.style()
            if st is not None:
                st.unpolish(self.txt_ocr_preview)
                st.polish(self.txt_ocr_preview)

        def _toggle_ocr_box(expanded: bool) -> None:
            self.txt_ocr_preview.setVisible(expanded)
            if not expanded and self.ocr_edit_bar.isVisible():
                _cancel_ocr_edit()
            self.btn_ocr_toggle.setText("▾ Text in Photo (OCR)" if expanded else "▸ Text in Photo (OCR)")

        def _copy_ocr_text() -> None:
            txt = self.txt_ocr_preview.toPlainText().strip()
            if txt:
                QApplication.clipboard().setText(txt)
                self.btn_copy_ocr.setText("Copied ✓")
                QTimer.singleShot(1500, lambda: self.btn_copy_ocr.setText("Copy Text"))

        def _begin_ocr_edit() -> None:
            self._ocr_pre_edit_text = self.txt_ocr_preview.toPlainText()
            if not self.btn_ocr_toggle.isChecked():
                self.btn_ocr_toggle.setChecked(True)
            self.txt_ocr_preview.setReadOnly(False)
            self.txt_ocr_preview.setTextInteractionFlags(Qt.TextEditorInteraction)
            _refresh_ocr_edit_style()
            self.ocr_edit_bar.setVisible(True)
            bottom_scroll.ensureWidgetVisible(self.ocr_edit_bar)
            self.txt_ocr_preview.setFocus()

        def _save_ocr_edit() -> None:
            if self.txt_ocr_preview.isReadOnly():
                return
            new_txt = self.txt_ocr_preview.toPlainText().strip()
            self.txt_ocr_preview.setReadOnly(True)
            self.txt_ocr_preview.setTextInteractionFlags(
                Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
            )
            _refresh_ocr_edit_style()
            self.ocr_edit_bar.setVisible(False)
            self.ocr_text_edited.emit(new_txt)

        def _cancel_ocr_edit() -> None:
            self.txt_ocr_preview.setPlainText(self._ocr_pre_edit_text)
            self.txt_ocr_preview.setReadOnly(True)
            self.txt_ocr_preview.setTextInteractionFlags(
                Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
            )
            _refresh_ocr_edit_style()
            self.ocr_edit_bar.setVisible(False)

        def _remove_ocr_text() -> None:
            if self.ocr_edit_bar.isVisible():
                _cancel_ocr_edit()
            self.txt_ocr_preview.clear()
            self.ocr_section_widget.setVisible(False)
            self.ocr_text_edited.emit("")

        self.btn_ocr_toggle.toggled.connect(_toggle_ocr_box)
        self.btn_copy_ocr.clicked.connect(_copy_ocr_text)
        act_copy_ocr.triggered.connect(_copy_ocr_text)
        act_edit_ocr.triggered.connect(_begin_ocr_edit)
        act_remove_ocr.triggered.connect(_remove_ocr_text)
        self.btn_ocr_save_edit.clicked.connect(_save_ocr_edit)
        self.btn_ocr_cancel_edit.clicked.connect(_cancel_ocr_edit)
        QShortcut(QKeySequence("Ctrl+Return"), self.txt_ocr_preview, _save_ocr_edit, context=Qt.WidgetShortcut)
        QShortcut(QKeySequence("Ctrl+Enter"), self.txt_ocr_preview, _save_ocr_edit, context=Qt.WidgetShortcut)
        QShortcut(QKeySequence(Qt.Key_Escape), self.txt_ocr_preview, _cancel_ocr_edit, context=Qt.WidgetShortcut)
        self.ocr_section_widget.setVisible(False)
        bottom_vbox.addWidget(self.ocr_section_widget)

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
        page_wrap = QWidget()
        page_vbox = QVBoxLayout(page_wrap)
        page_vbox.setContentsMargins(0, 0, 0, 0)
        page_vbox.setSpacing(4)

        # Header Bar: Basic | Advanced segmented toggle (always pinned at top)
        mode_bar_wrap = QWidget(page_wrap)
        mode_bar = QHBoxLayout(mode_bar_wrap)
        mode_bar.setContentsMargins(8, 0, 8, 2)
        mode_bar.setSpacing(4)
        lbl_mode = QLabel("Mode:", mode_bar_wrap)
        lbl_mode.setObjectName("SliderHeaderLabel")
        mode_bar.addWidget(lbl_mode)
        mode_bar.addStretch(1)

        self.btn_ft_basic = QPushButton("Basic", mode_bar_wrap)
        self.btn_ft_basic.setObjectName("FineTunePillBtn")
        self.btn_ft_basic.setCheckable(True)
        self.btn_ft_basic.setFixedHeight(22)
        self.btn_ft_basic.setCursor(Qt.PointingHandCursor)
        self.btn_ft_basic.clicked.connect(lambda: self.set_fine_tune_advanced_mode(False))

        self.btn_ft_advanced = QPushButton("Advanced", mode_bar_wrap)
        self.btn_ft_advanced.setObjectName("FineTunePillBtn")
        self.btn_ft_advanced.setCheckable(True)
        self.btn_ft_advanced.setFixedHeight(22)
        self.btn_ft_advanced.setCursor(Qt.PointingHandCursor)
        self.btn_ft_advanced.clicked.connect(lambda: self.set_fine_tune_advanced_mode(True))

        mode_bar.addWidget(self.btn_ft_basic)
        self.btn_ft_advanced.setFixedWidth(74)
        self.btn_ft_basic.setFixedWidth(56)
        mode_bar.addWidget(self.btn_ft_advanced)
        page_vbox.addWidget(mode_bar_wrap)

        # Pinned Histogram slot (Advanced mode: freezes Histogram above the scroll area)
        self.ft_pinned_hist_slot = QFrame(page_wrap)
        self.ft_pinned_hist_slot.setObjectName("FineTuneGroup")
        self._pinned_hist_layout = QVBoxLayout(self.ft_pinned_hist_slot)
        self._pinned_hist_layout.setContentsMargins(8, 4, 8, 4)
        self._pinned_hist_layout.setSpacing(2)
        self.ft_pinned_hist_slot.setVisible(False)
        page_vbox.addWidget(self.ft_pinned_hist_slot)

        scroll = _NoHorizontalScrollArea(page_wrap)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page_vbox.addWidget(scroll, 1)

        content = QWidget(scroll)
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(8, 0, 8, 8)
        vbox.setSpacing(8)

        # Phase 1 Regional Masking Scope Bar (Visible in Advanced Mode)
        # Group 0 (Advanced): Mask (styled identically to Adjust exposure)
        self._grp_mask = _FineTuneGroup("Mask", content)
        self.region_scope_widget = QWidget(self._grp_mask)
        reg_vbox = QVBoxLayout(self.region_scope_widget)
        reg_vbox.setContentsMargins(0, 2, 0, 2)
        reg_vbox.setSpacing(7)

        # Row 1: Mask: [ Show | Hide ]
        mask_vis_row = QHBoxLayout()
        mask_vis_row.setContentsMargins(0, 0, 0, 0)
        mask_vis_row.setSpacing(6)
        lbl_mvis = QLabel("Mask:", self.region_scope_widget)
        lbl_mvis.setObjectName("SliderHeaderLabel")
        lbl_mvis.setFixedWidth(68)
        mask_vis_row.addWidget(lbl_mvis)

        vis_seg = QHBoxLayout()
        vis_seg.setContentsMargins(0, 0, 0, 0)
        vis_seg.setSpacing(4)
        self.btn_mask_show = QPushButton("Show", self.region_scope_widget)
        self.btn_mask_show.setObjectName("FineTuneSubBtn")
        self.btn_mask_show.setCheckable(True)
        self.btn_mask_show.setFixedHeight(24)
        self.btn_mask_show.setCursor(Qt.PointingHandCursor)
        self.btn_mask_show.setToolTip("Show mask overlay on the photo (even when Whole Photo is selected)")

        self.btn_mask_hide = QPushButton("Hide", self.region_scope_widget)
        self.btn_mask_hide.setObjectName("FineTuneSubBtn")
        self.btn_mask_hide.setCheckable(True)
        self.btn_mask_hide.setChecked(True)
        self.btn_mask_hide.setFixedHeight(24)
        self.btn_mask_hide.setCursor(Qt.PointingHandCursor)
        self.btn_mask_hide.setToolTip("Hide mask overlay (adjustments still apply to the selected region)")

        self.btn_mask_fill = QPushButton("", self.region_scope_widget)
        self.btn_mask_fill.setObjectName("FineTuneSubBtn")
        self.btn_mask_fill.setCheckable(True)
        self.btn_mask_fill.setChecked(True)
        self.btn_mask_fill.setFixedSize(26, 24)
        self.btn_mask_fill.setIcon(self._make_mask_fill_icon(True, 16))
        self.btn_mask_fill.setIconSize(QSize(16, 16))
        self.btn_mask_fill.setCursor(Qt.PointingHandCursor)
        self.btn_mask_fill.setToolTip("Toggle red mask fill (on = red fill + outline, off = dotted outline only; temporarily hides fill while dragging sliders)")
        self.btn_mask_fill.toggled.connect(self._on_mask_fill_toggled)

        # Alias kept for backward compatibility
        self.btn_region_mask_preview = self.btn_mask_show
        self.btn_mask_show.clicked.connect(lambda _=False: self._set_mask_preview_visible(True))
        self.btn_mask_hide.clicked.connect(lambda _=False: self._set_mask_preview_visible(False))
        vis_seg.addWidget(self.btn_mask_show, 1)
        vis_seg.addWidget(self.btn_mask_hide, 1)
        vis_seg.addWidget(self.btn_mask_fill, 0)
        mask_vis_row.addLayout(vis_seg, 1)
        reg_vbox.addLayout(mask_vis_row)

        # Row 2: Auto Mask: Pure Icon-Only Tiles (hover tooltip shows name & description)
        auto_row = QHBoxLayout()
        auto_row.setContentsMargins(0, 0, 0, 0)
        auto_row.setSpacing(6)
        lbl_auto = QLabel("Auto Mask:", self.region_scope_widget)
        lbl_auto.setObjectName("SliderHeaderLabel")
        lbl_auto.setFixedWidth(68)
        auto_row.addWidget(lbl_auto)

        reg_pills = QHBoxLayout()
        reg_pills.setSpacing(4)
        reg_pills.setContentsMargins(0, 0, 0, 0)
        self._region_scope_buttons: dict[str, QPushButton] = {}
        self._region_scope_tips = {
            "whole": "Whole Photo\nApply Fine Tune adjustments to the entire image",
            "subject": "Subject\nAutomatically isolate and adjust the main person or subject",
            "background": "Background\nAutomatically isolate and adjust the background behind the subject",
            "sky": "Sky\nAutomatically detect and adjust the sky region",
        }
        for r_key in ("whole", "subject", "background", "sky"):
            rb = QPushButton("", self.region_scope_widget)
            rb.setObjectName("FineTuneIconTileBtn")
            rb.setCheckable(True)
            rb.setChecked(r_key == "whole")
            rb.setFixedHeight(30)
            rb.setIcon(self._make_mask_tool_icon(r_key, 20, is_checked=(r_key == "whole")))
            rb.setIconSize(QSize(20, 20))
            rb.setCursor(Qt.PointingHandCursor)
            rb.setToolTip(self._region_scope_tips[r_key])
            rb.clicked.connect(lambda _=False, rk=r_key: self.set_active_region_scope(rk))
            reg_pills.addWidget(rb, 1)
            self._region_scope_buttons[r_key] = rb
        auto_row.addLayout(reg_pills, 1)
        reg_vbox.addLayout(auto_row)

        # Row 3: Edit Mask: Photoshop-style Add (+) / Subtract (-) Smart Brush toggles
        tool_row = QHBoxLayout()
        tool_row.setContentsMargins(0, 0, 0, 0)
        tool_row.setSpacing(6)
        lbl_edit = QLabel("Edit Mask:", self.region_scope_widget)
        lbl_edit.setObjectName("SliderHeaderLabel")
        lbl_edit.setFixedWidth(68)
        tool_row.addWidget(lbl_edit)

        edit_bar = QHBoxLayout()
        edit_bar.setContentsMargins(0, 0, 0, 0)
        edit_bar.setSpacing(4)

        self.btn_region_paint = QPushButton("", self.region_scope_widget)
        self.btn_region_paint.setObjectName("FineTuneIconTileBtn")
        self.btn_region_paint.setCheckable(True)
        self.btn_region_paint.setFixedHeight(30)
        self.btn_region_paint.setIcon(self._make_mask_tool_icon("add", 18))
        self.btn_region_paint.setIconSize(QSize(18, 18))
        self.btn_region_paint.setCursor(Qt.PointingHandCursor)
        self.btn_region_paint.setToolTip("Add to Mask (+)\nSmart brush: drag on the photo to add areas to the mask (automatically snaps to edges)")
        self.btn_region_paint.clicked.connect(lambda checked: self._on_region_brush_toggled("paint", checked))
        edit_bar.addWidget(self.btn_region_paint, 1)

        self.btn_region_erase = QPushButton("", self.region_scope_widget)
        self.btn_region_erase.setObjectName("FineTuneIconTileBtn")
        self.btn_region_erase.setCheckable(True)
        self.btn_region_erase.setFixedHeight(30)
        self.btn_region_erase.setIcon(self._make_mask_tool_icon("subtract", 18))
        self.btn_region_erase.setIconSize(QSize(18, 18))
        self.btn_region_erase.setCursor(Qt.PointingHandCursor)
        self.btn_region_erase.setToolTip("Subtract from Mask (−)\nSmart brush: drag on the photo to remove areas from the mask (automatically snaps to edges)")
        self.btn_region_erase.clicked.connect(lambda checked: self._on_region_brush_toggled("erase", checked))
        edit_bar.addWidget(self.btn_region_erase, 1)

        self.btn_mask_invert = QPushButton("", self.region_scope_widget)
        self.btn_mask_invert.setObjectName("FineTuneIconTileBtn")
        self.btn_mask_invert.setFixedHeight(30)
        self.btn_mask_invert.setIcon(self._make_mask_tool_icon("invert", 18))
        self.btn_mask_invert.setIconSize(QSize(18, 18))
        self.btn_mask_invert.setCursor(Qt.PointingHandCursor)
        self.btn_mask_invert.setToolTip("Invert Mask\nFlip selected and unselected areas of the mask")
        self.btn_mask_invert.clicked.connect(self._on_mask_invert_clicked)
        edit_bar.addWidget(self.btn_mask_invert, 1)

        tool_row.addLayout(edit_bar, 1)
        reg_vbox.addLayout(tool_row)

        # Row 4: Brush size + Clear mask (visible when a sub-region or brush is active)
        self.region_brush_row_widget = QWidget(self.region_scope_widget)
        brush_col = QVBoxLayout(self.region_brush_row_widget)
        brush_col.setContentsMargins(0, 0, 0, 0)
        brush_col.setSpacing(5)

        size_row = QHBoxLayout()
        size_row.setContentsMargins(0, 0, 0, 0)
        size_row.setSpacing(6)
        lbl_bsz = QLabel("Brush size:", self.region_brush_row_widget)
        lbl_bsz.setObjectName("SliderHeaderLabel")
        lbl_bsz.setFixedWidth(68)
        size_row.addWidget(lbl_bsz)

        self.slider_region_brush_size = QSlider(Qt.Horizontal, self.region_brush_row_widget)
        self.slider_region_brush_size.setRange(10, 120)
        self.slider_region_brush_size.setValue(40)
        self.slider_region_brush_size.setFixedHeight(16)
        self.slider_region_brush_size.setCursor(Qt.SizeHorCursor)
        self.slider_region_brush_size.setToolTip("Adjust the brush radius (10–120 px)")
        self.slider_region_brush_size.valueChanged.connect(self._on_region_brush_size_changed)
        size_row.addWidget(self.slider_region_brush_size, 1)

        self.lbl_brush_size_val = QLabel("40px", self.region_brush_row_widget)
        self.lbl_brush_size_val.setObjectName("val_lbl")
        self.lbl_brush_size_val.setFixedWidth(30)
        self.lbl_brush_size_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        size_row.addWidget(self.lbl_brush_size_val)
        brush_col.addLayout(size_row)

        # Row 5: Clear mask + Reset edits
        clr_row = QHBoxLayout()
        clr_row.setContentsMargins(0, 0, 0, 0)
        clr_row.setSpacing(6)
        self.btn_region_brush_clear = QPushButton("Clear mask", self.region_brush_row_widget)
        self.btn_region_brush_clear.setObjectName("FineTuneSubBtn")
        self.btn_region_brush_clear.setFixedHeight(22)
        self.btn_region_brush_clear.setCursor(Qt.PointingHandCursor)
        self.btn_region_brush_clear.setToolTip("Clear all manual brush strokes on this mask")
        self.btn_region_brush_clear.clicked.connect(self._on_region_brush_clear)
        clr_row.addWidget(self.btn_region_brush_clear)

        self.btn_region_reset = QPushButton("Reset edits", self.region_scope_widget)
        self.btn_region_reset.setObjectName("FineTuneSubBtn")
        self.btn_region_reset.setFixedHeight(22)
        self.btn_region_reset.setVisible(False)
        self.btn_region_reset.setCursor(Qt.PointingHandCursor)
        self.btn_region_reset.setToolTip("Reset all exposure & color slider edits for this region")
        self.btn_region_reset.clicked.connect(self._reset_active_region)
        clr_row.addWidget(self.btn_region_reset)
        clr_row.addStretch(1)
        brush_col.addLayout(clr_row)

        self.region_brush_row_widget.setVisible(False)
        reg_vbox.addWidget(self.region_brush_row_widget)

        self._grp_mask.vbox.addWidget(self.region_scope_widget)
        vbox.addWidget(self._grp_mask)

        # Group 0b: Presets (Look Profiles + .cube 3D LUT + Amount)
        self._grp_presets = _FineTuneGroup("Presets", content)
        self.look_profile_widget = QWidget(self._grp_presets)
        look_vbox = QVBoxLayout(self.look_profile_widget)
        look_vbox.setContentsMargins(0, 0, 0, 0)
        look_vbox.setSpacing(4)
        look_top = QHBoxLayout()
        look_top.setContentsMargins(0, 0, 0, 0)
        look_top.setSpacing(5)
        lbl_look = QLabel("Look:", self.look_profile_widget)
        lbl_look.setObjectName("SliderHeaderLabel")
        look_top.addWidget(lbl_look)
        self.cmb_look_profile = QComboBox(self.look_profile_widget)
        self.cmb_look_profile.setFixedHeight(22)
        for lbl_p, key_p in (
            ("As Shot", "none"),
            ("Natural", "natural"),
            ("Warm Print", "warm_print"),
            ("Vivid", "vivid"),
            ("Portrait", "portrait"),
            ("Monochrome", "monochrome"),
        ):
            self.cmb_look_profile.addItem(lbl_p, key_p)
        self.cmb_look_profile.setToolTip("Apply a non-destructive color look profile or .cube 3D LUT")
        self.cmb_look_profile.currentIndexChanged.connect(self._on_look_profile_combo_changed)
        look_top.addWidget(self.cmb_look_profile, 1)

        self.btn_import_cube_lut = QPushButton("LUT...", self.look_profile_widget)
        self.btn_import_cube_lut.setObjectName("FineTuneSubBtn")
        self.btn_import_cube_lut.setFixedHeight(22)
        self.btn_import_cube_lut.setCursor(Qt.PointingHandCursor)
        self.btn_import_cube_lut.setToolTip("Import a .cube 3D LUT file")
        self.btn_import_cube_lut.clicked.connect(self._on_import_cube_lut_clicked)
        look_top.addWidget(self.btn_import_cube_lut)
        look_vbox.addLayout(look_top)

        self.ft_slider_look_amount = self._make_aero_slider(0, 200, 100)
        self.ft_slider_look_amount.setValue(100)
        self.ft_slider_look_amount.setEnabled(False)
        look_vbox.addWidget(self._make_slider_header("Profile amount", self.ft_slider_look_amount))
        look_vbox.addWidget(self.ft_slider_look_amount)
        self.ft_slider_look_amount.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
        self.ft_slider_look_amount.valueChanged.connect(self._emit_fine_tune_look)
        self.ft_slider_look_amount.sliderReleased.connect(
            lambda: self._on_fine_tune_slider_released("Look Profile Amount")
        )
        self._grp_presets.vbox.addWidget(self.look_profile_widget)
        vbox.addWidget(self._grp_presets)

        # Group 1: Adjust exposure
        self._grp_exp = _FineTuneGroup("Adjust exposure", content)
        grp_exp = self._grp_exp
        self.ft_slider_smart_enhance = self._make_aero_slider(0, 100, 0)
        self.ft_slider_smart_enhance.set_track_gradient([(0.0, "#CBD5E1"), (0.5, "#60A5FA"), (1.0, "#2563EB")])
        grp_exp.vbox.addWidget(self._make_slider_header("Smart enhance", self.ft_slider_smart_enhance))
        grp_exp.vbox.addWidget(self.ft_slider_smart_enhance)
        self.ft_slider_smart_enhance.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
        self.ft_slider_smart_enhance.valueChanged.connect(self._emit_fine_tune_smart_enhance)
        self.ft_slider_smart_enhance.sliderReleased.connect(
            lambda: self._on_fine_tune_slider_released("Smart Enhance")
        )

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

        # Advanced Exposure Controls (Whites & Blacks)
        self.adv_exposure_ext_widget = QWidget(grp_exp)
        adv_exp_layout = QVBoxLayout(self.adv_exposure_ext_widget)
        adv_exp_layout.setContentsMargins(0, 0, 0, 0)
        adv_exp_layout.setSpacing(5)

        self.ft_slider_whites = self._make_aero_slider(-100, 100, 0)
        adv_exp_layout.addWidget(self._make_slider_header("Whites", self.ft_slider_whites))
        adv_exp_layout.addWidget(self.ft_slider_whites)

        self.ft_slider_blacks = self._make_aero_slider(-100, 100, 0)
        adv_exp_layout.addWidget(self._make_slider_header("Blacks", self.ft_slider_blacks))
        adv_exp_layout.addWidget(self.ft_slider_blacks)

        for s_ext in (self.ft_slider_whites, self.ft_slider_blacks):
            s_ext.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
            s_ext.valueChanged.connect(self._emit_fine_tune_exposure_ext)
            s_ext.sliderReleased.connect(
                lambda: self._on_fine_tune_slider_released("Adjust Exposure")
            )

        self.ft_slider_structure = self._make_aero_slider(-100, 100, 0)
        adv_exp_layout.addWidget(self._make_slider_header("Structure", self.ft_slider_structure))
        adv_exp_layout.addWidget(self.ft_slider_structure)
        self.ft_slider_structure.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
        self.ft_slider_structure.valueChanged.connect(self._emit_fine_tune_structure)
        self.ft_slider_structure.sliderReleased.connect(
            lambda: self._on_fine_tune_slider_released("Structure")
        )
        grp_exp.vbox.addWidget(self.adv_exposure_ext_widget)

        self.ft_slider_clarity = self._make_aero_slider(-100, 100, 0)
        grp_exp.vbox.addWidget(self._make_slider_header("Clarity", self.ft_slider_clarity))
        grp_exp.vbox.addWidget(self.ft_slider_clarity)

        self.histogram_container = QWidget(grp_exp)
        hist_c_vbox = QVBoxLayout(self.histogram_container)
        hist_c_vbox.setContentsMargins(0, 0, 0, 0)
        hist_c_vbox.setSpacing(2)

        hist_header = self._make_slider_header("Histogram")
        self.lbl_hist_levels = hist_header.findChild(QLabel, "val_lbl")
        if self.lbl_hist_levels:
            self.lbl_hist_levels.setText("0 - 255")
        self.btn_pin_histogram = QPushButton("📌", hist_header)
        self.btn_pin_histogram.setObjectName("FineTuneSubBtn")
        self.btn_pin_histogram.setCheckable(True)
        self.btn_pin_histogram.setFixedSize(22, 18)
        self.btn_pin_histogram.setCursor(Qt.PointingHandCursor)
        self.btn_pin_histogram.setToolTip("Pin Histogram to top of Fine Tune pane")
        self.btn_pin_histogram.toggled.connect(lambda c: self.set_histogram_pinned(c))
        if isinstance(hist_header.layout(), QHBoxLayout):
            hist_header.layout().addWidget(self.btn_pin_histogram)
        hist_c_vbox.addWidget(hist_header)

        self.histogram_widget = HistogramWidget(self.histogram_container)
        self.histogram_widget.points_changed.connect(self._on_histogram_points_changed)
        self.histogram_widget.slider_pressed.connect(self.fine_tune_slider_pressed.emit)
        self.histogram_widget.slider_released.connect(self.fine_tune_slider_released.emit)
        hist_c_vbox.addWidget(self.histogram_widget)
        self._exp_hist_insert_idx = grp_exp.vbox.count()
        grp_exp.vbox.addWidget(self.histogram_container)

        # Advanced Drawer: Tone Curve (Coloured circle channels + taller plot + Point Curve presets & Reset below)
        self.drawer_tone_curve = _CollapsibleDrawer("Tone Curve", grp_exp, expanded=False)
        curve_ch_row = QHBoxLayout()
        curve_ch_row.setSpacing(6)
        curve_ch_row.setContentsMargins(2, 2, 2, 2)
        self._curve_ch_buttons: dict[str, QPushButton] = {}
        ch_tips = {
            "rgb": "RGB Master Tone Curve",
            "r": "Red Channel Tone Curve",
            "g": "Green Channel Tone Curve",
            "b": "Blue Channel Tone Curve",
        }
        for ch_key in ("rgb", "r", "g", "b"):
            b_ch = QPushButton("", self.drawer_tone_curve.body)
            b_ch.setObjectName("CurveCircleBtn")
            b_ch.setCheckable(True)
            b_ch.setChecked(ch_key == "rgb")
            b_ch.setFixedSize(26, 24)
            b_ch.setIcon(self._make_curve_channel_icon(ch_key, is_checked=(ch_key == "rgb")))
            b_ch.setIconSize(QSize(18, 20))
            b_ch.setCursor(Qt.PointingHandCursor)
            b_ch.setToolTip(ch_tips[ch_key])
            b_ch.clicked.connect(lambda _=False, k=ch_key: self._select_curve_channel(k))
            curve_ch_row.addWidget(b_ch)
            self._curve_ch_buttons[ch_key] = b_ch
        curve_ch_row.addStretch(1)
        self.drawer_tone_curve.body_layout.addLayout(curve_ch_row)

        self.tone_curve_widget = _ToneCurveCanvas(self.drawer_tone_curve.body)
        self.histogram_widget.histogram_updated.connect(self.tone_curve_widget.set_histogram_counts)
        self.tone_curve_widget.curve_pressed.connect(self.fine_tune_slider_pressed.emit)
        self.tone_curve_widget.curve_changed.connect(self._emit_fine_tune_curve)
        self.tone_curve_widget.curve_released.connect(
            lambda: self._on_fine_tune_slider_released("Tone Curve")
        )
        self.drawer_tone_curve.body_layout.addWidget(self.tone_curve_widget)

        curve_bot_row = QHBoxLayout()
        curve_bot_row.setSpacing(6)
        curve_bot_row.setContentsMargins(2, 2, 2, 0)
        lbl_pt_curve = QLabel("Point Curve", self.drawer_tone_curve.body)
        lbl_pt_curve.setObjectName("SliderHeaderLabel")
        curve_bot_row.addWidget(lbl_pt_curve)

        self.cmb_curve_preset = QComboBox(self.drawer_tone_curve.body)
        self.cmb_curve_preset.setFixedHeight(22)
        self.cmb_curve_preset.addItem("Linear", "linear")
        self.cmb_curve_preset.addItem("Medium Contrast (S-Curve)", "scurve_med")
        self.cmb_curve_preset.addItem("Strong Contrast (S-Curve)", "scurve_strong")
        self.cmb_curve_preset.setToolTip("Choose a Tone Curve preset")
        self.cmb_curve_preset.currentIndexChanged.connect(self._on_curve_preset_changed)
        curve_bot_row.addWidget(self.cmb_curve_preset, 1)

        self.btn_curve_scurve = QPushButton("S-Curve", self.drawer_tone_curve.body)
        self.btn_curve_scurve.setVisible(False)
        self.btn_curve_scurve.clicked.connect(lambda: self.tone_curve_widget.apply_s_curve_preset())

        self.btn_curve_reset = QPushButton("Reset", self.drawer_tone_curve.body)
        self.btn_curve_reset.setObjectName("FineTuneSubBtn")
        self.btn_curve_reset.setFixedHeight(22)
        self.btn_curve_reset.setMinimumWidth(48)
        self.btn_curve_reset.setCursor(Qt.PointingHandCursor)
        self.btn_curve_reset.clicked.connect(self._on_curve_reset_clicked)
        curve_bot_row.addWidget(self.btn_curve_reset)
        self.drawer_tone_curve.body_layout.addLayout(curve_bot_row)
        grp_exp.vbox.addWidget(self.drawer_tone_curve)

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

        # Group 2: Adjust colour
        self._grp_color = _FineTuneGroup("Adjust colour", content)
        grp_color = self._grp_color
        self._wb_presets = (
            ("as_shot", "As Shot", 0, 0),
            ("daylight", "Daylight (5500K)", 8, 2),
            ("cloudy", "Cloudy (6500K)", 22, 5),
            ("shade", "Shade (7500K)", 35, 8),
            ("tungsten", "Tungsten (2850K)", -42, -4),
            ("fluorescent", "Fluorescent (4000K)", -18, 18),
            ("flash", "Flash (5500K)", 12, 3),
            ("custom", "Custom", None, None),
        )
        wb_row = QHBoxLayout()
        wb_row.setContentsMargins(0, 0, 0, 2)
        wb_row.setSpacing(4)
        lbl_wb = QLabel("White Balance:", grp_color)
        lbl_wb.setObjectName("SliderHeaderLabel")
        wb_row.addWidget(lbl_wb)
        self.cmb_wb_preset = QComboBox(grp_color)
        self.cmb_wb_preset.setFixedHeight(22)
        for wb_key, wb_lbl, _t, _ti in self._wb_presets:
            self.cmb_wb_preset.addItem(wb_lbl, wb_key)
        self.cmb_wb_preset.setToolTip("White Balance lighting preset (As Shot, Daylight, Cloudy, Shade, Tungsten, Fluorescent, Flash)")
        self.cmb_wb_preset.currentIndexChanged.connect(self._on_wb_preset_changed)
        wb_row.addWidget(self.cmb_wb_preset, 1)

        self.btn_wb_eyedropper = QPushButton(grp_color)
        self.btn_wb_eyedropper.setObjectName("FineTuneActionBtn")
        self.btn_wb_eyedropper.setCheckable(True)
        self.btn_wb_eyedropper.setFixedSize(26, 22)
        self.btn_wb_eyedropper.setIcon(make_wlpg_icon("eyedropper", 15))
        self.btn_wb_eyedropper.setIconSize(QSize(15, 15))
        self.btn_wb_eyedropper.setCursor(Qt.PointingHandCursor)
        self.btn_wb_eyedropper.setToolTip(
            "White Balance Eyedropper\nClick any neutral gray or white area on the photo to auto-balance Temperature & Tint"
        )
        self.btn_wb_eyedropper.toggled.connect(self.wb_eyedropper_toggled.emit)
        wb_row.addWidget(self.btn_wb_eyedropper)
        self.wb_picker_row_widget = QWidget(grp_color)
        self.wb_picker_row_widget.setLayout(wb_row)
        grp_color.vbox.addWidget(self.wb_picker_row_widget)

        self.ft_slider_temp = self._make_aero_slider(-100, 100, 0)
        self.ft_slider_temp.set_track_gradient([(0.0, "#4B8BF5"), (0.5, "#E2E8F0"), (1.0, "#F5A623")])
        grp_color.vbox.addWidget(self._make_slider_header("Colour temperature", self.ft_slider_temp))
        grp_color.vbox.addWidget(self.ft_slider_temp)

        self.ft_slider_tint = self._make_aero_slider(-100, 100, 0)
        self.ft_slider_tint.set_track_gradient([(0.0, "#34C759"), (0.5, "#E2E8F0"), (1.0, "#D946EF")])
        grp_color.vbox.addWidget(self._make_slider_header("Tint", self.ft_slider_tint))
        grp_color.vbox.addWidget(self.ft_slider_tint)

        self.ft_slider_sat = self._make_aero_slider(-100, 100, 0)
        self.ft_slider_sat.set_track_gradient([(0.0, "#8E9AAF"), (0.45, "#CBD5E1"), (0.75, "#F97316"), (1.0, "#EC4899")])
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

        # Advanced Drawer: 8-Band Color Mixer (HSL)
        from src.imaging.opencv_engine import HSL_COLOR_BANDS

        self.drawer_color_mixer = _CollapsibleDrawer("Colour Mixer (HSL)", grp_color, expanded=False)
        hsl_tab_row = QHBoxLayout()
        hsl_tab_row.setSpacing(3)
        self._hsl_mode: str = "sat"
        self._hsl_hue_vals: dict[str, int] = {k: 0 for k, _, _, _ in HSL_COLOR_BANDS}
        self._hsl_sat_vals: dict[str, int] = {k: 0 for k, _, _, _ in HSL_COLOR_BANDS}
        self._hsl_lum_vals: dict[str, int] = {k: 0 for k, _, _, _ in HSL_COLOR_BANDS}
        self._hsl_tab_buttons: dict[str, QPushButton] = {}
        for m_key, m_lbl in (("hue", "Hue"), ("sat", "Saturation"), ("lum", "Luminance")):
            bt = QPushButton(m_lbl, self.drawer_color_mixer.body)
            bt.setObjectName("FineTuneSubBtn")
            bt.setCheckable(True)
            bt.setChecked(m_key == "sat")
            bt.setFixedHeight(20)
            bt.setMinimumWidth(38 if m_key == "hue" else 64)
            bt.setCursor(Qt.PointingHandCursor)
            bt.clicked.connect(lambda _=False, k=m_key: self._select_hsl_mode(k))
            hsl_tab_row.addWidget(bt)
            self._hsl_tab_buttons[m_key] = bt
        hsl_tab_row.addStretch(1)
        self.btn_hsl_reset = QPushButton("Reset", self.drawer_color_mixer.body)
        self.btn_hsl_reset.setObjectName("FineTuneSubBtn")
        self.btn_hsl_reset.setFixedHeight(20)
        self.btn_hsl_reset.setMinimumWidth(44)
        self.btn_hsl_reset.setCursor(Qt.PointingHandCursor)
        self.btn_hsl_reset.clicked.connect(self._reset_hsl_sliders)
        hsl_tab_row.addWidget(self.btn_hsl_reset)
        self.drawer_color_mixer.body_layout.addLayout(hsl_tab_row)

        self._hsl_sliders: dict[str, _AeroFineTuneSlider] = {}
        for band_key, band_name, _, hex_col in HSL_COLOR_BANDS:
            sl = self._make_aero_slider(-100, 100, 0)
            hdr = self._make_slider_header(f"● {band_name}", sl)
            lbl_hdr = hdr.findChild(QLabel, "SliderHeaderLabel")
            if lbl_hdr is not None:
                lbl_hdr.setStyleSheet(f"color: {hex_col}; font-weight: 600;")
            sl.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
            sl.valueChanged.connect(lambda val, bk=band_key: self._on_hsl_slider_changed(bk, val))
            sl.sliderReleased.connect(
                lambda: self._on_fine_tune_slider_released("Color Mixer (HSL)")
            )
            self.drawer_color_mixer.body_layout.addWidget(hdr)
            self.drawer_color_mixer.body_layout.addWidget(sl)
            self._hsl_sliders[band_key] = sl
        self._update_hsl_slider_gradients()
        grp_color.vbox.addWidget(self.drawer_color_mixer)

        vbox.addWidget(grp_color)

        # Group 3: Straighten photo
        self._grp_str = _FineTuneGroup("Straighten photo", content)
        grp_str = self._grp_str
        str_row = QHBoxLayout()
        str_row.setSpacing(6)
        self.btn_auto_straighten = QPushButton("Auto straighten", grp_str)
        self.btn_auto_straighten.setObjectName("FineTuneActionBtn")
        self.btn_auto_straighten.setFixedHeight(22)
        self.btn_auto_straighten.setCursor(Qt.PointingHandCursor)
        self.btn_auto_straighten.clicked.connect(self.auto_straighten_requested.emit)
        str_row.addWidget(self.btn_auto_straighten)

        self.btn_draw_straighten = QPushButton("", grp_str)
        self.btn_draw_straighten.setObjectName("FineTuneActionBtn")
        self.btn_draw_straighten.setIcon(make_wlpg_icon("straighten", 14))
        self.btn_draw_straighten.setIconSize(QSize(14, 14))
        self.btn_draw_straighten.setFixedSize(26, 22)
        self.btn_draw_straighten.setCursor(Qt.PointingHandCursor)
        self.btn_draw_straighten.setToolTip("Straighten the image by drawing a line on it")
        self.btn_draw_straighten.clicked.connect(self.draw_line_straighten_requested.emit)
        str_row.addWidget(self.btn_draw_straighten)

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

        # Group 4: Adjust detail (Reduce noise first, Sharpen second; each with its Advanced controls directly beneath)
        self._grp_det = _FineTuneGroup("Adjust detail", content)
        grp_det = self._grp_det

        self.ft_slider_denoise = self._make_aero_slider(0, 100, 0)
        grp_det.vbox.addWidget(self._make_slider_header("Reduce noise", self.ft_slider_denoise))
        grp_det.vbox.addWidget(self.ft_slider_denoise)
        self.ft_slider_denoise.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
        self.ft_slider_denoise.valueChanged.connect(self._emit_fine_tune_detail)
        self.ft_slider_denoise.sliderReleased.connect(
            lambda: self._on_fine_tune_slider_released("Adjust Detail")
        )

        # Advanced Noise Reduction Controls (directly below Reduce noise)
        self.adv_denoise_ext_widget = QWidget(grp_det)
        adv_dn_layout = QVBoxLayout(self.adv_denoise_ext_widget)
        adv_dn_layout.setContentsMargins(0, 2, 0, 0)
        adv_dn_layout.setSpacing(5)

        self.ft_slider_denoise_chroma = self._make_aero_slider(0, 100, 0)
        adv_dn_layout.addWidget(self._make_slider_header("Colour noise", self.ft_slider_denoise_chroma))
        adv_dn_layout.addWidget(self.ft_slider_denoise_chroma)
        self.ft_slider_denoise_chroma.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
        self.ft_slider_denoise_chroma.valueChanged.connect(self._emit_fine_tune_detail_ext)
        self.ft_slider_denoise_chroma.sliderReleased.connect(
            lambda: self._on_fine_tune_slider_released("Color Noise Reduction")
        )

        self.ft_slider_denoise_detail = self._make_aero_slider(0, 100, 50)
        self.ft_slider_denoise_detail.setValue(50)
        self.ft_slider_denoise_detail.setEnabled(False)
        adv_dn_layout.addWidget(self._make_slider_header("Detail preservation", self.ft_slider_denoise_detail))
        adv_dn_layout.addWidget(self.ft_slider_denoise_detail)
        self.ft_slider_denoise_detail.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
        self.ft_slider_denoise_detail.valueChanged.connect(self._emit_fine_tune_detail_ext)
        self.ft_slider_denoise_detail.sliderReleased.connect(
            lambda: self._on_fine_tune_slider_released("Denoise Detail")
        )
        grp_det.vbox.addWidget(self.adv_denoise_ext_widget)

        self.ft_slider_sharpen = self._make_aero_slider(0, 100, 0)
        grp_det.vbox.addWidget(self._make_slider_header("Sharpen", self.ft_slider_sharpen))
        grp_det.vbox.addWidget(self.ft_slider_sharpen)
        self.ft_slider_sharpen.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
        self.ft_slider_sharpen.valueChanged.connect(self._emit_fine_tune_detail)
        self.ft_slider_sharpen.sliderReleased.connect(
            lambda: self._on_fine_tune_slider_released("Adjust Detail")
        )

        # Advanced Sharpen Controls (directly below Sharpen)
        self.adv_detail_ext_widget = QWidget(grp_det)
        adv_det_layout = QVBoxLayout(self.adv_detail_ext_widget)
        adv_det_layout.setContentsMargins(0, 2, 0, 0)
        adv_det_layout.setSpacing(5)

        self.ft_slider_sharpen_radius = self._make_aero_slider(0, 100, 25)
        adv_det_layout.addWidget(self._make_slider_header("Sharpen radius", self.ft_slider_sharpen_radius))
        adv_det_layout.addWidget(self.ft_slider_sharpen_radius)
        self.ft_slider_sharpen_radius.setEnabled(False)
        self.ft_slider_sharpen_radius.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
        self.ft_slider_sharpen_radius.valueChanged.connect(self._emit_fine_tune_detail_ext)
        self.ft_slider_sharpen_radius.sliderReleased.connect(
            lambda: self._on_fine_tune_slider_released("Sharpen Radius")
        )

        self.ft_slider_sharpen_masking = self._make_aero_slider(0, 100, 0)
        adv_det_layout.addWidget(self._make_slider_header("Sharpen masking", self.ft_slider_sharpen_masking))
        adv_det_layout.addWidget(self.ft_slider_sharpen_masking)
        self.ft_slider_sharpen_masking.setEnabled(False)
        self.ft_slider_sharpen_masking.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
        self.ft_slider_sharpen_masking.valueChanged.connect(self._emit_fine_tune_detail_ext)
        self.ft_slider_sharpen_masking.sliderReleased.connect(
            lambda: self._on_fine_tune_slider_released("Sharpen Masking")
        )
        grp_det.vbox.addWidget(self.adv_detail_ext_widget)

        vbox.addWidget(grp_det)

        # Group 5: Vignette & Grain (Effects)
        self._grp_optics = _FineTuneGroup("Vignette & optics", content)
        grp_optics = self._grp_optics
        self.ft_slider_vignette = self._make_aero_slider(-100, 100, 0)
        grp_optics.vbox.addWidget(self._make_slider_header("Vignette", self.ft_slider_vignette))
        grp_optics.vbox.addWidget(self.ft_slider_vignette)

        self.ft_slider_vignette.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
        self.ft_slider_vignette.valueChanged.connect(self._emit_fine_tune_optics)
        self.ft_slider_vignette.sliderReleased.connect(
            lambda: self._on_fine_tune_slider_released("Adjust Vignette")
        )

        # Advanced Effects Controls: Vignette Feather + Film Grain
        self.adv_optics_ext_widget = QWidget(grp_optics)
        adv_opt_layout = QVBoxLayout(self.adv_optics_ext_widget)
        adv_opt_layout.setContentsMargins(0, 2, 0, 0)
        adv_opt_layout.setSpacing(5)

        self.ft_slider_vignette_feather = self._make_aero_slider(0, 100, 50)
        self.ft_slider_vignette_feather.setValue(50)
        self.ft_slider_vignette_feather.setEnabled(False)
        adv_opt_layout.addWidget(self._make_slider_header("Vignette feather", self.ft_slider_vignette_feather))
        adv_opt_layout.addWidget(self.ft_slider_vignette_feather)

        self.ft_slider_grain_amount = self._make_aero_slider(0, 100, 0)
        adv_opt_layout.addWidget(self._make_slider_header("Film grain", self.ft_slider_grain_amount))
        adv_opt_layout.addWidget(self.ft_slider_grain_amount)

        self.ft_slider_grain_size = self._make_aero_slider(0, 100, 25)
        self.ft_slider_grain_size.setValue(25)
        self.ft_slider_grain_size.setEnabled(False)
        adv_opt_layout.addWidget(self._make_slider_header("Grain size", self.ft_slider_grain_size))
        adv_opt_layout.addWidget(self.ft_slider_grain_size)

        self.ft_slider_vignette_feather.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
        self.ft_slider_vignette_feather.valueChanged.connect(self._emit_fine_tune_optics)
        self.ft_slider_vignette_feather.sliderReleased.connect(
            lambda: self._on_fine_tune_slider_released("Adjust Vignette Feather")
        )

        for s in (self.ft_slider_grain_amount, self.ft_slider_grain_size):
            s.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
            s.valueChanged.connect(self._emit_fine_tune_detail_ext)
            s.sliderReleased.connect(
                lambda: self._on_fine_tune_slider_released("Film Grain")
            )

        self.ft_slider_orton_glow = self._make_aero_slider(0, 100, 0)
        adv_opt_layout.addWidget(self._make_slider_header("Orton glow", self.ft_slider_orton_glow))
        adv_opt_layout.addWidget(self.ft_slider_orton_glow)

        self.ft_slider_golden_hour = self._make_aero_slider(0, 100, 0)
        self.ft_slider_golden_hour.set_track_gradient([(0.0, "#CBD5E1"), (0.5, "#FBBF24"), (1.0, "#EA580C")])
        adv_opt_layout.addWidget(self._make_slider_header("Golden hour", self.ft_slider_golden_hour))
        adv_opt_layout.addWidget(self.ft_slider_golden_hour)

        for s_glow in (self.ft_slider_orton_glow, self.ft_slider_golden_hour):
            s_glow.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
            s_glow.valueChanged.connect(self._emit_fine_tune_optics)
            s_glow.sliderReleased.connect(
                lambda: self._on_fine_tune_slider_released("Atmospheric Glow")
            )
        grp_optics.vbox.addWidget(self.adv_optics_ext_widget)

        vbox.addWidget(grp_optics)

        # Bottom Actions: Hold to Compare (Original) and Reset to default
        action_bar = QHBoxLayout()
        action_bar.setSpacing(6)
        action_bar.setContentsMargins(0, 6, 0, 4)

        self.btn_hold_compare = QPushButton("Hold to Compare (Original)", content)
        self.btn_hold_compare.setObjectName("FineTuneActionBtn")
        self.btn_hold_compare.setFixedHeight(26)
        self.btn_hold_compare.setCursor(Qt.PointingHandCursor)
        self.btn_hold_compare.setToolTip("Click and hold to preview original unedited photo (\\)")
        self.btn_hold_compare.pressed.connect(self.compare_started.emit)
        self.btn_hold_compare.released.connect(self.compare_ended.emit)
        action_bar.addWidget(self.btn_hold_compare, 1)

        self.btn_reset_default = QPushButton("Reset to default", content)
        self.btn_reset_default.setObjectName("FineTuneResetBtn")
        self.btn_reset_default.setFixedHeight(26)
        self.btn_reset_default.setCursor(Qt.PointingHandCursor)
        self.btn_reset_default.setToolTip("Reset all adjustments and levels back to default")
        self.btn_reset_default.clicked.connect(self.reset_all_adjustments)
        action_bar.addWidget(self.btn_reset_default, 1)

        vbox.addLayout(action_bar)
        vbox.addStretch(1)

        scroll.setWidget(content)

        # Initialize Basic | Advanced visibility from persisted QSettings (default Basic)
        try:
            from PySide6.QtCore import QSettings

            qs = QSettings("PhotoGallery64", "PhotoGallery64")
            saved_adv = bool(qs.value("ui/fine_tune_advanced", False, type=bool))
            saved_pin = bool(qs.value("ui/fine_tune_pin_histogram", True, type=bool))
        except Exception:
            saved_adv = False
            saved_pin = True
        self._histogram_pinned = saved_pin
        if hasattr(self, "btn_pin_histogram"):
            self.btn_pin_histogram.blockSignals(True)
            self.btn_pin_histogram.setChecked(saved_pin)
            self.btn_pin_histogram.blockSignals(False)
        self.set_fine_tune_advanced_mode(saved_adv, persist=False)

        self._apply_fine_tune_tooltips()
        return page_wrap

    def _apply_fine_tune_tooltips(self) -> None:
        tips = {
            "btn_ft_basic": "Basic Mode: clean, classic exposure, color, detail, and luminance histogram controls",
            "btn_ft_advanced": "Advanced Mode: unlocks RGB histogram, Tone Curve, WB Eyedropper, HSL Color Mixer, Film Grain, and Regional Masking",
            "cmb_look_profile": "Choose a non-destructive color Look Profile or custom .cube 3D LUT",
            "btn_import_cube_lut": "Import a custom .cube 3D LUT file",
            "ft_slider_look_amount": "Blend strength (0–200%) of the active Look Profile or .cube LUT",
            "cmb_wb_preset": "White Balance lighting preset (As Shot, Daylight, Cloudy, Shade, Tungsten, Fluorescent, Flash)",
            "histogram_widget": "Live tonal histogram — drag the bottom black/white point triangles to clip shadows or highlights",
            "tone_curve_widget": "Click to add control points, drag to shape tone response, right-click a point to remove it",
            "drawer_tone_curve": "Expand to shape contrast and color channels with a Tone Curve or 1-click S-Curve preset",
            "drawer_color_mixer": "Expand to adjust Hue, Saturation, and Luminance across 8 individual color bands",
            "ft_slider_smart_enhance": "One-slider adaptive enhance: balances shadows, midtone contrast, and vibrance automatically",
            "ft_slider_brightness": "Exposure: brighten or darken the photo with filmic highlight roll-off (double-click to reset)",
            "ft_slider_contrast": "Contrast: expand or soften tonal separation between shadows and highlights",
            "ft_slider_shadows": "Shadows: recover detail in dark areas without washing out blacks",
            "ft_slider_highlights": "Highlights: pull back bright skies and specular areas to recover detail",
            "ft_slider_whites": "Whites: set the upper white point brightness",
            "ft_slider_blacks": "Blacks: deepen or lift the darkest shadow tones",
            "ft_slider_clarity": "Clarity: boost midtone local contrast for crisp texture and punch",
            "ft_slider_structure": "Structure: enhance architectural and landscape micro-detail while protecting skin tones and skies",
            "ft_slider_temp": "Color temperature: shift white balance cooler (blue) or warmer (amber)",
            "ft_slider_tint": "Tint: fine-tune green vs. magenta white balance cast",
            "ft_slider_sat": "Saturation: boost color richness while protecting natural skin tones",
            "ft_slider_dehaze": "Dehaze: cut through atmospheric haze, fog, and low-contrast veil",
            "ft_slider_straighten": "Straighten angle (-15° to +15°) with automatic edge-free crop",
            "btn_auto_straighten": "Automatically detect the horizon or dominant lines and straighten the photo",
            "ft_slider_sharpen": "Sharpen: enhance fine edge crispness in luminance",
            "ft_slider_sharpen_radius": "Sharpen radius: adjust edge halo width (small for fine detail, larger for soft lenses)",
            "ft_slider_sharpen_masking": "Sharpen masking: restrict sharpening to real edges so smooth skies stay noise-free",
            "ft_slider_denoise": "Reduce noise: smooth luminance grain while preserving sharp edges",
            "ft_slider_denoise_detail": "Detail preservation: keep fine textures crisp while denoising flat areas",
            "ft_slider_denoise_chroma": "Color noise: remove red/green/blue chroma blotches from high-ISO shots",
            "ft_slider_vignette": "Vignette: darken or lighten photo corners to draw focus to the center",
            "ft_slider_vignette_feather": "Vignette feather: control how softly the vignette fades toward the center",
            "ft_slider_grain_amount": "Film grain: add organic analog film texture",
            "ft_slider_grain_size": "Grain size: adjust film grain particle coarseness",
            "ft_slider_orton_glow": "Orton glow: add dreamy highlight bloom and soft contrast",
            "ft_slider_golden_hour": "Golden hour: bathe highlights and midtones in warm sunset light",
            "btn_hold_compare": "Click and hold to compare with the original unedited photo (\\)",
            "btn_reset_default": "Reset all Fine Tune adjustments back to their defaults",
        }
        for attr, tip in tips.items():
            w = getattr(self, attr, None)
            if w is not None:
                w.setToolTip(tip)

    def is_fine_tune_advanced_mode(self) -> bool:
        return bool(getattr(self, "_fine_tune_advanced_mode", False))

    def set_fine_tune_advanced_mode(self, enabled: bool, persist: bool = True) -> None:
        adv = bool(enabled)
        self._fine_tune_advanced_mode = adv
        if hasattr(self, "btn_ft_basic"):
            self.btn_ft_basic.blockSignals(True)
            self.btn_ft_basic.setChecked(not adv)
            self.btn_ft_basic.blockSignals(False)
        if hasattr(self, "btn_ft_advanced"):
            self.btn_ft_advanced.blockSignals(True)
            self.btn_ft_advanced.setChecked(adv)
            self.btn_ft_advanced.blockSignals(False)
        if hasattr(self, "histogram_widget"):
            self.histogram_widget.set_rgb_mode(adv)
        for attr in (
            "_grp_mask",
            "btn_import_cube_lut",
            "adv_exposure_ext_widget",
            "drawer_tone_curve",
            "wb_picker_row_widget",
            "drawer_color_mixer",
            "adv_denoise_ext_widget",
            "adv_detail_ext_widget",
            "adv_optics_ext_widget",
        ):
            w = getattr(self, attr, None)
            if w is not None:
                w.setVisible(adv)
        if hasattr(self, "_grp_optics"):
            self._grp_optics.set_title("Vignette & grain" if adv else "Vignette & optics")
        for grp in ("_grp_mask", "_grp_presets", "_grp_exp", "_grp_color", "_grp_str", "_grp_det", "_grp_optics"):
            g = getattr(self, grp, None)
            if g is not None:
                g.set_collapsible(adv)
        if hasattr(self, "btn_pin_histogram"):
            self.btn_pin_histogram.setVisible(adv)
        self._apply_histogram_pin_state()
        if adv:
            if hasattr(self, "_grp_str") and abs(self.ft_slider_straighten.value()) == 0:
                self._grp_str.set_expanded(False)
            if hasattr(self, "_grp_det") and (
                self.ft_slider_sharpen.value() == 0
                and self.ft_slider_denoise.value() == 0
                and self.ft_slider_denoise_chroma.value() == 0
            ):
                self._grp_det.set_expanded(False)
        else:
            self.clear_mask_editing_session()
        if not adv and hasattr(self, "btn_wb_eyedropper") and self.btn_wb_eyedropper.isChecked():
            self.btn_wb_eyedropper.setChecked(False)
        if persist:
            try:
                from PySide6.QtCore import QSettings

                QSettings("PhotoGallery64", "PhotoGallery64").setValue("ui/fine_tune_advanced", adv)
            except Exception:
                pass

    def set_histogram_pinned(self, pinned: bool, persist: bool = True) -> None:
        self._histogram_pinned = bool(pinned)
        if hasattr(self, "btn_pin_histogram"):
            self.btn_pin_histogram.blockSignals(True)
            self.btn_pin_histogram.setChecked(self._histogram_pinned)
            self.btn_pin_histogram.blockSignals(False)
        self._apply_histogram_pin_state()
        if persist:
            try:
                from PySide6.QtCore import QSettings

                QSettings("PhotoGallery64", "PhotoGallery64").setValue(
                    "ui/fine_tune_pin_histogram", self._histogram_pinned
                )
            except Exception:
                pass

    def _apply_histogram_pin_state(self) -> None:
        if not hasattr(self, "histogram_container") or not hasattr(self, "ft_pinned_hist_slot"):
            return
        should_pin = bool(getattr(self, "_fine_tune_advanced_mode", False)) and bool(
            getattr(self, "_histogram_pinned", True)
        )
        if should_pin:
            if self.histogram_container.parentWidget() is not self.ft_pinned_hist_slot:
                self._pinned_hist_layout.addWidget(self.histogram_container)
            self.ft_pinned_hist_slot.setVisible(True)
        else:
            self.ft_pinned_hist_slot.setVisible(False)
            if hasattr(self, "_grp_exp") and self.histogram_container.parentWidget() is not self._grp_exp:
                idx = getattr(self, "_exp_hist_insert_idx", -1)
                if 0 <= idx <= self._grp_exp.vbox.count():
                    self._grp_exp.vbox.insertWidget(idx, self.histogram_container)
                else:
                    self._grp_exp.vbox.addWidget(self.histogram_container)

    def _on_mask_invert_clicked(self) -> None:
        reg = getattr(self, "_active_region_scope", "whole")
        if reg == "whole":
            self.set_active_region_scope("custom")
            reg = "custom"
        self._set_mask_preview_visible(True)
        self.region_mask_invert_requested.emit(reg)

    def _capture_active_region_sliders(self) -> dict[str, Any]:
        curves = self.tone_curve_widget.get_curves() if hasattr(self, "tone_curve_widget") else {}
        return {
            "brightness": int(self.ft_slider_brightness.value()),
            "contrast": int(self.ft_slider_contrast.value()),
            "shadows": int(self.ft_slider_shadows.value()),
            "highlights": int(self.ft_slider_highlights.value()),
            "whites": int(self.ft_slider_whites.value()) if hasattr(self, "ft_slider_whites") else 0,
            "blacks": int(self.ft_slider_blacks.value()) if hasattr(self, "ft_slider_blacks") else 0,
            "clarity": int(self.ft_slider_clarity.value()),
            "structure": int(self.ft_slider_structure.value()) if hasattr(self, "ft_slider_structure") else 0,
            "temperature": int(self.ft_slider_temp.value()),
            "tint": int(self.ft_slider_tint.value()),
            "saturation": int(self.ft_slider_sat.value()),
            "dehaze": int(self.ft_slider_dehaze.value()),
            "curve_rgb": [tuple(p) for p in curves.get("rgb", [(0.0, 0.0), (1.0, 1.0)])],
            "curve_r": [tuple(p) for p in curves.get("r", [(0.0, 0.0), (1.0, 1.0)])],
            "curve_g": [tuple(p) for p in curves.get("g", [(0.0, 0.0), (1.0, 1.0)])],
            "curve_b": [tuple(p) for p in curves.get("b", [(0.0, 0.0), (1.0, 1.0)])],
        }

    def _load_region_dict_into_sliders(self, data: dict[str, Any]) -> None:
        self._updating_ui = True
        try:
            self.ft_slider_brightness.setValue(int(round(data.get("brightness", 0))))
            self.ft_slider_contrast.setValue(int(round(data.get("contrast", 0))))
            self.ft_slider_shadows.setValue(int(round(data.get("shadows", 0))))
            self.ft_slider_highlights.setValue(int(round(data.get("highlights", 0))))
            if hasattr(self, "ft_slider_whites"):
                self.ft_slider_whites.setValue(int(round(data.get("whites", 0))))
            if hasattr(self, "ft_slider_blacks"):
                self.ft_slider_blacks.setValue(int(round(data.get("blacks", 0))))
            self.ft_slider_clarity.setValue(int(round(data.get("clarity", 0))))
            if hasattr(self, "ft_slider_structure"):
                self.ft_slider_structure.setValue(int(round(data.get("structure", 0))))
            self.ft_slider_temp.setValue(int(round(data.get("temperature", 0))))
            self.ft_slider_tint.setValue(int(round(data.get("tint", 0))))
            self.ft_slider_sat.setValue(int(round(data.get("saturation", 0))))
            self.ft_slider_dehaze.setValue(int(round(data.get("dehaze", 0))))
            if hasattr(self, "tone_curve_widget"):
                self.tone_curve_widget.set_curves(
                    {
                        "rgb": [tuple(p) for p in data.get("curve_rgb", [(0.0, 0.0), (1.0, 1.0)])],
                        "r": [tuple(p) for p in data.get("curve_r", [(0.0, 0.0), (1.0, 1.0)])],
                        "g": [tuple(p) for p in data.get("curve_g", [(0.0, 0.0), (1.0, 1.0)])],
                        "b": [tuple(p) for p in data.get("curve_b", [(0.0, 0.0), (1.0, 1.0)])],
                    }
                )
            self._sync_wb_preset_combo()
            self._update_drawer_indicators()
        finally:
            self._updating_ui = False

    def set_active_region_scope(self, region: str) -> None:
        reg = region if region in ("whole", "subject", "background", "sky", "custom") else "whole"
        prev = getattr(self, "_active_region_scope", "whole")
        if prev != reg:
            if prev == "whole":
                self._global_region_mirror = self._capture_active_region_sliders()
            else:
                self._region_edits[prev] = self._capture_active_region_sliders()
            self._active_region_scope = reg
            if reg == "whole":
                self._load_region_dict_into_sliders(self._global_region_mirror)
            else:
                self._load_region_dict_into_sliders(self._region_edits.get(reg, {}))

        for rk, btn in getattr(self, "_region_scope_buttons", {}).items():
            btn.blockSignals(True)
            btn.setChecked(rk == reg)
            btn.blockSignals(False)

        is_sub = reg != "whole"
        if is_sub:
            self._last_preview_region = reg
        if hasattr(self, "btn_mask_show") and hasattr(self, "btn_mask_hide"):
            self.btn_mask_show.setEnabled(True)
            self.btn_mask_hide.setEnabled(True)
            if prev == "whole" and is_sub:
                # Clicking a specific region (e.g. Sky/Subject/Bg) toggles Show mask on automatically
                self._set_mask_preview_visible(True)
            elif self.btn_mask_show.isChecked():
                preview_reg = reg if is_sub else getattr(self, "_last_preview_region", "subject")
                self.region_mask_preview_toggled.emit(preview_reg)
        if hasattr(self, "region_brush_row_widget"):
            self.region_brush_row_widget.setVisible(is_sub)
            if not is_sub:
                self._clear_region_brush_buttons()
                self.region_brush_mode_changed.emit("", "", self.slider_region_brush_size.value(), True)
            else:
                mode = self._active_region_brush_mode()
                if reg == "custom" and not mode:
                    self.btn_region_paint.blockSignals(True)
                    self.btn_region_paint.setChecked(True)
                    self.btn_region_paint.blockSignals(False)
                    self._set_mask_preview_visible(True)
                    mode = "paint"
                if mode:
                    self.region_brush_mode_changed.emit(reg, mode, self.slider_region_brush_size.value(), True)
        if hasattr(self, "ft_slider_smart_enhance"):
            self.ft_slider_smart_enhance.setEnabled(not is_sub)
        if hasattr(self, "histogram_widget"):
            self.histogram_widget.set_handles_enabled(not is_sub)
        if hasattr(self, "drawer_color_mixer"):
            self.drawer_color_mixer.setEnabled(not is_sub)
        self._refresh_region_indicators()
        self.region_scope_changed.emit(reg)

    def _active_region_brush_mode(self) -> str:
        if hasattr(self, "btn_region_paint") and self.btn_region_paint.isChecked():
            return "paint"
        if hasattr(self, "btn_region_erase") and self.btn_region_erase.isChecked():
            return "erase"
        return ""

    def _set_mask_preview_visible(self, visible: bool) -> None:
        reg = getattr(self, "_active_region_scope", "whole")
        vis = bool(visible)
        preview_reg = reg if reg != "whole" else getattr(self, "_last_preview_region", "subject")
        if hasattr(self, "btn_mask_show") and hasattr(self, "btn_mask_hide"):
            self.btn_mask_show.blockSignals(True)
            self.btn_mask_hide.blockSignals(True)
            self.btn_mask_show.setChecked(vis)
            self.btn_mask_hide.setChecked(not vis)
            self.btn_mask_show.blockSignals(False)
            self.btn_mask_hide.blockSignals(False)
        self.region_mask_preview_toggled.emit(preview_reg if vis else "")

    @staticmethod
    def _make_mask_fill_icon(filled: bool, size: int = 16) -> QIcon:
        """Draw a dotted circle icon; filled red when active, hollow dotted circle when off."""
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor("#E13860") if filled else QColor("#475569"), 1.6, Qt.DotLine)
        p.setPen(pen)
        p.setBrush(QColor(225, 56, 96, 170) if filled else Qt.NoBrush)
        p.drawEllipse(QRectF(2.5, 2.5, size - 5.0, size - 5.0))
        p.end()
        return QIcon(pm)

    def _on_mask_fill_toggled(self, checked: bool) -> None:
        if hasattr(self, "btn_mask_fill"):
            self.btn_mask_fill.setIcon(self._make_mask_fill_icon(bool(checked), 16))
        self.region_mask_fill_toggled.emit(bool(checked))

    def _on_brush_tool_clicked(self, *, smart: bool, checked: bool) -> None:
        if not checked:
            self._clear_region_brush_buttons()
            self.region_brush_mode_changed.emit("", "", self.slider_region_brush_size.value(), smart)
            return
        if hasattr(self, "btn_brush_normal") and hasattr(self, "btn_brush_smart"):
            self.btn_brush_normal.blockSignals(True)
            self.btn_brush_smart.blockSignals(True)
            self.btn_brush_normal.setChecked(not smart)
            self.btn_brush_smart.setChecked(smart)
            self.btn_brush_normal.blockSignals(False)
            self.btn_brush_smart.blockSignals(False)
        if getattr(self, "_active_region_scope", "whole") == "whole":
            self.set_active_region_scope("custom")
            return
        if not self._active_region_brush_mode():
            self.btn_region_paint.blockSignals(True)
            self.btn_region_paint.setChecked(True)
            self.btn_region_paint.blockSignals(False)
        self._set_mask_preview_visible(True)
        self.region_brush_mode_changed.emit(
            self._active_region_scope,
            self._active_region_brush_mode() or "paint",
            self.slider_region_brush_size.value(),
            smart,
        )

    def _clear_region_brush_buttons(self) -> None:
        for attr in ("btn_region_paint", "btn_region_erase", "btn_brush_normal", "btn_brush_smart"):
            b = getattr(self, attr, None)
            if b is not None:
                b.blockSignals(True)
                b.setChecked(False)
                b.blockSignals(False)

    def _on_region_brush_toggled(self, mode: str, checked: bool) -> None:
        reg = getattr(self, "_active_region_scope", "whole")
        if not checked:
            self._clear_region_brush_buttons()
            self._refresh_region_indicators()
            self.region_brush_mode_changed.emit("", "", self.slider_region_brush_size.value(), True)
            return
        other = self.btn_region_erase if mode == "paint" else self.btn_region_paint
        other.blockSignals(True)
        other.setChecked(False)
        other.blockSignals(False)
        if reg == "whole":
            # Clicking + or - while Whole is selected automatically starts a Custom Smart Brush mask
            self.btn_region_paint.blockSignals(True)
            self.btn_region_erase.blockSignals(True)
            self.btn_region_paint.setChecked(mode == "paint")
            self.btn_region_erase.setChecked(mode == "erase")
            self.btn_region_paint.blockSignals(False)
            self.btn_region_erase.blockSignals(False)
            self.set_active_region_scope("custom")
            self._set_mask_preview_visible(True)
            return
        if hasattr(self, "btn_region_mask_preview") and not self.btn_region_mask_preview.isChecked():
            self._set_mask_preview_visible(True)
        self._refresh_region_indicators()
        self.region_brush_mode_changed.emit(reg, mode, self.slider_region_brush_size.value(), True)

    def _make_mask_tool_icon(
        self,
        kind: str,
        size: int = 20,
        *,
        has_edits: bool = False,
        is_checked: bool = False,
        is_dark: bool | None = None,
    ) -> QIcon:
        """Render bespoke antialiased Photoshop/Lightroom-style vector icons for Auto Mask & Brush tools."""
        dark = getattr(self, "_is_dark", False) if is_dark is None else bool(is_dark)
        fg = QColor("#FFFFFF") if is_checked else (QColor("#E2E8F0") if dark else QColor("#1E395B"))
        accent = QColor("#DBEAFE") if is_checked else (QColor("#60A5FA") if dark else QColor("#2563EB"))
        fill_soft = QColor(255, 255, 255, 55) if is_checked else (QColor(96, 165, 250, 48) if dark else QColor(37, 99, 235, 38))

        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        s = float(size)

        if kind == "whole":
            # Full photo frame with mountain horizon & sun
            p.setPen(QPen(fg, 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(fill_soft)
            p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.76, s * 0.68), 2.5, 2.5)
            p.setBrush(accent)
            p.drawEllipse(QRectF(s * 0.24, s * 0.28, s * 0.14, s * 0.14))
            poly = QPainterPath()
            poly.moveTo(s * 0.15, s * 0.80)
            poly.lineTo(s * 0.42, s * 0.50)
            poly.lineTo(s * 0.58, s * 0.65)
            poly.lineTo(s * 0.72, s * 0.46)
            poly.lineTo(s * 0.85, s * 0.80)
            p.setBrush(Qt.NoBrush)
            p.drawPath(poly)

        elif kind == "subject":
            # Portrait subject silhouette (head + shoulders) inside focus brackets
            p.setPen(QPen(accent, 1.3, Qt.SolidLine, Qt.RoundCap))
            for x0, y0, dx, dy in (
                (0.14, 0.16, 0.12, 0.12),
                (0.86, 0.16, -0.12, 0.12),
                (0.14, 0.84, 0.12, -0.12),
                (0.86, 0.84, -0.12, -0.12),
            ):
                p.drawLine(QPointF(s * x0, s * y0), QPointF(s * (x0 + dx), s * y0))
                p.drawLine(QPointF(s * x0, s * y0), QPointF(s * x0, s * (y0 + dy)))
            p.setPen(QPen(fg, 1.5))
            p.setBrush(fill_soft)
            p.drawEllipse(QRectF(s * 0.36, s * 0.20, s * 0.28, s * 0.28))
            body = QPainterPath()
            body.moveTo(s * 0.24, s * 0.82)
            body.quadTo(s * 0.26, s * 0.54, s * 0.50, s * 0.54)
            body.quadTo(s * 0.74, s * 0.54, s * 0.76, s * 0.82)
            body.closeSubpath()
            p.drawPath(body)

        elif kind == "background":
            # Shaded background frame with dashed cutout person in center
            p.setPen(QPen(fg, 1.4))
            p.setBrush(fill_soft)
            p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.76, s * 0.68), 2.5, 2.5)
            p.setPen(QPen(accent, 1.1, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(s * 0.16, s * 0.34), QPointF(s * 0.34, s * 0.18))
            p.drawLine(QPointF(s * 0.66, s * 0.18), QPointF(s * 0.84, s * 0.34))
            dash_pen = QPen(fg, 1.3, Qt.DashLine)
            p.setPen(dash_pen)
            p.setBrush(QColor("#1E242E") if dark else QColor("#FFFFFF"))
            p.drawEllipse(QRectF(s * 0.39, s * 0.26, s * 0.22, s * 0.22))
            cut = QPainterPath()
            cut.moveTo(s * 0.30, s * 0.82)
            cut.quadTo(s * 0.32, s * 0.54, s * 0.50, s * 0.54)
            cut.quadTo(s * 0.68, s * 0.54, s * 0.70, s * 0.82)
            p.drawPath(cut)

        elif kind == "sky":
            # Cloud + sun above a crisp horizon line
            p.setPen(QPen(accent, 1.4, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(s * 0.12, s * 0.78), QPointF(s * 0.88, s * 0.78))
            p.setPen(QPen(fg, 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(fill_soft)
            cloud = QPainterPath()
            cloud.moveTo(s * 0.26, s * 0.62)
            cloud.cubicTo(s * 0.14, s * 0.62, s * 0.14, s * 0.42, s * 0.28, s * 0.42)
            cloud.cubicTo(s * 0.32, s * 0.24, s * 0.58, s * 0.22, s * 0.64, s * 0.38)
            cloud.cubicTo(s * 0.82, s * 0.38, s * 0.84, s * 0.62, s * 0.68, s * 0.62)
            cloud.closeSubpath()
            p.drawPath(cloud)

        elif kind in ("brush", "smart_brush"):
            # Angled artist paintbrush with soft tip
            p.setPen(QPen(fg, 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(fill_soft)
            # Handle
            p.drawLine(QPointF(s * 0.78, s * 0.18), QPointF(s * 0.44, s * 0.52))
            # Bristle tip drop
            tip = QPainterPath()
            tip.moveTo(s * 0.44, s * 0.52)
            tip.quadTo(s * 0.26, s * 0.52, s * 0.18, s * 0.78)
            tip.quadTo(s * 0.44, s * 0.72, s * 0.44, s * 0.52)
            p.drawPath(tip)
            if kind == "smart_brush":
                # Magic 4-point sparkle star + dashed edge-snap arc
                p.setPen(QPen(accent, 1.4, Qt.SolidLine, Qt.RoundCap))
                cx, cy = s * 0.28, s * 0.26
                p.drawLine(QPointF(cx - s * 0.12, cy), QPointF(cx + s * 0.12, cy))
                p.drawLine(QPointF(cx, cy - s * 0.12), QPointF(cx, cy + s * 0.12))
                p.setPen(QPen(accent, 1.2, Qt.DotLine, Qt.RoundCap))
                p.setBrush(Qt.NoBrush)
                p.drawArc(QRectF(s * 0.12, s * 0.40, s * 0.46, s * 0.46), -110 * 16, 170 * 16)

        elif kind in ("add", "subtract"):
            p.setPen(QPen(fg, 1.5, Qt.SolidLine, Qt.RoundCap))
            p.setBrush(fill_soft)
            p.drawEllipse(QRectF(s * 0.16, s * 0.16, s * 0.68, s * 0.68))
            p.setPen(QPen(fg, 1.9, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(s * 0.33, s * 0.50), QPointF(s * 0.67, s * 0.50))
            if kind == "add":
                p.drawLine(QPointF(s * 0.50, s * 0.33), QPointF(s * 0.50, s * 0.67))

        elif kind == "invert":
            # Split-contrast circle (half filled / half outline) representing Mask Invert
            p.setPen(QPen(fg, 1.5, Qt.SolidLine, Qt.RoundCap))
            p.setBrush(Qt.NoBrush)
            rect = QRectF(s * 0.16, s * 0.16, s * 0.68, s * 0.68)
            p.drawEllipse(rect)
            half = QPainterPath()
            half.moveTo(s * 0.50, s * 0.16)
            half.arcTo(rect, 90, 180)
            half.closeSubpath()
            p.setPen(Qt.NoPen)
            p.setBrush(fg)
            p.drawPath(half)

        elif kind == "eyedropper":
            # Crisp vector eyedropper oriented 45 deg
            p.setPen(QPen(fg, 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(fill_soft)
            # Tip + barrel
            barrel = QPainterPath()
            barrel.moveTo(s * 0.18, s * 0.82)
            barrel.lineTo(s * 0.24, s * 0.68)
            barrel.lineTo(s * 0.54, s * 0.38)
            barrel.lineTo(s * 0.62, s * 0.46)
            barrel.lineTo(s * 0.32, s * 0.76)
            barrel.closeSubpath()
            p.drawPath(barrel)
            # Collar + rubber bulb
            p.setPen(QPen(accent, 1.8, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(s * 0.49, s * 0.33), QPointF(s * 0.67, s * 0.51))
            bulb = QPainterPath()
            bulb.moveTo(s * 0.56, s * 0.36)
            bulb.quadTo(s * 0.74, s * 0.14, s * 0.84, s * 0.24)
            bulb.quadTo(s * 0.90, s * 0.32, s * 0.64, s * 0.44)
            p.setBrush(accent)
            p.setPen(QPen(fg, 1.3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPath(bulb)

        if has_edits:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#F59E0B") if is_checked else QColor("#2563EB"))
            p.drawEllipse(QRectF(s - 5.5, 1.5, 4.0, 4.0))

        p.end()
        return QIcon(pm)

    def _on_region_brush_size_changed(self, size: int) -> None:
        if hasattr(self, "lbl_brush_size_val"):
            self.lbl_brush_size_val.setText(f"{int(size)}px")
        reg = getattr(self, "_active_region_scope", "whole")
        mode = self._active_region_brush_mode()
        if reg != "whole" and mode:
            self.region_brush_mode_changed.emit(reg, mode, int(size), True)

    def _on_region_brush_clear(self) -> None:
        reg = getattr(self, "_active_region_scope", "whole")
        if reg != "whole":
            self.region_brush_cleared.emit(reg)

    def _on_region_mask_preview_toggled(self, checked: bool) -> None:
        reg = getattr(self, "_active_region_scope", "whole")
        preview_reg = reg if reg != "whole" else getattr(self, "_last_preview_region", "subject")
        self.region_mask_preview_toggled.emit(preview_reg if checked else "")

    def _reset_active_region(self) -> None:
        reg = getattr(self, "_active_region_scope", "whole")
        if reg == "whole":
            return
        self._region_edits.pop(reg, None)
        self._load_region_dict_into_sliders({})
        self._refresh_region_indicators()
        self.fine_tune_region_changed.emit(reg, {})
        self.fine_tune_slider_released.emit(f"Reset {reg.title()} Region")

    def _refresh_region_indicators(self) -> None:
        from src.imaging.opencv_engine import _region_dict_has_edits

        cur = getattr(self, "_active_region_scope", "whole")
        tips = getattr(self, "_region_scope_tips", {})
        for rk, btn in getattr(self, "_region_scope_buttons", {}).items():
            btn.setText("")
            if rk == "whole":
                has_ed = False
            else:
                data = self._capture_active_region_sliders() if cur == rk else self._region_edits.get(rk, {})
                has_ed = _region_dict_has_edits(data)
            btn.setIcon(self._make_mask_tool_icon(rk, 20, has_edits=has_ed, is_checked=btn.isChecked()))
            base_tip = tips.get(rk, rk.title())
            btn.setToolTip(f"{base_tip}\n(Has active adjustments)" if has_ed else base_tip)
        for t_btn, t_kind in (
            (getattr(self, "btn_region_paint", None), "add"),
            (getattr(self, "btn_region_erase", None), "subtract"),
            (getattr(self, "btn_mask_invert", None), "invert"),
        ):
            if t_btn is not None:
                t_btn.setIcon(self._make_mask_tool_icon(t_kind, 18, is_checked=t_btn.isChecked()))
        if hasattr(self, "btn_region_reset"):
            if cur != "whole":
                has_cur = _region_dict_has_edits(self._capture_active_region_sliders())
                self.btn_region_reset.setVisible(has_cur)
            else:
                self.btn_region_reset.setVisible(False)
        self._refresh_group_indicators()

    def _refresh_group_indicators(self) -> None:
        from src.imaging.opencv_engine import _is_identity_curve

        if hasattr(self, "_grp_exp"):
            exp_sliders = [
                self.ft_slider_brightness,
                self.ft_slider_contrast,
                self.ft_slider_shadows,
                self.ft_slider_highlights,
                self.ft_slider_clarity,
            ]
            for attr in ("ft_slider_smart_enhance", "ft_slider_whites", "ft_slider_blacks", "ft_slider_structure"):
                if hasattr(self, attr):
                    exp_sliders.append(getattr(self, attr))
            exp_active = (
                any(abs(s.value()) > 0 for s in exp_sliders)
                or self.histogram_widget.black_point != 0
                or self.histogram_widget.white_point != 255
                or any(not _is_identity_curve(pts) for pts in self.tone_curve_widget.get_curves().values())
            )
            self._grp_exp.set_active_indicator(exp_active)
        if hasattr(self, "_grp_color"):
            col_active = any(
                abs(s.value()) > 0
                for s in (self.ft_slider_temp, self.ft_slider_tint, self.ft_slider_sat, self.ft_slider_dehaze)
            ) or any(
                abs(v) > 0
                for d in (self._hsl_hue_vals, self._hsl_sat_vals, self._hsl_lum_vals)
                for v in d.values()
            )
            self._grp_color.set_active_indicator(col_active)
        if hasattr(self, "_grp_str"):
            self._grp_str.set_active_indicator(abs(self.ft_slider_straighten.value()) > 0)
        if hasattr(self, "_grp_det"):
            det_active = (
                self.ft_slider_sharpen.value() > 0
                or self.ft_slider_denoise.value() > 0
                or (hasattr(self, "ft_slider_denoise_chroma") and self.ft_slider_denoise_chroma.value() > 0)
            )
            self._grp_det.set_active_indicator(det_active)
        if hasattr(self, "_grp_optics"):
            opt_active = (
                abs(self.ft_slider_vignette.value()) > 0
                or (hasattr(self, "ft_slider_grain_amount") and self.ft_slider_grain_amount.value() > 0)
                or (hasattr(self, "ft_slider_orton_glow") and self.ft_slider_orton_glow.value() > 0)
                or (hasattr(self, "ft_slider_golden_hour") and self.ft_slider_golden_hour.value() > 0)
            )
            self._grp_optics.set_active_indicator(opt_active)
        if hasattr(self, "ft_slider_sharpen_radius"):
            self.ft_slider_sharpen_radius.setEnabled(self.ft_slider_sharpen.value() > 0)
        if hasattr(self, "ft_slider_sharpen_masking"):
            self.ft_slider_sharpen_masking.setEnabled(self.ft_slider_sharpen.value() > 0)
        if hasattr(self, "ft_slider_denoise_detail"):
            self.ft_slider_denoise_detail.setEnabled(self.ft_slider_denoise.value() > 0)
        if hasattr(self, "ft_slider_vignette_feather"):
            self.ft_slider_vignette_feather.setEnabled(abs(self.ft_slider_vignette.value()) > 0)
        if hasattr(self, "ft_slider_grain_size"):
            self.ft_slider_grain_size.setEnabled(self.ft_slider_grain_amount.value() > 0)

    def _make_curve_channel_icon(
        self,
        channel: str,
        size: int = 18,
        *,
        is_checked: bool = False,
        is_dark: bool | None = None,
    ) -> QIcon:
        """Render Lightroom-style coloured circle channel icons (RGB concentric / R / G / B)."""
        dark = getattr(self, "_is_dark", False) if is_dark is None else bool(is_dark)
        col_map = {
            "rgb": QColor("#F1F5F9") if dark else QColor("#334155"),
            "r": QColor("#EF4444"),
            "g": QColor("#22C55E"),
            "b": QColor("#3B82F6"),
        }
        ring_col = col_map.get(channel, QColor("#94A3B8"))
        pm = QPixmap(size, size + 2)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        cx, cy = size / 2.0, size / 2.0 - 0.5
        rad = size * 0.33
        p.setPen(QPen(ring_col, 2.0 if is_checked else 1.7))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), rad, rad)
        if channel == "rgb":
            p.setPen(Qt.NoPen)
            p.setBrush(ring_col)
            p.drawEllipse(QPointF(cx, cy), rad * 0.42, rad * 0.42)
        if is_checked:
            p.setPen(QPen(ring_col, 1.8, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(size * 0.22, size + 0.5), QPointF(size * 0.78, size + 0.5))
        p.end()
        return QIcon(pm)

    def _select_curve_channel(self, channel: str) -> None:
        for k, btn in getattr(self, "_curve_ch_buttons", {}).items():
            btn.blockSignals(True)
            btn.setChecked(k == channel)
            btn.setIcon(self._make_curve_channel_icon(k, is_checked=(k == channel)))
            btn.blockSignals(False)
        if hasattr(self, "tone_curve_widget"):
            self.tone_curve_widget.set_active_channel(channel)

    def _on_curve_preset_changed(self, idx: int) -> None:
        if self._updating_ui or not hasattr(self, "cmb_curve_preset") or not hasattr(self, "tone_curve_widget"):
            return
        preset = str(self.cmb_curve_preset.itemData(idx) or "linear")
        ch = self.tone_curve_widget._active_channel
        if preset == "linear":
            self.tone_curve_widget.reset_active_channel()
        elif preset == "scurve_med":
            self.tone_curve_widget.apply_s_curve_preset()
        elif preset == "scurve_strong":
            self.tone_curve_widget._curves[ch] = [
                (0.0, 0.0),
                (0.22, 0.14),
                (0.50, 0.50),
                (0.78, 0.86),
                (1.0, 1.0),
            ]
            self.tone_curve_widget.update()
            self.tone_curve_widget.curve_changed.emit()
            self.tone_curve_widget.curve_released.emit()

    def _on_curve_reset_clicked(self) -> None:
        if hasattr(self, "cmb_curve_preset"):
            self.cmb_curve_preset.blockSignals(True)
            self.cmb_curve_preset.setCurrentIndex(0)
            self.cmb_curve_preset.blockSignals(False)
        if hasattr(self, "tone_curve_widget"):
            self.tone_curve_widget.reset_active_channel()

    def _update_hsl_slider_gradients(self) -> None:
        from src.imaging.opencv_engine import HSL_COLOR_BANDS

        n = len(HSL_COLOR_BANDS)
        for idx, (band_key, _, _, hex_col) in enumerate(HSL_COLOR_BANDS):
            sl = self._hsl_sliders.get(band_key)
            if sl is None:
                continue
            if self._hsl_mode == "hue":
                prev_col = HSL_COLOR_BANDS[(idx - 1) % n][3]
                next_col = HSL_COLOR_BANDS[(idx + 1) % n][3]
                sl.set_track_gradient([(0.0, prev_col), (0.5, hex_col), (1.0, next_col)])
            elif self._hsl_mode == "sat":
                sl.set_track_gradient([(0.0, "#7A828E"), (0.5, "#C8D0DA"), (1.0, hex_col)])
            else:
                sl.set_track_gradient([(0.0, "#1E242B"), (0.5, hex_col), (1.0, "#F1F5F9")])

    def _select_hsl_mode(self, mode: str) -> None:
        self._hsl_mode = mode
        for k, btn in getattr(self, "_hsl_tab_buttons", {}).items():
            btn.blockSignals(True)
            btn.setChecked(k == mode)
            btn.blockSignals(False)
        self._update_hsl_slider_gradients()
        store = {"hue": self._hsl_hue_vals, "sat": self._hsl_sat_vals, "lum": self._hsl_lum_vals}[mode]
        self._updating_ui = True
        try:
            for band_key, sl in self._hsl_sliders.items():
                sl.setValue(int(store.get(band_key, 0)))
        finally:
            self._updating_ui = False

    def _on_hsl_slider_changed(self, band_key: str, val: int) -> None:
        store = {"hue": self._hsl_hue_vals, "sat": self._hsl_sat_vals, "lum": self._hsl_lum_vals}[self._hsl_mode]
        store[band_key] = int(val)
        self._update_drawer_indicators()
        if self._updating_ui:
            return
        self._emit_fine_tune_hsl()

    def _reset_hsl_sliders(self) -> None:
        for d in (self._hsl_hue_vals, self._hsl_sat_vals, self._hsl_lum_vals):
            for k in d:
                d[k] = 0
        self._updating_ui = True
        try:
            for sl in self._hsl_sliders.values():
                sl.setValue(0)
        finally:
            self._updating_ui = False
        self._update_drawer_indicators()
        self.fine_tune_hsl_changed.emit(
            dict(self._hsl_hue_vals), dict(self._hsl_sat_vals), dict(self._hsl_lum_vals)
        )
        self.fine_tune_slider_released.emit("Reset Color Mixer")

    def _update_drawer_indicators(self) -> None:
        if hasattr(self, "drawer_tone_curve") and hasattr(self, "tone_curve_widget"):
            from src.imaging.opencv_engine import _is_identity_curve

            curves = self.tone_curve_widget.get_curves()
            curve_active = any(not _is_identity_curve(pts) for pts in curves.values())
            self.drawer_tone_curve.set_active_indicator(curve_active)
        if hasattr(self, "drawer_color_mixer"):
            hsl_active = any(
                abs(v) > 0
                for d in (self._hsl_hue_vals, self._hsl_sat_vals, self._hsl_lum_vals)
                for v in d.values()
            )
            self.drawer_color_mixer.set_active_indicator(hsl_active)

    @staticmethod
    def _make_slider_header(
        text: str,
        slider: _AeroFineTuneSlider | None = None,
        unit: str = "",
        signed: bool | None = None,
    ) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent; border: none;")
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(4)

        lbl = QLabel(text, container)
        lbl.setObjectName("SliderHeaderLabel")
        h_layout.addWidget(lbl)
        h_layout.addStretch(1)

        use_signed = (slider.minimum() < 0) if (slider is not None and signed is None) else bool(signed)

        def _fmt(v: int) -> str:
            prefix = "+" if (use_signed and v > 0) else ""
            return f"{prefix}{v}{unit}"

        init_val = slider.value() if slider is not None else 0
        val_lbl = QLabel(_fmt(init_val), container)
        val_lbl.setObjectName("val_lbl")
        val_lbl.setMinimumWidth(42)
        val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h_layout.addWidget(val_lbl)

        if slider is not None:
            slider._header_widget = container
            container.setEnabled(slider.isEnabled())
            lbl.setToolTip(f"Double-click to reset {text} to default")
            val_lbl.setToolTip(f"Double-click to reset {text} to default")

            def on_double_click(event):
                if slider.isEnabled():
                    slider.setValue(slider.default_val)
                event.accept()

            lbl.mouseDoubleClickEvent = on_double_click
            val_lbl.mouseDoubleClickEvent = on_double_click

            def on_val_changed(val: int):
                val_lbl.setText(_fmt(val))

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
            for attr in (
                "ft_slider_smart_enhance",
                "ft_slider_whites",
                "ft_slider_blacks",
                "ft_slider_structure",
                "ft_slider_sharpen_radius",
                "ft_slider_sharpen_masking",
                "ft_slider_denoise_detail",
                "ft_slider_denoise_chroma",
                "ft_slider_grain_amount",
                "ft_slider_grain_size",
                "ft_slider_vignette_feather",
                "ft_slider_orton_glow",
                "ft_slider_golden_hour",
            ):
                sl = getattr(self, attr, None)
                if sl is not None:
                    sl.setValue(sl.default_val)
            self._region_edits = {}
            self._global_region_mirror = {
                "brightness": 0, "contrast": 0, "shadows": 0, "highlights": 0,
                "whites": 0, "blacks": 0, "clarity": 0, "structure": 0,
                "temperature": 0, "tint": 0, "saturation": 0, "dehaze": 0,
                "curve_rgb": [(0.0, 0.0), (1.0, 1.0)],
                "curve_r": [(0.0, 0.0), (1.0, 1.0)],
                "curve_g": [(0.0, 0.0), (1.0, 1.0)],
                "curve_b": [(0.0, 0.0), (1.0, 1.0)],
            }
            if getattr(self, "_active_region_scope", "whole") != "whole":
                self._active_region_scope = "whole"
                for rk, btn in getattr(self, "_region_scope_buttons", {}).items():
                    btn.blockSignals(True)
                    btn.setChecked(rk == "whole")
                    btn.blockSignals(False)
                self._set_mask_preview_visible(False)
            if hasattr(self, "tone_curve_widget"):
                self.tone_curve_widget.set_curves({})
            if hasattr(self, "cmb_look_profile"):
                self.cmb_look_profile.blockSignals(True)
                self.cmb_look_profile.setCurrentIndex(0)
                self.cmb_look_profile.blockSignals(False)
            if hasattr(self, "ft_slider_look_amount"):
                self.ft_slider_look_amount.setValue(100)
                self.ft_slider_look_amount.setEnabled(False)
            self._sync_wb_preset_combo()
            for d in (getattr(self, "_hsl_hue_vals", {}), getattr(self, "_hsl_sat_vals", {}), getattr(self, "_hsl_lum_vals", {})):
                for k in d:
                    d[k] = 0
            for sl in getattr(self, "_hsl_sliders", {}).values():
                sl.setValue(0)
            self._update_drawer_indicators()
            self._refresh_region_indicators()
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

        self._active_relight_preset_id: str | None = None

        # -------------------------------------------------------------
        # Group 1: Subject Mask & Cutout (Collapsible)
        # -------------------------------------------------------------
        grp_cutout = _FineTuneGroup("Subject Mask & Cutout", content)
        grp_cutout.set_collapsible(True)
        grp_cutout.set_expanded(True)
        self._dl_grp_cutout = grp_cutout

        cutout_row = QHBoxLayout()
        cutout_row.setSpacing(6)
        cutout_row.setContentsMargins(0, 2, 0, 4)

        self.btn_copy_subject = QPushButton("Copy Subject", grp_cutout)
        self.btn_copy_subject.setObjectName("DepthCutoutBtn")
        self.btn_copy_subject.setIcon(make_wlpg_icon("copy_subject", 16))
        self.btn_copy_subject.setIconSize(QSize(16, 16))
        self.btn_copy_subject.setFixedHeight(28)
        self.btn_copy_subject.setCursor(Qt.PointingHandCursor)
        self.btn_copy_subject.setToolTip(
            "Extract the foreground subject with a transparent background (RGBA) and copy to clipboard"
        )
        self.btn_copy_subject.clicked.connect(self.copy_subject_requested.emit)
        cutout_row.addWidget(self.btn_copy_subject, 1)

        self.btn_remove_background = QPushButton("Remove Background", grp_cutout)
        self.btn_remove_background.setObjectName("DepthCutoutBtn")
        self.btn_remove_background.setIcon(make_wlpg_icon("remove_background", 16))
        self.btn_remove_background.setIconSize(QSize(16, 16))
        self.btn_remove_background.setFixedHeight(28)
        self.btn_remove_background.setCursor(Qt.PointingHandCursor)
        self.btn_remove_background.setToolTip(
            "Remove the background from this photo (save as PNG, WebP, AVIF, or JXL to keep transparency)"
        )
        self.btn_remove_background.clicked.connect(self.remove_background_requested.emit)
        cutout_row.addWidget(self.btn_remove_background, 1)

        grp_cutout.vbox.addLayout(cutout_row)

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

        mask_row = QHBoxLayout()
        mask_row.setContentsMargins(0, 2, 0, 0)
        self.btn_toggle_mask_preview = QPushButton("Preview Subject Mask", grp_cutout)
        self.btn_toggle_mask_preview.setCheckable(True)
        self.btn_toggle_mask_preview.setFixedHeight(22)
        self.btn_toggle_mask_preview.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_mask_preview.setToolTip(
            "Show semi-transparent ruby mask overlay to verify edge alignment"
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
        # Group 2: Portrait Bokeh & Focus (Collapsible)
        # -------------------------------------------------------------
        grp_bokeh = _FineTuneGroup("Portrait Bokeh & Focus", content)
        grp_bokeh.set_collapsible(True)
        grp_bokeh.set_expanded(False)
        self._dl_grp_bokeh = grp_bokeh

        self.dl_slider_blur = self._make_aero_slider(0, 100, 0)
        grp_bokeh.vbox.addWidget(self._make_slider_header("Bokeh blur strength", self.dl_slider_blur, unit="%"))
        grp_bokeh.vbox.addWidget(self.dl_slider_blur)

        self.dl_slider_focal = self._make_aero_slider(0, 100, 0)
        grp_bokeh.vbox.addWidget(self._make_slider_header("Change focal depth", self.dl_slider_focal, unit="%"))
        grp_bokeh.vbox.addWidget(self.dl_slider_focal)

        self.dl_slider_glow = self._make_aero_slider(0, 100, 0)
        grp_bokeh.vbox.addWidget(self._make_slider_header("Specular highlight bloom", self.dl_slider_glow, unit="%"))
        grp_bokeh.vbox.addWidget(self.dl_slider_glow)

        for s in (self.dl_slider_blur, self.dl_slider_focal, self.dl_slider_glow):
            s.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
            s.valueChanged.connect(self._emit_depth_bokeh)
            s.sliderReleased.connect(lambda: self._on_depth_slider_released("Adjust Bokeh & Focus"))

        vbox.addWidget(grp_bokeh)

        # -------------------------------------------------------------
        # Group 3: Atmospheric Depth Haze (Collapsible)
        # -------------------------------------------------------------
        grp_haze = _FineTuneGroup("Atmospheric Depth Haze", content)
        grp_haze.set_collapsible(True)
        grp_haze.set_expanded(False)
        self._dl_grp_haze = grp_haze

        self.dl_slider_haze = self._make_aero_slider(0, 100, 0)
        grp_haze.vbox.addWidget(self._make_slider_header("Atmospheric haze / mist", self.dl_slider_haze, unit="%"))
        grp_haze.vbox.addWidget(self.dl_slider_haze)

        self.dl_slider_haze_warmth = self._make_aero_slider(-100, 100, 0)
        grp_haze.vbox.addWidget(self._make_slider_header("Haze warmth (Blue ↔ Amber)", self.dl_slider_haze_warmth))
        grp_haze.vbox.addWidget(self.dl_slider_haze_warmth)

        for s in (self.dl_slider_haze, self.dl_slider_haze_warmth):
            s.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
            s.valueChanged.connect(self._emit_depth_haze)
            s.sliderReleased.connect(lambda: self._on_depth_slider_released("Atmospheric Depth Haze"))

        vbox.addWidget(grp_haze)

        # -------------------------------------------------------------
        # Group 4: Near & Far Fill (Separate Collapsible Section above 3D Relight)
        # -------------------------------------------------------------
        grp_near_far = _FineTuneGroup("Near & Far Fill", content)
        grp_near_far.set_collapsible(True)
        grp_near_far.set_expanded(False)
        self._dl_grp_near_far = grp_near_far

        self.dl_slider_near = self._make_aero_slider(-100, 100, 0)
        grp_near_far.vbox.addWidget(self._make_slider_header("Near subject light", self.dl_slider_near))
        grp_near_far.vbox.addWidget(self.dl_slider_near)

        self.dl_slider_far = self._make_aero_slider(-100, 100, 0)
        grp_near_far.vbox.addWidget(self._make_slider_header("Background light", self.dl_slider_far))
        grp_near_far.vbox.addWidget(self.dl_slider_far)

        self.dl_slider_warmth_near = self._make_aero_slider(-100, 100, 0)
        grp_near_far.vbox.addWidget(self._make_slider_header("Near subject warmth", self.dl_slider_warmth_near))
        grp_near_far.vbox.addWidget(self.dl_slider_warmth_near)

        self.dl_slider_warmth_far = self._make_aero_slider(-100, 100, 0)
        grp_near_far.vbox.addWidget(self._make_slider_header("Background warmth", self.dl_slider_warmth_far))
        grp_near_far.vbox.addWidget(self.dl_slider_warmth_far)

        for s in (self.dl_slider_near, self.dl_slider_far, self.dl_slider_warmth_near, self.dl_slider_warmth_far):
            s.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
            s.valueChanged.connect(self._emit_depth_relight)
            s.sliderReleased.connect(lambda: self._on_depth_slider_released("Near & Far Fill"))

        vbox.addWidget(grp_near_far)

        # -------------------------------------------------------------
        # Group 5 (LAST): 3D Relight (Collapsible)
        # -------------------------------------------------------------
        grp_relight = _FineTuneGroup("3D Relight", content)
        grp_relight.set_collapsible(True)
        grp_relight.set_expanded(True)
        self._dl_grp_relight = grp_relight

        from src.imaging.creative_effects import RELIGHT_LOOK_PRESETS

        presets_hdr = QHBoxLayout()
        presets_hdr.setSpacing(4)
        lbl_looks = QLabel("Presets", grp_relight)
        set_ui_role(lbl_looks, "muted")
        presets_hdr.addWidget(lbl_looks)
        presets_hdr.addStretch(1)

        self.btn_dl_clear_relight = QPushButton("Clear", grp_relight)
        self.btn_dl_clear_relight.setFixedHeight(22)
        self.btn_dl_clear_relight.setCursor(Qt.PointingHandCursor)
        self.btn_dl_clear_relight.setToolTip("Reset all 3D Relight presets, virtual lights, and near/far fill")
        self.btn_dl_clear_relight.clicked.connect(self._on_clear_relight_clicked)
        presets_hdr.addWidget(self.btn_dl_clear_relight)
        grp_relight.vbox.addLayout(presets_hdr)

        looks_grid = QGridLayout()
        looks_grid.setSpacing(5)
        self.dl_look_buttons: list[_RelightSphereButton] = []
        for idx, preset in enumerate(RELIGHT_LOOK_PRESETS):
            btn_look = _RelightSphereButton(preset, grp_relight)
            btn_look.setToolTip(f"Apply {preset['name']} 3D lighting preset")
            btn_look.clicked.connect(lambda _=False, pr=preset: self._apply_relight_look_preset(pr))
            looks_grid.addWidget(btn_look, idx // 3, idx % 3)
            self.dl_look_buttons.append(btn_look)
        grp_relight.vbox.addLayout(looks_grid)

        self.dl_slider_original_light = self._make_aero_slider(0, 100, 100)
        self.dl_slider_original_light.setToolTip(
            "Original scene brightness (100% = untouched; drag left to darken ambient light)"
        )
        grp_relight.vbox.addWidget(
            self._make_slider_header("Original light", self.dl_slider_original_light, unit="%", signed=False)
        )
        grp_relight.vbox.addWidget(self.dl_slider_original_light)
        self.dl_slider_original_light.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
        self.dl_slider_original_light.valueChanged.connect(self._emit_depth_virtual_lights)
        self.dl_slider_original_light.sliderReleased.connect(
            lambda: self._on_depth_slider_released("3D Relight")
        )

        # Row 1: Light Slots (colored dots) + Add + Gizmos
        slot_btn_qss = """
            QPushButton {
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 8.5pt;
                font-weight: 600;
            }
            QPushButton:checked {
                background: rgba(37, 99, 235, 0.22);
                border: 1.5px solid #2563EB;
            }
        """

        vlight_bar = QHBoxLayout()
        vlight_bar.setSpacing(4)
        self.dl_light_slot_btns: list[QPushButton] = []
        for slot_i in range(3):
            b_slot = QPushButton(f"{slot_i + 1}", grp_relight)
            b_slot.setCheckable(True)
            b_slot.setFixedHeight(26)
            b_slot.setMinimumWidth(42)
            b_slot.setCursor(Qt.PointingHandCursor)
            b_slot.setStyleSheet(slot_btn_qss)
            b_slot.setVisible(False)
            b_slot.clicked.connect(lambda _=False, s_idx=slot_i: self._select_virtual_light_slot(s_idx))
            vlight_bar.addWidget(b_slot)
            self.dl_light_slot_btns.append(b_slot)

        self.btn_dl_add_light = QPushButton("+ Add", grp_relight)
        self.btn_dl_add_light.setFixedHeight(26)
        self.btn_dl_add_light.setMinimumWidth(56)
        self.btn_dl_add_light.setCursor(Qt.PointingHandCursor)
        self.btn_dl_add_light.setToolTip("Add a virtual 3D light (up to 3)")
        self.btn_dl_add_light.clicked.connect(self._on_add_virtual_light_clicked)
        vlight_bar.addWidget(self.btn_dl_add_light)

        vlight_bar.addStretch(1)

        self.btn_dl_show_gizmos = QPushButton("Gizmos", grp_relight)
        self.btn_dl_show_gizmos.setCheckable(True)
        self.btn_dl_show_gizmos.setChecked(True)
        self.btn_dl_show_gizmos.setFixedHeight(26)
        self.btn_dl_show_gizmos.setMinimumWidth(58)
        self.btn_dl_show_gizmos.setStyleSheet(slot_btn_qss)
        self.btn_dl_show_gizmos.setCursor(Qt.PointingHandCursor)
        self.btn_dl_show_gizmos.setToolTip("Show / hide draggable 3D light handles on photo")
        self.btn_dl_show_gizmos.toggled.connect(self.depth_light_gizmos_toggled.emit)
        vlight_bar.addWidget(self.btn_dl_show_gizmos)
        grp_relight.vbox.addLayout(vlight_bar)

        # Active Light Inspector
        self.dl_light_editor_box = QWidget(grp_relight)
        ed_lay = QVBoxLayout(self.dl_light_editor_box)
        ed_lay.setContentsMargins(0, 6, 0, 2)
        ed_lay.setSpacing(6)

        # Row 2: Full-width Type Segmented Control (Point | Spot | Rim)
        type_seg_row = QHBoxLayout()
        type_seg_row.setSpacing(4)
        self.btn_dl_type_point = QPushButton("Point", self.dl_light_editor_box)
        self.btn_dl_type_spot = QPushButton("Spot", self.dl_light_editor_box)
        self.btn_dl_type_rim = QPushButton("Rim", self.dl_light_editor_box)
        self._dl_type_btns = {
            "point": self.btn_dl_type_point,
            "spot": self.btn_dl_type_spot,
            "rim": self.btn_dl_type_rim,
        }
        for key, b_t in self._dl_type_btns.items():
            b_t.setCheckable(True)
            b_t.setFixedHeight(24)
            b_t.setCursor(Qt.PointingHandCursor)
            b_t.setStyleSheet(slot_btn_qss)
            b_t.clicked.connect(lambda _=False, k=key: self._set_active_light_type(k))
            type_seg_row.addWidget(b_t, 1)
        ed_lay.addLayout(type_seg_row)

        # Row 3: Uncramped Light On/Off + Delete Light
        state_row = QHBoxLayout()
        state_row.setSpacing(6)
        self.btn_dl_light_enabled = QPushButton("Light On", self.dl_light_editor_box)
        self.btn_dl_light_enabled.setCheckable(True)
        self.btn_dl_light_enabled.setChecked(True)
        self.btn_dl_light_enabled.setFixedHeight(24)
        self.btn_dl_light_enabled.setStyleSheet(slot_btn_qss)
        self.btn_dl_light_enabled.setCursor(Qt.PointingHandCursor)
        self.btn_dl_light_enabled.setToolTip("Temporarily toggle this light on or off")
        self.btn_dl_light_enabled.clicked.connect(self._on_light_enabled_toggled)
        state_row.addWidget(self.btn_dl_light_enabled, 1)

        self.btn_dl_remove_light = QPushButton("Delete Light", self.dl_light_editor_box)
        self.btn_dl_remove_light.setFixedHeight(24)
        self.btn_dl_remove_light.setCursor(Qt.PointingHandCursor)
        self.btn_dl_remove_light.setToolTip("Remove the selected virtual light")
        self.btn_dl_remove_light.clicked.connect(self._on_remove_virtual_light_clicked)
        state_row.addWidget(self.btn_dl_remove_light, 1)
        ed_lay.addLayout(state_row)

        # Row 4: Color + Swatches + Auto K
        col_row = QHBoxLayout()
        col_row.setSpacing(4)
        lbl_dl_color = QLabel("Colour", self.dl_light_editor_box)
        lbl_dl_color.setObjectName("SliderHeaderLabel")
        col_row.addWidget(lbl_dl_color)
        self.btn_dl_light_color = QPushButton("", self.dl_light_editor_box)
        self.btn_dl_light_color.setFixedSize(26, 20)
        self.btn_dl_light_color.setToolTip("Pick custom light color")
        self.btn_dl_light_color.clicked.connect(self._on_pick_virtual_light_color)
        col_row.addWidget(self.btn_dl_light_color)
        for sw_hex in ("#ff7e28", "#ffb85c", "#ffe8d5", "#ffffff", "#00e5ff", "#ff2a9d"):
            b_sw = QPushButton("", self.dl_light_editor_box)
            b_sw.setFixedSize(16, 16)
            b_sw.setStyleSheet(f"background-color: {sw_hex}; border: 1px solid #64748B; border-radius: 3px;")
            b_sw.setToolTip(f"Set color {sw_hex}")
            b_sw.clicked.connect(lambda _=False, hx=sw_hex: self._apply_swatch_to_active_light(hx))
            col_row.addWidget(b_sw)
        col_row.addStretch(1)
        self.btn_dl_color_kelvin_reset = QPushButton("Auto K", self.dl_light_editor_box)
        self.btn_dl_color_kelvin_reset.setFixedHeight(22)
        self.btn_dl_color_kelvin_reset.setCursor(Qt.PointingHandCursor)
        self.btn_dl_color_kelvin_reset.setToolTip("Clear custom swatch color and use Kelvin temperature")
        self.btn_dl_color_kelvin_reset.clicked.connect(self._on_clear_light_color_override)
        col_row.addWidget(self.btn_dl_color_kelvin_reset)
        ed_lay.addLayout(col_row)

        # Essential Light Sliders (unsigned formatted readouts)
        self.dl_slider_light_kelvin = self._make_aero_slider(1800, 10000, 4500)
        ed_lay.addWidget(self._make_slider_header("Temperature", self.dl_slider_light_kelvin, unit="K", signed=False))
        ed_lay.addWidget(self.dl_slider_light_kelvin)

        self.dl_slider_light_intensity = self._make_aero_slider(0, 100, 70)
        ed_lay.addWidget(self._make_slider_header("Intensity", self.dl_slider_light_intensity, unit="%", signed=False))
        ed_lay.addWidget(self.dl_slider_light_intensity)

        self.dl_slider_light_softness = self._make_aero_slider(0, 100, 45)
        ed_lay.addWidget(self._make_slider_header("Softness", self.dl_slider_light_softness, unit="%", signed=False))
        ed_lay.addWidget(self.dl_slider_light_softness)

        self.dl_slider_light_z = self._make_aero_slider(0, 100, 25)
        ed_lay.addWidget(self._make_slider_header("Distance (Close ↔ Far)", self.dl_slider_light_z, unit="%", signed=False))
        ed_lay.addWidget(self.dl_slider_light_z)

        # Conditional Spot Cone Slider (hidden unless type == "spot")
        self.dl_spot_cone_container = QWidget(self.dl_light_editor_box)
        cone_lay = QVBoxLayout(self.dl_spot_cone_container)
        cone_lay.setContentsMargins(0, 0, 0, 0)
        cone_lay.setSpacing(2)
        self.dl_slider_light_cone = self._make_aero_slider(10, 90, 40)
        cone_lay.addWidget(self._make_slider_header("Spot cone", self.dl_slider_light_cone, unit="°", signed=False))
        cone_lay.addWidget(self.dl_slider_light_cone)
        self.dl_spot_cone_container.setVisible(False)
        ed_lay.addWidget(self.dl_spot_cone_container)

        # Collapsible Fine Shaping Drawer (Gloss, Depth range, Light Pattern)
        self.dl_drawer_shaping = _CollapsibleDrawer("Fine Shaping (Gloss, Depth, Pattern)", self.dl_light_editor_box, expanded=False)
        self.dl_slider_light_gloss = self._make_aero_slider(0, 100, 30)
        self.dl_drawer_shaping.body_layout.addWidget(
            self._make_slider_header("Gloss / Specular", self.dl_slider_light_gloss, unit="%", signed=False)
        )
        self.dl_drawer_shaping.body_layout.addWidget(self.dl_slider_light_gloss)

        self.dl_slider_light_depth_from = self._make_aero_slider(0, 100, 0)
        self.dl_drawer_shaping.body_layout.addWidget(
            self._make_slider_header("Depth range: From", self.dl_slider_light_depth_from, unit="%", signed=False)
        )
        self.dl_drawer_shaping.body_layout.addWidget(self.dl_slider_light_depth_from)

        self.dl_slider_light_depth_to = self._make_aero_slider(0, 100, 100)
        self.dl_drawer_shaping.body_layout.addWidget(
            self._make_slider_header("Depth range: To", self.dl_slider_light_depth_to, unit="%", signed=False)
        )
        self.dl_drawer_shaping.body_layout.addWidget(self.dl_slider_light_depth_to)

        gobo_row = QHBoxLayout()
        gobo_row.setSpacing(6)
        lbl_pattern = QLabel("Pattern:", self.dl_drawer_shaping)
        lbl_pattern.setToolTip("Projects a shaped shadow pattern (like window frames or blinds) from this light")
        gobo_row.addWidget(lbl_pattern)
        self.cmb_dl_light_gobo = QComboBox(self.dl_drawer_shaping)
        self.cmb_dl_light_gobo.setToolTip("Projects a shaped shadow pattern (like window frames or blinds) from this light")
        self.cmb_dl_light_gobo.addItem("None", "none")
        self.cmb_dl_light_gobo.addItem("Venetian Blinds", "blinds")
        self.cmb_dl_light_gobo.addItem("Window Frame", "window")
        self.cmb_dl_light_gobo.addItem("Tree Foliage", "foliage")
        self.cmb_dl_light_gobo.currentIndexChanged.connect(self._on_virtual_light_combo_or_color_changed)
        gobo_row.addWidget(self.cmb_dl_light_gobo, 1)
        self.dl_drawer_shaping.body_layout.addLayout(gobo_row)
        ed_lay.addWidget(self.dl_drawer_shaping)

        for sl in (
            self.dl_slider_light_kelvin,
            self.dl_slider_light_intensity,
            self.dl_slider_light_softness,
            self.dl_slider_light_z,
            self.dl_slider_light_gloss,
            self.dl_slider_light_cone,
            self.dl_slider_light_depth_from,
            self.dl_slider_light_depth_to,
        ):
            sl.sliderPressed.connect(self.fine_tune_slider_pressed.emit)
            sl.valueChanged.connect(self._on_virtual_light_slider_changed)
            sl.sliderReleased.connect(lambda: self._on_depth_slider_released("Adjust Virtual Light"))

        self.dl_light_editor_box.setVisible(False)
        grp_relight.vbox.addWidget(self.dl_light_editor_box)

        vbox.addWidget(grp_relight)

        # Bottom Actions
        action_bar = QVBoxLayout()
        action_bar.setSpacing(5)
        action_bar.setContentsMargins(0, 4, 0, 4)

        self.btn_dl_hold_compare = QPushButton("Hold to Compare (Original)", content)
        self.btn_dl_hold_compare.setObjectName("FineTuneActionBtn")
        self.btn_dl_hold_compare.setFixedHeight(26)
        self.btn_dl_hold_compare.setCursor(Qt.PointingHandCursor)
        self.btn_dl_hold_compare.setToolTip("Click and hold to preview original unedited photo (\\)")
        self.btn_dl_hold_compare.pressed.connect(self.compare_started.emit)
        self.btn_dl_hold_compare.released.connect(self.compare_ended.emit)
        action_bar.addWidget(self.btn_dl_hold_compare)

        self.btn_dl_reset = QPushButton("Reset All Depth && Light", content)
        self.btn_dl_reset.setObjectName("FineTuneResetBtn")
        self.btn_dl_reset.setFixedHeight(26)
        self.btn_dl_reset.setCursor(Qt.PointingHandCursor)
        self.btn_dl_reset.setToolTip("Reset all cutout, bokeh, haze, and 3D Relight adjustments")
        self.btn_dl_reset.clicked.connect(self.depth_reset_requested.emit)
        action_bar.addWidget(self.btn_dl_reset)

        vbox.addLayout(action_bar)
        vbox.addStretch(1)

        scroll.setWidget(content)

        self._apply_fine_tune_tooltips()
        return scroll

    def _apply_fine_tune_tooltips(self) -> None:
        tips = {
            "btn_ft_basic": "Basic Mode: clean, classic exposure, color, detail, and luminance histogram controls",
            "btn_ft_advanced": "Advanced Mode: unlocks RGB histogram, Tone Curve, WB Eyedropper, HSL Color Mixer, Film Grain, and Regional Masking",
            "cmb_look_profile": "Choose a non-destructive color Look Profile or custom .cube 3D LUT",
            "btn_import_cube_lut": "Import a custom .cube 3D LUT file",
            "ft_slider_look_amount": "Blend strength (0–200%) of the active Look Profile or .cube LUT",
            "cmb_wb_preset": "White Balance lighting preset (As Shot, Daylight, Cloudy, Shade, Tungsten, Fluorescent, Flash)",
            "histogram_widget": "Live tonal histogram — drag the bottom black/white point triangles to clip shadows or highlights",
            "tone_curve_widget": "Click to add control points, drag to shape tone response, right-click a point to remove it",
            "drawer_tone_curve": "Expand to shape contrast and color channels with a Tone Curve or 1-click S-Curve preset",
            "drawer_color_mixer": "Expand to adjust Hue, Saturation, and Luminance across 8 individual color bands",
            "ft_slider_smart_enhance": "One-slider adaptive enhance: balances shadows, midtone contrast, and vibrance automatically",
            "ft_slider_brightness": "Exposure: brighten or darken the photo with filmic highlight roll-off (double-click to reset)",
            "ft_slider_contrast": "Contrast: expand or soften tonal separation between shadows and highlights",
            "ft_slider_shadows": "Shadows: recover detail in dark areas without washing out blacks",
            "ft_slider_highlights": "Highlights: pull back bright skies and specular areas to recover detail",
            "ft_slider_whites": "Whites: set the upper white point brightness",
            "ft_slider_blacks": "Blacks: deepen or lift the darkest shadow tones",
            "ft_slider_clarity": "Clarity: boost midtone local contrast for crisp texture and punch",
            "ft_slider_structure": "Structure: enhance architectural and landscape micro-detail while protecting skin tones and skies",
            "ft_slider_temp": "Color temperature: shift white balance cooler (blue) or warmer (amber)",
            "ft_slider_tint": "Tint: fine-tune green vs. magenta white balance cast",
            "ft_slider_sat": "Saturation: boost color richness while protecting natural skin tones",
            "ft_slider_dehaze": "Dehaze: cut through atmospheric haze, fog, and low-contrast veil",
            "ft_slider_straighten": "Straighten angle (-15° to +15°) with automatic edge-free crop",
            "btn_auto_straighten": "Automatically detect the horizon or dominant lines and straighten the photo",
            "ft_slider_sharpen": "Sharpen: enhance fine edge crispness in luminance",
            "ft_slider_sharpen_radius": "Sharpen radius: adjust edge halo width (small for fine detail, larger for soft lenses)",
            "ft_slider_sharpen_masking": "Sharpen masking: restrict sharpening to real edges so smooth skies stay noise-free",
            "ft_slider_denoise": "Reduce noise: smooth luminance grain while preserving sharp edges",
            "ft_slider_denoise_detail": "Detail preservation: keep fine textures crisp while denoising flat areas",
            "ft_slider_denoise_chroma": "Color noise: remove red/green/blue chroma blotches from high-ISO shots",
            "ft_slider_vignette": "Vignette: darken or lighten photo corners to draw focus to the center",
            "ft_slider_vignette_feather": "Vignette feather: control how softly the vignette fades toward the center",
            "ft_slider_grain_amount": "Film grain: add organic analog film texture",
            "ft_slider_grain_size": "Grain size: adjust film grain particle coarseness",
            "ft_slider_orton_glow": "Orton glow: add dreamy highlight bloom and soft contrast",
            "ft_slider_golden_hour": "Golden hour: bathe highlights and midtones in warm sunset light",
            "btn_hold_compare": "Click and hold to compare with the original unedited photo (\\)",
            "btn_reset_default": "Reset all Fine Tune adjustments back to their defaults",
        }
        for attr, tip in tips.items():
            w = getattr(self, attr, None)
            if w is not None:
                w.setToolTip(tip)

    @staticmethod
    def _make_aero_slider(min_val: int, max_val: int, default_val: int) -> _AeroFineTuneSlider:
        return _AeroFineTuneSlider(min_val, max_val, default_val)

    def _emit_fine_tune_exposure(self) -> None:
        self._refresh_region_indicators()
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_ft", False):
            self._is_sliding_ft = True
            self.fine_tune_slider_pressed.emit()
        if getattr(self, "_active_region_scope", "whole") != "whole":
            self._pending_ft_region = True
        else:
            self._pending_ft_exposure = True
        if not self._ft_timer.isActive():
            self._ft_timer.start()

    def _emit_fine_tune_exposure_ext(self) -> None:
        self._refresh_region_indicators()
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_ft", False):
            self._is_sliding_ft = True
            self.fine_tune_slider_pressed.emit()
        if getattr(self, "_active_region_scope", "whole") != "whole":
            self._pending_ft_region = True
        else:
            self._pending_ft_exposure_ext = True
        if not self._ft_timer.isActive():
            self._ft_timer.start()

    def _emit_fine_tune_smart_enhance(self) -> None:
        self._refresh_group_indicators()
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_ft", False):
            self._is_sliding_ft = True
            self.fine_tune_slider_pressed.emit()
        self._pending_ft_smart_enhance = True
        if not self._ft_timer.isActive():
            self._ft_timer.start()

    def _emit_fine_tune_structure(self) -> None:
        self._refresh_region_indicators()
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_ft", False):
            self._is_sliding_ft = True
            self.fine_tune_slider_pressed.emit()
        if getattr(self, "_active_region_scope", "whole") != "whole":
            self._pending_ft_region = True
        else:
            self._pending_ft_structure = True
        if not self._ft_timer.isActive():
            self._ft_timer.start()

    def _sync_wb_preset_combo(self) -> None:
        if not hasattr(self, "cmb_wb_preset"):
            return
        cur_t = int(self.ft_slider_temp.value())
        cur_ti = int(self.ft_slider_tint.value())
        matched_key = "custom"
        for wb_key, _lbl, pt, pti in getattr(self, "_wb_presets", ()):
            if pt is not None and pti is not None and pt == cur_t and pti == cur_ti:
                matched_key = wb_key
                break
        self.cmb_wb_preset.blockSignals(True)
        idx = self.cmb_wb_preset.findData(matched_key)
        if idx >= 0:
            self.cmb_wb_preset.setCurrentIndex(idx)
        self.cmb_wb_preset.blockSignals(False)

    def _on_wb_preset_changed(self, idx: int) -> None:
        if self._updating_ui or not hasattr(self, "cmb_wb_preset"):
            return
        key = self.cmb_wb_preset.itemData(idx)
        for wb_key, _lbl, pt, pti in getattr(self, "_wb_presets", ()):
            if wb_key == key and pt is not None and pti is not None:
                self._updating_ui = True
                try:
                    self.ft_slider_temp.setValue(int(pt))
                    self.ft_slider_tint.setValue(int(pti))
                finally:
                    self._updating_ui = False
                self._emit_fine_tune_color()
                self._on_fine_tune_slider_released("White Balance Preset")
                break

    def _on_look_profile_combo_changed(self, idx: int) -> None:
        if self._updating_ui or not hasattr(self, "cmb_look_profile"):
            return
        effect_key = str(self.cmb_look_profile.itemData(idx) or "none")
        if hasattr(self, "ft_slider_look_amount"):
            self.ft_slider_look_amount.setEnabled(effect_key not in ("none", "as_shot", ""))
        intensity = self.ft_slider_look_amount.value() if hasattr(self, "ft_slider_look_amount") else 100
        self.fine_tune_look_profile_changed.emit(effect_key, int(intensity))
        self.fine_tune_slider_released.emit("Look Profile")

    def _on_import_cube_lut_clicked(self) -> None:
        from pathlib import Path
        path, _ = QFileDialog.getOpenFileName(
            self, "Import 3D LUT (.cube)", "", "3D LUT Files (*.cube);;All Files (*)"
        )
        if not path:
            return
        key = f"cube:{path}"
        lbl = f"LUT: {Path(path).stem}"
        idx = self.cmb_look_profile.findData(key)
        if idx < 0:
            self.cmb_look_profile.addItem(lbl, key)
            idx = self.cmb_look_profile.count() - 1
        self.cmb_look_profile.setCurrentIndex(idx)

    def _emit_fine_tune_look(self) -> None:
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_ft", False):
            self._is_sliding_ft = True
            self.fine_tune_slider_pressed.emit()
        self._pending_ft_look = True
        if not self._ft_timer.isActive():
            self._ft_timer.start()

    def _emit_fine_tune_color(self) -> None:
        self._refresh_region_indicators()
        self._sync_wb_preset_combo()
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_ft", False):
            self._is_sliding_ft = True
            self.fine_tune_slider_pressed.emit()
        if getattr(self, "_active_region_scope", "whole") != "whole":
            self._pending_ft_region = True
        else:
            self._pending_ft_color = True
        if not self._ft_timer.isActive():
            self._ft_timer.start()

    def _emit_fine_tune_optics(self) -> None:
        self._refresh_group_indicators()
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_ft", False):
            self._is_sliding_ft = True
            self.fine_tune_slider_pressed.emit()
        self._pending_ft_optics = True
        if not self._ft_timer.isActive():
            self._ft_timer.start()

    def sync_straighten_angle(self, angle_deg: float) -> None:
        """Synchronize the Fine Tune straighten slider with live Crop & Straighten adjustments."""
        if not hasattr(self, "ft_slider_straighten"):
            return
        val = int(round(max(-15.0, min(15.0, float(angle_deg)))))
        self.ft_slider_straighten.blockSignals(True)
        self.ft_slider_straighten.setValue(val)
        self.ft_slider_straighten.blockSignals(False)
        hdr = getattr(self.ft_slider_straighten, "_header_widget", None)
        if hdr is not None:
            val_lbl = hdr.findChild(QLabel, "val_lbl")
            if val_lbl is not None:
                if abs(float(angle_deg)) < 0.05:
                    val_lbl.setText("0°")
                else:
                    val_lbl.setText(f"{float(angle_deg):+.1f}°")
        if hasattr(self, "_grp_str"):
            self._grp_str.set_active_indicator(abs(float(angle_deg)) >= 0.05)

    def _emit_fine_tune_straighten(self) -> None:
        self._refresh_group_indicators()
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_ft", False):
            self._is_sliding_ft = True
            self.fine_tune_slider_pressed.emit()
        self._pending_ft_straighten = True
        if not self._ft_timer.isActive():
            self._ft_timer.start()

    def _emit_fine_tune_detail(self) -> None:
        self._refresh_group_indicators()
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_ft", False):
            self._is_sliding_ft = True
            self.fine_tune_slider_pressed.emit()
        self._pending_ft_detail = True
        if not self._ft_timer.isActive():
            self._ft_timer.start()

    def _emit_fine_tune_detail_ext(self) -> None:
        self._refresh_group_indicators()
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_ft", False):
            self._is_sliding_ft = True
            self.fine_tune_slider_pressed.emit()
        self._pending_ft_detail_ext = True
        if not self._ft_timer.isActive():
            self._ft_timer.start()

    def _emit_fine_tune_curve(self) -> None:
        self._update_drawer_indicators()
        self._refresh_region_indicators()
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_ft", False):
            self._is_sliding_ft = True
            self.fine_tune_slider_pressed.emit()
        if getattr(self, "_active_region_scope", "whole") != "whole":
            self._pending_ft_region = True
        else:
            self._pending_ft_curve = True
        if not self._ft_timer.isActive():
            self._ft_timer.start()

    def _emit_fine_tune_hsl(self) -> None:
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_ft", False):
            self._is_sliding_ft = True
            self.fine_tune_slider_pressed.emit()
        self._pending_ft_hsl = True
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
        if self._pending_ft_exposure_ext:
            self._pending_ft_exposure_ext = False
            if hasattr(self, "ft_slider_whites") and hasattr(self, "ft_slider_blacks"):
                self.fine_tune_exposure_ext_changed.emit(
                    self.ft_slider_whites.value(),
                    self.ft_slider_blacks.value(),
                )
        if self._pending_ft_smart_enhance:
            self._pending_ft_smart_enhance = False
            if hasattr(self, "ft_slider_smart_enhance"):
                self.fine_tune_smart_enhance_changed.emit(self.ft_slider_smart_enhance.value())
        if self._pending_ft_structure:
            self._pending_ft_structure = False
            if hasattr(self, "ft_slider_structure"):
                self.fine_tune_structure_changed.emit(self.ft_slider_structure.value())
        if self._pending_ft_color:
            self._pending_ft_color = False
            self.fine_tune_color_changed.emit(
                self.ft_slider_temp.value(),
                self.ft_slider_tint.value(),
                self.ft_slider_sat.value(),
                self.ft_slider_dehaze.value(),
            )
        if getattr(self, "_pending_ft_look", False):
            self._pending_ft_look = False
            if hasattr(self, "cmb_look_profile") and hasattr(self, "ft_slider_look_amount"):
                effect_key = str(self.cmb_look_profile.currentData() or "none")
                self.fine_tune_look_profile_changed.emit(effect_key, int(self.ft_slider_look_amount.value()))
        if self._pending_ft_optics:
            self._pending_ft_optics = False
            self.fine_tune_optics_changed.emit(self.ft_slider_vignette.value())
            if hasattr(self, "ft_slider_vignette_feather"):
                self.fine_tune_optics_ext_changed.emit(
                    self.ft_slider_vignette.value(),
                    self.ft_slider_vignette_feather.value(),
                    self.ft_slider_orton_glow.value() if hasattr(self, "ft_slider_orton_glow") else 0,
                    self.ft_slider_golden_hour.value() if hasattr(self, "ft_slider_golden_hour") else 0,
                )
        if self._pending_ft_region:
            self._pending_ft_region = False
            reg = getattr(self, "_active_region_scope", "whole")
            if reg != "whole":
                data = self._capture_active_region_sliders()
                self._region_edits[reg] = data
                self.fine_tune_region_changed.emit(reg, dict(data))
        if self._pending_ft_straighten:
            self._pending_ft_straighten = False
            self.fine_tune_straighten_changed.emit(float(self.ft_slider_straighten.value()))
        if self._pending_ft_detail:
            self._pending_ft_detail = False
            self.fine_tune_detail_changed.emit(
                self.ft_slider_sharpen.value(),
                self.ft_slider_denoise.value(),
            )
        if self._pending_ft_detail_ext:
            self._pending_ft_detail_ext = False
            self.fine_tune_detail_ext_changed.emit(
                self.ft_slider_denoise_chroma.value(),
                self.ft_slider_grain_amount.value(),
                self.ft_slider_grain_size.value(),
                self.ft_slider_sharpen_masking.value() if hasattr(self, "ft_slider_sharpen_masking") else 0,
                self.ft_slider_sharpen_radius.value() if hasattr(self, "ft_slider_sharpen_radius") else 25,
                self.ft_slider_denoise_detail.value() if hasattr(self, "ft_slider_denoise_detail") else 50,
            )
        if self._pending_ft_curve:
            self._pending_ft_curve = False
            self.fine_tune_curve_changed.emit(self.tone_curve_widget.get_curves())
        if self._pending_ft_hsl:
            self._pending_ft_hsl = False
            self.fine_tune_hsl_changed.emit(
                dict(self._hsl_hue_vals),
                dict(self._hsl_sat_vals),
                dict(self._hsl_lum_vals),
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

    def _emit_depth_virtual_lights(self) -> None:
        if self._updating_ui:
            return
        if not getattr(self, "_is_sliding_depth", False):
            self._is_sliding_depth = True
            self.fine_tune_slider_pressed.emit()
        self._pending_dl_virtual_lights = True
        if not self._dl_timer.isActive():
            self._dl_timer.start()

    @staticmethod
    def _make_color_dot_icon(rgb: tuple[int, int, int]) -> QIcon:
        pix = QPixmap(10, 10)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setBrush(QBrush(QColor(rgb[0], rgb[1], rgb[2])))
        p.setPen(QPen(QColor(30, 41, 59, 180), 1.0))
        p.drawEllipse(QRectF(1.0, 1.0, 8.0, 8.0))
        p.end()
        return QIcon(pix)

    def _ambient_dim_value(self) -> int:
        if hasattr(self, "dl_slider_original_light"):
            return max(0, min(100, 100 - int(self.dl_slider_original_light.value())))
        return 0

    def _refresh_depth_light_group_indicators(self) -> None:
        if hasattr(self, "_dl_grp_cutout"):
            has_cutout = (
                self.dl_slider_boundary_shift.value() != 0
                or self.dl_slider_boundary_feather.value() != 0
            )
            self._dl_grp_cutout.set_active_indicator(has_cutout)
        if hasattr(self, "_dl_grp_bokeh"):
            has_bokeh = (
                self.dl_slider_blur.value() > 0
                or self.dl_slider_focal.value() > 0
                or self.dl_slider_glow.value() > 0
            )
            self._dl_grp_bokeh.set_active_indicator(has_bokeh)
        if hasattr(self, "_dl_grp_haze"):
            has_haze = self.dl_slider_haze.value() > 0 or self.dl_slider_haze_warmth.value() != 0
            self._dl_grp_haze.set_active_indicator(has_haze)
        has_near_far = any(
            getattr(self, a).value() != 0
            for a in ("dl_slider_near", "dl_slider_far", "dl_slider_warmth_near", "dl_slider_warmth_far")
            if hasattr(self, a)
        )
        if hasattr(self, "_dl_grp_near_far"):
            self._dl_grp_near_far.set_active_indicator(has_near_far)
        has_relight = bool(self._dl_virtual_lights) or self._ambient_dim_value() > 0
        if hasattr(self, "_dl_grp_relight"):
            self._dl_grp_relight.set_active_indicator(has_relight)
        if hasattr(self, "dl_drawer_shaping") and self._dl_virtual_lights:
            lt = self._dl_virtual_lights[self._dl_selected_light_idx]
            has_shape = (
                int(round(float(lt.get("gloss", 30.0)))) != 30
                or int(round(float(lt.get("depth_from", 0.0)) * 100.0)) != 0
                or int(round(float(lt.get("depth_to", 1.0)) * 100.0)) != 100
                or str(lt.get("gobo", "none")).lower() != "none"
            )
            self.dl_drawer_shaping.set_active_indicator(has_shape)

    def _refresh_virtual_lights_ui(self) -> None:
        from src.imaging.creative_effects import light_color_srgb

        lights = self._dl_virtual_lights
        if lights:
            self._dl_selected_light_idx = max(0, min(len(lights) - 1, self._dl_selected_light_idx))
        else:
            self._dl_selected_light_idx = 0

        for idx, btn in enumerate(getattr(self, "dl_light_slot_btns", [])):
            has_slot = idx < len(lights)
            btn.setVisible(has_slot)
            if has_slot:
                btn.blockSignals(True)
                btn.setChecked(idx == self._dl_selected_light_idx)
                btn.setIcon(self._make_color_dot_icon(light_color_srgb(lights[idx])))
                btn.setIconSize(QSize(10, 10))
                btn.setText(f"{idx + 1}")
                btn.blockSignals(False)

        for b_look in getattr(self, "dl_look_buttons", []):
            pid = str(b_look.preset.get("id", ""))
            b_look.blockSignals(True)
            b_look.setChecked(pid == self._active_relight_preset_id and bool(lights))
            b_look.blockSignals(False)

        if hasattr(self, "btn_dl_add_light"):
            self.btn_dl_add_light.setEnabled(len(lights) < 3)
        if hasattr(self, "dl_light_editor_box"):
            self.dl_light_editor_box.setVisible(len(lights) > 0)

        self._refresh_depth_light_group_indicators()
        if not lights:
            return

        lt = lights[self._dl_selected_light_idx]
        prev = self._updating_ui
        self._updating_ui = True
        try:
            ltype = str(lt.get("type", "point")).lower()
            for t_key, b_t in getattr(self, "_dl_type_btns", {}).items():
                b_t.blockSignals(True)
                b_t.setChecked(t_key == ltype)
                b_t.blockSignals(False)

            if hasattr(self, "btn_dl_light_enabled"):
                self.btn_dl_light_enabled.blockSignals(True)
                self.btn_dl_light_enabled.setChecked(bool(lt.get("enabled", True)))
                self.btn_dl_light_enabled.setText("Light On" if lt.get("enabled", True) else "Light Off")
                self.btn_dl_light_enabled.blockSignals(False)

            gobo = str(lt.get("gobo", "none")).lower()
            idx_g = self.cmb_dl_light_gobo.findData(gobo)
            self.cmb_dl_light_gobo.setCurrentIndex(max(0, idx_g))

            self.dl_slider_light_kelvin.setValue(int(round(float(lt.get("kelvin", 4500)))))
            self.dl_slider_light_intensity.setValue(int(round(float(lt.get("intensity", 70.0)))))
            self.dl_slider_light_softness.setValue(int(round(float(lt.get("softness", 45.0)))))
            self.dl_slider_light_z.setValue(int(round(float(lt.get("z", 0.25)) * 100.0)))
            self.dl_slider_light_gloss.setValue(int(round(float(lt.get("gloss", 30.0)))))
            self.dl_slider_light_cone.setValue(int(round(float(lt.get("cone_angle", 40.0)))))
            if hasattr(self, "dl_spot_cone_container"):
                self.dl_spot_cone_container.setVisible(ltype == "spot")
            self.dl_slider_light_depth_from.setValue(int(round(float(lt.get("depth_from", 0.0)) * 100.0)))
            self.dl_slider_light_depth_to.setValue(int(round(float(lt.get("depth_to", 1.0)) * 100.0)))

            r_c, g_c, b_c = light_color_srgb(lt)
            self.btn_dl_light_color.setStyleSheet(
                f"background-color: rgb({r_c}, {g_c}, {b_c}); border: 1px solid #64748B; border-radius: 3px;"
            )
            self._refresh_depth_light_group_indicators()
        finally:
            self._updating_ui = prev

    def _apply_relight_look_preset(self, preset: dict[str, Any]) -> None:
        from src.imaging.creative_effects import make_default_virtual_light

        self._active_relight_preset_id = str(preset.get("id", "")) or None
        self._dl_virtual_lights = [make_default_virtual_light(**lt) for lt in preset.get("lights", [])][:3]
        self._dl_selected_light_idx = 0
        self._updating_ui = True
        try:
            if hasattr(self, "dl_slider_original_light"):
                dim_v = int(round(float(preset.get("ambient_dim", 0.0))))
                self.dl_slider_original_light.setValue(max(0, min(100, 100 - dim_v)))
        finally:
            self._updating_ui = False
        self._refresh_virtual_lights_ui()
        self.depth_virtual_lights_changed.emit(
            self._ambient_dim_value(),
            [dict(lt) for lt in self._dl_virtual_lights],
            True,
        )

    def _on_clear_relight_clicked(self) -> None:
        self._active_relight_preset_id = None
        self._dl_virtual_lights = []
        self._dl_selected_light_idx = 0
        self._updating_ui = True
        try:
            if hasattr(self, "dl_slider_original_light"):
                self.dl_slider_original_light.setValue(100)
            for a in ("dl_slider_near", "dl_slider_far", "dl_slider_warmth_near", "dl_slider_warmth_far"):
                if hasattr(self, a):
                    getattr(self, a).setValue(0)
        finally:
            self._updating_ui = False
        self._refresh_virtual_lights_ui()
        self.depth_relight_clear_requested.emit()

    def _on_add_virtual_light_clicked(self) -> None:
        if len(self._dl_virtual_lights) >= 3:
            return
        from src.imaging.creative_effects import make_default_virtual_light

        self._active_relight_preset_id = None
        offsets = [(0.25, 0.25), (0.75, 0.25), (0.50, 0.15)]
        lx, ly = offsets[len(self._dl_virtual_lights) % len(offsets)]
        self._dl_virtual_lights.append(make_default_virtual_light(x=lx, y=ly))
        self._dl_selected_light_idx = len(self._dl_virtual_lights) - 1
        self._refresh_virtual_lights_ui()
        self.depth_selected_light_changed.emit(self._dl_selected_light_idx)
        self.depth_virtual_lights_changed.emit(
            self._ambient_dim_value(),
            [dict(lt) for lt in self._dl_virtual_lights],
            True,
        )

    def _on_remove_virtual_light_clicked(self) -> None:
        if not self._dl_virtual_lights:
            return
        self._active_relight_preset_id = None
        idx = max(0, min(len(self._dl_virtual_lights) - 1, self._dl_selected_light_idx))
        self._dl_virtual_lights.pop(idx)
        self._dl_selected_light_idx = max(0, min(len(self._dl_virtual_lights) - 1, idx - 1))
        self._refresh_virtual_lights_ui()
        self.depth_virtual_lights_changed.emit(
            self._ambient_dim_value(),
            [dict(lt) for lt in self._dl_virtual_lights],
            True,
        )

    def _select_virtual_light_slot(self, slot_idx: int) -> None:
        if 0 <= slot_idx < len(self._dl_virtual_lights):
            self._dl_selected_light_idx = slot_idx
            self._refresh_virtual_lights_ui()
            self.depth_selected_light_changed.emit(slot_idx)

    def _set_active_light_type(self, ltype: str) -> None:
        if self._updating_ui or not self._dl_virtual_lights:
            return
        self._active_relight_preset_id = None
        self._dl_virtual_lights[self._dl_selected_light_idx]["type"] = ltype
        self._refresh_virtual_lights_ui()
        self.depth_virtual_lights_changed.emit(
            self._ambient_dim_value(),
            [dict(x) for x in self._dl_virtual_lights],
            True,
        )

    def _on_light_enabled_toggled(self, checked: bool) -> None:
        if self._updating_ui or not self._dl_virtual_lights:
            return
        self._dl_virtual_lights[self._dl_selected_light_idx]["enabled"] = bool(checked)
        self._refresh_virtual_lights_ui()
        self.depth_virtual_lights_changed.emit(
            self._ambient_dim_value(),
            [dict(x) for x in self._dl_virtual_lights],
            True,
        )

    def _on_clear_light_color_override(self) -> None:
        if not self._dl_virtual_lights:
            return
        self._active_relight_preset_id = None
        self._dl_virtual_lights[self._dl_selected_light_idx]["color"] = ""
        self._refresh_virtual_lights_ui()
        self.depth_virtual_lights_changed.emit(
            self._ambient_dim_value(),
            [dict(lt) for lt in self._dl_virtual_lights],
            True,
        )

    def _on_virtual_light_slider_changed(self) -> None:
        if self._updating_ui or not self._dl_virtual_lights:
            return
        self._active_relight_preset_id = None
        idx = self._dl_selected_light_idx
        lt = self._dl_virtual_lights[idx]
        old_kelvin = int(round(float(lt.get("kelvin", 4500))))
        new_kelvin = int(self.dl_slider_light_kelvin.value())
        lt["kelvin"] = new_kelvin
        if new_kelvin != old_kelvin and lt.get("color"):
            lt["color"] = ""
        lt["intensity"] = float(self.dl_slider_light_intensity.value())
        lt["softness"] = float(self.dl_slider_light_softness.value())
        lt["z"] = float(self.dl_slider_light_z.value()) / 100.0
        lt["gloss"] = float(self.dl_slider_light_gloss.value())
        lt["cone_angle"] = float(self.dl_slider_light_cone.value())
        lt["depth_from"] = float(self.dl_slider_light_depth_from.value()) / 100.0
        lt["depth_to"] = float(self.dl_slider_light_depth_to.value()) / 100.0

        from src.imaging.creative_effects import light_color_srgb

        rgb = light_color_srgb(lt)
        self.btn_dl_light_color.setStyleSheet(
            f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 1px solid #64748B; border-radius: 3px;"
        )
        if 0 <= idx < len(getattr(self, "dl_light_slot_btns", [])):
            self.dl_light_slot_btns[idx].setIcon(self._make_color_dot_icon(rgb))
        for b_look in getattr(self, "dl_look_buttons", []):
            b_look.blockSignals(True)
            b_look.setChecked(False)
            b_look.blockSignals(False)
        self._refresh_depth_light_group_indicators()
        self._emit_depth_virtual_lights()

    def _on_virtual_light_combo_or_color_changed(self) -> None:
        if self._updating_ui or not self._dl_virtual_lights:
            return
        self._active_relight_preset_id = None
        lt = self._dl_virtual_lights[self._dl_selected_light_idx]
        lt["gobo"] = str(self.cmb_dl_light_gobo.currentData() or "none")
        self._refresh_virtual_lights_ui()
        self.depth_virtual_lights_changed.emit(
            self._ambient_dim_value(),
            [dict(x) for x in self._dl_virtual_lights],
            True,
        )

    def _apply_swatch_to_active_light(self, hex_col: str) -> None:
        if not self._dl_virtual_lights:
            return
        self._active_relight_preset_id = None
        self._dl_virtual_lights[self._dl_selected_light_idx]["color"] = hex_col
        self._refresh_virtual_lights_ui()
        self.depth_virtual_lights_changed.emit(
            self._ambient_dim_value(),
            [dict(lt) for lt in self._dl_virtual_lights],
            True,
        )

    def _on_pick_virtual_light_color(self) -> None:
        if not self._dl_virtual_lights:
            return
        from src.imaging.creative_effects import light_color_srgb

        lt = self._dl_virtual_lights[self._dl_selected_light_idx]
        r_c, g_c, b_c = light_color_srgb(lt)
        chosen = QColorDialog.getColor(QColor(r_c, g_c, b_c), self, "Select Light Color")
        if chosen.isValid():
            self._apply_swatch_to_active_light(chosen.name())

    def sync_virtual_lights_from_canvas(self, lights: list[dict], selected_idx: int) -> None:
        self._updating_ui = True
        try:
            self._dl_virtual_lights = [dict(lt) for lt in (lights or [])[:3]]
            self._dl_selected_light_idx = max(0, int(selected_idx))
            self._refresh_virtual_lights_ui()
        finally:
            self._updating_ui = False

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
        if self._pending_dl_virtual_lights:
            self._pending_dl_virtual_lights = False
            self.depth_virtual_lights_changed.emit(
                self._ambient_dim_value(),
                [dict(lt) for lt in self._dl_virtual_lights],
                False,
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
        self._refresh_depth_light_group_indicators()
        self.fine_tune_slider_released.emit(action_name)

    def set_denoise_analyzed_value(self, val: float) -> None:
        """Sets the noise reduction slider to the calculated noise analysis value."""
        self.ft_slider_denoise.setValue(int(round(val)))

    def sync_recipe(self, recipe: Any) -> None:
        """Synchronizes fine-tune and depth & light sliders with current PhotoAdjustmentRecipe without emitting signals."""
        self._updating_ui = True
        try:
            if recipe is None:
                b = se = c = s = h = w_val = bl_val = cl = st_val = t = ti = sat = dh = str_deg = sh = sh_mask = dn = vig = og_val = gh_val = 0
                sh_rad = 25
                dn_det = 50
                d_blur = d_focal = d_glow = d_bshift = d_bfeather = d_near = d_far = d_wn = d_wf = d_ambient_dim = d_haze = d_hw = 0
                d_vlights: list[dict[str, Any]] = []
            else:
                b = int(round(getattr(recipe, "brightness", 0.0)))
                se = int(round(getattr(recipe, "smart_enhance", 0.0)))
                c = int(round(getattr(recipe, "contrast", 0.0)))
                s = int(round(getattr(recipe, "shadows", 0.0)))
                h = int(round(getattr(recipe, "highlights", 0.0)))
                w_val = int(round(getattr(recipe, "whites", 0.0)))
                bl_val = int(round(getattr(recipe, "blacks", 0.0)))
                cl = int(round(getattr(recipe, "clarity", 0.0)))
                st_val = int(round(getattr(recipe, "structure", 0.0)))
                t = int(round(getattr(recipe, "temperature", 0.0)))
                ti = int(round(getattr(recipe, "tint", 0.0)))
                sat = int(round(getattr(recipe, "saturation", 0.0)))
                dh = int(round(getattr(recipe, "dehaze", 0.0)))
                str_deg = int(round(getattr(recipe, "straighten_deg", 0.0)))
                sh = int(round(getattr(recipe, "sharpen", 0.0)))
                sh_rad = int(round(getattr(recipe, "sharpen_radius", 25.0)))
                sh_mask = int(round(getattr(recipe, "sharpen_masking", 0.0)))
                dn = int(round(getattr(recipe, "denoise", 0.0)))
                dn_det = int(round(getattr(recipe, "denoise_detail", 50.0)))
                dn_c = int(round(getattr(recipe, "denoise_chroma", 0.0)))
                gr_amt = int(round(getattr(recipe, "grain_amount", 0.0)))
                gr_sz = int(round(getattr(recipe, "grain_size", 25.0)))
                vig = int(round(getattr(recipe, "vignette", 0.0)))
                vig_f = int(round(getattr(recipe, "vignette_feather", 50.0)))
                og_val = int(round(getattr(recipe, "orton_glow", 0.0)))
                gh_val = int(round(getattr(recipe, "golden_hour", 0.0)))

                d_blur = int(round(getattr(recipe, "depth_blur", 0.0)))
                d_focal = int(round(getattr(recipe, "focal_plane", 0.0)))
                d_glow = int(round(getattr(recipe, "bokeh_glow", 0.0)))
                d_bshift = int(round(getattr(recipe, "boundary_shift", 0.0)))
                d_bfeather = int(round(getattr(recipe, "boundary_feather", 0.0)))
                d_near = int(round(getattr(recipe, "relight_near", 0.0)))
                d_far = int(round(getattr(recipe, "relight_far", 0.0)))
                d_wn = int(round(getattr(recipe, "warmth_near", 0.0)))
                d_wf = int(round(getattr(recipe, "warmth_far", 0.0)))
                d_ambient_dim = int(round(getattr(recipe, "relight_ambient_dim", 0.0)))
                d_vlights = [dict(lt) for lt in (getattr(recipe, "virtual_lights", []) or [])[:3]]
                d_haze = int(round(getattr(recipe, "depth_haze", 0.0)))
                d_hw = int(round(getattr(recipe, "depth_haze_warmth", 0.0)))

            if hasattr(self, "ft_slider_brightness"):
                if hasattr(self, "ft_slider_smart_enhance"):
                    self.ft_slider_smart_enhance.setValue(se)
                self.ft_slider_brightness.setValue(b)
                self.ft_slider_contrast.setValue(c)
                self.ft_slider_shadows.setValue(s)
                self.ft_slider_highlights.setValue(h)
                self.ft_slider_clarity.setValue(cl)
                if hasattr(self, "ft_slider_structure"):
                    self.ft_slider_structure.setValue(st_val)
                self.ft_slider_temp.setValue(t)
                self.ft_slider_tint.setValue(ti)
                self.ft_slider_sat.setValue(sat)
                self.ft_slider_dehaze.setValue(dh)
                self.ft_slider_straighten.setValue(str_deg)
                self.ft_slider_sharpen.setValue(sh)
                self.ft_slider_denoise.setValue(dn)
                self.ft_slider_vignette.setValue(vig)
                if hasattr(self, "ft_slider_whites"):
                    self.ft_slider_whites.setValue(w_val)
                    self.ft_slider_blacks.setValue(bl_val)
                if hasattr(self, "ft_slider_vignette_feather"):
                    self.ft_slider_vignette_feather.setValue(50 if recipe is None else vig_f)
                if hasattr(self, "ft_slider_orton_glow"):
                    self.ft_slider_orton_glow.setValue(og_val)
                    self.ft_slider_golden_hour.setValue(gh_val)
                if hasattr(self, "ft_slider_sharpen_radius"):
                    self.ft_slider_sharpen_radius.setValue(sh_rad)
                if hasattr(self, "ft_slider_sharpen_masking"):
                    self.ft_slider_sharpen_masking.setValue(sh_mask)
                if hasattr(self, "ft_slider_denoise_detail"):
                    self.ft_slider_denoise_detail.setValue(dn_det)
                if hasattr(self, "ft_slider_denoise_chroma"):
                    self.ft_slider_denoise_chroma.setValue(0 if recipe is None else dn_c)
                    self.ft_slider_grain_amount.setValue(0 if recipe is None else gr_amt)
                    self.ft_slider_grain_size.setValue(25 if recipe is None else gr_sz)
                raw_regions = {} if recipe is None else getattr(recipe, "region_edits", {}) or {}
                self._region_edits = {
                    str(rk): {
                        fk: (list(fv) if str(fk).startswith("curve_") or fk == "mask_strokes" else fv)
                        for fk, fv in rd.items()
                    }
                    for rk, rd in raw_regions.items()
                }
                self._global_region_mirror = {
                    "brightness": b,
                    "contrast": c,
                    "shadows": s,
                    "highlights": h,
                    "whites": w_val,
                    "blacks": bl_val,
                    "clarity": cl,
                    "structure": st_val,
                    "temperature": t,
                    "tint": ti,
                    "saturation": sat,
                    "dehaze": dh,
                    "curve_rgb": [(0.0, 0.0), (1.0, 1.0)]
                    if recipe is None
                    else [tuple(p) for p in getattr(recipe, "curve_rgb", [(0.0, 0.0), (1.0, 1.0)])],
                    "curve_r": [(0.0, 0.0), (1.0, 1.0)]
                    if recipe is None
                    else [tuple(p) for p in getattr(recipe, "curve_r", [(0.0, 0.0), (1.0, 1.0)])],
                    "curve_g": [(0.0, 0.0), (1.0, 1.0)]
                    if recipe is None
                    else [tuple(p) for p in getattr(recipe, "curve_g", [(0.0, 0.0), (1.0, 1.0)])],
                    "curve_b": [(0.0, 0.0), (1.0, 1.0)]
                    if recipe is None
                    else [tuple(p) for p in getattr(recipe, "curve_b", [(0.0, 0.0), (1.0, 1.0)])],
                }
                cur_scope = getattr(self, "_active_region_scope", "whole")
                if cur_scope != "whole":
                    self._load_region_dict_into_sliders(self._region_edits.get(cur_scope, {}))
                elif hasattr(self, "tone_curve_widget"):
                    if recipe is None:
                        self.tone_curve_widget.set_curves({})
                    else:
                        self.tone_curve_widget.set_curves(
                            {
                                "rgb": getattr(recipe, "curve_rgb", [(0.0, 0.0), (1.0, 1.0)]),
                                "r": getattr(recipe, "curve_r", [(0.0, 0.0), (1.0, 1.0)]),
                                "g": getattr(recipe, "curve_g", [(0.0, 0.0), (1.0, 1.0)]),
                                "b": getattr(recipe, "curve_b", [(0.0, 0.0), (1.0, 1.0)]),
                            }
                        )
                if hasattr(self, "_hsl_sliders"):
                    src_h = {} if recipe is None else getattr(recipe, "hsl_hue", {})
                    src_s = {} if recipe is None else getattr(recipe, "hsl_sat", {})
                    src_l = {} if recipe is None else getattr(recipe, "hsl_lum", {})
                    for k in self._hsl_hue_vals:
                        self._hsl_hue_vals[k] = int(round(src_h.get(k, 0.0)))
                        self._hsl_sat_vals[k] = int(round(src_s.get(k, 0.0)))
                        self._hsl_lum_vals[k] = int(round(src_l.get(k, 0.0)))
                    active_store = {
                        "hue": self._hsl_hue_vals,
                        "sat": self._hsl_sat_vals,
                        "lum": self._hsl_lum_vals,
                    }[self._hsl_mode]
                    for k, sl in self._hsl_sliders.items():
                        sl.setValue(active_store.get(k, 0))
                self._update_drawer_indicators()
                self._refresh_region_indicators()
                self._sync_wb_preset_combo()
                if hasattr(self, "cmb_look_profile"):
                    from pathlib import Path
                    eff_val = "none" if recipe is None else str(getattr(recipe, "effect", "none") or "none")
                    eff_int = 100 if recipe is None else int(round(getattr(recipe, "effect_intensity", 100.0)))
                    self.cmb_look_profile.blockSignals(True)
                    idx = self.cmb_look_profile.findData(eff_val)
                    if idx < 0 and eff_val.startswith("cube:"):
                        self.cmb_look_profile.addItem(f"LUT: {Path(eff_val[5:]).stem}", eff_val)
                        idx = self.cmb_look_profile.count() - 1
                    self.cmb_look_profile.setCurrentIndex(max(0, idx))
                    self.cmb_look_profile.blockSignals(False)
                    if hasattr(self, "ft_slider_look_amount"):
                        self.ft_slider_look_amount.setValue(eff_int)
                        self.ft_slider_look_amount.setEnabled(eff_val not in ("none", "as_shot", ""))

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
                if hasattr(self, "dl_slider_original_light"):
                    self.dl_slider_original_light.setValue(max(0, min(100, 100 - d_ambient_dim)))
                    if not d_vlights and d_ambient_dim == 0:
                        self._active_relight_preset_id = None
                    self._dl_virtual_lights = d_vlights
                    self._refresh_virtual_lights_ui()
                self.dl_slider_haze.setValue(d_haze)
                self.dl_slider_haze_warmth.setValue(d_hw)
                self._refresh_depth_light_group_indicators()
                if d_blur > 0 or d_focal > 0 or d_glow > 0:
                    self._dl_grp_bokeh.set_expanded(True)
                if d_haze > 0 or d_hw != 0:
                    self._dl_grp_haze.set_expanded(True)
                if d_near != 0 or d_far != 0 or d_wn != 0 or d_wf != 0:
                    self._dl_grp_near_far.set_expanded(True)
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
                self._set_folder_display("")
                self.date_edit.setEnabled(False)
                self.time_edit.setEnabled(False)
                self.stack_date_time.finish_edit("-")
                self.stack_date_time.set_editable(False)
                self.btn_date_taken.setText("-")
                self.lbl_size.setText("-")
                self.lbl_size.setToolTip("")
                self.star_rating_widget.set_rating(0)
                self._update_flag_button(0, False)
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
                if hasattr(self, "ocr_section_widget"):
                    self.txt_ocr_preview.clear()
                    self.ocr_section_widget.setVisible(False)
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
                    is_boxless_val = bool(
                        count == 1
                        and st == 1
                        and (float(p.get("rect_w") or 0.0) <= 0.005 or float(p.get("rect_h") or 0.0) <= 0.005)
                    )
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
                self._set_folder_display(
                    str(summary.get("folder_path") or ""),
                    str(summary.get("folder_name") or ""),
                    multiple=False,
                )

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
                fl_val = -1 if int(summary.get("is_rejected") or 0) else int(summary.get("flag") or 0)
                fav_val = bool(summary.get("is_favourite") or 0)
                self._update_flag_button(fl_val, fav_val)

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

                cap_val = str(summary.get("caption") or "")
                self.txt_caption.setText(cap_val)
                self._last_caption_text = cap_val.strip()
                loc_name = str(summary.get("location_name") or "").strip()
                if not loc_name and lat is not None and lon is not None:
                    try:
                        rev = reverse_geocode(float(lat), float(lon))
                        if rev and rev.get("formatted_address"):
                            loc_name = rev["formatted_address"]
                    except Exception:
                        pass
                self.txt_geotag.setText(loc_name)
                self._last_valid_geotag = loc_name
                if hasattr(self, "ocr_section_widget"):
                    ocr_val = str(summary.get("ocr_text") or "").strip()
                    self.txt_ocr_preview.setPlainText(ocr_val)
                    self.ocr_section_widget.setVisible(bool(ocr_val))
            else:
                lead = summary.get("lead", {})
                self.lbl_selection_header.setText(f"{count} items selected")
                self.txt_filename.setText(f"Multiple files ({count})")
                self.txt_filename.setEnabled(False)
                self.stack_filename.set_display_text(f"Multiple files ({count})")
                self.stack_filename.finish_edit()
                self.stack_filename.set_editable(False)
                mult_folders = bool(summary.get("multiple_folders", False))
                if mult_folders:
                    self._set_folder_display("", multiple=True)
                else:
                    self._set_folder_display(
                        str(summary.get("folder_path") or lead.get("folder_path") or ""),
                        str(summary.get("folder_name") or lead.get("folder_name") or ""),
                        multiple=False,
                    )

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
                common_fav = summary.get("common_favourite")
                self._update_flag_button(
                    int(common_flag) if common_flag is not None else 0,
                    bool(common_fav) if common_fav is not None else False,
                )

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

                cap_val = str(lead.get("caption") or "")
                self.txt_caption.setText(cap_val)
                self._last_caption_text = cap_val.strip()
                common_loc = str(summary.get("common_location_name") or lead.get("location_name") or "").strip()
                if not common_loc and lead_lat is not None and lead_lon is not None:
                    try:
                        rev = reverse_geocode(float(lead_lat), float(lead_lon))
                        if rev and rev.get("formatted_address"):
                            common_loc = rev["formatted_address"]
                    except Exception:
                        pass
                self.txt_geotag.setText(common_loc)
                self._last_valid_geotag = common_loc
                if hasattr(self, "ocr_section_widget"):
                    self.txt_ocr_preview.clear()
                    self.ocr_section_widget.setVisible(False)
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

    def set_favourites_enabled(self, enabled: bool) -> None:
        self._favourites_enabled = bool(enabled)
        if hasattr(self, "btn_info_fav"):
            self.btn_info_fav.setVisible(self._favourites_enabled)

    def _make_info_cull_icon(self, kind: str, active: bool) -> QIcon:
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QLinearGradient, QPainterPath, QPolygonF

        pm = QPixmap(30, 30)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        muted = QColor("#94A3B8")
        if kind == "pick":
            banner = QPolygonF([QPointF(9.0, 5.5), QPointF(24.5, 10.5), QPointF(9.0, 16.0)])
            if active:
                grad = QLinearGradient(9.0, 5.5, 9.0, 16.5)
                grad.setColorAt(0.0, QColor("#F87171"))
                grad.setColorAt(0.5, QColor("#DC2626"))
                grad.setColorAt(1.0, QColor("#991B1B"))
                p.setPen(QPen(QColor("#7F1D1D"), 1.1))
                p.setBrush(grad)
                p.drawPolygon(banner)
                p.setPen(QPen(QColor("#CBD5E1"), 2.2, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(8, 5, 8, 25)
            else:
                p.setPen(QPen(muted, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                p.setBrush(Qt.NoBrush)
                p.drawLine(8, 5, 8, 25)
                p.drawPolygon(banner)
        elif kind == "reject":
            p.setPen(QPen(QColor("#CBD5E1") if active else muted, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(Qt.NoBrush)
            p.drawLine(8, 5, 8, 25)
            banner = QPolygonF([QPointF(8.0, 5.5), QPointF(23.5, 10.5), QPointF(8.0, 16.0)])
            p.drawPolygon(banner)
            p.setPen(QPen(QColor("#EF4444") if active else QColor("#94A3B8"), 2.6 if active else 2.0, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(5, 24, 25, 5)
        elif kind == "fav":
            from PySide6.QtCore import QRectF
            cx, cy, r = 15.0, 15.0, 7.8
            hpath = QPainterPath()
            hpath.moveTo(cx, cy + 1.0 * r)
            hpath.cubicTo(cx - 1.15 * r, cy + 0.30 * r, cx - 1.20 * r, cy - 0.72 * r, cx - 0.64 * r, cy - 0.96 * r)
            hpath.cubicTo(cx - 0.28 * r, cy - 1.12 * r, cx - 0.06 * r, cy - 0.76 * r, cx, cy - 0.46 * r)
            hpath.cubicTo(cx + 0.06 * r, cy - 0.76 * r, cx + 0.28 * r, cy - 1.12 * r, cx + 0.64 * r, cy - 0.96 * r)
            hpath.cubicTo(cx + 1.20 * r, cy - 0.72 * r, cx + 1.15 * r, cy + 0.30 * r, cx, cy + 1.0 * r)
            if active:
                hgrad = QLinearGradient(cx, cy - r, cx, cy + r)
                hgrad.setColorAt(0.0, QColor("#FB7185"))
                hgrad.setColorAt(1.0, QColor("#E11D48"))
                p.setPen(Qt.NoPen)
                p.setBrush(hgrad)
            else:
                p.setPen(QPen(muted, 2.0))
                p.setBrush(Qt.NoBrush)
            p.drawPath(hpath)
        p.end()
        pm.setDevicePixelRatio(2.0)
        return QIcon(pm)

    def _update_flag_button(self, flag: int, is_favourite: bool | None = None) -> None:
        self._current_flag = int(flag) if int(flag) in (-1, 0, 1) else (1 if flag else 0)
        if is_favourite is not None:
            self._current_favourite = bool(is_favourite)
        if self._current_flag == -1:
            self.btn_flag.setIcon(self._make_info_cull_icon("reject", True))
            self.btn_flag.setToolTip("Rejected — click to clear flag (P / X / U)")
            flag_bg = "rgba(239, 68, 68, 38)"
        elif self._current_flag == 1:
            self.btn_flag.setIcon(self._make_info_cull_icon("pick", True))
            self.btn_flag.setToolTip("Picked — click to mark Rejected (P / X / U)")
            flag_bg = "rgba(220, 38, 38, 42)"
        else:
            self.btn_flag.setIcon(self._make_info_cull_icon("pick", False))
            self.btn_flag.setToolTip("Unflagged — click to mark Picked (P / X / U)")
            flag_bg = "transparent"
        if hasattr(self, "btn_info_fav"):
            self.btn_info_fav.setVisible(bool(getattr(self, "_favourites_enabled", True)))
            self.btn_info_fav.setIcon(self._make_info_cull_icon("fav", self._current_favourite))
            self.btn_info_fav.setToolTip("Unfavourite (F)" if self._current_favourite else "Favourite (F)")

        for btn in (self.btn_flag, getattr(self, "btn_info_fav", None)):
            if btn is not None:
                btn.setStyleSheet(
                    "QPushButton { background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 0px; } "
                    "QPushButton:hover { background: rgba(148, 163, 184, 45); border: 1px solid rgba(148, 163, 184, 90); }"
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

    def _set_folder_display(self, folder_path: str, folder_name: str = "", *, multiple: bool = False) -> None:
        if not hasattr(self, "lbl_folder"):
            return
        clean_path = (folder_path or "").strip()
        if multiple:
            self._current_folder_path = ""
            self._folder_is_multiple = True
            self.lbl_folder.setText("Multiple folders")
            self.lbl_folder.setToolTip("Selected items are in multiple folders")
            self.lbl_folder.set_clickable(False)
        elif clean_path:
            display = (folder_name or "").strip() or Path(clean_path).name or clean_path
            self._current_folder_path = clean_path
            self._folder_is_multiple = False
            self.lbl_folder.setText(display)
            self.lbl_folder.setToolTip(f"{clean_path}\nClick to open folder in Explorer")
            self.lbl_folder.set_clickable(True)
        else:
            self._current_folder_path = ""
            self._folder_is_multiple = False
            self.lbl_folder.setText("-")
            self.lbl_folder.setToolTip("")
            self.lbl_folder.set_clickable(False)
        self._update_folder_link_style()

    def _update_folder_link_style(self) -> None:
        if not hasattr(self, "lbl_folder"):
            return
        is_dark = getattr(self, "_is_dark", False)
        if self.lbl_folder.is_clickable():
            col = "#60A5FA" if is_dark else "#0066CC"
            hover_col = "#93C5FD" if is_dark else "#004C99"
            self.lbl_folder.setStyleSheet(
                f"""
                QLabel#InfoFolderLinkLabel {{
                    color: {col};
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                    background: transparent;
                    border: none;
                }}
                QLabel#InfoFolderLinkLabel:hover {{
                    color: {hover_col};
                    text-decoration: underline;
                }}
                """
            )
        else:
            if getattr(self, "_folder_is_multiple", False):
                col = "#94A3B8" if is_dark else "#64748B"
            else:
                col = "#E2E8F0" if is_dark else "#1E293B"
            self.lbl_folder.setStyleSheet(
                f"""
                QLabel#InfoFolderLinkLabel {{
                    color: {col};
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                    background: transparent;
                    border: none;
                }}
                """
            )

    def _on_folder_label_clicked(self) -> None:
        if getattr(self, "_current_folder_path", ""):
            self.folder_open_requested.emit(self._current_folder_path)

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
        new_flag = 1 if self._current_flag == 0 else (-1 if self._current_flag == 1 else 0)
        self._update_flag_button(new_flag)
        self.flag_edited.emit(new_flag)

    def _on_favourite_button_clicked(self) -> None:
        if self._updating_ui or not self._selected_ids or not getattr(self, "_favourites_enabled", True):
            return
        new_fav = not self._current_favourite
        self._update_flag_button(self._current_flag, new_fav)
        self.favourite_edited.emit(new_fav)

    def _on_caption_return_pressed(self) -> None:
        self._on_caption_editing_finished()
        QTimer.singleShot(0, self.txt_caption.clearFocus)

    def keyPressEvent(self, event: Any) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)

    def _on_caption_editing_finished(self) -> None:
        if self._updating_ui or not self._selected_ids:
            return
        new_cap = self.txt_caption.text().strip()
        if new_cap == getattr(self, "_last_caption_text", ""):
            return
        self._last_caption_text = new_cap
        self.caption_committed.emit(new_cap)

    def commit_caption(self, caption_text: str | None = None) -> None:
        if caption_text is not None:
            self.txt_caption.setText(caption_text)
        new_cap = self.txt_caption.text().strip()
        self._last_caption_text = new_cap
        self.caption_committed.emit(new_cap)

    def _on_geotag_committed(self) -> None:
        if self._updating_ui or not self._selected_ids:
            return
        if getattr(self, "_ignore_next_geotag_return", False):
            self._ignore_next_geotag_return = False
            QTimer.singleShot(0, self.txt_geotag.clearFocus)
            return

        raw = self.txt_geotag.text().strip()
        popup = self.txt_geotag._completer.popup() if hasattr(self.txt_geotag, "_completer") else None
        if popup is not None and popup.isVisible():
            popup.hide()

        if not raw:
            if getattr(self, "_last_valid_geotag", ""):
                self._last_valid_geotag = ""
                self._on_geotag_cleared()
            QTimer.singleShot(0, self.txt_geotag.clearFocus)
            return

        if raw == getattr(self, "_last_valid_geotag", ""):
            QTimer.singleShot(0, self.txt_geotag.clearFocus)
            return

        # Auto-select the first valid item from the autocomplete list if available
        places = getattr(self.txt_geotag, "_current_places", None) or []
        chosen = next(
            (p for p in places if p.get("formatted_address", "").lower() == raw.lower()),
            None,
        )
        if chosen is None and places:
            chosen = next(
                (p for p in places if raw.lower() in p.get("formatted_address", "").lower()),
                places[0],
            )
        if chosen is not None and chosen.get("lat") is not None and chosen.get("lon") is not None:
            fmt = str(chosen["formatted_address"])
            lat, lon = float(chosen["lat"]), float(chosen["lon"])
            self.txt_geotag.setText(fmt)
            self._last_valid_geotag = fmt
            self.lbl_lat.setText(f"{lat:.5f}")
            self.lbl_lon.setText(f"{lon:.5f}")
            if self.lbl_geotag_status is not None:
                self.lbl_geotag_status.setVisible(False)
            self.geotag_edited.emit(fmt)
            QTimer.singleShot(0, self.txt_geotag.clearFocus)
            return

        lat, lon, fmt = resolve_location_query(
            raw, allow_network=False, db_engine=getattr(self.txt_geotag, "_db_engine", None)
        )
        if lat is not None and lon is not None and fmt:
            self.txt_geotag.setText(fmt)
            self._last_valid_geotag = fmt
            self.lbl_lat.setText(f"{float(lat):.5f}")
            self.lbl_lon.setText(f"{float(lon):.5f}")
            if self.lbl_geotag_status is not None:
                self.lbl_geotag_status.setVisible(False)
            self.geotag_edited.emit(fmt)
            QTimer.singleShot(0, self.txt_geotag.clearFocus)
            return

        # No offline match: let MainWindow try online lookup asynchronously (and reject if invalid)
        if self.lbl_geotag_status is not None:
            self.lbl_geotag_status.setText("Searching...")
            self.lbl_geotag_status.setVisible(True)
        self.geotag_edited.emit(raw)
        QTimer.singleShot(0, self.txt_geotag.clearFocus)

    def revert_invalid_geotag(self, invalid_query: str = "") -> None:
        """Restores the previous valid geotag and shows a non-blocking invalid-location indicator."""
        self.txt_geotag.setText(getattr(self, "_last_valid_geotag", ""))
        if self.lbl_geotag_status is not None:
            self.lbl_geotag_status.setText("Location not found")
            self.lbl_geotag_status.setVisible(True)
            QTimer.singleShot(
                3000,
                lambda: self.lbl_geotag_status.setVisible(False)
                if self.lbl_geotag_status is not None and self.lbl_geotag_status.text() == "Location not found"
                else None,
            )

    def _on_place_selected(self, place_name: str, lat: float, lon: float) -> None:
        if self._updating_ui or not self._selected_ids:
            return
        self._ignore_next_geotag_return = True
        QTimer.singleShot(0, lambda: setattr(self, "_ignore_next_geotag_return", False))
        self._last_valid_geotag = place_name
        self.txt_geotag.setCursorPosition(0)
        self.lbl_lat.setText(f"{lat:.5f}")
        self.lbl_lon.setText(f"{lon:.5f}")
        self.geotag_edited.emit(place_name)
        QTimer.singleShot(0, self.txt_geotag.clearFocus)

    def _on_geotag_cleared(self) -> None:
        if self._updating_ui or not self._selected_ids:
            return
        self._last_valid_geotag = ""
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

    def apply_theme_mode(self, theme_name: str) -> None:
        self._theme_name = theme_name or "wlpg_aero"
        self.set_dark_mode(self._theme_name == "dark")

    def set_dark_mode(self, is_dark: bool) -> None:
        """Applies Dark Studio or standard light styling to the Right Tag & Caption / Fine-Tune Dock."""
        self._is_dark = bool(is_dark)
        if not hasattr(self, "_theme_name") or (is_dark and self._theme_name != "dark") or (not is_dark and self._theme_name == "dark"):
            self._theme_name = "dark" if is_dark else "wlpg_aero"
        self._apply_stylesheet()
        if hasattr(self, "lbl_custom_inspector_title"):
            self.lbl_custom_inspector_title.setStyleSheet(
                f"color: {'#93C5FD' if is_dark else '#0F4C81'}; background: transparent;"
            )
        if hasattr(self, "page_custom_inspector"):
            bg = "#1B1E24" if is_dark else "#FFFFFF"
            self.page_custom_inspector.setStyleSheet(f"QWidget#CustomInspectorPage {{ background-color: {bg}; }}")
        if hasattr(self, "custom_inspector_scroll"):
            hosted = self.custom_inspector_scroll.widget()
            if hosted is not None and hasattr(hosted, "apply_theme_mode"):
                hosted.apply_theme_mode(self._theme_name)

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
                f"color: {'#94A3B8' if is_dark else '#475569'}; font-family: 'Segoe UI'; font-size: 9pt; padding-top: 2px;"
            )
        if hasattr(self, "lbl_selection_header"):
            self.lbl_selection_header.setStyleSheet(
                f"color: {'#94A3B8' if is_dark else '#64748B'}; font-family: 'Segoe UI'; font-size: 9pt;"
            )

        # Update form labels inside info_form_frame regardless of current visibility
        if hasattr(self, "info_form_frame") and hasattr(self, "_info_form"):
            form = self._info_form
            for row in range(form.rowCount()):
                lbl_item = form.itemAt(row, QFormLayout.LabelRole)
                if lbl_item and isinstance(lbl_item.widget(), QLabel):
                    lbl_item.widget().setStyleSheet(
                        f"color: {'#94A3B8' if is_dark else '#475569'}; font-family: 'Segoe UI'; font-size: 9pt;"
                    )
                fld_item = form.itemAt(row, QFormLayout.FieldRole)
                if fld_item and isinstance(fld_item.widget(), QLabel):
                    fld_widget = fld_item.widget()
                    if hasattr(self, "lbl_folder") and fld_widget is self.lbl_folder:
                        continue
                    fld_widget.setStyleSheet(
                        f"color: {'#E2E8F0' if is_dark else '#1E293B'}; font-family: 'Segoe UI'; font-size: 9pt;"
                    )
        if hasattr(self, "lbl_folder"):
            self._update_folder_link_style()

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
                self.page_info.setStyleSheet("QWidget#InfoPage { background-color: #1B1E24; }")
            if hasattr(self, "info_splitter"):
                self.info_splitter.setStyleSheet("QSplitter#InfoSplitter { background-color: #1B1E24; }")
                for i in range(self.info_splitter.count()):
                    w = self.info_splitter.widget(i)
                    if isinstance(w, QScrollArea):
                        w.setStyleSheet("QScrollArea { background-color: #1B1E24; border: none; }")
                        if w.widget():
                            w.widget().setStyleSheet("")
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
                QFrame#FineTuneGroup QLabel {
                    color: #E2E8F0;
                    background: transparent;
                }
                QPushButton#FineTuneGroupTitle {
                    color: #60A5FA;
                    font-family: 'Segoe UI';
                    font-size: 11.5pt;
                    font-weight: 500;
                    border: none;
                    background: transparent;
                    text-align: left;
                    padding: 0 0 2px 0;
                }
                QLabel#SliderHeaderLabel {
                    color: #E2E8F0;
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                    border: none;
                    background: transparent;
                }
                QLabel#val_lbl {
                    color: #94A3B8;
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                    font-weight: 600;
                    border: none;
                    background: transparent;
                    padding-right: 2px;
                }
                QLabel#SliderHeaderLabel:disabled,
                QLabel#val_lbl:disabled {
                    color: #64748B;
                }
                QPushButton#FineTuneActionBtn,
                QPushButton#FineTuneResetBtn,
                QPushButton#DepthCutoutBtn {
                    background: #262B34;
                    border: 1px solid #3B4252;
                    border-radius: 3px;
                    color: #E2E8F0;
                    font-family: 'Segoe UI';
                    font-size: 8pt;
                    font-weight: 600;
                    padding: 2px 8px;
                }
                QPushButton#DepthCutoutBtn {
                    padding: 3px 6px;
                    text-align: left;
                }
                QPushButton#FineTuneActionBtn:hover,
                QPushButton#DepthCutoutBtn:hover {
                    background: #2E3440;
                    border-color: #60A5FA;
                }
                QPushButton#FineTuneResetBtn:hover {
                    background: #3B1F24;
                    border-color: #F87171;
                    color: #FCA5A5;
                }
                QPushButton#FineTuneActionBtn:pressed,
                QPushButton#DepthCutoutBtn:pressed {
                    background: #1E3A5F;
                    border-color: #3B82F6;
                }
                QPushButton#FineTuneActionBtn:checked {
                    background: #2563EB;
                    border-color: #60A5FA;
                    color: #FFFFFF;
                }
                QFrame#RegionScopeBar {
                    background: #1C2129;
                    border: 1px solid #303846;
                    border-radius: 5px;
                }
                QPushButton#FineTuneSegmentBtn {
                    background: #222730;
                    border: 1px solid #3B4252;
                    border-right: none;
                    color: #CBD5E1;
                    font-family: 'Segoe UI';
                    font-size: 8pt;
                    padding: 2px 4px;
                }
                QPushButton#FineTuneSegmentBtn[seg="first"] {
                    border-top-left-radius: 4px;
                    border-bottom-left-radius: 4px;
                }
                QPushButton#FineTuneSegmentBtn[seg="last"] {
                    border-right: 1px solid #3B4252;
                    border-top-right-radius: 4px;
                    border-bottom-right-radius: 4px;
                }
                QPushButton#FineTuneSegmentBtn:hover {
                    background: #2E3440;
                    color: #F8FAFC;
                }
                QPushButton#FineTuneSegmentBtn:checked {
                    background: #2563EB;
                    color: #FFFFFF;
                    border-color: #3B82F6;
                    font-weight: 600;
                }
                QPushButton#FineTuneMaskToggleBtn {
                    background: #222730;
                    border: 1px solid #3B4252;
                    border-radius: 3px;
                    color: #CBD5E1;
                    font-family: 'Segoe UI';
                    font-size: 7.5pt;
                    padding: 1px 7px;
                }
                QPushButton#FineTuneMaskToggleBtn:hover {
                    background: #2E3440;
                    border-color: #60A5FA;
                    color: #F8FAFC;
                }
                QPushButton#FineTuneMaskToggleBtn:checked {
                    background: #DC2626;
                    border-color: #EF4444;
                    color: #FFFFFF;
                    font-weight: 600;
                }
                QPushButton#FineTuneMaskToggleBtn:disabled {
                    background: transparent;
                    border-color: #2A303C;
                    color: #64748B;
                }
                QPushButton#FineTuneSubBtn {
                    background-color: #222730;
                    border: 1px solid #3B4252;
                    border-radius: 3px;
                    color: #CBD5E1;
                    font-family: 'Segoe UI';
                    font-size: 7.5pt;
                    padding: 1px 6px;
                }
                QPushButton#FineTuneSubBtn {
                    background-color: #252B36;
                    border: 1px solid #3B4252;
                    border-radius: 3px;
                    color: #E2E8F0;
                    padding: 1px 6px;
                    font-size: 11px;
                }
                QPushButton#FineTuneSubBtn:hover {
                    background-color: #2E3440;
                    border-color: #60A5FA;
                    color: #F8FAFC;
                }
                QPushButton#FineTuneSubBtn:checked {
                    background-color: #2563EB;
                    border: 1px solid #3B82F6;
                    color: #FFFFFF;
                    font-weight: 600;
                }
                QPushButton#FineTunePillBtn {
                    padding: 2px 10px;
                    border: 1px solid #3B4252;
                    background: #222730;
                    color: #CBD5E1;
                    border-radius: 3px;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                }
                QPushButton#FineTunePillBtn:hover {
                    background: #2E3440;
                    border-color: #60A5FA;
                    color: #F8FAFC;
                }
                QPushButton#FineTunePillBtn:checked {
                    background: #2563EB;
                    color: #FFFFFF;
                    border-color: #1D4ED8;
                    font-weight: 600;
                }
                QToolButton#OcrToggleBtn {
                    border: none;
                    background: transparent;
                    color: #60A5FA;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    font-weight: 600;
                    padding: 0;
                }
                QToolButton#OcrCopyBtn {
                    font-size: 8pt;
                    padding: 1px 16px 1px 8px;
                    border: 1px solid #3B4252;
                    border-radius: 3px;
                    background: #262B34;
                    color: #E2E8F0;
                }
                QToolButton#OcrCopyBtn:hover {
                    background: #2E3440;
                    border-color: #60A5FA;
                }
                QToolButton#OcrCopyBtn::menu-button {
                    border-left: 1px solid #3B4252;
                    width: 14px;
                }
                QToolButton#OcrCopyBtn::menu-button:hover {
                    background: #3B4252;
                }
                QPlainTextEdit#OcrPreviewEdit {
                    background: #14171B;
                    border: 1px solid #333A45;
                    border-radius: 4px;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    color: #E2E8F0;
                    padding: 4px;
                }
                QPlainTextEdit#OcrPreviewEdit[readOnly="false"] {
                    background: #181E27;
                    border: 1px solid #60A5FA;
                }
                QLabel#SectionTitle {
                    color: #60A5FA;
                    font-family: 'Segoe UI';
                    font-size: 10.5pt;
                    font-weight: 600;
                    border: none;
                    background: transparent;
                    margin-top: 6px;
                }
                QLineEdit#InlineEdit {
                    background: transparent;
                    border: 1px solid transparent;
                    border-radius: 3px;
                    padding: 3px 6px;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    color: #E2E8F0;
                }
                QLineEdit#InlineEdit:hover {
                    background: #1E2530;
                    border: 1px solid #475569;
                }
                QLineEdit#InlineEdit:focus {
                    background: #14171B;
                    border: 1px solid #3B82F6;
                }
                """
            )
            self._apply_fine_tune_contrast_styles(True)
            return

        self.container.setStyleSheet(
            """
            QFrame#RightSidebarContainer {
                background-color: #FFFFFF;
                border-left: 1px solid qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #DCE7F5, stop:0.08 #B7D1EE, stop:0.92 #B7D1EE, stop:1 #DCE7F5);
            }
            """
        )
        if hasattr(self, "page_info"):
            self.page_info.setStyleSheet("QWidget#InfoPage { background-color: #FFFFFF; }")
        if hasattr(self, "info_splitter"):
            self.info_splitter.setStyleSheet("QSplitter#InfoSplitter { background-color: #FFFFFF; }")
            for i in range(self.info_splitter.count()):
                w = self.info_splitter.widget(i)
                if isinstance(w, QScrollArea):
                    w.setStyleSheet("QScrollArea { background-color: #FFFFFF; border: none; }")
                    if w.widget():
                        w.widget().setStyleSheet("")
        self.setStyleSheet(
            """
            QDockWidget#TagAndCaptionDock {
                background: #FFFFFF;
                background-color: #FFFFFF;
            }
            QFrame#RightSidebarContainer {
                background-color: #FFFFFF;
                border-left: 1px solid qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #DCE7F5, stop:0.08 #B7D1EE, stop:0.92 #B7D1EE, stop:1 #DCE7F5);
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
                color: #1B5E9E;
                font-family: 'Segoe UI';
                font-size: 10.5pt;
                font-weight: 500;
                border: none;
                background: transparent;
                margin-top: 6px;
            }
            QFrame#FineTuneGroup {
                background: #FFFFFF;
                border: 1px solid #D2DFEE;
                border-radius: 4px;
            }
            QPushButton#FineTuneGroupTitle {
                color: #1B5E9E;
                font-family: 'Segoe UI';
                font-size: 11.5pt;
                font-weight: 500;
                border: none;
                background: transparent;
                text-align: left;
                padding: 0 0 2px 0;
            }
            QLabel#SliderHeaderLabel {
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 9pt;
                border: none;
                background: transparent;
            }
            QLabel#val_lbl {
                color: #5A7A9E;
                font-family: 'Segoe UI';
                font-size: 9pt;
                font-weight: 600;
                border: none;
                background: transparent;
                padding-right: 2px;
            }
            QLabel#SliderHeaderLabel:disabled,
            QLabel#val_lbl:disabled {
                color: #94A3B8;
            }
            QPushButton#FineTuneActionBtn,
            QPushButton#FineTuneResetBtn,
            QPushButton#DepthCutoutBtn {
                background-color: #F7FAFD;
                border: 1px solid #B5D2ED;
                border-radius: 3px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
                font-weight: 600;
                padding: 2px 8px;
            }
            QPushButton#DepthCutoutBtn {
                padding: 3px 6px;
                text-align: left;
            }
            QPushButton#FineTuneActionBtn:hover,
            QPushButton#DepthCutoutBtn:hover {
                background: #E4F0FC;
                border-color: #7BA7C9;
            }
            QPushButton#FineTuneResetBtn:hover {
                background: #FFF1F0;
                border-color: #FFA39E;
                color: #CF1322;
            }
            QPushButton#FineTuneActionBtn:pressed,
            QPushButton#DepthCutoutBtn:pressed {
                background: #CCE4F7;
                border-color: #3399FF;
            }
            QPushButton#FineTuneActionBtn:checked {
                background: #2563EB;
                border-color: #1D4ED8;
                color: #FFFFFF;
            }
            QFrame#RegionScopeBar {
                background: #F4F8FC;
                border: 1px solid #D2DFEE;
                border-radius: 5px;
            }
            QPushButton#FineTuneSegmentBtn {
                background: #FFFFFF;
                border: 1px solid #B5D2ED;
                border-right: none;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
                padding: 2px 4px;
            }
            QPushButton#FineTuneSegmentBtn[seg="first"] {
                border-top-left-radius: 4px;
                border-bottom-left-radius: 4px;
            }
            QPushButton#FineTuneSegmentBtn[seg="last"] {
                border-right: 1px solid #B5D2ED;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
            }
            QPushButton#FineTuneSegmentBtn:hover {
                background: #E4F0FC;
            }
            QPushButton#FineTuneSegmentBtn:checked {
                background: #2563EB;
                color: #FFFFFF;
                border-color: #1D4ED8;
                font-weight: 600;
            }
            QPushButton#FineTuneMaskToggleBtn {
                background: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 3px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 7.5pt;
                padding: 1px 7px;
            }
            QPushButton#FineTuneMaskToggleBtn:hover {
                background: #E4F0FC;
                border-color: #7BA7C9;
            }
            QPushButton#FineTuneMaskToggleBtn:checked {
                background: #DC2626;
                border-color: #B91C1C;
                color: #FFFFFF;
                font-weight: 600;
            }
            QPushButton#FineTuneMaskToggleBtn:disabled {
                background: transparent;
                border-color: #DCE5F0;
                color: #94A3B8;
            }
            QPushButton#FineTuneSubBtn {
                background-color: #F1F5F9;
                border: 1px solid #CBD5E1;
                border-radius: 3px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 7.5pt;
                padding: 1px 6px;
            }
            QPushButton#FineTuneSubBtn:hover {
                background-color: #E4F0FC;
                border-color: #7BA7C9;
            }
            QPushButton#FineTuneSubBtn:checked {
                background-color: #2563EB;
                border: 1px solid #1D4ED8;
                color: #FFFFFF;
                font-weight: 600;
            }
            QPushButton#FineTunePillBtn {
                padding: 2px 10px;
                border: 1px solid #B5D2ED;
                background: #F7FAFD;
                color: #1E395B;
                border-radius: 3px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
            }
            QPushButton#FineTunePillBtn:hover {
                background: #E4F0FC;
                border-color: #7BA7C9;
            }
            QPushButton#FineTunePillBtn:checked {
                background: #2563EB;
                color: #FFFFFF;
                border-color: #1D4ED8;
                font-weight: 600;
            }
            QToolButton#OcrToggleBtn {
                border: none;
                background: transparent;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                font-weight: 600;
                padding: 0;
            }
            QToolButton#OcrCopyBtn {
                font-size: 8pt;
                padding: 1px 16px 1px 8px;
                border: 1px solid #CBD5E1;
                border-radius: 3px;
                background: #F8FAFC;
                color: #1E395B;
            }
            QToolButton#OcrCopyBtn:hover {
                background: #E2E8F0;
                border-color: #94A3B8;
            }
            QToolButton#OcrCopyBtn::menu-button {
                border-left: 1px solid #CBD5E1;
                width: 14px;
            }
            QToolButton#OcrCopyBtn::menu-button:hover {
                background: #CBD5E1;
            }
            QPlainTextEdit#OcrPreviewEdit {
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 4px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                color: #0F172A;
                padding: 4px;
            }
            QPlainTextEdit#OcrPreviewEdit[readOnly="false"] {
                background: #FFFFFF;
                border: 1px solid #0078D7;
            }
            QLineEdit#InlineEdit {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 3px 6px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                color: #1E395B;
            }
            QLineEdit#InlineEdit:hover {
                background: #EEF5FC;
                border: 1px solid #9EC0E0;
            }
            QLineEdit#InlineEdit:focus {
                background: #FFFFFF;
                border: 1px solid #1E293B;
            }
            """
        )
        self._apply_fine_tune_contrast_styles(is_dark)

    def _apply_fine_tune_contrast_styles(self, is_dark: bool) -> None:
        """Apply explicit widget-level contrast styles so theme QSS never turns buttons white-on-white."""
        sub_css = (
            "QPushButton { background-color: #252B36; border: 1px solid #475569; color: #F1F5F9; "
            "border-radius: 3px; padding: 1px 6px; font-size: 12px; }"
            "QPushButton:hover { background-color: #334155; border-color: #60A5FA; }"
            "QPushButton:checked { background-color: #2563EB; border-color: #3B82F6; color: #FFFFFF; font-weight: 600; }"
            "QPushButton:disabled { background-color: #1E242E; border-color: #2D3542; color: #64748B; }"
            if is_dark
            else "QPushButton { background-color: #F1F5F9; border: 1px solid #CBD5E1; color: #1E293B; "
            "border-radius: 3px; padding: 1px 6px; font-size: 12px; }"
            "QPushButton:hover { background-color: #E4F0FC; border-color: #7BA7C9; color: #0F172A; }"
            "QPushButton:checked { background-color: #2563EB; border-color: #1D4ED8; color: #FFFFFF; font-weight: 600; }"
            "QPushButton:disabled { background-color: #F8FAFC; border-color: #E2E8F0; color: #94A3B8; }"
        )
        widgets: list[Any] = [
            getattr(self, "btn_curve_scurve", None),
            getattr(self, "btn_curve_reset", None),
            getattr(self, "btn_wb_eyedropper", None),
            getattr(self, "btn_import_cube_lut", None),
            getattr(self, "btn_hsl_reset", None),
            getattr(self, "btn_mask_show", None),
            getattr(self, "btn_mask_hide", None),
            getattr(self, "btn_region_brush_clear", None),
            getattr(self, "btn_region_reset", None),
        ]
        widgets.extend(getattr(self, "_hsl_tab_buttons", {}).values())
        for w in widgets:
            if w is not None:
                w.setStyleSheet(sub_css)

        circle_css = (
            "QPushButton { background: transparent; border: none; border-radius: 4px; padding: 2px; }"
            "QPushButton:hover { background: rgba(255, 255, 255, 0.08); }"
            "QPushButton:checked { background: rgba(255, 255, 255, 0.12); }"
            if is_dark
            else "QPushButton { background: transparent; border: none; border-radius: 4px; padding: 2px; }"
            "QPushButton:hover { background: rgba(15, 23, 42, 0.06); }"
            "QPushButton:checked { background: rgba(37, 99, 235, 0.10); }"
        )
        for ch_key, cb in getattr(self, "_curve_ch_buttons", {}).items():
            cb.setStyleSheet(circle_css)
            cb.setIcon(self._make_curve_channel_icon(ch_key, is_checked=cb.isChecked(), is_dark=is_dark))

        if hasattr(self, "btn_wb_eyedropper") and self.btn_wb_eyedropper is not None:
            self.btn_wb_eyedropper.setIcon(
                self._make_mask_tool_icon(
                    "eyedropper", 16, is_checked=self.btn_wb_eyedropper.isChecked(), is_dark=is_dark
                )
            )

        tile_css = (
            "QPushButton { background-color: #252B36; border: 1px solid #475569; border-radius: 5px; padding: 2px; }"
            "QPushButton:hover { background-color: #334155; border-color: #60A5FA; }"
            "QPushButton:checked { background-color: #2563EB; border: 1px solid #3B82F6; }"
            "QPushButton:disabled { background-color: #1E242E; border-color: #2D3542; }"
            if is_dark
            else "QPushButton { background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 5px; padding: 2px; }"
            "QPushButton:hover { background-color: #E4F0FC; border-color: #7BA7C9; }"
            "QPushButton:checked { background-color: #2563EB; border: 1px solid #1D4ED8; }"
            "QPushButton:disabled { background-color: #F1F5F9; border-color: #E2E8F0; }"
        )
        tile_widgets: list[Any] = [
            getattr(self, "btn_region_paint", None),
            getattr(self, "btn_region_erase", None),
            getattr(self, "btn_mask_invert", None),
        ]
        tile_widgets.extend(getattr(self, "_region_scope_buttons", {}).values())
        for tw in tile_widgets:
            if tw is not None:
                tw.setStyleSheet(tile_css)
        if hasattr(self, "_refresh_region_indicators"):
            self._refresh_region_indicators()

        combo_css = (
            "QComboBox { background-color: #22262E; color: #E2E8F0; border: 1px solid #475569; "
            "border-radius: 3px; padding: 2px 6px; font-size: 12px; }"
            "QComboBox:hover { border-color: #60A5FA; }"
            "QComboBox QAbstractItemView { background-color: #1B1E24; color: #E2E8F0; "
            "border: 1px solid #3E4756; selection-background-color: #2563EB; selection-color: #FFFFFF; outline: none; }"
            if is_dark
            else "QComboBox { background-color: #FFFFFF; color: #1E293B; border: 1px solid #CBD5E1; "
            "border-radius: 3px; padding: 2px 6px; font-size: 12px; }"
            "QComboBox:hover { border-color: #7BA7C9; }"
            "QComboBox QAbstractItemView { background-color: #FFFFFF; color: #1E293B; "
            "border: 1px solid #CBD5E1; selection-background-color: #E4F0FC; selection-color: #0F172A; outline: none; }"
        )
        view_css = (
            "QAbstractItemView { background-color: #1B1E24; color: #E2E8F0; "
            "selection-background-color: #2563EB; selection-color: #FFFFFF; outline: none; }"
            if is_dark
            else "QAbstractItemView { background-color: #FFFFFF; color: #1E293B; "
            "selection-background-color: #E4F0FC; selection-color: #0F172A; outline: none; }"
        )
        for cmb_attr in ("cmb_look_profile", "cmb_curve_preset", "cmb_wb_preset"):
            cmb = getattr(self, cmb_attr, None)
            if cmb is not None:
                cmb.setStyleSheet(combo_css)
                if cmb.view() is not None:
                    cmb.view().setStyleSheet(view_css)

        drawer_css = (
            "QPushButton { text-align: left; padding: 3px 8px; background: #262C36; border: 1px solid #3B4252; "
            "border-radius: 4px; color: #F1F5F9; font-weight: 600; font-size: 12px; }"
            "QPushButton:hover { background: #303846; border-color: #60A5FA; }"
            if is_dark
            else "QPushButton { text-align: left; padding: 3px 8px; background: #F1F5F9; border: 1px solid #CBD5E1; "
            "border-radius: 4px; color: #1E293B; font-weight: 600; font-size: 12px; }"
            "QPushButton:hover { background: #E4F0FC; border-color: #7BA7C9; }"
        )
        for d_attr in ("drawer_tone_curve", "drawer_color_mixer"):
            drw = getattr(self, d_attr, None)
            if drw is not None and hasattr(drw, "btn_toggle"):
                drw.btn_toggle.setStyleSheet(drawer_css)


RightTagDock = TagAndCaptionDock
