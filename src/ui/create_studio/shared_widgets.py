"""Shared theme-aware UI building blocks for integrated Create Studio workspaces."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QSize, QPointF
from PySide6.QtGui import QColor, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from src.ui.ribbon_toolbar import make_wlpg_icon
from src.ui.theme_tokens import get_theme_tokens


def _cv2_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    """Converts a BGR numpy array to QPixmap safely without dangling memory."""
    if img_bgr is None or img_bgr.size == 0:
        return QPixmap()
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    h, w = img_bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _qcolor_to_bgr(color: QColor) -> tuple[int, int, int]:
    return color.blue(), color.green(), color.red()


def prompt_save_new_file(
    parent: QWidget,
    base_file_path: Path | None,
    suffix: str,
    ext: str = ".jpg",
    filter_str: str = "JPEG Image (*.jpg *.jpeg);;All Files (*.*)",
) -> Path | None:
    """Prompt the user to save a newly generated creation without touching source originals."""
    if base_file_path:
        stem = base_file_path.stem
        src_dir = base_file_path.parent
    else:
        stem = "Photo"
        src_dir = Path.home()

    default_name = f"{stem}_{suffix}{ext}"
    initial_path = (src_dir / default_name).as_posix()

    selected_path, _ = QFileDialog.getSaveFileName(
        parent,
        f"Save New File ({suffix})",
        initial_path,
        filter_str,
    )
    if not selected_path:
        return None
    return Path(selected_path)


def stop_qthread_safely(worker: QThread | None, timeout_ms: int = 2000) -> None:
    """Request interruption and wait cleanly so Qt never destroys an active QThread."""
    if worker is None:
        return
    try:
        worker.blockSignals(True)
        worker.requestInterruption()
        if hasattr(worker, "cancel") and callable(worker.cancel):
            worker.cancel()
        if worker.isRunning():
            worker.quit()
            worker.wait(max(200, int(timeout_ms)))
    except Exception:
        pass


def build_studio_panel_qss(theme_name: str = "wlpg_aero") -> str:
    """
    Build a complete, readable stylesheet for Create Studio inspector/tray widgets
    from ThemeTokens, strictly scoping containers by objectName so child buttons and
    comboboxes always retain clear borders and popup contrast.
    """
    tok = get_theme_tokens(theme_name)
    btn_bg = tok.bg_card if not tok.is_dark else "#272C36"
    btn_border = tok.border_subtle
    groove_bg = "#D6E2EE" if not tok.is_dark else "#333A45"
    handle_bg = "#FFFFFF" if not tok.is_dark else "#E2E8F0"
    handle_border = tok.accent
    return f"""
        QWidget#StudioInspectorRoot, QWidget#StudioCanvasRoot {{
            background-color: {tok.bg_surface};
            color: {tok.text_primary};
        }}
        QScrollArea#StudioInspectorScroll, QWidget#StudioInspectorScrollContent {{
            background-color: {tok.bg_surface};
            border: none;
        }}
        QFrame#StudioTrayRoot {{
            background-color: {tok.bg_surface};
            border: none;
        }}
        QScrollArea#StudioTrayScroll,
        QScrollArea#StudioTrayScroll > QWidget,
        QScrollArea#StudioTrayScroll > QWidget > QWidget,
        QWidget#StudioTrayHost {{
            background-color: {tok.bg_surface};
            border: none;
        }}
        QFrame#StudioCardGroup {{
            background-color: {tok.bg_elevated};
            border: 1px solid {tok.border_subtle};
            border-radius: 5px;
        }}
        QWidget#StudioCardBody {{
            background: transparent;
            border: none;
        }}
        QLabel#StudioCardHeader {{
            color: {tok.accent if not tok.is_dark else "#93C5FD"};
            font-family: 'Segoe UI';
            font-size: 9pt;
            font-weight: 600;
            background: transparent;
        }}
        QWidget#StudioInspectorRoot QLabel,
        QFrame#StudioTrayRoot QLabel {{
            color: {tok.text_primary};
            background: transparent;
        }}
        QWidget#StudioInspectorRoot QLabel[uiRole="muted"],
        QFrame#StudioTrayRoot QLabel[uiRole="muted"] {{
            color: {tok.text_secondary};
        }}
        QWidget#StudioInspectorRoot QCheckBox,
        QFrame#StudioTrayRoot QCheckBox {{
            color: {tok.text_primary};
            background: transparent;
            spacing: 6px;
        }}
        QWidget#StudioInspectorRoot QPushButton,
        QFrame#StudioTrayRoot QPushButton {{
            background-color: {btn_bg};
            color: {tok.text_primary};
            border: 1px solid {btn_border};
            border-radius: 4px;
            padding: 4px 10px;
            font-family: 'Segoe UI';
            font-size: 9pt;
        }}
        QWidget#StudioInspectorRoot QPushButton:hover,
        QFrame#StudioTrayRoot QPushButton:hover {{
            background-color: {tok.bg_hover};
            border-color: {tok.accent};
        }}
        QWidget#StudioInspectorRoot QPushButton:checked,
        QFrame#StudioTrayRoot QPushButton:checked {{
            background-color: {tok.selection_bg};
            color: {tok.selection_text};
            border: 1.5px solid {tok.accent};
            font-weight: 600;
        }}
        QWidget#StudioInspectorRoot QPushButton[uiRole="primary_btn"],
        QFrame#StudioTrayRoot QPushButton[uiRole="primary_btn"] {{
            background-color: {tok.accent};
            color: {tok.text_on_accent};
            border: 1px solid {tok.accent};
            font-weight: 600;
        }}
        QWidget#StudioInspectorRoot QPushButton[uiRole="primary_btn"]:hover,
        QFrame#StudioTrayRoot QPushButton[uiRole="primary_btn"]:hover {{
            background-color: {tok.accent_hover};
            border-color: {tok.accent_hover};
        }}
        QWidget#StudioInspectorRoot QPushButton:disabled,
        QFrame#StudioTrayRoot QPushButton:disabled {{
            opacity: 0.55;
        }}
        QFrame#StudioTrayRoot QPushButton#TrayMiniBtn {{
            padding: 0px;
            min-width: 18px;
            min-height: 18px;
            font-size: 9px;
            font-weight: 600;
            border-radius: 3px;
        }}
        QWidget#StudioInspectorRoot QComboBox,
        QFrame#StudioTrayRoot QComboBox,
        QWidget#StudioInspectorRoot QSpinBox,
        QFrame#StudioTrayRoot QSpinBox {{
            background-color: {tok.bg_input};
            color: {tok.text_primary};
            border: 1px solid {tok.border_subtle};
            border-radius: 4px;
            padding: 3px 8px;
            min-height: 22px;
        }}
        QWidget#StudioInspectorRoot QComboBox:hover,
        QFrame#StudioTrayRoot QComboBox:hover {{
            border-color: {tok.accent};
        }}
        QComboBox QAbstractItemView {{
            background-color: {tok.bg_elevated};
            color: {tok.text_primary};
            border: 1px solid {tok.border_subtle};
            selection-background-color: {tok.selection_bg};
            selection-color: {tok.selection_text};
            outline: none;
        }}
        QComboBox QAbstractItemView::item {{
            color: {tok.text_primary};
            background-color: transparent;
            padding: 4px 8px;
            min-height: 22px;
        }}
        QComboBox QAbstractItemView::item:selected,
        QComboBox QAbstractItemView::item:hover {{
            background-color: {tok.selection_bg};
            color: {tok.selection_text};
        }}
        QWidget#StudioInspectorRoot QListWidget {{
            background-color: {tok.bg_input};
            color: {tok.text_primary};
            border: 1px solid {tok.border_subtle};
            border-radius: 4px;
        }}
        QWidget#StudioInspectorRoot QListWidget::item {{
            padding: 4px 6px;
            border-radius: 3px;
        }}
        QWidget#StudioInspectorRoot QListWidget::item:selected {{
            background-color: {tok.selection_bg};
            color: {tok.selection_text};
        }}
        QWidget#StudioInspectorRoot QSlider::groove:horizontal,
        QFrame#StudioTrayRoot QSlider::groove:horizontal {{
            height: 4px;
            background: {groove_bg};
            border-radius: 2px;
        }}
        QWidget#StudioInspectorRoot QSlider::handle:horizontal,
        QFrame#StudioTrayRoot QSlider::handle:horizontal {{
            width: 12px;
            height: 12px;
            margin: -4px 0;
            background: {handle_bg};
            border: 1.5px solid {handle_border};
            border-radius: 6px;
        }}
        QWidget#StudioInspectorRoot QProgressBar {{
            background-color: {tok.bg_input};
            color: {tok.text_primary};
            border: 1px solid {tok.border_subtle};
            border-radius: 4px;
            text-align: center;
        }}
        QWidget#StudioInspectorRoot QProgressBar::chunk {{
            background-color: {tok.accent};
            border-radius: 3px;
        }}
    """


class StudioCardGroup(QFrame):
    """Theme-aware rounded card container used in Create Studio right inspectors."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StudioCardGroup")
        self.vbox = QVBoxLayout(self)
        self.vbox.setContentsMargins(10, 8, 10, 10)
        self.vbox.setSpacing(6)

        self.lbl_title = QLabel(title, self)
        self.lbl_title.setObjectName("StudioCardHeader")
        self.lbl_title.setFont(QFont("Segoe UI", 9, QFont.DemiBold))
        self.vbox.addWidget(self.lbl_title)

        self.body = QWidget(self)
        self.body.setObjectName("StudioCardBody")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(6)
        self.vbox.addWidget(self.body)

    def add_widget(self, w: QWidget) -> None:
        self.body_layout.addWidget(w)

    def add_layout(self, lay: Any) -> None:
        self.body_layout.addLayout(lay)


class AeroStudioSlider(QSlider):
    """Smooth horizontal slider supporting double-click reset to default."""

    def __init__(
        self,
        minimum: int,
        maximum: int,
        default_val: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setRange(minimum, maximum)
        self.setValue(default_val)
        self.default_val = default_val
        self.setFixedHeight(18)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        self.setValue(self.default_val)
        super().mouseDoubleClickEvent(event)


class CreateStudioZoomBar(QFrame):
    """Floating/bottom zoom bar providing Fit, 100%, and 10%-400% zoom slider."""

    fit_clicked = Signal()
    actual_size_clicked = Signal()
    zoom_in_clicked = Signal()
    zoom_out_clicked = Signal()
    zoom_slider_moved = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CreateStudioZoomBar")
        self.setFixedHeight(32)
        self._theme_name = "wlpg_aero"

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 3, 8, 3)
        lay.setSpacing(6)

        self.btn_fit = QPushButton("Fit", self)
        self.btn_fit.setIcon(make_wlpg_icon("fit_window", 14))
        self.btn_fit.setIconSize(QSize(14, 14))
        self.btn_fit.setFixedHeight(24)
        self.btn_fit.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_fit.setToolTip("Zoom to fit canvas (F)")
        self.btn_fit.clicked.connect(self.fit_clicked.emit)
        lay.addWidget(self.btn_fit)

        self.btn_100 = QPushButton("100%", self)
        self.btn_100.setIcon(make_wlpg_icon("actual_size", 14))
        self.btn_100.setIconSize(QSize(14, 14))
        self.btn_100.setFixedHeight(24)
        self.btn_100.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_100.setToolTip("Actual pixels 100% (1)")
        self.btn_100.clicked.connect(self.actual_size_clicked.emit)
        lay.addWidget(self.btn_100)

        self.btn_zoom_out = QPushButton("", self)
        self.btn_zoom_out.setIcon(make_wlpg_icon("zoom_out", 14))
        self.btn_zoom_out.setIconSize(QSize(14, 14))
        self.btn_zoom_out.setFixedSize(26, 24)
        self.btn_zoom_out.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_zoom_out.setToolTip("Zoom out (-)")
        self.btn_zoom_out.clicked.connect(self.zoom_out_clicked.emit)
        lay.addWidget(self.btn_zoom_out)

        self.slider_zoom = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_zoom.setRange(10, 400)
        self.slider_zoom.setValue(100)
        self.slider_zoom.setFixedWidth(120)
        self.slider_zoom.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slider_zoom.valueChanged.connect(self._on_slider_changed)
        lay.addWidget(self.slider_zoom)

        self.btn_zoom_in = QPushButton("", self)
        self.btn_zoom_in.setIcon(make_wlpg_icon("zoom_in", 14))
        self.btn_zoom_in.setIconSize(QSize(14, 14))
        self.btn_zoom_in.setFixedSize(26, 24)
        self.btn_zoom_in.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_zoom_in.setToolTip("Zoom in (+)")
        self.btn_zoom_in.clicked.connect(self.zoom_in_clicked.emit)
        lay.addWidget(self.btn_zoom_in)

        self.lbl_zoom_pct = QLabel("100%", self)
        self.lbl_zoom_pct.setFixedWidth(42)
        self.lbl_zoom_pct.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.lbl_zoom_pct)

        self.apply_theme_mode("wlpg_aero")

    def apply_theme_mode(self, theme_name: str) -> None:
        self._theme_name = theme_name or "wlpg_aero"
        tok = get_theme_tokens(self._theme_name)
        btn_bg = tok.bg_card if not tok.is_dark else "#272C36"
        groove = "#CBD5E1" if not tok.is_dark else "#334155"
        self.setStyleSheet(
            f"""
            QFrame#CreateStudioZoomBar {{
                background-color: {tok.bg_surface};
                border-top: 1px solid {tok.border_subtle};
            }}
            QFrame#CreateStudioZoomBar QPushButton {{
                background-color: {btn_bg};
                color: {tok.text_primary};
                border: 1px solid {tok.border_subtle};
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 8.5pt;
            }}
            QFrame#CreateStudioZoomBar QPushButton:hover {{
                background-color: {tok.bg_hover};
                border-color: {tok.accent};
            }}
            QFrame#CreateStudioZoomBar QLabel {{
                color: {tok.text_secondary};
                font-size: 8.5pt;
                font-weight: 600;
                background: transparent;
            }}
            QFrame#CreateStudioZoomBar QSlider::groove:horizontal {{
                height: 4px;
                background: {groove};
                border-radius: 2px;
            }}
            QFrame#CreateStudioZoomBar QSlider::handle:horizontal {{
                width: 11px;
                height: 11px;
                margin: -4px 0;
                background: {tok.bg_elevated};
                border: 1.5px solid {tok.accent};
                border-radius: 5px;
            }}
            """
        )
        self.btn_fit.setIcon(make_wlpg_icon("fit_window", 14))
        self.btn_100.setIcon(make_wlpg_icon("actual_size", 14))
        self.btn_zoom_out.setIcon(make_wlpg_icon("zoom_out", 14))
        self.btn_zoom_in.setIcon(make_wlpg_icon("zoom_in", 14))

    def _on_slider_changed(self, val: int) -> None:
        self.lbl_zoom_pct.setText(f"{val}%")
        self.zoom_slider_moved.emit(val)

    def set_zoom_level(self, percent: int) -> None:
        self.slider_zoom.blockSignals(True)
        self.slider_zoom.setValue(max(10, min(400, int(percent))))
        self.slider_zoom.blockSignals(False)
        self.lbl_zoom_pct.setText(f"{max(10, min(400, int(percent)))}%")

    def update_zoom_display(self, scale_val: float) -> None:
        pct = max(10, min(400, int(round(scale_val * 100))))
        self.set_zoom_level(pct)

    def set_is_fit(self, is_fit: bool) -> None:
        self.btn_fit.setEnabled(not is_fit)


class ZoomablePreviewCanvas(QGraphicsView):
    """Smooth zoom & pan QGraphicsView for Create Studio workspaces."""

    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PySide6.QtGui import QPainter

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.TextAntialiasing
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._space_panning = False
        self.apply_theme_mode("wlpg_aero")

    def apply_theme_mode(self, theme_name: str) -> None:
        tok = get_theme_tokens(theme_name)
        canvas_bg = "#111318" if tok.is_dark else "#232730"
        self.setStyleSheet(f"QGraphicsView {{ background-color: {canvas_bg}; border: none; }}")

    def wheelEvent(self, event: Any) -> None:
        if (event.modifiers() & Qt.KeyboardModifier.ControlModifier) or event.angleDelta().y() != 0:
            factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
            self.scale(factor, factor)
            self.zoom_changed.emit(self.transform().m11())
            event.accept()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._space_panning = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._space_panning:
            self._space_panning = False
            self.unsetCursor()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: Any) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_panning = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: Any) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_panning = False
            self.unsetCursor()
            event.accept()
            return
        super().keyReleaseEvent(event)
