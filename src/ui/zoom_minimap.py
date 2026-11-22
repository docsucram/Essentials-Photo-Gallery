"""
Zoom Minimap Navigator + Sharpness Heatmap Grid Overlay for SingleMediaViewer & CompareMediaView.
Appears automatically in the bottom-right corner when zoomed beyond fit (>100%),
allowing click-and-drag panning, scroll-wheel zooming, and an optional thumbnail-resolution
Laplacian sharpness grid overlay.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QFrame, QToolButton, QWidget


def compute_sharpness_grid(
    qimg: QImage, cols: int = 16, rows: int = 12
) -> list[tuple[int, int, float]]:
    """
    Computes normalized [0.0..1.0] per-cell edge/sharpness intensity on a small thumbnail
    QImage (never decodes full-res pixels). Returns `(col, row, intensity)` for active cells.
    """
    if qimg is None or qimg.isNull() or qimg.width() < 16 or qimg.height() < 16:
        return []
    small = (
        qimg.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        if max(qimg.width(), qimg.height()) > 256
        else qimg
    )
    gray = small.convertToFormat(QImage.Format.Format_Grayscale8)
    w, h = gray.width(), gray.height()
    if w < cols or h < rows:
        return []

    bpl = gray.bytesPerLine()
    raw = bytes(gray.constBits())
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, bpl))[:, :w].astype(np.float32)

    # Fast 3x3 Laplacian response |4*c - u - d - l - r|
    lap = np.zeros_like(arr)
    lap[1:-1, 1:-1] = np.abs(
        4.0 * arr[1:-1, 1:-1]
        - arr[:-2, 1:-1]
        - arr[2:, 1:-1]
        - arr[1:-1, :-2]
        - arr[1:-1, 2:]
    )

    cell_w = w / float(cols)
    cell_h = h / float(rows)
    scores: list[tuple[int, int, float]] = []
    max_score = 0.0
    for r in range(rows):
        y0 = int(r * cell_h)
        y1 = max(y0 + 1, int((r + 1) * cell_h))
        for c in range(cols):
            x0 = int(c * cell_w)
            x1 = max(x0 + 1, int((c + 1) * cell_w))
            val = float(np.mean(lap[y0:y1, x0:x1]))
            scores.append((c, r, val))
            if val > max_score:
                max_score = val

    if max_score <= 1e-3:
        return []

    thresh = max_score * 0.35
    return [
        (c, r, min(1.0, (val - thresh) / max(1e-3, max_score - thresh)))
        for c, r, val in scores
        if val >= thresh
    ]


class ZoomMinimapOverlay(QFrame):
    """
    Translucent corner minimap showing the full image, the active viewport rectangle,
    and an optional Sharpness Heatmap Grid toggle.
    """

    navigate_requested = Signal(float, float)  # norm_cx, norm_cy in [0..1]
    zoom_requested = Signal(float)             # wheel multiplier
    sharpness_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ZoomMinimapOverlay")
        self.setFixedSize(164, 118)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QFrame#ZoomMinimapOverlay { "
            "  background-color: rgba(18, 22, 28, 205); "
            "  border: 1px solid rgba(255, 255, 255, 55); "
            "  border-radius: 6px; "
            "}"
        )

        self._thumb: QPixmap = QPixmap()
        self._sharpness_cells: list[tuple[int, int, float]] = []
        self._sharpness_cols: int = 16
        self._sharpness_rows: int = 12
        self._sharpness_enabled: bool = False
        self._source_key: int | str | None = None

        # Normalized viewport rectangle in [0..1]
        self._vx0: float = 0.0
        self._vy0: float = 0.0
        self._vx1: float = 1.0
        self._vy1: float = 1.0
        self._dragging: bool = False

        self.btn_sharpness = QToolButton(self)
        self.btn_sharpness.setText("Focus")
        self.btn_sharpness.setCheckable(True)
        self.btn_sharpness.setToolTip("Toggle Sharpness Heatmap Grid")
        self.btn_sharpness.setFixedSize(44, 18)
        self.btn_sharpness.move(6, 6)
        self.btn_sharpness.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_sharpness.setStyleSheet(
            "QToolButton { "
            "  background: rgba(15, 23, 42, 190); color: #E2E8F0; "
            "  border: 1px solid rgba(255,255,255,60); border-radius: 3px; "
            "  font-size: 9px; font-weight: 600; padding: 0 3px; "
            "} "
            "QToolButton:checked { "
            "  background: #0078D7; color: #FFFFFF; border-color: #60A5FA; "
            "}"
        )
        self.btn_sharpness.toggled.connect(self._on_sharpness_toggled)
        self.hide()

    def _on_sharpness_toggled(self, checked: bool) -> None:
        self._sharpness_enabled = bool(checked)
        if self._sharpness_enabled and not self._sharpness_cells and not self._thumb.isNull():
            self._sharpness_cells = compute_sharpness_grid(
                self._thumb.toImage(), self._sharpness_cols, self._sharpness_rows
            )
        self.sharpness_toggled.emit(self._sharpness_enabled)
        self.update()

    def set_image(self, image: QPixmap | QImage | None, source_key: Any = None) -> None:
        if source_key is not None and source_key == self._source_key and not self._thumb.isNull():
            return
        self._source_key = source_key
        self._sharpness_cells = []
        if isinstance(image, QImage) and not image.isNull():
            small = image.scaled(
                QSize(152, 106),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._thumb = QPixmap.fromImage(small)
            if self._sharpness_enabled:
                self._sharpness_cells = compute_sharpness_grid(
                    small, self._sharpness_cols, self._sharpness_rows
                )
        elif isinstance(image, QPixmap) and not image.isNull():
            self._thumb = image.scaled(
                QSize(152, 106),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            if self._sharpness_enabled:
                self._sharpness_cells = compute_sharpness_grid(
                    self._thumb.toImage(), self._sharpness_cols, self._sharpness_rows
                )
        else:
            self._thumb = QPixmap()
        self.update()

    def set_viewport_norm(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self._vx0 = max(0.0, min(1.0, float(x0)))
        self._vy0 = max(0.0, min(1.0, float(y0)))
        self._vx1 = max(self._vx0, min(1.0, float(x1)))
        self._vy1 = max(self._vy0, min(1.0, float(y1)))
        self.update()

    def _image_rect(self) -> QRectF:
        pad = 6.0
        avail_w = max(1.0, float(self.width()) - pad * 2.0)
        avail_h = max(1.0, float(self.height()) - pad * 2.0)
        if self._thumb.isNull():
            return QRectF(pad, pad, avail_w, avail_h)
        tw = float(self._thumb.width())
        th = float(self._thumb.height())
        scale = min(avail_w / max(1.0, tw), avail_h / max(1.0, th))
        dw = tw * scale
        dh = th * scale
        x = (float(self.width()) - dw) * 0.5
        y = (float(self.height()) - dh) * 0.5
        return QRectF(x, y, dw, dh)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        img_rect = self._image_rect()
        if not self._thumb.isNull():
            painter.drawPixmap(img_rect.toRect(), self._thumb)

        if self._sharpness_enabled and self._sharpness_cells:
            # Darken photo slightly so the thermal focus heatmap pops clearly over bright/colorful areas
            painter.fillRect(img_rect, QColor(0, 0, 0, 125))
            cw = img_rect.width() / float(self._sharpness_cols)
            ch = img_rect.height() / float(self._sharpness_rows)
            painter.setPen(Qt.PenStyle.NoPen)
            for col, row, intensity in self._sharpness_cells:
                t = max(0.0, min(1.0, float(intensity)))
                if t < 0.5:
                    u = t * 2.0
                    r = int(34 + (250 - 34) * u)
                    g = int(197 + (204 - 197) * u)
                    b = int(94 + (21 - 94) * u)
                else:
                    u = (t - 0.5) * 2.0
                    r = int(250 + (239 - 250) * u)
                    g = int(204 + (68 - 204) * u)
                    b = int(21 + (68 - 21) * u)
                alpha = int(105 + t * 115)
                painter.setBrush(QColor(r, g, b, alpha))
                painter.drawRect(
                    QRectF(
                        img_rect.left() + col * cw,
                        img_rect.top() + row * ch,
                        cw + 0.5,
                        ch + 0.5,
                    )
                )

        # Dim outside the active viewport box
        vx0 = img_rect.left() + self._vx0 * img_rect.width()
        vy0 = img_rect.top() + self._vy0 * img_rect.height()
        vx1 = img_rect.left() + self._vx1 * img_rect.width()
        vy1 = img_rect.top() + self._vy1 * img_rect.height()
        vp_rect = QRectF(vx0, vy0, max(4.0, vx1 - vx0), max(4.0, vy1 - vy0))

        painter.setBrush(QBrush(QColor(0, 0, 0, 95)))
        painter.setPen(Qt.PenStyle.NoPen)
        if vp_rect.top() > img_rect.top():
            painter.drawRect(QRectF(img_rect.left(), img_rect.top(), img_rect.width(), vp_rect.top() - img_rect.top()))
        if vp_rect.bottom() < img_rect.bottom():
            painter.drawRect(QRectF(img_rect.left(), vp_rect.bottom(), img_rect.width(), img_rect.bottom() - vp_rect.bottom()))
        if vp_rect.left() > img_rect.left():
            painter.drawRect(QRectF(img_rect.left(), vp_rect.top(), vp_rect.left() - img_rect.left(), vp_rect.height()))
        if vp_rect.right() < img_rect.right():
            painter.drawRect(QRectF(vp_rect.right(), vp_rect.top(), img_rect.right() - vp_rect.right(), vp_rect.height()))

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#38BDF8"), 1.6))
        painter.drawRect(vp_rect)
        painter.end()

    def _emit_pos(self, pos: QPointF) -> None:
        r = self._image_rect()
        nx = max(0.0, min(1.0, (pos.x() - r.left()) / max(1.0, r.width())))
        ny = max(0.0, min(1.0, (pos.y() - r.top()) / max(1.0, r.height())))
        self.navigate_requested.emit(nx, ny)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._emit_pos(event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self._emit_pos(event.position())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta != 0:
            self.zoom_requested.emit(1.15 if delta > 0 else (1.0 / 1.15))
            event.accept()
            return
        super().wheelEvent(event)
