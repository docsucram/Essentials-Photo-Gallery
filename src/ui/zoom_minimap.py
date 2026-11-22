"""
Zoom Minimap Navigator Overlay for SingleMediaViewer & CompareMediaView.
Appears automatically in the bottom-right corner when zoomed beyond fit (>100%),
allowing click-and-drag panning and scroll-wheel zooming.
"""
from __future__ import annotations

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
from PySide6.QtWidgets import QFrame, QWidget


class ZoomMinimapOverlay(QFrame):
    """
    Translucent corner minimap showing the full image and the active viewport rectangle.
    """

    navigate_requested = Signal(float, float)  # norm_cx, norm_cy in [0..1]
    zoom_requested = Signal(float)             # wheel multiplier

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
        self._source_key: int | str | None = None

        # Normalized viewport rectangle in [0..1]
        self._vx0: float = 0.0
        self._vy0: float = 0.0
        self._vx1: float = 1.0
        self._vy1: float = 1.0
        self._dragging: bool = False
        self.hide()

    def set_image(self, image: QPixmap | QImage | None, source_key: Any = None) -> None:
        if source_key is not None and source_key == self._source_key and not self._thumb.isNull():
            return
        self._source_key = source_key
        if isinstance(image, QImage) and not image.isNull():
            small = image.scaled(
                QSize(152, 106),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._thumb = QPixmap.fromImage(small)
        elif isinstance(image, QPixmap) and not image.isNull():
            self._thumb = image.scaled(
                QSize(152, 106),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
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
