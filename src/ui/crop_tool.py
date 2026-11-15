"""
High-Performance Crop Overlay & Scene for PhotoGallery64.
Faithfully recreates Windows Live Photo Gallery's Crop Tool (`docs/reference_ui/fullsize and edit_ui.png`,
`media_1792235796451_8aed8686.jpg`) enhanced with Photoshop-style rotation & resolution badge
(`media_1792235984024_641c89d3.jpg`):
- Rule of Thirds 3x3 compositional grid lines (semi-transparent white).
- Dimmed / greyed-out outer area covering parts of the image to be cropped out.
- 8 Symmetrical handles (4 corners + 4 edge midpoints) dynamically scaled to stay easy to grab (~12 screen px).
- Edge line hit testing for effortless horizontal/vertical resizing.
- Corner-oriented curved double-arrow rotation cursors (TL, TR, BR, BL).
- Rotation zone strictly confined to outer corners; moving back toward lines instantly switches to resize mode.
- Live angle badge displaying current rotation (e.g. -29.2°).
- Live pixel resolution badge displaying current crop pixel dimensions (e.g. 2467 × 1863 px), crisp at any zoom level.
- Double-click or Enter to apply, Escape to cancel.
"""

from __future__ import annotations

import math
from typing import Literal

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QFontMetrics,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QStyleOptionGraphicsItem,
    QWidget,
)


def make_corner_rotation_cursor(angle_deg: float) -> QCursor:
    """Renders a crisp Photoshop-style curved double-arrow rotation cursor oriented for a specific corner."""
    pix = QPixmap(32, 32)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)

    # Rotate around center
    p.translate(16, 16)
    p.rotate(angle_deg)
    p.translate(-16, -16)

    center = QPointF(16, 16)
    radius = 9.5
    arc_rect = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)

    # Black drop shadow / border arc
    p.setPen(QPen(QColor(10, 10, 10, 230), 4.2, Qt.SolidLine, Qt.RoundCap))
    p.drawArc(arc_rect, int(35 * 16), int(190 * 16))

    # White inner line
    p.setPen(QPen(QColor(255, 255, 255, 255), 2.2, Qt.SolidLine, Qt.RoundCap))
    p.drawArc(arc_rect, int(35 * 16), int(190 * 16))

    # Arrowhead 1 (near 35 degrees)
    a1 = math.radians(35)
    pt1 = QPointF(center.x() + radius * math.cos(a1), center.y() - radius * math.sin(a1))
    poly1 = QPolygonF([
        pt1,
        QPointF(pt1.x() + 4.5, pt1.y() - 2.0),
        QPointF(pt1.x() + 2.0, pt1.y() + 4.5),
    ])

    # Arrowhead 2 (near 225 degrees)
    a2 = math.radians(225)
    pt2 = QPointF(center.x() + radius * math.cos(a2), center.y() - radius * math.sin(a2))
    poly2 = QPolygonF([
        pt2,
        QPointF(pt2.x() - 4.5, pt2.y() + 2.0),
        QPointF(pt2.x() - 2.0, pt2.y() - 4.5),
    ])

    p.setPen(QPen(QColor(10, 10, 10, 230), 1.0))
    p.setBrush(QBrush(QColor(255, 255, 255)))
    p.drawPolygon(poly1)
    p.drawPolygon(poly2)

    p.end()
    return QCursor(pix, 16, 16)


# Cached corner rotation cursors
_ROTATION_CURSORS: dict[str, QCursor] = {}


def get_rotation_cursor(corner: str = "tr") -> QCursor:
    """Returns the rotation cursor oriented for the specific corner: 'tr', 'br', 'bl', 'tl'."""
    angles = {"tr": 90.0, "br": 180.0, "bl": 270.0, "tl": 0.0}
    ang = angles.get(corner, 90.0)
    if corner not in _ROTATION_CURSORS:
        _ROTATION_CURSORS[corner] = make_corner_rotation_cursor(ang)
    return _ROTATION_CURSORS[corner]


def make_curved_rotation_cursor() -> QCursor:
    """Renders a curved double-arrow rotation cursor (default orientation)."""
    return make_corner_rotation_cursor(90.0)


class CropOverlayItem(QGraphicsRectItem):
    """
    Renders the WLPG Crop frame with Rule-of-Thirds grid, dimmed outer mask,
    8 resize handles, resolution badge, and live rotation badge.
    Dynamically scales handles and badges according to the view's current zoom level.
    """

    def __init__(self, parent: QGraphicsItem | None = None) -> None:
        super().__init__(parent)
        self.setZValue(100)
        self.setAcceptHoverEvents(True)

        self._rect = QRectF()
        self._image_rect = QRectF()
        self._aspect_ratio: float | None = None  # None = freeform/custom
        self._master_scale: float = 1.0  # multiplier to convert display pixels to master image pixels
        self._rotation_deg: float = 0.0
        self._is_rotating: bool = False
        self._rotation_badge_pos: QPointF = QPointF()

    def get_view_inv_scale(self) -> float:
        """Computes 1.0 / zoom_level from the active QGraphicsView to keep screen pixel sizes constant."""
        if self.scene() and self.scene().views():
            view = self.scene().views()[0]
            m11 = view.transform().m11()
            if m11 > 0.001:
                return 1.0 / m11
        return 1.0

    def setRect(self, rect: QRectF) -> None:
        self.prepareGeometryChange()
        self._rect = QRectF(rect).normalized()
        super().setRect(self._rect)
        self.update()

    def rect(self) -> QRectF:
        return self._rect

    def set_aspect_ratio(self, ratio: float | None) -> None:
        self._aspect_ratio = ratio
        if ratio is not None and ratio > 0 and not self._rect.isEmpty():
            cx, cy = self._rect.center().x(), self._rect.center().y()
            w = self._rect.width()
            h = w / ratio
            if self.scene() and self.scene().sceneRect().isValid():
                s_rect = self.scene().sceneRect()
                if h > s_rect.height() * 0.95:
                    h = s_rect.height() * 0.85
                    w = h * ratio
                if w > s_rect.width() * 0.95:
                    w = s_rect.width() * 0.85
                    h = w / ratio
            self.setRect(QRectF(cx - w / 2.0, cy - h / 2.0, w, h))

    def aspect_ratio(self) -> float | None:
        return self._aspect_ratio

    def set_master_scale(self, scale: float) -> None:
        self._master_scale = max(0.001, float(scale))
        self.update()

    def set_rotation_deg(self, deg: float, badge_pos: QPointF | None = None) -> None:
        self._rotation_deg = float(deg)
        if badge_pos is not None:
            self._rotation_badge_pos = badge_pos
        self.update()

    def rotation_deg(self) -> float:
        return self._rotation_deg

    def set_image_rect(self, rect: QRectF) -> None:
        self.prepareGeometryChange()
        self._image_rect = QRectF(rect)
        self.update()

    def _rotated_image_polygon(self, scene_rect: QRectF) -> QPolygonF:
        if scene_rect.isEmpty() or abs(self._rotation_deg) < 0.05:
            return QPolygonF(scene_rect)
        c = scene_rect.center()
        t = QTransform()
        t.translate(c.x(), c.y())
        t.rotate(self._rotation_deg)
        t.translate(-c.x(), -c.y())
        return t.map(QPolygonF(scene_rect))

    def boundingRect(self) -> QRectF:
        img_rect = getattr(self, "_image_rect", QRectF())
        if not img_rect.isEmpty():
            poly_rect = self._rotated_image_polygon(img_rect).boundingRect()
            return poly_rect.united(self._rect).adjusted(-32.0, -32.0, 32.0, 32.0)
        inv_scale = self.get_view_inv_scale()
        margin = max(100.0, 150.0 * inv_scale)
        return self._rect.adjusted(-margin, -margin, margin, margin)

    def get_handles(self) -> dict[str, QPointF]:
        r = self._rect
        return {
            "tl": r.topLeft(),
            "tr": r.topRight(),
            "bl": r.bottomLeft(),
            "br": r.bottomRight(),
            "tc": QPointF(r.center().x(), r.top()),
            "bc": QPointF(r.center().x(), r.bottom()),
            "lc": QPointF(r.left(), r.center().y()),
            "rc": QPointF(r.right(), r.center().y()),
        }

    def hit_test(self, pos: QPointF) -> tuple[str, QCursor | None]:
        """
        Evaluates mouse pos in scene coordinates taking current view scale into account.
        Returns (action_type, cursor).
        action_type is one of:
          'tl', 'tr', 'bl', 'br', 'tc', 'bc', 'lc', 'rc',
          'rotate', 'move', 'outside'
        """
        if not self.isVisible() or self._rect.isEmpty():
            return "outside", None

        inv_scale = self.get_view_inv_scale()
        # Generous hit radius on screen (16 screen pixels)
        handle_hit_r = max(10.0, 16.0 * inv_scale)
        line_hit_r = max(5.0, 8.0 * inv_scale)

        handles = self.get_handles()
        r = self._rect

        # 1. Handle hit test (corners & midpoints have top priority)
        for h_name, pt in handles.items():
            dist = math.hypot(pos.x() - pt.x(), pos.y() - pt.y())
            if dist <= handle_hit_r:
                if h_name in ("tl", "br"):
                    return h_name, QCursor(Qt.SizeFDiagCursor)
                elif h_name in ("tr", "bl"):
                    return h_name, QCursor(Qt.SizeBDiagCursor)
                elif h_name in ("tc", "bc"):
                    return h_name, QCursor(Qt.SizeVerCursor)
                elif h_name in ("lc", "rc"):
                    return h_name, QCursor(Qt.SizeHorCursor)

        # 2. Edge line hit test (effortless resizing on lines; takes priority over rotate)
        if r.left() - line_hit_r <= pos.x() <= r.right() + line_hit_r:
            if abs(pos.y() - r.top()) <= line_hit_r:
                return "tc", QCursor(Qt.SizeVerCursor)
            if abs(pos.y() - r.bottom()) <= line_hit_r:
                return "bc", QCursor(Qt.SizeVerCursor)
        if r.top() - line_hit_r <= pos.y() <= r.bottom() + line_hit_r:
            if abs(pos.x() - r.left()) <= line_hit_r:
                return "lc", QCursor(Qt.SizeHorCursor)
            if abs(pos.x() - r.right()) <= line_hit_r:
                return "rc", QCursor(Qt.SizeHorCursor)

        # 3. Inside crop box -> Move
        if r.contains(pos):
            return "move", QCursor(Qt.SizeAllCursor)

        # 4. Rotation zone: ONLY outside the box near the 4 corners (between 14 and 45 screen px)
        rot_min = max(10.0, 14.0 * inv_scale)
        rot_max = max(35.0, 48.0 * inv_scale)

        for corner_name in ("tl", "tr", "bl", "br"):
            c = handles[corner_name]
            dist = math.hypot(pos.x() - c.x(), pos.y() - c.y())
            if rot_min <= dist <= rot_max:
                return "rotate", get_rotation_cursor(corner_name)

        return "outside", QCursor(Qt.CrossCursor)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        if not self.isVisible() or self._rect.isEmpty():
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        r = self._rect
        w, h = r.width(), r.height()
        inv_scale = self.get_view_inv_scale()

        # 1. Dimmed / Greyed-out Outer Mask
        scene_rect = getattr(self, "_image_rect", QRectF())
        if scene_rect.isEmpty() and self.scene():
            scene_rect = self.scene().sceneRect()
        if not scene_rect.isEmpty():
            full_path = QPainterPath()
            full_path.addPolygon(self._rotated_image_polygon(scene_rect))
            crop_path = QPainterPath()
            crop_path.addRect(r)
            dimmed_path = full_path.subtracted(crop_path)
            # WLPG semi-transparent dark grey overlay (~48% opacity)
            painter.fillPath(dimmed_path, QColor(0, 0, 0, 122))

        # 2. Crop Rectangle Outer Border (White with subtle outer shadow)
        border_w = max(1.5, 1.8 * inv_scale)
        shadow_w = max(2.5, 3.2 * inv_scale)
        painter.setPen(QPen(QColor(20, 20, 20, 120), shadow_w, Qt.SolidLine))
        painter.drawRect(r)
        painter.setPen(QPen(QColor(255, 255, 255, 245), border_w, Qt.SolidLine))
        painter.drawRect(r)

        # 3. Rule of Thirds compositional grid (3x3)
        grid_min = 24.0 * inv_scale
        if w > grid_min and h > grid_min:
            grid_pen = QPen(QColor(255, 255, 255, 175), max(1.0, 1.0 * inv_scale), Qt.SolidLine)
            painter.setPen(grid_pen)
            # 2 vertical lines
            x1 = r.left() + w / 3.0
            x2 = r.left() + 2.0 * w / 3.0
            painter.drawLine(QPointF(x1, r.top()), QPointF(x1, r.bottom()))
            painter.drawLine(QPointF(x2, r.top()), QPointF(x2, r.bottom()))
            # 2 horizontal lines
            y1 = r.top() + h / 3.0
            y2 = r.top() + 2.0 * h / 3.0
            painter.drawLine(QPointF(r.left(), y1), QPointF(r.right(), y1))
            painter.drawLine(QPointF(r.left(), y2), QPointF(r.right(), y2))

        # 4. 8 Symmetrical Square Handles (WLPG style - 12 screen pixels across)
        hs = max(10.0, 12.0 * inv_scale)
        half_hs = hs / 2.0
        h_pen = QPen(QColor(30, 30, 30, 220), max(1.0, 1.2 * inv_scale))
        h_brush = QBrush(QColor(255, 255, 255, 255))
        painter.setPen(h_pen)
        painter.setBrush(h_brush)

        handles = self.get_handles()
        for pt in handles.values():
            painter.drawRect(QRectF(pt.x() - half_hs, pt.y() - half_hs, hs, hs))

        # 5. Photoshop-Style Live Pixel Resolution Badge (large & readable at any zoom)
        master_w = int(round(w * self._master_scale))
        master_h = int(round(h * self._master_scale))
        res_text = f"{master_w} x {master_h} px"

        badge_font = QFont("Segoe UI")
        badge_font.setPixelSize(int(round(max(11.0, 13.0 * inv_scale))))
        badge_font.setBold(True)
        painter.setFont(badge_font)
        fm = QFontMetrics(badge_font)
        text_w = fm.horizontalAdvance(res_text)
        text_h = fm.height()

        badge_padding_x = max(6.0, 8.0 * inv_scale)
        badge_padding_y = max(3.0, 4.0 * inv_scale)
        badge_w = text_w + badge_padding_x * 2
        badge_h = text_h + badge_padding_y * 2

        # Position inside top-left corner of crop rect (or above if small)
        badge_offset = max(6.0, 8.0 * inv_scale)
        badge_x = r.left() + badge_offset
        badge_y = r.top() + badge_offset
        if badge_y + badge_h > r.bottom() - badge_offset:
            badge_y = max(0.0, r.top() - badge_h - (4.0 * inv_scale))
        badge_rect = QRectF(badge_x, badge_y, badge_w, badge_h)

        badge_path = QPainterPath()
        badge_path.addRoundedRect(badge_rect, max(3.0, 4.0 * inv_scale), max(3.0, 4.0 * inv_scale))
        painter.fillPath(badge_path, QColor(20, 20, 20, 220))
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 90), max(1.0, 1.0 * inv_scale)))
        painter.drawPath(badge_path)

        painter.setPen(QColor(255, 255, 255, 255))
        painter.drawText(
            badge_rect,
            Qt.AlignCenter,
            res_text,
        )

        # 6. Photoshop-Style Live Rotation Angle Badge (e.g. -29.2°)
        if self._is_rotating or abs(self._rotation_deg) >= 0.05:
            deg_text = f"{self._rotation_deg:+.1f}°"
            deg_w = fm.horizontalAdvance(deg_text) + (16.0 * inv_scale)
            deg_h = text_h + (8.0 * inv_scale)

            # Near the active rotation handle or top-right
            if not self._rotation_badge_pos.isNull():
                rot_x = self._rotation_badge_pos.x() + (14.0 * inv_scale)
                rot_y = self._rotation_badge_pos.y() - deg_h - (4.0 * inv_scale)
            else:
                rot_x = r.right() - deg_w - (8.0 * inv_scale)
                rot_y = r.top() + (8.0 * inv_scale)

            rot_rect = QRectF(rot_x, rot_y, deg_w, deg_h)
            rot_path = QPainterPath()
            rot_path.addRoundedRect(rot_rect, max(3.0, 4.0 * inv_scale), max(3.0, 4.0 * inv_scale))
            painter.fillPath(rot_path, QColor(10, 10, 10, 235))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255, 90), max(1.0, 1.0 * inv_scale)))
            painter.drawPath(rot_path)

            painter.setPen(QColor(255, 255, 255, 255))
            painter.drawText(rot_rect, Qt.AlignCenter, deg_text)

        painter.restore()


class CropScene(QGraphicsScene):
    """
    Enhanced QGraphicsScene supporting authentic WLPG crop rectangle manipulation,
    rule of thirds grid, outer greyed-out overlay, 8 handles, and Photoshop-style rotation.
    """

    crop_rect_updated = Signal()
    crop_applied = Signal()
    crop_cancelled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.crop_rect_item = CropOverlayItem()
        self.addItem(self.crop_rect_item)
        self.crop_rect_item.hide()

        # Live rotation state
        self.crop_rotation_deg: float = 0.0
        self.pixmap_item: QGraphicsPixmapItem | None = None
        self.crop_enabled: bool = False
        self.allow_ai_expand: bool = False

        # Drag state
        self._is_dragging: bool = False
        self._drag_mode: str = "none"  # 'tl', 'tr', 'tc' etc, 'move', 'rotate', 'new'
        self._drag_start_pos: QPointF = QPointF()
        self._drag_start_rect: QRectF = QRectF()
        self._drag_start_angle: float = 0.0
        self._drag_initial_rot: float = 0.0

    def set_crop_mode(self, enabled: bool) -> None:
        self.crop_enabled = bool(enabled)
        if enabled:
            self.crop_rect_item.show()
        else:
            self.clear_crop_rect()

    def get_crop_rect(self) -> QRectF | None:
        return self.crop_rect_item.rect() if (self.crop_enabled and self.crop_rect_item.isVisible()) else None

    def clear_crop_rect(self) -> None:
        if self.crop_rect_item.isVisible():
            self.crop_rect_item.hide()
            self.crop_rotation_deg = 0.0
            self.crop_rect_item.set_rotation_deg(0.0)
            if self.pixmap_item:
                self.pixmap_item.setRotation(0.0)
            self.crop_rect_updated.emit()

    def set_allow_ai_expand(self, allow: bool) -> None:
        self.allow_ai_expand = bool(allow)
        if not self.allow_ai_expand and self.crop_rect_item.isVisible():
            self.crop_rect_item.setRect(self._clamp_rect_to_scene(self.crop_rect_item.rect()))
            self.crop_rect_item.update()
            self.update()
            for v in self.views():
                if v.viewport() is not None:
                    v.viewport().update()
            self.crop_rect_updated.emit()

    def _crop_bounds_rect(self) -> QRectF:
        s = self.sceneRect()
        if s.isEmpty():
            return s
        if getattr(self, "allow_ai_expand", False):
            mx = s.width() * 0.25
            my = s.height() * 0.25
            return s.adjusted(-mx, -my, mx, my)
        rot = abs(float(getattr(self, "crop_rotation_deg", 0.0)))
        if rot >= 0.05:
            rad = math.radians(rot)
            sin_a = math.sin(rad)
            cos_a = math.cos(rad)
            fw, fh = float(s.width()), float(s.height())
            factor = 1.0 / (cos_a + (fw / fh) * sin_a if fw > fh else cos_a + (fh / fw) * sin_a)
            in_w = max(16.0, fw * factor * 0.985)
            in_h = max(16.0, fh * factor * 0.985)
            c = s.center()
            return QRectF(c.x() - in_w / 2.0, c.y() - in_h / 2.0, in_w, in_h)
        return s

    def _clamp_rect_to_scene(self, r: QRectF) -> QRectF:
        s = self._crop_bounds_rect()
        if s.isEmpty():
            return r
        x = max(s.left(), min(r.left(), s.right() - 16.0))
        y = max(s.top(), min(r.top(), s.bottom() - 16.0))
        right = max(x + 16.0, min(r.right(), s.right()))
        bottom = max(y + 16.0, min(r.bottom(), s.bottom()))
        return QRectF(x, y, right - x, bottom - y)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self.crop_enabled:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.LeftButton:
            pos = event.scenePos()
            action, _ = self.crop_rect_item.hit_test(pos)

            self._is_dragging = True
            self._drag_start_pos = pos
            self._drag_start_rect = self.crop_rect_item.rect()

            if action in ("tl", "tr", "bl", "br", "tc", "bc", "lc", "rc"):
                self._drag_mode = action
            elif action == "move":
                self._drag_mode = "move"
            elif action == "rotate":
                self._drag_mode = "rotate"
                self.crop_rect_item._is_rotating = True
                center = self.crop_rect_item.rect().center()
                self._drag_start_angle = math.degrees(math.atan2(pos.y() - center.y(), pos.x() - center.x()))
                self._drag_initial_rot = self.crop_rotation_deg
            else:
                # Click outside: defer replacing existing crop until dragged past threshold
                self._drag_mode = "new"
                self._new_crop_started = False
                bounds_rect = self._crop_bounds_rect()
                clamped_x = max(bounds_rect.left(), min(pos.x(), bounds_rect.right()))
                clamped_y = max(bounds_rect.top(), min(pos.y(), bounds_rect.bottom()))
                self._drag_start_pos = QPointF(clamped_x, clamped_y)
                if not self.crop_rect_item.isVisible():
                    self.crop_rect_item.setRect(QRectF(self._drag_start_pos, self._drag_start_pos))
                    self.crop_rect_item.show()

            self.crop_rect_updated.emit()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self.crop_enabled:
            super().mouseMoveEvent(event)
            return

        pos = event.scenePos()

        # Update hover cursor when not actively dragging
        if not self._is_dragging:
            action, cur = self.crop_rect_item.hit_test(pos)
            if cur is not None and self.views():
                self.views()[0].viewport().setCursor(cur)
            elif self.views():
                self.views()[0].viewport().unsetCursor()
            super().mouseMoveEvent(event)
            return

        dx = pos.x() - self._drag_start_pos.x()
        dy = pos.y() - self._drag_start_pos.y()
        sr = self._drag_start_rect
        ratio = self.crop_rect_item.aspect_ratio()

        if self._drag_mode == "move":
            # Translate crop rectangle
            new_rect = sr.translated(dx, dy)
            # Clamp inside allowed bounds
            s = self._crop_bounds_rect()
            if not s.isEmpty():
                if new_rect.left() < s.left():
                    new_rect.moveLeft(s.left())
                if new_rect.top() < s.top():
                    new_rect.moveTop(s.top())
                if new_rect.right() > s.right():
                    new_rect.moveRight(s.right())
                if new_rect.bottom() > s.bottom():
                    new_rect.moveBottom(s.bottom())
            self.crop_rect_item.setRect(new_rect)
            self.crop_rect_updated.emit()

        elif self._drag_mode == "rotate":
            # Rotate crop / straighten angle
            center = self.crop_rect_item.rect().center()
            current_angle = math.degrees(math.atan2(pos.y() - center.y(), pos.x() - center.x()))
            angle_delta = current_angle - self._drag_start_angle
            # Clamp between -45.0 and +45.0 degrees
            rot = max(-45.0, min(self._drag_initial_rot + angle_delta, 45.0))
            self.crop_rotation_deg = rot
            self.crop_rect_item.set_rotation_deg(rot, badge_pos=pos)

            # Live hardware rotation on the background pixmap item (matches OpenCV CW direction for +rot)
            if self.pixmap_item is not None and self.sceneRect().isValid():
                s_center = self.sceneRect().center()
                self.pixmap_item.setTransformOriginPoint(s_center)
                self.pixmap_item.setRotation(rot)

            if not self.allow_ai_expand and self.crop_rect_item.isVisible():
                self.crop_rect_item.setRect(self._clamp_rect_to_scene(self.crop_rect_item.rect()))
                self.update()

            self.crop_rect_updated.emit()

        elif self._drag_mode == "new":
            if not getattr(self, "_new_crop_started", False):
                drag_thresh = max(8.0, 12.0 * self.crop_rect_item.get_view_inv_scale())
                if math.hypot(pos.x() - self._drag_start_pos.x(), pos.y() - self._drag_start_pos.y()) < drag_thresh:
                    event.accept()
                    return
                self._new_crop_started = True
            # Rubber-band drawing
            s = self._crop_bounds_rect()
            clamped_x = max(s.left(), min(pos.x(), s.right()))
            clamped_y = max(s.top(), min(pos.y(), s.bottom()))
            new_rect = QRectF(self._drag_start_pos, QPointF(clamped_x, clamped_y)).normalized()
            if ratio is not None and ratio > 0 and new_rect.width() > 0:
                h = new_rect.width() / ratio
                new_rect.setHeight(h)
            self.crop_rect_item.setRect(self._clamp_rect_to_scene(new_rect))
            self.crop_rect_updated.emit()

        elif self._drag_mode in ("tl", "tr", "bl", "br", "tc", "bc", "lc", "rc"):
            # Resizing edges and corners
            left, top, right, bottom = sr.left(), sr.top(), sr.right(), sr.bottom()

            if "l" in self._drag_mode:
                left = min(sr.right() - 16.0, sr.left() + dx)
            if "r" in self._drag_mode:
                right = max(sr.left() + 16.0, sr.right() + dx)
            if "t" in self._drag_mode:
                top = min(sr.bottom() - 16.0, sr.top() + dy)
            if "b" in self._drag_mode:
                bottom = max(sr.top() + 16.0, sr.bottom() + dy)

            new_w = right - left
            new_h = bottom - top

            # Enforce aspect ratio preset if active
            if ratio is not None and ratio > 0:
                if self._drag_mode in ("tc", "bc"):
                    new_w = new_h * ratio
                    cx = (left + right) / 2.0
                    left = cx - new_w / 2.0
                    right = cx + new_w / 2.0
                elif self._drag_mode in ("lc", "rc"):
                    new_h = new_w / ratio
                    cy = (top + bottom) / 2.0
                    top = cy - new_h / 2.0
                    bottom = cy + new_h / 2.0
                elif self._drag_mode in ("tl", "br"):
                    new_h = new_w / ratio
                    if "t" in self._drag_mode:
                        top = bottom - new_h
                    else:
                        bottom = top + new_h
                elif self._drag_mode in ("tr", "bl"):
                    new_h = new_w / ratio
                    if "t" in self._drag_mode:
                        top = bottom - new_h
                    else:
                        bottom = top + new_h

            res_rect = QRectF(left, top, right - left, bottom - top)
            self.crop_rect_item.setRect(self._clamp_rect_to_scene(res_rect))
            self.crop_rect_updated.emit()

        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if not self.crop_enabled:
            super().mouseReleaseEvent(event)
            return

        if event.button() == Qt.LeftButton and self._is_dragging:
            was_new_unstarted = (
                self._drag_mode == "new" and not getattr(self, "_new_crop_started", False)
            )
            self._is_dragging = False
            self.crop_rect_item._is_rotating = False
            self._drag_mode = "none"
            self.crop_rect_item.update()
            if not was_new_unstarted:
                self.crop_rect_updated.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            pos = event.scenePos()
            if self.crop_rect_item.isVisible() and self.crop_rect_item.rect().contains(pos):
                self.crop_applied.emit()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.crop_applied.emit()
            event.accept()
            return
        elif event.key() == Qt.Key_Escape:
            self.crop_cancelled.emit()
            event.accept()
            return
        super().keyPressEvent(event)
