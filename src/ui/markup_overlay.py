"""
Interactive Quick Markup, Callouts & Privacy Redaction Tool for Essentials Photo Gallery.
1. Freehand Pen & Fluorescent Highlighter with variable stroke widths.
2. Real-time WYSIWYG Vector Arrows and Rounded Rectangle Callouts.
3. Interactive Circular Magnifier Loupe (2.5x zoom) with glass chrome ring.
4. Live Privacy Redaction (Gaussian blur & Mosaic pixelation) with instant preview.
5. Draggable Text Labels with glassmorphic pill background.
6. Color swatches palette and Photoshop bracket shortcuts ([ and ]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
import cv2
import numpy as np

from PySide6.QtCore import Qt, QPoint, QPointF, QRect, QRectF, QSize, Signal
from PySide6.QtGui import (
    QColor,
    QPen,
    QBrush,
    QPainter,
    QPainterPath,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QPolygonF,
    QCursor,
    QImage,
    QPixmap,
    QIcon,
    QMouseEvent,
    QAction,
    QTransform,
)
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QToolButton,
    QSlider,
    QColorDialog,
    QInputDialog,
    QLabel,
    QFrame,
    QButtonGroup,
    QMenu,
)

from src.imaging.creative_effects import apply_pixelate, apply_blur_redact
from src.ui.ribbon_toolbar import make_wlpg_icon, make_symbol_icon, ARROW_DROPDOWN_PATH


_ROUND_ROTATION_CURSOR: QCursor | None = None
_SELECT_TOOL_CURSOR: QCursor | None = None


def make_select_tool_cursor() -> QCursor:
    """Returns a cached HiDPI precision dark-filled selection pointer cursor distinct from the OS arrow."""
    global _SELECT_TOOL_CURSOR
    if _SELECT_TOOL_CURSOR is not None:
        return _SELECT_TOOL_CURSOR

    pm = QPixmap(56, 56)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.scale(2.0, 2.0)

    arrow = QPolygonF([
        QPointF(2.5, 2.0),
        QPointF(2.5, 20.5),
        QPointF(7.2, 16.0),
        QPointF(11.0, 23.2),
        QPointF(13.8, 21.8),
        QPointF(10.0, 14.6),
        QPointF(16.4, 14.2),
    ])
    # Bold white outer halo for visibility over dark photos
    p.setPen(QPen(QColor(255, 255, 255, 245), 2.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.setBrush(QColor(255, 255, 255, 245))
    p.drawPolygon(arrow)

    # Crisp dark-slate / royal-blue Photoshop-style Move/Select pointer
    p.setPen(QPen(QColor("#0F172A"), 1.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.setBrush(QColor("#1E3A8A"))
    p.drawPolygon(arrow)

    # Subtle 4-way move indicator badge at bottom-right
    mx, my = 20.5, 20.5
    p.setPen(QPen(QColor(255, 255, 255, 230), 2.6, Qt.SolidLine, Qt.RoundCap))
    p.drawLine(QPointF(mx - 3.6, my), QPointF(mx + 3.6, my))
    p.drawLine(QPointF(mx, my - 3.6), QPointF(mx, my + 3.6))
    p.setPen(QPen(QColor("#1D4ED8"), 1.35, Qt.SolidLine, Qt.RoundCap))
    p.drawLine(QPointF(mx - 3.6, my), QPointF(mx + 3.6, my))
    p.drawLine(QPointF(mx, my - 3.6), QPointF(mx, my + 3.6))

    p.end()
    pm.setDevicePixelRatio(2.0)
    _SELECT_TOOL_CURSOR = QCursor(pm, 2, 2)
    return _SELECT_TOOL_CURSOR


def make_round_rotation_cursor() -> QCursor:
    """Renders a crisp round circular clockwise arrow cursor for markup rotation handles."""
    global _ROUND_ROTATION_CURSOR
    if _ROUND_ROTATION_CURSOR is not None:
        return _ROUND_ROTATION_CURSOR

    pm = QPixmap(56, 56)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)

    cx, cy = 28.0, 28.0
    r = 14.5
    arc_rect = QRectF(cx - r, cy - r, r * 2.0, r * 2.0)

    # Arc from ~40 deg sweeping -285 deg clockwise (in Qt 1/16 deg units)
    start_qt = int(45 * 16)
    span_qt = int(-280 * 16)
    end_rad = math.radians(45 - 280)  # Qt counter-clockwise from +X -> y inverted below
    # Convert Qt angle a_qt to screen coordinates: (cx + r*cos(a), cy - r*sin(a))
    end_angle = math.radians(45.0 - 280.0)
    tip_anchor = QPointF(cx + r * math.cos(end_angle), cy - r * math.sin(end_angle))
    # Clockwise tangent at end_angle: (sin(a), cos(a)) -> check direction toward arrowhead
    tx = math.sin(end_angle)
    ty = math.cos(end_angle)
    nx = math.cos(end_angle)
    ny = -math.sin(end_angle)

    tip = QPointF(tip_anchor.x() + tx * 6.5, tip_anchor.y() + ty * 6.5)
    wing_out = QPointF(tip_anchor.x() - tx * 3.0 + nx * 6.2, tip_anchor.y() - ty * 3.0 + ny * 6.2)
    wing_in = QPointF(tip_anchor.x() - tx * 3.0 - nx * 5.8, tip_anchor.y() - ty * 3.0 - ny * 5.8)
    head_poly = QPolygonF([tip, wing_out, wing_in])

    # White outer halo for high contrast on both dark & bright photos
    halo_pen = QPen(QColor(255, 255, 255, 245), 6.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(halo_pen)
    p.setBrush(QColor(255, 255, 255, 245))
    p.drawArc(arc_rect, start_qt, span_qt)
    p.drawPolygon(head_poly)

    # Crisp dark slate core + subtle center pivot dot
    core_pen = QPen(QColor(15, 23, 42), 3.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(core_pen)
    p.setBrush(QColor(15, 23, 42))
    p.drawArc(arc_rect, start_qt, span_qt)
    p.setPen(QPen(QColor(15, 23, 42), 1.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.drawPolygon(head_poly)

    p.setPen(QPen(QColor(255, 255, 255, 220), 1.4))
    p.setBrush(QColor(59, 130, 246))
    p.drawEllipse(QPointF(cx, cy), 3.2, 3.2)

    p.end()
    pm.setDevicePixelRatio(2.0)
    _ROUND_ROTATION_CURSOR = QCursor(pm, 14, 14)
    return _ROUND_ROTATION_CURSOR


CURATED_MARKUP_FONTS: tuple[str, ...] = (
    "Segoe UI",
    "Arial",
    "Calibri",
    "Verdana",
    "Trebuchet MS",
    "Georgia",
    "Times New Roman",
    "Consolas",
    "Comic Sans MS",
    "Impact",
)


def get_markup_font_families() -> list[str]:
    """
    Returns curated photographic/callout font families first (when available),
    followed by remaining installed system font families (excluding '@' vertical CJK aliases).
    """
    try:
        installed = [f for f in QFontDatabase.families() if f and not f.startswith("@")]
    except Exception:
        return list(CURATED_MARKUP_FONTS)

    if not installed:
        return list(CURATED_MARKUP_FONTS)

    installed_lower = {f.lower(): f for f in installed}
    ordered: list[str] = []
    seen: set[str] = set()

    for fam in CURATED_MARKUP_FONTS:
        matched = installed_lower.get(fam.lower())
        if matched and matched.lower() not in seen:
            ordered.append(matched)
            seen.add(matched.lower())

    for fam in installed:
        low = fam.lower()
        if low not in seen:
            ordered.append(fam)
            seen.add(low)

    return ordered or list(CURATED_MARKUP_FONTS)


ALL_SHAPE_KINDS: tuple[str, ...] = (
    "rect",
    "ellipse",
    "triangle",
    "pentagon",
    "hexagon",
    "star",
    "heart_shape",
    "speech_bubble",
)
ALL_LINE_KINDS: tuple[str, ...] = ("line", "arrow", "double_arrow")
ALL_STICKER_IDS: tuple[str, ...] = ("heart", "smiley", "star", "fire", "check")


@dataclass
class MarkupElement:
    kind: str  # 'pen', 'highlighter', 'line', 'arrow', 'double_arrow', shape kinds, 'pixelate', 'blur', 'sticker', 'magnifier', 'text'
    rect: QRectF | None = None
    start_pt: QPointF | None = None
    end_pt: QPointF | None = None
    points: list[QPointF] = field(default_factory=list)
    color: QColor = field(default_factory=lambda: QColor(230, 40, 40))
    bg_color: QColor | None = None
    stroke_width: int = 16
    text: str = ""
    zoom_factor: float = 2.5
    font_family: str = "Segoe UI"
    font_bold: bool = True
    font_italic: bool = False
    rotation_deg: float = 0.0
    text_style: str = "pill"   # 'pill', 'outline', 'shadow', 'plain'
    text_align: str = "left"   # 'left', 'center', 'right'
    opacity: float = 1.0       # 0.15 .. 1.0
    bg_opacity: float = 0.85   # 0.0 .. 1.0
    shape_filled: bool = False
    line_style: str = "arrow"  # 'line', 'arrow', 'double_arrow'
    dash_style: str = "solid"  # 'solid', 'dashed', 'dotted'
    sticker_id: str = ""       # 'heart', 'smiley', 'star', 'fire', 'check'
    obscure_mode: str = "brush"  # 'brush', 'box'
    obscure_strength: int = 55   # 15 .. 100


def build_arrow_geometry(p1: QPointF, p2: QPointF, stroke_width: int = 4) -> tuple[QPointF, QPolygonF]:
    """
    Computes arrowhead polygon and shortened shaft endpoint to create a clean,
    gapless vector arrow pointing from p1 to p2.
    """
    dx = p2.x() - p1.x()
    dy = p2.y() - p1.y()
    length = math.hypot(dx, dy)
    if length < 4.0:
        return p2, QPolygonF([p2, p2, p2])

    angle = math.atan2(dy, dx)
    arrow_size = max(14.0, stroke_width * 3.4)

    left_angle = angle + math.pi * 0.82
    right_angle = angle - math.pi * 0.82

    p_left = QPointF(p2.x() + math.cos(left_angle) * arrow_size, p2.y() + math.sin(left_angle) * arrow_size)
    p_right = QPointF(p2.x() + math.cos(right_angle) * arrow_size, p2.y() + math.sin(right_angle) * arrow_size)

    # Base center of the arrowhead where the shaft terminates
    p_base = QPointF((p_left.x() + p_right.x()) * 0.5, (p_left.y() + p_right.y()) * 0.5)
    arrowhead = QPolygonF([p2, p_left, p_right])
    return p_base, arrowhead


def build_star_polygon_in_rect(rect: QRectF) -> QPolygonF:
    """Constructs a clean 5-pointed star polygon inscribed in the given bounding rectangle."""
    cx = rect.center().x()
    cy = rect.center().y()
    outer_r = max(2.0, min(rect.width(), rect.height()) / 2.0)
    inner_r = outer_r * 0.42
    points: list[QPointF] = []
    for i in range(10):
        angle_deg = -90.0 + (i * 36.0)
        rad = math.radians(angle_deg)
        r = outer_r if i % 2 == 0 else inner_r
        points.append(
            QPointF(cx + r * math.cos(rad), cy + r * math.sin(rad))
        )
    return QPolygonF(points)


def build_regular_polygon_in_rect(rect: QRectF, sides: int, start_deg: float = -90.0) -> QPolygonF:
    """Constructs a regular N-gon (triangle, pentagon, hexagon) inscribed in *rect*."""
    rx = max(2.0, rect.width() / 2.0)
    ry = max(2.0, rect.height() / 2.0)
    cx, cy = rect.center().x(), rect.center().y()
    n = max(3, int(sides))
    pts: list[QPointF] = []
    for i in range(n):
        rad = math.radians(start_deg + (360.0 * i) / n)
        pts.append(QPointF(cx + rx * math.cos(rad), cy + ry * math.sin(rad)))
    return QPolygonF(pts)


def build_heart_path_in_rect(rect: QRectF) -> QPainterPath:
    """Constructs a balanced symmetric vector heart path inscribed in *rect*."""
    x, y, w, h = rect.x(), rect.y(), max(4.0, rect.width()), max(4.0, rect.height())
    path = QPainterPath()
    path.moveTo(x + w * 0.50, y + h * 0.90)
    path.cubicTo(x + w * 0.08, y + h * 0.62, x + w * 0.02, y + h * 0.28, x + w * 0.26, y + h * 0.14)
    path.cubicTo(x + w * 0.40, y + h * 0.06, x + w * 0.50, y + h * 0.22, x + w * 0.50, y + h * 0.28)
    path.cubicTo(x + w * 0.50, y + h * 0.22, x + w * 0.60, y + h * 0.06, x + w * 0.74, y + h * 0.14)
    path.cubicTo(x + w * 0.98, y + h * 0.28, x + w * 0.92, y + h * 0.62, x + w * 0.50, y + h * 0.90)
    path.closeSubpath()
    return path


def build_speech_bubble_path_in_rect(rect: QRectF) -> QPainterPath:
    """Constructs a clean rounded speech-bubble callout path with a bottom-left pointer tail."""
    x, y, w, h = rect.x(), rect.y(), max(4.0, rect.width()), max(4.0, rect.height())
    body_h = h * 0.76
    rad = min(14.0, w * 0.18, body_h * 0.28)
    tail_x1 = x + w * 0.22
    tail_x2 = x + w * 0.36
    tail_tip = QPointF(x + w * 0.14, y + h)

    path = QPainterPath()
    path.moveTo(x + rad, y)
    path.lineTo(x + w - rad, y)
    path.quadTo(x + w, y, x + w, y + rad)
    path.lineTo(x + w, y + body_h - rad)
    path.quadTo(x + w, y + body_h, x + w - rad, y + body_h)
    path.lineTo(tail_x2, y + body_h)
    path.lineTo(tail_tip)
    path.lineTo(tail_x1, y + body_h)
    path.lineTo(x + rad, y + body_h)
    path.quadTo(x, y + body_h, x, y + body_h - rad)
    path.lineTo(x, y + rad)
    path.quadTo(x, y, x + rad, y)
    path.closeSubpath()
    return path


def draw_vector_sticker(
    painter: QPainter,
    rect: QRectF,
    sticker_id: str,
    *,
    rotation_deg: float = 0.0,
    opacity: float = 1.0,
) -> None:
    """Renders crisp resolution-independent social vector stickers with glossy shading inside *rect*."""
    if rect.isEmpty():
        return
    from PySide6.QtGui import QLinearGradient, QRadialGradient

    sid = (sticker_id or "heart").strip().lower()
    painter.save()
    painter.setOpacity(max(0.15, min(1.0, float(opacity))))
    if abs(float(rotation_deg)) >= 0.05:
        c = rect.center()
        painter.translate(c.x(), c.y())
        painter.rotate(float(rotation_deg))
        painter.translate(-c.x(), -c.y())

    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
    edge_w = max(1.4, min(w, h) * 0.032)
    shadow_off = max(1.0, min(w, h) * 0.04)

    if sid == "heart":
        hp_shadow = build_heart_path_in_rect(rect.translated(0.0, shadow_off))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 48))
        painter.drawPath(hp_shadow)

        hp = build_heart_path_in_rect(rect)
        hgrad = QRadialGradient(QPointF(x + w * 0.34, y + h * 0.28), max(w, h) * 0.78)
        hgrad.setColorAt(0.0, QColor("#FDA4AF"))
        hgrad.setColorAt(0.28, QColor("#F43F5E"))
        hgrad.setColorAt(0.78, QColor("#E11D48"))
        hgrad.setColorAt(1.0, QColor("#9F1239"))
        painter.setPen(QPen(QColor("#881337"), edge_w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(hgrad)
        painter.drawPath(hp)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 135))
        painter.drawEllipse(QRectF(x + w * 0.19, y + h * 0.20, w * 0.17, h * 0.10))

    elif sid == "smiley":
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 46))
        painter.drawEllipse(rect.translated(0.0, shadow_off))

        sgrad = QRadialGradient(QPointF(x + w * 0.35, y + h * 0.28), max(w, h) * 0.72)
        sgrad.setColorAt(0.0, QColor("#FEF08A"))
        sgrad.setColorAt(0.45, QColor("#FACC15"))
        sgrad.setColorAt(1.0, QColor("#D97706"))
        painter.setPen(QPen(QColor("#B45309"), edge_w))
        painter.setBrush(sgrad)
        painter.drawEllipse(rect)

        # Top gloss arc & rosy cheeks
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 95))
        painter.drawEllipse(QRectF(x + w * 0.24, y + h * 0.10, w * 0.44, h * 0.16))
        painter.setBrush(QColor(244, 63, 94, 72))
        painter.drawEllipse(QRectF(x + w * 0.14, y + h * 0.48, w * 0.18, h * 0.13))
        painter.drawEllipse(QRectF(x + w * 0.68, y + h * 0.48, w * 0.18, h * 0.13))

        # Expressive eyes with tiny catchlights
        painter.setBrush(QColor("#1E293B"))
        ew, eh = w * 0.115, h * 0.155
        painter.drawEllipse(QRectF(x + w * 0.28, y + h * 0.30, ew, eh))
        painter.drawEllipse(QRectF(x + w * 0.605, y + h * 0.30, ew, eh))
        painter.setBrush(QColor(255, 255, 255, 210))
        painter.drawEllipse(QRectF(x + w * 0.30, y + h * 0.32, ew * 0.38, eh * 0.34))
        painter.drawEllipse(QRectF(x + w * 0.625, y + h * 0.32, ew * 0.38, eh * 0.34))

        painter.setPen(QPen(QColor("#1E293B"), max(2.2, min(w, h) * 0.068), Qt.SolidLine, Qt.RoundCap))
        painter.setBrush(Qt.NoBrush)
        smile_rect = QRectF(x + w * 0.25, y + h * 0.34, w * 0.50, h * 0.40)
        painter.drawArc(smile_rect, 205 * 16, 130 * 16)

    elif sid == "star":
        sp_shadow = build_star_polygon_in_rect(rect.translated(0.0, shadow_off))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 46))
        painter.drawPolygon(sp_shadow)

        sp = build_star_polygon_in_rect(rect)
        stgrad = QLinearGradient(QPointF(x, y), QPointF(x + w * 0.3, y + h))
        stgrad.setColorAt(0.0, QColor("#FEF08A"))
        stgrad.setColorAt(0.45, QColor("#FBBF24"))
        stgrad.setColorAt(1.0, QColor("#D97706"))
        painter.setPen(QPen(QColor("#92400E"), edge_w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(stgrad)
        painter.drawPolygon(sp)

        # Subtle inner facet highlight
        painter.setPen(QPen(QColor(255, 255, 255, 130), max(1.0, edge_w * 0.65), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(QPointF(x + w * 0.50, y + h * 0.14), QPointF(x + w * 0.42, y + h * 0.42))

    elif sid == "fire":
        flame_s = QPainterPath()
        flame_s.moveTo(x + w * 0.50, y + h * 0.05 + shadow_off)
        flame_s.cubicTo(x + w * 0.85, y + h * 0.32 + shadow_off, x + w * 0.94, y + h * 0.64 + shadow_off, x + w * 0.78, y + h * 0.86 + shadow_off)
        flame_s.cubicTo(x + w * 0.66, y + h * 0.98 + shadow_off, x + w * 0.34, y + h * 0.98 + shadow_off, x + w * 0.22, y + h * 0.86 + shadow_off)
        flame_s.cubicTo(x + w * 0.06, y + h * 0.64 + shadow_off, x + w * 0.16, y + h * 0.36 + shadow_off, x + w * 0.36, y + h * 0.22 + shadow_off)
        flame_s.cubicTo(x + w * 0.36, y + h * 0.38 + shadow_off, x + w * 0.48, y + h * 0.36 + shadow_off, x + w * 0.50, y + h * 0.05 + shadow_off)
        flame_s.closeSubpath()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 44))
        painter.drawPath(flame_s)

        flame = QPainterPath()
        flame.moveTo(x + w * 0.50, y + h * 0.05)
        flame.cubicTo(x + w * 0.85, y + h * 0.32, x + w * 0.94, y + h * 0.64, x + w * 0.78, y + h * 0.86)
        flame.cubicTo(x + w * 0.66, y + h * 0.98, x + w * 0.34, y + h * 0.98, x + w * 0.22, y + h * 0.86)
        flame.cubicTo(x + w * 0.06, y + h * 0.64, x + w * 0.16, y + h * 0.36, x + w * 0.36, y + h * 0.22)
        flame.cubicTo(x + w * 0.36, y + h * 0.38, x + w * 0.48, y + h * 0.36, x + w * 0.50, y + h * 0.05)
        flame.closeSubpath()
        fgrad = QLinearGradient(QPointF(x + w * 0.5, y), QPointF(x + w * 0.5, y + h))
        fgrad.setColorAt(0.0, QColor("#FB7185"))
        fgrad.setColorAt(0.35, QColor("#F97316"))
        fgrad.setColorAt(1.0, QColor("#DC2626"))
        painter.setPen(QPen(QColor("#9A3412"), edge_w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(fgrad)
        painter.drawPath(flame)

        inner = QPainterPath()
        inner.moveTo(x + w * 0.52, y + h * 0.36)
        inner.cubicTo(x + w * 0.74, y + h * 0.54, x + w * 0.72, y + h * 0.86, x + w * 0.50, y + h * 0.90)
        inner.cubicTo(x + w * 0.28, y + h * 0.86, x + w * 0.28, y + h * 0.54, x + w * 0.52, y + h * 0.36)
        igrad = QLinearGradient(QPointF(x + w * 0.5, y + h * 0.36), QPointF(x + w * 0.5, y + h * 0.9))
        igrad.setColorAt(0.0, QColor("#FEF9C3"))
        igrad.setColorAt(1.0, QColor("#FACC15"))
        painter.setPen(Qt.NoPen)
        painter.setBrush(igrad)
        painter.drawPath(inner)

    else:  # check
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 44))
        painter.drawEllipse(rect.translated(0.0, shadow_off))

        cgrad = QRadialGradient(QPointF(x + w * 0.35, y + h * 0.28), max(w, h) * 0.78)
        cgrad.setColorAt(0.0, QColor("#86EFAC"))
        cgrad.setColorAt(0.35, QColor("#22C55E"))
        cgrad.setColorAt(1.0, QColor("#15803D"))
        painter.setPen(QPen(QColor("#14532D"), edge_w))
        painter.setBrush(cgrad)
        painter.drawEllipse(rect)

        painter.setPen(QPen(QColor("#FFFFFF"), max(2.6, min(w, h) * 0.125), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        chk = QPainterPath()
        chk.moveTo(x + w * 0.27, y + h * 0.52)
        chk.lineTo(x + w * 0.44, y + h * 0.69)
        chk.lineTo(x + w * 0.74, y + h * 0.33)
        painter.drawPath(chk)

    painter.restore()


def compute_circle_tangent_lines(
    c1: QPointF, r1: float, c2: QPointF, r2: float
) -> list[tuple[QPointF, QPointF]]:
    """
    Computes external tangent lines connecting circle 1 (c1, r1) to circle 2 (c2, r2).
    Returns [(p1_top, p2_top), (p1_bottom, p2_bottom)] for callout connecting rays.
    """
    dx = c2.x() - c1.x()
    dy = c2.y() - c1.y()
    d = math.hypot(dx, dy)
    if d <= abs(r2 - r1) or d < 1e-4:
        return []

    base_angle = math.atan2(dy, dx)
    sin_phi = max(-1.0, min(1.0, (r2 - r1) / d))
    phi = math.asin(sin_phi)

    # Tangent 1
    ang1 = base_angle + (math.pi / 2.0) + phi
    p1_a = QPointF(c1.x() + r1 * math.cos(ang1), c1.y() + r1 * math.sin(ang1))
    p2_a = QPointF(c2.x() + r2 * math.cos(ang1), c2.y() + r2 * math.sin(ang1))

    # Tangent 2
    ang2 = base_angle - (math.pi / 2.0) - phi
    p1_b = QPointF(c1.x() + r1 * math.cos(ang2), c1.y() + r1 * math.sin(ang2))
    p2_b = QPointF(c2.x() + r2 * math.cos(ang2), c2.y() + r2 * math.sin(ang2))

    return [(p1_a, p2_a), (p1_b, p2_b)]


def create_magnifier_pixmap(
    base_rgb: np.ndarray,
    center_pt: QPointF,
    radius: float,
    zoom_factor: float = 2.5,
    ring_color: QColor | None = None,
) -> QPixmap:
    """
    Synthesizes a circular magnified loupe callout pixmap with anti-aliasing,
    glass chrome border, and custom accent ring directly from the base RGB array.
    """
    h, w = base_rgb.shape[:2]
    cx, cy = center_pt.x(), center_pt.y()
    crop_rad = radius / max(1.0, zoom_factor)

    src_x1 = max(0, int(round(cx - crop_rad)))
    src_y1 = max(0, int(round(cy - crop_rad)))
    src_x2 = min(w, int(round(cx + crop_rad)))
    src_y2 = min(h, int(round(cy + crop_rad)))

    target_d = int(round(radius * 2.0))
    canvas_size = target_d + 12
    pm = QPixmap(canvas_size, canvas_size)
    pm.fill(Qt.transparent)

    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return pm

    patch_rgb = np.ascontiguousarray(base_rgb[src_y1:src_y2, src_x1:src_x2])
    patch_h, patch_w = patch_rgb.shape[:2]
    qimg_patch = QImage(
        patch_rgb.data, patch_w, patch_h, 3 * patch_w, QImage.Format.Format_RGB888
    ).copy()
    scaled_patch = qimg_patch.scaled(
        target_d, target_d, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
    )

    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    center = canvas_size / 2.0

    # Antialiased circular fill via textured QBrush (avoids Qt's jagged integer setClipPath)
    brush = QBrush(scaled_patch)
    b_tr = QTransform()
    b_tr.translate(center - radius, center - radius)
    if scaled_patch.width() > 0 and scaled_patch.height() > 0:
        b_tr.scale((radius * 2.0) / float(scaled_patch.width()), (radius * 2.0) / float(scaled_patch.height()))
    brush.setTransform(b_tr)
    painter.setPen(Qt.NoPen)
    painter.setBrush(brush)
    painter.drawEllipse(QPointF(center, center), radius, radius)

    accent_col = ring_color if (ring_color is not None and ring_color.isValid()) else QColor(50, 130, 230)

    # Outer glass chrome ring
    painter.setPen(QPen(accent_col, 3.0))
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(QPointF(center, center), radius, radius)

    # Inner subtle highlight ring
    painter.setPen(QPen(QColor(255, 255, 255, 210), 1.2))
    painter.drawEllipse(QPointF(center, center), max(1.0, radius - 1.8), max(1.0, radius - 1.8))
    painter.end()

    return pm


def _measure_text_pill_size(el: MarkupElement) -> tuple[float, float, list[str], QFont, QFontMetrics]:
    """Computes consistent text box width/height, lines, QFont, and QFontMetrics."""
    fam = getattr(el, "font_family", "Segoe UI") or "Segoe UI"
    font = QFont(fam, max(10, int(round(el.stroke_width * 2.2))))
    font.setBold(getattr(el, "font_bold", True))
    font.setItalic(getattr(el, "font_italic", False))
    fm = QFontMetrics(font)
    raw_text = (el.text or "").replace("\r\n", "\n").replace("\u2028", "\n")
    lines = raw_text.split("\n")
    max_line_w = max((fm.horizontalAdvance(line) for line in lines), default=40)
    line_h = fm.lineSpacing()
    total_text_h = line_h * max(1, len(lines))
    pad_x, pad_y = 12.0, 8.0
    natural_w = float(max_line_w + 2.0 * pad_x)
    rect_w = float(el.rect.normalized().width()) if el.rect is not None else 0.0
    box_w = max(natural_w, rect_w)
    return box_w, float(total_text_h + 2.0 * pad_y), lines, font, fm


def _rotated_rect_bounding_box(rect: QRectF, angle_deg: float) -> QRectF:
    """Returns the axis-aligned bounding rect of *rect* rotated around its center by *angle_deg*."""
    if abs(float(angle_deg)) < 1e-4:
        return rect
    c = rect.center()
    tr = QTransform()
    tr.translate(c.x(), c.y())
    tr.rotate(float(angle_deg))
    tr.translate(-c.x(), -c.y())
    return tr.mapRect(rect)


def _pen_style_from_dash(dash_style: str | None) -> Qt.PenStyle:
    norm = str(dash_style or "solid").strip().lower()
    if norm == "dashed":
        return Qt.DashLine
    if norm == "dotted":
        return Qt.DotLine
    return Qt.SolidLine


def get_markup_element_unrotated_rect(el: MarkupElement | None) -> QRectF:
    """Returns the element's unrotated scene geometry rectangle (pill rect for text, normalized rect for shapes/stickers/lines)."""
    if el is None:
        return QRectF()
    if el.kind == "text" and el.rect is not None:
        if el.text:
            pill_w, pill_h, _, _, _ = _measure_text_pill_size(el)
            return QRectF(el.rect.x(), el.rect.y(), pill_w, pill_h)
        return el.rect.normalized()
    if el.kind in ALL_LINE_KINDS and el.start_pt is not None and el.end_pt is not None:
        x1 = min(el.start_pt.x(), el.end_pt.x())
        y1 = min(el.start_pt.y(), el.end_pt.y())
        x2 = max(el.start_pt.x(), el.end_pt.x())
        y2 = max(el.start_pt.y(), el.end_pt.y())
        w = x2 - x1
        h = y2 - y1
        if w < 16.0:
            cx = (x1 + x2) * 0.5
            x1, x2 = cx - 8.0, cx + 8.0
        if h < 16.0:
            cy = (y1 + y2) * 0.5
            y1, y2 = cy - 8.0, cy + 8.0
        return QRectF(x1, y1, x2 - x1, y2 - y1)
    if el.rect is not None:
        return el.rect.normalized()
    return QRectF()


def scale_markup_element(el: MarkupElement | None, new_rect: QRectF) -> None:
    """Resizes a shape/sticker/line/redaction rect or scales a text element's font size proportionally to *new_rect*."""
    if el is None or new_rect is None:
        return
    nr = new_rect.normalized()
    if nr.width() < 8.0 or nr.height() < 8.0:
        return
    if el.kind == "text":
        old_r = get_markup_element_unrotated_rect(el)
        if old_r.height() > 1.0:
            ratio = max(0.25, nr.height() / old_r.height())
            new_sw = max(2, min(48, int(round(el.stroke_width * ratio))))
            el.stroke_width = new_sw
        el.rect = QRectF(nr.x(), nr.y(), nr.width(), nr.height())
        # Enforce minimum text pill width while letting users stretch the box horizontally for left/center/right align
        adj = get_markup_element_unrotated_rect(el)
        el.rect = QRectF(nr.x(), nr.y(), adj.width(), adj.height())
    elif el.kind in ALL_LINE_KINDS and el.start_pt is not None and el.end_pt is not None:
        old_r = get_markup_element_unrotated_rect(el)
        ow = max(1.0, old_r.width())
        oh = max(1.0, old_r.height())
        sx1 = (el.start_pt.x() - old_r.left()) / ow
        sy1 = (el.start_pt.y() - old_r.top()) / oh
        sx2 = (el.end_pt.x() - old_r.left()) / ow
        sy2 = (el.end_pt.y() - old_r.top()) / oh
        el.start_pt = QPointF(nr.left() + sx1 * nr.width(), nr.top() + sy1 * nr.height())
        el.end_pt = QPointF(nr.left() + sx2 * nr.width(), nr.top() + sy2 * nr.height())
    elif el.kind in ALL_SHAPE_KINDS or el.kind in ("pixelate", "blur", "sticker"):
        el.rect = nr


def _draw_text_element(painter: QPainter, el: MarkupElement) -> None:
    """Shared WYSIWYG renderer for text elements across live preview and final master bake."""
    if el.rect is None or not el.text:
        return
    pill_w, pill_h, lines, font, metrics = _measure_text_pill_size(el)
    rx, ry = el.rect.x(), el.rect.y()
    pad_x, pad_y = 12.0, 8.0
    max_line_w = max(1.0, pill_w - 2.0 * pad_x)
    line_h = metrics.lineSpacing()
    pill_rect = QRectF(rx, ry, pill_w, pill_h)
    style = str(getattr(el, "text_style", "pill") or "pill").lower()
    align = str(getattr(el, "text_align", "left") or "left").lower()
    sec_col = getattr(el, "bg_color", None)
    bg_op = max(0.0, min(1.0, float(getattr(el, "bg_opacity", 0.85) if getattr(el, "bg_opacity", None) is not None else 0.85)))

    painter.save()
    rot = float(getattr(el, "rotation_deg", 0.0) or 0.0)
    if abs(rot) >= 0.05:
        c = pill_rect.center()
        painter.translate(c.x(), c.y())
        painter.rotate(rot)
        painter.translate(-c.x(), -c.y())

    painter.setFont(font)

    if style == "pill":
        if (sec_col is not None and sec_col.alpha() == 0) or bg_op <= 0.01:
            pass
        else:
            corner_rad = min(12.0, pill_h / 2.0)
            painter.setPen(QPen(QColor(255, 255, 255, int(round(180 * bg_op))), 1.2))
            if sec_col is None or not sec_col.isValid():
                sec_brush = QColor(15, 20, 28, int(round(255 * bg_op)))
            else:
                sec_brush = QColor(sec_col.red(), sec_col.green(), sec_col.blue(), int(round(255 * bg_op)))
            painter.setBrush(sec_brush)
            painter.drawRoundedRect(pill_rect, corner_rad, corner_rad)

    text_color = el.color if el.color not in (QColor(0, 0, 0), QColor("#000000")) else QColor(255, 255, 255)
    outline_col = (
        QColor(sec_col.red(), sec_col.green(), sec_col.blue(), max(25, int(round(255 * max(0.25, bg_op)))))
        if (sec_col is not None and sec_col.isValid() and sec_col.alpha() > 0)
        else QColor(17, 24, 39, max(25, int(round(255 * max(0.25, bg_op)))))
    )

    font_pt_val = max(10, float(font.pointSize()) if font.pointSize() > 0 else float(max(10, int(round(el.stroke_width * 2.2)))))
    cur_y = ry + pad_y + metrics.ascent()
    for line in lines:
        lw = float(metrics.horizontalAdvance(line))
        if align == "center":
            tx = rx + pad_x + (max_line_w - lw) * 0.5
        elif align == "right":
            tx = rx + pad_x + (max_line_w - lw)
        else:
            tx = rx + pad_x

        if style == "outline" and line:
            t_path = QPainterPath()
            t_path.addText(QPointF(tx, cur_y), font, line)
            stroke_w = max(3.0, font_pt_val * 0.22)
            painter.setPen(QPen(outline_col, stroke_w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(t_path)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(text_color))
            painter.drawPath(t_path)
        elif style == "shadow" and line:
            shadow_off = max(2.0, font_pt_val * 0.09)
            painter.setPen(QPen(QColor(0, 0, 0, 155)))
            painter.drawText(QPointF(tx + shadow_off, cur_y + shadow_off), line)
            painter.setPen(QPen(QColor(0, 0, 0, 75)))
            painter.drawText(QPointF(tx + shadow_off * 1.6, cur_y + shadow_off * 1.6), line)
            painter.setPen(QPen(text_color))
            painter.drawText(QPointF(tx, cur_y), line)
        else:
            painter.setPen(QPen(text_color))
            painter.drawText(QPointF(tx, cur_y), line)
        cur_y += line_h

    painter.restore()


def _draw_vector_shape(painter: QPainter, el: MarkupElement) -> None:
    """Draws all geometric & callout shapes with optional rotation, fill, and opacity."""
    if el.rect is None:
        return
    r = el.rect.normalized()
    el_opacity = max(0.15, min(1.0, float(getattr(el, "opacity", 1.0) if getattr(el, "opacity", None) is not None else 1.0)))
    rot = float(getattr(el, "rotation_deg", 0.0) or 0.0)
    filled = bool(getattr(el, "shape_filled", False))

    painter.save()
    painter.setOpacity(el_opacity)
    if abs(rot) >= 0.05:
        c = r.center()
        painter.translate(c.x(), c.y())
        painter.rotate(rot)
        painter.translate(-c.x(), -c.y())

    if el.stroke_width > 0:
        dash = str(getattr(el, "dash_style", "solid") or "solid").strip().lower()
        if dash == "outlined":
            bg = getattr(el, "bg_color", None)
            if bg is not None and bg.isValid() and bg.alpha() > 0:
                out_col = QColor(bg.red(), bg.green(), bg.blue(), bg.alpha())
            else:
                out_col = QColor("#FFFFFF") if el.color.lightness() < 150 else QColor("#111111")
            out_w = el.stroke_width + max(4, int(round(el.stroke_width * 0.5)))
            painter.setPen(QPen(out_col, out_w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(Qt.NoBrush)
            if el.kind == "rect":
                painter.drawRoundedRect(r, 4.0, 4.0)
            elif el.kind == "ellipse":
                painter.drawEllipse(r)
            elif el.kind == "triangle":
                painter.drawPolygon(build_regular_polygon_in_rect(r, 3, -90.0))
            elif el.kind == "pentagon":
                painter.drawPolygon(build_regular_polygon_in_rect(r, 5, -90.0))
            elif el.kind == "hexagon":
                painter.drawPolygon(build_regular_polygon_in_rect(r, 6, -90.0))
            elif el.kind == "star":
                painter.drawPolygon(build_star_polygon_in_rect(r))
            elif el.kind == "heart_shape":
                painter.drawPath(build_heart_path_in_rect(r))
            elif el.kind == "speech_bubble":
                painter.drawPath(build_speech_bubble_path_in_rect(r))
            pen = QPen(el.color, max(1, el.stroke_width), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        else:
            pen = QPen(el.color, max(1, el.stroke_width), _pen_style_from_dash(dash), Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
    else:
        painter.setPen(Qt.NoPen)
    if filled:
        bg = getattr(el, "bg_color", None)
        if bg is not None and bg.isValid():
            if bg.alpha() > 0:
                painter.setBrush(QBrush(QColor(bg.red(), bg.green(), bg.blue(), bg.alpha())))
            else:
                painter.setBrush(Qt.NoBrush)
        else:
            painter.setBrush(QBrush(QColor(el.color.red(), el.color.green(), el.color.blue(), 220)))
    else:
        painter.setBrush(Qt.NoBrush)

    if el.kind == "rect":
        painter.drawRoundedRect(r, 4.0, 4.0)
    elif el.kind == "ellipse":
        painter.drawEllipse(r)
    elif el.kind == "triangle":
        painter.drawPolygon(build_regular_polygon_in_rect(r, 3, -90.0))
    elif el.kind == "pentagon":
        painter.drawPolygon(build_regular_polygon_in_rect(r, 5, -90.0))
    elif el.kind == "hexagon":
        painter.drawPolygon(build_regular_polygon_in_rect(r, 6, -90.0))
    elif el.kind == "star":
        painter.drawPolygon(build_star_polygon_in_rect(r))
    elif el.kind == "heart_shape":
        painter.drawPath(build_heart_path_in_rect(r))
    elif el.kind == "speech_bubble":
        painter.drawPath(build_speech_bubble_path_in_rect(r))
    painter.restore()


def _draw_line_element(painter: QPainter, el: MarkupElement) -> None:
    """Draws plain lines, single arrows, and double-ended arrows with optional rotation and dash style."""
    if not el.start_pt or not el.end_pt:
        return
    p1, p2 = el.start_pt, el.end_pt
    if math.hypot(p2.x() - p1.x(), p2.y() - p1.y()) < 4.0:
        return
    el_opacity = max(0.15, min(1.0, float(getattr(el, "opacity", 1.0) if getattr(el, "opacity", None) is not None else 1.0)))
    style = el.kind if el.kind in ("line", "double_arrow") else str(getattr(el, "line_style", "arrow") or "arrow")
    rot = float(getattr(el, "rotation_deg", 0.0) or 0.0)

    painter.save()
    painter.setOpacity(el_opacity)
    if abs(rot) >= 0.05:
        c = get_markup_element_unrotated_rect(el).center()
        painter.translate(c.x(), c.y())
        painter.rotate(rot)
        painter.translate(-c.x(), -c.y())
    dash = str(getattr(el, "dash_style", "solid") or "solid").strip().lower()
    pen = QPen(el.color, max(1, el.stroke_width), _pen_style_from_dash(dash), Qt.RoundCap, Qt.RoundJoin)

    if dash == "outlined":
        bg = getattr(el, "bg_color", None)
        if bg is not None and bg.isValid() and bg.alpha() > 0:
            out_col = QColor(bg.red(), bg.green(), bg.blue(), bg.alpha())
        else:
            out_col = QColor("#FFFFFF") if el.color.lightness() < 150 else QColor("#111111")
        out_pad = max(4, int(round(el.stroke_width * 0.55)))
        out_pen = QPen(out_col, max(1, el.stroke_width) + out_pad, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        if style == "line":
            painter.setPen(out_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(p1, p2)
        elif style == "double_arrow":
            p_end_base, head_end = build_arrow_geometry(p1, p2, el.stroke_width)
            p_start_base, head_start = build_arrow_geometry(p2, p1, el.stroke_width)
            painter.setPen(out_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(p_start_base, p_end_base)
            painter.setPen(QPen(out_col, max(3.0, out_pad * 0.9), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(QBrush(out_col))
            painter.drawPolygon(head_end)
            painter.drawPolygon(head_start)
        else:
            p_base, arrowhead = build_arrow_geometry(p1, p2, el.stroke_width)
            painter.setPen(out_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(p1, p_base)
            painter.setPen(QPen(out_col, max(3.0, out_pad * 0.9), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(QBrush(out_col))
            painter.drawPolygon(arrowhead)
        pen = QPen(el.color, max(1, el.stroke_width), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

    if style == "line":
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(p1, p2)
    elif style == "double_arrow":
        p_end_base, head_end = build_arrow_geometry(p1, p2, el.stroke_width)
        p_start_base, head_start = build_arrow_geometry(p2, p1, el.stroke_width)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(p_start_base, p_end_base)
        painter.setPen(QPen(el.color, 1.0))
        painter.setBrush(QBrush(el.color))
        painter.drawPolygon(head_end)
        painter.drawPolygon(head_start)
    else:
        p_base, arrowhead = build_arrow_geometry(p1, p2, el.stroke_width)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(p1, p_base)
        painter.setPen(QPen(el.color, 1.0))
        painter.setBrush(QBrush(el.color))
        painter.drawPolygon(arrowhead)
    painter.restore()


def _draw_pen_stroke(painter: QPainter, el: MarkupElement, canvas_w: int, canvas_h: int) -> None:
    """Draws freehand pen/highlighter strokes with non-stacking opacity via an isolated mask."""
    if len(el.points) < 2:
        return
    el_opacity = max(0.15, min(1.0, float(getattr(el, "opacity", 1.0) if getattr(el, "opacity", None) is not None else 1.0)))
    is_hl = el.kind == "highlighter"
    eff_opacity = (0.42 * el_opacity) if is_hl else el_opacity
    stroke_w = max(8, int(el.stroke_width * 2.6)) if is_hl else max(1, el.stroke_width)

    path = QPainterPath()
    path.moveTo(el.points[0])
    for pt in el.points[1:]:
        path.lineTo(pt)

    if eff_opacity >= 0.99:
        painter.save()
        painter.setPen(QPen(el.color, stroke_w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
        painter.restore()
        return

    # Isolated transparent mask prevents self-intersecting strokes from darkening
    stroke_mask = QImage(max(1, canvas_w), max(1, canvas_h), QImage.Format.Format_ARGB32_Premultiplied)
    stroke_mask.fill(Qt.transparent)
    mp = QPainter(stroke_mask)
    mp.setRenderHint(QPainter.Antialiasing, True)
    solid_col = QColor(el.color.red(), el.color.green(), el.color.blue(), 255)
    mp.setPen(QPen(solid_col, stroke_w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    mp.setBrush(Qt.NoBrush)
    mp.drawPath(path)
    mp.end()

    painter.save()
    painter.setOpacity(eff_opacity)
    painter.drawImage(0, 0, stroke_mask)
    painter.restore()


def _apply_obscure_element_to_rgb(base_rgb: np.ndarray, el: MarkupElement) -> np.ndarray:
    """Applies box or circular brush blur/pixelate redaction onto an RGB numpy image."""
    if base_rgb is None or el.kind not in ("blur", "pixelate"):
        return base_rgb
    h, w = base_rgb.shape[:2]
    strength = max(15, min(100, int(getattr(el, "obscure_strength", 55) or 55)))

    if getattr(el, "points", None) and len(el.points) >= 1 and getattr(el, "obscure_mode", "box") == "brush":
        rad = max(10, int(round(el.stroke_width * 2.2)))
        xs = [p.x() for p in el.points]
        ys = [p.y() for p in el.points]
        pad = rad + 6
        x1 = max(0, int(math.floor(min(xs) - pad)))
        y1 = max(0, int(math.floor(min(ys) - pad)))
        x2 = min(w, int(math.ceil(max(xs) + pad)))
        y2 = min(h, int(math.ceil(max(ys) + pad)))
        rw, rh = x2 - x1, y2 - y1
        if rw <= 4 or rh <= 4:
            return base_rgb
        patch = base_rgb[y1:y2, x1:x2].copy()
        if el.kind == "pixelate":
            bs = max(6, int(round(6 + (strength / 100.0) * 26)))
            redacted = apply_pixelate(patch, (0, 0, rw, rh), block_size=bs)
        else:
            blur_rad = max(9, int(round(9 + (strength / 100.0) * 34))) | 1
            redacted = apply_blur_redact(patch, (0, 0, rw, rh), radius=blur_rad)

        mask = np.zeros((rh, rw), dtype=np.uint8)
        pts_int = [(int(round(p.x() - x1)), int(round(p.y() - y1))) for p in el.points]
        if len(pts_int) == 1:
            cv2.circle(mask, pts_int[0], rad, 255, -1)
        else:
            for a, b in zip(pts_int[:-1], pts_int[1:]):
                cv2.line(mask, a, b, 255, rad * 2, lineType=cv2.LINE_AA)
            cv2.circle(mask, pts_int[0], rad, 255, -1)
            cv2.circle(mask, pts_int[-1], rad, 255, -1)

        if el.kind == "blur":
            mask_f = (cv2.GaussianBlur(mask, (9, 9), 2.5).astype(np.float32) / 255.0)[:, :, None]
        else:
            mask_f = (mask.astype(np.float32) / 255.0)[:, :, None]

        out = base_rgb.copy()
        blended = np.clip(redacted.astype(np.float32) * mask_f + patch.astype(np.float32) * (1.0 - mask_f) + 0.5, 0, 255).astype(np.uint8)
        out[y1:y2, x1:x2] = blended
        return out

    if el.rect is not None:
        rx = max(0, int(round(el.rect.x())))
        ry = max(0, int(round(el.rect.y())))
        rw = max(1, min(w - rx, int(round(el.rect.width()))))
        rh = max(1, min(h - ry, int(round(el.rect.height()))))
        if rw > 4 and rh > 4:
            if el.kind == "pixelate":
                bs = max(6, int(round(min(rw, rh) * (0.04 + 0.14 * (strength / 100.0)))))
                return apply_pixelate(base_rgb, (rx, ry, rw, rh), block_size=bs)
            rad = max(9, int(round(min(rw, rh) * (0.10 + 0.25 * (strength / 100.0))))) | 1
            return apply_blur_redact(base_rgb, (rx, ry, rw, rh), radius=rad)
    return base_rgb


def get_markup_element_bounds(el: MarkupElement | None) -> QRectF:
    """Returns the scene bounding rectangle for an element, including rotation, stroke and arrowhead margin."""
    if el is None:
        return QRectF()
    margin = max(8.0, float(el.stroke_width) * 1.5)
    rot = float(getattr(el, "rotation_deg", 0.0) or 0.0)

    if el.kind in ALL_SHAPE_KINDS or el.kind in ("pixelate", "blur", "sticker"):
        if el.kind in ("pixelate", "blur") and getattr(el, "points", None) and getattr(el, "obscure_mode", "box") == "brush":
            rad = max(10.0, float(el.stroke_width) * 2.2) + 6.0
            xs = [p.x() for p in el.points]
            ys = [p.y() for p in el.points]
            return QRectF(min(xs), min(ys), max(1.0, max(xs) - min(xs)), max(1.0, max(ys) - min(ys))).adjusted(-rad, -rad, rad, rad)
        if el.rect is not None:
            base_r = el.rect.normalized()
            if abs(rot) >= 0.05 and (el.kind in ALL_SHAPE_KINDS or el.kind == "sticker"):
                base_r = _rotated_rect_bounding_box(base_r, rot)
            return base_r.adjusted(-margin, -margin, margin, margin)
        return QRectF()

    elif el.kind in ALL_LINE_KINDS:
        if el.start_pt is not None and el.end_pt is not None:
            base_r = get_markup_element_unrotated_rect(el)
            if abs(rot) >= 0.05:
                base_r = _rotated_rect_bounding_box(base_r, rot)
            arrow_m = max(margin, float(el.stroke_width) * 4.0)
            return base_r.adjusted(-arrow_m, -arrow_m, arrow_m, arrow_m)
        return QRectF()

    elif el.kind in ("pen", "highlighter"):
        if el.points:
            xs = [p.x() for p in el.points]
            ys = [p.y() for p in el.points]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            hl_m = margin if el.kind == "pen" else max(margin, float(el.stroke_width) * 2.8)
            return QRectF(x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)).adjusted(-hl_m, -hl_m, hl_m, hl_m)
        return QRectF()

    elif el.kind in ("magnifier", "magnify"):
        r = (el.rect.width() / 2.0) if el.rect else 45.0
        if el.start_pt is not None and el.end_pt is not None:
            zoom = getattr(el, "zoom_factor", 2.5)
            src_r = max(6.0, r / max(1.0, zoom))
            b1 = QRectF(el.start_pt.x() - src_r, el.start_pt.y() - src_r, src_r * 2.0, src_r * 2.0)
            b2 = QRectF(el.end_pt.x() - r, el.end_pt.y() - r, r * 2.0, r * 2.0)
            return b1.united(b2).adjusted(-margin, -margin, margin, margin)
        elif el.rect is not None:
            return el.rect.normalized().adjusted(-margin, -margin, margin, margin)
        return QRectF()

    elif el.kind == "text":
        if el.rect is not None and el.text:
            pill_w, pill_h, _, _, _ = _measure_text_pill_size(el)
            base_r = QRectF(el.rect.x(), el.rect.y(), pill_w, pill_h)
            if abs(rot) >= 0.05:
                base_r = _rotated_rect_bounding_box(base_r, rot)
            extra = 8.0 if getattr(el, "text_style", "pill") in ("outline", "shadow") else 4.0
            return base_r.adjusted(-extra, -extra, extra, extra)
        elif el.rect is not None:
            return el.rect.normalized().adjusted(-margin, -margin, margin, margin)
        return QRectF()

    return QRectF()


def translate_markup_element(el: MarkupElement | None, dx: float, dy: float) -> None:
    """Translates the scene coordinates of a markup element by (dx, dy)."""
    if el is None:
        return
    if el.rect is not None:
        el.rect.translate(dx, dy)
    if el.start_pt is not None:
        el.start_pt = QPointF(el.start_pt.x() + dx, el.start_pt.y() + dy)
    if el.end_pt is not None:
        el.end_pt = QPointF(el.end_pt.x() + dx, el.end_pt.y() + dy)
    if el.points:
        el.points = [QPointF(p.x() + dx, p.y() + dy) for p in el.points]


def render_markup_element_pixmap(
    el: MarkupElement,
    base_rgb: np.ndarray | None = None,
    crop_rect: tuple[int, int, int, int] | None = None,
) -> tuple[QPixmap, QPointF]:
    """
    Renders a single markup element into a transparent QPixmap and its top-left scene offset.
    For privacy tools (pixelate, blur) and magnifiers, samples directly from base_rgb.
    """
    bounds = get_markup_element_bounds(el)
    if bounds.isEmpty() or bounds.width() <= 0 or bounds.height() <= 0:
        return QPixmap(), QPointF()

    x1 = int(math.floor(bounds.left()))
    y1 = int(math.floor(bounds.top()))
    x2 = int(math.ceil(bounds.right()))
    y2 = int(math.ceil(bounds.bottom()))
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    top_left = QPointF(float(x1), float(y1))

    # For pixelate or blur:
    if el.kind in ("pixelate", "blur") and base_rgb is not None:
        base_h, base_w = base_rgb.shape[:2]
        cl, ct = (crop_rect[0], crop_rect[1]) if crop_rect is not None else (0, 0)
        pm = QPixmap(w, h)
        pm.fill(Qt.transparent)
        if getattr(el, "points", None) and getattr(el, "obscure_mode", "box") == "brush":
            shifted_pts = [QPointF(pt.x() + cl, pt.y() + ct) for pt in el.points]
            tmp_el = MarkupElement(
                kind=el.kind,
                points=shifted_pts,
                stroke_width=el.stroke_width,
                obscure_mode="brush",
                obscure_strength=getattr(el, "obscure_strength", 55),
            )
            sx1 = max(0, x1 + int(cl))
            sy1 = max(0, y1 + int(ct))
            sx2 = min(base_w, x2 + int(cl))
            sy2 = min(base_h, y2 + int(ct))
            if sx2 > sx1 + 2 and sy2 > sy1 + 2:
                local_el = MarkupElement(
                    kind=el.kind,
                    points=[QPointF(pt.x() - sx1, pt.y() - sy1) for pt in tmp_el.points],
                    stroke_width=el.stroke_width,
                    obscure_mode="brush",
                    obscure_strength=getattr(el, "obscure_strength", 55),
                )
                sub = _apply_obscure_element_to_rgb(base_rgb[sy1:sy2, sx1:sx2], local_el)
                sw, sh = sx2 - sx1, sy2 - sy1
                qimg_sub = QImage(sub.data, sw, sh, 3 * sw, QImage.Format.Format_RGB888).copy()
                p = QPainter(pm)
                p.drawImage(QPointF(float(sx1 - cl - x1), float(sy1 - ct - y1)), qimg_sub)
                p.end()
            return pm, top_left

        if el.rect is not None:
            rx = max(0, int(round(el.rect.x() + cl)))
            ry = max(0, int(round(el.rect.y() + ct)))
            rw = max(1, min(base_w - rx, int(round(el.rect.width()))))
            rh = max(1, min(base_h - ry, int(round(el.rect.height()))))
            if rw > 4 and rh > 4:
                strength = max(15, min(100, int(getattr(el, "obscure_strength", 55) or 55)))
                patch = base_rgb[ry:ry+rh, rx:rx+rw].copy()
                if el.kind == "pixelate":
                    bs = max(6, int(round(min(rw, rh) * (0.04 + 0.14 * (strength / 100.0)))))
                    patch = apply_pixelate(patch, (0, 0, rw, rh), block_size=bs)
                else:
                    rad = max(9, int(round(min(rw, rh) * (0.10 + 0.25 * (strength / 100.0))))) | 1
                    patch = apply_blur_redact(patch, (0, 0, rw, rh), radius=rad)
                qimg_patch = QImage(patch.data, rw, rh, 3 * rw, QImage.Format.Format_RGB888).copy()

                p = QPainter(pm)
                p.setRenderHint(QPainter.Antialiasing, True)
                rel_x = el.rect.x() - float(x1)
                rel_y = el.rect.y() - float(y1)
                p.drawImage(QRectF(rel_x, rel_y, el.rect.width(), el.rect.height()), qimg_patch)
                p.setPen(QPen(el.color, 1.5, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRectF(rel_x, rel_y, el.rect.width(), el.rect.height()))
                p.end()
        return pm, top_left

    # For magnifier:
    if el.kind in ("magnifier", "magnify") and base_rgb is not None:
        zoom = getattr(el, "zoom_factor", 2.5)
        p_src = getattr(el, "start_pt", None)
        p_tgt = getattr(el, "end_pt", None)
        ring_col = getattr(el, "color", QColor(230, 40, 40))
        rad = (el.rect.width() / 2.0) if el.rect else 45.0
        src_r = max(6.0, rad / max(1.0, zoom))
        cl, ct = (crop_rect[0], crop_rect[1]) if crop_rect is not None else (0, 0)

        pm = QPixmap(w, h)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)

        if p_src is not None and p_tgt is not None and math.hypot(p_tgt.x() - p_src.x(), p_tgt.y() - p_src.y()) > 6.0:
            rel_src = QPointF(p_src.x() - float(x1), p_src.y() - float(y1))
            rel_tgt = QPointF(p_tgt.x() - float(x1), p_tgt.y() - float(y1))

            tangents = compute_circle_tangent_lines(rel_src, src_r, rel_tgt, rad)
            pen_ray = QPen(ring_col, max(1.5, el.stroke_width * 0.4), Qt.SolidLine, Qt.RoundCap)
            p.setPen(pen_ray)
            for pt1, pt2 in tangents:
                p.drawLine(pt1, pt2)

            pen_src = QPen(ring_col, max(2.0, el.stroke_width * 0.5), Qt.SolidLine)
            p.setPen(pen_src)
            p.setBrush(QColor(ring_col.red(), ring_col.green(), ring_col.blue(), 25))
            p.drawEllipse(rel_src, src_r, src_r)
            p.setPen(QPen(QColor(255, 255, 255, 180), 1.0))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(rel_src, max(2.0, src_r - 1.2), max(2.0, src_r - 1.2))

            sample_pt = QPointF(p_src.x() + cl, p_src.y() + ct)
            loupe_pm = create_magnifier_pixmap(base_rgb, sample_pt, rad, zoom, ring_color=ring_col)
            offset = loupe_pm.width() / 2.0
            p.drawPixmap(QPointF(rel_tgt.x() - offset, rel_tgt.y() - offset), loupe_pm)

        elif el.rect is not None:
            cx = el.rect.center().x()
            cy = el.rect.center().y()
            rel_c = QPointF(cx - float(x1), cy - float(y1))
            sample_pt = QPointF(cx + cl, cy + ct)
            loupe_pm = create_magnifier_pixmap(base_rgb, sample_pt, rad, zoom, ring_color=ring_col)
            offset = loupe_pm.width() / 2.0
            p.drawPixmap(QPointF(rel_c.x() - offset, rel_c.y() - offset), loupe_pm)

        p.end()
        return pm, top_left

    # For vector annotations (pen, highlighter, arrow, rect, ellipse, star, text):
    shifted = MarkupElement(
        kind=el.kind,
        rect=QRectF(el.rect.x() - float(x1), el.rect.y() - float(y1), el.rect.width(), el.rect.height()) if el.rect else None,
        start_pt=QPointF(el.start_pt.x() - float(x1), el.start_pt.y() - float(y1)) if el.start_pt else None,
        end_pt=QPointF(el.end_pt.x() - float(x1), el.end_pt.y() - float(y1)) if el.end_pt else None,
        points=[QPointF(pt.x() - float(x1), pt.y() - float(y1)) for pt in el.points],
        color=el.color,
        bg_color=el.bg_color,
        stroke_width=el.stroke_width,
        text=el.text,
        zoom_factor=el.zoom_factor,
        font_family=getattr(el, "font_family", "Segoe UI") or "Segoe UI",
        font_bold=getattr(el, "font_bold", True),
        font_italic=getattr(el, "font_italic", False),
        rotation_deg=float(getattr(el, "rotation_deg", 0.0) or 0.0),
        text_style=str(getattr(el, "text_style", "pill") or "pill"),
        text_align=str(getattr(el, "text_align", "left") or "left"),
        opacity=float(getattr(el, "opacity", 1.0) if getattr(el, "opacity", None) is not None else 1.0),
        shape_filled=bool(getattr(el, "shape_filled", False)),
        line_style=str(getattr(el, "line_style", "arrow") or "arrow"),
        dash_style=str(getattr(el, "dash_style", "solid") or "solid"),
        sticker_id=str(getattr(el, "sticker_id", "") or ""),
        obscure_mode=str(getattr(el, "obscure_mode", "brush") or "brush"),
        obscure_strength=int(getattr(el, "obscure_strength", 55) or 55),
    )

    pm = QPixmap(w, h)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

    el_opacity = max(0.15, min(1.0, float(getattr(shifted, "opacity", 1.0) if getattr(shifted, "opacity", None) is not None else 1.0)))
    rot = float(getattr(shifted, "rotation_deg", 0.0) or 0.0)

    if shifted.kind in ("pen", "highlighter") and len(shifted.points) >= 2:
        _draw_pen_stroke(painter, shifted, w, h)

    elif shifted.kind in ALL_LINE_KINDS and shifted.start_pt and shifted.end_pt:
        _draw_line_element(painter, shifted)

    elif shifted.kind in ALL_SHAPE_KINDS and shifted.rect is not None:
        _draw_vector_shape(painter, shifted)

    elif shifted.kind == "sticker" and shifted.rect is not None:
        draw_vector_sticker(
            painter,
            shifted.rect.normalized(),
            getattr(shifted, "sticker_id", "heart"),
            rotation_deg=rot,
            opacity=el_opacity,
        )

    elif shifted.kind == "text" and shifted.rect is not None and shifted.text:
        painter.save()
        painter.setOpacity(el_opacity)
        _draw_text_element(painter, shifted)
        painter.restore()

    painter.end()
    return pm, top_left


def flatten_markup_element_roi(
    base_rgb: np.ndarray,
    el: MarkupElement,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
    """
    Fast ROI-only bake for a single non-magnifier MarkupElement directly into base_rgb in-place.
    Avoids copying or wrapping the full 24MP+ master array into a QImage.
    Returns ((y0, y1, x0, x1), before_roi, after_roi), or None to fall back to full-frame flatten.
    """
    if base_rgb is None or el is None or el.kind in ("magnifier", "magnify"):
        return None
    h, w = base_rgb.shape[:2]
    if h <= 0 or w <= 0:
        return None

    bounds = get_markup_element_bounds(el)
    if bounds.isNull() or bounds.width() <= 0 or bounds.height() <= 0:
        return None

    sw = max(2, int(getattr(el, "stroke_width", 4) or 4))
    pad = max(64, int(sw * 4.5) + 48)
    x0 = max(0, int(math.floor(bounds.left())) - pad)
    y0 = max(0, int(math.floor(bounds.top())) - pad)
    x1 = min(w, int(math.ceil(bounds.right())) + pad)
    y1 = min(h, int(math.ceil(bounds.bottom())) + pad)
    if x1 <= x0 or y1 <= y0:
        return None

    # If the ROI already spans > 85% of the image, full-frame flatten is equivalent
    if (x1 - x0) * (y1 - y0) > 0.85 * w * h:
        return None

    local_el = MarkupElement(
        kind=el.kind,
        rect=QRectF(el.rect.x() - x0, el.rect.y() - y0, el.rect.width(), el.rect.height()) if el.rect else None,
        start_pt=QPointF(el.start_pt.x() - x0, el.start_pt.y() - y0) if el.start_pt else None,
        end_pt=QPointF(el.end_pt.x() - x0, el.end_pt.y() - y0) if el.end_pt else None,
        points=[QPointF(pt.x() - x0, pt.y() - y0) for pt in el.points],
        color=el.color,
        bg_color=el.bg_color,
        stroke_width=el.stroke_width,
        text=el.text,
        zoom_factor=el.zoom_factor,
        font_family=getattr(el, "font_family", "Segoe UI") or "Segoe UI",
        font_bold=getattr(el, "font_bold", True),
        font_italic=getattr(el, "font_italic", False),
        rotation_deg=float(getattr(el, "rotation_deg", 0.0) or 0.0),
        text_style=str(getattr(el, "text_style", "pill") or "pill"),
        text_align=str(getattr(el, "text_align", "left") or "left"),
        opacity=float(getattr(el, "opacity", 1.0) if getattr(el, "opacity", None) is not None else 1.0),
        bg_opacity=float(getattr(el, "bg_opacity", 0.85) if getattr(el, "bg_opacity", None) is not None else 0.85),
        shape_filled=bool(getattr(el, "shape_filled", False)),
        line_style=str(getattr(el, "line_style", "arrow") or "arrow"),
        dash_style=str(getattr(el, "dash_style", "solid") or "solid"),
        sticker_id=str(getattr(el, "sticker_id", "") or ""),
        obscure_mode=str(getattr(el, "obscure_mode", "brush") or "brush"),
        obscure_strength=int(getattr(el, "obscure_strength", 55) or 55),
    )

    before_roi = np.ascontiguousarray(base_rgb[y0:y1, x0:x1].copy())
    after_roi = flatten_markup_onto_rgb_image(before_roi, [local_el])
    base_rgb[y0:y1, x0:x1] = after_roi
    return (y0, y1, x0, x1), before_roi, after_roi


def flatten_markup_onto_rgb_image(
    base_rgb: np.ndarray,
    elements: list[MarkupElement],
) -> np.ndarray:
    """
    Renders all markup elements (freehand pen, highlighter, arrows, rectangles,
    pixelation, blur, magnifiers, and text) directly onto a base RGB image,
    avoiding redundant RGB <-> BGR color conversions.
    """
    if base_rgb is None or not elements:
        return base_rgb

    result = base_rgb.copy()
    h, w = result.shape[:2]

    # 1. Apply privacy redactions (pixelate and blur: box or circular brush) directly in RGB
    for el in elements:
        if el.kind in ("pixelate", "blur"):
            result = _apply_obscure_element_to_rgb(result, el)

    # 2. Render vector annotations and callouts using QPainter directly in RGB888 / RGBA8888
    # Wrap result in contiguous RGB888 QImage directly without color conversions
    qimg = QImage(result.data, w, h, 3 * w, QImage.Format.Format_RGB888)

    painter = QPainter(qimg)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

    for el in elements:
        el_opacity = max(0.15, min(1.0, float(getattr(el, "opacity", 1.0) if getattr(el, "opacity", None) is not None else 1.0)))
        rot = float(getattr(el, "rotation_deg", 0.0) or 0.0)

        # Freehand Pen / Highlighter (Non-stacking opacity mask)
        if el.kind in ("pen", "highlighter") and len(el.points) >= 2:
            _draw_pen_stroke(painter, el, w, h)

        # Lines & Arrows (line, arrow, double_arrow)
        elif el.kind in ALL_LINE_KINDS and el.start_pt and el.end_pt:
            _draw_line_element(painter, el)

        # Shapes (rect, ellipse, triangle, pentagon, hexagon, star, heart_shape, speech_bubble)
        elif el.kind in ALL_SHAPE_KINDS and el.rect is not None:
            _draw_vector_shape(painter, el)

        # Vector Stickers
        elif el.kind == "sticker" and el.rect is not None:
            draw_vector_sticker(
                painter,
                el.rect.normalized(),
                getattr(el, "sticker_id", "heart"),
                rotation_deg=rot,
                opacity=el_opacity,
            )

        # Magnifier Loupe (supports two-point source -> target callout or single in-place loupe)
        elif el.kind == "magnifier":
            zoom = getattr(el, "zoom_factor", 2.5)
            p_src = getattr(el, "start_pt", None)
            p_tgt = getattr(el, "end_pt", None)
            ring_col = getattr(el, "color", QColor(230, 40, 40))

            if p_src is not None and p_tgt is not None and math.hypot(p_tgt.x() - p_src.x(), p_tgt.y() - p_src.y()) > 6.0:
                # Two-point callout: Source circle at p_src, Loupe at p_tgt
                tgt_cx, tgt_cy = p_tgt.x(), p_tgt.y()
                radius = max(20.0, min(el.rect.width(), el.rect.height()) / 2.0) if el.rect else 45.0
                src_r = max(6.0, radius / zoom)

                # 1. Connect tangent lines first (behind circles)
                tangents = compute_circle_tangent_lines(p_src, src_r, p_tgt, radius)
                pen_ray = QPen(ring_col, max(1.5, el.stroke_width * 0.4), Qt.SolidLine, Qt.RoundCap)
                painter.setPen(pen_ray)
                for pt1, pt2 in tangents:
                    painter.drawLine(pt1, pt2)

                # 2. Source circle around feature of interest
                pen_src = QPen(ring_col, max(2.0, el.stroke_width * 0.5), Qt.SolidLine)
                painter.setPen(pen_src)
                painter.setBrush(QColor(ring_col.red(), ring_col.green(), ring_col.blue(), 25))
                painter.drawEllipse(p_src, src_r, src_r)

                # Subtle inner white accent ring on source circle
                painter.setPen(QPen(QColor(255, 255, 255, 180), 1.0))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(p_src, max(2.0, src_r - 1.2), max(2.0, src_r - 1.2))

                # 3. Magnified loupe at target position sampling from p_src
                loupe_pm = create_magnifier_pixmap(result, p_src, radius, zoom, ring_color=ring_col)
                offset = loupe_pm.width() / 2.0
                painter.drawPixmap(QPointF(tgt_cx - offset, tgt_cy - offset), loupe_pm)

            elif el.rect is not None:
                # Classic single-point in-place loupe
                cx = el.rect.center().x()
                cy = el.rect.center().y()
                radius = max(20.0, min(el.rect.width(), el.rect.height()) / 2.0)
                loupe_pm = create_magnifier_pixmap(result, QPointF(cx, cy), radius, zoom, ring_color=ring_col)
                offset = loupe_pm.width() / 2.0
                painter.drawPixmap(QPointF(cx - offset, cy - offset), loupe_pm)

        # Text Callout Pill / Outline / Shadow / Plain (Supports multi-line, alignment, rotation & opacity)
        elif el.kind == "text" and el.rect is not None and el.text:
            painter.save()
            painter.setOpacity(el_opacity)
            _draw_text_element(painter, el)
            painter.restore()

    painter.end()

    # The buffer in qimg directly mutates result without any color conversions
    return result


def flatten_markup_onto_image(
    base_bgr: np.ndarray,
    elements: list[MarkupElement],
) -> np.ndarray:
    """
    Backward-compatible entry point for BGR images (converts BGR -> RGB -> flattened -> BGR).
    """
    if base_bgr is None or not elements:
        return base_bgr
    rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
    out_rgb = flatten_markup_onto_rgb_image(rgb, elements)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


class _DraggableGripHandle(QWidget):
    """Subtle drag handle with 2 columns of 4 clean dots on transparent background without button border."""
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("DraggableGripHandle")
        self.setFixedSize(14, 26)
        self.setCursor(Qt.SizeAllCursor)
        self.setToolTip("Drag toolbar to reposition")
        self.setStyleSheet(
            """
            _DraggableGripHandle, QWidget#DraggableGripHandle {
                background: transparent !important;
                border: none !important;
                border-radius: 0px !important;
                padding: 0px !important;
                margin: 0px !important;
            }
            """
        )
        self._hover: bool = False

    def enterEvent(self, event: Any) -> None:
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        p = self.parentWidget()
        if p is not None and hasattr(p, "mousePressEvent"):
            mapped = QMouseEvent(
                event.type(),
                self.mapTo(p, event.position().toPoint()),
                event.globalPosition(),
                event.button(),
                event.buttons(),
                event.modifiers(),
            )
            p.mousePressEvent(mapped)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        p = self.parentWidget()
        if p is not None and hasattr(p, "mouseMoveEvent"):
            mapped = QMouseEvent(
                event.type(),
                self.mapTo(p, event.position().toPoint()),
                event.globalPosition(),
                event.button(),
                event.buttons(),
                event.modifiers(),
            )
            p.mouseMoveEvent(mapped)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        p = self.parentWidget()
        if p is not None and hasattr(p, "mouseReleaseEvent"):
            mapped = QMouseEvent(
                event.type(),
                self.mapTo(p, event.position().toPoint()),
                event.globalPosition(),
                event.button(),
                event.buttons(),
                event.modifiers(),
            )
            p.mouseReleaseEvent(mapped)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2563EB") if self._hover else QColor("#8595A8"))
        for x in (3.5, 9.0):
            for y in (4.0, 10.0, 16.0, 22.0):
                p.drawEllipse(QPointF(x, y), 1.25, 1.25)
        p.end()


class _SwatchButton(QPushButton):
    """Color swatch supporting Left-click (Primary) and Right-click (Secondary), including Transparent."""
    left_clicked = Signal(QColor)
    right_clicked = Signal(QColor)

    def __init__(self, color: QColor, name: str, parent: QWidget | None = None, size: int = 16) -> None:
        super().__init__(parent)
        self.color = color
        self.color_name = name
        self.is_transparent = (color.alpha() == 0 or name.lower().startswith("transparent"))
        self._swatch_size = max(12, int(size))
        self.setFixedSize(self._swatch_size, self._swatch_size)
        tip_name = "Transparent / No fill" if self.is_transparent else f"{name} ({color.name()})"
        self.setToolTip(f"{tip_name}\n• Left-click: Set Primary (foreground)\n• Right-click: Set Secondary (background)")
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.left_clicked.emit(self.color)
            event.accept()
            return
        elif event.button() == Qt.RightButton:
            self.right_clicked.emit(self.color)
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event: Any) -> None:
        if not getattr(self, "is_transparent", False):
            super().paintEvent(event)
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect().adjusted(1, 1, -1, -1)

        # White disc background
        p.setPen(QPen(QColor("#7DA2CE"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(rect)

        # Red diagonal slash (universal symbol for transparent / no fill)
        p.setPen(QPen(QColor("#E53935"), 1.8, Qt.SolidLine, Qt.RoundCap))
        r = rect.width() / 2.0
        cx, cy = rect.center().x(), rect.center().y()
        offset = r * 0.707
        p.drawLine(QPointF(cx + offset, cy - offset), QPointF(cx - offset, cy + offset))

        # Active ring indicator if selected as Primary or Secondary
        txt = self.text()
        if txt == "•":
            p.setPen(QPen(QColor("#005FB8"), 2.0))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(self.rect().adjusted(0, 0, -1, -1))
        elif txt == "◦":
            p.setPen(QPen(QColor("#005FB8"), 1.4, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(self.rect().adjusted(0, 0, -1, -1))
        p.end()


class _PhotoshopColorPickerWidget(QWidget):
    """
    Photoshop-style overlapping Primary (foreground) and Secondary (background)
    color chips with swap button (⇄, shortcut X) and click-to-pick dialogs.
    """
    primary_color_changed = Signal(QColor)
    secondary_color_changed = Signal(QColor)
    swap_requested = Signal()

    def __init__(self, parent: QWidget | None = None, large: bool = False) -> None:
        super().__init__(parent)
        self._large = bool(large)
        self.setToolTip("Primary (Foreground) & Secondary (Background) Colors\n• Click chip to edit color\n• Click ⇄ or press 'X' to swap")
        self.primary_color: QColor = QColor(17, 17, 17)
        self.secondary_color: QColor = QColor(255, 255, 255)
        self._is_dark: bool = False

        if self._large:
            self.setFixedSize(48, 46)
            self.primary_rect = QRect(2, 4, 24, 24)
            self.secondary_rect = QRect(16, 17, 24, 24)
            self.swap_rect = QRect(28, 2, 18, 15)
        else:
            self.setFixedSize(38, 26)
            # Primary chip (top-left)
            self.primary_rect = QRect(2, 2, 16, 16)
            # Secondary chip (bottom-right)
            self.secondary_rect = QRect(11, 8, 16, 16)
            # Swap button hit area (top-right)
            self.swap_rect = QRect(25, 0, 13, 13)

    def set_primary_color(self, col: QColor) -> None:
        if col.isValid():
            self.primary_color = col
            self.update()
            self.primary_color_changed.emit(col)

    def set_secondary_color(self, col: QColor) -> None:
        if col.isValid():
            self.secondary_color = col
            self.update()
            self.secondary_color_changed.emit(col)

    def swap_colors(self) -> None:
        p, s = self.primary_color, self.secondary_color
        self.primary_color = s
        self.secondary_color = p
        self.update()
        self.primary_color_changed.emit(self.primary_color)
        self.secondary_color_changed.emit(self.secondary_color)
        self.swap_requested.emit()

    def set_dark_mode(self, is_dark: bool) -> None:
        self._is_dark = bool(is_dark)
        self.update()

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        is_dark = getattr(self, "_is_dark", False)

        # Secondary (background) chip behind
        rad = 3 if getattr(self, "_large", False) else 2
        p.setPen(QPen(QColor("#64748B") if is_dark else QColor("#7DA2CE"), 1.2 if self._large else 1.0))
        if self.secondary_color.alpha() == 0:
            p.setBrush(QColor("#FFFFFF"))
            p.drawRoundedRect(self.secondary_rect, rad, rad)
            p.setPen(QPen(QColor("#E53935"), 1.8 if self._large else 1.5, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(self.secondary_rect.topRight() + QPoint(-2, 2), self.secondary_rect.bottomLeft() + QPoint(2, -2))
        else:
            p.setBrush(self.secondary_color)
            p.drawRoundedRect(self.secondary_rect, rad, rad)

        # Primary (foreground) chip on top
        p.setPen(QPen(QColor("#E2E8F0") if is_dark else QColor("#1E395B"), 1.4 if self._large else 1.2))
        if self.primary_color.alpha() == 0:
            p.setBrush(QColor("#FFFFFF"))
            p.drawRoundedRect(self.primary_rect, rad, rad)
            p.setPen(QPen(QColor("#E53935"), 1.8 if self._large else 1.5, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(self.primary_rect.topRight() + QPoint(-2, 2), self.primary_rect.bottomLeft() + QPoint(2, -2))
        else:
            p.setBrush(self.primary_color)
            p.drawRoundedRect(self.primary_rect, rad, rad)

        # Crisp swap arrows in top-right
        p.setPen(QPen(QColor("#CBD5E1") if is_dark else QColor("#475569"), 1.35 if self._large else 1.2, Qt.SolidLine, Qt.RoundCap))
        if getattr(self, "_large", False):
            # Top arrow ->
            p.drawLine(31, 6, 42, 6)
            p.drawLine(39, 3, 42, 6)
            p.drawLine(39, 9, 42, 6)
            # Bottom arrow <-
            p.drawLine(42, 12, 31, 12)
            p.drawLine(34, 9, 31, 12)
            p.drawLine(34, 15, 31, 12)
        else:
            # Top arrow ->
            p.drawLine(27, 3, 34, 3)
            p.drawLine(32, 1, 34, 3)
            p.drawLine(32, 5, 34, 3)
            # Bottom arrow <-
            p.drawLine(34, 8, 27, 8)
            p.drawLine(29, 6, 27, 8)
            p.drawLine(29, 10, 27, 8)
        p.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            pt = event.pos()
            if self.swap_rect.contains(pt):
                self.swap_colors()
                event.accept()
                return
            if self.primary_rect.contains(pt):
                col = QColorDialog.getColor(self.primary_color, self, "Select Primary (Foreground) Color")
                if col.isValid():
                    self.set_primary_color(col)
                event.accept()
                return
            if self.secondary_rect.contains(pt):
                col = QColorDialog.getColor(self.secondary_color, self, "Select Secondary (Background) Color")
                if col.isValid():
                    self.set_secondary_color(col)
                event.accept()
                return
        super().mousePressEvent(event)


class MarkupFloatingToolbar(QFrame):
    """
    Comprehensive Windows Live Photo Gallery-styled floating toolbar for
    WYSIWYG Quick Markup, Privacy Redaction, and Callouts.
    """
    tool_changed = Signal(str)
    stroke_width_changed = Signal(int)
    color_changed = Signal(QColor)                 # Primary color
    secondary_color_changed = Signal(QColor)       # Secondary color
    colors_swapped = Signal(QColor, QColor)        # (primary, secondary)
    undo_requested = Signal()
    clear_requested = Signal()
    apply_requested = Signal()
    cancel_requested = Signal()
    save_requested = Signal()
    font_family_changed = Signal(str)              # font family name
    text_format_changed = Signal(bool, bool)       # is_bold, is_italic
    text_style_changed = Signal(str)               # 'pill', 'outline', 'shadow', 'plain'
    text_align_changed = Signal(str)               # 'left', 'center', 'right'
    opacity_changed = Signal(float)                # 0.15 .. 1.0
    bg_opacity_changed = Signal(float)             # 0.0 .. 1.0
    shape_filled_changed = Signal(bool)
    line_style_changed = Signal(str)               # 'line', 'arrow', 'double_arrow'
    dash_style_changed = Signal(str)               # 'solid', 'dashed', 'dotted'
    sticker_changed = Signal(str)                  # 'heart', 'smiley', 'star', 'fire', 'check'
    obscure_mode_changed = Signal(str)             # 'brush', 'box'
    obscure_strength_changed = Signal(int)         # 15 .. 100
    magnifier_zoom_changed = Signal(float)         # zoom_factor (e.g. 2.5)

    QUICK_COLORS = [
        ("#E53935", "Red"),
        ("#FFEB3B", "Yellow"),
        ("#4CAF50", "Green"),
        ("#2196F3", "Blue"),
        ("#FFFFFF", "White"),
        ("#212121", "Dark"),
        ("TRANSPARENT", "Transparent / None"),
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.active_tool: str = "pen"
        self.active_shape: str = "rect"  # 'rect', 'ellipse', 'star'
        self.current_color = QColor(17, 17, 17)
        self.secondary_color = QColor(255, 255, 255)
        self._last_opaque_secondary_color = QColor(255, 255, 255)
        self.stroke_width: int = 16
        self.magnifier_zoom: float = 2.5
        self.font_family: str = "Segoe UI"
        self.is_text_bold: bool = True
        self.is_text_italic: bool = False
        self.text_style: str = "pill"
        self.text_align: str = "left"
        self.opacity: float = 1.0
        self.bg_opacity: float = 0.85
        self.shape_filled: bool = False
        self.line_style: str = "arrow"
        self.dash_style: str = "solid"
        self.active_sticker: str = "smiley"
        self.obscure_mode: str = "box"
        self.obscure_strength: int = 55
        self._is_dragging: bool = False
        self._drag_pos: QPoint = QPoint()
        self._user_positioned: bool = False
        self._init_ui()

    def _init_ui(self) -> None:
        self.setStyleSheet(f"""
            MarkupFloatingToolbar {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FDFDFE, stop:1 #EFF4F9);
                border: 1px solid #70A6E8;
                border-radius: 6px;
                padding: 3px;
            }}
            QToolButton {{
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 2px 4px;
                min-width: 0px;
                min-height: 22px;
                font-size: 11px;
                font-weight: 500;
                color: #1E395B;
                background: transparent;
            }}
            QToolButton:hover {{
                background: #E1ECF9;
                border: 1px solid #85B5EB;
            }}
            QToolButton:checked {{
                background: #CBE2F8;
                border: 1px solid #3388E8;
                color: #0E294B;
            }}
            QToolButton#btn_shape {{
                min-width: 54px;
                padding-right: 15px;
            }}
            QToolButton#btn_shape::menu-button {{
                border-left: none;
                width: 16px;
                padding-right: 2px;
            }}
            QToolButton#btn_shape:hover::menu-button,
            QToolButton#btn_shape:checked::menu-button {{
                border-left: 1px solid rgba(112, 166, 232, 130);
            }}
            QToolButton::menu-arrow {{
                image: url("{ARROW_DROPDOWN_PATH}");
                width: 7px;
                height: 4px;
            }}
            QLabel {{
                font-size: 11px;
                color: #1E395B;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 3, 6, 3)
        layout.setSpacing(2)

        # Drag Grip Handle (Clean subtle dots without button border)
        self.grip_handle = _DraggableGripHandle(self)
        layout.addWidget(self.grip_handle)

        # Title label
        self.lbl_title = QLabel("Markup:")
        self.lbl_title.setObjectName("MarkupTitleLabel")
        layout.addWidget(self.lbl_title)

        # Tool switcher buttons
        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        self.tool_buttons: dict[str, QToolButton] = {}

        # 1. Pen
        btn_pen = QToolButton()
        btn_pen.setText("Pen")
        btn_pen.setIcon(make_wlpg_icon("pen", 16))
        btn_pen.setIconSize(QSize(16, 16))
        btn_pen.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        btn_pen.setToolTip("Freehand ink drawing")
        btn_pen.setCheckable(True)
        btn_pen.setChecked(True)
        btn_pen.clicked.connect(lambda: self._set_tool("pen"))
        self.btn_group.addButton(btn_pen)
        layout.addWidget(btn_pen)
        self.tool_buttons["pen"] = btn_pen

        # 2. Highlighter
        btn_hl = QToolButton()
        btn_hl.setText("Marker")
        btn_hl.setIcon(make_wlpg_icon("highlighter", 16))
        btn_hl.setIconSize(QSize(16, 16))
        btn_hl.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        btn_hl.setToolTip("Semi-transparent fluorescent highlighter marker")
        btn_hl.setCheckable(True)
        btn_hl.clicked.connect(lambda: self._set_tool("highlighter"))
        self.btn_group.addButton(btn_hl)
        layout.addWidget(btn_hl)
        self.tool_buttons["highlighter"] = btn_hl

        # 3. Arrow
        btn_arrow = QToolButton()
        btn_arrow.setText("Arrow")
        btn_arrow.setIcon(make_wlpg_icon("arrow", 16))
        btn_arrow.setIconSize(QSize(16, 16))
        btn_arrow.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        btn_arrow.setToolTip("Directional vector arrow")
        btn_arrow.setCheckable(True)
        btn_arrow.clicked.connect(lambda: self._set_tool("arrow"))
        self.btn_group.addButton(btn_arrow)
        layout.addWidget(btn_arrow)
        self.tool_buttons["arrow"] = btn_arrow

        # 4. Shape Tool with Photoshop-style drop-down arrow for Box / Oval / Star
        self.btn_shape = QToolButton()
        self.btn_shape.setObjectName("btn_shape")
        self.btn_shape.setPopupMode(QToolButton.MenuButtonPopup)
        self.btn_shape.setCheckable(True)
        self.btn_shape.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._update_shape_button_ui()

        shape_menu = QMenu(self.btn_shape)
        shape_menu.setStyleSheet("""
            QMenu {
                background: #FFFFFF;
                border: 1px solid #70A6E8;
                padding: 2px;
            }
            QMenu::item {
                padding: 4px 18px 4px 8px;
                color: #1E395B;
                font-size: 11px;
            }
            QMenu::item:selected {
                background: #E1ECF9;
                color: #0E294B;
            }
        """)
        act_box = shape_menu.addAction(make_wlpg_icon("rect", 16), "Box (Rectangle)")
        act_box.triggered.connect(lambda: self.set_active_shape("rect"))

        act_oval = shape_menu.addAction(make_wlpg_icon("ellipse", 16), "Oval (Circle)")
        act_oval.triggered.connect(lambda: self.set_active_shape("ellipse"))

        act_star = shape_menu.addAction(make_wlpg_icon("star_shape", 16), "Star (5-point)")
        act_star.triggered.connect(lambda: self.set_active_shape("star"))

        self.btn_shape.setMenu(shape_menu)
        self.btn_shape.clicked.connect(lambda: self._set_tool(self.active_shape))
        self.btn_group.addButton(self.btn_shape)
        layout.addWidget(self.btn_shape)

        self.tool_buttons["rect"] = self.btn_shape
        self.tool_buttons["ellipse"] = self.btn_shape
        self.tool_buttons["star"] = self.btn_shape

        # 5. Blur
        btn_blur = QToolButton()
        btn_blur.setText("Blur")
        btn_blur.setIcon(make_wlpg_icon("blur", 16))
        btn_blur.setIconSize(QSize(16, 16))
        btn_blur.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        btn_blur.setToolTip("Privacy Gaussian blur")
        btn_blur.setCheckable(True)
        btn_blur.clicked.connect(lambda: self._set_tool("blur"))
        self.btn_group.addButton(btn_blur)
        layout.addWidget(btn_blur)
        self.tool_buttons["blur"] = btn_blur

        # 6. Pixelate
        btn_pix = QToolButton()
        btn_pix.setText("Pixel")
        btn_pix.setIcon(make_wlpg_icon("pixelate", 16))
        btn_pix.setIconSize(QSize(16, 16))
        btn_pix.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        btn_pix.setToolTip("Mosaic privacy redaction")
        btn_pix.setCheckable(True)
        btn_pix.clicked.connect(lambda: self._set_tool("pixelate"))
        self.btn_group.addButton(btn_pix)
        layout.addWidget(btn_pix)
        self.tool_buttons["pixelate"] = btn_pix

        # 7. Magnify
        btn_mag = QToolButton()
        btn_mag.setObjectName("btn_magnify")
        btn_mag.setText("Magnify")
        btn_mag.setIcon(make_wlpg_icon("magnify", 16))
        btn_mag.setIconSize(QSize(16, 16))
        btn_mag.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        btn_mag.setToolTip("Magnified callout (Click source, then click or drag target)")
        btn_mag.setCheckable(True)
        btn_mag.clicked.connect(lambda: self._set_tool("magnify"))
        self.btn_group.addButton(btn_mag)
        layout.addWidget(btn_mag)
        self.tool_buttons["magnify"] = btn_mag
        self.tool_buttons["magnifier"] = btn_mag

        # 8. Text
        btn_txt = QToolButton()
        btn_txt.setText("Text")
        btn_txt.setIcon(make_wlpg_icon("text", 16))
        btn_txt.setIconSize(QSize(16, 16))
        btn_txt.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        btn_txt.setToolTip("Text label with frosted pill")
        btn_txt.setCheckable(True)
        btn_txt.clicked.connect(lambda: self._set_tool("text"))
        self.btn_group.addButton(btn_txt)
        layout.addWidget(btn_txt)
        self.tool_buttons["text"] = btn_txt

        # Divider
        div1 = QFrame()
        div1.setFrameShape(QFrame.VLine)
        div1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(div1)

        # Stroke Width / Font Size Slider
        self.lbl_size = QLabel("Size:")
        layout.addWidget(self.lbl_size)

        self.slider_width = QSlider(Qt.Horizontal)
        self.slider_width.setRange(2, 150)
        self.slider_width.setValue(self.stroke_width)
        self.slider_width.setFixedWidth(60)
        self.slider_width.setCursor(Qt.SizeHorCursor)
        self.slider_width.setToolTip("Adjust stroke width ([ and ] keys)")
        self.slider_width.valueChanged.connect(self._on_slider_width_changed)
        layout.addWidget(self.slider_width)

        self.lbl_width_val = QLabel(f"{self.stroke_width}px")
        self.lbl_width_val.setFixedWidth(26)
        layout.addWidget(self.lbl_width_val)

        # Magnifier Zoom Stepper (Cycles 1.5x -> 2.0x -> 2.5x -> 3.0x -> 4.0x)
        self.btn_magnify_zoom = QToolButton()
        self.btn_magnify_zoom.setText(f"{self.magnifier_zoom:.1f}x")
        self.btn_magnify_zoom.setToolTip("Magnification power (click to cycle)")
        self.btn_magnify_zoom.setStyleSheet("""
            QToolButton {
                font-weight: bold;
                font-size: 10px;
                padding: 2px 4px;
                border: 1px solid #94A3B8;
                border-radius: 3px;
                background: #F1F5F9;
                color: #1E3A8A;
            }
            QToolButton:hover {
                background: #DBEAFE;
                border-color: #3B82F6;
            }
        """)
        self.btn_magnify_zoom.clicked.connect(self._cycle_magnifier_zoom)
        layout.addWidget(self.btn_magnify_zoom)

        # Text Formatting: Bold & Italic Toggles
        self.btn_bold = QToolButton()
        self.btn_bold.setText("B")
        self.btn_bold.setCheckable(True)
        self.btn_bold.setChecked(self.is_text_bold)
        self.btn_bold.setToolTip("Bold text label")
        self.btn_bold.setStyleSheet("""
            QToolButton {
                font-weight: 900;
                font-size: 11px;
                min-width: 18px;
                padding: 2px 4px;
            }
        """)
        self.btn_bold.toggled.connect(self._on_bold_toggled)
        layout.addWidget(self.btn_bold)

        self.btn_italic = QToolButton()
        self.btn_italic.setText("I")
        self.btn_italic.setCheckable(True)
        self.btn_italic.setChecked(self.is_text_italic)
        self.btn_italic.setToolTip("Italic text label")
        self.btn_italic.setStyleSheet("""
            QToolButton {
                font-style: italic;
                font-weight: bold;
                font-size: 11px;
                min-width: 18px;
                padding: 2px 4px;
            }
        """)
        self.btn_italic.toggled.connect(self._on_italic_toggled)
        layout.addWidget(self.btn_italic)

        # Divider
        div2 = QFrame()
        div2.setFrameShape(QFrame.VLine)
        div2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(div2)

        # Photoshop Dual Color Picker (Primary foreground + Secondary background + Swap)
        self.color_picker = _PhotoshopColorPickerWidget(self)
        self.active_color_chip = self.color_picker  # Backward compatibility alias
        self.color_picker.set_primary_color(self.current_color)
        self.color_picker.set_secondary_color(self.secondary_color)
        self.color_picker.primary_color_changed.connect(self._on_primary_color_changed)
        self.color_picker.secondary_color_changed.connect(self._on_secondary_color_changed)
        self.color_picker.swap_requested.connect(self._on_colors_swapped)
        layout.addWidget(self.color_picker)

        # Quick Color Swatches with Left-click (Primary) and Right-click (Secondary)
        self.swatch_buttons: dict[str, _SwatchButton] = {}
        for hex_col, name in self.QUICK_COLORS:
            col = QColor(0, 0, 0, 0) if hex_col == "TRANSPARENT" else QColor(hex_col)
            btn_swatch = _SwatchButton(col, name, self)
            btn_swatch.left_clicked.connect(self.set_primary_color)
            btn_swatch.right_clicked.connect(self.set_secondary_color)
            layout.addWidget(btn_swatch)
            self.swatch_buttons[hex_col.upper()] = btn_swatch

        # Custom color picker
        self.btn_custom_color = QToolButton()
        self.btn_custom_color.setText("...")
        self.btn_custom_color.setToolTip("Choose custom color\n• Left-click: Primary color\n• Right-click: Secondary color")
        self.btn_custom_color.clicked.connect(self._pick_custom_color_primary)
        self.btn_custom_color.setContextMenuPolicy(Qt.CustomContextMenu)
        self.btn_custom_color.customContextMenuRequested.connect(lambda _: self._pick_custom_color_secondary())
        layout.addWidget(self.btn_custom_color)
        self._update_color_indicators()

        # Divider
        div3 = QFrame()
        div3.setFrameShape(QFrame.VLine)
        div3.setFrameShadow(QFrame.Sunken)
        layout.addWidget(div3)

        # Undo button
        self.btn_undo = QToolButton()
        self.btn_undo.setText("Undo")
        self.btn_undo.setIcon(make_wlpg_icon("undo", 14))
        self.btn_undo.setIconSize(QSize(14, 14))
        self.btn_undo.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_undo.setToolTip("Undo last markup element (Ctrl+Z)")
        self.btn_undo.clicked.connect(self.undo_requested.emit)
        layout.addWidget(self.btn_undo)

        # Clear All
        self.btn_clear = QToolButton()
        self.btn_clear.setText("Clear")
        self.btn_clear.setIcon(make_wlpg_icon("delete", 14))
        self.btn_clear.setIconSize(QSize(14, 14))
        self.btn_clear.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_clear.setToolTip("Clear all annotations from photo")
        self.btn_clear.clicked.connect(self.clear_requested.emit)
        layout.addWidget(self.btn_clear)

        layout.addSpacing(3)

        # Save photo with markups
        self.btn_save = QPushButton("Save")
        self.btn_save.setObjectName("MarkupSaveBtn")
        self.btn_save.setIcon(make_wlpg_icon("save", 14))
        self.btn_save.setIconSize(QSize(14, 14))
        self.btn_save.setToolTip("Save changes to original photo (Ctrl+S)")
        self.btn_save.clicked.connect(self.save_requested.emit)
        layout.addWidget(self.btn_save)

        # Apply / Done
        self.btn_apply = QPushButton("Done")
        self.btn_apply.setIcon(make_symbol_icon("check", 14))
        self.btn_apply.setIconSize(QSize(14, 14))
        self.btn_apply.setToolTip("Finish markup session (Esc)")
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3D8CE2, stop:1 #246CB8);
                color: #FFFFFF;
                font-weight: bold;
                border: 1px solid #1A5494;
                border-radius: 3px;
                padding: 3px 7px;
                min-width: 0px;
                font-size: 11px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4E9BF0, stop:1 #2F78C8);
            }
        """)
        self.btn_apply.clicked.connect(self.apply_requested.emit)
        layout.addWidget(self.btn_apply)

        # Cancel / Close
        self.btn_cancel = QPushButton("Close")
        self.btn_cancel.setObjectName("MarkupCancelBtn")
        self.btn_cancel.setIcon(make_symbol_icon("cross", 14))
        self.btn_cancel.setIconSize(QSize(14, 14))
        self.btn_cancel.clicked.connect(self.cancel_requested.emit)
        layout.addWidget(self.btn_cancel)

        # Initialize contextual visibility for tool-specific buttons
        self._set_tool("pen")
        self.set_dark_mode(False)

    def set_dark_mode(self, is_dark: bool) -> None:
        """Apply Dark Studio or classic light Aero styling to the floating Markup toolbar."""
        if is_dark:
            self.setStyleSheet(f"""
                MarkupFloatingToolbar {{
                    background: #1E242C;
                    border: 1px solid #3B4252;
                    border-radius: 6px;
                    padding: 3px;
                }}
                QLabel {{
                    font-size: 11px;
                    color: #E2E8F0;
                    background: transparent;
                }}
                QLabel#MarkupTitleLabel {{
                    font-weight: bold;
                    color: #F8FAFC;
                    padding-right: 2px;
                    background: transparent;
                }}
                QToolButton {{
                    border: 1px solid transparent;
                    border-radius: 3px;
                    padding: 2px 4px;
                    min-width: 0px;
                    min-height: 22px;
                    font-size: 11px;
                    font-weight: 500;
                    color: #E2E8F0;
                    background: transparent;
                }}
                QToolButton:hover {{
                    background: #2C3440;
                    border: 1px solid #4B5563;
                }}
                QToolButton:checked {{
                    background: #1E3A5F;
                    border: 1px solid #3B82F6;
                    color: #FFFFFF;
                }}
                QToolButton#btn_shape {{
                    min-width: 54px;
                    padding-right: 15px;
                }}
                QToolButton#btn_shape:hover::menu-button,
                QToolButton#btn_shape:checked::menu-button {{
                    border-left: 1px solid rgba(96, 165, 250, 130);
                }}
                QToolButton::menu-arrow {{
                    image: url("{ARROW_DROPDOWN_PATH}");
                    width: 7px;
                    height: 4px;
                }}
                QPushButton#MarkupSaveBtn,
                QPushButton#MarkupCancelBtn {{
                    background: #262B34;
                    border: 1px solid #3B4252;
                    border-radius: 3px;
                    padding: 3px 7px;
                    min-width: 0px;
                    font-size: 11px;
                    font-weight: 600;
                    color: #E2E8F0;
                }}
                QPushButton#MarkupSaveBtn:hover,
                QPushButton#MarkupCancelBtn:hover {{
                    background: #2E3440;
                    border-color: #60A5FA;
                }}
            """)
            if hasattr(self, "btn_magnify_zoom"):
                self.btn_magnify_zoom.setStyleSheet("""
                    QToolButton {
                        font-weight: bold;
                        font-size: 10px;
                        padding: 2px 4px;
                        border: 1px solid #475569;
                        border-radius: 3px;
                        background: #252B36;
                        color: #93C5FD;
                    }
                    QToolButton:hover {
                        background: #334155;
                        border-color: #60A5FA;
                    }
                """)
        else:
            self.setStyleSheet(f"""
                MarkupFloatingToolbar {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FDFDFE, stop:1 #EFF4F9);
                    border: 1px solid #70A6E8;
                    border-radius: 6px;
                    padding: 3px;
                }}
                QLabel {{
                    font-size: 11px;
                    color: #1E395B;
                    background: transparent;
                }}
                QLabel#MarkupTitleLabel {{
                    font-weight: bold;
                    color: #1E395B;
                    padding-right: 2px;
                    background: transparent;
                }}
                QToolButton {{
                    border: 1px solid transparent;
                    border-radius: 3px;
                    padding: 2px 4px;
                    min-width: 0px;
                    min-height: 22px;
                    font-size: 11px;
                    font-weight: 500;
                    color: #1E395B;
                    background: transparent;
                }}
                QToolButton:hover {{
                    background: #E1ECF9;
                    border: 1px solid #85B5EB;
                }}
                QToolButton:checked {{
                    background: #CBE2F8;
                    border: 1px solid #3388E8;
                    color: #0E294B;
                }}
                QToolButton#btn_shape {{
                    min-width: 54px;
                    padding-right: 15px;
                }}
                QToolButton#btn_shape:hover::menu-button,
                QToolButton#btn_shape:checked::menu-button {{
                    border-left: 1px solid rgba(112, 166, 232, 130);
                }}
                QToolButton::menu-arrow {{
                    image: url("{ARROW_DROPDOWN_PATH}");
                    width: 7px;
                    height: 4px;
                }}
                QPushButton#MarkupSaveBtn {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FDFDFE, stop:1 #EFF4F9);
                    border: 1px solid #7DA2CE;
                    border-radius: 3px;
                    padding: 3px 7px;
                    min-width: 0px;
                    font-size: 11px;
                    font-weight: 600;
                    color: #1E395B;
                }}
                QPushButton#MarkupSaveBtn:hover {{
                    background: #E1ECF9;
                    border-color: #3388E8;
                }}
                QPushButton#MarkupCancelBtn {{
                    background: #F8F9FA;
                    border: 1px solid #CBD5E1;
                    border-radius: 3px;
                    padding: 3px 7px;
                    min-width: 0px;
                    font-size: 11px;
                    color: #1E293B;
                }}
                QPushButton#MarkupCancelBtn:hover {{
                    background: #EDF2F7;
                    border-color: #94A3B8;
                }}
            """)
            if hasattr(self, "btn_magnify_zoom"):
                self.btn_magnify_zoom.setStyleSheet("""
                    QToolButton {
                        font-weight: bold;
                        font-size: 10px;
                        padding: 2px 4px;
                        border: 1px solid #94A3B8;
                        border-radius: 3px;
                        background: #F1F5F9;
                        color: #1E3A8A;
                    }
                    QToolButton:hover {
                        background: #DBEAFE;
                        border-color: #3B82F6;
                    }
                """)

    def _update_shape_button_ui(self) -> None:
        label_map = {"rect": "Box", "ellipse": "Oval", "star": "Star"}
        tip_map = {
            "rect": "Rounded rectangle callout (click arrow to swap to Oval or Star)",
            "ellipse": "Oval / Circle callout (click arrow to swap to Box or Star)",
            "star": "Star callout (click arrow to swap to Box or Oval)",
        }
        name = label_map.get(self.active_shape, "Box")
        self.btn_shape.setText(name)
        ico_name = "star_shape" if self.active_shape == "star" else self.active_shape
        self.btn_shape.setIcon(make_wlpg_icon(ico_name, 16))
        self.btn_shape.setIconSize(QSize(16, 16))
        self.btn_shape.setToolTip(tip_map.get(self.active_shape, "Shape callout"))

    def set_active_shape(self, shape_id: str) -> None:
        if shape_id in ("rect", "ellipse", "star"):
            self.active_shape = shape_id
            self._update_shape_button_ui()
            self._set_tool(shape_id)

    def _update_color_indicators(self) -> None:
        """Updates active swatch indications for Primary and Secondary colors."""
        p_hex = self.current_color.name().upper()
        s_hex = self.secondary_color.name().upper()

        if hasattr(self, "color_picker"):
            self.color_picker.setStyleSheet(f"/* {p_hex} {s_hex} */")

        for hex_col, btn in self.swatch_buttons.items():
            if getattr(btn, "is_transparent", False):
                is_p = (self.current_color.alpha() == 0)
                is_s = (self.secondary_color.alpha() == 0)
                if is_p and is_s:
                    btn.setText("•")
                elif is_p:
                    btn.setText("•")
                elif is_s:
                    btn.setText("◦")
                else:
                    btn.setText("")
                btn.update()
                continue

            is_p = (hex_col == p_hex and self.current_color.alpha() != 0)
            is_s = (hex_col == s_hex and self.secondary_color.alpha() != 0)

            if is_p and is_s:
                btn.setText("•")
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {hex_col};
                        color: #00FFCC;
                        font-weight: 900;
                        font-size: 14px;
                        border: 2px solid #005FB8;
                        border-radius: 8px;
                        padding-bottom: 2px;
                    }}
                """)
            elif is_p:
                dot_color = "#FFFFFF" if hex_col in ("#212121", "#E53935", "#2196F3", "#4CAF50") else "#000000"
                btn.setText("•")
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {hex_col};
                        color: {dot_color};
                        font-weight: 900;
                        font-size: 14px;
                        border: 2px solid #005FB8;
                        border-radius: 8px;
                        padding-bottom: 2px;
                    }}
                """)
            elif is_s:
                dot_color = "#FFFFFF" if hex_col in ("#212121", "#E53935", "#2196F3", "#4CAF50") else "#000000"
                btn.setText("○")
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {hex_col};
                        color: {dot_color};
                        font-weight: bold;
                        font-size: 11px;
                        border: 2px dashed #334155;
                        border-radius: 8px;
                        padding-bottom: 1px;
                    }}
                """)
            else:
                border_col = "#999999" if hex_col == "#FFFFFF" else hex_col
                btn.setText("")
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {hex_col};
                        border: 1px solid {border_col};
                        border-radius: 8px;
                    }}
                    QPushButton:hover {{
                        border: 2px solid #005FB8;
                    }}
                """)

    def set_compact_mode(self, compact: bool, ultra_compact: bool = False) -> None:
        """
        Dynamically adapts button presentation between ToolButtonTextBesideIcon (wide viewport),
        ToolButtonIconOnly (compact viewport), and hidden swatch circles (ultra-compact viewport).
        """
        self._is_compact = bool(compact)
        self._is_ultra_compact = bool(ultra_compact)
        style = Qt.ToolButtonIconOnly if compact else Qt.ToolButtonTextBesideIcon
        for btn in self.tool_buttons.values():
            btn.setToolButtonStyle(style)
            if btn is getattr(self, "btn_shape", None):
                continue
            if compact:
                btn.setFixedWidth(25)
            else:
                btn.setMinimumWidth(0)
                btn.setMaximumWidth(16777215)
        for attr in ("btn_undo", "btn_clear"):
            b = getattr(self, attr, None)
            if b is not None:
                b.setToolButtonStyle(style)
                if compact:
                    b.setFixedWidth(25)
                else:
                    b.setMinimumWidth(0)
                    b.setMaximumWidth(16777215)
        if hasattr(self, "btn_custom_color"):
            if compact:
                self.btn_custom_color.setFixedWidth(22)
            else:
                self.btn_custom_color.setMinimumWidth(0)
                self.btn_custom_color.setMaximumWidth(16777215)
        if hasattr(self, "lbl_title"):
            self.lbl_title.setVisible(not compact)
        if hasattr(self, "lbl_size"):
            self.lbl_size.setVisible(not compact)
        if hasattr(self, "slider_width"):
            self.slider_width.setFixedWidth(44 if compact else 60)
        if hasattr(self, "btn_shape"):
            if compact:
                self.btn_shape.setFixedWidth(36)
            else:
                self.btn_shape.setMinimumWidth(54)
                self.btn_shape.setMaximumWidth(16777215)
        for btn_swatch in getattr(self, "swatch_buttons", {}).values():
            btn_swatch.setVisible(not ultra_compact)
        if self.layout() is not None:
            self.layout().invalidate()
            self.layout().activate()
        self.adjustSize()

    def _set_tool(self, tool_id: str) -> None:
        norm = "magnify" if tool_id in ("magnify", "magnifier") else tool_id
        if norm in ALL_SHAPE_KINDS:
            self.active_shape = norm
            if norm in ("rect", "ellipse", "star"):
                self._update_shape_button_ui()
        elif norm in ALL_LINE_KINDS:
            self.line_style = norm

        self.active_tool = norm
        if norm in self.tool_buttons and not self.tool_buttons[norm].isChecked():
            self.tool_buttons[norm].setChecked(True)
        if hasattr(self, "lbl_size"):
            if norm == "text":
                self.lbl_size.setText("Font:")
                self.slider_width.setToolTip("Adjust font size ([ and ] keys)")
            else:
                self.lbl_size.setText("Size:")
                self.slider_width.setToolTip("Adjust stroke width ([ and ] keys)")
        if hasattr(self, "btn_magnify_zoom"):
            self.btn_magnify_zoom.setVisible(norm == "magnify")
        if hasattr(self, "btn_bold"):
            self.btn_bold.setVisible(norm == "text")
        if hasattr(self, "btn_italic"):
            self.btn_italic.setVisible(norm == "text")
        self.adjustSize()
        self.tool_changed.emit(norm)

    def set_tool(self, tool_id: str) -> None:
        self._set_tool(tool_id)

    def set_stroke_width(self, width: int) -> None:
        clamped = max(2, min(200, int(width)))
        self.stroke_width = clamped
        if self.slider_width.value() != clamped:
            self.slider_width.blockSignals(True)
            self.slider_width.setValue(clamped)
            self.slider_width.blockSignals(False)
        self.lbl_width_val.setText(f"{clamped}px")
        self.stroke_width_changed.emit(clamped)

    def _on_slider_width_changed(self, val: int) -> None:
        self.set_stroke_width(val)

    def set_color(self, col: QColor) -> None:
        self.set_primary_color(col)

    def set_primary_color(self, col: QColor) -> None:
        if col.isValid():
            self.current_color = col
            if hasattr(self, "color_picker"):
                self.color_picker.blockSignals(True)
                self.color_picker.set_primary_color(col)
                self.color_picker.blockSignals(False)
            self._update_color_indicators()
            self.color_changed.emit(col)

    def set_secondary_color(self, col: QColor) -> None:
        if col.isValid():
            self.secondary_color = col
            if col.alpha() > 0:
                self._last_opaque_secondary_color = QColor(col)
            if hasattr(self, "color_picker"):
                self.color_picker.blockSignals(True)
                self.color_picker.set_secondary_color(col)
                self.color_picker.blockSignals(False)
            self._update_color_indicators()
            self.secondary_color_changed.emit(col)

    def is_pill_bg_enabled(self) -> bool:
        return self.secondary_color.alpha() > 0

    def set_pill_bg_enabled(self, enabled: bool) -> None:
        if enabled:
            restore = getattr(self, "_last_opaque_secondary_color", QColor(20, 25, 35))
            if not restore.isValid() or restore.alpha() == 0:
                restore = QColor(20, 25, 35)
            self.set_secondary_color(QColor(restore))
            self.set_text_style("pill")
        else:
            self.set_secondary_color(QColor(0, 0, 0, 0))
            if self.text_style == "pill":
                self.set_text_style("plain")

    def swap_colors(self) -> None:
        if hasattr(self, "color_picker"):
            self.color_picker.blockSignals(True)
            self.color_picker.swap_colors()
            self.color_picker.blockSignals(False)
            self.current_color = self.color_picker.primary_color
            self.secondary_color = self.color_picker.secondary_color
        else:
            p, s = self.current_color, self.secondary_color
            self.current_color = s
            self.secondary_color = p
        if self.secondary_color.alpha() > 0:
            self._last_opaque_secondary_color = QColor(self.secondary_color)
        self._update_color_indicators()
        self.color_changed.emit(self.current_color)
        self.secondary_color_changed.emit(self.secondary_color)
        self.colors_swapped.emit(self.current_color, self.secondary_color)

    def _on_primary_color_changed(self, col: QColor) -> None:
        self.current_color = col
        self._update_color_indicators()
        self.color_changed.emit(col)

    def _on_secondary_color_changed(self, col: QColor) -> None:
        self.secondary_color = col
        if col.alpha() > 0:
            self._last_opaque_secondary_color = QColor(col)
        self._update_color_indicators()
        self.secondary_color_changed.emit(col)

    def _on_colors_swapped(self) -> None:
        self.current_color = self.color_picker.primary_color
        self.secondary_color = self.color_picker.secondary_color
        if self.secondary_color.alpha() > 0:
            self._last_opaque_secondary_color = QColor(self.secondary_color)
        self._update_color_indicators()
        self.colors_swapped.emit(self.current_color, self.secondary_color)

    def _pick_custom_color_primary(self) -> None:
        col = QColorDialog.getColor(self.current_color, self, "Select Primary (Foreground) Color")
        if col.isValid():
            self.set_primary_color(col)

    def _pick_custom_color_secondary(self) -> None:
        col = QColorDialog.getColor(self.secondary_color, self, "Select Secondary (Background) Color")
        if col.isValid():
            self.set_secondary_color(col)

    def _pick_color(self) -> None:
        self._pick_custom_color_primary()

    def _cycle_magnifier_zoom(self) -> None:
        steps = [1.5, 2.0, 2.5, 3.0, 4.0]
        cur = round(self.magnifier_zoom, 1)
        next_idx = 0
        for i, val in enumerate(steps):
            if abs(val - cur) < 0.15:
                next_idx = (i + 1) % len(steps)
                break
        self.set_magnifier_zoom(steps[next_idx])

    def set_magnifier_zoom(self, zoom: float) -> None:
        self.magnifier_zoom = max(1.2, min(5.0, float(zoom)))
        if hasattr(self, "btn_magnify_zoom"):
            self.btn_magnify_zoom.setText(f"{self.magnifier_zoom:.1f}x")
        self.magnifier_zoom_changed.emit(self.magnifier_zoom)

    def set_font_family(self, family: str) -> None:
        fam = (family or "").strip() or "Segoe UI"
        if fam == self.font_family:
            return
        self.font_family = fam
        self.font_family_changed.emit(fam)

    def set_text_bold(self, bold: bool) -> None:
        val = bool(bold)
        if val == self.is_text_bold:
            return
        self.is_text_bold = val
        if hasattr(self, "btn_bold") and self.btn_bold.isChecked() != val:
            self.btn_bold.blockSignals(True)
            self.btn_bold.setChecked(val)
            self.btn_bold.blockSignals(False)
        self.text_format_changed.emit(self.is_text_bold, self.is_text_italic)

    def set_text_italic(self, italic: bool) -> None:
        val = bool(italic)
        if val == self.is_text_italic:
            return
        self.is_text_italic = val
        if hasattr(self, "btn_italic") and self.btn_italic.isChecked() != val:
            self.btn_italic.blockSignals(True)
            self.btn_italic.setChecked(val)
            self.btn_italic.blockSignals(False)
        self.text_format_changed.emit(self.is_text_bold, self.is_text_italic)

    def set_text_style(self, style: str) -> None:
        norm = (style or "pill").strip().lower()
        if norm not in ("pill", "outline", "shadow", "plain"):
            norm = "pill"
        if norm == self.text_style:
            return
        self.text_style = norm
        if norm == "pill" and self.secondary_color.alpha() == 0:
            restore = getattr(self, "_last_opaque_secondary_color", QColor(20, 25, 35))
            if not restore.isValid() or restore.alpha() == 0:
                restore = QColor(20, 25, 35)
            self.set_secondary_color(QColor(restore))
        self.text_style_changed.emit(norm)

    def set_text_align(self, align: str) -> None:
        norm = (align or "left").strip().lower()
        if norm not in ("left", "center", "right"):
            norm = "left"
        if norm == self.text_align:
            return
        self.text_align = norm
        self.text_align_changed.emit(norm)

    def set_opacity(self, opacity: float) -> None:
        clamped = round(max(0.15, min(1.0, float(opacity))), 2)
        if abs(clamped - self.opacity) < 1e-3:
            return
        self.opacity = clamped
        self.opacity_changed.emit(clamped)

    def set_bg_opacity(self, opacity: float) -> None:
        clamped = round(max(0.0, min(1.0, float(opacity))), 2)
        if abs(clamped - self.bg_opacity) < 1e-3:
            return
        self.bg_opacity = clamped
        self.bg_opacity_changed.emit(clamped)

    def set_shape_filled(self, filled: bool) -> None:
        val = bool(filled)
        if val == self.shape_filled:
            return
        self.shape_filled = val
        self.shape_filled_changed.emit(val)

    def set_line_style(self, style: str) -> None:
        norm = (style or "arrow").strip().lower()
        if norm not in ALL_LINE_KINDS:
            norm = "arrow"
        self.line_style = norm
        if self.active_tool in ALL_LINE_KINDS:
            self.active_tool = norm
        self.line_style_changed.emit(norm)

    def set_dash_style(self, dash_style: str) -> None:
        norm = (dash_style or "solid").strip().lower()
        if norm not in ("solid", "outlined", "dashed", "dotted"):
            norm = "solid"
        if norm == getattr(self, "dash_style", "solid"):
            return
        self.dash_style = norm
        self.dash_style_changed.emit(norm)

    def set_active_sticker(self, sticker_id: str) -> None:
        sid = (sticker_id or "heart").strip().lower()
        if sid not in ALL_STICKER_IDS:
            sid = "heart"
        self.active_sticker = sid
        self.sticker_changed.emit(sid)

    def set_obscure_mode(self, mode: str) -> None:
        norm = (mode or "brush").strip().lower()
        if norm not in ("brush", "box"):
            norm = "brush"
        if norm == self.obscure_mode:
            return
        self.obscure_mode = norm
        self.obscure_mode_changed.emit(norm)

    def set_obscure_strength(self, strength: int) -> None:
        clamped = max(15, min(100, int(strength)))
        if clamped == self.obscure_strength:
            return
        self.obscure_strength = clamped
        self.obscure_strength_changed.emit(clamped)

    def _on_bold_toggled(self, checked: bool) -> None:
        self.is_text_bold = bool(checked)
        self.text_format_changed.emit(self.is_text_bold, self.is_text_italic)

    def _on_italic_toggled(self, checked: bool) -> None:
        self.is_text_italic = bool(checked)
        self.text_format_changed.emit(self.is_text_bold, self.is_text_italic)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._is_dragging = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if getattr(self, "_is_dragging", False) and (event.buttons() & Qt.LeftButton):
            p = self.parentWidget()
            if p is not None:
                new_pos = event.globalPosition().toPoint() - self._drag_pos
                max_x = max(10, p.width() - self.width() - 10)
                max_y = max(10, p.height() - self.height() - 10)
                self.move(max(10, min(new_pos.x(), max_x)), max(10, min(new_pos.y(), max_y)))
                self._user_positioned = True
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._is_dragging = False
        super().mouseReleaseEvent(event)
