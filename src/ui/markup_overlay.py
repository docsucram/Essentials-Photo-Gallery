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
    QFontMetrics,
    QPolygonF,
    QCursor,
    QImage,
    QPixmap,
    QIcon,
    QMouseEvent,
    QAction,
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


@dataclass
class MarkupElement:
    kind: str  # 'pen', 'highlighter', 'arrow', 'rect', 'ellipse', 'star', 'pixelate', 'blur', 'magnifier', 'text'
    rect: QRectF | None = None
    start_pt: QPointF | None = None
    end_pt: QPointF | None = None
    points: list[QPointF] = field(default_factory=list)
    color: QColor = field(default_factory=lambda: QColor(230, 40, 40))
    bg_color: QColor | None = None
    stroke_width: int = 4
    text: str = ""
    zoom_factor: float = 2.5
    font_bold: bool = True
    font_italic: bool = False


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

    # Clip circle for zoomed content
    painter.save()
    clip_path = QPainterPath()
    clip_path.addEllipse(QPointF(center, center), radius, radius)
    painter.setClipPath(clip_path)
    painter.drawImage(QRectF(center - radius, center - radius, radius * 2.0, radius * 2.0), scaled_patch)
    painter.restore()

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


def get_markup_element_bounds(el: MarkupElement | None) -> QRectF:
    """Returns the scene bounding rectangle for an element, including stroke and arrowhead margin."""
    if el is None:
        return QRectF()
    margin = max(8.0, float(el.stroke_width) * 1.5)

    if el.kind in ("rect", "ellipse", "star", "pixelate", "blur"):
        if el.rect is not None:
            return el.rect.normalized().adjusted(-margin, -margin, margin, margin)
        return QRectF()

    elif el.kind == "arrow":
        if el.start_pt is not None and el.end_pt is not None:
            x1 = min(el.start_pt.x(), el.end_pt.x())
            y1 = min(el.start_pt.y(), el.end_pt.y())
            x2 = max(el.start_pt.x(), el.end_pt.x())
            y2 = max(el.start_pt.y(), el.end_pt.y())
            arrow_m = max(margin, float(el.stroke_width) * 3.6)
            return QRectF(x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)).adjusted(-arrow_m, -arrow_m, arrow_m, arrow_m)
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
            font = QFont("Segoe UI", max(10, int(el.stroke_width * 2.2)))
            font.setBold(getattr(el, "font_bold", True))
            font.setItalic(getattr(el, "font_italic", False))
            fm = QFontMetrics(font)
            lines = el.text.split("\n")
            max_line_w = max((fm.horizontalAdvance(line) for line in lines), default=40)
            line_h = fm.lineSpacing()
            total_text_h = line_h * max(1, len(lines))
            pad_x, pad_y = 12.0, 8.0
            pill_w = max_line_w + 2 * pad_x
            pill_h = total_text_h + 2 * pad_y
            return QRectF(el.rect.x(), el.rect.y(), pill_w, pill_h).adjusted(-4, -4, 4, 4)
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
    if el.kind in ("pixelate", "blur") and base_rgb is not None and el.rect is not None:
        base_h, base_w = base_rgb.shape[:2]
        cl, ct = (crop_rect[0], crop_rect[1]) if crop_rect is not None else (0, 0)

        rx = max(0, int(round(el.rect.x() + cl)))
        ry = max(0, int(round(el.rect.y() + ct)))
        rw = max(1, min(base_w - rx, int(round(el.rect.width()))))
        rh = max(1, min(base_h - ry, int(round(el.rect.height()))))

        pm = QPixmap(w, h)
        pm.fill(Qt.transparent)
        if rw > 4 and rh > 4:
            patch = base_rgb[ry:ry+rh, rx:rx+rw].copy()
            if el.kind == "pixelate":
                bs = max(8, int(min(rw, rh) / 8))
                patch = apply_pixelate(patch, (0, 0, rw, rh), block_size=bs)
            else:
                rad = max(15, int(min(rw, rh) / 4)) | 1
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
        font_bold=getattr(el, "font_bold", True),
        font_italic=getattr(el, "font_italic", False),
    )

    pm = QPixmap(w, h)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

    if shifted.kind == "pen" and len(shifted.points) >= 2:
        pen = QPen(shifted.color, max(1, shifted.stroke_width), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        path = QPainterPath()
        path.moveTo(shifted.points[0])
        for pt in shifted.points[1:]:
            path.lineTo(pt)
        painter.drawPath(path)

    elif shifted.kind == "highlighter" and len(shifted.points) >= 2:
        hl_w = max(8, int(shifted.stroke_width * 2.6))
        path = QPainterPath()
        path.moveTo(shifted.points[0])
        for pt in shifted.points[1:]:
            path.lineTo(pt)
        stroke_mask = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        stroke_mask.fill(Qt.transparent)
        mp = QPainter(stroke_mask)
        mp.setRenderHint(QPainter.Antialiasing, True)
        solid_col = QColor(shifted.color.red(), shifted.color.green(), shifted.color.blue(), 255)
        mp.setPen(QPen(solid_col, hl_w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        mp.setBrush(Qt.NoBrush)
        mp.drawPath(path)
        mp.end()
        painter.save()
        painter.setOpacity(0.42)
        painter.drawImage(0, 0, stroke_mask)
        painter.restore()

    elif shifted.kind == "arrow" and shifted.start_pt and shifted.end_pt:
        p1 = shifted.start_pt
        p2 = shifted.end_pt
        if math.hypot(p2.x() - p1.x(), p2.y() - p1.y()) >= 4.0:
            p_base, arrowhead = build_arrow_geometry(p1, p2, shifted.stroke_width)
            pen = QPen(shifted.color, max(1, shifted.stroke_width), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(p1, p_base)
            painter.setPen(QPen(shifted.color, 1.0))
            painter.setBrush(QBrush(shifted.color))
            painter.drawPolygon(arrowhead)

    elif shifted.kind == "rect" and shifted.rect is not None:
        pen = QPen(shifted.color, max(1, shifted.stroke_width), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(shifted.rect, 4.0, 4.0)

    elif shifted.kind == "ellipse" and shifted.rect is not None:
        pen = QPen(shifted.color, max(1, shifted.stroke_width), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(shifted.rect)

    elif shifted.kind == "star" and shifted.rect is not None:
        pen = QPen(shifted.color, max(1, shifted.stroke_width), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPolygon(build_star_polygon_in_rect(shifted.rect))

    elif shifted.kind == "text" and shifted.rect is not None and shifted.text:
        font = QFont("Segoe UI", max(10, int(shifted.stroke_width * 2.2)))
        font.setBold(getattr(shifted, "font_bold", True))
        font.setItalic(getattr(shifted, "font_italic", False))
        painter.setFont(font)
        metrics = painter.fontMetrics()
        lines = shifted.text.split("\n")
        max_line_w = max((metrics.horizontalAdvance(l) for l in lines), default=40)
        line_h = metrics.lineSpacing()
        total_text_h = line_h * max(1, len(lines))
        pad_x, pad_y = 12.0, 8.0
        rx, ry = shifted.rect.x(), shifted.rect.y()
        pill_w = max_line_w + 2 * pad_x
        pill_h = total_text_h + 2 * pad_y
        pill_rect = QRectF(rx, ry, pill_w, pill_h)

        sec_col = getattr(shifted, "bg_color", None)
        if sec_col is not None and sec_col.alpha() == 0:
            pass
        else:
            corner_rad = min(12.0, pill_h / 2.0)
            painter.setPen(QPen(QColor(255, 255, 255, 180), 1.2))
            if sec_col is None or not sec_col.isValid():
                sec_brush = QColor(15, 20, 28, 220)
            else:
                sec_brush = QColor(sec_col.red(), sec_col.green(), sec_col.blue(), 220)
            painter.setBrush(sec_brush)
            painter.drawRoundedRect(pill_rect, corner_rad, corner_rad)

        text_color = shifted.color if shifted.color not in (QColor(0, 0, 0), QColor("#000000")) else QColor(255, 255, 255)
        painter.setPen(QPen(text_color))
        cur_y = ry + pad_y + metrics.ascent()
        for line in lines:
            painter.drawText(QPointF(rx + pad_x, cur_y), line)
            cur_y += line_h

    painter.end()
    return pm, top_left


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

    # 1. Apply privacy redactions (pixelate and blur) directly in RGB
    for el in elements:
        if el.kind == "pixelate" and el.rect is not None:
            rx = max(0, int(round(el.rect.x())))
            ry = max(0, int(round(el.rect.y())))
            rw = max(1, min(w - rx, int(round(el.rect.width()))))
            rh = max(1, min(h - ry, int(round(el.rect.height()))))
            if rw > 4 and rh > 4:
                bs = max(8, int(min(rw, rh) / 8))
                result = apply_pixelate(result, (rx, ry, rw, rh), block_size=bs)
        elif el.kind == "blur" and el.rect is not None:
            rx = max(0, int(round(el.rect.x())))
            ry = max(0, int(round(el.rect.y())))
            rw = max(1, min(w - rx, int(round(el.rect.width()))))
            rh = max(1, min(h - ry, int(round(el.rect.height()))))
            if rw > 4 and rh > 4:
                rad = max(15, int(min(rw, rh) / 4)) | 1
                result = apply_blur_redact(result, (rx, ry, rw, rh), radius=rad)

    # 2. Render vector annotations and callouts using QPainter directly in RGB888 / RGBA8888
    # Wrap result in contiguous RGB888 QImage directly without color conversions
    qimg = QImage(result.data, w, h, 3 * w, QImage.Format.Format_RGB888)

    painter = QPainter(qimg)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

    for el in elements:
        # Freehand Pen
        if el.kind == "pen" and len(el.points) >= 2:
            pen = QPen(el.color, max(1, el.stroke_width), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            path = QPainterPath()
            path.moveTo(el.points[0])
            for pt in el.points[1:]:
                path.lineTo(pt)
            painter.drawPath(path)

        # Fluorescent Highlighter (Uniform max-alpha blending without self-bleed)
        elif el.kind == "highlighter" and len(el.points) >= 2:
            hl_w = max(8, int(el.stroke_width * 2.6))
            path = QPainterPath()
            path.moveTo(el.points[0])
            for pt in el.points[1:]:
                path.lineTo(pt)

            # Render path to isolated transparent mask to cap overlapping stroke alpha at exactly alpha_val
            stroke_mask = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
            stroke_mask.fill(Qt.transparent)
            mp = QPainter(stroke_mask)
            mp.setRenderHint(QPainter.Antialiasing, True)
            solid_col = QColor(el.color.red(), el.color.green(), el.color.blue(), 255)
            mp.setPen(QPen(solid_col, hl_w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            mp.setBrush(Qt.NoBrush)
            mp.drawPath(path)
            mp.end()

            # Paint composite mask onto target image at controlled highlighter opacity (0.42)
            painter.save()
            painter.setOpacity(0.42)
            painter.drawImage(0, 0, stroke_mask)
            painter.restore()

        # Vector Arrow with filled arrowhead
        elif el.kind == "arrow" and el.start_pt and el.end_pt:
            p1 = el.start_pt
            p2 = el.end_pt
            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            if math.hypot(dx, dy) >= 4.0:
                p_base, arrowhead = build_arrow_geometry(p1, p2, el.stroke_width)
                # Shaft line
                pen = QPen(el.color, max(1, el.stroke_width), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawLine(p1, p_base)
                # Arrowhead
                painter.setPen(QPen(el.color, 1.0))
                painter.setBrush(QBrush(el.color))
                painter.drawPolygon(arrowhead)

        # Rectangle Callout
        elif el.kind == "rect" and el.rect is not None:
            pen = QPen(el.color, max(1, el.stroke_width), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(el.rect, 4.0, 4.0)

        # Ellipse / Oval Callout
        elif el.kind == "ellipse" and el.rect is not None:
            pen = QPen(el.color, max(1, el.stroke_width), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(el.rect)

        # Star Callout
        elif el.kind == "star" and el.rect is not None:
            pen = QPen(el.color, max(1, el.stroke_width), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(build_star_polygon_in_rect(el.rect))

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

        # Text Callout Pill (Supports multi-line, clean wrapping, and auto-sizing)
        elif el.kind == "text" and el.rect is not None and el.text:
            font = QFont("Segoe UI", max(10, int(el.stroke_width * 2.2)))
            font.setBold(getattr(el, "font_bold", True))
            font.setItalic(getattr(el, "font_italic", False))
            painter.setFont(font)
            metrics = painter.fontMetrics()

            lines = el.text.split("\n")
            max_line_w = max(metrics.horizontalAdvance(line) for line in lines) if lines else 40
            line_h = metrics.lineSpacing()
            total_text_h = line_h * max(1, len(lines))

            pad_x, pad_y = 12.0, 8.0
            rx, ry = el.rect.x(), el.rect.y()
            pill_w = max_line_w + 2 * pad_x
            pill_h = total_text_h + 2 * pad_y
            pill_rect = QRectF(rx, ry, pill_w, pill_h)

            sec_col = getattr(el, "bg_color", None)
            if sec_col is not None and sec_col.alpha() == 0:
                # Transparent secondary color: no background pill drawn!
                pass
            else:
                corner_rad = min(12.0, pill_h / 2.0)
                painter.setPen(QPen(QColor(255, 255, 255, 180), 1.2))
                if sec_col is None or not sec_col.isValid():
                    sec_brush = QColor(15, 20, 28, 220)
                else:
                    sec_brush = QColor(sec_col.red(), sec_col.green(), sec_col.blue(), 220)
                painter.setBrush(sec_brush)
                painter.drawRoundedRect(pill_rect, corner_rad, corner_rad)

            # Use primary color for text font
            text_color = el.color if el.color not in (QColor(0, 0, 0), QColor("#000000")) else QColor(255, 255, 255)
            painter.setPen(QPen(text_color))

            # Render each line with proper leading
            cur_y = ry + pad_y + metrics.ascent()
            for line in lines:
                painter.drawText(QPointF(rx + pad_x, cur_y), line)
                cur_y += line_h

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

    def __init__(self, color: QColor, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.color = color
        self.color_name = name
        self.is_transparent = (color.alpha() == 0 or name.lower().startswith("transparent"))
        self.setFixedSize(16, 16)
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(38, 26)
        self.setToolTip("Primary (Foreground) & Secondary (Background) Colors\n• Click chip to edit color\n• Click ⇄ or press 'X' to swap")
        self.primary_color: QColor = QColor(230, 40, 40)
        self.secondary_color: QColor = QColor(20, 25, 35)

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

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        # Secondary (background) chip behind
        p.setPen(QPen(QColor("#7DA2CE"), 1.0))
        if self.secondary_color.alpha() == 0:
            p.setBrush(QColor("#FFFFFF"))
            p.drawRoundedRect(self.secondary_rect, 2, 2)
            p.setPen(QPen(QColor("#E53935"), 1.5, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(self.secondary_rect.topRight() + QPoint(-1, 1), self.secondary_rect.bottomLeft() + QPoint(1, -1))
        else:
            p.setBrush(self.secondary_color)
            p.drawRoundedRect(self.secondary_rect, 2, 2)

        # Primary (foreground) chip on top
        p.setPen(QPen(QColor("#1E395B"), 1.2))
        if self.primary_color.alpha() == 0:
            p.setBrush(QColor("#FFFFFF"))
            p.drawRoundedRect(self.primary_rect, 2, 2)
            p.setPen(QPen(QColor("#E53935"), 1.5, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(self.primary_rect.topRight() + QPoint(-1, 1), self.primary_rect.bottomLeft() + QPoint(1, -1))
        else:
            p.setBrush(self.primary_color)
            p.drawRoundedRect(self.primary_rect, 2, 2)

        # Subtle swap arrows in top-right
        p.setPen(QPen(QColor("#475569"), 1.2, Qt.SolidLine, Qt.RoundCap))
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
    text_format_changed = Signal(bool, bool)       # is_bold, is_italic
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
        self.current_color = QColor(230, 40, 40)
        self.secondary_color = QColor(20, 25, 35)
        self.stroke_width: int = 4
        self.magnifier_zoom: float = 2.5
        self.is_text_bold: bool = True
        self.is_text_italic: bool = False
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
                padding: 2px 6px;
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
                min-width: 68px;
                padding-right: 18px;
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
        lbl = QLabel("Markup:")
        lbl.setStyleSheet("font-weight: bold; color: #1E395B; padding-right: 2px;")
        layout.addWidget(lbl)

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
        btn_mag.setMinimumWidth(66)
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
        self.slider_width.setRange(2, 36)
        self.slider_width.setValue(self.stroke_width)
        self.slider_width.setFixedWidth(60)
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
        self.btn_save.setIcon(make_wlpg_icon("save", 14))
        self.btn_save.setIconSize(QSize(14, 14))
        self.btn_save.setToolTip("Save changes to original photo (Ctrl+S)")
        self.btn_save.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FDFDFE, stop:1 #EFF4F9);
                border: 1px solid #7DA2CE;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 600;
                color: #1E395B;
            }
            QPushButton:hover {
                background: #E1ECF9;
                border-color: #3388E8;
            }
        """)
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
                padding: 3px 10px;
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
        self.btn_cancel.setIcon(make_symbol_icon("cross", 14))
        self.btn_cancel.setIconSize(QSize(14, 14))
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background: #F8F9FA;
                border: 1px solid #CBD5E1;
                border-radius: 3px;
                padding: 3px 7px;
                font-size: 11px;
                color: #1E293B;
            }
            QPushButton:hover {
                background: #EDF2F7;
                border-color: #94A3B8;
            }
        """)
        self.btn_cancel.clicked.connect(self.cancel_requested.emit)
        layout.addWidget(self.btn_cancel)

        # Initialize contextual visibility for tool-specific buttons
        self._set_tool("pen")

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

    def set_compact_mode(self, compact: bool) -> None:
        """
        Dynamically adapts button presentation between ToolButtonTextBesideIcon (wide viewport)
        and ToolButtonIconOnly (compact/narrow viewport) to eliminate text truncation.
        """
        style = Qt.ToolButtonIconOnly if compact else Qt.ToolButtonTextBesideIcon
        for btn in self.tool_buttons.values():
            btn.setToolButtonStyle(style)
        if hasattr(self, "btn_undo"):
            self.btn_undo.setToolButtonStyle(style)
        if hasattr(self, "btn_clear"):
            self.btn_clear.setToolButtonStyle(style)
        self.adjustSize()

    def _set_tool(self, tool_id: str) -> None:
        norm = "magnify" if tool_id in ("magnify", "magnifier") else tool_id
        if norm in ("rect", "ellipse", "star"):
            self.active_shape = norm
            self._update_shape_button_ui()
            norm = self.active_shape

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
        clamped = max(2, min(48, int(width)))
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
                self.color_picker.set_primary_color(col)
            self._update_color_indicators()
            self.color_changed.emit(col)

    def set_secondary_color(self, col: QColor) -> None:
        if col.isValid():
            self.secondary_color = col
            if hasattr(self, "color_picker"):
                self.color_picker.set_secondary_color(col)
            self._update_color_indicators()
            self.secondary_color_changed.emit(col)

    def swap_colors(self) -> None:
        if hasattr(self, "color_picker"):
            self.color_picker.swap_colors()
        else:
            p, s = self.current_color, self.secondary_color
            self.current_color = s
            self.secondary_color = p
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
        self._update_color_indicators()
        self.secondary_color_changed.emit(col)

    def _on_colors_swapped(self) -> None:
        self.current_color = self.color_picker.primary_color
        self.secondary_color = self.color_picker.secondary_color
        self._update_color_indicators()
        self.color_changed.emit(self.current_color)
        self.secondary_color_changed.emit(self.secondary_color)
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
