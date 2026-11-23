"""Integrated Interactive Collage & 'Then & Now' Workspace."""
from __future__ import annotations

import datetime
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QMimeData,
    QPoint,
    QPointF,
    QPropertyAnimation,
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
    QCursor,
    QDrag,
    QFont,
    QFontMetrics,
    QIcon,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.imaging.create_tools import (
    _apply_rounded_corners,
    _crop_to_aspect,
    _get_seamless_slots,
    compute_canvas_rotation_render_size,
    compute_grid_slots,
    compute_polaroid_card_size,
    compute_polaroid_metrics,
    create_blended_seamless_collage,
    create_grid_collage,
    create_polaroid_scatter_collage,
    create_then_and_now_card,
    default_polaroid_positions_norm,
    detect_faces_in_image,
    rotate_collage_canvas,
)
from src.imaging.creative_effects import load_bgr_exif_safe
from src.ui.create_studio.base import CreateWorkspaceBase, RibbonActionSpec
from src.ui.create_studio.shared_widgets import (
    AeroStudioSlider,
    CreateStudioZoomBar,
    StudioCardGroup,
    ZoomablePreviewCanvas,
    _cv2_to_qpixmap,
    _qcolor_to_bgr,
    build_studio_panel_qss,
    prompt_save_new_file,
)
from src.ui.create_studio_dialogs import recommended_grid_layouts
from src.ui.ribbon_toolbar import make_wlpg_icon
from src.ui.theme_tokens import get_theme_tokens

# Curated collage background colour swatches: (label, hex)
_BG_SWATCHES: list[tuple[str, str]] = [
    ("White", "#FFFFFF"),
    ("Warm Cream", "#F8F5F0"),
    ("Soft Grey", "#E5E7EB"),
    ("Charcoal", "#1F2937"),
    ("Black", "#0F1115"),
    ("Sky Tint", "#DBEAFE"),
    ("Sage Tint", "#DCFCE7"),
    ("Blush Tint", "#FCE7F3"),
]

# Combined visual aspect ratio / print presets:
# (key, short_label, ratio_w, ratio_h, high_px, std_px, tooltip)
_CANVAS_SHAPES: list[tuple[str, str, float, float, tuple[int, int], tuple[int, int], str]] = [
    ("1:1", "Square", 1.0, 1.0, (2880, 2880), (1600, 1600), "Square 1:1 — Social / Instagram / Square Print"),
    ("16:9", "16:9 Wide", 16.0, 9.0, (3840, 2160), (1920, 1080), "Widescreen 16:9 — Desktop / TV"),
    ("9:16", "9:16 Phone", 9.0, 16.0, (2160, 3840), (1080, 1920), "Tall 9:16 — Phone Wallpaper / Story"),
    ("4:3", "4:3 Wide", 4.0, 3.0, (3200, 2400), (1600, 1200), "Classic 4:3 Landscape"),
    ("3:4", "3:4 Tall", 3.0, 4.0, (2400, 3200), (1200, 1600), "Classic 3:4 Portrait"),
    ("3:2", "3:2 (6×4)", 3.0, 2.0, (3600, 2400), (1800, 1200), "6×4 Photo Print (Landscape)"),
    ("2:3", "2:3 (4×6)", 2.0, 3.0, (2400, 3600), (1200, 1800), "4×6 Photo Print (Portrait)"),
    ("5:4", "5:4 (10×8)", 5.0, 4.0, (3000, 2400), (1500, 1200), "10×8 Framed Print (Landscape)"),
    ("4:5", "4:5 Social", 4.0, 5.0, (2400, 3000), (1200, 1500), "4:5 Social Portrait / 8×10 Print"),
    ("a4_land", "A4 Wide", 1.414, 1.0, (3508, 2480), (1754, 1240), "A4 Landscape Print (300 DPI)"),
    ("a4_port", "A4 Tall", 1.0, 1.414, (2480, 3508), (1240, 1754), "A4 Portrait Print (300 DPI)"),
]


def _make_aspect_icon(rw: float, rh: float, fg_hex: str, size: int = 20) -> QIcon:
    """Draw a crisp proportional rectangle icon matching the exact aspect ratio."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    max_dim = size - 5.0
    if rw >= rh:
        w = max_dim
        h = max(5.0, max_dim * (rh / max(0.1, rw)))
    else:
        h = max_dim
        w = max(5.0, max_dim * (rw / max(0.1, rh)))
    x = (size - w) / 2.0
    y = (size - h) / 2.0
    col = QColor(fg_hex)
    fill = QColor(col)
    fill.setAlpha(38)
    p.setPen(QPen(col, 1.6))
    p.setBrush(QBrush(fill))
    p.drawRoundedRect(QRectF(x, y, w, h), 2.2, 2.2)
    p.end()
    return QIcon(pm)


def _make_style_icon(style_idx: int, fg_hex: str, size: int = 20) -> QIcon:
    """Draw a miniature schematic icon for each Collage style."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    col = QColor(fg_hex)
    fill = QColor(col)
    fill.setAlpha(45)
    p.setPen(QPen(col, 1.3))
    p.setBrush(QBrush(fill))
    if style_idx == 0:  # Blended overlapping
        p.drawRoundedRect(QRectF(2.5, 4.0, 10.0, 9.0), 2.0, 2.0)
        p.drawRoundedRect(QRectF(7.5, 7.0, 10.0, 9.0), 2.0, 2.0)
    elif style_idx == 1:  # Grid
        p.drawRoundedRect(QRectF(2.5, 3.0, 6.5, 6.5), 1.2, 1.2)
        p.drawRoundedRect(QRectF(11.0, 3.0, 6.5, 6.5), 1.2, 1.2)
        p.drawRoundedRect(QRectF(2.5, 11.0, 6.5, 6.5), 1.2, 1.2)
        p.drawRoundedRect(QRectF(11.0, 11.0, 6.5, 6.5), 1.2, 1.2)
    elif style_idx == 2:  # Polaroid scatter
        p.save()
        p.translate(7.5, 10.0)
        p.rotate(-12)
        p.drawRoundedRect(QRectF(-4.5, -5.5, 9.0, 11.0), 1.0, 1.0)
        p.restore()
        p.save()
        p.translate(12.8, 10.5)
        p.rotate(10)
        p.drawRoundedRect(QRectF(-4.5, -5.5, 9.0, 11.0), 1.0, 1.0)
        p.restore()
    else:  # Then & Now split
        p.drawRoundedRect(QRectF(2.5, 4.0, 6.8, 12.0), 1.5, 1.5)
        p.drawRoundedRect(QRectF(10.7, 4.0, 6.8, 12.0), 1.5, 1.5)
    p.end()
    return QIcon(pm)


def _make_grid_layout_icon(layout_label: str, count: int, fg_hex: str, w: int = 26, h: int = 18) -> QIcon:
    """Draw a miniature diagram of the actual slot layout for a grid preset."""
    pm = QPixmap(w, h)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    col = QColor(fg_hex)
    fill = QColor(col)
    fill.setAlpha(48)
    p.setPen(QPen(col, 1.1))
    p.setBrush(QBrush(fill))
    norm = layout_label.lower().replace(" ", "_")
    slots = compute_grid_slots(max(1, count), layout=norm, border_thickness=2, target_size=(w, h))
    for sx, sy, sw, sh in slots:
        p.drawRoundedRect(QRectF(sx + 0.5, sy + 0.5, max(2.0, sw - 1.0), max(2.0, sh - 1.0)), 1.2, 1.2)
    p.end()
    return QIcon(pm)


_COLLAGE_MIME = "application/x-collage-slot-idx"


class _DraggableCollageCard(QFrame):
    """Full-bleed filmstrip tile with hover-only remove and live smooth drag reordering."""

    card_clicked = Signal(int)
    card_dropped = Signal(int, int)  # (src_idx, dst_idx) for Qt QDrag compatibility
    remove_clicked = Signal(int)
    drag_started = Signal(int, int)  # (slot_idx, press_offset_x)
    drag_moved = Signal(int)  # global_x
    drag_finished = Signal()
    drag_cancelled = Signal()

    def __init__(self, slot_idx: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.slot_idx = slot_idx
        self._selected: bool = False
        self._hovered: bool = False
        self._dragging: bool = False
        self._can_remove: bool = False
        self._theme_name: str = "wlpg_aero"
        self._src_pixmap: QPixmap | None = None
        self._scaled_pm: QPixmap | None = None
        self._scaled_size: tuple[int, int] = (0, 0)
        self._press_pos: QPoint | None = None
        self._press_offset_x: int = 0
        self._drag_active: bool = False

        self.setObjectName("CollageTrayCard")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        self.btn_remove = QPushButton("✕", self)
        self.btn_remove.setObjectName("TrayMiniBtn")
        self.btn_remove.setFixedSize(18, 18)
        self.btn_remove.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_remove.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_remove.setToolTip("Remove photo from collage")
        self.btn_remove.setStyleSheet(
            """
            QFrame#CollageTrayCard QPushButton#TrayMiniBtn {
                background-color: rgba(15, 23, 42, 200);
                color: #F8FAFC;
                border: 1px solid rgba(255, 255, 255, 85);
                border-radius: 9px;
                padding: 0px;
                font-family: 'Segoe UI';
                font-size: 8pt;
                font-weight: 600;
            }
            QFrame#CollageTrayCard QPushButton#TrayMiniBtn:hover {
                background-color: #DC2626;
                border-color: #FCA5A5;
                color: #FFFFFF;
            }
            """
        )
        self.btn_remove.clicked.connect(lambda _c=False: self.remove_clicked.emit(self.slot_idx))
        self.btn_remove.hide()

    def set_slot_idx(self, idx: int) -> None:
        if self.slot_idx != int(idx):
            self.slot_idx = int(idx)
            self.update()

    def set_selected(self, selected: bool) -> None:
        val = bool(selected)
        if self._selected != val:
            self._selected = val
            self.update()

    def set_can_remove(self, can_remove: bool) -> None:
        val = bool(can_remove)
        if self._can_remove != val:
            self._can_remove = val
            self._sync_remove_btn_visibility()

    def set_theme_name(self, theme_name: str) -> None:
        t = theme_name or "wlpg_aero"
        if self._theme_name != t:
            self._theme_name = t
            self.update()

    def set_pixmap(self, pm: QPixmap | None) -> None:
        self._src_pixmap = pm
        self._scaled_pm = None
        self._scaled_size = (0, 0)
        self.update()

    def _sync_remove_btn_visibility(self) -> None:
        show = bool(self._hovered and self._can_remove and not self._dragging)
        self.btn_remove.setVisible(show)
        if show:
            self.btn_remove.raise_()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self.btn_remove.move(max(4, self.width() - self.btn_remove.width() - 6), 6)

    def enterEvent(self, event: Any) -> None:
        self._hovered = True
        self._sync_remove_btn_visibility()
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self._hovered = False
        self._sync_remove_btn_visibility()
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, _event: Any) -> None:
        tok = get_theme_tokens(self._theme_name)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        outer = self.rect().adjusted(2, 2, -2, -2)
        thumb_rect = outer.adjusted(2, 2, -2, -2)
        if thumb_rect.width() <= 2 or thumb_rect.height() <= 2:
            painter.end()
            return

        if self._dragging:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0, 75), 3.0))
            painter.drawRoundedRect(outer.adjusted(0, 1, 0, 1), 6, 6)
            painter.setBrush(QColor(tok.selection_bg))
            painter.setPen(QPen(QColor(tok.accent), 2.4))
            painter.drawRoundedRect(outer, 6, 6)
        elif self._selected:
            painter.setBrush(QColor(tok.selection_bg))
            painter.setPen(QPen(QColor(tok.accent), 2.2))
            painter.drawRoundedRect(outer, 6, 6)
        elif self._hovered:
            painter.setBrush(QColor(tok.bg_hover))
            painter.setPen(QPen(QColor(tok.border_subtle), 1.0))
            painter.drawRoundedRect(outer, 6, 6)
        else:
            painter.setBrush(QColor(tok.bg_card))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(outer, 6, 6)

        clip_path = QPainterPath()
        clip_path.addRoundedRect(thumb_rect.x(), thumb_rect.y(), thumb_rect.width(), thumb_rect.height(), 4, 4)
        painter.save()
        painter.setClipPath(clip_path)

        if self._src_pixmap is not None and not self._src_pixmap.isNull():
            target_wh = (thumb_rect.width(), thumb_rect.height())
            if self._scaled_pm is None or self._scaled_size != target_wh:
                expanded = self._src_pixmap.scaled(
                    thumb_rect.size(),
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                )
                sx = max(0, (expanded.width() - thumb_rect.width()) // 2)
                sy = max(0, (expanded.height() - thumb_rect.height()) // 2)
                self._scaled_pm = expanded.copy(sx, sy, thumb_rect.width(), thumb_rect.height())
                self._scaled_size = target_wh
            painter.drawPixmap(thumb_rect.topLeft(), self._scaled_pm)
        else:
            painter.fillRect(thumb_rect, QColor(tok.bg_input))

        if not self._selected and not self._hovered and not self._dragging:
            painter.fillRect(thumb_rect, QColor(0, 0, 0, 38))
        painter.restore()

        badge_font = QFont("Segoe UI", 8, QFont.Weight.DemiBold)
        badge_text = f"#{self.slot_idx + 1}"
        fm = QFontMetrics(badge_font)
        bw = max(20, fm.horizontalAdvance(badge_text) + 10)
        bh = 16
        badge_rect = QRect(thumb_rect.left() + 4, thumb_rect.top() + 4, bw, bh)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 23, 42, 185))
        painter.drawRoundedRect(badge_rect, 4, 4)
        painter.setFont(badge_font)
        painter.setPen(QColor("#F8FAFC"))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)
        painter.end()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            self._press_offset_x = int(round(event.position().x()))
            self._drag_active = False
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        if self._press_pos is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            if not self._drag_active:
                dist = (event.position().toPoint() - self._press_pos).manhattanLength()
                if dist >= max(4, QApplication.startDragDistance()):
                    self._drag_active = True
                    self._dragging = True
                    self._sync_remove_btn_visibility()
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                    self.raise_()
                    self.update()
                    self.drag_started.emit(self.slot_idx, self._press_offset_x)
                    self.drag_moved.emit(int(round(event.globalPosition().x())))
            else:
                self.drag_moved.emit(int(round(event.globalPosition().x())))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._press_pos is not None:
            was_dragging = self._drag_active
            self._press_pos = None
            self._drag_active = False
            self._dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self._sync_remove_btn_visibility()
            self.update()
            if was_dragging:
                self.drag_finished.emit()
            else:
                self.card_clicked.emit(self.slot_idx)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and self._drag_active:
            self._press_pos = None
            self._drag_active = False
            self._dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self._sync_remove_btn_visibility()
            self.update()
            self.drag_cancelled.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event: Any) -> None:
        if event.mimeData().hasFormat(_COLLAGE_MIME):
            parent = self.parentWidget()
            if isinstance(parent, _AnimatedCollageFilmstrip):
                try:
                    src_idx = int(bytes(event.mimeData().data(_COLLAGE_MIME)).decode("utf-8"))
                    if parent._drag_src_idx is None:
                        parent.begin_drag(src_idx)
                    parent.set_drag_target_slot(self.slot_idx, animate=True)
                except Exception:
                    pass
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event: Any) -> None:
        if event.mimeData().hasFormat(_COLLAGE_MIME):
            parent = self.parentWidget()
            if isinstance(parent, _AnimatedCollageFilmstrip):
                parent.set_drag_target_slot(self.slot_idx, animate=True)
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: Any) -> None:
        if event.mimeData().hasFormat(_COLLAGE_MIME):
            parent = self.parentWidget()
            if isinstance(parent, _AnimatedCollageFilmstrip) and parent._drag_src_idx is not None:
                parent.cancel_drag()
            try:
                src_idx = int(bytes(event.mimeData().data(_COLLAGE_MIME)).decode("utf-8"))
                if src_idx != self.slot_idx:
                    self.card_dropped.emit(src_idx, self.slot_idx)
                event.acceptProposedAction()
                return
            except Exception:
                pass
        event.ignore()


class _AnimatedCollageFilmstrip(QWidget):
    """Horizontal filmstrip container that animates sibling cards out of the way during drag-and-drop."""

    reorder_committed = Signal(int, int)  # (src_idx, dst_idx)

    PAD_X = 14
    PAD_Y = 4
    CARD_GAP = 6

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StudioTrayHost")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAcceptDrops(True)

        self._theme_name: str = "wlpg_aero"
        self._cards: list[_DraggableCollageCard] = []
        self._anims: list[QPropertyAnimation] = []
        self._drag_src_idx: int | None = None
        self._drag_target_idx: int | None = None
        self._drag_card: _DraggableCollageCard | None = None
        self._drag_offset_x: int = 48
        self._scroll_area: QScrollArea | None = None
        self._autoscroll_dir: int = 0

        self._autoscroll_timer = QTimer(self)
        self._autoscroll_timer.setInterval(16)
        self._autoscroll_timer.timeout.connect(self._on_autoscroll_tick)

    def set_scroll_area(self, scroll_area: QScrollArea) -> None:
        self._scroll_area = scroll_area

    def set_theme_name(self, theme_name: str) -> None:
        t = theme_name or "wlpg_aero"
        if self._theme_name != t:
            self._theme_name = t
            self.update()

    def _tile_size(self) -> tuple[int, int]:
        raw_h = self.height()
        avail_h = 84 if raw_h <= 20 else (raw_h - 2 * self.PAD_Y)
        card_h = max(64, min(124, avail_h))
        card_w = max(86, int(round(card_h * 4 / 3)))
        return card_w, card_h

    def slot_x(self, slot: int) -> int:
        card_w, _ = self._tile_size()
        return self.PAD_X + int(slot) * (card_w + self.CARD_GAP)

    def slot_y(self) -> int:
        _, card_h = self._tile_size()
        return max(self.PAD_Y, (self.height() - card_h) // 2)

    def _update_content_width(self) -> None:
        card_w, _ = self._tile_size()
        n = len(self._cards)
        content_w = self.PAD_X * 2 + n * card_w + max(0, n - 1) * self.CARD_GAP
        content_w = max(content_w, self.PAD_X * 2)
        if self.minimumWidth() != content_w:
            self.setMinimumWidth(content_w)

    def _stop_anims(self) -> None:
        for anim in list(self._anims):
            try:
                anim.stop()
                anim.deleteLater()
            except Exception:
                pass
        self._anims.clear()

    def _animate_card_to(self, card: _DraggableCollageCard, target_pos: QPoint, duration_ms: int = 165) -> None:
        if card.pos() == target_pos:
            for anim in list(self._anims):
                try:
                    if anim.targetObject() is card and anim.endValue() != target_pos:
                        anim.stop()
                except RuntimeError:
                    if anim in self._anims:
                        self._anims.remove(anim)
            return
        for anim in list(self._anims):
            try:
                if anim.targetObject() is card:
                    if anim.state() == QPropertyAnimation.State.Running and anim.endValue() == target_pos:
                        return
                    anim.stop()
                    if anim in self._anims:
                        self._anims.remove(anim)
                    anim.deleteLater()
            except RuntimeError:
                if anim in self._anims:
                    self._anims.remove(anim)

        anim = QPropertyAnimation(card, b"pos", self)
        anim.setDuration(max(60, int(duration_ms)))
        anim.setStartValue(card.pos())
        anim.setEndValue(target_pos)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        def _cleanup_anim(a: QPropertyAnimation = anim) -> None:
            if a in self._anims:
                self._anims.remove(a)
            try:
                a.deleteLater()
            except RuntimeError:
                pass

        anim.finished.connect(_cleanup_anim)
        self._anims.append(anim)
        anim.start()

    def clear_cards(self) -> None:
        self._autoscroll_timer.stop()
        self._autoscroll_dir = 0
        self._stop_anims()
        self._drag_src_idx = None
        self._drag_target_idx = None
        self._drag_card = None
        for card in self._cards:
            try:
                card.setParent(None)
                card.deleteLater()
            except Exception:
                pass
        self._cards.clear()
        self._update_content_width()
        self.update()

    def set_cards(self, cards: list[_DraggableCollageCard]) -> None:
        self._cards = list(cards)
        for card in self._cards:
            card.setParent(self)
            card.drag_started.connect(self.begin_drag)
            card.drag_moved.connect(self.update_drag_global_x)
            card.drag_finished.connect(self.finish_drag)
            card.drag_cancelled.connect(self.cancel_drag)
            card.show()
        self._update_content_width()
        self.layout_cards(animate=False)

    def _visual_slot_for(self, orig_idx: int) -> int:
        if self._drag_src_idx is None or self._drag_target_idx is None:
            return orig_idx
        src = self._drag_src_idx
        dst = self._drag_target_idx
        if src == dst:
            return orig_idx
        if src < dst and src < orig_idx <= dst:
            return orig_idx - 1
        if dst < src and dst <= orig_idx < src:
            return orig_idx + 1
        return orig_idx

    def layout_cards(self, *, animate: bool = False, skip_dragged: bool = True) -> None:
        self._update_content_width()
        card_w, card_h = self._tile_size()
        y = self.slot_y()
        for idx, card in enumerate(self._cards):
            if card.width() != card_w or card.height() != card_h:
                card.setFixedSize(card_w, card_h)
            if skip_dragged and card is self._drag_card:
                continue
            target_pos = QPoint(self.slot_x(self._visual_slot_for(idx)), y)
            if animate:
                self._animate_card_to(card, target_pos, duration_ms=165)
            else:
                card.move(target_pos)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self.layout_cards(animate=False)

    def begin_drag(self, src_idx: int, press_offset_x: int | None = None) -> None:
        if not (0 <= src_idx < len(self._cards)):
            return
        card_w, _ = self._tile_size()
        self._drag_src_idx = int(src_idx)
        self._drag_target_idx = int(src_idx)
        self._drag_card = self._cards[src_idx]
        self._drag_offset_x = card_w // 2 if press_offset_x is None else max(0, min(card_w, int(press_offset_x)))
        self._drag_card.raise_()
        lifted_y = max(0, self.slot_y() - 4)
        self._drag_card.move(self.slot_x(src_idx), lifted_y)
        self.update()

    def set_drag_target_slot(self, target_idx: int, *, animate: bool = True) -> None:
        if self._drag_src_idx is None or not self._cards:
            return
        clamped = max(0, min(len(self._cards) - 1, int(target_idx)))
        if clamped != self._drag_target_idx:
            self._drag_target_idx = clamped
            self.layout_cards(animate=animate, skip_dragged=True)
            self.update()

    def update_drag_global_x(self, global_x: int) -> None:
        if self._drag_card is None or self._drag_src_idx is None or not self._cards:
            return
        card_w, _ = self._tile_size()
        local_pt = self.mapFromGlobal(QPoint(int(global_x), 0))
        min_x = self.PAD_X - 12
        max_x = self.slot_x(len(self._cards) - 1) + 12
        new_x = max(min_x, min(max_x, local_pt.x() - self._drag_offset_x))
        lifted_y = max(0, self.slot_y() - 4)
        self._drag_card.move(new_x, lifted_y)

        center_x = new_x + (card_w / 2.0)
        slot_stride = max(1, card_w + self.CARD_GAP)
        target = max(0, min(len(self._cards) - 1, int((center_x - self.PAD_X) // slot_stride)))
        self.set_drag_target_slot(target, animate=True)

        if self._scroll_area is not None:
            vp = self._scroll_area.viewport()
            vp_x = vp.mapFromGlobal(QPoint(int(global_x), 0)).x()
            if vp_x < 36:
                self._autoscroll_dir = -1
                if not self._autoscroll_timer.isActive():
                    self._autoscroll_timer.start()
            elif vp_x > vp.width() - 36:
                self._autoscroll_dir = 1
                if not self._autoscroll_timer.isActive():
                    self._autoscroll_timer.start()
            else:
                self._autoscroll_dir = 0
                if self._autoscroll_timer.isActive():
                    self._autoscroll_timer.stop()

    def _on_autoscroll_tick(self) -> None:
        if self._scroll_area is None or self._autoscroll_dir == 0 or self._drag_card is None:
            self._autoscroll_timer.stop()
            return
        hbar = self._scroll_area.horizontalScrollBar()
        nxt = hbar.value() + (self._autoscroll_dir * 12)
        hbar.setValue(nxt)
        self.update_drag_global_x(QCursor.pos().x())

    def finish_drag(self) -> tuple[int, int] | None:
        self._autoscroll_timer.stop()
        self._autoscroll_dir = 0
        if self._drag_src_idx is None or self._drag_target_idx is None:
            self._drag_card = None
            return None
        src = self._drag_src_idx
        dst = self._drag_target_idx
        dragged = self._drag_card
        self._drag_src_idx = None
        self._drag_target_idx = None
        self._drag_card = None
        self.update()

        y = self.slot_y()
        if src != dst and 0 <= src < len(self._cards) and 0 <= dst < len(self._cards):
            moved = self._cards.pop(src)
            self._cards.insert(dst, moved)
            for i, card in enumerate(self._cards):
                card.set_slot_idx(i)
                if card is dragged:
                    self._animate_card_to(card, QPoint(self.slot_x(i), y), duration_ms=120)
                else:
                    self._animate_card_to(card, QPoint(self.slot_x(i), y), duration_ms=120)
            self.reorder_committed.emit(src, dst)
        elif dragged is not None:
            self._animate_card_to(dragged, QPoint(self.slot_x(src), y), duration_ms=120)
        return (src, dst)

    def cancel_drag(self) -> None:
        self._autoscroll_timer.stop()
        self._autoscroll_dir = 0
        self._drag_src_idx = None
        self._drag_target_idx = None
        self._drag_card = None
        self.layout_cards(animate=True, skip_dragged=False)
        self.update()

    def paintEvent(self, event: Any) -> None:
        super().paintEvent(event)
        if self._drag_src_idx is None or self._drag_target_idx is None or not self._cards:
            return
        tok = get_theme_tokens(self._theme_name)
        card_w, card_h = self._tile_size()
        socket_rect = QRect(self.slot_x(self._drag_target_idx), self.slot_y(), card_w, card_h).adjusted(3, 3, -3, -3)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        fill = QColor(tok.accent)
        fill.setAlpha(36)
        pen = QPen(QColor(tok.accent), 2.0, Qt.PenStyle.DashLine)
        painter.setBrush(fill)
        painter.setPen(pen)
        painter.drawRoundedRect(socket_rect, 6, 6)
        lbl_col = QColor(tok.accent)
        lbl_col.setAlpha(185)
        painter.setPen(lbl_col)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(socket_rect, Qt.AlignmentFlag.AlignCenter, f"#{self._drag_target_idx + 1}")
        painter.end()

    def dragEnterEvent(self, event: Any) -> None:
        if event.mimeData().hasFormat(_COLLAGE_MIME):
            try:
                src_idx = int(bytes(event.mimeData().data(_COLLAGE_MIME)).decode("utf-8"))
                self.begin_drag(src_idx)
                event.acceptProposedAction()
                return
            except Exception:
                pass
        event.ignore()

    def dragMoveEvent(self, event: Any) -> None:
        if event.mimeData().hasFormat(_COLLAGE_MIME) and self._cards:
            card_w, _ = self._tile_size()
            x = int(event.position().x())
            target = max(0, min(len(self._cards) - 1, int((x - self.PAD_X) // max(1, card_w + self.CARD_GAP))))
            self.set_drag_target_slot(target, animate=True)
            event.acceptProposedAction()
            return
        event.ignore()

    def dragLeaveEvent(self, event: Any) -> None:
        if self._drag_src_idx is not None:
            self.cancel_drag()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: Any) -> None:
        if event.mimeData().hasFormat(_COLLAGE_MIME):
            res = self.finish_drag()
            if res is None:
                try:
                    src_idx = int(bytes(event.mimeData().data(_COLLAGE_MIME)).decode("utf-8"))
                    card_w, _ = self._tile_size()
                    x = int(event.position().x())
                    dst_idx = max(0, min(len(self._cards) - 1, int((x - self.PAD_X) // max(1, card_w + self.CARD_GAP))))
                    if src_idx != dst_idx:
                        self.reorder_committed.emit(src_idx, dst_idx)
                except Exception:
                    pass
            event.acceptProposedAction()
            return
        event.ignore()


class _CollageTrayScrollArea(QScrollArea):
    """Horizontal scroll area with wheel-to-horizontal pan and edge chevron scroll overlays."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StudioTrayScroll")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

        def _make_chevron_icon(direction: str) -> QIcon:
            pm = QPixmap(32, 32)
            pm.fill(Qt.GlobalColor.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setPen(
                QPen(
                    QColor("#F8FAFC"),
                    2.8,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            if direction == "left":
                p.drawLine(19, 9, 12, 16)
                p.drawLine(12, 16, 19, 23)
            else:
                p.drawLine(13, 9, 20, 16)
                p.drawLine(20, 16, 13, 23)
            p.end()
            pm.setDevicePixelRatio(2.0)
            return QIcon(pm)

        _btn_qss = (
            "QPushButton { background-color: rgba(15, 23, 42, 135); "
            "border: 1px solid rgba(148, 163, 184, 85); border-radius: 9px; padding: 0px; }"
            "QPushButton:hover { background-color: rgba(15, 23, 42, 230); "
            "border: 1px solid rgba(56, 189, 248, 190); }"
        )
        self.btn_scroll_left = QPushButton(self)
        self.btn_scroll_left.setObjectName("FilmstripOverlayScrollBtn")
        self.btn_scroll_left.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_scroll_left.setIcon(_make_chevron_icon("left"))
        self.btn_scroll_left.setIconSize(QSize(16, 16))
        self.btn_scroll_left.setFixedSize(32, 32)
        self.btn_scroll_left.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_scroll_left.setToolTip("Scroll photos left")
        self.btn_scroll_left.setStyleSheet(_btn_qss)
        self.btn_scroll_left.clicked.connect(lambda: self._scroll_by_tiles(-3))

        self.btn_scroll_right = QPushButton(self)
        self.btn_scroll_right.setObjectName("FilmstripOverlayScrollBtn")
        self.btn_scroll_right.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_scroll_right.setIcon(_make_chevron_icon("right"))
        self.btn_scroll_right.setIconSize(QSize(16, 16))
        self.btn_scroll_right.setFixedSize(32, 32)
        self.btn_scroll_right.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_scroll_right.setToolTip("Scroll photos right")
        self.btn_scroll_right.setStyleSheet(_btn_qss)
        self.btn_scroll_right.clicked.connect(lambda: self._scroll_by_tiles(3))

        self.horizontalScrollBar().rangeChanged.connect(lambda _a, _b: self._sync_scroll_buttons())
        self.horizontalScrollBar().valueChanged.connect(lambda _v: self._sync_scroll_buttons())

    def _scroll_by_tiles(self, tiles: int) -> None:
        hbar = self.horizontalScrollBar()
        step = 112 * int(tiles)
        hbar.setValue(hbar.value() + step)

    def _sync_scroll_buttons(self) -> None:
        vp = self.viewport()
        w = vp.width()
        h = vp.height()
        cy = max(4, (h - 32) // 2)
        self.btn_scroll_left.move(6, cy)
        self.btn_scroll_right.move(max(6, w - 38), cy)
        hbar = self.horizontalScrollBar()
        max_v = hbar.maximum()
        val = hbar.value()
        show_overflow = max_v > 2 and w >= 120
        self.btn_scroll_left.setVisible(show_overflow and val > 2)
        self.btn_scroll_right.setVisible(show_overflow and val < max_v - 2)
        if self.btn_scroll_left.isVisible():
            self.btn_scroll_left.raise_()
        if self.btn_scroll_right.isVisible():
            self.btn_scroll_right.raise_()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._sync_scroll_buttons()

    def wheelEvent(self, event: Any) -> None:
        hbar = self.horizontalScrollBar()
        if hbar.maximum() > 0:
            dy = event.angleDelta().y() or event.angleDelta().x()
            if dy:
                hbar.setValue(hbar.value() - int(round(dy * 0.6)))
                event.accept()
                return
        super().wheelEvent(event)


class CollageWorkspace(CreateWorkspaceBase):
    """Full-window interactive Collage & 'Then & Now' workspace with bottom photo tray."""

    workspace_id = "collage"
    display_title = "Collage"
    tool_save_label = "collage"

    def __init__(self, photos_data: list[dict[str, Any]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.photos_data: list[dict[str, Any]] = [dict(p) for p in photos_data]
        self.output_dir = (
            Path(photos_data[0]["file_path"]).parent
            if photos_data and photos_data[0].get("file_path")
            else Path.home()
        )
        self.rendered_bgr: np.ndarray | None = None
        self._is_fit_mode: bool = True
        self._dirty: bool = False
        self._user_shuffled: bool = False
        self._last_preview_size: tuple[int, int] = (1280, 720)
        self._last_render_size: tuple[int, int] = (1280, 720)
        self._canvas_rotation_deg: float = 0.0
        self._selected_grid_slot: int | None = None
        self._grid_selection_rect: QGraphicsPolygonItem | None = None
        self._dragging_polaroid_idx: int | None = None
        self._drag_offset: QPointF = QPointF(0.0, 0.0)
        self._press_pos_vp: QPoint | None = None
        self._pending_grid_slot: int | None = None
        self._panning_crop_idx: int | None = None
        self._panning_crop_slot_size: tuple[int, int] = (100, 100)
        self._panning_crop_start_fc: tuple[float, float] = (0.5, 0.4)
        self._panning_crop_start_scene: QPointF = QPointF(0.0, 0.0)

        self.bg_color = QColor("#FFFFFF")
        self._active_photo_idx: int = 0
        self._active_shape_key: str = "16:9"
        self._tray_cards: list[_DraggableCollageCard] = []
        self._style_btns: list[QPushButton] = []
        self._grid_layout_btns: list[QPushButton] = []
        self._shape_btns: dict[str, QPushButton] = {}
        self._bg_swatch_btns: list[tuple[str, QPushButton]] = []
        self._hq_preview_timer = QTimer(self)
        self._hq_preview_timer.setSingleShot(True)
        self._hq_preview_timer.setInterval(90)
        self._hq_preview_timer.timeout.connect(lambda: self._render_preview(fast=False))

        self._build_canvas_ui()
        self._build_inspector_ui()
        self._build_bottom_tray_ui()

        # Filter out any unreadable photos and cap at 30 so tray, canvas, and grid slots stay 1:1 synced
        valid_photos: list[dict[str, Any]] = []
        for d in self.photos_data:
            if self._ensure_preview_proxy(d) is not None:
                valid_photos.append(d)
                if len(valid_photos) >= 30:
                    break
        if valid_photos:
            self.photos_data = valid_photos
            self._rebuild_grid_layout_combo()

        self._refresh_order_tray()
        self.apply_theme_mode("wlpg_aero")
        self._render_preview()

    # ------------------------------------------------------------------
    # Workspace Contract
    # ------------------------------------------------------------------
    def inspector_widget(self) -> QWidget:
        return self._inspector_root

    def bottom_tray_widget(self) -> QWidget:
        return self._tray_root

    def tray_title(self) -> str:
        return ""

    def ribbon_actions(self) -> list[RibbonActionSpec]:
        return [
            RibbonActionSpec(
                "shuffle",
                "Shuffle",
                "shuffle",
                "Shuffle photo placement order",
                group="Layout",
                large=True,
            ),
            RibbonActionSpec(
                "add_photos",
                "Add\nPhotos",
                "folder_add",
                "Add photos to collage",
                group="Layout",
                large=True,
            ),
            RibbonActionSpec(
                "reset_framing",
                "Reset\nFraming",
                "revert",
                "Reset all manual crop pan and Polaroid positions",
                group="Layout",
                large=True,
            ),
            RibbonActionSpec("zoom_fit", "Fit\nWindow", "fit_window", "Zoom to fit (F)", group="View", large=True),
            RibbonActionSpec("zoom_actual", "100%", "actual_size", "Actual pixels (1)", group="View", large=True),
        ]

    def ribbon_action_states(self) -> dict[str, dict[str, Any]]:
        style_idx = self.cmb_style.currentIndex() if hasattr(self, "cmb_style") else 0
        return {
            "create_ws_save": {"enabled": len(self.photos_data) > 0},
            "shuffle": {"enabled": style_idx != 3 and len(self.photos_data) > 1},
        }

    def handle_ribbon_action(self, action_id: str) -> None:
        if action_id == "shuffle":
            self._shuffle_photos()
        elif action_id == "add_photos":
            self._on_add_photos_clicked()
        elif action_id == "reset_framing":
            self._reset_photo_framing()
            self._reset_polaroid_positions()
            self._reset_canvas_rotation()
        else:
            super().handle_ribbon_action(action_id)

    def has_unsaved_changes(self) -> bool:
        return bool(self._dirty) and self.saved_file_path is None

    def save_result(self) -> Path | None:
        self._save_collage()
        return self.saved_file_path

    def zoom_in(self) -> None:
        self._apply_zoom(1.25)

    def zoom_out(self) -> None:
        self._apply_zoom(1.0 / 1.25)

    def zoom_fit(self) -> None:
        if self.preview_scene.sceneRect().isValid() and self.preview_scene.sceneRect().width() > 0:
            self._is_fit_mode = True
            self.preview_view.fitInView(self.preview_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._on_zoom_changed(self.preview_view.transform().m11())

    def zoom_actual(self) -> None:
        if self.preview_scene.sceneRect().isValid():
            self._is_fit_mode = False
            self.preview_view.resetTransform()
            self._on_zoom_changed(1.0)

    def set_zoom_percent(self, pct: int) -> None:
        self._on_slider_zoom_moved(pct)

    def handle_key_press(self, event: QKeyEvent) -> bool:
        if event.key() == Qt.Key.Key_F:
            self.zoom_fit()
            return True
        if event.key() == Qt.Key.Key_1:
            self.zoom_actual()
            return True
        return False

    def apply_theme_mode(self, theme_name: str) -> None:
        super().apply_theme_mode(theme_name)
        tok = get_theme_tokens(self._theme_name)
        qss = build_studio_panel_qss(self._theme_name)
        self.setStyleSheet(qss)
        self._inspector_root.setStyleSheet(qss)
        self._tray_root.setStyleSheet(qss)
        self.preview_view.apply_theme_mode(self._theme_name)
        self.lbl_info.setStyleSheet(f"color: {tok.text_secondary}; font-size: 9pt; background: transparent;")
        if hasattr(self, "rotation_bar"):
            canvas_bg = "#111318" if tok.is_dark else "#232730"
            self.rotation_bar.setStyleSheet(
                f"QFrame#CollageRotationBar {{ background-color: {canvas_bg}; border: none; }}"
                f"QFrame#CollageRotationBar QSlider::groove:horizontal {{ height: 4px; background: rgba(255,255,255,48); border-radius: 2px; }}"
                f"QFrame#CollageRotationBar QSlider::handle:horizontal {{ width: 12px; height: 12px; margin: -4px 0; "
                f"background: #F8FAFC; border: 1.5px solid {tok.accent}; border-radius: 6px; }}"
            )
            self.lbl_canvas_rotation.setStyleSheet(
                "color: #F8FAFC; font-size: 8.5pt; font-weight: 600; background: transparent;"
            )
            self.btn_reset_rotation.setStyleSheet(
                f"QPushButton {{ background-color: rgba(255,255,255,18); color: #F8FAFC; "
                f"border: 1px solid rgba(255,255,255,55); border-radius: 11px; padding: 2px 12px; font-size: 8.5pt; }}"
                f"QPushButton:hover {{ background-color: rgba(255,255,255,34); border-color: {tok.accent}; }}"
                f"QPushButton:disabled {{ color: rgba(248,250,252,95); border-color: rgba(255,255,255,28); }}"
            )
        self._refresh_visual_picker_icons()
        self._refresh_bg_swatch_styles()
        self._refresh_tray_card_styles()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_canvas_ui(self) -> None:
        self.setObjectName("StudioCanvasRoot")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.preview_view = ZoomablePreviewCanvas(self)
        self.preview_scene = QGraphicsScene(self)
        self.preview_view.setScene(self.preview_scene)
        self.preview_pixmap_item = QGraphicsPixmapItem()
        self.preview_scene.addItem(self.preview_pixmap_item)

        self._grid_selection_rect = QGraphicsPolygonItem()
        self._grid_selection_rect.setPen(QPen(QColor(50, 150, 255), 4))
        self._grid_selection_rect.setBrush(QBrush(QColor(50, 150, 255, 35)))
        self._grid_selection_rect.setZValue(10.0)
        self._grid_selection_rect.setVisible(False)
        self.preview_scene.addItem(self._grid_selection_rect)

        self.preview_view.setMouseTracking(True)
        self.preview_view.viewport().setMouseTracking(True)
        self.preview_view.viewport().installEventFilter(self)
        self.preview_view.zoom_changed.connect(self._on_zoom_changed)
        lay.addWidget(self.preview_view, 1)

        # Google Photos-style Canvas Rotation bar (inside dark canvas zone, above the grey info bar)
        self.rotation_bar = QFrame(self)
        self.rotation_bar.setObjectName("CollageRotationBar")
        rot_lay = QHBoxLayout(self.rotation_bar)
        rot_lay.setContentsMargins(12, 4, 12, 6)
        rot_lay.setSpacing(10)
        rot_lay.addStretch(1)

        self.lbl_canvas_rotation = QLabel("0°", self.rotation_bar)
        self.lbl_canvas_rotation.setFixedWidth(36)
        self.lbl_canvas_rotation.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rot_lay.addWidget(self.lbl_canvas_rotation)

        self.sld_canvas_rotation = AeroStudioSlider(-45, 45, 0, self.rotation_bar)
        self.sld_canvas_rotation.setFixedWidth(240)
        self.sld_canvas_rotation.setToolTip("Rotate collage canvas (-45° to +45°, double-click to reset)")
        self.sld_canvas_rotation.valueChanged.connect(self._on_canvas_rotation_slider_changed)
        rot_lay.addWidget(self.sld_canvas_rotation)

        self.btn_reset_rotation = QPushButton("Reset", self.rotation_bar)
        self.btn_reset_rotation.setFixedHeight(22)
        self.btn_reset_rotation.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset_rotation.setToolTip("Reset canvas rotation to 0°")
        self.btn_reset_rotation.setEnabled(False)
        self.btn_reset_rotation.clicked.connect(self._reset_canvas_rotation)
        rot_lay.addWidget(self.btn_reset_rotation)
        rot_lay.addStretch(1)
        lay.addWidget(self.rotation_bar)

        bottom_bar = QFrame(self)
        bottom_bar.setObjectName("StudioTrayRoot")
        b_lay = QHBoxLayout(bottom_bar)
        b_lay.setContentsMargins(10, 2, 10, 2)
        b_lay.setSpacing(8)

        self.lbl_info = QLabel("Interactive Preview", bottom_bar)
        b_lay.addWidget(self.lbl_info, 1)
        lay.addWidget(bottom_bar)

    def _build_inspector_ui(self) -> None:
        self._inspector_root = QWidget()
        self._inspector_root.setObjectName("StudioInspectorRoot")
        self._inspector_root.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        vbox = QVBoxLayout(self._inspector_root)
        vbox.setContentsMargins(4, 4, 4, 8)
        vbox.setSpacing(8)

        # 1. Style card (visual clickable 2x2 style tiles synced with cmb_style)
        grp_style = StudioCardGroup("Layout Style", self._inspector_root)
        self.cmb_style = QComboBox(grp_style)
        self.cmb_style.addItems([
            "Blended Collage",
            "Modern Grid / Masonry",
            "Polaroid Table Scatter",
            "Then & Now Comparison Card",
        ])
        self.cmb_style.setCurrentIndex(1)  # Default to Grid collage
        self.cmb_style.currentIndexChanged.connect(self._on_style_changed)
        self.cmb_style.setVisible(False)
        grp_style.add_widget(self.cmb_style)

        style_grid = QGridLayout()
        style_grid.setContentsMargins(0, 0, 0, 0)
        style_grid.setSpacing(6)
        self._style_btn_group = QButtonGroup(self)
        self._style_btn_group.setExclusive(True)
        self._style_btns = []
        style_labels = [
            (0, "Blended", "Seamless overlapping auto-collage"),
            (1, "Grid", "Clean bordered grid or masonry layout"),
            (2, "Polaroid", "Draggable Polaroid photo scatter"),
            (3, "Then & Now", "Chronological comparison card"),
        ]
        for idx, lbl, tip in style_labels:
            btn = QPushButton(lbl, grp_style)
            btn.setCheckable(True)
            btn.setChecked(idx == 1)
            btn.setToolTip(tip)
            btn.setIconSize(QSize(20, 20))
            btn.setMinimumHeight(32)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _c=False, i=idx: self.cmb_style.setCurrentIndex(i))
            self._style_btn_group.addButton(btn, idx)
            self._style_btns.append(btn)
            style_grid.addWidget(btn, idx // 2, idx % 2)
        grp_style.add_layout(style_grid)

        self.btn_shuffle = QPushButton("Shuffle Photo Order", grp_style)
        self.btn_shuffle.setIcon(make_wlpg_icon("shuffle", 16))
        self.btn_shuffle.clicked.connect(self._shuffle_photos)
        grp_style.add_widget(self.btn_shuffle)
        vbox.addWidget(grp_style)

        # 2. Canvas Shape & Export Quality card (Google Photos style visual aspect pills)
        grp_res = StudioCardGroup("Canvas Shape & Quality", self._inspector_root)
        shape_grid = QGridLayout()
        shape_grid.setContentsMargins(0, 0, 0, 0)
        shape_grid.setSpacing(5)
        self._shape_btn_group = QButtonGroup(self)
        self._shape_btn_group.setExclusive(True)
        self._shape_btns = {}
        for idx, (s_key, s_label, _rw, _rh, _hi, _std, s_tip) in enumerate(_CANVAS_SHAPES):
            btn = QPushButton(s_label, grp_res)
            btn.setCheckable(True)
            btn.setChecked(s_key == self._active_shape_key)
            btn.setToolTip(s_tip)
            btn.setIconSize(QSize(20, 20))
            btn.setMinimumHeight(28)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet("text-align: left; padding: 3px 7px;")
            btn.clicked.connect(lambda _c=False, k=s_key: self._select_canvas_shape(k))
            self._shape_btn_group.addButton(btn)
            self._shape_btns[s_key] = btn
            shape_grid.addWidget(btn, idx // 2, idx % 2)
        grp_res.add_layout(shape_grid)

        q_row = QHBoxLayout()
        q_row.setSpacing(6)
        lbl_q = QLabel("Quality:", grp_res)
        q_row.addWidget(lbl_q)
        self.cmb_quality = QComboBox(grp_res)
        self.cmb_quality.addItems([
            "High (4K / Print 300 DPI)",
            "Standard (Web / Share)",
        ])
        self.cmb_quality.currentIndexChanged.connect(lambda: self._mark_dirty_and_render())
        q_row.addWidget(self.cmb_quality, 1)
        grp_res.add_layout(q_row)
        vbox.addWidget(grp_res)

        # 3. Framing & Individual Photo Zoom card
        grp_frame = StudioCardGroup("Selected Photo Zoom & Framing", self._inspector_root)
        z_hdr = QHBoxLayout()
        self.lbl_photo_zoom = QLabel("Photo #1 Zoom: 100%", grp_frame)
        z_hdr.addWidget(self.lbl_photo_zoom, 1)
        grp_frame.add_layout(z_hdr)

        z_row = QHBoxLayout()
        z_row.setSpacing(4)
        self.btn_photo_zoom_out = QPushButton("−", grp_frame)
        self.btn_photo_zoom_out.setFixedSize(26, 24)
        self.btn_photo_zoom_out.setToolTip("Zoom selected photo out (-15%)")
        self.btn_photo_zoom_out.setStyleSheet("padding: 0px; font-weight: bold;")
        self.btn_photo_zoom_out.clicked.connect(lambda: self._nudge_active_photo_zoom(-15))
        z_row.addWidget(self.btn_photo_zoom_out)

        self.sld_photo_zoom = AeroStudioSlider(100, 300, 100, grp_frame)
        self.sld_photo_zoom.setToolTip("Zoom the selected photo in/out inside its slot (double-click to reset)")
        self.sld_photo_zoom.valueChanged.connect(self._on_photo_zoom_slider_changed)
        z_row.addWidget(self.sld_photo_zoom, 1)

        self.btn_photo_zoom_in = QPushButton("+", grp_frame)
        self.btn_photo_zoom_in.setFixedSize(26, 24)
        self.btn_photo_zoom_in.setToolTip("Zoom selected photo in (+15%)")
        self.btn_photo_zoom_in.setStyleSheet("padding: 0px; font-weight: bold;")
        self.btn_photo_zoom_in.clicked.connect(lambda: self._nudge_active_photo_zoom(15))
        z_row.addWidget(self.btn_photo_zoom_in)
        grp_frame.add_layout(z_row)

        lbl_hint = QLabel("Click a photo to select • Drag to pan • Shift+Scroll to zoom a photo.", grp_frame)
        lbl_hint.setProperty("uiRole", "muted")
        lbl_hint.setWordWrap(True)
        grp_frame.add_widget(lbl_hint)

        f_row = QHBoxLayout()
        self.btn_frame_mode = QPushButton("Hand Frame", grp_frame)
        self.btn_frame_mode.setCheckable(True)
        self.btn_reset_framing = QPushButton("Reset All", grp_frame)
        self.btn_reset_framing.clicked.connect(self._reset_photo_framing)
        f_row.addWidget(self.btn_frame_mode, 1)
        f_row.addWidget(self.btn_reset_framing, 1)
        grp_frame.add_layout(f_row)
        vbox.addWidget(grp_frame)

        # 4. Seamless Blend card
        self.grp_seamless = StudioCardGroup("Blended Collage Settings", self._inspector_root)
        self.grp_seamless.setVisible(False)
        self.grp_seamless.add_widget(QLabel("Edge Softness / Blending:", self.grp_seamless))
        self.sld_feather = AeroStudioSlider(10, 45, 22, self.grp_seamless)
        self.sld_feather.valueChanged.connect(lambda: self._mark_dirty_and_render())
        self.grp_seamless.add_widget(self.sld_feather)
        vbox.addWidget(self.grp_seamless)

        # 5. Grid Options card (visual grid layout diagrams + border + colour swatches)
        self.grp_grid = StudioCardGroup("Grid Layout & Border", self._inspector_root)
        self.grp_grid.setVisible(True)
        self.cmb_grid_layout = QComboBox(self.grp_grid)
        self.cmb_grid_layout.addItems(recommended_grid_layouts(len(self.photos_data)))
        self.cmb_grid_layout.currentIndexChanged.connect(self._on_grid_layout_changed)
        self.cmb_grid_layout.setVisible(False)
        self.grp_grid.add_widget(self.cmb_grid_layout)

        self._grid_layout_host = QWidget(self.grp_grid)
        self._grid_layout_grid = QGridLayout(self._grid_layout_host)
        self._grid_layout_grid.setContentsMargins(0, 0, 0, 0)
        self._grid_layout_grid.setSpacing(5)
        self._grid_layout_btn_group = QButtonGroup(self)
        self._grid_layout_btn_group.setExclusive(True)
        self.grp_grid.add_widget(self._grid_layout_host)
        self._rebuild_grid_layout_buttons()

        self.grp_grid.add_widget(QLabel("Border Thickness:", self.grp_grid))
        self.sld_border = AeroStudioSlider(0, 40, 12, self.grp_grid)
        self.sld_border.valueChanged.connect(lambda: self._mark_dirty_and_render())
        self.grp_grid.add_widget(self.sld_border)

        self.grp_grid.add_widget(QLabel("Corner Rounding:", self.grp_grid))
        self.sld_corner = AeroStudioSlider(0, 24, 8, self.grp_grid)
        self.sld_corner.valueChanged.connect(lambda: self._mark_dirty_and_render())
        self.grp_grid.add_widget(self.sld_corner)

        self.grp_grid.add_widget(QLabel("Background Colour:", self.grp_grid))
        swatch_row = QHBoxLayout()
        swatch_row.setContentsMargins(0, 2, 0, 2)
        swatch_row.setSpacing(5)
        self._bg_swatch_btns = []
        for s_name, s_hex in _BG_SWATCHES:
            sw = QPushButton("", self.grp_grid)
            sw.setFixedSize(22, 22)
            sw.setCursor(Qt.CursorShape.PointingHandCursor)
            sw.setToolTip(f"{s_name} ({s_hex})")
            sw.clicked.connect(lambda _c=False, hx=s_hex: self._set_bg_color(QColor(hx)))
            self._bg_swatch_btns.append((s_hex, sw))
            swatch_row.addWidget(sw)
        swatch_row.addStretch(1)
        self.grp_grid.add_layout(swatch_row)

        self.btn_bg_color = QPushButton("Custom Colour...", self.grp_grid)
        self.btn_bg_color.setIcon(make_wlpg_icon("eyedropper", 16))
        self.btn_bg_color.clicked.connect(self._pick_bg_color)
        self.grp_grid.add_widget(self.btn_bg_color)
        vbox.addWidget(self.grp_grid)

        # 6. Polaroid card
        self.grp_polaroid = StudioCardGroup("Polaroid Settings", self._inspector_root)
        self.grp_polaroid.setVisible(False)
        self.cmb_texture = QComboBox(self.grp_polaroid)
        self.cmb_texture.addItems(["Mahogany Wood", "Clean Linen", "Modern Slate"])
        self.cmb_texture.currentIndexChanged.connect(lambda: self._mark_dirty_and_render())
        self.grp_polaroid.add_widget(self.cmb_texture)

        cap_row = QHBoxLayout()
        cap_row.setSpacing(6)
        cap_row.addWidget(QLabel("Border Caption:", self.grp_polaroid))
        self.cmb_polaroid_caption = QComboBox(self.grp_polaroid)
        self.cmb_polaroid_caption.addItems([
            "Caption",
            "Date",
            "File Name",
            "Tagged People",
            "None",
        ])
        self.cmb_polaroid_caption.setToolTip(
            "Choose what text is printed on the lower Polaroid border"
        )
        self.cmb_polaroid_caption.currentIndexChanged.connect(lambda: self._mark_dirty_and_render())
        cap_row.addWidget(self.cmb_polaroid_caption, 1)
        self.grp_polaroid.add_layout(cap_row)

        self.btn_reset_polaroids = QPushButton("Reset Polaroid Positions", self.grp_polaroid)
        self.btn_reset_polaroids.clicked.connect(self._reset_polaroid_positions)
        self.grp_polaroid.add_widget(self.btn_reset_polaroids)
        vbox.addWidget(self.grp_polaroid)
        vbox.addStretch(1)

    def _build_bottom_tray_ui(self) -> None:
        self._tray_root = QFrame()
        self._tray_root.setObjectName("StudioTrayRoot")
        lay = QHBoxLayout(self._tray_root)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(6)

        self.tray_scroll = _CollageTrayScrollArea(self._tray_root)
        self.tray_host = _AnimatedCollageFilmstrip(self.tray_scroll)
        self.tray_host.set_scroll_area(self.tray_scroll)
        self.tray_host.reorder_committed.connect(self._on_animated_tray_reorder)
        self.tray_scroll.setWidget(self.tray_host)
        lay.addWidget(self.tray_scroll, 1)

        self.btn_add_photos = QPushButton("Add\nPhotos", self._tray_root)
        self.btn_add_photos.setObjectName("CollageAddPhotoBtn")
        self.btn_add_photos.setIcon(make_wlpg_icon("folder_add", 16))
        self.btn_add_photos.setFixedSize(72, 72)
        self.btn_add_photos.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add_photos.setToolTip("Add more photos to collage...")
        self.btn_add_photos.clicked.connect(self._on_add_photos_clicked)
        lay.addWidget(self.btn_add_photos, 0, Qt.AlignmentFlag.AlignVCenter)

    # ------------------------------------------------------------------
    # Bottom Photo Tray Management
    # ------------------------------------------------------------------
    def _refresh_order_tray(self, selected_slot: int | None = None) -> None:
        self.tray_host.clear_cards()
        self._tray_cards.clear()

        if selected_slot is not None:
            self._selected_grid_slot = selected_slot

        can_remove = len(self.photos_data) > 1
        cards: list[_DraggableCollageCard] = []
        for idx, d in enumerate(self.photos_data):
            img = self._ensure_preview_proxy(d)
            card = _DraggableCollageCard(idx, self.tray_host)
            fp = d.get("file_path") or d.get("filename") or ""
            name = Path(str(fp)).name if fp else f"Photo {idx + 1}"
            card.setToolTip(f"{name}\nClick to select • Drag left or right to reorder")
            card.set_can_remove(can_remove)
            card.set_theme_name(self._theme_name)
            card.card_clicked.connect(self._select_active_photo)
            card.card_dropped.connect(self.move_photo)
            card.remove_clicked.connect(self._remove_photo)

            pm: QPixmap | None = None
            if img is not None:
                ih, iw = img.shape[:2]
                sc = 220.0 / max(1, max(ih, iw))
                if sc < 1.0:
                    small = cv2.resize(
                        img,
                        (max(16, int(round(iw * sc))), max(16, int(round(ih * sc)))),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    small = img
                pm = _cv2_to_qpixmap(small)
            card.set_pixmap(pm)
            cards.append(card)

        self._tray_cards = cards
        self.tray_host.set_cards(cards)

        if self._active_photo_idx >= len(self.photos_data):
            self._active_photo_idx = max(0, len(self.photos_data) - 1)
        self._sync_active_photo_zoom_ui()
        self._refresh_tray_card_styles()

    def _preserve_polaroid_slot_positions(self, reorder_fn: Any) -> None:
        """Keep each Polaroid slot's position fixed when photos are reordered or swapped."""
        saved_positions = [d.get("polaroid_pos") for d in self.photos_data]
        reorder_fn()
        for i, d in enumerate(self.photos_data):
            pos = saved_positions[i] if i < len(saved_positions) else None
            if pos is not None:
                d["polaroid_pos"] = pos
            else:
                d.pop("polaroid_pos", None)

    def _on_animated_tray_reorder(self, src_idx: int, dst_idx: int) -> None:
        """Sync Collage state after an animated pointer drag reorder completes without rebuilding tiles."""
        n = len(self.photos_data)
        if src_idx == dst_idx or not (0 <= src_idx < n and 0 <= dst_idx < n):
            return

        def _do_move() -> None:
            item = self.photos_data.pop(src_idx)
            self.photos_data.insert(dst_idx, item)

        self._preserve_polaroid_slot_positions(_do_move)
        self._user_shuffled = True
        self._dirty = True
        self._active_photo_idx = dst_idx
        self._selected_grid_slot = dst_idx if self.cmb_style.currentIndex() == 1 else None
        self._tray_cards = list(self.tray_host._cards)
        for i, card in enumerate(self._tray_cards):
            fp = self.photos_data[i].get("file_path") or self.photos_data[i].get("filename") or ""
            name = Path(str(fp)).name if fp else f"Photo {i + 1}"
            card.setToolTip(f"{name}\nClick to select • Drag left or right to reorder")
        self._sync_active_photo_zoom_ui()
        self._refresh_tray_card_styles()
        self._update_grid_selection_overlay()
        self._render_preview(fast=False)

    def _on_tray_card_clicked(self, idx: int, _event: Any) -> None:
        self._select_active_photo(idx)

    def _refresh_tray_card_styles(self) -> None:
        tok = get_theme_tokens(self._theme_name)
        active_idx = self._selected_grid_slot if self._selected_grid_slot is not None else self._active_photo_idx
        can_remove = len(self.photos_data) > 1
        self.tray_host.set_theme_name(self._theme_name)
        for idx, card in enumerate(self._tray_cards):
            card.set_theme_name(self._theme_name)
            card.set_selected(idx == active_idx)
            card.set_can_remove(can_remove)
        if hasattr(self, "btn_add_photos"):
            self.btn_add_photos.setStyleSheet(
                f"""
                QFrame#StudioTrayRoot QPushButton#CollageAddPhotoBtn {{
                    background-color: transparent;
                    color: {tok.text_secondary};
                    border: 1.5px dashed {tok.border_subtle};
                    border-radius: 6px;
                    padding: 4px;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    font-weight: 600;
                }}
                QFrame#StudioTrayRoot QPushButton#CollageAddPhotoBtn:hover {{
                    background-color: {tok.bg_hover};
                    color: {tok.text_primary};
                    border-color: {tok.accent};
                }}
                """
            )

    def _on_add_photos_clicked(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Collage Photos",
            str(self.output_dir),
            "Images (*.jpg *.jpeg *.png *.webp *.tif *.tiff)",
        )
        added = False
        for f in files:
            p = Path(f)
            if p.is_file() and len(self.photos_data) < 30:
                img = load_bgr_exif_safe(p)
                if img is not None:
                    self.photos_data.append({"file_path": str(p), "filename": p.name, "image_bgr": img})
                    added = True
        if added:
            self._dirty = True
            self._rebuild_grid_layout_combo()
            self._refresh_order_tray()
            self._render_preview()
            self.ribbon_state_changed.emit()

    def _remove_photo(self, idx: int) -> None:
        if len(self.photos_data) <= 1 or not (0 <= idx < len(self.photos_data)):
            return
        self.photos_data.pop(idx)
        self._dirty = True
        self._selected_grid_slot = None
        self._rebuild_grid_layout_combo()
        self._refresh_order_tray()
        self._render_preview()
        self.ribbon_state_changed.emit()

    def _rebuild_grid_layout_combo(self) -> None:
        cur = self.cmb_grid_layout.currentText()
        self.cmb_grid_layout.blockSignals(True)
        self.cmb_grid_layout.clear()
        items = recommended_grid_layouts(len(self.photos_data))
        self.cmb_grid_layout.addItems(items)
        if cur in items:
            self.cmb_grid_layout.setCurrentText(cur)
        self.cmb_grid_layout.blockSignals(False)
        self._rebuild_grid_layout_buttons()

    def _rebuild_grid_layout_buttons(self) -> None:
        if not hasattr(self, "_grid_layout_grid"):
            return
        while self._grid_layout_grid.count():
            item = self._grid_layout_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                self._grid_layout_btn_group.removeButton(w)
                w.deleteLater()
        self._grid_layout_btns = []
        tok = get_theme_tokens(self._theme_name)
        cur_txt = self.cmb_grid_layout.currentText()
        count = max(1, len(self.photos_data))
        for idx in range(self.cmb_grid_layout.count()):
            lbl = self.cmb_grid_layout.itemText(idx)
            btn = QPushButton(lbl, self._grid_layout_host)
            btn.setCheckable(True)
            btn.setChecked(lbl == cur_txt)
            btn.setIcon(_make_grid_layout_icon(lbl, count, tok.text_primary))
            btn.setIconSize(QSize(26, 18))
            btn.setMinimumHeight(28)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet("text-align: left; padding: 3px 6px;")
            btn.clicked.connect(lambda _c=False, i=idx: self.cmb_grid_layout.setCurrentIndex(i))
            self._grid_layout_btn_group.addButton(btn, idx)
            self._grid_layout_btns.append(btn)
            self._grid_layout_grid.addWidget(btn, idx // 2, idx % 2)

    def _refresh_visual_picker_icons(self) -> None:
        tok = get_theme_tokens(self._theme_name)
        fg = tok.text_primary
        for idx, btn in enumerate(self._style_btns):
            btn.setIcon(_make_style_icon(idx, fg))
        for s_key, _lbl, rw, rh, _hi, _std, _tip in _CANVAS_SHAPES:
            btn = self._shape_btns.get(s_key)
            if btn is not None:
                btn.setIcon(_make_aspect_icon(rw, rh, fg))
        count = max(1, len(self.photos_data))
        for idx, btn in enumerate(self._grid_layout_btns):
            lbl = self.cmb_grid_layout.itemText(idx)
            btn.setIcon(_make_grid_layout_icon(lbl, count, fg))

    def _refresh_bg_swatch_styles(self) -> None:
        tok = get_theme_tokens(self._theme_name)
        cur_hex = self.bg_color.name().lower()
        for s_hex, btn in self._bg_swatch_btns:
            is_sel = s_hex.lower() == cur_hex
            border = f"2.5px solid {tok.accent}" if is_sel else f"1px solid {tok.border_subtle}"
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {s_hex}; border: {border}; border-radius: 11px; padding: 0px; }}"
                f"QPushButton:hover {{ border: 2px solid {tok.accent}; }}"
            )

    def _set_bg_color(self, col: QColor) -> None:
        if not col.isValid():
            return
        self.bg_color = col
        self._dirty = True
        self._refresh_bg_swatch_styles()
        self._render_preview()

    def _select_canvas_shape(self, shape_key: str) -> None:
        self._active_shape_key = shape_key
        for k, btn in self._shape_btns.items():
            btn.blockSignals(True)
            btn.setChecked(k == shape_key)
            btn.blockSignals(False)
        self._dirty = True
        self._render_preview()

    def _select_active_photo(self, idx: int) -> None:
        if not (0 <= idx < len(self.photos_data)):
            return
        self._active_photo_idx = idx
        if self.cmb_style.currentIndex() == 1:
            self._selected_grid_slot = idx
            self._update_grid_selection_overlay()
        self._sync_active_photo_zoom_ui()
        self._refresh_tray_card_styles()

    def _sync_active_photo_zoom_ui(self) -> None:
        if not hasattr(self, "sld_photo_zoom") or not self.photos_data:
            return
        idx = max(0, min(len(self.photos_data) - 1, self._active_photo_idx))
        self._active_photo_idx = idx
        z = float(self.photos_data[idx].get("crop_zoom", 1.0))
        pct = max(100, min(300, int(round(z * 100))))
        self.lbl_photo_zoom.setText(f"Photo #{idx + 1} Zoom: {pct}%")
        self.sld_photo_zoom.blockSignals(True)
        self.sld_photo_zoom.setValue(pct)
        self.sld_photo_zoom.blockSignals(False)

    def _ensure_preview_proxy(self, d: dict[str, Any]) -> np.ndarray | None:
        """Load image once and cache a fast <=900px preview proxy + faces for fluid interactive editing."""
        img = d.get("image_bgr")
        if img is None and d.get("file_path"):
            img = load_bgr_exif_safe(d["file_path"])
            d["image_bgr"] = img
        if img is None:
            return None
        proxy = d.get("_preview_bgr")
        if proxy is None:
            h, w = img.shape[:2]
            max_dim = max(h, w)
            if max_dim > 900:
                sc = 900.0 / float(max_dim)
                proxy = cv2.resize(
                    img,
                    (max(16, int(round(w * sc))), max(16, int(round(h * sc)))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                proxy = img
            d["_preview_bgr"] = proxy
        return proxy

    def move_photo(self, src_idx: int, dst_idx: int) -> None:
        """Reorder a photo by dragging it from src_idx to dst_idx."""
        n = len(self.photos_data)
        if src_idx == dst_idx or not (0 <= src_idx < n and 0 <= dst_idx < n):
            return

        def _do_move() -> None:
            item = self.photos_data.pop(src_idx)
            self.photos_data.insert(dst_idx, item)

        self._preserve_polaroid_slot_positions(_do_move)
        self._user_shuffled = True
        self._dirty = True
        self._active_photo_idx = dst_idx
        self._selected_grid_slot = dst_idx if self.cmb_style.currentIndex() == 1 else None
        self._refresh_order_tray(selected_slot=self._selected_grid_slot)
        self._render_preview(fast=False)

    def _set_photo_zoom(self, idx: int, pct: int, *, interactive: bool = False) -> None:
        if not (0 <= idx < len(self.photos_data)):
            return
        clamped = max(100, min(300, int(pct)))
        self._active_photo_idx = idx
        self.photos_data[idx]["crop_zoom"] = clamped / 100.0
        self._sync_active_photo_zoom_ui()
        self._dirty = True
        if interactive and self.cmb_style.currentIndex() == 1 and self._fast_update_grid_slot(idx):
            return
        self._render_preview(fast=interactive)

    def _on_photo_zoom_slider_changed(self, pct: int) -> None:
        self._set_photo_zoom(self._active_photo_idx, pct, interactive=True)

    def _nudge_active_photo_zoom(self, delta_pct: int) -> None:
        if not self.photos_data:
            return
        cur_z = float(self.photos_data[self._active_photo_idx].get("crop_zoom", 1.0))
        self._set_photo_zoom(self._active_photo_idx, int(round(cur_z * 100)) + delta_pct, interactive=False)

    # ------------------------------------------------------------------
    # Zoom & Settings Handlers
    # ------------------------------------------------------------------
    def _apply_zoom(self, factor: float) -> None:
        self._is_fit_mode = False
        self.preview_view.scale(factor, factor)
        self._on_zoom_changed(self.preview_view.transform().m11())

    def _on_slider_zoom_moved(self, pct: int) -> None:
        self._is_fit_mode = False
        target_scale = max(0.1, min(5.0, pct / 100.0))
        current_scale = self.preview_view.transform().m11()
        if abs(current_scale) > 1e-4:
            self.preview_view.scale(target_scale / current_scale, target_scale / current_scale)
        self._on_zoom_changed(target_scale)

    def _on_zoom_changed(self, scale_val: float) -> None:
        self.zoom_state_changed.emit(float(scale_val), bool(self._is_fit_mode))

    def _mark_dirty_and_render(self) -> None:
        self._dirty = True
        self._render_preview()

    def _on_style_changed(self, idx: int) -> None:
        self._dirty = True
        for i, btn in enumerate(self._style_btns):
            btn.blockSignals(True)
            btn.setChecked(i == idx)
            btn.blockSignals(False)
        self.grp_seamless.setVisible(idx == 0)
        self.grp_grid.setVisible(idx == 1)
        self.grp_polaroid.setVisible(idx == 2)
        self.btn_shuffle.setVisible(idx != 3)
        self._selected_grid_slot = None
        self._dragging_polaroid_idx = None
        self._panning_crop_idx = None
        self._pending_grid_slot = None
        self.preview_view.unsetCursor()
        self._refresh_tray_card_styles()
        self._render_preview()
        self.ribbon_state_changed.emit()

    def _on_grid_layout_changed(self) -> None:
        self._dirty = True
        cur_idx = self.cmb_grid_layout.currentIndex()
        for i, btn in enumerate(self._grid_layout_btns):
            btn.blockSignals(True)
            btn.setChecked(i == cur_idx)
            btn.blockSignals(False)
        self._selected_grid_slot = None
        self._refresh_tray_card_styles()
        self._render_preview()

    def swap_photos(self, idx_a: int, idx_b: int) -> None:
        if idx_a == idx_b:
            return
        if not (0 <= idx_a < len(self.photos_data) and 0 <= idx_b < len(self.photos_data)):
            return

        def _do_swap() -> None:
            self.photos_data[idx_a], self.photos_data[idx_b] = self.photos_data[idx_b], self.photos_data[idx_a]

        self._preserve_polaroid_slot_positions(_do_swap)
        self._user_shuffled = True
        self._dirty = True
        self._selected_grid_slot = None
        self._refresh_order_tray(selected_slot=None)
        self._update_grid_selection_overlay()
        self._refresh_tray_card_styles()
        self._render_preview()

    def _set_canvas_rotation(self, deg: float, *, interactive: bool = False) -> None:
        val = max(-45.0, min(45.0, float(deg)))
        if abs(val) < 0.01:
            val = 0.0
        self._canvas_rotation_deg = val
        iv = int(round(val))
        if hasattr(self, "sld_canvas_rotation"):
            self.sld_canvas_rotation.blockSignals(True)
            self.sld_canvas_rotation.setValue(max(-45, min(45, iv)))
            self.sld_canvas_rotation.blockSignals(False)
        if hasattr(self, "lbl_canvas_rotation"):
            self.lbl_canvas_rotation.setText(f"{iv}°")
        if hasattr(self, "btn_reset_rotation"):
            self.btn_reset_rotation.setEnabled(iv != 0)
        self._dirty = True
        self._render_preview(fast=interactive)

    def _on_canvas_rotation_slider_changed(self, deg: int) -> None:
        self._set_canvas_rotation(float(deg), interactive=True)

    def _reset_canvas_rotation(self) -> None:
        if abs(self._canvas_rotation_deg) > 1e-4:
            self._set_canvas_rotation(0.0, interactive=False)

    def _shuffle_photos(self) -> None:
        if len(self.photos_data) <= 1:
            return
        random.shuffle(self.photos_data)
        self._user_shuffled = True
        self._dirty = True
        self._selected_grid_slot = None
        self._refresh_order_tray()
        self._render_preview()

    def _reset_photo_framing(self) -> None:
        for d in self.photos_data:
            d.pop("focus_center", None)
            d.pop("crop_zoom", None)
        self._sync_active_photo_zoom_ui()
        self._dirty = True
        self._render_preview()

    def _reset_polaroid_positions(self) -> None:
        for d in self.photos_data:
            d.pop("polaroid_pos", None)
        self._dirty = True
        self._render_preview()

    def _pick_bg_color(self) -> None:
        col = QColorDialog.getColor(self.bg_color, self, "Select Background Colour")
        if col.isValid():
            self._set_bg_color(col)

    def _get_target_resolution(self) -> tuple[int, int]:
        use_std = hasattr(self, "cmb_quality") and self.cmb_quality.currentIndex() == 1
        for s_key, _lbl, _rw, _rh, hi_px, std_px, _tip in _CANVAS_SHAPES:
            if s_key == self._active_shape_key:
                return std_px if use_std else hi_px
        return (1920, 1080) if use_std else (3840, 2160)

    # ------------------------------------------------------------------
    # Rendering & Interactive Hit-Testing
    # ------------------------------------------------------------------
    def _format_polaroid_caption(self, d: dict[str, Any]) -> str:
        """Build the Polaroid border caption according to the user's selected mode."""
        mode = (
            self.cmb_polaroid_caption.currentText().strip().lower()
            if hasattr(self, "cmb_polaroid_caption")
            else "caption"
        )
        if mode == "none":
            return ""
        if mode == "caption":
            return str(d.get("caption") or "").strip()
        if mode == "file name":
            fp = d.get("filename") or d.get("file_path") or ""
            return Path(str(fp)).name if fp else ""
        if mode == "date":
            raw_dt = d.get("date_taken") or d.get("date_str") or d.get("mtime")
            if isinstance(raw_dt, (int, float)) and raw_dt > 100000000:
                try:
                    return datetime.datetime.fromtimestamp(raw_dt).strftime("%d %b %Y")
                except Exception:
                    return str(raw_dt)
            if raw_dt:
                return str(raw_dt).strip()
            return ""
        if mode == "tagged people":
            names: list[str] = []
            seen: set[str] = set()
            src_list = d.get("confirmed_people") or d.get("people") or d.get("face_regions") or []
            for entry in src_list:
                if isinstance(entry, str):
                    nm = entry.strip()
                elif isinstance(entry, dict):
                    nm = str(entry.get("person_name") or entry.get("name") or "").strip()
                else:
                    nm = ""
                if nm and nm.lower() not in seen:
                    seen.add(nm.lower())
                    names.append(nm)
            return " + ".join(names)
        return str(d.get("caption") or "").strip()

    def _scene_to_unrotated(self, scene_pos: QPointF) -> QPointF:
        """Map a point on the displayed preview canvas into unrotated render coordinates."""
        pw, ph = self._last_preview_size
        rw, rh = self._last_render_size
        dx = scene_pos.x() - (pw / 2.0)
        dy = scene_pos.y() - (ph / 2.0)
        ang = self._canvas_rotation_deg % 360.0
        if abs(ang) > 1e-4 and abs(ang - 360.0) > 1e-4:
            rad = math.radians(-ang)
            ca, sa = math.cos(rad), math.sin(rad)
            dx, dy = (dx * ca - dy * sa), (dx * sa + dy * ca)
        return QPointF(dx + (rw / 2.0), dy + (rh / 2.0))

    def _unrotated_to_scene(self, ux: float, uy: float) -> QPointF:
        """Map a point from unrotated render coordinates onto the displayed preview canvas."""
        pw, ph = self._last_preview_size
        rw, rh = self._last_render_size
        dx = ux - (rw / 2.0)
        dy = uy - (rh / 2.0)
        ang = self._canvas_rotation_deg % 360.0
        if abs(ang) > 1e-4 and abs(ang - 360.0) > 1e-4:
            rad = math.radians(ang)
            ca, sa = math.cos(rad), math.sin(rad)
            dx, dy = (dx * ca - dy * sa), (dx * sa + dy * ca)
        return QPointF(dx + (pw / 2.0), dy + (ph / 2.0))

    def _render_canvas(self, target_size: tuple[int, int], *, use_proxy: bool = False) -> np.ndarray:
        style_idx = self.cmb_style.currentIndex()
        images_bgr: list[np.ndarray] = []
        valid_entries: list[dict[str, Any]] = []
        for d in self.photos_data:
            img = self._ensure_preview_proxy(d) if use_proxy else d.get("image_bgr")
            if img is None and d.get("file_path"):
                img = load_bgr_exif_safe(d["file_path"])
                d["image_bgr"] = img
                if use_proxy:
                    img = self._ensure_preview_proxy(d)
            if img is not None:
                images_bgr.append(img)
                valid_entries.append(d)

        if not images_bgr:
            return np.full((target_size[1], target_size[0], 3), (245, 245, 248), dtype=np.uint8)

        render_size = compute_canvas_rotation_render_size(target_size, self._canvas_rotation_deg)
        focus_centers = [d.get("focus_center") for d in valid_entries]
        zooms = [float(d.get("crop_zoom", 1.0)) for d in valid_entries]

        if style_idx == 0:
            feather_ratio = self.sld_feather.value() / 100.0
            faces_per_image: list[list[tuple[int, int, int, int]]] = []
            for d, im in zip(valid_entries, images_bgr):
                fk = "_proxy_faces" if use_proxy else "_full_faces"
                if fk not in d:
                    d[fk] = d.get("faces") if d.get("faces") is not None else detect_faces_in_image(im)
                faces_per_image.append(d[fk])
            raw_bgr = create_blended_seamless_collage(
                images_bgr=images_bgr,
                target_size=render_size,
                blend_feather=feather_ratio,
                faces_per_image=faces_per_image,
                prioritize_faces=not self._user_shuffled,
                focus_centers=focus_centers,
                zooms=zooms,
            )
        elif style_idx == 1:
            layout_name = self.cmb_grid_layout.currentText().lower().replace(" ", "_")
            # Scale border/corner strictly by target resolution vs 1280px preview reference
            # (NEVER by rotation expansion factor, since rotation center-crops at 1:1 scale)
            px_scale = max(1, max(target_size)) / 1280.0
            raw_bgr = create_grid_collage(
                images_bgr=images_bgr,
                layout=layout_name,
                border_thickness=max(0, int(round(self.sld_border.value() * px_scale))),
                corner_radius=max(0, int(round(self.sld_corner.value() * px_scale))),
                bg_color=_qcolor_to_bgr(self.bg_color),
                target_size=render_size,
                focus_centers=focus_centers,
                zooms=zooms,
            )
        elif style_idx == 2:
            tex_txt = self.cmb_texture.currentText()
            tex_name = "dark_wood" if "Wood" in tex_txt else ("linen" if "Linen" in tex_txt else "slate")
            cap_mode = (
                self.cmb_polaroid_caption.currentText().strip().lower()
                if hasattr(self, "cmb_polaroid_caption")
                else "caption"
            )
            polaroid_data = []
            for i, d in enumerate(self.photos_data):
                if i < len(images_bgr):
                    cap_text = self._format_polaroid_caption(d)
                    p_entry: dict[str, Any] = {
                        "image_bgr": images_bgr[i],
                        "caption": cap_text,
                        "caption_override": cap_text,
                        "crop_zoom": float(d.get("crop_zoom", 1.0)),
                    }
                    if d.get("polaroid_pos") is not None:
                        p_entry["center_norm"] = d["polaroid_pos"]
                    if d.get("focus_center") is not None:
                        p_entry["focus_center"] = d["focus_center"]
                    polaroid_data.append(p_entry)
            raw_bgr = create_polaroid_scatter_collage(
                images_data=polaroid_data,
                target_size=render_size,
                bg_style=tex_name,
                print_captions=(cap_mode != "none"),
            )
        else:
            then_now_data = [
                {
                    "image_bgr": images_bgr[i],
                    "date_taken": d.get("date_taken") or d.get("date_str") or "",
                    "caption": d.get("caption") or "",
                    "mtime": d.get("mtime") or 0,
                    "focus_center": d.get("focus_center"),
                    "crop_zoom": float(d.get("crop_zoom", 1.0)),
                }
                for i, d in enumerate(self.photos_data)
                if i < len(images_bgr)
            ]
            raw_bgr = create_then_and_now_card(images_data=then_now_data, target_size=render_size)

        return rotate_collage_canvas(raw_bgr, target_size, self._canvas_rotation_deg)

    def _fast_update_grid_slot(self, idx: int) -> bool:
        """Ultra-fast single-slot partial blit during interactive Grid zoom/pan (<2ms)."""
        if self.rendered_bgr is None or self.cmb_style.currentIndex() != 1:
            return False
        if abs(self._canvas_rotation_deg % 360.0) > 1e-4:
            return False
        slots = self._current_grid_slots()
        if not (0 <= idx < len(slots) and idx < len(self.photos_data)):
            return False
        img = self._ensure_preview_proxy(self.photos_data[idx])
        if img is None:
            return False
        sx, sy, sw, sh = slots[idx]
        ch, cw = self.rendered_bgr.shape[:2]
        if sw <= 0 or sh <= 0 or sy + sh > ch or sx + sw > cw:
            return False
        fc = self.photos_data[idx].get("focus_center")
        z = float(self.photos_data[idx].get("crop_zoom", 1.0))
        cropped = _crop_to_aspect(img, sw, sh, focus_center=fc, zoom=z)
        cr = self.sld_corner.value()
        if cr > 0:
            bg_bgr = _qcolor_to_bgr(self.bg_color)
            self.rendered_bgr[sy : sy + sh, sx : sx + sw] = bg_bgr
            rounded_bgra = _apply_rounded_corners(cropped, cr)
            alpha = rounded_bgra[:, :, 3].astype(np.float32) / 255.0
            rgb = rounded_bgra[:, :, :3].astype(np.float32)
            roi = self.rendered_bgr[sy : sy + sh, sx : sx + sw].astype(np.float32)
            self.rendered_bgr[sy : sy + sh, sx : sx + sw] = (
                rgb * alpha[:, :, None] + roi * (1.0 - alpha[:, :, None])
            ).astype(np.uint8)
        else:
            self.rendered_bgr[sy : sy + sh, sx : sx + sw] = cropped
        self.preview_pixmap_item.setPixmap(_cv2_to_qpixmap(self.rendered_bgr))
        return True

    def _render_preview(self, *, fast: bool = False) -> None:
        full_w, full_h = self._get_target_resolution()
        # Keep scene coordinate size fixed at 1280px max-edge so fast previews never
        # alter sceneRect or jump the user's canvas zoom during interactive photo edits.
        base_scale = min(1.0, 1280.0 / max(full_w, full_h))
        scene_w = max(16, int(round(full_w * base_scale)))
        scene_h = max(16, int(round(full_h * base_scale)))
        self._last_preview_size = (scene_w, scene_h)
        self._last_render_size = compute_canvas_rotation_render_size(
            (scene_w, scene_h), self._canvas_rotation_deg
        )

        if fast:
            fast_scale = min(1.0, 640.0 / max(full_w, full_h))
            pw = max(16, int(round(full_w * fast_scale)))
            ph = max(16, int(round(full_h * fast_scale)))
            rendered_small = self._render_canvas(target_size=(pw, ph), use_proxy=True)
            rendered = (
                cv2.resize(rendered_small, (scene_w, scene_h), interpolation=cv2.INTER_LINEAR)
                if (pw, ph) != (scene_w, scene_h)
                else rendered_small
            )
            self._hq_preview_timer.start()
        else:
            rendered = self._render_canvas(target_size=(scene_w, scene_h), use_proxy=True)
            self._hq_preview_timer.stop()

        self.rendered_bgr = rendered
        pm = _cv2_to_qpixmap(rendered)
        prev_rect = self.preview_scene.sceneRect()
        new_rect = QRectF(pm.rect())
        size_changed = (not prev_rect.isValid()) or (prev_rect.size() != new_rect.size())
        self.preview_pixmap_item.setPixmap(pm)
        if size_changed:
            self.preview_scene.setSceneRect(new_rect)
        self._update_grid_selection_overlay()
        if self._is_fit_mode and size_changed:
            self.zoom_fit()

        style_idx = self.cmb_style.currentIndex()
        hint = " • Click 2 tiles to swap" if style_idx == 1 else (" • Drag Polaroids to move" if style_idx == 2 else "")
        rot_str = f" • {int(round(self._canvas_rotation_deg))}°" if abs(self._canvas_rotation_deg) > 0.4 else ""
        self.lbl_info.setText(
            f"{scene_w} × {scene_h} px (Export: {full_w} × {full_h}){rot_str} • {len(self.photos_data)} photo(s){hint}"
        )

    def _current_grid_slots(self) -> list[tuple[int, int, int, int]]:
        layout_name = self.cmb_grid_layout.currentText().lower().replace(" ", "_")
        px_scale = max(1, max(self._last_preview_size)) / 1280.0
        return compute_grid_slots(
            count=len(self.photos_data),
            layout=layout_name,
            border_thickness=max(0, int(round(self.sld_border.value() * px_scale))),
            target_size=self._last_render_size,
        )

    def _update_grid_selection_overlay(self) -> None:
        if self._grid_selection_rect is None:
            return
        if self.cmb_style.currentIndex() != 1 or self._selected_grid_slot is None:
            self._grid_selection_rect.setVisible(False)
            return
        slots = self._current_grid_slots()
        if 0 <= self._selected_grid_slot < len(slots):
            sx, sy, sw, sh = slots[self._selected_grid_slot]
            poly = QPolygonF([
                self._unrotated_to_scene(sx, sy),
                self._unrotated_to_scene(sx + sw, sy),
                self._unrotated_to_scene(sx + sw, sy + sh),
                self._unrotated_to_scene(sx, sy + sh),
            ])
            self._grid_selection_rect.setPolygon(poly)
            self._grid_selection_rect.setVisible(True)
        else:
            self._grid_selection_rect.setVisible(False)

    def _hit_test_grid_slot(self, scene_pos: QPointF) -> int | None:
        slots = self._current_grid_slots()
        u_pos = self._scene_to_unrotated(scene_pos)
        px, py = u_pos.x(), u_pos.y()
        for idx, (sx, sy, sw, sh) in enumerate(slots):
            if idx < len(self.photos_data) and sx <= px <= sx + sw and sy <= py <= sy + sh:
                return idx
        return None

    def _polaroid_center_px(self, idx: int, defaults: list[tuple[float, float]]) -> tuple[float, float]:
        rw, rh = self._last_render_size
        custom_pos = self.photos_data[idx].get("polaroid_pos")
        if custom_pos is not None and len(custom_pos) == 2:
            nx, ny = float(custom_pos[0]), float(custom_pos[1])
        else:
            nx, ny = defaults[idx % len(defaults)]
        return rw * nx, rh * ny

    def _hit_test_polaroid(self, scene_pos: QPointF) -> tuple[int, tuple[float, float]] | None:
        count = len(self.photos_data)
        if count == 0:
            return None
        card_w, card_h = compute_polaroid_card_size(self._last_render_size)
        defaults = default_polaroid_positions_norm(count)
        u_pos = self._scene_to_unrotated(scene_pos)
        px, py = u_pos.x(), u_pos.y()
        for idx in range(count - 1, -1, -1):
            cx, cy = self._polaroid_center_px(idx, defaults)
            if abs(px - cx) <= card_w * 0.52 and abs(py - cy) <= card_h * 0.52:
                return idx, (cx, cy)
        return None

    def _is_photo_cropped_for_slot(self, idx: int, slot_w: int, slot_h: int) -> bool:
        if not (0 <= idx < len(self.photos_data)) or slot_w <= 0 or slot_h <= 0:
            return False
        if float(self.photos_data[idx].get("crop_zoom", 1.0)) > 1.01:
            return True
        img = self.photos_data[idx].get("image_bgr")
        if img is None:
            return False
        ih, iw = img.shape[:2]
        if ih <= 0 or iw <= 0:
            return False
        return abs((iw / ih) - (slot_w / slot_h)) > 0.015

    def _hit_test_any_slot(self, scene_pos: QPointF) -> int | None:
        """Hit-test which photo index is under scene_pos regardless of whether it is cropped."""
        style_idx = self.cmb_style.currentIndex()
        rw, rh = self._last_render_size
        u_pos = self._scene_to_unrotated(scene_pos)
        px, py = u_pos.x(), u_pos.y()
        if style_idx == 0:
            n = len(self.photos_data)
            if n == 0:
                return None
            slots = _get_seamless_slots(n, rh > rw)
            for idx in range(min(n, len(slots)) - 1, -1, -1):
                cx_n, cy_n, sw_n, sh_n = slots[idx]
                sw = max(64, int(rw * sw_n))
                sh = max(64, int(rh * sh_n))
                sx = rw * cx_n - sw / 2.0
                sy = rh * cy_n - sh / 2.0
                if sx <= px <= sx + sw and sy <= py <= sy + sh:
                    return idx
            return None
        if style_idx == 1:
            return self._hit_test_grid_slot(scene_pos)
        if style_idx == 2:
            hit_pol = self._hit_test_polaroid(scene_pos)
            return hit_pol[0] if hit_pol is not None else None
        count = min(4, len(self.photos_data))
        if count > 0 and rw > 0:
            idx = int(px / max(1.0, rw / float(count)))
            return max(0, min(count - 1, idx))
        return None

    def _hit_test_cropped_slot(self, scene_pos: QPointF) -> tuple[int, int, int] | None:
        style_idx = self.cmb_style.currentIndex()
        rw, rh = self._last_render_size
        u_pos = self._scene_to_unrotated(scene_pos)
        px, py = u_pos.x(), u_pos.y()

        if style_idx == 0:
            n = len(self.photos_data)
            if n == 0:
                return None
            slots = _get_seamless_slots(n, rh > rw)
            for idx in range(min(n, len(slots)) - 1, -1, -1):
                cx_n, cy_n, sw_n, sh_n = slots[idx]
                sw = max(64, int(rw * sw_n))
                sh = max(64, int(rh * sh_n))
                sx = rw * cx_n - sw / 2.0
                sy = rh * cy_n - sh / 2.0
                if sx <= px <= sx + sw and sy <= py <= sy + sh and self._is_photo_cropped_for_slot(idx, sw, sh):
                    return idx, sw, sh
            return None

        if style_idx == 1:
            hit = self._hit_test_grid_slot(scene_pos)
            if hit is not None:
                slots = self._current_grid_slots()
                _, _, sw, sh = slots[hit]
                if self._is_photo_cropped_for_slot(hit, sw, sh):
                    return hit, sw, sh
            return None

        if style_idx == 2:
            hit_pol = self._hit_test_polaroid(scene_pos)
            if hit_pol is not None:
                idx, _ = hit_pol
                _cw, _ch, _pad, photo_w, photo_h, _bp = compute_polaroid_metrics(self._last_render_size)
                if self._is_photo_cropped_for_slot(idx, photo_w, photo_h):
                    return idx, photo_w, photo_h
            return None
        return None

    def _start_crop_pan(self, idx: int, sw: int, sh: int, scene_pos: QPointF) -> None:
        self._panning_crop_idx = idx
        self._panning_crop_slot_size = (sw, sh)
        fc = self.photos_data[idx].get("focus_center")
        self._panning_crop_start_fc = (float(fc[0]), float(fc[1])) if fc else (0.5, 0.40)
        self._panning_crop_start_scene = self._scene_to_unrotated(scene_pos)
        self._active_photo_idx = idx
        self._selected_grid_slot = idx
        self._sync_active_photo_zoom_ui()
        self._refresh_tray_card_styles()
        self.preview_view.setCursor(Qt.CursorShape.ClosedHandCursor)

    def eventFilter(self, obj: Any, event: QEvent) -> bool:
        if obj is self.preview_view.viewport():
            if event.type() == QEvent.Type.Resize:
                if self._is_fit_mode and self.preview_scene.sceneRect().isValid() and self.preview_scene.sceneRect().width() > 0:
                    self.preview_view.fitInView(self.preview_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
                    self._on_zoom_changed(self.preview_view.transform().m11())
            if event.type() == QEvent.Type.Wheel:
                mods = event.modifiers()
                per_photo_zoom = (
                    bool(mods & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.AltModifier))
                    or self.btn_frame_mode.isChecked()
                )
                if per_photo_zoom:
                    scene_pos = self.preview_view.mapToScene(event.position().toPoint())
                    hit_idx = self._hit_test_any_slot(scene_pos)
                    raw_delta = event.angleDelta().y() or event.angleDelta().x()
                    if hit_idx is not None and raw_delta != 0:
                        delta = 10 if raw_delta > 0 else -10
                        cur_z = float(self.photos_data[hit_idx].get("crop_zoom", 1.0))
                        self._set_photo_zoom(hit_idx, int(round(cur_z * 100)) + delta, interactive=True)
                        self._refresh_tray_card_styles()
                    # Always consume so per-photo zoom never leaks to canvas zoom
                    event.accept()
                    return True
            if not getattr(self.preview_view, "_space_panning", False):
                etype = event.type()
                if etype not in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease):
                    return super().eventFilter(obj, event)
                style_idx = self.cmb_style.currentIndex()
                frame_mode = self.btn_frame_mode.isChecked()
                alt_held = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)

                if etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                    scene_pos = self.preview_view.mapToScene(event.pos())
                    self._press_pos_vp = event.pos()
                    if style_idx == 0:
                        crop_hit = self._hit_test_cropped_slot(scene_pos)
                        if crop_hit is not None:
                            self._start_crop_pan(*crop_hit, scene_pos)
                            return True
                    elif style_idx == 1:
                        hit = self._hit_test_grid_slot(scene_pos)
                        if hit is not None:
                            if frame_mode or alt_held:
                                crop_hit = self._hit_test_cropped_slot(scene_pos)
                                if crop_hit is not None:
                                    self._start_crop_pan(*crop_hit, scene_pos)
                                    return True
                            self._pending_grid_slot = hit
                            return True
                    elif style_idx == 2:
                        if frame_mode or alt_held:
                            crop_hit = self._hit_test_cropped_slot(scene_pos)
                            if crop_hit is not None:
                                self._start_crop_pan(*crop_hit, scene_pos)
                                return True
                        hit = self._hit_test_polaroid(scene_pos)
                        if hit is not None:
                            idx, (cx, cy) = hit
                            u_pos = self._scene_to_unrotated(scene_pos)
                            self._dragging_polaroid_idx = idx
                            self._drag_offset = QPointF(cx - u_pos.x(), cy - u_pos.y())
                            self._selected_grid_slot = idx
                            self._refresh_tray_card_styles()
                            self.preview_view.setCursor(Qt.CursorShape.ClosedHandCursor)
                            return True

                elif etype == QEvent.Type.MouseMove:
                    scene_pos = self.preview_view.mapToScene(event.pos())
                    if self._pending_grid_slot is not None and self._press_pos_vp is not None:
                        if (event.pos() - self._press_pos_vp).manhattanLength() > 5:
                            crop_hit = self._hit_test_cropped_slot(self.preview_view.mapToScene(self._press_pos_vp))
                            self._pending_grid_slot = None
                            if crop_hit is not None:
                                self._start_crop_pan(*crop_hit, scene_pos)
                                return True

                    if self._panning_crop_idx is not None:
                        sw, sh = self._panning_crop_slot_size
                        u_pos = self._scene_to_unrotated(scene_pos)
                        dx = u_pos.x() - self._panning_crop_start_scene.x()
                        dy = u_pos.y() - self._panning_crop_start_scene.y()
                        sx0, sy0 = self._panning_crop_start_fc
                        self.photos_data[self._panning_crop_idx]["focus_center"] = (
                            max(0.0, min(1.0, sx0 - (dx / max(1, sw)))),
                            max(0.0, min(1.0, sy0 - (dy / max(1, sh)))),
                        )
                        self._dirty = True
                        if not (style_idx == 1 and self._fast_update_grid_slot(self._panning_crop_idx)):
                            self._render_preview(fast=True)
                        return True

                    if self._dragging_polaroid_idx is not None and style_idx == 2:
                        rw, rh = self._last_render_size
                        u_pos = self._scene_to_unrotated(scene_pos)
                        self.photos_data[self._dragging_polaroid_idx]["polaroid_pos"] = (
                            max(0.05, min(0.95, (u_pos.x() + self._drag_offset.x()) / max(1, rw))),
                            max(0.05, min(0.95, (u_pos.y() + self._drag_offset.y()) / max(1, rh))),
                        )
                        self._dirty = True
                        self._render_preview(fast=True)
                        return True

                    show_hand = style_idx in (0, 1) or (style_idx == 2 and (frame_mode or alt_held))
                    if show_hand and self._hit_test_cropped_slot(scene_pos) is not None:
                        self.preview_view.setCursor(Qt.CursorShape.OpenHandCursor)
                    else:
                        self.preview_view.unsetCursor()

                elif etype == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                    if self._panning_crop_idx is not None:
                        self._panning_crop_idx = None
                        self._press_pos_vp = None
                        self.preview_view.unsetCursor()
                        return True
                    if self._pending_grid_slot is not None and style_idx == 1:
                        hit = self._pending_grid_slot
                        self._pending_grid_slot = None
                        self._press_pos_vp = None
                        if self._selected_grid_slot is None:
                            self._selected_grid_slot = hit
                            self._active_photo_idx = hit
                            self._sync_active_photo_zoom_ui()
                            self._update_grid_selection_overlay()
                            self._refresh_tray_card_styles()
                        elif self._selected_grid_slot == hit:
                            self._selected_grid_slot = None
                            self._update_grid_selection_overlay()
                            self._refresh_tray_card_styles()
                        else:
                            self.swap_photos(self._selected_grid_slot, hit)
                        return True
                    if self._dragging_polaroid_idx is not None:
                        self._dragging_polaroid_idx = None
                        self._press_pos_vp = None
                        self.preview_view.unsetCursor()
                        return True
        return super().eventFilter(obj, event)

    def _save_collage(self) -> None:
        first_path = self.photos_data[0].get("file_path") if self.photos_data else None
        if first_path is not None and not isinstance(first_path, Path):
            first_path = Path(first_path)
        out_path = prompt_save_new_file(self, first_path, "collage", ".jpg")
        if not out_path:
            return
        self.lbl_info.setText("Rendering full-resolution collage for export...")
        QApplication.processEvents()
        full_res_bgr = self._render_canvas(target_size=self._get_target_resolution())
        try:
            cv2.imwrite(str(out_path), full_res_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            self.saved_file_path = out_path
            self._dirty = False
            self.saved.emit(out_path)
        except Exception as e:
            QMessageBox.critical(self, "Save Collage Failed", f"Could not save collage file:\n{e}")
