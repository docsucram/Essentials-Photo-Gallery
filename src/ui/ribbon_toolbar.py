"""
RibbonToolbar (QWidget) implementing Windows Live Photo Gallery's 5-Tab Ribbon & Search Scope Header:
- Tabs: Home, Edit, Find, Create, View (with painted WLPG Aero vector icons)
- Home Tab: New (Add folder), Manage (Rotate left/right, Delete), Organize (People tag,
  Descriptive tag, Adjust time, Flag, World Map), Quick find (Horizontal Face Avatar Carousel),
  Share (Slide show)
- View Tab: Panes (Navigation pane, Tag and caption pane), Arrange list (Date, Geotag,
  Person, Name, Tag, Published, Folder, Media type, Rating + Reverse sort),
  Show details (Rating, Caption, Flags, File name, File size, Date taken, Image size),
  Layout Mode (WLPG Classic Grid vs Justified Tessellated Mosaic), Zoom slider
- Search by Scope Bar: Sits above the grid ('Search by: Folder: Pictures | Search all' + instant search input)
"""

from __future__ import annotations

import calendar
import datetime
import math
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QRectF, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QColor,
    QContextMenuEvent,
    QCursor,
    QFont,
    QFontMetrics,
    QIcon,
    QImage,
    QKeyEvent,
    QLinearGradient,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QRadialGradient,
    QResizeEvent,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStyle,
    QStyleOptionButton,
    QStyleOptionToolButton,
    QStylePainter,
    QTabBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from src.db.repository import ArrangeMode
from src.imaging.opencv_engine import PhotoAdjustmentRecipe, render_adjusted_image
from src.utils.paths import get_icons_dir

import src.ui.icon_provider as icon_provider

ARROW_DROPDOWN_PATH = (get_icons_dir() / "arrow_dropdown_mini.svg").as_posix()

_ICON_CACHE: dict[tuple[str, int], QIcon] = {}


def _render_aero_raw(kind: str, size: int = 20) -> QIcon:
    """Renders crisp Windows Live Photo Gallery Aero-style vector icons onto a cached QIcon."""
    key = (kind, size)
    if key in _ICON_CACHE:
        return _ICON_CACHE[key]

    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.SmoothPixmapTransform, True)
    s = float(size)

    if kind in ("flower", "app_logo", "about", "photo_gallery", "about_photo_gallery"):
        for res_name in ("essentials_photo_gallery.png", "photogallery_live.png", "essentials_photo_gallery.ico"):
            p_ico = Path(__file__).resolve().parent.parent.parent / "resources" / "icons" / res_name
            if p_ico.exists():
                pix_ico = QPixmap(str(p_ico))
                if not pix_ico.isNull():
                    scaled = pix_ico.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    x_off = max(0, (size - scaled.width()) // 2)
                    y_off = max(0, (size - scaled.height()) // 2)
                    p.drawPixmap(x_off, y_off, scaled)
                    break
        p.end()
        icon = QIcon(pix)
        _ICON_CACHE[key] = icon
        return icon

    if kind in ("folder", "folder_add", "folder_open"):
        grad = QLinearGradient(0, s * 0.2, 0, s * 0.88)
        grad.setColorAt(0.0, QColor("#FFE88C"))
        grad.setColorAt(1.0, QColor("#E2B034"))
        p.setPen(QPen(QColor("#B6851A"), 1.0))
        p.setBrush(grad)
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.20, s * 0.36, s * 0.22), 2.0, 2.0)
        p.drawRoundedRect(QRectF(s * 0.08, s * 0.30, s * 0.84, s * 0.56), 2.2, 2.2)

        if kind == "folder_open":
            # Document sheet peeking out
            p.setPen(QPen(QColor("#CBD5E1"), 0.8))
            p.setBrush(QColor("#FFFFFF"))
            p.drawRect(QRectF(s * 0.22, s * 0.22, s * 0.44, s * 0.35))
            # Open front flap
            flap = QPolygonF([
                QPointF(s * 0.04, s * 0.86),
                QPointF(s * 0.20, s * 0.48),
                QPointF(s * 0.94, s * 0.48),
                QPointF(s * 0.80, s * 0.86),
            ])
            flap_grad = QLinearGradient(0, s * 0.48, 0, s * 0.86)
            flap_grad.setColorAt(0.0, QColor("#FFF0A6"))
            flap_grad.setColorAt(1.0, QColor("#EDB93B"))
            p.setPen(QPen(QColor("#B6851A"), 1.0))
            p.setBrush(flap_grad)
            p.drawPolygon(flap)

        elif kind == "folder_add":
            cx, cy, r = s * 0.74, s * 0.72, s * 0.22
            p.setPen(QPen(QColor("#1F7A1F"), 1.0))
            p.setBrush(QColor("#4CC94C"))
            p.drawEllipse(QPointF(cx, cy), r, r)
            p.setPen(QPen(QColor("#FFFFFF"), 1.8, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(cx - r * 0.5, cy), QPointF(cx + r * 0.5, cy))
            p.drawLine(QPointF(cx, cy - r * 0.5), QPointF(cx, cy + r * 0.5))

    elif kind in ("import", "import_media", "camera_import"):
        # Digital Camera Body + Optical Glass Lens + Blue Incoming Download Badge
        # 1. Shutter button top-left
        shutter_grad = QLinearGradient(s * 0.18, s * 0.16, s * 0.18, s * 0.26)
        shutter_grad.setColorAt(0.0, QColor("#E2E8F0"))
        shutter_grad.setColorAt(1.0, QColor("#64748B"))
        p.setPen(QPen(QColor("#334155"), 0.8))
        p.setBrush(shutter_grad)
        p.drawRoundedRect(QRectF(s * 0.18, s * 0.18, s * 0.16, s * 0.10), 1.5, 1.5)

        # 2. Viewfinder bump top-right
        p.setPen(QPen(QColor("#475569"), 0.8))
        p.setBrush(QColor("#94A3B8"))
        p.drawRoundedRect(QRectF(s * 0.44, s * 0.20, s * 0.26, s * 0.08), 1.5, 1.5)

        # 3. Main Camera Body
        body_rect = QRectF(s * 0.08, s * 0.25, s * 0.84, s * 0.58)
        body_grad = QLinearGradient(0, s * 0.25, 0, s * 0.83)
        body_grad.setColorAt(0.0, QColor("#FFFFFF"))
        body_grad.setColorAt(0.2, QColor("#EDF2F7"))
        body_grad.setColorAt(0.8, QColor("#CBD5E1"))
        body_grad.setColorAt(1.0, QColor("#94A3B8"))
        p.setPen(QPen(QColor("#475569"), 1.0))
        p.setBrush(body_grad)
        p.drawRoundedRect(body_rect, 3.5, 3.5)

        # Subtle grip accent line
        p.setPen(QPen(QColor("#E2E8F0"), 1.0))
        p.drawLine(QPointF(s * 0.12, s * 0.40), QPointF(s * 0.88, s * 0.40))
        p.setPen(QPen(QColor("#94A3B8"), 1.0))
        p.drawLine(QPointF(s * 0.12, s * 0.42), QPointF(s * 0.88, s * 0.42))

        # 4. Flash window top-right
        flash_rect = QRectF(s * 0.68, s * 0.31, s * 0.18, s * 0.11)
        p.setPen(QPen(QColor("#64748B"), 0.8))
        flash_grad = QLinearGradient(0, s * 0.31, 0, s * 0.42)
        flash_grad.setColorAt(0.0, QColor("#F8FAFC"))
        flash_grad.setColorAt(1.0, QColor("#CBD5E1"))
        p.setBrush(flash_grad)
        p.drawRoundedRect(flash_rect, 1.5, 1.5)

        # 5. Prominent circular optical lens
        lens_cx, lens_cy, lens_r = s * 0.42, s * 0.56, s * 0.23
        p.setPen(QPen(QColor("#334155"), 1.0))
        bezel_grad = QLinearGradient(0, lens_cy - lens_r, 0, lens_cy + lens_r)
        bezel_grad.setColorAt(0.0, QColor("#F1F5F9"))
        bezel_grad.setColorAt(0.5, QColor("#64748B"))
        bezel_grad.setColorAt(1.0, QColor("#1E293B"))
        p.setBrush(bezel_grad)
        p.drawEllipse(QPointF(lens_cx, lens_cy), lens_r, lens_r)

        # Inner deep optical glass
        glass_r = lens_r * 0.78
        glass_grad = QRadialGradient(lens_cx - glass_r * 0.3, lens_cy - glass_r * 0.3, glass_r * 1.2)
        glass_grad.setColorAt(0.0, QColor("#38BDF8"))
        glass_grad.setColorAt(0.4, QColor("#0284C7"))
        glass_grad.setColorAt(0.8, QColor("#0F172A"))
        glass_grad.setColorAt(1.0, QColor("#020617"))
        p.setPen(Qt.NoPen)
        p.setBrush(glass_grad)
        p.drawEllipse(QPointF(lens_cx, lens_cy), glass_r, glass_r)

        # Specular reflection arc on lens
        p.setPen(QPen(QColor(255, 255, 255, 200), 1.2, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(QRectF(lens_cx - glass_r * 0.7, lens_cy - glass_r * 0.7, glass_r * 1.4, glass_r * 1.4), 45 * 16, 90 * 16)

        # 6. Incoming Blue Download Arrow badge (bottom right)
        acx, acy = s * 0.76, s * 0.74
        badge_r = s * 0.20
        badge_grad = QLinearGradient(0, acy - badge_r, 0, acy + badge_r)
        badge_grad.setColorAt(0.0, QColor("#60A5FA"))
        badge_grad.setColorAt(1.0, QColor("#1D4ED8"))
        p.setPen(QPen(QColor("#FFFFFF"), 1.2))
        p.setBrush(badge_grad)
        p.drawEllipse(QPointF(acx, acy), badge_r, badge_r)

        arrow_path = QPainterPath()
        aw = badge_r * 0.55
        ah = badge_r * 0.65
        arrow_path.moveTo(acx, acy + ah * 0.7)
        arrow_path.lineTo(acx - aw, acy)
        arrow_path.lineTo(acx - aw * 0.4, acy)
        arrow_path.lineTo(acx - aw * 0.4, acy - ah * 0.7)
        arrow_path.lineTo(acx + aw * 0.4, acy - ah * 0.7)
        arrow_path.lineTo(acx + aw * 0.4, acy)
        arrow_path.lineTo(acx + aw, acy)
        arrow_path.closeSubpath()
        p.setPen(QPen(QColor("#1E3A8A"), 0.5))
        p.setBrush(QColor("#FFFFFF"))
        p.drawPath(arrow_path)

    elif kind in ("calendar", "dates", "dates_grid"):
        p.setPen(QPen(QColor("#6E87A2"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.76, s * 0.72), 2.0, 2.0)
        p.setBrush(QColor("#3A78B8"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.76, s * 0.22), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#5C7A99"))
        for r_idx in range(3 if kind != "calendar" else 2):
            for c_idx in range(4 if kind != "calendar" else 3):
                p.drawRect(
                    QRectF(
                        s * (0.20 + c_idx * (0.16 if kind != "calendar" else 0.20)),
                        s * (0.44 + r_idx * (0.13 if kind != "calendar" else 0.17)),
                        s * (0.09 if kind != "calendar" else 0.11),
                        s * (0.08 if kind != "calendar" else 0.10),
                    )
                )

    elif kind in ("months", "months_grid"):
        p.setPen(QPen(QColor("#45729F"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.76, s * 0.72), 2.0, 2.0)
        p.setBrush(QColor("#2468A8"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.76, s * 0.22), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#5C88B0"))
        for r_i in range(2):
            for c_i in range(2):
                p.drawRect(QRectF(s * (0.22 + c_i * 0.32), s * (0.46 + r_i * 0.18), s * 0.24, s * 0.12))

    elif kind in ("years", "years_stack"):
        p.setPen(QPen(QColor("#45729F"), 1.0))
        p.setBrush(QColor("#E8F1FA"))
        p.drawRoundedRect(QRectF(s * 0.26, s * 0.12, s * 0.62, s * 0.58), 2.0, 2.0)
        p.drawRoundedRect(QRectF(s * 0.18, s * 0.22, s * 0.62, s * 0.58), 2.0, 2.0)
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.32, s * 0.62, s * 0.58), 2.0, 2.0)
        p.setBrush(QColor("#2468A8"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.32, s * 0.62, s * 0.18), 2.0, 2.0)

    elif kind in ("text_search", "binoculars"):
        p.setPen(QPen(QColor("#1E5288"), 1.3))
        p.setBrush(QColor("#4A88C7"))
        p.drawRoundedRect(QRectF(s * 0.18, s * 0.30, s * 0.26, s * 0.52), 3.0, 3.0)
        p.drawRoundedRect(QRectF(s * 0.56, s * 0.30, s * 0.26, s * 0.52), 3.0, 3.0)
        p.setBrush(QColor("#1E3E66"))
        p.drawRoundedRect(QRectF(s * 0.22, s * 0.22, s * 0.18, s * 0.10), 2.0, 2.0)
        p.drawRoundedRect(QRectF(s * 0.60, s * 0.22, s * 0.18, s * 0.10), 2.0, 2.0)
        p.setPen(QPen(QColor("#1E3E66"), 2.0))
        p.drawLine(QPointF(s * 0.44, s * 0.50), QPointF(s * 0.56, s * 0.50))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 160))
        p.drawEllipse(QPointF(s * 0.28, s * 0.70), s * 0.06, s * 0.06)
        p.drawEllipse(QPointF(s * 0.66, s * 0.70), s * 0.06, s * 0.06)

    elif kind == "edited":
        p.setPen(QPen(QColor("#6B5218"), 1.0))
        p.setBrush(QColor("#F0B838"))
        pencil = QPolygonF(
            [
                QPointF(s * 0.76, s * 0.18),
                QPointF(s * 0.84, s * 0.26),
                QPointF(s * 0.34, s * 0.76),
                QPointF(s * 0.20, s * 0.82),
                QPointF(s * 0.26, s * 0.68),
            ]
        )
        p.drawPolygon(pencil)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#333333"))
        tip = QPolygonF(
            [
                QPointF(s * 0.20, s * 0.82),
                QPointF(s * 0.24, s * 0.73),
                QPointF(s * 0.29, s * 0.78),
            ]
        )
        p.drawPolygon(tip)

    elif kind in ("not_edited", "not-edited"):
        p.setPen(QPen(QColor("#6B5218"), 1.0))
        p.setBrush(QColor("#F0B838"))
        pencil = QPolygonF(
            [
                QPointF(s * 0.76, s * 0.18),
                QPointF(s * 0.84, s * 0.26),
                QPointF(s * 0.34, s * 0.76),
                QPointF(s * 0.20, s * 0.82),
                QPointF(s * 0.26, s * 0.68),
            ]
        )
        p.drawPolygon(pencil)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#333333"))
        tip = QPolygonF(
            [
                QPointF(s * 0.20, s * 0.82),
                QPointF(s * 0.24, s * 0.73),
                QPointF(s * 0.29, s * 0.78),
            ]
        )
        p.drawPolygon(tip)
        badge_r = s * 0.22
        badge_cx = s * 0.38
        badge_cy = s * 0.74
        p.setPen(QPen(QColor("#1B4E82"), 1.0))
        p.setBrush(QColor("#2468A8"))
        p.drawEllipse(QPointF(badge_cx, badge_cy), badge_r, badge_r)
        p.setPen(QPen(QColor("#FFFFFF"), max(1.2, s * 0.09), Qt.SolidLine, Qt.SquareCap))
        p.drawLine(QPointF(badge_cx - badge_r * 0.55, badge_cy), QPointF(badge_cx + badge_r * 0.55, badge_cy))

    elif kind == "media_type":
        p.setPen(QPen(QColor("#2468A8"), 1.0))
        p.setBrush(QColor("#E8F1FA"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.24, s * 0.52, s * 0.54), 2.0, 2.0)
        p.setBrush(QColor("#5488BD"))
        p.drawPolygon(
            QPolygonF(
                [
                    QPointF(s * 0.14, s * 0.70),
                    QPointF(s * 0.32, s * 0.46),
                    QPointF(s * 0.44, s * 0.60),
                    QPointF(s * 0.58, s * 0.70),
                ]
            )
        )
        p.setPen(QPen(QColor("#1A4E80"), 1.0))
        p.setBrush(QColor("#4A78A8"))
        p.drawRoundedRect(QRectF(s * 0.54, s * 0.38, s * 0.36, s * 0.42), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#FFFFFF"))
        tri = QPolygonF(
            [
                QPointF(s * 0.64, s * 0.46),
                QPointF(s * 0.80, s * 0.59),
                QPointF(s * 0.64, s * 0.72),
            ]
        )
        p.drawPolygon(tri)

    elif kind == "orientation":
        # Landscape frame (horizontal, behind)
        p.setPen(QPen(QColor("#2468A8"), 1.0))
        p.setBrush(QColor("#E8F1FA"))
        p.drawRoundedRect(QRectF(s * 0.08, s * 0.28, s * 0.58, s * 0.44), 1.5, 1.5)
        p.setBrush(QColor("#7DA2CE"))
        p.setPen(Qt.NoPen)
        p.drawPolygon(
            QPolygonF(
                [
                    QPointF(s * 0.12, s * 0.66),
                    QPointF(s * 0.26, s * 0.44),
                    QPointF(s * 0.38, s * 0.58),
                    QPointF(s * 0.54, s * 0.38),
                    QPointF(s * 0.62, s * 0.66),
                ]
            )
        )
        # Portrait frame (vertical, in front)
        p.setPen(QPen(QColor("#1A4E80"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.46, s * 0.14, s * 0.44, s * 0.66), 1.5, 1.5)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#5488BD"))
        p.drawRoundedRect(QRectF(s * 0.52, s * 0.22, s * 0.32, s * 0.48), 1.0, 1.0)

    elif kind in ("tag", "tag_add"):
        path = QPainterPath()
        path.moveTo(s * 0.14, s * 0.46)
        path.lineTo(s * 0.46, s * 0.14)
        path.lineTo(s * 0.84, s * 0.14)
        path.lineTo(s * 0.84, s * 0.52)
        path.lineTo(s * 0.52, s * 0.84)
        path.closeSubpath()
        grad = QLinearGradient(0, 0, s, s)
        grad.setColorAt(0.0, QColor("#74B2F0"))
        grad.setColorAt(1.0, QColor("#2D6FB7"))
        p.setPen(QPen(QColor("#1E5288"), 1.1))
        p.setBrush(grad)
        p.drawPath(path)
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(QPointF(s * 0.67, s * 0.31), s * 0.07, s * 0.07)

    elif kind == "not_tagged":
        path = QPainterPath()
        path.moveTo(s * 0.14, s * 0.46)
        path.lineTo(s * 0.46, s * 0.14)
        path.lineTo(s * 0.84, s * 0.14)
        path.lineTo(s * 0.84, s * 0.52)
        path.lineTo(s * 0.52, s * 0.84)
        path.closeSubpath()
        p.setPen(QPen(QColor("#7A8A9E"), 1.0))
        p.setBrush(QColor("#D9E2EC"))
        p.drawPath(path)
        p.setPen(QPen(QColor("#C5221F"), 1.6))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QRectF(s * 0.20, s * 0.20, s * 0.60, s * 0.60))
        p.drawLine(QPointF(s * 0.32, s * 0.32), QPointF(s * 0.68, s * 0.68))

    elif kind == "tag_hierarchy":
        p.setPen(QPen(QColor("#356AA0"), 1.0))
        p.setBrush(QColor("#F5FAFF"))
        p.drawLine(QPointF(s * 0.22, s * 0.24), QPointF(s * 0.22, s * 0.76))
        p.drawLine(QPointF(s * 0.22, s * 0.36), QPointF(s * 0.42, s * 0.36))
        p.drawLine(QPointF(s * 0.22, s * 0.68), QPointF(s * 0.42, s * 0.68))
        p.setBrush(QColor("#4B88C4"))
        p.drawRoundedRect(QRectF(s * 0.42, s * 0.24, s * 0.46, s * 0.22), 1.5, 1.5)
        p.drawRoundedRect(QRectF(s * 0.42, s * 0.56, s * 0.46, s * 0.22), 1.5, 1.5)

    elif kind in ("person", "people", "people_tag", "batch_people", "person_question", "person_none", "tag_as"):
        if kind == "batch_people":
            p.setPen(QPen(QColor("#2A5E8A"), 1.0))
            p.setBrush(QColor("#82B8E6"))
            p.drawEllipse(QPointF(s * 0.66, s * 0.30), s * 0.15, s * 0.15)
            p.drawRoundedRect(
                QRectF(s * 0.44, s * 0.50, s * 0.44, s * 0.32), s * 0.14, s * 0.14
            )
        if kind == "person_none":
            grad = QLinearGradient(0, s * 0.15, 0, s * 0.88)
            grad.setColorAt(0.0, QColor("#CFD8DC"))
            grad.setColorAt(1.0, QColor("#90A4AE"))
            p.setPen(QPen(QColor("#78909C"), 1.1))
            p.setBrush(grad)
        else:
            grad = QLinearGradient(0, s * 0.15, 0, s * 0.88)
            grad.setColorAt(0.0, QColor("#5CA4E2"))
            grad.setColorAt(1.0, QColor("#2468A8"))
            p.setPen(QPen(QColor("#1B4E80"), 1.1))
            p.setBrush(grad)
        cx = s * 0.42 if kind == "batch_people" else s * 0.50
        p.drawEllipse(QPointF(cx, s * 0.32), s * 0.18, s * 0.18)
        body = QPainterPath()
        body.moveTo(cx - s * 0.30, s * 0.86)
        body.quadTo(cx - s * 0.28, s * 0.54, cx, s * 0.54)
        body.quadTo(cx + s * 0.28, s * 0.54, cx + s * 0.30, s * 0.86)
        body.closeSubpath()
        p.drawPath(body)

        if kind == "person_question":
            p.setPen(QPen(QColor("#C68400"), 1.0))
            p.setBrush(QColor("#FFD54F"))
            p.drawEllipse(QPointF(s * 0.78, s * 0.72), s * 0.18, s * 0.18)
            p.setPen(QPen(QColor("#5D4037"), 1.2))
            p.setFont(QFont("Segoe UI", int(s * 0.24), QFont.Bold))
            p.drawText(QRectF(s * 0.60, s * 0.54, s * 0.36, s * 0.36), Qt.AlignCenter, "?")
        elif kind == "person_none":
            p.setPen(QPen(QColor("#C9302C"), 1.0))
            p.setBrush(QColor("#FFEBEE"))
            p.drawEllipse(QPointF(s * 0.78, s * 0.72), s * 0.18, s * 0.18)
            p.setPen(QPen(QColor("#C9302C"), 1.8, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(s * 0.68, s * 0.72), QPointF(s * 0.88, s * 0.72))
        elif kind == "tag_as":
            p.setPen(QPen(QColor("#1E5288"), 1.0))
            p.setBrush(QColor("#38BDF8"))
            tag_p = QPolygonF([
                QPointF(s * 0.62, s * 0.60),
                QPointF(s * 0.86, s * 0.60),
                QPointF(s * 0.86, s * 0.80),
                QPointF(s * 0.74, s * 0.92),
                QPointF(s * 0.62, s * 0.80),
            ])
            p.drawPolygon(tag_p)
            p.setBrush(QColor("#FFFFFF"))
            p.drawEllipse(QPointF(s * 0.74, s * 0.68), 1.5, 1.5)

    elif kind == "caption":
        # Landscape photo card with note/callout matching WLPG
        p.setPen(QPen(QColor("#6A8AA8"), 1.1))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.12, s * 0.80, s * 0.68), 2.0, 2.0)
        # Inner photo area (sky + hills)
        photo_rect = QRectF(s * 0.16, s * 0.18, s * 0.68, s * 0.38)
        p.setClipRect(photo_rect)
        sky = QLinearGradient(0, s * 0.18, 0, s * 0.45)
        sky.setColorAt(0.0, QColor("#5FA9EE"))
        sky.setColorAt(1.0, QColor("#BBE1FD"))
        p.setPen(Qt.NoPen)
        p.setBrush(sky)
        p.drawRect(photo_rect)
        # Green hills
        p.setBrush(QColor("#48A238"))
        p.drawChord(QRectF(s * 0.12, s * 0.32, s * 0.50, s * 0.36), 0, 180 * 16)
        p.setBrush(QColor("#66BB44"))
        p.drawChord(QRectF(s * 0.38, s * 0.35, s * 0.50, s * 0.32), 0, 180 * 16)
        p.setClipping(False)
        # Caption box at bottom
        p.setPen(QPen(QColor("#2B74B9"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.20, s * 0.52, s * 0.60, s * 0.22), 1.5, 1.5)
        p.setPen(QPen(QColor("#2B74B9"), 1.2, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.26, s * 0.59), QPointF(s * 0.70, s * 0.59))
        p.drawLine(QPointF(s * 0.26, s * 0.66), QPointF(s * 0.58, s * 0.66))

    elif kind in ("geotag", "globe"):
        grad = QLinearGradient(0, s * 0.1, s, s * 0.9)
        grad.setColorAt(0.0, QColor("#63B8F2"))
        grad.setColorAt(1.0, QColor("#1D6AB5"))
        p.setPen(QPen(QColor("#18528C"), 1.1))
        p.setBrush(grad)
        p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.38, s * 0.38)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#5CB85C"))
        p.drawEllipse(QPointF(s * 0.40, s * 0.40), s * 0.14, s * 0.10)
        p.drawEllipse(QPointF(s * 0.60, s * 0.56), s * 0.12, s * 0.15)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 130), 0.9))
        p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.20, s * 0.37)
        p.drawLine(QPointF(s * 0.13, s * 0.5), QPointF(s * 0.87, s * 0.5))

    elif kind in ("rotate_ccw", "rotate_cw"):
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#70A4D4"))
        p.drawPolygon(
            QPolygonF(
                [
                    QPointF(s * 0.16, s * 0.82),
                    QPointF(s * 0.44, s * 0.50),
                    QPointF(s * 0.60, s * 0.68),
                    QPointF(s * 0.84, s * 0.82),
                ]
            )
        )
        p.setPen(QPen(QColor("#2468A8"), 1.8, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        rect = QRectF(s * 0.20, s * 0.16, s * 0.60, s * 0.60)
        if kind == "rotate_ccw":
            p.drawArc(rect, 30 * 16, 210 * 16)
            head = QPolygonF(
                [
                    QPointF(s * 0.16, s * 0.40),
                    QPointF(s * 0.34, s * 0.20),
                    QPointF(s * 0.36, s * 0.44),
                ]
            )
        else:
            p.drawArc(rect, -60 * 16, 210 * 16)
            head = QPolygonF(
                [
                    QPointF(s * 0.84, s * 0.40),
                    QPointF(s * 0.66, s * 0.20),
                    QPointF(s * 0.64, s * 0.44),
                ]
            )
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2468A8"))
        p.drawPolygon(head)

    elif kind in ("zoom_out", "zoom_in"):
        # Magnifying glass with metallic frame and glass reflection
        p.setPen(QPen(QColor("#7A96B2"), 1.4))
        glass_grad = QLinearGradient(s * 0.2, s * 0.2, s * 0.7, s * 0.7)
        glass_grad.setColorAt(0.0, QColor("#EAF4FD"))
        glass_grad.setColorAt(1.0, QColor("#CCE4F7"))
        p.setBrush(glass_grad)
        p.drawEllipse(QPointF(s * 0.44, s * 0.44), s * 0.32, s * 0.32)
        # Handle
        p.setPen(QPen(QColor("#5A6E82"), 3.2, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.68, s * 0.68), QPointF(s * 0.88, s * 0.88))
        # Plus or minus sign inside
        p.setPen(QPen(QColor("#1976D2"), 2.2, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.28, s * 0.44), QPointF(s * 0.60, s * 0.44))
        if kind == "zoom_in":
            p.drawLine(QPointF(s * 0.44, s * 0.28), QPointF(s * 0.44, s * 0.60))

    elif kind in ("fit_window", "fit_photos"):
        p.setPen(QPen(QColor("#2B78C5"), 1.6, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        p.drawLine(QPointF(s * 0.18, s * 0.36), QPointF(s * 0.18, s * 0.20))
        p.drawLine(QPointF(s * 0.18, s * 0.20), QPointF(s * 0.34, s * 0.20))
        p.drawLine(QPointF(s * 0.66, s * 0.20), QPointF(s * 0.82, s * 0.20))
        p.drawLine(QPointF(s * 0.82, s * 0.20), QPointF(s * 0.82, s * 0.36))
        p.drawLine(QPointF(s * 0.18, s * 0.64), QPointF(s * 0.18, s * 0.80))
        p.drawLine(QPointF(s * 0.18, s * 0.80), QPointF(s * 0.34, s * 0.80))
        p.drawLine(QPointF(s * 0.66, s * 0.80), QPointF(s * 0.82, s * 0.80))
        p.drawLine(QPointF(s * 0.82, s * 0.80), QPointF(s * 0.82, s * 0.64))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2B78C5"))
        p.drawRect(QRectF(s * 0.42, s * 0.42, s * 0.16, s * 0.16))

    elif kind == "reverse_sort":
        p.setPen(QPen(QColor("#1A5C96"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.setBrush(QColor("#2468A8"))
        p.drawLine(QPointF(s * 0.35, s * 0.80), QPointF(s * 0.35, s * 0.24))
        tri_up = QPolygonF([QPointF(s * 0.20, s * 0.38), QPointF(s * 0.35, s * 0.18), QPointF(s * 0.50, s * 0.38)])
        p.drawPolygon(tri_up)
        p.drawLine(QPointF(s * 0.65, s * 0.20), QPointF(s * 0.65, s * 0.76))
        tri_dn = QPolygonF([QPointF(s * 0.50, s * 0.62), QPointF(s * 0.65, s * 0.82), QPointF(s * 0.80, s * 0.62)])
        p.drawPolygon(tri_dn)

    elif kind in ("actual_size", "actual_pixels"):
        p.setPen(QPen(QColor("#0078D7"), 1.6, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        # 4 Corner brackets
        p.drawLine(QPointF(s * 0.16, s * 0.36), QPointF(s * 0.16, s * 0.20))
        p.drawLine(QPointF(s * 0.16, s * 0.20), QPointF(s * 0.32, s * 0.20))
        p.drawLine(QPointF(s * 0.68, s * 0.20), QPointF(s * 0.84, s * 0.20))
        p.drawLine(QPointF(s * 0.84, s * 0.20), QPointF(s * 0.84, s * 0.36))
        p.drawLine(QPointF(s * 0.16, s * 0.64), QPointF(s * 0.16, s * 0.80))
        p.drawLine(QPointF(s * 0.16, s * 0.80), QPointF(s * 0.32, s * 0.80))
        p.drawLine(QPointF(s * 0.68, s * 0.80), QPointF(s * 0.84, s * 0.80))
        p.drawLine(QPointF(s * 0.84, s * 0.80), QPointF(s * 0.84, s * 0.64))
        # Central filled blue rectangle
        p.setPen(QPen(QColor("#005A9E"), 1.0))
        p.setBrush(QColor("#0078D7"))
        p.drawRoundedRect(QRectF(s * 0.36, s * 0.36, s * 0.28, s * 0.28), 1.0, 1.0)
        # 4 inward arrows
        p.setBrush(QColor("#005A9E"))
        p.drawPolygon(QPolygonF([QPointF(s * 0.22, s * 0.22), QPointF(s * 0.32, s * 0.26), QPointF(s * 0.26, s * 0.32)]))
        p.drawPolygon(QPolygonF([QPointF(s * 0.78, s * 0.22), QPointF(s * 0.68, s * 0.26), QPointF(s * 0.74, s * 0.32)]))
        p.drawPolygon(QPolygonF([QPointF(s * 0.22, s * 0.78), QPointF(s * 0.32, s * 0.74), QPointF(s * 0.26, s * 0.68)]))
        p.drawPolygon(QPolygonF([QPointF(s * 0.78, s * 0.78), QPointF(s * 0.68, s * 0.74), QPointF(s * 0.74, s * 0.68)]))

    elif kind in ("name", "sort_name"):
        p.setPen(QPen(QColor("#2468A8"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.14, s * 0.76, s * 0.72), 2.0, 2.0)
        p.setPen(QColor("#1A5C96"))
        p.setFont(QFont("Segoe UI", int(s * 0.34), QFont.Bold))
        p.drawText(QRectF(0, 0, s, s), Qt.AlignCenter, "AZ")

    elif kind == "published":
        p.setPen(QPen(QColor("#2468A8"), 1.2))
        p.setBrush(QColor("#D6EAF8"))
        p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.36, s * 0.36)
        p.drawLine(QPointF(s * 0.14, s * 0.5), QPointF(s * 0.86, s * 0.5))
        p.drawArc(QRectF(s * 0.30, s * 0.14, s * 0.40, s * 0.72), 0, 360 * 16)

    elif kind == "view_list":
        p.setPen(QPen(QColor("#2B78C5"), 1.8, Qt.SolidLine, Qt.RoundCap))
        for y_f in (0.30, 0.50, 0.70):
            p.drawPoint(QPointF(s * 0.24, s * y_f))
            p.drawLine(QPointF(s * 0.40, s * y_f), QPointF(s * 0.82, s * y_f))

    elif kind == "view_details":
        # Mini photo card on the left + horizontal detail lines on the right
        p.setPen(QPen(QColor("#2468A8"), 1.1))
        p.setBrush(QColor("#DCEBFA"))
        thumb_r = QRectF(s * 0.10, s * 0.22, s * 0.34, s * 0.56)
        p.drawRoundedRect(thumb_r, 1.2, 1.2)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#5290CF"))
        p.drawPolygon(
            QPolygonF(
                [
                    QPointF(thumb_r.left() + 1.0, thumb_r.bottom() - 1.0),
                    QPointF(thumb_r.left() + thumb_r.width() * 0.48, thumb_r.top() + thumb_r.height() * 0.52),
                    QPointF(thumb_r.right() - 1.0, thumb_r.bottom() - 1.0),
                ]
            )
        )
        p.setPen(QPen(QColor("#2468A8"), max(1.4, s * 0.085), Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.54, s * 0.31), QPointF(s * 0.89, s * 0.31))
        p.drawLine(QPointF(s * 0.54, s * 0.50), QPointF(s * 0.89, s * 0.50))
        p.drawLine(QPointF(s * 0.54, s * 0.69), QPointF(s * 0.78, s * 0.69))

    elif kind in ("prev_media", "next_media"):
        p.setPen(QPen(QColor("#2468A8"), 1.8, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(QColor("#2468A8"))
        if kind == "prev_media":
            p.drawLine(QPointF(s * 0.25, s * 0.25), QPointF(s * 0.25, s * 0.75))
            tri = QPolygonF(
                [
                    QPointF(s * 0.75, s * 0.25),
                    QPointF(s * 0.35, s * 0.50),
                    QPointF(s * 0.75, s * 0.75),
                ]
            )
        else:
            p.drawLine(QPointF(s * 0.75, s * 0.25), QPointF(s * 0.75, s * 0.75))
            tri = QPolygonF(
                [
                    QPointF(s * 0.25, s * 0.25),
                    QPointF(s * 0.65, s * 0.50),
                    QPointF(s * 0.25, s * 0.75),
                ]
            )
        p.drawPolygon(tri)

    elif kind in ("delete", "cross", "reject", "not_this_person", "close", "cancel"):
        p.setPen(QPen(QColor("#C9302C"), max(2.4, s * 0.12), Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.24, s * 0.24), QPointF(s * 0.76, s * 0.76))
        p.drawLine(QPointF(s * 0.76, s * 0.24), QPointF(s * 0.24, s * 0.76))

    elif kind in ("check", "confirm", "checkmark"):
        # Glossy Aero vibrant green checkmark
        path = QPainterPath()
        path.moveTo(s * 0.16, s * 0.52)
        path.lineTo(s * 0.38, s * 0.76)
        path.lineTo(s * 0.84, s * 0.22)
        p.setPen(QPen(QColor("#104B1F"), max(1.4, s * 0.08), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPath(path)
        grad = QLinearGradient(0, s * 0.2, 0, s * 0.8)
        grad.setColorAt(0.0, QColor("#54D26E"))
        grad.setColorAt(0.5, QColor("#28A745"))
        grad.setColorAt(1.0, QColor("#19692C"))
        p.setPen(QPen(grad, max(2.4, s * 0.16), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPath(path)

    elif kind in ("warning", "alert", "outlier"):
        # Amber / golden warning triangle with '!'
        p.setPen(QPen(QColor("#B45309"), 1.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        tri_grad = QLinearGradient(0, s * 0.14, 0, s * 0.88)
        tri_grad.setColorAt(0.0, QColor("#FDE047"))
        tri_grad.setColorAt(0.5, QColor("#F59E0B"))
        tri_grad.setColorAt(1.0, QColor("#D97706"))
        p.setBrush(tri_grad)
        tri = QPolygonF([
            QPointF(s * 0.50, s * 0.14),
            QPointF(s * 0.88, s * 0.86),
            QPointF(s * 0.12, s * 0.86),
        ])
        p.drawPolygon(tri)
        p.setPen(QPen(QColor("#78350F"), max(1.5, s * 0.10), Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.50, s * 0.38), QPointF(s * 0.50, s * 0.62))
        p.drawPoint(QPointF(s * 0.50, s * 0.74))

    elif kind in ("pin", "pin_active"):
        p.setPen(QPen(QColor("#24548A") if kind == "pin_active" else QColor("#5A738E"), 1.2))
        p.setBrush(QColor("#2B78C5") if kind == "pin_active" else QColor("#8AA4BE"))
        if kind == "pin_active":
            p.drawRoundedRect(QRectF(s * 0.35, s * 0.15, s * 0.30, s * 0.22), 1.5, 1.5)
            p.drawRect(QRectF(s * 0.42, s * 0.37, s * 0.16, s * 0.28))
            p.setPen(QPen(QColor("#1A3A60"), 1.8, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(s * 0.50, s * 0.65), QPointF(s * 0.50, s * 0.90))
        else:
            p.save()
            p.translate(s * 0.5, s * 0.5)
            p.rotate(-45)
            p.drawRoundedRect(QRectF(-s * 0.18, -s * 0.34, s * 0.36, s * 0.20), 1.5, 1.5)
            p.drawRect(QRectF(-s * 0.10, -s * 0.14, s * 0.20, s * 0.24))
            p.setPen(QPen(QColor("#455A6E"), 1.6, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(0, s * 0.10), QPointF(0, s * 0.38))
            p.restore()

    elif kind in ("flag", "flag_on", "flag_off"):
        p.setPen(QPen(QColor("#5A6978"), 1.5, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.26, s * 0.16), QPointF(s * 0.26, s * 0.86))
        banner = QPolygonF(
            [
                QPointF(s * 0.26, s * 0.18),
                QPointF(s * 0.80, s * 0.30),
                QPointF(s * 0.26, s * 0.56),
            ]
        )
        if kind == "flag_off":
            p.setPen(QPen(QColor("#7A8B9C"), 1.2))
            p.setBrush(QColor(255, 255, 255, 180))
        else:
            p.setPen(QPen(QColor("#B52B27"), 1.1))
            p.setBrush(QColor("#E64833"))
        p.drawPolygon(banner)

    elif kind in ("star", "rating", "star_filled", "star_empty", "rate"):
        pts: list[QPointF] = []
        cx, cy = s * 0.5, s * 0.5
        outer_r, inner_r = s * 0.44, s * 0.19
        for i in range(10):
            ang = math.radians(-90 + i * 36)
            r = outer_r if i % 2 == 0 else inner_r
            pts.append(QPointF(cx + r * math.cos(ang), cy + r * math.sin(ang)))
        if kind == "star_empty":
            p.setPen(QPen(QColor("#8C9BAE"), 1.2))
            p.setBrush(Qt.NoBrush)
            p.drawPolygon(QPolygonF(pts))
        else:
            grad = QLinearGradient(0, s * 0.1, 0, s * 0.9)
            grad.setColorAt(0.0, QColor("#FFF4A3"))
            grad.setColorAt(0.4, QColor("#F8D038"))
            grad.setColorAt(1.0, QColor("#D99E10"))
            p.setPen(QPen(QColor("#B27B08"), 1.0))
            p.setBrush(grad)
            p.drawPolygon(QPolygonF(pts))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 110))
            p.drawPolygon(QPolygonF([pts[0], pts[1], QPointF(cx, cy)]))
            p.drawPolygon(QPolygonF([pts[8], pts[9], QPointF(cx, cy)]))

    elif kind == "slideshow":
        # Projector screen on stand matching WLPG
        p.setPen(QPen(QColor("#455A64"), 1.2))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.12, s * 0.76, s * 0.52), 1.5, 1.5)
        p.setClipRect(QRectF(s * 0.15, s * 0.15, s * 0.70, s * 0.46))
        sky = QLinearGradient(0, s * 0.15, 0, s * 0.45)
        sky.setColorAt(0.0, QColor("#64B5F6"))
        sky.setColorAt(1.0, QColor("#BBDEFB"))
        p.setPen(Qt.NoPen)
        p.setBrush(sky)
        p.drawRect(QRectF(s * 0.15, s * 0.15, s * 0.70, s * 0.46))
        p.setBrush(QColor("#43A047"))
        p.drawChord(QRectF(s * 0.10, s * 0.32, s * 0.55, s * 0.36), 0, 180 * 16)
        p.setBrush(QColor("#66BB6A"))
        p.drawChord(QRectF(s * 0.40, s * 0.35, s * 0.50, s * 0.32), 0, 180 * 16)
        p.setClipping(False)
        p.setPen(QPen(QColor("#455A64"), 1.6, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.50, s * 0.64), QPointF(s * 0.50, s * 0.86))
        p.drawLine(QPointF(s * 0.50, s * 0.74), QPointF(s * 0.30, s * 0.88))
        p.drawLine(QPointF(s * 0.50, s * 0.74), QPointF(s * 0.70, s * 0.88))

    elif kind == "copy_subject":
        # Silhouette of a foreground person with a clean duplicate copy card and blue clipboard/copy badge
        p.setPen(QPen(QColor("#7BA2CE"), 1.0))
        p.setBrush(QColor("#F0F4FA"))
        p.drawRoundedRect(QRectF(s * 0.28, s * 0.12, s * 0.58, s * 0.62), 2.0, 2.0)
        # Front card containing subject silhouette
        p.setPen(QPen(QColor("#0066CC"), 1.2))
        p.setBrush(QColor("#FFFFFF"))
        card_r = QRectF(s * 0.12, s * 0.24, s * 0.58, s * 0.64)
        p.drawRoundedRect(card_r, 2.0, 2.0)
        # Stylized person silhouette in blue
        head_cx, head_cy, head_r = card_r.center().x(), card_r.top() + card_r.height() * 0.28, card_r.width() * 0.18
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#0066CC"))
        p.drawEllipse(QPointF(head_cx, head_cy), head_r, head_r)
        # Shoulders
        body_path = QPainterPath()
        body_path.moveTo(head_cx - head_r * 1.5, card_r.bottom() - 2.0)
        body_path.quadTo(head_cx, head_cy + head_r * 0.8, head_cx + head_r * 1.5, card_r.bottom() - 2.0)
        body_path.closeSubpath()
        p.drawPath(body_path)

    elif kind == "remove_background":
        # Checkerboard transparency grid on right side + solid cutout silhouette on left
        grid_r = QRectF(s * 0.12, s * 0.12, s * 0.76, s * 0.76)
        p.setPen(QPen(QColor("#9EADC0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(grid_r, 2.5, 2.5)
        p.setClipRect(grid_r)
        # Draw checkerboard pattern
        cell = s * 0.19
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#DDE5EE"))
        for row in range(4):
            for col in range(4):
                if (row + col) % 2 == 1:
                    p.drawRect(QRectF(grid_r.left() + col * cell, grid_r.top() + row * cell, cell, cell))
        # Draw subject cutout silhouette overlay
        scx, scy = grid_r.center().x() - s * 0.04, grid_r.top() + grid_r.height() * 0.30
        sr = s * 0.16
        p.setBrush(QColor("#1E395B"))
        p.drawEllipse(QPointF(scx, scy), sr, sr)
        sb = QPainterPath()
        sb.moveTo(scx - sr * 1.6, grid_r.bottom())
        sb.quadTo(scx, scy + sr * 0.85, scx + sr * 1.6, grid_r.bottom())
        sb.closeSubpath()
        p.drawPath(sb)
        # Magic wand / erase spark on top-right
        p.setClipping(False)
        p.setPen(QPen(QColor("#D9534F"), 1.5, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.76, s * 0.20), QPointF(s * 0.86, s * 0.10))
        p.drawLine(QPointF(s * 0.86, s * 0.20), QPointF(s * 0.76, s * 0.10))

    elif kind in ("detail_all", "all_details"):
        p.setPen(QPen(QColor("#6A85A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRect(QRectF(s * 0.16, s * 0.12, s * 0.68, s * 0.76))
        p.setPen(QPen(QColor("#2B74B9"), 1.2, Qt.SolidLine, Qt.RoundCap))
        for y_f in (0.30, 0.46, 0.62, 0.76):
            p.drawPoint(QPointF(s * 0.30, s * y_f))
            p.drawLine(QPointF(s * 0.42, s * y_f), QPointF(s * 0.72, s * y_f))

    elif kind in ("detail_filename", "filename"):
        p.setPen(QPen(QColor("#6A85A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRect(QRectF(s * 0.16, s * 0.12, s * 0.68, s * 0.76))
        p.setPen(QColor("#2468A8"))
        p.setFont(QFont("Segoe UI", int(s * 0.40), QFont.Bold))
        p.drawText(QRectF(s * 0.16, s * 0.16, s * 0.68, s * 0.68), Qt.AlignCenter, "ab")

    elif kind in ("detail_filesize", "filesize"):
        p.setPen(QPen(QColor("#6A85A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRect(QRectF(s * 0.16, s * 0.12, s * 0.68, s * 0.76))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#D3E3F5"))
        p.drawRect(QRectF(s * 0.26, s * 0.45, s * 0.48, s * 0.24))
        p.setBrush(QColor("#2B74B9"))
        p.drawRect(QRectF(s * 0.26, s * 0.45, s * 0.28, s * 0.24))

    elif kind in ("detail_date_taken", "date_taken"):
        p.setPen(QPen(QColor("#6A85A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.14, s * 0.16, s * 0.72, s * 0.70), 1.5, 1.5)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#D32F2F"))
        p.drawRoundedRect(QRectF(s * 0.14, s * 0.16, s * 0.72, s * 0.22), 1.5, 1.5)
        p.drawRect(QRectF(s * 0.14, s * 0.28, s * 0.72, s * 0.10))
        p.setPen(QPen(QColor("#78909C"), 1.0))
        p.drawLine(QPointF(s * 0.28, s * 0.52), QPointF(s * 0.72, s * 0.52))
        p.drawLine(QPointF(s * 0.28, s * 0.68), QPointF(s * 0.72, s * 0.68))

    elif kind in ("detail_dimensions", "dimensions"):
        p.setPen(QPen(QColor("#6A85A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRect(QRectF(s * 0.14, s * 0.18, s * 0.72, s * 0.64))
        p.setPen(QPen(QColor("#2B74B9"), 1.2, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.22, s * 0.30), QPointF(s * 0.78, s * 0.30))
        p.drawLine(QPointF(s * 0.78, s * 0.30), QPointF(s * 0.78, s * 0.72))

    elif kind in ("detail_date_modified", "date_modified"):
        p.setPen(QPen(QColor("#6A85A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.14, s * 0.16, s * 0.72, s * 0.70), 1.5, 1.5)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#1976D2"))
        p.drawRoundedRect(QRectF(s * 0.14, s * 0.16, s * 0.72, s * 0.22), 1.5, 1.5)
        p.drawRect(QRectF(s * 0.14, s * 0.28, s * 0.72, s * 0.10))
        p.setPen(QPen(QColor("#546E7A"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(QPointF(s * 0.62, s * 0.64), s * 0.20, s * 0.20)
        p.drawLine(QPointF(s * 0.62, s * 0.64), QPointF(s * 0.62, s * 0.52))
        p.drawLine(QPointF(s * 0.62, s * 0.64), QPointF(s * 0.72, s * 0.64))

    elif kind in ("layout_grid", "thumbnail_view", "thumbnail_grid"):
        # Crisp 2x2 grid of framed photo thumbnails
        tile_w = s * 0.35
        tile_h = s * 0.35
        offsets = (0.11, 0.54)
        for r_i in range(2):
            for c_i in range(2):
                tx = s * offsets[c_i]
                ty = s * offsets[r_i]
                tr = QRectF(tx, ty, tile_w, tile_h)
                p.setPen(QPen(QColor("#1F5D99"), max(1.0, s * 0.055)))
                p.setBrush(QColor("#FFFFFF"))
                p.drawRoundedRect(tr, 1.2, 1.2)
                inner = tr.adjusted(1.1, 1.1, -1.1, -1.1)
                if inner.width() > 2.0 and inner.height() > 2.0:
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor("#CFE6FC"))
                    p.drawRect(inner)
                    p.setBrush(QColor("#4B8AC9"))
                    p.drawPolygon(
                        QPolygonF(
                            [
                                QPointF(inner.left(), inner.bottom()),
                                QPointF(inner.left() + inner.width() * 0.45, inner.top() + inner.height() * 0.45),
                                QPointF(inner.right(), inner.bottom()),
                            ]
                        )
                    )

    elif kind == "layout_mosaic":
        p.setPen(QPen(QColor("#2B6299"), 1.1))
        p.setBrush(QColor("#69ACEA"))
        p.drawRect(QRectF(s * 0.12, s * 0.14, s * 0.44, s * 0.32))
        p.drawRect(QRectF(s * 0.62, s * 0.14, s * 0.26, s * 0.32))
        p.drawRect(QRectF(s * 0.12, s * 0.54, s * 0.26, s * 0.32))
        p.drawRect(QRectF(s * 0.44, s * 0.54, s * 0.44, s * 0.32))

    elif kind in ("all_media", "all_photos_videos"):
        p.setPen(QPen(QColor("#356B9E"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.16, s * 0.80, s * 0.68), 2.0, 2.0)
        grad_sky = QLinearGradient(0, s * 0.22, 0, s * 0.55)
        grad_sky.setColorAt(0.0, QColor("#68A6E2"))
        grad_sky.setColorAt(1.0, QColor("#CAE4FB"))
        p.setPen(Qt.NoPen)
        p.setBrush(grad_sky)
        p.drawRect(QRectF(s * 0.16, s * 0.22, s * 0.68, s * 0.36))
        p.setBrush(QColor("#54A045"))
        p.drawChord(QRectF(s * 0.10, s * 0.44, s * 0.80, s * 0.44), 0, 180 * 16)
        p.setBrush(QColor("#245380"))
        p.drawRoundedRect(QRectF(s * 0.08, s * 0.12, s * 0.84, s * 0.18), 1.5, 1.5)
        p.setBrush(QColor("#FFFFFF"))
        for sp_i in range(4):
            p.drawRect(QRectF(s * (0.14 + sp_i * 0.20), s * 0.15, s * 0.11, s * 0.10))

    elif kind in ("pictures_library", "picture_lib"):
        p.setPen(QPen(QColor("#2B689C"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.16, s * 0.80, s * 0.68), 2.0, 2.0)
        grad_sky = QLinearGradient(0, s * 0.22, 0, s * 0.55)
        grad_sky.setColorAt(0.0, QColor("#5DA2E8"))
        grad_sky.setColorAt(1.0, QColor("#D4E8FC"))
        p.setPen(Qt.NoPen)
        p.setBrush(grad_sky)
        p.drawRect(QRectF(s * 0.16, s * 0.22, s * 0.68, s * 0.40))
        p.setBrush(QColor("#F5C342"))
        p.drawEllipse(QPointF(s * 0.32, s * 0.34), s * 0.10, s * 0.10)
        p.setBrush(QColor("#4EA03C"))
        p.drawChord(QRectF(s * 0.16, s * 0.42, s * 0.68, s * 0.44), 0, 180 * 16)

    elif kind in ("videos_library", "video_lib"):
        p.setPen(QPen(QColor("#1F4B78"), 1.0))
        p.setBrush(QColor("#356B9E"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.76, s * 0.68), 2.0, 2.0)
        p.setBrush(QColor("#FFFFFF"))
        for sp_i in range(3):
            p.drawRect(QRectF(s * 0.16, s * (0.22 + sp_i * 0.20), s * 0.12, s * 0.12))
            p.drawRect(QRectF(s * 0.72, s * (0.22 + sp_i * 0.20), s * 0.12, s * 0.12))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#F6FAFE"))
        tri = QPolygonF([
            QPointF(s * 0.40, s * 0.36),
            QPointF(s * 0.62, s * 0.50),
            QPointF(s * 0.40, s * 0.64),
        ])
        p.drawPolygon(tri)

    elif kind in ("pane_left", "pane_right", "tag_caption_pane"):
        # Window frame with blue titlebar and right panel matching WLPG
        p.setPen(QPen(QColor("#3A6A9C"), 1.1))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.14, s * 0.80, s * 0.72), 2.0, 2.0)
        # Blue title bar
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2468A8"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.14, s * 0.80, s * 0.18), 2.0, 2.0)
        p.drawRect(QRectF(s * 0.10, s * 0.24, s * 0.80, s * 0.08))
        if kind == "pane_left":
            p.setBrush(QColor("#E1EFFC"))
            p.drawRect(QRectF(s * 0.10, s * 0.32, s * 0.32, s * 0.54))
            p.setPen(QPen(QColor("#8FBEE8"), 1.0))
            p.drawLine(QPointF(s * 0.42, s * 0.32), QPointF(s * 0.42, s * 0.86))
        else:
            # Right tag panel
            p.setBrush(QColor("#E1EFFC"))
            p.drawRect(QRectF(s * 0.58, s * 0.32, s * 0.32, s * 0.54))
            p.setPen(QPen(QColor("#8FBEE8"), 1.0))
            p.drawLine(QPointF(s * 0.58, s * 0.32), QPointF(s * 0.58, s * 0.86))
            # Blue Person silhouette badge overlay
            p.setPen(QPen(QColor("#185084"), 1.0))
            grad = QLinearGradient(0, s * 0.38, 0, s * 0.82)
            grad.setColorAt(0.0, QColor("#5CA4E2"))
            grad.setColorAt(1.0, QColor("#1F65A8"))
            p.setBrush(grad)
            p.drawEllipse(QPointF(s * 0.38, s * 0.44), s * 0.13, s * 0.13)
            body = QPainterPath()
            body.moveTo(s * 0.18, s * 0.84)
            body.quadTo(s * 0.20, s * 0.60, s * 0.38, s * 0.60)
            body.quadTo(s * 0.56, s * 0.60, s * 0.58, s * 0.84)
            body.closeSubpath()
            p.drawPath(body)

    elif kind == "auto_adjust":
        p.setPen(QPen(QColor("#7088A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.76, s * 0.68), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#6CB2E8"))
        p.drawRect(QRectF(s * 0.18, s * 0.22, s * 0.64, s * 0.56))
        p.setBrush(QColor("#48A038"))
        p.drawChord(QRectF(s * 0.16, s * 0.44, s * 0.68, s * 0.40), 0, 180 * 16)
        p.setBrush(QColor("#F9C846"))
        p.setPen(QPen(QColor("#D48817"), 0.8))
        cx, cy = s * 0.66, s * 0.34
        pts = [
            QPointF(cx, cy - s * 0.14), QPointF(cx + s * 0.04, cy - s * 0.04),
            QPointF(cx + s * 0.14, cy), QPointF(cx + s * 0.04, cy + s * 0.04),
            QPointF(cx, cy + s * 0.14), QPointF(cx - s * 0.04, cy + s * 0.04),
            QPointF(cx - s * 0.14, cy), QPointF(cx - s * 0.04, cy - s * 0.04),
        ]
        p.drawPolygon(QPolygonF(pts))

    elif kind == "straighten":
        p.save()
        p.translate(s * 0.5, s * 0.5)
        p.rotate(-8)
        p.translate(-s * 0.5, -s * 0.5)
        p.setPen(QPen(QColor("#7088A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.16, s * 0.18, s * 0.68, s * 0.60), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#6CB2E8"))
        p.drawRect(QRectF(s * 0.22, s * 0.24, s * 0.56, s * 0.48))
        p.setBrush(QColor("#48A038"))
        p.drawChord(QRectF(s * 0.20, s * 0.44, s * 0.60, s * 0.36), 0, 180 * 16)
        p.restore()
        p.setPen(QPen(QColor("#0078D7"), 1.6, Qt.SolidLine, Qt.RoundCap))
        path = QPainterPath()
        path.arcMoveTo(QRectF(s * 0.06, s * 0.06, s * 0.88, s * 0.88), 120)
        path.arcTo(QRectF(s * 0.06, s * 0.06, s * 0.88, s * 0.88), 120, -50)
        p.drawPath(path)

    elif kind == "noise_reduction":
        p.setPen(QPen(QColor("#7088A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.76, s * 0.68), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#6EB4F0"))
        p.drawRect(QRectF(s * 0.18, s * 0.22, s * 0.64, s * 0.56))
        p.setBrush(QColor("#2C6BA8"))
        p.drawPolygon(QPolygonF([QPointF(s * 0.18, s * 0.78), QPointF(s * 0.42, s * 0.42), QPointF(s * 0.66, s * 0.78)]))
        p.setBrush(QColor("#4A88C4"))
        p.drawPolygon(QPolygonF([QPointF(s * 0.46, s * 0.78), QPointF(s * 0.68, s * 0.48), QPointF(s * 0.82, s * 0.78)]))

    elif kind in ("adjust_color", "color_adjust", "color"):
        cx, cy = s * 0.5, s * 0.5
        r = s * 0.34
        colors = [QColor("#E81123"), QColor("#FFB900"), QColor("#107C41"), QColor("#0078D7")]
        for i, c in enumerate(colors):
            p.setPen(Qt.NoPen)
            p.setBrush(c)
            p.drawPie(QRectF(cx - r, cy - r, r * 2, r * 2), i * 90 * 16, 90 * 16)
        p.setPen(QPen(QColor("#404040"), 1.0))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)

    elif kind in ("color_spectrum", "rainbow_spectrum"):
        grad = QLinearGradient(s * 0.12, 0, s * 0.88, 0)
        grad.setColorAt(0.0, QColor("#E81123"))
        grad.setColorAt(0.2, QColor("#FF8C00"))
        grad.setColorAt(0.4, QColor("#FFD700"))
        grad.setColorAt(0.6, QColor("#107C41"))
        grad.setColorAt(0.8, QColor("#0078D7"))
        grad.setColorAt(1.0, QColor("#881798"))
        p.setPen(QPen(QColor("#708090"), 1.0))
        p.setBrush(grad)
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.12, s * 0.76, s * 0.76), 2.0, 2.0)

    elif kind in ("adjust_exposure", "exposure_adjust", "exposure"):
        cx, cy = s * 0.5, s * 0.5
        r = s * 0.34
        p.setPen(QPen(QColor("#242424"), 1.4))
        p.setBrush(QColor("#FFFFFF"))
        p.drawPie(QRectF(cx - r, cy - r, r * 2, r * 2), 90 * 16, 180 * 16)
        p.setBrush(QColor("#242424"))
        p.drawPie(QRectF(cx - r, cy - r, r * 2, r * 2), 270 * 16, 180 * 16)

    elif kind == "red_eye":
        p.setPen(QPen(QColor("#2B5E8C"), 1.5))
        p.setBrush(QColor("#FFFFFF"))
        p.drawChord(QRectF(s * 0.12, s * 0.28, s * 0.76, s * 0.44), 0, 180 * 16)
        p.drawChord(QRectF(s * 0.12, s * 0.28, s * 0.76, s * 0.44), 180 * 16, 180 * 16)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#D82626"))
        p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.14, s * 0.14)
        p.setBrush(QColor("#222222"))
        p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.07, s * 0.07)

    elif kind in ("retouch", "spot_fix"):
        p.setPen(QPen(QColor("#B8860B"), 1.2))
        p.setBrush(QColor("#F5DEB3"))
        p.save()
        p.translate(s * 0.5, s * 0.5)
        p.rotate(45)
        p.drawRoundedRect(QRectF(-s * 0.34, -s * 0.14, s * 0.68, s * 0.28), 4.0, 4.0)
        p.setBrush(QColor("#DEB887"))
        p.drawRect(QRectF(-s * 0.10, -s * 0.14, s * 0.20, s * 0.28))
        p.restore()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#0078D7"))
        p.drawEllipse(QPointF(s * 0.76, s * 0.24), s * 0.06, s * 0.06)
        p.drawEllipse(QPointF(s * 0.24, s * 0.76), s * 0.05, s * 0.05)

    elif kind == "markup":
        p.save()
        p.setPen(Qt.NoPen)
        # Highlight stroke behind tip
        p.setBrush(QColor(255, 215, 0, 190))
        p.drawRoundedRect(QRectF(s * 0.14, s * 0.74, s * 0.72, s * 0.16), 2.5, 2.5)

        p.translate(s * 0.52, s * 0.44)
        p.rotate(-45)
        pen_grad = QLinearGradient(-s * 0.10, -s * 0.36, s * 0.10, -s * 0.36)
        pen_grad.setColorAt(0.0, QColor("#60A5FA"))
        pen_grad.setColorAt(0.5, QColor("#2563EB"))
        pen_grad.setColorAt(1.0, QColor("#1D4ED8"))
        p.setPen(QPen(QColor("#1E3A8A"), 0.8))
        p.setBrush(pen_grad)
        p.drawRoundedRect(QRectF(-s * 0.12, -s * 0.38, s * 0.24, s * 0.44), 2.0, 2.0)

        p.setBrush(QColor("#CBD5E1"))
        p.drawRect(QRectF(-s * 0.09, s * 0.06, s * 0.18, s * 0.10))

        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#FBBF24"))
        tip_poly = QPolygonF([
            QPointF(-s * 0.07, s * 0.16),
            QPointF(s * 0.07, s * 0.16),
            QPointF(s * 0.04, s * 0.28),
            QPointF(-s * 0.07, s * 0.23),
        ])
        p.drawPolygon(tip_poly)
        p.restore()

    elif kind == "pen":
        p.save()
        p.translate(s * 0.5, s * 0.5)
        p.rotate(-45)
        p.setPen(QPen(QColor("#1E3A8A"), 0.8))
        p.setBrush(QColor("#2563EB"))
        p.drawRoundedRect(QRectF(-s * 0.10, -s * 0.40, s * 0.20, s * 0.50), 2.0, 2.0)
        p.setBrush(QColor("#CBD5E1"))
        p.drawPolygon(QPolygonF([
            QPointF(-s * 0.10, s * 0.10),
            QPointF(s * 0.10, s * 0.10),
            QPointF(0, s * 0.38),
        ]))
        p.setBrush(QColor("#1E293B"))
        p.drawEllipse(QPointF(0, s * 0.22), s * 0.03, s * 0.03)
        p.restore()

    elif kind == "highlighter":
        p.save()
        p.translate(s * 0.5, s * 0.5)
        p.rotate(-45)
        p.setPen(QPen(QColor("#B45309"), 0.8))
        p.setBrush(QColor("#F59E0B"))
        p.drawRoundedRect(QRectF(-s * 0.14, -s * 0.38, s * 0.28, s * 0.46), 2.0, 2.0)
        p.setBrush(QColor("#FDE047"))
        p.drawPolygon(QPolygonF([
            QPointF(-s * 0.11, s * 0.08),
            QPointF(s * 0.11, s * 0.08),
            QPointF(s * 0.06, s * 0.32),
            QPointF(-s * 0.11, s * 0.24),
        ]))
        p.restore()

    elif kind == "arrow":
        p.setPen(QPen(QColor("#0078D7"), max(1.5, s * 0.10), Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.18, s * 0.78), QPointF(s * 0.72, s * 0.26))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#0078D7"))
        p.drawPolygon(QPolygonF([
            QPointF(s * 0.84, s * 0.16),
            QPointF(s * 0.54, s * 0.24),
            QPointF(s * 0.74, s * 0.46),
        ]))

    elif kind == "rect":
        p.setPen(QPen(QColor("#0078D7"), max(1.4, s * 0.08)))
        p.setBrush(QColor(0, 120, 215, 35))
        p.drawRoundedRect(QRectF(s * 0.16, s * 0.20, s * 0.68, s * 0.60), 2.5, 2.5)

    elif kind == "ellipse":
        p.setPen(QPen(QColor("#0078D7"), max(1.4, s * 0.08)))
        p.setBrush(QColor(0, 120, 215, 35))
        p.drawEllipse(QRectF(s * 0.16, s * 0.20, s * 0.68, s * 0.60))

    elif kind == "star":
        from src.ui.grid_view import _build_star_polygon
        star_p = _build_star_polygon(s * 0.5, s * 0.5, s * 0.38, s * 0.16)
        p.setPen(QPen(QColor("#0078D7"), max(1.4, s * 0.08)))
        p.setBrush(QColor(0, 120, 215, 35))
        p.drawPolygon(star_p)

    elif kind == "blur":
        # Professional digital imaging standard water droplet for soften/blur
        drop_path = QPainterPath()
        drop_path.moveTo(s * 0.5, s * 0.16)
        drop_path.cubicTo(s * 0.52, s * 0.26, s * 0.82, s * 0.52, s * 0.82, s * 0.68)
        drop_path.cubicTo(s * 0.82, s * 0.86, s * 0.68, s * 0.90, s * 0.5, s * 0.90)
        drop_path.cubicTo(s * 0.32, s * 0.90, s * 0.18, s * 0.86, s * 0.18, s * 0.68)
        drop_path.cubicTo(s * 0.18, s * 0.52, s * 0.48, s * 0.26, s * 0.5, s * 0.16)
        drop_path.closeSubpath()
        grad = QLinearGradient(s * 0.5, s * 0.16, s * 0.5, s * 0.90)
        grad.setColorAt(0.0, QColor("#60A5FA"))
        grad.setColorAt(1.0, QColor("#1D4ED8"))
        p.setPen(QPen(QColor("#1E40AF"), 1.0))
        p.setBrush(grad)
        p.drawPath(drop_path)
        # Specular glint
        glint_path = QPainterPath()
        glint_path.moveTo(s * 0.38, s * 0.42)
        glint_path.cubicTo(s * 0.32, s * 0.52, s * 0.30, s * 0.64, s * 0.34, s * 0.72)
        p.setPen(QPen(QColor(255, 255, 255, 180), 1.2, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        p.drawPath(glint_path)

    elif kind == "pixelate":
        p.setPen(QPen(QColor("#475569"), 0.8))
        half = s * 0.32
        for ix, c in [(s * 0.18, "#3B82F6"), (s * 0.50, "#E2E8F0")]:
            for iy, c2 in [(s * 0.18, c), (s * 0.50, "#94A3B8" if c == "#3B82F6" else "#3B82F6")]:
                p.setBrush(QColor(c2))
                p.drawRect(QRectF(ix, iy, half, half))

    elif kind == "magnify":
        p.setPen(QPen(QColor("#475569"), max(1.5, s * 0.10)))
        p.setBrush(QColor(59, 130, 246, 50))
        p.drawEllipse(QPointF(s * 0.42, s * 0.42), s * 0.26, s * 0.26)
        p.setPen(QPen(QColor("#334155"), max(2.0, s * 0.12), Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.62, s * 0.62), QPointF(s * 0.84, s * 0.84))

    elif kind == "text":
        p.setPen(QPen(QColor("#1E3A8A"), 1.0))
        p.setBrush(QColor("#1E3A8A"))
        f = QFont("Segoe UI", int(s * 0.68), QFont.Bold)
        p.setFont(f)
        p.drawText(QRectF(0, 0, s, s), Qt.AlignCenter, "T")

    elif kind == "swap":
        p.setPen(QPen(QColor("#2563EB"), 1.4, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        p.drawLine(QPointF(s * 0.22, s * 0.35), QPointF(s * 0.72, s * 0.35))
        p.drawLine(QPointF(s * 0.60, s * 0.22), QPointF(s * 0.74, s * 0.35))
        p.drawLine(QPointF(s * 0.78, s * 0.65), QPointF(s * 0.28, s * 0.65))
        p.drawLine(QPointF(s * 0.40, s * 0.78), QPointF(s * 0.26, s * 0.65))

    elif kind == "fine_tune":
        p.setPen(QPen(QColor("#5A7A9A"), 1.4, Qt.SolidLine, Qt.RoundCap))
        for x_f in (0.28, 0.50, 0.72):
            p.drawLine(QPointF(s * x_f, s * 0.20), QPointF(s * x_f, s * 0.80))
        p.setPen(QPen(QColor("#005A9E"), 1.0))
        p.setBrush(QColor("#0078D7"))
        p.drawRoundedRect(QRectF(s * 0.21, s * 0.32, s * 0.14, s * 0.10), 1.5, 1.5)
        p.drawRoundedRect(QRectF(s * 0.43, s * 0.56, s * 0.14, s * 0.10), 1.5, 1.5)
        p.drawRoundedRect(QRectF(s * 0.65, s * 0.38, s * 0.14, s * 0.10), 1.5, 1.5)

    elif kind == "depth_light":
        p.setPen(QPen(QColor("#0078D7"), 1.2))
        p.setBrush(QColor("#EAF4FD"))
        p.drawEllipse(QPointF(s * 0.44, s * 0.52), s * 0.32, s * 0.32)
        p.setPen(QPen(QColor("#FFA000"), 1.2))
        p.setBrush(QColor("#FFD54F"))
        p.drawEllipse(QPointF(s * 0.72, s * 0.26), s * 0.12, s * 0.12)
        p.setPen(QPen(QColor("#005A9E"), 1.5))
        p.setBrush(QColor("#29B6F6"))
        p.drawEllipse(QPointF(s * 0.38, s * 0.56), s * 0.16, s * 0.16)

    elif kind == "save":
        p.setPen(QPen(QColor("#2B579A"), 1.2))
        p.setBrush(QColor("#4172B8"))
        p.drawRoundedRect(QRectF(s * 0.18, s * 0.18, s * 0.64, s * 0.64), 2.0, 2.0)
        p.setBrush(QColor("#FFFFFF"))
        p.drawRect(QRectF(s * 0.28, s * 0.20, s * 0.44, s * 0.28))
        p.setBrush(QColor("#A0C4E8"))
        p.drawRect(QRectF(s * 0.30, s * 0.58, s * 0.40, s * 0.22))

    elif kind in ("revert_original", "revert"):
        p.setPen(QPen(QColor("#7088A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.14, s * 0.18, s * 0.72, s * 0.64), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#EAF2FA"))
        p.drawRect(QRectF(s * 0.20, s * 0.24, s * 0.60, s * 0.52))
        p.setPen(QPen(QColor("#0078D7"), 2.0, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        path = QPainterPath()
        path.arcMoveTo(QRectF(s * 0.24, s * 0.28, s * 0.52, s * 0.52), 40)
        path.arcTo(QRectF(s * 0.24, s * 0.28, s * 0.52, s * 0.52), 40, 160)
        p.drawPath(path)
        p.setBrush(QColor("#0078D7"))
        p.drawPolygon(QPolygonF([QPointF(s * 0.20, s * 0.48), QPointF(s * 0.38, s * 0.36), QPointF(s * 0.38, s * 0.60)]))

    elif kind == "view_file":
        p.setPen(QPen(QColor("#7088A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.76, s * 0.68), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#5EA8E0"))
        p.drawRect(QRectF(s * 0.18, s * 0.22, s * 0.64, s * 0.56))
        p.setBrush(QColor("#3BA030"))
        p.drawChord(QRectF(s * 0.16, s * 0.44, s * 0.68, s * 0.42), 0, 180 * 16)
        p.setBrush(QColor("#F9C846"))
        p.drawEllipse(QPointF(s * 0.66, s * 0.36), s * 0.08, s * 0.08)

    elif kind == "make_copy":
        p.setPen(QPen(QColor("#7088A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.26, s * 0.12, s * 0.62, s * 0.56), 2.0, 2.0)
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.26, s * 0.62, s * 0.56), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#107C41"))
        p.drawEllipse(QPointF(s * 0.72, s * 0.72), s * 0.18, s * 0.18)
        p.setPen(QPen(QColor("#FFFFFF"), 2.0, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.62, s * 0.72), QPointF(s * 0.82, s * 0.72))
        p.drawLine(QPointF(s * 0.72, s * 0.62), QPointF(s * 0.72, s * 0.82))

    elif kind in ("close_file", "close_viewer"):
        p.setPen(QPen(QColor("#C42B1C"), 2.6, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.24, s * 0.24), QPointF(s * 0.76, s * 0.76))
        p.drawLine(QPointF(s * 0.76, s * 0.24), QPointF(s * 0.24, s * 0.76))

    elif kind == "crop":
        p.setPen(QPen(QColor("#2A5E8A"), 1.8))
        p.drawLine(QPointF(s * 0.26, s * 0.12), QPointF(s * 0.26, s * 0.74))
        p.drawLine(QPointF(s * 0.26, s * 0.74), QPointF(s * 0.88, s * 0.74))
        p.drawLine(QPointF(s * 0.12, s * 0.26), QPointF(s * 0.74, s * 0.26))
        p.drawLine(QPointF(s * 0.74, s * 0.26), QPointF(s * 0.74, s * 0.88))

    elif kind in ("options", "settings", "gear"):
        p.setPen(QPen(QColor("#244260"), 1.2))
        p.setBrush(QColor("#627D98"))
        center = QPointF(s * 0.5, s * 0.5)
        outer_r = s * 0.38
        inner_r = s * 0.28
        hole_r = s * 0.14
        num_teeth = 8
        cog_path = QPainterPath()
        for i in range(num_teeth * 2):
            angle = i * (math.pi / num_teeth)
            r = outer_r if (i % 2 == 0) else inner_r
            pt = QPointF(center.x() + r * math.cos(angle), center.y() + r * math.sin(angle))
            if i == 0:
                cog_path.moveTo(pt)
            else:
                cog_path.lineTo(pt)
        cog_path.closeSubpath()
        hole = QPainterPath()
        hole.addEllipse(center, hole_r, hole_r)
        p.drawPath(cog_path.subtracted(hole))

    elif kind in ("nav_back", "nav_forward"):
        grad = QLinearGradient(0, 0, 0, s)
        grad.setColorAt(0.0, QColor("#44A5F5"))
        grad.setColorAt(1.0, QColor("#0A68C2"))
        p.setPen(QPen(QColor("#0B4A8C"), 1.0))
        p.setBrush(grad)
        p.drawEllipse(QRectF(s * 0.08, s * 0.08, s * 0.84, s * 0.84))
        p.setPen(QPen(QColor("#FFFFFF"), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if kind == "nav_back":
            p.drawLine(QPointF(s * 0.58, s * 0.28), QPointF(s * 0.38, s * 0.50))
            p.drawLine(QPointF(s * 0.38, s * 0.50), QPointF(s * 0.58, s * 0.72))
        else:
            p.drawLine(QPointF(s * 0.42, s * 0.28), QPointF(s * 0.62, s * 0.50))
            p.drawLine(QPointF(s * 0.62, s * 0.50), QPointF(s * 0.42, s * 0.72))
        p.end()

        # Disabled pixmap
        dis_pix = QPixmap(size, size)
        dis_pix.fill(Qt.transparent)
        dp = QPainter(dis_pix)
        dp.setRenderHint(QPainter.Antialiasing, True)
        d_grad = QLinearGradient(0, 0, 0, s)
        d_grad.setColorAt(0.0, QColor("#E8EFF7"))
        d_grad.setColorAt(1.0, QColor("#D0DDEB"))
        dp.setPen(QPen(QColor("#B8CADB"), 1.0))
        dp.setBrush(d_grad)
        dp.drawEllipse(QRectF(s * 0.08, s * 0.08, s * 0.84, s * 0.84))
        dp.setPen(QPen(QColor("#8FA5B8"), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if kind == "nav_back":
            dp.drawLine(QPointF(s * 0.58, s * 0.28), QPointF(s * 0.38, s * 0.50))
            dp.drawLine(QPointF(s * 0.38, s * 0.50), QPointF(s * 0.58, s * 0.72))
        else:
            dp.drawLine(QPointF(s * 0.42, s * 0.28), QPointF(s * 0.62, s * 0.50))
            dp.drawLine(QPointF(s * 0.62, s * 0.50), QPointF(s * 0.42, s * 0.72))
        dp.end()

        icon = QIcon(pix)
        icon.addPixmap(dis_pix, QIcon.Disabled, QIcon.Off)
        icon.addPixmap(dis_pix, QIcon.Disabled, QIcon.On)
        _ICON_CACHE[key] = icon
        return icon

    elif kind in ("undo", "redo"):
        p.setPen(QPen(QColor("#1D5A92"), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.setBrush(Qt.NoBrush)
        path = QPainterPath()
        if kind == "undo":
            path.arcMoveTo(QRectF(s * 0.22, s * 0.24, s * 0.60, s * 0.60), 40)
            path.arcTo(QRectF(s * 0.22, s * 0.24, s * 0.60, s * 0.60), 40, 160)
            p.drawPath(path)
            p.setBrush(QColor("#1D5A92"))
            tri = QPolygonF([QPointF(s * 0.16, s * 0.44), QPointF(s * 0.34, s * 0.30), QPointF(s * 0.34, s * 0.56)])
            p.drawPolygon(tri)
        else:
            path.arcMoveTo(QRectF(s * 0.18, s * 0.24, s * 0.60, s * 0.60), 140)
            path.arcTo(QRectF(s * 0.18, s * 0.24, s * 0.60, s * 0.60), 140, -160)
            p.drawPath(path)
            p.setBrush(QColor("#1D5A92"))
            tri = QPolygonF([QPointF(s * 0.84, s * 0.44), QPointF(s * 0.66, s * 0.30), QPointF(s * 0.66, s * 0.56)])
            p.drawPolygon(tri)
        p.end()

        dis_pix = QPixmap(size, size)
        dis_pix.fill(Qt.transparent)
        dp = QPainter(dis_pix)
        dp.setRenderHint(QPainter.Antialiasing, True)
        dp.setPen(QPen(QColor("#9EB4CA"), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        dp.setBrush(Qt.NoBrush)
        dpath = QPainterPath()
        if kind == "undo":
            dpath.arcMoveTo(QRectF(s * 0.22, s * 0.24, s * 0.60, s * 0.60), 40)
            dpath.arcTo(QRectF(s * 0.22, s * 0.24, s * 0.60, s * 0.60), 40, 160)
            dp.drawPath(dpath)
            dp.setBrush(QColor("#9EB4CA"))
            tri = QPolygonF([QPointF(s * 0.16, s * 0.44), QPointF(s * 0.34, s * 0.30), QPointF(s * 0.34, s * 0.56)])
            dp.drawPolygon(tri)
        else:
            dpath.arcMoveTo(QRectF(s * 0.18, s * 0.24, s * 0.60, s * 0.60), 140)
            dpath.arcTo(QRectF(s * 0.18, s * 0.24, s * 0.60, s * 0.60), 140, -160)
            dp.drawPath(dpath)
            dp.setBrush(QColor("#9EB4CA"))
            tri = QPolygonF([QPointF(s * 0.84, s * 0.44), QPointF(s * 0.66, s * 0.30), QPointF(s * 0.66, s * 0.56)])
            dp.drawPolygon(tri)
        dp.end()

        icon = QIcon(pix)
        icon.addPixmap(dis_pix, QIcon.Disabled, QIcon.Off)
        icon.addPixmap(dis_pix, QIcon.Disabled, QIcon.On)
        _ICON_CACHE[key] = icon
        return icon

    elif kind == "qat_menu":
        p.setPen(QPen(QColor("#2C4661"), 1.2, Qt.SolidLine, Qt.FlatCap))
        p.drawLine(QPointF(s * 0.15, s * 0.28), QPointF(s * 0.85, s * 0.28))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2C4661"))
        tri = QPolygonF([
            QPointF(s * 0.22, s * 0.44),
            QPointF(s * 0.78, s * 0.44),
            QPointF(s * 0.50, s * 0.74),
        ])
        p.drawPolygon(tri)

    elif kind == "select_all":
        p.setPen(QPen(QColor("#245888"), 1.2, Qt.DashLine))
        p.setBrush(QColor(210, 230, 252, 100))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.12, s * 0.76, s * 0.76), 2.0, 2.0)
        p.setPen(QPen(QColor("#167A16"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawLine(QPointF(s * 0.30, s * 0.52), QPointF(s * 0.44, s * 0.68))
        p.drawLine(QPointF(s * 0.44, s * 0.68), QPointF(s * 0.72, s * 0.32))

    elif kind == "cut":
        p.setPen(QPen(QColor("#45729F"), 1.2, Qt.SolidLine, Qt.RoundCap))
        p.drawEllipse(QPointF(s * 0.32, s * 0.72), s * 0.13, s * 0.13)
        p.drawEllipse(QPointF(s * 0.68, s * 0.72), s * 0.13, s * 0.13)
        p.drawLine(QPointF(s * 0.38, s * 0.62), QPointF(s * 0.70, s * 0.20))
        p.drawLine(QPointF(s * 0.62, s * 0.62), QPointF(s * 0.30, s * 0.20))
        p.setBrush(QColor("#2B7CD6"))
        p.drawEllipse(QPointF(s * 0.50, s * 0.43), s * 0.04, s * 0.04)

    elif kind == "copy":
        p.setPen(QPen(QColor("#45729F"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.28, s * 0.12, s * 0.56, s * 0.60), 1.5, 1.5)
        p.drawRoundedRect(QRectF(s * 0.16, s * 0.28, s * 0.56, s * 0.60), 1.5, 1.5)

    elif kind == "paste":
        p.setPen(QPen(QColor("#8A6828"), 1.0))
        p.setBrush(QColor("#C99A46"))
        p.drawRoundedRect(QRectF(s * 0.20, s * 0.20, s * 0.60, s * 0.70), 2.0, 2.0)
        p.setPen(QPen(QColor("#45729F"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.28, s * 0.32, s * 0.48, s * 0.52), 1.5, 1.5)
        p.setBrush(QColor("#C0C0C0"))
        p.drawRoundedRect(QRectF(s * 0.36, s * 0.14, s * 0.28, s * 0.14), 1.5, 1.5)

    elif kind == "help":
        p.setPen(QPen(QColor("#246BB2"), 1.0))
        p.setBrush(QColor("#2B7CD6"))
        p.drawEllipse(QRectF(s * 0.12, s * 0.12, s * 0.76, s * 0.76))
        p.setPen(QPen(QColor("#FFFFFF"), 2.0, Qt.SolidLine, Qt.RoundCap))
        p.setFont(QFont("Segoe UI", int(s * 0.52), QFont.Bold))
        p.drawText(QRectF(0, 0, s, s), Qt.AlignCenter, "?")

    elif kind in ("gallery", "wlpg_app", "photo_gallery"):
        # Authentic Windows Live Photo Gallery overlapping orange & cyan photo cards
        # Back photo (orange)
        p.setPen(QPen(QColor("#D96B14"), 1.0))
        p.setBrush(QColor("#F58220"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.12, s * 0.58, s * 0.58), 2.0, 2.0)
        p.setBrush(QColor("#FFFFFF"))
        p.drawRect(QRectF(s * 0.14, s * 0.16, s * 0.50, s * 0.40))
        p.setBrush(QColor("#E87010"))
        p.drawEllipse(QPointF(s * 0.46, s * 0.28), s * 0.08, s * 0.08)

        # Front photo (cyan/blue)
        p.setPen(QPen(QColor("#1A6FA8"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.32, s * 0.30, s * 0.58, s * 0.58), 2.0, 2.0)
        p.setBrush(QColor("#2D9BE8"))
        p.drawRect(QRectF(s * 0.36, s * 0.34, s * 0.50, s * 0.38))
        p.setBrush(QColor("#F5C538"))
        p.drawEllipse(QPointF(s * 0.48, s * 0.44), s * 0.07, s * 0.07)
        p.setBrush(QColor("#4EA632"))
        p.drawChord(QRectF(s * 0.36, s * 0.50, s * 0.50, s * 0.38), 0, 180 * 16)

    elif kind in ("duplicates", "find_duplicates"):
        # Twin overlapping photos with matching landscape elements and comparison badge
        # Back photo card
        p.setPen(QPen(QColor("#7A96B2"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.12, s * 0.56, s * 0.58), 2.0, 2.0)
        p.setBrush(QColor("#6BA3D6"))
        p.drawRect(QRectF(s * 0.14, s * 0.16, s * 0.48, s * 0.40))
        p.setBrush(QColor("#F5C538"))
        p.drawEllipse(QPointF(s * 0.46, s * 0.28), s * 0.06, s * 0.06)

        # Front photo card (identical twin with vibrant Aero blue border)
        p.setPen(QPen(QColor("#2468A8"), 1.1))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.34, s * 0.30, s * 0.56, s * 0.58), 2.0, 2.0)
        p.setBrush(QColor("#389BE8"))
        p.drawRect(QRectF(s * 0.38, s * 0.34, s * 0.48, s * 0.40))
        p.setBrush(QColor("#F5C538"))
        p.drawEllipse(QPointF(s * 0.70, s * 0.46), s * 0.06, s * 0.06)
        p.setBrush(QColor("#4EA632"))
        p.drawChord(QRectF(s * 0.38, s * 0.52, s * 0.48, s * 0.38), 0, 180 * 16)

        # Subtle twin match indicator badge in bottom-left
        p.setPen(QPen(QColor("#15803D"), 1.6, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.12, s * 0.76), QPointF(s * 0.28, s * 0.76))
        p.drawLine(QPointF(s * 0.12, s * 0.83), QPointF(s * 0.28, s * 0.83))


    elif kind in ("rename", "rename_file", "rename_photo"):
        # Document sheet
        doc_pen = QPen(QColor("#7A96B2"), 1.0)
        p.setPen(doc_pen)
        p.setBrush(QColor("#FFFFFF"))
        path = QPainterPath()
        path.moveTo(s * 0.16, s * 0.12)
        path.lineTo(s * 0.54, s * 0.12)
        path.lineTo(s * 0.70, s * 0.28)
        path.lineTo(s * 0.70, s * 0.86)
        path.lineTo(s * 0.16, s * 0.86)
        path.closeSubpath()
        p.drawPath(path)
        flap = QPainterPath()
        flap.moveTo(s * 0.54, s * 0.12)
        flap.lineTo(s * 0.54, s * 0.28)
        flap.lineTo(s * 0.70, s * 0.28)
        p.setBrush(QColor("#D6E6F5"))
        p.drawPath(flap)
        p.setPen(QPen(QColor("#A0B8D0"), 1.0))
        p.drawLine(QPointF(s * 0.24, s * 0.40), QPointF(s * 0.62, s * 0.40))
        p.drawLine(QPointF(s * 0.24, s * 0.54), QPointF(s * 0.52, s * 0.54))
        p.save()
        p.translate(s * 0.60, s * 0.65)
        p.rotate(-45)
        p.setPen(QPen(QColor("#B87700"), 1.0))
        p.setBrush(QColor("#F5B82E"))
        p.drawRect(QRectF(-s * 0.08, -s * 0.32, s * 0.16, s * 0.36))
        wood = QPainterPath()
        wood.moveTo(-s * 0.08, s * 0.04)
        wood.lineTo(s * 0.08, s * 0.04)
        wood.lineTo(0, s * 0.18)
        wood.closeSubpath()
        p.setPen(QPen(QColor("#C89B65"), 1.0))
        p.setBrush(QColor("#F5D7B5"))
        p.drawPath(wood)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2C2C2C"))
        ptip = QPainterPath()
        ptip.moveTo(-s * 0.03, s * 0.12)
        ptip.lineTo(s * 0.03, s * 0.12)
        ptip.lineTo(0, s * 0.18)
        ptip.closeSubpath()
        p.drawPath(ptip)
        p.restore()

    elif kind in ("resize", "resize_photo"):
        p.setPen(QPen(QColor("#246BB2"), 1.0))
        p.setBrush(QColor("#E8F3FD"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.12, s * 0.56, s * 0.56), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#F5C538"))
        p.drawEllipse(QPointF(s * 0.24, s * 0.26), s * 0.06, s * 0.06)
        p.setBrush(QColor("#4EA632"))
        mount = QPainterPath()
        mount.moveTo(s * 0.12, s * 0.66)
        mount.lineTo(s * 0.32, s * 0.40)
        mount.lineTo(s * 0.48, s * 0.56)
        mount.lineTo(s * 0.64, s * 0.38)
        mount.lineTo(s * 0.64, s * 0.66)
        mount.closeSubpath()
        p.drawPath(mount)
        arr_pen = QPen(QColor("#D9530F"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(arr_pen)
        p.drawLine(QPointF(s * 0.52, s * 0.52), QPointF(s * 0.88, s * 0.88))
        p.drawLine(QPointF(s * 0.88, s * 0.68), QPointF(s * 0.88, s * 0.88))
        p.drawLine(QPointF(s * 0.68, s * 0.88), QPointF(s * 0.88, s * 0.88))
        p.drawLine(QPointF(s * 0.52, s * 0.72), QPointF(s * 0.52, s * 0.52))
        p.drawLine(QPointF(s * 0.72, s * 0.52), QPointF(s * 0.52, s * 0.52))

    elif kind in ("rated_higher", "arrow_up"):
        p.setPen(QPen(QColor("#246BB2"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawLine(QPointF(s * 0.50, s * 0.22), QPointF(s * 0.50, s * 0.78))
        p.drawLine(QPointF(s * 0.24, s * 0.46), QPointF(s * 0.50, s * 0.20))
        p.drawLine(QPointF(s * 0.76, s * 0.46), QPointF(s * 0.50, s * 0.20))

    elif kind in ("rated_lower", "arrow_down"):
        p.setPen(QPen(QColor("#246BB2"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawLine(QPointF(s * 0.50, s * 0.22), QPointF(s * 0.50, s * 0.78))
        p.drawLine(QPointF(s * 0.24, s * 0.54), QPointF(s * 0.50, s * 0.80))
        p.drawLine(QPointF(s * 0.76, s * 0.54), QPointF(s * 0.50, s * 0.80))

    elif kind in ("rated_only", "equals"):
        p.setPen(QPen(QColor("#246BB2"), 2.0, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.20, s * 0.38), QPointF(s * 0.80, s * 0.38))
        p.drawLine(QPointF(s * 0.20, s * 0.62), QPointF(s * 0.80, s * 0.62))

    elif kind in ("boolean_and", "plus", "add"):
        p.setPen(QPen(QColor("#246BB2"), 2.2, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.50, s * 0.18), QPointF(s * 0.50, s * 0.82))
        p.drawLine(QPointF(s * 0.18, s * 0.50), QPointF(s * 0.82, s * 0.50))

    elif kind in ("minus", "subtract"):
        p.setPen(QPen(QColor("#C9302C"), 2.2, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.18, s * 0.50), QPointF(s * 0.82, s * 0.50))

    elif kind in ("eyedropper", "pipette", "color_picker"):
        p.save()
        p.translate(s * 0.5, s * 0.5)
        p.rotate(-45)
        # Bulb at top
        p.setPen(QPen(QColor("#244E78"), 1.1))
        p.setBrush(QColor("#357EBD"))
        p.drawRoundedRect(QRectF(-s * 0.14, -s * 0.44, s * 0.28, s * 0.22), 2.0, 2.0)
        # Pipette glass tube
        p.setBrush(QColor(240, 246, 255, 220))
        p.drawRect(QRectF(-s * 0.08, -s * 0.22, s * 0.16, s * 0.38))
        # Color fluid inside glass
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#E81123"))
        p.drawRect(QRectF(-s * 0.06, -s * 0.06, s * 0.12, s * 0.22))
        # Pipette tapered tip
        p.setPen(QPen(QColor("#244E78"), 1.0))
        p.setBrush(QColor(240, 246, 255, 220))
        tip = QPolygonF([
            QPointF(-s * 0.08, s * 0.16),
            QPointF(s * 0.08, s * 0.16),
            QPointF(s * 0.03, s * 0.38),
            QPointF(-s * 0.03, s * 0.38),
        ])
        p.drawPolygon(tip)
        p.restore()
        # Liquid droplet at tip
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#0078D7"))
        p.drawEllipse(QPointF(s * 0.18, s * 0.80), s * 0.07, s * 0.07)

    elif kind in ("brush", "paint_brush"):
        p.save()
        p.translate(s * 0.5, s * 0.5)
        p.rotate(-45)
        # Handle
        p.setPen(QPen(QColor("#8C5A20"), 1.0))
        p.setBrush(QColor("#C68430"))
        p.drawRoundedRect(QRectF(-s * 0.06, -s * 0.44, s * 0.12, s * 0.42), 1.0, 1.0)
        # Ferrule
        p.setPen(QPen(QColor("#6A7885"), 1.0))
        p.setBrush(QColor("#CCD5DE"))
        p.drawRect(QRectF(-s * 0.09, -s * 0.02, s * 0.18, s * 0.16))
        # Bristles
        p.setPen(QPen(QColor("#1A5890"), 1.0))
        p.setBrush(QColor("#2B8EE8"))
        bristles = QPolygonF([
            QPointF(-s * 0.09, s * 0.14),
            QPointF(s * 0.09, s * 0.14),
            QPointF(s * 0.05, s * 0.38),
            QPointF(-s * 0.05, s * 0.38),
        ])
        p.drawPolygon(bristles)
        p.restore()

    elif kind in ("eraser",):
        p.save()
        p.translate(s * 0.5, s * 0.5)
        p.rotate(-35)
        p.setPen(QPen(QColor("#B24040"), 1.0))
        p.setBrush(QColor("#FF7A7A"))
        p.drawRoundedRect(QRectF(-s * 0.16, -s * 0.36, s * 0.32, s * 0.40), 2.0, 2.0)
        p.setPen(QPen(QColor("#6A7A8A"), 1.0))
        p.setBrush(QColor("#E8EDF2"))
        p.drawRoundedRect(QRectF(-s * 0.16, s * 0.04, s * 0.32, s * 0.32), 2.0, 2.0)
        p.restore()

    elif kind in ("compare", "split_view", "before_after"):
        p.setPen(QPen(QColor("#35618C"), 1.1))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.16, s * 0.80, s * 0.68), 2.0, 2.0)
        p.setClipRect(QRectF(s * 0.10, s * 0.16, s * 0.40, s * 0.68))
        p.setBrush(QColor("#A0AAB5"))
        p.drawRect(QRectF(s * 0.10, s * 0.16, s * 0.40, s * 0.68))
        p.setClipping(False)
        p.setClipRect(QRectF(s * 0.50, s * 0.16, s * 0.40, s * 0.68))
        p.setBrush(QColor("#3A96E8"))
        p.drawRect(QRectF(s * 0.50, s * 0.16, s * 0.40, s * 0.68))
        p.setClipping(False)
        p.setPen(QPen(QColor("#1A3A60"), 1.5))
        p.drawLine(QPointF(s * 0.50, s * 0.12), QPointF(s * 0.50, s * 0.88))

    elif kind in ("boolean_or", "slash"):
        p.setPen(QPen(QColor("#246BB2"), 2.0, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.25, s * 0.80), QPointF(s * 0.75, s * 0.20))

    elif kind == "print":
        # Classic Aero Printer icon
        p.setPen(QPen(QColor("#406892"), 1.0))
        # Top paper feeder sheet
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.28, s * 0.12, s * 0.44, s * 0.35), 1.0, 1.0)
        # Printer body
        p.setBrush(QColor("#E8F1FC"))
        p.drawRoundedRect(QRectF(s * 0.14, s * 0.36, s * 0.72, s * 0.42), 2.5, 2.5)
        # Output slot
        p.setBrush(QColor("#244268"))
        p.drawRect(QRectF(s * 0.24, s * 0.60, s * 0.52, s * 0.12))
        # Paper coming out
        p.setBrush(QColor("#FFFFFF"))
        p.drawRect(QRectF(s * 0.28, s * 0.64, s * 0.44, s * 0.24))
        # Blue status LED
        p.setBrush(QColor("#3399FF"))
        p.drawEllipse(QRectF(s * 0.72, s * 0.44, s * 0.08, s * 0.08))

    elif kind in ("panoramic", "panorama"):
        # Panoramic Wide-Angle 3-Panel Ribbon Icon
        p.setPen(QPen(QColor("#24568C"), 1.0))
        p.setBrush(QColor("#CCE5FF"))
        p.drawRoundedRect(QRectF(s * 0.06, s * 0.25, s * 0.88, s * 0.50), 2.0, 2.0)
        p.setPen(QPen(QColor("#3A7CBD"), 1.0, Qt.DashLine))
        p.drawLine(QPointF(s * 0.35, s * 0.25), QPointF(s * 0.35, s * 0.75))
        p.drawLine(QPointF(s * 0.65, s * 0.25), QPointF(s * 0.65, s * 0.75))
        # Sun & mountains
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#FFAE19"))
        p.drawEllipse(QRectF(s * 0.72, s * 0.32, s * 0.14, s * 0.14))
        p.setBrush(QColor("#2A8F4A"))
        poly = [QPointF(s * 0.12, s * 0.75), QPointF(s * 0.38, s * 0.45), QPointF(s * 0.62, s * 0.75)]
        p.drawPolygon(poly)

    elif kind in ("collage", "auto_collage"):
        # Multi-frame collage icon
        p.setPen(QPen(QColor("#3C648F"), 1.0))
        p.setBrush(QColor("#D6E8FC"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.14, s * 0.36, s * 0.34), 1.5, 1.5)
        p.setBrush(QColor("#EAF4FE"))
        p.drawRoundedRect(QRectF(s * 0.52, s * 0.14, s * 0.38, s * 0.42), 1.5, 1.5)
        p.setBrush(QColor("#BEDBFA"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.54, s * 0.38, s * 0.34), 1.5, 1.5)
        p.setBrush(QColor("#D6E8FC"))
        p.drawRoundedRect(QRectF(s * 0.54, s * 0.62, s * 0.36, s * 0.26), 1.5, 1.5)

    elif kind in ("photo_fuse", "fuse"):
        # Dual-frame face fuse icon with sparkle / swap arrow
        p.setPen(QPen(QColor("#35618C"), 1.0))
        p.setBrush(QColor("#E2EEFC"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.18, s * 0.45, s * 0.58), 2.0, 2.0)
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.42, s * 0.26, s * 0.48, s * 0.60), 2.0, 2.0)
        # Face silhouette inside front card
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2572C4"))
        p.drawEllipse(QRectF(s * 0.58, s * 0.38, s * 0.16, s * 0.18))
        p.drawChord(QRectF(s * 0.50, s * 0.60, s * 0.32, s * 0.22), 0, 180 * 16)
        # Magic sparkle star
        p.setPen(QPen(QColor("#FF9900"), 1.5))
        p.drawLine(QPointF(s * 0.28, s * 0.10), QPointF(s * 0.28, s * 0.26))
        p.drawLine(QPointF(s * 0.20, s * 0.18), QPointF(s * 0.36, s * 0.18))

    elif kind == "play":
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#0078D4"))
        triangle = QPolygonF([
            QPointF(s * 0.25, s * 0.18),
            QPointF(s * 0.82, s * 0.50),
            QPointF(s * 0.25, s * 0.82),
        ])
        p.drawPolygon(triangle)

    elif kind == "pause":
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#0078D4"))
        p.drawRoundedRect(QRectF(s * 0.22, s * 0.20, s * 0.22, s * 0.60), 1.5, 1.5)
        p.drawRoundedRect(QRectF(s * 0.56, s * 0.20, s * 0.22, s * 0.60), 1.5, 1.5)

    elif kind in ("volume", "volume_low"):
        p.setPen(QPen(QColor("#0078D4"), 1.2))
        p.setBrush(QColor("#0078D4"))
        horn = QPolygonF([
            QPointF(s * 0.15, s * 0.35),
            QPointF(s * 0.32, s * 0.35),
            QPointF(s * 0.55, s * 0.15),
            QPointF(s * 0.55, s * 0.85),
            QPointF(s * 0.32, s * 0.65),
            QPointF(s * 0.15, s * 0.65),
        ])
        p.drawPolygon(horn)
        p.setBrush(Qt.NoBrush)
        p.drawArc(QRectF(s * 0.58, s * 0.32, s * 0.26, s * 0.36), -50 * 16, 100 * 16)
        if kind == "volume":
            p.drawArc(QRectF(s * 0.58, s * 0.20, s * 0.38, s * 0.60), -60 * 16, 120 * 16)

    elif kind == "volume_mute":
        p.setPen(QPen(QColor("#777777"), 1.2))
        p.setBrush(QColor("#777777"))
        horn = QPolygonF([
            QPointF(s * 0.15, s * 0.35),
            QPointF(s * 0.32, s * 0.35),
            QPointF(s * 0.55, s * 0.15),
            QPointF(s * 0.55, s * 0.85),
            QPointF(s * 0.32, s * 0.65),
            QPointF(s * 0.15, s * 0.65),
        ])
        p.drawPolygon(horn)
        # Red/grey diagonal mute slash
        p.setPen(QPen(QColor("#C0392B"), 1.5))
        p.drawLine(QPointF(s * 0.62, s * 0.35), QPointF(s * 0.85, s * 0.65))
        p.drawLine(QPointF(s * 0.85, s * 0.35), QPointF(s * 0.62, s * 0.65))

    elif kind in ("shuffle", "random", "reorder"):
        p.setPen(QPen(QColor("#2468A8"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        # Top branch crossing down
        p.drawLine(QPointF(s * 0.16, s * 0.32), QPointF(s * 0.38, s * 0.32))
        p.drawLine(QPointF(s * 0.38, s * 0.32), QPointF(s * 0.64, s * 0.68))
        p.drawLine(QPointF(s * 0.64, s * 0.68), QPointF(s * 0.82, s * 0.68))
        # Bottom branch crossing up
        p.drawLine(QPointF(s * 0.16, s * 0.68), QPointF(s * 0.38, s * 0.68))
        p.drawLine(QPointF(s * 0.38, s * 0.68), QPointF(s * 0.64, s * 0.32))
        p.drawLine(QPointF(s * 0.64, s * 0.32), QPointF(s * 0.82, s * 0.32))
        # Arrow heads
        p.setBrush(QColor("#2468A8"))
        top_arrow = QPolygonF([QPointF(s * 0.84, s * 0.32), QPointF(s * 0.66, s * 0.20), QPointF(s * 0.66, s * 0.44)])
        bot_arrow = QPolygonF([QPointF(s * 0.84, s * 0.68), QPointF(s * 0.66, s * 0.56), QPointF(s * 0.66, s * 0.80)])
        p.drawPolygon(top_arrow)
        p.drawPolygon(bot_arrow)

    else:
        p.setPen(QPen(QColor("#3E6B99"), 1.1))
        p.setBrush(QColor("#D8EAF9"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.18, s * 0.76, s * 0.64), 2.0, 2.0)

    p.end()
    icon = QIcon(pix)
    _ICON_CACHE[key] = icon
    return icon


icon_provider.set_aero_renderer(_render_aero_raw)


def make_wlpg_icon(kind: str, size: int = 20) -> QIcon:
    """Returns a High-DPI icon respecting the active icon theme (WLPG Aero, Fluent, Minimalist)."""
    return icon_provider.get_icon(kind, size)


def make_symbol_icon(kind: str, size: int = 14) -> QIcon:
    """Draws a crisp antialiased checkmark or cross icon into a QIcon without font glyph clipping."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    if kind in ("check", "confirm"):
        p.setPen(QPen(QColor("#2E7D32"), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawLine(QPointF(size * 0.20, size * 0.52), QPointF(size * 0.42, size * 0.74))
        p.drawLine(QPointF(size * 0.42, size * 0.74), QPointF(size * 0.82, size * 0.26))
    elif kind in ("cross", "delete", "reject", "cancel"):
        p.setPen(QPen(QColor("#C50F1F"), 2.0, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(size * 0.28, size * 0.28), QPointF(size * 0.72, size * 0.72))
        p.drawLine(QPointF(size * 0.72, size * 0.28), QPointF(size * 0.28, size * 0.72))
    p.end()
    return QIcon(pm)


def make_person_avatar_pixmap(
    name: str, blob: bytes | None, size: int = 38, circular: bool = False, framed: bool = False
) -> QPixmap:
    """Renders a face avatar pixmap from JPEG/PNG bytes (or a silhouette fallback)."""
    canvas = QPixmap(size, size)
    canvas.fill(Qt.transparent)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

    if framed:
        frame_rect = QRectF(1.0, 1.0, size - 2.0, size - 2.0)
        painter.setPen(QPen(QColor("#9EB6CE"), 1.0))
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawRoundedRect(frame_rect, 3.0, 3.0)

        inner_rect = QRectF(2.5, 2.5, size - 5.0, size - 5.0)
        clip = QPainterPath()
        clip.addRoundedRect(inner_rect, 2.0, 2.0)
        painter.setClipPath(clip)
    else:
        inner_rect = QRectF(0.0, 0.0, float(size), float(size))
        clip = QPainterPath()
        if circular:
            clip.addEllipse(inner_rect)
        else:
            clip.addRoundedRect(inner_rect, 2.0, 2.0)
        painter.setClipPath(clip)

    drew_img = False
    if blob:
        qimg = QImage.fromData(blob)
        if not qimg.isNull():
            scaled = qimg.scaled(
                int(inner_rect.width()),
                int(inner_rect.height()),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
            sx = max(0, (scaled.width() - int(inner_rect.width())) // 2)
            sy = max(0, (scaled.height() - int(inner_rect.height())) // 2)
            painter.drawImage(
                inner_rect.topLeft(),
                scaled,
                QRectF(sx, sy, inner_rect.width(), inner_rect.height()),
            )
            drew_img = True

    if not drew_img:
        painter.fillRect(inner_rect, QColor("#DCEBFA"))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#5B9BD5"))
        s = float(size)
        painter.drawEllipse(QPointF(s * 0.5, s * 0.38), s * 0.17, s * 0.17)
        painter.drawEllipse(QPointF(s * 0.5, s * 0.82), s * 0.28, s * 0.24)

    painter.end()
    return canvas


@lru_cache(maxsize=8)
def make_default_variation_base_image(w: int = 54, h: int = 38) -> np.ndarray:
    """Generates the authentic WLPG poppy/flower reference image for adjustment menus and effect icons."""
    from PIL import Image as _PILImg, ImageDraw as _PILDProp, ImageFilter as _PILFilt
    # Supersample at 4x for smooth anti-aliased photographic rendering
    sw, sh = w * 4, h * 4
    canvas = _PILImg.new("RGB", (sw, sh), (0, 0, 0))
    draw = _PILDProp.Draw(canvas)

    # 1. Sky & Nature Bokeh Background
    for y in range(sh):
        ratio = y / sh
        if ratio < 0.45:
            r = int(140 + ratio * 80)
            g = int(185 + ratio * 55)
            b = int(235 - ratio * 40)
        elif ratio < 0.65:
            t = (ratio - 0.45) / 0.20
            r = int(176 - t * 80)
            g = int(210 - t * 60)
            b = int(217 - t * 130)
        else:
            t = (ratio - 0.65) / 0.35
            r = int(96 - t * 50)
            g = int(150 - t * 60)
            b = int(87 - t * 50)
        draw.line([(0, y), (sw, y)], fill=(r, g, b))

    # Soft bokeh light circles
    bokeh = _PILImg.new("RGBA", (sw, sh), (0, 0, 0, 0))
    bdraw = _PILDProp.Draw(bokeh)
    bokeh_spots = [
        (sw * 0.15, sh * 0.25, sw * 0.12, (255, 255, 255, 45)),
        (sw * 0.82, sh * 0.20, sw * 0.16, (240, 250, 255, 35)),
        (sw * 0.28, sh * 0.55, sw * 0.14, (200, 240, 160, 40)),
        (sw * 0.75, sh * 0.60, sw * 0.18, (180, 230, 140, 45)),
        (sw * 0.88, sh * 0.78, sw * 0.15, (130, 200, 100, 50)),
        (sw * 0.08, sh * 0.72, sw * 0.13, (150, 210, 110, 40)),
    ]
    for bx, by, br, bcol in bokeh_spots:
        bdraw.ellipse([bx - br, by - br, bx + br, by + br], fill=bcol)
    bokeh = bokeh.filter(_PILFilt.GaussianBlur(sw * 0.04))
    canvas.paste(bokeh, (0, 0), bokeh)

    # 2. Curved Stem & Leaf
    stem_layer = _PILImg.new("RGBA", (sw, sh), (0, 0, 0, 0))
    sdraw = _PILDProp.Draw(stem_layer)
    cx, cy = sw * 0.49, sh * 0.48
    stem_points = [
        (cx + sw * 0.02, cy + sh * 0.12),
        (cx + sw * 0.015, cy + sh * 0.24),
        (cx - sw * 0.01, cy + sh * 0.38),
        (cx - sw * 0.03, sh + 10)
    ]
    sdraw.line(stem_points, fill=(45, 115, 32, 255), width=int(sw * 0.032), joint="curve")
    sdraw.line(stem_points, fill=(70, 155, 45, 255), width=int(sw * 0.020), joint="curve")
    sdraw.line(stem_points, fill=(110, 190, 75, 180), width=int(sw * 0.008), joint="curve")

    leaf_poly = [
        (cx - sw * 0.01, cy + sh * 0.32),
        (cx - sw * 0.12, cy + sh * 0.28),
        (cx - sw * 0.20, cy + sh * 0.35),
        (cx - sw * 0.08, cy + sh * 0.40),
        (cx - sw * 0.015, cy + sh * 0.36),
    ]
    sdraw.polygon(leaf_poly, fill=(55, 130, 38, 255))
    sdraw.line([leaf_poly[0], leaf_poly[2]], fill=(95, 175, 60, 220), width=int(sw * 0.006))
    stem_layer = stem_layer.filter(_PILFilt.GaussianBlur(1))
    canvas.paste(stem_layer, (0, 0), stem_layer)

    # 3. Layered Poppy Petals
    flower_layer = _PILImg.new("RGBA", (sw, sh), (0, 0, 0, 0))
    fdraw = _PILDProp.Draw(flower_layer)
    pr = sw * 0.26

    back_petals = [
        (cx - pr * 0.7, cy - pr * 0.6, pr * 0.85, pr * 0.80, (195, 28, 32)),
        (cx + pr * 0.6, cy - pr * 0.65, pr * 0.90, pr * 0.85, (205, 32, 35)),
        (cx - pr * 0.85, cy + pr * 0.1, pr * 0.80, pr * 0.75, (185, 25, 30)),
        (cx + pr * 0.85, cy + pr * 0.1, pr * 0.85, pr * 0.78, (190, 26, 30)),
        (cx, cy - pr * 0.85, pr * 0.95, pr * 0.82, (215, 35, 38)),
    ]
    for px, py, rx, ry, col in back_petals:
        fdraw.ellipse([px - rx, py - ry, px + rx, py + ry], fill=col + (255,))

    mid_petals = [
        (cx - pr * 0.45, cy - pr * 0.35, pr * 0.88, pr * 0.82, (230, 42, 40)),
        (cx + pr * 0.45, cy - pr * 0.40, pr * 0.90, pr * 0.84, (242, 50, 45)),
        (cx - pr * 0.55, cy + pr * 0.35, pr * 0.86, pr * 0.80, (225, 38, 36)),
        (cx + pr * 0.55, cy + pr * 0.32, pr * 0.88, pr * 0.80, (238, 48, 42)),
        (cx - pr * 0.1, cy + pr * 0.55, pr * 0.92, pr * 0.78, (248, 58, 48)),
    ]
    for px, py, rx, ry, col in mid_petals:
        fdraw.ellipse([px - rx, py - ry, px + rx, py + ry], fill=col + (255,))

    fore_petals = [
        (cx - pr * 0.25, cy + pr * 0.25, pr * 0.72, pr * 0.62, (255, 78, 62)),
        (cx + pr * 0.30, cy + pr * 0.22, pr * 0.74, pr * 0.64, (255, 88, 70)),
        (cx + pr * 0.05, cy + pr * 0.42, pr * 0.78, pr * 0.58, (255, 98, 78)),
    ]
    for px, py, rx, ry, col in fore_petals:
        fdraw.ellipse([px - rx, py - ry, px + rx, py + ry], fill=col + (255,))

    # Radial shadow for velvety depth
    shadow = _PILImg.new("RGBA", (sw, sh), (0, 0, 0, 0))
    shdraw = _PILDProp.Draw(shadow)
    shdraw.ellipse([cx - pr * 0.42, cy - pr * 0.42, cx + pr * 0.42, cy + pr * 0.42], fill=(130, 10, 15, 140))
    shdraw.ellipse([cx - pr * 0.25, cy - pr * 0.25, cx + pr * 0.25, cy + pr * 0.25], fill=(70, 5, 10, 200))
    shadow = shadow.filter(_PILFilt.GaussianBlur(sw * 0.02))
    flower_layer.paste(shadow, (0, 0), shadow)

    # 4. Center Core, Stamens and Golden Pollen
    core_r = pr * 0.22
    fdraw.ellipse([cx - core_r, cy - core_r, cx + core_r, cy + core_r], fill=(24, 12, 14, 255))
    rng = np.random.RandomState(42)
    for _ in range(60):
        ang = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(core_r * 0.7, core_r * 1.55)
        sx = cx + math.cos(ang) * dist
        sy = cy + math.sin(ang) * dist
        fdraw.line([(cx + math.cos(ang) * core_r * 0.4, cy + math.sin(ang) * core_r * 0.4), (sx, sy)], fill=(50, 30, 20, 220), width=max(1, int(sw * 0.003)))
        pollen_r = rng.uniform(sw * 0.006, sw * 0.012)
        pollen_col = (255, rng.randint(210, 245), rng.randint(20, 60), 255)
        fdraw.ellipse([sx - pollen_r, sy - pollen_r, sx + pollen_r, sy + pollen_r], fill=pollen_col)

    # Seed capsule star
    cap_r = core_r * 0.48
    fdraw.ellipse([cx - cap_r, cy - cap_r, cx + cap_r, cy + cap_r], fill=(85, 115, 38, 255))
    for ang in np.linspace(0, 2 * math.pi, 8, endpoint=False):
        ex = cx + math.cos(ang) * cap_r
        ey = cy + math.sin(ang) * cap_r
        fdraw.line([(cx, cy), (ex, ey)], fill=(255, 230, 110, 240), width=max(1, int(sw * 0.004)))
    fdraw.ellipse([cx - sw * 0.008, cy - sw * 0.008, cx + sw * 0.008, cy + sw * 0.008], fill=(255, 240, 150, 255))

    # Dewdrop
    dx, dy, dr = cx - pr * 0.48, cy - pr * 0.15, sw * 0.018
    fdraw.ellipse([dx - dr, dy - dr * 0.75, dx + dr, dy + dr * 0.75], fill=(255, 255, 255, 180))
    fdraw.ellipse([dx - dr * 0.5, dy - dr * 0.4, dx, dy], fill=(255, 255, 255, 240))

    canvas.paste(flower_layer, (0, 0), flower_layer)
    res = canvas.resize((w, h), _PILImg.Resampling.LANCZOS)
    return np.array(res)


@lru_cache(maxsize=16)
def make_effect_preview_icon(effect_name: str) -> QIcon:
    """Generates an authentic photographic thumbnail preview matching WLPG Effects gallery."""
    img = make_default_variation_base_image(53, 64)
    h, w = img.shape[:2]
    rec = PhotoAdjustmentRecipe(effect=effect_name)
    rendered = render_adjusted_image(img, rec)
    qimg = QImage(rendered.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QIcon(QPixmap.fromImage(qimg))


_RATING_MENU_QSS = """
QMenu::item {
    padding: 3px 14px 3px 94px;
}
QMenu::icon {
    width: 85px;
    height: 16px;
    left: 4px;
}
"""


@lru_cache(maxsize=32)
def make_rating_stars_icon(rating: int, max_stars: int = 5, item_h: int = 16) -> QIcon:
    """Generates a crisp horizontal strip of authentic WLPG golden stars for menus."""
    slot_w = 17
    total_w = slot_w * max_stars
    pm = QPixmap(total_w, item_h)
    pm.fill(Qt.transparent)

    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)

    outer_r = item_h * 0.44
    inner_r = item_h * 0.19

    for s_idx in range(max_stars):
        cx = (s_idx + 0.5) * slot_w
        cy = item_h * 0.5
        is_filled = (s_idx < rating)

        pts: list[QPointF] = []
        for i in range(10):
            ang = math.radians(-90 + i * 36)
            r = outer_r if i % 2 == 0 else inner_r
            pts.append(QPointF(cx + r * math.cos(ang), cy + r * math.sin(ang)))

        poly = QPolygonF(pts)
        if is_filled:
            grad = QLinearGradient(0, cy - outer_r, 0, cy + outer_r)
            grad.setColorAt(0.0, QColor("#FFF4A3"))
            grad.setColorAt(0.4, QColor("#F8D038"))
            grad.setColorAt(1.0, QColor("#D99E10"))
            p.setPen(QPen(QColor("#B27B08"), 0.9))
            p.setBrush(grad)
            p.drawPolygon(poly)

            # Specular top highlight
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 120))
            p.drawPolygon(QPolygonF([pts[0], pts[1], QPointF(cx, cy)]))
            p.drawPolygon(QPolygonF([pts[8], pts[9], QPointF(cx, cy)]))
        else:
            p.setPen(QPen(QColor("#8C9BAE"), 1.0))
            p.setBrush(Qt.NoBrush)
            p.drawPolygon(poly)

    p.end()
    return QIcon(pm)


class _HoverableToolButton(QToolButton):
    hovered = Signal()
    unhovered = Signal()

    def enterEvent(self, event: Any) -> None:
        super().enterEvent(event)
        self.hovered.emit()

    def leaveEvent(self, event: Any) -> None:
        super().leaveEvent(event)
        self.unhovered.emit()


class _RibbonLargeToolButton(QToolButton):
    """
    Authentic Windows Live Photo Gallery Large Ribbon Tool Button.
    Paints a 32x32 icon tightly near top (y=4), followed immediately by 1-2 lines
    of text (y=38) with only 2px vertical spacing, avoiding Qt's default excessive gap.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("LargeRibbonBtn")
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedHeight(70)
        self._is_dark: bool = False
        self._has_menu: bool = False

    def paintEvent(self, event: Any) -> None:
        painter = QStylePainter(self)
        opt = QStyleOptionToolButton()
        self.initStyleOption(opt)

        # Clear text, icon, and menu indicator so style/QSS paints ONLY the hover/pressed/checked background frame
        opt.text = ""
        opt.icon = QIcon()
        opt.features &= ~QStyleOptionToolButton.ToolButtonFeature.HasMenu
        painter.drawComplexControl(QStyle.ComplexControl.CC_ToolButton, opt)

        r = self.rect()
        sz = self.iconSize()
        if sz.width() > 32 or sz.height() > 32:
            sz = QSize(32, 32)

        # In compact (icon-only) mode
        if self.toolButtonStyle() == Qt.ToolButtonIconOnly or not self.text():
            ix = (r.width() - sz.width()) // 2
            iy = (r.height() - sz.height()) // 2
            if not self.icon().isNull():
                mode = QIcon.Mode.Normal if self.isEnabled() else QIcon.Mode.Disabled
                if self.underMouse() and self.isEnabled():
                    mode = QIcon.Mode.Active
                state = QIcon.State.On if self.isChecked() else QIcon.State.Off
                self.icon().paint(painter, QRect(ix, iy, sz.width(), sz.height()), Qt.AlignmentFlag.AlignCenter, mode, state)
            return

        # 1. Paint Icon tightly at top (WLPG puts large 32x32 icons near top, y=4)
        ix = (r.width() - sz.width()) // 2
        iy = 4
        if not self.icon().isNull():
            mode = QIcon.Mode.Normal if self.isEnabled() else QIcon.Mode.Disabled
            if self.underMouse() and self.isEnabled():
                mode = QIcon.Mode.Active
            state = QIcon.State.On if self.isChecked() else QIcon.State.Off
            self.icon().paint(painter, QRect(ix, iy, sz.width(), sz.height()), Qt.AlignmentFlag.AlignCenter, mode, state)

        # 2. Text layout & color
        is_dark = getattr(self, "_is_dark", False)
        if is_dark:
            pen_color = QColor("#F1F5F9") if self.isEnabled() else QColor("#64748B")
        else:
            pen_color = QColor("#1E395B") if self.isEnabled() else QColor("#8E99A4")

        painter.setPen(pen_color)
        painter.setFont(self.font())

        has_menu = bool(
            self.menu() is not None
            or self.popupMode() in (QToolButton.ToolButtonPopupMode.MenuButtonPopup, QToolButton.ToolButtonPopupMode.InstantPopup)
            or getattr(self, "_has_menu", False)
        )

        fm = painter.fontMetrics()
        lines = self.text().split("\n")

        if has_menu and len(lines) == 1:
            # WLPG menu button: icon at y=4, text slightly higher (y=36)
            # giving comfortable, even spacing between the 32x32 icon and the dropdown arrow below
            ty = 36
            line_h = fm.height()
            text_rect = QRect(1, ty, r.width() - 2, line_h + 2)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self.text())
            arrow_y = 54
        elif len(lines) == 1:
            line_h = fm.height()
            ty = iy + sz.height() + 3
            text_rect = QRect(1, ty, r.width() - 2, line_h + 2)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self.text())
            arrow_y = ty + line_h + 1
        else:
            ty = iy + sz.height() + 1
            total_th = fm.lineSpacing() * len(lines)
            text_rect = QRect(1, ty, r.width() - 2, total_th + 4)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, self.text())
            arrow_y = ty + total_th + 1

        # 3. Draw single menu drop-down arrow directly beneath the label text
        if has_menu:
            arrow_x = (r.width() - 5) // 2
            painter.setPen(pen_color)
            painter.drawLine(arrow_x, arrow_y, arrow_x + 4, arrow_y)
            painter.drawLine(arrow_x + 1, arrow_y + 1, arrow_x + 3, arrow_y + 1)
            painter.drawPoint(arrow_x + 2, arrow_y + 2)


class _RibbonSplitToolButton(QToolButton):
    """
    Split ribbon tool button with distinct hover highlights:
    - Resting: transparent, NO divider line before the dropdown arrow.
    - Hovering either subcontrol:
      - Outer box / edge highlight wraps around both subcontrols.
      - Vertical divider line appears between main button and dropdown arrow.
      - Main highlight (active gradient) is ONLY over the subcontrol being hovered.
      - The other subcontrol maintains a soft subtle base fill.
    - Checked: checked style on main subcontrol.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CompactRibbonSplitBtn")
        self.setMouseTracking(True)
        self.setPopupMode(QToolButton.MenuButtonPopup)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)
        self._mouse_pos: QPoint = QPoint(-1, -1)
        self._is_dark: bool = False
        self._theme_mode: str = "aero"

    def enterEvent(self, event: Any) -> None:
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self._mouse_pos = QPoint(-1, -1)
        self.update()
        super().leaveEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        self._mouse_pos = event.pos()
        self.update()
        super().mouseMoveEvent(event)

    def sizeHint(self) -> QSize:
        sh = super().sizeHint()
        fm = self.fontMetrics()
        text_w = fm.horizontalAdvance(self.text()) if self.text() else 0
        icon_w = 16 if not self.icon().isNull() else 0
        spacing = 4 if (icon_w and text_w) else 0
        left_pad = 4
        right_pad = 6
        menu_w = 14
        calc_w = left_pad + icon_w + spacing + text_w + right_pad + menu_w
        return QSize(max(sh.width(), calc_w, 82), 22)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = self.width()
        h = self.height()
        r_menu_w = 13
        x_div = max(0, w - r_menu_w)

        is_hovered = self.underMouse() or (self._mouse_pos != QPoint(-1, -1) and self.rect().contains(self._mouse_pos))
        is_down = self.isDown()
        is_checked = self.isChecked()
        is_dark = getattr(self, "_is_dark", False)
        theme_mode = getattr(self, "_theme_mode", "dark" if is_dark else "aero")

        mpos = self._mouse_pos
        if is_hovered and mpos == QPoint(-1, -1):
            mpos = self.mapFromGlobal(QCursor.pos())

        hover_menu = is_hovered and (mpos.x() >= x_div)
        hover_btn = is_hovered and (mpos.x() < x_div)

        outer_rect = QRectF(0.5, 0.5, max(0.0, w - 1.0), max(0.0, h - 1.0))
        outer_path = QPainterPath()
        outer_path.addRoundedRect(outer_rect, 2.0, 2.0)

        # 1. Backgrounds and highlights
        if is_hovered:
            if theme_mode == "dark":
                base_color = QColor("#262F3D")
                border_color = QColor("#4B5563")
            elif theme_mode == "modern":
                base_color = QColor("#F3F4F6")
                border_color = QColor("#CBD5E1")
            else:
                base_color = QColor("#FFFDF5")
                border_color = QColor("#D9B152")

            painter.fillPath(outer_path, base_color)

            if hover_btn:
                clip_btn = QPainterPath()
                clip_btn.addRect(QRectF(0, 0, x_div, h))
                painter.save()
                painter.setClipPath(outer_path.intersected(clip_btn))

                if theme_mode == "dark":
                    if is_down:
                        b_grad = QLinearGradient(0, 0, 0, h)
                        b_grad.setColorAt(0, QColor("#1F2937"))
                        b_grad.setColorAt(1, QColor("#111827"))
                    else:
                        b_grad = QLinearGradient(0, 0, 0, h)
                        b_grad.setColorAt(0, QColor("#374151"))
                        b_grad.setColorAt(1, QColor("#1F2937"))
                elif theme_mode == "modern":
                    if is_down:
                        b_grad = QLinearGradient(0, 0, 0, h)
                        b_grad.setColorAt(0, QColor("#D1D5DB"))
                        b_grad.setColorAt(1, QColor("#9CA3AF"))
                    else:
                        b_grad = QLinearGradient(0, 0, 0, h)
                        b_grad.setColorAt(0, QColor("#E5E7EB"))
                        b_grad.setColorAt(1, QColor("#D1D5DB"))
                else:  # aero
                    b_grad = QLinearGradient(0, 0, 0, h)
                    if is_down:
                        b_grad.setColorAt(0, QColor("#FCE293"))
                        b_grad.setColorAt(1, QColor("#F7CE65"))
                    elif is_checked:
                        b_grad.setColorAt(0, QColor("#FDF4D7"))
                        b_grad.setColorAt(0.5, QColor("#FEE69C"))
                        b_grad.setColorAt(1, QColor("#FED768"))
                    else:
                        b_grad = QLinearGradient(0, 0, 0, h)
                        b_grad.setColorAt(0, QColor("#FFFDF5"))
                        b_grad.setColorAt(1, QColor("#FCE7A2"))

                painter.fillRect(QRectF(0, 0, x_div, h), b_grad)
                painter.restore()

            elif hover_menu:
                clip_menu = QPainterPath()
                clip_menu.addRect(QRectF(x_div, 0, r_menu_w, h))
                painter.save()
                painter.setClipPath(outer_path.intersected(clip_menu))

                if theme_mode == "dark":
                    m_grad = QLinearGradient(0, 0, 0, h)
                    m_grad.setColorAt(0, QColor("#374151"))
                    m_grad.setColorAt(1, QColor("#1F2937"))
                elif theme_mode == "modern":
                    m_grad = QLinearGradient(0, 0, 0, h)
                    m_grad.setColorAt(0, QColor("#E5E7EB"))
                    m_grad.setColorAt(1, QColor("#D1D5DB"))
                else:  # aero
                    m_grad = QLinearGradient(0, 0, 0, h)
                    m_grad.setColorAt(0, QColor("#FFFDF5"))
                    m_grad.setColorAt(1, QColor("#FCE7A2"))

                painter.fillRect(QRectF(x_div, 0, r_menu_w, h), m_grad)
                painter.restore()

                if is_checked:
                    clip_left = QPainterPath()
                    clip_left.addRect(QRectF(0, 0, x_div, h))
                    painter.save()
                    painter.setClipPath(outer_path.intersected(clip_left))
                    if theme_mode == "dark":
                        painter.fillRect(QRectF(0, 0, x_div, h), QColor("#1E3A8A"))
                    elif theme_mode == "modern":
                        painter.fillRect(QRectF(0, 0, x_div, h), QColor("#DBEAFE"))
                    else:
                        c_grad = QLinearGradient(0, 0, 0, h)
                        c_grad.setColorAt(0, QColor("#EAF3FD"))
                        c_grad.setColorAt(0.5, QColor("#D2E7FA"))
                        c_grad.setColorAt(1, QColor("#BBDDF7"))
                        painter.fillRect(QRectF(0, 0, x_div, h), c_grad)
                    painter.restore()

            # Outer boundary box wraps around both subcontrols
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(QPen(border_color, 1))
            painter.drawPath(outer_path)

            # Divider line appears ONLY on mouse over
            painter.setRenderHint(QPainter.Antialiasing, False)
            painter.drawLine(x_div, 1, x_div, h - 2)
            painter.setRenderHint(QPainter.Antialiasing, True)

        elif is_checked:
            btn_rect = QRectF(0.5, 0.5, max(0.0, x_div - 0.5), max(0.0, h - 1.0))
            btn_path = QPainterPath()
            btn_path.addRoundedRect(btn_rect, 2.0, 2.0)
            if theme_mode == "dark":
                painter.fillPath(btn_path, QColor("#1E3A8A"))
                painter.setPen(QPen(QColor("#60A5FA"), 1))
            elif theme_mode == "modern":
                painter.fillPath(btn_path, QColor("#DBEAFE"))
                painter.setPen(QPen(QColor("#3B82F6"), 1))
            else:
                c_grad = QLinearGradient(0, 0, 0, h)
                c_grad.setColorAt(0, QColor("#EAF3FD"))
                c_grad.setColorAt(0.5, QColor("#D2E7FA"))
                c_grad.setColorAt(1, QColor("#BBDDF7"))
                painter.fillPath(btn_path, c_grad)
                painter.setPen(QPen(QColor("#7DA2CE"), 1))
            painter.drawPath(btn_path)

        # 2. Paint Icon
        if not self.icon().isNull():
            ix = 4
            iy = (h - 16) // 2
            mode = QIcon.Mode.Normal if self.isEnabled() else QIcon.Mode.Disabled
            if is_hovered and self.isEnabled():
                mode = QIcon.Mode.Active
            state = QIcon.State.On if is_checked else QIcon.State.Off
            self.icon().paint(painter, QRect(ix, iy, 16, 16), Qt.AlignmentFlag.AlignCenter, mode, state)

        # 3. Paint Text
        if theme_mode == "dark":
            text_color = QColor("#F1F5F9") if self.isEnabled() else QColor("#64748B")
        else:
            text_color = QColor("#1E395B") if self.isEnabled() else QColor("#8E99A4")

        if self.toolButtonStyle() != Qt.ToolButtonIconOnly and self.text():
            tx = 24 if not self.icon().isNull() else 6
            tw = max(0, x_div - tx - 2)
            painter.setPen(text_color)
            painter.setFont(self.font())
            painter.drawText(QRect(tx, 0, tw, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())

        # 4. Paint Dropdown Arrow (identically matching Orientation and Media type)
        svg_file = "arrow_dropdown_mini_dark.svg" if theme_mode == "dark" else "arrow_dropdown_mini.svg"
        svg_path = get_icons_dir() / svg_file
        ax = w - 11
        ay = (h - 4) // 2 + 1
        if svg_path.exists():
            arrow_icon = QIcon(str(svg_path))
            arrow_icon.paint(painter, QRect(ax, ay, 7, 4))
        else:
            arrow_color = QColor("#F1F5F9") if theme_mode == "dark" else (QColor("#374151") if theme_mode == "modern" else QColor("#1E395B"))
            painter.setPen(Qt.NoPen)
            painter.setBrush(arrow_color)
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(ax, ay),
                        QPointF(ax + 7, ay),
                        QPointF(ax + 3.5, ay + 4),
                    ]
                )
            )

        painter.end()


class _VariationTile(QPushButton):
    hovered = Signal()
    unhovered = Signal()

    def __init__(self, is_center: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.is_center = is_center
        self.setFixedSize(58, 42)
        self.setCursor(Qt.PointingHandCursor)
        self._update_style(False)

    def _update_style(self, is_hover: bool) -> None:
        if is_hover:
            self.setStyleSheet(
                "QPushButton { border: 2px solid #3399FF; background-color: #CCE4F7; padding: 0px; border-radius: 2px; }"
            )
        elif self.is_center:
            self.setStyleSheet(
                "QPushButton { border: 2px solid #70A6E8; background-color: #F0F4FC; padding: 0px; border-radius: 2px; }"
            )
        else:
            self.setStyleSheet(
                "QPushButton { border: 1px solid #D5DFEB; background-color: #FFFFFF; padding: 1px; border-radius: 2px; }"
            )

    def enterEvent(self, event: Any) -> None:
        super().enterEvent(event)
        self._update_style(True)
        self.hovered.emit()

    def leaveEvent(self, event: Any) -> None:
        super().leaveEvent(event)
        self._update_style(False)
        self.unhovered.emit()


class _AutoAdjustTile(QToolButton):
    hovered = Signal()
    unhovered = Signal()

    def __init__(self, icon: QIcon, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(54, 54)
        self.setIcon(icon)
        self.setIconSize(QSize(38, 38))
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            """
            QToolButton {
                border: 1px solid #C4D5E8;
                border-radius: 2px;
                background-color: #F8FAFD;
            }
            QToolButton:hover {
                border: 1px solid #3399FF;
                background-color: #CCE4F7;
            }
            """
        )

    def enterEvent(self, event: Any) -> None:
        super().enterEvent(event)
        self.hovered.emit()

    def leaveEvent(self, event: Any) -> None:
        super().leaveEvent(event)
        self.unhovered.emit()


class _ExposureMenuWidget(QWidget):
    preview_requested = Signal(float, float)      # (brightness, contrast)
    auto_preview_requested = Signal()
    preview_cancelled = Signal()
    adjustment_selected = Signal(float, float)    # (brightness, contrast)
    auto_selected = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ExposureMenuWidget")
        self.setStyleSheet(
            """
            QWidget#ExposureMenuWidget {
                background-color: #FFFFFF;
                border: 1px solid #85B5F3;
            }
            QLabel {
                font-family: 'Segoe UI';
                font-size: 9pt;
                font-weight: 600;
                color: #1E395B;
            }
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        lbl_auto = QLabel("Auto adjust exposure", self)
        layout.addWidget(lbl_auto)

        self.btn_auto = _AutoAdjustTile(make_wlpg_icon("adjust_exposure", 36), self)
        self.btn_auto.hovered.connect(self.auto_preview_requested.emit)
        self.btn_auto.unhovered.connect(self.preview_cancelled.emit)
        self.btn_auto.clicked.connect(self.auto_selected.emit)
        layout.addWidget(self.btn_auto)

        sep = QFrame(self)
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Plain)
        sep.setStyleSheet("color: #DCE6F2; background-color: #DCE6F2; max-height: 1px;")
        layout.addWidget(sep)

        lbl_grid = QLabel("Choose exposure adjustment", self)
        layout.addWidget(lbl_grid)

        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setContentsMargins(0, 0, 0, 0)

        self.tiles: list[tuple[float, float, _VariationTile]] = []
        brightnesses = [-25.0, 0.0, 25.0]
        contrasts = [-20.0, 0.0, 20.0]

        for r_idx, b_val in enumerate(brightnesses):
            for c_idx, c_val in enumerate(contrasts):
                is_center = (r_idx == 1 and c_idx == 1)
                tile = _VariationTile(is_center=is_center, parent=self)
                tile.setIconSize(QSize(54, 38))
                tile.hovered.connect(lambda b=b_val, c=c_val: self.preview_requested.emit(b, c))
                tile.unhovered.connect(self.preview_cancelled.emit)
                tile.clicked.connect(lambda _, b=b_val, c=c_val: self.adjustment_selected.emit(b, c))
                grid.addWidget(tile, r_idx, c_idx)
                self.tiles.append((b_val, c_val, tile))

        layout.addLayout(grid)
        self._thumbnails_initialized = False

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        if not getattr(self, "_thumbnails_initialized", False):
            self.refresh_thumbnails()

    def refresh_thumbnails(self, base_img: np.ndarray | None = None) -> None:
        self._thumbnails_initialized = True
        if base_img is None:
            base_img = make_default_variation_base_image(54, 38)
        else:
            base_img = cv2.resize(base_img, (54, 38), interpolation=cv2.INTER_AREA)

        for b_val, c_val, tile in self.tiles:
            rec = PhotoAdjustmentRecipe(brightness=b_val, contrast=c_val)
            rendered = render_adjusted_image(base_img, rec)
            h, w = rendered.shape[:2]
            qimg = QImage(rendered.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            tile.setIcon(QIcon(QPixmap.fromImage(qimg)))


class _ColorMenuWidget(QWidget):
    preview_requested = Signal(float, float)      # (temperature, tint)
    auto_preview_requested = Signal()
    preview_cancelled = Signal()
    adjustment_selected = Signal(float, float)    # (temperature, tint)
    auto_selected = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ColorMenuWidget")
        self.setStyleSheet(
            """
            QWidget#ColorMenuWidget {
                background-color: #FFFFFF;
                border: 1px solid #85B5F3;
            }
            QLabel {
                font-family: 'Segoe UI';
                font-size: 9pt;
                font-weight: 600;
                color: #1E395B;
            }
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        lbl_auto = QLabel("Auto adjust color", self)
        layout.addWidget(lbl_auto)

        self.btn_auto = _AutoAdjustTile(make_wlpg_icon("color_spectrum", 36), self)
        self.btn_auto.hovered.connect(self.auto_preview_requested.emit)
        self.btn_auto.unhovered.connect(self.preview_cancelled.emit)
        self.btn_auto.clicked.connect(self.auto_selected.emit)
        layout.addWidget(self.btn_auto)

        sep = QFrame(self)
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Plain)
        sep.setStyleSheet("color: #DCE6F2; background-color: #DCE6F2; max-height: 1px;")
        layout.addWidget(sep)

        lbl_grid = QLabel("Choose color adjustment", self)
        layout.addWidget(lbl_grid)

        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setContentsMargins(0, 0, 0, 0)

        self.tiles: list[tuple[float, float, _VariationTile]] = []
        tints = [-25.0, 0.0, 25.0]
        temps = [-25.0, 0.0, 25.0]

        for r_idx, tint_val in enumerate(tints):
            for c_idx, temp_val in enumerate(temps):
                is_center = (r_idx == 1 and c_idx == 1)
                tile = _VariationTile(is_center=is_center, parent=self)
                tile.setIconSize(QSize(54, 38))
                tile.hovered.connect(lambda t=temp_val, ti=tint_val: self.preview_requested.emit(t, ti))
                tile.unhovered.connect(self.preview_cancelled.emit)
                tile.clicked.connect(lambda _, t=temp_val, ti=tint_val: self.adjustment_selected.emit(t, ti))
                grid.addWidget(tile, r_idx, c_idx)
                self.tiles.append((temp_val, tint_val, tile))

        layout.addLayout(grid)
        self._thumbnails_initialized = False

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        if not getattr(self, "_thumbnails_initialized", False):
            self.refresh_thumbnails()

    def refresh_thumbnails(self, base_img: np.ndarray | None = None) -> None:
        self._thumbnails_initialized = True
        if base_img is None:
            base_img = make_default_variation_base_image(54, 38)
        else:
            base_img = cv2.resize(base_img, (54, 38), interpolation=cv2.INTER_AREA)

        for temp_val, tint_val, tile in self.tiles:
            rec = PhotoAdjustmentRecipe(temperature=temp_val, tint=tint_val)
            rendered = render_adjusted_image(base_img, rec)
            h, w = rendered.shape[:2]
            qimg = QImage(rendered.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            tile.setIcon(QIcon(QPixmap.fromImage(qimg)))


class _RibbonEffectsGallery(QFrame):
    """
    Authentic WLPG Ribbon Effects Gallery with responsive clipping,
    scroll up/down arrows, and a dropdown menu showing a clean vertical list of all effects.
    """

    effect_selected = Signal(str)
    effect_hovered = Signal(str)
    effect_unhovered = Signal()

    CARD_WIDTH = 55
    CARD_HEIGHT = 66
    CARD_STRIDE = 56

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("RibbonEffectsGallery")
        self.setStyleSheet(
            """
            QFrame#RibbonEffectsGallery {
                background: #FFFFFF;
                border: 1px solid #C4D5E8;
                border-radius: 3px;
            }
            QToolTip {
                background-color: #FFFFFF;
                color: #1E395B;
                border: 1px solid #7DA2CE;
                border-radius: 2px;
                padding: 3px 6px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 8.5pt;
            }
            """
        )
        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(1, 1, 1, 1)
        hbox.setSpacing(1)

        # Viewport widget that clips the cards horizontally to exact card boundaries
        self.viewport_widget = QWidget(self)
        self.viewport_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.viewport_widget.setFixedHeight(68)
        self.viewport_widget.setStyleSheet("background: transparent;")

        self.cards_holder = QWidget(self.viewport_widget)
        self.cards_holder.setStyleSheet("background: transparent;")
        self.cards_layout = QHBoxLayout(self.cards_holder)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(1)

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        self._effects_data = [
            ("none", "None", "None — Original photo with no effect"),
            ("classic_chrome", "Chrome", "Classic Chrome — Fujifilm documentary style"),
            ("warm_portrait", "Portrait", "Warm Portrait — Portra warm golden skin tones"),
            ("vivid_landscape", "Vivid", "Vivid Landscape — Punchy Velvia saturated colors"),
            ("matte_vintage", "Matte", "Matte Vintage — Lifted film blacks & retro fade"),
            ("fine_art_bw", "B&W Art", "Fine Art B&W — Deep Acros monochrome tonal scale"),
            ("cool_cinema", "Cinema", "Cool Cinema — Teal & orange cinematic grade"),
            ("sepia", "Sepia", "Sepia — Warm antique photographic tone"),
        ]

        self.buttons: dict[str, _HoverableToolButton] = {}
        self._icons_initialized = False

        card_btn_style = """
            QToolButton {
                border: 1px solid transparent;
                border-radius: 2px;
                background: transparent;
                color: #1E395B;
                padding: 0px;
                margin: 0px;
            }
            QToolButton:hover {
                background: #EBF3FD;
                border: 1px solid #70A6E8;
            }
            QToolButton:checked {
                background: #CCE4F7;
                border: 1.5px solid #3399FF;
            }
            QToolTip {
                background-color: #FFFFFF;
                color: #1E395B;
                border: 1px solid #7DA2CE;
                border-radius: 2px;
                padding: 3px 6px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 8.5pt;
            }
        """

        for eff_id, label, tip in self._effects_data:
            btn = _HoverableToolButton(self.cards_holder)
            btn.setProperty("class", "EffectCard")
            btn.setCheckable(True)
            if eff_id == "none":
                btn.setChecked(True)
            btn.setText(label)
            btn.setIconSize(QSize(53, 64))
            btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
            btn.setFixedSize(self.CARD_WIDTH, self.CARD_HEIGHT)
            btn.setToolTip(tip)
            btn.setStyleSheet(card_btn_style)
            btn.hovered.connect(lambda eff=eff_id: self.effect_hovered.emit(eff))
            btn.unhovered.connect(self.effect_unhovered.emit)
            btn.clicked.connect(lambda _, eff=eff_id: self.effect_selected.emit(eff))
            self.btn_group.addButton(btn)
            self.cards_layout.addWidget(btn)
            self.buttons[eff_id] = btn

        self.cards_layout.addStretch(1)

        # Defer loading effect preview icons to avoid startup lag
        QTimer.singleShot(250, self._ensure_icons_loaded)

        # Legacy effect aliases for backwards compatibility with existing tests
        if "cool_cinema" in self.buttons:
            self.buttons["cool"] = self.buttons["cool_cinema"]
        if "fine_art_bw" in self.buttons:
            self.buttons["bw"] = self.buttons["fine_art_bw"]
            self.buttons["bw_contrast"] = self.buttons["fine_art_bw"]
        if "warm_portrait" in self.buttons:
            self.buttons["warm"] = self.buttons["warm_portrait"]

        total_w = len(self._effects_data) * self.CARD_STRIDE
        self.cards_holder.setGeometry(0, 0, total_w, 68)

        hbox.addWidget(self.viewport_widget, 1)

        v_ctrl = QVBoxLayout()
        v_ctrl.setContentsMargins(0, 0, 0, 0)
        v_ctrl.setSpacing(1)

        self.btn_up = QPushButton("▲", self)
        self.btn_up.setToolTip("Previous effects")
        self.btn_up.clicked.connect(self._scroll_prev)

        self.btn_dn = QPushButton("▼", self)
        self.btn_dn.setToolTip("Next effects")
        self.btn_dn.clicked.connect(self._scroll_next)

        self.btn_more = QPushButton(self)
        self.btn_more.setToolTip("Show all effects")
        self.btn_more.clicked.connect(self._show_all_effects_menu)

        btn_style = """
            QPushButton {
                border: 1px solid #D2DFEE;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F6F9FD, stop:1 #E2ECF7);
                color: #55687C;
                font-size: 7pt;
                padding: 0;
            }
            QPushButton:hover {
                background: #CCE4F7;
                border-color: #7BAEDB;
            }
            QPushButton:disabled {
                color: #B0C4DE;
                background: #F4F6F9;
                border-color: #DCE6F2;
            }
        """

        for b in (self.btn_up, self.btn_dn, self.btn_more):
            b.setFixedSize(14, 19)
            b.setFlat(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(btn_style)

        self._update_more_icon(is_dark=False)

        v_ctrl.addWidget(self.btn_up)
        v_ctrl.addWidget(self.btn_dn)
        v_ctrl.addWidget(self.btn_more)
        hbox.addLayout(v_ctrl)

        self._scroll_index: int = 0
        self._visible_tiles: int = len(self._effects_data)
        self.set_visible_tiles(len(self._effects_data))

    def _update_more_icon(self, is_dark: bool) -> None:
        more_pix = QPixmap(10, 10)
        more_pix.fill(Qt.transparent)
        mp = QPainter(more_pix)
        mp.setRenderHint(QPainter.Antialiasing)
        col = QColor("#94A3B8") if is_dark else QColor("#55687C")
        mp.setPen(QPen(col, 1.2))
        mp.drawLine(1, 2, 9, 2)
        mp.setBrush(col)
        tri = QPolygonF([QPointF(2, 4.5), QPointF(8, 4.5), QPointF(5, 7.5)])
        mp.drawPolygon(tri)
        mp.end()
        self.btn_more.setIcon(QIcon(more_pix))
        self.btn_more.setIconSize(QSize(10, 10))

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self._ensure_icons_loaded()

    def set_dark_mode(self, is_dark: bool) -> None:
        self._is_dark = bool(is_dark)
        self._update_more_icon(is_dark)
        if is_dark:
            self.setStyleSheet(
                """
                QFrame#RibbonEffectsGallery {
                    background: #14171C;
                    border: 1px solid #333A45;
                    border-radius: 2px;
                }
                QToolTip {
                    background-color: #1E242C;
                    color: #E2E8F0;
                    border: 1px solid #3B82F6;
                    border-radius: 2px;
                    padding: 3px 6px;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 8.5pt;
                }
                """
            )
            btn_style = """
                QPushButton {
                    border: 1px solid #333A45;
                    background: #20242B;
                    color: #94A3B8;
                    font-size: 7pt;
                    padding: 0;
                }
                QPushButton:hover {
                    background: #2E3642;
                    border-color: #60A5FA;
                    color: #FFFFFF;
                }
                QPushButton:disabled {
                    color: #475569;
                    background: #181B20;
                    border-color: #262B33;
                }
            """
            card_btn_style = """
                QToolButton {
                    border: 1px solid transparent;
                    border-radius: 2px;
                    background: transparent;
                    color: #E2E8F0;
                    padding: 0px;
                    margin: 0px;
                }
                QToolButton:hover {
                    background: #2A323D;
                    border: 1px solid #60A5FA;
                }
                QToolButton:checked {
                    background: #1E3A5F;
                    border: 1.5px solid #3B82F6;
                }
            """
        else:
            self.setStyleSheet(
                """
                QFrame#RibbonEffectsGallery {
                    background: #FFFFFF;
                    border: 1px solid #ABBFD3;
                    border-radius: 2px;
                }
                QToolTip {
                    background-color: #FFFFFF;
                    color: #1E395B;
                    border: 1px solid #7DA2CE;
                    border-radius: 2px;
                    padding: 3px 6px;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 8.5pt;
                }
                """
            )
            btn_style = """
                QPushButton {
                    border: 1px solid #D2DFEE;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F6F9FD, stop:1 #E2ECF7);
                    color: #55687C;
                    font-size: 7pt;
                    padding: 0;
                }
                QPushButton:hover {
                    background: #CCE4F7;
                    border-color: #7BAEDB;
                }
                QPushButton:disabled {
                    color: #B0C4DE;
                    background: #F4F6F9;
                    border-color: #DCE6F2;
                }
            """
            card_btn_style = """
                QToolButton {
                    border: 1px solid transparent;
                    border-radius: 2px;
                    background: transparent;
                    color: #1E395B;
                    padding: 0px;
                    margin: 0px;
                }
                QToolButton:hover {
                    background: #EBF3FD;
                    border: 1px solid #70A6E8;
                }
                QToolButton:checked {
                    background: #CCE4F7;
                    border: 1.5px solid #3399FF;
                }
            """
        for b in (self.btn_up, self.btn_dn, self.btn_more):
            b.setStyleSheet(btn_style)
        for eff_id, btn in self.buttons.items():
            if eff_id in ("cool", "bw", "bw_contrast", "warm"):
                continue
            btn.setStyleSheet(card_btn_style)

    def _ensure_icons_loaded(self) -> None:
        if getattr(self, "_icons_initialized", False) or getattr(self, "_live_thumbnails", None):
            return
        self._icons_initialized = True
        for eff_id, btn in self.buttons.items():
            if eff_id in ("cool", "bw", "bw_contrast", "warm"):
                continue
            btn.setIcon(make_effect_preview_icon(eff_id))
            btn.setIconSize(QSize(53, 64))

    def set_visible_tiles(self, count: int) -> None:
        self._visible_tiles = max(1, min(len(self._effects_data), count))
        vw = self._visible_tiles * self.CARD_STRIDE
        total_w = vw + 18
        self.viewport_widget.setFixedSize(vw, 68)
        self.setFixedSize(total_w, 70)
        total_holder_w = len(self._effects_data) * self.CARD_STRIDE
        self.cards_holder.setGeometry(0, 0, total_holder_w, 68)
        self._update_scroll_buttons()

    def sizeHint(self) -> QSize:
        count = getattr(self, "_visible_tiles", len(self._effects_data))
        total_w = count * self.CARD_STRIDE + 18
        return QSize(total_w, 70)

    def minimumSizeHint(self) -> QSize:
        return QSize(self.CARD_STRIDE + 18, 70)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scroll_buttons()

    def _update_scroll_buttons(self) -> None:
        visible_count = getattr(self, "_visible_tiles", 6)
        max_idx = max(0, len(self._effects_data) - visible_count)
        self._scroll_index = max(0, min(self._scroll_index, max_idx))
        self.cards_holder.move(-self._scroll_index * self.CARD_STRIDE, 0)
        self.btn_up.setEnabled(self._scroll_index > 0)
        self.btn_dn.setEnabled(self._scroll_index < max_idx)

    def _scroll_prev(self) -> None:
        if self._scroll_index > 0:
            self._scroll_index -= 1
            self._update_scroll_buttons()

    def _scroll_next(self) -> None:
        visible_count = getattr(self, "_visible_tiles", 6)
        max_idx = max(0, len(self._effects_data) - visible_count)
        if self._scroll_index < max_idx:
            self._scroll_index += 1
            self._update_scroll_buttons()

    def _show_all_effects_menu(self) -> None:
        menu = QMenu(self)
        menu.setMinimumWidth(320)
        menu.setStyleSheet(
            """
            QMenu {
                background-color: #FFFFFF;
                border: 1px solid #70A6E8;
                border-radius: 3px;
                padding: 4px;
            }
            """
        )
        active_eff = "none"
        for eff_id, btn in self.buttons.items():
            if btn.isChecked() and eff_id in [e[0] for e in self._effects_data]:
                active_eff = eff_id
                break

        from PySide6.QtWidgets import QWidgetAction

        for eff_id, label, tip in self._effects_data:
            btn = self.buttons[eff_id]
            action = QWidgetAction(menu)
            w = QWidget(menu)
            w.setCursor(Qt.PointingHandCursor)
            w.setFixedHeight(62)
            w_layout = QHBoxLayout(w)
            w_layout.setContentsMargins(8, 3, 10, 3)
            w_layout.setSpacing(12)

            # Left: preview image (2x size: 84x54) with center-crop to avoid squishing
            lbl_icon = QLabel(w)
            lbl_icon.setFixedSize(84, 54)
            lbl_icon.setStyleSheet("border: 1px solid #CBD5E1; border-radius: 2px;")
            live_pix = getattr(self, "_live_thumbnails", {}).get(eff_id)
            if live_pix and not live_pix.isNull():
                scaled_pix = live_pix.scaled(
                    84, 54, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
                )
                cx = max(0, (scaled_pix.width() - 84) // 2)
                cy = max(0, (scaled_pix.height() - 54) // 2)
                final_pix = scaled_pix.copy(cx, cy, 84, 54)
            else:
                raw_icon_pix = btn.icon().pixmap(128, 128)
                scaled_pix = raw_icon_pix.scaled(
                    84, 54, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
                )
                cx = max(0, (scaled_pix.width() - 84) // 2)
                cy = max(0, (scaled_pix.height() - 54) // 2)
                final_pix = scaled_pix.copy(cx, cy, 84, 54)
            lbl_icon.setPixmap(final_pix)
            lbl_icon.setScaledContents(False)
            w_layout.addWidget(lbl_icon)

            # Right: Effect Name and Description
            v_text = QVBoxLayout()
            v_text.setContentsMargins(0, 0, 0, 0)
            v_text.setSpacing(2)

            lbl_name = QLabel(label, w)
            lbl_name.setFont(QFont("Segoe UI", 9.5, QFont.Weight.Bold))
            lbl_name.setStyleSheet("color: #0F172A;")
            v_text.addWidget(lbl_name)

            lbl_desc = QLabel(tip, w)
            lbl_desc.setFont(QFont("Segoe UI", 8.5))
            lbl_desc.setStyleSheet("color: #556B82;")
            v_text.addWidget(lbl_desc)

            w_layout.addLayout(v_text, 1)

            lbl_icon.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            lbl_name.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            lbl_desc.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

            if eff_id == active_eff:
                lbl_check = QLabel("✓", w)
                lbl_check.setStyleSheet("color: #2563EB; font-weight: bold; font-size: 13pt;")
                lbl_check.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
                w_layout.addWidget(lbl_check)

            # Hover & Click handling
            def make_handlers(target_eff: str, item_widget: QWidget):
                def enter(ev):
                    item_widget.setStyleSheet("background-color: #EBF3FD; border-radius: 3px;")
                    self.effect_hovered.emit(target_eff)
                def leave(ev):
                    item_widget.setStyleSheet("background-color: transparent;")
                    self.effect_unhovered.emit()
                def click(ev):
                    menu.close()
                    self._select_from_menu(target_eff)
                item_widget.enterEvent = enter
                item_widget.leaveEvent = leave
                item_widget.mousePressEvent = click

            make_handlers(eff_id, w)
            action.setDefaultWidget(w)
            action.triggered.connect(lambda checked=False, e=eff_id: self._select_from_menu(e))
            menu.addAction(action)

        menu.aboutToHide.connect(self.effect_unhovered.emit)
        if not getattr(self, "_suppress_menu_exec", False):
            pos = self.mapToGlobal(QPoint(0, self.height()))
            screen = self.screen() or (QApplication.primaryScreen() if QApplication.instance() else None)
            if screen:
                avail_w = screen.availableGeometry().right()
                menu_w = menu.sizeHint().width()
                if pos.x() + menu_w > avail_w:
                    pos.setX(max(10, avail_w - menu_w - 10))
            menu.exec(pos)

    def _select_from_menu(self, eff_id: str) -> None:
        self.set_active_effect(eff_id)
        self.effect_selected.emit(eff_id)

    def set_active_effect(self, eff_id: str) -> None:
        target = eff_id if eff_id in self.buttons else "none"
        btn = self.buttons.get(target)
        if btn:
            btn.setChecked(True)

    def update_live_thumbnails(self, pixmap_dict: dict[str, QPixmap]) -> None:
        """Updates the effect buttons with live thumbnail previews of the active image."""
        self._live_thumbnails = dict(pixmap_dict)
        self._icons_initialized = True
        for eff_id, pix in pixmap_dict.items():
            btn = self.buttons.get(eff_id)
            if btn and not pix.isNull():
                btn.setIcon(QIcon(pix))
                btn.setIconSize(QSize(53, 64))


class _RibbonGroupBox(QFrame):
    """Visual WLPG Ribbon group with content row on top and subtle grey group caption at bottom."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("RibbonGroupBox")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(1, 0, 1, 1)
        vbox.setSpacing(1)

        self.content_widget = QWidget(self)
        self.content_widget.setFixedHeight(73)
        self.content_widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.content_layout = QHBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(1)
        self.content_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        vbox.addWidget(self.content_widget)

        self.lbl_title = QLabel(title, self)
        self.lbl_title.setFixedHeight(15)
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet(
            "color: #55687C; font-size: 8pt; font-family: 'Segoe UI'; border: none; background: transparent; padding-bottom: 1px;"
        )
        vbox.addWidget(self.lbl_title)

    def setTitle(self, title: str) -> None:
        self.lbl_title.setText(title)

    def adjust_to_content(self) -> None:
        self.content_layout.invalidate()
        self.content_layout.activate()
        cw = self.content_layout.sizeHint().width()
        tw = self.lbl_title.sizeHint().width()
        w = max(cw, tw) + 4
        self.content_widget.setFixedWidth(cw)
        self.setFixedWidth(w)
        self.updateGeometry()

    def sizeHint(self) -> QSize:
        cw = self.content_layout.sizeHint().width()
        tw = self.lbl_title.sizeHint().width()
        w = max(cw, tw) + 2
        return QSize(w, 90)

    def minimumSizeHint(self) -> QSize:
        cw = self.content_layout.minimumSize().width()
        tw = self.lbl_title.minimumSizeHint().width()
        w = max(cw, tw) + 2
        return QSize(w, 90)


class _RibbonPage(QWidget):
    """Ribbon tab page with relaxed minimumSizeHint to allow responsive window resizing down to 400px."""

    def minimumSizeHint(self) -> QSize:
        return QSize(100, 88)


class _FaceCarouselWidget(QWidget):
    """Horizontal scrollable strip of face avatar cards matching WLPG Home Quick find."""

    person_clicked = Signal(int)
    boolean_mode_selected = Signal(str)
    special_filter_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buttons: dict[int, QToolButton] = {}
        self._button_list: list[QToolButton] = []
        self._all_people_list: list[dict[str, Any]] = []
        self._top_people: list[dict[str, Any]] = []
        self._visible_faces: int = 5
        self._scroll_index: int = 0

        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(1)

        # Bordered container frame matching WLPG (62px tall)
        self.container = QFrame(self)
        self.container.setObjectName("FaceCarouselContainer")
        self.container.setFixedHeight(62)
        self.container.setStyleSheet(
            """
            QFrame#FaceCarouselContainer,
            QFrame#RibbonArrangeBox {
                background: #FFFFFF;
                border: 1px solid #ABBFD3;
                border-radius: 2px;
            }
            """
        )
        c_layout = QHBoxLayout(self.container)
        c_layout.setContentsMargins(1, 1, 1, 1)
        c_layout.setSpacing(1)

        self.scroll = QScrollArea(self.container)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setFixedHeight(58)
        self.scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.strip = QWidget()
        self.strip.setStyleSheet("background: transparent;")
        self.strip_layout = QHBoxLayout(self.strip)
        self.strip_layout.setContentsMargins(1, 0, 1, 0)
        self.strip_layout.setSpacing(2)
        self.strip_layout.addStretch(1)
        self.scroll.setWidget(self.strip)
        c_layout.addWidget(self.scroll)

        # Vertical 3-button stack on the right edge matching WLPG: ▲, ▼, and ▾ (More)
        btn_vbox = QVBoxLayout()
        btn_vbox.setContentsMargins(0, 0, 0, 0)
        btn_vbox.setSpacing(1)

        self.btn_scroll_left = QPushButton("▲", self.container)
        self.btn_scroll_left.setToolTip("Scroll left")
        self.btn_scroll_right = QPushButton("▼", self.container)
        self.btn_scroll_right.setToolTip("Scroll right")
        self.btn_expand = QPushButton(self.container)
        self.btn_expand.setToolTip("Show all people and search options")

        # Draw classic ribbon gallery "More" icon (top line + down arrow)
        more_pix = QPixmap(10, 10)
        more_pix.fill(Qt.transparent)
        mp = QPainter(more_pix)
        mp.setRenderHint(QPainter.Antialiasing)
        mp.setPen(QPen(QColor("#55687C"), 1.2))
        mp.drawLine(1, 2, 9, 2)
        mp.setBrush(QColor("#55687C"))
        tri = QPolygonF([QPointF(2, 4.5), QPointF(8, 4.5), QPointF(5, 7.5)])
        mp.drawPolygon(tri)
        mp.end()
        self.btn_expand.setIcon(QIcon(more_pix))
        self.btn_expand.setIconSize(QSize(10, 10))

        btn_style = """
            QPushButton {
                border: 1px solid #D2DFEE;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F6F9FD, stop:1 #E2ECF7);
                color: #55687C;
                font-size: 7pt;
                padding: 0;
            }
            QPushButton:hover {
                background: #CCE4F7;
                border-color: #7BAEDB;
            }
            QPushButton:disabled {
                color: #B0C4DE;
                background: #F4F6F9;
                border-color: #DCE6F2;
            }
        """
        for b in (self.btn_scroll_left, self.btn_scroll_right, self.btn_expand):
            b.setFixedSize(14, 19)
            b.setFlat(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(btn_style)

        self.btn_scroll_left.clicked.connect(self._scroll_left)
        self.btn_scroll_right.clicked.connect(self._scroll_right)
        self.btn_expand.clicked.connect(self._show_people_dropdown)
        btn_vbox.addWidget(self.btn_scroll_left)
        btn_vbox.addWidget(self.btn_scroll_right)
        btn_vbox.addWidget(self.btn_expand)
        c_layout.addLayout(btn_vbox)

        hbox.addWidget(self.container)
        self.setFixedHeight(62)
        self.set_visible_faces(5)

    def set_dark_mode(self, is_dark: bool) -> None:
        self._is_dark = bool(is_dark)
        if is_dark:
            self.container.setStyleSheet(
                """
                QFrame#FaceCarouselContainer {
                    background: #14171C;
                    border: 1px solid #333A45;
                    border-radius: 2px;
                }
                """
            )
            btn_style = """
                QPushButton {
                    border: 1px solid #333A45;
                    background: #20242B;
                    color: #94A3B8;
                    font-size: 7pt;
                    padding: 0;
                }
                QPushButton:hover {
                    background: #2E3642;
                    border-color: #60A5FA;
                    color: #FFFFFF;
                }
                QPushButton:disabled {
                    color: #475569;
                    background: #181B20;
                    border-color: #262B33;
                }
            """
        else:
            self.container.setStyleSheet(
                """
                QFrame#FaceCarouselContainer {
                    background: #FFFFFF;
                    border: 1px solid #ABBFD3;
                    border-radius: 2px;
                }
                """
            )
            btn_style = """
                QPushButton {
                    border: 1px solid #D2DFEE;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F6F9FD, stop:1 #E2ECF7);
                    color: #55687C;
                    font-size: 7pt;
                    padding: 0;
                }
                QPushButton:hover {
                    background: #CCE4F7;
                    border-color: #7BAEDB;
                }
                QPushButton:disabled {
                    color: #B0C4DE;
                    background: #F4F6F9;
                    border-color: #DCE6F2;
                }
            """
        for b in (self.btn_scroll_left, self.btn_scroll_right, self.btn_expand):
            b.setStyleSheet(btn_style)

    def set_visible_faces(self, count: int) -> None:
        self._visible_faces = max(1, min(10, count))
        scroll_w = self._visible_faces * 56
        total_w = scroll_w + 19
        self.scroll.setFixedWidth(scroll_w)
        self.container.setFixedWidth(total_w)
        self.setFixedWidth(total_w)
        self.setMinimumWidth(total_w)
        self.setMaximumWidth(total_w)
        self.setFixedHeight(62)
        self.container.setFixedHeight(62)
        self.scroll.setFixedHeight(58)
        self._update_visible_faces()

    def sizeHint(self) -> QSize:
        w = getattr(self, "_visible_faces", 5) * 56 + 19
        return QSize(w, 62)

    def minimumSizeHint(self) -> QSize:
        w = getattr(self, "_visible_faces", 1) * 56 + 19
        return QSize(w, 62)

    def _update_visible_faces(self) -> None:
        n = len(self._button_list)
        max_idx = max(0, n - self._visible_faces)
        self._scroll_index = max(0, min(self._scroll_index, max_idx))
        for i, btn in enumerate(self._button_list):
            btn.setVisible(self._scroll_index <= i < self._scroll_index + self._visible_faces)
        self.btn_scroll_left.setEnabled(self._scroll_index > 0)
        self.btn_scroll_right.setEnabled(self._scroll_index < max_idx)

    def _scroll_left(self) -> None:
        if self._scroll_index > 0:
            self._scroll_index -= 1
            self._update_visible_faces()

    def _scroll_right(self) -> None:
        n = len(self._button_list)
        max_idx = max(0, n - self._visible_faces)
        if self._scroll_index < max_idx:
            self._scroll_index += 1
            self._update_visible_faces()

    def populate(self, people: Sequence[dict[str, Any]]) -> None:
        while self.strip_layout.count() > 1:
            item = self.strip_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._buttons.clear()
        self._button_list.clear()
        self._scroll_index = 0
        self._all_people_list = list(people)

        # Sort by Top people (highest media count & confirmed count descending)
        self._top_people = sorted(
            self._all_people_list,
            key=lambda p: (int(p.get("media_count", 0)), int(p.get("confirmed_count", 0))),
            reverse=True,
        )

        for p in self._top_people:
            pid = int(p["id"])
            name = str(p["name"])
            cnt = int(p.get("media_count", 0))
            blob = p.get("thumbnail_blob")

            btn = QToolButton(self.strip)
            btn.setFixedSize(54, 54)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(f"{name} ({cnt} items)")

            pix = make_person_avatar_pixmap(name, blob, size=50)
            btn.setIcon(QIcon(pix))
            btn.setIconSize(QSize(50, 50))
            btn.setStyleSheet(
                """
                QToolButton {
                    border: 1px solid #C8D7E6;
                    border-radius: 2px;
                    background: transparent;
                    padding: 0px;
                }
                QToolButton:hover {
                    border: 1px solid #E2B034;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #FFFDF5, stop:1 #FCE7A2);
                }
                QToolTip {
                    background-color: #FFFFFF;
                    color: #1E395B;
                    border: 1px solid #7DA2CE;
                    border-radius: 2px;
                    padding: 3px 6px;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                }
                """
            )
            btn.clicked.connect(lambda _checked=False, person_id=pid: self.person_clicked.emit(person_id))
            self.strip_layout.insertWidget(self.strip_layout.count() - 1, btn)
            self._buttons[pid] = btn
            self._button_list.append(btn)

        self._update_visible_faces()

    def click_person(self, person_id: int) -> None:
        if person_id in self._buttons:
            self._buttons[person_id].click()
        else:
            self.person_clicked.emit(person_id)

    def _show_people_dropdown(self) -> None:
        if hasattr(self, "_active_popup") and self._active_popup is not None:
            try:
                if self._active_popup.isVisible():
                    self._active_popup.close()
                    self._active_popup = None
                    return
            except RuntimeError:
                self._active_popup = None

        popup = _WLPGPeopleDropdownPopup(self, self._all_people_list)
        self._active_popup = popup
        popup.person_selected.connect(self.person_clicked.emit)
        popup.boolean_mode_selected.connect(self.boolean_mode_selected.emit)
        popup.special_filter_selected.connect(self.special_filter_selected.emit)

        last_sz = getattr(self, "_last_dropdown_size", None)
        target_w = max(580, last_sz.width() if last_sz else 580)
        target_h = max(260, last_sz.height() if last_sz else 380)

        # In WLPG, the dropdown integrates Top people by directly overlaying the carousel box
        pos = self.container.mapToGlobal(QPoint(0, 0))
        screen = self.screen() or (QApplication.primaryScreen() if QApplication.instance() else None)
        if screen:
            avail = screen.availableGeometry()
            if pos.x() + target_w > avail.right():
                pos.setX(max(avail.left() + 10, avail.right() - target_w - 10))
            if pos.y() + target_h > avail.bottom():
                pos.setY(max(avail.top() + 10, avail.bottom() - target_h - 10))

        popup.move(pos)
        popup.resize(target_w, target_h)
        popup.show()


class _CornerResizeGrip(QWidget):
    """Bottom-right diagonal resize gripper matching classic Windows/Aero."""

    def __init__(self, target_window: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.target_window = target_window
        self.setFixedSize(14, 14)
        self.setCursor(Qt.SizeFDiagCursor)
        self._dragging = False
        self._drag_start_pos = QPoint()
        self._drag_start_size: QSize | None = None

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        dot_color = QColor("#7DA2CE")
        highlight = QColor("#FFFFFF")
        dots = [
            (10, 2),
            (6, 6), (10, 6),
            (2, 10), (6, 10), (10, 10)
        ]
        for x, y in dots:
            p.fillRect(x + 1, y + 1, 2, 2, highlight)
            p.fillRect(x, y, 2, 2, dot_color)
        p.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_start_pos = event.globalPosition().toPoint()
            self._drag_start_size = self.target_window.size()
            self.grabMouse()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging and self._drag_start_size is not None:
            delta = event.globalPosition().toPoint() - self._drag_start_pos
            min_w = self.target_window.minimumWidth()
            min_h = self.target_window.minimumHeight()
            new_w = max(min_w, self._drag_start_size.width() + delta.x())
            new_h = max(min_h, self._drag_start_size.height() + delta.y())
            self.target_window.resize(new_w, new_h)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self.releaseMouse()
            event.accept()


class _WLPGPeopleDropdownPopup(QWidget):
    """
    Authentic Windows Live Photo Gallery People Quick Find popup matching media_1794133844741_7f43b71b.jpg:
    - Same avatar size (54x54 button with 50x50 avatar) as in-ribbon strip
    - Minimum 10 columns (580px) wide, maintaining full width even when main window shrinks
    - Resizable via bottom-right corner grip, reflowing the grid dynamically
    - Integrates Top people at the top and Other people A - Z in a single scrollable area
    - Bottom frame with boolean & special filter options and resize grip
    """

    person_selected = Signal(int)
    boolean_mode_selected = Signal(str)
    special_filter_selected = Signal(str)

    def __init__(
        self,
        parent_carousel: "_FaceCarouselWidget",
        people: Sequence[dict[str, Any]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.parent_carousel = parent_carousel
        self.setObjectName("WLPGPeopleDropdownPopup")
        self.setMinimumSize(580, 260)

        is_dark = getattr(parent_carousel, "_is_dark", False)
        bg = "#1E222B" if is_dark else "#FFFFFF"
        border = "#4B5563" if is_dark else "#7DA2CE"
        self.setStyleSheet(
            f"""
            QWidget#WLPGPeopleDropdownPopup {{
                background: {bg};
                border: 1px solid {border};
            }}
            """
        )

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(1, 1, 1, 1)
        vbox.setSpacing(0)

        # Sort people into Top 10 and Other People A-Z
        all_people = list(people)
        top_people = sorted(
            all_people,
            key=lambda p: (int(p.get("media_count", 0)), int(p.get("confirmed_count", 0))),
            reverse=True,
        )
        self._top_10 = top_people[:10]
        top_ids = {int(p["id"]) for p in self._top_10}
        self._other_people = sorted(
            [p for p in all_people if int(p["id"]) not in top_ids],
            key=lambda p: str(p.get("name", "")).lower(),
        )

        # Scroll Area hosting Top people and Other people A-Z
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setStyleSheet(
            f"""
            QScrollArea {{
                background: {bg};
                border: none;
            }}
            QScrollBar:vertical {{
                border: none;
                background: transparent;
                width: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: #CBD5E1;
                min-height: 24px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: #94A3B8;
            }}
            QScrollBar::handle:vertical:pressed {{
                background: #64748B;
            }}
            QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {{
                border: none;
                background: transparent;
                height: 0px;
                width: 0px;
            }}
            QScrollBar::sub-page:vertical, QScrollBar::add-page:vertical {{
                border: none;
                background: transparent;
            }}
            """
        )

        self.scroll_content = QWidget()
        self.scroll_content.setStyleSheet(f"background: {bg};")
        self.content_layout = QVBoxLayout(self.scroll_content)
        self.content_layout.setContentsMargins(4, 4, 4, 4)
        self.content_layout.setSpacing(4)

        hdr_style = (
            "background: #2D333B; border-bottom: 1px solid #3E4651; color: #E2E8F0;"
            if is_dark
            else "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F9FBFE, stop:1 #EBF2F8);"
            "border-bottom: 1px solid #D5E0EA; color: #35506E;"
        ) + "font-weight: bold; font-size: 8.5pt; font-family: 'Segoe UI'; padding: 3px 6px;"

        # 1. Section Header: Top people
        self.lbl_top = QLabel("Top people", self.scroll_content)
        self.lbl_top.setStyleSheet(hdr_style)
        self.content_layout.addWidget(self.lbl_top)

        btn_style = (
            """
            QToolButton {
                border: 1px solid #3E4651;
                border-radius: 2px;
                background: #252A34;
                padding: 0px;
            }
            QToolButton:hover {
                border: 1px solid #60A5FA;
                background: #2E3646;
            }
            """
            if is_dark
            else """
            QToolButton {
                border: 1px solid #C8D7E6;
                border-radius: 2px;
                background: #FFFFFF;
                padding: 0px;
            }
            QToolButton:hover {
                border: 1px solid #E2B034;
                background: #FFF8E6;
            }
            """
        )

        self.top_grid_widget = QWidget(self.scroll_content)
        self.top_grid_widget.setStyleSheet("background: transparent;")
        self.top_grid = QGridLayout(self.top_grid_widget)
        self.top_grid.setContentsMargins(0, 2, 0, 4)
        self.top_grid.setSpacing(2)
        self.top_grid.setAlignment(Qt.AlignLeft)

        for col, p in enumerate(self._top_10):
            pid = int(p["id"])
            name = str(p.get("name", ""))
            cnt = int(p.get("media_count", 0))
            blob = p.get("thumbnail_blob")
            btn = QToolButton(self.top_grid_widget)
            btn.setFixedSize(54, 54)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(f"{name} ({cnt} items)")
            pix = make_person_avatar_pixmap(name, blob, size=50)
            btn.setIcon(QIcon(pix))
            btn.setIconSize(QSize(50, 50))
            btn.setStyleSheet(btn_style)
            btn.clicked.connect(lambda _c=False, person_id=pid: self._on_person_clicked(person_id))
            self.top_grid.addWidget(btn, 0, col)

        self.content_layout.addWidget(self.top_grid_widget)

        # 2. Section Header: Other people A - Z
        self._other_buttons: list[QToolButton] = []
        self._current_cols: int = 10
        if self._other_people:
            self.lbl_other = QLabel("Other people A - Z", self.scroll_content)
            self.lbl_other.setStyleSheet(hdr_style)
            self.content_layout.addWidget(self.lbl_other)

            self.other_grid_widget = QWidget(self.scroll_content)
            self.other_grid_widget.setStyleSheet("background: transparent;")
            self.other_grid = QGridLayout(self.other_grid_widget)
            self.other_grid.setContentsMargins(0, 2, 0, 4)
            self.other_grid.setSpacing(2)
            self.other_grid.setAlignment(Qt.AlignLeft)

            for p in self._other_people:
                pid = int(p["id"])
                name = str(p.get("name", ""))
                cnt = int(p.get("media_count", 0))
                blob = p.get("thumbnail_blob")
                btn = QToolButton(self.other_grid_widget)
                btn.setFixedSize(54, 54)
                btn.setCursor(Qt.PointingHandCursor)
                btn.setToolTip(f"{name} ({cnt} items)")
                pix = make_person_avatar_pixmap(name, blob, size=50)
                btn.setIcon(QIcon(pix))
                btn.setIconSize(QSize(50, 50))
                btn.setStyleSheet(btn_style)
                btn.clicked.connect(lambda _c=False, person_id=pid: self._on_person_clicked(person_id))
                self._other_buttons.append(btn)

            self.content_layout.addWidget(self.other_grid_widget)

        self.content_layout.addStretch(1)
        self.scroll.setWidget(self.scroll_content)
        vbox.addWidget(self.scroll, 1)

        # 3. Pinned Bottom Actions + Corner Resize Grip
        bottom_frame = QFrame(self)
        bottom_frame.setObjectName("WLPGPeopleDropdownBottomFrame")
        b_bg = "#181B20" if is_dark else "#F6F9FD"
        b_border = "#333A45" if is_dark else "#D5E0EA"
        bottom_frame.setStyleSheet(
            f"QFrame#WLPGPeopleDropdownBottomFrame {{ background: {b_bg}; border-top: 1px solid {b_border}; }}"
        )
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(6, 4, 4, 4)
        bottom_layout.setSpacing(0)

        b_vbox = QVBoxLayout()
        b_vbox.setContentsMargins(0, 0, 0, 0)
        b_vbox.setSpacing(2)

        opt_style = (
            """
            QToolButton {
                border: 1px solid transparent;
                border-radius: 2px;
                background: transparent;
                color: #E2E8F0;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                text-align: left;
                padding: 2px 6px;
            }
            QToolButton:hover {
                background: #2E3646;
                border: 1px solid #60A5FA;
            }
            """
            if is_dark
            else """
            QToolButton {
                border: 1px solid transparent;
                border-radius: 2px;
                background: transparent;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                text-align: left;
                padding: 2px 6px;
            }
            QToolButton:hover {
                background: #E5F1FB;
                border: 1px solid #A0C5E8;
            }
            """
        )

        self.btn_and = QToolButton(bottom_frame)
        self.btn_and.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_and.setText("&and")
        self.btn_and.setIcon(make_wlpg_icon("boolean_and", 14))
        self.btn_and.setIconSize(QSize(14, 14))
        self.btn_and.setStyleSheet(opt_style)
        self.btn_and.setFixedHeight(22)
        self.btn_and.clicked.connect(lambda: self._on_boolean_mode("AND"))
        b_vbox.addWidget(self.btn_and)

        self.btn_or = QToolButton(bottom_frame)
        self.btn_or.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_or.setText("&or")
        self.btn_or.setIcon(make_wlpg_icon("boolean_or", 14))
        self.btn_or.setIconSize(QSize(14, 14))
        self.btn_or.setStyleSheet(opt_style)
        self.btn_or.setFixedHeight(22)
        self.btn_or.clicked.connect(lambda: self._on_boolean_mode("OR"))
        b_vbox.addWidget(self.btn_or)

        self.btn_all_tagged = QToolButton(bottom_frame)
        self.btn_all_tagged.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_all_tagged.setText("All &tagged or detected")
        self.btn_all_tagged.setIcon(make_wlpg_icon("person", 14))
        self.btn_all_tagged.setIconSize(QSize(14, 14))
        self.btn_all_tagged.setStyleSheet(opt_style)
        self.btn_all_tagged.setFixedHeight(22)
        self.btn_all_tagged.clicked.connect(lambda: self._on_special_filter("all_people"))
        b_vbox.addWidget(self.btn_all_tagged)

        self.btn_detected = QToolButton(bottom_frame)
        self.btn_detected.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_detected.setText("All &detected but not tagged")
        self.btn_detected.setIcon(make_wlpg_icon("person_question", 14))
        self.btn_detected.setIconSize(QSize(14, 14))
        self.btn_detected.setStyleSheet(opt_style)
        self.btn_detected.setFixedHeight(22)
        self.btn_detected.clicked.connect(lambda: self._on_special_filter("detected_untagged"))
        b_vbox.addWidget(self.btn_detected)

        self.btn_untagged = QToolButton(bottom_frame)
        self.btn_untagged.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_untagged.setText("&Not tagged or detected")
        self.btn_untagged.setIcon(make_wlpg_icon("person_none", 14))
        self.btn_untagged.setIconSize(QSize(14, 14))
        self.btn_untagged.setStyleSheet(opt_style)
        self.btn_untagged.setFixedHeight(22)
        self.btn_untagged.clicked.connect(lambda: self._on_special_filter("untagged"))
        b_vbox.addWidget(self.btn_untagged)

        bottom_layout.addLayout(b_vbox, 1)

        grip_vbox = QVBoxLayout()
        grip_vbox.setContentsMargins(0, 0, 0, 0)
        grip_vbox.addStretch(1)
        self.grip = _CornerResizeGrip(self, bottom_frame)
        grip_vbox.addWidget(self.grip, 0, Qt.AlignRight | Qt.AlignBottom)
        bottom_layout.addLayout(grip_vbox, 0)

        vbox.addWidget(bottom_frame, 0)

    def _reflow_other_grid(self) -> None:
        if not hasattr(self, "other_grid") or not self._other_buttons:
            return
        viewport_w = self.scroll.viewport().width()
        cols = max(10, (viewport_w - 4) // 56)
        if cols == self._current_cols and self.other_grid.count() > 0:
            return
        self._current_cols = cols

        while self.other_grid.count() > 0:
            self.other_grid.takeAt(0)

        for i, btn in enumerate(self._other_buttons):
            row = i // cols
            col = i % cols
            self.other_grid.addWidget(btn, row, col)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._reflow_other_grid()
        if hasattr(self, "parent_carousel") and self.parent_carousel is not None:
            self.parent_carousel._last_dropdown_size = self.size()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._reflow_other_grid()

    def closeEvent(self, event: QCloseEvent) -> None:
        if hasattr(self, "parent_carousel") and self.parent_carousel is not None:
            if getattr(self.parent_carousel, "_active_popup", None) is self:
                self.parent_carousel._active_popup = None
        super().closeEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _on_person_clicked(self, pid: int) -> None:
        self.close()
        self.person_selected.emit(pid)

    def _on_boolean_mode(self, mode: str) -> None:
        self.close()
        self.boolean_mode_selected.emit(mode)

    def _on_special_filter(self, kind: str) -> None:
        self.close()
        self.special_filter_selected.emit(kind)


class _WLPGPeopleMenuWidget(QWidget):
    """
    Authentic Windows Live Photo Gallery People dropdown widget matching media_1792919727459_8c5fe8cf.png:
    - Top people (1 row of top 10 people by photo count)
    - Other people A - Z (10-column scrollable grid of all other people sorted alphabetically)
    - Search options: and, or, All tagged or detected, All detected but not tagged, Not tagged or detected
    """

    person_selected = Signal(int)
    boolean_mode_selected = Signal(str)
    special_filter_selected = Signal(str)

    def __init__(
        self,
        menu: QMenu,
        people: Sequence[dict[str, Any]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent or menu)
        self.menu = menu
        self.setFixedWidth(580)
        self.setObjectName("WLPGPeopleMenuWidget")
        self.setStyleSheet(
            """
            QWidget#WLPGPeopleMenuWidget {
                background: #FFFFFF;
                border: 1px solid #7DA2CE;
            }
            """
        )

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Sort people into Top 10 and Other People A-Z
        all_people = list(people)
        top_people = sorted(
            all_people,
            key=lambda p: (int(p.get("media_count", 0)), int(p.get("confirmed_count", 0))),
            reverse=True,
        )
        top_10 = top_people[:10]
        top_ids = {int(p["id"]) for p in top_10}
        other_people = sorted(
            [p for p in all_people if int(p["id"]) not in top_ids],
            key=lambda p: str(p.get("name", "")).lower(),
        )

        # Scroll Area hosting Top people and Other people A-Z
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)
        other_rows = (len(other_people) + 9) // 10 if other_people else 0
        needed_h = 95 + (32 + other_rows * 56 if other_rows > 0 else 0) + 12
        scroll.setFixedHeight(min(380, max(160, needed_h)))
        scroll.setStyleSheet(
            """
            QScrollArea {
                background: #FFFFFF;
                border: none;
            }
            QScrollBar:vertical {
                border: none;
                background: transparent;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #CBD5E1;
                min-height: 24px;
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

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: #FFFFFF;")
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(4)

        # 1. Section Header: Top people
        lbl_top = QLabel("Top people", scroll_content)
        lbl_top.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F9FBFE, stop:1 #EBF2F8);"
            "border-bottom: 1px solid #D5E0EA; color: #35506E; font-weight: bold; font-size: 8.5pt;"
            "font-family: 'Segoe UI'; padding: 3px 6px;"
        )
        content_layout.addWidget(lbl_top)

        top_grid_widget = QWidget(scroll_content)
        top_grid_widget.setStyleSheet("background: transparent;")
        top_grid = QGridLayout(top_grid_widget)
        top_grid.setContentsMargins(0, 2, 0, 4)
        top_grid.setSpacing(2)
        top_grid.setAlignment(Qt.AlignLeft)

        btn_style = """
            QToolButton {
                border: 1px solid #C8D7E6;
                border-radius: 2px;
                background: #FFFFFF;
                padding: 0px;
            }
            QToolButton:hover {
                border: 1px solid #E2B034;
                background: #FFF8E6;
            }
        """

        for col, p in enumerate(top_10):
            pid = int(p["id"])
            name = str(p.get("name", ""))
            cnt = int(p.get("media_count", 0))
            blob = p.get("thumbnail_blob")
            btn = QToolButton(top_grid_widget)
            btn.setFixedSize(54, 54)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(f"{name} ({cnt} items)")
            pix = make_person_avatar_pixmap(name, blob, size=50)
            btn.setIcon(QIcon(pix))
            btn.setIconSize(QSize(50, 50))
            btn.setStyleSheet(btn_style)
            btn.clicked.connect(lambda _c=False, person_id=pid: self._on_person_clicked(person_id))
            top_grid.addWidget(btn, 0, col)

        content_layout.addWidget(top_grid_widget)

        # 2. Section Header: Other people A - Z
        if other_people:
            lbl_other = QLabel("Other people A - Z", scroll_content)
            lbl_other.setStyleSheet(
                "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F9FBFE, stop:1 #EBF2F8);"
                "border-top: 1px solid #D5E0EA; border-bottom: 1px solid #D5E0EA; color: #35506E; font-weight: bold; font-size: 8.5pt;"
                "font-family: 'Segoe UI'; padding: 3px 6px;"
            )
            content_layout.addWidget(lbl_other)

            other_grid_widget = QWidget(scroll_content)
            other_grid_widget.setStyleSheet("background: transparent;")
            other_grid = QGridLayout(other_grid_widget)
            other_grid.setContentsMargins(0, 2, 0, 4)
            other_grid.setSpacing(2)
            other_grid.setAlignment(Qt.AlignLeft)

            for i, p in enumerate(other_people):
                r = i // 10
                c = i % 10
                pid = int(p["id"])
                name = str(p.get("name", ""))
                cnt = int(p.get("media_count", 0))
                blob = p.get("thumbnail_blob")
                btn = QToolButton(other_grid_widget)
                btn.setFixedSize(54, 54)
                btn.setCursor(Qt.PointingHandCursor)
                btn.setToolTip(f"{name} ({cnt} items)")
                pix = make_person_avatar_pixmap(name, blob, size=50)
                btn.setIcon(QIcon(pix))
                btn.setIconSize(QSize(50, 50))
                btn.setStyleSheet(btn_style)
                btn.clicked.connect(lambda _c=False, person_id=pid: self._on_person_clicked(person_id))
                other_grid.addWidget(btn, r, c)

            content_layout.addWidget(other_grid_widget)

        content_layout.addStretch(1)
        scroll.setWidget(scroll_content)
        vbox.addWidget(scroll)

        # 3. Pinned Bottom Actions
        bottom_frame = QFrame(self)
        bottom_frame.setStyleSheet(
            "background: #F6F9FD; border-top: 1px solid #D5E0EA;"
        )
        b_vbox = QVBoxLayout(bottom_frame)
        b_vbox.setContentsMargins(6, 4, 6, 4)
        b_vbox.setSpacing(2)

        opt_style = """
            QToolButton {
                border: 1px solid transparent;
                border-radius: 2px;
                background: transparent;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                text-align: left;
                padding: 2px 6px;
            }
            QToolButton:hover {
                background: #E5F1FB;
                border: 1px solid #A0C5E8;
            }
        """

        self.btn_and = QToolButton(bottom_frame)
        self.btn_and.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_and.setText("&and")
        self.btn_and.setIcon(make_wlpg_icon("boolean_and", 14))
        self.btn_and.setIconSize(QSize(14, 14))
        self.btn_and.setStyleSheet(opt_style)
        self.btn_and.setFixedHeight(22)
        self.btn_and.clicked.connect(lambda: self._on_boolean_mode("AND"))
        b_vbox.addWidget(self.btn_and)

        self.btn_or = QToolButton(bottom_frame)
        self.btn_or.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_or.setText("&or")
        self.btn_or.setIcon(make_wlpg_icon("boolean_or", 14))
        self.btn_or.setIconSize(QSize(14, 14))
        self.btn_or.setStyleSheet(opt_style)
        self.btn_or.setFixedHeight(22)
        self.btn_or.clicked.connect(lambda: self._on_boolean_mode("OR"))
        b_vbox.addWidget(self.btn_or)

        self.btn_all_tagged = QToolButton(bottom_frame)
        self.btn_all_tagged.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_all_tagged.setText("All &tagged or detected")
        self.btn_all_tagged.setIcon(make_wlpg_icon("person", 14))
        self.btn_all_tagged.setIconSize(QSize(14, 14))
        self.btn_all_tagged.setStyleSheet(opt_style)
        self.btn_all_tagged.setFixedHeight(22)
        self.btn_all_tagged.clicked.connect(lambda: self._on_special_filter("all_people"))
        b_vbox.addWidget(self.btn_all_tagged)

        self.btn_detected = QToolButton(bottom_frame)
        self.btn_detected.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_detected.setText("All &detected but not tagged")
        self.btn_detected.setIcon(make_wlpg_icon("person_question", 14))
        self.btn_detected.setIconSize(QSize(14, 14))
        self.btn_detected.setStyleSheet(opt_style)
        self.btn_detected.setFixedHeight(22)
        self.btn_detected.clicked.connect(lambda: self._on_special_filter("detected_untagged"))
        b_vbox.addWidget(self.btn_detected)

        self.btn_untagged = QToolButton(bottom_frame)
        self.btn_untagged.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_untagged.setText("&Not tagged or detected")
        self.btn_untagged.setIcon(make_wlpg_icon("person_none", 14))
        self.btn_untagged.setIconSize(QSize(14, 14))
        self.btn_untagged.setStyleSheet(opt_style)
        self.btn_untagged.setFixedHeight(22)
        self.btn_untagged.clicked.connect(lambda: self._on_special_filter("untagged"))
        b_vbox.addWidget(self.btn_untagged)

        vbox.addWidget(bottom_frame)

    def _on_person_clicked(self, pid: int) -> None:
        self.menu.close()
        self.person_selected.emit(pid)

    def _on_boolean_mode(self, mode: str) -> None:
        self.menu.close()
        self.boolean_mode_selected.emit(mode)

    def _on_special_filter(self, kind: str) -> None:
        self.menu.close()
        self.special_filter_selected.emit(kind)


class WLPGTagsMenuWidget(QWidget):
    """
    Authentic Windows Live Photo Gallery Tags dropdown widget matching media_1792184066235_7e468e48.png.
    Renders inside a QWidgetAction within the Tags QMenu.
    """

    tag_selected = Signal(int)
    boolean_mode_changed = Signal(str)
    untagged_selected = Signal()
    show_hierarchy_selected = Signal()

    def __init__(self, menu: QMenu, parent: QWidget | None = None) -> None:
        super().__init__(parent or menu)
        self.menu = menu
        self.setFixedWidth(380)
        self.setObjectName("WLPGTagsMenuWidget")
        self.setStyleSheet(
            """
            QWidget#WLPGTagsMenuWidget {
                background: #FFFFFF;
                border: 1px solid #7DA2CE;
            }
            """
        )

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # 1. Section Header: Most used tags
        self.lbl_hdr_most_used = QLabel("Most used tags", self)
        self.lbl_hdr_most_used.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F9FBFE, stop:1 #EBF2F8);"
            "border-bottom: 1px solid #D5E0EA; color: #35506E; font-weight: bold; font-size: 8.5pt;"
            "font-family: 'Segoe UI'; padding: 3px 8px;"
        )
        vbox.addWidget(self.lbl_hdr_most_used)

        # Container for top 10 most used tags
        self.top_tags_container = QWidget(self)
        self.top_tags_container.setStyleSheet("background: #FFFFFF;")
        self.top_tags_layout = QVBoxLayout(self.top_tags_container)
        self.top_tags_layout.setContentsMargins(0, 2, 0, 2)
        self.top_tags_layout.setSpacing(0)
        vbox.addWidget(self.top_tags_container)

        # 2. Section Header: All tags
        self.lbl_hdr_all_tags = QLabel("All tags", self)
        self.lbl_hdr_all_tags.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F9FBFE, stop:1 #EBF2F8);"
            "border-top: 1px solid #D5E0EA; border-bottom: 1px solid #D5E0EA; color: #35506E; font-weight: bold; font-size: 8.5pt;"
            "font-family: 'Segoe UI'; padding: 3px 8px;"
        )
        vbox.addWidget(self.lbl_hdr_all_tags)

        # Scrollable list for All tags
        self.list_all_tags = QListWidget(self)
        self.list_all_tags.setFixedHeight(220)
        self.list_all_tags.setFrameShape(QFrame.NoFrame)
        self.list_all_tags.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_all_tags.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list_all_tags.setStyleSheet(
            """
            QListWidget {
                background: #FFFFFF;
                border: none;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                color: #1E395B;
                padding: 1px 0;
            }
            QListWidget::item {
                height: 20px;
                padding-left: 8px;
            }
            QListWidget::item:hover {
                background: #E5F1FB;
                border: 1px solid #A0C5E8;
                border-radius: 2px;
            }
            QListWidget::item:selected {
                background: #CCE2F7;
                color: #000000;
            }
            """
        )
        self.list_all_tags.itemClicked.connect(self._on_all_tag_clicked)
        vbox.addWidget(self.list_all_tags)

        # 3. Pinned Bottom Bar
        bottom_frame = QFrame(self)
        bottom_frame.setStyleSheet(
            "background: #F6F9FD; border-top: 1px solid #D5E0EA;"
        )
        b_vbox = QVBoxLayout(bottom_frame)
        b_vbox.setContentsMargins(4, 3, 4, 3)
        b_vbox.setSpacing(2)

        # Mode toggles: [+ and] and [/ or]
        mode_hbox = QHBoxLayout()
        mode_hbox.setContentsMargins(2, 0, 2, 0)
        mode_hbox.setSpacing(4)

        btn_style = """
            QPushButton {
                border: 1px solid #BDD3E8;
                border-radius: 2px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FDFEFE, stop:1 #EBF2F9);
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
                padding: 0 8px;
            }
            QPushButton:hover {
                background: #E5F1FB;
                border-color: #7DA2CE;
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #EAF3FD, stop:1 #BBDDF7);
                border: 1px solid #7DA2CE;
                font-weight: 600;
            }
        """
        self.btn_mode_and = QPushButton("and", bottom_frame)
        self.btn_mode_and.setIcon(make_wlpg_icon("boolean_and", 12))
        self.btn_mode_and.setIconSize(QSize(12, 12))
        self.btn_mode_and.setCheckable(True)
        self.btn_mode_and.setFixedHeight(20)
        self.btn_mode_and.setStyleSheet(btn_style)

        self.btn_mode_or = QPushButton("or", bottom_frame)
        self.btn_mode_or.setIcon(make_wlpg_icon("boolean_or", 12))
        self.btn_mode_or.setIconSize(QSize(12, 12))
        self.btn_mode_or.setCheckable(True)
        self.btn_mode_or.setFixedHeight(20)
        self.btn_mode_or.setChecked(True)
        self.btn_mode_or.setStyleSheet(btn_style)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.btn_mode_and)
        self.mode_group.addButton(self.btn_mode_or)

        self.btn_mode_and.toggled.connect(
            lambda checked: self.boolean_mode_changed.emit("AND" if checked else "OR")
        )

        mode_hbox.addWidget(self.btn_mode_and)
        mode_hbox.addWidget(self.btn_mode_or)
        mode_hbox.addStretch(1)
        b_vbox.addLayout(mode_hbox)

        sep = QFrame(bottom_frame)
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #DDE6EF; margin: 1px 0;")
        b_vbox.addWidget(sep)

        action_btn_style = """
            QPushButton {
                text-align: left;
                border: 1px solid transparent;
                background: transparent;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                padding: 2px 6px;
                border-radius: 2px;
            }
            QPushButton:hover {
                background: #E5F1FB;
                border: 1px solid #A0C5E8;
            }
        """
        self.btn_not_tagged = QPushButton("  Not tagged", bottom_frame)
        self.btn_not_tagged.setIcon(make_wlpg_icon("not_tagged", 16))
        self.btn_not_tagged.setStyleSheet(action_btn_style)
        self.btn_not_tagged.clicked.connect(self._on_not_tagged_clicked)
        b_vbox.addWidget(self.btn_not_tagged)

        self.btn_show_hierarchy = QPushButton("Show tag hierarchy...", bottom_frame)
        self.btn_show_hierarchy.setIcon(make_wlpg_icon("tag_hierarchy", 16))
        self.btn_show_hierarchy.setStyleSheet(action_btn_style)
        self.btn_show_hierarchy.clicked.connect(self._on_show_hierarchy_clicked)
        b_vbox.addWidget(self.btn_show_hierarchy)

        vbox.addWidget(bottom_frame)

    def populate(self, tag_tree: Sequence[dict[str, Any]]) -> None:
        while self.top_tags_layout.count():
            item = self.top_tags_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

        sorted_by_usage = sorted(
            tag_tree,
            key=lambda t: int(t.get("media_count", t.get("direct_count", 0))),
            reverse=True,
        )
        top_10 = sorted_by_usage[:10]
        for t in top_10:
            tid = int(t["id"])
            full_path = str(t.get("full_path") or t.get("name") or "Tag")
            cnt = int(t.get("media_count", t.get("direct_count", 0)))
            text = f"{full_path} ({cnt})" if cnt > 0 else full_path

            btn = QPushButton(text, self.top_tags_container)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(
                """
                QPushButton {
                    text-align: left;
                    border: 1px solid transparent;
                    background: transparent;
                    color: #1E395B;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    padding: 2px 8px;
                    border-radius: 2px;
                    height: 18px;
                }
                QPushButton:hover {
                    background: #E5F1FB;
                    border: 1px solid #A0C5E8;
                }
                """
            )
            btn.clicked.connect(lambda _checked=False, tag_id=tid: self._select_tag(tag_id))
            self.top_tags_layout.addWidget(btn)

        self.list_all_tags.clear()
        alpha_tags = sorted(tag_tree, key=lambda t: str(t.get("full_path") or t.get("name") or "").lower())
        for t in alpha_tags:
            tid = int(t["id"])
            full_path = str(t.get("full_path") or t.get("name") or "Tag")
            item = QListWidgetItem(full_path, self.list_all_tags)
            item.setData(Qt.UserRole, tid)

    def _select_tag(self, tag_id: int) -> None:
        self.menu.close()
        self.tag_selected.emit(tag_id)

    def _on_all_tag_clicked(self, item: QListWidgetItem) -> None:
        tid = item.data(Qt.UserRole)
        if tid is not None:
            self._select_tag(int(tid))

    def _on_not_tagged_clicked(self) -> None:
        self.menu.close()
        self.untagged_selected.emit()

    def _on_show_hierarchy_clicked(self) -> None:
        self.menu.close()
        self.show_hierarchy_selected.emit()


class WLPGTagsMenu(QMenu):
    """QMenu container hosting the authentic WLPGTagsMenuWidget."""

    def __init__(self, ribbon: RibbonToolbar) -> None:
        super().__init__(ribbon)
        self.ribbon = ribbon
        self.setObjectName("WLPGTagsMenu")
        self.setStyleSheet(
            """
            QMenu#WLPGTagsMenu {
                background: #FFFFFF;
                border: 1px solid #7DA2CE;
                padding: 0px;
            }
            """
        )
        self.tags_widget = WLPGTagsMenuWidget(self)
        self._wa = QWidgetAction(self)
        self._wa.setDefaultWidget(self.tags_widget)
        super().addAction(self._wa)
        self._extra_actions: list[QAction] = []

        self.tags_widget.tag_selected.connect(
            lambda tid: self.ribbon.tag_filter_selected.emit(tid, "include")
        )
        self.tags_widget.boolean_mode_changed.connect(
            lambda mode: self.ribbon.tag_filter_selected.emit(None, mode)
        )
        self.tags_widget.untagged_selected.connect(
            lambda: self.ribbon.untagged_filter_selected.emit(True)
        )
        self.tags_widget.show_hierarchy_selected.connect(
            lambda: self.ribbon.action_triggered.emit("show_tag_hierarchy")
        )

    def populate(self, tag_tree: Sequence[dict[str, Any]]) -> None:
        self.tags_widget.populate(tag_tree)
        self._extra_actions.clear()
        for t in tag_tree:
            tid = int(t["id"])
            name = str(t.get("name") or t.get("full_path") or "Tag")
            act = QAction(name, self)
            act.setData(tid)
            act.triggered.connect(lambda _c=False, tag_id=tid: self.ribbon.tag_filter_selected.emit(tag_id, "include"))
            self._extra_actions.append(act)

        act_not_tagged = QAction("Not tagged", self)
        act_not_tagged.triggered.connect(lambda: self.ribbon.untagged_filter_selected.emit(True))
        self._extra_actions.append(act_not_tagged)

    def actions(self) -> list[QAction]:
        return [self._wa] + self._extra_actions


class _WLPGStarRowWidget(QWidget):
    """
    Renders an authentic WLPG golden star strip row inside rating popup menus,
    matching media_1793727030068_1c9c47dd.png exactly.
    """
    clicked = Signal(int)

    def __init__(self, rating: int, is_dark: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.rating = rating
        self.is_dark = is_dark
        self._hovered = False
        self.setFixedHeight(26)
        self.setFixedWidth(132)
        self.setCursor(Qt.PointingHandCursor)

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.rating)
        super().mousePressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()

        if self._hovered:
            if self.is_dark:
                grad = QLinearGradient(0, 0, 0, h)
                grad.setColorAt(0.0, QColor("#334155"))
                grad.setColorAt(1.0, QColor("#1E293B"))
                p.setPen(QPen(QColor("#3B82F6"), 1.0))
                p.setBrush(grad)
            else:
                grad = QLinearGradient(0, 0, 0, h)
                grad.setColorAt(0.0, QColor("#F4F9FE"))
                grad.setColorAt(0.5, QColor("#DCEBFB"))
                grad.setColorAt(1.0, QColor("#CBE2FA"))
                p.setPen(QPen(QColor("#7FAAD9"), 1.0))
                p.setBrush(grad)
            p.drawRoundedRect(QRectF(1.5, 1.5, w - 3, h - 3), 2.0, 2.0)

        # 5 Stars: each ~20px diameter, nicely centered
        star_dia = 20.0
        gap = 4.0
        total_stars_w = 5 * star_dia + 4 * gap
        start_x = (w - total_stars_w) / 2.0
        cy = h / 2.0
        outer_r = 9.5
        inner_r = 4.1

        for i in range(5):
            cx = start_x + i * (star_dia + gap) + star_dia / 2.0
            is_filled = (i < self.rating)

            pts: list[QPointF] = []
            for j in range(10):
                ang = math.radians(-90 + j * 36)
                r = outer_r if j % 2 == 0 else inner_r
                pts.append(QPointF(cx + r * math.cos(ang), cy + r * math.sin(ang)))
            poly = QPolygonF(pts)

            if is_filled:
                s_grad = QLinearGradient(cx, cy - outer_r, cx, cy + outer_r)
                s_grad.setColorAt(0.0, QColor("#FFF59E"))
                s_grad.setColorAt(0.4, QColor("#F8D038"))
                s_grad.setColorAt(1.0, QColor("#D99E10"))
                p.setPen(QPen(QColor("#B27B08"), 1.0))
                p.setBrush(s_grad)
                p.drawPolygon(poly)

                # Specular glint on upper point
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(255, 255, 255, 130))
                p.drawPolygon(QPolygonF([pts[0], pts[1], QPointF(cx, cy)]))
            else:
                # Authentic WLPG hollow silver/blue-grey outline
                outline_col = QColor("#64748B") if self.is_dark else QColor("#8DA7C4")
                p.setPen(QPen(outline_col, 1.2))
                p.setBrush(QColor(0, 0, 0, 20) if self.is_dark else QColor("#F4F8FB"))
                p.drawPolygon(poly)


def create_wlpg_rating_menu(
    parent: QWidget,
    on_selected: Callable[[int], None],
    show_all_ratings: bool = False,
    is_dark: bool = False,
) -> QMenu:
    """
    Creates an authentic WLPG golden star strip rating dropdown menu matching
    media_1793727030068_1c9c47dd.png.
    """
    menu = QMenu(parent)
    bg = "#1E242C" if is_dark else "#FFFFFF"
    border = "#3B82F6" if is_dark else "#97B5D5"
    item_col = "#F1F5F9" if is_dark else "#1E395B"
    menu.setStyleSheet(f"""
        QMenu {{
            background-color: {bg};
            border: 1px solid {border};
            padding: 3px;
        }}
        QMenu::item {{
            color: {item_col};
            font-family: 'Segoe UI';
            font-size: 8.5pt;
            padding: 4px 16px;
        }}
        QMenu::item:selected {{
            background-color: {"#334155" if is_dark else "#DCEBFB"};
            border-radius: 2px;
        }}
        QMenu::separator {{
            height: 1px;
            background: {"#334155" if is_dark else "#E2E8F0"};
            margin: 3px 4px;
        }}
    """)
    for r in (5, 4, 3, 2, 1, 0):
        row = _WLPGStarRowWidget(r, is_dark=is_dark, parent=menu)
        def _make_handler(val: int):
            def _handler():
                menu.close()
                on_selected(val)
            return _handler
        row.clicked.connect(_make_handler(r))
        act = QWidgetAction(menu)
        act.setDefaultWidget(row)
        menu.addAction(act)

    if show_all_ratings:
        menu.addSeparator()
        act_all = menu.addAction("All ratings")
        act_all.triggered.connect(lambda: on_selected(-1))

    return menu


class _WindowControlButton(QPushButton):
    """
    Native-styled vector window control button for Minimize, Maximize, Restore, Close.
    Provides pixel-perfect vector lines and overlapping boxes matching Windows 11 / Office.
    """

    def __init__(self, kind: str, parent: QWidget | None = None) -> None:
        text_map = {"min": "─", "max": "□", "restore": "❐", "close": "✕"}
        super().__init__(text_map.get(kind, ""), parent)
        self.kind = kind
        self.setMouseTracking(True)

    def set_kind(self, kind: str) -> None:
        self.kind = kind
        text_map = {"min": "─", "max": "□", "restore": "❐", "close": "✕"}
        self.setText(text_map.get(kind, ""))
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QStylePainter(self)
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        opt.text = ""
        painter.drawControl(QStyle.CE_PushButton, opt)

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, self.kind == "close")

        if self.kind == "close" and (self.underMouse() or self.isDown()):
            color = QColor("#FFFFFF")
        else:
            col = opt.palette.buttonText().color()
            if col.isValid() and col.alpha() > 0:
                color = col
            else:
                color = QColor("#CBD5E1") if getattr(self.window(), "_is_dark", False) else QColor("#1E395B")

        pen = QPen(color, 1.0)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        w = float(self.width())
        h = float(self.height())
        cx = w / 2.0
        cy = h / 2.0

        if self.kind == "min":
            # 10px horizontal line
            p.drawLine(QPointF(cx - 5.0, cy), QPointF(cx + 5.0, cy))
        elif self.kind == "max":
            # 10x10 square outline
            p.drawRect(QRectF(cx - 5.0, cy - 5.0, 10.0, 10.0))
        elif self.kind == "restore":
            # Native overlapping squares (back square offset top-right, front square bottom-left)
            fx = cx - 5.5
            fy = cy - 2.5
            bx = cx - 2.5
            by = cy - 5.5
            p.drawLine(QPointF(bx, by), QPointF(bx + 8.0, by))
            p.drawLine(QPointF(bx + 8.0, by), QPointF(bx + 8.0, by + 8.0))
            p.drawLine(QPointF(bx + 5.0, by + 8.0), QPointF(bx + 8.0, by + 8.0))
            p.drawLine(QPointF(bx, by), QPointF(bx, by + 3.0))
            p.drawRect(QRectF(fx, fy, 8.0, 8.0))
        elif self.kind == "close":
            pen.setWidthF(1.2)
            p.setPen(pen)
            s = 4.8
            p.drawLine(QPointF(cx - s, cy - s), QPointF(cx + s, cy + s))
            p.drawLine(QPointF(cx + s, cy - s), QPointF(cx - s, cy + s))
        p.end()


class _RibbonTitleBar(QWidget):
    """
    Custom integrated window title bar for Essentials Photo Gallery / Office Ribbon.
    Supports native OS dragging via windowHandle().startSystemMove() and double-click maximize/restore.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("RibbonTopHeader")
        self.setFixedHeight(32)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            win = self.window()
            handle = win.windowHandle() if win else None
            if handle is not None:
                handle.startSystemMove()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            win = self.window()
            if win:
                if win.isMaximized():
                    win.showNormal()
                else:
                    win.showMaximized()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)


class _RibbonTabBar(QTabBar):
    """
    Custom QTabBar supporting Fluent UI active tab indicator:
    - In Modern Fluent theme:
      * Selected heading: thin 2.0px Fluent blue indicator bar (#0F6CBD).
        - Normally (resting): line is centered under tab text.
        - On mouse over: smoothly and gradually grows wider (animated ease-out).
      * Non-selected heading: on mouse over, displays a faint grey line (#CBD5E1) indicating focus (matching Office Word / Microsoft 365 Fluent ribbon).
      * Sits along the bottom edge of the tab.
    - In Classic Aero / Dark themes:
      * Standard native tab painting without custom indicator.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self._hover_tab_idx: int = -1
        self._hover_progress: float = 0.0
        self._target_progress: float = 0.0
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._on_anim_tick)

    def _on_anim_tick(self) -> None:
        diff = self._target_progress - self._hover_progress
        if abs(diff) > 0.01:
            self._hover_progress += diff * 0.18
            self.update()
        else:
            self._hover_progress = self._target_progress
            self._anim_timer.stop()
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        super().mouseMoveEvent(event)
        idx = self.tabAt(event.pos())
        if idx != self._hover_tab_idx:
            self._hover_tab_idx = idx
            cur_idx = self.currentIndex()
            self._target_progress = 1.0 if idx == cur_idx else 0.0
            if not self._anim_timer.isActive():
                self._anim_timer.start()
            self.update()

    def leaveEvent(self, event: QEvent) -> None:
        super().leaveEvent(event)
        if self._hover_tab_idx != -1:
            self._hover_tab_idx = -1
            self._target_progress = 0.0
            if not self._anim_timer.isActive():
                self._anim_timer.start()
            self.update()

    def setCurrentIndex(self, index: int) -> None:
        super().setCurrentIndex(index)
        self._hover_progress = 0.0
        self._target_progress = 1.0 if self._hover_tab_idx == index else 0.0
        if not self._anim_timer.isActive():
            self._anim_timer.start()
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)

        parent_ribbon = self.window().findChild(RibbonToolbar) if self.window() else None
        if parent_ribbon is None:
            p = self.parent()
            while p and not isinstance(p, RibbonToolbar):
                p = p.parent()
            parent_ribbon = p

        theme_mode = getattr(parent_ribbon, "_current_theme_mode", "wlpg_aero")
        if theme_mode == "modern":
            cur_idx = self.currentIndex()
            is_dark = getattr(parent_ribbon, "_dark_mode", False)

            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setPen(Qt.NoPen)

            # 1. Non-selected hovered headings: faint grey line indicator
            if 0 <= self._hover_tab_idx < self.count() and self._hover_tab_idx != cur_idx:
                h_rect = self.tabRect(self._hover_tab_idx)
                if h_rect.isValid() and not h_rect.isEmpty():
                    hw = max(18.0, float(h_rect.width() - 28))
                    hx = h_rect.x() + (h_rect.width() - hw) / 2.0
                    hy = float(h_rect.bottom() - 3)
                    grey_color = QColor("#4B5563") if is_dark else QColor("#CBD5E1")
                    p.setBrush(grey_color)
                    p.drawRoundedRect(QRectF(hx, hy, hw, 2.0), 1.0, 1.0)

            # 2. Selected active heading: thinner 2.0px blue line with smooth, slower expansion on mouse over
            if 0 <= cur_idx < self.count():
                rect = self.tabRect(cur_idx)
                if rect.isValid() and not rect.isEmpty():
                    min_w = max(20.0, float(rect.width() - 32))
                    max_w = float(rect.width() - 8)
                    prog = max(0.0, min(1.0, self._hover_progress))
                    w = min_w + (max_w - min_w) * prog
                    x = rect.x() + (rect.width() - w) / 2.0
                    y = float(rect.bottom() - 3)
                    blue_color = QColor("#60A5FA") if is_dark else QColor("#0F6CBD")
                    p.setBrush(blue_color)
                    p.drawRoundedRect(QRectF(x, y, w, 2.0), 1.0, 1.0)

            p.end()


class RibbonToolbar(QWidget):
    """
    Top WLPG Ribbon (`Home`, `Edit`, `Find`, `Create`, `View`) + `Search by:` Scope Header Bar.
    """

    arrange_changed = Signal(str, bool)             # (ArrangeMode, reverse_sort)
    details_overlay_changed = Signal(set)           # set of active detail keys
    person_quick_find_selected = Signal(int)        # person_id
    search_text_changed = Signal(str)               # debounced search query
    search_all_requested = Signal()                 # clear folder/tag scope
    action_triggered = Signal(str)                  # ribbon command identifier
    layout_mode_changed = Signal(str)               # "wlpg_grid" | "mosaic"
    zoom_size_changed = Signal(int)                 # 72..320 px

    # Authentic WLPG Find Tab Filter Signals
    date_filter_selected = Signal(object, object, object)   # (year, month, day)
    rating_filter_selected = Signal(object)                 # tuple of ratings
    tag_filter_selected = Signal(object, str)              # (tag_id, boolean_mode)
    untagged_filter_selected = Signal(bool)
    flagged_filter_selected = Signal(bool)
    flag_filter_selected = Signal(object)                   # int | None: 1 = flagged, 0 = not flagged, None = all
    edited_filter_selected = Signal(bool)
    edited_filter_mode_selected = Signal(object)            # int | None: 1 = edited, 0 = not edited, None = all
    orientation_filter_selected = Signal(str)               # "all" | "landscape" | "portrait"
    media_type_filter_selected = Signal(str)                # "all" | "image" | "video"
    person_boolean_mode_changed = Signal(str)               # "AND" | "OR"
    people_special_filter_selected = Signal(str)            # "all_people" | "detected_untagged" | "untagged"
    text_search_toggle_requested = Signal()
    find_duplicates_requested = Signal()

    # Quick Access Toolbar & Navigation Signals
    back_requested = Signal()
    forward_requested = Signal()
    undo_requested = Signal()
    redo_requested = Signal()
    qat_thumbnails_clicked = Signal()
    qat_map_clicked = Signal()
    qat_people_clicked = Signal()
    filter_month_selected = Signal(int)                     # 0 for all, 1..12

    # Real-time preview & adjustment signals for Exposure and Color menus
    exposure_preview_requested = Signal(float, float)       # (brightness, contrast)
    auto_exposure_preview_requested = Signal()
    exposure_preview_cancelled = Signal()
    exposure_adjustment_selected = Signal(float, float)     # (brightness, contrast)
    auto_exposure_selected = Signal()

    color_preview_requested = Signal(float, float)          # (temperature, tint)
    auto_color_preview_requested = Signal()
    color_preview_cancelled = Signal()
    color_adjustment_selected = Signal(float, float)        # (temperature, tint)
    auto_color_selected = Signal()

    exposure_menu_about_to_show = Signal()
    color_menu_about_to_show = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("RibbonToolbar")

        self._arrange_by: ArrangeMode = "date"
        self._reverse_sort: bool = False
        self._visible_details: set[str] = {"rating"}
        self._detail_checkboxes: dict[str, QAbstractButton] = {}
        self._responsive_large_buttons: list[QToolButton] = []
        self._responsive_secondary_buttons: list[QToolButton] = []
        self._all_themed_buttons: list[QToolButton] = []
        self.current_slideshow_theme: str = "static"
        self._slideshow_theme_actions: list[QAction] = []
        icon_provider.signals.theme_changed.connect(self.refresh_icons)

        # Search input debounce timer (180ms)
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(180)
        self._search_timer.timeout.connect(self._emit_search_text)

        main_vbox = QVBoxLayout(self)
        main_vbox.setContentsMargins(0, 0, 0, 0)
        main_vbox.setSpacing(0)
        self.main_vbox = main_vbox

        # 0. Quick Access Toolbar & App Title Header
        self._qat_below_ribbon: bool = False

        self.qat_container = QWidget(self)
        self.qat_container.setObjectName("QATContainer")
        self.qat_layout = QHBoxLayout(self.qat_container)
        self.qat_layout.setContentsMargins(0, 0, 0, 0)
        self.qat_layout.setSpacing(2)

        self.btn_back = QToolButton(self.qat_container)
        self.btn_back.setObjectName("QATCircleBtn")
        self.btn_back.setIcon(make_wlpg_icon("nav_back", 20))
        self.btn_back.setIconSize(QSize(20, 20))
        self.btn_back.setFixedSize(24, 24)
        self.btn_back.setToolTip("Back (Alt+Left Arrow)")
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.setEnabled(False)
        self.btn_back.clicked.connect(self.back_requested.emit)
        self.qat_layout.addWidget(self.btn_back)

        self.btn_forward = QToolButton(self.qat_container)
        self.btn_forward.setObjectName("QATCircleBtn")
        self.btn_forward.setIcon(make_wlpg_icon("nav_forward", 20))
        self.btn_forward.setIconSize(QSize(20, 20))
        self.btn_forward.setFixedSize(24, 24)
        self.btn_forward.setToolTip("Forward (Alt+Right Arrow)")
        self.btn_forward.setCursor(Qt.PointingHandCursor)
        self.btn_forward.setEnabled(False)
        self.btn_forward.clicked.connect(self.forward_requested.emit)
        self.qat_layout.addWidget(self.btn_forward)

        self.qat_layout.addSpacing(2)

        self.btn_print = QToolButton(self.qat_container)
        self.btn_print.setObjectName("QATBtn")
        self.btn_print.setIcon(make_wlpg_icon("print", 18))
        self.btn_print.setIconSize(QSize(18, 18))
        self.btn_print.setFixedSize(24, 24)
        self.btn_print.setToolTip("Print (Ctrl+P)")
        self.btn_print.setCursor(Qt.PointingHandCursor)
        self.btn_print.setVisible(False)
        self.btn_print.clicked.connect(lambda: self.action_triggered.emit("print"))
        self.qat_layout.addWidget(self.btn_print)

        self.btn_undo = QToolButton(self.qat_container)
        self.btn_undo.setObjectName("QATBtn")
        self.btn_undo.setIcon(make_wlpg_icon("undo", 18))
        self.btn_undo.setIconSize(QSize(18, 18))
        self.btn_undo.setFixedSize(24, 24)
        self.btn_undo.setToolTip("Undo (Ctrl+Z)")
        self.btn_undo.setCursor(Qt.PointingHandCursor)
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self.undo_requested.emit)
        self.qat_layout.addWidget(self.btn_undo)

        self.btn_redo = QToolButton(self.qat_container)
        self.btn_redo.setObjectName("QATBtn")
        self.btn_redo.setIcon(make_wlpg_icon("redo", 18))
        self.btn_redo.setIconSize(QSize(18, 18))
        self.btn_redo.setFixedSize(24, 24)
        self.btn_redo.setToolTip("Redo (Ctrl+Y)")
        self.btn_redo.setCursor(Qt.PointingHandCursor)
        self.btn_redo.setEnabled(False)
        self.btn_redo.clicked.connect(self.redo_requested.emit)
        self.qat_layout.addWidget(self.btn_redo)

        # Gallery View button
        self.btn_qat_thumbnails = QToolButton(self.qat_container)
        self.btn_qat_thumbnails.setObjectName("QATBtn")
        self.btn_qat_thumbnails.setIcon(make_wlpg_icon("thumbnail_view", 18))
        self.btn_qat_thumbnails.setIconSize(QSize(18, 18))
        self.btn_qat_thumbnails.setFixedSize(24, 24)
        self.btn_qat_thumbnails.setToolTip("Gallery view (Ctrl+0)")
        self.btn_qat_thumbnails.setCursor(Qt.PointingHandCursor)
        self.btn_qat_thumbnails.setVisible(False)
        self.btn_qat_thumbnails.clicked.connect(self.qat_thumbnails_clicked.emit)
        self.qat_layout.addWidget(self.btn_qat_thumbnails)

        # World Map View button
        self.btn_qat_map = QToolButton(self.qat_container)
        self.btn_qat_map.setObjectName("QATBtn")
        self.btn_qat_map.setIcon(make_wlpg_icon("globe", 18))
        self.btn_qat_map.setIconSize(QSize(18, 18))
        self.btn_qat_map.setFixedSize(24, 24)
        self.btn_qat_map.setToolTip("World Map View")
        self.btn_qat_map.setCursor(Qt.PointingHandCursor)
        self.btn_qat_map.setVisible(False)
        self.btn_qat_map.clicked.connect(self.qat_map_clicked.emit)
        self.qat_layout.addWidget(self.btn_qat_map)

        # People View button
        self.btn_qat_people = QToolButton(self.qat_container)
        self.btn_qat_people.setObjectName("QATBtn")
        self.btn_qat_people.setIcon(make_wlpg_icon("person", 18))
        self.btn_qat_people.setIconSize(QSize(18, 18))
        self.btn_qat_people.setFixedSize(24, 24)
        self.btn_qat_people.setToolTip("People View")
        self.btn_qat_people.setCursor(Qt.PointingHandCursor)
        self.btn_qat_people.setVisible(False)
        self.btn_qat_people.clicked.connect(self.qat_people_clicked.emit)
        self.qat_layout.addWidget(self.btn_qat_people)

        # Rotate Left QAT button
        self.btn_qat_rot_left = QToolButton(self.qat_container)
        self.btn_qat_rot_left.setObjectName("QATBtn")
        self.btn_qat_rot_left.setIcon(make_wlpg_icon("rotate_ccw", 18))
        self.btn_qat_rot_left.setIconSize(QSize(18, 18))
        self.btn_qat_rot_left.setFixedSize(24, 24)
        self.btn_qat_rot_left.setToolTip("Rotate left (Ctrl+,)")
        self.btn_qat_rot_left.setCursor(Qt.PointingHandCursor)
        self.btn_qat_rot_left.setVisible(False)
        self.btn_qat_rot_left.clicked.connect(lambda: self.action_triggered.emit("rotate_left"))
        self.qat_layout.addWidget(self.btn_qat_rot_left)

        # Rotate Right QAT button
        self.btn_qat_rot_right = QToolButton(self.qat_container)
        self.btn_qat_rot_right.setObjectName("QATBtn")
        self.btn_qat_rot_right.setIcon(make_wlpg_icon("rotate_cw", 18))
        self.btn_qat_rot_right.setIconSize(QSize(18, 18))
        self.btn_qat_rot_right.setFixedSize(24, 24)
        self.btn_qat_rot_right.setToolTip("Rotate right (Ctrl+.)")
        self.btn_qat_rot_right.setCursor(Qt.PointingHandCursor)
        self.btn_qat_rot_right.setVisible(False)
        self.btn_qat_rot_right.clicked.connect(lambda: self.action_triggered.emit("rotate_right"))
        self.qat_layout.addWidget(self.btn_qat_rot_right)

        # Delete QAT button
        self.btn_qat_delete = QToolButton(self.qat_container)
        self.btn_qat_delete.setObjectName("QATBtn")
        self.btn_qat_delete.setIcon(make_wlpg_icon("delete", 18))
        self.btn_qat_delete.setIconSize(QSize(18, 18))
        self.btn_qat_delete.setFixedSize(24, 24)
        self.btn_qat_delete.setToolTip("Delete (Del)")
        self.btn_qat_delete.setCursor(Qt.PointingHandCursor)
        self.btn_qat_delete.setVisible(False)
        self.btn_qat_delete.clicked.connect(lambda: self.action_triggered.emit("delete"))
        self.qat_layout.addWidget(self.btn_qat_delete)

        # Slide Show QAT button
        self.btn_qat_slideshow = QToolButton(self.qat_container)
        self.btn_qat_slideshow.setObjectName("QATBtn")
        self.btn_qat_slideshow.setIcon(make_wlpg_icon("slideshow", 18))
        self.btn_qat_slideshow.setIconSize(QSize(18, 18))
        self.btn_qat_slideshow.setFixedSize(24, 24)
        self.btn_qat_slideshow.setToolTip("Slide show (F12)")
        self.btn_qat_slideshow.setCursor(Qt.PointingHandCursor)
        self.btn_qat_slideshow.setVisible(False)
        self.btn_qat_slideshow.clicked.connect(lambda: self.action_triggered.emit("slide_show"))
        self.qat_layout.addWidget(self.btn_qat_slideshow)

        self.btn_qat_menu = QToolButton(self.qat_container)
        self.btn_qat_menu.setObjectName("QATBtn")
        self.btn_qat_menu.setIcon(make_wlpg_icon("qat_menu", 14))
        self.btn_qat_menu.setIconSize(QSize(14, 14))
        self.btn_qat_menu.setFixedSize(14, 24)
        self.btn_qat_menu.setToolTip("Customize Quick Access Toolbar")
        self.btn_qat_menu.setCursor(Qt.PointingHandCursor)
        self.btn_qat_menu.setPopupMode(QToolButton.InstantPopup)

        # Build Customize QAT Menu
        qat_menu = QMenu(self.btn_qat_menu)
        qat_menu.setFont(QFont("Segoe UI", 9))

        act_title = qat_menu.addAction("Customize Quick Access Toolbar")
        act_title.setEnabled(False)
        bold_f = QFont(act_title.font())
        bold_f.setBold(True)
        act_title.setFont(bold_f)
        qat_menu.addSeparator()

        self.act_qat_print = qat_menu.addAction("Print")
        self.act_qat_print.setCheckable(True)
        self.act_qat_print.setChecked(False)
        self.act_qat_print.toggled.connect(self.btn_print.setVisible)
        self.act_qat_print.toggled.connect(self._save_qat_settings)

        self.act_qat_back = qat_menu.addAction("Back")
        self.act_qat_back.setCheckable(True)
        self.act_qat_back.setChecked(True)
        self.act_qat_back.toggled.connect(self.btn_back.setVisible)
        self.act_qat_back.toggled.connect(self._save_qat_settings)

        self.act_qat_forward = qat_menu.addAction("Forward")
        self.act_qat_forward.setCheckable(True)
        self.act_qat_forward.setChecked(True)
        self.act_qat_forward.toggled.connect(self.btn_forward.setVisible)
        self.act_qat_forward.toggled.connect(self._save_qat_settings)

        self.act_qat_undo = qat_menu.addAction("Undo")
        self.act_qat_undo.setCheckable(True)
        self.act_qat_undo.setChecked(True)
        self.act_qat_undo.toggled.connect(self.btn_undo.setVisible)
        self.act_qat_undo.toggled.connect(self._save_qat_settings)

        self.act_qat_redo = qat_menu.addAction("Redo")
        self.act_qat_redo.setCheckable(True)
        self.act_qat_redo.setChecked(True)
        self.act_qat_redo.toggled.connect(self.btn_redo.setVisible)
        self.act_qat_redo.toggled.connect(self._save_qat_settings)

        qat_menu.addSeparator()

        self.act_qat_thumbnails = qat_menu.addAction("Gallery View")
        self.act_qat_thumbnails.setCheckable(True)
        self.act_qat_thumbnails.setChecked(False)
        self.act_qat_thumbnails.toggled.connect(self.btn_qat_thumbnails.setVisible)
        self.act_qat_thumbnails.toggled.connect(self._save_qat_settings)

        self.act_qat_map = qat_menu.addAction("World Map View")
        self.act_qat_map.setCheckable(True)
        self.act_qat_map.setChecked(False)
        self.act_qat_map.toggled.connect(self.btn_qat_map.setVisible)
        self.act_qat_map.toggled.connect(self._save_qat_settings)

        self.act_qat_people = qat_menu.addAction("People View")
        self.act_qat_people.setCheckable(True)
        self.act_qat_people.setChecked(False)
        self.act_qat_people.toggled.connect(self.btn_qat_people.setVisible)
        self.act_qat_people.toggled.connect(self._save_qat_settings)

        qat_menu.addSeparator()

        self.act_qat_rot_left = qat_menu.addAction("Rotate Left")
        self.act_qat_rot_left.setCheckable(True)
        self.act_qat_rot_left.setChecked(False)
        self.act_qat_rot_left.toggled.connect(self.btn_qat_rot_left.setVisible)
        self.act_qat_rot_left.toggled.connect(self._save_qat_settings)

        self.act_qat_rot_right = qat_menu.addAction("Rotate Right")
        self.act_qat_rot_right.setCheckable(True)
        self.act_qat_rot_right.setChecked(False)
        self.act_qat_rot_right.toggled.connect(self.btn_qat_rot_right.setVisible)
        self.act_qat_rot_right.toggled.connect(self._save_qat_settings)

        self.act_qat_delete = qat_menu.addAction("Delete")
        self.act_qat_delete.setCheckable(True)
        self.act_qat_delete.setChecked(False)
        self.act_qat_delete.toggled.connect(self.btn_qat_delete.setVisible)
        self.act_qat_delete.toggled.connect(self._save_qat_settings)

        self.act_qat_slideshow = qat_menu.addAction("Slide Show")
        self.act_qat_slideshow.setCheckable(True)
        self.act_qat_slideshow.setChecked(False)
        self.act_qat_slideshow.toggled.connect(self.btn_qat_slideshow.setVisible)
        self.act_qat_slideshow.toggled.connect(self._save_qat_settings)

        qat_menu.addSeparator()

        self.act_qat_position = qat_menu.addAction("Show below the Ribbon")
        self.act_qat_position.triggered.connect(self._toggle_qat_position)

        self.act_qat_minimize = qat_menu.addAction("Minimize the Ribbon")
        self.act_qat_minimize.setCheckable(True)
        self.act_qat_minimize.setChecked(False)
        self.act_qat_minimize.triggered.connect(self._toggle_ribbon_collapse)

        qat_menu.addSeparator()
        self.act_qat_reset = qat_menu.addAction("Reset Quick Access Toolbar")
        self.act_qat_reset.triggered.connect(self.reset_qat_to_defaults)

        self.btn_qat_menu.setMenu(qat_menu)
        self.qat_layout.addWidget(self.btn_qat_menu)

        # Above Ribbon Header Bar (Integrated Custom Title Bar)
        self.top_header = _RibbonTitleBar(self)
        self.top_hbox = QHBoxLayout(self.top_header)
        self.top_hbox.setContentsMargins(6, 0, 0, 0)
        self.top_hbox.setSpacing(3)

        self.lbl_app_icon = QLabel(self.top_header)
        self.lbl_app_icon.setObjectName("RibbonAppIcon")
        self.lbl_app_icon.setFixedSize(22, 22)
        self.lbl_app_icon.setPixmap(make_wlpg_icon("app_logo", 20).pixmap(20, 20))
        self.lbl_app_icon.setAlignment(Qt.AlignCenter)
        self.lbl_app_icon.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.top_hbox.addWidget(self.lbl_app_icon)

        self.header_div_icon = QFrame(self.top_header)
        self.header_div_icon.setObjectName("RibbonHeaderDiv")
        self.header_div_icon.setFrameShape(QFrame.VLine)
        self.header_div_icon.setFrameShadow(QFrame.Plain)
        self.header_div_icon.setFixedSize(1, 16)
        self.header_div_icon.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.top_hbox.addWidget(self.header_div_icon)

        self.top_hbox.addWidget(self.qat_container)

        self.header_div = QFrame(self.top_header)
        self.header_div.setObjectName("RibbonHeaderDiv")
        self.header_div.setFrameShape(QFrame.VLine)
        self.header_div.setFrameShadow(QFrame.Plain)
        self.header_div.setFixedSize(1, 16)
        self.header_div.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.top_hbox.addWidget(self.header_div)

        self.lbl_app_title = QLabel("Essentials Photo Gallery", self.top_header)
        self.lbl_app_title.setObjectName("RibbonAppTitle")
        self.lbl_app_title.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.top_hbox.addWidget(self.lbl_app_title)

        self.top_hbox.addStretch(1)

        self.btn_win_min = _WindowControlButton("min", self.top_header)
        self.btn_win_min.setObjectName("WinMinBtn")
        self.btn_win_min.setFixedSize(46, 32)
        self.btn_win_min.setToolTip("Minimize")
        self.btn_win_min.setCursor(Qt.ArrowCursor)
        self.btn_win_min.clicked.connect(self._on_win_min_clicked)
        self.top_hbox.addWidget(self.btn_win_min)

        self.btn_win_max = _WindowControlButton("max", self.top_header)
        self.btn_win_max.setObjectName("WinMaxBtn")
        self.btn_win_max.setFixedSize(46, 32)
        self.btn_win_max.setToolTip("Maximize")
        self.btn_win_max.setCursor(Qt.ArrowCursor)
        self.btn_win_max.clicked.connect(self._on_win_max_clicked)
        self.top_hbox.addWidget(self.btn_win_max)

        self.btn_win_close = _WindowControlButton("close", self.top_header)
        self.btn_win_close.setObjectName("WinCloseBtn")
        self.btn_win_close.setFixedSize(46, 32)
        self.btn_win_close.setToolTip("Close (Alt+F4)")
        self.btn_win_close.setCursor(Qt.ArrowCursor)
        self.btn_win_close.clicked.connect(self._on_win_close_clicked)
        self.top_hbox.addWidget(self.btn_win_close)

        main_vbox.addWidget(self.top_header)

        # 1. 5-Tab Ribbon Widget
        self.tab_widget = QTabWidget(self)
        self.tab_widget.setTabBar(_RibbonTabBar(self.tab_widget))
        self.tab_widget.tabBar().setDrawBase(False)
        self.tab_widget.setObjectName("RibbonTabs")
        self.tab_widget.setFixedHeight(120)

        # Blue File Button in TopLeftCorner
        self.btn_file = QPushButton("File", self.tab_widget)
        self.btn_file.setObjectName("RibbonFileBtn")
        self.btn_file.setCursor(Qt.PointingHandCursor)
        file_menu = QMenu(self.btn_file)
        file_menu.setObjectName("RibbonFileMenu")
        file_menu.setMinimumWidth(260)
        self.file_menu = file_menu
        self._apply_file_menu_stylesheet(False)
        file_menu.addAction(make_wlpg_icon("import", 16), "Import from camera or scanner...", lambda: self.action_triggered.emit("import_media"))
        file_menu.addAction(make_wlpg_icon("folder_add", 16), "Include folder...", lambda: self.action_triggered.emit("add_folder"))
        file_menu.addAction(make_wlpg_icon("print", 16), "Print...\tCtrl+P", lambda: self.action_triggered.emit("print"))
        file_menu.addSeparator()
        file_menu.addAction("Backup Catalog...", lambda: self.action_triggered.emit("backup_catalog"))
        self.act_pause_write_back = QAction("Pause Background File Write-Back", self)
        self.act_pause_write_back.setCheckable(True)
        self.act_pause_write_back.setChecked(True)
        self.act_pause_write_back.triggered.connect(lambda: self.action_triggered.emit("toggle_pause_write_back"))
        file_menu.addAction(self.act_pause_write_back)
        file_menu.addSeparator()
        file_menu.addAction(make_wlpg_icon("options", 16), "Options...", lambda: self.action_triggered.emit("open_settings"))
        file_menu.addAction(make_wlpg_icon("app_logo", 16), "About Essentials Photo Gallery", lambda: self.action_triggered.emit("about_photo_gallery"))
        file_menu.addSeparator()
        file_menu.addAction("Exit", lambda: self.action_triggered.emit("exit_app"))
        self.btn_file.setMenu(file_menu)
        self.tab_widget.setCornerWidget(self.btn_file, Qt.TopLeftCorner)

        self.tab_home = self._build_home_tab()
        self.tab_edit = self._build_edit_tab()
        self.tab_find = self._build_find_tab()
        self.tab_create = self._build_create_tab()
        self.tab_view = self._build_view_tab()
        self.tab_batch_people = self._build_batch_people_tab()

        self.tab_widget.addTab(self.tab_home, "Home")
        self.tab_widget.addTab(self.tab_edit, "Edit")
        self.tab_widget.addTab(self.tab_find, "Find")
        self.tab_widget.addTab(self.tab_create, "Create")
        self.tab_widget.addTab(self.tab_view, "View")
        self.tab_widget.addTab(self.tab_batch_people, "Batch people tagging")
        self.tab_widget.tabBar().setTabVisible(5, False)

        self.tab_widget.tabBarDoubleClicked.connect(self._on_tab_bar_double_clicked)
        self.tab_widget.tabBarClicked.connect(self._on_tab_bar_clicked)

        # Right Corner: Minimize the Ribbon button and Exit Immersive Pill
        right_corner = QWidget(self.tab_widget)
        rc_layout = QHBoxLayout(right_corner)
        rc_layout.setContentsMargins(2, 0, 8, 0)
        rc_layout.setSpacing(4)

        # Exit Immersive Mode Button (subtle pill, hidden by default)
        self.btn_exit_immersive = QPushButton("✕ Exit Immersive (F4)", right_corner)
        self.btn_exit_immersive.setFixedHeight(22)
        self.btn_exit_immersive.setCursor(Qt.PointingHandCursor)
        self.btn_exit_immersive.setToolTip("Exit Immersive Viewer Mode (F4)")
        self.btn_exit_immersive.setStyleSheet(
            """
            QPushButton {
                border: 1px solid #70A6E8;
                border-radius: 3px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FFFFFF, stop:0.4 #E8F1FC, stop:1 #D2E4F8);
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
                font-weight: 600;
                padding: 1px 8px;
                margin-right: 4px;
            }
            QPushButton:hover {
                border: 1px solid #3399FF;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FFFFFF, stop:0.4 #D9EBFC, stop:1 #B9DAFB);
            }
            QPushButton:pressed {
                background: #A0C8F5;
                border: 1px solid #2272D0;
            }
            """
        )
        self.btn_exit_immersive.clicked.connect(lambda: self.action_triggered.emit("toggle_immersive"))
        self.btn_exit_immersive.hide()
        rc_layout.addWidget(self.btn_exit_immersive)

        self.btn_collapse_ribbon = QPushButton("▲", right_corner)
        self.btn_collapse_ribbon.setFixedSize(22, 22)
        self.btn_collapse_ribbon.setFlat(True)
        self.btn_collapse_ribbon.setToolTip("Minimize the Ribbon (Ctrl+F1)")
        self.btn_collapse_ribbon.setCursor(Qt.PointingHandCursor)
        self.btn_collapse_ribbon.setStyleSheet(
            """
            QPushButton {
                border: 1px solid transparent;
                color: #4B6B8B;
                font-size: 8pt;
                border-radius: 2px;
                padding: 0px;
            }
            QPushButton:hover {
                background: #D9E8F7;
                border: 1px solid #AFC8E2;
            }
            """
        )
        self.btn_collapse_ribbon.clicked.connect(self._toggle_ribbon_collapse)
        rc_layout.addWidget(self.btn_collapse_ribbon)

        self.tab_widget.setCornerWidget(right_corner, Qt.TopRightCorner)

        main_vbox.addWidget(self.tab_widget)

        # 1.5. Dedicated Quick Access Toolbar Below Ribbon (Authentic WLPG Option)
        self.below_ribbon_bar = QWidget(self)
        self.below_ribbon_bar.setObjectName("RibbonBelowQAT")
        self.below_ribbon_bar.setFixedHeight(25)
        self.below_ribbon_layout = QHBoxLayout(self.below_ribbon_bar)
        self.below_ribbon_layout.setContentsMargins(6, 1, 6, 1)
        self.below_ribbon_layout.setSpacing(2)
        self.below_ribbon_layout.addStretch(1)
        self.below_ribbon_bar.hide()
        main_vbox.addWidget(self.below_ribbon_bar)
        self.scope_bar = QFrame(self)
        self.scope_bar.setObjectName("SearchScopeBar")
        self.scope_bar.setFixedHeight(30)
        scope_hbox = QHBoxLayout(self.scope_bar)
        scope_hbox.setContentsMargins(10, 2, 10, 2)
        scope_hbox.setSpacing(8)

        lbl_prefix = QLabel("Search by:", self.scope_bar)
        lbl_prefix.setStyleSheet(
            "color: #55687C; font-weight: 600; font-family: 'Segoe UI'; font-size: 9pt;"
        )
        scope_hbox.addWidget(lbl_prefix)

        self.lbl_scope = QLabel("All photos and videos", self.scope_bar)
        self.lbl_scope.setStyleSheet(
            "color: #1E395B; font-weight: 600; font-family: 'Segoe UI'; font-size: 9pt;"
        )
        scope_hbox.addWidget(self.lbl_scope)

        self.btn_search_all = QPushButton("Search all", self.scope_bar)
        self.btn_search_all.setFlat(True)
        self.btn_search_all.setCursor(Qt.PointingHandCursor)
        self.btn_search_all.setStyleSheet(
            "color: #0066CC; text-decoration: underline; border: none; padding: 0 4px; font-family: 'Segoe UI'; font-size: 8.5pt;"
        )
        self.btn_search_all.clicked.connect(self.search_all_requested.emit)
        scope_hbox.addWidget(self.btn_search_all)

        scope_hbox.addStretch(1)

        self.lbl_search_tag = QLabel("Search:", self.scope_bar)
        self.lbl_search_tag.setStyleSheet(
            "color: #55687C; font-family: 'Segoe UI'; font-size: 9pt;"
        )
        scope_hbox.addWidget(self.lbl_search_tag)

        self.txt_search = QLineEdit(self.scope_bar)
        self.txt_search.setObjectName("ScopeSearchInput")
        self.txt_search.setPlaceholderText("Search photos and videos...")
        self.txt_search.setClearButtonEnabled(True)
        self.txt_search.setFixedWidth(220)
        self.txt_search.setFixedHeight(22)
        self.txt_search.textChanged.connect(self._on_search_input_changed)
        scope_hbox.addWidget(self.txt_search)

        self.btn_close_scope = QPushButton("✕", self.scope_bar)
        self.btn_close_scope.setObjectName("ScopeCloseBtn")
        self.btn_close_scope.setToolTip("Close search")
        self.btn_close_scope.setFixedSize(22, 22)
        self.btn_close_scope.setCursor(Qt.PointingHandCursor)
        self.btn_close_scope.clicked.connect(self.hide_search_scope)
        scope_hbox.addWidget(self.btn_close_scope)

        main_vbox.addWidget(self.scope_bar)

        for w in (self, self.top_header, self.tab_widget, self.tab_widget.tabBar(), self.below_ribbon_bar,
                  self.tab_home, self.tab_edit, self.tab_find, self.tab_create, self.tab_view, self.tab_batch_people):
            w.installEventFilter(self)

        self._load_qat_settings()
        self._apply_aero_stylesheet()

    def remove_embedded_scope_bar(self) -> None:
        """Removes the embedded ribbon scope bar so only the central one above the grid is used."""
        if hasattr(self, "scope_bar") and self.scope_bar is not None:
            self.scope_bar.hide()
            self.scope_bar.setVisible(False)
            if hasattr(self, "main_vbox") and self.main_vbox is not None:
                self.main_vbox.removeWidget(self.scope_bar)

    def hide_search_scope(self) -> None:
        if hasattr(self, "scope_bar") and self.scope_bar is not None:
            self.scope_bar.setVisible(False)
        if hasattr(self, "txt_search") and self.txt_search is not None:
            try:
                self.txt_search.clear()
            except RuntimeError:
                pass

    def show_search_scope(self) -> None:
        if hasattr(self, "scope_bar") and self.scope_bar is not None:
            self.scope_bar.setVisible(True)
        if hasattr(self, "txt_search") and self.txt_search is not None:
            try:
                self.txt_search.setFocus()
                self.txt_search.selectAll()
            except RuntimeError:
                pass

    def toggle_search_scope(self) -> None:
        if hasattr(self, "scope_bar") and self.scope_bar is not None and self.scope_bar.isVisible():
            self.hide_search_scope()
        else:
            self.show_search_scope()

    def _set_button_compact(self, btn: QToolButton | QPushButton, compact: bool) -> None:
        container = getattr(btn, "container_widget", None)
        orig = getattr(btn, "original_text", getattr(btn, "_orig_text", btn.text()))
        btn.original_text = orig
        btn._orig_text = orig
        is_large = getattr(btn, "_is_large", False) or (isinstance(btn, QToolButton) and btn.objectName() == "LargeRibbonBtn")

        if compact:
            if container is not None:
                if hasattr(btn, "btn_bottom"):
                    btn.btn_bottom.hide()
                container.setFixedSize(34, 73)
                btn.setFixedSize(34, 73)
                btn.setText("")
                if isinstance(btn, QToolButton):
                    btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
            else:
                if isinstance(btn, QToolButton):
                    btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
                btn.setText("")
                has_menu = False
                if isinstance(btn, QToolButton):
                    has_menu = btn.menu() is not None or btn.popupMode() in (
                        QToolButton.MenuButtonPopup,
                        QToolButton.InstantPopup,
                    )
                if is_large:
                    btn.setFixedSize(34, 70)
                else:
                    w = 32 if has_menu else 24
                    btn.setIconSize(QSize(16, 16))
                    btn.setFixedSize(w, 22)
                    if has_menu:
                        btn.setStyleSheet("QToolButton { padding: 1px 12px 1px 2px; }")
                    else:
                        btn.setStyleSheet("QToolButton, QPushButton { padding: 1px 2px 1px 2px; }")
            tip = getattr(btn, "_tooltip_override", None) or orig.replace("\n", " ")
            btn.setToolTip(tip)
        else:
            if container is not None:
                orig_w = getattr(container, "_orig_w", 80)
                container.setFixedSize(orig_w, 73)
                btn.setFixedSize(orig_w, 38)
                btn.setText("")
                if isinstance(btn, QToolButton):
                    btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
                if hasattr(btn, "btn_bottom"):
                    btn.btn_bottom.setFixedSize(orig_w, 35)
                    btn.btn_bottom.show()
            else:
                if isinstance(btn, QToolButton):
                    btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon if is_large else Qt.ToolButtonTextBesideIcon)
                btn.setText(orig)
                btn.setStyleSheet("")
                btn.setIconSize(QSize(32, 32) if is_large else QSize(16, 16))
                if is_large:
                    btn.setFixedHeight(70)
                    orig_w = getattr(btn, "_orig_min_w", 50)
                    btn.setFixedWidth(orig_w)
                else:
                    btn.setFixedHeight(22)
                    btn.setMinimumWidth(btn.sizeHint().width())
                    btn.setMaximumWidth(16777215)
            tip = getattr(btn, "_tooltip_override", None) or orig.replace("\n", " ")
            btn.setToolTip(tip)

    def _make_tool_btn(
        self,
        text: str,
        icon_kind: str,
        parent: QWidget,
        large: bool = True,
        popup_mode: QToolButton.ToolButtonPopupMode = QToolButton.InstantPopup,
    ) -> QToolButton:
        if large:
            btn = _RibbonLargeToolButton(parent)
            btn.setObjectName("LargeRibbonToolBtn")
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn.setIcon(make_wlpg_icon(icon_kind, 32))
            btn.setIconSize(QSize(32, 32))
            btn.setText(text)
            btn._has_menu = True
            btn._is_dark = getattr(self, "_dark_mode", False)
            fm = btn.fontMetrics()
            lines = text.split("\n")
            max_w = max((fm.horizontalAdvance(line) for line in lines), default=36)
            btn._orig_min_w = max(46, min(76, max_w + 10))
            btn.setFixedWidth(btn._orig_min_w)
            btn.setFixedHeight(70)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self._responsive_large_buttons.append(btn)
        else:
            if popup_mode == QToolButton.MenuButtonPopup:
                btn = _RibbonSplitToolButton(parent)
                btn._theme_mode = getattr(self, "_current_theme_mode", "aero")
                btn._is_dark = getattr(self, "_dark_mode", False)
            else:
                btn = QToolButton(parent)
                btn.setObjectName("CompactRibbonToolBtn")
            btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            btn.setIcon(make_wlpg_icon(icon_kind, 16))
            btn.setIconSize(QSize(16, 16))
            btn.setText(text)
            btn.setFixedHeight(21)
            btn.setMinimumWidth(btn.sizeHint().width())
            self._responsive_secondary_buttons.append(btn)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setPopupMode(popup_mode)
        btn.original_text = text
        btn._orig_text = text
        btn.setToolTip(text.replace("\n", " "))
        setattr(btn, "set_compact", lambda compact, b=btn: self._set_button_compact(b, compact))
        btn._icon_kind = icon_kind
        btn._is_large = large
        if hasattr(self, "_all_themed_buttons"):
            self._all_themed_buttons.append(btn)
        return btn

    def _make_slideshow_split_button(self, parent: QWidget) -> tuple[QWidget, QToolButton]:
        """
        Creates an authentic WLPG 2-button vertical split slideshow control:
        - Top button: Projector icon only -> clicking immediately plays the default slideshow (F12).
        - Bottom button: 'Slide\\nshow ▾' text with dropdown arrow in text -> clicking immediately opens the theme menu.
        """
        container = QWidget(parent)
        container.setObjectName("SlideshowSplitContainer")
        container.setFixedSize(54, 73)
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # 1. Top button (icon only)
        btn_top = QToolButton(container)
        btn_top.setObjectName("SlideshowTopBtn")
        btn_top.setFixedSize(54, 38)
        btn_top.setCursor(Qt.PointingHandCursor)
        btn_top.setIcon(make_wlpg_icon("slideshow", 32))
        btn_top.setIconSize(QSize(28, 28))
        btn_top.setToolTip("Slide show (F12)")
        btn_top._tooltip_override = "Slide show (F12)"
        btn_top.clicked.connect(lambda: self.action_triggered.emit("slide_show"))
        vbox.addWidget(btn_top)

        # 2. Bottom button (text: Slide\\nshow ▾)
        btn_bot = QToolButton(container)
        btn_bot.setObjectName("SlideshowBottomBtn")
        btn_bot.setFixedSize(54, 35)
        btn_bot.setCursor(Qt.PointingHandCursor)
        btn_bot.setText("Slide\nshow ▾")
        btn_bot.setToolTip("Slide show styles and themes")
        btn_bot.setPopupMode(QToolButton.InstantPopup)
        btn_bot.setMenu(self._build_slideshow_theme_menu())
        vbox.addWidget(btn_bot)

        # Attach delegators onto btn_top for backwards compatibility
        btn_top.menu = btn_bot.menu
        btn_top.setMenu = btn_bot.setMenu
        btn_top.popupMode = btn_bot.popupMode
        btn_top.container_widget = container
        btn_top.btn_bottom = btn_bot
        container._orig_w = 54
        if hasattr(self, "_all_themed_buttons"):
            btn_top._icon_kind = "slideshow"
            btn_top._is_large = True
            self._all_themed_buttons.append(btn_top)

        return container, btn_top

    def _make_split_save_copy_button(self, parent: QWidget) -> tuple[QWidget, QToolButton]:
        """
        Creates an authentic WLPG 2-button vertical split Save As Copy control:
        - Top button: 32px make_copy icon -> clicking triggers 'save_as_copy' (default behavior / Ctrl+Shift+S).
        - Bottom button: 'Save as\\ncopy ▾' text -> opens menu:
            - Save in original location
            - Choose Location...
            - Options...
        """
        container = QWidget(parent)
        container.setObjectName("SaveCopySplitContainer")
        container.setFixedSize(62, 73)
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # 1. Top button (icon only)
        btn_top = QToolButton(container)
        btn_top.setObjectName("SaveCopyTopBtn")
        btn_top.setFixedSize(62, 38)
        btn_top.setCursor(Qt.PointingHandCursor)
        btn_top.setIcon(make_wlpg_icon("make_copy", 32))
        btn_top.setIconSize(QSize(28, 28))
        btn_top.setToolTip("Make a copy (Ctrl+Shift+S)\nSave a new copy with your adjustments")
        btn_top._tooltip_override = "Make a copy (Ctrl+Shift+S)"
        btn_top.clicked.connect(lambda: self.action_triggered.emit("save_as_copy"))
        vbox.addWidget(btn_top)

        # 2. Bottom button (text: Save as\\ncopy ▾)
        btn_bot = QToolButton(container)
        btn_bot.setObjectName("SaveCopyBottomBtn")
        btn_bot.setFixedSize(62, 35)
        btn_bot.setCursor(Qt.PointingHandCursor)
        btn_bot.setText("Save as\ncopy ▾")
        btn_bot.setToolTip("Save as copy options")
        btn_bot.setPopupMode(QToolButton.InstantPopup)

        menu = QMenu(btn_bot)
        act_orig = menu.addAction(make_wlpg_icon("make_copy", 16), "Save in original location")
        act_orig.triggered.connect(lambda: self.action_triggered.emit("save_as_copy_original"))

        act_choose = menu.addAction(make_wlpg_icon("folder_open", 16), "Choose Location...")
        act_choose.triggered.connect(lambda: self.action_triggered.emit("save_as_copy_choose_location"))

        menu.addSeparator()
        act_opt = menu.addAction(make_wlpg_icon("options", 16), "Options...")
        act_opt.triggered.connect(lambda: self.action_triggered.emit("open_options_edit"))

        btn_bot.setMenu(menu)
        vbox.addWidget(btn_bot)

        btn_top.menu = btn_bot.menu
        btn_top.setMenu = btn_bot.setMenu
        btn_top.popupMode = btn_bot.popupMode
        btn_top.container_widget = container
        btn_top.btn_bottom = btn_bot
        container._orig_w = 62
        if hasattr(self, "_all_themed_buttons"):
            btn_top._icon_kind = "make_copy"
            btn_top._is_large = True
            self._all_themed_buttons.append(btn_top)

        return container, btn_top

    def _make_split_tag_desc_button(self, parent: QWidget) -> tuple[QWidget, QToolButton]:
        """
        Creates an authentic WLPG 2-button vertical split Descriptive tag control:
        - Top button: Tag icon only -> clicking immediately focuses descriptive tag input (Ctrl+T).
        - Bottom button: 'Descriptive\\ntag ▾' text with dropdown arrow in text -> clicking immediately opens tag menu.
        """
        container = QWidget(parent)
        container.setObjectName("TagDescSplitContainer")
        container.setFixedSize(80, 73)
        container._orig_w = 80
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # 1. Top button (icon only)
        btn_top = QToolButton(container)
        btn_top.setObjectName("TagDescTopBtn")
        btn_top.setFixedSize(80, 38)
        btn_top.setCursor(Qt.PointingHandCursor)
        btn_top.setIcon(make_wlpg_icon("tag", 32))
        btn_top.setIconSize(QSize(28, 28))
        btn_top.setToolButtonStyle(Qt.ToolButtonIconOnly)
        btn_top.setText("")
        btn_top.setToolTip("Descriptive tag (Ctrl+T)")
        btn_top._tooltip_override = "Descriptive tag (Ctrl+T)"
        btn_top.clicked.connect(lambda: self.action_triggered.emit("focus_tag_box"))
        vbox.addWidget(btn_top)

        # 2. Bottom button (text: Descriptive\ntag ▾)
        btn_bot = QToolButton(container)
        btn_bot.setObjectName("TagDescBottomBtn")
        btn_bot.setFixedSize(80, 35)
        btn_bot.setCursor(Qt.PointingHandCursor)
        btn_bot.setText("Descriptive\ntag ▾")
        btn_bot.setToolTip("Descriptive tag (Ctrl+T)")
        btn_bot.setPopupMode(QToolButton.InstantPopup)
        btn_bot.setMenu(self._build_descriptive_tag_menu(btn_bot))
        vbox.addWidget(btn_bot)

        # Attach delegators onto btn_top for backwards compatibility
        btn_top.menu = btn_bot.menu
        btn_top.setMenu = btn_bot.setMenu
        btn_top.popupMode = btn_bot.popupMode
        btn_top.container_widget = container
        btn_top.btn_bottom = btn_bot
        btn_top.original_text = ""
        btn_top._orig_text = ""

        if hasattr(self, "_all_themed_buttons"):
            btn_top._icon_kind = "tag"
            btn_top._is_large = True
            self._all_themed_buttons.append(btn_top)

        return container, btn_top

    def _build_descriptive_tag_menu(self, parent: QWidget | None = None) -> QMenu:
        menu = QMenu(parent or self)

        act_tag = menu.addAction(make_wlpg_icon("tag", 16), "Tag an item\tCtrl+T")
        act_tag.triggered.connect(lambda: self.action_triggered.emit("focus_tag_box"))

        menu.addSeparator()

        act_manage = menu.addAction(make_wlpg_icon("tag_hierarchy", 16), "Manage tags...")
        act_manage.triggered.connect(lambda: self.action_triggered.emit("show_tag_hierarchy"))

        menu_cleanup = QMenu("Clean up tags", menu)
        menu_cleanup.setIcon(make_wlpg_icon("tag", 16))
        menu.addMenu(menu_cleanup)

        act_cleanup = menu_cleanup.addAction(make_wlpg_icon("tag", 16), "Clean up unused tags")
        act_cleanup.triggered.connect(lambda: self.action_triggered.emit("delete_unused_tags"))

        act_cleanup_orphan = menu_cleanup.addAction(make_wlpg_icon("tag", 16), "Clean up duplicate root tags...")
        act_cleanup_orphan.triggered.connect(lambda: self.action_triggered.emit("merge_orphan_tags"))

        act_restore = menu_cleanup.addAction(make_wlpg_icon("tag", 16), "Restore last cleaned tags...")
        act_restore.triggered.connect(lambda: self.action_triggered.emit("restore_orphan_tags"))

        return menu

    def _make_mode_toggle_btn(
        self,
        text: str,
        parent: QWidget,
        height: int = 21,
        icon_kind: str | None = None,
    ) -> QPushButton:
        btn = QPushButton(text, parent)
        btn.setObjectName("RibbonModeToggleBtn")
        btn.setCheckable(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(height)
        if icon_kind:
            btn.setIcon(make_wlpg_icon(icon_kind, 12))
            btn.setIconSize(QSize(12, 12))
            btn._icon_kind = icon_kind
            btn._is_large = False
            if hasattr(self, "_all_themed_buttons"):
                self._all_themed_buttons.append(btn)
        return btn

    def _make_btn(
        self,
        text: str,
        icon_kind: str,
        parent: QWidget,
        large: bool = False,
        checkable: bool = False,
    ) -> QToolButton:
        if large:
            btn = _RibbonLargeToolButton(parent)
            btn.setObjectName("LargeRibbonBtn")
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn.setIcon(make_wlpg_icon(icon_kind, 32))
            btn.setIconSize(QSize(32, 32))
            btn.setText(text)
            btn._is_dark = getattr(self, "_dark_mode", False)
            fm = btn.fontMetrics()
            lines = text.split("\n")
            max_w = max((fm.horizontalAdvance(line) for line in lines), default=36)
            btn._orig_min_w = max(46, min(76, max_w + 10))
            btn.setFixedWidth(btn._orig_min_w)
            btn.setFixedHeight(70)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self._responsive_large_buttons.append(btn)
        else:
            btn = QToolButton(parent)
            btn.setObjectName("CompactRibbonBtn")
            btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            btn.setIcon(make_wlpg_icon(icon_kind, 16))
            btn.setIconSize(QSize(16, 16))
            btn.setText(text)
            btn.setFixedHeight(21)
            btn.setMinimumWidth(btn.sizeHint().width())
            self._responsive_secondary_buttons.append(btn)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFocusPolicy(Qt.NoFocus)
        btn.setCheckable(checkable)
        btn.original_text = text
        btn._orig_text = text
        btn.setToolTip(text.replace("\n", " "))
        setattr(btn, "set_compact", lambda compact, b=btn: self._set_button_compact(b, compact))
        btn._icon_kind = icon_kind
        btn._is_large = large
        if hasattr(self, "_all_themed_buttons"):
            self._all_themed_buttons.append(btn)
        return btn

    def refresh_icons(self, _theme_name: str = "") -> None:
        """Dynamically refreshes all ribbon button icons when the icon theme changes."""
        self.setUpdatesEnabled(False)
        try:
            for btn in getattr(self, "_all_themed_buttons", []):
                kind = getattr(btn, "_icon_kind", None)
                if kind:
                    size = 32 if getattr(btn, "_is_large", False) else 16
                    btn.setIcon(make_wlpg_icon(kind, size))
                    btn.setIconSize(QSize(size, size))
            if hasattr(self, "btn_back"):
                self.btn_back.setIcon(make_wlpg_icon("nav_back", 20))
            if hasattr(self, "btn_forward"):
                self.btn_forward.setIcon(make_wlpg_icon("nav_forward", 20))
            if hasattr(self, "btn_undo"):
                self.btn_undo.setIcon(make_wlpg_icon("undo", 18))
            if hasattr(self, "btn_redo"):
                self.btn_redo.setIcon(make_wlpg_icon("redo", 18))
            if hasattr(self, "btn_print"):
                self.btn_print.setIcon(make_wlpg_icon("print", 18))
            if hasattr(self, "btn_qat_thumbnails"):
                self.btn_qat_thumbnails.setIcon(make_wlpg_icon("thumbnail_view", 18))
            if hasattr(self, "btn_qat_map"):
                self.btn_qat_map.setIcon(make_wlpg_icon("globe", 18))
            if hasattr(self, "btn_qat_people"):
                self.btn_qat_people.setIcon(make_wlpg_icon("person", 18))
            if hasattr(self, "btn_qat_rot_left"):
                self.btn_qat_rot_left.setIcon(make_wlpg_icon("rotate_ccw", 18))
            if hasattr(self, "btn_qat_rot_right"):
                self.btn_qat_rot_right.setIcon(make_wlpg_icon("rotate_cw", 18))
            if hasattr(self, "btn_qat_delete"):
                self.btn_qat_delete.setIcon(make_wlpg_icon("delete", 18))
            if hasattr(self, "btn_qat_slideshow"):
                self.btn_qat_slideshow.setIcon(make_wlpg_icon("slideshow", 18))
            if hasattr(self, "lbl_app_icon"):
                self.lbl_app_icon.setPixmap(make_wlpg_icon("app_logo", 20).pixmap(20, 20))
        finally:
            self.setUpdatesEnabled(True)

    # -------------------------------------------------------------------------
    # Responsive Resizing & Layout Breakpoints (WP-07)
    # -------------------------------------------------------------------------
    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())

    def _apply_responsive_layout(self, width: int) -> None:
        """
        Authentic WLPG priority collapse order across all tabs:

        Home Tab:
        - width >= 1220: 5-6 faces in Quick find; all buttons Expanded.
        - 1050 <= width < 1220: Shrink Quick find to 2-3 faces; collapse Months / Rated / Flagged to Icon-Only.
        - 880 <= width < 1050: Shrink Quick find to 1 face; collapse Rotate left / Rotate right / Delete
          to Icon-Only while keeping Organize and Select all / Copy / Paste Expanded with full text!
        - width < 880: Collapse Geotag / Rate / Flag and Select all / Copy / Paste to Icon-Only.
        - width < 740: Compact Batch people tag / Descriptive tag / Caption to Icon-Only.

        Edit Tab:
        - Shrink visible Effects tiles: 6 (>=1220) -> 4 (1050-1219) -> 2 (<1050).
        - Noise reduction / Color / Exposure & Rotate left/right/Delete collapse to Icon-Only at < 1050.
        - Rate / Flag / Properties & Select all/copy/paste collapse to Icon-Only at < 880.

        Find Tab:
        - People face strip: 5 (>=1220) -> 3 (1050-1219) -> 1 (<1050).
        - Edited / Media type collapse to Icon-Only at < 1050.

        View Tab:
        - Show details checkboxes collapse to Icon-Only at < 1150.
        - Arrange list buttons collapse to Icon-Only at < 980.
        """
        # --- 1. Large Button Scaling (Always vertical, authentic WLPG style) ---
        for btn in self._responsive_large_buttons:
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn.setFixedHeight(73)
            if hasattr(btn, "_orig_min_w"):
                btn.setMinimumWidth(btn._orig_min_w if width >= 800 else 44)
            btn.setText(getattr(btn, "_orig_text", btn.text()))

        # --- 2. Home Tab Priority Collapse ---
        if hasattr(self, "home_face_carousel"):
            if width >= 1550:
                self.home_face_carousel.set_visible_faces(10)
            elif width >= 1450:
                self.home_face_carousel.set_visible_faces(9)
            elif width >= 1370:
                self.home_face_carousel.set_visible_faces(8)
            elif width >= 1290:
                self.home_face_carousel.set_visible_faces(7)
            elif width >= 1220:
                self.home_face_carousel.set_visible_faces(6)
            elif width >= 1150:
                self.home_face_carousel.set_visible_faces(5)
            elif width >= 1070:
                self.home_face_carousel.set_visible_faces(4)
            elif width >= 990:
                self.home_face_carousel.set_visible_faces(3)
            elif width >= 900:
                self.home_face_carousel.set_visible_faces(2)
            else:
                self.home_face_carousel.set_visible_faces(1)

        # Months / Rated / Flagged
        for btn in getattr(self, "home_qf_buttons", []):
            self._set_button_compact(btn, width < 1220)

        # Rotate left / Rotate right / Delete
        for btn in getattr(self, "home_manage_rot_delete", []):
            self._set_button_compact(btn, width < 1050)

        # Select all / Copy / Paste
        for btn in getattr(self, "home_manage_select_copy_paste", []):
            self._set_button_compact(btn, width < 880)

        # Geotag / Rate / Flag
        for btn in getattr(self, "home_org_geotag_rate_flag", []):
            self._set_button_compact(btn, width < 880)

        # Batch people tag / Descriptive tag / Caption
        for btn in getattr(self, "home_org_batch_desc_caption", []):
            self._set_button_compact(btn, width < 740)

        # --- 3. Edit Tab Priority Collapse ---
        if hasattr(self, "effects_gallery"):
            if width >= 1380:
                self.effects_gallery.set_visible_tiles(6)
            elif width >= 1240:
                self.effects_gallery.set_visible_tiles(5)
            elif width >= 1060:
                self.effects_gallery.set_visible_tiles(4)
            elif width >= 900:
                self.effects_gallery.set_visible_tiles(3)
            elif width >= 750:
                self.effects_gallery.set_visible_tiles(2)
            else:
                self.effects_gallery.set_visible_tiles(1)

        for btn in getattr(self, "edit_redeye_retouch_straighten", []):
            self._set_button_compact(btn, width < 1180)

        for btn in getattr(self, "edit_noise_color_exposure", []):
            self._set_button_compact(btn, width < 1180)

        for btn in getattr(self, "edit_manage_rot_delete", []):
            self._set_button_compact(btn, width < 1180)

        for btn in getattr(self, "edit_manage_select_copy_paste", []):
            self._set_button_compact(btn, width < 880)

        for btn in getattr(self, "edit_org_rate_flag_props", []):
            self._set_button_compact(btn, width < 880)

        for btn in getattr(self, "edit_grid_properties", []):
            self._set_button_compact(btn, width < 880)

        # --- 4. Find Tab Priority Collapse ---
        if hasattr(self, "find_face_carousel"):
            if width >= 1550:
                self.find_face_carousel.set_visible_faces(10)
            elif width >= 1450:
                self.find_face_carousel.set_visible_faces(9)
            elif width >= 1370:
                self.find_face_carousel.set_visible_faces(8)
            elif width >= 1290:
                self.find_face_carousel.set_visible_faces(7)
            elif width >= 1220:
                self.find_face_carousel.set_visible_faces(6)
            elif width >= 1150:
                self.find_face_carousel.set_visible_faces(5)
            elif width >= 1070:
                self.find_face_carousel.set_visible_faces(4)
            elif width >= 990:
                self.find_face_carousel.set_visible_faces(3)
            elif width >= 900:
                self.find_face_carousel.set_visible_faces(2)
            else:
                self.find_face_carousel.set_visible_faces(1)

        for btn in getattr(self, "find_file_edited_media", []):
            self._set_button_compact(btn, width < 1050)

        # --- 5. View Tab Priority Collapse ---
        for btn in getattr(self, "view_details_checkboxes", []):
            self._set_button_compact(btn, width < 1280)

        for btn in getattr(self, "view_arrange_buttons", []):
            self._set_button_compact(btn, width < 1180)

        for btn in (
            getattr(self, "btn_reverse_sort", None),
            getattr(self, "btn_detail_rating", None),
            getattr(self, "btn_detail_caption", None),
            getattr(self, "btn_detail_flags", None),
        ):
            if btn is not None:
                self._set_button_compact(btn, width < 1050)

        for btn in (
            getattr(self, "btn_zoom_in", None),
            getattr(self, "btn_zoom_out", None),
            getattr(self, "btn_view_thumbnails", None),
            getattr(self, "btn_view_map", None),
            getattr(self, "btn_view_slideshow", None),
            getattr(self, "btn_toggle_nav_pane", None),
            getattr(self, "btn_toggle_tag_pane", None),
            getattr(self, "btn_layout_grid", None),
            getattr(self, "btn_layout_mosaic", None),
        ):
            if btn is not None:
                self._set_button_compact(btn, width < 920)

        # --- 6. Group Box Minimum Width Synchronization ---
        for page in (
            getattr(self, "tab_home", None),
            getattr(self, "tab_edit", None),
            getattr(self, "tab_find", None),
            getattr(self, "tab_create", None),
            getattr(self, "tab_view", None),
        ):
            if page and page.layout():
                for i in range(page.layout().count()):
                    w = page.layout().itemAt(i).widget()
                    if w and hasattr(w, "content_layout") and hasattr(w, "content_widget"):
                        # Clear old min/max widths so group box can shrink when children are hidden
                        w.content_widget.setMinimumWidth(0)
                        w.content_widget.setMaximumWidth(16777215)
                        w.setMinimumWidth(0)
                        w.setMaximumWidth(16777215)
                        # Invalidate sub-layouts to ensure compacted button sizes are reflected
                        for j in range(w.content_layout.count()):
                            sub = w.content_layout.itemAt(j)
                            if sub and sub.layout():
                                sub.layout().invalidate()
                        w.content_layout.invalidate()
                        cw_min = w.content_layout.sizeHint().width()
                        w.content_widget.setMinimumWidth(cw_min)
                        w.setMinimumWidth(cw_min + 4)

    # -------------------------------------------------------------------------
    # Tab Builders & Minimization
    # -------------------------------------------------------------------------
    def _on_tab_bar_double_clicked(self, index: int) -> None:
        """Double clicking any tab header minimizes or restores the ribbon."""
        import time
        self._last_double_click_time = time.monotonic()
        self._last_double_click_index = index
        self._toggle_ribbon_collapse()

    def _on_tab_bar_clicked(self, index: int) -> None:
        """When ribbon is collapsed, clicking another tab header restores it."""
        import time
        if (
            index == getattr(self, "_last_double_click_index", -1)
            and time.monotonic() - getattr(self, "_last_double_click_time", 0.0) < 0.35
        ):
            return
        if self.tab_widget.height() <= 36:
            self.tab_widget.setFixedHeight(120)
            self.btn_collapse_ribbon.setText("▲")
            if hasattr(self, "act_qat_minimize"):
                self.act_qat_minimize.setChecked(False)

    @property
    def _ribbon_collapsed(self) -> bool:
        return self.tab_widget.height() <= 36

    @property
    def collapse_btn(self) -> QPushButton:
        return self.btn_collapse_ribbon

    def _toggle_ribbon_collapse(self) -> None:
        if self.tab_widget.height() > 36:
            self.tab_widget.setFixedHeight(28)
            self.btn_collapse_ribbon.setText("▼")
            if hasattr(self, "act_qat_minimize"):
                self.act_qat_minimize.setChecked(True)
        else:
            self.tab_widget.setFixedHeight(120)
            self.btn_collapse_ribbon.setText("▲")
            if hasattr(self, "act_qat_minimize"):
                self.act_qat_minimize.setChecked(False)
        self._save_qat_settings()

    def set_ribbon_collapsed(self, collapsed: bool) -> None:
        if collapsed:
            self.tab_widget.setFixedHeight(28)
            self.btn_collapse_ribbon.setText("▼")
            if hasattr(self, "act_qat_minimize"):
                self.act_qat_minimize.setChecked(True)
        else:
            self.tab_widget.setFixedHeight(120)
            self.btn_collapse_ribbon.setText("▲")
            if hasattr(self, "act_qat_minimize"):
                self.act_qat_minimize.setChecked(False)
        self._save_qat_settings()

    def _toggle_qat_position(self) -> None:
        self.set_qat_below_ribbon(not getattr(self, "_qat_below_ribbon", False))

    def set_qat_below_ribbon(self, below: bool) -> None:
        self._qat_below_ribbon = bool(below)
        if self._qat_below_ribbon:
            self.top_hbox.removeWidget(self.qat_container)
            self.qat_container.setParent(self.below_ribbon_bar)
            self.below_ribbon_layout.insertWidget(0, self.qat_container)
            self.qat_container.show()
            if hasattr(self, "header_div_icon"):
                self.header_div_icon.hide()
            if hasattr(self, "header_div"):
                self.header_div.hide()
            self.below_ribbon_bar.show()
            self.act_qat_position.setText("Show above the Ribbon")
        else:
            self.below_ribbon_layout.removeWidget(self.qat_container)
            self.qat_container.setParent(self.top_header)
            # Insert right after header_div_icon (index 2)
            self.top_hbox.insertWidget(2, self.qat_container)
            self.qat_container.show()
            if hasattr(self, "header_div_icon"):
                self.header_div_icon.show()
            if hasattr(self, "header_div"):
                self.header_div.show()
            self.below_ribbon_bar.hide()
            self.act_qat_position.setText("Show below the Ribbon")
        self._save_qat_settings()

    def _on_win_min_clicked(self) -> None:
        win = self.window()
        if win:
            win.showMinimized()

    def _on_win_max_clicked(self) -> None:
        win = self.window()
        if win:
            if win.isMaximized():
                win.showNormal()
            else:
                win.showMaximized()
            self.update_window_controls()

    def _on_win_close_clicked(self) -> None:
        win = self.window()
        if win:
            win.close()

    def update_window_controls(self) -> None:
        win = self.window()
        is_max = win.isMaximized() if win else False
        if hasattr(self, "btn_win_max"):
            if isinstance(self.btn_win_max, _WindowControlButton):
                self.btn_win_max.set_kind("restore" if is_max else "max")
            else:
                self.btn_win_max.setText("❐" if is_max else "□")
            self.btn_win_max.setToolTip("Restore Down" if is_max else "Maximize")

    def _save_qat_settings(self) -> None:
        try:
            settings = QSettings("PhotoGallery64", "PhotoGallery64")
            visible: list[str] = []
            action_map = {
                "print": getattr(self, "act_qat_print", None),
                "back": getattr(self, "act_qat_back", None),
                "forward": getattr(self, "act_qat_forward", None),
                "undo": getattr(self, "act_qat_undo", None),
                "redo": getattr(self, "act_qat_redo", None),
                "thumbnails": getattr(self, "act_qat_thumbnails", None),
                "map": getattr(self, "act_qat_map", None),
                "people": getattr(self, "act_qat_people", None),
                "rotate_left": getattr(self, "act_qat_rot_left", None),
                "rotate_right": getattr(self, "act_qat_rot_right", None),
                "delete": getattr(self, "act_qat_delete", None),
                "slideshow": getattr(self, "act_qat_slideshow", None),
            }
            for key, act in action_map.items():
                if act and act.isChecked():
                    visible.append(key)
            settings.setValue("qat_visible_actions", ",".join(visible))
            settings.setValue("qat_below_ribbon", bool(getattr(self, "_qat_below_ribbon", False)))
            settings.setValue("ribbon_minimized", bool(self.tab_widget.height() <= 36))
        except Exception:
            pass

    def reset_qat_to_defaults(self) -> None:
        """Resets Quick Access Toolbar to the authentic default actions: Back, Forward, Undo, Redo."""
        action_map = {
            "print": getattr(self, "act_qat_print", None),
            "back": getattr(self, "act_qat_back", None),
            "forward": getattr(self, "act_qat_forward", None),
            "undo": getattr(self, "act_qat_undo", None),
            "redo": getattr(self, "act_qat_redo", None),
            "thumbnails": getattr(self, "act_qat_thumbnails", None),
            "map": getattr(self, "act_qat_map", None),
            "people": getattr(self, "act_qat_people", None),
            "rotate_left": getattr(self, "act_qat_rot_left", None),
            "rotate_right": getattr(self, "act_qat_rot_right", None),
            "delete": getattr(self, "act_qat_delete", None),
            "slideshow": getattr(self, "act_qat_slideshow", None),
        }
        for key, act in action_map.items():
            if act:
                act.setChecked(key in ("back", "forward", "undo", "redo"))
        self._save_qat_settings()

    def _load_qat_settings(self) -> None:
        try:
            settings = QSettings("PhotoGallery64", "PhotoGallery64")
            # Migration check: v2 defaults ensure clean back, forward, undo, redo defaults
            v2_flag = settings.value("qat_defaults_v2", None)
            if v2_flag is None:
                settings.setValue("qat_defaults_v2", 1)
                settings.setValue("qat_visible_actions", "back,forward,undo,redo")

            raw_visible = settings.value("qat_visible_actions", None)
            action_map = {
                "print": getattr(self, "act_qat_print", None),
                "back": getattr(self, "act_qat_back", None),
                "forward": getattr(self, "act_qat_forward", None),
                "undo": getattr(self, "act_qat_undo", None),
                "redo": getattr(self, "act_qat_redo", None),
                "thumbnails": getattr(self, "act_qat_thumbnails", None),
                "map": getattr(self, "act_qat_map", None),
                "people": getattr(self, "act_qat_people", None),
                "rotate_left": getattr(self, "act_qat_rot_left", None),
                "rotate_right": getattr(self, "act_qat_rot_right", None),
                "delete": getattr(self, "act_qat_delete", None),
                "slideshow": getattr(self, "act_qat_slideshow", None),
            }
            if raw_visible is not None:
                if isinstance(raw_visible, str):
                    visible_set = set(raw_visible.split(",")) if raw_visible else set()
                elif isinstance(raw_visible, (list, tuple)):
                    visible_set = set(raw_visible)
                else:
                    visible_set = {"back", "forward", "undo", "redo"}
                for key, act in action_map.items():
                    if act:
                        act.setChecked(key in visible_set)

            below = settings.value("qat_below_ribbon", None)
            if below is not None:
                below_bool = str(below).lower() in ("true", "1")
                self.set_qat_below_ribbon(below_bool)

            minimized = settings.value("ribbon_minimized", None)
            if minimized is not None:
                min_bool = str(minimized).lower() in ("true", "1")
                self.set_ribbon_collapsed(min_bool)
        except Exception:
            pass

    def _get_qat_key_for_widget(self, w: QWidget | None) -> str | None:
        target = w
        while target and target != self:
            if target in (getattr(self, "btn_rot_ccw", None), getattr(self, "btn_edit_rot_l", None), getattr(self, "btn_qat_rot_left", None)):
                return "rotate_left"
            if target in (getattr(self, "btn_rot_cw", None), getattr(self, "btn_edit_rot_r", None), getattr(self, "btn_qat_rot_right", None)):
                return "rotate_right"
            if target in (getattr(self, "btn_delete", None), getattr(self, "btn_edit_del", None), getattr(self, "btn_qat_delete", None)):
                return "delete"
            if target in (getattr(self, "btn_slide", None), getattr(self, "btn_qat_slideshow", None)):
                return "slideshow"
            if target in (getattr(self, "btn_print", None),):
                return "print"
            if target in (getattr(self, "btn_back", None),):
                return "back"
            if target in (getattr(self, "btn_forward", None),):
                return "forward"
            if target in (getattr(self, "btn_undo", None),):
                return "undo"
            if target in (getattr(self, "btn_redo", None),):
                return "redo"
            if target in (getattr(self, "btn_qat_thumbnails", None),):
                return "thumbnails"
            if target in (getattr(self, "btn_qat_map", None), getattr(self, "btn_map", None)):
                return "map"
            if target in (getattr(self, "btn_qat_people", None), getattr(self, "btn_people_tag", None)):
                return "people"
            target = target.parentWidget()
        return None

    def _show_ribbon_context_menu(self, global_pos: QPoint, clicked_widget: QWidget | None = None) -> None:
        menu = QMenu(self)
        menu.setFont(QFont("Segoe UI", 9))

        target = clicked_widget or QApplication.widgetAt(global_pos)
        qat_key = self._get_qat_key_for_widget(target)

        action_map = {
            "print": getattr(self, "act_qat_print", None),
            "back": getattr(self, "act_qat_back", None),
            "forward": getattr(self, "act_qat_forward", None),
            "undo": getattr(self, "act_qat_undo", None),
            "redo": getattr(self, "act_qat_redo", None),
            "thumbnails": getattr(self, "act_qat_thumbnails", None),
            "map": getattr(self, "act_qat_map", None),
            "people": getattr(self, "act_qat_people", None),
            "rotate_left": getattr(self, "act_qat_rot_left", None),
            "rotate_right": getattr(self, "act_qat_rot_right", None),
            "delete": getattr(self, "act_qat_delete", None),
            "slideshow": getattr(self, "act_qat_slideshow", None),
        }

        if qat_key and action_map.get(qat_key):
            act = action_map[qat_key]
            if act.isChecked():
                a_toggle = menu.addAction("Remove from Quick Access Toolbar")
                a_toggle.triggered.connect(lambda: act.setChecked(False))
            else:
                a_toggle = menu.addAction("Add to Quick Access Toolbar")
                a_toggle.triggered.connect(lambda: act.setChecked(True))
        else:
            a_dummy = menu.addAction("Add to Quick Access Toolbar")
            a_dummy.setEnabled(False)

        menu.addSeparator()

        if getattr(self, "_qat_below_ribbon", False):
            act_pos = menu.addAction("Show Quick Access Toolbar above the Ribbon")
        else:
            act_pos = menu.addAction("Show Quick Access Toolbar below the Ribbon")
        act_pos.triggered.connect(self._toggle_qat_position)

        menu.addSeparator()

        is_min = (self.tab_widget.height() <= 36)
        act_min = menu.addAction("Minimize the Ribbon")
        act_min.setCheckable(True)
        act_min.setChecked(is_min)
        act_min.triggered.connect(self._toggle_ribbon_collapse)

        menu.exec(global_pos)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        self._show_ribbon_context_menu(event.globalPos(), None)
        event.accept()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.ContextMenu:
            if not isinstance(obj, (QLineEdit, QMenu)):
                if obj != getattr(self, "btn_file", None):
                    self._show_ribbon_context_menu(event.globalPos(), obj if isinstance(obj, QWidget) else None)
                    return True
        return super().eventFilter(obj, event)

    def set_back_enabled(self, enabled: bool) -> None:
        self.btn_back.setEnabled(enabled)

    def set_forward_enabled(self, enabled: bool) -> None:
        self.btn_forward.setEnabled(enabled)

    def set_undo_enabled(self, enabled: bool) -> None:
        self.btn_undo.setEnabled(enabled)

    def set_redo_enabled(self, enabled: bool) -> None:
        self.btn_redo.setEnabled(enabled)

    def set_revert_enabled(self, enabled: bool) -> None:
        """Enables or greys out the 'Revert to original' ribbon button."""
        if hasattr(self, "btn_edit_revert"):
            self.btn_edit_revert.setEnabled(enabled)

    def set_save_enabled(self, enabled: bool) -> None:
        """Enables or greys out the 'Save' ribbon button."""
        if hasattr(self, "btn_edit_save"):
            self.btn_edit_save.setEnabled(enabled)

    def set_save_copy_enabled(self, enabled: bool) -> None:
        """Enables or greys out the 'Save as copy' split ribbon button."""
        if hasattr(self, "btn_edit_save_copy"):
            self.btn_edit_save_copy.setEnabled(enabled)
            if hasattr(self.btn_edit_save_copy, "btn_bottom"):
                self.btn_edit_save_copy.btn_bottom.setEnabled(enabled)

    def set_flag_state(self, is_flagged: bool) -> None:
        """Updates the highlighted check state of the Flag ribbon buttons in Home & Edit ribbons."""
        if hasattr(self, "btn_home_flag") and self.btn_home_flag is not None:
            self.btn_home_flag.setChecked(bool(is_flagged))
        if hasattr(self, "btn_edit_flag") and self.btn_edit_flag is not None:
            self.btn_edit_flag.setChecked(bool(is_flagged))

    def set_write_back_paused(self, paused: bool) -> None:
        if hasattr(self, "act_pause_write_back"):
            self.act_pause_write_back.setChecked(bool(paused))

    def set_slideshow_theme(self, theme_id: str) -> None:
        """Sets the active slideshow theme and syncs action checkmarks."""
        self.current_slideshow_theme = str(theme_id)
        if hasattr(self, "_slideshow_theme_actions"):
            for act in self._slideshow_theme_actions:
                if act.data() == self.current_slideshow_theme:
                    act.setChecked(True)

    def _build_slideshow_theme_menu(self) -> QMenu:
        """Builds the authentic WLPG 'Slide show themes' menu with 48x36 preview icons."""
        menu = QMenu(self)
        menu.setObjectName("SlideshowThemesMenu")
        menu.setStyleSheet(
            """
            QMenu#SlideshowThemesMenu {
                background-color: #FFFFFF;
                border: 1px solid #70A6E8;
                padding: 4px;
                icon-size: 48px 36px;
            }
            QMenu#SlideshowThemesMenu::item {
                padding: 5px 24px 5px 8px;
                font-family: 'Segoe UI';
                font-size: 9pt;
                color: #1E395B;
            }
            QMenu#SlideshowThemesMenu::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FDFEFE, stop:1 #E4EFFB);
                border: 1px solid #7DA2CE;
                border-radius: 2px;
            }
            """
        )

        act_play = menu.addAction(make_wlpg_icon("play", 16), "Play slide show (F12)")
        act_play.setFont(QFont("Segoe UI", 9, QFont.Bold))
        act_play.triggered.connect(lambda: self.action_triggered.emit("slide_show"))
        menu.addSeparator()

        header_act = QWidgetAction(menu)
        header_lbl = QLabel("Slide show themes")
        header_lbl.setStyleSheet(
            """
            QLabel {
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                font-weight: 600;
                color: #55687C;
                padding: 6px 16px 3px 10px;
                background: transparent;
            }
            """
        )
        header_act.setDefaultWidget(header_lbl)
        menu.addAction(header_act)

        themes = [
            ("static", "Static"),
            ("pan_and_zoom", "Pan and Zoom"),
            ("polaroid", "Polaroid"),
            ("cinematic", "Cinematic"),
            ("fade", "Fade"),
            ("black_and_white", "Black and White"),
            ("sepia", "Sepia"),
        ]

        grp = QActionGroup(menu)
        grp.setExclusive(True)

        from src.ui.slideshow_view import make_slideshow_theme_icon

        for tid, label in themes:
            icon = make_slideshow_theme_icon(tid)
            act = menu.addAction(icon, label)
            act.setCheckable(True)
            act.setData(tid)
            grp.addAction(act)
            if hasattr(self, "_slideshow_theme_actions"):
                self._slideshow_theme_actions.append(act)
            if tid == self.current_slideshow_theme:
                act.setChecked(True)
            act.triggered.connect(lambda checked, t=tid: self._on_slideshow_menu_theme_selected(t))

        return menu

    def _on_slideshow_menu_theme_selected(self, theme_id: str) -> None:
        self.set_slideshow_theme(theme_id)
        self.action_triggered.emit("slide_show")

    def _build_flagged_menu(self, btn: QToolButton) -> None:
        menu = QMenu(btn)
        menu.setFont(QFont("Segoe UI", 9))
        act_all = menu.addAction("All items")
        act_flagged = menu.addAction("Flagged")
        act_unflagged = menu.addAction("Not flagged")

        def _on_all():
            self.flag_filter_selected.emit(None)
            self.flagged_filter_selected.emit(False)

        def _on_flagged():
            self.flag_filter_selected.emit(1)
            self.flagged_filter_selected.emit(True)

        def _on_unflagged():
            self.flag_filter_selected.emit(0)
            self.flagged_filter_selected.emit(False)

        act_all.triggered.connect(_on_all)
        act_flagged.triggered.connect(_on_flagged)
        act_unflagged.triggered.connect(_on_unflagged)
        btn.setMenu(menu)

    def _build_edited_menu(self, btn: QToolButton) -> None:
        menu = QMenu(btn)
        menu.setFont(QFont("Segoe UI", 9))
        act_edited = menu.addAction(make_wlpg_icon("edited", 16), "&Edited")
        act_not_edited = menu.addAction(make_wlpg_icon("not_edited", 16), "&Not edited")

        def _on_edited():
            btn.setChecked(True)
            self._current_edited_filter = 1
            btn.setToolTip("Filtering: Edited items (click to clear)")
            self.edited_filter_mode_selected.emit(1)
            self.edited_filter_selected.emit(True)

        def _on_not_edited():
            btn.setChecked(True)
            self._current_edited_filter = 0
            btn.setToolTip("Filtering: Not edited items (click to clear)")
            self.edited_filter_mode_selected.emit(0)
            self.edited_filter_selected.emit(False)

        act_edited.triggered.connect(_on_edited)
        act_not_edited.triggered.connect(_on_not_edited)
        btn.setMenu(menu)

    def _on_find_edited_clicked(self, checked: bool) -> None:
        if checked:
            self._current_edited_filter = 1
            self.btn_find_edited.setToolTip("Filtering: Edited items (click to clear)")
            self.edited_filter_mode_selected.emit(1)
            self.edited_filter_selected.emit(True)
        else:
            self._current_edited_filter = None
            self.btn_find_edited.setToolTip("Filter by edited items")
            self.edited_filter_mode_selected.emit(None)
            self.edited_filter_selected.emit(False)

    def minimumSizeHint(self) -> QSize:
        return QSize(350, 88)

    @staticmethod
    def _wrap_tab_page(page: QWidget) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll.setWidget(page)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        return scroll

    def _build_home_tab(self) -> QWidget:
        page = _RibbonPage()
        hbox = QHBoxLayout(page)
        hbox.setContentsMargins(4, 2, 4, 2)
        hbox.setSpacing(4)

        # ---------------------------------------------------------------------
        # Group 1: New (Import, Add folder)
        # ---------------------------------------------------------------------
        grp_new = _RibbonGroupBox("New", page)
        self.btn_import = self._make_btn("Import", "import", grp_new, large=True)
        self.btn_import.clicked.connect(lambda: self.action_triggered.emit("import_media"))
        self.btn_add_folder = self._make_btn("Add\nfolder", "folder_add", grp_new, large=True)
        self.btn_add_folder.setToolTip("Add a folder to the gallery")
        self.btn_add_folder.clicked.connect(lambda: self.action_triggered.emit("add_folder"))
        self.btn_new_folder = self.btn_add_folder
        grp_new.content_layout.addWidget(self.btn_import)
        grp_new.content_layout.addWidget(self.btn_add_folder)
        hbox.addWidget(grp_new)

        # ---------------------------------------------------------------------
        # Group 2: Manage (2 columns of 3 rows: Select all/Copy/Paste & Rotate left/right/Delete)
        # ---------------------------------------------------------------------
        grp_manage = _RibbonGroupBox("Manage", page)

        m_grid = QGridLayout()
        m_grid.setContentsMargins(0, 0, 0, 0)
        m_grid.setHorizontalSpacing(2)
        m_grid.setVerticalSpacing(1)

        self.btn_select_all = self._make_btn("Select all", "select_all", grp_manage)
        self.btn_select_all.setToolTip("Select all (Ctrl+A)")
        self.btn_select_all._tooltip_override = "Select all (Ctrl+A)"
        self.btn_select_all.clicked.connect(lambda: self.action_triggered.emit("select_all"))
        self.btn_copy = self._make_btn("Copy", "copy", grp_manage)
        self.btn_copy.setToolTip("Copy (Ctrl+C)")
        self.btn_copy._tooltip_override = "Copy (Ctrl+C)"
        self.btn_copy.clicked.connect(lambda: self.action_triggered.emit("copy"))
        self.btn_paste = self._make_btn("Paste", "paste", grp_manage)
        self.btn_paste.setToolTip("Paste (Ctrl+V)")
        self.btn_paste._tooltip_override = "Paste (Ctrl+V)"
        self.btn_paste.clicked.connect(lambda: self.action_triggered.emit("paste"))

        self.btn_rot_ccw = self._make_btn("Rotate left", "rotate_ccw", grp_manage)
        self.btn_rot_ccw.setToolTip("Rotate left (Ctrl+,)")
        self.btn_rot_ccw._tooltip_override = "Rotate left (Ctrl+,)"
        self.btn_rot_cw = self._make_btn("Rotate right", "rotate_cw", grp_manage)
        self.btn_rot_cw.setToolTip("Rotate right (Ctrl+.)")
        self.btn_rot_cw._tooltip_override = "Rotate right (Ctrl+.)"
        self.btn_delete = self._make_btn("Delete", "delete", grp_manage)
        self.btn_delete.setToolTip("Delete (Del)")
        self.btn_delete._tooltip_override = "Delete (Del)"
        self.btn_rot_ccw.clicked.connect(lambda: self.action_triggered.emit("rotate_left"))
        self.btn_rot_cw.clicked.connect(lambda: self.action_triggered.emit("rotate_right"))
        self.btn_delete.clicked.connect(lambda: self.action_triggered.emit("delete"))

        # Col 0 (left): Select all, Copy, Paste
        m_grid.addWidget(self.btn_select_all, 0, 0)
        m_grid.addWidget(self.btn_copy, 1, 0)
        m_grid.addWidget(self.btn_paste, 2, 0)

        # Col 1 (right): Rotate left, Rotate right, Delete
        m_grid.addWidget(self.btn_rot_ccw, 0, 1)
        m_grid.addWidget(self.btn_rot_cw, 1, 1)
        m_grid.addWidget(self.btn_delete, 2, 1)

        grp_manage.content_layout.addLayout(m_grid)
        hbox.addWidget(grp_manage)

        # ---------------------------------------------------------------------
        # Group 3: Organize (People tag, Batch people tag, Descriptive tag, Caption | Geotag, Rate, Flag)
        # ---------------------------------------------------------------------
        grp_org = _RibbonGroupBox("Organize", page)

        self.btn_people_tag = self._make_btn("People\ntag", "people_tag", grp_org, large=True)
        self.btn_people_tag.setToolTip("Identify a person in a photo")
        self.btn_people_tag.clicked.connect(lambda: self.action_triggered.emit("people_tag"))
        grp_org.content_layout.addWidget(self.btn_people_tag)

        self.btn_batch_people = self._make_btn("Batch\npeople tag", "batch_people", grp_org, large=True)
        self.btn_batch_people.setToolTip("Batch people tag")
        self.btn_batch_people.clicked.connect(lambda: self.action_triggered.emit("batch_people_tag"))
        grp_org.content_layout.addWidget(self.btn_batch_people)

        split_tag_desc, self.btn_tag_desc = self._make_split_tag_desc_button(grp_org)
        grp_org.content_layout.addWidget(split_tag_desc)

        self.btn_caption = self._make_btn("Caption", "caption", grp_org, large=True)
        self.btn_caption.setToolTip("Caption")
        self.btn_caption.clicked.connect(lambda: self.action_triggered.emit("edit_caption"))
        grp_org.content_layout.addWidget(self.btn_caption)

        # Right sub-column: Geotag, Rate, Flag (3 stacked rows)
        o_right_col = QVBoxLayout()
        o_right_col.setContentsMargins(0, 0, 0, 0)
        o_right_col.setSpacing(1)

        self.btn_geotag = self._make_btn("Geotag", "geotag", grp_org)
        self.btn_geotag.clicked.connect(lambda: self.action_triggered.emit("geotag"))

        self.btn_home_rate = self._make_btn("Rate", "rate", grp_org)
        self.btn_home_rate.setPopupMode(QToolButton.InstantPopup)
        self.btn_home_rate.setToolTip("Rate photo 0-5 stars (0-5)")
        self.btn_home_rate._tooltip_override = "Rate photo 0-5 stars (0-5)"
        rate_menu = create_wlpg_rating_menu(
            self.btn_home_rate,
            on_selected=lambda r: self.action_triggered.emit(f"rate_{r}"),
            show_all_ratings=False,
            is_dark=getattr(self, "_dark_mode", False),
        )
        self.btn_home_rate.setMenu(rate_menu)

        self.btn_home_flag = self._make_btn("Flag", "flag", grp_org, checkable=True)
        self.btn_home_flag.setToolTip("Toggle flag (Insert / \\)")
        self.btn_home_flag._tooltip_override = "Toggle flag (Insert / \\)"
        self.btn_home_flag.clicked.connect(lambda: self.action_triggered.emit("toggle_flag"))

        o_right_col.addWidget(self.btn_geotag)
        o_right_col.addWidget(self.btn_home_rate)
        o_right_col.addWidget(self.btn_home_flag)
        grp_org.content_layout.addLayout(o_right_col)

        hbox.addWidget(grp_org)

        # Home group button collections for priority collapse
        self.home_manage_rot_delete = [self.btn_rot_ccw, self.btn_rot_cw, self.btn_delete]
        self.home_manage_select_copy_paste = [self.btn_select_all, self.btn_copy, self.btn_paste]
        self.home_org_geotag_rate_flag = [self.btn_geotag, self.btn_home_rate, self.btn_home_flag]
        self.home_org_batch_desc_caption = [self.btn_batch_people, self.btn_tag_desc, self.btn_caption]

        # ---------------------------------------------------------------------
        # Group 4: Quick find (Face carousel + Months/Rated/Flagged + Text search)
        # ---------------------------------------------------------------------
        grp_qfind = _RibbonGroupBox("Quick find", page)

        # 1. Flexible Face Carousel (showing 1-6 faces)
        self.home_face_carousel = _FaceCarouselWidget(grp_qfind)
        self.home_face_carousel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.home_face_carousel.person_clicked.connect(self.person_quick_find_selected.emit)
        self.home_face_carousel.boolean_mode_selected.connect(self._on_carousel_boolean_mode)
        self.home_face_carousel.special_filter_selected.connect(self.people_special_filter_selected.emit)

        # 2. Dropdown Filter Buttons: Months, Rated, Flagged (3 stacked rows)
        qf_filters_col = QVBoxLayout()
        qf_filters_col.setContentsMargins(0, 0, 0, 0)
        qf_filters_col.setSpacing(1)
        qf_filters_col.setAlignment(Qt.AlignLeft)

        self.btn_qf_months = self._make_btn("Months", "months", grp_qfind)
        self.btn_qf_months.setFixedHeight(19)
        self.btn_qf_months.setPopupMode(QToolButton.InstantPopup)
        self.btn_qf_months.setStyleSheet("font-size: 8pt; padding: 0 10px 0 3px;")
        months_menu = QMenu(self.btn_qf_months)
        for m_num in range(1, 13):
            m_name = calendar.month_name[m_num]
            months_menu.addAction(
                m_name,
                lambda _c=False, mn=m_num: (
                    self.filter_month_selected.emit(mn),
                    self.date_filter_selected.emit(None, mn, None),
                ),
            )
        months_menu.addSeparator()
        months_menu.addAction(
            "All dates",
            lambda: (
                self.filter_month_selected.emit(0),
                self.date_filter_selected.emit(None, None, None),
            ),
        )
        self.btn_qf_months.setMenu(months_menu)

        self.btn_qf_rated = self._make_btn("Rated", "rate", grp_qfind)
        self.btn_qf_rated.setFixedHeight(19)
        self.btn_qf_rated.setPopupMode(QToolButton.InstantPopup)
        def _on_filter_rating_selected(r: int):
            if r == 5:
                self.rating_filter_selected.emit((5,))
            elif r == 4:
                self.rating_filter_selected.emit((4, 5))
            elif r == 3:
                self.rating_filter_selected.emit((3, 4, 5))
            elif r == 2:
                self.rating_filter_selected.emit((2, 3, 4, 5))
            elif r == 1:
                self.rating_filter_selected.emit((1, 2, 3, 4, 5))
            elif r == 0:
                self.rating_filter_selected.emit((0,))
            else:
                self.rating_filter_selected.emit(())

        rated_menu = create_wlpg_rating_menu(
            self.btn_qf_rated,
            on_selected=_on_filter_rating_selected,
            show_all_ratings=True,
            is_dark=getattr(self, "_dark_mode", False),
        )
        self.btn_qf_rated.setMenu(rated_menu)

        # Flagged Dropdown Filter Button
        self.btn_qf_flagged = self._make_btn("Flagged", "flag", grp_qfind)
        self.btn_qf_flagged.setFixedHeight(19)
        self.btn_qf_flagged.setPopupMode(QToolButton.InstantPopup)
        self.btn_qf_flagged.setStyleSheet("font-size: 8pt; padding: 0 10px 0 3px;")
        flagged_menu = QMenu(self.btn_qf_flagged)
        act_f_all = flagged_menu.addAction("All items")
        act_f_flagged = flagged_menu.addAction("Flagged")
        act_f_unflagged = flagged_menu.addAction("Not flagged")

        def _on_menu_all():
            self.flag_filter_selected.emit(None)
            self.flagged_filter_selected.emit(False)

        def _on_menu_flagged():
            self.flag_filter_selected.emit(1)
            self.flagged_filter_selected.emit(True)

        def _on_menu_unflagged():
            self.flag_filter_selected.emit(0)
            self.flagged_filter_selected.emit(False)

        act_f_all.triggered.connect(_on_menu_all)
        act_f_flagged.triggered.connect(_on_menu_flagged)
        act_f_unflagged.triggered.connect(_on_menu_unflagged)
        self.btn_qf_flagged.setMenu(flagged_menu)

        qf_filters_col.addWidget(self.btn_qf_months)
        qf_filters_col.addWidget(self.btn_qf_rated)
        qf_filters_col.addWidget(self.btn_qf_flagged)
        self.home_qf_buttons = [self.btn_qf_months, self.btn_qf_rated, self.btn_qf_flagged]

        # 3. Text search large button (toggles Search Scope Bar)
        btn_text_search = self._make_btn("Text\nsearch", "text_search", grp_qfind, large=True)
        btn_text_search.clicked.connect(lambda: self.action_triggered.emit("toggle_scope_bar"))

        grp_qfind.content_layout.setContentsMargins(0, 0, 0, 0)
        grp_qfind.content_layout.setSpacing(0)
        grp_qfind.content_layout.addWidget(self.home_face_carousel)
        grp_qfind.content_layout.addSpacing(2)
        grp_qfind.content_layout.addLayout(qf_filters_col)
        grp_qfind.content_layout.addSpacing(2)
        grp_qfind.content_layout.addWidget(btn_text_search)

        hbox.addWidget(grp_qfind)

        # ---------------------------------------------------------------------
        # Group 5: Share / Presentation (Slide show 2-button vertical split)
        # ---------------------------------------------------------------------
        grp_share = _RibbonGroupBox("Share", page)
        split_home, self.btn_home_slideshow = self._make_slideshow_split_button(grp_share)
        grp_share.content_layout.addWidget(split_home)
        hbox.addWidget(grp_share)

        hbox.addStretch(1)

        return page

    def _build_edit_tab(self) -> QWidget:
        """
        Contextual 'Edit' tab matching Windows Live Photo Gallery:
        - In Grid View: Manage, Properties, Quick adjustments, Effects, Revert to original, View file.
        - In Single View: Iterate, Manage, Organize, Adjustments, Effects, Revert to original, Close file.
        """
        page = _RibbonPage()
        hbox = QHBoxLayout(page)
        hbox.setContentsMargins(2, 2, 2, 2)
        hbox.setSpacing(2)

        # 1. Iterate Group (Single View only: Previous |<<, Next >>|)
        self.grp_edit_iterate = _RibbonGroupBox("Iterate", page)
        self.btn_edit_prev = self._make_btn("Previous", "prev_media", self.grp_edit_iterate, large=True)
        self.btn_edit_prev.setToolTip("Previous image (Left Arrow)")
        self.btn_edit_prev.clicked.connect(lambda: self.action_triggered.emit("prev_media"))

        self.btn_edit_next = self._make_btn("Next", "next_media", self.grp_edit_iterate, large=True)
        self.btn_edit_next.setToolTip("Next image (Right Arrow)")
        self.btn_edit_next.clicked.connect(lambda: self.action_triggered.emit("next_media"))

        self.grp_edit_iterate.content_layout.addWidget(self.btn_edit_prev)
        self.grp_edit_iterate.content_layout.addWidget(self.btn_edit_next)
        hbox.addWidget(self.grp_edit_iterate)

        # 2. Manage Group
        self.grp_edit_manage = _RibbonGroupBox("Manage", page)

        # Col 1: Grid only (Select all, Copy, Paste)
        self.col_grid_manage = QWidget(self.grp_edit_manage)
        cgm_vbox = QVBoxLayout(self.col_grid_manage)
        cgm_vbox.setContentsMargins(0, 0, 0, 0)
        cgm_vbox.setSpacing(1)

        self.btn_edit_select_all = self._make_btn("Select all", "select_all", self.col_grid_manage)
        self.btn_edit_select_all.setToolTip("Select all (Ctrl+A)")
        self.btn_edit_select_all.clicked.connect(lambda: self.action_triggered.emit("select_all"))

        self.btn_edit_copy = self._make_btn("Copy", "copy", self.col_grid_manage)
        self.btn_edit_copy.setToolTip("Copy (Ctrl+C)")
        self.btn_edit_copy.clicked.connect(lambda: self.action_triggered.emit("copy"))

        self.btn_edit_paste = self._make_btn("Paste", "paste", self.col_grid_manage)
        self.btn_edit_paste.setToolTip("Paste (Ctrl+V)")
        self.btn_edit_paste.clicked.connect(lambda: self.action_triggered.emit("paste"))

        cgm_vbox.addWidget(self.btn_edit_select_all)
        cgm_vbox.addWidget(self.btn_edit_copy)
        cgm_vbox.addWidget(self.btn_edit_paste)
        self.grp_edit_manage.content_layout.addWidget(self.col_grid_manage)

        # Col 2: Common (Rotate left, Rotate right, Delete)
        self.col_common_manage = QWidget(self.grp_edit_manage)
        ccm_vbox = QVBoxLayout(self.col_common_manage)
        ccm_vbox.setContentsMargins(0, 0, 0, 0)
        ccm_vbox.setSpacing(1)

        self.btn_edit_rot_l = self._make_btn("Rotate left", "rotate_ccw", self.col_common_manage)
        self.btn_edit_rot_l.setToolTip("Rotate left (Ctrl+,)")
        self.btn_edit_rot_l.clicked.connect(lambda: self.action_triggered.emit("rotate_left"))

        self.btn_edit_rot_r = self._make_btn("Rotate right", "rotate_cw", self.col_common_manage)
        self.btn_edit_rot_r.setToolTip("Rotate right (Ctrl+.)")
        self.btn_edit_rot_r.clicked.connect(lambda: self.action_triggered.emit("rotate_right"))

        self.btn_edit_del = self._make_btn("Delete", "delete", self.col_common_manage)
        self.btn_edit_del.setToolTip("Delete (Del)")
        self.btn_edit_del.clicked.connect(lambda: self.action_triggered.emit("delete"))

        ccm_vbox.addWidget(self.btn_edit_rot_l)
        ccm_vbox.addWidget(self.btn_edit_rot_r)
        ccm_vbox.addWidget(self.btn_edit_del)
        self.grp_edit_manage.content_layout.addWidget(self.col_common_manage)
        hbox.addWidget(self.grp_edit_manage)

        # 3. Properties Group (Grid view only)
        self.grp_edit_properties = _RibbonGroupBox("Properties", page)
        props_vbox = QVBoxLayout()
        props_vbox.setContentsMargins(0, 0, 0, 0)
        props_vbox.setSpacing(2)

        self.btn_edit_rename = self._make_btn("Rename", "rename", self.grp_edit_properties)
        self.btn_edit_rename.setToolTip("Rename file (F2)")
        self.btn_edit_rename.clicked.connect(lambda: self.action_triggered.emit("rename_file"))

        self.btn_edit_resize = self._make_btn("Resize", "resize", self.grp_edit_properties)
        self.btn_edit_resize.setToolTip("Resize photo...")
        self.btn_edit_resize.clicked.connect(lambda: self.action_triggered.emit("resize_photo"))

        self.btn_edit_time = self._make_btn("Adjust time", "calendar", self.grp_edit_properties)
        self.btn_edit_time.setToolTip("Adjust date and time taken...")
        self.btn_edit_time.clicked.connect(lambda: self.action_triggered.emit("change_time_taken"))

        props_vbox.addWidget(self.btn_edit_rename)
        props_vbox.addWidget(self.btn_edit_resize)
        props_vbox.addWidget(self.btn_edit_time)
        self.grp_edit_properties.content_layout.addLayout(props_vbox)
        hbox.addWidget(self.grp_edit_properties)

        # 4. Organize Group (Single view only: Tag and caption, Rate ▾, Flag, Properties ▾)
        self.grp_edit_organize = _RibbonGroupBox("Organize", page)

        self.btn_edit_tag_caption = self._make_btn("Tag and\ncaption", "tag", self.grp_edit_organize, large=True)
        self.btn_edit_tag_caption.setToolTip("Show information pane (Ctrl+I)")
        self.btn_edit_tag_caption.clicked.connect(lambda: self.action_triggered.emit("toggle_tag_pane"))
        self.grp_edit_organize.content_layout.addWidget(self.btn_edit_tag_caption)

        # Retain btn_edit_people_tag attribute for backwards compatibility
        self.btn_edit_people_tag = self._make_btn("People\ntag", "people_tag", self.grp_edit_organize, large=True)
        self.btn_edit_people_tag.setToolTip("Identify a person in this photo")
        self.btn_edit_people_tag.clicked.connect(lambda: self.action_triggered.emit("people_tag"))
        self.btn_edit_people_tag.setVisible(False)

        org_col = QVBoxLayout()
        org_col.setContentsMargins(0, 0, 0, 0)
        org_col.setSpacing(2)

        self.btn_edit_rate = self._make_btn("Rate", "rate", self.grp_edit_organize)
        self.btn_edit_rate.setPopupMode(QToolButton.InstantPopup)
        rate_menu = create_wlpg_rating_menu(
            self.btn_edit_rate,
            on_selected=lambda r: self.action_triggered.emit(f"rate_{r}"),
            show_all_ratings=False,
            is_dark=getattr(self, "_dark_mode", False),
        )
        self.btn_edit_rate.setMenu(rate_menu)

        self.btn_edit_flag = self._make_btn("Flag", "flag", self.grp_edit_organize, checkable=True)
        self.btn_edit_flag.setToolTip("Toggle flag")
        self.btn_edit_flag.clicked.connect(lambda: self.action_triggered.emit("toggle_flag"))

        self.btn_edit_prop_drop = self._make_btn("Properties", "options", self.grp_edit_organize)
        self.btn_edit_prop_drop.setPopupMode(QToolButton.InstantPopup)
        prop_menu = QMenu(self.btn_edit_prop_drop)
        prop_menu.addAction("Adjust date and time...", lambda: self.action_triggered.emit("change_time_taken"))
        prop_menu.addAction("Resize photo...", lambda: self.action_triggered.emit("resize_photo"))
        prop_menu.addAction("Rename photo...", lambda: self.action_triggered.emit("rename_file"))
        self.btn_edit_prop_drop.setMenu(prop_menu)

        org_col.addWidget(self.btn_edit_rate)
        org_col.addWidget(self.btn_edit_flag)
        org_col.addWidget(self.btn_edit_prop_drop)
        self.grp_edit_organize.content_layout.addLayout(org_col)
        hbox.addWidget(self.grp_edit_organize)

        # 5. Quick Adjustments / Adjustments Group
        self.grp_edit_adjust = _RibbonGroupBox("Quick adjustments", page)

        self.btn_edit_auto = self._make_btn("Auto\nadjust", "auto_adjust", self.grp_edit_adjust, large=True)
        self.btn_edit_auto.setToolTip("Automatically adjust brightness, contrast, and color")
        self.btn_edit_auto.clicked.connect(lambda: self.action_triggered.emit("auto_adjust"))
        self.grp_edit_adjust.content_layout.addWidget(self.btn_edit_auto)

        self.btn_edit_crop = self._make_btn("Crop", "crop", self.grp_edit_adjust, large=True)
        self.btn_edit_crop.setToolTip("Crop photo (draw crop rectangle)")
        self.btn_edit_crop.clicked.connect(lambda: self.action_triggered.emit("crop_photo"))
        self.grp_edit_adjust.content_layout.addWidget(self.btn_edit_crop)

        # Unified Spot Fix button (large icon at full width, direct click to open Spot Fix toolbar)
        self.btn_edit_spot_fix = self._make_btn("Spot\nfix", "spot_fix", self.grp_edit_adjust, large=True)
        self.btn_edit_spot_fix.setToolTip("Repair blemishes and remove red eye")
        self.btn_edit_spot_fix.clicked.connect(lambda: self.action_triggered.emit("retouch"))

        # Keep legacy aliases for backwards compatibility with any existing tests/references
        self.btn_edit_retouch = self.btn_edit_spot_fix
        self.btn_edit_red_eye = self.btn_edit_spot_fix
        self.grp_edit_adjust.content_layout.addWidget(self.btn_edit_spot_fix)

        # Markup tool (large icon at full width)
        self.btn_edit_markup = self._make_btn("Markup", "markup", self.grp_edit_adjust, large=True)
        self.btn_edit_markup.setToolTip("Quick markup, callouts & privacy redaction")
        self.btn_edit_markup.clicked.connect(lambda: self.action_triggered.emit("markup_redact"))
        self.grp_edit_adjust.content_layout.addWidget(self.btn_edit_markup)

        # Stack of adjustments: Straighten, Colour, and Exposure
        stack_adj = QVBoxLayout()
        stack_adj.setContentsMargins(0, 0, 0, 0)
        stack_adj.setSpacing(1)

        self.btn_edit_straighten = self._make_btn("Straighten", "straighten", self.grp_edit_adjust, large=False)
        self.btn_edit_straighten.setToolTip("Straighten photo horizon / vertical alignment")
        self.btn_edit_straighten.clicked.connect(lambda: self.action_triggered.emit("straighten"))
        stack_adj.addWidget(self.btn_edit_straighten)

        self.btn_edit_color = self._make_btn("Colour", "adjust_color", self.grp_edit_adjust, large=False)
        self.btn_edit_color.setToolTip("Adjust photo colour and white balance")
        self.btn_edit_color.setPopupMode(QToolButton.InstantPopup)
        self.menu_color = QMenu(self.btn_edit_color)
        self.menu_color.setStyleSheet("QMenu { background-color: #FFFFFF; border: 1px solid #70A6E8; }")
        self.color_menu_widget = _ColorMenuWidget(self.menu_color)
        self.color_menu_widget.preview_requested.connect(self.color_preview_requested.emit)
        self.color_menu_widget.auto_preview_requested.connect(self.auto_color_preview_requested.emit)
        self.color_menu_widget.preview_cancelled.connect(self.color_preview_cancelled.emit)
        self.color_menu_widget.adjustment_selected.connect(self.color_adjustment_selected.emit)
        self.color_menu_widget.auto_selected.connect(self.auto_color_selected.emit)
        act_col = QWidgetAction(self.menu_color)
        act_col.setDefaultWidget(self.color_menu_widget)
        self.menu_color.addAction(act_col)
        self.btn_edit_color.setMenu(self.menu_color)
        self.menu_color.aboutToShow.connect(self.color_menu_about_to_show.emit)
        self.menu_color.aboutToHide.connect(self.color_preview_cancelled.emit)
        stack_adj.addWidget(self.btn_edit_color)

        self.btn_edit_exposure = self._make_btn("Exposure", "adjust_exposure", self.grp_edit_adjust, large=False)
        self.btn_edit_exposure.setPopupMode(QToolButton.InstantPopup)
        self.menu_exposure = QMenu(self.btn_edit_exposure)
        self.menu_exposure.setStyleSheet("QMenu { background-color: #FFFFFF; border: 1px solid #70A6E8; }")
        self.exposure_menu_widget = _ExposureMenuWidget(self.menu_exposure)
        self.exposure_menu_widget.preview_requested.connect(self.exposure_preview_requested.emit)
        self.exposure_menu_widget.auto_preview_requested.connect(self.auto_exposure_preview_requested.emit)
        self.exposure_menu_widget.preview_cancelled.connect(self.exposure_preview_cancelled.emit)
        self.exposure_menu_widget.adjustment_selected.connect(self.exposure_adjustment_selected.emit)
        self.exposure_menu_widget.auto_selected.connect(self.auto_exposure_selected.emit)
        act_exp = QWidgetAction(self.menu_exposure)
        act_exp.setDefaultWidget(self.exposure_menu_widget)
        self.menu_exposure.addAction(act_exp)
        self.btn_edit_exposure.setMenu(self.menu_exposure)
        self.menu_exposure.aboutToShow.connect(self.exposure_menu_about_to_show.emit)
        self.menu_exposure.aboutToHide.connect(self.exposure_preview_cancelled.emit)
        stack_adj.addWidget(self.btn_edit_exposure)

        self.stack_adj = stack_adj
        self.grp_edit_adjust.content_layout.addLayout(self.stack_adj)

        self.btn_edit_fine_tune = self._make_btn("Fine\ntune", "fine_tune", self.grp_edit_adjust, large=True)
        self.btn_edit_fine_tune.setToolTip("Fine-tune exposure, color, straighten, and detail (Ctrl+F)")
        self.btn_edit_fine_tune.clicked.connect(lambda: self.action_triggered.emit("fine_tune"))
        self.grp_edit_adjust.content_layout.addWidget(self.btn_edit_fine_tune)

        self.btn_edit_depth_light = self._make_btn("Depth &\nlight", "depth_light", self.grp_edit_adjust, large=True)
        self.btn_edit_depth_light.setToolTip("Portrait bokeh, 3D depth relighting, and atmospheric haze")
        self.btn_edit_depth_light.clicked.connect(lambda: self.action_triggered.emit("depth_light_pane"))
        self.grp_edit_adjust.content_layout.addWidget(self.btn_edit_depth_light)
        hbox.addWidget(self.grp_edit_adjust)

        # 6. Effects Group (Thumbnail preview gallery matching WLPG)
        self.grp_edit_effects = _RibbonGroupBox("Effects", page)
        self.effects_gallery = _RibbonEffectsGallery(self.grp_edit_effects)
        self.effects_gallery.effect_selected.connect(
            lambda eff: self.action_triggered.emit(f"effect_{eff}")
        )
        self.grp_edit_effects.content_layout.addWidget(self.effects_gallery)
        hbox.addWidget(self.grp_edit_effects)

        # 7. Save Group
        self.grp_edit_save = _RibbonGroupBox("Save", page)
        self.btn_edit_save = self._make_btn("Save", "save", self.grp_edit_save, large=True)
        self.btn_edit_save.setToolTip("Save changes to original photo (Ctrl+S) — pristine original backed up to .OriginalFiles")
        self.btn_edit_save.clicked.connect(lambda: self.action_triggered.emit("save_photo"))
        self.btn_edit_save.setEnabled(False)
        self.grp_edit_save.content_layout.addWidget(self.btn_edit_save)

        split_copy, self.btn_edit_save_copy = self._make_split_save_copy_button(self.grp_edit_save)
        self.btn_edit_save_copy.setEnabled(False)
        if hasattr(self.btn_edit_save_copy, "btn_bottom"):
            self.btn_edit_save_copy.btn_bottom.setEnabled(False)
        self.grp_edit_save.content_layout.addWidget(split_copy)
        hbox.addWidget(self.grp_edit_save)

        # 8. Revert to original
        self.grp_edit_revert = _RibbonGroupBox("Revert", page)
        self.btn_edit_revert = self._make_btn(
            "Revert to\noriginal", "revert_original", self.grp_edit_revert, large=True
        )
        self.btn_edit_revert.setToolTip("Revert all edits back to the pristine original photo")
        self.btn_edit_revert.setEnabled(False)
        self.btn_edit_revert.clicked.connect(lambda: self.action_triggered.emit("revert_original"))
        self.grp_edit_revert.content_layout.addWidget(self.btn_edit_revert)
        hbox.addWidget(self.grp_edit_revert)

        # 8. View file / Close file Group
        self.grp_edit_close = _RibbonGroupBox("View", page)

        self.btn_edit_view_file = self._make_btn("View\nfile", "view_file", self.grp_edit_close, large=True)
        self.btn_edit_view_file.setToolTip("Open selected photo in viewer (Enter)")
        self.btn_edit_view_file.clicked.connect(lambda: self.action_triggered.emit("view_file"))
        self.grp_edit_close.content_layout.addWidget(self.btn_edit_view_file)

        self.btn_edit_close_file = self._make_btn("Close\nfile", "close_file", self.grp_edit_close, large=True)
        self.btn_edit_close_file.setToolTip("Close viewer and return to grid (Esc)")
        self.btn_edit_close_file.clicked.connect(lambda: self.action_triggered.emit("close_viewer"))
        self.grp_edit_close.content_layout.addWidget(self.btn_edit_close_file)
        hbox.addWidget(self.grp_edit_close)

        # Edit group button collections for priority collapse
        self.edit_redeye_retouch_straighten = [self.btn_edit_straighten]
        self.edit_noise_color_exposure = [self.btn_edit_color, self.btn_edit_exposure]
        self.edit_manage_rot_delete = [self.btn_edit_rot_l, self.btn_edit_rot_r, self.btn_edit_del]
        self.edit_manage_select_copy_paste = [self.btn_edit_select_all, self.btn_edit_copy, self.btn_edit_paste]
        self.edit_org_rate_flag_props = [self.btn_edit_rate, self.btn_edit_flag, self.btn_edit_prop_drop]
        self.edit_grid_properties = [self.btn_edit_rename, self.btn_edit_resize, self.btn_edit_time]

        hbox.addStretch(1)

        self._set_edit_tab_mode("grid")
        return page

    def _set_edit_tab_mode(self, mode: Literal["grid", "viewer"]) -> None:
        is_viewer = (mode == "viewer")
        if hasattr(self, "grp_edit_iterate"):
            self.grp_edit_iterate.setVisible(is_viewer)
        if hasattr(self, "col_grid_manage"):
            self.col_grid_manage.setVisible(not is_viewer)
        if hasattr(self, "grp_edit_properties"):
            self.grp_edit_properties.setVisible(not is_viewer)
        if hasattr(self, "grp_edit_organize"):
            self.grp_edit_organize.setVisible(is_viewer)
        # Canvas tools remain accessible in Grid View, routing to Single Viewer with the tool active
        if hasattr(self, "btn_edit_crop"):
            self.btn_edit_crop.setVisible(True)
        if hasattr(self, "btn_edit_red_eye"):
            self.btn_edit_red_eye.setVisible(True)
        if hasattr(self, "btn_edit_retouch"):
            self.btn_edit_retouch.setVisible(True)
        if hasattr(self, "btn_edit_straighten"):
            self.btn_edit_straighten.setVisible(True)
        if hasattr(self, "btn_edit_fine_tune"):
            self.btn_edit_fine_tune.setVisible(True)
        if hasattr(self, "btn_edit_depth_light"):
            self.btn_edit_depth_light.setVisible(True)
        if hasattr(self, "grp_edit_save"):
            self.grp_edit_save.setVisible(is_viewer)
        if hasattr(self, "btn_edit_view_file"):
            self.btn_edit_view_file.setVisible(not is_viewer)
        if hasattr(self, "btn_edit_close_file"):
            self.btn_edit_close_file.setVisible(is_viewer)
        if hasattr(self, "grp_edit_adjust"):
            self.grp_edit_adjust.setTitle("Adjustments" if is_viewer else "Quick adjustments")
        if hasattr(self, "grp_edit_close"):
            self.grp_edit_close.setTitle("Close" if is_viewer else "View")

        # Re-apply responsive layout to resize group boxes snugly around visible controls
        if hasattr(self, "grp_edit_manage"):
            self.grp_edit_manage.content_widget.setMinimumWidth(0)
            self.grp_edit_manage.setMinimumWidth(0)
        self._apply_responsive_layout(self.width())

    def _build_find_tab(self) -> QWidget:
        """
        'Find' tab matching Windows Live Photo Gallery reference screenshots:
        1. Date taken: Dates, Months, Years dropdown buttons
        2. People: Face carousel + stacked (+ and / / or) toggle buttons
        3. Rated: Rated dropdown button + stacked (↑ and higher / ↓ and lower / = only) mode buttons
        4. Tags: Tags dropdown button + stacked (+ and / / or) toggle buttons
        5. File details: Flagged, Edited, Media type dropdown button
        6. Text search: Large Text search binoculars button (Ctrl+F)
        """
        page = _RibbonPage()
        hbox = QHBoxLayout(page)
        hbox.setContentsMargins(4, 2, 4, 2)
        hbox.setSpacing(4)

        # ---------------------------------------------------------------------
        # 1. Date taken Group (Dates, Months, Years)
        # ---------------------------------------------------------------------
        grp_dates = _RibbonGroupBox("Date taken", page)
        d_hbox = QHBoxLayout()
        d_hbox.setContentsMargins(0, 0, 0, 0)
        d_hbox.setSpacing(2)

        self.btn_find_dates = self._make_tool_btn("Dates", "dates_grid", grp_dates, large=True)
        self.btn_find_dates.setToolTip("Filter by relative date range")
        self._build_dates_menu(self.btn_find_dates)
        d_hbox.addWidget(self.btn_find_dates)

        self.btn_find_months = self._make_tool_btn("Months", "months_grid", grp_dates, large=True)
        self.btn_find_months.setToolTip("Filter by month")
        self._build_months_menu(self.btn_find_months)
        d_hbox.addWidget(self.btn_find_months)

        self.btn_find_years = self._make_tool_btn("Years", "years_stack", grp_dates, large=True)
        self.btn_find_years.setToolTip("Filter by year")
        self._build_years_menu(self.btn_find_years)
        d_hbox.addWidget(self.btn_find_years)

        grp_dates.content_layout.addLayout(d_hbox)
        hbox.addWidget(grp_dates)

        # ---------------------------------------------------------------------
        # 2. People Group (Face Carousel + [+ and / / or])
        # ---------------------------------------------------------------------
        grp_people = _RibbonGroupBox("People", page)
        p_hbox = QHBoxLayout()
        p_hbox.setContentsMargins(0, 0, 0, 0)
        p_hbox.setSpacing(3)

        self.find_face_carousel = _FaceCarouselWidget(grp_people)
        self.find_face_carousel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.find_face_carousel.person_clicked.connect(
            self.person_quick_find_selected.emit
        )
        self.find_face_carousel.boolean_mode_selected.connect(
            self._on_carousel_boolean_mode
        )
        self.find_face_carousel.special_filter_selected.connect(
            self.people_special_filter_selected.emit
        )
        p_hbox.addWidget(self.find_face_carousel)

        p_mode_col = QVBoxLayout()
        p_mode_col.setContentsMargins(0, 0, 0, 0)
        p_mode_col.setSpacing(1)
        self.btn_people_and = self._make_mode_toggle_btn("and", grp_people, height=22, icon_kind="boolean_and")
        self.btn_people_and.setToolTip("Match all selected people (AND)")
        self.btn_people_or = self._make_mode_toggle_btn("or", grp_people, height=22, icon_kind="boolean_or")
        self.btn_people_or.setToolTip("Match any selected person (OR)")
        self.btn_people_or.setChecked(True)

        self._people_btn_group = QButtonGroup(self)
        self._people_btn_group.setExclusive(True)
        self._people_btn_group.addButton(self.btn_people_and)
        self._people_btn_group.addButton(self.btn_people_or)
        self.btn_people_and.toggled.connect(
            lambda checked: self.person_boolean_mode_changed.emit("AND" if checked else "OR")
        )
        p_mode_col.addWidget(self.btn_people_and)
        p_mode_col.addWidget(self.btn_people_or)
        p_hbox.addLayout(p_mode_col)

        grp_people.content_layout.addLayout(p_hbox)
        hbox.addWidget(grp_people)

        # ---------------------------------------------------------------------
        # 3. Rated Group (Rated tool button + [↑ and higher / ↓ and lower / = only])
        # ---------------------------------------------------------------------
        grp_rated = _RibbonGroupBox("Rated", page)
        r_hbox = QHBoxLayout()
        r_hbox.setContentsMargins(0, 0, 0, 0)
        r_hbox.setSpacing(3)

        self.btn_find_rated = self._make_tool_btn("Rated", "rate", grp_rated, large=True)
        self.btn_find_rated.setToolTip("Filter by star rating")
        self._build_rated_menu(self.btn_find_rated)
        r_hbox.addWidget(self.btn_find_rated)

        r_mode_col = QVBoxLayout()
        r_mode_col.setContentsMargins(0, 0, 0, 0)
        r_mode_col.setSpacing(1)

        self.btn_rated_higher = self._make_mode_toggle_btn("and higher", grp_rated, height=21, icon_kind="rated_higher")
        self.btn_rated_higher.setToolTip("Rated this or higher")
        self.btn_rated_lower = self._make_mode_toggle_btn("and lower", grp_rated, height=21, icon_kind="rated_lower")
        self.btn_rated_lower.setToolTip("Rated this or lower")
        self.btn_rated_only = self._make_mode_toggle_btn("only", grp_rated, height=21, icon_kind="rated_only")
        self.btn_rated_only.setToolTip("Rated this only")
        self.btn_rated_higher.setChecked(True)

        self._rated_btn_group = QButtonGroup(self)
        self._rated_btn_group.setExclusive(True)
        self._rated_btn_group.addButton(self.btn_rated_higher)
        self._rated_btn_group.addButton(self.btn_rated_lower)
        self._rated_btn_group.addButton(self.btn_rated_only)

        self.btn_rated_higher.toggled.connect(self._on_rating_mode_changed)
        self.btn_rated_lower.toggled.connect(self._on_rating_mode_changed)
        self.btn_rated_only.toggled.connect(self._on_rating_mode_changed)

        r_mode_col.addWidget(self.btn_rated_higher)
        r_mode_col.addWidget(self.btn_rated_lower)
        r_mode_col.addWidget(self.btn_rated_only)
        r_hbox.addLayout(r_mode_col)

        grp_rated.content_layout.addLayout(r_hbox)
        hbox.addWidget(grp_rated)

        # ---------------------------------------------------------------------
        # 4. File details Group (Tags, Flagged, Edited, Media type)
        # ---------------------------------------------------------------------
        grp_file = _RibbonGroupBox("File details", page)
        f_hbox = QHBoxLayout()
        f_hbox.setContentsMargins(0, 0, 0, 0)
        f_hbox.setSpacing(3)

        self.btn_find_tags = self._make_tool_btn("Tags", "tag", grp_file, large=True)
        self.btn_find_tags.setToolTip("Filter by descriptive tags")
        self.menu_find_tags = WLPGTagsMenu(self)
        self.btn_find_tags.setMenu(self.menu_find_tags)
        f_hbox.addWidget(self.btn_find_tags)

        self.btn_find_flagged = self._make_tool_btn("Flagged", "flag", grp_file, large=True)
        self.btn_find_flagged.setToolTip("Filter by flagged items")
        self._build_flagged_menu(self.btn_find_flagged)
        f_hbox.addWidget(self.btn_find_flagged)

        f_col = QVBoxLayout()
        f_col.setContentsMargins(0, 0, 0, 0)
        f_col.setSpacing(2)

        self.btn_find_orientation = self._make_tool_btn("Orientation", "orientation", grp_file, large=False)
        self.btn_find_orientation.setToolTip("Filter by landscape or portrait orientation")
        self.btn_find_orientation.setFixedHeight(22)
        self.btn_find_orientation.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._build_orientation_menu(self.btn_find_orientation)
        f_col.addWidget(self.btn_find_orientation)

        self.btn_find_edited = self._make_tool_btn(
            "Edited", "edited", grp_file, large=False, popup_mode=QToolButton.MenuButtonPopup
        )
        self.btn_find_edited.setCheckable(True)
        self.btn_find_edited.setFixedHeight(22)
        self.btn_find_edited.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._build_edited_menu(self.btn_find_edited)
        self.btn_find_edited.clicked.connect(self._on_find_edited_clicked)
        f_col.addWidget(self.btn_find_edited)

        self.btn_find_media_type = self._make_tool_btn("Media type", "media_type", grp_file, large=False)
        self.btn_find_media_type.setToolTip("Filter by photos or videos")
        self.btn_find_media_type.setFixedHeight(22)
        self.btn_find_media_type.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._build_media_type_menu(self.btn_find_media_type)
        f_col.addWidget(self.btn_find_media_type)

        f_hbox.addLayout(f_col)
        grp_file.content_layout.addLayout(f_hbox)
        hbox.addWidget(grp_file)

        self.find_file_edited_media = [
            self.btn_find_orientation,
            self.btn_find_edited,
            self.btn_find_media_type,
        ]

        # ---------------------------------------------------------------------
        # 5. Text search Group (Binoculars button, Ctrl+F)
        # ---------------------------------------------------------------------
        grp_search = _RibbonGroupBox("Text search", page)
        self.btn_find_text_search = self._make_btn("Text search", "text_search", grp_search, large=True)
        self.btn_find_text_search.setToolTip("Find text (Ctrl+F)")
        self.btn_find_text_search.clicked.connect(self.text_search_toggle_requested.emit)
        grp_search.content_layout.addWidget(self.btn_find_text_search)
        hbox.addWidget(grp_search)

        # ---------------------------------------------------------------------
        # 6. Clean up Group (Find Duplicates & Bursts)
        # ---------------------------------------------------------------------
        grp_cleanup = _RibbonGroupBox("Clean up", page)
        self.btn_find_duplicates = self._make_btn("Find\nduplicates", "duplicates", grp_cleanup, large=True)
        self.btn_find_duplicates.setToolTip("Find identical, similar, and burst photos to compare and recover disk space")
        self.btn_find_duplicates.clicked.connect(self.find_duplicates_requested.emit)
        grp_cleanup.content_layout.addWidget(self.btn_find_duplicates)
        hbox.addWidget(grp_cleanup)

        # Authentic stretch pushes all groups compactly to the left
        hbox.addStretch(1)

        return page

    # -------------------------------------------------------------------------
    # Find Tab Menu Builders & Signal Dispatchers
    # -------------------------------------------------------------------------
    def _build_dates_menu(self, btn: QToolButton) -> None:
        menu = QMenu(btn)
        dates_items = [
            ("Today", "today"),
            ("Yesterday", "yesterday"),
            ("This week", "this_week"),
            ("Last week", "last_week"),
            ("This month", "this_month"),
            ("Last month", "last_month"),
            ("This year", "this_year"),
            ("Last year", "last_year"),
        ]
        for label, code in dates_items:
            act = menu.addAction(label)
            act.triggered.connect(lambda _c=False, cd=code: self._on_date_code_selected(cd))
        menu.addSeparator()
        act_clear = menu.addAction("Clear date filter")
        act_clear.triggered.connect(lambda: self.date_filter_selected.emit(None, None, None))
        btn.setMenu(menu)

    def _on_date_code_selected(self, code: str) -> None:
        now = datetime.datetime.now()
        if code == "today":
            self.date_filter_selected.emit(now.year, now.month, now.day)
        elif code == "yesterday":
            y = now - datetime.timedelta(days=1)
            self.date_filter_selected.emit(y.year, y.month, y.day)
        elif code == "this_week":
            start = now - datetime.timedelta(days=now.weekday())
            self.date_filter_selected.emit(start.year, start.month, start.day)
        elif code == "last_week":
            start = now - datetime.timedelta(days=now.weekday() + 7)
            self.date_filter_selected.emit(start.year, start.month, start.day)
        elif code == "this_month":
            self.date_filter_selected.emit(now.year, now.month, None)
        elif code == "last_month":
            month = now.month - 1 if now.month > 1 else 12
            year = now.year if now.month > 1 else now.year - 1
            self.date_filter_selected.emit(year, month, None)
        elif code == "this_year":
            self.date_filter_selected.emit(now.year, None, None)
        elif code == "last_year":
            self.date_filter_selected.emit(now.year - 1, None, None)

    def _build_months_menu(self, btn: QToolButton) -> None:
        menu = QMenu(btn)
        for m in range(1, 13):
            month_name = calendar.month_name[m]
            act = menu.addAction(month_name)
            act.triggered.connect(lambda _c=False, mn=m: self.date_filter_selected.emit(None, mn, None))
        menu.addSeparator()
        act_clear = menu.addAction("Clear month filter")
        act_clear.triggered.connect(lambda: self.date_filter_selected.emit(None, None, None))
        btn.setMenu(menu)

    def _build_years_menu(self, btn: QToolButton) -> None:
        menu = QMenu(btn)
        current_year = datetime.datetime.now().year
        for yr in range(current_year, current_year - 12, -1):
            act = menu.addAction(str(yr))
            act.triggered.connect(lambda _c=False, y=yr: self.date_filter_selected.emit(y, None, None))
        menu.addSeparator()
        act_clear = menu.addAction("Clear year filter")
        act_clear.triggered.connect(lambda: self.date_filter_selected.emit(None, None, None))
        btn.setMenu(menu)

    def _build_rated_menu(self, btn: QToolButton) -> None:
        menu = create_wlpg_rating_menu(
            btn,
            on_selected=lambda v: self._on_rating_item_clicked(v),
            show_all_ratings=True,
            is_dark=getattr(self, "_dark_mode", False),
        )
        btn.setMenu(menu)

    def _on_rating_item_clicked(self, val: int) -> None:
        self._last_rated_val = val
        self._emit_current_rating_filter()

    def _on_rating_mode_changed(self, _checked: bool = True) -> None:
        if self._last_rated_val >= 0:
            self._emit_current_rating_filter()

    def _emit_current_rating_filter(self) -> None:
        val = self._last_rated_val
        if val == -1:
            self.rating_filter_selected.emit(())
            return
        if val == 0:
            self.rating_filter_selected.emit((0,))
            return
        if self.btn_rated_higher.isChecked():
            ratings = tuple(range(val, 6))
        elif self.btn_rated_lower.isChecked():
            ratings = tuple(range(1, val + 1))
        else:
            ratings = (val,)
        self.rating_filter_selected.emit(ratings)

    def _build_orientation_menu(self, btn: QToolButton) -> None:
        menu = QMenu(btn)
        menu.setFont(QFont("Segoe UI", 9))
        grp = QActionGroup(menu)
        grp.setExclusive(True)

        act_all = menu.addAction("All orientations")
        act_all.setCheckable(True)
        act_all.setChecked(True)
        grp.addAction(act_all)

        act_landscape = menu.addAction("Landscape")
        act_landscape.setCheckable(True)
        grp.addAction(act_landscape)

        act_portrait = menu.addAction("Portrait")
        act_portrait.setCheckable(True)
        grp.addAction(act_portrait)

        self._orientation_menu_actions = {
            "all": act_all,
            "landscape": act_landscape,
            "portrait": act_portrait,
        }

        act_all.triggered.connect(lambda: self._on_orientation_filter_selected("all", "All orientations"))
        act_landscape.triggered.connect(lambda: self._on_orientation_filter_selected("landscape", "Landscape"))
        act_portrait.triggered.connect(lambda: self._on_orientation_filter_selected("portrait", "Portrait"))
        btn.setMenu(menu)

    def _on_orientation_filter_selected(self, mode: str, label: str) -> None:
        self._current_orientation_filter = mode
        if hasattr(self, "btn_find_orientation"):
            if mode != "all":
                self.btn_find_orientation.setToolTip(f"Filtering: {label} photos (click menu to change)")
            else:
                self.btn_find_orientation.setToolTip("Filter by landscape or portrait orientation")
        self.orientation_filter_selected.emit(mode)

    def set_orientation_filter_ui(self, mode: str) -> None:
        self._current_orientation_filter = mode
        actions = getattr(self, "_orientation_menu_actions", {})
        if mode in actions:
            actions[mode].setChecked(True)
        if hasattr(self, "btn_find_orientation"):
            if mode != "all":
                label = "Landscape" if mode == "landscape" else "Portrait"
                self.btn_find_orientation.setToolTip(f"Filtering: {label} photos (click menu to change)")
            else:
                self.btn_find_orientation.setToolTip("Filter by landscape or portrait orientation")

    def _build_media_type_menu(self, btn: QToolButton) -> None:
        menu = QMenu(btn)
        act_all = menu.addAction("All media")
        act_photos = menu.addAction("Photos only")
        act_videos = menu.addAction("Videos only")
        act_raw = menu.addAction("RAW photos only")
        act_all.triggered.connect(lambda: self.media_type_filter_selected.emit("all"))
        act_photos.triggered.connect(lambda: self.media_type_filter_selected.emit("image"))
        act_videos.triggered.connect(lambda: self.media_type_filter_selected.emit("video"))
        act_raw.triggered.connect(lambda: self.media_type_filter_selected.emit("raw"))
        btn.setMenu(menu)

    def populate_tags_menu(self, tag_tree: Sequence[dict[str, Any]]) -> None:
        """Populates the Find tab's authentic WLPG 'Tags' popup menu."""
        if hasattr(self, "menu_find_tags") and isinstance(self.menu_find_tags, WLPGTagsMenu):
            self.menu_find_tags.populate(tag_tree)
        elif hasattr(self, "menu_find_tags"):
            self.menu_find_tags.clear()
            for t in tag_tree:
                tid = int(t["id"])
                name = str(t.get("name", t.get("full_path", "Tag")))
                act = self.menu_find_tags.addAction(name)
                act.triggered.connect(
                    lambda _c=False, tag_id=tid: self.tag_filter_selected.emit(tag_id, "include")
                )

    def is_people_and_mode(self) -> bool:
        return bool(hasattr(self, "btn_people_and") and self.btn_people_and.isChecked())

    def _on_carousel_boolean_mode(self, mode: str) -> None:
        if hasattr(self, "btn_people_and") and hasattr(self, "btn_people_or"):
            if mode == "AND":
                self.btn_people_and.setChecked(True)
            else:
                self.btn_people_or.setChecked(True)
        self.person_boolean_mode_changed.emit(mode)

    def _build_create_tab(self) -> QWidget:
        page = _RibbonPage()
        hbox = QHBoxLayout(page)
        hbox.setContentsMargins(4, 2, 4, 2)
        hbox.setSpacing(4)

        # ---------------------------------------------------------------------
        # Group 1: Tools (Panorama, Photo Fuse, Auto Collage, etc.)
        # ---------------------------------------------------------------------
        grp_tools = _RibbonGroupBox("Tools", page)
        self.grp_create_tools = grp_tools

        btn_pano = self._make_btn("Panorama", "panoramic", grp_tools, large=True)
        btn_pano.setToolTip("Stitch 2 or more overlapping photos into a wide-angle panorama (Microsoft ICE engine).")
        btn_pano.clicked.connect(lambda: self.action_triggered.emit("panorama"))
        self.btn_panorama = btn_pano
        grp_tools.content_layout.addWidget(btn_pano)

        btn_fuse = self._make_btn("Photo\nFuse", "photo_fuse", grp_tools, large=True)
        btn_fuse.setToolTip("Combine the best expressions and faces from 2–6 burst group shots into a single perfect photo.")
        btn_fuse.clicked.connect(lambda: self.action_triggered.emit("photo_fuse"))
        self.btn_photo_fuse = btn_fuse
        grp_tools.content_layout.addWidget(btn_fuse)

        btn_collage = self._make_btn("Auto\nCollage", "collage", grp_tools, large=True)
        btn_collage.setToolTip("Generate an interactive grid, vintage Polaroid scatter, or 'Then & Now' comparison card from selected photos.")
        btn_collage.clicked.connect(lambda: self.action_triggered.emit("collage"))
        self.btn_collage = btn_collage
        grp_tools.content_layout.addWidget(btn_collage)

        btn_pop = self._make_btn("Colour\nPop", "color_spectrum", grp_tools, large=True)
        btn_pop.setToolTip("Google Photos-Style Colour Pop: keep subject in vibrant colour while making background B&W.")
        btn_pop.clicked.connect(lambda: self.action_triggered.emit("color_pop"))
        self.btn_color_pop = btn_pop
        grp_tools.content_layout.addWidget(btn_pop)

        btn_cinematic = self._make_btn("3D\nCinematic", "slideshow", grp_tools, large=True)
        btn_cinematic.setToolTip("Generate a 3D Cinematic Motion .mp4 virtual camera parallax video from a single photo.")
        btn_cinematic.clicked.connect(lambda: self.action_triggered.emit("cinematic_motion"))
        self.btn_cinematic_motion = btn_cinematic
        grp_tools.content_layout.addWidget(btn_cinematic)

        hbox.addWidget(grp_tools)
        hbox.addStretch(1)
        return page

    def _build_view_tab(self) -> QWidget:
        """'View' tab matching WLPG reference `docs/reference_ui/view_tab.png` and single image view."""
        page = _RibbonPage()
        hbox = QHBoxLayout(page)
        hbox.setContentsMargins(4, 2, 4, 2)
        hbox.setSpacing(4)

        # ---------------------------------------------------------------------
        # Group 1: Arrange list (4x2 button matrix + Rating Gallery Box + Reverse sort)
        # ---------------------------------------------------------------------
        grp_arrange = _RibbonGroupBox("Arrange list", page)
        self.grp_arrange = grp_arrange
        arr_hbox = QHBoxLayout()
        arr_hbox.setContentsMargins(0, 0, 0, 0)
        arr_hbox.setSpacing(4)

        matrix_grid = QGridLayout()
        matrix_grid.setContentsMargins(0, 0, 0, 0)
        matrix_grid.setHorizontalSpacing(2)
        matrix_grid.setVerticalSpacing(2)

        self._arrange_buttons: dict[ArrangeMode, QPushButton | QToolButton] = {}
        self._arrange_btn_group = QButtonGroup(self)
        self._arrange_btn_group.setExclusive(True)

        # 5 columns x 2 rows matching WLPG
        # Col 0: Date, Geotag
        # Col 1: Person, Name
        # Col 2: Tag, Published
        # Col 3: Folder, Media type
        # Col 4: Rating
        matrix_items = [
            (0, 0, "Date", "calendar", "date"),
            (1, 0, "Geotag", "geotag", "geotag"),
            (0, 1, "Person", "person", "person"),
            (1, 1, "Name", "name", "name"),
            (0, 2, "Tag", "tag", "tag"),
            (1, 2, "Published", "published", "published"),
            (0, 3, "Folder", "folder", "folder"),
            (1, 3, "Media type", "media_type", "media_type"),
            (0, 4, "Rating", "star", "rating"),
        ]
        for row, col, label, icon_kind, mode in matrix_items:
            btn = self._make_btn(label, icon_kind, grp_arrange, checkable=True)
            btn.setFixedHeight(22)
            btn.setMinimumWidth(0)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(lambda _c=False, m=mode: self._on_arrange_button_clicked(m))
            matrix_grid.addWidget(btn, row, col)
            self._arrange_buttons[mode] = btn
            self._arrange_btn_group.addButton(btn)

        self.btn_arr_rating = self._arrange_buttons["rating"]

        # Container frame for Arrange list matching Quick Find People & Effects Gallery
        arrange_box = QFrame(grp_arrange)
        arrange_box.setObjectName("RibbonArrangeBox")
        arrange_box.setStyleSheet(
            """
            QFrame#RibbonArrangeBox {
                background: #FFFFFF;
                border: 1px solid #ABBFD3;
                border-radius: 2px;
            }
            """
        )
        box_layout = QHBoxLayout(arrange_box)
        box_layout.setContentsMargins(2, 2, 2, 2)
        box_layout.setSpacing(2)
        box_layout.addLayout(matrix_grid)

        # Mini vertical scroll buttons matching Aero Ribbon gallery
        v_arrows = QVBoxLayout()
        v_arrows.setContentsMargins(0, 0, 0, 0)
        v_arrows.setSpacing(1)

        btn_style = """
            QPushButton {
                border: 1px solid #D2DFEE;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F6F9FD, stop:1 #E2ECF7);
                color: #55687C;
                font-size: 7pt;
                padding: 0;
            }
            QPushButton:hover {
                background: #CCE4F7;
                border-color: #7BAEDB;
            }
            QPushButton:disabled {
                color: #B0C4DE;
                background: #F4F6F9;
                border-color: #DCE6F2;
            }
        """

        btn_up = QPushButton("▲", arrange_box)
        btn_up.setFixedSize(14, 19)
        btn_up.setFlat(True)
        btn_up.setCursor(Qt.PointingHandCursor)
        btn_up.setStyleSheet(btn_style)

        btn_dn = QPushButton("▼", arrange_box)
        btn_dn.setFixedSize(14, 19)
        btn_dn.setFlat(True)
        btn_dn.setCursor(Qt.PointingHandCursor)
        btn_dn.setStyleSheet(btn_style)

        btn_more = QPushButton("▾", arrange_box)
        btn_more.setFixedSize(14, 19)
        btn_more.setFlat(True)
        btn_more.setCursor(Qt.PointingHandCursor)
        btn_more.setStyleSheet(btn_style)

        self.btn_arrange_more = btn_more
        btn_more.clicked.connect(self._show_arrange_more_menu)
        v_arrows.addWidget(btn_up)
        v_arrows.addWidget(btn_dn)
        v_arrows.addWidget(btn_more)
        box_layout.addLayout(v_arrows)
        arr_hbox.addWidget(arrange_box)

        # Large Reverse sort button (with dual blue vertical arrows)
        self.btn_reverse_sort = self._make_btn("Reverse\nsort", "reverse_sort", grp_arrange, large=True, checkable=True)
        self.btn_reverse_sort.setFixedWidth(52)
        self.btn_reverse_sort.clicked.connect(self._on_reverse_sort_clicked)
        arr_hbox.addWidget(self.btn_reverse_sort)
        self.chk_reverse_sort = self.btn_reverse_sort  # Legacy alias

        # Hidden QComboBox for backward test compatibility
        self.cmb_arrange = QComboBox(page)
        self.cmb_arrange.hide()
        for _, _, label, _, mode in matrix_items:
            self.cmb_arrange.addItem(label, mode)
        self.cmb_arrange.addItem("Rating", "rating")

        # Set default active
        self._arrange_buttons["date"].setChecked(True)

        grp_arrange.content_layout.addLayout(arr_hbox)
        hbox.addWidget(grp_arrange)

        # ---------------------------------------------------------------------
        # Group 2: Show details (Large Rating, Large Caption, 2x3 checkboxes, Large Flags)
        # ---------------------------------------------------------------------
        grp_details = _RibbonGroupBox("Show details", page)
        self.grp_details = grp_details
        det_hbox = QHBoxLayout()
        det_hbox.setContentsMargins(0, 0, 0, 0)
        det_hbox.setSpacing(4)

        self.btn_detail_rating = self._make_btn("Rating", "star", grp_details, large=True, checkable=True)
        self.btn_detail_rating.setChecked("rating" in self._visible_details)
        det_hbox.addWidget(self.btn_detail_rating)

        self.btn_detail_caption = self._make_btn("Caption", "caption", grp_details, large=True, checkable=True)
        self.btn_detail_caption.setChecked("caption" in self._visible_details)
        det_hbox.addWidget(self.btn_detail_caption)

        chk_col1 = QVBoxLayout()
        chk_col1.setContentsMargins(0, 0, 0, 0)
        chk_col1.setSpacing(1)

        self.chk_all_details = self._make_btn("All details", "detail_all", grp_details, large=False, checkable=True)
        self.chk_filename = self._make_btn("File name", "detail_filename", grp_details, large=False, checkable=True)
        self.chk_filesize = self._make_btn("File size", "detail_filesize", grp_details, large=False, checkable=True)

        chk_col1.addWidget(self.chk_all_details)
        chk_col1.addWidget(self.chk_filename)
        chk_col1.addWidget(self.chk_filesize)

        chk_col2 = QVBoxLayout()
        chk_col2.setContentsMargins(0, 0, 0, 0)
        chk_col2.setSpacing(1)

        self.chk_date_taken = self._make_btn("Date taken", "detail_date_taken", grp_details, large=False, checkable=True)
        self.chk_dimensions = self._make_btn("Image size", "detail_dimensions", grp_details, large=False, checkable=True)
        self.chk_date_modified = self._make_btn("Date modified", "detail_date_modified", grp_details, large=False, checkable=True)

        chk_col2.addWidget(self.chk_date_taken)
        chk_col2.addWidget(self.chk_dimensions)
        chk_col2.addWidget(self.chk_date_modified)

        det_hbox.addLayout(chk_col1)
        det_hbox.addLayout(chk_col2)

        self.btn_detail_flags = self._make_btn("Flags", "flag", grp_details, large=True, checkable=True)
        self.btn_detail_flags.setChecked("flag" in self._visible_details)
        det_hbox.addWidget(self.btn_detail_flags)

        self._detail_checkboxes = {
            "rating": self.btn_detail_rating,
            "caption": self.btn_detail_caption,
            "flag": self.btn_detail_flags,
            "filename": self.chk_filename,
            "size": self.chk_filesize,
            "date_taken": self.chk_date_taken,
            "dimensions": self.chk_dimensions,
            "date_modified": self.chk_date_modified,
        }

        for chk in self._detail_checkboxes.values():
            chk.toggled.connect(self._on_detail_checkbox_toggled)

        self.chk_all_details.toggled.connect(self._on_all_details_toggled)

        self.view_details_checkboxes = [
            self.chk_all_details,
            self.chk_filename,
            self.chk_filesize,
            self.chk_date_taken,
            self.chk_dimensions,
            self.chk_date_modified,
        ]
        self.view_arrange_buttons = list(self._arrange_buttons.values())

        grp_details.content_layout.addLayout(det_hbox)
        hbox.addWidget(grp_details)

        # ---------------------------------------------------------------------
        # Group 3: Zoom Group (two large buttons matching WLPG in Grid, four in Viewer)
        # ---------------------------------------------------------------------
        grp_zoom = _RibbonGroupBox("Zoom", page)
        self.grp_zoom = grp_zoom
        grp_zoom.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        grp_zoom.content_layout.setSpacing(3)

        self.btn_zoom_in = self._make_btn("Zoom\nin", "zoom_in", grp_zoom, large=True)
        self.btn_zoom_in.setToolTip("Zoom in (+)")
        self.btn_zoom_in._tooltip_override = "Zoom in (+)"
        self.btn_zoom_in.clicked.connect(lambda: self.action_triggered.emit("zoom_in"))
        grp_zoom.content_layout.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = self._make_btn("Zoom\nout", "zoom_out", grp_zoom, large=True)
        self.btn_zoom_out.setToolTip("Zoom out (-)")
        self.btn_zoom_out._tooltip_override = "Zoom out (-)"
        self.btn_zoom_out.clicked.connect(lambda: self.action_triggered.emit("zoom_out"))
        grp_zoom.content_layout.addWidget(self.btn_zoom_out)

        self.btn_zoom_actual = self._make_btn("Actual\nsize", "actual_size", grp_zoom, large=True)
        self.btn_zoom_actual.setToolTip("Actual size (Ctrl+1)")
        self.btn_zoom_actual._tooltip_override = "Actual size (Ctrl+1)"
        self.btn_zoom_actual.clicked.connect(lambda: self.action_triggered.emit("viewer_zoom_actual"))
        self.btn_zoom_actual.setVisible(False)
        grp_zoom.content_layout.addWidget(self.btn_zoom_actual)

        self.btn_zoom_fit = self._make_btn("Fit to\nwindow", "fit_window", grp_zoom, large=True)
        self.btn_zoom_fit.clicked.connect(lambda: self.action_triggered.emit("viewer_zoom_fit"))
        self.btn_zoom_fit.setVisible(False)
        grp_zoom.content_layout.addWidget(self.btn_zoom_fit)

        # Thumbnail zoom slider kept for API/status bar compatibility, hidden from ribbon to match WLPG
        self.slider_zoom = QSlider(Qt.Horizontal, self)
        self.slider_zoom.setRange(72, 320)
        self.slider_zoom.setValue(150)
        self.slider_zoom.setFixedWidth(80)
        self.slider_zoom.valueChanged.connect(self.zoom_size_changed.emit)
        self.slider_zoom.setVisible(False)

        grp_zoom.adjust_to_content()
        hbox.addWidget(grp_zoom)

        # ---------------------------------------------------------------------
        # Group 4: Views Group (Thumbnails, World Map, Slide show)
        # ---------------------------------------------------------------------
        grp_slideshow = _RibbonGroupBox("Views", page)
        self.grp_slideshow = grp_slideshow

        self.btn_view_thumbnails = self._make_btn(
            "Gallery", "layout_grid", grp_slideshow, large=True, checkable=True
        )
        self.btn_view_thumbnails.setToolTip("View gallery photo and video thumbnails (Ctrl+0)")
        self.btn_view_thumbnails.setChecked(True)
        self.btn_view_thumbnails.clicked.connect(
            lambda: self.action_triggered.emit("show_thumbnails_view")
        )
        grp_slideshow.content_layout.addWidget(self.btn_view_thumbnails)

        self.btn_view_map = self._make_btn("World\nMap", "globe", grp_slideshow, large=True, checkable=True)
        self.btn_view_map.setToolTip("Explore geotagged photos on the World Map (Leaflet / Offline)")
        self.btn_view_map.clicked.connect(lambda: self.action_triggered.emit("toggle_map_view"))
        grp_slideshow.content_layout.addWidget(self.btn_view_map)

        split_view, self.btn_view_slideshow = self._make_slideshow_split_button(grp_slideshow)
        grp_slideshow.content_layout.addWidget(split_view)
        hbox.addWidget(grp_slideshow)

        # ---------------------------------------------------------------------
        # Group 5: Panes Group (Tag & Caption pane matching WLPG)
        # ---------------------------------------------------------------------
        grp_panes = _RibbonGroupBox("Panes", page)
        self.grp_panes = grp_panes

        self.btn_toggle_nav_pane = self._make_btn(
            "Navigation\npane", "pane_left", grp_panes, large=True, checkable=True
        )
        self.btn_toggle_nav_pane.setToolTip("Navigation pane (Ctrl+N)")
        self.btn_toggle_nav_pane._tooltip_override = "Navigation pane (Ctrl+N)"
        self.btn_toggle_nav_pane.setChecked(True)
        self.btn_toggle_nav_pane.clicked.connect(
            lambda: self.action_triggered.emit("toggle_nav_pane")
        )
        grp_panes.content_layout.addWidget(self.btn_toggle_nav_pane)

        self.btn_toggle_tag_pane = self._make_btn(
            "Tag and\ncaption pane", "pane_right", grp_panes, large=True, checkable=True
        )
        self.btn_toggle_tag_pane.setToolTip("Tag and caption pane (Ctrl+I)")
        self.btn_toggle_tag_pane._tooltip_override = "Tag and caption pane (Ctrl+I)"
        self.btn_toggle_tag_pane.clicked.connect(
            lambda: self.action_triggered.emit("toggle_tag_pane")
        )
        grp_panes.content_layout.addWidget(self.btn_toggle_tag_pane)
        grp_panes.adjust_to_content()

        hbox.addWidget(grp_panes)

        # ---------------------------------------------------------------------
        # Group 6: Photo layout (Grid mode only - keeping custom addition)
        # ---------------------------------------------------------------------
        grp_layout = _RibbonGroupBox("Photo layout", page)
        self.grp_layout = grp_layout
        self.btn_layout_grid = self._make_btn(
            "Grid", "layout_grid", grp_layout, large=True, checkable=True
        )
        self.btn_layout_grid.setChecked(True)
        self.btn_layout_mosaic = self._make_btn(
            "Mosaic", "layout_mosaic", grp_layout, large=True, checkable=True
        )

        self._layout_btn_group = QButtonGroup(self)
        self._layout_btn_group.setExclusive(True)
        self._layout_btn_group.addButton(self.btn_layout_grid)
        self._layout_btn_group.addButton(self.btn_layout_mosaic)

        self.btn_layout_grid.clicked.connect(
            lambda: self.layout_mode_changed.emit("wlpg_grid")
        )
        self.btn_layout_mosaic.clicked.connect(
            lambda: self.layout_mode_changed.emit("mosaic")
        )
        l_hbox = QHBoxLayout()
        l_hbox.setContentsMargins(0, 0, 0, 0)
        l_hbox.setSpacing(3)
        l_hbox.addWidget(self.btn_layout_grid)
        l_hbox.addWidget(self.btn_layout_mosaic)
        grp_layout.content_layout.addLayout(l_hbox)
        hbox.addWidget(grp_layout)

        hbox.addStretch(1)
        return page

    # -------------------------------------------------------------------------
    # Public Control & Inspection API
    # -------------------------------------------------------------------------
    def populate_face_carousel(self, people: Sequence[dict[str, Any]]) -> None:
        self.home_face_carousel.populate(people)
        self.find_face_carousel.populate(people)

    def set_arrange_by(
        self, mode: ArrangeMode, reverse_sort: bool | None = None
    ) -> None:
        self._arrange_by = mode
        if reverse_sort is not None:
            self._reverse_sort = bool(reverse_sort)

        if hasattr(self, "_arrange_buttons") and mode in self._arrange_buttons:
            btn = self._arrange_buttons[mode]
            btn.blockSignals(True)
            btn.setChecked(True)
            btn.blockSignals(False)

        if hasattr(self, "cmb_arrange"):
            self.cmb_arrange.blockSignals(True)
            for i in range(self.cmb_arrange.count()):
                if self.cmb_arrange.itemData(i) == mode:
                    self.cmb_arrange.setCurrentIndex(i)
                    break
            self.cmb_arrange.blockSignals(False)

        if hasattr(self, "btn_reverse_sort"):
            self.btn_reverse_sort.blockSignals(True)
            self.btn_reverse_sort.setChecked(self._reverse_sort)
            self.btn_reverse_sort.blockSignals(False)

        self.arrange_changed.emit(self._arrange_by, self._reverse_sort)

    def current_arrange_state(self) -> tuple[ArrangeMode, bool]:
        return (self._arrange_by, self._reverse_sort)

    def _on_arrange_button_clicked(self, mode: ArrangeMode) -> None:
        self._arrange_by = mode
        if hasattr(self, "_arrange_buttons") and mode in self._arrange_buttons:
            self._arrange_buttons[mode].setChecked(True)
        elif hasattr(self, "_arrange_btn_group"):
            checked_b = self._arrange_btn_group.checkedButton()
            if checked_b:
                self._arrange_btn_group.setExclusive(False)
                checked_b.setChecked(False)
                self._arrange_btn_group.setExclusive(True)
        if hasattr(self, "cmb_arrange"):
            self.cmb_arrange.blockSignals(True)
            for i in range(self.cmb_arrange.count()):
                if self.cmb_arrange.itemData(i) == mode:
                    self.cmb_arrange.setCurrentIndex(i)
                    break
            self.cmb_arrange.blockSignals(False)
        self.arrange_changed.emit(self._arrange_by, self._reverse_sort)

    def _show_arrange_more_menu(self) -> None:
        menu = QMenu(self)
        menu.setFont(QFont("Segoe UI", 9))
        menu.setStyleSheet(
            """
            QMenu {
                background-color: #FFFFFF;
                border: 1px solid #70A6E8;
                border-radius: 3px;
                padding: 4px;
            }
            QMenu::item {
                padding: 3px 20px 3px 24px;
                color: #1E395B;
            }
            QMenu::item:selected {
                background-color: #EBF3FD;
                color: #0F172A;
            }
            QMenu::separator {
                height: 1px;
                background-color: #DDE7F2;
                margin: 4px 6px;
            }
            """
        )
        grp = QActionGroup(menu)
        grp.setExclusive(True)

        group_criteria = [
            ("Auto", "auto"),
            ("Date taken", "date"),
            ("Month taken", "month"),
            ("Year taken", "year"),
            ("Time taken", "time"),
            ("File size", "size"),
            ("Image size", "image_size"),
            ("Orientation", "orientation"),
            ("File type", "file_type"),
            ("Media type", "media_type"),
            ("Rating", "rating"),
            ("Flag", "flag"),
            ("Folder", "folder"),
            ("Tag", "tag"),
            ("People", "people"),
            ("Published", "published"),
            ("Geotag", "geotag"),
            ("Camera", "camera"),
            ("None", "none"),
        ]

        curr = getattr(self, "_arrange_by", "date")

        for label, mode_key in group_criteria:
            act = menu.addAction(label)
            act.setCheckable(True)
            grp.addAction(act)
            if mode_key == curr:
                act.setChecked(True)
            elif curr == "date" and mode_key == "date":
                act.setChecked(True)

            def _make_handler(m=mode_key):
                return lambda: self._on_arrange_button_clicked(m)

            act.triggered.connect(_make_handler(mode_key))

        menu.addSeparator()

        act_asc = menu.addAction("Ascending")
        act_asc.setCheckable(True)
        act_desc = menu.addAction("Descending")
        act_desc.setCheckable(True)
        dir_grp = QActionGroup(menu)
        dir_grp.setExclusive(True)
        dir_grp.addAction(act_asc)
        dir_grp.addAction(act_desc)

        is_rev = getattr(self, "_reverse_sort", False)
        if is_rev:
            act_desc.setChecked(True)
        else:
            act_asc.setChecked(True)

        act_asc.triggered.connect(lambda: self._set_reverse_sort(False))
        act_desc.triggered.connect(lambda: self._set_reverse_sort(True))

        btn = getattr(self, "btn_arrange_more", None)
        if btn is not None:
            menu.exec(btn.mapToGlobal(QPoint(0, btn.height())))
        else:
            menu.exec(QCursor.pos())

    def _set_reverse_sort(self, reverse: bool) -> None:
        self._reverse_sort = bool(reverse)
        if hasattr(self, "btn_reverse_sort"):
            self.btn_reverse_sort.setChecked(self._reverse_sort)
        self.arrange_changed.emit(self._arrange_by, self._reverse_sort)

    def _on_reverse_sort_clicked(self) -> None:
        self._reverse_sort = self.btn_reverse_sort.isChecked()
        self.arrange_changed.emit(self._arrange_by, self._reverse_sort)

    def _on_arrange_ui_changed(self) -> None:
        mode = self.cmb_arrange.currentData()
        self._arrange_by = mode if mode else "date"
        self._reverse_sort = self.chk_reverse_sort.isChecked()
        self.arrange_changed.emit(self._arrange_by, self._reverse_sort)

    def set_detail_flag(self, key: str, enabled: bool) -> None:
        if key in self._detail_checkboxes:
            chk = self._detail_checkboxes[key]
            chk.blockSignals(True)
            chk.setChecked(bool(enabled))
            chk.blockSignals(False)
            self._on_detail_checkbox_toggled()

    def current_visible_details(self) -> set[str]:
        return set(self._visible_details)

    def _on_all_details_toggled(self, checked: bool) -> None:
        sub_chks = (
            self.chk_filename,
            self.chk_filesize,
            self.chk_date_taken,
            self.chk_dimensions,
            self.chk_date_modified,
        )
        for chk in sub_chks:
            chk.blockSignals(True)
            chk.setChecked(checked)
            chk.blockSignals(False)
        self._on_detail_checkbox_toggled()

    def _on_detail_checkbox_toggled(self) -> None:
        active = {
            k for k, chk in self._detail_checkboxes.items() if chk.isChecked()
        }
        self._visible_details = active
        sub_chks = (
            self.chk_filename,
            self.chk_filesize,
            self.chk_date_taken,
            self.chk_dimensions,
            self.chk_date_modified,
        )
        all_on = all(c.isChecked() for c in sub_chks)
        if hasattr(self, "chk_all_details"):
            self.chk_all_details.blockSignals(True)
            self.chk_all_details.setChecked(all_on)
            self.chk_all_details.blockSignals(False)
        self.details_overlay_changed.emit(set(active))

    def set_layout_mode(self, mode: Literal["wlpg_grid", "mosaic"]) -> None:
        if mode == "mosaic":
            self.btn_layout_mosaic.setChecked(True)
        else:
            self.btn_layout_grid.setChecked(True)
        self.layout_mode_changed.emit(mode)

    def set_scope_label(self, text: str) -> None:
        if hasattr(self, "lbl_scope") and self.lbl_scope is not None:
            try:
                self.lbl_scope.setText(text)
            except RuntimeError:
                pass
        if hasattr(self, "scope_bar") and hasattr(self.scope_bar, "set_scope_label"):
            try:
                self.scope_bar.set_scope_label(text)
            except RuntimeError:
                pass

    def set_search_text(self, text: str, immediate: bool = False) -> None:
        if hasattr(self, "txt_search") and self.txt_search is not None:
            try:
                self.txt_search.blockSignals(True)
                self.txt_search.setText(text)
                self.txt_search.blockSignals(False)
            except RuntimeError:
                pass
        if immediate:
            self._search_timer.stop()
            self.search_text_changed.emit(text.strip())
        else:
            self._search_timer.start()

    def _on_search_input_changed(self, _text: str) -> None:
        self._search_timer.start()

    def _emit_search_text(self) -> None:
        if hasattr(self, "txt_search") and self.txt_search is not None:
            try:
                self.search_text_changed.emit(self.txt_search.text().strip())
            except RuntimeError:
                pass

    def _build_batch_people_tab(self) -> QWidget:
        page = _RibbonPage()
        hbox = QHBoxLayout(page)
        hbox.setContentsMargins(4, 2, 4, 2)
        hbox.setSpacing(4)

        # ---------------------------------------------------------------------
        # Group 1: Selection
        # ---------------------------------------------------------------------
        grp_sel = _RibbonGroupBox("Selection", page)
        self.btn_batch_select_all = self._make_btn("Select all", "select_all", grp_sel, large=True)
        self.btn_batch_select_all.setToolTip("Select all face suggestions (Ctrl+A)")
        self.btn_batch_select_all.clicked.connect(lambda: self.action_triggered.emit("batch_select_all"))
        grp_sel.content_layout.addWidget(self.btn_batch_select_all)
        hbox.addWidget(grp_sel)

        # ---------------------------------------------------------------------
        # Group 2: Actions
        # ---------------------------------------------------------------------
        grp_act = _RibbonGroupBox("Actions", page)
        self.btn_batch_confirm = self._make_btn("Confirm", "check", grp_act, large=True)
        self.btn_batch_confirm.setToolTip("Confirm selected face tags (Enter)")
        self.btn_batch_confirm.clicked.connect(lambda: self.action_triggered.emit("batch_confirm"))

        self.btn_batch_reject = self._make_btn("Not this\nperson", "delete", grp_act, large=True)
        self.btn_batch_reject.setToolTip("Reject suggestion / not this person (Del)")
        self.btn_batch_reject.clicked.connect(lambda: self.action_triggered.emit("batch_reject"))

        self.btn_batch_tag_as = self._make_btn("Tag as", "tag_as", grp_act, large=True)
        self.btn_batch_tag_as.setToolTip("Tag selected face(s) as another person...")
        self.btn_batch_tag_as.clicked.connect(lambda: self.action_triggered.emit("batch_tag_as"))

        self.btn_batch_ignore = self._make_btn("Don't show\nagain", "person_none", grp_act, large=True)
        self.btn_batch_ignore.setToolTip("Ignore / don't suggest this face again (I)")
        self.btn_batch_ignore.clicked.connect(lambda: self.action_triggered.emit("batch_ignore"))

        grp_act.content_layout.addWidget(self.btn_batch_confirm)
        grp_act.content_layout.addWidget(self.btn_batch_reject)
        grp_act.content_layout.addWidget(self.btn_batch_tag_as)
        grp_act.content_layout.addWidget(self.btn_batch_ignore)
        hbox.addWidget(grp_act)

        # ---------------------------------------------------------------------
        # Group 3: Health & Filter
        # ---------------------------------------------------------------------
        grp_health = _RibbonGroupBox("Health & Mode", page)
        v_modes = QVBoxLayout()
        v_modes.setContentsMargins(0, 0, 0, 0)
        v_modes.setSpacing(1)

        self.btn_batch_mode_sugg = self._make_btn("Suggestions", "batch_people", grp_health, large=False, checkable=True)
        self.btn_batch_mode_sugg.setChecked(True)
        self.btn_batch_mode_sugg.setToolTip("View pending face recognition suggestions")
        self.btn_batch_mode_sugg.clicked.connect(lambda: self._on_batch_mode_clicked("suggestions"))

        self.btn_batch_mode_outliers = self._make_btn("Review Outliers", "warning", grp_health, large=False, checkable=True)
        self.btn_batch_mode_outliers.setToolTip("Review potential mis-tagged faces or visual outliers")
        self.btn_batch_mode_outliers.clicked.connect(lambda: self._on_batch_mode_clicked("outliers"))

        self.btn_batch_mode_boxless = self._make_btn("Missing Face Boxes", "tag", grp_health, large=False, checkable=True)
        self.btn_batch_mode_boxless.setToolTip("Review photos with person tags missing face bounding boxes")
        self.btn_batch_mode_boxless.clicked.connect(lambda: self._on_batch_mode_clicked("boxless"))

        v_modes.addWidget(self.btn_batch_mode_sugg)
        v_modes.addWidget(self.btn_batch_mode_outliers)
        v_modes.addWidget(self.btn_batch_mode_boxless)
        grp_health.content_layout.addLayout(v_modes)

        self.btn_batch_options = self._make_btn("Face\nOptions...", "options", grp_health, large=True)
        self.btn_batch_options.setToolTip("Configure face recognition models and rescan settings...")
        self.btn_batch_options.clicked.connect(lambda: self.action_triggered.emit("batch_face_options"))
        grp_health.content_layout.addWidget(self.btn_batch_options)
        hbox.addWidget(grp_health)

        # ---------------------------------------------------------------------
        # Group 4: Close
        # ---------------------------------------------------------------------
        grp_close = _RibbonGroupBox("Close", page)
        self.btn_batch_close = self._make_btn("Close", "cross", grp_close, large=True)
        self.btn_batch_close.setToolTip("Return to Gallery view (Esc)")
        self.btn_batch_close.clicked.connect(lambda: self.action_triggered.emit("batch_close"))
        grp_close.content_layout.addWidget(self.btn_batch_close)
        hbox.addWidget(grp_close)

        hbox.addStretch(1)

        return page

    def _on_batch_mode_clicked(self, mode: str) -> None:
        if hasattr(self, "btn_batch_mode_sugg"):
            self.btn_batch_mode_sugg.setChecked(mode == "suggestions")
        if hasattr(self, "btn_batch_mode_outliers"):
            self.btn_batch_mode_outliers.setChecked(mode == "outliers")
        if hasattr(self, "btn_batch_mode_boxless"):
            self.btn_batch_mode_boxless.setChecked(mode == "boxless")
        self.update_batch_action_labels(mode)
        self.action_triggered.emit(f"batch_mode_{mode}")

    def update_batch_action_labels(
        self,
        mode: str = "suggestions",
        swap_suggested_name: str | None = None,
        person_name: str | None = None,
    ) -> None:
        """Dynamically update Batch People Tagging action button labels & visibility based on mode and selection."""
        if not hasattr(self, "btn_batch_confirm"):
            return
        if mode == "outliers":
            # Outlier mode: explicit actions
            self.btn_batch_confirm.setText("Keep\ntag")
            self.btn_batch_confirm.setToolTip(f"Keep face as {person_name or 'current person'} (mark verified)")
            self.btn_batch_reject.setText("Remove\ntag")
            self.btn_batch_reject.setToolTip("Remove person tag from this photo")
            if hasattr(self, "btn_batch_tag_as"):
                self.btn_batch_tag_as.setText("Tag as")
            if hasattr(self, "btn_batch_ignore"):
                self.btn_batch_ignore.setText("Not a\nface")
                self.btn_batch_ignore.setToolTip("Mark as not a face and delete face box")
        elif mode == "boxless":
            self.btn_batch_confirm.setText("Add\nbox")
            self.btn_batch_confirm.setToolTip("Add face bounding box for this person")
            self.btn_batch_reject.setText("Remove\ntag")
            self.btn_batch_reject.setToolTip("Remove person tag from this photo")
            if hasattr(self, "btn_batch_tag_as"):
                self.btn_batch_tag_as.setText("Tag as")
            if hasattr(self, "btn_batch_ignore"):
                self.btn_batch_ignore.setText("Don't show\nagain")
                self.btn_batch_ignore.setToolTip("Ignore")
        else:
            # Suggestions mode
            self.btn_batch_confirm.setText("Confirm")
            self.btn_batch_confirm.setToolTip("Confirm selected face tags (Enter)")
            self.btn_batch_reject.setText("Not this\nperson")
            self.btn_batch_reject.setToolTip("Reject suggestion / not this person (Del)")
            if hasattr(self, "btn_batch_tag_as"):
                self.btn_batch_tag_as.setText("Tag as")
            if hasattr(self, "btn_batch_ignore"):
                self.btn_batch_ignore.setText("Don't show\nagain")
                self.btn_batch_ignore.setToolTip("Ignore / don't suggest this face again (I)")

    def set_active_mode(self, mode: Literal["grid", "viewer", "map", "batch_people"]) -> None:
        """Switches Ribbon context when entering/exiting Single Media Viewer, World Map, or Batch People Tagging."""
        if mode == "batch_people":
            for i in range(5):
                self.tab_widget.tabBar().setTabVisible(i, False)
            if hasattr(self, "tab_batch_people"):
                self.tab_widget.tabBar().setTabVisible(5, True)
                self.tab_widget.setCurrentWidget(self.tab_batch_people)
            if hasattr(self, "scope_bar") and self.scope_bar is not None:
                self.scope_bar.setVisible(False)
            return

        if hasattr(self, "tab_batch_people"):
            idx = self.tab_widget.indexOf(self.tab_batch_people)
            if idx >= 0:
                self.tab_widget.tabBar().setTabVisible(idx, False)
            if self.tab_widget.currentWidget() == self.tab_batch_people:
                self.tab_widget.setCurrentWidget(self.tab_home)

        if mode == "viewer":
            # In Single Image View, hide Home (0) and Find (2), showing File, Edit, Create, View
            self.tab_widget.tabBar().setTabVisible(0, False)
            self.tab_widget.tabBar().setTabVisible(1, True)
            self.tab_widget.tabBar().setTabVisible(2, False)
            self.tab_widget.tabBar().setTabVisible(3, True)
            self.tab_widget.tabBar().setTabVisible(4, True)
            self._set_edit_tab_mode("viewer")
            self.tab_widget.setCurrentWidget(self.tab_edit)
            if hasattr(self, "scope_bar") and self.scope_bar is not None:
                self.scope_bar.setVisible(False)
            if hasattr(self, "grp_arrange"):
                self.grp_arrange.setVisible(False)
            if hasattr(self, "grp_details"):
                self.grp_details.setVisible(False)
            if hasattr(self, "grp_layout"):
                self.grp_layout.setVisible(False)
            if hasattr(self, "btn_toggle_nav_pane"):
                self.btn_toggle_nav_pane.setVisible(False)
            if hasattr(self, "grp_panes"):
                self.grp_panes.adjust_to_content()
            if hasattr(self, "slider_zoom"):
                self.slider_zoom.setVisible(False)
            if hasattr(self, "btn_zoom_actual"):
                self.btn_zoom_actual.setVisible(True)
            if hasattr(self, "btn_zoom_fit"):
                self.btn_zoom_fit.setVisible(True)
            if hasattr(self, "grp_zoom"):
                self.grp_zoom.adjust_to_content()
        elif mode == "map":
            for i in range(5):
                self.tab_widget.tabBar().setTabVisible(i, True)
            self.tab_widget.setCurrentWidget(self.tab_view)
            if hasattr(self, "btn_view_map"):
                self.btn_view_map.setChecked(True)
            if hasattr(self, "btn_view_thumbnails"):
                self.btn_view_thumbnails.setChecked(False)
            if hasattr(self, "scope_bar") and self.scope_bar is not None:
                self.scope_bar.setVisible(False)
        else:
            # Grid mode: restore Home and Find tabs
            for i in range(5):
                self.tab_widget.tabBar().setTabVisible(i, True)
            if hasattr(self, "btn_view_map"):
                self.btn_view_map.setChecked(False)
            if hasattr(self, "btn_view_thumbnails"):
                self.btn_view_thumbnails.setChecked(True)
            self._set_edit_tab_mode("grid")
            if self.tab_widget.currentWidget() in (self.tab_edit, getattr(self, "tab_batch_people", None)):
                self.tab_widget.setCurrentWidget(self.tab_home)
            if hasattr(self, "scope_bar") and self.scope_bar is not None:
                self.scope_bar.setVisible(True)
            if hasattr(self, "grp_arrange"):
                self.grp_arrange.setVisible(True)
            if hasattr(self, "grp_details"):
                self.grp_details.setVisible(True)
            if hasattr(self, "grp_layout"):
                self.grp_layout.setVisible(True)
            if hasattr(self, "btn_toggle_nav_pane"):
                self.btn_toggle_nav_pane.setVisible(True)
            if hasattr(self, "grp_panes"):
                self.grp_panes.adjust_to_content()
            if hasattr(self, "slider_zoom"):
                self.slider_zoom.setVisible(False)
            if hasattr(self, "btn_zoom_actual"):
                self.btn_zoom_actual.setVisible(False)
            if hasattr(self, "btn_zoom_fit"):
                self.btn_zoom_fit.setVisible(False)
            if hasattr(self, "grp_zoom"):
                self.grp_zoom.adjust_to_content()


    def _apply_file_menu_stylesheet(self, is_dark: bool) -> None:
        menu = getattr(self, "file_menu", None) or (self.btn_file.menu() if hasattr(self, "btn_file") else None)
        if menu is None:
            return
        if is_dark:
            menu.setStyleSheet(
                """
                QMenu {
                    background-color: #1B1E24;
                    border: 1px solid #333A45;
                    border-radius: 2px;
                    padding: 4px 0px;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 9pt;
                    color: #E2E8F0;
                }
                QMenu::item {
                    padding: 6px 24px 6px 36px;
                    background-color: transparent;
                    border: 1px solid transparent;
                    border-radius: 2px;
                    margin: 1px 4px;
                    color: #E2E8F0;
                }
                QMenu::item:selected {
                    background-color: #2563EB;
                    border: 1px solid #3B82F6;
                    color: #FFFFFF;
                }
                QMenu::item:disabled {
                    color: #64748B;
                }
                QMenu::icon {
                    padding-left: 8px;
                }
                QMenu::separator {
                    height: 1px;
                    background: #2D323B;
                    margin: 4px 8px;
                }
                """
            )
        else:
            menu.setStyleSheet(
                """
                QMenu {
                    background-color: #FFFFFF;
                    border: 1px solid #7DA2CE;
                    border-radius: 2px;
                    padding: 4px 0px;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 9pt;
                    color: #1E395B;
                }
                QMenu::item {
                    padding: 6px 24px 6px 36px;
                    background-color: transparent;
                    border: 1px solid transparent;
                    border-radius: 2px;
                    margin: 1px 4px;
                    color: #1E395B;
                }
                QMenu::item:selected {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F0F6FD, stop:1 #D9E8F8);
                    border: 1px solid #A8CEF5;
                    color: #1E395B;
                }
                QMenu::item:disabled {
                    color: #A0B2C6;
                }
                QMenu::icon {
                    padding-left: 8px;
                }
                QMenu::separator {
                    height: 1px;
                    background: #E5EEF7;
                    margin: 4px 8px;
                }
                """
            )

    def _apply_aero_stylesheet(self) -> None:
        self._dark_mode = False
        self._is_dark = False
        self._current_theme_mode = "wlpg_aero"
        icon_check = (get_icons_dir() / "check_dark.svg").as_posix()
        arrow_drop_down = (get_icons_dir() / "arrow_dropdown_mini.svg").as_posix()
        self.setStyleSheet(
            """
            QTabWidget#RibbonTabs {
                border: none;
                background: transparent;
            }
            QTabWidget#RibbonTabs > QTabBar {
                qproperty-drawBase: 0;
                border: none;
                background: transparent;
            }
            QTabWidget#RibbonTabs::pane {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F4F8FD, stop:0.55 #EBF2FA, stop:1 #DDE8F5);
                border-top: 1px solid #9EB6CE;
                border-bottom: 1px solid #A8BDD4;
                border-left: none;
                border-right: none;
                top: -1px;
            }
            QTabWidget#RibbonTabs > QTabBar::tab {
                background: transparent;
                border: 1px solid transparent;
                border-bottom: none;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                padding: 3px 14px;
                margin-top: 2px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 9pt;
            }
            QTabWidget#RibbonTabs > QTabBar::tab:hover {
                background: rgba(255, 255, 255, 140);
                border-color: #B2C7DE;
            }
            QTabWidget#RibbonTabs > QTabBar::tab:selected {
                background: #F4F8FD;
                border: 1px solid #9EB6CE;
                border-bottom: 1px solid #F4F8FD;
                margin-bottom: -1px;
                font-weight: 600;
            }
            QPushButton#CornerOptionsBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 2px 8px;
                margin-right: 6px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
            }
            QPushButton#CornerOptionsBtn:hover {
                background: #E4F0FC;
                border-color: #9CB8D6;
            }
            QFrame#RibbonGroupBox {
                border: none;
                border-right: 1px solid #C5D5E6;
                background: transparent;
            }
            QFrame#RibbonGroupBox QLabel {
                color: #55687C;
                font-size: 8pt;
                font-family: 'Segoe UI';
                border: none;
                background: transparent;
                padding-bottom: 1px;
            }
            QFrame#FaceCarouselContainer,
            QFrame#RibbonArrangeBox {
                background: #FFFFFF;
                border: 1px solid #ABBFD3;
                border-radius: 2px;
            }
            QPushButton#LargeRibbonBtn,
            QToolButton#LargeRibbonBtn,
            QToolButton#LargeRibbonToolBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 0px 1px 2px 1px;
                margin: 0px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
                text-align: center;
            }
            QPushButton#CompactRibbonBtn,
            QToolButton#CompactRibbonBtn,
            QToolButton#CompactRibbonToolBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 2px;
                padding: 1px 4px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                text-align: left;
            }
            QPushButton#LargeRibbonBtn:hover,
            QToolButton#LargeRibbonBtn:hover,
            QToolButton#LargeRibbonToolBtn:hover,
            QPushButton#CompactRibbonBtn:hover,
            QToolButton#CompactRibbonBtn:hover,
            QToolButton#CompactRibbonToolBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFFDF5, stop:1 #FCE7A2);
                border: 1px solid #D9B152;
            }
            QPushButton#LargeRibbonBtn:pressed,
            QToolButton#LargeRibbonBtn:pressed,
            QPushButton#CompactRibbonBtn:pressed,
            QToolButton#CompactRibbonBtn:pressed,
            QToolButton#LargeRibbonToolBtn:pressed,
            QToolButton#CompactRibbonToolBtn:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FCE293, stop:1 #F7CE65);
                border: 1px solid #C28B10;
            }
            QPushButton#LargeRibbonBtn:checked,
            QToolButton#LargeRibbonBtn:checked,
            QToolButton#LargeRibbonToolBtn:checked,
            QPushButton#CompactRibbonBtn:checked,
            QToolButton#CompactRibbonBtn:checked,
            QToolButton#CompactRibbonToolBtn:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #EAF4FD, stop:1 #BBDDF7);
                border: 1px solid #7DA2CE;
            }
            QPushButton#LargeRibbonBtn:checked:hover,
            QToolButton#LargeRibbonBtn:checked:hover,
            QToolButton#LargeRibbonToolBtn:checked:hover,
            QPushButton#CompactRibbonBtn:checked:hover,
            QToolButton#CompactRibbonBtn:checked:hover,
            QToolButton#CompactRibbonToolBtn:checked:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F0F7FE, stop:1 #A8D3F5);
                border: 1px solid #6BA0D6;
            }
            QCheckBox {
                spacing: 4px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
            }
            QCheckBox::indicator {
                width: 12px;
                height: 12px;
                border: 1px solid #8E9CAE;
                border-radius: 1px;
                background: #FFFFFF;
            }
            QCheckBox::indicator:hover {
                border-color: #3399FF;
                background: #F0F7FE;
            }
            QCheckBox::indicator:checked {
                border-color: #2468A8;
                background: #E5F1FB;
                image: url({icon_check});
            }
            QToolButton#LargeRibbonToolBtn,
            QToolButton#LargeRibbonBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 0px 2px 1px 2px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
                text-align: center;
            }
            QToolButton#CompactRibbonBtn,
            QToolButton#CompactRibbonToolBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 2px;
                padding: 1px 4px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                text-align: left;
            }
            QToolButton#CompactRibbonToolBtn,
            QToolButton#CompactRibbonBtn[has_menu="true"] {
                padding: 1px 13px 1px 4px;
            }
            QToolButton#LargeRibbonToolBtn:hover,
            QToolButton#LargeRibbonBtn:hover,
            QToolButton#CompactRibbonToolBtn:hover,
            QToolButton#CompactRibbonBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFFDF5, stop:1 #FCE7A2);
                border: 1px solid #D9B152;
            }
            QToolButton::menu-indicator,
            QToolButton#CompactRibbonBtn::menu-indicator,
            QToolButton#CompactRibbonToolBtn::menu-indicator {
                image: url({arrow_drop_down});
                width: 7px;
                height: 4px;
                subcontrol-position: center right;
                subcontrol-origin: padding;
                right: 3px;
                top: 1px;
            }
            QToolButton#LargeRibbonBtn::menu-indicator,
            QToolButton#LargeRibbonToolBtn::menu-indicator {
                image: none;
                width: 0px;
                height: 0px;
            }
            QWidget#SlideshowSplitContainer,
            QWidget#TagDescSplitContainer,
            QWidget#SaveCopySplitContainer {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
            }
            QToolButton#SlideshowTopBtn,
            QToolButton#TagDescTopBtn,
            QToolButton#SaveCopyTopBtn {
                background: transparent;
                border: 1px solid transparent;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
                padding: 1px;
            }
            QToolButton#SlideshowTopBtn:hover,
            QToolButton#TagDescTopBtn:hover,
            QToolButton#SaveCopyTopBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFFDF5, stop:1 #FCE7A2);
                border: 1px solid #D9B152;
            }
            QToolButton#SlideshowTopBtn:pressed,
            QToolButton#TagDescTopBtn:pressed,
            QToolButton#SaveCopyTopBtn:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FCE293, stop:1 #F7CE65);
                border: 1px solid #C28B10;
            }
            QToolButton#SlideshowBottomBtn,
            QToolButton#TagDescBottomBtn,
            QToolButton#SaveCopyBottomBtn {
                background: transparent;
                border: 1px solid transparent;
                border-top-left-radius: 0px;
                border-top-right-radius: 0px;
                border-bottom-left-radius: 3px;
                border-bottom-right-radius: 3px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
                text-align: center;
                padding: 0px 1px 1px 1px;
            }
            QToolButton#SlideshowBottomBtn:hover,
            QToolButton#TagDescBottomBtn:hover,
            QToolButton#SaveCopyBottomBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFFDF5, stop:1 #FCE7A2);
                border: 1px solid #D9B152;
            }
            QToolButton#SlideshowBottomBtn:pressed,
            QToolButton#TagDescBottomBtn:pressed,
            QToolButton#SaveCopyBottomBtn:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FCE293, stop:1 #F7CE65);
                border: 1px solid #C28B10;
            }
            QToolButton#SlideshowBottomBtn::menu-indicator,
            QToolButton#TagDescBottomBtn::menu-indicator,
            QToolButton#SaveCopyBottomBtn::menu-indicator {
                image: none;
                width: 0px;
                height: 0px;
            }
            QToolButton#CompactRibbonSplitBtn {
                background: transparent;
                border: none;
                padding: 0px;
            }
            QToolButton#CompactRibbonSplitBtn::menu-button {
                border: none;
                background: transparent;
                width: 13px;
            }
            QToolButton#CompactRibbonSplitBtn::menu-arrow {
                image: none;
                width: 0px;
                height: 0px;
            }
            QToolButton::menu-button {
                border: none;
                border-left: 1px solid transparent;
                background: transparent;
                width: 14px;
            }
            QToolButton::menu-button:hover {
                border-left: 1px solid #C4D5E8;
                background: rgba(255, 255, 255, 0.45);
            }
            QToolButton::menu-arrow {
                image: url({arrow_drop_down});
                width: 7px;
                height: 4px;
            }
            QPushButton#RibbonModeToggleBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 2px;
                padding: 1px 4px;
                color: #33557A;
                font-family: 'Segoe UI';
                font-size: 8pt;
                text-align: left;
            }
            QPushButton#RibbonModeToggleBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFFDF5, stop:1 #FCE7A2);
                border: 1px solid #D9B152;
            }
            QPushButton#RibbonModeToggleBtn:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #EAF3FD, stop:0.5 #D2E7FA, stop:1 #BBDDF7);
                border: 1px solid #7DA2CE;
                font-weight: 600;
                color: #1E395B;
            }
            QFrame#SearchScopeBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F7FAFE, stop:1 #EAF1F9);
                border-bottom: 1px solid #C5D5E6;
            }
            QLineEdit#ScopeSearchInput {
                background: #FFFFFF;
                border: 1px solid #9CB4CC;
                border-radius: 2px;
                padding: 1px 6px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                color: #1E395B;
            }
            QLineEdit#ScopeSearchInput:focus {
                border: 1px solid #3380CC;
            }
            QWidget#RibbonTopHeader {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F4F8FD, stop:1 #E0EBF7);
                border-bottom: none;
            }
            QWidget#RibbonBelowQAT {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F4F8FC, stop:1 #E4ECF5);
                border-bottom: 1px solid #C4D3E3;
            }
            QToolButton#QATCircleBtn {
                border: none;
                border-radius: 11px;
                background: transparent;
                padding: 0px;
            }
            QToolButton#QATCircleBtn:hover {
                background: rgba(255, 255, 255, 160);
            }
            QToolButton#QATCircleBtn:pressed {
                background: rgba(0, 100, 200, 50);
            }
            QToolButton#QATBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 1px;
            }
            QToolButton#QATBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FDFEFE, stop:1 #E5F1FC);
                border: 1px solid #A8C8E8;
            }
            QToolButton#QATBtn:disabled {
                opacity: 0.35;
            }
            QPushButton#RibbonFileBtn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2A79C6, stop:1 #175B9E);
                color: #FFFFFF;
                font-family: 'Segoe UI';
                font-size: 9pt;
                font-weight: bold;
                border: 1px solid #104C85;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                padding: 3px 14px;
                margin-left: 2px;
                margin-right: 4px;
            }
            QPushButton#RibbonFileBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3B8BD8, stop:1 #1C67B0);
            }
            QPushButton#RibbonFileBtn::menu-indicator {
                image: none;
                width: 0px;
            }
            QWidget#RibbonTopHeader {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F0F6FC, stop:0.5 #E1EDF9, stop:1 #D2E4F6);
                border-bottom: none;
            }
            QLabel#RibbonAppTitle {
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                font-weight: 500;
                margin-left: 4px;
            }
            QFrame#RibbonHeaderDiv {
                background: #B6CEE8;
                border: none;
            }
            QPushButton#WinMinBtn, QPushButton#WinMaxBtn {
                border: none;
                background: transparent;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 9pt;
            }
            QPushButton#WinMinBtn:hover, QPushButton#WinMaxBtn:hover {
                background: #CFE3F7;
            }
            QPushButton#WinMinBtn:pressed, QPushButton#WinMaxBtn:pressed {
                background: #B8D6F2;
            }
            QPushButton#WinCloseBtn {
                border: none;
                background: transparent;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 9.5pt;
            }
            QPushButton#WinCloseBtn:hover {
                background: #E81123;
                color: #FFFFFF;
            }
            QPushButton#WinCloseBtn:pressed {
                background: #C42B1C;
                color: #FFFFFF;
            }
            QToolTip {
                background-color: #FFFFFF;
                color: #1E395B;
                border: 1px solid #7DA2CE;
                border-radius: 2px;
                padding: 3px 6px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
            }
            """.replace("{icon_check}", icon_check).replace("{arrow_drop_down}", arrow_drop_down)
        )
        for carousel in (getattr(self, "home_face_carousel", None), getattr(self, "find_face_carousel", None)):
            if carousel and hasattr(carousel, "set_dark_mode"):
                carousel.set_dark_mode(False)
        if hasattr(self, "effects_gallery") and hasattr(self.effects_gallery, "set_dark_mode"):
            self.effects_gallery.set_dark_mode(False)
        for btn in getattr(self, "_responsive_large_buttons", []):
            if hasattr(btn, "_is_dark"):
                btn._is_dark = False
                btn.update()
        for btn in getattr(self, "_responsive_secondary_buttons", []):
            if hasattr(btn, "_is_dark"):
                btn._is_dark = False
            if hasattr(btn, "_theme_mode"):
                btn._theme_mode = "modern"
            btn.update()
        for btn in getattr(self, "_responsive_secondary_buttons", []):
            if hasattr(btn, "_is_dark"):
                btn._is_dark = False
            if hasattr(btn, "_theme_mode"):
                btn._theme_mode = "aero"
            btn.update()
        self._apply_file_menu_stylesheet(False)

    def set_dark_mode(self, is_dark: bool) -> None:
        self._dark_mode = bool(is_dark)
        self._is_dark = self._dark_mode
        for btn in getattr(self, "_responsive_large_buttons", []):
            if hasattr(btn, "_is_dark"):
                btn._is_dark = self._dark_mode
                btn.update()
        for btn in getattr(self, "_responsive_secondary_buttons", []):
            if hasattr(btn, "_is_dark"):
                btn._is_dark = self._dark_mode
            if hasattr(btn, "_theme_mode"):
                btn._theme_mode = "dark" if self._dark_mode else "aero"
            btn.update()
        if not self._dark_mode:
            if getattr(self, "_current_theme_mode", "wlpg_aero") == "modern":
                self._apply_fluent_stylesheet()
            else:
                self._apply_aero_stylesheet()
            return

        icon_check = (get_icons_dir() / "check_white.svg").as_posix()
        arrow_drop_down = (get_icons_dir() / "arrow_dropdown_mini_dark.svg").as_posix()
        self.setStyleSheet(
            """
            QTabWidget#RibbonTabs {
                border: none;
                background: transparent;
            }
            QTabWidget#RibbonTabs > QTabBar {
                qproperty-drawBase: 0;
                border: none;
                background: transparent;
            }
            QTabWidget#RibbonTabs::pane {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #20242B, stop:0.55 #1B1E24, stop:1 #16191E);
                border-top: 1px solid #333A45;
                border-bottom: 1px solid #2D333D;
                border-left: none;
                border-right: none;
                top: -1px;
            }
            QTabWidget#RibbonTabs > QTabBar::tab {
                background: transparent;
                border: 1px solid transparent;
                border-bottom: none;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                padding: 3px 14px;
                margin-top: 2px;
                color: #D8E0EA;
                font-family: 'Segoe UI';
                font-size: 9pt;
            }
            QTabWidget#RibbonTabs > QTabBar::tab:hover {
                background: #282E38;
                border-color: #3E4756;
                color: #FFFFFF;
            }
            QTabWidget#RibbonTabs > QTabBar::tab:selected {
                background: #20242B;
                border: 1px solid #3E4756;
                border-bottom: 1px solid #20242B;
                margin-bottom: -1px;
                color: #60A5FA;
                font-weight: 600;
            }
            QPushButton#CornerOptionsBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 2px 8px;
                margin-right: 6px;
                color: #D8E0EA;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
            }
            QPushButton#CornerOptionsBtn:hover {
                background: #2A313C;
                border-color: #475569;
            }
            QFrame#RibbonGroupBox {
                border: none;
                border-right: 1px solid #2E3440;
                background: transparent;
            }
            QFrame#RibbonGroupBox QLabel {
                color: #94A3B8;
                font-size: 8pt;
                font-family: 'Segoe UI';
                border: none;
                background: transparent;
                padding-bottom: 1px;
            }
            QFrame#FaceCarouselContainer,
            QFrame#RibbonArrangeBox {
                background: #14171C;
                border: 1px solid #3B4252;
                border-radius: 2px;
            }
            QPushButton#LargeRibbonBtn,
            QToolButton#LargeRibbonBtn,
            QToolButton#LargeRibbonToolBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 0px 1px 2px 1px;
                margin: 0px;
                color: #E2E8F0;
                font-family: 'Segoe UI';
                font-size: 8pt;
                text-align: center;
            }
            QPushButton#CompactRibbonBtn,
            QToolButton#CompactRibbonBtn,
            QToolButton#CompactRibbonToolBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 2px;
                padding: 1px 4px;
                color: #E2E8F0;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                text-align: left;
            }
            QToolButton#CompactRibbonToolBtn,
            QToolButton#CompactRibbonBtn[has_menu="true"] {
                padding: 1px 13px 1px 4px;
            }
            QPushButton#LargeRibbonBtn:disabled,
            QToolButton#LargeRibbonBtn:disabled,
            QPushButton#CompactRibbonBtn:disabled,
            QToolButton#CompactRibbonBtn:disabled,
            QToolButton#LargeRibbonToolBtn:disabled,
            QToolButton#CompactRibbonToolBtn:disabled {
                color: #64748B;
            }
            QPushButton#LargeRibbonBtn:hover,
            QToolButton#LargeRibbonBtn:hover,
            QToolButton#LargeRibbonToolBtn:hover,
            QPushButton#CompactRibbonBtn:hover,
            QToolButton#CompactRibbonBtn:hover,
            QToolButton#CompactRibbonToolBtn:hover {
                background: #2A323D;
                border: 1px solid #475569;
                color: #FFFFFF;
            }
            QPushButton#LargeRibbonBtn:pressed,
            QToolButton#LargeRibbonBtn:pressed,
            QPushButton#CompactRibbonBtn:pressed,
            QToolButton#CompactRibbonBtn:pressed,
            QToolButton#LargeRibbonToolBtn:pressed,
            QToolButton#CompactRibbonToolBtn:pressed {
                background: #1E293B;
                border: 1px solid #3B82F6;
            }
            QPushButton#LargeRibbonBtn:checked,
            QToolButton#LargeRibbonBtn:checked,
            QPushButton#CompactRibbonBtn:checked,
            QToolButton#CompactRibbonBtn:checked,
            QToolButton#CompactRibbonToolBtn:checked {
                background: #1E3A5F;
                border: 1px solid #3B82F6;
                color: #FFFFFF;
            }
            QCheckBox {
                spacing: 4px;
                color: #E2E8F0;
                font-family: 'Segoe UI';
                font-size: 8pt;
            }
            QCheckBox::indicator {
                width: 12px;
                height: 12px;
                border: 1px solid #64748B;
                border-radius: 1px;
                background: #1E2228;
            }
            QCheckBox::indicator:hover {
                border-color: #60A5FA;
                background: #262B33;
            }
            QCheckBox::indicator:checked {
                border-color: #3B82F6;
                background: #2563EB;
                image: url({icon_check});
            }
            QToolButton::menu-indicator,
            QToolButton#CompactRibbonBtn::menu-indicator,
            QToolButton#CompactRibbonToolBtn::menu-indicator {
                image: url({arrow_drop_down});
                width: 7px;
                height: 4px;
                subcontrol-position: center right;
                subcontrol-origin: padding;
                right: 3px;
                top: 1px;
            }
            QToolButton#LargeRibbonBtn::menu-indicator,
            QToolButton#LargeRibbonToolBtn::menu-indicator {
                image: none;
                width: 0px;
                height: 0px;
            }
            QWidget#SlideshowSplitContainer,
            QWidget#TagDescSplitContainer,
            QWidget#SaveCopySplitContainer {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
            }
            QToolButton#SlideshowTopBtn,
            QToolButton#TagDescTopBtn,
            QToolButton#SaveCopyTopBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 1px;
            }
            QToolButton#SlideshowTopBtn:hover,
            QToolButton#SlideshowBottomBtn:hover,
            QToolButton#TagDescTopBtn:hover,
            QToolButton#TagDescBottomBtn:hover,
            QToolButton#SaveCopyTopBtn:hover,
            QToolButton#SaveCopyBottomBtn:hover {
                background: #2A323D;
                border: 1px solid #475569;
            }
            QToolButton#SlideshowBottomBtn,
            QToolButton#TagDescBottomBtn,
            QToolButton#SaveCopyBottomBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                color: #E2E8F0;
                font-family: 'Segoe UI';
                font-size: 8pt;
                text-align: center;
                padding: 0px 1px 1px 1px;
            }
            QToolButton#SlideshowBottomBtn::menu-indicator,
            QToolButton#TagDescBottomBtn::menu-indicator,
            QToolButton#SaveCopyBottomBtn::menu-indicator {
                image: none;
                width: 0px;
                height: 0px;
            }
            QToolButton#CompactRibbonSplitBtn {
                background: transparent;
                border: none;
                padding: 0px;
            }
            QToolButton#CompactRibbonSplitBtn::menu-button {
                border: none;
                background: transparent;
                width: 13px;
            }
            QToolButton#CompactRibbonSplitBtn::menu-arrow {
                image: none;
                width: 0px;
                height: 0px;
            }
            QToolButton::menu-button {
                border: none;
                border-left: 1px solid transparent;
                background: transparent;
                width: 14px;
            }
            QToolButton::menu-button:hover {
                border-left: 1px solid #333A45;
                background: rgba(255, 255, 255, 0.1);
            }
            QToolButton::menu-arrow {
                image: url({arrow_drop_down});
                width: 7px;
                height: 4px;
            }
            QPushButton#RibbonModeToggleBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 2px;
                padding: 1px 4px;
                color: #CBD5E1;
                font-family: 'Segoe UI';
                font-size: 8pt;
                text-align: left;
            }
            QPushButton#RibbonModeToggleBtn:hover {
                background: #2A323D;
                border: 1px solid #475569;
            }
            QPushButton#RibbonModeToggleBtn:checked {
                background: #1E3A5F;
                border: 1px solid #3B82F6;
                font-weight: 600;
                color: #FFFFFF;
            }
            QFrame#SearchScopeBar {
                background: #1A1D23;
                border-bottom: 1px solid #2E3440;
            }
            QLineEdit#ScopeSearchInput {
                background: #14161A;
                border: 1px solid #3E4756;
                border-radius: 2px;
                padding: 1px 6px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                color: #F1F5F9;
            }
            QLineEdit#ScopeSearchInput:focus {
                border: 1px solid #3B82F6;
            }
            QWidget#RibbonTopHeader {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1C2026, stop:1 #16191E);
                border-bottom: none;
            }
            QWidget#RibbonBelowQAT {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1C2026, stop:1 #16191E);
                border-bottom: 1px solid #2D333D;
            }
            QToolButton#QATCircleBtn {
                border: none;
                border-radius: 11px;
                background: transparent;
                padding: 0px;
            }
            QToolButton#QATCircleBtn:hover {
                background: rgba(255, 255, 255, 35);
            }
            QToolButton#QATBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 1px;
            }
            QToolButton#QATBtn:hover {
                background: #2A323D;
                border: 1px solid #475569;
            }
            QPushButton#RibbonFileBtn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2563EB, stop:1 #1D4ED8);
                color: #FFFFFF;
                font-family: 'Segoe UI';
                font-size: 9pt;
                font-weight: bold;
                border: 1px solid #1E40AF;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                padding: 3px 14px;
                margin-left: 2px;
                margin-right: 4px;
            }
            QPushButton#RibbonFileBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3B82F6, stop:1 #2563EB);
            }
            QPushButton#RibbonFileBtn::menu-indicator {
                image: none;
                width: 0px;
            }
            QWidget#RibbonTopHeader {
                background: #1B1E24;
                border-bottom: none;
            }
            QLabel#RibbonAppTitle {
                color: #94A3B8;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                font-weight: 500;
                margin-left: 4px;
            }
            QFrame#RibbonHeaderDiv {
                background: #333A45;
                border: none;
            }
            QPushButton#WinMinBtn, QPushButton#WinMaxBtn {
                border: none;
                background: transparent;
                color: #CBD5E1;
                font-family: 'Segoe UI';
                font-size: 9pt;
            }
            QPushButton#WinMinBtn:hover, QPushButton#WinMaxBtn:hover {
                background: #2E3642;
            }
            QPushButton#WinMinBtn:pressed, QPushButton#WinMaxBtn:pressed {
                background: #3B4452;
            }
            QPushButton#WinCloseBtn {
                border: none;
                background: transparent;
                color: #CBD5E1;
                font-family: 'Segoe UI';
                font-size: 9.5pt;
            }
            QPushButton#WinCloseBtn:hover {
                background: #E81123;
                color: #FFFFFF;
            }
            QPushButton#WinCloseBtn:pressed {
                background: #C42B1C;
                color: #FFFFFF;
            }
            QToolTip {
                background-color: #1E242C;
                color: #E2E8F0;
                border: 1px solid #333A45;
                border-radius: 2px;
                padding: 3px 6px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
            }
            """.replace("{icon_check}", icon_check).replace("{arrow_drop_down}", arrow_drop_down)
        )
        for carousel in (getattr(self, "home_face_carousel", None), getattr(self, "find_face_carousel", None)):
            if carousel and hasattr(carousel, "set_dark_mode"):
                carousel.set_dark_mode(True)
        if hasattr(self, "effects_gallery") and hasattr(self.effects_gallery, "set_dark_mode"):
            self.effects_gallery.set_dark_mode(True)
        self._apply_file_menu_stylesheet(True)

    def apply_theme_mode(self, theme_name: str) -> None:
        self._current_theme_mode = theme_name
        if theme_name == "dark":
            self.set_dark_mode(True)
        elif theme_name == "modern":
            self._apply_fluent_stylesheet()
        else:
            self.set_dark_mode(False)
        for btn in getattr(self, "_responsive_secondary_buttons", []):
            if hasattr(btn, "_theme_mode"):
                btn._theme_mode = theme_name
            btn.update()
        self.refresh_icons()
        if hasattr(self, "tab_widget") and self.tab_widget.tabBar():
            self.tab_widget.tabBar().update()

    def _apply_fluent_stylesheet(self) -> None:
        self._dark_mode = False
        self._is_dark = False
        self._current_theme_mode = "modern"
        arrow_drop_down = ARROW_DROPDOWN_PATH
        self.setStyleSheet(
            """
            QWidget#RibbonTopHeader {
                background: #F3F4F6;
                border-bottom: none;
            }
            QLabel#RibbonAppTitle {
                color: #374151;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                font-weight: 500;
                margin-left: 4px;
            }
            QFrame#RibbonHeaderDiv {
                background: #E5E7EB;
                border: none;
            }
            QPushButton#WinMinBtn, QPushButton#WinMaxBtn {
                border: none;
                background: transparent;
                color: #1F2937;
                font-family: 'Segoe UI';
                font-size: 9pt;
            }
            QPushButton#WinMinBtn:hover, QPushButton#WinMaxBtn:hover {
                background: #E5E7EB;
            }
            QPushButton#WinMinBtn:pressed, QPushButton#WinMaxBtn:pressed {
                background: #D1D5DB;
            }
            QPushButton#WinCloseBtn {
                border: none;
                background: transparent;
                color: #1F2937;
                font-family: 'Segoe UI';
                font-size: 9.5pt;
            }
            QPushButton#WinCloseBtn:hover {
                background: #E81123;
                color: #FFFFFF;
            }
            QPushButton#WinCloseBtn:pressed {
                background: #C42B1C;
                color: #FFFFFF;
            }
            QTabWidget#RibbonTabs {
                border: none;
                background: transparent;
            }
            QTabWidget#RibbonTabs > QTabBar {
                qproperty-drawBase: 0;
                border: none;
                background: transparent;
            }
            QTabWidget#RibbonTabs::pane {
                background: #FFFFFF;
                border-top: 1px solid #E5E7EB;
                border-bottom: 1px solid #E5E7EB;
                border-left: none;
                border-right: none;
                top: -1px;
            }
            QTabWidget#RibbonTabs > QTabBar::tab {
                background: transparent;
                border: none;
                padding: 4px 14px;
                margin-top: 2px;
                margin-bottom: 0px;
                color: #4B5563;
                font-family: 'Segoe UI';
                font-size: 9pt;
                font-weight: 500;
            }
            QTabWidget#RibbonTabs > QTabBar::tab:hover {
                background: #F3F4F6;
                border-radius: 4px;
                color: #111827;
            }
            QTabWidget#RibbonTabs > QTabBar::tab:selected {
                background: #FFFFFF;
                border: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
                color: #0F6CBD;
                font-weight: 600;
            }
            QTabWidget#RibbonTabs > QTabBar::tab:selected:hover {
                background: #FFFFFF;
                border: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }
            QPushButton#RibbonFileBtn {
                background: #0F6CBD;
                color: #FFFFFF;
                font-family: 'Segoe UI';
                font-size: 9pt;
                font-weight: 600;
                border: 1px solid #0F6CBD;
                border-radius: 4px;
                padding: 3px 14px;
                margin-left: 2px;
                margin-right: 4px;
                margin-top: 1px;
            }
            QPushButton#RibbonFileBtn:hover {
                background: #115EA3;
                border-color: #115EA3;
            }
            QPushButton#RibbonFileBtn::menu-indicator {
                image: none;
                width: 0px;
            }
            QFrame#RibbonGroupBox {
                border: none;
                border-right: 1px solid #E5E7EB;
                background: transparent;
            }
            QFrame#RibbonGroupBox QLabel {
                color: #6B7280;
                font-size: 8pt;
                font-family: 'Segoe UI';
                border: none;
                background: transparent;
                padding-bottom: 1px;
            }
            QFrame#FaceCarouselContainer,
            QFrame#RibbonArrangeBox {
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 4px;
            }
            QFrame#FaceCarouselContainer QToolButton {
                border: 1px solid #E5E7EB;
                border-radius: 3px;
                background: transparent;
                padding: 0px;
            }
            QFrame#FaceCarouselContainer QToolButton:hover {
                border: 1px solid #0F6CBD;
                background: #EFF6FF;
            }
            QPushButton#LargeRibbonBtn,
            QToolButton#LargeRibbonBtn,
            QToolButton#LargeRibbonToolBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 0px 1px 2px 1px;
                margin: 0px;
                color: #1F2937;
                font-family: 'Segoe UI';
                font-size: 8pt;
                text-align: center;
            }
            QPushButton#LargeRibbonBtn:hover,
            QToolButton#LargeRibbonBtn:hover,
            QToolButton#LargeRibbonToolBtn:hover {
                background: #E5E7EB;
                border: 1px solid #D1D5DB;
                color: #111827;
            }
            QPushButton#LargeRibbonBtn:pressed,
            QToolButton#LargeRibbonBtn:pressed,
            QToolButton#LargeRibbonToolBtn:pressed {
                background: #D1D5DB;
                border: 1px solid #9CA3AF;
            }
            QPushButton#LargeRibbonBtn:checked,
            QToolButton#LargeRibbonBtn:checked,
            QToolButton#LargeRibbonToolBtn:checked {
                background: #EFF6FF;
                border: 1px solid #BFDBFE;
                color: #0F6CBD;
            }
            QPushButton#LargeRibbonBtn:checked:hover,
            QToolButton#LargeRibbonBtn:checked:hover,
            QToolButton#LargeRibbonToolBtn:checked:hover {
                background: #DBEAFE;
                border: 1px solid #93C5FD;
                color: #0F6CBD;
            }
            QPushButton#LargeRibbonBtn:disabled,
            QToolButton#LargeRibbonBtn:disabled,
            QToolButton#LargeRibbonToolBtn:disabled {
                color: #9CA3AF;
            }
            QPushButton#CompactRibbonBtn,
            QToolButton#CompactRibbonBtn,
            QToolButton#CompactRibbonToolBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 1px 4px;
                color: #1F2937;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                text-align: left;
            }
            QPushButton#RibbonModeToggleBtn,
            QToolButton#RibbonModeToggleBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 0px 4px 1px 4px;
                color: #1F2937;
                font-family: 'Segoe UI';
                font-size: 8pt;
                text-align: left;
            }
            QPushButton#CompactRibbonBtn:hover,
            QToolButton#CompactRibbonBtn:hover,
            QToolButton#CompactRibbonToolBtn:hover,
            QPushButton#RibbonModeToggleBtn:hover,
            QToolButton#RibbonModeToggleBtn:hover {
                background: #E5E7EB;
                border: 1px solid #D1D5DB;
                color: #111827;
            }
            QPushButton#CompactRibbonBtn:pressed,
            QToolButton#CompactRibbonBtn:pressed,
            QToolButton#CompactRibbonToolBtn:pressed,
            QPushButton#RibbonModeToggleBtn:pressed,
            QToolButton#RibbonModeToggleBtn:pressed {
                background: #D1D5DB;
                border: 1px solid #9CA3AF;
            }
            QPushButton#CompactRibbonBtn:checked,
            QToolButton#CompactRibbonBtn:checked,
            QToolButton#CompactRibbonToolBtn:checked,
            QPushButton#RibbonModeToggleBtn:checked,
            QToolButton#RibbonModeToggleBtn:checked {
                background: #EFF6FF;
                border: 1px solid #BFDBFE;
                color: #0F6CBD;
                font-weight: 600;
            }
            QPushButton#CompactRibbonBtn:checked:hover,
            QToolButton#CompactRibbonBtn:checked:hover,
            QToolButton#CompactRibbonToolBtn:checked:hover,
            QPushButton#RibbonModeToggleBtn:checked:hover,
            QToolButton#RibbonModeToggleBtn:checked:hover {
                background: #DBEAFE;
                border: 1px solid #93C5FD;
                color: #0F6CBD;
            }
            QPushButton#CompactRibbonBtn:disabled,
            QToolButton#CompactRibbonBtn:disabled,
            QToolButton#CompactRibbonToolBtn:disabled,
            QPushButton#RibbonModeToggleBtn:disabled,
            QToolButton#RibbonModeToggleBtn:disabled {
                color: #9CA3AF;
            }
            QToolButton#CompactRibbonBtn[has_menu="true"],
            QToolButton#CompactRibbonToolBtn[has_menu="true"] {
                padding: 1px 13px 1px 4px;
            }
            QToolButton#CompactRibbonBtn::menu-indicator,
            QToolButton#CompactRibbonToolBtn::menu-indicator {
                image: url({arrow_drop_down});
                width: 7px;
                height: 4px;
                subcontrol-position: center right;
                subcontrol-origin: padding;
                right: 3px;
                top: 1px;
            }
            QToolButton#LargeRibbonBtn::menu-indicator,
            QToolButton#LargeRibbonToolBtn::menu-indicator {
                image: url({arrow_drop_down});
                width: 7px;
                height: 4px;
                subcontrol-position: bottom center;
                subcontrol-origin: padding;
                bottom: 2px;
            }
            QWidget#TagDescSplitContainer,
            QWidget#SlideshowSplitContainer,
            QWidget#SaveCopySplitContainer {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
            }
            QToolButton#SlideshowTopBtn,
            QToolButton#TagDescTopBtn,
            QToolButton#SaveCopyTopBtn {
                background: transparent;
                border: 1px solid transparent;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
                padding: 1px;
            }
            QToolButton#SlideshowTopBtn:hover,
            QToolButton#TagDescTopBtn:hover,
            QToolButton#SaveCopyTopBtn:hover {
                background: #E5E7EB;
                border: 1px solid #D1D5DB;
            }
            QToolButton#SlideshowTopBtn:pressed,
            QToolButton#TagDescTopBtn:pressed,
            QToolButton#SaveCopyTopBtn:pressed {
                background: #D1D5DB;
                border: 1px solid #9CA3AF;
            }
            QToolButton#SlideshowBottomBtn,
            QToolButton#TagDescBottomBtn,
            QToolButton#SaveCopyBottomBtn {
                background: transparent;
                border: 1px solid transparent;
                border-top-left-radius: 0px;
                border-top-right-radius: 0px;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
                color: #1F2937;
                font-family: 'Segoe UI';
                font-size: 8pt;
                text-align: center;
                padding: 0px 1px 1px 1px;
            }
            QToolButton#SlideshowBottomBtn:hover,
            QToolButton#TagDescBottomBtn:hover,
            QToolButton#SaveCopyBottomBtn:hover {
                background: #E5E7EB;
                border: 1px solid #D1D5DB;
            }
            QToolButton#SlideshowBottomBtn:pressed,
            QToolButton#TagDescBottomBtn:pressed,
            QToolButton#SaveCopyBottomBtn:pressed {
                background: #D1D5DB;
                border: 1px solid #9CA3AF;
            }
            QToolButton#SlideshowBottomBtn::menu-indicator,
            QToolButton#TagDescBottomBtn::menu-indicator,
            QToolButton#SaveCopyBottomBtn::menu-indicator {
                image: none;
                width: 0px;
                height: 0px;
            }
            QToolButton#CompactRibbonSplitBtn {
                background: transparent;
                border: none;
                padding: 0px;
            }
            QToolButton#CompactRibbonSplitBtn::menu-button {
                border: none;
                background: transparent;
                width: 13px;
            }
            QToolButton#CompactRibbonSplitBtn::menu-arrow {
                image: none;
                width: 0px;
                height: 0px;
            }
            QToolButton::menu-button {
                border: none;
                border-left: 1px solid transparent;
                background: transparent;
                width: 14px;
            }
            QToolButton::menu-button:hover {
                border-left: 1px solid #D1D5DB;
                background: #E5E7EB;
            }
            QToolButton::menu-arrow {
                image: url({arrow_drop_down});
                width: 7px;
                height: 4px;
            }
            QToolButton#QATCircleBtn,
            QToolButton#QATBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 1px;
            }
            QToolButton#QATCircleBtn:hover,
            QToolButton#QATBtn:hover {
                background: #E5E7EB;
                border: 1px solid #D1D5DB;
            }
            QToolButton#QATCircleBtn:pressed,
            QToolButton#QATBtn:pressed {
                background: #D1D5DB;
            }
            QToolButton#QATBtn:disabled,
            QToolButton#QATCircleBtn:disabled {
                opacity: 0.35;
            }
            QFrame#SearchScopeBar,
            QWidget#RibbonBelowQAT {
                background: #F9FAFB;
                border-bottom: 1px solid #E5E7EB;
            }
            QToolTip {
                background-color: #FFFFFF;
                color: #1F2937;
                border: 1px solid #E5E7EB;
                border-radius: 4px;
                padding: 3px 6px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
            }
            """.replace("{arrow_drop_down}", arrow_drop_down)
        )
        for carousel in (getattr(self, "home_face_carousel", None), getattr(self, "find_face_carousel", None)):
            if carousel and hasattr(carousel, "set_dark_mode"):
                carousel.set_dark_mode(False)
        if hasattr(self, "effects_gallery") and hasattr(self.effects_gallery, "set_dark_mode"):
            self.effects_gallery.set_dark_mode(False)
        for btn in getattr(self, "_responsive_large_buttons", []):
            if hasattr(btn, "_is_dark"):
                btn._is_dark = False
                btn.update()
        for btn in getattr(self, "_responsive_secondary_buttons", []):
            if hasattr(btn, "_is_dark"):
                btn._is_dark = False
            if hasattr(btn, "_theme_mode"):
                btn._theme_mode = "modern"
            btn.update()
        self._apply_file_menu_stylesheet(False)


# Backwards compatibility alias
PhotoGalleryRibbonToolbar = RibbonToolbar

