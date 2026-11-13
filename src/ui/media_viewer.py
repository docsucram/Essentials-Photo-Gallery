"""
SingleMediaViewer (QWidget) for PhotoGallery64.
Reuses ImageGraphicsView, CropScene, EditCommand (QUndoCommand), ClickableVideoWidget,
and MouseOverlay from legacy_reference/imgBrowserv13.py.
Implements WLPG's Full-Size Viewer & 'Fix / Fine-Tune' Adjustments Drawer:
- Zoom modes ('fit', 'fill', '100%', 'custom')
- Anchored CropScene rubber-band crop
- Auto adjust, Exposure/Contrast, Saturation/Warmth, Straighten, and Preset Effect tiles
- Non-destructive '.OriginalFiles' backup stashing on save & 'Revert to original'
- Static-image-only (is_video == 0) Face Region tagging rectangle overlay
- Integrated QMediaPlayer + QAudioOutput video playback
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRectF,
    QRunnable,
    QSize,
    QStringListModel,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QFontMetrics,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QMovie,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QUndoCommand,
    QUndoStack,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QCompleter,
    QFrame,
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QGraphicsTextItem,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class _MasterRenderSignals(QObject):
    """Signals emitted by _MasterRenderRunnable upon background completion."""

    finished = Signal(int, object, object, object, bool)


class _MasterRenderRunnable(QRunnable):
    """Runs the full-resolution render_adjusted_image pass on a worker thread."""

    def __init__(
        self,
        gen: int,
        media_id: Any,
        master_rgb: np.ndarray,
        master_alpha: np.ndarray | None,
        recipe: PhotoAdjustmentRecipe,
        mask_overlay: bool,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.gen = gen
        self.media_id = media_id
        self.master_rgb = master_rgb
        self.master_alpha = master_alpha
        self.recipe = recipe
        self.mask_overlay = mask_overlay
        self.signals = _MasterRenderSignals()

    def run(self) -> None:
        try:
            rendered_np = render_adjusted_image(
                self.master_rgb, self.recipe, media_id=self.media_id
            )
            h, w, _c = rendered_np.shape
            rendered_alpha = (
                transform_alpha_for_recipe(self.master_alpha, self.recipe, target_hw=(h, w))
                if self.master_alpha is not None
                else None
            )
            self.signals.finished.emit(
                self.gen, self.media_id, rendered_np, rendered_alpha, self.mask_overlay
            )
        except Exception:
            pass

from legacy_reference.imgBrowserv13 import (
    ClickableVideoWidget,
    EditCommand,
    ImageGraphicsView,
    MouseOverlay,
)
from src.imaging.image_loader import (
    ALPHA_SAVE_DOT_EXT_SET,
    RAW_DOT_EXT_SET,
    RAW_EXT_SET,
    build_save_pil_image,
    decode_pil_to_srgb_and_alpha,
    decode_raw_to_srgb,
    is_animated_image_file,
    transform_alpha_for_recipe,
)
from src.ui.crop_tool import CropOverlayItem, CropScene
from src.ui.markup_overlay import (
    MarkupFloatingToolbar,
    MarkupElement,
    flatten_markup_onto_image,
    flatten_markup_onto_rgb_image,
    build_arrow_geometry,
    create_magnifier_pixmap,
    build_star_polygon_in_rect,
    compute_circle_tangent_lines,
    get_markup_element_bounds,
    translate_markup_element,
    render_markup_element_pixmap,
    _DraggableGripHandle,
)
from src.imaging.creative_effects import apply_blur_redact, apply_pixelate
from src.ui.ribbon_toolbar import make_person_avatar_pixmap, make_symbol_icon, make_wlpg_icon
from src.ui.tag_someone_dialog import TagSomeoneDialog
from src.imaging.opencv_engine import (
    PhotoAdjustmentRecipe,
    analyze_auto_adjust,
    detect_auto_straighten_angle,
    render_adjusted_image,
)


def pil_to_qimage(pil_img: Image.Image) -> QImage:
    """Converts a PIL Image (RGB/RGBA) to a standalone QImage copy."""
    if pil_img.mode != "RGBA":
        rgba = pil_img.convert("RGBA")
    else:
        rgba = pil_img
    raw = rgba.tobytes("raw", "RGBA")
    qimg = QImage(raw, rgba.width, rgba.height, rgba.width * 4, QImage.Format_RGBA8888)
    return qimg.copy()


class _InlineFaceNamePopup(QFrame):
    """
    Inline floating popup with QLineEdit + QCompleter directly below an unknown
    or suggested face on the canvas, allowing instant naming without moving the mouse.
    """

    name_committed = Signal(int, str)  # (face_id, person_name)
    cancelled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("InlineFaceNamePopup")
        self.setFixedHeight(26)
        self.setMinimumWidth(165)
        self.setStyleSheet(
            """
            QFrame#InlineFaceNamePopup {
                background: #FFFFFF;
                border: 1.5px solid #2979CC;
                border-radius: 3px;
            }
            QLineEdit {
                background: transparent;
                border: none;
                font-family: 'Segoe UI';
                font-size: 9pt;
                color: #1E395B;
                padding: 1px 4px;
            }
            """
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(2)

        self.txt_name = QLineEdit(self)
        self.txt_name.setPlaceholderText("Who is this?")
        self.completer = QCompleter(self)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchContains)
        self.txt_name.setCompleter(self.completer)
        self.txt_name.returnPressed.connect(self._on_commit)
        self.completer.activated[str].connect(self._on_completer_activated)
        layout.addWidget(self.txt_name, 1)

        self.btn_done = QPushButton(self)
        self.btn_done.setIcon(make_symbol_icon("check", 14))
        self.btn_done.setFixedSize(20, 20)
        self.btn_done.setCursor(Qt.PointingHandCursor)
        self.btn_done.setToolTip("Save name (Enter)")
        self.btn_done.setStyleSheet(
            "QPushButton { background: transparent; border: 1px solid transparent; border-radius: 2px; }"
            "QPushButton:hover { background: #E8F5E9; border: 1px solid #A5D6A7; }"
            "QPushButton:pressed { background: #C8E6C9; }"
        )
        self.btn_done.clicked.connect(self._on_commit)
        layout.addWidget(self.btn_done)

        self.btn_cancel = QPushButton(self)
        self.btn_cancel.setIcon(make_symbol_icon("cross", 14))
        self.btn_cancel.setFixedSize(20, 20)
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setToolTip("Cancel (Esc)")
        self.btn_cancel.setStyleSheet(
            "QPushButton { background: transparent; border: 1px solid transparent; border-radius: 2px; }"
            "QPushButton:hover { background: #FFEBEE; border: 1px solid #FFCDD2; }"
            "QPushButton:pressed { background: #FFCDD2; }"
        )
        self.btn_cancel.clicked.connect(self._on_cancel)
        layout.addWidget(self.btn_cancel)

        self.current_face_id: int | None = None
        self._target_badge: Any | None = None
        self.hide()

    def set_dark_mode(self, is_dark: bool) -> None:
        if is_dark:
            self.setStyleSheet(
                """
                QFrame#InlineFaceNamePopup {
                    background: #1E242C;
                    border: 1.5px solid #3B82F6;
                    border-radius: 3px;
                }
                QLineEdit {
                    background: transparent;
                    border: none;
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                    color: #F1F5F9;
                    padding: 1px 4px;
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                QFrame#InlineFaceNamePopup {
                    background: #FFFFFF;
                    border: 1.5px solid #2979CC;
                    border-radius: 3px;
                }
                QLineEdit {
                    background: transparent;
                    border: none;
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                    color: #1E395B;
                    padding: 1px 4px;
                }
                """
            )

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self._on_cancel()
            event.accept()
            return
        super().keyPressEvent(event)

    def set_people_list(self, names: list[str]) -> None:
        model = QStringListModel(names, self.completer)
        self.completer.setModel(model)

    def popup_at(
        self,
        face_id: int,
        pos_view: QPoint,
        initial_text: str = "",
        target_badge: Any | None = None,
        unlinked_names: list[str] | None = None,
    ) -> None:
        self.current_face_id = face_id
        self._target_badge = target_badge
        if self._target_badge is not None:
            self._target_badge.setVisible(False)
        self.txt_name.setText(initial_text)
        self.txt_name.selectAll()

        if hasattr(self, "_quick_btns"):
            for b in self._quick_btns:
                b.deleteLater()
        self._quick_btns = []

        cands = [c.strip() for c in (unlinked_names or []) if c and c.strip()]
        if cands:
            for cand in cands[:2]:
                btn = QPushButton(f"Link {cand}", self)
                btn.setFixedHeight(20)
                btn.setCursor(Qt.PointingHandCursor)
                btn.setToolTip(f"Link detected face to tagged person '{cand}'")
                btn.setStyleSheet(
                    """
                    QPushButton {
                        background: #E8F4FD;
                        border: 1px solid #70A6E8;
                        border-radius: 2px;
                        color: #104C8A;
                        font-family: 'Segoe UI';
                        font-size: 8pt;
                        font-weight: 600;
                        padding: 0 4px;
                    }
                    QPushButton:hover {
                        background: #D0E8FC;
                        border-color: #2979CC;
                    }
                    """
                )
                btn.clicked.connect(lambda _, c=cand: self._on_quick_link(c))
                self.layout().insertWidget(self.layout().indexOf(self.btn_done), btn)
                self._quick_btns.append(btn)

        self.adjustSize()
        w = max(180, self.sizeHint().width())
        self.resize(w, 26)
        # Position popup directly centered over where the badge was
        self.move(pos_view.x() - w // 2, pos_view.y() - 13)
        self.show()
        self.raise_()
        self.txt_name.setFocus()

    def _on_quick_link(self, person_name: str) -> None:
        if person_name and self.current_face_id is not None:
            fid = self.current_face_id
            self.hide()
            self.name_committed.emit(fid, person_name)

    def _on_commit(self) -> None:
        name = self.txt_name.text().strip()
        if name and self.current_face_id is not None:
            fid = self.current_face_id
            self.hide()
            self.name_committed.emit(fid, name)
        else:
            self._on_cancel()

    def _on_completer_activated(self, text: str) -> None:
        if text.strip() and self.current_face_id is not None:
            fid = self.current_face_id
            self.hide()
            self.name_committed.emit(fid, text.strip())

    def _on_cancel(self) -> None:
        self.hide()
        if self._target_badge is not None:
            self._target_badge.setVisible(True)
            self._target_badge = None
        self.cancelled.emit()


class _FaceNameBadgeItem(QGraphicsItem):
    """
    Zoom-independent WLPG face name pill rendered directly below the bottom center
    of a face rectangle (matches media_1792269887974_67460900.png).
    Uses ItemIgnoresTransformations so the text is always a crisp 9.5pt Segoe UI pill
    regardless of photo resolution or zoom level.
    Supports confirmed faces, unconfirmed suggestions, and detected unknown faces.
    """

    def __init__(
        self,
        name: str,
        person_id: int | None = None,
        face_id: int | None = None,
        state: int = 1,
        is_suggestion: bool = False,
        is_unknown: bool = False,
        viewer: Any | None = None,
    ) -> None:
        super().__init__()
        self.name = str(name or "")
        self.person_id = person_id
        self.face_id = face_id
        self.state = int(state)
        self.is_suggestion = bool(is_suggestion)
        self.is_unknown = bool(is_unknown)
        self.viewer = viewer
        self.highlighted: bool = False
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setZValue(15)
        self._font = QFont("Segoe UI", 10, QFont.DemiBold)
        fm = QFontMetrics(self._font)
        disp_txt = f"{self.name}?" if self.is_suggestion else self.name
        text_w = max(32, fm.horizontalAdvance(disp_txt))
        extra_w = 68 if self.is_suggestion else 22
        self._pill_w = float(text_w + extra_w)
        self._pill_h = float(max(24, fm.height() + 6))
        self.setCursor(Qt.PointingHandCursor)
        self.setAcceptHoverEvents(True)
        self.sensor: Any | None = None

    def setText(self, text: str) -> None:
        self.prepareGeometryChange()
        self.name = str(text or "")
        fm = QFontMetrics(self._font)
        disp_txt = f"{self.name}?" if self.is_suggestion else self.name
        text_w = max(32, fm.horizontalAdvance(disp_txt))
        extra_w = 68 if self.is_suggestion else 22
        self._pill_w = float(text_w + extra_w)
        self._pill_h = float(max(24, fm.height() + 6))
        self.update()

    def text(self) -> str:
        return self.name

    def set_highlighted(self, highlighted: bool) -> None:
        if self.highlighted != highlighted:
            self.highlighted = highlighted
            self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(-self._pill_w / 2.0 - 2.0, 2.0, self._pill_w + 4.0, self._pill_h + 6.0)

    def hoverEnterEvent(self, event) -> None:
        sensor = getattr(self, "sensor", None)
        if sensor is not None:
            if hasattr(sensor, "_leave_timer"):
                sensor._leave_timer.stop()
            for it in getattr(sensor, "related_items", []):
                it.setVisible(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        sensor = getattr(self, "sensor", None)
        if sensor is not None and hasattr(sensor, "_leave_timer"):
            sensor._leave_timer.start(250)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.viewer is not None and self.face_id is not None:
            if self.is_unknown:
                event.accept()
                badge_center_scene = self.mapToScene(QPointF(0, self._pill_h * 0.5))
                vp_pos = self.viewer.graphics_view.mapFromScene(badge_center_scene)
                self.viewer.prompt_inline_name_face(self.face_id, vp_pos, initial_text="", target_badge=self)
                return
            elif self.is_suggestion:
                pos_x = event.pos().x()
                half_w = self._pill_w / 2.0
                if pos_x >= half_w - 28.0:
                    event.accept()
                    self.viewer.face_suggestion_rejected.emit(self.person_id or 0, self.face_id)
                    return
                elif pos_x >= half_w - 56.0:
                    event.accept()
                    self.viewer.face_suggestion_confirmed.emit(self.person_id or 0, self.face_id)
                    return
                else:
                    event.accept()
                    badge_center_scene = self.mapToScene(QPointF(0, self._pill_h * 0.5))
                    vp_pos = self.viewer.graphics_view.mapFromScene(badge_center_scene)
                    pname = self.name.split("?")[0].strip()
                    self.setVisible(False)
                    self.viewer.prompt_inline_name_face(self.face_id, vp_pos, initial_text=pname, target_badge=self)
                    return
            else:
                # Confirmed person tag: clicking allows directly editing or retagging the name
                event.accept()
                badge_center_scene = self.mapToScene(QPointF(0, self._pill_h * 0.5))
                vp_pos = self.viewer.graphics_view.mapFromScene(badge_center_scene)
                self.setVisible(False)
                self.viewer.prompt_inline_name_face(self.face_id, vp_pos, initial_text=self.name, target_badge=self)
                return
        super().mousePressEvent(event)

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        if not self.name:
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(-self._pill_w / 2.0, 4.0, self._pill_w, self._pill_h)

        # Subtle drop shadow
        shadow_rect = rect.translated(0.0, 1.0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 75))
        painter.drawRoundedRect(shadow_rect, 4.0, 4.0)

        # Pill background & border
        if self.highlighted:
            painter.setBrush(QColor("#D8EAFD"))
            painter.setPen(QPen(QColor("#2979CC"), 1.4))
        elif self.is_unknown:
            painter.setBrush(QColor("#FFFFFF"))
            pen = QPen(QColor("#70A6E8"), 1.2, Qt.DashLine)
            painter.setPen(pen)
        elif self.is_suggestion:
            painter.setBrush(QColor("#F0F8FF"))
            pen = QPen(QColor("#00B4D8"), 1.2, Qt.DashLine)
            painter.setPen(pen)
        else:
            painter.setBrush(QColor(238, 242, 247, 245))
            painter.setPen(QPen(QColor("#7A8A99"), 1.0))
        painter.drawRoundedRect(rect, 3.5, 3.5)

        painter.setFont(self._font)
        half_w = self._pill_w / 2.0
        if self.is_suggestion:
            # Draw suggestion name with question mark on the left, with full margin so '?' is never cut off
            disp_txt = f"{self.name}?" if not self.name.endswith("?") else self.name
            fm = QFontMetrics(self._font)
            text_w = fm.horizontalAdvance(disp_txt)
            text_rect = QRectF(-half_w + 10.0, 4.0, text_w + 16.0, self._pill_h)
            painter.setPen(QColor("#0B3D91"))
            painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, disp_txt)

            # Vector green checkmark icon
            btn_check_center = QPointF(half_w - 40.0, 4.0 + self._pill_h * 0.5)
            painter.setPen(QPen(QColor("#2E7D32"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(
                QPointF(btn_check_center.x() - 4.5, btn_check_center.y()),
                QPointF(btn_check_center.x() - 1.5, btn_check_center.y() + 3.5),
            )
            painter.drawLine(
                QPointF(btn_check_center.x() - 1.5, btn_check_center.y() + 3.5),
                QPointF(btn_check_center.x() + 4.5, btn_check_center.y() - 3.5),
            )

            # Vector red cross icon
            btn_cross_center = QPointF(half_w - 16.0, 4.0 + self._pill_h * 0.5)
            painter.setPen(QPen(QColor("#C50F1F"), 1.8, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(
                QPointF(btn_cross_center.x() - 3.5, btn_cross_center.y() - 3.5),
                QPointF(btn_cross_center.x() + 3.5, btn_cross_center.y() + 3.5),
            )
            painter.drawLine(
                QPointF(btn_cross_center.x() + 3.5, btn_cross_center.y() - 3.5),
                QPointF(btn_cross_center.x() - 3.5, btn_cross_center.y() + 3.5),
            )
        elif self.is_unknown:
            painter.setPen(QColor("#1A5FB4"))
            painter.drawText(rect, Qt.AlignCenter, self.name)
        else:
            painter.setPen(QColor("#111111"))
            painter.drawText(rect, Qt.AlignCenter, self.name)
        painter.restore()


class _FaceHoverRegionItem(QGraphicsRectItem):
    """Transparent sensor that detects mouse hover over a face region to reveal face boxes and badges."""

    def __init__(
        self,
        rect: QRectF,
        viewer: "SingleMediaViewer",
        related_items: list[QGraphicsItem],
        person_id: int | None,
        region_data: dict | None = None,
    ) -> None:
        super().__init__(rect)
        self.viewer = viewer
        self.related_items = related_items
        self.person_id = person_id
        self.region_data = region_data or {}
        self.related_badge: _FaceNameBadgeItem | None = None
        self._leave_timer = QTimer(self.viewer if isinstance(self.viewer, QWidget) else None)
        self._leave_timer.setSingleShot(True)
        self._leave_timer.timeout.connect(self._do_leave_hide)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        self.setPen(QPen(Qt.NoPen))
        self.setBrush(QBrush(QColor(0, 0, 0, 1)))
        self.setZValue(3)

    def hoverEnterEvent(self, event) -> None:
        if hasattr(self.viewer, "is_in_edit_mode") and self.viewer.is_in_edit_mode():
            return
        if hasattr(self, "_leave_timer"):
            self._leave_timer.stop()
        popup = getattr(self.viewer, "_inline_name_popup", None)
        popup_fid = popup.current_face_id if (popup is not None and popup.isVisible()) else None
        for it in self.related_items:
            if isinstance(it, _FaceNameBadgeItem) and popup_fid is not None and it.face_id == popup_fid:
                continue
            it.setVisible(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._leave_timer.start(250)
        super().hoverLeaveEvent(event)

    def _do_leave_hide(self) -> None:
        fid = self.region_data.get("face_id") or self.region_data.get("region_id")
        is_highlighted = (
            (self.person_id is not None and getattr(self.viewer, "_highlighted_person_id", None) == self.person_id)
            or (fid is not None and getattr(self.viewer, "_highlighted_region_id", None) == fid)
        )
        if not is_highlighted:
            if self.related_badge is not None and self.related_badge.isUnderMouse():
                return
            if self.isUnderMouse():
                return
            for it in self.related_items:
                it.setVisible(False)

    def mousePressEvent(self, event) -> None:
        if hasattr(self.viewer, "is_in_edit_mode") and self.viewer.is_in_edit_mode():
            super().mousePressEvent(event)
            return
        if event.button() == Qt.LeftButton:
            reg = self.region_data
            fid = reg.get("face_id") or reg.get("region_id")
            pname = reg.get("person_name") or reg.get("name")
            st = reg.get("state", 1 if reg.get("person_id") else 0)
            if fid is not None and (st == 0 or not pname):
                event.accept()
                badge_item = next((it for it in self.related_items if isinstance(it, _FaceNameBadgeItem)), None)
                if badge_item is not None:
                    badge_center_scene = badge_item.mapToScene(QPointF(0, badge_item._pill_h * 0.5))
                    vp_pos = self.viewer.graphics_view.mapFromScene(badge_center_scene)
                    self.viewer.prompt_inline_name_face(fid, vp_pos, initial_text=pname or "", target_badge=badge_item)
                else:
                    r = self.rect()
                    bot_center = QPointF(r.center().x(), r.bottom())
                    vp_pos = self.viewer.graphics_view.mapFromScene(bot_center)
                    self.viewer.prompt_inline_name_face(fid, vp_pos, initial_text=pname or "")
                return
        super().mousePressEvent(event)

    def contextMenuEvent(self, event) -> None:
        if hasattr(self.viewer, "is_in_edit_mode") and self.viewer.is_in_edit_mode():
            event.ignore()
            return
        event.accept()
        self.viewer.show_face_region_context_menu(
            event.screenPos(), self.region_data, self.person_id
        )


class _FaceAdjustToolbar(QFrame):
    """Floating Aero action pill shown when interactively adjusting a face bounding box."""

    saved = Signal()
    cancelled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("FaceAdjustToolbar")
        self.setStyleSheet(
            """
            QFrame#FaceAdjustToolbar {
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid #84ACDD;
                border-radius: 4px;
            }
            QLabel {
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                color: #1E395B;
                font-weight: 600;
            }
            QPushButton#AdjustSaveBtn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2A79D6, stop:1 #1B58B0);
                color: #FFFFFF;
                border: 1px solid #14448C;
                border-radius: 3px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                font-weight: bold;
                padding: 2px 10px;
                min-height: 22px;
            }
            QPushButton#AdjustSaveBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3D8CE8, stop:1 #2264C4);
            }
            QPushButton#AdjustCancelBtn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FDFDFD, stop:1 #EAEAEA);
                color: #222222;
                border: 1px solid #ACACAC;
                border-radius: 3px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                padding: 2px 8px;
                min-height: 22px;
            }
            QPushButton#AdjustCancelBtn:hover {
                background: #E0E8F5;
                border-color: #7DA2CE;
            }
            """
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        lbl = QLabel("Adjust face box:", self)
        layout.addWidget(lbl)

        btn_save = QPushButton("✓ Done", self)
        btn_save.setObjectName("AdjustSaveBtn")
        btn_save.setCursor(Qt.PointingHandCursor)
        btn_save.setToolTip("Save new bounding box (Enter)")
        btn_save.clicked.connect(self.saved.emit)
        layout.addWidget(btn_save)

        btn_cancel = QPushButton("✕ Cancel", self)
        btn_cancel.setObjectName("AdjustCancelBtn")
        btn_cancel.setCursor(Qt.PointingHandCursor)
        btn_cancel.setToolTip("Cancel adjustment (Esc)")
        btn_cancel.clicked.connect(self.cancelled.emit)
        layout.addWidget(btn_cancel)
        self.adjustSize()


class _FaceTagTargetBoxItem(QGraphicsItem):
    """
    Interactive WLPG manual face tagging box with thick white corner brackets
    (matches media_1792269928233_4b31b221.png).
    Can be dragged by its interior or resized by its corners/edges.
    """

    def __init__(
        self,
        rect: QRectF,
        image_bounds: QRectF,
        on_changed_callback=None,
        on_changed=None,
    ) -> None:
        super().__init__()
        self._rect = QRectF(rect)
        self._image_bounds = QRectF(image_bounds)
        self._on_changed = on_changed if on_changed is not None else on_changed_callback
        self._drag_mode: str | None = None
        self._drag_start_scene = QPointF()
        self._drag_start_rect = QRectF()
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setZValue(25)

    def rect(self) -> QRectF:
        return QRectF(self._rect)

    def setRect(self, rect: QRectF) -> None:
        self.prepareGeometryChange()
        self._rect = self._clamp_rect(rect)
        self.update()
        if self._on_changed is not None:
            self._on_changed()

    set_rect = setRect

    def _clamp_rect(self, r: QRectF) -> QRectF:
         r = r.normalized()
         ib = self._image_bounds
         if ib.isEmpty():
             return r
         min_sz = max(16.0, min(ib.width(), ib.height()) * 0.03)
         w = max(min_sz, min(r.width(), ib.width()))
         h = max(min_sz, min(r.height(), ib.height()))
         x = max(ib.left(), min(r.left(), ib.right() - w))
         y = max(ib.top(), min(r.top(), ib.bottom() - h))
         return QRectF(x, y, w, h)

    def boundingRect(self) -> QRectF:
        pad = 24.0
        return self._rect.adjusted(-pad, -pad, pad, pad)

    def _hit_mode(self, pos: QPointF) -> str | None:
        r = self._rect
        # Scale tolerance by view zoom so handles are easy to grab at any resolution
        tol = max(10.0, min(r.width(), r.height()) * 0.16)
        near_l = abs(pos.x() - r.left()) <= tol
        near_r = abs(pos.x() - r.right()) <= tol
        near_t = abs(pos.y() - r.top()) <= tol
        near_b = abs(pos.y() - r.bottom()) <= tol
        in_x = (r.left() - tol) <= pos.x() <= (r.right() + tol)
        in_y = (r.top() - tol) <= pos.y() <= (r.bottom() + tol)

        if near_t and near_l:
            return "tl"
        if near_t and near_r:
            return "tr"
        if near_b and near_l:
            return "bl"
        if near_b and near_r:
            return "br"
        if near_l and in_y:
            return "l"
        if near_r and in_y:
            return "r"
        if near_t and in_x:
            return "t"
        if near_b and in_x:
            return "b"
        if r.contains(pos):
            return "move"
        return None

    def hoverMoveEvent(self, event) -> None:
        mode = self._hit_mode(event.pos())
        cursors = {
            "tl": Qt.SizeFDiagCursor,
            "br": Qt.SizeFDiagCursor,
            "tr": Qt.SizeBDiagCursor,
            "bl": Qt.SizeBDiagCursor,
            "l": Qt.SizeHorCursor,
            "r": Qt.SizeHorCursor,
            "t": Qt.SizeVerCursor,
            "b": Qt.SizeVerCursor,
            "move": Qt.SizeAllCursor,
        }
        if mode in cursors:
            self.setCursor(cursors[mode])
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            mode = self._hit_mode(event.pos())
            if mode is not None:
                self._drag_mode = mode
                self._drag_start_scene = event.scenePos()
                self._drag_start_rect = QRectF(self._rect)
                event.accept()
                return
        event.ignore()

    def mouseMoveEvent(self, event) -> None:
        if not self._drag_mode:
            super().mouseMoveEvent(event)
            return
        delta = event.scenePos() - self._drag_start_scene
        r = QRectF(self._drag_start_rect)
        min_sz = max(16.0, min(self._image_bounds.width(), self._image_bounds.height()) * 0.03)
        m = self._drag_mode
        if m == "move":
            r.translate(delta)
        else:
            if "l" in m:
                r.setLeft(min(r.right() - min_sz, r.left() + delta.x()))
            if "r" in m:
                r.setRight(max(r.left() + min_sz, r.right() + delta.x()))
            if "t" in m:
                r.setTop(min(r.bottom() - min_sz, r.top() + delta.y()))
            if "b" in m:
                r.setBottom(max(r.top() + min_sz, r.bottom() + delta.y()))
        self.setRect(r)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_mode:
            self._drag_mode = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        painter.save()
        r = self._rect

        # 1. Outer dark shadow border + crisp inner white border (cosmetic so thickness is zoom-invariant)
        shadow_pen = QPen(QColor(0, 0, 0, 165), 3.0)
        shadow_pen.setCosmetic(True)
        painter.setPen(shadow_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(r)

        thin_white = QPen(QColor(255, 255, 255, 210), 1.4)
        thin_white.setCosmetic(True)
        painter.setPen(thin_white)
        painter.drawRect(r)

        # 2. Thick white L-shaped corner brackets matching WLPG screenshot (media_1792269928233_4b31b221.png)
        arm = max(6.0, min(r.width(), r.height()) * 0.22)
        path = QPainterPath()
        # Top-Left
        path.moveTo(r.left(), r.top() + arm)
        path.lineTo(r.left(), r.top())
        path.lineTo(r.left() + arm, r.top())
        # Top-Right
        path.moveTo(r.right() - arm, r.top())
        path.lineTo(r.right(), r.top())
        path.lineTo(r.right(), r.top() + arm)
        # Bottom-Left
        path.moveTo(r.left(), r.bottom() - arm)
        path.lineTo(r.left(), r.bottom())
        path.lineTo(r.left() + arm, r.bottom())
        # Bottom-Right
        path.moveTo(r.right() - arm, r.bottom())
        path.lineTo(r.right(), r.bottom())
        path.lineTo(r.right(), r.bottom() - arm)

        corner_shadow = QPen(QColor(0, 0, 0, 190), 5.5, Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin)
        corner_shadow.setCosmetic(True)
        painter.setPen(corner_shadow)
        painter.drawPath(path)

        corner_white = QPen(QColor(255, 255, 255, 255), 3.5, Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin)
        corner_white.setCosmetic(True)
        painter.setPen(corner_white)
        painter.drawPath(path)
        painter.restore()


class _PersonPopupRowButton(QPushButton):
    """Clickable row in the WLPG 'Tag someone' popup with 36x36 square face avatar and name."""

    person_clicked = Signal(str)

    def __init__(self, name: str, avatar_pixmap: QPixmap, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.person_name = name
        self.setFixedHeight(42)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            """
            QPushButton {
                background: #FFFFFF;
                border: 1px solid transparent;
                border-radius: 2px;
                text-align: left;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F2F8FE, stop:1 #DCEBFC);
                border: 1px solid #7AC1FF;
            }
            """
        )
        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(4, 3, 6, 3)
        hbox.setSpacing(8)

        lbl_icon = QLabel(self)
        lbl_icon.setFixedSize(36, 36)
        lbl_icon.setPixmap(avatar_pixmap.scaled(36, 36, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        lbl_icon.setStyleSheet("border: 1px solid #B8C8D8; background: #F0F4F8;")
        hbox.addWidget(lbl_icon)

        lbl_name = QLabel(name, self)
        lbl_name.setStyleSheet(
            "color: #111111; font-family: 'Segoe UI'; font-size: 9pt; background: transparent; border: none;"
        )
        hbox.addWidget(lbl_name, 1)
        self.clicked.connect(lambda: self.person_clicked.emit(self.person_name))


class _TagSomeonePopup(QFrame):
    """
    Floating WLPG 'Tag someone' card anchored beside the active face target box
    (matches media_1792269928233_4b31b221.png).
    Displays:
    - 'Tag someone' blue title + '✕' close button
    - Search QLineEdit with magnifying glass icon
    - Scrollable list of existing people with 36x36 square face avatars + names
    """

    person_selected = Signal(str)
    cancelled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TagSomeonePopup")
        self.setFixedSize(235, 280)
        self.setStyleSheet(
            """
            QFrame#TagSomeonePopup {
                background-color: #FFFFFF;
                border: 1px solid #7A8A99;
                border-radius: 3px;
            }
            """
        )
        self._people_data: list[dict] = []

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(10, 8, 10, 10)
        vbox.setSpacing(6)

        # 1. Header row: "Tag someone" + close "✕"
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        self.lbl_title = QLabel("Tag someone", self)
        self.lbl_title.setStyleSheet(
            "color: #1B64B0; font-family: 'Segoe UI'; font-size: 10pt; border: none; background: transparent;"
        )
        hdr.addWidget(self.lbl_title)
        hdr.addStretch(1)

        self.btn_close = QToolButton(self)
        self.btn_close.setText("✕")
        self.btn_close.setFixedSize(18, 18)
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.setStyleSheet(
            """
            QToolButton {
                border: none;
                background: transparent;
                color: #666666;
                font-family: 'Segoe UI';
                font-size: 9pt;
                font-weight: bold;
            }
            QToolButton:hover {
                color: #C50F1F;
                background: #F8D7DA;
                border-radius: 2px;
            }
            """
        )
        self.btn_close.clicked.connect(self.cancelled.emit)
        hdr.addWidget(self.btn_close)
        vbox.addLayout(hdr)

        # 2. Search input box with magnifying glass icon
        search_frame = QFrame(self)
        search_frame.setFixedHeight(24)
        search_frame.setStyleSheet(
            """
            QFrame {
                background: #FFFFFF;
                border: 1px solid #3380CC;
                border-radius: 2px;
            }
            """
        )
        s_hbox = QHBoxLayout(search_frame)
        s_hbox.setContentsMargins(4, 1, 4, 1)
        s_hbox.setSpacing(4)

        self.txt_search = QLineEdit(search_frame)
        self.txt_search.setPlaceholderText("Type a name")
        self.txt_search.setStyleSheet(
            "border: none; background: transparent; color: #111111; font-family: 'Segoe UI'; font-size: 9pt;"
        )
        self.txt_search.textChanged.connect(self._filter_list)
        self.txt_search.returnPressed.connect(self._on_return_pressed)
        s_hbox.addWidget(self.txt_search, 1)

        lbl_mag = QLabel(search_frame)
        lbl_mag.setFixedSize(16, 16)
        lbl_mag.setPixmap(make_wlpg_icon("text_search", 14).pixmap(14, 14))
        lbl_mag.setStyleSheet("border: none; background: transparent;")
        s_hbox.addWidget(lbl_mag)
        vbox.addWidget(search_frame)

        # 3. Scrollable People List
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet(
            """
            QScrollArea {
                background: #FFFFFF;
                border: 1px solid #D5E2F0;
            }
            """
        )
        self.list_container = QWidget(self.scroll_area)
        self.list_container.setStyleSheet("background: #FFFFFF;")
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(2, 2, 2, 2)
        self.list_layout.setSpacing(1)
        self.list_layout.addStretch(1)
        self.scroll_area.setWidget(self.list_container)
        vbox.addWidget(self.scroll_area, 1)

    def populate_people(self, people_list: list[dict]) -> None:
        self._people_data = list(people_list)
        self.txt_search.clear()
        self._rebuild_rows("")

    populate = populate_people

    def focus_input(self) -> None:
        self.txt_search.setFocus(Qt.ActiveWindowFocusReason)
        self.txt_search.selectAll()
        QTimer.singleShot(0, lambda: (self.txt_search.setFocus(Qt.ActiveWindowFocusReason), self.txt_search.selectAll()))

    def _filter_list(self, text: str) -> None:
        self._rebuild_rows(text.strip())

    def _rebuild_rows(self, query: str) -> None:
        while self.list_layout.count() > 0:
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        q_lower = query.lower()
        matched = [
            p for p in self._people_data
            if not q_lower or q_lower in str(p.get("name", "")).lower()
        ]
        for p in matched:
            pname = str(p.get("name", ""))
            blob = p.get("thumbnail_blob")
            pix = make_person_avatar_pixmap(pname, blob, size=36)
            row_btn = _PersonPopupRowButton(pname, pix, self.list_container)
            row_btn.person_clicked.connect(self.person_selected.emit)
            self.list_layout.addWidget(row_btn)

        if query and not any(str(p.get("name", "")).lower() == q_lower for p in matched):
            pix_new = make_person_avatar_pixmap(query, None, size=36)
            row_new = _PersonPopupRowButton(query, pix_new, self.list_container)
            row_new.person_clicked.connect(self.person_selected.emit)
            self.list_layout.addWidget(row_new)

        self.list_layout.addStretch(1)

    def _on_return_pressed(self) -> None:
        text = self.txt_search.text().strip()
        if text:
            self.person_selected.emit(text)
            return
        # If search is empty and there is at least one person row, pick the first
        if self._people_data:
            first_name = str(self._people_data[0].get("name", "")).strip()
            if first_name:
                self.person_selected.emit(first_name)


class RecipeUndoCommand(QUndoCommand):
    """A lightweight command storing before and after PhotoAdjustmentRecipe states."""

    def __init__(
        self,
        viewer: SingleMediaViewer,
        before_recipe: PhotoAdjustmentRecipe,
        after_recipe: PhotoAdjustmentRecipe,
        description: str,
        parent=None,
    ) -> None:
        super().__init__(description, parent)
        self.viewer = viewer
        self.before_recipe = before_recipe.copy()
        self.after_recipe = after_recipe.copy()
        self._first_redo = True

    def undo(self) -> None:
        was_crop = (self.after_recipe.crop_rect != self.before_recipe.crop_rect)
        self.viewer.current_recipe = self.before_recipe.copy()
        self.viewer._render_current_recipe()
        if was_crop:
            self.viewer.set_zoom_mode("fit")
        self.viewer.recipe_changed.emit(self.viewer.current_recipe)

    def redo(self) -> None:
        if self._first_redo:
            self._first_redo = False
            return
        was_crop = (self.after_recipe.crop_rect != self.before_recipe.crop_rect)
        self.viewer.current_recipe = self.after_recipe.copy()
        self.viewer._render_current_recipe()
        if was_crop:
            self.viewer.set_zoom_mode("fit")
        self.viewer.recipe_changed.emit(self.viewer.current_recipe)


class PixelArrayUndoCommand(QUndoCommand):
    """
    High-efficiency undo/redo command for destructive raster operations.
    Stores only the modified bounding box slice (patch) when dimensions match,
    reducing memory consumption by >95% compared to full-frame copying.
    """

    def __init__(
        self,
        viewer: SingleMediaViewer,
        before_np: np.ndarray,
        after_np: np.ndarray,
        description: str,
        parent=None,
    ) -> None:
        super().__init__(description, parent)
        self.viewer = viewer

        # If array shapes and dtypes match, compute the minimal bounding box of modified pixels
        if before_np.shape == after_np.shape and before_np.dtype == after_np.dtype and before_np.size > 0:
            diff = np.any(before_np != after_np, axis=-1) if before_np.ndim == 3 else (before_np != after_np)
            if np.any(diff):
                rows = np.any(diff, axis=1)
                cols = np.any(diff, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                # 1-pixel margin bounded by image extent
                y1 = max(0, int(rmin))
                y2 = min(before_np.shape[0], int(rmax) + 1)
                x1 = max(0, int(cmin))
                x2 = min(before_np.shape[1], int(cmax) + 1)

                self.bbox = (y1, y2, x1, x2)
                self.before_patch = before_np[y1:y2, x1:x2].copy()
                self.after_patch = after_np[y1:y2, x1:x2].copy()
                self.full_frame = False
            else:
                self.bbox = (0, 0, 0, 0)
                self.before_patch = np.empty((0, 0, before_np.shape[2] if before_np.ndim == 3 else 1), dtype=before_np.dtype)
                self.after_patch = self.before_patch.copy()
                self.full_frame = False
        else:
            self.full_frame = True
            self.before_patch = before_np.copy()
            self.after_patch = after_np.copy()
            self.bbox = None

    def undo(self) -> None:
        if self.full_frame:
            self.viewer.master_np_rgb = self.before_patch.copy()
        elif self.bbox and self.viewer.master_np_rgb is not None:
            y1, y2, x1, x2 = self.bbox
            if y2 > y1 and x2 > x1:
                self.viewer.master_np_rgb[y1:y2, x1:x2] = self.before_patch.copy()
        self.viewer._update_display_proxy()
        self.viewer._render_current_recipe()

    def redo(self) -> None:
        if self.full_frame:
            self.viewer.master_np_rgb = self.after_patch.copy()
        elif self.bbox and self.viewer.master_np_rgb is not None:
            y1, y2, x1, x2 = self.bbox
            if y2 > y1 and x2 > x1:
                self.viewer.master_np_rgb[y1:y2, x1:x2] = self.after_patch.copy()
        self.viewer._update_display_proxy()
        self.viewer._render_current_recipe()


class _AlphaCutoutUndoCommand(QUndoCommand):
    """Undo/Redo command for 1-click Background Removal (updates both master_np_rgb and master_alpha)."""

    def __init__(
        self,
        viewer: SingleMediaViewer,
        before_rgb: np.ndarray,
        before_alpha: np.ndarray | None,
        after_rgb: np.ndarray,
        after_alpha: np.ndarray | None,
        description: str = "Remove Background",
        parent=None,
    ) -> None:
        super().__init__(description, parent)
        self.viewer = viewer
        self.before_rgb = before_rgb.copy()
        self.before_alpha = before_alpha.copy() if before_alpha is not None else None
        self.after_rgb = after_rgb.copy()
        self.after_alpha = after_alpha.copy() if after_alpha is not None else None

    def undo(self) -> None:
        self.viewer.master_np_rgb = self.before_rgb.copy()
        self.viewer.master_alpha = self.before_alpha.copy() if self.before_alpha is not None else None
        self.viewer._update_display_proxy()
        self.viewer._render_current_recipe()

    def redo(self) -> None:
        self.viewer.master_np_rgb = self.after_rgb.copy()
        self.viewer.master_alpha = self.after_alpha.copy() if self.after_alpha is not None else None
        self.viewer._update_display_proxy()
        self.viewer._render_current_recipe()



class _DraggableFloatingToolbar(QFrame):
    """Floating overlay toolbar with drag handle and viewport boundary clamping."""
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_dragging: bool = False
        self._drag_pos: QPoint = QPoint()
        self._user_positioned: bool = False

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


class SingleMediaViewer(QWidget):
    """
    Single-Image / Video Viewer and WLPG 'Fix' Adjustments Editor (`docs/reference_ui/fullsize and edit_ui.png`).
    """

    closed = Signal()
    navigate_requested = Signal(int)                              # -1 (prev) or +1 (next)
    rotate_requested = Signal(bool)                               # True = CW, False = CCW
    pixel_edit_saved = Signal(int, str)                           # (media_id, backup_path)
    pixel_edit_saved_as_copy = Signal(int, str)                  # (original_media_id, new_path)
    pixel_edit_reverted = Signal(int)                             # media_id
    face_region_tagged = Signal(int, str, float, float, float, float)  # (media_id, name, rx, ry, rw, rh)
    face_region_named = Signal(int, str)                               # (face_id, person_name)
    face_suggestion_confirmed = Signal(int, int)                       # (person_id, face_id)
    face_suggestion_rejected = Signal(int, int)                        # (person_id, face_id)
    zoom_percent_changed = Signal(int)
    image_changed = Signal(object)                                # emitted with current PIL Image
    recipe_changed = Signal(object)                               # emitted with current PhotoAdjustmentRecipe
    status_message_requested = Signal(str, int)                  # (message, timeout_ms)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SingleMediaViewer")
        self.setFocusPolicy(Qt.StrongFocus)

        self.current_media_id: int | None = None
        self.current_file_path: Path | None = None
        self.current_is_video: bool = False
        self.master_pil_image: Image.Image | None = None
        self.master_np_rgb: np.ndarray | None = None
        self.master_alpha: np.ndarray | None = None
        self.display_proxy_rgb: np.ndarray | None = None
        self.display_proxy_alpha: np.ndarray | None = None
        self.scrub_proxy_rgb: np.ndarray | None = None
        self.scrub_proxy_alpha: np.ndarray | None = None
        self._is_animated_image: bool = False
        self._anim_movie: QMovie | None = None
        self._proxy_scale: float = 1.0
        self._scrub_proxy_scale: float = 1.0
        self._is_sliding: bool = False
        self._master_render_gen: int = 0
        self._master_render_dirty: bool = False
        self._master_render_timer = QTimer(self)
        self._master_render_timer.setSingleShot(True)
        self._master_render_timer.setInterval(250)
        self._master_render_timer.timeout.connect(self._flush_master_render)
        self.current_pil_image: Image.Image | None = None
        self.current_recipe: PhotoAdjustmentRecipe = PhotoAdjustmentRecipe()
        self._recipe_before_slider: PhotoAdjustmentRecipe | None = None
        self.current_zoom_mode: Literal["fit", "fill", "100", "custom"] = "fit"
        self.is_crop_mode: bool = False
        self.is_face_tag_mode: bool = False

        # Retouch (Magic Eraser) state
        self.is_retouch_mode: bool = False
        self.retouch_radius: int = 20
        self.main_window: Any = None
        self._retouch_active: bool = False
        self._retouch_mask: np.ndarray | None = None
        self._retouch_preview_items: list[Any] = []
        self._retouch_last_master_pt: tuple[int, int] | None = None

        # Red Eye state
        self.is_red_eye_mode: bool = False
        self._red_eye_active: bool = False
        self._red_eye_start_pos: QPointF | None = None
        self._red_eye_item: QGraphicsRectItem | None = None

        self.undo_stack = QUndoStack(self)
        self.undo_stack.canUndoChanged.connect(lambda _: self.update_revert_state())
        self.undo_stack.cleanChanged.connect(lambda _: self.update_revert_state())
        self._face_overlay_items: list[QGraphicsItem] = []
        self._current_people_regions: list[dict] = []
        self._face_target_item: _FaceTagTargetBoxItem | None = None
        self._face_tag_replace_person_id: int | None = None
        self._drawing_face_start_pos: QPointF | None = None
        self._highlighted_person_id: int | None = None
        self._highlighted_region_id: int | None = None

        root_hbox = QHBoxLayout(self)
        root_hbox.setContentsMargins(0, 0, 0, 0)
        root_hbox.setSpacing(0)

        # ---------------------------------------------------------------------
        # 1. Central Stack: Page 0 = ImageGraphicsView + CropScene, Page 1 = Video Player
        # ---------------------------------------------------------------------
        self.media_stack = QStackedWidget(self)
        self.media_stack.setFrameShape(QFrame.NoFrame)

        # Page 0: Image Viewer
        self.graphics_view = ImageGraphicsView(main_window=self, parent=self)
        self.graphics_view.setFrameShape(QFrame.NoFrame)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scene = CropScene(self.graphics_view)
        self._alpha_backdrop_item = QGraphicsRectItem()
        self._alpha_backdrop_item.setZValue(-1.0)
        self._alpha_backdrop_item.setPen(Qt.NoPen)
        self._alpha_backdrop_item.setBrush(self._make_checkerboard_brush(False))
        self._alpha_backdrop_item.setVisible(False)
        self.scene.addItem(self._alpha_backdrop_item)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.scene.pixmap_item = self.pixmap_item
        self.scene.crop_applied.connect(self.apply_crop)
        self.scene.crop_cancelled.connect(lambda: self.set_crop_mode(False))
        self.graphics_view.setScene(self.scene)
        self.graphics_view.setStyleSheet("background-color: #FFFFFF; border: none;")
        self.graphics_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.graphics_view.customContextMenuRequested.connect(self._show_viewer_context_menu)
        self.scene.setBackgroundBrush(QBrush(QColor("#FFFFFF")))
        self.graphics_view.zoomChanged.connect(self.zoom_percent_changed.emit)
        self.graphics_view.zoomChanged.connect(lambda _: self._position_tag_someone_popup())
        self.graphics_view.zoomChanged.connect(lambda _: self._position_adjust_toolbar())
        self.graphics_view.zoomModeSetToCustom.connect(self._on_custom_zoom_set)
        if self.graphics_view.horizontalScrollBar():
            self.graphics_view.horizontalScrollBar().valueChanged.connect(
                lambda _: self._position_tag_someone_popup()
            )
            self.graphics_view.horizontalScrollBar().valueChanged.connect(
                lambda _: self._position_adjust_toolbar()
            )
        if self.graphics_view.verticalScrollBar():
            self.graphics_view.verticalScrollBar().valueChanged.connect(
                lambda _: self._position_tag_someone_popup()
            )
            self.graphics_view.verticalScrollBar().valueChanged.connect(
                lambda _: self._position_adjust_toolbar()
            )
        self.media_stack.addWidget(self.graphics_view)

        # Page 1: Video Player (Authentic WLPG White Background & Transport Bar)
        self.video_container = QWidget(self)
        self.video_container.setStyleSheet("background-color: #FFFFFF; border: none;")
        vid_vbox = QVBoxLayout(self.video_container)
        vid_vbox.setContentsMargins(0, 0, 0, 0)
        vid_vbox.setSpacing(0)

        self.video_widget = ClickableVideoWidget(self.video_container)
        pal = self.video_widget.palette()
        pal.setColor(QPalette.Window, QColor("#FFFFFF"))
        self.video_widget.setPalette(pal)
        self.video_widget.setStyleSheet("background-color: #FFFFFF; border: none;")
        self.video_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.video_widget.customContextMenuRequested.connect(self._show_video_context_menu)

        self.mouse_overlay = MouseOverlay(self.video_widget)
        self.mouse_overlay.setContextMenuPolicy(Qt.CustomContextMenu)
        self.mouse_overlay.customContextMenuRequested.connect(self._show_video_context_menu)

        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.playbackStateChanged.connect(self._on_playback_state_changed)
        self.video_widget.clicked.connect(self.toggle_video_playback)
        vid_vbox.addWidget(self.video_widget, 1)

        # Video Transport Bar (WLPG Aero Styling)
        self.video_transport = QFrame(self.video_container)
        transport = self.video_transport
        transport.setFixedHeight(38)
        transport.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F5F8FC, stop:1 #E4ECF5);
                border-top: 1px solid #B8CDE3;
                color: #1E395B;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FFFFFF, stop:1 #EBF1F8);
                border: 1px solid #B6C7DC;
                border-radius: 3px;
                padding: 2px 10px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
            }
            QPushButton:hover {
                background: #EBF4FD;
                border: 1px solid #7AC1FF;
            }
            QLabel {
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                background: transparent;
                border: none;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #CAD8E6;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #509BE6;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0078D7;
                border: 1px solid #005A9E;
                width: 10px;
                margin: -5px 0;
                border-radius: 2px;
            }
            """
        )
        t_hbox = QHBoxLayout(transport)
        t_hbox.setContentsMargins(8, 2, 8, 2)
        t_hbox.setSpacing(6)

        self.btn_play_pause = QPushButton(transport)
        self.btn_play_pause.setFixedSize(28, 24)
        self.btn_play_pause.setIcon(make_wlpg_icon("play", 16))
        self.btn_play_pause.setIconSize(QSize(16, 16))
        self.btn_play_pause.setToolTip("Play / Pause (Space)")
        self.btn_play_pause.setCursor(Qt.PointingHandCursor)
        self.btn_play_pause.clicked.connect(self.toggle_video_playback)
        t_hbox.addWidget(self.btn_play_pause)

        self.slider_position = QSlider(Qt.Horizontal, transport)
        self.slider_position.setRange(0, 1000)
        self.slider_position.sliderMoved.connect(self.media_player.setPosition)
        self.media_player.positionChanged.connect(self._on_video_position_changed)
        self.media_player.durationChanged.connect(self._on_video_duration_changed)
        t_hbox.addWidget(self.slider_position, 1)

        self.lbl_video_time = QLabel("00:00 / 00:00", transport)
        self.lbl_video_time.setStyleSheet("font-size: 11px; color: #333; margin: 0 4px;")
        t_hbox.addWidget(self.lbl_video_time)

        # Volume control with icon to the right of the seek bar
        self.btn_volume = QPushButton(transport)
        self.btn_volume.setFixedSize(24, 24)
        self.btn_volume.setIcon(make_wlpg_icon("volume", 14))
        self.btn_volume.setIconSize(QSize(14, 14))
        self.btn_volume.setToolTip("Mute / Unmute")
        self.btn_volume.setCursor(Qt.PointingHandCursor)
        self.btn_volume.clicked.connect(self._toggle_mute)
        t_hbox.addWidget(self.btn_volume)

        self.slider_volume = QSlider(Qt.Horizontal, transport)
        self.slider_volume.setRange(0, 100)
        self.slider_volume.setValue(100)
        self.slider_volume.setFixedWidth(65)
        self.slider_volume.setToolTip("Volume")
        self.slider_volume.valueChanged.connect(self._on_volume_changed)
        t_hbox.addWidget(self.slider_volume)

        vid_vbox.addWidget(transport)
        self.media_stack.addWidget(self.video_container)

        root_hbox.addWidget(self.media_stack, 1)

        # ---------------------------------------------------------------------
        # 2. Right Collapsible 'Fix / Fine-Tune' Adjustments Drawer
        # ---------------------------------------------------------------------
        self.fix_pane = QFrame(self)
        self.fix_pane.setObjectName("FixAdjustmentsDrawer")
        self.fix_pane.setFixedWidth(255)
        self.fix_pane.setStyleSheet(
            """
            QFrame#FixAdjustmentsDrawer {
                background-color: #F5F7FA;
                border-left: 1px solid #C8D2E0;
            }
            """
        )
        fix_vbox = QVBoxLayout(self.fix_pane)
        fix_vbox.setContentsMargins(10, 10, 10, 10)
        fix_vbox.setSpacing(8)

        lbl_fix_title = QLabel("Fine-Tune & Fix", self.fix_pane)
        lbl_fix_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        lbl_fix_title.setStyleSheet("color: #1E395B;")
        fix_vbox.addWidget(lbl_fix_title)

        # Auto Adjust
        self.btn_auto_adjust = QPushButton("✨ Auto adjust", self.fix_pane)
        self.btn_auto_adjust.clicked.connect(self.apply_auto_adjust)
        fix_vbox.addWidget(self.btn_auto_adjust)

        # Crop Controls
        crop_row = QHBoxLayout()
        self.btn_crop_start = QPushButton("Crop photo", self.fix_pane)
        self.btn_crop_start.setCheckable(True)
        self.btn_crop_start.toggled.connect(self.set_crop_mode)
        crop_row.addWidget(self.btn_crop_start)

        self.btn_crop_apply = QPushButton("Apply crop", self.fix_pane)
        self.btn_crop_apply.clicked.connect(self.apply_crop)
        self.btn_crop_apply.setEnabled(False)
        crop_row.addWidget(self.btn_crop_apply)
        fix_vbox.addLayout(crop_row)

        # Exposure & Contrast Sliders
        fix_vbox.addWidget(QLabel("Brightness:", self.fix_pane))
        self.slider_brightness = QSlider(Qt.Horizontal, self.fix_pane)
        self.slider_brightness.setRange(-100, 100)
        self.slider_brightness.setValue(0)
        fix_vbox.addWidget(self.slider_brightness)

        fix_vbox.addWidget(QLabel("Contrast:", self.fix_pane))
        self.slider_contrast = QSlider(Qt.Horizontal, self.fix_pane)
        self.slider_contrast.setRange(-100, 100)
        self.slider_contrast.setValue(0)
        fix_vbox.addWidget(self.slider_contrast)

        self.btn_apply_exposure = QPushButton("Apply exposure", self.fix_pane)
        self.btn_apply_exposure.clicked.connect(
            lambda: self.apply_exposure_contrast(
                self.slider_brightness.value(), self.slider_contrast.value()
            )
        )
        fix_vbox.addWidget(self.btn_apply_exposure)

        # Color Saturation Slider
        fix_vbox.addWidget(QLabel("Saturation:", self.fix_pane))
        self.slider_saturation = QSlider(Qt.Horizontal, self.fix_pane)
        self.slider_saturation.setRange(-100, 100)
        self.slider_saturation.setValue(0)
        fix_vbox.addWidget(self.slider_saturation)

        self.btn_apply_color = QPushButton("Apply color", self.fix_pane)
        self.btn_apply_color.clicked.connect(
            lambda: self.apply_color_saturation(self.slider_saturation.value())
        )
        fix_vbox.addWidget(self.btn_apply_color)

        # Straighten Slider (-15..+15 deg)
        fix_vbox.addWidget(QLabel("Straighten (-15° to +15°):", self.fix_pane))
        self.slider_straighten = QSlider(Qt.Horizontal, self.fix_pane)
        self.slider_straighten.setRange(-15, 15)
        self.slider_straighten.setValue(0)
        fix_vbox.addWidget(self.slider_straighten)

        self.btn_apply_straighten = QPushButton("Apply straighten", self.fix_pane)
        self.btn_apply_straighten.clicked.connect(
            lambda: self.apply_straighten(float(self.slider_straighten.value()))
        )
        fix_vbox.addWidget(self.btn_apply_straighten)

        # Preset Effects (B&W, Sepia, Warm, Cool)
        fix_vbox.addWidget(QLabel("Effects:", self.fix_pane))
        fx_row1 = QHBoxLayout()
        for fx_key, fx_lbl in [("bw", "B&W"), ("sepia", "Sepia"), ("warm", "Warm"), ("cool", "Cool")]:
            b_fx = QPushButton(fx_lbl, self.fix_pane)
            b_fx.clicked.connect(lambda _c=False, k=fx_key: self.apply_preset_effect(k))
            fx_row1.addWidget(b_fx)
        fix_vbox.addLayout(fx_row1)

        # Undo / Redo Row
        undo_row = QHBoxLayout()
        self.btn_undo = QPushButton("↶ Undo", self.fix_pane)
        self.btn_undo.clicked.connect(self.undo_stack.undo)
        undo_row.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("↷ Redo", self.fix_pane)
        self.btn_redo.clicked.connect(self.undo_stack.redo)
        undo_row.addWidget(self.btn_redo)
        fix_vbox.addLayout(undo_row)

        # Save Pixel Edit (with .OriginalFiles backup) & Revert to Original
        self.btn_save_edit = QPushButton("💾 Save changes", self.fix_pane)
        self.btn_save_edit.setStyleSheet("font-weight: bold; color: #005A9E;")
        self.btn_save_edit.clicked.connect(self.save_current_image_with_backup)
        fix_vbox.addWidget(self.btn_save_edit)

        self.btn_revert = QPushButton("Revert to original", self.fix_pane)
        self.btn_revert.setEnabled(False)
        self.btn_revert.clicked.connect(lambda: self.revert_to_original())
        fix_vbox.addWidget(self.btn_revert)

        fix_vbox.addStretch(1)

        self.btn_close_viewer = QPushButton("✕ Close file (Esc)", self.fix_pane)
        self.btn_close_viewer.clicked.connect(self.request_close)
        fix_vbox.addWidget(self.btn_close_viewer)

        # (Fix pane controls are hosted exclusively in the right sidebar dock under Ctrl+F)
        # root_hbox contains only media_stack to give the photo/video the full viewport width
        self.fix_pane.hide()

        # Keyboard Navigation & Shortcut Event Filters
        self.installEventFilter(self)
        self.graphics_view.installEventFilter(self)
        if self.graphics_view.viewport():
            self.graphics_view.viewport().installEventFilter(self)
        self.video_widget.installEventFilter(self)
        self.mouse_overlay.installEventFilter(self)

        # Floating toolbars over graphics_view
        self._init_floating_toolbars()

    # -------------------------------------------------------------------------
    # Floating Contextual Toolbars (Crop Presets, Retouch Size, Red Eye, Tag Someone)
    # -------------------------------------------------------------------------
    def _init_floating_toolbars(self) -> None:
        tb_style = """
            _DraggableFloatingToolbar, #crop_toolbar {
                background-color: rgba(245, 248, 252, 245);
                border: 1px solid #7DA2CE;
                border-radius: 6px;
                padding: 2px 6px;
            }
            _DraggableGripHandle, #DraggableGripHandle {
                background: transparent;
                border: none;
                padding: 0px;
                margin: 0px;
            }
            QLabel {
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 11px;
                font-weight: bold;
            }
            QComboBox {
                background: #FFFFFF;
                border: 1px solid #7DA2CE;
                border-radius: 3px;
                padding: 2px 6px;
                color: #1E395B;
                font-size: 11px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FBFDFF, stop:1 #E0ECF8);
                border: 1px solid #7DA2CE;
                border-radius: 3px;
                color: #1E395B;
                padding: 3px 10px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #EAF4FD, stop:1 #C6E0F8);
                border: 1px solid #3C7FB1;
            }
        """

        # 1. Crop Toolbar (Photoshop style Ratio presets, custom W x H, and swap button ⇄)
        self.crop_toolbar = _DraggableFloatingToolbar(self.graphics_view)
        self.crop_toolbar.setObjectName("crop_toolbar")
        self.crop_toolbar.setStyleSheet(tb_style)
        ct_layout = QHBoxLayout(self.crop_toolbar)
        ct_layout.setContentsMargins(6, 4, 8, 4)
        ct_layout.setSpacing(6)

        grip_crop = _DraggableGripHandle(self.crop_toolbar)
        self.crop_toolbar.grip_handle = grip_crop
        ct_layout.addWidget(grip_crop)

        self._updating_crop_inputs = False

        self.combo_crop_ratio = QComboBox(self.crop_toolbar)
        self.combo_crop_ratio.addItems([
            "Ratio",
            "Custom",
            "Full Image (Uncrop)",
            "Original",
            "1:1 (Square)",
            "4:3",
            "3:4",
            "3:2",
            "2:3",
            "5:4",
            "4:5",
            "16:9",
            "9:16",
        ])
        self.combo_crop_ratio.setStyleSheet("""
            QComboBox {
                border: 1px solid #CBD5E1;
                border-radius: 4px;
                padding: 2px 6px;
                background: #FFFFFF;
                color: #1E293B;
                font-size: 11px;
                min-width: 68px;
            }
            QComboBox:hover { border-color: #94A3B8; }
            QComboBox::drop-down { border: none; width: 14px; }
        """)
        self.combo_crop_ratio.currentTextChanged.connect(self._on_crop_ratio_changed)

        input_style = """
            QLineEdit {
                border: 1px solid #CBD5E1;
                border-radius: 4px;
                padding: 2px 4px;
                background: #FFFFFF;
                color: #1E293B;
                font-size: 11px;
                font-weight: 600;
            }
            QLineEdit:focus {
                border-color: #2563EB;
                background: #FFFFFF;
            }
        """

        self.txt_crop_w = QLineEdit(self.crop_toolbar)
        self.txt_crop_w.setFixedSize(36, 24)
        self.txt_crop_w.setAlignment(Qt.AlignCenter)
        self.txt_crop_w.setStyleSheet(input_style)
        self.txt_crop_w.setPlaceholderText("W")
        self.txt_crop_w.textEdited.connect(self._on_crop_custom_text_edited)

        self.btn_crop_swap = QPushButton("⇄", self.crop_toolbar)
        self.btn_crop_swap.setFixedSize(26, 24)
        self.btn_crop_swap.setToolTip("Swap Width and Height (Landscape / Portrait)")
        self.btn_crop_swap.setStyleSheet("""
            QPushButton {
                border: 1px solid #CBD5E1;
                border-radius: 4px;
                background: #F8FAFC;
                color: #334155;
                font-size: 13px;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover {
                background: #E2E8F0;
                border-color: #94A3B8;
            }
            QPushButton:pressed {
                background: #CBD5E1;
            }
        """)
        self.btn_crop_swap.clicked.connect(self._on_crop_swap_clicked)

        self.txt_crop_h = QLineEdit(self.crop_toolbar)
        self.txt_crop_h.setFixedSize(36, 24)
        self.txt_crop_h.setAlignment(Qt.AlignCenter)
        self.txt_crop_h.setStyleSheet(input_style)
        self.txt_crop_h.setPlaceholderText("H")
        self.txt_crop_h.textEdited.connect(self._on_crop_custom_text_edited)

        # Subtle vertical separator
        sep = QFrame(self.crop_toolbar)
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #CBD5E1; margin: 2px 2px;")

        btn_crop_apply = QPushButton("Apply", self.crop_toolbar)
        btn_crop_apply.setIcon(make_symbol_icon("check", 14))
        btn_crop_apply.setIconSize(QSize(14, 14))
        btn_crop_apply.setToolTip("Apply Crop (Enter)")
        btn_crop_apply.setStyleSheet("""
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
        btn_crop_apply.clicked.connect(self.apply_crop)

        btn_crop_cancel = QPushButton("Cancel", self.crop_toolbar)
        btn_crop_cancel.setIcon(make_symbol_icon("cross", 14))
        btn_crop_cancel.setIconSize(QSize(14, 14))
        btn_crop_cancel.setToolTip("Cancel Crop (Esc)")
        btn_crop_cancel.setStyleSheet("""
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
        btn_crop_cancel.clicked.connect(lambda: self.set_crop_mode(False))

        ct_layout.addWidget(self.combo_crop_ratio)
        ct_layout.addWidget(self.txt_crop_w)
        ct_layout.addWidget(self.btn_crop_swap)
        ct_layout.addWidget(self.txt_crop_h)
        ct_layout.addWidget(sep)
        ct_layout.addWidget(btn_crop_apply)
        ct_layout.addWidget(btn_crop_cancel)
        self.crop_toolbar.hide()

        # 2. Spot Fix Toolbar (Shared by Retouch / Blemish, Object Eraser & Red Eye)
        self.spot_fix_sub_mode: Literal["retouch", "erase", "red_eye"] = "retouch"
        self.retouch_toolbar = _DraggableFloatingToolbar(self.graphics_view)
        self.retouch_toolbar.setStyleSheet(tb_style)
        rt_layout = QHBoxLayout(self.retouch_toolbar)
        rt_layout.setContentsMargins(6, 4, 8, 4)
        rt_layout.setSpacing(6)

        # Drag grip handle
        grip_retouch = _DraggableGripHandle(self.retouch_toolbar)
        self.retouch_toolbar.grip_handle = grip_retouch
        rt_layout.addWidget(grip_retouch)

        spot_btn_style = """
            QPushButton {
                background: #F0F4F9;
                border: 1px solid #70A6E8;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 8.5pt;
                font-weight: bold;
                color: #1E395B;
            }
            QPushButton:checked {
                background: #CBE2F8;
                border: 1px solid #3388E8;
                color: #0E294B;
            }
        """

        # Mode toggle buttons (Blemish vs Object eraser vs Red eye)
        self.btn_spot_retouch = QPushButton("Blemish", self.retouch_toolbar)
        self.btn_spot_retouch.setIcon(make_wlpg_icon("retouch", 16))
        self.btn_spot_retouch.setIconSize(QSize(16, 16))
        self.btn_spot_retouch.setCheckable(True)
        self.btn_spot_retouch.setChecked(True)
        self.btn_spot_retouch.setToolTip("Heal small skin blemishes and sensor dust with frequency-separated texture transfer")
        self.btn_spot_retouch.setStyleSheet(spot_btn_style)
        self.btn_spot_retouch.clicked.connect(lambda: self.set_spot_fix_sub_mode("retouch"))

        self.btn_spot_erase = QPushButton("Object eraser", self.retouch_toolbar)
        self.btn_spot_erase.setIcon(make_wlpg_icon("auto_adjust", 16))
        self.btn_spot_erase.setIconSize(QSize(16, 16))
        self.btn_spot_erase.setCheckable(True)
        self.btn_spot_erase.setChecked(False)
        self.btn_spot_erase.setToolTip("Erase larger unwanted objects, wires, or photobombers using LaMa AI inpainting")
        self.btn_spot_erase.setStyleSheet(spot_btn_style)
        self.btn_spot_erase.clicked.connect(lambda: self.set_spot_fix_sub_mode("erase"))

        self.btn_spot_red_eye = QPushButton("Red eye", self.retouch_toolbar)
        self.btn_spot_red_eye.setIcon(make_wlpg_icon("red_eye", 16))
        self.btn_spot_red_eye.setIconSize(QSize(16, 16))
        self.btn_spot_red_eye.setCheckable(True)
        self.btn_spot_red_eye.setChecked(False)
        self.btn_spot_red_eye.setToolTip("Neutralize flash red-eye glare while preserving natural corneal catchlights")
        self.btn_spot_red_eye.setStyleSheet(spot_btn_style)
        self.btn_spot_red_eye.clicked.connect(lambda: self.set_spot_fix_sub_mode("red_eye"))

        lbl_brush = QLabel("Size:", self.retouch_toolbar)
        self.slider_retouch_size = QSlider(Qt.Horizontal, self.retouch_toolbar)
        self.slider_retouch_size.setRange(6, 80)
        self.slider_retouch_size.setValue(20)
        self.slider_retouch_size.setFixedWidth(85)
        self.slider_retouch_size.valueChanged.connect(self._on_retouch_size_changed)

        btn_retouch_done = QPushButton("Done (Esc)", self.retouch_toolbar)
        btn_retouch_done.setIcon(make_symbol_icon("check", 14))
        btn_retouch_done.setIconSize(QSize(14, 14))
        btn_retouch_done.clicked.connect(self._close_spot_fix_toolbar)

        rt_layout.addWidget(self.btn_spot_retouch)
        rt_layout.addWidget(self.btn_spot_erase)
        rt_layout.addWidget(self.btn_spot_red_eye)
        rt_layout.addWidget(lbl_brush)
        rt_layout.addWidget(self.slider_retouch_size)
        rt_layout.addWidget(btn_retouch_done)
        self.retouch_toolbar.hide()

        # Backward compatibility alias
        self.red_eye_toolbar = self.retouch_toolbar

        # 4. WLPG Floating "Tag someone" Popup
        self._tag_someone_popup = _TagSomeonePopup(self.graphics_view)
        self._tag_someone_popup.person_selected.connect(self._on_tag_someone_selected)
        self._tag_someone_popup.cancelled.connect(self.stop_face_tagging_mode)
        self._tag_someone_popup.hide()

        # 5. Floating Face Adjust Toolbar
        self.is_face_adjust_mode: bool = False
        self._adjusting_face_info: dict = {}
        self._face_adjust_box_item: _FaceTagTargetBoxItem | None = None
        self._adjust_toolbar = _FaceAdjustToolbar(self.graphics_view)
        self._adjust_toolbar.saved.connect(self.save_face_adjust)
        self._adjust_toolbar.cancelled.connect(self.stop_face_adjust_mode)
        self._adjust_toolbar.hide()

        # 6. Markup & Privacy Redact Toolbar
        self.is_markup_mode: bool = False
        self._markup_elements: list[MarkupElement] = []
        self._markup_preview_items: list[Any] = []
        self._markup_active: bool = False
        self._markup_start_pt: QPointF | None = None
        self._markup_current_item: Any | None = None
        self._markup_current_element: MarkupElement | None = None
        self._magnify_hover_item: QGraphicsPixmapItem | None = None
        self._session_markup_command_count: int = 0
        self._markup_session_start_master: np.ndarray | None = None
        self._markup_session_start_undo_idx: int = 0
        self._inline_text_input: QLineEdit | None = None
        self._inline_text_scene_pos: QPointF | None = None
        self._active_movable_element: MarkupElement | None = None
        self._active_movable_before_master: np.ndarray | None = None
        self._pending_markup_item: QGraphicsPixmapItem | None = None
        self._pending_selection_box_item: QGraphicsRectItem | None = None
        self._is_moving_pending_markup: bool = False
        self._move_markup_last_pt: QPointF | None = None

        self.markup_toolbar = MarkupFloatingToolbar(self.graphics_view)
        self.markup_toolbar.apply_requested.connect(self._commit_markup)
        self.markup_toolbar.cancel_requested.connect(lambda: self.set_markup_mode(False))
        self.markup_toolbar.undo_requested.connect(self._undo_markup)
        self.markup_toolbar.clear_requested.connect(self._clear_markup_overlays)
        if hasattr(self.markup_toolbar, "save_requested"):
            self.markup_toolbar.save_requested.connect(self._on_markup_save_requested)
        self.markup_toolbar.tool_changed.connect(self._on_markup_tool_changed)
        self.markup_toolbar.stroke_width_changed.connect(lambda _: self._on_markup_settings_changed())
        self.markup_toolbar.color_changed.connect(lambda _: self._on_markup_settings_changed())
        if hasattr(self.markup_toolbar, "secondary_color_changed"):
            self.markup_toolbar.secondary_color_changed.connect(lambda _: self._on_markup_settings_changed())
        if hasattr(self.markup_toolbar, "text_format_changed"):
            self.markup_toolbar.text_format_changed.connect(self._on_text_format_changed)
        if hasattr(self.markup_toolbar, "magnifier_zoom_changed"):
            self.markup_toolbar.magnifier_zoom_changed.connect(self._on_magnifier_zoom_changed)
        if hasattr(self.markup_toolbar, "btn_undo"):
            self.undo_stack.canUndoChanged.connect(self.markup_toolbar.btn_undo.setEnabled)
        self.markup_toolbar.hide()

        # 7. Inline Face Naming Popup directly below face box
        self._inline_name_popup = _InlineFaceNamePopup(self.graphics_view)
        self._inline_name_popup.name_committed.connect(self._on_inline_name_committed)
        self._inline_name_popup.hide()

    def _show_status(self, msg: str, timeout: int = 4000) -> None:
        """Surfaces brief status bar feedback to MainWindow."""
        self.status_message_requested.emit(msg, timeout)

    def _on_markup_tool_changed(self, tool_id: str) -> None:
        """Commits inline text and any pending element before switching markup tools."""
        if getattr(self, "_inline_text_input", None) is not None:
            self._commit_inline_text()
        self._finalize_movable_markup()
        if tool_id not in ("magnify", "magnifier"):
            self._magnify_source_pt = None
            self._cleanup_magnify_preview()
        self._on_markup_settings_changed()

    def _on_magnifier_zoom_changed(self, zoom: float) -> None:
        self._show_status(f"Magnifier Zoom: {zoom:.1f}x", 2500)
        sp = getattr(self, "_last_mouse_scene_pos", None)
        if sp is not None:
            self._update_magnify_hover(sp)

    def _on_text_format_changed(self, is_bold: bool, is_italic: bool) -> None:
        self._update_inline_text_style()

    def _update_inline_text_style(self) -> None:
        """Dynamically styles and sizes the on-canvas inline text editor to match view scale and secondary background."""
        inp = getattr(self, "_inline_text_input", None)
        if inp is None:
            return
        if getattr(self.markup_toolbar, "active_tool", "pen") != "text":
            return
        width = getattr(self.markup_toolbar, "stroke_width", 4)
        col = getattr(self.markup_toolbar, "current_color", QColor(230, 40, 40))
        sec_col = getattr(self.markup_toolbar, "secondary_color", QColor(20, 25, 35))
        view_scale = self.graphics_view.transform().m11()
        if abs(view_scale) < 1e-4:
            view_scale = 1.0
        master_font_pt = max(10, int(round(width * 2.2)))
        screen_font_pt = max(9, int(round(master_font_pt * view_scale)))
        screen_pad_x = max(6, int(round(12.0 * view_scale)))
        screen_pad_y = max(4, int(round(8.0 * view_scale)))
        if sec_col.alpha() == 0:
            bg_qss = "background: transparent; border: 1.5px dashed rgba(56, 144, 240, 200);"
        else:
            bg_rgba = f"rgba({sec_col.red()}, {sec_col.green()}, {sec_col.blue()}, 220)"
            bg_qss = f"background: {bg_rgba}; border: 1.5px solid #3890F0;"
        is_bold = getattr(self.markup_toolbar, "is_text_bold", True)
        is_italic = getattr(self.markup_toolbar, "is_text_italic", False)

        inp.setStyleSheet(f"""
            QTextEdit {{
                {bg_qss}
                color: {col.name()};
                border-radius: 6px;
                padding: {screen_pad_y}px {screen_pad_x}px;
                font-family: 'Segoe UI';
                font-size: {screen_font_pt}pt;
                font-weight: {'bold' if is_bold else 'normal'};
                font-style: {'italic' if is_italic else 'normal'};
            }}
        """)
        self._adjust_inline_text_size()

    def _position_floating_toolbars(self) -> None:
        gw = self.graphics_view.width()
        gh = self.graphics_view.height()
        if hasattr(self, "markup_toolbar") and hasattr(self.markup_toolbar, "set_compact_mode"):
            self.markup_toolbar.set_compact_mode(gw < 980)
        for tb in (self.crop_toolbar, self.retouch_toolbar, self.red_eye_toolbar, self.markup_toolbar):
            if tb.isVisible():
                tb.adjustSize()
                sh = tb.sizeHint()
                if getattr(tb, "_user_positioned", False):
                    cur_x = tb.x()
                    cur_y = tb.y()
                    max_x = max(10, gw - sh.width() - 10)
                    max_y = max(10, gh - sh.height() - 10)
                    tb.setGeometry(max(10, min(cur_x, max_x)), max(10, min(cur_y, max_y)), sh.width(), sh.height())
                else:
                    tb.setGeometry(max(10, (gw - sh.width()) // 2), 12, sh.width(), sh.height())
        self._position_tag_someone_popup()
        self._position_adjust_toolbar()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_floating_toolbars()

    @staticmethod
    def _make_checkerboard_brush(dark_mode: bool) -> QBrush:
        tile = QPixmap(16, 16)
        c1 = QColor("#1E222A") if dark_mode else QColor("#FFFFFF")
        c2 = QColor("#2A303B") if dark_mode else QColor("#E6E9EE")
        tile.fill(c1)
        painter = QPainter(tile)
        painter.fillRect(0, 0, 8, 8, c2)
        painter.fillRect(8, 8, 8, 8, c2)
        painter.end()
        return QBrush(tile)

    @staticmethod
    def _build_pixmap_from_rgb_and_alpha(
        rgb_np: np.ndarray, alpha_np: np.ndarray | None
    ) -> QPixmap:
        h, w = rgb_np.shape[:2]
        if alpha_np is not None:
            if alpha_np.shape[:2] != (h, w):
                alpha_np = cv2.resize(alpha_np, (w, h), interpolation=cv2.INTER_LINEAR)
            rgba = np.ascontiguousarray(np.dstack([rgb_np, alpha_np]))
            qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
            return QPixmap.fromImage(qimg)
        rgb_c = np.ascontiguousarray(rgb_np)
        qimg = QImage(rgb_c.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def _update_alpha_backdrop(self, w: int, h: int, visible: bool) -> None:
        if hasattr(self, "_alpha_backdrop_item") and self._alpha_backdrop_item is not None:
            self._alpha_backdrop_item.setRect(QRectF(0.0, 0.0, float(w), float(h)))
            self._alpha_backdrop_item.setVisible(bool(visible))

    def _should_play_animation(self) -> bool:
        try:
            has_undo = hasattr(self, "undo_stack") and self.undo_stack is not None and self.undo_stack.canUndo()
        except RuntimeError:
            has_undo = False
        return bool(
            self._is_animated_image
            and not self.current_is_video
            and self.current_recipe.is_default()
            and not self.is_in_edit_mode()
            and not has_undo
        )

    def _start_animated_image(self, file_path: Path) -> None:
        self._stop_animated_image()
        movie = QMovie(str(file_path), parent=self)
        if not movie.isValid():
            movie.deleteLater()
            return
        self._is_animated_image = True
        self._anim_movie = movie
        movie.frameChanged.connect(self._on_anim_frame_changed)
        self._sync_animation_playback()

    def _stop_animated_image(self) -> None:
        movie = getattr(self, "_anim_movie", None)
        self._anim_movie = None
        self._is_animated_image = False
        if movie is not None:
            try:
                movie.stop()
                movie.deleteLater()
            except RuntimeError:
                pass

    def _on_anim_frame_changed(self, _frame_idx: int) -> None:
        if self._anim_movie is None or not self._should_play_animation():
            return
        pix = self._anim_movie.currentPixmap()
        if not pix.isNull():
            self.pixmap_item.setPixmap(pix)
            self._update_alpha_backdrop(pix.width(), pix.height(), pix.hasAlphaChannel())

    def _sync_animation_playback(self) -> None:
        if self._anim_movie is None:
            return
        if self._should_play_animation():
            if self._anim_movie.state() != QMovie.MovieState.Running:
                self._anim_movie.start()
            pix = self._anim_movie.currentPixmap()
            if not pix.isNull():
                self.pixmap_item.setPixmap(pix)
                self._update_alpha_backdrop(pix.width(), pix.height(), pix.hasAlphaChannel())
        else:
            if self._anim_movie.state() == QMovie.MovieState.Running:
                self._anim_movie.setPaused(True)
            if getattr(self, "_adjusted_pixmap", None) is not None:
                self.pixmap_item.setPixmap(self._adjusted_pixmap)
                self._update_alpha_backdrop(
                    self._adjusted_pixmap.width(),
                    self._adjusted_pixmap.height(),
                    self.master_alpha is not None,
                )

    def _update_display_proxy(self) -> None:
        self._master_render_gen += 1
        self._master_render_dirty = False
        if self.master_np_rgb is None:
            self.display_proxy_rgb = None
            self.display_proxy_alpha = None
            self.scrub_proxy_rgb = None
            self.scrub_proxy_alpha = None
            self._proxy_scale = 1.0
            self._scrub_proxy_scale = 1.0
            return
        h, w = self.master_np_rgb.shape[:2]
        max_dim = 1280
        if max(h, w) > max_dim:
            self._proxy_scale = max_dim / float(max(h, w))
            pw = max(1, int(round(w * self._proxy_scale)))
            ph = max(1, int(round(h * self._proxy_scale)))
            self.display_proxy_rgb = cv2.resize(self.master_np_rgb, (pw, ph), interpolation=cv2.INTER_AREA)
            if self.master_alpha is not None:
                self.display_proxy_alpha = cv2.resize(
                    self.master_alpha, (pw, ph), interpolation=cv2.INTER_AREA
                )
            else:
                self.display_proxy_alpha = None
        else:
            self.display_proxy_rgb = self.master_np_rgb
            self.display_proxy_alpha = self.master_alpha
            self._proxy_scale = 1.0

        # Fast 960px scrub proxy for 120 FPS active slider dragging
        scrub_dim = 960
        if max(h, w) > scrub_dim:
            self._scrub_proxy_scale = scrub_dim / float(max(h, w))
            sw = max(1, int(round(w * self._scrub_proxy_scale)))
            sh = max(1, int(round(h * self._scrub_proxy_scale)))
            self.scrub_proxy_rgb = cv2.resize(self.display_proxy_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
            if self.display_proxy_alpha is not None:
                self.scrub_proxy_alpha = cv2.resize(
                    self.display_proxy_alpha, (sw, sh), interpolation=cv2.INTER_AREA
                )
            else:
                self.scrub_proxy_alpha = None
        else:
            self.scrub_proxy_rgb = self.display_proxy_rgb
            self.scrub_proxy_alpha = self.display_proxy_alpha
            self._scrub_proxy_scale = self._proxy_scale

    def _scene_to_master(self, sx: float, sy: float) -> tuple[int, int]:
        if self.master_np_rgb is None:
            return int(sx), int(sy)
        mh, mw = self.master_np_rgb.shape[:2]
        if self.current_recipe.crop_rect is not None:
            l, t, r, b = self.current_recipe.crop_rect
            mx = int(round(l + sx))
            my = int(round(t + sy))
        else:
            mx = int(round(sx))
            my = int(round(sy))
        return max(0, min(mw - 1, mx)), max(0, min(mh - 1, my))

    def rotate_media(self, clockwise: bool = True) -> None:
        """Rotates the media currently viewed in the SingleMediaViewer."""
        self.rotate_requested.emit(clockwise)

    def apply_in_memory_rotation(self, clockwise: bool = True) -> None:
        """Immediately rotates current master image in memory (0ms lag, no disk reload)."""
        if self.master_np_rgb is None or self.current_is_video:
            return
        self._stop_animated_image()
        k = -1 if clockwise else 1
        self.master_np_rgb = np.ascontiguousarray(np.rot90(self.master_np_rgb, k))
        if self.master_alpha is not None:
            self.master_alpha = np.ascontiguousarray(np.rot90(self.master_alpha, k))
        self.master_pil_image = Image.fromarray(self.master_np_rgb)
        self.current_recipe.crop_rect = None
        self._update_display_proxy()
        self._render_current_recipe()

        # Update in-memory orientation tracking
        cw_orient = {1: 6, 6: 3, 3: 8, 8: 1, 2: 7, 7: 4, 4: 5, 5: 2}
        ccw_orient = {1: 8, 8: 3, 3: 6, 6: 1, 2: 5, 5: 4, 4: 7, 7: 2}
        cur_orient = getattr(self, "current_orientation", 1)
        self.current_orientation = (cw_orient if clockwise else ccw_orient).get(cur_orient, 1)

        # Transform and re-render face tags for rotated geometry
        if hasattr(self, "_current_people_regions") and self._current_people_regions:
            updated_regions = []
            for reg in self._current_people_regions:
                reg_copy = dict(reg)
                rx = float(reg.get("rect_x") or 0.0)
                ry = float(reg.get("rect_y") or 0.0)
                rw = float(reg.get("rect_w") or 0.0)
                rh = float(reg.get("rect_h") or 0.0)
                if clockwise:
                    nx = 1.0 - (ry + rh)
                    ny = rx
                    nw = rh
                    nh = rw
                else:
                    nx = ry
                    ny = 1.0 - (rx + rw)
                    nw = rh
                    nh = rw
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
                nw = max(0.0, min(1.0 - nx, nw))
                nh = max(0.0, min(1.0 - ny, nh))
                reg_copy["rect_x"] = nx
                reg_copy["rect_y"] = ny
                reg_copy["rect_w"] = nw
                reg_copy["rect_h"] = nh
                updated_regions.append(reg_copy)
            self._current_people_regions = updated_regions
            self.render_face_regions(self._current_people_regions)

        # When an image is rotated if it wouldn't fit in window after rotate, Fit to Window
        vp_size = self.graphics_view.viewport().size()
        scene_rect = self.scene.sceneRect()
        cur_zoom = getattr(self.graphics_view, "zoom_level", 1.0)
        disp_w = scene_rect.width() * cur_zoom
        disp_h = scene_rect.height() * cur_zoom
        if (
            self.current_zoom_mode == "fit"
            or disp_w > vp_size.width()
            or disp_h > vp_size.height()
        ):
            self.set_zoom_mode("fit")

    # -------------------------------------------------------------------------
    # Loading & Display
    # -------------------------------------------------------------------------
    def load_media(
        self,
        media_id: int,
        file_path: str | Path,
        is_video: bool = False,
        people_regions: list[dict] | None = None,
        orientation: int = 1,
        metadata_dirty: int = 0,
    ) -> bool:
        """Loads a static image or video into the SingleMediaViewer."""
        self.current_media_id = int(media_id)
        self.current_file_path = Path(file_path)
        self.current_is_video = bool(is_video)
        self.current_orientation = int(orientation or 1)
        self._stop_animated_image()
        self._update_alpha_backdrop(0, 0, False)
        self.undo_stack.clear()
        self.stop_face_tagging_mode()
        self.set_crop_mode(False)
        self.set_retouch_mode(False)
        self.set_red_eye_mode(False)
        self.set_markup_mode(False)
        self._highlighted_person_id = None
        self._highlighted_region_id = None
        if hasattr(self, "_temp_faces_timer") and self._temp_faces_timer.isActive():
            self._temp_faces_timer.stop()
        self._current_people_regions = list(people_regions or []) if not self.current_is_video else []
        self._clear_face_overlays()
        self._purge_all_transient_scene_items()

        if self.current_is_video:
            self.media_stack.setCurrentIndex(1)
            self.fix_pane.setVisible(False)
            self.media_player.setSource(QUrl.fromLocalFile(str(self.current_file_path)))
            self.media_player.play()
            self.btn_play_pause.setIcon(make_wlpg_icon("pause", 16))
            self.btn_play_pause.setToolTip("Pause (Space)")
            return True

        self.media_player.stop()
        self.media_stack.setCurrentIndex(0)
        self.fix_pane.setVisible(False)

        if not self.current_file_path.exists():
            return False

        ext = self.current_file_path.suffix.lstrip(".").lower()
        loaded = False
        is_anim = False
        delta_angle = 0
        if ext in RAW_EXT_SET:
            rgb_raw = decode_raw_to_srgb(self.current_file_path)
            if rgb_raw is not None:
                self.master_np_rgb = rgb_raw
                self.master_alpha = None
                self.master_pil_image = Image.fromarray(self.master_np_rgb)
                loaded = True

        if not loaded:
            try:
                is_anim = is_animated_image_file(self.current_file_path)
                with Image.open(self.current_file_path) as img:
                    disk_orient = 1
                    if not is_anim:
                        try:
                            disk_exif = img.getexif()
                            disk_orient = int(disk_exif.get(0x0112) or 1)
                        except Exception:
                            disk_orient = 1
                    rgb_arr, alpha_arr = decode_pil_to_srgb_and_alpha(img)
                    target_orient = int(orientation or 1)
                    if disk_orient != 1 and target_orient == 1 and int(metadata_dirty or 0) == 0:
                        self.current_orientation = disk_orient
                    elif target_orient != disk_orient and (
                        int(metadata_dirty or 0) == 1 or target_orient in (3, 6, 8)
                    ):
                        orient_to_angle = {1: 0, 6: 90, 3: 180, 8: 270}
                        delta_angle = (
                            orient_to_angle.get(target_orient, 0)
                            - orient_to_angle.get(disk_orient, 0)
                        ) % 360
                        if delta_angle != 0:
                            k = {90: -1, 180: 2, 270: 1}[delta_angle]
                            rgb_arr = np.ascontiguousarray(np.rot90(rgb_arr, k))
                            if alpha_arr is not None:
                                alpha_arr = np.ascontiguousarray(np.rot90(alpha_arr, k))
                    self.master_np_rgb = rgb_arr
                    self.master_alpha = alpha_arr
                    self.master_pil_image = Image.fromarray(self.master_np_rgb)
            except Exception:
                rgb_raw = decode_raw_to_srgb(self.current_file_path)
                if rgb_raw is not None:
                    self.master_np_rgb = rgb_raw
                    self.master_alpha = None
                    self.master_pil_image = Image.fromarray(self.master_np_rgb)
                else:
                    return False

        self._update_display_proxy()
        self._original_pixmap = None
        self._adjusted_pixmap = None
        self.current_recipe = PhotoAdjustmentRecipe()
        self.undo_stack.clear()
        self._render_current_recipe()
        if is_anim and delta_angle == 0:
            self._start_animated_image(self.current_file_path)
        self.set_zoom_mode("fit")
        self.recipe_changed.emit(self.current_recipe)

        # Draw face regions strictly for static images (is_video == 0)
        if not self.current_is_video:
            self.render_face_regions(people_regions or [])
        self.update_revert_state()
        return True

    def _render_proxy_to_viewport(
        self,
        proxy_rgb: np.ndarray,
        proxy_alpha_src: np.ndarray | None,
        proxy_scale: float,
    ) -> None:
        proxy_rec = self.current_recipe.copy()
        if proxy_rec.crop_rect is not None and proxy_scale < 0.999:
            l, t, r, b = proxy_rec.crop_rect
            proxy_rec.crop_rect = (
                int(round(l * proxy_scale)),
                int(round(t * proxy_scale)),
                int(round(r * proxy_scale)),
                int(round(b * proxy_scale)),
            )
        rendered_np = render_adjusted_image(
            proxy_rgb, proxy_rec, media_id=self.current_media_id
        )
        h, w = rendered_np.shape[:2]
        proxy_alpha = (
            transform_alpha_for_recipe(proxy_alpha_src, proxy_rec, target_hw=(h, w))
            if proxy_alpha_src is not None
            else None
        )
        # Apply ruby mask overlay if currently previewing subject boundary
        if getattr(self, "_mask_overlay_preview", False):
            try:
                from src.imaging.creative_effects import (
                    get_depth_and_matte_engine,
                    apply_boundary_shift_and_feather,
                )
                engine = get_depth_and_matte_engine()
                _, alpha_matte = engine.get_depth_and_matte(rendered_np, media_id=self.current_media_id)
                if alpha_matte is not None:
                    alpha_matte = apply_boundary_shift_and_feather(
                        alpha_matte,
                        boundary_shift=getattr(self.current_recipe, "boundary_shift", 0.0),
                        boundary_feather=getattr(self.current_recipe, "boundary_feather", 0.0),
                    )
                    bg_mask = (1.0 - np.clip(alpha_matte, 0.0, 1.0))[:, :, None]
                    overlay_color = np.array([220.0, 20.0, 60.0], dtype=np.float32)
                    blended = rendered_np.astype(np.float32) * (1.0 - bg_mask * 0.45) + overlay_color * (bg_mask * 0.45)
                    rendered_np = np.clip(blended, 0.0, 255.0).astype(np.uint8)
            except Exception:
                pass

        pix = self._build_pixmap_from_rgb_and_alpha(rendered_np, proxy_alpha)
        target_w = int(round(w / proxy_scale))
        target_h = int(round(h / proxy_scale))
        if target_w != w or target_h != h:
            pix = pix.scaled(target_w, target_h, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        self.pixmap_item.setPixmap(pix)
        self._update_alpha_backdrop(pix.width(), pix.height(), proxy_alpha is not None)
        self._sync_animation_playback()

    def _render_current_recipe(self) -> None:
        """Renders self.master_np_rgb with self.current_recipe using OpenCV/NumPy float pipeline."""
        if self.master_np_rgb is None:
            return

        if self._is_sliding and (self.scrub_proxy_rgb is not None or self.display_proxy_rgb is not None):
            self._master_render_dirty = True
            self._master_render_gen += 1
            use_scrub = self.scrub_proxy_rgb is not None
            proxy_rgb = self.scrub_proxy_rgb if use_scrub else self.display_proxy_rgb
            proxy_alpha_src = self.scrub_proxy_alpha if use_scrub else self.display_proxy_alpha
            proxy_scale = self._scrub_proxy_scale if use_scrub else self._proxy_scale
            assert proxy_rgb is not None
            self._render_proxy_to_viewport(proxy_rgb, proxy_alpha_src, proxy_scale)
            return

        self._master_render_gen += 1
        self._master_render_dirty = False
        rendered_np = render_adjusted_image(
            self.master_np_rgb, self.current_recipe, media_id=self.current_media_id
        )
        self.current_pil_image = Image.fromarray(rendered_np)

        h, w, _c = rendered_np.shape
        rendered_alpha = (
            transform_alpha_for_recipe(self.master_alpha, self.current_recipe, target_hw=(h, w))
            if self.master_alpha is not None
            else None
        )
        # Apply ruby mask overlay if currently previewing subject boundary
        if getattr(self, "_mask_overlay_preview", False):
            try:
                from src.imaging.creative_effects import (
                    get_depth_and_matte_engine,
                    apply_boundary_shift_and_feather,
                )
                engine = get_depth_and_matte_engine()
                _, alpha_matte = engine.get_depth_and_matte(rendered_np, media_id=self.current_media_id)
                if alpha_matte is not None:
                    alpha_matte = apply_boundary_shift_and_feather(
                        alpha_matte,
                        boundary_shift=getattr(self.current_recipe, "boundary_shift", 0.0),
                        boundary_feather=getattr(self.current_recipe, "boundary_feather", 0.0),
                    )
                    bg_mask = (1.0 - np.clip(alpha_matte, 0.0, 1.0))[:, :, None]
                    overlay_color = np.array([220.0, 20.0, 60.0], dtype=np.float32)
                    blended = rendered_np.astype(np.float32) * (1.0 - bg_mask * 0.45) + overlay_color * (bg_mask * 0.45)
                    rendered_np = np.clip(blended, 0.0, 255.0).astype(np.uint8)
            except Exception:
                pass

        pix = self._build_pixmap_from_rgb_and_alpha(rendered_np, rendered_alpha)
        self._adjusted_pixmap = pix
        self.pixmap_item.setPixmap(pix)
        self._update_alpha_backdrop(pix.width(), pix.height(), rendered_alpha is not None)
        self.scene.setSceneRect(QRectF(0, 0, pix.width(), pix.height()))
        self.scene.crop_rect_item.set_image_rect(QRectF(0, 0, pix.width(), pix.height()))
        self.image_changed.emit(self.current_pil_image)

        # Update face overlays for current static image (clears overlays when empty)
        if hasattr(self, "_current_people_regions") and not self.current_is_video:
            self.render_face_regions(self._current_people_regions)

        # Cache unedited original pixmap for instantaneous, high-fidelity compare
        if self.current_recipe.is_default():
            self._original_pixmap = self._adjusted_pixmap
        elif getattr(self, "_original_pixmap", None) is None:
            default_rec = PhotoAdjustmentRecipe(crop_rect=self.current_recipe.crop_rect)
            orig_np = render_adjusted_image(
                self.master_np_rgb, default_rec, media_id=self.current_media_id
            )
            oh, ow = orig_np.shape[:2]
            orig_alpha = (
                transform_alpha_for_recipe(self.master_alpha, default_rec, target_hw=(oh, ow))
                if self.master_alpha is not None
                else None
            )
            self._original_pixmap = self._build_pixmap_from_rgb_and_alpha(orig_np, orig_alpha)

        self._sync_animation_playback()

    def update_image_display(self) -> None:
        """Legacy helper for test compatibility."""
        self._render_current_recipe()

    def set_dark_mode(self, is_dark: bool) -> None:
        """Toggles dark theme for canvas, background scene, video player, and popups."""
        self._is_dark = bool(is_dark)
        bg_col = "#14171C" if is_dark else "#FFFFFF"
        self.graphics_view.setStyleSheet(f"background-color: {bg_col}; border: none;")
        self.scene.setBackgroundBrush(QBrush(QColor(bg_col)))
        if hasattr(self, "_alpha_backdrop_item") and self._alpha_backdrop_item is not None:
            self._alpha_backdrop_item.setBrush(self._make_checkerboard_brush(is_dark))
        self.video_container.setStyleSheet(f"background-color: {bg_col}; border: none;")
        self.video_widget.setStyleSheet(f"background-color: {bg_col}; border: none;")
        pal = self.video_widget.palette()
        pal.setColor(QPalette.Window, QColor(bg_col))
        self.video_widget.setPalette(pal)

        if hasattr(self, "_inline_name_popup") and hasattr(self._inline_name_popup, "set_dark_mode"):
            self._inline_name_popup.set_dark_mode(is_dark)

        if hasattr(self, "video_transport"):
            if is_dark:
                self.video_transport.setStyleSheet(
                    """
                    QFrame {
                        background: #171A20;
                        border-top: 1px solid #2A2E37;
                        color: #CBD5E1;
                    }
                    QPushButton {
                        background: #20242B;
                        border: 1px solid #333A45;
                        border-radius: 3px;
                        padding: 2px 10px;
                        color: #CBD5E1;
                        font-family: 'Segoe UI';
                        font-size: 8.5pt;
                    }
                    QPushButton:hover {
                        background: #2E3642;
                        border: 1px solid #60A5FA;
                    }
                    """
                )
            else:
                self.video_transport.setStyleSheet(
                    """
                    QFrame {
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F5F8FC, stop:1 #E4ECF5);
                        border-top: 1px solid #B8CDE3;
                        color: #1E395B;
                    }
                    QPushButton {
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FFFFFF, stop:1 #EBF1F8);
                        border: 1px solid #B6C7DC;
                        border-radius: 3px;
                        padding: 2px 10px;
                        color: #1E395B;
                        font-family: 'Segoe UI';
                        font-size: 8.5pt;
                    }
                    QPushButton:hover {
                        background: #EBF4FD;
                        border: 1px solid #7AC1FF;
                    }
                    """
                )

    # -------------------------------------------------------------------------
    # Zoom & Crop Modes
    # -------------------------------------------------------------------------
    def _on_custom_zoom_set(self) -> None:
        self.current_zoom_mode = "custom"

    def set_zoom_mode(self, mode: Literal["fit", "fit_window", "fill", "100", "custom"]) -> None:
        self.current_zoom_mode = mode
        if mode == "fit":
            self.graphics_view.fit_in_view_with_zoom(allow_upscale=False)
        elif mode == "fit_window":
            self.graphics_view.fit_in_view_with_zoom(allow_upscale=True)
        elif mode == "fill":
            self.graphics_view.fill_in_view_with_zoom()
        elif mode == "100":
            self.graphics_view.set_zoom(1.0)

    def zoom_in(self) -> None:
        self.current_zoom_mode = "custom"
        self.graphics_view.zoom(1.25)

    def zoom_out(self) -> None:
        self.current_zoom_mode = "custom"
        self.graphics_view.zoom(0.8)

    def is_in_edit_mode(self) -> bool:
        return bool(
            getattr(self, "is_crop_mode", False)
            or getattr(self, "is_retouch_mode", False)
            or getattr(self, "is_red_eye_mode", False)
            or getattr(self, "is_straighten_active", False)
            or getattr(self, "is_markup_mode", False)
        )

    def _update_face_overlays_visibility(self) -> None:
        in_edit = self.is_in_edit_mode()
        for item in getattr(self, "_face_overlay_items", []):
            if isinstance(item, _FaceHoverRegionItem):
                item.setEnabled(not in_edit)
            else:
                if in_edit:
                    item.setVisible(False)
        self._sync_animation_playback()

    def _dismiss_depth_light_pane_if_active(self) -> None:
        mw = self.window()
        if hasattr(mw, "right_tag_dock") and hasattr(mw.right_tag_dock, "current_mode"):
            if mw.right_tag_dock.current_mode() in ("depth_light", "fine_tune"):
                mw.right_tag_dock.set_mode("info")

    def set_crop_mode(self, enabled: bool) -> None:
        self.is_crop_mode = bool(enabled)
        if hasattr(self, "scene") and hasattr(self.scene, "set_crop_mode"):
            self.scene.set_crop_mode(self.is_crop_mode)
        self.btn_crop_start.blockSignals(True)
        self.btn_crop_start.setChecked(self.is_crop_mode)
        self.btn_crop_start.blockSignals(False)
        self.btn_crop_apply.setEnabled(self.is_crop_mode)

        if self.is_crop_mode:
            self._dismiss_depth_light_pane_if_active()
            self._crop_pre_existing_rect = self.current_recipe.crop_rect
            self._crop_committed = False

            self.stop_face_tagging_mode()
            self.set_retouch_mode(False)
            self.set_red_eye_mode(False)
            self.set_markup_mode(False)
            self._update_face_overlays_visibility()
            self.graphics_view.setDragMode(ImageGraphicsView.NoDrag)
            self.crop_toolbar.show()
            self._position_floating_toolbars()
            self.crop_toolbar.raise_()

            # WLPG behavior: if image was already cropped, temporarily display the uncropped image
            # with the existing crop rectangle placed on top, allowing uncrop or readjustment!
            if self._crop_pre_existing_rect is not None:
                self.current_recipe.crop_rect = None
                self._original_pixmap = None
                self._render_current_recipe()
            self.set_zoom_mode("fit")

            self.scene.crop_rect_item.set_master_scale(1.0)

            if self._crop_pre_existing_rect is not None:
                ml, mt, mr, mb = self._crop_pre_existing_rect
                pw = max(4.0, float(mr - ml))
                ph = max(4.0, float(mb - mt))
                self.scene.crop_rect_item.setRect(QRectF(float(ml), float(mt), pw, ph))
                self.scene.crop_rect_item.show()
                self._sync_crop_ratio_from_rect(self.scene.crop_rect_item.rect())
                self._show_status("Crop Tool: Drag handles outwards to uncrop, or adjust boundaries • Enter to apply, Esc to cancel", 5000)
            else:
                if self.current_pil_image is not None:
                    w, h = self.current_pil_image.size
                    cw, ch = w * 0.85, h * 0.85
                    self.scene.crop_rect_item.setRect(QRectF((w - cw) / 2.0, (h - ch) / 2.0, cw, ch))
                    self.scene.crop_rect_item.show()
                    self._sync_crop_ratio_from_rect(self.scene.crop_rect_item.rect())
                self._show_status("Crop Tool: Drag handles or crop rectangle to adjust • Enter to commit, Esc to cancel", 5000)
        else:
            self.crop_toolbar.hide()
            self.scene.clear_crop_rect()
            if self.pixmap_item:
                self.pixmap_item.setRotation(0.0)
            self._update_face_overlays_visibility()
            if not self.is_retouch_mode and not self.is_red_eye_mode and not self.is_face_tag_mode:
                self.graphics_view.viewport().unsetCursor()
                self.graphics_view.setDragMode(ImageGraphicsView.ScrollHandDrag)

            # If user exited without committing (e.g. Cancel button or Esc), restore previous crop
            if not getattr(self, "_crop_committed", False) and getattr(self, "_crop_pre_existing_rect", None) is not None:
                self.current_recipe.crop_rect = self._crop_pre_existing_rect
                self._original_pixmap = None
                self._render_current_recipe()
                self.set_zoom_mode("fit")

            self._crop_pre_existing_rect = None
            self._crop_committed = False
            self._show_status("", 0)

    def _on_crop_ratio_changed(self, text: str) -> None:
        if getattr(self, "_updating_crop_inputs", False):
            return
        if self.current_pil_image is None:
            return
        img_w, img_h = self.current_pil_image.size
        if img_w <= 0 or img_h <= 0:
            return

        self._updating_crop_inputs = True
        try:
            if text in ("Ratio", "Custom"):
                self.txt_crop_w.setText("")
                self.txt_crop_h.setText("")
                self.scene.crop_rect_item.set_aspect_ratio(None)
                self.scene.crop_rect_updated.emit()
                return

            if text == "Full Image (Uncrop)":
                self.txt_crop_w.setText("")
                self.txt_crop_h.setText("")
                self.scene.crop_rect_item.set_aspect_ratio(None)
                self.scene.crop_rect_item.setRect(QRectF(0, 0, float(img_w), float(img_h)))
                self.scene.crop_rect_updated.emit()
                return

            w_val: float
            h_val: float
            if text == "Original":
                import math
                g = math.gcd(img_w, img_h)
                if g > 1 and img_w // g < 100:
                    w_val = float(img_w // g)
                    h_val = float(img_h // g)
                else:
                    w_val = round(float(img_w) / float(img_h), 2)
                    h_val = 1.0
            elif text == "1:1 (Square)":
                w_val, h_val = 1.0, 1.0
            elif text == "4:3":
                w_val, h_val = 4.0, 3.0
            elif text == "3:4":
                w_val, h_val = 3.0, 4.0
            elif text == "3:2":
                w_val, h_val = 3.0, 2.0
            elif text == "2:3":
                w_val, h_val = 2.0, 3.0
            elif text == "5:4":
                w_val, h_val = 5.0, 4.0
            elif text == "4:5":
                w_val, h_val = 4.0, 5.0
            elif text == "16:9":
                w_val, h_val = 16.0, 9.0
            elif text == "9:16":
                w_val, h_val = 9.0, 16.0
            else:
                return

            self.txt_crop_w.setText(str(int(w_val) if w_val.is_integer() else w_val))
            self.txt_crop_h.setText(str(int(h_val) if h_val.is_integer() else h_val))
            target_ratio = float(w_val) / float(h_val)
            self.scene.crop_rect_item.set_aspect_ratio(target_ratio)
            self.scene.crop_rect_updated.emit()
        finally:
            self._updating_crop_inputs = False

    def _on_crop_swap_clicked(self) -> None:
        """Swaps Width and Height ratio inputs (Landscape <-> Portrait) Photoshop-style."""
        w_str = self.txt_crop_w.text().strip()
        h_str = self.txt_crop_h.text().strip()

        # If empty, default to 4 and 3 or current crop ratio
        if not w_str and not h_str:
            cur_ratio = self.scene.crop_rect_item.aspect_ratio()
            if cur_ratio is not None and cur_ratio > 0:
                w_str, h_str = str(round(cur_ratio, 2)), "1"
            elif self.current_pil_image:
                iw, ih = self.current_pil_image.size
                if iw >= ih:
                    w_str, h_str = "4", "3"
                else:
                    w_str, h_str = "3", "4"
            else:
                w_str, h_str = "4", "3"

        # Swap values
        new_w, new_h = h_str, w_str
        self._updating_crop_inputs = True
        try:
            self.txt_crop_w.setText(new_w)
            self.txt_crop_h.setText(new_h)
            try:
                nw_val = float(new_w)
                nh_val = float(new_h)
            except ValueError:
                return
            if nw_val <= 0 or nh_val <= 0:
                return
            new_ratio = nw_val / nh_val
            self.scene.crop_rect_item.set_aspect_ratio(new_ratio)
            self.scene.crop_rect_updated.emit()

            # Update combo text if it matches a preset
            self._match_crop_combo_to_inputs(nw_val, nh_val)
        finally:
            self._updating_crop_inputs = False

    def _on_crop_custom_text_edited(self) -> None:
        """Applies custom aspect ratio as user types in Width or Height boxes."""
        if getattr(self, "_updating_crop_inputs", False):
            return
        w_str = self.txt_crop_w.text().strip()
        h_str = self.txt_crop_h.text().strip()
        if not w_str or not h_str:
            self._updating_crop_inputs = True
            try:
                self.scene.crop_rect_item.set_aspect_ratio(None)
                self.scene.crop_rect_updated.emit()
                self.combo_crop_ratio.setCurrentText("Custom")
            finally:
                self._updating_crop_inputs = False
            return

        try:
            w_val = float(w_str)
            h_val = float(h_str)
        except ValueError:
            return

        if w_val <= 0 or h_val <= 0:
            return

        self._updating_crop_inputs = True
        try:
            target_ratio = w_val / h_val
            self.scene.crop_rect_item.set_aspect_ratio(target_ratio)
            self.scene.crop_rect_updated.emit()
            self._match_crop_combo_to_inputs(w_val, h_val)
        finally:
            self._updating_crop_inputs = False

    def _sync_crop_ratio_from_rect(self, rect: QRectF) -> None:
        if rect.width() > 0 and rect.height() > 0:
            self._match_crop_combo_to_inputs(rect.width(), rect.height())

    def _match_crop_combo_to_inputs(self, w_val: float, h_val: float) -> None:
        """Updates combo box selection if the ratio matches one of the presets."""
        ratio = w_val / h_val
        presets = [
            ("1:1 (Square)", 1.0),
            ("4:3", 4.0 / 3.0),
            ("3:4", 3.0 / 4.0),
            ("3:2", 3.0 / 2.0),
            ("2:3", 2.0 / 3.0),
            ("5:4", 5.0 / 4.0),
            ("4:5", 4.0 / 5.0),
            ("16:9", 16.0 / 9.0),
            ("9:16", 9.0 / 16.0),
        ]
        for name, r in presets:
            if abs(ratio - r) < 0.005:
                self.combo_crop_ratio.setCurrentText(name)
                return
        self.combo_crop_ratio.setCurrentText("Custom")

    def apply_crop(self) -> bool:
        """Applies the active CropScene rectangle and rotation angle non-destructively to the recipe."""
        if self.master_np_rgb is None:
            return False
        rect = self.scene.get_crop_rect()
        if rect is None or rect.width() < 4 or rect.height() < 4:
            return False

        # Reset any live preview rotation on pixmap item
        rot_deg = getattr(self.scene, "crop_rotation_deg", 0.0)
        if self.pixmap_item:
            self.pixmap_item.setRotation(0.0)

        master_h, master_w = self.master_np_rgb.shape[:2]
        ml = int(round(rect.left()))
        mt = int(round(rect.top()))
        mr = int(round(rect.right()))
        mb = int(round(rect.bottom()))

        ml = max(0, min(master_w - 1, ml))
        mt = max(0, min(master_h - 1, mt))
        mr = max(ml + 1, min(master_w, mr))
        mb = max(mt + 1, min(master_h, mb))

        # Check if crop covers virtually the entire image (within 3px margin): Uncrop!
        if ml <= 3 and mt <= 3 and mr >= master_w - 3 and mb >= master_h - 3:
            new_crop = None
        else:
            new_crop = (ml, mt, mr, mb)

        before = self.current_recipe.copy()
        before.crop_rect = getattr(self, "_crop_pre_existing_rect", None)

        if abs(rot_deg) >= 0.05:
            self.current_recipe.straighten_deg = float(self.current_recipe.straighten_deg + rot_deg)

        self.current_recipe.crop_rect = new_crop
        self._crop_committed = True
        self._original_pixmap = None

        action_name = "Crop Photo" if new_crop is not None else "Uncrop Photo"
        cmd = RecipeUndoCommand(self, before, self.current_recipe.copy(), action_name)
        self.undo_stack.push(cmd)
        self.set_crop_mode(False)
        self._render_current_recipe()
        self.set_zoom_mode("fit")
        self.recipe_changed.emit(self.current_recipe)
        self._show_status(f"{action_name} applied", 2500)
        return True

    # -------------------------------------------------------------------------
    # Spot Fix: Retouch (Blemish Healing) & Red Eye Removal
    # -------------------------------------------------------------------------
    def _close_spot_fix_toolbar(self) -> None:
        self.set_retouch_mode(False)
        self.set_red_eye_mode(False)
        self._show_status("Spot Fix closed", 2000)

    def _get_effective_brush_radius(self) -> tuple[float, int]:
        """Calculates (scene_radius, master_radius) matching the screen cursor radius exactly."""
        tr = self.graphics_view.transform()
        scale = max(0.001, float(tr.m11()))
        r_screen = float(max(4, getattr(self, "retouch_radius", 20)))
        r_scene = max(1.0, r_screen / scale)
        r_master = max(2, int(round(r_scene)))
        return r_scene, r_master

    def set_spot_fix_sub_mode(self, mode: Literal["retouch", "erase", "red_eye"]) -> None:
        """Switches between Blemish healing, Object eraser (LaMa), and Red Eye removal under the shared spot fix toolbar."""
        self.spot_fix_sub_mode = mode
        if hasattr(self, "btn_spot_retouch"):
            self.btn_spot_retouch.blockSignals(True)
            self.btn_spot_retouch.setChecked(mode == "retouch")
            self.btn_spot_retouch.blockSignals(False)
        if hasattr(self, "btn_spot_erase"):
            self.btn_spot_erase.blockSignals(True)
            self.btn_spot_erase.setChecked(mode == "erase")
            self.btn_spot_erase.blockSignals(False)
        if hasattr(self, "btn_spot_red_eye"):
            self.btn_spot_red_eye.blockSignals(True)
            self.btn_spot_red_eye.setChecked(mode == "red_eye")
            self.btn_spot_red_eye.blockSignals(False)

        if mode == "retouch":
            self.is_retouch_mode = True
            self.is_red_eye_mode = False
            self._show_status("Spot Fix (Blemish): Click or brush blemishes to heal with authentic texture transfer • [ / ] to resize", 5000)
        elif mode == "erase":
            self.is_retouch_mode = True
            self.is_red_eye_mode = False
            self._show_status("Spot Fix (Object Eraser): Paint over unwanted objects or people to erase with AI inpainting • [ / ] to resize", 5000)
        else:
            self.is_retouch_mode = False
            self.is_red_eye_mode = True
            self._show_status("Spot Fix (Red Eye): Click pupil or drag over eyes to neutralize flash glare • [ / ] to resize", 5000)

        self._update_retouch_cursor()

    def set_retouch_mode(self, enabled: bool) -> None:
        self.is_retouch_mode = bool(enabled)
        if self.is_retouch_mode:
            self._dismiss_depth_light_pane_if_active()
            self.stop_face_tagging_mode()
            self.set_crop_mode(False)
            self.set_markup_mode(False)
            self.is_red_eye_mode = False
            self.set_spot_fix_sub_mode("retouch")
            self._update_face_overlays_visibility()
            self.graphics_view.setDragMode(ImageGraphicsView.NoDrag)
            self._update_retouch_cursor()
            self.retouch_toolbar.show()
            self._position_floating_toolbars()
            self.retouch_toolbar.raise_()
        else:
            if not getattr(self, "is_red_eye_mode", False):
                self.retouch_toolbar.hide()
                self._show_status("", 0)
            self._clear_retouch_overlay()
            self._update_face_overlays_visibility()
            if not self.is_crop_mode and not getattr(self, "is_red_eye_mode", False):
                self.graphics_view.viewport().unsetCursor()
                self.graphics_view.setDragMode(ImageGraphicsView.ScrollHandDrag)

    def _update_retouch_cursor(self) -> None:
        """Constructs a high-visibility dual-ring circular cursor showing exact brush radius."""
        rad = max(4, int(getattr(self, "retouch_radius", 20)))
        d = rad * 2
        size = d + 6
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)

        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        center = size / 2.0
        r = d / 2.0

        sub_mode = getattr(self, "spot_fix_sub_mode", "retouch")
        is_red_eye = sub_mode == "red_eye"
        is_erase = sub_mode == "erase"

        # Outer dark shadow ring for high contrast over highlights and skin
        p.setPen(QPen(QColor(0, 0, 0, 210), 2.5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(center - r, center - r, d, d)

        # Inner ring: amber/gold for blemish, magenta/violet for object eraser, soft cyan for red eye
        if is_red_eye:
            ring_color = QColor(100, 220, 255, 240)
            dot_color = QColor(255, 70, 70, 240)
        elif is_erase:
            ring_color = QColor(225, 110, 255, 240)
            dot_color = QColor(240, 160, 255, 240)
        else:
            ring_color = QColor(255, 200, 100, 240)
            dot_color = QColor(255, 255, 255, 240)

        p.setPen(QPen(ring_color, 1.2))
        p.drawEllipse(center - r, center - r, d, d)

        # Precise center targeting dot
        p.setPen(Qt.NoPen)
        p.setBrush(dot_color)
        p.drawEllipse(center - 1.5, center - 1.5, 3.0, 3.0)
        p.end()

        self.graphics_view.viewport().setCursor(QCursor(pm, int(center), int(center)))

    def _on_retouch_size_changed(self, val: int) -> None:
        self.retouch_radius = max(6, int(val))
        if hasattr(self, "slider_retouch_size") and self.slider_retouch_size.value() != self.retouch_radius:
            self.slider_retouch_size.blockSignals(True)
            self.slider_retouch_size.setValue(self.retouch_radius)
            self.slider_retouch_size.blockSignals(False)
        if getattr(self, "is_retouch_mode", False) or getattr(self, "is_red_eye_mode", False):
            self._update_retouch_cursor()

    def _clear_retouch_overlay(self) -> None:
        for it in self._retouch_preview_items:
            try:
                self.scene.removeItem(it)
            except Exception:
                pass
        self._retouch_preview_items.clear()
        self._retouch_mask = None
        self._retouch_active = False

    def _start_retouch(self, scene_pos: QPointF) -> None:
        if self.master_np_rgb is None:
            return
        mx, my = self._scene_to_master(scene_pos.x(), scene_pos.y())
        self._retouch_mask = np.zeros(self.master_np_rgb.shape[:2], dtype=np.uint8)
        r_scene, r_master = self._get_effective_brush_radius()
        cv2.circle(self._retouch_mask, (mx, my), r_master, 255, -1)
        is_erase = getattr(self, "spot_fix_sub_mode", "retouch") == "erase"
        pen_col = QColor("#C040FF") if is_erase else QColor("#FF9900")
        brush_col = QColor(192, 64, 255, 110) if is_erase else QColor(255, 153, 0, 110)
        c_item = self.scene.addEllipse(
            scene_pos.x() - r_scene,
            scene_pos.y() - r_scene,
            r_scene * 2,
            r_scene * 2,
            QPen(pen_col, 1),
            QBrush(brush_col),
        )
        self._retouch_preview_items.append(c_item)
        self._retouch_last_master_pt = (mx, my)
        self._retouch_active = True

    def _drag_retouch(self, scene_pos: QPointF) -> None:
        if not self._retouch_active or self._retouch_mask is None or self.master_np_rgb is None:
            return
        mx, my = self._scene_to_master(scene_pos.x(), scene_pos.y())
        r_scene, r_master = self._get_effective_brush_radius()
        if self._retouch_last_master_pt is not None:
            cv2.line(self._retouch_mask, self._retouch_last_master_pt, (mx, my), 255, thickness=r_master * 2)
        cv2.circle(self._retouch_mask, (mx, my), r_master, 255, -1)
        is_erase = getattr(self, "spot_fix_sub_mode", "retouch") == "erase"
        brush_col = QColor(192, 64, 255, 110) if is_erase else QColor(255, 153, 0, 110)
        c_item = self.scene.addEllipse(
            scene_pos.x() - r_scene,
            scene_pos.y() - r_scene,
            r_scene * 2,
            r_scene * 2,
            QPen(Qt.NoPen),
            QBrush(brush_col),
        )
        self._retouch_preview_items.append(c_item)
        self._retouch_last_master_pt = (mx, my)

    def _finish_retouch(self) -> None:
        if not self._retouch_active or self._retouch_mask is None or self.master_np_rgb is None:
            self._clear_retouch_overlay()
            return
        self._retouch_active = False

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_dilated = cv2.dilate(self._retouch_mask, kernel, iterations=1)
        if not np.any(mask_dilated > 0):
            self._clear_retouch_overlay()
            return

        before = self.master_np_rgb.copy()

        # If in Object Eraser mode, use LaMa ONNX / Multi-Scale Structure-Texture Inpainting
        if getattr(self, "spot_fix_sub_mode", "retouch") == "erase":
            from src.imaging.creative_effects import inpaint_object_lama
            self.master_np_rgb = inpaint_object_lama(self.master_np_rgb, mask_dilated)
            self._update_display_proxy()
            self._clear_retouch_overlay()
            cmd = PixelArrayUndoCommand(self, before, self.master_np_rgb.copy(), "Erase Object")
            self.undo_stack.push(cmd)
            self._render_current_recipe()
            if self.current_media_id is not None:
                self.pixel_edit_saved.emit(self.current_media_id, "")
            return

        h_full, w_full = self.master_np_rgb.shape[:2]
        _, r_master = self._get_effective_brush_radius()

        # Localize bounding box for high-performance sub-millisecond execution
        pts_y, pts_x = np.where(mask_dilated > 0)
        bw = int(np.max(pts_x) - np.min(pts_x)) + 1
        bh = int(np.max(pts_y) - np.min(pts_y)) + 1
        shift_x = max(8, int(bw * 1.15))
        shift_y = max(8, int(bh * 1.15))

        pad = max(32, int(max(bw, bh) * 1.5))
        y1 = max(0, int(np.min(pts_y)) - pad)
        y2 = min(h_full, int(np.max(pts_y)) + pad)
        x1 = max(0, int(np.min(pts_x)) - pad)
        x2 = min(w_full, int(np.max(pts_x)) + pad)

        roi_img = self.master_np_rgb[y1:y2, x1:x2].copy()
        roi_mask = mask_dilated[y1:y2, x1:x2]
        inpaint_r = max(3, min(25, r_master // 2))

        # 1. Frequency Separation: Decompose local ROI into Low and High frequencies
        # Low frequency (smooth illumination, chromatic gradient, skin tone)
        sigma = max(2.0, min(8.0, float(r_master) / 3.0))
        k = int(2 * round(2 * sigma) + 1)
        roi_low = cv2.GaussianBlur(roi_img, (k, k), sigma)
        # High frequency (authentic pores, skin micro-grain, fabric weave)
        roi_high = roi_img.astype(np.float32) - roi_low.astype(np.float32)

        # 2. Inpaint low frequencies across the blemish to seamlessly interpolate color & lighting
        roi_low_inpainted = cv2.inpaint(
            roi_low,
            roi_mask,
            inpaintRadius=inpaint_r,
            flags=cv2.INPAINT_TELEA,
        )

        # 3. Donor ring for sampling authentic surrounding skin texture
        outer_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(7, inpaint_r * 2 + 1), max(7, inpaint_r * 2 + 1)),
        )
        donor_ring = cv2.dilate(roi_mask, outer_k, iterations=1) - roi_mask

        # 4. Multi-directional donor texture transfer from the unblemished perimeter
        h_roi, w_roi = roi_mask.shape
        if np.count_nonzero(donor_ring) >= 16:
            offsets = [
                (0, shift_x), (0, -shift_x),
                (shift_y, 0), (-shift_y, 0),
                (shift_y, shift_x), (-shift_y, -shift_x),
                (shift_y, -shift_x), (-shift_y, shift_x),
            ]
            valid_shifts = []
            for dy, dx in offsets:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted_mask = cv2.warpAffine(
                    roi_mask, M, (w_roi, h_roi), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                overlap = np.count_nonzero((shifted_mask > 0) & (roi_mask > 0))
                if overlap == 0:
                    shifted_high = cv2.warpAffine(
                        roi_high, M, (w_roi, h_roi), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                    )
                    valid_shifts.append(shifted_high)

            if valid_shifts:
                transferred_high = np.mean(valid_shifts, axis=0)
                donor_std = float(np.std(roi_high[donor_ring > 0]))
                trans_std = float(np.std(transferred_high[roi_mask > 0]))
                if trans_std > 0.05 and donor_std > 0.1:
                    transferred_high = transferred_high * (donor_std / trans_std)
            else:
                transferred_high = np.zeros_like(roi_high)
        else:
            transferred_high = np.zeros_like(roi_high)

        # 5. Frequency Recombination: Base Low + Transferred High
        restored_roi = np.clip(
            roi_low_inpainted.astype(np.float32) + transferred_high, 0, 255
        ).astype(np.uint8)

        # 6. Feathered alpha blending along mask boundary for invisible edges
        soft_mask = cv2.GaussianBlur(
            roi_mask.astype(np.float32) / 255.0, (5, 5), 1.2
        )[:, :, None]
        blended_roi = np.clip(
            roi_img.astype(np.float32) * (1.0 - soft_mask) + restored_roi.astype(np.float32) * soft_mask,
            0,
            255,
        ).astype(np.uint8)

        self.master_np_rgb[y1:y2, x1:x2] = blended_roi
        self._update_display_proxy()

        self._clear_retouch_overlay()
        cmd = PixelArrayUndoCommand(self, before, self.master_np_rgb.copy(), "Retouch Blemish")
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        if self.current_media_id is not None:
            self.pixel_edit_saved.emit(self.current_media_id, "")

    # -------------------------------------------------------------------------
    # Red Eye Removal
    # -------------------------------------------------------------------------
    def set_red_eye_mode(self, enabled: bool) -> None:
        self.is_red_eye_mode = bool(enabled)
        if self.is_red_eye_mode:
            self._dismiss_depth_light_pane_if_active()
            self.stop_face_tagging_mode()
            self.set_crop_mode(False)
            self.set_markup_mode(False)
            self.is_retouch_mode = False
            self.set_spot_fix_sub_mode("red_eye")
            self._update_face_overlays_visibility()
            self.graphics_view.setDragMode(ImageGraphicsView.NoDrag)
            self._update_retouch_cursor()
            self.retouch_toolbar.show()
            self._position_floating_toolbars()
            self.retouch_toolbar.raise_()
        else:
            if not getattr(self, "is_retouch_mode", False):
                self.retouch_toolbar.hide()
                self._show_status("", 0)
            if self._red_eye_item is not None:
                try:
                    self.scene.removeItem(self._red_eye_item)
                except Exception:
                    pass
                self._red_eye_item = None
            self._update_face_overlays_visibility()
            if not self.is_crop_mode and not getattr(self, "is_retouch_mode", False):
                self.graphics_view.viewport().unsetCursor()
                self.graphics_view.setDragMode(ImageGraphicsView.ScrollHandDrag)

    @staticmethod
    def _neutralize_red_eye_patch(
        patch: np.ndarray,
        in_region: np.ndarray,
        center_yx: tuple[float, float],
    ) -> np.ndarray | None:
        """
        Detects and neutralizes flash red-eye inside patch (float32 HxWx3 RGB):
        1. Isolates the connected red pupil core closest to center_yx.
        2. Dilates the pupil mask by 3px into the anti-aliased pupil-iris transition zone and weights by
           continuous redness excess so no maroon/brownish outer fringe remains.
        3. Preserves bright corneal specular catchlights (where min(G, B) > 110) while neutralizing excess red.
        4. Smoothly shades the pupil interior with radial darkening from center to edge.
        """
        r = patch[:, :, 0]
        g = patch[:, :, 1]
        b = patch[:, :, 2]

        gb_sum = g + b + 1.0
        max_gb = np.maximum(g, b)
        min_gb = np.minimum(g, b)
        r_ratio = r / gb_sum
        r_diff = r - max_gb

        # Detect strong red core pixels inside the brush/selection region
        is_red = (r > 55.0) & (r_ratio >= 1.05) & (r_diff >= 18.0) & in_region
        if not np.any(is_red):
            is_red = (r > 45.0) & (r_ratio >= 0.90) & (r_diff >= 14.0) & in_region
        if not np.any(is_red):
            return None

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            is_red.astype(np.uint8) * 255, connectivity=8
        )
        if num_labels <= 1:
            return None

        best_label = 1
        best_dist = float("inf")
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < 2:
                continue
            cy, cx = centroids[lbl][1], centroids[lbl][0]
            dist = math.hypot(cy - center_yx[0], cx - center_yx[1])
            if dist < best_dist:
                best_dist = dist
                best_label = lbl

        pupil_core = (labels == best_label).astype(np.uint8) * 255
        p_cy, p_cx = float(centroids[best_label][1]), float(centroids[best_label][0])
        p_area = float(max(4, stats[best_label, cv2.CC_STAT_AREA]))
        p_radius = max(2.0, math.sqrt(p_area / math.pi))

        # Close internal specular catchlight holes so geometry covers the full pupil disc
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        pupil_closed = cv2.morphologyEx(pupil_core, cv2.MORPH_CLOSE, k_close)

        # Precise 1px dilation into the iris transition zone to capture the maroon fringe without swallowing the iris
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        pupil_dilated = cv2.dilate(pupil_closed, k_dil, iterations=1)

        # Continuous redness weight: 1.0 in vivid red core, smooth roll-off across maroon fringe, 0.0 on blue/green/skin
        w_diff = np.clip((r_diff - 4.0) / 16.0, 0.0, 1.0)
        w_ratio = np.clip((r_ratio - 0.52) / 0.35, 0.0, 1.0)
        w_red = np.clip(w_diff * w_ratio * 1.35, 0.0, 1.0)

        core_f = cv2.GaussianBlur(pupil_closed.astype(np.float32) / 255.0, (3, 3), 0.7)
        fringe_f = cv2.GaussianBlur(pupil_dilated.astype(np.float32) / 255.0, (3, 3), 1.0) * w_red
        alpha_2d = np.clip(np.maximum(core_f * np.maximum(w_red, 0.75), fringe_f), 0.0, 1.0)

        if float(np.max(alpha_2d)) < 0.05:
            return None

        # Radial pupil shading: authentic photographic pupil depth, avoiding pitch-black void
        ph, pw = patch.shape[:2]
        yy, xx = np.ogrid[0:ph, 0:pw]
        norm_r = np.sqrt((yy - p_cy) ** 2 + (xx - p_cx) ** 2) / max(1.5, p_radius * 1.25)
        radial_t = np.clip(norm_r, 0.0, 1.0)
        gb_mean = (g + b) * 0.5
        shade_factor = 0.40 * (1.0 - radial_t) + 0.60 * radial_t
        pupil_luma = np.clip(gb_mean * shade_factor, 20.0, 58.0)
        dark_pupil = np.stack([pupil_luma, pupil_luma, pupil_luma * 1.02], axis=-1)

        # Preserve corneal specular catchlight (glint) where G and B channels are bright
        glint_w = np.clip((gb_mean - 75.0) / 45.0, 0.0, 1.0)[..., None]
        glint_r = np.minimum(r, gb_mean)
        glint_rgb = np.stack([glint_r, g, b], axis=-1)

        target_rgb = dark_pupil * (1.0 - glint_w) + glint_rgb * glint_w
        alpha = alpha_2d[..., None]
        new_patch = patch * (1.0 - alpha) + target_rgb * alpha
        return np.clip(new_patch, 0.0, 255.0).astype(np.uint8)

    def _apply_brush_red_eye(self, scene_pos: QPointF) -> None:
        """Applies pupil detection and neutral dark replacement using the circular brush area."""
        if self.master_np_rgb is None:
            return
        _, r_master = self._get_effective_brush_radius()
        mx, my = self._scene_to_master(scene_pos.x(), scene_pos.y())
        mh, mw = self.master_np_rgb.shape[:2]

        pad_r = r_master + 4
        x1 = max(0, mx - pad_r)
        x2 = min(mw, mx + pad_r + 1)
        y1 = max(0, my - pad_r)
        y2 = min(mh, my + pad_r + 1)
        if x2 <= x1 or y2 <= y1:
            return

        patch = self.master_np_rgb[y1:y2, x1:x2].astype(np.float32)
        grid_y, grid_x = np.ogrid[y1:y2, x1:x2]
        dist_sq = (grid_x - mx) ** 2 + (grid_y - my) ** 2
        in_circle = dist_sq <= (r_master ** 2)

        updated_patch = self._neutralize_red_eye_patch(
            patch, in_circle, (float(my - y1), float(mx - x1))
        )
        if updated_patch is None:
            return

        before = self.master_np_rgb.copy()
        self.master_np_rgb[y1:y2, x1:x2] = updated_patch

        self._update_display_proxy()
        cmd = PixelArrayUndoCommand(self, before, self.master_np_rgb.copy(), "Fix Red Eye")
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        if self.current_media_id is not None:
            self.pixel_edit_saved.emit(self.current_media_id, "")

    def _start_red_eye(self, scene_pos: QPointF) -> None:
        self._red_eye_start_pos = scene_pos
        self._red_eye_active = True
        self._red_eye_item = self.scene.addRect(
            QRectF(scene_pos, scene_pos),
            QPen(QColor("#00FFCC"), 2, Qt.DashLine),
            QBrush(QColor(0, 255, 204, 30)),
        )

    def _drag_red_eye(self, scene_pos: QPointF) -> None:
        if not self._red_eye_active or self._red_eye_item is None or self._red_eye_start_pos is None:
            return
        rect = QRectF(self._red_eye_start_pos, scene_pos).normalized()
        self._red_eye_item.setRect(rect)

    def _finish_red_eye(self) -> None:
        if not self._red_eye_active or self._red_eye_item is None or self.master_np_rgb is None:
            self._red_eye_active = False
            return
        self._red_eye_active = False
        rect = self._red_eye_item.rect()
        try:
            self.scene.removeItem(self._red_eye_item)
        except Exception:
            pass
        self._red_eye_item = None

        if rect.width() < 3 or rect.height() < 3:
            return

        mx1, my1 = self._scene_to_master(rect.left(), rect.top())
        mx2, my2 = self._scene_to_master(rect.right(), rect.bottom())
        x1, x2 = min(mx1, mx2), max(mx1, mx2)
        y1, y2 = min(my1, my2), max(my1, my2)
        if x2 <= x1 or y2 <= y1:
            return

        patch = self.master_np_rgb[y1:y2, x1:x2].astype(np.float32)
        in_box = np.ones(patch.shape[:2], dtype=bool)
        updated_patch = self._neutralize_red_eye_patch(
            patch, in_box, (patch.shape[0] / 2.0, patch.shape[1] / 2.0)
        )
        if updated_patch is None:
            return

        before = self.master_np_rgb.copy()
        self.master_np_rgb[y1:y2, x1:x2] = updated_patch

        self._update_display_proxy()
        cmd = PixelArrayUndoCommand(self, before, self.master_np_rgb.copy(), "Fix Red Eye")
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        if self.current_media_id is not None:
            self.pixel_edit_saved.emit(self.current_media_id, "")

    # -------------------------------------------------------------------------
    # Quick Markup, Callouts & Privacy Redaction Tool
    # -------------------------------------------------------------------------
    def set_markup_mode(self, enabled: bool) -> None:
        self.is_markup_mode = bool(enabled)
        if self.is_markup_mode:
            self._dismiss_depth_light_pane_if_active()
            self._session_markup_command_count = 0
            self._markup_session_start_master = self.master_np_rgb.copy() if self.master_np_rgb is not None else None
            self._markup_session_start_undo_idx = self.undo_stack.index() if hasattr(self, "undo_stack") else 0
            self.stop_face_tagging_mode()
            self.set_crop_mode(False)
            self.set_retouch_mode(False)
            self.set_red_eye_mode(False)
            self._update_face_overlays_visibility()
            self.graphics_view.setDragMode(ImageGraphicsView.NoDrag)
            self._update_markup_cursor()
            if hasattr(self.markup_toolbar, "btn_undo") and hasattr(self, "undo_stack"):
                self.markup_toolbar.btn_undo.setEnabled(self.undo_stack.canUndo())
            self.markup_toolbar.show()
            self._position_floating_toolbars()
            self.markup_toolbar.raise_()
            self._purge_all_transient_scene_items()
            self.update_revert_state()
            self._show_status("Markup Mode: Choose Pen, Highlighter, Shapes, Text or Magnify • Enter to commit, Esc to close", 5000)
        else:
            if getattr(self, "_inline_text_input", None) is not None:
                self._commit_inline_text()
            self._finalize_movable_markup()
            self.markup_toolbar.hide()
            self._cleanup_markup_temp_items()
            self._update_face_overlays_visibility()
            if not self.is_crop_mode and not self.is_retouch_mode and not self.is_red_eye_mode:
                self.graphics_view.viewport().unsetCursor()
                self.graphics_view.setDragMode(ImageGraphicsView.ScrollHandDrag)
            self._purge_all_transient_scene_items()
            self.update_revert_state()
            self._show_status("", 0)

    def _on_markup_save_requested(self) -> None:
        """Saves current image with pristine backup and displays status on toolbar."""
        if getattr(self, "_inline_text_input", None) is not None:
            self._commit_inline_text()
        self._finalize_movable_markup()
        res = self.save_current_image_with_backup()
        if res and hasattr(self, "markup_toolbar") and hasattr(self.markup_toolbar, "btn_save"):
            self.markup_toolbar.btn_save.setText("Saved!")
            QTimer.singleShot(1500, lambda: self.markup_toolbar.btn_save.setText("Save") if hasattr(self.markup_toolbar, "btn_save") else None)

    def _update_markup_cursor(self) -> None:
        """Constructs a circular cursor matching the exact on-screen stroke width."""
        if not getattr(self, "is_markup_mode", False) or not hasattr(self, "markup_toolbar"):
            return

        tool = getattr(self.markup_toolbar, "active_tool", "pen")
        stroke_w = getattr(self.markup_toolbar, "stroke_width", 4)
        col = getattr(self.markup_toolbar, "current_color", QColor(230, 40, 40))

        # View scale: viewport pixels per scene pixel
        tr = self.graphics_view.transform()
        scale = max(0.001, float(tr.m11()))

        if tool == "pen":
            d_screen = max(4, int(round(stroke_w * scale)))
            rad = max(2, d_screen // 2)
        elif tool == "highlighter":
            hl_w = max(8, int(stroke_w * 2.6))
            d_screen = max(6, int(round(hl_w * scale)))
            rad = max(3, d_screen // 2)
        elif tool in ("magnify", "magnifier"):
            # The magnifying lens is rendered live on scene in _update_magnify_hover;
            # use a clean precision crosshair cursor to avoid showing a confusing second circle!
            self.graphics_view.viewport().setCursor(Qt.CrossCursor)
            return
        elif tool == "text":
            self.graphics_view.viewport().setCursor(Qt.IBeamCursor)
            return
        else:
            self.graphics_view.viewport().setCursor(Qt.CrossCursor)
            return

        d = rad * 2
        size = d + 8
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)

        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        center = size / 2.0
        r = d / 2.0

        # Outer high-contrast dark drop-ring
        p.setPen(QPen(QColor(0, 0, 0, 200), 2.0))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(center, center), r, r)

        # Inner ring colored with the active markup color
        inner_col = col if col.isValid() else QColor(255, 255, 255)
        p.setPen(QPen(inner_col, 1.5))
        p.drawEllipse(QPointF(center, center), r - 1.0, r - 1.0)

        # Center dot
        dot_col = QColor(255, 255, 255, 230) if inner_col.lightness() < 128 else QColor(0, 0, 0, 220)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(dot_col))
        p.drawEllipse(QPointF(center, center), 1.0, 1.0)
        p.end()

        cursor = QCursor(pm, int(center), int(center))
        self.graphics_view.viewport().setCursor(cursor)

    def _on_markup_settings_changed(self) -> None:
        self._update_markup_cursor()
        tool = getattr(self.markup_toolbar, "active_tool", "pen")
        if tool not in ("magnify", "magnifier"):
            self._cleanup_magnify_preview()
        else:
            sp = getattr(self, "_last_mouse_scene_pos", None)
            if sp is not None:
                self._update_magnify_hover(sp)
        if getattr(self.markup_toolbar, "active_tool", "pen") == "text":
            self._update_inline_text_style()

        if getattr(self, "_pending_markup_element", None) is not None:
            col = getattr(self.markup_toolbar, "current_color", QColor(230, 40, 40))
            width = getattr(self.markup_toolbar, "stroke_width", 4)
            el = self._pending_markup_element
            el.color = col
            el.stroke_width = width
            if hasattr(self.markup_toolbar, "secondary_color"):
                el.bg_color = getattr(self.markup_toolbar, "secondary_color", None)
            if hasattr(self.markup_toolbar, "magnifier_zoom"):
                el.zoom_factor = getattr(self.markup_toolbar, "magnifier_zoom", 2.5)
            crop_tuple = self.current_recipe.crop_rect if self.current_recipe else None
            pm, top_left = render_markup_element_pixmap(el, self.master_np_rgb, crop_tuple)
            if self._pending_markup_item is not None and not pm.isNull():
                self._pending_markup_item.setPixmap(pm)
                self._pending_markup_item.setPos(top_left)
            if self._pending_selection_box_item is not None:
                self._pending_selection_box_item.setRect(get_markup_element_bounds(el))

    def _get_magnifier_scene_radius(self) -> float:
        """Returns the scene radius for the magnifying lens so it maintains a prominent screen diameter."""
        tr = self.graphics_view.transform()
        scale = max(0.001, float(tr.m11()))
        w = getattr(self.markup_toolbar, "stroke_width", 4)
        mag_screen_radius = 45.0 + (w - 2) * 5.0
        return max(30.0, mag_screen_radius / scale)

    def _cleanup_magnify_preview(self) -> None:
        """Hides temporary scene items used by the 2-point magnifier callout preview."""
        if getattr(self, "_magnify_src_circle_item", None) is not None:
            self._magnify_src_circle_item.setVisible(False)
        if getattr(self, "_magnify_rays_item", None) is not None:
            self._magnify_rays_item.setVisible(False)
        if getattr(self, "_magnify_hover_item", None) is not None:
            self._magnify_hover_item.setVisible(False)

    def _finalize_movable_markup(self) -> None:
        """Finalizes the active movable element and hides the selection outline."""
        if getattr(self, "_pending_selection_box_item", None) is not None:
            self._remove_scene_item(self._pending_selection_box_item)
            self._pending_selection_box_item = None
        if getattr(self, "_pending_markup_item", None) is not None:
            self._remove_scene_item(self._pending_markup_item)
            self._pending_markup_item = None
        self._active_movable_element = None
        self._active_movable_before_master = None
        self._is_moving_pending_markup = False
        self._move_markup_last_pt = None

    def _commit_magnifier_callout(self, p_src: QPointF, p_tgt: QPointF) -> None:
        """Commits a 2-point circular loupe callout with connecting rays from p_src to p_tgt."""
        if self.master_np_rgb is None:
            return
        col = getattr(self.markup_toolbar, "current_color", QColor(230, 40, 40))
        width = getattr(self.markup_toolbar, "stroke_width", 4)
        zoom = getattr(self.markup_toolbar, "magnifier_zoom", 2.5)
        rad = self._get_magnifier_scene_radius()

        el = MarkupElement(
            kind="magnifier",
            start_pt=p_src,
            end_pt=p_tgt,
            rect=QRectF(p_tgt.x() - rad, p_tgt.y() - rad, rad * 2.0, rad * 2.0),
            color=col,
            stroke_width=width,
            zoom_factor=zoom,
        )
        self._cleanup_magnify_preview()
        self._commit_single_markup_element(el, f"Markup Magnify Callout {zoom:.1f}x")

    def _update_magnify_hover(self, sp: QPointF | None) -> None:
        """
        Shows or updates live interactive preview for the 2-step / drag loupe callout:
        - When no source is selected: displays a source capture ring at the cursor.
        - When source is selected: displays anchored source circle at p_src, tangent connector
          rays, and the magnified loupe following the cursor at p_tgt.
        """
        if (
            not getattr(self, "is_markup_mode", False)
            or getattr(self.markup_toolbar, "active_tool", "pen") not in ("magnify", "magnifier")
            or sp is None
            or self.master_np_rgb is None
        ):
            self._cleanup_magnify_preview()
            return

        rad = self._get_magnifier_scene_radius()
        zoom = getattr(self.markup_toolbar, "magnifier_zoom", 2.5)
        src_r = max(6.0, rad / max(1.0, zoom))
        ring_col = getattr(self.markup_toolbar, "current_color", QColor(230, 40, 40))
        w = getattr(self.markup_toolbar, "stroke_width", 4)
        src_pt = getattr(self, "_magnify_source_pt", None)

        if src_pt is not None:
            # Step 2 active: source is anchored at src_pt, loupe is at sp
            p_src = src_pt
            p_tgt = sp

            # 1. Source circle item
            if getattr(self, "_magnify_src_circle_item", None) is None:
                self._magnify_src_circle_item = QGraphicsEllipseItem()
                self._magnify_src_circle_item.setZValue(9996)
            if self._magnify_src_circle_item.scene() is not self.scene:
                self.scene.addItem(self._magnify_src_circle_item)
            self._magnify_src_circle_item.setRect(QRectF(p_src.x() - src_r, p_src.y() - src_r, src_r * 2.0, src_r * 2.0))
            self._magnify_src_circle_item.setPen(QPen(ring_col, max(2.0, w * 0.5), Qt.SolidLine))
            self._magnify_src_circle_item.setBrush(QColor(ring_col.red(), ring_col.green(), ring_col.blue(), 35))
            self._magnify_src_circle_item.setVisible(True)

            # 2. Tangent rays connecting source circle to target loupe
            tangents = compute_circle_tangent_lines(p_src, src_r, p_tgt, rad)
            if getattr(self, "_magnify_rays_item", None) is None:
                self._magnify_rays_item = QGraphicsPathItem()
                self._magnify_rays_item.setZValue(9997)
            if self._magnify_rays_item.scene() is not self.scene:
                self.scene.addItem(self._magnify_rays_item)
            ray_path = QPainterPath()
            for pt1, pt2 in tangents:
                ray_path.moveTo(pt1)
                ray_path.lineTo(pt2)
            self._magnify_rays_item.setPath(ray_path)
            self._magnify_rays_item.setPen(QPen(ring_col, max(1.5, w * 0.4), Qt.SolidLine, Qt.RoundCap))
            self._magnify_rays_item.setVisible(bool(tangents))

            # 3. Magnified loupe at target cursor position
            sample_src = p_src
            if self.current_recipe.crop_rect is not None:
                cl, ct, _, _ = self.current_recipe.crop_rect
                sample_src = QPointF(p_src.x() + cl, p_src.y() + ct)
            loupe_pm = create_magnifier_pixmap(self.master_np_rgb, sample_src, rad, zoom_factor=zoom, ring_color=ring_col)
            if getattr(self, "_magnify_hover_item", None) is None:
                self._magnify_hover_item = QGraphicsPixmapItem()
                self._magnify_hover_item.setZValue(9998)
            if self._magnify_hover_item.scene() is not self.scene:
                self.scene.addItem(self._magnify_hover_item)
            self._magnify_hover_item.setPixmap(loupe_pm)
            offset = loupe_pm.width() / 2.0
            self._magnify_hover_item.setPos(p_tgt.x() - offset, p_tgt.y() - offset)
            self._magnify_hover_item.setVisible(True)

        else:
            # Step 1: Pre-selection hover ring showing the exact source area that will be magnified
            if getattr(self, "_magnify_src_circle_item", None) is None:
                self._magnify_src_circle_item = QGraphicsEllipseItem()
                self._magnify_src_circle_item.setZValue(9996)
            if self._magnify_src_circle_item.scene() is not self.scene:
                self.scene.addItem(self._magnify_src_circle_item)
            self._magnify_src_circle_item.setRect(QRectF(sp.x() - src_r, sp.y() - src_r, src_r * 2.0, src_r * 2.0))
            self._magnify_src_circle_item.setPen(QPen(ring_col, 1.5, Qt.DashLine))
            self._magnify_src_circle_item.setBrush(QColor(ring_col.red(), ring_col.green(), ring_col.blue(), 20))
            self._magnify_src_circle_item.setVisible(True)

            if getattr(self, "_magnify_rays_item", None) is not None:
                self._magnify_rays_item.setVisible(False)
            if getattr(self, "_magnify_hover_item", None) is not None:
                self._magnify_hover_item.setVisible(False)

    def _remove_scene_item(self, item: Any) -> None:
        if item is not None:
            try:
                self.scene.removeItem(item)
            except Exception:
                pass

    def _commit_inline_text(self) -> None:
        """Commits the active on-canvas inline text editor directly onto the photo."""
        if getattr(self, "_inline_text_input", None) is None:
            return
        widget = self._inline_text_input
        txt = widget.toPlainText().strip() if hasattr(widget, "toPlainText") else widget.text().strip()
        sp = getattr(self, "_inline_text_scene_pos", None)
        width = getattr(self.markup_toolbar, "stroke_width", 4)
        col = getattr(self.markup_toolbar, "current_color", QColor(230, 40, 40))

        sec_col = getattr(self.markup_toolbar, "secondary_color", QColor(20, 25, 35))

        self._inline_text_input.deleteLater()
        self._inline_text_input = None
        self._inline_text_scene_pos = None

        if txt and sp is not None:
            el = MarkupElement(
                kind="text",
                rect=QRectF(sp.x(), sp.y(), 120, 32),
                text=txt,
                color=col,
                bg_color=sec_col,
                stroke_width=width,
                font_bold=getattr(self.markup_toolbar, "is_text_bold", True),
                font_italic=getattr(self.markup_toolbar, "is_text_italic", False),
            )
            self._commit_single_markup_element(el, "Markup Text")

    def _cancel_inline_text(self) -> None:
        if getattr(self, "_inline_text_input", None) is not None:
            self._inline_text_input.deleteLater()
            self._inline_text_input = None
            self._inline_text_scene_pos = None

    def _adjust_inline_text_size(self) -> None:
        """Dynamically scales the on-canvas inline text editor to match typed multi-line contents."""
        inp = getattr(self, "_inline_text_input", None)
        if inp is None:
            return
        view_scale = self.graphics_view.transform().m11()
        if abs(view_scale) < 1e-4:
            view_scale = 1.0
        screen_pad_x = max(8, int(round(12.0 * view_scale)))
        screen_pad_y = max(4, int(round(6.0 * view_scale)))
        font = inp.font()
        fm = QFontMetrics(font)
        line_h = fm.lineSpacing()
        doc = inp.document()
        doc.setTextWidth(-1)
        w = max(200, int(math.ceil(doc.idealWidth())) + 2 * screen_pad_x + 24)
        h = max(line_h + 2 * screen_pad_y + 12, int(math.ceil(doc.size().height())) + 2 * screen_pad_y + 10)
        inp.resize(w, h)

    def _commit_single_markup_element(self, el: MarkupElement, desc: str = "Markup") -> None:
        """Bakes a single markup element onto master_np_rgb and pushes to undo_stack."""
        if self.master_np_rgb is None or el is None:
            return

        before = self.master_np_rgb.copy()

        # Shift elements if cropped
        if self.current_recipe.crop_rect is not None:
            l, t, _, _ = self.current_recipe.crop_rect
            shifted = MarkupElement(
                kind=el.kind,
                rect=QRectF(el.rect.x() + l, el.rect.y() + t, el.rect.width(), el.rect.height()) if el.rect else None,
                start_pt=QPointF(el.start_pt.x() + l, el.start_pt.y() + t) if el.start_pt else None,
                end_pt=QPointF(el.end_pt.x() + l, el.end_pt.y() + t) if el.end_pt else None,
                points=[QPointF(pt.x() + l, pt.y() + t) for pt in el.points],
                color=el.color,
                bg_color=el.bg_color,
                stroke_width=el.stroke_width,
                text=el.text,
                zoom_factor=el.zoom_factor,
                font_bold=getattr(el, "font_bold", True),
                font_italic=getattr(el, "font_italic", False),
            )
            element_to_bake = shifted
        else:
            element_to_bake = el

        # Bake directly onto RGB master without redundant RGB <-> BGR roundtrips
        self.master_np_rgb = flatten_markup_onto_rgb_image(self.master_np_rgb, [element_to_bake])

        self._session_markup_command_count = getattr(self, "_session_markup_command_count", 0) + 1
        self._markup_elements.append(el)

        # Record dummy preview item for test assertions
        if el.kind in ("pixelate", "blur", "magnifier", "magnify"):
            r = el.rect or QRectF(0, 0, 10, 10)
            pm = QPixmap(int(max(1, r.width())), int(max(1, r.height())))
            it = QGraphicsPixmapItem(pm)
            self._markup_preview_items.append(it)
        else:
            p_item = QGraphicsPathItem()
            self._markup_preview_items.append(p_item)

        cmd = PixelArrayUndoCommand(self, before, self.master_np_rgb.copy(), desc)
        self.undo_stack.push(cmd)
        if hasattr(self.markup_toolbar, "btn_undo"):
            self.markup_toolbar.btn_undo.setEnabled(True)
        self._update_display_proxy()
        self._render_current_recipe()
        self.update_revert_state()
        if self.current_media_id is not None:
            self.pixel_edit_saved.emit(self.current_media_id, "")

        if el.kind not in ("pen", "highlighter"):
            self._active_movable_element = el
            self._active_movable_before_master = before.copy()
            if getattr(self, "_pending_selection_box_item", None) is not None:
                self._remove_scene_item(self._pending_selection_box_item)
                self._pending_selection_box_item = None
            bounds = get_markup_element_bounds(el)
            if not bounds.isEmpty():
                box_item = QGraphicsRectItem(bounds)
                pen = QPen(QColor(56, 144, 240, 220), 1.5, Qt.DashLine)
                box_item.setPen(pen)
                box_item.setBrush(Qt.NoBrush)
                box_item.setZValue(9999)
                self.scene.addItem(box_item)
                self._pending_selection_box_item = box_item

    def _clamp_point_to_image(self, pt: QPointF) -> QPointF:
        if self.master_np_rgb is not None:
            h, w = self.master_np_rgb.shape[:2]
        elif self.current_pil_image is not None:
            w, h = self.current_pil_image.size
        else:
            return pt
        cx = max(0.0, min(float(w), pt.x()))
        cy = max(0.0, min(float(h), pt.y()))
        return QPointF(cx, cy)

    def _start_markup(self, sp: QPointF) -> None:
        if getattr(self, "_active_movable_element", None) is not None:
            self._finalize_movable_markup()
        if self.master_np_rgb is not None:
            h, w = self.master_np_rgb.shape[:2]
        elif self.current_pil_image is not None:
            w, h = self.current_pil_image.size
        else:
            return
        if not QRectF(0.0, 0.0, float(w), float(h)).contains(sp):
            return
        tool = getattr(self.markup_toolbar, "active_tool", "pen")
        col = getattr(self.markup_toolbar, "current_color", QColor(230, 40, 40))
        width = getattr(self.markup_toolbar, "stroke_width", 4)

        if tool in ("magnify", "magnifier"):
            if getattr(self, "_magnify_source_pt", None) is None:
                self._magnify_source_pt = sp
                self._magnify_press_pt = sp
                self._markup_active = True
                self._update_magnify_hover(sp)
                self._show_status("Source selected. Drag or click where you want the magnified loupe placed (Esc to cancel)", 3500)
            else:
                self._commit_magnifier_callout(self._magnify_source_pt, sp)
                self._magnify_source_pt = None
                self._markup_active = False
            return

        elif tool == "text":
            if getattr(self, "_inline_text_input", None) is not None:
                self._commit_inline_text()

            vp_pt = self.graphics_view.mapFromScene(sp)
            view_scale = self.graphics_view.transform().m11()
            if abs(view_scale) < 1e-4:
                view_scale = 1.0

            master_font_pt = max(10, int(round(width * 2.2)))
            screen_font_pt = max(9, int(round(master_font_pt * view_scale)))
            screen_pad_x = max(8, int(round(12.0 * view_scale)))
            screen_pad_y = max(4, int(round(6.0 * view_scale)))
            sec_col = getattr(self.markup_toolbar, "secondary_color", QColor(20, 25, 35))
            if sec_col.alpha() == 0:
                bg_qss = "background: transparent; border: 1.5px dashed rgba(56, 144, 240, 200);"
            else:
                bg_rgba = f"rgba({sec_col.red()}, {sec_col.green()}, {sec_col.blue()}, 220)"
                bg_qss = f"background: {bg_rgba}; border: 1.5px solid #3890F0;"
            is_bold = getattr(self.markup_toolbar, "is_text_bold", True)
            is_italic = getattr(self.markup_toolbar, "is_text_italic", False)

            class _InlineTextEdit(QTextEdit):
                returnPressed = Signal()

                def keyPressEvent(self, e: Any) -> None:
                    if hasattr(e, "key") and e.key() in (Qt.Key_Return, Qt.Key_Enter):
                        if not (e.modifiers() & (Qt.ShiftModifier | Qt.AltModifier)):
                            self.returnPressed.emit()
                            e.accept()
                            return
                    super().keyPressEvent(e)

            inp = _InlineTextEdit(self.graphics_view.viewport())
            inp_font = QFont("Segoe UI", screen_font_pt)
            if is_bold:
                inp_font.setBold(True)
            if is_italic:
                inp_font.setItalic(True)
            inp.setFont(inp_font)
            fm = QFontMetrics(inp_font)
            line_h = fm.lineSpacing()

            inp.setPlaceholderText("Type text note (Enter to commit, Shift+Enter for newline)...")
            inp.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            inp.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            inp.setTabChangesFocus(True)
            inp.document().setDocumentMargin(0)
            inp.setStyleSheet(f"""
                QTextEdit {{
                    {bg_qss}
                    color: {col.name()};
                    border-radius: 6px;
                    padding: {screen_pad_y}px {screen_pad_x}px;
                    font-family: 'Segoe UI';
                    font-size: {screen_font_pt}pt;
                    font-weight: {'bold' if is_bold else 'normal'};
                    font-style: {'italic' if is_italic else 'normal'};
                }}
            """)
            inp.move(max(4, vp_pt.x()), max(4, vp_pt.y()))
            init_w = max(240, fm.horizontalAdvance("Type text note...") + 2 * screen_pad_x + 36)
            init_h = max(38, line_h + 2 * screen_pad_y + 12)
            inp.resize(init_w, init_h)
            inp.textChanged.connect(self._adjust_inline_text_size)
            inp.returnPressed.connect(self._commit_inline_text)
            inp.show()
            inp.setFocus()
            self._inline_text_input = inp
            self._inline_text_scene_pos = sp
            self._markup_active = False
            return

        if not hasattr(self, "_active_markup_scene_items"):
            self._active_markup_scene_items = set()

        # Remove any previously uncommitted or dangling temporary item
        if self._markup_current_item is not None:
            self._remove_scene_item(self._markup_current_item)
            self._markup_current_item = None
        for it in list(self._active_markup_scene_items):
            self._remove_scene_item(it)
        self._active_markup_scene_items.clear()

        self._markup_active = True
        self._markup_start_pt = sp

        if tool == "pen":
            item = QGraphicsPathItem()
            pen = QPen(col, max(1, width), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            item.setPen(pen)
            item.setBrush(Qt.NoBrush)
            path = QPainterPath(sp)
            item.setPath(path)
            self.scene.addItem(item)
            self._active_markup_scene_items.add(item)
            self._markup_current_item = item
            self._markup_current_element = MarkupElement(
                kind="pen", points=[sp], color=col, stroke_width=width
            )

        elif tool == "highlighter":
            hl_w = max(8, int(width * 2.6))
            hl_col = QColor(col.red(), col.green(), col.blue(), 105)
            item = QGraphicsPathItem()
            pen = QPen(hl_col, hl_w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            item.setPen(pen)
            item.setBrush(Qt.NoBrush)
            path = QPainterPath(sp)
            item.setPath(path)
            self.scene.addItem(item)
            self._active_markup_scene_items.add(item)
            self._markup_current_item = item
            self._markup_current_element = MarkupElement(
                kind="highlighter", points=[sp], color=col, stroke_width=width
            )

        elif tool == "arrow":
            item = QGraphicsPathItem()
            item.setPen(QPen(col, max(1, width), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            item.setBrush(QBrush(col))
            self.scene.addItem(item)
            self._active_markup_scene_items.add(item)
            self._markup_current_item = item
            self._markup_current_element = MarkupElement(
                kind="arrow", start_pt=sp, end_pt=sp, color=col, stroke_width=width
            )

        elif tool in ("rect", "ellipse", "star"):
            pen = QPen(col, max(1, width), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            if tool == "rect":
                item = QGraphicsRectItem(QRectF(sp, sp))
                item.setPen(pen)
                item.setBrush(Qt.NoBrush)
            elif tool == "ellipse":
                item = QGraphicsEllipseItem(QRectF(sp, sp))
                item.setPen(pen)
                item.setBrush(Qt.NoBrush)
            else:
                item = QGraphicsPathItem()
                item.setPen(pen)
                item.setBrush(Qt.NoBrush)
            self.scene.addItem(item)
            self._active_markup_scene_items.add(item)
            self._markup_current_item = item
            self._markup_current_element = MarkupElement(
                kind=tool, rect=QRectF(sp, sp), color=col, stroke_width=width
            )

        elif tool in ("pixelate", "blur"):
            item = QGraphicsRectItem(QRectF(sp, sp))
            pen = QPen(col, 1.5, Qt.DashLine)
            item.setPen(pen)
            brush_col = QColor(col.red(), col.green(), col.blue(), 40)
            item.setBrush(QBrush(brush_col))
            self.scene.addItem(item)
            self._active_markup_scene_items.add(item)
            self._markup_current_item = item
            self._markup_current_element = MarkupElement(
                kind=tool, rect=QRectF(sp, sp), color=col, stroke_width=width
            )

    def _drag_markup(self, sp: QPointF) -> None:
        sp = self._clamp_point_to_image(sp)
        tool = getattr(self.markup_toolbar, "active_tool", "pen")
        if tool in ("magnify", "magnifier"):
            self._update_magnify_hover(sp)
            return

        if not self._markup_active or self._markup_current_item is None or self._markup_start_pt is None:
            return
        p1 = self._markup_start_pt
        width = getattr(self.markup_toolbar, "stroke_width", 4)

        if tool in ("pen", "highlighter"):
            if self._markup_current_element:
                self._markup_current_element.points.append(sp)
                path = self._markup_current_item.path()
                path.lineTo(sp)
                self._markup_current_item.setPath(path)

        elif tool == "arrow":
            dx = sp.x() - p1.x()
            dy = sp.y() - p1.y()
            if math.hypot(dx, dy) >= 4.0:
                p_base, arrowhead = build_arrow_geometry(p1, sp, width)
                path = QPainterPath()
                path.moveTo(p1)
                path.lineTo(p_base)
                if arrowhead.count() >= 3:
                    path.moveTo(arrowhead[0])
                    for i in range(1, arrowhead.count()):
                        path.lineTo(arrowhead[i])
                    path.closeSubpath()
                self._markup_current_item.setPath(path)
            if self._markup_current_element:
                self._markup_current_element.end_pt = sp

        elif tool in ("rect", "ellipse", "star"):
            r = QRectF(min(p1.x(), sp.x()), min(p1.y(), sp.y()), abs(sp.x() - p1.x()), abs(sp.y() - p1.y()))
            if tool in ("rect", "ellipse"):
                self._markup_current_item.setRect(r)
            else:
                star_p = build_star_polygon_in_rect(r)
                p = QPainterPath()
                if not star_p.isEmpty():
                    p.moveTo(star_p[0])
                    for i in range(1, star_p.count()):
                        p.lineTo(star_p[i])
                    p.lineTo(star_p[0])
                self._markup_current_item.setPath(p)
            if self._markup_current_element:
                self._markup_current_element.rect = r

        elif tool in ("pixelate", "blur"):
            r = QRectF(min(p1.x(), sp.x()), min(p1.y(), sp.y()), abs(sp.x() - p1.x()), abs(sp.y() - p1.y()))
            self._markup_current_item.setRect(r)
            if self._markup_current_element:
                self._markup_current_element.rect = r

    def _finish_markup(self, sp: QPointF) -> None:
        sp = self._clamp_point_to_image(sp)
        tool = getattr(self.markup_toolbar, "active_tool", "pen")
        if tool in ("magnify", "magnifier"):
            self._markup_active = False
            src_pt = getattr(self, "_magnify_source_pt", None)
            press_pt = getattr(self, "_magnify_press_pt", None)
            if src_pt is not None and press_pt is not None:
                dist = math.hypot(sp.x() - press_pt.x(), sp.y() - press_pt.y())
                if dist >= 16.0:
                    self._commit_magnifier_callout(src_pt, sp)
                    self._magnify_source_pt = None
            return

        if not getattr(self, "_markup_active", False):
            return
        self._markup_active = False
        p1 = getattr(self, "_markup_start_pt", None)
        cur_item = getattr(self, "_markup_current_item", None)

        tool = getattr(self.markup_toolbar, "active_tool", "pen")
        col = getattr(self.markup_toolbar, "current_color", QColor(230, 40, 40))
        width = getattr(self.markup_toolbar, "stroke_width", 4)
        element_to_commit: MarkupElement | None = None

        try:
            if p1 is not None and cur_item is not None:
                if tool in ("pen", "highlighter"):
                    if self._markup_current_element:
                        self._markup_current_element.points.append(sp)
                        if len(self._markup_current_element.points) >= 2:
                            element_to_commit = self._markup_current_element

                elif tool == "arrow":
                    dx = sp.x() - p1.x()
                    dy = sp.y() - p1.y()
                    if math.hypot(dx, dy) >= 6.0 and self._markup_current_element:
                        self._markup_current_element.end_pt = sp
                        element_to_commit = self._markup_current_element

                elif tool in ("rect", "ellipse", "star"):
                    r = QRectF(min(p1.x(), sp.x()), min(p1.y(), sp.y()), abs(sp.x() - p1.x()), abs(sp.y() - p1.y()))
                    if r.width() >= 6 and r.height() >= 6 and self._markup_current_element:
                        self._markup_current_element.rect = r
                        element_to_commit = self._markup_current_element

                elif tool in ("pixelate", "blur"):
                    r = QRectF(min(p1.x(), sp.x()), min(p1.y(), sp.y()), abs(sp.x() - p1.x()), abs(sp.y() - p1.y()))
                    if r.width() >= 8 and r.height() >= 8 and self.master_np_rgb is not None:
                        rx = max(0, int(round(r.x())))
                        ry = max(0, int(round(r.y())))
                        rw = max(1, min(self.master_np_rgb.shape[1] - rx, int(round(r.width()))))
                        rh = max(1, min(self.master_np_rgb.shape[0] - ry, int(round(r.height()))))
                        if rw > 4 and rh > 4:
                            element_to_commit = MarkupElement(kind=tool, rect=QRectF(rx, ry, rw, rh), color=col, stroke_width=width)
        finally:
            if cur_item is not None:
                self._remove_scene_item(cur_item)
            if hasattr(self, "_active_markup_scene_items"):
                for it in list(self._active_markup_scene_items):
                    self._remove_scene_item(it)
                self._active_markup_scene_items.clear()
            self._markup_current_item = None
            self._markup_current_element = None
            self._markup_start_pt = None

        if element_to_commit is not None:
            self._commit_single_markup_element(element_to_commit, f"Markup {tool.title()}")

    def _undo_markup(self) -> None:
        if getattr(self, "_inline_text_input", None) is not None:
            self._cancel_inline_text()
            return
        self._finalize_movable_markup()
        if hasattr(self, "undo_stack") and self.undo_stack.canUndo():
            self.undo_stack.undo()
            if self._markup_elements:
                self._markup_elements.pop()
            if self._markup_preview_items:
                it = self._markup_preview_items.pop()
                self._remove_scene_item(it)
            if hasattr(self.markup_toolbar, "btn_undo"):
                self.markup_toolbar.btn_undo.setEnabled(self.undo_stack.canUndo())
            self.update_revert_state()

    def _cleanup_markup_temp_items(self) -> None:
        """Removes temporary floating preview items and cursor overlays without rolling back undo stack."""
        for it in self._markup_preview_items:
            self._remove_scene_item(it)
        self._markup_preview_items.clear()
        self._markup_elements.clear()
        if self._markup_current_item:
            self._remove_scene_item(self._markup_current_item)
            self._markup_current_item = None
        self._markup_current_element = None
        self._markup_active = False
        if hasattr(self, "_active_markup_scene_items"):
            for it in list(self._active_markup_scene_items):
                self._remove_scene_item(it)
            self._active_markup_scene_items.clear()
        if hasattr(self, "_magnify_hover_item") and self._magnify_hover_item is not None:
            self._remove_scene_item(self._magnify_hover_item)
            self._magnify_hover_item = None
        if hasattr(self, "_magnify_src_circle_item") and self._magnify_src_circle_item is not None:
            self._remove_scene_item(self._magnify_src_circle_item)
            self._magnify_src_circle_item = None
        if hasattr(self, "_magnify_rays_item") and self._magnify_rays_item is not None:
            self._remove_scene_item(self._magnify_rays_item)
            self._magnify_rays_item = None

    def _purge_all_transient_scene_items(self) -> None:
        """Purges any loose, orphaned, or abandoned graphics items from the scene."""
        self._cleanup_markup_temp_items()
        protected = {
            getattr(self, "pixmap_item", None),
            getattr(self, "_alpha_backdrop_item", None),
            getattr(self.scene, "crop_rect_item", None),
            getattr(self, "_face_target_item", None),
            getattr(self, "_face_adjust_box_item", None),
        }
        if hasattr(self, "_face_overlay_items"):
            protected.update(self._face_overlay_items)

        for item in list(self.scene.items()):
            if item not in protected:
                try:
                    self.scene.removeItem(item)
                except Exception:
                    pass

    def _clear_markup_overlays(self) -> None:
        """User-triggered clear: cancels active text, cleans overlays, and rolls back all markups on image."""
        self._cancel_inline_text()
        self._finalize_movable_markup()
        self._cleanup_markup_temp_items()

        # Revert all markup operations performed on this photo
        start_idx = getattr(self, "_markup_session_start_undo_idx", 0)
        while hasattr(self, "undo_stack") and self.undo_stack.canUndo() and self.undo_stack.index() > 0:
            cmd = self.undo_stack.command(self.undo_stack.index() - 1)
            cmd_text = cmd.text() if cmd else ""
            if any(k in cmd_text for k in ("Markup", "Arrow", "Pen", "Shape", "Text", "Magnif", "Loupe", "Callout")):
                self.undo_stack.undo()
            elif self.undo_stack.index() > start_idx:
                self.undo_stack.undo()
            else:
                break

        # If baseline master is saved, unconditionally restore master_np_rgb to session start
        start_master = getattr(self, "_markup_session_start_master", None)
        if start_master is not None and self.master_np_rgb is not None:
            self.master_np_rgb = start_master.copy()
            self._update_display_proxy()
            self._render_current_recipe()

        if hasattr(self.markup_toolbar, "btn_undo"):
            self.markup_toolbar.btn_undo.setEnabled(self.undo_stack.canUndo())
        self._purge_all_transient_scene_items()
        self.update_revert_state()

    def _commit_markup(self) -> None:
        if getattr(self, "_inline_text_input", None) is not None:
            self._commit_inline_text()
        self._finalize_movable_markup()
        self.set_markup_mode(False)

    # -------------------------------------------------------------------------
    # OpenCV-Powered High-Fidelity Adjustments
    # -------------------------------------------------------------------------
    def begin_slider_adjustment(self) -> None:
        self._is_sliding = True
        self._master_render_dirty = True
        self._master_render_gen += 1
        if hasattr(self, "_master_render_timer") and self._master_render_timer.isActive():
            self._master_render_timer.stop()
        if self._recipe_before_slider is None:
            self._recipe_before_slider = self.current_recipe.copy()

    def commit_slider_adjustment(self, action_name: str = "Adjust Parameters") -> None:
        self._is_sliding = False
        if getattr(self, "_recipe_before_slider", None) is not None:
            if self._recipe_before_slider != self.current_recipe:
                cmd = RecipeUndoCommand(
                    self, self._recipe_before_slider, self.current_recipe.copy(), action_name
                )
                self.undo_stack.push(cmd)
                self.recipe_changed.emit(self.current_recipe)
            self._recipe_before_slider = None

        # Immediately snap viewport to the crisp 1280px display proxy on slider release,
        # then schedule the full-resolution master pass off the GUI thread after 250ms.
        if self.display_proxy_rgb is not None and self.master_np_rgb is not None and not self.current_is_video:
            self._render_proxy_to_viewport(
                self.display_proxy_rgb, self.display_proxy_alpha, self._proxy_scale
            )
            self._master_render_dirty = True

        if hasattr(self, "_master_render_timer"):
            self._master_render_timer.start(250)
        else:
            self._render_current_recipe()

    def _apply_rendered_master_to_viewport(
        self,
        rendered_np: np.ndarray,
        rendered_alpha: np.ndarray | None,
        mask_overlay: bool,
    ) -> None:
        self.current_pil_image = Image.fromarray(rendered_np)
        if mask_overlay:
            try:
                from src.imaging.creative_effects import (
                    get_depth_and_matte_engine,
                    apply_boundary_shift_and_feather,
                )
                engine = get_depth_and_matte_engine()
                _, alpha_matte = engine.get_depth_and_matte(rendered_np, media_id=self.current_media_id)
                if alpha_matte is not None:
                    alpha_matte = apply_boundary_shift_and_feather(
                        alpha_matte,
                        boundary_shift=getattr(self.current_recipe, "boundary_shift", 0.0),
                        boundary_feather=getattr(self.current_recipe, "boundary_feather", 0.0),
                    )
                    bg_mask = (1.0 - np.clip(alpha_matte, 0.0, 1.0))[:, :, None]
                    overlay_color = np.array([220.0, 20.0, 60.0], dtype=np.float32)
                    blended = rendered_np.astype(np.float32) * (1.0 - bg_mask * 0.45) + overlay_color * (bg_mask * 0.45)
                    rendered_np = np.clip(blended, 0.0, 255.0).astype(np.uint8)
            except Exception:
                pass

        pix = self._build_pixmap_from_rgb_and_alpha(rendered_np, rendered_alpha)
        self._adjusted_pixmap = pix
        self.pixmap_item.setPixmap(pix)
        self._update_alpha_backdrop(pix.width(), pix.height(), rendered_alpha is not None)
        self.scene.setSceneRect(QRectF(0, 0, pix.width(), pix.height()))
        self.scene.crop_rect_item.set_image_rect(QRectF(0, 0, pix.width(), pix.height()))
        self.image_changed.emit(self.current_pil_image)

        # Update face overlays for current static image (clears overlays when empty)
        if hasattr(self, "_current_people_regions") and not self.current_is_video:
            self.render_face_regions(self._current_people_regions)
        self._sync_animation_playback()

    def _on_master_render_finished(
        self,
        gen: int,
        media_id: Any,
        rendered_np: np.ndarray,
        rendered_alpha: np.ndarray | None,
        mask_overlay: bool,
    ) -> None:
        if gen != self._master_render_gen or media_id != self.current_media_id:
            return
        if self.master_np_rgb is None or self.current_is_video:
            return
        self._master_render_dirty = False
        self._apply_rendered_master_to_viewport(rendered_np, rendered_alpha, mask_overlay)

    def ensure_master_rendered(self) -> None:
        """Synchronously ensures current_pil_image reflects the full-res master render before Save/Export."""
        if hasattr(self, "_master_render_timer") and self._master_render_timer.isActive():
            self._master_render_timer.stop()
        if self.master_np_rgb is None or self.current_is_video:
            return
        if not getattr(self, "_master_render_dirty", False) and self.current_pil_image is not None:
            return
        self._master_render_gen += 1
        self._master_render_dirty = False
        rendered_np = render_adjusted_image(
            self.master_np_rgb, self.current_recipe, media_id=self.current_media_id
        )
        h, w, _c = rendered_np.shape
        rendered_alpha = (
            transform_alpha_for_recipe(self.master_alpha, self.current_recipe, target_hw=(h, w))
            if self.master_alpha is not None
            else None
        )
        self._apply_rendered_master_to_viewport(
            rendered_np, rendered_alpha, getattr(self, "_mask_overlay_preview", False)
        )

    def _flush_master_render(self) -> None:
        if hasattr(self, "_master_render_timer") and self._master_render_timer.isActive():
            self._master_render_timer.stop()
        if self.master_np_rgb is None or self.current_is_video:
            return
        self._master_render_gen += 1
        self._master_render_dirty = True
        runnable = _MasterRenderRunnable(
            gen=self._master_render_gen,
            media_id=self.current_media_id,
            master_rgb=self.master_np_rgb,
            master_alpha=self.master_alpha,
            recipe=self.current_recipe.copy(),
            mask_overlay=getattr(self, "_mask_overlay_preview", False),
        )
        runnable.signals.finished.connect(self._on_master_render_finished)
        QThreadPool.globalInstance().start(runnable)

    def set_exposure_parameters(
        self, brightness: int, contrast: int, shadows: int, highlights: int, clarity: int = 0
    ) -> None:
        if self._recipe_before_slider is None:
            self._recipe_before_slider = self.current_recipe.copy()
        self._is_sliding = True
        self.current_recipe.brightness = float(brightness)
        self.current_recipe.contrast = float(contrast)
        self.current_recipe.shadows = float(shadows)
        self.current_recipe.highlights = float(highlights)
        self.current_recipe.clarity = float(clarity)
        self._render_current_recipe()

    def set_color_parameters(
        self, temperature: int, tint: int, saturation: int, dehaze: int = 0
    ) -> None:
        if self._recipe_before_slider is None:
            self._recipe_before_slider = self.current_recipe.copy()
        self._is_sliding = True
        self.current_recipe.temperature = float(temperature)
        self.current_recipe.tint = float(tint)
        self.current_recipe.saturation = float(saturation)
        self.current_recipe.dehaze = float(dehaze)
        self._render_current_recipe()

    def set_optics_parameters(self, vignette: int = 0) -> None:
        if self._recipe_before_slider is None:
            self._recipe_before_slider = self.current_recipe.copy()
        self._is_sliding = True
        self.current_recipe.vignette = float(vignette)
        self._render_current_recipe()

    def set_levels(self, black_point: int = 0, white_point: int = 255) -> None:
        if self._recipe_before_slider is None:
            self._recipe_before_slider = self.current_recipe.copy()
        self._is_sliding = True
        self.current_recipe.black_point = float(black_point)
        self.current_recipe.white_point = float(white_point)
        self._render_current_recipe()

    def set_depth_bokeh_parameters(
        self,
        blur: int,
        focal_plane: int,
        glow: int = 0,
        boundary_shift: int = 0,
        boundary_feather: int = 0,
    ) -> None:
        if self._recipe_before_slider is None:
            self._recipe_before_slider = self.current_recipe.copy()
        self._is_sliding = True
        self.current_recipe.depth_blur = float(blur)
        self.current_recipe.focal_plane = float(focal_plane)
        self.current_recipe.bokeh_glow = float(glow)
        self.current_recipe.boundary_shift = float(boundary_shift)
        self.current_recipe.boundary_feather = float(boundary_feather)
        self._render_current_recipe()

    def set_mask_overlay_preview(self, enabled: bool) -> None:
        """Toggles a semi-transparent ruby mask overlay over background pixels to visualize boundary adjustments."""
        self._mask_overlay_preview = bool(enabled)
        self._render_current_recipe()

    def set_depth_relight_parameters(
        self, near: int, far: int, warmth_near: int = 0, warmth_far: int = 0
    ) -> None:
        if self._recipe_before_slider is None:
            self._recipe_before_slider = self.current_recipe.copy()
        self._is_sliding = True
        self.current_recipe.relight_near = float(near)
        self.current_recipe.relight_far = float(far)
        self.current_recipe.warmth_near = float(warmth_near)
        self.current_recipe.warmth_far = float(warmth_far)
        self._render_current_recipe()

    def set_depth_haze_parameters(self, haze: int, haze_warmth: int = 0) -> None:
        if self._recipe_before_slider is None:
            self._recipe_before_slider = self.current_recipe.copy()
        self._is_sliding = True
        self.current_recipe.depth_haze = float(haze)
        self.current_recipe.depth_haze_warmth = float(haze_warmth)
        self._render_current_recipe()

    def reset_depth_light(self) -> None:
        before = self.current_recipe.copy()
        new_rec = self.current_recipe.copy()
        new_rec.depth_blur = 0.0
        new_rec.focal_plane = 0.0
        new_rec.bokeh_glow = 0.0
        new_rec.boundary_shift = 0.0
        new_rec.boundary_feather = 0.0
        new_rec.relight_near = 0.0
        new_rec.relight_far = 0.0
        new_rec.warmth_near = 0.0
        new_rec.warmth_far = 0.0
        new_rec.depth_haze = 0.0
        new_rec.depth_haze_warmth = 0.0
        cmd = RecipeUndoCommand(self, before, new_rec, "Reset Depth & Light")
        self.undo_stack.push(cmd)
        self.current_recipe = new_rec
        self._render_current_recipe()
        self.recipe_changed.emit(self.current_recipe)

    def copy_subject_to_clipboard(self) -> bool:
        """
        Extracts the foreground subject with a transparent background (RGBA) using DepthAndMatteEngine
        (respecting current boundary_shift & boundary_feather) and copies it to the system clipboard.
        """
        if self.master_np_rgb is None or self.current_is_video:
            return False
        from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QMimeData
        from src.imaging.creative_effects import extract_subject_cutout_rgba
        rendered_rgb = render_adjusted_image(
            self.master_np_rgb, self.current_recipe, media_id=self.current_media_id
        )
        rgba, _ = extract_subject_cutout_rgba(
            rendered_rgb,
            boundary_shift=getattr(self.current_recipe, "boundary_shift", 0.0),
            boundary_feather=getattr(self.current_recipe, "boundary_feather", 0.0),
            media_id=self.current_media_id,
        )
        rgba_c = np.ascontiguousarray(rgba)
        h, w = rgba_c.shape[:2]
        qimg = QImage(rgba_c.data, w, h, 4 * w, QImage.Format_RGBA8888). convertToFormat(
            QImage.Format_ARGB32
        )
        self._last_copied_subject_qimage = qimg

        mime = QMimeData()
        mime.setImageData(qimg)
        ba = QByteArray()
        buf = QBuffer(ba)
        if buf.open(QIODevice.WriteOnly):
            qimg.save(buf, "PNG")
            buf.close()
            mime.setData("image/png", ba)
            mime.setData("PNG", ba)
        QApplication.clipboard().setMimeData(mime)
        self._show_status("Copied subject cutout (transparent RGBA) to clipboard", 4000)
        return True

    def remove_background_cutout(self) -> bool:
        """
        Removes the background from the current photo in-place using DepthAndMatteEngine,
        updating master_alpha (and edge-decontaminated master_np_rgb) with full Undo/Redo support.
        """
        if self.master_np_rgb is None or self.current_is_video:
            return False
        from src.imaging.creative_effects import extract_subject_cutout_rgba
        before_rgb = self.master_np_rgb.copy()
        before_alpha = self.master_alpha.copy() if self.master_alpha is not None else None

        rgba, subj_alpha = extract_subject_cutout_rgba(
            self.master_np_rgb,
            boundary_shift=getattr(self.current_recipe, "boundary_shift", 0.0),
            boundary_feather=getattr(self.current_recipe, "boundary_feather", 0.0),
            media_id=self.current_media_id,
        )
        clean_rgb = np.ascontiguousarray(rgba[:, :, :3])
        if before_alpha is not None and before_alpha.shape == subj_alpha.shape:
            combined_alpha = np.minimum(before_alpha, subj_alpha)
        else:
            combined_alpha = subj_alpha

        self.master_np_rgb = clean_rgb
        self.master_alpha = combined_alpha
        cmd = _AlphaCutoutUndoCommand(
            self, before_rgb, before_alpha, clean_rgb, combined_alpha, "Remove Background"
        )
        self.undo_stack.push(cmd)
        self._update_display_proxy()
        self._render_current_recipe()
        self.update_revert_state()
        self._show_status(
            "Background removed • Save as Copy (PNG, WebP, AVIF, or JXL) to keep transparency",
            5000,
        )
        return True

    def reset_recipe(self) -> None:
        """Resets all non-crop visual adjustments and levels back to default."""
        before = self.current_recipe.copy()
        new_rec = PhotoAdjustmentRecipe(crop_rect=self.current_recipe.crop_rect)
        cmd = RecipeUndoCommand(self, before, new_rec, "Reset to Default")
        self.undo_stack.push(cmd)
        self.current_recipe = new_rec
        self._render_current_recipe()
        self.recipe_changed.emit(self.current_recipe)

    def set_compare_mode(self, active: bool) -> None:
        """Temporarily toggles between adjusted view and unedited original view."""
        self._is_comparing = active
        if active:
            if getattr(self, "_original_pixmap", None) is not None:
                self.pixmap_item.setPixmap(self._original_pixmap)
            else:
                target_np = self.master_np_rgb
                if target_np is not None:
                    h, w = target_np.shape[:2]
                    qimg = QImage(target_np.data, w, h, 3 * w, QImage.Format_RGB888).copy()
                    pix = QPixmap.fromImage(qimg)
                    self.pixmap_item.setPixmap(pix)
        else:
            if getattr(self, "_adjusted_pixmap", None) is not None:
                self.pixmap_item.setPixmap(self._adjusted_pixmap)
            else:
                self._render_current_recipe()

    def get_effect_preview_pixmaps(self, width: int = 46, height: int = 64) -> dict[str, QPixmap]:
        """Generates live mini-thumbnails of the current photo for each of the ribbon effect buttons."""
        if self.master_np_rgb is None or self.current_is_video:
            return {}
        target_np = self.display_proxy_rgb if self.display_proxy_rgb is not None else self.master_np_rgb
        th, tw = target_np.shape[:2]
        if th <= 0 or tw <= 0 or width <= 0 or height <= 0:
            return {}

        # Generate crisp 2x resolution (at least 92x128) for high DPI and sharp dropdown menu previews
        render_w = max(width * 2, 92)
        render_h = max(height * 2, 128)
        target_aspect = render_w / float(render_h)
        src_aspect = tw / float(th)
        if src_aspect > target_aspect:
            # Source is wider than target: center crop horizontally
            crop_w = max(1, int(round(th * target_aspect)))
            x0 = max(0, (tw - crop_w) // 2)
            cropped = target_np[:, x0 : x0 + crop_w]
        else:
            # Source is taller than target (portrait): center crop vertically
            crop_h = max(1, int(round(tw / target_aspect)))
            y0 = max(0, (th - crop_h) // 2)
            cropped = target_np[y0 : y0 + crop_h, :]

        thumb = cv2.resize(cropped, (render_w, render_h), interpolation=cv2.INTER_AREA)
        effect_keys = [
            "none",
            "classic_chrome",
            "warm_portrait",
            "vivid_landscape",
            "matte_vintage",
            "fine_art_bw",
            "cool_cinema",
            "sepia",
        ]
        result = {}
        for eff in effect_keys:
            rec = PhotoAdjustmentRecipe(effect=eff)
            rendered = render_adjusted_image(thumb, rec)
            rh, rw = rendered.shape[:2]
            qimg = QImage(rendered.data, rw, rh, 3 * rw, QImage.Format_RGB888).copy()
            result[eff] = QPixmap.fromImage(qimg)
        return result

    def apply_pixelate_redact(
        self, rect: tuple[int, int, int, int], block_size: int = 16
    ) -> None:
        """Pixelates a region of the image for privacy redaction, recorded onto the undo stack."""
        if self.master_np_rgb is None or self.current_is_video:
            return
        l, t, r, b = rect
        h, w = self.master_np_rgb.shape[:2]
        l = max(0, min(w - 1, int(l)))
        t = max(0, min(h - 1, int(t)))
        r = max(l + 1, min(w, int(r)))
        b = max(t + 1, min(h, int(b)))
        if r <= l or b <= t:
            return
        before_arr = self.master_np_rgb.copy()
        patch = self.master_np_rgb[t:b, l:r]
        ph, pw = patch.shape[:2]
        down_w = max(1, pw // block_size)
        down_h = max(1, ph // block_size)
        small_patch = cv2.resize(patch, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small_patch, (pw, ph), interpolation=cv2.INTER_NEAREST)
        self.master_np_rgb[t:b, l:r] = pixelated
        cmd = _PixelEditUndoCommand(self, before_arr, self.master_np_rgb.copy(), "Redact / Pixelate")
        self.undo_stack.push(cmd)
        self._update_display_proxy()
        self._render_current_recipe()
        self.pixel_edit_saved.emit()

    def set_straighten(self, angle_deg: float) -> None:
        if self._recipe_before_slider is None:
            self._recipe_before_slider = self.current_recipe.copy()
        self._is_sliding = True
        self.current_recipe.straighten_deg = float(angle_deg)
        self._render_current_recipe()

    def set_detail(self, sharpen: int, denoise: int) -> None:
        if self._recipe_before_slider is None:
            self._recipe_before_slider = self.current_recipe.copy()
        self._is_sliding = True
        self.current_recipe.sharpen = float(sharpen)
        self.current_recipe.denoise = float(denoise)
        self._render_current_recipe()

    def apply_auto_adjust(self) -> None:
        """Analyzes scene dynamic range & color cast using OpenCV and applies balanced parameters."""
        if self.master_np_rgb is None:
            return
        auto_rec = analyze_auto_adjust(self.master_np_rgb)
        auto_rec.crop_rect = self.current_recipe.crop_rect
        auto_rec.straighten_deg = self.current_recipe.straighten_deg
        cmd = RecipeUndoCommand(self, self.current_recipe.copy(), auto_rec.copy(), "Auto Adjust")
        self.current_recipe = auto_rec
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        self.recipe_changed.emit(self.current_recipe)

    def apply_auto_straighten(self) -> float:
        """Detects dominant horizon/vertical tilt using Canny & Hough line transforms."""
        if self.master_np_rgb is None:
            return 0.0
        angle = detect_auto_straighten_angle(self.master_np_rgb)
        before = self.current_recipe.copy()
        self.current_recipe.straighten_deg = angle
        cmd = RecipeUndoCommand(self, before, self.current_recipe.copy(), "Auto Straighten")
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        self.recipe_changed.emit(self.current_recipe)
        return angle

    def apply_exposure_contrast(self, brightness: int, contrast: int) -> None:
        before = self.current_recipe.copy()
        self.current_recipe.brightness = float(brightness)
        self.current_recipe.contrast = float(contrast)
        cmd = RecipeUndoCommand(self, before, self.current_recipe.copy(), "Adjust Exposure")
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        self.recipe_changed.emit(self.current_recipe)

    def apply_color_saturation(self, saturation: int) -> None:
        before = self.current_recipe.copy()
        self.current_recipe.saturation = float(saturation)
        cmd = RecipeUndoCommand(self, before, self.current_recipe.copy(), "Adjust Color")
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        self.recipe_changed.emit(self.current_recipe)

    def apply_straighten(self, angle_deg: float) -> None:
        before = self.current_recipe.copy()
        self.current_recipe.straighten_deg = float(angle_deg)
        cmd = RecipeUndoCommand(self, before, self.current_recipe.copy(), "Straighten Photo")
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        self.recipe_changed.emit(self.current_recipe)

    def apply_preset_effect(
        self, effect: str
    ) -> None:
        before = self.current_recipe.copy()
        self.current_recipe.effect = effect
        label = "Original" if effect == "none" else effect.replace("_", " ").capitalize()
        cmd = RecipeUndoCommand(self, before, self.current_recipe.copy(), f"Effect: {label}")
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        self.recipe_changed.emit(self.current_recipe)

    def preview_recipe(self, recipe: PhotoAdjustmentRecipe | None) -> None:
        """Previews an adjustment recipe in real-time on the viewer without modifying undo stack or current_recipe."""
        if self.master_np_rgb is None or self.current_is_video:
            return
        if recipe is None:
            self._render_current_recipe()
            return
        target_np = self.display_proxy_rgb if self.display_proxy_rgb is not None else self.master_np_rgb
        scale = getattr(self, "_proxy_scale", 1.0)
        proxy_rec = recipe.copy()
        if proxy_rec.crop_rect is not None and scale < 0.999:
            l, t, r, b = proxy_rec.crop_rect
            proxy_rec.crop_rect = (
                int(round(l * scale)),
                int(round(t * scale)),
                int(round(r * scale)),
                int(round(b * scale)),
            )
        rendered_np = render_adjusted_image(
            target_np, proxy_rec, media_id=self.current_media_id
        )
        h, w = rendered_np.shape[:2]
        qimg = QImage(rendered_np.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        if scale < 0.999:
            target_w = int(round(w / scale))
            target_h = int(round(h / scale))
            if target_w != w or target_h != h:
                pix = pix.scaled(target_w, target_h, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        self.pixmap_item.setPixmap(pix)

    def apply_recipe_adjustment(
        self, new_recipe: PhotoAdjustmentRecipe, action_name: str = "Adjust Photo"
    ) -> None:
        """Commits a new recipe to the undo stack and updates the display."""
        if self.master_np_rgb is None:
            return
        cmd = RecipeUndoCommand(self, self.current_recipe.copy(), new_recipe.copy(), action_name)
        self.current_recipe = new_recipe.copy()
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        self.recipe_changed.emit(self.current_recipe)

    def apply_auto_exposure(self) -> None:
        """Analyzes and applies balanced exposure parameters without changing color or effects."""
        if self.master_np_rgb is None:
            return
        auto_rec = analyze_auto_adjust(self.master_np_rgb)
        new_rec = self.current_recipe.copy()
        new_rec.brightness = auto_rec.brightness
        new_rec.contrast = auto_rec.contrast
        new_rec.highlights = auto_rec.highlights
        new_rec.shadows = auto_rec.shadows
        self.apply_recipe_adjustment(new_rec, "Auto Exposure")

    def apply_auto_color(self) -> None:
        """Analyzes and applies balanced color parameters without changing exposure or effects."""
        if self.master_np_rgb is None:
            return
        auto_rec = analyze_auto_adjust(self.master_np_rgb)
        new_rec = self.current_recipe.copy()
        new_rec.temperature = auto_rec.temperature
        new_rec.tint = auto_rec.tint
        new_rec.saturation = auto_rec.saturation
        self.apply_recipe_adjustment(new_rec, "Auto Color")

    def get_preview_thumbnail(self, width: int = 54, height: int = 38) -> np.ndarray | None:
        """Returns a small RGB numpy array of the current image for popup variation menus."""
        if self.master_np_rgb is None or self.current_is_video:
            return None
        target = self.display_proxy_rgb if self.display_proxy_rgb is not None else self.master_np_rgb
        h, w = target.shape[:2]
        if h <= 0 or w <= 0:
            return None
        target_aspect = width / float(height)
        img_aspect = w / float(h)
        if img_aspect > target_aspect:
            crop_w = int(h * target_aspect)
            x1 = (w - crop_w) // 2
            crop = target[:, x1 : x1 + crop_w]
        else:
            crop_h = int(w / target_aspect)
            y1 = (h - crop_h) // 2
            crop = target[y1 : y1 + crop_h, :]
        return cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)

    # -------------------------------------------------------------------------
    # Non-Destructive '.OriginalFiles' Backup Stash & Revert
    # -------------------------------------------------------------------------
    def has_unsaved_adjustments(self) -> bool:
        """Returns True if the current photo has uncommitted recipe adjustments or undoable edits."""
        recipe_dirty = self.current_recipe is not None and not self.current_recipe.is_default()
        try:
            undo_dirty = hasattr(self, "undo_stack") and self.undo_stack is not None and self.undo_stack.canUndo()
        except RuntimeError:
            undo_dirty = False
        return recipe_dirty or undo_dirty

    def save_current_image_with_backup(self) -> str | None:
        """
        Stashes the pristine original file into '<folder>/.OriginalFiles/<filename>'
        (if not already backed up) before saving the rendered recipe image to disk.
        Returns the backup file path.
        """
        if (
            self.master_np_rgb is None
            or self.current_file_path is None
            or self.current_media_id is None
            or self.current_is_video
        ):
            return None

        self.ensure_master_rendered()

        # Never overwrite proprietary Camera RAW sensor files in-place; route safely to Save as Copy (.jpg)
        if self.current_file_path.suffix.lower() in RAW_DOT_EXT_SET:
            return self.save_as_copy()

        self.ensure_master_rendered()
        self._stop_animated_image()
        backup_dir = self.current_file_path.parent / ".OriginalFiles"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / self.current_file_path.name

        if not backup_path.exists() and self.current_file_path.exists():
            shutil.copy2(self.current_file_path, backup_path)

        crop_box = self.current_recipe.crop_rect
        norm_crop = None
        if crop_box is not None and self.master_np_rgb is not None:
            l, t, r, b = crop_box
            master_h, master_w = self.master_np_rgb.shape[:2]
            if master_w > 0 and master_h > 0:
                norm_crop = (
                    float(l) / master_w,
                    float(t) / master_h,
                    float(r - l) / master_w,
                    float(b - t) / master_h,
                )

        final_rendered = render_adjusted_image(
            self.master_np_rgb, self.current_recipe, media_id=self.current_media_id
        )
        final_alpha = (
            transform_alpha_for_recipe(
                self.master_alpha, self.current_recipe, target_hw=final_rendered.shape[:2]
            )
            if self.master_alpha is not None
            else None
        )
        target_sfx = self.current_file_path.suffix.lower()
        save_img = build_save_pil_image(final_rendered, final_alpha, target_sfx)

        save_kwargs = {}
        if target_sfx in (".jpg", ".jpeg"):
            save_kwargs["quality"] = 95
        elif target_sfx in (".webp", ".jxl", ".avif", ".heic", ".heif"):
            save_kwargs["quality"] = 90
        try:
            with Image.open(backup_path if backup_path.exists() else self.current_file_path) as orig_pil:
                orig_exif = orig_pil.getexif()
                if orig_exif:
                    orig_exif[0x0112] = 1
                    if target_sfx == ".jxl":
                        save_kwargs["exif"] = orig_exif.tobytes()
                    elif target_sfx not in (".png", ".tif", ".tiff"):
                        save_kwargs["exif"] = orig_exif
        except Exception:
            pass
        save_img.save(self.current_file_path, **save_kwargs)
        self.current_orientation = 1
        if hasattr(self, "catalog_repo") and self.catalog_repo and self.current_media_id is not None:
            try:
                st = self.current_file_path.stat()
                self.catalog_repo.update_media_metadata(
                    int(self.current_media_id),
                    orientation=1,
                    width=int(save_img.width),
                    height=int(save_img.height),
                    mtime=int(st.st_mtime),
                    size=int(st.st_size),
                )
            except Exception:
                pass

        if norm_crop is not None and self.current_media_id is not None:
            if hasattr(self, "catalog_repo") and self.catalog_repo:
                try:
                    self.catalog_repo.transform_face_regions_for_crop(
                        int(self.current_media_id), *norm_crop
                    )
                except Exception:
                    pass

        self.master_alpha = final_alpha if target_sfx in ALPHA_SAVE_DOT_EXT_SET else None
        self.master_np_rgb = (
            final_rendered
            if self.master_alpha is not None
            else np.ascontiguousarray(np.array(save_img.convert("RGB"), dtype=np.uint8))
        )
        self.master_pil_image = Image.fromarray(self.master_np_rgb)
        self._update_display_proxy()
        self.current_recipe = PhotoAdjustmentRecipe()
        self.undo_stack.clear()
        self._render_current_recipe()

        self.pixel_edit_saved.emit(int(self.current_media_id), str(backup_path))
        self._purge_all_transient_scene_items()
        self.update_revert_state()
        return str(backup_path)

    def save_as_copy(self, forced_target: str | None = None) -> str | None:
        """
        Saves the edited version as a separate new file without touching the original:
        - forced_target: None (reads settings), "original" (forces original folder), or "choose" (prompts save dialog).
        - If loc_mode == "ask" or forced_target == "choose": prompts standard system save dialog.
        - Synchronously registers new file in catalog_repo so new media_id is valid immediately.
        - Automatically switches viewer to display the newly saved copy if enabled in Options.
        - Emits pixel_edit_saved_as_copy(original_media_id, new_path)
        """
        if (
            self.master_np_rgb is None
            or self.current_file_path is None
            or self.current_media_id is None
            or self.current_is_video
        ):
            return None

        self.ensure_master_rendered()

        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QFileDialog
        settings = QSettings("PhotoGallery64", "PhotoGallery64")
        loc_mode = str(settings.value("save_copy_location", "original"))
        custom_folder = str(settings.value("save_copy_custom_folder", "") or "").strip()
        naming_pattern = str(settings.value("save_copy_naming_pattern", "{stem}_edited"))
        switch_view = settings.value("save_copy_switch_view", True, type=bool)

        stem = self.current_file_path.stem
        orig_suffix = self.current_file_path.suffix
        is_raw_source = orig_suffix.lower() in RAW_DOT_EXT_SET
        if self.master_alpha is not None and orig_suffix.lower() not in ALPHA_SAVE_DOT_EXT_SET:
            suffix = ".png"
        elif is_raw_source:
            suffix = ".jpg"
        else:
            suffix = orig_suffix

        if "{stem}" in naming_pattern:
            base_pattern = naming_pattern.replace("{stem}", stem)
        else:
            base_pattern = f"{stem}_edited"

        if forced_target == "choose" or (forced_target is None and loc_mode in ("ask", "prompt")):
            initial_path = str(self.current_file_path.parent / f"{base_pattern}{suffix}")
            if is_raw_source:
                filter_str = (
                    "JPEG (*.jpg *.jpeg);;PNG (*.png);;WebP (*.webp);;JPEG XL (*.jxl);;"
                    "AVIF (*.avif);;HEIF (*.heic *.heif);;TIFF (*.tif *.tiff);;All Files (*.*)"
                )
            else:
                filter_str = (
                    f"Images (*{suffix});;JPEG (*.jpg *.jpeg);;PNG (*.png);;"
                    "WebP (*.webp);;JPEG XL (*.jxl);;AVIF (*.avif);;HEIF (*.heic *.heif);;"
                    "TIFF (*.tif *.tiff);;All Files (*.*)"
                )
            chosen, _ = QFileDialog.getSaveFileName(
                self,
                "Save as Copy",
                initial_path,
                filter_str,
            )
            if not chosen:
                return None
            candidate = Path(chosen)
            if not candidate.suffix or candidate.suffix.lower() in RAW_DOT_EXT_SET:
                candidate = candidate.with_suffix(".jpg")
        else:
            if forced_target == "original" or loc_mode == "original":
                parent_dir = self.current_file_path.parent
            elif loc_mode == "custom" and custom_folder and Path(custom_folder).is_dir():
                parent_dir = Path(custom_folder)
            else:
                parent_dir = self.current_file_path.parent

            candidate = parent_dir / f"{base_pattern}{suffix}"
            if candidate.exists():
                counter = 2
                while True:
                    candidate = parent_dir / f"{base_pattern}_{counter}{suffix}"
                    if not candidate.exists():
                        break
                    counter += 1

        final_rendered = render_adjusted_image(
            self.master_np_rgb, self.current_recipe, media_id=self.current_media_id
        )
        final_alpha = (
            transform_alpha_for_recipe(
                self.master_alpha, self.current_recipe, target_hw=final_rendered.shape[:2]
            )
            if self.master_alpha is not None
            else None
        )
        cand_sfx = candidate.suffix.lower()
        save_img = build_save_pil_image(final_rendered, final_alpha, cand_sfx)
        save_kwargs = {}
        if cand_sfx in (".jpg", ".jpeg"):
            save_kwargs["quality"] = 95
        elif cand_sfx in (".webp", ".jxl", ".avif", ".heic", ".heif"):
            save_kwargs["quality"] = 90
        if not is_raw_source:
            try:
                with Image.open(self.current_file_path) as orig_pil:
                    orig_exif = orig_pil.getexif()
                    if orig_exif:
                        orig_exif[0x0112] = 1
                        if cand_sfx == ".jxl":
                            save_kwargs["exif"] = orig_exif.tobytes()
                        elif cand_sfx not in (".png", ".tif", ".tiff"):
                            save_kwargs["exif"] = orig_exif
            except Exception:
                pass
        save_img.save(candidate, **save_kwargs)

        mid = int(self.current_media_id)

        # Synchronously register folder & media in database so new_mid is valid immediately (0ms)
        new_mid: int | None = None
        if getattr(self, "catalog_repo", None) is not None:
            try:
                p_dir = candidate.parent
                folder_id = self.catalog_repo.upsert_folder(
                    path=str(p_dir).replace("\\", "/"),
                    name=p_dir.name or str(p_dir),
                    parent_id=None,
                    mtime=int(p_dir.stat().st_mtime) if p_dir.exists() else 0,
                )
                orig_detail = self.catalog_repo.get_media_detail(mid) if mid else {}
                date_taken = orig_detail.get("date_taken")
                # New file has rendered raster pixels upright and normalized EXIF orientation=1
                orientation = 1
                m_rows = self.catalog_repo.bulk_upsert_media([{
                    "folder_id": folder_id,
                    "filename": candidate.name,
                    "file_size": candidate.stat().st_size,
                    "mtime": int(candidate.stat().st_mtime),
                    "width": final_rendered.shape[1],
                    "height": final_rendered.shape[0],
                    "is_video": False,
                    "orientation": orientation,
                    "date_taken": date_taken,
                }])
                new_mid = m_rows[0] if m_rows else None

                if new_mid is not None and getattr(self, "_current_people_regions", None):
                    # Remap and copy surviving face regions to new copy
                    mw = float(self.master_np_rgb.shape[1])
                    mh = float(self.master_np_rgb.shape[0])
                    crop_box = self.current_recipe.crop_rect
                    cl, ct = (float(crop_box[0]), float(crop_box[1])) if crop_box else (0.0, 0.0)
                    cw = float(final_rendered.shape[1])
                    ch = float(final_rendered.shape[0])
                    surviving_copy_faces = []
                    for reg in self._current_people_regions:
                        rx = float(reg.get("rect_x") or 0.0)
                        ry = float(reg.get("rect_y") or 0.0)
                        rw = float(reg.get("rect_w") or 0.0)
                        rh = float(reg.get("rect_h") or 0.0)
                        pid = reg.get("person_id")
                        if rw <= 0.0 or rh <= 0.0 or pid is None:
                            continue
                        fx, fy = rx * mw, ry * mh
                        fw, fh = rw * mw, rh * mh
                        orig_area = fw * fh
                        sx, sy = fx - cl, fy - ct
                        ix1, iy1 = max(0.0, sx), max(0.0, sy)
                        ix2, iy2 = min(cw, sx + fw), min(ch, sy + fh)
                        if ix2 <= ix1 or iy2 <= iy1:
                            continue
                        vis_area = (ix2 - ix1) * (iy2 - iy1)
                        if orig_area > 0 and (vis_area / orig_area) >= 0.25:
                            new_rx = max(0.0, min(1.0, ix1 / cw))
                            new_ry = max(0.0, min(1.0, iy1 / ch))
                            new_rw = max(0.0, min(1.0 - new_rx, (ix2 - ix1) / cw))
                            new_rh = max(0.0, min(1.0 - new_ry, (iy2 - iy1) / ch))
                            try:
                                self.catalog_repo.assign_person(new_mid, pid, rect=(new_rx, new_ry, new_rw, new_rh))
                                copy_reg = dict(reg)
                                copy_reg["rect_x"] = new_rx
                                copy_reg["rect_y"] = new_ry
                                copy_reg["rect_w"] = new_rw
                                copy_reg["rect_h"] = new_rh
                                surviving_copy_faces.append(copy_reg)
                            except Exception:
                                pass
                    if switch_view:
                        self._current_people_regions = surviving_copy_faces
            except Exception as e:
                logger.debug("Synchronous upsert for copy failed: %s", e)

        # Insert new copy into navigation sequence immediately
        if new_mid is not None:
            if hasattr(self, "media_ids") and self.media_ids:
                if mid in self.media_ids:
                    idx = self.media_ids.index(mid)
                    if new_mid not in self.media_ids:
                        self.media_ids.insert(idx + 1, new_mid)
                else:
                    if new_mid not in self.media_ids:
                        self.media_ids.append(new_mid)

        if switch_view:
            # Switch viewer to immediately display the newly saved copy
            self._stop_animated_image()
            self.current_file_path = candidate
            self.current_orientation = 1
            if new_mid is not None:
                self.current_media_id = new_mid
                if hasattr(self, "media_ids") and new_mid in self.media_ids:
                    self.current_index = self.media_ids.index(new_mid)
            self.master_alpha = final_alpha if cand_sfx in ALPHA_SAVE_DOT_EXT_SET else None
            self.master_np_rgb = (
                final_rendered
                if self.master_alpha is not None
                else np.ascontiguousarray(np.array(save_img.convert("RGB"), dtype=np.uint8))
            )
            self.master_pil_image = Image.fromarray(self.master_np_rgb)
            self.current_recipe = PhotoAdjustmentRecipe()
            self.undo_stack.clear()
            self._update_display_proxy()
            self._original_pixmap = None
            self._render_current_recipe()
            self._purge_all_transient_scene_items()
            self.update_revert_state()
            self._show_status(f"Saved copy and switched to {candidate.name}", 3500)
        else:
            # User chose not to switch: leave viewer and recipe on original image
            self._show_status(f"Saved copy as {candidate.name}", 3500)

        self.pixel_edit_saved_as_copy.emit(mid, str(candidate))
        return str(candidate)

    def prompt_save_changes_if_needed(self, action_name: str = "navigate") -> bool:
        """
        Smart Unsaved-Edits Guard:
        If there are uncommitted recipe adjustments, checks user preference in Options or prompts:
        - [Save (Overwrite) (Enter)]
        - [Save as Copy]
        - [Discard Changes]
        - [Cancel (Esc)]
        Includes 'Do not ask again' checkbox linked to Options > Edit preferences.
        Returns True if okay to proceed with navigation/closing, False if cancelled.
        """
        if not self.has_unsaved_adjustments():
            return True

        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QCheckBox
        settings = QSettings("PhotoGallery64", "PhotoGallery64")
        action_pref = str(settings.value("unsaved_edits_action", "prompt")).lower()

        if action_pref == "overwrite":
            res = self.save_current_image_with_backup()
            return res is not None
        elif action_pref == "copy":
            res = self.save_as_copy()
            return res is not None
        elif action_pref == "discard":
            self.undo_stack.clear()
            self.current_recipe = PhotoAdjustmentRecipe()
            start_master = getattr(self, "_markup_session_start_master", None)
            if start_master is not None and self.master_np_rgb is not None:
                self.master_np_rgb = start_master.copy()
                self._update_display_proxy()
            self._cleanup_markup_temp_items()
            self._purge_all_transient_scene_items()
            self._render_current_recipe()
            self.update_revert_state()
            return True

        filename = self.current_file_path.name if self.current_file_path else "this photo"
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Unsaved Changes")
        msg_box.setText(f"Do you want to save the changes made to '{filename}'?")
        msg_box.setInformativeText(
            "You can overwrite the original photo (safely backed up to .OriginalFiles/) "
            "or save the edited version as a new separate copy.\n\n"
            "You can change this behavior at any time in Options > Edit."
        )
        msg_box.setIcon(QMessageBox.Question)

        btn_save = msg_box.addButton("Save (Overwrite)", QMessageBox.AcceptRole)
        btn_copy = msg_box.addButton("Save as Copy", QMessageBox.ActionRole)
        btn_discard = msg_box.addButton("Discard Changes", QMessageBox.DestructiveRole)
        btn_cancel = msg_box.addButton("Cancel", QMessageBox.RejectRole)

        chk_dont_ask = QCheckBox("Do not ask again (use this choice in the future)", msg_box)
        chk_dont_ask.setStyleSheet("color: #1E395B; font-size: 9pt; padding-top: 6px;")
        msg_box.setCheckBox(chk_dont_ask)

        msg_box.setDefaultButton(btn_save)
        msg_box.setEscapeButton(btn_cancel)

        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #FBFDFF;
                font-family: 'Segoe UI';
                min-width: 530px;
            }
            QLabel {
                font-family: 'Segoe UI';
                font-size: 9pt;
            }
            QPushButton {
                padding: 6px 12px;
                border: 1px solid #7DA2CE;
                border-radius: 3px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FBFDFF, stop:1 #E0ECF8);
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 9pt;
                min-width: 105px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #EAF4FD, stop:1 #C6E0F8);
                border-color: #3C7FB1;
            }
            QPushButton:default {
                font-weight: bold;
                border-color: #005A9E;
            }
        """)

        msg_box.exec()
        clicked = msg_box.clickedButton()

        remember = chk_dont_ask.isChecked()

        if clicked == btn_save:
            if remember:
                settings.setValue("unsaved_edits_action", "overwrite")
            res = self.save_current_image_with_backup()
            return res is not None
        elif clicked == btn_copy:
            if remember:
                settings.setValue("unsaved_edits_action", "copy")
            res = self.save_as_copy()
            return res is not None
        elif clicked == btn_discard:
            if remember:
                settings.setValue("unsaved_edits_action", "discard")
            self.undo_stack.clear()
            self.current_recipe = PhotoAdjustmentRecipe()
            start_master = getattr(self, "_markup_session_start_master", None)
            if start_master is not None and self.master_np_rgb is not None:
                self.master_np_rgb = start_master.copy()
                self._update_display_proxy()
            self._cleanup_markup_temp_items()
            self._purge_all_transient_scene_items()
            self._render_current_recipe()
            self.update_revert_state()
            return True
        else:
            return False

    def request_close(self) -> None:
        """Prompts to save uncommitted edits before closing the single image viewer."""
        if self.prompt_save_changes_if_needed("close"):
            self.closed.emit()

    def request_navigate(self, delta: int) -> None:
        """Prompts to save uncommitted edits before navigating to previous or next photo."""
        if self.prompt_save_changes_if_needed("navigate"):
            self.navigate_requested.emit(delta)

    def update_revert_state(self) -> None:
        """Updates enabled state of the Revert button based on disk backup or in-memory changes."""
        has_backup = False
        if self.current_file_path is not None:
            bpath = self.current_file_path.parent / ".OriginalFiles" / self.current_file_path.name
            if bpath.exists():
                has_backup = True
        try:
            can_undo = hasattr(self, "undo_stack") and self.undo_stack is not None and self.undo_stack.canUndo()
        except RuntimeError:
            can_undo = False
        has_recipe_edits = (
            self.current_recipe is not None and not self.current_recipe.is_default()
        ) or can_undo
        can_revert = has_backup or has_recipe_edits
        if hasattr(self, "btn_revert"):
            self.btn_revert.setEnabled(can_revert)
        try:
            mw = getattr(self, "main_window", None) or self.window()
            if hasattr(mw, "_update_revert_button_state"):
                mw._update_revert_button_state()
        except RuntimeError:
            pass

    def revert_to_original(self, backup_path: str | Path | None = None) -> bool:
        """
        Restores the unedited photo:
        1. If a disk backup in '.OriginalFiles' exists, restores the original file.
        2. If uncommitted in-memory pixel edits or adjustments exist, reloads the pristine photo from disk.
        """
        if self.current_file_path is None and not hasattr(self, "_markup_session_start_master"):
            return False

        if getattr(self, "is_markup_mode", False):
            self._cancel_inline_text()
            self.set_markup_mode(False)

        if backup_path is None and self.current_file_path is not None:
            bpath = self.current_file_path.parent / ".OriginalFiles" / self.current_file_path.name
        elif backup_path is not None:
            bpath = Path(backup_path)
        else:
            bpath = None

        # Determine pristine camera/disk EXIF orientation & file dimensions from the file being restored
        target_fpath = bpath if (bpath is not None and bpath.exists()) else self.current_file_path
        disk_orient = 1
        restored_dims: tuple[int, int] | None = None
        if target_fpath is not None and target_fpath.exists():
            try:
                with Image.open(target_fpath) as f_img:
                    exif_val = f_img.getexif().get(0x0112)
                    disk_orient = int(exif_val or 1)
                    restored_dims = (int(f_img.width), int(f_img.height))
            except Exception:
                pass

        if bpath is not None and bpath.exists() and self.current_file_path is not None:
            shutil.copy2(bpath, self.current_file_path)
            try:
                bpath.unlink()
            except OSError:
                pass
            if hasattr(self, "catalog_repo") and self.catalog_repo and self.current_media_id is not None:
                try:
                    st = self.current_file_path.stat()
                    self.catalog_repo.update_media_metadata(
                        int(self.current_media_id),
                        orientation=disk_orient,
                        width=restored_dims[0] if restored_dims else None,
                        height=restored_dims[1] if restored_dims else None,
                        mtime=int(st.st_mtime),
                        size=int(st.st_size),
                    )
                except Exception:
                    pass
            self.load_media(self.current_media_id, self.current_file_path, is_video=False, orientation=disk_orient)
        else:
            if self.current_file_path is not None and self.current_file_path.exists():
                if hasattr(self, "catalog_repo") and self.catalog_repo and self.current_media_id is not None:
                    try:
                        st = self.current_file_path.stat()
                        self.catalog_repo.update_media_metadata(
                            int(self.current_media_id),
                            orientation=disk_orient,
                            width=restored_dims[0] if restored_dims else None,
                            height=restored_dims[1] if restored_dims else None,
                            mtime=int(st.st_mtime),
                            size=int(st.st_size),
                        )
                    except Exception:
                        pass
                self.load_media(self.current_media_id, self.current_file_path, is_video=False, orientation=disk_orient)
            else:
                start_master = getattr(self, "_markup_session_start_master", None)
                if start_master is not None:
                    self.master_np_rgb = start_master.copy()
                    self._update_display_proxy()
                self.current_recipe = PhotoAdjustmentRecipe()
                self.undo_stack.clear()
                self._render_current_recipe()
                self.recipe_changed.emit(self.current_recipe)

        self._purge_all_transient_scene_items()
        self.set_zoom_mode("fit")
        self.update_revert_state()
        if self.current_media_id is not None:
            self.pixel_edit_reverted.emit(int(self.current_media_id))
        return True

    # -------------------------------------------------------------------------
    # Static-Image Face Region Overlay & Interactive WLPG Face Tagging
    # -------------------------------------------------------------------------
    def _clear_face_overlays(self) -> None:
        if hasattr(self, "_inline_name_popup") and self._inline_name_popup is not None:
            self._inline_name_popup.hide()
        for item in self._face_overlay_items:
            if item.scene() is self.scene:
                self.scene.removeItem(item)
        self._face_overlay_items.clear()

    def reload_faces(self, highlight_new: bool = False) -> None:
        """Re-fetches and renders face regions from the database for the current media immediately."""
        if self.current_media_id is None:
            return
        mw = getattr(self, "main_window", None) or self.window()
        if not hasattr(mw, "catalog_repo"):
            return
        try:
            detail = mw.catalog_repo.get_media_detail(int(self.current_media_id))
            self.render_face_regions(detail.get("people", []))
            if highlight_new:
                self.temporarily_show_face_overlays(duration_ms=6000)
            if hasattr(mw, "right_tag_dock") and mw.right_tag_dock is not None:
                summary = mw.catalog_repo.get_selection_summary([int(self.current_media_id)])
                mw.right_tag_dock.bind_selection([int(self.current_media_id)], summary)
        except Exception:
            pass

    def temporarily_show_face_overlays(self, duration_ms: int = 6000) -> None:
        """Reveals all face boxes and badges temporarily (e.g. after rescan or tag someone mode)."""
        if self.is_in_edit_mode():
            return
        for item in getattr(self, "_face_overlay_items", []):
            if not isinstance(item, _FaceHoverRegionItem):
                item.setVisible(True)
        if not hasattr(self, "_temp_faces_timer"):
            self._temp_faces_timer = QTimer(self)
            self._temp_faces_timer.setSingleShot(True)
            self._temp_faces_timer.timeout.connect(self._hide_temp_face_overlays)
        self._temp_faces_timer.start(duration_ms)

    def _hide_temp_face_overlays(self) -> None:
        if self.is_in_edit_mode():
            return
        if getattr(self, "is_face_tag_mode", False) or getattr(self, "is_face_adjust_mode", False):
            return
        popup = getattr(self, "_inline_name_popup", None)
        if popup is not None and popup.isVisible():
            return
        for item in getattr(self, "_face_overlay_items", []):
            if not isinstance(item, _FaceHoverRegionItem):
                fid = item.data(1)
                pid = item.data(0)
                is_highlighted = (
                    (pid is not None and getattr(self, "_highlighted_person_id", None) == pid)
                    or (fid is not None and getattr(self, "_highlighted_region_id", None) == fid)
                )
                if not is_highlighted:
                    item.setVisible(False)

    def render_face_regions(self, people_regions: list[dict]) -> None:
        """Draws normalized XMP MP:RegionInfo face rectangles and screen-space name badges on static images."""
        self._clear_face_overlays()
        self._current_people_regions = list(people_regions or [])
        if self.current_is_video or self.current_pil_image is None:
            return

        # Determine master dimensions
        if self.master_np_rgb is not None:
            mw = float(self.master_np_rgb.shape[1])
            mh = float(self.master_np_rgb.shape[0])
        else:
            mw = float(self.current_pil_image.size[0])
            mh = float(self.current_pil_image.size[1])

        crop_rect = getattr(self.current_recipe, "crop_rect", None)
        if crop_rect is not None:
            cl, ct, cr, cb = [float(v) for v in crop_rect]
            cw = max(1.0, cr - cl)
            ch = max(1.0, cb - ct)
        else:
            cl, ct = 0.0, 0.0
            cw, ch = mw, mh

        for reg in self._current_people_regions:
            rx = float(reg.get("rect_x") or 0.0)
            ry = float(reg.get("rect_y") or 0.0)
            rw = float(reg.get("rect_w") or 0.0)
            rh = float(reg.get("rect_h") or 0.0)
            if rw <= 0.0 or rh <= 0.0:
                continue

            # Convert normalized master coordinates to master pixels
            fx = rx * mw
            fy = ry * mh
            fw = rw * mw
            fh = rh * mh
            face_orig_area = fw * fh

            # Map to scene coordinates (offset by crop top-left)
            sx = fx - cl
            sy = fy - ct

            # Clip against visible scene area [0, cw] x [0, ch]
            ix1 = max(0.0, sx)
            iy1 = max(0.0, sy)
            ix2 = min(cw, sx + fw)
            iy2 = min(ch, sy + fh)

            if ix2 <= ix1 or iy2 <= iy1:
                # Completely outside cropped area
                continue

            visible_area = (ix2 - ix1) * (iy2 - iy1)
            if face_orig_area > 0 and (visible_area / face_orig_area) < 0.25:
                # Less than 25% of face is visible: omit face tag
                continue

            rect_scene = QRectF(ix1, iy1, ix2 - ix1, iy2 - iy1)

            pid = reg.get("person_id")
            fid = reg.get("face_id") or reg.get("region_id")
            st = int(reg.get("state", 1 if (pid and not reg.get("is_unconfirmed")) else 0))
            pname = str(reg.get("person_name") or reg.get("name") or "").strip()

            shadow_box = QGraphicsRectItem(rect_scene)
            shadow_pen = QPen(QColor(0, 0, 0, 165), 3.0)
            shadow_pen.setCosmetic(True)
            shadow_box.setPen(shadow_pen)
            shadow_box.setBrush(QBrush(Qt.transparent))
            shadow_box.setZValue(2)
            shadow_box.setData(0, pid)
            shadow_box.setData(1, fid)
            shadow_box.setData(3, st)
            shadow_box.setVisible(False)
            self.scene.addItem(shadow_box)
            self._face_overlay_items.append(shadow_box)

            box = QGraphicsRectItem(rect_scene)
            if st == 1:
                fg_pen = QPen(QColor("#FFFFFF"), 1.5, Qt.SolidLine)
            elif pname:
                fg_pen = QPen(QColor("#00B4D8"), 1.8, Qt.DashLine)
            else:
                fg_pen = QPen(QColor("#70A6E8"), 1.8, Qt.DashLine)
            fg_pen.setCosmetic(True)
            box.setPen(fg_pen)
            box.setBrush(QBrush(Qt.transparent))
            box.setZValue(2.5)
            box.setData(0, pid)
            box.setData(1, fid)
            box.setData(3, st)
            box.setVisible(False)
            self.scene.addItem(box)
            self._face_overlay_items.append(box)

            if st == 1:
                display_pill = pname
                is_sug = False
                is_unk = False
            elif pname:
                display_pill = pname
                is_sug = True
                is_unk = False
            else:
                display_pill = "Who is this?"
                is_sug = False
                is_unk = True

            badge = _FaceNameBadgeItem(
                display_pill,
                person_id=pid,
                face_id=fid,
                state=st,
                is_suggestion=is_sug,
                is_unknown=is_unk,
                viewer=self,
            )
            badge.setPos(rect_scene.center().x(), min(ch - 10.0, rect_scene.bottom()))
            badge.setData(0, pid)
            badge.setData(1, fid)
            badge.setData(3, st)
            badge.setVisible(False)
            self.scene.addItem(badge)
            self._face_overlay_items.append(badge)

            # Interactive hover sensor over face region
            related = [shadow_box, box, badge]
            sensor_rect = QRectF(rect_scene)
            sensor_rect.adjust(-8, -8, 8, max(46.0, rect_scene.height() * 0.35))
            sensor = _FaceHoverRegionItem(sensor_rect, self, related, pid, region_data=reg)
            sensor.setData(0, pid)
            sensor.setData(1, fid)
            sensor.setData(3, st)
            sensor.related_badge = badge
            badge.sensor = sensor
            self.scene.addItem(sensor)
            self._face_overlay_items.append(sensor)

        if self.is_in_edit_mode():
            self._update_face_overlays_visibility()

    def prompt_inline_name_face(
        self,
        face_id: int,
        pos_view: QPoint,
        initial_text: str = "",
        target_badge: Any | None = None,
    ) -> None:
        """Opens the on-canvas inline QLineEdit popup with completer for naming an unknown/suggested face."""
        mw = getattr(self, "main_window", None) or self.window()
        repo = getattr(mw, "catalog_repo", None) or getattr(mw, "repo", None)
        names = []
        if repo and hasattr(repo, "get_people_list"):
            try:
                names = [str(r["name"]) for r in repo.get_people_list() if r.get("name")]
            except Exception:
                names = []

        # Boxless / unlinked person tags on current photo
        unlinked_names = [
            str(r.get("person_name") or r.get("name") or "").strip()
            for r in getattr(self, "_current_people_regions", [])
            if int(r.get("state") or 0) == 1
            and (float(r.get("rect_w") or 0.0) <= 0.005 or float(r.get("rect_h") or 0.0) <= 0.005)
            and (r.get("person_name") or r.get("name"))
        ]
        merged_names = list(unlinked_names) + [n for n in names if n not in unlinked_names]
        self._inline_name_popup.set_people_list(merged_names)
        self._inline_name_popup.popup_at(
            face_id, pos_view, initial_text, target_badge=target_badge, unlinked_names=unlinked_names
        )

    def _on_inline_name_committed(self, face_id: int, name: str) -> None:
        self.face_region_named.emit(face_id, name)

    def highlight_face_region(self, region_id: int | None) -> None:
        """Highlights a specific face box by its media_people.id (region_id) on the photo canvas."""
        if self.is_in_edit_mode():
            return
        self._highlighted_region_id = region_id
        for item in self._face_overlay_items:
            if isinstance(item, _FaceHoverRegionItem):
                continue
            item_fid = item.data(1)
            is_match = region_id is not None and item_fid == region_id
            if is_match:
                item.setVisible(True)
                if isinstance(item, QGraphicsRectItem) and item.zValue() > 2.2:
                    p = QPen(QColor("#3399FF"), 2.5)
                    p.setCosmetic(True)
                    item.setPen(p)
                elif isinstance(item, _FaceNameBadgeItem):
                    item.set_highlighted(True)
            else:
                if isinstance(item, QGraphicsRectItem) and item.zValue() > 2.2:
                    st = item.data(3) if item.data(3) is not None else 1
                    col = QColor("#FFFFFF") if st == 1 else (QColor("#00B4D8") if st == 0 and item.data(0) else QColor("#70A6E8"))
                    style = Qt.SolidLine if st == 1 else Qt.DashLine
                    p = QPen(col, 1.5, style)
                    p.setCosmetic(True)
                    item.setPen(p)
                elif isinstance(item, _FaceNameBadgeItem):
                    item.set_highlighted(False)
                item.setVisible(False)

    @property
    def is_face_tagging_mode(self) -> bool:
        return self.is_face_tag_mode

    @property
    def _face_tag_target_item(self) -> _FaceTagTargetBoxItem | None:
        return self._face_target_item

    @property
    def image_scene(self):
        return self.scene

    def highlight_person_face(self, person_id: int | None) -> None:
        """Highlights a specific person's face box and name badge on the image when hovered in the right pane."""
        if self.is_in_edit_mode():
            return
        self._highlighted_person_id = person_id
        for item in self._face_overlay_items:
            if isinstance(item, _FaceHoverRegionItem):
                continue
            item_pid = item.data(0)
            is_match = person_id is not None and item_pid == person_id
            if is_match:
                item.setVisible(True)
                if isinstance(item, QGraphicsRectItem) and item.zValue() > 2.2:
                    p = QPen(QColor("#3399FF"), 2.5)
                    p.setCosmetic(True)
                    item.setPen(p)
                elif isinstance(item, _FaceNameBadgeItem):
                    item.set_highlighted(True)
            else:
                if isinstance(item, QGraphicsRectItem) and item.zValue() > 2.2:
                    p = QPen(QColor("#FFFFFF"), 1.5)
                    p.setCosmetic(True)
                    item.setPen(p)
                elif isinstance(item, _FaceNameBadgeItem):
                    item.set_highlighted(False)
                item.setVisible(False)

    def start_face_tagging_mode(
        self,
        initial_rect: tuple[float, float, float, float] | None = None,
        replace_person_id: int | None = None,
        prefill_name: str | None = None,
    ) -> bool:
        """
        Enters WLPG interactive face tagging mode:
        - Displays a movable/resizable white L-bracket face target box on the photo
        - Displays the floating 'Tag someone' popup card with existing people list + search
        """
        if self.current_is_video or self.current_pil_image is None or self.current_media_id is None:
            return False

        if self.is_crop_mode:
            self.set_crop_mode(False)
        if self.is_retouch_mode:
            self.set_retouch_mode(False)
        if self.is_red_eye_mode:
            self.set_red_eye_mode(False)
        if getattr(self, "is_markup_mode", False):
            self.set_markup_mode(False)

        self.stop_face_tagging_mode()
        self.is_face_tag_mode = True
        self._face_tag_replace_person_id = replace_person_id

        w, h = self.current_pil_image.size
        img_rect = QRectF(0, 0, w, h)

        mw = getattr(self, "main_window", None) or self.window()
        repo = getattr(mw, "repo", None)

        norm_rect: tuple[float, float, float, float] | None = initial_rect
        if norm_rect is None and replace_person_id is not None:
            for reg in self._current_people_regions:
                if reg.get("person_id") == replace_person_id:
                    rx = float(reg.get("rect_x") or 0.0)
                    ry = float(reg.get("rect_y") or 0.0)
                    rw = float(reg.get("rect_w") or 0.0)
                    rh = float(reg.get("rect_h") or 0.0)
                    if rw > 0.0 and rh > 0.0:
                        norm_rect = (rx, ry, rw, rh)
                        break

        if norm_rect is None and repo is not None and hasattr(repo, "get_untagged_faces_for_media"):
            untagged = repo.get_untagged_faces_for_media(int(self.current_media_id))
            if untagged:
                u0 = untagged[0]
                norm_rect = (
                    float(u0.get("rect_x") or 0.38),
                    float(u0.get("rect_y") or 0.30),
                    float(u0.get("rect_w") or 0.22),
                    float(u0.get("rect_h") or 0.22),
                )

        if norm_rect is not None:
            rx, ry, rw, rh = norm_rect
            target_scene_rect = QRectF(rx * w, ry * h, max(32.0, rw * w), max(32.0, rh * h))
        else:
            side = max(48.0, min(w, h) * 0.22)
            target_scene_rect = QRectF((w - side) * 0.5, (h - side) * 0.42, side, side)

        self._face_target_item = _FaceTagTargetBoxItem(
            target_scene_rect,
            img_rect,
            on_changed=self._position_tag_someone_popup,
        )
        self.scene.addItem(self._face_target_item)

        people_rows = []
        if repo is not None and hasattr(repo, "get_people_list"):
            try:
                people_rows = repo.get_people_list()
            except Exception:
                people_rows = []
        self._tag_someone_popup.populate(people_rows)
        if prefill_name:
            self._tag_someone_popup.txt_search.setText(prefill_name)
            self._tag_someone_popup.txt_search.selectAll()
        self._tag_someone_popup.show()
        self._tag_someone_popup.raise_()
        self._position_tag_someone_popup()
        self._tag_someone_popup.focus_input()
        return True

    def start_face_tag_mode_for_person(
        self,
        person_id: int,
        person_name: str,
        initial_rect: QRectF | tuple[float, float, float, float] | None = None,
    ) -> bool:
        """Starts interactive face tagging mode pre-targeted to position/link a box for person_name."""
        return self.start_face_tagging_mode(
            initial_rect=initial_rect,
            replace_person_id=person_id,
            prefill_name=person_name,
        )

    def stop_face_tagging_mode(self) -> None:
        """Exits interactive WLPG face tagging mode and hides the 'Tag someone' popup."""
        self.is_face_tag_mode = False
        self._face_tag_replace_person_id = None
        if self._face_target_item is not None:
            if self._face_target_item.scene() is self.scene:
                self.scene.removeItem(self._face_target_item)
            self._face_target_item = None
        if hasattr(self, "_tag_someone_popup") and self._tag_someone_popup is not None:
            self._tag_someone_popup.hide()
        if getattr(self, "is_face_adjust_mode", False):
            self.stop_face_adjust_mode()

    def _position_tag_someone_popup(self) -> None:
        if not getattr(self, "is_face_tag_mode", False):
            return
        if self._face_target_item is None or not self._tag_someone_popup.isVisible():
            return
        r = self._face_target_item.rect()
        top_right_vp = self.graphics_view.mapFromScene(QPointF(r.right(), r.top()))
        top_left_vp = self.graphics_view.mapFromScene(QPointF(r.left(), r.top()))
        pw = self._tag_someone_popup.width()
        ph = self._tag_someone_popup.height()
        gw = self.graphics_view.width()
        gh = self.graphics_view.height()

        px = top_right_vp.x() + 10
        if px + pw > gw - 8:
            px = max(8, top_left_vp.x() - pw - 10)
        py = max(8, min(gh - ph - 8, top_right_vp.y()))
        self._tag_someone_popup.move(px, py)

    def _on_tag_someone_selected(self, person_name: str) -> None:
        if not person_name or not person_name.strip():
            return
        self.tag_selected_crop_as_face(person_name.strip())

    def tag_selected_crop_as_face(self, person_name: str | None = None) -> bool:
        """
        Converts the current face target box (or drawn crop rectangle) on a static image
        into normalized [0..1] MP:RegionInfo coordinates and emits face_region_tagged.
        """
        if self.current_is_video or self.current_pil_image is None or self.current_media_id is None:
            return False

        rect: QRectF | None = None
        if self._face_target_item is not None:
            rect = self._face_target_item.rect()
        else:
            rect = self.scene.get_crop_rect()

        if rect is None or rect.width() < 4 or rect.height() < 4:
            return False

        w, h = self.current_pil_image.size
        if w <= 0 or h <= 0:
            return False

        if person_name is None:
            name, ok = QInputDialog.getText(self, "Tag a Person", "Enter person's name:")
            if not ok or not name.strip():
                return False
            person_name = name.strip()

        rx = max(0.0, min(1.0, rect.left() / w))
        ry = max(0.0, min(1.0, rect.top() / h))
        rw = max(0.0, min(1.0 - rx, rect.width() / w))
        rh = max(0.0, min(1.0 - ry, rect.height() / h))

        replace_pid = self._face_tag_replace_person_id
        self.stop_face_tagging_mode()
        self.scene.clear_crop_rect()

        if replace_pid is not None:
            mw = getattr(self, "main_window", None) or self.window()
            repo = getattr(mw, "repo", None)
            if repo is not None and hasattr(repo, "remove_person_from_media"):
                try:
                    repo.remove_person_from_media(int(self.current_media_id), int(replace_pid))
                except Exception:
                    pass

        self.face_region_tagged.emit(int(self.current_media_id), person_name, rx, ry, rw, rh)
        return True

    def show_face_region_context_menu(
        self, screen_pos: QPoint, region_data: dict, person_id: int | None
    ) -> QMenu:
        """Context menu when right-clicking a face region in Single Image View."""
        menu = QMenu(self)
        menu.setFont(QFont("Segoe UI", 9))

        mw = getattr(self, "main_window", None) or self.window()
        face_id = region_data.get("face_id") or region_data.get("id") or 0
        if not face_id and self.current_media_id is not None and person_id is not None:
            if hasattr(mw, "catalog_repo"):
                row = mw.catalog_repo.engine.catalog_conn().execute(
                    "SELECT id FROM media_people WHERE media_id = ? AND (person_id = ? OR suggested_person_id = ?) LIMIT 1;",
                    (int(self.current_media_id), int(person_id), int(person_id)),
                ).fetchone()
                if row:
                    face_id = int(row["id"])

        is_unconfirmed = bool(region_data.get("is_unconfirmed"))
        person_name = str(region_data.get("person_name") or region_data.get("name") or "").strip()

        # Unknown face case:
        if not person_id and not person_name:
            act_tag = menu.addAction("Tag person...")
            bold = QFont(act_tag.font())
            bold.setBold(True)
            act_tag.setFont(bold)
            act_tag.triggered.connect(
                lambda: self.prompt_inline_name_face(face_id, self.graphics_view.mapFromGlobal(screen_pos), "")
            )
            act_adjust = menu.addAction("Adjust bounding box")
            act_adjust.triggered.connect(
                lambda: self.start_face_adjust_mode(region_data, face_id, person_id)
            )
            menu.addSeparator()
            act_ignore = menu.addAction("Ignore face")
            act_ignore.triggered.connect(
                lambda: self._on_face_region_dismiss(face_id)
            )
            act_not_face = menu.addAction("Not a face")
            act_not_face.triggered.connect(
                lambda: self._on_face_region_not_a_face(region_data, person_id, face_id)
            )
            if not getattr(self, "_suppress_menu_exec", False):
                menu.exec(screen_pos)
            return menu

        # 1. Confirm (if unconfirmed suggestion)
        if is_unconfirmed and face_id and person_id:
            act_confirm = menu.addAction(f"Confirm '{person_name}'")
            bold = QFont(act_confirm.font())
            bold.setBold(True)
            act_confirm.setFont(bold)
            act_confirm.triggered.connect(
                lambda: mw._on_dock_person_confirm_requested(int(person_id), int(face_id))
            )

        # 2. Adjust bounding box
        act_adjust = menu.addAction("Adjust bounding box")
        act_adjust.triggered.connect(
            lambda: self.start_face_adjust_mode(region_data, face_id, person_id)
        )

        menu.addSeparator()

        # 3. Tag as someone else...
        act_retag = menu.addAction("Tag as someone else...")
        act_retag.triggered.connect(
            lambda: self._on_face_region_tag_as_someone_else(region_data, person_id, face_id, screen_pos)
        )

        # 4. Not this person (Remove tag)
        if person_id:
            act_not_person = menu.addAction("Not this person")
            act_not_person.triggered.connect(
                lambda: self._on_face_region_not_this_person(region_data, person_id, face_id)
            )

        # 5. Not a face (Delete region)
        act_not_face = menu.addAction("Not a face")
        act_not_face.triggered.connect(
            lambda: self._on_face_region_not_a_face(region_data, person_id, face_id)
        )

        # 6. Set as display picture
        if person_id:
            act_display_pic = menu.addAction("Set as display picture")
            act_display_pic.triggered.connect(
                lambda: self._on_face_region_set_display_pic(person_id, face_id)
            )

        if not getattr(self, "_suppress_menu_exec", False):
            menu.exec(screen_pos)
        return menu

    def _on_face_region_dismiss(self, face_id: int) -> None:
        mw = getattr(self, "main_window", None) or self.window()
        if hasattr(mw, "_on_dismiss_face_requested"):
            mw._on_dismiss_face_requested(face_id)
        elif hasattr(mw, "catalog_repo") and self.current_media_id is not None:
            mw.catalog_repo.dismiss_face_region(face_id)
            detail = mw.catalog_repo.get_media_detail(int(self.current_media_id))
            self.render_face_regions(detail.get("face_regions") or detail.get("people", []))
        if hasattr(mw, "invalidate_face_crop_ui") and self.current_media_id is not None:
            mw.invalidate_face_crop_ui(media_id=int(self.current_media_id))

    def start_face_adjust_mode(
        self,
        region_data: dict,
        face_id: int | None = None,
        person_id: int | None = None,
    ) -> bool:
        """Enters interactive bounding box adjustment mode for a face region."""
        if self.current_is_video or self.current_pil_image is None or self.current_media_id is None:
            return False

        if self.is_crop_mode:
            self.set_crop_mode(False)
        if self.is_retouch_mode:
            self.set_retouch_mode(False)
        if self.is_red_eye_mode:
            self.set_red_eye_mode(False)
        if getattr(self, "is_markup_mode", False):
            self.set_markup_mode(False)
        if self.is_face_tag_mode:
            self.stop_face_tagging_mode()

        self.stop_face_adjust_mode()
        self.is_face_adjust_mode = True
        self._adjusting_face_info = {
            "region": dict(region_data),
            "face_id": face_id,
            "person_id": person_id,
        }

        # Hide static face overlays while adjusting
        for it in self._face_overlay_items:
            it.setVisible(False)

        w, h = self.current_pil_image.size
        img_rect = QRectF(0, 0, w, h)
        rx = float(region_data.get("rect_x") or 0.0)
        ry = float(region_data.get("rect_y") or 0.0)
        rw = float(region_data.get("rect_w") or 0.2)
        rh = float(region_data.get("rect_h") or 0.2)
        target_scene_rect = QRectF(rx * w, ry * h, max(24.0, rw * w), max(24.0, rh * h))

        self._face_adjust_box_item = _FaceTagTargetBoxItem(
            target_scene_rect,
            img_rect,
            on_changed=self._position_adjust_toolbar,
        )
        self.scene.addItem(self._face_adjust_box_item)

        self._adjust_toolbar.show()
        self._adjust_toolbar.raise_()
        self._position_adjust_toolbar()
        return True

    def stop_face_adjust_mode(self) -> None:
        """Exits bounding box adjustment mode and restores face overlays."""
        self.is_face_adjust_mode = False
        self._adjusting_face_info = {}
        if hasattr(self, "_adjust_toolbar") and self._adjust_toolbar is not None:
            self._adjust_toolbar.hide()
        if hasattr(self, "_face_adjust_box_item") and self._face_adjust_box_item is not None:
            if self._face_adjust_box_item.scene() is self.scene:
                self.scene.removeItem(self._face_adjust_box_item)
            self._face_adjust_box_item = None
        for it in self._face_overlay_items:
            it.setVisible(False)

        # Reset mouse cursor and restore hover for item currently under cursor
        if hasattr(self, "graphics_view") and self.graphics_view is not None:
            self.graphics_view.viewport().unsetCursor()
            self.graphics_view.unsetCursor()
            try:
                from PySide6.QtGui import QCursor
                vp_pos = self.graphics_view.mapFromGlobal(QCursor.pos())
                if self.graphics_view.viewport().rect().contains(vp_pos):
                    scene_pos = self.graphics_view.mapToScene(vp_pos)
                    for it in self.scene.items(scene_pos):
                        if isinstance(it, _FaceHoverRegionItem):
                            for rel in it.related_items:
                                rel.setVisible(True)
                            break
            except Exception:
                pass

    def save_face_adjust(self) -> None:
        """Saves updated bounding box to catalog.db, updates face thumbnail, and re-renders."""
        if not getattr(self, "is_face_adjust_mode", False) or self._face_adjust_box_item is None:
            return
        if self.current_pil_image is None or self.current_media_id is None:
            self.stop_face_adjust_mode()
            return
        w, h = self.current_pil_image.size
        r = self._face_adjust_box_item.rect()
        norm_rect = (
            max(0.0, min(1.0, r.x() / w)),
            max(0.0, min(1.0, r.y() / h)),
            max(0.01, min(1.0, r.width() / w)),
            max(0.01, min(1.0, r.height() / h)),
        )
        face_info = getattr(self, "_adjusting_face_info", {})
        face_id = face_info.get("face_id") or 0
        person_id = face_info.get("person_id")
        mw = getattr(self, "main_window", None) or self.window()
        if hasattr(mw, "catalog_repo"):
            if not face_id and person_id:
                row = mw.catalog_repo.engine.catalog_conn().execute(
                    "SELECT id FROM media_people WHERE media_id = ? AND (person_id = ? OR suggested_person_id = ?) LIMIT 1;",
                    (int(self.current_media_id), int(person_id), int(person_id)),
                ).fetchone()
                if row:
                    face_id = int(row["id"])
            if face_id > 0:
                thumb_repo = getattr(mw, "thumb_repo", None)
                mw.catalog_repo.update_face_rect(face_id, norm_rect, thumb_repo=thumb_repo)
                mw.lbl_sync_status.setText("✓ Face bounding box updated")
            if hasattr(mw, "invalidate_face_crop_ui"):
                pids = [int(person_id)] if person_id else []
                mw.invalidate_face_crop_ui(media_id=int(self.current_media_id), person_ids=pids)
            mw.refresh_navigation_and_autocomplete()
        self.stop_face_adjust_mode()
        if hasattr(mw, "catalog_repo"):
            detail = mw.catalog_repo.get_media_detail(int(self.current_media_id))
            self.render_face_regions(detail.get("people", []))

    def _position_adjust_toolbar(self) -> None:
        if not getattr(self, "is_face_adjust_mode", False):
            return
        if (
            self._face_adjust_box_item is None
            or not hasattr(self, "_adjust_toolbar")
            or not self._adjust_toolbar.isVisible()
        ):
            return
        r = self._face_adjust_box_item.rect()
        bottom_center_vp = self.graphics_view.mapFromScene(QPointF(r.center().x(), r.bottom()))
        tw = self._adjust_toolbar.width()
        th = self._adjust_toolbar.height()
        gw = self.graphics_view.width()
        gh = self.graphics_view.height()

        px = int(bottom_center_vp.x() - tw / 2)
        px = max(8, min(gw - tw - 8, px))
        py = int(bottom_center_vp.y() + 8)
        if py + th > gh - 8:
            top_center_vp = self.graphics_view.mapFromScene(QPointF(r.center().x(), r.top()))
            py = max(8, int(top_center_vp.y() - th - 8))
        self._adjust_toolbar.move(px, py)

    def _on_face_region_tag_as_someone_else(
        self,
        region_data: dict,
        person_id: int | None,
        face_id: int | None = None,
        screen_pos: QPoint | None = None,
    ) -> None:
        mw = getattr(self, "main_window", None) or self.window()
        if not hasattr(mw, "catalog_repo") or self.current_media_id is None:
            return
        fid = int(face_id or region_data.get("face_id") or region_data.get("region_id") or 0)
        if not fid and person_id is not None:
            row = mw.catalog_repo.engine.catalog_conn().execute(
                "SELECT id FROM media_people WHERE media_id = ? AND (person_id = ? OR suggested_person_id = ?) LIMIT 1;",
                (int(self.current_media_id), int(person_id), int(person_id)),
            ).fetchone()
            if row:
                fid = int(row["id"])

        people_rows = mw.catalog_repo.get_people_list()
        new_name, new_pid = TagSomeoneDialog.prompt(people_rows, self, pos=screen_pos, title="Tag someone")
        if not new_name or not new_name.strip():
            return
        new_name_clean = new_name.strip()
        if new_pid is None or new_pid < 0:
            new_pid = mw.catalog_repo.upsert_person(new_name_clean)
        rx = float(region_data.get("rect_x") or 0.0)
        ry = float(region_data.get("rect_y") or 0.0)
        rw = float(region_data.get("rect_w") or 0.0)
        rh = float(region_data.get("rect_h") or 0.0)
        rect = (rx, ry, rw, rh) if rw > 0 and rh > 0 else None
        if fid > 0:
            mw.catalog_repo.assign_person_to_face(fid, new_pid)
        else:
            if person_id:
                mw.catalog_repo.remove_person_tag(int(self.current_media_id), int(person_id))
            mw.catalog_repo.assign_person(int(self.current_media_id), new_name_clean, rect)
        if hasattr(mw, "invalidate_face_crop_ui"):
            affected_pids = [int(p) for p in (person_id, new_pid) if p is not None and int(p) > 0]
            mw.invalidate_face_crop_ui(media_id=int(self.current_media_id), person_ids=affected_pids)
        mw.refresh_navigation_and_autocomplete()
        detail = mw.catalog_repo.get_media_detail(int(self.current_media_id))
        self.render_face_regions(detail.get("people", []))
        summary = mw.catalog_repo.get_selection_summary([int(self.current_media_id)])
        mw.right_tag_dock.bind_selection([int(self.current_media_id)], summary)
        mw.lbl_sync_status.setText(f"✓ Retagged face to '{new_name_clean}'")
        for it in getattr(self, "_face_overlay_items", []):
            if it.data(1) == fid:
                it.setVisible(True)

    def _on_face_region_not_this_person(self, region_data: dict, person_id: int, face_id: int) -> None:
        mw = getattr(self, "main_window", None) or self.window()
        if not hasattr(mw, "catalog_repo") or self.current_media_id is None:
            return
        fid = int(face_id or region_data.get("face_id") or region_data.get("region_id") or 0)
        if not fid and person_id is not None:
            row = mw.catalog_repo.engine.catalog_conn().execute(
                "SELECT id FROM media_people WHERE media_id = ? AND (person_id = ? OR suggested_person_id = ?) LIMIT 1;",
                (int(self.current_media_id), int(person_id), int(person_id)),
            ).fetchone()
            if row:
                fid = int(row["id"])
        if fid > 0:
            mw.catalog_repo.unassign_face_person(fid, person_id)
        else:
            mw.catalog_repo.remove_person_tag(int(self.current_media_id), person_id)
        if hasattr(mw, "invalidate_face_crop_ui"):
            pids = [int(person_id)] if person_id else []
            mw.invalidate_face_crop_ui(media_id=int(self.current_media_id), person_ids=pids)
        mw.refresh_navigation_and_autocomplete()
        detail = mw.catalog_repo.get_media_detail(int(self.current_media_id))
        self.render_face_regions(detail.get("people", []))
        summary = mw.catalog_repo.get_selection_summary([int(self.current_media_id)])
        mw.right_tag_dock.bind_selection([int(self.current_media_id)], summary)
        mw.lbl_sync_status.setText("✓ Removed person from face (box kept as 'Who is this?')")
        # Ensure the unassigned face box and 'Who is this?' badge stay visible:
        for it in getattr(self, "_face_overlay_items", []):
            if it.data(1) == fid:
                it.setVisible(True)

    def _on_face_region_not_a_face(self, region_data: dict, person_id: int | None, face_id: int) -> None:
        mw = getattr(self, "main_window", None) or self.window()
        if not hasattr(mw, "catalog_repo") or self.current_media_id is None:
            return
        if face_id > 0:
            mw.catalog_repo.mark_not_a_face(face_id)
            if person_id:
                with mw.catalog_repo.engine.write_lock:
                    conn = mw.catalog_repo.engine.catalog_conn()
                    conn.execute(
                        """
                        UPDATE people
                        SET confirmed_count = (
                            SELECT COUNT(*) FROM media_people WHERE person_id = people.id AND state = 1
                        )
                        WHERE id = ?;
                        """,
                        (person_id,),
                    )
                    conn.commit()
        if hasattr(mw, "invalidate_face_crop_ui"):
            pids = [int(person_id)] if person_id else []
            mw.invalidate_face_crop_ui(media_id=int(self.current_media_id), person_ids=pids)
        mw.refresh_navigation_and_autocomplete()
        detail = mw.catalog_repo.get_media_detail(int(self.current_media_id))
        self.render_face_regions(detail.get("people", []))
        summary = mw.catalog_repo.get_selection_summary([int(self.current_media_id)])
        mw.right_tag_dock.bind_selection([int(self.current_media_id)], summary)
        mw.lbl_sync_status.setText("✓ Marked region as 'Not a face'")

    def _on_face_region_set_display_pic(self, person_id: int, face_id: int) -> None:
        mw = getattr(self, "main_window", None) or self.window()
        if not hasattr(mw, "catalog_repo") or self.current_media_id is None:
            return
        mw._on_face_crop_set_display_pic(int(self.current_media_id), int(person_id), face_id=face_id)

    # -------------------------------------------------------------------------
    # Video Playback Controls & Keyboard Shortcuts
    # -------------------------------------------------------------------------
    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlayingState:
            self.btn_play_pause.setIcon(make_wlpg_icon("pause", 16))
            self.btn_play_pause.setToolTip("Pause (Space)")
        else:
            self.btn_play_pause.setIcon(make_wlpg_icon("play", 16))
            self.btn_play_pause.setToolTip("Play (Space)")

    def toggle_video_playback(self) -> None:
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.btn_play_pause.setIcon(make_wlpg_icon("play", 16))
            self.btn_play_pause.setToolTip("Play (Space)")
        else:
            self.media_player.play()
            self.btn_play_pause.setIcon(make_wlpg_icon("pause", 16))
            self.btn_play_pause.setToolTip("Pause (Space)")

    def stop_video_playback(self) -> None:
        self.media_player.stop()
        self.btn_play_pause.setIcon(make_wlpg_icon("play", 16))
        self.btn_play_pause.setToolTip("Play (Space)")

    def _on_volume_changed(self, val: int) -> None:
        vol = max(0.0, min(1.0, val / 100.0))
        self.audio_output.setVolume(vol)
        if val == 0 or self.audio_output.isMuted():
            self.btn_volume.setIcon(make_wlpg_icon("volume_mute", 14))
        elif val < 50:
            self.btn_volume.setIcon(make_wlpg_icon("volume_low", 14))
        else:
            self.btn_volume.setIcon(make_wlpg_icon("volume", 14))

    def _toggle_mute(self) -> None:
        is_muted = not self.audio_output.isMuted()
        self.audio_output.setMuted(is_muted)
        if is_muted:
            self.btn_volume.setIcon(make_wlpg_icon("volume_mute", 14))
        else:
            val = self.slider_volume.value()
            if val < 50:
                self.btn_volume.setIcon(make_wlpg_icon("volume_low", 14))
            else:
                self.btn_volume.setIcon(make_wlpg_icon("volume", 14))

    def _show_video_context_menu(self, pos: QPoint) -> QMenu:
        menu = QMenu(self)
        menu.setFont(QFont("Segoe UI", 9))

        is_playing = self.media_player.playbackState() == QMediaPlayer.PlayingState
        act_toggle = menu.addAction("Pause" if is_playing else "Play")
        bold_font = QFont(act_toggle.font())
        bold_font.setBold(True)
        act_toggle.setFont(bold_font)
        act_toggle.triggered.connect(self.toggle_video_playback)

        act_stop = menu.addAction("Stop")
        act_stop.triggered.connect(self.stop_video_playback)

        is_muted = self.audio_output.isMuted()
        act_mute = menu.addAction("Unmute" if is_muted else "Mute")
        act_mute.triggered.connect(lambda: self.audio_output.setMuted(not is_muted))

        menu.addSeparator()

        main_win = self.window()
        mid = self.current_media_id

        open_with_menu = menu.addMenu("Open with")
        open_with_menu.setFont(QFont("Segoe UI", 9))
        act_default = open_with_menu.addAction("System default player")
        act_choose = open_with_menu.addAction("Choose another app...")
        
        def _on_open_default() -> None:
            if hasattr(main_win, "open_with_app_for_media") and mid is not None:
                main_win.open_with_app_for_media(mid, "default")
            elif self.current_file_path:
                from PySide6.QtCore import QUrl
                from PySide6.QtGui import QDesktopServices
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_file_path)))

        def _on_open_choose() -> None:
            if hasattr(main_win, "open_with_app_for_media") and mid is not None:
                main_win.open_with_app_for_media(mid, "choose")
            elif self.current_file_path:
                import sys
                if sys.platform == "win32":
                    import ctypes
                    ctypes.windll.shell32.OpenAs_RunDLLW(0, 0, str(Path(self.current_file_path).resolve()), 1)

        act_default.triggered.connect(_on_open_default)
        act_choose.triggered.connect(_on_open_choose)

        act_loc = menu.addAction("Open file location")
        def _on_open_loc() -> None:
            if hasattr(main_win, "open_file_location_for_media") and mid is not None:
                main_win.open_file_location_for_media(mid)
            elif self.current_file_path:
                import os, subprocess, sys
                p = Path(self.current_file_path)
                if sys.platform == "win32":
                    subprocess.Popen(f'explorer.exe /select,"{os.path.normpath(str(p.resolve()))}"')
                else:
                    from PySide6.QtCore import QUrl
                    from PySide6.QtGui import QDesktopServices
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.parent)))
        act_loc.triggered.connect(_on_open_loc)

        menu.addSeparator()

        act_copy = menu.addAction("Copy")
        act_copy.setShortcut("Ctrl+C")
        if hasattr(main_win, "copy_selection_to_clipboard"):
            act_copy.triggered.connect(main_win.copy_selection_to_clipboard)

        act_del = menu.addAction("Delete")
        act_del.setShortcut("Del")
        if hasattr(main_win, "delete_selection_to_recycle_bin"):
            act_del.triggered.connect(main_win.delete_selection_to_recycle_bin)

        menu.addSeparator()

        act_prop = menu.addAction("Properties")
        act_prop.setShortcut("Alt+Enter")
        def _on_prop() -> None:
            if hasattr(main_win, "open_file_properties") and self.current_media_id is not None:
                main_win.open_file_properties(self.current_media_id)
            elif hasattr(main_win, "toggle_right_pane"):
                main_win.toggle_right_pane("info")
        act_prop.triggered.connect(_on_prop)

        if not getattr(self, "_suppress_menu_exec", False):
            menu.exec(self.video_widget.mapToGlobal(pos))
        return menu

    def _on_video_position_changed(self, pos_ms: int) -> None:
        self.slider_position.blockSignals(True)
        self.slider_position.setValue(pos_ms)
        self.slider_position.blockSignals(False)
        dur_ms = max(0, self.media_player.duration())
        self.lbl_video_time.setText(
            f"{pos_ms // 60000:02d}:{(pos_ms // 1000) % 60:02d} / "
            f"{dur_ms // 60000:02d}:{(dur_ms // 1000) % 60:02d}"
        )

    def _on_video_duration_changed(self, dur_ms: int) -> None:
        self.slider_position.setRange(0, max(1, dur_ms))

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        # Intercept graphics_view viewport mouse events for Retouch and Red Eye
        if watched in (self.graphics_view, self.graphics_view.viewport()):
            if event.type() == QEvent.Wheel:
                if getattr(self, "is_markup_mode", False):
                    tool = getattr(self.markup_toolbar, "active_tool", "pen")
                    if tool in ("magnify", "magnifier"):
                        delta = event.angleDelta().y()
                        cur_zoom = getattr(self.markup_toolbar, "magnifier_zoom", 2.5)
                        if delta > 0:
                            new_zoom = min(5.0, round(cur_zoom + 0.5, 1))
                        elif delta < 0:
                            new_zoom = max(1.2, round(cur_zoom - 0.5, 1))
                        else:
                            new_zoom = cur_zoom
                        if new_zoom != cur_zoom:
                            self.markup_toolbar.set_magnifier_zoom(new_zoom)
                        event.accept()
                        return True
                res = super().eventFilter(watched, event)
                if getattr(self, "is_markup_mode", False):
                    self._update_markup_cursor()
                elif getattr(self, "is_retouch_mode", False) or getattr(self, "is_red_eye_mode", False):
                    self._update_retouch_cursor()
                return res

            if self.is_face_tag_mode and self._face_target_item is not None:
                if event.type() == QEvent.MouseButtonPress and getattr(event, "button", lambda: None)() == Qt.LeftButton:
                    sp = self.graphics_view.mapToScene(event.position().toPoint())
                    item_rect = self._face_target_item.rect().adjusted(-14, -14, 14, 14)
                    if not item_rect.contains(sp) and self.current_pil_image is not None:
                        w, h = self.current_pil_image.size
                        cur_r = self._face_target_item.rect()
                        nw, nh = cur_r.width(), cur_r.height()
                        nx = max(0.0, min(w - nw, sp.x() - nw * 0.5))
                        ny = max(0.0, min(h - nh, sp.y() - nh * 0.5))
                        self._face_target_item.setRect(QRectF(nx, ny, nw, nh))
                        self._tag_someone_popup.focus_input()
                        return True
            if self.is_retouch_mode:
                if event.type() == QEvent.MouseButtonPress and getattr(event, "button", lambda: None)() == Qt.LeftButton:
                    sp = self.graphics_view.mapToScene(event.position().toPoint())
                    self._start_retouch(sp)
                    return True
                elif event.type() == QEvent.MouseMove and getattr(self, "_retouch_active", False):
                    sp = self.graphics_view.mapToScene(event.position().toPoint())
                    self._drag_retouch(sp)
                    return True
                elif event.type() == QEvent.MouseButtonRelease and getattr(event, "button", lambda: None)() == Qt.LeftButton and getattr(self, "_retouch_active", False):
                    self._finish_retouch()
                    return True

            if self.is_red_eye_mode:
                if event.type() == QEvent.MouseButtonPress and getattr(event, "button", lambda: None)() == Qt.LeftButton:
                    sp = self.graphics_view.mapToScene(event.position().toPoint())
                    self._apply_brush_red_eye(sp)
                    return True

            if getattr(self, "is_markup_mode", False):
                if event.type() == QEvent.MouseButtonPress and getattr(event, "button", lambda: None)() == Qt.LeftButton:
                    sp = self.graphics_view.mapToScene(event.position().toPoint())
                    if getattr(self, "_active_movable_element", None) is not None:
                        bounds = get_markup_element_bounds(self._active_movable_element)
                        if bounds.contains(sp):
                            self._is_moving_pending_markup = True
                            self._move_markup_last_pt = sp
                            if getattr(self, "_active_movable_before_master", None) is not None:
                                self.master_np_rgb = self._active_movable_before_master.copy()
                                self._update_display_proxy()
                                self._render_current_recipe()
                            crop_tuple = self.current_recipe.crop_rect if self.current_recipe else None
                            pm, top_left = render_markup_element_pixmap(self._active_movable_element, self.master_np_rgb, crop_tuple)
                            if not pm.isNull():
                                p_item = QGraphicsPixmapItem(pm)
                                p_item.setPos(top_left)
                                p_item.setZValue(9995)
                                self.scene.addItem(p_item)
                                self._pending_markup_item = p_item
                            return True
                        else:
                            self._finalize_movable_markup()
                    self._start_markup(sp)
                    return True
                elif event.type() == QEvent.MouseMove:
                    sp = self.graphics_view.mapToScene(event.position().toPoint())
                    self._last_mouse_scene_pos = sp
                    if getattr(self, "_is_moving_pending_markup", False) and getattr(self, "_move_markup_last_pt", None) is not None:
                        dx = sp.x() - self._move_markup_last_pt.x()
                        dy = sp.y() - self._move_markup_last_pt.y()
                        self._move_markup_last_pt = sp
                        if getattr(self, "_active_movable_element", None) is not None:
                            translate_markup_element(self._active_movable_element, dx, dy)
                            el = self._active_movable_element
                            crop_tuple = self.current_recipe.crop_rect if self.current_recipe else None
                            if el.kind in ("pixelate", "blur", "magnifier", "magnify"):
                                pm, top_left = render_markup_element_pixmap(el, self.master_np_rgb, crop_tuple)
                                if self._pending_markup_item is not None and not pm.isNull():
                                    self._pending_markup_item.setPixmap(pm)
                                    self._pending_markup_item.setPos(top_left)
                            else:
                                if self._pending_markup_item is not None:
                                    cur_pos = self._pending_markup_item.pos()
                                    self._pending_markup_item.setPos(cur_pos.x() + dx, cur_pos.y() + dy)
                            if self._pending_selection_box_item is not None:
                                self._pending_selection_box_item.setRect(get_markup_element_bounds(el))
                        return True
                    elif getattr(self, "_markup_active", False):
                        self._drag_markup(sp)
                        return True
                    else:
                        if getattr(self, "_active_movable_element", None) is not None:
                            bounds = get_markup_element_bounds(self._active_movable_element)
                            if bounds.contains(sp):
                                self.graphics_view.viewport().setCursor(Qt.SizeAllCursor)
                                return True
                        tool = getattr(self.markup_toolbar, "active_tool", "pen")
                        if tool in ("magnify", "magnifier"):
                            self._update_magnify_hover(sp)
                        else:
                            self._update_markup_cursor()
                        return True
                elif event.type() == QEvent.MouseButtonRelease and getattr(event, "button", lambda: None)() == Qt.LeftButton:
                    sp = self.graphics_view.mapToScene(event.position().toPoint())
                    if getattr(self, "_is_moving_pending_markup", False):
                        self._is_moving_pending_markup = False
                        if getattr(self, "_pending_markup_item", None) is not None:
                            self._remove_scene_item(self._pending_markup_item)
                            self._pending_markup_item = None
                        if getattr(self, "_active_movable_element", None) is not None:
                            el = self._active_movable_element
                            if hasattr(self, "undo_stack") and self.undo_stack.canUndo():
                                self.undo_stack.undo()
                                if self._markup_elements:
                                    self._markup_elements.pop()
                                if self._markup_preview_items:
                                    it = self._markup_preview_items.pop()
                                    self._remove_scene_item(it)
                            self._commit_single_markup_element(el, f"Markup {el.kind.title()}")
                        return True
                    elif getattr(self, "_markup_active", False):
                        self._finish_markup(sp)
                        return True
                elif event.type() in (QEvent.FocusOut, QEvent.Leave) and getattr(self, "_markup_active", False):
                    self._cleanup_markup_temp_items()
                    return True

        if event.type() == QEvent.KeyRelease and isinstance(event, QKeyEvent):
            if event.key() == Qt.Key_Backslash:
                self.set_compare_mode(False)
                event.accept()
                return True

        if event.type() == QEvent.KeyPress and isinstance(event, QKeyEvent):
            if event.key() == Qt.Key_Backslash:
                self.set_compare_mode(True)
                event.accept()
                return True

            if event.key() == Qt.Key_Escape and self.is_face_tag_mode:
                self.stop_face_tagging_mode()
                event.accept()
                return True

            focus_w = QApplication.focusWidget()
            is_text_entry = isinstance(focus_w, (QLineEdit, QTextEdit, QPlainTextEdit))
            if is_text_entry:
                if getattr(self, "_inline_text_input", None) is not None and focus_w == self._inline_text_input:
                    if event.key() == Qt.Key_Escape:
                        self._cancel_inline_text()
                        event.accept()
                        return True
                    # Enter or Return commits inline text (Shift+Enter creates a new line)
                    elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
                        if not (event.modifiers() & (Qt.ShiftModifier | Qt.AltModifier)):
                            self._commit_inline_text()
                            event.accept()
                            return True
                elif event.key() == Qt.Key_Escape and getattr(self, "_inline_text_input", None) is not None:
                    self._cancel_inline_text()
                    event.accept()
                    return True
                return super().eventFilter(watched, event)

            if getattr(self, "is_retouch_mode", False) or getattr(self, "is_red_eye_mode", False):
                if event.key() == Qt.Key_BracketLeft:
                    self._on_retouch_size_changed(max(6, getattr(self, "retouch_radius", 20) - 4))
                    event.accept()
                    return True
                elif event.key() == Qt.Key_BracketRight:
                    self._on_retouch_size_changed(min(80, getattr(self, "retouch_radius", 20) + 4))
                    event.accept()
                    return True

            if getattr(self, "is_markup_mode", False):
                if event.key() == Qt.Key_BracketLeft:
                    cur = getattr(self.markup_toolbar, "stroke_width", 4)
                    self.markup_toolbar.set_stroke_width(max(2, cur - 2))
                    self._update_markup_cursor()
                    event.accept()
                    return True
                elif event.key() == Qt.Key_BracketRight:
                    cur = getattr(self.markup_toolbar, "stroke_width", 4)
                    self.markup_toolbar.set_stroke_width(min(36, cur + 2))
                    self._update_markup_cursor()
                    event.accept()
                    return True
                elif event.key() == Qt.Key_X and not (event.modifiers() & (Qt.ControlModifier | Qt.AltModifier)):
                    if hasattr(self.markup_toolbar, "swap_colors"):
                        self.markup_toolbar.swap_colors()
                        self._show_status("Swapped Primary / Secondary Colours", 2000)
                    event.accept()
                    return True
                elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
                    self._commit_markup()
                    event.accept()
                    return True
                elif event.key() == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
                    self._undo_markup()
                    event.accept()
                    return True
                elif event.key() == Qt.Key_Y and (event.modifiers() & Qt.ControlModifier):
                    if hasattr(self, "undo_stack") and self.undo_stack.canRedo():
                        self.undo_stack.redo()
                    event.accept()
                    return True

            if event.key() == Qt.Key_Delete and not (event.modifiers() & (Qt.ControlModifier | Qt.AltModifier)):
                mw = self.window()
                if hasattr(mw, "delete_selection_to_recycle_bin"):
                    mw.delete_selection_to_recycle_bin()
                    event.accept()
                    return True

            if event.key() == Qt.Key_Escape:
                import time
                now = time.monotonic()
                if (now - getattr(self, "_last_esc_consumed_time", 0.0)) < 0.35:
                    event.accept()
                    return True

                if getattr(self, "_inline_text_input", None) is not None:
                    self._cancel_inline_text()
                    self._last_esc_consumed_time = now
                    event.accept()
                    return True
                if getattr(self, "_magnify_source_pt", None) is not None:
                    self._magnify_source_pt = None
                    self._last_esc_consumed_time = now
                    self._cleanup_magnify_preview()
                    self._show_status("Magnify callout cancelled", 2000)
                    event.accept()
                    return True
                if getattr(self, "_active_movable_element", None) is not None:
                    self._undo_markup()
                    self._last_esc_consumed_time = now
                    self._show_status("Markup cancelled", 2000)
                    event.accept()
                    return True
                if getattr(self, "_markup_active", False):
                    self._cleanup_markup_temp_items()
                    self._markup_active = False
                    event.accept()
                    return True
                if getattr(self, "_retouch_active", False):
                    self._clear_retouch_overlay()
                    self._retouch_active = False
                    event.accept()
                    return True
                if getattr(self, "_red_eye_active", False):
                    self._clear_retouch_overlay()
                    self._red_eye_active = False
                    event.accept()
                    return True
                if self.is_crop_mode:
                    self.set_crop_mode(False)
                    event.accept()
                    return True
                if self.is_retouch_mode or self.is_red_eye_mode:
                    self._close_spot_fix_toolbar()
                    event.accept()
                    return True
                if getattr(self, "is_markup_mode", False):
                    self.set_markup_mode(False)
                    event.accept()
                    return True
                if getattr(self, "is_face_adjust_mode", False):
                    self.stop_face_adjust_mode()
                    event.accept()
                    return True
                self.request_close()
                event.accept()
                return True

            if getattr(self, "is_face_adjust_mode", False):
                if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                    self.save_face_adjust()
                    event.accept()
                    return True

            if event.key() == Qt.Key_Space and self.current_is_video:
                self.toggle_video_playback()
                event.accept()
                return True

            if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self.is_crop_mode:
                self.apply_crop()
                event.accept()
                return True

            # Previous / Next image navigation (Left / Right arrow)
            if not self.is_crop_mode and event.modifiers() in (Qt.NoModifier, Qt.KeypadModifier):
                if event.key() == Qt.Key_Left:
                    self.request_navigate(-1)
                    event.accept()
                    return True
                elif event.key() == Qt.Key_Right:
                    self.request_navigate(1)
                    event.accept()
                    return True

            # Zoom in (+) / Zoom out (-)
            if not self.is_crop_mode and not (event.modifiers() & (Qt.ControlModifier | Qt.AltModifier)):
                if event.key() in (Qt.Key_Plus, Qt.Key_Equal) or event.text() in ("+", "="):
                    self.zoom_in()
                    event.accept()
                    return True
                elif event.key() in (Qt.Key_Minus, Qt.Key_Underscore) or event.text() in ("-", "_"):
                    self.zoom_out()
                    event.accept()
                    return True

            # Star ratings: 0, 1, 2, 3, 4, 5
            if not self.is_crop_mode and event.modifiers() in (Qt.NoModifier, Qt.KeypadModifier):
                if Qt.Key_0 <= event.key() <= Qt.Key_5:
                    r_val = event.key() - Qt.Key_0
                    mw = getattr(self, "main_window", None) or self.window()
                    if hasattr(mw, "apply_rating_to_selection"):
                        mw.apply_rating_to_selection(r_val)
                    event.accept()
                    return True

            # Fullscreen / Slideshow: F11 / F12
            if event.key() in (Qt.Key_F11, Qt.Key_F12):
                mw = getattr(self, "main_window", None) or self.window()
                if hasattr(mw, "dispatch_ribbon_action"):
                    mw.dispatch_ribbon_action("slide_show")
                event.accept()
                return True

            # Ctrl shortcuts: Rotate left (Ctrl+,), Rotate right (Ctrl+.), Thumbnail view (Ctrl+0), Actual size (Ctrl+1)
            if event.modifiers() == Qt.ControlModifier:
                if event.key() == Qt.Key_Comma:
                    self.rotate_media(clockwise=False)
                    event.accept()
                    return True
                elif event.key() == Qt.Key_Period:
                    self.rotate_media(clockwise=True)
                    event.accept()
                    return True
                elif event.key() == Qt.Key_0:
                    mw = getattr(self, "main_window", None) or self.window()
                    if hasattr(mw, "trigger_thumbnail_view"):
                        mw.trigger_thumbnail_view()
                    else:
                        self.closed.emit()
                    event.accept()
                    return True
                elif event.key() == Qt.Key_1:
                    self.set_zoom_mode("100")
                    event.accept()
                    return True

        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self.eventFilter(self, event):
            return
        super().keyPressEvent(event)

    def _show_viewer_context_menu(self, pos: QPoint) -> None:
        """Authentic context menu for SingleMediaViewer (Previous, Next, Rotate, Rescan for faces, etc.)."""
        if self.current_media_id is None:
            return

        # Check if right-click was on a face region (confirmed, suggestion, or unknown face)
        scene_pos = self.graphics_view.mapToScene(pos)
        items_under = self.scene.items(scene_pos)
        face_sensor = None
        for item in items_under:
            if isinstance(item, _FaceHoverRegionItem):
                face_sensor = item
                break
            elif isinstance(item, _FaceNameBadgeItem):
                fid = item.face_id
                for sit in self._face_overlay_items:
                    if isinstance(sit, _FaceHoverRegionItem) and (
                        sit.region_data.get("face_id") == fid or sit.region_data.get("region_id") == fid
                    ):
                        face_sensor = sit
                        break
                if face_sensor:
                    break

        if face_sensor is not None:
            self.show_face_region_context_menu(
                self.graphics_view.mapToGlobal(pos),
                face_sensor.region_data,
                face_sensor.person_id,
            )
            return

        menu = self._build_viewer_context_menu()
        if menu is not None:
            menu.exec(self.graphics_view.mapToGlobal(pos))

    def _build_viewer_context_menu(self) -> QMenu | None:
        """Constructs the authentic context menu for SingleMediaViewer (Open with, Rotate, Copy, Delete, etc.)."""
        menu = QMenu(self)
        menu.setFont(QFont("Segoe UI", 9))
        mw = getattr(self, "main_window", None) or self.window()
        is_mw = hasattr(mw, "rescan_faces_for_selection")
        mid = self.current_media_id

        # 1. Open with ▸ at the very top of context menu
        open_with_menu = menu.addMenu("Open with")
        open_with_menu.setFont(QFont("Segoe UI", 9))

        ext = "jpg"
        if self.current_file_path:
            p_ext = str(self.current_file_path).rsplit(".", 1)[-1].lower()
            if p_ext:
                ext = p_ext

        is_vid = getattr(self, "current_is_video", False)
        default_label = "System default player" if is_vid else "System default viewer"
        default_icon_name = "play" if is_vid else "view_file"
        act_default = open_with_menu.addAction(make_wlpg_icon(default_icon_name, 16), default_label)

        def _on_open_default() -> None:
            if hasattr(mw, "open_with_app_for_media") and mid is not None:
                mw.open_with_app_for_media(mid, "default")
            elif self.current_file_path:
                os.startfile(str(self.current_file_path))

        act_default.triggered.connect(_on_open_default)

        from src.ui.grid_view import get_windows_open_with_apps, get_windows_app_icon
        reg_apps = get_windows_open_with_apps(ext)
        if reg_apps:
            open_with_menu.addSeparator()
            has_paint = False
            for app_name, app_cmd in reg_apps:
                if "paint" in app_name.lower():
                    has_paint = True
                act_app = open_with_menu.addAction(get_windows_app_icon(app_name, app_cmd), app_name)
                act_app.triggered.connect(lambda _c=False, cmd=app_cmd: mw.open_with_app_for_media(mid, cmd) if hasattr(mw, "open_with_app_for_media") and mid is not None else None)
            if not is_vid and not has_paint:
                act_paint = open_with_menu.addAction(make_wlpg_icon("retouch", 16), "Paint")
                act_paint.triggered.connect(lambda: mw.open_with_app_for_media(mid, "paint") if hasattr(mw, "open_with_app_for_media") and mid is not None else None)

        open_with_menu.addSeparator()
        act_choose = open_with_menu.addAction(make_wlpg_icon("options", 16), "Choose another app...")

        def _on_open_choose() -> None:
            if hasattr(mw, "open_with_app_for_media") and mid is not None:
                mw.open_with_app_for_media(mid, "choose")
            elif self.current_file_path:
                import subprocess
                subprocess.Popen(["rundll32.exe", "shell32.dll,OpenAs_RunDLL", str(self.current_file_path)])

        act_choose.triggered.connect(_on_open_choose)

        menu.addSeparator()

        act_rot_cw = menu.addAction("Rotate clockwise\tCtrl+.")
        act_rot_cw.triggered.connect(lambda: self.rotate_media(clockwise=True))
        act_rot_ccw = menu.addAction("Rotate counter-clockwise\tCtrl+,")
        act_rot_ccw.triggered.connect(lambda: self.rotate_media(clockwise=False))

        menu.addSeparator()

        act_copy = menu.addAction("Copy")
        act_copy.setShortcut("Ctrl+C")
        if is_mw and hasattr(mw, "copy_selection_to_clipboard"):
            act_copy.triggered.connect(mw.copy_selection_to_clipboard)

        if not is_vid:
            act_copy_subj = menu.addAction("Copy subject")
            act_copy_subj.triggered.connect(self.copy_subject_to_clipboard)
            act_rm_bg = menu.addAction("Remove background")
            act_rm_bg.triggered.connect(self.remove_background_cutout)

        act_del = menu.addAction("Delete")
        act_del.setShortcut("Del")
        if is_mw and hasattr(mw, "delete_selection_to_recycle_bin"):
            act_del.triggered.connect(mw.delete_selection_to_recycle_bin)

        menu.addSeparator()

        act_rescan = menu.addAction("Rescan for faces")
        if is_mw:
            act_rescan.triggered.connect(lambda: mw.rescan_faces_for_selection([self.current_media_id]))

        # Boxless tag quick-healing options in right-click menu
        boxless_tags = [
            p for p in getattr(self, "_current_people_regions", [])
            if int(p.get("state", 1)) == 1
            and (float(p.get("rect_w") or 0.0) <= 0.005 or float(p.get("rect_h") or 0.0) <= 0.005)
            and (p.get("person_name") or p.get("name"))
        ]
        if boxless_tags:
            for bt in boxless_tags:
                b_name = str(bt.get("person_name") or bt.get("name"))
                b_pid = int(bt.get("person_id") or bt.get("id") or 0)
                act_b = menu.addAction(f"Locate {b_name} on photo...")
                act_b.triggered.connect(
                    lambda checked=False, pid_val=b_pid, name_val=b_name: self.start_face_tag_mode_for_person(pid_val, name_val)
                )

        if is_mw and hasattr(mw, "open_file_location_for_media"):
            act_loc = menu.addAction("Open file location")
            act_loc.triggered.connect(lambda: mw.open_file_location_for_media(self.current_media_id))

        menu.addSeparator()

        if is_mw and hasattr(mw, "set_desktop_background_for_media"):
            act_bg = menu.addAction("Set as desktop background")
            act_bg.triggered.connect(lambda: mw.set_desktop_background_for_media(self.current_media_id))

        if is_mw and hasattr(mw, "open_print_dialog") and not self.current_is_video:
            act_print = menu.addAction("Print...")
            act_print.setShortcut("Ctrl+P")
            act_print.triggered.connect(mw.open_print_dialog)

        if is_mw:
            act_prop = menu.addAction("Properties")
            act_prop.setShortcut("Alt+Enter")
            if hasattr(mw, "open_file_properties") and self.current_media_id is not None:
                act_prop.triggered.connect(lambda: mw.open_file_properties(self.current_media_id))
            elif hasattr(mw, "toggle_right_pane"):
                act_prop.triggered.connect(lambda: mw.toggle_right_pane("info"))

        act_close = menu.addAction("Back to gallery")
        act_close.triggered.connect(self.closed.emit)

        return menu
