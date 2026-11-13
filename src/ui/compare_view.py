"""
PhotoGallery 64 — Multi-Photo Compare / Light Table View (`CompareMediaView`).
Supports 2-Up (Side-by-Side / Top-Bottom), 3-Up, and 2x2 Quad layouts,
normalized [0.0..1.0] pan/zoom synchronization with Shift-nudge override,
1-click multi-face "Zoom to Face", Laplacian sharpness scoring + Focus Peaking,
Pin Champion & Cycle Challengers filmstrip, and inline culling controls.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QImageReader,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

try:
    from src.imaging.image_loader import RAW_EXT_SET, decode_wic_to_srgb, register_nextgen_codecs

    register_nextgen_codecs()
    _RAW_EXTS = set(RAW_EXT_SET) | {
        "arw", "cr2", "cr3", "nef", "dng", "raf", "rw2", "orf", "ori", "pef",
        "srw", "sr2", "srf", "mrw", "kdc", "dcr", "erf", "mef", "mos",
        "nrw", "raw", "rwl", "x3f", "3fr", "fff", "iiq",
    }
except Exception:
    decode_wic_to_srgb = None  # type: ignore[assignment]
    _RAW_EXTS = {
        "arw", "cr2", "cr3", "nef", "dng", "raf", "rw2", "orf", "ori", "pef",
        "srw", "sr2", "srf", "mrw", "kdc", "dcr", "erf", "mef", "mos",
        "nrw", "raw", "rwl", "x3f", "3fr", "fff", "iiq",
    }


def _decode_media_image(file_path: Path | str, max_dim: int = 2400) -> QImage:
    """Decode an image (including camera RAW, raw-in-TIFF, next-gen, and WIC formats) with EXIF auto-orientation."""
    path_str = str(file_path)
    if not path_str or not os.path.exists(path_str):
        return QImage()

    def _scale_if_needed(img: QImage) -> QImage:
        if not img.isNull() and max_dim > 0 and max(img.width(), img.height()) > max_dim:
            return img.scaled(
                max_dim,
                max_dim,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        return img

    ext = Path(path_str).suffix.lstrip(".").lower()

    # 1. Direct rawpy path for known camera RAW extensions
    if ext in _RAW_EXTS:
        try:
            import rawpy
            from PySide6.QtCore import QBuffer, QByteArray, QIODevice

            with rawpy.imread(path_str) as raw:
                qimg: QImage | None = None
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        ba = QByteArray(thumb.data)
                        buf = QBuffer(ba)
                        buf.open(QIODevice.OpenModeFlag.ReadOnly)
                        t_reader = QImageReader(buf)
                        t_reader.setAutoTransform(True)
                        qimg = t_reader.read()
                        buf.close()
                    elif thumb.format == rawpy.ThumbFormat.BITMAP:
                        th, tw, ch = thumb.data.shape
                        qimg = QImage(thumb.data.data, tw, th, ch * tw, QImage.Format.Format_RGB888).copy()
                except Exception:
                    qimg = None

                if qimg is None or qimg.isNull():
                    try:
                        rgb = raw.postprocess(half_size=True, use_camera_wb=True)
                    except Exception:
                        rgb = raw.postprocess(half_size=True, use_auto_wb=True)
                    rh, rw, ch = rgb.shape
                    qimg = QImage(rgb.data, rw, rh, ch * rw, QImage.Format.Format_RGB888).copy()

                if qimg is not None and not qimg.isNull():
                    return _scale_if_needed(qimg)
        except Exception as exc:
            logger.debug("rawpy direct decode failed for %s: %s", path_str, exc)

    # 2. QImageReader with EXIF auto-orientation
    try:
        reader = QImageReader(path_str)
        reader.setAutoTransform(True)
        orig_size = reader.size()
        if orig_size.isValid() and max_dim > 0:
            max_side = max(orig_size.width(), orig_size.height())
            if max_side > max_dim:
                scale = max_dim / float(max_side)
                reader.setScaledSize(
                    QSize(max(1, int(orig_size.width() * scale)), max(1, int(orig_size.height() * scale)))
                )
        img = reader.read()
        if not img.isNull():
            return img
    except Exception:
        pass

    # 3. Pillow with next-gen codec plugins and EXIF transpose
    try:
        from PIL import Image, ImageOps

        with Image.open(path_str) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGBA")
            if max(pil_img.width, pil_img.height) > max_dim > 0:
                pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            data = pil_img.tobytes("raw", "RGBA")
            qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format.Format_RGBA8888).copy()
            if not qimg.isNull():
                return qimg
    except Exception:
        pass

    # 4. Fallback for camera RAW files named as .tiff / .tif or unhandled formats via rawpy
    try:
        import rawpy

        with rawpy.imread(path_str) as raw:
            try:
                rgb = raw.postprocess(half_size=True, use_camera_wb=True)
            except Exception:
                rgb = raw.postprocess(half_size=True, use_auto_wb=True)
            rh, rw, ch = rgb.shape
            qimg = QImage(rgb.data, rw, rh, ch * rw, QImage.Format.Format_RGB888).copy()
            if not qimg.isNull():
                return _scale_if_needed(qimg)
    except Exception:
        pass

    # 5. Native Windows Imaging Component (WIC) fallback
    if decode_wic_to_srgb is not None:
        try:
            rgb_wic = decode_wic_to_srgb(path_str)
            if rgb_wic is not None:
                rh, rw, ch = rgb_wic.shape
                qimg = QImage(rgb_wic.data, rw, rh, ch * rw, QImage.Format.Format_RGB888).copy()
                if not qimg.isNull():
                    return _scale_if_needed(qimg)
        except Exception:
            pass

    return QImage()


def _qimage_to_rgb_array(qimg: QImage) -> np.ndarray | None:
    if qimg.isNull():
        return None
    conv = qimg.convertToFormat(QImage.Format.Format_RGB888)
    w, h = conv.width(), conv.height()
    if w <= 0 or h <= 0:
        return None
    bpl = conv.bytesPerLine()
    ptr = conv.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8, count=bpl * h).reshape((h, bpl))
    return np.array(arr[:, : w * 3].reshape((h, w, 3)), copy=True, order="C")


def compute_laplacian_sharpness(
    qimg: QImage,
    face_boxes: Sequence[tuple[float, float, float, float]] | None = None,
) -> float:
    """
    Compute Laplacian variance sharpness score on the primary face region if present,
    otherwise on the central 50% crop.
    """
    rgb = _qimage_to_rgb_array(qimg)
    if rgb is None or rgb.size == 0:
        return 0.0

    h, w, _ = rgb.shape
    if h < 4 or w < 4:
        return 0.0

    if face_boxes:
        rx, ry, rw, rh = face_boxes[0]
        x0 = max(0, min(w - 2, int(rx * w)))
        y0 = max(0, min(h - 2, int(ry * h)))
        x1 = max(x0 + 2, min(w, int((rx + rw) * w)))
        y1 = max(y0 + 2, min(h, int((ry + rh) * h)))
        crop = rgb[y0:y1, x0:x1]
    else:
        x0, x1 = w // 4, max(w // 4 + 2, (3 * w) // 4)
        y0, y1 = h // 4, max(h // 4 + 2, (3 * h) // 4)
        crop = rgb[y0:y1, x0:x1]

    if crop.size == 0 or crop.ndim != 3 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return 0.0

    try:
        crop = np.ascontiguousarray(crop)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        # Normalize analysis width so scores are comparable across different resolutions
        if max(gray.shape) > 512:
            scale = 512.0 / float(max(gray.shape))
            gray = cv2.resize(
                gray,
                (max(2, int(gray.shape[1] * scale)), max(2, int(gray.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return round(score, 1)
    except Exception:
        return 0.0


def compute_focus_peaking_overlay(qimg: QImage) -> QImage:
    """Generate a transparent RGBA overlay highlighting high-frequency in-focus edges in neon cyan-green."""
    rgb = _qimage_to_rgb_array(qimg)
    if rgb is None or rgb.size == 0:
        return QImage()

    try:
        h, w, _ = rgb.shape
        max_side = max(h, w)
        work = rgb
        if max_side > 1200:
            scale = 1200.0 / float(max_side)
            work = cv2.resize(
                rgb,
                (max(4, int(w * scale)), max(4, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.cvtColor(np.ascontiguousarray(work), cv2.COLOR_RGB2GRAY)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        p95 = float(np.percentile(lap, 96.5))
        thresh = max(18.0, p95)
        mask = lap >= thresh

        overlay = np.zeros((work.shape[0], work.shape[1], 4), dtype=np.uint8)
        overlay[mask] = (0, 255, 170, 225)
        out = QImage(
            overlay.data,
            overlay.shape[1],
            overlay.shape[0],
            overlay.shape[1] * 4,
            QImage.Format.Format_RGBA8888,
        )
        return out.copy()
    except Exception:
        return QImage()


def compute_clipping_overlay(qimg: QImage, high_thresh: int = 250, low_thresh: int = 5) -> QImage:
    """Return an RGBA overlay highlighting clipped highlights (red) and crushed shadows (blue)."""
    rgb = _qimage_to_rgb_array(qimg)
    if rgb is None or rgb.size == 0:
        return QImage()
    try:
        h, w, _ = rgb.shape
        work = rgb
        if max(w, h) > 1024:
            scale = 1024.0 / float(max(w, h))
            work = cv2.resize(
                rgb,
                (max(16, int(w * scale)), max(16, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        overlay = np.zeros((work.shape[0], work.shape[1], 4), dtype=np.uint8)
        hi_mask = np.max(work, axis=2) >= int(high_thresh)
        lo_mask = np.all(work <= int(low_thresh), axis=2)
        overlay[lo_mask] = (40, 125, 255, 195)
        overlay[hi_mask] = (255, 45, 45, 200)
        out = QImage(
            overlay.data,
            overlay.shape[1],
            overlay.shape[0],
            overlay.shape[1] * 4,
            QImage.Format.Format_RGBA8888,
        )
        return out.copy()
    except Exception:
        return QImage()


def detect_faces_fallback(qimg: QImage) -> list[tuple[float, float, float, float]]:
    """Fast OpenCV Haar cascade fallback returning normalized (rx, ry, rw, rh) boxes."""
    rgb = _qimage_to_rgb_array(qimg)
    if rgb is None or rgb.size == 0:
        return []
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if not os.path.exists(cascade_path):
            return []
        clf = cv2.CascadeClassifier(cascade_path)
        if clf.empty():
            return []
        h, w, _ = rgb.shape
        scale = 1.0
        work = rgb
        if max(w, h) > 640:
            scale = 640.0 / float(max(w, h))
            work = cv2.resize(rgb, (max(16, int(w * scale)), max(16, int(h * scale))))
        gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
        rects = clf.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(24, 24))
        wh, ww = gray.shape
        boxes = [
            (float(x) / ww, float(y) / wh, float(bw) / ww, float(bh) / wh)
            for (x, y, bw, bh) in rects
        ]
        boxes.sort(key=lambda b: b[0])
        return boxes
    except Exception:
        return []


def _format_exif_summary(detail: dict[str, Any]) -> dict[str, str]:
    """Return structured EXIF parts for difference highlighting."""
    make = str(detail.get("camera_make") or "").strip()
    model = str(detail.get("camera_model") or "").strip()
    if make and model:
        camera = model if model.lower().startswith(make.lower()) else f"{make} {model}"
    else:
        camera = model or make
    exp = str(detail.get("exposure_time") or "").strip()
    if exp and not exp.endswith("s"):
        exp = f"{exp}s"
    fnum = detail.get("f_number")
    ap = f"f/{float(fnum):g}" if fnum not in (None, "", 0, 0.0) else ""
    iso_val = detail.get("iso")
    iso = f"ISO {int(iso_val)}" if iso_val not in (None, "", 0) else ""
    fl = detail.get("focal_length")
    focal = f"{float(fl):g}mm" if fl not in (None, "", 0, 0.0) else ""
    return {
        "camera": camera,
        "exposure": exp,
        "aperture": ap,
        "iso": iso,
        "focal": focal,
    }


class _CompareCanvas(QWidget):
    """
    Zoomable, pannable image canvas with normalized [0.0..1.0] viewport coordinates,
    Shift-nudge override, Focus Peaking overlay, and face-box zoom support.
    """

    viewport_changed = Signal(float, float, float)  # cx_norm, cy_norm, zoom_factor
    focused = Signal()

    def __init__(self, is_dark: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_dark = is_dark
        self._raw_qimg: QImage = QImage()
        self._qimg: QImage = QImage()
        self._pixmap: QPixmap = QPixmap()
        self._rotation_deg: int = 0
        self._peaking_pixmap: QPixmap | None = None
        self._focus_peaking: bool = False
        self._clipping_pixmap: QPixmap | None = None
        self._clipping_warnings: bool = False
        self._cx_norm: float = 0.5
        self._cy_norm: float = 0.5
        self._zoom_factor: float = 1.0  # 1.0 = fit to pane
        self._fit_mode: bool = True
        self._dragging: bool = False
        self._drag_last_pos = QPointF()
        self._drag_shift: bool = False
        self._face_boxes: list[tuple[float, float, float, float]] = []
        self._active_face_idx: int = -1

        self.setMouseTracking(True)
        self.setMinimumSize(140, 120)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def set_dark_mode(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        self.update()

    def set_image(
        self,
        qimg: QImage,
        face_boxes: Sequence[tuple[float, float, float, float]] = (),
        rotation_deg: int = 0,
    ) -> None:
        self._raw_qimg = qimg
        self._rotation_deg = int(rotation_deg) % 360
        self._rebuild_rotated_pixmap()
        self._face_boxes = list(face_boxes)
        self._active_face_idx = -1
        self.update()

    def _rebuild_rotated_pixmap(self) -> None:
        if self._raw_qimg.isNull():
            self._qimg = QImage()
            self._pixmap = QPixmap()
        elif self._rotation_deg:
            from PySide6.QtGui import QTransform

            self._qimg = self._raw_qimg.transformed(
                QTransform().rotate(self._rotation_deg),
                Qt.TransformationMode.SmoothTransformation,
            )
            self._pixmap = QPixmap.fromImage(self._qimg)
        else:
            self._qimg = self._raw_qimg
            self._pixmap = QPixmap.fromImage(self._qimg)
        self._peaking_pixmap = None
        self._clipping_pixmap = None
        if self._focus_peaking and not self._qimg.isNull():
            ov = compute_focus_peaking_overlay(self._qimg)
            self._peaking_pixmap = QPixmap.fromImage(ov) if not ov.isNull() else QPixmap()
        if self._clipping_warnings and not self._qimg.isNull():
            ov = compute_clipping_overlay(self._qimg)
            self._clipping_pixmap = QPixmap.fromImage(ov) if not ov.isNull() else QPixmap()

    def rotate_by(self, delta_deg: int) -> int:
        self._rotation_deg = (self._rotation_deg + int(delta_deg)) % 360
        self._rebuild_rotated_pixmap()
        self.update()
        return self._rotation_deg

    def set_face_boxes(self, face_boxes: Sequence[tuple[float, float, float, float]]) -> None:
        self._face_boxes = list(face_boxes)
        self.update()

    def set_focus_peaking(self, enabled: bool) -> None:
        self._focus_peaking = bool(enabled)
        if self._focus_peaking and self._peaking_pixmap is None and not self._qimg.isNull():
            ov = compute_focus_peaking_overlay(self._qimg)
            self._peaking_pixmap = QPixmap.fromImage(ov) if not ov.isNull() else QPixmap()
        self.update()

    def set_clipping_warnings(self, enabled: bool) -> None:
        self._clipping_warnings = bool(enabled)
        if self._clipping_warnings and self._clipping_pixmap is None and not self._qimg.isNull():
            ov = compute_clipping_overlay(self._qimg)
            self._clipping_pixmap = QPixmap.fromImage(ov) if not ov.isNull() else QPixmap()
        self.update()

    def viewport_state(self) -> tuple[float, float, float, bool]:
        return (self._cx_norm, self._cy_norm, self._zoom_factor, self._fit_mode)

    def set_normalized_viewport(
        self,
        cx_norm: float,
        cy_norm: float,
        zoom_factor: float,
        *,
        fit_mode: bool | None = None,
    ) -> None:
        self._cx_norm = max(0.0, min(1.0, float(cx_norm)))
        self._cy_norm = max(0.0, min(1.0, float(cy_norm)))
        self._zoom_factor = max(0.25, min(24.0, float(zoom_factor)))
        if fit_mode is not None:
            self._fit_mode = fit_mode
        else:
            self._fit_mode = self._zoom_factor <= 1.001 and abs(self._cx_norm - 0.5) < 1e-3 and abs(self._cy_norm - 0.5) < 1e-3
        self.update()

    def zoom_fit(self) -> tuple[float, float, float]:
        self._cx_norm = 0.5
        self._cy_norm = 0.5
        self._zoom_factor = 1.0
        self._fit_mode = True
        self._active_face_idx = -1
        self.update()
        return (self._cx_norm, self._cy_norm, self._zoom_factor)

    def zoom_actual_1_to_1(self) -> tuple[float, float, float]:
        if self._pixmap.isNull():
            return (0.5, 0.5, 1.0)
        fit_scale = self._compute_fit_scale()
        target_zoom = max(1.0, 1.0 / max(1e-4, fit_scale))
        self._zoom_factor = min(24.0, target_zoom)
        self._fit_mode = False
        self.update()
        return (self._cx_norm, self._cy_norm, self._zoom_factor)

    def zoom_to_face_index(self, face_idx: int) -> bool:
        if not self._face_boxes:
            return False
        idx = face_idx % len(self._face_boxes)
        self._active_face_idx = idx
        rx, ry, rw, rh = self._face_boxes[idx]
        cx = max(0.0, min(1.0, rx + rw * 0.5))
        cy = max(0.0, min(1.0, ry + rh * 0.5))
        # Comfortable ~1.8x head/eyes framing margin relative to fit viewport
        box_span = max(0.04, max(rw, rh) * 1.8)
        target_zoom = max(1.5, min(12.0, 1.0 / box_span))
        self.set_normalized_viewport(cx, cy, target_zoom, fit_mode=False)
        return True

    def _compute_fit_scale(self) -> float:
        if self._pixmap.isNull():
            return 1.0
        pw = max(1, self._pixmap.width())
        ph = max(1, self._pixmap.height())
        vw = max(1, self.width())
        vh = max(1, self.height())
        return min(vw / pw, vh / ph)

    def _target_draw_rect(self) -> QRectF:
        if self._pixmap.isNull():
            return QRectF()
        pw = float(self._pixmap.width())
        ph = float(self._pixmap.height())
        vw = float(self.width())
        vh = float(self.height())
        fit_scale = self._compute_fit_scale()
        scale = fit_scale if self._fit_mode else fit_scale * self._zoom_factor
        dw = pw * scale
        dh = ph * scale
        left = (vw * 0.5) - (self._cx_norm * dw)
        top = (vh * 0.5) - (self._cy_norm * dh)
        return QRectF(left, top, dw, dh)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        bg = QColor("#0F1318") if self._is_dark else QColor("#E2E8F0")
        painter.fillRect(self.rect(), bg)

        if self._pixmap.isNull():
            painter.setPen(QColor("#94A3B8") if self._is_dark else QColor("#475569"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded")
            painter.end()
            return

        target_rect = self._target_draw_rect()
        src_rect = QRectF(0, 0, self._pixmap.width(), self._pixmap.height())
        painter.drawPixmap(target_rect, self._pixmap, src_rect)

        if self._focus_peaking and self._peaking_pixmap is not None and not self._peaking_pixmap.isNull():
            ov_src = QRectF(0, 0, self._peaking_pixmap.width(), self._peaking_pixmap.height())
            painter.drawPixmap(target_rect, self._peaking_pixmap, ov_src)

        if self._clipping_warnings and self._clipping_pixmap is not None and not self._clipping_pixmap.isNull():
            cl_src = QRectF(0, 0, self._clipping_pixmap.width(), self._clipping_pixmap.height())
            painter.drawPixmap(target_rect, self._clipping_pixmap, cl_src)

        # Subtle highlight box on active zoomed face
        if 0 <= self._active_face_idx < len(self._face_boxes) and not self._fit_mode:
            rx, ry, rw, rh = self._face_boxes[self._active_face_idx]
            face_rect = QRectF(
                target_rect.left() + rx * target_rect.width(),
                target_rect.top() + ry * target_rect.height(),
                rw * target_rect.width(),
                rh * target_rect.height(),
            )
            painter.setPen(QPen(QColor("#38BDF8"), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(face_rect)

        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.focused.emit()
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_last_pos = event.position()
            self._drag_shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging and not self._pixmap.isNull():
            cur = event.position()
            dx = cur.x() - self._drag_last_pos.x()
            dy = cur.y() - self._drag_last_pos.y()
            self._drag_last_pos = cur
            rect = self._target_draw_rect()
            if rect.width() > 1 and rect.height() > 1:
                self._fit_mode = False
                self._cx_norm = max(0.0, min(1.0, self._cx_norm - dx / rect.width()))
                self._cy_norm = max(0.0, min(1.0, self._cy_norm - dy / rect.height()))
                self.update()
                is_shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier) or self._drag_shift
                if not is_shift:
                    self.viewport_changed.emit(self._cx_norm, self._cy_norm, self._zoom_factor)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.focused.emit()
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.18 if delta > 0 else (1.0 / 1.18)
        new_zoom = max(0.25, min(24.0, self._zoom_factor * factor))
        self._fit_mode = False
        self._zoom_factor = new_zoom
        self.update()
        if not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.viewport_changed.emit(self._cx_norm, self._cy_norm, self._zoom_factor)
        event.accept()


class _CompareWipeCanvas(QWidget):
    """
    Before/After Split-Wipe curtain canvas comparing Slot A (left) and Slot B (right)
    with a draggable vertical divider and synchronized pan/zoom.
    """

    def __init__(self, is_dark: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_dark = is_dark
        self._qimg_a = QImage()
        self._qimg_b = QImage()
        self._pix_a = QPixmap()
        self._pix_b = QPixmap()
        self._name_a = "A"
        self._name_b = "B"
        self._peak_a: QPixmap | None = None
        self._peak_b: QPixmap | None = None
        self._clip_a: QPixmap | None = None
        self._clip_b: QPixmap | None = None
        self._focus_peaking = False
        self._clipping_warnings = False
        self._cx_norm = 0.5
        self._cy_norm = 0.5
        self._zoom_factor = 1.0
        self._fit_mode = True
        self._wipe_ratio = 0.5
        self._dragging_wipe = False
        self._dragging_pan = False
        self._drag_last_pos = QPointF()
        self.setMouseTracking(True)
        self.setMinimumSize(200, 160)

    def set_dark_mode(self, is_dark: bool) -> None:
        self._is_dark = bool(is_dark)
        self.update()

    def set_images(
        self,
        qimg_a: QImage,
        qimg_b: QImage,
        name_a: str = "A",
        name_b: str = "B",
        rot_a: int = 0,
        rot_b: int = 0,
    ) -> None:
        from PySide6.QtGui import QTransform

        if rot_a % 360 and not qimg_a.isNull():
            qimg_a = qimg_a.transformed(QTransform().rotate(rot_a % 360), Qt.TransformationMode.SmoothTransformation)
        if rot_b % 360 and not qimg_b.isNull():
            qimg_b = qimg_b.transformed(QTransform().rotate(rot_b % 360), Qt.TransformationMode.SmoothTransformation)
        self._qimg_a = qimg_a
        self._qimg_b = qimg_b
        self._pix_a = QPixmap.fromImage(qimg_a) if not qimg_a.isNull() else QPixmap()
        self._pix_b = QPixmap.fromImage(qimg_b) if not qimg_b.isNull() else QPixmap()
        self._name_a = name_a or "A"
        self._name_b = name_b or "B"
        self._peak_a = None
        self._peak_b = None
        self._clip_a = None
        self._clip_b = None
        self._refresh_overlays()
        self.update()

    def _refresh_overlays(self) -> None:
        if self._focus_peaking:
            if self._peak_a is None and not self._qimg_a.isNull():
                ov = compute_focus_peaking_overlay(self._qimg_a)
                self._peak_a = QPixmap.fromImage(ov) if not ov.isNull() else QPixmap()
            if self._peak_b is None and not self._qimg_b.isNull():
                ov = compute_focus_peaking_overlay(self._qimg_b)
                self._peak_b = QPixmap.fromImage(ov) if not ov.isNull() else QPixmap()
        if self._clipping_warnings:
            if self._clip_a is None and not self._qimg_a.isNull():
                ov = compute_clipping_overlay(self._qimg_a)
                self._clip_a = QPixmap.fromImage(ov) if not ov.isNull() else QPixmap()
            if self._clip_b is None and not self._qimg_b.isNull():
                ov = compute_clipping_overlay(self._qimg_b)
                self._clip_b = QPixmap.fromImage(ov) if not ov.isNull() else QPixmap()

    def set_focus_peaking(self, enabled: bool) -> None:
        self._focus_peaking = bool(enabled)
        self._refresh_overlays()
        self.update()

    def set_clipping_warnings(self, enabled: bool) -> None:
        self._clipping_warnings = bool(enabled)
        self._refresh_overlays()
        self.update()

    def zoom_fit(self) -> None:
        self._cx_norm = 0.5
        self._cy_norm = 0.5
        self._zoom_factor = 1.0
        self._fit_mode = True
        self.update()

    def zoom_actual_1_to_1(self) -> None:
        ref = self._pix_a if not self._pix_a.isNull() else self._pix_b
        if ref.isNull():
            return
        fit = min(max(1, self.width()) / max(1, ref.width()), max(1, self.height()) / max(1, ref.height()))
        self._zoom_factor = max(0.25, min(24.0, 1.0 / max(1e-4, fit)))
        self._fit_mode = False
        self.update()

    def zoom_to_face(self, face_box: tuple[float, float, float, float]) -> None:
        rx, ry, rw, rh = face_box
        self._cx_norm = rx + rw * 0.5
        self._cy_norm = ry + rh * 0.5
        self._zoom_factor = max(1.5, min(12.0, 0.45 / max(0.05, rh)))
        self._fit_mode = False
        self.update()

    def _target_rect_for(self, pix: QPixmap) -> QRectF:
        if pix.isNull():
            return QRectF()
        vw = max(1.0, float(self.width()))
        vh = max(1.0, float(self.height()))
        iw = float(pix.width())
        ih = float(pix.height())
        fit_scale = min(vw / iw, vh / ih)
        scale = fit_scale if self._fit_mode else fit_scale * self._zoom_factor
        dw = iw * scale
        dh = ih * scale
        left = vw * 0.5 - self._cx_norm * dw
        top = vh * 0.5 - self._cy_norm * dh
        return QRectF(left, top, dw, dh)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor("#14171D" if self._is_dark else "#E8ECF1"))

        split_x = max(8.0, min(float(self.width() - 8), self._wipe_ratio * float(self.width())))

        # Left side: A
        if not self._pix_a.isNull():
            painter.save()
            painter.setClipRect(QRectF(0, 0, split_x, float(self.height())))
            ra = self._target_rect_for(self._pix_a)
            painter.drawPixmap(ra, self._pix_a, QRectF(0, 0, self._pix_a.width(), self._pix_a.height()))
            if self._focus_peaking and self._peak_a and not self._peak_a.isNull():
                painter.drawPixmap(ra, self._peak_a, QRectF(0, 0, self._peak_a.width(), self._peak_a.height()))
            if self._clipping_warnings and self._clip_a and not self._clip_a.isNull():
                painter.drawPixmap(ra, self._clip_a, QRectF(0, 0, self._clip_a.width(), self._clip_a.height()))
            painter.restore()

        # Right side: B
        if not self._pix_b.isNull():
            painter.save()
            painter.setClipRect(QRectF(split_x, 0, float(self.width()) - split_x, float(self.height())))
            rb = self._target_rect_for(self._pix_b)
            painter.drawPixmap(rb, self._pix_b, QRectF(0, 0, self._pix_b.width(), self._pix_b.height()))
            if self._focus_peaking and self._peak_b and not self._peak_b.isNull():
                painter.drawPixmap(rb, self._peak_b, QRectF(0, 0, self._peak_b.width(), self._peak_b.height()))
            if self._clipping_warnings and self._clip_b and not self._clip_b.isNull():
                painter.drawPixmap(rb, self._clip_b, QRectF(0, 0, self._clip_b.width(), self._clip_b.height()))
            painter.restore()

        # Divider line & handle
        painter.setPen(QPen(QColor("#38BDF8"), 2.0))
        painter.drawLine(QPointF(split_x, 0), QPointF(split_x, float(self.height())))
        cy = float(self.height()) * 0.5
        painter.setBrush(QColor("#0F172A"))
        painter.setPen(QPen(QColor("#38BDF8"), 2.0))
        painter.drawEllipse(QPointF(split_x, cy), 14, 14)
        painter.setPen(QColor("#F8FAFC"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(QRectF(split_x - 14, cy - 14, 28, 28), Qt.AlignmentFlag.AlignCenter, "◀▶")

        # Corner A / B pills
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        for label, x_pos in ((f"A: {self._name_a}", 10), (f"B: {self._name_b}", max(10, self.width() - 210))):
            pill = QRectF(x_pos, 10, 196, 22)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(15, 23, 42, 185))
            painter.drawRoundedRect(pill, 4, 4)
            painter.setPen(QColor("#F8FAFC"))
            painter.drawText(pill.adjusted(6, 0, -6, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label[:28])

        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            split_x = self._wipe_ratio * float(max(1, self.width()))
            if abs(event.position().x() - split_x) <= 14:
                self._dragging_wipe = True
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self._dragging_pan = True
                self._drag_last_pos = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        cur = event.position()
        if self._dragging_wipe:
            self._wipe_ratio = max(0.02, min(0.98, cur.x() / max(1.0, float(self.width()))))
            self.update()
            event.accept()
            return
        if self._dragging_pan:
            dx = cur.x() - self._drag_last_pos.x()
            dy = cur.y() - self._drag_last_pos.y()
            self._drag_last_pos = cur
            ref = self._pix_a if not self._pix_a.isNull() else self._pix_b
            rect = self._target_rect_for(ref)
            if rect.width() > 1 and rect.height() > 1:
                self._fit_mode = False
                self._cx_norm = max(0.0, min(1.0, self._cx_norm - dx / rect.width()))
                self._cy_norm = max(0.0, min(1.0, self._cy_norm - dy / rect.height()))
                self.update()
            event.accept()
            return
        split_x = self._wipe_ratio * float(max(1, self.width()))
        self.setCursor(Qt.CursorShape.SplitHCursor if abs(cur.x() - split_x) <= 14 else Qt.CursorShape.OpenHandCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_wipe = False
            self._dragging_pan = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self._wipe_ratio = 0.5
        self.update()
        event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self._fit_mode = False
        self._zoom_factor = max(0.25, min(24.0, self._zoom_factor * factor))
        self.update()
        event.accept()


class _ComparePane(QFrame):
    """Single photo comparison pane with translucent header, canvas, sharpness badge, and culling footer."""

    focused = Signal(int)                # slot_index
    pin_toggled = Signal(int, bool)      # slot_index, is_pinned
    rating_changed = Signal(int, int)    # media_id, rating
    flag_toggled = Signal(int, int)      # media_id, new_flag
    dismiss_requested = Signal(int)      # slot_index
    delete_requested = Signal(int)       # slot_index
    rotate_requested = Signal(int, int)  # slot_index, delta_deg
    viewport_changed = Signal(int, float, float, float)  # slot_index, cx, cy, zoom

    def __init__(self, slot_index: int, is_dark: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.slot_index = slot_index
        self._is_dark = is_dark
        self.media_id: int | None = None
        self.detail: dict[str, Any] = {}
        self.exif_parts: dict[str, str] = {}
        self.sharpness_score: float = 0.0
        self.is_pinned: bool = False
        self.is_focused: bool = False
        self._rating: int = 0
        self._flag: int = 0

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._build_ui()
        self.set_dark_mode(is_dark)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top translucent header bar
        self.header = QWidget(self)
        self.header.setFixedHeight(34)
        h_lay = QHBoxLayout(self.header)
        h_lay.setContentsMargins(8, 3, 8, 3)
        h_lay.setSpacing(6)

        self.btn_pin = QPushButton("📌 Pin", self.header)
        self.btn_pin.setCheckable(True)
        self.btn_pin.setFixedHeight(24)
        self.btn_pin.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_pin.setToolTip(
            "Pin this pane as your reference ('champion') photo.\n"
            "While pinned, Left/Right arrow keys or clicking the filmstrip cycles the other pane through remaining candidates."
        )
        self.btn_pin.clicked.connect(lambda checked: self.pin_toggled.emit(self.slot_index, checked))
        h_lay.addWidget(self.btn_pin)

        self.lbl_title = QLabel("Empty Slot", self.header)
        self.lbl_title.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        h_lay.addWidget(self.lbl_title, 1)

        self.lbl_exif = QLabel("", self.header)
        self.lbl_exif.setFont(QFont("Segoe UI", 8))
        self.lbl_exif.setToolTip("Camera & exposure details (highlighted in amber where panes differ)")
        h_lay.addWidget(self.lbl_exif)

        self.lbl_sharpest = QLabel("★ Sharpest", self.header)
        self.lbl_sharpest.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        self.lbl_sharpest.setVisible(False)
        self.lbl_sharpest.setToolTip(
            "Highest Laplacian sharpness score among visible panes (face-weighted when faces are detected)."
        )
        self.lbl_sharpest.setStyleSheet(
            "background-color: #16A34A; color: #FFFFFF; border-radius: 9px; padding: 2px 8px;"
        )
        h_lay.addWidget(self.lbl_sharpest)

        self.lbl_sharp_score = QLabel("", self.header)
        self.lbl_sharp_score.setFont(QFont("Segoe UI", 8))
        self.lbl_sharp_score.setToolTip("Relative Laplacian sharpness score (higher = crisper focus / detail)")
        h_lay.addWidget(self.lbl_sharp_score)

        root.addWidget(self.header)

        # Center zoomable canvas
        self.canvas = _CompareCanvas(is_dark=self._is_dark, parent=self)
        self.canvas.focused.connect(lambda: self.focused.emit(self.slot_index))
        self.canvas.viewport_changed.connect(
            lambda cx, cy, z: self.viewport_changed.emit(self.slot_index, cx, cy, z)
        )
        root.addWidget(self.canvas, 1)

        # Bottom culling bar
        self.footer = QWidget(self)
        self.footer.setFixedHeight(34)
        f_lay = QHBoxLayout(self.footer)
        f_lay.setContentsMargins(8, 3, 8, 3)
        f_lay.setSpacing(4)

        self.star_btns: list[QPushButton] = []
        for s in range(1, 6):
            b = QPushButton("☆", self.footer)
            b.setFixedSize(22, 22)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setToolTip(f"Rate {s} star{'s' if s > 1 else ''} (Key {s}; click active to clear)")
            b.clicked.connect(lambda _chk=False, star=s: self._on_star_clicked(star))
            self.star_btns.append(b)
            f_lay.addWidget(b)

        self.btn_flag = QPushButton("⚑", self.footer)
        self.btn_flag.setCheckable(True)
        self.btn_flag.setFixedSize(26, 22)
        self.btn_flag.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_flag.setToolTip("Flag / Pick photo (P / Insert)")
        self.btn_flag.clicked.connect(self._on_flag_clicked)
        f_lay.addWidget(self.btn_flag)

        self.btn_rot_ccw = QPushButton("⟲", self.footer)
        self.btn_rot_ccw.setFixedSize(24, 22)
        self.btn_rot_ccw.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_rot_ccw.setToolTip("Quick rotate left 90° (view only)")
        self.btn_rot_ccw.clicked.connect(lambda: self.rotate_requested.emit(self.slot_index, -90))
        f_lay.addWidget(self.btn_rot_ccw)

        self.btn_rot_cw = QPushButton("⟳", self.footer)
        self.btn_rot_cw.setFixedSize(24, 22)
        self.btn_rot_cw.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_rot_cw.setToolTip("Quick rotate right 90° (view only)")
        self.btn_rot_cw.clicked.connect(lambda: self.rotate_requested.emit(self.slot_index, 90))
        f_lay.addWidget(self.btn_rot_cw)

        f_lay.addStretch(1)

        self.btn_dismiss = QPushButton("✕ Dismiss", self.footer)
        self.btn_dismiss.setFixedHeight(24)
        self.btn_dismiss.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_dismiss.setToolTip("Remove from comparison without deleting (X)")
        self.btn_dismiss.clicked.connect(lambda: self.dismiss_requested.emit(self.slot_index))
        f_lay.addWidget(self.btn_dismiss)

        self.btn_delete = QPushButton("🗑 Delete", self.footer)
        self.btn_delete.setFixedHeight(24)
        self.btn_delete.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_delete.setToolTip("Move file to Recycle Bin (Del)")
        self.btn_delete.clicked.connect(lambda: self.delete_requested.emit(self.slot_index))
        f_lay.addWidget(self.btn_delete)

        root.addWidget(self.footer)

    def set_dark_mode(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        self.canvas.set_dark_mode(is_dark)
        self._refresh_chrome_style()

    def set_focused(self, focused: bool) -> None:
        self.is_focused = focused
        self._refresh_chrome_style()

    def set_pinned(self, pinned: bool) -> None:
        self.is_pinned = pinned
        self.btn_pin.blockSignals(True)
        self.btn_pin.setChecked(pinned)
        self.btn_pin.setText("📌 Pinned" if pinned else "📌 Pin")
        self.btn_pin.blockSignals(False)
        self._refresh_chrome_style()

    def _refresh_chrome_style(self) -> None:
        border_col = "#3B82F6" if self.is_focused else ("#2A3240" if self._is_dark else "#CBD5E1")
        border_w = 2 if self.is_focused else 1
        bar_bg = "#171C24" if self._is_dark else "#F8FAFC"
        bar_fg = "#F1F5F9" if self._is_dark else "#1E293B"
        muted = "#94A3B8" if self._is_dark else "#64748B"

        self.setStyleSheet(f"_ComparePane {{ border: {border_w}px solid {border_col}; background: {bar_bg}; }}")
        self.header.setStyleSheet(f"background-color: {bar_bg}; color: {bar_fg}; border-bottom: 1px solid {border_col};")
        self.footer.setStyleSheet(f"background-color: {bar_bg}; color: {bar_fg}; border-top: 1px solid {border_col};")
        self.lbl_title.setStyleSheet(f"color: {bar_fg}; border: none; background: transparent;")
        self.lbl_sharp_score.setStyleSheet(f"color: {muted}; border: none; background: transparent;")

        pin_style = (
            "QPushButton { background-color: #2563EB; color: #FFFFFF; border: 1px solid #3B82F6; border-radius: 3px; padding: 1px 7px; font-weight: bold; }"
            if self.is_pinned
            else f"QPushButton {{ background-color: {'#232A36' if self._is_dark else '#E2E8F0'}; color: {bar_fg}; border: 1px solid {'#3E4B5B' if self._is_dark else '#CBD5E1'}; border-radius: 3px; padding: 1px 7px; }}"
        )
        self.btn_pin.setStyleSheet(pin_style)

        btn_bg = "#232A36" if self._is_dark else "#F1F5F9"
        btn_bd = "#3E4B5B" if self._is_dark else "#CBD5E1"
        rot_style = (
            f"QPushButton {{ background-color: {btn_bg}; color: {bar_fg}; border: 1px solid {btn_bd}; border-radius: 3px; font-size: 12px; padding: 0; }}"
            f"QPushButton:hover {{ border-color: #3B82F6; }}"
        )
        self.btn_rot_ccw.setStyleSheet(rot_style)
        self.btn_rot_cw.setStyleSheet(rot_style)
        self.btn_dismiss.setStyleSheet(
            f"QPushButton {{ background-color: {btn_bg}; color: {bar_fg}; border: 1px solid {btn_bd}; border-radius: 3px; padding: 1px 8px; }}"
        )
        self.btn_delete.setStyleSheet(
            "QPushButton { background-color: #B91C1C; color: #FFFFFF; border: 1px solid #DC2626; border-radius: 3px; padding: 1px 8px; }"
        )
        self._refresh_rating_flag_buttons()

    def _refresh_rating_flag_buttons(self) -> None:
        star_off = "#64748B" if self._is_dark else "#94A3B8"
        for idx, b in enumerate(self.star_btns, start=1):
            filled = idx <= self._rating
            b.setText("★" if filled else "☆")
            b.setStyleSheet(
                f"QPushButton {{ background: transparent; border: none; color: {'#F59E0B' if filled else star_off}; font-size: 13px; padding: 0; }}"
            )
        self.btn_flag.blockSignals(True)
        self.btn_flag.setChecked(bool(self._flag))
        self.btn_flag.blockSignals(False)
        self.btn_flag.setStyleSheet(
            "QPushButton { background-color: #DC2626; color: #FFFFFF; border: 1px solid #EF4444; border-radius: 3px; padding: 0; }"
            if self._flag
            else f"QPushButton {{ background: transparent; color: {star_off}; border: 1px solid {'#334155' if self._is_dark else '#CBD5E1'}; border-radius: 3px; padding: 0; }}"
        )

    def _on_star_clicked(self, star: int) -> None:
        if self.media_id is None:
            return
        new_rating = 0 if self._rating == star else star
        self.set_rating_ui(new_rating)
        self.rating_changed.emit(self.media_id, new_rating)

    def _on_flag_clicked(self, checked: bool) -> None:
        if self.media_id is None:
            return
        new_flag = 1 if checked else 0
        self.set_flag_ui(new_flag)
        self.flag_toggled.emit(self.media_id, new_flag)

    def set_rating_ui(self, rating: int) -> None:
        self._rating = max(0, min(5, int(rating)))
        self._refresh_rating_flag_buttons()

    def set_flag_ui(self, flag: int) -> None:
        self._flag = 1 if flag else 0
        self._refresh_rating_flag_buttons()

    def load_media(
        self,
        media_id: int,
        detail: dict[str, Any],
        qimg: QImage,
        face_boxes: Sequence[tuple[float, float, float, float]],
        sharpness: float,
    ) -> None:
        self.media_id = int(media_id)
        self.detail = dict(detail)
        self.exif_parts = _format_exif_summary(detail)
        self.sharpness_score = float(sharpness)
        self._rating = int(detail.get("rating") or 0)
        self._flag = int(detail.get("flag") or 0)

        fname = str(detail.get("filename") or f"Media #{media_id}")
        w = int(detail.get("width") or qimg.width() or 0)
        h = int(detail.get("height") or qimg.height() or 0)
        dims = f"  ({w}×{h})" if w > 0 and h > 0 else ""
        self.lbl_title.setText(f"{fname}{dims}")
        self.lbl_sharp_score.setText(f"Sharp: {sharpness:.0f}")
        self.canvas.set_image(qimg, face_boxes=face_boxes)
        self._refresh_rating_flag_buttons()

    def update_exif_html(self, differing_keys: set[str]) -> None:
        hl_color = "#FBBF24" if self._is_dark else "#B45309"
        muted = "#CBD5E1" if self._is_dark else "#475569"
        spans: list[str] = []
        for k in ("camera", "exposure", "aperture", "iso", "focal"):
            val = self.exif_parts.get(k, "")
            if not val:
                continue
            if k in differing_keys:
                spans.append(f"<span style='color:{hl_color}; font-weight:bold;'>{val}</span>")
            else:
                spans.append(f"<span style='color:{muted};'>{val}</span>")
        self.lbl_exif.setText(" &nbsp; ".join(spans))


class CompareMediaView(QWidget):
    """
    Multi-Photo Compare / Light Table View integrated into MainWindow's central_stack.
    """

    close_requested = Signal(object)            # last_winner_media_id | None
    rating_changed = Signal(int, int)           # media_id, rating
    flag_toggled = Signal(int, int)             # media_id, flag
    media_deleted = Signal(int)                 # media_id
    open_in_editor_requested = Signal(int)      # media_id
    keep_winner_requested = Signal(int)         # media_id

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CompareMediaView")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._repo: Any = None
        self._thumb_repo: Any = None
        self._is_dark: bool = True
        self._lock_sync: bool = True
        self._focus_peaking: bool = False
        self._clipping_warnings: bool = False
        self._auto_advance: bool = False
        self._layout_mode: str = "side_by_side"  # side_by_side | top_bottom | three_up | quad | wipe
        self._candidate_ids: list[int] = []
        self._active_ids: list[int] = []
        self._pinned_slot: int | None = None
        self._focused_slot: int = 0
        self._face_cycle_index: int = -1
        self._sync_guard: bool = False

        self._detail_cache: dict[int, dict[str, Any]] = {}
        self._image_cache: dict[int, QImage] = {}
        self._faces_cache: dict[int, list[tuple[float, float, float, float]]] = {}
        self._sharpness_cache: dict[int, float] = {}
        self._rotation_map: dict[int, int] = {}

        self._panes: list[_ComparePane] = []
        self._wipe_canvas: _CompareWipeCanvas | None = None
        self._filmstrip_buttons: dict[int, QPushButton] = {}
        self._filmstrip_base_pixmaps: dict[int, QPixmap] = {}

        self._build_ui()
        self.set_dark_mode(False)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Workspace area for splitters
        self.workspace_host = QWidget(self)
        self.workspace_layout = QVBoxLayout(self.workspace_host)
        self.workspace_layout.setContentsMargins(0, 0, 0, 0)
        self.workspace_layout.setSpacing(0)
        root.addWidget(self.workspace_host, 1)

        # Bottom Candidate Filmstrip (shown for 5+ items or whenever candidates > active panes)
        self.filmstrip_bar = QFrame(self)
        self.filmstrip_bar.setFixedHeight(86)
        fb_lay = QVBoxLayout(self.filmstrip_bar)
        fb_lay.setContentsMargins(8, 4, 8, 4)
        fb_lay.setSpacing(2)

        self.lbl_filmstrip = QLabel("Candidates (Click to load challenger, Left/Right arrows to cycle)", self.filmstrip_bar)
        self.lbl_filmstrip.setFont(QFont("Segoe UI", 8))
        fb_lay.addWidget(self.lbl_filmstrip)

        self.filmstrip_scroll = QScrollArea(self.filmstrip_bar)
        self.filmstrip_scroll.setWidgetResizable(True)
        self.filmstrip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.filmstrip_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.filmstrip_host = QWidget(self.filmstrip_scroll)
        self.filmstrip_row = QHBoxLayout(self.filmstrip_host)
        self.filmstrip_row.setContentsMargins(0, 0, 0, 0)
        self.filmstrip_row.setSpacing(6)
        self.filmstrip_row.addStretch(1)
        self.filmstrip_scroll.setWidget(self.filmstrip_host)
        fb_lay.addWidget(self.filmstrip_scroll, 1)

        self.filmstrip_bar.setVisible(False)
        root.addWidget(self.filmstrip_bar)

    def bind_repositories(self, repo: Any, thumb_repo: Any = None) -> None:
        self._repo = repo
        self._thumb_repo = thumb_repo

    def set_dark_mode(self, is_dark: bool) -> None:
        self._is_dark = bool(is_dark)
        bg = "#11151B" if is_dark else "#F1F5F9"
        fg = "#CBD5E1" if is_dark else "#334155"
        bd = "#2A3240" if is_dark else "#CBD5E1"
        self.setStyleSheet(
            f"#CompareMediaView {{ background-color: {bg}; }}\n"
            f"#CompareMediaView QSplitter {{ background-color: {bg}; }}\n"
            f"#CompareMediaView QSplitter::handle {{ background-color: {bd}; }}\n"
            f"#CompareMediaView QSplitter::handle:hover {{ background-color: #3B82F6; }}"
        )
        self.filmstrip_bar.setStyleSheet(f"background-color: {bg}; border-top: 1px solid {bd};")
        self.lbl_filmstrip.setStyleSheet(f"color: {fg}; border: none; background: transparent;")
        for p in self._panes:
            p.set_dark_mode(is_dark)
        if self._wipe_canvas is not None:
            self._wipe_canvas.set_dark_mode(is_dark)
        self._refresh_filmstrip_styles()

    @property
    def active_media_ids(self) -> list[int]:
        return list(self._active_ids)

    @property
    def layout_mode(self) -> str:
        return self._layout_mode

    @property
    def pinned_slot(self) -> int | None:
        return self._pinned_slot

    @property
    def lock_sync(self) -> bool:
        return self._lock_sync

    @property
    def focus_peaking(self) -> bool:
        return self._focus_peaking

    @property
    def clipping_warnings(self) -> bool:
        return self._clipping_warnings

    @property
    def auto_advance(self) -> bool:
        return self._auto_advance

    def open_media_ids(
        self,
        selected_ids: Sequence[int],
        all_grid_ids: Sequence[int] | None = None,
        repo: Any = None,
        thumb_repo: Any = None,
    ) -> None:
        """
        Launch Compare View with smart selection handling:
        - 1 photo: pair with next photo in grid (or burst neighbor < 3s).
        - 2-4 photos: load all into active slots (2-Up, 3-Up, 2x2 Quad).
        - 5+ photos: load first 2 into Slot A & Slot B + show Candidate Filmstrip.
        """
        if repo is not None:
            self._repo = repo
        if thumb_repo is not None:
            self._thumb_repo = thumb_repo

        self._detail_cache.clear()
        self._image_cache.clear()
        self._faces_cache.clear()
        self._sharpness_cache.clear()
        self._pinned_slot = None
        self._focused_slot = 0
        self._face_cycle_index = -1

        clean_ids = [int(mid) for mid in dict.fromkeys(selected_ids)]
        if len(clean_ids) == 1:
            pair_id = self._find_auto_pair(clean_ids[0], all_grid_ids or [])
            if pair_id is not None and pair_id != clean_ids[0]:
                clean_ids.append(pair_id)

        self._candidate_ids = list(clean_ids)
        if len(clean_ids) >= 5:
            self._active_ids = clean_ids[:2]
            self._layout_mode = "side_by_side"
        elif len(clean_ids) == 4:
            self._active_ids = clean_ids[:4]
            self._layout_mode = "quad"
        elif len(clean_ids) == 3:
            self._active_ids = clean_ids[:3]
            self._layout_mode = "three_up"
        else:
            self._active_ids = clean_ids[:2]
            self._layout_mode = "side_by_side"

        self._rebuild_panes_and_filmstrip()
        self._refresh_all_panes()

    def _find_auto_pair(self, media_id: int, all_grid_ids: Sequence[int]) -> int | None:
        grid_list = [int(x) for x in all_grid_ids]
        if media_id in grid_list:
            idx = grid_list.index(media_id)
            if idx + 1 < len(grid_list):
                return grid_list[idx + 1]
            if idx - 1 >= 0:
                return grid_list[idx - 1]
        if self._repo is not None:
            try:
                conn = self._repo.engine.catalog_conn()
                row = conn.execute("SELECT date_taken FROM media WHERE id = ?;", (media_id,)).fetchone()
                if row and row["date_taken"]:
                    dt = int(row["date_taken"])
                    nbr = conn.execute(
                        "SELECT id FROM media WHERE id != ? AND COALESCE(is_missing, 0) = 0 "
                        "AND ABS(date_taken - ?) <= 3 ORDER BY ABS(date_taken - ?) ASC LIMIT 1;",
                        (media_id, dt, dt),
                    ).fetchone()
                    if nbr:
                        return int(nbr["id"])
            except Exception:
                pass
        return None

    def _ensure_loaded(self, media_id: int) -> tuple[dict[str, Any], QImage, list[tuple[float, float, float, float]], float]:
        mid = int(media_id)
        if mid not in self._detail_cache:
            detail: dict[str, Any] = {"id": mid, "filename": f"Media #{mid}"}
            if self._repo is not None:
                try:
                    detail = self._repo.get_media_detail(mid) or detail
                except Exception:
                    pass
            self._detail_cache[mid] = detail
        detail = self._detail_cache[mid]

        if mid not in self._image_cache:
            fpath = detail.get("file_path") or ""
            if not fpath and detail.get("folder_path") and detail.get("filename"):
                fpath = str(Path(detail["folder_path"]) / str(detail["filename"]))
            self._image_cache[mid] = _decode_media_image(fpath)
        qimg = self._image_cache[mid]

        if mid not in self._faces_cache:
            boxes: list[tuple[float, float, float, float]] = []
            if self._repo is not None:
                try:
                    conn = self._repo.engine.catalog_conn()
                    rows = conn.execute(
                        "SELECT rect_x, rect_y, rect_w, rect_h FROM media_people "
                        "WHERE media_id = ? AND COALESCE(state, 0) >= 0 ORDER BY rect_x ASC;",
                        (mid,),
                    ).fetchall()
                    boxes = [
                        (float(r["rect_x"]), float(r["rect_y"]), float(r["rect_w"]), float(r["rect_h"]))
                        for r in rows
                        if float(r["rect_w"] or 0.0) > 0 and float(r["rect_h"] or 0.0) > 0
                    ]
                except Exception:
                    boxes = []
            if not boxes and not qimg.isNull():
                boxes = detect_faces_fallback(qimg)
            self._faces_cache[mid] = boxes
        faces = self._faces_cache[mid]

        if mid not in self._sharpness_cache:
            self._sharpness_cache[mid] = compute_laplacian_sharpness(qimg, face_boxes=faces)
        sharp = self._sharpness_cache[mid]

        return detail, qimg, faces, sharp

    def _clear_workspace(self) -> None:
        while self.workspace_layout.count() > 0:
            item = self.workspace_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self._panes.clear()
        self._wipe_canvas = None

    def _make_pane(self, slot_idx: int) -> _ComparePane:
        pane = _ComparePane(slot_idx, is_dark=self._is_dark, parent=self.workspace_host)
        pane.focused.connect(self._on_pane_focused)
        pane.pin_toggled.connect(self._on_pane_pin_toggled)
        pane.rating_changed.connect(self._on_pane_rating_changed)
        pane.flag_toggled.connect(self._on_pane_flag_toggled)
        pane.dismiss_requested.connect(self.dismiss_slot)
        pane.delete_requested.connect(self.delete_slot)
        pane.rotate_requested.connect(self._on_pane_rotate_requested)
        pane.viewport_changed.connect(self._on_pane_viewport_changed)
        pane.canvas.set_focus_peaking(self._focus_peaking)
        pane.canvas.set_clipping_warnings(self._clipping_warnings)
        return pane

    def _rebuild_panes_and_filmstrip(self) -> None:
        self._clear_workspace()
        count = len(self._active_ids)
        if count == 0:
            return

        mode = self._layout_mode
        if mode == "wipe" and count >= 2:
            self._wipe_canvas = _CompareWipeCanvas(is_dark=self._is_dark, parent=self.workspace_host)
            self._wipe_canvas.set_focus_peaking(self._focus_peaking)
            self._wipe_canvas.set_clipping_warnings(self._clipping_warnings)
            self.workspace_layout.addWidget(self._wipe_canvas)
            self._rebuild_filmstrip()
            return

        for idx in range(count):
            self._panes.append(self._make_pane(idx))

        if count == 1:
            self.workspace_layout.addWidget(self._panes[0])
        elif count == 2:
            orient = Qt.Orientation.Vertical if mode == "top_bottom" else Qt.Orientation.Horizontal
            sp = QSplitter(orient, self.workspace_host)
            sp.addWidget(self._panes[0])
            sp.addWidget(self._panes[1])
            sp.setSizes([500, 500])
            self.workspace_layout.addWidget(sp)
        elif count == 3:
            sp = QSplitter(Qt.Orientation.Horizontal, self.workspace_host)
            for p in self._panes:
                sp.addWidget(p)
            sp.setSizes([333, 333, 333])
            self.workspace_layout.addWidget(sp)
        else:
            # 2x2 Quad Grid
            v_sp = QSplitter(Qt.Orientation.Vertical, self.workspace_host)
            top_sp = QSplitter(Qt.Orientation.Horizontal, v_sp)
            bot_sp = QSplitter(Qt.Orientation.Horizontal, v_sp)
            top_sp.addWidget(self._panes[0])
            top_sp.addWidget(self._panes[1])
            bot_sp.addWidget(self._panes[2])
            if len(self._panes) > 3:
                bot_sp.addWidget(self._panes[3])
            top_sp.setSizes([500, 500])
            bot_sp.setSizes([500, 500])
            v_sp.addWidget(top_sp)
            v_sp.addWidget(bot_sp)
            v_sp.setSizes([500, 500])
            self.workspace_layout.addWidget(v_sp)

        if self._focused_slot >= len(self._panes):
            self._focused_slot = 0
        for idx, p in enumerate(self._panes):
            p.set_focused(idx == self._focused_slot)
            p.set_pinned(self._pinned_slot == idx)

        self._rebuild_filmstrip()

    def _rebuild_filmstrip(self) -> None:
        while self.filmstrip_row.count() > 1:
            item = self.filmstrip_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._filmstrip_buttons.clear()
        self._filmstrip_base_pixmaps.clear()

        show_strip = len(self._candidate_ids) > len(self._active_ids) or len(self._candidate_ids) >= 5
        self.filmstrip_bar.setVisible(show_strip)
        if not show_strip:
            return

        blobs: dict[int, bytes] = {}
        if self._thumb_repo is not None:
            try:
                blobs = self._thumb_repo.get_webp_blobs_batch(self._candidate_ids)
            except Exception:
                blobs = {}

        for mid in self._candidate_ids:
            btn = QPushButton(self.filmstrip_host)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setFixedSize(64, 54)
            btn.setIconSize(QSize(56, 46))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            pm = QPixmap()
            if mid in blobs:
                pm.loadFromData(blobs[mid], "WEBP")
            if pm.isNull():
                detail, qimg, _, _ = self._ensure_loaded(mid)
                if not qimg.isNull():
                    pm = QPixmap.fromImage(qimg.scaled(56, 46, Qt.AspectRatioMode.KeepAspectRatio))
                btn.setToolTip(str(detail.get("filename") or f"Media #{mid}"))
            if not pm.isNull():
                self._filmstrip_base_pixmaps[mid] = pm.scaled(
                    56, 46, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
            else:
                btn.setText(str(mid))
            btn.clicked.connect(lambda _chk=False, m=mid: self._on_filmstrip_clicked(m))
            self._filmstrip_buttons[mid] = btn
            self.filmstrip_row.insertWidget(self.filmstrip_row.count() - 1, btn)

        self._refresh_filmstrip_badges()
        self._refresh_filmstrip_styles()

    def _refresh_filmstrip_badges(self) -> None:
        from PySide6.QtGui import QIcon

        pinned_mid = (
            self._active_ids[self._pinned_slot]
            if self._pinned_slot is not None and 0 <= self._pinned_slot < len(self._active_ids)
            else None
        )
        for mid, btn in self._filmstrip_buttons.items():
            base = self._filmstrip_base_pixmaps.get(mid)
            if base is None or base.isNull():
                continue
            detail = self._detail_cache.get(mid)
            if detail is None and self._repo is not None:
                try:
                    detail = self._repo.get_media_detail(mid) or {}
                    self._detail_cache[mid] = detail
                except Exception:
                    detail = {}
            detail = detail or {}
            rating = int(detail.get("rating") or 0)
            flag = int(detail.get("flag") or 0)
            is_pinned = pinned_mid == mid

            if rating <= 0 and flag <= 0 and not is_pinned:
                btn.setIcon(QIcon(base))
                continue

            canvas = QPixmap(56, 46)
            canvas.fill(Qt.GlobalColor.transparent)
            p = QPainter(canvas)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            dx = (56 - base.width()) // 2
            dy = (46 - base.height()) // 2
            p.drawPixmap(dx, dy, base)
            p.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))

            if is_pinned:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor("#2563EB"))
                p.drawRoundedRect(1, 1, 16, 13, 3, 3)
                p.setPen(QColor("#FFFFFF"))
                p.drawText(QRectF(1, 1, 16, 13), Qt.AlignmentFlag.AlignCenter, "P")

            if flag > 0:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor("#16A34A"))
                p.drawRoundedRect(39, 1, 16, 13, 3, 3)
                p.setPen(QColor("#FFFFFF"))
                p.drawText(QRectF(39, 1, 16, 13), Qt.AlignmentFlag.AlignCenter, "⚑")

            if rating > 0:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(15, 23, 42, 210))
                p.drawRoundedRect(29, 32, 26, 13, 3, 3)
                p.setPen(QColor("#FBBF24"))
                p.drawText(QRectF(29, 32, 26, 13), Qt.AlignmentFlag.AlignCenter, f"★{rating}")

            p.end()
            btn.setIcon(QIcon(canvas))

    def _refresh_filmstrip_styles(self) -> None:
        for mid, btn in self._filmstrip_buttons.items():
            is_active = mid in self._active_ids
            if is_active:
                btn.setStyleSheet("QPushButton { border: 2px solid #3B82F6; border-radius: 4px; background: #1E3A5F; }")
            else:
                bd = "#334155" if self._is_dark else "#CBD5E1"
                bg = "#1E242E" if self._is_dark else "#FFFFFF"
                btn.setStyleSheet(f"QPushButton {{ border: 1px solid {bd}; border-radius: 4px; background: {bg}; }}")

    def _refresh_all_panes(self) -> None:
        if self._wipe_canvas is not None and len(self._active_ids) >= 2:
            mid_a, mid_b = self._active_ids[0], self._active_ids[1]
            det_a, q_a, _, _ = self._ensure_loaded(mid_a)
            det_b, q_b, _, _ = self._ensure_loaded(mid_b)
            self._wipe_canvas.set_images(
                q_a,
                q_b,
                name_a=str(det_a.get("filename") or f"#{mid_a}"),
                name_b=str(det_b.get("filename") or f"#{mid_b}"),
                rot_a=self._rotation_map.get(mid_a, 0),
                rot_b=self._rotation_map.get(mid_b, 0),
            )
            self._refresh_filmstrip_badges()
            self._refresh_filmstrip_styles()
            return

        if not self._panes:
            return
        for pane, mid in zip(self._panes, self._active_ids):
            detail, qimg, faces, sharp = self._ensure_loaded(mid)
            pane.load_media(mid, detail, qimg, faces, sharp)
            rot = self._rotation_map.get(mid, 0)
            if rot:
                pane.canvas.set_image(qimg, face_boxes=faces, rotation_deg=rot)

        # Highlight sharpest pane
        if len(self._panes) >= 2:
            best_score = max(p.sharpness_score for p in self._panes)
            best_assigned = False
            for p in self._panes:
                is_best = best_score > 0 and abs(p.sharpness_score - best_score) < 1e-6 and not best_assigned
                p.lbl_sharpest.setVisible(is_best)
                if is_best:
                    best_assigned = True
        else:
            for p in self._panes:
                p.lbl_sharpest.setVisible(False)

        # Highlight differing EXIF tags across panes
        diff_keys: set[str] = set()
        if len(self._panes) >= 2:
            for k in ("camera", "exposure", "aperture", "iso", "focal"):
                vals = {p.exif_parts.get(k, "") for p in self._panes}
                if len(vals) > 1:
                    diff_keys.add(k)
        for p in self._panes:
            p.update_exif_html(diff_keys)

        self._refresh_filmstrip_badges()
        self._refresh_filmstrip_styles()

    # ------------------------------------------------------------------
    # Public Ribbon / Keyboard Actions
    # ------------------------------------------------------------------
    def set_layout_mode(self, mode: str) -> None:
        if mode == "quad" and len(self._candidate_ids) >= 4 and len(self._active_ids) < 4:
            self._active_ids = list(self._candidate_ids[:4])
        elif mode == "three_up" and len(self._candidate_ids) >= 3 and len(self._active_ids) < 3:
            self._active_ids = list(self._candidate_ids[:3])
        elif mode in ("side_by_side", "top_bottom", "wipe") and len(self._active_ids) > 2:
            self._active_ids = list(self._active_ids[:2])
        self._layout_mode = mode
        self._rebuild_panes_and_filmstrip()
        self._refresh_all_panes()

    def swap_panes(self) -> None:
        if len(self._active_ids) >= 2:
            self._active_ids[0], self._active_ids[1] = self._active_ids[1], self._active_ids[0]
            self._refresh_all_panes()

    def set_lock_sync(self, enabled: bool) -> None:
        self._lock_sync = bool(enabled)

    def set_focus_peaking(self, enabled: bool) -> None:
        self._focus_peaking = bool(enabled)
        for p in self._panes:
            p.canvas.set_focus_peaking(self._focus_peaking)
        if self._wipe_canvas is not None:
            self._wipe_canvas.set_focus_peaking(self._focus_peaking)

    def set_clipping_warnings(self, enabled: bool) -> None:
        self._clipping_warnings = bool(enabled)
        for p in self._panes:
            p.canvas.set_clipping_warnings(self._clipping_warnings)
        if self._wipe_canvas is not None:
            self._wipe_canvas.set_clipping_warnings(self._clipping_warnings)

    def set_auto_advance(self, enabled: bool) -> None:
        self._auto_advance = bool(enabled)

    def zoom_actual_1_to_1(self) -> None:
        if self._wipe_canvas is not None:
            self._wipe_canvas.zoom_actual_1_to_1()
            return
        if not self._panes:
            return
        idx = min(self._focused_slot, len(self._panes) - 1)
        cx, cy, z = self._panes[idx].canvas.zoom_actual_1_to_1()
        if self._lock_sync:
            self._broadcast_viewport(idx, cx, cy, z, fit_mode=False)

    def zoom_fit(self) -> None:
        self._face_cycle_index = -1
        if self._wipe_canvas is not None:
            self._wipe_canvas.zoom_fit()
            return
        for p in self._panes:
            p.canvas.zoom_fit()

    def zoom_to_faces(self) -> None:
        """Zoom each pane individually to its Nth detected face, cycling on repeated clicks."""
        if self._wipe_canvas is not None and self._active_ids:
            faces = self._faces_cache.get(self._active_ids[0], [])
            if faces:
                self._face_cycle_index = (self._face_cycle_index + 1) % len(faces)
                self._wipe_canvas.zoom_to_face(faces[self._face_cycle_index])
            else:
                self._wipe_canvas.zoom_to_face((0.35, 0.35, 0.3, 0.2))
            return
        if not self._panes:
            return
        max_faces = max((len(p.canvas._face_boxes) for p in self._panes), default=0)
        if max_faces <= 0:
            # No detected faces: zoom into center crop cleanly
            for p in self._panes:
                p.canvas.set_normalized_viewport(0.5, 0.5, 2.5, fit_mode=False)
            return
        self._face_cycle_index = (self._face_cycle_index + 1) % max_faces
        for p in self._panes:
            if not p.canvas.zoom_to_face_index(self._face_cycle_index):
                p.canvas.set_normalized_viewport(0.5, 0.5, 2.5, fit_mode=False)

    def toggle_pin_focused_pane(self) -> None:
        if not self._panes:
            return
        idx = min(self._focused_slot, len(self._panes) - 1)
        self._on_pane_pin_toggled(idx, self._pinned_slot != idx)

    def cycle_challenger(self, delta: int) -> None:
        """Cycle the unpinned challenger pane through remaining candidates."""
        if len(self._candidate_ids) <= 1 or len(self._active_ids) < 2:
            return
        target_slot = 1 if self._pinned_slot != 1 else 0
        current_mid = self._active_ids[target_slot]
        other_active = {mid for i, mid in enumerate(self._active_ids) if i != target_slot}
        available = [mid for mid in self._candidate_ids if mid not in other_active]
        if not available:
            return
        cur_pos = available.index(current_mid) if current_mid in available else 0
        next_mid = available[(cur_pos + delta) % len(available)]
        self._active_ids[target_slot] = next_mid
        self._refresh_all_panes()

    def dismiss_focused_pane(self) -> None:
        if self._panes:
            self.dismiss_slot(min(self._focused_slot, len(self._panes) - 1))

    def dismiss_slot(self, slot_idx: int) -> None:
        if slot_idx < 0 or slot_idx >= len(self._active_ids):
            return
        removed_mid = self._active_ids[slot_idx]
        if removed_mid in self._candidate_ids:
            self._candidate_ids.remove(removed_mid)

        # Promote next candidate if available
        replacement = next((m for m in self._candidate_ids if m not in self._active_ids), None)
        if replacement is not None:
            self._active_ids[slot_idx] = replacement
            self._refresh_all_panes()
            self._rebuild_filmstrip()
            return

        self._active_ids.pop(slot_idx)
        if self._pinned_slot == slot_idx:
            self._pinned_slot = None
        elif self._pinned_slot is not None and self._pinned_slot > slot_idx:
            self._pinned_slot -= 1

        if len(self._active_ids) <= 1:
            winner = self._active_ids[0] if self._active_ids else None
            self.close_requested.emit(winner)
            return

        if len(self._active_ids) == 3:
            self._layout_mode = "three_up"
        elif len(self._active_ids) == 2 and self._layout_mode in ("three_up", "quad"):
            self._layout_mode = "side_by_side"
        self._rebuild_panes_and_filmstrip()
        self._refresh_all_panes()

    def delete_slot(self, slot_idx: int) -> None:
        if slot_idx < 0 or slot_idx >= len(self._active_ids):
            return
        mid = self._active_ids[slot_idx]
        detail = self._detail_cache.get(mid) or {}
        fpath = detail.get("file_path") or ""
        if not fpath and detail.get("folder_path") and detail.get("filename"):
            fpath = str(Path(detail["folder_path"]) / str(detail["filename"]))
        if fpath and os.path.exists(fpath):
            try:
                from send2trash import send2trash

                send2trash(fpath)
            except Exception as exc:
                logger.warning("send2trash failed for %s: %s", fpath, exc)
        if self._repo is not None:
            try:
                conn = self._repo.engine.catalog_conn()
                with self._repo.engine.write_lock:
                    conn.execute("DELETE FROM media WHERE id = ?;", (mid,))
                    conn.commit()
            except Exception as exc:
                logger.warning("Failed to delete media %s from catalog: %s", mid, exc)
        self.media_deleted.emit(mid)
        self.dismiss_slot(slot_idx)

    def keep_winner_only(self) -> None:
        winner_slot = self._pinned_slot if self._pinned_slot is not None else self._focused_slot
        if 0 <= winner_slot < len(self._active_ids):
            self._on_keep_winner_only(self._active_ids[winner_slot])

    def open_winner_in_editor(self) -> None:
        winner_slot = self._pinned_slot if self._pinned_slot is not None else self._focused_slot
        if 0 <= winner_slot < len(self._active_ids):
            self.open_in_editor_requested.emit(self._active_ids[winner_slot])

    def close_compare(self) -> None:
        winner_slot = self._pinned_slot if self._pinned_slot is not None else self._focused_slot
        winner_id = self._active_ids[winner_slot] if 0 <= winner_slot < len(self._active_ids) else None
        self.close_requested.emit(winner_id)

    # ------------------------------------------------------------------
    # Internal Slots
    # ------------------------------------------------------------------
    def _on_pane_focused(self, slot_idx: int) -> None:
        self._focused_slot = slot_idx
        for idx, p in enumerate(self._panes):
            p.set_focused(idx == slot_idx)

    def _on_pane_pin_toggled(self, slot_idx: int, checked: bool) -> None:
        self._pinned_slot = slot_idx if checked else None
        for idx, p in enumerate(self._panes):
            p.set_pinned(self._pinned_slot == idx)
        self._refresh_filmstrip_badges()

    def _on_pane_rotate_requested(self, slot_idx: int, delta_deg: int) -> None:
        if 0 <= slot_idx < len(self._active_ids) and slot_idx < len(self._panes):
            mid = self._active_ids[slot_idx]
            new_deg = self._panes[slot_idx].canvas.rotate_by(delta_deg)
            self._rotation_map[mid] = new_deg

    def _maybe_auto_advance_for_media(self, media_id: int) -> None:
        if not self._auto_advance:
            return
        if len(self._candidate_ids) <= len(self._active_ids):
            return
        if media_id not in self._active_ids:
            return
        if self._pinned_slot is not None:
            pinned_mid = (
                self._active_ids[self._pinned_slot]
                if 0 <= self._pinned_slot < len(self._active_ids)
                else None
            )
            if media_id != pinned_mid:
                self.cycle_challenger(1)
            return
        # No pane pinned: advance the slot that was just rated/flagged to the next unshown candidate
        slot_idx = self._active_ids.index(media_id)
        other_active = {mid for i, mid in enumerate(self._active_ids) if i != slot_idx}
        available = [mid for mid in self._candidate_ids if mid not in other_active]
        if not available:
            return
        cur_pos = available.index(media_id) if media_id in available else 0
        self._active_ids[slot_idx] = available[(cur_pos + 1) % len(available)]
        self._refresh_all_panes()

    def rate_focused_pane(self, rating: int) -> None:
        if 0 <= self._focused_slot < len(self._panes):
            p = self._panes[self._focused_slot]
            if p.media_id is not None:
                p.set_rating_ui(rating)
                self._on_pane_rating_changed(p.media_id, rating)

    def _on_pane_rating_changed(self, media_id: int, rating: int) -> None:
        if self._repo is not None:
            try:
                self._repo.set_rating([media_id], rating)
            except Exception:
                pass
        if media_id in self._detail_cache:
            self._detail_cache[media_id]["rating"] = rating
        self.rating_changed.emit(media_id, rating)
        self._refresh_filmstrip_badges()
        self._maybe_auto_advance_for_media(media_id)

    def _on_pane_flag_toggled(self, media_id: int, flag: int) -> None:
        if self._repo is not None:
            try:
                self._repo.set_flag([media_id], flag)
            except Exception:
                pass
        if media_id in self._detail_cache:
            self._detail_cache[media_id]["flag"] = flag
        self.flag_toggled.emit(media_id, flag)
        self._refresh_filmstrip_badges()
        self._maybe_auto_advance_for_media(media_id)

    def _on_keep_winner_only(self, media_id: int) -> None:
        self._on_pane_flag_toggled(media_id, 1)
        self.keep_winner_requested.emit(media_id)
        self.close_requested.emit(media_id)

    def _on_filmstrip_clicked(self, media_id: int) -> None:
        if media_id in self._active_ids or len(self._active_ids) < 2:
            return
        target_slot = 1 if self._pinned_slot != 1 else 0
        self._active_ids[target_slot] = media_id
        self._refresh_all_panes()

    def _on_pane_viewport_changed(self, source_slot: int, cx_norm: float, cy_norm: float, zoom_factor: float) -> None:
        if not self._lock_sync or self._sync_guard:
            return
        self._broadcast_viewport(source_slot, cx_norm, cy_norm, zoom_factor, fit_mode=False)

    def _broadcast_viewport(
        self, source_slot: int, cx_norm: float, cy_norm: float, zoom_factor: float, *, fit_mode: bool | None = None
    ) -> None:
        self._sync_guard = True
        try:
            for idx, pane in enumerate(self._panes):
                if idx != source_slot:
                    pane.canvas.set_normalized_viewport(cx_norm, cy_norm, zoom_factor, fit_mode=fit_mode)
        finally:
            self._sync_guard = False

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.close_compare()
            event.accept()
            return
        if key == Qt.Key.Key_Right:
            self.cycle_challenger(+1)
            event.accept()
            return
        if key == Qt.Key.Key_Left:
            self.cycle_challenger(-1)
            event.accept()
            return
        if key == Qt.Key.Key_F:
            self.zoom_to_faces()
            event.accept()
            return
        if key == Qt.Key.Key_X:
            self.dismiss_focused_pane()
            event.accept()
            return
        if key == Qt.Key.Key_Delete and 0 <= self._focused_slot < len(self._panes):
            self.delete_slot(self._focused_slot)
            event.accept()
            return
        if key in (Qt.Key.Key_P, Qt.Key.Key_Insert) and 0 <= self._focused_slot < len(self._panes):
            p = self._panes[self._focused_slot]
            if p.media_id is not None:
                p._on_flag_clicked(not bool(p._flag))
            event.accept()
            return
        if Qt.Key.Key_0 <= key <= Qt.Key.Key_5 and 0 <= self._focused_slot < len(self._panes):
            self.rate_focused_pane(key - Qt.Key.Key_0)
            event.accept()
            return
        super().keyPressEvent(event)
