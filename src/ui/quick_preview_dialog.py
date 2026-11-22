"""
Spacebar Quick Preview Fullscreen Lightbox for ThumbnailGridView.
Displays the photo borderless over a full-screen dimmed backdrop with floating
filename/rating/flag and shortcut pills, plus Left/Right navigation, 0-5 rating,
P flag toggle, Enter to open full Single View, and Space/Esc or backdrop click to close.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QObject,
    QPointF,
    QPropertyAnimation,
    QRect,
    QRectF,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QImageReader,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from src.utils.paths import is_cloud_only_file


class CullFlashHud(QWidget):
    """
    Brief (~500 ms) translucent center-screen confirmation HUD for rapid culling
    (star rating, favourite heart, Pick / Reject / Unflag) alongside instant auto-advance.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("CullFlashHud")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._kind: str = ""
        self._value: Any = None
        self._label_text: str = ""

        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity)

        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.setInterval(240)
        self._hold_timer.timeout.connect(self._start_fade_out)

        self._fade_anim = QPropertyAnimation(self._opacity, b"opacity", self)
        self._fade_anim.setDuration(260)
        self._fade_anim.setStartValue(1.0)
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade_anim.finished.connect(self.hide)

        self.hide()

    def flash(self, kind: str, value: Any) -> None:
        self._kind = str(kind)
        self._value = value
        if self._kind == "rating":
            r = max(0, min(5, int(value or 0)))
            self._value = r
            self._label_text = f"{r} Star{'s' if r != 1 else ''}" if r > 0 else "Unrated"
            pill_w = 186 if r > 0 else 148
            pill_h = 64 if r > 0 else 54
        elif self._kind == "favourite":
            fav = bool(value)
            self._value = fav
            self._label_text = "Favourited" if fav else "Unfavourited"
            pill_w = 176
            pill_h = 56
        elif self._kind == "flag":
            fl = int(value or 0)
            self._value = fl
            self._label_text = "Picked" if fl == 1 else ("Rejected" if fl == -1 else "Unflagged")
            pill_w = 160
            pill_h = 56
        else:
            self._label_text = str(value or "")
            fm = QFontMetrics(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
            pill_w = max(140, fm.horizontalAdvance(self._label_text) + 48)
            pill_h = 54

        self.setFixedSize(pill_w, pill_h)
        self.recenter()
        self._hold_timer.stop()
        self._fade_anim.stop()
        self._opacity.setOpacity(1.0)
        self.show()
        self.raise_()
        self.update()
        self._hold_timer.start()

    def flash_rating(self, rating: int) -> None:
        self.flash("rating", int(rating))

    def flash_favourite(self, is_fav: bool) -> None:
        self.flash("favourite", bool(is_fav))

    def flash_flag(self, flag: int) -> None:
        self.flash("flag", int(flag))

    def recenter(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        pw = parent.width()
        ph = parent.height()
        if pw <= 0 or ph <= 0:
            return
        x = max(8, (pw - self.width()) // 2)
        y = max(8, (ph - self.height()) // 2)
        self.move(x, y)

    def _start_fade_out(self) -> None:
        if not self.isVisible():
            return
        self._fade_anim.stop()
        self._fade_anim.setStartValue(1.0)
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.start()

    @staticmethod
    def _star_poly(cx: float, cy: float, r_outer: float, r_inner: float) -> QPolygonF:
        pts: list[QPointF] = []
        for i in range(10):
            ang = math.radians(-90.0 + i * 36.0)
            rad = r_outer if (i % 2 == 0) else r_inner
            pts.append(QPointF(cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
        return QPolygonF(pts)

    def paintEvent(self, _event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        p.setPen(QPen(QColor(148, 163, 184, 110), 1.2))
        p.setBrush(QColor(15, 23, 42, 226))
        p.drawRoundedRect(rect, 15.0, 15.0)

        if self._kind == "rating" and int(self._value or 0) > 0:
            r = int(self._value)
            total_w = 5 * 26.0
            start_x = (self.width() - total_w) / 2.0 + 13.0
            cy = self.height() / 2.0 - 6.0
            for idx in range(1, 6):
                cx = start_x + (idx - 1) * 26.0
                poly = self._star_poly(cx, cy, 10.0, 4.4)
                if idx <= r:
                    p.setPen(QPen(QColor("#F59E0B"), 1.0))
                    p.setBrush(QColor("#FBBF24"))
                else:
                    p.setPen(QPen(QColor("#64748B"), 1.0))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolygon(poly)
            p.setPen(QColor("#E2E8F0"))
            p.setFont(QFont("Segoe UI", 8.5, QFont.Weight.DemiBold))
            p.drawText(QRect(0, self.height() - 22, self.width(), 16), Qt.AlignmentFlag.AlignCenter, self._label_text)
        elif self._kind == "favourite":
            fav = bool(self._value)
            hcx = 34.0
            hcy = self.height() / 2.0
            hp = QPainterPath()
            hp.moveTo(hcx, hcy + 9.0)
            hp.cubicTo(hcx - 12.0, hcy + 1.0, hcx - 11.0, hcy - 9.0, hcx - 5.0, hcy - 9.0)
            hp.cubicTo(hcx - 1.5, hcy - 9.0, hcx, hcy - 5.5, hcx, hcy - 4.0)
            hp.cubicTo(hcx, hcy - 5.5, hcx + 1.5, hcy - 9.0, hcx + 5.0, hcy - 9.0)
            hp.cubicTo(hcx + 11.0, hcy - 9.0, hcx + 12.0, hcy + 1.0, hcx, hcy + 9.0)
            if fav:
                p.setPen(QPen(QColor("#FB7185"), 1.2))
                p.setBrush(QColor("#F43F5E"))
            else:
                p.setPen(QPen(QColor("#CBD5E1"), 1.6))
                p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(hp)
            p.setPen(QColor("#F8FAFC" if fav else "#CBD5E1"))
            p.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
            p.drawText(QRect(52, 0, self.width() - 62, self.height()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._label_text)
        elif self._kind == "flag":
            fl = int(self._value or 0)
            fx = 26.0
            fy = self.height() / 2.0 - 10.0
            if fl == 1:
                p.setPen(QPen(QColor("#F87171"), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(QPointF(fx, fy), QPointF(fx, fy + 20.0))
                poly = QPolygonF([QPointF(fx, fy), QPointF(fx + 15.0, fy + 5.0), QPointF(fx, fy + 10.5)])
                p.setPen(QPen(QColor("#EF4444"), 1.0))
                p.setBrush(QColor("#EF4444"))
                p.drawPolygon(poly)
            elif fl == -1:
                p.setPen(QPen(QColor("#EF4444"), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(QPointF(fx, fy + 3.0), QPointF(fx + 14.0, fy + 17.0))
                p.drawLine(QPointF(fx + 14.0, fy + 3.0), QPointF(fx, fy + 17.0))
            else:
                p.setPen(QPen(QColor("#94A3B8"), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(QPointF(fx, fy), QPointF(fx, fy + 20.0))
                poly = QPolygonF([QPointF(fx, fy), QPointF(fx + 14.0, fy + 5.0), QPointF(fx, fy + 10.0)])
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolygon(poly)
            p.setPen(QColor("#F8FAFC" if fl != 0 else "#CBD5E1"))
            p.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
            p.drawText(QRect(52, 0, self.width() - 60, self.height()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._label_text)
        else:
            p.setPen(QColor("#E2E8F0"))
            p.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._label_text)
        p.end()


class QuickPreviewDialog(QDialog):
    """Borderless full-screen Spacebar QuickLook overlay for rapid culling from Grid View."""

    open_full_requested = Signal(int)
    rate_requested = Signal(int, int)      # media_id, rating
    flag_requested = Signal(int)           # legacy toggle signal (media_id)
    flag_set_requested = Signal(int, int)  # media_id, new_flag (1=Pick, -1=Reject, 0=Unflag)
    favourite_requested = Signal(int)      # media_id
    navigate_requested = Signal(int)       # -1 or +1

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Quick Preview")
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setModal(False)
        self.setStyleSheet("QDialog { background: transparent; }")

        self._current_media_id: int | None = None
        self._current_pixmap: QPixmap = QPixmap()
        self._filename: str = ""
        self._rating: int = 0
        self._flag: int = 0
        self._is_favourite: bool = False
        self._favourites_enabled: bool = True
        self._image_rect: QRect = QRect()
        self._app_filter_installed: bool = False

        # Borderless image canvas
        self.lbl_image = QLabel(self)
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setStyleSheet("background: transparent; border: none;")
        self.lbl_image.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        # Floating top-center metadata pill
        self._top_pill = QFrame(self)
        self._top_pill.setStyleSheet(
            "QFrame { background-color: rgba(15, 23, 42, 205); "
            "border: 1px solid rgba(148, 163, 184, 85); border-radius: 16px; }"
        )
        top_lay = QHBoxLayout(self._top_pill)
        top_lay.setContentsMargins(14, 6, 14, 6)
        top_lay.setSpacing(12)

        self.lbl_title = QLabel("", self._top_pill)
        self.lbl_title.setStyleSheet("color: #F8FAFC; font-size: 10pt; font-weight: 600; border: none; background: transparent;")
        top_lay.addWidget(self.lbl_title)

        self.lbl_meta = QLabel("", self._top_pill)
        self.lbl_meta.setStyleSheet("color: #FBBF24; font-size: 9.5pt; font-weight: 600; border: none; background: transparent;")
        top_lay.addWidget(self.lbl_meta)

        self.lbl_flag_icon = QLabel(self._top_pill)
        self.lbl_flag_icon.setFixedSize(16, 16)
        self.lbl_flag_icon.setStyleSheet("border: none; background: transparent;")
        self.lbl_flag_icon.hide()
        top_lay.addWidget(self.lbl_flag_icon)

        self.lbl_flag_text = QLabel("", self._top_pill)
        self.lbl_flag_text.setStyleSheet("color: #F8FAFC; font-size: 9pt; font-weight: 600; border: none; background: transparent;")
        self.lbl_flag_text.hide()
        top_lay.addWidget(self.lbl_flag_text)

        self.lbl_fav_icon = QLabel(self._top_pill)
        self.lbl_fav_icon.setFixedSize(16, 16)
        self.lbl_fav_icon.setStyleSheet("border: none; background: transparent;")
        self.lbl_fav_icon.hide()
        top_lay.addWidget(self.lbl_fav_icon)

        # Floating bottom-center hint pill
        self._hint_pill = QFrame(self)
        self._hint_pill.setStyleSheet(
            "QFrame { background-color: rgba(15, 23, 42, 180); "
            "border: 1px solid rgba(148, 163, 184, 65); border-radius: 13px; }"
        )
        hint_lay = QHBoxLayout(self._hint_pill)
        hint_lay.setContentsMargins(14, 5, 14, 5)
        self.lbl_hint = QLabel(
            "Space / Esc: Close   •   ← / →: Prev / Next   •   0–5: Rate   •   P: Pick   •   X: Reject   •   F: Fav   •   Enter: Full View",
            self._hint_pill,
        )
        self.lbl_hint.setStyleSheet("color: #CBD5E1; font-size: 8.5pt; border: none; background: transparent;")
        hint_lay.addWidget(self.lbl_hint)

        self._cull_flash_hud = CullFlashHud(self)

    def _make_badge_pixmap(self, kind: str) -> QPixmap:
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QLinearGradient, QPainterPath, QPen, QPolygonF

        pm = QPixmap(32, 32)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if kind == "picked":
            banner = QPolygonF([QPointF(9.0, 5.5), QPointF(26.0, 11.0), QPointF(9.0, 17.0)])
            grad = QLinearGradient(9.0, 5.5, 9.0, 17.5)
            grad.setColorAt(0.0, QColor("#F87171"))
            grad.setColorAt(0.5, QColor("#DC2626"))
            grad.setColorAt(1.0, QColor("#991B1B"))
            p.setPen(QPen(QColor("#7F1D1D"), 1.2))
            p.setBrush(grad)
            p.drawPolygon(banner)
            p.setPen(QPen(QColor("#CBD5E1"), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(8, 5, 8, 27)
        elif kind == "rejected":
            p.setPen(QPen(QColor("#CBD5E1"), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawLine(8, 5, 8, 27)
            banner = QPolygonF([QPointF(8.0, 5.5), QPointF(25.0, 11.0), QPointF(8.0, 17.0)])
            p.drawPolygon(banner)
            p.setPen(QPen(QColor("#EF4444"), 2.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(5, 26, 27, 5)
        elif kind == "fav":
            grad = QLinearGradient(16.0, 6.0, 16.0, 26.0)
            grad.setColorAt(0.0, QColor("#FB7185"))
            grad.setColorAt(1.0, QColor("#E11D48"))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(grad)
            hpath = QPainterPath()
            hpath.moveTo(16.0, 26.5)
            hpath.cubicTo(5.5, 19.8, 4.0, 11.5, 8.8, 7.2)
            hpath.cubicTo(12.2, 4.4, 15.0, 6.6, 16.0, 9.2)
            hpath.cubicTo(17.0, 6.6, 19.8, 4.4, 23.2, 7.2)
            hpath.cubicTo(28.0, 11.5, 26.5, 19.8, 16.0, 26.5)
            p.drawPath(hpath)
        p.end()
        pm.setDevicePixelRatio(2.0)
        return pm

    def _update_header(self) -> None:
        stars = "★" * int(self._rating or 0) + "☆" * max(0, 5 - int(self._rating or 0))
        self.lbl_title.setText(self._filename)
        self.lbl_meta.setText(stars)

        if self._flag == 1:
            self.lbl_flag_icon.setPixmap(self._make_badge_pixmap("picked"))
            self.lbl_flag_icon.show()
            self.lbl_flag_text.setText("Picked")
            self.lbl_flag_text.setStyleSheet("color: #FCA5A5; font-size: 9pt; font-weight: 600; border: none; background: transparent;")
            self.lbl_flag_text.show()
        elif self._flag == -1:
            self.lbl_flag_icon.setPixmap(self._make_badge_pixmap("rejected"))
            self.lbl_flag_icon.show()
            self.lbl_flag_text.setText("Rejected")
            self.lbl_flag_text.setStyleSheet("color: #CBD5E1; font-size: 9pt; font-weight: 600; border: none; background: transparent;")
            self.lbl_flag_text.show()
        else:
            self.lbl_flag_icon.hide()
            self.lbl_flag_text.hide()

        if self._favourites_enabled and self._is_favourite:
            self.lbl_fav_icon.setPixmap(self._make_badge_pixmap("fav"))
            self.lbl_fav_icon.show()
        else:
            self.lbl_fav_icon.hide()

        if self._favourites_enabled:
            self.lbl_hint.setText(
                "Space / Esc: Close   •   ← / →: Prev / Next   •   0–5: Rate   •   P: Pick   •   X: Reject   •   F: Fav   •   Enter: Full View"
            )
        else:
            self.lbl_hint.setText(
                "Space / Esc: Close   •   ← / →: Prev / Next   •   0–5: Rate   •   P: Pick   •   X: Reject   •   Enter: Full View"
            )
        self._position_floating_overlays()

    def _position_floating_overlays(self) -> None:
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        self.lbl_image.setGeometry(48, 56, max(100, w - 96), max(100, h - 112))
        self._top_pill.adjustSize()
        self._top_pill.move(max(16, (w - self._top_pill.width()) // 2), 16)
        self._top_pill.raise_()

        self._hint_pill.adjustSize()
        self._hint_pill.move(max(16, (w - self._hint_pill.width()) // 2), max(16, h - self._hint_pill.height() - 16))
        self._hint_pill.raise_()

    def show_media(
        self,
        media_id: int,
        file_path: Path | str,
        filename: str,
        rating: int = 0,
        flag: int = 0,
        fallback_pixmap: QPixmap | None = None,
        is_favourite: bool = False,
        favourites_enabled: bool = True,
    ) -> None:
        self._current_media_id = int(media_id)
        self._filename = filename
        self._rating = int(rating or 0)
        self._flag = int(flag or 0)
        self._is_favourite = bool(is_favourite)
        self._favourites_enabled = bool(favourites_enabled)

        # Span the full screen containing the parent window
        screen = None
        pw = self.parentWidget()
        if pw is not None:
            screen = QApplication.screenAt(pw.geometry().center())
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())

        self._update_header()
        pix = self._load_preview_pixmap(Path(file_path), fallback_pixmap)
        self._current_pixmap = pix
        self._position_floating_overlays()
        self._update_scaled_pixmap()
        if not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus()

    def _load_preview_pixmap(
        self, path: Path, fallback_pixmap: QPixmap | None
    ) -> QPixmap:
        if not path.is_file() or is_cloud_only_file(path):
            return fallback_pixmap if fallback_pixmap is not None else QPixmap()

        ext = path.suffix.lstrip(".").lower()
        if ext in {"nef", "nrw", "arw", "sr2", "srf", "cr2", "dng", "orf", "rw2", "pef", "srw"}:
            try:
                from PySide6.QtCore import QBuffer, QByteArray, QIODevice
                from src.imaging.raw_fast_thumb import extract_tiff_embedded_jpeg

                res = extract_tiff_embedded_jpeg(path, target_size=1920)
                if res is not None:
                    ba = QByteArray(res[0])
                    buf = QBuffer(ba)
                    buf.open(QIODevice.OpenModeFlag.ReadOnly)
                    reader = QImageReader(buf)
                    reader.setAutoTransform(True)
                    qimg = reader.read()
                    buf.close()
                    if not qimg.isNull():
                        return QPixmap.fromImage(qimg)
            except Exception:
                pass

        try:
            reader = QImageReader(str(path))
            reader.setAutoTransform(True)
            sz = reader.size()
            if sz.isValid() and max(sz.width(), sz.height()) > 2048:
                reader.setScaledSize(sz.scaled(2048, 2048, Qt.AspectRatioMode.KeepAspectRatio))
            qimg = reader.read()
            if not qimg.isNull():
                return QPixmap.fromImage(qimg)
        except Exception:
            pass
        return fallback_pixmap if fallback_pixmap is not None else QPixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._current_pixmap.isNull():
            self.lbl_image.setText("Preview unavailable")
            self._image_rect = QRect()
            return
        avail = self.lbl_image.size()
        if avail.width() <= 0 or avail.height() <= 0:
            return
        scaled = self._current_pixmap.scaled(
            avail,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.lbl_image.setPixmap(scaled)
        gx = self.lbl_image.x() + (avail.width() - scaled.width()) // 2
        gy = self.lbl_image.y() + (avail.height() - scaled.height()) // 2
        self._image_rect = QRect(gx, gy, scaled.width(), scaled.height())

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(6, 10, 16, 225))
        p.end()

    def mousePressEvent(self, event) -> None:
        pt = event.position().toPoint()
        if (
            not self._image_rect.contains(pt)
            and not self._top_pill.geometry().contains(pt)
            and not self._hint_pill.geometry().contains(pt)
        ):
            self.hide()
            event.accept()
            return
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_floating_overlays()
        self._update_scaled_pixmap()
        if hasattr(self, "_cull_flash_hud"):
            self._cull_flash_hud.recenter()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        app = QApplication.instance()
        if app is not None and not self._app_filter_installed:
            app.installEventFilter(self)
            self._app_filter_installed = True
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    def hideEvent(self, event) -> None:
        app = QApplication.instance()
        if app is not None and self._app_filter_installed:
            app.removeEventFilter(self)
            self._app_filter_installed = False
        super().hideEvent(event)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if self.isVisible() and event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            if obj is not self and not self.isAncestorOf(obj if isinstance(obj, QWidget) else None):
                self.keyPressEvent(event)
                if event.isAccepted():
                    return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        shift_held = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if key == Qt.Key.Key_Space and event.isAutoRepeat():
            event.accept()
            return
        if key in (Qt.Key.Key_Space, Qt.Key.Key_Escape):
            self.hide()
            event.accept()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self._current_media_id is not None:
            mid = self._current_media_id
            self.hide()
            self.open_full_requested.emit(mid)
            event.accept()
            return
        if key == Qt.Key.Key_Left:
            self.navigate_requested.emit(-1)
            event.accept()
            return
        if key == Qt.Key.Key_Right:
            self.navigate_requested.emit(1)
            event.accept()
            return
        _SHIFT_DIGIT_KEYS = {
            Qt.Key.Key_ParenRight: 0,
            Qt.Key.Key_Exclam: 1,
            Qt.Key.Key_At: 2,
            Qt.Key.Key_QuoteDbl: 2,
            Qt.Key.Key_NumberSign: 3,
            Qt.Key.Key_sterling: 3,
            Qt.Key.Key_Dollar: 4,
            Qt.Key.Key_Percent: 5,
        }
        rating_val: int | None = None
        if Qt.Key.Key_0 <= key <= Qt.Key.Key_5:
            rating_val = key - Qt.Key.Key_0
        elif shift_held and key in _SHIFT_DIGIT_KEYS:
            rating_val = _SHIFT_DIGIT_KEYS[key]
        elif shift_held and 0x30 <= int(event.nativeVirtualKey() or 0) <= 0x35:
            rating_val = int(event.nativeVirtualKey()) - 0x30
        if rating_val is not None and self._current_media_id is not None:
            self._rating = rating_val
            self._update_header()
            self._cull_flash_hud.flash_rating(self._rating)
            self.rate_requested.emit(self._current_media_id, self._rating)
            if shift_held:
                self.navigate_requested.emit(1)
            event.accept()
            return
        if key == Qt.Key.Key_P and self._current_media_id is not None:
            self._flag = 0 if self._flag == 1 else 1
            self._update_header()
            self._cull_flash_hud.flash_flag(self._flag)
            self.flag_set_requested.emit(self._current_media_id, self._flag)
            self.flag_requested.emit(self._current_media_id)
            if shift_held:
                self.navigate_requested.emit(1)
            event.accept()
            return
        if key == Qt.Key.Key_X and self._current_media_id is not None:
            self._flag = 0 if self._flag == -1 else -1
            self._update_header()
            self._cull_flash_hud.flash_flag(self._flag)
            self.flag_set_requested.emit(self._current_media_id, self._flag)
            if shift_held:
                self.navigate_requested.emit(1)
            event.accept()
            return
        if key == Qt.Key.Key_U and self._current_media_id is not None:
            self._flag = 0
            self._update_header()
            self._cull_flash_hud.flash_flag(0)
            self.flag_set_requested.emit(self._current_media_id, 0)
            if shift_held:
                self.navigate_requested.emit(1)
            event.accept()
            return
        if key == Qt.Key.Key_F and self._favourites_enabled and self._current_media_id is not None:
            self._is_favourite = not self._is_favourite
            self._update_header()
            self._cull_flash_hud.flash_favourite(self._is_favourite)
            self.favourite_requested.emit(self._current_media_id)
            if shift_held:
                self.navigate_requested.emit(1)
            event.accept()
            return
        super().keyPressEvent(event)
