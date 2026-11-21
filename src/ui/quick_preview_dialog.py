"""
Spacebar Quick Preview Fullscreen Lightbox for ThumbnailGridView.
Displays the photo borderless over a full-screen dimmed backdrop with floating
filename/rating/flag and shortcut pills, plus Left/Right navigation, 0-5 rating,
P flag toggle, Enter to open full Single View, and Space/Esc or backdrop click to close.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QImageReader, QKeyEvent, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QDialog, QFrame, QHBoxLayout, QLabel, QWidget

from src.utils.paths import is_cloud_only_file


class QuickPreviewDialog(QDialog):
    """Borderless full-screen Spacebar QuickLook overlay for rapid culling from Grid View."""

    open_full_requested = Signal(int)
    rate_requested = Signal(int, int)   # media_id, rating
    flag_requested = Signal(int)        # media_id
    navigate_requested = Signal(int)    # -1 or +1

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
        self._image_rect: QRect = QRect()

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

        # Floating bottom-center hint pill
        self._hint_pill = QFrame(self)
        self._hint_pill.setStyleSheet(
            "QFrame { background-color: rgba(15, 23, 42, 180); "
            "border: 1px solid rgba(148, 163, 184, 65); border-radius: 13px; }"
        )
        hint_lay = QHBoxLayout(self._hint_pill)
        hint_lay.setContentsMargins(14, 5, 14, 5)
        self.lbl_hint = QLabel(
            "Space / Esc: Close   •   ← / →: Prev / Next   •   0–5: Rate   •   P: Flag   •   Enter: Full View",
            self._hint_pill,
        )
        self.lbl_hint.setStyleSheet("color: #CBD5E1; font-size: 8.5pt; border: none; background: transparent;")
        hint_lay.addWidget(self.lbl_hint)

    def _update_header(self) -> None:
        stars = "★" * int(self._rating or 0) + "☆" * max(0, 5 - int(self._rating or 0))
        flag_str = "   ⚑ Picked" if self._flag == 1 else ("   ✕ Rejected" if self._flag == -1 else "")
        self.lbl_title.setText(self._filename)
        self.lbl_meta.setText(f"{stars}{flag_str}")
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
    ) -> None:
        self._current_media_id = int(media_id)
        self._filename = filename
        self._rating = int(rating or 0)
        self._flag = int(flag or 0)

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

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
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
        if Qt.Key.Key_0 <= key <= Qt.Key.Key_5 and self._current_media_id is not None:
            self._rating = key - Qt.Key.Key_0
            self._update_header()
            self.rate_requested.emit(self._current_media_id, self._rating)
            event.accept()
            return
        if key == Qt.Key.Key_P and self._current_media_id is not None:
            self._flag = 0 if self._flag == 1 else 1
            self._update_header()
            self.flag_requested.emit(self._current_media_id)
            event.accept()
            return
        super().keyPressEvent(event)
