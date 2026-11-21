"""
Spacebar Quick Preview Popover for ThumbnailGridView.
Provides instant, lightweight full-resolution / embedded-RAW preview with Left/Right
arrow navigation, 1-5 star rating, P flag toggle, Enter to open full Single View,
and Space/Esc to close.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImageReader, QKeyEvent, QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from src.utils.paths import is_cloud_only_file


class QuickPreviewDialog(QDialog):
    """Lightweight Spacebar QuickLook popover for rapid culling from Grid View."""

    open_full_requested = Signal(int)
    rate_requested = Signal(int, int)   # media_id, rating
    flag_requested = Signal(int)        # media_id
    navigate_requested = Signal(int)    # -1 or +1

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Quick Preview")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setModal(False)
        self.resize(860, 620)
        self.setStyleSheet(
            "QDialog { background-color: #11161D; border: 1px solid #334155; border-radius: 8px; }"
        )

        self._current_media_id: int | None = None
        self._current_pixmap: QPixmap = QPixmap()

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(12, 10, 12, 10)
        vbox.setSpacing(8)

        hdr = QHBoxLayout()
        self.lbl_title = QLabel("", self)
        self.lbl_title.setStyleSheet("color: #F8FAFC; font-size: 10pt; font-weight: 600;")
        hdr.addWidget(self.lbl_title, 1)

        self.lbl_meta = QLabel("", self)
        self.lbl_meta.setStyleSheet("color: #94A3B8; font-size: 9pt;")
        hdr.addWidget(self.lbl_meta)
        vbox.addLayout(hdr)

        self.lbl_image = QLabel(self)
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setMinimumSize(400, 320)
        self.lbl_image.setStyleSheet("background-color: #0B0F14; border-radius: 4px;")
        vbox.addWidget(self.lbl_image, 1)

        self.lbl_hint = QLabel(
            "Space / Esc: Close   •   ← / →: Prev / Next   •   1–5: Rate   •   P: Flag   •   Enter: Full View",
            self,
        )
        self.lbl_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_hint.setStyleSheet("color: #64748B; font-size: 8.5pt;")
        vbox.addWidget(self.lbl_hint)

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
        stars = "★" * int(rating or 0) + "☆" * max(0, 5 - int(rating or 0))
        flag_str = "  ⚑ Flagged" if flag else ""
        self.lbl_title.setText(filename)
        self.lbl_meta.setText(f"{stars}{flag_str}")

        pix = self._load_preview_pixmap(Path(file_path), fallback_pixmap)
        self._current_pixmap = pix
        self._update_scaled_pixmap()
        if not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()

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

                res = extract_tiff_embedded_jpeg(path, target_size=1280)
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
            if sz.isValid() and max(sz.width(), sz.height()) > 1440:
                reader.setScaledSize(sz.scaled(1440, 1440, Qt.AspectRatioMode.KeepAspectRatio))
            qimg = reader.read()
            if not qimg.isNull():
                return QPixmap.fromImage(qimg)
        except Exception:
            pass
        return fallback_pixmap if fallback_pixmap is not None else QPixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._current_pixmap.isNull():
            self.lbl_image.setText("Preview unavailable")
            return
        scaled = self._current_pixmap.scaled(
            self.lbl_image.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.lbl_image.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
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
            self.rate_requested.emit(self._current_media_id, key - Qt.Key.Key_0)
            event.accept()
            return
        if key == Qt.Key.Key_P and self._current_media_id is not None:
            self.flag_requested.emit(self._current_media_id)
            event.accept()
            return
        super().keyPressEvent(event)
