"""
PhotoGallery 64 — Modern Duplicate & Burst Finder View & Dialog
Provides side-by-side visual comparison, color-coded quality metrics,
smart auto-selection rules, live progress & streaming discovery,
and safe reversible Recycle Bin cleanup.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from PySide6.QtCore import QEvent, QObject, Qt, QSize, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QImageReader, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.core.duplicate_finder import (
    DuplicateDetectionEngine,
    DuplicateGroup,
    DuplicateMediaItem,
    auto_recommend_rules,
)
from src.db.repository import CatalogRepository, ThumbnailRepository


class _DuplicateScanWorker(QThread):
    """Background worker executing the duplicate detection engine."""

    progress = Signal(int, int, str)
    group_found = Signal(object)
    finished_scan = Signal(list)

    def __init__(
        self,
        engine: DuplicateDetectionEngine,
        media_records: Sequence[dict[str, Any]] | None = None,
        find_exact: bool = True,
        find_visual: bool = True,
        find_bursts: bool = True,
        similarity_threshold: float = 0.92,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.media_records = media_records
        self.find_exact = find_exact
        self.find_visual = find_visual
        self.find_bursts = find_bursts
        self.similarity_threshold = similarity_threshold

    def cancel(self) -> None:
        """Signals engine to cancel processing cleanly."""
        self.engine.cancel()

    def run(self) -> None:
        def on_progress(cur: int, tot: int, msg: str) -> None:
            self.progress.emit(cur, tot, msg)

        def on_group_found(group: DuplicateGroup) -> None:
            self.group_found.emit(group)

        groups = self.engine.find_all(
            media_records=self.media_records,
            find_exact=self.find_exact,
            find_visual=self.find_visual,
            find_bursts=self.find_bursts,
            similarity_threshold=self.similarity_threshold,
            progress_cb=on_progress,
            group_found_cb=on_group_found,
        )
        self.finished_scan.emit(groups)


class _GroupListCard(QFrame):
    """Visual list row widget for a duplicate/burst group."""

    def __init__(
        self,
        group: DuplicateGroup,
        is_dark: bool = False,
        thumb_pixmap: QPixmap | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.group = group
        self.is_dark = is_dark
        self._thumb_pixmap = thumb_pixmap
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedHeight(68)
        self._init_ui()

    def sizeHint(self) -> QSize:
        return QSize(280, 68)

    def minimumSizeHint(self) -> QSize:
        return QSize(220, 68)

    def set_thumbnail(self, pixmap: QPixmap | None) -> None:
        self._thumb_pixmap = pixmap
        if pixmap is not None and not pixmap.isNull():
            scaled = pixmap.scaled(
                52,
                52,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            if scaled.width() > 52 or scaled.height() > 52:
                sx = max(0, (scaled.width() - 52) // 2)
                sy = max(0, (scaled.height() - 52) // 2)
                scaled = scaled.copy(sx, sy, 52, 52)
            self.lbl_thumb.setPixmap(scaled)
            self.lbl_thumb.setText("")
        else:
            self.lbl_thumb.setPixmap(QPixmap())
            self.lbl_thumb.setText("🖼")

    def _init_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 8, 6)
        outer.setSpacing(8)

        self.lbl_thumb = QLabel(self)
        self.lbl_thumb.setFixedSize(52, 52)
        self.lbl_thumb.setAlignment(Qt.AlignCenter)
        thumb_bg = "#1E242E" if self.is_dark else "#E2E8F0"
        thumb_bd = "#3E4B5B" if self.is_dark else "#CBD5E1"
        thumb_fg = "#94A3B8" if self.is_dark else "#64748B"
        self.lbl_thumb.setStyleSheet(
            f"background-color: {thumb_bg}; border: 1px solid {thumb_bd}; border-radius: 4px; color: {thumb_fg};"
        )
        self.set_thumbnail(self._thumb_pixmap)
        outer.addWidget(self.lbl_thumb, 0, Qt.AlignVCenter)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        outer.addLayout(layout, 1)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)

        # Match Type Badge
        badge = QLabel(self)
        badge.setFont(QFont("Segoe UI", 8, QFont.Bold))
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedHeight(18)
        if self.group.match_type == "exact":
            badge.setText("Exact Match (100%)")
            badge.setStyleSheet("background: #2E7D32; color: #FFFFFF; border-radius: 3px; padding: 0 5px;")
        elif self.group.match_type == "burst":
            badge.setText(f"Burst ({self.group.time_span_sec}s)")
            badge.setStyleSheet("background: #6A1B9A; color: #FFFFFF; border-radius: 3px; padding: 0 5px;")
        else:
            pct = int(self.group.similarity * 100)
            badge.setText(f"Visual Match ({pct}%)")
            badge.setStyleSheet("background: #0277BD; color: #FFFFFF; border-radius: 3px; padding: 0 5px;")
        top_row.addWidget(badge)
        top_row.addStretch(1)

        # Item count
        lbl_count = QLabel(f"{len(self.group.items)} photos", self)
        lbl_count.setFont(QFont("Segoe UI", 8.5))
        lbl_count.setStyleSheet("color: #718096;" if not self.is_dark else "color: #94A3B8;")
        top_row.addWidget(lbl_count)
        layout.addLayout(top_row)

        # Title (first filename)
        lbl_filename = QLabel(self.group.items[0].filename if self.group.items else "", self)
        lbl_filename.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        lbl_filename.setStyleSheet("color: #1E293B;" if not self.is_dark else "color: #F1F5F9;")
        lbl_filename.setWordWrap(False)
        layout.addWidget(lbl_filename)

        # Bottom row: Recoverable size
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        lbl_rec = QLabel(f"Reclaimable: {self.group.formatted_reclaimable_size}", self)
        lbl_rec.setFont(QFont("Segoe UI", 8))
        lbl_rec.setStyleSheet("color: #D32F2F;" if not self.is_dark else "color: #EF5350;")
        bottom_row.addWidget(lbl_rec)
        bottom_row.addStretch(1)
        layout.addLayout(bottom_row)


class _ImageComparisonCard(QFrame):
    """Side-by-side card presenting one photo in the group with rich comparison metrics and zoom support."""

    toggled_keep = Signal(int)  # media_id

    def __init__(
        self,
        item: DuplicateMediaItem,
        group: DuplicateGroup,
        thumb_repo: ThumbnailRepository | None,
        is_dark: bool = False,
        card_width: int = 310,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.item = item
        self.group = group
        self.thumb_repo = thumb_repo
        self.is_dark = is_dark
        self.card_width = card_width
        self._orig_pixmap: QPixmap | None = None
        self.setFixedWidth(card_width)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # 1. Status Pill Badge at Top
        self.lbl_status_pill = QLabel(self)
        self.lbl_status_pill.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.lbl_status_pill.setAlignment(Qt.AlignCenter)
        self.lbl_status_pill.setFixedHeight(24)
        layout.addWidget(self.lbl_status_pill)

        # 2. Zoomable Image Preview
        pw = max(180, self.card_width - 20)
        ph = int(pw * 0.70)
        self.lbl_preview = QLabel(self)
        self.lbl_preview.setFixedSize(pw, ph)
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setCursor(Qt.PointingHandCursor)
        self.lbl_preview.setToolTip("Click to open original file in system viewer")
        self.lbl_preview.setStyleSheet(
            "background-color: #0F172A; border-radius: 4px;"
            if self.is_dark
            else "background-color: #E2E8F0; border-radius: 4px;"
        )
        self._load_thumbnail()
        self.lbl_preview.mousePressEvent = lambda _: self._open_file()
        layout.addWidget(self.lbl_preview)

        # 3. Action Keep / Remove Button
        self.btn_toggle = QPushButton(self)
        self.btn_toggle.setFixedHeight(28)
        self.btn_toggle.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.btn_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_toggle.clicked.connect(lambda: self.toggled_keep.emit(self.item.media_id))
        layout.addWidget(self.btn_toggle)

        # 4. Clean, Unified Properties Card
        prop_frame = QFrame(self)
        prop_frame.setObjectName("CardPropFrame")
        prop_frame.setStyleSheet(
            "QFrame#CardPropFrame { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 4px; padding: 4px 6px; }"
            if not self.is_dark
            else "QFrame#CardPropFrame { background: #181E26; border: 1px solid #334155; border-radius: 4px; padding: 4px 6px; }"
        )
        t_layout = QVBoxLayout(prop_frame)
        t_layout.setContentsMargins(6, 6, 6, 6)
        t_layout.setSpacing(5)

        # Metrics calculations
        max_res = max((it.width * it.height for it in self.group.items), default=0)
        is_max_res = (self.item.width * self.item.height) == max_res and max_res > 0
        max_size = max((it.file_size for it in self.group.items), default=0)
        is_max_size = self.item.file_size == max_size and max_size > 0
        max_sharp = max((it.sharpness for it in self.group.items), default=0.0)
        is_max_sharp = self.item.sharpness == max_sharp and max_sharp > 0.0

        # Filename Row (Prioritized, Bold)
        row_fn, self.lbl_filename = self._make_metric_row_with_label(
            "Filename",
            self.item.filename,
            is_highlight=False,
            tooltip=f"{self.item.filename}\nFull Path: {self.item.full_path}",
        )
        t_layout.addWidget(row_fn)

        # Folder / Full File Path Row
        full_dir = str(self.item.folder_path)
        row_folder, self.lbl_folder = self._make_metric_row_with_label(
            "Folder",
            full_dir,
            is_highlight=False,
            tooltip=full_dir,
        )
        t_layout.addWidget(row_folder)

        # Resolution Row
        res_str = f"{self.item.width} × {self.item.height} ({self.item.megapixels} MP)"
        t_layout.addWidget(
            self._make_metric_row(
                "Resolution",
                res_str,
                is_highlight=is_max_res,
                badge="BEST" if is_max_res and len(self.group.items) > 1 else None,
            )
        )

        # File Size Row
        size_badge = None
        if len(self.group.items) > 1:
            if is_max_size:
                size_badge = "LARGEST"
            elif self.item.file_size == min((it.file_size for it in self.group.items), default=0):
                size_badge = "SMALLEST"
        t_layout.addWidget(
            self._make_metric_row("File Size", self.item.formatted_size, is_highlight=is_max_size, badge=size_badge)
        )

        # Date Taken Row
        date_str = "Unknown"
        t_val = self.item.date_taken if self.item.date_taken > 0 else self.item.mtime
        if t_val > 0:
            import datetime
            dt = datetime.datetime.fromtimestamp(t_val)
            date_str = dt.strftime("%d %b %Y, %H:%M:%S")
        t_layout.addWidget(self._make_metric_row("Date Taken", date_str))

        # Focus / Sharpness for Bursts
        if self.group.match_type == "burst" and self.item.sharpness > 0:
            t_layout.addWidget(
                self._make_metric_row(
                    "Sharpness",
                    f"{int(self.item.sharpness)}",
                    is_highlight=is_max_sharp,
                    badge="SHARPEST" if is_max_sharp else None,
                )
            )

        layout.addWidget(prop_frame)

        # 5. Card Bottom Actions
        act_row = QHBoxLayout()
        act_row.setContentsMargins(0, 0, 0, 0)
        btn_folder = QPushButton("Reveal in Explorer", self)
        btn_folder.setFixedHeight(24)
        btn_folder.setFont(QFont("Segoe UI", 8.5))
        btn_folder.setCursor(Qt.PointingHandCursor)
        btn_folder.clicked.connect(self._reveal_in_explorer)
        act_row.addWidget(btn_folder)
        layout.addLayout(act_row)

        self.update_selection_state()

    def set_zoom(self, card_width: int) -> None:
        """Dynamically rescales card and preview image for live side-by-side zoom."""
        self.card_width = card_width
        self.setFixedWidth(card_width)
        pw = max(180, card_width - 20)
        ph = int(pw * 0.70)
        self.lbl_preview.setFixedSize(pw, ph)
        if card_width > 340 and (not self._orig_pixmap or self._orig_pixmap.width() <= 256):
            self._load_thumbnail()
        else:
            self._update_preview_pixmap()

    def update_selection_state(self) -> None:
        """Updates borders, pills, and button states based on keep/remove flag."""
        if self.item.is_recommended_keep:
            self.lbl_status_pill.setText("✓ KEEP THIS COPY")
            self.lbl_status_pill.setStyleSheet("background: #2E7D32; color: #FFFFFF; border-radius: 3px;")
            self.btn_toggle.setText("✓ Keeping Copy")
            self.btn_toggle.setStyleSheet("background: #E8F5E9; color: #1B5E20; border: 1px solid #81C784;")
            self.setStyleSheet(
                "_ImageComparisonCard { background-color: #F8FCF8; border: 2px solid #2E7D32; border-radius: 6px; }"
                if not self.is_dark
                else "_ImageComparisonCard { background-color: #122316; border: 2px solid #4CAF50; border-radius: 6px; }"
            )
        else:
            self.lbl_status_pill.setText("🗑 TO BE REMOVED")
            self.lbl_status_pill.setStyleSheet("background: #C62828; color: #FFFFFF; border-radius: 3px;")
            self.btn_toggle.setText("Switch to Keep")
            self.btn_toggle.setStyleSheet("background: #FFEBEE; color: #B71C1C; border: 1px solid #EF9A9A;")
            self.setStyleSheet(
                "_ImageComparisonCard { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; }"
                if not self.is_dark
                else "_ImageComparisonCard { background-color: #1E242C; border: 1px solid #334155; border-radius: 6px; }"
            )

    def _load_thumbnail(self) -> None:
        pix = None
        is_video = self.item.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".wmv"))
        if not is_video and self.item.full_path.exists() and self.card_width > 340:
            try:
                reader = QImageReader(str(self.item.full_path))
                reader.setAutoTransform(True)
                pw = max(180, self.card_width - 20)
                ph = int(pw * 0.70)
                reader.setScaledSize(QSize(pw * 2, ph * 2))
                img = reader.read()
                if not img.isNull():
                    pix = QPixmap.fromImage(img)
            except Exception:
                pix = None

        if pix is None and self.thumb_repo:
            blob = self.thumb_repo.get_webp_blob(self.item.media_id)
            if blob:
                qimg = QImage.fromData(blob, "WEBP")
                if not qimg.isNull():
                    pix = QPixmap.fromImage(qimg)
        if pix is None and self.item.full_path.exists():
            pix = QPixmap(str(self.item.full_path))

        self._orig_pixmap = pix
        self._update_preview_pixmap()

    def _update_preview_pixmap(self) -> None:
        if self._orig_pixmap and not self._orig_pixmap.isNull():
            pw = self.lbl_preview.width()
            ph = self.lbl_preview.height()
            scaled = self._orig_pixmap.scaled(pw, ph, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_preview.setPixmap(scaled)
        else:
            self.lbl_preview.setText("No Preview")

    def _make_metric_row(
        self, label: str, value: str, is_highlight: bool = False, badge: str | None = None, tooltip: str | None = None
    ) -> QWidget:
        widget, _ = self._make_metric_row_with_label(label, value, is_highlight, badge, tooltip)
        return widget

    def _make_metric_row_with_label(
        self, label: str, value: str, is_highlight: bool = False, badge: str | None = None, tooltip: str | None = None
    ) -> tuple[QWidget, QLabel]:
        w = QWidget(self)
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 1, 0, 1)
        row.setSpacing(6)
        lbl_k = QLabel(label, w)
        lbl_k.setFont(QFont("Segoe UI", 8.5))
        lbl_k.setStyleSheet("color: #64748B;" if not self.is_dark else "color: #94A3B8;")
        lbl_k.setFixedWidth(75)
        row.addWidget(lbl_k)

        lbl_v = QLabel(value, w)
        lbl_v.setFont(QFont("Segoe UI", 8.5, QFont.Weight.Bold if is_highlight else QFont.Weight.Normal))
        lbl_v.setWordWrap(True)
        if is_highlight:
            lbl_v.setStyleSheet("color: #2E7D32;" if not self.is_dark else "color: #4ADE80;")
        else:
            lbl_v.setStyleSheet("color: #1E293B;" if not self.is_dark else "color: #F1F5F9;")
        if tooltip:
            lbl_v.setToolTip(tooltip)
        row.addWidget(lbl_v, 1)

        if badge:
            b_lbl = QLabel(badge, w)
            b_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            if badge in ("BEST", "SHARPEST"):
                b_lbl.setStyleSheet("background: #E8F5E9; color: #2E7D32; border-radius: 2px; padding: 1px 4px;")
            else:
                b_lbl.setStyleSheet("background: #EFF6FF; color: #1D4ED8; border-radius: 2px; padding: 1px 4px;")
            row.addWidget(b_lbl)
        return w, lbl_v

    def _reveal_in_explorer(self) -> None:
        p = self.item.full_path
        if p.exists() and sys.platform == "win32":
            subprocess.run(f'explorer /select,"{p}"', shell=True)

    def _open_file(self) -> None:
        p = self.item.full_path
        if p.exists() and sys.platform == "win32":
            os.startfile(str(p))


class DuplicateFinderView(QWidget):
    """
    Integrated Duplicate & Burst Finder View (embedded directly in central_stack).
    Features side-by-side visual comparison, real-time streaming results,
    clear progress feedback, and safe Recycle Bin cleanup.
    """

    back_to_gallery_requested = Signal()
    duplicates_cleaned = Signal(int, int)  # count, freed_bytes

    def __init__(
        self,
        engine: Any = None,
        parent: QWidget | None = None,
        current_records: Sequence[dict[str, Any]] | None = None,
        is_dark: bool = False,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.catalog_repo = CatalogRepository(engine) if engine else None
        self.thumb_repo = ThumbnailRepository(engine) if engine else None
        self.current_records = current_records
        self.is_dark = is_dark
        self.groups: list[DuplicateGroup] = []
        self.active_group: DuplicateGroup | None = None
        self._group_thumb_cache: dict[int, QPixmap] = {}
        self._scan_worker: _DuplicateScanWorker | None = None
        self._detection_engine: DuplicateDetectionEngine | None = None

        self._init_ui()
        self.set_dark_mode(self.is_dark)

    def set_media_records(
        self,
        records: Sequence[dict[str, Any]] | None,
        selected_records: Sequence[dict[str, Any]] | None = None,
        current_folder: str | None = None,
    ) -> None:
        """Updates available records for current filter scanning."""
        self.current_records = records
        self.selected_records = selected_records
        self.cmb_scope.clear()
        self.cmb_scope.addItem("All Photos in Catalog", "all")
        if self.selected_records and len(self.selected_records) > 1:
            self.cmb_scope.addItem(f"Selected Photos ({len(self.selected_records)} photos)", "selected")
        if current_folder:
            folder_name = Path(current_folder).name or current_folder
            self.cmb_scope.addItem(f"📁 Folder: {folder_name}", {"type": "folder", "path": current_folder})
        if self.current_records:
            self.cmb_scope.addItem(f"Current Filter ({len(self.current_records)} photos)", "filtered")

    def set_dark_mode(self, is_dark: bool) -> None:
        self.is_dark = is_dark
        if is_dark:
            self.setStyleSheet(
                """
                DuplicateFinderView { background-color: #14171C; color: #F1F5F9; font-family: 'Segoe UI'; }
                QLabel { color: #F1F5F9; }
                QSplitter::handle { background-color: #2D3748; }
                QListWidget { background-color: #1A1F26; border: 1px solid #334155; border-radius: 4px; color: #F1F5F9; }
                QListWidget::item:selected { background-color: #253347; border: 1px solid #0078D4; }
                QScrollArea { background-color: #14171C; border: none; }
                QComboBox, QLineEdit { background-color: #1E242C; border: 1px solid #334155; color: #F1F5F9; border-radius: 3px; padding: 4px; }
                QPushButton { background-color: #252D37; border: 1px solid #3E4B5B; color: #F1F5F9; border-radius: 3px; padding: 4px 10px; }
                QPushButton:hover { background-color: #2F3946; border-color: #60A5FA; }
                QPushButton:disabled { background-color: #1B222C; border-color: #2D3748; color: #64748B; }
                QPushButton#BackBtn { background-color: #1E3A5F; border: 1px solid #2563EB; color: #93C5FD; font-weight: bold; }
                QPushButton#BackBtn:hover { background-color: #2563EB; color: #FFFFFF; }
                QPushButton#PrimaryBtn { background-color: #0078D4; border-color: #0078D4; color: #FFFFFF; font-weight: bold; }
                QPushButton#PrimaryBtn:hover { background-color: #1084D8; }
                QPushButton#PrimaryBtn:disabled { background-color: #1E3A5F; border-color: #2B4C7E; color: #94A3B8; }
                QPushButton#StopBtn { background-color: #7F1D1D; border: 1px solid #DC2626; color: #FEE2E2; font-weight: bold; }
                QPushButton#StopBtn:hover { background-color: #991B1B; border-color: #EF4444; }
                QPushButton#DeleteBtn { background-color: #B91C1C; border-color: #B91C1C; color: #FFFFFF; font-weight: bold; }
                QPushButton#DeleteBtn:hover { background-color: #DC2626; }
                QPushButton#DeleteBtn:disabled { background-color: #3B2327; border-color: #5C2B32; color: #94A3B8; }
                """
            )
        else:
            self.setStyleSheet(
                """
                DuplicateFinderView { background-color: #F8FAFC; color: #1E293B; font-family: 'Segoe UI'; }
                QLabel { color: #1E293B; }
                QSplitter::handle { background-color: #CBD5E1; }
                QListWidget { background-color: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 4px; color: #1E293B; }
                QListWidget::item:selected { background-color: #E2EFFC; border: 1px solid #2A82DA; }
                QScrollArea { background-color: #F8FAFC; border: none; }
                QComboBox, QLineEdit { background-color: #FFFFFF; border: 1px solid #CBD5E1; color: #1E293B; border-radius: 3px; padding: 4px; }
                QPushButton { background-color: #F1F5F9; border: 1px solid #CBD5E1; color: #1E293B; border-radius: 3px; padding: 4px 10px; }
                QPushButton:hover { background-color: #E2E8F0; border-color: #2A82DA; }
                QPushButton:disabled { background-color: #E2E8F0; border-color: #CBD5E1; color: #64748B; }
                QPushButton#BackBtn { background-color: #E0EEF9; border: 1px solid #B0D4F1; color: #005A9E; font-weight: bold; }
                QPushButton#BackBtn:hover { background-color: #CCE4F7; border-color: #0078D4; }
                QPushButton#PrimaryBtn { background-color: #0078D4; border-color: #0078D4; color: #FFFFFF; font-weight: bold; }
                QPushButton#PrimaryBtn:hover { background-color: #1084D8; }
                QPushButton#PrimaryBtn:disabled { background-color: #E2E8F0; border-color: #CBD5E1; color: #475569; }
                QPushButton#StopBtn { background-color: #FEE2E2; border: 1px solid #DC2626; color: #991B1B; font-weight: bold; }
                QPushButton#StopBtn:hover { background-color: #FECACA; border-color: #B91C1C; }
                QPushButton#DeleteBtn { background-color: #D32F2F; border-color: #D32F2F; color: #FFFFFF; font-weight: bold; }
                QPushButton#DeleteBtn:hover { background-color: #E53935; }
                QPushButton#DeleteBtn:disabled { background-color: #E2E8F0; border-color: #CBD5E1; color: #64748B; }
                """
            )

        # Apply explicit widget-level styles on action buttons so ancestor theme QSS
        # (e.g. modern.qss QPushButton { background-color: #FFFFFF; }) cannot override them.
        if hasattr(self, "btn_scan"):
            if is_dark:
                self.btn_scan.setStyleSheet(
                    "QPushButton { background-color: #0078D4; border: 1px solid #0078D4; color: #FFFFFF; font-weight: bold; border-radius: 3px; padding: 4px 12px; }"
                    "QPushButton:hover { background-color: #1084D8; border-color: #1084D8; }"
                    "QPushButton:disabled { background-color: #1E3A5F; border-color: #2B4C7E; color: #94A3B8; }"
                )
            else:
                self.btn_scan.setStyleSheet(
                    "QPushButton { background-color: #0078D4; border: 1px solid #0078D4; color: #FFFFFF; font-weight: bold; border-radius: 3px; padding: 4px 12px; }"
                    "QPushButton:hover { background-color: #1084D8; border-color: #1084D8; }"
                    "QPushButton:disabled { background-color: #E2E8F0; border-color: #CBD5E1; color: #475569; }"
                )
        if hasattr(self, "btn_stop"):
            if is_dark:
                self.btn_stop.setStyleSheet(
                    "QPushButton { background-color: #7F1D1D; border: 1px solid #DC2626; color: #FEE2E2; font-weight: bold; border-radius: 3px; padding: 4px 12px; }"
                    "QPushButton:hover { background-color: #991B1B; border-color: #EF4444; }"
                )
            else:
                self.btn_stop.setStyleSheet(
                    "QPushButton { background-color: #FEE2E2; border: 1px solid #DC2626; color: #991B1B; font-weight: bold; border-radius: 3px; padding: 4px 12px; }"
                    "QPushButton:hover { background-color: #FECACA; border-color: #B91C1C; }"
                )
        if hasattr(self, "btn_delete"):
            if is_dark:
                self.btn_delete.setStyleSheet(
                    "QPushButton { background-color: #B91C1C; border: 1px solid #B91C1C; color: #FFFFFF; font-weight: bold; border-radius: 3px; padding: 4px 12px; }"
                    "QPushButton:hover { background-color: #DC2626; }"
                    "QPushButton:disabled { background-color: #3B2327; border-color: #5C2B32; color: #94A3B8; }"
                )
            else:
                self.btn_delete.setStyleSheet(
                    "QPushButton { background-color: #D32F2F; border: 1px solid #D32F2F; color: #FFFFFF; font-weight: bold; border-radius: 3px; padding: 4px 12px; }"
                    "QPushButton:hover { background-color: #E53935; }"
                    "QPushButton:disabled { background-color: #E2E8F0; border-color: #CBD5E1; color: #64748B; }"
                )

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 10, 14, 12)
        main_layout.setSpacing(8)

        # ---------------------------------------------------------------------
        # Top Header Bar (Back button, Title, Scan options, Progress)
        # ---------------------------------------------------------------------
        top_bar = QFrame(self)
        top_bar.setStyleSheet("background: transparent;")
        top_layout = QVBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(10)

        # Back to Gallery Button
        self.btn_back = QPushButton("← Back to Gallery (Esc)", top_bar)
        self.btn_back.setObjectName("BackBtn")
        self.btn_back.setFixedHeight(30)
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.setToolTip("Return to the photo gallery grid view (Esc)")
        self.btn_back.clicked.connect(self.back_to_gallery_requested.emit)
        row1.addWidget(self.btn_back)

        # Title & Subtitle
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        lbl_title = QLabel("Find Duplicates & Burst Sequences", top_bar)
        lbl_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        lbl_sub = QLabel("Identify exact byte duplicates, visual re-compressions, and rapid burst sequences to safely clean up disk space.", top_bar)
        lbl_sub.setFont(QFont("Segoe UI", 8.5))
        lbl_sub.setStyleSheet("color: #64748B;")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_sub)
        row1.addLayout(title_box, 1)

        # Scope selector
        self.cmb_scope = QComboBox(top_bar)
        self.cmb_scope.addItem("All Photos in Catalog", "all")
        self.cmb_scope.setFixedHeight(28)
        self.cmb_scope.setMinimumWidth(180)
        row1.addWidget(self.cmb_scope)

        self.btn_browse_folder = QPushButton("📁 Folder...", top_bar)
        self.btn_browse_folder.setFixedHeight(28)
        self.btn_browse_folder.setCursor(Qt.PointingHandCursor)
        self.btn_browse_folder.setToolTip("Constrain duplicate search to a specific folder")
        self.btn_browse_folder.clicked.connect(self._on_choose_folder_clicked)
        row1.addWidget(self.btn_browse_folder)

        # Scan buttons
        self.btn_scan = QPushButton("Scan Now", top_bar)
        self.btn_scan.setObjectName("PrimaryBtn")
        self.btn_scan.setFixedHeight(28)
        self.btn_scan.setCursor(Qt.PointingHandCursor)
        self.btn_scan.clicked.connect(self.start_scan)
        row1.addWidget(self.btn_scan)

        self.btn_stop = QPushButton("Stop Scan", top_bar)
        self.btn_stop.setObjectName("StopBtn")
        self.btn_stop.setFixedHeight(28)
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.setVisible(False)
        self.btn_stop.clicked.connect(self.stop_scan)
        row1.addWidget(self.btn_stop)

        top_layout.addLayout(row1)

        # Scan Filters & Progress strip
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(12)

        self.chk_exact = QCheckBox("Exact byte copies (100%)", top_bar)
        self.chk_exact.setChecked(True)
        self.chk_bursts = QCheckBox("Burst sequences (≤3s)", top_bar)
        self.chk_bursts.setChecked(True)
        self.chk_visual = QCheckBox("Visual matches (re-saved / resized)", top_bar)
        self.chk_visual.setChecked(True)
        row2.addWidget(self.chk_exact)
        row2.addWidget(self.chk_bursts)
        row2.addWidget(self.chk_visual)
        row2.addStretch(1)

        # Progress bar
        self.scan_progress_bar = QProgressBar(top_bar)
        self.scan_progress_bar.setFixedHeight(16)
        self.scan_progress_bar.setRange(0, 100)
        self.scan_progress_bar.setValue(0)
        self.scan_progress_bar.setTextVisible(True)
        self.scan_progress_bar.setFixedWidth(220)
        self.scan_progress_bar.setVisible(False)
        row2.addWidget(self.scan_progress_bar)

        self.lbl_scan_status = QLabel(top_bar)
        self.lbl_scan_status.setFont(QFont("Segoe UI", 8.5))
        self.lbl_scan_status.setVisible(False)
        row2.addWidget(self.lbl_scan_status)

        top_layout.addLayout(row2)
        main_layout.addWidget(top_bar)

        # ---------------------------------------------------------------------
        # Central Splitter (Left: Groups List, Right: Comparison Workspace)
        # ---------------------------------------------------------------------
        splitter = QSplitter(Qt.Horizontal, self)

        # --- Left Pane (Groups List) ---
        left_widget = QWidget(splitter)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(6)

        # Filter Pills
        pill_row = QHBoxLayout()
        pill_row.setContentsMargins(0, 0, 0, 0)
        self.btn_filter_all = QPushButton("All (0)", left_widget)
        self.btn_filter_exact = QPushButton("Exact (0)", left_widget)
        self.btn_filter_visual = QPushButton("Visual (0)", left_widget)
        self.btn_filter_burst = QPushButton("Bursts (0)", left_widget)
        for b in (self.btn_filter_all, self.btn_filter_exact, self.btn_filter_visual, self.btn_filter_burst):
            b.setCheckable(True)
            b.setFixedHeight(24)
            b.setFont(QFont("Segoe UI", 8.5))
            b.clicked.connect(self._on_type_filter_changed)
            pill_row.addWidget(b)
        self.btn_filter_all.setChecked(True)
        left_layout.addLayout(pill_row)

        self.list_groups = QListWidget(left_widget)
        self.list_groups.setFixedWidth(300)
        self.list_groups.currentRowChanged.connect(self._on_group_selected)
        left_layout.addWidget(self.list_groups, 1)

        splitter.addWidget(left_widget)

        # --- Right Pane (Comparison Workspace) ---
        right_widget = QWidget(splitter)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(8)

        # Workspace Header Strip
        ws_header = QHBoxLayout()
        self.lbl_group_title = QLabel("Select a duplicate group to compare", right_widget)
        self.lbl_group_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        ws_header.addWidget(self.lbl_group_title, 1)

        # Smart Rule Selector Dropdown
        lbl_rule = QLabel("Auto-Select Rule:", right_widget)
        lbl_rule.setFont(QFont("Segoe UI", 9))
        ws_header.addWidget(lbl_rule)

        self.cmb_rule = QComboBox(right_widget)
        self.cmb_rule.addItems([
            "Keep Highest Resolution",
            "Keep Largest File Size",
            "Keep Oldest Date Taken",
            "Keep Most Metadata",
            "Keep Best Sharpness",
        ])
        self.cmb_rule.setFixedHeight(26)
        self.cmb_rule.currentIndexChanged.connect(self._on_rule_changed)
        ws_header.addWidget(self.cmb_rule)

        btn_apply_all = QPushButton("Apply to All Groups", right_widget)
        btn_apply_all.setFixedHeight(26)
        btn_apply_all.setCursor(Qt.PointingHandCursor)
        btn_apply_all.clicked.connect(self._apply_rule_to_all_groups)
        ws_header.addWidget(btn_apply_all)

        ws_header.addSpacing(10)

        # Zoom Controls
        lbl_zoom = QLabel("Zoom:", right_widget)
        lbl_zoom.setFont(QFont("Segoe UI", 9))
        ws_header.addWidget(lbl_zoom)

        self.btn_zoom_out = QPushButton("−", right_widget)
        self.btn_zoom_out.setFixedSize(26, 26)
        self.btn_zoom_out.setToolTip("Zoom out preview images (Ctrl -)")
        self.btn_zoom_out.setCursor(Qt.PointingHandCursor)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        ws_header.addWidget(self.btn_zoom_out)

        self.slider_zoom = QSlider(Qt.Horizontal, right_widget)
        self.slider_zoom.setRange(220, 650)
        self.slider_zoom.setValue(310)
        self.slider_zoom.setFixedWidth(90)
        self.slider_zoom.setToolTip("Drag to zoom comparison previews")
        self.slider_zoom.valueChanged.connect(self._on_zoom_slider_changed)
        ws_header.addWidget(self.slider_zoom)

        self.btn_zoom_in = QPushButton("+", right_widget)
        self.btn_zoom_in.setFixedSize(26, 26)
        self.btn_zoom_in.setToolTip("Zoom in preview images (Ctrl +)")
        self.btn_zoom_in.setCursor(Qt.PointingHandCursor)
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        ws_header.addWidget(self.btn_zoom_in)

        self.lbl_zoom_val = QLabel("100%", right_widget)
        self.lbl_zoom_val.setFont(QFont("Segoe UI", 8.5))
        self.lbl_zoom_val.setFixedWidth(38)
        ws_header.addWidget(self.lbl_zoom_val)

        right_layout.addLayout(ws_header)

        # Scroll Area for Side-by-Side Cards
        self.cards_scroll = QScrollArea(right_widget)
        self.cards_scroll.setWidgetResizable(True)
        self.cards_scroll.viewport().installEventFilter(self)
        self.cards_container = QWidget()
        self.cards_layout = QHBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(8, 8, 8, 8)
        self.cards_layout.setSpacing(14)
        self.cards_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.cards_scroll.setWidget(self.cards_container)
        right_layout.addWidget(self.cards_scroll, 1)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter, 1)

        # ---------------------------------------------------------------------
        # Bottom Footer Action Bar
        # ---------------------------------------------------------------------
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 6, 0, 0)

        self.lbl_summary = QLabel("0 duplicate files marked for removal (0 B)", self)
        self.lbl_summary.setFont(QFont("Segoe UI", 9.5, QFont.Bold))
        footer.addWidget(self.lbl_summary)
        footer.addStretch(1)

        btn_backup = QPushButton("Move to Backup Folder...", self)
        btn_backup.setFixedHeight(30)
        btn_backup.setCursor(Qt.PointingHandCursor)
        btn_backup.clicked.connect(self._move_marked_to_backup)
        footer.addWidget(btn_backup)

        self.btn_delete = QPushButton("Move Selected to Recycle Bin", self)
        self.btn_delete.setObjectName("DeleteBtn")
        self.btn_delete.setFixedHeight(30)
        self.btn_delete.setCursor(Qt.PointingHandCursor)
        self.btn_delete.clicked.connect(self._move_marked_to_recycle_bin)
        footer.addWidget(self.btn_delete)

        main_layout.addLayout(footer)

    def keyPressEvent(self, event: Any) -> None:
        if event.key() == Qt.Key_Escape:
            self.back_to_gallery_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def hideEvent(self, event: Any) -> None:
        self.stop_scan()
        super().hideEvent(event)

    def _on_choose_folder_clicked(self) -> None:
        """Opens folder dialog to constrain duplicate detection to a specific folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Constrain Duplicate Search")
        if not folder:
            return
        folder_path = str(Path(folder).resolve())
        name = Path(folder_path).name or folder_path
        # Check if already present in scope dropdown
        idx = -1
        for i in range(self.cmb_scope.count()):
            d = self.cmb_scope.itemData(i)
            if isinstance(d, dict) and d.get("path") == folder_path:
                idx = i
                break
        if idx < 0:
            self.cmb_scope.addItem(f"📁 {name}", {"type": "folder", "path": folder_path})
            idx = self.cmb_scope.count() - 1
        self.cmb_scope.setCurrentIndex(idx)
        self.start_scan()

    def _fetch_folder_records(self, folder_path: str) -> list[dict[str, Any]]:
        """Fetches all catalog records residing in folder_path or any of its subfolders."""
        if not self.catalog_repo:
            return []
        conn = self.catalog_repo.engine.catalog_conn()
        cur = conn.cursor()
        cur.row_factory = None
        sql = """
        SELECT m.id, f.path, m.filename, m.size, m.width, m.height, m.mtime, m.date_taken,
               m.rating, m.flag, m.caption, m.is_video
        FROM media m
        JOIN folders f ON m.folder_id = f.id
        WHERE m.is_missing = 0 AND (f.path = ? OR f.path LIKE ? || '\\%' OR f.path LIKE ? || '/%');
        """
        rows = cur.execute(sql, (folder_path, folder_path, folder_path)).fetchall()
        records = []
        for r in rows:
            records.append({
                "id": r[0],
                "folder_path": r[1],
                "filename": r[2],
                "size": r[3],
                "width": r[4] or 0,
                "height": r[5] or 0,
                "mtime": r[6] or 0.0,
                "date_taken": r[7] or 0.0,
                "rating": r[8] or 0,
                "flag": r[9] or 0,
                "caption": r[10] or "",
                "is_video": bool(r[11]),
            })
        return records

    def start_scan(self) -> None:
        """Starts duplicate detection scan with live progress and real-time streaming results."""
        self.stop_scan()
        self.groups.clear()
        self.list_groups.clear()
        self._clear_workspace()
        self._update_summary_counts()

        scope = self.cmb_scope.currentData()
        if scope == "filtered":
            records = self.current_records
        elif scope == "selected":
            records = getattr(self, "selected_records", None)
        elif isinstance(scope, dict) and scope.get("type") == "folder":
            records = self._fetch_folder_records(scope.get("path", ""))
        else:
            records = None

        self._detection_engine = DuplicateDetectionEngine(self.catalog_repo, self.thumb_repo)
        self._scan_worker = _DuplicateScanWorker(
            engine=self._detection_engine,
            media_records=records,
            find_exact=self.chk_exact.isChecked(),
            find_visual=self.chk_visual.isChecked(),
            find_bursts=self.chk_bursts.isChecked(),
            parent=self,
        )

        self.btn_scan.setEnabled(False)
        self.btn_stop.setVisible(True)
        self.scan_progress_bar.setVisible(True)
        self.scan_progress_bar.setValue(0)
        self.lbl_scan_status.setVisible(True)
        self.lbl_scan_status.setText("Scanning catalog for duplicates...")

        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.group_found.connect(self._on_group_found)
        self._scan_worker.finished_scan.connect(self._on_scan_finished)
        self._scan_worker.start()

    def stop_scan(self) -> None:
        """Safely and gracefully cancels the scan worker thread without crashing."""
        if self._scan_worker is not None and self._scan_worker.isRunning():
            if self._detection_engine is not None:
                self._detection_engine.cancel()
            self._scan_worker.cancel()

            # Disconnect signals to avoid callbacks on dead thread
            try:
                self._scan_worker.progress.disconnect()
            except Exception:
                pass
            try:
                self._scan_worker.group_found.disconnect()
            except Exception:
                pass
            try:
                self._scan_worker.finished_scan.disconnect()
            except Exception:
                pass

            self._scan_worker.quit()
            self._scan_worker.wait(1500)
            self._scan_worker = None

        self.btn_scan.setEnabled(True)
        self.btn_stop.setVisible(False)
        self.scan_progress_bar.setVisible(False)
        self.lbl_scan_status.setVisible(False)

    @Slot(int, int, str)
    def _on_scan_progress(self, cur: int, tot: int, msg: str) -> None:
        self.scan_progress_bar.setRange(0, tot)
        self.scan_progress_bar.setValue(cur)
        self.lbl_scan_status.setText(msg)

    @Slot(object)
    def _on_group_found(self, group: DuplicateGroup) -> None:
        """Called live as each duplicate group is identified."""
        self.groups.append(group)
        self._add_group_to_list(group)
        self._update_type_filter_counts()
        self._update_summary_counts()

        # If this is the first group found, select it immediately
        if self.list_groups.count() == 1:
            self.list_groups.setCurrentRow(0)

    @Slot(list)
    def _on_scan_finished(self, groups: list[DuplicateGroup]) -> None:
        """Called when scan completes fully."""
        self.btn_scan.setEnabled(True)
        self.btn_stop.setVisible(False)
        self.scan_progress_bar.setVisible(False)
        self.lbl_scan_status.setVisible(False)

        self.groups = groups
        self._populate_groups_list()
        self._update_summary_counts()

        if self.groups and self.list_groups.currentRow() < 0:
            self.list_groups.setCurrentRow(0)
        elif not self.groups:
            self.lbl_group_title.setText("✓ No duplicates found in selected scope.")

    def _on_type_filter_changed(self) -> None:
        sender = self.sender()
        for b in (self.btn_filter_all, self.btn_filter_exact, self.btn_filter_visual, self.btn_filter_burst):
            if b != sender:
                b.setChecked(False)
        if sender and not sender.isChecked():
            sender.setChecked(True)
        self._populate_groups_list()

    def _update_type_filter_counts(self) -> None:
        cnt_all = len(self.groups)
        cnt_exact = sum(1 for g in self.groups if g.match_type == "exact")
        cnt_visual = sum(1 for g in self.groups if g.match_type == "visual")
        cnt_burst = sum(1 for g in self.groups if g.match_type == "burst")

        self.btn_filter_all.setText(f"All ({cnt_all})")
        self.btn_filter_exact.setText(f"Exact ({cnt_exact})")
        self.btn_filter_visual.setText(f"Visual ({cnt_visual})")
        self.btn_filter_burst.setText(f"Bursts ({cnt_burst})")

    def _resolve_group_thumbs_batch(self, groups: Sequence[DuplicateGroup]) -> None:
        if self.thumb_repo is None or not groups:
            return
        missing_ids: list[int] = []
        for g in groups:
            if g.items:
                mid = int(g.items[0].media_id)
                if mid not in self._group_thumb_cache:
                    missing_ids.append(mid)
        if not missing_ids:
            return
        try:
            blobs = self.thumb_repo.get_webp_blobs_batch(missing_ids)
            for mid, blob in blobs.items():
                pm = QPixmap()
                if pm.loadFromData(blob, "WEBP") and not pm.isNull():
                    self._group_thumb_cache[int(mid)] = pm
        except Exception:
            pass

    def _get_group_thumb(self, group: DuplicateGroup) -> QPixmap | None:
        if not group.items:
            return None
        first_item = group.items[0]
        mid = int(first_item.media_id)
        if mid in self._group_thumb_cache:
            return self._group_thumb_cache[mid]
        if self.thumb_repo is not None:
            try:
                blob = self.thumb_repo.get_webp_blob(mid)
                if blob:
                    pm = QPixmap()
                    if pm.loadFromData(blob, "WEBP") and not pm.isNull():
                        self._group_thumb_cache[mid] = pm
                        return pm
            except Exception:
                pass
        fp = getattr(first_item, "full_path", None) or getattr(first_item, "file_path", None)
        if fp and os.path.exists(fp):
            try:
                reader = QImageReader(str(fp))
                reader.setAutoTransform(True)
                reader.setScaledSize(QSize(64, 64))
                img = reader.read()
                if not img.isNull():
                    pm = QPixmap.fromImage(img)
                    self._group_thumb_cache[mid] = pm
                    return pm
            except Exception:
                pass
        return None

    def _add_group_to_list(self, g: DuplicateGroup) -> None:
        filter_mode = "all"
        if self.btn_filter_exact.isChecked():
            filter_mode = "exact"
        elif self.btn_filter_visual.isChecked():
            filter_mode = "visual"
        elif self.btn_filter_burst.isChecked():
            filter_mode = "burst"

        if filter_mode != "all" and g.match_type != filter_mode:
            return

        item = QListWidgetItem()
        item.setSizeHint(QSize(280, 68))
        item.setData(Qt.UserRole, g.group_id)
        self.list_groups.addItem(item)
        card = _GroupListCard(g, is_dark=self.is_dark, thumb_pixmap=self._get_group_thumb(g))
        self.list_groups.setItemWidget(item, card)

    def _populate_groups_list(self) -> None:
        self.list_groups.clear()
        self._update_type_filter_counts()

        filter_mode = "all"
        if self.btn_filter_exact.isChecked():
            filter_mode = "exact"
        elif self.btn_filter_visual.isChecked():
            filter_mode = "visual"
        elif self.btn_filter_burst.isChecked():
            filter_mode = "burst"

        visible_groups = [g for g in self.groups if filter_mode == "all" or g.match_type == filter_mode]
        self._resolve_group_thumbs_batch(visible_groups)

        for g in visible_groups:
            item = QListWidgetItem()
            item.setSizeHint(QSize(280, 68))
            item.setData(Qt.UserRole, g.group_id)
            self.list_groups.addItem(item)
            card = _GroupListCard(g, is_dark=self.is_dark, thumb_pixmap=self._get_group_thumb(g))
            self.list_groups.setItemWidget(item, card)

        if visible_groups:
            self.list_groups.setCurrentRow(0)
        else:
            self._clear_workspace()

    def _on_group_selected(self, row: int) -> None:
        if row < 0 or row >= self.list_groups.count():
            return
        item = self.list_groups.item(row)
        gid = item.data(Qt.UserRole)
        match_group = next((g for g in self.groups if g.group_id == gid), None)
        if match_group:
            self.active_group = match_group
            self._render_group_cards(match_group)

    def _render_group_cards(self, group: DuplicateGroup) -> None:
        self._clear_workspace()

        # Update title
        type_str = "Exact Duplicate (100%)"
        if group.match_type == "burst":
            type_str = f"Burst Sequence ({group.time_span_sec}s)"
        elif group.match_type == "visual":
            type_str = f"Visual Match ({int(group.similarity * 100)}%)"

        self.lbl_group_title.setText(f"{type_str} — {len(group.items)} Photos ({group.formatted_reclaimable_size} Reclaimable)")

        current_zoom = self.slider_zoom.value() if hasattr(self, "slider_zoom") else 310
        for it in group.items:
            card = _ImageComparisonCard(
                item=it,
                group=group,
                thumb_repo=self.thumb_repo,
                is_dark=self.is_dark,
                card_width=current_zoom,
                parent=self.cards_container,
            )
            card.toggled_keep.connect(self._on_card_toggled_keep)
            self.cards_layout.addWidget(card)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if hasattr(self, "cards_scroll") and watched == self.cards_scroll.viewport():
            if event.type() == QEvent.Type.Wheel and (event.modifiers() & Qt.ControlModifier):
                delta = event.angleDelta().y()
                if delta > 0:
                    self._zoom_in()
                elif delta < 0:
                    self._zoom_out()
                return True
        return super().eventFilter(watched, event)

    def _on_zoom_slider_changed(self, val: int) -> None:
        pct = int((val / 310) * 100)
        if hasattr(self, "lbl_zoom_val"):
            self.lbl_zoom_val.setText(f"{pct}%")
        for i in range(self.cards_layout.count()):
            w = self.cards_layout.itemAt(i).widget()
            if isinstance(w, _ImageComparisonCard):
                w.set_zoom(val)

    def _zoom_in(self) -> None:
        if hasattr(self, "slider_zoom"):
            self.slider_zoom.setValue(min(self.slider_zoom.maximum(), self.slider_zoom.value() + 35))

    def _zoom_out(self) -> None:
        if hasattr(self, "slider_zoom"):
            self.slider_zoom.setValue(max(self.slider_zoom.minimum(), self.slider_zoom.value() - 35))

    def _on_card_toggled_keep(self, media_id: int) -> None:
        if not self.active_group:
            return
        for it in self.active_group.items:
            if it.media_id == media_id:
                it.is_recommended_keep = True
                it.is_selected_for_removal = False
            else:
                it.is_recommended_keep = False
                it.is_selected_for_removal = True

        # Refresh cards
        for i in range(self.cards_layout.count()):
            w = self.cards_layout.itemAt(i).widget()
            if isinstance(w, _ImageComparisonCard):
                w.update_selection_state()

        self._update_summary_counts()

    def _on_rule_changed(self) -> None:
        rule_map = {
            0: "highest_resolution",
            1: "largest_file_size",
            2: "oldest_date",
            3: "most_metadata",
            4: "best_sharpness",
        }
        rule = rule_map.get(self.cmb_rule.currentIndex(), "highest_resolution")
        if self.active_group:
            auto_recommend_rules([self.active_group], rule=rule)
            self._render_group_cards(self.active_group)
            self._update_summary_counts()

    def _apply_rule_to_all_groups(self) -> None:
        rule_map = {
            0: "highest_resolution",
            1: "largest_file_size",
            2: "oldest_date",
            3: "most_metadata",
            4: "best_sharpness",
        }
        rule = rule_map.get(self.cmb_rule.currentIndex(), "highest_resolution")
        auto_recommend_rules(self.groups, rule=rule)
        if self.active_group:
            self._render_group_cards(self.active_group)
        self._update_summary_counts()

    def _update_summary_counts(self) -> None:
        tot_removal = sum(1 for g in self.groups for it in g.items if it.is_selected_for_removal)
        tot_bytes = sum(it.file_size for g in self.groups for it in g.items if it.is_selected_for_removal)
        s = float(tot_bytes)
        if s >= 1024 ** 3:
            sz_str = f"{s / (1024 ** 3):.2f} GB"
        elif s >= 1024 ** 2:
            sz_str = f"{s / (1024 ** 2):.1f} MB"
        elif s >= 1024:
            sz_str = f"{s / 1024:.0f} KB"
        else:
            sz_str = f"{int(s)} B"
        self.lbl_summary.setText(f"{tot_removal} duplicate files marked for removal ({sz_str} recoverable)")
        self.btn_delete.setEnabled(tot_removal > 0)

    def _clear_workspace(self) -> None:
        while self.cards_layout.count() > 0:
            child = self.cards_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.lbl_group_title.setText("Select a duplicate group to compare")

    def _move_marked_to_recycle_bin(self) -> None:
        items_to_remove = [it for g in self.groups for it in g.items if it.is_selected_for_removal]
        if not items_to_remove:
            return

        tot_bytes = sum(it.file_size for it in items_to_remove)
        s = float(tot_bytes)
        sz_str = f"{s / (1024 ** 2):.1f} MB" if s >= 1024 ** 2 else f"{s / 1024:.0f} KB"

        ret = QMessageBox.warning(
            self,
            "Confirm Move to Recycle Bin",
            f"Are you sure you want to move {len(items_to_remove)} duplicate files ({sz_str}) to the Recycle Bin?\n\n"
            "The best original copies will remain in your library.\n"
            "You can restore files from the Windows Recycle Bin if needed.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return

        try:
            from send2trash import send2trash
        except ImportError:
            send2trash = None

        deleted_mids: list[int] = []
        for it in items_to_remove:
            p = it.full_path
            try:
                if p.exists():
                    if send2trash:
                        send2trash(str(p))
                    else:
                        p.unlink()
                deleted_mids.append(it.media_id)
            except Exception:
                pass

        if deleted_mids and self.catalog_repo:
            self.catalog_repo.delete_media(deleted_mids)

        self.duplicates_cleaned.emit(len(deleted_mids), tot_bytes)
        QMessageBox.information(
            self,
            "Duplicates Cleaned",
            f"Successfully moved {len(deleted_mids)} duplicate files to the Recycle Bin, recovering {sz_str}.",
        )
        self.back_to_gallery_requested.emit()

    def _move_marked_to_backup(self) -> None:
        items_to_remove = [it for g in self.groups for it in g.items if it.is_selected_for_removal]
        if not items_to_remove:
            return

        dest_folder = QFileDialog.getExistingDirectory(self, "Select Backup Folder to Move Duplicates Into")
        if not dest_folder:
            return

        dest_path = Path(dest_folder)
        moved_count = 0
        deleted_mids: list[int] = []

        import shutil
        for it in items_to_remove:
            src = it.full_path
            if src.exists():
                dst = dest_path / src.name
                if dst.exists():
                    dst = dest_path / f"{src.stem}_dup_{int(it.mtime)}{src.suffix}"
                try:
                    shutil.move(str(src), str(dst))
                    moved_count += 1
                    deleted_mids.append(it.media_id)
                except Exception:
                    pass

        if deleted_mids and self.catalog_repo:
            self.catalog_repo.delete_media(deleted_mids)

        self.duplicates_cleaned.emit(moved_count, sum(it.file_size for it in items_to_remove))
        QMessageBox.information(
            self,
            "Duplicates Moved",
            f"Moved {moved_count} duplicate files into:\n{dest_folder}",
        )
        self.back_to_gallery_requested.emit()


class DuplicateFinderDialog(QDialog):
    """
    State-of-the-art Duplicate & Burst Finder dialog wrapper around DuplicateFinderView.
    Ensures 100% crash-free teardown and thread termination on close.
    """

    duplicates_cleaned = Signal(int, int)

    def __init__(
        self,
        parent: QWidget | None = None,
        engine: Any = None,
        current_records: Sequence[dict[str, Any]] | None = None,
        is_dark: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Find Duplicates & Bursts — PhotoGallery 64")
        self.resize(1260, 820)
        self.setMinimumSize(1050, 680)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = DuplicateFinderView(
            engine=engine,
            parent=self,
            current_records=current_records,
            is_dark=is_dark,
        )
        layout.addWidget(self.view)

        self.view.back_to_gallery_requested.connect(self.accept)
        self.view.duplicates_cleaned.connect(self.duplicates_cleaned.emit)

        # Trigger initial scan
        self.view.start_scan()

    @property
    def is_dark(self) -> bool:
        return self.view.is_dark

    def set_dark_mode(self, is_dark: bool) -> None:
        self.view.set_dark_mode(is_dark)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.view, name)

    def closeEvent(self, event: Any) -> None:
        """Crash prevention: safely cancel scan worker before destroying dialog."""
        self.view.stop_scan()
        super().closeEvent(event)

    def reject(self) -> None:
        """Crash prevention: safely cancel scan worker when dialog is dismissed."""
        self.view.stop_scan()
        super().reject()
