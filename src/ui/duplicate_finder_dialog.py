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

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSettings, Qt, QSize, QThread, Signal, Slot
from src.ui.theme_tokens import menu_stylesheet
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QImage, QImageReader, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.duplicate_finder import (
    DuplicateDetectionEngine,
    DuplicateGroup,
    DuplicateMediaItem,
    auto_recommend_rules,
    merge_duplicate_metadata,
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
        seed_media_id: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.media_records = media_records
        self.find_exact = find_exact
        self.find_visual = find_visual
        self.find_bursts = find_bursts
        self.similarity_threshold = similarity_threshold
        self.seed_media_id = seed_media_id

    def cancel(self) -> None:
        """Signals engine to cancel processing cleanly."""
        self.engine.cancel()

    def run(self) -> None:
        def on_progress(cur: int, tot: int, msg: str) -> None:
            self.progress.emit(cur, tot, msg)

        def on_group_found(group: DuplicateGroup) -> None:
            self.group_found.emit(group)

        if self.seed_media_id is not None:
            grp = self.engine.find_similar_to_photo(
                seed_media_id=self.seed_media_id,
                similarity_threshold=self.similarity_threshold,
                media_records=self.media_records,
                progress_cb=on_progress,
            )
            groups = [grp] if grp is not None else []
            if grp is not None:
                on_group_found(grp)
        else:
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
        elif self.group.match_type == "similar":
            pct = int(self.group.similarity * 100)
            badge.setText(f"Similar ({pct}%)")
            badge.setStyleSheet("background: #0F766E; color: #FFFFFF; border-radius: 3px; padding: 0 5px;")
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


class _FlowLayout(QLayout):
    """Responsive wrapping layout so comparison cards wrap cleanly without horizontal cutoff."""

    def __init__(self, parent: QWidget | None = None, margin: int = 0, spacing: int = 10) -> None:
        super().__init__(parent)
        self._items: list[Any] = []
        self._spacing = spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def __del__(self) -> None:
        while self._items:
            self.takeAt(0)

    def addItem(self, item: Any) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> Any:
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> Any:
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientations:
        return Qt.Orientations(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        m = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x = eff.x()
        y = eff.y()
        line_h = 0
        for item in self._items:
            w = item.widget()
            if w is not None and not w.isVisible():
                continue
            sz = item.sizeHint()
            next_x = x + sz.width() + self._spacing
            if next_x - self._spacing > eff.right() + 1 and line_h > 0:
                x = eff.x()
                y = y + line_h + self._spacing
                next_x = x + sz.width() + self._spacing
                line_h = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), sz))
            x = next_x
            line_h = max(line_h, sz.height())
        return (y + line_h - rect.y()) + m.bottom()


_CARD_DETAIL_KEYS: tuple[tuple[str, str], ...] = (
    ("filename", "Filename"),
    ("folder", "Folder Path"),
    ("metrics", "Resolution & File Size"),
    ("date_sharpness", "Date, Sharpness & Match %"),
)


def _get_card_detail_prefs() -> dict[str, bool]:
    s = QSettings("PhotoGallery64", "PhotoGallery64")
    return {
        k: bool(s.value(f"dup_finder/show_{k}", True, type=bool))
        for k, _ in _CARD_DETAIL_KEYS
    }


def _set_card_detail_pref(key: str, enabled: bool) -> None:
    s = QSettings("PhotoGallery64", "PhotoGallery64")
    s.setValue(f"dup_finder/show_{key}", bool(enabled))


class _ImageComparisonCard(QFrame):
    """Side-by-side WLPG card presenting one photo with corner keep/remove pill, crisp metrics, and clean actions."""

    toggled_keep = Signal(int)  # media_id
    recycle_single_requested = Signal(int)  # media_id
    find_similar_requested = Signal(int)  # media_id
    compare_requested = Signal(int)  # media_id
    card_selected = Signal(int)  # media_id

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
        self._is_selected: bool = False
        self._full_folder_str: str = str(self.item.folder_path)
        self.setFixedWidth(card_width)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # 1. Larger Zoomable Image Preview with floating top-left overlay status pill
        pw = max(180, self.card_width - 20)
        ph = int(pw * 0.72)
        self.lbl_preview = QLabel(self)
        self.lbl_preview.setFixedSize(pw, ph)
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setCursor(Qt.PointingHandCursor)
        self.lbl_preview.setToolTip("Double-click to inspect this group in Compare Mode")
        self.lbl_preview.setStyleSheet(
            "background-color: #0F172A; border-radius: 5px;"
            if self.is_dark
            else "background-color: #E2E8F0; border-radius: 5px;"
        )
        self._load_thumbnail()
        self.lbl_preview.mouseDoubleClickEvent = lambda _: self.compare_requested.emit(self.item.media_id)
        layout.addWidget(self.lbl_preview)

        self.lbl_status_pill = QLabel(self.lbl_preview)
        self.lbl_status_pill.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        self.lbl_status_pill.setAlignment(Qt.AlignCenter)
        self.lbl_status_pill.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.lbl_status_pill.move(8, 8)

        # 2. Compact 3-line metadata summary (matching MediaCardDelegate aesthetic)
        self.lbl_filename = QLabel(self.item.filename, self)
        self.lbl_filename.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.lbl_filename.setWordWrap(False)
        self.lbl_filename.setToolTip(f"{self.item.filename}\nFull Path: {self.item.full_path}")
        self.lbl_filename.setStyleSheet(
            "color: #F1F5F9; background: transparent; border: none;"
            if self.is_dark
            else "color: #0F172A; background: transparent; border: none;"
        )
        layout.addWidget(self.lbl_filename)

        self.lbl_folder = QLabel(self._full_folder_str, self)
        self.lbl_folder.setFont(QFont("Segoe UI", 8))
        self.lbl_folder.setWordWrap(False)
        self.lbl_folder.setToolTip(self._full_folder_str)
        self.lbl_folder.setStyleSheet(
            "color: #94A3B8; background: transparent; border: none;"
            if self.is_dark
            else "color: #64748B; background: transparent; border: none;"
        )
        self._update_folder_label()
        layout.addWidget(self.lbl_folder)

        res_vals = [it.width * it.height for it in self.group.items]
        max_res = max(res_vals, default=0)
        min_res = min(res_vals, default=0)
        res_differs = min_res < max_res
        is_max_res = (self.item.width * self.item.height) == max_res and max_res > 0
        is_equal_res = len(self.group.items) > 1 and (not res_differs) and max_res > 0

        size_vals = [it.file_size for it in self.group.items]
        max_size = max(size_vals, default=0)
        min_size = min(size_vals, default=0)
        size_differs = min_size < max_size
        is_max_size = self.item.file_size == max_size and max_size > 0

        sharp_vals = [it.sharpness for it in self.group.items]
        max_sharp = max(sharp_vals, default=0.0)
        min_sharp = min(sharp_vals, default=0.0)
        sharp_differs = min_sharp < max_sharp
        is_max_sharp = self.item.sharpness == max_sharp and max_sharp > 0.0

        self.metrics_widget = QWidget(self)
        self.metrics_widget.setStyleSheet("background: transparent; border: none;")
        metrics_row = QHBoxLayout(self.metrics_widget)
        metrics_row.setContentsMargins(0, 0, 0, 0)
        metrics_row.setSpacing(4)
        res_str = f"{self.item.width}×{self.item.height} ({self.item.megapixels} MP) • {self.item.formatted_size}"
        lbl_metrics = QLabel(res_str, self)
        lbl_metrics.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold if (res_differs and is_max_res) or (size_differs and is_max_size) else QFont.Weight.Normal))
        lbl_metrics.setStyleSheet(
            ("color: #4ADE80;" if (res_differs and is_max_res) else "color: #CBD5E1;") + " background: transparent; border: none;"
            if self.is_dark
            else ("color: #166534;" if (res_differs and is_max_res) else "color: #334155;") + " background: transparent; border: none;"
        )
        metrics_row.addWidget(lbl_metrics)
        if len(self.group.items) > 1 and res_differs and is_max_res:
            b_best = QLabel("BEST", self)
            b_best.setFont(QFont("Segoe UI", 6.5, QFont.Weight.Bold))
            b_best.setStyleSheet("background: #DCFCE7; color: #166534; border-radius: 3px; padding: 1px 4px; border: none;")
            metrics_row.addWidget(b_best)
        elif is_equal_res and not size_differs:
            b_eq = QLabel("=", self)
            b_eq.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            b_eq.setToolTip("Same resolution and file size across copies in this group")
            b_eq.setStyleSheet(
                "background: #1E293B; color: #94A3B8; border-radius: 3px; padding: 1px 5px; border: none;"
                if self.is_dark
                else "background: #F1F5F9; color: #64748B; border-radius: 3px; padding: 1px 5px; border: none;"
            )
            metrics_row.addWidget(b_eq)
        if len(self.group.items) > 1 and size_differs and is_max_size and not (res_differs and is_max_res):
            b_lg = QLabel("LARGEST", self)
            b_lg.setFont(QFont("Segoe UI", 6.5, QFont.Weight.Bold))
            b_lg.setStyleSheet("background: #EFF6FF; color: #1D4ED8; border-radius: 3px; padding: 1px 4px; border: none;")
            metrics_row.addWidget(b_lg)
        metrics_row.addStretch(1)
        layout.addWidget(self.metrics_widget)

        # Sub-detail row: Date Taken + Burst Sharpness + Similarity
        sub_parts: list[str] = []
        t_val = self.item.date_taken if self.item.date_taken > 0 else self.item.mtime
        if t_val > 0:
            import datetime
            sub_parts.append(datetime.datetime.fromtimestamp(t_val).strftime("%d %b %Y, %H:%M"))
        if self.group.match_type == "burst" and self.item.sharpness > 0:
            sharp_lbl = "Eye/Face" if getattr(self.item, "has_faces", False) else "Sharp"
            sharp_val = int(self.item.face_sharpness if getattr(self.item, "has_faces", False) else self.item.sharpness)
            sub_parts.append(f"{sharp_lbl} {sharp_val}" + (" ★" if (sharp_differs and is_max_sharp) else ""))
        if getattr(self.item, "similarity_score", 0.0) > 0.0:
            sub_parts.append(f"{int(round(self.item.similarity_score * 100))}% match")

        self.lbl_sub = QLabel(" • ".join(sub_parts) if sub_parts else "Unknown date", self)
        self.lbl_sub.setFont(QFont("Segoe UI", 7.5))
        self.lbl_sub.setStyleSheet(
            "color: #94A3B8; background: transparent; border: none;"
            if self.is_dark
            else "color: #64748B; background: transparent; border: none;"
        )
        layout.addWidget(self.lbl_sub)

        # 3. Clean Single Bottom Action Row
        act_row = QHBoxLayout()
        act_row.setContentsMargins(0, 2, 0, 0)
        act_row.setSpacing(5)

        self.btn_toggle = QPushButton(self)
        self.btn_toggle.setFixedHeight(26)
        self.btn_toggle.setFont(QFont("Segoe UI", 8.5, QFont.Weight.Bold))
        self.btn_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_toggle.clicked.connect(lambda: self.toggled_keep.emit(self.item.media_id))
        act_row.addWidget(self.btn_toggle, 1)

        self.btn_similar = QPushButton("Similar", self)
        self.btn_similar.setFixedHeight(26)
        self.btn_similar.setFont(QFont("Segoe UI", 8))
        self.btn_similar.setCursor(Qt.PointingHandCursor)
        self.btn_similar.setToolTip("Find photos visually similar to this image across your catalog")
        self.btn_similar.setStyleSheet(
            "QPushButton { background-color: #252D37; border: 1px solid #3E4B5B; color: #E2E8F0; border-radius: 3px; padding: 2px 6px; }"
            "QPushButton:hover { background-color: #2F3946; border-color: #60A5FA; }"
            if self.is_dark
            else "QPushButton { background-color: #F8FAFC; border: 1px solid #CBD5E1; color: #334155; border-radius: 3px; padding: 2px 6px; }"
            "QPushButton:hover { background-color: #EFF6FF; border-color: #0078D4; color: #005A9E; }"
        )
        self.btn_similar.clicked.connect(lambda: self.find_similar_requested.emit(self.item.media_id))
        act_row.addWidget(self.btn_similar)

        btn_folder = QPushButton(self)
        btn_folder.setFixedSize(26, 26)
        btn_folder.setCursor(Qt.PointingHandCursor)
        btn_folder.setToolTip("Open containing folder and select file in Windows Explorer")
        try:
            from src.ui.ribbon_toolbar import make_wlpg_icon
            btn_folder.setIcon(make_wlpg_icon("folder", 14))
            btn_folder.setIconSize(QSize(14, 14))
        except Exception:
            btn_folder.setText("📁")
        btn_folder.setStyleSheet(
            "QPushButton { background-color: #252D37; border: 1px solid #3E4B5B; color: #E2E8F0; border-radius: 3px; padding: 0px; }"
            "QPushButton:hover { background-color: #2F3946; border-color: #60A5FA; }"
            if self.is_dark
            else "QPushButton { background-color: #F8FAFC; border: 1px solid #CBD5E1; color: #334155; border-radius: 3px; padding: 0px; }"
            "QPushButton:hover { background-color: #EFF6FF; border-color: #0078D4; }"
        )
        btn_folder.clicked.connect(self._reveal_in_explorer)
        act_row.addWidget(btn_folder)

        self.btn_recycle_single = QPushButton("🗑", self)
        self.btn_recycle_single.setObjectName("CardDeleteBtn")
        self.btn_recycle_single.setFixedSize(26, 26)
        self.btn_recycle_single.setFont(QFont("Segoe UI", 9))
        self.btn_recycle_single.setCursor(Qt.PointingHandCursor)
        self.btn_recycle_single.setToolTip("Move this individual file to the Recycle Bin immediately")
        self.btn_recycle_single.setStyleSheet(
            "QPushButton { background-color: #3B1821; border: 1px solid #881337; color: #FDA4AF; border-radius: 3px; padding: 0px; }"
            "QPushButton:hover { background-color: #B91C1C; border-color: #DC2626; color: #FFFFFF; }"
            if self.is_dark
            else "QPushButton { background-color: #FFF1F2; border: 1px solid #FECDD3; color: #BE123C; border-radius: 3px; padding: 0px; }"
            "QPushButton:hover { background-color: #DC2626; border-color: #B91C1C; color: #FFFFFF; }"
        )
        self.btn_recycle_single.clicked.connect(lambda: self.recycle_single_requested.emit(self.item.media_id))
        act_row.addWidget(self.btn_recycle_single)
        layout.addLayout(act_row)

        self.update_selection_state()

        self.apply_detail_visibility()

    def apply_detail_visibility(self) -> None:
        prefs = _get_card_detail_prefs()
        if hasattr(self, "lbl_filename"):
            self.lbl_filename.setVisible(prefs.get("filename", True))
        if hasattr(self, "lbl_folder"):
            self.lbl_folder.setVisible(prefs.get("folder", True))
        if hasattr(self, "metrics_widget"):
            self.metrics_widget.setVisible(prefs.get("metrics", True))
        if hasattr(self, "lbl_sub"):
            self.lbl_sub.setVisible(prefs.get("date_sharpness", True))

    def _update_folder_label(self) -> None:
        if not hasattr(self, "lbl_folder"):
            return
        avail_w = max(120, self.card_width - 24)
        fm = QFontMetrics(self.lbl_folder.font())
        self.lbl_folder.setText(fm.elidedText(self._full_folder_str, Qt.TextElideMode.ElideMiddle, avail_w))

    def set_selected(self, selected: bool) -> None:
        self._is_selected = bool(selected)
        self.update_selection_state()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton:
            self.card_selected.emit(self.item.media_id)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event: Any) -> None:
        self.card_selected.emit(self.item.media_id)
        menu = QMenu(self)
        menu_qss = menu_stylesheet(is_dark=self.is_dark)
        menu.setStyleSheet(menu_qss)

        act_keep = menu.addAction("✓ Select to Keep" if not self.item.is_recommended_keep else "✓ Keeping This Copy")
        act_keep.triggered.connect(lambda: self.toggled_keep.emit(self.item.media_id))
        act_cmp = menu.addAction("View in Compare Mode")
        act_cmp.triggered.connect(lambda: self.compare_requested.emit(self.item.media_id))
        act_sim = menu.addAction("Find Visually Similar Photos")
        act_sim.triggered.connect(lambda: self.find_similar_requested.emit(self.item.media_id))
        act_exp = menu.addAction("Show in Explorer")
        act_exp.triggered.connect(self._reveal_in_explorer)

        menu.addSeparator()
        details_menu = menu.addMenu("Show Card Details")
        details_menu.setStyleSheet(menu_qss)
        prefs = _get_card_detail_prefs()
        for key, label in _CARD_DETAIL_KEYS:
            act_d = details_menu.addAction(label)
            act_d.setCheckable(True)
            act_d.setChecked(prefs.get(key, True))
            act_d.toggled.connect(lambda checked, k=key: self._on_toggle_card_detail(k, checked))

        menu.addSeparator()
        act_del = menu.addAction("🗑 Move File to Recycle Bin")
        act_del.triggered.connect(lambda: self.recycle_single_requested.emit(self.item.media_id))
        menu.exec(event.globalPos())

    def _on_toggle_card_detail(self, key: str, checked: bool) -> None:
        _set_card_detail_pref(key, checked)
        root = self.window() or self.parent()
        if root is not None:
            for card in root.findChildren(_ImageComparisonCard):
                card.apply_detail_visibility()
        else:
            self.apply_detail_visibility()

    def set_zoom(self, card_width: int) -> None:
        """Dynamically rescales card and preview image for live side-by-side zoom."""
        self.card_width = card_width
        self.setFixedWidth(card_width)
        pw = max(180, card_width - 20)
        ph = int(pw * 0.72)
        self.lbl_preview.setFixedSize(pw, ph)
        self.lbl_status_pill.raise_()
        self._update_folder_label()
        if card_width > 340 and (not self._orig_pixmap or self._orig_pixmap.width() <= 256):
            self._load_thumbnail()
        else:
            self._update_preview_pixmap()
        self.update_selection_state()

    def update_selection_state(self) -> None:
        """Updates borders, floating overlay pill, and button states based on keep/remove flag."""
        compact = self.card_width < 255
        if self.item.is_recommended_keep:
            self.lbl_status_pill.setText("✓ KEEP")
            self.lbl_status_pill.setStyleSheet(
                "background: rgba(21, 128, 61, 228); color: #FFFFFF; border-radius: 9px; padding: 2px 9px; border: 1px solid rgba(255,255,255,80);"
            )
            self.btn_toggle.setText("✓ Keep" if compact else "✓ Keeping")
            self.btn_toggle.setStyleSheet(
                "QPushButton { background-color: #14532D; color: #DCFCE7; border: 1px solid #22C55E; border-radius: 3px; padding: 2px 6px; }"
                "QPushButton:hover { background-color: #166534; }"
                if self.is_dark
                else "QPushButton { background-color: #E8F5E9; color: #1B5E20; border: 1px solid #66BB6A; border-radius: 3px; padding: 2px 6px; }"
                "QPushButton:hover { background-color: #C8E6C9; }"
            )
            self.setStyleSheet(
                "_ImageComparisonCard { background-color: #102316; border: 2px solid #22C55E; border-radius: 7px; }"
                if self.is_dark
                else "_ImageComparisonCard { background-color: #F4FBF6; border: 2px solid #16A34A; border-radius: 7px; }"
            )
        elif self.item.is_selected_for_removal:
            self.lbl_status_pill.setText("🗑 REMOVE")
            self.lbl_status_pill.setStyleSheet(
                "background: rgba(185, 28, 28, 228); color: #FFFFFF; border-radius: 9px; padding: 2px 9px; border: 1px solid rgba(255,255,255,80);"
            )
            self.btn_toggle.setText("Keep" if compact else "Keep Instead")
            self.btn_toggle.setStyleSheet(
                "QPushButton { background-color: #2B1B1F; color: #FECACA; border: 1px solid #B91C1C; border-radius: 3px; padding: 2px 6px; }"
                "QPushButton:hover { background-color: #3F2127; border-color: #EF4444; }"
                if self.is_dark
                else "QPushButton { background-color: #FFF5F5; color: #B91C1C; border: 1px solid #FCA5A5; border-radius: 3px; padding: 2px 6px; }"
                "QPushButton:hover { background-color: #FEE2E2; border-color: #EF4444; }"
            )
            self.setStyleSheet(
                f"_ImageComparisonCard {{ background-color: #1B2129; border: {'2px solid #0078D4' if self._is_selected else '1px solid #334155'}; border-radius: 7px; }}"
                if self.is_dark
                else f"_ImageComparisonCard {{ background-color: #FFFFFF; border: {'2px solid #0078D4' if self._is_selected else '1px solid #CBD5E1'}; border-radius: 7px; }}"
            )
        else:
            self.lbl_status_pill.setText("≈ SIMILAR")
            self.lbl_status_pill.setStyleSheet(
                "background: rgba(15, 118, 110, 228); color: #FFFFFF; border-radius: 9px; padding: 2px 9px; border: 1px solid rgba(255,255,255,80);"
            )
            self.btn_toggle.setText("Remove" if compact else "Mark Remove")
            self.btn_toggle.setStyleSheet(
                "QPushButton { background-color: #132A2E; color: #99F6E4; border: 1px solid #0F766E; border-radius: 3px; padding: 2px 8px; }"
                "QPushButton:hover { background-color: #164E63; }"
                if self.is_dark
                else "QPushButton { background-color: #F0FDFA; color: #115E59; border: 1px solid #5EEAD4; border-radius: 3px; padding: 2px 8px; }"
                "QPushButton:hover { background-color: #CCFBF1; }"
            )
            self.setStyleSheet(
                f"_ImageComparisonCard {{ background-color: #1B2129; border: {'2px solid #0078D4' if self._is_selected else '1px solid #334155'}; border-radius: 7px; }}"
                if self.is_dark
                else f"_ImageComparisonCard {{ background-color: #FFFFFF; border: {'2px solid #0078D4' if self._is_selected else '1px solid #CBD5E1'}; border-radius: 7px; }}"
            )
        self.lbl_status_pill.adjustSize()
        self.lbl_status_pill.raise_()

    def _load_thumbnail(self) -> None:
        pix = None
        is_video = self.item.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".wmv"))
        pw = max(180, self.card_width - 20)
        ph = int(pw * 0.72)

        # Fast path: use cached WebP thumbnail from SQLite (<0.2ms) when card <= 380px
        # so Sort & Group is near-instant instead of decoding 60+ full-res JPEGs on the UI thread.
        if self.thumb_repo and self.card_width <= 380:
            blob = self.thumb_repo.get_webp_blob(self.item.media_id)
            if blob:
                qimg = QImage.fromData(blob, "WEBP")
                if not qimg.isNull():
                    pix = QPixmap.fromImage(qimg)

        if pix is None and not is_video and self.item.full_path.exists():
            try:
                reader = QImageReader(str(self.item.full_path))
                reader.setAutoTransform(True)
                orig_sz = reader.size()
                if orig_sz.isValid() and (orig_sz.width() > pw * 2 or orig_sz.height() > ph * 2):
                    scaled_sz = orig_sz.scaled(pw * 2, ph * 2, Qt.AspectRatioMode.KeepAspectRatio)
                    reader.setScaledSize(scaled_sz)
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


class _WLPGSectionHeader(QFrame):
    """Collapsible WLPG-style section header matching BatchPeopleView's Aero header strip."""

    recycle_section_requested = Signal(object)  # list[DuplicateGroup]

    def __init__(
        self,
        title: str,
        group_count: int = 0,
        reclaimable_text: str = "",
        is_dark: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.title = title
        self.group_count = group_count
        self.reclaimable_text = reclaimable_text
        self.is_dark = is_dark
        self.collapsed = False
        self.child_sections: list[QWidget] = []
        self.section_groups: list[DuplicateGroup] = []
        self.setFixedHeight(38)
        self.setCursor(Qt.PointingHandCursor)

        bg = "#1E232B" if self.is_dark else "#F4F7FB"
        bd = "#2D333F" if self.is_dark else "#D0DBE5"
        self.setObjectName("WLPGSectionHeaderFrame")
        self.setStyleSheet(
            f"QFrame#WLPGSectionHeaderFrame {{ background-color: {bg}; border: 1px solid {bd}; border-radius: 4px; }}"
        )

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 4, 8, 4)
        row.setSpacing(6)

        self.lbl_chevron = QLabel("▼", self)
        self.lbl_chevron.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.lbl_chevron.setStyleSheet(
            f"color: {'#93C5FD' if self.is_dark else '#1E395B'}; background: transparent; border: none;"
        )
        row.addWidget(self.lbl_chevron)

        self.lbl_title = QLabel(self.title, self)
        self.lbl_title.setFont(QFont("Segoe UI", 10.5, QFont.Weight.Bold))
        self.lbl_title.setStyleSheet(
            f"color: {'#60A5FA' if self.is_dark else '#005A9E'}; background: transparent; border: none;"
        )
        row.addWidget(self.lbl_title)

        self.lbl_sub = QLabel(self._format_sub(), self)
        self.lbl_sub.setFont(QFont("Segoe UI", 9))
        self.lbl_sub.setStyleSheet(
            f"color: {'#94A3B8' if self.is_dark else '#55687C'}; background: transparent; border: none;"
        )
        row.addWidget(self.lbl_sub, 1)

        self.btn_recycle_section = QPushButton("🗑 Recycle Section", self)
        self.btn_recycle_section.setFixedHeight(26)
        self.btn_recycle_section.setFont(QFont("Segoe UI", 8.5, QFont.Weight.Bold))
        self.btn_recycle_section.setCursor(Qt.PointingHandCursor)
        self.btn_recycle_section.setToolTip("Move all marked duplicate photos in this section to the Recycle Bin")
        self.btn_recycle_section.setStyleSheet(
            "QPushButton { background-color: #B91C1C; border: 1px solid #B91C1C; color: #FFFFFF; font-weight: bold; border-radius: 3px; padding: 2px 10px; }"
            "QPushButton:hover { background-color: #DC2626; border-color: #EF4444; }"
            "QPushButton:disabled { background-color: #3B2327; border-color: #5C2B32; color: #94A3B8; }"
            if self.is_dark
            else "QPushButton { background-color: #D32F2F; border: 1px solid #D32F2F; color: #FFFFFF; font-weight: bold; border-radius: 3px; padding: 2px 10px; }"
            "QPushButton:hover { background-color: #E53935; border-color: #C62828; }"
            "QPushButton:disabled { background-color: #E2E8F0; border-color: #CBD5E1; color: #64748B; }"
        )
        self.btn_recycle_section.clicked.connect(lambda: self.recycle_section_requested.emit(list(self.section_groups)))
        row.addWidget(self.btn_recycle_section)
        self.refresh_recycle_button()

    def _format_sub(self) -> str:
        g_word = "group" if self.group_count == 1 else "groups"
        if self.reclaimable_text:
            return f"({self.group_count} {g_word}, {self.reclaimable_text} recoverable)"
        return f"({self.group_count} {g_word})"

    def refresh_recycle_button(self) -> None:
        marked = sum(
            1 for g in self.section_groups for it in g.items if it.is_selected_for_removal
        )
        self.btn_recycle_section.setText(f"🗑 Recycle Section ({marked})" if marked > 0 else "🗑 Recycle Section")
        self.btn_recycle_section.setEnabled(marked > 0)

    def set_summary(self, group_count: int, reclaimable_text: str) -> None:
        self.group_count = group_count
        self.reclaimable_text = reclaimable_text
        self.lbl_sub.setText(self._format_sub())

    def toggle_collapsed(self) -> None:
        self.collapsed = not self.collapsed
        self.lbl_chevron.setText("▶" if self.collapsed else "▼")
        for w in self.child_sections:
            w.setVisible(not self.collapsed)

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton:
            self.toggle_collapsed()
            event.accept()
            return
        super().mousePressEvent(event)


class DuplicateFinderView(QWidget):
    """
    Integrated Duplicate & Burst Finder View (embedded directly in central_stack).
    Features side-by-side visual comparison, real-time streaming results,
    clear progress feedback, and safe Recycle Bin cleanup.
    """

    back_to_gallery_requested = Signal()
    duplicates_cleaned = Signal(int, int)  # count, freed_bytes
    scan_state_changed = Signal(bool)  # is_scanning
    summary_state_changed = Signal(int)  # marked_for_removal_count
    find_similar_requested = Signal(int)  # media_id
    compare_group_requested = Signal(object)  # list[int] media_ids
    zoom_changed = Signal(int)  # card_width
    media_selected = Signal(int)  # media_id

    _DEFAULT_BANNER_TEXT = (
        "ℹ Tip: The sharpest/highest-resolution copy in each group is automatically marked Keep. "
        "Click any group's Compare button for 2-Up / 3-Up / Quad light-table inspection."
    )

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
        self._applied_dark_mode: bool | None = None
        self._scope_signature: Any = None
        self.groups: list[DuplicateGroup] = []
        self.active_group: DuplicateGroup | None = None
        self._group_thumb_cache: dict[int, QPixmap] = {}
        self._scan_worker: _DuplicateScanWorker | None = None
        self._detection_engine: DuplicateDetectionEngine | None = None
        self._scan_generation: int = 0
        self._use_ribbon_mode: bool = False
        self._similarity_threshold: float = 0.96
        self._metadata_mode: str = str(
            QSettings("PhotoGallery64", "PhotoGallery64").value("ui/dup_metadata_mode", "ask") or "ask"
        )
        if self._metadata_mode not in ("ask", "always", "never"):
            self._metadata_mode = "ask"
        self._merge_metadata_on_clean: bool = (
            self._metadata_mode == "always"
            or (
                self._metadata_mode == "ask"
                and bool(QSettings("PhotoGallery64", "PhotoGallery64").value("ui/dup_merge_metadata", True, type=bool))
            )
        )
        self._banner_dismissed: bool = False
        self._seed_media_id: int | None = None
        self._seed_filename: str = ""
        self._selected_media_id: int | None = None
        self._feed_visible_groups: list[DuplicateGroup] = []
        self._feed_loaded_count: int = 0
        self._init_ui()
        self.set_dark_mode(self.is_dark)

    def ensure_workspace_visible(self) -> None:
        """Ensure the workspace stack shows results immediately upon re-entering Duplicate Finder."""
        if not hasattr(self, "workspace_stack"):
            return
        if self.groups:
            self.workspace_stack.setCurrentWidget(self.cards_scroll)
        elif hasattr(self, "empty_state_widget"):
            self.workspace_stack.setCurrentWidget(self.empty_state_widget)

    def set_media_records(
        self,
        records: Sequence[dict[str, Any]] | None,
        selected_records: Sequence[dict[str, Any]] | None = None,
        current_folder: str | None = None,
    ) -> None:
        """Updates available records for current filter scanning."""
        self.current_records = records
        self.selected_records = selected_records
        sig = (
            len(records) if records is not None else -1,
            len(selected_records) if selected_records is not None else 0,
            current_folder or "",
        )
        if sig != self._scope_signature:
            self._scope_signature = sig
            self.cmb_scope.clear()
            self.cmb_scope.addItem("All Photos in Catalog", "all")
            if self.selected_records and len(self.selected_records) > 1:
                self.cmb_scope.addItem(f"Selected Photos ({len(self.selected_records)} photos)", "selected")
            if current_folder:
                folder_name = Path(current_folder).name or current_folder
                self.cmb_scope.addItem(f"Folder: {folder_name}", {"type": "folder", "path": current_folder})
            if self.current_records:
                self.cmb_scope.addItem(f"Current Filter ({len(self.current_records)} photos)", "filtered")
            self._populate_scope_tree(current_folder=current_folder)
        if not self.groups:
            self._show_empty_state(post_scan=False)
        else:
            self.ensure_workspace_visible()

    def set_dark_mode(self, is_dark: bool) -> None:
        is_dark = bool(is_dark)
        if self._applied_dark_mode == is_dark:
            return
        self._applied_dark_mode = is_dark
        self.is_dark = is_dark
        if is_dark:
            self.setStyleSheet(
                """
                DuplicateFinderView { background-color: #14171C; color: #F1F5F9; font-family: 'Segoe UI'; }
                QLabel { color: #F1F5F9; }
                QSplitter::handle { background-color: #2D333F; width: 1px; }
                QWidget#DupScopePane { background-color: #1A1F26; border-right: 1px solid #2D333F; }
                QListWidget { background-color: #1A1F26; border: none; color: #F1F5F9; }
                QListWidget::item:selected { background-color: #253347; border: 1px solid #0078D4; }
                QTreeWidget { background-color: transparent; border: none; color: #F1F5F9; }
                QTreeWidget::item { padding: 2px 4px; }
                QTreeWidget::item:selected { background-color: #253347; color: #F1F5F9; border-radius: 3px; }
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
                DuplicateFinderView { background-color: #FFFFFF; color: #1E293B; font-family: 'Segoe UI'; }
                QLabel { color: #1E293B; }
                QSplitter::handle { background-color: #D6E2F0; width: 1px; }
                QWidget#DupScopePane { background-color: #F4F7FC; border-right: 1px solid #D6E2F0; }
                QListWidget { background-color: #FFFFFF; border: none; color: #1E293B; }
                QListWidget::item:selected { background-color: #E2EFFC; border: 1px solid #2A82DA; }
                QTreeWidget { background-color: transparent; border: none; color: #1E395B; }
                QTreeWidget::item { padding: 2px 4px; }
                QTreeWidget::item:selected { background-color: #D9E7F8; color: #1E395B; border-radius: 3px; }
                QScrollArea { background-color: #FFFFFF; border: none; }
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

        if hasattr(self, "info_banner"):
            if is_dark:
                self.info_banner.setStyleSheet(
                    "QFrame#DupInfoBanner { background-color: #2B2512; border: 1px solid #785E18; border-radius: 4px; }"
                )
                self.lbl_banner_text.setStyleSheet("color: #FDE68A; background: transparent; border: none;")
                self.btn_dismiss_banner.setStyleSheet(
                    "QPushButton { background: transparent; border: none; color: #FDE68A; font-weight: bold; }"
                    "QPushButton:hover { background-color: #423514; border-radius: 3px; }"
                )
            else:
                self.info_banner.setStyleSheet(
                    "QFrame#DupInfoBanner { background-color: #FFFBEB; border: 1px solid #FDE68A; border-radius: 4px; }"
                )
                self.lbl_banner_text.setStyleSheet("color: #665200; background: transparent; border: none;")
                self.btn_dismiss_banner.setStyleSheet(
                    "QPushButton { background: transparent; border: none; color: #665200; font-weight: bold; }"
                    "QPushButton:hover { background-color: #FEF3C7; border-radius: 3px; }"
                )

        pill_css = (
            "QPushButton { background-color: #1E2530; border: 1px solid #334155; color: #E2E8F0; "
            "border-radius: 13px; padding: 3px 12px; font-weight: 600; }"
            "QPushButton:hover { background-color: #283241; border-color: #60A5FA; }"
            "QPushButton:checked { background-color: #0078D4; border-color: #0078D4; color: #FFFFFF; }"
            if is_dark
            else "QPushButton { background-color: #FFFFFF; border: 1px solid #CBD5E1; color: #334155; "
            "border-radius: 13px; padding: 3px 12px; font-weight: 600; }"
            "QPushButton:hover { background-color: #F1F5F9; border-color: #0078D4; color: #005A9E; }"
            "QPushButton:checked { background-color: #0078D4; border-color: #0078D4; color: #FFFFFF; }"
        )
        for attr in ("btn_filter_all", "btn_filter_exact", "btn_filter_visual", "btn_filter_burst", "btn_filter_similar"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setStyleSheet(pill_css)

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 10, 14, 12)
        main_layout.setSpacing(8)

        # ---------------------------------------------------------------------
        # Top Header Bar (Back button, Title, Scan options)
        # ---------------------------------------------------------------------
        self.top_bar = QFrame(self)
        top_bar = self.top_bar
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
        self.chk_bursts.setChecked(False)
        self.chk_visual = QCheckBox("Visual matches (re-saved / resized)", top_bar)
        self.chk_visual.setChecked(True)
        row2.addWidget(self.chk_exact)
        row2.addWidget(self.chk_bursts)
        row2.addWidget(self.chk_visual)
        row2.addStretch(1)

        top_layout.addLayout(row2)
        main_layout.addWidget(top_bar)

        # Progress Strip (remains visible even when top_bar is hidden in Ribbon mode)
        progress_strip = QHBoxLayout()
        progress_strip.setContentsMargins(0, 0, 0, 0)
        progress_strip.setSpacing(8)
        self.scan_progress_bar = QProgressBar(self)
        self.scan_progress_bar.setFixedHeight(16)
        self.scan_progress_bar.setRange(0, 100)
        self.scan_progress_bar.setValue(0)
        self.scan_progress_bar.setTextVisible(True)
        self.scan_progress_bar.setFixedWidth(240)
        self.scan_progress_bar.setVisible(False)
        progress_strip.addWidget(self.scan_progress_bar)

        self.lbl_scan_status = QLabel(self)
        self.lbl_scan_status.setFont(QFont("Segoe UI", 8.5))
        self.lbl_scan_status.setVisible(False)
        progress_strip.addWidget(self.lbl_scan_status, 1)
        main_layout.addLayout(progress_strip)

        # WLPG Soft-Yellow Workflow / Context Banner (dismissible)
        self.info_banner = QFrame(self)
        self.info_banner.setObjectName("DupInfoBanner")
        banner_row = QHBoxLayout(self.info_banner)
        banner_row.setContentsMargins(10, 6, 8, 6)
        banner_row.setSpacing(8)
        self.lbl_banner_text = QLabel(self._DEFAULT_BANNER_TEXT, self.info_banner)
        self.lbl_banner_text.setFont(QFont("Segoe UI", 8.5))
        self.lbl_banner_text.setWordWrap(True)
        banner_row.addWidget(self.lbl_banner_text, 1)

        self.btn_dismiss_banner = QPushButton("✕", self.info_banner)
        self.btn_dismiss_banner.setFixedSize(20, 20)
        self.btn_dismiss_banner.setCursor(Qt.PointingHandCursor)
        self.btn_dismiss_banner.setToolTip("Dismiss tip")
        self.btn_dismiss_banner.clicked.connect(self._dismiss_info_banner)
        banner_row.addWidget(self.btn_dismiss_banner)
        main_layout.addWidget(self.info_banner)

        # ---------------------------------------------------------------------
        # Central Splitter (Left: Persistent Scan Scope + Loupe List, Right: Workspace)
        # ---------------------------------------------------------------------
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setHandleWidth(1)

        # --- Left Pane (Persistent Scan Scope + 2-Up Loupe Group Navigator) ---
        self.left_widget = QWidget(splitter)
        self.left_widget.setObjectName("DupScopePane")
        left_widget = self.left_widget
        left_widget.setFixedWidth(245)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(6)

        lbl_scope_hdr = QLabel("Scan Scope", left_widget)
        lbl_scope_hdr.setFont(QFont("Segoe UI", 9.5, QFont.Bold))
        left_layout.addWidget(lbl_scope_hdr)

        self.txt_scope_filter = QLineEdit(left_widget)
        self.txt_scope_filter.setPlaceholderText("Filter folders & albums...")
        self.txt_scope_filter.setClearButtonEnabled(True)
        self.txt_scope_filter.textChanged.connect(self._filter_scope_tree)
        left_layout.addWidget(self.txt_scope_filter)

        self.scope_tree = QTreeWidget(left_widget)
        self.scope_tree.setHeaderHidden(True)
        self.scope_tree.setIndentation(14)
        self.scope_tree.currentItemChanged.connect(self._on_scope_tree_item_selected)
        self.scope_tree.itemDoubleClicked.connect(self._on_scope_tree_item_double_clicked)
        left_layout.addWidget(self.scope_tree, 1)

        splitter.addWidget(left_widget)

        # --- Right Pane (Comparison Workspace) ---
        right_widget = QWidget(splitter)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(8)

        # Filter Pills + Sort / Group Strip
        pill_row = QHBoxLayout()
        pill_row.setContentsMargins(0, 0, 0, 0)
        pill_row.setSpacing(6)
        lbl_flt = QLabel("Filter:", right_widget)
        lbl_flt.setFont(QFont("Segoe UI", 8.5, QFont.Weight.DemiBold))
        lbl_flt.setStyleSheet("color: #64748B;")
        pill_row.addWidget(lbl_flt)
        self.btn_filter_all = QPushButton("All (0)", right_widget)
        self.btn_filter_all.setToolTip("Show all detected groups (Exact Copies, Visually Similar Pairs, and Burst Sequences)")
        self.btn_filter_exact = QPushButton("Exact (0)", right_widget)
        self.btn_filter_exact.setToolTip("Exact Copies (100%) — bit-identical files with matching file size and SHA-256 hash")
        self.btn_filter_visual = QPushButton("Visually Similar (0)", right_widget)
        self.btn_filter_visual.setToolTip("Visually Similar Pairs — resized, re-compressed, or slightly cropped photos that look alike")
        self.btn_filter_burst = QPushButton("Bursts (0)", right_widget)
        self.btn_filter_burst.setToolTip("Burst Sequences — rapid-fire shots captured within 3 seconds of each other")
        self.btn_filter_similar = QPushButton("Similar to Photo (0)", right_widget)
        self.btn_filter_similar.setToolTip("Photos visually similar to your chosen reference photo")
        self.btn_filter_similar.setVisible(False)
        for b in (
            self.btn_filter_all,
            self.btn_filter_exact,
            self.btn_filter_visual,
            self.btn_filter_burst,
            self.btn_filter_similar,
        ):
            b.setCheckable(True)
            b.setMinimumWidth(82)
            b.setFixedHeight(26)
            b.setFont(QFont("Segoe UI", 8.5, QFont.Weight.DemiBold))
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(self._on_type_filter_changed)
            pill_row.addWidget(b)
        self.btn_filter_all.setChecked(True)
        pill_row.addStretch(1)

        self.txt_group_filter = QLineEdit(right_widget)
        self.txt_group_filter.setPlaceholderText("Filter by filename or folder...")
        self.txt_group_filter.setClearButtonEnabled(True)
        self.txt_group_filter.setFixedHeight(26)
        self.txt_group_filter.setFixedWidth(220)
        self.txt_group_filter.setToolTip("Show groups where at least one photo's filename or folder contains this text")
        self.txt_group_filter.textChanged.connect(lambda _t: self._populate_groups_list())
        pill_row.addWidget(self.txt_group_filter)

        self.lbl_sort_groups = QLabel("Sort:", right_widget)
        self.lbl_sort_groups.setFont(QFont("Segoe UI", 8.5))
        pill_row.addWidget(self.lbl_sort_groups)
        self.cmb_sort_groups = QComboBox(right_widget)
        for s_lbl, s_key in (
            ("Recoverable Space", "reclaimable"),
            ("Similarity %", "similarity"),
            ("Photo Count", "count"),
            ("Newest Date", "date_desc"),
            ("Oldest Date", "date_asc"),
            ("Folder (A–Z)", "folder"),
        ):
            self.cmb_sort_groups.addItem(s_lbl, s_key)
        self.cmb_sort_groups.setFixedHeight(26)
        self.cmb_sort_groups.currentIndexChanged.connect(lambda _idx: self._populate_groups_list())
        pill_row.addWidget(self.cmb_sort_groups)

        self.lbl_group_by = QLabel("Group:", right_widget)
        self.lbl_group_by.setFont(QFont("Segoe UI", 8.5))
        pill_row.addWidget(self.lbl_group_by)
        self.cmb_group_by = QComboBox(right_widget)
        for g_lbl, g_key in (
            ("None", "none"),
            ("Match Type", "match_type"),
            ("Folder", "folder"),
            ("Date (Month)", "month"),
        ):
            self.cmb_group_by.addItem(g_lbl, g_key)
        self.cmb_group_by.setFixedHeight(26)
        self.cmb_group_by.currentIndexChanged.connect(lambda _idx: self._populate_groups_list())
        pill_row.addWidget(self.cmb_group_by)

        right_layout.addLayout(pill_row)

        # Workspace Header Strip
        ws_header = QHBoxLayout()
        self.lbl_group_title = QLabel("Select a duplicate group to compare", right_widget)
        self.lbl_group_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        ws_header.addWidget(self.lbl_group_title, 1)

        # Controls widget (hidden when Ribbon mode is active)
        self.ws_controls_widget = QWidget(right_widget)
        ws_ctrl_layout = QHBoxLayout(self.ws_controls_widget)
        ws_ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ws_ctrl_layout.setSpacing(6)

        lbl_rule = QLabel("Auto-Select Rule:", self.ws_controls_widget)
        lbl_rule.setFont(QFont("Segoe UI", 9))
        ws_ctrl_layout.addWidget(lbl_rule)

        self.cmb_rule = QComboBox(self.ws_controls_widget)
        self.cmb_rule.addItems([
            "Keep Highest Resolution",
            "Keep Largest File Size",
            "Keep Oldest Date Taken",
            "Keep Most Metadata",
            "Keep Best Sharpness",
            "Keep Original Filename",
            "Keep Preferred Folder Priority",
        ])
        self.cmb_rule.setFixedHeight(26)
        self.cmb_rule.currentIndexChanged.connect(self._on_rule_changed)
        ws_ctrl_layout.addWidget(self.cmb_rule)

        self.txt_preferred_folder = QLineEdit(self.ws_controls_widget)
        self.txt_preferred_folder.setPlaceholderText("Preferred folder(s), ; separated...")
        self.txt_preferred_folder.setFixedWidth(180)
        self.txt_preferred_folder.setFixedHeight(26)
        self.txt_preferred_folder.setVisible(False)
        ws_ctrl_layout.addWidget(self.txt_preferred_folder)

        self.btn_browse_pref_folder = QPushButton("...", self.ws_controls_widget)
        self.btn_browse_pref_folder.setFixedSize(26, 26)
        self.btn_browse_pref_folder.setToolTip("Choose preferred keep folder")
        self.btn_browse_pref_folder.setVisible(False)
        self.btn_browse_pref_folder.clicked.connect(self._browse_preferred_folder)
        ws_ctrl_layout.addWidget(self.btn_browse_pref_folder)

        btn_apply_all = QPushButton("Apply to All Groups", self.ws_controls_widget)
        btn_apply_all.setFixedHeight(26)
        btn_apply_all.setCursor(Qt.PointingHandCursor)
        btn_apply_all.clicked.connect(self._apply_rule_to_all_groups)
        ws_ctrl_layout.addWidget(btn_apply_all)

        ws_ctrl_layout.addSpacing(10)

        lbl_zoom = QLabel("Zoom:", self.ws_controls_widget)
        lbl_zoom.setFont(QFont("Segoe UI", 9))
        ws_ctrl_layout.addWidget(lbl_zoom)

        self.btn_zoom_out = QPushButton("−", self.ws_controls_widget)
        self.btn_zoom_out.setFixedSize(26, 26)
        self.btn_zoom_out.setToolTip("Zoom out preview images (Ctrl -)")
        self.btn_zoom_out.setCursor(Qt.PointingHandCursor)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        ws_ctrl_layout.addWidget(self.btn_zoom_out)

        self.slider_zoom = QSlider(Qt.Horizontal, self.ws_controls_widget)
        self.slider_zoom.setCursor(Qt.SizeHorCursor)
        self.slider_zoom.setRange(220, 650)
        self.slider_zoom.setValue(310)
        self.slider_zoom.setFixedWidth(90)
        self.slider_zoom.setToolTip("Drag to zoom comparison previews")
        self.slider_zoom.valueChanged.connect(self._on_zoom_slider_changed)
        ws_ctrl_layout.addWidget(self.slider_zoom)

        self.btn_zoom_in = QPushButton("+", self.ws_controls_widget)
        self.btn_zoom_in.setFixedSize(26, 26)
        self.btn_zoom_in.setToolTip("Zoom in preview images (Ctrl +)")
        self.btn_zoom_in.setCursor(Qt.PointingHandCursor)
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        ws_ctrl_layout.addWidget(self.btn_zoom_in)

        self.lbl_zoom_val = QLabel("100%", self.ws_controls_widget)
        self.lbl_zoom_val.setFont(QFont("Segoe UI", 8.5))
        self.lbl_zoom_val.setFixedWidth(38)
        ws_ctrl_layout.addWidget(self.lbl_zoom_val)

        ws_header.addWidget(self.ws_controls_widget)
        right_layout.addLayout(ws_header)

        # Workspace Stack: Card Grid vs 2-Up Synchronized Loupe
        self.workspace_stack = QStackedWidget(right_widget)

        # Scroll Area for All Groups Vertical Feed
        self.cards_scroll = QScrollArea(self.workspace_stack)
        self.cards_scroll.setWidgetResizable(True)
        self.cards_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.cards_scroll.viewport().installEventFilter(self)
        self.cards_scroll.verticalScrollBar().valueChanged.connect(self._on_feed_scrolled)
        self.cards_container = QWidget()
        self.cards_container.setStyleSheet("background: transparent;")
        self.cards_container.setContextMenuPolicy(Qt.CustomContextMenu)
        self.cards_container.customContextMenuRequested.connect(self._on_workspace_context_menu)
        self.cards_container.installEventFilter(self)
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(8, 8, 8, 8)
        self.cards_layout.setSpacing(14)
        self.cards_layout.setAlignment(Qt.AlignTop)
        self.cards_scroll.setWidget(self.cards_container)
        self._group_sections: dict[str, QFrame] = {}
        self.workspace_stack.addWidget(self.cards_scroll)

        # WLPG-style Empty State ("Scan Now" clickable blue link)
        self.empty_state_widget = QWidget(self.workspace_stack)
        es_lay = QVBoxLayout(self.empty_state_widget)
        es_lay.setContentsMargins(20, 20, 20, 20)
        es_lay.setSpacing(8)
        es_lay.setAlignment(Qt.AlignCenter)
        self.btn_empty_scan_action = QPushButton("Scan Now for duplicate, similar, and burst photos", self.empty_state_widget)
        self.btn_empty_scan_action.setCursor(Qt.PointingHandCursor)
        self.btn_empty_scan_action.setStyleSheet(
            """
            QPushButton {
                color: #0066CC;
                font-family: 'Segoe UI', sans-serif;
                font-size: 11pt;
                border: none;
                background: transparent;
                padding: 0;
            }
            QPushButton:hover {
                color: #004499;
                text-decoration: underline;
            }
            """
        )
        self.btn_empty_scan_action.clicked.connect(self._on_empty_state_action_clicked)
        es_lay.addWidget(self.btn_empty_scan_action, 0, Qt.AlignCenter)

        self.lbl_empty_subtitle = QLabel(
            "Choose a folder or album in the Scan Scope pane on the left, or click above to scan.",
            self.empty_state_widget,
        )
        self.lbl_empty_subtitle.setAlignment(Qt.AlignCenter)
        self.lbl_empty_subtitle.setStyleSheet(
            "color: #64748B; font-family: 'Segoe UI', sans-serif; font-size: 9.5pt; border: none; background: transparent;"
        )
        es_lay.addWidget(self.lbl_empty_subtitle, 0, Qt.AlignCenter)
        self._empty_action_mode = "scan_now"
        self.workspace_stack.addWidget(self.empty_state_widget)
        self.workspace_stack.setCurrentWidget(self.empty_state_widget)
        self._populate_scope_tree()

        right_layout.addWidget(self.workspace_stack, 1)

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

        self.footer_actions_widget = QWidget(self)
        fa_layout = QHBoxLayout(self.footer_actions_widget)
        fa_layout.setContentsMargins(0, 0, 0, 0)
        fa_layout.setSpacing(8)

        btn_backup = QPushButton("Move Duplicates...", self.footer_actions_widget)
        btn_backup.setFixedHeight(30)
        btn_backup.setCursor(Qt.PointingHandCursor)
        btn_backup.setToolTip("Move all duplicate photos (marked REMOVE) to a folder outside your library")
        btn_backup.clicked.connect(self._move_marked_to_backup)
        fa_layout.addWidget(btn_backup)

        self.btn_delete = QPushButton("Delete All Duplicates", self.footer_actions_widget)
        self.btn_delete.setObjectName("DeleteBtn")
        self.btn_delete.setFixedHeight(30)
        self.btn_delete.setCursor(Qt.PointingHandCursor)
        self.btn_delete.setToolTip("Move all duplicate photos (marked REMOVE) to the Windows Recycle Bin")
        self.btn_delete.clicked.connect(self._move_marked_to_recycle_bin)
        fa_layout.addWidget(self.btn_delete)

        footer.addWidget(self.footer_actions_widget)
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

    def open_scope_menu(self, anchor_widget: QWidget | None = None) -> None:
        """Shows the left Scan Scope pane and focuses its folder/album filter input."""
        if hasattr(self, "left_widget"):
            self.left_widget.setVisible(True)
        if hasattr(self, "txt_scope_filter"):
            self.txt_scope_filter.setFocus()
            self.txt_scope_filter.selectAll()

    def _populate_scope_tree(self, current_folder: str | None = None) -> None:
        if not hasattr(self, "scope_tree"):
            return
        self.scope_tree.blockSignals(True)
        self.scope_tree.clear()

        try:
            from src.ui.ribbon_toolbar import make_wlpg_icon
            icn_all = make_wlpg_icon("all_media", 16)
            icn_view = make_wlpg_icon("pictures_library", 16)
            icn_sel = make_wlpg_icon("select_all", 16)
            icn_folder = make_wlpg_icon("folder", 16)
            icn_albums = make_wlpg_icon("pictures_library", 16)
            icn_smart = make_wlpg_icon("star_filled", 16)
        except Exception:
            icn_all = icn_view = icn_sel = icn_folder = icn_albums = icn_smart = QIcon()

        item_all = QTreeWidgetItem(self.scope_tree, ["All Photos in Catalog"])
        item_all.setIcon(0, icn_all)
        item_all.setData(0, Qt.UserRole, {"type": "all"})

        if self.current_records:
            item_flt = QTreeWidgetItem(self.scope_tree, [f"Current Gallery View ({len(self.current_records)})"])
            item_flt.setIcon(0, icn_view)
            item_flt.setData(0, Qt.UserRole, {"type": "filtered"})

        if getattr(self, "selected_records", None) and len(self.selected_records) > 1:
            item_sel = QTreeWidgetItem(self.scope_tree, [f"Selected Photos ({len(self.selected_records)})"])
            item_sel.setIcon(0, icn_sel)
            item_sel.setData(0, Qt.UserRole, {"type": "selected"})

        folders_root = QTreeWidgetItem(self.scope_tree, ["Library Folders"])
        folders_root.setIcon(0, icn_folder)
        folders_root.setFlags(folders_root.flags() & ~Qt.ItemIsSelectable)
        folders: list[dict[str, Any]] = []
        if self.catalog_repo and hasattr(self.catalog_repo, "get_folders"):
            try:
                folders = self.catalog_repo.get_folders()
            except Exception:
                folders = []

        folder_nodes: dict[int, QTreeWidgetItem] = {}
        active_node: QTreeWidgetItem | None = None
        for f in folders:
            fid = int(f["id"])
            fpath = str(f.get("path") or "")
            fname = str(f.get("name") or Path(fpath).name or fpath)
            node = QTreeWidgetItem([fname])
            node.setIcon(0, icn_folder)
            node.setToolTip(0, fpath)
            node.setData(0, Qt.UserRole, {"type": "folder", "path": fpath, "name": fname})
            folder_nodes[fid] = node
            if current_folder and fpath == current_folder:
                active_node = node

        for f in folders:
            fid = int(f["id"])
            pid = f.get("parent_id")
            parent_node = folder_nodes.get(int(pid)) if pid is not None else None
            if parent_node is not None and parent_node is not folder_nodes[fid]:
                parent_node.addChild(folder_nodes[fid])
            else:
                folders_root.addChild(folder_nodes[fid])

        albums_root = QTreeWidgetItem(self.scope_tree, ["Albums"])
        albums_root.setIcon(0, icn_albums)
        albums_root.setFlags(albums_root.flags() & ~Qt.ItemIsSelectable)
        albums: list[dict[str, Any]] = []
        if self.catalog_repo and hasattr(self.catalog_repo, "get_albums_with_counts"):
            try:
                albums = self.catalog_repo.get_albums_with_counts()
            except Exception:
                albums = []
        for alb in albums:
            aid = int(alb["id"])
            aname = str(alb.get("name") or f"Album {aid}")
            cnt = int(alb.get("media_count") or alb.get("count") or 0)
            node = QTreeWidgetItem(albums_root, [f"{aname} ({cnt})"])
            node.setIcon(0, icn_folder)
            node.setData(0, Qt.UserRole, {"type": "album", "album_id": aid, "name": aname})

        smart_root = QTreeWidgetItem(self.scope_tree, ["Smart Albums"])
        smart_root.setIcon(0, icn_smart)
        smart_root.setFlags(smart_root.flags() & ~Qt.ItemIsSelectable)
        smart_albums: list[dict[str, Any]] = []
        if self.catalog_repo and hasattr(self.catalog_repo, "get_smart_albums_with_counts"):
            try:
                smart_albums = self.catalog_repo.get_smart_albums_with_counts()
            except Exception:
                smart_albums = []
        for sa in smart_albums:
            sid = int(sa["id"])
            sname = str(sa.get("name") or f"Smart Album {sid}")
            cnt = int(sa.get("media_count") or sa.get("count") or 0)
            node = QTreeWidgetItem(smart_root, [f"{sname} ({cnt})"])
            node.setIcon(0, icn_smart)
            node.setData(0, Qt.UserRole, {"type": "smart_album", "smart_album_id": sid, "name": sname})

        self.scope_tree.expandAll()
        self.scope_tree.setCurrentItem(active_node or item_all)
        self.scope_tree.blockSignals(False)

    def _filter_scope_tree(self, text: str) -> None:
        if not hasattr(self, "scope_tree"):
            return
        q = text.strip().lower()
        def _walk(item: QTreeWidgetItem) -> bool:
            child_match = False
            for i in range(item.childCount()):
                if _walk(item.child(i)):
                    child_match = True
            self_match = (not q) or (q in item.text(0).lower())
            visible = self_match or child_match
            item.setHidden(not visible)
            if q and child_match:
                item.setExpanded(True)
            return visible
        for i in range(self.scope_tree.topLevelItemCount()):
            _walk(self.scope_tree.topLevelItem(i))

    def _sync_scope_from_data(self, data: dict[str, Any]) -> None:
        t = data.get("type")
        if t in ("all", "filtered", "selected"):
            for i in range(self.cmb_scope.count()):
                if self.cmb_scope.itemData(i) == t:
                    self.cmb_scope.setCurrentIndex(i)
                    break
        elif t == "folder":
            folder_path = data["path"]
            folder_name = data["name"]
            idx = -1
            for i in range(self.cmb_scope.count()):
                d = self.cmb_scope.itemData(i)
                if isinstance(d, dict) and d.get("type") == "folder" and d.get("path") == folder_path:
                    idx = i
                    break
            if idx < 0:
                self.cmb_scope.addItem(f"📁 {folder_name}", {"type": "folder", "path": folder_path})
                idx = self.cmb_scope.count() - 1
            self.cmb_scope.setCurrentIndex(idx)
        elif t == "album":
            album_id = int(data["album_id"])
            album_name = data["name"]
            idx = -1
            for i in range(self.cmb_scope.count()):
                d = self.cmb_scope.itemData(i)
                if isinstance(d, dict) and d.get("type") == "album" and int(d.get("album_id", -1)) == album_id:
                    idx = i
                    break
            if idx < 0:
                self.cmb_scope.addItem(f"Album: {album_name}", {"type": "album", "album_id": album_id})
                idx = self.cmb_scope.count() - 1
            self.cmb_scope.setCurrentIndex(idx)
        elif t == "smart_album":
            sa_id = int(data["smart_album_id"])
            sa_name = data["name"]
            idx = -1
            for i in range(self.cmb_scope.count()):
                d = self.cmb_scope.itemData(i)
                if isinstance(d, dict) and d.get("type") == "smart_album" and int(d.get("smart_album_id", -1)) == sa_id:
                    idx = i
                    break
            if idx < 0:
                self.cmb_scope.addItem(f"Smart Album: {sa_name}", {"type": "smart_album", "smart_album_id": sa_id})
                idx = self.cmb_scope.count() - 1
            self.cmb_scope.setCurrentIndex(idx)

    def _on_scope_tree_item_selected(self, current: QTreeWidgetItem | None, _prev: QTreeWidgetItem | None = None) -> None:
        if current is None:
            return
        data = current.data(0, Qt.UserRole)
        if isinstance(data, dict):
            self._sync_scope_from_data(data)

    def _on_scope_tree_item_double_clicked(self, item: QTreeWidgetItem | None, _col: int = 0) -> None:
        if item is None:
            return
        data = item.data(0, Qt.UserRole)
        if isinstance(data, dict):
            self._sync_scope_from_data(data)
            self.start_scan()

    def _show_empty_state(self, *, post_scan: bool = False) -> None:
        if not hasattr(self, "empty_state_widget"):
            return
        if not post_scan:
            self._empty_action_mode = "scan_now"
            self.btn_empty_scan_action.setText("Scan Now for duplicate, similar, and burst photos")
            self.lbl_empty_subtitle.setText(
                "Choose a folder or album in the Scan Scope pane on the left, or click above to scan."
            )
        else:
            if self.cmb_scope.currentData() != "all":
                self._empty_action_mode = "scan_all"
                self.btn_empty_scan_action.setText("Try scanning all photos in catalog")
                self.lbl_empty_subtitle.setText("No duplicate or burst photos were found in the selected scope.")
            else:
                self._empty_action_mode = "scan_now"
                self.btn_empty_scan_action.setText("Scan again")
                self.lbl_empty_subtitle.setText("✓ No duplicate or burst photos were found in your catalog.")
        if self.cards_layout.count() == 0:
            self.workspace_stack.setCurrentWidget(self.empty_state_widget)

    def _on_empty_state_action_clicked(self) -> None:
        if getattr(self, "_empty_action_mode", "scan_now") == "scan_all":
            self._select_scope_and_scan("all")
        else:
            self.start_scan()

    def _select_scope_and_scan(self, scope_data: str) -> None:
        for i in range(self.cmb_scope.count()):
            if self.cmb_scope.itemData(i) == scope_data:
                self.cmb_scope.setCurrentIndex(i)
                break
        self.start_scan()

    def _select_folder_scope_and_scan(self, folder_path: str, folder_name: str) -> None:
        idx = -1
        for i in range(self.cmb_scope.count()):
            d = self.cmb_scope.itemData(i)
            if isinstance(d, dict) and d.get("type") == "folder" and d.get("path") == folder_path:
                idx = i
                break
        if idx < 0:
            self.cmb_scope.addItem(f"📁 {folder_name}", {"type": "folder", "path": folder_path})
            idx = self.cmb_scope.count() - 1
        self.cmb_scope.setCurrentIndex(idx)
        self.start_scan()

    def _select_album_scope_and_scan(self, album_id: int, album_name: str) -> None:
        idx = -1
        for i in range(self.cmb_scope.count()):
            d = self.cmb_scope.itemData(i)
            if isinstance(d, dict) and d.get("type") == "album" and int(d.get("album_id", -1)) == int(album_id):
                idx = i
                break
        if idx < 0:
            self.cmb_scope.addItem(f"🗂️ {album_name}", {"type": "album", "album_id": int(album_id)})
            idx = self.cmb_scope.count() - 1
        self.cmb_scope.setCurrentIndex(idx)
        self.start_scan()

    def _fetch_album_records(self, album_id: int) -> list[dict[str, Any]]:
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
        JOIN album_items ai ON ai.media_id = m.id
        WHERE ai.album_id = ?
        """
        rows = cur.execute(sql, (int(album_id),)).fetchall()
        return [
            {
                "id": r[0],
                "folder_path": r[1],
                "filename": r[2],
                "file_size": r[3],
                "width": r[4],
                "height": r[5],
                "mtime": r[6],
                "date_taken": r[7],
                "rating": r[8],
                "flag": r[9],
                "caption": r[10],
                "is_video": r[11],
            }
            for r in rows
        ]

    def _on_choose_folder_clicked(self) -> None:
        """Opens the Library Scope & Folder menu anchored to the Scope button."""
        self.open_scope_menu(getattr(self, "btn_browse_folder", None))

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

    def _fetch_smart_album_records(self, smart_album_id: int) -> list[dict[str, Any]]:
        if not self.catalog_repo:
            return []
        sa = self.catalog_repo.get_smart_album(int(smart_album_id)) if hasattr(self.catalog_repo, "get_smart_album") else None
        if not sa:
            return []
        from src.db.repository import compile_smart_album_where
        where_sql, params = compile_smart_album_where(sa.get("rules_dict") or {}, alias="m")
        priv_clause = self.catalog_repo._private_folder_clause("f") if hasattr(self.catalog_repo, "_private_folder_clause") else "1=1"
        conn = self.catalog_repo.engine.catalog_conn()
        sql = f"""
        SELECT m.id, f.path AS folder_path, m.filename, m.size, m.width, m.height, m.mtime, m.date_taken,
               m.rating, m.flag, m.caption, m.is_video
        FROM media m
        JOIN folders f ON m.folder_id = f.id
        WHERE COALESCE(m.is_missing, 0) = 0 AND {priv_clause} AND ({where_sql});
        """
        rows = conn.execute(sql, params).fetchall()
        return [
            {
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
            }
            for r in rows
        ]

    def _dismiss_info_banner(self) -> None:
        self._banner_dismissed = True
        if hasattr(self, "info_banner"):
            self.info_banner.setVisible(False)

    def _update_workflow_banner(self, seed_media_id: int | None) -> None:
        if not hasattr(self, "info_banner"):
            return
        if seed_media_id is not None:
            seed_name = self._seed_filename or f"Photo #{seed_media_id}"
            self.lbl_banner_text.setText(
                f"ℹ Showing photos visually similar to \"{seed_name}\" — adjust Similarity % in the Ribbon "
                "or click Scan Now to return to duplicate scanning."
            )
            self.info_banner.setVisible(True)
        else:
            self.lbl_banner_text.setText(self._DEFAULT_BANNER_TEXT)
            self.info_banner.setVisible(not self._banner_dismissed)

    def find_similar_to(self, seed_media_id: int, similarity_threshold: float = 0.78) -> None:
        """Scan the catalog for photos visually similar to seed_media_id using LAB Haar + pHash."""
        for b in (
            self.btn_filter_all,
            self.btn_filter_exact,
            self.btn_filter_visual,
            self.btn_filter_burst,
            self.btn_filter_similar,
        ):
            b.setChecked(b is self.btn_filter_all)
        self._similarity_threshold = max(0.55, min(0.99, float(similarity_threshold)))
        self.start_scan(seed_media_id=seed_media_id)

    def start_scan(self, seed_media_id: int | None = None) -> None:
        """Starts duplicate detection (or Find Similar when seed_media_id is set)."""
        self.stop_scan()
        self._seed_media_id = seed_media_id
        self._seed_filename = ""
        if seed_media_id is not None and self.catalog_repo is not None:
            try:
                recs = self.catalog_repo.get_media_by_ids([int(seed_media_id)])
                if recs:
                    self._seed_filename = str(getattr(recs[0], "filename", "") or "")
            except Exception:
                self._seed_filename = ""
        self._update_workflow_banner(seed_media_id)
        if seed_media_id is None and self.window() is not None:
            wtitle = self.window().windowTitle()
            if wtitle.startswith("Find Similar Photos"):
                self.window().setWindowTitle("Find Duplicates & Bursts - Essentials Photo Gallery")
        self._scan_generation += 1
        scan_id = self._scan_generation
        self.groups.clear()
        self._clear_workspace()
        self._update_type_filter_counts()
        self._update_summary_counts()

        scope = self.cmb_scope.currentData()
        if seed_media_id is not None:
            records = None
        elif scope == "filtered":
            records = self.current_records
        elif scope == "selected":
            records = getattr(self, "selected_records", None)
        elif isinstance(scope, dict) and scope.get("type") == "folder":
            records = self._fetch_folder_records(scope.get("path", ""))
        elif isinstance(scope, dict) and scope.get("type") == "album":
            records = self._fetch_album_records(int(scope.get("album_id", 0)))
        elif isinstance(scope, dict) and scope.get("type") == "smart_album":
            records = self._fetch_smart_album_records(int(scope.get("smart_album_id", 0)))
        else:
            records = None

        self._detection_engine = DuplicateDetectionEngine(self.catalog_repo, self.thumb_repo)
        self._scan_worker = _DuplicateScanWorker(
            engine=self._detection_engine,
            media_records=records,
            find_exact=self.chk_exact.isChecked(),
            find_visual=self.chk_visual.isChecked(),
            find_bursts=self.chk_bursts.isChecked(),
            similarity_threshold=self._similarity_threshold,
            seed_media_id=seed_media_id,
            parent=self,
        )

        self.btn_scan.setEnabled(False)
        self.btn_stop.setVisible(True)
        self.scan_progress_bar.setVisible(True)
        self.scan_progress_bar.setValue(0)
        self.lbl_scan_status.setVisible(True)
        self.lbl_scan_status.setText(
            "Searching catalog for visually similar photos..." if seed_media_id is not None else "Scanning catalog for duplicates..."
        )
        self.scan_state_changed.emit(True)

        self._scan_worker.progress.connect(lambda c, t, m, sid=scan_id: self._on_scan_progress(c, t, m, sid))
        self._scan_worker.group_found.connect(lambda g, sid=scan_id: self._on_group_found(g, sid))
        self._scan_worker.finished_scan.connect(lambda grps, sid=scan_id: self._on_scan_finished(grps, sid))
        self._scan_worker.start()

    def stop_scan(self) -> None:
        """Safely and gracefully cancels the scan worker thread without crashing."""
        self._scan_generation += 1
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
        self.scan_state_changed.emit(False)

    @Slot(int, int, str)
    def _on_scan_progress(self, cur: int, tot: int, msg: str, scan_id: int | None = None) -> None:
        if scan_id is not None and scan_id != self._scan_generation:
            return
        self.scan_progress_bar.setRange(0, tot)
        self.scan_progress_bar.setValue(cur)
        self.lbl_scan_status.setText(msg)

    @Slot(object)
    def _on_group_found(self, group: DuplicateGroup, scan_id: int | None = None) -> None:
        """Called live as each duplicate group is identified."""
        if scan_id is not None and scan_id != self._scan_generation:
            return
        self.groups.append(group)
        self._add_group_to_list(group)
        self._update_type_filter_counts()
        self._update_summary_counts()

        if self.active_group is None:
            self.active_group = group

    @Slot(list)
    def _on_scan_finished(self, groups: list[DuplicateGroup], scan_id: int | None = None) -> None:
        """Called when scan completes fully."""
        if scan_id is not None and scan_id != self._scan_generation:
            return
        self.btn_scan.setEnabled(True)
        self.btn_stop.setVisible(False)
        self.scan_progress_bar.setVisible(False)
        self.lbl_scan_status.setVisible(False)
        self.scan_state_changed.emit(False)

        self.groups = groups
        self._populate_groups_list()
        self._update_summary_counts()

        if not self.groups:
            self.lbl_group_title.setText("✓ No duplicates found in selected scope.")
            self._show_empty_state(post_scan=True)

    def _on_type_filter_changed(self) -> None:
        sender = self.sender()
        for b in (
            self.btn_filter_all,
            self.btn_filter_exact,
            self.btn_filter_visual,
            self.btn_filter_burst,
            self.btn_filter_similar,
        ):
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
        cnt_similar = sum(1 for g in self.groups if g.match_type == "similar")

        self.btn_filter_all.setText(f"All ({cnt_all})")
        self.btn_filter_exact.setText(f"Exact ({cnt_exact})")
        self.btn_filter_visual.setText(f"Visually Similar ({cnt_visual})")
        self.btn_filter_burst.setText(f"Bursts ({cnt_burst})")
        if hasattr(self, "btn_filter_similar"):
            self.btn_filter_similar.setText(f"Similar to Photo ({cnt_similar})")
            self.btn_filter_similar.setVisible(cnt_similar > 0)
            if cnt_similar == 0 and self.btn_filter_similar.isChecked():
                self.btn_filter_similar.setChecked(False)
                self.btn_filter_all.setChecked(True)

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

    def _active_filter_mode(self) -> str:
        if self.btn_filter_exact.isChecked():
            return "exact"
        if self.btn_filter_visual.isChecked():
            return "visual"
        if self.btn_filter_burst.isChecked():
            return "burst"
        if hasattr(self, "btn_filter_similar") and self.btn_filter_similar.isChecked():
            return "similar"
        return "all"

    def _add_group_to_list(self, g: DuplicateGroup) -> None:
        filter_mode = self._active_filter_mode()
        if filter_mode != "all" and g.match_type != filter_mode:
            return

        if len(self._group_sections) < 30:
            if hasattr(self, "workspace_stack"):
                self.workspace_stack.setCurrentWidget(self.cards_scroll)
            current_zoom = self.slider_zoom.value() if hasattr(self, "slider_zoom") else 310
            sec = self._make_group_section(g, current_zoom)
            self._group_sections[g.group_id] = sec
            self.cards_layout.addWidget(sec)
        self.lbl_group_title.setText(
            f"Found {len(self.groups)} Duplicate / Similar Groups — Scroll to compare all"
        )

    def _sort_visible_groups(self, groups_list: list[DuplicateGroup]) -> list[DuplicateGroup]:
        sort_key = self.cmb_sort_groups.currentData() if hasattr(self, "cmb_sort_groups") else "reclaimable"
        group_by = self.cmb_group_by.currentData() if hasattr(self, "cmb_group_by") else "none"

        def _item_sort_tuple(g: DuplicateGroup) -> Any:
            if sort_key == "similarity":
                return (-float(g.similarity), -int(g.reclaimable_bytes))
            if sort_key == "count":
                return (-len(g.items), -int(g.reclaimable_bytes))
            if sort_key == "date_desc":
                dmax = max(((it.date_taken if it.date_taken > 0 else it.mtime) for it in g.items), default=0.0)
                return (-float(dmax), -int(g.reclaimable_bytes))
            if sort_key == "date_asc":
                dmin = min(((it.date_taken if it.date_taken > 0 else it.mtime) for it in g.items), default=0.0)
                return (float(dmin), -int(g.reclaimable_bytes))
            if sort_key == "folder":
                f = str(g.items[0].folder_path).lower() if g.items else ""
                return (f, -int(g.reclaimable_bytes))
            return (-int(g.reclaimable_bytes), -float(g.similarity))

        if group_by == "match_type":
            order = {"exact": 0, "visual": 1, "burst": 2, "similar": 3}
            return sorted(groups_list, key=lambda g: (order.get(g.match_type, 9), _item_sort_tuple(g)))
        if group_by == "folder":
            return sorted(groups_list, key=lambda g: (str(g.items[0].folder_path).lower() if g.items else "", _item_sort_tuple(g)))
        if group_by == "month":
            def _month_sort_key(g: DuplicateGroup) -> tuple[int, int]:
                ts = max(((it.date_taken if it.date_taken > 0 else it.mtime) for it in g.items), default=0.0)
                if ts > 0:
                    import datetime
                    dt = datetime.datetime.fromtimestamp(ts)
                    return (-dt.year, -dt.month)
                return (9999, 99)
            return sorted(groups_list, key=lambda g: (_month_sort_key(g), _item_sort_tuple(g)))
        return sorted(groups_list, key=_item_sort_tuple)

    def _populate_groups_list(self) -> None:
        self._update_type_filter_counts()

        filter_mode = self._active_filter_mode()
        q = self.txt_group_filter.text().strip().lower() if hasattr(self, "txt_group_filter") else ""
        visible_groups = self._sort_visible_groups(
            [
                g for g in self.groups
                if (filter_mode == "all" or g.match_type == filter_mode)
                and (
                    not q
                    or any(q in it.filename.lower() or q in str(it.folder_path).lower() for it in g.items)
                )
            ]
        )

        if visible_groups:
            self._render_all_groups_feed(visible_groups)
            if hasattr(self, "workspace_stack"):
                self.workspace_stack.setCurrentWidget(self.cards_scroll)
            self.active_group = visible_groups[0]
            self.lbl_group_title.setText(
                f"Showing {len(visible_groups)} Duplicate / Similar Groups — Scroll to compare all"
            )
        else:
            self._clear_workspace()
            self._show_empty_state(post_scan=bool(self._scan_generation > 0))

    def _format_group_header_text(self, group: DuplicateGroup) -> str:
        type_str = "Exact Duplicate (100%)"
        if group.match_type == "burst":
            type_str = f"Burst Sequence ({group.time_span_sec}s)"
        elif group.match_type == "similar":
            type_str = f"Similar Photos ({int(group.similarity * 100)}%)"
        elif group.match_type == "visual":
            type_str = f"Visual Match ({int(group.similarity * 100)}%)"
        return f"{type_str} — {len(group.items)} Photos ({group.formatted_reclaimable_size} Reclaimable)"

    def _make_group_section(self, group: DuplicateGroup, current_zoom: int) -> QFrame:
        section = QFrame(self.cards_container)
        section.setObjectName("DupGroupSection")
        section.setStyleSheet("QFrame#DupGroupSection { background: transparent; border: none; }")
        sv = QVBoxLayout(section)
        sv.setContentsMargins(0, 2, 0, 10)
        sv.setSpacing(8)

        hdr_bar = QFrame(section)
        hdr_bar.setObjectName("DupGroupHeaderBar")
        hdr_bar.setFixedHeight(36)
        h_bg = "#1E232B" if self.is_dark else "#F4F7FB"
        h_bd = "#2D333F" if self.is_dark else "#D0DBE5"
        hdr_bar.setStyleSheet(
            f"QFrame#DupGroupHeaderBar {{ background-color: {h_bg}; border: 1px solid {h_bd}; border-radius: 4px; }}"
        )
        hdr = QHBoxLayout(hdr_bar)
        hdr.setContentsMargins(10, 4, 8, 4)
        hdr.setSpacing(8)
        lbl_hdr = QLabel(self._format_group_header_text(group), hdr_bar)
        lbl_hdr.setFont(QFont("Segoe UI", 9.5, QFont.Bold))
        lbl_hdr.setStyleSheet(
            f"color: {'#60A5FA' if self.is_dark else '#005A9E'}; border: none; background: transparent;"
        )
        hdr.addWidget(lbl_hdr, 1)

        btn_loupe = QPushButton("Compare", section)
        btn_loupe.setFixedHeight(24)
        btn_loupe.setCursor(Qt.PointingHandCursor)
        btn_loupe.setToolTip("Open this group in Light Table Compare Mode (2-Up, 3-Up, 2×2 Quad, Focus Peaking & Zoom-to-Face)")
        btn_loupe.setStyleSheet(
            "QPushButton { background-color: #252D37; border: 1px solid #3E4B5B; color: #F1F5F9; border-radius: 3px; padding: 2px 10px; }"
            "QPushButton:hover { background-color: #2F3946; border-color: #60A5FA; }"
            if self.is_dark
            else "QPushButton { background-color: #FFFFFF; border: 1px solid #CBD5E1; color: #1E293B; border-radius: 3px; padding: 2px 10px; }"
            "QPushButton:hover { background-color: #EFF6FF; border-color: #0078D4; color: #005A9E; }"
        )
        btn_loupe.clicked.connect(
            lambda _chk=False, g=group: self.compare_group_requested.emit([int(it.media_id) for it in g.items])
        )
        hdr.addWidget(btn_loupe)

        grp_marked_cnt = sum(1 for it in group.items if it.is_selected_for_removal)
        btn_clean_grp = QPushButton(
            f"🗑 Delete Duplicates ({grp_marked_cnt})" if grp_marked_cnt > 0 else "🗑 Delete Duplicates",
            section,
        )
        btn_clean_grp.setObjectName("GroupDeleteBtn")
        btn_clean_grp.setFixedHeight(24)
        btn_clean_grp.setEnabled(grp_marked_cnt > 0)
        btn_clean_grp.setFont(QFont("Segoe UI", 8.5, QFont.Weight.Bold))
        btn_clean_grp.setCursor(Qt.PointingHandCursor)
        btn_clean_grp.setToolTip("Move the duplicate photos marked REMOVE in this group to the Recycle Bin")
        btn_clean_grp.setStyleSheet(
            "QPushButton { background-color: #B91C1C; border: 1px solid #B91C1C; color: #FFFFFF; font-weight: bold; border-radius: 3px; padding: 2px 10px; }"
            "QPushButton:hover { background-color: #DC2626; border-color: #EF4444; }"
            "QPushButton:disabled { background-color: #3B2327; border-color: #5C2B32; color: #94A3B8; }"
            if self.is_dark
            else "QPushButton { background-color: #D32F2F; border: 1px solid #D32F2F; color: #FFFFFF; font-weight: bold; border-radius: 3px; padding: 2px 10px; }"
            "QPushButton:hover { background-color: #E53935; border-color: #C62828; }"
            "QPushButton:disabled { background-color: #E2E8F0; border-color: #CBD5E1; color: #64748B; }"
        )
        btn_clean_grp.clicked.connect(lambda _chk=False, g=group: self._recycle_group_marked(g))
        hdr.addWidget(btn_clean_grp)
        sv.addWidget(hdr_bar)

        row_container = QWidget(section)
        row_container.setStyleSheet("background: transparent;")
        row_layout = _FlowLayout(row_container, margin=0, spacing=10)

        cards: list[_ImageComparisonCard] = []
        for it in group.items:
            card = _ImageComparisonCard(
                item=it,
                group=group,
                thumb_repo=self.thumb_repo,
                is_dark=self.is_dark,
                card_width=current_zoom,
                parent=row_container,
            )
            card.toggled_keep.connect(
                lambda mid, g=group, sec=section, lbl=lbl_hdr, clist=cards, bc=btn_clean_grp: self._on_group_card_toggled_keep(
                    g, mid, lbl, clist, bc
                )
            )
            card.recycle_single_requested.connect(
                lambda mid, g=group: self._recycle_single_item(g, mid)
            )
            card.find_similar_requested.connect(self.find_similar_requested.emit)
            card.compare_requested.connect(
                lambda _mid, g=group: self.compare_group_requested.emit([int(x.media_id) for x in g.items])
            )
            card.card_selected.connect(self._on_card_selected)
            cards.append(card)
            row_layout.addWidget(card)
        sv.addWidget(row_container)
        section._lbl_hdr = lbl_hdr  # type: ignore[attr-defined]
        section._btn_clean_grp = btn_clean_grp  # type: ignore[attr-defined]
        section._flow_layout = row_layout  # type: ignore[attr-defined]
        section._cards = cards  # type: ignore[attr-defined]
        return section

    def _render_all_groups_feed(self, visible_groups: list[DuplicateGroup], *, limit: int = 30) -> None:
        self.cards_container.setUpdatesEnabled(False)
        try:
            self._clear_workspace()
            self._group_sections = {}
            self._feed_visible_groups = list(visible_groups)
            self._feed_loaded_count = 0
            current_zoom = self.slider_zoom.value() if hasattr(self, "slider_zoom") else 310
            self._append_groups_to_feed(self._feed_visible_groups, start_idx=0, limit=limit, current_zoom=current_zoom)
        finally:
            self.cards_container.setUpdatesEnabled(True)

    def _on_feed_scrolled(self, value: int) -> None:
        if not self._feed_visible_groups or self._feed_loaded_count >= len(self._feed_visible_groups):
            return
        sb = self.cards_scroll.verticalScrollBar()
        if sb.maximum() > 0 and value >= sb.maximum() - 350:
            current_zoom = self.slider_zoom.value() if hasattr(self, "slider_zoom") else 310
            self._append_groups_to_feed(
                self._feed_visible_groups,
                start_idx=self._feed_loaded_count,
                limit=30,
                current_zoom=current_zoom,
            )

    def _on_card_selected(self, media_id: int) -> None:
        self._selected_media_id = int(media_id)
        for c in self.cards_container.findChildren(_ImageComparisonCard):
            c.set_selected(c.item.media_id == self._selected_media_id)
        self.media_selected.emit(self._selected_media_id)

    def selected_media_id(self) -> int | None:
        return self._selected_media_id

    def _on_workspace_context_menu(self, pos: QPoint) -> None:
        child = self.cards_container.childAt(pos)
        w = child
        while w is not None and w is not self.cards_container:
            if isinstance(w, _ImageComparisonCard):
                return
            w = w.parentWidget()
        menu = QMenu(self)
        cur_grp = self.cmb_group_by.currentData() if hasattr(self, "cmb_group_by") else "none"
        grp_sub = menu.addMenu("Group by")
        for g_lbl, g_key in (
            ("None (Flat Feed)", "none"),
            ("Match Type", "match_type"),
            ("Folder", "folder"),
            ("Month taken", "month"),
        ):
            act = grp_sub.addAction(g_lbl)
            act.setCheckable(True)
            act.setChecked(cur_grp == g_key)
            act.triggered.connect(lambda _=False, k=g_key: self.set_group_by_key(k))

        cur_sort = self.cmb_sort_groups.currentData() if hasattr(self, "cmb_sort_groups") else "reclaimable"
        sort_sub = menu.addMenu("Sort by")
        for s_lbl, s_key in (
            ("Recoverable Space", "reclaimable"),
            ("Similarity %", "similarity"),
            ("Photo Count", "count"),
            ("Newest Date", "date_desc"),
            ("Oldest Date", "date_asc"),
            ("Folder (A–Z)", "folder"),
        ):
            act = sort_sub.addAction(s_lbl)
            act.setCheckable(True)
            act.setChecked(cur_sort == s_key)
            act.triggered.connect(lambda _=False, k=s_key: self.set_sort_by_key(k))

        menu.addSeparator()
        act_refresh = menu.addAction("Refresh Scan")
        act_refresh.triggered.connect(self.start_scan)
        menu.exec(self.cards_container.mapToGlobal(pos))

    def _append_groups_to_feed(
        self,
        visible_groups: list[DuplicateGroup],
        *,
        start_idx: int,
        limit: int,
        current_zoom: int,
    ) -> None:
        # Remove trailing 'Show More' button if present
        count = self.cards_layout.count()
        if count > 0:
            last_w = self.cards_layout.itemAt(count - 1).widget()
            if last_w is not None and last_w.objectName() == "BtnLoadMoreGroups":
                self.cards_layout.takeAt(count - 1)
                last_w.deleteLater()

        end_idx = min(len(visible_groups), start_idx + limit)
        group_by = self.cmb_group_by.currentData() if hasattr(self, "cmb_group_by") else "none"
        last_bucket = getattr(self, "_feed_last_bucket", None) if start_idx > 0 else None
        active_wlpg_hdr: _WLPGSectionHeader | None = getattr(self, "_feed_active_wlpg_hdr", None) if start_idx > 0 else None

        def _group_bucket(grp: DuplicateGroup) -> tuple[Any, str]:
            if group_by == "match_type":
                b = grp.match_type
                lbl = {
                    "exact": "Exact Duplicates (100%)",
                    "visual": "Visual Matches",
                    "burst": "Burst Sequences",
                    "similar": "Similar Photos",
                }.get(b, b.title())
                return b, lbl
            if group_by == "folder":
                b = str(grp.items[0].folder_path) if grp.items else ""
                return b, (b if b else "Unknown Folder")
            if group_by == "month":
                ts = max(((it.date_taken if it.date_taken > 0 else it.mtime) for it in grp.items), default=0.0)
                if ts > 0:
                    import datetime
                    dt = datetime.datetime.fromtimestamp(ts)
                    return (dt.year, dt.month), dt.strftime("%B %Y")
                return (0, 0), "Unknown Date"
            return None, ""

        # Precompute bucket summaries for all visible_groups when grouped
        bucket_stats: dict[Any, tuple[int, str]] = {}
        if group_by in ("match_type", "folder", "month"):
            counts_b: dict[Any, int] = {}
            bytes_b: dict[Any, int] = {}
            for vg in visible_groups:
                bk, _ = _group_bucket(vg)
                counts_b[bk] = counts_b.get(bk, 0) + 1
                bytes_b[bk] = bytes_b.get(bk, 0) + int(vg.reclaimable_bytes)
            for bk, cnt_v in counts_b.items():
                bt = float(bytes_b[bk])
                sz_b = (
                    f"{bt / (1024 ** 3):.2f} GB" if bt >= 1024 ** 3
                    else f"{bt / (1024 ** 2):.1f} MB" if bt >= 1024 ** 2
                    else f"{bt / 1024:.0f} KB" if bt >= 1024
                    else f"{int(bt)} B"
                )
                bucket_stats[bk] = (cnt_v, sz_b)

        for g in visible_groups[start_idx:end_idx]:
            bucket, bucket_label = _group_bucket(g)
            if bucket is not None and bucket != last_bucket:
                last_bucket = bucket
                cnt_b, sz_b = bucket_stats.get(bucket, (1, ""))
                active_wlpg_hdr = _WLPGSectionHeader(
                    title=bucket_label,
                    group_count=cnt_b,
                    reclaimable_text=sz_b,
                    is_dark=self.is_dark,
                    parent=self.cards_container,
                )
                active_wlpg_hdr.recycle_section_requested.connect(self._recycle_section_marked)
                self.cards_layout.addWidget(active_wlpg_hdr)
            sec = self._make_group_section(g, current_zoom)
            if active_wlpg_hdr is not None and bucket is not None:
                active_wlpg_hdr.child_sections.append(sec)
                active_wlpg_hdr.section_groups.append(g)
                active_wlpg_hdr.refresh_recycle_button()
                if active_wlpg_hdr.collapsed:
                    sec.setVisible(False)
            self._group_sections[g.group_id] = sec
            self.cards_layout.addWidget(sec)
        self._feed_last_bucket = last_bucket
        self._feed_active_wlpg_hdr = active_wlpg_hdr
        self._feed_loaded_count = end_idx

        remaining = len(visible_groups) - end_idx
        if remaining > 0:
            btn_more = QPushButton(f"Show Next {min(limit, remaining)} Groups ({remaining} remaining)", self.cards_container)
            btn_more.setObjectName("BtnLoadMoreGroups")
            btn_more.setFixedHeight(32)
            btn_more.setCursor(Qt.PointingHandCursor)
            btn_more.clicked.connect(
                lambda _chk=False, vg=visible_groups, nxt=end_idx, lim=limit: self._append_groups_to_feed(
                    vg, start_idx=nxt, limit=lim, current_zoom=(self.slider_zoom.value() if hasattr(self, "slider_zoom") else 310)
                )
            )
            self.cards_layout.addWidget(btn_more)

    def _on_group_card_toggled_keep(
        self,
        group: DuplicateGroup,
        media_id: int,
        lbl_hdr: QLabel,
        cards: list[_ImageComparisonCard],
        btn_clean_grp: QPushButton | None = None,
    ) -> None:
        self.active_group = group
        if group.match_type == "similar":
            for it in group.items:
                if it.media_id == media_id:
                    if it.is_recommended_keep:
                        it.is_recommended_keep = False
                        it.is_selected_for_removal = True
                    elif it.is_selected_for_removal:
                        it.is_selected_for_removal = False
                    else:
                        it.is_selected_for_removal = True
        else:
            for it in group.items:
                if it.media_id == media_id:
                    it.is_recommended_keep = True
                    it.is_selected_for_removal = False
                else:
                    it.is_recommended_keep = False
                    it.is_selected_for_removal = True

        for c in cards:
            c.update_selection_state()
        lbl_hdr.setText(self._format_group_header_text(group))
        if btn_clean_grp is not None:
            mc = sum(1 for it in group.items if it.is_selected_for_removal)
            btn_clean_grp.setText(f"🗑 Delete Duplicates ({mc})" if mc > 0 else "🗑 Delete Duplicates")
            btn_clean_grp.setEnabled(mc > 0)
        for hdr_w in self.cards_container.findChildren(_WLPGSectionHeader):
            hdr_w.refresh_recycle_button()
        self._update_summary_counts()

    def _render_group_cards(self, group: DuplicateGroup) -> None:
        self._clear_workspace()
        if hasattr(self, "workspace_stack"):
            self.workspace_stack.setCurrentWidget(self.cards_scroll)
        self.lbl_group_title.setText(self._format_group_header_text(group))

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
            card.recycle_single_requested.connect(
                lambda mid, g=group: self._recycle_single_item(g, mid)
            )
            card.find_similar_requested.connect(self.find_similar_requested.emit)
            self.cards_layout.addWidget(card)

    def set_use_ribbon_mode(self, enabled: bool) -> None:
        """Hide redundant internal header controls while keeping batch action bar accessible."""
        self._use_ribbon_mode = bool(enabled)
        if hasattr(self, "top_bar"):
            self.top_bar.setVisible(not self._use_ribbon_mode)
        if hasattr(self, "ws_controls_widget"):
            self.ws_controls_widget.setVisible(not self._use_ribbon_mode)
        for attr in ("lbl_sort_groups", "cmb_sort_groups", "lbl_group_by", "cmb_group_by"):
            w = getattr(self, attr, None)
            if w is not None:
                w.setVisible(not self._use_ribbon_mode)
        if hasattr(self, "footer_actions_widget"):
            self.footer_actions_widget.setVisible(True)

    def compare_active_group(self) -> None:
        """Open the active duplicate group (or first visible group) in Light Table Compare Mode."""
        grp = self.active_group
        if grp is None and getattr(self, "_feed_visible_groups", None):
            grp = self._feed_visible_groups[0]
        elif grp is None and self.groups:
            grp = self.groups[0]
        if grp is not None and grp.items:
            self.compare_group_requested.emit([int(it.media_id) for it in grp.items])

    def set_similarity_threshold(self, pct: int) -> None:
        self._similarity_threshold = max(0.80, min(0.99, float(pct) / 100.0))

    def set_merge_metadata_enabled(self, enabled: bool) -> None:
        self._merge_metadata_on_clean = bool(enabled)

    def set_metadata_mode(self, mode: str) -> None:
        mode = mode if mode in ("ask", "always", "never") else "ask"
        self._metadata_mode = mode
        QSettings("PhotoGallery64", "PhotoGallery64").setValue("ui/dup_metadata_mode", mode)
        if mode == "always":
            self._merge_metadata_on_clean = True
        elif mode == "never":
            self._merge_metadata_on_clean = False
        else:
            self._merge_metadata_on_clean = bool(
                QSettings("PhotoGallery64", "PhotoGallery64").value("ui/dup_merge_metadata", True, type=bool)
            )

    def set_sort_by_key(self, sort_key: str) -> None:
        if hasattr(self, "cmb_sort_groups"):
            idx = self.cmb_sort_groups.findData(sort_key)
            if idx >= 0:
                if self.cmb_sort_groups.currentIndex() == idx:
                    self._populate_groups_list()
                else:
                    self.cmb_sort_groups.setCurrentIndex(idx)

    def set_group_by_key(self, group_key: str) -> None:
        if hasattr(self, "cmb_group_by"):
            idx = self.cmb_group_by.findData(group_key)
            if idx >= 0:
                if self.cmb_group_by.currentIndex() == idx:
                    self._populate_groups_list()
                else:
                    self.cmb_group_by.setCurrentIndex(idx)

    def set_rule_by_name(self, rule_label: str) -> None:
        mapping = {
            "Highest Resolution": 0,
            "Largest File": 1,
            "Oldest Date": 2,
            "Most Metadata": 3,
            "Best Sharpness": 4,
            "Original Filename": 5,
        }
        idx = mapping.get(rule_label)
        if idx is not None and hasattr(self, "cmb_rule"):
            self.cmb_rule.setCurrentIndex(idx)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if hasattr(self, "cards_scroll") and watched in (self.cards_scroll.viewport(), self.cards_container):
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
        for w in self.cards_container.findChildren(_ImageComparisonCard):
            w.set_zoom(val)
        self.zoom_changed.emit(int(val))

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

    def _current_rule_and_prefs(self) -> tuple[str, list[str]]:
        rule_map = {
            0: "highest_resolution",
            1: "largest_file_size",
            2: "oldest_date",
            3: "most_metadata",
            4: "best_sharpness",
            5: "original_filename",
            6: "preferred_folder",
        }
        rule = rule_map.get(self.cmb_rule.currentIndex(), "highest_resolution")
        raw_prefs = self.txt_preferred_folder.text() if hasattr(self, "txt_preferred_folder") else ""
        prefs = [p.strip() for p in raw_prefs.split(";") if p.strip()]
        return rule, prefs

    def _browse_preferred_folder(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        chosen = QFileDialog.getExistingDirectory(self, "Choose Preferred Keep Folder")
        if chosen:
            cur = self.txt_preferred_folder.text().strip()
            self.txt_preferred_folder.setText(f"{cur}; {chosen}" if cur else chosen)
            self._on_rule_changed()

    def _on_rule_changed(self) -> None:
        rule, prefs = self._current_rule_and_prefs()
        is_pref = rule == "preferred_folder"
        if hasattr(self, "txt_preferred_folder"):
            self.txt_preferred_folder.setVisible(is_pref)
        if hasattr(self, "btn_browse_pref_folder"):
            self.btn_browse_pref_folder.setVisible(is_pref)
        if self.active_group:
            auto_recommend_rules([self.active_group], rule=rule, preferred_folders=prefs)
            for c in self.cards_container.findChildren(_ImageComparisonCard):
                if c.group is self.active_group:
                    c.update_selection_state()
            self._update_summary_counts()

    def _apply_rule_to_all_groups(self) -> None:
        rule, prefs = self._current_rule_and_prefs()
        auto_recommend_rules(self.groups, rule=rule, preferred_folders=prefs)
        self._populate_groups_list()
        self._update_summary_counts()

    def _update_summary_counts(self) -> None:
        tot_removal = sum(1 for g in self.groups for it in g.items if it.is_selected_for_removal)
        tot_bytes = sum(it.file_size for g in self.groups for it in g.items if it.is_selected_for_removal)
        tot_possible = sum(int(getattr(g, "reclaimable_bytes", 0) or 0) for g in self.groups)

        def _fmt_bytes(val: float) -> str:
            if val >= 1024 ** 3:
                return f"{val / (1024 ** 3):.2f} GB"
            if val >= 1024 ** 2:
                return f"{val / (1024 ** 2):.1f} MB"
            if val >= 1024:
                return f"{val / 1024:.0f} KB"
            return f"{int(val)} B"

        sz_str = _fmt_bytes(float(tot_bytes))
        pos_str = _fmt_bytes(float(tot_possible))
        self.lbl_summary.setText(
            f"{tot_removal} duplicate files marked for removal ({sz_str} reclaimable / {pos_str} total)"
        )
        self.btn_delete.setEnabled(tot_removal > 0)
        self.btn_delete.setText(
            f"🗑 Delete All Duplicates ({tot_removal})" if tot_removal > 0 else "Delete All Duplicates"
        )
        self.summary_state_changed.emit(tot_removal)

    def _clear_workspace(self) -> None:
        self._group_sections = {}
        while self.cards_layout.count() > 0:
            child = self.cards_layout.takeAt(0)
            w = child.widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        self.lbl_group_title.setText("Select a duplicate group to compare")

    def _merge_group_metadata_before_delete(self, deleted_mids: set[int]) -> None:
        if not self._merge_metadata_on_clean or not self.catalog_repo or not deleted_mids:
            return
        for grp in self.groups:
            rem_ids = [it.media_id for it in grp.items if it.media_id in deleted_mids]
            if not rem_ids:
                continue
            keep_item = next((it for it in grp.items if it.media_id not in deleted_mids), None)
            if keep_item is not None:
                try:
                    merge_duplicate_metadata(self.catalog_repo, keep_item.media_id, rem_ids)
                except Exception:
                    pass

    def _recycle_items_in_place(
        self,
        group: DuplicateGroup,
        items_to_remove: Sequence[DuplicateMediaItem],
    ) -> list[int]:
        """Move specified items in a group to Recycle Bin, merge metadata, and remove from catalog."""
        if not items_to_remove:
            return []
        try:
            from send2trash import send2trash
        except ImportError:
            send2trash = None

        rem_set = {it.media_id for it in items_to_remove}
        keep_item = next((it for it in group.items if it.media_id not in rem_set and not it.is_selected_for_removal), None)
        if keep_item is None:
            keep_item = next((it for it in group.items if it.media_id not in rem_set), None)

        deleted_mids: list[int] = []
        freed = 0
        for it in items_to_remove:
            p = it.full_path
            try:
                if p.exists():
                    if send2trash:
                        send2trash(str(p))
                    else:
                        p.unlink()
                deleted_mids.append(it.media_id)
                freed += int(it.file_size)
            except Exception:
                pass

        if deleted_mids and self.catalog_repo:
            if self._merge_metadata_on_clean and keep_item is not None:
                try:
                    merge_duplicate_metadata(self.catalog_repo, keep_item.media_id, deleted_mids)
                except Exception:
                    pass
            self.catalog_repo.delete_media(deleted_mids)

        if deleted_mids:
            del_set = set(deleted_mids)
            group.items = [it for it in group.items if it.media_id not in del_set]
            sec = self._group_sections.get(group.group_id)
            if len(group.items) < 2:
                self.groups = [g for g in self.groups if g.group_id != group.group_id]
                if sec is not None:
                    self._group_sections.pop(group.group_id, None)
                    for hdr_w in self.cards_container.findChildren(_WLPGSectionHeader):
                        if sec in hdr_w.child_sections:
                            hdr_w.child_sections.remove(sec)
                        if group in hdr_w.section_groups:
                            hdr_w.section_groups.remove(group)
                        hdr_w.refresh_recycle_button()
                    self.cards_layout.removeWidget(sec)
                    sec.hide()
                    sec.deleteLater()
                if not self.groups:
                    self._clear_workspace()
                    self._show_empty_state(post_scan=True)
            else:
                if not any(it.is_recommended_keep for it in group.items):
                    group.items[0].is_recommended_keep = True
                    group.items[0].is_selected_for_removal = False
                if sec is not None:
                    cards_list: list[_ImageComparisonCard] = getattr(sec, "_cards", [])
                    flow_lay = getattr(sec, "_flow_layout", None)
                    kept_cards: list[_ImageComparisonCard] = []
                    for c in cards_list:
                        if c.item.media_id in del_set:
                            if flow_lay is not None:
                                flow_lay.removeWidget(c)
                            c.hide()
                            c.deleteLater()
                        else:
                            c.update_selection_state()
                            kept_cards.append(c)
                    sec._cards = kept_cards  # type: ignore[attr-defined]
                    lbl_hdr = getattr(sec, "_lbl_hdr", None)
                    if lbl_hdr is not None:
                        lbl_hdr.setText(self._format_group_header_text(group))
                    btn_cg = getattr(sec, "_btn_clean_grp", None)
                    if btn_cg is not None:
                        mc = sum(1 for it in group.items if it.is_selected_for_removal)
                        btn_cg.setText(f"🗑 Delete Duplicates ({mc})" if mc > 0 else "🗑 Delete Duplicates")
                        btn_cg.setEnabled(mc > 0)
                for hdr_w in self.cards_container.findChildren(_WLPGSectionHeader):
                    hdr_w.refresh_recycle_button()
            self._update_type_filter_counts()
            self._update_summary_counts()
            self.duplicates_cleaned.emit(len(deleted_mids), freed)
        return deleted_mids

    def _recycle_single_item(self, group: DuplicateGroup, media_id: int, *, confirm: bool | None = None) -> None:
        item = next((it for it in group.items if it.media_id == media_id), None)
        if item is None:
            return
        should_confirm = confirm if confirm is not None else (self._metadata_mode == "ask")
        if should_confirm:
            s = float(item.file_size)
            sz_str = f"{s / (1024 ** 2):.1f} MB" if s >= 1024 ** 2 else f"{s / 1024:.0f} KB"
            if not self._confirm_cleanup_dialog(1, sz_str, title=f"Delete '{item.filename}'"):
                return
        self._recycle_items_in_place(group, [item])

    def _recycle_group_marked(self, group: DuplicateGroup, *, confirm: bool | None = None) -> None:
        marked = [it for it in group.items if it.is_selected_for_removal]
        if not marked:
            return
        should_confirm = confirm if confirm is not None else (self._metadata_mode == "ask")
        if should_confirm:
            tot_bytes = sum(int(it.file_size) for it in marked)
            s = float(tot_bytes)
            sz_str = f"{s / (1024 ** 2):.1f} MB" if s >= 1024 ** 2 else f"{s / 1024:.0f} KB"
            if not self._confirm_cleanup_dialog(len(marked), sz_str, title="Delete Group Duplicates"):
                return
        self._recycle_items_in_place(group, marked)

    def _confirm_cleanup_dialog(
        self,
        count: int,
        sz_str: str,
        *,
        is_backup: bool = False,
        title: str | None = None,
    ) -> bool:
        """WLPG cleanup confirmation with Option 1 metadata-merge checkbox persisted in QSettings."""
        dlg = QDialog(self)
        dlg.setWindowTitle(title or ("Move Duplicate Photos to Folder" if is_backup else "Delete Duplicate Photos"))
        dlg.setMinimumWidth(440)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(10)

        if is_backup:
            msg_html = (
                f"Move <b>{count}</b> duplicate photo(s) marked <b>REMOVE</b> (<b>{sz_str}</b>) "
                "to a folder outside your library and remove them from the catalog?<br><br>"
                "Your kept originals (marked <b>KEEP</b>) will stay untouched in place."
            )
        else:
            msg_html = (
                f"Move <b>{count}</b> duplicate photo(s) marked <b>REMOVE</b> (<b>{sz_str}</b>) "
                "to the Windows Recycle Bin?<br><br>"
                "Your kept originals (marked <b>KEEP</b>) will stay untouched in your library. "
                "You can restore deleted files from the Recycle Bin if needed."
            )
        lbl_msg = QLabel(msg_html, dlg)
        lbl_msg.setWordWrap(True)
        lay.addWidget(lbl_msg)

        chk_merge: QCheckBox | None = None
        if self._metadata_mode == "ask":
            chk_merge = QCheckBox(
                "Copy unique tags, people, captions, and higher star ratings onto kept copies",
                dlg,
            )
            chk_merge.setChecked(bool(self._merge_metadata_on_clean))
            lay.addWidget(chk_merge)
        else:
            note_txt = (
                "✓ Metadata policy: unique tags, people, captions, and higher ratings will be auto-merged onto kept copies."
                if self._metadata_mode == "always"
                else "ℹ Metadata policy: Do Not Merge (duplicate metadata will not be copied onto kept copies)."
            )
            lbl_meta_note = QLabel(note_txt, dlg)
            lbl_meta_note.setWordWrap(True)
            lbl_meta_note.setStyleSheet("color: #64748B; font-size: 11px;")
            lay.addWidget(lbl_meta_note)

        btns = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No, dlg)
        btns.button(QDialogButtonBox.Yes).setText("Choose Folder & Move..." if is_backup else "Move to Recycle Bin")
        btns.button(QDialogButtonBox.No).setDefault(True)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() != QDialog.Accepted:
            return False

        if chk_merge is not None:
            self._merge_metadata_on_clean = bool(chk_merge.isChecked())
            QSettings("PhotoGallery64", "PhotoGallery64").setValue("ui/dup_merge_metadata", self._merge_metadata_on_clean)
        elif self._metadata_mode == "always":
            self._merge_metadata_on_clean = True
        else:
            self._merge_metadata_on_clean = False
        return True

    def _recycle_section_marked(
        self,
        section_groups: list[DuplicateGroup],
        *,
        confirm: bool = True,
    ) -> list[int]:
        """Move all marked photos in the given section groups to the Recycle Bin in-place."""
        self.stop_scan()
        pairs: list[tuple[DuplicateGroup, list[DuplicateMediaItem]]] = []
        tot_bytes = 0
        tot_count = 0
        for g in section_groups:
            marked = [it for it in g.items if it.is_selected_for_removal]
            if marked:
                pairs.append((g, marked))
                tot_count += len(marked)
                tot_bytes += sum(int(it.file_size) for it in marked)
        if tot_count == 0:
            return []

        s = float(tot_bytes)
        sz_str = f"{s / (1024 ** 2):.1f} MB" if s >= 1024 ** 2 else f"{s / 1024:.0f} KB"
        if confirm and not self._confirm_cleanup_dialog(tot_count, sz_str, title="Recycle Section Duplicates"):
            return []

        try:
            from send2trash import send2trash
        except ImportError:
            send2trash = None

        all_deleted: list[int] = []
        freed_bytes = 0
        for g, marked in pairs:
            rem_set = {it.media_id for it in marked}
            keep_item = next((it for it in g.items if it.media_id not in rem_set and not it.is_selected_for_removal), None)
            if keep_item is None:
                keep_item = next((it for it in g.items if it.media_id not in rem_set), None)
            grp_deleted: list[int] = []
            for it in marked:
                p = it.full_path
                try:
                    if p.exists():
                        if send2trash:
                            send2trash(str(p))
                        else:
                            p.unlink()
                    grp_deleted.append(it.media_id)
                    freed_bytes += int(it.file_size)
                except Exception:
                    pass
            if grp_deleted and self.catalog_repo:
                if self._merge_metadata_on_clean and keep_item is not None:
                    try:
                        merge_duplicate_metadata(self.catalog_repo, keep_item.media_id, grp_deleted)
                    except Exception:
                        pass
                self.catalog_repo.delete_media(grp_deleted)
            if grp_deleted:
                del_set = set(grp_deleted)
                g.items = [it for it in g.items if it.media_id not in del_set]
                if len(g.items) < 2:
                    self.groups = [existing for existing in self.groups if existing.group_id != g.group_id]
                elif not any(it.is_recommended_keep for it in g.items):
                    g.items[0].is_recommended_keep = True
                    g.items[0].is_selected_for_removal = False
                all_deleted.extend(grp_deleted)

        if all_deleted:
            self._populate_groups_list()
            self._update_summary_counts()
            self.duplicates_cleaned.emit(len(all_deleted), freed_bytes)
        return all_deleted

    def _move_marked_to_recycle_bin(self) -> None:
        self.stop_scan()
        items_to_remove = [it for g in self.groups for it in g.items if it.is_selected_for_removal]
        if not items_to_remove:
            return

        tot_bytes = sum(it.file_size for it in items_to_remove)
        s = float(tot_bytes)
        sz_str = f"{s / (1024 ** 2):.1f} MB" if s >= 1024 ** 2 else f"{s / 1024:.0f} KB"

        if not self._confirm_cleanup_dialog(len(items_to_remove), sz_str, is_backup=False):
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
            self._merge_group_metadata_before_delete(set(deleted_mids))
            self.catalog_repo.delete_media(deleted_mids)

        self.duplicates_cleaned.emit(len(deleted_mids), tot_bytes)
        QMessageBox.information(
            self,
            "Duplicates Cleaned",
            f"Successfully moved {len(deleted_mids)} duplicate files to the Recycle Bin, recovering {sz_str}.",
        )
        self.back_to_gallery_requested.emit()

    def _move_marked_to_backup(self) -> None:
        self.stop_scan()
        items_to_remove = [it for g in self.groups for it in g.items if it.is_selected_for_removal]
        if not items_to_remove:
            return

        tot_bytes = sum(it.file_size for it in items_to_remove)
        s = float(tot_bytes)
        sz_str = f"{s / (1024 ** 2):.1f} MB" if s >= 1024 ** 2 else f"{s / 1024:.0f} KB"

        if not self._confirm_cleanup_dialog(len(items_to_remove), sz_str, is_backup=True):
            return

        dest_folder = QFileDialog.getExistingDirectory(self, "Select Folder to Move Marked Photos Into")
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
            self._merge_group_metadata_before_delete(set(deleted_mids))
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
