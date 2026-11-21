from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from PySide6.QtCore import QEvent, QPoint, QSize, Qt
from PySide6.QtGui import QColor, QCursor, QGuiApplication, QIcon, QImageReader, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.gps_matcher import (
    DonorPoint,
    ProposedMatch,
    TargetPhoto,
    format_time_gap,
    match_timed_photos,
    parse_gpx_file,
    resolve_missing_place_names,
)
from src.ui.theme_tokens import get_theme_tokens, set_ui_role


def _fmt_ts(ts: int) -> str:
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"


def _fmt_short_ts(ts: int) -> str:
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%b %d, %H:%M:%S")
    except Exception:
        return "-"


class _SortableTableItem(QTableWidgetItem):
    """QTableWidgetItem that sorts by an explicit Qt.UserRole + 1 key when present."""

    SORT_ROLE = Qt.UserRole + 1

    def __lt__(self, other: QTableWidgetItem) -> bool:
        k1 = self.data(self.SORT_ROLE)
        k2 = other.data(self.SORT_ROLE)
        if k1 is not None and k2 is not None:
            try:
                return k1 < k2
            except TypeError:
                pass
        return super().__lt__(other)


def _make_gpx_icon(size: int = 96) -> QIcon:
    """Create a clean GPX route badge icon for GPX-sourced matches."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setBrush(QColor("#EFF6FF"))
    p.setPen(QPen(QColor("#93C5FD"), 2.0))
    p.drawRoundedRect(3, 3, size - 6, size - 6, 10, 10)
    p.setPen(QPen(QColor("#2563EB"), 3.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.drawPolyline([
        (int(size * 0.22), int(size * 0.72)),
        (int(size * 0.40), int(size * 0.48)),
        (int(size * 0.58), int(size * 0.58)),
        (int(size * 0.76), int(size * 0.26)),
    ])
    p.setBrush(QColor("#16A34A"))
    p.setPen(Qt.NoPen)
    p.drawEllipse(int(size * 0.17), int(size * 0.67), 9, 9)
    p.setBrush(QColor("#DC2626"))
    p.drawEllipse(int(size * 0.71), int(size * 0.21), 9, 9)
    p.end()
    return QIcon(pm)


class _HoverPhotoPreviewPopup(QFrame):
    """Floating enlarged preview card shown when hovering over Destination or GPS Source cells."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setFrameShape(QFrame.StyledPanel)
        is_dark = bool(parent is not None and parent.palette().color(parent.backgroundRole()).lightness() < 128)
        tok = get_theme_tokens(is_dark=is_dark)
        self.setStyleSheet(
            f"QFrame {{ background-color: {tok.bg_elevated}; border: 1px solid {tok.border_subtle}; border-radius: 6px; }}"
            f"QLabel {{ color: {tok.text_primary}; background: transparent; border: none; }}"
        )
        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(10, 10, 10, 10)
        vbox.setSpacing(6)

        self.lbl_role = QLabel("", self)
        self.lbl_role.setStyleSheet(f"font-size: 9pt; font-weight: 600; color: {tok.accent};")
        vbox.addWidget(self.lbl_role)

        self.lbl_img = QLabel(self)
        self.lbl_img.setAlignment(Qt.AlignCenter)
        self.lbl_img.setFixedSize(260, 195)
        self.lbl_img.setStyleSheet(f"background-color: {tok.bg_card}; border-radius: 4px;")
        vbox.addWidget(self.lbl_img)

        self.lbl_meta = QLabel("", self)
        self.lbl_meta.setWordWrap(True)
        self.lbl_meta.setMaximumWidth(260)
        self.lbl_meta.setStyleSheet("font-size: 9pt; line-height: 1.3;")
        vbox.addWidget(self.lbl_meta)

    def show_preview(
        self,
        *,
        role: str,
        pixmap: QPixmap,
        filename: str,
        camera: str,
        capture_time: str,
        location: str,
        global_pos: QPoint,
    ) -> None:
        self.lbl_role.setText(role)
        if not pixmap.isNull():
            self.lbl_img.setPixmap(
                pixmap.scaled(260, 195, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            self.lbl_img.setVisible(True)
        else:
            self.lbl_img.setVisible(False)

        lines = [f"<b>{filename}</b>"]
        if camera:
            lines.append(f"Camera: {camera}")
        if capture_time:
            lines.append(f"Taken: {capture_time}")
        if location:
            lines.append(f"Location: {location}")
        self.lbl_meta.setText("<br>".join(lines))
        self.adjustSize()

        pos = QPoint(global_pos.x() + 18, global_pos.y() + 14)
        screen = QGuiApplication.screenAt(global_pos) or QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            if pos.x() + self.width() > avail.right():
                pos.setX(max(avail.left(), global_pos.x() - self.width() - 16))
            if pos.y() + self.height() > avail.bottom():
                pos.setY(max(avail.top(), avail.bottom() - self.height() - 8))
        self.move(pos)
        self.show()


class _CalibrateClockDialog(QDialog):
    """Two-photo picker to auto-calculate camera clock difference = t_phone - t_camera."""

    def __init__(
        self,
        targets: list[TargetPhoto],
        donors: list[DonorPoint],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sync Camera Clock Using 2 Photos")
        self.resize(680, 400)
        self.computed_offset_sec: int = 0

        root = QVBoxLayout(self)
        root.setSpacing(8)
        intro = QLabel(
            "Pick one camera photo (without GPS) and one phone photo (with GPS) that were taken at the exact same moment.\n"
            "We will calculate the clock difference automatically.",
            self,
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        cols = QHBoxLayout()
        left_box = QVBoxLayout()
        left_box.addWidget(QLabel("1. Camera Photo (Without GPS):", self))
        self.list_targets = QListWidget(self)
        for t in targets[:400]:
            lbl = f"{_fmt_ts(t.timestamp)}  •  {t.filename}"
            if t.camera_model:
                lbl += f"  ({t.camera_model})"
            item = QListWidgetItem(lbl)
            item.setData(Qt.UserRole, int(t.timestamp))
            self.list_targets.addItem(item)
        left_box.addWidget(self.list_targets, 1)
        cols.addLayout(left_box, 1)

        right_box = QVBoxLayout()
        right_box.addWidget(QLabel("2. Phone / GPS Photo (Same Moment):", self))
        self.list_donors = QListWidget(self)
        for d in donors[:400]:
            if d.source == "gpx":
                continue
            lbl = f"{_fmt_ts(d.timestamp)}  •  {d.filename or d.location_name or 'GPS Point'}"
            if d.location_name:
                lbl += f"  [{d.location_name}]"
            item = QListWidgetItem(lbl)
            item.setData(Qt.UserRole, int(d.timestamp))
            self.list_donors.addItem(item)
        right_box.addWidget(self.list_donors, 1)
        cols.addLayout(right_box, 1)
        root.addLayout(cols, 1)

        self.lbl_preview = QLabel("Select one photo from each list.", self)
        self.lbl_preview.setStyleSheet("font-weight: 600;")
        root.addWidget(self.lbl_preview)

        self.list_targets.currentItemChanged.connect(self._recalc)
        self.list_donors.currentItemChanged.connect(self._recalc)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.btn_ok = btns.button(QDialogButtonBox.Ok)
        self.btn_ok.setText("Use Difference")
        self.btn_ok.setEnabled(False)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def _recalc(self) -> None:
        t_it = self.list_targets.currentItem()
        d_it = self.list_donors.currentItem()
        if t_it is None or d_it is None:
            self.btn_ok.setEnabled(False)
            return
        t_cam = int(t_it.data(Qt.UserRole))
        t_ph = int(d_it.data(Qt.UserRole))
        diff = t_ph - t_cam
        self.computed_offset_sec = diff
        sign = "+" if diff >= 0 else "-"
        self.lbl_preview.setText(f"Calculated Difference: {sign}{format_time_gap(abs(diff))} ({diff:+d}s)")
        self.btn_ok.setEnabled(True)


class GpsMatchDialog(QDialog):
    """
    Companion Phone GPS Matching dialog:
    Shows a clear Source (Phone GPS Photo / GPX Track) -> Destination (Camera Photo)
    preview with in-dialog Scope switching and optional clock-difference controls.
    """

    def __init__(
        self,
        targets: list[TargetPhoto],
        donors: list[DonorPoint],
        *,
        scope_label: str = "",
        db_engine: Any = None,
        thumb_repo: Any = None,
        scope_options: list[tuple[str, str]] | None = None,
        initial_scope: str | None = None,
        scope_loader: Callable[[str], tuple[list[TargetPhoto], list[DonorPoint], str]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Match Locations from Timed Photos")
        self.resize(1060, 720)
        self._targets = list(targets)
        self._base_donors = list(donors)
        self._gpx_donors: list[DonorPoint] = []
        self._donors = list(donors)
        self._db_engine = db_engine
        self._thumb_repo = thumb_repo
        self._scope_loader = scope_loader
        self._scope_label = scope_label
        self._matches: list[ProposedMatch] = []
        self._gpx_files_loaded: list[str] = []
        self._thumb_icon_cache: dict[int, QIcon] = {}
        self._large_pixmap_cache: dict[int, QPixmap] = {}
        self._gpx_icon: QIcon | None = None
        self._path_map: dict[int, str] = {int(t.media_id): str(t.path) for t in self._targets if t.path}
        for d in self._donors:
            if d.media_id is not None and d.path:
                self._path_map[int(d.media_id)] = str(d.path)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        # 1. Plain-English Intro
        lbl_title = QLabel("<b>Copy GPS locations from your phone photos (or a GPX track) to your camera photos</b>", self)
        lbl_title.setStyleSheet("font-size: 10.5pt;")
        lbl_title.setWordWrap(True)
        root.addWidget(lbl_title)

        lbl_sub = QLabel(
            "Step 1: Choose which photos to scan (or load a GPX file).  "
            "Step 2: Review the matched pairs below, uncheck any you don't want, and click Apply.",
            self,
        )
        lbl_sub.setWordWrap(True)
        set_ui_role(lbl_sub, "muted")
        root.addWidget(lbl_sub)

        # 2. Scope & Source Bar (modeled after Find Duplicates)
        scope_frame = QFrame(self)
        scope_frame.setFrameShape(QFrame.StyledPanel)
        scope_vbox = QVBoxLayout(scope_frame)
        scope_vbox.setContentsMargins(10, 8, 10, 8)
        scope_vbox.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        top_row.addWidget(QLabel("<b>1. Photos to scan:</b>", scope_frame))

        self.combo_scope = QComboBox(scope_frame)
        if scope_options:
            for key, text in scope_options:
                self.combo_scope.addItem(text, key)
            if initial_scope:
                idx = self.combo_scope.findData(initial_scope)
                if idx >= 0:
                    self.combo_scope.setCurrentIndex(idx)
        else:
            self.combo_scope.addItem(scope_label or "Current Selection", "default")
            self.combo_scope.setEnabled(False)
        self.combo_scope.currentIndexChanged.connect(self._on_scope_changed)
        top_row.addWidget(self.combo_scope)

        self.btn_import_gpx = QPushButton("Load GPX Track...", scope_frame)
        self.btn_import_gpx.setToolTip("Optional: load a .gpx GPS track log recorded on your phone or GPS watch.")
        self.btn_import_gpx.clicked.connect(self._import_gpx)
        top_row.addWidget(self.btn_import_gpx)

        self.lbl_scope_counts = QLabel("", scope_frame)
        set_ui_role(self.lbl_scope_counts, "muted")
        top_row.addWidget(self.lbl_scope_counts, 1)

        self.btn_toggle_timing = QPushButton("Clock & Time Window...", scope_frame)
        self.btn_toggle_timing.setCheckable(True)
        self.btn_toggle_timing.setToolTip("Adjust how close in time photos must be, or fix a wrong camera clock.")
        self.btn_toggle_timing.toggled.connect(self._on_timing_toggled)
        top_row.addWidget(self.btn_toggle_timing)
        scope_vbox.addLayout(top_row)

        # Collapsible Timing & Clock Fix Panel
        self.timing_panel = QWidget(scope_frame)
        timing_vbox = QVBoxLayout(self.timing_panel)
        timing_vbox.setContentsMargins(0, 4, 0, 0)
        timing_vbox.setSpacing(6)

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        lbl_gap = QLabel("Match photos within:", self.timing_panel)
        lbl_gap.setToolTip("Maximum time allowed between a camera photo and the nearest GPS-tagged photo or GPX point.")
        row1.addWidget(lbl_gap)
        self.combo_max_gap = QComboBox(self.timing_panel)
        self.combo_max_gap.setToolTip(lbl_gap.toolTip())
        for label, secs in (
            ("15 minutes", 900),
            ("30 minutes", 1800),
            ("1 hour", 3600),
            ("2 hours", 7200),
            ("4 hours", 14400),
            ("12 hours", 43200),
        ):
            self.combo_max_gap.addItem(label, secs)
        self.combo_max_gap.setCurrentIndex(3)  # 2 hours default
        self.combo_max_gap.currentIndexChanged.connect(self._recompute_matches)
        row1.addWidget(self.combo_max_gap)

        row1.addSpacing(12)
        lbl_clk = QLabel("Camera clock difference:", self.timing_panel)
        lbl_clk.setToolTip("If your camera's clock was ahead of or behind your phone, adjust it here.")
        row1.addWidget(lbl_clk)
        self.combo_sign = QComboBox(self.timing_panel)
        self.combo_sign.addItem("+", 1)
        self.combo_sign.addItem("-", -1)
        self.combo_sign.setFixedWidth(44)
        self.combo_sign.currentIndexChanged.connect(self._recompute_matches)
        row1.addWidget(self.combo_sign)

        self.spin_hrs = QSpinBox(self.timing_panel)
        self.spin_hrs.setRange(0, 23)
        self.spin_hrs.setSuffix("h")
        self.spin_hrs.valueChanged.connect(self._recompute_matches)
        row1.addWidget(self.spin_hrs)

        self.spin_mins = QSpinBox(self.timing_panel)
        self.spin_mins.setRange(0, 59)
        self.spin_mins.setSuffix("m")
        self.spin_mins.valueChanged.connect(self._recompute_matches)
        row1.addWidget(self.spin_mins)

        self.spin_secs = QSpinBox(self.timing_panel)
        self.spin_secs.setRange(0, 59)
        self.spin_secs.setSuffix("s")
        self.spin_secs.valueChanged.connect(self._recompute_matches)
        row1.addWidget(self.spin_secs)

        self.btn_calibrate = QPushButton("Sync Clock Using 2 Photos...", self.timing_panel)
        self.btn_calibrate.setToolTip("Pick a camera photo and a phone photo taken at the same moment.")
        self.btn_calibrate.clicked.connect(self._open_calibrate_dialog)
        row1.addWidget(self.btn_calibrate)
        row1.addStretch(1)
        timing_vbox.addLayout(row1)

        row2 = QHBoxLayout()
        self.chk_fix_timestamps = QCheckBox(
            "Also fix the capture time saved on matched camera photos", self.timing_panel
        )
        self.chk_fix_timestamps.setEnabled(False)
        row2.addWidget(self.chk_fix_timestamps)
        row2.addStretch(1)
        self.lbl_gpx_info = QLabel("", self.timing_panel)
        set_ui_role(self.lbl_gpx_info, "muted")
        row2.addWidget(self.lbl_gpx_info)
        timing_vbox.addLayout(row2)

        self.timing_panel.setVisible(False)
        scope_vbox.addWidget(self.timing_panel)
        root.addWidget(scope_frame)

        # 3. Side-by-Side Source -> Destination Match Table
        self.table = QTableWidget(0, 6, self)
        self.table.setHorizontalHeaderLabels([
            "Apply",
            "Destination Photo (Needs GPS)",
            "",
            "GPS Source (Matched From)",
            "Location to Apply",
            "Match Quality",
        ])
        self.table.setIconSize(QSize(96, 96))
        self.table.verticalHeader().setDefaultSectionSize(108)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.setMouseTracking(True)
        self.table.viewport().setMouseTracking(True)
        self.table.cellEntered.connect(self._on_table_cell_entered)
        self.table.viewport().installEventFilter(self)
        self._hover_popup = _HoverPhotoPreviewPopup(self)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.Stretch)
        hdr.setSectionResizeMode(4, QHeaderView.Stretch)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.itemChanged.connect(self._on_table_item_changed)
        root.addWidget(self.table, 1)

        # 4. Empty / Guidance Label (shown when 0 matches)
        self.lbl_empty_hint = QLabel("", self)
        self.lbl_empty_hint.setWordWrap(True)
        self.lbl_empty_hint.setAlignment(Qt.AlignCenter)
        set_ui_role(self.lbl_empty_hint, "empty_hint")
        self.lbl_empty_hint.setVisible(False)
        root.addWidget(self.lbl_empty_hint)

        # 5. Footer
        footer = QHBoxLayout()
        self.btn_sel_all = QPushButton("Select All", self)
        self.btn_sel_all.clicked.connect(lambda: self._set_all_checked(True))
        footer.addWidget(self.btn_sel_all)

        self.btn_sel_none = QPushButton("Select None", self)
        self.btn_sel_none.clicked.connect(lambda: self._set_all_checked(False))
        footer.addWidget(self.btn_sel_none)

        self.lbl_summary = QLabel("0 matches", self)
        self.lbl_summary.setStyleSheet("font-weight: 600; margin-left: 8px;")
        footer.addWidget(self.lbl_summary)
        footer.addStretch(1)

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.clicked.connect(self.reject)
        footer.addWidget(self.btn_cancel)

        self.btn_apply = QPushButton("Apply Locations to 0 Photos", self)
        self.btn_apply.setDefault(True)
        set_ui_role(self.btn_apply, "primary_btn")
        self.btn_apply.clicked.connect(self.accept)
        footer.addWidget(self.btn_apply)
        root.addLayout(footer)

        self._refresh_scope_counts()
        self._recompute_matches()

    def _on_timing_toggled(self, checked: bool) -> None:
        self.timing_panel.setVisible(bool(checked))

    def _on_scope_changed(self) -> None:
        if self._scope_loader is None:
            return
        scope_key = str(self.combo_scope.currentData() or "")
        if not scope_key:
            return
        targets, donors, scope_label = self._scope_loader(scope_key)
        self._targets = list(targets)
        self._base_donors = list(donors)
        self._donors = [*self._base_donors, *self._gpx_donors]
        self._scope_label = scope_label
        for t in self._targets:
            if t.path:
                self._path_map[int(t.media_id)] = str(t.path)
        for d in self._donors:
            if d.media_id is not None and d.path:
                self._path_map[int(d.media_id)] = str(d.path)
        self._refresh_scope_counts()
        self._recompute_matches()

    def _refresh_scope_counts(self) -> None:
        gpx_note = f" + {len(self._gpx_donors)} GPX points" if self._gpx_donors else ""
        self.lbl_scope_counts.setText(
            f"<b>{len(self._targets)}</b> without GPS  •  <b>{len(self._base_donors)}</b> GPS source photo(s){gpx_note}"
        )

    def clock_offset_seconds(self) -> int:
        sign = int(self.combo_sign.currentData() or 1)
        secs = (
            int(self.spin_hrs.value()) * 3600
            + int(self.spin_mins.value()) * 60
            + int(self.spin_secs.value())
        )
        return sign * secs

    def set_clock_offset_seconds(self, offset_sec: int) -> None:
        sign = 1 if offset_sec >= 0 else -1
        rem = abs(int(offset_sec))
        hrs = min(23, rem // 3600)
        rem %= 3600
        mins = rem // 60
        secs = rem % 60
        for w in (self.combo_sign, self.spin_hrs, self.spin_mins, self.spin_secs):
            w.blockSignals(True)
        self.combo_sign.setCurrentIndex(0 if sign > 0 else 1)
        self.spin_hrs.setValue(hrs)
        self.spin_mins.setValue(mins)
        self.spin_secs.setValue(secs)
        for w in (self.combo_sign, self.spin_hrs, self.spin_mins, self.spin_secs):
            w.blockSignals(False)
        self._recompute_matches()

    def selected_matches(self) -> list[ProposedMatch]:
        return [m for m in self._matches if m.checked]

    def should_shift_timestamps(self) -> bool:
        return bool(self.chk_fix_timestamps.isChecked() and self.clock_offset_seconds() != 0)

    def _open_calibrate_dialog(self) -> None:
        dlg = _CalibrateClockDialog(self._targets, self._donors, parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.set_clock_offset_seconds(dlg.computed_offset_sec)

    def _import_gpx(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load GPX Track File(s)",
            "",
            "GPX Tracks (*.gpx);;All Files (*.*)",
        )
        if not paths:
            return
        added = 0
        for p in paths:
            try:
                pts = parse_gpx_file(p)
                if pts:
                    self._gpx_donors.extend(pts)
                    self._gpx_files_loaded.append(p)
                    added += len(pts)
            except Exception:
                pass
        if added > 0:
            self._donors = [*self._base_donors, *self._gpx_donors]
            self.lbl_gpx_info.setText(f"+{len(self._gpx_donors)} GPX trackpoint(s) loaded")
            self._refresh_scope_counts()
            self._recompute_matches()

    def _recompute_matches(self) -> None:
        offset = self.clock_offset_seconds()
        self.chk_fix_timestamps.setEnabled(offset != 0)
        if offset == 0:
            self.chk_fix_timestamps.setChecked(False)

        max_gap = int(self.combo_max_gap.currentData() or 7200)
        self._matches = match_timed_photos(
            self._targets,
            self._donors,
            clock_offset_sec=offset,
            max_gap_sec=max_gap,
        )
        resolve_missing_place_names(self._matches, db_engine=self._db_engine)
        self._populate_table()

    def _load_large_pixmap(self, media_id: int | None, fallback_path: str = "") -> QPixmap:
        if media_id is None:
            return QPixmap()
        mid = int(media_id)
        cached = self._large_pixmap_cache.get(mid)
        if cached is not None:
            return cached
        pix = QPixmap()
        fpath = fallback_path or self._path_map.get(mid, "")
        if fpath:
            try:
                reader = QImageReader(fpath)
                reader.setAutoTransform(True)
                img_sz = reader.size()
                if img_sz.isValid():
                    reader.setScaledSize(img_sz.scaled(260, 195, Qt.KeepAspectRatio))
                qimg = reader.read()
                if not qimg.isNull():
                    pix = QPixmap.fromImage(qimg)
            except Exception:
                pass
        if pix.isNull() and self._thumb_repo is not None:
            try:
                data = None
                if hasattr(self._thumb_repo, "get_webp_blob"):
                    data = self._thumb_repo.get_webp_blob(mid)
                elif hasattr(self._thumb_repo, "get_thumbnail_bytes"):
                    data = self._thumb_repo.get_thumbnail_bytes(mid)
                if data:
                    if data.startswith(b"PG64THMB"):
                        data = data[8:]
                    pix.loadFromData(data)
            except Exception:
                pass
        self._large_pixmap_cache[mid] = pix
        return pix

    def _load_thumb_icon(self, media_id: int | None, fallback_path: str = "") -> QIcon:
        if media_id is None:
            return QIcon()
        mid = int(media_id)
        cached = self._thumb_icon_cache.get(mid)
        if cached is not None:
            return cached
        icon = QIcon()
        if self._thumb_repo is not None:
            try:
                data = None
                if hasattr(self._thumb_repo, "get_webp_blob"):
                    data = self._thumb_repo.get_webp_blob(mid)
                elif hasattr(self._thumb_repo, "get_thumbnail_bytes"):
                    data = self._thumb_repo.get_thumbnail_bytes(mid)
                if data:
                    if data.startswith(b"PG64THMB"):
                        data = data[8:]
                    pix = QPixmap()
                    if pix.loadFromData(data):
                        icon = QIcon(pix.scaled(96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            except Exception:
                pass
        if icon.isNull():
            fpath = fallback_path or self._path_map.get(mid, "")
            if fpath:
                try:
                    reader = QImageReader(fpath)
                    reader.setAutoTransform(True)
                    img_sz = reader.size()
                    if img_sz.isValid():
                        reader.setScaledSize(img_sz.scaled(96, 96, Qt.KeepAspectRatio))
                    qimg = reader.read()
                    if not qimg.isNull():
                        icon = QIcon(QPixmap.fromImage(qimg))
                except Exception:
                    pass
        self._thumb_icon_cache[mid] = icon
        return icon

    def eventFilter(self, watched: Any, event: QEvent) -> bool:
        if watched is self.table.viewport() and event.type() in (QEvent.Leave, QEvent.HoverLeave):
            self._hover_popup.hide()
        return super().eventFilter(watched, event)

    def _on_table_cell_entered(self, row: int, col: int) -> None:
        if col not in (1, 3):
            self._hover_popup.hide()
            return
        chk = self.table.item(row, 0)
        match_idx = chk.data(Qt.UserRole) if chk is not None else row
        if not isinstance(match_idx, int) or not (0 <= match_idx < len(self._matches)):
            self._hover_popup.hide()
            return
        m = self._matches[match_idx]
        if col == 1:
            pix = self._load_large_pixmap(m.media_id, m.path)
            loc_str = m.location_name or f"{m.lat:.4f}, {m.lon:.4f}"
            self._hover_popup.show_preview(
                role="Destination Photo (Will Receive Location)",
                pixmap=pix,
                filename=m.filename,
                camera=m.camera_model,
                capture_time=_fmt_ts(m.target_ts),
                location=f"Will be set to: {loc_str}",
                global_pos=QCursor.pos(),
            )
        else:
            if m.donor_source == "gpx":
                if self._gpx_icon is None:
                    self._gpx_icon = _make_gpx_icon(96)
                pix = self._gpx_icon.pixmap(96, 96)
                loc_str = m.donor_location_name or m.location_name or f"{m.lat:.4f}, {m.lon:.4f}"
                self._hover_popup.show_preview(
                    role="GPS Source (GPX Track)",
                    pixmap=pix,
                    filename=m.donor_filename or "GPX Track",
                    camera="GPX Track",
                    capture_time=_fmt_ts(m.donor_timestamp) if m.donor_timestamp else "",
                    location=loc_str,
                    global_pos=QCursor.pos(),
                )
            else:
                pix = self._load_large_pixmap(m.donor_media_id, m.donor_path)
                loc_str = m.donor_location_name or (
                    f"{m.donor_lat:.4f}, {m.donor_lon:.4f}"
                    if m.donor_lat is not None and m.donor_lon is not None
                    else (m.location_name or "")
                )
                self._hover_popup.show_preview(
                    role="GPS Source Photo",
                    pixmap=pix,
                    filename=m.donor_filename or "GPS-tagged photo",
                    camera=m.donor_camera_model,
                    capture_time=_fmt_ts(m.donor_timestamp) if m.donor_timestamp else "",
                    location=loc_str,
                    global_pos=QCursor.pos(),
                )

    def _populate_table(self) -> None:
        self.table.setSortingEnabled(False)
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._matches))

        if self._thumb_repo is not None and hasattr(self._thumb_repo, "get_webp_blobs_batch"):
            needed_ids: list[int] = []
            for m in self._matches:
                if int(m.media_id) not in self._thumb_icon_cache:
                    needed_ids.append(int(m.media_id))
                if m.donor_media_id is not None and int(m.donor_media_id) not in self._thumb_icon_cache:
                    needed_ids.append(int(m.donor_media_id))
            if needed_ids:
                try:
                    blobs = self._thumb_repo.get_webp_blobs_batch(list(dict.fromkeys(needed_ids)))
                    for mid_i, blob in blobs.items():
                        raw = blob[8:] if blob.startswith(b"PG64THMB") else blob
                        pix = QPixmap()
                        if pix.loadFromData(raw):
                            self._thumb_icon_cache[mid_i] = QIcon(
                                pix.scaled(96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            )
                except Exception:
                    pass

        conf_colors = {
            "exact": QColor("#15803D"),
            "nearby": QColor("#1D4ED8"),
            "interpolated": QColor("#B45309"),
        }
        for row, m in enumerate(self._matches):
            # Col 0: Apply Checkbox
            chk_item = _SortableTableItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            chk_item.setCheckState(Qt.Checked if m.checked else Qt.Unchecked)
            chk_item.setData(Qt.UserRole, row)
            chk_item.setData(_SortableTableItem.SORT_ROLE, 0 if m.checked else 1)
            self.table.setItem(row, 0, chk_item)

            # Col 1: Destination Photo (Thumbnail + Filename + Camera + Time)
            cam_parts = [p for p in (m.camera_model, _fmt_short_ts(m.target_ts)) if p]
            dest_txt = f"{m.filename}\n{' • '.join(cam_parts)}"
            dest_item = _SortableTableItem(dest_txt)
            dest_item.setData(_SortableTableItem.SORT_ROLE, (m.filename.lower(), int(m.target_ts)))
            dest_icon = self._load_thumb_icon(m.media_id, m.path)
            if not dest_icon.isNull():
                dest_item.setIcon(dest_icon)
            self.table.setItem(row, 1, dest_item)

            # Col 2: Arrow
            arrow_item = _SortableTableItem("←")
            arrow_item.setTextAlignment(Qt.AlignCenter)
            arrow_item.setToolTip("Receives GPS location from")
            self.table.setItem(row, 2, arrow_item)

            # Col 3: GPS Source (Phone Photo Thumbnail or GPX Badge + Name + Time)
            if m.donor_source == "gpx":
                if self._gpx_icon is None:
                    self._gpx_icon = _make_gpx_icon(96)
                src_icon = self._gpx_icon
                src_name = m.donor_filename or "GPX Track"
                src_sub = f"GPX Track • {_fmt_short_ts(m.donor_timestamp)}" if m.donor_timestamp else "GPX Track"
            else:
                src_icon = self._load_thumb_icon(m.donor_media_id, m.donor_path)
                src_name = m.donor_filename or "GPS-tagged photo"
                src_parts = [p for p in (m.donor_camera_model, _fmt_short_ts(m.donor_timestamp)) if p]
                src_sub = " • ".join(src_parts)
            src_item = _SortableTableItem(f"{src_name}\n{src_sub}" if src_sub else src_name)
            src_item.setData(_SortableTableItem.SORT_ROLE, (src_name.lower(), int(m.donor_timestamp or 0)))
            if not src_icon.isNull():
                src_item.setIcon(src_icon)
            self.table.setItem(row, 3, src_item)

            # Col 4: Matched Location
            place = m.location_name or f"{m.lat:.4f}, {m.lon:.4f}"
            loc_item = _SortableTableItem(place)
            loc_item.setData(_SortableTableItem.SORT_ROLE, place.lower())
            self.table.setItem(row, 4, loc_item)

            # Col 5: Match Quality
            gap_str = format_time_gap(m.time_gap_sec)
            if m.confidence == "exact":
                qual_txt = f"Close match\n{gap_str} apart"
            elif m.confidence == "nearby":
                qual_txt = f"Nearby\n{gap_str} apart"
            else:
                qual_txt = f"Estimated on route\n{gap_str} apart"
            qual_item = _SortableTableItem(qual_txt)
            qual_item.setData(_SortableTableItem.SORT_ROLE, int(m.time_gap_sec))
            qual_item.setForeground(conf_colors.get(m.confidence, QColor("#334155")))
            self.table.setItem(row, 5, qual_item)

        self.table.blockSignals(False)
        self.table.setSortingEnabled(True)
        self._update_summary()

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        match_idx = item.data(Qt.UserRole)
        if not isinstance(match_idx, int):
            match_idx = item.row()
        if 0 <= match_idx < len(self._matches):
            checked = item.checkState() == Qt.Checked
            self._matches[match_idx].checked = checked
            item.setData(_SortableTableItem.SORT_ROLE, 0 if checked else 1)
            self._update_summary()

    def _set_all_checked(self, checked: bool) -> None:
        self.table.blockSignals(True)
        state = Qt.Checked if checked else Qt.Unchecked
        for m in self._matches:
            m.checked = checked
        for row in range(self.table.rowCount()):
            it = self.table.item(row, 0)
            if it is not None:
                it.setCheckState(state)
                it.setData(_SortableTableItem.SORT_ROLE, 0 if checked else 1)
        self.table.blockSignals(False)
        self._update_summary()

    def _update_summary(self) -> None:
        total = len(self._matches)
        sel = sum(1 for m in self._matches if m.checked)
        unmatched = max(0, len(self._targets) - total)
        gap_txt = self.combo_max_gap.currentText() if hasattr(self, "combo_max_gap") else "2 hours"

        if total == 0:
            self.table.setVisible(False)
            self.lbl_empty_hint.setVisible(True)
            if not self._targets:
                self.lbl_empty_hint.setText(
                    "No photos without GPS were found in this scope.\n\n"
                    "Try switching 'Photos to scan' above to 'Current Folder / View', or open a folder containing untagged camera photos."
                )
            elif not self._donors:
                self.lbl_empty_hint.setText(
                    f"Found {len(self._targets)} photo(s) without GPS, but no GPS-tagged source photos were found for the same dates.\n\n"
                    "Make sure your phone photos for those dates are in the catalog, or click 'Load GPX Track...' above."
                )
            else:
                self.lbl_empty_hint.setText(
                    f"No matches found within {gap_txt}.\n\n"
                    "Click 'Clock & Time Window...' above to increase 'Match photos within' or sync a wrong camera clock."
                )
            self.lbl_summary.setText("0 matches")
        else:
            self.table.setVisible(True)
            self.lbl_empty_hint.setVisible(False)
            if unmatched > 0:
                self.lbl_summary.setText(
                    f"{sel} of {total} matched selected  ({unmatched} had no GPS match within {gap_txt})"
                )
            else:
                self.lbl_summary.setText(f"{sel} of {total} matched photo(s) selected")

        self.btn_apply.setText(f"Apply Locations to {sel} Photo{'s' if sel != 1 else ''}")
        self.btn_apply.setEnabled(sel > 0)
