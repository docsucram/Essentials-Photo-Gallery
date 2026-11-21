"""
Modal Dialogs for PhotoGallery64:
1. ChangeTimeTakenDialog — Exact visual and mathematical clone of WLPG's 'Change Time Taken'
   dialog (docs/reference_ui/adjust_time.png) calculating relative Δt = T_lead_new - T_lead_original.
2. AppSettingsDialog — WLPG Options & Database Maintenance dialog (docs/reference_ui/database.png)
   with 30-day '.OriginalFiles' backup retention policy, theme selector, and SQLite compaction.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageOps
from PySide6.QtCore import QDate, QObject, QRect, QSettings, QThread, QTime, Qt, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QIntValidator,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from src.db.repository import _EPOCH_UTC, _epoch_to_datetime_utc
from src.ui.ribbon_toolbar import make_wlpg_icon


def _format_delta_human(delta_sec: int, selected_count: int) -> str:
    if delta_sec == 0:
        return "ⓘ  No time adjustment has been made."
    sign = "+" if delta_sec > 0 else "-"
    abs_sec = abs(int(delta_sec))
    days, rem = divmod(abs_sec, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    file_word = "file" if selected_count == 1 else "files"
    if days > 0:
        day_word = "day" if days == 1 else "days"
        return (
            f"ⓘ  Adjustment: {sign}{days} {day_word}, "
            f"{hours:02d}:{minutes:02d}:{seconds:02d} applied to {selected_count} {file_word}."
        )
    return (
        f"ⓘ  Adjustment: {sign}{hours:02d}:{minutes:02d}:{seconds:02d} "
        f"applied to {selected_count} {file_word}."
    )


class ChangeTimeTakenDialog(QDialog):
    """
    Implements Windows Live Photo Gallery's 'Change Time Taken' dialog (`docs/reference_ui/adjust_time.png`).
    Computes Δt = T_lead_new - T_lead_original and applies that exact relative shift across all selected items.
    Supports pre-1970 timestamps on Windows.
    """

    def __init__(
        self,
        lead_filename: str,
        lead_date_taken: int,
        selected_count: int = 1,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Change Time Taken")
        self.setFixedWidth(430)

        self.lead_filename = lead_filename
        self.lead_date_taken = int(lead_date_taken)
        self.selected_count = max(1, int(selected_count))

        orig_dt = _epoch_to_datetime_utc(self.lead_date_taken)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(16, 16, 16, 14)
        vbox.setSpacing(12)

        # 1. Instructional Header (Blue Segoe UI Title + Explanation)
        lbl_header = QLabel(
            "Change the time the photo or video was taken. If multiple files "
            "are selected, the time for all of the files will be adjusted by "
            "the same amount.",
            self,
        )
        lbl_header.setWordWrap(True)
        lbl_header.setFont(QFont("Segoe UI", 10))
        lbl_header.setStyleSheet("color: #1E395B;")
        vbox.addWidget(lbl_header)

        # 2. Live Status Box ("ⓘ No time adjustment has been made.")
        self.status_frame = QFrame(self)
        self.status_frame.setStyleSheet(
            """
            QFrame {
                background-color: #F4F8FC;
                border: 1px solid #C6DDF0;
                border-radius: 3px;
            }
            QLabel {
                border: none;
                background: transparent;
                color: #333333;
            }
            """
        )
        status_layout = QHBoxLayout(self.status_frame)
        status_layout.setContentsMargins(8, 6, 8, 6)
        self.lbl_status = QLabel(
            _format_delta_human(0, self.selected_count), self.status_frame
        )
        status_layout.addWidget(self.lbl_status)
        vbox.addWidget(self.status_frame)

        # 3. Side-by-side QTimeEdit (HH:mm:ss) and QDateEdit (d MMMM yyyy)
        pickers_row = QHBoxLayout()
        pickers_row.setSpacing(8)

        self.time_edit = QTimeEdit(self)
        self.time_edit.setDisplayFormat("HH:mm:ss")
        self.time_edit.setTime(QTime(orig_dt.hour, orig_dt.minute, orig_dt.second))
        self.time_edit.setFixedWidth(115)
        self.time_edit.timeChanged.connect(lambda _: self._on_picker_changed())
        pickers_row.addWidget(self.time_edit)

        self.date_edit = QDateEdit(self)
        self.date_edit.setDisplayFormat("d MMMM yyyy")
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setMinimumDate(QDate(1753, 1, 1))
        self.date_edit.setMaximumDate(QDate(2200, 12, 31))
        self.date_edit.setDate(QDate(orig_dt.year, orig_dt.month, orig_dt.day))
        self.date_edit.dateChanged.connect(lambda _: self._on_picker_changed())
        pickers_row.addWidget(self.date_edit, 1)

        vbox.addLayout(pickers_row)

        # 4. Current (original) time of the lead photo
        orig_str = orig_dt.strftime("%d/%m/%Y %H:%M:%S")
        self.lbl_current_time = QLabel(
            f"Current time:   {orig_str}  ({self.lead_filename})", self
        )
        self.lbl_current_time.setStyleSheet("color: #444444;")
        vbox.addWidget(self.lbl_current_time)

        vbox.addSpacing(4)

        # 5. OK / Cancel Dialog Button Box
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        vbox.addWidget(self.button_box)

    def _on_picker_changed(self) -> None:
        delta = self.delta_seconds()
        self.lbl_status.setText(_format_delta_human(delta, self.selected_count))

    def set_new_datetime(
        self,
        year: int,
        month: int,
        day: int,
        hour: int,
        minute: int,
        second: int = 0,
    ) -> None:
        """Programmatically sets the date and time pickers and refreshes the live Δt preview."""
        self.date_edit.setDate(QDate(year, month, day))
        self.time_edit.setTime(QTime(hour, minute, second))
        self._on_picker_changed()

    def new_lead_timestamp(self) -> int:
        """Returns the newly chosen UTC Unix epoch timestamp (supports negative pre-1970 dates)."""
        qd = self.date_edit.date()
        qt = self.time_edit.time()
        dt = datetime.datetime(
            qd.year(),
            qd.month(),
            qd.day(),
            qt.hour(),
            qt.minute(),
            qt.second(),
            tzinfo=datetime.timezone.utc,
        )
        return int((dt - _EPOCH_UTC).total_seconds())

    def delta_seconds(self) -> int:
        """Returns Δt = T_lead_new - T_lead_original in seconds."""
        return self.new_lead_timestamp() - self.lead_date_taken


class TagMergeConfirmDialog(QDialog):
    """
    WLPG-style confirmation dialog when moving a tag into a parent where a tag
    with the same name already exists (media_1793525510394_ecf2beb5.png).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Photo Gallery")
        self.setFixedSize(410, 160)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #FFFFFF;
            }
            QPushButton {
                min-width: 75px;
                height: 23px;
                font-family: 'Segoe UI';
                font-size: 9pt;
                border: 1px solid #707070;
                border-radius: 2px;
                background-color: #E1E1E1;
            }
            QPushButton:hover {
                background-color: #E5F1FB;
                border: 1px solid #0078D7;
            }
            QPushButton:pressed {
                background-color: #CCE4F7;
                border: 1px solid #005499;
            }
            QPushButton:focus {
                border: 1px solid #0078D7;
            }
            QCheckBox {
                font-family: 'Segoe UI';
                font-size: 9pt;
                color: #000000;
            }
            """
        )

        main_vbox = QVBoxLayout(self)
        main_vbox.setContentsMargins(18, 16, 18, 14)
        main_vbox.setSpacing(14)

        # Top area: Icon + text
        top_hbox = QHBoxLayout()
        top_hbox.setSpacing(14)
        top_hbox.setAlignment(Qt.AlignTop)

        lbl_icon = QLabel(self)
        lbl_icon.setFixedSize(36, 36)
        lbl_icon.setPixmap(self._make_question_pixmap(36))
        top_hbox.addWidget(lbl_icon, 0, Qt.AlignTop)

        text_vbox = QVBoxLayout()
        text_vbox.setSpacing(6)

        self.lbl_title = QLabel("A tag with the same name already exists here", self)
        self.lbl_title.setStyleSheet("color: #003399; font-family: 'Segoe UI'; font-size: 11pt; font-weight: normal;")
        self.lbl_title.setWordWrap(True)
        text_vbox.addWidget(self.lbl_title)

        self.lbl_msg = QLabel("These tags are about to be merged. Do you want to continue?", self)
        self.lbl_msg.setStyleSheet("color: #000000; font-family: 'Segoe UI'; font-size: 9pt;")
        self.lbl_msg.setWordWrap(True)
        text_vbox.addWidget(self.lbl_msg)

        top_hbox.addLayout(text_vbox, 1)
        main_vbox.addLayout(top_hbox)

        main_vbox.addStretch(1)

        # Bottom row: Checkbox on left, Yes/No buttons on right
        bottom_hbox = QHBoxLayout()
        self.chk_dont_show = QCheckBox("Don't show this message again", self)
        bottom_hbox.addWidget(self.chk_dont_show, 0, Qt.AlignVCenter)

        bottom_hbox.addStretch(1)

        self.btn_yes = QPushButton("Yes", self)
        self.btn_yes.setDefault(True)
        self.btn_yes.clicked.connect(self.accept)
        bottom_hbox.addWidget(self.btn_yes)

        self.btn_no = QPushButton("No", self)
        self.btn_no.clicked.connect(self.reject)
        bottom_hbox.addWidget(self.btn_no)

        main_vbox.addLayout(bottom_hbox)

    @staticmethod
    def _make_question_pixmap(size: int = 36) -> QPixmap:
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing, True)
        grad = QLinearGradient(0, 0, 0, size)
        grad.setColorAt(0, QColor("#3C88E8"))
        grad.setColorAt(1, QColor("#1452B5"))
        p.setBrush(grad)
        p.setPen(QPen(QColor("#0F3D87"), 1.2))
        p.drawEllipse(1, 1, size - 2, size - 2)
        p.setPen(QColor("#FFFFFF"))
        font = QFont("Segoe UI", int(size * 0.58), QFont.Bold)
        p.setFont(font)
        p.drawText(QRect(0, 0, size, size - 2), Qt.AlignCenter, "?")
        p.end()
        return pix

    @property
    def dont_show_again(self) -> bool:
        return self.chk_dont_show.isChecked()


class AppSettingsDialog(QDialog):
    """
    Windows Live Photo Gallery Options Dialog with multi-tab interface:
    - General: Tooltips, Face Detection & Recognition, Navigation Pane visibility, Visual theme
    - Import: Device settings, destination folders, naming rules, and auto-rotation
    - Originals: Original photos backup retention and Recycle Bin threshold, directory link
    - Edit: Auto-adjust options (Straighten, Noise, Color, Exposure) and JPEG quality slider
    - Database: Database & Backup Maintenance (compaction, expired original cleanup)
    """

    settings_applied = Signal()
    picasa_import_requested = Signal()

    def __init__(
        self,
        retention_days: int = 180,
        current_theme: Literal["wlpg_aero", "modern", "dark"] = "wlpg_aero",
        show_dates: bool = True,
        show_tags: bool = True,
        show_people: bool = True,
        show_ratings: bool = True,
        show_places: bool = True,
        show_albums: bool = True,
        favourites_enabled: bool = True,
        find_people_enabled: bool = False,
        tooltip_previews: bool = True,
        jpeg_quality: int = 83,
        auto_straighten: bool = True,
        auto_noise: bool = False,
        auto_color: bool = True,
        auto_exposure: bool = True,
        save_copy_location: str | None = None,
        save_copy_custom_folder: str | None = None,
        save_copy_naming: str | None = None,
        save_copy_switch_view: bool | None = None,
        originals_dir: str | None = None,
        pause_write_back: bool = True,
        icon_theme: str = "wlpg_aero",
        engine: Any | None = None,
        initial_tab: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Photo Gallery Options")
        self.engine = engine
        self._is_dark: bool = current_theme == "dark"
        self.resize(600, 550)
        self.setMinimumSize(560, 500)

        self.compact_requested: bool = False
        self.rebuild_thumbs_requested: bool = False
        self.prune_backups_requested: bool = False
        self.backup_requested: bool = False
        self.fresh_start_requested: bool = False
        self.heal_faces_requested: bool = False
        self.purge_rescan_requested: bool = False
        self.fast_rematch_requested: bool = False
        self.rescan_ocr_requested: bool = False
        self.originals_dir: str = originals_dir or str(Path.home() / "Pictures" / ".OriginalFiles")

        # Load Save as copy settings from QSettings if not explicitly passed
        settings = QSettings("PhotoGallery64", "PhotoGallery64")
        self._init_save_copy_location = save_copy_location if save_copy_location is not None else str(settings.value("save_copy_location", "original"))
        self._init_save_copy_custom_folder = save_copy_custom_folder if save_copy_custom_folder is not None else str(settings.value("save_copy_custom_folder", "") or "")
        self._init_save_copy_naming = save_copy_naming if save_copy_naming is not None else str(settings.value("save_copy_naming_pattern", "{stem}_edited"))
        self._init_save_copy_switch_view = save_copy_switch_view if save_copy_switch_view is not None else settings.value("save_copy_switch_view", True, type=bool)
        self._init_unsaved_edits_action = str(settings.value("unsaved_edits_action", "prompt"))

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Tab Widget
        self.tab_widget = QTabWidget(self)
        self.tabs = self.tab_widget  # Compatibility alias
        self.tab_widget.setFont(QFont("Segoe UI", 9))
        self._apply_dialog_theme(self._is_dark)

        # -------------------------------------------------------------
        # Tab 1: General (docs: screenshot 2)
        # -------------------------------------------------------------
        tab_general = QWidget()
        v_gen = QVBoxLayout(tab_general)
        v_gen.setContentsMargins(14, 14, 14, 14)
        v_gen.setSpacing(12)

        # Face detection and recognition
        lbl_face = QLabel("Face detection and recognition", tab_general)
        lbl_face.setFont(QFont("Segoe UI", 9, QFont.Bold))
        v_gen.addWidget(lbl_face)

        self.chk_find_people = QCheckBox("Find people in your photos.", tab_general)
        self.chk_find_people.setChecked(find_people_enabled)
        v_gen.addWidget(self.chk_find_people)

        self.btn_configure_face = QPushButton("⚙ Configure Face Recognition & Tuning Options...", tab_general)
        self.btn_configure_face.setFixedWidth(290)
        self.btn_configure_face.clicked.connect(lambda: self.tab_widget.setCurrentWidget(self.face_tab))
        v_gen.addWidget(self.btn_configure_face)

        sep2 = QFrame(tab_general)
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        v_gen.addWidget(sep2)

        # Navigation pane
        lbl_nav = QLabel("Navigation pane", tab_general)
        lbl_nav.setFont(QFont("Segoe UI", 9, QFont.Bold))
        v_gen.addWidget(lbl_nav)

        nav_grid = QGridLayout()
        nav_grid.setHorizontalSpacing(24)
        nav_grid.setVerticalSpacing(8)

        self.chk_show_dates = QCheckBox("Show Date taken", tab_general)
        self.chk_show_dates.setChecked(show_dates)
        nav_grid.addWidget(self.chk_show_dates, 0, 0)

        self.chk_show_tags = QCheckBox("Show Descriptive tags", tab_general)
        self.chk_show_tags.setChecked(show_tags)
        nav_grid.addWidget(self.chk_show_tags, 0, 1)

        self.chk_show_people = QCheckBox("Show People", tab_general)
        self.chk_show_people.setChecked(show_people)
        nav_grid.addWidget(self.chk_show_people, 1, 0)

        self.chk_show_ratings = QCheckBox("Show Ratings", tab_general)
        self.chk_show_ratings.setChecked(show_ratings)
        nav_grid.addWidget(self.chk_show_ratings, 1, 1)

        self.chk_show_places = QCheckBox("Show Places", tab_general)
        self.chk_show_places.setChecked(show_places)
        nav_grid.addWidget(self.chk_show_places, 2, 0)

        self.chk_show_albums = QCheckBox("Show Albums", tab_general)
        self.chk_show_albums.setChecked(show_albums)
        nav_grid.addWidget(self.chk_show_albums, 2, 1)

        v_gen.addLayout(nav_grid)

        self.chk_favourites_enabled = QCheckBox(
            "Enable Favourites (heart icon, Favourites smart album, Group by & filters)",
            tab_general,
        )
        self.chk_favourites_enabled.setChecked(bool(favourites_enabled))
        v_gen.addWidget(self.chk_favourites_enabled)

        sep3 = QFrame(tab_general)
        sep3.setFrameShape(QFrame.HLine)
        sep3.setFrameShadow(QFrame.Sunken)
        v_gen.addWidget(sep3)

        # Visual Theme selector
        h_theme = QHBoxLayout()
        lbl_theme = QLabel("Visual theme:", tab_general)
        lbl_theme.setFixedWidth(85)
        self.cmb_theme = QComboBox(tab_general)
        self.cmb_theme.addItem("Classic Aero", "wlpg_aero")
        self.cmb_theme.addItem("Modern Fluent", "modern")
        self.cmb_theme.addItem("Dark Studio", "dark")
        t_idx = self.cmb_theme.findData(current_theme)
        if t_idx >= 0:
            self.cmb_theme.setCurrentIndex(t_idx)
        h_theme.addWidget(lbl_theme)
        h_theme.addWidget(self.cmb_theme, 1)
        v_gen.addLayout(h_theme)

        # Icon Theme selector
        h_icon_theme = QHBoxLayout()
        lbl_icon_theme = QLabel("Icon style:", tab_general)
        lbl_icon_theme.setFixedWidth(85)
        self.cmb_icon_theme = QComboBox(tab_general)
        self.cmb_icon_theme.addItem("Classic Aero (Colourful)", "wlpg_aero")
        self.cmb_icon_theme.addItem("Microsoft Fluent UI (Windows 11)", "fluent")
        self.cmb_icon_theme.addItem("Modern Minimalist", "minimal")
        i_idx = self.cmb_icon_theme.findData(icon_theme)
        if i_idx >= 0:
            self.cmb_icon_theme.setCurrentIndex(i_idx)
        h_icon_theme.addWidget(lbl_icon_theme)
        h_icon_theme.addWidget(self.cmb_icon_theme, 1)
        v_gen.addLayout(h_icon_theme)

        def _on_theme_selection_changed(idx: int) -> None:
            theme_val = str(self.cmb_theme.itemData(idx) or "")
            target_icon = "fluent" if theme_val == "modern" else "wlpg_aero" if theme_val == "wlpg_aero" else None
            if target_icon:
                match_idx = self.cmb_icon_theme.findData(target_icon)
                if match_idx >= 0:
                    self.cmb_icon_theme.setCurrentIndex(match_idx)

        self.cmb_theme.currentIndexChanged.connect(_on_theme_selection_changed)

        sep4 = QFrame(tab_general)
        sep4.setFrameShape(QFrame.HLine)
        sep4.setFrameShadow(QFrame.Sunken)
        v_gen.addWidget(sep4)

        # File Metadata Writing
        lbl_safe = QLabel("File Metadata", tab_general)
        lbl_safe.setFont(QFont("Segoe UI", 9, QFont.Bold))
        v_gen.addWidget(lbl_safe)

        self.chk_write_to_files = QCheckBox(
            "Write metadata changes to original files", tab_general
        )
        self.chk_write_to_files.setChecked(not pause_write_back)
        self.chk_write_to_files.setToolTip(
            "When checked, tags, ratings, captions, and face coordinates are written back to\n"
            "the original image files (EXIF/XMP). When unchecked, changes are saved only to the\n"
            "PhotoGallery64 catalog database without modifying original files on disk."
        )
        v_gen.addWidget(self.chk_write_to_files)
        self.chk_pause_write_back = self.chk_write_to_files

        lbl_safe_hint = QLabel(
            "When unchecked (default), tags, ratings, captions, and people are safely saved to the "
            "gallery database without modifying your original image files on disk.",
            tab_general,
        )
        lbl_safe_hint.setWordWrap(True)
        lbl_safe_hint.setStyleSheet("font-size: 8.5pt; padding-left: 20px;")
        v_gen.addWidget(lbl_safe_hint)

        v_gen.addStretch(1)

        self.tab_widget.addTab(tab_general, "General")
        self.general_tab = tab_general

        # -------------------------------------------------------------
        # Tab 2: Face Recognition
        # -------------------------------------------------------------
        from src.ui.options_dialog import FaceRecognitionOptionsTab

        self.face_tab = FaceRecognitionOptionsTab(engine=self.engine, parent=self)
        self.face_tab.purge_rescan_requested.connect(self._on_face_purge_rescan)
        self.face_tab.fast_rematch_requested.connect(self._on_face_fast_rematch)
        self.face_tab.heal_faces_requested.connect(self._on_face_heal_requested)
        self.face_tab.rescan_unscanned_requested.connect(self._on_face_rescan_unscanned)
        self.tab_widget.addTab(self.face_tab, "Face Recognition")

        # -------------------------------------------------------------
        # Tab 3: Import (docs: screenshot 3)
        # -------------------------------------------------------------
        tab_import = QWidget()
        v_imp = QVBoxLayout(tab_import)
        v_imp.setContentsMargins(14, 14, 14, 14)
        v_imp.setSpacing(10)

        h_dev = QHBoxLayout()
        lbl_set_for = QLabel("Settings for:", tab_import)
        self.cmb_import_device = QComboBox(tab_import)
        self.cmb_import_device.addItems(["Cameras", "CD/DVD", "Flash drive"])
        h_dev.addWidget(lbl_set_for)
        h_dev.addWidget(self.cmb_import_device, 1)
        v_imp.addLayout(h_dev)

        imp_frame = QFrame(tab_import)
        imp_frame.setFrameShape(QFrame.StyledPanel)
        imp_frame_layout = QVBoxLayout(imp_frame)
        imp_frame_layout.setContentsMargins(12, 10, 12, 10)
        imp_frame_layout.setSpacing(8)

        # Import to row
        h_to = QHBoxLayout()
        lbl_to = QLabel("Import to:   ", imp_frame)
        self.cmb_import_dest = QComboBox(imp_frame)
        self.cmb_import_dest.addItem("Pictures", str(Path.home() / "Pictures"))
        self.cmb_import_dest.addItem("Documents", str(Path.home() / "Documents"))
        self.btn_browse_dest = QPushButton("Browse...", imp_frame)
        self.btn_browse_dest.setFixedWidth(75)
        self.btn_browse_dest.clicked.connect(self._on_browse_import_dest)
        h_to.addWidget(lbl_to)
        h_to.addWidget(self.cmb_import_dest, 1)
        h_to.addWidget(self.btn_browse_dest)
        imp_frame_layout.addLayout(h_to)

        # Folder name row
        h_fn = QHBoxLayout()
        lbl_fn = QLabel("Folder name:", imp_frame)
        self.cmb_folder_naming = QComboBox(imp_frame)
        self.cmb_folder_naming.addItems(["Name", "Date + Name", "Name + Date", "Date"])
        self.cmb_folder_naming.currentIndexChanged.connect(self._update_import_example)
        h_fn.addWidget(lbl_fn)
        h_fn.addWidget(self.cmb_folder_naming, 1)
        imp_frame_layout.addLayout(h_fn)

        # File name row
        h_fin = QHBoxLayout()
        lbl_fin = QLabel("File name:    ", imp_frame)
        self.cmb_file_naming = QComboBox(imp_frame)
        self.cmb_file_naming.addItems(["Original File Name", "Name + Number", "Date + Name"])
        self.cmb_file_naming.currentIndexChanged.connect(self._update_import_example)
        h_fin.addWidget(lbl_fin)
        h_fin.addWidget(self.cmb_file_naming, 1)
        imp_frame_layout.addLayout(h_fin)

        # Example label
        h_ex = QHBoxLayout()
        lbl_ex_title = QLabel("Example:     ", imp_frame)
        self.lbl_import_example = QLabel("Himalayas\\DSC_1234.jpg", imp_frame)
        h_ex.addWidget(lbl_ex_title)
        h_ex.addWidget(self.lbl_import_example, 1)
        imp_frame_layout.addLayout(h_ex)

        # Other options checkboxes
        lbl_other = QLabel("Other options:", imp_frame)
        imp_frame_layout.addWidget(lbl_other)

        self.chk_open_after_import = QCheckBox("Open Photo Gallery after importing files", imp_frame)
        self.chk_open_after_import.setChecked(True)
        self.chk_delete_after_import = QCheckBox("Delete files from device after importing", imp_frame)
        self.chk_delete_after_import.setChecked(False)
        self.chk_rotate_during_import = QCheckBox("Rotate photos during import", imp_frame)
        self.chk_rotate_during_import.setChecked(False)

        imp_frame_layout.addWidget(self.chk_open_after_import)
        imp_frame_layout.addWidget(self.chk_delete_after_import)
        imp_frame_layout.addWidget(self.chk_rotate_during_import)

        v_imp.addWidget(imp_frame)

        lbl_autoplay = QLabel("<a href='autoplay' style='color:#0066CC;'>Change default autoplay options</a>", tab_import)
        lbl_autoplay.setOpenExternalLinks(False)
        v_imp.addWidget(lbl_autoplay)

        self.btn_restore_import_defaults = QPushButton("Restore defaults", tab_import)
        self.btn_restore_import_defaults.setFixedWidth(110)
        self.btn_restore_import_defaults.clicked.connect(self._restore_import_defaults)
        v_imp.addWidget(self.btn_restore_import_defaults)

        # Legacy Photo Manager Metadata Import (Google Picasa)
        picasa_group = QGroupBox("Import from Other Photo Managers", tab_import)
        picasa_lay = QVBoxLayout(picasa_group)
        picasa_lay.setSpacing(6)
        lbl_picasa = QLabel(
            "Bring in starred photos, captions, tags, people, and albums from Google Picasa folders.",
            picasa_group,
        )
        lbl_picasa.setWordWrap(True)
        picasa_lay.addWidget(lbl_picasa)
        self.btn_import_picasa = QPushButton("Import Picasa Metadata...", picasa_group)
        self.btn_import_picasa.setFixedWidth(190)
        self.btn_import_picasa.clicked.connect(self.picasa_import_requested.emit)
        picasa_lay.addWidget(self.btn_import_picasa)
        v_imp.addWidget(picasa_group)

        v_imp.addStretch(1)

        self.tab_widget.addTab(tab_import, "Import")
        self.import_tab = tab_import

        # -------------------------------------------------------------
        # Tab 3: Originals (docs: screenshot 4)
        # -------------------------------------------------------------
        tab_originals = QWidget()
        v_orig = QVBoxLayout(tab_originals)
        v_orig.setContentsMargins(14, 14, 14, 14)
        v_orig.setSpacing(12)

        orig_group = QGroupBox("Original photos", tab_originals)
        orig_layout = QVBoxLayout(orig_group)
        orig_layout.setContentsMargins(12, 12, 12, 12)
        orig_layout.setSpacing(10)

        lbl_orig_desc = QLabel(
            "When you edit photos, Photo Gallery saves a copy of the originals\n"
            "in your original photos folder so you can revert the changes you've\n"
            "made at a later time. To save hard drive space, set Photo Gallery to\n"
            "periodically move your original photos to the Recycle Bin.",
            orig_group,
        )
        lbl_orig_desc.setWordWrap(True)
        orig_layout.addWidget(lbl_orig_desc)

        lbl_go_orig = QLabel(
            "<a href='open_originals' style='color: #0066CC; text-decoration: underline;'>Go to original photos folder</a>",
            orig_group,
        )
        lbl_go_orig.setOpenExternalLinks(False)
        lbl_go_orig.linkActivated.connect(self._on_open_originals_link)
        orig_layout.addWidget(lbl_go_orig)

        lbl_recycle_after = QLabel("Move originals to Recycle Bin after:", orig_group)
        orig_layout.addWidget(lbl_recycle_after)

        self.cmb_retention = QComboBox(orig_group)
        self.cmb_retention.addItem("Never", 99999)
        self.cmb_retention.addItem("One day", 1)
        self.cmb_retention.addItem("One week", 7)
        self.cmb_retention.addItem("One month", 30)
        self.cmb_retention.addItem("Six months", 180)
        self.cmb_retention.addItem("One year", 365)

        # Select matching or closest preset
        best_idx = 4  # default: Six months
        for i in range(self.cmb_retention.count()):
            if self.cmb_retention.itemData(i) == retention_days:
                best_idx = i
                break
        self.cmb_retention.setCurrentIndex(best_idx)
        self.cmb_retention.currentIndexChanged.connect(self._on_retention_cmb_changed)
        orig_layout.addWidget(self.cmb_retention)

        v_orig.addWidget(orig_group)

        # Retained compatibility widget for tests
        self.spin_retention = QSpinBox(tab_originals)
        self.spin_retention.setRange(1, 99999)
        self.spin_retention.setValue(max(1, int(retention_days)))
        self.spin_retention.setVisible(False)
        self.spin_retention.valueChanged.connect(self._on_spin_retention_changed)
        v_orig.addWidget(self.spin_retention)
        v_orig.addStretch(1)

        self.tab_widget.addTab(tab_originals, "Originals")
        self.originals_tab = tab_originals

        # -------------------------------------------------------------
        # Tab 4: Edit (docs: screenshot 5)
        # -------------------------------------------------------------
        tab_edit = QWidget()
        v_edit = QVBoxLayout(tab_edit)
        v_edit.setContentsMargins(14, 14, 14, 14)
        v_edit.setSpacing(12)

        # Auto adjust settings
        grp_auto = QGroupBox("Auto adjust settings", tab_edit)
        v_auto = QVBoxLayout(grp_auto)
        v_auto.setContentsMargins(12, 10, 12, 10)
        v_auto.setSpacing(6)

        lbl_auto_desc = QLabel(
            "Select which adjustments you would like auto adjust to perform:",
            grp_auto,
        )
        v_auto.addWidget(lbl_auto_desc)

        self.chk_auto_straighten = QCheckBox("Straighten", grp_auto)
        self.chk_auto_straighten.setChecked(auto_straighten)
        v_auto.addWidget(self.chk_auto_straighten)

        self.chk_auto_noise = QCheckBox("Noise reduction", grp_auto)
        self.chk_auto_noise.setChecked(auto_noise)
        v_auto.addWidget(self.chk_auto_noise)

        self.chk_auto_color = QCheckBox("Color", grp_auto)
        self.chk_auto_color.setChecked(auto_color)
        v_auto.addWidget(self.chk_auto_color)

        self.chk_auto_exposure = QCheckBox("Exposure", grp_auto)
        self.chk_auto_exposure.setChecked(auto_exposure)
        v_auto.addWidget(self.chk_auto_exposure)

        v_edit.addWidget(grp_auto)

        # JPEG quality group
        self.grp_jpeg = QGroupBox(f"JPEG quality: {jpeg_quality}%", tab_edit)
        v_jpeg = QVBoxLayout(self.grp_jpeg)
        v_jpeg.setContentsMargins(12, 10, 12, 10)
        v_jpeg.setSpacing(8)

        lbl_jpeg_desc = QLabel(
            "JPEG files are compressed to save space and to make them\n"
            "faster to upload or email. The JPEG quality level controls how\n"
            "much compression is applied to the file. A higher value will\n"
            "result in a higher quality image and a larger file size.",
            self.grp_jpeg,
        )
        lbl_jpeg_desc.setWordWrap(True)
        v_jpeg.addWidget(lbl_jpeg_desc)

        self.slider_jpeg_quality = QSlider(Qt.Horizontal, self.grp_jpeg)
        self.slider_jpeg_quality.setRange(50, 100)
        self.slider_jpeg_quality.setValue(max(50, min(100, int(jpeg_quality))))
        self.slider_jpeg_quality.setTickPosition(QSlider.TicksBelow)
        self.slider_jpeg_quality.setTickInterval(10)
        self.slider_jpeg_quality.valueChanged.connect(self._on_jpeg_slider_changed)
        v_jpeg.addWidget(self.slider_jpeg_quality)

        h_labels = QHBoxLayout()
        lbl_low = QLabel("Low", self.grp_jpeg)
        lbl_high = QLabel("High", self.grp_jpeg)
        h_labels.addWidget(lbl_low)
        h_labels.addStretch(1)
        h_labels.addWidget(lbl_high)
        v_jpeg.addLayout(h_labels)

        v_edit.addWidget(self.grp_jpeg)

        # Save as copy settings group
        self.grp_save_copy = QGroupBox("Save as copy settings", tab_edit)
        v_copy = QVBoxLayout(self.grp_save_copy)
        v_copy.setContentsMargins(12, 10, 12, 10)
        v_copy.setSpacing(6)

        lbl_naming = QLabel("File naming format:", self.grp_save_copy)
        v_copy.addWidget(lbl_naming)

        self.cmb_save_naming = QComboBox(self.grp_save_copy)
        self.cmb_save_naming.addItem("Original name + _edited (e.g. photo_edited.jpg)", "{stem}_edited")
        self.cmb_save_naming.addItem("Original name + (edited) (e.g. photo (edited).jpg)", "{stem} (edited)")
        self.cmb_save_naming.addItem("Original name + _copy (e.g. photo_copy.jpg)", "{stem}_copy")
        self.cmb_save_naming.addItem("Original name + -edit (e.g. photo-edit.jpg)", "{stem}-edit")
        v_copy.addWidget(self.cmb_save_naming)

        naming_idx = self.cmb_save_naming.findData(self._init_save_copy_naming)
        if naming_idx >= 0:
            self.cmb_save_naming.setCurrentIndex(naming_idx)

        v_copy.addSpacing(2)
        lbl_dest = QLabel("Destination folder:", self.grp_save_copy)
        v_copy.addWidget(lbl_dest)

        self.rad_save_orig = QRadioButton("Save to original photo's folder", self.grp_save_copy)
        self.rad_save_custom = QRadioButton("Save to a specific folder:", self.grp_save_copy)
        self.rad_save_ask = QRadioButton("Ask each time", self.grp_save_copy)
        self.rad_save_ask.setToolTip("Prompt with a File Explorer save dialog each time Save as copy is clicked")

        if self._init_save_copy_location == "custom":
            self.rad_save_custom.setChecked(True)
        elif self._init_save_copy_location in ("ask", "prompt"):
            self.rad_save_ask.setChecked(True)
        else:
            self.rad_save_orig.setChecked(True)
        v_copy.addWidget(self.rad_save_orig)
        v_copy.addWidget(self.rad_save_custom)

        h_cust = QHBoxLayout()
        h_cust.setContentsMargins(20, 0, 0, 0)
        self.txt_save_custom_folder = QLineEdit(self.grp_save_copy)
        self.txt_save_custom_folder.setText(self._init_save_copy_custom_folder)
        self.txt_save_custom_folder.setEnabled(self.rad_save_custom.isChecked())
        self.btn_browse_custom_folder = QPushButton("Browse...", self.grp_save_copy)
        self.btn_browse_custom_folder.setEnabled(self.rad_save_custom.isChecked())
        self.btn_browse_custom_folder.clicked.connect(self._on_browse_save_copy_folder)
        self.rad_save_custom.toggled.connect(self.txt_save_custom_folder.setEnabled)
        self.rad_save_custom.toggled.connect(self.btn_browse_custom_folder.setEnabled)
        h_cust.addWidget(self.txt_save_custom_folder)
        h_cust.addWidget(self.btn_browse_custom_folder)
        v_copy.addLayout(h_cust)

        v_copy.addWidget(self.rad_save_ask)

        v_copy.addSpacing(2)
        self.chk_save_switch_view = QCheckBox("Switch to viewing the new copy after saving", self.grp_save_copy)
        self.chk_save_switch_view.setChecked(self._init_save_copy_switch_view)
        v_copy.addWidget(self.chk_save_switch_view)

        v_edit.addWidget(self.grp_save_copy)

        # Unsaved edits confirmation
        self.grp_unsaved = QGroupBox("Unsaved edits confirmation", tab_edit)
        v_unsaved = QVBoxLayout(self.grp_unsaved)
        v_unsaved.setContentsMargins(10, 10, 10, 10)
        v_unsaved.setSpacing(6)

        lbl_unsaved_desc = QLabel("When closing or navigating away from a photo with unsaved adjustments:")
        lbl_unsaved_desc.setStyleSheet(
            f"color: {'#94A3B8' if self._is_dark else '#1E395B'}; font-size: 9pt;"
        )
        v_unsaved.addWidget(lbl_unsaved_desc)

        self.cmb_unsaved_action = QComboBox(self.grp_unsaved)
        self.cmb_unsaved_action.addItem("Prompt what to do each time", "prompt")
        self.cmb_unsaved_action.addItem("Automatically Save (Overwrite original)", "overwrite")
        self.cmb_unsaved_action.addItem("Automatically Save as copy", "copy")
        self.cmb_unsaved_action.addItem("Discard changes without asking", "discard")

        cur_idx = self.cmb_unsaved_action.findData(self._init_unsaved_edits_action)
        if cur_idx >= 0:
            self.cmb_unsaved_action.setCurrentIndex(cur_idx)
        else:
            self.cmb_unsaved_action.setCurrentIndex(0)
        v_unsaved.addWidget(self.cmb_unsaved_action)

        v_edit.addWidget(self.grp_unsaved)
        v_edit.addStretch(1)

        self.tab_widget.addTab(tab_edit, "Edit")
        self.edit_tab = tab_edit

        # -------------------------------------------------------------
        # Tab 5: AI & Downloads (Optional Models & OCR Engines)
        # -------------------------------------------------------------
        tab_ai = QWidget()
        v_ai = QVBoxLayout(tab_ai)
        v_ai.setContentsMargins(14, 14, 14, 14)
        v_ai.setSpacing(12)

        ai_intro = QLabel(
            "Manage optional AI models and external OCR engines. Models are stored in your local AppData folder "
            "and can be uninstalled at any time."
        )
        ai_intro.setWordWrap(True)
        ai_intro.setStyleSheet("color: #666; font-size: 11px;")
        v_ai.addWidget(ai_intro)

        # 1. AI Neural Inpainting (LaMa)
        grp_lama = QGroupBox("AI Neural Inpainting (LaMa)", tab_ai)
        v_lama = QVBoxLayout(grp_lama)
        v_lama.setSpacing(6)
        lama_desc = QLabel(
            "Optional deep-learning fill model (~208 MB) that upgrades Object Eraser, Crop AI Expand, "
            "Panorama corner fill, and 3D Cinematic background fill. Built-in basic inpainting is used when not installed.",
            grp_lama,
        )
        lama_desc.setWordWrap(True)
        lama_desc.setStyleSheet("color: #666; font-size: 10.5px;")
        v_lama.addWidget(lama_desc)

        self.lbl_opt_lama_status = QLabel("Checking status...", grp_lama)
        self.lbl_opt_lama_status.setWordWrap(True)
        v_lama.addWidget(self.lbl_opt_lama_status)

        self.chk_opt_lama_enabled = QCheckBox(
            "Enable AI Object Eraser (LaMa) when model is installed", grp_lama
        )
        self.chk_opt_lama_enabled.toggled.connect(self._on_opt_lama_enabled_toggled)
        v_lama.addWidget(self.chk_opt_lama_enabled)

        lama_btn_row = QHBoxLayout()
        self.btn_opt_lama_download = QPushButton("Download AI Object Eraser", grp_lama)
        self.btn_opt_lama_uninstall = QPushButton("Uninstall Model", grp_lama)
        self.btn_opt_lama_download.clicked.connect(self._on_opt_lama_download)
        self.btn_opt_lama_uninstall.clicked.connect(self._on_opt_lama_uninstall)
        lama_btn_row.addWidget(self.btn_opt_lama_download)
        lama_btn_row.addWidget(self.btn_opt_lama_uninstall)
        lama_btn_row.addStretch(1)
        v_lama.addLayout(lama_btn_row)
        v_ai.addWidget(grp_lama)

        # 2. Offline Text Recognition (OCR)
        grp_ocr = QGroupBox("Offline Text Recognition (OCR)", tab_ai)
        v_ocr = QVBoxLayout(grp_ocr)
        v_ocr.setSpacing(7)
        ocr_desc = QLabel(
            "• Windows Native OCR: Built-in Windows 10/11 OCR — zero download, best for photos (used first by Auto).\n"
            "• RapidOCR (~15 MB): Self-contained offline AI text recognition downloaded to AppData.\n"
            "• Tesseract OCR: Uses an external tesseract.exe installation if already on your PC.",
            grp_ocr,
        )
        ocr_desc.setWordWrap(True)
        ocr_desc.setStyleSheet("color: #666; font-size: 10.5px;")
        v_ocr.addWidget(ocr_desc)

        self.lbl_opt_ocr_status = QLabel("Detecting OCR engine...", grp_ocr)
        self.lbl_opt_ocr_status.setWordWrap(True)
        self.lbl_opt_ocr_status.setStyleSheet("font-weight: 600;")
        v_ocr.addWidget(self.lbl_opt_ocr_status)

        ocr_engine_row = QHBoxLayout()
        ocr_engine_row.addWidget(QLabel("Preferred Engine:", grp_ocr))
        self.cmb_opt_ocr_backend = QComboBox(grp_ocr)
        self.cmb_opt_ocr_backend.addItem("Auto (Recommended)", "auto")
        self.cmb_opt_ocr_backend.addItem("Windows Native OCR (WinRT)", "windows_media_ocr")
        self.cmb_opt_ocr_backend.addItem("RapidOCR (ONNX)", "rapidocr")
        self.cmb_opt_ocr_backend.addItem("Tesseract OCR", "pytesseract")
        self.cmb_opt_ocr_backend.currentIndexChanged.connect(self._on_opt_ocr_backend_changed)
        ocr_engine_row.addWidget(self.cmb_opt_ocr_backend, 1)
        self.btn_opt_rescan_ocr = QPushButton("Re-scan All OCR...", grp_ocr)
        self.btn_opt_rescan_ocr.setToolTip(
            "Re-run OCR indexing across all OCR-enabled folders using the selected engine."
        )
        self.btn_opt_rescan_ocr.clicked.connect(self._on_opt_rescan_ocr)
        ocr_engine_row.addWidget(self.btn_opt_rescan_ocr)
        v_ocr.addLayout(ocr_engine_row)

        self.lbl_opt_rapid_status = QLabel("RapidOCR Add-on: Checking...", grp_ocr)
        self.lbl_opt_rapid_status.setWordWrap(True)
        v_ocr.addWidget(self.lbl_opt_rapid_status)

        rapid_dl_row = QHBoxLayout()
        self.btn_opt_rapid_download = QPushButton("Download RapidOCR (~15 MB)", grp_ocr)
        self.btn_opt_rapid_uninstall = QPushButton("Uninstall RapidOCR", grp_ocr)
        self.btn_opt_rapid_download.clicked.connect(self._on_opt_rapid_download)
        self.btn_opt_rapid_uninstall.clicked.connect(self._on_opt_rapid_uninstall)
        rapid_dl_row.addWidget(self.btn_opt_rapid_download)
        rapid_dl_row.addWidget(self.btn_opt_rapid_uninstall)
        rapid_dl_row.addStretch(1)
        v_ocr.addLayout(rapid_dl_row)

        rapid_row = QHBoxLayout()
        rapid_lbl = QLabel("RapidOCR Folder:", grp_ocr)
        rapid_lbl.setMinimumWidth(105)
        rapid_row.addWidget(rapid_lbl)
        self.txt_opt_rapid_dir = QLabel("Default (AppData)", grp_ocr)
        self.txt_opt_rapid_dir.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.txt_opt_rapid_dir.setStyleSheet("color: #444; font-size: 11px;")
        self.txt_opt_rapid_dir.setWordWrap(True)
        self.btn_opt_browse_rapid = QPushButton("Browse...", grp_ocr)
        self.btn_opt_reset_rapid = QPushButton("Reset", grp_ocr)
        self.btn_opt_browse_rapid.clicked.connect(self._on_opt_browse_rapidocr)
        self.btn_opt_reset_rapid.clicked.connect(self._on_opt_reset_rapidocr)
        rapid_row.addWidget(self.txt_opt_rapid_dir, 1)
        rapid_row.addWidget(self.btn_opt_browse_rapid)
        rapid_row.addWidget(self.btn_opt_reset_rapid)
        v_ocr.addLayout(rapid_row)

        tess_row = QHBoxLayout()
        tess_lbl = QLabel("Tesseract (.exe):", grp_ocr)
        tess_lbl.setMinimumWidth(105)
        tess_row.addWidget(tess_lbl)
        self.txt_opt_tesseract_path = QLabel("Not detected", grp_ocr)
        self.txt_opt_tesseract_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.txt_opt_tesseract_path.setStyleSheet("color: #444; font-size: 11px;")
        self.txt_opt_tesseract_path.setWordWrap(True)
        self.btn_opt_browse_tess = QPushButton("Browse...", grp_ocr)
        self.btn_opt_detect_tess = QPushButton("Auto-Detect", grp_ocr)
        self.btn_opt_browse_tess.clicked.connect(self._on_opt_browse_tesseract)
        self.btn_opt_detect_tess.clicked.connect(self._on_opt_detect_tesseract)
        tess_row.addWidget(self.txt_opt_tesseract_path, 1)
        tess_row.addWidget(self.btn_opt_browse_tess)
        tess_row.addWidget(self.btn_opt_detect_tess)
        v_ocr.addLayout(tess_row)

        v_ai.addWidget(grp_ocr)
        v_ai.addStretch(1)

        self._opt_lama_dl_thread = None
        self._opt_rapid_dl_thread = None
        self._refresh_ai_downloads_tab()

        self.addons_tab = tab_ai
        self.tab_widget.addTab(tab_ai, "Add-ons")
        self.ai_downloads_tab = tab_ai

        # -------------------------------------------------------------
        # Tab 6: Database
        # -------------------------------------------------------------
        tab_database = QWidget()
        v_db = QVBoxLayout(tab_database)
        v_db.setContentsMargins(14, 14, 14, 14)
        v_db.setSpacing(12)

        # Catalog & Cache Statistics Group
        stats_group = QGroupBox("Catalog & Cache Statistics", tab_database)
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(12, 10, 12, 10)
        stats_layout.setSpacing(6)

        self.lbl_db_stats = QLabel(stats_group)
        self.lbl_db_stats.setWordWrap(True)
        stats_layout.addWidget(self.lbl_db_stats)

        self.lbl_open_db_dir = QLabel(
            "<a href='open_db_dir' style='color: #3B82F6; text-decoration: underline;'>Open database folder in Explorer</a>",
            stats_group,
        )
        self.lbl_open_db_dir.setOpenExternalLinks(False)
        self.lbl_open_db_dir.linkActivated.connect(self._on_open_db_dir_link)
        stats_layout.addWidget(self.lbl_open_db_dir)

        v_db.addWidget(stats_group)
        self._refresh_db_stats()

        maint_frame = QGroupBox("Database & Backup Maintenance", tab_database)
        maint_layout = QVBoxLayout(maint_frame)
        maint_layout.setContentsMargins(12, 10, 12, 12)
        maint_layout.setSpacing(8)

        lbl_maint_desc = QLabel(
            "Maintain optimal performance for large photo catalogs by compacting SQLite\n"
            "databases and removing expired backup copies of edited photos.",
            maint_frame,
        )
        lbl_maint_desc.setWordWrap(True)
        maint_layout.addWidget(lbl_maint_desc)

        self.btn_compact_db = QPushButton(
            "Compact && Optimize Databases (catalog.db + thumbnails.db)", maint_frame
        )
        self.btn_compact_db.clicked.connect(self._on_compact_clicked)
        maint_layout.addWidget(self.btn_compact_db)

        self.btn_rebuild_thumbs = QPushButton(
            "Rebuild Thumbnail Cache (HiDPI && Panoramas)", maint_frame
        )
        self.btn_rebuild_thumbs.clicked.connect(self._on_rebuild_thumbs_clicked)
        maint_layout.addWidget(self.btn_rebuild_thumbs)

        self.btn_prune_backups = QPushButton(
            "Clean Up Expired Original Backups Now", maint_frame
        )
        self.btn_prune_backups.clicked.connect(self._on_prune_clicked)
        maint_layout.addWidget(self.btn_prune_backups)

        self.btn_backup_catalog = QPushButton(
            "Backup Catalog Database Now...", maint_frame
        )
        self.btn_backup_catalog.clicked.connect(self._on_backup_clicked)
        maint_layout.addWidget(self.btn_backup_catalog)

        self.btn_fresh_start = QPushButton(
            "Fresh Start (Reset Catalog && Thumbnails)...", maint_frame
        )
        self.btn_fresh_start.setStyleSheet(
            "QPushButton { color: #B91C1C; border: 1px solid #FCA5A5; background: #FEF2F2; "
            "border-radius: 3px; padding: 5px 10px; font-weight: 600; }"
            "QPushButton:hover { background: #FEE2E2; border-color: #EF4444; }"
            if not self._is_dark
            else "QPushButton { color: #FCA5A5; border: 1px solid #7F1D1D; background: #2D181B; "
            "border-radius: 3px; padding: 5px 10px; font-weight: 600; }"
            "QPushButton:hover { background: #451A20; border-color: #EF4444; }"
        )
        self.btn_fresh_start.setToolTip(
            "Deletes catalog.db and thumbnails.db to start with a clean empty gallery (your photo files on disk are NOT deleted)"
        )
        self.btn_fresh_start.clicked.connect(self._on_fresh_start_clicked)
        maint_layout.addWidget(self.btn_fresh_start)

        self.lbl_maint_status = QLabel("", maint_frame)
        self.lbl_maint_status.setStyleSheet("color: #3B82F6; font-weight: 600;")
        maint_layout.addWidget(self.lbl_maint_status)

        v_db.addWidget(maint_frame)
        v_db.addStretch(1)

        self.tab_widget.addTab(tab_database, "Database")
        self.database_tab = tab_database

        main_layout.addWidget(self.tab_widget)

        # OK / Cancel / Apply Buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply, self
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        apply_btn = self.button_box.button(QDialogButtonBox.Apply)
        if apply_btn:
            apply_btn.clicked.connect(self._on_apply)
        main_layout.addWidget(self.button_box)

        # Initial tab routing
        if initial_tab:
            itab = str(initial_tab).lower()
            if itab in ("edit", "editing", "edits"):
                self.tab_widget.setCurrentWidget(self.edit_tab)
            elif itab in ("face_recognition", "face", "faces"):
                self.tab_widget.setCurrentWidget(self.face_tab)
            elif itab in ("general",):
                self.tab_widget.setCurrentWidget(self.general_tab)
            elif itab in ("import", "importing"):
                self.tab_widget.setCurrentWidget(self.import_tab)
            elif itab in ("originals", "original"):
                self.tab_widget.setCurrentWidget(self.originals_tab)
            elif itab in ("addons", "add-ons", "ocr", "ai"):
                self.tab_widget.setCurrentWidget(self.addons_tab)
            elif itab in ("database", "db"):
                self.tab_widget.setCurrentWidget(self.database_tab)
            elif itab in ("downloads", "ai_downloads", "ai", "addons", "optional_downloads"):
                self.tab_widget.setCurrentWidget(self.ai_downloads_tab)

    def _apply_dialog_theme(self, is_dark: bool) -> None:
        """Apply light or dark stylesheet to this dialog and its tab widget (supports live Apply)."""
        self._is_dark = bool(is_dark)
        if self._is_dark:
            self.setStyleSheet(
                """
                QDialog { background: #1A1D24; color: #E2E8F0; }
                QLabel, QCheckBox, QRadioButton, QGroupBox { color: #E2E8F0; }
                QGroupBox { border: 1px solid #3E4756; border-radius: 4px; margin-top: 8px; padding-top: 10px; }
                QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #94A3B8; }
                QLineEdit, QSpinBox, QComboBox { background: #262B35; color: #E2E8F0; border: 1px solid #3E4756; border-radius: 3px; padding: 3px 6px; }
                QPushButton { background: #262B35; color: #E2E8F0; border: 1px solid #3E4756; border-radius: 3px; padding: 4px 10px; }
                QPushButton:hover { background: #303744; }
                """
            )
            self.tab_widget.setStyleSheet(
                """
                QTabWidget::pane {
                    border: 1px solid #3E4756;
                    background: #1E222A;
                    top: -1px;
                }
                QTabBar::tab {
                    background: #262B35;
                    border: 1px solid #3E4756;
                    border-top-left-radius: 3px;
                    border-top-right-radius: 3px;
                    padding: 4px 12px;
                    margin-right: 2px;
                    color: #CBD5E1;
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                }
                QTabBar::tab:hover {
                    background: #303744;
                    border-color: #4B5563;
                    color: #FFFFFF;
                }
                QTabBar::tab:selected {
                    background: #1E222A;
                    border: 1px solid #3E4756;
                    border-bottom: 1px solid #1E222A;
                    margin-bottom: -1px;
                    font-weight: 600;
                    color: #FFFFFF;
                }
                """
            )
        else:
            self.setStyleSheet("")
            self.tab_widget.setStyleSheet(
                """
                QTabWidget::pane {
                    border: 1px solid #9EB6CE;
                    background: #FFFFFF;
                    top: -1px;
                }
                QTabBar::tab {
                    background: #F0F4F9;
                    border: 1px solid #9EB6CE;
                    border-top-left-radius: 3px;
                    border-top-right-radius: 3px;
                    padding: 4px 12px;
                    margin-right: 2px;
                    color: #1E395B;
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                }
                QTabBar::tab:hover {
                    background: #E5EEF8;
                    border-color: #7DA2CE;
                }
                QTabBar::tab:selected {
                    background: #FFFFFF;
                    border: 1px solid #9EB6CE;
                    border-bottom: 1px solid #FFFFFF;
                    margin-bottom: -1px;
                    font-weight: 600;
                    color: #000000;
                }
                """
            )

    def _refresh_ai_downloads_tab(self) -> None:
        try:
            from src.imaging.creative_effects import get_lama_model_status, is_lama_enabled

            st = get_lama_model_status()
            enabled = is_lama_enabled()
            self.chk_opt_lama_enabled.blockSignals(True)
            self.chk_opt_lama_enabled.setChecked(enabled)
            self.chk_opt_lama_enabled.blockSignals(False)
            if st.get("installed"):
                state_txt = "Enabled" if enabled else "Disabled (using built-in synthesizer)"
                self.lbl_opt_lama_status.setText(
                    f"Status: Installed ({st.get('filename')}, {st.get('size_mb', 0):.0f} MB) — {state_txt}"
                )
                self.btn_opt_lama_download.setText("Reinstall / Repair")
                self.btn_opt_lama_uninstall.setVisible(True)
                self.btn_opt_lama_uninstall.setEnabled(True)
            else:
                self.lbl_opt_lama_status.setText(
                    f"Status: Not installed (using built-in multi-scale texture synthesizer). "
                    f"Optional download: ~{st.get('size_mb', 50)} MB."
                )
                self.btn_opt_lama_download.setText(f"Download AI Object Eraser (~{st.get('size_mb', 50)} MB)")
                self.btn_opt_lama_uninstall.setVisible(False)
                self.btn_opt_lama_uninstall.setEnabled(False)
        except Exception as exc:
            self.lbl_opt_lama_status.setText(f"Status unavailable: {exc}")

        try:
            from src.core.ocr_engine import OcrEngine, get_ocr_engine_status

            pref = OcrEngine.get_preferred_backend()
            self.cmb_opt_ocr_backend.blockSignals(True)
            p_idx = self.cmb_opt_ocr_backend.findData(pref)
            self.cmb_opt_ocr_backend.setCurrentIndex(max(0, p_idx))
            self.cmb_opt_ocr_backend.blockSignals(False)

            ocr_st = get_ocr_engine_status()
            labels = {
                "rapidocr": "RapidOCR (ONNX)",
                "winocr": "Windows Native OCR (WinRT)",
                "windows_media_ocr": "Windows.Media.Ocr",
                "pytesseract": "Tesseract OCR",
                "stub": "Not configured (MSER stub)",
            }
            b_label = labels.get(ocr_st["backend"], ocr_st["backend"])
            self.lbl_opt_ocr_status.setText(f"Active OCR Engine: {b_label}")
            tess_det = ocr_st.get("tesseract_detected") or ""
            self.txt_opt_tesseract_path.setText(tess_det or "Not found on PC")
            self.txt_opt_tesseract_path.setToolTip(tess_det)

            rapid_installed = bool(ocr_st.get("rapidocr_installed"))
            rapid_mb = ocr_st.get("rapidocr_size_mb", 15)
            custom_dir = ocr_st.get("custom_model_dir") or ""
            if custom_dir:
                from src.core.ocr_engine import _find_rapidocr_files
                det_p, rec_p, _ = _find_rapidocr_files(Path(custom_dir))
                if det_p and rec_p:
                    self.txt_opt_rapid_dir.setText(custom_dir)
                else:
                    self.txt_opt_rapid_dir.setText(f"{custom_dir} (no .onnx models found — click Reset)")
            else:
                self.txt_opt_rapid_dir.setText("Default (AppData)")

            if rapid_installed:
                self.lbl_opt_rapid_status.setText(f"RapidOCR Add-on: Installed ({rapid_mb} MB)")
                self.lbl_opt_rapid_status.setStyleSheet("color: #1B6E2D; font-size: 11px;")
                self.btn_opt_rapid_download.setText("Reinstall / Repair")
                self.btn_opt_rapid_uninstall.setVisible(True)
                self.btn_opt_rapid_uninstall.setEnabled(True)
            else:
                self.lbl_opt_rapid_status.setText(f"RapidOCR Add-on: Not installed (~{rapid_mb} MB download)")
                self.lbl_opt_rapid_status.setStyleSheet("color: #666; font-size: 11px;")
                self.btn_opt_rapid_download.setText(f"Download RapidOCR (~{rapid_mb} MB)")
                self.btn_opt_rapid_uninstall.setVisible(False)
                self.btn_opt_rapid_uninstall.setEnabled(False)
        except Exception as exc:
            self.lbl_opt_ocr_status.setText(f"OCR status unavailable: {exc}")

    def _on_opt_lama_enabled_toggled(self, checked: bool) -> None:
        from src.imaging.creative_effects import set_lama_enabled

        set_lama_enabled(bool(checked))
        self._refresh_ai_downloads_tab()

    def _on_opt_ocr_backend_changed(self, _idx: int) -> None:
        from src.core.ocr_engine import OcrEngine

        backend = str(self.cmb_opt_ocr_backend.currentData() or "auto")
        OcrEngine.set_preferred_backend(backend)
        self._refresh_ai_downloads_tab()

    def _on_opt_lama_download(self) -> None:
        if self._opt_lama_dl_thread is not None and self._opt_lama_dl_thread.isRunning():
            return
        from PySide6.QtCore import QThread, Signal as QtSignal

        class _LamaOptDlWorker(QThread):
            progress = QtSignal(int, int)
            finished_ok = QtSignal(bool)

            def run(self) -> None:
                from src.imaging.creative_effects import download_lama_model

                path = download_lama_model(
                    progress_cb=lambda d, t: self.progress.emit(int(d), int(t))
                )
                self.finished_ok.emit(path is not None)

        self.btn_opt_lama_download.setEnabled(False)
        self.btn_opt_lama_download.setText("Downloading...")
        self.btn_opt_lama_uninstall.setVisible(False)
        worker = _LamaOptDlWorker(self)

        def _on_prog(done: int, total: int) -> None:
            if total > 0:
                pct = int(min(99, (done * 100) // total))
                self.btn_opt_lama_download.setText(f"Downloading {pct}%...")
            else:
                self.btn_opt_lama_download.setText(f"Downloading {done // (1024 * 1024)} MB...")

        def _on_done(ok: bool) -> None:
            self._opt_lama_dl_thread = None
            self.btn_opt_lama_download.setEnabled(True)
            self._refresh_ai_downloads_tab()
            if not ok:
                QMessageBox.warning(
                    self,
                    "Download Failed",
                    "Could not download the AI Object Eraser model. Check your internet connection and try again.",
                )

        worker.progress.connect(_on_prog)
        worker.finished_ok.connect(_on_done)
        self._opt_lama_dl_thread = worker
        worker.start()

    def _on_opt_lama_uninstall(self) -> None:
        from src.imaging.creative_effects import uninstall_lama_model

        if uninstall_lama_model():
            QMessageBox.information(self, "Uninstalled", "AI Object Eraser model removed from AppData.")
        self._refresh_ai_downloads_tab()

    def _on_opt_browse_tesseract(self) -> None:
        from src.core.ocr_engine import OcrEngine, resolve_tesseract_executable

        start_dir = r"C:\Program Files\Tesseract-OCR" if Path(r"C:\Program Files\Tesseract-OCR").is_dir() else ""
        exe, _ = QFileDialog.getOpenFileName(
            self,
            "Select tesseract.exe (e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe)",
            start_dir,
            "Tesseract Executable (tesseract.exe);;Executables (*.exe);;All Files (*)",
        )
        if exe:
            resolved = resolve_tesseract_executable(exe) or exe
            OcrEngine.save_ocr_paths(tesseract_path=resolved)
            self._refresh_ai_downloads_tab()

    def _on_opt_detect_tesseract(self) -> None:
        from src.core.ocr_engine import OcrEngine, detect_tesseract_executable

        OcrEngine.save_ocr_paths(tesseract_path="")
        detected = detect_tesseract_executable()
        self._refresh_ai_downloads_tab()
        if detected:
            QMessageBox.information(self, "Tesseract Detected", f"Detected Tesseract at:\n{detected}")
        else:
            QMessageBox.information(
                self,
                "Tesseract Not Found",
                "Could not find tesseract.exe in PATH or standard Program Files folders.",
            )

    def _on_opt_browse_rapidocr(self) -> None:
        from src.core.ocr_engine import OcrEngine, _find_rapidocr_files

        folder = QFileDialog.getExistingDirectory(self, "Select RapidOCR ONNX Models Folder")
        if folder:
            det_p, rec_p, _ = _find_rapidocr_files(Path(folder))
            if not (det_p and rec_p):
                QMessageBox.warning(
                    self,
                    "No RapidOCR Models Found",
                    "The selected folder does not contain RapidOCR ONNX model files (*det*.onnx and *rec*.onnx).",
                )
                return
            OcrEngine.save_ocr_paths(custom_model_dir=folder)
            self._refresh_ai_downloads_tab()

    def _on_opt_reset_rapidocr(self) -> None:
        from src.core.ocr_engine import OcrEngine

        OcrEngine.save_ocr_paths(custom_model_dir="")
        self._refresh_ai_downloads_tab()

    def _on_opt_rapid_download(self) -> None:
        if self._opt_rapid_dl_thread is not None and self._opt_rapid_dl_thread.isRunning():
            return
        from PySide6.QtCore import QThread, Signal as QtSignal

        class _RapidOptDlWorker(QThread):
            progress = QtSignal(int, int)
            finished_ok = QtSignal(bool)

            def run(self) -> None:
                from src.core.ocr_engine import download_rapidocr_models

                path = download_rapidocr_models(
                    progress_cb=lambda d, t: self.progress.emit(int(d), int(t))
                )
                self.finished_ok.emit(path is not None)

        self.btn_opt_rapid_download.setEnabled(False)
        self.btn_opt_rapid_download.setText("Downloading...")
        self.btn_opt_rapid_uninstall.setVisible(False)
        worker = _RapidOptDlWorker(self)

        def _on_prog(done: int, total: int) -> None:
            if total > 0:
                pct = int(min(99, (done * 100) // total))
                self.btn_opt_rapid_download.setText(f"Downloading {pct}%...")
            else:
                self.btn_opt_rapid_download.setText(f"Downloading {done // (1024 * 1024)} MB...")

        def _on_done(ok: bool) -> None:
            self._opt_rapid_dl_thread = None
            self.btn_opt_rapid_download.setEnabled(True)
            self._refresh_ai_downloads_tab()
            if not ok:
                QMessageBox.warning(
                    self,
                    "Download Failed",
                    "Could not download RapidOCR models. Check your internet connection and try again.",
                )

        worker.progress.connect(_on_prog)
        worker.finished_ok.connect(_on_done)
        self._opt_rapid_dl_thread = worker
        worker.start()

    def _on_opt_rapid_uninstall(self) -> None:
        from src.core.ocr_engine import uninstall_rapidocr_models

        if uninstall_rapidocr_models():
            QMessageBox.information(self, "Uninstalled", "RapidOCR models removed from AppData.")
        self._refresh_ai_downloads_tab()

    def _refresh_db_stats(self) -> None:
        if not hasattr(self, "lbl_db_stats"):
            return
        if self.engine is None:
            self.lbl_db_stats.setText("Database statistics unavailable.")
            return
        try:
            cat_path = Path(self.engine.catalog_path)
            thumb_path = Path(self.engine.thumbnails_path)
            cat_mb = cat_path.stat().st_size / (1024 * 1024) if cat_path.exists() else 0.0
            thumb_mb = thumb_path.stat().st_size / (1024 * 1024) if thumb_path.exists() else 0.0
            conn = self.engine.catalog_conn()
            media_cnt = int(conn.execute("SELECT COUNT(*) FROM media;").fetchone()[0])
            folder_cnt = int(conn.execute("SELECT COUNT(*) FROM folders;").fetchone()[0])
            people_cnt = int(conn.execute("SELECT COUNT(*) FROM people;").fetchone()[0])
            dirty_cnt = int(conn.execute("SELECT COUNT(*) FROM media WHERE metadata_dirty = 1;").fetchone()[0])
            self.lbl_db_stats.setText(
                f"Indexed Media: {media_cnt:,} items across {folder_cnt:,} folders  |  "
                f"People: {people_cnt:,}  |  Pending file writes: {dirty_cnt:,}\n"
                f"Storage: catalog.db ({cat_mb:.1f} MB)  •  thumbnails.db ({thumb_mb:.1f} MB)\n"
                f"Location: {cat_path.parent}"
            )
        except Exception:
            self.lbl_db_stats.setText("Database statistics ready.")

    def _on_open_db_dir_link(self, _link: str) -> None:
        if self.engine is not None and hasattr(self.engine, "catalog_path"):
            db_dir = Path(self.engine.catalog_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(db_dir)))

    def _on_face_purge_rescan(self) -> None:
        self.purge_rescan_requested = True

    def _on_face_fast_rematch(self) -> None:
        self.fast_rematch_requested = True

    def _on_face_rescan_unscanned(self) -> None:
        self.rescan_unscanned_requested = True

    def _on_opt_rescan_ocr(self) -> None:
        engine_label = self.cmb_opt_ocr_backend.currentText()
        reply = QMessageBox.question(
            self,
            "Re-scan All OCR",
            f"Re-scan OCR text across all OCR-enabled folders using '{engine_label}'?\n\n"
            "This runs in the background and will update previously scanned photos.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.rescan_ocr_requested = True
            self.accept()

    def _on_apply(self) -> None:
        if hasattr(self, "face_tab") and self.face_tab:
            self.face_tab.save_settings()
        self._save_copy_preferences()
        self.settings_applied.emit()
        self._apply_dialog_theme(self.selected_theme() == "dark")

    def accept(self) -> None:
        if hasattr(self, "face_tab") and self.face_tab:
            self.face_tab.save_settings()
        self._save_copy_preferences()
        super().accept()

    def _save_copy_preferences(self) -> None:
        settings = QSettings("PhotoGallery64", "PhotoGallery64")
        settings.setValue("save_copy_location", self.save_copy_location())
        settings.setValue("save_copy_custom_folder", self.save_copy_custom_folder())
        settings.setValue("save_copy_naming_pattern", self.save_copy_naming_pattern())
        settings.setValue("save_copy_switch_view", self.save_copy_switch_view())
        settings.setValue("unsaved_edits_action", self.unsaved_edits_action())

    def _on_browse_save_copy_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Save as Copy Destination Folder")
        if folder:
            self.txt_save_custom_folder.setText(folder)

    def _on_browse_import_dest(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Import Folder")
        if folder:
            idx = self.cmb_import_dest.findData(folder)
            if idx < 0:
                self.cmb_import_dest.addItem(Path(folder).name, folder)
                idx = self.cmb_import_dest.count() - 1
            self.cmb_import_dest.setCurrentIndex(idx)

    def _update_import_example(self) -> None:
        fn = self.cmb_folder_naming.currentText()
        fin = self.cmb_file_naming.currentText()
        folder_ex = "Himalayas" if "Name" in fn else "2026-10-17"
        if fn == "Date + Name":
            folder_ex = "2026-10-17 Himalayas"
        elif fn == "Name + Date":
            folder_ex = "Himalayas 2026-10-17"

        file_ex = "DSC_1234.jpg"
        if fin == "Name + Number":
            file_ex = "Himalayas 001.jpg"
        elif fin == "Date + Name":
            file_ex = "2026-10-17 Himalayas.jpg"

        self.lbl_import_example.setText(f"{folder_ex}\\{file_ex}")

    def _restore_import_defaults(self) -> None:
        self.cmb_import_device.setCurrentIndex(0)
        self.cmb_import_dest.setCurrentIndex(0)
        self.cmb_folder_naming.setCurrentIndex(0)
        self.cmb_file_naming.setCurrentIndex(0)
        self.chk_open_after_import.setChecked(True)
        self.chk_delete_after_import.setChecked(False)
        self.chk_rotate_during_import.setChecked(False)
        self._update_import_example()

    def _on_open_originals_link(self, _link: str) -> None:
        p = Path(self.originals_dir)
        p.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def _on_retention_cmb_changed(self, idx: int) -> None:
        val = self.cmb_retention.itemData(idx)
        if val is not None:
            self.spin_retention.blockSignals(True)
            self.spin_retention.setValue(int(val))
            self.spin_retention.blockSignals(False)

    def _on_spin_retention_changed(self, val: int) -> None:
        for i in range(self.cmb_retention.count()):
            if self.cmb_retention.itemData(i) == val:
                self.cmb_retention.blockSignals(True)
                self.cmb_retention.setCurrentIndex(i)
                self.cmb_retention.blockSignals(False)
                return
        # If custom number entered not in dropdown, add it
        self.cmb_retention.addItem(f"{val} days", val)
        self.cmb_retention.setCurrentIndex(self.cmb_retention.count() - 1)

    def _on_jpeg_slider_changed(self, val: int) -> None:
        self.grp_jpeg.setTitle(f"JPEG quality: {val}%")

    def _on_compact_clicked(self) -> None:
        self.compact_requested = True
        self.lbl_maint_status.setText("✓ Database optimization queued on close.")

    def _on_rebuild_thumbs_clicked(self) -> None:
        self.rebuild_thumbs_requested = True
        self.lbl_maint_status.setText("✓ Thumbnail cache rebuild queued on close (regenerates visible thumbnails at HiDPI quality).")

    def _on_prune_clicked(self) -> None:
        self.prune_backups_requested = True
        self.lbl_maint_status.setText("✓ Expired backup cleanup queued on close.")

    def _on_backup_clicked(self) -> None:
        self.backup_requested = True
        self.lbl_maint_status.setText("✓ Catalog backup queued on close.")

    def _on_fresh_start_clicked(self) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Fresh Start — Reset Catalog?")
        box.setText("Delete the photo catalog and start completely fresh?")
        box.setInformativeText(
            "This will permanently delete your catalog database (catalog.db) and cached "
            "thumbnails (thumbnails.db), clearing all indexed folders, tags, people, albums, "
            "and face matches.\n\n"
            "Your original photo and video files on disk will NOT be deleted."
        )
        btn_reset = box.addButton("Reset && Fresh Start", QMessageBox.DestructiveRole)
        btn_cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(btn_cancel)
        box.exec()
        if box.clickedButton() == btn_reset:
            self.fresh_start_requested = True
            self.accept()

    def _on_heal_faces_clicked(self) -> None:
        self.heal_faces_requested = True
        self.lbl_maint_status.setText("✓ Re-parse & heal face regions queued on close.")

    def _on_face_heal_requested(self) -> None:
        self.heal_faces_requested = True

    def pause_write_back_enabled(self) -> bool:
        return not self.chk_write_to_files.isChecked()

    def write_to_files_enabled(self) -> bool:
        return self.chk_write_to_files.isChecked()

    def retention_days(self) -> int:
        return int(self.spin_retention.value())

    def selected_theme(self) -> str:
        return str(self.cmb_theme.currentData() or "wlpg_aero")

    def selected_icon_theme(self) -> str:
        return str(self.cmb_icon_theme.currentData() or "wlpg_aero")

    def show_dates(self) -> bool:
        return self.chk_show_dates.isChecked()

    def show_tags(self) -> bool:
        return self.chk_show_tags.isChecked()

    def show_people(self) -> bool:
        return self.chk_show_people.isChecked()

    def show_ratings(self) -> bool:
        return self.chk_show_ratings.isChecked()

    def show_places(self) -> bool:
        return self.chk_show_places.isChecked()

    def show_albums(self) -> bool:
        return self.chk_show_albums.isChecked()

    def favourites_enabled(self) -> bool:
        return self.chk_favourites_enabled.isChecked()

    def tooltip_previews_enabled(self) -> bool:
        return True

    def find_people_enabled(self) -> bool:
        return self.chk_find_people.isChecked()

    def jpeg_quality(self) -> int:
        return int(self.slider_jpeg_quality.value())

    def auto_straighten(self) -> bool:
        return self.chk_auto_straighten.isChecked()

    def auto_noise(self) -> bool:
        return self.chk_auto_noise.isChecked()

    def auto_color(self) -> bool:
        return self.chk_auto_color.isChecked()

    def auto_exposure(self) -> bool:
        return self.chk_auto_exposure.isChecked()

    def save_copy_location(self) -> str:
        if getattr(self, "rad_save_ask", None) and self.rad_save_ask.isChecked():
            return "ask"
        if getattr(self, "rad_save_custom", None) and self.rad_save_custom.isChecked():
            return "custom"
        return "original"

    def save_copy_custom_folder(self) -> str:
        return self.txt_save_custom_folder.text().strip() if hasattr(self, "txt_save_custom_folder") else ""

    def save_copy_naming_pattern(self) -> str:
        return str(self.cmb_save_naming.currentData()) if hasattr(self, "cmb_save_naming") else "{stem}_edited"

    def save_copy_switch_view(self) -> bool:
        return self.chk_save_switch_view.isChecked() if hasattr(self, "chk_save_switch_view") else True

    def unsaved_edits_action(self) -> str:
        return str(self.cmb_unsaved_action.currentData()) if hasattr(self, "cmb_unsaved_action") else "prompt"


def compute_resize_target(
    src_w: int,
    src_h: int,
    mode: str = "long_edge",
    max_dim: int = 1024,
    target_w: int = 1024,
    target_h: int = 768,
    percent: int = 50,
    lock_aspect: bool = True,
    center_crop: bool = False,
    no_upscale: bool = True,
) -> tuple[int, int, bool]:
    """
    Computes (out_w, out_h, use_center_crop) for the given source dimensions and resize settings.
    """
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))

    if mode == "percent":
        pct = max(1, min(100, int(percent))) if no_upscale else max(1, int(percent))
        scale = pct / 100.0
        return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale))), False

    if mode == "exact":
        tw = max(1, int(target_w))
        th = max(1, int(target_h))
        if no_upscale and src_w <= tw and src_h <= th:
            return src_w, src_h, False
        if not lock_aspect and center_crop:
            return tw, th, True
        if lock_aspect:
            scale = min(tw / float(src_w), th / float(src_h))
            if no_upscale and scale >= 1.0:
                return src_w, src_h, False
            return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale))), False
        if no_upscale:
            return min(src_w, tw), min(src_h, th), False
        return tw, th, False

    # Default: "long_edge"
    md = max(1, int(max_dim))
    cur_max = max(src_w, src_h)
    if no_upscale and cur_max <= md:
        return src_w, src_h, False
    scale = md / float(cur_max)
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale))), False


def _estimate_resized_file_bytes(out_w: int, out_h: int, fmt_ext: str, quality: int) -> int:
    """Estimates output file size in bytes for live UI preview."""
    px = max(1, int(out_w) * int(out_h))
    ext = fmt_ext.lower()
    if ext == ".png":
        return int(px * 1.8)
    q_norm = max(0.1, min(1.0, float(quality) / 100.0))
    jpeg_bpp = 0.12 + 1.85 * (q_norm ** 2.2)
    if ext == ".webp":
        return int(px * jpeg_bpp * 0.72 / 1.0)
    return int(px * jpeg_bpp)


def _format_bytes_human(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"~{num_bytes} B"
    kb = num_bytes / 1024.0
    if kb < 1024.0:
        return f"~{int(round(kb))} KB"
    return f"~{kb / 1024.0:.1f} MB"


def _resolve_output_format_and_ext(src_ext: str, out_format: str) -> tuple[str, str]:
    fmt_key = (out_format or "same").lower()
    if fmt_key == "jpeg":
        return "JPEG", ".jpg"
    if fmt_key == "webp":
        return "WEBP", ".webp"
    if fmt_key == "png":
        return "PNG", ".png"

    ext = (src_ext or ".jpg").lower()
    if ext in (".jpg", ".jpeg"):
        return "JPEG", src_ext or ".jpg"
    if ext == ".webp":
        return "WEBP", ".webp"
    if ext == ".png":
        return "PNG", ".png"
    return "JPEG", ".jpg"


def _resolve_export_filename_stem(
    src: Path,
    *,
    naming_pattern: str = "auto",
    resize_enabled: bool = False,
    mode: str = "long_edge",
    out_w: int = 0,
    out_h: int = 0,
    ext: str = ".jpg",
) -> str:
    """Resolve the output filename stem for Export / Resize using friendly presets or custom tokens."""
    import time as _time

    stem = src.stem
    pat = (naming_pattern or "auto").strip()
    size_tag = (
        f"{max(out_w, out_h)}px"
        if (resize_enabled and mode == "long_edge")
        else (f"{out_w}x{out_h}" if resize_enabled else "orig")
    )
    if pat in ("", "auto"):
        if resize_enabled:
            return f"{stem}_{size_tag}"
        src_ext = (src.suffix or "").lower()
        return stem if src_ext != ext.lower() else f"{stem}_export"

    if "{stem}" not in pat and "{size}" not in pat and "{w}" not in pat and "{h}" not in pat and "{date}" not in pat:
        # Plain suffix or exact stem
        if pat.startswith("_") or pat.startswith("-"):
            return f"{stem}{pat}"
        return pat

    date_str = _time.strftime("%Y%m%d")
    resolved = (
        pat.replace("{stem}", stem)
        .replace("{size}", size_tag)
        .replace("{w}", str(out_w))
        .replace("{h}", str(out_h))
        .replace("{date}", date_str)
    )
    # Sanitize Windows-invalid filename characters
    for bad in '<>:"/\\|?*':
        resolved = resolved.replace(bad, "_")
    return resolved.strip() or f"{stem}_export"


def _process_single_resize(
    src: Path,
    out_dir: Path,
    *,
    resize_enabled: bool = True,
    mode: str = "long_edge",
    max_dim: int = 1024,
    target_w: int = 1024,
    target_h: int = 768,
    percent: int = 50,
    lock_aspect: bool = True,
    center_crop: bool = False,
    no_upscale: bool = True,
    out_format: str = "same",
    quality: int = 88,
    strip_metadata: bool = False,
    naming_pattern: str = "auto",
    in_memory_image: Image.Image | None = None,
) -> Path | None:
    """Resizes/exports one image (or in-memory rendered image) and writes a non-destructive copy."""
    exif_bytes: bytes | None = None
    if in_memory_image is not None:
        img = in_memory_image.copy()
        if not strip_metadata and src.exists():
            try:
                with Image.open(src) as disk_img:
                    exif_bytes = disk_img.info.get("exif")
            except Exception:
                exif_bytes = None
    else:
        if not src.exists():
            return None
        with Image.open(src) as opened:
            exif_bytes = None if strip_metadata else opened.info.get("exif")
            img = ImageOps.exif_transpose(opened).copy()

    src_w, src_h = img.size
    if resize_enabled:
        out_w, out_h, use_crop = compute_resize_target(
            src_w,
            src_h,
            mode=mode,
            max_dim=max_dim,
            target_w=target_w,
            target_h=target_h,
            percent=percent,
            lock_aspect=lock_aspect,
            center_crop=center_crop,
            no_upscale=no_upscale,
        )
        if use_crop:
            res_img = ImageOps.fit(img, (out_w, out_h), method=Image.Resampling.LANCZOS)
        elif (out_w, out_h) != (src_w, src_h):
            res_img = img.resize((out_w, out_h), Image.Resampling.LANCZOS)
        else:
            res_img = img
    else:
        out_w, out_h = src_w, src_h
        res_img = img

    pil_fmt, ext = _resolve_output_format_and_ext(src.suffix or ".jpg", out_format)
    if pil_fmt == "JPEG" and res_img.mode not in ("RGB", "L"):
        res_img = res_img.convert("RGB")

    base_target_stem = _resolve_export_filename_stem(
        src,
        naming_pattern=naming_pattern,
        resize_enabled=resize_enabled,
        mode=mode,
        out_w=out_w,
        out_h=out_h,
        ext=ext,
    )

    target = out_dir / f"{base_target_stem}{ext}"
    if target.exists() or (src.exists() and target.resolve() == src.resolve()):
        c = 2
        while (out_dir / f"{base_target_stem}_{c}{ext}").exists():
            c += 1
        target = out_dir / f"{base_target_stem}_{c}{ext}"

    save_kwargs: dict[str, Any] = {"format": pil_fmt}
    if pil_fmt in ("JPEG", "WEBP"):
        save_kwargs["quality"] = max(1, min(100, int(quality)))
    if strip_metadata:
        for meta_key in ("exif", "XML:com.adobe.xmp", "xmp", "photoshop", "iptc"):
            res_img.info.pop(meta_key, None)
    elif exif_bytes and pil_fmt in ("JPEG", "WEBP"):
        save_kwargs["exif"] = exif_bytes

    res_img.save(target, **save_kwargs)
    return target


class _ResizeWorker(QThread):
    progress = Signal(int, int)  # current, total
    finished_batch = Signal(list)  # list[Path]

    def __init__(
        self,
        file_paths: list[Path],
        max_dim: int = 1024,
        out_dir: Path = Path("."),
        parent: QObject | None = None,
        *,
        resize_enabled: bool = True,
        mode: str = "long_edge",
        target_w: int = 1024,
        target_h: int = 768,
        percent: int = 50,
        lock_aspect: bool = True,
        center_crop: bool = False,
        no_upscale: bool = True,
        out_format: str = "same",
        quality: int = 88,
        strip_metadata: bool = False,
        naming_pattern: str = "auto",
        in_memory_image: Image.Image | None = None,
    ) -> None:
        super().__init__(parent)
        self.file_paths = file_paths
        self.max_dim = max_dim
        self.out_dir = out_dir
        self.resize_enabled = bool(resize_enabled)
        self.mode = mode
        self.target_w = target_w
        self.target_h = target_h
        self.percent = percent
        self.lock_aspect = lock_aspect
        self.center_crop = center_crop
        self.no_upscale = no_upscale
        self.out_format = out_format
        self.quality = quality
        self.strip_metadata = strip_metadata
        self.naming_pattern = naming_pattern
        self.in_memory_image = in_memory_image

    def run(self) -> None:
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        generated: list[Path] = []
        total = len(self.file_paths)
        for i, src in enumerate(self.file_paths):
            try:
                mem_img = self.in_memory_image if (i == 0 and total == 1) else None
                out_path = _process_single_resize(
                    src,
                    self.out_dir,
                    resize_enabled=self.resize_enabled,
                    mode=self.mode,
                    max_dim=self.max_dim,
                    target_w=self.target_w,
                    target_h=self.target_h,
                    percent=self.percent,
                    lock_aspect=self.lock_aspect,
                    center_crop=self.center_crop,
                    no_upscale=self.no_upscale,
                    out_format=self.out_format,
                    quality=self.quality,
                    strip_metadata=self.strip_metadata,
                    naming_pattern=self.naming_pattern,
                    in_memory_image=mem_img,
                )
                if out_path is not None:
                    generated.append(out_path)
            except Exception:
                pass
            self.progress.emit(i + 1, total)

        self.finished_batch.emit(generated)


class ResizePhotoDialog(QDialog):
    """
    Unified Resize dialog for both Gallery batch resizing and Single Image View
    ('Save Resized Copy...'). Supports Fit within (Long Edge), Exact WxH (with
    aspect lock and optional center-crop to fill), Percentage (%), saved custom
    presets in QSettings, live resolution/size readout, format/quality, and
    metadata stripping.
    """

    _SETTINGS_KEY = "resize/custom_presets"
    _session_state: dict[str, Any] | None = None

    def __init__(
        self,
        file_paths: Sequence[Path | str],
        parent: QWidget | None = None,
        *,
        in_memory_image: Image.Image | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export / Resize")
        self.setWindowIcon(make_wlpg_icon("gallery", 16))
        self.setMinimumWidth(480)

        self.file_paths = [Path(p) for p in file_paths]
        self._in_memory_image = in_memory_image
        self.generated_files: list[Path] = []
        self._updating_exact_inputs = False

        # Probe single-image source dimensions for live preview
        self._single_src_size: tuple[int, int] | None = None
        if len(self.file_paths) == 1:
            if self._in_memory_image is not None:
                self._single_src_size = self._in_memory_image.size
            elif self.file_paths[0].exists():
                try:
                    with Image.open(self.file_paths[0]) as im:
                        im_t = ImageOps.exif_transpose(im)
                        self._single_src_size = im_t.size
                except Exception:
                    self._single_src_size = None

        initial_dir = Path.home()
        for p in self.file_paths:
            if p.parent.exists():
                initial_dir = p.parent
                break

        self._custom_presets: list[dict[str, Any]] = self._load_custom_presets()

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(18, 16, 18, 16)
        vbox.setSpacing(12)

        from src.ui.theme_tokens import get_theme_tokens

        is_dark = bool(
            (str(getattr(parent, "current_theme", "")).strip().lower() == "dark")
            or (self.palette().window().color().lightness() < 128)
        )
        tok = get_theme_tokens(getattr(parent, "current_theme", None), is_dark=is_dark)

        # 1. Header + Friendly Subtitle
        lbl_title = QLabel("Export / Resize", self)
        lbl_title.setFont(QFont("Segoe UI", 14, QFont.DemiBold))
        lbl_title.setProperty("uiRole", "heading")
        vbox.addWidget(lbl_title)

        n_files = len(self.file_paths)
        lbl_sub = QLabel(
            f"Exporting {n_files} {'photo' if n_files == 1 else 'photos'} — originals on disk are never modified.",
            self,
        )
        lbl_sub.setFont(QFont("Segoe UI", 8.5))
        lbl_sub.setStyleSheet(f"color: {tok.text_secondary}; background: transparent;")
        vbox.addWidget(lbl_sub)

        # 2. Photo Size Group Box (Size preset selector first, custom dimensions tucked cleanly inside)
        class _ResizeModeContainer(QGroupBox):
            toggled = Signal(bool)

            def __init__(self, title: str, parent_w: QWidget | None = None) -> None:
                super().__init__(title, parent_w)
                self._checked_state: bool = False

            def isChecked(self) -> bool:
                return self._checked_state

            def setChecked(self, checked: bool) -> None:
                new_v = bool(checked)
                if self._checked_state != new_v:
                    self._checked_state = new_v
                    if not self.signalsBlocked():
                        self.toggled.emit(new_v)

        self.mode_group = _ResizeModeContainer("Photo Size", self)
        mode_group = self.mode_group
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setSpacing(8)

        # Size Preset Row (Original size default; Delete button appears for saved custom presets)
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        lbl_preset = QLabel("Size:", mode_group)
        lbl_preset.setFont(QFont("Segoe UI", 9))
        preset_row.addWidget(lbl_preset)

        self.cmb_size = QComboBox(mode_group)
        self.cmb_size.setFont(QFont("Segoe UI", 9))
        self._rebuild_presets_combo(select_index=0)  # Default: Original size (No resize)
        preset_row.addWidget(self.cmb_size, 1)

        self.btn_delete_preset = QPushButton("✕", mode_group)
        self.btn_delete_preset.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.btn_delete_preset.setMinimumWidth(30)
        self.btn_delete_preset.setFixedWidth(30)
        self.btn_delete_preset.setToolTip("Delete selected custom preset")
        self.btn_delete_preset.clicked.connect(self._on_delete_preset_clicked)
        self.btn_delete_preset.setVisible(False)
        preset_row.addWidget(self.btn_delete_preset)
        mode_layout.addLayout(preset_row)

        # Progressive-disclosure toggle for custom dimensions
        self.btn_toggle_custom_dims = QPushButton("▸ Customize dimensions && scaling", mode_group)
        self.btn_toggle_custom_dims.setFlat(True)
        self.btn_toggle_custom_dims.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_custom_dims.setStyleSheet(
            f"QPushButton {{ border: none; background: transparent; color: {tok.accent}; "
            "text-align: left; padding: 2px 0px; font-size: 8.5pt; }}"
            "QPushButton:hover { text-decoration: underline; }"
        )
        mode_layout.addWidget(self.btn_toggle_custom_dims)

        self._custom_dims_box = QWidget(mode_group)
        custom_lay = QVBoxLayout(self._custom_dims_box)
        custom_lay.setContentsMargins(6, 4, 0, 2)
        custom_lay.setSpacing(7)

        self.mode_btn_group = QButtonGroup(self)

        # Mode A: Fit within (Long Edge)
        row_long = QHBoxLayout()
        self.rb_mode_long_edge = QRadioButton("Fit within (Long Edge):", self._custom_dims_box)
        self.rb_mode_long_edge.setChecked(True)
        self.mode_btn_group.addButton(self.rb_mode_long_edge)
        row_long.addWidget(self.rb_mode_long_edge)

        self.txt_dim = QLineEdit(self._custom_dims_box)
        self.txt_dim.setValidator(QIntValidator(1, 99999, self))
        self.txt_dim.setText("1024")
        self.txt_dim.setFixedWidth(84)
        row_long.addWidget(self.txt_dim)
        row_long.addWidget(QLabel("px", self._custom_dims_box))
        row_long.addStretch(1)
        custom_lay.addLayout(row_long)

        # Mode B: Exact W x H
        row_exact = QHBoxLayout()
        self.rb_mode_exact = QRadioButton("Exact W × H:", self._custom_dims_box)
        self.mode_btn_group.addButton(self.rb_mode_exact)
        row_exact.addWidget(self.rb_mode_exact)

        init_w = str(self._single_src_size[0]) if self._single_src_size else "1024"
        init_h = str(self._single_src_size[1]) if self._single_src_size else "768"
        self.txt_exact_w = QLineEdit(init_w, self._custom_dims_box)
        self.txt_exact_w.setValidator(QIntValidator(1, 99999, self))
        self.txt_exact_w.setFixedWidth(72)
        self.txt_exact_h = QLineEdit(init_h, self._custom_dims_box)
        self.txt_exact_h.setValidator(QIntValidator(1, 99999, self))
        self.txt_exact_h.setFixedWidth(72)
        row_exact.addWidget(self.txt_exact_w)
        row_exact.addWidget(QLabel("×", self._custom_dims_box))
        row_exact.addWidget(self.txt_exact_h)
        row_exact.addWidget(QLabel("px", self._custom_dims_box))
        row_exact.addStretch(1)
        custom_lay.addLayout(row_exact)

        row_exact_opts = QHBoxLayout()
        row_exact_opts.setContentsMargins(22, 0, 0, 0)
        self.chk_lock_aspect = QCheckBox("Lock aspect ratio", self._custom_dims_box)
        self.chk_lock_aspect.setChecked(True)
        self.chk_center_crop = QCheckBox("Center-crop to fill", self._custom_dims_box)
        self.chk_center_crop.setChecked(False)
        self.chk_center_crop.setEnabled(False)
        row_exact_opts.addWidget(self.chk_lock_aspect)
        row_exact_opts.addWidget(self.chk_center_crop)
        row_exact_opts.addStretch(1)
        custom_lay.addLayout(row_exact_opts)

        # Mode C: Percentage (%)
        row_pct = QHBoxLayout()
        self.rb_mode_percent = QRadioButton("Percentage:", self._custom_dims_box)
        self.mode_btn_group.addButton(self.rb_mode_percent)
        row_pct.addWidget(self.rb_mode_percent)

        self.spin_percent = QSpinBox(self._custom_dims_box)
        self.spin_percent.setRange(10, 100)
        self.spin_percent.setValue(50)
        self.spin_percent.setSuffix("%")
        self.spin_percent.setFixedWidth(84)
        row_pct.addWidget(self.spin_percent)
        row_pct.addStretch(1)
        custom_lay.addLayout(row_pct)

        no_up_row = QHBoxLayout()
        self.chk_no_upscale = QCheckBox("Do not enlarge smaller photos", self._custom_dims_box)
        self.chk_no_upscale.setChecked(True)
        no_up_row.addWidget(self.chk_no_upscale)
        no_up_row.addStretch(1)

        self.btn_save_preset = QPushButton("+ Save Preset", self._custom_dims_box)
        self.btn_save_preset.setFont(QFont("Segoe UI", 8.5))
        self.btn_save_preset.setToolTip("Save these custom dimensions & format settings as a reusable preset")
        self.btn_save_preset.clicked.connect(self._on_save_preset_clicked)
        no_up_row.addWidget(self.btn_save_preset)
        custom_lay.addLayout(no_up_row)

        self._custom_dims_box.setVisible(False)
        mode_layout.addWidget(self._custom_dims_box)
        self.btn_toggle_custom_dims.clicked.connect(self._toggle_custom_dims_drawer)

        vbox.addWidget(mode_group)

        # 4. Output Format, Quality & Privacy
        out_group = QGroupBox("Format && Privacy", self)
        out_grid = QGridLayout(out_group)
        out_grid.setHorizontalSpacing(10)
        out_grid.setVerticalSpacing(6)

        out_grid.addWidget(QLabel("Format:", out_group), 0, 0)
        self.cmb_format = QComboBox(out_group)
        self.cmb_format.addItem("Same as original", "same")
        self.cmb_format.addItem("JPEG", "jpeg")
        self.cmb_format.addItem("WebP", "webp")
        self.cmb_format.addItem("PNG", "png")
        out_grid.addWidget(self.cmb_format, 0, 1)

        out_grid.addWidget(QLabel("Quality:", out_group), 0, 2)
        q_row = QHBoxLayout()
        self.slider_quality = QSlider(Qt.Horizontal, out_group)
        self.slider_quality.setRange(50, 100)
        self.slider_quality.setValue(88)
        self.lbl_quality_val = QLabel("88%", out_group)
        self.lbl_quality_val.setFixedWidth(36)
        q_row.addWidget(self.slider_quality, 1)
        q_row.addWidget(self.lbl_quality_val)
        out_grid.addLayout(q_row, 0, 3)

        self.chk_strip_metadata = QCheckBox("Strip GPS && EXIF metadata", out_group)
        self.chk_strip_metadata.setChecked(False)
        out_grid.addWidget(self.chk_strip_metadata, 1, 0, 1, 4)

        out_grid.addWidget(QLabel("File name:", out_group), 2, 0)
        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        self.cmb_naming = QComboBox(out_group)
        self.cmb_naming.addItem("Auto (name + size / _export)", "auto")
        self.cmb_naming.addItem("name_export", "{stem}_export")
        self.cmb_naming.addItem("name_size (e.g. _1024px)", "{stem}_{size}")
        self.cmb_naming.addItem("name_edited", "{stem}_edited")
        self.cmb_naming.addItem("Keep original name", "{stem}")
        self.cmb_naming.addItem("Custom...", "__custom__")
        name_row.addWidget(self.cmb_naming, 1)

        self.txt_naming_custom = QLineEdit(out_group)
        self.txt_naming_custom.setPlaceholderText("{stem}_{size}")
        self.txt_naming_custom.setToolTip("Tokens: {stem}, {size}, {w}, {h}, {date}")
        self.txt_naming_custom.setVisible(False)
        name_row.addWidget(self.txt_naming_custom, 1)
        out_grid.addLayout(name_row, 2, 1, 1, 3)

        vbox.addWidget(out_group)

        # 5. Live Output Summary Pill
        self.lbl_preview_summary = QLabel("", self)
        self.lbl_preview_summary.setProperty("uiRole", "muted")
        self.lbl_preview_summary.setWordWrap(True)
        self.lbl_preview_summary.setStyleSheet(
            f"background-color: {tok.bg_card}; color: {tok.text_primary}; "
            f"border: 1px solid {tok.border_subtle}; border-radius: 6px; padding: 7px 10px; font-size: 9pt;"
        )
        vbox.addWidget(self.lbl_preview_summary)

        # 6. Save to: Path + Browse...
        lbl_save = QLabel("Save to folder:", self)
        lbl_save.setFont(QFont("Segoe UI", 9))
        vbox.addWidget(lbl_save)

        save_row = QHBoxLayout()
        save_row.setSpacing(8)
        self.txt_save_path = QLineEdit(str(initial_dir), self)
        self.txt_save_path.setFont(QFont("Segoe UI", 9))
        save_row.addWidget(self.txt_save_path, 1)

        self.btn_browse = QPushButton("Browse...", self)
        self.btn_browse.setFont(QFont("Segoe UI", 9))
        self.btn_browse.clicked.connect(self._on_browse_clicked)
        save_row.addWidget(self.btn_browse)
        vbox.addLayout(save_row)

        # Progress bar (visible during batch resizing)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setVisible(False)
        vbox.addWidget(self.progress_bar)

        # 7. Action Buttons (Right-aligned with Primary Export CTA)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch(1)

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setFont(QFont("Segoe UI", 9))
        self.btn_cancel.setMinimumWidth(84)
        self.btn_cancel.setFixedHeight(30)
        self.btn_cancel.clicked.connect(self.reject)

        export_label = "Export" if len(self.file_paths) == 1 else f"Export {len(self.file_paths)} Photos"
        self.btn_resize = QPushButton(export_label, self)
        self.btn_resize.setFont(QFont("Segoe UI", 9, QFont.DemiBold))
        self.btn_resize.setDefault(True)
        self.btn_resize.setMinimumWidth(120)
        self.btn_resize.setFixedHeight(30)
        self.btn_resize.setStyleSheet(
            f"QPushButton {{ background-color: {tok.accent}; color: {tok.text_on_accent}; "
            f"border: none; border-radius: 5px; padding: 0 16px; }}"
            f"QPushButton:hover {{ background-color: {tok.accent_hover}; }}"
            "QPushButton:disabled { opacity: 0.5; }"
        )
        self.btn_resize.clicked.connect(self._on_resize_clicked)

        btn_row.addWidget(self.btn_resize)
        btn_row.addWidget(self.btn_cancel)
        vbox.addLayout(btn_row)

        self._worker: _ResizeWorker | None = None

        # Wire signals
        self.mode_group.toggled.connect(self._on_resize_toggled)
        self.cmb_size.currentIndexChanged.connect(self._on_size_combo_changed)
        self.txt_dim.textChanged.connect(self._on_dim_text_changed)
        self.rb_mode_long_edge.toggled.connect(self._on_mode_changed)
        self.rb_mode_exact.toggled.connect(self._on_mode_changed)
        self.rb_mode_percent.toggled.connect(self._on_mode_changed)
        self.txt_exact_w.textEdited.connect(self._on_exact_w_edited)
        self.txt_exact_h.textEdited.connect(self._on_exact_h_edited)
        self.chk_lock_aspect.toggled.connect(self._on_lock_aspect_toggled)
        self.chk_center_crop.toggled.connect(self._update_preview_summary)
        self.spin_percent.valueChanged.connect(self._update_preview_summary)
        self.chk_no_upscale.toggled.connect(self._update_preview_summary)
        self.cmb_format.currentIndexChanged.connect(self._on_format_or_quality_changed)
        self.slider_quality.valueChanged.connect(self._on_format_or_quality_changed)
        self.cmb_naming.currentIndexChanged.connect(self._on_naming_combo_changed)
        self.txt_naming_custom.textChanged.connect(self._update_preview_summary)

        self._restore_session_state()
        self._sync_mode_enabled_states()
        self._on_format_or_quality_changed()
        self._update_preview_summary()
        self.adjustSize()

    # ------------------------------------------------------------------
    # Preset Persistence
    # ------------------------------------------------------------------
    def _load_custom_presets(self) -> list[dict[str, Any]]:
        settings = QSettings("PhotoGallery64", "PhotoGallery64")
        raw = settings.value(self._SETTINGS_KEY, "[]")
        try:
            items = json.loads(raw) if isinstance(raw, str) else list(raw or [])
        except Exception:
            items = []
        return [it for it in items if isinstance(it, dict) and "name" in it]

    def _save_custom_presets(self) -> None:
        settings = QSettings("PhotoGallery64", "PhotoGallery64")
        settings.setValue(self._SETTINGS_KEY, json.dumps(self._custom_presets))

    def _rebuild_presets_combo(self, select_index: int | None = None, select_name: str | None = None) -> None:
        self.cmb_size.blockSignals(True)
        try:
            self.cmb_size.clear()
            self.cmb_size.addItem("Original size (No resize)", {"resize_enabled": False})
            self.cmb_size.addItem("Small — 640 px", {"resize_enabled": True, "mode": "long_edge", "dim": 640})
            self.cmb_size.addItem("Medium — 1024 px", {"resize_enabled": True, "mode": "long_edge", "dim": 1024})
            self.cmb_size.addItem("Full HD — 1920 px", {"resize_enabled": True, "mode": "long_edge", "dim": 1920})
            self.cmb_size.addItem("4K — 3840 px", {"resize_enabled": True, "mode": "long_edge", "dim": 3840})
            self.cmb_size.addItem("Custom", None)
            for p in self._custom_presets:
                self.cmb_size.addItem(str(p["name"]), {"custom": True, **p})

            if select_name is not None:
                idx = self.cmb_size.findText(select_name)
                self.cmb_size.setCurrentIndex(idx if idx >= 0 else 5)
            elif select_index is not None:
                self.cmb_size.setCurrentIndex(select_index)
        finally:
            self.cmb_size.blockSignals(False)

    def save_custom_preset(self, name: str) -> None:
        cfg = self._current_resize_config()
        entry: dict[str, Any] = {
            "name": name.strip(),
            "resize_enabled": bool(cfg["resize_enabled"]),
            "mode": cfg["mode"],
            "no_upscale": bool(cfg["no_upscale"]),
            "out_format": str(cfg["out_format"]),
            "quality": int(cfg["quality"]),
            "strip_metadata": bool(cfg["strip_metadata"]),
        }
        if cfg["mode"] == "long_edge":
            entry["dim"] = cfg["max_dim"]
        elif cfg["mode"] == "exact":
            entry["w"] = cfg["target_w"]
            entry["h"] = cfg["target_h"]
            entry["lock_aspect"] = cfg["lock_aspect"]
            entry["center_crop"] = cfg["center_crop"]
        else:
            entry["percent"] = cfg["percent"]
        self._custom_presets = [p for p in self._custom_presets if p.get("name") != entry["name"]]
        self._custom_presets.append(entry)
        self._save_custom_presets()
        self._rebuild_presets_combo(select_name=entry["name"])
        self.btn_delete_preset.setVisible(True)

    def _on_save_preset_clicked(self) -> None:
        cfg = self._current_resize_config()
        if not cfg["resize_enabled"]:
            fmt = str(cfg["out_format"]).upper() if cfg["out_format"] != "same" else "Original"
            default_name = f"Original ({fmt})"
        elif cfg["mode"] == "long_edge":
            default_name = f"Custom {cfg['max_dim']} px"
        elif cfg["mode"] == "exact":
            default_name = f"Custom {cfg['target_w']}×{cfg['target_h']}"
        else:
            default_name = f"Scale {cfg['percent']}%"
        name, ok = QInputDialog.getText(self, "Save Export Preset", "Preset name:", text=default_name)
        if ok and name.strip():
            self.save_custom_preset(name.strip())

    def _on_delete_preset_clicked(self) -> None:
        data = self.cmb_size.currentData()
        if not isinstance(data, dict) or not data.get("custom"):
            return
        name = self.cmb_size.currentText()
        self._custom_presets = [p for p in self._custom_presets if p.get("name") != name]
        self._save_custom_presets()
        self._rebuild_presets_combo(select_index=2)
        self._on_size_combo_changed(2)

    # ------------------------------------------------------------------
    # Mode & Input Synchronization
    # ------------------------------------------------------------------
    def active_mode(self) -> str:
        if self.rb_mode_exact.isChecked():
            return "exact"
        if self.rb_mode_percent.isChecked():
            return "percent"
        return "long_edge"

    def _toggle_custom_dims_drawer(self) -> None:
        vis = self._custom_dims_box.isHidden()
        self._custom_dims_box.setVisible(vis)
        self.btn_toggle_custom_dims.setText(
            "▾ Customize dimensions && scaling" if vis else "▸ Customize dimensions && scaling"
        )
        self.adjustSize()

    def _on_resize_toggled(self, checked: bool) -> None:
        if not checked:
            self.cmb_size.blockSignals(True)
            self.cmb_size.setCurrentIndex(0)  # Original size (No resize)
            self.cmb_size.blockSignals(False)
            self.btn_delete_preset.setVisible(False)
            self.btn_toggle_custom_dims.setVisible(False)
            self._custom_dims_box.setVisible(False)
        else:
            self.btn_toggle_custom_dims.setVisible(True)
            if self.cmb_size.currentIndex() == 0:
                self.cmb_size.blockSignals(True)
                self.cmb_size.setCurrentIndex(5)  # Custom
                self.cmb_size.blockSignals(False)
        self._sync_mode_enabled_states()
        self._update_preview_summary()
        self.adjustSize()

    def _sync_mode_enabled_states(self) -> None:
        is_resizing = self.mode_group.isChecked()
        self.btn_toggle_custom_dims.setVisible(is_resizing)
        if not is_resizing:
            self._custom_dims_box.setVisible(False)
            return
        mode = self.active_mode()
        self.txt_dim.setEnabled(mode == "long_edge")
        self.txt_exact_w.setEnabled(mode == "exact")
        self.txt_exact_h.setEnabled(mode == "exact")
        self.chk_lock_aspect.setEnabled(mode == "exact")
        self.chk_center_crop.setEnabled(mode == "exact" and not self.chk_lock_aspect.isChecked())
        self.spin_percent.setEnabled(mode == "percent")

    def _on_mode_changed(self) -> None:
        if not self.mode_group.isChecked():
            self.mode_group.blockSignals(True)
            self.mode_group.setChecked(True)
            self.mode_group.blockSignals(False)
            if self.cmb_size.currentIndex() == 0:
                self.cmb_size.blockSignals(True)
                self.cmb_size.setCurrentIndex(5)
                self.cmb_size.blockSignals(False)
        self._sync_mode_enabled_states()
        self._update_preview_summary()

    def _on_size_combo_changed(self, _index: int) -> None:
        data = self.cmb_size.currentData()
        is_custom_saved = isinstance(data, dict) and bool(data.get("custom"))
        self.btn_delete_preset.setVisible(is_custom_saved)
        if not isinstance(data, dict):
            # "Custom" selected -> enable resize and expand custom dimensions drawer
            self.mode_group.blockSignals(True)
            self.mode_group.setChecked(True)
            self.mode_group.blockSignals(False)
            self.btn_toggle_custom_dims.setVisible(True)
            self._custom_dims_box.setVisible(True)
            self.btn_toggle_custom_dims.setText("▾ Customize dimensions && scaling")
            self._sync_mode_enabled_states()
            self._update_preview_summary()
            self.adjustSize()
            return

        resize_enabled = bool(data.get("resize_enabled", True))
        self.mode_group.blockSignals(True)
        self.mode_group.setChecked(resize_enabled)
        self.mode_group.blockSignals(False)

        if "no_upscale" in data:
            self.chk_no_upscale.setChecked(bool(data["no_upscale"]))
        if "out_format" in data:
            fmt_idx = self.cmb_format.findData(str(data["out_format"]))
            if fmt_idx >= 0:
                self.cmb_format.blockSignals(True)
                self.cmb_format.setCurrentIndex(fmt_idx)
                self.cmb_format.blockSignals(False)
        if "quality" in data:
            self.slider_quality.blockSignals(True)
            self.slider_quality.setValue(int(data["quality"]))
            self.slider_quality.blockSignals(False)
        if "strip_metadata" in data:
            self.chk_strip_metadata.setChecked(bool(data["strip_metadata"]))

        mode = data.get("mode", "long_edge")
        if mode == "long_edge" and "dim" in data:
            self.rb_mode_long_edge.setChecked(True)
            self.txt_dim.blockSignals(True)
            self.txt_dim.setText(str(int(data.get("dim", 1024))))
            self.txt_dim.blockSignals(False)
        elif mode == "exact":
            self.rb_mode_exact.setChecked(True)
            self._updating_exact_inputs = True
            try:
                self.txt_exact_w.setText(str(int(data.get("w", 1024))))
                self.txt_exact_h.setText(str(int(data.get("h", 768))))
                self.chk_lock_aspect.setChecked(bool(data.get("lock_aspect", True)))
                self.chk_center_crop.setChecked(bool(data.get("center_crop", False)))
            finally:
                self._updating_exact_inputs = False
        elif mode == "percent":
            self.rb_mode_percent.setChecked(True)
            self.spin_percent.setValue(int(data.get("percent", 50)))

        self._sync_mode_enabled_states()
        self._on_format_or_quality_changed()
        self._update_preview_summary()

    def _on_dim_text_changed(self, text: str) -> None:
        val = int(text) if text.strip().isdigit() else None
        self.cmb_size.blockSignals(True)
        preset_map = {640: 1, 1024: 2, 1920: 3, 3840: 4}
        self.cmb_size.setCurrentIndex(preset_map.get(val, 5))
        self.cmb_size.blockSignals(False)
        self.btn_delete_preset.setVisible(False)
        self._update_preview_summary()

    def _on_exact_w_edited(self, text: str) -> None:
        if self._updating_exact_inputs:
            return
        if self.chk_lock_aspect.isChecked() and self._single_src_size and text.strip().isdigit():
            w_val = int(text.strip())
            sw, sh = self._single_src_size
            if w_val > 0 and sw > 0 and sh > 0:
                self._updating_exact_inputs = True
                try:
                    self.txt_exact_h.setText(str(max(1, int(round(w_val * (sh / float(sw)))))))
                finally:
                    self._updating_exact_inputs = False
        self._update_preview_summary()

    def _on_exact_h_edited(self, text: str) -> None:
        if self._updating_exact_inputs:
            return
        if self.chk_lock_aspect.isChecked() and self._single_src_size and text.strip().isdigit():
            h_val = int(text.strip())
            sw, sh = self._single_src_size
            if h_val > 0 and sw > 0 and sh > 0:
                self._updating_exact_inputs = True
                try:
                    self.txt_exact_w.setText(str(max(1, int(round(h_val * (sw / float(sh)))))))
                finally:
                    self._updating_exact_inputs = False
        self._update_preview_summary()

    def _on_lock_aspect_toggled(self, checked: bool) -> None:
        self.chk_center_crop.setEnabled(self.active_mode() == "exact" and not checked)
        if checked:
            self.chk_center_crop.setChecked(False)
        self._update_preview_summary()

    def _on_format_or_quality_changed(self) -> None:
        q = self.slider_quality.value()
        self.lbl_quality_val.setText(f"{q}%")
        fmt_choice = str(self.cmb_format.currentData() or "same")
        src_ext = self.file_paths[0].suffix if self.file_paths else ".jpg"
        pil_fmt, _ = _resolve_output_format_and_ext(src_ext, fmt_choice)
        self.slider_quality.setEnabled(pil_fmt != "PNG")
        self._update_preview_summary()

    def _on_naming_combo_changed(self) -> None:
        is_custom = str(self.cmb_naming.currentData() or "auto") == "__custom__"
        self.txt_naming_custom.setVisible(is_custom)
        if is_custom and not self.txt_naming_custom.text().strip():
            self.txt_naming_custom.setText("{stem}_{size}")
        self._update_preview_summary()

    def _effective_naming_pattern(self) -> str:
        val = str(self.cmb_naming.currentData() or "auto")
        if val == "__custom__":
            return self.txt_naming_custom.text().strip() or "{stem}_export"
        return val

    def _save_session_state(self) -> None:
        cfg = self._current_resize_config()
        ResizePhotoDialog._session_state = {
            **cfg,
            "size_text": self.cmb_size.currentText(),
            "naming_data": str(self.cmb_naming.currentData() or "auto"),
            "naming_custom": self.txt_naming_custom.text(),
            "custom_dims_open": not self._custom_dims_box.isHidden(),
        }

    def _restore_session_state(self) -> None:
        st = ResizePhotoDialog._session_state
        if not isinstance(st, dict):
            return
        size_text = str(st.get("size_text") or "")
        s_idx = self.cmb_size.findText(size_text) if size_text else -1
        if s_idx >= 0:
            self.cmb_size.blockSignals(True)
            self.cmb_size.setCurrentIndex(s_idx)
            self.cmb_size.blockSignals(False)
            data = self.cmb_size.currentData()
            self.btn_delete_preset.setVisible(isinstance(data, dict) and bool(data.get("custom")))

        self.mode_group.blockSignals(True)
        self.mode_group.setChecked(bool(st.get("resize_enabled", False)))
        self.mode_group.blockSignals(False)

        mode = str(st.get("mode") or "long_edge")
        if mode == "exact":
            self.rb_mode_exact.setChecked(True)
        elif mode == "percent":
            self.rb_mode_percent.setChecked(True)
        else:
            self.rb_mode_long_edge.setChecked(True)

        self.txt_dim.blockSignals(True)
        self.txt_dim.setText(str(int(st.get("max_dim", 1024))))
        self.txt_dim.blockSignals(False)

        self._updating_exact_inputs = True
        try:
            if "target_w" in st and len(self.file_paths) != 1:
                self.txt_exact_w.setText(str(int(st["target_w"])))
            if "target_h" in st and len(self.file_paths) != 1:
                self.txt_exact_h.setText(str(int(st["target_h"])))
        finally:
            self._updating_exact_inputs = False

        self.spin_percent.setValue(int(st.get("percent", 50)))
        self.chk_lock_aspect.setChecked(bool(st.get("lock_aspect", True)))
        self.chk_center_crop.setChecked(bool(st.get("center_crop", False)))
        self.chk_no_upscale.setChecked(bool(st.get("no_upscale", True)))

        fmt_idx = self.cmb_format.findData(str(st.get("out_format", "same")))
        if fmt_idx >= 0:
            self.cmb_format.setCurrentIndex(fmt_idx)
        self.slider_quality.setValue(int(st.get("quality", 88)))
        self.chk_strip_metadata.setChecked(bool(st.get("strip_metadata", False)))

        n_idx = self.cmb_naming.findData(str(st.get("naming_data", "auto")))
        if n_idx >= 0:
            self.cmb_naming.setCurrentIndex(n_idx)
        self.txt_naming_custom.setText(str(st.get("naming_custom") or ""))
        self.txt_naming_custom.setVisible(str(st.get("naming_data")) == "__custom__")

        if bool(st.get("custom_dims_open", False)) and self.mode_group.isChecked():
            self._custom_dims_box.setVisible(True)
            self.btn_toggle_custom_dims.setText("▾ Customize dimensions && scaling")

    def _current_resize_config(self) -> dict[str, Any]:
        dim_str = self.txt_dim.text().strip()
        w_str = self.txt_exact_w.text().strip()
        h_str = self.txt_exact_h.text().strip()
        return {
            "resize_enabled": self.mode_group.isChecked(),
            "mode": self.active_mode(),
            "max_dim": int(dim_str) if dim_str.isdigit() and int(dim_str) > 0 else 1024,
            "target_w": int(w_str) if w_str.isdigit() and int(w_str) > 0 else 1024,
            "target_h": int(h_str) if h_str.isdigit() and int(h_str) > 0 else 768,
            "percent": int(self.spin_percent.value()),
            "lock_aspect": self.chk_lock_aspect.isChecked(),
            "center_crop": self.chk_center_crop.isChecked(),
            "no_upscale": self.chk_no_upscale.isChecked(),
            "out_format": str(self.cmb_format.currentData() or "same"),
            "quality": int(self.slider_quality.value()),
            "strip_metadata": self.chk_strip_metadata.isChecked(),
            "naming_pattern": self._effective_naming_pattern(),
        }

    def _update_preview_summary(self) -> None:
        cfg = self._current_resize_config()
        if len(self.file_paths) == 1 and self._single_src_size is not None:
            sw, sh = self._single_src_size
            if cfg["resize_enabled"]:
                ow, oh, _ = compute_resize_target(
                    sw,
                    sh,
                    mode=cfg["mode"],
                    max_dim=cfg["max_dim"],
                    target_w=cfg["target_w"],
                    target_h=cfg["target_h"],
                    percent=cfg["percent"],
                    lock_aspect=cfg["lock_aspect"],
                    center_crop=cfg["center_crop"],
                    no_upscale=cfg["no_upscale"],
                )
            else:
                ow, oh = sw, sh
            _, ext = _resolve_output_format_and_ext(self.file_paths[0].suffix or ".jpg", cfg["out_format"])
            out_stem = _resolve_export_filename_stem(
                self.file_paths[0],
                naming_pattern=cfg["naming_pattern"],
                resize_enabled=cfg["resize_enabled"],
                mode=cfg["mode"],
                out_w=ow,
                out_h=oh,
                ext=ext,
            )
            est = _format_bytes_human(_estimate_resized_file_bytes(ow, oh, ext, cfg["quality"]))
            self.lbl_preview_summary.setText(
                f"File: {out_stem}{ext}   •   Original: {sw} × {sh} px  →  Output: {ow} × {oh} px ({est})"
            )
        else:
            count = len(self.file_paths)
            if not cfg["resize_enabled"]:
                detail = "Original resolution"
            elif cfg["mode"] == "long_edge":
                detail = f"Longest edge capped at {cfg['max_dim']} px"
            elif cfg["mode"] == "exact":
                sub = "center-crop" if (not cfg["lock_aspect"] and cfg["center_crop"]) else (
                    "fit" if cfg["lock_aspect"] else "exact"
                )
                detail = f"Target size: {cfg['target_w']} × {cfg['target_h']} px ({sub})"
            else:
                detail = f"Scaled to {cfg['percent']}%"
            sample_line = ""
            if self.file_paths:
                _, s_ext = _resolve_output_format_and_ext(self.file_paths[0].suffix or ".jpg", cfg["out_format"])
                s_stem = _resolve_export_filename_stem(
                    self.file_paths[0],
                    naming_pattern=cfg["naming_pattern"],
                    resize_enabled=cfg["resize_enabled"],
                    mode=cfg["mode"],
                    out_w=cfg["max_dim"] if cfg["mode"] == "long_edge" else cfg["target_w"],
                    out_h=cfg["max_dim"] if cfg["mode"] == "long_edge" else cfg["target_h"],
                    ext=s_ext,
                )
                sample_line = f" • e.g. {s_stem}{s_ext}"
            self.lbl_preview_summary.setText(f"{count} photos selected • {detail}{sample_line}")

    def _on_browse_clicked(self) -> None:
        curr = self.txt_save_path.text()
        chosen = QFileDialog.getExistingDirectory(self, "Select Folder to Save Resized Photos", curr)
        if chosen:
            self.txt_save_path.setText(chosen)

    def _on_resize_clicked(self) -> None:
        self._save_session_state()
        cfg = self._current_resize_config()
        out_dir = Path(self.txt_save_path.text().strip() or ".")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        self.btn_resize.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self._worker = _ResizeWorker(
            self.file_paths,
            cfg["max_dim"],
            out_dir,
            self,
            resize_enabled=cfg["resize_enabled"],
            mode=cfg["mode"],
            target_w=cfg["target_w"],
            target_h=cfg["target_h"],
            percent=cfg["percent"],
            lock_aspect=cfg["lock_aspect"],
            center_crop=cfg["center_crop"],
            no_upscale=cfg["no_upscale"],
            out_format=cfg["out_format"],
            quality=cfg["quality"],
            strip_metadata=cfg["strip_metadata"],
            naming_pattern=cfg["naming_pattern"],
            in_memory_image=self._in_memory_image,
        )
        self._worker.progress.connect(self._on_resize_progress)
        self._worker.finished_batch.connect(self._on_resize_finished)
        self._worker.start()

    def _on_resize_progress(self, current: int, total: int) -> None:
        if total > 0:
            self.progress_bar.setValue(int(100 * current / total))

    def _on_resize_finished(self, generated: list[Path]) -> None:
        self.generated_files = generated
        self.accept()

    def execute_resize(self) -> list[Path]:
        """Synchronously resizes all input photos using the current dialog configuration."""
        self._save_session_state()
        cfg = self._current_resize_config()
        out_dir = Path(self.txt_save_path.text().strip() or ".")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        generated: list[Path] = []
        total = len(self.file_paths)
        for i, src in enumerate(self.file_paths):
            try:
                mem_img = self._in_memory_image if (i == 0 and total == 1) else None
                out_path = _process_single_resize(
                    src,
                    out_dir,
                    resize_enabled=cfg["resize_enabled"],
                    mode=cfg["mode"],
                    max_dim=cfg["max_dim"],
                    target_w=cfg["target_w"],
                    target_h=cfg["target_h"],
                    percent=cfg["percent"],
                    lock_aspect=cfg["lock_aspect"],
                    center_crop=cfg["center_crop"],
                    no_upscale=cfg["no_upscale"],
                    out_format=cfg["out_format"],
                    quality=cfg["quality"],
                    strip_metadata=cfg["strip_metadata"],
                    naming_pattern=cfg["naming_pattern"],
                    in_memory_image=mem_img,
                )
                if out_path is not None:
                    generated.append(out_path)
            except Exception:
                pass

        self.generated_files = generated
        return generated


class AboutDialog(QDialog):
    """About dialog for Essentials Photo Gallery."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About Essentials Photo Gallery")
        self.setObjectName("AboutDialog")
        self.setFixedSize(480, 310)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        from src.ui.theme_tokens import get_theme_tokens

        theme_name = getattr(parent, "current_theme", None)
        self._is_dark = bool(
            (str(theme_name or "").strip().lower() == "dark")
            or (self.palette().window().color().lightness() < 128)
        )
        self._theme_tokens = get_theme_tokens(theme_name, is_dark=self._is_dark)
        tok = self._theme_tokens

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(24, 20, 24, 18)
        vbox.setSpacing(12)

        # Header with Logo & Title
        header_hbox = QHBoxLayout()
        header_hbox.setSpacing(16)

        lbl_icon = QLabel(self)
        try:
            from src.ui.icon_provider import make_wlpg_icon
            lbl_icon.setPixmap(make_wlpg_icon("app_logo", 48).pixmap(48, 48))
        except Exception:
            try:
                from src.ui.icon_provider import make_wlpg_icon
                lbl_icon.setPixmap(make_wlpg_icon("flower", 48).pixmap(48, 48))
            except Exception:
                lbl_icon.setText("🌸")
                lbl_icon.setStyleSheet("font-size: 32pt;")

        header_hbox.addWidget(lbl_icon, 0, Qt.AlignTop)

        info_vbox = QVBoxLayout()
        info_vbox.setSpacing(4)

        lbl_title = QLabel("Essentials Photo Gallery", self)
        title_color = tok.text_primary if tok.is_dark else "#1E395B"
        lbl_title.setStyleSheet(
            f"font-family: 'Segoe UI', sans-serif; font-size: 13pt; font-weight: bold; color: {title_color}; background: transparent;"
        )
        info_vbox.addWidget(lbl_title)

        from src.version import APP_VERSION

        lbl_ver = QLabel(f"Version {APP_VERSION} (64-bit)", self)
        ver_color = tok.text_secondary if tok.is_dark else "#55687D"
        lbl_ver.setStyleSheet(
            f"font-family: 'Segoe UI', sans-serif; font-size: 8.5pt; color: {ver_color}; background: transparent;"
        )
        info_vbox.addWidget(lbl_ver)

        lbl_desc = QLabel(
            "Essentials Photo Gallery recreates the beloved fluid ribbon interface and Aero experience "
            "with modern AI intelligence, hardware-accelerated monocular depth & portrait lighting, "
            "and lossless non-destructive photo management.",
            self,
        )
        lbl_desc.setWordWrap(True)
        desc_color = tok.text_primary if tok.is_dark else "#333333"
        lbl_desc.setStyleSheet(
            f"font-family: 'Segoe UI', sans-serif; font-size: 9pt; color: {desc_color}; margin-top: 4px; background: transparent;"
        )
        info_vbox.addWidget(lbl_desc)

        header_hbox.addLayout(info_vbox, 1)
        vbox.addLayout(header_hbox)

        # Separator line
        sep = QFrame(self)
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet(f"color: {tok.border_subtle if tok.is_dark else '#DDE5EE'};")
        vbox.addWidget(sep)

        # Technical details
        lbl_tech = QLabel(
            "Powered by PySide6 (Qt 6.8), OpenCV, ONNX Runtime (DirectML / CPU), "
            "and SQLite WAL.\n© 2026 Essentials Photo Gallery Contributors.",
            self,
        )
        tech_color = tok.text_secondary if tok.is_dark else "#667788"
        lbl_tech.setStyleSheet(
            f"font-family: 'Segoe UI', sans-serif; font-size: 8pt; color: {tech_color}; background: transparent;"
        )
        vbox.addWidget(lbl_tech)

        vbox.addStretch(1)

        # Update status label + Check for Updates & OK buttons
        self.lbl_update_status = QLabel("", self)
        status_color = "#60A5FA" if tok.is_dark else "#3A6EA5"
        self.lbl_update_status.setStyleSheet(
            f"font-family: 'Segoe UI', sans-serif; font-size: 8.5pt; color: {status_color}; background: transparent;"
        )
        self.lbl_update_status.setWordWrap(True)
        vbox.addWidget(self.lbl_update_status)

        btn_box = QHBoxLayout()
        self.btn_check_updates = QPushButton("Check for Updates...", self)
        self.btn_check_updates.setFixedHeight(26)
        self.btn_check_updates.clicked.connect(self._on_check_updates)
        btn_box.addWidget(self.btn_check_updates)
        btn_box.addStretch(1)
        self._update_thread = None
        btn_ok = QPushButton("OK", self)
        btn_ok.setFixedSize(85, 26)
        btn_ok.setDefault(True)
        if tok.is_dark:
            btn_ok.setStyleSheet(
                f"""
                QPushButton {{
                    background: {tok.accent};
                    border: 1px solid {tok.accent_hover};
                    border-radius: 3px;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 9pt;
                    color: {tok.text_on_accent};
                }}
                QPushButton:hover {{
                    background: {tok.accent_hover};
                }}
                """
            )
        else:
            btn_ok.setStyleSheet(
                """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FDFDFD, stop:1 #E5EBF2);
                    border: 1px solid #7DA2CE;
                    border-radius: 3px;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 9pt;
                    color: #1E395B;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F0F6FD, stop:1 #D9E8F8);
                    border: 1px solid #4888C8;
                }
                QPushButton:pressed {
                    background: #CCE0F4;
                    border: 1px solid #3377B8;
                }
                """
            )
        btn_ok.clicked.connect(self.accept)
        btn_box.addWidget(btn_ok)
        vbox.addLayout(btn_box)

        # Scope strictly to #AboutDialog so child QMessageBox / QProgressDialog
        # never inherit a forced white QDialog background on top of Dark Theme text.
        bg_dlg = tok.bg_surface if tok.is_dark else "#FFFFFF"
        self.setStyleSheet(
            f"""
            QDialog#AboutDialog {{
                background-color: {bg_dlg};
            }}
            """
        )

    def _on_check_updates(self) -> None:
        if self._update_thread is not None and self._update_thread.isRunning():
            return
        from PySide6.QtCore import QThread, Signal as QtSignal

        class _UpdateCheckWorker(QThread):
            result_ready = QtSignal(bool, object, str)

            def run(self) -> None:
                try:
                    from src.version import fetch_latest_release
                    info = fetch_latest_release()
                    self.result_ready.emit(True, info, "")
                except Exception as exc:
                    self.result_ready.emit(False, None, str(exc))

        self.btn_check_updates.setEnabled(False)
        self.lbl_update_status.setText("Checking GitHub Releases for updates...")
        worker = _UpdateCheckWorker(self)

        def _on_result(ok: bool, info: object, err: str) -> None:
            self._update_thread = None
            self.btn_check_updates.setEnabled(True)
            if not ok or not isinstance(info, dict):
                self.lbl_update_status.setText("Could not check for updates (offline or GitHub unreachable).")
                return
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            from src.version import APP_VERSION, is_newer_version, download_installer_exe

            latest_ver = str(info.get("version") or "")
            if not is_newer_version(latest_ver, APP_VERSION):
                self.lbl_update_status.setText(f"✓ You are running the latest version ({APP_VERSION}).")
                return

            self.lbl_update_status.setText(f"Update available: v{latest_ver} (current: {APP_VERSION}).")
            installer_url = str(info.get("installer_url") or "")
            html_url = str(info.get("html_url") or "")
            release_body = str(info.get("body") or "").strip()
            tok = self._theme_tokens
            msg = QMessageBox(self)
            msg.setWindowTitle("Update Available")
            msg.setIcon(QMessageBox.Information)
            msg.setText(f"A new version of Essentials Photo Gallery (v{latest_ver}) is available.")
            info_text = f"You are currently running version {APP_VERSION}."
            if installer_url:
                info_text += "\n\nThe app will close automatically once the installer is downloaded and ready to run."
            msg.setInformativeText(info_text)
            if release_body:
                msg.setDetailedText(release_body[:3000])
            msg.setStyleSheet(
                f"QMessageBox {{ background-color: {tok.bg_surface}; color: {tok.text_primary}; }} "
                f"QMessageBox QLabel {{ color: {tok.text_primary}; background: transparent; }} "
                f"QMessageBox QTextEdit {{ background-color: {tok.bg_card}; color: {tok.text_primary}; "
                f"border: 1px solid {tok.border_subtle}; border-radius: 4px; }} "
                f"QMessageBox QPushButton {{ background-color: {tok.bg_elevated}; color: {tok.text_primary}; "
                f"border: 1px solid {tok.border_subtle}; border-radius: 4px; padding: 5px 14px; min-width: 76px; }} "
                f"QMessageBox QPushButton:hover {{ background-color: {tok.bg_hover}; border-color: {tok.accent}; }}"
            )
            btn_dl = msg.addButton("Download && Install", QMessageBox.AcceptRole) if installer_url else None
            btn_web = msg.addButton("Open Release Page", QMessageBox.ActionRole)
            msg.addButton(QMessageBox.Close)
            msg.exec()
            clicked = msg.clickedButton()
            if btn_dl is not None and clicked == btn_dl:
                from PySide6.QtCore import QTimer
                from PySide6.QtWidgets import QApplication, QProgressDialog

                prog = QProgressDialog(f"Downloading v{latest_ver} installer...", "Cancel", 0, 100, self)
                prog.setWindowTitle("Updating Essentials Photo Gallery")
                prog.setWindowModality(Qt.WindowModal)
                prog.setMinimumWidth(380)
                prog.setStyleSheet(
                    f"QProgressDialog {{ background-color: {tok.bg_surface}; color: {tok.text_primary}; }} "
                    f"QProgressDialog QLabel {{ color: {tok.text_primary}; background: transparent; }}"
                )
                prog.show()
                cancelled = {"v": False}
                prog.canceled.connect(lambda: cancelled.__setitem__("v", True))

                class _InstallerDlWorker(QThread):
                    progress = QtSignal(int, int)
                    done = QtSignal(object)

                    def run(self) -> None:
                        p = download_installer_exe(
                            installer_url,
                            str(info.get("installer_name") or "EssentialsPhotoGallery_Setup_x64.exe"),
                            progress_cb=lambda d, t: self.progress.emit(int(d), int(t)),
                            cancel_cb=lambda: cancelled["v"],
                        )
                        self.done.emit(p)

                dl_w = _InstallerDlWorker(self)

                def _on_dl_progress(done_b: int, total_b: int) -> None:
                    if total_b > 0:
                        pct = int((done_b * 100) // total_b)
                        prog.setValue(pct)
                        prog.setLabelText(
                            f"Downloading v{latest_ver} installer... "
                            f"{done_b / (1024 * 1024):.1f} / {total_b / (1024 * 1024):.1f} MB ({pct}%)"
                        )
                    else:
                        prog.setLabelText(
                            f"Downloading v{latest_ver} installer... {done_b / (1024 * 1024):.1f} MB"
                        )

                dl_w.progress.connect(_on_dl_progress)

                def _on_dl_done(path_obj: object) -> None:
                    self._update_thread = None
                    prog.close()
                    if path_obj:
                        self.lbl_update_status.setText("Launching installer and closing app...")
                        launched = False
                        try:
                            if hasattr(os, "startfile"):
                                os.startfile(str(path_obj))
                                launched = True
                        except Exception:
                            launched = False
                        if not launched:
                            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path_obj)))
                        # Cleanly close AboutDialog & MainWindow so Inno Setup never hits file/mutex locks
                        parent_win = self.parentWidget()
                        self.accept()
                        if parent_win is not None:
                            QTimer.singleShot(120, parent_win.close)
                        app = QApplication.instance()
                        if app is not None:
                            QTimer.singleShot(350, app.quit)
                    elif not cancelled["v"]:
                        warn = QMessageBox(self)
                        warn.setWindowTitle("Download Failed")
                        warn.setIcon(QMessageBox.Warning)
                        warn.setText("Could not download the installer.")
                        warn.setInformativeText("You can download the update manually from the GitHub Releases page.")
                        warn.setStyleSheet(msg.styleSheet())
                        warn.exec()

                dl_w.done.connect(_on_dl_done)
                self._update_thread = dl_w
                dl_w.start()
            elif clicked == btn_web and html_url:
                QDesktopServices.openUrl(QUrl(html_url))

        worker.result_ready.connect(_on_result)
        self._update_thread = worker
        worker.start()


def prompt_and_download_lama_addon(
    parent: QWidget | None = None,
    feature_name: str = "AI Neural Inpainting",
) -> bool:
    """
    Contextually prompt the user once to download the optional ~208 MB LaMa model
    when using Object Eraser / AI Expand / Panorama Fill for the first time.
    Returns True if the caller should proceed (downloaded or user chose Basic Inpainting),
    or False if the user cancelled.
    """
    from PySide6.QtCore import QSettings, QThread, Signal as QtSignal
    from PySide6.QtWidgets import QProgressDialog
    from src.imaging.creative_effects import (
        LAMA_MODEL_SIZE_MB,
        download_lama_model,
        get_lama_model_status,
    )

    if get_lama_model_status().get("installed"):
        return True

    settings = QSettings("PhotoGallery64", "PhotoGallery64")
    if bool(settings.value("ai/lama_prompted", False, type=bool)):
        return True

    box = QMessageBox(parent)
    box.setWindowTitle(f"Install AI Neural Inpainting (~{LAMA_MODEL_SIZE_MB} MB)?")
    box.setIcon(QMessageBox.Question)
    box.setText(f"Download the optional AI Neural Inpainting add-on (~{LAMA_MODEL_SIZE_MB} MB)?")
    box.setInformativeText(
        f"You're using {feature_name}. Installing the LaMa neural model greatly improves quality "
        "when removing objects or expanding borders.\n\n"
        "You can manage optional add-ons anytime in Options → Add-ons."
    )
    btn_dl = box.addButton(f"Download (~{LAMA_MODEL_SIZE_MB} MB)", QMessageBox.AcceptRole)
    btn_basic = box.addButton("Use Basic Inpainting", QMessageBox.NoRole)
    box.addButton(QMessageBox.Cancel)
    box.exec()
    clicked = box.clickedButton()

    if clicked == btn_basic:
        settings.setValue("ai/lama_prompted", True)
        return True
    if clicked != btn_dl:
        return False

    settings.setValue("ai/lama_prompted", True)
    prog = QProgressDialog(f"Downloading AI Neural Inpainting (~{LAMA_MODEL_SIZE_MB} MB)...", "Cancel", 0, 100, parent)
    prog.setWindowTitle("Downloading Add-on")
    prog.setWindowModality(Qt.WindowModal)
    prog.setAutoClose(False)
    prog.setAutoReset(False)
    prog.setMinimumDuration(0)
    prog.show()

    cancelled = {"v": False}
    prog.canceled.connect(lambda: cancelled.__setitem__("v", True))
    result_holder = {"path": None}

    class _DlThread(QThread):
        progress = QtSignal(int, int)

        def run(self) -> None:
            result_holder["path"] = download_lama_model(
                progress_cb=lambda d, t: self.progress.emit(int(d), int(t)),
                cancel_cb=lambda: cancelled["v"],
            )

    t = _DlThread(parent)
    t.progress.connect(lambda d, tot: prog.setValue(int(min(100, (d * 100) // tot))) if tot > 0 else None)
    t.start()
    while t.isRunning():
        QApplication.processEvents()
        t.wait(30)
    if result_holder["path"] is not None and not cancelled["v"]:
        prog.setValue(100)
    prog.close()
    prog.deleteLater()

    if result_holder["path"] is None and not cancelled["v"]:
        QMessageBox.warning(
            parent,
            "Download Failed",
            "Could not download the AI Neural Inpainting model. Falling back to Basic Inpainting.",
        )
    return True


def prompt_and_download_rapidocr_addon(
    parent: QWidget | None = None,
    on_more_options: Any = None,
) -> bool:
    """
    When OCR is enabled by the user and no dedicated OCR engine (RapidOCR / Tesseract / winocr)
    is installed, prompt to 1-click download RapidOCR (~15 MB), open More Options, or Cancel.
    Returns True if an OCR engine is ready (or was downloaded), False if cancelled.
    """
    from PySide6.QtCore import QThread, Signal as QtSignal
    from PySide6.QtWidgets import QProgressDialog
    from src.core.ocr_engine import (
        RAPIDOCR_SIZE_MB,
        download_rapidocr_models,
        has_configured_ocr_engine,
    )

    if has_configured_ocr_engine():
        return True

    box = QMessageBox(parent)
    box.setWindowTitle("Enable Text Recognition (OCR)")
    box.setIcon(QMessageBox.Question)
    box.setText(f"Download RapidOCR (~{RAPIDOCR_SIZE_MB} MB) for offline text recognition?")
    box.setInformativeText(
        "No offline OCR engine (RapidOCR or Tesseract) is currently installed.\n\n"
        "You can also configure Tesseract or custom model paths in Options → Add-ons."
    )
    btn_dl = box.addButton(f"Download RapidOCR (~{RAPIDOCR_SIZE_MB} MB)", QMessageBox.AcceptRole)
    btn_more = box.addButton("More Options...", QMessageBox.ActionRole)
    box.addButton(QMessageBox.Cancel)
    box.exec()
    clicked = box.clickedButton()

    if clicked == btn_more:
        if callable(on_more_options):
            on_more_options()
        else:
            dlg = AppSettingsDialog(parent=parent)
            if hasattr(dlg, "ai_downloads_tab"):
                dlg.tab_widget.setCurrentWidget(dlg.ai_downloads_tab)
            dlg.exec()
        return has_configured_ocr_engine()

    if clicked != btn_dl:
        return False

    prog = QProgressDialog(f"Downloading RapidOCR (~{RAPIDOCR_SIZE_MB} MB)...", "Cancel", 0, 100, parent)
    prog.setWindowTitle("Downloading OCR Add-on")
    prog.setWindowModality(Qt.WindowModal)
    prog.setMinimumDuration(0)
    prog.show()

    cancelled = {"v": False}
    prog.canceled.connect(lambda: cancelled.__setitem__("v", True))
    result_holder = {"dir": None}

    class _DlThread(QThread):
        progress = QtSignal(int, int)

        def run(self) -> None:
            result_holder["dir"] = download_rapidocr_models(
                progress_cb=lambda d, t: self.progress.emit(int(d), int(t)),
                cancel_cb=lambda: cancelled["v"],
            )

    t = _DlThread(parent)
    t.progress.connect(lambda d, tot: prog.setValue(int(min(99, (d * 100) // tot))) if tot > 0 else None)
    t.start()
    while t.isRunning():
        QApplication.processEvents()
        t.wait(30)
    prog.close()

    if result_holder["dir"] is None and not cancelled["v"]:
        QMessageBox.warning(
            parent,
            "Download Failed",
            "Could not download RapidOCR models. Check your internet connection and try again.",
        )
        return False
    return result_holder["dir"] is not None


