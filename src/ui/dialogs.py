"""
Modal Dialogs for PhotoGallery64:
1. ChangeTimeTakenDialog — Exact visual and mathematical clone of WLPG's 'Change Time Taken'
   dialog (docs/reference_ui/adjust_time.png) calculating relative Δt = T_lead_new - T_lead_original.
2. AppSettingsDialog — WLPG Options & Database Maintenance dialog (docs/reference_ui/database.png)
   with 30-day '.OriginalFiles' backup retention policy, theme selector, and SQLite compaction.
"""

from __future__ import annotations

import datetime
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
    QLabel,
    QLineEdit,
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

    def __init__(
        self,
        retention_days: int = 180,
        current_theme: Literal["wlpg_aero", "modern", "dark"] = "wlpg_aero",
        show_dates: bool = True,
        show_tags: bool = True,
        show_people: bool = True,
        show_ratings: bool = True,
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
        self.resize(580, 540)
        self.setMinimumSize(540, 500)

        self.compact_requested: bool = False
        self.prune_backups_requested: bool = False
        self.backup_requested: bool = False
        self.heal_faces_requested: bool = False
        self.purge_rescan_requested: bool = False
        self.fast_rematch_requested: bool = False
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
        self.tab_widget.setStyleSheet(
            """
            QTabBar::tab {
                padding: 4px 9px;
            }
            """
        )

        # -------------------------------------------------------------
        # Tab 1: General (docs: screenshot 2)
        # -------------------------------------------------------------
        tab_general = QWidget()
        v_gen = QVBoxLayout(tab_general)
        v_gen.setContentsMargins(14, 14, 14, 14)
        v_gen.setSpacing(12)

        # Tooltips group
        lbl_tooltips = QLabel("Tooltips", tab_general)
        lbl_tooltips.setFont(QFont("Segoe UI", 9, QFont.Bold))
        v_gen.addWidget(lbl_tooltips)

        self.chk_tooltip_previews = QCheckBox("Show photo and video previews in tooltips", tab_general)
        self.chk_tooltip_previews.setChecked(tooltip_previews)
        v_gen.addWidget(self.chk_tooltip_previews)

        sep1 = QFrame(tab_general)
        sep1.setFrameShape(QFrame.HLine)
        sep1.setFrameShadow(QFrame.Sunken)
        v_gen.addWidget(sep1)

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

        v_gen.addLayout(nav_grid)

        sep3 = QFrame(tab_general)
        sep3.setFrameShape(QFrame.HLine)
        sep3.setFrameShadow(QFrame.Sunken)
        v_gen.addWidget(sep3)

        # Visual Theme selector
        h_theme = QHBoxLayout()
        lbl_theme = QLabel("Visual theme:", tab_general)
        self.cmb_theme = QComboBox(tab_general)
        self.cmb_theme.addItem("WLPG Classic Aero (Light)", "wlpg_aero")
        self.cmb_theme.addItem("Modern Neutral", "modern")
        self.cmb_theme.addItem("Dark Studio", "dark")
        t_idx = self.cmb_theme.findData(current_theme)
        if t_idx >= 0:
            self.cmb_theme.setCurrentIndex(t_idx)
        h_theme.addWidget(lbl_theme)
        h_theme.addWidget(self.cmb_theme, 1)
        v_gen.addLayout(h_theme)

        # Icon Theme selector
        h_icon_theme = QHBoxLayout()
        lbl_icon_theme = QLabel("Icon style:   ", tab_general)
        self.cmb_icon_theme = QComboBox(tab_general)
        self.cmb_icon_theme.addItem("WLPG Classic Aero (Colourful)", "wlpg_aero")
        self.cmb_icon_theme.addItem("Microsoft Fluent UI (Windows 11)", "fluent")
        self.cmb_icon_theme.addItem("Modern Minimalist", "minimal")
        i_idx = self.cmb_icon_theme.findData(icon_theme)
        if i_idx >= 0:
            self.cmb_icon_theme.setCurrentIndex(i_idx)
        h_icon_theme.addWidget(lbl_icon_theme)
        h_icon_theme.addWidget(self.cmb_icon_theme, 1)
        v_gen.addLayout(h_icon_theme)

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
        lbl_unsaved_desc.setStyleSheet("color: #1E395B; font-size: 9pt;")
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
        # Tab 5: Database
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

        self.btn_heal_faces = QPushButton(
            "Re-parse & Heal Face Regions from Files", maint_frame
        )
        self.btn_heal_faces.clicked.connect(self._on_heal_faces_clicked)
        maint_layout.addWidget(self.btn_heal_faces)

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
            elif itab in ("database", "db"):
                self.tab_widget.setCurrentWidget(self.database_tab)

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

    def _on_apply(self) -> None:
        if hasattr(self, "face_tab") and self.face_tab:
            self.face_tab.save_settings()
        self._save_copy_preferences()
        self.settings_applied.emit()

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

    def _on_prune_clicked(self) -> None:
        self.prune_backups_requested = True
        self.lbl_maint_status.setText("✓ Expired backup cleanup queued on close.")

    def _on_backup_clicked(self) -> None:
        self.backup_requested = True
        self.lbl_maint_status.setText("✓ Catalog backup queued on close.")

    def _on_heal_faces_clicked(self) -> None:
        self.heal_faces_requested = True
        self.lbl_maint_status.setText("✓ Re-parse & heal face regions queued on close.")

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

    def tooltip_previews_enabled(self) -> bool:
        return self.chk_tooltip_previews.isChecked()

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


class _ResizeWorker(QThread):
    progress = Signal(int, int)  # current, total
    finished_batch = Signal(list)  # list[Path]

    def __init__(
        self,
        file_paths: list[Path],
        max_dim: int,
        out_dir: Path,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.file_paths = file_paths
        self.max_dim = max_dim
        self.out_dir = out_dir

    def run(self) -> None:
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        generated: list[Path] = []
        total = len(self.file_paths)
        for i, src in enumerate(self.file_paths):
            if not src.exists():
                self.progress.emit(i + 1, total)
                continue
            try:
                with Image.open(src) as img:
                    img = ImageOps.exif_transpose(img)
                    w, h = img.size
                    if max(w, h) > self.max_dim:
                        scale = self.max_dim / float(max(w, h))
                        nw = max(1, int(round(w * scale)))
                        nh = max(1, int(round(h * scale)))
                        res_img = img.resize((nw, nh), Image.Resampling.LANCZOS)
                    else:
                        res_img = img.copy()

                    stem = src.stem
                    ext = src.suffix or ".jpg"
                    target = self.out_dir / f"{stem}{ext}"
                    if target.exists():
                        c = 2
                        while (self.out_dir / f"{stem} ({c}){ext}").exists():
                            c += 1
                        target = self.out_dir / f"{stem} ({c}){ext}"

                    if ext.lower() in (".jpg", ".jpeg"):
                        res_img.save(target, format="JPEG", quality=92)
                    elif ext.lower() == ".png":
                        res_img.save(target, format="PNG")
                    else:
                        res_img.save(target)
                    generated.append(target)
            except Exception:
                pass
            self.progress.emit(i + 1, total)

        self.finished_batch.emit(generated)


class ResizePhotoDialog(QDialog):
    """
    Authentic Windows Live Photo Gallery 'Resize' dialog (`media_1792184754917_353219bb.png`).
    Allows selecting standard presets (Smaller: 640, Medium: 1024, Large: 1280, Custom),
    specifying maximum dimensions, picking target destination directory, and resizing images
    using high-quality Lanczos resampling with collision avoidance (e.g. 'filename (2).jpg').
    """

    def __init__(
        self,
        file_paths: Sequence[Path | str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Photo Gallery")
        self.setWindowIcon(make_wlpg_icon("gallery", 16))
        self.setFixedWidth(440)

        self.file_paths = [Path(p) for p in file_paths]
        self.generated_files: list[Path] = []

        # Determine initial save folder
        initial_dir = Path.home()
        for p in self.file_paths:
            if p.parent.exists():
                initial_dir = p.parent
                break

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(18, 16, 18, 16)
        vbox.setSpacing(12)

        # 1. Blue WLPG Header ("Resize")
        lbl_title = QLabel("Resize", self)
        lbl_title.setFont(QFont("Segoe UI", 14))
        lbl_title.setStyleSheet("color: #0066CC; font-weight: normal;")
        vbox.addWidget(lbl_title)

        vbox.addSpacing(2)

        # 2. Form Grid (Select a size | Maximum dimensions)
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)

        lbl_size = QLabel("Select a size:", self)
        lbl_size.setFont(QFont("Segoe UI", 9))
        grid.addWidget(lbl_size, 0, 0)

        lbl_dim = QLabel("Maximum dimensions:", self)
        lbl_dim.setFont(QFont("Segoe UI", 9))
        grid.addWidget(lbl_dim, 0, 1)

        self.cmb_size = QComboBox(self)
        self.cmb_size.setFont(QFont("Segoe UI", 9))
        self.cmb_size.addItem("Smaller: 640", 640)
        self.cmb_size.addItem("Medium: 1024", 1024)
        self.cmb_size.addItem("Large: 1280", 1280)
        self.cmb_size.addItem("Custom", None)
        self.cmb_size.setCurrentIndex(1)  # Default: Medium: 1024
        grid.addWidget(self.cmb_size, 1, 0)

        self.txt_dim = QLineEdit(self)
        self.txt_dim.setFont(QFont("Segoe UI", 9))
        self.txt_dim.setValidator(QIntValidator(1, 99999, self))
        self.txt_dim.setText("1024")
        grid.addWidget(self.txt_dim, 1, 1)

        vbox.addLayout(grid)

        vbox.addSpacing(6)

        # 3. Save to: Path + Browse...
        lbl_save = QLabel("Save to:", self)
        lbl_save.setFont(QFont("Segoe UI", 9))
        vbox.addWidget(lbl_save)

        save_row = QHBoxLayout()
        save_row.setSpacing(8)

        self.txt_save_path = QLineEdit(str(initial_dir), self)
        self.txt_save_path.setFont(QFont("Segoe UI", 9))
        self.txt_save_path.setReadOnly(False)
        save_row.addWidget(self.txt_save_path, 1)

        self.btn_browse = QPushButton("Browse...", self)
        self.btn_browse.setFont(QFont("Segoe UI", 9))
        self.btn_browse.clicked.connect(self._on_browse_clicked)
        save_row.addWidget(self.btn_browse)

        vbox.addLayout(save_row)

        vbox.addSpacing(12)

        # Progress bar (visible during batch resizing)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setVisible(False)
        vbox.addWidget(self.progress_bar)

        # 4. Action Buttons (Resize and Save | Cancel)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch(1)

        self.btn_resize = QPushButton("Resize and Save", self)
        self.btn_resize.setFont(QFont("Segoe UI", 9))
        self.btn_resize.setDefault(True)
        self.btn_resize.setMinimumWidth(110)
        self.btn_resize.clicked.connect(self._on_resize_clicked)
        btn_row.addWidget(self.btn_resize)

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setFont(QFont("Segoe UI", 9))
        self.btn_cancel.setMinimumWidth(80)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        btn_row.addStretch(1)
        vbox.addLayout(btn_row)

        self._worker: _ResizeWorker | None = None

        # Connect synchronization between preset combo and dimension text box
        self.cmb_size.currentIndexChanged.connect(self._on_size_combo_changed)
        self.txt_dim.textChanged.connect(self._on_dim_text_changed)

    def _on_size_combo_changed(self, index: int) -> None:
        dim = self.cmb_size.currentData()
        if dim is not None:
            self.txt_dim.blockSignals(True)
            self.txt_dim.setText(str(dim))
            self.txt_dim.blockSignals(False)

    def _on_dim_text_changed(self, text: str) -> None:
        val = int(text) if text.strip().isdigit() else None
        self.cmb_size.blockSignals(True)
        if val == 640:
            self.cmb_size.setCurrentIndex(0)
        elif val == 1024:
            self.cmb_size.setCurrentIndex(1)
        elif val == 1280:
            self.cmb_size.setCurrentIndex(2)
        else:
            self.cmb_size.setCurrentIndex(3)  # Custom
        self.cmb_size.blockSignals(False)

    def _on_browse_clicked(self) -> None:
        curr = self.txt_save_path.text()
        chosen = QFileDialog.getExistingDirectory(self, "Select Folder to Save Resized Photos", curr)
        if chosen:
            self.txt_save_path.setText(chosen)

    def _on_resize_clicked(self) -> None:
        dim_str = self.txt_dim.text().strip()
        max_dim = int(dim_str) if dim_str.isdigit() and int(dim_str) > 0 else 1024
        out_dir = Path(self.txt_save_path.text().strip() or ".")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        self.btn_resize.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self._worker = _ResizeWorker(self.file_paths, max_dim, out_dir, self)
        self._worker.progress.connect(self._on_resize_progress)
        self._worker.finished_batch.connect(self._on_resize_finished)
        self._worker.start()

    def _on_resize_progress(self, current: int, total: int) -> None:
        if total > 0:
            pct = int(100 * current / total)
            self.progress_bar.setValue(pct)

    def _on_resize_finished(self, generated: list[Path]) -> None:
        self.generated_files = generated
        self.accept()

    def execute_resize(self) -> list[Path]:
        """Resizes all input photos to the requested max dimension using Lanczos interpolation."""
        dim_str = self.txt_dim.text().strip()
        max_dim = int(dim_str) if dim_str.isdigit() and int(dim_str) > 0 else 1024
        out_dir = Path(self.txt_save_path.text().strip() or ".")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        generated: list[Path] = []
        for src in self.file_paths:
            if not src.exists():
                continue
            try:
                with Image.open(src) as img:
                    img = ImageOps.exif_transpose(img)
                    w, h = img.size
                    if max(w, h) > max_dim:
                        scale = max_dim / float(max(w, h))
                        nw = max(1, int(round(w * scale)))
                        nh = max(1, int(round(h * scale)))
                        res_img = img.resize((nw, nh), Image.Resampling.LANCZOS)
                    else:
                        res_img = img.copy()

                    stem = src.stem
                    ext = src.suffix or ".jpg"
                    target = out_dir / f"{stem}{ext}"
                    # Collision avoidance: filename (2).ext, filename (3).ext...
                    if target.exists():
                        c = 2
                        while (out_dir / f"{stem} ({c}){ext}").exists():
                            c += 1
                        target = out_dir / f"{stem} ({c}){ext}"

                    # Save preserving format / quality
                    if ext.lower() in (".jpg", ".jpeg"):
                        res_img.save(target, format="JPEG", quality=92)
                    elif ext.lower() == ".png":
                        res_img.save(target, format="PNG")
                    else:
                        res_img.save(target)
                    generated.append(target)
            except Exception:
                pass

        self.generated_files = generated
        return generated


class AboutDialog(QDialog):
    """About dialog for Essentials Photo Gallery."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About Essentials Photo Gallery")
        self.setFixedSize(480, 310)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

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
        lbl_title.setStyleSheet("font-family: 'Segoe UI', sans-serif; font-size: 13pt; font-weight: bold; color: #1E395B;")
        info_vbox.addWidget(lbl_title)

        lbl_ver = QLabel("Build 2026 (v16.4.3528.0331)", self)
        lbl_ver.setStyleSheet("font-family: 'Segoe UI', sans-serif; font-size: 8.5pt; color: #55687D;")
        info_vbox.addWidget(lbl_ver)

        lbl_desc = QLabel(
            "Essentials Photo Gallery recreates the beloved fluid ribbon interface and Aero experience "
            "with modern AI intelligence, hardware-accelerated monocular depth & portrait lighting, "
            "and lossless non-destructive photo management.",
            self,
        )
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet("font-family: 'Segoe UI', sans-serif; font-size: 9pt; color: #333333; margin-top: 4px;")
        info_vbox.addWidget(lbl_desc)

        header_hbox.addLayout(info_vbox, 1)
        vbox.addLayout(header_hbox)

        # Separator line
        sep = QFrame(self)
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet("color: #DDE5EE;")
        vbox.addWidget(sep)

        # Technical details
        lbl_tech = QLabel(
            "Powered by PySide6 (Qt 6.8), OpenCV, ONNX Runtime (DirectML / CPU), "
            "and SQLite WAL.\n© 2026 Essentials Photo Gallery Contributors.",
            self,
        )
        lbl_tech.setStyleSheet("font-family: 'Segoe UI', sans-serif; font-size: 8pt; color: #667788;")
        vbox.addWidget(lbl_tech)

        vbox.addStretch(1)

        # Bottom OK button
        btn_box = QHBoxLayout()
        btn_box.addStretch(1)
        btn_ok = QPushButton("OK", self)
        btn_ok.setFixedSize(85, 26)
        btn_ok.setDefault(True)
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

        self.setStyleSheet(
            """
            QDialog {
                background-color: #FFFFFF;
            }
            """
        )


