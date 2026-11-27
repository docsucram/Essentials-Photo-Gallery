"""
Options and Settings Dialog for PhotoGallery64.
Provides configuration tabs for General, Face Recognition, Import, Originals, Edit, and Database.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.core.face_engine import FaceEngine
from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository

logger = logging.getLogger(__name__)


class FaceRecognitionOptionsTab(QWidget):
    """
    Dedicated Face Recognition settings tab providing:
    - AI Engine status (InsightFace buffalo_l / buffalo_s ONNX, Execution Provider)
    - Detection confidence slider (50% - 90%, default 65%)
    - Suggestion match threshold slider (40% - 80%, default 52%)
    - Minimum face size slider (24px - 120px, default 40px)
    - Maintenance: Purge unconfirmed suggestions & re-scan, Fast recalculate signatures
    """

    match_threshold_changed = Signal(float)
    settings_changed = Signal()
    purge_rescan_requested = Signal()
    fast_rematch_requested = Signal()
    heal_faces_requested = Signal()
    rescan_unscanned_requested = Signal()

    def __init__(
        self,
        engine: DualDatabaseEngine | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.repo = CatalogRepository(engine) if engine is not None else None
        self._settings = QSettings("PhotoGallery64", "PhotoGallery64")
        self._face_dl_thread: Any = None
        self._face_models_installed: bool = False

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(10)

        # 1. Engine Status Group
        status_group = QGroupBox("AI Recognition Engine Status", self)
        status_layout = QVBoxLayout(status_group)
        status_layout.setContentsMargins(12, 8, 12, 8)
        status_layout.setSpacing(6)

        self.lbl_status_badge = QLabel(self)
        self.lbl_status_badge.setWordWrap(True)
        self.lbl_hw_provider = QLabel(self)
        self.lbl_hw_provider.setWordWrap(True)

        status_layout.addWidget(self.lbl_status_badge)
        status_layout.addWidget(self.lbl_hw_provider)

        # Download / Reinstall / Uninstall action row
        face_btn_row = QHBoxLayout()
        self.btn_face_download = QPushButton("Download Face Models (~281 MB)", self)
        self.btn_face_download.setToolTip(
            "Downloads or repairs the offline AI models used to detect and recognize faces (InsightFace buffalo_l ONNX, ~281 MB)."
        )
        self.btn_face_uninstall = QPushButton("Uninstall Models", self)
        self.btn_face_uninstall.setToolTip(
            "Removes the cached face recognition models to free ~281 MB; your existing tagged people are kept."
        )
        self.btn_face_download.clicked.connect(self._on_face_download_clicked)
        self.btn_face_uninstall.clicked.connect(self._on_face_uninstall_clicked)
        face_btn_row.addWidget(self.btn_face_download)
        face_btn_row.addWidget(self.btn_face_uninstall)
        face_btn_row.addStretch(1)
        status_layout.addLayout(face_btn_row)

        # Technical details toggleable view
        self.btn_toggle_tech = QPushButton("▸ Show Technical Details", self)
        self.btn_toggle_tech.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_tech.setStyleSheet(
            "QPushButton { border: none; background: transparent; color: #2563EB; text-align: left; padding: 2px 0px; font-size: 8pt; }"
            "QPushButton:hover { text-decoration: underline; color: #1D4ED8; }"
        )
        self.lbl_tech_details = QLabel("", self)
        self.lbl_tech_details.setStyleSheet(
            "background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 3px; padding: 6px 8px; color: #475569; font-family: monospace; font-size: 8pt;"
        )
        self.lbl_tech_details.setVisible(False)

        def _toggle_tech():
            should_show = self.lbl_tech_details.isHidden()
            self.lbl_tech_details.setVisible(should_show)
            self.btn_toggle_tech.setText("▾ Hide Technical Details" if should_show else "▸ Show Technical Details")

        self.btn_toggle_tech.clicked.connect(_toggle_tech)
        status_layout.addWidget(self.btn_toggle_tech)
        status_layout.addWidget(self.lbl_tech_details)
        self._refresh_engine_status()

        main_layout.addWidget(status_group)

        # 2. Tuning Sliders Group
        tuning_group = QGroupBox("Face Detection && Matching Sensitivity", self)
        tuning_layout = QFormLayout(tuning_group)
        tuning_layout.setContentsMargins(12, 10, 12, 10)
        tuning_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        tuning_layout.setSpacing(8)

        # 1. Face Finder Sensitivity (Detection Confidence, 50% - 90%, default 65%)
        v_det = QVBoxLayout()
        v_det.setSpacing(2)
        h_box_det = QHBoxLayout()
        self.slider_confidence = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_confidence.setRange(50, 90)
        self.slider_confidence.setValue(65)
        self.slider_confidence.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_confidence.setTickInterval(5)
        tip_det = (
            "Controls how certain the detector must be before marking a region as a face—raise it to reduce false alarms, or lower it to catch distant or side-profile faces (Face Finder Sensitivity / SCRFD threshold, default 65%)."
        )
        self.slider_confidence.setToolTip(tip_det)
        self.lbl_conf_val = QLabel("65%", self)
        self.lbl_conf_val.setFixedWidth(40)
        self.slider_confidence.valueChanged.connect(lambda v: self.lbl_conf_val.setText(f"{v}%"))
        h_box_det.addWidget(self.slider_confidence, 1)
        h_box_det.addWidget(self.lbl_conf_val)
        v_det.addLayout(h_box_det)
        self.lbl_det_hint = QLabel("Higher = fewer false faces; Lower = catches faint or profile faces", self)
        self.lbl_det_hint.setWordWrap(True)
        self.lbl_det_hint.setStyleSheet("color: #64748B; font-size: 8pt;")
        v_det.addWidget(self.lbl_det_hint)
        tuning_layout.addRow("Face Finder Sensitivity:", v_det)

        # 2. Match Strictness (Match Threshold, 40% - 80%, default 52%)
        v_match = QVBoxLayout()
        v_match.setSpacing(2)
        h_box_match = QHBoxLayout()
        self.slider_match = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_match.setRange(40, 80)
        self.slider_match.setValue(52)
        self.slider_match.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_match.setTickInterval(5)
        tip_match = (
            "Controls how closely a face must match someone you've already named before Photo Gallery suggests them—raise it if lookalikes get mixed up (Name Matching Strictness / ArcFace cosine threshold, default 52%)."
        )
        self.slider_match.setToolTip(tip_match)
        self.lbl_match_val = QLabel("52%", self)
        self.lbl_match_val.setFixedWidth(40)
        self.slider_match.valueChanged.connect(self._on_match_slider_changed)
        h_box_match.addWidget(self.slider_match, 1)
        h_box_match.addWidget(self.lbl_match_val)
        v_match.addLayout(h_box_match)
        self.lbl_match_hint = QLabel("Higher = stricter (fewer wrong guesses); Lower = groups more faces automatically", self)
        self.lbl_match_hint.setWordWrap(True)
        self.lbl_match_hint.setStyleSheet("color: #64748B; font-size: 8pt;")
        v_match.addWidget(self.lbl_match_hint)
        tuning_layout.addRow("Match Strictness:", v_match)

        # 3. Minimum Face Size (24px - 120px, default 40px)
        v_size = QVBoxLayout()
        v_size.setSpacing(2)
        h_box_size = QHBoxLayout()
        self.slider_size = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_size.setRange(24, 120)
        self.slider_size.setValue(40)
        self.slider_size.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_size.setTickInterval(10)
        tip_size = (
            "Ignores tiny background faces smaller than this width so crowd shots stay clutter-free, or lower it to catch distant faces (Minimum Face Size bounding box in pixels, default 40 px)."
        )
        self.slider_size.setToolTip(tip_size)
        self.lbl_size_val = QLabel("40 px", self)
        self.lbl_size_val.setFixedWidth(45)
        self.slider_size.valueChanged.connect(lambda v: self.lbl_size_val.setText(f"{v} px"))
        h_box_size.addWidget(self.slider_size, 1)
        h_box_size.addWidget(self.lbl_size_val)
        v_size.addLayout(h_box_size)
        self.lbl_size_hint = QLabel("Faces smaller than this size (in pixels) are ignored during scans", self)
        self.lbl_size_hint.setWordWrap(True)
        self.lbl_size_hint.setStyleSheet("color: #64748B; font-size: 8pt;")
        v_size.addWidget(self.lbl_size_hint)
        tuning_layout.addRow("Minimum Face Size:", v_size)

        h_reset_tuning = QHBoxLayout()
        h_reset_tuning.addStretch(1)
        self.btn_restore_face_defaults = QPushButton("Restore Default Sensitivity", self)
        self.btn_restore_face_defaults.setToolTip(
            "Resets all three face detection and matching sliders back to their balanced defaults (65% sensitivity, 52% match strictness, 40 px minimum size)."
        )
        self.btn_restore_face_defaults.clicked.connect(self._restore_default_sensitivity)
        h_reset_tuning.addWidget(self.btn_restore_face_defaults)
        tuning_layout.addRow("", h_reset_tuning)

        main_layout.addWidget(tuning_group)

        # 3. Maintenance Group
        maint_group = QGroupBox("Face Tagging Maintenance && Recovery", self)
        maint_layout = QVBoxLayout(maint_group)
        maint_layout.setContentsMargins(12, 8, 12, 10)
        maint_layout.setSpacing(8)

        self.lbl_maint_info = QLabel(
            "Manage face recognition data, re-match unnamed faces, or re-sync face tags from photo files:",
            self,
        )
        self.lbl_maint_info.setWordWrap(True)
        self.lbl_maint_info.setStyleSheet("font-size: 8.5pt; color: #55687D;")
        maint_layout.addWidget(self.lbl_maint_info)

        maint_grid = QGridLayout()
        maint_grid.setContentsMargins(0, 0, 0, 0)
        maint_grid.setHorizontalSpacing(8)
        maint_grid.setVerticalSpacing(6)
        maint_grid.setColumnStretch(0, 1)
        maint_grid.setColumnStretch(1, 1)

        self.btn_rescan_unscanned = QPushButton("Scan Unscanned Photos", self)
        self.btn_rescan_unscanned.setToolTip(
            "Queues photos that haven't been scanned yet or failed face embedding so missed people can be found; all confirmed tags are kept (checks missing or zero-face records)."
        )
        self.btn_rescan_unscanned.clicked.connect(self._on_rescan_unscanned_clicked)
        maint_grid.addWidget(self.btn_rescan_unscanned, 0, 0)

        self.btn_fast_rematch = QPushButton("Fast Re-Match People", self)
        self.btn_fast_rematch.setToolTip(
            "Instantly refreshes name suggestions using your latest confirmed faces—useful right after tagging several photos or adjusting Match Strictness (recalculates person centroids in under 1s)."
        )
        self.btn_fast_rematch.clicked.connect(self._on_fast_rematch_clicked)
        maint_grid.addWidget(self.btn_fast_rematch, 0, 1)

        self.btn_heal_faces = QPushButton("Re-import Face Boxes (XMP)", self)
        self.btn_heal_faces.setToolTip(
            "Re-reads face rectangles embedded inside your photo files to fix shifted, rotated, or missing face boxes without changing confirmed names (parses XMP-mwg-rs / Microsoft RegionInfo metadata)."
        )
        self.btn_heal_faces.clicked.connect(self._on_heal_faces_clicked)
        maint_grid.addWidget(self.btn_heal_faces, 1, 0)

        self.btn_purge_rescan = QPushButton("Purge Suggestions && Re-Scan", self)
        self.btn_purge_rescan.setToolTip(
            "Clears unconfirmed face guesses and re-scans photos from scratch to clean up noisy suggestions; confirmed people tags are always preserved (resets face_scan_version for unconfirmed faces)."
        )
        self.btn_purge_rescan.setStyleSheet(
            "QPushButton { color: #B91C1C; border: 1px solid #FCA5A5; background: #FEF2F2; "
            "border-radius: 3px; padding: 4px 10px; font-weight: 600; }"
            "QPushButton:hover { background: #FEE2E2; border-color: #EF4444; }"
        )
        self.btn_purge_rescan.clicked.connect(self._on_purge_rescan_clicked)
        maint_grid.addWidget(self.btn_purge_rescan, 1, 1)

        maint_layout.addLayout(maint_grid)
        main_layout.addWidget(maint_group)

        main_layout.addStretch(1)

    def _load_settings(self) -> None:
        conf = int(round(float(self._settings.value("face_min_score", 0.65)) * 100))
        match = int(round(float(self._settings.value("face_match_threshold", 0.52)) * 100))
        size = int(self._settings.value("face_min_size", 40))

        self.slider_confidence.setValue(max(50, min(90, conf)))
        self.slider_match.setValue(max(40, min(80, match)))
        self.slider_size.setValue(max(24, min(120, size)))

        self.lbl_conf_val.setText(f"{self.slider_confidence.value()}%")
        self.lbl_match_val.setText(f"{self.slider_match.value()}%")
        self.lbl_size_val.setText(f"{self.slider_size.value()} px")

    def save_settings(self) -> None:
        conf_float = self.slider_confidence.value() / 100.0
        match_float = self.slider_match.value() / 100.0
        size_int = self.slider_size.value()

        self._settings.setValue("face_min_score", conf_float)
        self._settings.setValue("face_match_threshold", match_float)
        self._settings.setValue("face_min_size", size_int)
        self._settings.sync()
        self.settings_changed.emit()

    def _restore_default_sensitivity(self) -> None:
        self.slider_confidence.setValue(65)
        self.slider_match.setValue(52)
        self.slider_size.setValue(40)
        self.save_settings()

    def _on_match_slider_changed(self, value: int) -> None:
        self.lbl_match_val.setText(f"{value}%")
        threshold = value / 100.0
        self._settings.setValue("face_match_threshold", threshold)
        self.match_threshold_changed.emit(threshold)

    def _on_fast_rematch_clicked(self) -> None:
        if not self.repo:
            QMessageBox.warning(self, "Fast Re-Match", "Database repository is unavailable.")
            return

        match_score = self.slider_match.value() / 100.0
        try:
            count = self.repo.recalculate_signatures_and_rematch(min_match_score=match_score)
            QMessageBox.information(
                self,
                "Signatures Recalculated",
                f"Successfully recalculated person centroids and updated {count} face suggestion(s).",
            )
            self.fast_rematch_requested.emit()
        except Exception as e:
            logger.exception("Fast rematch failed: %s", e)
            QMessageBox.critical(self, "Error", f"Failed to recalculate signatures: {e}")

    def _on_purge_rescan_clicked(self) -> None:
        if not self.repo:
            QMessageBox.warning(self, "Purge Suggestions", "Database repository is unavailable.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Purge & Re-Scan",
            "This will delete all unconfirmed face suggestions and reset scan progress on photos "
            "to re-scan them cleanly with the InsightFace anatomical filter.\n\n"
            "All confirmed people tags will be safely preserved 100%.\n\n"
            "Do you want to proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                deleted_count = self.repo.purge_unconfirmed_suggestions()
                QMessageBox.information(
                    self,
                    "Purge Complete",
                    f"Purged {deleted_count} unconfirmed suggestions. "
                    f"Background detection will now re-scan photos cleanly.",
                )
                self.purge_rescan_requested.emit()
            except Exception as e:
                logger.exception("Purge suggestions failed: %s", e)
                QMessageBox.critical(self, "Error", f"Failed to purge suggestions: {e}")

    def _on_heal_faces_clicked(self) -> None:
        if not self.repo:
            QMessageBox.warning(self, "Re-import Face Boxes", "Database repository is unavailable.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Re-import Face Boxes",
            "This will scan your original photo files on disk, read their raw XMP face metadata, "
            "and correct any rotated, misplaced, or missing face boxes in the catalog.\n\n"
            "All confirmed people tags are 100% safely preserved.\n\n"
            "Do you want to proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                count = self.repo.reparse_and_heal_face_regions_from_files()
                QMessageBox.information(
                    self,
                    "Face Tags Re-imported",
                    f"Successfully re-read and synced face boxes for {count} photo(s) from disk metadata.",
                )
                self.heal_faces_requested.emit()
            except Exception as e:
                logger.exception("Re-import face boxes failed: %s", e)
                QMessageBox.critical(self, "Error", f"Failed to re-import face boxes: {e}")

    def _on_rescan_unscanned_clicked(self) -> None:
        if not self.repo:
            QMessageBox.warning(self, "Scan Unscanned Photos", "Database repository is unavailable.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Scan Photos",
            "This will find all photos that have no faces or missing embeddings and queue them "
            "for background face detection.\n\n"
            "All existing confirmed people and tags are completely preserved.\n\n"
            "Do you want to proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                count = self.repo.reset_unscanned_and_zero_face_media()
                QMessageBox.information(
                    self,
                    "Scan Queued",
                    f"Successfully queued {count:,} photo(s) for background face scanning.",
                )
                self.rescan_unscanned_requested.emit()
            except Exception as e:
                logger.exception("Scan unscanned photos failed: %s", e)
                QMessageBox.critical(self, "Error", f"Failed to queue photos: {e}")

    def _refresh_engine_status(self) -> None:
        try:
            st = FaceEngine.probe_status()
            installed = bool(st.get("available"))
            self._face_models_installed = installed
            if installed:
                m_name = str(st.get("model_name") or "buffalo_l")
                p_name = str(st.get("active_provider") or "CPUExecutionProvider")
                self.lbl_status_badge.setText("●  Engine Status: Ready (High Accuracy Model)")
                self.lbl_status_badge.setStyleSheet("color: #16A34A; font-weight: bold; font-size: 9.5pt;")
                hw_desc = "GPU Accelerated" if any(k in p_name for k in ("CUDA", "DirectML", "Tensorrt", "Dml")) else f"Standard CPU ({p_name})"
                self.lbl_hw_provider.setText(f"Hardware Processing: {hw_desc}")
                self.lbl_hw_provider.setStyleSheet("color: #475569; font-size: 8.5pt;")
                tech_text = (
                    f"Model: InsightFace {m_name} ONNX\n"
                    f"Architecture: ArcFace 512-D + SCRFD Landmark Geometry\n"
                    f"Active Execution Provider: {p_name}"
                )
                self.btn_face_download.setText("Reinstall / Repair")
                self.btn_face_uninstall.setVisible(True)
                self.btn_face_uninstall.setEnabled(True)
            else:
                self.lbl_status_badge.setText("●  Engine Status: Not Downloaded / Offline")
                self.lbl_status_badge.setStyleSheet("color: #D97706; font-weight: bold; font-size: 9.5pt;")
                self.lbl_hw_provider.setText("Download the face models below, or they will download automatically on first scan.")
                self.lbl_hw_provider.setStyleSheet("color: #475569; font-size: 8.5pt;")
                tech_text = f"InsightFace models not yet cached in {st.get('models_root', '~/.insightface/models')}."
                self.btn_face_download.setText("Download Face Models (~281 MB)")
                self.btn_face_uninstall.setVisible(False)
                self.btn_face_uninstall.setEnabled(False)
            self.btn_face_download.setEnabled(bool(st.get("insightface_installed", True)))
        except Exception as ex:
            self._face_models_installed = False
            self.lbl_status_badge.setText("●  Engine Status: Error Checking Status")
            self.lbl_status_badge.setStyleSheet("color: #DC2626; font-weight: bold; font-size: 9.5pt;")
            self.lbl_hw_provider.setText(str(ex))
            self.lbl_hw_provider.setStyleSheet("color: #EF4444; font-size: 8.5pt;")
            tech_text = str(ex)
            self.btn_face_download.setText("Download Face Models (~281 MB)")
            self.btn_face_uninstall.setVisible(False)
            self.btn_face_uninstall.setEnabled(False)
        self.lbl_tech_details.setText(tech_text)

    def apply_theme(self, is_dark: bool) -> None:
        """Apply light or dark styling to labels, technical details, and danger button."""
        muted = "#94A3B8" if is_dark else "#55687D"
        link_col = "#60A5FA" if is_dark else "#2563EB"
        link_hover = "#93C5FD" if is_dark else "#1D4ED8"
        if hasattr(self, "lbl_hw_provider"):
            self.lbl_hw_provider.setStyleSheet(f"color: {muted}; font-size: 8.5pt;")
        if hasattr(self, "btn_toggle_tech"):
            self.btn_toggle_tech.setStyleSheet(
                f"QPushButton {{ border: none; background: transparent; color: {link_col}; text-align: left; padding: 2px 0px; font-size: 8pt; }}"
                f"QPushButton:hover {{ text-decoration: underline; color: {link_hover}; }}"
            )
        if hasattr(self, "lbl_tech_details"):
            if is_dark:
                self.lbl_tech_details.setStyleSheet(
                    "background: #232832; border: 1px solid #3E4756; border-radius: 3px; padding: 6px 8px; color: #CBD5E1; font-family: monospace; font-size: 8pt;"
                )
            else:
                self.lbl_tech_details.setStyleSheet(
                    "background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 3px; padding: 6px 8px; color: #475569; font-family: monospace; font-size: 8pt;"
                )
        for lbl_attr in ("lbl_det_hint", "lbl_match_hint", "lbl_size_hint"):
            lbl = getattr(self, lbl_attr, None)
            if lbl is not None:
                lbl.setStyleSheet(f"color: {muted}; font-size: 8pt;")
        if hasattr(self, "lbl_maint_info"):
            self.lbl_maint_info.setStyleSheet(f"color: {muted}; font-size: 8.5pt;")
        if hasattr(self, "btn_purge_rescan"):
            self.btn_purge_rescan.setStyleSheet(
                "QPushButton { color: #FCA5A5; border: 1px solid #7F1D1D; background: #2D181B; "
                "border-radius: 3px; padding: 4px 10px; font-weight: 600; }"
                "QPushButton:hover { background: #451A20; border-color: #EF4444; }"
                if is_dark
                else "QPushButton { color: #B91C1C; border: 1px solid #FCA5A5; background: #FEF2F2; "
                "border-radius: 3px; padding: 4px 10px; font-weight: 600; }"
                "QPushButton:hover { background: #FEE2E2; border-color: #EF4444; }"
            )

    def _on_face_download_clicked(self) -> None:
        if self._face_dl_thread is not None and self._face_dl_thread.isRunning():
            return
        from PySide6.QtCore import QThread, Signal as QtSignal

        force_reinstall = bool(self._face_models_installed)

        class _FaceDlWorker(QThread):
            finished_ok = QtSignal(bool)

            def run(self) -> None:
                ok = FaceEngine.download_models("buffalo_l", force_reinstall=force_reinstall)
                self.finished_ok.emit(bool(ok))

        self.btn_face_download.setEnabled(False)
        self.btn_face_download.setText("Reinstalling..." if force_reinstall else "Downloading (~281 MB)...")
        self.btn_face_uninstall.setVisible(False)

        worker = _FaceDlWorker(self)

        def _on_done(ok: bool) -> None:
            self._face_dl_thread = None
            self.btn_face_download.setEnabled(True)
            self._refresh_engine_status()
            if not ok:
                QMessageBox.warning(
                    self,
                    "Download Failed",
                    "Could not download or initialize InsightFace models. Check your internet connection and try again.",
                )

        worker.finished_ok.connect(_on_done)
        self._face_dl_thread = worker
        worker.start()

    def _on_face_uninstall_clicked(self) -> None:
        reply = QMessageBox.question(
            self,
            "Uninstall Face Models",
            "Remove cached InsightFace recognition models from this PC?\n\n"
            "Existing confirmed people and face tags in your catalog are kept, "
            "but new face scanning will pause until models are downloaded again.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            if FaceEngine.uninstall_models():
                QMessageBox.information(self, "Uninstalled", "InsightFace models removed.")
            self._refresh_engine_status()


from src.ui.dialogs import AppSettingsDialog


class OptionsDialog(AppSettingsDialog):
    """
    Main PhotoGallery64 Options / Preferences Dialog.
    Contains General, Face Recognition, Import, Originals, Edit, and Database tabs.
    """

    def __init__(
        self,
        engine: DualDatabaseEngine | None = None,
        initial_tab: str = "general",
        parent: QWidget | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            engine=engine,
            initial_tab=initial_tab,
            parent=parent,
            **kwargs,
        )


FaceOptionsDialog = OptionsDialog
