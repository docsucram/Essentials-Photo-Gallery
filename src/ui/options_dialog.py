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

    def __init__(
        self,
        engine: DualDatabaseEngine | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.repo = CatalogRepository(engine) if engine is not None else None
        self._settings = QSettings("PhotoGallery64", "PhotoGallery64")

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        # 1. Engine Status Group
        status_group = QGroupBox("AI Recognition Engine Status", self)
        status_layout = QVBoxLayout(status_group)

        self.lbl_engine_status = QLabel(self)
        self.lbl_engine_status.setWordWrap(True)
        # Query active engine status
        try:
            fe = FaceEngine()
            if fe.is_available:
                m_name = getattr(fe, "model_name", "buffalo_l")
                p_name = getattr(fe, "active_provider", "CPUExecutionProvider")
                self.lbl_engine_status.setText(
                    f"✔ Active Engine: InsightFace {m_name} ONNX\n"
                    f"   Execution Provider: {p_name}\n"
                    f"   Status: Operational (ArcFace 512-D + SCRFD Landmark Geometry)"
                )
                self.lbl_engine_status.setStyleSheet("color: #22C55E; font-weight: bold;")
            else:
                self.lbl_engine_status.setText(
                    "⚠ Face Engine: Unavailable (InsightFace models missing). Fallback disabled."
                )
                self.lbl_engine_status.setStyleSheet("color: #EF4444; font-weight: bold;")
        except Exception as ex:
            self.lbl_engine_status.setText(f"Status check error: {ex}")
            self.lbl_engine_status.setStyleSheet("color: #EF4444;")

        status_layout.addWidget(self.lbl_engine_status)
        main_layout.addWidget(status_group)

        # 2. Tuning Sliders Group
        tuning_group = QGroupBox("Face Detection & Matching Sensitivity", self)
        tuning_layout = QFormLayout(tuning_group)
        tuning_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Detection Confidence Slider (50% - 90%, default 65%)
        h_box_det = QHBoxLayout()
        self.slider_confidence = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_confidence.setRange(50, 90)
        self.slider_confidence.setValue(65)
        self.slider_confidence.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_confidence.setTickInterval(5)
        self.lbl_conf_val = QLabel("65%", self)
        self.lbl_conf_val.setFixedWidth(40)
        self.slider_confidence.valueChanged.connect(
            lambda v: self.lbl_conf_val.setText(f"{v}%")
        )
        h_box_det.addWidget(self.slider_confidence, 1)
        h_box_det.addWidget(self.lbl_conf_val)
        tuning_layout.addRow("Detection Confidence:", h_box_det)

        # Suggestion Match Threshold Slider (40% - 80%, default 52%)
        h_box_match = QHBoxLayout()
        self.slider_match = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_match.setRange(40, 80)
        self.slider_match.setValue(52)
        self.slider_match.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_match.setTickInterval(5)
        self.lbl_match_val = QLabel("52%", self)
        self.lbl_match_val.setFixedWidth(40)
        self.slider_match.valueChanged.connect(self._on_match_slider_changed)
        h_box_match.addWidget(self.slider_match, 1)
        h_box_match.addWidget(self.lbl_match_val)
        tuning_layout.addRow("Match Threshold:", h_box_match)

        # Minimum Face Size Slider (24px - 120px, default 40px)
        h_box_size = QHBoxLayout()
        self.slider_size = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_size.setRange(24, 120)
        self.slider_size.setValue(40)
        self.slider_size.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_size.setTickInterval(10)
        self.lbl_size_val = QLabel("40 px", self)
        self.lbl_size_val.setFixedWidth(45)
        self.slider_size.valueChanged.connect(
            lambda v: self.lbl_size_val.setText(f"{v} px")
        )
        h_box_size.addWidget(self.slider_size, 1)
        h_box_size.addWidget(self.lbl_size_val)
        tuning_layout.addRow("Minimum Face Size:", h_box_size)

        h_reset_tuning = QHBoxLayout()
        h_reset_tuning.addStretch(1)
        self.btn_restore_face_defaults = QPushButton("Restore Default Sensitivity", self)
        self.btn_restore_face_defaults.setToolTip("Reset Detection Confidence (65%), Match Threshold (52%), and Minimum Face Size (40 px) to defaults.")
        self.btn_restore_face_defaults.clicked.connect(self._restore_default_sensitivity)
        h_reset_tuning.addWidget(self.btn_restore_face_defaults)
        tuning_layout.addRow("", h_reset_tuning)

        main_layout.addWidget(tuning_group)

        # 3. Maintenance Group
        maint_group = QGroupBox("Database Maintenance & Re-Learning", self)
        maint_layout = QVBoxLayout(maint_group)

        lbl_maint_info = QLabel(
            "If false detections (e.g. logos, architecture) exist from earlier scans, "
            "purge unconfirmed suggestions to re-scan with the strict anatomical filter.",
            self,
        )
        lbl_maint_info.setWordWrap(True)
        lbl_maint_info.setStyleSheet("font-size: 11px;")
        maint_layout.addWidget(lbl_maint_info)

        v_btn_layout = QVBoxLayout()
        v_btn_layout.setSpacing(8)

        self.btn_fast_rematch = QPushButton("⚡ Recalculate Signatures & Re-Match (Fast)", self)
        self.btn_fast_rematch.setMinimumHeight(32)
        self.btn_fast_rematch.setToolTip("Rebuilds person centroids from confirmed faces and re-scores unassigned faces (<1s).")
        self.btn_fast_rematch.clicked.connect(self._on_fast_rematch_clicked)
        v_btn_layout.addWidget(self.btn_fast_rematch)

        self.btn_purge_rescan = QPushButton("🗑 Purge Unconfirmed Suggestions & Re-Scan", self)
        self.btn_purge_rescan.setMinimumHeight(32)
        self.btn_purge_rescan.setToolTip("Deletes all unconfirmed suggestions (retaining all confirmed people) and re-scans photos.")
        self.btn_purge_rescan.setStyleSheet("QPushButton { color: #C62828; font-weight: bold; }")
        self.btn_purge_rescan.clicked.connect(self._on_purge_rescan_clicked)
        v_btn_layout.addWidget(self.btn_purge_rescan)

        maint_layout.addLayout(v_btn_layout)
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
