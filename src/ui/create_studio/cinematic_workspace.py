"""Integrated 3D Cinematic Motion Video Workspace."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QSize, QTimer, Qt
from PySide6.QtGui import QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.imaging.creative_effects import (
    compute_target_dimensions,
    prepare_3d_parallax_layers,
    render_3d_cinematic_motion_frame,
)
from src.ui.create_studio.base import CreateWorkspaceBase, RibbonActionSpec
from src.ui.create_studio.shared_widgets import (
    AeroStudioSlider,
    StudioCardGroup,
    _cv2_to_qpixmap,
    build_studio_panel_qss,
    prompt_save_new_file,
    stop_qthread_safely,
)
from src.ui.create_studio_dialogs import CinematicMotionWorker
from src.ui.ribbon_toolbar import make_wlpg_icon
from src.ui.theme_tokens import get_theme_tokens

_MOTION_STYLES = [
    ("Dolly Zoom (Vertigo)", "dolly_zoom"),
    ("Parallax Drift", "parallax_drift"),
    ("3D Orbit", "3d_orbit"),
    ("Cinematic Pan", "cinematic_pan"),
]


class CinematicWorkspace(CreateWorkspaceBase):
    """Full-window integrated 3D Cinematic Motion Workspace with presets & timeline scrub bar."""

    workspace_id = "cinematic_motion"
    display_title = "3D Cinematic"
    tool_save_label = "3D Cinematic video"

    def __init__(
        self,
        image_bgr: np.ndarray | list[np.ndarray],
        file_path: Path | list[Path] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if isinstance(image_bgr, list):
            self.multi_images = [
                cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                if len(img.shape) == 2
                else (cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if img.shape[2] == 4 else img.copy())
                for img in image_bgr
                if img is not None and img.size > 0
            ]
            self.original_bgr = self.multi_images[0]
        else:
            if len(image_bgr.shape) == 2:
                img = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
            elif image_bgr.shape[2] == 4:
                img = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
            else:
                img = image_bgr.copy()
            self.original_bgr = img
            self.multi_images = [self.original_bgr]

        if isinstance(file_path, list):
            self.file_paths = file_path
            self.file_path = file_path[0] if file_path else None
        else:
            self.file_path = file_path
            self.file_paths = [file_path] if file_path else []

        self.output_dir = self.file_path.parent if self.file_path else Path.home()
        self._worker: CinematicMotionWorker | None = None
        self._preview_frames: list[QPixmap] = []
        self._current_frame_idx: int = 0
        self._playing: bool = True
        self._dirty: bool = False

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._advance_preview_frame)

        h, w = self.original_bgr.shape[:2]
        pw, ph = compute_target_dimensions(w, h, max_long_edge=520)
        self._proxy_bgr = cv2.resize(self.original_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        self._proxy_layers = prepare_3d_parallax_layers(self._proxy_bgr)
        self._proxy_size = (pw, ph)

        self._build_canvas_ui()
        self._build_inspector_ui()
        self._build_bottom_tray_ui()
        self.apply_theme_mode("wlpg_aero")
        self._build_preview_loop()

    # ------------------------------------------------------------------
    # Workspace Contract
    # ------------------------------------------------------------------
    def inspector_widget(self) -> QWidget:
        return self._inspector_root

    def bottom_tray_widget(self) -> QWidget:
        return self._tray_root

    def tray_title(self) -> str:
        return "Motion Presets & Timeline"

    def ribbon_actions(self) -> list[RibbonActionSpec]:
        return [
            RibbonActionSpec("toggle_play", "Pause", "pause", "Play or pause preview (Space)", group="Playback", large=True),
            RibbonActionSpec("reset_motion", "Reset", "revert", "Reset trajectory to default 3D Orbit", group="Playback", large=True),
        ]

    def ribbon_action_states(self) -> dict[str, dict[str, Any]]:
        running = self._worker is not None and self._worker.isRunning()
        return {
            "create_ws_save": {"enabled": not running, "text": "Export..."},
            "toggle_play": {
                "enabled": not running,
                "text": "Pause" if self._playing else "Play",
                "icon_kind": "pause" if self._playing else "play",
            },
        }

    def handle_ribbon_action(self, action_id: str) -> None:
        if action_id == "toggle_play":
            self._toggle_playback()
        elif action_id == "reset_motion":
            self.cmb_motion.setCurrentIndex(2)
            self.sld_intensity.setValue(50)
            self.chk_bokeh.setChecked(True)
        else:
            super().handle_ribbon_action(action_id)

    def has_unsaved_changes(self) -> bool:
        return bool(self._dirty) and self.saved_file_path is None

    def save_result(self) -> Path | None:
        self._start_render()
        return self.saved_file_path

    def handle_key_press(self, event: QKeyEvent) -> bool:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._toggle_playback()
            return True
        return False

    def apply_theme_mode(self, theme_name: str) -> None:
        super().apply_theme_mode(theme_name)
        tok = get_theme_tokens(self._theme_name)
        qss = build_studio_panel_qss(self._theme_name)
        self.setStyleSheet(qss)
        self._inspector_root.setStyleSheet(qss)
        self._tray_root.setStyleSheet(qss)
        canvas_bg = "#111318" if tok.is_dark else "#232730"
        self.lbl_thumb.setStyleSheet(f"background-color: {canvas_bg}; border: none;")
        self.lbl_status.setStyleSheet(f"color: {tok.text_secondary}; font-size: 9pt; background: transparent;")
        self.lbl_res.setStyleSheet(f"color: {tok.text_secondary}; font-size: 9pt; background: transparent;")
        self.lbl_scrub_time.setStyleSheet(f"color: {tok.text_secondary}; font-size: 8.5pt; background: transparent;")

    def cleanup(self) -> None:
        self._timer.stop()
        stop_qthread_safely(self._worker, timeout_ms=3000)
        self._worker = None

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_canvas_ui(self) -> None:
        self.setObjectName("StudioCanvasRoot")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.lbl_thumb = QLabel(self)
        self.lbl_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_thumb.setMinimumHeight(240)
        lay.addWidget(self.lbl_thumb, 1)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setVisible(False)
        lay.addWidget(self.progress_bar)

        bottom_bar = QFrame(self)
        bottom_bar.setObjectName("StudioTrayRoot")
        b_lay = QHBoxLayout(bottom_bar)
        b_lay.setContentsMargins(10, 4, 10, 4)
        b_lay.setSpacing(10)

        self.lbl_status = QLabel("Live 3D Parallax Preview", bottom_bar)
        b_lay.addWidget(self.lbl_status, 1)

        h_orig, w_orig = self.original_bgr.shape[:2]
        exp_w, exp_h = compute_target_dimensions(w_orig, h_orig, max_long_edge=1920)
        self.lbl_res = QLabel(f"Export: {exp_w} × {exp_h} @ 30fps", bottom_bar)
        b_lay.addWidget(self.lbl_res)
        lay.addWidget(bottom_bar)

    def _build_inspector_ui(self) -> None:
        self._inspector_root = QWidget()
        self._inspector_root.setObjectName("StudioInspectorRoot")
        self._inspector_root.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        vbox = QVBoxLayout(self._inspector_root)
        vbox.setContentsMargins(4, 4, 4, 8)
        vbox.setSpacing(8)

        # 1. Camera Trajectory card
        grp_motion = StudioCardGroup("Camera & Motion", self._inspector_root)
        self.cmb_motion = QComboBox(grp_motion)
        for label, _ in _MOTION_STYLES:
            self.cmb_motion.addItem(label)
        self.cmb_motion.setCurrentIndex(2)
        self.cmb_motion.currentIndexChanged.connect(self._on_motion_combo_changed)
        grp_motion.add_widget(self.cmb_motion)

        lbl_i_hdr = QHBoxLayout()
        lbl_i_title = QLabel("Motion Amplitude:", grp_motion)
        self.lbl_intensity_val = QLabel("50%", grp_motion)
        lbl_i_hdr.addWidget(lbl_i_title)
        lbl_i_hdr.addStretch(1)
        lbl_i_hdr.addWidget(self.lbl_intensity_val)
        grp_motion.add_layout(lbl_i_hdr)

        self.sld_intensity = AeroStudioSlider(10, 100, 50, grp_motion)
        self.sld_intensity.valueChanged.connect(self._on_intensity_changed)
        grp_motion.add_widget(self.sld_intensity)

        self.chk_bokeh = QCheckBox("Optical depth blur (Bokeh)", grp_motion)
        self.chk_bokeh.setChecked(True)
        self.chk_bokeh.toggled.connect(lambda _c: self._mark_dirty_and_rebuild())
        grp_motion.add_widget(self.chk_bokeh)
        vbox.addWidget(grp_motion)

        # 2. Export Settings card
        grp_export = StudioCardGroup("Export Settings", self._inspector_root)
        grp_export.add_widget(QLabel("Format:", grp_export))
        self.cmb_format = QComboBox(grp_export)
        self.cmb_format.addItems(["MP4 Video (H.264)", "Animated GIF", "Animated WebP"])
        self.cmb_format.currentIndexChanged.connect(self._update_res_summary)
        grp_export.add_widget(self.cmb_format)

        grp_export.add_widget(QLabel("Clip Duration:", grp_export))
        self.cmb_duration = QComboBox(grp_export)
        self.cmb_duration.addItems(["3 seconds", "5 seconds", "8 seconds"])
        self.cmb_duration.setCurrentIndex(1)
        self.cmb_duration.currentIndexChanged.connect(self._update_res_summary)
        grp_export.add_widget(self.cmb_duration)

        grp_export.add_widget(QLabel("Max Resolution:", grp_export))
        self.cmb_max_res = QComboBox(grp_export)
        self.cmb_max_res.addItem("Full HD (1920 px)", 1920)
        self.cmb_max_res.addItem("HD (1280 px)", 1280)
        self.cmb_max_res.addItem("Web Share (800 px)", 800)
        self.cmb_max_res.currentIndexChanged.connect(self._update_res_summary)
        grp_export.add_widget(self.cmb_max_res)

        if len(self.multi_images) > 1:
            grp_export.add_widget(QLabel("Target:", grp_export))
            self.cmb_export_mode = QComboBox(grp_export)
            self.cmb_export_mode.addItem("Current Photo")
            self.cmb_export_mode.addItem(f"3D Memories Reel ({len(self.multi_images)} Photos)")
            self.cmb_export_mode.setCurrentIndex(1)
            grp_export.add_widget(self.cmb_export_mode)

        vbox.addWidget(grp_export)
        vbox.addStretch(1)

    def _build_bottom_tray_ui(self) -> None:
        self._tray_root = QFrame()
        self._tray_root.setObjectName("StudioTrayRoot")
        lay = QVBoxLayout(self._tray_root)
        lay.setContentsMargins(10, 4, 10, 6)
        lay.setSpacing(6)

        # Row 1: Motion Style preset pills
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        self._preset_btns: list[QPushButton] = []
        for idx, (label, _key) in enumerate(_MOTION_STYLES):
            b = QPushButton(label, self._tray_root)
            b.setCheckable(True)
            b.setChecked(idx == 2)
            b.setFixedHeight(28)
            b.clicked.connect(lambda _c=False, i=idx: self._select_motion_preset(i))
            self._preset_btns.append(b)
            preset_row.addWidget(b)
        preset_row.addStretch(1)
        lay.addLayout(preset_row)

        # Row 2: Play/Pause + timeline scrub slider
        scrub_row = QHBoxLayout()
        scrub_row.setSpacing(8)
        self.btn_play_pause = QPushButton("Pause", self._tray_root)
        self.btn_play_pause.setIcon(make_wlpg_icon("pause", 14))
        self.btn_play_pause.setFixedSize(82, 26)
        self.btn_play_pause.clicked.connect(self._toggle_playback)
        scrub_row.addWidget(self.btn_play_pause)

        self.sld_scrub = QSlider(Qt.Orientation.Horizontal, self._tray_root)
        self.sld_scrub.setRange(0, 23)
        self.sld_scrub.setValue(0)
        self.sld_scrub.sliderPressed.connect(self._on_scrub_pressed)
        self.sld_scrub.valueChanged.connect(self._on_scrub_moved)
        scrub_row.addWidget(self.sld_scrub, 1)

        self.lbl_scrub_time = QLabel("Frame 1 / 24", self._tray_root)
        self.lbl_scrub_time.setFixedWidth(92)
        scrub_row.addWidget(self.lbl_scrub_time)
        lay.addLayout(scrub_row)

    # ------------------------------------------------------------------
    # Playback & Preset Controls
    # ------------------------------------------------------------------
    def _select_motion_preset(self, idx: int) -> None:
        self.cmb_motion.setCurrentIndex(idx)

    def _on_motion_combo_changed(self, idx: int) -> None:
        for i, b in enumerate(self._preset_btns):
            b.blockSignals(True)
            b.setChecked(i == idx)
            b.blockSignals(False)
        self._mark_dirty_and_rebuild()

    def _mark_dirty_and_rebuild(self) -> None:
        self._dirty = True
        self._build_preview_loop()

    def _on_intensity_changed(self, val: int) -> None:
        self.lbl_intensity_val.setText(f"{val}%")
        self._mark_dirty_and_rebuild()

    def _update_res_summary(self) -> None:
        h_orig, w_orig = self.original_bgr.shape[:2]
        max_edge = int(self.cmb_max_res.currentData() or 1920)
        exp_w, exp_h = compute_target_dimensions(w_orig, h_orig, max_long_edge=max_edge)
        fmt_idx = self.cmb_format.currentIndex()
        fps = 30 if fmt_idx == 0 else 15
        self.lbl_res.setText(f"Export: {exp_w} × {exp_h} @ {fps}fps")

    def _toggle_playback(self) -> None:
        self._playing = not self._playing
        if self._playing:
            self._timer.start(33)
            self.btn_play_pause.setText("Pause")
            self.btn_play_pause.setIcon(make_wlpg_icon("pause", 14))
        else:
            self._timer.stop()
            self.btn_play_pause.setText("Play")
            self.btn_play_pause.setIcon(make_wlpg_icon("play", 14))
        self.ribbon_state_changed.emit()

    def _on_scrub_pressed(self) -> None:
        if self._playing:
            self._toggle_playback()

    def _on_scrub_moved(self, frame_idx: int) -> None:
        if 0 <= frame_idx < len(self._preview_frames):
            self._current_frame_idx = frame_idx
            self.lbl_scrub_time.setText(f"Frame {frame_idx + 1} / {len(self._preview_frames)}")
            self._update_current_pixmap()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._update_current_pixmap()

    def _update_current_pixmap(self) -> None:
        if not self._preview_frames or self._current_frame_idx >= len(self._preview_frames):
            return
        pm = self._preview_frames[self._current_frame_idx]
        avail_size = self.lbl_thumb.size()
        if avail_size.width() > 60 and avail_size.height() > 60:
            scaled = pm.scaled(QSize(avail_size.width() - 12, avail_size.height() - 12), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.lbl_thumb.setPixmap(scaled)
        else:
            self.lbl_thumb.setPixmap(pm)

    def _active_style_key(self) -> str:
        idx = max(0, min(len(_MOTION_STYLES) - 1, self.cmb_motion.currentIndex()))
        return _MOTION_STYLES[idx][1]

    def _build_preview_loop(self) -> None:
        self._timer.stop()
        style = self._active_style_key()
        intensity = self.sld_intensity.value() / 100.0
        enable_bokeh = self.chk_bokeh.isChecked()
        pw, ph = self._proxy_size

        self._preview_frames.clear()
        num_frames = 24
        for i in range(num_frames):
            frame = render_3d_cinematic_motion_frame(
                foreground_bgr=self._proxy_layers,
                t=i / float(num_frames),
                canvas_size=(pw, ph),
                motion_style=style,
                intensity=intensity,
                enable_bokeh=enable_bokeh,
            )
            self._preview_frames.append(_cv2_to_qpixmap(frame))

        if self._preview_frames:
            self._current_frame_idx = 0
            self.sld_scrub.blockSignals(True)
            self.sld_scrub.setRange(0, len(self._preview_frames) - 1)
            self.sld_scrub.setValue(0)
            self.sld_scrub.blockSignals(False)
            self.lbl_scrub_time.setText(f"Frame 1 / {len(self._preview_frames)}")
            self._update_current_pixmap()
            if self._playing:
                self._timer.start(33)

    def _advance_preview_frame(self) -> None:
        if not self._preview_frames:
            return
        self._current_frame_idx = (self._current_frame_idx + 1) % len(self._preview_frames)
        self.sld_scrub.blockSignals(True)
        self.sld_scrub.setValue(self._current_frame_idx)
        self.sld_scrub.blockSignals(False)
        self.lbl_scrub_time.setText(f"Frame {self._current_frame_idx + 1} / {len(self._preview_frames)}")
        self._update_current_pixmap()

    def _start_render(self) -> None:
        fmt_idx = self.cmb_format.currentIndex()
        is_reel = (
            hasattr(self, "cmb_export_mode")
            and self.cmb_export_mode.currentIndex() == 1
            and len(self.multi_images) > 1
        )
        default_tag = "3D_Memories_Reel" if is_reel else "cinematic3D"

        if fmt_idx == 1:
            out_path = prompt_save_new_file(self, self.file_path, default_tag, ".gif", "Animated GIF (*.gif)")
        elif fmt_idx == 2:
            out_path = prompt_save_new_file(self, self.file_path, default_tag, ".webp", "Animated WebP (*.webp)")
        else:
            out_path = prompt_save_new_file(self, self.file_path, default_tag, ".mp4", "MP4 Video (*.mp4)")
        if not out_path:
            return

        dur_text = self.cmb_duration.currentText()
        duration = 3.0 if "3" in dur_text else (8.0 if "8" in dur_text else 5.0)
        style = self._active_style_key()
        intensity = self.sld_intensity.value() / 100.0
        enable_bokeh = self.chk_bokeh.isChecked()
        max_edge = int(self.cmb_max_res.currentData() or 1920)

        if fmt_idx in (1, 2):
            # Synchronously render animated GIF / WebP via PIL
            self._timer.stop()
            self.progress_bar.setVisible(True)
            h_o, w_o = self.original_bgr.shape[:2]
            tw, th = compute_target_dimensions(w_o, h_o, max_long_edge=min(960, max_edge))
            resized = cv2.resize(self.original_bgr, (tw, th), interpolation=cv2.INTER_AREA)
            layers = prepare_3d_parallax_layers(resized)
            fps = 15
            total_frames = max(12, int(round(duration * fps)))
            pil_frames: list[Image.Image] = []
            for i in range(total_frames):
                bgr = render_3d_cinematic_motion_frame(
                    foreground_bgr=layers,
                    t=i / float(total_frames),
                    canvas_size=(tw, th),
                    motion_style=style,
                    intensity=intensity,
                    enable_bokeh=enable_bokeh,
                )
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil_frames.append(Image.fromarray(rgb))
                self.progress_bar.setValue(int(100 * (i + 1) / total_frames))
            pil_frames[0].save(
                str(out_path),
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(round(1000 / fps)),
                loop=0,
            )
            self.progress_bar.setVisible(False)
            self.saved_file_path = out_path
            self._dirty = False
            self.saved.emit(out_path)
            return

        self._timer.stop()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Preparing 3D depth layers and rendering MP4...")
        self.ribbon_state_changed.emit()

        images_payload = self.multi_images if is_reel else self.original_bgr
        self._worker = CinematicMotionWorker(
            image_bgr=images_payload,
            out_path=out_path,
            duration=duration,
            motion_style=style,
            intensity=intensity,
            enable_bokeh=enable_bokeh,
            is_reel=is_reel,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(lambda ok, msg: self._on_finished(ok, msg, out_path))
        self._worker.start()

    def _on_progress(self, val: int) -> None:
        self.progress_bar.setValue(val)
        self.lbl_status.setText(f"Encoding 3D virtual camera frames: {val}%")

    def _on_finished(self, ok: bool, msg: str, out_path: Path) -> None:
        self.progress_bar.setVisible(False)
        self.ribbon_state_changed.emit()
        if ok and out_path.exists():
            self.saved_file_path = out_path
            self._dirty = False
            self.saved.emit(out_path)
        else:
            QMessageBox.critical(self, "Render Error", f"Could not render 3D video:\n{msg}")
            if self._playing:
                self._timer.start(33)
