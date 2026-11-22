"""Integrated Microsoft ICE-Style Panorama Stitcher Workspace."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QRectF, QSize, Qt
from PySide6.QtGui import QIcon, QKeyEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.imaging.create_tools import postprocess_panorama, slice_panorama_carousel
from src.imaging.creative_effects import load_bgr_exif_safe
from src.ui.create_studio.base import CreateWorkspaceBase, RibbonActionSpec
from src.ui.create_studio.shared_widgets import (
    CreateStudioZoomBar,
    StudioCardGroup,
    ZoomablePreviewCanvas,
    _cv2_to_qpixmap,
    build_studio_panel_qss,
    prompt_save_new_file,
    stop_qthread_safely,
)
from src.ui.create_studio_dialogs import PanoramaWorker
from src.ui.ribbon_toolbar import make_wlpg_icon
from src.ui.theme_tokens import get_theme_tokens


class PanoramaWorkspace(CreateWorkspaceBase):
    """Full-window integrated Panorama Stitcher with interactive source frame strip."""

    workspace_id = "panorama"
    display_title = "Panorama"
    tool_save_label = "panorama"

    def __init__(self, photo_paths: list[Path], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.photo_paths: list[Path] = [Path(p) for p in photo_paths]
        self.output_dir = self.photo_paths[0].parent if self.photo_paths else Path.home()
        self.current_stitched_bgr: np.ndarray | None = None
        self.raw_stitched_bgr: np.ndarray | None = None
        self._worker: PanoramaWorker | None = None
        self._is_fit_mode: bool = True
        self._dirty: bool = False
        self._frame_entries: list[dict[str, Any]] = [
            {"path": p, "included": True, "thumb": None} for p in self.photo_paths
        ]

        self._build_canvas_ui()
        self._build_inspector_ui()
        self._build_bottom_tray_ui()
        self._populate_frame_strip()
        self.apply_theme_mode("wlpg_aero")
        self._load_and_stitch(mark_dirty=False)

    # ------------------------------------------------------------------
    # Workspace Contract
    # ------------------------------------------------------------------
    def inspector_widget(self) -> QWidget:
        return self._inspector_root

    def bottom_tray_widget(self) -> QWidget:
        return self._tray_root

    def tray_title(self) -> str:
        return "Source Photos (Reorder or Exclude Frames)"

    def ribbon_actions(self) -> list[RibbonActionSpec]:
        return [
            RibbonActionSpec(
                "restitch",
                "Re-stitch",
                "refresh",
                "Re-stitch included frames with active projection",
                group="Stitch",
                large=True,
            ),
            RibbonActionSpec(
                "auto_crop",
                "Auto-Crop",
                "crop",
                "Crop wavy black borders to largest clean rectangle",
                group="Stitch",
                large=True,
                checkable=True,
                checked=True,
            ),
            RibbonActionSpec(
                "fill_gaps",
                "Fill Gaps",
                "retouch",
                "Inpaint minor border corner gaps before cropping",
                group="Stitch",
                large=True,
                checkable=True,
                checked=False,
            ),
            RibbonActionSpec("zoom_fit", "Fit\nWindow", "fit_window", "Zoom to fit (F)", group="View", large=True),
            RibbonActionSpec("zoom_actual", "100%", "actual_size", "Actual pixels (1)", group="View", large=True),
        ]

    def ribbon_action_states(self) -> dict[str, dict[str, Any]]:
        running = self._worker is not None and self._worker.isRunning()
        included_cnt = sum(1 for e in self._frame_entries if e["included"])
        ready = self.current_stitched_bgr is not None and not running
        return {
            "create_ws_save": {"enabled": ready},
            "restitch": {"enabled": (not running) and included_cnt >= 2},
            "auto_crop": {"enabled": not running, "checked": self.chk_autocrop.isChecked()},
            "fill_gaps": {"enabled": not running, "checked": self.chk_inpaint.isChecked()},
        }

    def handle_ribbon_action(self, action_id: str) -> None:
        if action_id == "restitch":
            self._load_and_stitch(mark_dirty=True)
        elif action_id == "auto_crop":
            self.chk_autocrop.setChecked(not self.chk_autocrop.isChecked())
        elif action_id == "fill_gaps":
            self.chk_inpaint.setChecked(not self.chk_inpaint.isChecked())
        else:
            super().handle_ribbon_action(action_id)

    def has_unsaved_changes(self) -> bool:
        return bool(self._dirty) and self.current_stitched_bgr is not None and self.saved_file_path is None

    def save_result(self) -> Path | None:
        self._save_panorama()
        return self.saved_file_path

    def zoom_in(self) -> None:
        self._apply_zoom(1.25)

    def zoom_out(self) -> None:
        self._apply_zoom(1.0 / 1.25)

    def zoom_fit(self) -> None:
        if self.preview_scene.sceneRect().isValid() and self.preview_scene.sceneRect().width() > 0:
            self._is_fit_mode = True
            self.preview_view.fitInView(self.preview_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._on_zoom_changed(self.preview_view.transform().m11())

    def zoom_actual(self) -> None:
        if self.preview_scene.sceneRect().isValid():
            self._is_fit_mode = False
            self.preview_view.resetTransform()
            self._on_zoom_changed(1.0)

    def set_zoom_percent(self, pct: int) -> None:
        self._on_slider_zoom_moved(pct)

    def handle_key_press(self, event: QKeyEvent) -> bool:
        if event.key() == Qt.Key.Key_F:
            self.zoom_fit()
            return True
        if event.key() == Qt.Key.Key_1:
            self.zoom_actual()
            return True
        return False

    def apply_theme_mode(self, theme_name: str) -> None:
        super().apply_theme_mode(theme_name)
        tok = get_theme_tokens(self._theme_name)
        qss = build_studio_panel_qss(self._theme_name)
        self.setStyleSheet(qss)
        self._inspector_root.setStyleSheet(qss)
        self._tray_root.setStyleSheet(qss)
        self.preview_view.apply_theme_mode(self._theme_name)
        self.lbl_status.setStyleSheet(f"color: {tok.text_secondary}; font-size: 9pt; background: transparent;")
        self.lbl_dims.setStyleSheet(f"color: {tok.text_secondary}; font-size: 9pt; background: transparent;")
        self._refresh_frame_card_styles()

    def cleanup(self) -> None:
        stop_qthread_safely(self._worker, timeout_ms=2500)
        self._worker = None

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_canvas_ui(self) -> None:
        self.setObjectName("StudioCanvasRoot")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.preview_view = ZoomablePreviewCanvas(self)
        self.preview_scene = QGraphicsScene(self)
        self.preview_view.setScene(self.preview_scene)
        self.preview_pixmap_item = QGraphicsPixmapItem()
        self.preview_scene.addItem(self.preview_pixmap_item)
        self.preview_view.zoom_changed.connect(self._on_zoom_changed)
        self.preview_view.viewport().installEventFilter(self)
        lay.addWidget(self.preview_view, 1)

        bottom_bar = QFrame(self)
        bottom_bar.setObjectName("StudioTrayRoot")
        b_lay = QHBoxLayout(bottom_bar)
        b_lay.setContentsMargins(10, 2, 10, 2)
        b_lay.setSpacing(8)

        self.progress_bar = QProgressBar(bottom_bar)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedSize(120, 14)
        self.progress_bar.setVisible(False)
        b_lay.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Ready", bottom_bar)
        b_lay.addWidget(self.lbl_status, 1)

        self.lbl_dims = QLabel("", bottom_bar)
        b_lay.addWidget(self.lbl_dims)
        lay.addWidget(bottom_bar)

    def _build_inspector_ui(self) -> None:
        self._inspector_root = QWidget()
        self._inspector_root.setObjectName("StudioInspectorRoot")
        self._inspector_root.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        vbox = QVBoxLayout(self._inspector_root)
        vbox.setContentsMargins(4, 4, 4, 8)
        vbox.setSpacing(8)

        grp_proj = StudioCardGroup("Projection Mode", self._inspector_root)
        self.cmb_projection = QComboBox(grp_proj)
        self.cmb_projection.addItems([
            "Cylindrical (Standard Wide)",
            "Spherical (360 / Multi-Row)",
            "Planar / Perspective (Document / Whiteboard)",
        ])
        self.cmb_projection.currentIndexChanged.connect(self._on_projection_changed)
        grp_proj.add_widget(self.cmb_projection)

        self.btn_restitch = QPushButton("Re-stitch Panorama", grp_proj)
        self.btn_restitch.setIcon(make_wlpg_icon("refresh", 16))
        self.btn_restitch.clicked.connect(lambda: self._load_and_stitch(mark_dirty=True))
        grp_proj.add_widget(self.btn_restitch)
        vbox.addWidget(grp_proj)

        grp_border = StudioCardGroup("Border & Edge Treatment", self._inspector_root)
        self.chk_autocrop = QCheckBox("Auto-crop black edges (ICE)", grp_border)
        self.chk_autocrop.setChecked(True)
        self.chk_autocrop.setToolTip("Computes the largest interior rectangle with zero black border pixels.")
        self.chk_autocrop.toggled.connect(lambda _c: self._on_crop_or_inpaint_toggled(mark_dirty=True))
        grp_border.add_widget(self.chk_autocrop)

        self.chk_inpaint = QCheckBox("AI Fill Border Gaps (Gen AI)", grp_border)
        self.chk_inpaint.setChecked(False)
        self.chk_inpaint.setToolTip(
            "Uses the LaMa generative AI inpainting engine (same as Crop AI Expand) to fill wavy stitched border gaps."
        )
        self.chk_inpaint.toggled.connect(self._on_inpaint_checkbox_toggled)
        grp_border.add_widget(self.chk_inpaint)
        vbox.addWidget(grp_border)

        grp_carousel = StudioCardGroup("Instagram Carousel Slicer", self._inspector_root)
        c_row = QHBoxLayout()
        c_row.setSpacing(6)
        c_row.addWidget(QLabel("Slides (4:5):", grp_carousel))
        self.cmb_carousel_slides = QComboBox(grp_carousel)
        self.cmb_carousel_slides.addItems(["2 Slides", "3 Slides", "4 Slides", "5 Slides"])
        self.cmb_carousel_slides.setCurrentIndex(1)
        c_row.addWidget(self.cmb_carousel_slides, 1)
        grp_carousel.add_layout(c_row)

        self.btn_slice_carousel = QPushButton("Export 4:5 Carousel...", grp_carousel)
        self.btn_slice_carousel.setIcon(make_wlpg_icon("collage", 16))
        self.btn_slice_carousel.setToolTip("Slice wide panorama into seamless 1080×1350 (4:5) swipeable carousel slides")
        self.btn_slice_carousel.clicked.connect(self._export_carousel_slides)
        grp_carousel.add_widget(self.btn_slice_carousel)
        vbox.addWidget(grp_carousel)

        grp_info = StudioCardGroup("Output Info", self._inspector_root)
        self.lbl_source_info = QLabel("", grp_info)
        self.lbl_source_info.setProperty("uiRole", "muted")
        self.lbl_source_info.setWordWrap(True)
        grp_info.add_widget(self.lbl_source_info)
        vbox.addWidget(grp_info)
        vbox.addStretch(1)

    def _build_bottom_tray_ui(self) -> None:
        self._tray_root = QFrame()
        self._tray_root.setObjectName("StudioTrayRoot")
        lay = QHBoxLayout(self._tray_root)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(8)

        self.frame_scroll = QScrollArea(self._tray_root)
        self.frame_scroll.setObjectName("StudioTrayScroll")
        self.frame_scroll.setWidgetResizable(True)
        self.frame_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.frame_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.frame_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.frame_host = QWidget(self.frame_scroll)
        self.frame_host.setObjectName("StudioTrayHost")
        self.frame_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.frame_layout = QHBoxLayout(self.frame_host)
        self.frame_layout.setContentsMargins(2, 2, 2, 2)
        self.frame_layout.setSpacing(8)
        self.frame_layout.addStretch(1)
        self.frame_scroll.setWidget(self.frame_host)
        lay.addWidget(self.frame_scroll, 1)

        self.btn_add_photos = QPushButton("+ Add Photos...", self._tray_root)
        self.btn_add_photos.setIcon(make_wlpg_icon("folder_add", 16))
        self.btn_add_photos.setFixedHeight(32)
        self.btn_add_photos.clicked.connect(self._on_add_photos_clicked)
        lay.addWidget(self.btn_add_photos)

        self._frame_cards: list[QFrame] = []

    def _populate_frame_strip(self) -> None:
        while self.frame_layout.count() > 1:
            child = self.frame_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self._frame_cards.clear()

        for idx, entry in enumerate(self._frame_entries):
            p: Path = entry["path"]
            if entry["thumb"] is None:
                img = load_bgr_exif_safe(p)
                if img is not None:
                    h, w = img.shape[:2]
                    sc = 54.0 / max(1, max(h, w))
                    small = cv2.resize(img, (max(16, int(round(w * sc))), max(16, int(round(h * sc)))), interpolation=cv2.INTER_AREA)
                    entry["thumb"] = _cv2_to_qpixmap(small)
            card = QFrame(self.frame_host)
            card.setObjectName("PanoTrayCard")
            card.setProperty("_frame_idx", idx)
            card.setFixedSize(114, 92)
            c_lay = QVBoxLayout(card)
            c_lay.setContentsMargins(4, 3, 4, 3)
            c_lay.setSpacing(2)

            top_row = QHBoxLayout()
            top_row.setSpacing(3)
            chk = QCheckBox(f"#{idx + 1}", card)
            chk.setChecked(bool(entry["included"]))
            chk.setToolTip(f"Include {p.name} in panorama stitch")
            chk.toggled.connect(lambda val, i=idx: self._toggle_frame_included(i, val))
            top_row.addWidget(chk)
            top_row.addStretch(1)

            btn_left = QPushButton("◀", card)
            btn_left.setObjectName("TrayMiniBtn")
            btn_left.setFixedSize(18, 18)
            btn_left.setEnabled(idx > 0)
            btn_left.setToolTip("Move frame left")
            btn_left.clicked.connect(lambda _c=False, i=idx: self._move_frame(i, -1))
            top_row.addWidget(btn_left)

            btn_right = QPushButton("▶", card)
            btn_right.setObjectName("TrayMiniBtn")
            btn_right.setFixedSize(18, 18)
            btn_right.setEnabled(idx < len(self._frame_entries) - 1)
            btn_right.setToolTip("Move frame right")
            btn_right.clicked.connect(lambda _c=False, i=idx: self._move_frame(i, 1))
            top_row.addWidget(btn_right)
            c_lay.addLayout(top_row)

            lbl_img = QLabel(card)
            lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_img.setStyleSheet("border: none; background: transparent;")
            if entry["thumb"] is not None:
                lbl_img.setPixmap(entry["thumb"].scaled(102, 50, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            c_lay.addWidget(lbl_img, 1)

            lbl_name = QLabel(p.name, card)
            lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_name.setMaximumWidth(104)
            lbl_name.setStyleSheet("border: none; background: transparent; font-size: 8pt;")
            c_lay.addWidget(lbl_name)

            self._frame_cards.append(card)
            self.frame_layout.insertWidget(self.frame_layout.count() - 1, card)

        self._refresh_frame_card_styles()
        self._update_source_info()

    def _refresh_frame_card_styles(self) -> None:
        tok = get_theme_tokens(self._theme_name)
        for idx, card in enumerate(self._frame_cards):
            inc = idx < len(self._frame_entries) and self._frame_entries[idx]["included"]
            border_c = tok.accent if inc else tok.border_subtle
            bg_c = tok.bg_elevated if inc else tok.bg_card
            card.setStyleSheet(
                f"QFrame#PanoTrayCard {{ background: {bg_c}; border: 1.5px solid {border_c}; border-radius: 6px; }}"
            )

    def _update_source_info(self) -> None:
        inc = sum(1 for e in self._frame_entries if e["included"])
        self.lbl_source_info.setText(
            f"Stitching {inc} of {len(self._frame_entries)} photo(s) into a wide-angle composite."
        )

    def _toggle_frame_included(self, idx: int, included: bool) -> None:
        if 0 <= idx < len(self._frame_entries):
            self._frame_entries[idx]["included"] = bool(included)
            self._dirty = True
            self._refresh_frame_card_styles()
            self._update_source_info()
            self.ribbon_state_changed.emit()

    def _move_frame(self, idx: int, delta: int) -> None:
        target = idx + delta
        if not (0 <= idx < len(self._frame_entries) and 0 <= target < len(self._frame_entries)):
            return
        self._frame_entries[idx], self._frame_entries[target] = self._frame_entries[target], self._frame_entries[idx]
        self._dirty = True
        self._populate_frame_strip()
        self.ribbon_state_changed.emit()

    def _on_add_photos_clicked(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Panorama Frames",
            str(self.output_dir),
            "Images (*.jpg *.jpeg *.png *.webp *.tif *.tiff)",
        )
        added = False
        existing = {str(e["path"].resolve()) for e in self._frame_entries if e["path"].exists()}
        for f in files:
            p = Path(f)
            if str(p.resolve()) not in existing:
                self._frame_entries.append({"path": p, "included": True, "thumb": None})
                existing.add(str(p.resolve()))
                added = True
        if added:
            self._dirty = True
            self._populate_frame_strip()
            self.ribbon_state_changed.emit()

    # ------------------------------------------------------------------
    # EventFilter & Zoom
    # ------------------------------------------------------------------
    def eventFilter(self, watched: Any, event: QEvent) -> bool:
        if watched is self.preview_view.viewport() and event.type() == QEvent.Type.Resize:
            if self._is_fit_mode and self.preview_scene.sceneRect().isValid() and self.preview_scene.sceneRect().width() > 0:
                self.preview_view.fitInView(self.preview_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
                self._on_zoom_changed(self.preview_view.transform().m11())
        return super().eventFilter(watched, event)

    def _apply_zoom(self, factor: float) -> None:
        self._is_fit_mode = False
        self.preview_view.scale(factor, factor)
        self._on_zoom_changed(self.preview_view.transform().m11())

    def _on_slider_zoom_moved(self, pct: int) -> None:
        self._is_fit_mode = False
        target_scale = max(0.1, min(5.0, pct / 100.0))
        current_scale = self.preview_view.transform().m11()
        if abs(current_scale) > 1e-4:
            self.preview_view.scale(target_scale / current_scale, target_scale / current_scale)
        self._on_zoom_changed(target_scale)

    def _on_zoom_changed(self, scale_val: float) -> None:
        self.zoom_state_changed.emit(float(scale_val), bool(self._is_fit_mode))

    # ------------------------------------------------------------------
    # Stitching & Post-Processing
    # ------------------------------------------------------------------
    def _on_projection_changed(self) -> None:
        self._load_and_stitch(mark_dirty=True)

    def _load_and_stitch(self, *, mark_dirty: bool = True) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        active_paths = [e["path"] for e in self._frame_entries if e["included"]]
        if len(active_paths) < 2:
            self.lbl_status.setText("Include at least 2 frames to stitch a panorama.")
            self.progress_bar.setVisible(False)
            self.ribbon_state_changed.emit()
            return

        if mark_dirty:
            self._dirty = True
        self.progress_bar.setVisible(True)
        self.btn_restitch.setEnabled(False)
        self.lbl_status.setText("Loading photos & initializing stitcher...")
        self.ribbon_state_changed.emit()

        mode_text = self.cmb_projection.currentText()
        proj_mode = "planar" if "Planar" in mode_text else ("spherical" if "Spherical" in mode_text else "cylindrical")

        self._worker = PanoramaWorker(
            images_bgr=None,
            projection_mode=proj_mode,
            photo_paths=active_paths,
        )
        self._worker.progress_updated.connect(self.lbl_status.setText)
        self._worker.raw_stitching_finished.connect(self._on_raw_stitch_completed)
        self._worker.start()

    def _on_raw_stitch_completed(self, success: bool, raw_bgr: np.ndarray, msg: str) -> None:
        self.progress_bar.setVisible(False)
        self.btn_restitch.setEnabled(True)

        if not success or raw_bgr is None or raw_bgr.size == 0:
            self.lbl_status.setText(msg)
            self.ribbon_state_changed.emit()
            QMessageBox.warning(self, "Panorama Stitching", f"Could not stitch panorama:\n{msg}")
            return

        self.raw_stitched_bgr = raw_bgr
        self._on_crop_or_inpaint_toggled(mark_dirty=False)
        self.lbl_status.setText(msg)

    def _on_inpaint_checkbox_toggled(self, checked: bool) -> None:
        if checked:
            try:
                from src.imaging.creative_effects import get_lama_inpaint_engine

                if not get_lama_inpaint_engine().has_neural_model():
                    from src.ui.dialogs import prompt_and_download_lama_addon

                    prompt_and_download_lama_addon(self, "Panorama AI Gap Fill")
            except Exception:
                pass
        self._on_crop_or_inpaint_toggled(mark_dirty=True)

    def _on_crop_or_inpaint_toggled(self, *, mark_dirty: bool = True) -> None:
        if self.raw_stitched_bgr is None:
            return
        if mark_dirty:
            self._dirty = True

        ai_fill = self.chk_inpaint.isChecked()
        self.current_stitched_bgr = postprocess_panorama(
            self.raw_stitched_bgr,
            auto_crop=self.chk_autocrop.isChecked(),
            fill_minor_gaps=ai_fill,
            allow_full_border_ai=ai_fill,
        )
        h, w = self.current_stitched_bgr.shape[:2]
        self.lbl_dims.setText(f"{w} × {h} px ({max(1, (w * h) // 1_000_000)} MP)")
        if ai_fill:
            try:
                from src.imaging.creative_effects import get_lama_inpaint_engine

                engine_tag = "LaMa Gen AI" if get_lama_inpaint_engine().has_neural_model() else "built-in inpaint"
                self.lbl_status.setText(f"Border gaps filled ({engine_tag}).")
            except Exception:
                pass

        pm = _cv2_to_qpixmap(self.current_stitched_bgr)
        self.preview_pixmap_item.setPixmap(pm)
        self.preview_scene.setSceneRect(QRectF(pm.rect()))
        if self._is_fit_mode:
            self.zoom_fit()
        self.ribbon_state_changed.emit()

    def _export_carousel_slides(self) -> None:
        if self.current_stitched_bgr is None:
            return
        slide_count = self.cmb_carousel_slides.currentIndex() + 2
        out_dir = QFileDialog.getExistingDirectory(self, "Choose Folder for 4:5 Carousel Slides", str(self.output_dir))
        if not out_dir:
            return
        slides = slice_panorama_carousel(self.current_stitched_bgr, slide_count=slide_count)
        stem = self.photo_paths[0].stem if self.photo_paths else "panorama"
        dest_dir = Path(out_dir)
        saved_paths: list[Path] = []
        for idx, slide in enumerate(slides, 1):
            out_p = dest_dir / f"{stem}_carousel_{idx}_of_{len(slides)}.jpg"
            cv2.imwrite(str(out_p), slide, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_paths.append(out_p)
        if saved_paths:
            self.lbl_status.setText(f"Exported {len(saved_paths)} seamless 4:5 carousel slides to {dest_dir.name}.")
            self.status_message.emit(f"✓ Exported {len(saved_paths)} carousel slides", 4000)

    def _save_panorama(self) -> None:
        if self.current_stitched_bgr is None:
            return
        first_path = self.photo_paths[0] if self.photo_paths else None
        out_path = prompt_save_new_file(self, first_path, "panorama", ".jpg")
        if not out_path:
            return
        try:
            cv2.imwrite(str(out_path), self.current_stitched_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            self.saved_file_path = out_path
            self._dirty = False
            self.saved.emit(out_path)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save panorama:\n{e}")
