"""
Creation Studio Dialogs:
Authentic Windows Live Photo Gallery Create Tab Studio Dialogs:
1. PanoramaDialog — Microsoft ICE-Style Panorama Stitcher with auto-crop of wavy black borders & gap inpainting.
2. CollageDialog — Interactive QGraphicsView Collage & 'Then & Now' Studio (Grid, Polaroid Scatter, and Then & Now).
3. PhotoFuseDialog — Best-Face Group Shot Combiner with burst homography alignment and seamless Poisson face cloning.
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Any
import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, QRectF, QPointF, QPoint, QTimer, QSize
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QColor,
    QIcon,
    QPainter,
    QPen,
    QBrush,
    QFont,
    QCursor,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QSlider,
    QSpinBox,
    QColorDialog,
    QProgressBar,
    QMessageBox,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsEllipseItem,
    QGraphicsTextItem,
    QScrollArea,
    QWidget,
    QFrame,
    QFileDialog,
    QButtonGroup,
    QToolButton,
)

from src.imaging.create_tools import (
    stitch_panorama,
    stitch_panorama_raw,
    postprocess_panorama,
    create_grid_collage,
    create_blended_seamless_collage,
    create_polaroid_scatter_collage,
    create_then_and_now_card,
    align_burst_frame,
    detect_faces_in_image,
    detect_faces_with_landmarks,
    fuse_face,
)
from src.imaging.creative_effects import (
    load_bgr_exif_safe,
    guided_filter_matte,
    estimate_subject_mask,
    estimate_depth_map,
    compute_color_similarity_mask,
    compute_multi_color_similarity_mask,
    apply_color_pop,
    refine_mask_brush,
    build_bokeh_pyramid,
    apply_portrait_bokeh,
    apply_portrait_bokeh_and_fill_light,
    compute_target_dimensions,
    prepare_3d_parallax_layers,
    render_3d_cinematic_motion_frame,
    render_3d_cinematic_motion_video,
    render_3d_cinematic_motion_reel,
)
from src.ui.ribbon_toolbar import make_wlpg_icon


# =============================================================================
# Helper Utilities
# =============================================================================

def _cv2_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    """Converts a BGR numpy array to QPixmap safely without dangling memory."""
    if img_bgr is None or img_bgr.size == 0:
        return QPixmap()
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    h, w = img_bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _qcolor_to_bgr(color: QColor) -> tuple[int, int, int]:
    return color.blue(), color.green(), color.red()


def prompt_save_new_file(
    parent: QWidget,
    base_file_path: Path | None,
    suffix: str,
    ext: str = ".jpg",
    filter_str: str = "JPEG Image (*.jpg *.jpeg);;All Files (*.*)",
) -> Path | None:
    """
    Prompts the user to save the newly created item:
    - Makes it crystal clear that a new file is being created.
    - Suggests a default filename based on the first photo's stem + suffix (e.g. DSC3125_colourPop.jpg).
    - Opens a standard Save File dialog rooted in the source directory.
    """
    if base_file_path:
        stem = base_file_path.stem
        src_dir = base_file_path.parent
    else:
        stem = "Photo"
        src_dir = Path.home()

    default_name = f"{stem}_{suffix}{ext}"
    initial_path = (src_dir / default_name).as_posix()

    selected_path, _ = QFileDialog.getSaveFileName(
        parent,
        f"Save New File ({suffix})",
        initial_path,
        filter_str,
    )
    if not selected_path:
        return None
    return Path(selected_path)


# =============================================================================
# Reusable WLPG Studio UI Components (Fine-Tune Style)
# =============================================================================

class _FineTuneGroup(QFrame):
    """Clean WLPG white rounded card matching right_tag_dock."""
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("FineTuneGroup")
        self.setStyleSheet("""
            QFrame#FineTuneGroup {
                background: #FFFFFF;
                border: 1px solid #D2DFEE;
                border-radius: 4px;
            }
        """)
        self.vbox = QVBoxLayout(self)
        self.vbox.setContentsMargins(10, 8, 10, 10)
        self.vbox.setSpacing(6)
        if title:
            lbl = QLabel(title, self)
            lbl.setStyleSheet("color: #29588C; font-family: 'Segoe UI'; font-size: 9.5pt; font-weight: 600; border: none;")
            self.vbox.addWidget(lbl)


class _AeroFineTuneSlider(QSlider):
    """High-fidelity Aero horizontal slider with double-click to reset."""
    def __init__(
        self,
        min_val: int = 0,
        max_val: int = 100,
        default_val: int = 50,
        parent: QWidget | None = None,
        default_value: int | None = None,
    ) -> None:
        super().__init__(Qt.Horizontal, parent)
        if default_value is not None:
            default_val = default_value
        self.default_val = default_val
        self.setRange(min_val, max_val)
        self.setValue(default_val)
        self.setFixedHeight(18)
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #D8DFE6;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0078D7;
                border: 1px solid #005A9E;
                width: 10px;
                margin: -5px 0;
                border-radius: 2px;
            }
            QSlider::handle:horizontal:hover {
                background: #2B88D8;
            }
        """)

    def mouseDoubleClickEvent(self, event) -> None:
        self.setValue(self.default_val)
        event.accept()


class CreateStudioZoomBar(QFrame):
    """
    Standard Windows Live Photo Gallery zoom status bar component:
    Matches main_window bottom status bar controls exactly (Fit/100%, Zoom Out, Slider, Zoom In, Pct).
    """
    zoom_in_clicked = Signal()
    zoom_out_clicked = Signal()
    fit_toggle_clicked = Signal()
    zoom_slider_moved = Signal(int)

    # Aliases
    fitRequested = fit_toggle_clicked
    zoomInRequested = zoom_in_clicked
    zoomOutRequested = zoom_out_clicked
    zoomSliderMoved = zoom_slider_moved

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(24)
        self.setStyleSheet("""
            QFrame {
                background: transparent;
                border: none;
            }
            QPushButton {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 2px;
                padding: 1px;
            }
            QPushButton:hover {
                background: #E5F1FB;
                border: 1px solid #70C0E7;
            }
            QPushButton:pressed {
                background: #CCE4F7;
                border: 1px solid #0078D7;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #B4B4B4;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0078D7;
                border: 1px solid #005A9E;
                width: 10px;
                margin: -5px 0;
                border-radius: 2px;
            }
            QSlider::handle:horizontal:hover {
                background: #2B88D8;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.btn_fit = QPushButton(self)
        self.btn_fit.setFixedSize(22, 22)
        self.btn_fit.setIcon(make_wlpg_icon("actual_size", 16))
        self.btn_fit.setToolTip("Fit to window / Actual size (Ctrl+0)")
        self.btn_fit.clicked.connect(self.fit_toggle_clicked.emit)
        layout.addWidget(self.btn_fit)

        self.btn_zoom_out = QPushButton(self)
        self.btn_zoom_out.setFixedSize(22, 22)
        self.btn_zoom_out.setIcon(make_wlpg_icon("zoom_out", 16))
        self.btn_zoom_out.setToolTip("Zoom out (-)")
        self.btn_zoom_out.clicked.connect(self.zoom_out_clicked.emit)
        layout.addWidget(self.btn_zoom_out)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(20, 300)
        self.slider.setValue(100)
        self.slider.setFixedWidth(95)
        self.slider.setToolTip("Zoom level")
        self.slider.valueChanged.connect(self.zoom_slider_moved.emit)
        layout.addWidget(self.slider)

        self.btn_zoom_in = QPushButton(self)
        self.btn_zoom_in.setFixedSize(22, 22)
        self.btn_zoom_in.setIcon(make_wlpg_icon("zoom_in", 16))
        self.btn_zoom_in.setToolTip("Zoom in (+)")
        self.btn_zoom_in.clicked.connect(self.zoom_in_clicked.emit)
        layout.addWidget(self.btn_zoom_in)

        self.lbl_pct = QLabel("100%", self)
        self.lbl_pct.setStyleSheet("font-size: 11px; font-weight: 600; color: #444; min-width: 38px; padding-left: 2px;")
        layout.addWidget(self.lbl_pct)

    def set_zoom_level(self, percent: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(max(20, min(300, percent)))
        self.slider.blockSignals(False)
        self.lbl_pct.setText(f"{percent}%")

    def update_zoom_display(self, scale_val: float) -> None:
        pct = int(round(scale_val * 100))
        self.set_zoom_level(pct)

    def set_is_fit(self, is_fit: bool) -> None:
        if is_fit:
            self.btn_fit.setToolTip("Switch to 100% actual size (Ctrl+0)")
        else:
            self.btn_fit.setToolTip("Fit photo to canvas window (Ctrl+0)")


class ZoomablePreviewCanvas(QGraphicsView):
    """
    Interactive zoomable/pannable canvas for creative tools previews:
    - Wheel zoom with zoom_changed signal.
    - Space+Drag and Middle-button drag for fluid panning.
    """
    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setStyleSheet("background: #1C1C1E; border: 1px solid #555558; border-radius: 4px;")
        self._space_pressed: bool = False

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)
        self.zoom_changed.emit(self.transform().m11())
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() in (Qt.MiddleButton, Qt.LeftButton) and (self._space_pressed or event.button() == Qt.MiddleButton):
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            super().mousePressEvent(event)
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() in (Qt.MiddleButton, Qt.LeftButton):
            if not self._space_pressed:
                self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space and not self._space_pressed:
            self._space_pressed = True
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key_Space:
            self._space_pressed = False
            self.setDragMode(QGraphicsView.NoDrag)
            event.accept()
            return
        super().keyReleaseEvent(event)


# =============================================================================
# 1. Panorama Stitcher Dialog (Microsoft ICE-Style)
# =============================================================================

class PanoramaWorker(QThread):
    """Background worker for multi-image loading & stitching."""
    progress_updated = Signal(str)
    raw_stitching_finished = Signal(bool, object, str)

    def __init__(
        self,
        images_bgr: list[np.ndarray] | None = None,
        projection_mode: str = "cylindrical",
        photo_paths: list[Path] | None = None,
    ) -> None:
        super().__init__()
        self.images_bgr = images_bgr
        self.photo_paths = photo_paths or []
        self.projection_mode = projection_mode

    def run(self) -> None:
        imgs = self.images_bgr
        if imgs is None:
            imgs = []
            for idx, p in enumerate(self.photo_paths, start=1):
                if self.isInterruptionRequested():
                    return
                self.progress_updated.emit(f"Loading photo {idx} of {len(self.photo_paths)}...")
                img = load_bgr_exif_safe(p)
                if img is not None:
                    imgs.append(img)
        if len(imgs) < 2:
            self.raw_stitching_finished.emit(False, np.empty((0, 0, 3), dtype=np.uint8), "Could not load 2+ valid images.")
            return
        if self.isInterruptionRequested():
            return
        self.progress_updated.emit("Analyzing feature keypoints & camera trajectories...")
        success, raw_bgr, msg = stitch_panorama_raw(
            imgs,
            projection_mode=self.projection_mode,
        )
        self.raw_stitching_finished.emit(success, raw_bgr, msg)


class PanoramaDialog(QDialog):
    """
    Microsoft ICE-Style Panorama Stitcher Dialog:
    - Projection modes: Cylindrical, Spherical, Planar/Perspective.
    - Signature ICE auto-crop of wavy black borders to largest clean interior rectangle.
    - Telea inpainting of minor border gaps.
    - High-res export to same directory as source photos.
    """

    def __init__(
        self,
        photo_paths: list[Path],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.photo_paths = photo_paths
        self.output_dir = photo_paths[0].parent if photo_paths else Path.home()
        self.saved_file_path: Path | None = None
        self.current_stitched_bgr: np.ndarray | None = None
        self.raw_stitched_bgr: np.ndarray | None = None
        self._worker: PanoramaWorker | None = None
        self._is_fit_mode: bool = True

        self.setWindowTitle("Create Panorama — Essentials Photo Gallery")
        self.setWindowIcon(make_wlpg_icon("panoramic", 24))
        self.resize(1240, 800)
        self.setMinimumSize(960, 620)
        self._init_ui()
        self._load_and_stitch()

    def _init_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ---------------- Left: Canvas & Status / Zoom Bar ----------------
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.preview_view = ZoomablePreviewCanvas(self)
        self.preview_scene = QGraphicsScene(self)
        self.preview_view.setScene(self.preview_scene)
        self.preview_view.setStyleSheet("background: #1C1C1E; border: 1px solid #555558; border-radius: 4px;")
        self.preview_pixmap_item = QGraphicsPixmapItem()
        self.preview_scene.addItem(self.preview_pixmap_item)
        self.preview_view.zoom_changed.connect(self._on_zoom_changed)
        left_layout.addWidget(self.preview_view, 1)

        # Status & Zoom Controls Bar
        status_bar = QHBoxLayout()
        status_bar.setContentsMargins(4, 2, 4, 2)
        status_bar.setSpacing(8)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedSize(110, 12)
        self.progress_bar.setStyleSheet(
            "QProgressBar { height: 12px; border: 1px solid #c0c0c0; border-radius: 2px; background: #e0e0e0; } "
            "QProgressBar::chunk { background: #2D72BC; }"
        )
        self.progress_bar.setVisible(False)
        status_bar.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Ready", self)
        self.lbl_status.setStyleSheet("color: #444; font-size: 11px;")
        status_bar.addWidget(self.lbl_status)

        status_bar.addStretch(1)

        self.lbl_dims = QLabel("", self)
        self.lbl_dims.setStyleSheet("color: #666; font-size: 11px; margin-right: 8px;")
        status_bar.addWidget(self.lbl_dims)

        self.zoom_bar = CreateStudioZoomBar(self)
        self.zoom_bar.fitRequested.connect(self._toggle_fit_zoom)
        self.zoom_bar.zoomInRequested.connect(lambda: self._apply_zoom(1.25))
        self.zoom_bar.zoomOutRequested.connect(lambda: self._apply_zoom(1.0 / 1.25))
        self.zoom_bar.zoomSliderMoved.connect(self._on_slider_zoom_moved)
        status_bar.addWidget(self.zoom_bar)

        left_layout.addLayout(status_bar)
        main_layout.addWidget(left_widget, 1)

        # ---------------- Right: Settings Sidebar ----------------
        sidebar_scroll = QScrollArea(self)
        sidebar_scroll.setFixedWidth(320)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        sidebar_content = QWidget()
        sidebar_content.setMaximumWidth(312)
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(2, 0, 4, 0)
        sidebar_layout.setSpacing(8)

        # 1. Projection Mode
        grp_proj = _FineTuneGroup("Projection Mode")
        self.cmb_projection = QComboBox(self)
        self.cmb_projection.addItems([
            "Cylindrical (Standard Wide)",
            "Spherical (360 / Multi-Row)",
            "Planar / Perspective (Document / Whiteboard Scan)",
        ])
        self.cmb_projection.currentIndexChanged.connect(self._on_settings_changed)
        grp_proj.layout().addWidget(self.cmb_projection)

        self.btn_restitch = QPushButton(" Re-Stitch Panorama", self)
        self.btn_restitch.setIcon(make_wlpg_icon("refresh", 16))
        self.btn_restitch.setStyleSheet("padding: 5px 10px; border: 1px solid #7a96df; border-radius: 3px; background: #eef3fa;")
        self.btn_restitch.clicked.connect(self._load_and_stitch)
        grp_proj.layout().addWidget(self.btn_restitch)
        sidebar_layout.addWidget(grp_proj)

        # 2. Border & Edge Treatment
        grp_border = _FineTuneGroup("Border & Edge Treatment")
        self.chk_autocrop = QCheckBox("Auto-crop black edges (ICE)")
        self.chk_autocrop.setChecked(True)
        self.chk_autocrop.setToolTip("Computes the largest axis-aligned interior rectangle containing zero black border pixels.")
        self.chk_autocrop.toggled.connect(self._on_crop_or_inpaint_toggled)
        grp_border.layout().addWidget(self.chk_autocrop)

        self.chk_inpaint = QCheckBox("Fill minor border gaps (Inpaint)")
        self.chk_inpaint.setChecked(False)
        self.chk_inpaint.setToolTip("Inpaints minor sky/grass corner gaps using Telea diffusion before cropping.")
        self.chk_inpaint.toggled.connect(self._on_crop_or_inpaint_toggled)
        grp_border.layout().addWidget(self.chk_inpaint)
        sidebar_layout.addWidget(grp_border)

        # 3. Source Info
        grp_info = _FineTuneGroup("Source Photos")
        lbl_info = QLabel(f"Combining {len(self.photo_paths)} overlapping photos into a wide-angle composite.", self)
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet("color: #555; font-size: 11px;")
        grp_info.layout().addWidget(lbl_info)
        sidebar_layout.addWidget(grp_info)

        sidebar_layout.addStretch(1)

        # Save and Cancel
        btn_box = QHBoxLayout()
        btn_box.setSpacing(6)

        self.btn_save = QPushButton("Save New...", self)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3c82e6, stop:1 #1c52ba);
                color: white; font-weight: bold; border: 1px solid #143e8c;
                border-radius: 4px; padding: 6px 14px; min-width: 90px;
            }
            QPushButton:hover { background: #4d8ff2; }
            QPushButton:disabled { background: #cccccc; color: #888888; border: 1px solid #aaaaaa; }
        """)
        self.btn_save.clicked.connect(self._save_panorama)
        btn_box.addWidget(self.btn_save, 1)

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setStyleSheet("padding: 6px 12px; border-radius: 4px; border: 1px solid #ccc; background: #f0f0f0;")
        self.btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(self.btn_cancel)

        sidebar_layout.addLayout(btn_box)

        sidebar_scroll.setWidget(sidebar_content)
        main_layout.addWidget(sidebar_scroll)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if hasattr(self, "preview_scene") and self.preview_scene.sceneRect().isValid() and self.preview_scene.sceneRect().width() > 0:
            self.preview_view.fitInView(self.preview_scene.sceneRect(), Qt.KeepAspectRatio)
            self._is_fit_mode = True
            current_scale = self.preview_view.transform().m11()
            self._on_zoom_changed(current_scale)
            if hasattr(self, "zoom_bar"):
                self.zoom_bar.set_is_fit(True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, "_is_fit_mode", True) and hasattr(self, "preview_view") and hasattr(self, "preview_scene"):
            if self.preview_scene.sceneRect().isValid() and self.preview_scene.sceneRect().width() > 0:
                self.preview_view.fitInView(self.preview_scene.sceneRect(), Qt.KeepAspectRatio)

    def _toggle_fit_zoom(self) -> None:
        if not hasattr(self, "preview_scene") or not self.preview_scene.sceneRect().isValid() or self.preview_scene.sceneRect().width() == 0:
            return
        if self._is_fit_mode:
            self._is_fit_mode = False
            self.preview_view.resetTransform()
            self._on_zoom_changed(1.0)
            self.zoom_bar.set_is_fit(False)
        else:
            self._is_fit_mode = True
            self.preview_view.fitInView(self.preview_scene.sceneRect(), Qt.KeepAspectRatio)
            current_scale = self.preview_view.transform().m11()
            self._on_zoom_changed(current_scale)
            self.zoom_bar.set_is_fit(True)

    def _apply_zoom(self, factor: float) -> None:
        self._is_fit_mode = False
        self.zoom_bar.set_is_fit(False)
        self.preview_view.scale(factor, factor)
        current_scale = self.preview_view.transform().m11()
        self._on_zoom_changed(current_scale)

    def _on_slider_zoom_moved(self, pct: int) -> None:
        self._is_fit_mode = False
        self.zoom_bar.set_is_fit(False)
        target_scale = max(0.1, min(5.0, pct / 100.0))
        current_scale = self.preview_view.transform().m11()
        if abs(current_scale) > 1e-4:
            factor = target_scale / current_scale
            self.preview_view.scale(factor, factor)
        self._on_zoom_changed(target_scale)

    def _on_zoom_changed(self, scale_val: float) -> None:
        if hasattr(self, "zoom_bar"):
            self.zoom_bar.update_zoom_display(scale_val)

    def _load_and_stitch(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if len(self.photo_paths) < 2:
            self.lbl_status.setText("Failed: Need at least 2 photos to stitch.")
            self.progress_bar.setVisible(False)
            self.btn_restitch.setEnabled(True)
            return

        self.progress_bar.setVisible(True)
        self.btn_save.setEnabled(False)
        self.btn_restitch.setEnabled(False)
        self.lbl_status.setText("Loading photos & initializing stitcher...")

        mode_text = self.cmb_projection.currentText()
        proj_mode = "planar" if "Planar" in mode_text else ("spherical" if "Spherical" in mode_text else "cylindrical")

        self._worker = PanoramaWorker(
            images_bgr=None,
            projection_mode=proj_mode,
            photo_paths=self.photo_paths,
        )
        self._worker.progress_updated.connect(self.lbl_status.setText)
        self._worker.raw_stitching_finished.connect(self._on_raw_stitch_completed)
        self._worker.start()

    def _stop_worker(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(5000)

    def reject(self) -> None:
        self._stop_worker()
        super().reject()

    def closeEvent(self, event) -> None:
        self._stop_worker()
        super().closeEvent(event)

    def _on_raw_stitch_completed(self, success: bool, raw_bgr: np.ndarray, msg: str) -> None:
        self.progress_bar.setVisible(False)
        self.btn_restitch.setEnabled(True)

        if not success or raw_bgr is None or raw_bgr.size == 0:
            self.lbl_status.setText(msg)
            QMessageBox.warning(self, "Panorama Stitching", f"Could not stitch panorama:\n{msg}")
            return

        self.raw_stitched_bgr = raw_bgr
        self._on_crop_or_inpaint_toggled()
        self.lbl_status.setText(msg)

    def _on_crop_or_inpaint_toggled(self) -> None:
        """Fast <50ms cache-based postprocessing without re-stitching."""
        if self.raw_stitched_bgr is None:
            return

        self.current_stitched_bgr = postprocess_panorama(
            self.raw_stitched_bgr,
            auto_crop=self.chk_autocrop.isChecked(),
            fill_minor_gaps=self.chk_inpaint.isChecked(),
        )
        self.btn_save.setEnabled(True)

        h, w = self.current_stitched_bgr.shape[:2]
        self.lbl_dims.setText(f"{w} × {h} px ({w * h // 1_000_000} MP)")

        # Render preview pixmap
        pm = _cv2_to_qpixmap(self.current_stitched_bgr)
        self.preview_pixmap_item.setPixmap(pm)
        self.preview_scene.setSceneRect(QRectF(pm.rect()))
        if self._is_fit_mode:
            self.preview_view.fitInView(self.preview_scene.sceneRect(), Qt.KeepAspectRatio)
            self._on_zoom_changed(self.preview_view.transform().m11())

    def _on_settings_changed(self) -> None:
        # Re-stitch when projection mode changes
        self._load_and_stitch()

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
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save panorama to disk:\n{e}")


# =============================================================================
# 2. Interactive Collage & 'Then & Now' Studio Dialog
# =============================================================================

class CollageDialog(QDialog):
    """
    Interactive QGraphicsView Collage & 'Then & Now' Studio:
    - 3 Layout Styles:
      1. Modern Grid / Masonry (2x1, 2x2, 3x2, Center Hero, Masonry)
      2. Classic WLPG Polaroid Table Scatter with drop shadows & handwritten caption
      3. 'Then & Now' Comparison Card with chronological sorting & badges
    - Adjustable borders, corner rounding, background themes.
    - Full print resolution export (4K UHD or 300 DPI A4).
    """

    def __init__(
        self,
        photos_data: list[dict[str, Any]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.photos_data = photos_data
        self.output_dir = Path(photos_data[0]["file_path"]).parent if photos_data and photos_data[0].get("file_path") else Path.home()
        self.saved_file_path: Path | None = None
        self.rendered_bgr: np.ndarray | None = None
        self._is_fit_mode: bool = True

        self.bg_color = QColor(245, 245, 248)
        self.bg_texture = "dark_wood"

        self.setWindowTitle("Create Collage — Essentials Photo Gallery")
        self.setWindowIcon(make_wlpg_icon("collage", 24))
        self.resize(1240, 800)
        self.setMinimumSize(960, 620)
        self._init_ui()
        self._render_preview()

    def _init_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ---------------- Left: Canvas & Status / Zoom Bar ----------------
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.preview_view = ZoomablePreviewCanvas(self)
        self.preview_scene = QGraphicsScene(self)
        self.preview_view.setScene(self.preview_scene)
        self.preview_view.setStyleSheet("background: #1C1C1E; border: 1px solid #555558; border-radius: 4px;")
        self.preview_pixmap_item = QGraphicsPixmapItem()
        self.preview_scene.addItem(self.preview_pixmap_item)
        self.preview_view.zoom_changed.connect(self._on_zoom_changed)
        left_layout.addWidget(self.preview_view, 1)

        # Status & Zoom bar
        status_bar = QHBoxLayout()
        status_bar.setContentsMargins(4, 2, 4, 2)
        status_bar.setSpacing(8)

        self.lbl_info = QLabel("Interactive Preview", self)
        self.lbl_info.setStyleSheet("color: #555; font-size: 11px;")
        status_bar.addWidget(self.lbl_info)

        status_bar.addStretch(1)

        self.zoom_bar = CreateStudioZoomBar(self)
        self.zoom_bar.fitRequested.connect(self._toggle_fit_zoom)
        self.zoom_bar.zoomInRequested.connect(lambda: self._apply_zoom(1.25))
        self.zoom_bar.zoomOutRequested.connect(lambda: self._apply_zoom(1.0 / 1.25))
        self.zoom_bar.zoomSliderMoved.connect(self._on_slider_zoom_moved)
        status_bar.addWidget(self.zoom_bar)

        left_layout.addLayout(status_bar)
        main_layout.addWidget(left_widget, 1)

        # ---------------- Right: Settings Sidebar ----------------
        sidebar_scroll = QScrollArea(self)
        sidebar_scroll.setFixedWidth(320)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        sidebar_content = QWidget()
        sidebar_content.setMaximumWidth(312)
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(2, 0, 4, 0)
        sidebar_layout.setSpacing(8)

        # 1. Layout Style
        grp_style = _FineTuneGroup("Layout Style")
        self.cmb_style = QComboBox(self)
        self.cmb_style.addItems([
            "Blended Collage",
            "Modern Grid / Masonry",
            "Polaroid Table Scatter",
            "Then & Now Comparison Card",
        ])
        self.cmb_style.currentIndexChanged.connect(self._on_style_changed)
        grp_style.layout().addWidget(self.cmb_style)

        self.btn_shuffle = QPushButton(" Shuffle Photo Order", self)
        self.btn_shuffle.setIcon(make_wlpg_icon("shuffle", 16))
        self.btn_shuffle.setCursor(Qt.PointingHandCursor)
        self.btn_shuffle.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                border: 1px solid #7A96DF;
                border-radius: 4px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F4F8FD, stop:1 #DDE9F8);
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 9pt;
                font-weight: 500;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #EBF3FC, stop:1 #C8DEF5);
                border: 1px solid #3C7FB1;
            }
            QPushButton:pressed {
                background: #B8D6F2;
                border: 1px solid #2C628B;
            }
        """)
        self.btn_shuffle.clicked.connect(self._shuffle_photos)
        grp_style.layout().addWidget(self.btn_shuffle)
        sidebar_layout.addWidget(grp_style)

        # 2. Seamless Blend Options (Default WLPG mode)
        self.grp_seamless = _FineTuneGroup("Blended Collage Settings")
        lbl_feather = QLabel("Edge Softness / Blending:")
        lbl_feather.setStyleSheet("font-size: 11px; color: #444; margin-top: 4px;")
        self.grp_seamless.layout().addWidget(lbl_feather)
        self.sld_feather = _AeroFineTuneSlider(default_value=22)
        self.sld_feather.setRange(10, 45)
        self.sld_feather.setValue(22)
        self.sld_feather.valueChanged.connect(self._render_preview)
        self.grp_seamless.layout().addWidget(self.sld_feather)
        sidebar_layout.addWidget(self.grp_seamless)

        # 3. Grid Options
        self.grp_grid = _FineTuneGroup("Grid Arrangement & Border")
        self.grp_grid.setVisible(False)
        self.cmb_grid_layout = QComboBox(self)
        self.cmb_grid_layout.addItems(["2x2", "2x1", "3x2", "Center Hero", "Masonry"])
        self.cmb_grid_layout.currentIndexChanged.connect(self._render_preview)
        self.grp_grid.layout().addWidget(self.cmb_grid_layout)

        lbl_b = QLabel("Border Thickness:")
        lbl_b.setStyleSheet("font-size: 11px; color: #444; margin-top: 4px;")
        self.grp_grid.layout().addWidget(lbl_b)
        self.sld_border = _AeroFineTuneSlider(default_value=12)
        self.sld_border.setRange(0, 40)
        self.sld_border.setValue(12)
        self.sld_border.valueChanged.connect(self._render_preview)
        self.grp_grid.layout().addWidget(self.sld_border)

        lbl_c = QLabel("Corner Rounding:")
        lbl_c.setStyleSheet("font-size: 11px; color: #444; margin-top: 4px;")
        self.grp_grid.layout().addWidget(lbl_c)
        self.sld_corner = _AeroFineTuneSlider(default_value=8)
        self.sld_corner.setRange(0, 24)
        self.sld_corner.setValue(8)
        self.sld_corner.valueChanged.connect(self._render_preview)
        self.grp_grid.layout().addWidget(self.sld_corner)

        self.btn_bg_color = QPushButton(" Pick Background Colour", self)
        self.btn_bg_color.setIcon(make_wlpg_icon("pipette", 16))
        self.btn_bg_color.setStyleSheet("padding: 4px 8px; border: 1px solid #c0c0c0; border-radius: 3px; background: #ffffff;")
        self.btn_bg_color.clicked.connect(self._pick_bg_color)
        self.grp_grid.layout().addWidget(self.btn_bg_color)
        sidebar_layout.addWidget(self.grp_grid)

        # 4. Polaroid Options
        self.grp_polaroid = _FineTuneGroup("Polaroid Settings")
        self.grp_polaroid.setVisible(False)
        self.cmb_texture = QComboBox(self)
        self.cmb_texture.addItems(["Mahogany Wood", "Clean Linen", "Modern Slate"])
        self.cmb_texture.currentIndexChanged.connect(self._render_preview)
        self.grp_polaroid.layout().addWidget(self.cmb_texture)

        self.chk_captions = QCheckBox("Print Date / Caption on Border")
        self.chk_captions.setChecked(True)
        self.chk_captions.toggled.connect(self._render_preview)
        self.grp_polaroid.layout().addWidget(self.chk_captions)
        sidebar_layout.addWidget(self.grp_polaroid)

        # 5. Canvas Orientation & Export Resolution
        grp_res = _FineTuneGroup("Orientation & Resolution")
        lbl_orient = QLabel("Orientation:")
        lbl_orient.setStyleSheet("font-size: 11px; color: #444;")
        grp_res.layout().addWidget(lbl_orient)
        self.cmb_orientation = QComboBox(self)
        self.cmb_orientation.addItems(["Landscape (Horizontal)", "Portrait (Vertical)"])
        self.cmb_orientation.currentIndexChanged.connect(self._render_preview)
        grp_res.layout().addWidget(self.cmb_orientation)

        lbl_res = QLabel("Resolution:")
        lbl_res.setStyleSheet("font-size: 11px; color: #444; margin-top: 4px;")
        grp_res.layout().addWidget(lbl_res)
        self.cmb_resolution = QComboBox(self)
        self.cmb_resolution.addItems([
            "4K UHD (3840 × 2160)",
            "Full HD (1920 × 1080)",
            "A4 Print 300 DPI (3508 × 2480)",
        ])
        self.cmb_resolution.currentIndexChanged.connect(self._render_preview)
        grp_res.layout().addWidget(self.cmb_resolution)
        sidebar_layout.addWidget(grp_res)

        sidebar_layout.addStretch(1)

        # Save and Cancel
        btn_box = QHBoxLayout()
        btn_box.setSpacing(6)

        self.btn_save = QPushButton("Save New...", self)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3c82e6, stop:1 #1c52ba);
                color: white; font-weight: bold; border: 1px solid #143e8c;
                border-radius: 4px; padding: 6px 14px; min-width: 90px;
            }
            QPushButton:hover { background: #4d8ff2; }
        """)
        self.btn_save.clicked.connect(self._save_collage)
        btn_box.addWidget(self.btn_save, 1)

        btn_cancel = QPushButton("Cancel", self)
        btn_cancel.setStyleSheet("padding: 6px 12px; border-radius: 4px; border: 1px solid #ccc; background: #f0f0f0;")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        sidebar_layout.addLayout(btn_box)

        sidebar_scroll.setWidget(sidebar_content)
        main_layout.addWidget(sidebar_scroll)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if hasattr(self, "preview_scene") and self.preview_scene.sceneRect().isValid() and self.preview_scene.sceneRect().width() > 0:
            self.preview_view.fitInView(self.preview_scene.sceneRect(), Qt.KeepAspectRatio)
            self._is_fit_mode = True
            current_scale = self.preview_view.transform().m11()
            self._on_zoom_changed(current_scale)
            if hasattr(self, "zoom_bar"):
                self.zoom_bar.set_is_fit(True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, "_is_fit_mode", True) and hasattr(self, "preview_view") and hasattr(self, "preview_scene"):
            if self.preview_scene.sceneRect().isValid() and self.preview_scene.sceneRect().width() > 0:
                self.preview_view.fitInView(self.preview_scene.sceneRect(), Qt.KeepAspectRatio)

    def _toggle_fit_zoom(self) -> None:
        if not hasattr(self, "preview_scene") or not self.preview_scene.sceneRect().isValid() or self.preview_scene.sceneRect().width() == 0:
            return
        if self._is_fit_mode:
            self._is_fit_mode = False
            self.preview_view.resetTransform()
            self._on_zoom_changed(1.0)
            self.zoom_bar.set_is_fit(False)
        else:
            self._is_fit_mode = True
            self.preview_view.fitInView(self.preview_scene.sceneRect(), Qt.KeepAspectRatio)
            current_scale = self.preview_view.transform().m11()
            self._on_zoom_changed(current_scale)
            self.zoom_bar.set_is_fit(True)

    def _apply_zoom(self, factor: float) -> None:
        self._is_fit_mode = False
        self.zoom_bar.set_is_fit(False)
        self.preview_view.scale(factor, factor)
        current_scale = self.preview_view.transform().m11()
        self._on_zoom_changed(current_scale)

    def _on_slider_zoom_moved(self, pct: int) -> None:
        self._is_fit_mode = False
        self.zoom_bar.set_is_fit(False)
        target_scale = max(0.1, min(5.0, pct / 100.0))
        current_scale = self.preview_view.transform().m11()
        if abs(current_scale) > 1e-4:
            factor = target_scale / current_scale
            self.preview_view.scale(factor, factor)
        self._on_zoom_changed(target_scale)

    def _on_zoom_changed(self, scale_val: float) -> None:
        if hasattr(self, "zoom_bar"):
            self.zoom_bar.update_zoom_display(scale_val)

    def _on_style_changed(self, idx: int) -> None:
        # idx 0: Blended Collage
        # idx 1: Modern Grid / Masonry
        # idx 2: Polaroid Table Scatter
        # idx 3: Then & Now Comparison Card
        self.grp_seamless.setVisible(idx == 0)
        self.grp_grid.setVisible(idx == 1)
        self.grp_polaroid.setVisible(idx == 2)
        # Shuffle button doesn't make sense for chronological Then & Now
        self.btn_shuffle.setVisible(idx != 3)
        self._render_preview()

    def _pick_bg_color(self) -> None:
        col = QColorDialog.getColor(self.bg_color, self, "Select Background Colour")
        if col.isValid():
            self.bg_color = col
            self._render_preview()

    def _get_target_resolution(self) -> tuple[int, int]:
        txt = self.cmb_resolution.currentText()
        if "4K" in txt:
            w, h = 3840, 2160
        elif "A4" in txt:
            w, h = 3508, 2480
        else:
            w, h = 1920, 1080

        # Check portrait orientation toggle
        if hasattr(self, "cmb_orientation") and "Portrait" in self.cmb_orientation.currentText():
            if w > h:
                w, h = h, w
        return w, h

    def _shuffle_photos(self) -> None:
        import random
        random.shuffle(self.photos_data)
        self._user_shuffled = True
        self._render_preview()

    def _render_canvas(self, target_size: tuple[int, int]) -> np.ndarray:
        style_idx = self.cmb_style.currentIndex()

        # Build list of numpy images respecting EXIF
        images_bgr = []
        for d in self.photos_data:
            img = d.get("image_bgr")
            if img is None and d.get("file_path"):
                img = load_bgr_exif_safe(d["file_path"])
                d["image_bgr"] = img
            if img is not None:
                images_bgr.append(img)

        if not images_bgr:
            return np.full((target_size[1], target_size[0], 3), (245, 245, 248), dtype=np.uint8)

        if style_idx == 0:
            feather_ratio = self.sld_feather.value() / 100.0 if hasattr(self, "sld_feather") else 0.22
            faces_per_image = [d.get("faces") for d in self.photos_data]
            prioritize = not getattr(self, "_user_shuffled", False)
            return create_blended_seamless_collage(
                images_bgr=images_bgr,
                target_size=target_size,
                blend_feather=feather_ratio,
                faces_per_image=faces_per_image,
                prioritize_faces=prioritize,
            )
        elif style_idx == 1:
            layout_name = self.cmb_grid_layout.currentText().lower().replace(" ", "_")
            return create_grid_collage(
                images_bgr=images_bgr,
                layout=layout_name,
                border_thickness=self.sld_border.value(),
                corner_radius=self.sld_corner.value(),
                bg_color=_qcolor_to_bgr(self.bg_color),
                target_size=target_size,
            )
        elif style_idx == 2:
            tex_name = "dark_wood" if "Wood" in self.cmb_texture.currentText() else ("linen" if "Linen" in self.cmb_texture.currentText() else "slate")
            polaroid_data = []
            for i, d in enumerate(self.photos_data):
                if i < len(images_bgr):
                    raw_dt = d.get("date_taken")
                    dt_str = ""
                    if isinstance(raw_dt, (int, float)) and raw_dt > 100000000:
                        try:
                            import datetime
                            dt_str = datetime.datetime.fromtimestamp(raw_dt).strftime("%d %b %Y")
                        except Exception:
                            dt_str = str(raw_dt)
                    elif raw_dt:
                        dt_str = str(raw_dt)

                    caption = d.get("caption") or dt_str or d.get("filename") or ""
                    polaroid_data.append({
                        "image_bgr": images_bgr[i],
                        "caption": str(caption),
                        "date_str": dt_str,
                    })
            return create_polaroid_scatter_collage(
                images_data=polaroid_data,
                target_size=target_size,
                bg_style=tex_name,
                print_captions=self.chk_captions.isChecked(),
            )
        else:
            then_now_data = []
            for i, d in enumerate(self.photos_data):
                if i < len(images_bgr):
                    then_now_data.append({
                        "image_bgr": images_bgr[i],
                        "date_taken": d.get("date_taken") or d.get("date_str") or "",
                        "caption": d.get("caption") or "",
                        "mtime": d.get("mtime") or 0,
                    })
            return create_then_and_now_card(
                images_data=then_now_data,
                target_size=target_size,
            )

    def _render_preview(self) -> None:
        full_w, full_h = self._get_target_resolution()
        # Scale proxy to max 1280px for instant 60 FPS slider response
        scale = min(1.0, 1280.0 / max(full_w, full_h))
        pw = max(16, int(round(full_w * scale)))
        ph = max(16, int(round(full_h * scale)))

        rendered = self._render_canvas(target_size=(pw, ph))
        self.rendered_bgr = rendered

        # Display in QGraphicsView
        pm = _cv2_to_qpixmap(rendered)
        self.preview_pixmap_item.setPixmap(pm)
        self.preview_scene.setSceneRect(QRectF(pm.rect()))
        if self._is_fit_mode:
            self.preview_view.fitInView(self.preview_scene.sceneRect(), Qt.KeepAspectRatio)
            self._on_zoom_changed(self.preview_view.transform().m11())

        self.lbl_info.setText(f"{pw} × {ph} px (Full: {full_w} × {full_h}) • {len(self.photos_data)} photos")

    def _save_collage(self) -> None:
        first_path = self.photos_data[0].get("file_path") if self.photos_data else None
        if first_path is not None and not isinstance(first_path, Path):
            first_path = Path(first_path)
        out_path = prompt_save_new_file(self, first_path, "collage", ".jpg")
        if not out_path:
            return

        self.lbl_info.setText("Rendering full-resolution collage for export...")
        QApplication.processEvents()

        full_size = self._get_target_resolution()
        full_res_bgr = self._render_canvas(target_size=full_size)

        try:
            cv2.imwrite(str(out_path), full_res_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            self.saved_file_path = out_path
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Save Collage Failed", f"Could not save collage file:\n{e}")


# =============================================================================
# 3. Photo Fuse Dialog (Best-Face Group Shot Combiner)
# =============================================================================

class _InteractiveFaceItem(QGraphicsRectItem):
    """Interactive face box with robust click handling, high-contrast cosmetic rendering, and hover feedback."""
    def __init__(
        self,
        rect: QRectF,
        index: int,
        on_clicked: Any,
        is_selected: bool = False,
        is_swapped: bool = False,
        on_dismiss: Any = None,
    ) -> None:
        super().__init__(rect)
        self.index = index
        self.on_clicked = on_clicked
        self.is_selected = is_selected
        self.is_swapped = is_swapped
        self.on_dismiss = on_dismiss
        self._is_hovered = False
        self.setCursor(Qt.PointingHandCursor)
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event) -> None:
        self._is_hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._is_hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.on_clicked(self.index)
            event.accept()
            return
        if event.button() == Qt.RightButton and self.on_dismiss is not None:
            self.on_dismiss(self.index)
            event.accept()
            return
        super().mousePressEvent(event)

    def paint(self, painter: QPainter, option: Any, widget: Any = None) -> None:
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        r = self.rect()

        if self.is_selected:
            fill_col = QColor(0, 120, 215, 55)
            border_col = QColor("#0078D7")
            pen_w = 2.8
        elif self.is_swapped:
            fill_col = QColor(16, 163, 74, 30)
            border_col = QColor("#16A34A")
            pen_w = 2.4
        elif self._is_hovered:
            fill_col = QColor(0, 162, 237, 45)
            border_col = QColor("#00A2ED")
            pen_w = 2.4
        else:
            fill_col = QColor(0, 120, 215, 20)
            border_col = QColor("#0086E6")
            pen_w = 2.2

        # 1. Subtle semi-transparent Aero blue fill
        painter.fillRect(r, fill_col)

        # 2. Outer high-contrast dark drop-shadow stroke (ensures visibility on white/bright backgrounds)
        outer_pen = QPen(QColor(0, 0, 0, 150), pen_w + 1.4)
        outer_pen.setCosmetic(True)
        painter.setPen(outer_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(r)

        # 3. Main vibrant frame
        main_pen = QPen(border_col, pen_w)
        main_pen.setCosmetic(True)
        painter.setPen(main_pen)
        painter.drawRect(r)

        # 4. Corner bracket accents for authentic face-detection framing
        accent_col = QColor("#FFFFFF") if self.is_selected else border_col
        accent_pen = QPen(accent_col, pen_w + 0.8)
        accent_pen.setCosmetic(True)
        painter.setPen(accent_pen)
        cw = min(r.width() * 0.2, 14.0)
        ch = min(r.height() * 0.2, 14.0)
        # Top-left corner
        painter.drawLine(QPointF(r.left(), r.top()), QPointF(r.left() + cw, r.top()))
        painter.drawLine(QPointF(r.left(), r.top()), QPointF(r.left(), r.top() + ch))
        # Top-right corner
        painter.drawLine(QPointF(r.right(), r.top()), QPointF(r.right() - cw, r.top()))
        painter.drawLine(QPointF(r.right(), r.top()), QPointF(r.right(), r.top() + ch))
        # Bottom-left corner
        painter.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.left() + cw, r.bottom()))
        painter.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.left(), r.bottom() - ch))
        # Bottom-right corner
        painter.drawLine(QPointF(r.right(), r.bottom()), QPointF(r.right() - cw, r.bottom()))
        painter.drawLine(QPointF(r.right(), r.bottom()), QPointF(r.right(), r.bottom() - ch))

        painter.restore()


class _PhotoFuseWorker(QThread):
    """Background worker for loading, aligning burst shots, and detecting faces."""
    progress_updated = Signal(str)
    finished_ok = Signal(int, object, object, object, object)  # base_idx, frames_bgr, aligned_frames_bgr, align_ok, detections
    finished_err = Signal(str)

    def __init__(
        self,
        photo_paths: list[Path],
        frames_bgr: list[np.ndarray],
        base_idx: int = 0,
    ) -> None:
        super().__init__()
        self.photo_paths = photo_paths
        self.frames_bgr = frames_bgr
        self.base_idx = base_idx

    def run(self) -> None:
        frames = list(self.frames_bgr)
        if not frames:
            for idx, p in enumerate(self.photo_paths, start=1):
                if self.isInterruptionRequested():
                    return
                self.progress_updated.emit(f"Loading photo {idx} of {len(self.photo_paths)}...")
                img = load_bgr_exif_safe(p)
                if img is not None:
                    frames.append(img)

        if len(frames) < 2:
            self.finished_err.emit("Need at least 2 valid photos to fuse.")
            return

        base_idx = max(0, min(self.base_idx, len(frames) - 1))
        base = frames[base_idx]
        aligned_frames: list[np.ndarray] = []
        align_ok: list[bool] = []

        for i, target in enumerate(frames):
            if self.isInterruptionRequested():
                return
            if i == base_idx:
                aligned_frames.append(base.copy())
                align_ok.append(True)
            else:
                self.progress_updated.emit(f"Aligning Shot {i + 1} to Base Shot {base_idx + 1}...")
                ok, aligned = align_burst_frame(base, target)
                aligned_frames.append(aligned)
                align_ok.append(bool(ok))

        if self.isInterruptionRequested():
            return
        self.progress_updated.emit("Detecting faces...")
        detections = detect_faces_with_landmarks(base)
        self.finished_ok.emit(base_idx, frames, aligned_frames, align_ok, detections)


class PhotoFuseDialog(QDialog):
    """
    Photo Fuse — Best-Face Group Shot Combiner:
    - Visual Base Shot selector with cached per-base alignment & detections
    - Clear warning when selected photos cannot be aligned
    - InsightFace (or strict Haar) face + landmark detection
    - Per-face local landmark/ECC alignment + multi-band Laplacian blending
    - Show/Hide face boxes toggle and clear Before/After comparison
    """

    def __init__(
        self,
        photo_paths: list[Path],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.photo_paths = photo_paths
        self.output_dir = photo_paths[0].parent if photo_paths else Path.home()
        self.saved_file_path: Path | None = None

        self.frames_bgr: list[np.ndarray] = []
        self.aligned_frames_bgr: list[np.ndarray] = []
        self.alignment_ok: list[bool] = []
        self.base_fused_bgr: np.ndarray | None = None
        self.base_shot_idx: int = 0
        self.face_boxes: list[tuple[int, int, int, int]] = []
        self.face_kps: list[np.ndarray | None] = []
        self.face_applied_shot: dict[int, int] = {}
        self._src_kps_cache: dict[tuple[int, int], np.ndarray | None] = {}
        self._base_cache: dict[int, tuple[list[np.ndarray], list[bool], list[dict[str, Any]]]] = {}
        self._base_buttons: list[QToolButton] = []
        self.selected_face_idx: int = -1
        self._showing_original: bool = False
        self._show_face_boxes: bool = True
        self._is_fit_mode: bool = True
        self._worker: _PhotoFuseWorker | None = None

        self.setWindowTitle("Photo Fuse — Essentials Photo Gallery")
        self.setWindowIcon(make_wlpg_icon("photo_fuse", 24))
        self.resize(1240, 820)
        self.setMinimumSize(960, 640)
        self._init_ui()
        self._align_and_detect()

    def _init_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ---------------- Left: Banner + Canvas + Expression Strip + Zoom Bar ----------------
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Alignment warning banner (hidden unless alignment fails)
        self.lbl_align_warning = QLabel("", self)
        self.lbl_align_warning.setWordWrap(True)
        self.lbl_align_warning.setVisible(False)
        self.lbl_align_warning.setStyleSheet(
            "background: #FEF3C7; color: #92400E; border: 1px solid #F59E0B; "
            "border-radius: 4px; padding: 6px 10px; font-size: 11px; font-weight: bold;"
        )
        left_layout.addWidget(self.lbl_align_warning)

        # Before/After mode banner (shown only when viewing original base)
        self.lbl_preview_mode = QLabel("", self)
        self.lbl_preview_mode.setVisible(False)
        self.lbl_preview_mode.setAlignment(Qt.AlignCenter)
        self.lbl_preview_mode.setStyleSheet(
            "background: #1E3A8A; color: #FFFFFF; border-radius: 4px; padding: 4px 10px; "
            "font-size: 11px; font-weight: bold;"
        )
        left_layout.addWidget(self.lbl_preview_mode)

        # Canvas
        self.view = ZoomablePreviewCanvas(self)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.view.setStyleSheet("background: #1C1C1E; border: 1px solid #555558; border-radius: 4px;")
        self.bg_item = QGraphicsPixmapItem()
        self.scene.addItem(self.bg_item)
        self.view.zoom_changed.connect(self._on_zoom_changed)
        left_layout.addWidget(self.view, 1)

        # Bottom face replacement strip (tall enough for 64x64 thumbs + full label + scrollbar)
        self.strip_frame = QFrame()
        self.strip_frame.setFixedHeight(162)
        self.strip_frame.setStyleSheet("background: #F4F6FA; border: 1px solid #D2DFEE; border-radius: 4px;")
        strip_layout = QVBoxLayout(self.strip_frame)
        strip_layout.setContentsMargins(8, 6, 8, 6)
        strip_layout.setSpacing(4)

        self.lbl_strip_info = QLabel("Click any highlighted face box above to view expressions from all shots:")
        self.lbl_strip_info.setStyleSheet("font-weight: bold; font-size: 11px; color: #1e395b;")
        strip_layout.addWidget(self.lbl_strip_info)

        self.strip_scroll = QScrollArea()
        self.strip_scroll.setWidgetResizable(True)
        self.strip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.strip_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.strip_content = QWidget()
        self.strip_content_layout = QHBoxLayout(self.strip_content)
        self.strip_content_layout.setContentsMargins(4, 2, 4, 2)
        self.strip_content_layout.setSpacing(10)
        self.strip_content_layout.addStretch(1)
        self.strip_scroll.setWidget(self.strip_content)
        strip_layout.addWidget(self.strip_scroll, 1)

        left_layout.addWidget(self.strip_frame)

        # Status & Zoom bar
        status_bar = QHBoxLayout()
        status_bar.setContentsMargins(4, 2, 4, 2)
        status_bar.setSpacing(8)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedSize(110, 12)
        self.progress_bar.setStyleSheet(
            "QProgressBar { height: 12px; border: 1px solid #c0c0c0; border-radius: 2px; background: #e0e0e0; } "
            "QProgressBar::chunk { background: #2D72BC; }"
        )
        self.progress_bar.setVisible(False)
        status_bar.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Loading & aligning burst frames...", self)
        self.lbl_status.setStyleSheet("color: #444; font-size: 11px;")
        status_bar.addWidget(self.lbl_status)

        status_bar.addStretch(1)

        self.zoom_bar = CreateStudioZoomBar(self)
        self.zoom_bar.fitRequested.connect(self._toggle_fit_zoom)
        self.zoom_bar.zoomInRequested.connect(lambda: self._apply_zoom(1.25))
        self.zoom_bar.zoomOutRequested.connect(lambda: self._apply_zoom(1.0 / 1.25))
        self.zoom_bar.zoomSliderMoved.connect(self._on_slider_zoom_moved)
        status_bar.addWidget(self.zoom_bar)

        left_layout.addLayout(status_bar)
        main_layout.addWidget(left_widget, 1)

        # ---------------- Right: Sidebar ----------------
        sidebar_scroll = QScrollArea(self)
        sidebar_scroll.setFixedWidth(290)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(0, 0, 4, 0)
        sidebar_layout.setSpacing(8)

        # Base Photo Selector (visual thumbnail strip)
        grp_base = _FineTuneGroup("Base Photo")
        lbl_base_hint = QLabel("Pick the starting shot:", self)
        lbl_base_hint.setStyleSheet("color: #555; font-size: 11px;")
        grp_base.layout().addWidget(lbl_base_hint)

        self.base_strip_scroll = QScrollArea(self)
        self.base_strip_scroll.setFixedHeight(104)
        self.base_strip_scroll.setWidgetResizable(True)
        self.base_strip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.base_strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.base_strip_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.base_strip_widget = QWidget(self)
        self.base_strip_layout = QHBoxLayout(self.base_strip_widget)
        self.base_strip_layout.setContentsMargins(2, 2, 2, 2)
        self.base_strip_layout.setSpacing(6)
        self.base_strip_layout.addStretch(1)
        self.base_strip_scroll.setWidget(self.base_strip_widget)
        grp_base.layout().addWidget(self.base_strip_scroll)
        sidebar_layout.addWidget(grp_base)

        # View & Actions
        grp_actions = _FineTuneGroup("View & Actions")
        self.lbl_fuse_summary = QLabel(f"Processing {len(self.photo_paths)} shots...", self)
        self.lbl_fuse_summary.setStyleSheet("color: #555; font-size: 11px;")
        self.lbl_fuse_summary.setWordWrap(True)
        grp_actions.layout().addWidget(self.lbl_fuse_summary)

        self.chk_show_boxes = QCheckBox("Show face boxes", self)
        self.chk_show_boxes.setChecked(True)
        self.chk_show_boxes.toggled.connect(self._on_show_boxes_toggled)
        grp_actions.layout().addWidget(self.chk_show_boxes)

        self.btn_compare_orig = QPushButton(" View Original Base", self)
        self.btn_compare_orig.setCheckable(True)
        self.btn_compare_orig.setToolTip("Toggle between the fused composite and the unmodified base shot")
        self.btn_compare_orig.setStyleSheet("padding: 5px 10px; border: 1px solid #8ea4c8; border-radius: 3px; background: #f5f7fb;")
        self.btn_compare_orig.toggled.connect(self._toggle_compare_original)
        grp_actions.layout().addWidget(self.btn_compare_orig)

        self.btn_reset = QPushButton(" Reset to Base Shot 1", self)
        self.btn_reset.setIcon(make_wlpg_icon("undo", 16))
        self.btn_reset.setStyleSheet("padding: 5px 10px; border: 1px solid #7a96df; border-radius: 3px; background: #eef3fa;")
        self.btn_reset.clicked.connect(self._reset_to_base)
        grp_actions.layout().addWidget(self.btn_reset)
        sidebar_layout.addWidget(grp_actions)

        # Instructions
        grp_how = _FineTuneGroup("Tips")
        lbl_how = QLabel(
            "• Click a face box on the photo, then pick an expression below.\n"
            "• Right-click a box to dismiss a false detection.",
            self,
        )
        lbl_how.setWordWrap(True)
        lbl_how.setStyleSheet("color: #444; font-size: 11px; line-height: 1.4;")
        grp_how.layout().addWidget(lbl_how)
        sidebar_layout.addWidget(grp_how)

        sidebar_layout.addStretch(1)

        # Save and Cancel
        btn_box = QHBoxLayout()
        btn_box.setSpacing(6)

        self.btn_save = QPushButton("Save New...", self)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3c82e6, stop:1 #1c52ba);
                color: white; font-weight: bold; border: 1px solid #143e8c;
                border-radius: 4px; padding: 6px 14px; min-width: 90px;
            }
            QPushButton:hover { background: #4d8ff2; }
            QPushButton:disabled { background: #ccc; color: #888; }
        """)
        self.btn_save.clicked.connect(self._save_fused_photo)
        btn_box.addWidget(self.btn_save, 1)

        btn_cancel = QPushButton("Cancel", self)
        btn_cancel.setStyleSheet("padding: 6px 12px; border-radius: 4px; border: 1px solid #ccc; background: #f0f0f0;")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        sidebar_layout.addLayout(btn_box)

        sidebar_scroll.setWidget(sidebar_content)
        main_layout.addWidget(sidebar_scroll)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if hasattr(self, "scene") and self.scene.sceneRect().isValid() and self.scene.sceneRect().width() > 0:
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self._is_fit_mode = True
            current_scale = self.view.transform().m11()
            self._on_zoom_changed(current_scale)
            if hasattr(self, "zoom_bar"):
                self.zoom_bar.set_is_fit(True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, "_is_fit_mode", True) and hasattr(self, "view") and hasattr(self, "scene"):
            if self.scene.sceneRect().isValid() and self.scene.sceneRect().width() > 0:
                self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
                self._on_zoom_changed(self.view.transform().m11())
                if hasattr(self, "zoom_bar"):
                    self.zoom_bar.set_is_fit(True)

    def _toggle_fit_zoom(self) -> None:
        if not hasattr(self, "scene") or not self.scene.sceneRect().isValid() or self.scene.sceneRect().width() == 0:
            return
        if self._is_fit_mode:
            self._is_fit_mode = False
            self.view.resetTransform()
            self._on_zoom_changed(1.0)
            self.zoom_bar.set_is_fit(False)
        else:
            self._is_fit_mode = True
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            current_scale = self.view.transform().m11()
            self._on_zoom_changed(current_scale)
            self.zoom_bar.set_is_fit(True)

    def _apply_zoom(self, factor: float) -> None:
        self._is_fit_mode = False
        self.zoom_bar.set_is_fit(False)
        self.view.scale(factor, factor)
        current_scale = self.view.transform().m11()
        self._on_zoom_changed(current_scale)

    def _on_slider_zoom_moved(self, pct: int) -> None:
        self._is_fit_mode = False
        self.zoom_bar.set_is_fit(False)
        target_scale = max(0.1, min(5.0, pct / 100.0))
        current_scale = self.view.transform().m11()
        if abs(current_scale) > 1e-4:
            factor = target_scale / current_scale
            self.view.scale(factor, factor)
        self._on_zoom_changed(target_scale)

    def _on_zoom_changed(self, scale_val: float) -> None:
        if hasattr(self, "zoom_bar"):
            self.zoom_bar.update_zoom_display(scale_val)

    def _align_and_detect(self) -> None:
        """Starts background loading, alignment, and face detection for Shot 1."""
        if len(self.photo_paths) < 2:
            self.lbl_status.setText("Need at least 2 photos to fuse.")
            return
        self._rebuild_from_base(0)

    def _populate_base_strip(self) -> None:
        """Populates visual thumbnail buttons for choosing the Base Shot."""
        while self.base_strip_layout.count() > 1:
            child = self.base_strip_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self._base_buttons.clear()

        for idx, frame in enumerate(self.frames_bgr):
            h, w = frame.shape[:2]
            scale = 52.0 / max(1, max(h, w))
            tw = max(16, int(round(w * scale)))
            th = max(16, int(round(h * scale)))
            thumb_bgr = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
            pm = _cv2_to_qpixmap(thumb_bgr)

            btn = QToolButton(self)
            btn.setIcon(QIcon(pm))
            btn.setIconSize(QSize(54, 54))
            btn.setText(f"Shot {idx + 1}")
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn.setFixedSize(70, 82)
            name = self.photo_paths[idx].name if idx < len(self.photo_paths) else f"Shot {idx + 1}"
            btn.setToolTip(f"Use Shot {idx + 1} ({name}) as Base Photo")
            btn.clicked.connect(lambda _, i=idx: self._on_base_shot_selected(i))
            self._base_buttons.append(btn)
            self.base_strip_layout.insertWidget(self.base_strip_layout.count() - 1, btn)

        self._update_base_strip_styles()

    def _update_base_strip_styles(self) -> None:
        for idx, btn in enumerate(self._base_buttons):
            is_cur = idx == self.base_shot_idx
            if is_cur:
                btn.setStyleSheet(
                    "QToolButton { border: 2px solid #0078d4; background: #e8f2fc; "
                    "border-radius: 4px; padding: 3px; font-size: 10px; font-weight: bold; color: #1e395b; }"
                )
            else:
                btn.setStyleSheet(
                    "QToolButton { border: 1px solid #b0c0d8; background: white; "
                    "border-radius: 4px; padding: 3px; font-size: 10px; color: #1f2937; }"
                    "QToolButton:hover { border: 2px solid #0078d4; background: #eef6fc; }"
                )

    def _on_base_shot_selected(self, base_idx: int) -> None:
        if base_idx < 0 or base_idx >= len(self.frames_bgr):
            return
        if base_idx == self.base_shot_idx and self.aligned_frames_bgr:
            return
        self._rebuild_from_base(base_idx)

    def _rebuild_from_base(self, base_idx: int) -> None:
        """Restores cached alignment/detections for base_idx if available, else runs worker."""
        if self._worker is not None and self._worker.isRunning():
            return

        self.base_shot_idx = base_idx
        self.face_applied_shot.clear()
        self._src_kps_cache.clear()
        self._showing_original = False
        self.lbl_preview_mode.setVisible(False)
        if hasattr(self, "btn_compare_orig"):
            self.btn_compare_orig.blockSignals(True)
            self.btn_compare_orig.setChecked(False)
            self.btn_compare_orig.setText(" View Original Base")
            self.btn_compare_orig.blockSignals(False)

        self._update_base_strip_styles()

        # Instant cache hit when returning to a previously computed base shot
        if base_idx in self._base_cache and self.frames_bgr:
            aligned_frames, align_ok, detections = self._base_cache[base_idx]
            self._apply_base_results(base_idx, self.frames_bgr, aligned_frames, align_ok, detections)
            return

        self.aligned_frames_bgr = []
        self.alignment_ok = []
        self.progress_bar.setVisible(True)
        self.btn_save.setEnabled(False)
        for b in self._base_buttons:
            b.setEnabled(False)
        self.btn_reset.setEnabled(False)
        self.lbl_status.setText("Loading & aligning burst photos...")

        self._worker = _PhotoFuseWorker(
            photo_paths=self.photo_paths,
            frames_bgr=self.frames_bgr,
            base_idx=base_idx,
        )
        self._worker.progress_updated.connect(self.lbl_status.setText)
        self._worker.finished_ok.connect(self._on_fuse_worker_finished)
        self._worker.finished_err.connect(self._on_fuse_worker_error)
        self._worker.start()

    def _on_fuse_worker_error(self, msg: str) -> None:
        self.progress_bar.setVisible(False)
        for b in self._base_buttons:
            b.setEnabled(True)
        self.lbl_status.setText(msg)

    def _on_fuse_worker_finished(
        self,
        base_idx: int,
        frames: list[np.ndarray],
        aligned_frames: list[np.ndarray],
        align_ok: list[bool],
        detections: list[dict[str, Any]],
    ) -> None:
        self.progress_bar.setVisible(False)
        for b in self._base_buttons:
            b.setEnabled(True)
        self.btn_reset.setEnabled(True)

        self._base_cache[base_idx] = (aligned_frames, align_ok, detections)
        self._apply_base_results(base_idx, frames, aligned_frames, align_ok, detections)

    def _apply_base_results(
        self,
        base_idx: int,
        frames: list[np.ndarray],
        aligned_frames: list[np.ndarray],
        align_ok: list[bool],
        detections: list[dict[str, Any]],
    ) -> None:
        first_populate = not self.frames_bgr
        self.frames_bgr = frames
        if first_populate or len(self._base_buttons) != len(frames):
            self._populate_base_strip()

        self.aligned_frames_bgr = aligned_frames
        self.alignment_ok = align_ok
        self.base_shot_idx = base_idx
        self._update_base_strip_styles()

        base = self.frames_bgr[base_idx]
        self.base_fused_bgr = base.copy()

        self.face_boxes = [d["rect"] for d in detections]
        self.face_kps = [d["kps"] for d in detections]

        if not self.face_boxes:
            h, w = base.shape[:2]
            cx, cy = w // 2, h // 2
            fw, fh = int(w * 0.2), int(h * 0.25)
            self.face_boxes = [(cx - fw // 2, cy - fh // 2, fw, fh)]
            self.face_kps = [None]

        # Check alignment quality across other shots
        failed_shots = [f"Shot {i + 1}" for i, ok in enumerate(align_ok) if i != base_idx and not ok]
        other_count = max(0, len(frames) - 1)
        if failed_shots:
            if len(failed_shots) == other_count:
                self.lbl_align_warning.setText(
                    "Could not align the selected photos. Photo Fuse requires burst or tripod shots of the same scene."
                )
            else:
                self.lbl_align_warning.setText(
                    f"Could not align {', '.join(failed_shots)} to Shot {base_idx + 1}. "
                    "Swapping faces from unaligned shots may not line up."
                )
            self.lbl_align_warning.setVisible(True)
        else:
            self.lbl_align_warning.setVisible(False)

        self.btn_reset.setText(f" Reset to Base Shot {base_idx + 1}")
        self.lbl_status.setText(
            f"Base: Shot {base_idx + 1} — {len(self.face_boxes)} face(s) detected"
        )
        self.lbl_fuse_summary.setText(
            f"Base: Shot {base_idx + 1} ({len(self.frames_bgr)} shots, {len(self.face_boxes)} face(s))."
        )
        self.btn_save.setEnabled(True)

        self.selected_face_idx = 0 if self.face_boxes else -1
        self._update_scene()
        if self.selected_face_idx >= 0:
            self._populate_face_strip(self.selected_face_idx)

        if self._is_fit_mode and self.scene.sceneRect().isValid():
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self._on_zoom_changed(self.view.transform().m11())

    def _stop_worker(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(5000)

    def reject(self) -> None:
        self._stop_worker()
        super().reject()

    def closeEvent(self, event) -> None:
        self._stop_worker()
        super().closeEvent(event)

    def _on_show_boxes_toggled(self, checked: bool) -> None:
        self._show_face_boxes = bool(checked)
        self._update_scene()

    def _update_scene(self) -> None:
        display_bgr = (
            self.frames_bgr[self.base_shot_idx]
            if self._showing_original and self.frames_bgr
            else self.base_fused_bgr
        )
        if display_bgr is None:
            return

        pm = _cv2_to_qpixmap(display_bgr)
        self.bg_item.setPixmap(pm)
        self.scene.setSceneRect(QRectF(pm.rect()))

        # Remove existing face items
        for item in list(self.scene.items()):
            if isinstance(item, _InteractiveFaceItem):
                self.scene.removeItem(item)

        # Draw interactive face rectangles only when enabled and not viewing original
        if self._show_face_boxes and not self._showing_original:
            for i, (fx, fy, fw, fh) in enumerate(self.face_boxes):
                is_sel = (i == self.selected_face_idx)
                is_swapped = i in self.face_applied_shot and self.face_applied_shot[i] != self.base_shot_idx
                rect_item = _InteractiveFaceItem(
                    QRectF(fx, fy, fw, fh),
                    i,
                    self._select_face,
                    is_selected=is_sel,
                    is_swapped=is_swapped,
                    on_dismiss=self._dismiss_face,
                )
                self.scene.addItem(rect_item)

    def _select_face(self, face_idx: int) -> None:
        if face_idx < 0 or face_idx >= len(self.face_boxes):
            return
        self.selected_face_idx = face_idx
        self._update_scene()
        self._populate_face_strip(face_idx)

    def _dismiss_face(self, face_idx: int) -> None:
        if len(self.face_boxes) <= 1 or face_idx < 0 or face_idx >= len(self.face_boxes):
            return
        self.face_boxes.pop(face_idx)
        if face_idx < len(self.face_kps):
            self.face_kps.pop(face_idx)

        new_applied: dict[int, int] = {}
        for k, v in self.face_applied_shot.items():
            if k < face_idx:
                new_applied[k] = v
            elif k > face_idx:
                new_applied[k - 1] = v
        self.face_applied_shot = new_applied
        self._src_kps_cache.clear()

        if self.selected_face_idx >= len(self.face_boxes):
            self.selected_face_idx = len(self.face_boxes) - 1
        elif self.selected_face_idx == face_idx:
            self.selected_face_idx = min(face_idx, len(self.face_boxes) - 1)

        self._recompute_composite()
        self._update_scene()
        if self.selected_face_idx >= 0:
            self._populate_face_strip(self.selected_face_idx)
        self.lbl_fuse_summary.setText(
            f"Base: Shot {self.base_shot_idx + 1} ({len(self.frames_bgr)} shots, {len(self.face_boxes)} face(s))."
        )
        self.lbl_status.setText("Removed face box.")

    def _toggle_compare_original(self, checked: bool) -> None:
        self._showing_original = bool(checked)
        self.btn_compare_orig.setText(" Viewing Original Base" if checked else " View Original Base")
        if checked:
            self.lbl_preview_mode.setText(f"ORIGINAL — Shot {self.base_shot_idx + 1} (Base)")
            self.lbl_preview_mode.setVisible(True)
            self.lbl_status.setText(f"Viewing original Shot {self.base_shot_idx + 1}")
        else:
            self.lbl_preview_mode.setVisible(False)
            self.lbl_status.setText("Viewing fused composite")
        self._update_scene()

    def _get_aligned_face_kps(self, shot_idx: int, face_idx: int) -> np.ndarray | None:
        key = (shot_idx, face_idx)
        if key in self._src_kps_cache:
            return self._src_kps_cache[key]

        kps_found: np.ndarray | None = None
        if 0 <= shot_idx < len(self.aligned_frames_bgr) and 0 <= face_idx < len(self.face_boxes):
            aligned_img = self.aligned_frames_bgr[shot_idx]
            h, w = aligned_img.shape[:2]
            fx, fy, fw, fh = self.face_boxes[face_idx]
            pad_x = int(fw * 0.45)
            pad_y = int(fh * 0.45)
            rx1 = max(0, fx - pad_x)
            ry1 = max(0, fy - pad_y)
            rx2 = min(w, fx + fw + pad_x)
            ry2 = min(h, fy + fh + pad_y)
            roi = aligned_img[ry1:ry2, rx1:rx2]
            if roi.size > 0:
                dets = detect_faces_with_landmarks(roi)
                if dets:
                    # Pick detection closest to expected center
                    target_cx = (fx + fw * 0.5) - rx1
                    target_cy = (fy + fh * 0.5) - ry1
                    best = min(
                        dets,
                        key=lambda d: (
                            (d["rect"][0] + d["rect"][2] * 0.5 - target_cx) ** 2
                            + (d["rect"][1] + d["rect"][3] * 0.5 - target_cy) ** 2
                        ),
                    )
                    if best["kps"] is not None:
                        kps_found = best["kps"] + np.array([rx1, ry1], dtype=np.float32)

        self._src_kps_cache[key] = kps_found
        return kps_found

    def _populate_face_strip(self, face_idx: int) -> None:
        while self.strip_content_layout.count() > 1:
            child = self.strip_content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if face_idx < 0 or face_idx >= len(self.face_boxes):
            return

        fx, fy, fw, fh = self.face_boxes[face_idx]
        active_shot = self.face_applied_shot.get(face_idx, self.base_shot_idx)
        self.lbl_strip_info.setText(f"Expressions for Person {face_idx + 1} (Choose to replace in photo):")

        for shot_idx, aligned_img in enumerate(self.aligned_frames_bgr):
            h, w = aligned_img.shape[:2]
            margin_x = int(fw * 0.25)
            margin_y = int(fh * 0.25)
            x1 = max(0, fx - margin_x)
            y1 = max(0, fy - margin_y)
            x2 = min(w, fx + fw + margin_x)
            y2 = min(h, fy + fh + margin_y)

            crop_bgr = aligned_img[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                continue

            crop_thumb = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
            pm_thumb = _cv2_to_qpixmap(crop_thumb)

            is_base = shot_idx == self.base_shot_idx
            is_active = shot_idx == active_shot
            is_aligned = shot_idx < len(self.alignment_ok) and self.alignment_ok[shot_idx]
            label = f"Shot {shot_idx + 1}"
            if is_base:
                label += " (Base)"
            elif not is_aligned:
                label += " ⚠"
            elif is_active:
                label += " ✓"

            btn_card = QToolButton()
            btn_card.setIcon(QIcon(pm_thumb))
            btn_card.setIconSize(pm_thumb.size())
            btn_card.setText(label)
            btn_card.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn_card.setFixedSize(96, 98)
            if not is_aligned and not is_base:
                btn_card.setToolTip(f"Shot {shot_idx + 1} could not be aligned to Shot {self.base_shot_idx + 1}")
                border_style = "1px solid #f59e0b; background: #fffbeb;"
            elif is_active:
                border_style = "2px solid #0078d4; background: #e8f2fc; font-weight: bold;"
            else:
                border_style = "1px solid #b0c0d8; background: white;"
            btn_card.setStyleSheet(f"""
                QToolButton {{
                    border: {border_style}
                    border-radius: 4px; padding: 4px; font-size: 10px; color: #1f2937;
                }}
                QToolButton:hover {{
                    border: 2px solid #0078d4; background: #eef6fc;
                }}
            """)
            btn_card.clicked.connect(lambda _, s=shot_idx: self._apply_face_fuse(s))
            self.strip_content_layout.insertWidget(self.strip_content_layout.count() - 1, btn_card)

    def _recompute_composite(self) -> None:
        if not self.frames_bgr:
            return
        result = self.frames_bgr[self.base_shot_idx].copy()
        for f_idx, s_idx in sorted(self.face_applied_shot.items()):
            if s_idx == self.base_shot_idx or not (0 <= f_idx < len(self.face_boxes)):
                continue
            source_aligned = self.aligned_frames_bgr[s_idx]
            face_rect = self.face_boxes[f_idx]
            base_kps = self.face_kps[f_idx] if f_idx < len(self.face_kps) else None
            src_kps = self._get_aligned_face_kps(s_idx, f_idx)
            result = fuse_face(result, source_aligned, face_rect, base_kps=base_kps, src_kps=src_kps)
        self.base_fused_bgr = result

    def _apply_face_fuse(self, shot_idx: int) -> None:
        if self.selected_face_idx < 0 or self.base_fused_bgr is None:
            return

        if shot_idx == self.base_shot_idx:
            self.face_applied_shot.pop(self.selected_face_idx, None)
        else:
            self.face_applied_shot[self.selected_face_idx] = shot_idx

        self._showing_original = False
        self.lbl_preview_mode.setVisible(False)
        if hasattr(self, "btn_compare_orig") and self.btn_compare_orig.isChecked():
            self.btn_compare_orig.blockSignals(True)
            self.btn_compare_orig.setChecked(False)
            self.btn_compare_orig.setText(" View Original Base")
            self.btn_compare_orig.blockSignals(False)

        self._recompute_composite()
        self._update_scene()
        self._populate_face_strip(self.selected_face_idx)
        if shot_idx == self.base_shot_idx:
            self.lbl_status.setText(f"Person {self.selected_face_idx + 1}: using Base (Shot {shot_idx + 1})")
        else:
            self.lbl_status.setText(f"Person {self.selected_face_idx + 1}: using Shot {shot_idx + 1}")

    def _reset_to_base(self) -> None:
        if self.frames_bgr:
            self.face_applied_shot.clear()
            self._showing_original = False
            self.lbl_preview_mode.setVisible(False)
            if hasattr(self, "btn_compare_orig"):
                self.btn_compare_orig.blockSignals(True)
                self.btn_compare_orig.setChecked(False)
                self.btn_compare_orig.setText(" View Original Base")
                self.btn_compare_orig.blockSignals(False)
            self.base_fused_bgr = self.frames_bgr[self.base_shot_idx].copy()
            self._update_scene()
            if self.selected_face_idx >= 0:
                self._populate_face_strip(self.selected_face_idx)
            self.lbl_status.setText(f"Reset to Shot {self.base_shot_idx + 1}")

    def _save_fused_photo(self) -> None:
        if self.base_fused_bgr is None:
            return

        first_path = self.photo_paths[self.base_shot_idx] if self.base_shot_idx < len(self.photo_paths) else (self.photo_paths[0] if self.photo_paths else None)
        out_path = prompt_save_new_file(self, first_path, "photoFuse", ".jpg")
        if not out_path:
            return

        try:
            cv2.imwrite(str(out_path), self.base_fused_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            self.saved_file_path = out_path
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save photo:\n{e}")


# =============================================================================
# 4. Google Photos-Style "Colour Pop" (Subject Colour Splash) Dialog & Canvas
# =============================================================================

class ColourPopCanvasView(QGraphicsView):
    """
    High-performance interactive canvas for Colour Pop brush editing:
    - Real-time circular brush cursor following the pointer.
    - Continuous stroke interpolation between mouse moves so fast drags do not leave gaps.
    - Eyedropper mode for clicking and sampling colours directly from the image.
    - Middle-click or Space+Drag for panning, mouse wheel for zoom.
    """
    brush_applied = Signal(int, int)
    color_picked = Signal(int, int, str)
    stroke_started = Signal()
    stroke_finished = Signal()
    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.brush_radius: int = 28
        self._is_painting: bool = False
        self._space_pressed: bool = False
        self.eyedropper_mode: bool = False
        self._last_pt: QPointF | None = None

        self._cursor_item = QGraphicsEllipseItem()
        self._cursor_item.setPen(QPen(QColor(0, 160, 255, 230), 1.5, Qt.DashLine))
        self._cursor_item.setBrush(QBrush(QColor(0, 160, 255, 35)))
        self._cursor_item.setZValue(100)
        self._cursor_item.hide()

    def _get_eyedropper_cursor(self, mode: str = "normal") -> QCursor:
        pm = QPixmap(32, 32)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)

        # Pipette nozzle path (tip at 4, 27)
        nozzle = QPolygonF([
            QPointF(4, 27),
            QPointF(7, 24),
            QPointF(9, 26),
        ])
        p.setBrush(QBrush(QColor(220, 220, 220)))
        p.setPen(QPen(QColor(20, 20, 20), 1.2))
        p.drawPolygon(nozzle)

        # Pipette glass tube
        tube = QPolygonF([
            QPointF(7, 24),
            QPointF(17, 14),
            QPointF(19, 16),
            QPointF(9, 26),
        ])
        p.setBrush(QBrush(QColor(245, 245, 250, 230)))
        p.drawPolygon(tube)

        # Rubber bulb
        bulb = QPolygonF([
            QPointF(16, 15),
            QPointF(22, 9),
            QPointF(27, 4),
            QPointF(29, 6),
            QPointF(24, 11),
            QPointF(18, 17),
        ])
        p.setBrush(QBrush(QColor(50, 50, 50)))
        p.drawPolygon(bulb)

        # Tip dot
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 120, 215)))
        p.drawEllipse(QPointF(4.5, 26.5), 1.5, 1.5)

        # Mode badge
        if mode == "add":
            # Blue circle badge with white '+'
            p.setBrush(QBrush(QColor(0, 120, 215)))
            p.setPen(QPen(QColor(255, 255, 255), 1.2))
            p.drawEllipse(18, 18, 12, 12)
            # '+' symbol
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawLine(21, 24, 27, 24)
            p.drawLine(24, 21, 24, 27)
        elif mode == "remove":
            # Orange/Red circle badge with white '-'
            p.setBrush(QBrush(QColor(216, 59, 1)))
            p.setPen(QPen(QColor(255, 255, 255), 1.2))
            p.drawEllipse(18, 18, 12, 12)
            # '-' symbol
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawLine(21, 24, 27, 24)

        p.end()
        return QCursor(pm, 4, 27)

    def _update_eyedropper_cursor(self) -> None:
        if not self.eyedropper_mode:
            return
        mods = QApplication.keyboardModifiers()
        if bool(mods & (Qt.AltModifier | Qt.ControlModifier)):
            mode = "remove"
        elif bool(mods & Qt.ShiftModifier):
            mode = "add"
        else:
            mode = "normal"
        self.setCursor(self._get_eyedropper_cursor(mode))

    def set_brush_radius(self, r: int) -> None:
        self.brush_radius = max(4, r)
        center = self._cursor_item.rect().center()
        rad = self.brush_radius
        self._cursor_item.setRect(QRectF(center.x() - rad, center.y() - rad, 2 * rad, 2 * rad))

    def set_eyedropper_mode(self, enabled: bool) -> None:
        self.eyedropper_mode = bool(enabled)
        if self.eyedropper_mode:
            self._cursor_item.hide()
            self._update_eyedropper_cursor()
        else:
            self.setCursor(Qt.BlankCursor)
            self._cursor_item.show()

    def mousePressEvent(self, event) -> None:
        if self.eyedropper_mode and (event.button() == Qt.LeftButton or event.button() == Qt.RightButton):
            sp = self.mapToScene(event.pos())
            mods = event.modifiers()
            if event.button() == Qt.RightButton or bool(mods & (Qt.AltModifier | Qt.ControlModifier)):
                pick_mode = "remove"
            elif bool(mods & Qt.ShiftModifier):
                pick_mode = "add"
            else:
                pick_mode = "replace"
            self.color_picked.emit(int(round(sp.x())), int(round(sp.y())), pick_mode)
            self._update_eyedropper_cursor()
            event.accept()
            return

        if event.button() == Qt.LeftButton and not self._space_pressed:
            self._is_painting = True
            sp = self.mapToScene(event.pos())
            self._last_pt = sp
            self.stroke_started.emit()
            self.brush_applied.emit(int(sp.x()), int(sp.y()))
            event.accept()
            return
        elif event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self._space_pressed):
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            super().mousePressEvent(event)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.eyedropper_mode:
            self._cursor_item.hide()
            self._update_eyedropper_cursor()
            super().mouseMoveEvent(event)
            return

        sp = self.mapToScene(event.pos())
        rad = self.brush_radius
        self._cursor_item.setRect(QRectF(sp.x() - rad, sp.y() - rad, 2 * rad, 2 * rad))

        if self.scene() and self._cursor_item.scene() != self.scene():
            self.scene().addItem(self._cursor_item)
        if not self._cursor_item.isVisible() and not self._space_pressed:
            self._cursor_item.show()

        if self._is_painting and (event.buttons() & Qt.LeftButton) and not self._space_pressed:
            if self._last_pt is not None:
                p1 = self._last_pt
                p2 = sp
                dx = p2.x() - p1.x()
                dy = p2.y() - p1.y()
                dist = math.hypot(dx, dy)
                step = max(3.0, self.brush_radius * 0.35)
                steps = max(1, int(dist / step))
                for i in range(1, steps + 1):
                    ix = p1.x() + (dx * i) / steps
                    iy = p1.y() + (dy * i) / steps
                    self.brush_applied.emit(int(ix), int(iy))
            else:
                self.brush_applied.emit(int(sp.x()), int(sp.y()))
            self._last_pt = sp
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._is_painting:
            self._is_painting = False
            self._last_pt = None
            self.stroke_finished.emit()
            event.accept()
            return
        elif event.button() in (Qt.MiddleButton, Qt.LeftButton):
            if not self._space_pressed:
                self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if self.eyedropper_mode:
            self._update_eyedropper_cursor()
        if event.key() == Qt.Key_Space and not self._space_pressed:
            self._space_pressed = True
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._cursor_item.hide()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if self.eyedropper_mode:
            self._update_eyedropper_cursor()
        if event.key() == Qt.Key_Space:
            self._space_pressed = False
            self.setDragMode(QGraphicsView.NoDrag)
            if not self.eyedropper_mode:
                self._cursor_item.show()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        if self.eyedropper_mode:
            self._cursor_item.hide()
            self._update_eyedropper_cursor()
        else:
            self.setCursor(Qt.BlankCursor)
            if self.scene() and self._cursor_item.scene() != self.scene():
                self.scene().addItem(self._cursor_item)
            self._cursor_item.show()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self.unsetCursor()
        self._cursor_item.hide()

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)
        self.zoom_changed.emit(self.transform().m11())
        event.accept()


class ColourPopDialog(QDialog):
    """
    Google Photos-Style Colour Pop Studio:
    - Subject remains in full vibrant colour while background is converted to dramatic B&W.
    - Seeding via GrabCut segmentation / face detection.
    - Continuous 60 FPS proxy painting with bilateral edge-aware brush snapping.
    - Sidebar adjustment controls matching WLPG Fine-Tune layout.
    """

    def __init__(
        self,
        image_bgr: np.ndarray,
        file_path: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if len(image_bgr.shape) == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

        self.original_bgr = image_bgr.copy()
        self.file_path = file_path
        self.output_dir = file_path.parent if file_path else Path.home()
        self.saved_file_path: Path | None = None

        h, w = self.original_bgr.shape[:2]

        max_proxy_dim = 1200
        if max(h, w) > max_proxy_dim:
            self.proxy_scale = max_proxy_dim / float(max(h, w))
            pw = max(1, int(round(w * self.proxy_scale)))
            ph = max(1, int(round(h * self.proxy_scale)))
            self.proxy_bgr = cv2.resize(self.original_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        else:
            self.proxy_scale = 1.0
            self.proxy_bgr = self.original_bgr.copy()

        full_mask = estimate_subject_mask(self.original_bgr)
        self.proxy_mask = cv2.resize(full_mask, (self.proxy_bgr.shape[1], self.proxy_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

        self.undo_stack: list[np.ndarray] = [self.proxy_mask.copy()]
        self.redo_stack: list[np.ndarray] = []

        self.brush_mode = "keep_color"
        self.brush_radius = 28
        self.bg_contrast = 1.15
        self.bg_brightness = 0.0
        self.color_tolerance = 0.25
        self.show_mask_overlay = False
        self.last_picked_color: tuple[int, int, int] | None = None
        self._first_fit = True
        self._is_fit_mode = True
        self.picked_colors: list[tuple[int, int, int]] = []

        self.proxy_bw: np.ndarray | None = None
        self._precompute_proxy_bw()

        self.setWindowTitle("Colour Pop Studio")
        self.setWindowIcon(make_wlpg_icon("color_spectrum", 24))
        self.resize(1240, 800)
        self.setMinimumSize(960, 620)
        self._init_ui()
        self._refresh_canvas()

    def _init_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # -------------------------------------------------------------
        # 1. Left Canvas & Zoom Status Bar
        # -------------------------------------------------------------
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.view = ColourPopCanvasView(self)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.view.set_brush_radius(self.brush_radius)
        self.view.brush_applied.connect(self._on_brush_applied)
        self.view.color_picked.connect(self._on_color_picked)
        self.view.stroke_started.connect(self._on_stroke_started)
        self.view.zoom_changed.connect(self._on_zoom_changed)
        self.view.setStyleSheet("background: #1C1C1E; border: 1px solid #555558; border-radius: 4px;")

        self.bg_item = QGraphicsPixmapItem()
        self.scene.addItem(self.bg_item)
        left_layout.addWidget(self.view, 1)

        # Bottom Bar: Status on Left, Dims, and Zoom on Right
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(4, 2, 4, 2)
        bottom_bar.setSpacing(12)

        self.lbl_status = QLabel("Paint over photo to add or remove colour • Click to pick hue")
        self.lbl_status.setStyleSheet("color: #444; font-size: 11px;")
        bottom_bar.addWidget(self.lbl_status, 1)

        h, w = self.original_bgr.shape[:2]
        self.lbl_dims = QLabel(f"{w} × {h} px")
        self.lbl_dims.setStyleSheet("color: #666; font-size: 11px;")
        bottom_bar.addWidget(self.lbl_dims)

        self.zoom_bar = CreateStudioZoomBar(self)
        self.zoom_bar.fit_toggle_clicked.connect(self._toggle_fit_zoom)
        self.zoom_bar.zoom_in_clicked.connect(lambda: self._apply_zoom(1.25))
        self.zoom_bar.zoom_out_clicked.connect(lambda: self._apply_zoom(1.0 / 1.25))
        self.zoom_bar.zoom_slider_moved.connect(self._on_slider_zoom_moved)
        bottom_bar.addWidget(self.zoom_bar)

        left_layout.addLayout(bottom_bar)
        main_layout.addWidget(left_widget, 1)

        # -------------------------------------------------------------
        # 2. Right Sidebar (WLPG Fine-Tune Style Cards)
        # -------------------------------------------------------------
        sidebar_scroll = QScrollArea(self)
        sidebar_scroll.setFixedWidth(280)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setStyleSheet("QScrollArea { border: 1px solid #D2DFEE; border-radius: 4px; background: #F4F6FA; }")

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(8)

        # Mode Selection Card
        grp_mode = _FineTuneGroup("Colour Selection Mode", sidebar)
        self.btn_group_mode = QButtonGroup(self)
        self.btn_group_mode.setExclusive(True)

        mode_btn_style = """
            QPushButton {
                background: #FFFFFF; border: 1px solid #B0C0D8;
                border-radius: 3px; padding: 6px 12px; font-weight: 500; text-align: left;
            }
            QPushButton:hover { background: #EAF2FB; border-color: #70C0E7; }
            QPushButton:checked {
                background: #CCE5FF; border: 1px solid #0078D4;
                font-weight: bold; color: #004578;
            }
        """

        self.btn_keep = QPushButton("Keep Vibrant Colour (+)", grp_mode)
        self.btn_keep.setIcon(make_wlpg_icon("plus", 16))
        self.btn_keep.setIconSize(QSize(16, 16))
        self.btn_keep.setCheckable(True)
        self.btn_keep.setChecked(True)
        self.btn_keep.setStyleSheet(mode_btn_style)
        self.btn_keep.clicked.connect(lambda: self._set_brush_mode("keep_color"))
        self.btn_group_mode.addButton(self.btn_keep)
        grp_mode.vbox.addWidget(self.btn_keep)

        self.btn_bw = QPushButton("Make Black and White (-)", grp_mode)
        self.btn_bw.setIcon(make_wlpg_icon("minus", 16))
        self.btn_bw.setIconSize(QSize(16, 16))
        self.btn_bw.setCheckable(True)
        self.btn_bw.setStyleSheet(mode_btn_style)
        self.btn_bw.clicked.connect(lambda: self._set_brush_mode("make_bw"))
        self.btn_group_mode.addButton(self.btn_bw)
        grp_mode.vbox.addWidget(self.btn_bw)

        self.btn_pick_color = QPushButton("Pick Colour from Photo", grp_mode)
        self.btn_pick_color.setIcon(make_wlpg_icon("eyedropper", 16))
        self.btn_pick_color.setIconSize(QSize(16, 16))
        self.btn_pick_color.setCheckable(True)
        self.btn_pick_color.setStyleSheet(mode_btn_style)
        self.btn_pick_color.setToolTip("Click to isolate colour.\nShift+Click: Add colour (+)\nAlt/Ctrl+Click: Remove colour (-)")
        self.btn_pick_color.clicked.connect(self._toggle_eyedropper)
        self.btn_group_mode.addButton(self.btn_pick_color)
        grp_mode.vbox.addWidget(self.btn_pick_color)

        # Swatches row for multiple picked colors
        self.swatches_container = QWidget(grp_mode)
        self.swatches_layout = QHBoxLayout(self.swatches_container)
        self.swatches_layout.setContentsMargins(2, 4, 2, 2)
        self.swatches_layout.setSpacing(6)
        self.swatches_container.setVisible(False)
        grp_mode.vbox.addWidget(self.swatches_container)

        sidebar_layout.addWidget(grp_mode)

        # Brush & Mask Settings Card
        grp_brush = _FineTuneGroup("Brush & Mask Settings", sidebar)
        self.lbl_radius_val = QLabel(f"Brush Size: {self.brush_radius} px", grp_brush)
        self.lbl_radius_val.setStyleSheet("color: #333; font-size: 11px;")
        grp_brush.vbox.addWidget(self.lbl_radius_val)

        self.sld_radius = _AeroFineTuneSlider(8, 120, 28, grp_brush)
        self.sld_radius.setValue(self.brush_radius)
        self.sld_radius.valueChanged.connect(self._on_radius_changed)
        grp_brush.vbox.addWidget(self.sld_radius)

        self.lbl_tol_val = QLabel("Colour Tolerance: 25%", grp_brush)
        self.lbl_tol_val.setStyleSheet("color: #333; font-size: 11px;")
        grp_brush.vbox.addWidget(self.lbl_tol_val)

        self.sld_tol = _AeroFineTuneSlider(5, 80, 25, grp_brush)
        self.sld_tol.setValue(int(self.color_tolerance * 100))
        self._is_tol_sliding = False
        self.sld_tol.sliderPressed.connect(self._on_tolerance_pressed)
        self.sld_tol.sliderReleased.connect(self._on_tolerance_released)
        self.sld_tol.valueChanged.connect(self._on_tolerance_changed)
        grp_brush.vbox.addWidget(self.sld_tol)

        self.chk_overlay = QCheckBox("Show Mask Overlay (Red)", grp_brush)
        self.chk_overlay.setStyleSheet("font-size: 11px; color: #333; margin-top: 4px;")
        self.chk_overlay.toggled.connect(self._on_overlay_toggled)
        grp_brush.vbox.addWidget(self.chk_overlay)

        # Action Buttons Row
        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        self.btn_invert = QPushButton("Invert", grp_brush)
        self.btn_invert.setIcon(make_wlpg_icon("invert", 16))
        self.btn_invert.setToolTip("Swap colour subject and B&W background")
        self.btn_invert.setStyleSheet("background: #FFFFFF; border: 1px solid #B0C0D8; border-radius: 3px; padding: 4px;")
        self.btn_invert.clicked.connect(self._invert_mask)
        action_row.addWidget(self.btn_invert)

        self.btn_auto_detect = QPushButton("Auto-Detect", grp_brush)
        self.btn_auto_detect.setIcon(make_wlpg_icon("refresh", 16))
        self.btn_auto_detect.setToolTip("Re-run AI face and subject segmentation")
        self.btn_auto_detect.setStyleSheet("background: #FFFFFF; border: 1px solid #B0C0D8; border-radius: 3px; padding: 4px;")
        self.btn_auto_detect.clicked.connect(self._auto_detect_mask)
        action_row.addWidget(self.btn_auto_detect)
        grp_brush.vbox.addLayout(action_row)

        sidebar_layout.addWidget(grp_brush)

        # Background Adjustments Card
        grp_tone = _FineTuneGroup("Background Tone", sidebar)
        self.lbl_contrast_val = QLabel("Contrast: +15%", grp_tone)
        self.lbl_contrast_val.setStyleSheet("color: #333; font-size: 11px;")
        grp_tone.vbox.addWidget(self.lbl_contrast_val)

        self.sld_contrast = _AeroFineTuneSlider(80, 180, 115, grp_tone)
        self.sld_contrast.setValue(115)
        self.sld_contrast.valueChanged.connect(self._on_contrast_changed)
        grp_tone.vbox.addWidget(self.sld_contrast)

        self.lbl_bright_val = QLabel("Brightness: 0", grp_tone)
        self.lbl_bright_val.setStyleSheet("color: #333; font-size: 11px;")
        grp_tone.vbox.addWidget(self.lbl_bright_val)

        self.sld_bright = _AeroFineTuneSlider(-40, 40, 0, grp_tone)
        self.sld_bright.setValue(0)
        self.sld_bright.valueChanged.connect(self._on_bright_changed)
        grp_tone.vbox.addWidget(self.sld_bright)

        sidebar_layout.addWidget(grp_tone)

        # History & Preview Card
        grp_hist = _FineTuneGroup("History & Preview", sidebar)
        hist_row = QHBoxLayout()
        self.btn_undo = QPushButton("Undo", grp_hist)
        self.btn_undo.setIcon(make_wlpg_icon("undo", 16))
        self.btn_undo.setEnabled(False)
        self.btn_undo.setStyleSheet("background: #FFFFFF; border: 1px solid #B0C0D8; border-radius: 3px; padding: 4px;")
        self.btn_undo.clicked.connect(self._undo_mask)
        hist_row.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("Redo", grp_hist)
        self.btn_redo.setIcon(make_wlpg_icon("redo", 16))
        self.btn_redo.setEnabled(False)
        self.btn_redo.setStyleSheet("background: #FFFFFF; border: 1px solid #B0C0D8; border-radius: 3px; padding: 4px;")
        self.btn_redo.clicked.connect(self._redo_mask)
        hist_row.addWidget(self.btn_redo)
        grp_hist.vbox.addLayout(hist_row)

        self.btn_compare = QPushButton("Hold to Compare", grp_hist)
        self.btn_compare.setToolTip("Press and hold to preview original photo")
        self.btn_compare.setStyleSheet("background: #F0F4FA; border: 1px solid #7A96DF; font-weight: bold; padding: 6px 12px; border-radius: 3px;")
        self.btn_compare.pressed.connect(self._show_original)
        self.btn_compare.released.connect(self._restore_color_pop)
        grp_hist.vbox.addWidget(self.btn_compare)

        sidebar_layout.addWidget(grp_hist)
        sidebar_layout.addStretch(1)

        # Save and Cancel Buttons
        btn_save = QPushButton("Save New Photo...")
        btn_save.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3c82e6, stop:1 #1c52ba);
                color: white; font-weight: bold; border: 1px solid #143e8c;
                border-radius: 4px; padding: 8px 16px; font-size: 11px;
            }
            QPushButton:hover { background: #4d8ff2; }
        """)
        btn_save.clicked.connect(self._save_photo)
        sidebar_layout.addWidget(btn_save)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("background: #FFFFFF; border: 1px solid #B0C0D8; border-radius: 4px; padding: 6px;")
        btn_cancel.clicked.connect(self.reject)
        sidebar_layout.addWidget(btn_cancel)

        sidebar_scroll.setWidget(sidebar)
        main_layout.addWidget(sidebar_scroll)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._first_fit and self.scene.sceneRect().width() > 0:
            self._first_fit = False
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self._is_fit_mode = True
            self._on_zoom_changed(self.view.transform().m11())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, "_is_fit_mode", True) and hasattr(self, "view") and hasattr(self, "scene"):
            if self.scene.sceneRect().isValid() and self.scene.sceneRect().width() > 0:
                self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def _toggle_fit_zoom(self) -> None:
        if self._is_fit_mode:
            self.view.resetTransform()
            self._is_fit_mode = False
            self._on_zoom_changed(1.0)
        else:
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self._is_fit_mode = True
            self._on_zoom_changed(self.view.transform().m11())

    def _apply_zoom(self, factor: float) -> None:
        self._is_fit_mode = False
        self.view.scale(factor, factor)
        self._on_zoom_changed(self.view.transform().m11())

    def _on_slider_zoom_moved(self, val: int) -> None:
        self._is_fit_mode = False
        target_scale = val / 100.0
        current_scale = self.view.transform().m11()
        if abs(current_scale) > 1e-4:
            factor = target_scale / current_scale
            self.view.scale(factor, factor)
            self._on_zoom_changed(target_scale)

    def _on_zoom_changed(self, scale_val: float) -> None:
        pct = int(round(scale_val * 100))
        if hasattr(self, "zoom_bar"):
            self.zoom_bar.set_zoom_level(pct)

    def _precompute_proxy_bw(self) -> None:
        gray = cv2.cvtColor(self.proxy_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray -= 128.0
        gray *= self.bg_contrast
        gray += 128.0 + self.bg_brightness
        np.clip(gray, 0.0, 255.0, out=gray)
        self.proxy_bw = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self._proxy_bw_f32 = self.proxy_bw.astype(np.float32)
        if not hasattr(self, "_proxy_bgr_f32"):
            self._proxy_bgr_f32 = self.proxy_bgr.astype(np.float32)
        if not hasattr(self, "_proxy_lab_f32"):
            self._proxy_lab_f32 = cv2.cvtColor(self.proxy_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    def _toggle_eyedropper(self) -> None:
        self.brush_mode = "eyedropper"
        self.view.set_eyedropper_mode(True)
        self.lbl_status.setText("Eyedropper: Click to pick colour. Shift+Click to add (+), Alt/Ctrl+Click to remove (-).")

    def _set_brush_mode(self, mode: str) -> None:
        self.brush_mode = mode
        self.view.set_eyedropper_mode(False)
        self.btn_keep.setChecked(mode == "keep_color")
        self.btn_bw.setChecked(mode == "make_bw")
        self.btn_pick_color.setChecked(False)
        self.lbl_status.setText(f"Active Brush: {'Keep Vibrant Colour (+)' if mode == 'keep_color' else 'Make Black and White (-)'}")

    def _on_color_picked(self, sx: int, sy: int, mode: str = "replace") -> None:
        ph, pw = self.proxy_bgr.shape[:2]
        if not (0 <= sx < pw and 0 <= sy < ph):
            return
        target_bgr = tuple(int(c) for c in self.proxy_bgr[sy, sx])
        self.last_picked_color = target_bgr

        if mode == "replace":
            self.picked_colors = [target_bgr]
            action_desc = "Selected colour"
        elif mode == "add":
            # Avoid adding duplicate close colors
            is_dup = any(np.linalg.norm(np.array(c) - np.array(target_bgr)) < 10 for c in self.picked_colors)
            if not is_dup:
                self.picked_colors.append(target_bgr)
            action_desc = "Added colour"
        elif mode == "remove":
            if self.picked_colors:
                closest_idx = min(
                    range(len(self.picked_colors)),
                    key=lambda idx: np.linalg.norm(np.array(self.picked_colors[idx]) - np.array(target_bgr)),
                )
                self.picked_colors.pop(closest_idx)
                action_desc = "Removed colour"
            else:
                action_desc = "No colours to remove"

        self._on_stroke_started()
        self._recompute_picked_colors_mask()
        self._update_color_swatches_ui()
        self.lbl_status.setText(
            f"{action_desc} at ({sx}, {sy}). Active colours: {len(self.picked_colors)}. (Shift+Click to add, Alt/Ctrl+Click to remove)"
        )

    def _recompute_picked_colors_mask(self, refine_edges: bool = True) -> None:
        if not self.picked_colors:
            self.proxy_mask = np.zeros(self.proxy_bgr.shape[:2], dtype=np.float32)
        else:
            if not hasattr(self, "_proxy_lab_f32"):
                self._proxy_lab_f32 = cv2.cvtColor(self.proxy_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            self.proxy_mask = compute_multi_color_similarity_mask(
                self.proxy_bgr,
                self.picked_colors,
                tolerance=self.color_tolerance,
                refine_edges=refine_edges,
                lab_img=self._proxy_lab_f32,
            )
        self._fast_refresh_proxy()

    def _update_color_swatches_ui(self) -> None:
        if not hasattr(self, "swatches_layout"):
            return
        while self.swatches_layout.count():
            item = self.swatches_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if not self.picked_colors:
            self.swatches_container.setVisible(False)
            return

        self.swatches_container.setVisible(True)
        lbl_info = QLabel(f"Colours ({len(self.picked_colors)}):", self.swatches_container)
        lbl_info.setStyleSheet("font-size: 10px; color: #555; font-weight: bold;")
        self.swatches_layout.addWidget(lbl_info)

        max_visible_chips = 6
        for idx in range(min(max_visible_chips, len(self.picked_colors))):
            b, g, r = self.picked_colors[idx]
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            chip = QPushButton(self.swatches_container)
            chip.setFixedSize(16, 16)
            chip.setCursor(Qt.PointingHandCursor)
            chip.setToolTip(f"Colour {hex_color.upper()}\nClick to remove this colour")
            chip.setStyleSheet(f"""
                QPushButton {{
                    background-color: {hex_color};
                    border: 1.5px solid #FFFFFF;
                    border-radius: 8px;
                }}
                QPushButton:hover {{
                    border: 1.5px solid #E81123;
                }}
            """)
            chip.clicked.connect(lambda checked=False, i=idx: self._remove_picked_color(i))
            self.swatches_layout.addWidget(chip)

        if len(self.picked_colors) > max_visible_chips:
            overflow_count = len(self.picked_colors) - max_visible_chips
            lbl_more = QLabel(f"+{overflow_count}", self.swatches_container)
            lbl_more.setStyleSheet("font-size: 10px; color: #666; font-weight: bold; padding: 0 2px;")
            lbl_more.setToolTip(f"{len(self.picked_colors)} total isolated colours")
            self.swatches_layout.addWidget(lbl_more)

        self.swatches_layout.addStretch(1)

        btn_clear = QPushButton("Clear", self.swatches_container)
        btn_clear.setCursor(Qt.PointingHandCursor)
        btn_clear.setStyleSheet("font-size: 10px; color: #0078D4; background: transparent; border: none; text-decoration: underline;")
        btn_clear.clicked.connect(self._clear_picked_colors)
        self.swatches_layout.addWidget(btn_clear)

    def _remove_picked_color(self, idx: int) -> None:
        if 0 <= idx < len(self.picked_colors):
            self.picked_colors.pop(idx)
            self._on_stroke_started()
            self._recompute_picked_colors_mask()
            self._update_color_swatches_ui()
            self.lbl_status.setText(f"Removed colour. Remaining active: {len(self.picked_colors)}.")

    def _clear_picked_colors(self) -> None:
        self.picked_colors.clear()
        self._on_stroke_started()
        self._recompute_picked_colors_mask()
        self._update_color_swatches_ui()
        self.lbl_status.setText("Cleared all picked colours. Eyedropper ready.")

    def _on_tolerance_pressed(self) -> None:
        self._is_tol_sliding = True

    def _on_tolerance_released(self) -> None:
        self._is_tol_sliding = False
        self._on_tolerance_changed(self.sld_tol.value())

    def _on_tolerance_changed(self, val: int) -> None:
        self.color_tolerance = val / 100.0
        self.lbl_tol_val.setText(f"Colour Tolerance: {val}%")
        refine = not getattr(self, "_is_tol_sliding", False)
        if self.picked_colors and self.brush_mode == "eyedropper":
            self._recompute_picked_colors_mask(refine_edges=refine)
            self.lbl_status.setText(f"Adjusted colour tolerance to {val}% across {len(self.picked_colors)} colour(s).")
        elif self.last_picked_color is not None and self.brush_mode == "eyedropper":
            if not hasattr(self, "_proxy_lab_f32"):
                self._proxy_lab_f32 = cv2.cvtColor(self.proxy_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            self.proxy_mask = compute_multi_color_similarity_mask(
                self.proxy_bgr,
                [self.last_picked_color],
                tolerance=self.color_tolerance,
                refine_edges=refine,
                lab_img=self._proxy_lab_f32,
            )
            self._fast_refresh_proxy()
            self.lbl_status.setText(f"Adjusted colour tolerance to {val}%.")

    def _on_overlay_toggled(self, checked: bool) -> None:
        self.show_mask_overlay = bool(checked)
        self._fast_refresh_proxy()

    def _on_radius_changed(self, val: int) -> None:
        self.brush_radius = val
        self.lbl_radius_val.setText(f"Brush Size: {val} px")
        self.view.set_brush_radius(val)

    def _on_contrast_changed(self, val: int) -> None:
        self.bg_contrast = val / 100.0
        delta = val - 100
        sign = "+" if delta >= 0 else ""
        self.lbl_contrast_val.setText(f"Contrast: {sign}{delta}%")
        self._precompute_proxy_bw()
        self._refresh_canvas()

    def _on_bright_changed(self, val: int) -> None:
        self.bg_brightness = float(val)
        sign = "+" if val > 0 else ""
        self.lbl_bright_val.setText(f"Brightness: {sign}{val}")
        self._precompute_proxy_bw()
        self._refresh_canvas()

    def _on_stroke_started(self) -> None:
        self.undo_stack.append(self.proxy_mask.copy())
        if len(self.undo_stack) > 25:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.btn_undo.setEnabled(True)
        self.btn_redo.setEnabled(False)

    def _on_brush_applied(self, sx: int, sy: int) -> None:
        ph, pw = self.proxy_bgr.shape[:2]
        if 0 <= sx < pw and 0 <= sy < ph:
            self.proxy_mask = refine_mask_brush(
                self.proxy_mask,
                self.proxy_bgr,
                (sx, sy),
                radius=self.brush_radius,
                mode=self.brush_mode,
            )
            self._fast_refresh_proxy()

    def _fast_refresh_proxy(self) -> None:
        if self.proxy_bw is None or not hasattr(self, "_proxy_bw_f32"):
            self._precompute_proxy_bw()
        if not hasattr(self, "_proxy_bgr_f32"):
            self._proxy_bgr_f32 = self.proxy_bgr.astype(np.float32)
        m3 = self.proxy_mask[:, :, None]
        blended_f = self._proxy_bgr_f32 * m3
        blended_f += self._proxy_bw_f32 * (1.0 - m3)
        if self.show_mask_overlay:
            bg_alpha = (1.0 - self.proxy_mask) * 0.40
            inv_alpha = 1.0 - bg_alpha
            blended_f[:, :, 0] *= inv_alpha
            blended_f[:, :, 1] *= inv_alpha
            blended_f[:, :, 2] *= inv_alpha
            blended_f[:, :, 2] += 230.0 * bg_alpha
        np.clip(blended_f, 0.0, 255.0, out=blended_f)
        self.bg_item.setPixmap(_cv2_to_qpixmap(blended_f.astype(np.uint8)))

    def _refresh_canvas(self) -> None:
        self._fast_refresh_proxy()
        ph, pw = self.proxy_bgr.shape[:2]
        self.scene.setSceneRect(QRectF(0, 0, pw, ph))

    def _invert_mask(self) -> None:
        self._on_stroke_started()
        self.proxy_mask = 1.0 - self.proxy_mask
        self._fast_refresh_proxy()
        self.lbl_status.setText("Inverted mask: subject and background swapped.")

    def _auto_detect_mask(self) -> None:
        self._on_stroke_started()
        full_mask = estimate_subject_mask(self.original_bgr)
        self.proxy_mask = cv2.resize(full_mask, (self.proxy_bgr.shape[1], self.proxy_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        self._fast_refresh_proxy()
        self.lbl_status.setText("Re-detected subject segmentation.")

    def _undo_mask(self) -> None:
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.undo_stack.pop())
            self.proxy_mask = self.undo_stack[-1].copy()
            self._fast_refresh_proxy()
            self.btn_undo.setEnabled(len(self.undo_stack) > 1)
            self.btn_redo.setEnabled(True)

    def _redo_mask(self) -> None:
        if self.redo_stack:
            restored = self.redo_stack.pop()
            self.undo_stack.append(restored)
            self.proxy_mask = restored.copy()
            self._fast_refresh_proxy()
            self.btn_undo.setEnabled(True)
            self.btn_redo.setEnabled(len(self.redo_stack) > 0)

    def _show_original(self) -> None:
        self.bg_item.setPixmap(_cv2_to_qpixmap(self.proxy_bgr))

    def _restore_color_pop(self) -> None:
        self._fast_refresh_proxy()

    def _save_photo(self) -> None:
        out_path = prompt_save_new_file(self, self.file_path, "colourPop", ".jpg")
        if not out_path:
            return

        self.lbl_status.setText("Rendering high-resolution Colour Pop photo...")
        QApplication.processEvents()

        h, w = self.original_bgr.shape[:2]
        if self.proxy_scale != 1.0:
            full_mask = cv2.resize(self.proxy_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            full_mask = self.proxy_mask.copy()

        full_mask = guided_filter_matte(self.original_bgr, full_mask, radius=16, eps=1e-3)

        full_color_pop = apply_color_pop(
            self.original_bgr,
            full_mask,
            bg_contrast=self.bg_contrast,
            bg_brightness=self.bg_brightness,
        )

        try:
            cv2.imwrite(str(out_path), full_color_pop, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            self.saved_file_path = out_path
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save photo:\n{e}")


# Backwards compatibility aliases
ColorPopCanvasView = ColourPopCanvasView
ColorPopDialog = ColourPopDialog


# =============================================================================
# 5. Portrait Depth Bokeh & Foreground Fill Light Dialog
# =============================================================================

class PortraitBokehDialog(QDialog):
    """
    Portrait Bokeh & Fill Light Dialog:
    - Background lens blur kernel with real-time slider readout.
    - Soft-knee foreground face/subject fill light.
    - Hold to Compare and instant display proxy rendering.
    """

    def __init__(
        self,
        image_bgr: np.ndarray,
        file_path: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if len(image_bgr.shape) == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

        self.original_bgr = image_bgr.copy()
        self.file_path = file_path
        self.output_dir = file_path.parent if file_path else Path.home()
        self.saved_file_path: Path | None = None
        self._is_fit_mode: bool = True

        h, w = self.original_bgr.shape[:2]
        max_proxy_dim = 1200
        if max(h, w) > max_proxy_dim:
            self.proxy_scale = max_proxy_dim / float(max(h, w))
            pw = max(1, int(round(w * self.proxy_scale)))
            ph = max(1, int(round(h * self.proxy_scale)))
            self.proxy_bgr = cv2.resize(self.original_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        else:
            self.proxy_scale = 1.0
            self.proxy_bgr = self.original_bgr.copy()

        full_mask = estimate_subject_mask(self.original_bgr)
        self.proxy_mask = cv2.resize(full_mask, (self.proxy_bgr.shape[1], self.proxy_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        self.proxy_bokeh_pyramid = build_bokeh_pyramid(self.proxy_bgr, max_blur_radius=36.0)
        self.rendered_proxy: np.ndarray | None = None

        self.setWindowTitle("Portrait Depth Bokeh & Fill Light")
        self.setWindowIcon(make_wlpg_icon("noise_reduction", 24))
        self.resize(1240, 800)
        self.setMinimumSize(960, 620)
        self._init_ui()
        self._render_proxy()

    def _init_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ---------------- Left: Canvas & Zoom Bar ----------------
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.view = ZoomablePreviewCanvas(self)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.view.setStyleSheet("background: #1C1C1E; border: 1px solid #555558; border-radius: 4px;")
        self.bg_item = QGraphicsPixmapItem()
        self.scene.addItem(self.bg_item)
        self.view.zoom_changed.connect(self._on_zoom_changed)
        left_layout.addWidget(self.view, 1)

        # Status & Zoom bar
        status_bar = QHBoxLayout()
        status_bar.setContentsMargins(4, 2, 4, 2)
        status_bar.setSpacing(8)

        lbl_res = QLabel(f"Resolution: {self.original_bgr.shape[1]} × {self.original_bgr.shape[0]} px", self)
        lbl_res.setStyleSheet("color: #666; font-size: 11px;")
        status_bar.addWidget(lbl_res)

        status_bar.addStretch(1)

        self.zoom_bar = CreateStudioZoomBar(self)
        self.zoom_bar.fitRequested.connect(self._toggle_fit_zoom)
        self.zoom_bar.zoomInRequested.connect(lambda: self._apply_zoom(1.25))
        self.zoom_bar.zoomOutRequested.connect(lambda: self._apply_zoom(1.0 / 1.25))
        self.zoom_bar.zoomSliderMoved.connect(self._on_slider_zoom_moved)
        status_bar.addWidget(self.zoom_bar)

        left_layout.addLayout(status_bar)
        main_layout.addWidget(left_widget, 1)

        # ---------------- Right: Settings Sidebar ----------------
        sidebar_scroll = QScrollArea(self)
        sidebar_scroll.setFixedWidth(320)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        sidebar_content = QWidget()
        sidebar_content.setMaximumWidth(312)
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(2, 0, 4, 0)
        sidebar_layout.setSpacing(8)

        # 1. Depth Bokeh
        grp_bokeh = _FineTuneGroup("Lens Bokeh Blur")
        lbl_b_hdr = QHBoxLayout()
        lbl_b_title = QLabel("Blur Amount:")
        lbl_b_title.setStyleSheet("font-size: 11px; color: #444;")
        self.lbl_blur_val = QLabel("16 px")
        self.lbl_blur_val.setStyleSheet("font-weight: bold; font-size: 11px; color: #1e395b;")
        lbl_b_hdr.addWidget(lbl_b_title)
        lbl_b_hdr.addStretch(1)
        lbl_b_hdr.addWidget(self.lbl_blur_val)
        grp_bokeh.layout().addLayout(lbl_b_hdr)

        self.sld_blur = _AeroFineTuneSlider(default_value=16)
        self.sld_blur.setRange(0, 40)
        self.sld_blur.setValue(16)
        self.sld_blur.valueChanged.connect(self._on_blur_changed)
        grp_bokeh.layout().addWidget(self.sld_blur)
        sidebar_layout.addWidget(grp_bokeh)

        # 2. Portrait Fill Light
        grp_light = _FineTuneGroup("Portrait Lighting")
        lbl_l_hdr = QHBoxLayout()
        lbl_l_title = QLabel("Subject Fill Light:")
        lbl_l_title.setStyleSheet("font-size: 11px; color: #444;")
        self.lbl_fill_val = QLabel("22%")
        self.lbl_fill_val.setStyleSheet("font-weight: bold; font-size: 11px; color: #1e395b;")
        lbl_l_hdr.addWidget(lbl_l_title)
        lbl_l_hdr.addStretch(1)
        lbl_l_hdr.addWidget(self.lbl_fill_val)
        grp_light.layout().addLayout(lbl_l_hdr)

        self.sld_fill = _AeroFineTuneSlider(default_value=22)
        self.sld_fill.setRange(0, 50)
        self.sld_fill.setValue(22)
        self.sld_fill.valueChanged.connect(self._on_fill_changed)
        grp_light.layout().addWidget(self.sld_fill)
        sidebar_layout.addWidget(grp_light)

        # 3. Preview & Compare
        grp_preview = _FineTuneGroup("Preview")
        self.btn_compare = QPushButton("Hold to Compare Original", self)
        self.btn_compare.setStyleSheet("padding: 6px 12px; border: 1px solid #7a96df; border-radius: 3px; background: #eef3fa; font-weight: bold;")
        self.btn_compare.pressed.connect(self._show_original)
        self.btn_compare.released.connect(self._restore_bokeh)
        grp_preview.layout().addWidget(self.btn_compare)
        sidebar_layout.addWidget(grp_preview)

        sidebar_layout.addStretch(1)

        # Save and Cancel
        btn_box = QHBoxLayout()
        btn_box.setSpacing(6)

        self.btn_save = QPushButton("Save New...", self)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3c82e6, stop:1 #1c52ba);
                color: white; font-weight: bold; border: 1px solid #143e8c;
                border-radius: 4px; padding: 6px 14px; min-width: 90px;
            }
            QPushButton:hover { background: #4d8ff2; }
        """)
        self.btn_save.clicked.connect(self._save_photo)
        btn_box.addWidget(self.btn_save, 1)

        btn_cancel = QPushButton("Cancel", self)
        btn_cancel.setStyleSheet("padding: 6px 12px; border-radius: 4px; border: 1px solid #ccc; background: #f0f0f0;")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        sidebar_layout.addLayout(btn_box)

        sidebar_scroll.setWidget(sidebar_content)
        main_layout.addWidget(sidebar_scroll)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if hasattr(self, "scene") and self.scene.sceneRect().isValid() and self.scene.sceneRect().width() > 0:
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self._is_fit_mode = True
            current_scale = self.view.transform().m11()
            self._on_zoom_changed(current_scale)
            if hasattr(self, "zoom_bar"):
                self.zoom_bar.set_is_fit(True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, "_is_fit_mode", True) and hasattr(self, "view") and hasattr(self, "scene"):
            if self.scene.sceneRect().isValid() and self.scene.sceneRect().width() > 0:
                self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
                self._on_zoom_changed(self.view.transform().m11())
                if hasattr(self, "zoom_bar"):
                    self.zoom_bar.set_is_fit(True)

    def _toggle_fit_zoom(self) -> None:
        if not hasattr(self, "scene") or not self.scene.sceneRect().isValid() or self.scene.sceneRect().width() == 0:
            return
        if self._is_fit_mode:
            self._is_fit_mode = False
            self.view.resetTransform()
            self._on_zoom_changed(1.0)
            self.zoom_bar.set_is_fit(False)
        else:
            self._is_fit_mode = True
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            current_scale = self.view.transform().m11()
            self._on_zoom_changed(current_scale)
            self.zoom_bar.set_is_fit(True)

    def _apply_zoom(self, factor: float) -> None:
        self._is_fit_mode = False
        self.zoom_bar.set_is_fit(False)
        self.view.scale(factor, factor)
        current_scale = self.view.transform().m11()
        self._on_zoom_changed(current_scale)

    def _on_slider_zoom_moved(self, pct: int) -> None:
        self._is_fit_mode = False
        self.zoom_bar.set_is_fit(False)
        target_scale = max(0.1, min(5.0, pct / 100.0))
        current_scale = self.view.transform().m11()
        if abs(current_scale) > 1e-4:
            factor = target_scale / current_scale
            self.view.scale(factor, factor)
        self._on_zoom_changed(target_scale)

    def _on_zoom_changed(self, scale_val: float) -> None:
        if hasattr(self, "zoom_bar"):
            self.zoom_bar.update_zoom_display(scale_val)

    def _on_blur_changed(self, val: int) -> None:
        self.lbl_blur_val.setText(f"{val} px")
        self._render_proxy()

    def _on_fill_changed(self, val: int) -> None:
        self.lbl_fill_val.setText(f"{val}%")
        self._render_proxy()

    def _render_proxy(self) -> None:
        b_rad = max(1, int(self.sld_blur.value() * self.proxy_scale))
        fill_val = self.sld_fill.value() / 100.0
        self.rendered_proxy = apply_portrait_bokeh_and_fill_light(
            self.proxy_bgr,
            self.proxy_mask,
            bokeh_radius=b_rad,
            fill_light=fill_val,
            bokeh_pyramid=self.proxy_bokeh_pyramid,
        )
        pm = _cv2_to_qpixmap(self.rendered_proxy)
        self.bg_item.setPixmap(pm)
        ph, pw = self.proxy_bgr.shape[:2]
        self.scene.setSceneRect(QRectF(0, 0, pw, ph))
        if self._is_fit_mode and self.isVisible():
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self._on_zoom_changed(self.view.transform().m11())
            if hasattr(self, "zoom_bar"):
                self.zoom_bar.set_is_fit(True)

    def _show_original(self) -> None:
        self.bg_item.setPixmap(_cv2_to_qpixmap(self.proxy_bgr))

    def _restore_bokeh(self) -> None:
        if self.rendered_proxy is not None:
            self.bg_item.setPixmap(_cv2_to_qpixmap(self.rendered_proxy))
        ph, pw = self.proxy_bgr.shape[:2]
        self.scene.setSceneRect(QRectF(0, 0, pw, ph))

    def _show_original(self) -> None:
        self.bg_item.setPixmap(_cv2_to_qpixmap(self.proxy_bgr))

    def _restore_bokeh(self) -> None:
        if self.rendered_proxy is not None:
            self.bg_item.setPixmap(_cv2_to_qpixmap(self.rendered_proxy))

    def _save_photo(self) -> None:
        out_path = prompt_save_new_file(self, self.file_path, "portraitBokeh", ".jpg")
        if not out_path:
            return

        h, w = self.original_bgr.shape[:2]
        if self.proxy_scale != 1.0:
            full_mask = cv2.resize(self.proxy_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            full_mask = self.proxy_mask.copy()

        full_mask = guided_filter_matte(self.original_bgr, full_mask, radius=16, eps=1e-3)

        full_bokeh = apply_portrait_bokeh_and_fill_light(
            self.original_bgr,
            full_mask,
            bokeh_radius=self.sld_blur.value(),
            fill_light=self.sld_fill.value() / 100.0,
        )

        try:
            cv2.imwrite(str(out_path), full_bokeh, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            self.saved_file_path = out_path
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save photo:\n{e}")


# =============================================================================
# 6. 3D Cinematic Motion (Faux-3D Parallax MP4 Generator) Dialog
# =============================================================================

class CinematicMotionWorker(QThread):
    progress = Signal(int)
    finished = Signal(bool, str)

    def __init__(
        self,
        image_bgr: np.ndarray | list[np.ndarray],
        out_path: Path,
        duration: float,
        motion_style: str = "orbital",
        intensity: float = 0.5,
        enable_bokeh: bool = True,
        is_reel: bool = False,
    ) -> None:
        super().__init__()
        self.image_bgr = image_bgr
        self.out_path = out_path
        self.duration = duration
        self.motion_style = motion_style
        self.intensity = intensity
        self.enable_bokeh = enable_bokeh
        self.is_reel = is_reel

    def run(self) -> None:
        try:
            if self.is_reel and isinstance(self.image_bgr, list):
                success = render_3d_cinematic_motion_reel(
                    images_bgr=self.image_bgr,
                    output_path=self.out_path,
                    duration_per_clip=self.duration,
                    transition_sec=0.8,
                    fps=30,
                    motion_style=self.motion_style,
                    intensity=self.intensity,
                    enable_bokeh=self.enable_bokeh,
                    progress_callback=self.progress.emit,
                )
            else:
                img = self.image_bgr[0] if isinstance(self.image_bgr, list) else self.image_bgr
                h, w = img.shape[:2]
                target_res = compute_target_dimensions(w, h, max_long_edge=1920)
                success = render_3d_cinematic_motion_video(
                    image_bgr=img,
                    output_path=self.out_path,
                    duration_sec=self.duration,
                    fps=30,
                    resolution=target_res,
                    motion_style=self.motion_style,
                    intensity=self.intensity,
                    enable_bokeh=self.enable_bokeh,
                    progress_callback=self.progress.emit,
                )
            self.finished.emit(success, "Video rendered successfully" if success else "Failed to open video encoder")
        except Exception as e:
            self.finished.emit(False, str(e))


class CinematicMotionDialog(QDialog):
    """
    3D Cinematic Motion Video Generator:
    - Previews animated 3D virtual camera parallax loop in real time from precomputed 2.5D layers.
    - Estimates foreground/background layers with disocclusion background inpainting to eliminate rubber edge stretching.
    - Renders smooth 30fps .mp4 virtual camera dolly video or multi-photo 3D Memories Reel.
    """

    def __init__(
        self,
        image_bgr: np.ndarray | list[np.ndarray],
        file_path: Path | list[Path] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if isinstance(image_bgr, list):
            self.multi_images = [
                cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2
                else (cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if img.shape[2] == 4 else img.copy())
                for img in image_bgr if img is not None and img.size > 0
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
        self.saved_file_path: Path | None = None
        self._worker: CinematicMotionWorker | None = None

        self._preview_frames: list[QPixmap] = []
        self._current_frame_idx: int = 0
        self._timer: QTimer | None = None

        # Precompute proxy layers once for display to ensure zero per-frame lag
        h, w = self.original_bgr.shape[:2]
        pw, ph = compute_target_dimensions(w, h, max_long_edge=520)
        self._proxy_bgr = cv2.resize(self.original_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        self._proxy_layers = prepare_3d_parallax_layers(self._proxy_bgr)
        self._proxy_size = (pw, ph)

        self.setWindowTitle("3D Cinematic Motion Video Generator")
        self.setWindowIcon(make_wlpg_icon("slideshow", 24))
        self.resize(1240, 800)
        self.setMinimumSize(960, 620)
        self._init_ui()
        self._build_preview_loop()

    def _init_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ---------------- Left: Live Animated View & Progress / Status ----------------
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.lbl_thumb = QLabel(self)
        self.lbl_thumb.setAlignment(Qt.AlignCenter)
        self.lbl_thumb.setStyleSheet("background: #1C1C1E; border: 1px solid #555558; border-radius: 4px;")
        left_layout.addWidget(self.lbl_thumb, 1)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet("QProgressBar { height: 8px; border: none; background: #e0e0e0; border-radius: 4px; } QProgressBar::chunk { background: #2D72BC; border-radius: 4px; }")
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Status bar
        status_bar = QHBoxLayout()
        status_bar.setContentsMargins(4, 2, 4, 2)
        status_bar.setSpacing(8)

        self.lbl_status = QLabel("Live 3D Parallax Preview (Interactive Loop)", self)
        self.lbl_status.setStyleSheet("color: #444; font-size: 11px;")
        status_bar.addWidget(self.lbl_status)

        status_bar.addStretch(1)

        h_orig, w_orig = self.original_bgr.shape[:2]
        exp_w, exp_h = compute_target_dimensions(w_orig, h_orig, max_long_edge=1920)
        lbl_res = QLabel(f"Export: {exp_w} × {exp_h} @ 30fps", self)
        lbl_res.setStyleSheet("color: #666; font-size: 11px;")
        status_bar.addWidget(lbl_res)

        left_layout.addLayout(status_bar)
        main_layout.addWidget(left_widget, 1)

        # ---------------- Right: Settings Sidebar ----------------
        sidebar_scroll = QScrollArea(self)
        sidebar_scroll.setFixedWidth(320)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        sidebar_content = QWidget()
        sidebar_content.setMaximumWidth(312)
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(2, 0, 4, 0)
        sidebar_layout.setSpacing(8)

        # 1. Camera Motion Presets
        grp_motion = _FineTuneGroup("Camera Trajectory")
        self.cmb_motion = QComboBox(self)
        self.cmb_motion.addItems([
            "Dolly Zoom (Vertigo)",
            "Parallax Drift",
            "3D Orbit",
            "Cinematic Pan",
        ])
        self.cmb_motion.setCurrentIndex(2)  # 3D Orbit default
        self.cmb_motion.currentIndexChanged.connect(self._build_preview_loop)
        grp_motion.layout().addWidget(self.cmb_motion)

        lbl_i_hdr = QHBoxLayout()
        lbl_i_title = QLabel("Motion Amplitude:")
        lbl_i_title.setStyleSheet("font-size: 11px; color: #444;")
        self.lbl_intensity_val = QLabel("50%")
        self.lbl_intensity_val.setStyleSheet("font-weight: bold; font-size: 11px; color: #1e395b;")
        lbl_i_hdr.addWidget(lbl_i_title)
        lbl_i_hdr.addStretch(1)
        lbl_i_hdr.addWidget(self.lbl_intensity_val)
        grp_motion.layout().addLayout(lbl_i_hdr)

        self.sld_intensity = _AeroFineTuneSlider(default_value=50)
        self.sld_intensity.setRange(10, 100)
        self.sld_intensity.setValue(50)
        self.sld_intensity.setToolTip("Parallax virtual camera displacement intensity (10% to 100%)")
        self.sld_intensity.valueChanged.connect(self._on_intensity_changed)
        grp_motion.layout().addWidget(self.sld_intensity)

        # Subtle background bokeh toggle
        self.chk_bokeh = QCheckBox("Enhance 3D depth with optical background blur", self)
        self.chk_bokeh.setChecked(True)
        self.chk_bokeh.setStyleSheet("color: #333; font-size: 11px; margin-top: 4px;")
        self.chk_bokeh.setToolTip("Applies depth-weighted optical blur to background plate to accentuate foreground separation")
        self.chk_bokeh.toggled.connect(self._build_preview_loop)
        grp_motion.layout().addWidget(self.chk_bokeh)

        sidebar_layout.addWidget(grp_motion)

        # 2. Video Duration
        grp_video = _FineTuneGroup("Video Settings")
        lbl_dur = QLabel("Clip Duration:")
        lbl_dur.setStyleSheet("font-size: 11px; color: #444;")
        grp_video.layout().addWidget(lbl_dur)
        self.cmb_duration = QComboBox(self)
        self.cmb_duration.addItems(["3 seconds (90 frames)", "5 seconds (150 frames)", "8 seconds (240 frames)"])
        self.cmb_duration.setCurrentIndex(1)
        grp_video.layout().addWidget(self.cmb_duration)

        lbl_enc = QLabel("Format: H.264 / MP4 (30 FPS)\nSeamless cosine loop trajectory", self)
        lbl_enc.setStyleSheet("color: #666; font-size: 10px; margin-top: 4px;")
        grp_video.layout().addWidget(lbl_enc)
        sidebar_layout.addWidget(grp_video)

        # 3. Multi-Photo Target (if multiple photos provided)
        if len(self.multi_images) > 1:
            grp_export = _FineTuneGroup("Export Target")
            self.cmb_export_mode = QComboBox(self)
            self.cmb_export_mode.addItem("Export Current Photo (MP4)")
            self.cmb_export_mode.addItem(f"Export 3D Memories Reel ({len(self.multi_images)} Photos, 0.8s Cross-Fades)")
            self.cmb_export_mode.setCurrentIndex(1)
            grp_export.layout().addWidget(self.cmb_export_mode)
            sidebar_layout.addWidget(grp_export)

        sidebar_layout.addStretch(1)

        # Save and Cancel
        btn_box = QHBoxLayout()
        btn_box.setSpacing(6)

        self.btn_render = QPushButton("Save Video...", self)
        self.btn_render.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3c82e6, stop:1 #1c52ba);
                color: white; font-weight: bold; border: 1px solid #143e8c;
                border-radius: 4px; padding: 6px 14px; min-width: 90px;
            }
            QPushButton:hover { background: #4d8ff2; }
            QPushButton:disabled { background: #ccc; color: #888; }
        """)
        self.btn_render.clicked.connect(self._start_render)
        btn_box.addWidget(self.btn_render, 1)

        btn_cancel = QPushButton("Cancel", self)
        btn_cancel.setStyleSheet("padding: 6px 12px; border-radius: 4px; border: 1px solid #ccc; background: #f0f0f0;")
        btn_cancel.clicked.connect(self._on_cancel)
        btn_box.addWidget(btn_cancel)

        sidebar_layout.addLayout(btn_box)

        sidebar_scroll.setWidget(sidebar_content)
        main_layout.addWidget(sidebar_scroll)

    def _on_intensity_changed(self, val: int) -> None:
        self.lbl_intensity_val.setText(f"{val}%")
        self._build_preview_loop()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_current_pixmap()

    def _update_current_pixmap(self) -> None:
        if not self._preview_frames or self._current_frame_idx >= len(self._preview_frames):
            return
        pm = self._preview_frames[self._current_frame_idx]
        avail_size = self.lbl_thumb.size()
        if avail_size.width() > 60 and avail_size.height() > 60:
            target_size = QSize(avail_size.width() - 12, avail_size.height() - 12)
            scaled = pm.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_thumb.setPixmap(scaled)
        else:
            self.lbl_thumb.setPixmap(pm)

    def _build_preview_loop(self) -> None:
        """Renders smooth parallax frames from precomputed proxy layers for instant 30 FPS preview."""
        if self._timer:
            self._timer.stop()

        style_txt = self.cmb_motion.currentText()
        if "Dolly" in style_txt:
            style = "dolly_zoom"
        elif "Drift" in style_txt:
            style = "parallax_drift"
        elif "Pan" in style_txt:
            style = "cinematic_pan"
        else:
            style = "3d_orbit"

        intensity = self.sld_intensity.value() / 100.0
        enable_bokeh = self.chk_bokeh.isChecked() if hasattr(self, "chk_bokeh") else True

        pw, ph = self._proxy_size
        self._preview_frames.clear()
        num_frames = 24
        for i in range(num_frames):
            t = i / float(num_frames)
            frame = render_3d_cinematic_motion_frame(
                foreground_bgr=self._proxy_layers,
                t=t,
                canvas_size=(pw, ph),
                motion_style=style,
                intensity=intensity,
                enable_bokeh=enable_bokeh,
            )
            self._preview_frames.append(_cv2_to_qpixmap(frame))

        if self._preview_frames:
            self._current_frame_idx = 0
            self._update_current_pixmap()
            if not self._timer:
                self._timer = QTimer(self)
                self._timer.timeout.connect(self._advance_preview_frame)
            self._timer.start(33)  # Smooth 30 FPS playback

    def _advance_preview_frame(self) -> None:
        if not self._preview_frames:
            return
        self._current_frame_idx = (self._current_frame_idx + 1) % len(self._preview_frames)
        self._update_current_pixmap()

    def _start_render(self) -> None:
        is_reel = (
            hasattr(self, "cmb_export_mode")
            and self.cmb_export_mode.currentIndex() == 1
            and len(self.multi_images) > 1
        )
        default_tag = "3D_Memories_Reel" if is_reel else "cinematic3D"
        out_path = prompt_save_new_file(
            self,
            self.file_path,
            default_tag,
            ".mp4",
            "MP4 Video (*.mp4);;All Files (*.*)",
        )
        if not out_path:
            return

        if self._timer:
            self._timer.stop()
        self.btn_render.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Preparing 3D depth layers and inpainting disocclusions...")

        dur_text = self.cmb_duration.currentText()
        duration = 3.0 if "3" in dur_text else (8.0 if "8" in dur_text else 5.0)

        style_txt = self.cmb_motion.currentText()
        if "Dolly" in style_txt:
            style = "dolly_zoom"
        elif "Drift" in style_txt:
            style = "parallax_drift"
        elif "Pan" in style_txt:
            style = "cinematic_pan"
        else:
            style = "3d_orbit"

        intensity = self.sld_intensity.value() / 100.0
        enable_bokeh = self.chk_bokeh.isChecked() if hasattr(self, "chk_bokeh") else True

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
        self.btn_render.setEnabled(True)
        if ok and out_path.exists():
            self.saved_file_path = out_path
            self.lbl_status.setText(f"✓ Video saved: {out_path.name}")
            self.accept()
        else:
            QMessageBox.critical(self, "Render Error", f"Could not render 3D video:\n{msg}")
            if self._timer:
                self._timer.start(33)

    def _on_cancel(self) -> None:
        if self._timer:
            self._timer.stop()
        self.reject()

    def closeEvent(self, event) -> None:
        if self._timer:
            self._timer.stop()
        super().closeEvent(event)

