"""Integrated Google Photos-Style Colour Pop Workspace."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QRectF, QSize, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.imaging.creative_effects import (
    apply_color_pop,
    compute_multi_color_similarity_mask,
    estimate_subject_mask,
    guided_filter_matte,
    refine_mask_brush,
)
from src.ui.create_studio.base import CreateWorkspaceBase, RibbonActionSpec
from src.ui.create_studio.shared_widgets import (
    AeroStudioSlider,
    CreateStudioZoomBar,
    StudioCardGroup,
    _cv2_to_qpixmap,
    build_studio_panel_qss,
    prompt_save_new_file,
)
from src.ui.create_studio_dialogs import ColourPopCanvasView
from src.ui.ribbon_toolbar import make_wlpg_icon
from src.ui.theme_tokens import get_theme_tokens


class ColourPopWorkspace(CreateWorkspaceBase):
    """Full-window integrated Colour Pop workspace with brush, eyedropper, and Quick Styles tray."""

    workspace_id = "colour_pop"
    display_title = "Colour Pop"
    tool_save_label = "Colour Pop photo"

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
        self.proxy_mask = cv2.resize(
            full_mask, (self.proxy_bgr.shape[1], self.proxy_bgr.shape[0]), interpolation=cv2.INTER_LINEAR
        )

        self.undo_stack: list[np.ndarray] = [self.proxy_mask.copy()]
        self.redo_stack: list[np.ndarray] = []

        self.brush_mode = "keep_color"
        self.brush_radius = 28
        self.bg_contrast = 1.15
        self.bg_brightness = 0.0
        self.color_tolerance = 0.25
        self.show_mask_overlay = False
        self._comparing_original = False
        self.last_picked_color: tuple[int, int, int] | None = None
        self._is_fit_mode = True
        self._dirty = False
        self.picked_colors: list[tuple[int, int, int]] = []
        self._is_tol_sliding = False
        self.smart_brush = True
        self._active_quick_style: str | None = None

        self.proxy_bw: np.ndarray | None = None
        self._precompute_proxy_bw()

        self._build_canvas_ui()
        self._build_inspector_ui()
        self.apply_theme_mode("wlpg_aero")
        self._refresh_canvas()

    # ------------------------------------------------------------------
    # Workspace Contract
    # ------------------------------------------------------------------
    def inspector_widget(self) -> QWidget:
        return self._inspector_root

    def bottom_tray_widget(self) -> QWidget | None:
        return None

    def tray_title(self) -> str:
        return ""

    def ribbon_actions(self) -> list[RibbonActionSpec]:
        return [
            RibbonActionSpec("auto_subject", "Auto\nSubject", "face", "Auto-detect main subject (AI)", group="Mask", large=True),
            RibbonActionSpec("eyedropper", "Eyedropper", "eyedropper", "Sample photo hues to isolate", group="Mask", large=True, checkable=True),
            RibbonActionSpec("paint_color", "Paint\nColour", "brush", "Paint to keep colour (follows contours when Smart Edge is on)", group="Mask", large=True, checkable=True, checked=True),
            RibbonActionSpec("erase_color", "Erase\nColour", "eraser", "Paint to convert to B&W (follows contours when Smart Edge is on)", group="Mask", large=True, checkable=True),
            RibbonActionSpec("smart_edge", "Smart\nEdge", "auto_adjust", "Follow object contours with Lab edge-aware brush snapping", group="Mask", large=True, checkable=True, checked=True),
            RibbonActionSpec("invert_mask", "Invert", "invert", "Swap subject and B&W background", group="Mask", large=True),
            RibbonActionSpec("style_red", "Red\nPop", "swatch_red", "Isolate vivid reds & crimsons", group="Quick Styles", large=True, checkable=True),
            RibbonActionSpec("style_blue", "Blue\nPop", "swatch_blue", "Isolate rich blues & skies", group="Quick Styles", large=True, checkable=True),
            RibbonActionSpec("style_green", "Green\nPop", "swatch_green", "Isolate lush greens & foliage", group="Quick Styles", large=True, checkable=True),
            RibbonActionSpec("style_warm", "Warm\nTones", "swatch_warm", "Isolate warm reds, oranges & golden tones", group="Quick Styles", large=True, checkable=True),
            RibbonActionSpec("style_cool", "Cool\nTones", "swatch_cool", "Isolate cool cyans, blues & teals", group="Quick Styles", large=True, checkable=True),
            RibbonActionSpec("show_mask", "Show\nMask", "view_details", "Toggle red mask overlay & contour outlines", group="View", large=True, checkable=True),
            RibbonActionSpec("compare_orig", "Compare\nOriginal", "compare", "Toggle original colour photo preview", group="View", large=True, checkable=True),
            RibbonActionSpec("zoom_fit", "Fit\nWindow", "fit_window", "Zoom to fit (F)", group="View", large=True),
            RibbonActionSpec("zoom_actual", "100%", "actual_size", "Actual pixels (1)", group="View", large=True),
        ]

    def ribbon_action_states(self) -> dict[str, dict[str, Any]]:
        return {
            "eyedropper": {"checked": self.brush_mode == "eyedropper"},
            "paint_color": {"checked": self.brush_mode == "keep_color"},
            "erase_color": {"checked": self.brush_mode == "make_bw"},
            "smart_edge": {"checked": self.smart_brush},
            "style_red": {"checked": self._active_quick_style == "style_red"},
            "style_blue": {"checked": self._active_quick_style == "style_blue"},
            "style_green": {"checked": self._active_quick_style == "style_green"},
            "style_warm": {"checked": self._active_quick_style == "style_warm"},
            "style_cool": {"checked": self._active_quick_style == "style_cool"},
            "show_mask": {"checked": self.show_mask_overlay},
            "compare_orig": {"checked": self._comparing_original},
        }

    def handle_ribbon_action(self, action_id: str) -> None:
        if action_id == "auto_subject":
            self._active_quick_style = None
            self._auto_detect_mask()
        elif action_id == "eyedropper":
            self._active_quick_style = None
            self.btn_pick_color.setChecked(True)
            self._toggle_eyedropper()
        elif action_id == "paint_color":
            self._set_brush_mode("keep_color")
        elif action_id == "erase_color":
            self._set_brush_mode("make_bw")
        elif action_id == "smart_edge":
            self.chk_smart_brush.setChecked(not self.chk_smart_brush.isChecked())
        elif action_id == "invert_mask":
            self._active_quick_style = None
            self._invert_mask()
        elif action_id == "style_red":
            self._active_quick_style = "style_red"
            self._apply_quick_hue_preset([(30, 30, 220), (45, 25, 175)], "Red Pop")
        elif action_id == "style_blue":
            self._active_quick_style = "style_blue"
            self._apply_quick_hue_preset([(220, 95, 25), (210, 150, 50)], "Blue Pop")
        elif action_id == "style_green":
            self._active_quick_style = "style_green"
            self._apply_quick_hue_preset([(45, 195, 45), (55, 150, 75)], "Green Pop")
        elif action_id == "style_warm":
            self._active_quick_style = "style_warm"
            self._apply_quick_hue_preset([(30, 50, 230), (25, 140, 240), (35, 200, 240)], "Warm Tones")
        elif action_id == "style_cool":
            self._active_quick_style = "style_cool"
            self._apply_quick_hue_preset([(230, 120, 30), (210, 180, 45), (190, 95, 115)], "Cool Tones")
        elif action_id == "show_mask":
            self.chk_overlay.setChecked(not self.chk_overlay.isChecked())
        elif action_id == "compare_orig":
            self._set_compare_original(not self._comparing_original)
        else:
            super().handle_ribbon_action(action_id)

    def has_unsaved_changes(self) -> bool:
        return (len(self.undo_stack) > 1 or self._dirty) and self.saved_file_path is None

    def save_result(self) -> Path | None:
        self._save_photo()
        return self.saved_file_path

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 1

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def undo(self) -> None:
        self._undo_mask()

    def redo(self) -> None:
        self._redo_mask()

    def zoom_in(self) -> None:
        self._apply_zoom(1.25)

    def zoom_out(self) -> None:
        self._apply_zoom(1.0 / 1.25)

    def zoom_fit(self) -> None:
        if self.scene.sceneRect().isValid() and self.scene.sceneRect().width() > 0:
            self._is_fit_mode = True
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._on_zoom_changed(self.view.transform().m11())
            self.zoom_bar.set_is_fit(True)

    def zoom_actual(self) -> None:
        if self.scene.sceneRect().isValid():
            self._is_fit_mode = False
            self.view.resetTransform()
            self._on_zoom_changed(1.0)
            self.zoom_bar.set_is_fit(False)

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
        canvas_bg = "#111318" if tok.is_dark else "#232730"
        self.view.setStyleSheet(f"QGraphicsView {{ background-color: {canvas_bg}; border: none; }}")
        self.zoom_bar.apply_theme_mode(self._theme_name)
        self.lbl_status.setStyleSheet(f"color: {tok.text_secondary}; font-size: 9pt; background: transparent;")
        self.lbl_dims.setStyleSheet(f"color: {tok.text_secondary}; font-size: 9pt; background: transparent;")
        self._update_color_swatches_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_canvas_ui(self) -> None:
        self.setObjectName("StudioCanvasRoot")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.view = ColourPopCanvasView(self)
        self.view.setFrameShape(QFrame.Shape.NoFrame)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.view.set_brush_radius(self.brush_radius)
        self.view.brush_applied.connect(self._on_brush_applied)
        self.view.color_picked.connect(self._on_color_picked)
        self.view.stroke_started.connect(self._on_stroke_started)
        self.view.zoom_changed.connect(self._on_zoom_changed)
        self.view.viewport().installEventFilter(self)

        self.bg_item = QGraphicsPixmapItem()
        self.scene.addItem(self.bg_item)
        lay.addWidget(self.view, 1)

        bottom_bar = QFrame(self)
        bottom_bar.setObjectName("StudioTrayRoot")
        b_lay = QHBoxLayout(bottom_bar)
        b_lay.setContentsMargins(10, 2, 10, 2)
        b_lay.setSpacing(10)

        self.lbl_status = QLabel("Paint over photo to keep or erase colour • Eyedropper picks hue", bottom_bar)
        b_lay.addWidget(self.lbl_status, 1)

        h, w = self.original_bgr.shape[:2]
        self.lbl_dims = QLabel(f"{w} × {h} px", bottom_bar)
        b_lay.addWidget(self.lbl_dims)

        self.zoom_bar = CreateStudioZoomBar(bottom_bar)
        self.zoom_bar.fit_clicked.connect(self.zoom_fit)
        self.zoom_bar.actual_size_clicked.connect(self.zoom_actual)
        self.zoom_bar.zoom_in_clicked.connect(self.zoom_in)
        self.zoom_bar.zoom_out_clicked.connect(self.zoom_out)
        self.zoom_bar.zoom_slider_moved.connect(self._on_slider_zoom_moved)
        b_lay.addWidget(self.zoom_bar)
        lay.addWidget(bottom_bar)

    def _build_inspector_ui(self) -> None:
        self._inspector_root = QWidget()
        self._inspector_root.setObjectName("StudioInspectorRoot")
        self._inspector_root.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        vbox = QVBoxLayout(self._inspector_root)
        vbox.setContentsMargins(4, 4, 4, 8)
        vbox.setSpacing(8)

        # 1. Selection Mode card
        grp_mode = StudioCardGroup("Selection Tool", self._inspector_root)
        self.btn_group_mode = QButtonGroup(self)
        self.btn_group_mode.setExclusive(True)

        self.btn_keep = QPushButton("Keep Vibrant Colour (+)", grp_mode)
        self.btn_keep.setIcon(make_wlpg_icon("brush", 16))
        self.btn_keep.setCheckable(True)
        self.btn_keep.setChecked(True)
        self.btn_keep.clicked.connect(lambda: self._set_brush_mode("keep_color"))
        self.btn_group_mode.addButton(self.btn_keep)
        grp_mode.add_widget(self.btn_keep)

        self.btn_bw = QPushButton("Make Black & White (-)", grp_mode)
        self.btn_bw.setIcon(make_wlpg_icon("eraser", 16))
        self.btn_bw.setCheckable(True)
        self.btn_bw.clicked.connect(lambda: self._set_brush_mode("make_bw"))
        self.btn_group_mode.addButton(self.btn_bw)
        grp_mode.add_widget(self.btn_bw)

        self.btn_pick_color = QPushButton("Pick Colour from Photo", grp_mode)
        self.btn_pick_color.setIcon(make_wlpg_icon("eyedropper", 16))
        self.btn_pick_color.setCheckable(True)
        self.btn_pick_color.setToolTip("Click: pick colour • Shift+Click: add (+) • Alt/Ctrl+Click: remove (-)")
        self.btn_pick_color.clicked.connect(self._toggle_eyedropper)
        self.btn_group_mode.addButton(self.btn_pick_color)
        grp_mode.add_widget(self.btn_pick_color)

        self.swatches_container = QWidget(grp_mode)
        self.swatches_layout = QHBoxLayout(self.swatches_container)
        self.swatches_layout.setContentsMargins(2, 4, 2, 2)
        self.swatches_layout.setSpacing(6)
        self.swatches_container.setVisible(False)
        grp_mode.add_widget(self.swatches_container)
        vbox.addWidget(grp_mode)

        # 2. Brush & Mask card
        grp_brush = StudioCardGroup("Brush & Mask", self._inspector_root)
        self.lbl_radius_val = QLabel(f"Brush Size: {self.brush_radius} px", grp_brush)
        grp_brush.add_widget(self.lbl_radius_val)

        self.sld_radius = AeroStudioSlider(8, 120, self.brush_radius, grp_brush)
        self.sld_radius.valueChanged.connect(self._on_radius_changed)
        grp_brush.add_widget(self.sld_radius)

        self.lbl_tol_val = QLabel("Colour Tolerance: 25%", grp_brush)
        grp_brush.add_widget(self.lbl_tol_val)

        self.sld_tol = AeroStudioSlider(5, 80, int(self.color_tolerance * 100), grp_brush)
        self.sld_tol.sliderPressed.connect(lambda: setattr(self, "_is_tol_sliding", True))
        self.sld_tol.sliderReleased.connect(self._on_tolerance_released)
        self.sld_tol.valueChanged.connect(self._on_tolerance_changed)
        grp_brush.add_widget(self.sld_tol)

        self.chk_smart_brush = QCheckBox("Smart Edge Snap (Follow Contours)", grp_brush)
        self.chk_smart_brush.setChecked(True)
        self.chk_smart_brush.setToolTip(
            "Automatically snaps Paint Colour (+) and Erase Colour (-) strokes to object edges & contours"
        )
        self.chk_smart_brush.toggled.connect(self._on_smart_brush_toggled)
        grp_brush.add_widget(self.chk_smart_brush)

        self.chk_overlay = QCheckBox("Show Mask Overlay & Contours", grp_brush)
        self.chk_overlay.toggled.connect(self._on_overlay_toggled)
        grp_brush.add_widget(self.chk_overlay)

        act_row = QHBoxLayout()
        self.btn_invert = QPushButton("Invert", grp_brush)
        self.btn_invert.setIcon(make_wlpg_icon("invert", 14))
        self.btn_invert.clicked.connect(self._invert_mask)
        act_row.addWidget(self.btn_invert)

        self.btn_auto_detect = QPushButton("Auto Subject", grp_brush)
        self.btn_auto_detect.setIcon(make_wlpg_icon("face", 14))
        self.btn_auto_detect.clicked.connect(self._auto_detect_mask)
        act_row.addWidget(self.btn_auto_detect)
        grp_brush.add_layout(act_row)
        vbox.addWidget(grp_brush)

        # 3. B&W Background Tone card
        grp_tone = StudioCardGroup("B&W Background Tone", self._inspector_root)
        self.lbl_contrast_val = QLabel("Contrast: +15%", grp_tone)
        grp_tone.add_widget(self.lbl_contrast_val)
        self.sld_contrast = AeroStudioSlider(80, 180, 115, grp_tone)
        self.sld_contrast.valueChanged.connect(self._on_contrast_changed)
        grp_tone.add_widget(self.sld_contrast)

        self.lbl_bright_val = QLabel("Brightness: 0", grp_tone)
        grp_tone.add_widget(self.lbl_bright_val)
        self.sld_bright = AeroStudioSlider(-40, 40, 0, grp_tone)
        self.sld_bright.valueChanged.connect(self._on_bright_changed)
        grp_tone.add_widget(self.sld_bright)
        vbox.addWidget(grp_tone)

        # 4. History & Compare card
        grp_hist = StudioCardGroup("History & Compare", self._inspector_root)
        h_row = QHBoxLayout()
        self.btn_undo = QPushButton("Undo", grp_hist)
        self.btn_undo.setIcon(make_wlpg_icon("undo", 14))
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self._undo_mask)
        h_row.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("Redo", grp_hist)
        self.btn_redo.setIcon(make_wlpg_icon("redo", 14))
        self.btn_redo.setEnabled(False)
        self.btn_redo.clicked.connect(self._redo_mask)
        h_row.addWidget(self.btn_redo)
        grp_hist.add_layout(h_row)

        self.btn_compare = QPushButton("Hold / Compare Original", grp_hist)
        self.btn_compare.setIcon(make_wlpg_icon("compare", 14))
        self.btn_compare.pressed.connect(lambda: self._set_compare_original(True))
        self.btn_compare.released.connect(lambda: self._set_compare_original(False))
        grp_hist.add_widget(self.btn_compare)
        vbox.addWidget(grp_hist)
        vbox.addStretch(1)

    def _build_bottom_tray_ui(self) -> None:
        self._tray_root = QFrame()
        self._tray_root.setObjectName("StudioTrayRoot")
        lay = QHBoxLayout(self._tray_root)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)

        presets = [
            ("Auto Subject", "auto", None),
            ("Red Pop", "hue", (30, 30, 220)),
            ("Blue Pop", "hue", (220, 90, 25)),
            ("Green Pop", "hue", (45, 195, 45)),
            ("Warm Tones", "hue", (40, 140, 230)),
            ("Cool Tones", "hue", (200, 150, 40)),
        ]
        self._quick_style_btns: list[QPushButton] = []
        for label, kind, color_bgr in presets:
            b = QPushButton(label, self._tray_root)
            b.setMinimumHeight(34)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            if kind == "auto":
                b.setIcon(make_wlpg_icon("face", 16))
                b.clicked.connect(self._auto_detect_mask)
            else:
                b.setIcon(make_wlpg_icon("eyedropper", 16))
                b.clicked.connect(lambda _c=False, c=color_bgr, name=label: self._apply_quick_hue_preset(c, name))
            self._quick_style_btns.append(b)
            lay.addWidget(b)
        lay.addStretch(1)

    # ------------------------------------------------------------------
    # EventFilter & Zoom
    # ------------------------------------------------------------------
    def eventFilter(self, obj: Any, event: QEvent) -> bool:
        if obj is self.view.viewport() and event.type() == QEvent.Type.Resize:
            if self._is_fit_mode and self.scene.sceneRect().isValid() and self.scene.sceneRect().width() > 0:
                self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
                self._on_zoom_changed(self.view.transform().m11())
                self.zoom_bar.set_is_fit(True)
        return super().eventFilter(obj, event)

    def _apply_zoom(self, factor: float) -> None:
        self._is_fit_mode = False
        self.zoom_bar.set_is_fit(False)
        self.view.scale(factor, factor)
        self._on_zoom_changed(self.view.transform().m11())

    def _on_slider_zoom_moved(self, val: int) -> None:
        self._is_fit_mode = False
        self.zoom_bar.set_is_fit(False)
        target_scale = val / 100.0
        current_scale = self.view.transform().m11()
        if abs(current_scale) > 1e-4:
            self.view.scale(target_scale / current_scale, target_scale / current_scale)
            self._on_zoom_changed(target_scale)

    def _on_zoom_changed(self, scale_val: float) -> None:
        self.zoom_bar.update_zoom_display(scale_val)

    # ------------------------------------------------------------------
    # Masking & Colour Selection
    # ------------------------------------------------------------------
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

    def _apply_quick_hue_preset(
        self, bgr: tuple[int, int, int] | list[tuple[int, int, int]], name: str
    ) -> None:
        self._on_stroke_started()
        colors = list(bgr) if isinstance(bgr, list) else [bgr]
        self.picked_colors = colors
        self.last_picked_color = colors[0] if colors else None
        self.btn_pick_color.setChecked(True)
        self.brush_mode = "eyedropper"
        self.view.set_eyedropper_mode(True)
        self._recompute_picked_colors_mask()
        self._update_color_swatches_ui()
        self.lbl_status.setText(f"Applied Quick Style: {name} • Use Tolerance slider or Paint/Erase to refine")
        self.ribbon_state_changed.emit()

    def _toggle_eyedropper(self) -> None:
        self.brush_mode = "eyedropper"
        self.view.set_eyedropper_mode(True)
        self.lbl_status.setText("Eyedropper: Click to pick colour • Shift+Click add • Alt+Click remove")
        self.ribbon_state_changed.emit()

    def _set_brush_mode(self, mode: str) -> None:
        self.brush_mode = mode
        self.view.set_eyedropper_mode(False)
        self.btn_keep.setChecked(mode == "keep_color")
        self.btn_bw.setChecked(mode == "make_bw")
        self.btn_pick_color.setChecked(False)
        self.lbl_status.setText(
            f"Active Brush: {'Keep Vibrant Colour (+)' if mode == 'keep_color' else 'Make Black & White (-)'}"
        )
        self.ribbon_state_changed.emit()

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
            if not any(np.linalg.norm(np.array(c) - np.array(target_bgr)) < 10 for c in self.picked_colors):
                self.picked_colors.append(target_bgr)
            action_desc = "Added colour"
        else:
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
        self.lbl_status.setText(f"{action_desc} • Active colours: {len(self.picked_colors)}")

    def _recompute_picked_colors_mask(self, refine_edges: bool = True) -> None:
        if not self.picked_colors:
            self.proxy_mask = np.zeros(self.proxy_bgr.shape[:2], dtype=np.float32)
        else:
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
            if item.widget():
                item.widget().deleteLater()
        if not self.picked_colors:
            self.swatches_container.setVisible(False)
            return

        tok = get_theme_tokens(self._theme_name)
        self.swatches_container.setVisible(True)
        lbl_info = QLabel(f"Colours ({len(self.picked_colors)}):", self.swatches_container)
        self.swatches_layout.addWidget(lbl_info)

        for idx, (b, g, r) in enumerate(self.picked_colors[:6]):
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            chip = QPushButton("", self.swatches_container)
            chip.setFixedSize(16, 16)
            chip.setToolTip(f"Colour {hex_color.upper()} — click to remove")
            chip.setStyleSheet(
                f"QPushButton {{ background-color: {hex_color}; border: 1.5px solid #FFFFFF; border-radius: 8px; padding: 0; }}"
            )
            chip.clicked.connect(lambda _c=False, i=idx: self._remove_picked_color(i))
            self.swatches_layout.addWidget(chip)

        self.swatches_layout.addStretch(1)
        btn_clear = QPushButton("Clear", self.swatches_container)
        btn_clear.setStyleSheet(
            f"QPushButton {{ font-size: 8.5pt; color: {tok.accent}; background: transparent; border: none; padding: 0 2px; }}"
        )
        btn_clear.clicked.connect(self._clear_picked_colors)
        self.swatches_layout.addWidget(btn_clear)

    def _remove_picked_color(self, idx: int) -> None:
        if 0 <= idx < len(self.picked_colors):
            self.picked_colors.pop(idx)
            self._on_stroke_started()
            self._recompute_picked_colors_mask()
            self._update_color_swatches_ui()

    def _clear_picked_colors(self) -> None:
        self.picked_colors.clear()
        self._on_stroke_started()
        self._recompute_picked_colors_mask()
        self._update_color_swatches_ui()

    def _on_tolerance_released(self) -> None:
        self._is_tol_sliding = False
        self._on_tolerance_changed(self.sld_tol.value())

    def _on_tolerance_changed(self, val: int) -> None:
        self.color_tolerance = val / 100.0
        self.lbl_tol_val.setText(f"Colour Tolerance: {val}%")
        refine = not self._is_tol_sliding
        if self.picked_colors and self.brush_mode == "eyedropper":
            self._recompute_picked_colors_mask(refine_edges=refine)
        elif self.last_picked_color is not None and self.brush_mode == "eyedropper":
            self.proxy_mask = compute_multi_color_similarity_mask(
                self.proxy_bgr,
                [self.last_picked_color],
                tolerance=self.color_tolerance,
                refine_edges=refine,
                lab_img=self._proxy_lab_f32,
            )
            self._fast_refresh_proxy()

    def _on_overlay_toggled(self, checked: bool) -> None:
        self.show_mask_overlay = bool(checked)
        self._fast_refresh_proxy()
        self.ribbon_state_changed.emit()

    def _on_radius_changed(self, val: int) -> None:
        self.brush_radius = val
        self.lbl_radius_val.setText(f"Brush Size: {val} px")
        self.view.set_brush_radius(val)

    def _on_contrast_changed(self, val: int) -> None:
        self.bg_contrast = val / 100.0
        self._dirty = True
        delta = val - 100
        self.lbl_contrast_val.setText(f"Contrast: {'+' if delta >= 0 else ''}{delta}%")
        self._precompute_proxy_bw()
        self._refresh_canvas()

    def _on_bright_changed(self, val: int) -> None:
        self.bg_brightness = float(val)
        self._dirty = True
        self.lbl_bright_val.setText(f"Brightness: {'+' if val > 0 else ''}{val}")
        self._precompute_proxy_bw()
        self._refresh_canvas()

    def _on_stroke_started(self) -> None:
        self._dirty = True
        self.undo_stack.append(self.proxy_mask.copy())
        if len(self.undo_stack) > 25:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._sync_undo_buttons()

    def _sync_undo_buttons(self) -> None:
        can_u = self.can_undo()
        can_r = self.can_redo()
        self.btn_undo.setEnabled(can_u)
        self.btn_redo.setEnabled(can_r)
        self.undo_redo_state_changed.emit(can_u, can_r)
        self.ribbon_state_changed.emit()

    def _on_smart_brush_toggled(self, checked: bool) -> None:
        self.smart_brush = bool(checked)
        self.lbl_status.setText(
            "Smart Edge Snap ON — brush follows object contours"
            if self.smart_brush
            else "Smart Edge Snap OFF — soft freehand brush"
        )
        self.ribbon_state_changed.emit()

    def _on_brush_applied(self, sx: int, sy: int) -> None:
        ph, pw = self.proxy_bgr.shape[:2]
        if 0 <= sx < pw and 0 <= sy < ph:
            self.proxy_mask = refine_mask_brush(
                self.proxy_mask,
                self.proxy_bgr,
                (sx, sy),
                radius=self.brush_radius,
                mode=self.brush_mode,
                smart=self.smart_brush,
                tolerance=self.color_tolerance,
            )
            self._fast_refresh_proxy()

    def _fast_refresh_proxy(self) -> None:
        if self._comparing_original:
            self.bg_item.setPixmap(_cv2_to_qpixmap(self.proxy_bgr))
            return
        m3 = self.proxy_mask[:, :, None]
        blended_f = self._proxy_bgr_f32 * m3 + self._proxy_bw_f32 * (1.0 - m3)
        if self.show_mask_overlay:
            bg_alpha = (1.0 - self.proxy_mask) * 0.40
            inv_alpha = 1.0 - bg_alpha
            blended_f[:, :, 0] *= inv_alpha
            blended_f[:, :, 1] *= inv_alpha
            blended_f[:, :, 2] *= inv_alpha
            blended_f[:, :, 2] += 230.0 * bg_alpha
        np.clip(blended_f, 0.0, 255.0, out=blended_f)
        out_u8 = blended_f.astype(np.uint8)
        if self.show_mask_overlay:
            bin_u8 = (self.proxy_mask >= 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(bin_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(out_u8, contours, -1, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.drawContours(out_u8, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
        self.bg_item.setPixmap(_cv2_to_qpixmap(out_u8))

    def _refresh_canvas(self) -> None:
        self._fast_refresh_proxy()
        ph, pw = self.proxy_bgr.shape[:2]
        self.scene.setSceneRect(QRectF(0, 0, pw, ph))

    def _invert_mask(self) -> None:
        self._on_stroke_started()
        self.proxy_mask = 1.0 - self.proxy_mask
        self._fast_refresh_proxy()
        self.lbl_status.setText("Inverted mask: subject and B&W background swapped.")

    def _auto_detect_mask(self) -> None:
        self._on_stroke_started()
        full_mask = estimate_subject_mask(self.original_bgr)
        self.proxy_mask = cv2.resize(
            full_mask, (self.proxy_bgr.shape[1], self.proxy_bgr.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        self._fast_refresh_proxy()
        self.lbl_status.setText("Re-detected subject segmentation.")

    def _undo_mask(self) -> None:
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.undo_stack.pop())
            self.proxy_mask = self.undo_stack[-1].copy()
            self._fast_refresh_proxy()
            self._sync_undo_buttons()

    def _redo_mask(self) -> None:
        if self.redo_stack:
            restored = self.redo_stack.pop()
            self.undo_stack.append(restored)
            self.proxy_mask = restored.copy()
            self._fast_refresh_proxy()
            self._sync_undo_buttons()

    def _set_compare_original(self, comparing: bool) -> None:
        self._comparing_original = bool(comparing)
        self._fast_refresh_proxy()
        self.lbl_status.setText("Previewing original photo" if comparing else "Previewing Colour Pop")
        self.ribbon_state_changed.emit()

    def _save_photo(self) -> None:
        out_path = prompt_save_new_file(self, self.file_path, "colourPop", ".jpg")
        if not out_path:
            return
        self.lbl_status.setText("Rendering high-resolution Colour Pop photo...")
        QApplication.processEvents()
        h, w = self.original_bgr.shape[:2]
        full_mask = (
            cv2.resize(self.proxy_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            if self.proxy_scale != 1.0
            else self.proxy_mask.copy()
        )
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
            self._dirty = False
            self.saved.emit(out_path)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save photo:\n{e}")
