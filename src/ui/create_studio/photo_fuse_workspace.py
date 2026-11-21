"""WLPG-faithful Photo Fuse Workspace integrated into MainWindow."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.imaging.create_tools import (
    align_burst_frame,
    detect_faces_with_landmarks,
    fill_warped_border_slivers,
    fuse_face,
)
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
from src.ui.create_studio_dialogs import _InteractiveFaceItem, _PhotoFuseWorker
from src.ui.ribbon_toolbar import make_wlpg_icon
from src.ui.theme_tokens import get_theme_tokens


class _FaceCandidatePopover(QFrame):
    """WLPG-style floating on-canvas 'Which do you like best?' candidate picker."""

    shot_chosen = Signal(int)
    closed = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("PhotoFuseCandidatePopover")
        self._theme_name = "wlpg_aero"
        self.setMinimumWidth(156)
        self.setMaximumWidth(238)
        self.setMaximumHeight(340)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(6, 5, 6, 6)
        vbox.setSpacing(5)

        hdr = QHBoxLayout()
        hdr.setContentsMargins(2, 0, 0, 0)
        hdr.setSpacing(4)
        self.lbl_title = QLabel("Which do you like best?", self)
        self.lbl_title.setFont(QFont("Segoe UI", 8.5, QFont.DemiBold))
        hdr.addWidget(self.lbl_title, 1)

        self.btn_close = QPushButton("✕", self)
        self.btn_close.setObjectName("PopoverCloseBtn")
        self.btn_close.setFixedSize(18, 18)
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.setToolTip("Close picker")
        self.btn_close.clicked.connect(self._on_close)
        hdr.addWidget(self.btn_close)
        vbox.addLayout(hdr)

        self.scroll = QScrollArea(self)
        self.scroll.setObjectName("PopoverScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.grid_host = QWidget(self.scroll)
        self.grid_layout = QGridLayout(self.grid_host)
        self.grid_layout.setContentsMargins(2, 2, 2, 2)
        self.grid_layout.setSpacing(6)
        self.scroll.setWidget(self.grid_host)
        vbox.addWidget(self.scroll, 1)

        self.apply_theme_mode("wlpg_aero")
        self.hide()

    def _on_close(self) -> None:
        self.hide()
        self.closed.emit()

    def set_candidates(self, candidates: list[dict[str, Any]], theme_name: str) -> None:
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

        tok = get_theme_tokens(theme_name)
        cols = 2 if len(candidates) > 3 else 1
        for i, cand in enumerate(candidates):
            btn = QToolButton(self.grid_host)
            btn.setObjectName("PopoverCandBtn")
            btn.setIcon(QIcon(cand["pixmap"]))
            btn.setIconSize(QSize(72, 72))
            btn.setText(str(cand["label"]))
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            btn.setFixedSize(92 if cols == 2 else 116, 98)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            if cand.get("tooltip"):
                btn.setToolTip(str(cand["tooltip"]))

            is_active = bool(cand.get("is_active"))
            is_unaligned = bool(cand.get("is_unaligned"))
            if is_active:
                btn.setStyleSheet(
                    f"QToolButton {{ border: 2px solid {tok.accent}; background: {tok.selection_bg}; "
                    f"color: {tok.selection_text}; border-radius: 4px; padding: 3px; font-size: 8pt; font-weight: 600; }}"
                )
            elif is_unaligned:
                btn.setStyleSheet(
                    f"QToolButton {{ border: 1.5px solid {tok.warning}; background: {tok.bg_card}; "
                    f"color: {tok.text_primary}; border-radius: 4px; padding: 3px; font-size: 8pt; }}"
                )
            else:
                btn.setStyleSheet(
                    f"QToolButton {{ border: 1px solid {tok.border_subtle}; background: {tok.bg_input}; "
                    f"color: {tok.text_primary}; border-radius: 4px; padding: 3px; font-size: 8pt; }}"
                    f"QToolButton:hover {{ border: 1.5px solid {tok.accent}; background: {tok.bg_hover}; }}"
                )
            shot_idx = int(cand["shot_idx"])
            btn.clicked.connect(lambda _c=False, s=shot_idx: self.shot_chosen.emit(s))
            self.grid_layout.addWidget(btn, i // cols, i % cols)

        rows = max(1, (len(candidates) + cols - 1) // cols)
        w = 216 if cols == 2 else 164
        h = min(338, 42 + rows * 104)
        self.resize(w, h)
        self.apply_theme_mode(theme_name)

    def apply_theme_mode(self, theme_name: str) -> None:
        self._theme_name = theme_name or "wlpg_aero"
        tok = get_theme_tokens(self._theme_name)
        self.setStyleSheet(
            f"""
            QFrame#PhotoFuseCandidatePopover {{
                background-color: {tok.bg_elevated};
                border: 1.5px solid {tok.accent};
                border-radius: 6px;
            }}
            QFrame#PhotoFuseCandidatePopover QLabel {{
                color: {tok.text_primary};
                background: transparent;
            }}
            QScrollArea#PopoverScroll, QScrollArea#PopoverScroll > QWidget > QWidget {{
                background: transparent;
                border: none;
            }}
            QPushButton#PopoverCloseBtn {{
                border: none;
                background: transparent;
                color: {tok.text_secondary};
                font-weight: bold;
                border-radius: 3px;
            }}
            QPushButton#PopoverCloseBtn:hover {{
                background: {tok.danger};
                color: {tok.text_on_accent};
            }}
            """
        )


class PhotoFuseWorkspace(CreateWorkspaceBase):
    """Integrated WLPG-style Photo Fuse Best-Face Group Shot Combiner workspace."""

    workspace_id = "photo_fuse"
    display_title = "Photo Fuse"
    tool_save_label = "fused photo"

    def __init__(self, photo_paths: list[Path], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.photo_paths: list[Path] = [Path(p) for p in photo_paths]
        self.output_dir = self.photo_paths[0].parent if self.photo_paths else Path.home()

        self.frames_bgr: list[np.ndarray] = []
        self.aligned_frames_bgr: list[np.ndarray] = []
        self.alignment_ok: list[bool] = []
        self.base_fused_bgr: np.ndarray | None = None
        self.base_shot_idx: int = 0
        self.face_boxes: list[tuple[int, int, int, int]] = []
        self.face_kps: list[np.ndarray | None] = []
        self.face_is_custom: list[bool] = []
        self.face_applied_shot: dict[int, int] = {}
        self._src_kps_cache: dict[tuple[int, int], np.ndarray | None] = {}
        self._base_cache: dict[int, tuple[list[np.ndarray], list[bool], list[dict[str, Any]]]] = {}
        self._base_buttons: list[QToolButton] = []
        self.selected_face_idx: int = -1
        self._showing_original: bool = False
        self._show_face_boxes: bool = True
        self._add_region_mode: bool = False
        self._region_drag_start: QPointF | None = None
        self._rubber_rect_item: QGraphicsRectItem | None = None
        self._is_fit_mode: bool = True
        self._worker: _PhotoFuseWorker | None = None

        self._build_canvas_ui()
        self._build_inspector_ui()
        self._build_bottom_tray_ui()
        self.apply_theme_mode("wlpg_aero")
        self._align_and_detect()

    # ------------------------------------------------------------------
    # Workspace Contract
    # ------------------------------------------------------------------
    def inspector_widget(self) -> QWidget:
        return self._inspector_root

    def bottom_tray_widget(self) -> QWidget:
        return self._tray_root

    def tray_title(self) -> str:
        return "Source Shots (Pick Base Shot)"

    def ribbon_actions(self) -> list[RibbonActionSpec]:
        return [
            RibbonActionSpec(
                "add_region",
                "Add\nRegion",
                "add_region",
                "Draw a custom rectangle anywhere on the photo to replace",
                group="Tools",
                large=True,
                checkable=True,
                checked=False,
            ),
            RibbonActionSpec(
                "show_boxes",
                "Show\nBoxes",
                "face",
                "Show or hide face and region outlines",
                group="Tools",
                large=True,
                checkable=True,
                checked=True,
            ),
            RibbonActionSpec(
                "compare_orig",
                "Compare\nOriginal",
                "compare",
                "Toggle between fused result and unmodified base shot",
                group="Tools",
                large=True,
                checkable=True,
                checked=False,
            ),
            RibbonActionSpec(
                "reset_base",
                "Reset",
                "revert",
                "Discard all swaps and reset to the active Base Shot",
                group="Tools",
                large=True,
                enabled=False,
            ),
            RibbonActionSpec("zoom_fit", "Fit\nWindow", "fit_window", "Zoom to fit (F)", group="View", large=True),
            RibbonActionSpec("zoom_actual", "100%", "actual_size", "Actual pixels (1)", group="View", large=True),
        ]

    def ribbon_action_states(self) -> dict[str, dict[str, Any]]:
        ready = self.base_fused_bgr is not None and not (self._worker is not None and self._worker.isRunning())
        return {
            "create_ws_save": {"enabled": ready},
            "add_region": {"enabled": ready, "checked": self._add_region_mode},
            "show_boxes": {"enabled": ready, "checked": self._show_face_boxes},
            "compare_orig": {"enabled": ready, "checked": self._showing_original},
            "reset_base": {"enabled": ready and bool(self.face_applied_shot)},
        }

    def handle_ribbon_action(self, action_id: str) -> None:
        if action_id == "add_region":
            self._set_add_region_mode(not self._add_region_mode)
        elif action_id == "show_boxes":
            self.chk_show_boxes.setChecked(not self.chk_show_boxes.isChecked())
        elif action_id == "compare_orig":
            self.btn_compare_orig.setChecked(not self.btn_compare_orig.isChecked())
        elif action_id == "reset_base":
            self._reset_to_base()
        else:
            super().handle_ribbon_action(action_id)

    def has_unsaved_changes(self) -> bool:
        return bool(self.face_applied_shot) and self.saved_file_path is None

    def save_result(self) -> Path | None:
        self._save_fused_photo()
        return self.saved_file_path

    def zoom_in(self) -> None:
        self._apply_zoom(1.25)

    def zoom_out(self) -> None:
        self._apply_zoom(1.0 / 1.25)

    def zoom_fit(self) -> None:
        if hasattr(self, "scene") and self.scene.sceneRect().isValid() and self.scene.sceneRect().width() > 0:
            self._is_fit_mode = True
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._on_zoom_changed(self.view.transform().m11())
            self.zoom_bar.set_is_fit(True)
            self._reposition_popover()

    def zoom_actual(self) -> None:
        if hasattr(self, "scene") and self.scene.sceneRect().isValid():
            self._is_fit_mode = False
            self.view.resetTransform()
            self._on_zoom_changed(1.0)
            self.zoom_bar.set_is_fit(False)
            self._reposition_popover()

    def handle_key_press(self, event: QKeyEvent) -> bool:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat() and self.base_fused_bgr is not None:
            self.btn_compare_orig.setChecked(not self.btn_compare_orig.isChecked())
            return True
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
        self.view.apply_theme_mode(self._theme_name)
        self.zoom_bar.apply_theme_mode(self._theme_name)
        self.candidate_popover.apply_theme_mode(self._theme_name)
        self.lbl_align_warning.setStyleSheet(
            f"background: {'#451A03' if tok.is_dark else '#FEF3C7'}; "
            f"color: {'#FDE68A' if tok.is_dark else '#92400E'}; "
            f"border: 1px solid {tok.warning}; border-radius: 4px; padding: 6px 10px; "
            "font-size: 9pt; font-weight: 600;"
        )
        self.lbl_preview_mode.setStyleSheet(
            f"background: {tok.accent}; color: {tok.text_on_accent}; "
            "border-radius: 4px; padding: 4px 10px; font-size: 9pt; font-weight: 600;"
        )
        self.lbl_status.setStyleSheet(f"color: {tok.text_secondary}; font-size: 9pt; background: transparent;")
        self._update_base_strip_styles()
        if self.candidate_popover.isVisible() and self.selected_face_idx >= 0:
            self._show_candidate_popover(self.selected_face_idx)

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

        self.lbl_align_warning = QLabel("", self)
        self.lbl_align_warning.setWordWrap(True)
        self.lbl_align_warning.setVisible(False)
        lay.addWidget(self.lbl_align_warning)

        self.lbl_preview_mode = QLabel("", self)
        self.lbl_preview_mode.setVisible(False)
        self.lbl_preview_mode.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.lbl_preview_mode)

        self.view = ZoomablePreviewCanvas(self)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.bg_item = QGraphicsPixmapItem()
        self.scene.addItem(self.bg_item)
        self.view.zoom_changed.connect(self._on_zoom_changed)
        self.view.viewport().installEventFilter(self)
        lay.addWidget(self.view, 1)

        # Floating WLPG-style 'Which do you like best?' popover anchored inside viewport
        self.candidate_popover = _FaceCandidatePopover(self.view.viewport())
        self.candidate_popover.shot_chosen.connect(self._apply_face_fuse)

        # Bottom canvas bar: progress + status + ZoomBar
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

        self.lbl_status = QLabel("Loading & aligning burst photos...", bottom_bar)
        b_lay.addWidget(self.lbl_status, 1)

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

        # 1. View & Regions card
        grp_actions = StudioCardGroup("View & Regions", self._inspector_root)
        self.lbl_fuse_summary = QLabel(f"Processing {len(self.photo_paths)} shots...", grp_actions)
        self.lbl_fuse_summary.setProperty("uiRole", "muted")
        self.lbl_fuse_summary.setWordWrap(True)
        grp_actions.add_widget(self.lbl_fuse_summary)

        self.chk_show_boxes = QCheckBox("Show face boxes", grp_actions)
        self.chk_show_boxes.setChecked(True)
        self.chk_show_boxes.toggled.connect(self._on_show_boxes_toggled)
        grp_actions.add_widget(self.chk_show_boxes)

        self.btn_add_region = QPushButton("Add Custom Region", grp_actions)
        self.btn_add_region.setIcon(make_wlpg_icon("add_region", 16))
        self.btn_add_region.setCheckable(True)
        self.btn_add_region.setToolTip("Drag a rectangle anywhere on the photo to swap")
        self.btn_add_region.clicked.connect(lambda chk: self._set_add_region_mode(chk))
        grp_actions.add_widget(self.btn_add_region)

        self.btn_compare_orig = QPushButton("Compare Original", grp_actions)
        self.btn_compare_orig.setIcon(make_wlpg_icon("compare", 16))
        self.btn_compare_orig.setCheckable(True)
        self.btn_compare_orig.toggled.connect(self._toggle_compare_original)
        grp_actions.add_widget(self.btn_compare_orig)

        self.btn_reset = QPushButton("Reset to Base Shot 1", grp_actions)
        self.btn_reset.setIcon(make_wlpg_icon("revert", 16))
        self.btn_reset.clicked.connect(self._reset_to_base)
        grp_actions.add_widget(self.btn_reset)
        vbox.addWidget(grp_actions)

        # 2. Detected People & Custom Regions card
        grp_people = StudioCardGroup("Detected Faces & Regions", self._inspector_root)
        self.list_regions = QListWidget(grp_people)
        self.list_regions.setMinimumHeight(130)
        self.list_regions.itemClicked.connect(self._on_region_list_clicked)
        grp_people.add_widget(self.list_regions)

        self.btn_dismiss_region = QPushButton("Dismiss Selected Box", grp_people)
        self.btn_dismiss_region.setIcon(make_wlpg_icon("cancel", 14))
        self.btn_dismiss_region.clicked.connect(lambda: self._dismiss_face(self.selected_face_idx))
        grp_people.add_widget(self.btn_dismiss_region)
        vbox.addWidget(grp_people)

        # 3. Tips card
        grp_tips = StudioCardGroup("Tips", self._inspector_root)
        lbl_tips = QLabel(
            "• Click any face box on the photo to open 'Which do you like best?' right beside it.\n"
            "• Use Add Custom Region to blend any non-face area (e.g. a hand or object).\n"
            "• Pick a different Base Shot from the bottom strip anytime.\n"
            "• Right-click a box to dismiss a false detection.",
            grp_tips,
        )
        lbl_tips.setProperty("uiRole", "muted")
        lbl_tips.setWordWrap(True)
        grp_tips.add_widget(lbl_tips)
        vbox.addWidget(grp_tips)
        vbox.addStretch(1)

    def _build_bottom_tray_ui(self) -> None:
        self._tray_root = QFrame()
        self._tray_root.setObjectName("StudioTrayRoot")
        lay = QHBoxLayout(self._tray_root)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(8)

        self.base_strip_scroll = QScrollArea(self._tray_root)
        self.base_strip_scroll.setObjectName("StudioTrayScroll")
        self.base_strip_scroll.setWidgetResizable(True)
        self.base_strip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.base_strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.base_strip_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.base_strip_widget = QWidget(self.base_strip_scroll)
        self.base_strip_widget.setObjectName("StudioTrayHost")
        self.base_strip_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.base_strip_layout = QHBoxLayout(self.base_strip_widget)
        self.base_strip_layout.setContentsMargins(2, 2, 2, 2)
        self.base_strip_layout.setSpacing(8)
        self.base_strip_layout.addStretch(1)
        self.base_strip_scroll.setWidget(self.base_strip_widget)
        lay.addWidget(self.base_strip_scroll, 1)

        self.btn_add_shots = QPushButton("+ Add Shots...", self._tray_root)
        self.btn_add_shots.setIcon(make_wlpg_icon("folder_add", 16))
        self.btn_add_shots.setFixedHeight(32)
        self.btn_add_shots.setToolTip("Add more burst photos (up to 6 shots total)")
        self.btn_add_shots.clicked.connect(self._on_add_shots_clicked)
        lay.addWidget(self.btn_add_shots)

    # ------------------------------------------------------------------
    # Custom Region Draw EventFilter + Resize Sync
    # ------------------------------------------------------------------
    def eventFilter(self, watched: Any, event: QEvent) -> bool:
        if watched is self.view.viewport():
            if self._add_region_mode:
                if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                    self._region_drag_start = self.view.mapToScene(event.pos())
                    if self._rubber_rect_item is not None and self._rubber_rect_item.scene() is self.scene:
                        self.scene.removeItem(self._rubber_rect_item)
                    pen = QPen(QColor("#0078D7"), 2.0, Qt.PenStyle.DashLine)
                    pen.setCosmetic(True)
                    self._rubber_rect_item = self.scene.addRect(QRectF(self._region_drag_start, QSize(1, 1)), pen)
                    return True
                if event.type() == QEvent.Type.MouseMove and self._region_drag_start is not None:
                    cur = self.view.mapToScene(event.pos())
                    if self._rubber_rect_item is not None:
                        self._rubber_rect_item.setRect(QRectF(self._region_drag_start, cur).normalized())
                    return True
                if event.type() == QEvent.Type.MouseButtonRelease and self._region_drag_start is not None:
                    end_pt = self.view.mapToScene(event.pos())
                    rect = QRectF(self._region_drag_start, end_pt).normalized()
                    self._region_drag_start = None
                    if self._rubber_rect_item is not None and self._rubber_rect_item.scene() is self.scene:
                        self.scene.removeItem(self._rubber_rect_item)
                    self._rubber_rect_item = None
                    self._finish_custom_region_draw(rect)
                    return True
            if event.type() == QEvent.Type.Resize:
                if self._is_fit_mode and self.scene.sceneRect().isValid() and self.scene.sceneRect().width() > 0:
                    self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
                    self._on_zoom_changed(self.view.transform().m11())
                    self.zoom_bar.set_is_fit(True)
                self._reposition_popover()
        return super().eventFilter(watched, event)

    def _set_add_region_mode(self, enabled: bool) -> None:
        self._add_region_mode = bool(enabled)
        self.btn_add_region.blockSignals(True)
        self.btn_add_region.setChecked(self._add_region_mode)
        self.btn_add_region.blockSignals(False)
        if self._add_region_mode:
            self.candidate_popover.hide()
            self.view.setDragMode(self.view.DragMode.NoDrag)
            self.view.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self.lbl_status.setText("Drag a rectangle over any area you want to replace...")
        else:
            self.view.setDragMode(self.view.DragMode.ScrollHandDrag)
            self.view.viewport().unsetCursor()
        self.ribbon_state_changed.emit()

    def _finish_custom_region_draw(self, rect: QRectF) -> None:
        self._set_add_region_mode(False)
        if self.base_fused_bgr is None:
            return
        h, w = self.base_fused_bgr.shape[:2]
        clamped = rect.intersected(QRectF(0, 0, w, h))
        if clamped.width() < 16 or clamped.height() < 16:
            self.lbl_status.setText("Region too small — drag a larger rectangle to add a custom region.")
            return
        box = (int(round(clamped.x())), int(round(clamped.y())), int(round(clamped.width())), int(round(clamped.height())))
        self.face_boxes.append(box)
        self.face_kps.append(None)
        self.face_is_custom.append(True)
        if not self._show_face_boxes:
            self.chk_show_boxes.setChecked(True)
        new_idx = len(self.face_boxes) - 1
        self._refresh_region_list()
        self._select_face(new_idx, open_popover=True)
        self._update_summary_label()
        self.lbl_status.setText(f"Added custom Region {new_idx + 1}. Choose a shot from the popup.")

    # ------------------------------------------------------------------
    # Alignment & Base Shot Management
    # ------------------------------------------------------------------
    def _align_and_detect(self) -> None:
        if len(self.photo_paths) < 2:
            self.lbl_status.setText("Need at least 2 photos to fuse.")
            return
        self._rebuild_from_base(0)

    def _on_add_shots_clicked(self) -> None:
        if len(self.photo_paths) >= 6:
            QMessageBox.information(self, "Photo Fuse", "Photo Fuse supports up to 6 burst shots.")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Burst Shots",
            str(self.output_dir),
            "Images (*.jpg *.jpeg *.png *.webp *.tif *.tiff)",
        )
        added = 0
        existing = {str(p.resolve()) for p in self.photo_paths if p.exists()}
        for f in files:
            p = Path(f)
            if str(p.resolve()) not in existing and len(self.photo_paths) < 6:
                self.photo_paths.append(p)
                img = load_bgr_exif_safe(p)
                if img is not None:
                    self.frames_bgr.append(img)
                existing.add(str(p.resolve()))
                added += 1
        if added > 0:
            self._base_cache.clear()
            self._src_kps_cache.clear()
            self._rebuild_from_base(self.base_shot_idx)

    def _populate_base_strip(self) -> None:
        while self.base_strip_layout.count() > 1:
            child = self.base_strip_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self._base_buttons.clear()

        for idx, frame in enumerate(self.frames_bgr):
            h, w = frame.shape[:2]
            scale = 54.0 / max(1, max(h, w))
            tw = max(16, int(round(w * scale)))
            th = max(16, int(round(h * scale)))
            thumb_bgr = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
            pm = _cv2_to_qpixmap(thumb_bgr)

            btn = QToolButton(self.base_strip_widget)
            btn.setIcon(QIcon(pm))
            btn.setIconSize(QSize(56, 56))
            btn.setText(f"Shot {idx + 1}" + (" (Base)" if idx == self.base_shot_idx else ""))
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            btn.setFixedSize(82, 86)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            name = self.photo_paths[idx].name if idx < len(self.photo_paths) else f"Shot {idx + 1}"
            btn.setToolTip(f"Use Shot {idx + 1} ({name}) as Base Photo")
            btn.clicked.connect(lambda _c=False, i=idx: self._on_base_shot_selected(i))
            self._base_buttons.append(btn)
            self.base_strip_layout.insertWidget(self.base_strip_layout.count() - 1, btn)

        self._update_base_strip_styles()

    def _update_base_strip_styles(self) -> None:
        tok = get_theme_tokens(self._theme_name)
        for idx, btn in enumerate(self._base_buttons):
            is_cur = idx == self.base_shot_idx
            btn.setText(f"Shot {idx + 1}" + (" (Base)" if is_cur else ""))
            if is_cur:
                btn.setStyleSheet(
                    f"QToolButton {{ border: 2px solid {tok.accent}; background: {tok.selection_bg}; "
                    f"color: {tok.selection_text}; border-radius: 5px; padding: 3px; font-size: 8.5pt; font-weight: 600; }}"
                )
            else:
                btn.setStyleSheet(
                    f"QToolButton {{ border: 1px solid {tok.border_subtle}; background: {tok.bg_elevated}; "
                    f"color: {tok.text_primary}; border-radius: 5px; padding: 3px; font-size: 8.5pt; }}"
                    f"QToolButton:hover {{ border: 1.5px solid {tok.accent}; background: {tok.bg_hover}; }}"
                )

    def _on_base_shot_selected(self, base_idx: int) -> None:
        if base_idx < 0 or base_idx >= len(self.frames_bgr):
            return
        if base_idx == self.base_shot_idx and self.aligned_frames_bgr:
            return
        self._rebuild_from_base(base_idx)

    def _rebuild_from_base(self, base_idx: int) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        self.base_shot_idx = base_idx
        self.face_applied_shot.clear()
        self._src_kps_cache.clear()
        self._showing_original = False
        self.candidate_popover.hide()
        self.lbl_preview_mode.setVisible(False)
        self.btn_compare_orig.blockSignals(True)
        self.btn_compare_orig.setChecked(False)
        self.btn_compare_orig.blockSignals(False)
        self._update_base_strip_styles()

        if base_idx in self._base_cache and self.frames_bgr:
            aligned_frames, align_ok, detections = self._base_cache[base_idx]
            self._apply_base_results(base_idx, self.frames_bgr, aligned_frames, align_ok, detections)
            return

        self.aligned_frames_bgr = []
        self.alignment_ok = []
        self.progress_bar.setVisible(True)
        for b in self._base_buttons:
            b.setEnabled(False)
        self.btn_reset.setEnabled(False)
        self.lbl_status.setText("Loading & aligning burst photos...")
        self.ribbon_state_changed.emit()

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
        self.ribbon_state_changed.emit()

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
        self.face_is_custom = [False] * len(self.face_boxes)

        if not self.face_boxes:
            h, w = base.shape[:2]
            cx, cy = w // 2, h // 2
            fw, fh = int(w * 0.2), int(h * 0.25)
            self.face_boxes = [(cx - fw // 2, cy - fh // 2, fw, fh)]
            self.face_kps = [None]
            self.face_is_custom = [True]

        failed_shots = [f"Shot {i + 1}" for i, ok in enumerate(align_ok) if i != base_idx and not ok]
        other_count = max(0, len(frames) - 1)
        if failed_shots:
            if len(failed_shots) == other_count:
                self.lbl_align_warning.setText(
                    "Could not align the selected photos. Photo Fuse works best with burst or tripod shots of the same scene."
                )
            else:
                self.lbl_align_warning.setText(
                    f"Could not align {', '.join(failed_shots)} to Shot {base_idx + 1}. Swapping from unaligned shots may drift."
                )
            self.lbl_align_warning.setVisible(True)
        else:
            self.lbl_align_warning.setVisible(False)

        self.btn_reset.setText(f"Reset to Base Shot {base_idx + 1}")
        self.lbl_status.setText(f"Base: Shot {base_idx + 1} — Click any box to pick the best expression")
        self._update_summary_label()
        self._refresh_region_list()

        self.selected_face_idx = 0 if self.face_boxes else -1
        self._update_scene()
        self.zoom_fit()
        self.ribbon_state_changed.emit()

    def _update_summary_label(self) -> None:
        self.lbl_fuse_summary.setText(
            f"Base: Shot {self.base_shot_idx + 1} • {len(self.frames_bgr)} shots • {len(self.face_boxes)} box(es)"
        )

    def _refresh_region_list(self) -> None:
        self.list_regions.blockSignals(True)
        self.list_regions.clear()
        for idx, _box in enumerate(self.face_boxes):
            is_custom = idx < len(self.face_is_custom) and self.face_is_custom[idx]
            prefix = "Region" if is_custom else "Person"
            swapped = self.face_applied_shot.get(idx, self.base_shot_idx)
            suffix = f"Shot {swapped + 1} ✓" if swapped != self.base_shot_idx else "Base"
            it = QListWidgetItem(f"{prefix} {idx + 1} — {suffix}")
            self.list_regions.addItem(it)
            if idx == self.selected_face_idx:
                self.list_regions.setCurrentRow(idx)
        self.list_regions.blockSignals(False)
        self.btn_dismiss_region.setEnabled(len(self.face_boxes) > 1 and self.selected_face_idx >= 0)

    def _on_region_list_clicked(self, item: QListWidgetItem) -> None:
        row = self.list_regions.row(item)
        if row >= 0:
            self._select_face(row, open_popover=True)

    # ------------------------------------------------------------------
    # Zoom & Scene Rendering
    # ------------------------------------------------------------------
    def _apply_zoom(self, factor: float) -> None:
        self._is_fit_mode = False
        self.zoom_bar.set_is_fit(False)
        self.view.scale(factor, factor)
        self._on_zoom_changed(self.view.transform().m11())
        self._reposition_popover()

    def _on_slider_zoom_moved(self, pct: int) -> None:
        self._is_fit_mode = False
        self.zoom_bar.set_is_fit(False)
        target_scale = max(0.1, min(5.0, pct / 100.0))
        current_scale = self.view.transform().m11()
        if abs(current_scale) > 1e-4:
            self.view.scale(target_scale / current_scale, target_scale / current_scale)
        self._on_zoom_changed(target_scale)
        self._reposition_popover()

    def _on_zoom_changed(self, scale_val: float) -> None:
        self.zoom_bar.update_zoom_display(scale_val)
        self._reposition_popover()

    def _on_show_boxes_toggled(self, checked: bool) -> None:
        self._show_face_boxes = bool(checked)
        if not self._show_face_boxes:
            self.candidate_popover.hide()
        self._update_scene()
        self.ribbon_state_changed.emit()

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

        for item in list(self.scene.items()):
            if isinstance(item, _InteractiveFaceItem):
                self.scene.removeItem(item)

        if self._show_face_boxes and not self._showing_original:
            for i, (fx, fy, fw, fh) in enumerate(self.face_boxes):
                is_sel = i == self.selected_face_idx
                is_swapped = i in self.face_applied_shot and self.face_applied_shot[i] != self.base_shot_idx
                rect_item = _InteractiveFaceItem(
                    QRectF(fx, fy, fw, fh),
                    i,
                    lambda idx: self._select_face(idx, open_popover=True),
                    is_selected=is_sel,
                    is_swapped=is_swapped,
                    on_dismiss=self._dismiss_face,
                )
                self.scene.addItem(rect_item)

    # ------------------------------------------------------------------
    # Interactive Face / Region Selection & Popover Picker
    # ------------------------------------------------------------------
    def _select_face(self, face_idx: int, *, open_popover: bool = True) -> None:
        if face_idx < 0 or face_idx >= len(self.face_boxes):
            return
        self.selected_face_idx = face_idx
        self._refresh_region_list()
        self._update_scene()
        if open_popover and not self._showing_original:
            self._show_candidate_popover(face_idx)

    def _show_candidate_popover(self, face_idx: int) -> None:
        if face_idx < 0 or face_idx >= len(self.face_boxes) or not self.aligned_frames_bgr:
            self.candidate_popover.hide()
            return
        fx, fy, fw, fh = self.face_boxes[face_idx]
        active_shot = self.face_applied_shot.get(face_idx, self.base_shot_idx)

        candidates: list[dict[str, Any]] = []
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
            crop_thumb = cv2.resize(crop_bgr, (72, 72), interpolation=cv2.INTER_AREA)
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
            candidates.append({
                "shot_idx": shot_idx,
                "pixmap": pm_thumb,
                "label": label,
                "is_active": is_active,
                "is_unaligned": (not is_aligned and not is_base),
                "tooltip": (
                    f"Shot {shot_idx + 1} could not align cleanly to Shot {self.base_shot_idx + 1}"
                    if (not is_aligned and not is_base)
                    else f"Use expression from Shot {shot_idx + 1}"
                ),
            })

        if not candidates:
            self.candidate_popover.hide()
            return
        self.candidate_popover.set_candidates(candidates, self._theme_name)
        self.candidate_popover.show()
        self.candidate_popover.raise_()
        self._reposition_popover()

    def _reposition_popover(self) -> None:
        if not self.candidate_popover.isVisible() or self.selected_face_idx < 0 or self.selected_face_idx >= len(self.face_boxes):
            return
        vp = self.view.viewport()
        if vp is None or vp.width() <= 0 or vp.height() <= 0:
            return
        fx, fy, fw, fh = self.face_boxes[self.selected_face_idx]
        tr_vp = self.view.mapFromScene(QPointF(fx + fw, fy))
        tl_vp = self.view.mapFromScene(QPointF(fx, fy))

        pw = self.candidate_popover.width()
        ph = self.candidate_popover.height()
        margin = 10

        # Prefer right side of box; fall back to left side if clipped
        px = tr_vp.x() + margin
        if px + pw > vp.width() - margin:
            px = tl_vp.x() - pw - margin
        px = max(margin, min(px, max(margin, vp.width() - pw - margin)))

        py = max(margin, min(tr_vp.y(), max(margin, vp.height() - ph - margin)))
        self.candidate_popover.move(QPoint(px, py))

    def _dismiss_face(self, face_idx: int) -> None:
        if len(self.face_boxes) <= 1 or face_idx < 0 or face_idx >= len(self.face_boxes):
            return
        self.face_boxes.pop(face_idx)
        if face_idx < len(self.face_kps):
            self.face_kps.pop(face_idx)
        if face_idx < len(self.face_is_custom):
            self.face_is_custom.pop(face_idx)

        new_applied: dict[int, int] = {}
        for k, v in self.face_applied_shot.items():
            if k < face_idx:
                new_applied[k] = v
            elif k > face_idx:
                new_applied[k - 1] = v
        self.face_applied_shot = new_applied
        self._src_kps_cache.clear()
        self.candidate_popover.hide()

        if self.selected_face_idx >= len(self.face_boxes):
            self.selected_face_idx = len(self.face_boxes) - 1
        elif self.selected_face_idx == face_idx:
            self.selected_face_idx = min(face_idx, len(self.face_boxes) - 1)

        self._recompute_composite()
        self._update_scene()
        self._refresh_region_list()
        self._update_summary_label()
        self.lbl_status.setText("Removed selected box.")
        self.ribbon_state_changed.emit()

    def _toggle_compare_original(self, checked: bool) -> None:
        self._showing_original = bool(checked)
        self.btn_compare_orig.setText("Viewing Original Base" if checked else "Compare Original")
        if checked:
            self.candidate_popover.hide()
            self.lbl_preview_mode.setText(f"ORIGINAL — Shot {self.base_shot_idx + 1} (Base)")
            self.lbl_preview_mode.setVisible(True)
            self.lbl_status.setText(f"Viewing original Shot {self.base_shot_idx + 1}")
        else:
            self.lbl_preview_mode.setVisible(False)
            self.lbl_status.setText("Viewing fused composite")
        self._update_scene()
        self.ribbon_state_changed.emit()

    def _get_aligned_face_kps(self, shot_idx: int, face_idx: int) -> np.ndarray | None:
        if face_idx < len(self.face_is_custom) and self.face_is_custom[face_idx]:
            return None
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

    def _recompute_composite(self) -> None:
        if not self.frames_bgr:
            return
        result = self.frames_bgr[self.base_shot_idx].copy()
        for f_idx, s_idx in sorted(self.face_applied_shot.items()):
            if s_idx == self.base_shot_idx or not (0 <= f_idx < len(self.face_boxes)):
                continue
            source_aligned = self.aligned_frames_bgr[s_idx]
            face_rect = self.face_boxes[f_idx]
            fx, fy, fw, fh = face_rect
            h_s, w_s = source_aligned.shape[:2]
            roi_src = source_aligned[max(0, fy) : min(h_s, fy + fh), max(0, fx) : min(w_s, fx + fw)]
            if roi_src.size > 0 and np.any(np.max(roi_src, axis=2) <= 1):
                source_aligned = fill_warped_border_slivers(source_aligned)
                self.aligned_frames_bgr[s_idx] = source_aligned
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
        self.btn_compare_orig.blockSignals(True)
        self.btn_compare_orig.setChecked(False)
        self.btn_compare_orig.setText("Compare Original")
        self.btn_compare_orig.blockSignals(False)

        self._recompute_composite()
        self._update_scene()
        self._refresh_region_list()
        if self.candidate_popover.isVisible():
            self._show_candidate_popover(self.selected_face_idx)
        self.lbl_status.setText(
            f"Box {self.selected_face_idx + 1}: using Base (Shot {shot_idx + 1})"
            if shot_idx == self.base_shot_idx
            else f"Box {self.selected_face_idx + 1}: replaced from Shot {shot_idx + 1}"
        )
        self.ribbon_state_changed.emit()

    def _reset_to_base(self) -> None:
        if self.frames_bgr:
            self.face_applied_shot.clear()
            self._showing_original = False
            self.lbl_preview_mode.setVisible(False)
            self.btn_compare_orig.blockSignals(True)
            self.btn_compare_orig.setChecked(False)
            self.btn_compare_orig.setText("Compare Original")
            self.btn_compare_orig.blockSignals(False)
            self.base_fused_bgr = self.frames_bgr[self.base_shot_idx].copy()
            self._update_scene()
            self._refresh_region_list()
            if self.candidate_popover.isVisible() and self.selected_face_idx >= 0:
                self._show_candidate_popover(self.selected_face_idx)
            self.lbl_status.setText(f"Reset to Base Shot {self.base_shot_idx + 1}")
            self.ribbon_state_changed.emit()

    def _save_fused_photo(self) -> None:
        if self.base_fused_bgr is None:
            return
        first_path = (
            self.photo_paths[self.base_shot_idx]
            if self.base_shot_idx < len(self.photo_paths)
            else (self.photo_paths[0] if self.photo_paths else None)
        )
        out_path = prompt_save_new_file(self, first_path, "photoFuse", ".jpg")
        if not out_path:
            return
        try:
            cv2.imwrite(str(out_path), self.base_fused_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            self.saved_file_path = out_path
            self.saved.emit(out_path)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save photo:\n{e}")
