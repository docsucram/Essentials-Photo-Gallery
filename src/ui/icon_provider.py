"""
Centralized High-DPI Icon Provider with Theme Support.
Supports:
  - 'wlpg_aero': Windows Live Photo Gallery Classic Aero glossy/colorful vectors & PNGs.
  - 'fluent': Microsoft Fluent UI (Windows 11) style icons using qtawesome / vector glyphs.
  - 'minimal': Monochrome vector icons.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

# Ensure QtPy uses PySide6 if imported via qtawesome
os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QIcon, QPixmap

from src.utils.paths import get_icons_dir

_CURRENT_ICON_THEME: str = "wlpg_aero"
_ICON_CACHE: dict[tuple[str, str, int], QIcon] = {}
_AERO_RENDERER: Callable[[str, int], QIcon] | None = None
_QTA_MODULE: Any = None

# Fluent UI / Modern vector glyph mapping using QtAwesome
# Prefixes: msc (Microsoft / VS icons), ri (Remix Icon), fa5s (FontAwesome 5 Solid), mdi6 (Material Design 6)
_FLUENT_MAP: dict[str, tuple[str, str]] = {
    # File / Folder / Import
    "folder": ("msc.folder", "#0078D4"),
    "folder_open": ("msc.folder-opened", "#0078D4"),
    "folder_add": ("msc.new-folder", "#107C41"),
    "import": ("ri.download-cloud-2-line", "#107C41"),
    "options": ("msc.settings-gear", "#484644"),
    "settings": ("msc.settings-gear", "#484644"),
    # Manage
    "delete": ("msc.trash", "#D83B01"),
    "copy": ("msc.copy", "#0078D4"),
    "duplicates": ("msc.files", "#0078D4"),
    "find_duplicates": ("msc.files", "#0078D4"),
    "cut": ("ri.scissors-cut-line", "#0078D4"),
    "paste": ("ri.clipboard-line", "#0078D4"),
    "select_all": ("msc.checklist", "#0078D4"),
    "undo": ("msc.undo", "#0078D4"),
    "redo": ("msc.redo", "#0078D4"),
    "rotate_ccw": ("ri.anticlockwise-2-line", "#0078D4"),
    "rotate_cw": ("ri.clockwise-2-line", "#0078D4"),
    # Find / Dates / Media
    "print": ("ri.printer-line", "#0078D4"),
    "panoramic": ("ri.panorama-line", "#0078D4"),
    "panorama": ("ri.panorama-line", "#0078D4"),
    "collage": ("ri.layout-masonry-line", "#0078D4"),
    "photo_fuse": ("ri.camera-lens-line", "#0078D4"),
    "fuse": ("ri.camera-lens-line", "#0078D4"),
    "calendar": ("msc.calendar", "#0078D4"),
    "dates": ("msc.calendar", "#0078D4"),
    "dates_grid": ("msc.calendar", "#0078D4"),
    "months": ("ri.calendar-event-line", "#0078D4"),
    "years": ("ri.calendar-2-line", "#0078D4"),
    "text_search": ("msc.search", "#0078D4"),
    "binoculars": ("msc.search", "#0078D4"),
    "edited": ("msc.edit", "#0078D4"),
    "media_type": ("ri.folder-video-line", "#881798"),
    "all_media": ("ri.gallery-line", "#0078D4"),
    # Tags & People
    "tag": ("msc.tag", "#0078D4"),
    "tag_add": ("ri.price-tag-3-line", "#107C41"),
    "not_tagged": ("ri.price-tag-3-line", "#888888"),
    "tag_hierarchy": ("msc.list-tree", "#0078D4"),
    "person": ("msc.account", "#0078D4"),
    "people_tag": ("msc.account", "#0078D4"),
    "batch_people": ("ri.group-line", "#0078D4"),
    "caption": ("ri.chat-new-line", "#0078D4"),
    "geotag": ("msc.globe", "#0078D4"),
    "globe": ("msc.globe", "#0078D4"),
    # Ratings & Flags
    "flag": ("msc.flag", "#D83B01"),
    "flag_on": ("msc.flag", "#D83B01"),
    "flag_off": ("msc.flag", "#888888"),
    "pin": ("msc.pin", "#0078D4"),
    "pin_active": ("msc.pinned", "#D83B01"),
    "star": ("ri.star-fill", "#FFB900"),
    "rating": ("ri.star-fill", "#FFB900"),
    "star_filled": ("ri.star-fill", "#FFB900"),
    "star_empty": ("ri.star-line", "#888888"),
    # Navigation & View
    "nav_back": ("msc.arrow-left", "#0078D4"),
    "nav_forward": ("msc.arrow-right", "#0078D4"),
    "slideshow": ("ri.slideshow-3-line", "#0078D4"),
    "layout_grid": ("ri.grid-fill", "#0078D4"),
    "thumbnail_view": ("ri.grid-fill", "#0078D4"),
    "thumbnail_grid": ("ri.grid-fill", "#0078D4"),
    "view_details": ("ri.list-check-2", "#0078D4"),
    "layout_mosaic": ("ri.dashboard-line", "#0078D4"),
    "pane_left": ("msc.layout-sidebar-left", "#0078D4"),
    "pane_right": ("msc.layout-sidebar-right", "#0078D4"),
    "tag_caption_pane": ("msc.layout-sidebar-right", "#0078D4"),
    "zoom_in": ("msc.zoom-in", "#0078D4"),
    "zoom_out": ("msc.zoom-out", "#0078D4"),
    "fit_window": ("ri.aspect-ratio-line", "#0078D4"),
    "actual_size": ("ri.fullscreen-line", "#0078D4"),
    "actual_pixels": ("ri.fullscreen-line", "#0078D4"),
    # Photo Adjustments (Edit Tab)
    "auto_adjust": ("msc.wand", "#0078D4"),
    "crop": ("ri.crop-line", "#0078D4"),
    "red_eye": ("ri.eye-line", "#D83B01"),
    "retouch": ("fa5s.paint-brush", "#0078D4"),
    "spot_fix": ("fa5s.band-aid", "#0078D4"),
    "markup": ("ri.markup-line", "#0078D4"),
    "pen": ("ri.pen-nib-line", "#0078D4"),
    "highlighter": ("fa5s.highlighter", "#FFB900"),
    "arrow": ("ri.arrow-right-line", "#0078D4"),
    "rect": ("ri.checkbox-blank-line", "#0078D4"),
    "ellipse": ("ri.circle-line", "#0078D4"),
    "star_shape": ("ri.star-line", "#0078D4"),
    "blur": ("ri.blur-off-line", "#0078D4"),
    "pixelate": ("ri.grid-line", "#0078D4"),
    "magnify": ("ri.search-eye-line", "#0078D4"),
    "text": ("ri.text", "#0078D4"),
    "swap": ("ri.arrow-left-right-line", "#0078D4"),
    "straighten": ("ri.ruler-line", "#0078D4"),
    "noise_reduction": ("ri.sparkling-line", "#0078D4"),
    "adjust_color": ("ri.palette-line", "#0078D4"),
    "color_adjust": ("ri.palette-line", "#0078D4"),
    "color_spectrum": ("ri.rainbow-line", "#FFB900"),
    "adjust_exposure": ("ri.sun-line", "#FFB900"),
    "exposure_adjust": ("ri.sun-line", "#FFB900"),
    "fine_tune": ("ri.equalizer-line", "#0078D4"),
    "revert_original": ("msc.discard", "#D83B01"),
    "revert": ("msc.discard", "#D83B01"),
    "make_copy": ("ri.file-copy-line", "#0078D4"),
    "view_file": ("msc.file", "#0078D4"),
    "close_file": ("msc.close", "#484644"),
    "rename": ("msc.edit", "#0078D4"),
    "rename_file": ("msc.edit", "#0078D4"),
    "rename_photo": ("msc.edit", "#0078D4"),
    "resize": ("ri.aspect-ratio-line", "#0078D4"),
    "resize_photo": ("ri.aspect-ratio-line", "#0078D4"),
    "rated_higher": ("msc.arrow-up", "#0078D4"),
    "rated_lower": ("msc.arrow-down", "#0078D4"),
    "rated_only": ("ri.equal-line", "#0078D4"),
    "boolean_and": ("ri.add-line", "#0078D4"),
    # Create Tools & Creative UI
    "eyedropper": ("ri.sip-line", "#0078D4"),
    "pipette": ("ri.sip-line", "#0078D4"),
    "brush": ("fa5s.paint-brush", "#0078D4"),
    "eraser": ("fa5s.eraser", "#D83B01"),
    "compare": ("ri.split-cells-horizontal", "#0078D4"),
    "shuffle": ("ri.shuffle-line", "#0078D4"),
    "face": ("msc.account", "#0078D4"),
    "refresh": ("msc.refresh", "#0078D4"),
    "invert": ("ri.contrast-drop-line", "#0078D4"),
    "play": ("ri.play-fill", "#0078D4"),
    "pause": ("ri.pause-fill", "#0078D4"),
    "volume": ("ri.volume-up-fill", "#0078D4"),
    "volume_low": ("ri.volume-down-fill", "#0078D4"),
    "volume_mute": ("ri.volume-mute-fill", "#666666"),
}


class _IconProviderSignals(QObject):
    theme_changed = Signal(str)


signals = _IconProviderSignals()


def set_aero_renderer(renderer: Callable[[str, int], QIcon]) -> None:
    """Registers the procedural WLPG Classic Aero icon renderer from ribbon_toolbar."""
    global _AERO_RENDERER
    _AERO_RENDERER = renderer


def set_icon_theme(theme_name: str) -> None:
    """Sets the active icon theme ('wlpg_aero', 'fluent', 'minimal')."""
    global _CURRENT_ICON_THEME
    if theme_name in ("wlpg_aero", "fluent", "minimal"):
        _CURRENT_ICON_THEME = theme_name
        _ICON_CACHE.clear()
        signals.theme_changed.emit(theme_name)


def get_icon_theme() -> str:
    """Returns the current icon theme name."""
    return _CURRENT_ICON_THEME


def get_icon(kind: str, size: int = 20, color_override: str | None = None) -> QIcon:
    """
    Returns a cached high-DPI QIcon for the given action/kind, tailored to the active icon theme.
    Falls back gracefully if an asset or vector font is missing.
    """
    key = (kind, _CURRENT_ICON_THEME, size)
    if key in _ICON_CACHE and color_override is None:
        return _ICON_CACHE[key]

    icon: QIcon | None = None

    # Check for direct PNG/SVG file asset in assets/icons/<theme>/
    assets_dir = get_icons_dir()
    theme_dir = assets_dir / _CURRENT_ICON_THEME
    asset_file = theme_dir / f"{kind}.png"
    if not asset_file.exists():
        asset_file = theme_dir / f"{kind}.svg"
    if asset_file.exists():
        icon = QIcon(str(asset_file))

    # Fluent Theme via QtAwesome
    if icon is None and _CURRENT_ICON_THEME in ("fluent", "minimal"):
        global _QTA_MODULE
        if _QTA_MODULE is None:
            try:
                import qtawesome as qta
                _QTA_MODULE = qta
            except Exception:
                _QTA_MODULE = False

        if _QTA_MODULE:
            try:
                glyph_spec = _FLUENT_MAP.get(kind)
                if glyph_spec is not None:
                    glyph_name, default_color = glyph_spec
                    tint = (
                        color_override
                        if color_override
                        else ("#333333" if _CURRENT_ICON_THEME == "minimal" else default_color)
                    )
                    icon = _QTA_MODULE.icon(glyph_name, color=tint)
            except Exception:
                icon = None

    # WLPG Classic Aero Theme (or fallback if fluent glyph wasn't found)
    if icon is None and _AERO_RENDERER is not None:
        try:
            icon = _AERO_RENDERER(kind, size)
        except Exception:
            icon = None

    if icon is None:
        icon = QIcon()

    if color_override is None:
        _ICON_CACHE[key] = icon
    return icon
