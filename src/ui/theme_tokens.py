"""
Central ThemeTokens palette & semantic QSS role helpers for Essentials Photo Gallery.
Provides a single source of truth for Classic Aero ('wlpg_aero'), Fluent ('modern'),
and Dark Studio ('dark') themes across both QSS-styled widgets and QPainter delegates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget


@dataclass(frozen=True, slots=True)
class ThemeTokens:
    """Semantic color palette for a UI theme."""

    name: str
    is_dark: bool
    bg_surface: str
    bg_elevated: str
    bg_card: str
    bg_input: str
    bg_hover: str
    text_primary: str
    text_secondary: str
    text_on_accent: str
    border_subtle: str
    accent: str
    accent_hover: str
    danger: str
    danger_hover: str
    success: str
    warning: str
    selection_bg: str
    selection_text: str


AERO_TOKENS = ThemeTokens(
    name="wlpg_aero",
    is_dark=False,
    bg_surface="#F5F8FC",
    bg_elevated="#FFFFFF",
    bg_card="#F8FAFC",
    bg_input="#FFFFFF",
    bg_hover="#E5F1FB",
    text_primary="#0F172A",
    text_secondary="#475569",
    text_on_accent="#FFFFFF",
    border_subtle="#CBD5E1",
    accent="#0078D7",
    accent_hover="#106EBE",
    danger="#DC2626",
    danger_hover="#B91C1C",
    success="#15803D",
    warning="#B45309",
    selection_bg="#CCE4F7",
    selection_text="#0F172A",
)

MODERN_TOKENS = ThemeTokens(
    name="modern",
    is_dark=False,
    bg_surface="#F8FAFC",
    bg_elevated="#FFFFFF",
    bg_card="#F1F5F9",
    bg_input="#FFFFFF",
    bg_hover="#E2E8F0",
    text_primary="#0F172A",
    text_secondary="#475569",
    text_on_accent="#FFFFFF",
    border_subtle="#CBD5E1",
    accent="#2563EB",
    accent_hover="#1D4ED8",
    danger="#DC2626",
    danger_hover="#B91C1C",
    success="#15803D",
    warning="#B45309",
    selection_bg="#DBEAFE",
    selection_text="#0F172A",
)

DARK_TOKENS = ThemeTokens(
    name="dark",
    is_dark=True,
    bg_surface="#1B1E24",
    bg_elevated="#20242C",
    bg_card="#14171C",
    bg_input="#14171C",
    bg_hover="#2E3440",
    text_primary="#E2E8F0",
    text_secondary="#94A3B8",
    text_on_accent="#FFFFFF",
    border_subtle="#334155",
    accent="#2563EB",
    accent_hover="#1D4ED8",
    danger="#DC2626",
    danger_hover="#B91C1C",
    success="#4ADE80",
    warning="#FBBF24",
    selection_bg="#2C415A",
    selection_text="#F8FAFC",
)

_THEME_MAP: dict[str, ThemeTokens] = {
    "wlpg_aero": AERO_TOKENS,
    "aero": AERO_TOKENS,
    "classic": AERO_TOKENS,
    "modern": MODERN_TOKENS,
    "fluent": MODERN_TOKENS,
    "dark": DARK_TOKENS,
}


def get_theme_tokens(theme_name: str | None = None, *, is_dark: bool | None = None) -> ThemeTokens:
    """Return the ThemeTokens for a theme name (or boolean is_dark fallback)."""
    if theme_name:
        key = str(theme_name).strip().lower()
        if key in _THEME_MAP:
            return _THEME_MAP[key]
    if is_dark:
        return DARK_TOKENS
    return AERO_TOKENS


def set_ui_role(widget: QWidget, role: str) -> None:
    """Assign a semantic `uiRole` QSS property and re-polish only the target widget."""
    widget.setProperty("uiRole", role)
    st = widget.style()
    if st is not None:
        st.unpolish(widget)
        st.polish(widget)


def menu_stylesheet(*, is_dark: bool = False, theme_name: str | None = None) -> str:
    """Build a guaranteed-readable QMenu stylesheet from ThemeTokens."""
    tok = get_theme_tokens(theme_name, is_dark=is_dark)
    return (
        f"QMenu {{ background-color: {tok.bg_elevated}; color: {tok.text_primary}; "
        f"border: 1px solid {tok.border_subtle}; padding: 4px; }}"
        f"QMenu::item {{ padding: 5px 20px 5px 12px; border-radius: 3px; "
        f"color: {tok.text_primary}; background: transparent; }}"
        f"QMenu::item:selected {{ background-color: {tok.selection_bg}; color: {tok.selection_text}; }}"
        f"QMenu::separator {{ height: 1px; background: {tok.border_subtle}; margin: 4px 8px; }}"
    )


def _srgb_channel_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    """Compute WCAG 2.1 relative luminance in [0.0, 1.0] from #RRGGBB."""
    s = hex_color.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    r = _srgb_channel_to_linear(int(s[0:2], 16) / 255.0)
    g = _srgb_channel_to_linear(int(s[2:4], 16) / 255.0)
    b = _srgb_channel_to_linear(int(s[4:6], 16) / 255.0)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg_hex: str, bg_hex: str) -> float:
    """Compute WCAG 2.1 contrast ratio (>= 1.0) between two #RRGGBB colors."""
    l1 = relative_luminance(fg_hex)
    l2 = relative_luminance(bg_hex)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)
