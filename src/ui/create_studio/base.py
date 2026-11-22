"""Base contract and contextual ribbon descriptor for integrated Create Studio workspaces."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QWidget


@dataclass
class RibbonActionSpec:
    """Declarative action entry exposed to the contextual Create Studio ribbon tab."""

    action_id: str
    label: str
    icon_kind: str
    tooltip: str = ""
    group: str = "Tools"
    large: bool = True
    checkable: bool = False
    checked: bool = False
    enabled: bool = True


class CreateWorkspaceBase(QWidget):
    """
    Base class for integrated Create Studio workspaces hosted inside MainWindow:
    - canvas_widget() -> MainWindow.central_stack
    - inspector_widget() -> RightTagDock ("custom_inspector" slot)
    - bottom_tray_widget() -> BottomFilmstripDock (optional workspace tray)
    - ribbon_actions() -> contextual RibbonToolbar tab
    """

    saved = Signal(object)  # emits Path of newly created file
    cancelled = Signal()
    status_message = Signal(str, int)
    ribbon_state_changed = Signal()
    undo_redo_state_changed = Signal(bool, bool)
    zoom_state_changed = Signal(float, bool)  # (scale_factor, is_fit_mode)

    workspace_id: str = "create_workspace"
    display_title: str = "Create Studio"
    tool_save_label: str = "creation"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.saved_file_path: Path | None = None
        self._theme_name: str = "wlpg_aero"

    def canvas_widget(self) -> QWidget:
        """Widget mounted in MainWindow.central_stack."""
        return self

    def inspector_widget(self) -> QWidget | None:
        """Optional settings/inspector widget hosted in RightTagDock."""
        return None

    def bottom_tray_widget(self) -> QWidget | None:
        """Optional bottom filmstrip/preset tray hosted in BottomFilmstripDock."""
        return None

    def tray_title(self) -> str:
        """Header title shown when bottom_tray_widget() is active."""
        return self.display_title

    def ribbon_actions(self) -> list[RibbonActionSpec]:
        """Contextual ribbon actions displayed alongside Save New... and Cancel."""
        return []

    def ribbon_action_states(self) -> dict[str, dict[str, Any]]:
        """Runtime overrides for contextual ribbon button states."""
        return {}

    def handle_ribbon_action(self, action_id: str) -> None:
        """Handle a click on one of the workspace's contextual ribbon actions."""
        if action_id == "zoom_fit":
            self.zoom_fit()
        elif action_id == "zoom_actual":
            self.zoom_actual()
        elif action_id == "zoom_in":
            self.zoom_in()
        elif action_id == "zoom_out":
            self.zoom_out()

    def has_unsaved_changes(self) -> bool:
        """Return True if leaving should prompt to Save / Discard / Cancel."""
        return False

    def save_result(self) -> Path | None:
        """Save the output creation and return its Path (or None if cancelled/failed)."""
        return None

    def can_undo(self) -> bool:
        return False

    def can_redo(self) -> bool:
        return False

    def undo(self) -> None:
        pass

    def redo(self) -> None:
        pass

    def zoom_in(self) -> None:
        pass

    def zoom_out(self) -> None:
        pass

    def zoom_fit(self) -> None:
        pass

    def zoom_actual(self) -> None:
        pass

    def set_zoom_percent(self, _pct: int) -> None:
        """Set explicit canvas zoom percentage (e.g. 20..400) from MainWindow status slider."""
        pass

    def handle_key_press(self, _event: QKeyEvent) -> bool:
        """Handle workspace-specific key shortcuts; return True if consumed."""
        return False

    def apply_theme_mode(self, theme_name: str) -> None:
        """Propagate theme change ('wlpg_aero', 'modern', 'dark') to workspace widgets."""
        self._theme_name = theme_name or "wlpg_aero"

    def cleanup(self) -> None:
        """Stop background workers/timers safely prior to teardown."""
        pass
