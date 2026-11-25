"""
Keyboard Shortcuts Cheat-Sheet (F1) & Quick-Tag Hotkeys (Alt+1..Alt+9) Dialog.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

from PySide6.QtCore import QSettings, QStringListModel, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.ui.ribbon_toolbar import make_wlpg_icon
from src.ui.theme_tokens import get_theme_tokens

_QUICK_TAG_SETTINGS_KEY = "quick_tag_hotkeys"

# (Category, Shortcut, Description)
SHORTCUT_CATALOG: list[tuple[str, str, str]] = [
    ("Navigation & Views", "F1", "Open Keyboard Shortcuts & Quick-Tag Hotkeys"),
    ("Navigation & Views", "F4", "Toggle Immersive Viewer Mode (hide toolbars & panes)"),
    ("Navigation & Views", "Ctrl+N", "Show / hide Navigation Pane"),
    ("Navigation & Views", "Ctrl+I", "Show / hide Information & Tags Pane"),
    ("Navigation & Views", "Ctrl+Shift+F", "Show / hide Viewer Filmstrip"),
    ("Navigation & Views", "Ctrl+F", "Open Text Search Scope Bar"),
    ("Navigation & Views", "Alt+Left / Alt+Right", "Back / Forward navigation history"),
    ("Navigation & Views", "Escape", "Exit Single Image View / close active overlay or tool"),
    ("Culling, Rating & Flags", "0 – 5", "Set star rating (0 clears rating)"),
    ("Culling, Rating & Flags", "Shift + 0 – 5", "Set star rating and auto-advance to next photo"),
    ("Culling, Rating & Flags", "P / Shift+P", "Toggle Picked flag (+ auto-advance with Shift)"),
    ("Culling, Rating & Flags", "X / Shift+X", "Toggle Rejected flag (+ auto-advance with Shift)"),
    ("Culling, Rating & Flags", "U / Shift+U", "Clear flag to Unreviewed (+ auto-advance with Shift)"),
    ("Culling, Rating & Flags", "F / Shift+F", "Toggle Favourite (+ auto-advance with Shift)"),
    ("Spacebar Quick Preview", "Space", "Open / close borderless fullscreen Quick Preview"),
    ("Spacebar Quick Preview", "Left / Right", "Browse previous / next photo in Quick Preview"),
    ("Spacebar Quick Preview", "Enter", "Open current preview photo in Single Image View"),
    ("Spacebar Quick Preview", "Shift + 0–5 / P / X / U / F", "Rate/flag/favourite and auto-advance inside Quick Preview"),
    ("Single Image Viewer & Zoom", "Left / Right", "Previous / Next photo"),
    ("Single Image Viewer & Zoom", "+ / =", "Zoom in"),
    ("Single Image Viewer & Zoom", "- / _", "Zoom out"),
    ("Single Image Viewer & Zoom", "Ctrl+0", "Fit image to window"),
    ("Single Image Viewer & Zoom", "Ctrl+1", "Zoom to 100% actual pixels"),
    ("Single Image Viewer & Zoom", "A", "Toggle camera Autofocus (AF) focus point overlay"),
    ("Single Image Viewer & Zoom", "F11 / F12", "Start fullscreen Slideshow"),
    ("Photo Grouping & Organization", "S", "Expand / collapse selected photo group (or click bottom-left group badge)"),
    ("Photo Grouping & Organization", "Ctrl+T", "Focus Add Tag input"),
    ("Photo Grouping & Organization", "Alt+1 – Alt+9", "Apply assigned Quick-Tag hotkey to selection"),
    ("Photo Grouping & Organization", "F2", "Rename selected file / Batch Rename multi-selection"),
    ("Photo Grouping & Organization", "C", "Open Compare Light Table for selected photos or collapsed photo group"),
    ("Photo Grouping & Organization", "Ctrl+A / Ctrl+C / Ctrl+X / Ctrl+V", "Select All / Copy / Cut / Paste"),
    ("Photo Grouping & Organization", "Delete", "Move selection to Recycle Bin"),
    ("Editing & Files", "Ctrl+,", "Rotate Left (counter-clockwise)"),
    ("Editing & Files", "Ctrl+.", "Rotate Right (clockwise)"),
    ("Editing & Files", "Ctrl+S", "Save edits (backs up original to .OriginalFiles)"),
    ("Editing & Files", "Ctrl+Shift+S", "Save as copy..."),
    ("Editing & Files", "Ctrl+Z / Ctrl+Y", "Undo / Redo"),
    ("Editing & Files", "Ctrl+P", "Print selected photo(s)"),
    ("Editing & Files", "Alt+Enter", "Show file Properties"),
]


def load_quick_tag_hotkeys() -> dict[int, str]:
    """Load `{slot (1..9): tag_name}` quick-tag hotkey assignments from QSettings."""
    settings = QSettings("PhotoGallery64", "PhotoGallery64")
    raw = settings.value(_QUICK_TAG_SETTINGS_KEY, "{}")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    except Exception:
        return {}
    out: dict[int, str] = {}
    if isinstance(parsed, dict):
        for k, v in parsed.items():
            try:
                slot = int(k)
            except (TypeError, ValueError):
                continue
            name = str(v or "").strip()
            if 1 <= slot <= 9 and name:
                out[slot] = name
    return out


def save_quick_tag_hotkeys(mapping: dict[int, str]) -> None:
    """Persist `{slot (1..9): tag_name}` quick-tag hotkey assignments to QSettings."""
    cleaned: dict[str, str] = {}
    for k, v in (mapping or {}).items():
        try:
            slot = int(k)
        except (TypeError, ValueError):
            continue
        name = str(v or "").strip()
        if 1 <= slot <= 9 and name:
            cleaned[str(slot)] = name
    settings = QSettings("PhotoGallery64", "PhotoGallery64")
    settings.setValue(_QUICK_TAG_SETTINGS_KEY, json.dumps(cleaned))


def assign_quick_tag_hotkey(tag_path: str, slot: int | None) -> dict[int, str]:
    """Assign ``tag_path`` to ``Alt+slot`` (1..9), or clear its slot when ``slot is None``."""
    clean = str(tag_path or "").strip()
    mapping = load_quick_tag_hotkeys()
    if clean:
        low = clean.lower()
        for existing_slot in list(mapping.keys()):
            if mapping[existing_slot].strip().lower() == low:
                mapping.pop(existing_slot, None)
    if slot is not None and 1 <= int(slot) <= 9 and clean:
        mapping[int(slot)] = clean
    save_quick_tag_hotkeys(mapping)
    return mapping


def get_quick_tag_slot_for_tag(tag_path: str) -> int | None:
    """Return the assigned slot (1..9) for ``tag_path`` (case-insensitive), or None."""
    low = str(tag_path or "").strip().lower()
    if not low:
        return None
    for slot, name in load_quick_tag_hotkeys().items():
        if name.strip().lower() == low:
            return slot
    return None


def populate_quick_tag_shortcut_submenu(
    parent_menu: Any,
    tag_path: str,
    on_changed: Any = None,
) -> Any:
    """Populate and return an 'Assign Quick-Tag Shortcut (Alt+1..9)' submenu on ``parent_menu``."""
    clean = str(tag_path or "").strip()
    mapping = load_quick_tag_hotkeys()
    cur_slot = get_quick_tag_slot_for_tag(clean)
    title = (
        f"Quick-Tag Shortcut (Alt+{cur_slot})"
        if cur_slot is not None
        else "Assign Quick-Tag Shortcut"
    )
    sub = parent_menu.addMenu(title)
    if not clean:
        sub.setEnabled(False)
        return sub

    for slot in range(1, 10):
        existing = mapping.get(slot, "")
        if cur_slot == slot:
            lbl = f"Alt+{slot}  (This tag)"
        elif existing:
            lbl = f"Alt+{slot}  — {existing}"
        else:
            lbl = f"Alt+{slot}"
        act = sub.addAction(lbl)
        act.setCheckable(True)
        act.setChecked(cur_slot == slot)

        def _on_pick(_checked: bool = False, s: int = slot) -> None:
            assign_quick_tag_hotkey(clean, s)
            if callable(on_changed):
                on_changed()

        act.triggered.connect(_on_pick)

    if cur_slot is not None:
        sub.addSeparator()
        act_clear = sub.addAction(f"Clear Shortcut (Alt+{cur_slot})")

        def _on_clear(_checked: bool = False) -> None:
            assign_quick_tag_hotkey(clean, None)
            if callable(on_changed):
                on_changed()

        act_clear.triggered.connect(_on_clear)
    return sub


class _FilterableTagComboBox(QComboBox):
    """
    Editable tag picker mirroring `_DescriptiveTagInput` in `right_tag_dock.py`:
    browse all tags via the dropdown arrow, or type any substring to filter
    with leaf-prefix matches ranked first.
    """

    def __init__(
        self, all_tags: Sequence[str] = (), parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._all_tags: list[str] = list(
            dict.fromkeys(str(t).strip() for t in all_tags if str(t or "").strip())
        )
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.addItem("")
        if self._all_tags:
            self.addItems(self._all_tags)

        self._completer_model = QStringListModel(self._all_tags, self)
        self.completer_model = self._completer_model
        self._completer = QCompleter(self._completer_model, self)
        self._completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self._completer.setCompletionPrefix("")
        self.setCompleter(self._completer)
        self._completer.activated[str].connect(self._on_completion_activated)

        le = self.lineEdit()
        if le is not None:
            le.setClearButtonEnabled(True)
            le.setPlaceholderText("Type to filter tags or pick...")
            le.textEdited.connect(self._on_text_edited)

    def _sync_filter(self, text: str) -> None:
        val = (text or "").strip()
        if not val:
            matched = list(self._all_tags)
        else:
            q = val.lower()
            matched = [t for t in self._all_tags if q in t.lower()]

            def _sort_key(tag_str: str) -> tuple[int, str, str]:
                low = tag_str.lower()
                leaf = low.rsplit("/", 1)[-1]
                if leaf.startswith(q):
                    rank = 0
                elif low.startswith(q):
                    rank = 1
                else:
                    rank = 2
                return (rank, leaf, low)

            matched.sort(key=_sort_key)

        self._completer_model.setStringList(matched)
        self._completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self._completer.setCompletionPrefix("")

    def _on_text_edited(self, text: str) -> None:
        self._sync_filter(text)
        if self._completer_model.rowCount() > 0:
            self._completer.complete()
        elif self._completer.popup() is not None:
            self._completer.popup().hide()

    def _on_completion_activated(self, text: str) -> None:
        idx = self.findText(text)
        if idx >= 0:
            self.setCurrentIndex(idx)
        else:
            self.setEditText(text)


class KeyboardShortcutsDialog(QDialog):
    """Searchable F1 Keyboard Shortcuts reference & Alt+1..9 Quick-Tag configurator."""

    hotkeys_saved = Signal(dict)

    def __init__(
        self,
        available_tags: Sequence[str] = (),
        *,
        is_dark: bool = False,
        initial_tab: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts & Quick-Tag Hotkeys")
        self.setWindowIcon(make_wlpg_icon("gallery", 16))
        self.resize(620, 520)
        self.setMinimumSize(520, 420)
        self._is_dark = bool(is_dark)
        tok = get_theme_tokens("dark" if self._is_dark else "wlpg_aero", is_dark=self._is_dark)

        self.setStyleSheet(
            f"""
            QDialog {{ background-color: {tok.bg_surface}; color: {tok.text_primary}; }}
            QLabel {{ color: {tok.text_primary}; background: transparent; }}
            QLineEdit, QComboBox, QTreeWidget {{
                background-color: {tok.bg_card};
                color: {tok.text_primary};
                border: 1px solid {tok.border_subtle};
                border-radius: 4px;
                padding: 4px 6px;
            }}
            QAbstractItemView {{
                background-color: {tok.bg_card};
                color: {tok.text_primary};
                border: 1px solid {tok.border_subtle};
                selection-background-color: {tok.selection_bg};
                selection-color: {tok.text_primary};
            }}
            QTreeWidget::item:selected {{
                background-color: {tok.selection_bg};
                color: {tok.text_primary};
            }}
            QTabWidget::pane {{
                border: 1px solid {tok.border_subtle};
                background-color: {tok.bg_surface};
                border-radius: 4px;
            }}
            """
        )

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(14, 12, 14, 12)
        vbox.setSpacing(10)

        hdr = QLabel("Keyboard Shortcuts", self)
        hdr.setFont(QFont("Segoe UI", 13, QFont.DemiBold))
        vbox.addWidget(hdr)

        self.tabs = QTabWidget(self)
        vbox.addWidget(self.tabs, 1)

        # Tab 1: Searchable Shortcuts Reference
        tab_ref = QWidget(self.tabs)
        v_ref = QVBoxLayout(tab_ref)
        v_ref.setContentsMargins(10, 10, 10, 10)
        v_ref.setSpacing(8)

        self.txt_filter = QLineEdit(tab_ref)
        self.txt_filter.setPlaceholderText("Filter shortcuts (e.g. zoom, flag, filmstrip, space, undo)...")
        self.txt_filter.setClearButtonEnabled(True)
        self.txt_filter.textChanged.connect(self._apply_shortcut_filter)
        v_ref.addWidget(self.txt_filter)

        self.tree_shortcuts = QTreeWidget(tab_ref)
        self.tree_shortcuts.setHeaderLabels(["Shortcut", "Action"])
        self.tree_shortcuts.setColumnWidth(0, 185)
        self.tree_shortcuts.setRootIsDecorated(True)
        self.tree_shortcuts.setAlternatingRowColors(True)
        self._populate_shortcuts_tree("")
        v_ref.addWidget(self.tree_shortcuts, 1)
        self.tabs.addTab(tab_ref, "Shortcuts Reference (F1)")

        # Tab 2: Quick-Tag Hotkeys (Alt+1..Alt+9)
        tab_tags = QWidget(self.tabs)
        v_tags = QVBoxLayout(tab_tags)
        v_tags.setContentsMargins(12, 10, 12, 10)
        v_tags.setSpacing(8)

        lbl_info = QLabel(
            "Assign up to 9 descriptive tags to Alt+1 through Alt+9 (type to filter tags, or pick from the list):",
            tab_tags,
        )
        lbl_info.setWordWrap(True)
        v_tags.addWidget(lbl_info)

        form = QFormLayout()
        form.setSpacing(7)
        current_map = load_quick_tag_hotkeys()
        sorted_tags = sorted({str(t).strip() for t in available_tags if str(t or "").strip()}, key=str.casefold)
        self.hotkey_combos: dict[int, _FilterableTagComboBox] = {}
        for slot in range(1, 10):
            cmb = _FilterableTagComboBox(sorted_tags, tab_tags)
            if slot in current_map:
                val = current_map[slot]
                idx = cmb.findText(val)
                if idx >= 0:
                    cmb.setCurrentIndex(idx)
                else:
                    cmb.setEditText(val)
            else:
                cmb.setEditText("")
            self.hotkey_combos[slot] = cmb
            form.addRow(f"Alt + {slot}:", cmb)
        v_tags.addLayout(form)

        row_clear = QHBoxLayout()
        self.btn_clear_hotkeys = QPushButton("Clear All", tab_tags)
        self.btn_clear_hotkeys.setAutoDefault(False)
        self.btn_clear_hotkeys.clicked.connect(self._on_clear_hotkeys)
        row_clear.addWidget(self.btn_clear_hotkeys)
        row_clear.addStretch(1)
        v_tags.addLayout(row_clear)
        v_tags.addStretch(1)
        self.tabs.addTab(tab_tags, "Quick-Tag Hotkeys (Alt+1–9)")

        if 0 <= int(initial_tab) < self.tabs.count():
            self.tabs.setCurrentIndex(int(initial_tab))

        # Dialog Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_save = QPushButton("Save Hotkeys", self)
        self.btn_save.setAutoDefault(False)
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.btn_close = QPushButton("Close", self)
        self.btn_close.setAutoDefault(False)
        self.btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_close)
        vbox.addLayout(btn_row)

    def _populate_shortcuts_tree(self, query: str) -> None:
        self.tree_shortcuts.clear()
        q = (query or "").strip().lower()
        cat_items: dict[str, QTreeWidgetItem] = {}
        cat_font = QFont("Segoe UI", 9, QFont.DemiBold)
        for cat, keys, desc in SHORTCUT_CATALOG:
            if q and q not in cat.lower() and q not in keys.lower() and q not in desc.lower():
                continue
            parent_item = cat_items.get(cat)
            if parent_item is None:
                parent_item = QTreeWidgetItem([cat, ""])
                parent_item.setFont(0, cat_font)
                parent_item.setFirstColumnSpanned(True)
                self.tree_shortcuts.addTopLevelItem(parent_item)
                cat_items[cat] = parent_item
            row_item = QTreeWidgetItem([keys, desc])
            parent_item.addChild(row_item)
            parent_item.setExpanded(True)

    def _apply_shortcut_filter(self, text: str) -> None:
        self._populate_shortcuts_tree(text)

    def _on_clear_hotkeys(self) -> None:
        for cmb in self.hotkey_combos.values():
            cmb.setCurrentIndex(0)
            cmb.setEditText("")

    def collect_hotkeys(self) -> dict[int, str]:
        out: dict[int, str] = {}
        for slot, cmb in self.hotkey_combos.items():
            val = cmb.currentText().strip()
            if val:
                out[slot] = val
        return out

    def _on_save_clicked(self) -> None:
        mapping = self.collect_hotkeys()
        save_quick_tag_hotkeys(mapping)
        self.hotkeys_saved.emit(mapping)
        self.accept()
