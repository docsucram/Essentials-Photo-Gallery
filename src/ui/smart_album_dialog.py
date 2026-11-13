from __future__ import annotations

import calendar
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.db.repository import CatalogRepository


_UNARY_OPS = {
    "is_empty",
    "is_not_empty",
    "has_gps",
    "no_gps",
    "is_flagged",
    "is_unflagged",
}


# Declarative field registry driving SmartRuleRowWidget uniformly.
# Each entry: (field_key, label, [(op_key, op_label), ...], editor_kind, choices_or_placeholder)
SMART_FIELD_SPECS: list[tuple[str, str, list[tuple[str, str]], str, Any]] = [
    (
        "people",
        "People",
        [
            ("includes", "includes"),
            ("excludes", "does not include"),
            ("is_empty", "has no people"),
            ("is_not_empty", "has any person"),
        ],
        "autocomplete",
        "Person name...",
    ),
    (
        "face_count",
        "Detected faces",
        [
            ("gte", "is at least (>=)"),
            ("lte", "is at most (<=)"),
            ("eq", "is exactly (=)"),
            ("between", "is between (min-max)"),
        ],
        "text",
        "e.g. 2 or 1-5",
    ),
    (
        "tag",
        "Descriptive tag",
        [
            ("includes", "includes"),
            ("excludes", "does not include"),
            ("is_empty", "has no tags"),
            ("is_not_empty", "has any tag"),
        ],
        "autocomplete",
        "Tag name or path...",
    ),
    (
        "place",
        "Geotag / Place",
        [
            ("contains", "contains"),
            ("not_contains", "does not contain"),
            ("has_gps", "has GPS coordinates"),
            ("no_gps", "has no GPS coordinates"),
        ],
        "autocomplete",
        "Country, state, or city...",
    ),
    (
        "time_of_day",
        "Time of day",
        [
            ("is_preset", "is"),
            ("between_hours", "between hours"),
        ],
        "time_of_day",
        [
            ("morning", "Morning (05:00 - 11:59)"),
            ("afternoon", "Afternoon (12:00 - 16:59)"),
            ("evening", "Evening (17:00 - 20:59)"),
            ("night", "Night (21:00 - 04:59)"),
        ],
    ),
    (
        "on_this_day",
        "On This Day",
        [
            ("is", "matches today's month & day"),
            ("is_not", "does not match today"),
        ],
        "unary",
        "",
    ),
    (
        "date_taken",
        "Date taken",
        [
            ("eq", "equals"),
            ("gte", "on or after"),
            ("lte", "on or before"),
            ("between", "between"),
            ("neq", "does not equal"),
        ],
        "text",
        "YYYY-MM-DD (or YYYY-MM-DD..YYYY-MM-DD)",
    ),
    (
        "year",
        "Year",
        [
            ("eq", "equals"),
            ("neq", "does not equal"),
            ("gte", "is at least (>=)"),
            ("lte", "is at most (<=)"),
            ("between", "between"),
        ],
        "text",
        "e.g. 2024 or 2020-2024",
    ),
    (
        "month",
        "Month",
        [
            ("eq", "is"),
            ("neq", "is not"),
        ],
        "choices",
        [(str(m), calendar.month_name[m]) for m in range(1, 13)],
    ),
    (
        "camera",
        "Camera",
        [
            ("contains", "contains"),
            ("equals", "equals"),
            ("not_contains", "does not contain"),
            ("is_empty", "is unknown"),
            ("is_not_empty", "is known"),
        ],
        "autocomplete",
        "e.g. X-S20 or Canon...",
    ),
    (
        "camera_make",
        "Camera Manufacturer",
        [
            ("contains", "contains"),
            ("equals", "equals"),
            ("not_contains", "does not contain"),
            ("is_empty", "is unknown"),
            ("is_not_empty", "is known"),
        ],
        "autocomplete",
        "e.g. Fujifilm, Canon, Sony, Apple...",
    ),
    (
        "rating",
        "Rating",
        [
            ("eq", "equals"),
            ("gte", "at least (>=)"),
            ("lte", "at most (<=)"),
            ("neq", "does not equal"),
        ],
        "choices",
        [
            (0, "Unrated (0)"),
            (1, "1 Star"),
            (2, "2 Stars"),
            (3, "3 Stars"),
            (4, "4 Stars"),
            (5, "5 Stars"),
        ],
    ),
    (
        "flag",
        "Flag",
        [
            ("is_flagged", "is flagged"),
            ("is_unflagged", "is unflagged"),
        ],
        "unary",
        "",
    ),
    (
        "media_type",
        "Media type",
        [
            ("is", "is"),
            ("is_not", "is not"),
        ],
        "choices",
        [
            ("photo", "Photo"),
            ("video", "Video"),
            ("raw", "RAW Photo"),
        ],
    ),
    (
        "file_ext",
        "File type",
        [
            ("is", "is"),
            ("is_not", "is not"),
        ],
        "autocomplete",
        "e.g. jpg, heic, jxl, mp4, raw",
    ),
    (
        "orientation",
        "Orientation",
        [
            ("is", "is"),
            ("is_not", "is not"),
        ],
        "choices",
        [
            ("landscape", "Landscape"),
            ("portrait", "Portrait"),
            ("square", "Square"),
        ],
    ),
    (
        "file_size_mb",
        "File size (MB)",
        [
            ("gte", "at least (>= MB)"),
            ("lte", "at most (<= MB)"),
            ("between", "between (min-max MB)"),
        ],
        "text",
        "e.g. 10 or 5-25",
    ),
    (
        "megapixels",
        "Megapixels",
        [
            ("gte", "at least (>= MP)"),
            ("lte", "at most (<= MP)"),
            ("between", "between (min-max MP)"),
        ],
        "text",
        "e.g. 20 or 12-45",
    ),
    (
        "folder",
        "Folder",
        [
            ("in_folder", "is in folder (recursive)"),
            ("not_in_folder", "is not in folder"),
        ],
        "autocomplete",
        "Folder name or path...",
    ),
    (
        "caption_filename",
        "Caption / Filename",
        [
            ("contains", "contains"),
            ("not_contains", "does not contain"),
            ("is_empty", "is empty"),
            ("is_not_empty", "is not empty"),
        ],
        "text",
        "Search text...",
    ),
]

_FIELD_SPEC_MAP: dict[str, tuple[str, str, list[tuple[str, str]], str, Any]] = {
    spec[0]: spec for spec in SMART_FIELD_SPECS
}

_FIELD_ALIASES: dict[str, str] = {
    "person": "people",
    "tags": "tag",
    "descriptive_tag": "tag",
    "geotag": "place",
    "location": "place",
    "time_taken": "time_of_day",
    "date": "date_taken",
    "flagged": "flag",
    "ext": "file_ext",
    "file_type": "file_ext",
    "file_size": "file_size_mb",
    "size_mb": "file_size_mb",
    "caption": "caption_filename",
    "filename": "caption_filename",
    "camera_manufacturer": "camera_make",
    "make": "camera_make",
    "detected_faces": "face_count",
}


class SmartRuleRowWidget(QWidget):
    """Single dynamic boolean condition row inside :class:`SmartAlbumDialog`."""

    changed = Signal()
    remove_requested = Signal(object)

    def __init__(
        self,
        suggestions: dict[str, list[str]],
        initial_rule: dict[str, Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._suggestions = suggestions
        self._building = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)

        self.field_combo = QComboBox(self)
        self.field_combo.setMinimumWidth(145)
        for fkey, flabel, _, _, _ in SMART_FIELD_SPECS:
            self.field_combo.addItem(flabel, fkey)
        layout.addWidget(self.field_combo)

        self.op_combo = QComboBox(self)
        self.op_combo.setMinimumWidth(155)
        layout.addWidget(self.op_combo)

        self._value_container = QWidget(self)
        self._value_layout = QHBoxLayout(self._value_container)
        self._value_layout.setContentsMargins(0, 0, 0, 0)
        self._value_layout.setSpacing(0)
        layout.addWidget(self._value_container, 1)

        self._value_widget: QWidget | None = None

        self.remove_btn = QPushButton("✕", self)
        self.remove_btn.setFixedSize(28, 28)
        self.remove_btn.setToolTip("Remove condition")
        self.remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.remove_btn.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 11pt; color: #DC2626; border: 1px solid #D1D5DB; border-radius: 4px; background: #FFFFFF; } "
            "QPushButton:hover { background: #FEE2E2; border-color: #EF4444; color: #B91C1C; } "
            "QPushButton:disabled { color: #CBD5E1; border-color: #F1F5F9; background: #F8FAFC; }"
        )
        layout.addWidget(self.remove_btn)

        self.field_combo.currentIndexChanged.connect(self._on_field_changed)
        self.op_combo.currentIndexChanged.connect(self._on_op_changed)
        self.remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))

        if initial_rule:
            self.set_rule(initial_rule)
        else:
            self._rebuild_operators_and_editor()

    def _clear_value_widget(self) -> None:
        while self._value_layout.count():
            item = self._value_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._value_widget = None

    def _current_field_key(self) -> str:
        return str(self.field_combo.currentData() or "people")

    def _current_op_key(self) -> str:
        return str(self.op_combo.currentData() or "")

    def _on_field_changed(self) -> None:
        if self._building:
            return
        self._rebuild_operators_and_editor()
        self.changed.emit()

    def _on_op_changed(self) -> None:
        if self._building:
            return
        self._rebuild_value_editor(preserve_value=self._read_raw_value())
        self.changed.emit()

    def _rebuild_operators_and_editor(
        self, op_key: str | None = None, value: Any = None
    ) -> None:
        fkey = self._current_field_key()
        spec = _FIELD_SPEC_MAP.get(fkey, SMART_FIELD_SPECS[0])
        _, _, ops, _, _ = spec

        self._building = True
        try:
            self.op_combo.clear()
            for ok, olabel in ops:
                self.op_combo.addItem(olabel, ok)
            if op_key:
                idx = self.op_combo.findData(op_key)
                if idx >= 0:
                    self.op_combo.setCurrentIndex(idx)
            self._rebuild_value_editor(preserve_value=value)
        finally:
            self._building = False

    def _rebuild_value_editor(self, preserve_value: Any = None) -> None:
        fkey = self._current_field_key()
        op = self._current_op_key()
        spec = _FIELD_SPEC_MAP.get(fkey, SMART_FIELD_SPECS[0])
        _, _, _, kind, extra = spec

        self._clear_value_widget()

        if kind == "unary" or op in _UNARY_OPS:
            lbl = QLabel("(no value needed)", self._value_container)
            lbl.setStyleSheet("color: #6b7280; font-style: italic; padding-left: 4px;")
            self._value_layout.addWidget(lbl)
            self._value_widget = lbl
            return

        if kind == "time_of_day":
            if op == "between_hours":
                edit = QLineEdit(self._value_container)
                edit.setPlaceholderText("Hours e.g. 17-20")
                if preserve_value is not None and "-" in str(preserve_value):
                    edit.setText(str(preserve_value))
                edit.textChanged.connect(self.changed)
                self._value_layout.addWidget(edit)
                self._value_widget = edit
                return
            combo = QComboBox(self._value_container)
            for vkey, vlabel in extra:
                combo.addItem(vlabel, vkey)
            if preserve_value is not None:
                idx = combo.findData(str(preserve_value).lower())
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.currentIndexChanged.connect(self.changed)
            self._value_layout.addWidget(combo)
            self._value_widget = combo
            return

        if kind == "choices":
            combo = QComboBox(self._value_container)
            for vkey, vlabel in extra:
                combo.addItem(vlabel, vkey)
            if preserve_value is not None:
                for i in range(combo.count()):
                    if str(combo.itemData(i)).lower() == str(preserve_value).lower():
                        combo.setCurrentIndex(i)
                        break
            combo.currentIndexChanged.connect(self.changed)
            self._value_layout.addWidget(combo)
            self._value_widget = combo
            return

        if kind == "autocomplete":
            combo = QComboBox(self._value_container)
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            items = self._suggestions.get(fkey, [])
            combo.addItems(items)
            combo.setCurrentText("" if preserve_value is None else str(preserve_value))
            le = combo.lineEdit()
            if le is not None:
                le.setPlaceholderText(str(extra))
                comp = QCompleter(items, combo)
                comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
                comp.setFilterMode(Qt.MatchFlag.MatchContains)
                le.setCompleter(comp)
            combo.currentTextChanged.connect(self.changed)
            self._value_layout.addWidget(combo)
            self._value_widget = combo
            return

        edit = QLineEdit(self._value_container)
        edit.setPlaceholderText(str(extra))
        if preserve_value is not None:
            if isinstance(preserve_value, (list, tuple)) and len(preserve_value) >= 2:
                edit.setText(f"{preserve_value[0]}..{preserve_value[1]}")
            else:
                edit.setText(str(preserve_value))
        edit.textChanged.connect(self.changed)
        self._value_layout.addWidget(edit)
        self._value_widget = edit

    def _read_raw_value(self) -> Any:
        w = self._value_widget
        if isinstance(w, QComboBox):
            if w.isEditable():
                return w.currentText().strip()
            return w.currentData()
        if isinstance(w, QLineEdit):
            return w.text().strip()
        return ""

    def set_rule(self, rule: dict[str, Any]) -> None:
        """Populate row widgets from a serialized rule dict."""
        raw_field = str(rule.get("field") or "people").strip().lower()
        fkey = _FIELD_ALIASES.get(raw_field, raw_field)
        self._building = True
        try:
            idx = self.field_combo.findData(fkey)
            if idx >= 0:
                self.field_combo.setCurrentIndex(idx)
        finally:
            self._building = False
        self._rebuild_operators_and_editor(
            op_key=str(rule.get("op") or "").strip().lower(),
            value=rule.get("value"),
        )

    def get_rule(self) -> dict[str, Any]:
        """Return ``{'field': ..., 'op': ..., 'value': ...}`` for the compiler."""
        fkey = self._current_field_key()
        op = self._current_op_key()
        val: Any = "" if op in _UNARY_OPS or fkey == "on_this_day" else self._read_raw_value()
        if fkey == "rating" and isinstance(val, str) and val.isdigit():
            val = int(val)
        return {"field": fkey, "op": op, "value": val}

    def validate_rule(self) -> tuple[bool, str]:
        """Return (is_valid, error_message) for this condition row."""
        from src.db.repository import _compile_single_smart_rule

        fkey = self._current_field_key()
        op = self._current_op_key()
        label = str(_FIELD_SPEC_MAP.get(fkey, (fkey, fkey))[1])
        if op in _UNARY_OPS or fkey == "on_this_day":
            return True, ""

        raw = self._read_raw_value()
        val_str = "" if raw is None else str(raw).strip()
        if not val_str:
            return False, f"'{label}' cannot be blank."

        if _compile_single_smart_rule(self.get_rule()) is None:
            return False, f"'{label}' has an invalid value ('{val_str}')."
        return True, ""

    def set_invalid_highlight(self, invalid: bool, message: str = "") -> None:
        w = self._value_widget
        if w is not None:
            w.setStyleSheet("border: 1.5px solid #DC2626;" if invalid else "")
            w.setToolTip(message if invalid else "")

    def focus_value_input(self) -> None:
        w = self._value_widget
        if w is not None:
            w.setFocus()


class SmartAlbumDialog(QDialog):
    """Interactive Boolean Smart Album rule builder with live match preview."""

    def __init__(
        self,
        catalog_repo: CatalogRepository,
        initial_name: str = "",
        initial_rules: dict[str, Any] | None = None,
        title: str = "New Smart Album",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._repo = catalog_repo
        self.setWindowTitle(title)
        self.setMinimumSize(720, 460)
        self.resize(760, 520)

        self._suggestions = self._load_suggestions()
        self._rows: list[SmartRuleRowWidget] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # Header: Name + Match Mode
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Smart Album Name:", self))
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("e.g. Evening Photos in England")
        self.name_edit.setText(initial_name)
        name_row.addWidget(self.name_edit, 1)
        root.addLayout(name_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Match:", self))
        self.match_mode_combo = QComboBox(self)
        self.match_mode_combo.addItem("Match ALL of the following rules (AND)", "all")
        self.match_mode_combo.addItem("Match ANY of the following rules (OR)", "any")
        mode_row.addWidget(self.match_mode_combo, 1)
        root.addLayout(mode_row)

        # Scrollable dynamic condition rows
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.StyledPanel)
        self._rows_host = QWidget(self._scroll)
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(8, 8, 8, 8)
        self._rows_layout.setSpacing(4)
        self._rows_layout.addStretch(1)
        self._scroll.setWidget(self._rows_host)
        root.addWidget(self._scroll, 1)

        # Footer controls: [+ Add Condition] + Live Preview count
        ctrl_row = QHBoxLayout()
        self.add_condition_btn = QPushButton("+ Add Condition", self)
        ctrl_row.addWidget(self.add_condition_btn)
        ctrl_row.addStretch(1)
        self.live_preview_label = QLabel("Live Preview: 0 matching items", self)
        self.live_preview_label.setStyleSheet("font-weight: 600; color: #1d4ed8;")
        ctrl_row.addWidget(self.live_preview_label)
        root.addLayout(ctrl_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        root.addWidget(buttons)

        self.add_condition_btn.clicked.connect(lambda: self.add_rule_row())
        self.match_mode_combo.currentIndexChanged.connect(self._update_live_preview)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        self.set_rules_dict(initial_rules or {"match_mode": "all", "rules": []})

    def _load_suggestions(self) -> dict[str, list[str]]:
        """Populate autocomplete values directly from catalog.db."""
        out: dict[str, list[str]] = {
            "people": [],
            "tag": [],
            "place": [],
            "camera": [],
            "camera_make": [],
            "file_ext": [],
            "folder": [],
        }
        try:
            people = self._repo.get_people_list()
            out["people"] = [
                str(p.get("name") or "").strip()
                for p in people
                if str(p.get("name") or "").strip()
            ]
        except Exception:
            pass
        try:
            tags = self._repo.get_tag_tree()
            seen_tags: list[str] = []
            for t in tags:
                fp = str(t.get("full_path") or t.get("name") or "").strip()
                if fp and fp not in seen_tags:
                    seen_tags.append(fp)
            out["tag"] = seen_tags
        except Exception:
            pass
        try:
            out["place"] = self._repo.get_distinct_places(limit=250)
        except Exception:
            pass
        try:
            out["camera"] = self._repo.get_camera_models()
        except Exception:
            pass
        try:
            out["camera_make"] = self._repo.get_camera_makes()
        except Exception:
            pass
        try:
            exts = self._repo.get_distinct_extensions()
            if "raw" not in exts:
                exts = list(exts) + ["raw"]
            out["file_ext"] = exts
        except Exception:
            pass
        try:
            folders = self._repo.get_folders()
            seen_f: list[str] = []
            for f in folders:
                name = str(f.get("name") or "").strip()
                if name and name not in seen_f:
                    seen_f.append(name)
            out["folder"] = seen_f
        except Exception:
            pass
        return out

    def set_rules_dict(self, rules_dict: dict[str, Any]) -> None:
        """Reset dialog rule rows and match mode from a rule dictionary."""
        for r in list(self._rows):
            self._remove_row_widget(r, refresh=False)

        mode = str(rules_dict.get("match_mode") or "all").strip().lower()
        idx = self.match_mode_combo.findData(mode)
        if idx >= 0:
            self.match_mode_combo.setCurrentIndex(idx)

        rules = rules_dict.get("rules") or []
        if not rules:
            self.add_rule_row({"field": "people", "op": "includes", "value": ""}, refresh=False)
        else:
            for rule in rules:
                if isinstance(rule, dict):
                    self.add_rule_row(rule, refresh=False)
        self._sync_remove_buttons()
        self._update_live_preview()

    def add_rule_row(
        self, initial_rule: dict[str, Any] | None = None, refresh: bool = True
    ) -> SmartRuleRowWidget:
        """Append a new dynamic condition row."""
        row = SmartRuleRowWidget(self._suggestions, initial_rule=initial_rule, parent=self._rows_host)
        row.changed.connect(self._update_live_preview)
        row.remove_requested.connect(self._on_remove_row_requested)
        insert_pos = max(0, self._rows_layout.count() - 1)
        self._rows_layout.insertWidget(insert_pos, row)
        self._rows.append(row)
        self._sync_remove_buttons()
        if refresh:
            self._update_live_preview()
        return row

    def _on_remove_row_requested(self, row: SmartRuleRowWidget) -> None:
        if len(self._rows) <= 1:
            return
        self._remove_row_widget(row, refresh=True)

    def _remove_row_widget(self, row: SmartRuleRowWidget, refresh: bool = True) -> None:
        if row in self._rows:
            self._rows.remove(row)
        self._rows_layout.removeWidget(row)
        row.deleteLater()
        self._sync_remove_buttons()
        if refresh:
            self._update_live_preview()

    def _sync_remove_buttons(self) -> None:
        can_remove = len(self._rows) > 1
        for r in self._rows:
            r.remove_btn.setEnabled(can_remove)

    def get_name(self) -> str:
        """Return trimmed Smart Album name."""
        return self.name_edit.text().strip()

    def get_rules_dict(self) -> dict[str, Any]:
        """Serialize current match mode and rule rows for ``compile_smart_album_where``."""
        mode = str(self.match_mode_combo.currentData() or "all")
        return {
            "match_mode": mode,
            "rules": [r.get_rule() for r in self._rows],
        }

    def _update_live_preview(self) -> int:
        invalid_count = 0
        for row in self._rows:
            ok, msg = row.validate_rule()
            if not ok:
                invalid_count += 1
            else:
                row.set_invalid_highlight(False)

        try:
            cnt = self._repo.count_smart_album_matches(self.get_rules_dict())
        except Exception:
            cnt = 0
        noun = "item" if cnt == 1 else "items"
        if invalid_count > 0:
            self.live_preview_label.setText(
                f"Live Preview: {cnt:,} matching {noun} ({invalid_count} incomplete condition{'s' if invalid_count != 1 else ''})"
            )
            self.live_preview_label.setStyleSheet("font-weight: 600; color: #B45309;")
        else:
            self.live_preview_label.setText(f"Live Preview: {cnt:,} matching {noun}")
            self.live_preview_label.setStyleSheet("font-weight: 600; color: #1d4ed8;")
        return cnt

    def _on_accept(self) -> None:
        if not self.get_name():
            QMessageBox.warning(
                self,
                "Smart Album Name Required",
                "Please enter a name for this Smart Album.",
            )
            self.name_edit.setFocus()
            return

        first_bad_row: SmartRuleRowWidget | None = None
        errors: list[str] = []
        for idx, row in enumerate(self._rows, start=1):
            ok, msg = row.validate_rule()
            row.set_invalid_highlight(not ok, msg)
            if not ok:
                errors.append(f"Condition {idx}: {msg}")
                if first_bad_row is None:
                    first_bad_row = row

        if errors:
            QMessageBox.warning(
                self,
                "Incomplete Smart Album Condition",
                "Please complete or remove the blank/invalid condition(s) before saving:\n\n"
                + "\n".join(errors),
            )
            if first_bad_row is not None:
                first_bad_row.focus_value_input()
            return
        self.accept()
