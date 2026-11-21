"""
Batch Rename Dialog & Atomic Companion-Safe Rename Engine (`F2` on multi-selection).
Supports token templates (`{name}`, `{num}`, `{date}`, `{camera}`), Windows reserved-name
validation, collision detection, two-phase atomic rename with full rollback, and automatic
renaming of `.xmp` sidecars and `.OriginalFiles/` backups.
"""
from __future__ import annotations

import datetime
import os
import re
import uuid
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_WINDOWS_RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
_FORBIDDEN_CHARS_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def validate_windows_filename(filename: str) -> str | None:
    """Returns an error string if `filename` is invalid on Windows, or None if valid."""
    if not filename or not filename.strip():
        return "Filename cannot be empty."
    if filename.endswith((" ", ".")):
        return "Filename cannot end with a space or period."
    if _FORBIDDEN_CHARS_RE.search(filename):
        return 'Filename contains invalid characters (\\ / : * ? " < > |).'
    stem = filename.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED_STEMS:
        return f"'{stem}' is a reserved Windows device name."
    if len(filename) > 255:
        return "Filename exceeds 255 characters."
    return None


def _sanitize_token_value(val: str) -> str:
    cleaned = _FORBIDDEN_CHARS_RE.sub("_", (val or "").strip())
    return cleaned.rstrip(" .")


def _find_companions(file_path: Path) -> list[tuple[Path, str]]:
    """
    Finds companion files that must move with `file_path`:
    - `<stem>.xmp` and `<stem><ext>.xmp`
    - `<folder>/.OriginalFiles/<filename>`
    Returns list of `(existing_path, kind)` where kind is `'ext_xmp' | 'stem_xmp' | 'backup'`.
    """
    companions: list[tuple[Path, str]] = []
    ext_xmp = file_path.with_suffix(file_path.suffix + ".xmp")
    if ext_xmp.is_file():
        companions.append((ext_xmp, "ext_xmp"))
    stem_xmp = file_path.with_suffix(".xmp")
    if stem_xmp.is_file() and stem_xmp != ext_xmp:
        companions.append((stem_xmp, "stem_xmp"))
    backup = file_path.parent / ".OriginalFiles" / file_path.name
    if backup.is_file():
        companions.append((backup, "backup"))
    return companions


def _companion_target(new_file_path: Path, kind: str) -> Path:
    if kind == "ext_xmp":
        return new_file_path.with_suffix(new_file_path.suffix + ".xmp")
    if kind == "stem_xmp":
        return new_file_path.with_suffix(".xmp")
    return new_file_path.parent / ".OriginalFiles" / new_file_path.name


def build_batch_rename_plan(
    items: list[dict[str, Any]],
    pattern: str,
    start_num: int = 1,
    digits: int = 3,
    case_mode: str = "keep",
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Computes a preview/execution plan for `items` (each with id, folder_path, filename, date_taken, mtime, camera_model).
    Returns `(plan_rows, errors)`.
    """
    errors: list[str] = []
    plan: list[dict[str, Any]] = []
    if not items:
        return plan, ["No files selected."]

    pat = (pattern or "").strip() or "{name}_{num}"
    digits = max(1, min(8, int(digits)))
    start_num = max(0, int(start_num))

    seen_targets: dict[str, int] = {}
    old_paths_lower: set[str] = set()
    for it in items:
        p = Path(it["folder_path"]) / str(it["filename"])
        old_paths_lower.add(str(p).lower())

    for idx, it in enumerate(items):
        mid = int(it["id"])
        folder_p = Path(it["folder_path"])
        old_name = str(it["filename"])
        old_path = folder_p / old_name
        stem = Path(old_name).stem
        ext = Path(old_name).suffix  # includes leading dot

        ts = it.get("date_taken") or it.get("mtime") or 0
        try:
            date_str = datetime.datetime.fromtimestamp(int(ts)).strftime("%Y%m%d") if ts else "00000000"
        except Exception:
            date_str = "00000000"
        cam_str = _sanitize_token_value(str(it.get("camera_model") or "")) or "Camera"
        num_str = f"{start_num + idx:0{digits}d}"

        new_stem = (
            pat.replace("{name}", stem)
            .replace("{num}", num_str)
            .replace("{date}", date_str)
            .replace("{camera}", cam_str)
        )
        new_stem = new_stem.strip()
        if case_mode == "lower":
            new_stem = new_stem.lower()
            ext_out = ext.lower()
        elif case_mode == "upper":
            new_stem = new_stem.upper()
            ext_out = ext.upper()
        else:
            ext_out = ext

        new_name = f"{new_stem}{ext_out}"
        new_path = folder_p / new_name
        companions = _find_companions(old_path) if old_path.exists() else []

        row_err = validate_windows_filename(new_name)
        target_key = str(new_path).lower()
        if row_err is None:
            if target_key in seen_targets:
                row_err = f"Duplicate target name '{new_name}' in batch."
            elif target_key != str(old_path).lower() and new_path.exists() and target_key not in old_paths_lower:
                row_err = f"Target '{new_name}' already exists on disk."
            elif not old_path.exists():
                row_err = "Source file is missing on disk."

        seen_targets[target_key] = mid
        if row_err:
            errors.append(f"{old_name}: {row_err}")

        plan.append(
            {
                "media_id": mid,
                "folder_path": folder_p,
                "old_name": old_name,
                "new_name": new_name,
                "old_path": old_path,
                "new_path": new_path,
                "companions": companions,
                "error": row_err,
            }
        )

    return plan, errors


def execute_batch_rename_plan(
    plan: list[dict[str, Any]],
    catalog_repo: Any,
    media_meta_lookup: dict[int, Any] | None = None,
) -> tuple[int, list[str]]:
    """
    Executes `plan` using two-phase atomic staging so swap/cycle renames succeed and
    any OS error rolls back all filesystem moves cleanly before updating `catalog.db`.
    Returns `(renamed_count, errors)`.
    """
    to_move = [
        row for row in plan
        if row.get("error") is None and row["old_name"] != row["new_name"]
    ]
    if not to_move:
        return 0, []

    batch_tag = uuid.uuid4().hex[:8]
    staged_moves: list[tuple[Path, Path]] = []  # (src, staged_tmp)
    final_moves: list[tuple[Path, Path]] = []   # (staged_tmp, dst)
    completed_staged: list[tuple[Path, Path]] = []
    completed_final: list[tuple[Path, Path]] = []

    try:
        # Phase 1: move every file + its companions to unique temporary names
        for idx, row in enumerate(to_move):
            old_p: Path = row["old_path"]
            new_p: Path = row["new_path"]
            tmp_p = old_p.with_name(f".__epg_ren_{batch_tag}_{idx}_{old_p.name}")
            old_p.rename(tmp_p)
            completed_staged.append((old_p, tmp_p))
            staged_moves.append((old_p, tmp_p))
            final_moves.append((tmp_p, new_p))

            for c_idx, (comp_p, comp_kind) in enumerate(row.get("companions") or []):
                comp_tmp = comp_p.with_name(f".__epg_ren_{batch_tag}_{idx}_{c_idx}_{comp_p.name}")
                comp_p.rename(comp_tmp)
                completed_staged.append((comp_p, comp_tmp))
                final_moves.append((comp_tmp, _companion_target(new_p, comp_kind)))

        # Phase 2: move staged temps to final destination names
        for tmp_p, dst_p in final_moves:
            tmp_p.rename(dst_p)
            completed_final.append((tmp_p, dst_p))

    except Exception as exc:
        # Roll back Phase 2 first, then Phase 1
        for tmp_p, dst_p in reversed(completed_final):
            try:
                if dst_p.exists():
                    dst_p.rename(tmp_p)
            except Exception:
                pass
        for orig_p, tmp_p in reversed(completed_staged):
            try:
                if tmp_p.exists():
                    tmp_p.rename(orig_p)
            except Exception:
                pass
        return 0, [f"Batch rename rolled back due to OS error: {exc}"]

    # Phase 3: commit renames in catalog.db
    renamed = 0
    errors: list[str] = []
    for row in to_move:
        mid = int(row["media_id"])
        try:
            catalog_repo.rename_media(mid, row["new_name"])
            if media_meta_lookup is not None:
                media_meta_lookup.pop(mid, None)
            renamed += 1
        except Exception as exc:
            errors.append(f"DB update failed for {row['new_name']}: {exc}")
    return renamed, errors


class BatchRenameDialog(QDialog):
    """Interactive Batch Rename dialog with live preview, validation, and companion sync."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Batch Rename ({len(items)} items)")
        self.resize(700, 460)
        self._items = list(items)
        self._plan: list[dict[str, Any]] = []
        self._errors: list[str] = []

        vbox = QVBoxLayout(self)
        vbox.setSpacing(10)

        form = QFormLayout()
        form.setSpacing(6)

        pres_row = QHBoxLayout()
        self.combo_preset = QComboBox(self)
        self.combo_preset.addItem("Original Name + Sequence ({name}_{num})", "{name}_{num}")
        self.combo_preset.addItem("Date + Sequence ({date}_{num})", "{date}_{num}")
        self.combo_preset.addItem("Date + Camera + Sequence ({date}_{camera}_{num})", "{date}_{camera}_{num}")
        self.combo_preset.addItem("Custom Pattern...", "")
        pres_row.addWidget(self.combo_preset, 1)
        form.addRow("Preset:", pres_row)

        self.edit_pattern = QLineEdit("{name}_{num}", self)
        self.edit_pattern.setPlaceholderText("Tokens: {name}, {num}, {date}, {camera}")
        form.addRow("Pattern:", self.edit_pattern)

        num_row = QHBoxLayout()
        self.spin_start = QSpinBox(self)
        self.spin_start.setRange(0, 999999)
        self.spin_start.setValue(1)
        self.spin_digits = QSpinBox(self)
        self.spin_digits.setRange(1, 8)
        self.spin_digits.setValue(3)
        self.combo_case = QComboBox(self)
        self.combo_case.addItem("Keep Original Case", "keep")
        self.combo_case.addItem("lowercase", "lower")
        self.combo_case.addItem("UPPERCASE", "upper")
        num_row.addWidget(QLabel("Start #:", self))
        num_row.addWidget(self.spin_start)
        num_row.addWidget(QLabel("Digits:", self))
        num_row.addWidget(self.spin_digits)
        num_row.addWidget(QLabel("Case:", self))
        num_row.addWidget(self.combo_case, 1)
        form.addRow("Numbering:", num_row)
        vbox.addLayout(form)

        self.table = QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(["Current Name", "New Name", "Companions", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        vbox.addWidget(self.table, 1)

        self.lbl_status = QLabel("", self)
        self.lbl_status.setWordWrap(True)
        vbox.addWidget(self.lbl_status)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.btn_ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.btn_ok.setText("Rename Files")
        self.buttons.accepted.connect(self._on_accept)
        self.buttons.rejected.connect(self.reject)
        vbox.addWidget(self.buttons)

        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        self.edit_pattern.textChanged.connect(self._refresh_preview)
        self.spin_start.valueChanged.connect(self._refresh_preview)
        self.spin_digits.valueChanged.connect(self._refresh_preview)
        self.combo_case.currentIndexChanged.connect(self._refresh_preview)

        self._refresh_preview()

    def _on_preset_changed(self) -> None:
        pat = self.combo_preset.currentData()
        if pat:
            self.edit_pattern.setText(pat)

    def _refresh_preview(self) -> None:
        self._plan, self._errors = build_batch_rename_plan(
            self._items,
            self.edit_pattern.text(),
            start_num=self.spin_start.value(),
            digits=self.spin_digits.value(),
            case_mode=str(self.combo_case.currentData() or "keep"),
        )
        self.table.setRowCount(len(self._plan))
        for i, row in enumerate(self._plan):
            self.table.setItem(i, 0, QTableWidgetItem(row["old_name"]))
            self.table.setItem(i, 1, QTableWidgetItem(row["new_name"]))
            comps = len(row.get("companions") or [])
            self.table.setItem(i, 2, QTableWidgetItem(f"{comps} sidecar/backup" if comps else "—"))
            status = row["error"] if row["error"] else ("Unchanged" if row["old_name"] == row["new_name"] else "Ready")
            st_item = QTableWidgetItem(status)
            if row["error"]:
                st_item.setForeground(Qt.GlobalColor.red)
            self.table.setItem(i, 3, st_item)

        if self._errors:
            self.lbl_status.setText(f"⚠ {len(self._errors)} issue(s): {self._errors[0]}")
            self.lbl_status.setStyleSheet("color: #DC2626; font-weight: 600;")
            self.btn_ok.setEnabled(False)
        else:
            changes = sum(1 for r in self._plan if r["old_name"] != r["new_name"])
            self.lbl_status.setText(f"✓ Ready to rename {changes} file(s) atomically.")
            self.lbl_status.setStyleSheet("color: #16A34A; font-weight: 600;")
            self.btn_ok.setEnabled(changes > 0)

    def _on_accept(self) -> None:
        if self._errors:
            QMessageBox.warning(self, "Batch Rename", self._errors[0])
            return
        self.accept()

    def rename_plan(self) -> list[dict[str, Any]]:
        return list(self._plan)
