"""
Dry-run & import dialog + library folder manager for Google Picasa 3 metadata (.picasa.ini & contacts.xml).
"""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.picasa_importer import PicasaImportPlan, apply_picasa_import, scan_picasa_tree
from src.db.repository import CatalogRepository


def _norm_folder_key(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def get_imported_picasa_folders() -> set[str]:
    """Returns normalized folder paths whose Picasa metadata has already been imported."""
    settings = QSettings("PhotoGallery64", "PhotoGallery64")
    raw = settings.value("picasa_imported_folders", [])
    if isinstance(raw, (list, tuple)):
        return {_norm_folder_key(p) for p in raw if p}
    if raw:
        return {_norm_folder_key(str(raw))}
    return set()


def mark_picasa_folders_imported(paths: list[str | Path] | set[str | Path]) -> None:
    """Persists normalized folder paths as imported in QSettings."""
    if not paths:
        return
    existing = get_imported_picasa_folders()
    for p in paths:
        if p:
            existing.add(_norm_folder_key(p))
    settings = QSettings("PhotoGallery64", "PhotoGallery64")
    settings.setValue("picasa_imported_folders", sorted(existing))


class PicasaImportDialog(QDialog):
    """Scans a folder tree for .picasa.ini files, previews counts, and imports into catalog.db."""

    def __init__(
        self,
        repo: CatalogRepository,
        initial_dir: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.repo = repo
        self._plan: PicasaImportPlan | None = None
        self.import_stats: dict[str, int] | None = None

        self.setWindowTitle("Import from Google Picasa")
        self.setMinimumWidth(480)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(16, 16, 16, 14)
        vbox.setSpacing(10)

        lbl_intro = QLabel(
            "Choose a photo folder (and optionally its subfolders) to import your Google Picasa stars, "
            "captions, tags, people, and albums non-destructively.",
            self,
        )
        lbl_intro.setWordWrap(True)
        vbox.addWidget(lbl_intro)

        row_dir = QHBoxLayout()
        self.txt_root = QLineEdit(initial_dir, self)
        self.txt_root.setPlaceholderText("Select photo folder...")
        self.txt_root.returnPressed.connect(self._on_scan)
        self.btn_browse = QPushButton("Browse...", self)
        row_dir.addWidget(self.txt_root, 1)
        row_dir.addWidget(self.btn_browse)
        vbox.addLayout(row_dir)

        self.chk_subfolders = QCheckBox("Include subfolders", self)
        self.chk_subfolders.setChecked(True)
        self.chk_subfolders.toggled.connect(self._on_scan)
        vbox.addWidget(self.chk_subfolders)

        self.lbl_summary = QLabel("Select a folder to scan for Picasa metadata.", self)
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setStyleSheet(
            "padding: 10px 12px; border-radius: 6px; background: rgba(128, 128, 128, 28);"
        )
        vbox.addWidget(self.lbl_summary)

        # Collapsible options (all enabled by default for clean 1-click import)
        self.btn_customize = QPushButton("▸ Customize what to import", self)
        self.btn_customize.setFlat(True)
        self.btn_customize.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_customize.setStyleSheet(
            "QPushButton { border: none; background: transparent; text-align: left; padding: 2px 0px; font-size: 8.5pt; opacity: 0.85; }"
        )
        vbox.addWidget(self.btn_customize)

        self._opts_box = QWidget(self)
        opts_lay = QVBoxLayout(self._opts_box)
        opts_lay.setContentsMargins(12, 2, 0, 4)
        opts_lay.setSpacing(5)

        self.chk_stars = QCheckBox("Starred photos (as Favourites)", self._opts_box)
        self.chk_stars.setChecked(True)
        self.chk_captions = QCheckBox("Captions & descriptions", self._opts_box)
        self.chk_captions.setChecked(True)
        self.chk_tags = QCheckBox("Keywords & tags", self._opts_box)
        self.chk_tags.setChecked(True)
        self.chk_faces = QCheckBox("People & face regions", self._opts_box)
        self.chk_faces.setChecked(True)
        self.chk_albums = QCheckBox("Albums", self._opts_box)
        self.chk_albums.setChecked(True)

        for chk in (self.chk_stars, self.chk_captions, self.chk_tags, self.chk_faces, self.chk_albums):
            opts_lay.addWidget(chk)
        self._opts_box.setVisible(False)
        vbox.addWidget(self._opts_box)

        def _toggle_opts() -> None:
            vis = self._opts_box.isHidden()
            self._opts_box.setVisible(vis)
            self.btn_customize.setText("▾ Customize what to import" if vis else "▸ Customize what to import")
            self.adjustSize()

        self.btn_customize.clicked.connect(_toggle_opts)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal,
            self,
        )
        self.btn_import = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.btn_import.setText("Import")
        self.btn_import.setEnabled(False)
        vbox.addWidget(self.buttons)

        self.btn_browse.clicked.connect(self._on_browse)
        self.buttons.accepted.connect(self._on_accept)
        self.buttons.rejected.connect(self.reject)

        if initial_dir and Path(initial_dir).is_dir():
            self._on_scan()

    def _on_browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Select Picasa Photo Folder", self.txt_root.text())
        if chosen:
            self.txt_root.setText(chosen)
            self._on_scan()

    def _on_scan(self) -> None:
        root_str = self.txt_root.text().strip()
        if not root_str or not Path(root_str).is_dir():
            self.lbl_summary.setText("Please select a valid folder.")
            self.btn_import.setEnabled(False)
            return
        plan = scan_picasa_tree(root_str, include_subfolders=self.chk_subfolders.isChecked())
        self._plan = plan
        if not plan.entries:
            scope = "this folder or its subfolders" if self.chk_subfolders.isChecked() else "this folder"
            self.lbl_summary.setText(f"No Google Picasa metadata was found in {scope}.")
            self.btn_import.setEnabled(False)
            return
        parts: list[str] = []
        if plan.stars_count:
            parts.append(f"• {plan.stars_count} starred photo(s)")
        if plan.captions_count:
            parts.append(f"• {plan.captions_count} caption(s)")
        if plan.tags_count:
            parts.append(f"• {plan.tags_count} keyword tag(s)")
        if plan.faces_count:
            parts.append(f"• {plan.faces_count} face region(s)")
        if plan.albums:
            parts.append(f"• {len(plan.albums)} album(s)")
        details_html = "<br>".join(parts) if parts else "• Metadata entries ready to import"
        folder_suffix = (
            f" across <b>{plan.ini_files}</b> folder(s)"
            if plan.ini_files > 1
            else ""
        )
        self.lbl_summary.setText(
            f"Found Picasa metadata for <b>{len(plan.entries)}</b> photo(s){folder_suffix}:<br>{details_html}"
        )
        self.btn_import.setEnabled(True)

    def _on_accept(self) -> None:
        if self._plan is None:
            return
        self.import_stats = apply_picasa_import(
            self.repo,
            self._plan,
            import_stars=self.chk_stars.isChecked(),
            import_captions=self.chk_captions.isChecked(),
            import_tags=self.chk_tags.isChecked(),
            import_faces=self.chk_faces.isChecked(),
            import_albums=self.chk_albums.isChecked(),
        )
        imported_dirs = {e.folder_path for e in self._plan.entries}
        imported_dirs.add(self._plan.root_dir)
        mark_picasa_folders_imported(imported_dirs)
        self.accept()


class PicasaFoldersDialog(QDialog):
    """Lists all library folders containing .picasa.ini metadata with import status and 1-click actions."""

    def __init__(self, repo: CatalogRepository, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self.any_imported: bool = False
        self._plans_by_folder: dict[str, PicasaImportPlan] = {}

        self.setWindowTitle("Google Picasa Folders in Library")
        self.setMinimumSize(720, 420)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(14, 14, 14, 12)
        vbox.setSpacing(10)

        self.lbl_header = QLabel(
            "Folders in your library containing Google Picasa metadata (.picasa.ini):",
            self,
        )
        self.lbl_header.setWordWrap(True)
        vbox.addWidget(self.lbl_header)

        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Folder", "Metadata", "Status", "Actions"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        hdr = self.tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.tree.setColumnWidth(3, 185)
        vbox.addWidget(self.tree, 1)

        btn_row = QHBoxLayout()
        self.btn_import_all = QPushButton("Import All Pending", self)
        self.btn_import_all.clicked.connect(self._on_import_all_pending)
        btn_row.addWidget(self.btn_import_all)
        btn_row.addStretch(1)
        btn_close = QPushButton("Close", self)
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        vbox.addLayout(btn_row)

        self._populate()

    def _populate(self) -> None:
        self.tree.clear()
        self._plans_by_folder.clear()
        imported_set = get_imported_picasa_folders()
        ini_names = (".picasa.ini", "Picasa.ini", "picasa.ini")

        folders = self.repo.get_folders()
        found_count = 0
        pending_count = 0

        for f in folders:
            fpath = Path(str(f["path"]))
            if not fpath.is_dir():
                continue
            if not any((fpath / nm).is_file() for nm in ini_names):
                continue
            plan = scan_picasa_tree(fpath, include_subfolders=False)
            if not plan.entries:
                continue
            found_count += 1
            norm_key = _norm_folder_key(fpath)
            self._plans_by_folder[norm_key] = plan
            is_imported = norm_key in imported_set
            if not is_imported:
                pending_count += 1

            parts: list[str] = [f"{len(plan.entries)} photo(s)"]
            if plan.faces_count:
                parts.append(f"{plan.faces_count} faces")
            if plan.stars_count:
                parts.append(f"{plan.stars_count} ★")
            if plan.tags_count:
                parts.append(f"{plan.tags_count} tags")
            if plan.albums:
                parts.append(f"{len(plan.albums)} album(s)")

            item = QTreeWidgetItem(
                [
                    str(fpath),
                    ", ".join(parts),
                    "✓ Imported" if is_imported else "Not imported",
                    "",
                ]
            )
            item.setToolTip(0, str(fpath))
            item.setData(0, Qt.ItemDataRole.UserRole, norm_key)
            self.tree.addTopLevelItem(item)

            act_w = QWidget(self.tree)
            act_l = QHBoxLayout(act_w)
            act_l.setContentsMargins(4, 2, 4, 2)
            act_l.setSpacing(6)
            btn_imp = QPushButton("Re-import" if is_imported else "Import", act_w)
            btn_imp.setFixedHeight(24)
            btn_imp.clicked.connect(lambda _c=False, k=norm_key, fp=fpath: self._import_one(k, fp))
            btn_open = QPushButton("Open Folder", act_w)
            btn_open.setFixedHeight(24)
            btn_open.clicked.connect(lambda _c=False, fp=fpath: self._open_folder(fp))
            act_l.addWidget(btn_imp)
            act_l.addWidget(btn_open)
            self.tree.setItemWidget(item, 3, act_w)

        if found_count == 0:
            self.lbl_header.setText("No folders with Google Picasa metadata (.picasa.ini) were found in your library.")
        else:
            self.lbl_header.setText(
                f"Found {found_count} library folder(s) with Google Picasa metadata ({pending_count} pending import)."
            )
        self.btn_import_all.setEnabled(pending_count > 0)

    def _open_folder(self, folder_path: Path) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder_path)))

    def _import_one(self, norm_key: str, folder_path: Path) -> None:
        plan = self._plans_by_folder.get(norm_key)
        if plan is None:
            return
        apply_picasa_import(self.repo, plan)
        mark_picasa_folders_imported([folder_path])
        self.any_imported = True
        self._populate()

    def _on_import_all_pending(self) -> None:
        imported_set = get_imported_picasa_folders()
        newly_imported: list[Path] = []
        for norm_key, plan in self._plans_by_folder.items():
            if norm_key in imported_set:
                continue
            apply_picasa_import(self.repo, plan)
            newly_imported.append(plan.root_dir)
        if newly_imported:
            mark_picasa_folders_imported(newly_imported)
            self.any_imported = True
            self._populate()
