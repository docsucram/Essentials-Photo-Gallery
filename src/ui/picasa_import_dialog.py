"""
Dry-run & import dialog for Google Picasa 3 metadata (.picasa.ini & contacts.xml).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.picasa_importer import PicasaImportPlan, apply_picasa_import, scan_picasa_tree
from src.db.repository import CatalogRepository


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

        self.setWindowTitle("Import Picasa Metadata (.picasa.ini)")
        self.setMinimumWidth(520)

        vbox = QVBoxLayout(self)
        vbox.setSpacing(10)

        vbox.addWidget(
            QLabel(
                "Select a root folder containing Google Picasa (.picasa.ini) files to preview and import "
                "starred favourites, captions, tags, face boxes, and albums non-destructively.",
                self,
            )
        )

        row_dir = QHBoxLayout()
        self.txt_root = QLineEdit(initial_dir, self)
        self.txt_root.setPlaceholderText("Root photo folder...")
        self.btn_browse = QPushButton("Browse...", self)
        self.btn_scan = QPushButton("Preview Scan", self)
        row_dir.addWidget(self.txt_root, 1)
        row_dir.addWidget(self.btn_browse)
        row_dir.addWidget(self.btn_scan)
        vbox.addLayout(row_dir)

        self.lbl_summary = QLabel("Click 'Preview Scan' to inspect .picasa.ini files.", self)
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setStyleSheet(
            "padding: 8px; border-radius: 4px; background: rgba(128, 128, 128, 28);"
        )
        vbox.addWidget(self.lbl_summary)

        self.chk_stars = QCheckBox("Import Picasa Stars (star=yes) as ♥ Favourites", self)
        self.chk_stars.setChecked(True)
        self.chk_captions = QCheckBox("Import Captions / Descriptions", self)
        self.chk_captions.setChecked(True)
        self.chk_tags = QCheckBox("Import Keywords as Tags", self)
        self.chk_tags.setChecked(True)
        self.chk_faces = QCheckBox("Import Face Regions & People ([Contacts2] / contacts.xml)", self)
        self.chk_faces.setChecked(True)
        self.chk_albums = QCheckBox("Import Picasa Albums", self)
        self.chk_albums.setChecked(True)

        for chk in (self.chk_stars, self.chk_captions, self.chk_tags, self.chk_faces, self.chk_albums):
            vbox.addWidget(chk)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal,
            self,
        )
        self.btn_import = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.btn_import.setText("Import Metadata")
        self.btn_import.setEnabled(False)
        vbox.addWidget(self.buttons)

        self.btn_browse.clicked.connect(self._on_browse)
        self.btn_scan.clicked.connect(self._on_scan)
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
        plan = scan_picasa_tree(root_str)
        self._plan = plan
        self.lbl_summary.setText(
            f"Found <b>{plan.ini_files}</b> .picasa.ini file(s) covering <b>{len(plan.entries)}</b> photo(s):<br>"
            f"• {plan.stars_count} Starred (♥ Favourites)<br>"
            f"• {plan.captions_count} Captions<br>"
            f"• {plan.tags_count} Keywords<br>"
            f"• {plan.faces_count} Face Boxes<br>"
            f"• {len(plan.albums)} Album(s)"
        )
        self.btn_import.setEnabled(len(plan.entries) > 0)

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
        self.accept()
