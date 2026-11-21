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

        self.setWindowTitle("Import from Google Picasa")
        self.setMinimumWidth(480)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(16, 16, 16, 14)
        vbox.setSpacing(10)

        lbl_intro = QLabel(
            "Choose a photo folder to import your Google Picasa stars, captions, tags, people, and albums non-destructively.",
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
        plan = scan_picasa_tree(root_str)
        self._plan = plan
        if not plan.entries:
            self.lbl_summary.setText("No Google Picasa metadata was found in this folder.")
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
        self.lbl_summary.setText(
            f"Found Picasa metadata for <b>{len(plan.entries)}</b> photo(s):<br>{details_html}"
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
        self.accept()
