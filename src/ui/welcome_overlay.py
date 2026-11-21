"""
Lap-inspired Empty-Library Welcome Canvas for Essentials Photo Gallery.
Displayed inside the main workspace when no folders/photos have been added yet
(first launch or after Fresh Start), using ThemeTokens for Dark Studio, Fluent, and Aero.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.ui.ribbon_toolbar import make_wlpg_icon
from src.ui.theme_tokens import get_theme_tokens


class WelcomeEmptyOverlay(QWidget):
    """Centered 3-card onboarding canvas shown when the photo library is empty."""

    add_folder_requested = Signal()
    import_picasa_requested = Signal()
    open_settings_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("WelcomeEmptyOverlay")
        self._theme_name: str = "wlpg_aero"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 32, 32, 32)
        outer.setSpacing(0)
        outer.addStretch(2)

        # Centered header: Logo + Title + Tagline
        self._logo_badge = QLabel(self)
        self._logo_badge.setFixedSize(72, 72)
        self._logo_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        try:
            self._logo_badge.setPixmap(make_wlpg_icon("app_logo", 52).pixmap(52, 52))
        except Exception:
            self._logo_badge.setText("🌸")
            self._logo_badge.setFont(QFont("Segoe UI", 28))
        outer.addWidget(self._logo_badge, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addSpacing(14)

        self._lbl_title = QLabel("Essentials Photo Gallery", self)
        self._lbl_title.setFont(QFont("Segoe UI", 17, QFont.Weight.Bold))
        self._lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._lbl_title, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addSpacing(6)

        self._lbl_sub = QLabel(
            "An offline-first photo manager for large local libraries",
            self,
        )
        self._lbl_sub.setFont(QFont("Segoe UI", 10))
        self._lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._lbl_sub, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addSpacing(28)

        # 3-Card Row
        cards_row = QHBoxLayout()
        cards_row.setSpacing(14)
        cards_row.addStretch(1)

        self._cards: list[QFrame] = []
        self._card_titles: list[QLabel] = []
        self._card_descs: list[QLabel] = []

        # Card 1: Add your photos (primary CTA)
        c1 = self._make_card(
            "folder_add",
            "Start with a photo folder",
            "Select a folder on your PC to browse, rate, and organize your photos and videos without moving originals.",
        )
        self.btn_add_folder = QPushButton("+ Add Folder", c1)
        self.btn_add_folder.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add_folder.setFixedHeight(32)
        self.btn_add_folder.clicked.connect(self.add_folder_requested.emit)
        c1.layout().addWidget(self.btn_add_folder, 0, Qt.AlignmentFlag.AlignLeft)
        cards_row.addWidget(c1)

        # Card 2: Automatic metadata & Picasa import
        c2 = self._make_card(
            "import",
            "Bring existing metadata",
            "Ratings, tags, captions, and face tags from XMP or Windows Live Photo Gallery import automatically.",
        )
        self.btn_import_picasa = QPushButton("Import from Picasa...", c2)
        self.btn_import_picasa.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_import_picasa.setFixedHeight(30)
        self.btn_import_picasa.clicked.connect(self.import_picasa_requested.emit)
        c2.layout().addWidget(self.btn_import_picasa, 0, Qt.AlignmentFlag.AlignLeft)
        cards_row.addWidget(c2)

        # Card 3: Smart Albums & Local AI
        c3 = self._make_card(
            "star_filled",
            "Smart Albums & local AI",
            "Automatically group Favourites, Picked/Rejected shots, and optionally enable private on-device Face Recognition.",
        )
        self.btn_open_settings = QPushButton("Customize Options...", c3)
        self.btn_open_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_open_settings.setFixedHeight(30)
        self.btn_open_settings.clicked.connect(self.open_settings_requested.emit)
        c3.layout().addWidget(self.btn_open_settings, 0, Qt.AlignmentFlag.AlignLeft)
        cards_row.addWidget(c3)

        cards_row.addStretch(1)
        outer.addLayout(cards_row)
        outer.addStretch(3)

        self.apply_theme("wlpg_aero")

    def _make_card(self, icon_name: str, title: str, desc: str) -> QFrame:
        card = QFrame(self)
        card.setObjectName("WelcomeCard")
        card.setFixedSize(232, 186)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(18, 18, 18, 16)
        lay.setSpacing(8)

        ico = QLabel(card)
        ico.setFixedSize(22, 22)
        try:
            ico.setPixmap(make_wlpg_icon(icon_name, 18).pixmap(18, 18))
        except Exception:
            pass
        ico.setStyleSheet("background: transparent; border: none;")
        lay.addWidget(ico)

        lbl_t = QLabel(title, card)
        lbl_t.setFont(QFont("Segoe UI", 9.5, QFont.Weight.DemiBold))
        lbl_t.setWordWrap(True)
        lay.addWidget(lbl_t)

        lbl_d = QLabel(desc, card)
        lbl_d.setFont(QFont("Segoe UI", 8.5))
        lbl_d.setWordWrap(True)
        lay.addWidget(lbl_d)
        lay.addStretch(1)

        self._cards.append(card)
        self._card_titles.append(lbl_t)
        self._card_descs.append(lbl_d)
        return card

    def apply_theme(self, theme_name: str) -> None:
        self._theme_name = theme_name or "wlpg_aero"
        tok = get_theme_tokens(self._theme_name)

        self.setStyleSheet(f"QWidget#WelcomeEmptyOverlay {{ background-color: {tok.bg_surface}; }}")

        logo_bg = "#262B35" if tok.is_dark else "#FFFFFF"
        self._logo_badge.setStyleSheet(
            f"background-color: {logo_bg}; border: 1px solid {tok.border_subtle}; border-radius: 16px;"
        )
        self._lbl_title.setStyleSheet(f"color: {tok.text_primary}; background: transparent;")
        self._lbl_sub.setStyleSheet(f"color: {tok.text_secondary}; background: transparent;")

        card_bg = tok.bg_elevated if tok.is_dark else "#FFFFFF"
        card_qss = (
            f"QFrame#WelcomeCard {{ background-color: {card_bg}; "
            f"border: 1px solid {tok.border_subtle}; border-radius: 10px; }}"
        )
        for card in self._cards:
            card.setStyleSheet(card_qss)
        for t in self._card_titles:
            t.setStyleSheet(f"color: {tok.text_primary}; background: transparent; border: none;")
        for d in self._card_descs:
            d.setStyleSheet(f"color: {tok.text_secondary}; background: transparent; border: none;")

        self.btn_add_folder.setStyleSheet(
            f"QPushButton {{ background-color: {tok.accent}; color: {tok.text_on_accent}; "
            f"border: none; border-radius: 6px; padding: 0 14px; font-weight: 600; font-size: 9pt; }}"
            f"QPushButton:hover {{ background-color: {tok.accent_hover}; }}"
        )
        sec_btn_qss = (
            f"QPushButton {{ background-color: transparent; color: {tok.text_primary}; "
            f"border: 1px solid {tok.border_subtle}; border-radius: 6px; padding: 0 11px; font-size: 8.5pt; }}"
            f"QPushButton:hover {{ background-color: {tok.bg_hover}; border-color: {tok.accent}; }}"
        )
        self.btn_import_picasa.setStyleSheet(sec_btn_qss)
        self.btn_open_settings.setStyleSheet(sec_btn_qss)
