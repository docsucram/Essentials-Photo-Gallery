"""
SearchScopeBar (QWidget) — Authentic Windows Live Photo Gallery Search Scope Bar.
Located directly above the ThumbnailGridView in the central area.
Provides:
- Header row with 'Search by:' title, Pushpin toggle (keeps bar pinned open), and Close ('✕') button.
- Query row with Scope description, 'Search all' hyperlink, Search text input (with Boolean search syntax),
  and interactive removable filter chips for People tags, Descriptive tags, Ratings, Flags, and Dates.
"""

from __future__ import annotations

import calendar
from typing import Any

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.db.repository import CatalogRepository


def _make_pin_icon(pinned: bool, size: int = 16) -> QIcon:
    """Draws a crisp pushpin vector icon (vertical pinned when active, angled when unpinned)."""
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    s = float(size)

    if pinned:
        # Vertical pin (active / latched)
        p.setPen(QPen(QColor("#1A4B7D"), 1.0))
        p.setBrush(QColor("#3A84C8"))
        p.drawRoundedRect(QRectF(s * 0.32, s * 0.14, s * 0.36, s * 0.22), 1.5, 1.5)
        p.setBrush(QColor("#236AA8"))
        p.drawRect(QRectF(s * 0.40, s * 0.36, s * 0.20, s * 0.26))
        p.setPen(QPen(QColor("#1A3A5C"), 1.8, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.50, s * 0.62), QPointF(s * 0.50, s * 0.90))
    else:
        # Angled pushpin (unpinned / inactive)
        p.save()
        p.translate(s * 0.5, s * 0.5)
        p.rotate(-45)
        p.setPen(QPen(QColor("#5C728A"), 1.0))
        p.setBrush(QColor("#8AA4BE"))
        p.drawRoundedRect(QRectF(-s * 0.18, -s * 0.34, s * 0.36, s * 0.20), 1.5, 1.5)
        p.setBrush(QColor("#708CA8"))
        p.drawRect(QRectF(-s * 0.10, -s * 0.14, s * 0.20, s * 0.24))
        p.setPen(QPen(QColor("#3C5268"), 1.6, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(0, s * 0.10), QPointF(0, s * 0.38))
        p.restore()

    p.end()
    return QIcon(pix)


_AERO_TOOLTIP_STYLE = """
QToolTip {
    background-color: #FFFFFF;
    color: #1E395B;
    border: 1px solid #7DA2CE;
    border-radius: 2px;
    padding: 3px 6px;
    font-family: 'Segoe UI', sans-serif;
    font-size: 9pt;
}
"""


class FilterChip(QFrame):
    """
    Subtle aero chip displaying an active filter facet matching authentic Windows Live Photo Gallery.
    - Default unhovered state: clean plain Segoe UI text with no background, border, or visible '✕'.
    - Hovered state: light blue aero outline box (#7EB4EA) and subtle background (#E5F1FB) with an interactive '✕' remove button.
    """

    remove_clicked = Signal()

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("FilterChip")
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_Hover, True)

        self._hbox = QHBoxLayout(self)
        self._hbox.setContentsMargins(4, 1, 3, 1)
        self._hbox.setSpacing(3)

        self.lbl = QLabel(text, self)
        self.lbl.setStyleSheet(
            "color: #1E395B; font-family: 'Segoe UI'; font-size: 8.5pt; border: none; background: transparent;"
        )
        self.lbl.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._hbox.addWidget(self.lbl)

        self.btn_close = QPushButton("✕", self)
        self.btn_close.setFixedSize(13, 13)
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.setToolTip(f"Remove '{text}' filter")
        self.btn_close.setStyleSheet(
            """
            QPushButton {
                border: none;
                background: transparent;
                color: #55708C;
                font-family: 'Segoe UI';
                font-size: 7.5pt;
                font-weight: bold;
                padding: 0;
            }
            QPushButton:hover {
                color: #C50F1F;
                background: #F8D7DA;
                border-radius: 2px;
            }
            """ + _AERO_TOOLTIP_STYLE
        )
        self.btn_close.clicked.connect(self.remove_clicked.emit)
        self._hbox.addWidget(self.btn_close)
        self.btn_close.setVisible(False)

        self._set_hover_state(False)

    def text(self) -> str:
        return self.lbl.text()

    def _set_hover_state(self, hovered: bool) -> None:
        if hovered:
            self.setStyleSheet(
                """
                QFrame#FilterChip {
                    background: #E5F1FB;
                    border: 1px solid #7EB4EA;
                    border-radius: 2px;
                }
                """
            )
            self.btn_close.setVisible(True)
        else:
            self.setStyleSheet(
                """
                QFrame#FilterChip {
                    background: transparent;
                    border: 1px solid transparent;
                    border-radius: 2px;
                }
                """
            )
            self.btn_close.setVisible(False)

    def enterEvent(self, event: Any) -> None:
        self._set_hover_state(True)
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self._set_hover_state(False)
        super().leaveEvent(event)


class SearchScopeBar(QFrame):
    """
    Windows Live Photo Gallery Search Scope Bar.
    Renders directly above the ThumbnailGridView in the central pane.
    """

    search_text_changed = Signal(str)           # Debounced search query
    search_all_requested = Signal()             # Resets scope to All Photos
    pin_toggled = Signal(bool)                  # Pinned state changed
    close_requested = Signal()                  # Close button clicked
    person_remove_requested = Signal(int)       # person_id
    tag_remove_requested = Signal(int)          # tag_id
    rating_clear_requested = Signal()
    flagged_clear_requested = Signal()
    date_clear_requested = Signal()
    location_clear_requested = Signal()
    special_people_clear_requested = Signal()
    media_type_clear_requested = Signal()
    orientation_clear_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SearchScopeBar")
        self._is_pinned = False

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(180)
        self._debounce_timer.timeout.connect(self._emit_search_text)

        self._apply_style()

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(12, 4, 10, 4)
        vbox.setSpacing(2)

        # ---------------------------------------------------------------------
        # Row 1: Header ("Search by:" title + Pin & Close buttons)
        # ---------------------------------------------------------------------
        r1_layout = QHBoxLayout()
        r1_layout.setContentsMargins(0, 0, 0, 0)
        r1_layout.setSpacing(4)

        self.lbl_title = QLabel("Search by:", self)
        self.lbl_title.setStyleSheet(
            "color: #1F497D; font-family: 'Segoe UI'; font-size: 9.5pt; font-weight: bold; border: none; background: transparent;"
        )
        r1_layout.addWidget(self.lbl_title)

        r1_layout.addStretch(1)

        # Pin button (latches open)
        self.btn_pin = QToolButton(self)
        self.btn_pin.setCheckable(True)
        self.btn_pin.setFixedSize(20, 20)
        self.btn_pin.setIcon(_make_pin_icon(False, 16))
        self.btn_pin.setIconSize(QSize(16, 16))
        self.btn_pin.setCursor(Qt.PointingHandCursor)
        self.btn_pin.setToolTip("Pin search bar (keep open)")
        self.btn_pin.setStyleSheet(
            """
            QToolButton {
                border: 1px solid transparent;
                background: transparent;
                border-radius: 2px;
            }
            QToolButton:hover {
                background: #E4EFF9;
                border-color: #A0C2E2;
            }
            QToolButton:checked {
                background: #CCE2F7;
                border-color: #7AAAD4;
            }
            """ + _AERO_TOOLTIP_STYLE
        )
        self.btn_pin.toggled.connect(self._on_pin_toggled)
        r1_layout.addWidget(self.btn_pin)

        # Close button ('✕')
        self.btn_close = QPushButton("✕", self)
        self.btn_close.setFixedSize(18, 18)
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.setToolTip("Close search (Esc)")
        self.btn_close.setStyleSheet(
            """
            QPushButton {
                border: 1px solid transparent;
                background: transparent;
                color: #55708C;
                font-family: 'Segoe UI', sans-serif;
                font-size: 8.5pt;
                font-weight: bold;
                padding: 0px;
                border-radius: 2px;
            }
            QPushButton:hover {
                color: #C50F1F;
                background: #F8D7DA;
                border: 1px solid #E8B4B8;
            }
            """ + _AERO_TOOLTIP_STYLE
        )
        self.btn_close.clicked.connect(self._on_close_clicked)
        r1_layout.addWidget(self.btn_close)

        vbox.addLayout(r1_layout)

        # ---------------------------------------------------------------------
        # Row 2: Scope label, 'Search all', 'Search:' QLineEdit & Active Chips
        # ---------------------------------------------------------------------
        r2_scroll = QScrollArea(self)
        r2_scroll.setWidgetResizable(True)
        r2_scroll.setFrameShape(QFrame.NoFrame)
        r2_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        r2_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        r2_scroll.setFixedHeight(28)
        r2_scroll.setStyleSheet("background: transparent; border: none;")

        r2_container = QWidget(r2_scroll)
        r2_container.setStyleSheet("background: transparent;")
        self.r2_layout = QHBoxLayout(r2_container)
        self.r2_layout.setContentsMargins(0, 0, 0, 0)
        self.r2_layout.setSpacing(6)

        # Scope text: e.g. "Folder: 2022" (hidden when not filtering a specific folder)
        self.lbl_scope_label = QLabel("Folder:", r2_container)
        self.lbl_scope_label.setStyleSheet("color: #55687C; font-family: 'Segoe UI'; font-size: 8.5pt;")
        self.r2_layout.addWidget(self.lbl_scope_label)

        self.lbl_scope_value = QLabel("", r2_container)
        self.lbl_scope_value.setStyleSheet(
            "color: #1E395B; font-weight: bold; font-family: 'Segoe UI'; font-size: 8.5pt;"
        )
        self.r2_layout.addWidget(self.lbl_scope_value)

        self.btn_search_all = QPushButton("Search all", r2_container)
        self.btn_search_all.setFlat(True)
        self.btn_search_all.setCursor(Qt.PointingHandCursor)
        self.btn_search_all.setToolTip("Search across all gallery photos and videos")
        self.btn_search_all.setStyleSheet(
            """
            QPushButton {
                color: #0066CC;
                text-decoration: underline;
                border: none;
                background: transparent;
                padding: 0 4px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
            }
            QPushButton:hover {
                color: #004499;
            }
            """ + _AERO_TOOLTIP_STYLE
        )
        self.btn_search_all.clicked.connect(self.search_all_requested.emit)
        self.r2_layout.addWidget(self.btn_search_all)

        # Spacer between folder scope and search box
        self.sep_folder = QLabel(" ", r2_container)
        self.sep_folder.setFixedWidth(6)
        self.r2_layout.addWidget(self.sep_folder)

        # By default start with folder scope hidden until a specific folder is active
        self.lbl_scope_label.setVisible(False)
        self.lbl_scope_value.setVisible(False)
        self.btn_search_all.setVisible(False)
        self.sep_folder.setVisible(False)

        # Instant Text Search Input
        self.lbl_search_kw = QLabel("Search:", r2_container)
        self.lbl_search_kw.setStyleSheet("color: #55687C; font-family: 'Segoe UI'; font-size: 8.5pt;")
        self.r2_layout.addWidget(self.lbl_search_kw)

        self.txt_search = QLineEdit(r2_container)
        self.txt_search.setObjectName("ScopeBarSearchInput")
        self.txt_search.setPlaceholderText("Search...")
        self.txt_search.setClearButtonEnabled(True)
        self.txt_search.setFixedWidth(160)
        self.txt_search.setFixedHeight(22)
        self.txt_search.setToolTip(
            "Supports Boolean searches (cat AND dog, cat OR dog, -term) and prefixes (person:name, tag:name)"
        )
        self.txt_search.setStyleSheet(
            """
            QLineEdit#ScopeBarSearchInput {
                background: #FFFFFF;
                border: 1px solid #7F9DB9;
                border-radius: 2px;
                padding: 1px 4px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                color: #000000;
            }
            QLineEdit#ScopeBarSearchInput:focus {
                border: 1px solid #3380CC;
            }
            """ + _AERO_TOOLTIP_STYLE
        )
        self.txt_search.textChanged.connect(self._on_search_text_edited)
        self.txt_search.returnPressed.connect(self._emit_search_text)
        self.r2_layout.addWidget(self.txt_search)

        # Dynamic Chips Layout for Active Filter Facets
        self.chips_widget = QWidget(r2_container)
        self.chips_widget.setStyleSheet("background: transparent;")
        self.chips_layout = QHBoxLayout(self.chips_widget)
        self.chips_layout.setContentsMargins(6, 0, 0, 0)
        self.chips_layout.setSpacing(4)
        self.r2_layout.addWidget(self.chips_widget)

        self.r2_layout.addStretch(1)
        r2_scroll.setWidget(r2_container)
        vbox.addWidget(r2_scroll)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QFrame#SearchScopeBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F4F8FD, stop:0.55 #EBF2FA, stop:1 #DDE8F5);
                border: 1px solid #B8CDE3;
                border-radius: 2px;
            }
            QToolTip {
                background-color: #FFFFFF;
                color: #1E395B;
                border: 1px solid #7DA2CE;
                border-radius: 2px;
                padding: 3px 6px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 9pt;
            }
            """
        )

    def set_dark_mode(self, is_dark: bool) -> None:
        self._dark_mode = bool(is_dark)
        self._is_dark = self._dark_mode
        if not self._dark_mode:
            self._apply_style()
            self.lbl_title.setStyleSheet(
                "color: #1F497D; font-family: 'Segoe UI'; font-size: 9.5pt; font-weight: bold; border: none; background: transparent;"
            )
            self.lbl_scope_label.setStyleSheet("color: #55687C; font-family: 'Segoe UI'; font-size: 8.5pt; background: transparent;")
            self.lbl_scope_value.setStyleSheet("color: #1E395B; font-weight: bold; font-family: 'Segoe UI'; font-size: 8.5pt; background: transparent;")
            self.lbl_search_kw.setStyleSheet("color: #55687C; font-family: 'Segoe UI'; font-size: 8.5pt; background: transparent;")
            self.txt_search.setStyleSheet(
                """
                QLineEdit#ScopeBarSearchInput {
                    background: #FFFFFF;
                    border: 1px solid #7F9DB9;
                    border-radius: 2px;
                    padding: 1px 4px;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    color: #000000;
                }
                QLineEdit#ScopeBarSearchInput:focus {
                    border: 1px solid #3380CC;
                }
                """ + _AERO_TOOLTIP_STYLE
            )
        else:
            self.setStyleSheet(
                """
                QFrame#SearchScopeBar {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #20242B, stop:1 #1A1D23);
                    border: 1px solid #2E3440;
                    border-radius: 2px;
                }
                QToolTip {
                    background-color: #FFFFFF;
                    color: #1E395B;
                    border: 1px solid #7DA2CE;
                    border-radius: 2px;
                    padding: 3px 6px;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 9pt;
                }
                """
            )
            self.lbl_title.setStyleSheet(
                "color: #60A5FA; font-family: 'Segoe UI'; font-size: 9.5pt; font-weight: bold; border: none; background: transparent;"
            )
            self.lbl_scope_label.setStyleSheet("color: #94A3B8; font-family: 'Segoe UI'; font-size: 8.5pt; background: transparent;")
            self.lbl_scope_value.setStyleSheet("color: #E2E8F0; font-weight: bold; font-family: 'Segoe UI'; font-size: 8.5pt; background: transparent;")
            self.lbl_search_kw.setStyleSheet("color: #94A3B8; font-family: 'Segoe UI'; font-size: 8.5pt; background: transparent;")
            self.txt_search.setStyleSheet(
                """
                QLineEdit#ScopeBarSearchInput {
                    background: #14171B;
                    border: 1px solid #3E4756;
                    border-radius: 2px;
                    padding: 1px 4px;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    color: #F1F5F9;
                }
                QLineEdit#ScopeBarSearchInput:focus {
                    border: 1px solid #3B82F6;
                }
                """ + _AERO_TOOLTIP_STYLE
            )

    # -------------------------------------------------------------------------
    # Public Control & Query API
    # -------------------------------------------------------------------------
    def set_scope_label(self, desc: str | None) -> None:
        """Updates the 'Folder: Pictures' label, or hides folder scope if no folder is active."""
        if not desc:
            self.lbl_scope_label.setVisible(False)
            self.lbl_scope_value.setVisible(False)
            self.btn_search_all.setVisible(False)
            self.sep_folder.setVisible(False)
            return

        desc_str = str(desc).strip()
        if desc_str.lower().startswith("folder:"):
            _, val = desc_str.split(":", 1)
            folder_name = val.strip()
            self.lbl_scope_label.setText("Folder:")
            self.lbl_scope_value.setText(folder_name)
            self.lbl_scope_label.setVisible(True)
            self.lbl_scope_value.setVisible(True)
            self.btn_search_all.setVisible(True)
            self.sep_folder.setVisible(True)
        else:
            # If not an actual folder filter, do not display as folder scope
            self.lbl_scope_label.setVisible(False)
            self.lbl_scope_value.setVisible(False)
            self.btn_search_all.setVisible(False)
            self.sep_folder.setVisible(False)

    def update_active_filter_chips(
        self,
        filter_state: dict[str, Any],
        catalog_repo: CatalogRepository,
        location_label: str | None = None,
        special_people_label: str | None = None,
    ) -> None:
        """Re-renders active People, Tag, Rating, Flag, Date, Location, Media type, and Orientation facet chips matching WLPG."""
        while self.chips_layout.count():
            item = self.chips_layout.takeAt(0)
            if item.widget() is not None:
                w = item.widget()
                w.setParent(None)
                w.deleteLater()

        # 1. Active People Chips: "People: Alice + Bob" or Special: "People: All tagged or detected"
        person_ids = filter_state.get("person_ids", ())
        if person_ids:
            lbl_p = QLabel("People:", self.chips_widget)
            lbl_p.setStyleSheet("color: #55687C; font-family: 'Segoe UI'; font-size: 8.5pt;")
            self.chips_layout.addWidget(lbl_p)

            person_mode = filter_state.get("person_boolean_mode", "AND")
            sep_str = " / " if str(person_mode).upper() == "OR" else " + "

            for idx, pid in enumerate(person_ids):
                if idx > 0:
                    lbl_sep = QLabel(sep_str, self.chips_widget)
                    lbl_sep.setStyleSheet(
                        "color: #1E395B; font-weight: bold; font-family: 'Segoe UI'; font-size: 8.5pt;"
                    )
                    self.chips_layout.addWidget(lbl_sep)

                p_name = self._resolve_person_name(int(pid), catalog_repo)
                chip = FilterChip(p_name, self.chips_widget)
                chip.remove_clicked.connect(lambda p_id=int(pid): self.person_remove_requested.emit(p_id))
                self.chips_layout.addWidget(chip)
        elif special_people_label:
            lbl_p = QLabel("People:", self.chips_widget)
            lbl_p.setStyleSheet("color: #55687C; font-family: 'Segoe UI'; font-size: 8.5pt;")
            self.chips_layout.addWidget(lbl_p)

            chip = FilterChip(special_people_label, self.chips_widget)
            chip.remove_clicked.connect(self.special_people_clear_requested.emit)
            self.chips_layout.addWidget(chip)

        # 2. Active Descriptive Tag Chips: "Tagged: Burma + Nansumay" or "Tagged: Burma / Nansumay"
        include_tag_ids = filter_state.get("include_tag_ids", ())
        if include_tag_ids:
            lbl_t = QLabel("Tagged:", self.chips_widget)
            lbl_t.setStyleSheet("color: #55687C; font-family: 'Segoe UI'; font-size: 8.5pt;")
            self.chips_layout.addWidget(lbl_t)

            tag_mode = filter_state.get("tag_boolean_mode", "AND")
            sep_str = " / " if str(tag_mode).upper() == "OR" else " + "

            for idx, tid in enumerate(include_tag_ids):
                if idx > 0:
                    lbl_sep = QLabel(sep_str, self.chips_widget)
                    lbl_sep.setStyleSheet(
                        "color: #1E395B; font-weight: bold; font-family: 'Segoe UI'; font-size: 8.5pt;"
                    )
                    self.chips_layout.addWidget(lbl_sep)

                t_name = self._resolve_tag_name(int(tid), catalog_repo)
                chip = FilterChip(t_name, self.chips_widget)
                chip.remove_clicked.connect(lambda t_id=int(tid): self.tag_remove_requested.emit(t_id))
                self.chips_layout.addWidget(chip)

        # 3. Active Rating Chip: "Rating: ★ 4 or higher"
        ratings = filter_state.get("ratings", ())
        if ratings:
            lbl_r = QLabel("Rating:", self.chips_widget)
            lbl_r.setStyleSheet("color: #55687C; font-family: 'Segoe UI'; font-size: 8.5pt;")
            self.chips_layout.addWidget(lbl_r)

            if tuple(ratings) == (0,):
                r_text = "Unrated"
            else:
                min_r = min(ratings)
                r_text = f"★ {min_r} or higher" if len(ratings) > 1 else f"★ {min_r} only"
            chip = FilterChip(r_text, self.chips_widget)
            chip.remove_clicked.connect(self.rating_clear_requested.emit)
            self.chips_layout.addWidget(chip)

        # 4. Active Flagged Chip: "Flagged only"
        if filter_state.get("flagged_only"):
            chip = FilterChip("Flagged only", self.chips_widget)
            chip.remove_clicked.connect(self.flagged_clear_requested.emit)
            self.chips_layout.addWidget(chip)

        # 5. Active Month / Date Chip: "Date: October 2022"
        d_year = filter_state.get("date_year")
        d_month = filter_state.get("date_month")
        if d_year is not None or d_month is not None:
            lbl_d = QLabel("Date:", self.chips_widget)
            lbl_d.setStyleSheet("color: #55687C; font-family: 'Segoe UI'; font-size: 8.5pt;")
            self.chips_layout.addWidget(lbl_d)

            d_str = ""
            if d_month is not None and 1 <= d_month <= 12:
                d_str += calendar.month_name[d_month] + " "
            if d_year is not None:
                d_str += str(d_year)
            chip = FilterChip(d_str.strip() or "Date", self.chips_widget)
            chip.remove_clicked.connect(self.date_clear_requested.emit)
            self.chips_layout.addWidget(chip)

        # 6. Active Location Chip: "Location: Paris"
        if location_label:
            lbl_loc = QLabel("Location:", self.chips_widget)
            lbl_loc.setStyleSheet("color: #55687C; font-family: 'Segoe UI'; font-size: 8.5pt;")
            self.chips_layout.addWidget(lbl_loc)

            chip = FilterChip(location_label, self.chips_widget)
            chip.remove_clicked.connect(self.location_clear_requested.emit)
            self.chips_layout.addWidget(chip)

        # 7. Active Media Type Chip: "Media type: Video" (or "Photos", "RAW") - ignore "all"
        media_type = filter_state.get("media_type")
        m_val = str(media_type or "").lower().strip()
        if m_val and m_val not in ("all", "none"):
            lbl_m = QLabel("Media type:", self.chips_widget)
            lbl_m.setStyleSheet("color: #55687C; font-family: 'Segoe UI'; font-size: 8.5pt;")
            self.chips_layout.addWidget(lbl_m)

            if m_val in ("video", "videos"):
                m_label = "Video"
            elif m_val in ("image", "images", "photo", "photos", "picture", "pictures"):
                m_label = "Photos"
            elif m_val in ("raw", "raws"):
                m_label = "RAW"
            else:
                m_label = str(media_type).capitalize()

            chip = FilterChip(m_label, self.chips_widget)
            chip.remove_clicked.connect(self.media_type_clear_requested.emit)
            self.chips_layout.addWidget(chip)

        # 8. Active Orientation Chip: "Orientation: Landscape" / "Orientation: Portrait" - ignore "all"
        orient_val = str(filter_state.get("orientation_filter") or "").lower().strip()
        if orient_val and orient_val not in ("all", "none"):
            lbl_o = QLabel("Orientation:", self.chips_widget)
            lbl_o.setStyleSheet("color: #55687C; font-family: 'Segoe UI'; font-size: 8.5pt;")
            self.chips_layout.addWidget(lbl_o)

            o_label = "Landscape" if orient_val == "landscape" else (
                "Portrait" if orient_val == "portrait" else orient_val.capitalize()
            )
            chip = FilterChip(o_label, self.chips_widget)
            chip.remove_clicked.connect(self.orientation_clear_requested.emit)
            self.chips_layout.addWidget(chip)

    def _resolve_person_name(self, person_id: int, catalog_repo: CatalogRepository) -> str:
        conn = catalog_repo.engine.catalog_conn()
        row = conn.execute("SELECT name FROM people WHERE id = ?;", (person_id,)).fetchone()
        if row:
            return str(row[0])
        return f"Person #{person_id}"

    def _resolve_tag_name(self, tag_id: int, catalog_repo: CatalogRepository) -> str:
        conn = catalog_repo.engine.catalog_conn()
        row = conn.execute("SELECT name FROM tags WHERE id = ?;", (tag_id,)).fetchone()
        if row:
            return str(row[0])
        return f"Tag #{tag_id}"

    # -------------------------------------------------------------------------
    # Pinning & Visibility
    # -------------------------------------------------------------------------
    def is_pinned(self) -> bool:
        return self._is_pinned

    def set_pinned(self, pinned: bool) -> None:
        self._is_pinned = bool(pinned)
        self.btn_pin.blockSignals(True)
        self.btn_pin.setChecked(self._is_pinned)
        self.btn_pin.setIcon(_make_pin_icon(self._is_pinned, 16))
        self.btn_pin.blockSignals(False)

    def _on_pin_toggled(self, checked: bool) -> None:
        self._is_pinned = checked
        self.btn_pin.setIcon(_make_pin_icon(checked, 16))
        self.btn_pin.setToolTip("Unpin search bar" if checked else "Pin search bar (keep open)")
        self.pin_toggled.emit(checked)

    def _on_close_clicked(self) -> None:
        self.set_pinned(False)
        self.txt_search.clear()
        self.close_requested.emit()

    def focus_search_input(self) -> None:
        self.txt_search.setFocus()
        self.txt_search.selectAll()

    def clear_search_input(self) -> None:
        self.txt_search.clear()

    def search_text(self) -> str:
        return self.txt_search.text()

    def _on_search_text_edited(self, _text: str) -> None:
        self._debounce_timer.start()

    def _emit_search_text(self) -> None:
        self.search_text_changed.emit(self.txt_search.text().strip())
