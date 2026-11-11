"""
Immich / Google Photos-Inspired People Directory View for PhotoGallery64.
Displays a rich, dense visual directory grid of all people in the catalog:
- Section 1: Named People (circular avatars, photo count, subtle '+N' suggestion badge)
- Section 2: People to Name with inline clickable 'Add a name' field & autocomplete suggestions
- Real-time search/filter bar, crisp solid context menus (Rename, Merge, Review suggestions, Cover face).
- Guaranteed pixel-perfect vertical column alignment between sections.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QObject, QPoint, QRect, QRunnable, QSettings, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.ui.person_header_bar import _AutoPopupLineEdit, make_circular_avatar
from src.ui.ribbon_toolbar import make_wlpg_icon

logger = logging.getLogger(__name__)


class MergePeopleDialog(QDialog):
    """
    Dialog allowing user to merge one person contact into another or rename.
    Supports typing with instant autocomplete suggestions and a visual candidate picker.
    """

    def __init__(
        self,
        source_person_name: str,
        candidates: list[dict[str, Any]],
        source_person_id: int | None = None,
        source_photo_count: int = 0,
        source_thumb_blob: bytes | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.source_person_name = source_person_name
        self.source_person_id = source_person_id
        self.source_photo_count = source_photo_count
        self.source_thumb_blob = source_thumb_blob
        self.candidates = candidates

        self.setWindowTitle(f"Merge '{source_person_name}'")
        self.setFixedSize(460, 480)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #FFFFFF;
                border-radius: 8px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        # 1. Header with Source Info
        h_src = QHBoxLayout()
        h_src.setSpacing(12)

        lbl_src_av = QLabel(self)
        lbl_src_av.setFixedSize(48, 48)
        lbl_src_av.setPixmap(make_circular_avatar(source_thumb_blob, source_person_name, size=48))
        h_src.addWidget(lbl_src_av)

        v_src_txt = QVBoxLayout()
        v_src_txt.setSpacing(2)
        lbl_title = QLabel(f"Merge <b>{source_person_name}</b> into:", self)
        lbl_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        lbl_title.setStyleSheet("color: #0F172A;")
        v_src_txt.addWidget(lbl_title)

        cnt_str = f"{source_photo_count} photo{'s' if source_photo_count != 1 else ''}"
        lbl_sub = QLabel(f"All photos ({cnt_str}) and face tags will be consolidated.", self)
        lbl_sub.setFont(QFont("Segoe UI", 9))
        lbl_sub.setStyleSheet("color: #64748B;")
        v_src_txt.addWidget(lbl_sub)
        h_src.addLayout(v_src_txt, 1)

        layout.addLayout(h_src)

        # 2. Searchable Name Input (single unified list below)
        lbl_prompt = QLabel("Type to filter contacts or enter a new name:", self)
        lbl_prompt.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        lbl_prompt.setStyleSheet("color: #334155; margin-top: 4px;")
        layout.addWidget(lbl_prompt)

        self.txt_target = QLineEdit(self)
        self.txt_target.setPlaceholderText("Search contacts or type name...")
        self.txt_target.setFont(QFont("Segoe UI", 10))
        self.txt_target.setFixedHeight(34)
        self.txt_target.setStyleSheet(
            """
            QLineEdit {
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 4px 8px;
                background-color: #F8FAFC;
                color: #0F172A;
            }
            QLineEdit:focus {
                border-color: #2563EB;
                background-color: #FFFFFF;
            }
            """
        )
        self.txt_target.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.txt_target)

        # 3. Candidate Unified Contact List with Avatars & Photo Counts
        self.list_candidates = QListWidget(self)
        self.list_candidates.setIconSize(QSize(28, 28))
        self.list_candidates.setStyleSheet(
            """
            QListWidget {
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                background-color: #FFFFFF;
                color: #0F172A;
            }
            QListWidget::item {
                padding: 5px 8px;
                border-bottom: 1px solid #F1F5F9;
                border-radius: 4px;
            }
            QListWidget::item:hover {
                background-color: #F1F5F9;
            }
            QListWidget::item:selected {
                background-color: #EFF6FF;
                color: #1D4ED8;
                border: 1px solid #BFDBFE;
                font-weight: bold;
            }
            """
        )
        self._populate_candidate_list(self.candidates)
        self.list_candidates.itemClicked.connect(self._on_list_item_clicked)
        layout.addWidget(self.list_candidates, 1)

        # 4. Status / Live Preview Label
        self.lbl_preview = QLabel(self)
        self.lbl_preview.setFont(QFont("Segoe UI", 9))
        self.lbl_preview.setWordWrap(True)
        self.lbl_preview.setStyleSheet("color: #475569; padding: 2px;")
        layout.addWidget(self.lbl_preview)

        # 5. Buttons
        h_btn = QHBoxLayout()
        h_btn.setSpacing(8)
        h_btn.addStretch(1)

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setFixedSize(80, 32)
        self.btn_cancel.setStyleSheet(
            """
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                color: #334155;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #F1F5F9;
            }
            """
        )
        self.btn_cancel.clicked.connect(self.reject)
        h_btn.addWidget(self.btn_cancel)

        self.btn_action = QPushButton("Merge", self)
        self.btn_action.setFixedSize(110, 32)
        self.btn_action.setStyleSheet(
            """
            QPushButton {
                background-color: #2563EB;
                border: none;
                border-radius: 6px;
                color: #FFFFFF;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1D4ED8;
            }
            QPushButton:disabled {
                background-color: #94A3B8;
                color: #E2E8F0;
            }
            """
        )
        self.btn_action.setEnabled(False)
        self.btn_action.clicked.connect(self.accept)
        h_btn.addWidget(self.btn_action)

        layout.addLayout(h_btn)

        # Initial state update
        self._update_preview()

    def _populate_candidate_list(self, items: list[dict[str, Any]]) -> None:
        self.list_candidates.clear()
        for c in items:
            cnt = c.get("photo_count", c.get("confirmed_count", 0))
            name = c.get("name", "")
            icon = QIcon(make_circular_avatar(c.get("thumbnail_blob"), name, size=28))
            item = QListWidgetItem(icon, f"  {name}   ({cnt} photo{'s' if cnt != 1 else ''})")
            item.setSizeHint(QSize(400, 38))
            item.setData(Qt.ItemDataRole.UserRole, c)
            self.list_candidates.addItem(item)

    def _on_list_item_clicked(self, item: QListWidgetItem) -> None:
        c = item.data(Qt.ItemDataRole.UserRole)
        if c:
            self.txt_target.blockSignals(True)
            self.txt_target.setText(c["name"])
            self.txt_target.blockSignals(False)
            self._update_preview()

    def _on_text_changed(self, text: str) -> None:
        query = text.strip().lower()
        matched_item = None
        for i in range(self.list_candidates.count()):
            itm = self.list_candidates.item(i)
            c = itm.data(Qt.ItemDataRole.UserRole)
            matches = not query or (c and query in c["name"].lower())
            itm.setHidden(not matches)
            if c and c["name"].lower() == query:
                matched_item = itm

        if matched_item:
            self.list_candidates.setCurrentItem(matched_item)
        else:
            self.list_candidates.clearSelection()

        self._update_preview()

    def _update_preview(self) -> None:
        txt = self.txt_target.text().strip()
        if not txt:
            self.lbl_preview.setText("Select a contact to merge into.")
            self.btn_action.setEnabled(False)
            self.btn_action.setText("Merge")
            return

        if txt.lower() == self.source_person_name.lower():
            self.lbl_preview.setText("<span style='color: #DC2626;'>Cannot merge a person into themselves.</span>")
            self.btn_action.setEnabled(False)
            return

        matched = next((c for c in self.candidates if c["name"].lower() == txt.lower()), None)
        if matched:
            cnt = matched.get("photo_count", matched.get("confirmed_count", 0))
            self.lbl_preview.setText(
                f"Consolidating <b>{self.source_person_name}</b> into <b>{matched['name']}</b> ({cnt} photos)."
            )
            self.btn_action.setText("Merge")
            self.btn_action.setEnabled(True)
        else:
            self.lbl_preview.setText(
                f"<b>'{txt}'</b> is a new name. This will rename <b>{self.source_person_name}</b>."
            )
            self.btn_action.setText("Rename")
            self.btn_action.setEnabled(True)

    def target_person_id(self) -> int | None:
        txt = self.txt_target.text().strip()
        matched = next((c for c in self.candidates if c["name"].lower() == txt.lower()), None)
        return matched["id"] if matched else None

    def target_person_name(self) -> str:
        return self.txt_target.text().strip()

    def is_rename(self) -> bool:
        return self.target_person_id() is None


_MergePeopleDialog = MergePeopleDialog  # Backward compatibility alias


class _MergeConfirmDialog(QDialog):
    """
    Visual comparison confirmation modal for merging people (Immich / Google Photos inspired).
    Shows the two face circular avatars side-by-side:
    'Are these the same person?'
    'These photos will be merged together into [Name]'
    [ No ]  [ Yes, Merge ]
    """

    def __init__(
        self,
        new_face_blob: bytes | None,
        existing_person_name: str,
        existing_thumb_blob: bytes | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Merge People")
        self.setFixedSize(380, 240)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #FFFFFF;
                border-radius: 12px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 1. Avatars Row
        h_avatars = QHBoxLayout()
        h_avatars.setSpacing(16)
        h_avatars.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl_av1 = QLabel(self)
        lbl_av1.setFixedSize(72, 72)
        lbl_av1.setPixmap(make_circular_avatar(new_face_blob, "?", size=72))
        h_avatars.addWidget(lbl_av1)

        lbl_arrow = QLabel("➔", self)
        lbl_arrow.setStyleSheet("font-size: 16px; color: #94A3B8; font-weight: bold;")
        h_avatars.addWidget(lbl_arrow)

        lbl_av2 = QLabel(self)
        lbl_av2.setFixedSize(72, 72)
        lbl_av2.setPixmap(make_circular_avatar(existing_thumb_blob, existing_person_name, size=72))
        h_avatars.addWidget(lbl_av2)

        layout.addLayout(h_avatars)

        # 2. Text
        lbl_title = QLabel("Are these the same person?", self)
        lbl_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("color: #0F172A;")
        layout.addWidget(lbl_title)

        lbl_desc = QLabel(f"These photos will be merged into <b>{existing_person_name}</b>.", self)
        lbl_desc.setFont(QFont("Segoe UI", 9))
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_desc.setStyleSheet("color: #64748B;")
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        layout.addSpacing(4)

        # 3. Buttons (No / Yes, Merge)
        h_btn = QHBoxLayout()
        h_btn.setSpacing(10)
        h_btn.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_no = QPushButton("No", self)
        btn_no.setFixedSize(85, 32)
        btn_no.setStyleSheet(
            """
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 16px;
                color: #334155;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #F1F5F9;
            }
            """
        )
        btn_no.clicked.connect(self.reject)
        h_btn.addWidget(btn_no)

        btn_yes = QPushButton("Yes, Merge", self)
        btn_yes.setFixedSize(95, 32)
        btn_yes.setStyleSheet(
            """
            QPushButton {
                background-color: #2563EB;
                border: none;
                border-radius: 16px;
                color: #FFFFFF;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #1D4ED8;
            }
            """
        )
        btn_yes.clicked.connect(self.accept)
        h_btn.addWidget(btn_yes)

        layout.addLayout(h_btn)


class _NamedPersonCard(QFrame):
    """108x108 circular avatar card for named people (Immich-styled with smart elision)."""

    clicked = Signal(int)                   # person_id
    rename_requested = Signal(int, str)     # (person_id, current_name)
    merge_requested = Signal(int, str)      # (person_id, name)
    review_requested = Signal(int)          # person_id
    cover_requested = Signal(int)           # person_id

    CARD_WIDTH = 162
    CARD_HEIGHT = 198

    def __init__(
        self,
        person_id: int,
        name: str,
        photo_count: int,
        suggestion_count: int,
        thumb_blob: bytes | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.person_id = person_id
        self.person_name = name
        self.photo_count = photo_count
        self.suggestion_count = suggestion_count
        self.thumb_blob = thumb_blob

        self.setFixedSize(self.CARD_WIDTH, self.CARD_HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("NamedPersonCard")
        self.setStyleSheet(
            """
            #NamedPersonCard {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 12px;
            }
            #NamedPersonCard:hover {
                background-color: #F1F5F9;
            }
            """
        )

        v_layout = QVBoxLayout(self)
        v_layout.setContentsMargins(6, 8, 6, 8)
        v_layout.setSpacing(4)
        v_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 1. Avatar Container with Suggestion Badge
        self.avatar_container = QWidget(self)
        self.avatar_container.setFixedSize(130, 130)
        av_layout = QVBoxLayout(self.avatar_container)
        av_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_avatar = QLabel(self.avatar_container)
        self.lbl_avatar.setFixedSize(130, 130)
        self.lbl_avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = make_circular_avatar(thumb_blob, name, size=130)
        self.lbl_avatar.setPixmap(pix)
        av_layout.addWidget(self.lbl_avatar)

        # Pending suggestions badge
        if suggestion_count > 0:
            self.lbl_badge = QLabel(f"+{suggestion_count}", self.avatar_container)
            self.lbl_badge.setStyleSheet(
                "QLabel { background-color: #F59E0B; color: #FFFFFF; font-weight: bold; "
                "font-size: 9px; border-radius: 8px; padding: 1px 5px; border: 1.5px solid #FFFFFF; }"
            )
            self.lbl_badge.adjustSize()
            self.lbl_badge.move(self.avatar_container.width() - self.lbl_badge.width() - 2, 2)
            self.lbl_badge.show()

        v_layout.addWidget(self.avatar_container, 0, Qt.AlignmentFlag.AlignCenter)

        # 2. Name Label with smart elision (starts with name, ends with ...)
        name_font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        fm = QFontMetrics(name_font)
        elided_name = fm.elidedText(name, Qt.TextElideMode.ElideRight, self.CARD_WIDTH - 10)

        self.lbl_name = QLabel(elided_name, self)
        self.lbl_name.setFont(name_font)
        self.lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_name.setStyleSheet("color: #0F172A;")
        self.lbl_name.setToolTip(name)
        v_layout.addWidget(self.lbl_name)

        # 3. Photo Count Label
        cnt_str = f"{photo_count:,} photo{'s' if photo_count != 1 else ''}"
        self.lbl_count = QLabel(cnt_str, self)
        self.lbl_count.setFont(QFont("Segoe UI", 8))
        self.lbl_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_count.setStyleSheet("color: #64748B;")
        v_layout.addWidget(self.lbl_count)

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.person_id)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event: Any) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            """
            QMenu {
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 4px;
                color: #0F172A;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
            }
            QMenu::item {
                padding: 6px 20px 6px 12px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #F1F5F9;
                color: #0F172A;
            }
            QMenu::separator {
                height: 1px;
                background-color: #E2E8F0;
                margin: 4px 6px;
            }
            """
        )
        act_open = menu.addAction(f"Open {self.person_name}'s photos")
        act_rename = menu.addAction("Rename person...")
        act_merge = menu.addAction("Merge with another person...")
        act_review = menu.addAction(f"Review suggestions ({self.suggestion_count})")
        menu.addSeparator()
        act_cover = menu.addAction("Set as cover face")

        chosen = menu.exec(event.globalPos())
        if chosen == act_open:
            self.clicked.emit(self.person_id)
        elif chosen == act_rename:
            self.rename_requested.emit(self.person_id, self.person_name)
        elif chosen == act_merge:
            self.merge_requested.emit(self.person_id, self.person_name)
        elif chosen == act_review:
            self.review_requested.emit(self.person_id)
        elif chosen == act_cover:
            self.cover_requested.emit(self.person_id)


class _UnnamedPersonCard(QFrame):
    """
    Immich-style avatar card for unnamed people with an inline 'Add a name' field.
    Provides autocomplete suggestions matching existing people.
    Left click on the avatar / card navigates to view the person's photos.
    """

    clicked = Signal(list, object)       # (face_ids, thumb_blob)
    named = Signal(str, list)            # (name, face_ids)
    ignore_requested = Signal(list)      # face_ids

    CARD_WIDTH = 162
    CARD_HEIGHT = 198

    def __init__(
        self,
        cluster_name: str,
        face_ids: list[int],
        face_count: int,
        thumb_blob: bytes | None,
        existing_names: list[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.cluster_name = cluster_name
        self.face_ids = face_ids
        self.face_count = face_count
        self.thumb_blob = thumb_blob

        self.setFixedSize(self.CARD_WIDTH, self.CARD_HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("UnnamedPersonCard")
        self.setStyleSheet(
            """
            #UnnamedPersonCard {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 12px;
            }
            #UnnamedPersonCard:hover {
                background-color: #F1F5F9;
            }
            """
        )

        v_layout = QVBoxLayout(self)
        v_layout.setContentsMargins(6, 8, 6, 8)
        v_layout.setSpacing(4)
        v_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 1. Avatar (130x130 circular face photo)
        self.lbl_avatar = QLabel(self)
        self.lbl_avatar.setFixedSize(130, 130)
        self.lbl_avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = make_circular_avatar(thumb_blob, "?", size=130)
        self.lbl_avatar.setPixmap(pix)
        v_layout.addWidget(self.lbl_avatar, 0, Qt.AlignmentFlag.AlignCenter)

        # 2. Inline "Add a name" input (Immich pill style with instant click-to-suggest)
        self.txt_name = _AutoPopupLineEdit(self)
        self.txt_name.setPlaceholderText("Add a name")
        self.txt_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.txt_name.setFont(QFont("Segoe UI", 9))
        self.txt_name.setFixedWidth(146)
        self.txt_name.setStyleSheet(
            """
            QLineEdit {
                border: 1px solid #E2E8F0;
                border-radius: 11px;
                padding: 2px 6px;
                background-color: #FFFFFF;
                color: #0F172A;
            }
            QLineEdit:hover {
                border-color: #94A3B8;
            }
            QLineEdit:focus {
                border-color: #2563EB;
                background-color: #FFFFFF;
                font-weight: bold;
            }
            """
        )

        # Autocomplete suggestions matching existing people
        if existing_names:
            completer = QCompleter(existing_names, self)
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            self.txt_name.setCompleter(completer)
            completer.activated.connect(self._on_completer_activated)

        self.txt_name.returnPressed.connect(self._on_name_entered)
        v_layout.addWidget(self.txt_name)

        # 3. Photo count label
        cnt_str = f"{face_count} photo{'s' if face_count != 1 else ''}"
        self.lbl_count = QLabel(cnt_str, self)
        self.lbl_count.setFont(QFont("Segoe UI", 8))
        self.lbl_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_count.setStyleSheet("color: #94A3B8;")
        v_layout.addWidget(self.lbl_count)

    def _on_completer_activated(self, text: str) -> None:
        self.txt_name.setText(text)
        self._on_name_entered()

    def _on_name_entered(self) -> None:
        name = self.txt_name.text().strip()
        if name:
            self.named.emit(name, self.face_ids)

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.txt_name.underMouse():
                self.clicked.emit(self.face_ids, self.thumb_blob)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event: Any) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            """
            QMenu {
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 4px;
                color: #0F172A;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
            }
            QMenu::item {
                padding: 6px 20px 6px 12px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #F1F5F9;
                color: #0F172A;
            }
            QMenu::separator {
                height: 1px;
                background-color: #E2E8F0;
                margin: 4px 6px;
            }
            """
        )
        act_view = menu.addAction("View photos")
        act_name = menu.addAction("Add name...")
        menu.addSeparator()
        act_ignore = menu.addAction("Hide this person (don't show again)")

        chosen = menu.exec(event.globalPos())
        if chosen == act_view:
            self.clicked.emit(self.face_ids, self.thumb_blob)
        elif chosen == act_name:
            self.txt_name.setFocus()
            self.txt_name.selectAll()
        elif chosen == act_ignore:
            self.ignore_requested.emit(self.face_ids)


_UnnamedClusterCard = _UnnamedPersonCard  # Backward compatibility alias


class _UnnamedClusterSignals(QObject):
    clusters_ready = Signal(int, list)


class _UnnamedClusterWorker(QRunnable):
    """Background worker that computes unnamed face clusters and representative crops off the GUI thread."""

    def __init__(
        self,
        repo: CatalogRepository,
        gen_id: int,
        signals: _UnnamedClusterSignals,
        thumb_repo: ThumbnailRepository | None = None,
        min_confidence: float = 0.52,
    ) -> None:
        super().__init__()
        self.repo = repo
        self.gen_id = gen_id
        self.signals = signals
        self.thumb_repo = thumb_repo
        self.min_confidence = float(min_confidence)

    def run(self) -> None:
        try:
            clusters = self.repo.get_unnamed_face_clusters(min_confidence=self.min_confidence)
            if clusters and self.thumb_repo is not None:
                rep_fids = [
                    int(c["representative_face_id"])
                    for c in clusters
                    if c.get("representative_face_id")
                ]
                thumbs = self.thumb_repo.get_face_thumbnails_batch(rep_fids) if rep_fids else {}
                for c in clusters:
                    rep_fid = c.get("representative_face_id")
                    c["thumb_blob"] = thumbs.get(int(rep_fid)) if rep_fid else None
            self.signals.clusters_ready.emit(self.gen_id, clusters)
        except Exception:
            pass
        finally:
            import threading
            if threading.current_thread() is not threading.main_thread():
                self.repo.engine.close_thread_connections()


class PeopleDirectoryView(QWidget):
    """
    Main People Directory visual directory view (Immich / Google Photos inspired).
    Displays Section 1 (Named People) and Section 2 (People to Name).
    """

    person_selected = Signal(int)              # Navigates to person's photo timeline
    unnamed_selected = Signal(list, object)    # (face_ids, thumb_blob)
    review_suggestions_requested = Signal()     # Opens BatchPeopleView
    directory_updated = Signal()

    def __init__(
        self,
        engine: DualDatabaseEngine,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.repo = CatalogRepository(engine)
        self.thumb_repo = ThumbnailRepository(engine)
        self._all_named_people: list[dict[str, Any]] = []
        self._all_unnamed_clusters: list[dict[str, Any]] = []
        self._cluster_gen_id: int = 0
        self._cluster_signals = _UnnamedClusterSignals()
        self._cluster_signals.clusters_ready.connect(self._on_unnamed_clusters_ready)

        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setObjectName("PeopleDirectoryView")
        self.setStyleSheet("#PeopleDirectoryView { background-color: #FFFFFF; }")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---------------------------------------------------------------------
        # 1. Header Toolbar
        # ---------------------------------------------------------------------
        self.header_bar = QFrame(self)
        header_bar = self.header_bar
        header_bar.setFixedHeight(56)
        header_bar.setStyleSheet("background-color: #FFFFFF; border-bottom: 1px solid #E2E8F0;")
        h_layout = QHBoxLayout(header_bar)
        h_layout.setContentsMargins(20, 8, 20, 8)
        h_layout.setSpacing(14)

        # Title & Count
        self.lbl_title = QLabel("People", header_bar)
        self.lbl_title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        self.lbl_title.setStyleSheet("color: #0F172A;")
        h_layout.addWidget(self.lbl_title)

        self.lbl_total_count = QLabel("(0)", header_bar)
        self.lbl_total_count.setFont(QFont("Segoe UI", 12))
        self.lbl_total_count.setStyleSheet("color: #64748B;")
        h_layout.addWidget(self.lbl_total_count)

        h_layout.addSpacing(16)

        # Search Bar
        self.txt_search = QLineEdit(header_bar)
        self.txt_search.setPlaceholderText("🔍 Search people...")
        self.txt_search.setFixedWidth(240)
        self.txt_search.setStyleSheet(
            "QLineEdit { border: 1px solid #CBD5E1; border-radius: 16px; padding: 5px 12px; background: #F8FAFC; color: #0F172A; }"
            "QLineEdit:focus { border-color: #3B82F6; background: #FFFFFF; }"
        )
        self.txt_search.textChanged.connect(self._filter_cards)
        h_layout.addWidget(self.txt_search)

        h_layout.addStretch(1)

        # Batch Review Suggestions Button
        self.btn_review_suggestions = QPushButton("Review Face Suggestions", header_bar)
        self.btn_review_suggestions.setObjectName("btnReviewSuggestions")
        self.btn_review_suggestions.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_review_suggestions.setIcon(make_wlpg_icon("batch_people", 16))
        self.btn_review_suggestions.setIconSize(QSize(16, 16))
        self.btn_review_suggestions.setStyleSheet(
            """
            QPushButton#btnReviewSuggestions {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2E79BA, stop:1 #1E5C94);
                border: 1px solid #184E80;
                border-radius: 4px;
                color: #FFFFFF;
                font-family: 'Segoe UI';
                font-size: 9pt;
                font-weight: 600;
                padding: 5px 14px;
            }
            QPushButton#btnReviewSuggestions:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3D8CCD, stop:1 #256BA6);
                border-color: #1F5F9E;
            }
            QPushButton#btnReviewSuggestions:pressed {
                background: #184E80;
            }
            """
        )
        self.btn_review_suggestions.clicked.connect(self.review_suggestions_requested.emit)
        h_layout.addWidget(self.btn_review_suggestions)

        # Refresh Button
        self.btn_refresh = QPushButton("Refresh", header_bar)
        self.btn_refresh.setObjectName("btnRefreshDirectory")
        self.btn_refresh.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_refresh.setIcon(make_wlpg_icon("rotate_right", 14))
        self.btn_refresh.setIconSize(QSize(14, 14))
        self.btn_refresh.setStyleSheet(
            """
            QPushButton#btnRefreshDirectory {
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 4px;
                color: #334155;
                font-family: 'Segoe UI';
                font-size: 9pt;
                padding: 5px 12px;
            }
            QPushButton#btnRefreshDirectory:hover {
                background-color: #F8FAFC;
                border-color: #94A3B8;
                color: #1E293B;
            }
            """
        )
        self.btn_refresh.clicked.connect(self.refresh_directory)
        h_layout.addWidget(self.btn_refresh)

        main_layout.addWidget(header_bar)

        # ---------------------------------------------------------------------
        # 2. Scrollable Directory Area (with minimalist vertical scrollbar)
        # ---------------------------------------------------------------------
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet(
            """
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                border: none;
                background: transparent;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #CBD5E1;
                min-height: 28px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #94A3B8;
            }
            QScrollBar::handle:vertical:pressed {
                background: #64748B;
            }
            QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {
                border: none;
                background: transparent;
                height: 0px;
                width: 0px;
            }
            QScrollBar::sub-page:vertical, QScrollBar::add-page:vertical {
                border: none;
                background: transparent;
            }
            QScrollBar:horizontal {
                border: none;
                background: transparent;
                height: 0px;
                width: 0px;
            }
            QScrollBar::handle:horizontal, QScrollBar::sub-line:horizontal, QScrollBar::add-line:horizontal {
                border: none;
                background: transparent;
                width: 0px;
                height: 0px;
            }
            QScrollBar::sub-page:horizontal, QScrollBar::add-page:horizontal {
                border: none;
                background: transparent;
            }
            QAbstractScrollArea::corner {
                background: transparent;
                border: none;
            }
            """
        )

        self.scroll_content = QWidget()
        self.scroll_content.setStyleSheet("background-color: transparent;")
        self.content_layout = QVBoxLayout(self.scroll_content)
        self.content_layout.setContentsMargins(20, 16, 20, 20)
        self.content_layout.setSpacing(22)

        # Section 1: Named People
        self.sec1_container = QWidget(self.scroll_content)
        v_sec1 = QVBoxLayout(self.sec1_container)
        v_sec1.setContentsMargins(0, 0, 0, 0)
        v_sec1.setSpacing(10)

        self.lbl_sec1_title = QLabel("Named People", self.sec1_container)
        self.lbl_sec1_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.lbl_sec1_title.setStyleSheet("color: #1E293B;")
        v_sec1.addWidget(self.lbl_sec1_title)

        self.grid_named = QGridLayout()
        self.grid_named.setSpacing(10)
        self.grid_named.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        v_sec1.addLayout(self.grid_named)

        self.content_layout.addWidget(self.sec1_container)

        # Section 2: People to Name ("Add a name")
        self.sec2_container = QWidget(self.scroll_content)
        v_sec2 = QVBoxLayout(self.sec2_container)
        v_sec2.setContentsMargins(0, 0, 0, 0)
        v_sec2.setSpacing(10)

        self.lbl_sec2_title = QLabel("People to Name", self.sec2_container)
        self.lbl_sec2_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.lbl_sec2_title.setStyleSheet("color: #1E293B;")
        v_sec2.addWidget(self.lbl_sec2_title)

        self.grid_unnamed = QGridLayout()
        self.grid_unnamed.setSpacing(10)
        self.grid_unnamed.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        v_sec2.addLayout(self.grid_unnamed)

        self.content_layout.addWidget(self.sec2_container)

        # Empty State Label
        self.lbl_empty = QLabel("No people found in your photo library.\n\nTag faces in photos or enable face detection in Options.", self.scroll_content)
        self.lbl_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_empty.setStyleSheet("color: #64748B; font-size: 14px; margin-top: 60px;")
        self.lbl_empty.hide()
        self.content_layout.addWidget(self.lbl_empty)

        self.content_layout.addStretch(1)
        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area, 1)

    # -------------------------------------------------------------------------
    # Directory Data Refresh & Rendering
    # -------------------------------------------------------------------------
    def refresh_directory(self, recluster_unnamed: bool = True) -> None:
        """Fetches named people immediately (<60ms) and computes unnamed clusters in background."""
        try:
            min_conf = float(QSettings("PhotoGallery64", "PhotoGallery64").value("face_match_threshold", 0.52))
        except Exception:
            min_conf = 0.52
        self._all_named_people = self.repo.get_named_people_directory_data(min_confidence=min_conf)

        total_cnt = len(self._all_named_people)
        self.lbl_total_count.setText(f"({total_cnt:,})")

        # Update Review suggestions button badge & styling
        total_suggestions = sum(p.get("suggestion_count", 0) for p in self._all_named_people)
        if total_suggestions > 0:
            self.btn_review_suggestions.setText(f"Review Suggestions ({total_suggestions})")
            self.btn_review_suggestions.setIcon(make_wlpg_icon("batch_people", 16))
            self.btn_review_suggestions.setStyleSheet(
                """
                QPushButton#btnReviewSuggestions {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2E79BA, stop:1 #1E5C94);
                    border: 1px solid #184E80;
                    border-radius: 4px;
                    color: #FFFFFF;
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                    font-weight: 600;
                    padding: 5px 14px;
                }
                QPushButton#btnReviewSuggestions:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3D8CCD, stop:1 #256BA6);
                    border-color: #1F5F9E;
                }
                QPushButton#btnReviewSuggestions:pressed {
                    background: #184E80;
                }
                """
            )
        else:
            self.btn_review_suggestions.setText("Review Face Suggestions")
            self.btn_review_suggestions.setIcon(make_wlpg_icon("batch_people", 16))
            self.btn_review_suggestions.setStyleSheet(
                """
                QPushButton#btnReviewSuggestions {
                    background-color: #FFFFFF;
                    border: 1px solid #CBD5E1;
                    border-radius: 4px;
                    color: #475569;
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                    font-weight: normal;
                    padding: 5px 14px;
                }
                QPushButton#btnReviewSuggestions:hover {
                    background-color: #F8FAFC;
                    border-color: #94A3B8;
                    color: #1E293B;
                }
                """
            )

        # Update Section 1 immediately; Section 2 either stays as-is or updates once worker finishes
        self._render_named_grid()
        if not recluster_unnamed:
            self._render_unnamed_grid()
        self._update_empty_state()

        if recluster_unnamed:
            self._cluster_gen_id += 1
            worker = _UnnamedClusterWorker(
                self.repo,
                self._cluster_gen_id,
                self._cluster_signals,
                thumb_repo=self.thumb_repo,
                min_confidence=min_conf,
            )
            QThreadPool.globalInstance().start(worker)

    def _on_unnamed_clusters_ready(self, gen_id: int, clusters: list[dict[str, Any]]) -> None:
        if gen_id != self._cluster_gen_id:
            return
        self._all_unnamed_clusters = clusters
        self._render_unnamed_grid()
        self._update_empty_state()

    def reload_directory(self) -> None:
        """Alias for refresh_directory."""
        self.refresh_directory()

    def _calc_cols(self) -> int:
        # Dynamic column calculation (card width 162 + spacing 12 = 174)
        # Always subtract 10px to account for vertical scrollbar and prevent horizontal scrollbar triggering
        vw = self.scroll_area.viewport().width() if self.scroll_area.viewport().width() > 100 else self.width()
        avail_w = max(200, vw - 10)
        return max(2, (avail_w - 40) // 174)

    def _update_empty_state(self) -> None:
        search_filter = self.txt_search.text().strip().lower()
        has_named = any(
            not search_filter or search_filter in p["name"].lower()
            for p in self._all_named_people
        )
        has_any = bool(has_named or (self._all_unnamed_clusters and not search_filter))
        self.lbl_empty.setVisible(not has_any)

    def _render_grids(self) -> None:
        self._render_named_grid()
        self._render_unnamed_grid()
        self._update_empty_state()

    def _render_named_grid(self) -> None:
        search_filter = self.txt_search.text().strip().lower()
        cols = self._calc_cols()
        self._last_rendered_cols = cols

        self.sec1_container.setUpdatesEnabled(False)
        try:
            self._clear_layout(self.grid_named)

            filtered_named = [
                p for p in self._all_named_people
                if not search_filter or search_filter in p["name"].lower()
            ]
            self.lbl_sec1_title.setText(f"Named People ({len(filtered_named)})")
            self.sec1_container.setVisible(bool(filtered_named))

            for idx, p in enumerate(filtered_named):
                card = _NamedPersonCard(
                    person_id=p["id"],
                    name=p["name"],
                    photo_count=p["photo_count"],
                    suggestion_count=p["suggestion_count"],
                    thumb_blob=p["thumbnail_blob"],
                    parent=self.sec1_container,
                )
                if getattr(self, "_is_dark", False):
                    card.setStyleSheet(
                        """
                        #NamedPersonCard {
                            background-color: transparent;
                            border: 1px solid transparent;
                            border-radius: 12px;
                        }
                        #NamedPersonCard:hover {
                            background-color: #242A36;
                        }
                        """
                    )
                    card.lbl_name.setStyleSheet("color: #E2E8F0;")
                    card.lbl_count.setStyleSheet("color: #94A3B8;")
                card.clicked.connect(self.person_selected.emit)
                card.rename_requested.connect(self._on_rename_person)
                card.merge_requested.connect(self._on_merge_person)
                card.review_requested.connect(self._on_review_person)
                card.cover_requested.connect(self._on_set_cover_face)
                row = idx // cols
                col = idx % cols
                self.grid_named.addWidget(card, row, col)

            # Enforce exact column width alignment across both grids
            for c in range(cols):
                self.grid_named.setColumnMinimumWidth(c, _NamedPersonCard.CARD_WIDTH)
        finally:
            self.sec1_container.setUpdatesEnabled(True)

    def _render_unnamed_grid(self) -> None:
        search_filter = self.txt_search.text().strip().lower()
        cols = self._calc_cols()
        self._last_rendered_cols = cols

        self.sec2_container.setUpdatesEnabled(False)
        try:
            self._clear_layout(self.grid_unnamed)

            self.sec2_container.setVisible(bool(self._all_unnamed_clusters) and not search_filter)
            if self._all_unnamed_clusters and not search_filter:
                self.lbl_sec2_title.setText(f"People to Name ({len(self._all_unnamed_clusters)})")
                existing_names = [p["name"] for p in self._all_named_people if p.get("name")]

                # Batch-fetch any missing representative blobs once instead of querying per-card
                missing_rep_fids = [
                    int(c["representative_face_id"])
                    for c in self._all_unnamed_clusters
                    if c.get("representative_face_id") and "thumb_blob" not in c and self.thumb_repo
                ]
                fallback_thumbs = (
                    self.thumb_repo.get_face_thumbnails_batch(missing_rep_fids)
                    if missing_rep_fids and self.thumb_repo
                    else {}
                )

                for idx, c in enumerate(self._all_unnamed_clusters):
                    rep_fid = c.get("representative_face_id")
                    if "thumb_blob" in c:
                        thumb_blob = c["thumb_blob"]
                    elif rep_fid:
                        thumb_blob = fallback_thumbs.get(int(rep_fid))
                        c["thumb_blob"] = thumb_blob
                    else:
                        thumb_blob = None

                    card_names = list(existing_names)
                    sugg_nm = c.get("suggested_person_name")
                    if sugg_nm and sugg_nm in card_names:
                        card_names.remove(sugg_nm)
                        card_names.insert(0, sugg_nm)
                    elif sugg_nm:
                        card_names.insert(0, sugg_nm)

                    card_u = _UnnamedPersonCard(
                        cluster_name=c["cluster_name"],
                        face_ids=c["face_ids"],
                        face_count=c["face_count"],
                        thumb_blob=thumb_blob,
                        existing_names=card_names,
                        parent=self.sec2_container,
                    )
                    if getattr(self, "_is_dark", False):
                        card_u.setStyleSheet(
                            """
                            #UnnamedPersonCard {
                                background-color: transparent;
                                border: 1px solid transparent;
                                border-radius: 12px;
                            }
                            #UnnamedPersonCard:hover {
                                background-color: #242A36;
                            }
                            """
                        )
                        card_u.txt_name.setStyleSheet(
                            """
                            QLineEdit {
                                border: 1px solid #333A45;
                                border-radius: 11px;
                                padding: 2px 6px;
                                background-color: #14171B;
                                color: #E2E8F0;
                            }
                            QLineEdit:hover {
                                border-color: #475569;
                            }
                            QLineEdit:focus {
                                border-color: #3B82F6;
                                background-color: #1B1E24;
                                font-weight: bold;
                            }
                            """
                        )
                    card_u.clicked.connect(lambda fids, tb: self.unnamed_selected.emit(fids, tb))
                    card_u.named.connect(self._on_name_cluster)
                    card_u.ignore_requested.connect(self._on_ignore_cluster_faces)
                    row = idx // cols
                    col = idx % cols
                    self.grid_unnamed.addWidget(card_u, row, col)

                for c in range(cols):
                    self.grid_unnamed.setColumnMinimumWidth(c, _UnnamedPersonCard.CARD_WIDTH)
        finally:
            self.sec2_container.setUpdatesEnabled(True)

    def _clear_layout(self, layout: QGridLayout) -> None:
        while layout.count() > 0:
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _filter_cards(self, text: str) -> None:
        self._render_grids()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if self._calc_cols() != getattr(self, "_last_rendered_cols", -1):
            self._render_grids()

    def set_dark_mode(self, is_dark: bool) -> None:
        """Applies Dark Studio or standard light styling to PeopleDirectoryView."""
        self._is_dark = bool(is_dark)
        if is_dark:
            self.setStyleSheet("#PeopleDirectoryView { background-color: #121417; }")
            if hasattr(self, "header_bar"):
                self.header_bar.setStyleSheet("background-color: #1B1E24; border-bottom: 1px solid #2D323B;")
            self.lbl_title.setStyleSheet("color: #E2E8F0;")
            self.lbl_total_count.setStyleSheet("color: #94A3B8;")
            self.txt_search.setStyleSheet(
                "QLineEdit { border: 1px solid #333A45; border-radius: 16px; padding: 5px 12px; background: #14171B; color: #E2E8F0; }"
                "QLineEdit:focus { border-color: #3B82F6; background: #1B1E24; }"
            )
            self.btn_refresh.setStyleSheet(
                """
                QPushButton#btnRefreshDirectory {
                    background-color: #242A36;
                    border: 1px solid #3E4756;
                    border-radius: 4px;
                    color: #E2E8F0;
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                    padding: 5px 12px;
                }
                QPushButton#btnRefreshDirectory:hover {
                    background-color: #2D3542;
                    border-color: #3B82F6;
                    color: #FFFFFF;
                }
                """
            )
            self.lbl_sec1_title.setStyleSheet("color: #E2E8F0;")
            self.lbl_sec2_title.setStyleSheet("color: #E2E8F0;")
            self.lbl_empty.setStyleSheet("color: #94A3B8; font-size: 14px; margin-top: 60px;")
        else:
            self.setStyleSheet("#PeopleDirectoryView { background-color: #FFFFFF; }")
            if hasattr(self, "header_bar"):
                self.header_bar.setStyleSheet("background-color: #FFFFFF; border-bottom: 1px solid #E2E8F0;")
            self.lbl_title.setStyleSheet("color: #0F172A;")
            self.lbl_total_count.setStyleSheet("color: #64748B;")
            self.txt_search.setStyleSheet(
                "QLineEdit { border: 1px solid #CBD5E1; border-radius: 16px; padding: 5px 12px; background: #F8FAFC; color: #0F172A; }"
                "QLineEdit:focus { border-color: #3B82F6; background: #FFFFFF; }"
            )
            self.btn_refresh.setStyleSheet(
                """
                QPushButton#btnRefreshDirectory {
                    background-color: #FFFFFF;
                    border: 1px solid #CBD5E1;
                    border-radius: 4px;
                    color: #334155;
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                    padding: 5px 12px;
                }
                QPushButton#btnRefreshDirectory:hover {
                    background-color: #F8FAFC;
                    border-color: #94A3B8;
                    color: #1E293B;
                }
                """
            )
            self.lbl_sec1_title.setStyleSheet("color: #1E293B;")
            self.lbl_sec2_title.setStyleSheet("color: #1E293B;")
            self.lbl_empty.setStyleSheet("color: #64748B; font-size: 14px; margin-top: 60px;")
        if self.isVisible():
            self._render_grids()

    # -------------------------------------------------------------------------
    # Actions: Name Cluster, Rename, Merge, Review Suggestions
    # -------------------------------------------------------------------------
    def _on_name_cluster(self, name: str, face_ids: list[int]) -> None:
        clean_name = name.strip()
        if not clean_name or not face_ids:
            return

        # Check if clean_name matches an existing person (case-insensitive)
        matched_person = None
        for p in self._all_named_people:
            if p["name"].lower() == clean_name.lower():
                matched_person = p
                break

        if matched_person is not None:
            # Show Immich merge confirmation dialog
            new_thumb = self.thumb_repo.get_face_thumbnail(face_ids[0]) if face_ids else None
            dlg = _MergeConfirmDialog(
                new_face_blob=new_thumb,
                existing_person_name=matched_person["name"],
                existing_thumb_blob=matched_person.get("thumbnail_blob"),
                parent=self,
            )
            if dlg.exec() != QDialog.Accepted:
                return
            p_id = matched_person["id"]
        else:
            p_id = self.repo.upsert_person(clean_name)

        self.repo.confirm_faces(face_ids, p_id)

        # Update centroid / sub-centroids
        self.repo.recalculate_person_centroid(p_id)

        # Set cover face if person has none
        try:
            if matched_person is None or not matched_person.get("thumbnail_blob"):
                thumb = self.thumb_repo.get_face_thumbnail(face_ids[0]) if (self.thumb_repo and face_ids) else None
                if thumb:
                    self.repo.set_person_thumbnail_blob(p_id, thumb)
        except Exception:
            pass

        self._prune_unnamed_cluster_faces(face_ids)
        self.refresh_directory(recluster_unnamed=False)
        self.directory_updated.emit()

    def _prune_unnamed_cluster_faces(self, face_ids: list[int]) -> None:
        """Removes confirmed/ignored face_ids from cached unnamed clusters in memory."""
        if not face_ids or not self._all_unnamed_clusters:
            return
        fid_set = set(face_ids)
        updated: list[dict[str, Any]] = []
        for c in self._all_unnamed_clusters:
            remaining = [fid for fid in c["face_ids"] if fid not in fid_set]
            if len(remaining) >= 3:
                c_copy = dict(c)
                c_copy["face_ids"] = remaining
                c_copy["face_count"] = len(remaining)
                if c_copy.get("representative_face_id") in fid_set:
                    c_copy["representative_face_id"] = remaining[0]
                    c_copy.pop("thumb_blob", None)
                updated.append(c_copy)
        self._all_unnamed_clusters = updated

    def _on_ignore_cluster_faces(self, face_ids: list[int]) -> None:
        self.repo.ignore_face(face_ids)
        self._prune_unnamed_cluster_faces(face_ids)
        self.refresh_directory(recluster_unnamed=False)
        self.directory_updated.emit()

    def _on_rename_person(self, person_id: int, current_name: str) -> None:
        from PySide6.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(
            self, "Rename Person", "Enter new name for person:", QLineEdit.Normal, current_name
        )
        if ok and new_name.strip() and new_name.strip() != current_name:
            try:
                self.repo.rename_person(person_id, new_name.strip())
                self.refresh_directory(recluster_unnamed=False)
                self.directory_updated.emit()
            except Exception as e:
                QMessageBox.warning(self, "Rename Failed", f"Could not rename person: {e}")

    def _on_merge_person(self, person_id: int, source_name: str) -> None:
        candidates = [p for p in self._all_named_people if p["id"] != person_id]
        if not candidates:
            QMessageBox.information(self, "Merge Person", "No other people available to merge into.")
            return

        src_info = next((p for p in self._all_named_people if p["id"] == person_id), None)
        src_thumb = src_info.get("thumbnail_blob") if src_info else None
        src_cnt = src_info.get("photo_count", 0) if src_info else 0

        dlg = MergePeopleDialog(
            source_person_name=source_name,
            candidates=candidates,
            source_person_id=person_id,
            source_photo_count=src_cnt,
            source_thumb_blob=src_thumb,
            parent=self,
        )
        if dlg.exec() == QDialog.Accepted:
            if dlg.is_rename():
                new_name = dlg.target_person_name()
                try:
                    self.repo.rename_person(person_id, new_name)
                    self.refresh_directory(recluster_unnamed=False)
                    self.directory_updated.emit()
                except Exception as e:
                    QMessageBox.warning(self, "Rename Failed", f"Could not rename person: {e}")
            else:
                dst_id = dlg.target_person_id()
                if dst_id is not None:
                    try:
                        self.repo.merge_people(person_id, dst_id)
                        self.refresh_directory(recluster_unnamed=False)
                        self.directory_updated.emit()
                    except Exception as e:
                        QMessageBox.warning(self, "Merge Failed", f"Could not merge people: {e}")

    def _on_review_person(self, person_id: int) -> None:
        self.person_selected.emit(person_id)

    def _on_set_cover_face(self, person_id: int) -> None:
        self.person_selected.emit(person_id)
