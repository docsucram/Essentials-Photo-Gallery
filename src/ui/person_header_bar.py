"""
Person Header Bar & Inline Suggestions Banner for PhotoGallery64.
Provides a modern Google Photos / Immich-style header bar displayed at the top of the photo grid
when viewing a specific person:
- Circular face avatar, editable Name, photo count, and [Full Photos | Face Crops] view toggle.
- Google Photos-style pending face suggestions banner with inline '[Yes] [No]' rapid verification strip.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QByteArray, QPoint, QRect, QSize, Qt, Signal
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
    QButtonGroup,
    QCompleter,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository

logger = logging.getLogger(__name__)


def make_circular_avatar(blob: bytes | None, name: str, size: int = 52) -> QPixmap:
    """Renders a crisp anti-aliased circular avatar from WebP blob, or initial letter fallback."""
    pm = QPixmap()
    if blob:
        pm.loadFromData(blob)
    if pm.isNull():
        # Fallback colored circle with initials
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Consistent color based on hash of name
        safe_name = (str(name) if name is not None else "").strip()
        h = abs(hash(safe_name or "Unknown")) % 360
        bg_color = QColor.fromHsl(h, 110, 140)
        p.setBrush(QBrush(bg_color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(0, 0, size, size)

        # Initials
        initials = (safe_name[:2] if safe_name else "?").upper()
        p.setPen(QColor("#FFFFFF"))
        font = QFont("Segoe UI", max(10, size // 3), QFont.Weight.Bold)
        p.setFont(font)
        p.drawText(QRect(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, initials)
        p.end()
        return pm

    # Scale and circular crop
    scaled = pm.scaled(
        size, size, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation
    )
    result = QPixmap(size, size)
    result.fill(Qt.GlobalColor.transparent)
    p = QPainter(result)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    path = QPainterPath()
    path.addEllipse(0, 0, size, size)
    p.setClipPath(path)

    # Center crop
    sx = (scaled.width() - size) // 2
    sy = (scaled.height() - size) // 2
    p.drawPixmap(0, 0, scaled, sx, sy, size, size)

    # Subtle border
    p.setClipping(False)
    p.setPen(QPen(QColor(0, 0, 0, 40), 1.0))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(0, 0, size - 1, size - 1)
    p.end()
    return result


def make_square_face_thumbnail(blob: bytes | None, size: int = 60) -> QPixmap:
    """Renders a crisp square face crop thumbnail filling the square without letterboxing."""
    pm = QPixmap()
    if blob:
        pm.loadFromData(blob)
    if pm.isNull():
        pm = QPixmap(size, size)
        pm.fill(QColor("#E2E8F0"))
        p = QPainter(pm)
        p.setPen(QColor("#64748B"))
        p.setFont(QFont("Segoe UI", 12))
        p.drawText(QRect(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, "?")
        p.end()
        return pm

    # Scale with KeepAspectRatioByExpanding to fill the square completely
    scaled = pm.scaled(
        size, size, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation
    )
    result = QPixmap(size, size)
    result.fill(Qt.GlobalColor.transparent)
    p = QPainter(result)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    # Center crop
    sx = max(0, (scaled.width() - size) // 2)
    sy = max(0, (scaled.height() - size) // 2)
    p.drawPixmap(0, 0, scaled, sx, sy, size, size)
    # Subtle 1px border
    p.setPen(QPen(QColor(0, 0, 0, 40), 1.0))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(0, 0, size - 1, size - 1)
    p.end()
    return result


class _AutoPopupLineEdit(QLineEdit):
    """
    A QLineEdit subclass that immediately opens its QCompleter dropdown on mouse click or focus,
    showing all candidate items when empty or filtered items when text is present.
    """

    def mousePressEvent(self, event: Any) -> None:
        super().mousePressEvent(event)
        self.show_completion_popup()

    def focusInEvent(self, event: Any) -> None:
        super().focusInEvent(event)
        self.show_completion_popup()

    def show_completion_popup(self) -> None:
        comp = self.completer()
        if comp:
            comp.setCompletionPrefix(self.text().strip())
            comp.complete()


class _QuickFaceSuggestionCard(QFrame):
    """
    Square 66x66 face thumbnail card matching authentic WLPG.
    Displays a clear 60x60 square face crop with explicit [✓] confirm and [✕] reject actions.
    Clicking the face image itself opens the photo for inspection without silently confirming.
    """

    confirmed = Signal(int)         # face_id
    rejected = Signal(int)          # face_id
    preview_requested = Signal(int) # media_id

    def __init__(
        self,
        face_id: int,
        thumb_blob: bytes | None,
        confidence: float,
        parent: QWidget | None = None,
        media_id: int = 0,
    ) -> None:
        super().__init__(parent)
        self.face_id = face_id
        self.media_id = media_id
        self.thumb_blob = thumb_blob
        self.confidence = confidence
        self.setFixedSize(66, 66)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click photo to inspect • Click ✓ to confirm or ✕ to reject")
        self.setStyleSheet(
            "_QuickFaceSuggestionCard { background-color: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 4px; }"
            "_QuickFaceSuggestionCard:hover { border: 2px solid #2563EB; background-color: #F8FAFC; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)

        # Avatar Label
        self.lbl_avatar = QLabel(self)
        self.lbl_avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = make_square_face_thumbnail(thumb_blob, size=60)
        self.lbl_avatar.setPixmap(pix)
        layout.addWidget(self.lbl_avatar)

        # Reject [✕] Overlay Button (top-right)
        self.btn_no = QPushButton("✕", self)
        self.btn_no.setFixedSize(17, 17)
        self.btn_no.setToolTip("Not this person")
        self.btn_no.setStyleSheet(
            "QPushButton { background-color: rgba(220, 38, 38, 0.85); color: white; border-radius: 3px; "
            "font-size: 10px; font-weight: bold; border: none; padding: 0px; }"
            "QPushButton:hover { background-color: #B91C1C; }"
        )
        self.btn_no.move(46, 3)
        self.btn_no.clicked.connect(self._on_reject_clicked)

        # Confirm [✓] Overlay Button (bottom-right)
        self.btn_yes = QPushButton("✓", self)
        self.btn_yes.setFixedSize(17, 17)
        self.btn_yes.setToolTip("Confirm this person")
        self.btn_yes.setStyleSheet(
            "QPushButton { background-color: rgba(34, 197, 94, 0.9); color: white; border-radius: 3px; "
            "font-size: 10px; font-weight: bold; border: none; padding: 0px; }"
            "QPushButton:hover { background-color: #16A34A; }"
        )
        self.btn_yes.move(46, 46)
        self.btn_yes.clicked.connect(self._on_confirm_clicked)

    def _on_confirm_clicked(self) -> None:
        self.confirmed.emit(self.face_id)

    def _on_reject_clicked(self) -> None:
        self.rejected.emit(self.face_id)

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self.media_id > 0:
                self.preview_requested.emit(self.media_id)
            event.accept()
            return
        super().mousePressEvent(event)


class PersonHeaderBar(QFrame):
    """
    Compact Person Header Bar & Inline Suggestions Banner.
    Rendered directly above the photo grid when a person is filtered in the Left Navigation Tree.
    """

    back_to_directory_requested = Signal()
    review_suggestions_requested = Signal(int)   # person_id
    view_mode_changed = Signal(str)             # "photos" or "crops"
    person_renamed = Signal(int, str)           # (person_id, new_name)
    cover_face_changed = Signal(int)            # person_id
    unnamed_person_named = Signal(str, list)    # (name, face_ids)
    open_media_requested = Signal(int)          # media_id
    geometry_changed = Signal()

    def __init__(
        self,
        engine: DualDatabaseEngine | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.repo = CatalogRepository(engine) if engine else None
        self.thumb_repo = ThumbnailRepository(engine) if engine else None
        self.current_person_id: int | None = None
        self.current_person_name: str = ""
        self.current_unnamed_face_ids: list[int] = []
        self.current_view_mode: str = "photos"

        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setObjectName("PersonHeaderBar")
        self.setStyleSheet(
            "#PersonHeaderBar { background-color: #F8FAFC; border-bottom: 1px solid #DDE4ED; }"
            "QToolTip { background-color: #FFFFFF; color: #1E395B; border: 1px solid #7DA2CE; "
            "border-radius: 2px; padding: 3px 6px; font-family: 'Segoe UI', sans-serif; font-size: 9pt; }"
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 8, 14, 8)
        main_layout.setSpacing(6)

        # ---------------------------------------------------------------------
        # 1. Top Identity & Toggle Row
        # ---------------------------------------------------------------------
        h_row = QHBoxLayout()
        h_row.setContentsMargins(0, 0, 0, 0)
        h_row.setSpacing(12)

        # Back to People Directory Button
        self.btn_back_people = QPushButton("← People", self)
        self.btn_back_people.setToolTip("Back to all People directory")
        self.btn_back_people.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; border: 1px solid #CBD5E1; color: #1E395B; "
            "font-weight: bold; padding: 4px 10px; border-radius: 4px; font-size: 11px; }"
            "QPushButton:hover { background-color: #EDF2F7; border-color: #94A3B8; }"
        )
        self.btn_back_people.clicked.connect(self.back_to_directory_requested.emit)
        h_row.addWidget(self.btn_back_people)

        # Circular Face Avatar
        self.lbl_avatar = QLabel(self)
        self.lbl_avatar.setFixedSize(48, 48)
        self.lbl_avatar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lbl_avatar.setToolTip("Click to change cover face...")
        self.lbl_avatar.mousePressEvent = self._on_avatar_clicked
        h_row.addWidget(self.lbl_avatar)

        # Name and Count VBox
        v_info = QVBoxLayout()
        v_info.setContentsMargins(0, 0, 0, 0)
        v_info.setSpacing(2)
        v_info.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.txt_name = _AutoPopupLineEdit(self)
        self.txt_name.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.txt_name.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.txt_name.setToolTip("Click to rename")
        self.txt_name.setTextMargins(0, 0, 0, 0)
        self.txt_name.setStyleSheet(
            "QLineEdit { border: 1px solid transparent; background: transparent; color: #0F172A; padding: 0px; margin: 0px; border-radius: 3px; }"
            "QLineEdit:hover { border-color: #CBD5E1; background: #FFFFFF; padding: 0px 4px; }"
            "QLineEdit:focus { border-color: #3B82F6; background: #FFFFFF; padding: 0px 4px; }"
        )
        self.txt_name.returnPressed.connect(self._on_name_edited)
        v_info.addWidget(self.txt_name, 0, Qt.AlignmentFlag.AlignLeft)

        self.lbl_photo_count = QLabel(self)
        self.lbl_photo_count.setContentsMargins(0, 0, 0, 0)
        self.lbl_photo_count.setStyleSheet("background: transparent; color: #64748B; font-size: 11px; padding: 0px; margin: 0px;")
        v_info.addWidget(self.lbl_photo_count, 0, Qt.AlignmentFlag.AlignLeft)

        h_row.addLayout(v_info, 1)

        # View Mode Toggle: [🖼 Full Photos | 👤 Face Crops]
        self.btn_group_view = QButtonGroup(self)
        h_toggle = QHBoxLayout()
        h_toggle.setContentsMargins(0, 0, 0, 0)
        h_toggle.setSpacing(0)

        self.btn_full_photos = QPushButton("🖼 Full Photos", self)
        self.btn_full_photos.setCheckable(True)
        self.btn_full_photos.setChecked(True)
        self.btn_full_photos.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; border: 1px solid #CBD5E1; border-top-left-radius: 4px; border-bottom-left-radius: 4px; "
            "padding: 5px 12px; font-size: 11px; color: #334155; }"
            "QPushButton:checked { background-color: #1E395B; color: #FFFFFF; border-color: #1E395B; font-weight: bold; }"
        )
        self.btn_group_view.addButton(self.btn_full_photos)
        h_toggle.addWidget(self.btn_full_photos)

        self.btn_face_crops = QPushButton("👤 Face Crops", self)
        self.btn_face_crops.setCheckable(True)
        self.btn_face_crops.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; border: 1px solid #CBD5E1; border-left: none; border-top-right-radius: 4px; border-bottom-right-radius: 4px; "
            "padding: 5px 12px; font-size: 11px; color: #334155; }"
            "QPushButton:checked { background-color: #1E395B; color: #FFFFFF; border-color: #1E395B; font-weight: bold; }"
        )
        self.btn_group_view.addButton(self.btn_face_crops)
        h_toggle.addWidget(self.btn_face_crops)

        self.btn_full_photos.clicked.connect(lambda: self._set_view_mode("photos"))
        self.btn_face_crops.clicked.connect(lambda: self._set_view_mode("crops"))

        h_row.addLayout(h_toggle)
        main_layout.addLayout(h_row)

        # ---------------------------------------------------------------------
        # 2. WLPG-Style Unconfirmed Suggestions Banner & Horizontal Strip
        # ---------------------------------------------------------------------
        self.banner_frame = QFrame(self)
        self.banner_frame.setObjectName("SuggestionBanner")
        self.banner_frame.setStyleSheet(
            "#SuggestionBanner { background-color: #F0F6FF; border: 1px solid #C7D9EE; border-radius: 6px; padding: 4px; }"
            "QToolTip { background-color: #FFFFFF; color: #1E395B; border: 1px solid #7DA2CE; "
            "border-radius: 2px; padding: 3px 6px; font-family: 'Segoe UI', sans-serif; font-size: 9pt; }"
        )
        self.banner_frame.hide()

        v_banner = QVBoxLayout(self.banner_frame)
        v_banner.setContentsMargins(8, 4, 8, 4)
        v_banner.setSpacing(4)

        # Banner Header Row (WLPG: Are these <Name>? [✓ Confirm All] [View all suggestions] count)
        h_banner_top = QHBoxLayout()
        h_banner_top.setContentsMargins(0, 0, 0, 0)
        h_banner_top.setSpacing(10)

        self.lbl_banner_title = QLabel(self.banner_frame)
        self.lbl_banner_title.setStyleSheet("font-weight: bold; color: #1E395B; font-size: 12px;")
        h_banner_top.addWidget(self.lbl_banner_title)

        self.btn_confirm_all = QPushButton("✓ Confirm all", self.banner_frame)
        self.btn_confirm_all.setToolTip("Confirm all face suggestions displayed below")
        self.btn_confirm_all.setStyleSheet(
            "QPushButton { background-color: #2E8B57; color: white; font-weight: bold; padding: 3px 12px; "
            "border-radius: 3px; font-size: 11px; border: 1px solid #236B43; }"
            "QPushButton:hover { background-color: #3CB371; }"
        )
        self.btn_confirm_all.clicked.connect(self._on_confirm_all_clicked)
        h_banner_top.addWidget(self.btn_confirm_all)

        self.btn_review_all = QPushButton("View all suggestions...", self.banner_frame)
        self.btn_review_all.setToolTip("Open full Batch People Tagging to review suggestions")
        self.btn_review_all.setStyleSheet(
            "QPushButton { background: transparent; color: #1A5FB4; font-size: 11px; text-decoration: underline; border: none; padding: 2px 6px; }"
            "QPushButton:hover { color: #0B3D91; }"
        )
        self.btn_review_all.clicked.connect(self._on_review_all_clicked)
        h_banner_top.addWidget(self.btn_review_all)

        h_banner_top.addStretch(1)

        self.lbl_banner_count = QLabel(self.banner_frame)
        self.lbl_banner_count.setStyleSheet("color: #64748B; font-size: 11px;")
        h_banner_top.addWidget(self.lbl_banner_count)

        v_banner.addLayout(h_banner_top)

        # Horizontal square face crop strip (WLPG style)
        self.strip_scroll = QScrollArea(self.banner_frame)
        self.strip_scroll.setFixedHeight(74)
        self.strip_scroll.setWidgetResizable(True)
        self.strip_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.strip_scroll.setStyleSheet("background: transparent;")

        self.strip_content = QWidget()
        self.strip_content.setStyleSheet("background: transparent;")
        self.strip_layout = QHBoxLayout(self.strip_content)
        self.strip_layout.setContentsMargins(0, 2, 0, 2)
        self.strip_layout.setSpacing(6)
        self.strip_layout.addStretch(1)
        self.strip_scroll.setWidget(self.strip_content)

        v_banner.addWidget(self.strip_scroll)
        main_layout.addWidget(self.banner_frame)

    def set_dark_mode(self, is_dark: bool) -> None:
        self._dark_mode = bool(is_dark)
        self._is_dark = self._dark_mode
        if not self._dark_mode:
            self.setStyleSheet(
                "#PersonHeaderBar { background-color: #F8FAFC; border-bottom: 1px solid #DDE4ED; }"
                "QToolTip { background-color: #FFFFFF; color: #1E395B; border: 1px solid #7DA2CE; "
                "border-radius: 2px; padding: 3px 6px; font-family: 'Segoe UI', sans-serif; font-size: 9pt; }"
            )
            self.btn_back_people.setStyleSheet(
                "QPushButton { background-color: #FFFFFF; border: 1px solid #CBD5E1; color: #1E395B; "
                "font-weight: bold; padding: 4px 10px; border-radius: 4px; font-size: 11px; }"
                "QPushButton:hover { background-color: #EDF2F7; border-color: #94A3B8; }"
            )
            self.txt_name.setStyleSheet(
                "QLineEdit { border: 1px solid transparent; background: transparent; color: #0F172A; padding: 0px; margin: 0px; border-radius: 3px; }"
                "QLineEdit:hover { border-color: #CBD5E1; background: #FFFFFF; padding: 0px 4px; }"
                "QLineEdit:focus { border-color: #3B82F6; background: #FFFFFF; padding: 0px 4px; }"
            )
            self.lbl_photo_count.setStyleSheet("background: transparent; color: #64748B; font-size: 11px; padding: 0px; margin: 0px;")
            self.btn_full_photos.setStyleSheet(
                "QPushButton { background-color: #FFFFFF; border: 1px solid #CBD5E1; border-top-left-radius: 4px; border-bottom-left-radius: 4px; "
                "padding: 5px 12px; font-size: 11px; color: #334155; }"
                "QPushButton:checked { background-color: #1E395B; color: #FFFFFF; border-color: #1E395B; font-weight: bold; }"
            )
            self.btn_face_crops.setStyleSheet(
                "QPushButton { background-color: #FFFFFF; border: 1px solid #CBD5E1; border-left: none; border-top-right-radius: 4px; border-bottom-right-radius: 4px; "
                "padding: 5px 12px; font-size: 11px; color: #334155; }"
                "QPushButton:checked { background-color: #1E395B; color: #FFFFFF; border-color: #1E395B; font-weight: bold; }"
            )
            self.banner_frame.setStyleSheet(
                "#SuggestionBanner { background-color: #F0F6FF; border: 1px solid #C7D9EE; border-radius: 6px; padding: 4px; }"
            )
            self.lbl_banner_title.setStyleSheet("background: transparent; font-weight: bold; color: #1E395B; font-size: 12px;")
            self.lbl_banner_count.setStyleSheet("background: transparent; color: #64748B; font-size: 11px;")
        else:
            self.setStyleSheet(
                "#PersonHeaderBar { background-color: #1E2228; border-bottom: 1px solid #2E3440; }"
                "QToolTip { background-color: #FFFFFF; color: #1E395B; border: 1px solid #7DA2CE; "
                "border-radius: 2px; padding: 3px 6px; font-family: 'Segoe UI', sans-serif; font-size: 9pt; }"
            )
            self.btn_back_people.setStyleSheet(
                "QPushButton { background-color: #262C36; border: 1px solid #3E4756; color: #E2E8F0; "
                "font-weight: bold; padding: 4px 10px; border-radius: 4px; font-size: 11px; }"
                "QPushButton:hover { background-color: #323946; border-color: #60A5FA; }"
            )
            self.txt_name.setStyleSheet(
                "QLineEdit { border: 1px solid transparent; background: transparent; color: #F1F5F9; padding: 0px; margin: 0px; border-radius: 3px; }"
                "QLineEdit:hover { border-color: #475569; background: #14171B; padding: 0px 4px; }"
                "QLineEdit:focus { border-color: #3B82F6; background: #14171B; padding: 0px 4px; }"
            )
            self.lbl_photo_count.setStyleSheet("background: transparent; color: #94A3B8; font-size: 11px; padding: 0px; margin: 0px;")
            self.btn_full_photos.setStyleSheet(
                "QPushButton { background-color: #262C36; border: 1px solid #3E4756; border-top-left-radius: 4px; border-bottom-left-radius: 4px; "
                "padding: 5px 12px; font-size: 11px; color: #CBD5E1; }"
                "QPushButton:checked { background-color: #2563EB; color: #FFFFFF; border-color: #2563EB; font-weight: bold; }"
            )
            self.btn_face_crops.setStyleSheet(
                "QPushButton { background-color: #262C36; border: 1px solid #3E4756; border-left: none; border-top-right-radius: 4px; border-bottom-right-radius: 4px; "
                "padding: 5px 12px; font-size: 11px; color: #CBD5E1; }"
                "QPushButton:checked { background-color: #2563EB; color: #FFFFFF; border-color: #2563EB; font-weight: bold; }"
            )
            self.banner_frame.setStyleSheet(
                "#SuggestionBanner { background-color: #1A2433; border: 1px solid #2E4057; border-radius: 6px; padding: 4px; }"
            )
            self.lbl_banner_title.setStyleSheet("background: transparent; font-weight: bold; color: #93C5FD; font-size: 12px;")
            self.lbl_banner_count.setStyleSheet("background: transparent; color: #94A3B8; font-size: 11px;")

    def load_unnamed_person(
        self,
        face_ids: list[int],
        thumb_blob: bytes | None = None,
        photo_count: int = 0,
    ) -> None:
        """Configures header bar to display an unnamed person group."""
        self.current_person_id = None
        self.current_unnamed_face_ids = list(face_ids)
        self.current_person_name = ""
        self.txt_name.setText("")
        self.txt_name.setPlaceholderText("Add a name...")
        self.txt_name.setCursorPosition(0)
        self.txt_name.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.txt_name.setMinimumWidth(200)
        cnt = photo_count or len(face_ids)
        cnt_text = f"{cnt:,} photo{'s' if cnt != 1 else ''} • Find them quickly by name with search"
        self.lbl_photo_count.setText(cnt_text)
        self.lbl_avatar.setPixmap(make_circular_avatar(thumb_blob, "?", size=48))
        self._update_name_completer()
        self.btn_full_photos.setChecked(True)
        self.btn_face_crops.setChecked(False)
        self.current_view_mode = "photos"
        self.banner_frame.hide()
        self.updateGeometry()
        self.geometry_changed.emit()
        self.show()

    def sizeHint(self) -> QSize:
        if hasattr(self, "banner_frame") and self.banner_frame.isVisible():
            return QSize(self.width(), 176)
        return QSize(self.width(), 64)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def load_person(self, person_id: int) -> None:
        """Loads person data, updates avatar, counts, and inline suggestions banner."""
        self.current_unnamed_face_ids = []
        if not self.repo:
            return
        self.current_person_id = person_id
        self.btn_full_photos.setChecked(True)
        self.btn_face_crops.setChecked(False)
        self.current_view_mode = "photos"
        conn = self.repo.engine.catalog_conn()
        p_row = conn.execute(
            """
            SELECT p.id, p.name, p.thumbnail_blob, p.confirmed_count,
                   COUNT(DISTINCT mp.media_id) AS photo_count
            FROM people p
            LEFT JOIN media_people mp ON mp.person_id = p.id AND mp.state = 1
            WHERE p.id = ?
            GROUP BY p.id;
            """,
            (person_id,),
        ).fetchone()

        if not p_row:
            self.hide()
            return

        self.current_person_name = str(p_row["name"])
        self.txt_name.setText(self.current_person_name)
        self.txt_name.setCursorPosition(0)
        self.txt_name.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        from PySide6.QtGui import QFontMetrics
        fm = QFontMetrics(self.txt_name.font())
        req_w = min(500, max(160, fm.horizontalAdvance(self.current_person_name) + 28))
        self.txt_name.setMinimumWidth(req_w)

        photo_cnt = int(p_row["photo_count"] or p_row["confirmed_count"] or 0)
        cnt_text = f"{photo_cnt:,} photo{'s' if photo_cnt != 1 else ''}"
        self.lbl_photo_count.setText(cnt_text)

        # Avatar
        thumb_blob = bytes(p_row["thumbnail_blob"]) if p_row["thumbnail_blob"] else None
        self.lbl_avatar.setPixmap(make_circular_avatar(thumb_blob, self.current_person_name, size=48))
        self._update_name_completer()

        # Check for pending suggestions
        suggestions = self.repo.get_unconfirmed_suggestions_for_person(person_id, limit=12)
        if suggestions:
            s_cnt = len(suggestions)
            self.lbl_banner_title.setText(f"Are these {self.current_person_name}? ({self.current_person_name} has {s_cnt} unconfirmed face suggestion{'s' if s_cnt != 1 else ''})")
            self.lbl_banner_count.setText(f"{s_cnt} suggestion{'s' if s_cnt != 1 else ''}")
            self._populate_suggestion_strip(suggestions)
            self.banner_frame.show()
        else:
            self.banner_frame.hide()

        self.updateGeometry()
        self.geometry_changed.emit()
        self.show()

    def _populate_suggestion_strip(self, suggestions: list[dict[str, Any]]) -> None:
        """Populates the inline face cards in the suggestions banner."""
        # Clear old items (except trailing stretch)
        while self.strip_layout.count() > 1:
            item = self.strip_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for s in suggestions[:8]:
            fid = int(s["face_id"])
            mid = int(s.get("media_id") or 0)
            thumb_blob = self.thumb_repo.get_face_thumbnail(fid) if self.thumb_repo else None
            conf = float(s.get("confidence") or 0.0)
            card = _QuickFaceSuggestionCard(fid, thumb_blob, conf, self.strip_content, media_id=mid)
            card.confirmed.connect(self._on_quick_confirm)
            card.rejected.connect(self._on_quick_reject)
            card.preview_requested.connect(self.open_media_requested.emit)
            self.strip_layout.insertWidget(self.strip_layout.count() - 1, card)

    def _on_quick_confirm(self, face_id: int) -> None:
        if self.repo and self.current_person_id:
            self.repo.confirm_faces([face_id], self.current_person_id)
            # Remove confirmed card
            self._remove_card_from_strip(face_id)

    def _on_quick_reject(self, face_id: int) -> None:
        if self.repo and self.current_person_id:
            self.repo.reject_face_suggestion(face_id, self.current_person_id)
            self._remove_card_from_strip(face_id)

    def _remove_card_from_strip(self, face_id: int) -> None:
        for i in range(self.strip_layout.count() - 1):
            w = self.strip_layout.itemAt(i).widget()
            if isinstance(w, _QuickFaceSuggestionCard) and w.face_id == face_id:
                w.deleteLater()
                break
        remaining = sum(
            1
            for i in range(self.strip_layout.count() - 1)
            if isinstance(self.strip_layout.itemAt(i).widget(), _QuickFaceSuggestionCard)
            and self.strip_layout.itemAt(i).widget().face_id != face_id
        )
        if remaining <= 0:
            self.banner_frame.hide()
            self.updateGeometry()
            self.geometry_changed.emit()
        else:
            self.lbl_banner_title.setText(
                f"Are these {self.current_person_name}? ({self.current_person_name} has {remaining} unconfirmed face suggestion{'s' if remaining != 1 else ''})"
            )
            self.lbl_banner_count.setText(f"{remaining} suggestion{'s' if remaining != 1 else ''}")

    def _on_confirm_all_clicked(self) -> None:
        """Confirms all visible face suggestion cards for this person at once."""
        if not self.repo or not self.current_person_id:
            return
        fids = []
        for i in range(self.strip_layout.count() - 1):
            w = self.strip_layout.itemAt(i).widget()
            if isinstance(w, _QuickFaceSuggestionCard):
                fids.append(w.face_id)
        if fids:
            self.repo.confirm_faces(fids, self.current_person_id)
            self.load_person(self.current_person_id)
            self.updateGeometry()
            self.geometry_changed.emit()

    def _on_review_all_clicked(self) -> None:
        if self.current_person_id:
            self.review_suggestions_requested.emit(self.current_person_id)

    def _update_name_completer(self) -> None:
        """Populates the name line edit with autocomplete suggestions from existing people."""
        if not self.repo:
            return
        try:
            people_rows = self.repo.get_people_list()
            names = [
                str(p["name"]) for p in people_rows
                if p.get("name") and str(p["name"]).strip() and str(p["name"]) != self.current_person_name
            ]
            if names:
                comp = QCompleter(names, self.txt_name)
                comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
                comp.setFilterMode(Qt.MatchFlag.MatchContains)
                self.txt_name.setCompleter(comp)
                comp.activated.connect(self._on_completer_activated)
            else:
                self.txt_name.setCompleter(None)
        except Exception:
            pass

    def _on_completer_activated(self, text: str) -> None:
        self.txt_name.setText(text)
        self._on_name_edited()

    def _on_name_edited(self) -> None:
        new_name = self.txt_name.text().strip()
        if not new_name or not self.repo:
            return
        if self.current_unnamed_face_ids:
            self.unnamed_person_named.emit(new_name, self.current_unnamed_face_ids)
            return
        if not self.current_person_id or new_name == self.current_person_name:
            return
        try:
            actual_id = self.repo.rename_person(self.current_person_id, new_name)
            self.current_person_id = actual_id
            self.current_person_name = new_name
            self.person_renamed.emit(actual_id, new_name)
            self.load_person(actual_id)
        except Exception as e:
            QMessageBox.warning(self, "Rename Person", f"Failed to rename person: {e}")

    def _set_view_mode(self, mode: str) -> None:
        self.current_view_mode = mode
        self.view_mode_changed.emit(mode)

    def _on_avatar_clicked(self, event: Any) -> None:
        if not self.current_person_id or not self.repo:
            return
        self.cover_face_changed.emit(self.current_person_id)
