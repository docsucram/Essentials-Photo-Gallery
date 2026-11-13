"""
AlbumsView — Visual Photo-Stack Directory for Static Albums and Smart Albums.
Mirrors the visual polish of PeopleDirectoryView with fanned/layered photo stacks,
inline "+ New Album" and "+ New Smart Album" actions, search filtering, and
context menus for opening, renaming, editing rules, and deleting albums.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.ui.ribbon_toolbar import make_wlpg_icon

logger = logging.getLogger(__name__)


def render_album_stack_pixmap(
    thumbs: list[QPixmap],
    is_smart: bool = False,
    is_dark: bool = False,
    width: int = 176,
    height: int = 136,
) -> QPixmap:
    """
    Composite up to 3 thumbnails into a fanned photo-stack preview pixmap.
    Falls back to a clean stacked placeholder when the album has no photos yet.
    """
    pm = QPixmap(width, height)
    pm.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    card_w = 142
    card_h = 110
    cx = width / 2.0
    cy = height / 2.0 + 1.0

    frame_bg = QColor("#242933") if is_dark else QColor("#FFFFFF")
    frame_border = QColor("#3E4756") if is_dark else QColor("#CBD5E1")
    empty_fill = QColor("#1E242E") if is_dark else QColor("#F1F5F9")
    shadow_col = QColor(0, 0, 0, 55 if is_dark else 36)

    def _draw_layer(
        angle_deg: float,
        dx: float,
        dy: float,
        img_pm: QPixmap | None,
        draw_placeholder_icon: bool = False,
    ) -> None:
        painter.save()
        painter.translate(cx + dx, cy + dy)
        painter.rotate(angle_deg)

        outer = QRect(-card_w // 2, -card_h // 2, card_w, card_h)
        # Soft shadow
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(shadow_col)
        painter.drawRoundedRect(outer.adjusted(2, 3, 2, 3), 5, 5)

        # Card paper border
        painter.setBrush(frame_bg)
        painter.setPen(QPen(frame_border, 1.0))
        painter.drawRoundedRect(outer, 5, 5)

        inner = outer.adjusted(3, 3, -3, -3)
        clip = QPainterPath()
        clip.addRoundedRect(inner, 3, 3)
        painter.setClipPath(clip)

        if img_pm is not None and not img_pm.isNull():
            scaled = img_pm.scaled(
                inner.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            sx = (scaled.width() - inner.width()) // 2
            sy = (scaled.height() - inner.height()) // 2
            painter.drawPixmap(inner.topLeft(), scaled, QRect(sx, sy, inner.width(), inner.height()))
        else:
            painter.fillRect(inner, empty_fill)
            if draw_placeholder_icon:
                ico = make_wlpg_icon("star_filled" if is_smart else "pictures_library", 28)
                ico_pm = ico.pixmap(28, 28)
                painter.drawPixmap(
                    inner.center().x() - 14,
                    inner.center().y() - 14,
                    ico_pm,
                )
        painter.restore()

    if len(thumbs) >= 3:
        _draw_layer(-5.0, -7.0, 2.0, thumbs[2])
        _draw_layer(4.5, 7.0, 1.0, thumbs[1])
        _draw_layer(0.0, 0.0, 0.0, thumbs[0])
    elif len(thumbs) == 2:
        _draw_layer(-4.5, -6.0, 1.0, thumbs[1])
        _draw_layer(1.5, 3.0, 0.0, thumbs[0])
    elif len(thumbs) == 1:
        _draw_layer(4.0, 5.0, 2.0, None)
        _draw_layer(-1.0, -2.0, 0.0, thumbs[0])
    else:
        _draw_layer(3.5, 5.0, 2.0, None)
        _draw_layer(-1.0, -2.0, 0.0, None, draw_placeholder_icon=True)

    # Smart Album pill badge on top-right of the stack
    if is_smart:
        badge_rect = QRect(width - 52, 6, 46, 18)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#2563EB") if not is_dark else QColor("#3B82F6"))
        painter.drawRoundedRect(badge_rect, 9, 9)
        painter.setPen(QColor("#FFFFFF"))
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, "SMART")

    painter.end()
    return pm


class _AlbumStackCard(QFrame):
    """Clickable album card displaying a 3-photo stack, album title, and item count."""

    open_requested = Signal(int)
    rename_requested = Signal(int, str)
    delete_requested = Signal(int, str)
    edit_rules_requested = Signal(int)

    def __init__(
        self,
        album_id: int,
        name: str,
        item_count: int,
        thumbs: list[QPixmap],
        is_smart: bool = False,
        is_dark: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.album_id = album_id
        self.album_name = name
        self.item_count = item_count
        self.is_smart = is_smart
        self._is_dark = is_dark

        self.setFixedSize(194, 204)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"{name} ({item_count} item{'s' if item_count != 1 else ''})")

        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        v.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        self.lbl_stack = QLabel(self)
        self.lbl_stack.setFixedSize(176, 136)
        self.lbl_stack.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_stack.setPixmap(render_album_stack_pixmap(thumbs, is_smart=is_smart, is_dark=is_dark))
        v.addWidget(self.lbl_stack, 0, Qt.AlignmentFlag.AlignHCenter)

        fm = QFontMetrics(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        elided_name = fm.elidedText(name, Qt.TextElideMode.ElideRight, 174)
        self.lbl_name = QLabel(elided_name, self)
        self.lbl_name.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        self.lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.lbl_name)

        cnt_text = f"{item_count:,} item{'s' if item_count != 1 else ''}"
        self.lbl_count = QLabel(cnt_text, self)
        self.lbl_count.setFont(QFont("Segoe UI", 8))
        self.lbl_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.lbl_count)

        self._apply_style()

    def _apply_style(self) -> None:
        if self._is_dark:
            self.setStyleSheet(
                "_AlbumStackCard { background-color: #181C24; border: 1px solid #2A313D; border-radius: 8px; }"
                "_AlbumStackCard:hover { background-color: #1F2530; border: 1px solid #3B82F6; }"
            )
            self.lbl_name.setStyleSheet("color: #F1F5F9; background: transparent; border: none;")
            self.lbl_count.setStyleSheet("color: #94A3B8; background: transparent; border: none;")
        else:
            self.setStyleSheet(
                "_AlbumStackCard { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; }"
                "_AlbumStackCard:hover { background-color: #F8FAFC; border: 1px solid #2563EB; }"
            )
            self.lbl_name.setStyleSheet("color: #0F172A; background: transparent; border: none;")
            self.lbl_count.setStyleSheet("color: #64748B; background: transparent; border: none;")

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_requested.emit(self.album_id)
            event.accept()
            return
        super().mousePressEvent(event)

    def contextMenuEvent(self, event: Any) -> None:
        menu = QMenu(self)
        act_open = menu.addAction("Open Album" if not self.is_smart else "Open Smart Album")
        act_edit = menu.addAction("Edit Rules...") if self.is_smart else None
        act_rename = menu.addAction("Rename Album..." if not self.is_smart else "Rename Smart Album...")
        menu.addSeparator()
        act_delete = menu.addAction("Delete Album" if not self.is_smart else "Delete Smart Album")

        chosen = menu.exec(event.globalPos())
        if chosen is None:
            return
        if chosen == act_open:
            self.open_requested.emit(self.album_id)
        elif act_edit is not None and chosen == act_edit:
            self.edit_rules_requested.emit(self.album_id)
        elif chosen == act_rename:
            self.rename_requested.emit(self.album_id, self.album_name)
        elif chosen == act_delete:
            self.delete_requested.emit(self.album_id, self.album_name)


class _CreateAlbumCard(QFrame):
    """Dashed '+ New Album' / '+ New Smart Album' action tile inside each section."""

    clicked = Signal()

    def __init__(self, label: str, is_dark: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_dark = is_dark
        self.setFixedSize(194, 204)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 16, 12, 16)
        v.setSpacing(8)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_plus = QLabel("+", self)
        self.lbl_plus.setFont(QFont("Segoe UI", 26, QFont.Weight.Light))
        self.lbl_plus.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.lbl_plus)

        self.lbl_text = QLabel(label, self)
        self.lbl_text.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        self.lbl_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_text.setWordWrap(True)
        v.addWidget(self.lbl_text)

        self._apply_style()

    def _apply_style(self) -> None:
        if self._is_dark:
            self.setStyleSheet(
                "_CreateAlbumCard { background-color: #14171D; border: 2px dashed #334155; border-radius: 8px; }"
                "_CreateAlbumCard:hover { background-color: #1A202C; border-color: #3B82F6; }"
            )
            self.lbl_plus.setStyleSheet("color: #60A5FA; background: transparent; border: none;")
            self.lbl_text.setStyleSheet("color: #CBD5E1; background: transparent; border: none;")
        else:
            self.setStyleSheet(
                "_CreateAlbumCard { background-color: #F8FAFC; border: 2px dashed #CBD5E1; border-radius: 8px; }"
                "_CreateAlbumCard:hover { background-color: #EFF6FF; border-color: #2563EB; }"
            )
            self.lbl_plus.setStyleSheet("color: #2563EB; background: transparent; border: none;")
            self.lbl_text.setStyleSheet("color: #334155; background: transparent; border: none;")

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class AlbumsView(QWidget):
    """
    Visual Albums & Smart Albums directory view with stacked photo thumbnails,
    search filtering, and direct '+ New Album' / '+ New Smart Album' creation.
    """

    album_selected = Signal(int, str)          # (album_id, album_name)
    smart_album_selected = Signal(int, str)    # (smart_album_id, smart_album_name)
    new_album_requested = Signal()
    rename_album_requested = Signal(int, str)
    delete_album_requested = Signal(int, str)
    new_smart_album_requested = Signal()
    edit_smart_album_requested = Signal(int)
    rename_smart_album_requested = Signal(int, str)
    delete_smart_album_requested = Signal(int, str)

    def __init__(self, engine: DualDatabaseEngine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.repo = CatalogRepository(engine)
        self.thumb_repo = ThumbnailRepository(engine)
        self._is_dark: bool = False
        self._static_albums: list[dict[str, Any]] = []
        self._smart_albums: list[dict[str, Any]] = []
        self._last_cols: int = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header Bar
        self.header_bar = QFrame(self)
        self.header_bar.setFixedHeight(54)
        h_hdr = QHBoxLayout(self.header_bar)
        h_hdr.setContentsMargins(20, 8, 20, 8)
        h_hdr.setSpacing(12)

        self.lbl_title = QLabel("Albums", self.header_bar)
        self.lbl_title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        h_hdr.addWidget(self.lbl_title)

        h_hdr.addStretch(1)

        self.txt_search = QLineEdit(self.header_bar)
        self.txt_search.setPlaceholderText("Search albums...")
        self.txt_search.setClearButtonEnabled(True)
        self.txt_search.setFixedWidth(240)
        self.txt_search.setFixedHeight(30)
        self.txt_search.textChanged.connect(self._rebuild_grids)
        h_hdr.addWidget(self.txt_search)

        root.addWidget(self.header_bar)

        # Scroll Area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self.scroll_content = QWidget(self.scroll_area)
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(24, 18, 24, 28)
        self.scroll_layout.setSpacing(22)

        # Section 1: My Albums
        sec_static_hdr = QHBoxLayout()
        sec_static_hdr.setSpacing(10)
        self.lbl_static_hdr = QLabel("My Albums", self.scroll_content)
        self.lbl_static_hdr.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        sec_static_hdr.addWidget(self.lbl_static_hdr)

        self.btn_new_album = QPushButton("+ New Album", self.scroll_content)
        self.btn_new_album.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_new_album.setFixedHeight(26)
        self.btn_new_album.clicked.connect(self.new_album_requested.emit)
        sec_static_hdr.addWidget(self.btn_new_album)
        sec_static_hdr.addStretch(1)
        self.scroll_layout.addLayout(sec_static_hdr)

        self.static_grid_host = QWidget(self.scroll_content)
        self.static_grid = QGridLayout(self.static_grid_host)
        self.static_grid.setContentsMargins(0, 0, 0, 0)
        self.static_grid.setSpacing(14)
        self.static_grid.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.scroll_layout.addWidget(self.static_grid_host)

        # Section 2: Smart Albums
        sec_smart_hdr = QHBoxLayout()
        sec_smart_hdr.setSpacing(10)
        self.lbl_smart_hdr = QLabel("Smart Albums", self.scroll_content)
        self.lbl_smart_hdr.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        sec_smart_hdr.addWidget(self.lbl_smart_hdr)

        self.btn_new_smart_album = QPushButton("+ New Smart Album", self.scroll_content)
        self.btn_new_smart_album.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_new_smart_album.setFixedHeight(26)
        self.btn_new_smart_album.clicked.connect(self.new_smart_album_requested.emit)
        sec_smart_hdr.addWidget(self.btn_new_smart_album)
        sec_smart_hdr.addStretch(1)
        self.scroll_layout.addLayout(sec_smart_hdr)

        self.smart_grid_host = QWidget(self.scroll_content)
        self.smart_grid = QGridLayout(self.smart_grid_host)
        self.smart_grid.setContentsMargins(0, 0, 0, 0)
        self.smart_grid.setSpacing(14)
        self.smart_grid.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.scroll_layout.addWidget(self.smart_grid_host)

        self.scroll_layout.addStretch(1)
        self.scroll_area.setWidget(self.scroll_content)
        root.addWidget(self.scroll_area, 1)

        self._apply_theme()

    def set_dark_mode(self, is_dark: bool) -> None:
        self._is_dark = bool(is_dark)
        self._apply_theme()
        self._rebuild_grids()

    def _apply_theme(self) -> None:
        if self._is_dark:
            self.header_bar.setStyleSheet(
                "QFrame { background-color: #171A20; border-bottom: 1px solid #2A313D; }"
            )
            self.lbl_title.setStyleSheet("color: #F8FAFC; border: none;")
            self.txt_search.setStyleSheet(
                "QLineEdit { background-color: #1F242D; color: #F1F5F9; border: 1px solid #334155; "
                "border-radius: 15px; padding: 0px 12px; font-size: 12px; }"
                "QLineEdit:focus { border-color: #3B82F6; }"
            )
            self.scroll_area.setStyleSheet("QScrollArea { background-color: #121417; border: none; }")
            self.scroll_content.setStyleSheet("background-color: #121417;")
            self.lbl_static_hdr.setStyleSheet("color: #E2E8F0;")
            self.lbl_smart_hdr.setStyleSheet("color: #E2E8F0;")
            btn_style = (
                "QPushButton { background-color: #1E293B; color: #93C5FD; border: 1px solid #334155; "
                "border-radius: 13px; padding: 0px 12px; font-size: 11px; font-weight: 600; }"
                "QPushButton:hover { background-color: #2563EB; color: #FFFFFF; border-color: #2563EB; }"
            )
        else:
            self.header_bar.setStyleSheet(
                "QFrame { background-color: #FFFFFF; border-bottom: 1px solid #E2E8F0; }"
            )
            self.lbl_title.setStyleSheet("color: #0F172A; border: none;")
            self.txt_search.setStyleSheet(
                "QLineEdit { background-color: #F8FAFC; color: #0F172A; border: 1px solid #CBD5E1; "
                "border-radius: 15px; padding: 0px 12px; font-size: 12px; }"
                "QLineEdit:focus { border-color: #2563EB; background-color: #FFFFFF; }"
            )
            self.scroll_area.setStyleSheet("QScrollArea { background-color: #F8FAFC; border: none; }")
            self.scroll_content.setStyleSheet("background-color: #F8FAFC;")
            self.lbl_static_hdr.setStyleSheet("color: #1E293B;")
            self.lbl_smart_hdr.setStyleSheet("color: #1E293B;")
            btn_style = (
                "QPushButton { background-color: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; "
                "border-radius: 13px; padding: 0px 12px; font-size: 11px; font-weight: 600; }"
                "QPushButton:hover { background-color: #2563EB; color: #FFFFFF; border-color: #2563EB; }"
            )
        self.btn_new_album.setStyleSheet(btn_style)
        self.btn_new_smart_album.setStyleSheet(btn_style)

    def _compute_columns(self) -> int:
        avail = max(240, self.scroll_area.viewport().width() - 48)
        return max(1, avail // 208)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        cols = self._compute_columns()
        if cols != self._last_cols and (self._static_albums or self._smart_albums):
            self._rebuild_grids()

    def _load_preview_pixmaps(self, media_ids: list[int]) -> list[QPixmap]:
        if not media_ids:
            return []
        blobs = self.thumb_repo.get_webp_blobs_batch(media_ids)
        pixmaps: list[QPixmap] = []
        for mid in media_ids:
            blob = blobs.get(mid)
            if not blob:
                continue
            pm = QPixmap()
            if pm.loadFromData(blob) and not pm.isNull():
                pixmaps.append(pm)
        return pixmaps

    def refresh(self) -> None:
        """Reload static and smart albums + stack preview thumbnails from catalog.db."""
        try:
            self._static_albums = self.repo.get_albums_with_counts()
            for a in self._static_albums:
                preview_ids = self.repo.get_album_preview_media_ids(int(a["id"]), limit=3)
                a["_thumbs"] = self._load_preview_pixmaps(preview_ids)
        except Exception as exc:
            logger.debug("Failed to load static albums: %s", exc)
            self._static_albums = []

        try:
            self._smart_albums = self.repo.get_smart_albums_with_counts()
            for s in self._smart_albums:
                preview_ids = self.repo.get_smart_album_preview_media_ids(
                    s.get("rules_dict") or s.get("rules_json") or {}, limit=3
                )
                s["_thumbs"] = self._load_preview_pixmaps(preview_ids)
        except Exception as exc:
            logger.debug("Failed to load smart albums: %s", exc)
            self._smart_albums = []

        self._rebuild_grids()

    def _clear_grid(self, grid: QGridLayout) -> None:
        while grid.count() > 0:
            item = grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()

    def _rebuild_grids(self) -> None:
        cols = self._compute_columns()
        self._last_cols = cols
        query = self.txt_search.text().strip().lower()

        filtered_static = [
            a for a in self._static_albums if not query or query in str(a.get("name", "")).lower()
        ]
        filtered_smart = [
            s for s in self._smart_albums if not query or query in str(s.get("name", "")).lower()
        ]

        self.lbl_static_hdr.setText(f"My Albums ({len(filtered_static)})")
        self.lbl_smart_hdr.setText(f"Smart Albums ({len(filtered_smart)})")

        # 1. Static Albums grid
        self._clear_grid(self.static_grid)
        idx = 0
        for alb in filtered_static:
            aid = int(alb["id"])
            aname = str(alb["name"])
            cnt = int(alb.get("item_count") or 0)
            card = _AlbumStackCard(
                album_id=aid,
                name=aname,
                item_count=cnt,
                thumbs=alb.get("_thumbs") or [],
                is_smart=False,
                is_dark=self._is_dark,
                parent=self.static_grid_host,
            )
            card.open_requested.connect(lambda _aid, n=aname: self.album_selected.emit(_aid, n))
            card.rename_requested.connect(self.rename_album_requested.emit)
            card.delete_requested.connect(self.delete_album_requested.emit)
            self.static_grid.addWidget(card, idx // cols, idx % cols)
            idx += 1

        if not query:
            create_static = _CreateAlbumCard("New Album", is_dark=self._is_dark, parent=self.static_grid_host)
            create_static.clicked.connect(self.new_album_requested.emit)
            self.static_grid.addWidget(create_static, idx // cols, idx % cols)

        # 2. Smart Albums grid
        self._clear_grid(self.smart_grid)
        s_idx = 0
        for salb in filtered_smart:
            sid = int(salb["id"])
            sname = str(salb["name"])
            cnt = int(salb.get("item_count") or 0)
            card = _AlbumStackCard(
                album_id=sid,
                name=sname,
                item_count=cnt,
                thumbs=salb.get("_thumbs") or [],
                is_smart=True,
                is_dark=self._is_dark,
                parent=self.smart_grid_host,
            )
            card.open_requested.connect(lambda _sid, n=sname: self.smart_album_selected.emit(_sid, n))
            card.edit_rules_requested.connect(self.edit_smart_album_requested.emit)
            card.rename_requested.connect(self.rename_smart_album_requested.emit)
            card.delete_requested.connect(self.delete_smart_album_requested.emit)
            self.smart_grid.addWidget(card, s_idx // cols, s_idx % cols)
            s_idx += 1

        if not query:
            create_smart = _CreateAlbumCard(
                "New Smart Album", is_dark=self._is_dark, parent=self.smart_grid_host
            )
            create_smart.clicked.connect(self.new_smart_album_requested.emit)
            self.smart_grid.addWidget(create_smart, s_idx // cols, s_idx % cols)
