"""Reusable Bottom Dock hosting either Single Image View Filmstrip or active Create Workspace Tray."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

from PySide6.QtCore import (
    QAbstractListModel,
    QEvent,
    QModelIndex,
    QRect,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QImage, QImageReader, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QPushButton,
    QStackedWidget,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from src.ui.theme_tokens import get_theme_tokens

_FILMSTRIP_ID_ROLE = Qt.ItemDataRole.UserRole + 101
_FILMSTRIP_IS_VIDEO_ROLE = Qt.ItemDataRole.UserRole + 102
_FILMSTRIP_NAME_ROLE = Qt.ItemDataRole.UserRole + 103
_FILMSTRIP_PATH_ROLE = Qt.ItemDataRole.UserRole + 104
_FILMSTRIP_FAV_ROLE = Qt.ItemDataRole.UserRole + 105
_FILMSTRIP_FLAG_ROLE = Qt.ItemDataRole.UserRole + 106
_FILMSTRIP_RATING_ROLE = Qt.ItemDataRole.UserRole + 107


class _FilmstripListModel(QAbstractListModel):
    """Lightweight horizontal model filtering out group headers from CatalogListModel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: list[Any] = []
        self._id_to_row: dict[int, int] = {}

    def set_records(self, records: list[Any]) -> None:
        self.beginResetModel()
        self._items = [
            rec
            for rec in records
            if not getattr(rec, "is_group_header", False) and getattr(rec, "id", None) is not None
        ]
        self._id_to_row = {int(rec.id): idx for idx, rec in enumerate(self._items)}
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._items)

    def row_for_media_id(self, media_id: int | None) -> int | None:
        if media_id is None:
            return None
        return self._id_to_row.get(int(media_id))

    def media_id_at_row(self, row: int) -> int | None:
        if 0 <= row < len(self._items):
            return int(self._items[row].id)
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._items)):
            return None
        rec = self._items[index.row()]
        if role == _FILMSTRIP_ID_ROLE:
            return int(rec.id)
        if role == _FILMSTRIP_IS_VIDEO_ROLE:
            return bool(getattr(rec, "is_video", 0))
        if role == _FILMSTRIP_NAME_ROLE or role == Qt.ItemDataRole.ToolTipRole:
            return str(getattr(rec, "filename", ""))
        if role == _FILMSTRIP_PATH_ROLE:
            return str(getattr(rec, "file_path", "") or "")
        if role == _FILMSTRIP_FAV_ROLE:
            return bool(getattr(rec, "is_favourite", 0))
        if role == _FILMSTRIP_FLAG_ROLE:
            return int(getattr(rec, "flag", 0) or 0)
        if role == _FILMSTRIP_RATING_ROLE:
            return int(getattr(rec, "rating", 0) or 0)
        return None


class _FilmstripItemDelegate(QStyledItemDelegate):
    """Crisp rounded thumbnail item delegate for Single Image View Filmstrip."""

    ITEM_W = 112
    ITEM_H = 84

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap_cache: dict[int, QPixmap] | None = None
        self._thumb_repo: Any = None
        self._local_thumbs: OrderedDict[int, QPixmap] = OrderedDict()
        self._theme_name: str = "wlpg_aero"
        self.favourites_enabled: bool = True

    def set_sources(self, pixmap_cache: dict[int, QPixmap] | None, thumb_repo: Any) -> None:
        self._pixmap_cache = pixmap_cache
        self._thumb_repo = thumb_repo
        self._local_thumbs.clear()

    def set_theme_name(self, theme_name: str) -> None:
        self._theme_name = theme_name or "wlpg_aero"

    def set_item_height(self, h: int) -> bool:
        new_h = max(72, min(144, int(h)))
        new_w = int(round(new_h * 4 / 3))
        if new_h == self.ITEM_H and new_w == self.ITEM_W:
            return False
        self.ITEM_H = new_h
        self.ITEM_W = new_w
        return True

    def sizeHint(self, _option: QStyleOptionViewItem, _index: QModelIndex) -> QSize:
        return QSize(self.ITEM_W, self.ITEM_H)

    def _resolve_pixmap(self, media_id: int, file_path: str = "", is_video: bool = False) -> QPixmap | None:
        if self._pixmap_cache is not None and media_id in self._pixmap_cache:
            pm = self._pixmap_cache[media_id]
            if isinstance(pm, QPixmap) and not pm.isNull():
                return pm
        if media_id in self._local_thumbs:
            self._local_thumbs.move_to_end(media_id)
            return self._local_thumbs[media_id]
        if self._thumb_repo is not None:
            try:
                blob = self._thumb_repo.get_webp_blob(media_id)
                if blob:
                    qimg = QImage.fromData(blob)
                    if not qimg.isNull():
                        pm = QPixmap.fromImage(qimg)
                        self._local_thumbs[media_id] = pm
                        if len(self._local_thumbs) > 256:
                            self._local_thumbs.popitem(last=False)
                        return pm
            except Exception:
                pass
        if file_path and not is_video:
            try:
                reader = QImageReader(file_path)
                reader.setAutoTransform(True)
                orig = reader.size()
                if orig.isValid() and orig.width() > 0 and orig.height() > 0:
                    scaled_sz = orig.scaled(220, 165, Qt.AspectRatioMode.KeepAspectRatioByExpanding)
                    reader.setScaledSize(scaled_sz)
                qimg = reader.read()
                if not qimg.isNull():
                    pm = QPixmap.fromImage(qimg)
                    self._local_thumbs[media_id] = pm
                    if len(self._local_thumbs) > 256:
                        self._local_thumbs.popitem(last=False)
                    return pm
            except Exception:
                pass
        return None

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        mid = index.data(_FILMSTRIP_ID_ROLE)
        if mid is None:
            return
        is_video = bool(index.data(_FILMSTRIP_IS_VIDEO_ROLE))
        is_fav = bool(index.data(_FILMSTRIP_FAV_ROLE))
        flag_val = int(index.data(_FILMSTRIP_FLAG_ROLE) or 0)
        rating_val = int(index.data(_FILMSTRIP_RATING_ROLE) or 0)
        tok = get_theme_tokens(self._theme_name)
        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        is_hover = bool(option.state & QStyle.StateFlag.State_MouseOver)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        outer = option.rect.adjusted(2, 3, -2, -3)
        thumb_rect = outer.adjusted(2, 2, -2, -2)

        # Card / selection ring
        if is_selected:
            painter.setBrush(QColor(tok.selection_bg))
            painter.setPen(QPen(QColor(tok.accent), 2.2))
            painter.drawRoundedRect(outer, 6, 6)
        elif is_hover:
            painter.setBrush(QColor(tok.bg_hover))
            painter.setPen(QPen(QColor(tok.border_subtle), 1.0))
            painter.drawRoundedRect(outer, 6, 6)
        else:
            painter.setBrush(QColor(tok.bg_card))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(outer, 6, 6)

        pm = self._resolve_pixmap(int(mid), str(index.data(_FILMSTRIP_PATH_ROLE) or ""), is_video)
        clip_path = QPainterPath()
        clip_path.addRoundedRect(thumb_rect.x(), thumb_rect.y(), thumb_rect.width(), thumb_rect.height(), 4, 4)
        painter.setClipPath(clip_path)

        if pm is not None and not pm.isNull():
            scaled = pm.scaled(
                thumb_rect.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            sx = max(0, (scaled.width() - thumb_rect.width()) // 2)
            sy = max(0, (scaled.height() - thumb_rect.height()) // 2)
            painter.drawPixmap(
                thumb_rect.topLeft(),
                scaled,
                QRect(sx, sy, thumb_rect.width(), thumb_rect.height()),
            )
        else:
            painter.fillRect(thumb_rect, QColor(tok.bg_input))

        # Subtle focus dimming on inactive items (or stronger dim on rejected items)
        if flag_val == -1:
            painter.fillRect(thumb_rect, QColor(15, 23, 42, 125))
        elif not is_selected and not is_hover:
            painter.fillRect(thumb_rect, QColor(0, 0, 0, 38))

        painter.setClipping(False)

        # Top-right: Picked / Crossed-out Rejected Flag badge
        if flag_val in (1, -1):
            from PySide6.QtCore import QPointF
            from PySide6.QtGui import QPolygonF

            fr = QRect(thumb_rect.right() - 17, thumb_rect.top() + 3, 14, 14)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(15, 23, 42, 195))
            painter.drawRoundedRect(fr, 4, 4)
            px = fr.left() + 4
            painter.setPen(QPen(QColor("#E81123" if flag_val == 1 else "#CBD5E1"), 1.2))
            painter.drawLine(px, fr.top() + 2, px, fr.bottom() - 2)
            poly = QPolygonF([QPointF(px, fr.top() + 2), QPointF(fr.right() - 2, fr.top() + 5), QPointF(px, fr.top() + 8)])
            painter.setBrush(QColor("#E81123") if flag_val == 1 else Qt.BrushStyle.NoBrush)
            painter.drawPolygon(poly)
            if flag_val == -1:
                painter.setPen(QPen(QColor("#EF4444"), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(fr.left() + 2, fr.bottom() - 2, fr.right() - 2, fr.top() + 2)

        # Bottom-right squircle pill: Lap-style star-turns-into-heart when favourited (+ rating digit)
        show_fav = bool(is_fav and getattr(self, "favourites_enabled", True))
        if show_fav or rating_val > 0:
            from PySide6.QtCore import QPointF, QRectF
            from PySide6.QtGui import QLinearGradient, QPolygonF

            has_num = rating_val > 0
            pill_w = 28 if has_num else 18
            pill_h = 18
            # Shift left if video badge occupies bottom-right
            right_offset = 24 if is_video else 4
            pr = QRectF(
                thumb_rect.right() - pill_w - right_offset,
                thumb_rect.bottom() - pill_h - 4,
                pill_w,
                pill_h,
            )
            painter.setPen(QPen(QColor(255, 255, 255, 55), 0.9))
            painter.setBrush(QColor(15, 23, 42, 195))
            painter.drawRoundedRect(pr, 5.0, 5.0)

            icx = (pr.left() + 8.5) if has_num else pr.center().x()
            icy = pr.center().y()
            if show_fav:
                r_h = 4.3
                hp = QPainterPath()
                hp.moveTo(icx, icy + 1.0 * r_h)
                hp.cubicTo(icx - 1.15 * r_h, icy + 0.30 * r_h, icx - 1.20 * r_h, icy - 0.72 * r_h, icx - 0.64 * r_h, icy - 0.96 * r_h)
                hp.cubicTo(icx - 0.28 * r_h, icy - 1.12 * r_h, icx - 0.06 * r_h, icy - 0.76 * r_h, icx, icy - 0.46 * r_h)
                hp.cubicTo(icx + 0.06 * r_h, icy - 0.76 * r_h, icx + 0.28 * r_h, icy - 1.12 * r_h, icx + 0.64 * r_h, icy - 0.96 * r_h)
                hp.cubicTo(icx + 1.20 * r_h, icy - 0.72 * r_h, icx + 1.15 * r_h, icy + 0.30 * r_h, icx, icy + 1.0 * r_h)
                hgrad = QLinearGradient(icx, icy - r_h, icx, icy + r_h)
                hgrad.setColorAt(0.0, QColor("#FB7185"))
                hgrad.setColorAt(1.0, QColor("#E11D48"))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(hgrad)
                painter.drawPath(hp)
            else:
                # 5-point gold star
                import math
                pts: list[QPointF] = []
                for k in range(10):
                    ang = -math.pi / 2.0 + k * (math.pi / 5.0)
                    rad = 4.6 if (k % 2 == 0) else 2.0
                    pts.append(QPointF(icx + math.cos(ang) * rad, icy + math.sin(ang) * rad))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#FBBF24"))
                painter.drawPolygon(QPolygonF(pts))

            if has_num:
                painter.setPen(QColor("#F8FAFC"))
                painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                painter.drawText(
                    QRectF(pr.left() + 14.0, pr.top(), pr.width() - 16.0, pr.height()),
                    Qt.AlignmentFlag.AlignCenter,
                    str(rating_val),
                )

        if is_video:
            badge_w, badge_h = 18, 13
            bx = thumb_rect.right() - badge_w - 3
            by = thumb_rect.bottom() - badge_h - 3
            painter.setBrush(QColor(15, 23, 42, 190))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(bx, by, badge_w, badge_h, 3, 3)
            painter.setBrush(QColor("#FFFFFF"))
            tri_x, tri_y = bx + 6, by + 3
            from PySide6.QtCore import QPointF
            from PySide6.QtGui import QPolygonF

            painter.drawPolygon(
                QPolygonF([
                    QPointF(tri_x, tri_y),
                    QPointF(tri_x + 7, tri_y + 3.5),
                    QPointF(tri_x, tri_y + 7),
                ])
            )

        painter.restore()


class BottomFilmstripDock(QDockWidget):
    """
    Reusable bottom dock supporting:
    - Page 0: Optional Single Image View Filmstrip
    - Page 1: Create Studio Workspace Tray
    """

    media_selected = Signal(int)
    close_requested = Signal()
    height_anim_tick = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("", parent)
        self.setObjectName("BottomFilmstripDock")
        self.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        _empty_tb = QWidget(self)
        _empty_tb.setFixedHeight(0)
        self.setTitleBarWidget(_empty_tb)  # 0px hidden QDockWidget title bar

        self._theme_name: str = "wlpg_aero"
        self.favourites_enabled: bool = True
        self._catalog_model: Any = None
        self._pixmap_cache: dict[int, QPixmap] | None = None
        self._thumb_repo: Any = None
        self._hosted_tray: QWidget | None = None
        self._syncing_selection = False
        self._saved_filmstrip_height: int = 96
        self._animating_visibility: bool = False

        self.container = QFrame(self)
        self.container.setObjectName("BottomDockContainer")
        outer = QVBoxLayout(self.container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Compact Header Bar (shown in Create Workspace Tray mode; hidden in Filmstrip mode)
        header = QWidget(self.container)
        self._header = header
        header.setObjectName("BottomDockHeader")
        header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        header.setFixedHeight(22)
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(10, 1, 6, 1)
        h_lay.setSpacing(8)

        self.lbl_title = QLabel("Filmstrip", header)
        self.lbl_title.setFont(QFont("Segoe UI", 8.5, QFont.DemiBold))
        h_lay.addWidget(self.lbl_title)

        self.lbl_count = QLabel("", header)
        self.lbl_count.setFont(QFont("Segoe UI", 8))
        h_lay.addWidget(self.lbl_count)
        h_lay.addStretch(1)

        self.btn_close = QPushButton("✕", header)
        self.btn_close.setObjectName("BottomDockCloseBtn")
        self.btn_close.setFixedSize(20, 18)
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.setToolTip("Hide bottom panel")
        self.btn_close.clicked.connect(self.close_requested.emit)
        h_lay.addWidget(self.btn_close)
        outer.addWidget(header)

        # Stacked content: Page 0 = Filmstrip, Page 1 = Workspace Tray
        self.stack = QStackedWidget(self.container)
        outer.addWidget(self.stack, 1)

        # Page 0: Filmstrip with edge-overlay ‹ / › scroll pills
        self.page_filmstrip = QWidget(self.stack)
        self.page_filmstrip.setObjectName("FilmstripPage")
        self.page_filmstrip.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        p0_lay = QHBoxLayout(self.page_filmstrip)
        p0_lay.setContentsMargins(2, 0, 2, 2)
        p0_lay.setSpacing(0)

        self.filmstrip_model = _FilmstripListModel(self.page_filmstrip)
        self.filmstrip_delegate = _FilmstripItemDelegate(self.page_filmstrip)
        self.filmstrip_view = QListView(self.page_filmstrip)
        self.filmstrip_view.setObjectName("FilmstripListView")
        self.filmstrip_view.setFlow(QListView.Flow.LeftToRight)
        self.filmstrip_view.setWrapping(False)
        self.filmstrip_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.filmstrip_view.setUniformItemSizes(True)
        self.filmstrip_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.filmstrip_view.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.filmstrip_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.filmstrip_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.filmstrip_view.setFrameShape(QFrame.Shape.NoFrame)
        self.filmstrip_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.filmstrip_view.setModel(self.filmstrip_model)
        self.filmstrip_view.setItemDelegate(self.filmstrip_delegate)
        self.filmstrip_view.viewport().installEventFilter(self)
        self.page_filmstrip.installEventFilter(self)
        self.filmstrip_view.selectionModel().currentChanged.connect(self._on_filmstrip_current_changed)
        self.filmstrip_view.clicked.connect(self._on_filmstrip_clicked)
        p0_lay.addWidget(self.filmstrip_view, 1)

        from PySide6.QtGui import QIcon

        def _make_chevron_icon(direction: str) -> QIcon:
            pm = QPixmap(32, 32)
            pm.fill(Qt.GlobalColor.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setPen(QPen(QColor("#F8FAFC"), 2.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            if direction == "left":
                p.drawLine(19, 9, 12, 16)
                p.drawLine(12, 16, 19, 23)
            else:
                p.drawLine(13, 9, 20, 16)
                p.drawLine(20, 16, 13, 23)
            p.end()
            pm.setDevicePixelRatio(2.0)
            return QIcon(pm)

        _overlay_btn_qss = (
            "QPushButton { background-color: rgba(15, 23, 42, 135); "
            "border: 1px solid rgba(148, 163, 184, 85); border-radius: 9px; }"
            "QPushButton:hover { background-color: rgba(15, 23, 42, 230); "
            "border: 1px solid rgba(56, 189, 248, 190); }"
        )
        self.btn_scroll_left = QPushButton(self.page_filmstrip)
        self.btn_scroll_left.setObjectName("FilmstripOverlayScrollBtn")
        self.btn_scroll_left.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_scroll_left.setIcon(_make_chevron_icon("left"))
        self.btn_scroll_left.setIconSize(QSize(16, 16))
        self.btn_scroll_left.setFixedSize(32, 32)
        self.btn_scroll_left.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_scroll_left.setToolTip("Scroll filmstrip left")
        self.btn_scroll_left.setStyleSheet(_overlay_btn_qss)
        self.btn_scroll_left.clicked.connect(lambda: self._scroll_filmstrip(-3))

        self.btn_scroll_right = QPushButton(self.page_filmstrip)
        self.btn_scroll_right.setObjectName("FilmstripOverlayScrollBtn")
        self.btn_scroll_right.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_scroll_right.setIcon(_make_chevron_icon("right"))
        self.btn_scroll_right.setIconSize(QSize(16, 16))
        self.btn_scroll_right.setFixedSize(32, 32)
        self.btn_scroll_right.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_scroll_right.setToolTip("Scroll filmstrip right")
        self.btn_scroll_right.setStyleSheet(_overlay_btn_qss)
        self.btn_scroll_right.clicked.connect(lambda: self._scroll_filmstrip(3))

        # Page 1: Create Studio Workspace Tray Host
        self.page_tray = QWidget(self.stack)
        self.page_tray.setObjectName("BottomTrayHost")
        self.page_tray.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.tray_layout = QVBoxLayout(self.page_tray)
        self.tray_layout.setContentsMargins(6, 2, 6, 4)
        self.tray_layout.setSpacing(0)

        self.stack.addWidget(self.page_filmstrip)
        self.stack.addWidget(self.page_tray)

        self.setWidget(self.container)
        self.setMinimumHeight(92)
        self.setMaximumHeight(148)
        self.apply_theme_mode("wlpg_aero")

    def _position_scroll_buttons(self) -> None:
        host = getattr(self, "page_filmstrip", None)
        if host is None:
            return
        w, h = host.width(), host.height()
        cy = max(4, (h - self.btn_scroll_right.height()) // 2)
        self.btn_scroll_left.move(8, cy)
        self.btn_scroll_right.move(max(8, w - self.btn_scroll_right.width() - 8), cy)
        self.btn_scroll_left.raise_()
        self.btn_scroll_right.raise_()

    def saved_filmstrip_height(self) -> int:
        if self.isVisible() and self.stack.currentIndex() == 0 and not getattr(self, "_animating_visibility", False):
            h = self.height()
            if h >= 92:
                self._saved_filmstrip_height = max(92, min(148, h))
        return int(getattr(self, "_saved_filmstrip_height", 96))

    def set_saved_filmstrip_height(self, height: int) -> None:
        self._saved_filmstrip_height = max(92, min(148, int(height)))
        if self.stack.currentIndex() == 0 and self.isVisible():
            self._apply_saved_filmstrip_height()

    def _apply_saved_filmstrip_height(self) -> None:
        from PySide6.QtWidgets import QMainWindow
        h = max(92, min(148, int(getattr(self, "_saved_filmstrip_height", 96))))
        mw = self.parentWidget()
        if isinstance(mw, QMainWindow):
            mw.resizeDocks([self], [h], Qt.Orientation.Vertical)

    def animate_visible(self, show: bool) -> None:
        """Smoothly and quickly slide the filmstrip dock up or down."""
        from PySide6.QtCore import QEasingCurve, QPropertyAnimation

        old_anim = getattr(self, "_slide_anim", None)
        if old_anim is not None:
            try:
                old_anim.stop()
            except Exception:
                pass

        if not show and self.isVisible() and self.stack.currentIndex() == 0 and self.height() >= 92:
            self._saved_filmstrip_height = max(92, min(148, self.height()))

        target_h = max(92, min(148, int(getattr(self, "_saved_filmstrip_height", 96))))
        self._animating_visibility = True
        self.setMinimumHeight(0)
        anim = QPropertyAnimation(self, b"maximumHeight", self)
        anim.setDuration(140)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.valueChanged.connect(lambda _v: self.height_anim_tick.emit() if hasattr(self, "height_anim_tick") else None)
        if show:
            if not self.isVisible():
                self.setMaximumHeight(0)
                self.show()
            anim.setStartValue(self.height() if self.height() > 0 else 0)
            anim.setEndValue(target_h)

            def _on_shown() -> None:
                self._animating_visibility = False
                self.setMinimumHeight(92)
                self.setMaximumHeight(148)
                self._apply_saved_filmstrip_height()
                self._position_scroll_buttons()
                self.height_anim_tick.emit()

            anim.finished.connect(_on_shown)
        else:
            if not self.isVisible():
                self._animating_visibility = False
                return
            anim.setStartValue(self.height())
            anim.setEndValue(0)

            def _on_hidden() -> None:
                self._animating_visibility = False
                self.hide()
                self.setMinimumHeight(92)
                self.setMaximumHeight(148)

            anim.finished.connect(_on_hidden)

        self._slide_anim = anim
        anim.start()

    def eventFilter(self, watched: Any, event: QEvent) -> bool:
        if watched is self.page_filmstrip and event.type() == QEvent.Type.Resize:
            self._position_scroll_buttons()
        elif watched is self.filmstrip_view.viewport():
            if event.type() == QEvent.Type.Resize:
                if (
                    self.isVisible()
                    and self.stack.currentIndex() == 0
                    and not getattr(self, "_animating_visibility", False)
                    and self.height() >= 92
                ):
                    self._saved_filmstrip_height = max(92, min(148, self.height()))
                vp_h = self.filmstrip_view.viewport().height()
                if vp_h >= 70 and self.filmstrip_delegate.set_item_height(vp_h - 4):
                    self.filmstrip_view.doItemsLayout()
                self._position_scroll_buttons()
                self.height_anim_tick.emit()
            elif event.type() == QEvent.Type.Wheel:
                sb = self.filmstrip_view.horizontalScrollBar()
                if sb is not None:
                    delta = event.angleDelta().y() or event.angleDelta().x()
                    sb.setValue(sb.value() - int(delta * 0.6))
                    return True
        return super().eventFilter(watched, event)

    def bind_sources(
        self,
        catalog_model: Any,
        pixmap_cache: dict[int, QPixmap] | None = None,
        thumb_repo: Any = None,
    ) -> None:
        self._catalog_model = catalog_model
        self._pixmap_cache = pixmap_cache
        self._thumb_repo = thumb_repo
        self.filmstrip_delegate.set_sources(pixmap_cache, thumb_repo)
        self.refresh_items()

    def refresh_items(self) -> None:
        if self._catalog_model is None:
            self.filmstrip_model.set_records([])
            self.lbl_count.setText("")
            return
        records = list(getattr(self._catalog_model, "_records", []) or [])
        self.filmstrip_model.set_records(records)
        cnt = self.filmstrip_model.rowCount()
        self.lbl_count.setText(f"({cnt:,} items)" if cnt else "")

    def _scroll_filmstrip(self, step_items: int) -> None:
        sb = self.filmstrip_view.horizontalScrollBar()
        if sb is not None:
            sb.setValue(sb.value() + step_items * _FilmstripItemDelegate.ITEM_W)

    def set_active_media_id(self, media_id: int | None) -> None:
        row = self.filmstrip_model.row_for_media_id(media_id)
        if row is None:
            return
        self._syncing_selection = True
        try:
            idx = self.filmstrip_model.index(row, 0)
            self.filmstrip_view.setCurrentIndex(idx)
            self.filmstrip_view.scrollTo(idx, QAbstractItemView.ScrollHint.PositionAtCenter)
            cnt = self.filmstrip_model.rowCount()
            if self.stack.currentIndex() == 0 and cnt > 0:
                self.lbl_count.setText(f"({row + 1:,} of {cnt:,})")
        finally:
            self._syncing_selection = False

    def _on_filmstrip_current_changed(self, current: QModelIndex, _prev: QModelIndex) -> None:
        if self._syncing_selection or not current.isValid():
            return
        mid = self.filmstrip_model.media_id_at_row(current.row())
        if mid is not None:
            self.media_selected.emit(mid)

    def _on_filmstrip_clicked(self, index: QModelIndex) -> None:
        if not index.isValid():
            return
        mid = self.filmstrip_model.media_id_at_row(index.row())
        if mid is not None:
            self.media_selected.emit(mid)

    def set_filmstrip_mode(self) -> None:
        self.clear_tray_widget()
        self.stack.setCurrentIndex(0)
        if hasattr(self, "_header"):
            self._header.setVisible(False)
        self.setMinimumHeight(92)
        self.setMaximumHeight(148)
        self._apply_saved_filmstrip_height()
        self._position_scroll_buttons()

    def set_tray_widget(self, widget: QWidget | None, title: str = "") -> None:
        self.clear_tray_widget()
        if widget is None:
            return
        self._hosted_tray = widget
        widget.setParent(self.page_tray)
        self.tray_layout.addWidget(widget)
        self.stack.setCurrentIndex(1)
        if hasattr(self, "_header"):
            self._header.setVisible(True)
        self.lbl_title.setText(title or "Tools Tray")
        self.lbl_count.setText("")
        self.btn_close.setVisible(False)
        self.setMinimumHeight(104)
        self.setMaximumHeight(168)

    def clear_tray_widget(self) -> None:
        if self._hosted_tray is not None:
            self.tray_layout.removeWidget(self._hosted_tray)
            self._hosted_tray.setParent(None)
            self._hosted_tray = None

    def set_dark_mode(self, is_dark: bool) -> None:
        self.apply_theme_mode("dark" if is_dark else "wlpg_aero")

    def apply_theme_mode(self, theme_name: str) -> None:
        self._theme_name = theme_name or "wlpg_aero"
        tok = get_theme_tokens(self._theme_name)
        self.filmstrip_delegate.set_theme_name(self._theme_name)
        self.container.setStyleSheet(
            f"""
            QFrame#BottomDockContainer {{
                background-color: {tok.bg_surface};
                border-top: 1px solid {tok.border_subtle};
            }}
            QWidget#BottomDockHeader {{
                background-color: {tok.bg_card};
                border-bottom: 1px solid {tok.border_subtle};
            }}
            QWidget#BottomDockHeader QLabel {{
                color: {tok.text_primary};
                background: transparent;
            }}
            QPushButton#BottomDockCloseBtn {{
                border: none;
                background: transparent;
                color: {tok.text_secondary};
                font-weight: bold;
                border-radius: 3px;
            }}
            QPushButton#BottomDockCloseBtn:hover {{
                background-color: {tok.danger};
                color: {tok.text_on_accent};
            }}
            QPushButton#FilmstripScrollBtn {{
                background-color: {tok.bg_card};
                color: {tok.text_primary};
                border: 1px solid {tok.border_subtle};
                border-radius: 4px;
                padding: 0px;
                font-size: 13pt;
                font-weight: 600;
            }}
            QPushButton#FilmstripScrollBtn:hover {{
                background-color: {tok.bg_hover};
                border-color: {tok.accent};
            }}
            QWidget#FilmstripPage {{
                background-color: {tok.bg_surface};
                border: none;
            }}
            QListView#FilmstripListView {{
                background-color: {tok.bg_surface};
                border: none;
            }}
            QListView#FilmstripListView::viewport {{
                background-color: {tok.bg_surface};
            }}
            QWidget#BottomTrayHost {{
                background-color: {tok.bg_surface};
            }}
            """
        )
        self.lbl_count.setStyleSheet(f"color: {tok.text_secondary}; background: transparent;")
        self.filmstrip_view.viewport().update()
