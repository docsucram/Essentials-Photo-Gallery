"""
Stripped-down reusable components extracted from legacy imgBrowserv13.py.
Contains only the classes and helpers retained for PhotoGallery64:
- CancellationFlag, LruCache
- EditCommand (QUndoCommand)
- ImageGraphicsView, CropScene
- ClickableVideoWidget, MouseOverlay, FavouritesMenu
- Video dark-frame detection & filmstrip overlay helpers
- EXIF human-readable formatting helpers
"""

import collections
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QIcon,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
    QTransform,
    QUndoCommand,
    QWheelEvent,
)
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QMenu,
    QWidget,
)

# --- Constants ---
IMAGE_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.ico', '*.tiff', '*.webp']
VIDEO_EXTENSIONS = ['*.mp4', '*.avi', '*.mkv', '*.mov', '*.wmv', '*.flv']
DEFAULT_THUMBNAIL_SIZE = 150
MASTER_THUMBNAIL_SIZE = 256
THUMBNAIL_PADDING = 10

ZOOM_STEP = 0.1
MIN_ZOOM = 0.1
MAX_ZOOM = 8.0


@dataclass
class CancellationFlag:
    """A simple mutable flag to signal cancellation to a worker thread."""
    is_set: bool = False

    def set(self) -> None:
        self.is_set = True

    def clear(self) -> None:
        self.is_set = False


class LruCache:
    """Thread-safe LRU Cache implementation for visible viewport QPixmaps."""
    def __init__(self, max_size: int = 2500):
        if max_size <= 0:
            raise ValueError("LruCache max_size must be positive")
        self._max_size = max_size
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._lock = threading.Lock()

    @property
    def max_size(self) -> int:
        return self._max_size

    @max_size.setter
    def max_size(self, value: int) -> None:
        if value <= 0:
            raise ValueError("LruCache max_size must be positive")
        with self._lock:
            self._max_size = value
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def get(self, key):
        with self._lock:
            try:
                value = self._cache.pop(key)
                self._cache[key] = value
                return value
            except KeyError:
                return None

    def put(self, key, value) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            elif len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def __getitem__(self, key):
        with self._lock:
            try:
                value = self._cache.pop(key)
                self._cache[key] = value
                return value
            except KeyError:
                raise KeyError(key)

    def __setitem__(self, key, value) -> None:
        self.put(key, value)

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._cache

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __delitem__(self, key) -> None:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            else:
                raise KeyError(key)

    def remove(self, key) -> None:
        """Safely removes key if present without raising KeyError."""
        with self._lock:
            self._cache.pop(key, None)


class EditCommand(QUndoCommand):
    """A command for the undo/redo framework for single-view image pixel edits."""
    def __init__(self, viewer_target, before_image, description: str, parent=None):
        super().__init__(description, parent)
        self.viewer_target = viewer_target
        self.before_image = before_image.copy()
        self.after_image = viewer_target.current_pil_image.copy()

    def undo(self) -> None:
        self.viewer_target.current_pil_image = self.before_image.copy()
        self.viewer_target.update_image_display()

    def redo(self) -> None:
        self.viewer_target.current_pil_image = self.after_image.copy()
        self.viewer_target.update_image_display()


class ImageGraphicsView(QGraphicsView):
    """Custom QGraphicsView for smooth panning and zoom control."""
    zoomChanged = Signal(int)
    clicked = Signal()
    mouseMovedInFullscreen = Signal()
    zoomModeSetToCustom = Signal()

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.zoom_level = 1.0

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self.scene() or self.scene().sceneRect().isEmpty():
            return
        zoom_mode = getattr(self.main_window, "current_zoom_mode", "fit") if self.main_window else "fit"
        if zoom_mode == "fit":
            self.fit_in_view_with_zoom(allow_upscale=False)
        elif zoom_mode == "fit_window":
            self.fit_in_view_with_zoom(allow_upscale=True)
        elif zoom_mode == "fill":
            self.fill_in_view_with_zoom()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if not self.isVisible() or not self.scene() or self.scene().sceneRect().isEmpty():
            return
        zoom_mode = getattr(self.main_window, "current_zoom_mode", "fit") if self.main_window else "fit"
        if zoom_mode == "fit":
            self.fit_in_view_with_zoom(allow_upscale=False)
        elif zoom_mode == "fit_window":
            self.fit_in_view_with_zoom(allow_upscale=True)
        elif zoom_mode == "fill":
            self.fill_in_view_with_zoom()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() == Qt.ControlModifier:
            super().wheelEvent(event)
            return
        self.zoomModeSetToCustom.emit()
        self.zoom(1 + ZOOM_STEP if event.angleDelta().y() > 0 else 1 - ZOOM_STEP)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self.mouseMovedInFullscreen.emit()
        super().mouseMoveEvent(event)

    def zoom(self, factor: float) -> None:
        self.zoom_level *= factor
        self.set_zoom(self.zoom_level)

    def set_zoom(self, zoom_level: float) -> None:
        self.zoom_level = max(MIN_ZOOM, min(zoom_level, MAX_ZOOM))
        transform = QTransform()
        transform.scale(self.zoom_level, self.zoom_level)
        self.setTransform(transform)
        self.zoomChanged.emit(int(self.zoom_level * 100))

    def fit_in_view_with_zoom(self, allow_upscale: bool = False) -> None:
        if not self.scene() or self.scene().sceneRect().isEmpty():
            return
        sr = self.scene().sceneRect()
        vp = self.viewport().rect()
        if (
            not allow_upscale
            and vp.width() > 0
            and vp.height() > 0
            and sr.width() <= vp.width()
            and sr.height() <= vp.height()
        ):
            self.setTransform(QTransform())
            self.centerOn(sr.center())
            self.zoom_level = 1.0
            self.zoomChanged.emit(100)
            return
        self.fitInView(sr, Qt.KeepAspectRatio)
        self.zoom_level = self.transform().m11()
        self.zoomChanged.emit(int(self.zoom_level * 100))

    def fill_in_view_with_zoom(self) -> None:
        if not self.scene() or self.scene().sceneRect().isEmpty():
            return
        view_rect = self.viewport().rect()
        scene_rect = self.scene().sceneRect()
        if scene_rect.width() > 0 and scene_rect.height() > 0:
            self.set_zoom(max(view_rect.width() / scene_rect.width(), view_rect.height() / scene_rect.height()))


from src.ui.crop_tool import CropScene


class ClickableVideoWidget(QVideoWidget):
    """A QVideoWidget that emits 'clicked' on mouse press and 'volumeWheelChanged' on scroll."""
    clicked = Signal()
    volumeWheelChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.angleDelta().y() > 0:
            self.volumeWheelChanged.emit(1)
        elif event.angleDelta().y() < 0:
            self.volumeWheelChanged.emit(-1)
        event.accept()


class MouseOverlay(QWidget):
    """Transparent widget sitting over video playback to capture mouse moves while passing clicks through."""
    mouseMoved = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background:transparent;")
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self.mouseMoved.emit()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        event.ignore()
        super().mousePressEvent(event)


class FavouritesMenu(QMenu):
    """QMenu supporting right-click removal of favourite folder items."""
    def __init__(self, owner_instance, parent=None):
        super().__init__(parent)
        self.owner = owner_instance

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.RightButton:
            action_under_mouse = self.actionAt(event.pos())
            if action_under_mouse:
                folder_path_to_remove = action_under_mouse.data()
                if isinstance(folder_path_to_remove, str):
                    try:
                        path_obj = Path(folder_path_to_remove)
                        item_context_menu = QMenu(self)
                        remove_action = QAction(f"Remove '{path_obj.name}'", self)
                        remove_action.triggered.connect(
                            lambda checked=False, p=folder_path_to_remove: self.owner._remove_folder_from_favourites(p)
                        )
                        item_context_menu.addAction(remove_action)
                        item_context_menu.exec(event.globalPos())
                        event.accept()
                        return
                    except Exception:
                        pass
        super().mouseReleaseEvent(event)


# --- Video Frame & Filmstrip Helpers ---

def is_frame_too_dark(frame, threshold: float = 10.0) -> bool:
    """Checks if a video BGR frame is predominantly black/dark."""
    if frame is None:
        return True
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    average_brightness = cv2.mean(gray_frame)[0]
    return average_brightness < threshold


def create_filmstrip_for_pixmap(frame_pixmap: QPixmap, target_size: int = MASTER_THUMBNAIL_SIZE) -> QPixmap:
    """Composites a video frame QPixmap onto a perforated filmstrip canvas."""
    filmstrip_w = 400
    filmstrip_h = 300
    strip_width = 30
    hole_size = 18
    hole_padding = 10

    filmstrip_pixmap = QPixmap(filmstrip_w, filmstrip_h)
    filmstrip_pixmap.fill(QColor("black"))

    available_width = filmstrip_w - (2 * strip_width)
    scaled_frame = frame_pixmap.scaledToWidth(available_width, Qt.SmoothTransformation)

    painter = QPainter(filmstrip_pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    frame_x = strip_width
    frame_y = (filmstrip_h - scaled_frame.height()) / 2
    painter.drawPixmap(int(frame_x), int(frame_y), scaled_frame)

    painter.setBrush(QColor(200, 200, 200))
    painter.setPen(Qt.NoPen)
    hole_rects = [
        QRectF(hole_x_left, current_y, hole_size, hole_size)
        for current_y in range(hole_padding, filmstrip_h - hole_size, hole_size + hole_padding)
        for hole_x_left in (
            (strip_width - hole_size) / 2,
            filmstrip_w - strip_width + (strip_width - hole_size) / 2,
        )
    ]
    painter.drawRects(hole_rects)
    painter.end()

    return filmstrip_pixmap.scaled(target_size, target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# --- EXIF / Metadata Display Formatting Helpers ---

def format_file_size(num_bytes: float) -> str:
    """Converts bytes to a human-readable string (KB, MB, GB)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.0f} {unit}" if unit == 'B' else f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} EB"


def _format_exif_tag_name(tag_name) -> str:
    if isinstance(tag_name, int):
        return f"Tag 0x{tag_name:04X}"
    s = str(tag_name)
    s = ''.join([' ' + char if char.isupper() and i > 0 and s[i - 1].islower() else char for i, char in enumerate(s)]).strip()
    return s.replace('_', ' ').title()


def _format_folder_path_for_display(full_file_path_str: str) -> tuple[str, str]:
    path_obj = Path(full_file_path_str)
    parent_path = path_obj.parent
    if not parent_path or str(parent_path) == '.':
        return str(parent_path), str(parent_path)
    if parent_path.parent and parent_path.parent != parent_path:
        display_name = f".../{parent_path.parent.name}/{parent_path.name}"
    else:
        display_name = parent_path.name if parent_path.name else str(parent_path)
    return display_name, str(parent_path)


def _format_exif_datetime(value_str, exif_data_readable=None) -> str:
    try:
        dt = time.strptime(str(value_str).strip(), "%Y:%m:%d %H:%M:%S")
        return time.strftime("%Y-%m-%d %H:%M:%S", dt)
    except ValueError:
        return str(value_str)


def _format_exif_flash(value: int) -> str:
    flash_modes = {
        0x00: "No Flash",
        0x01: "Fired",
        0x05: "Fired, Return not detected",
        0x07: "Fired, Return detected",
        0x09: "Fired, Compulsory",
        0x0D: "Fired, Compulsory, Return not detected",
        0x0F: "Fired, Compulsory, Return detected",
        0x10: "Not Fired, Compulsory",
        0x18: "Not Fired, Auto",
        0x19: "Fired, Auto",
        0x1D: "Fired, Auto, Return not detected",
        0x1F: "Fired, Auto, Return detected",
        0x20: "No flash function",
        0x41: "Fired, Red-eye reduction",
    }
    return flash_modes.get(value, f"Unknown (0x{value:04X})" if isinstance(value, int) else str(value))


def _format_exif_orientation(value: int) -> str:
    orientations = {
        1: "Normal (top-left)",
        2: "Mirrored horizontal",
        3: "Rotated 180°",
        4: "Mirrored vertical",
        5: "Mirrored horizontal, rotated 90° CW",
        6: "Rotated 90° CW",
        7: "Mirrored horizontal, rotated 270° CW",
        8: "Rotated 270° CW",
    }
    return orientations.get(value, f"Unknown ({value})")


def _format_exif_colorspace(value: int) -> str:
    return "sRGB" if value == 1 else "Uncalibrated" if value == 0xFFFF else f"Unknown ({value})"


def _get_exif_resolution_unit(exif_data_readable: dict) -> str:
    unit_val = exif_data_readable.get("ResolutionUnit")
    if unit_val == 2:
        return "dpi"
    if unit_val == 3:
        return "dpcm"
    return ""


def _to_degrees_from_exif_gps(value_tuple) -> float:
    d, m, s = value_tuple[0], value_tuple[1], value_tuple[2]
    return float(d) + (float(m) / 60.0) + (float(s) / 3600.0)


def _format_exif_gps(gps_data_dict: dict) -> str:
    try:
        lat_val = gps_data_dict.get(2) or gps_data_dict.get("GPSLatitude")
        lat_ref = gps_data_dict.get(1) or gps_data_dict.get("GPSLatitudeRef")
        lon_val = gps_data_dict.get(4) or gps_data_dict.get("GPSLongitude")
        lon_ref = gps_data_dict.get(3) or gps_data_dict.get("GPSLongitudeRef")
        if lat_val and lat_ref and lon_val and lon_ref:
            lat_deg = _to_degrees_from_exif_gps(lat_val)
            lon_deg = _to_degrees_from_exif_gps(lon_val)
            if str(lat_ref).upper().startswith('S'):
                lat_deg = -lat_deg
            if str(lon_ref).upper().startswith('W'):
                lon_deg = -lon_deg
            return f"{lat_deg:.6f}, {lon_deg:.6f}"
    except Exception:
        pass
    return "Unavailable"
