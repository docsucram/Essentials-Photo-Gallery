"""
ImportMediaDialog (QDialog) — Authentic Windows Live Photo Gallery Camera, Memory Card & Folder Import Wizard.
Provides:
1. Device & Source Discovery:
   - Scans QStorageInfo.mountedVolumes() for removable drives, SD cards, and DCIM folders.
   - Provides Source dropdown + 'Browse for Folder...' option.
2. Background Scan & Deduplication:
   - Fast background scanner with 'Skip files already in catalog (N duplicates hidden)' deduplication.
3. Chronological Event Grouping:
   - Groups files by calendar date (DateTimeOriginal / mtime).
   - Each group rendered as collapsible card with inline 'Event / Folder Name' and 'Tags to apply'.
4. Destination Folder Structure & File Renaming:
   - Subfolder templates: YYYY\\YYYY-MM-DD [Event], YYYY\\YYYY_MM [Event], YYYY-MM-DD [Event], Single Folder.
   - File renaming: Original filename or YYYY-MM-DD HH.MM.SS.ext with burst collision resolution (_1, _2).
   - Live Path Preview label updating in real-time.
   - Safe copy worker with byte-size verification, optional source deletion, folder sync, and tag assignment.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from PIL import ExifTags, Image, ImageOps
from PySide6.QtCore import (
    QObject,
    QPoint,
    QRect,
    QRectF,
    QRunnable,
    QSettings,
    QSize,
    QStandardPaths,
    QStorageInfo,
    Qt,
    QThreadPool,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.db.repository import CatalogRepository
from src.imaging.image_loader import NEXTGEN_DOT_EXT_SET, RAW_DOT_EXT_SET
from src.ui.ribbon_toolbar import make_wlpg_icon
from src.utils.wpd_mtp import MtpDeviceSession, enumerate_mtp_devices


SUPPORTED_PHOTO_EXTS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".ico"
} | set(RAW_DOT_EXT_SET) | set(NEXTGEN_DOT_EXT_SET)
SUPPORTED_VIDEO_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v", ".flv", ".webm", ".3gp", ".ts", ".mts", ".m2ts"
}
SUPPORTED_EXTS = SUPPORTED_PHOTO_EXTS | SUPPORTED_VIDEO_EXTS


@dataclass
class DiscoveredItem:
    source_path: Path
    filename: str
    file_size: int
    date_taken: datetime
    is_video: bool
    is_duplicate: bool = False
    selected: bool = True
    mtp_device_id: str | None = None
    mtp_object_id: str | None = None
    thumbnail_bytes: bytes | None = None


@dataclass
class ImportSource:
    label: str
    source_type: str  # "folder" | "mtp"
    path: Path | None = None
    has_dcim: bool = False
    mtp_device_id: str | None = None
    mtp_device_name: str | None = None


def extract_file_datetime(path: Path) -> datetime:
    """Extracts date taken from EXIF or falls back to file modification time."""
    ext = path.suffix.lower()
    if ext in SUPPORTED_PHOTO_EXTS:
        try:
            with Image.open(path) as im:
                ex = im.getexif()
                if ex:
                    dt_val = None
                    ex_ifd = ex.get_ifd(ExifTags.IFD.Exif)
                    if ex_ifd:
                        dt_val = ex_ifd.get(36867) or ex_ifd.get(36868)
                    if not dt_val:
                        dt_val = ex.get(36867) or ex.get(36868) or ex.get(306)
                    if dt_val:
                        s = str(dt_val).strip().replace("-", ":")
                        parts = s.split(" ")
                        if len(parts) == 2:
                            d = [int(p) for p in parts[0].split(":")]
                            t = [int(p) for p in parts[1].split(":")]
                            return datetime(d[0], d[1], d[2], t[0], t[1], t[2])
        except Exception:
            pass
    try:
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime)
    except Exception:
        return datetime.now()


def generate_destination_subfolder(template_idx: int, dt: datetime, event_name: str) -> Path:
    """Computes destination subfolder path for given template index and event name."""
    year = dt.strftime("%Y")
    ymd = dt.strftime("%Y-%m-%d")
    ym = dt.strftime("%Y_%m")
    ev = event_name.strip()
    suffix = f" {ev}" if ev else ""

    if template_idx == 0:
        return Path(year) / f"{ymd}{suffix}"
    elif template_idx == 1:
        return Path(year) / f"{ym}{suffix}"
    elif template_idx == 2:
        return Path(f"{ymd}{suffix}")
    else:
        return Path(ev) if ev else Path(".")


def resolve_target_filename(
    renaming_idx: int,
    orig_filename: str,
    dt: datetime,
    allocated_names: set[str],
    dest_dir: Path | None = None,
) -> str:
    """Computes target filename resolving same-second burst collisions with _1, _2."""
    ext = Path(orig_filename).suffix
    if renaming_idx == 0:
        base = Path(orig_filename).stem
    else:
        base = dt.strftime("%Y-%m-%d %H.%M.%S")

    candidate = f"{base}{ext}"
    idx = 1
    while candidate.lower() in allocated_names or (dest_dir and (dest_dir / candidate).exists()):
        candidate = f"{base}_{idx}{ext}"
        idx += 1
    allocated_names.add(candidate.lower())
    return candidate


def detect_available_sources() -> list[tuple[str, Path, bool]]:
    """Discovers removable volumes, DCIM directories, and standard photo staging folders."""
    sources: list[tuple[str, Path, bool]] = []
    seen_paths: set[Path] = set()

    for v in QStorageInfo.mountedVolumes():
        if not v.isReady():
            continue
        root = Path(v.rootPath())
        if root in seen_paths:
            continue
        seen_paths.add(root)

        dcim = root / "DCIM"
        has_dcim = dcim.exists() and dcim.is_dir()
        name = v.displayName() or v.name() or "Removable Disk"
        root_str = str(root).rstrip("\\/")
        if has_dcim:
            label = f"Camera / Memory Card ({root_str}) - {name} [DCIM]"
            sources.insert(0, (label, dcim, True))
        else:
            label = f"Drive ({root_str}) - {name}"
            sources.append((label, root, False))

    # Standard user folders
    pics = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
    if pics and Path(pics).exists() and Path(pics) not in seen_paths:
        sources.append((f"Pictures Library ({pics})", Path(pics), False))
    dls = QStandardPaths.writableLocation(QStandardPaths.DownloadLocation)
    if dls and Path(dls).exists() and Path(dls) not in seen_paths:
        sources.append((f"Downloads Folder ({dls})", Path(dls), False))

    return sources


def detect_all_import_sources() -> list[ImportSource]:
    """Discovers MTP/PTP cameras & smartphones plus mounted volumes and standard folders."""
    all_sources: list[ImportSource] = []

    # 1. MTP / PTP Cameras & Smartphones via Windows Portable Devices (WPD)
    try:
        for dev in enumerate_mtp_devices():
            all_sources.append(
                ImportSource(
                    label=dev.label,
                    source_type="mtp",
                    has_dcim=True,
                    mtp_device_id=dev.pnp_id,
                    mtp_device_name=dev.friendly_name or dev.description or "Portable Device",
                )
            )
    except Exception:
        pass

    # 2. Mounted volumes, DCIM memory cards, and standard user folders
    for label, path, has_dcim in detect_available_sources():
        all_sources.append(
            ImportSource(
                label=label,
                source_type="folder",
                path=path,
                has_dcim=has_dcim,
            )
        )

    return all_sources


class _ScannerSignals(QObject):
    finished = Signal(list, int)  # (items: list[DiscoveredItem], duplicate_count: int)
    progress = Signal(int)
    error = Signal(str)


class _BackgroundScanner(QRunnable):
    def __init__(
        self,
        source_dir: Path | None,
        existing_name_size: set[tuple[str, int]],
        existing_size_date: set[tuple[int, int]],
        signals: _ScannerSignals,
        mtp_device_id: str | None = None,
        mtp_device_name: str | None = None,
    ) -> None:
        super().__init__()
        self.source_dir = source_dir
        self.existing_name_size = existing_name_size
        self.existing_size_date = existing_size_date
        self.signals = signals
        self.mtp_device_id = mtp_device_id
        self.mtp_device_name = mtp_device_name or "Device"

    def _is_duplicate(self, fname: str, size: int, dt: datetime) -> bool:
        dt_epoch = int(dt.timestamp())
        return (
            (fname.lower(), size) in self.existing_name_size
            or (size, dt_epoch) in self.existing_size_date
        )

    def _run_mtp_scan(self) -> tuple[list[DiscoveredItem], int]:
        discovered: list[DiscoveredItem] = []
        dup_count = 0
        with MtpDeviceSession(self.mtp_device_id or "") as sess:
            mtp_items = sess.scan_media(
                SUPPORTED_EXTS,
                SUPPORTED_VIDEO_EXTS,
                progress_cb=self.signals.progress.emit,
            )
            for obj in mtp_items:
                is_dup = self._is_duplicate(obj.filename, obj.file_size, obj.date_taken)
                if is_dup:
                    dup_count += 1
                rel = f"{obj.relative_folder}/{obj.filename}".strip("/")
                discovered.append(
                    DiscoveredItem(
                        source_path=Path(rel),
                        filename=obj.filename,
                        file_size=obj.file_size,
                        date_taken=obj.date_taken,
                        is_video=obj.is_video,
                        is_duplicate=is_dup,
                        selected=not is_dup,
                        mtp_device_id=obj.device_pnp_id,
                        mtp_object_id=obj.object_id,
                        thumbnail_bytes=obj.thumbnail_bytes,
                    )
                )
        return discovered, dup_count

    def run(self) -> None:
        discovered: list[DiscoveredItem] = []
        dup_count = 0
        try:
            if self.mtp_device_id:
                try:
                    discovered, dup_count = self._run_mtp_scan()
                except Exception as exc:
                    self.signals.error.emit(str(exc))
                    return
                self.signals.finished.emit(discovered, dup_count)
                return

            if not self.source_dir:
                self.signals.finished.emit([], 0)
                return

            entries: list[Path] = []
            for root, _, files in os.walk(self.source_dir):
                for f in files:
                    p = Path(root) / f
                    if p.suffix.lower() in SUPPORTED_EXTS:
                        entries.append(p)

            for i, p in enumerate(entries):
                try:
                    st = p.stat()
                    size = st.st_size
                    dt = extract_file_datetime(p)
                    fname = p.name
                    is_vid = p.suffix.lower() in SUPPORTED_VIDEO_EXTS
                    is_dup = self._is_duplicate(fname, size, dt)
                    if is_dup:
                        dup_count += 1

                    discovered.append(
                        DiscoveredItem(
                            source_path=p,
                            filename=fname,
                            file_size=size,
                            date_taken=dt,
                            is_video=is_vid,
                            is_duplicate=is_dup,
                            selected=not is_dup,
                        )
                    )
                except Exception:
                    pass

                if i % 25 == 0:
                    self.signals.progress.emit(i)
        except Exception:
            pass

        self.signals.finished.emit(discovered, dup_count)


class ImportCopySignals(QObject):
    progress = Signal(int, int, str)  # (copied, total, filename)
    finished = Signal(list, list)      # (created_folders, imported_items)
    error = Signal(str)


class SafeCopyWorker(QRunnable):
    def __init__(
        self,
        dest_root: Path,
        template_idx: int,
        renaming_idx: int,
        delete_source: bool,
        groups_data: list[dict[str, Any]],
        signals: ImportCopySignals,
    ) -> None:
        super().__init__()
        self.dest_root = dest_root
        self.template_idx = template_idx
        self.renaming_idx = renaming_idx
        self.delete_source = delete_source
        self.groups_data = groups_data
        self.signals = signals

    def run(self) -> None:
        total_items = sum(len(g["items"]) for g in self.groups_data)
        copied = 0
        created_folders: set[Path] = set()
        imported_entries: list[tuple[Path, list[str]]] = []
        mtp_sessions: dict[str, MtpDeviceSession] = {}

        try:
            for g in self.groups_data:
                ev_name = str(g.get("event_name", ""))
                tags = [t.strip() for t in str(g.get("tags", "")).split(",") if t.strip()]
                allocated_names: set[str] = set()

                for item in g["items"]:
                    src: Path = item.source_path
                    dt: datetime = item.date_taken

                    subfolder = generate_destination_subfolder(self.template_idx, dt, ev_name)
                    target_dir = self.dest_root / subfolder
                    target_dir.mkdir(parents=True, exist_ok=True)
                    created_folders.add(target_dir)

                    target_name = resolve_target_filename(
                        self.renaming_idx,
                        item.filename,
                        dt,
                        allocated_names,
                        target_dir,
                    )
                    dest = target_dir / target_name

                    if item.mtp_device_id and item.mtp_object_id:
                        sess = mtp_sessions.get(item.mtp_device_id)
                        if sess is None:
                            sess = MtpDeviceSession(item.mtp_device_id).__enter__()
                            mtp_sessions[item.mtp_device_id] = sess
                        sess.copy_object_to_file(item.mtp_object_id, dest, expected_size=item.file_size)
                        if self.delete_source:
                            try:
                                sess.delete_object(item.mtp_object_id)
                            except Exception:
                                pass
                    else:
                        # Safe copy with metadata preservation
                        shutil.copy2(src, dest)

                        # Byte size verification
                        if dest.stat().st_size != src.stat().st_size:
                            raise IOError(f"Byte size mismatch verifying copied file: {dest.name}")

                        # Optional safe delete from source
                        if self.delete_source:
                            try:
                                src.unlink()
                            except Exception:
                                pass

                    imported_entries.append((dest, tags))
                    copied += 1
                    self.signals.progress.emit(copied, total_items, item.filename)

            self.signals.finished.emit(list(created_folders), imported_entries)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            for sess in mtp_sessions.values():
                try:
                    sess.__exit__(None, None, None)
                except Exception:
                    pass


class _ThumbnailItemWidget(QFrame):
    """Small thumbnail card preview for a single photo in the event group."""

    selection_changed = Signal(bool)

    def __init__(self, item: DiscoveredItem, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.item = item
        self.setFixedSize(92, 108)
        self.setStyleSheet(
            """
            _ThumbnailItemWidget {
                background: #FFFFFF;
                border: 1px solid #D5DFEB;
                border-radius: 3px;
            }
            _ThumbnailItemWidget:hover {
                background: #F0F6FC;
                border: 1px solid #70A6E8;
            }
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Top row: Checkbox
        h_top = QHBoxLayout()
        h_top.setContentsMargins(0, 0, 0, 0)
        self.chk = QCheckBox(self)
        self.chk.setChecked(item.selected)
        self.chk.toggled.connect(self._on_toggled)
        h_top.addWidget(self.chk)
        h_top.addStretch(1)

        if item.is_video:
            lbl_vid = QLabel("▶", self)
            lbl_vid.setStyleSheet("color: #0078D7; font-size: 8pt; font-weight: bold;")
            h_top.addWidget(lbl_vid)
        layout.addLayout(h_top)

        # Center thumbnail
        self.lbl_thumb = QLabel(self)
        self.lbl_thumb.setFixedSize(84, 58)
        self.lbl_thumb.setAlignment(Qt.AlignCenter)
        self.lbl_thumb.setStyleSheet("background: #EAEFF5; border-radius: 2px;")
        layout.addWidget(self.lbl_thumb)

        # Bottom label: name or time
        time_str = item.date_taken.strftime("%H:%M:%S")
        self.lbl_text = QLabel(time_str, self)
        self.lbl_text.setAlignment(Qt.AlignCenter)
        self.lbl_text.setStyleSheet("font-size: 7.5pt; color: #444444;")
        self.lbl_text.setToolTip(f"{item.filename} ({item.file_size // 1024:,} KB)")
        layout.addWidget(self.lbl_text)

        self._load_fast_preview()

    def _on_toggled(self, checked: bool) -> None:
        self.item.selected = checked
        self.selection_changed.emit(checked)

    def _load_fast_preview(self) -> None:
        try:
            if self.item.thumbnail_bytes:
                qimg = QImage.fromData(self.item.thumbnail_bytes)
                if not qimg.isNull():
                    pix = QPixmap.fromImage(qimg).scaled(
                        84, 58, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    self.lbl_thumb.setPixmap(pix)
                    return

            if self.item.mtp_device_id:
                self.lbl_thumb.setText("🎬 Video" if self.item.is_video else "📷 Photo")
                return

            if not self.item.source_path.exists():
                return
            if not self.item.is_video:
                with Image.open(self.item.source_path) as im:
                    im.thumbnail((84, 58), Image.Resampling.BILINEAR)
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    data = im.tobytes("raw", "RGB")
                    qimg = QImage(data, im.width, im.height, im.width * 3, QImage.Format_RGB888)
                    self.lbl_thumb.setPixmap(QPixmap.fromImage(qimg))
            else:
                self.lbl_thumb.setText("🎬 Video")
        except Exception:
            self.lbl_thumb.setText("Photo")


class _DateGroupCard(QFrame):
    """Collapsible date group card with group checkbox, inline event name, and tags."""

    counts_changed = Signal()

    def __init__(self, date_str: str, items: list[DiscoveredItem], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.date_str = date_str
        self.items = items
        self.is_collapsed: bool = False

        self.setStyleSheet(
            """
            _DateGroupCard {
                background: #FFFFFF;
                border: 1px solid #C4D5E8;
                border-radius: 4px;
            }
            """
        )
        v_layout = QVBoxLayout(self)
        v_layout.setContentsMargins(8, 8, 8, 8)
        v_layout.setSpacing(6)

        # Header Frame
        self.header_frame = QFrame(self)
        self.header_frame.setStyleSheet(
            "QFrame { background-color: #F4F7FB; border: 1px solid #D0DBE5; border-radius: 3px; padding: 2px; }"
        )
        h_head = QHBoxLayout(self.header_frame)
        h_head.setContentsMargins(6, 4, 6, 4)
        h_head.setSpacing(8)

        self.btn_toggle = QToolButton(self.header_frame)
        self.btn_toggle.setText("▼")
        self.btn_toggle.setStyleSheet("border: none; font-size: 9pt; color: #444444;")
        self.btn_toggle.clicked.connect(self._toggle_collapse)
        h_head.addWidget(self.btn_toggle)

        self.chk_group = QCheckBox(self.header_frame)
        self.chk_group.setStyleSheet("font-weight: bold; color: #1E395B; font-size: 9.5pt;")
        self.chk_group.setChecked(True)
        self.chk_group.toggled.connect(self._on_group_toggled)
        h_head.addWidget(self.chk_group)

        # Inline Event Name input
        lbl_ev = QLabel("Event / Folder Name:", self.header_frame)
        lbl_ev.setStyleSheet("color: #445566; font-size: 8.5pt;")
        h_head.addWidget(lbl_ev)
        self.txt_event_name = QLineEdit(self.header_frame)
        self.txt_event_name.setPlaceholderText("e.g. Dorset Trip")
        self.txt_event_name.setFixedWidth(160)
        self.txt_event_name.setStyleSheet("background: #FFFFFF; border: 1px solid #B0C4DE; padding: 2px 4px; border-radius: 2px;")
        h_head.addWidget(self.txt_event_name)

        # Inline Tags input
        lbl_tags = QLabel("Tags:", self.header_frame)
        lbl_tags.setStyleSheet("color: #445566; font-size: 8.5pt;")
        h_head.addWidget(lbl_tags)
        self.txt_tags = QLineEdit(self.header_frame)
        self.txt_tags.setPlaceholderText("e.g. Holiday, Beach")
        self.txt_tags.setFixedWidth(140)
        self.txt_tags.setStyleSheet("background: #FFFFFF; border: 1px solid #B0C4DE; padding: 2px 4px; border-radius: 2px;")
        h_head.addWidget(self.txt_tags)

        h_head.addStretch(1)
        v_layout.addWidget(self.header_frame)

        # Items flow grid
        self.grid_container = QWidget(self)
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(4, 4, 4, 4)
        self.grid_layout.setSpacing(6)

        self.item_widgets: list[_ThumbnailItemWidget] = []
        col = 0
        row = 0
        max_cols = 8
        for it in self.items:
            w = _ThumbnailItemWidget(it, self.grid_container)
            w.selection_changed.connect(self._on_item_selection_changed)
            self.item_widgets.append(w)
            self.grid_layout.addWidget(w, row, col)
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        v_layout.addWidget(self.grid_container)
        self._update_header_title()

    def _toggle_collapse(self) -> None:
        self.is_collapsed = not self.is_collapsed
        self.btn_toggle.setText("►" if self.is_collapsed else "▼")
        self.grid_container.setVisible(not self.is_collapsed)

    def _on_group_toggled(self, checked: bool) -> None:
        for w in self.item_widgets:
            w.chk.blockSignals(True)
            w.chk.setChecked(checked)
            w.item.selected = checked
            w.chk.blockSignals(False)
        self._update_header_title()
        self.counts_changed.emit()

    def _on_item_selection_changed(self, _: bool) -> None:
        sel_count = sum(1 for it in self.items if it.selected)
        self.chk_group.blockSignals(True)
        self.chk_group.setChecked(sel_count > 0)
        self.chk_group.blockSignals(False)
        self._update_header_title()
        self.counts_changed.emit()

    def _update_header_title(self) -> None:
        sel_count = sum(1 for it in self.items if it.selected)
        total = len(self.items)
        try:
            dt = datetime.strptime(self.date_str, "%Y-%m-%d")
            display_date = dt.strftime("%d %b %Y")
        except Exception:
            display_date = self.date_str
        self.chk_group.setText(f"{display_date} ({sel_count}/{total} selected)")

    def get_selected_items(self) -> list[DiscoveredItem]:
        return [it for it in self.items if it.selected]


class ImportMediaDialog(QDialog):
    """
    Windows Live Photo Gallery Camera, Memory Card & Folder Import Wizard.
    """

    import_completed = Signal(list)  # (created_folders: list[Path])

    SUBFOLDER_TEMPLATES = [
        "YYYY \\ YYYY-MM-DD [Event Name]",
        "YYYY \\ YYYY_MM [Event Name]",
        "YYYY-MM-DD [Event Name]",
        "Single Folder (No date subfolders)",
    ]

    RENAMING_TEMPLATES = [
        "Keep original filename (IMG_1234.jpg)",
        "Date & Time taken (YYYY-MM-DD HH.MM.SS.jpg)",
    ]

    def __init__(
        self,
        catalog_repo: CatalogRepository,
        initial_source: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.catalog_repo = catalog_repo
        self.setWindowTitle("Import Photos and Videos — Photo Gallery")
        self.resize(980, 680)
        self.setMinimumSize(800, 560)

        self._thread_pool = QThreadPool(self)
        self._thread_pool.setMaxThreadCount(3)
        self._scanner_signals = _ScannerSignals()
        self._scanner_signals.finished.connect(self._on_scan_finished)
        self._scanner_signals.progress.connect(self._on_scan_progress)
        self._scanner_signals.error.connect(self._on_scan_error)

        self._copy_signals = ImportCopySignals()
        self._copy_signals.progress.connect(self._on_copy_progress)
        self._copy_signals.finished.connect(self._on_copy_finished)
        self._copy_signals.error.connect(self._on_copy_error)

        self._all_items: list[DiscoveredItem] = []
        self._group_cards: list[_DateGroupCard] = []
        self._duplicate_count: int = 0

        self._existing_name_size: set[tuple[str, int]] = set()
        self._existing_size_date: set[tuple[int, int]] = set()
        self._load_catalog_cache()

        self._build_ui()
        self._load_settings()

        if initial_source and initial_source.exists():
            self._set_source_path(initial_source)
        else:
            self._populate_sources()

    def _load_catalog_cache(self) -> None:
        """Caches existing media (filename, size) and (size, date_taken) for instant deduplication."""
        try:
            conn = self.catalog_repo.engine.catalog_conn()
            rows = conn.execute("SELECT filename, size, date_taken FROM media;").fetchall()
            for r in rows:
                fn = str(r["filename"]).lower()
                sz = int(r["size"] or 0)
                dt = int(r["date_taken"]) if r["date_taken"] is not None else None
                self._existing_name_size.add((fn, sz))
                if dt is not None:
                    self._existing_size_date.add((sz, dt))
        except Exception:
            pass

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # ---------------------------------------------------------------------
        # 1. Top Source & Deduplication Bar
        # ---------------------------------------------------------------------
        top_frame = QFrame(self)
        top_frame.setStyleSheet(
            "QFrame { background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FDFEFE, stop:1 #EBF2F9); "
            "border: 1px solid #D5DFEB; border-radius: 4px; padding: 4px; }"
        )
        h_top = QHBoxLayout(top_frame)
        h_top.setContentsMargins(8, 6, 8, 6)
        h_top.setSpacing(8)

        lbl_src = QLabel("<b>Source:</b>", top_frame)
        lbl_src.setStyleSheet("color: #1E395B; font-size: 9pt;")
        h_top.addWidget(lbl_src)

        self.combo_source = QComboBox(top_frame)
        self.combo_source.setMinimumWidth(320)
        self.combo_source.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.combo_source.currentIndexChanged.connect(self._on_source_combo_changed)
        h_top.addWidget(self.combo_source)

        self.btn_browse_source = QPushButton("Browse for Folder...", top_frame)
        self.btn_browse_source.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; border: 1px solid #B0C4DE; padding: 4px 10px; border-radius: 2px; }"
            "QPushButton:hover { background-color: #E2EBF5; }"
        )
        self.btn_browse_source.clicked.connect(self._browse_source_folder)
        h_top.addWidget(self.btn_browse_source)

        self.btn_refresh_sources = QPushButton("Refresh", top_frame)
        self.btn_refresh_sources.setToolTip("Re-scan for connected cameras, smartphones (MTP/PTP), and drives")
        self.btn_refresh_sources.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; border: 1px solid #B0C4DE; padding: 4px 10px; border-radius: 2px; }"
            "QPushButton:hover { background-color: #E2EBF5; }"
        )
        self.btn_refresh_sources.clicked.connect(self._populate_sources)
        h_top.addWidget(self.btn_refresh_sources)

        h_top.addSpacing(12)

        self.chk_skip_duplicates = QCheckBox("Skip files already in catalog", top_frame)
        self.chk_skip_duplicates.setChecked(True)
        self.chk_skip_duplicates.setStyleSheet("font-weight: bold; color: #1E395B; font-size: 9pt;")
        self.chk_skip_duplicates.toggled.connect(self._on_skip_duplicates_toggled)
        h_top.addWidget(self.chk_skip_duplicates)

        h_top.addStretch(1)
        main_layout.addWidget(top_frame)

        # Status Summary Label
        self.lbl_status = QLabel("Ready to scan", self)
        self.lbl_status.setStyleSheet("color: #445566; font-size: 9pt; padding-left: 4px;")
        main_layout.addWidget(self.lbl_status)

        # ---------------------------------------------------------------------
        # 2. Center Groups Scroll Area
        # ---------------------------------------------------------------------
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { background: #F8FAFD; border: 1px solid #D5DFEB; border-radius: 4px; }")

        self.scroll_content = QWidget()
        self.groups_vbox = QVBoxLayout(self.scroll_content)
        self.groups_vbox.setContentsMargins(8, 8, 8, 8)
        self.groups_vbox.setSpacing(8)
        self.groups_vbox.addStretch(1)
        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area, 1)

        # ---------------------------------------------------------------------
        # 3. Bottom Import Settings Box
        # ---------------------------------------------------------------------
        self.grp_settings = QGroupBox("Import Settings", self)
        self.grp_settings.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #1E395B; border: 1px solid #D0DBE5; border-radius: 4px; margin-top: 6px; padding-top: 10px; }"
        )
        s_layout = QVBoxLayout(self.grp_settings)
        s_layout.setContentsMargins(10, 8, 10, 8)
        s_layout.setSpacing(6)

        # Destination Root
        h_dest = QHBoxLayout()
        h_dest.setSpacing(8)
        lbl_dest = QLabel("Import to:", self.grp_settings)
        lbl_dest.setFixedWidth(70)
        h_dest.addWidget(lbl_dest)

        self.txt_dest_root = QLineEdit(self.grp_settings)
        self.txt_dest_root.setReadOnly(True)
        self.txt_dest_root.setStyleSheet("background: #FFFFFF; border: 1px solid #B0C4DE; padding: 3px 6px; border-radius: 2px;")
        h_dest.addWidget(self.txt_dest_root, 1)

        self.btn_browse_dest = QPushButton("Browse...", self.grp_settings)
        self.btn_browse_dest.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; border: 1px solid #B0C4DE; padding: 3px 10px; border-radius: 2px; }"
            "QPushButton:hover { background-color: #E2EBF5; }"
        )
        self.btn_browse_dest.clicked.connect(self._browse_dest_root)
        h_dest.addWidget(self.btn_browse_dest)
        s_layout.addLayout(h_dest)

        # Templates Row
        h_tmpl = QHBoxLayout()
        h_tmpl.setSpacing(12)

        lbl_t1 = QLabel("Folder structure:", self.grp_settings)
        h_tmpl.addWidget(lbl_t1)
        self.combo_subfolder = QComboBox(self.grp_settings)
        self.combo_subfolder.addItems(self.SUBFOLDER_TEMPLATES)
        self.combo_subfolder.currentIndexChanged.connect(self._update_live_preview)
        h_tmpl.addWidget(self.combo_subfolder, 1)

        lbl_t2 = QLabel("File naming:", self.grp_settings)
        h_tmpl.addWidget(lbl_t2)
        self.combo_renaming = QComboBox(self.grp_settings)
        self.combo_renaming.addItems(self.RENAMING_TEMPLATES)
        self.combo_renaming.currentIndexChanged.connect(self._update_live_preview)
        h_tmpl.addWidget(self.combo_renaming, 1)
        s_layout.addLayout(h_tmpl)

        # Live Path Preview Label
        self.lbl_preview = QLabel("Preview:", self.grp_settings)
        self.lbl_preview.setStyleSheet("color: #2E6DA4; font-family: 'Segoe UI'; font-size: 8.5pt;")
        s_layout.addWidget(self.lbl_preview)

        # Delete Option
        self.chk_delete_source = QCheckBox("Delete original files after verifying import", self.grp_settings)
        self.chk_delete_source.setStyleSheet("color: #660000; font-size: 8.5pt;")
        s_layout.addWidget(self.chk_delete_source)

        main_layout.addWidget(self.grp_settings)

        # ---------------------------------------------------------------------
        # 4. Action Bar (Progress + Import / Cancel)
        # ---------------------------------------------------------------------
        b_bar = QHBoxLayout()
        b_bar.setContentsMargins(4, 4, 4, 4)
        b_bar.setSpacing(8)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #B0C4DE; border-radius: 3px; text-align: center; height: 20px; }"
            "QProgressBar::chunk { background-color: #2E8B57; }"
        )
        b_bar.addWidget(self.progress_bar, 1)

        b_bar.addStretch(1)

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setFixedWidth(80)
        self.btn_cancel.clicked.connect(self.reject)
        b_bar.addWidget(self.btn_cancel)

        self.btn_import = QPushButton("Import", self)
        self.btn_import.setFixedWidth(130)
        self.btn_import.setStyleSheet(
            "QPushButton { background-color: #0078D7; color: white; font-weight: bold; border-radius: 3px; padding: 6px 12px; }"
            "QPushButton:hover { background-color: #1A88E1; }"
            "QPushButton:disabled { background-color: #A0C5E8; color: #E0E0E0; }"
        )
        self.btn_import.clicked.connect(self._start_import)
        b_bar.addWidget(self.btn_import)

        main_layout.addLayout(b_bar)

    def _load_settings(self) -> None:
        settings = QSettings("PhotoGallery64", "ImportWizard")
        def_dest = settings.value("dest_root")
        if not def_dest or not Path(def_dest).exists():
            # Fallback to library root or Pictures
            pics = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
            def_dest = pics or str(Path.home() / "Pictures")
        self.txt_dest_root.setText(def_dest)

        sub_idx = int(settings.value("subfolder_template", 0))
        ren_idx = int(settings.value("renaming_template", 0))
        self.combo_subfolder.setCurrentIndex(max(0, min(sub_idx, len(self.SUBFOLDER_TEMPLATES) - 1)))
        self.combo_renaming.setCurrentIndex(max(0, min(ren_idx, len(self.RENAMING_TEMPLATES) - 1)))
        self._update_live_preview()

    def _save_settings(self) -> None:
        settings = QSettings("PhotoGallery64", "ImportWizard")
        settings.setValue("dest_root", self.txt_dest_root.text())
        settings.setValue("subfolder_template", self.combo_subfolder.currentIndex())
        settings.setValue("renaming_template", self.combo_renaming.currentIndex())

    def _populate_sources(self) -> None:
        self.combo_source.blockSignals(True)
        self.combo_source.clear()
        detected = detect_all_import_sources()
        for src in detected:
            self.combo_source.addItem(src.label, src)
        self.combo_source.blockSignals(False)

        if self.combo_source.count() > 0:
            src = self.combo_source.currentData()
            if isinstance(src, ImportSource):
                self._start_scan_source(src)

    def _set_source_path(self, path: Path) -> None:
        src = ImportSource(label=f"Custom Folder ({path})", source_type="folder", path=path)
        self.combo_source.blockSignals(True)
        self.combo_source.insertItem(0, src.label, src)
        self.combo_source.setCurrentIndex(0)
        self.combo_source.blockSignals(False)
        self._start_scan_source(src)

    def _browse_source_folder(self) -> None:
        from PySide6.QtCore import QStandardPaths
        pic = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation)
        start_dir = pic if (pic and Path(pic).is_dir()) else str(Path.home() / "Pictures" if (Path.home() / "Pictures").is_dir() else Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder / Camera Stage", start_dir)
        if folder:
            self._set_source_path(Path(folder))

    def _browse_dest_root(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Import Destination Folder", self.txt_dest_root.text())
        if folder:
            self.txt_dest_root.setText(folder)
            self._update_live_preview()

    def _on_source_combo_changed(self, idx: int) -> None:
        data = self.combo_source.itemData(idx)
        if isinstance(data, ImportSource):
            self._start_scan_source(data)
        elif data:
            self._start_scan(Path(data))

    def _start_scan_source(self, src: ImportSource) -> None:
        if src.source_type == "mtp" and src.mtp_device_id:
            dev_label = src.mtp_device_name or src.label
            self.lbl_status.setText(f"Scanning '{dev_label}' (MTP/PTP) for photos and videos...")
            self.btn_import.setEnabled(False)
            self._clear_groups()
            worker = _BackgroundScanner(
                source_dir=None,
                existing_name_size=self._existing_name_size,
                existing_size_date=self._existing_size_date,
                signals=self._scanner_signals,
                mtp_device_id=src.mtp_device_id,
                mtp_device_name=dev_label,
            )
            self._thread_pool.start(worker)
        elif src.path is not None:
            self._start_scan(src.path)

    def _start_scan(self, source_dir: Path) -> None:
        if not source_dir.exists():
            self.lbl_status.setText(f"Source folder not found: {source_dir}")
            return
        self.lbl_status.setText(f"Scanning '{source_dir.name}' for photos and videos...")
        self.btn_import.setEnabled(False)
        self._clear_groups()

        worker = _BackgroundScanner(
            source_dir=source_dir,
            existing_name_size=self._existing_name_size,
            existing_size_date=self._existing_size_date,
            signals=self._scanner_signals,
        )
        self._thread_pool.start(worker)

    def _on_scan_progress(self, count: int) -> None:
        self.lbl_status.setText(f"Scanning... ({count:,} files examined)")

    def _on_scan_error(self, err_msg: str) -> None:
        self._all_items = []
        self._duplicate_count = 0
        self._clear_groups()
        self.btn_import.setEnabled(False)
        self.lbl_status.setText(
            f"Could not read device ({err_msg}). Unlock your phone/camera, enable File Transfer (MTP), and click Refresh."
        )

    def _on_scan_finished(self, items: list[DiscoveredItem], dup_count: int) -> None:
        self._all_items = items
        self._duplicate_count = dup_count
        self._rebuild_groups()

    def _on_skip_duplicates_toggled(self) -> None:
        self._rebuild_groups()

    def _clear_groups(self) -> None:
        for card in self._group_cards:
            card.deleteLater()
        self._group_cards.clear()

    def _rebuild_groups(self) -> None:
        self._clear_groups()
        skip_dups = self.chk_skip_duplicates.isChecked()

        visible_items = [it for it in self._all_items if (not it.is_duplicate or not skip_dups)]
        dup_text = f" ({self._duplicate_count} duplicate{'s' if self._duplicate_count != 1 else ''} hidden)" if self._duplicate_count > 0 and skip_dups else ""
        self.chk_skip_duplicates.setText(f"Skip files already in catalog{dup_text}")

        # Group by date YYYY-MM-DD
        grouped: dict[str, list[DiscoveredItem]] = {}
        for it in visible_items:
            d_str = it.date_taken.strftime("%Y-%m-%d")
            grouped.setdefault(d_str, []).append(it)

        # Sort dates descending (newest first)
        for d_str in sorted(grouped.keys(), reverse=True):
            card = _DateGroupCard(d_str, grouped[d_str], self.scroll_content)
            card.counts_changed.connect(self._update_counts)
            self._group_cards.append(card)
            self.groups_vbox.insertWidget(self.groups_vbox.count() - 1, card)

        self._update_counts()
        self._update_live_preview()

    def _update_counts(self) -> None:
        total_selected = sum(len(c.get_selected_items()) for c in self._group_cards)
        total_size = sum(it.file_size for c in self._group_cards for it in c.get_selected_items())
        size_mb = total_size / (1024.0 * 1024.0)
        size_str = f"{size_mb / 1024.0:.2f} GB" if size_mb >= 1024.0 else f"{size_mb:.1f} MB"

        self.lbl_status.setText(f"Found {len(self._all_items)} item(s) • {total_selected} selected for import ({size_str})")
        self.btn_import.setText(f"Import ({total_selected} item{'s' if total_selected != 1 else ''})")
        self.btn_import.setEnabled(total_selected > 0)

    def _update_live_preview(self) -> None:
        dest_root = Path(self.txt_dest_root.text())
        t_idx = self.combo_subfolder.currentIndex()
        r_idx = self.combo_renaming.currentIndex()

        sample_dt = datetime.now()
        ev_sample = "Dorset Trip"
        if self._group_cards and self._group_cards[0].txt_event_name.text().strip():
            ev_sample = self._group_cards[0].txt_event_name.text().strip()

        sub = generate_destination_subfolder(t_idx, sample_dt, ev_sample)
        alloc: set[str] = set()
        fname = resolve_target_filename(r_idx, "IMG_1234.JPG", sample_dt, alloc)
        sample_path = dest_root / sub / fname
        self.lbl_preview.setText(f"<b>Example destination:</b> {sample_path}")

    def _start_import(self) -> None:
        dest_root = Path(self.txt_dest_root.text())
        if not dest_root.exists():
            try:
                dest_root.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                QMessageBox.critical(self, "Invalid Destination", f"Cannot create destination directory:\n{exc}")
                return

        groups_data: list[dict[str, Any]] = []
        for card in self._group_cards:
            sel = card.get_selected_items()
            if sel:
                groups_data.append({
                    "date_str": card.date_str,
                    "event_name": card.txt_event_name.text().strip(),
                    "tags": card.txt_tags.text().strip(),
                    "items": sel,
                })

        if not groups_data:
            return

        self._save_settings()

        # Disable buttons and show progress bar
        self.btn_import.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.btn_browse_source.setEnabled(False)
        self.btn_refresh_sources.setEnabled(False)
        self.btn_browse_dest.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        worker = SafeCopyWorker(
            dest_root=dest_root,
            template_idx=self.combo_subfolder.currentIndex(),
            renaming_idx=self.combo_renaming.currentIndex(),
            delete_source=self.chk_delete_source.isChecked(),
            groups_data=groups_data,
            signals=self._copy_signals,
        )
        self._thread_pool.start(worker)

    def _on_copy_progress(self, current: int, total: int, fname: str) -> None:
        pct = int((current / float(total)) * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(f"Copying {current} of {total}: {fname}...")

    def _on_copy_finished(self, created_folders: list[Path], imported_items: list[tuple[Path, list[str]]]) -> None:
        self.lbl_status.setText(f"✓ Successfully imported {len(imported_entries := imported_items)} items!")
        self.progress_bar.setValue(100)

        # Index created folders and apply tags in catalog.db
        try:
            from src.workers.scanner_thumb_worker import FolderSyncWorker
            for folder_path in created_folders:
                worker = FolderSyncWorker(self.catalog_repo.engine, folder_path, include_subfolders=False)
                worker.run()

            conn = self.catalog_repo.engine.catalog_conn()
            for dest_path, tags in imported_entries:
                if tags:
                    row = conn.execute(
                        """
                        SELECT m.id FROM media m
                        JOIN folders f ON m.folder_id = f.id
                        WHERE f.path = ? AND m.filename = ?;
                        """,
                        (str(dest_path.parent), dest_path.name),
                    ).fetchone()
                    if row:
                        mid = int(row["id"])
                        for tag_str in tags:
                            tag_id = self.catalog_repo.create_tag(tag_str)
                            self.catalog_repo.assign_tag([mid], tag_id)
        except Exception:
            pass

        self.import_completed.emit(created_folders)
        self.accept()

    def _on_copy_error(self, err_msg: str) -> None:
        self.btn_import.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.btn_browse_source.setEnabled(True)
        self.btn_refresh_sources.setEnabled(True)
        self.btn_browse_dest.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Import Error", f"An error occurred during file import:\n{err_msg}")
