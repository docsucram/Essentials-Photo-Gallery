"""
Picasa-Style Folder Manager Dialog:
Provides per-folder monitoring policy ('always' | 'once' | 'excluded'),
Background AI indexing toggles (Faces, OCR), Private Folder isolation,
and cascading subfolder policy application.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from pathlib import Path
from typing import Any

from PySide6.QtCore import SIGNAL, QModelIndex, QPointF, QRect, QRectF, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QStyle,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.db.repository import CatalogRepository
from src.ui.theme_tokens import set_ui_role


def _privacy_settings() -> QSettings:
    return QSettings("PhotoGallery64", "PhotoGallery64")


def has_private_pin() -> bool:
    return bool(str(_privacy_settings().value("privacy/pin_hash", "") or "").strip())


def hash_private_pin(pin: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), bytes.fromhex(salt), 100_000)
    return f"{salt}:{dk.hex()}"


def verify_private_pin(pin: str) -> bool:
    stored = str(_privacy_settings().value("privacy/pin_hash", "") or "").strip()
    if ":" not in stored:
        return False
    salt_hex, expected_hex = stored.split(":", 1)
    try:
        dk = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), bytes.fromhex(salt_hex), 100_000)
        return hmac.compare_digest(dk.hex(), expected_hex)
    except Exception:
        return False


def get_private_auto_lock_minutes() -> int:
    try:
        return max(0, int(_privacy_settings().value("privacy/auto_lock_minutes", 0, type=int)))
    except Exception:
        return 0


def get_private_auth_mode() -> str:
    """Return private folder unlock mode: 'none', 'pin', or 'windows'."""
    mode = str(_privacy_settings().value("privacy/auth_mode", "") or "").strip().lower()
    if mode in ("none", "pin", "windows"):
        return mode
    return "pin" if has_private_pin() else "none"


def verify_windows_login(message: str = "Unlock Private Folders in Essentials Photo Gallery") -> bool:
    """Prompt Windows Hello / Windows Login verification on Windows 10/11."""
    import shutil
    import subprocess
    import sys

    if sys.platform != "win32":
        return False
    ps_exe = shutil.which("powershell.exe") or shutil.which("pwsh")
    if not ps_exe:
        return False
    escaped = message.replace("'", "''")
    ps_script = (
        "Add-Type -AssemblyName System.Runtime.WindowsRuntime; "
        "$null = [Windows.Security.Credentials.UI.UserConsentVerifier,Windows.Security.Credentials.UI,ContentType=WindowsRuntime]; "
        "$m = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.IsGenericMethod })[0]; "
        "$op = [Windows.Security.Credentials.UI.UserConsentVerifier]::RequestVerificationAsync('" + escaped + "'); "
        "$t = $m.MakeGenericMethod([Windows.Security.Credentials.UI.UserConsentVerificationResult]).Invoke($null, @($op)); "
        "$t.Wait(); Write-Output $t.Result"
    )
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.run(
            [ps_exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            creationflags=flags,
        )
        return proc.returncode == 0 and "Verified" in (proc.stdout or "")
    except Exception:
        return False


def set_private_lock_settings(
    *,
    pin: str | None = None,
    clear_pin: bool = False,
    auto_lock_minutes: int | None = None,
    auth_mode: str | None = None,
) -> None:
    s = _privacy_settings()
    if clear_pin:
        s.setValue("privacy/pin_hash", "")
    elif pin is not None and pin != "":
        s.setValue("privacy/pin_hash", hash_private_pin(pin))
    if auth_mode in ("none", "pin", "windows"):
        s.setValue("privacy/auth_mode", auth_mode)
    elif clear_pin and get_private_auth_mode() == "pin":
        s.setValue("privacy/auth_mode", "none")
    elif pin is not None and pin != "" and auth_mode is None:
        s.setValue("privacy/auth_mode", "pin")
    if auto_lock_minutes is not None:
        s.setValue("privacy/auto_lock_minutes", max(0, int(auto_lock_minutes)))


def format_folder_badges(folder: dict[str, Any]) -> str:
    """Build compact Picasa-style status prefix + badges for a folder row."""
    mode = str(folder.get("monitor_mode") or "always").lower()
    if mode == "once":
        prefix = "✓"
    elif mode == "excluded":
        prefix = "❌"
    else:
        prefix = "👁️"

    name = str(folder.get("name") or Path(str(folder.get("path") or "")).name or folder.get("path") or "")
    badges: list[str] = []
    if int(folder.get("is_private") or 0) == 1:
        badges.append("🔒 Private")
    if int(folder.get("scan_faces") if folder.get("scan_faces") is not None else 1) == 0:
        badges.append("No Faces")
    if int(folder.get("scan_ocr") if folder.get("scan_ocr") is not None else 1) == 0:
        badges.append("No OCR")
    if mode == "excluded":
        badges.append("Excluded")

    badge_str = f"  [{', '.join(badges)}]" if badges else ""
    return f"{prefix} {name}{badge_str}"


def _draw_status_pixmap(mode: str, size: int = 16) -> QPixmap:
    """Paint a crisp Picasa-style status icon (refresh, green check, or red X)."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    m = mode.lower()
    if m == "once":
        pen = QPen(QColor("#16A34A"), 2.3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.drawLine(int(size * 0.2), int(size * 0.55), int(size * 0.42), int(size * 0.78))
        p.drawLine(int(size * 0.42), int(size * 0.78), int(size * 0.82), int(size * 0.25))
    elif m == "excluded":
        pen = QPen(QColor("#DC2626"), 2.3, Qt.SolidLine, Qt.RoundCap)
        p.setPen(pen)
        pad = int(size * 0.22)
        p.drawLine(pad, pad, size - pad, size - pad)
        p.drawLine(size - pad, pad, pad, size - pad)
    else:
        # Blue circular refresh arrow (Scan Always)
        blue = QColor("#1D6FE8")
        pen = QPen(blue, 2.0, Qt.SolidLine, Qt.RoundCap)
        p.setPen(pen)
        rect = QRectF(2.5, 2.5, size - 5.0, size - 5.0)
        p.drawArc(rect, 35 * 16, 285 * 16)
        # Arrowhead at top-right
        p.setBrush(blue)
        p.setPen(Qt.NoPen)
        ax = int(size * 0.78)
        ay = int(size * 0.30)
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QPolygon
        tri = QPolygon([QPoint(ax + 2, ay + 2), QPoint(ax - 4, ay + 1), QPoint(ax + 1, ay - 4)])
        p.drawPolygon(tri)
    p.end()
    return pm


_STATUS_ICON_CACHE: dict[tuple[str, int], QIcon] = {}
_FOLDER_ROW_ICON_CACHE: dict[str, QIcon] = {}
_FEATURE_ICON_CACHE: dict[tuple[str, bool], QIcon] = {}


def _make_status_icon(mode: str, size: int = 16) -> QIcon:
    key = (mode.lower(), int(size))
    cached = _STATUS_ICON_CACHE.get(key)
    if cached is not None:
        return cached
    icon = QIcon(_draw_status_pixmap(mode, size))
    _STATUS_ICON_CACHE[key] = icon
    return icon


def _make_folder_row_icon(mode: str) -> QIcon:
    """Composite [Status Icon] + [Standard OS Folder Icon] for the left tree."""
    key = mode.lower()
    cached = _FOLDER_ROW_ICON_CACHE.get(key)
    if cached is not None:
        return cached
    w, h = 36, 16
    pm = QPixmap(w, h)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    status_pm = _draw_status_pixmap(mode, 16)
    p.drawPixmap(0, 0, status_pm)

    app = QApplication.instance()
    if app is not None:
        folder_icon = app.style().standardIcon(QStyle.SP_DirIcon)
        folder_pm = folder_icon.pixmap(16, 16)
        p.drawPixmap(20, 0, folder_pm)
    p.end()
    icon = QIcon(pm)
    _FOLDER_ROW_ICON_CACHE[key] = icon
    return icon


def _make_feature_icon(kind: str, enabled: bool) -> QIcon:
    """Paint compact 16x16 right-side indicator icons for Faces, OCR, and Private."""
    key = (kind, bool(enabled))
    cached = _FEATURE_ICON_CACHE.get(key)
    if cached is not None:
        return cached
    size = 16
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)

    if kind == "private":
        color = QColor("#D97706")
        p.setPen(QPen(color, 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawArc(QRectF(5, 2, 6, 6), 0, 180 * 16)
        p.setBrush(color)
        p.drawRoundedRect(QRectF(3.5, 6.5, 9, 7.5), 1.5, 1.5)
    elif kind == "faces":
        color = QColor("#2563EB") if enabled else QColor("#CBD5E1")
        p.setPen(QPen(color, 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QRectF(5.0, 2.5, 6.0, 6.0))
        p.drawArc(QRectF(2.5, 9.0, 11.0, 7.0), 0, 180 * 16)
        if not enabled:
            p.setPen(QPen(QColor("#94A3B8"), 1.5, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(3, 13, 13, 3)
    elif kind == "ocr":
        color = QColor("#0D9488") if enabled else QColor("#CBD5E1")
        p.setPen(QPen(color, 1.4))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(2.5, 2.0, 11.0, 12.0), 1.5, 1.5)
        f = QFont()
        f.setPixelSize(9)
        f.setBold(True)
        p.setFont(f)
        p.drawText(QRectF(2.5, 2.0, 11.0, 12.0), Qt.AlignCenter, "T")
        if not enabled:
            p.setPen(QPen(QColor("#94A3B8"), 1.5, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(3, 13, 13, 3)
    p.end()
    icon = QIcon(pm)
    _FEATURE_ICON_CACHE[key] = icon
    return icon


class _FolderManagerTree(QTreeWidget):
    """QTreeWidget with high-contrast branch chevrons and subtle hierarchy guide lines."""

    def __init__(self, parent: QWidget | None = None, is_dark: bool = False) -> None:
        super().__init__(parent)
        self._is_dark = bool(is_dark)

    def set_dark_mode(self, is_dark: bool) -> None:
        self._is_dark = bool(is_dark)
        self.viewport().update()

    def drawBranches(self, painter: QPainter, rect: QRect, index: QModelIndex) -> None:
        if not index.isValid():
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        indent = max(16, self.indentation())
        guide_col = QColor("#334155") if self._is_dark else QColor("#CBD5E1")
        arrow_col = QColor("#E2E8F0") if self._is_dark else QColor("#334155")

        ancestors: list[QModelIndex] = []
        cur = index
        while cur.isValid():
            ancestors.append(cur)
            cur = cur.parent()
        ancestors.reverse()
        depth = len(ancestors) - 1

        cy = rect.top() + rect.height() / 2.0
        half_indent = indent / 2.0

        painter.setPen(QPen(guide_col, 1.0))
        for lvl in range(depth):
            x_lvl = rect.left() + lvl * indent + half_indent
            if lvl < depth - 1:
                anc = ancestors[lvl + 1]
                parent_idx = anc.parent()
                sib_count = self.model().rowCount(parent_idx)
                if anc.row() < sib_count - 1:
                    painter.drawLine(QPointF(x_lvl, rect.top()), QPointF(x_lvl, rect.bottom()))
            else:
                parent_idx = index.parent()
                sib_count = self.model().rowCount(parent_idx)
                has_next = index.row() < sib_count - 1
                y_end = float(rect.bottom()) if has_next else cy
                painter.drawLine(QPointF(x_lvl, rect.top()), QPointF(x_lvl, y_end))
                painter.drawLine(QPointF(x_lvl, cy), QPointF(x_lvl + half_indent - 2.0, cy))

        if self.model().hasChildren(index):
            cx = rect.left() + depth * indent + half_indent
            painter.setPen(QPen(arrow_col, 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            if self.isExpanded(index):
                painter.drawPolyline(
                    QPolygonF([
                        QPointF(cx - 4.0, cy - 2.0),
                        QPointF(cx, cy + 2.5),
                        QPointF(cx + 4.0, cy - 2.0),
                    ])
                )
            else:
                painter.drawPolyline(
                    QPolygonF([
                        QPointF(cx - 2.0, cy - 4.0),
                        QPointF(cx + 2.5, cy),
                        QPointF(cx - 2.0, cy + 4.0),
                    ])
                )

        painter.restore()


class FolderManagerDialog(QDialog):
    """Picasa-style Folder Manager for monitoring, AI indexing, and Private Folder policies."""

    policies_changed = Signal()
    scan_ocr_requested = Signal(int)
    open_ocr_options_requested = Signal()
    folder_add_requested = Signal(str)
    folder_removed = Signal(list)

    def __init__(
        self,
        repo: CatalogRepository,
        parent: QWidget | None = None,
        *,
        initial_folder_id: int | None = None,
        is_dark: bool | None = None,
    ) -> None:
        super().__init__(parent)
        self.repo = repo
        self._initial_folder_id = initial_folder_id
        self._folder_map: dict[int, dict[str, Any]] = {}
        self._item_map: dict[int, QTreeWidgetItem] = {}
        self._loading_ui = False
        if is_dark is None:
            is_dark = getattr(parent, "current_theme", "") == "dark" or (
                self.palette().color(self.backgroundRole()).lightness() < 128
            )
        self._is_dark = bool(is_dark)

        self.setWindowTitle("Folder Manager")
        self.resize(860, 680)
        self.setMinimumSize(780, 620)
        self._setup_ui()
        self._apply_theme_styles()
        self.refresh_tree(select_folder_id=initial_folder_id)

    def _setup_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        header = QLabel(
            "Configure how folders are monitored, indexed for Faces & Text (OCR), or hidden in Private Mode."
        )
        header.setWordWrap(True)
        root_layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal, self)
        root_layout.addWidget(splitter, 1)

        # Left Pane: Hierarchical Folder Tree
        left_frame = QFrame(splitter)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.addWidget(QLabel("Catalog & Monitored Folders:"))

        nav_row = QHBoxLayout()
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.setSpacing(4)
        self.txt_filter_folders = QLineEdit(left_frame)
        self.txt_filter_folders.setPlaceholderText("Filter folders...")
        self.txt_filter_folders.setClearButtonEnabled(True)
        self.txt_filter_folders.textChanged.connect(self._apply_folder_filter)
        nav_row.addWidget(self.txt_filter_folders, 1)

        self.btn_expand_all = QPushButton("Expand", left_frame)
        self.btn_expand_all.setToolTip("Expand all folders")
        self.btn_expand_all.clicked.connect(lambda: self.tree_folders.expandAll())
        nav_row.addWidget(self.btn_expand_all)

        self.btn_collapse_all = QPushButton("Collapse", left_frame)
        self.btn_collapse_all.setToolTip("Collapse to top-level folders")
        self.btn_collapse_all.clicked.connect(lambda: self.tree_folders.collapseAll())
        nav_row.addWidget(self.btn_collapse_all)
        left_layout.addLayout(nav_row)

        self.tree_folders = _FolderManagerTree(left_frame, is_dark=self._is_dark)
        self.tree_folders.setIndentation(20)
        self.tree_folders.setColumnCount(4)
        self.tree_folders.setHeaderHidden(True)
        self.tree_folders.setAlternatingRowColors(True)
        self.tree_folders.setUniformRowHeights(True)
        self.tree_folders.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree_folders.setIconSize(QSize(36, 16))
        hdr = self.tree_folders.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2, 3):
            hdr.setSectionResizeMode(col, QHeaderView.Fixed)
            self.tree_folders.setColumnWidth(col, 24)
        self.tree_folders.itemSelectionChanged.connect(self._on_tree_item_selection_changed)
        self.tree_folders.itemClicked.connect(self._on_tree_item_clicked)
        left_layout.addWidget(self.tree_folders, 1)

        folder_ops_row = QHBoxLayout()
        folder_ops_row.setContentsMargins(0, 2, 0, 0)
        folder_ops_row.setSpacing(6)
        self.btn_add_folder = QPushButton("+ Add Folder...", left_frame)
        self.btn_add_folder.setToolTip("Add a folder from disk to the catalog")
        self.btn_add_folder.clicked.connect(self._on_add_folder_clicked)
        folder_ops_row.addWidget(self.btn_add_folder)

        self.btn_remove_folder = QPushButton("Remove from Catalog...", left_frame)
        self.btn_remove_folder.setToolTip("Remove selected folder(s) from catalog (files on disk are untouched)")
        self.btn_remove_folder.clicked.connect(self._on_remove_folder_clicked)
        folder_ops_row.addWidget(self.btn_remove_folder)
        folder_ops_row.addStretch(1)
        left_layout.addLayout(folder_ops_row)

        self.legend_label = QLabel("↻ Scan Always  •  ✓ Scan Once  •  ✕ Exclude  |  Click Faces/OCR icons to toggle  •  Ctrl/Shift: multi-select")
        self.legend_label.setStyleSheet("font-size: 11px;")
        set_ui_role(self.legend_label, "muted")
        left_layout.addWidget(self.legend_label)

        # Right Pane: Settings for Selected Folder
        right_frame = QFrame(splitter)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(10)

        self.lbl_selected_title = QLabel("Settings for Selected Folder")
        self.lbl_selected_title.setStyleSheet("font-size: 13px;")
        set_ui_role(self.lbl_selected_title, "title")
        right_layout.addWidget(self.lbl_selected_title)

        self.lbl_selected_path = QLabel("Select a folder on the left.")
        self.lbl_selected_path.setWordWrap(True)
        self.lbl_selected_path.setStyleSheet("font-size: 11px;")
        set_ui_role(self.lbl_selected_path, "muted")
        right_layout.addWidget(self.lbl_selected_path)

        # 1. Monitoring Policy Radio Group
        grp_monitor = QGroupBox("Monitoring Policy", right_frame)
        mon_layout = QVBoxLayout(grp_monitor)
        self.radio_always = QRadioButton("Scan Always (Monitor for new or changed files)", grp_monitor)
        self.radio_always.setIcon(_make_status_icon("always"))
        self.radio_once = QRadioButton(
            "Scan Once (Keep in catalog, skip live file watcher & startup rescans)", grp_monitor
        )
        self.radio_once.setIcon(_make_status_icon("once"))
        self.radio_excluded = QRadioButton(
            "Exclude from catalog (keeps files on disk)", grp_monitor
        )
        self.radio_excluded.setIcon(_make_status_icon("excluded"))
        self.monitor_group = QButtonGroup(grp_monitor)
        self.monitor_group.addButton(self.radio_always)
        self.monitor_group.addButton(self.radio_once)
        self.monitor_group.addButton(self.radio_excluded)
        mon_layout.addWidget(self.radio_always)
        mon_layout.addWidget(self.radio_once)
        mon_layout.addWidget(self.radio_excluded)
        right_layout.addWidget(grp_monitor)

        self.radio_always.toggled.connect(self._on_policy_control_changed)
        self.radio_once.toggled.connect(self._on_policy_control_changed)
        self.radio_excluded.toggled.connect(self._on_policy_control_changed)

        # 2. Background AI Indexing Checkboxes
        grp_ai = QGroupBox("Background AI Indexing", right_frame)
        ai_layout = QVBoxLayout(grp_ai)
        self.chk_scan_faces = QCheckBox("Scan this folder for Faces & People", grp_ai)
        self.chk_scan_faces.stateChanged.connect(self._on_policy_control_changed)
        ai_layout.addWidget(self.chk_scan_faces)

        self.btn_clear_faces = QPushButton("Clear Unconfirmed Face Suggestions in Folder", grp_ai)
        self.btn_clear_faces.clicked.connect(self._on_clear_unconfirmed_faces)
        ai_layout.addWidget(self.btn_clear_faces)

        self.chk_scan_ocr = QCheckBox("Scan this folder for Text in Photos (OCR)", grp_ai)
        self.chk_scan_ocr.stateChanged.connect(self._on_scan_ocr_toggled)
        ai_layout.addWidget(self.chk_scan_ocr)

        self.btn_scan_ocr_now = QPushButton("Scan Folder for Text Now", grp_ai)
        self.btn_scan_ocr_now.clicked.connect(self._on_scan_ocr_now)
        ai_layout.addWidget(self.btn_scan_ocr_now)

        ocr_cfg_row = QHBoxLayout()
        self.lbl_ocr_backend = QLabel("OCR Engine: detecting...", grp_ai)
        self.lbl_ocr_backend.setStyleSheet("font-size: 11px;")
        set_ui_role(self.lbl_ocr_backend, "muted")
        self.btn_configure_ocr = QPushButton("OCR Settings in Options...", grp_ai)
        self.btn_configure_ocr.setToolTip("Open Options → Add-ons to choose the OCR engine or set model paths")
        self.btn_configure_ocr.clicked.connect(self._on_configure_ocr_path)
        ocr_cfg_row.addWidget(self.lbl_ocr_backend, 1)
        ocr_cfg_row.addWidget(self.btn_configure_ocr)
        ai_layout.addLayout(ocr_cfg_row)
        QTimer.singleShot(0, self._refresh_ocr_backend_label)
        right_layout.addWidget(grp_ai)

        # 3. Private Folders Checkbox + Unlock / Auto-Lock
        grp_priv = QGroupBox("Private Folders & Access", right_frame)
        priv_layout = QVBoxLayout(grp_priv)
        self.chk_is_private = QCheckBox(
            "Private Folder (Hide photos, people, tags & places when Private Mode is locked)",
            grp_priv,
        )
        self.chk_is_private.stateChanged.connect(self._on_policy_control_changed)
        priv_layout.addWidget(self.chk_is_private)

        self.btn_private_lock_settings = QPushButton("Unlock & Auto-Lock Settings...", grp_priv)
        self.btn_private_lock_settings.clicked.connect(self._on_private_lock_settings)
        priv_layout.addWidget(self.btn_private_lock_settings)
        right_layout.addWidget(grp_priv)

        # Subfolder Cascade Button
        self.btn_apply_subfolders = QPushButton("Apply Settings to All Subfolders", right_frame)
        self.btn_apply_subfolders.clicked.connect(self._on_apply_to_subfolders)
        right_layout.addWidget(self.btn_apply_subfolders)

        right_layout.addStretch(1)

        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 6)

        # Bottom Close row
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_close = QPushButton("Close", self)
        self.btn_close.setDefault(True)
        self.btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_close)
        root_layout.addLayout(btn_row)

        self._set_controls_enabled(False)

    def _apply_theme_styles(self) -> None:
        if self._is_dark:
            muted = "#94A3B8"
            self.setStyleSheet(
                """
                QDialog {
                    background-color: #1B1E24;
                    color: #E2E8F0;
                }
                QLabel {
                    color: #E2E8F0;
                    background: transparent;
                }
                QGroupBox {
                    color: #60A5FA;
                    font-weight: 600;
                    border: 1px solid #333A45;
                    border-radius: 5px;
                    margin-top: 8px;
                    padding-top: 10px;
                    background-color: #20242C;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 4px;
                    color: #60A5FA;
                }
                QRadioButton, QCheckBox {
                    color: #E2E8F0;
                    background: transparent;
                    spacing: 6px;
                }
                QRadioButton:disabled, QCheckBox:disabled {
                    color: #64748B;
                }
                QPushButton {
                    background-color: #262B34;
                    color: #E2E8F0;
                    border: 1px solid #3B4252;
                    border-radius: 4px;
                    padding: 4px 10px;
                }
                QPushButton:hover {
                    background-color: #2E3440;
                    border: 1px solid #60A5FA;
                }
                QPushButton:disabled {
                    color: #64748B;
                    background-color: #1E222A;
                    border-color: #2D323B;
                }
                QTreeWidget {
                    background-color: #14171C;
                    alternate-background-color: #191D24;
                    color: #E2E8F0;
                    border: 1px solid #2D323B;
                }
                QTreeWidget::item:selected {
                    background-color: #2C415A;
                    color: #F8FAFC;
                }
                """
            )
            self.lbl_selected_path.setStyleSheet(f"color: {muted}; font-size: 11px;")
            self.lbl_ocr_backend.setStyleSheet(f"color: {muted}; font-size: 11px;")
            self.legend_label.setStyleSheet(f"color: {muted}; font-size: 11px;")
        else:
            self.lbl_selected_path.setStyleSheet("color: #555; font-size: 11px;")
            self.lbl_ocr_backend.setStyleSheet("color: #555; font-size: 11px;")
            self.legend_label.setStyleSheet("color: #666; font-size: 11px;")

    def _set_controls_enabled(self, enabled: bool) -> None:
        for w in (
            self.radio_always,
            self.radio_once,
            self.radio_excluded,
            self.chk_scan_faces,
            self.btn_clear_faces,
            self.chk_scan_ocr,
            self.btn_scan_ocr_now,
            self.chk_is_private,
            self.btn_apply_subfolders,
        ):
            w.setEnabled(enabled)

    def _apply_folder_to_item(self, item: QTreeWidgetItem, f: dict[str, Any]) -> None:
        """Update an existing QTreeWidgetItem's icons and tooltips from folder policy dict."""
        mode = str(f.get("monitor_mode") or "always").lower()
        faces_on = int(f.get("scan_faces") if f.get("scan_faces") is not None else 1) == 1
        ocr_on = int(f.get("scan_ocr") if f.get("scan_ocr") is not None else 0) == 1
        is_priv = int(f.get("is_private") or 0) == 1

        item.setIcon(0, _make_folder_row_icon(mode))
        item.setToolTip(0, format_folder_badges(f))

        item.setIcon(1, _make_feature_icon("faces", faces_on))
        item.setToolTip(1, f"Scan Faces: {'On' if faces_on else 'Off'} (click to toggle)")

        item.setIcon(2, _make_feature_icon("ocr", ocr_on))
        item.setToolTip(2, f"Scan Text (OCR): {'On' if ocr_on else 'Off'} (click to toggle)")

        if is_priv:
            item.setIcon(3, _make_feature_icon("private", True))
            item.setToolTip(3, "Private Folder (Hidden when locked)")
        else:
            item.setIcon(3, QIcon())
            item.setToolTip(3, "")

    def refresh_tree(
        self,
        select_folder_id: int | None = None,
        select_folder_ids: list[int] | None = None,
    ) -> None:
        vbar = self.tree_folders.verticalScrollBar()
        prev_scroll = vbar.value() if vbar is not None else 0

        folders = self.repo.get_all_folders_with_policies()
        self._folder_map = {int(f["id"]): f for f in folders}
        self._item_map.clear()

        self.tree_folders.setUpdatesEnabled(False)
        self.tree_folders.blockSignals(True)
        try:
            self.tree_folders.clear()

            for f in folders:
                fid = int(f["id"])
                name = str(f.get("name") or Path(str(f.get("path") or "")).name or f.get("path") or "")

                item = QTreeWidgetItem([name, "", "", ""])
                item.setData(0, Qt.UserRole, fid)
                if f.get("parent_id") is None:
                    root_font = item.font(0)
                    root_font.setBold(True)
                    item.setFont(0, root_font)
                self._apply_folder_to_item(item, f)
                self._item_map[fid] = item

            for f in folders:
                fid = int(f["id"])
                pid = f.get("parent_id")
                item = self._item_map[fid]
                if pid is not None and int(pid) in self._item_map:
                    self._item_map[int(pid)].addChild(item)
                else:
                    self.tree_folders.addTopLevelItem(item)

            self.tree_folders.expandAll()

            targets: list[int] = []
            if select_folder_ids:
                targets = [int(i) for i in select_folder_ids if int(i) in self._item_map]
            elif select_folder_id is not None and int(select_folder_id) in self._item_map:
                targets = [int(select_folder_id)]
            elif folders:
                targets = [int(folders[0]["id"])]

            if targets:
                for tid in targets:
                    self._item_map[tid].setSelected(True)
                self.tree_folders.setCurrentItem(self._item_map[targets[0]])
        finally:
            self.tree_folders.blockSignals(False)
            self.tree_folders.setUpdatesEnabled(True)

        if vbar is not None:
            vbar.setValue(prev_scroll)
            if prev_scroll > 0:
                QTimer.singleShot(0, lambda v=prev_scroll, bar=vbar: bar.setValue(v))

        if targets:
            self._load_folders_into_controls(targets)
        else:
            self._set_controls_enabled(False)
        if hasattr(self, "btn_remove_folder"):
            self.btn_remove_folder.setEnabled(bool(targets))
        if hasattr(self, "txt_filter_folders") and self.txt_filter_folders.text().strip():
            self._apply_folder_filter(self.txt_filter_folders.text())

    def _apply_folder_filter(self, query: str) -> None:
        q = query.strip().lower()

        def _filter_item(item: QTreeWidgetItem) -> bool:
            name_match = not q or q in item.text(0).lower()
            child_match = False
            for i in range(item.childCount()):
                if _filter_item(item.child(i)):
                    child_match = True
            visible = name_match or child_match
            item.setHidden(not visible)
            if q and child_match:
                item.setExpanded(True)
            return visible

        self.tree_folders.setUpdatesEnabled(False)
        try:
            for i in range(self.tree_folders.topLevelItemCount()):
                _filter_item(self.tree_folders.topLevelItem(i))
        finally:
            self.tree_folders.setUpdatesEnabled(True)

    def _on_add_folder_clicked(self) -> None:
        from PySide6.QtCore import QStandardPaths
        pic = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation)
        start_dir = pic if (pic and Path(pic).is_dir()) else str(Path.home() / "Pictures" if (Path.home() / "Pictures").is_dir() else Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Add Folder to Catalog", start_dir)
        if not folder:
            return
        if self.receivers(SIGNAL("folder_add_requested(QString)")) > 0:
            self.folder_add_requested.emit(folder)
        else:
            p = Path(folder)
            self.repo.upsert_folder(str(p), p.name or str(p), None, int(p.stat().st_mtime))
            self.policies_changed.emit()
        self.refresh_tree()

    def _on_remove_folder_clicked(self) -> None:
        fids = self.selected_folder_ids()
        if not fids:
            return
        # Skip descendants whose ancestor is also selected
        fid_set = set(fids)
        top_fids: list[int] = []
        for fid in fids:
            f = self._folder_map.get(fid)
            pid = f.get("parent_id") if f else None
            anc_selected = False
            while pid is not None:
                if int(pid) in fid_set:
                    anc_selected = True
                    break
                pf = self._folder_map.get(int(pid))
                pid = pf.get("parent_id") if pf else None
            if not anc_selected:
                top_fids.append(fid)

        label = (
            f"'{self._folder_map[top_fids[0]].get('name')}'"
            if len(top_fids) == 1 and top_fids[0] in self._folder_map
            else f"{len(top_fids)} selected folders"
        )
        ans = QMessageBox.question(
            self,
            "Remove Folder from Catalog",
            f"Remove {label} (and any subfolders) from the catalog?\n\n"
            "Your original files on disk will NOT be deleted.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return
        removed_ids: list[int] = []
        for fid in top_fids:
            removed_ids.extend(self.repo.remove_folder_tree(fid))
        if removed_ids:
            self.folder_removed.emit(removed_ids)
        self.policies_changed.emit()
        self.refresh_tree()

    def selected_folder_ids(self) -> list[int]:
        ids: list[int] = []
        for item in self.tree_folders.selectedItems():
            val = item.data(0, Qt.UserRole)
            if val is not None:
                fid = int(val)
                if fid not in ids:
                    ids.append(fid)
        if not ids:
            cur = self.tree_folders.currentItem()
            if cur is not None:
                val = cur.data(0, Qt.UserRole)
                if val is not None:
                    ids.append(int(val))
        return ids

    def selected_folder_id(self) -> int | None:
        ids = self.selected_folder_ids()
        return ids[0] if ids else None

    def _on_tree_item_selection_changed(self) -> None:
        ids = self.selected_folder_ids()
        if not ids:
            self._set_controls_enabled(False)
            return
        self._load_folders_into_controls(ids)

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        if item is None or column not in (1, 2):
            return
        fid_val = item.data(0, Qt.UserRole)
        if fid_val is None:
            return
        clicked_fid = int(fid_val)
        sel_ids = self.selected_folder_ids()
        target_ids = sel_ids if (len(sel_ids) > 1 and clicked_fid in sel_ids) else [clicked_fid]

        f0 = self._folder_map.get(clicked_fid)
        if not f0:
            return
        if column == 1:
            cur_on = int(f0.get("scan_faces") if f0.get("scan_faces") is not None else 1) == 1
            new_val = not cur_on
            if hasattr(self.repo, "update_folders_policy_batch"):
                self.repo.update_folders_policy_batch(target_ids, scan_faces=new_val)
            else:
                for tid in target_ids:
                    self.repo.update_folder_policy(tid, scan_faces=new_val)
            for tid in target_ids:
                if tid in self._folder_map:
                    self._folder_map[tid]["scan_faces"] = 1 if new_val else 0
                    if tid in self._item_map:
                        self._apply_folder_to_item(self._item_map[tid], self._folder_map[tid])
        elif column == 2:
            cur_on = int(f0.get("scan_ocr") if f0.get("scan_ocr") is not None else 0) == 1
            new_val = not cur_on
            if hasattr(self.repo, "update_folders_policy_batch"):
                self.repo.update_folders_policy_batch(target_ids, scan_ocr=new_val)
            else:
                for tid in target_ids:
                    self.repo.update_folder_policy(tid, scan_ocr=new_val)
            for tid in target_ids:
                if tid in self._folder_map:
                    self._folder_map[tid]["scan_ocr"] = 1 if new_val else 0
                    if tid in self._item_map:
                        self._apply_folder_to_item(self._item_map[tid], self._folder_map[tid])

        self._load_folders_into_controls(sel_ids or target_ids)
        self.policies_changed.emit()

    def _load_folder_into_controls(self, folder_id: int) -> None:
        self._load_folders_into_controls([folder_id])

    def _load_folders_into_controls(self, folder_ids: list[int]) -> None:
        rows = [self._folder_map[fid] for fid in folder_ids if fid in self._folder_map]
        if not rows:
            self._set_controls_enabled(False)
            return

        self._loading_ui = True
        try:
            self._set_controls_enabled(True)
            if len(rows) == 1:
                f = rows[0]
                self.lbl_selected_title.setText("Settings for Selected Folder")
                self.lbl_selected_path.setText(
                    f"{f['path']}  ({int(f.get('media_count') or 0):,} cataloged items)"
                )
            else:
                total_items = sum(int(r.get("media_count") or 0) for r in rows)
                self.lbl_selected_title.setText(f"Settings for {len(rows)} Selected Folders")
                self.lbl_selected_path.setText(
                    f"{len(rows)} folders selected  ({total_items:,} cataloged items total)"
                )

            modes = {str(r.get("monitor_mode") or "always").lower() for r in rows}
            self.monitor_group.setExclusive(False)
            if len(modes) == 1:
                m = next(iter(modes))
                self.radio_always.setChecked(m == "always")
                self.radio_once.setChecked(m == "once")
                self.radio_excluded.setChecked(m == "excluded")
            else:
                self.radio_always.setChecked(False)
                self.radio_once.setChecked(False)
                self.radio_excluded.setChecked(False)
            self.monitor_group.setExclusive(True)

            def _apply_check(chk: QCheckBox, key: str, default: int) -> None:
                vals = {int(r.get(key) if r.get(key) is not None else default) == 1 for r in rows}
                if len(vals) == 1:
                    chk.setTristate(False)
                    chk.setChecked(next(iter(vals)))
                else:
                    chk.setTristate(True)
                    chk.setCheckState(Qt.PartiallyChecked)

            _apply_check(self.chk_scan_faces, "scan_faces", 1)
            _apply_check(self.chk_scan_ocr, "scan_ocr", 1)
            _apply_check(self.chk_is_private, "is_private", 0)
        finally:
            self._loading_ui = False

    def _current_mode_value(self) -> str | None:
        if self.radio_always.isChecked():
            return "always"
        if self.radio_once.isChecked():
            return "once"
        if self.radio_excluded.isChecked():
            return "excluded"
        return None

    def _on_policy_control_changed(self) -> None:
        if self._loading_ui:
            return
        fids = self.selected_folder_ids()
        if not fids:
            return

        new_mode = self._current_mode_value()
        if new_mode == "excluded":
            need_confirm = any(
                str(self._folder_map.get(fid, {}).get("monitor_mode") or "always").lower() != "excluded"
                for fid in fids
            )
            if need_confirm:
                msg = (
                    f"Remove {len(fids)} selected folder(s) and their subfolders from the Photo Gallery catalog?\n\n"
                    if len(fids) > 1
                    else "Remove this folder and its subfolders from the Photo Gallery catalog?\n\n"
                )
                reply = QMessageBox.question(
                    self,
                    "Exclude Folder from Catalog",
                    msg + "Your original files on disk will NOT be deleted, and parent folder scans will skip them.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    self._load_folders_into_controls(fids)
                    return
                for fid in fids:
                    self.repo.exclude_folder_from_catalog(fid)

        faces_arg = None if self.chk_scan_faces.checkState() == Qt.PartiallyChecked else self.chk_scan_faces.isChecked()
        ocr_arg = None if self.chk_scan_ocr.checkState() == Qt.PartiallyChecked else self.chk_scan_ocr.isChecked()
        priv_arg = None if self.chk_is_private.checkState() == Qt.PartiallyChecked else self.chk_is_private.isChecked()

        if hasattr(self.repo, "update_folders_policy_batch"):
            self.repo.update_folders_policy_batch(
                fids,
                monitor_mode=new_mode,
                scan_faces=faces_arg,
                scan_ocr=ocr_arg,
                is_private=priv_arg,
            )
        else:
            for fid in fids:
                self.repo.update_folder_policy(
                    fid,
                    monitor_mode=new_mode,
                    scan_faces=faces_arg,
                    scan_ocr=ocr_arg,
                    is_private=priv_arg,
                )

        if new_mode == "excluded":
            self.refresh_tree(select_folder_ids=fids)
        else:
            for fid in fids:
                f_dict = self._folder_map.get(fid)
                if f_dict is not None:
                    if new_mode is not None:
                        f_dict["monitor_mode"] = new_mode
                    if faces_arg is not None:
                        f_dict["scan_faces"] = 1 if faces_arg else 0
                    if ocr_arg is not None:
                        f_dict["scan_ocr"] = 1 if ocr_arg else 0
                    if priv_arg is not None:
                        f_dict["is_private"] = 1 if priv_arg else 0
                    item = self._item_map.get(fid)
                    if item is not None:
                        self._apply_folder_to_item(item, f_dict)
        self.policies_changed.emit()

    def _on_apply_to_subfolders(self) -> None:
        fids = self.selected_folder_ids()
        if not fids:
            return
        updated = 0
        is_excluded = self._current_mode_value() == "excluded"
        for fid in fids:
            updated += self.repo.apply_folder_policy_to_subfolders(fid)
            if is_excluded:
                self.repo.exclude_folder_from_catalog(fid)

        if is_excluded:
            self.refresh_tree(select_folder_ids=fids)
        else:
            # Update descendant items in place so tree scroll position does not jump
            children_by_parent: dict[int, list[int]] = {}
            for cid, f_row in self._folder_map.items():
                pid = f_row.get("parent_id")
                if pid is not None:
                    children_by_parent.setdefault(int(pid), []).append(int(cid))
            for fid in fids:
                src = self._folder_map.get(fid)
                if src is None:
                    continue
                queue = list(children_by_parent.get(fid, []))
                while queue:
                    cid = queue.pop(0)
                    queue.extend(children_by_parent.get(cid, []))
                    dst = self._folder_map.get(cid)
                    if dst is not None:
                        dst["monitor_mode"] = src.get("monitor_mode")
                        dst["scan_faces"] = src.get("scan_faces")
                        dst["scan_ocr"] = src.get("scan_ocr")
                        dst["is_private"] = src.get("is_private")
                        item = self._item_map.get(cid)
                        if item is not None:
                            self._apply_folder_to_item(item, dst)
        self.policies_changed.emit()
        QMessageBox.information(
            self,
            "Applied to Subfolders",
            f"Updated settings on {updated} subfolder(s).",
        )

    def _on_clear_unconfirmed_faces(self) -> None:
        fids = self.selected_folder_ids()
        if not fids:
            return
        cleared = sum(self.repo.clear_unconfirmed_faces_in_folder(fid) for fid in fids)
        self.policies_changed.emit()
        QMessageBox.information(
            self,
            "Cleared Unconfirmed Faces",
            f"Removed {cleared} unconfirmed face suggestion(s) in selected folder(s).",
        )

    def _ensure_ocr_ready_prompt(self) -> bool:
        from src.core.ocr_engine import has_configured_ocr_engine
        if has_configured_ocr_engine():
            return True
        from src.ui.dialogs import prompt_and_download_rapidocr_addon
        ok = prompt_and_download_rapidocr_addon(self, on_more_options=self._on_configure_ocr_path)
        self._refresh_ocr_backend_label()
        return ok

    def _on_scan_ocr_toggled(self, state: int) -> None:
        if self._loading_ui:
            return
        if self.chk_scan_ocr.isChecked() and not self._ensure_ocr_ready_prompt():
            self.chk_scan_ocr.blockSignals(True)
            self.chk_scan_ocr.setChecked(False)
            self.chk_scan_ocr.blockSignals(False)
            return
        self._on_policy_control_changed()

    def _on_scan_ocr_now(self) -> None:
        fids = self.selected_folder_ids()
        if not fids:
            return
        if not self._ensure_ocr_ready_prompt():
            return
        if not self.chk_scan_ocr.isChecked():
            self.chk_scan_ocr.blockSignals(True)
            self.chk_scan_ocr.setChecked(True)
            self.chk_scan_ocr.blockSignals(False)
            self._on_policy_control_changed()
        for fid in fids:
            self.scan_ocr_requested.emit(fid)

    def _refresh_ocr_backend_label(self) -> None:
        from src.core.ocr_engine import get_ocr_engine_status

        st = get_ocr_engine_status()
        labels = {
            "rapidocr": "RapidOCR (ONNX)",
            "winocr": "Windows Native OCR (WinRT)",
            "windows_media_ocr": "Windows.Media.Ocr",
            "pytesseract": "Tesseract OCR",
            "stub": "Not configured (MSER stub)",
        }
        b_name = st["backend"]
        desc = labels.get(b_name, b_name)
        if b_name == "pytesseract" and st.get("tesseract_detected"):
            desc = f"Tesseract ({Path(st['tesseract_detected']).name})"
        self.lbl_ocr_backend.setText(f"OCR Engine: {desc}")

    def _on_configure_ocr_path(self) -> None:
        from PySide6.QtCore import SIGNAL
        if hasattr(self.parent(), "_open_settings_dialog") or self.receivers(SIGNAL("open_ocr_options_requested()")) > 0:
            self.open_ocr_options_requested.emit()
            return
        from src.core.ocr_engine import (
            OcrEngine,
            detect_tesseract_executable,
            get_ocr_engine_status,
            resolve_tesseract_executable,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("Configure Offline OCR Engine")
        dlg.setMinimumWidth(460)
        v = QVBoxLayout(dlg)

        st = get_ocr_engine_status()
        info = QLabel(
            f"Active Engine: <b>{st['backend']}</b><br>"
            f"Detected Tesseract: <code>{st['tesseract_detected'] or 'Not found'}</code><br>"
            f"Custom RapidOCR Folder: <code>{st['custom_model_dir'] or 'Auto / None'}</code>",
            dlg,
        )
        info.setWordWrap(True)
        v.addWidget(info)

        hint = QLabel(
            "For Tesseract OCR, select <b>tesseract.exe</b> (usually inside "
            "<code>C:\\Program Files\\Tesseract-OCR\\</code>). For RapidOCR, select the folder "
            "containing <code>*det*.onnx</code> and <code>*rec*.onnx</code>.",
            dlg,
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555; font-size: 11px;")
        v.addWidget(hint)

        btn_tess = QPushButton("Select tesseract.exe...", dlg)
        btn_rapid = QPushButton("Select RapidOCR Models Folder...", dlg)
        btn_reset = QPushButton("Auto-Detect / Reset to Default", dlg)
        btn_close = QPushButton("Close", dlg)

        def _pick_tess() -> None:
            start_dir = r"C:\Program Files\Tesseract-OCR" if Path(r"C:\Program Files\Tesseract-OCR").is_dir() else ""
            exe, _ = QFileDialog.getOpenFileName(
                dlg,
                "Select tesseract.exe (e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe)",
                start_dir,
                "Tesseract Executable (tesseract.exe);;Executables (*.exe);;All Files (*)",
            )
            if exe:
                resolved = resolve_tesseract_executable(exe) or exe
                OcrEngine.save_ocr_paths(tesseract_path=resolved)
                self._refresh_ocr_backend_label()
                dlg.accept()

        def _pick_rapid() -> None:
            folder = QFileDialog.getExistingDirectory(dlg, "Select RapidOCR ONNX Models Folder")
            if folder:
                OcrEngine.save_ocr_paths(custom_model_dir=folder)
                self._refresh_ocr_backend_label()
                dlg.accept()

        def _reset_auto() -> None:
            OcrEngine.save_ocr_paths(custom_model_dir="", tesseract_path="")
            detected = detect_tesseract_executable()
            self._refresh_ocr_backend_label()
            QMessageBox.information(
                dlg,
                "OCR Auto-Detect",
                f"Reset to auto-detect.\nDetected Tesseract: {detected or 'None'}\n"
                f"Active Engine: {OcrEngine().backend_name}",
            )
            dlg.accept()

        btn_tess.clicked.connect(_pick_tess)
        btn_rapid.clicked.connect(_pick_rapid)
        btn_reset.clicked.connect(_reset_auto)
        btn_close.clicked.connect(dlg.reject)

        v.addWidget(btn_tess)
        v.addWidget(btn_rapid)
        v.addWidget(btn_reset)
        v.addWidget(btn_close)
        dlg.exec()

    def _on_private_lock_settings(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Private Folders Unlock & Auto-Lock")
        form = QFormLayout(dlg)

        cmb_auth = QComboBox(dlg)
        cmb_auth.addItem("1-Click Unlock (No Passcode)", "none")
        cmb_auth.addItem("Passcode / PIN", "pin")
        cmb_auth.addItem("Windows Login / Windows Hello", "windows")
        cur_auth = get_private_auth_mode()
        auth_idx = cmb_auth.findData(cur_auth)
        cmb_auth.setCurrentIndex(max(0, auth_idx))
        form.addRow("Unlock Method:", cmb_auth)

        txt_pin = QLineEdit(dlg)
        txt_pin.setEchoMode(QLineEdit.Password)
        txt_pin.setPlaceholderText("Leave blank to keep current" if has_private_pin() else "Enter new PIN / Passcode")
        form.addRow("Passcode / PIN:", txt_pin)

        def _sync_pin_fields() -> None:
            is_pin = cmb_auth.currentData() == "pin"
            txt_pin.setEnabled(is_pin)

        cmb_auth.currentIndexChanged.connect(_sync_pin_fields)
        _sync_pin_fields()

        cmb_timeout = QComboBox(dlg)
        for label, mins in [("Off (Manual lock only)", 0), ("5 minutes", 5), ("15 minutes", 15), ("30 minutes", 30)]:
            cmb_timeout.addItem(label, mins)
        cur_mins = get_private_auto_lock_minutes()
        idx = cmb_timeout.findData(cur_mins)
        cmb_timeout.setCurrentIndex(max(0, idx))
        form.addRow("Auto-Lock Idle Timeout:", cmb_timeout)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec() == QDialog.Accepted:
            chosen_mode = str(cmb_auth.currentData() or "none")
            pin_txt = txt_pin.text().strip()
            if chosen_mode == "pin" and not pin_txt and not has_private_pin():
                QMessageBox.warning(self, "Passcode Required", "Please enter a Passcode / PIN or choose another unlock method.")
                return
            set_private_lock_settings(
                pin=pin_txt or None,
                clear_pin=(chosen_mode == "none"),
                auto_lock_minutes=int(cmb_timeout.currentData() or 0),
                auth_mode=chosen_mode,
            )
            self.policies_changed.emit()
