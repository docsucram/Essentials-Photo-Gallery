"""
Path and resource resolution helpers for development and frozen PyInstaller bundles.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    """Returns True if the application is running inside a PyInstaller or cx_Freeze bundle."""
    return bool(getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"))


_CACHED_BUNDLE_DIR: Path | None = None


def get_bundle_dir() -> Path:
    """Returns the root directory where bundled assets live."""
    global _CACHED_BUNDLE_DIR
    if _CACHED_BUNDLE_DIR is not None:
        return _CACHED_BUNDLE_DIR
    if is_frozen():
        _CACHED_BUNDLE_DIR = Path(getattr(sys, "_MEIPASS"))
    else:
        # In dev mode, project root is 2 levels up from src/utils/
        _CACHED_BUNDLE_DIR = Path(__file__).resolve().parents[2]
    return _CACHED_BUNDLE_DIR


def get_resource_path(relative_path: str | Path) -> Path:
    """
    Returns an absolute path to a resource file, resolving correctly in both
    standard Python development environments and frozen PyInstaller packages.
    """
    return get_bundle_dir() / relative_path


def get_app_data_dir() -> Path:
    """
    Returns the persistent user data directory in %LOCALAPPDATA%/PhotoGallery
    (or ~/.local/share/PhotoGallery on Linux). Creates the directory if missing.
    """
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            p = Path(base) / "PhotoGallery"
        else:
            p = Path.home() / "AppData" / "Local" / "PhotoGallery"
    else:
        p = Path.home() / ".local" / "share" / "PhotoGallery"

    p.mkdir(parents=True, exist_ok=True)
    return p


def get_custom_db_dir() -> Path | None:
    """Returns the user-configured custom database directory from QSettings, or None if using default."""
    try:
        from PySide6.QtCore import QSettings

        raw = str(QSettings("PhotoGallery64", "PhotoGallery64").value("storage/custom_db_dir", "") or "").strip()
        if raw:
            return Path(raw)
    except Exception:
        pass
    return None


def set_custom_db_dir(path: Path | str | None) -> None:
    """Persists or clears the custom database directory in QSettings."""
    from PySide6.QtCore import QSettings

    settings = QSettings("PhotoGallery64", "PhotoGallery64")
    if path is None or not str(path).strip():
        settings.remove("storage/custom_db_dir")
    else:
        settings.setValue("storage/custom_db_dir", str(Path(path)))
    settings.sync()


def is_cloud_or_network_path(path: Path | str) -> bool:
    """
    Returns True if ``path`` points to a UNC network share (\\\\server\\...) or sits inside
    a cloud-synced folder (OneDrive, Dropbox, Google Drive, iCloudDrive) where SQLite WAL locking is unsafe.
    """
    s = str(path).strip()
    if s.startswith("\\\\") or s.startswith("//"):
        return True
    lowered_parts = [part.lower() for part in Path(s).parts]
    cloud_markers = ("onedrive", "dropbox", "google drive", "googledrive", "iclouddrive", "icloud drive")
    return any(any(marker in part for marker in cloud_markers) for part in lowered_parts)


def get_default_db_dir(*, ignore_custom: bool = False) -> Path:
    """
    Returns the directory where catalog.db and thumbnails.db should be stored.
    Respects ``storage/custom_db_dir`` when configured (outside pytest unless ``ignore_custom=True``).
    In frozen/production mode, defaults to %LOCALAPPDATA%/PhotoGallery/db to avoid
    Windows 'Program Files' write-permission errors. In dev mode, prefers local 'db/'
    if writable, falling back to user AppData.
    """
    if not ignore_custom and "PYTEST_CURRENT_TEST" not in os.environ:
        custom = get_custom_db_dir()
        if custom is not None:
            try:
                if custom.exists() or (custom.parent.exists() and not is_volume_offline(custom)):
                    custom.mkdir(parents=True, exist_ok=True)
                    return custom
            except OSError:
                return custom

    if is_frozen():
        db_dir = get_app_data_dir() / "db"
    else:
        local_db = get_bundle_dir() / "db"
        try:
            local_db.mkdir(parents=True, exist_ok=True)
            test_file = local_db / ".write_test"
            test_file.touch()
            test_file.unlink()
            return local_db
        except (PermissionError, OSError):
            db_dir = get_app_data_dir() / "db"

    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


def get_schema_path() -> Path:
    """Returns the absolute path to db/schema.sql."""
    return get_resource_path("db/schema.sql")


def get_icons_dir() -> Path:
    """Returns the absolute path to assets/icons/."""
    return get_resource_path("assets/icons")


def get_themes_dir() -> Path:
    """Returns the absolute path to assets/themes/."""
    return get_resource_path("assets/themes")


# Windows cloud placeholder / OneDrive Files On-Demand attribute bitmasks.
# Reading these flags via os.stat() never triggers cloud file hydration.
FILE_ATTRIBUTE_OFFLINE = 0x1000
FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x40000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x400000
_CLOUD_ONLY_ATTR_MASK = (
    FILE_ATTRIBUTE_OFFLINE
    | FILE_ATTRIBUTE_RECALL_ON_OPEN
    | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
)


def is_cloud_only_stat(st: os.stat_result | None) -> bool:
    """
    Returns True if stat attributes indicate an online-only cloud placeholder
    (e.g., Windows OneDrive Files On-Demand) whose content is not resident locally.
    """
    if st is None:
        return False
    attrs = getattr(st, "st_file_attributes", 0) or 0
    return bool(int(attrs) & _CLOUD_ONLY_ATTR_MASK)


def is_cloud_only_file(path: Path | str) -> bool:
    """
    Checks live filesystem attributes without opening the file or triggering
    an OS cloud-placeholder download (OneDrive Files On-Demand).
    """
    try:
        st = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return is_cloud_only_stat(st)


def is_volume_offline(path: Path | str) -> bool:
    """
    Returns True if the drive/UNC share hosting `path` is currently unplugged or unreachable
    (distinct from a folder/file that was genuinely deleted on a mounted drive).
    """
    p = Path(path)
    anchor = p.anchor
    if anchor:
        try:
            return not os.path.exists(anchor)
        except OSError:
            return True
    return False
