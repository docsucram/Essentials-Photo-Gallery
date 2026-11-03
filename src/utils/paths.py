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


def get_default_db_dir() -> Path:
    """
    Returns the directory where catalog.db and thumbnails.db should be stored.
    In frozen/production mode, always uses %LOCALAPPDATA%/PhotoGallery/db to avoid
    Windows 'Program Files' write-permission errors. In dev mode, prefers local 'db/'
    if writable, falling back to user AppData.
    """
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
