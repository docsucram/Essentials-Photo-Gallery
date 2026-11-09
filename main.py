"""
PhotoGallery64 — Modern 64-Bit Python / PySide6 Clone of Windows Live Photo Gallery.
Application entrypoint initializing High-DPI Qt rendering, DualDatabaseEngine
(db/catalog.db + db/thumbnails.db), and PhotoGalleryMainWindow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QLoggingCategory, QObject, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QToolTip, QWidget

from src.db.connection import DualDatabaseEngine
from src.imaging.image_loader import register_nextgen_codecs
from src.ui.main_window import PhotoGalleryMainWindow
from src.utils.paths import get_default_db_dir, get_resource_path
from src.utils.scan_logger import (
    install_global_crash_logger,
    install_universal_tooltip_filter,
)


def create_app(argv: list[str] | None = None) -> QApplication:
    existing = QApplication.instance()
    if existing is not None and isinstance(existing, QApplication):
        return existing
    QLoggingCategory.setFilterRules(
        "qt.gui.imageio.jpeg.warning=false\n"
        "qt.gui.imageio.warning=false\n"
        "qt.imageformats.tiff=false\n"
        "qt.imageformats.*.warning=false\n"
    )
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    register_nextgen_codecs()
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Essentials Photo Gallery")
    app.setOrganizationName("Essentials Photo Gallery")
    app.setApplicationVersion("0.10.0")
    ico_path = get_resource_path("resources/icons/essentials_photo_gallery.ico")
    if not ico_path.exists():
        ico_path = get_resource_path("resources/icons/photogallery_live.ico")
    if ico_path.exists():
        from PySide6.QtGui import QIcon
        app.setWindowIcon(QIcon(str(ico_path)))
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#1E395B"))
    app.setPalette(palette)
    QToolTip.setPalette(palette)
    app.setStyleSheet(
        "QToolTip, QLabel#qtooltip_label { "
        "background-color: #FFFFFF; "
        "color: #1E395B; "
        "border: 1px solid #7DA2CE; "
        "border-radius: 2px; "
        "padding: 3px 6px; "
        "font-family: 'Segoe UI', sans-serif; "
        "font-size: 9pt; "
        "}"
    )
    install_universal_tooltip_filter(app)
    return app


def main(argv: list[str] | None = None) -> int:
    install_global_crash_logger()

    parser = argparse.ArgumentParser(
        description="Essentials Photo Gallery — Windows Live Photo Gallery 64-Bit Clone"
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        type=Path,
        default=None,
        help="Optional initial image or video file path (e.g. from Explorer 'Open with...')",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=None,
        help="Directory for catalog.db and thumbnails.db (defaults to %LOCALAPPDATA%/PhotoGallery/db or local db/)",
    )
    parser.add_argument(
        "--scan-folder",
        type=Path,
        default=None,
        help="Optional initial folder path to scan on startup",
    )
    parser.add_argument(
        "--theme",
        choices=["wlpg_aero", "modern", "dark"],
        default=None,
        help="Initial visual stylesheet theme",
    )
    args = parser.parse_args(argv)

    app = create_app(argv)
    db_dir = args.db_dir if args.db_dir is not None else get_default_db_dir()
    db_dir.mkdir(parents=True, exist_ok=True)

    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )

    window = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=True)
    if args.theme:
        window.apply_theme(args.theme)
    window.show()

    # If a file was passed via Explorer 'Open with...' or CLI positional argument
    if args.file_path is not None and args.file_path.exists():
        if args.file_path.is_file():
            window.open_file_on_startup(args.file_path)
        elif args.file_path.is_dir():
            window.scan_folder(args.file_path, include_subfolders=True)
    elif args.scan_folder is not None and args.scan_folder.exists():
        window.scan_folder(args.scan_folder, include_subfolders=True)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
