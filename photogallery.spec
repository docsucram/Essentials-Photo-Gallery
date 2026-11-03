# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification for Essentials Photo Gallery (PhotoGallery64).
Configures an optimized --onedir standalone Windows x64 build.
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

project_root = Path.cwd().resolve()

# 1. Base Project Data Files
datas = [
    (str(project_root / "assets"), "assets"),
    (str(project_root / "resources"), "resources"),
    (str(project_root / "db" / "schema.sql"), "db"),
]

# 2. Collect library data files & binaries
datas += collect_data_files("cv2")
datas += collect_data_files("onnxruntime")
datas += collect_data_files("insightface")

binaries = []
binaries += collect_dynamic_libs("onnxruntime")

# 3. Hidden Imports
hiddenimports = [
    # PySide6 Qt Modules
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtXml",
    
    # Core Data & Image Libraries
    "sqlite3",
    "cv2",
    "numpy",
    "PIL",
    "PIL.Image",
    "PIL.ImageOps",
    "PIL.ExifTags",
    "scipy",
    "scipy.ndimage",
    "scipy.spatial",
    "onnxruntime",
    
    # InsightFace & ML Modules
    "insightface",
    "insightface.app",
    "insightface.model_zoo",
    "insightface.utils",
    "skimage",
    "skimage.transform",

    # System & Icon Utilities
    "send2trash",
    "qtawesome",
]

# Collect additional submodules safely
hiddenimports += collect_submodules("insightface.model_zoo")

# 4. Exclusions (developer tooling, unused large frameworks)
# Note: Do NOT exclude 'unittest' because numpy.testing / scipy / insightface import it at runtime.
excludes = [
    "pytest",
    "_pytest",
    "tkinter",
    "IPython",
    "jupyter",
    "matplotlib",
    "pandas",
]

a = Analysis(
    ["main.py"],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EssentialsPhotoGallery",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Set False for faster builds and rock-solid antivirus compatibility
    console=False,  # Windowed GUI app (no black console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / "resources" / "icons" / "essentials_photo_gallery.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="EssentialsPhotoGallery",
)
