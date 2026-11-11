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
if (project_root / "models").exists():
    datas.append((str(project_root / "models"), "models"))

# 2. Collect library data files & binaries
datas += collect_data_files("cv2")
datas += collect_data_files("onnxruntime")
datas += collect_data_files("insightface")

# InsightFace's pickle_object.get_object() checks sys._MEIPASS/objects/ in frozen builds
# instead of insightface/data/objects/. Ship objects/ at the bundle root too.
try:
    import insightface as _insightface_pkg
    _insight_objects_dir = Path(_insightface_pkg.__file__).resolve().parent / "data" / "objects"
    if _insight_objects_dir.is_dir():
        datas.append((str(_insight_objects_dir), "objects"))
except Exception:
    pass

for pkg in ("pillow_heif", "pillow_jxl", "pillow_avif"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

binaries = []
binaries += collect_dynamic_libs("onnxruntime")

for pkg in ("pillow_heif", "pillow_jxl", "pillow_avif"):
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:
        pass

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
    "PIL.ImageCms",
    "PIL.ImageOps",
    "PIL.ExifTags",
    "rawpy",
    "onnxruntime",

    # Next-Gen Image Codecs (HEIF, AVIF, JPEG XL)
    "pillow_heif",
    "pillow_jxl",
    "pillow_jxl.JpegXLImagePlugin",
    "pillow_avif",
    "pillow_avif.AvifImagePlugin",
    
    # InsightFace & ML Modules
    "insightface",
    "insightface.app",
    "insightface.model_zoo",
    "insightface.utils",

    # System & Icon Utilities
    "send2trash",
    "qtawesome",
]

# Collect additional submodules safely
hiddenimports += collect_submodules("insightface.model_zoo")
hiddenimports += collect_submodules("onnxruntime")

# 4. Exclusions (developer tooling, unused large frameworks; scipy/skimage replaced by pure NumPy/OpenCV)
# Note: Do NOT exclude 'unittest' because numpy.testing / insightface import it at runtime.
excludes = [
    "pytest",
    "_pytest",
    "tkinter",
    "IPython",
    "jupyter",
    "matplotlib",
    "pandas",
    "scipy",
    "skimage",
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
