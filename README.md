# Essentials Photo Gallery

[![Release](https://img.shields.io/github/v/release/docsucram/Essentials-Photo-Gallery?label=Release&color=blue)](https://github.com/docsucram/Essentials-Photo-Gallery/releases/latest)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%20%7C%2011%20(x64)-0078D6.svg)](https://github.com/docsucram/Essentials-Photo-Gallery/releases/latest)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/GUI-PySide6%20%2F%20Qt6-green.svg)](https://wiki.qt.io/Qt_for_Python)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

**Essentials Photo Gallery** is a high-performance, 64-bit Windows desktop photo management and editing workstation inspired by the classic **Windows Live Photo Gallery (WLPG)** and elevated with modern computer vision, biometric face clustering, interactive world map geotagging, a multi-photo creation studio, and 32-bit non-destructive image processing.

Engineered to fluidly manage libraries exceeding **500,000+ photos and videos** with sub-10ms queries and 60 FPS virtualized scrolling, 100% locally and offline.

---

## 📥 Download Windows Installer (x64)

Ready-to-run standalone installers (`EssentialsPhotoGallery_Setup_x64.exe`) are available on the **[GitHub Releases](https://github.com/docsucram/Essentials-Photo-Gallery/releases/latest)** page—no Python installation required.

> **Note on Windows SmartScreen:** Because this open-source installer is not signed with a commercial EV certificate, Windows SmartScreen may display an *"Unknown publisher"* prompt on first launch. Click **More info** → **Run anyway** to proceed with setup.

---

## Key Features

### 🎨 Authentic WLPG Ribbon Experience & Full Dark Theme
- **Classic Fluent & Aero Styling:** Faithful recreation of the Windows Live Photo Gallery ribbon tabs (*Home*, *Edit*, *Find*, *Create*, *View*) with responsive compaction and Quick Access Toolbar (QAT).
- **Effects Gallery:** Live preview carousel and scrollable dropdown (*Black and white*, *Sepia*, *Warm*, *Cool*, *Chrome*, *Vivid*, *Matte*, and *B&W filter effects*) with 0–100% intensity blending.
- **Full Dark Theme Support:** Dynamic dark slate styling across all ribbon tabs, carousels, status bars, and the single-image viewport canvas (`#14171C`).
- **Browser-Style Navigation History:** Seamless back and forward navigation stack with mouse thumb button (`BackButton`, `ForwardButton`) and keyboard shortcut (`Alt+Left` / `Alt+Right`) support.

### 👤 Biometric Face Recognition & People Workflows
- **InsightFace & ArcFace AI Backend:** SCRFD anatomical face detector paired with 512-dimensional ArcFace vector embeddings running on ONNX Runtime (CPU or DirectML GPU acceleration).
- **Batch People Tagging:** Virtualized face confirmation grid rendering 96×96 WebP crops with 1-click group confirmations and instant full-photo preview popovers on `Space`.
- **Single-Image View Face Overlay:** Bounding boxes, hover regions, suggestion accept/reject chips, automatic face-box remapping on crop/rotate, and direct **click-to-rename/retag** affordance on confirmed face badges.
- **People Directory & Unnamed Clusters:** Visual portrait directory grid with live search filtering, plus recurring unnamed face clusters (`>= 3` faces) with inline *"Add a name..."* pills and merge confirmation dialogs.

### 🛠 Modern 60 FPS Non-Destructive Editing Engine
- **Non-Destructive Pipeline & True Uncrop:** Edits are stored as parametric recipes and applied non-destructively while preserving pristine originals in `.OriginalFiles/` (with configurable Save Overwrite vs. Save as Copy workflows). Re-opening Crop reveals the full uncropped canvas with the existing crop box pre-positioned.
- **Fine-Tune & Depth & Light Adjustments:**
  - Interactive RGB/Luminance histogram with draggable black and white point clipping markers.
  - Clarity, vibrancy, color temperature, exposure, shadows/highlights, dehaze, unsharp mask, and bilateral denoising.
  - Depth-aware subject relighting and Snapseed-style U-Point selective control points.
- **Spot Fix, Red-Eye & Interactive Markup Suite:**
  - Multi-radius Poisson Spot Fix healing brush and circular click/drag pupil-masked Red-Eye removal.
  - **Vector Markup & Privacy Redaction:** Subpixel arrows, boxes, circles, stars, inline text callouts, two-click magnification loupes with tangent callout lines (crop-aware and scroll-wheel zoomable), Gaussian blur, and mosaic pixelation—all interactively movable before commit with strict LIFO undo.

### 🪄 Multi-Photo Creation Studio & Creative Effects
- **Microsoft ICE-Style Panorama Stitcher:** Cylindrical/spherical stitching with automatic maximal interior rectangle crop and boundary inpainting.
- **Collage Studio:** Modern Grid, Masonry, Polaroid table scatter with customizable captions, and *"Then & Now"* chronological comparison cards.
- **Photo Fuse:** Combines the best smiles/expressions from a burst of group photos using color-matched Poisson seamless cloning.
- **Colour Pop & Portrait Depth Bokeh:** Interactive multi-colour eyedropper (`Shift`/`Ctrl` `+`/`-` selection), edge-aware bilateral masking, and multi-scale pyramid bokeh blur.
- **Hybrid 2.5D 3D Cinematic Motion:** Disocclusion-inpainted dual-layer depth-mesh parallax video synthesis (`Depth-Anything-V2` + `MODNet` ONNX with guided filter fallback) exporting smooth MP4 camera orbits, dollies, and pans without rubber-edge stretching.

### 🏷 Hierarchical Tag Management & Left Navigation Dock
- **Left Navigation Tree:** Folders (with subfolder recursion), Date Taken tree (`Year -> Month -> Day`), People, and Ratings (5★–1★).
- **Descriptive Tag Tree:** Multi-level nesting, drag-and-drop reparenting, live item count badges, and 3-state Boolean checkboxes (`AND`, `OR`, `NOT`).
- **Tag Hierarchy Dialog:** Interactive tag manager with drag-and-drop and *Clean Up Duplicate Root Tags*.

### 🔍 Perceptual Duplicate & Burst Finder (Find Ribbon Tab)
- **Multi-Stage Detection Engine:** Instant byte-identical deduplication, multi-index 64-bit `pHash`/`dHash` for recompressed/edited images, and photo burst sequences with Laplacian focus sharpness scoring.
- **Side-by-Side Comparison & Smart Auto-Select:** Clean comparison cards with 1-click recommendations (*Keep Highest Resolution*, *Keep Largest File Size*, *Keep Oldest Date*, *Keep Most Metadata*, *Keep Best Sharpness*) and safe removal via the Windows Recycle Bin (`Send2Trash`).

### 🗺 Interactive World Map Geotagging
- Embedded OpenStreetMap Leaflet integration with click-to-pin coordinate tagging, reverse-geocoded location search, and SQLite `media_rtree` bounding-box spatial clustering.

### ⚡ Dual-Database Architecture & 500k-Item Scale
- **Catalog Database (`catalog.db`):** High-speed SQLite WAL storing media records, EXIF/XMP metadata, hierarchical tags, and ArcFace embeddings.
- **Thumbnails Database (`thumbnails.db`):** Dedicated BLOB storage caching EXIF-oriented 256×256 WebP thumbnails and 96×96 face crops for instantaneous scrolling.
- **Production Safeguards:** Safe Mode file write-back pause, worker task monitor with 1-click pause, online SQLite backup API, and RAW file container isolation via `.xmp` sidecars.

---

## Running from Source

### Prerequisites
- Python 3.11, 3.12, or 3.13 (64-bit) on Windows 10/11
- Git

### Setup
```bash
# 1. Clone repository
git clone https://github.com/docsucram/Essentials-Photo-Gallery.git
cd Essentials-Photo-Gallery

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows (pwsh / cmd)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch application
python main.py
```

### Building Standalone Executable & Installer
```cmd
1_build_exe.bat         # Compiles standalone 64-bit Windows executable via PyInstaller
2_test_exe.bat          # Launches and tests the compiled binary
3_build_installer.bat   # Builds Windows installer setup (EssentialsPhotoGallery_Setup_x64.exe) via Inno Setup 6
```

---

## Architecture & Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md): Comprehensive as-built architecture, database schemas, threading models, and specifications.
- [`SPEC.md`](SPEC.md): Executive functional and technical specifications summary.
- [`editing_engine.md`](editing_engine.md): Mathematical and algorithmic specifications for non-destructive image processing.
- [`edit_and_create_engine.md`](edit_and_create_engine.md) & [`create_engine.md`](create_engine.md): Technical review of the Edit, Depth & Light, and Multi-Photo Creation Studio engines.
- [`futureideas.md`](futureideas.md): Forward-looking innovation roadmap (local semantic search, OCR, offline places tree).

---

## License

This project is licensed under the [MIT License](LICENSE).
