# Essentials Photo Gallery

[![Release](https://img.shields.io/github/v/release/docsucram/Essentials-Photo-Gallery?label=Release&color=blue)](https://github.com/docsucram/Essentials-Photo-Gallery/releases/latest)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%20%7C%2011%20(x64)-0078D6.svg)](https://github.com/docsucram/Essentials-Photo-Gallery/releases/latest)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

**Essentials Photo Gallery** is a modern, 64-bit update to Microsoft's long-abandoned **Windows Live Photo Gallery (WLPG)**.

Microsoft discontinued Windows Essentials back in 2017, and for anyone with a big local folder collection of family photos, nothing since has quite matched how fast, intuitive, and practical WLPG was for browsing, organising, tagging, and doing quick edits. This project is an attempt to bring that exact feel and workflow back on Windows 10 and 11—fixing WLPG's old limitations while adding a handful of genuinely useful modern tools.

> [!IMPORTANT]
> **Project Status — In Active Development**
> This project is still actively being developed and tested. File write-back is **turned off by default**, and if you are trying it out on an existing photo collection, **it is strongly recommended to leave file write-back turned off** so tag and metadata changes stay in the local database rather than modifying your original image files. As always, please keep backups of important photos.

---

## Download

Standalone 64-bit Windows installers (`EssentialsPhotoGallery_Setup_x64.exe`) are available on the **[Releases](https://github.com/docsucram/Essentials-Photo-Gallery/releases/latest)** page—no Python setup needed.

*Note: Because this is a hobby open-source project without a paid code-signing certificate, Windows SmartScreen will likely show an "Unknown publisher" warning the first time you run the installer (**More info** -> **Run anyway**).*

---

## Highlights & Features

- **Classic WLPG Ribbon Look & Feel (with Modern Themes & Integrated QAT)**
  Recreates the authentic ribbon layout (*Home*, *Edit*, *Find*, *Create*, *View*), integrated title bar with Quick Access Toolbar (QAT), left navigation tree, and right-hand tag/information pane. Supports three distinct themes: **Classic Aero (Light)**, modern **Microsoft Fluent (Windows 11)** with vector icons and dynamic tab indicator, and **Dark Studio**.

- **WLPG Hierarchical Tagging**
  Full support for nested descriptive tags (`Place/England/London`, `Subject/Occasions/...`), drag-and-drop tag tree management, star ratings, flags, captions, and fast multi-tag filtering—compatible with existing WLPG XMP/EXIF tags.

- **Built for Large Libraries (64-Bit Architecture)**
  WLPG was a 32-bit app that would frequently hit out-of-memory errors on large collections or high-megapixel cameras. Essentials Photo Gallery is native 64-bit with virtualised grid scrolling and separate SQLite catalog and thumbnail databases, designed to stay snappy even with **500,000+ photos and videos**.

- **Updated Facial Recognition & People View**
  Replaces WLPG's ageing face detector with local **SCRFD + ArcFace (ONNX)** models for much faster, more accurate face detection and suggestions. Includes a dedicated People view, expandable Quick Find People dropdown with corner resizing, face crop view, and batch confirmation tools—all running 100% locally offline.

- **Updated Creation Tools**
  Brings back WLPG's multi-photo creation workflows without needing old external plugins:
  - **Photo Collage** (grid, masonry, scattered Polaroid, and Then & Now layouts)
  - **Photo Fuse** (combine the best smiles/expressions from a group burst into one photo)
  - **Panoramic Stitching** (automatic panorama stitcher inspired by Microsoft ICE)

- **Interactive World Map & Hierarchical Places Tree**
  Full interactive Leaflet world map with persistent disk tile caching (Clean Voyager, Street, Satellite, Terrain) and offline vector canvas fallback. Features sub-5ms spatial clustering backed by SQLite R*Tree (`media_rtree`), an offline gazetteer with 3,500+ world cities and UK settlements, Nominatim reverse geocoding with local SQLite caching, and a collapsible **Places** hierarchy (`Country -> State / County -> City / Town`) in the navigation dock with live item counts.

- **Duplicate & Burst Photo Finder**
  Finds exact file duplicates, resized/re-saved copies (via perceptual hashing), and rapid burst sequences, with side-by-side sharpness comparison to help clean up clutter safely via the Recycle Bin.

- **Markup & Privacy Redaction Tools**
  Built-in annotation toolbar in single-image view for drawing arrows, boxes, circles, text callouts, zoom magnifier loupes, and blur/pixelate redaction before saving or copying.

- **Portrait Blur & Relighting Tools**
  Alongside classic WLPG quick edits (crop, straighten, red-eye, spot fix/retouch, exposure, colour, and effects), adds depth-based background portrait blur, subject relighting, and colour pop.

---

## Running from Source

Requires **Python 3.11–3.13 (64-bit)** on Windows 10 or 11.

```powershell
# Clone the repo
git clone https://github.com/docsucram/Essentials-Photo-Gallery.git
cd Essentials-Photo-Gallery

# Set up a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies and launch
pip install -r requirements.txt
python main.py
```

### Building the `.exe` & Installer

If you want to build a standalone executable or installer locally (installer requires [Inno Setup 6](https://jrsoftware.org/isinfo.php)):

```cmd
1_build_exe.bat         :: Builds standalone 64-bit app via PyInstaller
2_test_exe.bat          :: Runs the compiled build
3_build_installer.bat   :: Packages EssentialsPhotoGallery_Setup_x64.exe
```

---

## License

Released under the [MIT License](LICENSE).
