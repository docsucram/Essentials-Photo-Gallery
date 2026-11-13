# Essentials Photo Gallery — Releases

Release history matching the published GitHub Releases (`EssentialsPhotoGallery_Setup_x64.exe`).

---

## Essentials Photo Gallery v0.12.0

- **Photo Editor & Colour Pop Performance Overhaul (`src/ui/media_viewer.py`, `src/imaging/opencv_engine.py`, `src/imaging/creative_effects.py`, `src/ui/create_studio_dialogs.py`):**
  - **Background Full-Resolution Master Pass:** Moved the 250ms debounced `render_adjusted_image` master render onto a `QRunnable`/`QThreadPool` worker with generation-counter cancellation and synchronous `ensure_master_rendered()` before Save/Export, eliminating UI freezes after releasing sliders.
  - **Dirty-Stage Checkpoint Caching:** Added a thread-safe bounded `_STAGE_CACHE` (`dd_img`, `tl_img`, `bokeh_data`) in `render_adjusted_image` plus in-place `float32` buffers so downstream adjustments skip upstream Denoise, Dehaze, Tone/LAB, and Bokeh Blur passes.
  - **Two-Tier Adaptive Proxy:** Added a `960px` `scrub_proxy_rgb` during active slider dragging (`_is_sliding`) that snaps to the crisp `1280px` `display_proxy_rgb` immediately on release.
  - **Faster Colour Pop Selection & Tolerance Scrubbing:** Added `compute_multi_color_similarity_mask()` (single LAB conversion and single `guided_filter_matte` pass across all picked colours, skipping the guided filter during active tolerance slider drags) and cached `float32` proxy buffers in `ColourPopDialog`.
- **Photo Fuse Overhaul (`src/imaging/create_tools.py`, `src/ui/create_studio_dialogs.py`):**
  - Added InsightFace `det_10g` (with `CPUExecutionProvider`) + Haar fallback face detection, robust ECC/homography burst alignment with unalignable-shot reporting, cached alignment across base-shot switches, landmark-guided seamless Poisson face blending (`cv2.seamlessClone`), cleaner base-shot selection UX, and show/hide face box overlays.
- **Panorama Stitcher Stability (`src/imaging/create_tools.py`, `src/ui/create_studio_dialogs.py`):**
  - Disabled OpenCL (`cv2.ocl.setUseOpenCL(False)`) during stitching and downscaled oversized inputs to prevent native OpenCV/OpenCL driver crashes during panorama synthesis.
- **Batch People Tagging & Navigation Restore (`src/ui/batch_people_view.py`, `src/ui/main_window.py`):**
  - Guarded `None` `outlier_reason` values in `QStyledItemDelegate` `paint`/`editorEvent` and selection handlers, and restored Left Navigation pane visibility when returning from Batch People Tag to Thumbnail, People, or Album views.
- **Multi-Photo Compare & Light Table View (`src/ui/compare_view.py`):**
  - Added native multi-photo side-by-side light table comparison accessible via `Compare` (`C`) on the View ribbon tab (`Views` group) or thumbnail selection context menu.
  - **Multi-Slot Adaptive Split Layouts:** Resizable split panes supporting `2-Up Side-by-Side`, `2-Up Top / Bottom`, `3-Up Columns`, `2×2 Quad Grid`, and an interactive Before/After **Wipe Slider** with a draggable `◀▶` curtain divider (double-click to reset to 50%).
  - **Pin Champion & Cycle Challenger Workflow:** Pin any pane (`📌 Pin`) to lock a champion photo while cycling remaining candidate photos through the active pane using `Left`/`Right` arrow keys or the bottom Candidate Filmstrip.
  - **Candidate Filmstrip with Badges:** Automatically appears for 5+ selected photos or whenever candidates exceed active panes, displaying live thumbnails with Pick (`⚑`), Pin (`P`), and Star rating (`★N`) status badges.
  - **Auto-Advance Mode:** Optional ribbon toggle (`Auto-Advance`) that automatically promotes the next challenger candidate upon rating (`0–5`) or flagging (`P`).
  - **Synchronized Viewport Pan & Zoom:** 60 FPS real-time normalized `[0.0..1.0]` pan and zoom synchronization with `Shift + Drag` and `Shift + Wheel` independent nudge override, `1:1 Actual Pixels (100%)`, and `Fit to Pane`.
  - **1-Click "Zoom to Face (F)":** Detects faces and zooms all panes synchronously onto matching subjects (~45% viewport height), cycling through detected individuals on successive presses.
  - **Focus Peaking & Clipping Warnings:** Live green edge-contrast focus peaking overlay and highlight/shadow clipping warnings ("blinkies" - blown highlights in red $\ge 250$, crushed shadows in blue $\le 5$).
  - **Universal RAW, TIFF & Next-Gen Support:** Full camera RAW decoding (`.arw`, `.cr2`, `.cr3`, `.nef`, `.dng`, `.raf`, `.rw2`, `.orf`, `.ori`, etc.), raw-sensor TIFF containers (e.g. Pentax 14-bit, Leica 16-bit, Nikon Z9, Blackmagic 12-bit RAW TIFFs) via `rawpy` and Windows Imaging Component (WIC), with EXIF auto-orientation.
  - **Camera & EXIF Header Difference Highlighting:** Pane headers display camera make/model and exposure metadata (shutter speed, aperture, ISO, focal length), automatically highlighted in amber where panes differ.
  - **Non-Destructive Culling & Quick Rotate:** Integrated 1–5 star ratings, Pick flag (`P`), quick 90° view rotate (`⟲` and `⟳`), Dismiss (`X`), and Recycle Bin delete (`Del`), plus 1-click **Open Selected in Editor**.
  - **Dedicated Compare Ribbon & Dark Backdrop:** Contextual `Compare` tab with custom layout icons, half-and-half contrast dark backdrop toggle, and dark splitter dividers (`#2A3240`).
- **Albums View & Smart Album Extensions:**
  - Added dedicated **Albums View** (`src/ui/albums_view.py`) with stacked preview thumbnails, quick album creation (`New Album +`, `New Smart Album +`), and customizable Left Navigation ordering/visibility in Options.
  - Extended Smart Albums with **Camera Manufacturer** (`camera_make`) and **Detected Face Count** conditions (`at least`, `at most`, `exactly X faces`), with graceful handling of empty/unspecified criteria.
  - Fixed `update_smart_album()` argument mismatch when modifying smart albums and refined condition removal buttons with clear red `✕` delete actions.
- **Static Albums & 17-Dimension Boolean Smart Albums:**
  - Added `My Albums` (non-destructive static collections via `Add to Album ▸` / `New Album from Selection...` / `Remove from Album`) and `Smart Albums` to the top of the Left Navigation pane.
  - Seeded built-in Smart Albums (*On This Day*, *4+ Star & Flagged Favorites*, *All Videos*).
  - Added interactive `SmartAlbumDialog` boolean rule builder (`ALL`/`ANY`) with live catalog match preview and *Save Current Filter as Smart Album...*.
  - Added EXIF `camera_make` / `camera_model` indexing for fast camera-based Smart Album filtering.
- **Contextual World Map Ribbon & Navigation Upgrade:**
  - Switching to World Map view activates the contextual **Map** ribbon tab (Location Search, Clean/Street/Satellite styles, Year/Person/Rating filters synced to the Search Scope bar, Fit to photos, View area in grid, Panes, and Close map) while displaying geotagged item and cluster counts in the status bar.
  - Added **People** view switch button directly in `View -> Views`, drag-rectangle zoom on the map canvas, and status bar / background task menu shortcuts to Options.
- **Windows Native Framing, Snap & Drop Shadow:**
  - Restored native Win32/DWM frame integration (`WS_THICKFRAME | WS_CAPTION | WS_MAXIMIZEBOX | WS_MINIMIZEBOX`), Aero Snap (`Win+Left/Right/Up/Down` and drag-to-edge snapping), native DWM drop shadow and Windows 11 rounded corners, and DPI-accurate `WM_NCHITTEST` resize borders while keeping the custom Ribbon title bar.
- **People Suggestions & Batch Tagging UX Fixes:**
  - Clicking an inline face suggestion thumbnail in People View now opens the photo for inspection instead of silently confirming and hiding it; explicit `✓` and `✕` buttons handle confirm/reject.
  - Fixed `View all suggestions...` missing suggestions outside the global 5,000-row / confidence threshold, fixed clicking `All Suggestions` after opening filtered to one person so all other people's sections appear, and added a `Filtered to: <Name>` bar with a **`✕ Show All People`** reset button.
- **Orientation & Thumbnail Polish:** Preserved EXIF/visual orientation consistency between gallery thumbnails and single-image color/adjustment saves.

---

## Essentials Photo Gallery v0.11.0

- **Depth & Light & Cutout Tools:** Added 1-click Subject Cutout (Copy Subject & Remove Background to transparent PNG), interactive mask boundary shift/feather with live preview, and an atmospheric 3D haze engine (optical bloom, horizon layering, soft-knee veiling).
- **Spot Fix & Inpainting:** Integrated AI Object Eraser (LaMa inpainting) alongside smart healing and red-eye correction.
- **Next-Gen Image Format Support:** Native viewing and thumbnail support for AVIF, WebP, HEIC/HEIF, and extended RAW formats.
- **Map View Overhaul:** Smooth zooming without flicker, reliable offline tile caching, cluster pin navigation, and seamless view switching.
- **Face Detection:** Added full library Rescan option, improved face thumbnail caching, and grouping stability.
- **Navigation & Filtering:** Added Orientation filter (Landscape / Portrait), persistent selection highlights across background refreshes, and smooth drag-and-drop file organization.
- **Static Slideshow:** Added a static fullscreen presentation mode alongside animated pan-and-zoom.
- **Fluent Style & UI Themes:** Updated Fluent icons, refined Ribbon controls, and improved theme switching.
- **Lightweight Distribution:** Removed `scipy` and `scikit-image` dependencies to reduce installer and disk size while improving startup time.
- **Groups:** Added group scroll indicator and Group Context Menu Options (Collapse Group, Collapse all groups, Select All Group Items).
- **Performance & Packaging:** 60 FPS interactive slider scrubbing for Fine Tune controls, synchronized Ctrl + wheel zoom, and PyInstaller ONNX model bundling and resolution fixes.

---

## Essentials Photo Gallery v0.10.0

### Key Changes and Improvements

#### Places and Offline Reverse Geotagging
- Offline world gazetteer with 25,000+ cities and regions.
- Places tree added to the Left Navigation dock (Country / State / City hierarchy).

#### World Map View Overhaul
- Multiple map styles supported: OpenStreetMap Standard, Carto Light, Carto Dark, Satellite/Aerial, and Stamen Terrain.
- Clustered photo pins with preview thumbnail bubbles and responsive zoom.
- Updated map view UI.

#### Batch People Tagging and Face Health
- **Review Outliers Mode:** Identifies potential mis-tagged faces by comparing face embeddings against person cluster centroids.
- **Missing Face Boxes Mode:** Audits photos tagged with a person that lack bounding box coordinates.
- **Metadata Recovery:** Re-parses and restores face regions directly from original photo XMP metadata on disk.
- Batch People Tagging Ribbon implemented.
- Cleaned up Face Recognition Options dialog.

#### Modern Fluent Theme and Title Bar QAT
- Added Fluent (Windows 11) light theme alongside Classic Aero and Dark modes.
- Integrated Quick Access Toolbar (QAT) into the custom title bar header (defaults to Back, Forward, Undo, Redo).

#### Media Pipeline and RAW Support
- Expanded camera RAW decoding (`DNG`, `CR2`, `CR3`, `NEF`, `ARW`, `RW2`, `ORF`, `PEF`).
- Thumbnail generation and metadata reading for video files (`MP4`, `MOV`, `WMV`, `AVI`, `MKV`, `M4V`).
- Automatic EXIF orientation normalization and color profile preservation.

#### Navigation and Interface Fixes
- Fixed back/forward history navigation across grid, single viewer, map, and batch tagging views.
- Improved docking pane auto-hide and restore consistency.
- Corrected dropdown border overlap in Quick Find People.
- Fixed Gallery arrow key navigation.

---

## Essentials Photo Gallery v0.9.0 (Beta)

A modern, 64-bit continuation of Windows Live Photo Gallery (WLPG) for Windows 10 and 11.

> **Important (Development Beta):** This project is in active development. File write-back is turned off by default, and we recommend leaving it turned off when using existing photo libraries so tag and metadata updates stay in the local database rather than modifying original files. Always keep backups of important photos.

### Highlights
- **WLPG Ribbon look & feel** (Home, Edit, Find, Create, View) with Light and Dark theme support.
- **WLPG hierarchical tagging** — nested descriptive tags, ratings, flags, captions, and fast multi-tag filtering.
- **64-bit large-library architecture** — eliminates 32-bit out-of-memory crashes and stays responsive on 500,000+ photo libraries.
- **Updated facial recognition & People view** — local SCRFD + ArcFace (ONNX) detection, face crops view, and batch confirmation.
- **Updated creation tools** — Photo Collage, Photo Fuse, and Panoramic Stitching.
- **New:** Duplicate & burst photo finder, interactive Markup tools, Portrait blur & relighting, and Map view (WIP).
