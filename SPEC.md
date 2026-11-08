# Essentials Photo Gallery — System Specification Summary (`SPEC.md`)

This document summarizes the core functional, technical, and operational requirements for **Essentials Photo Gallery**. For the complete as-built architecture, database schemas, threading models, signal flow tables, and implementation details, refer to the master architecture document:

👉 **[Complete System Architecture & Technical Specification (`ARCHITECTURE.md`)](ARCHITECTURE.md)**  
👉 **[Image Editing & Multi-Photo Studio Mathematical Engine (`editing_engine.md`)](editing_engine.md)**  
👉 **[Future Innovation Roadmap (`futureideas.md`)](futureideas.md)**

---

## 1. Project Goal & Scale Requirements

Build a high-performance, 64-bit Windows desktop replacement for **Windows Live Photo Gallery (WLPG)** in Python 3.11–3.13 and PySide6, capable of managing **500,000+ photos and videos** with:
- Sub-10ms UI query and filtering responsiveness.
- 60 FPS smooth scrolling in the virtualized grid view.
- 100% local, private, and offline execution with zero cloud telemetry.
- Non-destructive image editing preserving originals in `.OriginalFiles/`.
- Standards-compliant dual XMP / MWG metadata synchronization.

---

## 2. Core Functional Requirements

### 2.1 Virtualized Grid & Navigation
- **Virtualization Budget:** Strict rule against naive widget lists (`QListWidget`, `QTableWidget`). All grids must use `QListView` in `IconMode` backed by lightweight tuple models (`CatalogListModel`), with $QPixmap$ objects generated asynchronously via thread-safe LRU caching.
- **Collapsible Section Headers:** Grouping by Date, Rating, Folder, Person, or Tag using $O(\log G)$ binary-searched header resolution.
- **Polaroid Card Delegate:** Interactive on-card 5-star rating widget, flag toggle pin, selection checkbox, video duration badge, and hover highlights.
- **Browser-Style History:** Forward/Back view history navigation (`Alt+Left` / `Alt+Right`, mouse thumb buttons).

### 2.2 Authentic WLPG Ribbon Toolbar, Integrated Title Bar & Themes
- **Tabs:** `Home`, `Edit`, `Find`, `Create`, `View`.
- **Integrated Title Bar & Quick Access Toolbar (QAT):** Title bar merges custom application branding, customizable QAT actions (New folder, Open with, Save, Undo, Redo, Delete, Slideshow, Thumbnails, Map, People, Rotate), and native Windows window control buttons.
- **Ribbon Context Menu:** Right-click actions for customizing QAT, toggling QAT above/below ribbon, and minimizing ribbon (<kbd>Ctrl</kbd> + <kbd>F1</kbd>).
- **Quick Find People Carousel & Resizable Dropdown:** Horizontal circular avatar carousel with names for instant 1-click person filtering, paired with an integrated dropdown combining Top People and Other People A-Z, resizable via a bottom-right corner grip.
- **Effects Preview Gallery:** 8 photographic preview tiles (*None, Classic Chrome, Warm Portrait, Vivid Landscape, Matte Vintage, Fine Art B&W, Cool Cinema, Sepia*) with smooth $0..100\%$ intensity blending.
- **Descriptive Tag Split Button:** Main button click opens the tag pane / focuses tag entry (`Ctrl+T`); dropdown arrow accesses *Manage tags...* and *Clean up unused tags*.
- **Three Visual Themes:** Dynamic styling across **Classic Aero (Light)**, **Microsoft Fluent (Windows 11)** with vector icon sets and smooth tab indicator, and **Dark Studio**.

### 2.3 Left Navigation Dock & Hierarchical Filtering
- **Folders:** Directory tree with *"Include subfolders"* recursion and drag-and-drop file organization.
- **Date Taken:** Hierarchical tree (`Year -> Month -> Day`) with live item counts.
- **Places:** Hierarchical tree (`Country -> State / County -> City / Town`) with live item counts and instant grid filtering.
- **Descriptive Tags:** Nested tag hierarchy with drag-and-drop reparenting, live count badges, and 3-state Boolean checkboxes (`Checked = Include (AND/OR)`, `Unchecked = Neutral`, `Alt-Check/Right-Click = Exclude (NOT)`).
- **Tag Hierarchy Manager:** Dedicated dialog with interactive tree, drag-and-drop, and *Clean Up Duplicate Root Tags* tool to merge orphan leaf tags into hierarchical paths with user confirmation and restore tracking.
- **People & Ratings:** Alphabetic directory of tagged people with face avatars, plus 5-star down to 1-star ratings filters.

### 2.4 Single Media Viewer & Non-Destructive Studio
- **60 FPS Viewport:** Interactive zoom/pan centered on cursor with hardware acceleration.
- **Dual-Pipeline Execution:** Real-time parametric slider adjustments against a $1280\text{px}$ display proxy with lossless commit against camera master arrays.
- **Face Tagging Overlay:**
  - Automatic hover detection with bounding boxes and name pills.
  - **Click-to-Rename Affordance:** Left-clicking any confirmed face badge opens the inline autocomplete rename popup pre-filled with the current name.
  - Suggested face accept (`[✓]`) and reject (`[✗]`) chips.
  - Right-click context menu (*Tag as someone else...*, *Not this person*, *Remove face box*).
- **Integrated Video Player:** Play/pause, scrubber bar, and timecode overlays.
- **Dark Theme Canvas:** Letterboxing canvas switches to `#14171C` in Dark Theme.

### 2.5 Biometric Face Recognition & "Batch People Tag"
- **AI Face Pipeline:** InsightFace SCRFD detector paired with 512-dimensional ArcFace vector embeddings via ONNX Runtime (CPU / DirectML GPU), with OpenCV Haar fallback.
- **Batch People Tag View:** Virtualized face confirmation grid rendering $96 \times 96$ WebP crops directly from `thumbnails.db`, grouped by person with collapsible headers and 1-click `[Confirm Selected]`.
- **Quick Full-Photo Preview Popover (`Space`):** Centered overlay highlighting the face in full-photo context without leaving the batch workflow.
- **Ergonomics:** `Enter` (Confirm), `Delete`/`Backspace` (Not this person), `I` (Don't show again), `Esc` (Back to Gallery).
- **People Directory:** Immich-style 130px circular portrait cards, recurring unnamed clusters, and timeline verification strip.

### 2.6 Multi-Photo Creation & Creative Engines
- **Microsoft ICE-Style Panorama Stitcher:** Cylindrical, Spherical, and Planar homography stitching with maximal interior rectangle auto-crop and boundary gap inpainting.
- **Interactive Collages & "Then & Now" Studio:** Grid, Masonry, and Polaroid table scatter layouts with randomized rotation jitter and drop shadows.
- **Photo Fuse:** Best-face group shot combiner with Poisson seamless cloning (`cv2.seamlessClone`).
- **Creative Effects:** Color Pop (subject segmentation with bilateral brush), Portrait Depth Bokeh, and 3D Cinematic Motion parallax video synthesis.
- **Vector Markup & Privacy Redaction:** Subpixel arrows, text pills, 2.5x loupe callouts, mosaic pixelation, and blur redactions.

### 2.7 Relative "Change Time Taken" Logic
- Configures timestamp on lead selected photo; computes $\Delta t = T_{\text{lead\_new}} - T_{\text{lead\_original}}$ and applies the relative offset to all selected items in a single transaction, flagging `metadata_dirty = 1` for background disk write-back.

### 2.8 Perceptual Duplicate & Burst Finder
- Integrated directly into the **Find** ribbon tab (`Clean up` group).
- Combines $O(1)$ fast byte hashing for exact duplicates, multi-index 64-bit `pHash`/`dHash` for perceptual near-duplicates, and time-delta clustering with Laplacian sharpness variance scoring for photo bursts.
- Side-by-side visual comparison workspace with attribute metric matrix, 1-click smart auto-selection rules (*Highest Resolution*, *Largest File Size*, *Oldest Date*, *Most Metadata*, *Best Sharpness*), and safe removal via the Windows Recycle Bin (`send2trash`).

### 2.9 Geotagging, Interactive World Map & Places Engine
- **Interactive Leaflet Map View:** Embedded world map supporting 4 raster layers (Clean Voyager, OpenStreetMap, Satellite, Terrain) with persistent tile caching and offline vector canvas fallback.
- **Spatial R*Tree Queries:** Queries SQLite `media_rtree` in $<5\text{ms}$ to cluster and plot markers within the active bounding box; clicking any cluster or pin filters the main gallery.
- **Offline Gazetteer & Nominatim Caching:** Embedded database of 3,500+ world cities and UK settlements with sub-2ms KD-tree reverse geocoding, plus OpenStreetMap Nominatim resolution with SQLite caching in `saved_places`.
- **Hierarchical Places Tree:** Collapsible `Country -> State/County -> City` navigation in the Left Navigation Dock with live item counts.
- **Metadata Synchronization:** Bidirectional GPS coordinates and IPTC/XMP location data sync (`Iptc4xmpExt:LocationCreated`).

---

## 3. Storage & Concurrency Architecture

- **`catalog.db` (SQLite WAL):** Manages folders, media records, hierarchical tags (`tag_closure`), ArcFace face clusters, permanent rejections, and spatial R*Tree (`media_rtree`).
- **`thumbnails.db` (SQLite WAL):** Dedicated BLOB storage for $256 \times 256$ WebP thumbnails (`PG64THMB` header) and $96 \times 96$ WebP face crops.
- **Threading Model:** Multi-threaded pipeline (Qt GUI thread, Scanner pool, Folder watcher, Thumbnail pool, Face detection worker, and Background metadata sync daemon) with zero UI-thread blocking file I/O.
- **Production Safeguards:** Safe Mode write-back pause, worker task monitor with 1-click pause, online SQLite backup API (`sqlite3.Connection.backup()`), and RAW file container isolation via `.xmp` sidecars.

---

## 4. Verification & Scale Standards

Core catalog queries, face recognition pipelines, non-destructive OpenCV editing recipes, and UI workflows are verified against automated test suites and benchmarked up to 100,000 and 500,000 item catalogs.
