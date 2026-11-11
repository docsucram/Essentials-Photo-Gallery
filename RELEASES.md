# Essentials Photo Gallery — Releases

Release history and user-facing highlights matching published Windows installers (`EssentialsPhotoGallery_Setup_x64.exe`) and repository release tags.

---

## Unreleased (Post-v0.11.0)

- **Static Albums & 17-Dimension Boolean Smart Albums:**
  - Added `My Albums` (non-destructive static collections via `Add to Album ▸` / `New Album from Selection...` / `Remove from Album`) and `Smart Albums` to the top of the Left Navigation pane.
  - Seeded built-in Smart Albums (*On This Day*, *4+ Star & Flagged Favorites*, *All Videos*).
  - Added interactive `SmartAlbumDialog` boolean rule builder (`ALL`/`ANY`) with live catalog match preview and *Save Current Filter as Smart Album...*.
  - Added EXIF `camera_make` / `camera_model` indexing for fast camera-based Smart Album filtering.
- **Contextual World Map Ribbon & Navigation Upgrade:**
  - Switching to World Map view activates the contextual **Map** ribbon tab (Location Search, Clean/Street/Satellite styles, Year/Person/Rating filters synced to the Search Scope bar, Fit to photos, View area in grid, Panes, and Close map) while displaying geotagged item and cluster counts in the status bar.
  - Added **People** view switch button directly in `View -> Views`, drag-rectangle zoom on the map canvas, and status bar / background task menu shortcuts to Options.
- **Orientation & Thumbnail Polish:** Preserved orientation consistency between gallery thumbnails and single-image color/adjustment saves.

---

## v0.11.0 (2026-11-10)

- **Next-Generation Image Formats & Color Management:**
  - Added native **HEIC / HEIF / HIF**, **AVIF**, and **JPEG XL (`.jxl`)** support across scanning, thumbnailing, Single Image View, and export/Save as Copy.
  - Automatic Display P3 ICC-to-sRGB conversion (`ImageCms`), alpha preservation, and PyInstaller DLL/plugin bundling.
- **Depth & Light, Markup & Spot Fix Improvements:**
  - Normalized 3D Depth & Light bokeh/boundary controls (`depth_blur`, `focal_plane`, `bokeh_glow`, near/far relighting, warmth, and atmospheric depth haze).
  - Upgraded object eraser / spot repair, 1-click subject cutout improvements, red-eye reduction refinements, and WYSIWYG markup tools.
  - Removed heavy `scipy` and `scikit-image` dependencies in favor of lean NumPy/OpenCV pipelines.

---

## v0.10.0 (2026-11-08)

- **Reverse Geotagging, Places Tree & Interactive World Map:**
  - Embedded offline gazetteer (3,500+ world cities and UK settlements) plus SQLite-cached Nominatim reverse geocoding (`saved_places`).
  - Collapsible `Country -> State / County -> City` **Places** hierarchy in the navigation pane and interactive Leaflet World Map view with R*Tree bounding-box clustering.
- **Modern Fluent UI Theme & Integrated Title Bar QAT:**
  - Custom window title bar with Quick Access Toolbar, dynamic Fluent tab indicators, Dark Studio polish, and `People` dropdown sizing improvements.
- **Batch People Tag Health & Pipeline Hardening:** Outlier health review tools, RAW & media pipeline fixes, and options menu upgrades.

---

## v0.9.0 (2026-11-04)

- **Pre-release Stabilization:**
  - Fixed hierarchical descriptive tag synchronization, face crop refresh, small-image zoom fit, `Ctrl+0` toggle, View Details spacing, and `.OriginalFiles` backup safety.
