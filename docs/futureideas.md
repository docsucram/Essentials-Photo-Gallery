# Future Ideas & Architectural Roadmap

This document tracks **genuinely unbuilt future enhancements** for **Essentials Photo Gallery**, maintaining the classic Windows Live Photo Gallery (WLPG) ease-of-use, clean UI, and speed while expanding optional local AI, heritage migration, and enthusiast editing capabilities.

> **Graduated Features:** Completed initiatives—including **v0.18.0 (Scoped / Incremental Nav Refresh, Startup Stall Fixes `~14.2 s → ~3.2 s`, `media_signatures` Reuse + Sharpness Tie-Breaker in Duplicate Finder, Non-Blocking Progressive NumPy-Vectorized Semantic-Blended `Find Similar`, Semantic Subject / Natural-Language Rules in Smart Albums, Priority 1–3 SQL/Cache Optimizations, Mosaic Flag Squircles & 3 Default-Off Monochrome Tag Badges, Always-On Favourites, Unified Image Right-Click Menu & Compare Ribbon AF Focus Points, 2-Column Manage Tags Shortcuts & Delete Tag, Geotag Hover/Copy & Compact Folder Path, Video Frame Capture, Stack-Aware Delete/Move/Rename, Upgraded Key-Aware Auto Exposure & Neutral-Lab Auto Color, Fine Tune Look Profiles + `.cube` 3D LUT Importer, and Map View Coalesced Cluster Thumbnails)**, **v0.17.0 (Thumbnail-Time Visual Signatures `media_signatures` + Undoable Hardened Auto-Stacking, 4-Mode Photo Layout Engine `Grid`/`Tiles`/`Mosaic`/`Masonry`, Optional `Cameras` & `Subjects` Nav Sections, `F1` Keyboard Shortcuts Modal + `Alt+1..9` Quick-Tag Hotkeys, and Optional `MobileCLIP-S1` ONNX `SemanticVectorRepository` + FTS5 RRF Search)**, **v0.16.0 World Map View Speed Overhaul, P2 Grid/RAW/Background Perf, Photographer Culling Toolkit (Zoom Minimap, Batch Rename `F2`, Borderless Fullscreen Spacebar Quick Preview + `Shift+Rating/Flag/Fav` Auto-Advance, Camera AF Area Overlay, Animated Filmstrip with Attached Pull-Tab), Photo Favourites (`is_favourite`) & Tri-State Pick/Reject Flags, Auto-Detecting Google Picasa Importer, Apple Live Photos & Android Motion Photos, Offline Drive/NAS Browsing (`volume_offline`), and Duplicate Preferred Folder Priority**, **Depth & Light 2.0**, **UI Theme Architecture Consolidation**, **Visual Similarity Search (`Find Similar`)**, **Companion Phone GPS Matching & GPX Interpolation**, **Fine Tune Two-Tier `Basic | Advanced` Mode**, **Phase 1 Regional Masking (`Whole | Subject | Bg | Sky`)**, SQLite FTS5 RapidOCR search, Per-Folder Scan & Privacy Controls with PBKDF2 PIN lock, and Unified `LaMa` ONNX Generative Inpainting—have been graduated into [`docs/ARCHITECTURE.md`](ARCHITECTURE.md), [`RELEASES.md`](../RELEASES.md), and [`docs/editing_engine.md`](editing_engine.md).

---

## v0.19.0 — Editing Studio, Ribbon Integration & Maintenance

* **Dedicated Contextual `Markup` Ribbon Tab & Typography Upgrades:**
  * Move Markup tools from the floating toolbar into a clean contextual **`Markup`** Ribbon tab (keeping WLPG's clean visual language rather than Canva-Lite bloat) and add **Font Family** choice alongside size, bold/italic, colour, and background pill toggle.
* **Spot Fix Improvements & Archival Old-Photo Restoration Presets:**
  * **Auto "Remove Orange Date Stamp" Preset:** Detect saturated orange/red LED quartz date-stamp digits in bottom corners of scanned film/compact photos and feed the dilated mask into `LaMaInpaintEngine`.
  * **Dust & Film Scratch Line Mode:** Hold `Shift` in Spot Fix to click-drag straight scratch lines across scanned slides/negatives and heal them in one stroke.
* **Perspective / Keystone Sliders in Crop & Straighten:**
  * Add `Vertical Perspective (-100..+100)` and `Horizontal Perspective (-100..+100)` sliders solved via 4-point projective homography (`cv2.getPerspectiveTransform` + `cv2.warpPerspective(..., flags=cv2.INTER_LANCZOS4)`), paired with analytical auto-crop or optional `LaMa` corner-wedge fill.
* **Relocatable Database & Thumbnail Cache + SHA-256 Add-on Integrity:**
  * Allow moving `thumbnails.db` and `catalog.db` to a secondary drive from **File → Options → Database** with a cancellable migration progress dialog.
  * Add `sha256sums.txt` integrity verification across optional ONNX add-on downloads in **File → Options → Add-ons**.
* **Auto-Update Checking & Release Notes in About Dialog:**
  * Optional automatic background update check on launch (configurable in Options) plus readable inline release notes right inside the **About / Update Available** dialog.

---

## v0.20.0 — Discovery, Color & Creative Effects

* **Automatic Trip & Event Clustering (`Group by` Option):**
  * Combine temporal gap detection ($\Delta t > 6\text{ hours}$ or multi-day travel away from home centroid) with `location_city` / `location_country` transitions to offer an **`Arrange / Group by: Events / Trips`** option (provided grouping stays fast on large libraries), with optional one-click **"Save Trip as Album"**.
* **Dominant Color Swatch Filtering (`Find` Tab):**
  * Add a 10-swatch clickable color filter bar to the `Find` ribbon tab matching precomputed Tier 1 `media_signatures` CIE LAB color data.

---

## Longer-Term / Unscheduled Ideas

* **Quick Social Share / Upload (Instagram / Facebook):** Streamlined share/upload or social framing workflow once the Markup Ribbon is settled.
* **Zero-Shot Auto-Tagging (`concept_presets.bin`) & Multilingual CLIP Pack:** Optional `~120` pre-computed categories driving **Suggested Smart Tags** plus multilingual CLIP text encoder pack.
* **Metadata Merge on Duplicate Cleanup:** Non-destructive **"Merge Tags, People, Caption & Rating into Kept Photo"** option when recycling duplicates.
* **Multi-Language OCR & Interactive Text Selection:** Optional CJK / Latin / Cyrillic RapidOCR packs and click-to-copy text bounding box overlay in Single Image View.
* **Microsoft Ribbon `Alt` KeyTips:** Classic `Alt` overlay badges (`H`, `E`, `F`, `C`, `V`) for two-stroke Ribbon keyboard activation.
* **Photo Fuse Boundary Fill (`LaMa`):** Fill thin outer edge slivers on handheld burst alignment instead of cropping in.
* **RAW Display Preference:** Configurable RAW display mode (*Embedded Preview* vs. *Full Demosaic* vs. *Paired JPEG*).
* **Map Polygon Geofencing & KML Track Import:** Google Timeline `.kml` support in `GpsMatchDialog` and lasso/polygon filtering on **World Map View**.
* **Beat-Synced Slideshow Music & MP4 Export:** Short-time Fourier flux onset detection (NumPy FFT) snapping Ken Burns transitions to musical bars.
* **Visual Calendar Density Heatmap:** Optional GitHub-style activity heatmap inside the **Date Taken** navigation panel.
* **"On This Day" Memories & Automatic Smart Album Examples *(Low Priority)*:** Optional future enhancement inside the **Smart Albums** view rather than cluttering the main Thumbnail Grid.

---

## Removed / Rethink

The following previously proposed ideas have been **removed or placed on hold for rethinking** to keep Essentials Photo Gallery fast, clean, and easy to understand:

* **4th AI Conceptual Similarity Mode inside `Find Duplicates`:** *Removed.* Putting conceptual/semantic similarity inside Duplicate Finder confuses "exact/burst duplicates to delete" with "similar-looking scenes to explore." Instead, `Find Duplicates` stays focused on fast cached `media_signatures` + sharpness-aware keeper picking, while `Find Similar` (right-click) transparently blends MobileCLIP-S1 semantic embeddings behind the scenes.
* **Manual Stack Creation UI:** *Removed.* Previously prototyped and removed because it added extra dropdown complexity and overlapped confusingly with Albums.
* **Semantic Masking Phase 2 (`MobileSAM` / `EdgeSAM` + `sam3_presets.bin`):** *Removed / Rethink.* Heavy model footprint and complex Lightroom-style masking UX that goes beyond WLPG's clean, approachable editing philosophy.
* **Portable `.epgbackup` Catalog Bundles:** *Removed / Rethink.* High maintenance and path-remapping edge-case complexity relative to standard folder/sidecar backup workflows.
* **Customizable Status Bar / Info Pane Field Reordering:** *Removed / Rethink.* Unnecessary settings bloat; sensible built-in defaults keep the UI consistent and uncluttered.
