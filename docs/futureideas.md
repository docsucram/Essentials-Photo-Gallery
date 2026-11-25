# Future Ideas & Architectural Roadmap

This document tracks **genuinely unbuilt future enhancements** for **Essentials Photo Gallery**, maintaining the classic Windows Live Photo Gallery (WLPG) ease-of-use, clean UI, and speed while expanding optional local AI, heritage migration, and enthusiast editing capabilities.

> **Graduated Features:** Completed initiatives—including **v0.18.0 Early Foundations (`media_signatures` Reuse + Sharpness Tie-Breaker in Duplicate Finder, Semantic-Blended `Find Similar`, Semantic Subject / Natural-Language Rules in Smart Albums, and Priority 1–3 SQL/Cache Optimizations)**, **v0.17.0 (Thumbnail-Time Visual Signatures `media_signatures` + Undoable Hardened Auto-Stacking, 4-Mode Photo Layout Engine `Grid`/`Tiles`/`Mosaic`/`Masonry`, Optional `Cameras` & `Subjects` Nav Sections, `F1` Keyboard Shortcuts Modal + `Alt+1..9` Quick-Tag Hotkeys, and Optional `MobileCLIP-S1` ONNX `SemanticVectorRepository` + FTS5 RRF Search)**, **v0.16.0 World Map View Speed Overhaul, P2 Grid/RAW/Background Perf, Photographer Culling Toolkit (Zoom Minimap, Batch Rename `F2`, Borderless Fullscreen Spacebar Quick Preview + `Shift+Rating/Flag/Fav` Auto-Advance, Camera AF Area Overlay, Animated Filmstrip with Attached Pull-Tab), Optional Photo Favourites (`is_favourite`) & Tri-State Pick/Reject Flags, Auto-Detecting Google Picasa Importer, Apple Live Photos & Android Motion Photos, Offline Drive/NAS Browsing (`volume_offline`), and Duplicate Preferred Folder Priority**, **Depth & Light 2.0**, **UI Theme Architecture Consolidation**, **Visual Similarity Search (`Find Similar`)**, **Companion Phone GPS Matching & GPX Interpolation**, **Fine Tune Two-Tier `Basic | Advanced` Mode**, **Phase 1 Regional Masking (`Whole | Subject | Bg | Sky`)**, SQLite FTS5 RapidOCR search, Per-Folder Scan & Privacy Controls with PBKDF2 PIN lock, and Unified `LaMa` ONNX Generative Inpainting—have been graduated into [`docs/ARCHITECTURE.md`](ARCHITECTURE.md), [`RELEASES.md`](../RELEASES.md), and [`docs/editing_engine.md`](editing_engine.md).

---

## 1. Contextual Markup Ribbon & Future Social Sharing

### 1.1 Dedicated Contextual `Markup` (and Optional `Spot Fix`) Ribbon Tab
* **User Value:** Replace the cramped floating Markup toolbar with a clean, focused contextual **`Markup`** Ribbon tab (potentially also housing or pairing cleanly with **`Spot Fix`** tools) so annotation stays intuitive and fits WLPG's visual language without turning into a cluttered design suite.
* **Planned Scope (v0.18.0):**
  * **Tools & Shapes:** Existing freehand pen, highlighter, arrows/lines, rectangles/circles, and blur/pixelate moved cleanly into Ribbon groups.
  * **Typography Upgrade:** Add **Font Family** selection alongside font size, bold/italic, colour, and background pill toggle.
  * **Keep It Simple:** Avoid heavy multi-layer Canva-style canvas bloat; focus on fast, effortless photo annotation.

### 1.2 Quick Social Share / Upload (Future — Not v0.18.0)
* **Future Goal:** Streamlined **Quick Share / Upload** to Instagram and Facebook (e.g. one-click aspect-ratio framing/padding or copy-to-clipboard / share workflow) once the core Markup Ribbon is settled.

---

## 2. RAW + JPEG Lifecycle & Media Organization

### 2.1 Unified RAW + JPEG/HEIC Lifecycle & Preview Controls
* Keep paired RAW + JPEG/HEIC files together during move, rename, copy, and Duplicate Finder scans; prompt with a deletion-scope choice (*Delete Both / RAW Only / JPEG Only*); and add a configurable RAW display preference (*Embedded Preview* vs. *Full Demosaic* vs. *Paired JPEG*).

### 2.2 Automatic Trip & Event Clustering (`Group by` Option)
* Combine temporal gap detection ($\Delta t > 6\text{ hours}$ or multi-day travel away from the user's home coordinate centroid) with `location_city` / `location_country` transitions to offer an **`Arrange / Group by: Events / Trips`** option in the gallery view (provided grouping stays fast on large libraries), with optional one-click **"Save Trip as Album"**.

---

## 3. Semantic AI, Multi-Language OCR & Integrity

### 3.1 Extensions to the `MobileCLIP-S1` Add-on & Duplicate Cleanup
* **Verified Integrity & Multilingual CLIP Pack:** Add `sha256sums.txt` integrity checks across optional ONNX add-on downloads in `File → Options → Add-ons`, plus an optional multilingual CLIP text encoder pack.
* **Zero-Shot Auto-Tagging (`concept_presets.bin`):** Expand beyond the 12 built-in Left Nav **Subjects** with `~120` pre-computed photographic categories driving optional **Suggested Smart Tags** (analogous to face suggestions).
* **Metadata Merge on Duplicate Cleanup:** Non-destructive **"Merge Tags, People, Caption & Rating into Kept Photo"** option when recycling duplicates.

### 3.2 Multi-Language OCR & Interactive Text Selection
* **Multi-Language Recognition Packs:** Optional CJK / Latin / Cyrillic RapidOCR packs in **File → Options → Add-ons**.
* **Interactive Text Selection in Single Image View:** Optional overlay button when viewing screenshots or documents that highlights detected text bounding boxes for direct click-to-copy on the image canvas.

---

## 4. Ribbon KeyTips

### 4.1 Microsoft Ribbon `Alt` KeyTips
* *(Unified `F1` Keyboard Shortcuts modal and configurable `Alt+1..9` Quick-Tag hotkeys shipped in v0.17.0.)*
* **Microsoft Ribbon `Alt` KeyTips:** Pressing `Alt` overlays classic keyboard navigation badges (`H` Home, `E` Edit, `F` Find, `C` Create, `V` View) and two-stroke command activation across the Ribbon.

---

## 5. Additional Editing & Generative Restoration

### 5.1 `LaMa` Context-Aware Fill Remaining Extensions (`Photo Fuse` & `Perspective/Keystone`)
* Extend the implemented `LaMaInpaintEngine` (`lama_fp32.onnx`) to the remaining geometric tools:
  * **Photo Fuse Boundary Fill:** Fill thin outer edge slivers when handheld burst shots have slight framing drift after homography alignment, instead of cropping in.
  * **Perspective / Keystone Correction & Corner Fill:** Add `Vertical Perspective (-100..+100)` and `Horizontal Perspective (-100..+100)` sliders inside the Crop & Straighten tool, solved via 4-point projective homography (`cv2.getPerspectiveTransform` + `cv2.warpPerspective(..., flags=cv2.INTER_LANCZOS4)`), paired with analytical auto-crop or optional `LaMa` corner-wedge fill.

### 5.2 Archival Restoration Presets in Spot Fix
* **Auto "Remove Orange Date Stamp" Preset:** Automatically detect saturated orange/red LED quartz date-stamp digits in the bottom corners of 1980s–2000s film/compact scans and feed the dilated digit mask into `LaMaInpaintEngine`.
* **Dust & Film Scratch Line Mode:** Hold `Shift` in Spot Fix to click-drag straight scratch lines across scanned slides/negatives and heal them in one stroke.

### 5.3 Creative `.cube` 3D LUT Importer & Look Profiles
* **Look Profiles Dropdown (`As Shot`, `Natural`, `Warm Print`, `Vivid`, `Portrait`, `Monochrome`)** with an importable **`.cube` 3D LUT** parser evaluated via fast trilinear `numpy` lookup and continuous `Amount (0–200%)` intensity blending.

---

## 6. Geolocation, Video, Search & Storage Extensions

### 6.1 Map Polygon Geofencing & KML Track Import
* **KML Track Import:** Extend **Companion Phone GPS Matching** (`GpsMatchDialog`) alongside `.gpx` to also parse Google Timeline `.kml` exports.
* **Map Polygon Geofencing & Geographic Smart Albums:** Draw a lasso or polygon on **World Map View** to filter photos inside custom geographic regions or save the boundary as a Smart Album rule.

### 6.2 Video Frame Capture ("Save Still Photo") & Beat-Synced MP4 Slideshows
* **Video Frame Capture:** Add a **"Save Frame as Photo"** action (e.g. as a clean right-click context menu item on video playback in Single Media Viewer) that exports the current full-resolution decoded frame as `.jpg`/`.png` into the same folder, copies `date_taken` (offset by playback position) + GPS coordinates, and immediately indexes it into the catalog.
* **Beat-Synced Slideshow Music & MP4 Export:** Detect onset peaks in `.mp3`/`.wav` audio tracks via short-time Fourier flux (NumPy FFT, zero heavy audio dependencies) and snap Ken Burns pan/zoom slide transitions to musical bars in 1080p MP4 exports.

### 6.3 Dominant Color Swatch Filtering (`Find` Tab)
* **Dominant Color Swatch Filtering:** Add a 10-swatch clickable color filter bar to the `Find` ribbon tab matching precomputed Tier 1 `media_signatures` CIE LAB color data.

### 6.4 Relocatable Database & Thumbnail Cache
* **Relocatable Cache:** Allow moving `thumbnails.db` (which grows to several GB on 500k libraries) and `catalog.db` to a secondary drive from **File → Options → Database** with a cancellable migration progress dialog.

### 6.5 Visual Calendar Density Heatmap
* **Calendar Heatmap:** Optional GitHub-style Year/Month/Day activity heatmap inside the **Date Taken** navigation panel to spot busy trips and events at a glance.

### 6.6 "On This Day" Memories & Automatic Smart Album Examples *(Low Priority)*
* **Smart Albums View Enhancement:** In the future, optionally surface **"On This Day"** memories and curated automatic Smart Album examples inside the **Smart Albums** view rather than cluttering the main Thumbnail Grid.

---

## 7. Removed / Rethink

The following previously proposed ideas have been **removed or placed on hold for rethinking** to keep Essentials Photo Gallery fast, clean, and easy to understand:

* **4th AI Conceptual Similarity Mode inside `Find Duplicates`:** *Removed.* Putting conceptual/semantic similarity inside Duplicate Finder confuses "exact/burst duplicates to delete" with "similar-looking scenes to explore." Instead, `Find Duplicates` stays focused on fast cached `media_signatures` + sharpness-aware keeper picking, while `Find Similar` (right-click) transparently blends MobileCLIP-S1 semantic embeddings behind the scenes.
* **Manual Stack Creation UI:** *Removed.* Previously prototyped and removed because it added extra dropdown complexity and overlapped confusingly with Albums.
* **Semantic Masking Phase 2 (`MobileSAM` / `EdgeSAM` + `sam3_presets.bin`):** *Removed / Rethink.* Heavy model footprint and complex Lightroom-style masking UX that goes beyond WLPG's clean, approachable editing philosophy.
* **Portable `.epgbackup` Catalog Bundles:** *Removed / Rethink.* High maintenance and path-remapping edge-case complexity relative to standard folder/sidecar backup workflows.
* **Customizable Status Bar / Info Pane Field Reordering:** *Removed / Rethink.* Unnecessary settings bloat; sensible built-in defaults keep the UI consistent and uncluttered.
