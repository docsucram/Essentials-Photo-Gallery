# Future Ideas & Architectural Roadmap

This document tracks **genuinely unbuilt future enhancements** for **Essentials Photo Gallery**, maintaining the classic Windows Live Photo Gallery (WLPG) ease-of-use and speed while expanding optional local AI, heritage migration, and enthusiast editing capabilities.

> **Graduated Features:** Completed initiatives—including **v0.17.0 Pre-Release Roadmap (Thumbnail-Time Visual Signatures `media_signatures` + Undoable Hardened Auto-Stacking, 4-Mode Photo Layout Engine `Grid`/`Tiles`/`Mosaic`/`Masonry`, Optional `Cameras` & `Subjects` Nav Sections, `F1` Keyboard Shortcuts Modal + `Alt+1..9` Quick-Tag Hotkeys, and Optional `MobileCLIP-S1` INT8 ONNX `SemanticVectorRepository` + FTS5 RRF Search)**, **v0.16.0 World Map View Speed Overhaul (Bundled Leaflet + `gallerythumb://` + Web Worker Supercluster + Map Scope + 3s Pre-Warm), P2 Grid/RAW/Background Perf (Two-Tier Cache, Batched Commits, Tier-0 RAW Embedded JPEG, `ScanCrashTrace`, Panorama-Aware HiDPI Thumbnails + Rebuild Cache), Photographer Culling Toolkit (Zoom Minimap, Batch Rename `F2`, Borderless Fullscreen Spacebar Quick Preview + `Shift+Rating/Flag/Fav` Auto-Advance, Camera AF Area Overlay, Animated Filmstrip with Attached Pull-Tab), Optional Photo Favourites (`is_favourite`) & Tri-State Pick/Reject Flags + Permanent Built-In Flagged Smart Albums + Group by Favourites, Auto-Detecting Google Picasa `.picasa.ini` & `contacts.xml` Heritage Importer, `Lap`-Style Empty-Library Welcome Canvas, Apple Live Photos & Android Motion Photos, Offline Drive/NAS Browsing (`volume_offline`) & Duplicate Preferred Folder Priority, and Thumbnail Grid Image Stacking (`stack_parent_id`) + RAW/JPEG Pairs**, **Depth & Light 2.0 (3D Surface-Normal Relighting & Virtual Light Studio with Joint-Guided Depth Refinement, Hemispherical Volume Normals, Albedo-Preserving Ambient Dimming, 3 Interactive On-Canvas Point/Spot/Rim Lights, Blinn-Phong Gloss, Gobo Projections & 6 Relight Look Presets)**, **UI Theme Architecture Consolidation (`ThemeTokens`, semantic `uiRole` QSS properties, and automated headless contrast auditing)**, **Visual Similarity Search (`Find Similar` with 24-byte LAB Haar + pHash/dHash, `81%` default & `70%–98%` Scope Bar slider)**, **Database Fresh Start Reset & Empty-Library Prompt**, **Session-Cached `FaceEngine` & Folder Manager Performance Optimizations**, **Companion Phone GPS Matching & GPX Interpolation**, **Fine Tune Two-Tier `Basic | Advanced` Mode** (`Smart Enhance`, `Whites`/`Blacks`, content-aware `Structure`, `Tone Curve`, `WB Eyedropper`, `8-Band Color Mixer (HSL)`, `Sharpen Radius/Masking`, `Split Denoise`, `Orton Glow`/`Golden Hour`, `Film Grain`), **Phase 1 Regional Masking (`Whole | Subject | Bg | Sky`) with Interactive Paint/Erase Mask Brush & Live Masked Histogram**, **Home Ribbon `People` & `Geotag` Split Buttons**, SQLite FTS5 RapidOCR search, Per-Folder Scan & Privacy Controls with PBKDF2 PIN lock, Unified `LaMa` ONNX Generative Inpainting, Embedded JPEG Fast-Path for Camera RAW, AgX Highlight Chroma Roll-Off, Focus-Aware Sliders, and Interactive Map Pin Adjustment—have been graduated into [`docs/ARCHITECTURE.md`](ARCHITECTURE.md), [`RELEASES.md`](../RELEASES.md), and [`docs/editing_engine.md`](editing_engine.md).

---

## 1. Contextual Markup & Social Post Design Studio ("Canva-Lite" Workflow)

### 1.1 Promote Markup to a Dedicated Contextual `Markup & Design` Ribbon Tab
* **User Value:** Replace the cramped floating Markup toolbar with a full contextual **`Markup`** Ribbon tab (whilst keeping **`Spot Fix`** strictly focused on `Blemish` | `Object Eraser` | `Red Eye` healing), unlocking rich Canva-style vector graphics, typography, and social media framing right inside Single Image View.
* **Architecture:**
  * **Contextual `Markup` Ribbon Tab Groups:**
    1. **Insert & Tools:** `Select / Transform (Move, Resize, Rotate)`, `Text Box / Headline`, `Speech / Thought Callout`, `Arrow / Line`, `Shapes (Rectangle, Rounded Pill, Circle, Star, Badge, Polygon)`, `Sticker / Emoji / Watermark Image`, `Freehand Pen`, `Highlighter`, `Magnifier Loupe`, `Privacy Blur / Pixelate`.
    2. **Shape & Border Style:** `Fill Colour` (Solid, Semi-Transparent Frosted Glass, Gradient, or Transparent), `Border / Stroke Colour`, `Border Thickness (0–24px)`, `Dash Style`, `Corner Rounding`, `Drop Shadow (Soft / Retro Hard)`.
    3. **Typography:** `Font Family` (System + bundled clean display fonts), `Font Size`, `Bold / Italic / Uppercase`, `Text Colour`, `Text Outline / Shadow`, `Background Pill Toggle`, `Layer Order (Bring Forward / Send Backward)`.
    4. **Social Canvas & Aspect Ratio Pad (`Frame for Social`):** One-click aspect-ratio canvas presets that pad or frame the photo without destructive cropping:
       * `Instagram Post (4:5 Portrait — 1080×1350)`
       * `Instagram Square (1:1 — 1080×1080)`
       * `Instagram Story / Reel (9:16 — 1080×1920)`
       * `Facebook / X Landscape (16:9 / 1.91:1)`
       * **Pad Background Styles:** *Blurred Photo Backdrop*, *Dominant Palette Colour*, *Clean White Gallery Mat (with optional EXIF camera/lens caption stamp)*, or *Dark Charcoal*.
* **Quick Social Export & Carousel Slicer:**
  * **`Copy Social Post to Clipboard` (`Ctrl+Shift+C`) + Quick Launch** to paste straight into Instagram, Facebook, WhatsApp Web, or Discord.
  * **Seamless Panorama Carousel Slicer:** Slice a wide panorama or collage into 2–5 swipeable `1080×1350` (`4:5`) Instagram carousel frames automatically.
    * Auto-stacking helpers: **`Auto-Stack RAW + JPEG Pairs`** and **`Auto-Stack Rapid Bursts (< 2s)`** (picking the highest Laplacian sharpness shot as the stack parent).
    * **Unified RAW + JPEG/HEIC Lifecycle & Preview Controls:** Keep paired RAW + JPEG/HEIC files together during move, rename, copy, and Duplicate Finder scans; prompt with a deletion-scope choice (*Delete Both / RAW Only / JPEG Only*); and add a configurable RAW display preference (*Embedded Preview* vs. *Full Demosaic* vs. *Paired JPEG*).

---

## 4. Semantic Auto-Tagging, Multi-Language OCR & Redesigned Duplicates Workspace

### 4.1 Extensions to the `MobileCLIP-S1` Add-on & Contextual `Duplicates` Ribbon
* **Expanding on v0.17.0's `MobileCLIP-S1` Search & Subjects + `media_signatures` Foundation:**
  * **Verified Integrity & Multilingual CLIP Pack:** Add `sha256sums.txt` integrity checks across optional ONNX add-on downloads in `File → Options → Add-ons`, plus an optional multilingual CLIP text encoder pack.
  * **Zero-Shot Auto-Tagging (`concept_presets.bin`):** Expand beyond the 12 built-in Left Nav **Subjects** with `~120` pre-computed photographic categories driving optional **Suggested Smart Tags** (analogous to face suggestions).
  * **4th AI Conceptual Similarity Mode inside `Duplicates & Similar` Ribbon Tab:** Expose semantic similarity inside the contextual Duplicates workspace alongside *Exact Copies*, *Visual & Bursts*, and *Color / Haar Similar*, with a non-destructive **"Merge Tags, People, Caption & Rating into Kept Photo"** action on recycle.

### 4.2 Multi-Language OCR & Interactive Text Selection
* **Multi-Language Recognition Packs:** Optional CJK / Latin / Cyrillic RapidOCR packs in **File → Options → Add-ons**.
* **Interactive Text Selection in Single Image View:** Optional overlay button when viewing screenshots or documents that highlights detected text bounding boxes for direct click-to-copy on the image canvas.

### 4.3 Automatic Trip & Event Clustering
* Combine temporal gap detection ($\Delta t > 6\text{ hours}$ or multi-day travel away from the user's home coordinate centroid) with `location_city` / `location_country` transitions to power an **`Arrange by: Events / Trips`** mode and one-click **"Save Trip as Album"**.

---

## 5. Ribbon KeyTips

### 5.1 Microsoft Ribbon `Alt` KeyTips
* *(Unified `F1` Keyboard Shortcuts modal and configurable `Alt+1..9` Quick-Tag hotkeys shipped in v0.17.0.)*
* **Microsoft Ribbon `Alt` KeyTips:** Pressing `Alt` overlays classic keyboard navigation badges (`H` Home, `E` Edit, `F` Find, `C` Create, `V` View) and two-stroke command activation across the Ribbon.

---

## 6. Additional Editing, Generative Restoration & Semantic Masking

### 6.1 `LaMa` Context-Aware Fill Remaining Extensions (`Photo Fuse` & `Perspective/Keystone`)
* Extend the implemented `LaMaInpaintEngine` (`lama_fp32.onnx`) to the remaining geometric tools:
  * **Photo Fuse Boundary Fill:** Fill thin outer edge slivers when handheld burst shots have slight framing drift after homography alignment, instead of cropping in.
  * **Perspective / Keystone Correction & Corner Fill:** Add `Vertical Perspective (-100..+100)` and `Horizontal Perspective (-100..+100)` sliders inside the Crop & Straighten tool, solved via 4-point projective homography (`cv2.getPerspectiveTransform` + `cv2.warpPerspective(..., flags=cv2.INTER_LANCZOS4)`), paired with analytical auto-crop or optional `LaMa` corner-wedge fill.

### 6.2 Archival Restoration Presets in Spot Fix
* **Auto "Remove Orange Date Stamp" Preset:** Automatically detect saturated orange/red LED quartz date-stamp digits in the bottom corners of 1980s–2000s film/compact scans and feed the dilated digit mask into `LaMaInpaintEngine`.
* **Dust & Film Scratch Line Mode:** Hold `Shift` in Spot Fix to click-drag straight scratch lines across scanned slides/negatives and heal them in one stroke.

### 6.3 Creative `.cube` 3D LUT Importer & Look Profiles
* **Look Profiles Dropdown (`As Shot`, `Natural`, `Warm Print`, `Vivid`, `Portrait`, `Monochrome`)** with an importable **`.cube` 3D LUT** parser evaluated via fast trilinear `numpy` lookup and continuous `Amount (0–200%)` intensity blending.

### 6.4 Semantic Masking Phase 2 (`MobileSAM` / `EdgeSAM` + Pre-Computed `sam3_presets.bin`)
* **Optional Add-on in `File → Options → Add-ons`:** Adds `mobile_sam_encoder_int8.onnx` (~18 MB) + `sam_mask_decoder.onnx` (~16 MB) extending **Fine Tune Regional Masking** (`Whole | Subject | Bg | Sky` + Mask Brush) with interactive **Click / Box-to-Select Any Object**, cached per `media_id`.
* **Pre-Computed Text Embeddings (`sam3_presets.bin` / `semantic_mask_presets.bin`, ~524 KB):** Ship pre-computed text embeddings for ~150 common masking categories (*Foliage, Water, Architecture, Road, Skin, Pet, Vehicle*) so instant 1-click semantic masks run without bundling a heavy text encoder.
* **Live Histogram "Subject-Only" Scope Toggle:** Compute the live histogram strictly inside the detected foreground/subject mask so bright skies don't skew skin-tone exposure decisions.

---

## 7. Geolocation, Memories, Video & Library Portability Extensions

### 7.1 Map Polygon Geofencing, KML Track Import & Memories Strip
* **KML Track Import:** Extend **Companion Phone GPS Matching** (`GpsMatchDialog`) alongside `.gpx` to also parse Google Timeline `.kml` exports.
* **Map Polygon Geofencing & Geographic Smart Albums:** Draw a lasso or polygon on **World Map View** to filter photos inside custom geographic regions or save the boundary as a Smart Album rule.
* **"On This Day" Collapsible Memories Strip:** Optional collapsible horizontal strip above the Thumbnail Grid highlighting photos from 1, 2, 5, and 10 years ago today.

### 7.2 Video Frame Capture ("Save Still Photo") & Beat-Synced MP4 Slideshows
* **Video Frame Capture:** Add a **"Save Frame as Photo"** button in the Single Media Viewer video transport bar that exports the current full-resolution decoded frame as `.jpg`/`.png` into the same folder, copies `date_taken` (offset by playback position) + GPS coordinates, and immediately indexes it into the catalog.
* **Beat-Synced Slideshow Music & MP4 Export:** Detect onset peaks in `.mp3`/`.wav` audio tracks via short-time Fourier flux (NumPy FFT, zero heavy audio dependencies) and snap Ken Burns pan/zoom slide transitions to musical bars in 1080p MP4 exports.

### 7.3 Dominant Color Swatch Filtering (`Find` Tab) & Portable Catalog Bundles (`.epgbackup`)
* **Dominant Color Swatch Filtering:** Add a 10-swatch clickable color filter bar to the `Find` ribbon tab matching precomputed Tier 1 `media_signatures` CIE LAB color data.
* **Multi-PC Library Portability (`.epgbackup`):** Extend **File → Options → Database** with **"Export / Import Portable Catalog Bundle..."** packaging `catalog.db`, `thumbnails.db`, and relative root-folder path mappings so drive-letter differences (`D:\Pictures` vs. `C:\Users\...\Pictures`) remap automatically on the target PC.

---

## 8. Photographer Culling Toolkit, Motion Photos, Offline Drives & Customization

### 8.1 Offline External Drive / NAS Browsing & Relocatable Cache
* **Offline Volume Browsing (`volume_offline` vs. Deleted):** Distinguish unplugged USB drives and offline NAS mounts from deleted files so catalogued folders remain browsable via cached `thumbnails.db` thumbnails with an **`Offline`** badge, disabling only actions requiring original disk bytes and recovering automatically when the drive reconnects.
* **Relocatable Database & Thumbnail Cache:** Allow moving `thumbnails.db` (which grows to several GB on 500k libraries) and `catalog.db` to a secondary drive from **File → Options → Database** with a cancellable migration progress dialog.

### 8.2 Apple Live Photos & Android Motion Photos
* **Unified Motion Photo Support:** Detect Android/Samsung/Pixel Motion Photos via a fast 2-byte JPEG EOI tail check (`FF D9`) + XMP/MP4 `ftyp` offset scan, and pair iPhone `.HEIC`/`.JPG` + `.MOV` Live Photos into a single `[LIVE]` card with smooth in-viewer playback and unified Move/Rename/Delete lifecycle.

### 8.3 Pro Culling Ergonomics (AF Overlay, Zoom Minimap, Batch Rename & Space QuickLook)
* **Camera Autofocus (AF) Area Overlay:** On-demand MakerNote AF point parser (cached per `(path, mtime, size)`, never run during folder scans) displaying locked/selected autofocus boxes in Single Image View and Compare Mode.
* **Zoom Minimap Navigator:** When zoomed >100% in Single Image View or Compare Mode, display a corner minimap with a draggable viewport box. *(Sharpness heatmap overlay removed in v0.16 in favour of clean viewport navigation.)*
* **Batch Rename with Live Preview & Sidecar Sync (`F2` on Multi-Selection):** Token/sequence renaming with collision detection, Windows reserved-name validation (`CON`/`PRN`/`AUX`/`NUL`/`COM1-9`/`LPT1-9`), atomic rollback, and automatic renaming of `.xmp` sidecars, `.OriginalFiles` backups, and RAW+JPEG companions.
* **Spacebar Quick Preview & Global Auto-Advance:** Instant `Space` QuickLook preview in Thumbnail Grid View, plus `Shift + 1..5` / `Shift + P` auto-advance culling outside Compare Mode.

### 8.4 Duplicate Finder Keep Rules, Calendar Heatmap & Custom Fields
* **Preferred Folders & Keep Rules:** Extend **Duplicate Finder** Auto-Pick with an ordered **Preferred Folder Priority** list, RAW+JPEG/Live Photo pair protection, and a total reclaimable-disk-space summary banner.
* **Visual Calendar Density Heatmap:** Optional GitHub-style Year/Month/Day activity heatmap inside the **Date Taken** navigation panel to spot busy trips and events at a glance.
* **Customizable Status Bar & Info Pane Fields:** Right-click dialogs to show, hide, and reorder displayed metadata fields (dimensions, file size, lens, exposure, color space, path, GPS) with optional automatic hiding of empty fields.
