# Essentials Photo Gallery — Future Ideas & Innovation Roadmap (`futureideas.md`)

This document tracks forward-looking enhancements and next-generation capabilities planned for **Essentials Photo Gallery**. Features that have already been designed, implemented, and verified in the core engine (such as the 32-bit floating-point perceptual tone engine, U-Point selective control points, auto-straighten with inscribed crop, ICE-style panorama stitching, multi-photo collages, Photo Fuse, portrait bokeh, Color Pop, 3D motion synthesis, vector privacy redaction, the **Perceptual Duplicate & Burst Finder**, and the **Interactive Leaflet World Map & Hierarchical Places Tree**) are documented in [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`editing_engine.md`](editing_engine.md).

All future ideas follow the core design philosophy of **Essentials Photo Gallery**: **100% local, private, and offline execution** with zero external cloud dependencies or telemetry, maintaining sub-10ms UI query fluidity at 500,000+ photo scale.

---

## 1. Local Semantic & Natural-Language Visual Search

### 1.1 Local Multi-Modal CLIP Embeddings (MobileCLIP / SigLIP2 ONNX)
- **Concept:** Search by conceptual descriptions (e.g., `"red convertible in rain"`, `"family playing board game"`, `"snowy garden sunset"`, `"plate of pasta"`) with zero manual descriptive tags required.
- **Architecture:**
  - Background worker takes pre-decoded $256 \times 256$ WebP thumbnails from `thumbnails.db` and passes them through an ONNX-quantized `MobileCLIP-S1` or `SigLIP2-Base` vision encoder running via `onnxruntime-directml` (or multi-threaded CPU).
  - Generates an L2-normalized 512-dimensional `float16` vector per photo (~1 KB storage per asset = ~400 MB total for 400,000 items).
  - Stored in a memory-mapped binary file (`clip_embeddings.bin`) paired with an SQLite lookup index (`media_embeddings(media_id, offset)`).
  - Text search queries run through the MobileCLIP text encoder in `<10 ms`. A single NumPy/BLAS dot product (`embeddings @ query_vec`) ranks all 400,000 photos in `<15 ms`.
- **UI Integration:**
  - Integrated into the existing top-right Search Box with an indicator toggle (`[🔍 AI Semantic Search]`).

### 1.2 "Find Visually Similar Photos" Right-Click Action
- **UX:** Right-click any photo $\to$ *"Find visually similar photos"*.
- **Math:** Computes cosine similarity between the selected photo's 512-d CLIP vector and all catalog vectors in `<5 ms`.
- **Results:** Instantly filters the Grid View to the top 100 matching photos, ordered by visual affinity (useful for finding alternative angles, related photo sets, or matching artistic styles).

---

## 2. Offline Text-in-Photo OCR Search

### 2.1 SQLite FTS5 Full-Text Search Integration
- **Concept:** Instantly find photos containing visible text: street signs, receipts, handwritten notes, whiteboards, book covers, name badges, certificates, wine labels, and screenshots.
- **Architecture:**
  - Background worker utilizes Windows native `Windows.Media.Ocr` (built into Windows 10/11 with zero extra DLL footprint) or quantized `RapidOCR` ONNX.
  - Extracts bounding boxes and text strings; writes into an SQLite virtual table:
    ```sql
    CREATE VIRTUAL TABLE media_ocr_fts USING fts5(
        media_id UNINDEXED,
        ocr_text,
        tokenize = 'unicode61 remove_diacritics 2'
    );
    ```
  - Sub-millisecond keyword lookup natively supported in the main Search Box alongside tag and caption filtering.

---

## 3. Geolocation, Places & Mapping (As-Built & Next Steps)

### 3.1 Hierarchical Offline Reverse Geocoding & Map View [IMPLEMENTED]
- **Status:** **Completed & Operational** (documented in [`ARCHITECTURE.md`](ARCHITECTURE.md#33-center-area-qstackedwidget)).
- **Implemented Capabilities:**
  - **Comprehensive Offline Gazetteer (`src/core/gazetteer_data.py`):** 3,500+ world cities and detailed UK settlements built-in with sub-2ms prefix/substring search and KD-Tree reverse geocoding via `reverse_geocoder` / `scipy.spatial.cKDTree` or `numpy`.
  - **Online Nominatim Fallback & SQLite Caching:** Automatic resolution for remote areas with persistent caching in `saved_places` to minimize network calls.
  - **Collapsible Places Hierarchy in Left Navigation Dock:** Automatically populates `Country -> State / County -> City / Town` (e.g. `United Kingdom -> Surrey -> Claygate`) with live item counts and instant grid filtering.
  - **Interactive Leaflet World Map View (`src/ui/map_view.py`):** Hardware-accelerated interactive map with persistent disk tile caching (Clean Voyager, Street, Satellite, Terrain), SQLite R*Tree spatial query (`media_rtree`), instant cluster/pin filtering, and offline vector map canvas fallback.
  - **Ribbon & Dock Geotagging Controls:** Instant geotagging dialog, right-pane geotag editing with hover clear 'x', coordinate format parsing, and bidirectional XMP/EXIF synchronization.

### 3.2 GPX Track Log Auto-Geotagging (Future Idea)
- **Concept:** Automatically geotag photos captured on cameras without built-in GPS by correlating photo timestamps with a recorded GPX or KML track log (from a smartphone, smartwatch, or GPS logger).
- **Workflow:**
  - Select a set of photos $\to$ Right-click $\to$ *"Geotag from GPX Track..."*.
  - User chooses a `.gpx` file and sets a camera clock time-offset calibration if the camera clock differed from satellite time.
  - Linearly interpolates latitude, longitude, and elevation between adjacent track points based on UTC `date_taken` timestamps.
  - Previews matched points on the Leaflet map with confidence indicators before committing to database and XMP sidecars.

### 3.3 Map Polygon Geofencing & Geographic Smart Albums (Future Idea)
- **Concept:** Interactive polygon / lasso and circular radius selection tools directly on the World Map View.
- **Features:**
  - Draw a custom boundary (e.g., surrounding a national park, holiday resort, or city neighborhood).
  - Filter grid photos falling within the polygon via SQLite spatial R*Tree bounding box filter followed by ray-casting point-in-polygon tests.
  - Save geographic boundaries as persistent Smart Albums in the Left Navigation Dock.

---

## 4. Dynamic Library Curation & Smart Collections

### 4.1 Smart Albums (Saved Dynamic Boolean Queries)
- **Concept:** Allow users to save any complex combination of filters as a persistent Smart Album in the Left Navigation Dock:
  - Example: `(Person = "Emma" OR Person = "Craig") AND Rating >= 4 AND Year = 2024 AND Flagged = 1`.
- **Implementation:**
  - Stored in a lightweight SQLite table `smart_albums(id, name, query_spec_json)`.
  - Smart Albums evaluate dynamically upon selection, ensuring newly imported or tagged photos appear immediately without manual list maintenance.

### 4.2 "On This Day" / Rediscover Memories Bar
- **Concept:** An optional collapsible header strip above Grid View highlighting favorite memories from today's calendar date across past years (`1 year ago`, `5 years ago`, `10 years ago`, `25+ years ago`).
- **Algorithm:**
  - Efficiently queries `date_taken` where `strftime('%m-%d', date_taken, 'unixepoch') = strftime('%m-%d', 'now') AND rating >= 3`.
  - Presents a curated horizontal carousel with direct click-to-view.

---

## 5. Local Neural Inpainting & Generative Power Tools

### 5.1 Quantized LaMa Inpainting ONNX (~45 MB)
- **Current State:** Retouch and healing currently use Navier-Stokes / Telea fast-marching inpainting with multi-radius Poisson cloning.
- **Future Enhancement:**
  - Add an optional quantized LaMa (Large Mask Inpainting) ONNX model for large brush strokes.
  - Erases complex power lines, photobombers, trash bins, and street furniture with texture-aware synthesis in `<250 ms` on DirectML GPU.

### 5.2 1-Click Subject Cutout / Sticker Generation (RMBG-1.4 / ModNet ONNX)
- **Current State:** GrabCut and guided-filter matting are implemented in `creative_effects.py` for Color Pop and Portrait Bokeh.
- **Future Enhancement:**
  - Add single-click subject background removal exporting clean transparent PNGs or copying transparent bitmaps straight to the Windows clipboard for instant pasting into presentations and documents.

### 5.3 Archival Scan Restoration & 2x Super-Resolution (Real-ESRGAN / GFPGAN ONNX)
- **Concept:** Dedicated *"Restore Archival Scan"* tool for early-2000s low-resolution digital captures or scanned 1950s–1980s family film prints.
- **Pipeline:**
  - Eliminates JPEG compression block artifacts and print scanner dust scratches.
  - Gently enhances facial clarity using lightweight GFPGAN face restoration.

---

## 6. 3D LUT Presets (.cube) Importer

### 6.1 Custom Film Emulation & Professional LUT Engine
- **Current State:** 8 calibrated photographic profiles (Classic Chrome, Warm Portrait, Vivid Landscape, Matte Vintage, Fine Art B&W, Cool Cinema, Sepia, None) are built into `opencv_engine.py` with 0..100% intensity blending.
- **Future Enhancement:**
  - Add a `.cube` file loader supporting industry-standard 3D LUT files ($17 \times 17 \times 17$ up to $65 \times 65 \times 65$).
  - Trilinear interpolation in NumPy/C++ executing in `<8 ms` on preview proxies.
  - Allow users to import LUT collections (e.g., Kodak Portra, CineStill, Kodak Kodachrome, Fuji Velvia) and adjust blend intensity.

---

## 7. Next-Generation Image Formats & Lightweight Codec Additions [IMPLEMENTED]

### 7.1 HEIC / HEIF & AVIF Support (`pillow-heif`, `pillow-avif-plugin`) [IMPLEMENTED]
- **Status:** **Completed & Operational**
- **Implemented Capabilities:**
  - Integrated `pillow-heif` and `pillow-avif-plugin` via centralized `register_nextgen_codecs()` in `src/imaging/image_loader.py` and `main.py`.
  - Enables native 8-bit and 10-bit **HEIC, HEIF, HIF, and AVIF** decoding for both background thumbnail generation (`_decode_nextgen_thumbnail()` in `scanner_thumb_worker.py`) and full-resolution Single Image View (`media_viewer.py` and `image_loader.py`).
  - Automatically converts embedded Display P3 ICC color profiles (standard on Apple iPhone captures) and wide-gamut profiles to calibrated sRGB via LittleCMS2 (`convert_icc_to_srgb()`).
  - Full transparency / alpha preservation (`ALPHA_SAVE_DOT_EXT_SET`) and EXIF / XMP metadata extraction.
  - Export & Save as Copy support with configurable quality and metadata preservation in `SingleMediaViewer`.
  - **PyInstaller Packaging:** Includes dynamic libraries, data files, and hidden imports for `pillow_heif` and `pillow_avif` in `photogallery.spec`.

### 7.2 JPEG XL (`.jxl`) Support (`pillow-jxl-plugin`) [IMPLEMENTED]
- **Status:** **Completed & Operational**
- **Implemented Capabilities:**
  - Integrated `pillow-jxl-plugin` (wrapping `libjxl`) with automatic Pillow plugin registration in `register_nextgen_codecs()`.
  - Added `.jxl` across all scanner extension sets (`NEXTGEN_EXT_SET`, `IMAGE_EXT_SET`, `ALL_MEDIA_EXT_SET`), folder watchers, import dialogs, and catalog grouping (`JPEG XL Image (*.jxl)` in `repository.py`).
  - Full alpha transparency support, bit-depth normalization, and LittleCMS2 color management.
  - High-efficiency `.jxl` export in **Save as Copy** and **Save Adjustments to Disk** with serialized EXIF preservation.
  - **PyInstaller Packaging:** Includes dynamic libraries, data files, and hidden imports for `pillow_jxl` in `photogallery.spec`.

