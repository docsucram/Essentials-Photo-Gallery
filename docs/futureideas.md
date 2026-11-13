# Essentials Photo Gallery — Future Ideas & Innovation Roadmap (`docs/futureideas.md`)

This document tracks forward-looking enhancements and unimplemented capabilities planned for **Essentials Photo Gallery**. Completed systems—such as the 32-bit perceptual tone engine, U-Point selective control points, Depth & Light bokeh/relighting, ICE-style panorama stitching, Photo Fuse, vector privacy redaction, Duplicate & Burst Finder, Interactive Leaflet World Map & Hierarchical Places Tree, **Static & 17-Dimension Boolean Smart Albums**, and **Next-Gen Codecs (`HEIC`/`AVIF`/`JPEG XL`)**—are documented in [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`editing_engine.md`](editing_engine.md).

All future ideas follow the core philosophy of **100% local, private, offline execution** with zero external cloud telemetry and sub-10ms UI query responsiveness at 500,000+ item scale.

---

## 1. Local Semantic & Natural-Language Visual Search

### 1.1 Local Multi-Modal CLIP Embeddings (`MobileCLIP` / `SigLIP2` ONNX)
- **Concept:** Search by conceptual descriptions (e.g., `"red convertible in rain"`, `"family playing board game"`, `"snowy garden sunset"`, `"plate of pasta"`) with zero manual descriptive tags required.
- **Architecture:**
  - Background worker feeds pre-decoded $256 \times 256$ WebP thumbnails from `thumbnails.db` through an ONNX-quantized `MobileCLIP-S1` or `SigLIP2-Base` vision encoder via `onnxruntime-directml` (or multi-threaded CPU).
  - Stores L2-normalized 512-dimensional `float16` vectors (~1 KB per asset) in a memory-mapped binary file (`clip_embeddings.bin`) paired with an SQLite offset lookup table (`media_embeddings(media_id, offset)`).
  - Ranks 400,000+ vectors via single-matrix dot product (`embeddings @ query_vec`) in `<15 ms`.
- **UI Integration:** Mode toggle inside the top-right Search Box (`[AI Semantic Search]`).

### 1.2 "Find Visually Similar Photos" Right-Click Action
- **UX:** Right-click any photo $\to$ *"Find visually similar photos"*.
- Computes cosine similarity against the catalog embedding matrix in `<5 ms` and filters the grid to the top 100 matches ordered by visual affinity.

---

## 2. Offline Text-in-Photo OCR Search

### 2.1 SQLite FTS5 Full-Text Search Integration
- **Concept:** Find photos containing visible text (street signs, receipts, handwritten notes, whiteboards, book covers, certificates, screenshots).
- **Architecture:**
  - Background worker runs Windows native `Windows.Media.Ocr` (built into Windows 10/11) or quantized `RapidOCR` ONNX.
  - Indexes extracted strings in an SQLite virtual FTS5 table:
    ```sql
    CREATE VIRTUAL TABLE media_ocr_fts USING fts5(
        media_id UNINDEXED,
        ocr_text,
        tokenize = 'unicode61 remove_diacritics 2'
    );
    ```
  - Integrated directly into the main Search Box alongside tags and captions.

---

## 3. Geolocation & Mapping Next Steps

### 3.1 GPX / KML Track Log Auto-Geotagging
- **Concept:** Correlate timestamps of photos shot on cameras without GPS against a smartphone or smartwatch GPX/KML track log.
- **Workflow:** Right-click selection $\to$ *"Geotag from GPX Track..."*, adjust camera clock offset if needed, linearly interpolate `(lat, lon, elevation)` between track points, preview pins on the Leaflet map, and commit to `catalog.db` + XMP sidecars.

### 3.2 Map Polygon Geofencing & Geographic Smart Albums
- **Concept:** Interactive polygon / lasso and circular radius selection directly on the World Map View.
- **Pipeline:** Spatial R*Tree bounding-box pre-filter followed by ray-casting point-in-polygon tests, with 1-click export to persistent Smart Album rules.

---

## 4. Memories & Discovery UX

### 4.1 "On This Day" Collapsible Strip Above Grid View
- **Current State:** `"On This Day"` is seeded as a dynamic Smart Album in the left `Albums -> Smart Albums` tree.
- **Next Step:** Optional collapsible horizontal header strip above Grid View highlighting top-rated memories from today's month/day across past years (`1 year ago`, `5 years ago`, `10+ years ago`).

---

## 5. Archival Restoration & Inpainting Upgrades

- **Archival Scan Restoration & 2x Super-Resolution (`Real-ESRGAN` / `GFPGAN` ONNX):**
  Dedicated tool for scanned 1950s–1980s film prints and early-2000s low-res digital captures to remove JPEG block artifacts, scanner dust scratches, and gently restore facial clarity.
- **Large-Mask Generative Inpainting (`LaMa` ONNX):**
  Optional ONNX inpainting backend for erasing large photobombers, power lines, and street clutter where exemplar frequency separation isn't enough.

---

## 6. 3D LUT Presets (`.cube`) Importer

- Add a standard `.cube` 3D LUT loader ($17^3$ to $65^3$ lattices) with NumPy trilinear interpolation (`<8 ms` per proxy) alongside the 8 built-in presets in `opencv_engine.py`.

---

## 7. Per-Folder Scan & Privacy Controls

- Folder-level preferences accessible from the Left Navigation Dock:
  - Mark folders as **Private / Hidden** (excluded from default `"All photos and videos"` unless unlocked).
  - Toggle background passes per folder root (**Face detection**, future **OCR / CLIP indexing**, **Live file watcher**).

---

## 8. Social Media Export & Crop Presets

- Quick export workflow tailored for Instagram, Facebook, X, and messaging platforms:
  - Aspect-ratio templates (`1:1`, `4:5`, `9:16`, `16:9`), optional EXIF/GPS metadata stripping for privacy, maximum pixel dimension scaling, and quality/size presets.



---

## 10. Lessons & Pipeline Upgrades Inspired by Greycard

### 10.1 Pre-Computed Text Embeddings for Zero-Bloat Semantic Masking (`sam3_presets.bin`)
- **Concept:** Enable instant 1-click semantic masking (*Sky, Subject, Person, Face, Skin, Eyes, Lips, Teeth, Hair, Clothes, Background, Building, Vegetation, Water, Ground*) without shipping a massive 500 MB text encoder model.
- **Architecture:**
  - Pre-compute the text prompt embeddings offline once for the 16 standard photographic masking categories and pack them into a tiny (~524 KB) binary asset (`sam3_presets.bin` / `semantic_mask_presets.bin`).
  - At runtime, only the lightweight vision encoder/decoder (e.g., quantized MobileSAM / SAM-3 ONNX) runs against the image, multiplying its image features against the pre-computed `524 KB` preset vectors in `<15 ms` to generate instant, edge-accurate local adjustment masks.

### 10.2 Embedded JPEG Fast-Path for Camera RAW Files (`.RAF`, `.CR3`, `.NEF`, `.ARW`)
- **Concept:** Achieve instantaneous (`<5 ms`) thumbnail generation and initial Single Image View loading for 40–80 MB camera RAW files (such as Fujifilm X-Trans `.RAF` files) while preserving the camera's built-in Film Simulation preview.
- **Architecture:**
  - Every modern camera RAW container embeds a full-resolution (or half-resolution) camera-processed JPEG preview at a deterministic header byte offset with the camera's exact color profile (e.g., Fujifilm *Provia*, *Classic Chrome*, *Velvia*) already applied.
  - `ThumbnailDecodeWorker` and `SingleMediaViewer` extract and decode this embedded JPEG stream directly from the byte range for instant browsing and culling, deferring heavy 16-bit linear RAW demosaicing (`rawpy` / `LibRaw` Markesteijn/AMaZE) strictly until the user enters active slider editing.

### 10.3 AgX-Style Highlight Chroma Roll-Off ("Film Shoulder Desaturation")
- **Concept:** Prevent bright specular highlights (sunset clouds, neon signs, sun-lit skin) from looking artificially clipped or posterized yellow/magenta when boosting exposure or contrast.
- **Formulation (`src/imaging/opencv_engine.py`):**
  - Inside the Unified CIE $L^*a^*b^*$ pass, couple the Filmic Soft-Knee luminance curve with an AgX-inspired chromaticity roll-off above the highlight shoulder threshold ($L^* > 220.0$ on the $0..255$ scale):
    $$w_{\text{desat}} = \text{clip}\left(\frac{L^* - 220.0}{35.0}, 0.0, 1.0\right)^2$$
    $$A^{*\prime} = 128.0 + (A^* - 128.0) \cdot (1.0 - 0.65 \cdot w_{\text{desat}})$$
    $$B^{*\prime} = 128.0 + (B^* - 128.0) \cdot (1.0 - 0.65 \cdot w_{\text{desat}})$$
  - Gently desaturates extreme highlights toward clean paper-white as $L^* \to 255$, emulating analog color negative film behavior.

### 10.4 Focus-Aware Mouse-Wheel Slider Ergonomics
- **Concept:** Prevent accidental slider changes when scrolling the right-hand editing sidebar, while enabling fast precision mouse-wheel scrubbing when a slider is intentionally targeted.
- **UX Rules (`_AeroFineTuneSlider` in `src/ui/right_tag_dock.py`):**
  - **Unfocused Hover:** Scrolling the mouse wheel over the sidebar scrolls the vertical panel normally; unfocused sliders ignore wheel events (`event.ignore()`).
  - **Focused / Active Slider (or `Alt + Wheel`):** Once a slider is clicked/focused (or while holding `Alt`), mouse-wheel ticks increment/decrement the slider in clean `±1` (`Shift + Wheel` for `±5`) steps without scrolling the parent sidebar away from the cursor.

### 10.5 Live RGB / Luminance Histogram with "Subject-Only" Scope Toggle
- **Concept:** Provide
