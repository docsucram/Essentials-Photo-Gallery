# Essentials Photo Gallery — Future Ideas & Innovation Roadmap (`docs/futureideas.md`)

This document tracks forward-looking enhancements and unimplemented capabilities planned for **Essentials Photo Gallery**. Completed systems—such as the 32-bit perceptual tone engine, U-Point selective control points, Depth & Light bokeh/relighting, ICE-style panorama stitching, Photo Fuse, vector privacy redaction, Duplicate & Burst Finder, Interactive Leaflet World Map & Hierarchical Places Tree, **Static & 17-Dimension Boolean Smart Albums**, **Next-Gen Codecs (`HEIC`/`AVIF`/`JPEG XL`)**, **Offline Text-in-Photo OCR Search (`FTS5` + Tesseract/RapidOCR/WinRT)**, **LaMa ONNX Generative Inpainting** (Spot Fix Object Eraser, 3D Cinematic Motion disocclusion fill, Panorama gap fill, optional Straighten/Crop AI fill), and **Picasa-Style Folder Manager & Private Mode (with PIN/Auto-Lock)**—are implemented in the codebase.

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

## 2. Offline Text-in-Photo OCR Search *(Implemented — Future Extensions)*

### 2.1 SQLite FTS5 Full-Text Search Integration ✅ *(Implemented)*
- Implemented in `src/core/ocr_engine.py`, `src/workers/ocr_worker.py`, `media_ocr_fts`, Smart Album `OCR Text` rule, and Right Info Dock copyable OCR preview.
- **Next Steps:**
  - Multi-language RapidOCR ONNX model pack downloads inside **Options → AI & Downloads**.
  - On-image bounding-box text highlighting / click-to-copy directly in Single Image View.

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

## 5. Archival Restoration, OpenCV 5.0 & ONNXRuntime Opportunities

- **Archival Scan Restoration & 2x Super-Resolution (`Real-ESRGAN` / `GFPGAN` ONNX):**
  Dedicated tool for scanned 1950s–1980s film prints and early-2000s low-res digital captures to remove JPEG block artifacts, scanner dust scratches, and gently restore facial clarity (downloadable via **Options → AI & Downloads**).
- **Large-Mask Generative Inpainting (`LaMa` ONNX) ✅ *(Implemented for Object Eraser, 3D Motion, Panorama, Optional Crop/Straighten)*:**
  - **Next Extensions:** 4-point Perspective / Keystone corner fill, Photo Fuse burst-alignment edge fill, and 1-click **"Remove Date Stamp"** / **"Heal Film Scratch"** archival presets.
- **Learned Panorama Feature Matching (`ALIKED` / `DISK` + `LightGlue` ONNX):**
  Surface the OpenCV 5.0 learned matcher models (`_find_learned_matcher_models` in `src/imaging/create_tools.py`) in **Options → AI & Downloads** to improve panorama alignment on low-texture skies, water, and indoor scenes.

---

## 6. 3D LUT Presets (`.cube`) Importer

- Add a standard `.cube` 3D LUT loader ($17^3$ to $65^3$ lattices) with NumPy trilinear interpolation (`<8 ms` per proxy) alongside the 8 built-in presets in `opencv_engine.py`.

---

## 7. Per-Folder Scan & Privacy Controls ✅ *(Implemented — Future Extensions)*

- Implemented in `FolderManagerDialog` (`Scan Always`, `Scan Once`, `Exclude from catalog`, per-folder `Scan Faces` / `Scan OCR`, `Private Folder` isolation, and optional PBKDF2 PIN + idle auto-lock).
- **Next Steps:** Windows Hello biometric unlock integration on Windows 10/11 and per-folder CLIP semantic indexing toggle.

---

## 8. Social Media Export & Crop Presets

- Quick export workflow tailored for Instagram, Facebook, X, and messaging platforms:
  - Aspect-ratio templates (`1:1`, `4:5`, `9:16`, `16:9`), optional EXIF/GPS metadata stripping for privacy, maximum pixel dimension scaling, and quality/size presets.



---

## 9. Fine Tune & Pipeline Upgrades (Inspired by Greycard)

### 9.1 Pre-Computed Text Embeddings for Zero-Bloat Semantic Masking (`sam3_presets.bin`)
- **Concept:** Enable instant 1-click semantic masking (*Sky, Subject, Person, Face, Skin, Eyes, Lips, Teeth, Hair, Clothes, Background, Building, Vegetation, Water, Ground*) without shipping a massive 500 MB text encoder model.
- **Architecture:**
  - Pre-compute the text prompt embeddings offline once for the 16 standard photographic masking categories and pack them into a tiny (~524 KB) binary asset (`sam3_presets.bin` / `semantic_mask_presets.bin`).
  - At runtime, only the lightweight vision encoder/decoder (e.g., quantized MobileSAM / SAM-3 ONNX) runs against the image, multiplying its image features against the pre-computed `524 KB` preset vectors in `<15 ms` to generate instant, edge-accurate local adjustment masks.
- **Selective / Masked Fine Tune Integration:** Allow Fine Tune sliders (Exposure, Contrast, Saturation, Clarity, Shadows/Highlights) to target an AI-generated *Subject*, *Sky*, or *Background* mask or brush mask.

### 9.2 Embedded JPEG Fast-Path for Camera RAW Files (`.RAF`, `.CR3`, `.NEF`, `.ARW`) ✅ *(Implemented)*
- **Concept:** Achieve instantaneous (`<5 ms`) thumbnail generation and initial Single Image View loading for 40–80 MB camera RAW files (such as Fujifilm X-Trans `.RAF` files) while preserving the camera's built-in Film Simulation preview.
- **Implemented (`src/imaging/image_loader.py`, `src/workers/scanner_thumb_worker.py`, `src/ui/media_viewer.py`):**
  - `decode_raw_embedded_preview()` and `ThumbnailDecodeWorker` extract and decode the camera's embedded JPEG/bitmap stream (with EXIF orientation applied) for instant browsing and culling, deferring heavy `rawpy` demosaicing strictly until active slider editing (`_upgrade_raw_to_full_demosaic_async()`).

### 9.3 AgX-Style Highlight Chroma Roll-Off ("Film Shoulder Desaturation") ✅ *(Implemented)*
- **Concept:** Prevent bright specular highlights (sunset clouds, neon signs, sun-lit skin) from looking artificially clipped or posterized yellow/magenta when boosting exposure or contrast.
- **Implemented (`src/imaging/opencv_engine.py`):**
  - Inside the Unified CIE $L^*a^*b^*$ pass (`need_tone_lab`), couples the Filmic Soft-Knee luminance curve with an AgX-inspired chromaticity roll-off above the highlight shoulder threshold ($L^* > 220.0$ on the $0..255$ scale):
    $$w_{\text{desat}} = \text{clip}\left(\frac{L^* - 220.0}{35.0}, 0.0, 1.0\right)^2$$
    $$A^{*\prime} = 128.0 + (A^* - 128.0) \cdot (1.0 - 0.65 \cdot w_{\text{desat}})$$
    $$B^{*\prime} = 128.0 + (B^* - 128.0) \cdot (1.0 - 0.65 \cdot w_{\text{desat}})$$
  - Gently desaturates extreme highlights toward clean paper-white as $L^* \to 255$, emulating analog color negative film behavior.

### 9.4 Focus-Aware Mouse-Wheel Slider Ergonomics ✅ *(Implemented)*
- **Concept:** Prevent accidental slider changes when scrolling the right-hand editing sidebar, while enabling fast precision mouse-wheel scrubbing when a slider is intentionally targeted.
- **Implemented (`_AeroFineTuneSlider` in `src/ui/right_tag_dock.py`):**
  - **Unfocused Hover:** Uses `Qt.StrongFocus`; unfocused sliders ignore wheel events (`event.ignore()`) so the parent Fine Tune sidebar scrolls normally.
  - **Focused / Active Slider (or `Alt + Wheel`):** Accepts wheel events in clean `±1` (`Shift + Wheel` for `±5`) steps without scrolling the parent sidebar away from the cursor.

### 9.5 Live RGB / Luminance Histogram ✅ *(Implemented — Future: "Subject-Only" Scope Toggle)*
- **Implemented (`HistogramWidget` in `src/ui/right_tag_dock.py`):** Renders translucent `R`, `G`, and `B` channel curves behind the Luminance histogram with interactive Black/White point markers.
- **Next Step:** Add an optional **"Subject-Only"** scope toggle that computes the histogram strictly inside the detected foreground/subject mask so bright skies don't skew skin-tone exposure decisions.
