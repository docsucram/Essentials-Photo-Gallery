# Scale Benchmark Results: Before vs. After Optimization (100,000 & 500,000 Items)

**Benchmark Script:** [`tests/benchmark_scale.py`](../tests/benchmark_scale.py)  
**Baseline JSON Data:** [`tests/benchmark_results_100k_500k.json`](../tests/benchmark_results_100k_500k.json)  
**Post-Fix JSON Data:** [`tests/benchmark_results_100k_500k_after.json`](../tests/benchmark_results_100k_500k_after.json)
*Measured: 2026-10-27*

---

## 1. Executive Summary: Verified Before vs. After Improvements

All bottlenecks identified in the 100,000- and 500,000-item scale benchmark have been resolved and verified:

- **Thumbnail Arrival UI Stall Eliminated (`408×` to `4,500×` faster):**
  - **100,000 items:** `1× notify_pixmap_ready()` dropped from **`940.3 ms` -> `2.30 ms`** (`40×` viewport dropped from **`37.6 s` -> `92.1 ms`**).
  - **500,000 items:** `1× notify_pixmap_ready()` dropped from **`4,589.3 ms` -> `1.02 ms`** (`40×` viewport dropped from **`183.6 s` -> `40.8 ms`**).
- **Zero Scroll Hitches at 100k & 500k Items (`60+ FPS` sustained):**
  - Scrolling while thumbnails decode and arrive dropped from **`953.6 ms` max frame (4 hitches) -> `10.49 ms` max frame (0 hitches)** at 100k, and from **`4,448.7 ms` max frame -> `13.89 ms` max frame (0 hitches)** at 500k.
  - Contended scrolling with active background workers dropped from **`111.2 ms` -> `14.40 ms` (0 hitches)** at 100k, and from **`300.9 ms` -> `19.49 ms` (0 hitches >50ms)** at 500k.
- **Catalog Load & Layout (`set_records` + `query_media_records` + Mosaic):**
  - `set_records() + endResetModel()` at 500k dropped from **`4,541.4 ms` -> `25.7 ms` (`176× faster`)**.
  - `query_media_records()` dropped from **`551.1 ms` -> `212.9 ms` (`2.6× faster`)** at 100k, and **`3,712.0 ms` -> `1,123.3 ms` (`3.3× faster`)** at 500k.
  - `compute_mosaic_geometry()` dropped from **`185.6 ms` -> `45.7 ms` (`4.1× faster`)** at 100k, and **`1,267.9 ms` -> `243.4 ms` (`5.2× faster`)** at 500k.
- **Cold Viewport First-Thumbnail Latency Cut in Half (`~2.2× faster`):**
  - Emitting `thumbnail_ready` immediately after JPEG decode (before WebP encode & DB commit) + keeping worker SQLite connections alive cut cold first-thumbnail latency from **`46.0 ms` -> `28.4 ms`** at 100k and **`176.9 ms` -> `78.1 ms`** at 500k (with zero UI freeze afterward).

---

## 2. Before vs. After Benchmark Tables (100,000 & 500,000 Items)

### A. Model & Qt `QListView` Virtualization Layer

| Metric | 100k Before | 100k After | 500k Before | 500k After | Speedup |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`set_records()` + Qt `endResetModel()`** | `1,510.4 ms` | **`557.8 ms`** | `4,541.4 ms` | **`25.7 ms`** | **2.7× – 176.7× faster** |
| **1× `notify_pixmap_ready()` UI Cost** | `940.3 ms` | **`2.30 ms`** | `4,589.3 ms` | **`1.02 ms`** | **408× – 4,500× faster** |
| **40× `notify_pixmap_ready()` (1 Viewport)** | `37,611.8 ms` | **`92.1 ms`** | `183,571.0 ms` | **`40.8 ms`** | **408× – 4,500× faster** |
| **Mosaic Layout `compute_mosaic_geometry()`** | `185.6 ms` | **`45.7 ms`** | `1,267.9 ms` | **`243.4 ms`** | **4.1× – 5.2× faster** |
| **Lazy `_ensure_id_index()` dict build** | `6.17 ms` | **`5.83 ms`** | `34.74 ms` | **`33.17 ms`** | $O(1)$ amortized lookup |

---

### B. Continuous Scroll Frame Times (200 Steps Across Entire Catalog)

| Scroll Scenario | 100k Before (p95 / Max / Hitches) | 100k After (p95 / Max / Hitches) | 500k Before (p95 / Max / Hitches) | 500k After (p95 / Max / Hitches) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Empty Cache Scroll** | `10.25 / 12.03 ms` (0) | **`9.88 / 11.49 ms` (0)** | `10.44 / 12.78 ms` (0) | **`10.30 / 12.66 ms` (0)** |
| **2. Warm `256×256` Pixmap Scroll** | `14.26 / 16.28 ms` (0, 2 >16.6ms) | **`11.08 / 13.01 ms` (0, 0 >16.6ms)** | `14.74 / 17.44 ms` (0, 4 >16.6ms) | **`11.20 / 15.52 ms` (0, 0 >16.6ms)** |
| **3. Contended Scroll (BG Workers)** | `13.29 / 111.23 ms` (**1 hitch**) | **`11.15 / 14.40 ms` (0 hitches)** | `14.90 / 300.90 ms` (**1 hitch**) | **`11.58 / 19.49 ms` (0 hitches)** |
| **4. Scroll + Thumbnail Arrivals** | `907.7 / 953.6 ms` (**4 hitches**) | **`8.82 / 10.49 ms` (0 hitches)** | `4,399 / 4,448 ms` (**4 hitches**) | **`11.56 / 13.89 ms` (0 hitches)** |

---

### C. SQLite Query & Navigation Refresh Latencies (`CatalogRepository`)

| SQL Operation | 100k Before | 100k After | 500k Before | 500k After | Speedup |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`query_media_records` (All Items, Date Sort)** | `551.1 ms` | **`212.9 ms`** | `3,712.0 ms` | **`1,123.3 ms`** | **2.6× – 3.3× faster** |
| **`query_media_records` (Folder + 100 Subfolders)** | `503.5 ms` | **`227.5 ms`** | `3,957.5 ms` | **`1,516.7 ms`** | **2.2× – 2.6× faster** |
| **`get_date_tree_counts()` (Nav Tree Refresh)** | `64.7 ms` | **`45.8 ms`** | `281.5 ms` | **`74.1 ms`** | **1.4× – 3.8× faster** |
| **`get_rating_counts()` (Nav Tree Refresh)** | `6.4 ms` | **`5.9 ms`** | `47.3 ms` | **`46.4 ms`** | Raw tuple cursor |
| **Total `refresh_navigation_and_autocomplete` SQL** | `71.3 ms` | **`51.9 ms`** | `329.1 ms` | **`120.7 ms`** | **1.4× – 2.7× faster** |
| **Scroll Path Lookup (`_ensure_media_path_lookup`)** | `0.18 ms` (SQL) | **`0.00 ms` (In-Memory)** | `0.24 ms` (SQL) | **`0.00 ms` (In-Memory)** | **Zero main-thread SQL** |

---

### D. Thumbnail Decode & Cache Pipeline (40 Visible Items = 1 Full Viewport)

| Thumbnail Pipeline Metric | 100k Before | 100k After | 500k Before | 500k After | Improvement |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **40× Worker SQLite Reads** | `106.54 ms` (close/reopen) | **`0.26 ms` (persistent)** | `112.14 ms` (close/reopen) | **`0.25 ms` (persistent)** | **410× – 448× faster** |
| **Cold 40-Item Viewport (First Thumbnail Ready)** | `46.0 ms` (+940ms UI stall) | **`28.4 ms` (+0ms stall)** | `176.9 ms` (+4.5s UI stall) | **`78.1 ms` (+0ms stall)** | **~2.2× faster worker + no UI stall** |
| **Cold 40-Item Viewport (All 40 Decoded)** | `258.1 ms` | **`204.1 ms`** | `182.0 ms` | **`120.2 ms`** | **1.3× – 1.5× faster** |
| **Warm 40-Item Viewport (Batched SQL Read)** | `23.8 ms` | **`19.6 ms`** | `16.7 ms` | **`12.4 ms`** | **Sub-20ms full viewport** |

---

## 3. v0.12.0 Verification Run & Editor/Colour Pop Micro-Benchmarks

*Measured: 2026-11-13 (`QT_QPA_PLATFORM=offscreen`, `tests/benchmark_scale.py` + v0.12.0 pipeline micro-benchmarks)*

### A. 100,000 & 500,000 Item Scale Regression Check (`tests/benchmark_scale.py`)

| Metric | 100k (v0.12.0) | 500k (v0.12.0) | Status |
| :--- | :---: | :---: | :--- |
| **`set_records()` + Qt `endResetModel()`** | `13.3 ms` | `37.4 ms` | Pass (no regression) |
| **1× `notify_pixmap_ready()` / 40× Est.** | `1.1 ms` / `45.7 ms` | `1.8 ms` / `70.1 ms` | Pass (direct update `3.53 ms` / `6.50 ms`) |
| **Mosaic Layout `compute_mosaic_geometry()`** | `44.3 ms` | `293.3 ms` | Pass |
| **SQL `query_all` / `query_folder` / `nav_refresh`** | `205.2 / 235.2 / 50.3 ms` | `1,389.5 / 1,880.2 / 148.1 ms` | Pass |
| **Scroll: Empty / Warm `p95`** | `10.34 / 12.84 ms` | `10.25 / 12.71 ms` | Pass (`60+ FPS`) |
| **Scroll: Contended / Arrival Max** | `15.27 / 11.17 ms` | `17.07 / 10.62 ms` | Pass (0 hitches) |
| **Cold 40-Item Viewport (First / Total)** | `24.3 ms` / `211.6 ms` | `26.5 ms` / `116.9 ms` | Pass |
| **Warm 40-Item Viewport (Workers / Batched SQL)** | `45.8 ms` / `22.6 ms` | `45.6 ms` / `23.0 ms` | Pass |

### B. Editor Stage Cache & Colour Pop Multi-Pick Speedups (`v0.12.0`)

| Pipeline Benchmark (`1280×853` Proxy, `960×640` Scrub) | Baseline / Cold | v0.12.0 Optimized | Speedup |
| :--- | :---: | :---: | :--- |
| **Editor `render_adjusted_image` (`1280px` cold, Denoise+Dehaze+LAB+Vignette)** | `270.3 ms` | — | Baseline |
| **Editor `dd_img` Stage Cache Hit (adjusting Exposure/LAB)** | `270.3 ms` | **`81.5 ms`** | **3.3× faster** |
| **Editor `tl_img` Stage Cache Hit (adjusting Vignette/Preset)** | `270.3 ms` | **`25.8 ms`** | **10.5× faster** |
| **Editor `960px` Active Scrub Proxy (cold)** | `270.3 ms` | **`87.0 ms`** | **3.1× faster** |
| **Colour Pop 4-Colour Mask (active tolerance scrub)** | `112.1 ms` | **`32.6 ms`** | **3.4× faster** |
| **Colour Pop 4-Colour Mask (released, `guided_filter_matte`)** | `112.1 ms` | **`43.9 ms`** | **2.6× faster** |

---

## 4. v0.16.0 P1/P2 & New Subsystem Micro-Benchmarks (20,000-Item Catalog)

*Measured: 2026-11-21 (Temporary SSD `DualDatabaseEngine`, 20,000 geotagged rows)*

| Operation (v0.16.0) | Measured Latency | Throughput / Notes |
| :--- | :---: | :--- |
| **`query_media_records()` (20k rows + `_annotate_media_records`)** | **`43.01 ms`** | Scales linearly with `100k` (`212.9 ms` in §2C) — zero regression from stacking/offline/live-photo annotations |
| **`query_map_cluster_media_ids()` (lazy click, 2,040 IDs)** | **`1.47 ms`** | Replaces upfront `GROUP_CONCAT` string splitting across all clusters |
| **`put_webp_blobs_batch()` (256 × 4 KB blobs)** | **`3.39 ms`** | Batched single-transaction commits (`ThumbnailWriteCoalescer`) |
| **`get_webp_blobs_batch()` (256 cached blobs)** | **`0.27 ms`** | **~1.05 µs/thumb** (vs `23.8 ms` for 40× individual reads in §2D baseline) |
| **`extract_tiff_embedded_jpeg()` (Tier-0 RAW fast-path)** | **`0.291 ms/file`** | `14.56 ms` for 50 files — avoids `rawpy.imread()` completely |
| **`detect_motion_photo_offset()` (EOI/XMP tail scan)** | **`0.277 ms/file`** | `27.73 ms` for 100 JPEGs |
| **`query_map_overview_cells()` (20k *distinct* 0.01° cells)** | `84.17 ms` | Worst-case 1-item-per-cell spread; runs **once** on map load, 0 SQL on pan/zoom |


---

## 5. v0.17.0 Pre-Release Benchmarks & Architecture Review

*Measured: 2026-11-25 (`QT_QPA_PLATFORM=offscreen`, `tests/benchmark_scale.py`, `tests/benchmark_scale_v2.py`, and v0.17.0 subsystem micro-benchmarks on Windows CPU / SSD)*

### A. 100,000 & 500,000 Item Scale Regression Check (`tests/benchmark_scale.py`)

| Metric | 100k (v0.12.0) | 100k (v0.17.0-pre) | 500k (v0.12.0) | 500k (v0.17.0-pre) | Trend / Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`set_records()` + Qt `endResetModel()`** | `13.3 ms` | `26.9 ms` | `37.4 ms` | `44.8 ms` | Pass (sub-50ms at 500k) |
| **1× `notify_pixmap_ready()` / 40× Est.** | `1.1 / 45.7 ms` | `1.6 / 62.9 ms` | `1.8 / 70.1 ms` | `4.4 / 177.9 ms` | Direct update: `4.04 ms` / `7.15 ms` |
| **Mosaic Layout `compute_mosaic_geometry()`** | `44.3 ms` | `51.6 ms` | `293.3 ms` | `330.9 ms` | +13% vs v0.12.0 |
| **SQL `query_all` (Date Sort)** | `205.2 ms` | `246.3 ms` | `1,389.5 ms` | `1,720.8 ms` | +20–24% (extra `media` columns/annotations) |
| **SQL `query_folder` (100 Subfolders)** | `235.2 ms` | `277.9 ms` | `1,880.2 ms` | `2,348.3 ms` | +18–25% vs v0.12.0 |
| **SQL `nav_refresh_sql` (Nav Tree Counts)** | `50.3 ms` | **`98.8 ms`** | `148.1 ms` | **`665.8 ms`** | **Watchlist (~2×–4.5× vs v0.12.0)** |
| **Scroll: Empty / Warm `p95`** | `10.34 / 12.84 ms` | `10.67 / 12.77 ms` | `10.25 / 12.71 ms` | `15.69 / 13.05 ms` | Pass (`60+ FPS`) |
| **Scroll: Contended / Arrival Max** | `15.27 / 11.17 ms` | `16.06 / 11.16 ms` | `17.07 / 10.62 ms` | `15.58 / 7.18 ms` | Pass (0 hitches) |
| **Cold 40-Item Viewport (First / Total)** | `24.3 / 211.6 ms` | `63.7 / 457.3 ms` | `26.5 / 116.9 ms` | `40.0 / 228.8 ms` | Varies with CPU thermal/background load |
| **Warm 40-Item Viewport (Workers / Batched SQL)** | `45.8 / 22.6 ms` | `53.7 / 39.1 ms` | `45.6 / 23.0 ms` | `49.6 / 33.1 ms` | Persistent conn: `0.24–0.25 ms` vs `95 ms` reopen |

---

### B. People Directory & Batch People Tagging (`tests/benchmark_scale_v2.py`)

| Operation | 50,000 Images (33k Faces) | 500,000 Images (333k Faces) | Notes |
| :--- | :---: | :---: | :--- |
| **`get_total_media_count()`** | `10.91 ms` | `229.60 ms` | Full table count without cached summary |
| **`get_folders()`** | `0.20 ms` | `1.02 ms` | 26 / 251 folders |
| **`get_date_tree_counts()`** | `11.32 ms` | `149.16 ms` | 2 / 7 date buckets |
| **`get_people_directory_data()`** | `20.75 ms` | `252.54 ms` | 100 / 300 named people |
| **`get_pending_face_suggestions(limit=5000)`** | `50.98 ms` | **`275.57 ms`** | Uses `idx_media_people_state` + `USE TEMP B-TREE FOR ORDER BY` |
| **Avatar Picker Strip Query** | `0.54 ms` | `0.36 ms` | Fast PK `IN (...)` lookup |

---

### C. v0.17.0 Subsystem Micro-Benchmarks (20k Catalog, Semantic AI, Editor, Colour Pop, LaMa)

| Subsystem / Operation | Measured Latency | Notes |
| :--- | :---: | :--- |
| **`query_media_records()` (20k rows)** | `43.67 ms` | Flat vs v0.16.0 (`43.01 ms`) |
| **`query_media_records()` Multi-Select (5 folders + 3 cameras, 20k)** | `15.34 ms` | 800 matching rows |
| **`get_places_hierarchy()` / `get_date_tree_counts()` (20k)** | `11.91 ms` / `3.51 ms` | Fast at 20k; grows linearly at 500k |
| **`query_map_overview_cells()` (20k distinct 0.01° cells)** | `85.83 ms` | Flat vs v0.16.0 (`84.17 ms`) |
| **`put_webp_blobs_batch(256)` / `get_webp_blobs_batch(256)`** | `2.44 ms` / `0.61 ms` | ~2.4 µs/thumb cached read |
| **`detect_motion_photo_offset()` (100 JPEGs)** | `0.721 ms/file` | `72.09 ms` total |
| **Semantic AI `load_matrix()` (20k / 100k fp16 vectors)** | `118.94 ms` / **`675.95 ms`** | One-time cold load into RAM; keep off GUI thread |
| **Semantic AI `compute_subject_counts()` (12 presets, 20k / 100k)** | **`2.62 ms`** / **`11.36 ms`** | Batched `mat @ subj_mat.T` after matrix load |
| **Semantic AI `query_by_vector()` (20k / 100k)** | **`0.54 ms`** / **`4.10 ms`** | Sub-5ms semantic search at 100k photos |
| **Editor `render_adjusted_image` (`1280×853` cold / `dd_img` / `tl_img`)** | `282.61` / `92.13` / `24.44 ms` | Consistent with v0.12.0 stage cache |
| **Colour Pop Quick Styles (`1280×853`, `style_red` / `style_warm`)** | `57.54 ms` / `56.53 ms` | Interactive single-click response |
| **LaMa Object Eraser (`inpaint_object_lama`, 4K ROI, warm CPU)** | **`1,427.22 ms`** | Runs asynchronously off GUI thread in Spot Fix |

---

### D. Candidate Improvements, Further Testing & Trade-Offs (Pre-Release Review)

#### Potential Improvements (Not Yet Implemented)
1. **Navigation Tree Refresh SQL (`nav_refresh_sql`: `98.8 ms` @ 100k, `665.8 ms` @ 500k):**
   - Investigate why `nav_refresh_sql` grew 2×–4.5× since v0.12.0 (likely additional `WHERE is_missing = 0 AND is_rejected = 0` / `folders.is_private = 0` joins or unindexed `date_taken` / `location_*` group-bys) and move full `refresh_trees()` SQL off the main thread or maintain lightweight summary counts.
2. **Covering Index for `get_pending_face_suggestions()` (`275.6 ms` @ 500k):**
   - `EXPLAIN QUERY PLAN` shows `USE TEMP B-TREE FOR ORDER BY` over all `state = 0` rows. A partial covering index on `media_people(state, suggested_person_id, similarity_score DESC) WHERE state = 0` would eliminate the temp B-tree sort at 500k scale.
3. **Full-Catalog `query_media_records()` (`1.72 s` @ 500k, `2.35 s` for 100-subfolder query):**
   - Materializing 500k Python `MediaItemRecord` objects in one synchronous call on the GUI thread causes a noticeable pause when switching views; consider incremental/background query execution or a lighter projection for the initial viewport.
4. **Semantic Matrix Cold Load (`675.95 ms` @ 100k):**
   - Ensure `SemanticVectorRepository.load_matrix()` is always warmed on a background thread (never on the first synchronous text/subject click on the GUI thread), and consider storing the matrix as a contiguous memory-mapped `.npy` sidecar or reading blobs via `np.frombuffer` without per-row Python loops.
5. **Synchronous LaMa Inpainting in Straighten (`apply_crop` → `straighten_with_ai_fill`):**
   - Like Object Eraser (`~1.43 s`), `straighten_with_ai_fill` still runs LaMa synchronously on the GUI thread (~1.4 s stall in `freeze_traces.log`); moving it to the same async worker pattern would eliminate that stall.

#### Useful Further Testing
- **Real-world cold startup & view-switch profiling (`tests/benchmark_startup_and_views.py`)** against an actual multi-100k user `catalog.db` (measuring `MainWindow.__init__`, People Directory, Person Grid, Face Crops toggle, and return to Gallery).
- **Grouped Mosaic scroll & resize benchmark** (100k items grouped by Camera / Date / Folder, measuring scroll frame `p95` and zoom slider responsiveness).
- **Concurrent background worker contention**: measure UI frame times while Face Scan + Semantic Indexer + Thumbnail Writer + Folder Watcher run simultaneously.
- **Memory footprint tracking (RSS)** at 100k & 500k items with `catalog_model`, `SemanticVectorRepository` float32 matrix (~205 MB at 100k), and `LruCache` active.

#### Compromises & Trade-Offs to Weigh
- **SQLite indexes vs. scan/write speed & DB size:** Adding composite/partial indexes speeds up `nav_refresh_sql` and `get_pending_face_suggestions`, at the cost of ~5–15% slower initial folder ingestion and larger `catalog.db` size.
- **Semantic matrix RAM (`float32` vs `float16`):** Keeping the cached matrix in `float32` gives `4.1 ms` search at 100k (~205 MB RAM); keeping `float16` halves RAM (~102 MB) but slows CPU BLAS matmul significantly because NumPy upcasts `float16` on CPU.
- **Async queries vs. UI complexity:** Moving `query_media_records` and `nav_refresh_sql` fully off the GUI thread prevents >600 ms stalls at 500k items, but requires generation-token cancellation so rapid filter clicks never display stale results.
- **LaMa CPU latency vs. quality:** `lama_fp32.onnx` on CPU takes ~1.4 s per 512×512 ROI; quantized int8 LaMa or DirectML GPU (`onnxruntime-directml`) could cut latency 3–5×, at the trade-off of model packaging size or GPU driver compatibility.
