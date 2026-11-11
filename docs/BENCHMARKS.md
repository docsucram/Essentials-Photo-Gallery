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
