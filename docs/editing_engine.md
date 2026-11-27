# Essentials Photo Gallery — Edit & Create Engine Specification (`docs/editing_engine.md`)

Unified technical architecture, mathematical formulations, Depth & Light subsystem, Multi-Photo Create studio algorithms, and interactive UI/UX behavior for **Essentials Photo Gallery** (`src/imaging/` and `src/ui/`).

---

## 1. Architectural Overview & Core Design Principles

The **Edit & Create Engine** pairs deterministic floating-point / LUT-accelerated image pipelines with responsive PySide6 graphics viewports across three layers:

```text
+-------------------------------------------------------------------------+
|                            UI & Viewer Layer                            |
|  (src/ui/media_viewer.py, src/ui/right_tag_dock.py, src/ui/crop_tool.py,|
|   src/ui/markup_overlay.py, src/ui/create_studio_dialogs.py,            |
|   src/ui/ribbon_toolbar.py)                                             |
|  - SingleMediaViewer with hardware-accelerated zoom/pan & proxy preview |
|  - Right dock Fine Tune + Depth & Light controls (16ms timer coalesce)  |
|  - Save to Original (.OriginalFiles backup) vs Save as Copy split flows |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                         Imaging Pipeline Layer                          |
|  (src/imaging/opencv_engine.py, src/imaging/creative_effects.py,        |
|   src/imaging/create_tools.py, src/imaging/image_loader.py,             |
|   src/imaging/video_frame_capture.py)                                   |
|  - Non-destructive PhotoAdjustmentRecipe parametric stack & .cube LUTs  |
|  - Precomputed sRGB <-> Linear LUTs & unified 256-entry cv2.LUT tonal   |
|  - Monocular 3D depth maps & portrait alpha matte proxy caching         |
|  - Next-gen codec decoding, Display P3 -> sRGB, & video frame capture   |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                    Computer Vision & AI Runtime Layer                   |
|  - OpenCV 5.0 (cv2): SIFT/ORB, ALIKED/DISK+LightGlue, USAC_MAGSAC,      |
|    Telea/NS Inpainting, CIE LAB                                         |
|  - ONNX Runtime: Depth-Anything-V2 / MiDaS & MODNet / RMBG matting      |
|  - NumPy 2.5.4: Vectorized AVX2/AVX-512 float / Laplacian math          |
|  - Pillow 13.0.0: Next-gen codecs (HEIC/AVIF/JXL), ICC & EXIF handling  |
+-------------------------------------------------------------------------+
```

### Core Performance & Fidelity Guarantees
1. **Two-Tier Display Proxies (`scrub_proxy_rgb` & `display_proxy_rgb`):** Active slider scrubbing (`_is_sliding`) evaluates against a fast `960px` `scrub_proxy_rgb` with `16ms` timer coalescing (`_ft_timer`, `_dl_timer` in `right_tag_dock.py`), snapping immediately to the crisp `1280px` `display_proxy_rgb` on release.
2. **Background Master Commit & Dirty-Stage Cache (`_MasterRenderRunnable` & `_STAGE_CACHE`):** Releasing sliders schedules a debounced (`250ms` `_master_render_timer`) full-resolution `master_np_rgb` pass on `QThreadPool` with generation-counter cancellation (and synchronous `ensure_master_rendered()` before Save/Export). `render_adjusted_image()` caches intermediate `dd_img`, `tl_img`, and `bokeh_data` checkpoints per `(media_id, H, W)` so downstream sliders skip upstream passes.
3. **Precomputed Look-Up Tables (`cv2.LUT`):** Levels, Filmic Exposure, and Filmic Contrast combine into a single 256-entry `uint8` LUT executing in **`1.1 ms`**, paired with precomputed sRGB/linear (`_LUT_SRGB_TO_LIN`, `_LUT_LIN_TO_SRGB`, `_VIBRANCE_POW_LUT`) and 4x downsampled base/detail illumination decomposition.
4. **Non-Destructive File Protocol (`.OriginalFiles/`):** Saving over an original file stashes a byte-identical untouched copy inside `<folder>/.OriginalFiles/<filename>` (never overwriting an existing backup), enabling exact **Revert to original** and preserving EXIF/XMP blocks and spatial face coordinates.
5. **Embedded JPEG Fast-Path for Camera RAW Files (`decode_raw_embedded_preview`):** Thumbnails and initial Single Image View opens extract the camera's embedded JPEG/bitmap preview (`raw.extract_thumb()` with EXIF orientation) for instant browsing, deferring full `rawpy` demosaicing (`_upgrade_raw_to_full_demosaic_async()`) until active slider editing.
6. **Two-Tier Fine Tune (`Basic | Advanced`), Look Profiles / `.cube` 3D LUTs & Live Masked Histogram:** Defaults to classic WLPG simplicity with a `94px` single-color luminance histogram (`HistogramWidget`), while toggling **Advanced** unlocks a `128px` multi-channel `R/G/B` histogram, **Presets** (`Look` profiles `As Shot` / `Natural` / `Warm Print` / `Vivid` / `Portrait` / `Monochrome`, importable `.cube` 3D LUTs via `LUT...`, and `Profile amount 0–200%`), **Smart Enhance**, **Whites/Blacks**, content-aware **Structure**, **Tone Curve** (`RGB/R/G/B` + S-Curve), **WB Presets & Pipette Eyedropper**, **8-Band Color Mixer (HSL)**, **Sharpen Radius/Masking**, **Denoise Detail/Color**, **Orton Glow / Golden Hour**, **Film Grain**, and **Regional Masking (`Whole | Subject | Bg | Sky`)** with mask-restricted live histogram updates (`set_histogram_mask`) and neutral red-dot mask fill toggle. `_AeroFineTuneSlider` uses `Qt.StrongFocus` (ignoring wheel events when unfocused unless `Alt` is held, stepping `±1` or `±5` with `Shift`) and propagates disabled states cleanly to paired headers.

---

## 2. Parametric Recipe Stack (`PhotoAdjustmentRecipe`)

Defined in [`src/imaging/opencv_engine.py`](../src/imaging/opencv_engine.py), all non-destructive tonal, color, geometry, and 3D depth controls serialize cleanly:

```python
@dataclass
class PhotoAdjustmentRecipe:
    # Tone & Exposure
    brightness: float = 0.0          # -100.0 .. +100.0 (Filmic shoulder roll-off)
    smart_enhance: float = 0.0       # 0.0 .. 100.0 (Adaptive single-slider balance)
    contrast: float = 0.0            # -100.0 .. +100.0 (Filmic S-curve on L*)
    shadows: float = 0.0             # -100.0 .. +100.0 (Edge-aware shadow lift/crush)
    highlights: float = 0.0          # -100.0 .. +100.0 (Highlight recovery)
    whites: float = 0.0              # -100.0 .. +100.0 (Upper highlight stretch)
    blacks: float = 0.0              # -100.0 .. +100.0 (Deep shadow toe stretch)
    clarity: float = 0.0             # -100.0 .. +100.0 (Midtone local contrast)
    structure: float = 0.0           # -100.0 .. +100.0 (Skin/sky-safe detail)
    black_point: float = 0.0         # 0.0 .. 255.0 (Histogram shadow clip)
    white_point: float = 255.0       # 0.0 .. 255.0 (Histogram highlight clip)
    curve_rgb/r/g/b: list[tuple]     # PCHIP tone curve control points

    # Color, Glow & Atmospheric
    temperature: float = 0.0         # -100.0 .. +100.0 (Cool Blue <-> Warm Amber)
    tint: float = 0.0                # -100.0 .. +100.0 (Green <-> Magenta)
    saturation: float = 0.0          # -100.0 .. +100.0 (Skin-safe vibrance)
    dehaze: float = 0.0              # -100.0 .. +100.0 (Dark channel prior recovery)
    hsl_hue/sat/lum: dict            # 8-band perceptual HSL color mixer
    vignette: float = 0.0            # -100.0 .. +100.0 (Elliptical corner shading)
    vignette_feather: float = 50.0   # 0.0 .. 100.0 (Vignette falloff softness)
    orton_glow: float = 0.0          # 0.0 .. 100.0 (Dreamy highlight bloom)
    golden_hour: float = 0.0         # 0.0 .. 100.0 (Warm sunset highlight glow)

    # Detail, Grain & Geometry
    straighten_deg: float = 0.0      # -15.0 .. +15.0 degrees
    perspective_v: float = 0.0       # -100.0 .. +100.0 Vertical keystone pitch
    perspective_h: float = 0.0       # -100.0 .. +100.0 Horizontal keystone yaw
    flip_h: bool = False             # Horizontal mirror toggle
    flip_v: bool = False             # Vertical flip toggle
    sharpen: float = 0.0             # 0.0 .. 100.0 (Luminance unsharp mask)
    sharpen_radius: float = 25.0     # 0.0 .. 100.0 (0.5x .. 2.5x radius scale)
    sharpen_masking: float = 0.0     # 0.0 .. 100.0 (Sobel edge-only mask)
    denoise: float = 0.0             # 0.0 .. 100.0 (2D Sobel edge-guided luma denoise)
    denoise_detail: float = 50.0     # 0.0 .. 100.0 (Luma edge preservation weight)
    denoise_chroma: float = 0.0      # 0.0 .. 100.0 (Independent chroma denoise)
    grain_amount: float = 0.0        # 0.0 .. 100.0 (Organic film grain)
    grain_size: float = 25.0         # 0.0 .. 100.0 (Grain particle size)
    region_edits: dict[str, dict]    # Per-region overrides ("subject"|"background"|"sky" + mask_strokes)
    crop_rect: tuple[int, ...] | None = None  # (left, top, right, bottom)

    # Creative Presets & Selective Control Points
    effect: str = "none"
    effect_intensity: float = 100.0
    selective_points: list[dict] = field(default_factory=list)

    # 3D Depth & Light
    depth_blur: float = 0.0          # Optical bokeh defocus strength
    focal_plane: float = 0.0         # Target focal depth slice
    bokeh_glow: float = 0.0          # Specular bokeh highlight bloom
    relight_near: float = 0.0        # Foreground subject fill lighting
    relight_far: float = 0.0         # Background illumination balance
    warmth_near: float = 0.0         # Foreground warmth
    warmth_far: float = 0.0          # Background warmth
    depth_haze: float = 0.0          # Beer-Lambert atmospheric depth fog
    depth_haze_warmth: float = 0.0   # Mist warmth tint
```

---

## 3. Mathematical & Algorithmic Specifications

### 3.1 Dual-Domain Denoise, 2D Sobel Edge Preservation & MAD Noise Estimation
Laplacian MAD noise estimate:
$$\hat{\sigma} = \frac{\text{median}(|\nabla^2 I - \text{median}(\nabla^2 I)|)}{0.6745}$$
Resolution-scaled bilateral smoothing separates chrominance (`denoise_chroma`, wide $\sigma_c$) from luminance (`denoise`). When `denoise >= 1.0`, full 2D Sobel gradient magnitude (`cv2.magnitude(gx, gy)`) computes `edge_weight`, modulated by **Detail preservation** (`denoise_detail`, scaling edge protection between `0.35` and `0.95`) so skies and flat shadows smooth cleanly while textures remain crisp.

### 3.2 Filmic Exposure, Smart Enhance, Content-Aware Structure & AgX Roll-Off
- **Levels & Whites/Blacks Stretching:** Baked with Filmic Exposure into the 256-entry uint8 `cv2.LUT` (`1.1 ms`).
- **Positive Exposure Shoulder:** $f_{\text{scaled}} = f \cdot (1 + 1.4 b)$, asymptotically compressed above $\text{knee} = 0.65$ via $f_{\text{compressed}} = \text{knee} + \frac{\Delta}{1 + 1.8\Delta}$ to avoid harsh highlight clipping.
- **Negative Exposure Toe:** Attenuates in stops ($2^{1.5 b}$) with subtle shadow toe lift $0.05 |b| f (1 - f)$ preserving deep shadow detail.
- **Smart Enhance (`0..100`):** Measures mean frame luminance to adaptively lift underexposed shadows/midtones, gently pull harsh highlights, add mild midtone contrast, and boost skin-safe vibrance from a single slider.
- **Content-Aware Structure (`-100..+100`):** Extracts high-frequency luminance detail and attenuates positive boosts by `75%` across warm human skin locus and `65%` across smooth bright sky/highlights.
- **CIE LAB Shadows/Highlights + Ambience & AgX Roll-Off:** Bilateral base/detail separation ($L = L_{\text{base}} + L_{\text{detail}}$) lifts dark regions while keeping micro-contrast punchy; highlights ($L^* > 220$) smoothly desaturate toward paper-white.
- **Clarity & Orton Glow / Golden Hour:** Midtone bell-curve `Clarity`, screen-blended `Orton Glow`, and warm highlight-weighted `Golden Hour`.

### 3.3 Color Temperature, Tint, Skin-Safe Vibrance & Upgraded Auto Adjust in CIE LAB
- **Temperature ($B^*$) & Tint ($A^*$):** Orthogonal chromatic offsets in perceptual LAB space. The Edit Ribbon **Exposure** (Brighter top → Darker bottom × Lower→Higher contrast) and **Color** (High/magenta tint top → Low/green tint bottom × Cooler→Warmer temperature) 3×3 variation menus match WLPG ordering with descriptive hover tooltips and live image previews.
- **Upgraded Configurable Auto Adjust (`analyze_auto_adjust`, `DEFAULT_AUTO_ADJUST_OPTIONS`):**
  - Supports 7 independent toggleable passes controlled via **Edit → Auto adjust ▾** (`exposure_contrast`, `colour_wb`, `detail_clarity`, `denoise`, `dehaze`, `straighten`, `perspective`).
  - **Key-Aware & Highlight-Safe Auto Exposure + Black Point Anchor:** Dampens median shifts on intentional low-key/high-key scenes, scales back positive brightness when `p98 > 232`, and lifts crushed/washed-out black points gently without clipping.
  - **Near-Neutral CIE Lab White Balance & Adaptive Vibrance:** Estimates color cast from low-chroma midtone pixels (`chroma < 24`, `28 < L < 228`), falls back to damped Gray-World when neutral pixels are scarce, and gates Detail/Sharpening by estimated Laplacian MAD noise $\hat{\sigma}$.
- **Skin-Safe Vibrance (Zero HSV Round-Trip):** Measures Euclidean chroma $C = \sqrt{(A^*-128)^2 + (B^*-128)^2}$ and proximity to the warm human skin locus ($A^* \approx 146, B^* \approx 152$):
  $$W_{\text{vibrance}} = \left(1 - \left(\frac{C}{128}\right)^{0.8}\right) \cdot \left(1 - 0.6 \cdot \text{clip}(1 - D_{\text{skin}}, 0, 1)\right)$$
  Boosting muted blues/greens cleanly without turning portraits orange.

### 3.4 Atmospheric Dehaze & Halo-Free Sharpening
- **Koschmieder Dark Channel Prior:** Estimates atmospheric light $A$ (99th percentile luminance $\ge 100$) and refines the transmission map $t(x)$ with a bilateral filter before recovering $J(x) = \frac{I(x) - A}{t(x)} + A$ without sky/horizon halos.
- **Luminance Unsharp Mask with Radius, Masking & Halo Clamping:** Evaluates Gaussian radius from `sharpen_radius` (`0.5x..2.5x`, floored proxy `res_scale >= 0.5`), applies $\pm 3.0$ dead-zone soft thresholding, clamps `sharp_delta` to `[-14, +18]` in `L*` to eliminate overshoot halos, and optionally gates flat regions via `sharpen_masking` (`Sobel` edge magnitude).

### 3.5 Regional Masking (`Whole | Subject | Bg | Sky`) & Interactive Mask Brush
- **Region Segmentation (`DepthAndMatteEngine.get_region_mask`):** Derives `subject` and `background` from `alpha_matte` and segments `sky` by combining upper-frame depth priors, blue/overcast/golden-hour sunset chromaticity, $5 \times 5$ local luminance standard deviation + 4-neighbor Laplacian texture rejection (excluding foliage/roofs), top-edge connected components, $3 \times 3$ morphological close, and Joint Guided Filter edge snapping.
- **Interactive Brush Touch-Up (`apply_region_mask_strokes`):** Supports normalized `Paint` and `Erase` edge-aware bilateral brush strokes per region (`mask_strokes`), reflected live in both the **Show mask** overlay and the region-masked histogram (`HistogramWidget.set_histogram_mask`).

### 3.6 Contextual Crop, Perspective Homography & Zero-Wedge Auto-Crop
- **Projective Keystone & Live GPU Proxy (`build_perspective_homography`, `apply_geometry_transform`):** Solves 4-point projective homographies from `perspective_v`, `perspective_h`, `straighten_deg`, `flip_h`, and `flip_v` (`cv2.getPerspectiveTransform` + `cv2.warpPerspective(..., flags=cv2.INTER_LANCZOS4)`), anchored at scene `(0, 0)` during interactive slider scrubbing and evaluated via `estimate_auto_perspective()` for architectural vertical convergence.
- **Zero-Border Auto-Straighten / Inscribed Homography Crop & Async AI Fill:** Analytically solves the largest axis-aligned inscribed rectangle (`compute_inscribed_rect_for_homography`) or matches original aspect ratio $R = W_0 / H_0$ (`detect_auto_straighten_angle`) with optional non-blocking background `LaMa` AI Expand corner fill (`chk_gen_fill` / `btn_ribbon_crop_ai_fill`):
  $$W_{\text{crop}} = \frac{W_0\cos\alpha - H_0\sin\alpha}{\cos 2\alpha}, \quad H_{\text{crop}} = \frac{H_0\cos\alpha - W_0\sin\alpha}{\cos 2\alpha}$$
- **U-Point Control Points:** Combines spatial falloff $w_s$ and LAB photometric affinity $w_c = \exp(-(\Delta E / 28)^2)$ using normalized $[0, 1]$ coordinates identical across proxy and master renders.

---

## 4. Photographic Preset Effects Engine

| Effect ID | Display Name | Emulation Target | Core Formulation |
| :--- | :--- | :--- | :--- |
| `none` | **None** | Baseline | Pass-through |
| `classic_chrome` | **Classic Chrome** | Fujifilm Classic Chrome | Reduced saturation (-18%), midtone contrast boost, warm highlights, teal shadows |
| `warm_portrait` | **Warm Portrait** | Kodak Portra | Golden highlight glow, lifted shadows, skin protection, gentle micro-contrast |
| `vivid_landscape` | **Vivid Landscape** | Fujifilm Velvia 50 | Strong saturation (+38%), deepened sky blues, crisp blacks, +25 clarity |
| `matte_vintage` | **Matte Vintage** | 1970s Matte Print | Raised black point ($L=22$), highlight roll-off, warm cross-fade |
| `fine_art_bw` | **Fine Art B&W** | Neopan Acros 100 | Panchromatic luminance ($0.299R + 0.587G + 0.114B$), steep S-curve, +30 structure |
| `cool_cinema` | **Cool Cinema** | 35mm Teal & Orange | Shadow teal split-tone, amber highlight split-tone, filmic curve |
| `sepia` | **Sepia** | Silver Bromide Print | Panchromatic mono + warm duotone matrix ($R\times 1.15, G\times 0.98, B\times 0.80$) |

All presets blend continuously from $0\%$ to $100\%$ intensity.

---

## 5. Interactive Spot Fix (Retouch & Red Eye) and WYSIWYG Markup

### 5.1 Unified Spot Fix & Neural LaMa Inpainting (`src/ui/media_viewer.py`, `src/imaging/lama_inpaint.py`)
- **Mutual-Exclusion Modes:** Compact floating toolbar toggles **Blemish** healing, **Object eraser** (powered by `LaMaInpaintEngine` with automatic frequency-separation + Telea fallback when the add-on is not installed), and **Red eye** reduction sharing a continuous $6\text{px}..80\text{px}$ slider, `[` / `]` keyboard sizing, unified **`B`** Hold-to-Compare (also shared by **Fine Tune** and **Depth & Light**, preserving `_original_pixmap` before the first bake), and **`Shift+Drag` Straight Line Mode** (live brush-width + dashed center-guide preview healing film scratches or power lines in a single stroke).
- **Unified LaMa ONNX Inpainting (`LaMaInpaintEngine`):** Runs `lama_fp32.onnx` (Fast Fourier Convolutions with image-wide receptive field) inside padded $512 \times 512$ local bounding-box crops with $21\text{px}$ feathered Gaussian alpha blending into surrounding pixels. Reused across **Spot Fix Object Eraser**, **Crop & Straighten Generative Corner Fill** (`chk_gen_fill`), **Panorama Stitcher Boundary Fill**, **3D Cinematic Motion Disocclusion Fill**, and **Depth & Light Background Removal Fill**. Managed via *File → Options → Add-ons*, where `get_lama_model_status()` inspects disk state without eagerly loading ONNX Runtime into memory.
- **Exact 1:1 View-Scale Geometry:** Brush radius maps dynamically through graphics transform scale $M_{11}$ ($r_{\text{scene}} = r_{\text{screen}} / \text{scale}$, $r_{\text{master}} = \max(2, \text{round}(r_{\text{scene}}))$) paired with a dual-ring high-contrast cursor.
- **Frequency-Separation Texture Transfer (Blemish):** Splits local illumination via Gaussian blur ($L_{\text{low}}$ inpainted via Telea), extracts high-frequency pores/grain ($I_{\text{high}} = I - I_{\text{low}}$), copies unblemished texture from an 8-direction surrounding donor ring via `cv2.warpAffine`, variance-normalizes ($\sigma_{\text{donor}} / \sigma_{\text{transferred}}$), and recombines cleanly without smudging.
- **Dual-Mode Red Eye:** Connected-component pupil detection (`r_ratio >= 1.25`, morphological closing to protect white corneal glints, neutral dark replacement) triggered either by brush click/stroke or legacy box drag.

### 5.2 Contextual Markup Studio Ribbon & WYSIWYG Engine (`src/ui/markup_overlay.py`, `src/ui/ribbon_toolbar.py`, `src/ui/media_viewer.py`)
- **Contextual `Markup` Ribbon Tab & 9 Tools:** Entering Markup opens a dedicated `Markup` tab (`Edit | Create | View | Markup`) with 9 WLPG tools:
  - **Select (`V` / Hold `Alt` temporary Select):** Reselects any session markup item for interactive move, 4-corner resize, on-canvas rotation (round top handle + corner outer-zone rotate), `Edit Text`, `Duplicate` (`Ctrl+D`), undoable `Delete` (`Del`), and global `Ctrl+C` / `Ctrl+V` copy/paste from any active tool. Uses an app-level `Key_Alt` event filter so holding `Alt` temporarily switches to Select without Windows menu-bar focus stealing.
  - **Pen:** Freehand strokes with non-cumulative live opacity (`item.setOpacity`) so overlapping strokes never darken mid-stroke.
  - **Text:** True WYSIWYG inline editor (transparent canvas, live `QTextCharFormat` outline stroke and `QGraphicsDropShadowEffect` shadow preview while typing, first click-away commits/deselects, `Enter` commits, `Shift+Enter` newline) with Font, Bold/Italic, Alignment, an editable Canva/Fluent-style `[ − | pt | + ]` **Text Size** stepper + popup preset list (`12–440 pt`), and icon-equipped `Pill / Outline / Shadow / Plain` style dropdown.
  - **Shapes ▾ (4×2 Visual Gallery):** Rectangle, Rounded Rectangle, Ellipse, Triangle, Star, Speech Bubble, Heart, Callout; `Outline / Solid / Both` fill style (filled shapes use the Secondary/Background colour) and `Solid / Outlined / Dashed / Dotted` border line styles.
  - **Line ▾:** Straight Line, Single Arrow, and Double Arrow with `Solid / Outlined / Dashed / Dotted` styles, live post-draw style switching, 4-corner resize, and on-canvas rotation.
  - **Obscure ▾:** Privacy Blur ($\sigma$ scaled by Level) and Mosaic Pixelation in either Brush or Box mode (Brush Size slider hidden in Box mode; neutral frosted preview with no blue selection box).
  - **Stickers ▾ & Stamp ▾:** Drag-to-draw 5-sticker gallery and live-editable metadata/watermark stamps (Place GPS, Date, Camera, Custom Watermark), both dynamically updating their Ribbon button icon to reflect the active selection.
  - **Magnify:** Antialiased `QBrush` glass loupe (`1.5x..4.0x` zoom + configurable initial circle Size).
- **Resolution-Aware Defaults & Paint-Style Colours:** Automatically scales initial stroke/text width to the photo's short edge (`12..48px` on photos `>= 600px`, `16px` fallback) so annotations are immediately legible on 12MP–45MP photos. Defaults to Black (`#111111`) foreground / White (`#FFFFFF`) background with a 3×6 swatch grid (`18×18`), enlarged `48×46` overlapping foreground/background chip (`⇄` / `X` swap), and a `36×36` Paint-style rainbow colour wheel (`make_rainbow_color_wheel_icon()`).
- **Zero-Jump Full-Res ROI Baking & Safe Undo:** `flatten_markup_element_roi()` renders only the element's padded bounding box (with explicit `.copy()` to avoid NumPy view aliasing) and paints the dirty ROI directly onto the full-resolution `QPixmap` in place (`scale=1.0`), eliminating viewport jumps and proxy flashes. `PixelArrayUndoCommand` stores only modified bounding-box slices (`<15 KB` vs full 24MP frames) plus `_markup_elements` snapshots so Undo/Redo and Delete/Reselect remain strictly consistent.

---

## 6. 3D Depth & Light Subsystem (`src/imaging/creative_effects.py`)

### 6.1 Dual Inference Engine + Proxy Caching
- **ONNX Runtime:** Uses `DepthAndMatteEngine` (`depth_anything_v2_vits.onnx` / `midas_v21_small_256.onnx` for continuous depth $D \in [0, 1]$ and `modnet.onnx` / `rmbg-1.4.onnx` for alpha matting $M \in [0, 1]$) with zero-dependency Haar + GrabCut + Joint Guided Filter fallback. Computed once per `media_id` at $640\text{px}$ (`_base_cache`) and bilinearly scaled (`<0.5ms`).

### 6.2 Vectorized Depth Algorithms
- **Direct RGB Relighting (`14.1 ms` at 1280px):** Independent near ($W_{\text{near}} = \max(M, 1 - D)$) and far ($W_{\text{far}} = D(1 - M)$) luminance/warmth weighting directly in RGB without LAB bounces.
- **Optical Bokeh Blur (`35.0 ms` at 1280px):** Defocus map $\delta = |D - z_f| \cdot (1 - M \cdot [z_f < 0.25])$ evaluated at $640\text{px}$ in linear light (`_LUT_LIN_TO_SRGB`) with specular highlight bloom.
- **Beer-Lambert Depth Haze (`26.0 ms` at 1280px):** Transmittance $T = 1 - \exp(-\beta D^2)$ masked cleanly behind foreground subjects.

---

## 7. Multi-Photo Creation Studio (`src/imaging/create_tools.py`, `src/ui/create_studio_dialogs.py`)

1. **ICE-Style Panorama Stitcher (`PanoramaDialog`):** Background `PanoramaWorker` loads EXIF-safe frames and runs OpenCV `Stitcher` (or `estimate_homography_with_fallback`: SIFT/ORB → OpenCV 5.0 ALIKED/DISK + LightGlue learned matcher when ONNX weights are present → CLAHE-boosted SIFT, all solved with `USAC_MAGSAC` homography) with OpenCL disabled (`cv2.ocl.setUseOpenCL(False)`) and a `2400px` input long-edge cap for stability, plus exposure compensation, multi-band Laplacian feathering, Navier-Stokes/Telea corner gap fill, and **Largest Interior Rectangle Auto-Crop** (`find_largest_interior_rectangle`) via 2D histogram monotonic stacks.
2. **Interactive Collage & "Then & Now" (`CollageDialog`):** Smart hero face slot protection (`>94%` opacity lock, $40\%$ upward headroom crop) guaranteeing $100\%$ gapless grid/masonry coverage (1–35 photos), tactile **Polaroid Table Scatter** with drop shadows and date/caption stamping, chronological **Then & Now** comparison cards, 2-row curated backdrop swatches (neutrals + editorial pastels) paired with the shared Paint-style rainbow colour wheel picker, and per-slot zoom/pan/shape framing.
3. **Photo Fuse (`PhotoFuseDialog`):** Background `_PhotoFuseWorker` loads frames, aligns burst shots via `estimate_homography_with_fallback` (SIFT/ORB + `USAC_MAGSAC` homography with OpenCV 5.0 ALIKED/DISK + LightGlue and CLAHE fallbacks, $\ge 6$ inliers, determinant sanity gate, and warning banner for unalignable shots), and detects faces with a CPU-only InsightFace detection session (`allowed_modules=["detection"]`, `det_score >= 0.55`, 5-point landmark gate, strict Haar fallback). Features a visual Base Shot thumbnail picker with per-base caching (`_base_cache`), per-face local alignment (`_align_face_patch_local` via 5-point landmark `estimateAffinePartial2D` or Euclidean ECC fallback), mask-aware CIE LAB color matching, multi-band Laplacian pyramid blending (up to 5 levels), a **Show face boxes** toggle, and a **View Original Base** comparison banner.
4. **Colour Pop (`ColourPopDialog`):** Subject mask seeded from face detections or center saliency + Photoshop-style multi-color eyedropper (`Click` set, `Shift+Click` add, `Right/Alt+Click` remove) refined with He et al. Joint Guided Filter edge snapping.
5. **3D Cinematic Motion (`CinematicMotionDialog`):** Telea disocclusion background inpainting behind foreground masks + `cv2.remap` bicubic camera parallax profiles (*Dolly Zoom*, *Drift*, *Orbit*, *Pan*) exported to 1080p 30 FPS MP4.

---

## 8. Viewer Workflow, Save Controls & Grid Polish

- **Save to Original vs. Save as Copy Split Button:** Edit ribbon houses `SaveCopySplitContainer` (*Save in original location*, *Choose Location...*, *Options...*). `Options -> Edit` configures destination (original folder, custom path, or `"Ask each time"`), naming tokens (`{stem}_edited`, `{stem} (edited)`, `{stem}_copy`, `{stem}-edit`), collision suffixing (`_2`, `_3`), and `save_copy_switch_view` auto-switching.
- **WLPG Crop / Uncrop:** Entering Crop mode on previously cropped photos displays the full uncropped image with the existing crop rectangle adjustable outward or resettable via `"Full Image (Uncrop)"`.
- **Metadata & Face Coordinate Healing:** Spatial edits (crop, straighten, 90° rotate) re-project `[0, 1]` face boxes, purge cached thumbnails, preserve orientation consistency, and mark `metadata_dirty = 1` for XMP/EXIF sync.
- **Authentic Windows `Open with ▸` & Edit Nomenclature:** Inspects Windows Registry (`QFileIconProvider`) for registered apps, opens native `rundll32 shell32.dll,OpenAs_RunDLL`, standardizes **Colour** spelling, and maintains a self-healing thumbnail paint queue (`visible_media_ids(buffer_rows=4)` + decode error flag cleanup).
- **Single Media Viewer Context Actions & Video Frame Capture (`src/imaging/video_frame_capture.py`):** Unified photo right-click menu with **Flag ▸**, **Edit ▸**, **Scan ▸**, **Find Similar Photos...** (non-blocking progressive search), and **Compare** integration alongside non-destructive Fine Tune, Crop, Spot Fix, and Markup modes. Video playback adds **"Save still frame as photo"**, extracting the current full-resolution frame as a collision-safe JPEG in the same folder with offset EXIF capture date + GPS coordinates and immediate catalog indexing.

---

## 9. Verification & Automated Test Coverage

| Subsystem | Test Suite | Verified Coverage |
| :--- | :--- | :--- |
| **Core Edit & OpenCV** | [`tests/test_modern_opencv_engine.py`](../tests/test_modern_opencv_engine.py), [`tests/test_modern_editor_pipeline.py`](../tests/test_modern_editor_pipeline.py), [`tests/test_raw_fastpath_and_finetune_ux.py`](../tests/test_raw_fastpath_and_finetune_ux.py) | LUT parity, exposure shoulder/toe, AgX highlight chroma roll-off, CIE LAB ambience, vibrance skin locus, U-Point, auto-straighten, RAW embedded JPEG fast-path, focus-aware sliders & RGB histogram |
| **Markup & Callouts** | [`tests/test_markup_ribbon_and_video_immersive.py`](../tests/test_markup_ribbon_and_video_immersive.py), [`tests/test_markup_wysiwyg_and_tools.py`](../tests/test_markup_wysiwyg_and_tools.py), [`tests/test_markup_interactive_move_and_crop.py`](../tests/test_markup_interactive_move_and_crop.py) | Contextual Markup Ribbon, Select/Alt-hold, 9 tools, WYSIWYG inline text, shapes/lines/obscure/stickers/stamp, ROI full-res baking, `PixelArrayUndoCommand`, `.OriginalFiles` save & clear |
| **Spot Fix & Brush** | [`tests/test_spot_fix_consolidation_and_brush_scale.py`](../tests/test_spot_fix_consolidation_and_brush_scale.py) | Frequency separation transfer, 1:1 cursor view scale, `[`/`]` shortcuts, dual-mode red eye, `Esc` cascade |
| **Depth & Light** | [`tests/test_depth_light_and_save_workflow.py`](../tests/test_depth_light_and_save_workflow.py) | `DepthAndMatteEngine` proxy/master cache, bokeh, relight, haze, Save vs Save as Copy, dock bindings |
| **Create Studio** | [`tests/test_create_tools.py`](../tests/test_create_tools.py), [`tests/test_create_tools_overhaul.py`](../tests/test_create_tools_overhaul.py), [`tests/test_photo_fuse_and_ribbon_effects_fixes.py`](../tests/test_photo_fuse_and_ribbon_effects_fixes.py), [`tests/test_creative_and_markup.py`](../tests/test_creative_and_markup.py) | Panorama inscribed crop, Photo Fuse landmark alignment & Laplacian blending, Base Shot caching & UI, Collage face safety, Colour Pop, 3D Motion remap |
| **Orientation, Codecs & Ribbon** | [`tests/test_nextgen_codecs.py`](../tests/test_nextgen_codecs.py), [`tests/test_thumbnail_orientation_and_markup_fit.py`](../tests/test_thumbnail_orientation_and_markup_fit.py), [`tests/test_wp07_drag_openwith_ribbon.py`](../tests/test_wp07_drag_openwith_ribbon.py), [`tests/test_perspective_and_crop_ribbon.py`](../tests/test_perspective_and_crop_ribbon.py) | HEIC/AVIF/JXL + Display P3 ICC + magic-byte sniffing, post-edit orientation preservation, Windows `Open with`, contextual Crop Ribbon & keystone perspective, configurable Auto Adjust |
