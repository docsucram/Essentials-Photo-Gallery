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
|   src/imaging/create_tools.py, src/imaging/image_loader.py)             |
|  - Non-destructive PhotoAdjustmentRecipe parametric stack               |
|  - Precomputed sRGB <-> Linear LUTs & unified 256-entry cv2.LUT tonal   |
|  - Monocular 3D depth maps & portrait alpha matte proxy caching         |
|  - Next-gen codec decoding & LittleCMS2 Display P3 -> sRGB conversion   |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                    Computer Vision & AI Runtime Layer                   |
|  - OpenCV (cv2): SIFT/ORB, Homography, Telea/NS Inpainting, CIE LAB     |
|  - ONNX Runtime: Depth-Anything-V2 / MiDaS & MODNet / RMBG matting      |
|  - NumPy: Vectorized floating-point / 4x hierarchical Laplacian math    |
|  - Pillow (PIL): Next-gen codecs (HEIC/AVIF/JXL), ICC & EXIF handling   |
+-------------------------------------------------------------------------+
```

### Core Performance & Fidelity Guarantees
1. **Two-Tier Display Proxies (`scrub_proxy_rgb` & `display_proxy_rgb`):** Active slider scrubbing (`_is_sliding`) evaluates against a fast `960px` `scrub_proxy_rgb` with `16ms` timer coalescing (`_ft_timer`, `_dl_timer` in `right_tag_dock.py`), snapping immediately to the crisp `1280px` `display_proxy_rgb` on release.
2. **Background Master Commit & Dirty-Stage Cache (`_MasterRenderRunnable` & `_STAGE_CACHE`):** Releasing sliders schedules a debounced (`250ms` `_master_render_timer`) full-resolution `master_np_rgb` pass on `QThreadPool` with generation-counter cancellation (and synchronous `ensure_master_rendered()` before Save/Export). `render_adjusted_image()` caches intermediate `dd_img`, `tl_img`, and `bokeh_data` checkpoints per `(media_id, H, W)` so downstream sliders skip upstream passes.
3. **Precomputed Look-Up Tables (`cv2.LUT`):** Levels, Filmic Exposure, and Filmic Contrast combine into a single 256-entry `uint8` LUT executing in **`1.1 ms`**, paired with precomputed sRGB/linear (`_LUT_SRGB_TO_LIN`, `_LUT_LIN_TO_SRGB`, `_VIBRANCE_POW_LUT`) and 4x downsampled base/detail illumination decomposition.
4. **Non-Destructive File Protocol (`.OriginalFiles/`):** Saving over an original file stashes a byte-identical untouched copy inside `<folder>/.OriginalFiles/<filename>` (never overwriting an existing backup), enabling exact **Revert to original** and preserving EXIF/XMP blocks and spatial face coordinates.

---

## 2. Parametric Recipe Stack (`PhotoAdjustmentRecipe`)

Defined in [`src/imaging/opencv_engine.py`](../src/imaging/opencv_engine.py), all non-destructive tonal, color, geometry, and 3D depth controls serialize cleanly:

```python
@dataclass
class PhotoAdjustmentRecipe:
    # Tone & Exposure
    brightness: float = 0.0          # -100.0 .. +100.0 (Filmic shoulder roll-off)
    contrast: float = 0.0            # -100.0 .. +100.0 (Filmic S-curve on L*)
    shadows: float = 0.0             # -100.0 .. +100.0 (Edge-aware shadow lift/crush)
    highlights: float = 0.0          # -100.0 .. +100.0 (Highlight recovery)
    clarity: float = 0.0             # -100.0 .. +100.0 (Midtone local contrast)
    black_point: float = 0.0         # 0.0 .. 255.0 (Histogram shadow clip)
    white_point: float = 255.0       # 0.0 .. 255.0 (Histogram highlight clip)

    # Color & Atmospheric
    temperature: float = 0.0         # -100.0 .. +100.0 (Cool Blue <-> Warm Amber)
    tint: float = 0.0                # -100.0 .. +100.0 (Green <-> Magenta)
    saturation: float = 0.0          # -100.0 .. +100.0 (Skin-safe vibrance)
    dehaze: float = 0.0              # -100.0 .. +100.0 (Dark channel prior recovery)
    vignette: float = 0.0            # -100.0 .. +100.0 (Elliptical corner shading)

    # Detail & Geometry
    straighten_deg: float = 0.0      # -15.0 .. +15.0 degrees
    sharpen: float = 0.0             # 0.0 .. 100.0 (Luminance unsharp mask)
    denoise: float = 0.0             # 0.0 .. 100.0 (Dual-domain bilateral filter)
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

### 3.1 Dual-Domain Denoise & MAD Noise Estimation
Laplacian MAD noise estimate:
$$\hat{\sigma} = \frac{\text{median}(|\nabla^2 I - \text{median}(\nabla^2 I)|)}{0.6745}$$
Resolution-scaled bilateral smoothing runs with $\text{res\_scale} = \max(W, H) / 1280.0$ across luminance ($L^*$, stronger detail protection $\sigma_c = 0.45 \cdot \text{denoise}$) and chrominance ($A^*, B^*$, wider $\sigma_c = 0.85 \cdot \text{denoise}$) to remove color mottling without blurring edges.

### 3.2 Filmic Exposure, Contrast, Levels & Ambience
- **Levels Stretching:** $f_{\text{out}} = \text{clip}\left(\frac{f_{\text{in}} - bp/255}{wp/255 - bp/255}, 0, 1\right)$.
- **Positive Exposure Shoulder:** $f_{\text{scaled}} = f \cdot (1 + 1.4 b)$, asymptotically compressed above $\text{knee} = 0.65$ via $f_{\text{compressed}} = \text{knee} + \frac{\Delta}{1 + 1.8\Delta}$ to avoid harsh highlight clipping.
- **Negative Exposure Toe:** Attenuates in stops ($2^{1.5 b}$) with subtle shadow toe lift $0.05 |b| f (1 - f)$ preserving deep shadow detail.
- **CIE LAB Shadows/Highlights + Ambience:** Bilateral base/detail separation ($L = L_{\text{base}} + L_{\text{detail}}$) lifts dark regions ($L_{\text{base}}/255 < 0.65$) while boosting micro-contrast on $L_{\text{detail}}$ so lifted shadows stay punchy rather than muddy.
- **Clarity:** Midtone bell-curve weight $w_{\text{mid}} = \text{clip}(1 - 4(L_{\text{base}}/255 - 0.5)^2, 0, 1)$ modulates $L_{\text{detail}}$ selectively in midtones.

### 3.3 Color Temperature, Tint & Skin-Safe Vibrance in CIE LAB
- **Temperature ($B^*$) & Tint ($A^*$):** Orthogonal chromatic offsets in perceptual LAB space.
- **Skin-Safe Vibrance (Zero HSV Round-Trip):** Measures Euclidean chroma $C = \sqrt{(A^*-128)^2 + (B^*-128)^2}$ and proximity to the warm human skin locus ($A^* \approx 146, B^* \approx 152$):
  $$W_{\text{vibrance}} = \left(1 - \left(\frac{C}{128}\right)^{0.8}\right) \cdot \left(1 - 0.6 \cdot \text{clip}(1 - D_{\text{skin}}, 0, 1)\right)$$
  Boosting muted blues/greens cleanly without turning portraits orange.

### 3.4 Atmospheric Dehaze & Halo-Free Sharpening
- **Koschmieder Dark Channel Prior:** Estimates atmospheric light $A$ (99th percentile luminance $\ge 100$) and refines the transmission map $t(x)$ with a bilateral filter before recovering $J(x) = \frac{I(x) - A}{t(x)} + A$ without sky/horizon halos.
- **Luminance Unsharp Mask:** Applies $\pm 3.0$ dead-zone soft thresholding on $d = L - G_{\sigma}*L$ so sensor grain isn't amplified.

### 3.5 Analytical Inscribed-Rectangle Auto-Straighten & U-Point Selective Controls
- **Zero-Border Auto-Straighten:** Rotates by $|\theta| \le 15^\circ$ and analytically crops to the maximum inscribed rectangle matching original aspect ratio $R = W_0 / H_0$:
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

### 5.1 Unified Spot Fix (`src/ui/media_viewer.py`)
- **Mutual-Exclusion Modes:** Compact floating toolbar toggles **Blemish** healing and **Red eye** reduction sharing a continuous $6\text{px}..80\text{px}$ slider and `[` / `]` keyboard sizing.
- **Exact 1:1 View-Scale Geometry:** Brush radius maps dynamically through graphics transform scale $M_{11}$ ($r_{\text{scene}} = r_{\text{screen}} / \text{scale}$, $r_{\text{master}} = \max(2, \text{round}(r_{\text{scene}}))$) paired with a dual-ring high-contrast cursor.
- **Frequency-Separation Texture Transfer (Blemish):** Splits local illumination via Gaussian blur ($L_{\text{low}}$ inpainted via Telea), extracts high-frequency pores/grain ($I_{\text{high}} = I - I_{\text{low}}$), copies unblemished texture from an 8-direction surrounding donor ring via `cv2.warpAffine`, variance-normalizes ($\sigma_{\text{donor}} / \sigma_{\text{transferred}}$), and recombines cleanly without smudging.
- **Dual-Mode Red Eye:** Connected-component pupil detection (`r_ratio >= 1.25`, morphological closing to protect white corneal glints, neutral dark replacement) triggered either by brush click/stroke or legacy box drag.

### 5.2 WYSIWYG Markup & Privacy Redaction (`src/ui/markup_overlay.py`)
- **Eight Tools:** Freehand Pen, Highlighter (alpha-isolated so overlapping strokes never darken), Dynamic Vector Arrow, Shape (Box / Oval / 5-point Star dropdown), Privacy Blur ($\sigma = 24$), Mosaic Pixelation ($16\text{px}$ blocks), `2.5x` Glass Magnifier Loupe (`1.5x..4.0x` zoom stepper), and Multi-line Inline Text (`Enter` commits, `Shift+Enter` newline).
- **Photoshop-Style Color & Undo:** Overlapping Primary / Secondary color swatches (`⇄` / `X` swap; Left-click sets foreground/text, Right-click sets frosted pill background). `PixelArrayUndoCommand` stores only modified bounding-box slices (`<15 KB` vs full 24MP frames), saving `>95%` undo RAM.
- **Esc Cascade & Boundaries:** Floating toolbars (`_DraggableFloatingToolbar`, `MarkupFloatingToolbar`) use borderless antialiased `_DraggableGripHandle` dots, clamp inside viewport with $10\text{px}$ margins, and unwind cleanly via an 8-level `Esc` cascade.

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

1. **ICE-Style Panorama Stitcher (`PanoramaDialog`):** Background `PanoramaWorker` loads EXIF-safe frames and runs SIFT (`2,500` keypoints) + FLANN Lowe's ratio ($0.75$) / cylindrical bundle adjustment or affine scan homography with OpenCL disabled (`cv2.ocl.setUseOpenCL(False)`) and a `2400px` input long-edge cap for stability, plus exposure compensation, multi-band Laplacian feathering, Navier-Stokes/Telea corner gap fill, and **Largest Interior Rectangle Auto-Crop** (`find_largest_interior_rectangle`) via 2D histogram monotonic stacks.
2. **Interactive Collage & "Then & Now" (`CollageDialog`):** Smart hero face slot protection (`>94%` opacity lock, $40\%$ upward headroom crop) guaranteeing $100\%$ gapless grid/masonry coverage (1–35 photos), tactile **Polaroid Table Scatter** with drop shadows and date/caption stamping, and chronological **Then & Now** comparison cards.
3. **Photo Fuse (`PhotoFuseDialog`):** Background `_PhotoFuseWorker` loads frames, aligns burst shots via SIFT/ORB + RANSAC homography ($\ge 6$ inliers, determinant sanity gate, and warning banner for unalignable shots), and detects faces with a CPU-only InsightFace detection session (`allowed_modules=["detection"]`, `det_score >= 0.55`, 5-point landmark gate, strict Haar fallback). Features a visual Base Shot thumbnail picker with per-base caching (`_base_cache`), per-face local alignment (`_align_face_patch_local` via 5-point landmark `estimateAffinePartial2D` or Euclidean ECC fallback), mask-aware CIE LAB color matching, multi-band Laplacian pyramid blending (up to 5 levels), a **Show face boxes** toggle, and a **View Original Base** comparison banner.
4. **Colour Pop (`ColourPopDialog`):** Subject mask seeded from face detections or center saliency + Photoshop-style multi-color eyedropper (`Click` set, `Shift+Click` add, `Right/Alt+Click` remove) refined with He et al. Joint Guided Filter edge snapping.
5. **3D Cinematic Motion (`CinematicMotionDialog`):** Telea disocclusion background inpainting behind foreground masks + `cv2.remap` bicubic camera parallax profiles (*Dolly Zoom*, *Drift*, *Orbit*, *Pan*) exported to 1080p 30 FPS MP4.

---

## 8. Viewer Workflow, Save Controls & Grid Polish

- **Save to Original vs. Save as Copy Split Button:** Edit ribbon houses `SaveCopySplitContainer` (*Save in original location*, *Choose Location...*, *Options...*). `Options -> Edit` configures destination (original folder, custom path, or `"Ask each time"`), naming tokens (`{stem}_edited`, `{stem} (edited)`, `{stem}_copy`, `{stem}-edit`), collision suffixing (`_2`, `_3`), and `save_copy_switch_view` auto-switching.
- **WLPG Crop / Uncrop:** Entering Crop mode on previously cropped photos displays the full uncropped image with the existing crop rectangle adjustable outward or resettable via `"Full Image (Uncrop)"`.
- **Metadata & Face Coordinate Healing:** Spatial edits (crop, straighten, 90° rotate) re-project `[0, 1]` face boxes, purge cached thumbnails, preserve orientation consistency, and mark `metadata_dirty = 1` for XMP/EXIF sync.
- **Authentic Windows `Open with ▸` & Edit Nomenclature:** Inspects Windows Registry (`QFileIconProvider`) for registered apps, opens native `rundll32 shell32.dll,OpenAs_RunDLL`, standardizes **Colour** spelling, and maintains a self-healing thumbnail paint queue (`visible_media_ids(buffer_rows=4)` + decode error flag cleanup).

---

## 9. Verification & Automated Test Coverage

| Subsystem | Test Suite | Verified Coverage |
| :--- | :--- | :--- |
| **Core Edit & OpenCV** | [`tests/test_modern_opencv_engine.py`](../tests/test_modern_opencv_engine.py), [`tests/test_modern_editor_pipeline.py`](../tests/test_modern_editor_pipeline.py) | LUT parity, exposure shoulder/toe, CIE LAB ambience, vibrance skin locus, U-Point, auto-straighten |
| **Markup & Callouts** | [`tests/test_markup_wysiwyg_and_tools.py`](../tests/test_markup_wysiwyg_and_tools.py), [`tests/test_markup_interactive_move_and_crop.py`](../tests/test_markup_interactive_move_and_crop.py) | 8 tools, inline text WYSIWYG, shape swap, dual color swatches, `PixelArrayUndoCommand`, `.OriginalFiles` save & clear |
| **Spot Fix & Brush** | [`tests/test_spot_fix_consolidation_and_brush_scale.py`](../tests/test_spot_fix_consolidation_and_brush_scale.py) | Frequency separation transfer, 1:1 cursor view scale, `[`/`]` shortcuts, dual-mode red eye, `Esc` cascade |
| **Depth & Light** | [`tests/test_depth_light_and_save_workflow.py`](../tests/test_depth_light_and_save_workflow.py) | `DepthAndMatteEngine` proxy/master cache, bokeh, relight, haze, Save vs Save as Copy, dock bindings |
| **Create Studio** | [`tests/test_create_tools.py`](../tests/test_create_tools.py), [`tests/test_create_tools_overhaul.py`](../tests/test_create_tools_overhaul.py), [`tests/test_photo_fuse_and_ribbon_effects_fixes.py`](../tests/test_photo_fuse_and_ribbon_effects_fixes.py), [`tests/test_creative_and_markup.py`](../tests/test_creative_and_markup.py) | Panorama inscribed crop, Photo Fuse landmark alignment & Laplacian blending, Base Shot caching & UI, Collage face safety, Colour Pop, 3D Motion remap |
| **Orientation, Codecs & Ribbon** | [`tests/test_nextgen_codecs.py`](../tests/test_nextgen_codecs.py), [`tests/test_thumbnail_orientation_and_markup_fit.py`](../tests/test_thumbnail_orientation_and_markup_fit.py), [`tests/test_wp07_drag_openwith_ribbon.py`](../tests/test_wp07_drag_openwith_ribbon.py) | HEIC/AVIF/JXL + Display P3 ICC, post-edit orientation preservation, Windows `Open with`, Edit ribbon & Crop/Save UI |
