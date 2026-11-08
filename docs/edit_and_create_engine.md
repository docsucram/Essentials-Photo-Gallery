# Edit & Create Engine — Technical Architecture & Implementation Review

## 1. Architectural Overview

The **Edit & Create Engine** powers the professional editing, creative synthesis, and depth-aware photo manipulation capabilities in Essentials Photo Gallery. The subsystem consists of three tightly integrated layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          UI & Viewer Layer                             │
│  (src/ui/media_viewer.py, src/ui/right_tag_dock.py,                   │
│   src/ui/create_studio_dialogs.py & src/ui/ribbon_toolbar.py)         │
│  - Interactive single image viewer with 60 FPS display proxy (~1280px) │
│  - Side-dock Fine Tune & Depth & Light controls with Aero sliders      │
│  - Smart Unsaved-Edits Guard: Save to Original vs Save as Copy         │
│  - Interactive Create Studio dialogs with zoom bars and compare modes  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     Imaging Pipeline Layer                             │
│  (src/imaging/opencv_engine.py, src/imaging/creative_effects.py,       │
│   src/imaging/create_tools.py)                                         │
│  - Non-destructive floating-point recipe stack (PhotoAdjustmentRecipe)│
│  - Precomputed sRGB <-> Linear floating-point LUTs (0.01ms lookup)     │
│  - Real-time display proxy evaluation with asynchronous master commit  │
│  - Monocular 3D depth and hair-accurate alpha matting cache            │
│  - High-resolution final export pipelines (24MP/48MP lossless)        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Computer Vision & Core AI Layer                      │
│  - OpenCV (cv2): SIFT/ORB, Homography, Inpainting, GrabCut, LAB, LUT   │
│  - ONNX Runtime: Depth-Anything-V2 / MiDaS and MODNet / RMBG-1.4       │
│  - NumPy: Vectorized multi-channel floating-point arithmetic           │
│  - Pillow (PIL): File IO, scalable typography, EXIF preservation       │
│  - PySide6: Hardware-accelerated graphics scenes & 60 FPS coalescing   │
└────────────────────────────────────────────────────────────────────────┘
```

### Core Design Principles
1. **Interactive Display Proxies**: All non-destructive adjustments (exposure, color, curves, bokeh blur, relighting, and depth haze) run on a calibrated display proxy (bounded to 1920px for high-fidelity 1080p/1440p scrubbing). Sliders update fluidly at 30–60+ FPS without freezing the user interface.
2. **Lossless Master Commit**: Original camera files are never overwritten destructively. Edits saved to the original photo stash a byte-identical copy in `<folder>/.OriginalFiles/<filename>`. "Save as Copy" generates independent files leaving originals untouched. Full-resolution master rendering on slider release is debounced via a 250ms timer (`_master_render_timer`), ensuring sequential adjustments across multiple sliders stay 100% fluid with zero UI freezes.
3. **60 FPS Event Coalescing**: All slider inputs (Fine Tune & Depth & Light) are coalesced via high-resolution 16ms timers (`_ft_timer` and `_dl_timer` in `right_tag_dock.py`), guaranteeing zero event queue backlog when rapidly dragging sliders.
4. **Precomputed Look-Up Tables (LUTs)**:
   - Levels (Black/White Point), Filmic Exposure, and Filmic S-curve Contrast are combined into a single 256-element uint8 LUT applied with `cv2.LUT` (executing in **1.1 ms** instead of ~120 ms float math).
   - Costly floating-point power calculations ($x^{2.2}$, $x^{1/2.2}$, and vibrance $x^{0.8}$) are precomputed into static C-contiguous arrays (`_LUT_SRGB_TO_LIN`, `_LUT_LIN_TO_SRGB`, `_VIBRANCE_POW_LUT`).
   - 4x hierarchical base/detail decomposition accelerates Gaussian blur for shadows/highlights/clarity by 14x.

---

## 2. Technical Specification — Edit Engine

### A. Non-Destructive Adjustment Recipe (`PhotoAdjustmentRecipe`)
Defined in [`src/imaging/opencv_engine.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/imaging/opencv_engine.py). Every image adjustment is parameterized as a lightweight dataclass:
- **Exposure & Tone**: `brightness` (filmic shoulder roll-off), `contrast` (filmic S-curve), `shadows` (edge-aware recovery), `highlights` (specular suppression), `clarity` (midtone local contrast).
- **Color & White Balance**: `temperature` (cool blue $\leftrightarrow$ warm amber), `tint` (green $\leftrightarrow$ magenta), `saturation` (skin-safe vibrance), `dehaze` (dark-channel prior haze removal).
- **Detail & Optics**: `sharpen` (halo-free unsharp mask with dead-zone), `denoise` (dual-domain bilateral filter), `vignette` (aspect-aware elliptical fall-off), `straighten_deg` (analytical inscribed rectangle rotation).
- **3D Depth & Light**: `depth_blur` (optical defocus), `focal_plane` (depth plane selection), `bokeh_glow` (specular highlight bloom), `relight_near` / `relight_far` (near/distant fill lighting), `warmth_near` / `warmth_far` (near/distant color temperature), `depth_haze` / `depth_haze_warmth` (exponential Beer-Lambert depth fog).

### B. Mathematical Formulations & Algorithms

#### 1. Linear Exposure & Filmic Soft-Knee Compression
$$I_{\text{lin}} = \left( \frac{I}{255} \right)^{2.2}$$
$$\text{gain} = 2.0^{\frac{\text{brightness}}{35.0}}$$
Highlights are rolled off smoothly using a soft-knee shoulder function to prevent harsh clipping:
$$I_{\text{scaled}} = \begin{cases} I_{\text{lin}} \cdot \text{gain} & \text{if } I_{\text{lin}} \cdot \text{gain} \le T \\ 1.0 - (1.0 - T) \cdot \exp\left( -\frac{I_{\text{lin}} \cdot \text{gain} - T}{1.0 - T} \right) & \text{if } I_{\text{lin}} \cdot \text{gain} > T \end{cases}$$
where $T = 0.72$ is the threshold transition point.

#### 2. Highlight & Shadow Recovery via CIE LAB Ambience
The image is transformed into perceptual CIE $L^*a^*b^*$ color space. Edge-preserving bilateral decomposition splits luminance $L$ into base illumination $L_{\text{base}}$ and detail texture $L_{\text{detail}}$:
- **Shadow Lift**: $W_{\text{shadow}} = \text{clip}\left( 1.0 - \frac{L_{\text{base}}}{128.0}, 0.0, 1.0 \right)^2$
- **Highlight Recovery**: $W_{\text{highlight}} = \text{clip}\left( \frac{L_{\text{base}} - 128.0}{127.0}, 0.0, 1.0 \right)^2$
- $L^*_{\text{new}} = L^* + \Delta_{\text{shadow}} \cdot W_{\text{shadow}} + \Delta_{\text{highlight}} \cdot W_{\text{highlight}} + L_{\text{detail}}$

#### 3. Skin-Safe Vibrance Direct in LAB
Unlike naive saturation that oversaturates human faces, vibrance selectively protects warm skin tones:
$$\text{Chroma } C = \sqrt{(a^* - 128)^2 + (b^* - 128)^2}$$
$$\text{Skin Distance } D_{\text{skin}} = \sqrt{ \left( \frac{a^* - 146}{14} \right)^2 + \left( \frac{b^* - 152}{18} \right)^2 }$$
$$W_{\text{vibrance}} = \left( 1.0 - \left( \frac{C}{128} \right)^{0.8} \right) \cdot \left( 1.0 - 0.6 \cdot \text{clip}(1.0 - D_{\text{skin}}, 0, 1) \right)$$
Chroma is expanded proportionally to $W_{\text{vibrance}}$, ensuring natural skin tones remain lifelike while dull skies and foliage receive vivid saturation.

### C. Spot Fix Subsystem: Retouch & Red Eye Unification

The interactive canvas repair tools are consolidated under a single, unified **Spot Fix** paradigm in [`src/ui/media_viewer.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/media_viewer.py):

#### 1. Unified Floating Toolbar
- **Tool Toggles**: Removed verbose text instructions ("drag over blemishes to heal") in favor of compact, mutual-exclusion buttons: **Blemish** (inpainting) and **Red eye** (pupil correction).
- **Shared Brush Sizing**: A single continuous slider controls brush size ($6\text{px}$ to $80\text{px}$).
- **Keyboard Shortcuts**: Photoshop-standard bracket keys (`[` to decrease brush size, `]` to increase) allow instant adjustment without mouse diversion.
- **Unified Exit**: `Done (Esc)` safely terminates either tool, restores standard scroll/hand drag navigation, and unsets custom cursors.

#### 2. Exact 1:1 View-Scale Brush Radius Geometry
When viewing high-resolution photographs fitted or zoomed inside the viewport, view scale $\text{scale} = \mathbf{M}_{11}$ (ratio of viewport pixels to scene/master pixels). To eliminate selection vs. cursor discrepancies:
$$r_{\text{screen}} = \text{self.retouch\_radius}$$
$$r_{\text{scene}} = \frac{r_{\text{screen}}}{\text{scale}}$$
$$r_{\text{master}} = \max\left(2, \text{round}(r_{\text{scene}})\right)$$
- The screen cursor circle of radius $r_{\text{screen}}$ perfectly circumscribes the scene preview item of radius $r_{\text{scene}}$.
- The bitmap mask applied to `master_np_rgb` uses $r_{\text{master}}$, guaranteeing that what the user sees under the cursor is identical down to the individual pixel.

#### 3. Dual-Ring High-Visibility Cursor
Constructed with anti-aliasing directly in memory:
- Outer shadow ring ($\text{RGBA}(0, 0, 0, 210)$) provides contrast against bright highlights.
- Inner ring provides contrast over deep shadows (amber/gold for blemish healing, cyan for red eye).
- Center targeting dot provides pixel-precise alignment (white for blemish, red for red eye).

#### 4. Dual-Mode Red Eye Repair
- **Brush Click/Stroke**: Single-clicking or brushing over a red pupil isolates the connected pupil component, closes specular reflections via morphological closing, and feathers the replacement using Gaussian falloff.
- **Box Dragging**: Dragging a marquee rectangle over an eye provides backward-compatible area targeting for larger or angled faces.
- **Skin Preservation**: Skin tones outside the pupil reflection are completely protected ($\alpha = 0$).

#### 5. True Frequency Separation Texture Transfer
Blemish healing decomposes the region of interest into separate frequency domains to avoid unnatural smudging:
- **Low-Frequency Base Decomposition**: Gaussian blur ($\sigma \in [2.0, 8.0]$ scaled to brush radius) captures underlying skin illumination, tone gradients, and shadows without high-frequency pore detail. Telea inpainting interpolates the low-frequency base seamlessly across the blemish.
- **High-Frequency Detail Extraction**: $I_{\text{high}} = I - I_{\text{low}}$ isolates microscopic skin grain, authentic pores, and fabric weave.
- **Multi-Directional Donor Ring Texture Transfer**: An unblemished donor ring surrounding the blemish is sampled across 8 cardinal and diagonal directions offset by the blemish bounding box. Shifts are translated via `cv2.warpAffine` (eliminating toroidal wrap artifacts) with collision checking against the blemish mask.
- **Variance Normalization & Recombination**: The transferred high frequencies are normalized to match the donor ring's standard deviation:
  $$\hat{I}_{\text{high}} = I_{\text{transferred}} \cdot \frac{\sigma_{\text{donor}}}{\sigma_{\text{transferred}}}$$
  The healed low-frequency base is recombined with the normalized high-frequency texture, producing invisible repairs with 100% authentic micro-texture preservation.

#### 6. Draggable Floating Toolbars & Boundary Clamping
Both the Spot Fix toolbar (`_DraggableFloatingToolbar`) and Markup toolbar (`MarkupFloatingToolbar`) feature:
- Ergonomic grip handles (`⋮⋮`) with `SizeAllCursor` drag interaction.
- Strict boundary clamping inside the viewer viewport ($10\text{px}$ inset margin), preventing toolbars from drifting offscreen during window resize.
- Persistent user positioning flag (`_user_positioned`) ensuring user-placed toolbar coordinates are honored during zoom and pan operations.

### D. WYSIWYG Quick Markup & Callout Subsystem

The Markup tool in [`src/ui/markup_overlay.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/markup_overlay.py) and [`src/ui/media_viewer.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/media_viewer.py) provides a professional, real-time WYSIWYG annotation, callout, and redaction environment:

#### 1. Eight Specialized Markup Tools
- **Freehand Pen**: Smooth anti-aliased vector paths with circular brush cursor matching exact stroke width and selected color.
- **Fluorescent Highlighter**: Uniform opacity rendering using an isolated alpha mask to prevent self-intersection darkening/bleeding, simulating authentic chisel highlighter across photo and document details.
- **Dynamic Vector Arrow**: Analytical geometry calculation (`build_arrow_geometry`) rendering a proportional arrowhead and shaft dynamically following mouse drag in real time.
- **Rectangle / Bounding Box**: Crisp rectangular outline callout for highlighting regions of interest.
- **Interactive Privacy Blur**: Rubberband drag selection instantly replaced with a live, Gaussian-blurred image patch on the canvas (`apply_blur_redact`).
- **Interactive Mosaic Pixelate**: Drag selection instantly replaced with an authentic block-mosaic pixelated patch (`apply_pixelate`).
- **Magnify Callout (WYSIWYG Lens)**: Interactive magnifying tool with clean glass chrome ring (no distracting labels) and a live hovering lens preview tracking the cursor. Uses a precision crosshair cursor to eliminate conflicting double circles, with screen diameter expanded from 90px to 430px; single-click stamps the magnified callout directly onto the photo. Includes a magnification power stepper button (`1.5x` to `4.0x`) and mouse wheel magnification control.
- **Multi-line Inline Text Tool**: On-canvas `QTextEdit` spawned at click coordinates with live font scaling, color binding, auto-sizing text pill, bold (`B`) and italic (`I`) toggle buttons, and multi-line support (`Enter` commits, `Shift+Enter` inserts newlines, `Esc` cancels).

#### 2. Floating Aero Toolbar (`MarkupFloatingToolbar`)
- **Tool Switcher**: Mutual-exclusion tool buttons with Aero hover and selection glows (Pen, Highlight, Arrow, Box, Blur, Pixelate, Magnify, Text).
- **Contextual Tool Controls**: Magnifier zoom stepper (`btn_magnify_zoom`) and text formatting toggles (`btn_bold`, `btn_italic`) automatically display only when their respective tools are active.
- **Stroke Sizing & Photoshop Shortcuts**: Continuous slider (2px to 36px) with dynamic label ("Size:" or "Font:") and `[` / `]` hotkeys that dynamically scale the circular cursor on canvas to match view scale.
- **Active Color Indication**: Dedicated 18x18px active color preview chip showing current color, paired with prominent active selection dot markers on preset swatches.
- **Dedicated Save Button**: Integrated "Save" button directly on the floating toolbar with instant visual feedback ("Saved!") creating pristine backups in `.OriginalFiles`.

#### 3. Status Bar Hints & Hierarchical Escape Chain
- **Status Bar Integration**: Mode switches (Crop, Spot Fix Blemish, Spot Fix Red Eye, Markup) broadcast actionable shortcuts and usage tips to the main window status bar via `status_message_requested`.
- **Hierarchical `Esc` Event Cascade**: Esc sequentially unwinds user activity without premature viewer dismissal:
  1. Active inline text input $\rightarrow$ Cancel text edit
  2. Active markup stroke $\rightarrow$ Cancel stroke in progress
  3. Active retouch stroke $\rightarrow$ Cancel brush drag
  4. Active crop mode $\rightarrow$ Cancel crop rectangle
  5. Active Spot Fix (retouch or red eye) mode $\rightarrow$ Close Spot Fix toolbar
  6. Active Markup mode $\rightarrow$ Close Markup toolbar
  7. Active Face Adjust mode $\rightarrow$ Exit Face Adjust
  8. Base view $\rightarrow$ Close Single Viewer and return to grid navigation
- **Mutual Exclusion Guard**: Opening Spot Fix (Blemish / Red Eye), Crop, or Face Tagging automatically exits Markup mode and closes the toolbar; entering Markup similarly dismisses all other active editing modes.

#### 3. High-Efficiency Undo/Redo Engine & Direct RGB Pipeline
- **Bounding Box Patch Undo**: [`PixelArrayUndoCommand`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/media_viewer.py) calculates the exact bounding box slice of modified pixels `(y1:y2, x1:x2)`, storing only localized patches (<15 KB per stroke instead of full 72 MB 24MP frames), cutting undo stack RAM usage by over 95%.
- **Zero Color Conversion Bounces**: [`flatten_markup_onto_rgb_image`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/markup_overlay.py) operates directly on the contiguous RGB master buffer wrapped in `Format_RGB888`, eliminating redundant RGB $\leftrightarrow$ BGR conversions on stroke commit.
- Fully respects the application-wide undo queue: `Ctrl+Z` immediately undos individual strokes, `Ctrl+Y` redoes them, and the ribbon/toolbar Undo and Redo buttons reflect state dynamically.
- **Clear vs. Done Separation**: Clicking "Clear" rolls back all markups performed in the active session to the session baseline; closing markup mode ("Done" or Esc) cleanly cleans up temporary preview items while preserving all committed strokes.
- **Unsaved Edits Guard & Revert to Original**: Uncommitted markups are recognized by `has_unsaved_adjustments()`. Both the Ribbon "Save" (`Ctrl+S`) and "Revert to original" buttons are enabled. "Revert to original" reloads the pristine file from disk, clearing all uncommitted markups and resetting the canvas immediately.

---

## 3. 3D Depth & Light Subsystem (`creative_effects.py`)

### A. Dual Engine Architecture (ONNX + Fallback)
1. **Local ONNX Inference (`DepthAndMatteEngine`)**:
   - Monocular Depth: `depth_anything_v2_vits.onnx` / `midas_v21_small_256.onnx` outputs a continuous depth map $D(x, y) \in [0.0, 1.0]$ ($0.0 = \text{near}, 1.0 = \text{far}$).
   - Portrait Matting: `modnet.onnx` / `rmbg-1.4.onnx` outputs hair-accurate alpha matte $M(x, y) \in [0.0, 1.0]$.
2. **Zero-Dependency Fallback**:
   - Rapid Haar cascade face detection ($<5$ms) seeds a geometric trimap.
   - Saliency-informed GrabCut segments the subject.
   - Joint Guided Filtering snaps borders to pixel edges.
3. **Per-Media Proxy Caching**:
   - Depth and alpha maps are computed once at max 640px base resolution and stored in `_base_cache[media_id]`.
   - Scaling to display proxy or master resolution takes $<0.5$ms via bilinear resizing.

### B. Depth Effect Algorithms & High-Speed Implementations

#### 1. Vectorized Direct RGB Relighting
Near and far regions are lit independently without slow $RGB \leftrightarrow LAB$ round trips:
$$W_{\text{near}} = \max(M, 1.0 - D), \quad W_{\text{far}} = D \cdot (1.0 - M)$$
$$\Delta L = (\text{relight}_{\text{near}} \cdot 0.45) \cdot W_{\text{near}} + (\text{relight}_{\text{far}} \cdot 0.45) \cdot W_{\text{far}}$$
$$\Delta W = (\text{warmth}_{\text{near}} \cdot 0.35) \cdot W_{\text{near}} + (\text{warmth}_{\text{far}} \cdot 0.35) \cdot W_{\text{far}}$$
$$\begin{bmatrix} R \\ G \\ B \end{bmatrix}_{\text{lit}} = \begin{bmatrix} R + \Delta L + \Delta W \\ G + \Delta L + 0.2 \cdot \Delta W \\ B + \Delta L - \Delta W \end{bmatrix}$$
- **Execution Speed**: **14.1 ms** (71.1 FPS) at 1280px.

#### 2. Optical Bokeh Blur with Specular Highlight Bloom
- Focus depth $z_f = \text{focal\_plane} / 100.0$.
- Defocus amount: $\delta(x, y) = |D(x, y) - z_f| \cdot (1.0 - M(x, y) \cdot [z_f < 0.25])$.
- Defocus blur contains zero spatial frequencies above 640px. The linear buffer is downsampled to 640px, convolved via `cv2.GaussianBlur`, converted back to sRGB via `_LUT_LIN_TO_SRGB`, and upsampled to target dimensions.
- **Execution Speed**: **35.0 ms** at 1280px; **601 ms** at 24MP (down from 3,245 ms).

#### 3. Beer-Lambert Atmospheric Depth Haze
Transmittance modeled after physical optical scattering in mist/fog:
$$T(x, y) = 1.0 - \exp\left( -\beta \cdot D(x, y)^2 \right)$$
$$H(x, y) = T(x, y) \cdot (1.0 - 0.85 \cdot M(x, y))$$
$$I_{\text{haze}} = I + (C_{\text{mist}} - I) \cdot H$$
- **Execution Speed**: **26.0 ms** (38.4 FPS) at 1280px.

---

## 4. Technical Specification — Create Tools Engine

### Tool 1: Panoramic Photo Stitching (`PanoramaDialog`)
- **Keypoint Matching**: SIFT (2,500 points) with FLANN matcher and Lowe's ratio test ($d_1 / d_2 < 0.75$).
- **Stitching Core**: Cylindrical projection bundle adjustment with fail-safe fallback to pairwise RANSAC homography.
- **Gap Inpainting**: Boundary scallops ($<2.5\%$ area) are filled using Alexandru Telea fast marching (`cv2.INPAINT_TELEA`), followed by maximum inscribed rectangular bounding crop.

### Tool 2: Photo Fuse (`PhotoFuseDialog`)
- **Sub-Pixel Registration**: Source and destination face patches are aligned via feature homography warp.
- **CIE LAB Illumination Transfer**: Matches $\mu_L, \sigma_L, \mu_a, \sigma_a, \mu_b, \sigma_b$ to prevent flash or exposure shifts across seams.
- **Hermite Boundary Blending**: Smooth cubic Hermite polynomial feathering eliminates visible composite boundaries.

### Tool 3: Auto-Collage Engine (`CollageDialog`)
- **Blended Seamless Collage**:
  - Automatically identifies the most prominent face across photos and assigns it to **Slot 0 (Hero focal slot)**.
  - Headroom-preserving smart crop ($40\%$ upward offset from face center).
  - Face protection map locks facial regions to $>94\%$ opacity to prevent ghosting artifacts from background overlaps.
  - Guarantees $100\%$ canvas coverage without unassigned gaps for any selection from 1 to 35 photos.
- **Modern Grid / Masonry**: Tessellated layouts with anti-aliased rounded corners.
- **Polaroid Table Scatter**: Realistic 3.5" $\times$ 4.2" Polaroid cards with procedural mahogany/slate woodgrain and Gaussian drop shadows.
- **Then & Now**: Automatic chronological EXIF sequencing with archival vintage tone curve applied to the earlier photograph.

### Tool 4: Colour Pop Studio (`ColourPopDialog`)
- **Multi-Color Eyedropper with Photoshop Shortcuts**:
  - Click: Set primary color.
  - Shift + Click: Add color to active selection.
  - Alt/Ctrl/Right-Click: Remove color from selection.
- **Guided Filter Edge Matting**: He et al. Joint Guided Filtering snaps the Euclidean color distance mask to true physical object boundaries.
- **High-Contrast B&W Synthesis**: Desaturates and tone-maps unselected background pixels.

### Tool 5: 3D Cinematic Motion (`CinematicMotionDialog`)
- **Depth Mesh Warping**: Converts 2D stills into 3D video sequences with camera parallax via `cv2.remap` bicubic grid displacement.
- **Motion Profiles**: Dolly Zoom (Vertigo effect), Parallax Drift, 3D Orbit, and Cinematic Pan.
- **Export**: H.264 MP4 video encoding with even dimension constraints.

---

## 5. Save vs. Save as Copy Architecture

```
User Edits Photo in Single View
              │
              ├──► [Save to Original]
              │         │
              │         ├──► Backup original to <folder>/.OriginalFiles/<filename>
              │         ├──► Render master at 100% quality (JPEG quality from Options)
              │         ├──► Update file on disk & sync catalog
              │         └──► Enable "Revert to original" button in Ribbon
              │
              └──► [Save as Copy]
                        │
                        ├──► Check Photo Gallery Options (Edit Tab):
                        │      • Destination: Original folder vs. Custom folder
                        │      • Naming: <stem>_edited, <stem> (edited), <stem>_copy, <stem>-edit
                        │      • Collision protection: auto-increment suffix (_2, _3)
                        │      • Switch view: immediately load new copy into viewer
                        │
                        ├──► Render master to destination file
                        ├──► Scan folder & index new item in catalog
                        └──► Switch Single Viewer to new file (original untouched)
```

### Options Configuration (`src/ui/dialogs.py` - Tab 4: Edit)
- **Save destination**: Radio selection between original photo's folder or custom directory with browse picker.
- **File naming format**: Dropdown supporting `{stem}_edited`, `{stem} (edited)`, `{stem}_copy`, and `{stem}-edit`.
- **Post-save behavior**: "Switch to viewing the new copy after saving" checkbox (enabled by default).
- **Settings Persistence**: Stored via `QSettings("EssentialsPhotoGallery", "EssentialsPhotoGallery")`.

---

## 6. Verification & Test Coverage Matrix

All components of the Edit & Create Engine are validated via automated pytest suites:

| Test Suite | File | Key Assertions Verified |
| :--- | :--- | :--- |
| **Markup & Callout Engine** | [`tests/test_markup_wysiwyg_and_tools.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/tests/test_markup_wysiwyg_and_tools.py) | • Floating toolbar tools, stroke slider, color chips, and color picker<br>• Freehand pen and highlighter path generation and RGBA blending<br>• Dynamic vector arrow and bounding box coordinate calculation<br>• Real-time interactive privacy blur and mosaic pixelate patch items<br>• Anti-aliased 2.5x circular magnifier loupe creation, precision crosshair cursor, and expanded radius<br>• Direct on-canvas inline text typing, live font sizing & color, and Return commit<br>• Keyboard shortcuts (`[` / `]`, Enter, Ctrl+Z, Esc) and master undo stack commit<br>• Floating toolbar Save button with pristine backup to `.OriginalFiles`<br>• Toolbar Clear vs Done session commit separation and Revert to Original canvas wipe |
| **Depth & Light Engine** | [`tests/test_depth_light_and_save_workflow.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/tests/test_depth_light_and_save_workflow.py) | • `PhotoAdjustmentRecipe` default states<br>• `DepthAndMatteEngine` caching & proxy-to-master scaling<br>• `render_adjusted_image` depth bokeh, relighting, and haze<br>• Save vs Save as Copy backup and file creation<br>• Right tag dock Depth & Light mode UI binding |
| **Create Tools Engine** | [`tests/test_create_tools_overhaul.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/tests/test_create_tools_overhaul.py) | • Guided filter edge matting & subject trimap<br>• Color Pop multi-color sampling & B&W synthesis<br>• Portrait bokeh pyramid performance & halo prevention<br>• 3D Cinematic remap stability & aspect ratio<br>• Auto-collage 100% canvas coverage & face protection<br>• Panorama inscribed rectangular crop & illumination transfer |
| **Spot Fix & Brush Engine** | [`tests/test_spot_fix_consolidation_and_brush_scale.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/tests/test_spot_fix_consolidation_and_brush_scale.py) | • Unified Spot Fix toolbar with Blemish / Red eye toggles<br>• True Frequency Separation texture transfer blemish repair with multi-directional donor sampling<br>• 1:1 View-scale dynamic brush radius inverse mapping<br>• Photoshop `[` and `]` brush size keyboard shortcuts<br>• Circular brush-based red eye detection & desaturation<br>• Draggable floating toolbars with viewport boundary clamping<br>• Status bar tool switch hints and hierarchical escape chain |
| **Grid Thumbnail Pipeline** | [`tests/test_wlpg_ui_refinements.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/tests/test_wlpg_ui_refinements.py) | • High-density viewports (>64 visible thumbnails) scheduled immediately without arbitrary batch capping<br>• Decode error signal handling prevents leaking active flags into `_active_thumb_flags`<br>• Continuous pump loop on worker completion/error prevents blank square stalling at viewport end<br>• 3-row pre-buffer margins above and below viewport |
| **Ribbon, Open With & Drag UI** | [`tests/test_wp07_drag_openwith_ribbon.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/tests/test_wp07_drag_openwith_ribbon.py) | • Single view right-click context menu features "Open with ▸" at top with registered system app icons<br>• Genuine native Windows Open With dialog (`rundll32.exe shell32.dll,OpenAs_RunDLL`) on Choose another app<br>• Removal of Next/Previous actions from image context menu<br>• Edit ribbon full-width Spot Fix and Markup large button sizing (32px)<br>• Straighten stacked cleanly above British English "Colour" and Exposure |

---

## 7. Windows Live Photo Gallery & Photoshop Edit Tool Enhancements

### A. Authentic Windows System "Open with" Integration
1. **Registered System Application Icons**:
   - `get_windows_app_icon(app_name, app_cmd)` inspects Windows Registry keys (`App Paths`, `Applications\<cmd>\shell\open\command`, `shutil.which`) and extracts high-fidelity icons via `QFileIconProvider`.
   - The thumbnail context menu and single image view context menu render authentic application icons beside every registered photo and video application.
2. **Native Windows Dialog**:
   - Selecting "Choose another app..." invokes the genuine Windows system shell dialog via `rundll32.exe shell32.dll,OpenAs_RunDLL <file>`, displaying recommended Windows Store and desktop apps with full association options.
3. **Single Image View Context Menu**:
   - "Open with ▸" is placed prominently at the top of the context menu.
   - "Next" and "Previous" actions are removed to maintain a clean, distraction-free editing workflow.

### B. Edit Ribbon Layout & Nomenclature
1. **Large Full-Width Tool Buttons**:
   - "Spot fix" and "Markup" are promoted to large buttons (`large=True`, 32px vector icons) in wide viewports, providing immediate visual affordance.
2. **Stacked Adjustment Hierarchy**:
   - In the "Adjust" group, "Straighten" is positioned directly above "Colour" and "Exposure", grouping geometrical corrections before tonal adjustments.
3. **British English Nomenclature**:
   - "Color" is standardized to British English `"Colour"` across the Ribbon, menus, and status messages.

### C. Floating Toolbars UI Polish
1. **Button-Free Floating Drag Handle**:
   - Replaced framed button drag dots with `_DraggableGripHandle(QWidget)`, rendering two subtle columns of 4 antialiased dots (`#8595A8`, active blue `#3388E8` on hover) on a transparent, borderless background.
   - Refactored toolbar styles to use `_DraggableFloatingToolbar` and `#crop_toolbar` specific selectors, preventing stylesheet leakage to child widgets.
2. **High-Quality Vector Icons**:
   - Both Spot Fix and Markup floating toolbars feature crisp High-DPI icons across all tools: Pen, Highlighter, Arrow, Shape, Blur, Pixelate, Magnify, Text, Undo, Clear, Save, Done, and Close.

### D. Markup Photoshop-Style Tooling & WYSIWYG Parity
1. **WYSIWYG Inline Text Parity**:
   - On-canvas editor scales dynamically with the graphics view matrix (`view_scale = transform().m11()`), synchronizing screen font points, padding, and corner radii with master image pixels so text position and appearance during typing match the confirmed pill exactly.
2. **Photoshop-Style Shape Tool Swapping**:
   - The shape tool button includes a dropdown arrow with a popup menu allowing instant swapping between **Box** (Rectangle), **Oval** (Circle), and **Star** (5-point vector star), dynamically updating its active icon and text label.
3. **Photoshop-Style Dual Color Picker & Secondary Text Background**:
   - Overlapping Primary (foreground) and Secondary (background) color chips with a swap button (`⇄`, keyboard shortcut `X`).
   - Left-clicking a swatch sets the Primary colour; Right-clicking a swatch sets the Secondary colour.
   - Text callout uses the Secondary colour for the frosted pill background and the Primary colour for the font.


---

## 8. WLPG Crop / Uncrop, Save Split Button & Resilient Thumbnail Queue

### A. Authentic WLPG Crop / Uncrop & Re-adjust
1. **Interactive Crop Revelation**:
   - Entering Crop mode on an already-cropped image displays the complete, uncropped proxy image with the previous crop boundary pre-selected on top.
   - Users can expand handles outwards beyond existing bounds, reposition the crop rectangle, or select `"Full Image (Uncrop)"` from the aspect ratio presets.
   - Cancelling (Esc or Cancel button) cleanly restores the active crop without altering the adjustment recipe.

### B. Save as Copy Split Button & Options
1. **Two-Button Vertical Split Control**:
   - Edit ribbon features a dedicated split button container (`SaveCopySplitContainer`) with a 32px top action button (`SaveCopyTopBtn`) and a bottom instant popup button (`SaveCopyBottomBtn`, `Save as\ncopy ▾`).
   - Menu options provide: "Save in original location", "Choose Location...", and "Options...".
2. **Destination Folder "Ask each time"**:
   - Added `"Ask each time"` radio option in Options -> Edit. When enabled, clicking the primary Save as copy button immediately prompts the native file save dialog.
3. **Synchronous Catalog Indexing & Switch View Settings**:
   - `save_as_copy` synchronously indexes the new file in `CatalogRepository`, enabling immediate next/previous navigation.
   - Respects `save_copy_switch_view`: switches active viewer when enabled, or keeps viewer on the original edited master when disabled.

### C. Self-Healing Thumbnail Queue & Drag Handle Parity
1. **Zero Blank Placeholders on Small Thumbnails**:
   - Increased visible buffer rows in `visible_media_ids(buffer_rows=4)` (`buf_px >= 240`).
   - Added self-healing missing thumbnail detection in `paintEvent`: if any visible on-screen card lacks a cached pixmap, `_scroll_debounce` triggers an immediate background decode without requiring manual scrolling.
2. **Universal Borderless Drag Handles**:
   - Crop, Spot Fix, and Markup floating toolbars all feature `_DraggableGripHandle`, rendering antialiased grip dots (`#8595A8`, hover `#2563EB`) on a 100% transparent, borderless background with seamless mouse forwarding.
