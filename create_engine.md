# Create Tools Engine — Technical Architecture & Implementation Review

## 1. Architectural Overview

The **Create Tools Engine** powers the creative photo manipulation and synthesis features in Essentials Photo Gallery. The subsystem consists of three primary layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        UI Studio Layer                                 │
│  (src/ui/create_studio_dialogs.py & src/ui/ribbon_toolbar.py)          │
│  - Interactive dialogs with 60 FPS display proxy                       │
│  - High-DPI Aero slider controls & zoom bars                           │
│  - Photoshop-style multi-color eyedropper & brush masking              │
│  - Instant non-destructive "Hold to Compare" previews                  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     Imaging Pipeline Layer                             │
│  (src/imaging/create_tools.py & src/imaging/creative_effects.py)       │
│  - Fast interactive downscaled proxy rendering                         │
│  - High-resolution final export pipelines                              │
│  - Memory-efficient NumPy array operations                            │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Computer Vision & Core AI Layer                      │
│  - OpenCV (cv2): SIFT/ORB, Homography, Inpainting, GrabCut, LAB        │
│  - Pillow (PIL): Scalable typography, anti-aliased geometry, drop shadows│
│  - PySide6: Hardware-accelerated graphics scenes & interactive canvas  │
└────────────────────────────────────────────────────────────────────────┘
```

### Core Design Principles
1. **Interactive Display Proxies**: Heavy computer vision operations (depth estimation, GrabCut, optical bokeh pyramids) are computed on display proxies (typically bounded to 1280px max dimension). This ensures that slider dragging and brush strokes update at a smooth 60 FPS.
2. **Lossless Full-Resolution Export**: All user parameters, strokes, and masks are stored in normalized floating-point coordinates. When the user clicks "Save New...", masks are upsampled via guided filtering and applied to the full-resolution source image (e.g. 24 MP/48 MP RAW/JPEG) with zero loss of quality.
3. **Non-Destructive Workflows**: Original image files are never modified. The engine uses a strict copy-on-write model with file collision protection (`filename (2).jpg`).

---

## 2. Technical Specification by Create Tool

---

### Tool 1: Panoramic Photo Stitching (`PanoramaDialog`)

#### Implementation Files
- **UI Controller**: `PanoramaDialog` in [`src/ui/create_studio_dialogs.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work%20Code/Python/Gallery/src/ui/create_studio_dialogs.py)
- **Algorithm Engine**: `postprocess_panorama`, `inpaint_minor_border_gaps`, `fallback_homography_stitch` in [`src/imaging/create_tools.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/imaging/create_tools.py)

#### Algorithmic Pipeline
1. **Feature Extraction & Matching**:
   - Sequential image pairs are analyzed using SIFT (Scale-Invariant Feature Transform) with a feature budget of 2,500 keypoints per image (falling back to ORB with 3,000 points if SIFT is unavailable).
   - Keypoints are matched using FLANN (Fast Library for Approximate Nearest Neighbors) or Brute-Force Hamming distance with Lowe's ratio test ($d_1 / d_2 < 0.75$).
2. **Stitching Execution**:
   - Standard execution utilizes `cv2.Stitcher_create(cv2.Stitcher_PANORAMA)` configured with cylindrical camera projection and bundle adjustment.
   - **Fail-Safe Fallback**: If OpenCV's native stitcher returns an error code (common with handheld scenes or uneven overlap), execution transitions automatically to `fallback_homography_stitch()`. This computes pairwise RANSAC homography matrices ($H$), warps successive frames into a shared canvas, and applies linear multi-band feathering across overlap regions.
3. **Border Straightening & Gap Inpainting**:
   - Raw stitched panoramas feature irregular curved black borders caused by projective distortion.
   - `postprocess_panorama()` detects the outer boundary of the stitched canvas.
   - Minor missing edge scallops (area $< 2.5\%$ of total resolution) are inpainted using the Alexandru Telea fast marching algorithm (`cv2.inpaint(..., inpaintRadius=3, flags=cv2.INPAINT_TELEA)`).
   - A maximum inscribed rectangular bounding box is calculated to automatically crop away remaining black regions without user intervention.

---

### Tool 2: Photo Fuse (`PhotoFuseDialog`)

#### Implementation Files
- **UI Controller**: `PhotoFuseDialog` in [`src/ui/create_studio_dialogs.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/create_studio_dialogs.py)
- **Algorithm Engine**: `fuse_face`, `detect_faces_in_image` in [`src/imaging/create_tools.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/imaging/create_tools.py)

#### Algorithmic Pipeline
Photo Fuse replicates the signature Windows Live Photo Gallery feature: merging the best expressions from burst or bracketed group shots into a single composite.

1. **Sub-Pixel Perspective Alignment**:
   - The selected replacement photo is registered to the base photo using feature matching and a projective homography matrix $H$:
     $$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
   - `cv2.warpPerspective` aligns the images so background and body contours align with sub-pixel precision.
2. **CIE LAB Illumination Transfer**:
   - Photos taken seconds apart frequently suffer from auto-exposure drift or flash recycling variations.
   - The face patch in the aligned source image is converted to CIE $L^*a^*b^*$ color space.
   - The mean and standard deviation of each channel are matched to the base image's face region:
     $$L^*_{\text{new}} = \left( L^*_{\text{src}} - \mu_{L,\text{src}} \right) \cdot \frac{\sigma_{L,\text{base}}}{\sigma_{L,\text{src}}} + \mu_{L,\text{base}}$$
   - This eliminates noticeable exposure or color temperature shifts across the boundary.
3. **Feathered Elliptical Boundary Blending**:
   - An elliptical mask is generated around the selected face with a $25\%$ outer safety margin.
   - Cosine Hermite polynomial feathering smooths the alpha transition:
     $$t = \frac{d}{d_{\text{feather}}}, \quad \alpha(t) = t^2 \cdot (3 - 2t)$$
   - The resulting blend is seamlessly integrated with no visible seam lines.

---

### Tool 3: Auto-Collage Engine (`CollageDialog`)

#### Implementation Files
- **UI Controller**: `AutoCollageDialog` in [`src/ui/create_studio_dialogs.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/create_studio_dialogs.py)
- **Algorithm Engine**: `create_blended_seamless_collage`, `create_grid_collage`, `create_polaroid_scatter_collage`, `create_then_and_now_card` in [`src/imaging/create_tools.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/imaging/create_tools.py)

#### Sub-Mode Technical Breakdown

#### A. Blended Collage
Inspired by Microsoft Research AutoCollage (Rother et al., SIGGRAPH):
1. **Face Detection & Salience Prominence**:
   - Each input photo is evaluated via Haar cascade / DNN face detection (`detect_faces_in_image`).
   - For responsive performance, images with dimension $> 960$px are evaluated on downscaled proxies with inverse coordinates projected back to native scale.
   - A face prominence score $P$ is computed:
     $$P = \frac{\max_{(x,y,w,h) \in \text{Faces}} (w \cdot h)}{W_{\text{img}} \cdot H_{\text{img}}}$$
   - The photo with the highest prominence score is automatically prioritized and positioned in **Slot 0 (Hero focal slot)**.
2. **Headroom-Preserving Smart Cropping**:
   - Rather than static center crops, `_crop_to_aspect_with_rect` centers the crop around $(f_{cx}, f_{cy})$, where:
     $$f_{cx} = \frac{x_{\text{face}} + 0.5 \cdot w_{\text{face}}}{W}, \quad f_{cy} = \frac{y_{\text{face}} + 0.40 \cdot h_{\text{face}}}{H}$$
   - The upward $0.40$ offset ensures eyes and foreheads are never chopped off.
3. **Face Protection Against Overlap Ghosting**:
   - Standard overlapping alpha blending ($C = \sum P_i M_i / \sum M_i$) causes semi-transparent ghostly overlays when background elements of Photo B overlap people's faces in Photo A.
   - The engine generates a `face_boost` map:
     - Inside any detected face bounding box $+ 25\%$ margin, `mask` is locked to $1.0$ and `face_boost` is set to $5.0$.
     - Blending weights become $W = M \cdot (1.0 + \text{face\_boost})$.
     - As a result, the face maintains $> 94\%$ opacity even in overlapping regions, completely preserving crisp, sharp facial features.
4. **100% Canvas Coverage Guarantee**:
   - `_get_seamless_slots(N, is_portrait)` dynamically partitions the canvas into overlapping geometric envelopes covering $[0, W] \times [0, H]$ for any $N \in [1..35]$, guaranteeing zero holes or unassigned gray gaps.

#### B. Modern Grid / Masonry
- Generates dynamic 2x2, 2x1, 3x2, Center Hero, or Masonry tessellations.
- Renders rounded corners with anti-aliasing via 4-channel BGRA buffers (`_apply_rounded_corners`).

#### C. Polaroid Table Scatter
- Individual cards are generated with authentic Polaroid dimensions ($3.5" \times 4.2"$).
- Procedural backgrounds (Dark Mahogany Wood, Slate, Fine Linen) are rendered using high-resolution Perlin-style noise and directional grain filters.
- Gaussian drop shadows ($r = 12$) simulate physical elevation.
- Timestamps and EXIF dates are parsed and formatted into human-readable strings (`"14 Sep 2022"`), safely handling epoch numbers and string timestamps.

#### D. Then & Now Comparison Card
- Automatically sequences pairs of photos chronologically by EXIF date.
- Applies archival sepia, vignette, and film grain tone mapping to the older photograph, while preserving modern HDR color on the recent photograph.

---

### Tool 4: Colour Pop Studio (`ColourPopDialog`)

#### Implementation Files
- **UI Controller**: `ColourPopDialog`, `_InteractiveMaskCanvasView` in [`src/ui/create_studio_dialogs.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/create_studio_dialogs.py)
- **Algorithm Engine**: `apply_color_pop`, `compute_color_similarity_mask`, `estimate_subject_mask`, `guided_filter_matte` in [`src/imaging/creative_effects.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/imaging/creative_effects.py)

#### Algorithmic Pipeline
1. **Initial Automatic Subject Segmentation**:
   - Automatically initializes foreground seeds using face detection bounding boxes and a central saliency prior.
   - Computes an initial trimap (definite foreground, probable foreground, probable background, definite background).
   - Executes GrabCut graph-cut optimization (`cv2.grabCut`) to isolate the main subject.
2. **Multi-Colour Eyedropper with Photoshop Shortcuts**:
   - The user can sample colors directly from the interactive canvas.
   - **Photoshop Shortcuts**:
     - **Click**: Replace selection with target color.
     - **Shift + Click**: Add another color to the selection.
     - **Alt + Click / Ctrl + Click / Right-Click**: Remove a color from the selection.
   - For each active color $C_k = (B_k, G_k, R_k)$ and tolerance $\tau \in [0.05, 0.60]$:
     $$d_k(x, y) = \frac{1}{\sqrt{3} \cdot 255} \sqrt{(B - B_k)^2 + (G - G_k)^2 + (R - R_k)^2}$$
     $$M_k(x, y) = \text{clip}\left( 1.0 - \frac{d_k(x, y)}{\tau}, 0.0, 1.0 \right)$$
   - The composite mask is computed as the maximum across all picked colors:
     $$M_{\text{composite}}(x, y) = \max_k M_k(x, y)$$
   - Active colors are displayed in the UI as interactive swatch chips with one-click removal.
3. **Guided Filter Edge Matting**:
   - Edge snapping is achieved via He et al. Guided Image Filtering (`guided_filter_matte`):
     $$q_i = a_k I_i + b_k, \quad \forall i \in \omega_k$$
   - The high-resolution RGB image acts as the guide, pulling the blurred mask edges precisely to physical hair strands, garments, and object boundaries without haloing.
4. **High-Contrast B&W Synthesis**:
   - Background pixels are desaturated into grayscale and tone-mapped using adjustable contrast and brightness:
     $$I_{\text{bg}}(x, y) = \text{clip}\left( (I_{\text{gray}} - 128) \cdot C_{\text{bg}} + 128 + B_{\text{bg}}, 0, 255 \right)$$
     $$I_{\text{final}} = I_{\text{orig}} \cdot M + I_{\text{bg}} \cdot (1 - M)$$

---

### Tool 5: Portrait Blur / Bokeh Studio (`PortraitBokehDialog`)

#### Implementation Files
- **UI Controller**: `PortraitBokehDialog` in [`src/ui/create_studio_dialogs.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/create_studio_dialogs.py)
- **Algorithm Engine**: `apply_portrait_bokeh`, `build_bokeh_pyramid`, `apply_portrait_bokeh_and_fill_light` in [`src/imaging/creative_effects.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/imaging/creative_effects.py)

#### Algorithmic Pipeline
1. **Multi-Scale Optical Disc Bokeh Pyramid**:
   - True photographic lens blur exhibits a circular disc Point Spread Function (PSF), not a Gaussian curve.
   - Direct spatial convolution with a large disc kernel ($r = 40$) is computationally prohibitive for live preview ($O(r^2)$ per pixel).
   - `build_bokeh_pyramid()` pre-computes an octave pyramid where each successive level is generated by spatial downsampling followed by disc convolution with a kernel:
     $$K_{\text{disc}}(x, y) = \begin{cases} \frac{1}{\pi r^2} & \text{if } x^2 + y^2 \le r^2 \\ 0 & \text{otherwise} \end{cases}$$
   - During live rendering, pixels are interpolated across pyramid levels in constant time $O(1)$, delivering authentic circular optical highlights (circles of confusion) at 60 FPS.
2. **Anti-Halo Edge Bleeding Prevention**:
   - A critical problem in portrait blur is "halo bleeding": bright foreground colors blurring into the dark background, or background blur leaking over sharp hair contours.
   - The engine masks out foreground pixels before computing the background blur levels, filling the occlusion boundary via normalized cross-correlation. When foreground and blurred background are recombined, edges remain razor-sharp.
3. **Soft-Knee Subject Fill Lighting**:
   - Studio portraiture requires flattering lighting on the subject.
   - The foreground face/subject region receives an adjustable soft-knee gamma boost ($0\% - 50\%$):
     $$I_{\text{lit}} = I \cdot \left( 1.0 + \gamma_{\text{fill}} \cdot (1.0 - I_{\text{norm}}) \right)$$
   - Shadow areas on faces are gently illuminated while highlights are protected from clipping.

---

### Tool 6: 3D Cinematic Motion (`CinematicMotionDialog`)

#### Implementation Files
- **UI Controller**: `CinematicMotionDialog` in [`src/ui/create_studio_dialogs.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/create_studio_dialogs.py)
- **Algorithm Engine**: `render_3d_cinematic_motion_frame`, `estimate_depth_map`, `compute_target_dimensions` in [`src/imaging/creative_effects.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/imaging/creative_effects.py)

#### Algorithmic Pipeline
Converts a single 2D still photograph into a cinematic 3D video sequence (MP4/H.264) with camera parallax.

1. **Monocular Depth Map Estimation**:
   - An estimated continuous depth map $D(x, y) \in [0.0, 1.0]$ is derived from face/body priors, focus gradients, and vertical ground plane perspective ($0.0 = \text{near/foreground}$, $1.0 = \text{far/background}$).
2. **Motion Trajectory Curves**:
   - Four distinct camera motion profiles are evaluated over normalized time $t \in [0.0, 1.0]$:
     - **Dolly Zoom (Vertigo Effect)**: Simultaneous camera zoom-in and reverse background perspective scaling:
       $$S(t) = 1.0 + 0.15 \cdot t, \quad \Delta_{\text{parallax}} \propto (D - 0.5) \cdot t$$
     - **Parallax Drift**: Gentle horizontal floating camera move:
       $$\Delta x(t) = A_x \cdot \sin(2\pi t) \cdot (1.0 - D(x, y))$$
     - **3D Orbit**: Elliptical orbit around the subject's face:
       $$\Delta x(t) = R_x \cdot \cos(2\pi t) \cdot (1.0 - D), \quad \Delta y(t) = R_y \cdot \sin(2\pi t) \cdot (1.0 - D)$$
     - **Cinematic Pan**: Slow diagonal sweep across the visual axis.
3. **Non-Linear Mesh Grid Warping**:
   - An $(X, Y)$ coordinate displacement mesh is evaluated via `cv2.remap` with bicubic interpolation (`cv2.INTER_CUBIC`).
   - Foreground subjects displace more than the background, creating physical depth separation.
   - Outer frame boundaries utilize reflective border replication (`cv2.BORDER_REFLECT_101`) to eliminate black edge artifacts during camera movement.

---

## 3. UI Iconography & Aero Styling Standards

All Create Studio dialogs adhere strictly to Windows Live Photo Gallery Aero design language:

| UI Control | Visual Specification | Implementation Path |
| :--- | :--- | :--- |
| **Shuffle Photo Order** | Blue crossed arrows with arrowheads (`#2468A8`) | `make_wlpg_icon("shuffle", 16)` in `_render_aero_raw` |
| **Keep Vibrant Colour (+)** | Crisp vibrant plus icon (`#246BB2`) | `make_wlpg_icon("plus", 16)` in `_render_aero_raw` |
| **Make Black and White (-)** | Crisp red/neutral minus icon (`#C9302C`) | `make_wlpg_icon("minus", 16)` in `_render_aero_raw` |
| **Pick Colour from Photo** | Angled glass pipette with color fluid & droplet | `make_wlpg_icon("eyedropper", 16)` in `_render_aero_raw` |
| **Hold to Compare** | Clean native button without distracting placeholder square | `btn_compare` (text-only, hover state `#EAF2FB`) |
| **Fine-Tune Sliders** | Antialiased Aero pill slider with real-time value tooltip | `_AeroFineTuneSlider` |

---

## 4. Test Coverage & Verification Matrix

The Create Tools engine is covered by an automated test suite in [`tests/test_create_tools_overhaul.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/tests/test_create_tools_overhaul.py) and [`tests/test_create_tools.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/tests/test_create_tools.py):

| Test Case | Scope & Verification Target | Status |
| :--- | :--- | :--- |
| `test_guided_filter_matte_and_subject_mask` | Guided filter edge snapping, trimap extraction, face-seeded GrabCut | **PASS** |
| `test_color_pop_hue_and_subject` | Euclidean color distance mask, B&W background synthesis | **PASS** |
| `test_portrait_bokeh_pyramid_speed_and_halos` | Optical disc bokeh pyramid, sub-100ms proxy evaluation, halo prevention | **PASS** |
| `test_3d_cinematic_motion_curves_and_bounds` | Mesh remap stability, aspect preservation, zero division safeguards | **PASS** |
| `test_smart_crop_and_scalable_typography` | Headroom face awareness, scalable typography font metrics | **PASS** |
| `test_panorama_fast_cache_and_fuse_color_match` | Inscribed crop bounding box, CIE LAB color mean/std transfer | **PASS** |
| `test_create_studio_dialogs_ui_layout_and_zoom_controls` | Minimum dimensions, zoom bar synchronization, event handling | **PASS** |
| `test_auto_collage_face_awareness_and_polaroid_int_caption` | 7-photo 100% canvas coverage, epoch timestamp caption parsing | **PASS** |
| `test_colour_pop_multi_pick_and_ui_labels` | Multi-color picking (`replace`, `add`, `remove`), UI swatch chips, no square on Compare | **PASS** |
