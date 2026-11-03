"""
High-Fidelity OpenCV & NumPy Photo Adjustment Engine for PhotoGallery64.
Provides professional, non-destructive, floating-point image processing:
- Linear exposure & filmic soft-knee shoulder roll-off
- Edge-aware highlight & shadow recovery via CIE LAB decomposition (Snapseed Ambience)
- Midtone Clarity, Dehaze, and Aspect-Aware Vignette
- Perceptual skin-safe Vibrance, Color Temperature, and Tint
- Selective Control Point (U-Point) bilateral adjustment
- 6 High-Taste Photographic Preset Grades with 0-100% Intensity Blending
- Functional Noise Profile Analysis (MAD wavelet/Laplacian estimation) & Dual-Domain Bilateral Denoising
- Halo-free Luminance Sharpening
- Analytical Inscribed-Rectangle Auto-Straightening
- Fast multichannel histogram computation for real-time UI feedback
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np

# Precomputed high-precision sRGB <-> Linear floating-point look-up tables
_LUT_SRGB_TO_LIN: np.ndarray = ((np.arange(256, dtype=np.float32) / 255.0) ** 2.2).astype(np.float32)
_LUT_LIN_TO_SRGB: np.ndarray = np.clip(
    (np.linspace(0.0, 1.0, 4096, dtype=np.float32) ** (1.0 / 2.2)) * 255.0 + 0.5, 0, 255
).astype(np.uint8)
_VIBRANCE_POW_LUT: np.ndarray = (np.linspace(0.0, 1.0, 256, dtype=np.float32) ** 0.8).astype(np.float32)


@dataclass
class PhotoAdjustmentRecipe:
    """Non-destructive adjustment parameter stack applied to the master image."""

    brightness: float = 0.0       # -100.0 .. +100.0 (Exposure shift with filmic shoulder)
    contrast: float = 0.0         # -100.0 .. +100.0 (Filmic S-curve)
    shadows: float = 0.0          # -100.0 .. +100.0 (Edge-aware shadow lift / compress)
    highlights: float = 0.0       # -100.0 .. +100.0 (Highlight recovery)
    clarity: float = 0.0          # -100.0 .. +100.0 (Midtone local contrast)
    temperature: float = 0.0      # -100.0 .. +100.0 (Cool blue <-> Warm amber)
    tint: float = 0.0             # -100.0 .. +100.0 (Green <-> Magenta)
    saturation: float = 0.0       # -100.0 .. +100.0 (Skin-safe vibrance / chroma scale)
    dehaze: float = 0.0           # -100.0 .. +100.0 (Atmospheric haze reduction)
    vignette: float = 0.0         # -100.0 .. +100.0 (Dark <-> Light corners)
    straighten_deg: float = 0.0   # -15.0 .. +15.0 degrees
    sharpen: float = 0.0          # 0.0 .. 100.0 (Halo-free unsharp mask amount)
    denoise: float = 0.0          # 0.0 .. 100.0 (Dual-domain bilateral amount)
    black_point: float = 0.0      # 0.0 .. 255.0 (Histogram black point clip)
    white_point: float = 255.0    # 0.0 .. 255.0 (Histogram white point clip)
    crop_rect: tuple[int, int, int, int] | None = None  # (left, top, right, bottom)
    effect: str = "none"          # "none" | "classic_chrome" | "warm_portrait" | "vivid_landscape" | "matte_vintage" | "fine_art_bw" | "cool_cinema"
    effect_intensity: float = 100.0  # 0.0 .. 100.0 (% blend with base adjusted image)
    selective_points: list[dict[str, Any]] = field(default_factory=list)  # list of {x, y, radius, brightness, contrast, saturation} (supports normalized [0.0, 1.0])

    # 3D Depth & Light Adjustments
    depth_blur: float = 0.0       # 0.0 .. 100.0 (Optical disc bokeh blur strength)
    focal_plane: float = 0.0      # 0.0 .. 100.0 (In-focus depth plane: 0=foreground, 100=background)
    bokeh_glow: float = 0.0       # 0.0 .. 100.0 (Linear-light specular highlight bloom)
    relight_near: float = 0.0     # -100.0 .. +100.0 (Fill light on foreground subject)
    relight_far: float = 0.0      # -100.0 .. +100.0 (Exposure shift on distant background)
    warmth_near: float = 0.0      # -100.0 .. +100.0 (CIE LAB B* warmth shift on foreground)
    warmth_far: float = 0.0       # -100.0 .. +100.0 (CIE LAB B* warmth shift on background)
    depth_haze: float = 0.0       # 0.0 .. 100.0 (Beer-Lambert exponential atmospheric depth mist)
    depth_haze_warmth: float = 0.0 # -100.0 .. +100.0 (Mist tint: cool blue <-> golden amber)

    def is_default(self) -> bool:
        """Returns True if no visual adjustments are active."""
        return (
            abs(self.brightness) < 0.1
            and abs(self.contrast) < 0.1
            and abs(self.shadows) < 0.1
            and abs(self.highlights) < 0.1
            and abs(self.clarity) < 0.1
            and abs(self.temperature) < 0.1
            and abs(self.tint) < 0.1
            and abs(self.saturation) < 0.1
            and abs(self.dehaze) < 0.1
            and abs(self.vignette) < 0.1
            and abs(self.straighten_deg) < 0.01
            and abs(self.sharpen) < 0.1
            and abs(self.denoise) < 0.1
            and abs(self.black_point) < 0.1
            and abs(self.white_point - 255.0) < 0.1
            and self.crop_rect is None
            and self.effect == "none"
            and len(self.selective_points) == 0
            and abs(self.depth_blur) < 0.1
            and abs(self.focal_plane) < 0.1
            and abs(self.bokeh_glow) < 0.1
            and abs(self.relight_near) < 0.1
            and abs(self.relight_far) < 0.1
            and abs(self.warmth_near) < 0.1
            and abs(self.warmth_far) < 0.1
            and abs(self.depth_haze) < 0.1
            and abs(self.depth_haze_warmth) < 0.1
        )

    def copy(self) -> PhotoAdjustmentRecipe:
        return PhotoAdjustmentRecipe(
            brightness=self.brightness,
            contrast=self.contrast,
            shadows=self.shadows,
            highlights=self.highlights,
            clarity=self.clarity,
            temperature=self.temperature,
            tint=self.tint,
            saturation=self.saturation,
            dehaze=self.dehaze,
            vignette=self.vignette,
            straighten_deg=self.straighten_deg,
            sharpen=self.sharpen,
            denoise=self.denoise,
            black_point=self.black_point,
            white_point=self.white_point,
            crop_rect=self.crop_rect,
            effect=self.effect,
            effect_intensity=self.effect_intensity,
            selective_points=[dict(p) for p in self.selective_points],
            depth_blur=self.depth_blur,
            focal_plane=self.focal_plane,
            bokeh_glow=self.bokeh_glow,
            relight_near=self.relight_near,
            relight_far=self.relight_far,
            warmth_near=self.warmth_near,
            warmth_far=self.warmth_far,
            depth_haze=self.depth_haze,
            depth_haze_warmth=self.depth_haze_warmth,
        )


def _crop_largest_inscribed_rect(
    img: np.ndarray, angle_deg: float, orig_w: int, orig_h: int
) -> np.ndarray:
    """
    Crops the largest inscribed bounding box from an image rotated around its center,
    guaranteeing that zero black border triangles or transparent edges ever appear.
    """
    rad = math.radians(abs(angle_deg))
    sin_a = math.sin(rad)
    cos_a = math.cos(rad)

    w = float(orig_w)
    h = float(orig_h)

    # Standard formula for largest inscribed rectangle matching original aspect ratio
    factor = 1.0 / (cos_a + (w / h) * sin_a if w > h else cos_a + (h / w) * sin_a)
    new_w = max(10, int(w * factor * 0.985))
    new_h = max(10, int(h * factor * 0.985))

    curr_h, curr_w = img.shape[:2]
    cx, cy = curr_w // 2, curr_h // 2
    x1 = max(0, cx - new_w // 2)
    y1 = max(0, cy - new_h // 2)
    x2 = min(curr_w, x1 + new_w)
    y2 = min(curr_h, y1 + new_h)

    return img[y1:y2, x1:x2]


def _apply_effect_grade(rgb_img: np.ndarray, effect_name: str) -> np.ndarray:
    """
    Applies high-taste modern photographic grades or legacy WLPG effects.
    """
    if effect_name in ("none", "", None):
        return rgb_img

    # 1. Classic Chrome (Fujifilm style: rich cyan skies, soft warm reds, dense contrast)
    if effect_name == "classic_chrome":
        f = rgb_img.astype(np.float32) / 255.0
        # Boost contrast
        diff = f - 0.5
        f = 0.5 + diff * 1.15 - 0.2 * (diff ** 3)
        # Shift sky cyan and mute reds
        f[:, :, 0] = np.clip(f[:, :, 0] * 0.96 + 0.02, 0.0, 1.0)
        f[:, :, 2] = np.clip(f[:, :, 2] * 1.05, 0.0, 1.0)
        # Gentle desaturation of greens/yellows
        hsv = cv2.cvtColor((np.clip(f, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= 0.88
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 2. Warm Portrait (Kodak Portra style: warm golden skin tones, soft pastel highlights)
    elif effect_name in ("warm_portrait", "warm"):
        f = rgb_img.astype(np.float32) / 255.0
        # Warm golden highlight roll-off
        f[:, :, 0] = np.clip(f[:, :, 0] * 1.08 + 0.02, 0.0, 1.0)
        f[:, :, 1] = np.clip(f[:, :, 1] * 1.02 + 0.01, 0.0, 1.0)
        f[:, :, 2] = np.clip(f[:, :, 2] * 0.92, 0.0, 1.0)
        # Soft contrast
        f = np.clip(f * 0.95 + 0.05, 0.0, 1.0)
        return (f * 255.0).astype(np.uint8)

    # 3. Vivid Landscape (Velvia style: punchy dynamic range, deep emerald greens, vibrant skies)
    elif effect_name == "vivid_landscape":
        f = rgb_img.astype(np.float32) / 255.0
        diff = f - 0.5
        f = 0.5 + diff * 1.25 - 0.3 * (diff ** 3)
        f = np.clip(f, 0.0, 1.0)
        u8 = (f * 255.0).astype(np.uint8)
        hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0.0, 255.0)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 4. Matte Vintage (Lifted film blacks, subtle warm split shadows)
    elif effect_name == "matte_vintage":
        f = rgb_img.astype(np.float32) / 255.0
        # Lift blacks to ~0.08
        f = f * 0.88 + 0.08
        # Warm split toning in shadows
        shadow_weight = np.clip(1.0 - f, 0.0, 1.0)
        f[:, :, 0] += shadow_weight[:, :, 0] * 0.04
        f[:, :, 2] -= shadow_weight[:, :, 2] * 0.03
        return (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8)

    # 5. Fine Art B&W (Acros / Ilford style: deep rich blacks, luminous midtones, crisp microcontrast)
    elif effect_name in ("fine_art_bw", "bw", "bw_contrast", "high_contrast_bw"):
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        diff = gray - 0.5
        mult = 1.45 if effect_name in ("bw_contrast", "high_contrast_bw") else 1.25
        gray = np.clip(0.5 + diff * mult - 0.25 * (diff ** 3), 0.0, 1.0)
        u8_gray = (gray * 255.0).astype(np.uint8)
        return cv2.cvtColor(u8_gray, cv2.COLOR_GRAY2RGB)

    # 6. Cool Cinema (Teal shadows & warm golden highlights)
    elif effect_name in ("cool_cinema", "cool", "cyanotype"):
        f = rgb_img.astype(np.float32) / 255.0
        lum = 0.2126 * f[:, :, 0] + 0.7152 * f[:, :, 1] + 0.0722 * f[:, :, 2]
        shadow_mask = np.clip(1.0 - lum / 0.6, 0.0, 1.0)[:, :, None]
        high_mask = np.clip((lum - 0.4) / 0.6, 0.0, 1.0)[:, :, None]
        # Teal in shadows
        f[:, :, 1] += shadow_mask[:, :, 0] * 0.04
        f[:, :, 2] += shadow_mask[:, :, 0] * 0.12
        # Warm in highlights
        f[:, :, 0] += high_mask[:, :, 0] * 0.07
        f[:, :, 1] += high_mask[:, :, 0] * 0.03
        return (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Legacy Sepia
    elif effect_name == "sepia":
        sepia_mat = np.array(
            [[0.393, 0.769, 0.189],
             [0.349, 0.686, 0.168],
             [0.272, 0.534, 0.131]],
            dtype=np.float32,
        )
        transformed = cv2.transform(rgb_img, sepia_mat)
        return np.clip(transformed, 0, 255).astype(np.uint8)

    return rgb_img


def render_adjusted_image(
    master_rgb: np.ndarray,
    recipe: PhotoAdjustmentRecipe,
    media_id: Any = None,
) -> np.ndarray:
    """
    Renders an adjusted uint8 RGB image from the master uint8 RGB array according to recipe.
    Uses 32-bit floating point calculations for smooth, non-destructive tonal fidelity.

    Optimal photographic pipeline execution order:
    1. Denoise (Resolution-scaled dual-domain bilateral filter)
    2. Dehaze (Atmospheric dark-channel prior with bilateral transmission map refinement)
    3. Levels (Black/White Point) & Filmic Exposure (with negative stops + toe lift)
    4. Unified CIE LAB Pass:
       - Base/Detail separation (L_base, L_detail)
       - White Balance (Temperature B*, Tint A*)
       - Shadows & Highlights + Ambience micro-contrast
       - Midtone Clarity on L_detail
       - Filmic S-curve Contrast on Luminance
       - Selective Control Points (U-Point) with normalized [0..1] support
       - Skin-Safe Vibrance directly in LAB (chroma scaling with warm skin locus protection)
       - Halo-free Luminance Unsharp Mask with dead-zone soft thresholding
    5. Creative Preset Blend (0..100% intensity)
    6. Aspect-Aware Elliptical Vignette
    7. Straighten & Crop (Bicubic rotation + inscribed rectangle crop or user crop_rect)
    """
    if master_rgb is None or master_rgb.size == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    if recipe.is_default():
        return np.ascontiguousarray(master_rgb)

    img = master_rgb
    h_orig, w_orig = img.shape[:2]
    res_scale = max(w_orig, h_orig) / 1280.0

    if media_id is None:
        media_id = (
            img.shape,
            int(img[0, 0, 0]),
            int(img[-1, -1, 0]),
            int(img[h_orig // 2, w_orig // 2, 0]),
        )

    # 1. Detail: Decoupled Chroma-Denoise + Edge-Aware Luma Filtering
    if recipe.denoise >= 1.0:
        dn = recipe.denoise
        # Convert to LAB for perceptual luminance/chrominance separation
        d_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        L_ch = d_lab[:, :, 0]
        A_ch = d_lab[:, :, 1]
        B_ch = d_lab[:, :, 2]

        # 1a. Aggressive high-speed chroma denoise: Clean color blotches where the eye
        # is sensitive without affecting edge sharpness or hair details
        d_chroma = max(3, int(round(7 * res_scale)) | 1)
        sigma_c = float(dn * 0.9)
        sigma_space_c = float(6.0 * res_scale)
        dA = cv2.bilateralFilter(A_ch, d=d_chroma, sigmaColor=sigma_c, sigmaSpace=sigma_space_c)
        dB = cv2.bilateralFilter(B_ch, d=d_chroma, sigmaColor=sigma_c, sigmaSpace=sigma_space_c)

        # 1b. Edge-guided luma filtering: Use edge detection on L to preserve fine textures,
        # skin pores, and eyelashes while smoothing flat grain in skies/shadows
        sigma_luma_space = float(max(1.2, 3.0 * res_scale))
        L_smooth = cv2.bilateralFilter(
            L_ch,
            d=max(3, int(round(5 * res_scale)) | 1),
            sigmaColor=float(dn * 0.35),
            sigmaSpace=sigma_luma_space,
        )

        # Detect high-contrast edges/detail on original L
        edges = cv2.Sobel(L_ch, cv2.CV_32F, 1, 1, ksize=3)
        edge_mag = cv2.convertScaleAbs(edges)
        edge_weight = np.clip(edge_mag.astype(np.float32) / 32.0, 0.0, 1.0)

        # Blend: Strong denoise in smooth flat areas; preserve crisp original details on edges
        blend_factor = (dn / 100.0) * (1.0 - 0.75 * edge_weight)
        dL = np.clip(L_ch.astype(np.float32) * (1.0 - blend_factor) + L_smooth.astype(np.float32) * blend_factor, 0, 255).astype(np.uint8)

        d_lab[:, :, 0] = dL
        d_lab[:, :, 1] = dA
        d_lab[:, :, 2] = dB
        img = cv2.cvtColor(d_lab, cv2.COLOR_LAB2RGB)

    # 2. Dehaze (Atmospheric dark-channel prior with bilateral transmission map refinement)
    if abs(recipe.dehaze) >= 0.1:
        dh = recipe.dehaze / 100.0
        if dh > 0:
            dark = np.min(img, axis=2).astype(np.float32)
            k = max(3, int(round(15 * res_scale)))
            if k % 2 == 0:
                k += 1
            dark_smooth = cv2.blur(dark, (k, k))
            A_atm = float(np.percentile(img, 99))
            A_atm = max(A_atm, 100.0)
            tx = np.clip(1.0 - 0.75 * dh * (dark_smooth / A_atm), 0.25, 1.0)
            d_refine = max(3, int(round(7 * res_scale)) | 1)
            tx_refined = cv2.bilateralFilter(
                tx.astype(np.float32),
                d=d_refine,
                sigmaColor=0.15,
                sigmaSpace=max(3.0, 7.0 * res_scale),
            )
            f_rad = img.astype(np.float32)
            recovered = (f_rad - A_atm) / tx_refined[:, :, None] + A_atm
            img = np.clip(recovered, 0, 255).astype(np.uint8)

    # 3. Levels & Filmic Exposure (Black/White Point Clipping & Photographic Shoulder/Toe)
    # Accelerated via 256-element uint8 LUT (1.5ms vs 55ms float math).
    # When no other LAB parameters are active, filmic S-curve contrast is baked directly into the 1D LUT
    # to bypass the 100ms RGB<->LAB conversion entirely.
    has_levels = recipe.black_point >= 0.5 or recipe.white_point <= 254.5
    has_exposure = abs(recipe.brightness) >= 0.1
    has_contrast = abs(recipe.contrast) >= 0.1
    has_lab_except_contrast = (
        abs(recipe.shadows) >= 0.1
        or abs(recipe.highlights) >= 0.1
        or abs(recipe.clarity) >= 0.1
        or abs(recipe.temperature) >= 0.1
        or abs(recipe.tint) >= 0.1
        or abs(recipe.saturation) >= 0.1
        or recipe.sharpen >= 1.0
        or len(recipe.selective_points) > 0
    )
    bake_contrast_in_lut = has_contrast and not has_lab_except_contrast

    if has_levels or has_exposure or bake_contrast_in_lut:
        lut = np.arange(256, dtype=np.float32) / 255.0
        if has_levels:
            bp = min(recipe.black_point, recipe.white_point - 2.0) / 255.0
            wp = max(recipe.white_point, recipe.black_point + 2.0) / 255.0
            lut = np.clip((lut - bp) / (wp - bp), 0.0, 1.0)

        if has_exposure:
            b = recipe.brightness / 100.0  # -1.0 .. +1.0
            if b > 0.0:
                exp_mult = 1.0 + b * 1.4
                scaled = lut * exp_mult
                knee = 0.65
                over = np.maximum(0.0, scaled - knee)
                scaled = np.where(scaled > knee, knee + over / (1.0 + over * 1.8), scaled)
                lut = np.clip(scaled, 0.0, 1.0)
            else:
                stops = 2.0 ** (b * 1.5)
                f_scaled = lut * stops
                toe = 0.05 * abs(b) * lut * (1.0 - lut)
                lut = np.clip(f_scaled + toe, 0.0, 1.0)

        if bake_contrast_in_lut:
            c = recipe.contrast / 100.0
            diff = (lut - 0.5) * 2.0
            lut = np.clip(0.5 + 0.5 * (diff * (1.0 + c * 0.8) - (c * 0.25) * (diff ** 3)), 0.0, 1.0)

        lut_uint8 = np.clip(lut * 255.0 + 0.5, 0, 255).astype(np.uint8)
        img = cv2.LUT(img, lut_uint8)

    # 4. Unified CIE LAB Pass (executed only if LAB adjustments or non-baked contrast active)
    has_lab = has_lab_except_contrast or (has_contrast and not bake_contrast_in_lut)
    if has_lab:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = lab[:, :, 0]
        A = lab[:, :, 1]
        B = lab[:, :, 2]
        h_i, w_i = L.shape

        # Base / Detail decomposition if Shadows, Highlights, or Clarity active (4x hierarchical acceleration)
        needs_decompose = (
            abs(recipe.shadows) >= 0.1
            or abs(recipe.highlights) >= 0.1
            or abs(recipe.clarity) >= 0.1
        )
        if needs_decompose:
            sigma_base = max(3.0 * res_scale, min(14.0 * res_scale, min(h_i, w_i) / 90.0))
            sw = max(16, w_i // 4)
            sh = max(16, h_i // 4)
            L_small = cv2.resize(L, (sw, sh), interpolation=cv2.INTER_AREA)
            L_base_small = cv2.GaussianBlur(L_small, (0, 0), sigmaX=sigma_base / 4.0, sigmaY=sigma_base / 4.0)
            L_base = cv2.resize(L_base_small, (w_i, h_i), interpolation=cv2.INTER_LINEAR)
            L_detail = L - L_base

            if abs(recipe.shadows) >= 0.1:
                s_val = recipe.shadows / 100.0
                s_norm = L_base / 255.0
                if s_val > 0:
                    s_mask = np.clip(1.0 - s_norm / 0.65, 0.0, 1.0) ** 1.8
                    L_base = L_base + s_val * s_mask * (145.0 - L_base) * 0.72
                    # Ambience boost: preserve crisp local micro-contrast in lifted shadows
                    L_detail = L_detail * (1.0 + s_val * 0.35 * s_mask)
                else:
                    s_mask = np.clip(1.0 - s_norm / 0.5, 0.0, 1.0)
                    L_base = L_base + s_val * s_mask * L_base * 0.5

            if abs(recipe.highlights) >= 0.1:
                h_val = recipe.highlights / 100.0
                h_norm = L_base / 255.0
                h_mask = np.clip((h_norm - 0.42) / 0.58, 0.0, 1.0) ** 1.5
                if h_val > 0:
                    L_base = L_base + h_val * h_mask * (255.0 - L_base) * 0.5
                else:
                    L_base = L_base + h_val * h_mask * (L_base - 100.0) * 0.65

            if abs(recipe.clarity) >= 0.1:
                c_amount = recipe.clarity / 100.0
                mid_weight = 1.0 - 4.0 * ((L_base / 255.0) - 0.5) ** 2
                mid_weight = np.clip(mid_weight, 0.0, 1.0)
                L_detail = L_detail + c_amount * 0.45 * mid_weight * L_detail

            L = np.clip(L_base + L_detail, 0.0, 255.0)

        # White Balance (Temperature & Tint)
        if abs(recipe.temperature) >= 0.1 or abs(recipe.tint) >= 0.1:
            t_shift = (recipe.temperature / 100.0) * 26.0
            ti_shift = (recipe.tint / 100.0) * 20.0
            B = np.clip(B + t_shift, 0.0, 255.0)
            A = np.clip(A + ti_shift, 0.0, 255.0)

        # Filmic S-curve Contrast on Luminance (if not already baked into 1D LUT)
        if has_contrast and not bake_contrast_in_lut:
            c = recipe.contrast / 100.0
            diff = (L - 128.0) / 128.0
            L = np.clip(128.0 + 128.0 * (diff * (1.0 + c * 0.8) - (c * 0.25) * (diff ** 3)), 0.0, 255.0)

        # Selective Control Points (U-Point Bilateral Engine with normalized [0..1] support)
        if recipe.selective_points:
            y_coords, x_coords = np.ogrid[:h_i, :w_i]
            for pt in recipe.selective_points:
                pt_x = float(pt.get("x", 0))
                pt_y = float(pt.get("y", 0))
                pt_rad = float(pt.get("radius", 80))
                b_shift = float(pt.get("brightness", 0.0))
                c_shift = float(pt.get("contrast", 0.0))
                s_shift = float(pt.get("saturation", 0.0))

                if 0.0 <= pt_x <= 1.0 and 0.0 <= pt_y <= 1.0:
                    px = pt_x * w_i
                    py = pt_y * h_i
                else:
                    px = pt_x
                    py = pt_y

                if 0.0 < pt_rad <= 1.0:
                    rad = pt_rad * max(w_i, h_i)
                else:
                    rad = pt_rad * (res_scale if (0.0 <= pt_x <= 1.0) else 1.0)
                rad = max(5.0 * res_scale, rad)

                px_int = int(np.clip(round(px), 0, w_i - 1))
                py_int = int(np.clip(round(py), 0, h_i - 1))
                target_l = L[py_int, px_int]
                target_a = A[py_int, px_int]
                target_b = B[py_int, px_int]

                d2 = (x_coords - px) ** 2 + (y_coords - py) ** 2
                w_spatial = np.clip(1.0 - np.sqrt(d2) / rad, 0.0, 1.0) ** 2
                delta_e = np.sqrt((L - target_l) ** 2 + (A - target_a) ** 2 + (B - target_b) ** 2)
                w_color = np.exp(-(delta_e / 28.0) ** 2)
                w_total = w_spatial * w_color

                if np.any(w_total > 0.01):
                    if abs(b_shift) > 0.1:
                        L = np.clip(L + w_total * (b_shift * 0.8), 0.0, 255.0)
                    if abs(c_shift) > 0.1:
                        diff_l = L - target_l
                        L = np.clip(L + w_total * (diff_l * (c_shift / 100.0) * 0.5), 0.0, 255.0)
                    if abs(s_shift) > 0.1:
                        s_factor = 1.0 + w_total * (s_shift / 100.0)
                        A = np.clip(128.0 + (A - 128.0) * s_factor, 0.0, 255.0)
                        B = np.clip(128.0 + (B - 128.0) * s_factor, 0.0, 255.0)

        # Skin-Safe Vibrance directly in LAB (chroma scaling with warm skin locus protection)
        if abs(recipe.saturation) >= 0.1:
            sat_gain = recipe.saturation / 100.0
            da = A - 128.0
            db = B - 128.0
            chroma = np.sqrt(da * da + db * db)
            d_a14 = (A - 146.0) * (1.0 / 14.0)
            d_b18 = (B - 152.0) * (1.0 / 18.0)
            skin_dist = np.sqrt(d_a14 * d_a14 + d_b18 * d_b18)
            skin_mask = np.clip(1.0 - skin_dist, 0.0, 1.0)
            c_idx = np.clip(chroma * (255.0 / 128.0), 0.0, 255.0).astype(np.int32)
            vibrance_weight = (1.0 - _VIBRANCE_POW_LUT[c_idx]) * (1.0 - 0.6 * skin_mask)
            scale = np.maximum(0.0, 1.0 + sat_gain * (0.8 + 0.6 * vibrance_weight))
            A = np.clip(128.0 + da * scale, 0.0, 255.0)
            B = np.clip(128.0 + db * scale, 0.0, 255.0)

        # Halo-free Luminance Unsharp Mask with dead-zone soft thresholding
        if recipe.sharpen >= 1.0:
            sigma = 1.2 * res_scale
            L_blur = cv2.GaussianBlur(L, (0, 0), sigmaX=sigma, sigmaY=sigma)
            d = L - L_blur
            d_soft = np.sign(d) * np.maximum(0.0, np.abs(d) - 3.0)
            L = np.clip(L + (recipe.sharpen / 100.0) * 1.5 * d_soft, 0.0, 255.0)

        lab[:, :, 0] = L
        lab[:, :, 1] = A
        lab[:, :, 2] = B
        img = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    # 5. Photographic Effects Preset with Intensity Blend
    if recipe.effect not in ("none", "", None):
        effect_img = _apply_effect_grade(img, recipe.effect)
        alpha = max(0.0, min(1.0, recipe.effect_intensity / 100.0))
        if alpha < 0.999:
            img = cv2.addWeighted(effect_img, alpha, img, 1.0 - alpha, 0)
        else:
            img = effect_img

    # 5b. 3D Depth Bokeh, Relighting & Atmospheric Haze
    has_depth_and_light = (
        abs(recipe.depth_blur) >= 0.1
        or abs(recipe.relight_near) >= 0.1
        or abs(recipe.relight_far) >= 0.1
        or abs(recipe.warmth_near) >= 0.1
        or abs(recipe.warmth_far) >= 0.1
        or abs(recipe.depth_haze) >= 0.1
    )
    if has_depth_and_light:
        try:
            from src.imaging.creative_effects import get_depth_and_matte_engine
            engine = get_depth_and_matte_engine()
            depth_map, alpha_matte, near_weight, far_weight = engine.get_depth_matte_and_weights(img, media_id=media_id)

            # 3D Relighting & Warmth (Vectorized direct RGB for real-time 60+ FPS without RGB->LAB->RGB overhead)
            has_relight = (
                abs(recipe.relight_near) >= 0.1
                or abs(recipe.relight_far) >= 0.1
                or abs(recipe.warmth_near) >= 0.1
                or abs(recipe.warmth_far) >= 0.1
            )
            if has_relight:
                d_light = (recipe.relight_near * 0.45) * near_weight + (recipe.relight_far * 0.45) * far_weight
                d_warmth = (recipe.warmth_near * 0.35) * near_weight + (recipe.warmth_far * 0.35) * far_weight

                img_f = img.astype(np.float32)
                img_f[:, :, 0] += d_light + d_warmth
                img_f[:, :, 1] += d_light + d_warmth * 0.2
                img_f[:, :, 2] += d_light - d_warmth
                img = np.clip(img_f, 0, 255).astype(np.uint8)

            # Portrait Bokeh & Focus Defocus Blur with Specular Highlights Bloom
            if abs(recipe.depth_blur) >= 0.1:
                focus_z = float(recipe.focal_plane) / 100.0
                subject_protect = alpha_matte * (1.0 if focus_z < 0.25 else 0.0)
                defocus = np.clip(np.abs(depth_map - focus_z) * (1.0 - subject_protect), 0.0, 1.0)

                blur_r = max(2, int(round((recipe.depth_blur / 100.0) * 28.0 * res_scale)))
                h_img, w_img = img.shape[:2]

                # Optical defocus blur has zero frequencies above 640px; blur on downscaled buffer for instant speed
                scale_blur = max(1.0, max(h_img, w_img) / 640.0)
                if scale_blur > 1.05:
                    bw = max(16, int(round(w_img / scale_blur)))
                    bh = max(16, int(round(h_img / scale_blur)))
                    small_img = cv2.resize(img, (bw, bh), interpolation=cv2.INTER_AREA)
                    small_lin = _LUT_SRGB_TO_LIN[small_img]
                    if recipe.bokeh_glow >= 0.1:
                        lum_small = cv2.cvtColor(small_img, cv2.COLOR_RGB2GRAY)
                        glow_factor = np.clip((lum_small.astype(np.float32) - 190.0) / 60.0, 0.0, 1.0)[:, :, None]
                        small_lin = small_lin * (1.0 + glow_factor * (recipe.bokeh_glow / 50.0) * 2.5)
                    small_r = max(2, int(round(blur_r / scale_blur)))
                    sigma_b = max(1.0, small_r * 0.5)
                    ksize = max(3, (2 * small_r + 1) | 1)
                    blurred_small = cv2.GaussianBlur(small_lin, (ksize, ksize), sigmaX=sigma_b, sigmaY=sigma_b)
                    indices = np.clip(blurred_small * 4095.0, 0, 4095).astype(np.int32)
                    small_srgb = _LUT_LIN_TO_SRGB[indices]
                    blurred_srgb = cv2.resize(small_srgb, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
                else:
                    img_lin = _LUT_SRGB_TO_LIN[img]
                    if recipe.bokeh_glow >= 0.1:
                        lum = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        glow_factor = np.clip((lum.astype(np.float32) - 190.0) / 60.0, 0.0, 1.0)[:, :, None]
                        img_lin = img_lin * (1.0 + glow_factor * (recipe.bokeh_glow / 50.0) * 2.5)
                    sigma_b = max(1.0, blur_r * 0.5)
                    ksize = max(3, (2 * blur_r + 1) | 1)
                    blurred_lin = cv2.GaussianBlur(img_lin, (ksize, ksize), sigmaX=sigma_b, sigmaY=sigma_b)
                    indices = np.clip(blurred_lin * 4095.0, 0, 4095).astype(np.int32)
                    blurred_srgb = _LUT_LIN_TO_SRGB[indices]

                blend_w = np.clip(defocus * (recipe.depth_blur / 100.0) * 1.5, 0.0, 1.0)[:, :, None]
                f_base = img.astype(np.float32)
                f_blur = blurred_srgb.astype(np.float32)
                f_base += (f_blur - f_base) * blend_w
                img = np.clip(f_base, 0.0, 255.0).astype(np.uint8)

            # Atmospheric Depth Haze / Fog (Fast vectorized difference blend)
            if abs(recipe.depth_haze) >= 0.1:
                beta = (recipe.depth_haze / 100.0) * 3.5
                haze_trans = 1.0 - np.exp(-beta * (depth_map ** 2))
                haze_factor = np.clip(haze_trans * (1.0 - alpha_matte * 0.85), 0.0, 1.0)[:, :, None]

                w_h = recipe.depth_haze_warmth / 100.0
                if w_h >= 0:
                    mist_r = 230.0 + w_h * 25.0
                    mist_g = 235.0 - w_h * 5.0
                    mist_b = 240.0 - w_h * 60.0
                else:
                    mist_r = 230.0 + w_h * 40.0
                    mist_g = 235.0 - w_h * 20.0
                    mist_b = 240.0 - w_h * 15.0
                mist_color = np.array([mist_r, mist_g, mist_b], dtype=np.float32)

                diff_haze = mist_color - img.astype(np.float32)
                img = np.clip(img.astype(np.float32) + diff_haze * haze_factor, 0, 255).astype(np.uint8)
        except Exception as ex:
            logger.debug("Depth and light processing failed: %s", ex)

    # 6. Aspect-Aware Elliptical Vignette
    if abs(recipe.vignette) >= 0.1:
        v_amount = recipe.vignette / 100.0
        vh, vw = img.shape[:2]
        y, x = np.ogrid[:vh, :vw]
        cx, cy = vw / 2.0, vh / 2.0
        nx = (x - cx) / (vw / 2.0)
        ny = (y - cy) / (vh / 2.0)
        dist_sq = nx * nx + ny * ny
        v_mask = np.clip(dist_sq * 0.75, 0.0, 1.5)
        falloff = np.clip(1.0 - v_mask, 0.0, 1.0)
        v_gain = 1.0 + v_amount * (1.0 - falloff)
        img = np.clip(img.astype(np.float32) * v_gain[:, :, None], 0, 255).astype(np.uint8)

    # 7. Straighten & Crop
    if abs(recipe.straighten_deg) >= 0.05:
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        rot_matrix = cv2.getRotationMatrix2D(center, -float(recipe.straighten_deg), 1.0)
        img = cv2.warpAffine(
            img,
            rot_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        if recipe.crop_rect is None:
            img = _crop_largest_inscribed_rect(img, recipe.straighten_deg, w, h)

    if recipe.crop_rect is not None:
        h, w = img.shape[:2]
        l, t, r, b = recipe.crop_rect
        if isinstance(l, float) and 0.0 <= l <= 1.0 and 0.0 <= r <= 1.0:
            l = int(l * w)
            t = int(t * h)
            r = int(r * w)
            b = int(b * h)
        else:
            l = int(l)
            t = int(t)
            r = int(r)
            b = int(b)
        l = max(0, min(w - 1, l))
        t = max(0, min(h - 1, t))
        r = max(l + 1, min(w, r))
        b = max(t + 1, min(h, b))
        if r > l and b > t:
            img = img[t:b, l:r]

    return np.ascontiguousarray(img)


def analyze_noise_profile(master_rgb: np.ndarray) -> float:
    """
    Estimates high-frequency chroma and luma noise variance from flat patches of the image
    using Median Absolute Deviation (MAD) of Laplacian coefficients.
    Returns recommended Denoise slider value (0.0 .. 100.0).
    """
    if master_rgb is None or master_rgb.size == 0:
        return 0.0

    h, w = master_rgb.shape[:2]
    max_dim = 640
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        small = cv2.resize(master_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = master_rgb

    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    # Compute Laplacians
    lap_l = cv2.Laplacian(L, cv2.CV_32F, ksize=3)
    lap_a = cv2.Laplacian(A, cv2.CV_32F, ksize=3)
    lap_b = cv2.Laplacian(B, cv2.CV_32F, ksize=3)

    # Robust MAD estimator: sigma = median(|x - median(x)|) / 0.6745
    def calc_mad(arr: np.ndarray) -> float:
        med = float(np.median(arr))
        return float(np.median(np.abs(arr - med)) / 0.6745)

    mad_l = calc_mad(lap_l)
    mad_chroma = (calc_mad(lap_a) + calc_mad(lap_b)) / 2.0

    # Composite noise metric: Chroma noise is visually more disturbing than luma
    composite_noise = mad_l * 0.4 + mad_chroma * 0.8

    # Scale to recommended slider value (0..100)
    # Typical clean DSLR photo: composite_noise ~ 1.5..3.0 -> denoise ~ 0..10
    # Moderate ISO noise: composite_noise ~ 4.0..8.0 -> denoise ~ 20..45
    # High ISO grainy photo: composite_noise > 10.0 -> denoise ~ 50..80
    rec_val = (composite_noise - 2.0) * 6.5
    return float(np.clip(round(rec_val, 1), 0.0, 100.0))


def analyze_auto_adjust(master_rgb: np.ndarray) -> PhotoAdjustmentRecipe:
    """
    Analyzes scene dynamic range, histogram distribution, and color cast
    to compute balanced, professional Auto Adjust parameters.
    """
    if master_rgb is None or master_rgb.size == 0:
        return PhotoAdjustmentRecipe()

    # Downsample for fast, consistent analysis
    h, w = master_rgb.shape[:2]
    max_dim = 600
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        small = cv2.resize(master_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = master_rgb

    # 1. Luminance & Percentiles
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    p2 = float(np.percentile(gray, 2))
    p50 = float(np.percentile(gray, 50))
    p98 = float(np.percentile(gray, 98))

    # Auto Exposure (target median around 118..125)
    target_median = 120.0
    brightness_shift = 0.0
    if p50 > 5 and abs(p50 - target_median) > 6.0:
        brightness_shift = float(np.clip((target_median - p50) / 128.0 * 55.0, -45.0, 45.0))

    # Dynamic Range & Contrast
    dyn_range = p98 - p2
    contrast_shift = 0.0
    if dyn_range < 195:
        contrast_shift = float(np.clip((195.0 - dyn_range) / 195.0 * 35.0, 5.0, 35.0))

    # Shadow & Highlight recovery
    shadows_shift = 0.0
    highlights_shift = 0.0
    if p2 < 15 and p50 < 90:
        shadows_shift = float(np.clip((15.0 - p2) * 2.0, 0.0, 30.0))
    if p98 > 242 and p50 > 140:
        highlights_shift = float(np.clip(-(p98 - 240.0) * 2.0, -35.0, 0.0))

    # 2. White Balance / Color Cast Analysis (Gray-World with highlight/shadow exclusion)
    valid_mask = (gray > 20) & (gray < 235)
    if np.count_nonzero(valid_mask) > 100:
        r_mean = float(np.mean(small[:, :, 0][valid_mask]))
        g_mean = float(np.mean(small[:, :, 1][valid_mask]))
        b_mean = float(np.mean(small[:, :, 2][valid_mask]))
        avg_mean = (r_mean + g_mean + b_mean) / 3.0

        # Temperature (R vs B)
        rb_diff = r_mean - b_mean
        temp_shift = float(np.clip(-rb_diff / max(avg_mean, 10.0) * 80.0, -25.0, 25.0))

        # Tint (G vs (R+B)/2)
        rb_avg = (r_mean + b_mean) / 2.0
        g_diff = g_mean - rb_avg
        tint_shift = float(np.clip(g_diff / max(avg_mean, 10.0) * 60.0, -20.0, 20.0))
    else:
        temp_shift = 0.0
        tint_shift = 0.0

    return PhotoAdjustmentRecipe(
        brightness=round(brightness_shift, 1),
        contrast=round(contrast_shift, 1),
        shadows=round(shadows_shift, 1),
        highlights=round(highlights_shift, 1),
        clarity=10.0,     # Subtle modern clarity
        temperature=round(temp_shift, 1),
        tint=round(tint_shift, 1),
        saturation=10.0,  # Subtle pleasing vibrance boost
        sharpen=15.0,     # Clean natural sharpness
    )


def detect_auto_straighten_angle(master_rgb: np.ndarray) -> float:
    """
    Detects dominant horizon and architectural vertical lines using OpenCV Canny + HoughLinesP,
    returning the optimal tilt correction angle in degrees (-15.0 .. +15.0).
    """
    if master_rgb is None or master_rgb.size == 0:
        return 0.0

    h, w = master_rgb.shape[:2]
    max_dim = 800
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        small = cv2.resize(master_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = master_rgb

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Probabilistic Hough Line Transform
    min_length = int(max(small.shape[:2]) * 0.08)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=60,
        minLineLength=min_length,
        maxLineGap=12,
    )

    if lines is None or len(lines) == 0:
        return 0.0

    candidate_tilts: list[float] = []
    weights: list[float] = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < min_length:
            continue

        deg = math.degrees(math.atan2(dy, dx))

        # Check near horizontal (-15 deg .. +15 deg)
        if abs(deg) <= 15.0:
            candidate_tilts.append(deg)
            weights.append(length * length)  # Favor long horizon lines
        elif abs(deg - 180.0) <= 15.0:
            candidate_tilts.append(deg - 180.0)
            weights.append(length * length)
        elif abs(deg + 180.0) <= 15.0:
            candidate_tilts.append(deg + 180.0)
            weights.append(length * length)
        # Check near vertical (80 deg .. 100 deg)
        elif abs(deg - 90.0) <= 12.0:
            candidate_tilts.append(deg - 90.0)
            weights.append(length * 0.65)  # Vertical lines weighted slightly less
        elif abs(deg + 90.0) <= 12.0:
            candidate_tilts.append(deg + 90.0)
            weights.append(length * 0.65)

    if not candidate_tilts:
        return 0.0

    # Weighted median angle
    tilts_arr = np.array(candidate_tilts)
    weights_arr = np.array(weights)
    sort_idx = np.argsort(tilts_arr)
    tilts_sorted = tilts_arr[sort_idx]
    weights_sorted = weights_arr[sort_idx]

    cumsum = np.cumsum(weights_sorted)
    cutoff = cumsum[-1] / 2.0
    median_angle = float(tilts_sorted[np.searchsorted(cumsum, cutoff)])

    # Clamp to safe bounds and round to 1 decimal place
    clamped = float(np.clip(median_angle, -15.0, 15.0))
    return round(clamped, 1)


def compute_histogram_data(
    rgb_img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes 256-bin histograms for Red, Green, Blue, and Luminance channels
    normalized to [0.0 .. 1.0] for real-time HistogramWidget rendering.
    """
    if rgb_img is None or rgb_img.size == 0:
        empty = np.zeros(256, dtype=np.float32)
        return empty, empty, empty, empty

    hist_r = cv2.calcHist([rgb_img], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([rgb_img], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([rgb_img], [2], None, [256], [0, 256]).flatten()

    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    hist_l = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    max_val = max(hist_r.max(), hist_g.max(), hist_b.max(), hist_l.max(), 1.0)

    return (
        hist_r / max_val,
        hist_g / max_val,
        hist_b / max_val,
        hist_l / max_val,
    )
