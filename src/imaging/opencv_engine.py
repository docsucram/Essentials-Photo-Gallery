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

import logging
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Thread-safe bounded checkpoint cache for render_adjusted_image
_STAGE_CACHE_LOCK = threading.Lock()
_STAGE_CACHE: OrderedDict[tuple[Any, int, int], dict[str, Any]] = OrderedDict()
_STAGE_CACHE_MAX_ENTRIES = 6


def _get_stage_slot(cache_id: tuple[Any, int, int]) -> dict[str, Any]:
    with _STAGE_CACHE_LOCK:
        slot = _STAGE_CACHE.get(cache_id)
        if slot is None:
            slot = {}
            _STAGE_CACHE[cache_id] = slot
            while len(_STAGE_CACHE) > _STAGE_CACHE_MAX_ENTRIES:
                _STAGE_CACHE.popitem(last=False)
        else:
            _STAGE_CACHE.move_to_end(cache_id)
        return slot


def clear_stage_cache() -> None:
    """Clears the intermediate render stage cache."""
    with _STAGE_CACHE_LOCK:
        _STAGE_CACHE.clear()

# Precomputed high-precision sRGB <-> Linear floating-point look-up tables
_LUT_SRGB_TO_LIN: np.ndarray = ((np.arange(256, dtype=np.float32) / 255.0) ** 2.2).astype(np.float32)
_LUT_LIN_TO_SRGB: np.ndarray = np.clip(
    (np.linspace(0.0, 1.0, 4096, dtype=np.float32) ** (1.0 / 2.2)) * 255.0 + 0.5, 0, 255
).astype(np.uint8)
_VIBRANCE_POW_LUT: np.ndarray = (np.linspace(0.0, 1.0, 256, dtype=np.float32) ** 0.8).astype(np.float32)


HSL_COLOR_BANDS: tuple[tuple[str, str, int, str], ...] = (
    # (key, display_name, opencv_hue_center_0_180, swatch_hex)
    ("red", "Red", 0, "#E04848"),
    ("orange", "Orange", 14, "#E88B38"),
    ("yellow", "Yellow", 28, "#DFC438"),
    ("green", "Green", 58, "#52B856"),
    ("aqua", "Aqua", 88, "#38B6B6"),
    ("blue", "Blue", 112, "#4680E0"),
    ("purple", "Purple", 136, "#9658DC"),
    ("magenta", "Magenta", 160, "#DC50A8"),
)


def _default_curve_points() -> list[tuple[float, float]]:
    return [(0.0, 0.0), (1.0, 1.0)]


def _is_identity_curve(pts: Sequence[tuple[float, float]] | None) -> bool:
    if not pts or len(pts) < 2:
        return True
    if len(pts) == 2:
        (x0, y0), (x1, y1) = pts[0], pts[1]
        return abs(x0) < 1e-4 and abs(y0) < 1e-4 and abs(x1 - 1.0) < 1e-4 and abs(y1 - 1.0) < 1e-4
    return all(abs(float(x) - float(y)) < 1e-3 for x, y in pts)


def _build_curve_lut(pts: Sequence[tuple[float, float]]) -> np.ndarray:
    """Build a smooth 256-entry uint8 LUT from normalized (x, y) control points using PCHIP interpolation."""
    clean = sorted((max(0.0, min(1.0, float(x))), max(0.0, min(1.0, float(y)))) for x, y in pts)
    dedup: list[tuple[float, float]] = []
    for x, y in clean:
        if dedup and abs(x - dedup[-1][0]) < 1e-4:
            dedup[-1] = (dedup[-1][0], y)
        else:
            dedup.append((x, y))
    if len(dedup) < 2:
        return np.arange(256, dtype=np.uint8)

    xs = np.array([p[0] for p in dedup], dtype=np.float64)
    ys = np.array([p[1] for p in dedup], dtype=np.float64)
    n = len(xs)
    x_eval = np.linspace(0.0, 1.0, 256, dtype=np.float64)

    if n == 2:
        y_eval = np.interp(x_eval, xs, ys)
        return np.clip(np.round(y_eval * 255.0), 0, 255).astype(np.uint8)

    # Fritsch-Carlson monotonic cubic Hermite spline (PCHIP) — zero overshoot
    h = np.diff(xs)
    delta = np.diff(ys) / np.maximum(h, 1e-8)
    d = np.zeros(n, dtype=np.float64)
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] > 0.0:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    d[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / max(h[0] + h[1], 1e-8)
    if d[0] * delta[0] <= 0.0:
        d[0] = 0.0
    elif abs(d[0]) > 3.0 * abs(delta[0]):
        d[0] = 3.0 * delta[0]
    d[-1] = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / max(h[-1] + h[-2], 1e-8)
    if d[-1] * delta[-1] <= 0.0:
        d[-1] = 0.0
    elif abs(d[-1]) > 3.0 * abs(delta[-1]):
        d[-1] = 3.0 * delta[-1]

    y_eval = np.empty(256, dtype=np.float64)
    idx = np.clip(np.searchsorted(xs, x_eval, side="right") - 1, 0, n - 2)
    t = (x_eval - xs[idx]) / np.maximum(h[idx], 1e-8)
    t = np.where(x_eval <= xs[0], 0.0, np.where(x_eval >= xs[-1], 1.0, t))
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    y_eval = (
        h00 * ys[idx]
        + h10 * h[idx] * d[idx]
        + h01 * ys[idx + 1]
        + h11 * h[idx] * d[idx + 1]
    )
    y_eval = np.where(x_eval <= xs[0], ys[0], np.where(x_eval >= xs[-1], ys[-1], y_eval))
    return np.clip(np.round(y_eval * 255.0), 0, 255).astype(np.uint8)


def _apply_tone_curves(img: np.ndarray, recipe: PhotoAdjustmentRecipe) -> np.ndarray:
    has_rgb = not _is_identity_curve(recipe.curve_rgb)
    has_r = not _is_identity_curve(recipe.curve_r)
    has_g = not _is_identity_curve(recipe.curve_g)
    has_b = not _is_identity_curve(recipe.curve_b)
    if not (has_rgb or has_r or has_g or has_b):
        return img

    lut_rgb = _build_curve_lut(recipe.curve_rgb) if has_rgb else np.arange(256, dtype=np.uint8)
    if has_r or has_g or has_b:
        lut_r = _build_curve_lut(recipe.curve_r) if has_r else np.arange(256, dtype=np.uint8)
        lut_g = _build_curve_lut(recipe.curve_g) if has_g else np.arange(256, dtype=np.uint8)
        lut_b = _build_curve_lut(recipe.curve_b) if has_b else np.arange(256, dtype=np.uint8)
        # Compose master RGB curve followed by per-channel curve into a single 3-channel LUT
        lut_3 = np.stack([lut_r[lut_rgb], lut_g[lut_rgb], lut_b[lut_rgb]], axis=-1).reshape(1, 256, 3)
        return cv2.LUT(img, lut_3)
    return cv2.LUT(img, lut_rgb)


def _apply_hsl_color_mixer(img: np.ndarray, recipe: PhotoAdjustmentRecipe) -> np.ndarray:
    active_bands = [
        (key, center, float(recipe.hsl_hue.get(key, 0.0)), float(recipe.hsl_sat.get(key, 0.0)), float(recipe.hsl_lum.get(key, 0.0)))
        for key, _, center, _ in HSL_COLOR_BANDS
        if abs(recipe.hsl_hue.get(key, 0.0)) >= 0.1
        or abs(recipe.hsl_sat.get(key, 0.0)) >= 0.1
        or abs(recipe.hsl_lum.get(key, 0.0)) >= 0.1
    ]
    if not active_bands:
        return img

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    H = hsv[:, :, 0]  # 0 .. 180
    S = hsv[:, :, 1]  # 0 .. 255
    V = hsv[:, :, 2]  # 0 .. 255

    # Attenuate adjustments on near-neutral grays so neutral highlights/shadows stay clean
    chroma_gate = np.clip(S / 28.0, 0.0, 1.0)
    dh = np.zeros_like(H)
    ds = np.zeros_like(S)
    dv = np.zeros_like(V)

    half_width = 22.0  # ~44 deg in 360-space
    for _, center, h_adj, s_adj, l_adj in active_bands:
        dist = np.abs(H - float(center))
        dist = np.minimum(dist, 180.0 - dist)
        w = np.clip(1.0 - dist / half_width, 0.0, 1.0)
        w = w * w * (3.0 - 2.0 * w) * chroma_gate
        if abs(h_adj) >= 0.1:
            dh += w * (h_adj * 0.14)  # +/- 14 in [0..180] = +/- 28 deg
        if abs(s_adj) >= 0.1:
            ds += w * (s_adj / 100.0)
        if abs(l_adj) >= 0.1:
            dv += w * (l_adj / 100.0)

    hsv[:, :, 0] = np.mod(H + dh, 180.0)
    hsv[:, :, 1] = np.clip(S * (1.0 + ds), 0.0, 255.0)
    # Luminance adjustment applied softly in V so highlights don't clip abruptly
    v_norm = V / 255.0
    v_shift = dv * 0.45 * (1.0 - 0.5 * v_norm)
    hsv[:, :, 2] = np.clip((v_norm + v_shift * v_norm) * 255.0, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _apply_film_grain(img: np.ndarray, amount: float, size: float) -> np.ndarray:
    if amount < 0.5:
        return img
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed=42)
    scale_factor = 1.0 + (max(0.0, min(100.0, size)) / 100.0) * 2.2
    gh = max(8, int(round(h / scale_factor)))
    gw = max(8, int(round(w / scale_factor)))
    noise_small = rng.normal(0.0, 1.0, size=(gh, gw)).astype(np.float32)
    if gh != h or gw != w:
        noise = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        noise = noise_small

    img_f = img.astype(np.float32)
    lum = (0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]) / 255.0
    mid_weight = 0.35 + 0.65 * np.clip(4.0 * lum * (1.0 - lum), 0.0, 1.0)
    strength = (amount / 100.0) * 24.0
    img_f += (noise * mid_weight * strength)[:, :, None]
    return np.clip(img_f, 0.0, 255.0).astype(np.uint8)


def compute_wb_from_rgb_sample(r: float, g: float, b: float) -> tuple[float, float]:
    """
    Given a sampled neutral gray/white RGB value (0..255), return (temperature, tint)
    in [-100, 100] that neutralizes the color cast in CIE LAB space.
    """
    px = np.array([[[np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)]]], dtype=np.uint8)
    lab = cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0]
    a_val = float(lab[1]) - 128.0
    b_val = float(lab[2]) - 128.0
    # In render_adjusted_image: B += temperature * 0.35, A += tint * 0.35
    temp = float(np.clip(round(-b_val / 0.35), -100.0, 100.0))
    tint = float(np.clip(round(-a_val / 0.35), -100.0, 100.0))
    return temp, tint


def _region_dict_has_edits(reg_dict: dict[str, Any] | None) -> bool:
    if not reg_dict:
        return False
    for key in (
        "brightness",
        "contrast",
        "shadows",
        "highlights",
        "whites",
        "blacks",
        "clarity",
        "structure",
        "temperature",
        "tint",
        "saturation",
        "dehaze",
    ):
        if abs(float(reg_dict.get(key, 0.0))) >= 0.1:
            return True
    for c_key in ("curve_rgb", "curve_r", "curve_g", "curve_b"):
        if not _is_identity_curve(reg_dict.get(c_key)):
            return True
    if reg_dict.get("mask_strokes"):
        return True
    return False


@dataclass
class PhotoAdjustmentRecipe:
    """Non-destructive adjustment parameter stack applied to the master image."""

    brightness: float = 0.0       # -100.0 .. +100.0 (Exposure shift with filmic shoulder)
    smart_enhance: float = 0.0    # 0.0 .. 100.0 (Adaptive Accent-AI single-slider balance)
    contrast: float = 0.0         # -100.0 .. +100.0 (Filmic S-curve)
    shadows: float = 0.0          # -100.0 .. +100.0 (Edge-aware shadow lift / compress)
    highlights: float = 0.0       # -100.0 .. +100.0 (Highlight recovery)
    whites: float = 0.0           # -100.0 .. +100.0 (Upper highlight endpoint stretch/compress)
    blacks: float = 0.0           # -100.0 .. +100.0 (Deep shadow toe stretch/compress)
    clarity: float = 0.0          # -100.0 .. +100.0 (Midtone local contrast)
    structure: float = 0.0        # -100.0 .. +100.0 (Content-aware detail with skin/sky protection)
    temperature: float = 0.0      # -100.0 .. +100.0 (Cool blue <-> Warm amber)
    tint: float = 0.0             # -100.0 .. +100.0 (Green <-> Magenta)
    saturation: float = 0.0       # -100.0 .. +100.0 (Skin-safe vibrance / chroma scale)
    dehaze: float = 0.0           # -100.0 .. +100.0 (Atmospheric haze reduction)
    vignette: float = 0.0         # -100.0 .. +100.0 (Dark <-> Light corners)
    straighten_deg: float = 0.0   # -15.0 .. +15.0 degrees
    straighten_ai_fill: bool = False  # Fill rotated corner wedges instead of cropping inward
    sharpen: float = 0.0          # 0.0 .. 100.0 (Halo-free unsharp mask amount)
    sharpen_radius: float = 25.0  # 0.0 .. 100.0 (25 = 1.0px nominal radius)
    sharpen_masking: float = 0.0  # 0.0 .. 100.0 (Edge-only sharpening threshold)
    denoise: float = 0.0          # 0.0 .. 100.0 (Luminance / combined bilateral amount)
    denoise_detail: float = 50.0  # 0.0 .. 100.0 (Luma denoise edge/detail preservation)
    denoise_chroma: float = 0.0   # 0.0 .. 100.0 (Independent chroma denoise; 0 = follow denoise)
    grain_amount: float = 0.0     # 0.0 .. 100.0 (Organic film grain strength)
    grain_size: float = 25.0      # 0.0 .. 100.0 (Film grain particle scale)
    orton_glow: float = 0.0       # 0.0 .. 100.0 (Dreamy highlight bloom + soft contrast)
    golden_hour: float = 0.0      # 0.0 .. 100.0 (Warm golden-hour highlight/midtone glow)
    black_point: float = 0.0      # 0.0 .. 255.0 (Histogram black point clip)
    white_point: float = 255.0    # 0.0 .. 255.0 (Histogram white point clip)
    curve_rgb: list[tuple[float, float]] = field(default_factory=_default_curve_points)
    curve_r: list[tuple[float, float]] = field(default_factory=_default_curve_points)
    curve_g: list[tuple[float, float]] = field(default_factory=_default_curve_points)
    curve_b: list[tuple[float, float]] = field(default_factory=_default_curve_points)
    hsl_hue: dict[str, float] = field(default_factory=dict)
    hsl_sat: dict[str, float] = field(default_factory=dict)
    hsl_lum: dict[str, float] = field(default_factory=dict)
    vignette_feather: float = 50.0  # 0.0 .. 100.0 (Vignette falloff softness)
    region_edits: dict[str, dict[str, Any]] = field(default_factory=dict)  # "subject" | "background" | "sky"
    crop_rect: tuple[int, int, int, int] | None = None  # (left, top, right, bottom)
    effect: str = "none"          # "none" | "classic_chrome" | "warm_portrait" | "vivid_landscape" | "matte_vintage" | "fine_art_bw" | "cool_cinema"
    effect_intensity: float = 100.0  # 0.0 .. 100.0 (% blend with base adjusted image)
    selective_points: list[dict[str, Any]] = field(default_factory=list)  # list of {x, y, radius, brightness, contrast, saturation} (supports normalized [0.0, 1.0])

    # 3D Depth & Light Adjustments
    depth_blur: float = 0.0       # 0.0 .. 100.0 (Optical disc bokeh blur strength)
    focal_plane: float = 0.0      # 0.0 .. 100.0 (In-focus depth plane: 0=foreground, 100=background)
    bokeh_glow: float = 0.0       # 0.0 .. 100.0 (Linear-light specular highlight bloom)
    boundary_shift: float = 0.0   # -20.0 .. +20.0 (Shrink <-> Grow subject mask boundary)
    boundary_feather: float = 0.0 # 0.0 .. 20.0 (Soft feather radius along subject boundary)
    relight_near: float = 0.0     # -100.0 .. +100.0 (Fill light on foreground subject)
    relight_far: float = 0.0      # -100.0 .. +100.0 (Exposure shift on distant background)
    warmth_near: float = 0.0      # -100.0 .. +100.0 (CIE LAB B* warmth shift on foreground)
    warmth_far: float = 0.0       # -100.0 .. +100.0 (CIE LAB B* warmth shift on background)
    relight_ambient_dim: float = 0.0  # 0.0 .. 100.0 (Dim original ambient light in linear space)
    virtual_lights: list[dict[str, Any]] = field(default_factory=list)  # Up to 3 virtual 3D lights
    depth_haze: float = 0.0       # 0.0 .. 100.0 (Beer-Lambert exponential atmospheric depth mist)
    depth_haze_warmth: float = 0.0 # -100.0 .. +100.0 (Mist tint: cool blue <-> golden amber)

    def is_default(self) -> bool:
        """Returns True if no visual adjustments are active."""
        return (
            abs(self.brightness) < 0.1
            and abs(self.smart_enhance) < 0.1
            and abs(self.contrast) < 0.1
            and abs(self.shadows) < 0.1
            and abs(self.highlights) < 0.1
            and abs(self.whites) < 0.1
            and abs(self.blacks) < 0.1
            and abs(self.clarity) < 0.1
            and abs(self.structure) < 0.1
            and abs(self.temperature) < 0.1
            and abs(self.tint) < 0.1
            and abs(self.saturation) < 0.1
            and abs(self.dehaze) < 0.1
            and abs(self.vignette) < 0.1
            and abs(self.orton_glow) < 0.1
            and abs(self.golden_hour) < 0.1
            and abs(self.straighten_deg) < 0.01
            and abs(self.sharpen) < 0.1
            and abs(self.sharpen_radius - 25.0) < 0.1
            and abs(self.sharpen_masking) < 0.1
            and abs(self.denoise) < 0.1
            and abs(self.denoise_detail - 50.0) < 0.1
            and abs(self.denoise_chroma) < 0.1
            and abs(self.grain_amount) < 0.1
            and abs(self.black_point) < 0.1
            and abs(self.white_point - 255.0) < 0.1
            and _is_identity_curve(self.curve_rgb)
            and _is_identity_curve(self.curve_r)
            and _is_identity_curve(self.curve_g)
            and _is_identity_curve(self.curve_b)
            and all(abs(v) < 0.1 for v in self.hsl_hue.values())
            and all(abs(v) < 0.1 for v in self.hsl_sat.values())
            and all(abs(v) < 0.1 for v in self.hsl_lum.values())
            and not any(_region_dict_has_edits(d) for d in self.region_edits.values())
            and self.crop_rect is None
            and self.effect in ("none", "as_shot", "")
            and len(self.selective_points) == 0
            and abs(self.depth_blur) < 0.1
            and abs(self.focal_plane) < 0.1
            and abs(self.bokeh_glow) < 0.1
            and abs(self.boundary_shift) < 0.1
            and abs(self.boundary_feather) < 0.1
            and abs(self.relight_near) < 0.1
            and abs(self.relight_far) < 0.1
            and abs(self.warmth_near) < 0.1
            and abs(self.warmth_far) < 0.1
            and abs(self.relight_ambient_dim) < 0.1
            and not any(
                bool(lt.get("enabled", True)) and float(lt.get("intensity", 0.0)) >= 0.1
                for lt in self.virtual_lights
            )
            and abs(self.depth_haze) < 0.1
            and abs(self.depth_haze_warmth) < 0.1
        )

    def copy(self) -> PhotoAdjustmentRecipe:
        return PhotoAdjustmentRecipe(
            brightness=self.brightness,
            smart_enhance=self.smart_enhance,
            contrast=self.contrast,
            shadows=self.shadows,
            highlights=self.highlights,
            whites=self.whites,
            blacks=self.blacks,
            clarity=self.clarity,
            structure=self.structure,
            temperature=self.temperature,
            tint=self.tint,
            saturation=self.saturation,
            dehaze=self.dehaze,
            vignette=self.vignette,
            orton_glow=self.orton_glow,
            golden_hour=self.golden_hour,
            straighten_deg=self.straighten_deg,
            straighten_ai_fill=self.straighten_ai_fill,
            sharpen=self.sharpen,
            sharpen_radius=self.sharpen_radius,
            sharpen_masking=self.sharpen_masking,
            denoise=self.denoise,
            denoise_detail=self.denoise_detail,
            denoise_chroma=self.denoise_chroma,
            grain_amount=self.grain_amount,
            grain_size=self.grain_size,
            black_point=self.black_point,
            white_point=self.white_point,
            curve_rgb=[tuple(p) for p in self.curve_rgb],
            curve_r=[tuple(p) for p in self.curve_r],
            curve_g=[tuple(p) for p in self.curve_g],
            curve_b=[tuple(p) for p in self.curve_b],
            hsl_hue=dict(self.hsl_hue),
            hsl_sat=dict(self.hsl_sat),
            hsl_lum=dict(self.hsl_lum),
            vignette_feather=self.vignette_feather,
            region_edits={
                k: {
                    fk: ([tuple(pt) for pt in fv] if fk in ("curve_rgb", "curve_r", "curve_g", "curve_b") else fv)
                    for fk, fv in d.items()
                }
                for k, d in self.region_edits.items()
            },
            crop_rect=self.crop_rect,
            effect=self.effect,
            effect_intensity=self.effect_intensity,
            selective_points=[dict(p) for p in self.selective_points],
            depth_blur=self.depth_blur,
            focal_plane=self.focal_plane,
            bokeh_glow=self.bokeh_glow,
            boundary_shift=self.boundary_shift,
            boundary_feather=self.boundary_feather,
            relight_near=self.relight_near,
            relight_far=self.relight_far,
            warmth_near=self.warmth_near,
            warmth_far=self.warmth_far,
            relight_ambient_dim=self.relight_ambient_dim,
            virtual_lights=[dict(lt) for lt in self.virtual_lights],
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


_CUBE_LUT_CACHE: dict[str, np.ndarray] = {}


def _load_cube_lut(lut_path: str) -> np.ndarray | None:
    """Parse a standard 3D .cube LUT file into shape (N, N, N, 3) indexed [B, G, R]."""
    cached = _CUBE_LUT_CACHE.get(lut_path)
    if cached is not None:
        return cached
    try:
        size = 0
        domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        entries: list[tuple[float, float, float]] = []
        with open(lut_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                upper = line.upper()
                if upper.startswith("LUT_3D_SIZE"):
                    parts = line.split()
                    if len(parts) >= 2:
                        size = int(parts[1])
                    continue
                if upper.startswith("DOMAIN_MIN"):
                    parts = line.split()
                    if len(parts) >= 4:
                        domain_min = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
                    continue
                if upper.startswith("DOMAIN_MAX"):
                    parts = line.split()
                    if len(parts) >= 4:
                        domain_max = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
                    continue
                if upper.startswith(("TITLE", "LUT_1D_SIZE")):
                    continue
                parts = line.split()
                if len(parts) == 3:
                    entries.append((float(parts[0]), float(parts[1]), float(parts[2])))
        if size <= 1 or len(entries) != size * size * size:
            return None
        lut = np.array(entries, dtype=np.float32).reshape((size, size, size, 3))
        span = np.maximum(domain_max - domain_min, 1e-6)
        if np.any(domain_min != 0.0) or np.any(domain_max != 1.0):
            lut = np.clip((lut - domain_min) / span, 0.0, 1.0)
        _CUBE_LUT_CACHE[lut_path] = lut
        return lut
    except Exception:
        return None


def _load_and_apply_cube_lut(rgb_img: np.ndarray, lut_path: str) -> np.ndarray:
    """Apply a cached 3D .cube LUT via vectorized trilinear interpolation."""
    lut = _load_cube_lut(lut_path)
    if lut is None:
        return rgb_img
    n = lut.shape[0]
    scale = float(n - 1) / 255.0
    rf = rgb_img[:, :, 0].astype(np.float32) * scale
    gf = rgb_img[:, :, 1].astype(np.float32) * scale
    bf = rgb_img[:, :, 2].astype(np.float32) * scale

    r0 = np.clip(np.floor(rf).astype(np.int32), 0, n - 1)
    g0 = np.clip(np.floor(gf).astype(np.int32), 0, n - 1)
    b0 = np.clip(np.floor(bf).astype(np.int32), 0, n - 1)
    r1 = np.minimum(r0 + 1, n - 1)
    g1 = np.minimum(g0 + 1, n - 1)
    b1 = np.minimum(b0 + 1, n - 1)

    dr = (rf - r0)[:, :, None]
    dg = (gf - g0)[:, :, None]
    db = (bf - b0)[:, :, None]

    # Standard .cube order: R fastest -> axes (B, G, R)
    c000 = lut[b0, g0, r0]
    c001 = lut[b0, g0, r1]
    c010 = lut[b0, g1, r0]
    c011 = lut[b0, g1, r1]
    c100 = lut[b1, g0, r0]
    c101 = lut[b1, g0, r1]
    c110 = lut[b1, g1, r0]
    c111 = lut[b1, g1, r1]

    c00 = c000 * (1.0 - dr) + c001 * dr
    c01 = c010 * (1.0 - dr) + c011 * dr
    c10 = c100 * (1.0 - dr) + c101 * dr
    c11 = c110 * (1.0 - dr) + c111 * dr

    c0 = c00 * (1.0 - dg) + c01 * dg
    c1 = c10 * (1.0 - dg) + c11 * dg
    out = c0 * (1.0 - db) + c1 * db
    return np.clip(out * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def _apply_effect_grade(rgb_img: np.ndarray, effect_name: str) -> np.ndarray:
    """
    Applies high-taste modern photographic grades, Look Profiles, .cube 3D LUTs, or legacy WLPG effects.
    """
    if effect_name in ("none", "as_shot", "", None):
        return rgb_img

    if str(effect_name).startswith("cube:"):
        return _load_and_apply_cube_lut(rgb_img, str(effect_name)[5:])

    if effect_name == "natural":
        f = rgb_img.astype(np.float32) / 255.0
        diff = f - 0.5
        f = np.clip(0.5 + diff * 1.10 - 0.15 * (diff ** 3), 0.0, 1.0)
        hsv = cv2.cvtColor((f * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.06, 0.0, 255.0)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if effect_name == "portrait":
        f = rgb_img.astype(np.float32) / 255.0
        diff = f - 0.5
        f = np.clip(0.5 + diff * 0.94 + 0.02, 0.0, 1.0)
        f[:, :, 0] = np.clip(f[:, :, 0] * 1.04 + 0.01, 0.0, 1.0)
        f[:, :, 2] = np.clip(f[:, :, 2] * 0.97, 0.0, 1.0)
        return (f * 255.0).astype(np.uint8)

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

    # 2. Warm Portrait / Warm Print (Kodak Portra style: warm golden skin tones, soft pastel highlights)
    elif effect_name in ("warm_print", "warm_portrait", "warm"):
        f = rgb_img.astype(np.float32) / 255.0
        # Warm golden highlight roll-off
        f[:, :, 0] = np.clip(f[:, :, 0] * 1.08 + 0.02, 0.0, 1.0)
        f[:, :, 1] = np.clip(f[:, :, 1] * 1.02 + 0.01, 0.0, 1.0)
        f[:, :, 2] = np.clip(f[:, :, 2] * 0.92, 0.0, 1.0)
        # Soft contrast
        f = np.clip(f * 0.95 + 0.05, 0.0, 1.0)
        return (f * 255.0).astype(np.uint8)

    # 3. Vivid / Vivid Landscape (Velvia style: punchy dynamic range, deep emerald greens, vibrant skies)
    elif effect_name in ("vivid", "vivid_landscape"):
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

    # 5. Monochrome / Fine Art B&W (Acros / Ilford style: deep rich blacks, luminous midtones, crisp microcontrast)
    elif effect_name in ("monochrome", "fine_art_bw", "bw", "bw_contrast", "high_contrast_bw"):
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

    slot = _get_stage_slot((media_id, h_orig, w_orig)) if media_id is not None else None
    dd_key = (
        round(recipe.denoise, 2),
        round(recipe.denoise_detail, 2),
        round(recipe.denoise_chroma, 2),
        round(recipe.dehaze, 2),
    )
    selective_key = tuple(
        (p.get("x"), p.get("y"), p.get("radius"), p.get("brightness"), p.get("contrast"), p.get("saturation"))
        for p in recipe.selective_points
    )
    curve_key = (
        tuple((round(x, 3), round(y, 3)) for x, y in recipe.curve_rgb),
        tuple((round(x, 3), round(y, 3)) for x, y in recipe.curve_r),
        tuple((round(x, 3), round(y, 3)) for x, y in recipe.curve_g),
        tuple((round(x, 3), round(y, 3)) for x, y in recipe.curve_b),
    )
    hsl_key = (
        tuple(sorted((k, round(v, 2)) for k, v in recipe.hsl_hue.items() if abs(v) >= 0.1)),
        tuple(sorted((k, round(v, 2)) for k, v in recipe.hsl_sat.items() if abs(v) >= 0.1)),
        tuple(sorted((k, round(v, 2)) for k, v in recipe.hsl_lum.items() if abs(v) >= 0.1)),
    )
    tl_key = (
        dd_key,
        round(recipe.black_point, 2),
        round(recipe.white_point, 2),
        round(recipe.brightness, 2),
        round(recipe.smart_enhance, 2),
        round(recipe.contrast, 2),
        round(recipe.shadows, 2),
        round(recipe.highlights, 2),
        round(recipe.whites, 2),
        round(recipe.blacks, 2),
        round(recipe.clarity, 2),
        round(recipe.structure, 2),
        round(recipe.temperature, 2),
        round(recipe.tint, 2),
        round(recipe.saturation, 2),
        round(recipe.sharpen, 2),
        round(recipe.sharpen_radius, 2),
        round(recipe.sharpen_masking, 2),
        selective_key,
        curve_key,
        hsl_key,
    )

    tl_hit = False
    if slot is not None:
        with _STAGE_CACHE_LOCK:
            if slot.get("tl_key") == tl_key and slot.get("tl_img") is not None:
                img = slot["tl_img"]
                tl_hit = True

    if not tl_hit:
        dd_hit = False
        if slot is not None:
            with _STAGE_CACHE_LOCK:
                if slot.get("dd_key") == dd_key and slot.get("dd_img") is not None:
                    img = slot["dd_img"]
                    dd_hit = True

        if not dd_hit:
            # 1. Detail: Decoupled Chroma-Denoise + Edge-Aware Luma Filtering
            if recipe.denoise >= 1.0 or recipe.denoise_chroma >= 1.0:
                dn = recipe.denoise
                dn_c = recipe.denoise_chroma if recipe.denoise_chroma >= 1.0 else dn
                # Convert to LAB for perceptual luminance/chrominance separation
                d_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                L_ch = d_lab[:, :, 0]
                A_ch = d_lab[:, :, 1]
                B_ch = d_lab[:, :, 2]

                # 1a. Aggressive high-speed chroma denoise: Clean color blotches where the eye
                # is sensitive without affecting edge sharpness or hair details
                if dn_c >= 1.0:
                    d_chroma = max(3, int(round(7 * res_scale)) | 1)
                    sigma_c = float(dn_c * 0.9)
                    sigma_space_c = float(6.0 * res_scale)
                    dA = cv2.bilateralFilter(A_ch, d=d_chroma, sigmaColor=sigma_c, sigmaSpace=sigma_space_c)
                    dB = cv2.bilateralFilter(B_ch, d=d_chroma, sigmaColor=sigma_c, sigmaSpace=sigma_space_c)
                else:
                    dA = A_ch
                    dB = B_ch

                # 1b. Edge-guided luma filtering: Use 2D gradient magnitude on L to preserve fine textures,
                # skin pores, and eyelashes while smoothing flat grain in skies/shadows
                if dn >= 1.0:
                    sigma_luma_space = float(max(1.2, 3.0 * res_scale))
                    L_smooth = cv2.bilateralFilter(
                        L_ch,
                        d=max(3, int(round(5 * res_scale)) | 1),
                        sigmaColor=float(dn * 0.35),
                        sigmaSpace=sigma_luma_space,
                    )
                    gx = cv2.Sobel(L_ch, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(L_ch, cv2.CV_32F, 0, 1, ksize=3)
                    edge_mag = cv2.magnitude(gx, gy)
                    edge_weight = np.clip(edge_mag / 45.0, 0.0, 1.0)
                    edge_protect = 0.35 + (float(np.clip(recipe.denoise_detail, 0.0, 100.0)) / 100.0) * 0.60
                    blend_factor = (dn / 100.0) * (1.0 - edge_protect * edge_weight)
                    dL = np.clip(
                        L_ch.astype(np.float32) * (1.0 - blend_factor) + L_smooth.astype(np.float32) * blend_factor,
                        0.0,
                        255.0,
                    ).astype(np.uint8)
                else:
                    dL = L_ch

                d_lab[:, :, 0] = dL
                d_lab[:, :, 1] = dA
                d_lab[:, :, 2] = dB
                img = cv2.cvtColor(d_lab, cv2.COLOR_LAB2RGB)

            # 2. Dehaze (Fast 1/4-res dark-channel & bilateral transmission map, bilinearly upsampled)
            if abs(recipe.dehaze) >= 0.1:
                dh = recipe.dehaze / 100.0
                if dh > 0:
                    sh_d = max(16, h_orig // 4)
                    sw_d = max(16, w_orig // 4)
                    small_rgb = cv2.resize(img, (sw_d, sh_d), interpolation=cv2.INTER_AREA)
                    dark_s = np.min(small_rgb, axis=2).astype(np.float32)
                    k_s = max(3, int(round(4 * res_scale)) | 1)
                    dark_smooth_s = cv2.blur(dark_s, (k_s, k_s))
                    A_atm = max(float(np.percentile(small_rgb, 99)), 100.0)
                    tx_s = np.clip(1.0 - 0.75 * dh * (dark_smooth_s / A_atm), 0.25, 1.0).astype(np.float32)
                    d_ref_s = max(3, int(round(3 * res_scale)) | 1)
                    tx_ref_s = cv2.bilateralFilter(
                        tx_s,
                        d=d_ref_s,
                        sigmaColor=0.15,
                        sigmaSpace=max(1.5, 2.5 * res_scale),
                    )
                    tx_refined = cv2.resize(tx_ref_s, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                    f_rad = img.astype(np.float32)
                    f_rad -= A_atm
                    f_rad /= tx_refined[:, :, None]
                    f_rad += A_atm
                    np.clip(f_rad, 0, 255, out=f_rad)
                    img = f_rad.astype(np.uint8)

            if slot is not None:
                with _STAGE_CACHE_LOCK:
                    slot["dd_key"] = dd_key
                    slot["dd_img"] = img

        # 3. Levels & Filmic Exposure (Black/White Point Clipping & Photographic Shoulder/Toe)
        # Accelerated via 256-element uint8 LUT (1.5ms vs 55ms float math).
        # When no other LAB parameters are active, filmic S-curve contrast is baked directly into the 1D LUT
        # to bypass the 100ms RGB<->LAB conversion entirely.
        has_levels = recipe.black_point >= 0.5 or recipe.white_point <= 254.5
        has_exposure = abs(recipe.brightness) >= 0.1
        has_whites_blacks = abs(recipe.whites) >= 0.1 or abs(recipe.blacks) >= 0.1
        has_contrast = abs(recipe.contrast) >= 0.1
        has_lab_except_contrast = (
            abs(recipe.shadows) >= 0.1
            or abs(recipe.highlights) >= 0.1
            or abs(recipe.clarity) >= 0.1
            or abs(recipe.structure) >= 0.1
            or recipe.smart_enhance >= 0.5
            or abs(recipe.temperature) >= 0.1
            or abs(recipe.tint) >= 0.1
            or abs(recipe.saturation) >= 0.1
            or recipe.sharpen >= 1.0
            or len(recipe.selective_points) > 0
        )
        bake_contrast_in_lut = has_contrast and not has_lab_except_contrast

        if has_levels or has_exposure or has_whites_blacks or bake_contrast_in_lut:
            lut = np.arange(256, dtype=np.float32) / 255.0
            if has_levels:
                bp = min(recipe.black_point, recipe.white_point - 2.0) / 255.0
                wp = max(recipe.white_point, recipe.black_point + 2.0) / 255.0
                lut = np.clip((lut - bp) / (wp - bp), 0.0, 1.0)

            if has_whites_blacks:
                w_adj = (recipe.whites / 100.0) * 0.22
                k_adj = (recipe.blacks / 100.0) * 0.22
                # Smoothly stretch/compress upper highlights (whites) and deep shadow toe (blacks)
                lut = np.clip(lut + w_adj * (lut ** 2.2) + k_adj * ((1.0 - lut) ** 2.2) * np.where(k_adj > 0, 1.0 - lut, lut + 0.35), 0.0, 1.0)

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

        # 3b. Parametric / Point Tone Curves (RGB + R/G/B Channels via PCHIP LUT)
        img = _apply_tone_curves(img, recipe)

        # 4. Unified CIE LAB Pass (executed only if LAB adjustments or non-baked contrast active)
        has_lab = has_lab_except_contrast or (has_contrast and not bake_contrast_in_lut)
        if has_lab:
            has_ab = (
                abs(recipe.temperature) >= 0.1
                or abs(recipe.tint) >= 0.1
                or abs(recipe.saturation) >= 0.1
                or recipe.smart_enhance >= 0.5
                or abs(recipe.structure) >= 0.1
                or bool(recipe.selective_points)
            )
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            L = lab[:, :, 0].astype(np.float32)
            if has_ab:
                A = lab[:, :, 1].astype(np.float32)
                B = lab[:, :, 2].astype(np.float32)
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
                B += t_shift
                np.clip(B, 0.0, 255.0, out=B)
                A += ti_shift
                np.clip(A, 0.0, 255.0, out=A)

            # Filmic S-curve Contrast on Luminance (if not already baked into 1D LUT)
            if has_contrast and not bake_contrast_in_lut:
                c = recipe.contrast / 100.0
                diff = (L - 128.0) / 128.0
                L = np.clip(128.0 + 128.0 * (diff * (1.0 + c * 0.8) - (c * 0.25) * (diff ** 3)), 0.0, 255.0)

            # Smart Enhance (Adaptive Accent-AI single-slider balance)
            if recipe.smart_enhance >= 0.5:
                se = float(np.clip(recipe.smart_enhance, 0.0, 100.0)) / 100.0
                mean_l = float(np.mean(L)) / 255.0
                l_norm = L / 255.0
                # Adaptive exposure lift for underexposed images + shadow recovery & highlight roll-off
                exp_lift = se * max(0.0, 0.50 - mean_l) * 48.0 * (1.0 - l_norm * 0.5)
                sh_gain = se * float(np.clip(0.68 - mean_l, 0.16, 0.55))
                hl_pull = se * 0.18
                s_w = np.clip(1.0 - l_norm / 0.62, 0.0, 1.0) ** 1.5
                h_w = np.clip((l_norm - 0.55) / 0.45, 0.0, 1.0) ** 1.4
                L = L + exp_lift + sh_gain * s_w * (148.0 - L) * 0.72 - hl_pull * h_w * (L - 110.0) * 0.55
                # Gentle midtone S-curve contrast without crushing lifted shadows
                diff_se = (L - 128.0) / 128.0
                L = np.clip(128.0 + 128.0 * (diff_se * (1.0 + se * 0.12) - (se * 0.05) * (diff_se ** 3)), 0.0, 255.0)

            # Content-Aware Structure (Detail boost/soften with skin & smooth-sky protection)
            if abs(recipe.structure) >= 0.1:
                st = float(np.clip(recipe.structure, -100.0, 100.0)) / 100.0
                sigma_st = max(1.0, 2.2 * max(0.5, res_scale))
                L_st_blur = cv2.GaussianBlur(L, (0, 0), sigmaX=sigma_st, sigmaY=sigma_st)
                L_st_detail = L - L_st_blur
                if st > 0 and has_ab:
                    d_a14 = (A - 146.0) * (1.0 / 14.0)
                    d_b18 = (B - 152.0) * (1.0 / 18.0)
                    skin_m = np.clip(1.0 - np.sqrt(d_a14 * d_a14 + d_b18 * d_b18), 0.0, 1.0)
                    # Reduce positive structure by 75% on skin tones & 65% on smooth bright highlights/sky
                    hl_smooth = np.clip((L - 195.0) / 50.0, 0.0, 1.0) * np.clip(1.0 - np.abs(L_st_detail) / 12.0, 0.0, 1.0)
                    st_weight = (1.0 - 0.75 * skin_m) * (1.0 - 0.65 * hl_smooth)
                else:
                    st_weight = 1.0
                L = np.clip(L + st * 0.90 * st_weight * L_st_detail, 0.0, 255.0)

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
            effective_sat = recipe.saturation + recipe.smart_enhance * 0.36
            if abs(effective_sat) >= 0.1:
                sat_gain = effective_sat / 100.0
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

            # Halo-free Luminance Unsharp Mask with configurable radius, halo clamping & optional edge masking
            if recipe.sharpen >= 1.0:
                r_mult = 0.5 + (float(np.clip(recipe.sharpen_radius, 0.0, 100.0)) / 100.0) * 2.0
                sigma = max(0.45, 1.2 * r_mult * max(0.5, res_scale))
                L_blur = cv2.GaussianBlur(L, (0, 0), sigmaX=sigma, sigmaY=sigma)
                d = L - L_blur
                d_soft = np.sign(d) * np.maximum(0.0, np.abs(d) - 3.0)
                sharp_delta = (recipe.sharpen / 100.0) * 1.5 * d_soft
                sharp_delta = np.clip(sharp_delta, -14.0, 18.0)
                if recipe.sharpen_masking >= 1.0:
                    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
                    grad_mag = np.sqrt(gx * gx + gy * gy)
                    thresh = (recipe.sharpen_masking / 100.0) * 55.0
                    edge_w = np.clip((grad_mag - thresh * 0.5) / max(8.0, thresh * 0.6), 0.0, 1.0)
                    sharp_delta = sharp_delta * edge_w
                L = np.clip(L + sharp_delta, 0.0, 255.0)

            lab[:, :, 0] = np.clip(L + 0.5, 0.0, 255.0).astype(np.uint8)
            if has_ab:
                # AgX-style highlight chroma roll-off: gently desaturate extreme highlights (L > 220)
                # toward paper-white so boosted highlights don't clip to harsh yellow/magenta.
                w_desat = np.clip((L - 220.0) / 35.0, 0.0, 1.0) ** 2
                rolloff = 1.0 - 0.65 * w_desat
                A = 128.0 + (A - 128.0) * rolloff
                B = 128.0 + (B - 128.0) * rolloff
                lab[:, :, 1] = np.clip(A + 0.5, 0.0, 255.0).astype(np.uint8)
                lab[:, :, 2] = np.clip(B + 0.5, 0.0, 255.0).astype(np.uint8)
            else:
                hl_mask = L > 220.0
                if np.any(hl_mask):
                    w_desat = np.clip((L[hl_mask] - 220.0) / 35.0, 0.0, 1.0) ** 2
                    rolloff = 1.0 - 0.65 * w_desat
                    a_hl = 128.0 + (lab[:, :, 1][hl_mask].astype(np.float32) - 128.0) * rolloff
                    b_hl = 128.0 + (lab[:, :, 2][hl_mask].astype(np.float32) - 128.0) * rolloff
                    lab[:, :, 1][hl_mask] = np.clip(a_hl + 0.5, 0.0, 255.0).astype(np.uint8)
                    lab[:, :, 2][hl_mask] = np.clip(b_hl + 0.5, 0.0, 255.0).astype(np.uint8)
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 4b. 8-Band HSL Color Mixer
        img = _apply_hsl_color_mixer(img, recipe)

        if slot is not None:
            with _STAGE_CACHE_LOCK:
                slot["tl_key"] = tl_key
                slot["tl_img"] = img

    # 4c. Phase 1 Regional Masking (Background / Sky / Subject)
    if recipe.region_edits:
        active_regions = [
            (rname, recipe.region_edits[rname])
            for rname in ("background", "sky", "subject", "custom")
            if _region_dict_has_edits(recipe.region_edits.get(rname))
        ]
        if active_regions:
            try:
                from src.imaging.creative_effects import (
                    apply_region_mask_strokes,
                    get_depth_and_matte_engine,
                )

                depth_engine = get_depth_and_matte_engine()
                for rname, rdict in active_regions:
                    sub_rec = PhotoAdjustmentRecipe(
                        brightness=float(rdict.get("brightness", 0.0)),
                        contrast=float(rdict.get("contrast", 0.0)),
                        shadows=float(rdict.get("shadows", 0.0)),
                        highlights=float(rdict.get("highlights", 0.0)),
                        whites=float(rdict.get("whites", 0.0)),
                        blacks=float(rdict.get("blacks", 0.0)),
                        clarity=float(rdict.get("clarity", 0.0)),
                        structure=float(rdict.get("structure", 0.0)),
                        temperature=float(rdict.get("temperature", 0.0)),
                        tint=float(rdict.get("tint", 0.0)),
                        saturation=float(rdict.get("saturation", 0.0)),
                        dehaze=float(rdict.get("dehaze", 0.0)),
                        curve_rgb=[tuple(p) for p in rdict["curve_rgb"]]
                        if rdict.get("curve_rgb")
                        else _default_curve_points(),
                        curve_r=[tuple(p) for p in rdict["curve_r"]]
                        if rdict.get("curve_r")
                        else _default_curve_points(),
                        curve_g=[tuple(p) for p in rdict["curve_g"]]
                        if rdict.get("curve_g")
                        else _default_curve_points(),
                        curve_b=[tuple(p) for p in rdict["curve_b"]]
                        if rdict.get("curve_b")
                        else _default_curve_points(),
                    )
                    reg_img = render_adjusted_image(img, sub_rec, media_id=None)
                    raw_mask = depth_engine.get_region_mask(master_rgb, rname, media_id=media_id)
                    strokes = rdict.get("mask_strokes")
                    if strokes:
                        raw_mask = apply_region_mask_strokes(
                            raw_mask, cv2.cvtColor(master_rgb, cv2.COLOR_RGB2BGR), strokes
                        )
                    mask = raw_mask[:, :, None]
                    blended = img.astype(np.float32) * (1.0 - mask) + reg_img.astype(np.float32) * mask
                    img = np.clip(blended, 0.0, 255.0).astype(np.uint8)
            except Exception as ex:
                logger.debug("Regional masked adjustment skipped: %s", ex)

    # 5. Look Profiles / Photographic Effects Preset / .cube LUT with 0..200% Intensity Blend
    if recipe.effect not in ("none", "as_shot", "", None):
        effect_img = _apply_effect_grade(img, recipe.effect)
        alpha = max(0.0, min(2.0, recipe.effect_intensity / 100.0))
        if abs(alpha - 1.0) < 1e-3:
            img = effect_img
        elif alpha < 1.0:
            img = cv2.addWeighted(effect_img, alpha, img, 1.0 - alpha, 0)
        else:
            base_f = img.astype(np.float32)
            eff_f = effect_img.astype(np.float32)
            img = np.clip(base_f + alpha * (eff_f - base_f), 0.0, 255.0).astype(np.uint8)

    # 5b. 3D Depth Bokeh, Relighting & Atmospheric Haze
    has_virtual_lights = any(
        bool(lt.get("enabled", True)) and float(lt.get("intensity", 0.0)) >= 0.1
        for lt in getattr(recipe, "virtual_lights", [])
    )
    has_ambient_dim = abs(getattr(recipe, "relight_ambient_dim", 0.0)) >= 0.1
    has_depth_and_light = (
        abs(recipe.depth_blur) >= 0.1
        or abs(recipe.relight_near) >= 0.1
        or abs(recipe.relight_far) >= 0.1
        or abs(recipe.warmth_near) >= 0.1
        or abs(recipe.warmth_far) >= 0.1
        or has_ambient_dim
        or has_virtual_lights
        or abs(recipe.depth_haze) >= 0.1
        or abs(recipe.boundary_shift) >= 0.1
        or abs(recipe.boundary_feather) >= 0.1
    )
    if has_depth_and_light:
        try:
            from src.imaging.creative_effects import (
                apply_virtual_3d_relighting,
                get_depth_and_matte_engine,
            )
            engine = get_depth_and_matte_engine()
            depth_map, alpha_matte, near_weight, far_weight = engine.get_depth_matte_and_weights(img, media_id=media_id)

            # Apply interactive Boundary Shift (shrink/grow) and Boundary Feather to alpha_matte
            has_boundary_adj = abs(recipe.boundary_shift) >= 0.1 or abs(recipe.boundary_feather) >= 0.1
            if has_boundary_adj:
                adj_matte = alpha_matte.copy()
                if abs(recipe.boundary_shift) >= 0.1:
                    r_px = max(1, int(round(abs(recipe.boundary_shift) * 1.5 * res_scale)))
                    k_b = 2 * r_px + 1
                    kern_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_b, k_b))
                    if recipe.boundary_shift > 0:
                        adj_matte = cv2.dilate(adj_matte, kern_b)
                    else:
                        adj_matte = cv2.erode(adj_matte, kern_b)
                if recipe.boundary_feather >= 0.1:
                    sigma_f = max(0.8, (recipe.boundary_feather * 0.8) * res_scale)
                    k_f = max(3, int(round(sigma_f * 4)) | 1)
                    adj_matte = cv2.GaussianBlur(adj_matte, (k_f, k_f), sigmaX=sigma_f, sigmaY=sigma_f)
                alpha_matte = np.clip(adj_matte, 0.0, 1.0).astype(np.float32)
                near_weight = np.maximum(alpha_matte, 1.0 - depth_map).astype(np.float32)
                far_weight = (depth_map * (1.0 - alpha_matte)).astype(np.float32)

            # 3D Surface-Normal Relighting, Albedo-Preserving De-Lighting & Virtual Lights
            has_relight = (
                abs(recipe.relight_near) >= 0.1
                or abs(recipe.relight_far) >= 0.1
                or abs(recipe.warmth_near) >= 0.1
                or abs(recipe.warmth_far) >= 0.1
                or has_ambient_dim
                or has_virtual_lights
            )
            if has_relight:
                refined_depth, _, normal_map = engine.get_refined_depth_and_normals(img, media_id=media_id)
                img = apply_virtual_3d_relighting(
                    img,
                    refined_depth,
                    alpha_matte,
                    normal_map,
                    near_weight,
                    far_weight,
                    relight_near=recipe.relight_near,
                    relight_far=recipe.relight_far,
                    warmth_near=recipe.warmth_near,
                    warmth_far=recipe.warmth_far,
                    ambient_dim=getattr(recipe, "relight_ambient_dim", 0.0),
                    virtual_lights=getattr(recipe, "virtual_lights", ()),
                )

            # Portrait Bokeh & Focus Defocus Blur with Normalized Convolution (Zero Foreground Color Bleed)
            if abs(recipe.depth_blur) >= 0.1:
                vlights_key = tuple(
                    tuple(sorted((str(k), round(v, 3) if isinstance(v, float) else v) for k, v in lt.items()))
                    for lt in getattr(recipe, "virtual_lights", [])
                )
                relight_key = (
                    round(recipe.relight_near, 2),
                    round(recipe.relight_far, 2),
                    round(recipe.warmth_near, 2),
                    round(recipe.warmth_far, 2),
                    round(getattr(recipe, "relight_ambient_dim", 0.0), 2),
                    vlights_key,
                )
                region_key = tuple(
                    (
                        rk,
                        tuple(
                            sorted(
                                (
                                    fk,
                                    tuple(
                                        (round(float(st.get("x", 0.0)), 3), round(float(st.get("y", 0.0)), 3), round(float(st.get("r", 0.0)), 3), str(st.get("mode", "paint")))
                                        if isinstance(st, dict)
                                        else (round(st[0], 3), round(st[1], 3))
                                        for st in fv
                                    )
                                    if isinstance(fv, list)
                                    else round(float(fv), 2),
                                )
                                for fk, fv in rd.items()
                            )
                        ),
                    )
                    for rk, rd in sorted(recipe.region_edits.items())
                    if _region_dict_has_edits(rd)
                )
                bokeh_key = (
                    tl_key,
                    region_key,
                    recipe.effect,
                    round(recipe.effect_intensity, 2),
                    relight_key,
                    round(recipe.depth_blur, 2),
                    round(recipe.focal_plane, 2),
                    round(recipe.bokeh_glow, 2),
                    round(recipe.boundary_shift, 2),
                    round(recipe.boundary_feather, 2),
                )
                bokeh_cached = None
                if slot is not None:
                    with _STAGE_CACHE_LOCK:
                        if slot.get("bokeh_key") == bokeh_key and slot.get("bokeh_data") is not None:
                            bokeh_cached = slot["bokeh_data"]

                if bokeh_cached is not None:
                    blurred_srgb, defocus = bokeh_cached
                else:
                    focus_z = float(recipe.focal_plane) / 100.0
                    subject_protect = alpha_matte * (1.0 if focus_z < 0.25 else 0.0)
                    defocus = np.clip(np.abs(depth_map - focus_z) * (1.0 - subject_protect), 0.0, 1.0)
                    w_bg = np.clip(1.0 - subject_protect, 1e-3, 1.0).astype(np.float32)

                    blur_r = max(2, int(round((recipe.depth_blur / 100.0) * 28.0 * res_scale)))
                    h_img, w_img = img.shape[:2]

                    # Optical defocus blur has zero frequencies above 640px; blur on downscaled buffer for instant speed
                    scale_blur = max(1.0, max(h_img, w_img) / 640.0)
                    if scale_blur > 1.05:
                        bw = max(16, int(round(w_img / scale_blur)))
                        bh = max(16, int(round(h_img / scale_blur)))
                        small_img = cv2.resize(img, (bw, bh), interpolation=cv2.INTER_AREA)
                        small_w_bg = cv2.resize(w_bg, (bw, bh), interpolation=cv2.INTER_AREA)
                        small_lin = _LUT_SRGB_TO_LIN[small_img]
                        if recipe.bokeh_glow >= 0.1:
                            lum_small = cv2.cvtColor(small_img, cv2.COLOR_RGB2GRAY)
                            glow_factor = np.clip((lum_small.astype(np.float32) - 190.0) / 60.0, 0.0, 1.0)[:, :, None]
                            small_lin = small_lin * (1.0 + glow_factor * (recipe.bokeh_glow / 50.0) * 2.5)
                        small_r = max(2, int(round(blur_r / scale_blur)))
                        sigma_b = max(1.0, small_r * 0.5)
                        ksize = max(3, (2 * small_r + 1) | 1)
                        premult_small = small_lin * small_w_bg[:, :, None]
                        blurred_premult = cv2.GaussianBlur(premult_small, (ksize, ksize), sigmaX=sigma_b, sigmaY=sigma_b)
                        blurred_w = cv2.GaussianBlur(small_w_bg, (ksize, ksize), sigmaX=sigma_b, sigmaY=sigma_b)[:, :, None]
                        blurred_small = blurred_premult / np.maximum(blurred_w, 1e-4)
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
                        premult_lin = img_lin * w_bg[:, :, None]
                        blurred_premult = cv2.GaussianBlur(premult_lin, (ksize, ksize), sigmaX=sigma_b, sigmaY=sigma_b)
                        blurred_w = cv2.GaussianBlur(w_bg, (ksize, ksize), sigmaX=sigma_b, sigmaY=sigma_b)[:, :, None]
                        blurred_lin = blurred_premult / np.maximum(blurred_w, 1e-4)
                        indices = np.clip(blurred_lin * 4095.0, 0, 4095).astype(np.int32)
                        blurred_srgb = _LUT_LIN_TO_SRGB[indices]

                    if slot is not None:
                        with _STAGE_CACHE_LOCK:
                            slot["bokeh_key"] = bokeh_key
                            slot["bokeh_data"] = (blurred_srgb, defocus)

                blend_w = np.clip(defocus * (recipe.depth_blur / 100.0) * 1.5, 0.0, 1.0)[:, :, None]
                f_base = img.astype(np.float32)
                f_blur = blurred_srgb.astype(np.float32)
                f_blur -= f_base
                f_blur *= blend_w
                f_base += f_blur
                np.clip(f_base, 0.0, 255.0, out=f_base)
                img = f_base.astype(np.uint8)

            # Atmospheric Depth Haze / Fog (Luminar-style optical scattering, soft veiling & horizon depth layering)
            if abs(recipe.depth_haze) >= 0.1:
                h_img, w_img = img.shape[:2]
                amount = np.clip(recipe.depth_haze / 100.0, 0.0, 1.0)
                
                # 1. Distance-based optical transmission with smooth exponential extinction
                beta = 1.8 * amount
                haze_distance_trans = 1.0 - np.exp(-beta * (depth_map ** 1.6))
                
                # 2. Natural vertical / altitude falloff (haze settles gently along ground/horizon plane)
                y_coords = np.linspace(0.0, 1.0, h_img, dtype=np.float32)[:, None]
                vertical_falloff = 0.55 + 0.45 * (y_coords ** 0.8)
                
                # Composite haze transmission factor protecting the segmented foreground subject
                haze_factor = np.clip(
                    haze_distance_trans * vertical_falloff * (1.0 - alpha_matte * 0.92),
                    0.0,
                    0.75 * amount,
                )[:, :, None]
                
                # 3. Atmospheric Airlight with forward Mie scattering bloom:
                # Downsample background to compute a wide, soft optical scatter bloom from scene light sources
                scale_haze = max(1.0, max(h_img, w_img) / 480.0)
                sh_haze = max(16, int(round(h_img / scale_haze)))
                sw_haze = max(16, int(round(w_img / scale_haze)))
                small_scene = cv2.resize(img, (sw_haze, sh_haze), interpolation=cv2.INTER_AREA)
                
                # Wide Gaussian scatter kernel
                sigma_scatter = max(3.0, 18.0 / scale_haze)
                scatter_bloom = cv2.GaussianBlur(small_scene, (0, 0), sigmaX=sigma_scatter, sigmaY=sigma_scatter)
                scatter_full = cv2.resize(scatter_bloom, (w_img, h_img), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                
                # 4. Color temperature & tint of the atmospheric mist
                w_h = recipe.depth_haze_warmth / 100.0
                if w_h >= 0:
                    base_mist = np.array([215.0 + w_h * 35.0, 222.0 + w_h * 8.0, 235.0 - w_h * 45.0], dtype=np.float32)
                else:
                    base_mist = np.array([215.0 + w_h * 45.0, 222.0 + w_h * 15.0, 235.0 - w_h * 15.0], dtype=np.float32)
                
                # Synthesize airlight: 60% ambient mist color + 40% forward-scattered scene luminance
                airlight = 0.58 * base_mist[None, None, :] + 0.42 * scatter_full
                
                # 5. Photographic Veiling Blend (Soft-knee screen blend preserving highlight contrast,
                # lifting milky shadow toe without blowing out into opaque flat white)
                img_f = img.astype(np.float32)
                # Distance-based slight chroma desaturation in deep mist for photographic realism
                gray_f = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)[:, :, None]
                img_f *= 0.75
                img_f += gray_f * 0.25
                img_f *= (1.0 - haze_factor * 0.55)
                img_f += airlight * haze_factor
                np.clip(img_f, 0.0, 255.0, out=img_f)
                img = img_f.astype(np.uint8)
        except Exception as ex:
            logger.debug("Depth and light processing failed: %s", ex)

    # 6a. Orton Glow & Golden Hour (Luminar-inspired atmospheric glow)
    if recipe.orton_glow >= 0.5:
        og = float(np.clip(recipe.orton_glow, 0.0, 100.0)) / 100.0
        sigma_og = max(4.0, 18.0 * max(0.5, res_scale))
        blur_og = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_og, sigmaY=sigma_og).astype(np.float32) / 255.0
        base_f = img.astype(np.float32) / 255.0
        screen_og = 1.0 - (1.0 - base_f) * (1.0 - blur_og)
        # Soft highlight-weighted screen bloom + gentle contrast
        og_blend = np.clip(base_f * (1.0 - 0.55 * og) + screen_og * (0.55 * og), 0.0, 1.0)
        img = np.clip(og_blend * 255.0, 0.0, 255.0).astype(np.uint8)

    if recipe.golden_hour >= 0.5:
        gh = float(np.clip(recipe.golden_hour, 0.0, 100.0)) / 100.0
        img_f = img.astype(np.float32)
        lum_gh = (0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]) / 255.0
        w_warm = np.clip(0.25 + 0.75 * lum_gh, 0.0, 1.0)
        img_f[:, :, 0] += gh * 24.0 * w_warm
        img_f[:, :, 1] += gh * 10.0 * w_warm
        img_f[:, :, 2] -= gh * 20.0 * w_warm
        np.clip(img_f, 0.0, 255.0, out=img_f)
        img = img_f.astype(np.uint8)

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
        feather_exp = 0.45 + (max(0.0, min(100.0, float(recipe.vignette_feather))) / 100.0) * 1.10
        falloff = np.clip(1.0 - v_mask, 0.0, 1.0) ** feather_exp
        v_gain = 1.0 + v_amount * (1.0 - falloff)
        img_f = img.astype(np.float32)
        img_f *= v_gain[:, :, None]
        np.clip(img_f, 0, 255, out=img_f)
        img = img_f.astype(np.uint8)

    # 6b. Organic Film Grain
    if recipe.grain_amount >= 0.5:
        img = _apply_film_grain(img, recipe.grain_amount, recipe.grain_size)

    # 7. Straighten & Crop
    if abs(recipe.straighten_deg) >= 0.05:
        if getattr(recipe, "straighten_ai_fill", False):
            img = straighten_with_ai_fill(img, recipe.straighten_deg)
        else:
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


def straighten_with_ai_fill(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate an RGB uint8 image by angle_deg while preserving the full (w, h) frame
    and filling empty rotated corner wedges via LaMa (or built-in texture fallback).
    """
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_matrix = cv2.getRotationMatrix2D(center, -float(angle_deg), 1.0)
    # Use BORDER_REPLICATE instead of black BORDER_CONSTANT so INTER_CUBIC never
    # introduces a 1-2px dark anti-aliased fringe along the rotated image boundary.
    rot_img = cv2.warpAffine(
        img,
        rot_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    ones = np.full((h, w), 255, dtype=np.uint8)
    rot_mask = cv2.warpAffine(
        ones,
        rot_matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    gap_mask = np.where(rot_mask == 0, np.uint8(255), np.uint8(0))
    if np.any(gap_mask > 0):
        from src.imaging.creative_effects import inpaint_object_lama

        seam_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        inpaint_mask = cv2.dilate(gap_mask, seam_k, iterations=1)
        filled = inpaint_object_lama(rot_img, inpaint_mask)
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        keep_bin = cv2.erode((gap_mask == 0).astype(np.uint8) * 255, erode_k, iterations=1)
        alpha_keep = cv2.GaussianBlur(keep_bin.astype(np.float32) / 255.0, (7, 7), 1.8)
        alpha_keep = np.where(gap_mask > 0, 0.0, alpha_keep)[:, :, None]
        blended = (
            rot_img.astype(np.float32) * alpha_keep
            + filled.astype(np.float32) * (1.0 - alpha_keep)
        )
        return np.clip(blended, 0, 255).astype(np.uint8)
    return rot_img


def expand_canvas_with_ai_fill(
    img: np.ndarray, left: int, top: int, right: int, bottom: int
) -> np.ndarray:
    """
    Crop/expand an RGB uint8 image to (left, top, right, bottom) where bounds may
    extend outside [0..w, 0..h], filling any out-of-bounds margins via LaMa (or
    built-in texture synthesis fallback when the neural model is not installed).
    """
    h, w = img.shape[:2]
    out_w = max(1, int(right - left))
    out_h = max(1, int(bottom - top))
    mask = np.full((out_h, out_w), 255, dtype=np.uint8)

    src_x1 = max(0, left)
    src_y1 = max(0, top)
    src_x2 = min(w, right)
    src_y2 = min(h, bottom)

    if src_x2 > src_x1 and src_y2 > src_y1:
        # Pre-fill expanded margins with edge-replicated pixels instead of black (0,0,0)
        # so neural/texture context never sees a hard black cliff at the boundary.
        pad_l = max(0, -left)
        pad_t = max(0, -top)
        pad_r = max(0, right - w)
        pad_b = max(0, bottom - h)
        sub = img[src_y1:src_y2, src_x1:src_x2]
        canvas = cv2.copyMakeBorder(
            sub, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_REPLICATE
        )
        if canvas.shape[0] != out_h or canvas.shape[1] != out_w:
            canvas = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        dst_x1 = src_x1 - left
        dst_y1 = src_y1 - top
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        mask[dst_y1:dst_y2, dst_x1:dst_x2] = 0
    else:
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    if np.any(mask > 0):
        from src.imaging.creative_effects import inpaint_object_lama

        # Dilate mask slightly (5px) into the original border so LaMa synthesizes across
        # any edge JPEG ringing/dark border, then erode+feather-blend with the crisp interior.
        seam_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        inpaint_mask = cv2.dilate(mask, seam_k, iterations=1)
        filled = inpaint_object_lama(canvas, inpaint_mask)

        if src_x2 > src_x1 and src_y2 > src_y1:
            # Smooth feathered transition across the boundary (erode 2px so edge artifacts are replaced)
            erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            keep_bin = cv2.erode((mask == 0).astype(np.uint8) * 255, erode_k, iterations=1)
            alpha_keep = cv2.GaussianBlur(keep_bin.astype(np.float32) / 255.0, (9, 9), 2.0)
            alpha_keep = np.where(mask > 0, 0.0, alpha_keep)[:, :, None]
            blended = (
                canvas.astype(np.float32) * alpha_keep
                + filled.astype(np.float32) * (1.0 - alpha_keep)
            )
            return np.clip(blended, 0, 255).astype(np.uint8)
        return filled
    return canvas


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

    # OpenCV 5 returns (N, 4) for HoughLinesP while OpenCV 4 returned (N, 1, 4)
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
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
