"""
Creative Effects Engine:
1. Google Photos-Style 'Color Pop' (Subject Color Splash) with bilateral & hue-aware matting.
2. Portrait Depth Bokeh (spatially varying optical disc lens blur) & Foreground Fill Light.
3. '3D Cinematic Motion' (Continuous Depth-Map Mesh Warping via cv2.remap with Parallax Presets).
4. Privacy Redaction (Mosaic pixelation and Gaussian blur).
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

_SHARED_FACE_ENGINE: Any = None


def _get_face_engine() -> Any:
    """Lazily initializes and caches the shared FaceEngine."""
    global _SHARED_FACE_ENGINE
    if _SHARED_FACE_ENGINE is None:
        try:
            import os
            os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
            models_dir = os.path.expanduser("~/.insightface/models")
            has_local_insight = os.path.exists(models_dir) and any(os.path.isdir(os.path.join(models_dir, m)) for m in ("buffalo_l", "buffalo_s"))
            backend = "insightface" if has_local_insight else "opencv_haar"
            from src.core.face_engine import FaceEngine
            _SHARED_FACE_ENGINE = FaceEngine(backend=backend, allow_fallback=True)
        except BaseException as e:
            logger.debug("FaceEngine initialization in creative_effects failed: %s", e)
            _SHARED_FACE_ENGINE = False
    return _SHARED_FACE_ENGINE if _SHARED_FACE_ENGINE is not False else None


# =============================================================================
# EXIF-Safe Loading & Edge-Preserving Guided Filter
# =============================================================================

def load_bgr_exif_safe(path: str | Path) -> np.ndarray | None:
    """
    Load an image into BGR uint8 format respecting EXIF orientation tags.
    Eliminates sideways orientation on JPEG photos.
    """
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with Image.open(str(p)) as img:
            img = ImageOps.exif_transpose(img)
            rgb = np.array(img.convert("RGB"), dtype=np.uint8)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        try:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            pass
        if p.suffix.lower() in {".cr2", ".cr3", ".nef", ".nrw", ".arw", ".dng", ".orf", ".rw2", ".pef", ".raf", ".srw"}:
            try:
                import rawpy

                with rawpy.imread(str(p)) as raw:
                    rgb = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=False)
                    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                return None
        return None


def guided_filter_matte(
    guide_bgr: np.ndarray,
    src_mask: np.ndarray,
    radius: int = 16,
    eps: float = 1e-3,
    subsample: int = 2,
    unknown_band: np.ndarray | None = None,
) -> np.ndarray:
    """
    Edge-preserving Joint Guided Filter (He et al.) implemented via pure NumPy + cv2.boxFilter.
    Snaps rough or upscaled alpha masks [0..1] directly to full-resolution luminance/color edges
    eliminating blurry halo bleed. Runs in ~15ms on large images via subsampling.
    When unknown_band is provided, guided alpha refinement is strictly restricted to unknown_band
    pixels, preventing foreground weight propagation across distant high-contrast background lines.
    """
    if guide_bgr is None or src_mask is None:
        return src_mask

    h, w = guide_bgr.shape[:2]
    I_full = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    p_full = src_mask.astype(np.float32)
    if p_full.shape[:2] != (h, w):
        p_full = cv2.resize(p_full, (w, h), interpolation=cv2.INTER_LINEAR)

    subsample = max(1, int(subsample))
    sh, sw = max(16, h // subsample), max(16, w // subsample)
    I = cv2.resize(I_full, (sw, sh), interpolation=cv2.INTER_AREA) if subsample > 1 else I_full
    p = cv2.resize(p_full, (sw, sh), interpolation=cv2.INTER_AREA) if subsample > 1 else p_full
    r = max(2, radius // subsample)
    ksize = (2 * r + 1, 2 * r + 1)

    mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    corr_I = cv2.boxFilter(I * I, cv2.CV_32F, ksize)
    corr_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)

    var_I = np.maximum(corr_I - mean_I * mean_I, 0.0)
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    if subsample > 1:
        mean_a_up = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b_up = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        mean_a_up = mean_a
        mean_b_up = mean_b

    q = mean_a_up * I_full + mean_b_up
    q = np.clip(q, 0.0, 1.0)
    if unknown_band is not None:
        ub = (unknown_band > 0.5) if unknown_band.dtype != bool else unknown_band
        if ub.shape[:2] != (h, w):
            ub = cv2.resize(ub.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        res = p_full.copy()
        res[ub] = q[ub]
        return np.clip(res, 0.0, 1.0)
    return q


def refine_mask_narrow_band(
    guide_bgr: np.ndarray,
    src_mask: np.ndarray,
    radius: int | None = None,
    subsample: int = 2,
) -> np.ndarray:
    """
    Restricts alpha matting strictly to the narrow boundary unknown band:
    - Interior core (eroded) remains strictly 1.0.
    - Exterior background (outside dilated) remains strictly 0.0.
    - Only the narrow boundary transition band receives guided filter subpixel snapping.
    Guarantees zero background halo bleed onto road stripes, railings, or distant objects.
    """
    if guide_bgr is None or src_mask is None:
        return src_mask

    h, w = guide_bgr.shape[:2]
    mask = src_mask.astype(np.float32)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    k_size = radius if radius is not None else max(5, int(min(h, w) * 0.015)) | 1
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    th_mask = (mask > 0.35).astype(np.float32)
    dilated = cv2.dilate(th_mask, kernel)
    eroded = cv2.erode(th_mask, kernel)
    unknown_band = (dilated > 0.5) & (eroded < 0.5)

    refined = guided_filter_matte(guide_bgr, mask, radius=k_size, eps=1e-3, subsample=subsample)

    final_mask = np.zeros((h, w), dtype=np.float32)
    final_mask[eroded > 0.5] = 1.0
    final_mask[unknown_band] = refined[unknown_band]
    return np.clip(final_mask, 0.0, 1.0).astype(np.float32)


def _download_u2netp_model(dest_path: Path, timeout: float = 5.0) -> bool:
    """Downloads lightweight u2netp.onnx (~4.5 MB) from official release with a clean timeout."""
    import urllib.request
    urls = [
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        "https://huggingface.co/schirrmacher/u2netp/resolve/main/u2netp.onnx",
    ]
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".tmp")
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    data = resp.read()
                    if len(data) > 1_000_000:
                        tmp_path.write_bytes(data)
                        tmp_path.replace(dest_path)
                        logger.info("Successfully downloaded %s (%d bytes)", dest_path.name, len(data))
                        return True
        except Exception as e:
            logger.debug("Download from %s failed: %s", url, e)
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
    return False


# =============================================================================
# 1. Subject Segmentation, Depth Estimation & Color Pop
# =============================================================================

def _detect_faces_for_mask(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detects faces and returns pixel coordinates [(x, y, w, h)]. Uses fast multi-scale Haar cascade."""
    h, w = image_bgr.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []

    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        scale = max(1.0, max(h, w) / 800.0)
        sw, sh = max(10, int(round(w / scale))), max(10, int(round(h / scale)))
        small_gray = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA) if scale > 1.05 else gray
        raw_faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        for rx, ry, rw, rh in raw_faces:
            bx = max(0, min(w - 1, int(round(rx * scale))))
            by = max(0, min(h - 1, int(round(ry * scale))))
            bw = max(16, min(w - bx, int(round(rw * scale))))
            bh = max(16, min(h - by, int(round(rh * scale))))
            boxes.append((bx, by, bw, bh))
    except Exception:
        pass

    return sorted(boxes, key=lambda rect: rect[0])


def estimate_subject_mask(
    image_bgr: np.ndarray,
    face_boxes: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """
    Estimates a clean, edge-aware float32 subject alpha mask in [0.0, 1.0]:
    - Automatically detects faces via FaceEngine if face_boxes is not provided.
    - If faces are found: builds anatomical trimap with extended arm envelope (5x face width)
      and mid-frame probable background so extended limbs are never clipped.
    - If no faces are found: saliency-informed seeding with background road stripe rejection.
    - Background Color Rejection Gate: removes uniform road/sky patches inside probable-foreground.
    - Edge refinement is strictly restricted to the narrow boundary band, preventing halo leakage
      across high-contrast background stripes or fences.
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((10, 10), dtype=np.float32)

    h, w = image_bgr.shape[:2]

    # Downsample for fast GrabCut (~400px long edge)
    scale = max(1.0, max(h, w) / 450.0)
    sw = max(10, int(w / scale))
    sh = max(10, int(h / scale))
    small_bgr = cv2.resize(image_bgr, (sw, sh), interpolation=cv2.INTER_AREA)

    if face_boxes is None:
        face_boxes = _detect_faces_for_mask(image_bgr)

    # 1. Sample background color distribution from outer 4% frame border
    lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bw_border = max(1, int(sw * 0.04))
    bh_border = max(1, int(sh * 0.04))
    border_pixels = np.concatenate([
        lab[0:bh_border, :, :].reshape(-1, 3),
        lab[-bh_border:, :, :].reshape(-1, 3),
        lab[:, 0:bw_border, :].reshape(-1, 3),
        lab[:, -bw_border:, :].reshape(-1, 3),
    ], axis=0)
    bg_mean = np.mean(border_pixels, axis=0)
    dl_bg = (lab[:, :, 0] - bg_mean[0]) * 0.6
    da_bg = lab[:, :, 1] - bg_mean[1]
    db_bg = lab[:, :, 2] - bg_mean[2]
    dist_to_bg = np.sqrt(dl_bg * dl_bg + da_bg * da_bg + db_bg * db_bg)

    # Gradient magnitude for silhouette detection
    gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_norm = cv2.normalize(cv2.magnitude(sobelx, sobely), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Initialize trimap as Probable Background everywhere
    gc_mask = np.full((sh, sw), cv2.GC_PR_BGD, dtype=np.uint8)
    # Definite background strictly on outer 4% frame borders
    gc_mask[:, :bw_border] = cv2.GC_BGD
    gc_mask[:, -bw_border:] = cv2.GC_BGD

    if face_boxes:
        for fx, fy, fw, fh in face_boxes:
            s_fx = fx / scale
            s_fy = fy / scale
            s_fw = fw / scale
            s_fh = fh / scale

            cx = int(s_fx + s_fw * 0.5)
            cy = int(s_fy + s_fh * 0.48)
            rx = max(4, int(s_fw * 0.65))
            ry = max(4, int(s_fh * 0.72))

            # Sky above head marked as definite background
            sky_y = max(0, int(s_fy - s_fh * 0.4))
            if sky_y > 0:
                gc_mask[0:sky_y, :] = cv2.GC_BGD

            # Head ellipses
            cv2.ellipse(gc_mask, (cx, cy), (int(1.15 * rx), int(1.15 * ry)), 0, 0, 360, cv2.GC_PR_FGD, -1)
            cv2.ellipse(gc_mask, (cx, cy), (max(2, int(0.75 * rx)), max(2, int(0.75 * ry))), 0, 0, 360, cv2.GC_FGD, -1)

            # Neck and upper torso
            neck_y = min(sh - 1, max(0, int(s_fy + s_fh * 0.90)))
            shld_y = min(sh - 1, max(neck_y + 1, int(s_fy + s_fh * 1.45)))
            torso_bottom = min(sh - 1, int(s_fy + s_fh * 4.0))

            torso_pts = np.array([
                [int(cx - s_fw * 0.45), neck_y],
                [int(cx + s_fw * 0.45), neck_y],
                [int(cx + s_fw * 1.50), shld_y],
                [int(cx + s_fw * 1.50), torso_bottom],
                [int(cx - s_fw * 1.50), torso_bottom],
                [int(cx - s_fw * 1.50), shld_y],
            ], dtype=np.int32)
            torso_pts[:, 0] = np.clip(torso_pts[:, 0], 0, sw - 1)
            torso_pts[:, 1] = np.clip(torso_pts[:, 1], 0, sh - 1)
            cv2.fillPoly(gc_mask, [torso_pts], cv2.GC_PR_FGD)

            # Central spine column marked as definite foreground
            spine_x1 = max(0, int(cx - s_fw * 0.7))
            spine_x2 = min(sw, int(cx + s_fw * 0.7))
            spine_y1 = min(sh - 1, max(0, int(s_fy + s_fh * 1.0)))
            spine_y2 = min(sh, int(s_fy + s_fh * 3.5))
            if spine_x2 > spine_x1 and spine_y2 > spine_y1:
                gc_mask[spine_y1:spine_y2, spine_x1:spine_x2] = cv2.GC_FGD

            cv2.ellipse(gc_mask, (cx, cy), (max(2, int(0.75 * rx)), max(2, int(0.75 * ry))), 0, 0, 360, cv2.GC_FGD, -1)

            # Extended arm envelope (5.0x face width from y = fy + 1.15*fh to fy + 3.5*fh)
            # Beside torso remains GC_PR_BGD (Probable Background) so arms are never cut off
            arm_y1 = min(sh - 1, max(0, int(s_fy + s_fh * 1.15)))
            arm_y2 = min(sh, max(arm_y1 + 1, int(s_fy + s_fh * 3.5)))
            arm_x1 = max(bw_border, int(cx - s_fw * 2.5))
            arm_x2 = min(sw - bw_border, int(cx + s_fw * 2.5))
            if arm_x2 > arm_x1 and arm_y2 > arm_y1:
                arm_reg = gc_mask[arm_y1:arm_y2, arm_x1:arm_x2]
                gc_mask[arm_y1:arm_y2, arm_x1:arm_x2] = np.where(
                    arm_reg == cv2.GC_FGD, cv2.GC_FGD, cv2.GC_PR_FGD
                )
    else:
        # Saliency-informed seed: Center-weighted color contrast without geometric oval bias
        saliency = dist_to_bg * (1.0 + grad_norm * 1.5)
        th_fg = max(18.0, float(np.percentile(saliency, 75)))
        th_def_fg = max(28.0, float(np.percentile(saliency, 90)))
        cand_fg = (saliency > th_fg).astype(np.uint8)

        # Reject wide spanning background lines (road stripes, railings, horizons)
        num_lbls, lbls, stats, _ = cv2.connectedComponentsWithStats(cand_fg)
        for lbl in range(1, num_lbls):
            lx, ly, lw, lh, larea = stats[lbl]
            touches_both = (lx <= bw_border and (lx + lw) >= sw - bw_border)
            aspect = max(lw, lh) / max(1, min(lw, lh))
            if touches_both or (lw > sw * 0.55 and aspect > 3.5):
                cand_fg[lbls == lbl] = 0

        gc_mask[cand_fg > 0] = cv2.GC_PR_FGD
        def_fg = cand_fg & (saliency > th_def_fg)
        gc_mask[def_fg > 0] = cv2.GC_FGD

    # Run GrabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            small_bgr,
            gc_mask,
            None,
            bgd_model,
            fgd_model,
            iterCount=3,
            mode=cv2.GC_INIT_WITH_MASK,
        )
    except Exception:
        pass

    # Background Color Rejection Gate:
    # Samples core foreground (GC_FGD) vs border background (GC_BGD).
    # Discards any GC_PR_FGD pixel whose LAB color closely matches surrounding road/sky.
    core_fg = (gc_mask == cv2.GC_FGD)
    if np.count_nonzero(core_fg) > 5:
        fg_mean = np.mean(lab[core_fg], axis=0)
        dl_f = (lab[:, :, 0] - fg_mean[0]) * 0.6
        da_f = lab[:, :, 1] - fg_mean[1]
        db_f = lab[:, :, 2] - fg_mean[2]
        dist_to_fg = np.sqrt(dl_f * dl_f + da_f * da_f + db_f * db_f)
    else:
        dist_to_fg = np.full_like(dist_to_bg, 999.0)

    is_fg = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
    is_pr_fgd = (gc_mask == cv2.GC_PR_FGD)
    reject_bg = is_pr_fgd & (dist_to_bg < 18.0) & (dist_to_bg < dist_to_fg * 0.75) & (grad_norm < 0.20)
    binary_mask = (is_fg & (~reject_bg)).astype(np.float32)

    # Upscale to full camera resolution
    full_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    # Refine strictly within narrow unknown boundary band (zero leakage along background lines)
    return refine_mask_narrow_band(image_bgr, full_mask)


def estimate_depth_map(
    image_bgr: np.ndarray,
    subject_mask: np.ndarray | None = None,
    face_boxes: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """
    Produces a continuous, smooth float32 depth map D(x, y) in [0.0, 1.0]:
    - 1.0 = nearest foreground, 0.0 = farthest background.
    - Combines subject segmentation, ground-plane perspective gradient, and high-frequency sharpness.
    - Refined with Joint Guided Filter so depth transitions align with real visual boundaries.
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((10, 10), dtype=np.float32)

    h, w = image_bgr.shape[:2]
    if subject_mask is None:
        subject_mask = estimate_subject_mask(image_bgr, face_boxes=face_boxes)

    # 1. Ground plane depth gradient (bottom of frame is near, top is far)
    y_coords = np.arange(h, dtype=np.float32).reshape(h, 1)
    ground_gradient = np.clip(y_coords / float(h), 0.0, 1.0)
    ground_gradient = np.tile(ground_gradient, (1, w))

    # 2. Defocus sharpness map via Laplacian energy
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_mag = np.abs(lap)
    sharpness = cv2.boxFilter(lap_mag, cv2.CV_32F, (31, 31))
    sharp_norm = cv2.normalize(sharpness, None, 0.0, 1.0, cv2.NORM_MINMAX)

    # 3. Composite depth estimation
    # Subject has highest foreground weight (0.65), ground gradient (0.25), sharpness (0.10)
    raw_depth = 0.65 * subject_mask + 0.25 * ground_gradient + 0.10 * sharp_norm
    raw_depth = np.clip(raw_depth, 0.0, 1.0)

    # 4. Refine with Guided Filter for smooth, edge-aware depth transitions
    depth_smooth = guided_filter_matte(image_bgr, raw_depth, radius=24, eps=5e-3, subsample=2)
    return np.clip(depth_smooth, 0.0, 1.0).astype(np.float32)


def compute_multi_color_similarity_mask(
    image_bgr: np.ndarray,
    target_bgrs: list[tuple[int, int, int] | np.ndarray] | tuple[tuple[int, int, int], ...],
    tolerance: float = 0.25,
    refine_edges: bool = True,
    lab_img: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generates a unified smooth alpha mask [0.0, 1.0] isolating pixels similar to ANY colour in target_bgrs.
    Computes the image LAB conversion once and runs guided_filter_matte at most once.
    When refine_edges=False (e.g. during active slider scrubbing), skips guided_filter_matte for instant speed.
    """
    if image_bgr is None or not target_bgrs:
        return np.zeros((10, 10) if image_bgr is None else image_bgr.shape[:2], dtype=np.float32)

    if lab_img is None:
        lab_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    targets_arr = np.array([target_bgrs], dtype=np.uint8)
    lab_targets = cv2.cvtColor(targets_arr, cv2.COLOR_BGR2LAB).astype(np.float32)[0]

    L = lab_img[:, :, 0]
    A = lab_img[:, :, 1]
    B = lab_img[:, :, 2]

    min_dist_sq: np.ndarray | None = None
    for lab_t in lab_targets:
        dl = (L - lab_t[0]) * 0.5
        da = A - lab_t[1]
        db = B - lab_t[2]
        d2 = dl * dl + da * da + db * db
        if min_dist_sq is None:
            min_dist_sq = d2
        else:
            np.minimum(min_dist_sq, d2, out=min_dist_sq)

    assert min_dist_sq is not None
    dist = np.sqrt(min_dist_sq)
    max_dist = max(10.0, float(tolerance) * 110.0)
    raw_mask = np.clip(1.0 - (dist / max_dist), 0.0, 1.0)
    smooth_mask = np.power(raw_mask, 1.5)

    if not refine_edges:
        return smooth_mask.astype(np.float32)
    return guided_filter_matte(image_bgr, smooth_mask, radius=12, eps=1e-3, subsample=2)


def compute_color_similarity_mask(
    image_bgr: np.ndarray,
    target_bgr: tuple[int, int, int] | np.ndarray,
    tolerance: float = 0.25,
    refine_edges: bool = True,
) -> np.ndarray:
    """
    Generates a smooth alpha mask [0.0, 1.0] isolating pixels similar in color to target_bgr:
    - Uses CIE LAB perceptual distance.
    - tolerance: in [0.05, 0.80].
    - Edge-snapped using guided_filter_matte when refine_edges=True.
    """
    if image_bgr is None:
        return np.zeros((10, 10), dtype=np.float32)
    return compute_multi_color_similarity_mask(
        image_bgr,
        [target_bgr],
        tolerance=tolerance,
        refine_edges=refine_edges,
    )


def apply_color_pop(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bg_contrast: float = 1.15,
    bg_brightness: float = 0.0,
) -> np.ndarray:
    """
    Produces the signature Color Pop effect:
    - Subject (where mask=1.0) remains in full vibrant color.
    - Background (where mask=0.0) is converted to crisp high-contrast B&W.
    """
    if image_bgr is None or mask is None:
        return image_bgr

    # Background B&W with contrast adjustment
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray -= 128.0
    gray *= bg_contrast
    gray += 128.0 + bg_brightness
    np.clip(gray, 0.0, 255.0, out=gray)
    bw_bgr = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)

    fg_bgr = image_bgr.astype(np.float32)
    m3 = mask[:, :, None]
    fg_bgr *= m3
    bw_bgr *= (1.0 - m3)
    fg_bgr += bw_bgr
    np.clip(fg_bgr, 0.0, 255.0, out=fg_bgr)
    return fg_bgr.astype(np.uint8)


def refine_mask_brush(
    mask: np.ndarray,
    image_bgr: np.ndarray,
    brush_center: tuple[int, int],
    radius: int = 24,
    mode: str = "keep_color",
    smart: bool = True,
    tolerance: float | None = None,
) -> np.ndarray:
    """
    Edge-aware bilateral brush for interactive mask touch-up:
    - Guaranteed responsive painting with smooth radial falloff.
    - When smart=True, bilateral edge affinity snaps to object boundaries (Smart Select style).
    """
    if mask is None or image_bgr is None:
        return mask

    h, w = mask.shape[:2]
    cx, cy = brush_center
    r = max(4, int(radius))

    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r + 1)
    y2 = min(h, cy + r + 1)

    if x2 <= x1 or y2 <= y1:
        return mask

    roi_img = image_bgr[y1:y2, x1:x2].astype(np.float32)
    sample_col = image_bgr[min(h - 1, max(0, cy)), min(w - 1, max(0, cx))].astype(np.float32)

    # Radial distance and smooth falloff
    ys, xs = np.ogrid[y1 - cy : y2 - cy, x1 - cx : x2 - cx]
    dist = np.sqrt(xs * xs + ys * ys)
    spatial_w = np.clip(1.0 - dist / float(r), 0.0, 1.0)
    spatial_w = spatial_w * spatial_w * (3.0 - 2.0 * spatial_w)

    if smart:
        # Photoshop-style connected Quick Selection inside the brush ROI:
        # Lab perceptual distance + C++ floodFill connectivity from brush center + soft feather.
        roi_u8 = np.clip(roi_img, 0, 255).astype(np.uint8)
        roi_lab = cv2.cvtColor(roi_u8, cv2.COLOR_BGR2Lab).astype(np.float32)
        sy_loc = min(roi_lab.shape[0] - 1, max(0, cy - y1))
        sx_loc = min(roi_lab.shape[1] - 1, max(0, cx - x1))
        seed_lab = roi_lab[sy_loc, sx_loc]
        d_lab = roi_lab - seed_lab
        # Weight chroma (a, b) slightly higher than luminance so hue boundaries snap cleanly
        dist_lab = np.sqrt(0.55 * d_lab[:, :, 0] ** 2 + 1.25 * d_lab[:, :, 1] ** 2 + 1.25 * d_lab[:, :, 2] ** 2)
        tol_lab = 28.0 if tolerance is None else max(10.0, min(52.0, float(tolerance) * 90.0))
        tol_mask = (dist_lab <= tol_lab).astype(np.uint8) * 255
        tol_mask[sy_loc, sx_loc] = 255
        ff_mask = np.zeros((tol_mask.shape[0] + 2, tol_mask.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(tol_mask, ff_mask, (sx_loc, sy_loc), 128)
        connected = (tol_mask == 128).astype(np.float32)
        soft_edge = np.exp(-dist_lab / max(10.0, tol_lab * 0.78)).astype(np.float32)
        smart_affinity = np.clip(0.85 * connected + 0.15 * soft_edge, 0.0, 1.0)
        smart_affinity = cv2.GaussianBlur(smart_affinity, (3, 3), 0.7)
        brush_alpha = np.clip(spatial_w * smart_affinity, 0.0, 1.0).astype(np.float32)
    else:
        brush_alpha = spatial_w.astype(np.float32)

    updated = mask.copy()
    if mode == "keep_color":
        updated[y1:y2, x1:x2] = updated[y1:y2, x1:x2] * (1.0 - brush_alpha) + 1.0 * brush_alpha
    else:  # "make_bw"
        updated[y1:y2, x1:x2] = updated[y1:y2, x1:x2] * (1.0 - brush_alpha) + 0.0 * brush_alpha

    return np.clip(updated, 0.0, 1.0).astype(np.float32)


# =============================================================================
# 2. Portrait Depth Bokeh & Foreground Fill Light
# =============================================================================

def create_disc_kernel(radius: int) -> np.ndarray:
    """Generates an anti-aliased circular aperture disc kernel normalized to sum to 1.0."""
    r = max(1, int(radius))
    d = 2 * r + 1
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    dist = np.sqrt(x * x + y * y)
    # Anti-aliased outer edge (1px boundary)
    kernel = np.clip(r + 0.5 - dist, 0.0, 1.0)
    s = np.sum(kernel)
    return kernel / s if s > 0 else kernel


def build_bokeh_pyramid(
    image_bgr: np.ndarray,
    max_blur_radius: float = 28.0,
    num_levels: int = 4,
) -> list[np.ndarray]:
    """
    Precomputes a multi-level optical bokeh blur pyramid in gamma-linear space (gamma=2.2):
    - Level 0: Original sharp image.
    - Level 1..3: Progressively stronger optical disc bokeh blur with specular highlight bloom.
    - Allows interactive slider dragging to interpolate in <5ms at 60 FPS.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    # Linearize (gamma = 2.2) and add highlight bloom
    bgr_norm = image_bgr.astype(np.float32) / 255.0
    bgr_linear = np.power(bgr_norm, 2.2)

    # Specular boost for bright highlights
    lum = 0.114 * image_bgr[:, :, 0] + 0.587 * image_bgr[:, :, 1] + 0.299 * image_bgr[:, :, 2]
    highlight_factor = np.clip((lum - 215.0) / 40.0, 0.0, 1.0)[..., None]
    bgr_bloomed = bgr_linear * (1.0 + highlight_factor * 2.0)

    pyramid: list[np.ndarray] = [image_bgr.copy()]

    radii = np.linspace(max_blur_radius * 0.33, max_blur_radius, num_levels - 1)
    for r in radii:
        disc = create_disc_kernel(int(round(r)))
        blurred_lin = cv2.filter2D(bgr_bloomed, -1, disc, borderType=cv2.BORDER_REFLECT)
        # De-linearize to sRGB
        blurred_srgb = np.clip(np.power(np.clip(blurred_lin, 0.0, 1.0), 1.0 / 2.2) * 255.0, 0.0, 255.0).astype(np.uint8)
        pyramid.append(blurred_srgb)

    return pyramid


def apply_subject_fill_light(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    fill_light: float = 0.0,
) -> np.ndarray:
    """
    Applies gentle foreground fill light lifting midtones without blowing out highlights:
    - Guided by a heavily feathered subject mask to eliminate harsh boundary seams.
    """
    if image_bgr is None or mask is None or fill_light <= 0.01:
        return image_bgr

    h, w = image_bgr.shape[:2]
    sigma_fill = max(15.0, max(w, h) * 0.04)
    feather_mask = cv2.GaussianBlur(mask, (0, 0), sigma_fill)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2]
    # Soft-knee fill light formula
    v_lift = v + fill_light * (255.0 - v) * (v / 255.0)
    hsv[:, :, 2] = np.clip(v * (1.0 - feather_mask) + v_lift * feather_mask, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_portrait_bokeh(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bokeh_radius: int = 16,
    bokeh_pyramid: list[np.ndarray] | None = None,
    background_warmth: float = 0.0,
) -> np.ndarray:
    """
    Renders DSLR-quality Portrait Bokeh with optical lens simulation:
    - Uses precomputed bokeh_pyramid if provided for instantaneous <5ms rendering.
    - Normalized convolution prevents subject edge bleeding into background blur.
    - Specular highlight boost in linear space (gamma = 2.2) for crisp circular bokeh discs.
    - Optional background warmth adjustment (-50 to +50).
    """
    if image_bgr is None or mask is None:
        return image_bgr

    h, w = image_bgr.shape[:2]

    # If bokeh_pyramid is provided, interpolate between levels for instant 60 FPS slider drag
    if bokeh_pyramid and len(bokeh_pyramid) >= 2:
        # bokeh_radius in [0..40] -> level float in [0 .. len-1]
        max_r = 36.0
        lvl_f = float(np.clip(bokeh_radius / max_r * (len(bokeh_pyramid) - 1), 0.0, len(bokeh_pyramid) - 1))
        idx0 = int(math.floor(lvl_f))
        idx1 = min(len(bokeh_pyramid) - 1, idx0 + 1)
        frac = lvl_f - idx0

        if idx0 == idx1 or frac <= 1e-4:
            bg_bgr = bokeh_pyramid[idx0].astype(np.float32)
        else:
            bg_bgr = (1.0 - frac) * bokeh_pyramid[idx0].astype(np.float32) + frac * bokeh_pyramid[idx1].astype(np.float32)
    else:
        # On-the-fly optical disc blur
        r = max(1, int(bokeh_radius))
        disc = create_disc_kernel(r)

        # 1. Linearize BGR (gamma = 2.2) and highlight boost
        bgr_norm = image_bgr.astype(np.float32) / 255.0
        bgr_linear = np.power(bgr_norm, 2.2)
        lum = 0.114 * image_bgr[:, :, 0] + 0.587 * image_bgr[:, :, 1] + 0.299 * image_bgr[:, :, 2]
        highlight_factor = np.clip((lum - 215.0) / 40.0, 0.0, 1.0)[..., None]
        bgr_linear = bgr_linear * (1.0 + highlight_factor * 2.0)

        # 2. Normalized convolution using background weight
        W_bg = np.clip(1.0 - mask, 1e-3, 1.0)
        premult = bgr_linear * W_bg[..., None]
        blurred_premult = cv2.filter2D(premult, -1, disc, borderType=cv2.BORDER_REFLECT)
        blurred_weight = cv2.filter2D(W_bg, -1, disc, borderType=cv2.BORDER_REFLECT)[..., None]
        bg_linear = blurred_premult / np.maximum(blurred_weight, 1e-4)

        # De-linearize to sRGB
        bg_bgr = np.clip(np.power(np.clip(bg_linear, 0.0, 1.0), 1.0 / 2.2) * 255.0, 0.0, 255.0)

    # Optional background warmth
    if abs(background_warmth) > 0.5:
        # Warmth shifts Blue down and Red up
        shift = float(background_warmth) * 0.4
        bg_bgr[:, :, 0] = np.clip(bg_bgr[:, :, 0] - shift, 0.0, 255.0)  # B
        bg_bgr[:, :, 2] = np.clip(bg_bgr[:, :, 2] + shift, 0.0, 255.0)  # R

    # 3. Composite foreground subject over blurred background
    m3 = mask[..., None]
    composite = image_bgr.astype(np.float32) * m3 + bg_bgr * (1.0 - m3)
    return np.clip(composite, 0.0, 255.0).astype(np.uint8)


def apply_portrait_bokeh_and_fill_light(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bokeh_radius: int = 16,
    fill_light: float = 0.0,
    bokeh_pyramid: list[np.ndarray] | None = None,
    background_warmth: float = 0.0,
) -> np.ndarray:
    """Combines portrait depth lens blur, foreground fill light, and background warmth."""
    fg = apply_subject_fill_light(image_bgr, mask, fill_light=fill_light)
    return apply_portrait_bokeh(
        fg,
        mask,
        bokeh_radius=bokeh_radius,
        bokeh_pyramid=bokeh_pyramid,
        background_warmth=background_warmth,
    )


# =============================================================================
# ONNX Depth & Alpha Matting Engine + Cache
# =============================================================================

class DepthAndMatteEngine:
    """
    Unified high-performance Monocular Depth Map and Portrait Alpha Matte inference engine.
    - Loads local ONNX models via onnxruntime (DirectML / CPU):
        * Monocular 3D Depth Map: depth_anything_v2_vits.onnx or midas_v21_small_256.onnx
          Outputs normalized continuous depth map D(x, y) in [0.0, 1.0] where 0.0 = closest foreground
          and 1.0 = farthest background.
        * Portrait Alpha Matte: modnet.onnx, u2netp.onnx, or rmbg-1.4.onnx
          Outputs continuous human subject alpha matte M(x, y) in [0.0, 1.0].
    - Zero-dependency instant fallback when ONNX weights are not present:
        * Anatomical face-seeded Guided Filter + perspective ground gradient & sharpness estimator.
    - High-speed per-image cache keyed by (cache_id, width, height) enabling 60 FPS real-time
      slider manipulation in pure NumPy.
    """

    _instance: DepthAndMatteEngine | None = None

    def __init__(self, models_dir: Path | str | None = None) -> None:
        self.search_dirs: list[Path] = []
        if models_dir is not None:
            self.search_dirs.append(Path(models_dir))

        # 1. Bundled models/ (PyInstaller sys._MEIPASS / models, or repo / models)
        try:
            from src.utils.paths import get_bundle_dir
            bundle_models = get_bundle_dir() / "models"
            if bundle_models not in self.search_dirs:
                self.search_dirs.append(bundle_models)
        except Exception:
            pass

        # 2. User AppData models/ (%LOCALAPPDATA%/PhotoGallery/models)
        try:
            from src.utils.paths import get_app_data_dir
            app_data_models = get_app_data_dir() / "models"
            if app_data_models not in self.search_dirs:
                self.search_dirs.append(app_data_models)
        except Exception:
            pass

        # 3. Local workspace models/
        self.search_dirs.append(Path("models").resolve())
        # AppData models/
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            self.search_dirs.append(Path(local_app_data) / "PhotoGallery" / "models")
            self.search_dirs.append(Path(local_app_data) / "WindowsMediaGallery" / "models")
        # Relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        self.search_dirs.append(project_root / "models")

        self._depth_session: Any = None
        self._depth_model_name: str | None = None
        self._matte_session: Any = None
        self._matte_model_name: str | None = None
        self._normal_session: Any = None
        self._normal_model_name: str | None = None
        self._cache: dict[tuple[Any, int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._base_cache: dict[Any, tuple[np.ndarray, np.ndarray]] = {}

        self._init_onnx_sessions()
        if self._matte_session is None:
            self.ensure_depth_models(download_if_missing=True)

    def _find_model_file(self, filenames: list[str]) -> Path | None:
        for d in self.search_dirs:
            for fn in filenames:
                cand = d / fn
                if cand.is_file():
                    return cand
        return None

    def _init_single_matte_session(self, matte_path: Path) -> bool:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            prov_list = []
            if "DmlExecutionProvider" in providers:
                prov_list.append("DmlExecutionProvider")
            prov_list.append("CPUExecutionProvider")

            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._matte_session = ort.InferenceSession(str(matte_path), sess_opts, providers=prov_list)
            self._matte_model_name = matte_path.name
            logger.info("Matte session loaded: %s with providers %s", matte_path.name, prov_list)
            return True
        except Exception as ex:
            logger.debug("Failed to initialize matte ONNX session from %s: %s", matte_path, ex)
            return False

    def _init_onnx_sessions(self) -> None:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            prov_list = []
            if "DmlExecutionProvider" in providers:
                prov_list.append("DmlExecutionProvider")
            prov_list.append("CPUExecutionProvider")

            depth_path = self._find_model_file([
                "depth_anything_v2_vits.onnx",
                "midas_v21_small_256.onnx",
                "depth_anything_v2.onnx",
            ])
            if depth_path:
                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._depth_session = ort.InferenceSession(str(depth_path), sess_opts, providers=prov_list)
                self._depth_model_name = depth_path.name
                logger.info("Depth session loaded: %s with providers %s", depth_path.name, prov_list)

            matte_path = self._find_model_file([
                "u2netp.onnx",
                "modnet.onnx",
                "rmbg-1.4.onnx",
            ])
            if matte_path:
                self._init_single_matte_session(matte_path)

            normal_path = self._find_model_file([
                "dsine.onnx",
                "normal_bae.onnx",
            ])
            if normal_path:
                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._normal_session = ort.InferenceSession(str(normal_path), sess_opts, providers=prov_list)
                self._normal_model_name = normal_path.name
                logger.info("Normal session loaded: %s with providers %s", normal_path.name, prov_list)
        except Exception as ex:
            logger.debug("ONNX session initialization skipped / fallback active: %s", ex)

    def ensure_depth_models(self, download_if_missing: bool = True) -> dict[str, Any]:
        """
        Checks availability of ONNX depth and portrait matte models.
        If download_if_missing is True and matte model is missing, downloads
        the lightweight ONNX human segmentation model (u2netp.onnx ~4.5 MB)
        with a 5-second timeout and initializes onnxruntime session with
        DmlExecutionProvider or CPUExecutionProvider.
        """
        if self._matte_session is None:
            matte_path = self._find_model_file([
                "u2netp.onnx",
                "modnet.onnx",
                "rmbg-1.4.onnx",
            ])
            if not matte_path and download_if_missing:
                try:
                    from src.utils.paths import get_app_data_dir
                    target_dir = get_app_data_dir() / "models"
                except Exception:
                    target_dir = Path("models").resolve()
                target_dir.mkdir(parents=True, exist_ok=True)
                target_dest = target_dir / "u2netp.onnx"
                if not target_dest.is_file():
                    _download_u2netp_model(target_dest, timeout=5.0)
                if target_dest.is_file():
                    matte_path = target_dest

            if matte_path and self._matte_session is None:
                self._init_single_matte_session(matte_path)

        return {
            "has_depth_model": self._depth_session is not None,
            "depth_model": self._depth_model_name,
            "has_matte_model": self._matte_session is not None,
            "matte_model": self._matte_model_name,
            "using_fallback": self._depth_session is None or self._matte_session is None,
        }

    def clear_cache(self) -> None:
        self._cache.clear()
        self._base_cache.clear()

    def get_depth_and_matte(
        self,
        image_rgb: np.ndarray,
        media_id: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (depth_map, alpha_matte):
        - depth_map: continuous float32 [0.0, 1.0] where 0.0 = closest foreground, 1.0 = farthest background.
        - alpha_matte: continuous float32 [0.0, 1.0] where 1.0 = subject, 0.0 = background.
        Cached by (cache_key, h, w) for instantaneous 60 FPS slider manipulation.
        """
        if image_rgb is None or image_rgb.size == 0:
            dummy = np.zeros((10, 10), dtype=np.float32)
            return dummy, dummy

        h, w = image_rgb.shape[:2]
        if media_id is not None:
            cache_id = media_id
        else:
            cache_id = (
                image_rgb.shape,
                int(image_rgb[0, 0, 0]),
                int(image_rgb[-1, -1, 0]),
                int(image_rgb[h // 2, w // 2, 0]),
            )
        cache_key = (cache_id, w, h)

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check if base maps are already computed for this media_id
        if cache_id in self._base_cache:
            base_depth, base_matte = self._base_cache[cache_id]
            d_out = cv2.resize(base_depth, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            m_out = cv2.resize(base_matte, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            res = (d_out, m_out)
            self._cache[cache_key] = res
            return res

        # Downsample image for high-speed base map computation (~50ms max)
        base_scale = min(1.0, 640.0 / max(w, h))
        if base_scale < 0.95:
            bw = max(16, int(round(w * base_scale)))
            bh = max(16, int(round(h * base_scale)))
            comp_rgb = cv2.resize(image_rgb, (bw, bh), interpolation=cv2.INTER_AREA)
        else:
            bw, bh = w, h
            comp_rgb = image_rgb

        depth_map: np.ndarray | None = None
        alpha_matte: np.ndarray | None = None

        # 1. Monocular 3D Depth Map
        if self._depth_session is not None:
            try:
                input_name = self._depth_session.get_inputs()[0].name
                in_shape = self._depth_session.get_inputs()[0].shape
                target_h = in_shape[2] if len(in_shape) == 4 and isinstance(in_shape[2], int) and in_shape[2] > 0 else 518
                target_w = in_shape[3] if len(in_shape) == 4 and isinstance(in_shape[3], int) and in_shape[3] > 0 else 518

                img_f = cv2.resize(comp_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                blob = ((img_f - mean) / std).transpose(2, 0, 1)[None, ...]

                out = self._depth_session.run(None, {input_name: blob})[0]
                out_2d = out[0, 0] if out.ndim == 4 else (out[0] if out.ndim == 3 else out)
                d_min, d_max = float(np.min(out_2d)), float(np.max(out_2d))
                if d_max > d_min:
                    norm_disp = (out_2d - d_min) / (d_max - d_min)
                else:
                    norm_disp = np.zeros_like(out_2d)

                norm_depth = 1.0 - norm_disp
                depth_map = cv2.resize(norm_depth, (bw, bh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            except Exception as e:
                logger.debug("ONNX depth inference error, using fallback: %s", e)
                depth_map = None

        # 2. Portrait Alpha Matte
        if self._matte_session is not None:
            try:
                input_name = self._matte_session.get_inputs()[0].name
                in_shape = self._matte_session.get_inputs()[0].shape
                is_u2net = "u2net" in (self._matte_model_name or "").lower()
                target_h = in_shape[2] if len(in_shape) == 4 and isinstance(in_shape[2], int) and in_shape[2] > 0 else (320 if is_u2net else 512)
                target_w = in_shape[3] if len(in_shape) == 4 and isinstance(in_shape[3], int) and in_shape[3] > 0 else (320 if is_u2net else 512)

                img_f = cv2.resize(comp_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                if is_u2net:
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    blob = ((img_f - mean) / std).transpose(2, 0, 1)[None, ...].astype(np.float32)
                else:
                    blob = ((img_f - 0.5) / 0.5).transpose(2, 0, 1)[None, ...].astype(np.float32)

                out = self._matte_session.run(None, {input_name: blob})[0]
                out_2d = out[0, 0] if out.ndim == 4 else (out[0] if out.ndim == 3 else out)
                m_min, m_max = float(np.min(out_2d)), float(np.max(out_2d))
                if m_max > m_min:
                    norm_m = (out_2d - m_min) / (m_max - m_min)
                else:
                    norm_m = np.zeros_like(out_2d)
                raw_matte = cv2.resize(norm_m, (bw, bh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                alpha_matte = refine_mask_narrow_band(cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR), raw_matte)
            except Exception as e:
                logger.debug("ONNX matte inference error, using fallback: %s", e)
                alpha_matte = None

        # Zero-dependency fallback on base image
        bgr = cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR)
        if alpha_matte is None:
            alpha_matte = estimate_subject_mask(bgr)

        if depth_map is None:
            inv_depth = estimate_depth_map(bgr, subject_mask=alpha_matte)
            depth_map = np.clip(1.0 - inv_depth, 0.0, 1.0).astype(np.float32)

        base_res = (
            np.clip(depth_map, 0.0, 1.0).astype(np.float32),
            np.clip(alpha_matte, 0.0, 1.0).astype(np.float32),
        )
        self._base_cache[cache_id] = base_res

        # Scale to requested (w, h)
        if (bw, bh) == (w, h):
            res = base_res
        else:
            res = (
                cv2.resize(base_res[0], (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32),
                cv2.resize(base_res[1], (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32),
            )
        self._cache[cache_key] = res
        return res

    def get_depth_matte_and_weights(
        self,
        image_rgb: np.ndarray,
        media_id: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (depth_map, alpha_matte, near_weight, far_weight) cached for 60 FPS slider manipulation.
        """
        depth_map, alpha_matte = self.get_depth_and_matte(image_rgb, media_id=media_id)
        h, w = image_rgb.shape[:2]
        if media_id is not None:
            cache_id = media_id
        else:
            cache_id = (
                image_rgb.shape,
                int(image_rgb[0, 0, 0]),
                int(image_rgb[-1, -1, 0]),
                int(image_rgb[h // 2, w // 2, 0]),
            )
        w_key = (cache_id, w, h, "weights")
        if w_key in self._cache:
            near_w, far_w = self._cache[w_key]
            return depth_map, alpha_matte, near_w, far_w

        near_w = np.maximum(alpha_matte, 1.0 - depth_map).astype(np.float32)
        far_w = (depth_map * (1.0 - alpha_matte)).astype(np.float32)
        self._cache[w_key] = (near_w, far_w)
        return depth_map, alpha_matte, near_w, far_w

    def get_refined_depth_and_normals(
        self,
        image_rgb: np.ndarray,
        media_id: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (refined_depth, alpha_matte, normal_map) where:
        - refined_depth: float32 (H, W) [0.0=near, 1.0=far] snapped via Joint Guided Filter.
        - alpha_matte: float32 (H, W) [0.0=bg, 1.0=subject].
        - normal_map: float32 (H, W, 3) unit 3D surface normals (Nx, Ny, Nz) with Nz > 0.
        Cached by (cache_id, w, h, 'normals') for real-time virtual light manipulation.
        """
        depth_map, alpha_matte = self.get_depth_and_matte(image_rgb, media_id=media_id)
        h, w = image_rgb.shape[:2]
        if media_id is not None:
            cache_id = media_id
        else:
            cache_id = (
                image_rgb.shape,
                int(image_rgb[0, 0, 0]),
                int(image_rgb[-1, -1, 0]),
                int(image_rgb[h // 2, w // 2, 0]),
            )
        n_key = (cache_id, w, h, "normals")
        if n_key in self._cache:
            refined_d, normals = self._cache[n_key]
            return refined_d, alpha_matte, normals

        scale = min(1.0, 512.0 / float(max(1, max(h, w))))
        ww = max(16, int(round(w * scale)))
        wh = max(16, int(round(h * scale)))
        if (ww, wh) != (w, h):
            work_rgb = cv2.resize(image_rgb, (ww, wh), interpolation=cv2.INTER_AREA)
            work_depth = cv2.resize(depth_map, (ww, wh), interpolation=cv2.INTER_LINEAR)
            work_matte = cv2.resize(alpha_matte, (ww, wh), interpolation=cv2.INTER_LINEAR)
        else:
            work_rgb = image_rgb
            work_depth = depth_map
            work_matte = alpha_matte

        work_bgr = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2BGR)
        refined_work = np.clip(
            guided_filter_matte(work_bgr, work_depth, radius=12, eps=2e-3, subsample=2),
            0.0,
            1.0,
        ).astype(np.float32)

        normal_work: np.ndarray | None = None
        if self._normal_session is not None:
            try:
                input_name = self._normal_session.get_inputs()[0].name
                in_shape = self._normal_session.get_inputs()[0].shape
                th = in_shape[2] if len(in_shape) == 4 and isinstance(in_shape[2], int) and in_shape[2] > 0 else 384
                tw = in_shape[3] if len(in_shape) == 4 and isinstance(in_shape[3], int) and in_shape[3] > 0 else 384
                img_f = cv2.resize(work_rgb, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                blob = ((img_f - 0.5) / 0.5).transpose(2, 0, 1)[None, ...].astype(np.float32)
                out = self._normal_session.run(None, {input_name: blob})[0]
                out_chw = out[0] if out.ndim == 4 else out
                n_hw3 = np.transpose(out_chw[:3], (1, 2, 0)).astype(np.float32)
                n_hw3 = cv2.resize(n_hw3, (ww, wh), interpolation=cv2.INTER_LINEAR)
                n_hw3[:, :, 2] = np.maximum(np.abs(n_hw3[:, :, 2]), 0.1)
                norm = np.linalg.norm(n_hw3, axis=2, keepdims=True)
                normal_work = (n_hw3 / np.maximum(norm, 1e-6)).astype(np.float32)
            except Exception:
                normal_work = None

        if normal_work is None:
            elev = 1.0 - refined_work
            z_fine = cv2.GaussianBlur(elev, (0, 0), sigmaX=1.5)
            z_coarse = cv2.GaussianBlur(elev, (0, 0), sigmaX=5.0)
            z_blend = (0.55 * z_fine + 0.45 * z_coarse).astype(np.float32)
            gx = cv2.Scharr(z_blend, cv2.CV_32F, 1, 0) / 32.0
            gy = cv2.Scharr(z_blend, cv2.CV_32F, 0, 1) / 32.0

            fg_bin = (work_matte > 0.35).astype(np.uint8)
            if np.any(fg_bin):
                dt = cv2.distanceTransform(fg_bin, cv2.DIST_L2, 5).astype(np.float32)
                dt_max = float(np.max(dt))
                if dt_max > 1.0:
                    dome = np.sqrt(np.clip(dt / dt_max, 0.0, 1.0)) * work_matte
                    dome = cv2.GaussianBlur(dome, (0, 0), sigmaX=2.5)
                    dgx = (cv2.Scharr(dome, cv2.CV_32F, 1, 0) / 32.0) * (dt_max * 0.04)
                    dgy = (cv2.Scharr(dome, cv2.CV_32F, 0, 1) / 32.0) * (dt_max * 0.04)
                    gx = gx + 0.45 * dgx
                    gy = gy + 0.45 * dgy

            # Subtle luminance high-pass micro-relief so specular highlights glint on texture
            gray_f = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            hp = gray_f - cv2.GaussianBlur(gray_f, (0, 0), sigmaX=2.0)
            gx = gx + 0.14 * (cv2.Scharr(hp, cv2.CV_32F, 1, 0) / 32.0)
            gy = gy + 0.14 * (cv2.Scharr(hp, cv2.CV_32F, 0, 1) / 32.0)

            nx = -gx * 3.2
            ny = -gy * 3.2
            nz = np.ones_like(nx, dtype=np.float32)
            n_stack = np.stack([nx, ny, nz], axis=-1).astype(np.float32)
            norm = np.linalg.norm(n_stack, axis=2, keepdims=True)
            normal_work = (n_stack / np.maximum(norm, 1e-6)).astype(np.float32)

        if (ww, wh) != (w, h):
            refined_out = cv2.resize(refined_work, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            normal_out = cv2.resize(normal_work, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            n_norm = np.linalg.norm(normal_out, axis=2, keepdims=True)
            normal_out = (normal_out / np.maximum(n_norm, 1e-6)).astype(np.float32)
        else:
            refined_out = refined_work
            normal_out = normal_work

        self._cache[n_key] = (refined_out, normal_out)
        return refined_out, alpha_matte, normal_out

    def get_region_mask(
        self,
        image_rgb: np.ndarray,
        region: str,
        media_id: Any = None,
    ) -> np.ndarray:
        """
        Returns a soft float32 [0.0, 1.0] mask of shape (H, W) for regional Fine Tune edits:
        - 'subject': foreground alpha matte
        - 'background': 1.0 - alpha matte
        - 'sky': far upper-frame sky segmented via depth + chromaticity + top-edge connected
          components + Joint Guided Filter edge snapping.
        """
        h, w = image_rgb.shape[:2]
        reg = (region or "whole").lower()
        if reg == "custom":
            return np.zeros((h, w), dtype=np.float32)
        if reg not in ("subject", "background", "sky"):
            return np.ones((h, w), dtype=np.float32)

        depth_map, alpha_matte = self.get_depth_and_matte(image_rgb, media_id=media_id)
        if media_id is not None:
            cache_id = media_id
        else:
            cache_id = (
                image_rgb.shape,
                int(image_rgb[0, 0, 0]),
                int(image_rgb[-1, -1, 0]),
                int(image_rgb[h // 2, w // 2, 0]),
            )
        r_key = (cache_id, w, h, f"region_{reg}")
        if r_key in self._cache:
            return self._cache[r_key]

        if reg == "subject":
            mask = np.clip(alpha_matte, 0.0, 1.0).astype(np.float32)
        elif reg == "background":
            mask = np.clip(1.0 - alpha_matte, 0.0, 1.0).astype(np.float32)
        else:
            # Sky detection on fast 320px proxy
            scale = min(1.0, 320.0 / max(1, max(h, w)))
            sw = max(24, int(round(w * scale)))
            sh = max(24, int(round(h * scale)))
            small_rgb = cv2.resize(image_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
            small_depth = cv2.resize(depth_map, (sw, sh), interpolation=cv2.INTER_LINEAR)
            small_bg = 1.0 - cv2.resize(alpha_matte, (sw, sh), interpolation=cv2.INTER_LINEAR)

            hsv = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)
            rf = small_rgb[:, :, 0].astype(np.float32)
            gf = small_rgb[:, :, 1].astype(np.float32)
            bf = small_rgb[:, :, 2].astype(np.float32)
            lum = 0.299 * rf + 0.587 * gf + 0.114 * bf
            sat = hsv[:, :, 1].astype(np.float32)

            # Texture variance + 4-neighbor Laplacian check so rough foliage/brick/roofs in upper frame are suppressed
            mean_l = cv2.blur(lum, (5, 5))
            mean_l2 = cv2.blur(lum * lum, (5, 5))
            local_std = np.sqrt(np.maximum(0.0, mean_l2 - mean_l * mean_l))
            lap_l = cv2.blur(np.abs(cv2.Laplacian(lum, cv2.CV_32F, ksize=1)) * 0.35, (5, 5))
            grad = np.maximum(local_std * 1.4, lap_l)
            smooth_prior = np.clip(1.0 - (grad - 18.0) / 42.0, 0.0, 1.0)

            y_norm = np.linspace(0.0, 1.0, sh, dtype=np.float32)[:, None]
            # Blue/cyan sky, low-sat overcast sky, OR warm golden-hour / sunset sky
            is_blue_sky = (bf >= rf - 8.0) & (bf >= 75.0) & (gf >= 55.0)
            is_overcast_sky = (lum >= 145.0) & (sat <= 95.0)
            is_sunset_sky = (
                (lum >= 115.0)
                & (rf >= gf)
                & (gf >= bf * 0.65)
                & ((rf - bf) >= 18.0)
                & (y_norm <= 0.62)
                & (grad <= 28.0)
            )
            upper_prior = np.clip((0.78 - y_norm) / 0.25, 0.0, 1.0)
            far_prior = np.clip((small_depth - 0.35) / 0.35, 0.0, 1.0)

            cand = (
                (is_blue_sky | is_overcast_sky | is_sunset_sky).astype(np.float32)
                * (0.15 + 0.85 * smooth_prior)
                * (upper_prior > 0.05).astype(np.float32)
                * (small_bg > 0.35).astype(np.float32)
                * (0.35 + 0.65 * far_prior)
            )
            bin_cand = (cand > 0.30).astype(np.uint8)

            # Keep only connected components touching the top 15% of the frame
            num_labels, labels = cv2.connectedComponents(bin_cand, connectivity=8)
            top_band_h = max(2, int(round(sh * 0.15)))
            top_labels = set(np.unique(labels[:top_band_h, :]).tolist()) - {0}
            if top_labels:
                anchored = np.isin(labels, list(top_labels)).astype(np.float32)
                anchored = cv2.morphologyEx(anchored, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            else:
                anchored = np.zeros((sh, sw), dtype=np.float32)

            soft_small = anchored * upper_prior * np.clip(small_bg, 0.0, 1.0)
            guide_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            soft_up = cv2.resize(soft_small, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = guided_filter_matte(guide_bgr, soft_up, radius=12, eps=1e-3, subsample=2)
            mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

        self._cache[r_key] = mask
        return mask


_STROKE_MASK_CACHE: dict[tuple[Any, ...], np.ndarray] = {}


def apply_region_mask_strokes(
    mask: np.ndarray,
    image_bgr: np.ndarray,
    strokes: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> np.ndarray:
    """Applies normalized paint/erase brush strokes onto a [0..1] float32 region mask with O(1) incremental cache."""
    if mask is None or image_bgr is None or not strokes:
        return mask
    h, w = mask.shape[:2]
    dim = max(1, max(h, w))

    stroke_sig = tuple(
        (
            round(float(st.get("x", 0.5)), 4),
            round(float(st.get("y", 0.5)), 4),
            round(float(st.get("r", 0.04)), 4),
            str(st.get("mode", "paint")).lower(),
            bool(st.get("smart", True)),
            bool(st.get("cont", False)),
        )
        for st in strokes
    )
    img_sig = (
        h,
        w,
        int(image_bgr[0, 0, 0]),
        int(image_bgr[h // 2, w // 2, 1]),
        float(mask[0, 0]),
        float(mask[h // 2, w // 2]),
    )
    full_key = (img_sig, stroke_sig)
    if full_key in _STROKE_MASK_CACHE:
        return _STROKE_MASK_CACHE[full_key].copy()

    start_idx = 0
    updated: np.ndarray | None = None
    if len(stroke_sig) > 1:
        prev_key = (img_sig, stroke_sig[:-1])
        if prev_key in _STROKE_MASK_CACHE:
            updated = _STROKE_MASK_CACHE[prev_key].copy()
            start_idx = len(stroke_sig) - 1
    if updated is None:
        updated = mask.astype(np.float32, copy=True)

    prev_pt: tuple[float, float, str] | None = None
    if start_idx > 0:
        ps = strokes[start_idx - 1]
        pmode = "keep_color" if str(ps.get("mode", "paint")).lower() == "paint" else "make_bw"
        prev_pt = (float(np.clip(float(ps.get("x", 0.5)), 0.0, 1.0)), float(np.clip(float(ps.get("y", 0.5)), 0.0, 1.0)), pmode)

    for st in strokes[start_idx:]:
        try:
            raw_mode = str(st.get("mode", "paint")).lower()
            if raw_mode == "invert":
                updated = np.clip(1.0 - updated, 0.0, 1.0)
                prev_pt = None
                continue
            nx = float(np.clip(float(st.get("x", 0.5)), 0.0, 1.0))
            ny = float(np.clip(float(st.get("y", 0.5)), 0.0, 1.0))
            nr = float(st.get("r", 0.04))
            mode = "keep_color" if raw_mode == "paint" else "make_bw"
            smart = bool(st.get("smart", True))
            cont = bool(st.get("cont", False))
            rad_px = max(4, int(round(nr * dim)))

            pts: list[tuple[float, float]] = [(nx, ny)]
            if cont and prev_pt is not None and prev_pt[2] == mode:
                px0, py0, _ = prev_pt
                dist_norm = math.hypot(nx - px0, ny - py0)
                step = max(1e-4, nr * 0.25)
                if dist_norm > step:
                    steps = min(32, int(math.ceil(dist_norm / step)))
                    pts = [
                        (px0 + (nx - px0) * (k / float(steps)), py0 + (ny - py0) * (k / float(steps)))
                        for k in range(1, steps + 1)
                    ]

            for sx, sy in pts:
                px = int(round(sx * (w - 1)))
                py = int(round(sy * (h - 1)))
                updated = refine_mask_brush(
                    updated, image_bgr, (px, py), radius=rad_px, mode=mode, smart=smart
                )
            prev_pt = (nx, ny, mode)
        except Exception:
            continue

    res = np.clip(updated, 0.0, 1.0).astype(np.float32)
    if len(_STROKE_MASK_CACHE) >= 32:
        _STROKE_MASK_CACHE.pop(next(iter(_STROKE_MASK_CACHE)))
    _STROKE_MASK_CACHE[full_key] = res.copy()
    return res


_SHARED_DEPTH_ENGINE: DepthAndMatteEngine | None = None


def get_depth_and_matte_engine() -> DepthAndMatteEngine:
    global _SHARED_DEPTH_ENGINE
    if _SHARED_DEPTH_ENGINE is None:
        _SHARED_DEPTH_ENGINE = DepthAndMatteEngine()
    return _SHARED_DEPTH_ENGINE


# =============================================================================
# 2b. Depth & Light 2.0: 3D Surface-Normal Relighting & Virtual Light Studio
# =============================================================================

_SRGB_TO_LIN_LUT = np.power(np.arange(256, dtype=np.float32) / 255.0, 2.2).astype(np.float32)
_LIN_TO_SRGB_LUT = np.clip(
    np.round(np.power(np.linspace(0.0, 1.0, 4096, dtype=np.float32), 1.0 / 2.2) * 255.0),
    0,
    255,
).astype(np.uint8)
_UV_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _get_cached_uv_grids(w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = (int(w), int(h))
    cached = _UV_GRID_CACHE.get(key)
    if cached is not None:
        return cached
    aspect = float(w) / float(max(1, h))
    u = np.linspace(0.0, aspect, w, dtype=np.float32)[None, :]
    v = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    u_norm = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    v_norm = v
    if len(_UV_GRID_CACHE) >= 16:
        _UV_GRID_CACHE.pop(next(iter(_UV_GRID_CACHE)))
    val = (u, v, u_norm, v_norm)
    _UV_GRID_CACHE[key] = val
    return val


def make_default_virtual_light(**overrides: Any) -> dict[str, Any]:
    """Creates a normalized virtual 3D light descriptor for PhotoAdjustmentRecipe."""
    light: dict[str, Any] = {
        "enabled": True,
        "type": "point",
        "x": 0.25,
        "y": 0.25,
        "z": 0.25,
        "aim_x": 0.50,
        "aim_y": 0.50,
        "cone_angle": 40.0,
        "kelvin": 4500,
        "color": "",
        "intensity": 70.0,
        "softness": 45.0,
        "gloss": 30.0,
        "depth_from": 0.0,
        "depth_to": 1.0,
        "gobo": "none",
    }
    light.update(overrides)
    return light


def kelvin_to_rgb_srgb(kelvin: float) -> tuple[int, int, int]:
    """Converts color temperature in Kelvin (1800K..10000K) to (R, G, B) uint8 sRGB."""
    temp = max(1000.0, min(40000.0, float(kelvin))) / 100.0
    if temp <= 66.0:
        r = 255.0
        g = 99.4708025861 * math.log(max(1.0, temp)) - 161.1195681661
        if temp <= 19.0:
            b = 0.0
        else:
            b = 138.5177312231 * math.log(max(1.0, temp - 10.0)) - 305.0447927307
    else:
        r = 329.698727446 * ((temp - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((temp - 60.0) ** -0.0755148492)
        b = 255.0
    return (
        int(max(0, min(255, round(r)))),
        int(max(0, min(255, round(g)))),
        int(max(0, min(255, round(b)))),
    )


def parse_hex_color(hex_str: str) -> tuple[int, int, int] | None:
    s = (hex_str or "").strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return None


def light_color_srgb(light: dict[str, Any]) -> tuple[int, int, int]:
    parsed = parse_hex_color(str(light.get("color", "")))
    if parsed is not None:
        return parsed
    return kelvin_to_rgb_srgb(float(light.get("kelvin", 4500)))


def light_color_linear(light: dict[str, Any]) -> np.ndarray:
    r, g, b = light_color_srgb(light)
    return np.power(np.array([r, g, b], dtype=np.float32) / 255.0, 2.2).astype(np.float32)


RELIGHT_LOOK_PRESETS: list[dict[str, Any]] = [
    {
        "id": "warm_streetlamp",
        "name": "Warm Streetlamp",
        "preview_style": "warm_streetlamp",
        "preview_colors": ("#ff7e28", "#2d1a10"),
        "ambient_dim": 45.0,
        "lights": [
            make_default_virtual_light(
                type="spot", x=0.18, y=0.15, z=0.20, aim_x=0.52, aim_y=0.52,
                cone_angle=48.0, kelvin=2100, color="#ff7e28", intensity=85.0,
                softness=40.0, gloss=55.0, depth_from=0.0, depth_to=0.85,
            ),
        ],
    },
    {
        "id": "golden_rim",
        "name": "Golden Rim",
        "preview_style": "golden_rim",
        "preview_colors": ("#ffb85c", "#3a2b1c"),
        "ambient_dim": 20.0,
        "lights": [
            make_default_virtual_light(
                type="rim", x=0.84, y=0.18, z=0.75, aim_x=0.48, aim_y=0.45,
                kelvin=3200, color="#ffb85c", intensity=88.0, softness=45.0,
                gloss=40.0, depth_from=0.0, depth_to=0.70,
            ),
            make_default_virtual_light(
                type="point", x=0.22, y=0.35, z=0.15, kelvin=5200, color="",
                intensity=30.0, softness=75.0, gloss=15.0, depth_from=0.0, depth_to=0.55,
            ),
        ],
    },
    {
        "id": "studio_rembrandt",
        "name": "Rembrandt",
        "preview_style": "studio_rembrandt",
        "preview_colors": ("#ffe8d5", "#23252a"),
        "ambient_dim": 35.0,
        "lights": [
            make_default_virtual_light(
                type="spot", x=0.22, y=0.20, z=0.18, aim_x=0.52, aim_y=0.48,
                cone_angle=38.0, kelvin=4600, color="#ffe8d5", intensity=82.0,
                softness=55.0, gloss=28.0, depth_from=0.0, depth_to=0.65,
            ),
            make_default_virtual_light(
                type="point", x=0.80, y=0.55, z=0.30, kelvin=6200, color="",
                intensity=22.0, softness=85.0, gloss=5.0, depth_from=0.0, depth_to=0.70,
            ),
        ],
    },
    {
        "id": "dual_neon",
        "name": "Dual Neon",
        "preview_style": "dual_neon",
        "preview_colors": ("#00e5ff", "#ff2a9d"),
        "ambient_dim": 55.0,
        "lights": [
            make_default_virtual_light(
                type="rim", x=0.12, y=0.38, z=0.40, aim_x=0.50, aim_y=0.48,
                kelvin=6500, color="#00e5ff", intensity=88.0, softness=35.0,
                gloss=65.0, depth_from=0.0, depth_to=0.75,
            ),
            make_default_virtual_light(
                type="rim", x=0.88, y=0.36, z=0.45, aim_x=0.50, aim_y=0.48,
                kelvin=6500, color="#ff2a9d", intensity=88.0, softness=35.0,
                gloss=65.0, depth_from=0.0, depth_to=0.75,
            ),
        ],
    },
    {
        "id": "moonlit_backlight",
        "name": "Moonlit",
        "preview_style": "moonlit_backlight",
        "preview_colors": ("#9cc5ff", "#101a2e"),
        "ambient_dim": 50.0,
        "lights": [
            make_default_virtual_light(
                type="rim", x=0.55, y=0.10, z=0.85, aim_x=0.50, aim_y=0.52,
                kelvin=8800, color="#9cc5ff", intensity=90.0, softness=50.0,
                gloss=40.0, depth_from=0.0, depth_to=1.0,
            ),
        ],
    },
    {
        "id": "window_gobo",
        "name": "Window Gobo",
        "preview_style": "window_gobo",
        "preview_colors": ("#ffd296", "#2c2a29"),
        "ambient_dim": 40.0,
        "lights": [
            make_default_virtual_light(
                type="spot", x=0.14, y=0.16, z=0.22, aim_x=0.55, aim_y=0.55,
                cone_angle=52.0, kelvin=3800, color="#ffd296", intensity=90.0,
                softness=30.0, gloss=25.0, depth_from=0.0, depth_to=1.0,
                gobo="window",
            ),
        ],
    },
]


def _compute_gobo_mask(
    gobo: str,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    lx: float,
    ly: float,
    aim_x: float,
    aim_y: float,
    softness: float,
) -> np.ndarray | float:
    g = (gobo or "none").lower()
    if g == "none" or g == "":
        return 1.0
    du = u_grid - aim_x
    dv = v_grid - aim_y
    adx = aim_x - lx
    ady = aim_y - ly
    ang = math.atan2(ady, adx) if math.hypot(adx, ady) > 1e-4 else 0.0
    ca, sa = math.cos(ang), math.sin(ang)
    rx = du * ca + dv * sa
    ry = -du * sa + dv * ca
    soft_t = max(0.05, min(1.0, float(softness) / 100.0))

    if g == "blinds":
        wave = np.sin((ry * 12.0) * math.pi)
        thresh = -0.15 + 0.1 * (1.0 - soft_t)
        sharp = 3.0 + 10.0 * (1.0 - soft_t)
        return np.clip(0.15 + 0.85 * (1.0 / (1.0 + np.exp(-sharp * (wave - thresh)))), 0.12, 1.0).astype(np.float32)

    if g == "window":
        bar_w = 0.045 + 0.03 * soft_t
        mullion_x = np.exp(-0.5 * (rx / bar_w) ** 2)
        mullion_y = np.exp(-0.5 * (ry / bar_w) ** 2)
        border_mask = np.clip(1.0 - np.maximum(mullion_x, mullion_y) * 0.9, 0.10, 1.0)
        return border_mask.astype(np.float32)

    if g == "foliage":
        p1 = np.sin(rx * 22.0 + np.cos(ry * 17.0)) * np.cos(ry * 21.0 - np.sin(rx * 13.0))
        p2 = np.sin((rx + ry) * 31.0) * np.cos((rx - ry) * 29.0)
        pat = 0.6 * p1 + 0.4 * p2
        sharp = 2.0 + 5.0 * (1.0 - soft_t)
        return np.clip(0.20 + 0.80 * (1.0 / (1.0 + np.exp(-sharp * pat))), 0.15, 1.0).astype(np.float32)

    return 1.0


def apply_virtual_3d_relighting(
    img_rgb: np.ndarray,
    refined_depth: np.ndarray,
    alpha_matte: np.ndarray,
    normal_map: np.ndarray,
    near_weight: np.ndarray,
    far_weight: np.ndarray,
    *,
    relight_near: float = 0.0,
    relight_far: float = 0.0,
    warmth_near: float = 0.0,
    warmth_far: float = 0.0,
    ambient_dim: float = 0.0,
    virtual_lights: Sequence[dict[str, Any]] = (),
) -> np.ndarray:
    """
    Albedo-preserving linear-light 3D shader supporting:
    - Dim Original Light ('Old Light' suppression) in linear space
    - Near / Far fill light and warmth shifts
    - Up to 3 virtual 3D Point / Spot / Rim lights with 3D surface normals,
      Blinn-Phong specular gloss, depth-band masking, and Gobo projections.
    """
    h, w = img_rgb.shape[:2]
    lin = _SRGB_TO_LIN_LUT[img_rgb]
    lum = (0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]).astype(np.float32)

    # 1. Albedo-preserving ambient dimming (0% Original light -> 97% darker for dramatic studio look)
    dim_t = float(np.clip(float(ambient_dim) / 100.0, 0.0, 1.0))
    ambient_gain = ((1.0 - 0.97 * dim_t) * (1.0 - 0.12 * dim_t * np.clip(lum, 0.0, 1.0))).astype(np.float32)

    # 2. Near / Far linear fill & warmth
    ev_shift = (float(relight_near) * 0.014) * near_weight + (float(relight_far) * 0.014) * far_weight
    near_far_gain = np.power(2.0, np.clip(ev_shift, -2.0, 2.0)).astype(np.float32)

    base_gain = (ambient_gain * near_far_gain)[:, :, None]
    w_t = np.clip(((float(warmth_near) * near_weight + float(warmth_far) * far_weight) / 100.0) * 0.35, -0.45, 0.45)
    warmth_rgb = np.stack([1.0 + w_t, 1.0 + 0.15 * w_t, 1.0 - w_t], axis=-1).astype(np.float32)

    # 3. Multi-Light 3D Shader (decoupled <=640px lighting field for 3-5x speedup, applied to sharp albedo)
    active_lights = [
        lt for lt in list(virtual_lights)[:3]
        if isinstance(lt, dict) and bool(lt.get("enabled", True)) and float(lt.get("intensity", 0.0)) >= 0.1
    ]

    if not active_lights:
        out_lin = lin * (base_gain * warmth_rgb)
    else:
        max_side = max(h, w)
        if max_side > 640:
            l_scale = 640.0 / float(max_side)
            lw = max(24, int(round(w * l_scale)))
            lh = max(24, int(round(h * l_scale)))
            work_depth = cv2.resize(refined_depth, (lw, lh), interpolation=cv2.INTER_AREA).astype(np.float32)
            work_matte = cv2.resize(alpha_matte, (lw, lh), interpolation=cv2.INTER_AREA).astype(np.float32)
            work_normals = cv2.resize(normal_map, (lw, lh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            w_nrm = np.linalg.norm(work_normals, axis=2, keepdims=True)
            work_normals = (work_normals / np.maximum(w_nrm, 1e-6)).astype(np.float32)
            work_lum = cv2.resize(lum, (lw, lh), interpolation=cv2.INTER_AREA).astype(np.float32)
        else:
            lw, lh = w, h
            work_depth = refined_depth
            work_matte = alpha_matte
            work_normals = normal_map
            work_lum = lum

        diffuse_field = np.zeros((lh, lw, 3), dtype=np.float32)
        spec_field = np.zeros((lh, lw, 3), dtype=np.float32)

        aspect = float(lw) / float(max(1, lh))
        u, v, u_norm, v_norm = _get_cached_uv_grids(lw, lh)
        pz = (-work_depth).astype(np.float32)

        nx = work_normals[:, :, 0]
        ny = work_normals[:, :, 1]
        nz = work_normals[:, :, 2]

        blur_a = cv2.GaussianBlur(work_matte, (0, 0), sigmaX=max(2.0, min(lw, lh) * 0.015))
        inner_rim = np.clip(work_matte * np.maximum(0.0, 1.0 - blur_a) * 3.8, 0.0, 1.35).astype(np.float32)

        for lt in active_lights:
            ltype = str(lt.get("type", "point")).lower()
            lx_n = float(lt.get("x", 0.25))
            ly_n = float(lt.get("y", 0.25))
            lx = lx_n * aspect
            ly = ly_n
            lz_norm = float(np.clip(float(lt.get("z", 0.25)), 0.0, 1.0))
            lz = -lz_norm + 0.22
            aim_x = float(lt.get("aim_x", 0.5)) * aspect
            aim_y = float(lt.get("aim_y", 0.5))
            softness = float(np.clip(float(lt.get("softness", 45.0)), 0.0, 100.0))
            soft_t = softness / 100.0
            intensity = float(np.clip(float(lt.get("intensity", 70.0)), 0.0, 100.0))
            gloss = float(np.clip(float(lt.get("gloss", 0.0)), 0.0, 100.0))
            cone_deg = float(np.clip(float(lt.get("cone_angle", 40.0)), 5.0, 120.0))
            d_from = float(np.clip(float(lt.get("depth_from", 0.0)), 0.0, 1.0))
            d_to = float(np.clip(float(lt.get("depth_to", 1.0)), 0.0, 1.0))
            if d_to < d_from:
                d_from, d_to = d_to, d_from

            dx = lx - u
            dy = ly - v
            dz = lz - pz
            dist_sq = dx * dx + dy * dy + dz * dz + 1e-6
            dist = np.sqrt(dist_sq)
            inv_dist = 1.0 / dist
            ldx = dx * inv_dist
            ldy = dy * inv_dist
            ldz = dz * inv_dist

            radius = 0.45 + 1.20 * soft_t
            atten = (1.0 / (1.0 + dist_sq / (radius * radius))).astype(np.float32)

            if ltype == "spot":
                aim_col = max(0, min(lw - 1, int(round(float(lt.get("aim_x", 0.5)) * (lw - 1)))))
                aim_row = max(0, min(lh - 1, int(round(float(lt.get("aim_y", 0.5)) * (lh - 1)))))
                aim_pz = -float(work_depth[aim_row, aim_col])
                adx = aim_x - lx
                ady = aim_y - ly
                adz = aim_pz - lz
                if abs(adx) < 1e-3 and abs(ady) < 1e-3 and abs(adz) < 0.05:
                    adz = -0.35
                anorm = math.sqrt(adx * adx + ady * ady + adz * adz + 1e-6)
                ax, ay, az = adx / anorm, ady / anorm, adz / anorm
                spot_cos = -(ldx * ax + ldy * ay + ldz * az)
                inner_cos = math.cos(math.radians(cone_deg * 0.5))
                outer_cos = math.cos(math.radians(min(89.0, cone_deg * (0.55 + 0.55 * soft_t))))
                denom = max(1e-4, inner_cos - outer_cos)
                cone_mask = np.clip((spot_cos - outer_cos) / denom, 0.0, 1.0)
                cone_mask = cone_mask * cone_mask * (3.0 - 2.0 * cone_mask)
                type_weight = (atten * cone_mask * 1.35).astype(np.float32)
            elif ltype == "rim":
                silhouette = 4.0 * work_matte * (1.0 - work_matte)
                grazing = np.power(np.clip(1.0 - nz, 0.0, 1.0), 1.25) + 0.75 * silhouette + inner_rim
                side_align = np.clip(nx * ldx + ny * ldy + 0.45, 0.0, 1.0)
                type_weight = (atten * np.clip(grazing, 0.0, 1.6) * side_align * 1.45).astype(np.float32)
            else:
                type_weight = atten

            # 3-step depth-aware occlusion shadowing for Point / Spot lights
            if ltype in ("point", "spot"):
                du_n = lx_n - u_norm
                dv_n = ly_n - v_norm
                d2d = np.sqrt(du_n * du_n + dv_n * dv_n + 1e-6)
                dir_u = du_n / d2d
                dir_v = dv_n / d2d
                occluded = np.zeros((lh, lw), dtype=np.float32)
                for step_d in (0.06, 0.14, 0.24):
                    t_frac = np.minimum(1.0, step_d / d2d)
                    su = np.clip(((u_norm + dir_u * np.minimum(step_d, d2d)) * (lw - 1)).astype(np.int32), 0, lw - 1)
                    sv = np.clip(((v_norm + dir_v * np.minimum(step_d, d2d)) * (lh - 1)).astype(np.int32), 0, lh - 1)
                    sample_d = work_depth[sv, su]
                    ray_d = work_depth + t_frac * (lz_norm - work_depth)
                    occluded = np.maximum(occluded, np.clip((ray_d - sample_d - 0.07) / 0.12, 0.0, 1.0))
                type_weight = (type_weight * (1.0 - occluded * (0.62 * (1.0 - 0.55 * soft_t)))).astype(np.float32)

            wrap = 0.15 + 0.65 * soft_t
            ndotl = np.clip((nx * ldx + ny * ldy + nz * ldz + wrap) / (1.0 + wrap), 0.0, 1.0)

            if d_from > 0.001 or d_to < 0.999:
                f_in = np.clip((work_depth - d_from + 0.06) / 0.06, 0.0, 1.0)
                f_out = np.clip((d_to + 0.06 - work_depth) / 0.06, 0.0, 1.0)
                depth_band = np.minimum(f_in, f_out).astype(np.float32)
            else:
                depth_band = 1.0

            gobo_mask = _compute_gobo_mask(
                str(lt.get("gobo", "none")), u, v, lx, ly, aim_x, aim_y, softness
            )

            gain_scalar = (intensity / 100.0) * 2.2
            l_col = light_color_linear(lt)[None, None, :]
            eff = (type_weight * depth_band * gobo_mask * gain_scalar)[:, :, None]

            diffuse_field += eff * ndotl[:, :, None] * l_col

            if gloss >= 0.5:
                hx = ldx
                hy = ldy
                hz = ldz + 1.0
                h_inv = 1.0 / np.sqrt(hx * hx + hy * hy + hz * hz + 1e-6)
                ndoth = np.clip((nx * hx + ny * hy + nz * hz) * h_inv, 0.0, 1.0)
                shininess = 12.0 + 36.0 * (1.0 - soft_t)
                spec_term = (
                    (gloss / 100.0)
                    * 0.85
                    * np.power(ndoth, shininess)
                    * np.clip(work_lum * 2.2, 0.05, 1.0)
                )
                spec_field += eff * spec_term[:, :, None] * l_col

        if (lw, lh) != (w, h):
            diffuse_acc = cv2.resize(diffuse_field, (w, h), interpolation=cv2.INTER_LINEAR)
            spec_acc = cv2.resize(spec_field, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            diffuse_acc = diffuse_field
            spec_acc = spec_field

        out_lin = lin * (base_gain * warmth_rgb + diffuse_acc) + spec_acc

    # Chroma-preserving luminance shoulder above 0.85
    y_out = 0.2126 * out_lin[:, :, 0] + 0.7152 * out_lin[:, :, 1] + 0.0722 * out_lin[:, :, 2]
    excess = np.maximum(y_out - 0.85, 0.0)
    y_comp = np.where(
        y_out > 0.85,
        0.85 + excess / (1.0 + excess * 1.45),
        y_out,
    )
    chroma_scale = np.where(y_out > 1e-5, y_comp / np.maximum(y_out, 1e-5), 1.0)[:, :, None]
    out_lin = out_lin * chroma_scale

    idx = np.clip(np.round(out_lin * 4095.0).astype(np.int32), 0, 4095)
    return _LIN_TO_SRGB_LUT[idx]


# =============================================================================
# 3. 3D Cinematic Motion (Continuous Depth-Map Mesh Warping via cv2.remap)
# =============================================================================

def compute_target_dimensions(w: int, h: int, max_long_edge: int = 1920) -> tuple[int, int]:
    """
    Computes dimensions preserving exact aspect ratio with even width & height
    required by H.264 / mp4v video encoders.
    """
    scale = min(1.0, float(max_long_edge) / max(w, h))
    tw = int(round(w * scale))
    th = int(round(h * scale))
    # Must be even integers for video encoders
    tw = max(16, tw & ~1)
    th = max(16, th & ~1)
    return (tw, th)


def inpaint_disocclusion_background(
    image_bgr: np.ndarray,
    subject_mask: np.ndarray,
) -> np.ndarray:
    """
    Inpaints the background behind the subject so parallax virtual camera moves
    reveal realistic background content rather than black holes.
    Uses neural LaMa when available, otherwise falls back to fast Telea inpainting.
    """
    h, w = image_bgr.shape[:2]
    bin_mask = (subject_mask > 0.25).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated_mask = cv2.dilate(bin_mask, kernel)

    engine = get_lama_inpaint_engine()
    if engine.has_neural_model() and np.any(dilated_mask > 0):
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inpainted_rgb = inpaint_object_lama(img_rgb, dilated_mask)
        return cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)

    scale = max(1.0, max(h, w) / 600.0)
    if scale > 1.2:
        sw, sh = int(w / scale), int(h / scale)
        small_img = cv2.resize(image_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(dilated_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
        inpainted_small = cv2.inpaint(small_img, small_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return cv2.resize(inpainted_small, (w, h), interpolation=cv2.INTER_LINEAR)

    return cv2.inpaint(image_bgr, dilated_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


_CINEMATIC_SRGB_TO_LIN: np.ndarray = ((np.arange(256, dtype=np.float32) / 255.0) ** 2.2).astype(np.float32)
_CINEMATIC_LIN_TO_SRGB: np.ndarray = np.clip(
    (np.linspace(0.0, 1.0, 4096, dtype=np.float32) ** (1.0 / 2.2)) * 255.0 + 0.5, 0, 255
).astype(np.uint8)


def _decontaminate_foreground_edges(bgr: np.ndarray, alpha_matte: np.ndarray) -> np.ndarray:
    """
    Extends solid foreground colors slightly outward into semi-transparent matte edges
    (hair, shoulders) so background colors do not bleed into the moving foreground cutout.
    """
    core_w = np.clip((alpha_matte - 0.55) / 0.35, 0.0, 1.0).astype(np.float32)
    rim_mask = (alpha_matte > 0.02) & (alpha_matte < 0.92)
    if not np.any(rim_mask) or not np.any(core_w > 0.1):
        return bgr

    h, w = bgr.shape[:2]
    scale = max(1.0, max(h, w) / 720.0)
    k = max(5, int(round(9 * scale)) | 1)
    sigma = max(1.5, 3.0 * scale)

    fg_f = bgr.astype(np.float32)
    premult = fg_f * core_w[:, :, None]
    num = cv2.GaussianBlur(premult, (k, k), sigmaX=sigma, sigmaY=sigma)
    den = cv2.GaussianBlur(core_w, (k, k), sigmaX=sigma, sigmaY=sigma)[:, :, None]
    valid = den[:, :, 0] > 1e-3
    ext_fg = fg_f.copy()
    ext_fg[valid] = num[valid] / den[valid]

    # Blend extended foreground color into the soft matte rim only
    rim_blend = np.where(rim_mask, np.clip(1.0 - core_w, 0.0, 1.0), 0.0)[:, :, None].astype(np.float32)
    out_f = fg_f * (1.0 - rim_blend) + ext_fg * rim_blend
    np.clip(out_f, 0.0, 255.0, out=out_f)
    return out_f.astype(np.uint8)


def _build_cinematic_bokeh_plate(
    bg_canvas: np.ndarray,
    depth_bg: np.ndarray,
    focal_plane: float = 0.5,
) -> np.ndarray:
    """
    Precomputes a linear-light optical bokeh background plate weighted by distance from focal_plane.
    """
    h_img, w_img = bg_canvas.shape[:2]
    focus_z = float(np.clip(focal_plane, 0.0, 1.0))
    defocus = np.clip(np.abs(depth_bg - focus_z) * 1.35, 0.0, 1.0)

    # Blur in linear light on a <=640px buffer for photographic highlight bloom & speed
    scale_blur = max(1.0, max(h_img, w_img) / 640.0)
    bw = max(16, int(round(w_img / scale_blur)))
    bh = max(16, int(round(h_img / scale_blur)))
    small_bg = cv2.resize(bg_canvas, (bw, bh), interpolation=cv2.INTER_AREA) if scale_blur > 1.05 else bg_canvas
    small_lin = _CINEMATIC_SRGB_TO_LIN[small_bg]

    # Subtle specular highlight boost in linear space
    lum = cv2.cvtColor(small_bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    glow = np.clip((lum - 195.0) / 60.0, 0.0, 1.0)[:, :, None]
    small_lin = small_lin * (1.0 + glow * 0.6)

    sigma_b = max(2.0, 6.0 / scale_blur) if scale_blur > 1.05 else 6.0
    blurred_lin = cv2.GaussianBlur(small_lin, (0, 0), sigmaX=sigma_b, sigmaY=sigma_b)
    indices = np.clip(blurred_lin * 4095.0, 0, 4095).astype(np.int32)
    small_srgb = _CINEMATIC_LIN_TO_SRGB[indices]
    blurred_srgb = (
        cv2.resize(small_srgb, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        if scale_blur > 1.05
        else small_srgb
    )

    w_blend = (defocus * 0.72)[:, :, None]
    bg_f = bg_canvas.astype(np.float32)
    blur_f = blurred_srgb.astype(np.float32)
    blur_f -= bg_f
    blur_f *= w_blend
    bg_f += blur_f
    np.clip(bg_f, 0.0, 255.0, out=bg_f)
    return bg_f.astype(np.uint8)


def prepare_3d_parallax_layers(
    bgr_array: np.ndarray,
    face_rects: list[tuple[int, int, int, int]] | None = None,
) -> dict[str, np.ndarray]:
    """
    Precomputes hybrid 2.5D layered depth-mesh parallax layers:
    1. Obtains (depth_map, alpha_matte) via DepthAndMatteEngine (Depth-Anything-V2 + MODNet with fallback),
       fusing near-depth foreground cues when MODNet has low coverage (non-portrait scenes).
    2. Decontaminates soft foreground edges so background colors do not bleed into moving cutouts.
    3. Constructs a clean background plate (bg_plate) by dilating the foreground mask
       (alpha_matte > 0.15) and inpainting disocclusions at up to 1024px resolution.
    4. Inpaints and smooths the background depth map (depth_bg) across the disocclusion zone.
    """
    if bgr_array is None or bgr_array.size == 0:
        empty = np.zeros((10, 10, 3), dtype=np.uint8)
        empty_f = np.zeros((10, 10), dtype=np.float32)
        return {
            "fg_image": empty,
            "bg_plate": empty,
            "depth_map": empty_f,
            "depth_bg": empty_f,
            "alpha_matte": empty_f,
        }

    h, w = bgr_array.shape[:2]
    rgb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    depth_engine = get_depth_and_matte_engine()
    depth_map, alpha_matte = depth_engine.get_depth_and_matte(rgb)

    # For non-portrait scenes where MODNet finds < 2% foreground, fuse near-depth foreground cues
    if float(np.mean(alpha_matte > 0.25)) < 0.02:
        depth_fg_cue = np.clip((0.40 - depth_map) / 0.18, 0.0, 1.0).astype(np.float32)
        alpha_matte = np.maximum(alpha_matte, depth_fg_cue)

    # Decontaminate foreground rim colors before warping
    fg_clean = _decontaminate_foreground_edges(bgr_array, alpha_matte)

    # Clean foreground mask (threshold at 0.15)
    inpaint_scale = max(1.0, max(h, w) / 1024.0)
    res_scale = max(1.0, max(h, w) / 600.0)
    dilate_px = max(5, int(14 * res_scale))
    ksize = dilate_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bin_mask = (alpha_matte > 0.15).astype(np.uint8) * 255
    dilated_mask = cv2.dilate(bin_mask, kernel)

    # Inpaint background plate (neural LaMa if available, otherwise Telea) and background depth map
    lama_engine = get_lama_inpaint_engine()
    if lama_engine.has_neural_model() and np.any(dilated_mask > 0):
        bgr_rgb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        bg_plate = cv2.cvtColor(inpaint_object_lama(bgr_rgb, dilated_mask), cv2.COLOR_RGB2BGR)
        depth_u8 = (np.clip(depth_map, 0.0, 1.0) * 255.0).astype(np.uint8)
        if inpaint_scale > 1.15:
            sw, sh = max(16, int(w / inpaint_scale)), max(16, int(h / inpaint_scale))
            small_mask = cv2.resize(dilated_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
            small_depth = cv2.resize(depth_u8, (sw, sh), interpolation=cv2.INTER_AREA)
            inpainted_depth_small = cv2.inpaint(small_depth, small_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            depth_bg_u8 = cv2.resize(inpainted_depth_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            depth_bg_u8 = cv2.inpaint(depth_u8, dilated_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        depth_bg = depth_bg_u8.astype(np.float32) / 255.0
    elif inpaint_scale > 1.15:
        sw, sh = max(16, int(w / inpaint_scale)), max(16, int(h / inpaint_scale))
        small_img = cv2.resize(bgr_array, (sw, sh), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(dilated_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
        inpainted_small = cv2.inpaint(small_img, small_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        bg_plate = cv2.resize(inpainted_small, (w, h), interpolation=cv2.INTER_LINEAR)

        depth_u8 = (np.clip(depth_map, 0.0, 1.0) * 255.0).astype(np.uint8)
        small_depth = cv2.resize(depth_u8, (sw, sh), interpolation=cv2.INTER_AREA)
        inpainted_depth_small = cv2.inpaint(small_depth, small_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        depth_bg_u8 = cv2.resize(inpainted_depth_small, (w, h), interpolation=cv2.INTER_LINEAR)
        depth_bg = depth_bg_u8.astype(np.float32) / 255.0
    else:
        bg_plate = cv2.inpaint(bgr_array, dilated_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        depth_u8 = (np.clip(depth_map, 0.0, 1.0) * 255.0).astype(np.uint8)
        depth_bg_u8 = cv2.inpaint(depth_u8, dilated_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        depth_bg = depth_bg_u8.astype(np.float32) / 255.0

    # Ensure depth_bg is smoothly continuous behind the subject
    blur_k = max(7, (int(9 * res_scale) // 2) * 2 + 1)
    blurred_depth = cv2.GaussianBlur(depth_bg, (blur_k, blur_k), 0)
    blend_mask = (dilated_mask.astype(np.float32) / 255.0)
    depth_bg = depth_bg * (1.0 - blend_mask) + blurred_depth * blend_mask

    return {
        "fg_image": fg_clean,
        "bg_plate": bg_plate,
        "depth_map": depth_map,
        "depth_bg": depth_bg,
        "alpha_matte": alpha_matte,
    }


def render_3d_cinematic_motion_frame(
    foreground_bgr: np.ndarray | dict[str, np.ndarray],
    background_bgr: np.ndarray | None = None,
    subject_mask: np.ndarray | None = None,
    t: float = 0.0,
    canvas_size: tuple[int, int] | None = None,
    depth_map: np.ndarray | None = None,
    motion_style: str = "orbital",
    intensity: float = 0.5,
    focal_plane: float = 0.5,
    enable_bokeh: bool = True,
    depth_bg: np.ndarray | None = None,
    overscan_scale: float = 1.04,
) -> np.ndarray:
    """
    Renders a single frame of continuous 3D cinematic virtual camera parallax motion:
    - 2.5D Layered separation: Background plate and foreground subject are warped independently,
      completely eliminating single-mesh rubber edge stretching artifacts.
    - Default 1.04x (4%) overscan prevents border reflection artifacts at the outer edges.
    - motion_style: 'orbital' / '3d orbit', 'dolly_zoom' / 'dolly_in', 'parallax_drift' / 'pan_lr',
      'cinematic_pan', 'breathe'.
    - intensity: parallax displacement amplitude in [0.1, 1.0].
    - enable_bokeh: applies subtle optical disc blur to background plate proportional to depth_bg.
    """
    if isinstance(foreground_bgr, dict):
        fg_src = foreground_bgr["fg_image"]
        bg_src = foreground_bgr.get("bg_plate", fg_src)
        depth_fg = foreground_bgr.get("depth_map")
        depth_bg_arr = foreground_bgr.get("depth_bg", depth_fg)
        alpha_matte = foreground_bgr.get("alpha_matte")
    else:
        fg_src = foreground_bgr
        bg_src = background_bgr if background_bgr is not None else foreground_bgr
        depth_fg = depth_map
        depth_bg_arr = depth_bg if depth_bg is not None else depth_map
        alpha_matte = subject_mask

    h, w = fg_src.shape[:2]
    if canvas_size is None:
        tw, th = w, h
    else:
        tw, th = canvas_size

    # Ensure masks & depth exist
    if alpha_matte is None:
        alpha_matte = np.ones((h, w), dtype=np.float32)

    if depth_fg is None:
        depth_fg = estimate_depth_map(fg_src, subject_mask=alpha_matte)
    if depth_bg_arr is None:
        depth_bg_arr = depth_fg.copy()

    intensity = max(0.05, min(1.0, float(intensity)))
    max_disp = float(min(tw, th)) * 0.045 * intensity
    cx, cy = tw * 0.5, th * 0.5

    # Cache resized canvas layers & meshgrid inside dict when rendering multi-frame videos/previews
    layers_dict = foreground_bgr if isinstance(foreground_bgr, dict) else None
    canvas_cache = layers_dict.get("_canvas_cache") if layers_dict is not None else None
    if canvas_cache is not None and canvas_cache[0] == (tw, th):
        _, depth_fg, depth_bg_arr, fg_canvas, bg_canvas, alpha_canvas, xs, ys = canvas_cache
    else:
        if depth_fg.shape[:2] != (th, tw):
            depth_fg = cv2.resize(depth_fg, (tw, th), interpolation=cv2.INTER_LINEAR)
        if depth_bg_arr.shape[:2] != (th, tw):
            depth_bg_arr = cv2.resize(depth_bg_arr, (tw, th), interpolation=cv2.INTER_LINEAR)
        if fg_src.shape[:2] != (th, tw):
            fg_canvas = cv2.resize(fg_src, (tw, th), interpolation=cv2.INTER_LINEAR)
        else:
            fg_canvas = fg_src
        if bg_src.shape[:2] != (th, tw):
            bg_canvas = cv2.resize(bg_src, (tw, th), interpolation=cv2.INTER_LINEAR)
        else:
            bg_canvas = bg_src
        if alpha_matte.shape[:2] != (th, tw):
            alpha_canvas = cv2.resize(alpha_matte, (tw, th), interpolation=cv2.INTER_LINEAR)
        else:
            alpha_canvas = alpha_matte
        xs, ys = np.meshgrid(np.arange(tw, dtype=np.float32), np.arange(th, dtype=np.float32))
        if layers_dict is not None:
            layers_dict["_canvas_cache"] = (
                (tw, th),
                depth_fg,
                depth_bg_arr,
                fg_canvas,
                bg_canvas,
                alpha_canvas,
                xs,
                ys,
            )

    # Dynamic overscan eliminates outer border reflection artifacts even at intensity=1.0
    overscan = max(float(overscan_scale), 1.0 + 0.065 * intensity)
    base_x = cx + (xs - cx) / overscan
    base_y = cy + (ys - cy) / overscan

    # Relative depth displacement: near foreground (0.0) -> +0.5, far background (1.0) -> -0.5
    rel_fg = 0.5 - depth_fg
    rel_bg = 0.5 - depth_bg_arr

    m_style = motion_style.lower()
    if "dolly" in m_style or "vertigo" in m_style:
        # Hitchcock Vertigo / Dolly Zoom: Foreground stays steady, background zooms in/out
        cycle_t = 0.5 - 0.5 * math.cos(t * 2.0 * math.pi)
        fg_zoom = 1.0 + 0.015 * cycle_t * intensity
        bg_zoom = 1.0 + 0.08 * cycle_t * intensity
        cam_x = math.sin(t * 2.0 * math.pi) * (max_disp * 0.3)
        cam_y = math.sin(t * 2.0 * math.pi) * (max_disp * 0.15)
        map_bg_x = cx + (base_x - cx) / bg_zoom + cam_x * rel_bg
        map_bg_y = cy + (base_y - cy) / bg_zoom + cam_y * rel_bg
        map_fg_x = cx + (base_x - cx) / fg_zoom + cam_x * rel_fg
        map_fg_y = cy + (base_y - cy) / fg_zoom + cam_y * rel_fg

    elif "drift" in m_style or "pan_lr" in m_style:
        # Smooth horizontal / diagonal parallax drift
        cam_x = math.sin(t * 2.0 * math.pi) * max_disp
        cam_y = math.sin(t * 2.0 * math.pi) * (max_disp * 0.25)
        map_bg_x = base_x + cam_x * rel_bg
        map_bg_y = base_y + cam_y * rel_bg
        map_fg_x = base_x + cam_x * rel_fg
        map_fg_y = base_y + cam_y * rel_fg

    elif "cinematic" in m_style or "pan" in m_style:
        # Sweeping cinematic pan with subtle vertical arc
        cam_x = math.sin(t * 2.0 * math.pi) * (max_disp * 1.3)
        cam_y = (1.0 - math.cos(t * 2.0 * math.pi)) * (max_disp * 0.22)
        zoom = 1.0 + 0.015 * (1.0 - math.cos(t * 2.0 * math.pi)) * intensity
        map_bg_x = cx + (base_x - cx) / zoom + cam_x * rel_bg
        map_bg_y = cy + (base_y - cy) / zoom + cam_y * rel_bg
        map_fg_x = cx + (base_x - cx) / zoom + cam_x * rel_fg
        map_fg_y = cy + (base_y - cy) / zoom + cam_y * rel_fg

    elif "breathe" in m_style:
        # Gentle floating breathe
        cam_y = math.sin(t * 2.0 * math.pi) * (max_disp * 0.5)
        zoom = 1.0 + 0.02 * math.sin(t * 2.0 * math.pi) * intensity
        map_bg_x = cx + (base_x - cx) / zoom
        map_bg_y = cy + (base_y - cy) / zoom + cam_y * rel_bg
        map_fg_x = cx + (base_x - cx) / zoom
        map_fg_y = cy + (base_y - cy) / zoom + cam_y * rel_fg

    else:  # "3d orbit" / "orbital" (default)
        cam_x = math.sin(t * 2.0 * math.pi) * max_disp
        cam_y = math.cos(t * 2.0 * math.pi) * (max_disp * 0.45)
        zoom = 1.0 + 0.025 * (0.5 - 0.5 * math.cos(t * 2.0 * math.pi)) * intensity
        map_bg_x = cx + (base_x - cx) / zoom + cam_x * rel_bg
        map_bg_y = cy + (base_y - cy) / zoom + cam_y * rel_bg
        map_fg_x = cx + (base_x - cx) / zoom + cam_x * rel_fg
        map_fg_y = cy + (base_y - cy) / zoom + cam_y * rel_fg

    # Linear-light depth-weighted optical blur on background plate (precomputed once per focal_plane)
    if enable_bokeh:
        bokeh_key = (tw, th, round(float(focal_plane), 3))
        bokeh_cache = layers_dict.get("_bokeh_cache") if layers_dict is not None else None
        if bokeh_cache is not None and bokeh_cache[0] == bokeh_key:
            bg_to_warp = bokeh_cache[1]
        else:
            bg_to_warp = _build_cinematic_bokeh_plate(bg_canvas, depth_bg_arr, focal_plane=focal_plane)
            if layers_dict is not None:
                layers_dict["_bokeh_cache"] = (bokeh_key, bg_to_warp)
    else:
        bg_to_warp = bg_canvas

    # Warp background plate and foreground subject & matte independently (INTER_CUBIC keeps subject sharp)
    warped_bg = cv2.remap(bg_to_warp, map_bg_x, map_bg_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    warped_fg = cv2.remap(fg_canvas, map_fg_x, map_fg_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    warped_alpha = cv2.remap(alpha_canvas, map_fg_x, map_fg_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    m3 = np.clip(warped_alpha, 0.0, 1.0)[..., None]
    fg_f = warped_fg.astype(np.float32)
    bg_f = warped_bg.astype(np.float32)
    fg_f *= m3
    bg_f *= (1.0 - m3)
    fg_f += bg_f
    np.clip(fg_f, 0.0, 255.0, out=fg_f)
    return fg_f.astype(np.uint8)


def render_3d_cinematic_motion_video(
    image_bgr: np.ndarray,
    output_path: Path | str,
    duration_sec: float = 4.0,
    fps: int = 30,
    resolution: tuple[int, int] | None = None,
    subject_mask: np.ndarray | None = None,
    depth_map: np.ndarray | None = None,
    motion_style: str = "orbital",
    intensity: float = 0.5,
    focal_plane: float = 0.5,
    enable_bokeh: bool = True,
    progress_callback: Any | None = None,
) -> bool:
    """
    Renders a 3D Cinematic Motion .mp4 video from a single photo preserving native aspect ratio.
    """
    if image_bgr is None or image_bgr.size == 0:
        return False

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    h, w = image_bgr.shape[:2]
    if resolution is None:
        target_size = compute_target_dimensions(w, h, max_long_edge=1920)
    else:
        target_size = (resolution[0] & ~1, resolution[1] & ~1)

    tw, th = target_size

    # Precompute hybrid 2.5D parallax layers once
    layers = prepare_3d_parallax_layers(image_bgr)
    if subject_mask is not None:
        layers["alpha_matte"] = subject_mask
    if depth_map is not None:
        layers["depth_map"] = depth_map

    total_frames = int(duration_sec * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_p), fourcc, float(fps), (tw, th))

    if not writer.isOpened():
        return False

    try:
        for frame_idx in range(total_frames):
            t = frame_idx / float(max(1, total_frames - 1))
            frame = render_3d_cinematic_motion_frame(
                foreground_bgr=layers,
                t=t,
                canvas_size=(tw, th),
                motion_style=motion_style,
                intensity=intensity,
                focal_plane=focal_plane,
                enable_bokeh=enable_bokeh,
            )
            writer.write(frame)
            if progress_callback:
                progress_callback(int((frame_idx + 1) * 100 / total_frames))
    finally:
        writer.release()

    return out_p.exists() and out_p.stat().st_size > 1000


def render_3d_cinematic_motion_reel(
    images_bgr: list[np.ndarray],
    output_path: Path | str,
    duration_per_clip: float = 4.0,
    transition_sec: float = 0.8,
    fps: int = 30,
    resolution: tuple[int, int] | None = None,
    motion_style: str = "orbital",
    intensity: float = 0.5,
    enable_bokeh: bool = True,
    progress_callback: Any | None = None,
) -> bool:
    """
    Renders a multi-photo 3D Memories Reel with smooth 0.8s cross-fades between photos.
    """
    valid_images = [img for img in images_bgr if img is not None and img.size > 0]
    if not valid_images:
        return False

    if len(valid_images) == 1:
        return render_3d_cinematic_motion_video(
            valid_images[0],
            output_path=output_path,
            duration_sec=duration_per_clip,
            fps=fps,
            resolution=resolution,
            motion_style=motion_style,
            intensity=intensity,
            enable_bokeh=enable_bokeh,
            progress_callback=progress_callback,
        )

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Base target canvas resolution on first image
    h0, w0 = valid_images[0].shape[:2]
    if resolution is None:
        tw, th = compute_target_dimensions(w0, h0, max_long_edge=1920)
    else:
        tw, th = (resolution[0] & ~1, resolution[1] & ~1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_p), fourcc, float(fps), (tw, th))
    if not writer.isOpened():
        return False

    clip_frames_count = max(30, int(duration_per_clip * fps))
    trans_frames = max(6, int(transition_sec * fps))

    # Pre-render or stream each photo clip
    styles = [motion_style, "cinematic_pan", "dolly_zoom", "3d_orbit"]
    num_clips = len(valid_images)
    total_steps = num_clips * clip_frames_count
    current_step = 0

    try:
        prev_tail_frames: list[np.ndarray] = []

        for clip_idx, img in enumerate(valid_images):
            style = styles[clip_idx % len(styles)] if motion_style == "orbital" else motion_style
            layers = prepare_3d_parallax_layers(img)

            clip_frames: list[np.ndarray] = []
            for f_idx in range(clip_frames_count):
                t = f_idx / float(max(1, clip_frames_count - 1))
                frame = render_3d_cinematic_motion_frame(
                    foreground_bgr=layers,
                    t=t,
                    canvas_size=(tw, th),
                    motion_style=style,
                    intensity=intensity,
                    enable_bokeh=enable_bokeh,
                )
                clip_frames.append(frame)

            # If not first clip, write cross-fade transition
            if clip_idx > 0 and prev_tail_frames:
                for step in range(trans_frames):
                    alpha = (step + 1) / float(trans_frames + 1)
                    blend = cv2.addWeighted(
                        prev_tail_frames[step], 1.0 - alpha,
                        clip_frames[step], alpha, 0.0
                    )
                    writer.write(blend)
                    current_step += 1
                    if progress_callback:
                        progress_callback(min(99, int(current_step * 100 / total_steps)))

                # Write remaining frames of this clip
                for frame in clip_frames[trans_frames:-trans_frames if clip_idx < num_clips - 1 else None]:
                    writer.write(frame)
                    current_step += 1
                    if progress_callback:
                        progress_callback(min(99, int(current_step * 100 / total_steps)))
            else:
                # First clip: write up to tail
                for frame in clip_frames[:-trans_frames]:
                    writer.write(frame)
                    current_step += 1
                    if progress_callback:
                        progress_callback(min(99, int(current_step * 100 / total_steps)))

            prev_tail_frames = clip_frames[-trans_frames:]

        if progress_callback:
            progress_callback(100)
    finally:
        writer.release()

    return out_p.exists() and out_p.stat().st_size > 1000


# =============================================================================
# 4. Privacy Redaction (Mosaic Pixelate & Gaussian Blur)
# =============================================================================

def apply_pixelate(
    image_bgr: np.ndarray,
    rect: tuple[int, int, int, int],
    block_size: int = 16,
) -> np.ndarray:
    """Applies heavy privacy mosaic pixelation over rect (x, y, w, h)."""
    if image_bgr is None:
        return image_bgr

    h, w = image_bgr.shape[:2]
    rx, ry, rw, rh = rect
    x1, y1 = max(0, rx), max(0, ry)
    x2, y2 = min(w, rx + rw), min(h, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return image_bgr

    result = image_bgr.copy()
    roi = result[y1:y2, x1:x2]
    bs = max(4, block_size)

    small_w = max(1, (x2 - x1) // bs)
    small_h = max(1, (y2 - y1) // bs)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    result[y1:y2, x1:x2] = pixelated
    return result


def apply_blur_redact(
    image_bgr: np.ndarray,
    rect: tuple[int, int, int, int],
    radius: int = 25,
) -> np.ndarray:
    """Applies heavy Gaussian privacy blur over rect (x, y, w, h)."""
    if image_bgr is None:
        return image_bgr

    h, w = image_bgr.shape[:2]
    rx, ry, rw, rh = rect
    x1, y1 = max(0, rx), max(0, ry)
    x2, y2 = min(w, rx + rw), min(h, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return image_bgr

    result = image_bgr.copy()
    roi = result[y1:y2, x1:x2]
    ksize = max(3, radius * 2 + 1)
    if ksize % 2 == 0:
        ksize += 1

    blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
    result[y1:y2, x1:x2] = blurred
    return result


# =============================================================================
# 5. 1-Click Subject Cutout & Background Removal
# =============================================================================

def apply_boundary_shift_and_feather(
    alpha_matte: np.ndarray,
    boundary_shift: float = 0.0,
    boundary_feather: float = 0.0,
) -> np.ndarray:
    """
    Applies morphological boundary shift (grow/shrink in [-10..+10]) and
    Gaussian boundary feathering ([0..20]) to a float32 [0..1] alpha matte.
    """
    if alpha_matte is None or alpha_matte.size == 0:
        return alpha_matte
    h, w = alpha_matte.shape[:2]
    res_scale = max(0.5, max(w, h) / 1000.0)
    out = np.clip(alpha_matte.astype(np.float32), 0.0, 1.0)

    if abs(boundary_shift) > 0.05:
        shift_px = max(1, int(round(abs(boundary_shift) * 1.5 * res_scale)))
        k_shift = shift_px * 2 + 1
        kern_shift = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_shift, k_shift))
        if boundary_shift > 0:
            out = cv2.dilate(out, kern_shift)
        else:
            out = cv2.erode(out, kern_shift)

    if boundary_feather > 0.05:
        feather_sigma = max(0.8, float(boundary_feather) * 1.2 * res_scale)
        k_f = int(math.ceil(feather_sigma * 3.0)) * 2 + 1
        out = cv2.GaussianBlur(out, (k_f, k_f), feather_sigma)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def extract_subject_cutout_rgba(
    image_rgb: np.ndarray,
    boundary_shift: float = 0.0,
    boundary_feather: float = 0.0,
    media_id: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts the foreground subject as an RGBA uint8 image and alpha uint8 mask:
    - Computes high-precision subject alpha matte via DepthAndMatteEngine (U2NetP / MODNet + Guided Filter).
    - Applies user boundary_shift and boundary_feather adjustments.
    - Performs edge color decontamination on semi-transparent boundary pixels so background halo colors
      do not fringe the cutout subject.
    Returns (rgba_uint8, alpha_uint8).
    """
    if image_rgb is None or image_rgb.size == 0:
        empty_rgba = np.zeros((10, 10, 4), dtype=np.uint8)
        empty_a = np.zeros((10, 10), dtype=np.uint8)
        return empty_rgba, empty_a

    h, w = image_rgb.shape[:2]
    engine = get_depth_and_matte_engine()
    _, alpha_matte = engine.get_depth_and_matte(image_rgb, media_id=media_id)
    if alpha_matte.shape[:2] != (h, w):
        alpha_matte = cv2.resize(alpha_matte, (w, h), interpolation=cv2.INTER_LINEAR)

    alpha_matte = apply_boundary_shift_and_feather(
        alpha_matte,
        boundary_shift=boundary_shift,
        boundary_feather=boundary_feather,
    )

    # Edge color decontamination: propagate pure interior foreground colors into the semi-transparent edge
    rgb_f = image_rgb.astype(np.float32)
    core_fg = (alpha_matte > 0.92).astype(np.float32)
    if np.count_nonzero(core_fg) > 16:
        k_dec = max(5, (int(min(h, w) * 0.015) // 2) * 2 + 1)
        premult = rgb_f * core_fg[..., None]
        blur_rgb = cv2.GaussianBlur(premult, (k_dec, k_dec), 0)
        blur_w = cv2.GaussianBlur(core_fg, (k_dec, k_dec), 0)[..., None]
        pure_fg_rgb = np.where(blur_w > 1e-3, blur_rgb / np.maximum(blur_w, 1e-3), rgb_f)
        edge_band = (alpha_matte > 0.02) & (alpha_matte < 0.92)
        decontam_weight = np.clip(1.0 - alpha_matte, 0.0, 0.85)[..., None]
        rgb_decontam = np.where(
            edge_band[..., None],
            rgb_f * (1.0 - decontam_weight) + pure_fg_rgb * decontam_weight,
            rgb_f,
        )
        rgb_clean = np.clip(rgb_decontam, 0.0, 255.0).astype(np.uint8)
    else:
        rgb_clean = image_rgb.copy()

    alpha_u8 = np.clip(np.round(alpha_matte * 255.0), 0.0, 255.0).astype(np.uint8)
    rgba = np.dstack([rgb_clean, alpha_u8])
    return rgba, alpha_u8


# =============================================================================
# 6. LaMa ONNX Object Eraser & Multi-Scale Structure/Texture Inpainting
# =============================================================================

def transfer_best_donor_texture(
    roi_rgb: np.ndarray,
    roi_high: np.ndarray,
    mask: np.ndarray,
    donor_ring: np.ndarray,
    bw: int,
    bh: int,
    min_donor_pixels: int = 16,
) -> tuple[np.ndarray, bool]:
    """
    Score 8 directional donor shifts along ``donor_ring`` by boundary MAD error
    and blend the top-2 best-matching shifts with std matching.

    Preserves crisp directional textures (skin pores, hair, fabric, grass, wood)
    without washing them out via a uniform 8-way average.
    Returns ``(transferred_high, used_directional_shift)``.
    """
    rh, rw = mask.shape[:2]
    mask_u8 = (mask > 0).astype(np.uint8)
    mask_bool = mask_u8 > 0
    donor_bool = donor_ring > 0
    if np.count_nonzero(donor_bool) < min_donor_pixels or not np.any(mask_bool):
        return np.zeros_like(roi_high), False

    shift_x = max(10, int(bw * 1.15))
    shift_y = max(10, int(bh * 1.15))
    shifts = [
        (0, shift_x), (0, -shift_x), (shift_y, 0), (-shift_y, 0),
        (shift_y, shift_x), (-shift_y, -shift_x), (shift_y, -shift_x), (-shift_y, shift_x),
    ]

    roi_rgb_f = roi_rgb.astype(np.float32)
    ref_ring = roi_rgb_f[donor_bool]
    scored: list[tuple[float, np.ndarray]] = []
    for dy, dx in shifts:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        sm = cv2.warpAffine(
            mask_u8, M, (rw, rh),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        if np.count_nonzero((sm > 0) & mask_bool) == 0:
            sh_rgb = cv2.warpAffine(
                roi_rgb_f, M, (rw, rh),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
            )
            mad = float(np.mean(np.abs(sh_rgb[donor_bool] - ref_ring)))
            sh_high = cv2.warpAffine(
                roi_high, M, (rw, rh),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
            )
            scored.append((mad, sh_high))

    if not scored:
        return np.zeros_like(roi_high), False

    scored.sort(key=lambda item: item[0])
    top = scored[:2]
    if len(top) == 1:
        trans_high = top[0][1].copy()
    else:
        w0 = 1.0 / (top[0][0] + 1.0)
        w1 = 1.0 / (top[1][0] + 1.0)
        w_sum = w0 + w1
        trans_high = top[0][1] * (w0 / w_sum) + top[1][1] * (w1 / w_sum)

    d_std = float(np.std(roi_high[donor_bool]))
    t_std = float(np.std(trans_high[mask_bool]))
    if t_std > 0.05 and d_std > 0.1:
        gain = min(2.0, max(0.5, d_std / t_std))
        trans_high = trans_high * gain

    return trans_high, True


def _match_boundary_and_blend(
    roi_rgb: np.ndarray,
    restored: np.ndarray,
    bin_mask: np.ndarray,
    feather_ksize: int = 9,
    feather_sigma: float = 2.5,
) -> np.ndarray:
    """
    Match mean LAB tone on a narrow perimeter ring around ``bin_mask`` so
    upscaled 512x512 neural outputs blend without brightness/color seams.
    """
    mask_u8 = (bin_mask > 0).astype(np.uint8)
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    ring = (cv2.dilate(mask_u8, ring_k) > 0) & (mask_u8 == 0)

    matched = restored
    if np.count_nonzero(ring) >= 16:
        src_lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        dst_lab = cv2.cvtColor(restored, cv2.COLOR_RGB2LAB).astype(np.float32)
        delta = np.mean(src_lab[ring], axis=0) - np.mean(dst_lab[ring], axis=0)
        delta = np.clip(delta, -25.0, 25.0)
        dst_lab = np.clip(dst_lab + delta[None, None, :], 0.0, 255.0).astype(np.uint8)
        matched = cv2.cvtColor(dst_lab, cv2.COLOR_LAB2RGB)

    feather = cv2.GaussianBlur(
        mask_u8.astype(np.float32), (feather_ksize, feather_ksize), feather_sigma
    )[..., None]
    blended = roi_rgb.astype(np.float32) * (1.0 - feather) + matched.astype(np.float32) * feather
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


class LaMaInpaintEngine:
    """
    Local Neural Object Eraser & Inpainting Engine:
    - Supports ONNX LaMa models (lama_fp32.onnx / lama_quant.onnx / lama.onnx) in models/
      executed via OpenCV 5.0's graph-based CPU DNN engine (cv2.dnn.readNetFromONNX)
      or onnxruntime (DirectML / CPU) on a 512x512 context ROI crop.
    - Includes a zero-dependency Multi-Scale Structure + Best-Match Exemplar Texture
      Synthesis fallback when the ONNX model file is not present locally.
    """

    _instance: LaMaInpaintEngine | None = None

    def __init__(self, models_dir: Path | str | None = None, backend: str = "auto") -> None:
        self.search_dirs: list[Path] = []
        if models_dir is not None:
            self.search_dirs.append(Path(models_dir))

        # 1. Bundled models/ (PyInstaller sys._MEIPASS / models, or repo / models)
        try:
            from src.utils.paths import get_bundle_dir
            bundle_models = get_bundle_dir() / "models"
            if bundle_models not in self.search_dirs:
                self.search_dirs.append(bundle_models)
        except Exception:
            pass

        # 2. User AppData models/ (%LOCALAPPDATA%/PhotoGallery/models)
        try:
            from src.utils.paths import get_app_data_dir
            app_data_models = get_app_data_dir() / "models"
            if app_data_models not in self.search_dirs:
                self.search_dirs.append(app_data_models)
        except Exception:
            pass

        self.search_dirs.append(Path("models").resolve())
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            self.search_dirs.append(Path(local_app_data) / "PhotoGallery" / "models")
            self.search_dirs.append(Path(local_app_data) / "WindowsMediaGallery" / "models")
        project_root = Path(__file__).resolve().parent.parent.parent
        self.search_dirs.append(project_root / "models")

        self._backend = backend
        self._cv_net: Any = None
        self._session: Any = None
        self._model_name: str | None = None
        self._backend_used: str | None = None
        self._init_session()

    @staticmethod
    def default_search_dirs(model_dir: str | Path | None = None) -> list[Path]:
        dirs: list[Path] = []
        if model_dir is not None:
            dirs.append(Path(model_dir))
        try:
            from src.utils.paths import get_app_data_dir

            app_data_models = get_app_data_dir() / "models"
            if app_data_models not in dirs:
                dirs.append(app_data_models)
        except Exception:
            pass
        dirs.append(Path("models").resolve())
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            dirs.append(Path(local_app_data) / "PhotoGallery" / "models")
            dirs.append(Path(local_app_data) / "WindowsMediaGallery" / "models")
        project_root = Path(__file__).resolve().parent.parent.parent
        dirs.append(project_root / "models")
        return dirs

    @classmethod
    def find_model_file(cls, search_dirs: list[Path] | None = None) -> Path | None:
        candidates = [
            "lama_quant.onnx",
            "lama_fp32.onnx",
            "lama.onnx",
            "big-lama.onnx",
        ]
        for d in (search_dirs if search_dirs is not None else cls.default_search_dirs()):
            for fn in candidates:
                p = d / fn
                if p.is_file():
                    return p
        return None

    def _find_model_file(self) -> Path | None:
        return self.find_model_file(self.search_dirs)

    def _init_opencv_dnn(self, model_path: Path) -> bool:
        """Load ONNX LaMa model into OpenCV 5.0's graph-based CPU DNN engine."""
        try:
            net = cv2.dnn.readNetFromONNX(str(model_path))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._cv_net = net
            self._model_name = model_path.name
            self._backend_used = "opencv_dnn"
            logger.info("LaMa OpenCV DNN inpaint session loaded: %s", model_path.name)
            return True
        except Exception as e:
            logger.debug("LaMa OpenCV DNN init skipped: %s", e)
            self._cv_net = None
            return False

    def _init_onnxruntime(self, model_path: Path) -> bool:
        """Load ONNX LaMa model into onnxruntime (DirectML / CPU)."""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            prov_list = []
            if "DmlExecutionProvider" in providers:
                prov_list.append("DmlExecutionProvider")
            prov_list.append("CPUExecutionProvider")

            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(str(model_path), sess_opts, providers=prov_list)
            self._model_name = model_path.name
            if self._backend_used is None:
                self._backend_used = "onnxruntime"
            logger.info("LaMa ONNXRuntime inpaint session loaded: %s", model_path.name)
            return True
        except Exception as e:
            logger.debug("LaMa ONNXRuntime session init skipped: %s", e)
            self._session = None
            return False

    def has_neural_model(self) -> bool:
        """Return True when a neural LaMa ONNX session is loaded and enabled."""
        if not is_lama_enabled():
            return False
        return self._session is not None or self._cv_net is not None

    def reset_session(self) -> None:
        """Release loaded neural sessions so model files can be reloaded or deleted."""
        self._session = None
        self._cv_net = None
        self._model_name = None
        self._backend_used = None

    def _init_session(self) -> bool:
        model_path = self._find_model_file()
        if not model_path:
            return False

        loaded = False
        if self._backend in ("auto", "onnxruntime") and self._session is None:
            loaded = self._init_onnxruntime(model_path) or loaded
        if self._backend in ("auto", "opencv_dnn") and self._cv_net is None:
            if self._backend == "opencv_dnn" or not loaded:
                loaded = self._init_opencv_dnn(model_path) or loaded
        return loaded

    def _postprocess_neural_output(
        self, out: np.ndarray, roi_rgb: np.ndarray, bin_mask: np.ndarray
    ) -> np.ndarray:
        """Convert NCHW/CHW neural output to RGB uint8 and blend with boundary LAB matching."""
        rh, rw = roi_rgb.shape[:2]
        out_chw = out[0] if out.ndim == 4 else out
        out_hwc = out_chw.transpose(1, 2, 0)
        if float(np.max(out_hwc)) <= 1.5:
            out_hwc = out_hwc * 255.0
        out_u8 = np.clip(out_hwc, 0.0, 255.0).astype(np.uint8)
        restored = cv2.resize(out_u8, (rw, rh), interpolation=cv2.INTER_CUBIC)
        return _match_boundary_and_blend(roi_rgb, restored, bin_mask, feather_ksize=9, feather_sigma=2.5)

    def _run_opencv_dnn(
        self, roi_rgb: np.ndarray, bin_mask: np.ndarray, target_w: int = 512, target_h: int = 512
    ) -> np.ndarray:
        """Run LaMa ONNX inference on OpenCV 5.0's graph-based CPU DNN engine."""
        img_512 = cv2.resize(roi_rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        mask_512 = cv2.resize(bin_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        mask_512 = (mask_512 > 0).astype(np.float32)

        img_tensor = np.ascontiguousarray(
            (img_512.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        )
        mask_tensor = np.ascontiguousarray(mask_512[None, None, ...])

        try:
            self._cv_net.setInput(img_tensor, "image")
            self._cv_net.setInput(mask_tensor, "mask")
        except Exception:
            self._cv_net.setInput(img_tensor)

        out = self._cv_net.forward()
        return self._postprocess_neural_output(out, roi_rgb, bin_mask)

    def _run_onnxruntime(self, roi_rgb: np.ndarray, bin_mask: np.ndarray) -> np.ndarray:
        """Run LaMa ONNX inference via onnxruntime (DirectML / CPU)."""
        inputs = self._session.get_inputs()
        in_shape = inputs[0].shape
        target_h = in_shape[2] if len(in_shape) == 4 and isinstance(in_shape[2], int) and in_shape[2] > 0 else 512
        target_w = in_shape[3] if len(in_shape) == 4 and isinstance(in_shape[3], int) and in_shape[3] > 0 else 512

        img_512 = cv2.resize(roi_rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        mask_512 = cv2.resize(bin_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        mask_512 = (mask_512 > 0).astype(np.float32)

        img_tensor = (img_512.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        mask_tensor = mask_512[None, None, ...]

        feed = {inputs[0].name: img_tensor}
        if len(inputs) >= 2:
            feed[inputs[1].name] = mask_tensor

        out = self._session.run(None, feed)[0]
        return self._postprocess_neural_output(out, roi_rgb, bin_mask)

    def inpaint_roi(self, roi_rgb: np.ndarray, roi_mask_u8: np.ndarray) -> np.ndarray:
        """
        Inpaints a cropped ROI (H, W, 3) RGB uint8 where roi_mask_u8 > 0 marks the object to erase.
        Uses OpenCV 5.0 DNN or ONNXRuntime LaMa inference if loaded, otherwise runs multi-scale
        coarse-to-fine structure + best-match exemplar texture synthesis.
        """
        rh, rw = roi_rgb.shape[:2]
        bin_mask = (roi_mask_u8 > 127).astype(np.uint8)
        if np.count_nonzero(bin_mask) == 0:
            return roi_rgb.copy()

        # 1. Try LaMa neural session (ONNXRuntime first for 2.7x faster CPU inference, then OpenCV 5.0 DNN)
        if is_lama_enabled():
            if self._cv_net is None and self._session is None:
                self._init_session()

            if self._session is not None and self._backend in ("auto", "onnxruntime"):
                try:
                    return self._run_onnxruntime(roi_rgb, bin_mask)
                except Exception as e:
                    logger.debug("LaMa ONNXRuntime inference failed: %s", e)
                    if self._backend == "auto" and self._cv_net is None:
                        model_path = self._find_model_file()
                        if model_path is not None:
                            self._init_opencv_dnn(model_path)

            if self._cv_net is not None and self._backend in ("auto", "opencv_dnn"):
                try:
                    return self._run_opencv_dnn(roi_rgb, bin_mask)
                except Exception as e:
                    logger.debug("LaMa OpenCV DNN inference failed, using multi-scale fallback: %s", e)

        # 2. High-Fidelity Frequency Separation + Coherent 2D Spatial Texture Transfer Fallback
        pts_y, pts_x = np.where(bin_mask > 0)
        bw = int(np.max(pts_x) - np.min(pts_x)) + 1
        bh = int(np.max(pts_y) - np.min(pts_y)) + 1

        # 2a. Reconstruct smooth base illumination and color via Telea inpainting directly on unmasked borders
        inpaint_r = max(3, min(15, int(max(bw, bh) * 0.15)))
        low_inp = cv2.inpaint(
            roi_rgb,
            (bin_mask * 255).astype(np.uint8),
            inpaintRadius=inpaint_r,
            flags=cv2.INPAINT_TELEA,
        )

        # 2b. High frequencies (authentic gravel, asphalt grain, skin pores, fabric weave)
        sigma = max(2.5, min(8.0, float(max(bw, bh)) / 6.0))
        k = int(2 * round(2 * sigma) + 1)
        roi_high = roi_rgb.astype(np.float32) - cv2.GaussianBlur(roi_rgb, (k, k), sigma).astype(np.float32)
        roi_high[bin_mask > 0] = 0.0

        # 2c. Donor ring around the erase boundary to harvest authentic surface texture
        ring_r = max(11, int(max(bw, bh) * 0.45))
        outer_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_r * 2 + 1, ring_r * 2 + 1))
        donor_ring = (cv2.dilate(bin_mask, outer_k) > 0) & (bin_mask == 0)

        # 2d. Best-match directional 2D affine donor transfer (preserves pebble, pore, hair, fabric structure)
        trans_high, used_shift = transfer_best_donor_texture(
            roi_rgb, roi_high, bin_mask, donor_ring, bw, bh
        )
        if not used_shift:
            # Fallback for large centered objects: tile 2D spatial patches from donor perimeter
            trans_high = np.zeros_like(roi_high)
            patch_sz = max(12, min(36, min(rw, rh) // 6))
            ys, xs = np.where(donor_ring)
            if len(ys) > 20:
                rng = np.random.default_rng(seed=int(rw * 31 + rh * 17))
                step = max(4, patch_sz - 4)
                for py in range(0, rh, step):
                    for px in range(0, rw, step):
                        sub_mask = bin_mask[py:min(rh, py + patch_sz), px:min(rw, px + patch_sz)]
                        if np.any(sub_mask):
                            idx = rng.integers(0, len(ys))
                            sy = max(0, min(rh - patch_sz, ys[idx] - patch_sz // 2))
                            sx = max(0, min(rw - patch_sz, xs[idx] - patch_sz // 2))
                            patch = roi_high[sy:sy + patch_sz, sx:sx + patch_sz]
                            ph, pw = patch.shape[:2]
                            trans_high[py:py + ph, px:px + pw] = patch

        restored = np.clip(low_inp.astype(np.float32) + trans_high, 0.0, 255.0).astype(np.uint8)
        feather = cv2.GaussianBlur(bin_mask.astype(np.float32), (5, 5), 1.2)[..., None]
        final_rgb = roi_rgb.astype(np.float32) * (1.0 - feather) + restored.astype(np.float32) * feather
        return np.clip(final_rgb, 0.0, 255.0).astype(np.uint8)


_SHARED_LAMA_ENGINE: LaMaInpaintEngine | None = None


def get_lama_inpaint_engine() -> LaMaInpaintEngine:
    global _SHARED_LAMA_ENGINE
    if _SHARED_LAMA_ENGINE is None:
        _SHARED_LAMA_ENGINE = LaMaInpaintEngine()
    return _SHARED_LAMA_ENGINE


def inpaint_object_lama(
    image_rgb: np.ndarray,
    mask_u8: np.ndarray,
) -> np.ndarray:
    """
    Erases an unwanted object marked by mask_u8 (> 0) from image_rgb (H, W, 3) uint8.
    Automatically crops a padded bounding-box ROI around the mask for fast, high-resolution
    local inpainting via LaMaInpaintEngine.
    """
    if image_rgb is None or image_rgb.size == 0 or mask_u8 is None:
        return image_rgb
    h, w = image_rgb.shape[:2]
    if mask_u8.shape[:2] != (h, w):
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return image_rgb.copy()

    # Dilate mask slightly to cover anti-aliased object boundary fringes
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_mask = cv2.dilate((mask_u8 > 0).astype(np.uint8) * 255, k_dil)

    x1, x2 = int(np.min(xs)), int(np.max(xs))
    y1, y2 = int(np.min(ys)), int(np.max(ys))
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    pad = max(48, int(max(bw, bh) * 0.65))

    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad)
    cy2 = min(h, y2 + pad)

    result = image_rgb.copy()
    roi_rgb = result[cy1:cy2, cx1:cx2]
    roi_mask = dilated_mask[cy1:cy2, cx1:cx2]

    engine = get_lama_inpaint_engine()
    inpainted_roi = engine.inpaint_roi(roi_rgb, roi_mask)
    result[cy1:cy2, cx1:cx2] = inpainted_roi
    return result


LAMA_MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
LAMA_MODEL_FILENAME = "lama_fp32.onnx"
LAMA_MODEL_SIZE_MB = 208


def _get_lama_appdata_dir() -> Path:
    try:
        from src.utils.paths import get_app_data_dir
        return get_app_data_dir() / "models"
    except Exception:
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            return Path(local_app) / "PhotoGallery" / "models"
        return Path("models").resolve()


def is_lama_enabled() -> bool:
    """Return True if the user has enabled neural LaMa inpainting (default True)."""
    try:
        from PySide6.QtCore import QSettings

        s = QSettings("PhotoGallery64", "PhotoGallery64")
        return bool(s.value("ai/lama_enabled", True, type=bool))
    except Exception:
        return True


def set_lama_enabled(enabled: bool) -> None:
    """Enable or disable neural LaMa inpainting without uninstalling the model file."""
    try:
        from PySide6.QtCore import QSettings

        s = QSettings("PhotoGallery64", "PhotoGallery64")
        s.setValue("ai/lama_enabled", bool(enabled))
    except Exception:
        pass


def get_lama_model_status() -> dict[str, Any]:
    """Return installation status of the optional neural LaMa ONNX model without loading ONNX Runtime."""
    engine = _SHARED_LAMA_ENGINE
    model_path = engine._find_model_file() if engine is not None else LaMaInpaintEngine.find_model_file()
    enabled = is_lama_enabled()
    if model_path is not None and model_path.is_file():
        size_mb = round(model_path.stat().st_size / (1024 * 1024), 1)
        backend_label = (engine._backend_used or "ready") if engine is not None else "ready"
        return {
            "installed": True,
            "enabled": enabled,
            "path": model_path,
            "filename": model_path.name,
            "size_mb": size_mb,
            "backend": backend_label if enabled else "disabled",
        }
    return {
        "installed": False,
        "enabled": enabled,
        "path": None,
        "filename": LAMA_MODEL_FILENAME,
        "size_mb": LAMA_MODEL_SIZE_MB,
        "backend": None,
    }


def download_lama_model(
    progress_cb: Any = None,
    cancel_cb: Any = None,
    timeout: float = 30.0,
) -> Path | None:
    """
    Download ``lama_fp32.onnx`` (~208 MB) into user AppData ``models/`` atomically.
    ``progress_cb(bytes_done, total_bytes)`` and ``cancel_cb() -> bool`` are optional.
    Reloads the shared LaMa engine on success and returns the model path, or None on failure/cancel.
    """
    import urllib.request

    target_dir = _get_lama_appdata_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    dest_path = target_dir / LAMA_MODEL_FILENAME
    tmp_path = target_dir / f"{LAMA_MODEL_FILENAME}.part"

    try:
        req = urllib.request.Request(
            LAMA_MODEL_URL,
            headers={"User-Agent": "EssentialsPhotoGallery/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            chunk_size = 262144  # 256 KB
            with open(tmp_path, "wb") as f:
                while True:
                    if cancel_cb is not None and cancel_cb():
                        raise InterruptedError("Download cancelled")
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if progress_cb is not None:
                        progress_cb(done, total)

        if total > 0 and done != total:
            raise ValueError(f"Incomplete LaMa download ({done} of {total} bytes)")
        if tmp_path.stat().st_size < 150_000_000:
            raise ValueError("Downloaded LaMa model file is unexpectedly small")
        with open(tmp_path, "rb") as chk:
            head = chk.read(32).lstrip()
            if head.startswith(b"<") or b"<html" in head.lower():
                raise ValueError("Downloaded LaMa file appears to be HTML rather than an ONNX model")

        tmp_path.replace(dest_path)
        engine = get_lama_inpaint_engine()
        engine.reset_session()
        if progress_cb is not None and total > 0:
            progress_cb(total, total)
        return dest_path
    except Exception as e:
        logger.warning("LaMa model download failed or cancelled: %s", e)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return None


def uninstall_lama_model() -> bool:
    """
    Release the shared LaMa session and remove downloaded ``lama*.onnx`` files
    from user AppData / local ``models/`` directories.
    """
    engine = get_lama_inpaint_engine()
    engine.reset_session()

    removed = False
    candidate_names = ("lama_quant.onnx", "lama_fp32.onnx", "lama.onnx", "big-lama.onnx")
    for d in list(engine.search_dirs) + [_get_lama_appdata_dir()]:
        for name in candidate_names:
            p = d / name
            if p.is_file():
                try:
                    p.unlink()
                    removed = True
                except Exception as e:
                    logger.warning("Could not delete %s: %s", p, e)
    return removed


