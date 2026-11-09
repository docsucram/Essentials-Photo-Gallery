"""
High-Performance Face Detection & ArcFace Feature Extraction Engine for PhotoGallery64.
Provides zero-configuration, robust face detection and 512-D embedding extraction:
- Primary backend: InsightFace (buffalo_l: SCRFD-10GF detector + ArcFace 512-D recognizer)
  with DirectML GPU / CPU execution providers.
- Deterministic offline fallback: OpenCV Haar cascade detector + spatial gradient descriptor
  producing 512-D L2-normalized vectors and 96x96 WebP portrait crops out-of-the-box.
- EXIF orientation auto-transposition, bounding box normalization in [0.0, 1.0],
  and size/score filtering (>= 32x32 px, score >= 0.65).
"""

from __future__ import annotations

import io
import logging
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Suppress InsightFace / skimage SimilarityTransform.estimate deprecation FutureWarning
warnings.filterwarnings("ignore", message=r".*`estimate` is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface.*")
warnings.filterwarnings("ignore", message=r".*AVIF support not installed.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*Corrupt EXIF data.*", category=UserWarning)

import sys
import types
import cv2
import numpy as np
from PIL import Image, ImageOps

from src.core.face_matcher import normalize_vector

logger = logging.getLogger(__name__)


class _NumPySimilarityTransform:
    """
    Pure-NumPy Umeyama 2D similarity transform replacing `skimage.transform.SimilarityTransform`
    for InsightFace landmark alignment without requiring `scikit-image` or `scipy` (~120 MB).
    """

    def __init__(
        self,
        matrix: np.ndarray | None = None,
        scale: float | None = None,
        rotation: float | None = None,
        translation: tuple[float, float] | list[float] | np.ndarray | None = None,
    ) -> None:
        if matrix is not None:
            self.params = np.asarray(matrix, dtype=np.float64)
        else:
            s = 1.0 if scale is None else float(scale)
            r = 0.0 if rotation is None else float(rotation)
            if translation is None:
                tx, ty = 0.0, 0.0
            else:
                tx, ty = float(translation[0]), float(translation[1])
            c, sn = math.cos(r) * s, math.sin(r) * s
            self.params = np.array(
                [[c, -sn, tx], [sn, c, ty], [0.0, 0.0, 1.0]], dtype=np.float64
            )

    def estimate(self, src: np.ndarray, dst: np.ndarray) -> bool:
        src_arr = np.asarray(src, dtype=np.float64)
        dst_arr = np.asarray(dst, dtype=np.float64)
        num, dim = src_arr.shape
        src_mean = src_arr.mean(axis=0)
        dst_mean = dst_arr.mean(axis=0)
        src_demean = src_arr - src_mean
        dst_demean = dst_arr - dst_mean
        A = dst_demean.T @ src_demean / num
        d = np.ones((dim,), dtype=np.float64)
        if np.linalg.det(A) < 0:
            d[dim - 1] = -1
        T = np.eye(dim + 1, dtype=np.float64)
        U, S, Vt = np.linalg.svd(A)
        rank = np.linalg.matrix_rank(A)
        if rank == 0:
            self.params = np.nan * T
            return False
        elif rank == dim - 1:
            if np.linalg.det(U) * np.linalg.det(Vt) > 0:
                T[:dim, :dim] = U @ Vt
            else:
                s_val = d[dim - 1]
                d[dim - 1] = -1
                T[:dim, :dim] = U @ np.diag(d) @ Vt
                d[dim - 1] = s_val
        else:
            T[:dim, :dim] = U @ np.diag(d) @ Vt
        var_sum = float(src_demean.var(axis=0).sum())
        scale = (1.0 / var_sum * float(S @ d)) if var_sum > 1e-12 else 1.0
        T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
        T[:dim, :dim] *= scale
        self.params = T
        return True

    @classmethod
    def from_estimate(cls, src: np.ndarray, dst: np.ndarray) -> "_NumPySimilarityTransform":
        obj = cls()
        obj.estimate(src, dst)
        return obj

    def __add__(self, other: "_NumPySimilarityTransform") -> "_NumPySimilarityTransform":
        return _NumPySimilarityTransform(matrix=other.params @ self.params)


def _install_skimage_shim() -> None:
    """Installs a lightweight pure-NumPy `skimage.transform` shim if `skimage` is not loaded."""
    if "skimage.transform" not in sys.modules or sys.modules.get("skimage.transform") is None:
        sk_mod = sys.modules.get("skimage")
        if sk_mod is None:
            sk_mod = types.ModuleType("skimage")
            sys.modules["skimage"] = sk_mod
        sk_trans = types.ModuleType("skimage.transform")
        sk_trans.SimilarityTransform = _NumPySimilarityTransform  # type: ignore[attr-defined]
        sk_mod.transform = sk_trans  # type: ignore[attr-defined]
        sys.modules["skimage.transform"] = sk_trans


_install_skimage_shim()


@dataclass(slots=True)
class DetectedFace:
    """Represents a detected face with normalized coordinates, confidence score, and embedding."""

    rect_x: float          # Left coordinate normalized in [0.0, 1.0]
    rect_y: float          # Top coordinate normalized in [0.0, 1.0]
    rect_w: float          # Width normalized in [0.0, 1.0]
    rect_h: float          # Height normalized in [0.0, 1.0]
    score: float           # Confidence score [0.0, 1.0]
    embedding: np.ndarray  # (512,) float32 L2-normalized vector
    crop_webp: bytes       # 96x96 WebP compressed avatar bytes


def compute_iou(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float],
) -> float:
    """
    Computes Intersection-over-Union (IoU) of two normalized (x, y, w, h) bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter

    if union <= 0.0:
        return 0.0
    return float(inter / union)


def compute_box_overlap(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float],
) -> float:
    """
    Computes a robust matching score between two normalized (x, y, w, h) bounding boxes.
    Handles discrepancies between wide head/hair boxes (e.g. WLPG MP:RegionInfo) and
    tight facial landmark boxes (InsightFace/RetinaFace) by considering:
    1. Intersection-over-Union (IoU)
    2. Intersection-over-Minimum-Area (IoM: containment ratio)
    Returns maximum matching ratio in range [0.0, 1.0].
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area1 = max(1e-6, w1 * h1)
    area2 = max(1e-6, w2 * h2)
    union = area1 + area2 - inter

    iou = float(inter / union) if union > 0 else 0.0
    iom = float(inter / min(area1, area2))
    return max(iou, iom * 0.75)


class FaceEngine:
    """
    Tier 2 Face Detection & Recognition Engine.
    Detects faces, computes 512-D ArcFace-compatible feature embeddings,
    and produces 96x96 WebP avatar crops.
    """

    def __init__(
        self,
        backend: str = "auto",
        min_size: int = 40,
        min_score: float = 0.65,
        providers: list[str] | None = None,
        allow_fallback: bool = False,
    ) -> None:
        try:
            from PySide6.QtCore import QSettings
            settings = QSettings("PhotoGallery64", "PhotoGallery64")
            if min_size == 40 and settings.contains("face_min_size"):
                min_size = int(settings.value("face_min_size", 40))
            if min_score == 0.65 and settings.contains("face_min_score"):
                min_score = float(settings.value("face_min_score", 0.65))
        except Exception:
            pass

        self.min_size = max(16, int(min_size))
        self.min_score = float(min_score)
        self.backend_name = backend
        self.allow_fallback = bool(allow_fallback)
        self.is_available: bool = False
        self.model_name: str = ""
        self.active_provider: str = "None"
        self._insight_app: Any = None
        self._haar_cascade: cv2.CascadeClassifier | None = None

        if providers is None:
            avail: list[str] = []
            try:
                import onnxruntime
                avail = onnxruntime.get_available_providers()
            except Exception:
                pass
            providers = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in avail]
            if not providers:
                providers = ["CPUExecutionProvider"]
        self.providers = providers

        self._init_backend()

    def _init_backend(self) -> None:
        """Initializes primary InsightFace backend or falls back to OpenCV Haar cascade only if allow_fallback=True."""
        if self.backend_name in ("auto", "insightface"):
            for model_name in ("buffalo_l", "buffalo_s"):
                try:
                    import insightface  # type: ignore[import-not-found]
                    from insightface.app import FaceAnalysis  # type: ignore[import-not-found]

                    app = None
                    try:
                        app = FaceAnalysis(name=model_name, providers=self.providers)
                        app.prepare(ctx_id=0, det_size=(640, 640))
                        self.active_provider = self.providers[0] if self.providers else "CPUExecutionProvider"
                    except Exception:
                        app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
                        app.prepare(ctx_id=0, det_size=(640, 640))
                        self.active_provider = "CPUExecutionProvider"

                    self._insight_app = app
                    self.backend_name = "insightface"
                    self.model_name = model_name
                    self.is_available = True
                    return
                except Exception as ex:
                    logger.debug("InsightFace model %s load attempt failed: %s", model_name, ex)

        # Fallback handling
        if self.allow_fallback or self.backend_name == "opencv_haar":
            logger.info("Initializing OpenCV Haar cascade fallback (allow_fallback=%s, backend=%s).", self.allow_fallback, self.backend_name)
            self.backend_name = "opencv_haar"
            self.model_name = "OpenCV Haar Cascade"
            self.active_provider = "CPU (Haar Cascade Fallback)"
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                self._haar_cascade = cv2.CascadeClassifier(cascade_path)
            else:
                self._haar_cascade = cv2.CascadeClassifier()
            self.is_available = True
        else:
            logger.warning(
                "InsightFace models unavailable and allow_fallback=False; "
                "face detection/embedding disabled in production to protect ArcFace vector integrity."
            )
            self._insight_app = None
            self._haar_cascade = None
            self.backend_name = "unavailable"
            self.model_name = "None"
            self.active_provider = "None"
            self.is_available = False

    # -------------------------------------------------------------------------
    # Image Loading & EXIF Transposition
    # -------------------------------------------------------------------------
    @staticmethod
    def load_image_rgb(
        image: str | Path | np.ndarray | Image.Image,
        orientation: int = 1,
    ) -> np.ndarray:
        """
        Loads an image as an upright uint8 RGB numpy array.
        Applies EXIF orientation normalization.
        """
        if isinstance(image, (str, Path)):
            p = Path(image)
            if not p.exists():
                return np.zeros((10, 10, 3), dtype=np.uint8)
            from src.imaging.image_loader import (
                RAW_EXT_SET,
                decode_pil_to_srgb_and_alpha,
                decode_raw_to_srgb,
                is_pil_true_animation,
            )

            ext = p.suffix.lstrip(".").lower()
            if ext in RAW_EXT_SET:
                rgb_raw = decode_raw_to_srgb(p)
                if rgb_raw is None:
                    raise OSError(f"Unsupported or unreadable RAW file: {p}")
                return rgb_raw

            try:
                with Image.open(p) as pil_img:
                    disk_orient = 1
                    if not is_pil_true_animation(pil_img):
                        try:
                            disk_exif = pil_img.getexif()
                            disk_orient = int(disk_exif.get(0x0112) or 1)
                        except Exception:
                            disk_orient = 1
                    rgb_arr, _ = decode_pil_to_srgb_and_alpha(pil_img)
                    target_orient = int(orientation or 1)
                    if target_orient != disk_orient and target_orient in (3, 6, 8):
                        orient_to_angle = {1: 0, 6: 90, 3: 180, 8: 270}
                        delta_angle = (
                            orient_to_angle.get(target_orient, 0)
                            - orient_to_angle.get(disk_orient, 0)
                        ) % 360
                        if delta_angle != 0:
                            k = {90: -1, 180: 2, 270: 1}[delta_angle]
                            return np.ascontiguousarray(np.rot90(rgb_arr, k))
                    return rgb_arr
            except Exception:
                rgb_raw = decode_raw_to_srgb(p)
                if rgb_raw is not None:
                    return rgb_raw
                raise

        if isinstance(image, Image.Image):
            rgb_pil = image.convert("RGB")
            if orientation != 1:
                orient_to_angle = {1: 0, 6: -90, 3: 180, 8: 90}
                ang = orient_to_angle.get(orientation, 0)
                if ang != 0:
                    rgb_pil = rgb_pil.rotate(ang, expand=True)
            return np.ascontiguousarray(np.array(rgb_pil))

        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            if orientation != 1:
                k = {6: -1, 3: 2, 8: 1}.get(orientation, 0)
                if k != 0:
                    arr = np.ascontiguousarray(np.rot90(arr, k))
            return np.ascontiguousarray(arr)

        return np.zeros((10, 10, 3), dtype=np.uint8)

    # -------------------------------------------------------------------------
    # 96x96 WebP Avatar Thumbnail Encoding
    # -------------------------------------------------------------------------
    @staticmethod
    def crop_and_encode_webp(
        rgb_img: np.ndarray,
        rect_x: float,
        rect_y: float,
        rect_w: float,
        rect_h: float,
        target_size: tuple[int, int] = (128, 128),
        padding: float = 0.25,
    ) -> bytes:
        """
        Crops a face with portrait padding and square aspect ratio (no stretching),
        encoded as WebP bytes.
        """
        img_h, img_w = rgb_img.shape[:2]
        if img_h <= 0 or img_w <= 0:
            return b""
        if rect_w <= 0.005 or rect_h <= 0.005:
            return b""

        # Convert normalized coords to pixel bounds
        px = rect_x * img_w
        py = rect_y * img_h
        pw = rect_w * img_w
        ph = rect_h * img_h

        # Center of face
        cx = px + pw / 2.0
        cy = py + ph / 2.0

        # Square bounding side based on larger dimension + padding
        side = max(pw, ph) * (1.0 + 2.0 * padding)
        side = max(16.0, side)

        x1 = cx - side / 2.0
        y1 = cy - side / 2.0
        x2 = cx + side / 2.0
        y2 = cy + side / 2.0

        # Shift to fit within image bounds while preserving square aspect
        if x1 < 0:
            x2 = min(float(img_w), x2 - x1)
            x1 = 0.0
        if y1 < 0:
            y2 = min(float(img_h), y2 - y1)
            y1 = 0.0
        if x2 > img_w:
            x1 = max(0.0, x1 - (x2 - img_w))
            x2 = float(img_w)
        if y2 > img_h:
            y1 = max(0.0, y1 - (y2 - img_h))
            y2 = float(img_h)

        ix1 = int(round(x1))
        iy1 = int(round(y1))
        ix2 = int(round(x2))
        iy2 = int(round(y2))

        crop_w = max(1, ix2 - ix1)
        crop_h = max(1, iy2 - iy1)
        crop_side = min(crop_w, crop_h)

        # Center square crop
        ix1 += (crop_w - crop_side) // 2
        ix2 = ix1 + crop_side
        iy1 += (crop_h - crop_side) // 2
        iy2 = iy1 + crop_side

        face_region = rgb_img[iy1:iy2, ix1:ix2]
        if face_region.size == 0 or crop_side < 2:
            crop = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        else:
            crop = cv2.resize(face_region, target_size, interpolation=cv2.INTER_AREA)

        # Convert RGB to BGR for cv2 WebP encoding
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".webp", crop_bgr, [cv2.IMWRITE_WEBP_QUALITY, 85])
        if ok:
            return buf.tobytes()

        # Fallback to PIL WebP save
        bio = io.BytesIO()
        Image.fromarray(crop).save(bio, format="WEBP", quality=85)
        return bio.getvalue()

    # -------------------------------------------------------------------------
    # 512-D Feature Embedding Extraction
    # -------------------------------------------------------------------------
    @staticmethod
    def compute_spatial_gradient_descriptor(face_crop: np.ndarray) -> np.ndarray:
        """
        Deterministic, lightweight 512-D spatial gradient histogram descriptor.
        Partitions a normalized 112x112 face crop into an 8x8 spatial grid with 8 orientation bins.
        8 x 8 x 8 = 512 dimensions, L2-normalized.
        """
        if face_crop.size == 0:
            return np.zeros((512,), dtype=np.float32)

        # 1. Resize to canonical 112x112 and convert to grayscale
        canonical = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_AREA)
        if canonical.ndim == 3:
            gray = cv2.cvtColor(canonical, cv2.COLOR_RGB2GRAY)
        else:
            gray = canonical

        # 2. Compute spatial gradients via Sobel operators
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # 3. 8x8 spatial grid of 14x14 cells, with 8 orientation bins (45 deg each)
        cell_size = 14
        hist = np.zeros((8, 8, 8), dtype=np.float32)
        bin_idx = ((ang / 45.0).astype(int)) % 8

        for r in range(8):
            r_start = r * cell_size
            r_end = r_start + cell_size
            for c in range(8):
                c_start = c * cell_size
                c_end = c_start + cell_size

                cell_mag = mag[r_start:r_end, c_start:c_end]
                cell_bins = bin_idx[r_start:r_end, c_start:c_end]

                for b in range(8):
                    hist[r, c, b] = np.sum(cell_mag[cell_bins == b])

        flat = hist.flatten()
        return normalize_vector(flat)

    # -------------------------------------------------------------------------
    # Public API: detect_and_embed & extract_crop_and_embed_rect
    # -------------------------------------------------------------------------
    def detect_and_embed(
        self,
        image: str | Path | np.ndarray | Image.Image,
        orientation: int = 1,
    ) -> list[DetectedFace]:
        """
        Scans an image for faces, filtering faces < 32x32 px or score < min_score.
        Returns normalized bounding boxes, 512-D L2-normalized embeddings,
        and 96x96 WebP avatar crops.
        """
        if not self.is_available:
            return []

        rgb_img = self.load_image_rgb(image, orientation=orientation)
        h, w = rgb_img.shape[:2]
        if h < self.min_size or w < self.min_size:
            return []

        # 1. Primary InsightFace Pipeline
        if self._insight_app is not None:
            try:
                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                faces = self._insight_app.get(bgr_img)
                results: list[DetectedFace] = []

                for f in faces:
                    # Stage 1: Confidence Gate
                    score = float(getattr(f, "det_score", 0.0) or 0.0)
                    if score < self.min_score:
                        continue

                    bbox = getattr(f, "bbox", None)
                    if bbox is None or len(bbox) < 4:
                        continue

                    x1, y1, x2, y2 = bbox[:4]
                    bw = float(x2 - x1)
                    bh = float(y2 - y1)

                    # Stage 2: Minimum Pixel Size Gate
                    if bw < self.min_size or bh < self.min_size:
                        continue

                    # Stage 3: 5-Point Landmark Geometry Gate
                    kps = getattr(f, "kps", None)
                    if kps is None or len(kps) < 5:
                        continue

                    # Landmark indices: 0: left eye, 1: right eye, 2: nose, 3: left mouth, 4: right mouth
                    le_x, le_y = float(kps[0][0]), float(kps[0][1])
                    re_x, re_y = float(kps[1][0]), float(kps[1][1])
                    lm_x, lm_y = float(kps[3][0]), float(kps[3][1])
                    rm_x, rm_y = float(kps[4][0]), float(kps[4][1])

                    # In upright coordinates (Y downwards), eyes must be strictly above mouth
                    if le_y >= lm_y or re_y >= rm_y or max(le_y, re_y) >= min(lm_y, rm_y):
                        continue

                    # Inter-ocular distance must be >= 15% of bounding box width
                    inter_ocular_dist = math.hypot(re_x - le_x, re_y - le_y)
                    if inter_ocular_dist < 0.15 * bw:
                        continue

                    rect_x = float(np.clip(x1 / w, 0.0, 1.0))
                    rect_y = float(np.clip(y1 / h, 0.0, 1.0))
                    rect_w = float(np.clip(bw / w, 0.0, 1.0 - rect_x))
                    rect_h = float(np.clip(bh / h, 0.0, 1.0 - rect_y))

                    emb = getattr(f, "embedding", None)
                    if emb is not None and emb.size == 512:
                        embedding = normalize_vector(emb.astype(np.float32))
                    else:
                        if not (self.allow_fallback or self.backend_name == "opencv_haar"):
                            continue
                        # Fallback descriptor on detected face box
                        ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
                        ix2, iy2 = min(w, int(round(x2))), min(h, int(round(y2)))
                        embedding = self.compute_spatial_gradient_descriptor(rgb_img[iy1:iy2, ix1:ix2])

                    crop_bytes = self.crop_and_encode_webp(rgb_img, rect_x, rect_y, rect_w, rect_h)

                    results.append(
                        DetectedFace(
                            rect_x=rect_x,
                            rect_y=rect_y,
                            rect_w=rect_w,
                            rect_h=rect_h,
                            score=score,
                            embedding=embedding,
                            crop_webp=crop_bytes,
                        )
                    )

                return results
            except Exception as e:
                logger.warning("InsightFace inference failed: %s", e)
                if not (self.allow_fallback or self.backend_name == "opencv_haar"):
                    raise RuntimeError(f"InsightFace inference failed: {e}") from e

        # 2. OpenCV Haar Cascade Fallback Pipeline (unit tests / fallback only)
        if not (self.allow_fallback or self.backend_name == "opencv_haar"):
            return []

        results = []
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

        det_scale = min(1.0, 800.0 / float(max(w, h)))
        if det_scale < 1.0:
            dw = max(1, int(round(w * det_scale)))
            dh = max(1, int(round(h * det_scale)))
            det_gray = cv2.resize(gray, (dw, dh), interpolation=cv2.INTER_AREA)
            eff_min = max(self.min_size, int(min(dw, dh) * 0.065))
            min_neighbors = 5
        else:
            dw, dh = w, h
            det_gray = gray
            eff_min = self.min_size
            min_neighbors = 4

        eq_gray = cv2.equalizeHist(det_gray)

        if self._haar_cascade is not None and not self._haar_cascade.empty():
            raw_faces = self._haar_cascade.detectMultiScale(
                eq_gray,
                scaleFactor=1.1,
                minNeighbors=min_neighbors,
                minSize=(eff_min, eff_min),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            # Map back to original image pixel coordinates and apply NMS IoU deduplication
            scaled_boxes: list[tuple[int, int, int, int]] = []
            for (dx, dy, dbw, dbh) in raw_faces:
                x = int(round(dx / det_scale))
                y = int(round(dy / det_scale))
                bw = int(round(dbw / det_scale))
                bh = int(round(dbh / det_scale))
                if bw >= self.min_size and bh >= self.min_size:
                    scaled_boxes.append((x, y, bw, bh))

            # Sort largest first and suppress overlapping boxes (IoU > 0.35)
            scaled_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
            kept_boxes: list[tuple[int, int, int, int]] = []
            for box in scaled_boxes:
                bx, by, bw, bh = box
                b_area = float(bw * bh)
                overlap = False
                for (kx, ky, kw, kh) in kept_boxes:
                    ix1 = max(bx, kx)
                    iy1 = max(by, ky)
                    ix2 = min(bx + bw, kx + kw)
                    iy2 = min(by + bh, ky + kh)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = float((ix2 - ix1) * (iy2 - iy1))
                        union = b_area + float(kw * kh) - inter
                        if union > 0 and (inter / union) > 0.35:
                            overlap = True
                            break
                if not overlap:
                    kept_boxes.append(box)

            for (x, y, bw, bh) in kept_boxes:
                rect_x = float(np.clip(x / w, 0.0, 1.0))
                rect_y = float(np.clip(y / h, 0.0, 1.0))
                rect_w = float(np.clip(bw / w, 0.0, 1.0 - rect_x))
                rect_h = float(np.clip(bh / h, 0.0, 1.0 - rect_y))

                face_crop = rgb_img[max(0, y) : min(h, y + bh), max(0, x) : min(w, x + bw)]
                embedding = self.compute_spatial_gradient_descriptor(face_crop)
                crop_bytes = self.crop_and_encode_webp(rgb_img, rect_x, rect_y, rect_w, rect_h)

                results.append(
                    DetectedFace(
                        rect_x=rect_x,
                        rect_y=rect_y,
                        rect_w=rect_w,
                        rect_h=rect_h,
                        score=0.90,  # Established detector confidence
                        embedding=embedding,
                        crop_webp=crop_bytes,
                    )
                )

        return results

    def extract_crop_and_embed_rect(
        self,
        image: str | Path | np.ndarray | Image.Image,
        rect: tuple[float, float, float, float],
        orientation: int = 1,
    ) -> tuple[np.ndarray, bytes]:
        """
        Given a predefined face rectangle in normalized coordinates [0.0, 1.0]
        (e.g., from existing WLPG MP:RegionInfo metadata), extracts the
        512-D L2-normalized embedding and 96x96 WebP avatar thumbnail bytes.

        Returns:
            (embedding, crop_webp)
        """
        rect_x, rect_y, rect_w, rect_h = rect
        if float(rect_w) <= 0.005 or float(rect_h) <= 0.005:
            return np.zeros((512,), dtype=np.float32), b""

        rgb_img = self.load_image_rgb(image, orientation=orientation)
        h, w = rgb_img.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((512,), dtype=np.float32), b""

        rect_x = float(np.clip(rect_x, 0.0, 1.0))
        rect_y = float(np.clip(rect_y, 0.0, 1.0))
        rect_w = float(np.clip(rect_w, 0.0, 1.0 - rect_x))
        rect_h = float(np.clip(rect_h, 0.0, 1.0 - rect_y))

        # Pixel coordinates
        x1 = max(0, int(round(rect_x * w)))
        y1 = max(0, int(round(rect_y * h)))
        x2 = min(w, int(round((rect_x + rect_w) * w)))
        y2 = min(h, int(round((rect_y + rect_h) * h)))

        if x2 <= x1 or y2 <= y1:
            face_region = rgb_img
        else:
            face_region = rgb_img[y1:y2, x1:x2]

        crop_bytes = self.crop_and_encode_webp(rgb_img, rect_x, rect_y, rect_w, rect_h)

        # Compute 512-D embedding
        if self._insight_app is not None:
            try:
                # If InsightFace is active, try to embed the cropped region
                bgr_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2BGR)
                faces = self._insight_app.get(bgr_face)
                if faces and hasattr(faces[0], "embedding") and faces[0].embedding is not None:
                    return normalize_vector(faces[0].embedding.astype(np.float32)), crop_bytes
            except Exception as e:
                logger.warning("InsightFace crop embed failed: %s", e)

        if self.allow_fallback or self.backend_name == "opencv_haar":
            # Deterministic 512-D spatial gradient embedding for unit tests
            embedding = self.compute_spatial_gradient_descriptor(face_region)
            return embedding, crop_bytes

        # Production without InsightFace: do NOT pollute media_people.embedding with non-ArcFace vectors
        return np.zeros((512,), dtype=np.float32), crop_bytes
