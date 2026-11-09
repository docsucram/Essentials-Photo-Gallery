"""
Zero-dependency color-managed, bit-depth-normalized, alpha-preserving, and animation-aware
image decoding & saving helpers for Essentials Photo Gallery.
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

warnings.filterwarnings("ignore", message=r".*AVIF support not installed.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*Corrupt EXIF data.*", category=UserWarning)

try:
    from PIL import ImageCms
except Exception:  # pragma: no cover
    ImageCms = None  # type: ignore[assignment]

_CODECS_REGISTERED = False


def register_nextgen_codecs() -> dict[str, bool]:
    """
    Registers next-generation image codecs with Pillow:
      - `pillow_heif` for HEIC, HEIF, HIF
      - `pillow_avif` for AVIF
      - `pillow_jxl` for JPEG XL (.jxl)
    Returns a dict indicating the availability of each format.
    """
    global _CODECS_REGISTERED
    status = {"heif": False, "avif": False, "jxl": False}

    # 1. HEIF / HEIC / HIF
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        status["heif"] = True
    except Exception:
        pass

    # 2. AVIF
    try:
        import pillow_avif  # registers AVIF plugin with Pillow on import
        status["avif"] = True
    except Exception:
        pass

    # 3. JPEG XL (.jxl)
    try:
        import pillow_jxl  # registers JXL plugin with Pillow on import
        status["jxl"] = True
    except Exception:
        pass

    _CODECS_REGISTERED = True
    return status


# Automatically register on import
register_nextgen_codecs()

from src.imaging.opencv_engine import PhotoAdjustmentRecipe, _crop_largest_inscribed_rect

HEIF_EXT_SET: frozenset[str] = frozenset({"heic", "heif", "hif"})
AVIF_EXT_SET: frozenset[str] = frozenset({"avif"})
JXL_EXT_SET: frozenset[str] = frozenset({"jxl"})
NEXTGEN_EXT_SET: frozenset[str] = HEIF_EXT_SET | AVIF_EXT_SET | JXL_EXT_SET
NEXTGEN_DOT_EXT_SET: frozenset[str] = frozenset(f".{ext}" for ext in NEXTGEN_EXT_SET)

RAW_EXT_SET: frozenset[str] = frozenset(
    {
        "raf",
        "cr2",
        "cr3",
        "nef",
        "nrw",
        "arw",
        "dng",
        "orf",
        "ori",
        "rw2",
        "rwl",
        "raw",
        "pef",
        "srw",
        "3fr",
        "mrw",
        "sr2",
        "srf",
        "x3f",
        "erf",
        "kdc",
        "dcr",
        "mos",
        "mef",
        "iiq",
    }
)
RAW_DOT_EXT_SET: frozenset[str] = frozenset(f".{ext}" for ext in RAW_EXT_SET)
ALPHA_SAVE_DOT_EXT_SET: frozenset[str] = frozenset(
    {".png", ".webp", ".tif", ".tiff", ".ico", ".avif", ".jxl", ".heic", ".heif"}
)

_SRGB_CMS_PROFILE: Any = None


def _get_srgb_cms_profile() -> Any:
    global _SRGB_CMS_PROFILE
    if _SRGB_CMS_PROFILE is None and ImageCms is not None:
        try:
            _SRGB_CMS_PROFILE = ImageCms.createProfile("sRGB")
        except Exception:
            _SRGB_CMS_PROFILE = None
    return _SRGB_CMS_PROFILE


def normalize_pil_bit_depth(img: Image.Image) -> Image.Image:
    """
    Normalizes 16-bit integer ('I;16', 'I;16L', 'I;16B', 'I') and 32-bit float ('F')
    TIFF/PNG images cleanly to 8-bit without clipping all highlights to white, while
    preserving embedded ICC profiles in `img.info`.
    """
    mode = img.mode
    if not (mode.startswith("I") or mode == "F"):
        # Note: Pillow >= 10 may also expose multi-channel 16-bit via custom modes in rare builds
        return img

    icc = img.info.get("icc_profile")
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return img

    if arr.dtype == np.uint16:
        u8 = np.clip(arr.astype(np.float32) / 257.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    elif np.issubdtype(arr.dtype, np.integer):
        f_arr = arr.astype(np.float32)
        hi = float(f_arr.max()) if f_arr.size else 0.0
        lo = float(f_arr.min()) if f_arr.size else 0.0
        if hi <= 255.0 and lo >= 0.0:
            u8 = f_arr.astype(np.uint8)
        elif hi <= 65535.0 and lo >= 0.0:
            u8 = np.clip(f_arr / 257.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        elif hi > lo:
            u8 = np.clip((f_arr - lo) * (255.0 / (hi - lo)), 0.0, 255.0).astype(np.uint8)
        else:
            u8 = np.zeros_like(f_arr, dtype=np.uint8)
    else:
        f_arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        hi = float(f_arr.max()) if f_arr.size else 0.0
        lo = float(f_arr.min()) if f_arr.size else 0.0
        if hi <= 1.05 and lo >= -0.05:
            u8 = np.clip(f_arr * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        elif hi <= 255.0 and lo >= 0.0:
            u8 = np.clip(f_arr + 0.5, 0.0, 255.0).astype(np.uint8)
        elif hi > lo:
            u8 = np.clip((f_arr - lo) * (255.0 / (hi - lo)), 0.0, 255.0).astype(np.uint8)
        else:
            u8 = np.zeros(f_arr.shape, dtype=np.uint8)

    out = Image.fromarray(u8, mode="L" if u8.ndim == 2 else "RGB")
    if icc:
        out.info["icc_profile"] = icc
    return out


def convert_icc_to_srgb(img: Image.Image) -> Image.Image:
    """
    Converts images carrying wide-gamut or non-sRGB ICC profiles (Display P3, Adobe RGB,
    ProPhoto RGB, CMYK, calibrated Grayscale) to standard sRGB via LittleCMS2 (`PIL.ImageCms`),
    preserving any alpha channel intact. Falls back safely if no profile or corrupt profile.
    """
    if ImageCms is None:
        return img
    icc_bytes = img.info.get("icc_profile")
    if not icc_bytes:
        return img
    srgb_prof = _get_srgb_cms_profile()
    if srgb_prof is None:
        return img

    try:
        src_prof = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        desc = (ImageCms.getProfileDescription(src_prof) or "").strip().lower()
        if "srgb" in desc and img.mode in ("RGB", "RGBA"):
            return img

        alpha_band: Image.Image | None = None
        work = img
        if work.mode == "P" and "transparency" in work.info:
            work = work.convert("RGBA")
        if work.mode == "RGBA":
            alpha_band = work.getchannel("A")
            work = work.convert("RGB")
        elif work.mode == "LA":
            alpha_band = work.getchannel("A")
            work = work.getchannel("L")
        elif work.mode not in ("RGB", "CMYK", "L"):
            work = work.convert("RGB")

        converted = ImageCms.profileToProfile(
            work,
            src_prof,
            srgb_prof,
            renderingIntent=ImageCms.Intent.PERCEPTUAL,
            outputMode="RGB",
        )
        if converted is None:
            return img
        if alpha_band is not None:
            converted = converted.convert("RGBA")
            converted.putalpha(alpha_band)
        return converted
    except Exception:
        return img


def is_pil_true_animation(img: Image.Image) -> bool:
    """
    Returns True ONLY for genuine multi-frame animated formats (GIF, WebP, APNG/PNG).
    Crucially excludes `MPO` (Multi-Picture Object JPEGs used by modern smartphones for HDR gainmaps
    and portrait depth maps), where Pillow sets `img.is_animated = True` even though frame 0 is a
    static photo requiring normal EXIF orientation transposition!
    """
    fmt = (getattr(img, "format", None) or "").upper()
    if fmt not in ("GIF", "WEBP", "PNG", "APNG"):
        return False
    return bool(getattr(img, "is_animated", False) and int(getattr(img, "n_frames", 1)) > 1)


def decode_pil_to_srgb_and_alpha(
    img: Image.Image,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Decodes an open PIL Image (applying EXIF transpose on static/MPO photos, bit-depth
    normalization, and wide-gamut ICC -> sRGB conversion) into:
      - `rgb`: contiguous `uint8` `(H, W, 3)` array (with fully transparent `alpha == 0`
        pixels filled white so spatial filters never bleed black halos into edges)
      - `alpha`: contiguous `uint8` `(H, W)` array if the image has genuine non-opaque
        pixels (`min(alpha) < 255`), else `None`.
    """
    if is_pil_true_animation(img):
        try:
            img.seek(0)
        except Exception:
            pass
        work = img.copy()
    else:
        work = ImageOps.exif_transpose(img)

    work = normalize_pil_bit_depth(work)
    work = convert_icc_to_srgb(work)

    has_alpha = (work.mode in ("RGBA", "LA", "PA")) or (
        work.mode == "P" and "transparency" in work.info
    )
    if has_alpha:
        rgba = np.ascontiguousarray(np.array(work.convert("RGBA"), dtype=np.uint8))
        alpha = np.ascontiguousarray(rgba[:, :, 3])
        if alpha.size > 0 and int(alpha.min()) < 255:
            rgb = np.ascontiguousarray(rgba[:, :, :3].copy())
            zero_mask = alpha == 0
            if np.any(zero_mask):
                rgb[zero_mask] = 255
            return rgb, alpha
        return np.ascontiguousarray(rgba[:, :, :3]), None

    rgb = np.ascontiguousarray(np.array(work.convert("RGB"), dtype=np.uint8))
    return rgb, None


def decode_wic_to_srgb(file_path: Path | str) -> np.ndarray | None:
    """
    Decodes an image or camera RAW file via Windows Imaging Component (WIC) using native
    `windowscodecs.dll` and installed manufacturer codecs (e.g. Nikon NEF, Canon, Sony, Apple ProRAW).
    Returns an sRGB 8-bit `(H, W, 3)` numpy array, or None if unavailable/unsupported.
    """
    import sys

    if sys.platform != "win32":
        return None

    try:
        import ctypes
        from ctypes import POINTER, Structure, WINFUNCTYPE, byref, c_int32, c_ubyte, c_uint32, c_void_p, c_wchar, c_wchar_p, wintypes

        wic = ctypes.windll.LoadLibrary("windowscodecs.dll")
        ole32 = ctypes.windll.ole32
        ole32.CoInitialize(None)

        class GUID(Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", c_ubyte * 8),
            ]

        def _guid(s: str) -> GUID:
            g = GUID()
            ole32.CLSIDFromString(s, byref(g))
            return g

        GUID_24bppBGR = _guid("{6fddc324-4e03-4b15-be80-c75b0d095012}")
        GUID_32bppBGRA = _guid("{6fddc324-4e03-4b15-be80-c75b0d095020}")

        factory = c_void_p()
        if wic.WICCreateImagingFactory_Proxy(c_uint32(0x0237), byref(factory)) != 0 or not factory.value:
            return None
        f_vtable = ctypes.cast(factory, POINTER(POINTER(c_void_p))).contents

        create_dec_fn = WINFUNCTYPE(
            c_int32, c_void_p, c_wchar_p, c_void_p, wintypes.DWORD, c_uint32, POINTER(c_void_p)
        )(f_vtable[3])

        decoder = c_void_p()
        if create_dec_fn(factory, str(file_path), None, 0x80000000, 0, byref(decoder)) != 0 or not decoder.value:
            return None
        d_vtable = ctypes.cast(decoder, POINTER(POINTER(c_void_p))).contents

        # Try sources: Preview first (full-res JPEG/RGB for RAW formats), then Frame 0, then Thumbnail
        sources: list[c_void_p] = []
        pv = c_void_p()
        if WINFUNCTYPE(c_int32, c_void_p, POINTER(c_void_p))(d_vtable[9])(decoder, byref(pv)) == 0 and pv.value:
            sources.append(pv)
        fr = c_void_p()
        if WINFUNCTYPE(c_int32, c_void_p, c_uint32, POINTER(c_void_p))(d_vtable[13])(decoder, 0, byref(fr)) == 0 and fr.value:
            sources.append(fr)
        tb = c_void_p()
        if WINFUNCTYPE(c_int32, c_void_p, POINTER(c_void_p))(d_vtable[11])(decoder, byref(tb)) == 0 and tb.value:
            sources.append(tb)

        for src in sources:
            s_vtable = ctypes.cast(src, POINTER(POINTER(c_void_p))).contents
            get_size_fn = WINFUNCTYPE(c_int32, c_void_p, POINTER(c_uint32), POINTER(c_uint32))(s_vtable[3])
            w = c_uint32()
            h = c_uint32()
            if get_size_fn(src, byref(w), byref(h)) != 0 or w.value == 0 or h.value == 0:
                continue

            get_fmt_fn = WINFUNCTYPE(c_int32, c_void_p, POINTER(GUID))(s_vtable[4])
            src_fmt = GUID()
            get_fmt_fn(src, byref(src_fmt))
            guid_buf = (c_wchar * 40)()
            ole32.StringFromGUID2(byref(src_fmt), guid_buf, 40)
            fmt_str = guid_buf.value.upper()

            copy_pixels_fn = WINFUNCTYPE(c_int32, c_void_p, c_void_p, c_uint32, c_uint32, c_void_p)(s_vtable[7])

            # Native 24bpp RGB
            if fmt_str == "{6FDDC324-4E03-4BFE-B185-3D77768DC90C}":
                stride = w.value * 3
                buf_size = stride * h.value
                buf = (c_ubyte * buf_size)()
                if copy_pixels_fn(src, None, stride, buf_size, buf) == 0:
                    arr = np.frombuffer(buf, dtype=np.uint8).reshape((h.value, w.value, 3))
                    return np.ascontiguousarray(arr)

            # Native 24bpp BGR
            if fmt_str == "{6FDDC324-4E03-4B15-BE80-C75B0D095012}":
                stride = w.value * 3
                buf_size = stride * h.value
                buf = (c_ubyte * buf_size)()
                if copy_pixels_fn(src, None, stride, buf_size, buf) == 0:
                    arr = np.frombuffer(buf, dtype=np.uint8).reshape((h.value, w.value, 3))
                    return np.ascontiguousarray(arr[:, :, ::-1])

            # Native 32bpp BGRA or RGBA
            if fmt_str in ("{6FDDC324-4E03-4B15-BE80-C75B0D095020}", "{F5C7AD2D-6A8D-43DD-A7A8-A29935261A20}"):
                stride = w.value * 4
                buf_size = stride * h.value
                buf = (c_ubyte * buf_size)()
                if copy_pixels_fn(src, None, stride, buf_size, buf) == 0:
                    arr = np.frombuffer(buf, dtype=np.uint8).reshape((h.value, w.value, 4))
                    if fmt_str.startswith("{6FDDC324"):
                        return np.ascontiguousarray(arr[:, :, [2, 1, 0]])
                    return np.ascontiguousarray(arr[:, :, :3])

            # Convert to 32bppBGRA or 24bppBGR via WIC converter
            for target_guid, is_bgra in [(GUID_32bppBGRA, True), (GUID_24bppBGR, False)]:
                conv = c_void_p()
                if wic.WICConvertBitmapSource(byref(target_guid), src, byref(conv)) == 0 and conv.value:
                    c_vt = ctypes.cast(conv, POINTER(POINTER(c_void_p))).contents
                    c_copy_fn = WINFUNCTYPE(c_int32, c_void_p, c_void_p, c_uint32, c_uint32, c_void_p)(c_vt[7])
                    bpp = 4 if is_bgra else 3
                    stride = w.value * bpp
                    buf_size = stride * h.value
                    buf = (c_ubyte * buf_size)()
                    if c_copy_fn(conv, None, stride, buf_size, buf) == 0:
                        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h.value, w.value, bpp))
                        if is_bgra:
                            return np.ascontiguousarray(arr[:, :, [2, 1, 0]])
                        return np.ascontiguousarray(arr[:, :, ::-1])
    except Exception:
        pass
    return None


def extract_wic_thumbnail(file_path: Path | str, max_dim: int = 256) -> Any | None:
    """
    Extracts an embedded thumbnail or fast preview via Windows Imaging Component (WIC),
    returning a PySide6 `QImage` scaled to `max_dim`, or None if unsupported.
    """
    arr = decode_wic_to_srgb(file_path)
    if arr is None:
        return None
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage

        h, w, ch = arr.shape
        bytes_per_line = ch * w
        qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        if qimg.width() > max_dim or qimg.height() > max_dim:
            qimg = qimg.scaled(max_dim, max_dim, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return qimg
    except Exception:
        return None


def decode_raw_to_srgb(file_path: Path | str) -> np.ndarray | None:
    """Demosaics a camera RAW file to sRGB 8-bit `(H, W, 3)` using camera WB with auto-WB and WIC fallbacks."""
    try:
        import rawpy

        with rawpy.imread(str(file_path)) as raw:
            try:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_color=rawpy.ColorSpace.sRGB,
                    output_bps=8,
                )
            except Exception:
                rgb = raw.postprocess(
                    use_auto_wb=True,
                    output_color=rawpy.ColorSpace.sRGB,
                    output_bps=8,
                )
            return np.ascontiguousarray(rgb)
    except Exception:
        pass

    # Fallback to Windows Imaging Component (handles Nikon Z9 High Efficiency RAW and OEM codecs)
    return decode_wic_to_srgb(file_path)


def transform_alpha_for_recipe(
    master_alpha: np.ndarray,
    recipe: PhotoAdjustmentRecipe,
    target_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Applies identical straighten + crop geometry as `render_adjusted_image` to an alpha channel."""
    alpha = master_alpha
    if abs(recipe.straighten_deg) >= 0.05:
        h, w = alpha.shape[:2]
        center = (w / 2.0, h / 2.0)
        rot_matrix = cv2.getRotationMatrix2D(center, -float(recipe.straighten_deg), 1.0)
        alpha = cv2.warpAffine(
            alpha,
            rot_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if recipe.crop_rect is None:
            alpha = _crop_largest_inscribed_rect(alpha, recipe.straighten_deg, w, h)

    if recipe.crop_rect is not None:
        h, w = alpha.shape[:2]
        l, t, r, b = recipe.crop_rect
        if isinstance(l, float) and 0.0 <= l <= 1.0 and 0.0 <= r <= 1.0:
            l = int(l * w)
            t = int(t * h)
            r = int(r * w)
            b = int(b * h)
        else:
            l, t, r, b = int(l), int(t), int(r), int(b)
        l = max(0, min(w - 1, l))
        t = max(0, min(h - 1, t))
        r = max(l + 1, min(w, r))
        b = max(t + 1, min(h, b))
        alpha = alpha[t:b, l:r]

    if target_hw is not None:
        th, tw = target_hw
        if alpha.shape[0] != th or alpha.shape[1] != tw:
            alpha = cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_LINEAR)

    return np.ascontiguousarray(alpha)


def build_save_pil_image(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    target_suffix: str,
) -> Image.Image:
    """
    Constructs a PIL Image ready for disk export:
      - Re-attaches RGBA alpha when saving to `.png`, `.webp`, `.tif`, `.tiff`, or `.ico`
      - Composites cleanly over white `(255, 255, 255)` when saving to opaque formats like `.jpg`.
    """
    sfx = target_suffix.lower()
    if not sfx.startswith("."):
        sfx = f".{sfx}"
    if alpha is None:
        return Image.fromarray(rgb, mode="RGB")

    if alpha.shape[:2] != rgb.shape[:2]:
        alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    if sfx in ALPHA_SAVE_DOT_EXT_SET:
        rgba = np.dstack([rgb, alpha]).astype(np.uint8)
        return Image.fromarray(np.ascontiguousarray(rgba), mode="RGBA")

    a_f = alpha.astype(np.float32)[:, :, None] / 255.0
    comp = np.clip(rgb.astype(np.float32) * a_f + 255.0 * (1.0 - a_f) + 0.5, 0.0, 255.0).astype(
        np.uint8
    )
    return Image.fromarray(np.ascontiguousarray(comp), mode="RGB")


def is_animated_image_file(file_path: Path | str) -> bool:
    """Returns True if `file_path` is a multi-frame animated GIF or WebP."""
    p = Path(file_path)
    if p.suffix.lstrip(".").lower() not in ("gif", "webp"):
        return False
    try:
        from PySide6.QtGui import QImageReader

        reader = QImageReader(str(p))
        if reader.supportsAnimation() and reader.imageCount() > 1:
            return True
    except Exception:
        pass
    try:
        with Image.open(p) as im:
            return bool(getattr(im, "is_animated", False) and int(getattr(im, "n_frames", 1)) > 1)
    except Exception:
        return False
