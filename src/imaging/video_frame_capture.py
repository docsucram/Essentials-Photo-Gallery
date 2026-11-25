"""
Video Frame Capture ("Save Still Frame as Photo").

Extracts a full-resolution still frame from a video at the current playback
timestamp using OpenCV (with optional QImage fallback), writes EXIF capture
date (offset by playback position) and GPS coordinates, and returns the saved
file path + dimensions for immediate catalog indexing.
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

logger = logging.getLogger(__name__)


def _deg_to_dms_rational(val: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Convert decimal degrees to EXIF rational DMS tuple ((deg,1),(min,1),(sec*100,100))."""
    abs_v = abs(float(val))
    deg = int(abs_v)
    min_f = (abs_v - deg) * 60.0
    minute = int(min_f)
    sec = int(round((min_f - minute) * 60.0 * 100.0))
    if sec >= 6000:
        sec -= 6000
        minute += 1
    if minute >= 60:
        minute -= 60
        deg += 1
    return ((deg, 1), (minute, 1), (sec, 100))


def _build_exif_bytes(
    *,
    date_taken_ts: int | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> bytes | None:
    """Build EXIF payload with DateTimeOriginal and optional GPS coordinates."""
    try:
        import piexif

        zeroth: dict[int, Any] = {}
        exif_ifd: dict[int, Any] = {}
        gps_ifd: dict[int, Any] = {}

        if date_taken_ts is not None:
            dt = datetime.datetime.fromtimestamp(int(date_taken_ts))
            dt_bytes = dt.strftime("%Y:%m:%d %H:%M:%S").encode("ascii")
            zeroth[piexif.ImageIFD.DateTime] = dt_bytes
            exif_ifd[piexif.ExifIFD.DateTimeOriginal] = dt_bytes
            exif_ifd[piexif.ExifIFD.DateTimeDigitized] = dt_bytes

        if lat is not None and lon is not None:
            gps_ifd[piexif.GPSIFD.GPSLatitudeRef] = b"N" if float(lat) >= 0 else b"S"
            gps_ifd[piexif.GPSIFD.GPSLatitude] = _deg_to_dms_rational(float(lat))
            gps_ifd[piexif.GPSIFD.GPSLongitudeRef] = b"E" if float(lon) >= 0 else b"W"
            gps_ifd[piexif.GPSIFD.GPSLongitude] = _deg_to_dms_rational(float(lon))

        if zeroth or exif_ifd or gps_ifd:
            return piexif.dump({"0th": zeroth, "Exif": exif_ifd, "GPS": gps_ifd})
    except Exception:
        pass

    # Fallback: Pillow native Exif (DateTimeOriginal)
    if date_taken_ts is not None:
        try:
            dt = datetime.datetime.fromtimestamp(int(date_taken_ts))
            dt_str = dt.strftime("%Y:%m:%d %H:%M:%S")
            ex = Image.Exif()
            ex[306] = dt_str    # DateTime
            ex[36867] = dt_str  # DateTimeOriginal
            ex[36868] = dt_str  # DateTimeDigitized
            return ex.tobytes()
        except Exception:
            pass
    return None


def _next_still_frame_path(video_path: Path, position_ms: int) -> Path:
    """Return a collision-safe '<stem>_frame_<MM>m<SS>s.jpg' path in the same folder."""
    total_sec = max(0, int(position_ms // 1000))
    mins, secs = divmod(total_sec, 60)
    base_stem = f"{video_path.stem}_frame_{mins:02d}m{secs:02d}s"
    cand = video_path.parent / f"{base_stem}.jpg"
    idx = 2
    while cand.exists():
        cand = video_path.parent / f"{base_stem}_{idx}.jpg"
        idx += 1
    return cand


def capture_video_still_frame(
    video_path: Path | str,
    position_ms: int = 0,
    *,
    fallback_rgb_image: Image.Image | None = None,
    base_date_taken: int | None = None,
    lat: float | None = None,
    lon: float | None = None,
    jpeg_quality: int = 95,
) -> tuple[Path, int, int, int | None] | None:
    """
    Decode the video frame at ``position_ms`` and save it as a high-quality JPEG
    in the same folder as ``video_path``.

    Returns ``(saved_path, width, height, effective_date_taken)`` on success, or ``None``.
    """
    vpath = Path(video_path)
    if not vpath.is_file() and fallback_rgb_image is None:
        return None

    pil_img: Image.Image | None = None
    if vpath.is_file():
        cap = cv2.VideoCapture(str(vpath))
        try:
            if cap.isOpened():
                if position_ms > 0:
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(position_ms))
                ok, frame = cap.read()
                if (not ok or frame is None) and position_ms > 0:
                    cap.set(cv2.CAP_PROP_POS_MSEC, 0.0)
                    ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
        except Exception as exc:
            logger.warning("OpenCV video frame capture failed for %s: %s", vpath, exc)
        finally:
            cap.release()

    if pil_img is None and fallback_rgb_image is not None:
        pil_img = fallback_rgb_image.convert("RGB")
    if pil_img is None:
        return None

    offset_sec = max(0, int(position_ms // 1000))
    eff_dt: int | None = None
    if base_date_taken is not None and int(base_date_taken) > 0:
        eff_dt = int(base_date_taken) + offset_sec
    elif vpath.is_file():
        try:
            eff_dt = int(vpath.stat().st_mtime) + offset_sec
        except OSError:
            eff_dt = None

    dest_path = _next_still_frame_path(vpath, position_ms)
    exif_bytes = _build_exif_bytes(date_taken_ts=eff_dt, lat=lat, lon=lon)
    save_kwargs: dict[str, Any] = {
        "format": "JPEG",
        "quality": max(75, min(98, int(jpeg_quality))),
        "subsampling": 0,
    }
    if exif_bytes:
        save_kwargs["exif"] = exif_bytes

    pil_img.save(str(dest_path), **save_kwargs)
    w, h = pil_img.size
    return (dest_path, int(w), int(h), eff_dt)
