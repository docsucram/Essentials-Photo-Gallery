"""
Read-only, on-demand Autofocus (AF) Area parser for Single Image View.
Never runs during folder scans and never opens Windows OneDrive cloud-only placeholders.
Results are cached in an LRU dict keyed by `(path, mtime, size)`.

Supports:
- Standard EXIF `SubjectArea` (0x9214) & `SubjectLocation` (0xA214)
- Nikon MakerNote `AFInfo` / `AFInfo2`
- Canon MakerNote `AFInfo` / `FocalPlane`
- Sony MakerNote `FocusLocation` (0x2027)
- Fujifilm MakerNote `FocusPixel` (0x1023)
"""
from __future__ import annotations

import os
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.utils.paths import is_cloud_only_stat


@dataclass(frozen=True)
class AfArea:
    """Normalized [0..1] autofocus box in display image coordinates."""
    x: float
    y: float
    width: float
    height: float
    focused: bool | None = True  # True = locked, False = active/unconfirmed, None = selected


_CACHE_CAPACITY = 64
_AF_CACHE: OrderedDict[tuple[str, int, int], list[AfArea]] = OrderedDict()


def _clamp_area(cx: float, cy: float, w: float, h: float, focused: bool | None = True) -> AfArea | None:
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return None
    w = max(0.025, min(0.5, float(w)))
    h = max(0.025, min(0.5, float(h)))
    x = max(0.0, min(1.0 - w, cx - w * 0.5))
    y = max(0.0, min(1.0 - h, cy - h * 0.5))
    return AfArea(x=x, y=y, width=w, height=h, focused=focused)


def _rotate_area(area: AfArea, orientation: int) -> AfArea:
    """Transforms normalized top-left AF box from raw sensor space into upright display orientation."""
    x, y, w, h = area.x, area.y, area.width, area.height
    if orientation == 6:  # 90° CW
        return AfArea(max(0.0, 1.0 - (y + h)), x, h, w, area.focused)
    if orientation == 8:  # 270° CW
        return AfArea(y, max(0.0, 1.0 - (x + w)), h, w, area.focused)
    if orientation == 3:  # 180°
        return AfArea(max(0.0, 1.0 - (x + w)), max(0.0, 1.0 - (y + h)), w, h, area.focused)
    return area


def _parse_tiff_ifd_tags(buf: bytes, tiff_start: int, ifd_offset: int, endian: str) -> dict[int, tuple[int, int, bytes]]:
    """Parses raw TIFF IFD entries into `{tag_id: (ftype, count, raw_value_or_offset_bytes)}`."""
    tags: dict[int, tuple[int, int, bytes]] = {}
    pos = tiff_start + ifd_offset
    if pos + 2 > len(buf):
        return tags
    num = struct.unpack(endian + "H", buf[pos : pos + 2])[0]
    if num == 0 or num > 256:
        return tags
    pos += 2
    for _ in range(num):
        if pos + 12 > len(buf):
            break
        tag, ftype, count = struct.unpack(endian + "HHI", buf[pos : pos + 8])
        val_bytes = buf[pos + 8 : pos + 12]
        tags[tag] = (ftype, count, val_bytes)
        pos += 12
    return tags


def _read_short_list(buf: bytes, tiff_start: int, endian: str, ftype: int, count: int, val_bytes: bytes) -> list[int]:
    if ftype != 3 or count <= 0 or count > 64:
        return []
    total = count * 2
    if total <= 4:
        raw = val_bytes[:total]
    else:
        off = struct.unpack(endian + "I", val_bytes)[0]
        abs_off = tiff_start + off
        if abs_off + total > len(buf):
            return []
        raw = buf[abs_off : abs_off + total]
    return list(struct.unpack(endian + f"{count}H", raw))


def _parse_makernote_af(
    make: str,
    maker_bytes: bytes,
    img_w: int,
    img_h: int,
) -> list[AfArea]:
    """Best-effort MakerNote AF box extraction for Nikon, Sony, and Fujifilm."""
    areas: list[AfArea] = []
    mk = (make or "").lower()

    # Fujifilm: FocusPixel (tag 0x1023) -> [x_px, y_px]
    if "fuji" in mk and maker_bytes.startswith(b"FUJIFILM"):
        if len(maker_bytes) >= 12:
            ifd_off = struct.unpack("<I", maker_bytes[8:12])[0]
            tags = _parse_tiff_ifd_tags(maker_bytes, 0, ifd_off, "<")
            if 0x1023 in tags:
                vals = _read_short_list(maker_bytes, 0, "<", *tags[0x1023])
                if len(vals) >= 2 and img_w > 0 and img_h > 0:
                    a = _clamp_area(vals[0] / float(img_w), vals[1] / float(img_h), 0.05, 0.05, True)
                    if a:
                        areas.append(a)
        return areas

    # Nikon MakerNote ("Nikon\x00\x02...")
    if "nikon" in mk and maker_bytes.startswith(b"Nikon\x00") and len(maker_bytes) >= 18:
        tiff_start = 10
        endian = "<" if maker_bytes[10:12] == b"II" else ">"
        ifd_off = struct.unpack(endian + "I", maker_bytes[14:18])[0]
        tags = _parse_tiff_ifd_tags(maker_bytes, tiff_start, ifd_off, endian)
        # Tag 0x0088 (AFInfo) or 0x00B7 (AFInfo2)
        if 0x00B7 in tags:
            ftype, count, val_bytes = tags[0x00B7]
            if count >= 20:
                off = struct.unpack(endian + "I", val_bytes)[0]
                abs_off = tiff_start + off
                if abs_off + 20 <= len(maker_bytes):
                    af = maker_bytes[abs_off : abs_off + count]
                    # Nikon AFInfo2: [0:4] version, [4] contrastDetectAF, [5] afAreaMode, ...
                    # When contrast/phase point coords exist at bytes [16..20] (image_w, image_h) or [10..14]
                    if len(af) >= 20:
                        aw, ah, ax, ay = struct.unpack(endian + "HHHH", af[12:20])
                        if aw > 0 and ah > 0 and ax <= aw and ay <= ah:
                            a = _clamp_area(ax / float(aw), ay / float(ah), 0.05, 0.05, True)
                            if a:
                                areas.append(a)
        return areas

    # Sony MakerNote ("SONY DSC \x00\x00\x00")
    if "sony" in mk and len(maker_bytes) >= 24:
        tiff_start = 12 if maker_bytes.startswith(b"SONY ") else 0
        if tiff_start < len(maker_bytes):
            tags = _parse_tiff_ifd_tags(maker_bytes, tiff_start, 0 if tiff_start else 0, "<")
            # Tag 0x2027 FocusLocation: [img_w, img_h, fx, fy]
            if 0x2027 in tags:
                vals = _read_short_list(maker_bytes, tiff_start, "<", *tags[0x2027])
                if len(vals) >= 4 and vals[0] > 0 and vals[1] > 0:
                    a = _clamp_area(vals[2] / float(vals[0]), vals[3] / float(vals[1]), 0.05, 0.05, True)
                    if a:
                        areas.append(a)
        return areas

    return areas


def _extract_uncached(file_path: Path) -> list[AfArea]:
    areas: list[AfArea] = []
    try:
        with Image.open(file_path) as im:
            raw_w, raw_h = im.size
            exif = im.getexif()
            if not exif:
                return []
            orient = int(exif.get(0x0112) or 1)
            make = str(exif.get(0x010F) or "")
            ifd = exif.get_ifd(0x8769)  # ExifIFD

            # 1. Standard EXIF SubjectArea (0x9214) / SubjectLocation (0xA214)
            subj = ifd.get(0x9214) or ifd.get(0xA214)
            if isinstance(subj, (tuple, list)) and raw_w > 0 and raw_h > 0:
                if len(subj) == 2:
                    a = _clamp_area(subj[0] / float(raw_w), subj[1] / float(raw_h), 0.06, 0.06, True)
                    if a:
                        areas.append(a)
                elif len(subj) == 3:
                    d = subj[2] / float(max(raw_w, raw_h))
                    a = _clamp_area(subj[0] / float(raw_w), subj[1] / float(raw_h), d, d, True)
                    if a:
                        areas.append(a)
                elif len(subj) >= 4:
                    a = _clamp_area(
                        subj[0] / float(raw_w),
                        subj[1] / float(raw_h),
                        subj[2] / float(raw_w),
                        subj[3] / float(raw_h),
                        True,
                    )
                    if a:
                        areas.append(a)

            # 2. MakerNote AF areas
            if not areas:
                maker_raw = ifd.get(0x927C)
                if isinstance(maker_raw, (bytes, bytearray)):
                    areas.extend(_parse_makernote_af(make, bytes(maker_raw), raw_w, raw_h))

            if orient in (3, 6, 8):
                areas = [_rotate_area(a, orient) for a in areas]
    except Exception:
        return []
    return areas


def get_af_areas(file_path: Path | str) -> list[AfArea]:
    """
    On-demand, LRU-cached Autofocus area lookup.
    Returns empty list for missing or Windows cloud-only placeholder files without hydrating.
    """
    p = Path(file_path)
    try:
        st = p.stat(follow_symlinks=False)
    except OSError:
        return []
    if is_cloud_only_stat(st) or not p.is_file():
        return []

    key = (str(p), int(st.st_mtime), int(st.st_size))
    cached = _AF_CACHE.get(key)
    if cached is not None:
        _AF_CACHE.move_to_end(key)
        return list(cached)

    areas = _extract_uncached(p)
    _AF_CACHE[key] = list(areas)
    while len(_AF_CACHE) > _CACHE_CAPACITY:
        _AF_CACHE.popitem(last=False)
    return areas
