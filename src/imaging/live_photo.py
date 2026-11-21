"""
Apple Live Photos (.HEIC/.JPG + .MOV) & Android Motion Photos (Samsung/Pixel embedded MP4)
detection and extraction helpers.
Never hydrates Windows OneDrive cloud-only placeholders.
"""
from __future__ import annotations

import hashlib
import re
import struct
import tempfile
from pathlib import Path

from src.utils.paths import is_cloud_only_stat

_LIVE_STILL_EXTS = {"jpg", "jpeg", "heic", "heif"}
_MICRO_VIDEO_OFFSET_RE = re.compile(rb'MicroVideoOffset="(\d+)"')
_MOTION_PHOTO_RE = re.compile(rb'MotionPhoto="1"')
_CONTAINER_VIDEO_LEN_RE = re.compile(rb'Item:Mime="video/mp4"[^>]*?Item:Length="(\d+)"')


def detect_motion_photo_offset(file_path: Path | str) -> int | None:
    """
    Returns the byte offset of an embedded MP4 inside a Samsung/Pixel Motion Photo JPEG,
    or None if the file is a regular still image (or cloud-only placeholder).
    """
    p = Path(file_path)
    ext = p.suffix.lstrip(".").lower()
    if ext not in ("jpg", "jpeg"):
        return None
    try:
        st = p.stat(follow_symlinks=False)
    except OSError:
        return None
    if is_cloud_only_stat(st) or not p.is_file():
        return None

    file_size = int(st.st_size)
    if file_size < 128:
        return None

    try:
        with p.open("rb") as fh:
            head = fh.read(min(65536, file_size))

            # 1. Google Pixel XMP MicroVideoOffset (distance from EOF)
            m_off = _MICRO_VIDEO_OFFSET_RE.search(head)
            if m_off:
                tail_len = int(m_off.group(1))
                cand = file_size - tail_len
                if 0 < cand < file_size - 16:
                    fh.seek(cand)
                    box = fh.read(12)
                    if len(box) >= 8 and box[4:8] == b"ftyp":
                        return cand

            # 2. UltraHDR / MotionPhoto XMP Container Item:Length="N"
            if _MOTION_PHOTO_RE.search(head):
                m_len = _CONTAINER_VIDEO_LEN_RE.search(head)
                if m_len:
                    tail_len = int(m_len.group(1))
                    cand = file_size - tail_len
                    if 0 < cand < file_size - 16:
                        fh.seek(cand)
                        box = fh.read(12)
                        if len(box) >= 8 and box[4:8] == b"ftyp":
                            return cand

            # 3. Fast tail scan for Samsung MotionPhoto_Data / JPEG EOI (\xff\xd9) + ftyp MP4
            scan_window = min(file_size, 20 * 1024 * 1024)
            start_pos = file_size - scan_window
            fh.seek(start_pos)
            tail = fh.read(scan_window)

            # Search backwards for 'ftyp' atom
            search_end = len(tail)
            while True:
                idx = tail.rfind(b"ftyp", 4, search_end)
                if idx < 4:
                    break
                box_start = idx - 4
                box_size = struct.unpack(">I", tail[box_start : box_start + 4])[0]
                if 16 <= box_size <= 64 and box_start + box_size <= len(tail):
                    abs_offset = start_pos + box_start
                    # Verify either Samsung MotionPhoto_Data marker or JPEG EOI (\xff\xd9) right before
                    prefix = tail[max(0, box_start - 32) : box_start]
                    if b"\xff\xd9" in prefix or b"MotionPhoto_Data" in prefix:
                        return abs_offset
                search_end = idx
    except OSError:
        return None
    return None


def find_paired_live_video(image_path: Path | str) -> Path | None:
    """
    Checks for an Apple Live Photo companion .MOV file with the same stem in the same folder.
    """
    p = Path(image_path)
    if p.suffix.lstrip(".").lower() not in _LIVE_STILL_EXTS:
        return None
    for cand_ext in (".mov", ".MOV"):
        cand = p.with_suffix(cand_ext)
        try:
            st = cand.stat(follow_symlinks=False)
        except OSError:
            continue
        if not is_cloud_only_stat(st) and cand.is_file():
            return cand
    return None


def extract_motion_mp4_to_temp(image_path: Path | str, offset: int) -> Path | None:
    """
    Extracts an embedded Motion Photo MP4 starting at `offset` into a cached temp .mp4 file.
    """
    p = Path(image_path)
    if offset <= 0:
        return None
    try:
        st = p.stat(follow_symlinks=False)
    except OSError:
        return None
    if is_cloud_only_stat(st) or not p.is_file() or offset >= st.st_size:
        return None

    cache_dir = Path(tempfile.gettempdir()) / "gallery_live_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(f"{p}:{int(st.st_mtime)}:{offset}".encode("utf-8")).hexdigest()[:16]
    out_path = cache_dir / f"live_{digest}.mp4"
    if out_path.exists() and out_path.stat().st_size == (st.st_size - offset):
        return out_path

    try:
        with p.open("rb") as src, out_path.open("wb") as dst:
            src.seek(offset)
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        return out_path
    except OSError:
        return None
