"""
Bounded TIFF/IFD Embedded JPEG Preview Extractor for Camera RAW files.
Reads only the bounded TIFF directory chain (<= 1 MiB) and seeks straight to the
best embedded JPEG slice (`JPEGInterchangeFormat` / `JPEGInterchangeFormatLength`),
avoiding full LibRaw (`rawpy.imread`) container parsing for fast grid thumbnails
on `.nef`, `.arw`, `.cr2`, `.dng`, `.orf`, `.rw2`, `.pef`, and `.srw` files.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

_MAX_METADATA_BYTES = 1024 * 1024  # 1 MiB cap for IFD header traversal
_MAX_JPEG_BYTES = 32 * 1024 * 1024
_MAX_IFDS = 12

_TAG_ORIENTATION = 0x0112
_TAG_SUBIFDS = 0x014A
_TAG_JPEG_OFFSET = 0x0201
_TAG_JPEG_LENGTH = 0x0202

# TIFF type sizes in bytes
_TYPE_SIZES = {
    1: 1,   # BYTE
    2: 1,   # ASCII
    3: 2,   # SHORT
    4: 4,   # LONG
    5: 8,   # RATIONAL
    7: 1,   # UNDEFINED
    9: 4,   # SLONG
    10: 8,  # SRATIONAL
    13: 4,  # IFD
}

_TIFF_RAW_EXTS = frozenset({"nef", "nrw", "arw", "sr2", "srf", "cr2", "dng", "orf", "rw2", "pef", "srw"})


def _parse_jpeg_sof_dimensions(header_bytes: bytes) -> tuple[int, int] | None:
    """Parses (width, height) from a JPEG SOF marker in a small header buffer without decoding pixels."""
    n = len(header_bytes)
    if n < 4 or header_bytes[0] != 0xFF or header_bytes[1] != 0xD8:
        return None
    pos = 2
    while pos + 3 < n:
        if header_bytes[pos] != 0xFF:
            pos += 1
            continue
        while pos < n and header_bytes[pos] == 0xFF:
            pos += 1
        if pos >= n:
            break
        marker = header_bytes[pos]
        pos += 1
        if marker in (0xD8, 0xD9, 0x01) or (0xD0 <= marker <= 0xD7):
            continue
        if pos + 2 > n:
            break
        seg_len = struct.unpack(">H", header_bytes[pos : pos + 2])[0]
        if seg_len < 2:
            break
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if pos + 7 <= n:
                height, width = struct.unpack(">HH", header_bytes[pos + 3 : pos + 7])
                if width > 0 and height > 0:
                    return int(width), int(height)
            return None
        if marker == 0xDA:  # Start of Scan
            break
        pos += seg_len
    return None


def _read_u32_values(
    f, endian: str, field_type: int, count: int, val_bytes: bytes, file_len: int
) -> list[int]:
    type_size = _TYPE_SIZES.get(field_type, 0)
    if type_size == 0 or count <= 0 or count > 16:
        return []
    total = type_size * count
    if total <= 4:
        raw = val_bytes[:total]
    else:
        offset = struct.unpack(endian + "I", val_bytes)[0]
        if offset + total > file_len:
            return []
        f.seek(offset)
        raw = f.read(total)
        if len(raw) < total:
            return []
    if field_type == 3:  # SHORT
        return list(struct.unpack(endian + f"{count}H", raw))
    if field_type in (4, 13):  # LONG / IFD
        return list(struct.unpack(endian + f"{count}I", raw))
    return []


def extract_tiff_embedded_jpeg(
    file_path: Path | str, target_size: int = 256
) -> tuple[bytes, int] | None:
    """
    Extracts the best-fitting embedded JPEG preview `(jpeg_bytes, orientation)` from a
    classic TIFF-based camera RAW file without invoking LibRaw.
    Returns `None` if the file is not a classic TIFF container or has no suitable JPEG preview.
    """
    p = Path(file_path)
    ext = p.suffix.lstrip(".").lower()
    if ext not in _TIFF_RAW_EXTS:
        return None

    try:
        file_len = p.stat().st_size
    except OSError:
        return None
    if file_len < 16:
        return None

    try:
        with open(p, "rb") as f:
            hdr = f.read(8)
            if len(hdr) < 8:
                return None
            if hdr[:4] == b"II\x2a\x00":
                endian = "<"
            elif hdr[:4] == b"MM\x00\x2a":
                endian = ">"
            else:
                return None

            first_ifd = struct.unpack(endian + "I", hdr[4:8])[0]
            if first_ifd == 0 or first_ifd >= file_len:
                return None

            queue: list[int] = [first_ifd]
            visited: set[int] = set()
            candidates: list[tuple[int, int, int, int, int]] = []  # (offset, length, w, h, orient)
            root_orientation = 1

            while queue and len(visited) < _MAX_IFDS:
                ifd_offset = queue.pop(0)
                if ifd_offset == 0 or ifd_offset in visited or ifd_offset + 2 > file_len:
                    continue
                if ifd_offset > min(file_len, _MAX_METADATA_BYTES * 8):
                    continue
                visited.add(ifd_offset)

                f.seek(ifd_offset)
                cnt_raw = f.read(2)
                if len(cnt_raw) < 2:
                    continue
                num_entries = struct.unpack(endian + "H", cnt_raw)[0]
                if num_entries == 0 or num_entries > 512:
                    continue

                entries_bytes = f.read(num_entries * 12 + 4)
                if len(entries_bytes) < num_entries * 12:
                    continue

                jpg_offset: int | None = None
                jpg_length: int | None = None
                ifd_orient: int | None = None

                for idx in range(num_entries):
                    entry = entries_bytes[idx * 12 : (idx + 1) * 12]
                    tag, ftype, count = struct.unpack(endian + "HHI", entry[:8])
                    val_bytes = entry[8:12]
                    if tag == _TAG_ORIENTATION:
                        vals = _read_u32_values(f, endian, ftype, count, val_bytes, file_len)
                        if vals and 1 <= vals[0] <= 8:
                            ifd_orient = vals[0]
                            if ifd_offset == first_ifd:
                                root_orientation = vals[0]
                    elif tag == _TAG_JPEG_OFFSET:
                        vals = _read_u32_values(f, endian, ftype, count, val_bytes, file_len)
                        if vals:
                            jpg_offset = vals[0]
                    elif tag == _TAG_JPEG_LENGTH:
                        vals = _read_u32_values(f, endian, ftype, count, val_bytes, file_len)
                        if vals:
                            jpg_length = vals[0]
                    elif tag == _TAG_SUBIFDS:
                        sub_offsets = _read_u32_values(f, endian, ftype, count, val_bytes, file_len)
                        for sub_off in sub_offsets:
                            if sub_off and sub_off not in visited:
                                queue.append(sub_off)

                # Follow next IFD pointer in chain
                if len(entries_bytes) >= num_entries * 12 + 4:
                    next_ifd = struct.unpack(
                        endian + "I",
                        entries_bytes[num_entries * 12 : num_entries * 12 + 4],
                    )[0]
                    if next_ifd and next_ifd not in visited:
                        queue.append(next_ifd)

                if (
                    jpg_offset is not None
                    and jpg_length is not None
                    and 256 <= jpg_length <= _MAX_JPEG_BYTES
                    and jpg_offset + jpg_length <= file_len
                ):
                    f.seek(jpg_offset)
                    probe = f.read(min(jpg_length, 65536))
                    dims = _parse_jpeg_sof_dimensions(probe)
                    if dims is not None:
                        w, h = dims
                        orient = ifd_orient if ifd_orient is not None else root_orientation
                        candidates.append((jpg_offset, jpg_length, w, h, orient))

            if not candidates:
                return None

            # Pick the smallest embedded preview whose long edge satisfies target_size,
            # or fall back to the largest available preview if all are smaller.
            qualifying = [c for c in candidates if max(c[2], c[3]) >= target_size]
            if qualifying:
                chosen = min(qualifying, key=lambda c: (max(c[2], c[3]), c[1]))
            else:
                chosen = max(candidates, key=lambda c: (max(c[2], c[3]), c[1]))

            off, length, _w, _h, orient = chosen
            f.seek(off)
            jpeg_bytes = f.read(length)
            if len(jpeg_bytes) >= 4 and jpeg_bytes[:2] == b"\xff\xd8":
                return jpeg_bytes, orient
    except OSError:
        return None
    return None
