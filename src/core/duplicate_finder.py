"""
PhotoGallery 64 — Core Duplicate & Burst Detection Engine
Provides high-speed exact byte hashing, multi-index perceptual hashing (dHash/pHash),
and burst sequence analysis with sharpness scoring and smart auto-recommendation rules.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass


@dataclass
class DuplicateMediaItem:
    media_id: int
    folder_path: str
    filename: str
    file_size: int
    width: int
    height: int
    mtime: float
    date_taken: float
    rating: int = 0
    flag: int = 0
    caption: str = ""
    people_names: list[str] = field(default_factory=list)
    tag_names: list[str] = field(default_factory=list)
    is_video: bool = False
    phash: int = 0
    dhash: int = 0
    sharpness: float = 0.0
    face_sharpness: float = 0.0
    has_faces: bool = False
    lab_haar: bytes = b""
    similarity_score: float = 0.0
    is_selected_for_removal: bool = False
    is_recommended_keep: bool = False

    @property
    def full_path(self) -> Path:
        return Path(self.folder_path) / self.filename

    @property
    def megapixels(self) -> float:
        if self.width <= 0 or self.height <= 0:
            return 0.0
        return round((self.width * self.height) / 1_000_000.0, 2)

    @property
    def formatted_size(self) -> str:
        s = float(self.file_size)
        if s >= 1024 * 1024 * 1024:
            return f"{s / (1024 ** 3):.2f} GB"
        if s >= 1024 * 1024:
            return f"{s / (1024 ** 2):.1f} MB"
        if s >= 1024:
            return f"{s / 1024:.0f} KB"
        return f"{int(s)} B"

    @property
    def metadata_score(self) -> int:
        score = len(self.people_names) * 3 + len(self.tag_names) * 2 + self.rating
        if self.caption:
            score += 2
        if self.flag:
            score += 1
        return score


@dataclass
class DuplicateGroup:
    group_id: str
    match_type: str  # "exact", "visual", "burst"
    similarity: float  # 0.0 .. 1.0 (1.0 = 100%)
    items: list[DuplicateMediaItem]
    time_span_sec: float = 0.0

    @property
    def reclaimable_bytes(self) -> int:
        return sum(it.file_size for it in self.items if it.is_selected_for_removal)

    @property
    def formatted_reclaimable_size(self) -> str:
        s = float(self.reclaimable_bytes)
        if s >= 1024 * 1024 * 1024:
            return f"{s / (1024 ** 3):.2f} GB"
        if s >= 1024 * 1024:
            return f"{s / (1024 ** 2):.1f} MB"
        if s >= 1024:
            return f"{s / 1024:.0f} KB"
        return f"{int(s)} B"


def compute_image_dhash(img_gray: np.ndarray) -> int:
    """Computes a 64-bit difference hash (dHash) from a 2D grayscale image array."""
    if img_gray is None or img_gray.size == 0:
        return 0
    resized = cv2.resize(img_gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return int.from_bytes(np.packbits(diff.flatten()), "big")


def compute_image_phash(img_gray: np.ndarray) -> int:
    """Computes a 64-bit perceptual hash (pHash / DCT) from a 2D grayscale image array."""
    if img_gray is None or img_gray.size == 0:
        return 0
    resized = cv2.resize(img_gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    dct_low = dct[:8, :8]
    # Exclude DC component (0, 0)
    flat = dct_low.flatten()
    median_val = float(np.median(flat[1:]))
    bits = dct_low > median_val
    return int.from_bytes(np.packbits(bits.flatten()), "big")


def compute_sharpness_score(img_gray: np.ndarray) -> float:
    """Computes Laplacian gradient variance as an indicator of focus sharpness."""
    if img_gray is None or img_gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def compute_face_eye_sharpness(
    img_gray: np.ndarray,
    face_rects: Sequence[tuple[float, float, float, float]],
) -> tuple[float, bool]:
    """Computes focus sharpness + eye-band contrast inside detected face boxes.

    Returns (score, has_valid_face). Focuses on the upper 60% of each face box
    (eye/brow band) so background sharpness does not fool portrait burst culling.
    """
    if img_gray is None or img_gray.size == 0 or not face_rects:
        return 0.0, False
    h, w = img_gray.shape[:2]
    best_score = 0.0
    valid = False
    for rx, ry, rw, rh in face_rects:
        if rw <= 0.01 or rh <= 0.01:
            continue
        x0 = max(0, min(w - 1, int((rx + 0.10 * rw) * w)))
        x1 = max(x0 + 4, min(w, int((rx + 0.90 * rw) * w)))
        y0 = max(0, min(h - 1, int((ry + 0.14 * rh) * h)))
        y1 = max(y0 + 4, min(h, int((ry + 0.58 * rh) * h)))
        roi = img_gray[y0:y1, x0:x1]
        if roi.size < 16:
            continue
        valid = True
        lap_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        eye_edge = float(np.mean(np.abs(sobel_y)))
        score = lap_var * (1.0 + min(0.35, eye_edge / 60.0))
        if score > best_score:
            best_score = score
    return best_score, valid


def compute_lab_haar_similarity(sig_a: bytes, sig_b: bytes) -> float:
    """Computes 0.0..1.0 spatial color+composition similarity between two 24-byte LAB Haar signatures."""
    if not sig_a or not sig_b or len(sig_a) != 24 or len(sig_b) != 24:
        return 0.0
    a = np.frombuffer(sig_a, dtype=np.uint8).astype(np.float32).reshape(8, 3)
    b = np.frombuffer(sig_b, dtype=np.uint8).astype(np.float32).reshape(8, 3)
    diff = np.abs(a - b)
    weighted = diff[:, 0] * 0.26 + diff[:, 1] * 0.37 + diff[:, 2] * 0.37
    mean_diff = float(np.mean(weighted))
    return max(0.0, min(1.0, 1.0 - (mean_diff / 75.0)))


def compute_lab_palette_similarity(sig_a: bytes, sig_b: bytes) -> float:
    """
    Computes 0.0..1.0 position-invariant color palette similarity between two 24-byte
    LAB signatures (symmetric Chamfer distance over the 8 spatial cells), so shifted or
    cropped shots with the same distinctive colors still match strongly.
    """
    if not sig_a or not sig_b or len(sig_a) != 24 or len(sig_b) != 24:
        return 0.0
    a = np.frombuffer(sig_a, dtype=np.uint8).astype(np.float32).reshape(8, 3)
    b = np.frombuffer(sig_b, dtype=np.uint8).astype(np.float32).reshape(8, 3)
    # Pairwise 8x8 distance emphasizing chromaticity (a*, b*) over brightness (L*)
    diff = np.abs(a[:, None, :] - b[None, :, :])
    dist_mat = diff[:, :, 0] * 0.15 + diff[:, :, 1] * 0.425 + diff[:, :, 2] * 0.425
    min_ab = float(np.mean(np.min(dist_mat, axis=1)))
    min_ba = float(np.mean(np.min(dist_mat, axis=0)))
    sym_dist = 0.5 * (min_ab + min_ba)
    return max(0.0, min(1.0, 1.0 - (sym_dist / 30.0)))


def compute_lab_haar_signature(img_bgr: np.ndarray) -> bytes:
    """
    Computes a compact 24-byte spatial CIE LAB signature (4x2 grid x 3 channels)
    for fast digiKam-style visual similarity & color matching.
    """
    if img_bgr is None or img_bgr.size == 0:
        return b""
    small = cv2.resize(img_bgr, (4, 2), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    return lab.astype(np.uint8).tobytes()


def compute_signatures_from_qimage(qimg: Any) -> tuple[int, int, float, bytes] | None:
    """
    Extracts `(phash64, dhash64, sharpness, lab_haar)` directly from an in-memory
    `QImage` (no extra disk read or WebP re-decode), returning None on invalid/tiny inputs.
    """
    if qimg is None or qimg.isNull() or qimg.width() < 16 or qimg.height() < 16:
        return None
    from PySide6.QtGui import QImage

    rgb = (
        qimg
        if qimg.format() == QImage.Format.Format_RGB888
        else qimg.convertToFormat(QImage.Format.Format_RGB888)
    )
    w, h = rgb.width(), rgb.height()
    bpl = rgb.bytesPerLine()
    raw = bytes(rgb.constBits())
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size < h * bpl:
        return None
    rgb_arr = arr[: h * bpl].reshape((h, bpl))[:, : w * 3].reshape((h, w, 3))
    bgr = np.ascontiguousarray(rgb_arr[:, :, ::-1])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    phash64 = compute_image_phash(gray)
    dhash64 = compute_image_dhash(gray)
    sharpness = compute_sharpness_score(gray)
    lab_haar = compute_lab_haar_signature(bgr)
    return (phash64, dhash64, sharpness, lab_haar)


def hamming_distance(h1: int, h2: int) -> int:
    """Returns the bitwise Hamming distance between two 64-bit integer hashes."""
    return (h1 ^ h2).bit_count()


def similarity_from_distance(dist: int) -> float:
    """Converts 64-bit Hamming distance to similarity ratio (0.0 to 1.0)."""
    return max(0.0, 1.0 - (dist / 64.0))


def fast_file_hash(path: Path | str, chunk_size: int = 8192) -> str | None:
    """Computes a fast partial hash of file header + tail + size for O(1) exact candidate grouping."""
    from src.utils.paths import is_cloud_only_stat

    p = Path(path)
    try:
        st = p.stat(follow_symlinks=False)
    except OSError:
        return None
    if is_cloud_only_stat(st) or not p.is_file():
        return None
    try:
        size = st.st_size
        if size == 0:
            return "empty"
        h = hashlib.md5()
        h.update(str(size).encode("ascii"))
        with open(p, "rb") as f:
            h.update(f.read(chunk_size))
            if size > chunk_size * 2:
                f.seek(-chunk_size, 2)
                h.update(f.read(chunk_size))
        return h.hexdigest()
    except Exception:
        return None


def full_file_hash(path: Path | str) -> str | None:
    """Computes full SHA-256 hash of file for 100% exact verification."""
    from src.utils.paths import is_cloud_only_file

    p = Path(path)
    if is_cloud_only_file(p) or not p.is_file():
        return None
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


class DuplicateDetectionEngine:
    """
    High-speed duplicate detection engine operating on CatalogRepository records
    and ThumbnailRepository WebP blobs.
    """

    def __init__(
        self,
        catalog_repo: Any,
        thumb_repo: Any,
        semantic_repo: Any = None,
        semantic_engine: Any = None,
    ) -> None:
        self.catalog_repo = catalog_repo
        self.thumb_repo = thumb_repo
        self.semantic_repo = semantic_repo
        self.semantic_engine = semantic_engine
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def find_all(
        self,
        media_records: Sequence[dict[str, Any]] | None = None,
        find_exact: bool = True,
        find_visual: bool = True,
        find_bursts: bool = True,
        similarity_threshold: float = 0.92,
        burst_time_window_sec: float = 2.5,
        progress_cb: Callable[[int, int, str], None] | None = None,
        group_found_cb: Callable[[DuplicateGroup], None] | None = None,
    ) -> list[DuplicateGroup]:
        """
        Runs the multi-stage duplicate and burst detection pipeline.
        Returns a list of DuplicateGroup objects sorted by reclaimable space descending.
        """
        self._is_cancelled = False
        all_groups: list[DuplicateGroup] = []
        handled_exact_mids: set[int] = set()

        # 1. Load media items
        if media_records is None:
            if progress_cb:
                progress_cb(0, 100, "Loading catalog media records...")
            media_records = self._fetch_all_catalog_records()

        if not media_records:
            return []

        items_dict = self._build_items_dict(media_records)
        all_items = list(items_dict.values())
        total_items = len(all_items)

        # -----------------------------------------------------------------
        # Stage 1: Exact Byte Duplicates (O(N) file size bucketing + hash)
        # -----------------------------------------------------------------
        if find_exact and not self._is_cancelled:
            if progress_cb:
                progress_cb(5, 100, "Scanning for exact byte-for-byte duplicates...")
            exact_groups = self._find_exact_duplicates(all_items, progress_cb, group_found_cb)
            for g in exact_groups:
                all_groups.append(g)
                for it in g.items:
                    handled_exact_mids.add(it.media_id)

        # -----------------------------------------------------------------
        # Stage 2: Photo Bursts (Grouped by time delta <= window)
        # -----------------------------------------------------------------
        if find_bursts and not self._is_cancelled:
            if progress_cb:
                progress_cb(30, 100, "Analyzing photo burst sequences...")
            burst_groups = self._find_burst_sequences(
                all_items,
                max_time_diff=burst_time_window_sec,
                exclude_mids=handled_exact_mids,
                progress_cb=progress_cb,
                group_found_cb=group_found_cb,
            )
            for g in burst_groups:
                all_groups.append(g)

        # -----------------------------------------------------------------
        # Stage 3: Perceptual / Visual Duplicates (dHash + pHash)
        # -----------------------------------------------------------------
        if find_visual and not self._is_cancelled:
            if progress_cb:
                progress_cb(50, 100, "Computing perceptual hashes and visual matches...")
            visual_groups = self._find_visual_duplicates(
                all_items,
                exclude_mids=handled_exact_mids,
                similarity_threshold=similarity_threshold,
                progress_cb=progress_cb,
                group_found_cb=group_found_cb,
            )
            for g in visual_groups:
                all_groups.append(g)

        # Apply default auto-recommendation rules (Highest Resolution / Best Quality)
        auto_recommend_rules(all_groups, rule="highest_resolution")

        if progress_cb:
            progress_cb(100, 100, f"Scan complete: found {len(all_groups)} duplicate/burst groups.")

        # Sort groups: largest recoverable bytes first
        all_groups.sort(key=lambda g: g.reclaimable_bytes, reverse=True)
        return all_groups

    def _fetch_all_catalog_records(self) -> list[dict[str, Any]]:
        conn = self.catalog_repo.engine.catalog_conn()
        cur = conn.cursor()
        cur.row_factory = None
        sql = """
        SELECT m.id, f.path, m.filename, m.size, m.width, m.height, m.mtime, m.date_taken,
               m.rating, m.flag, m.caption, m.is_video
        FROM media m
        JOIN folders f ON m.folder_id = f.id
        WHERE m.is_missing = 0;
        """
        rows = cur.execute(sql).fetchall()
        records = []
        for r in rows:
            records.append({
                "id": r[0],
                "folder_path": r[1],
                "filename": r[2],
                "size": r[3],
                "width": r[4] or 0,
                "height": r[5] or 0,
                "mtime": r[6] or 0.0,
                "date_taken": r[7] or 0.0,
                "rating": r[8] or 0,
                "flag": r[9] or 0,
                "caption": r[10] or "",
                "is_video": bool(r[11]),
            })
        return records

    def _build_items_dict(self, records: Sequence[Any]) -> dict[int, DuplicateMediaItem]:
        items: dict[int, DuplicateMediaItem] = {}
        folder_map: dict[int, str] | None = None

        def _resolve_folder_path(fid_raw: Any) -> str:
            nonlocal folder_map
            if not fid_raw or self.catalog_repo is None:
                return ""
            if folder_map is None:
                try:
                    conn = self.catalog_repo.engine.catalog_conn()
                    folder_map = {
                        int(row[0]): str(row[1] or "")
                        for row in conn.execute("SELECT id, path FROM folders;").fetchall()
                    }
                except Exception:
                    folder_map = {}
            return folder_map.get(int(fid_raw), "")

        for r in records:
            if isinstance(r, dict):
                mid = int(r["id"])
                folder_path = str(r.get("folder_path") or "")
                if not folder_path and r.get("folder_id"):
                    folder_path = _resolve_folder_path(r.get("folder_id"))
                filename = str(r.get("filename") or "")
                file_size = int(r.get("size") or r.get("file_size") or 0)
                width = int(r.get("width") or 0)
                height = int(r.get("height") or 0)
                mtime = float(r.get("mtime") or 0.0)
                date_taken = float(r.get("date_taken") or 0.0)
                rating = int(r.get("rating") or 0)
                flag = int(r.get("flag") or 0)
                caption = str(r.get("caption") or "")
                is_video = bool(r.get("is_video"))
            else:
                mid = int(getattr(r, "id"))
                folder_path = str(getattr(r, "folder_path", "") or "")
                if not folder_path:
                    folder_path = _resolve_folder_path(getattr(r, "folder_id", None))
                filename = str(getattr(r, "filename", "") or "")
                file_size = int(getattr(r, "size", 0) or getattr(r, "file_size", 0) or 0)
                width = int(getattr(r, "width", 0) or 0)
                height = int(getattr(r, "height", 0) or 0)
                mtime = float(getattr(r, "mtime", 0.0) or 0.0)
                date_taken = float(getattr(r, "date_taken", 0.0) or 0.0)
                rating = int(getattr(r, "rating", 0) or 0)
                flag = int(getattr(r, "flag", 0) or 0)
                caption = str(getattr(r, "caption", "") or "")
                is_video = bool(getattr(r, "is_video", False))
            items[mid] = DuplicateMediaItem(
                media_id=mid,
                folder_path=folder_path,
                filename=filename,
                file_size=file_size,
                width=width,
                height=height,
                mtime=mtime,
                date_taken=date_taken,
                rating=rating,
                flag=flag,
                caption=caption,
                is_video=is_video,
            )
        return items

    def _find_exact_duplicates(
        self,
        items: list[DuplicateMediaItem],
        progress_cb: Any = None,
        group_found_cb: Any = None,
    ) -> list[DuplicateGroup]:
        size_buckets: dict[int, list[DuplicateMediaItem]] = {}
        for it in items:
            if self._is_cancelled:
                return []
            if it.file_size > 0:
                size_buckets.setdefault(it.file_size, []).append(it)

        candidate_buckets = [b for b in size_buckets.values() if len(b) > 1]
        groups: list[DuplicateGroup] = []
        group_idx = 1
        total_buckets = len(candidate_buckets)

        for b_idx, bucket in enumerate(candidate_buckets, start=1):
            if self._is_cancelled:
                break
            if progress_cb:
                pct = int(5 + (b_idx / max(1, total_buckets)) * 25)
                progress_cb(pct, 100, f"Phase 1/3: Checking exact byte matches ({b_idx}/{total_buckets} candidate groups)...")

            # Compute partial fast hash
            hash_buckets: dict[str, list[DuplicateMediaItem]] = {}
            for it in bucket:
                if self._is_cancelled:
                    break
                ph = fast_file_hash(it.full_path)
                if ph:
                    hash_buckets.setdefault(ph, []).append(it)

            for cand_items in hash_buckets.values():
                if self._is_cancelled:
                    break
                if len(cand_items) < 2:
                    continue
                # Full hash verification
                full_buckets: dict[str, list[DuplicateMediaItem]] = {}
                for it in cand_items:
                    if self._is_cancelled:
                        break
                    fh = full_file_hash(it.full_path)
                    if fh:
                        full_buckets.setdefault(fh, []).append(it)

                for exact_matches in full_buckets.values():
                    if len(exact_matches) >= 2:
                        grp = DuplicateGroup(
                            group_id=f"exact_{group_idx}",
                            match_type="exact",
                            similarity=1.0,
                            items=list(exact_matches),
                        )
                        auto_recommend_rules([grp], rule="highest_resolution")
                        groups.append(grp)
                        group_idx += 1
                        if group_found_cb:
                            group_found_cb(grp)
        return groups

    def _find_burst_sequences(
        self,
        items: list[DuplicateMediaItem],
        max_time_diff: float = 2.5,
        exclude_mids: set[int] | None = None,
        progress_cb: Any = None,
        group_found_cb: Any = None,
    ) -> list[DuplicateGroup]:
        exclude = exclude_mids or set()
        candidates = [
            it for it in items
            if not it.is_video
            and it.media_id not in exclude
            and it.date_taken > 0
            and it.width > 0
            and it.height > 0
        ]
        candidates.sort(key=lambda it: (it.folder_path, it.date_taken, it.filename))

        groups: list[DuplicateGroup] = []
        group_idx = 1
        n = len(candidates)
        i = 0

        while i < n:
            if self._is_cancelled:
                break
            if progress_cb and i % 15 == 0:
                pct = int(30 + (i / max(1, n)) * 20)
                progress_cb(pct, 100, f"Phase 2/3: Finding burst photo sequences ({i}/{n} photos)...")

            seq = [candidates[i]]
            t_prev = candidates[i].date_taken
            j = i + 1
            while j < n:
                if self._is_cancelled:
                    break
                it_next = candidates[j]
                if it_next.folder_path != seq[0].folder_path:
                    break
                delta = it_next.date_taken - t_prev
                if 0 <= delta <= max_time_diff:
                    if (it_next.width, it_next.height) == (seq[0].width, seq[0].height):
                        seq.append(it_next)
                    t_prev = it_next.date_taken
                    j += 1
                else:
                    break

            if len(seq) >= 2:
                signed_seq = self._populate_visual_signatures(seq)
                signed_map = {it.media_id: it for it in signed_seq}
                plausible_seq: list[DuplicateMediaItem] = [signed_map.get(seq[0].media_id, seq[0])]
                for cand in seq[1:]:
                    cur = signed_map.get(cand.media_id, cand)
                    prev = plausible_seq[-1]
                    if (prev.dhash or prev.lab_haar) and (cur.dhash or cur.lab_haar):
                        d_dist = hamming_distance(prev.dhash, cur.dhash)
                        haar_s = compute_lab_haar_similarity(prev.lab_haar, cur.lab_haar)
                        if d_dist > 26 and haar_s < 0.55:
                            continue
                    plausible_seq.append(cur)

                if len(plausible_seq) >= 2:
                    self._enrich_burst_face_sharpness(plausible_seq)
                    for it in plausible_seq:
                        if it.sharpness <= 0.0:
                            it.sharpness = self._get_item_sharpness(it)

                    t_start = plausible_seq[0].date_taken
                    t_end = plausible_seq[-1].date_taken
                    span = max(0.1, t_end - t_start)

                    grp = DuplicateGroup(
                        group_id=f"burst_{group_idx}",
                        match_type="burst",
                        similarity=0.98,
                        items=plausible_seq,
                        time_span_sec=round(span, 1),
                    )
                    auto_recommend_rules([grp], rule="highest_resolution")
                    groups.append(grp)
                    group_idx += 1
                    if group_found_cb:
                        group_found_cb(grp)
                i = j
            else:
                i += 1

        return groups

    def _populate_visual_signatures(
        self,
        cands: list[DuplicateMediaItem],
        progress_cb: Any = None,
    ) -> list[DuplicateMediaItem]:
        """Load cached pHash/dHash/sharpness/lab_haar signatures and compute missing/stale entries in chunks."""
        mids = [it.media_id for it in cands]
        cached_sigs = (
            self.thumb_repo.get_signatures_batch(mids)
            if (self.thumb_repo and mids and hasattr(self.thumb_repo, "get_signatures_batch"))
            else {}
        )
        uncached_cands: list[DuplicateMediaItem] = []
        valid_cands: list[DuplicateMediaItem] = []
        for it in cands:
            sig = cached_sigs.get(it.media_id)
            if sig is not None and sig[0] == int(it.mtime) and (sig[1] != 0 or sig[2] != 0) and sig[4]:
                it.phash = sig[1]
                it.dhash = sig[2]
                if sig[3] > 0.0:
                    it.sharpness = sig[3]
                it.lab_haar = sig[4]
                valid_cands.append(it)
            else:
                uncached_cands.append(it)

        total_uncached = len(uncached_cands)
        chunk_size = 200
        for chunk_start in range(0, total_uncached, chunk_size):
            if self._is_cancelled:
                break
            chunk_items = uncached_cands[chunk_start : chunk_start + chunk_size]
            chunk_mids = [it.media_id for it in chunk_items]
            blobs = self.thumb_repo.get_webp_blobs_batch(chunk_mids) if (self.thumb_repo and chunk_mids) else {}
            to_persist: list[tuple[int, int, int, int, float, bytes]] = []
            for offset, it in enumerate(chunk_items, start=1):
                if self._is_cancelled:
                    break
                idx = chunk_start + offset
                if progress_cb and idx % 20 == 0:
                    pct = int(50 + (idx / max(1, total_uncached)) * 38)
                    progress_cb(pct, 100, f"Computing visual signatures ({idx}/{total_uncached} uncached photos)...")

                arr_bgr = None
                blob = blobs.get(it.media_id)
                if blob:
                    arr_bgr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                elif it.full_path.exists():
                    from src.utils.paths import is_cloud_only_file
                    if not is_cloud_only_file(it.full_path):
                        try:
                            arr_bgr = cv2.imread(str(it.full_path), cv2.IMREAD_REDUCED_COLOR_8)
                            if arr_bgr is None or arr_bgr.size == 0:
                                arr_bgr = cv2.imread(str(it.full_path), cv2.IMREAD_COLOR)
                        except Exception:
                            arr_bgr = None

                if arr_bgr is not None and arr_bgr.size > 0:
                    arr_gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)
                    it.phash = compute_image_phash(arr_gray)
                    it.dhash = compute_image_dhash(arr_gray)
                    if it.sharpness <= 0.0:
                        it.sharpness = compute_sharpness_score(arr_gray)
                    it.lab_haar = compute_lab_haar_signature(arr_bgr)
                    valid_cands.append(it)
                    to_persist.append((it.media_id, int(it.mtime), it.phash, it.dhash, it.sharpness, it.lab_haar))

            if to_persist and self.thumb_repo and hasattr(self.thumb_repo, "upsert_signatures_batch"):
                self.thumb_repo.upsert_signatures_batch(to_persist)

        return valid_cands

    def _compute_item_signature(
        self,
        it: DuplicateMediaItem,
        blob: bytes | None,
    ) -> tuple[int, int, int, int, float, bytes] | None:
        """Compute visual signatures for a single DuplicateMediaItem and return DB tuple if valid."""
        arr_bgr = None
        if blob:
            arr_bgr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        elif it.full_path.exists():
            from src.utils.paths import is_cloud_only_file
            if not is_cloud_only_file(it.full_path):
                try:
                    arr_bgr = cv2.imread(str(it.full_path), cv2.IMREAD_REDUCED_COLOR_8)
                    if arr_bgr is None or arr_bgr.size == 0:
                        arr_bgr = cv2.imread(str(it.full_path), cv2.IMREAD_COLOR)
                except Exception:
                    arr_bgr = None
        if arr_bgr is None or arr_bgr.size == 0:
            return None
        arr_gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)
        it.phash = compute_image_phash(arr_gray)
        it.dhash = compute_image_dhash(arr_gray)
        if it.sharpness <= 0.0:
            it.sharpness = compute_sharpness_score(arr_gray)
        it.lab_haar = compute_lab_haar_signature(arr_bgr)
        return (it.media_id, int(it.mtime), it.phash, it.dhash, it.sharpness, it.lab_haar)

    def find_similar_to_photo(
        self,
        seed_media_id: int,
        similarity_threshold: float = 0.81,
        media_records: Sequence[dict[str, Any]] | None = None,
        max_results: int = 40,
        progress_cb: Any = None,
        partial_cb: Any = None,
    ) -> DuplicateGroup | None:
        """
        Rank photos visually similar to seed_media_id using 24-byte LAB Haar + pHash/dHash
        (and optional MobileCLIP-S1 semantic boost).

        When ``partial_cb(group, done_uncached, total_uncached)`` is provided, immediately scores
        already-cached signatures (<200ms) and streams progressive updates while indexing any
        remaining uncached photos in chunks.
        """
        import time as _time

        self._is_cancelled = False
        if progress_cb:
            progress_cb(0, 100, "Loading catalog photos for similarity search...")
        if media_records is None:
            media_records = self._fetch_all_catalog_records()
        if not media_records:
            return None
        items = list(self._build_items_dict(media_records).values())
        cands = [it for it in items if not it.is_video]
        if not cands:
            return None

        # 1. Fast load of all already-cached visual signatures
        mids = [it.media_id for it in cands]
        cached_sigs = (
            self.thumb_repo.get_signatures_batch(mids)
            if (self.thumb_repo and mids and hasattr(self.thumb_repo, "get_signatures_batch"))
            else {}
        )
        cached_cands: list[DuplicateMediaItem] = []
        uncached_cands: list[DuplicateMediaItem] = []
        seed: DuplicateMediaItem | None = None
        for it in cands:
            sig = cached_sigs.get(it.media_id)
            if sig is not None and sig[0] == int(it.mtime) and (sig[1] != 0 or sig[2] != 0) and sig[4]:
                it.phash = sig[1]
                it.dhash = sig[2]
                if sig[3] > 0.0:
                    it.sharpness = sig[3]
                it.lab_haar = sig[4]
                cached_cands.append(it)
            else:
                uncached_cands.append(it)
            if it.media_id == int(seed_media_id):
                seed = it

        if seed is None:
            return None

        # Ensure the seed photo itself has a signature before scoring anything
        if not seed.lab_haar:
            seed_blobs = (
                self.thumb_repo.get_webp_blobs_batch([seed.media_id])
                if self.thumb_repo
                else {}
            )
            seed_row = self._compute_item_signature(seed, seed_blobs.get(seed.media_id))
            if seed_row is None:
                return None
            if self.thumb_repo and hasattr(self.thumb_repo, "upsert_signatures_batch"):
                self.thumb_repo.upsert_signatures_batch([seed_row])
            uncached_cands = [it for it in uncached_cands if it.media_id != seed.media_id]
            cached_cands.append(seed)

        # Optional MobileCLIP-S1 semantic boost when installed & seed is indexed
        sem_score_map: dict[int, float] = {}
        if (
            self.semantic_repo is not None
            and self.semantic_engine is not None
            and getattr(self.semantic_engine, "is_available", lambda: False)()
        ):
            try:
                sem_mids, sem_mat = self.semantic_repo.load_matrix()
                if sem_mat.shape[0] > 0:
                    seed_matches = np.where(sem_mids == int(seed.media_id))[0]
                    if seed_matches.size > 0:
                        seed_vec = sem_mat[int(seed_matches[0])]
                        cos_sims = sem_mat @ seed_vec
                        for mid_val, cs in zip(sem_mids.tolist(), cos_sims.tolist()):
                            if mid_val != seed.media_id:
                                sem_score_map[int(mid_val)] = float(max(0.0, min(1.0, cs)))
            except Exception:
                sem_score_map = {}

        best_by_id: dict[int, tuple[float, DuplicateMediaItem]] = {}
        seed_lab = np.frombuffer(seed.lab_haar, dtype=np.uint8).astype(np.float32).reshape(8, 3)
        seed_ph = np.uint64(int(seed.phash) & 0xFFFFFFFFFFFFFFFF)
        seed_dh = np.uint64(int(seed.dhash) & 0xFFFFFFFFFFFFFFFF)
        _popcnt_lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
        has_sem = bool(sem_score_map)

        def _score_batch(batch: list[DuplicateMediaItem]) -> bool:
            """Vectorized NumPy similarity scoring (releases GIL; ~200x faster than Python loop)."""
            valid = [
                other for other in batch
                if other.media_id != seed.media_id and other.lab_haar and len(other.lab_haar) == 24
            ]
            if not valid:
                return False
            added_or_improved = False
            sub_chunk = 4096
            for start in range(0, len(valid), sub_chunk):
                if self._is_cancelled:
                    break
                sub = valid[start : start + sub_chunk]
                n = len(sub)
                # 1. Vectorized LAB Haar + symmetric Chamfer palette similarity
                raw_bytes = b"".join(it.lab_haar for it in sub)
                B = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32).reshape(n, 8, 3)

                diff_h = np.abs(B - seed_lab[None, :, :])
                mean_diff = np.mean(
                    diff_h[:, :, 0] * 0.26 + diff_h[:, :, 1] * 0.37 + diff_h[:, :, 2] * 0.37,
                    axis=1,
                )
                haar_sim = np.clip(1.0 - (mean_diff / 75.0), 0.0, 1.0)

                diff_p = np.abs(seed_lab[None, :, None, :] - B[:, None, :, :])
                dist_mat = (
                    diff_p[:, :, :, 0] * 0.15
                    + diff_p[:, :, :, 1] * 0.425
                    + diff_p[:, :, :, 2] * 0.425
                )
                sym_dist = 0.5 * (
                    np.mean(np.min(dist_mat, axis=2), axis=1)
                    + np.mean(np.min(dist_mat, axis=1), axis=1)
                )
                pal_sim = np.clip(1.0 - (sym_dist / 30.0), 0.0, 1.0)

                # 2. Vectorized 64-bit pHash & dHash Hamming distances
                ph_arr = np.fromiter((int(it.phash) & 0xFFFFFFFFFFFFFFFF for it in sub), dtype=np.uint64, count=n)
                dh_arr = np.fromiter((int(it.dhash) & 0xFFFFFFFFFFFFFFFF for it in sub), dtype=np.uint64, count=n)
                ph_xor = np.bitwise_xor(ph_arr, seed_ph).view(np.uint8).reshape(n, 8)
                dh_xor = np.bitwise_xor(dh_arr, seed_dh).view(np.uint8).reshape(n, 8)
                ph_dist = _popcnt_lut[ph_xor].sum(axis=1)
                dh_dist = _popcnt_lut[dh_xor].sum(axis=1)
                min_dist = np.minimum(ph_dist, dh_dist)
                hash_sim = np.clip(1.0 - (min_dist.astype(np.float32) / 64.0), 0.0, 1.0)

                visual_combined = 0.35 * haar_sim + 0.40 * pal_sim + 0.25 * hash_sim

                if has_sem:
                    sem_vals = np.fromiter(
                        (sem_score_map.get(it.media_id, -1.0) for it in sub),
                        dtype=np.float32,
                        count=n,
                    )
                    has_s = sem_vals >= 0.0
                    keep_vis = (hash_sim >= 0.42) & (pal_sim >= 0.62)
                    keep = np.where(has_s, (sem_vals >= 0.72) | keep_vis, keep_vis)
                    combined = np.where(has_s, 0.45 * sem_vals + 0.55 * visual_combined, visual_combined)
                else:
                    keep = (hash_sim >= 0.42) & (pal_sim >= 0.62)
                    combined = visual_combined

                hit_indices = np.flatnonzero(keep & (combined >= similarity_threshold))
                for idx_i in hit_indices.tolist():
                    other = sub[idx_i]
                    score_f = float(combined[idx_i])
                    other.similarity_score = round(score_f, 3)
                    other.is_recommended_keep = False
                    other.is_selected_for_removal = False
                    prev = best_by_id.get(other.media_id)
                    if prev is None or score_f > prev[0]:
                        best_by_id[other.media_id] = (score_f, other)
                        added_or_improved = True
            return added_or_improved

        def _build_group() -> DuplicateGroup | None:
            if not best_by_id:
                return None
            scored = sorted(best_by_id.values(), key=lambda pair: pair[0], reverse=True)
            top_items = [it for _s, it in scored[:max_results]]
            seed.similarity_score = 1.0
            seed.is_recommended_keep = True
            seed.is_selected_for_removal = False
            avg_sim = round(float(np.mean([s for s, _it in scored[:max_results]])), 2)
            return DuplicateGroup(
                group_id=f"similar_{seed.media_id}",
                match_type="similar",
                similarity=avg_sim,
                items=[seed, *top_items],
            )

        # 2. Score all already-cached photos immediately
        _score_batch(cached_cands)
        total_uncached = len(uncached_cands)
        if partial_cb is not None:
            partial_cb(_build_group(), 0, total_uncached)

        # 3. Incrementally index & score any remaining uncached photos
        if total_uncached > 0:
            chunk_size = 200
            last_emit = _time.monotonic()
            dirty_matches = False
            for chunk_start in range(0, total_uncached, chunk_size):
                if self._is_cancelled:
                    break
                chunk_items = uncached_cands[chunk_start : chunk_start + chunk_size]
                chunk_mids = [it.media_id for it in chunk_items]
                blobs = self.thumb_repo.get_webp_blobs_batch(chunk_mids) if (self.thumb_repo and chunk_mids) else {}
                to_persist: list[tuple[int, int, int, int, float, bytes]] = []
                newly_valid: list[DuplicateMediaItem] = []
                for it in chunk_items:
                    if self._is_cancelled:
                        break
                    row = self._compute_item_signature(it, blobs.get(it.media_id))
                    if row is not None:
                        newly_valid.append(it)
                        to_persist.append(row)
                if to_persist and self.thumb_repo and hasattr(self.thumb_repo, "upsert_signatures_batch"):
                    self.thumb_repo.upsert_signatures_batch(to_persist)
                if newly_valid and _score_batch(newly_valid):
                    dirty_matches = True

                done_cnt = min(total_uncached, chunk_start + len(chunk_items))
                now = _time.monotonic()
                if progress_cb:
                    progress_cb(done_cnt, total_uncached, f"Indexing ({done_cnt:,}/{total_uncached:,})...")
                if partial_cb is not None and (
                    (dirty_matches and (now - last_emit) >= 0.8)
                    or (now - last_emit) >= 1.5
                    or done_cnt >= total_uncached
                ):
                    partial_cb(_build_group(), done_cnt, total_uncached)
                    last_emit = now
                    dirty_matches = False

        final_grp = _build_group()
        if progress_cb:
            cnt = max(0, len(final_grp.items) - 1) if final_grp else 0
            progress_cb(100, 100, f"Found {cnt} visually similar photos.")
        return final_grp

    def _find_visual_duplicates(
        self,
        items: list[DuplicateMediaItem],
        exclude_mids: set[int] | None = None,
        similarity_threshold: float = 0.92,
        progress_cb: Any = None,
        group_found_cb: Any = None,
    ) -> list[DuplicateGroup]:
        exclude = exclude_mids or set()
        cands = [it for it in items if not it.is_video and it.media_id not in exclude]
        if len(cands) < 2:
            return []

        max_dist = int(round((1.0 - similarity_threshold) * 64))
        max_dist = max(1, min(10, max_dist))

        valid_cands = self._populate_visual_signatures(cands, progress_cb=progress_cb)

        if self._is_cancelled:
            return []

        if progress_cb:
            progress_cb(88, 100, "Phase 3/3: Comparing visual similarity...")

        # 2. Multi-Index Inverted Table on 16-bit chunks of pHash
        # By Pigeonhole Principle: 4 chunks of 16 bits. If dist <= 3, at least 1 chunk is identical.
        chunk_index: dict[tuple[int, int], list[DuplicateMediaItem]] = {}
        for it in valid_cands:
            h = it.phash
            for c_idx in range(4):
                chunk_val = (h >> (c_idx * 16)) & 0xFFFF
                chunk_index.setdefault((c_idx, chunk_val), []).append(it)

        matched_pairs: set[tuple[int, int]] = set()
        for it in valid_cands:
            if self._is_cancelled:
                break
            h1 = it.phash
            candidate_neighbors: set[int] = set()
            for c_idx in range(4):
                chunk_val = (h1 >> (c_idx * 16)) & 0xFFFF
                for other in chunk_index.get((c_idx, chunk_val), []):
                    if other.media_id != it.media_id:
                        candidate_neighbors.add(other.media_id)

            for other_id in candidate_neighbors:
                pair = (min(it.media_id, other_id), max(it.media_id, other_id))
                if pair not in matched_pairs:
                    matched_pairs.add(pair)

        # Build adjacency graph
        adj: dict[int, set[int]] = {}
        id_to_item = {it.media_id: it for it in valid_cands}
        pair_similarities: dict[tuple[int, int], float] = {}

        for m1, m2 in matched_pairs:
            it1 = id_to_item.get(m1)
            it2 = id_to_item.get(m2)
            if not it1 or not it2:
                continue
            dist_p = hamming_distance(it1.phash, it2.phash)
            dist_d = hamming_distance(it1.dhash, it2.dhash)
            effective_dist = min(dist_p, dist_d)
            if effective_dist <= max_dist:
                if it1.lab_haar and it2.lab_haar and compute_lab_haar_similarity(it1.lab_haar, it2.lab_haar) < 0.55:
                    continue
                sim = similarity_from_distance(effective_dist)
                pair_similarities[(m1, m2)] = sim
                adj.setdefault(m1, set()).add(m2)
                adj.setdefault(m2, set()).add(m1)

        # Connected components -> Duplicate Groups
        visited: set[int] = set()
        groups: list[DuplicateGroup] = []
        group_idx = 1

        for mid in adj:
            if self._is_cancelled:
                break
            if mid in visited:
                continue
            comp: list[int] = []
            queue = [mid]
            visited.add(mid)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj.get(curr, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(comp) >= 2:
                group_items = [id_to_item[m] for m in comp]
                # Average similarity across component edges
                edge_sims = [
                    pair_similarities.get((min(a, b), max(a, b)), 0.95)
                    for a in comp for b in comp if a < b and (min(a, b), max(a, b)) in pair_similarities
                ]
                avg_sim = round(float(np.mean(edge_sims)) if edge_sims else 0.95, 2)
                grp = DuplicateGroup(
                    group_id=f"visual_{group_idx}",
                    match_type="visual",
                    similarity=avg_sim,
                    items=group_items,
                )
                auto_recommend_rules([grp], rule="highest_resolution")
                groups.append(grp)
                group_idx += 1
                if group_found_cb:
                    group_found_cb(grp)

        return groups

    def _load_face_rects_batch(self, media_ids: Sequence[int]) -> dict[int, list[tuple[float, float, float, float]]]:
        if not self.catalog_repo or not media_ids:
            return {}
        try:
            conn = self.catalog_repo.engine.catalog_conn()
            placeholders = ",".join("?" for _ in media_ids)
            rows = conn.execute(
                f"""
                SELECT media_id, rect_x, rect_y, rect_w, rect_h
                FROM media_people
                WHERE media_id IN ({placeholders})
                  AND state >= 0
                  AND rect_w > 0.01 AND rect_h > 0.01
                """,
                [int(m) for m in media_ids],
            ).fetchall()
            out: dict[int, list[tuple[float, float, float, float]]] = {}
            for r in rows:
                out.setdefault(int(r["media_id"]), []).append(
                    (float(r["rect_x"]), float(r["rect_y"]), float(r["rect_w"]), float(r["rect_h"]))
                )
            return out
        except Exception:
            return {}

    def _enrich_burst_face_sharpness(self, items: Sequence[DuplicateMediaItem]) -> None:
        """Compute face/eye sharpness for burst items with detected face bounding boxes.
        Skips decoding items that have no detected faces and already carry cached sharpness from media_signatures.
        """
        mids = [it.media_id for it in items]
        face_map = self._load_face_rects_batch(mids)
        need_decode = [
            it for it in items
            if bool(face_map.get(it.media_id)) or it.sharpness <= 0.0
        ]
        if not need_decode:
            return
        need_mids = [it.media_id for it in need_decode]
        blobs = self.thumb_repo.get_webp_blobs_batch(need_mids) if (self.thumb_repo and need_mids) else {}
        for it in need_decode:
            arr_gray = None
            blob = blobs.get(it.media_id)
            if blob:
                arr_gray = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)
            elif it.full_path.exists():
                from src.utils.paths import is_cloud_only_file
                if not is_cloud_only_file(it.full_path):
                    try:
                        arr_gray = cv2.imread(str(it.full_path), cv2.IMREAD_GRAYSCALE)
                    except Exception:
                        arr_gray = None
            if arr_gray is None or arr_gray.size == 0:
                continue
            global_sharp = it.sharpness if it.sharpness > 0.0 else compute_sharpness_score(arr_gray)
            rects = face_map.get(it.media_id) or []
            face_score, has_face = compute_face_eye_sharpness(arr_gray, rects)
            if has_face:
                it.has_faces = True
                it.face_sharpness = face_score
                it.sharpness = 0.75 * face_score + 0.25 * global_sharp
            else:
                it.sharpness = global_sharp

    def _get_item_sharpness(self, item: DuplicateMediaItem) -> float:
        if item.sharpness > 0.0:
            return item.sharpness
        if self.thumb_repo:
            blob = self.thumb_repo.get_webp_blob(item.media_id)
            if blob:
                arr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)
                if arr is not None and arr.size > 0:
                    return compute_sharpness_score(arr)
        if item.full_path.exists():
            from src.utils.paths import is_cloud_only_file
            if not is_cloud_only_file(item.full_path):
                try:
                    arr = cv2.imread(str(item.full_path), cv2.IMREAD_GRAYSCALE)
                    if arr is not None and arr.size > 0:
                        return compute_sharpness_score(arr)
                except Exception:
                    pass
        return 0.0


def _filename_copy_penalty(filename: str) -> int:
    """Return 0 for clean original stems, 1 for numbered duplicates (-1, (2)), 2 for copy/edited stems."""
    import re
    stem = Path(filename).stem.lower().strip()
    if re.search(r"(?:^copy\s+of\s+|[-_\s]+copy(?:\s*\(\d+\)|\s*\d*)?$|[-_\s]+edited(?:[-_\s]*\d+)?$)", stem):
        return 2
    if re.search(r"(?:\s*\(\d+\)|[-_]\d{1,3})$", stem):
        return 1
    return 0


def compute_reclaimable_summary(groups: list[DuplicateGroup]) -> tuple[int, int]:
    """Returns `(marked_for_removal_count, marked_reclaimable_bytes)` across all groups."""
    count = 0
    total_bytes = 0
    for g in groups:
        for it in g.items:
            if it.is_selected_for_removal:
                count += 1
                total_bytes += int(it.file_size or 0)
    return count, total_bytes


def auto_recommend_rules(
    groups: list[DuplicateGroup],
    rule: str = "highest_resolution",
    preferred_folders: list[str] | tuple[str, ...] = (),
) -> None:
    """
    Applies an auto-selection rule across all duplicate and burst groups:
    - 'highest_resolution': Recommends keeping the photo with the most megapixels (and largest file size).
    - 'largest_file_size': Recommends keeping the photo with the largest byte size (least compressed).
    - 'oldest_date': Recommends keeping the original photo with the earliest date taken.
    - 'most_metadata': Recommends keeping the photo with the most people, tags, caption, rating.
    - 'best_sharpness': Recommends keeping the sharpest, least-blurry photo in the group.
    - 'original_filename': Prefer clean original filenames without copy/numbered suffixes.
    - 'preferred_folder': Prefer keeping files inside the user's prioritized folder prefixes.
    """
    norm_prefs = [
        os.path.normcase(str(p).strip()).rstrip("\\/")
        for p in (preferred_folders or ())
        if str(p).strip()
    ]

    def _folder_priority_score(folder_path: str) -> int:
        fp = os.path.normcase(str(folder_path or "")).rstrip("\\/")
        for idx, pref in enumerate(norm_prefs):
            if fp == pref or fp.startswith(pref + os.sep) or fp.startswith(pref + "/"):
                return len(norm_prefs) - idx
        return 0

    for g in groups:
        if not g.items:
            continue

        if rule == "preferred_folder" and norm_prefs:
            sorted_items = sorted(
                g.items,
                key=lambda it: (
                    _folder_priority_score(getattr(it, "folder_path", "")),
                    it.width * it.height,
                    it.file_size,
                    it.metadata_score,
                ),
                reverse=True,
            )
        elif rule == "original_filename":
            sorted_items = sorted(
                g.items,
                key=lambda it: (
                    -_filename_copy_penalty(it.filename),
                    -len(it.filename),
                    it.width * it.height,
                    it.file_size,
                    it.metadata_score,
                ),
                reverse=True,
            )
        elif g.match_type == "burst" or rule == "best_sharpness":
            # For bursts & sharpness culling, face/eye sharpness wins when faces are detected
            sorted_items = sorted(
                g.items,
                key=lambda it: (
                    it.face_sharpness if it.has_faces else it.sharpness,
                    it.sharpness,
                    it.width * it.height,
                    it.file_size,
                    -_filename_copy_penalty(it.filename),
                    -len(it.filename),
                ),
                reverse=True,
            )
        elif rule == "largest_file_size":
            sorted_items = sorted(
                g.items,
                key=lambda it: (
                    it.file_size,
                    it.width * it.height,
                    it.metadata_score,
                    -_filename_copy_penalty(it.filename),
                    -len(it.filename),
                ),
                reverse=True,
            )
        elif rule == "oldest_date":
            sorted_items = sorted(
                g.items,
                key=lambda it: (
                    it.date_taken if it.date_taken > 0 else it.mtime,
                    -(it.width * it.height),
                    _filename_copy_penalty(it.filename),
                    len(it.filename),
                ),
            )
        elif rule == "most_metadata":
            sorted_items = sorted(
                g.items,
                key=lambda it: (
                    it.metadata_score,
                    it.width * it.height,
                    it.file_size,
                    -_filename_copy_penalty(it.filename),
                    -len(it.filename),
                ),
                reverse=True,
            )
        else:  # "highest_resolution" (default)
            sorted_items = sorted(
                g.items,
                key=lambda it: (
                    it.width * it.height,
                    it.file_size,
                    round(float(it.sharpness or 0.0), 1),
                    it.metadata_score,
                    -_filename_copy_penalty(it.filename),
                    -len(it.filename),
                ),
                reverse=True,
            )

        best_item = sorted_items[0]
        for it in g.items:
            if it.media_id == best_item.media_id:
                it.is_recommended_keep = True
                it.is_selected_for_removal = False
            else:
                it.is_recommended_keep = False
                it.is_selected_for_removal = True


def merge_duplicate_metadata(
    catalog_repo: Any,
    keep_media_id: int,
    remove_media_ids: Sequence[int],
) -> int:
    """
    Safely merges descriptive tags, person assignments, non-empty caption, and max rating
    from duplicates about to be recycled into the kept master photo.
    Returns count of merged items.
    """
    rem_ids = [int(m) for m in remove_media_ids if int(m) != int(keep_media_id)]
    if not rem_ids or catalog_repo is None:
        return 0
    try:
        keep_detail = catalog_repo.get_media_detail(int(keep_media_id))
        if not keep_detail:
            return 0
        keep_tags = {
            str(t.get("full_path") or t.get("name") or t if isinstance(t, dict) else t).strip()
            for t in (keep_detail.get("tags") or [])
            if str(t.get("full_path") or t.get("name") or t if isinstance(t, dict) else t).strip()
        }
        keep_people = {
            str(p.get("name") or p.get("person_name") or "").strip().lower()
            for p in (keep_detail.get("people") or [])
            if str(p.get("name") or p.get("person_name") or "").strip()
        }
        keep_caption = str(keep_detail.get("caption") or "").strip()
        keep_rating = int(keep_detail.get("rating") or 0)

        new_tags: list[str] = []
        new_people: list[str] = []
        best_caption = keep_caption
        best_rating = keep_rating

        for rid in rem_ids:
            det = catalog_repo.get_media_detail(rid)
            if not det:
                continue
            for t in det.get("tags") or []:
                ts = str(t.get("full_path") or t.get("name") or t if isinstance(t, dict) else t).strip()
                if ts and ts not in keep_tags:
                    keep_tags.add(ts)
                    new_tags.append(ts)
            for p in det.get("people") or []:
                pn = str(p.get("name") or p.get("person_name") or "").strip()
                if pn and pn.lower() not in keep_people:
                    keep_people.add(pn.lower())
                    new_people.append(pn)
            if not best_caption:
                c = str(det.get("caption") or "").strip()
                if c:
                    best_caption = c
            best_rating = max(best_rating, int(det.get("rating") or 0))

        for t in new_tags:
            if hasattr(catalog_repo, "create_tag") and hasattr(catalog_repo, "assign_tag"):
                tid = catalog_repo.create_tag(t)
                catalog_repo.assign_tag([int(keep_media_id)], tid)
        for pn in new_people:
            if hasattr(catalog_repo, "assign_person"):
                catalog_repo.assign_person(int(keep_media_id), pn)
        if best_caption and not keep_caption and hasattr(catalog_repo, "set_caption"):
            catalog_repo.set_caption([int(keep_media_id)], best_caption)
        if best_rating > keep_rating and hasattr(catalog_repo, "set_rating"):
            catalog_repo.set_rating([int(keep_media_id)], best_rating)
        return len(rem_ids)
    except Exception:
        return 0
