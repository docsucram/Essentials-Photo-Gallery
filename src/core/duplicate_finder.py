"""
PhotoGallery 64 — Core Duplicate & Burst Detection Engine
Provides high-speed exact byte hashing, multi-index perceptual hashing (dHash/pHash),
and burst sequence analysis with sharpness scoring and smart auto-recommendation rules.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np


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


def hamming_distance(h1: int, h2: int) -> int:
    """Returns the bitwise Hamming distance between two 64-bit integer hashes."""
    return (h1 ^ h2).bit_count()


def similarity_from_distance(dist: int) -> float:
    """Converts 64-bit Hamming distance to similarity ratio (0.0 to 1.0)."""
    return max(0.0, 1.0 - (dist / 64.0))


def fast_file_hash(path: Path | str, chunk_size: int = 8192) -> str | None:
    """Computes a fast partial hash of file header + tail + size for O(1) exact candidate grouping."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        size = p.stat().st_size
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
    p = Path(path)
    if not p.is_file():
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

    def __init__(self, catalog_repo: Any, thumb_repo: Any) -> None:
        self.catalog_repo = catalog_repo
        self.thumb_repo = thumb_repo
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
        burst_time_window_sec: float = 3.0,
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

    def _build_items_dict(self, records: Sequence[dict[str, Any]]) -> dict[int, DuplicateMediaItem]:
        items: dict[int, DuplicateMediaItem] = {}
        for r in records:
            mid = int(r["id"])
            items[mid] = DuplicateMediaItem(
                media_id=mid,
                folder_path=str(r.get("folder_path") or ""),
                filename=str(r.get("filename") or ""),
                file_size=int(r.get("size") or 0),
                width=int(r.get("width") or 0),
                height=int(r.get("height") or 0),
                mtime=float(r.get("mtime") or 0.0),
                date_taken=float(r.get("date_taken") or 0.0),
                rating=int(r.get("rating") or 0),
                flag=int(r.get("flag") or 0),
                caption=str(r.get("caption") or ""),
                is_video=bool(r.get("is_video")),
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
        max_time_diff: float = 3.0,
        exclude_mids: set[int] | None = None,
        progress_cb: Any = None,
        group_found_cb: Any = None,
    ) -> list[DuplicateGroup]:
        exclude = exclude_mids or set()
        candidates = [
            it for it in items
            if not it.is_video and it.media_id not in exclude and (it.date_taken > 0 or it.mtime > 0)
        ]
        candidates.sort(key=lambda it: it.date_taken if it.date_taken > 0 else it.mtime)

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
            t_prev = candidates[i].date_taken if candidates[i].date_taken > 0 else candidates[i].mtime
            j = i + 1
            while j < n:
                if self._is_cancelled:
                    break
                it_next = candidates[j]
                t_next = it_next.date_taken if it_next.date_taken > 0 else it_next.mtime
                delta = t_next - t_prev
                if 0 <= delta <= max_time_diff:
                    if it_next.folder_path == seq[0].folder_path:
                        seq.append(it_next)
                        t_prev = t_next
                        j += 1
                    else:
                        break
                else:
                    break

            if len(seq) >= 2:
                for it in seq:
                    it.sharpness = self._get_item_sharpness(it)

                t_start = seq[0].date_taken if seq[0].date_taken > 0 else seq[0].mtime
                t_end = seq[-1].date_taken if seq[-1].date_taken > 0 else seq[-1].mtime
                span = max(0.1, t_end - t_start)

                grp = DuplicateGroup(
                    group_id=f"burst_{group_idx}",
                    match_type="burst",
                    similarity=0.98,
                    items=seq,
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

        # 1. Compute perceptual hashes
        max_dist = int(round((1.0 - similarity_threshold) * 64))
        max_dist = max(1, min(6, max_dist))

        # Batch fetch thumbnails from thumbnails.db
        mids = [it.media_id for it in cands]
        blobs = self.thumb_repo.get_webp_blobs_batch(mids) if self.thumb_repo else {}

        valid_cands: list[DuplicateMediaItem] = []
        total_cands = len(cands)
        for idx, it in enumerate(cands, start=1):
            if self._is_cancelled:
                break
            if progress_cb and idx % 20 == 0:
                pct = int(50 + (idx / max(1, total_cands)) * 38)
                progress_cb(pct, 100, f"Phase 3/3: Computing visual hashes ({idx}/{total_cands} photos)...")

            blob = blobs.get(it.media_id)
            if blob:
                # Fast decode from WebP blob
                arr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)
                if arr is not None and arr.size > 0:
                    it.phash = compute_image_phash(arr)
                    it.dhash = compute_image_dhash(arr)
                    valid_cands.append(it)
            elif it.full_path.exists():
                # Direct file fallback
                try:
                    arr = cv2.imread(str(it.full_path), cv2.IMREAD_GRAYSCALE)
                    if arr is not None and arr.size > 0:
                        it.phash = compute_image_phash(arr)
                        it.dhash = compute_image_dhash(arr)
                        valid_cands.append(it)
                except Exception:
                    pass

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
            try:
                arr = cv2.imread(str(item.full_path), cv2.IMREAD_GRAYSCALE)
                if arr is not None and arr.size > 0:
                    return compute_sharpness_score(arr)
            except Exception:
                pass
        return 0.0


def auto_recommend_rules(groups: list[DuplicateGroup], rule: str = "highest_resolution") -> None:
    """
    Applies an auto-selection rule across all duplicate and burst groups:
    - 'highest_resolution': Recommends keeping the photo with the most megapixels (and largest file size).
    - 'largest_file_size': Recommends keeping the photo with the largest byte size (least compressed).
    - 'oldest_date': Recommends keeping the original photo with the earliest date taken.
    - 'most_metadata': Recommends keeping the photo with the most people, tags, caption, rating.
    - 'best_sharpness': Recommends keeping the sharpest, least-blurry photo in the group.
    """
    for g in groups:
        if not g.items:
            continue

        if g.match_type == "burst" or rule == "best_sharpness":
            # For bursts, sharpness always wins
            sorted_items = sorted(
                g.items,
                key=lambda it: (it.sharpness, it.width * it.height, it.file_size),
                reverse=True,
            )
        elif rule == "largest_file_size":
            sorted_items = sorted(
                g.items,
                key=lambda it: (it.file_size, it.width * it.height, it.metadata_score),
                reverse=True,
            )
        elif rule == "oldest_date":
            sorted_items = sorted(
                g.items,
                key=lambda it: (
                    it.date_taken if it.date_taken > 0 else it.mtime,
                    -(it.width * it.height),
                ),
            )
        elif rule == "most_metadata":
            sorted_items = sorted(
                g.items,
                key=lambda it: (it.metadata_score, it.width * it.height, it.file_size),
                reverse=True,
            )
        else:  # "highest_resolution" (default)
            sorted_items = sorted(
                g.items,
                key=lambda it: (it.width * it.height, it.file_size, it.metadata_score),
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
