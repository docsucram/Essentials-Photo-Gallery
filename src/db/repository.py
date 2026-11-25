"""
Repository layer for catalog.db and thumbnails.db.
Provides <10ms indexed queries, Boolean tag_closure filtering, R*Tree spatial lookups,
lossless orientation updates, relative Δt time shifts, and 30-day Original Files backup tracking.
"""

from __future__ import annotations

import calendar
import datetime
import io
import json
import os
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageOps

from src.db.connection import DualDatabaseEngine


def _iter_id_chunks(media_ids: Sequence[int], chunk_size: int = 900) -> Iterator[Sequence[int]]:
    """Yields consecutive slices of media_ids to avoid SQLite parameter limit (32,766)."""
    for i in range(0, len(media_ids), chunk_size):
        yield media_ids[i : i + chunk_size]

ArrangeMode = Literal[
    "auto",
    "date",
    "month",
    "year",
    "time",
    "size",
    "image_size",
    "orientation",
    "file_type",
    "media_type",
    "rating",
    "flag",
    "favourites",
    "folder",
    "tag",
    "people",
    "person",
    "published",
    "geotag",
    "camera",
    "name",
    "none",
]

# EXIF Orientation rotation lookup tables
_ROTATE_CW_MAP: dict[int, int] = {
    1: 6,
    6: 3,
    3: 8,
    8: 1,
    2: 7,
    7: 4,
    4: 5,
    5: 2,
}
_ROTATE_CCW_MAP: dict[int, int] = {v: k for k, v in _ROTATE_CW_MAP.items()}
_ROTATE_180_MAP: dict[int, int] = {1: 3, 3: 1, 6: 8, 8: 6, 2: 4, 4: 2, 5: 7, 7: 5}

_RATING_LABELS: dict[int, str] = {
    5: "5 stars",
    4: "4 stars",
    3: "3 stars",
    2: "2 stars",
    1: "1 star",
    0: "Unrated",
}


def _build_fts5_prefix_query(term: str) -> str | None:
    """Convert an arbitrary search string into a safe FTS5 prefix MATCH expression."""
    tokens = re.findall(r"[^\W_]+", term or "", re.UNICODE)
    if not tokens:
        return None
    return " ".join(f'"{tok}"*' for tok in tokens)


def _build_boolean_search_sql(search_text: str, use_ocr_fts: bool = False) -> tuple[str, list[Any]]:
    """
    Parses a Boolean search string supporting:
    - AND / OR / NOT operators (e.g. 'cat AND dog', 'cat OR dog', 'sunset -winter')
    - Quoted exact phrases (e.g. '"Mu Rai" AND beach')
    - Person/People tag syntax (e.g. 'person:"Mu Rai"', 'people:Craig')
    - Descriptive tag syntax (e.g. 'tag:Vacation', 'tag:"Family Reunion"')
    - General terms matching filename, caption, location, linked people, or descriptive tags.
    """
    text = search_text.strip()
    if not text:
        return "", []

    token_pattern = re.compile(
        r'(-?(?:person|people|tag|ocr):"[^"]+"|-?(?:person|people|tag|ocr):\S+|"[^"]+"|\bAND\b|\bOR\b|\bNOT\b|-\S+|\S+)',
        re.IGNORECASE,
    )
    raw_tokens = token_pattern.findall(text)
    if not raw_tokens:
        return "", []

    clauses: list[str] = []
    params: list[Any] = []
    current_connector = "AND"
    next_not = False

    for token in raw_tokens:
        upper = token.upper()
        if upper == "AND":
            current_connector = "AND"
            continue
        elif upper == "OR":
            current_connector = "OR"
            continue
        elif upper == "NOT":
            next_not = True
            continue

        is_negated = next_not or token.startswith("-")
        next_not = False
        val = token[1:] if token.startswith("-") else token

        field_match = re.match(r"^(person|people|tag|ocr):(.*)$", val, re.IGNORECASE)
        if field_match:
            field_type = field_match.group(1).lower()
            field_val = field_match.group(2).strip('"')
            like_val = f"%{field_val}%"
            if field_type in ("person", "people"):
                atom = """EXISTS (
                    SELECT 1 FROM media_people mp
                    JOIN people p ON mp.person_id = p.id
                    WHERE mp.media_id = m.id AND mp.state = 1 AND p.name LIKE ? COLLATE NOCASE
                )"""
                p_list = [like_val]
            elif field_type == "ocr":
                fts_q = _build_fts5_prefix_query(field_val) if use_ocr_fts else None
                if fts_q:
                    atom = "m.id IN (SELECT media_id FROM media_ocr_fts WHERE media_ocr_fts MATCH ?)"
                    p_list = [fts_q]
                else:
                    atom = "COALESCE(m.ocr_text, '') LIKE ? COLLATE NOCASE"
                    p_list = [like_val]
            else:
                atom = """EXISTS (
                    SELECT 1 FROM media_tags mt
                    JOIN tags t ON mt.tag_id = t.id
                    WHERE mt.media_id = m.id AND (t.name LIKE ? COLLATE NOCASE OR t.full_path LIKE ? COLLATE NOCASE)
                )"""
                p_list = [like_val, like_val]
        else:
            term = val.strip('"')
            like_val = f"%{term}%"
            fts_q = _build_fts5_prefix_query(term) if use_ocr_fts else None
            if fts_q:
                ocr_pred = "m.id IN (SELECT media_id FROM media_ocr_fts WHERE media_ocr_fts MATCH ?)"
                ocr_param = fts_q
            else:
                ocr_pred = "COALESCE(m.ocr_text, '') LIKE ? COLLATE NOCASE"
                ocr_param = like_val
            atom = f"""(
                m.filename LIKE ? COLLATE NOCASE
                OR m.caption LIKE ? COLLATE NOCASE
                OR m.location_name LIKE ? COLLATE NOCASE
                OR {ocr_pred}
                OR EXISTS (
                    SELECT 1 FROM media_tags mt
                    JOIN tags t ON mt.tag_id = t.id
                    WHERE mt.media_id = m.id AND t.name LIKE ? COLLATE NOCASE
                )
                OR EXISTS (
                    SELECT 1 FROM media_people mp
                    JOIN people p ON mp.person_id = p.id
                    WHERE mp.media_id = m.id AND mp.state = 1 AND p.name LIKE ? COLLATE NOCASE
                )
            )"""
            p_list = [like_val, like_val, like_val, ocr_param, like_val, like_val]

        if is_negated:
            atom = f"NOT ({atom})"

        if not clauses:
            clauses.append(atom)
        else:
            clauses.append(f"{current_connector} {atom}")
            current_connector = "AND"

        params.extend(p_list)

    final_sql = f"({' '.join(clauses)})"
    return final_sql, params


RAW_EXTENSIONS: tuple[str, ...] = (
    "raf", "cr2", "cr3", "nef", "nrw", "arw", "dng", "orf", "ori",
    "rw2", "rwl", "raw", "pef", "srw", "3fr", "mrw", "sr2", "srf",
    "x3f", "erf", "kdc", "dcr", "mos", "mef", "iiq"
)


@dataclass(frozen=True, slots=True)
class FilterQuerySpec:
    """Immutable specification of active navigation tree, ribbon, and search filters."""

    folder_id: int | None = None
    folder_ids: tuple[int, ...] = ()
    include_subfolders: bool = True
    date_year: int | None = None
    date_month: int | None = None
    date_day: int | None = None
    dates: tuple[tuple[int, int | None, int | None], ...] = ()
    date_from: int | None = None
    date_to: int | None = None
    include_tag_ids: tuple[int, ...] = ()
    tag_boolean_mode: Literal["AND", "OR"] = "AND"
    exclude_tag_ids: tuple[int, ...] = ()
    untagged_only: bool = False
    person_ids: tuple[int, ...] = ()
    person_boolean_mode: Literal["AND", "OR"] = "OR"
    ratings: tuple[int, ...] = ()
    flagged_only: bool = False
    flag_filter: int | None = None  # 1 = flagged only, 0 = unflagged only, None = all
    edited_only: bool = False
    edited_filter: int | None = None  # 1 = edited only, 0 = unedited only, None = all
    media_type: Literal["all", "image", "video", "raw", "motion"] = "all"
    orientation_filter: Literal["all", "landscape", "portrait"] = "all"
    search_text: str = ""
    bbox: tuple[float, float, float, float] | None = None  # (min_lat, max_lat, min_lon, max_lon)
    media_ids: tuple[int, ...] | None = None  # Explicit media ID filtering (e.g. from map cluster clicks)
    location_country: str | None = None
    location_state: str | None = None
    location_city: str | None = None
    locations: tuple[tuple[str, str, str], ...] = ()
    camera_model: str | None = None
    camera_models: tuple[str, ...] = ()
    group_by: str = "month"  # "month", "year", "date", "rating", "folder", "media_type", "flag", "size", "none"
    sort_by: str = "date"  # "date", "name", "rating", "size", "media_type"
    sort_reverse: bool = False
    group_reverse: bool = False
    arrange_by: ArrangeMode = "date"
    reverse_sort: bool = False
    album_id: int | None = None
    smart_album_id: int | None = None
    smart_album_rules: str | None = None
    collapse_stacks: bool = True
    expanded_stack_ids: tuple[int, ...] = ()
    favourites_only: bool = False


@dataclass(slots=True)
class MediaItemRecord:
    """Compact in-memory representation of a catalog row (~120 bytes) for QAbstractListModel."""

    id: int
    folder_id: int
    filename: str
    size: int
    mtime: int
    date_taken: int
    rating: int
    flag: int
    width: int
    height: int
    orientation: int
    is_video: int
    caption: str
    group_key: str
    is_live_photo: int = 0
    is_offline: int = 0
    stack_count: int = 1
    stack_parent_id: int | None = None
    stack_pos: int = 0
    is_raw_jpg_stack: bool = False
    is_favourite: int = 0


_EPOCH_UTC = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def _epoch_to_datetime_utc(epoch: int) -> datetime.datetime:
    """Converts Unix epoch seconds (including negative pre-1970 timestamps on Windows) to UTC datetime."""
    return _EPOCH_UTC + datetime.timedelta(seconds=int(epoch))


def _compute_date_bounds_utc(
    year: int, month: int | None = None, day: int | None = None
) -> tuple[int, int]:
    """Computes inclusive [start_epoch, end_epoch] UTC bounds for indexed date_taken filtering."""
    if month is None:
        start_dt = datetime.datetime(year, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        end_dt = datetime.datetime(year, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc)
    elif day is None:
        last_day = calendar.monthrange(year, month)[1]
        start_dt = datetime.datetime(year, month, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        end_dt = datetime.datetime(year, month, last_day, 23, 59, 59, tzinfo=datetime.timezone.utc)
    else:
        start_dt = datetime.datetime(year, month, day, 0, 0, 0, tzinfo=datetime.timezone.utc)
        end_dt = datetime.datetime(year, month, day, 23, 59, 59, tzinfo=datetime.timezone.utc)
    return int((start_dt - _EPOCH_UTC).total_seconds()), int((end_dt - _EPOCH_UTC).total_seconds())


def _format_month_year_group(epoch: int | None) -> str:
    if epoch is None:
        return "Unknown Date"
    dt = _epoch_to_datetime_utc(epoch)
    return dt.strftime("%B %Y")


class CatalogRepository:
    """High-speed repository for catalog.db operations."""

    def __init__(self, engine: DualDatabaseEngine) -> None:
        self.engine = engine
        self.include_private: bool = False
        self._has_ocr_fts: bool | None = None

    def _supports_ocr_fts(self) -> bool:
        if self._has_ocr_fts is None:
            try:
                row = self.engine.catalog_conn().execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='media_ocr_fts';"
                ).fetchone()
                self._has_ocr_fts = row is not None
            except Exception:
                self._has_ocr_fts = False
        return bool(self._has_ocr_fts)

    def set_include_private(self, include_private: bool) -> None:
        """Toggle session visibility of Private Folders across all catalog queries."""
        self.include_private = bool(include_private)

    def has_private_folders(self) -> bool:
        """Returns True if at least one folder in catalog.db is marked is_private = 1."""
        conn = self.engine.catalog_conn()
        row = conn.execute(
            "SELECT 1 FROM folders WHERE COALESCE(is_private, 0) = 1 LIMIT 1;"
        ).fetchone()
        return row is not None

    def _private_folder_clause(self, alias: str = "f") -> str:
        """SQL predicate enforcing excluded-folder and Private Folder session isolation."""
        base = f"COALESCE({alias}.monitor_mode, 'always') != 'excluded'"
        if self.include_private:
            return base
        return f"({base} AND COALESCE({alias}.is_private, 0) = 0)"

    # -------------------------------------------------------------------------
    # Folder Operations
    # -------------------------------------------------------------------------
    def upsert_folder(
        self,
        path: str,
        name: str,
        parent_id: int | None,
        mtime: int,
        is_favourite: int = 0,
    ) -> int:
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            cur = conn.execute(
                """
                INSERT INTO folders (path, name, parent_id, mtime, is_favourite)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    name = excluded.name,
                    parent_id = COALESCE(excluded.parent_id, folders.parent_id),
                    mtime = excluded.mtime
                RETURNING id;
                """,
                (path, name, parent_id, mtime, is_favourite),
            )
            folder_id = int(cur.fetchone()[0])
            conn.commit()
            return folder_id

    def set_folder_favourite(self, folder_id: int, is_favourite: bool) -> None:
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute(
                "UPDATE folders SET is_favourite = ? WHERE id = ?;",
                (1 if is_favourite else 0, folder_id),
            )
            conn.commit()

    def get_folders(self) -> list[dict[str, Any]]:
        """Returns active catalog folders respecting Private Mode session isolation."""
        conn = self.engine.catalog_conn()
        clause = self._private_folder_clause("f")
        rows = conn.execute(
            f"""
            SELECT f.id, f.path, f.name, f.parent_id, f.mtime, f.is_favourite,
                   COALESCE(f.monitor_mode, 'always') AS monitor_mode,
                   COALESCE(f.scan_faces, 1) AS scan_faces,
                   COALESCE(f.scan_ocr, 0) AS scan_ocr,
                   COALESCE(f.is_private, 0) AS is_private,
                   COALESCE(f.ocr_scanned_count, 0) AS ocr_scanned_count
            FROM folders f
            WHERE {clause}
            ORDER BY f.name COLLATE NOCASE ASC, f.path COLLATE NOCASE ASC;
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_folders_with_policies(self) -> list[dict[str, Any]]:
        """Returns all folders (including excluded and private) for the Folder Manager Dialog."""
        conn = self.engine.catalog_conn()
        rows = conn.execute(
            """
            SELECT f.id, f.path, f.name, f.parent_id, f.mtime, f.is_favourite,
                   COALESCE(f.monitor_mode, 'always') AS monitor_mode,
                   COALESCE(f.scan_faces, 1) AS scan_faces,
                   COALESCE(f.scan_ocr, 0) AS scan_ocr,
                   COALESCE(f.is_private, 0) AS is_private,
                   COALESCE(f.ocr_scanned_count, 0) AS ocr_scanned_count,
                   COALESCE(mc.media_count, 0) AS media_count
            FROM folders f
            LEFT JOIN (
                SELECT folder_id, COUNT(*) AS media_count
                FROM media
                WHERE COALESCE(is_missing, 0) = 0
                GROUP BY folder_id
            ) mc ON mc.folder_id = f.id
            ORDER BY f.name COLLATE NOCASE ASC, f.path COLLATE NOCASE ASC;
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def get_folder_by_id(self, folder_id: int) -> dict[str, Any] | None:
        conn = self.engine.catalog_conn()
        row = conn.execute(
            """
            SELECT id, path, name, parent_id, mtime, is_favourite,
                   COALESCE(monitor_mode, 'always') AS monitor_mode,
                   COALESCE(scan_faces, 1) AS scan_faces,
                   COALESCE(scan_ocr, 0) AS scan_ocr,
                   COALESCE(is_private, 0) AS is_private,
                   COALESCE(ocr_scanned_count, 0) AS ocr_scanned_count
            FROM folders WHERE id = ?;
            """,
            (folder_id,),
        ).fetchone()
        return dict(row) if row else None

    def update_folder_policy(
        self,
        folder_id: int,
        *,
        monitor_mode: str | None = None,
        scan_faces: bool | None = None,
        scan_ocr: bool | None = None,
        is_private: bool | None = None,
    ) -> None:
        """Persist Folder Manager policy flags for a single folder."""
        self.update_folders_policy_batch(
            [int(folder_id)],
            monitor_mode=monitor_mode,
            scan_faces=scan_faces,
            scan_ocr=scan_ocr,
            is_private=is_private,
        )

    def update_folders_policy_batch(
        self,
        folder_ids: Sequence[int],
        *,
        monitor_mode: str | None = None,
        scan_faces: bool | None = None,
        scan_ocr: bool | None = None,
        is_private: bool | None = None,
    ) -> None:
        """Persist Folder Manager policy flags for one or more folders in a single transaction."""
        fids = [int(fid) for fid in folder_ids]
        if not fids:
            return
        updates: list[str] = []
        params: list[Any] = []
        if monitor_mode is not None:
            mode_val = str(monitor_mode).strip().lower()
            if mode_val not in ("always", "once", "excluded"):
                mode_val = "always"
            updates.append("monitor_mode = ?")
            params.append(mode_val)
        if scan_faces is not None:
            updates.append("scan_faces = ?")
            params.append(1 if scan_faces else 0)
        if scan_ocr is not None:
            updates.append("scan_ocr = ?")
            params.append(1 if scan_ocr else 0)
        if is_private is not None:
            updates.append("is_private = ?")
            params.append(1 if is_private else 0)
        if not updates:
            return
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            set_clause = ", ".join(updates)
            for chunk in _iter_id_chunks(fids, 900):
                ph = ",".join("?" for _ in chunk)
                conn.execute(
                    f"UPDATE folders SET {set_clause} WHERE id IN ({ph});",
                    [*params, *chunk],
                )
            conn.commit()

    def apply_folder_policy_to_subfolders(self, folder_id: int) -> int:
        """Cascade monitor_mode, scan_faces, scan_ocr, and is_private to all descendant folders."""
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            row = conn.execute(
                """
                SELECT COALESCE(monitor_mode, 'always') AS monitor_mode,
                       COALESCE(scan_faces, 1) AS scan_faces,
                       COALESCE(scan_ocr, 1) AS scan_ocr,
                       COALESCE(is_private, 0) AS is_private
                FROM folders WHERE id = ?;
                """,
                (int(folder_id),),
            ).fetchone()
            if not row:
                return 0
            desc_rows = conn.execute(
                """
                WITH RECURSIVE descendants(id) AS (
                    SELECT id FROM folders WHERE parent_id = ?
                    UNION ALL
                    SELECT f.id FROM folders f
                    JOIN descendants d ON f.parent_id = d.id
                )
                SELECT id FROM descendants;
                """,
                (int(folder_id),),
            ).fetchall()
            desc_ids = [int(r[0]) for r in desc_rows]
            if not desc_ids:
                return 0
            placeholders = ",".join("?" for _ in desc_ids)
            conn.execute(
                f"""
                UPDATE folders
                SET monitor_mode = ?, scan_faces = ?, scan_ocr = ?, is_private = ?
                WHERE id IN ({placeholders});
                """,
                [
                    row["monitor_mode"],
                    int(row["scan_faces"]),
                    int(row["scan_ocr"]),
                    int(row["is_private"]),
                    *desc_ids,
                ],
            )
            conn.commit()
            return len(desc_ids)

    def exclude_folder_from_catalog(self, folder_id: int) -> int:
        """
        Mark folder_id and all descendants as monitor_mode='excluded' and remove their
        cataloged media records without deleting files on disk.
        """
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            desc_rows = conn.execute(
                """
                WITH RECURSIVE sub(id) AS (
                    SELECT ?
                    UNION ALL
                    SELECT f.id FROM folders f JOIN sub s ON f.parent_id = s.id
                )
                SELECT id FROM sub;
                """,
                (int(folder_id),),
            ).fetchall()
            folder_ids = [int(r[0]) for r in desc_rows]
            if not folder_ids:
                return 0

            placeholders = ",".join("?" for _ in folder_ids)
            media_rows = conn.execute(
                f"SELECT id FROM media WHERE folder_id IN ({placeholders});",
                folder_ids,
            ).fetchall()
            media_ids = [int(r[0]) for r in media_rows]

            conn.execute(
                f"UPDATE folders SET monitor_mode = 'excluded' WHERE id IN ({placeholders});",
                folder_ids,
            )
            if media_ids:
                for chunk in _iter_id_chunks(media_ids, 900):
                    m_ph = ",".join("?" for _ in chunk)
                    try:
                        conn.execute(f"DELETE FROM media_ocr_fts WHERE media_id IN ({m_ph});", chunk)
                    except sqlite3.OperationalError:
                        pass
                    conn.execute(f"DELETE FROM media WHERE id IN ({m_ph});", chunk)
            conn.commit()

        if media_ids and self.engine.thumbnails_db_path is not None:
            thumb_conn = self.engine.thumbnails_conn()
            for chunk in _iter_id_chunks(media_ids, 900):
                m_ph = ",".join("?" for _ in chunk)
                thumb_conn.execute(f"DELETE FROM thumbnails WHERE media_id IN ({m_ph});", chunk)
            thumb_conn.commit()
        return len(media_ids)

    def clear_unconfirmed_faces_in_folder(self, folder_id: int) -> int:
        """Delete unconfirmed (state = 0) face detections/suggestions for media in folder_id tree."""
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            cur = conn.execute(
                """
                DELETE FROM media_people
                WHERE state = 0
                  AND media_id IN (
                      SELECT m.id FROM media m
                      WHERE m.folder_id IN (
                          WITH RECURSIVE sub(id) AS (
                              SELECT ?
                              UNION ALL
                              SELECT f.id FROM folders f JOIN sub s ON f.parent_id = s.id
                          )
                          SELECT id FROM sub
                      )
                  );
                """,
                (int(folder_id),),
            )
            conn.commit()
            return int(cur.rowcount or 0)

    def get_monitored_folder_paths(self) -> list[str]:
        """Return folder paths configured for live monitoring (monitor_mode = 'always')."""
        conn = self.engine.catalog_conn()
        rows = conn.execute(
            "SELECT path FROM folders WHERE COALESCE(monitor_mode, 'always') = 'always' ORDER BY path ASC;"
        ).fetchall()
        return [str(r[0]) for r in rows]

    def get_excluded_folder_paths(self) -> set[str]:
        """Return normalized folder paths marked as excluded so parent scans skip them."""
        conn = self.engine.catalog_conn()
        rows = conn.execute(
            "SELECT path FROM folders WHERE COALESCE(monitor_mode, 'always') = 'excluded';"
        ).fetchall()
        return {str(Path(r[0])).replace("\\", "/").rstrip("/").lower() for r in rows}

    def update_media_ocr(self, media_id: int, ocr_text: str) -> None:
        """Store extracted OCR text in media + media_ocr_fts and mark ocr_scanned = 1."""
        clean_text = (ocr_text or "").strip()
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute(
                "UPDATE media SET ocr_scanned = 1, ocr_text = ? WHERE id = ?;",
                (clean_text, int(media_id)),
            )
            try:
                conn.execute("DELETE FROM media_ocr_fts WHERE media_id = ?;", (int(media_id),))
                if clean_text:
                    conn.execute(
                        "INSERT INTO media_ocr_fts (media_id, ocr_text) VALUES (?, ?);",
                        (int(media_id), clean_text),
                    )
            except sqlite3.OperationalError:
                pass
            conn.execute(
                """
                UPDATE folders
                SET ocr_scanned_count = (
                    SELECT COUNT(*) FROM media m
                    WHERE m.folder_id = folders.id AND COALESCE(m.ocr_scanned, 0) = 1
                )
                WHERE id = (SELECT folder_id FROM media WHERE id = ?);
                """,
                (int(media_id),),
            )
            conn.commit()

    def reset_ocr_scan_flags(self) -> int:
        """Reset ocr_scanned = 0 for all non-video media in OCR-enabled folders so they are re-scanned."""
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            cur = conn.execute(
                """
                UPDATE media
                SET ocr_scanned = 0
                WHERE is_video = 0
                  AND folder_id IN (
                      SELECT id FROM folders
                      WHERE COALESCE(scan_ocr, 1) = 1
                        AND COALESCE(monitor_mode, 'always') != 'excluded'
                  );
                """
            )
            queued = int(cur.rowcount or 0)
            conn.execute("UPDATE folders SET ocr_scanned_count = 0;")
            conn.commit()
            return queued

    def rename_folder(self, old_path: str, new_path: str, new_name: str) -> None:
        """
        Renames a folder in catalog.db, updating the folder row itself,
        any descendant folders, and any media items located within the folder tree.
        """
        norm_old = str(Path(old_path)).replace("\\", "/")
        norm_new = str(Path(new_path)).replace("\\", "/")
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            # 1. Update the folder itself
            conn.execute(
                """
                UPDATE folders 
                SET path = ?, name = ?
                WHERE REPLACE(LOWER(path), '\\', '/') = LOWER(?);
                """,
                (str(Path(new_path)), new_name, norm_old),
            )
            # 2. Update all descendant subfolders
            sub_rows = conn.execute(
                """
                SELECT id, path FROM folders
                WHERE REPLACE(LOWER(path), '\\', '/') LIKE LOWER(?) || '/%';
                """,
                (norm_old,),
            ).fetchall()
            for r in sub_rows:
                old_sub = str(r["path"]).replace("\\", "/")
                rel = old_sub[len(norm_old):]
                new_sub_path = str(Path(norm_new + rel))
                conn.execute("UPDATE folders SET path = ? WHERE id = ?;", (new_sub_path, r["id"]))

            conn.commit()

    def remove_folder_tree(self, folder_id: int) -> list[int]:
        """
        Removes a folder and all its subfolders from catalog.db and thumbnails.db.
        Crucially, does NOT delete any files or directories on disk.
        Returns the list of removed media IDs.
        """
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            row = conn.execute("SELECT id, path FROM folders WHERE id = ?;", (folder_id,)).fetchone()
            if row is None:
                return []
            root_path = str(row["path"])

            subfolder_rows = conn.execute(
                """
                WITH RECURSIVE subfolder_tree AS (
                    SELECT id FROM folders WHERE id = ?
                    UNION ALL
                    SELECT f.id FROM folders f
                    JOIN subfolder_tree st ON f.parent_id = st.id
                )
                SELECT id FROM subfolder_tree;
                """,
                (folder_id,),
            ).fetchall()
            folder_ids = {int(r["id"]) for r in subfolder_rows}

            prefix_slash = (root_path.rstrip("/\\") + "/").lower()
            prefix_bslash = (root_path.rstrip("/\\") + "\\").lower()
            all_folders = conn.execute("SELECT id, path FROM folders;").fetchall()
            for f in all_folders:
                p_lower = str(f["path"]).lower()
                if p_lower.startswith(prefix_slash) or p_lower.startswith(prefix_bslash):
                    folder_ids.add(int(f["id"]))

            if not folder_ids:
                return []

            f_placeholders = ",".join("?" for _ in folder_ids)
            f_ids_list = list(folder_ids)

            media_rows = conn.execute(
                f"SELECT id FROM media WHERE folder_id IN ({f_placeholders});",
                f_ids_list,
            ).fetchall()
            media_ids = [int(r["id"]) for r in media_rows]

            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(
                f"DELETE FROM folders WHERE id IN ({f_placeholders});",
                f_ids_list,
            )
            conn.commit()

            if media_ids:
                thumb_conn = self.engine.thumbnails_conn()
                m_placeholders = ",".join("?" for _ in media_ids)
                thumb_conn.execute(
                    f"DELETE FROM thumbnails WHERE media_id IN ({m_placeholders});",
                    media_ids,
                )
                thumb_conn.commit()

            return media_ids

    def relocate_folder_tree(self, folder_id: int, new_root_path: str) -> int:
        """
        Updates a root folder and all its descendant subfolders to a new disk path,
        preserving all media IDs, tags, faces, albums, and thumbnails.
        Returns the number of folder rows updated.
        """
        norm_new = os.path.normpath(str(new_root_path))
        new_name = os.path.basename(norm_new) or norm_new
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            row = conn.execute("SELECT id, path FROM folders WHERE id = ?;", (folder_id,)).fetchone()
            if row is None:
                return 0
            old_root = os.path.normpath(str(row["path"]))
            old_prefix_slash = (old_root.rstrip("/\\") + "/").lower()
            old_prefix_bslash = (old_root.rstrip("/\\") + "\\").lower()

            all_folders = conn.execute("SELECT id, path FROM folders;").fetchall()
            updates: list[tuple[str, str, int]] = []
            for f in all_folders:
                fid = int(f["id"])
                f_path = str(f["path"])
                if fid == folder_id:
                    updates.append((norm_new, new_name, fid))
                    continue
                f_lower = os.path.normpath(f_path).lower()
                if f_lower.startswith(old_prefix_slash) or f_lower.startswith(old_prefix_bslash):
                    rel = os.path.normpath(f_path)[len(old_root.rstrip("/\\")) + 1:]
                    updated_path = os.path.normpath(os.path.join(norm_new, rel))
                    updates.append((updated_path, os.path.basename(updated_path) or updated_path, fid))

            if updates:
                conn.executemany(
                    "UPDATE folders SET path = ?, name = ? WHERE id = ?;",
                    updates,
                )
                conn.commit()
            return len(updates)

    def delete_media(self, media_ids: Sequence[int]) -> int:
        """Deletes media records from catalog.db and associated thumbnails from thumbnails.db."""
        if not media_ids:
            return 0
        total_count = 0
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            thumb_conn = self.engine.thumbnails_conn()
            for chunk in _iter_id_chunks(media_ids, 900):
                placeholders = ",".join("?" for _ in chunk)
                total_count += conn.execute(
                    f"DELETE FROM media WHERE id IN ({placeholders});", list(chunk)
                ).rowcount
                thumb_conn.execute(
                    f"DELETE FROM thumbnails WHERE media_id IN ({placeholders});", list(chunk)
                )
            conn.commit()
            thumb_conn.commit()
        return total_count

    def get_rating_counts(self) -> dict[int, int]:
        """Returns item count per rating star (0..5) for the Left Navigation Dock."""
        conn = self.engine.catalog_conn()
        cur = conn.cursor()
        cur.row_factory = None
        counts = {r: 0 for r in range(6)}
        priv_clause = self._private_folder_clause("f")
        for r_val, cnt in cur.execute(
            f"""
            SELECT m.rating, COUNT(*)
            FROM media m
            JOIN folders f ON f.id = m.folder_id
            WHERE COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
            GROUP BY m.rating;
            """
        ).fetchall():
            if r_val is not None:
                counts[int(r_val)] = int(cnt)
        return counts

    def get_date_tree_counts(self) -> list[dict[str, Any]]:
        """
        Aggregates media counts by (year, month, day) with formatted WLPG day label
        (e.g. '1 Jan, Thursday') supporting pre-1970 timestamps on Windows.
        Strictly groups by unique integer (year, month, day) via SQL-side day bucket aggregation.
        """
        conn = self.engine.catalog_conn()
        cur = conn.cursor()
        cur.row_factory = None
        day_buckets: dict[int, int] = {}
        priv_clause = self._private_folder_clause("f")
        for (day_key, cnt) in cur.execute(
            f"""
            SELECT CAST(FLOOR((CASE WHEN m.date_taken > 100000000000 THEN m.date_taken / 1000.0 ELSE m.date_taken * 1.0 END) / 86400.0) AS INTEGER) AS day_key,
                   COUNT(*) AS cnt
            FROM media m
            JOIN folders f ON f.id = m.folder_id
            WHERE m.date_taken IS NOT NULL AND m.date_taken != 0
              AND COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
            GROUP BY day_key;
            """
        ).fetchall():
            if day_key is not None:
                day_buckets[int(day_key)] = int(cnt)

        ymd_counts: dict[tuple[int, int, int], int] = {}
        for day_key, cnt in day_buckets.items():
            try:
                dt = _epoch_to_datetime_utc(day_key * 86400)
                k = (int(dt.year), int(dt.month), int(dt.day))
                ymd_counts[k] = ymd_counts.get(k, 0) + cnt
            except (ValueError, OverflowError, OSError):
                continue

        results: list[dict[str, Any]] = []
        for (y, m, d), cnt in sorted(ymd_counts.items(), key=lambda x: (x[0][0], x[0][1], x[0][2]), reverse=True):
            try:
                dt = datetime.date(y, m, d)
                day_label = f"{d} {dt.strftime('%b')}, {dt.strftime('%A')}"
                m_name = dt.strftime("%B")
            except (ValueError, OverflowError):
                day_label = f"{d}/{m}/{y}"
                m_name = f"Month {m}"

            results.append(
                {
                    "year": int(y),
                    "month": int(m),
                    "month_name": m_name,
                    "day": int(d),
                    "day_label": day_label,
                    "count": int(cnt),
                }
            )
        return results

    # -------------------------------------------------------------------------
    # Media Upsert & Lookup
    # -------------------------------------------------------------------------
    def bulk_upsert_media(self, rows: Sequence[dict[str, Any]]) -> list[int]:
        """Batch-inserts or updates media records in a single transaction, returning their IDs."""
        if not rows:
            return []
        from src.core.geotag_resolver import resolve_location_hierarchy

        ids: list[int] = []
        dim_changed_ids: list[int] = []
        coord_cache: dict[tuple[float, float], tuple[str | None, str | None, str | None, str | None]] = {}
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            cur = conn.cursor()
            cur.row_factory = None
            for r in rows:
                prev = cur.execute(
                    "SELECT id, width, height FROM media WHERE folder_id = ? AND filename = ?;",
                    (r["folder_id"], r["filename"]),
                ).fetchone()
                if prev is not None:
                    prev_id, prev_w, prev_h = prev
                    new_w = r.get("width")
                    new_h = r.get("height")
                    if prev_w is not None and prev_h is not None and new_w is not None and new_h is not None:
                        if (new_w, new_h) != (prev_w, prev_h):
                            dim_changed_ids.append(int(prev_id))

                loc_country = r.get("location_country")
                loc_state = r.get("location_state")
                loc_city = r.get("location_city")
                loc_name = r.get("location_name")
                if not (loc_country or loc_state or loc_city) and (
                    (r.get("lat") is not None and r.get("lon") is not None) or loc_name
                ):
                    loc_country, loc_state, loc_city, resolved_fmt = resolve_location_hierarchy(
                        r.get("lat"), r.get("lon"), loc_name, cache=coord_cache
                    )
                    if not loc_name and resolved_fmt:
                        loc_name = resolved_fmt

                res = cur.execute(
                    """
                    INSERT INTO media (
                        folder_id, filename, ext, size, mtime, date_taken,
                        rating, flag, width, height, orientation,
                        lat, lon, location_name, location_country, location_state, location_city,
                        caption, camera_make, camera_model, author, exposure_time,
                        f_number, focal_length, iso, published,
                        is_video, duration_ms, video_codec, video_fps, metadata_dirty
                    ) VALUES (
                        :folder_id, :filename, :ext, :size, :mtime, :date_taken,
                        :rating, :flag, :width, :height, :orientation,
                        :lat, :lon, :location_name, :location_country, :location_state, :location_city,
                        :caption, :camera_make, :camera_model, :author, :exposure_time,
                        :f_number, :focal_length, :iso, :published,
                        :is_video, :duration_ms, :video_codec, :video_fps, :metadata_dirty
                    )
                    ON CONFLICT(folder_id, filename) DO UPDATE SET
                        size = excluded.size,
                        mtime = excluded.mtime,
                        date_taken = CASE
                            WHEN excluded.date_taken IS NOT NULL THEN excluded.date_taken
                            WHEN media.date_taken < 315532800 AND (media.is_video = 1 OR media.date_taken < -2000000000) THEN NULL
                            ELSE media.date_taken
                        END,
                        rating = CASE WHEN excluded.rating > 0 THEN excluded.rating ELSE media.rating END,
                        flag = CASE WHEN excluded.flag > 0 THEN excluded.flag ELSE media.flag END,
                        face_scan_version = CASE
                            WHEN (excluded.width IS NOT NULL AND media.width IS NOT NULL AND excluded.width != media.width)
                              OR (excluded.height IS NOT NULL AND media.height IS NOT NULL AND excluded.height != media.height)
                            THEN 0
                            ELSE media.face_scan_version
                        END,
                        metadata_dirty = CASE
                            WHEN (media.rating > 0 AND excluded.rating = 0) THEN 1
                            ELSE media.metadata_dirty
                        END,
                        width = COALESCE(excluded.width, media.width),
                        height = COALESCE(excluded.height, media.height),
                        orientation = COALESCE(excluded.orientation, media.orientation),
                        lat = COALESCE(excluded.lat, media.lat),
                        lon = COALESCE(excluded.lon, media.lon),
                        location_name = COALESCE(excluded.location_name, media.location_name),
                        location_country = COALESCE(excluded.location_country, media.location_country),
                        location_state = COALESCE(excluded.location_state, media.location_state),
                        location_city = COALESCE(excluded.location_city, media.location_city),
                        caption = COALESCE(NULLIF(excluded.caption, ''), media.caption),
                        camera_make = COALESCE(excluded.camera_make, media.camera_make),
                        camera_model = COALESCE(excluded.camera_model, media.camera_model),
                        author = COALESCE(excluded.author, media.author),
                        exposure_time = COALESCE(excluded.exposure_time, media.exposure_time),
                        f_number = COALESCE(excluded.f_number, media.f_number),
                        focal_length = COALESCE(excluded.focal_length, media.focal_length),
                        iso = COALESCE(excluded.iso, media.iso),
                        duration_ms = COALESCE(excluded.duration_ms, media.duration_ms),
                        video_codec = COALESCE(excluded.video_codec, media.video_codec),
                        video_fps = COALESCE(excluded.video_fps, media.video_fps)
                    RETURNING id;
                    """,
                    {
                        "folder_id": r["folder_id"],
                        "filename": r["filename"],
                        "ext": r.get("ext", ""),
                        "size": r.get("size", 0),
                        "mtime": r.get("mtime", 0),
                        "date_taken": r.get("date_taken"),
                        "rating": r.get("rating", 0),
                        "flag": r.get("flag", 0),
                        "width": r.get("width"),
                        "height": r.get("height"),
                        "orientation": r.get("orientation", 1),
                        "lat": r.get("lat"),
                        "lon": r.get("lon"),
                        "location_name": loc_name,
                        "location_country": loc_country,
                        "location_state": loc_state,
                        "location_city": loc_city,
                        "caption": r.get("caption", ""),
                        "camera_make": r.get("camera_make"),
                        "camera_model": r.get("camera_model"),
                        "author": r.get("author"),
                        "exposure_time": r.get("exposure_time"),
                        "f_number": r.get("f_number"),
                        "focal_length": r.get("focal_length"),
                        "iso": r.get("iso"),
                        "published": r.get("published", 0),
                        "is_video": r.get("is_video", 0),
                        "duration_ms": r.get("duration_ms"),
                        "video_codec": r.get("video_codec"),
                        "video_fps": r.get("video_fps"),
                        "metadata_dirty": r.get("metadata_dirty", 0),
                    },
                ).fetchone()
                if res:
                    ids.append(int(res[0]))
            conn.commit()

            if dim_changed_ids:
                thumb_conn = self.engine.thumbnails_conn()
                with thumb_conn:
                    for chunk in _iter_id_chunks(dim_changed_ids, 900):
                        ph = ",".join("?" for _ in chunk)
                        thumb_conn.execute(
                            f"DELETE FROM thumbnails WHERE media_id IN ({ph});",
                            list(chunk),
                        )
        return ids

    # -------------------------------------------------------------------------
    # Sub-10ms Virtualized Grid Query Engine
    # -------------------------------------------------------------------------
    def query_media_records(self, spec: FilterQuerySpec) -> list[MediaItemRecord]:
        """Executes an indexed catalog query and returns compact MediaItemRecord instances."""
        conn = self.engine.catalog_conn()
        cur = conn.cursor()
        cur.row_factory = None

        where_clauses: list[str] = [
            f"EXISTS (SELECT 1 FROM folders f WHERE f.id = m.folder_id AND {self._private_folder_clause('f')})",
            "m.live_companion_for_id IS NULL",
        ]
        params: list[Any] = []

        if getattr(spec, "collapse_stacks", True):
            exp_ids = tuple(int(x) for x in (getattr(spec, "expanded_stack_ids", ()) or ()) if x)
            if exp_ids:
                ph = ",".join("?" for _ in exp_ids)
                where_clauses.append(
                    f"(m.stack_parent_id IS NULL OR m.stack_parent_id = m.id OR m.stack_parent_id IN ({ph}))"
                )
                params.extend(exp_ids)
            else:
                where_clauses.append("(m.stack_parent_id IS NULL OR m.stack_parent_id = m.id)")

        # 1. Folder scope (supports single or multi-folder selection)
        raw_fids = list(getattr(spec, "folder_ids", ()) or ())
        if spec.folder_id is not None:
            raw_fids.append(spec.folder_id)
        eff_fids = tuple(dict.fromkeys(int(x) for x in raw_fids if x is not None))
        if eff_fids:
            if spec.include_subfolders:
                if len(eff_fids) == 1:
                    where_clauses.append(
                        """
                        m.folder_id IN (
                            WITH RECURSIVE subfolder_tree(id) AS (
                                SELECT ?
                                UNION ALL
                                SELECT f.id FROM folders f
                                JOIN subfolder_tree st ON f.parent_id = st.id
                            )
                            SELECT id FROM subfolder_tree
                        )
                        """
                    )
                    params.append(eff_fids[0])
                else:
                    values_sql = ", ".join("(?)" for _ in eff_fids)
                    where_clauses.append(
                        f"""
                        m.folder_id IN (
                            WITH RECURSIVE subfolder_tree(id) AS (
                                VALUES {values_sql}
                                UNION ALL
                                SELECT f.id FROM folders f
                                JOIN subfolder_tree st ON f.parent_id = st.id
                            )
                            SELECT id FROM subfolder_tree
                        )
                        """
                    )
                    params.extend(eff_fids)
            else:
                if len(eff_fids) == 1:
                    where_clauses.append("m.folder_id = ?")
                    params.append(eff_fids[0])
                else:
                    placeholders = ",".join("?" for _ in eff_fids)
                    where_clauses.append(f"m.folder_id IN ({placeholders})")
                    params.extend(eff_fids)

        # 2. Date Taken range (uses idx_media_date_taken without per-row strftime overhead)
        if spec.date_from is not None and spec.date_to is not None:
            where_clauses.append("m.date_taken BETWEEN ? AND ?")
            params.extend([int(spec.date_from), int(spec.date_to)])
        else:
            eff_dates = list(getattr(spec, "dates", ()) or ())
            if not eff_dates and spec.date_year is not None:
                eff_dates = [(spec.date_year, spec.date_month, spec.date_day)]
            if eff_dates:
                date_subclauses: list[str] = []
                for d_y, d_m, d_d in eff_dates:
                    if d_y is None:
                        continue
                    start_ts, end_ts = _compute_date_bounds_utc(int(d_y), d_m, d_d)
                    date_subclauses.append("(m.date_taken BETWEEN ? AND ?)")
                    params.extend([start_ts, end_ts])
                if date_subclauses:
                    where_clauses.append(f"({' OR '.join(date_subclauses)})")

        # 3. Hierarchical Descriptive Tags (Boolean AND / OR / NOT via tag_closure)
        if spec.untagged_only:
            where_clauses.append(
                "NOT EXISTS (SELECT 1 FROM media_tags mt WHERE mt.media_id = m.id)"
            )
        elif spec.include_tag_ids:
            if spec.tag_boolean_mode == "OR" or len(spec.include_tag_ids) == 1:
                placeholders = ",".join("?" for _ in spec.include_tag_ids)
                where_clauses.append(
                    f"""
                    m.id IN (
                        SELECT mt.media_id
                        FROM media_tags mt
                        JOIN tag_closure tc ON mt.tag_id = tc.descendant_id
                        WHERE tc.ancestor_id IN ({placeholders})
                    )
                    """
                )
                params.extend(spec.include_tag_ids)
            else:
                # Boolean AND across hierarchical tag subtrees
                intersections = []
                for tid in spec.include_tag_ids:
                    intersections.append(
                        """
                        SELECT mt.media_id
                        FROM media_tags mt
                        JOIN tag_closure tc ON mt.tag_id = tc.descendant_id
                        WHERE tc.ancestor_id = ?
                        """
                    )
                    params.append(tid)
                intersect_sql = " INTERSECT ".join(intersections)
                where_clauses.append(f"m.id IN ({intersect_sql})")

        if spec.exclude_tag_ids:
            placeholders = ",".join("?" for _ in spec.exclude_tag_ids)
            where_clauses.append(
                f"""
                m.id NOT IN (
                    SELECT mt.media_id
                    FROM media_tags mt
                    JOIN tag_closure tc ON mt.tag_id = tc.descendant_id
                    WHERE tc.ancestor_id IN ({placeholders})
                )
                """
            )
            params.extend(spec.exclude_tag_ids)

        # 4. People filter
        if spec.person_ids:
            placeholders = ",".join("?" for _ in spec.person_ids)
            if getattr(spec, "person_boolean_mode", "OR") == "AND" and len(spec.person_ids) > 1:
                where_clauses.append(
                    f"m.id IN (SELECT mp.media_id FROM media_people mp WHERE mp.person_id IN ({placeholders}) AND mp.state = 1 GROUP BY mp.media_id HAVING COUNT(DISTINCT mp.person_id) = {len(spec.person_ids)})"
                )
            else:
                where_clauses.append(
                    f"m.id IN (SELECT mp.media_id FROM media_people mp WHERE mp.person_id IN ({placeholders}) AND mp.state = 1)"
                )
            params.extend(spec.person_ids)

        # 5. Ratings & Flag filter
        if spec.ratings:
            placeholders = ",".join("?" for _ in spec.ratings)
            where_clauses.append(f"m.rating IN ({placeholders})")
            params.extend(spec.ratings)

        flag_val = getattr(spec, "flag_filter", None)
        if flag_val == 1 or spec.flagged_only:
            where_clauses.append("m.flag = 1")
        elif flag_val == -1:
            where_clauses.append("m.is_rejected = 1")
        elif flag_val == 0:
            where_clauses.append("m.flag = 0 AND m.is_rejected = 0")

        if getattr(spec, "favourites_only", False):
            where_clauses.append("m.is_favourite = 1")

        edited_val = getattr(spec, "edited_filter", None)
        if edited_val == 1 or spec.edited_only:
            where_clauses.append("m.original_backup_path IS NOT NULL")
        elif edited_val == 0:
            where_clauses.append("m.original_backup_path IS NULL")

        # 6. Media Type filter
        if spec.media_type == "image":
            raw_clauses = " OR ".join(f"m.filename LIKE '%.{ext}'" for ext in RAW_EXTENSIONS)
            where_clauses.append(f"(m.is_video = 0 AND NOT ({raw_clauses}))")
        elif spec.media_type == "video":
            where_clauses.append("m.is_video = 1")
        elif spec.media_type == "raw":
            raw_clauses = " OR ".join(f"m.filename LIKE '%.{ext}'" for ext in RAW_EXTENSIONS)
            where_clauses.append(f"(m.is_video = 0 AND ({raw_clauses}))")
        elif spec.media_type == "motion":
            where_clauses.append("(m.is_video = 0 AND COALESCE(m.is_live_photo, 0) = 1)")

        # 6b. Orientation filter (landscape vs portrait)
        # media.width and media.height are already stored in upright display dimensions.
        orient_val = getattr(spec, "orientation_filter", "all")
        if orient_val == "landscape":
            where_clauses.append(
                "(COALESCE(m.width, 0) > 0 AND COALESCE(m.height, 0) > 0 AND m.width > m.height)"
            )
        elif orient_val == "portrait":
            where_clauses.append(
                "(COALESCE(m.width, 0) > 0 AND COALESCE(m.height, 0) > 0 AND m.height > m.width)"
            )

        # 6b. Explicit media_ids filter (e.g. from World Map cluster click)
        if spec.media_ids is not None:
            if not spec.media_ids:
                where_clauses.append("1 = 0")
            elif len(spec.media_ids) <= 900:
                placeholders = ",".join("?" for _ in spec.media_ids)
                where_clauses.append(f"m.id IN ({placeholders})")
                params.extend(spec.media_ids)
            else:
                # For very large cluster selections (>900 IDs), chunk into OR clauses to prevent SQLite limit errors
                id_subclauses = []
                for chunk in _iter_id_chunks(spec.media_ids, 900):
                    placeholders = ",".join("?" for _ in chunk)
                    id_subclauses.append(f"m.id IN ({placeholders})")
                    params.extend(chunk)
                where_clauses.append(f"({' OR '.join(id_subclauses)})")

        # 6c. Place / Location hierarchy filter (single or multi-selection from Places nav tree)
        eff_locs: list[tuple[str, str, str]] = [
            ((c or "").strip(), (s or "").strip(), (ci or "").strip())
            for c, s, ci in (getattr(spec, "locations", ()) or ())
            if (c or "").strip() or (s or "").strip() or (ci or "").strip()
        ]
        if not eff_locs:
            loc_country = (getattr(spec, "location_country", None) or "").strip()
            loc_state = (getattr(spec, "location_state", None) or "").strip()
            loc_city = (getattr(spec, "location_city", None) or "").strip()
            if loc_country or loc_state or loc_city:
                eff_locs = [(loc_country, loc_state, loc_city)]
        if eff_locs:
            uk_aliases_map = {
                "England": ("ENG", "ENGLAND"),
                "Wales": ("WLS", "WALES"),
                "Scotland": ("SCT", "SCOTLAND"),
                "Northern Ireland": ("NIR", "NORTHERN IRELAND"),
            }
            loc_or_parts: list[str] = []
            for loc_c, loc_s, loc_ci in eff_locs:
                sub_parts: list[str] = []
                if loc_c:
                    sub_parts.append("TRIM(COALESCE(m.location_country, '')) = ?")
                    params.append(loc_c)
                if loc_s:
                    if loc_c in ("United Kingdom", "GB", "UK") and loc_s in uk_aliases_map:
                        sub_parts.append("UPPER(TRIM(COALESCE(m.location_state, ''))) IN (?, ?)")
                        params.extend(uk_aliases_map[loc_s])
                    else:
                        sub_parts.append("TRIM(COALESCE(m.location_state, '')) = ?")
                        params.append(loc_s)
                if loc_ci:
                    sub_parts.append("TRIM(COALESCE(m.location_city, '')) = ?")
                    params.append(loc_ci)
                if sub_parts:
                    loc_or_parts.append(f"({' AND '.join(sub_parts)})")
            if loc_or_parts:
                where_clauses.append(f"({' OR '.join(loc_or_parts)})")

        # 6d. Camera Make / Model filter (single or multi-selection from Cameras nav section)
        eff_cams: list[str] = [
            c.strip() for c in (getattr(spec, "camera_models", ()) or ()) if (c or "").strip()
        ]
        if not eff_cams:
            cam_filter = (getattr(spec, "camera_model", None) or "").strip()
            if cam_filter:
                eff_cams = [cam_filter]
        if eff_cams:
            norm_cam_expr = (
                "TRIM(CASE "
                "WHEN COALESCE(TRIM(m.camera_make), '') != '' AND COALESCE(TRIM(m.camera_model), '') != '' "
                "AND LOWER(TRIM(m.camera_model)) NOT LIKE LOWER(TRIM(m.camera_make)) || '%' "
                "THEN TRIM(m.camera_make) || ' ' || TRIM(m.camera_model) "
                "WHEN COALESCE(TRIM(m.camera_model), '') != '' THEN TRIM(m.camera_model) "
                "ELSE COALESCE(TRIM(m.camera_make), '') END)"
            )
            cam_or_parts: list[str] = []
            for cam_entry in eff_cams:
                if cam_entry == "__NO_CAMERA__":
                    cam_or_parts.append(
                        "(COALESCE(TRIM(m.camera_make), '') = '' AND COALESCE(TRIM(m.camera_model), '') = '')"
                    )
                else:
                    cam_or_parts.append(f"LOWER({norm_cam_expr}) = LOWER(?)")
                    params.append(cam_entry)
            if cam_or_parts:
                where_clauses.append(f"({' OR '.join(cam_or_parts)})")

        # 7. World Map Bounding Box filter (via media_rtree virtual table or media.lat/lon fallback)
        if spec.bbox is not None:
            min_lat, max_lat, min_lon, max_lon = spec.bbox
            real_min_lat = min(min_lat, max_lat)
            real_max_lat = max(min_lat, max_lat)
            real_min_lon = min(min_lon, max_lon)
            real_max_lon = max(min_lon, max_lon)
            where_clauses.append(
                """
                (
                    (m.lat IS NOT NULL AND m.lon IS NOT NULL
                     AND m.lat >= ? AND m.lat <= ?
                     AND m.lon >= ? AND m.lon <= ?)
                    OR m.id IN (
                        SELECT r.id FROM media_rtree r
                        WHERE r.max_lat >= ? AND r.min_lat <= ?
                          AND r.max_lon >= ? AND r.min_lon <= ?
                    )
                )
                """
            )
            params.extend([
                real_min_lat, real_max_lat, real_min_lon, real_max_lon,
                real_min_lat, real_max_lat, real_min_lon, real_max_lon,
            ])

        # 8. Full Boolean text search & people/tag/OCR query
        if spec.search_text.strip():
            b_sql, b_params = _build_boolean_search_sql(
                spec.search_text.strip(),
                use_ocr_fts=self._supports_ocr_fts(),
            )
            if b_sql:
                where_clauses.append(b_sql)
                params.extend(b_params)

        # 8b. Static Album & Smart Album filters
        if getattr(spec, "album_id", None) is not None:
            where_clauses.append(
                "m.id IN (SELECT ai.media_id FROM album_items ai WHERE ai.album_id = ?)"
            )
            params.append(int(spec.album_id))

        sa_rules_raw = getattr(spec, "smart_album_rules", None)
        sa_id = getattr(spec, "smart_album_id", None)
        if sa_rules_raw is None and sa_id is not None:
            sa_row = conn.execute(
                "SELECT rules_json FROM smart_albums WHERE id = ?;", (int(sa_id),)
            ).fetchone()
            sa_rules_raw = sa_row[0] if sa_row else None
        if sa_rules_raw:
            sa_sql, sa_params = compile_smart_album_where(sa_rules_raw, alias="m")
            if sa_sql:
                where_clauses.append(f"({sa_sql})")
                params.extend(sa_params)
        elif sa_id is not None:
            where_clauses.append("1 = 0")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # 9. Sort / Arrange clause: Group first, then sort within group
        group_mode = (getattr(spec, "group_by", "month") or "month").lower()
        sort_mode = (getattr(spec, "sort_by", "date") or "date").lower()

        # Backward compatibility with arrange_by
        if spec.arrange_by != "date" and group_mode in ("month", "auto"):
            group_mode = spec.arrange_by

        grp_dir = "ASC" if getattr(spec, "group_reverse", False) else "DESC"
        grp_expr = None
        extra_joins = ""
        extra_col = "'' AS group_name"

        if group_mode in ("auto",):
            grp_expr = "strftime('%Y-%m', datetime(COALESCE(m.date_taken, m.mtime, 0), 'unixepoch'))"
        elif group_mode in ("date", "date taken", "day"):
            grp_expr = "strftime('%Y-%m-%d', datetime(COALESCE(m.date_taken, m.mtime, 0), 'unixepoch'))"
        elif group_mode in ("month", "month taken"):
            grp_expr = "strftime('%Y-%m', datetime(COALESCE(m.date_taken, m.mtime, 0), 'unixepoch'))"
        elif group_mode in ("year", "year taken"):
            grp_expr = "strftime('%Y', datetime(COALESCE(m.date_taken, m.mtime, 0), 'unixepoch'))"
        elif group_mode in ("time", "time taken"):
            grp_expr = "CASE WHEN m.date_taken IS NULL THEN 5 WHEN strftime('%H', datetime(m.date_taken, 'unixepoch')) BETWEEN '06' AND '11' THEN 1 WHEN strftime('%H', datetime(m.date_taken, 'unixepoch')) BETWEEN '12' AND '17' THEN 2 WHEN strftime('%H', datetime(m.date_taken, 'unixepoch')) BETWEEN '18' AND '21' THEN 3 ELSE 4 END"
            grp_dir = "DESC" if getattr(spec, "group_reverse", False) else "ASC"
        elif group_mode in ("size", "file size", "file_size"):
            grp_expr = "CASE WHEN m.size >= 104857600 THEN 1 WHEN m.size >= 26214400 THEN 2 WHEN m.size >= 5242880 THEN 3 WHEN m.size >= 1048576 THEN 4 ELSE 5 END"
            grp_dir = "DESC" if getattr(spec, "group_reverse", False) else "ASC"
        elif group_mode in ("image_size", "image size", "dimensions"):
            grp_expr = "CASE WHEN (COALESCE(m.width, 0) * COALESCE(m.height, 0)) >= 16000000 THEN 1 WHEN (COALESCE(m.width, 0) * COALESCE(m.height, 0)) >= 8000000 THEN 2 WHEN (COALESCE(m.width, 0) * COALESCE(m.height, 0)) >= 2000000 THEN 3 WHEN (COALESCE(m.width, 0) * COALESCE(m.height, 0)) > 0 THEN 4 ELSE 5 END"
            grp_dir = "DESC" if getattr(spec, "group_reverse", False) else "ASC"
        elif group_mode in ("orientation",):
            grp_expr = "CASE WHEN COALESCE(m.width, 0) > 0 AND COALESCE(m.height, 0) > 0 THEN CASE WHEN m.width > m.height THEN 1 WHEN m.height > m.width THEN 2 ELSE 3 END ELSE 4 END"
            grp_dir = "DESC" if getattr(spec, "group_reverse", False) else "ASC"
        elif group_mode in ("file_type", "file type"):
            grp_expr = "LOWER(COALESCE(m.ext, ''))"
            grp_dir = "DESC" if getattr(spec, "group_reverse", False) else "ASC"
        elif group_mode in ("media_type", "media type"):
            grp_expr = "m.is_video"
            grp_dir = "DESC" if getattr(spec, "group_reverse", False) else "ASC"
        elif group_mode in ("rating",):
            grp_expr = "m.rating"
            grp_dir = "ASC" if getattr(spec, "group_reverse", False) else "DESC"
        elif group_mode in ("flag",):
            grp_expr = "CASE WHEN COALESCE(m.flag, 0) = 1 THEN 2 WHEN COALESCE(m.is_rejected, 0) = 1 THEN 1 ELSE 0 END"
            grp_dir = "ASC" if getattr(spec, "group_reverse", False) else "DESC"
            extra_col = "CASE WHEN COALESCE(m.flag, 0) = 1 THEN 'Picked / Flagged' WHEN COALESCE(m.is_rejected, 0) = 1 THEN 'Rejected' ELSE 'Not flagged' END AS group_name"
        elif group_mode in ("favourite", "favourites"):
            grp_expr = "COALESCE(m.is_favourite, 0)"
            grp_dir = "ASC" if getattr(spec, "group_reverse", False) else "DESC"
            extra_col = "CASE WHEN COALESCE(m.is_favourite, 0) = 1 THEN 'Favourites' ELSE 'Not favourited' END AS group_name"
        elif group_mode in ("folder",):
            grp_expr = "COALESCE(f.name, '') COLLATE NOCASE"
            grp_dir = "DESC" if getattr(spec, "group_reverse", False) else "ASC"
        elif group_mode in ("published",):
            grp_expr = "m.published"
            grp_dir = "ASC" if getattr(spec, "group_reverse", False) else "DESC"
        elif group_mode in ("geotag", "location"):
            grp_expr = "CASE WHEN m.location_name IS NOT NULL AND m.location_name != '' THEN 1 WHEN m.lat IS NOT NULL THEN 2 ELSE 3 END, COALESCE(m.location_name, '') COLLATE NOCASE"
            grp_dir = "DESC" if getattr(spec, "group_reverse", False) else "ASC"
        elif group_mode in ("camera",):
            grp_expr = "CASE WHEN (m.camera_make IS NULL OR m.camera_make = '') AND (m.camera_model IS NULL OR m.camera_model = '') THEN 1 ELSE 0 END, COALESCE(NULLIF(TRIM(COALESCE(m.camera_make, '') || ' ' || COALESCE(m.camera_model, '')), ''), NULLIF(m.camera_model, ''), NULLIF(m.camera_make, ''), 'Unknown Camera') COLLATE NOCASE"
            grp_dir = "DESC" if getattr(spec, "group_reverse", False) else "ASC"
        elif group_mode in ("person", "people"):
            extra_joins = "LEFT JOIN (SELECT DISTINCT media_id, person_id FROM media_people WHERE state = 1) mp ON mp.media_id = m.id LEFT JOIN people p ON p.id = mp.person_id"
            extra_col = "COALESCE(p.name, 'Not tagged') AS group_name"
            p_dir = "DESC" if getattr(spec, "group_reverse", False) else "ASC"
            grp_expr = f"CASE WHEN p.name IS NULL THEN 1 ELSE 0 END, COALESCE(p.name, 'Not tagged') COLLATE NOCASE {p_dir}"
        elif group_mode in ("tag", "tags"):
            extra_joins = "LEFT JOIN (SELECT DISTINCT media_id, tag_id FROM media_tags) mt ON mt.media_id = m.id LEFT JOIN tags t ON t.id = mt.tag_id"
            extra_col = "COALESCE(t.name, 'Not tagged') AS group_name"
            t_dir = "DESC" if getattr(spec, "group_reverse", False) else "ASC"
            grp_expr = f"CASE WHEN t.name IS NULL THEN 1 ELSE 0 END, COALESCE(t.name, 'Not tagged') COLLATE NOCASE {t_dir}"
        elif group_mode == "name":
            name_dir = "DESC" if (getattr(spec, "group_reverse", False) or getattr(spec, "sort_reverse", False) or getattr(spec, "reverse_sort", False)) else "ASC"
            grp_expr = f"m.filename COLLATE NOCASE {name_dir}"
        elif group_mode in ("none", "flat"):
            grp_expr = None

        is_sort_rev = getattr(spec, "sort_reverse", False) or getattr(spec, "reverse_sort", False)
        if sort_mode == "name":
            sort_dir = "DESC" if is_sort_rev else "ASC"
            sort_expr = "m.filename COLLATE NOCASE"
        elif sort_mode == "rating":
            sort_dir = "ASC" if is_sort_rev else "DESC"
            sort_expr = "m.rating"
        elif sort_mode == "size":
            sort_dir = "ASC" if is_sort_rev else "DESC"
            sort_expr = "m.size"
        elif sort_mode == "media_type":
            sort_dir = "ASC" if is_sort_rev else "DESC"
            sort_expr = "m.is_video"
        else:  # date
            sort_dir = "ASC" if is_sort_rev else "DESC"
            sort_expr = "COALESCE(m.date_taken, m.mtime, 0)"

        # Fast-path ORDER BY: when grouping by year/month/date and sorting by date in the same direction,
        # chronological order is already strictly grouped by year/month/day, allowing SQLite to use
        # idx_media_effective_date directly without 500k strftime() calls or temporary B-tree sorting.
        is_chrono_group = group_mode in (
            "auto",
            "month",
            "month taken",
            "date",
            "date taken",
            "day",
            "year",
            "year taken",
        )
        if is_chrono_group and sort_mode == "date" and grp_dir == sort_dir:
            order_sql = f"ORDER BY COALESCE(m.date_taken, m.mtime, 0) {sort_dir}, m.id ASC"
        elif grp_expr is not None:
            if " COLLATE " in grp_expr or grp_expr.endswith(" ASC") or grp_expr.endswith(" DESC"):
                order_sql = f"ORDER BY {grp_expr}, {sort_expr} {sort_dir}, m.id ASC"
            else:
                order_sql = f"ORDER BY {grp_expr} {grp_dir}, {sort_expr} {sort_dir}, m.id ASC"
        else:
            order_sql = f"ORDER BY {sort_expr} {sort_dir}, m.id ASC"

        # Fast-path 13-column SELECT when extra joins/columns are not needed
        if group_mode in ("month", "month taken", "auto", "none", "flat", "year", "year taken", "date", "date taken", "day"):
            sql_fast = f"""
                SELECT
                    m.id,
                    m.folder_id,
                    m.filename,
                    m.size,
                    m.mtime,
                    m.date_taken,
                    m.rating,
                    m.flag,
                    COALESCE(m.width, 0),
                    COALESCE(m.height, 0),
                    m.orientation,
                    m.is_video,
                    m.caption
                FROM media m
                {where_sql}
                {order_sql};
            """
            raw_rows = cur.execute(sql_fast, params).fetchall()
            _Record = MediaItemRecord
            _epoch_to_dt = _epoch_to_datetime_utc
            _date_bounds = _compute_date_bounds_utc
            records: list[MediaItemRecord] = []

            if group_mode in ("month", "month taken", "auto"):
                cur_m_start = 1
                cur_m_end = 0
                cur_m_label = "Unknown Date"
                for mid, fid, fname, size, mtime, dt_taken, rating, flag, w, h, orient, is_vid, caption in raw_rows:
                    eff_dt = dt_taken if dt_taken is not None else (mtime if mtime else None)
                    if eff_dt is None:
                        gkey = "Unknown Date"
                        dt_val = 0
                    else:
                        dt_val = eff_dt
                        if cur_m_start <= dt_val <= cur_m_end:
                            gkey = cur_m_label
                        else:
                            dt = _epoch_to_dt(dt_val)
                            cur_m_start, cur_m_end = _date_bounds(dt.year, dt.month)
                            cur_m_label = dt.strftime("%B %Y")
                            gkey = cur_m_label
                    records.append(
                        _Record(mid, fid, fname, size, mtime, dt_val, rating, flag, w, h, orient, is_vid, caption, gkey)
                    )
                return self._annotate_media_records(cur, records)

            if group_mode in ("none", "flat"):
                return self._annotate_media_records(
                    cur,
                    [
                        _Record(mid, fid, fname, size, mtime, dt_taken if dt_taken is not None else (mtime or 0), rating, flag, w, h, orient, is_vid, caption, "")
                        for mid, fid, fname, size, mtime, dt_taken, rating, flag, w, h, orient, is_vid, caption in raw_rows
                    ],
                )

            if group_mode in ("year", "year taken"):
                cur_y_start = 1
                cur_y_end = 0
                cur_y_label = "Unknown Date"
                for mid, fid, fname, size, mtime, dt_taken, rating, flag, w, h, orient, is_vid, caption in raw_rows:
                    eff_dt = dt_taken if dt_taken is not None else (mtime if mtime else None)
                    if eff_dt is None:
                        gkey = "Unknown Date"
                        dt_val = 0
                    else:
                        dt_val = eff_dt
                        if cur_y_start <= dt_val <= cur_y_end:
                            gkey = cur_y_label
                        else:
                            dt = _epoch_to_dt(dt_val)
                            cur_y_start, cur_y_end = _date_bounds(dt.year)
                            cur_y_label = str(dt.year)
                            gkey = cur_y_label
                    records.append(
                        _Record(mid, fid, fname, size, mtime, dt_val, rating, flag, w, h, orient, is_vid, caption, gkey)
                    )
                return self._annotate_media_records(cur, records)

            if group_mode in ("date", "date taken", "day"):
                cur_d_start = 1
                cur_d_end = 0
                cur_d_label = "Unknown Date"
                for mid, fid, fname, size, mtime, dt_taken, rating, flag, w, h, orient, is_vid, caption in raw_rows:
                    eff_dt = dt_taken if dt_taken is not None else (mtime if mtime else None)
                    if eff_dt is None:
                        gkey = "Unknown Date"
                        dt_val = 0
                    else:
                        dt_val = eff_dt
                        if cur_d_start <= dt_val <= cur_d_end:
                            gkey = cur_d_label
                        else:
                            dt = _epoch_to_dt(dt_val)
                            cur_d_start, cur_d_end = _date_bounds(dt.year, dt.month, dt.day)
                            cur_d_label = dt.strftime("%A, %d %B %Y")
                            gkey = cur_d_label
                    records.append(
                        _Record(mid, fid, fname, size, mtime, dt_val, rating, flag, w, h, orient, is_vid, caption, gkey)
                    )
                return self._annotate_media_records(cur, records)

        sql = f"""
            SELECT
                m.id,
                m.folder_id,
                m.filename,
                m.size,
                m.mtime,
                m.date_taken,
                m.rating,
                m.flag,
                COALESCE(m.width, 0),
                COALESCE(m.height, 0),
                m.orientation,
                m.is_video,
                m.caption,
                COALESCE(m.location_name, ''),
                COALESCE(m.ext, ''),
                COALESCE(m.published, 0),
                COALESCE(NULLIF(TRIM(COALESCE(m.camera_make, '') || ' ' || COALESCE(m.camera_model, '')), ''), NULLIF(m.camera_model, ''), NULLIF(m.camera_make, ''), 'Unknown Camera') AS camera_name,
                COALESCE(f.name, '') AS folder_name,
                m.lat,
                m.lon,
                {extra_col}
            FROM media m
            LEFT JOIN folders f ON f.id = m.folder_id
            {extra_joins}
            {where_sql}
            {order_sql};
        """

        raw_rows = cur.execute(sql, params).fetchall()

        cur_m_start = 1
        cur_m_end = 0
        cur_m_label = "Unknown Date"
        records = []

        for row in raw_rows:
            (
                mid,
                fid,
                fname,
                size,
                mtime,
                dt_taken,
                rating,
                flag,
                w,
                h,
                orient,
                is_vid,
                caption,
                loc_name,
                ext,
                published,
                camera_name,
                folder_name,
                lat,
                lon,
            ) = row[:20]
            g_extra = row[20] if len(row) > 20 else ""

            eff_dt = dt_taken if dt_taken is not None else (mtime if mtime else None)
            if group_mode in ("none", "flat"):
                gkey = ""
            elif group_mode in ("person", "people"):
                gkey = g_extra or "Not tagged"
            elif group_mode in ("tag", "tags"):
                gkey = g_extra or "Not tagged"
            elif group_mode == "name":
                gkey = "All photos and videos"
            elif group_mode in ("date", "date taken", "day"):
                if eff_dt is None:
                    gkey = "Unknown Date"
                else:
                    dt = _epoch_to_datetime_utc(eff_dt)
                    gkey = dt.strftime("%A, %d %B %Y")
            elif group_mode in ("month", "month taken", "auto"):
                if eff_dt is None:
                    gkey = "Unknown Date"
                elif cur_m_start <= eff_dt <= cur_m_end:
                    gkey = cur_m_label
                else:
                    dt = _epoch_to_datetime_utc(eff_dt)
                    cur_m_start, cur_m_end = _compute_date_bounds_utc(dt.year, dt.month)
                    cur_m_label = dt.strftime("%B %Y")
                    gkey = cur_m_label
            elif group_mode in ("year", "year taken"):
                if eff_dt is None:
                    gkey = "Unknown Date"
                else:
                    dt = _epoch_to_datetime_utc(eff_dt)
                    gkey = dt.strftime("%Y")
            elif group_mode in ("time", "time taken"):
                if dt_taken is None:
                    gkey = "Unknown Time"
                else:
                    dt = _epoch_to_datetime_utc(dt_taken)
                    hour = dt.hour
                    if 6 <= hour < 12:
                        gkey = "Morning (6:00 AM - 12:00 PM)"
                    elif 12 <= hour < 18:
                        gkey = "Afternoon (12:00 PM - 6:00 PM)"
                    elif 18 <= hour < 22:
                        gkey = "Evening (6:00 PM - 10:00 PM)"
                    else:
                        gkey = "Night (10:00 PM - 6:00 AM)"
            elif group_mode in ("size", "file size", "file_size"):
                if size >= 104857600:
                    gkey = "Huge (> 100 MB)"
                elif size >= 26214400:
                    gkey = "Large (25 - 100 MB)"
                elif size >= 5242880:
                    gkey = "Medium (5 - 25 MB)"
                elif size >= 1048576:
                    gkey = "Small (1 - 5 MB)"
                else:
                    gkey = "Tiny (< 1 MB)"
            elif group_mode in ("image_size", "image size", "dimensions"):
                mp = (w * h) / 1_000_000.0 if (w and h) else 0.0
                if mp >= 16.0:
                    gkey = "Extra Large (> 16 MP)"
                elif mp >= 8.0:
                    gkey = "Large (8 - 16 MP)"
                elif mp >= 2.0:
                    gkey = "Medium (2 - 8 MP)"
                elif mp > 0.0:
                    gkey = "Small (< 2 MP)"
                else:
                    gkey = "Unknown Size"
            elif group_mode in ("orientation",):
                if w > 0 and h > 0 and w > h:
                    gkey = "Landscape"
                elif w > 0 and h > 0 and h > w:
                    gkey = "Portrait"
                elif w > 0 and w == h:
                    gkey = "Square"
                else:
                    gkey = "Unknown Orientation"
            elif group_mode in ("file_type", "file type"):
                e = (ext or "").lower().strip()
                if e in (".jpg", ".jpeg"):
                    gkey = "JPEG Image (*.jpg, *.jpeg)"
                elif e == ".png":
                    gkey = "PNG Image (*.png)"
                elif e in (".dng", ".cr2", ".nef", ".arw", ".raf", ".rw2", ".orf"):
                    gkey = "RAW Digital Negative (*.raw, *.dng)"
                elif e == ".mp4":
                    gkey = "MP4 Video (*.mp4)"
                elif e in (".mov", ".qt"):
                    gkey = "QuickTime Video (*.mov)"
                elif e == ".webp":
                    gkey = "WebP Image (*.webp)"
                elif e in (".heic", ".heif", ".hif"):
                    gkey = "HEIF Image (*.heic, *.heif)"
                elif e == ".avif":
                    gkey = "AVIF Image (*.avif)"
                elif e == ".jxl":
                    gkey = "JPEG XL Image (*.jxl)"
                elif e in (".tif", ".tiff"):
                    gkey = "TIFF Image (*.tif, *.tiff)"
                elif e == ".gif":
                    gkey = "GIF Image (*.gif)"
                elif e:
                    gkey = f"{e[1:].upper()} File (*{e})"
                else:
                    gkey = "Unknown Format"
            elif group_mode in ("media_type", "media type"):
                gkey = "Videos" if is_vid else "Photos"
            elif group_mode in ("rating",):
                gkey = _RATING_LABELS.get(rating, "Unrated")
            elif group_mode in ("flag",):
                gkey = g_extra or ("Picked / Flagged" if flag else "Not flagged")
            elif group_mode in ("favourite", "favourites"):
                gkey = g_extra or "Not favourited"
            elif group_mode in ("folder",):
                gkey = folder_name or f"Folder {fid}"
            elif group_mode in ("published",):
                gkey = "Published" if published else "Not published"
            elif group_mode in ("geotag", "location"):
                if loc_name:
                    gkey = loc_name
                elif lat is not None and lon is not None:
                    gkey = "Geotagged"
                else:
                    gkey = "No Geotag"
            elif group_mode in ("camera",):
                gkey = camera_name or "Unknown Camera"
            else:
                if eff_dt is None:
                    gkey = "Unknown Date"
                elif cur_m_start <= eff_dt <= cur_m_end:
                    gkey = cur_m_label
                else:
                    dt = _epoch_to_datetime_utc(eff_dt)
                    cur_m_start, cur_m_end = _compute_date_bounds_utc(dt.year, dt.month)
                    cur_m_label = dt.strftime("%B %Y")
                    gkey = cur_m_label

            records.append(
                MediaItemRecord(
                    id=mid,
                    folder_id=fid,
                    filename=fname,
                    size=size,
                    mtime=mtime,
                    date_taken=0 if eff_dt is None else eff_dt,
                    rating=rating,
                    flag=flag,
                    width=w,
                    height=h,
                    orientation=orient,
                    is_video=is_vid,
                    caption=caption,
                    group_key=gkey,
                )
            )
        return self._annotate_media_records(cur, records)

    def _annotate_media_records(
        self, cur: sqlite3.Cursor, records: list[MediaItemRecord]
    ) -> list[MediaItemRecord]:
        """Annotates live-photo, offline-volume, and stack metadata with zero overhead when unused."""
        if not records:
            return records
        offline_folders = {
            int(r[0]) for r in cur.execute("SELECT id FROM folders WHERE volume_offline = 1").fetchall()
        }
        # Fast-path: for compact result sets (<=500 items), look up only those IDs via PK index
        if len(records) <= 500:
            rec_ids = [int(r.id) for r in records]
            placeholders = ",".join("?" for _ in rec_ids)
            meta_rows = cur.execute(
                f"""
                SELECT id, COALESCE(is_live_photo, 0), COALESCE(is_favourite, 0),
                       COALESCE(is_rejected, 0), stack_parent_id, LOWER(COALESCE(ext, '')),
                       live_companion_for_id
                FROM media
                WHERE (id IN ({placeholders}) OR stack_parent_id IN ({placeholders}))
                  AND (is_live_photo = 1 OR is_favourite = 1 OR is_rejected = 1 OR stack_parent_id IS NOT NULL);
                """,
                list(rec_ids) + list(rec_ids),
            ).fetchall()
        else:
            # Avoid COALESCE() wrapper on filter predicates so SQLite can evaluate/index directly
            meta_rows = cur.execute(
                """
                SELECT id, COALESCE(is_live_photo, 0), COALESCE(is_favourite, 0),
                       COALESCE(is_rejected, 0), stack_parent_id, LOWER(COALESCE(ext, '')),
                       live_companion_for_id
                FROM media
                WHERE is_live_photo = 1
                   OR is_favourite = 1
                   OR is_rejected = 1
                   OR stack_parent_id IS NOT NULL;
                """
            ).fetchall()
        live_ids: set[int] = set()
        fav_ids: set[int] = set()
        rej_ids: set[int] = set()
        stack_rows: list[tuple[int, int, str]] = []
        for mid_raw, lp, fav, rej, sp_raw, ext_raw, live_comp in meta_rows:
            mid = int(mid_raw)
            if lp:
                live_ids.add(mid)
            if fav:
                fav_ids.add(mid)
            if rej:
                rej_ids.add(mid)
            if sp_raw is not None and live_comp is None:
                stack_rows.append((mid, int(sp_raw), str(ext_raw or "")))
        if not offline_folders and not live_ids and not fav_ids and not rej_ids and not stack_rows:
            return records

        raw_exts = {"nef", "cr2", "cr3", "arw", "dng", "raf", "orf", "rw2", "pef", "srw"}
        jpg_exts = {"jpg", "jpeg"}
        stack_members: dict[int, list[str]] = {}
        stack_parents: dict[int, int] = {}
        for mid_raw, sp_raw, ext_raw in stack_rows:
            mid = int(mid_raw)
            sp = int(sp_raw)
            stack_parents[mid] = sp
            stack_members.setdefault(sp, []).append(str(ext_raw or "").lstrip("."))

        raw_jpg_heads: set[int] = set()
        for sp, exts in stack_members.items():
            if len(exts) == 2:
                has_raw = any(e in raw_exts for e in exts)
                has_jpg = any(e in jpg_exts for e in exts)
                if has_raw and has_jpg:
                    raw_jpg_heads.add(sp)

        in_view_by_sp: dict[int, list[MediaItemRecord]] = {}
        for rec in records:
            if rec.folder_id in offline_folders:
                rec.is_offline = 1
            if rec.id in live_ids:
                rec.is_live_photo = 1
            if rec.id in fav_ids:
                rec.is_favourite = 1
            if rec.id in rej_ids:
                rec.flag = -1
            sp = stack_parents.get(rec.id)
            if sp is not None:
                rec.stack_parent_id = sp
                in_view_by_sp.setdefault(sp, []).append(rec)

        for sp, view_recs in in_view_by_sp.items():
            total_cnt = max(1, len(stack_members.get(sp, ())))
            if total_cnt <= 1:
                continue
            is_rj = sp in raw_jpg_heads
            if len(view_recs) > 1:
                for pos_idx, r in enumerate(view_recs, start=1):
                    r.stack_count = total_cnt
                    r.stack_pos = pos_idx
                    r.is_raw_jpg_stack = is_rj
            else:
                r = view_recs[0]
                if r.id == sp:
                    r.stack_count = total_cnt
                    r.stack_pos = 0
                    r.is_raw_jpg_stack = is_rj
        return records

    def batch_update_favourite(self, media_ids: Sequence[int], is_favourite: bool) -> int:
        ids = [int(x) for x in dict.fromkeys(media_ids) if int(x) > 0]
        if not ids:
            return 0
        val = 1 if is_favourite else 0
        changed = 0
        with self.engine.catalog_transaction() as conn:
            cur = conn.cursor()
            for chunk in _iter_id_chunks(ids, 900):
                ph = ",".join("?" for _ in chunk)
                cur.execute(
                    f"UPDATE media SET is_favourite = ? WHERE id IN ({ph});",
                    [val, *chunk],
                )
                changed += int(cur.rowcount or 0)
        return changed

    def get_rejected_media_ids(self) -> list[int]:
        conn = self.engine.catalog_conn()
        rows = conn.execute(
            "SELECT id FROM media WHERE is_rejected = 1 AND live_companion_for_id IS NULL ORDER BY id ASC;"
        ).fetchall()
        return [int(r[0]) for r in rows]

    def get_favourite_and_flag_counts(self) -> dict[str, int]:
        conn = self.engine.catalog_conn()
        row = conn.execute(
            """
            SELECT
                SUM(CASE WHEN is_favourite = 1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN flag = 1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN is_rejected = 1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN COALESCE(flag, 0) = 0 AND COALESCE(is_rejected, 0) = 0 THEN 1 ELSE 0 END)
            FROM media
            WHERE live_companion_for_id IS NULL;
            """
        ).fetchone()
        return {
            "favourites": int((row[0] if row else 0) or 0),
            "picked": int((row[1] if row else 0) or 0),
            "rejected": int((row[2] if row else 0) or 0),
            "unflagged": int((row[3] if row else 0) or 0),
        }

    # -------------------------------------------------------------------------
    # Offline Volumes, Live/Motion Photos & Image Stacking
    # -------------------------------------------------------------------------
    def set_folder_volume_offline(self, folder_id: int, offline: bool) -> None:
        with self.engine.catalog_transaction() as conn:
            conn.execute(
                "UPDATE folders SET volume_offline = ? WHERE id = ?;",
                (1 if offline else 0, int(folder_id)),
            )

    def update_live_photo_metadata(
        self, updates: Sequence[tuple[int, int, str, int]]
    ) -> None:
        """Batch-updates `(is_live_photo, motion_offset, live_video_path, media_id)`."""
        if not updates:
            return
        with self.engine.catalog_transaction() as conn:
            conn.executemany(
                "UPDATE media SET is_live_photo = ?, motion_offset = ?, live_video_path = ? WHERE id = ?;",
                [(int(lp), int(off), str(vp or ""), int(mid)) for lp, off, vp, mid in updates],
            )

    def mark_live_companion_videos(self, pairs: Sequence[tuple[int, int]]) -> None:
        """Marks `(still_media_id, mov_media_id)` so the companion .MOV is hidden from grid clutter."""
        if not pairs:
            return
        with self.engine.catalog_transaction() as conn:
            conn.executemany(
                "UPDATE media SET live_companion_for_id = ? WHERE id = ?;",
                [(int(still_id), int(mov_id)) for still_id, mov_id in pairs],
            )

    def create_stack(self, media_ids: Sequence[int], head_id: int | None = None) -> int | None:
        ids = [int(x) for x in dict.fromkeys(media_ids) if int(x) > 0]
        if len(ids) < 2:
            return None
        chosen_head = int(head_id) if head_id is not None and int(head_id) in ids else ids[0]
        with self.engine.catalog_transaction() as conn:
            conn.executemany(
                "UPDATE media SET stack_parent_id = ? WHERE id = ?;",
                [(chosen_head, mid) for mid in ids],
            )
        return chosen_head

    def unstack_media(self, media_ids: Sequence[int]) -> int:
        ids = [int(x) for x in dict.fromkeys(media_ids) if int(x) > 0]
        if not ids:
            return 0
        with self.engine.catalog_transaction() as conn:
            cur = conn.cursor()
            placeholders = ",".join("?" for _ in ids)
            # Clear both selected items and any children whose stack_parent_id is in ids
            cur.execute(
                f"UPDATE media SET stack_parent_id = NULL WHERE id IN ({placeholders}) OR stack_parent_id IN ({placeholders});",
                ids + ids,
            )
            changed = int(cur.rowcount or 0)
        return changed

    def get_stack_member_ids(self, media_id: int) -> list[int]:
        conn = self.engine.catalog_conn()
        cur = conn.cursor()
        row = cur.execute(
            "SELECT stack_parent_id FROM media WHERE id = ?;", (int(media_id),)
        ).fetchone()
        if not row or row[0] is None:
            return [int(media_id)]
        head_id = int(row[0])
        rows = cur.execute(
            "SELECT id FROM media WHERE stack_parent_id = ? AND live_companion_for_id IS NULL ORDER BY id = ? DESC, filename ASC;",
            (head_id, head_id),
        ).fetchall()
        return [int(r[0]) for r in rows] or [int(media_id)]

    def get_stack_parent_map(
        self, media_ids: Sequence[int] | None = None
    ) -> dict[int, int | None]:
        """Returns `{media_id: stack_parent_id}` for given IDs (or all non-companion items)."""
        conn = self.engine.catalog_conn()
        cur = conn.cursor()
        out: dict[int, int | None] = {}
        if media_ids is not None:
            ids = [int(m) for m in media_ids]
            for chunk in _iter_id_chunks(ids, 900):
                ph = ",".join("?" for _ in chunk)
                rows = cur.execute(
                    f"SELECT id, stack_parent_id FROM media WHERE id IN ({ph});",
                    list(chunk),
                ).fetchall()
                for mid, sp in rows:
                    out[int(mid)] = int(sp) if sp is not None else None
        else:
            rows = cur.execute(
                "SELECT id, stack_parent_id FROM media WHERE live_companion_for_id IS NULL;"
            ).fetchall()
            for mid, sp in rows:
                out[int(mid)] = int(sp) if sp is not None else None
        return out

    def restore_stack_parent_map(self, mapping: dict[int, int | None]) -> None:
        """Restores `stack_parent_id` values in a single catalog transaction (for Undo/Redo)."""
        if not mapping:
            return
        with self.engine.catalog_transaction() as tx_conn:
            tx_conn.executemany(
                "UPDATE media SET stack_parent_id = ? WHERE id = ?;",
                [(int(sp) if sp is not None else None, int(mid)) for mid, sp in mapping.items()],
            )

    def reconcile_stack_after_delete(self, deleted_id: int) -> None:
        """Promotes a clean survivor head (or unstacks a lone survivor) after deleting a stack member."""
        did = int(deleted_id)
        conn = self.engine.catalog_conn()
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT id, LOWER(COALESCE(ext, '')), COALESCE(rating, 0), COALESCE(flag, 0)
            FROM media
            WHERE stack_parent_id = ? AND id != ? AND live_companion_for_id IS NULL
            ORDER BY id ASC;
            """,
            (did, did),
        ).fetchall()
        if not rows:
            # Check if deleted_id belonged to another head and left only 1 item in that stack
            other_rows = cur.execute(
                """
                SELECT stack_parent_id FROM media
                WHERE stack_parent_id IS NOT NULL AND live_companion_for_id IS NULL
                GROUP BY stack_parent_id HAVING COUNT(*) = 1;
                """
            ).fetchall()
            lone_heads = [int(r[0]) for r in other_rows if r[0] is not None]
            if lone_heads:
                with self.engine.catalog_transaction() as tx_conn:
                    for lh in lone_heads:
                        tx_conn.execute(
                            "UPDATE media SET stack_parent_id = NULL WHERE stack_parent_id = ?;",
                            (lh,),
                        )
            return

        if len(rows) == 1:
            surv_id = int(rows[0][0])
            with self.engine.catalog_transaction() as tx_conn:
                tx_conn.execute(
                    "UPDATE media SET stack_parent_id = NULL WHERE id = ?;",
                    (surv_id,),
                )
            return

        raw_exts = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw",
                    "nef", "cr2", "cr3", "arw", "dng", "raf", "orf", "rw2", "pef", "srw"}
        best_row = max(
            rows,
            key=lambda r: (
                0 if str(r[1]) in raw_exts else 1,
                1 if int(r[3]) == 1 else 0,
                int(r[2]),
                -int(r[0]),
            ),
        )
        new_head = int(best_row[0])
        with self.engine.catalog_transaction() as tx_conn:
            tx_conn.executemany(
                "UPDATE media SET stack_parent_id = ? WHERE id = ?;",
                [(new_head, int(r[0])) for r in rows],
            )

    def set_stack_head(self, media_id: int) -> int | None:
        """Promote `media_id` to be the cover (`stack_parent_id`) of its stack."""
        members = self.get_stack_member_ids(int(media_id))
        if len(members) < 2 or int(media_id) not in members:
            return None
        new_head = int(media_id)
        with self.engine.catalog_transaction() as tx_conn:
            tx_conn.executemany(
                "UPDATE media SET stack_parent_id = ? WHERE id = ?;",
                [(new_head, int(mid)) for mid in members],
            )
        return new_head

    def auto_stack_raw_jpeg_pairs(
        self,
        folder_id: int | None = None,
        *,
        folder_ids: Sequence[int] | None = None,
        skip_existing_stacks: bool = True,
        capture_undo_out: dict[int, int | None] | None = None,
    ) -> int:
        """Pairs RAW + JPEG files with matching `(folder_id, stem)` into stacks with JPEG as head."""
        raw_exts = {"nef", "cr2", "cr3", "arw", "dng", "raf", "orf", "rw2", "pef", "srw"}
        jpg_exts = {"jpg", "jpeg"}
        conn = self.engine.catalog_conn()
        cur = conn.cursor()
        where = "WHERE is_video = 0 AND live_companion_for_id IS NULL"
        params: list[Any] = []
        if folder_ids:
            fids = [int(f) for f in dict.fromkeys(folder_ids)]
            ph = ",".join("?" for _ in fids)
            where += f" AND folder_id IN ({ph})"
            params.extend(fids)
        elif folder_id is not None:
            where += " AND folder_id = ?"
            params.append(int(folder_id))
        rows = cur.execute(
            f"SELECT id, folder_id, filename, LOWER(COALESCE(ext, '')), stack_parent_id FROM media {where};",
            params,
        ).fetchall()

        groups: dict[tuple[int, str], dict[str, list[tuple[int, int | None]]]] = {}
        for mid, fid, fname, ext, cur_sp in rows:
            if skip_existing_stacks and cur_sp is not None:
                continue
            clean_ext = ext.lstrip(".")
            if clean_ext not in raw_exts and clean_ext not in jpg_exts:
                continue
            stem = Path(str(fname)).stem.lower()
            bucket = groups.setdefault((int(fid), stem), {"jpg": [], "raw": []})
            sp_val = int(cur_sp) if cur_sp is not None else None
            if clean_ext in jpg_exts:
                bucket["jpg"].append((int(mid), sp_val))
            else:
                bucket["raw"].append((int(mid), sp_val))

        updates: list[tuple[int, int]] = []
        stacks_created = 0
        for bucket in groups.values():
            if bucket["jpg"] and bucket["raw"]:
                head_id = bucket["jpg"][0][0]
                members = bucket["jpg"] + bucket["raw"]
                for m_id, prev_sp in members:
                    if prev_sp != head_id:
                        if capture_undo_out is not None and m_id not in capture_undo_out:
                            capture_undo_out[m_id] = prev_sp
                        updates.append((head_id, m_id))
                stacks_created += 1

        if updates:
            with self.engine.catalog_transaction() as tx_conn:
                tx_conn.executemany(
                    "UPDATE media SET stack_parent_id = ?, stack_source = 'raw_jpeg' WHERE id = ?;", updates
                )
        return stacks_created

    def auto_stack_bursts(
        self,
        folder_id: int | None = None,
        max_gap_sec: float = 2.0,
        *,
        folder_ids: Sequence[int] | None = None,
        max_span_sec: float = 30.0,
        thumb_repo: Any | None = None,
        max_dhash_hamming: int = 16,
        skip_existing_stacks: bool = True,
        capture_undo_out: dict[int, int | None] | None = None,
    ) -> int:
        """
        Groups visually similar consecutive photos in the same folder taken within `max_gap_sec`
        seconds (capped at `max_span_sec`), picking the sharpest/highest-rated item as head.
        """
        conn = self.engine.catalog_conn()
        cur = conn.cursor()
        where = "WHERE is_video = 0 AND live_companion_for_id IS NULL AND date_taken IS NOT NULL AND date_taken > 0"
        params: list[Any] = []
        if folder_ids:
            fids = [int(f) for f in dict.fromkeys(folder_ids)]
            ph = ",".join("?" for _ in fids)
            where += f" AND folder_id IN ({ph})"
            params.extend(fids)
        elif folder_id is not None:
            where += " AND folder_id = ?"
            params.append(int(folder_id))
        rows = cur.execute(
            f"""
            SELECT id, folder_id, date_taken, COALESCE(rating, 0), COALESCE(flag, 0),
                   COALESCE(is_favourite, 0), stack_parent_id, filename
            FROM media {where}
            ORDER BY folder_id ASC, date_taken ASC, id ASC;
            """,
            params,
        ).fetchall()

        filtered_rows = [
            r for r in rows if not (skip_existing_stacks and r[6] is not None)
        ]
        sigs: dict[int, tuple[int, int, int, float, bytes]] = {}
        if thumb_repo is not None and filtered_rows and hasattr(thumb_repo, "get_signatures_batch"):
            sigs = thumb_repo.get_signatures_batch([int(r[0]) for r in filtered_rows])

        stacks_created = 0
        updates: list[tuple[int, int]] = []
        cur_group: list[tuple[int, int, int, int, int, int | None, str]] = []
        prev_fid: int | None = None
        prev_dt: int | None = None
        anchor_id: int | None = None
        anchor_dt: int | None = None

        def _flush_group(grp: list[tuple[int, int, int, int, int, int | None, str]]) -> None:
            nonlocal stacks_created
            if len(grp) < 2:
                return
            # A single RAW+JPEG pair shares one stem; leave 1-stem pairs to auto_stack_raw_jpeg_pairs
            if len({stem for *_rest, stem in grp}) < 2:
                return
            best_pos = 0
            best_score: tuple[int, int, float, int] | None = None
            for idx, (m_id, _rating, flag, is_fav, _dt, _sp, _stem) in enumerate(grp):
                sig = sigs.get(m_id)
                sharp = float(sig[3]) if sig is not None else 0.0
                cand = (1 if (flag == 1 or is_fav) else 0, _rating, sharp, -idx)
                if best_score is None or cand > best_score:
                    best_score = cand
                    best_pos = idx
            head = grp[best_pos][0]
            for m_id, _r, _f, _fav, _dt, prev_sp, _stem in grp:
                if prev_sp != head:
                    if capture_undo_out is not None and m_id not in capture_undo_out:
                        capture_undo_out[m_id] = prev_sp
                    updates.append((head, m_id))
            stacks_created += 1

        for mid, fid, dt, rating, flag, is_fav, cur_sp, fname in filtered_rows:
            imid = int(mid)
            ifid = int(fid)
            idt = int(dt)
            sp_val = int(cur_sp) if cur_sp is not None else None
            stem_l = Path(str(fname or "")).stem.lower()
            same_burst = False
            if (
                prev_fid == ifid
                and prev_dt is not None
                and anchor_dt is not None
                and (idt - prev_dt) <= max_gap_sec
                and (idt - anchor_dt) <= max_span_sec
            ):
                same_burst = True
                if anchor_id is not None and anchor_id in sigs and imid in sigs:
                    dh_a = sigs[anchor_id][2]
                    dh_b = sigs[imid][2]
                    if dh_a != 0 and dh_b != 0:
                        ham = ((dh_a ^ dh_b) & 0xFFFFFFFFFFFFFFFF).bit_count()
                        if ham > max_dhash_hamming:
                            same_burst = False
            if same_burst:
                cur_group.append((imid, int(rating), int(flag), int(is_fav), idt, sp_val, stem_l))
            else:
                _flush_group(cur_group)
                cur_group = [(imid, int(rating), int(flag), int(is_fav), idt, sp_val, stem_l)]
                anchor_id = imid
                anchor_dt = idt
            prev_fid = ifid
            prev_dt = idt
        _flush_group(cur_group)

        if updates:
            with self.engine.catalog_transaction() as tx_conn:
                tx_conn.executemany(
                    "UPDATE media SET stack_parent_id = ?, stack_source = 'burst' WHERE id = ?;", updates
                )
        return stacks_created

    def unstack_by_source(self, source: str, *, folder_ids: Sequence[int] | None = None) -> int:
        """Ungroup all photo groups created by `source` ('raw_jpeg' or 'burst'), including legacy untagged stacks."""
        src_key = str(source or "").strip().lower()
        if src_key not in ("raw_jpeg", "burst"):
            return 0
        raw_exts = {"nef", "cr2", "cr3", "arw", "dng", "raf", "orf", "rw2", "pef", "srw"}
        jpg_exts = {"jpg", "jpeg"}
        conn = self.engine.catalog_conn()
        cur = conn.cursor()
        where = "WHERE stack_parent_id IS NOT NULL AND live_companion_for_id IS NULL"
        params: list[Any] = []
        if folder_ids:
            fids = [int(f) for f in dict.fromkeys(folder_ids)]
            ph = ",".join("?" for _ in fids)
            where += f" AND folder_id IN ({ph})"
            params.extend(fids)
        rows = cur.execute(
            f"SELECT id, stack_parent_id, LOWER(COALESCE(ext, '')), COALESCE(stack_source, '') FROM media {where};",
            params,
        ).fetchall()
        by_sp: dict[int, list[tuple[int, str, str]]] = {}
        for mid, sp, ext, ssrc in rows:
            by_sp.setdefault(int(sp), []).append((int(mid), str(ext or "").lstrip("."), str(ssrc or "").lower()))

        to_clear: list[int] = []
        for sp, members in by_sp.items():
            sources = {m[2] for m in members if m[2]}
            if src_key in sources:
                to_clear.extend(m[0] for m in members)
            elif not sources:
                exts = [m[1] for m in members]
                is_rj = len(exts) == 2 and any(e in raw_exts for e in exts) and any(e in jpg_exts for e in exts)
                if (src_key == "raw_jpeg" and is_rj) or (src_key == "burst" and not is_rj):
                    to_clear.extend(m[0] for m in members)
        if not to_clear:
            return 0
        changed = 0
        with self.engine.catalog_transaction() as tx_conn:
            tx_cur = tx_conn.cursor()
            for chunk in _iter_id_chunks(to_clear, 900):
                ph = ",".join("?" for _ in chunk)
                tx_cur.execute(
                    f"UPDATE media SET stack_parent_id = NULL, stack_source = NULL WHERE id IN ({ph});",
                    list(chunk),
                )
                changed += int(tx_cur.rowcount or 0)
        return changed

    # -------------------------------------------------------------------------
    # Instant UI Metadata Writes (Marks metadata_dirty = 1)
    # -------------------------------------------------------------------------
    def get_field_map_for_ids(self, media_ids: Sequence[int], column: str) -> dict[int, Any]:
        """Fetches a specific column value for media_ids in batches of 900 without N+1 queries."""
        allowed_cols = {"rating", "flag", "is_favourite", "caption", "orientation", "date_taken", "author", "location_name"}
        if column not in allowed_cols:
            raise ValueError(f"Disallowed column for field map: {column}")
        if not media_ids:
            return {}
        conn = self.engine.catalog_conn()
        cursor = conn.cursor()
        result: dict[int, Any] = {}
        for chunk in _iter_id_chunks(media_ids, 900):
            placeholders = ",".join("?" for _ in chunk)
            cursor.execute(
                f"SELECT id, {column} FROM media WHERE id IN ({placeholders});",
                list(chunk),
            )
            for row in cursor.fetchall():
                result[int(row[0])] = row[1]
        return result

    def restore_field_map(self, field_map: dict[int, Any], column: str) -> None:
        """Restores column values in bulk by grouping IDs with matching values into batched transactions."""
        allowed_cols = {"rating", "flag", "is_favourite", "caption", "orientation", "date_taken", "author", "location_name"}
        if column not in allowed_cols:
            raise ValueError(f"Disallowed column for field map: {column}")
        if not field_map:
            return
        val_to_ids: dict[Any, list[int]] = {}
        for mid, val in field_map.items():
            val_to_ids.setdefault(val, []).append(mid)

        with self.engine.write_lock:
            with self.engine.catalog_transaction() as conn:
                cursor = conn.cursor()
                for val, mids in val_to_ids.items():
                    for chunk in _iter_id_chunks(mids, 900):
                        placeholders = ",".join("?" for _ in chunk)
                        cursor.execute(
                            f"UPDATE media SET {column} = ?, metadata_dirty = 1 WHERE id IN ({placeholders});",
                            [val, *chunk],
                        )

    def update_media_metadata(self, media_id: int, **fields: Any) -> None:
        """Updates allowlisted media metadata columns (orientation, dimensions, file stats) in catalog.db."""
        allowed_cols = {"orientation", "width", "height", "mtime", "size", "metadata_dirty"}
        updates = {k: v for k, v in fields.items() if k in allowed_cols and v is not None}
        if not updates:
            return
        assignments = ", ".join(f"{col} = ?" for col in updates)
        params: list[Any] = [*updates.values(), int(media_id)]
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute(f"UPDATE media SET {assignments} WHERE id = ?;", params)
            conn.commit()

    def set_rating(self, media_ids: Sequence[int], rating: int) -> None:
        if not media_ids:
            return
        if not (0 <= rating <= 5):
            raise ValueError("rating must be between 0 and 5")
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            for chunk in _iter_id_chunks(media_ids, 900):
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"UPDATE media SET rating = ?, metadata_dirty = 1 WHERE id IN ({placeholders});",
                    [rating, *chunk],
                )
            conn.commit()

    update_rating = set_rating

    def set_flag(self, media_ids: Sequence[int], flag: int) -> None:
        if not media_ids:
            return
        ival = int(flag or 0)
        db_flag = 1 if ival > 0 else 0
        db_rej = 1 if ival < 0 else 0
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            for chunk in _iter_id_chunks(media_ids, 900):
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"UPDATE media SET flag = ?, is_rejected = ?, metadata_dirty = 1 WHERE id IN ({placeholders});",
                    [db_flag, db_rej, *chunk],
                )
            conn.commit()

    def set_caption(self, media_ids: Sequence[int], caption: str) -> None:
        if not media_ids:
            return
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            for chunk in _iter_id_chunks(media_ids, 900):
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"UPDATE media SET caption = ?, metadata_dirty = 1 WHERE id IN ({placeholders});",
                    [caption, *chunk],
                )
            conn.commit()

    def set_author(self, media_ids: Sequence[int], author: str) -> None:
        if not media_ids:
            return
        clean_author = author.strip()
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            for chunk in _iter_id_chunks(media_ids, 900):
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"UPDATE media SET author = ?, metadata_dirty = 1 WHERE id IN ({placeholders});",
                    [clean_author, *chunk],
                )
            conn.commit()

    def set_date_taken(self, media_ids: Sequence[int], timestamp: int) -> None:
        if not media_ids:
            return
        ts_val = int(timestamp)
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            for chunk in _iter_id_chunks(media_ids, 900):
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"UPDATE media SET date_taken = ?, metadata_dirty = 1 WHERE id IN ({placeholders});",
                    [ts_val, *chunk],
                )
            conn.commit()

    def set_geotag(
        self,
        media_ids: Sequence[int],
        lat: float | None,
        lon: float | None,
        location_name: str | None = None,
    ) -> None:
        """
        Updates GPS coordinates, resolved Places hierarchy (country/state/city), and
        optional location name for the specified media items.
        Automatically updates SQLite's media_rtree spatial index via triggers.
        Marks metadata_dirty = 1 for background XMP write-back.
        """
        if not media_ids:
            return
        lat_val = float(lat) if lat is not None else None
        lon_val = float(lon) if lon is not None else None
        loc_val = str(location_name).strip() if location_name else None
        if lat_val is None and lon_val is None and not loc_val:
            country = state = city = None
        else:
            from src.core.geotag_resolver import resolve_location_hierarchy

            country, state, city, resolved_fmt = resolve_location_hierarchy(lat_val, lon_val, loc_val)
            if not loc_val and resolved_fmt:
                loc_val = resolved_fmt

        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            for chunk in _iter_id_chunks(media_ids, 900):
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"""
                    UPDATE media
                    SET lat = ?,
                        lon = ?,
                        location_name = ?,
                        location_country = ?,
                        location_state = ?,
                        location_city = ?,
                        metadata_dirty = 1
                    WHERE id IN ({placeholders});
                    """,
                    [lat_val, lon_val, loc_val, country, state, city, *chunk],
                )
            conn.commit()

    def get_geotag_map(
        self, media_ids: Sequence[int]
    ) -> dict[int, tuple[float | None, float | None, str | None]]:
        """Returns snapshot mapping {media_id: (lat, lon, location_name)} for bulk undo commands."""
        if not media_ids:
            return {}
        result: dict[int, tuple[float | None, float | None, str | None]] = {}
        conn = self.engine.catalog_conn()
        for chunk in _iter_id_chunks(media_ids, 900):
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT id, lat, lon, location_name FROM media WHERE id IN ({placeholders});",
                list(chunk),
            ).fetchall()
            for r in rows:
                result[int(r["id"])] = (
                    float(r["lat"]) if r["lat"] is not None else None,
                    float(r["lon"]) if r["lon"] is not None else None,
                    str(r["location_name"]) if r["location_name"] is not None else None,
                )
        return result

    def restore_geotag_map(
        self, geotag_map: dict[int, tuple[float | None, float | None, str | None]]
    ) -> None:
        """Restores previously snapshotted `(lat, lon, location_name)` and Place hierarchy during undo."""
        if not geotag_map:
            return
        from src.core.geotag_resolver import resolve_location_hierarchy

        coord_cache: dict[tuple[float, float], tuple[str | None, str | None, str | None, str | None]] = {}
        items: list[tuple[Any, ...]] = []
        for mid, (lat, lon, loc) in geotag_map.items():
            if lat is None and lon is None and not (loc or "").strip():
                country = state = city = None
            else:
                country, state, city, _ = resolve_location_hierarchy(lat, lon, loc, cache=coord_cache)
            items.append((lat, lon, loc, country, state, city, mid))
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.executemany(
                """
                UPDATE media
                SET lat = ?,
                    lon = ?,
                    location_name = ?,
                    location_country = ?,
                    location_state = ?,
                    location_city = ?,
                    metadata_dirty = 1
                WHERE id = ?;
                """,
                items,
            )
            conn.commit()

    def backfill_missing_places(self) -> int:
        """Populate missing `(location_country, location_state, location_city)` from lat/lon or location_name."""
        from src.core.geotag_resolver import resolve_location_hierarchy

        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            rows = conn.execute(
                """
                SELECT id, lat, lon, location_name
                FROM media
                WHERE (
                    (lat IS NOT NULL AND lon IS NOT NULL)
                    OR (location_name IS NOT NULL AND TRIM(location_name) != '')
                )
                AND COALESCE(TRIM(location_country), '') = ''
                AND COALESCE(TRIM(location_state), '') = ''
                AND COALESCE(TRIM(location_city), '') = '';
                """
            ).fetchall()
            if not rows:
                return 0
            coord_cache: dict[tuple[float, float], tuple[str | None, str | None, str | None, str | None]] = {}
            updates: list[tuple[str | None, str | None, str | None, str | None, int]] = []
            for r in rows:
                country, state, city, fmt = resolve_location_hierarchy(
                    r["lat"], r["lon"], r["location_name"], cache=coord_cache
                )
                if country or state or city or fmt:
                    updates.append((country, state, city, fmt, int(r["id"])))
            if not updates:
                return 0
            conn.executemany(
                """
                UPDATE media
                SET location_country = ?,
                    location_state = ?,
                    location_city = ?,
                    location_name = COALESCE(NULLIF(TRIM(location_name), ''), ?)
                WHERE id = ?;
                """,
                updates,
            )
            conn.commit()
            return len(updates)

    def get_gps_matching_candidates(
        self,
        *,
        selected_ids: Sequence[int] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        pad_seconds: int = 172800,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        """
        Loads (targets, donors, scope_label) for Companion Phone GPS Matching.
        Targets are untagged photos with valid date_taken; donors are geotagged photos
        with valid date_taken.
        """
        conn = self.engine.catalog_conn()
        priv_clause = self._private_folder_clause("f")
        sel = [int(i) for i in (selected_ids or [])]

        def _row_to_dict(r: Any) -> dict[str, Any]:
            return {
                "id": int(r["id"]),
                "path": str(r["path"] or ""),
                "filename": str(r["filename"] or ""),
                "date_taken": int(r["date_taken"] or 0),
                "camera_model": str(r["camera_model"] or ""),
                "lat": float(r["lat"]) if r["lat"] is not None else None,
                "lon": float(r["lon"]) if r["lon"] is not None else None,
                "location_name": str(r["location_name"] or ""),
            }

        base_cols = (
            "m.id, (f.path || '/' || m.filename) AS path, m.filename, m.date_taken, "
            "m.camera_model, m.lat, m.lon, m.location_name"
        )

        # Case 1: World Map View (viewport bbox or clicked cluster)
        if bbox is not None:
            min_lat, max_lat, min_lon, max_lon = bbox
            if sel:
                donors: list[dict[str, Any]] = []
                for chunk in _iter_id_chunks(sel, 900):
                    ph = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"""
                        SELECT {base_cols}
                        FROM media m
                        JOIN folders f ON f.id = m.folder_id
                        WHERE COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
                          AND m.date_taken IS NOT NULL AND m.date_taken > 0
                          AND m.lat IS NOT NULL AND m.lon IS NOT NULL
                          AND m.id IN ({ph})
                        ORDER BY m.date_taken ASC;
                        """,
                        list(chunk),
                    ).fetchall()
                    donors.extend(_row_to_dict(r) for r in rows)
                scope_label = f"Selected map cluster ({len(donors)} geotagged donor(s))"
            else:
                rows = conn.execute(
                    f"""
                    SELECT {base_cols}
                    FROM media m
                    JOIN folders f ON f.id = m.folder_id
                    WHERE COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
                      AND m.date_taken IS NOT NULL AND m.date_taken > 0
                      AND m.lat BETWEEN ? AND ?
                      AND m.lon BETWEEN ? AND ?
                    ORDER BY m.date_taken ASC;
                    """,
                    (float(min_lat), float(max_lat), float(min_lon), float(max_lon)),
                ).fetchall()
                donors = [_row_to_dict(r) for r in rows]
                scope_label = f"Map viewport ({len(donors)} geotagged donor(s))"

            if not donors:
                return [], [], scope_label

            min_ts = min(d["date_taken"] for d in donors) - int(pad_seconds)
            max_ts = max(d["date_taken"] for d in donors) + int(pad_seconds)
            t_rows = conn.execute(
                f"""
                SELECT {base_cols}
                FROM media m
                JOIN folders f ON f.id = m.folder_id
                WHERE COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
                  AND m.date_taken BETWEEN ? AND ?
                  AND (m.lat IS NULL OR m.lon IS NULL)
                ORDER BY m.date_taken ASC
                LIMIT 2500;
                """,
                (min_ts, max_ts),
            ).fetchall()
            targets = [_row_to_dict(r) for r in t_rows]
            return targets, donors, scope_label

        # Case 2: Gallery / Grid selection
        if sel:
            sel_rows: list[dict[str, Any]] = []
            for chunk in _iter_id_chunks(sel, 900):
                ph = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""
                    SELECT {base_cols}
                    FROM media m
                    JOIN folders f ON f.id = m.folder_id
                    WHERE COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
                      AND m.date_taken IS NOT NULL AND m.date_taken > 0
                      AND m.id IN ({ph})
                    ORDER BY m.date_taken ASC;
                    """,
                    list(chunk),
                ).fetchall()
                sel_rows.extend(_row_to_dict(r) for r in rows)

            sel_donors = [r for r in sel_rows if r["lat"] is not None and r["lon"] is not None]
            sel_targets = [r for r in sel_rows if r["lat"] is None or r["lon"] is None]

            if len(sel) >= 2 and sel_donors and sel_targets:
                return sel_targets, sel_donors, f"Selection ({len(sel_targets)} untagged, {len(sel_donors)} donors)"

            if sel_targets:
                min_ts = min(t["date_taken"] for t in sel_targets) - int(pad_seconds)
                max_ts = max(t["date_taken"] for t in sel_targets) + int(pad_seconds)
                d_rows = conn.execute(
                    f"""
                    SELECT {base_cols}
                    FROM media m
                    JOIN folders f ON f.id = m.folder_id
                    WHERE COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
                      AND m.date_taken BETWEEN ? AND ?
                      AND m.lat IS NOT NULL AND m.lon IS NOT NULL
                    ORDER BY m.date_taken ASC;
                    """,
                    (min_ts, max_ts),
                ).fetchall()
                donors = [_row_to_dict(r) for r in d_rows]
                return sel_targets, donors, f"Selected {len(sel_targets)} untagged photo(s) + catalog donors"

            if sel_donors:
                min_ts = min(d["date_taken"] for d in sel_donors) - int(pad_seconds)
                max_ts = max(d["date_taken"] for d in sel_donors) + int(pad_seconds)
                t_rows = conn.execute(
                    f"""
                    SELECT {base_cols}
                    FROM media m
                    JOIN folders f ON f.id = m.folder_id
                    WHERE COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
                      AND m.date_taken BETWEEN ? AND ?
                      AND (m.lat IS NULL OR m.lon IS NULL)
                    ORDER BY m.date_taken ASC
                    LIMIT 2500;
                    """,
                    (min_ts, max_ts),
                ).fetchall()
                targets = [_row_to_dict(r) for r in t_rows]
                return targets, sel_donors, f"Selected {len(sel_donors)} donor(s) → untagged catalog photos"

        # Case 3: Catalog-wide fallback
        t_rows = conn.execute(
            f"""
            SELECT {base_cols}
            FROM media m
            JOIN folders f ON f.id = m.folder_id
            WHERE COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
              AND m.date_taken IS NOT NULL AND m.date_taken > 0
              AND (m.lat IS NULL OR m.lon IS NULL)
            ORDER BY m.date_taken DESC
            LIMIT 2500;
            """
        ).fetchall()
        d_rows = conn.execute(
            f"""
            SELECT {base_cols}
            FROM media m
            JOIN folders f ON f.id = m.folder_id
            WHERE COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
              AND m.date_taken IS NOT NULL AND m.date_taken > 0
              AND m.lat IS NOT NULL AND m.lon IS NOT NULL
            ORDER BY m.date_taken ASC;
            """
        ).fetchall()
        targets = sorted((_row_to_dict(r) for r in t_rows), key=lambda r: r["date_taken"])
        donors = [_row_to_dict(r) for r in d_rows]
        return targets, donors, f"Catalog ({len(targets)} untagged, {len(donors)} donors)"

    def get_geotag_bounds(self) -> tuple[float, float, float, float] | None:
        """
        Returns bounding box (min_lat, max_lat, min_lon, max_lon) across all geotagged media,
        or None if no photos have geotags.
        """
        conn = self.engine.catalog_conn()
        row = conn.execute(
            """
            SELECT MIN(lat) AS min_lat, MAX(lat) AS max_lat,
                   MIN(lon) AS min_lon, MAX(lon) AS max_lon
            FROM media
            WHERE lat IS NOT NULL AND lon IS NOT NULL;
            """
        ).fetchone()
        if (
            row
            and row["min_lat"] is not None
            and row["max_lat"] is not None
            and row["min_lon"] is not None
            and row["max_lon"] is not None
        ):
            return (
                float(row["min_lat"]),
                float(row["max_lat"]),
                float(row["min_lon"]),
                float(row["max_lon"]),
            )
        return None

    def rename_media(self, media_id: int, new_filename: str) -> None:
        """Renames a media file record in catalog.db."""
        ext = Path(new_filename).suffix.lstrip(".").lower()
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute(
                "UPDATE media SET filename = ?, ext = ?, metadata_dirty = 1 WHERE id = ?;",
                (new_filename, ext, media_id),
            )
            conn.commit()

    def move_media_folder(
        self, media_id: int, new_folder_id: int, new_filename: str | None = None
    ) -> None:
        """Moves a media record to a different folder in catalog.db (and optionally updates filename)."""
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            if new_filename:
                ext = Path(new_filename).suffix.lstrip(".").lower()
                conn.execute(
                    "UPDATE media SET folder_id = ?, filename = ?, ext = ?, metadata_dirty = 1 WHERE id = ?;",
                    (new_folder_id, new_filename, ext, media_id),
                )
            else:
                conn.execute(
                    "UPDATE media SET folder_id = ?, metadata_dirty = 1 WHERE id = ?;",
                    (new_folder_id, media_id),
                )
            conn.commit()

    def move_media_folder_batch(
        self,
        moves: Sequence[tuple[int, int, str | None, str | None]],
    ) -> None:
        """
        Batch-moves multiple media records to new folders in a single SQLite transaction.
        Each tuple is (media_id, new_folder_id, new_filename, new_backup_path).
        """
        if not moves:
            return
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            for media_id, new_folder_id, new_filename, new_backup_path in moves:
                if new_filename and new_backup_path is not None:
                    ext = Path(new_filename).suffix.lstrip(".").lower()
                    conn.execute(
                        "UPDATE media SET folder_id = ?, filename = ?, ext = ?, original_backup_path = ?, metadata_dirty = 1 WHERE id = ?;",
                        (new_folder_id, new_filename, ext, new_backup_path, media_id),
                    )
                elif new_filename:
                    ext = Path(new_filename).suffix.lstrip(".").lower()
                    conn.execute(
                        "UPDATE media SET folder_id = ?, filename = ?, ext = ?, metadata_dirty = 1 WHERE id = ?;",
                        (new_folder_id, new_filename, ext, media_id),
                    )
                elif new_backup_path is not None:
                    conn.execute(
                        "UPDATE media SET folder_id = ?, original_backup_path = ?, metadata_dirty = 1 WHERE id = ?;",
                        (new_folder_id, new_backup_path, media_id),
                    )
                else:
                    conn.execute(
                        "UPDATE media SET folder_id = ?, metadata_dirty = 1 WHERE id = ?;",
                        (new_folder_id, media_id),
                    )
            conn.commit()


    def rotate_lossless(
        self,
        media_ids: Sequence[int],
        clockwise: bool = True,
        delta_degrees: int | None = None,
    ) -> dict[int, int]:
        """
        Updates EXIF orientation losslessly in catalog.db, swaps width/height (for 90°),
        and transforms all face bounding boxes in media_people.
        Supports clockwise boolean, or explicit delta_degrees: 90, -90, 180, 270, -270.
        """
        if not media_ids:
            return {}

        if delta_degrees is not None:
            if delta_degrees in (180, -180):
                is_180 = True
                is_cw = False
                rot_map = _ROTATE_180_MAP
            elif delta_degrees in (90, -270):
                is_180 = False
                is_cw = True
                rot_map = _ROTATE_CW_MAP
            elif delta_degrees in (-90, 270):
                is_180 = False
                is_cw = False
                rot_map = _ROTATE_CCW_MAP
            else:
                is_180 = False
                is_cw = clockwise
                rot_map = _ROTATE_CW_MAP if clockwise else _ROTATE_CCW_MAP
        else:
            is_180 = False
            is_cw = clockwise
            rot_map = _ROTATE_CW_MAP if clockwise else _ROTATE_CCW_MAP

        updated_orientations: dict[int, int] = {}

        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            for chunk in _iter_id_chunks(media_ids, 900):
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT id, orientation, width, height, is_video FROM media WHERE id IN ({placeholders});",
                    list(chunk),
                ).fetchall()
                for r in rows:
                    if r["is_video"]:
                        continue
                    mid = int(r["id"])
                    old_orient = int(r["orientation"] or 1)
                    fallback_default = 3 if is_180 else (6 if is_cw else 8)
                    new_orient = rot_map.get(old_orient, fallback_default)
                    updated_orientations[mid] = new_orient

                    if is_180:
                        # 180 deg does not swap width and height
                        conn.execute(
                            """
                            UPDATE media
                            SET orientation = ?,
                                metadata_dirty = 1
                            WHERE id = ?;
                            """,
                            (new_orient, mid),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE media
                            SET orientation = ?,
                                width = height,
                                height = width,
                                metadata_dirty = 1
                            WHERE id = ?;
                            """,
                            (new_orient, mid),
                        )

                    # Transform all face bounding boxes for this media_id
                    face_rows = conn.execute(
                        "SELECT id, rect_x, rect_y, rect_w, rect_h FROM media_people WHERE media_id = ?;",
                        (mid,),
                    ).fetchall()
                    for f in face_rows:
                        fid = int(f["id"])
                        rx = float(f["rect_x"])
                        ry = float(f["rect_y"])
                        rw = float(f["rect_w"])
                        rh = float(f["rect_h"])

                        if is_180:
                            nx = 1.0 - (rx + rw)
                            ny = 1.0 - (ry + rh)
                            nw = rw
                            nh = rh
                        elif is_cw:
                            nx = 1.0 - (ry + rh)
                            ny = rx
                            nw = rh
                            nh = rw
                        else:  # CCW
                            nx = ry
                            ny = 1.0 - (rx + rw)
                            nw = rh
                            nh = rw

                        nx = max(0.0, min(1.0, nx))
                        ny = max(0.0, min(1.0, ny))
                        nw = max(0.0, min(1.0 - nx, nw))
                        nh = max(0.0, min(1.0 - ny, nh))

                        conn.execute(
                            """
                            UPDATE media_people
                            SET rect_x = ?, rect_y = ?, rect_w = ?, rect_h = ?
                            WHERE id = ?;
                            """,
                            (nx, ny, nw, nh, fid),
                        )
            conn.commit()
        return updated_orientations

    def transform_face_regions_for_crop(
        self,
        media_id: int,
        norm_crop_x: float,
        norm_crop_y: float,
        norm_crop_w: float,
        norm_crop_h: float,
    ) -> int:
        """
        Remaps each face bounding box on media_id into the normalized crop rectangle:
        rx = (rect_x - norm_crop_x) / norm_crop_w
        ry = (rect_y - norm_crop_y) / norm_crop_h
        rw = rect_w / norm_crop_w
        rh = rect_h / norm_crop_h
        If a face's visible area in the cropped space is < 25% of its pre-crop area,
        it is deleted (cropped out); otherwise its bounding box is clamped to [0, 1]
        and updated, and media.metadata_dirty is set to 1.
        Returns the number of retained face regions.
        """
        if norm_crop_w <= 0.0 or norm_crop_h <= 0.0:
            return 0

        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            faces = conn.execute(
                """
                SELECT id, person_id, state, rect_x, rect_y, rect_w, rect_h
                FROM media_people
                WHERE media_id = ?;
                """,
                (media_id,),
            ).fetchall()

            if not faces:
                return 0

            retained_count = 0
            for f in faces:
                fid = int(f["id"])
                pid = f["person_id"]
                state = int(f["state"])
                rx = float(f["rect_x"])
                ry = float(f["rect_y"])
                rw = float(f["rect_w"])
                rh = float(f["rect_h"])

                # Remap into normalized crop box
                new_rx = (rx - norm_crop_x) / norm_crop_w
                new_ry = (ry - norm_crop_y) / norm_crop_h
                new_rw = rw / norm_crop_w
                new_rh = rh / norm_crop_h
                pre_area = new_rw * new_rh

                # Clamp to [0.0, 1.0]
                x1 = max(0.0, min(1.0, new_rx))
                y1 = max(0.0, min(1.0, new_ry))
                x2 = max(0.0, min(1.0, new_rx + new_rw))
                y2 = max(0.0, min(1.0, new_ry + new_rh))
                clamped_w = max(0.0, x2 - x1)
                clamped_h = max(0.0, y2 - y1)
                clamped_area = clamped_w * clamped_h

                if pre_area <= 0.0 or clamped_area < (pre_area * 0.25):
                    # Face cropped out: delete row and decrement confirmed_count if confirmed
                    conn.execute("DELETE FROM media_people WHERE id = ?;", (fid,))
                    if state == 1 and pid is not None:
                        conn.execute(
                            "UPDATE people SET confirmed_count = MAX(0, confirmed_count - 1) WHERE id = ?;",
                            (pid,),
                        )
                else:
                    # Update with clamped coordinates
                    conn.execute(
                        """
                        UPDATE media_people
                        SET rect_x = ?, rect_y = ?, rect_w = ?, rect_h = ?
                        WHERE id = ?;
                        """,
                        (x1, y1, clamped_w, clamped_h, fid),
                    )
                    retained_count += 1

            conn.execute(
                "UPDATE media SET metadata_dirty = 1 WHERE id = ?;", (media_id,)
            )
            conn.commit()
            return retained_count

    def restore_original_faces(
        self, media_id: int, original_people: list[dict[str, Any]]
    ) -> None:
        """
        Restores original face regions (e.g. from restored original file on revert).
        Clears existing media_people rows (adjusting confirmed_counts) and re-assigns
        the original people.
        """
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            existing = conn.execute(
                "SELECT person_id, state FROM media_people WHERE media_id = ?;",
                (media_id,),
            ).fetchall()
            for row in existing:
                if int(row["state"]) == 1 and row["person_id"] is not None:
                    conn.execute(
                        "UPDATE people SET confirmed_count = MAX(0, confirmed_count - 1) WHERE id = ?;",
                        (int(row["person_id"]),),
                    )
            conn.execute("DELETE FROM media_people WHERE media_id = ?;", (media_id,))
            conn.commit()

        for p in original_people:
            name = p.get("name")
            if not name:
                continue
            rect = p.get("rect", (0.0, 0.0, 0.0, 0.0))
            self.assign_person(media_id, name, rect=rect)

    # -------------------------------------------------------------------------
    # Relative "Change Time Taken" Logic (SPEC Section 4)
    # -------------------------------------------------------------------------
    def apply_relative_time_shift(
        self, media_ids: Sequence[int], lead_id: int, new_lead_epoch: int
    ) -> int:
        """
        Computes Δt = T_lead_new - T_lead_original and applies it relatively to all selected IDs.
        Returns Δt in seconds.
        """
        if not media_ids:
            return 0
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            row = conn.execute(
                "SELECT date_taken, mtime FROM media WHERE id = ?;", (lead_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Lead media_id {lead_id} not found in catalog.db")
            lead_original = int(row["date_taken"] if row["date_taken"] is not None else row["mtime"])
            delta_seconds = int(new_lead_epoch) - lead_original

            for chunk in _iter_id_chunks(media_ids, 900):
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"""
                    UPDATE media
                    SET date_taken = COALESCE(date_taken, mtime) + ?,
                        metadata_dirty = 1
                    WHERE id IN ({placeholders});
                    """,
                    [delta_seconds, *chunk],
                )
            conn.commit()
            return delta_seconds

    # -------------------------------------------------------------------------
    # Hierarchical Descriptive Tags & Closure Table
    # -------------------------------------------------------------------------
    def create_tag(self, full_path: str) -> int:
        """
        Creates a hierarchical tag (e.g. 'PeopleTags/Family/Craig'), creating intermediate
        parent tags automatically. Relies on trg_tags_after_insert to maintain tag_closure.
        """
        parts = [p.strip() for p in full_path.replace("|", "/").strip("/").split("/") if p.strip()]
        if not parts:
            raise ValueError("Tag full_path cannot be empty")

        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            parent_id: int | None = None
            current_path = ""
            for part in parts:
                current_path = f"{current_path}/{part}" if current_path else part
                existing = conn.execute(
                    "SELECT id, full_path FROM tags WHERE LOWER(full_path) = LOWER(?);",
                    (current_path,),
                ).fetchone()
                if existing is not None:
                    parent_id = int(existing["id"])
                    current_path = str(existing["full_path"])
                else:
                    cur = conn.execute(
                        "INSERT INTO tags (parent_id, name, full_path) VALUES (?, ?, ?) RETURNING id;",
                        (parent_id, part, current_path),
                    )
                    parent_id = int(cur.fetchone()[0])
            conn.commit()
            assert parent_id is not None
            return parent_id

    def move_tag(self, tag_id: int, new_parent_id: int | None) -> None:
        """Moves a tag to a new parent in the hierarchy, triggering trg_tags_after_update_parent."""
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            tag_row = conn.execute(
                "SELECT name, full_path FROM tags WHERE id = ?;", (tag_id,)
            ).fetchone()
            if tag_row is None:
                raise KeyError(f"Tag {tag_id} not found")

            if new_parent_id is not None:
                parent_row = conn.execute(
                    "SELECT full_path FROM tags WHERE id = ?;", (new_parent_id,)
                ).fetchone()
                if parent_row is None:
                    raise KeyError(f"Parent tag {new_parent_id} not found")
                # Cycle check: new_parent cannot be a descendant of tag_id
                is_cycle = conn.execute(
                    "SELECT 1 FROM tag_closure WHERE ancestor_id = ? AND descendant_id = ?;",
                    (tag_id, new_parent_id),
                ).fetchone()
                if is_cycle is not None:
                    raise ValueError(f"Cannot move tag {tag_id} into its own descendant {new_parent_id}")
                new_full_path = f"{parent_row['full_path']}/{tag_row['name']}"
            else:
                new_full_path = str(tag_row["name"])

            old_prefix = str(tag_row["full_path"])
            conn.execute(
                "UPDATE tags SET parent_id = ?, full_path = ? WHERE id = ?;",
                (new_parent_id, new_full_path, tag_id),
            )
            # Update full_path of any descendants
            descendants = conn.execute(
                """
                SELECT t.id, t.full_path
                FROM tags t
                JOIN tag_closure tc ON t.id = tc.descendant_id
                WHERE tc.ancestor_id = ? AND tc.depth > 0;
                """,
                (tag_id,),
            ).fetchall()
            for d in descendants:
                rel_suffix = str(d["full_path"])[len(old_prefix) :]
                conn.execute(
                    "UPDATE tags SET full_path = ? WHERE id = ?;",
                    (f"{new_full_path}{rel_suffix}", int(d["id"])),
                )
            conn.commit()

    def merge_tags(self, source_tag_id: int, target_tag_id: int) -> None:
        """
        Merges source_tag_id into target_tag_id:
        1. Copies all media associations from source_tag_id to target_tag_id.
        2. Recursively re-parents any child tags of source_tag_id under target_tag_id (merging duplicates).
        3. Deletes source_tag_id.
        """
        if source_tag_id == target_tag_id:
            return

        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            # 1. Transfer media associations
            conn.execute(
                """
                INSERT OR IGNORE INTO media_tags (media_id, tag_id)
                SELECT media_id, ? FROM media_tags WHERE tag_id = ?;
                """,
                (target_tag_id, source_tag_id),
            )
            conn.execute("DELETE FROM media_tags WHERE tag_id = ?;", (source_tag_id,))

            # 2. Transfer child tags
            children = conn.execute(
                "SELECT id, name FROM tags WHERE parent_id = ?;", (source_tag_id,)
            ).fetchall()
            for child in children:
                cid = int(child["id"])
                cname = str(child["name"])
                existing_c = conn.execute(
                    "SELECT id FROM tags WHERE parent_id = ? AND name = ? COLLATE NOCASE;",
                    (target_tag_id, cname),
                ).fetchone()
                if existing_c is not None:
                    self.merge_tags(cid, int(existing_c["id"]))
                else:
                    self.move_tag(cid, target_tag_id)

            # 3. Delete source tag
            conn.execute("DELETE FROM tags WHERE id = ?;", (source_tag_id,))
            conn.commit()

    def find_child_tag_by_name(self, parent_id: int | None, name: str) -> dict[str, Any] | None:
        """Finds a direct child tag with the specified name under parent_id (or root if None)."""
        conn = self.engine.catalog_conn()
        if parent_id is not None:
            row = conn.execute(
                "SELECT id, parent_id, name, full_path FROM tags WHERE parent_id = ? AND name = ? COLLATE NOCASE;",
                (parent_id, name),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id, parent_id, name, full_path FROM tags WHERE parent_id IS NULL AND name = ? COLLATE NOCASE;",
                (name,),
            ).fetchone()
        return dict(row) if row else None

    def find_tag_by_leaf_name(self, leaf_name: str) -> dict[str, Any] | None:
        """Finds an existing tag with the specified leaf name, preferring hierarchical tags and most-used tags."""
        conn = self.engine.catalog_conn()
        row = conn.execute(
            """
            SELECT t.id, t.parent_id, t.name, t.full_path, COUNT(mt.media_id) AS cnt
            FROM tags t
            LEFT JOIN media_tags mt ON mt.tag_id = t.id
            WHERE LOWER(t.name) = LOWER(?)
            GROUP BY t.id
            ORDER BY (t.parent_id IS NOT NULL) DESC, cnt DESC, LENGTH(t.full_path) DESC
            LIMIT 1;
            """,
            (leaf_name.strip(),),
        ).fetchone()
        return dict(row) if row else None

    def assign_tag(
        self,
        media_ids: Sequence[int],
        tag_id: int,
        *,
        mark_dirty: bool = True,
    ) -> None:
        if not media_ids:
            return
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            clean_ids: list[int] = []
            if not mark_dirty:
                for chunk in _iter_id_chunks(media_ids, 900):
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"SELECT id FROM media WHERE id IN ({placeholders}) AND metadata_dirty = 0;",
                        list(chunk),
                    ).fetchall()
                    clean_ids.extend(int(r[0]) for r in rows)
            conn.executemany(
                "INSERT OR IGNORE INTO media_tags (media_id, tag_id) VALUES (?, ?);",
                [(mid, tag_id) for mid in media_ids],
            )
            if mark_dirty:
                for chunk in _iter_id_chunks(media_ids, 900):
                    placeholders = ",".join("?" for _ in chunk)
                    conn.execute(
                        f"UPDATE media SET metadata_dirty = 1 WHERE id IN ({placeholders});",
                        list(chunk),
                    )
            elif clean_ids:
                # Reset any AFTER INSERT trigger dirtiness atomically before commit
                for chunk in _iter_id_chunks(clean_ids, 900):
                    placeholders = ",".join("?" for _ in chunk)
                    conn.execute(
                        f"UPDATE media SET metadata_dirty = 0 WHERE id IN ({placeholders});",
                        list(chunk),
                    )
            conn.commit()

    def remove_tag(self, media_ids: Sequence[int], tag_id: int) -> None:
        if not media_ids:
            return
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            for chunk in _iter_id_chunks(media_ids, 900):
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"DELETE FROM media_tags WHERE tag_id = ? AND media_id IN ({placeholders});",
                    [tag_id, *chunk],
                )
                conn.execute(
                    f"UPDATE media SET metadata_dirty = 1 WHERE id IN ({placeholders});",
                    list(chunk),
                )
            conn.commit()

    def delete_tag(self, tag_id: int) -> None:
        """Deletes a tag and all its descendant tags, cascading to media_tags and tag_closure."""
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute("DELETE FROM tags WHERE id = ?;", (tag_id,))
            conn.commit()

    def delete_unused_tags(self) -> int:
        """Deletes all tags that have 0 assigned media across themselves and all descendants."""
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            used_ids = [
                row[0]
                for row in conn.execute(
                    """
                    SELECT DISTINCT tc.ancestor_id
                    FROM tag_closure tc
                    JOIN media_tags mt ON mt.tag_id = tc.descendant_id;
                    """
                ).fetchall()
            ]
            if used_ids:
                placeholders = ",".join("?" for _ in used_ids)
                cur = conn.execute(
                    f"DELETE FROM tags WHERE id NOT IN ({placeholders});",
                    used_ids,
                )
            else:
                cur = conn.execute("DELETE FROM tags;")
            count = cur.rowcount
            conn.commit()
            return count

    def deduplicate_media_tags(self) -> int:
        """
        Cleans up redundant root tag assignments where a media item is tagged with
        both a hierarchical tag (e.g. 'Subject/Occasions/Christmas') and a redundant
        flat root tag sharing the same leaf name (e.g. 'Christmas' from dc:subject).
        Returns the number of redundant media_tag associations removed.
        """
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            rows = conn.execute(
                """
                SELECT mt_root.media_id, mt_root.tag_id AS root_tag_id
                FROM media_tags mt_root
                JOIN tags t_root ON mt_root.tag_id = t_root.id
                JOIN media_tags mt_hier ON mt_root.media_id = mt_hier.media_id AND mt_root.tag_id != mt_hier.tag_id
                JOIN tags t_hier ON mt_hier.tag_id = t_hier.id
                WHERE t_root.parent_id IS NULL
                  AND t_hier.parent_id IS NOT NULL
                  AND LOWER(t_root.name) = LOWER(t_hier.name);
                """
            ).fetchall()
            if not rows:
                return 0

            conn.executemany(
                "DELETE FROM media_tags WHERE media_id = ? AND tag_id = ?;",
                [(r["media_id"], r["root_tag_id"]) for r in rows],
            )
            conn.execute(
                """
                DELETE FROM tags
                WHERE parent_id IS NULL
                  AND id NOT IN (SELECT DISTINCT tag_id FROM media_tags)
                  AND id NOT IN (SELECT DISTINCT parent_id FROM tags WHERE parent_id IS NOT NULL);
                """
            )
            conn.commit()
            return len(rows)

    def get_tag_tree(self) -> list[dict[str, Any]]:
        conn = self.engine.catalog_conn()
        priv_clause = self._private_folder_clause("f")
        rows = conn.execute(
            f"""
            WITH tag_counts AS (
                SELECT tc.ancestor_id AS tag_id, COUNT(DISTINCT mt.media_id) AS cnt
                FROM tag_closure tc
                JOIN media_tags mt ON mt.tag_id = tc.descendant_id
                JOIN media m ON m.id = mt.media_id
                JOIN folders f ON f.id = m.folder_id
                WHERE COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
                GROUP BY tc.ancestor_id
            )
            SELECT
                t.id,
                t.parent_id,
                t.name,
                t.full_path,
                COALESCE(tc.cnt, 0) AS media_count,
                COALESCE(tc.cnt, 0) AS direct_count
            FROM tags t
            LEFT JOIN tag_counts tc ON tc.tag_id = t.id
            ORDER BY t.full_path COLLATE NOCASE ASC;
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def get_tag_full_path(self, tag_id: int) -> str | None:
        """Returns the canonical full_path for tag_id, or None if not found."""
        conn = self.engine.catalog_conn()
        row = conn.execute("SELECT full_path FROM tags WHERE id = ?;", (int(tag_id),)).fetchone()
        return str(row["full_path"]) if row else None

    def get_recent_tags(self, limit: int = 10) -> list[str]:
        """Returns the most recently created / recently modified tag full_paths as initial suggestions."""
        conn = self.engine.catalog_conn()
        rows = conn.execute(
            """
            SELECT t.full_path
            FROM tags t
            JOIN media_tags mt ON mt.tag_id = t.id
            JOIN media m ON m.id = mt.media_id
            GROUP BY t.id
            ORDER BY MAX(m.metadata_dirty) DESC, t.id DESC
            LIMIT ?;
            """,
            (limit,),
        ).fetchall()
        if not rows:
            rows = conn.execute(
                "SELECT full_path FROM tags ORDER BY id DESC LIMIT ?;",
                (limit,),
            ).fetchall()
        return [str(r["full_path"]) for r in rows]

    # -------------------------------------------------------------------------
    # People & Static-Image Face Regions
    # -------------------------------------------------------------------------
    @staticmethod
    def extract_face_crop_blob(
        file_path: str | Path,
        rect: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        target_size: int = 128,
        orientation: int = 1,
    ) -> bytes | None:
        """
        Extracts a square face avatar JPEG blob from `file_path` using the normalized
        rectangle `(rx, ry, rw, rh)` on the upright image oriented according to `orientation`.
        """
        try:
            p = Path(file_path)
            if not p.exists() or not p.is_file():
                return None
            with Image.open(p) as raw_img:
                disk_orient = 1
                try:
                    disk_exif = raw_img.getexif()
                    disk_orient = int(disk_exif.get(0x0112) or 1)
                except Exception:
                    pass

                img = ImageOps.exif_transpose(raw_img)
                target_orient = int(orientation or 1)
                if target_orient != 1 and target_orient != disk_orient:
                    orient_to_angle = {1: 0, 6: 90, 3: 180, 8: 270}
                    delta_angle = (
                        orient_to_angle.get(target_orient, 0)
                        - orient_to_angle.get(disk_orient, 0)
                    ) % 360
                    if delta_angle == 90:
                        img = img.rotate(-90, expand=True)
                    elif delta_angle == 180:
                        img = img.rotate(180, expand=True)
                    elif delta_angle == 270:
                        img = img.rotate(90, expand=True)

                if img.mode != "RGB":
                    img = img.convert("RGB")
                w, h = img.size
                if w <= 0 or h <= 0:
                    return None
                rx, ry, rw, rh = rect
                if rw > 0.005 and rh > 0.005:
                    cx = (rx + rw * 0.5) * w
                    cy = (ry + rh * 0.5) * h
                    side = max(rw * w, rh * h) * 1.38
                    side = max(24.0, min(side, float(min(w, h))))
                    x0 = max(0, int(round(cx - side * 0.5)))
                    y0 = max(0, int(round(cy - side * 0.5)))
                    x1 = min(w, int(round(x0 + side)))
                    y1 = min(h, int(round(y0 + side)))
                    if x1 - x0 < 8 or y1 - y0 < 8:
                        side_i = min(w, h)
                        x0 = (w - side_i) // 2
                        y0 = (h - side_i) // 2
                        x1 = x0 + side_i
                        y1 = y0 + side_i
                else:
                    side_i = min(w, h)
                    x0 = (w - side_i) // 2
                    y0 = (h - side_i) // 2
                    x1 = x0 + side_i
                    y1 = y0 + side_i

                cropped = img.crop((x0, y0, x1, y1)).resize(
                    (target_size, target_size), Image.Resampling.LANCZOS
                )
                buf = io.BytesIO()
                cropped.save(buf, format="JPEG", quality=90)
                return buf.getvalue()
        except Exception:
            return None

    def _put_face_thumbnail_direct(self, face_id: int, blob: bytes) -> None:
        if not blob or face_id <= 0:
            return
        try:
            with self.engine.write_lock:
                tconn = self.engine.thumbnails_conn()
                tconn.execute(
                    """
                    INSERT INTO face_thumbnails (face_id, crop_webp)
                    VALUES (?, ?)
                    ON CONFLICT(face_id) DO UPDATE SET crop_webp = excluded.crop_webp;
                    """,
                    (int(face_id), blob),
                )
                tconn.commit()
        except Exception:
            pass

    def _invalidate_face_thumbnails_direct(self, face_ids: Sequence[int]) -> None:
        clean_ids = [int(fid) for fid in face_ids if fid and int(fid) > 0]
        if not clean_ids:
            return
        try:
            with self.engine.write_lock:
                tconn = self.engine.thumbnails_conn()
                for chunk in _iter_id_chunks(clean_ids, 900):
                    placeholders = ",".join("?" for _ in chunk)
                    tconn.execute(
                        f"DELETE FROM face_thumbnails WHERE face_id IN ({placeholders});",
                        list(chunk),
                    )
                tconn.commit()
        except Exception:
            pass

    def get_face_thumbnail(self, media_people_id: int) -> bytes | None:
        """Retrieves cached face thumbnail crop from thumbnails database if available."""
        try:
            conn = self.engine.thumbnails_conn()
            row = conn.execute(
                "SELECT crop_webp FROM face_thumbnails WHERE face_id = ?;",
                (int(media_people_id),),
            ).fetchone()
            return bytes(row["crop_webp"]) if row and row["crop_webp"] else None
        except Exception:
            return None

    def refresh_person_avatar(
        self,
        person_id: int,
        invalidated_face_ids: Sequence[int] = (),
        updated_face_id: int | None = None,
    ) -> None:
        """
        Ensures `people.thumbnail_blob` and `people.avatar_media_people_id` stay fresh when a
        face box is moved, retagged, unassigned, or deleted.
        """
        if not person_id or int(person_id) <= 0:
            return
        pid = int(person_id)
        inv_set = {int(f) for f in invalidated_face_ids if f and int(f) > 0}
        conn_read = self.engine.catalog_conn()
        p_row = conn_read.execute(
            "SELECT thumbnail_blob, avatar_media_people_id FROM people WHERE id = ?;",
            (pid,),
        ).fetchone()
        if p_row is None:
            return
        cur_blob = p_row["thumbnail_blob"]
        cur_avatar_fid = int(p_row["avatar_media_people_id"]) if p_row["avatar_media_people_id"] is not None else None

        needs_refresh = False
        target_face_id: int | None = None
        if updated_face_id is not None and int(updated_face_id) > 0:
            ufid = int(updated_face_id)
            if cur_blob is None or cur_avatar_fid is None or cur_avatar_fid == ufid:
                needs_refresh = True
                target_face_id = ufid
        elif inv_set:
            if cur_blob is None or cur_avatar_fid is None or cur_avatar_fid in inv_set:
                needs_refresh = True
        elif cur_blob is None:
            needs_refresh = True

        if not needs_refresh:
            return

        cand_row = None
        if target_face_id is not None:
            cand_row = conn_read.execute(
                """
                SELECT mp.id AS face_id, mp.rect_x, mp.rect_y, mp.rect_w, mp.rect_h,
                       m.filename, m.orientation, f.path AS folder_path
                FROM media_people mp
                JOIN media m ON m.id = mp.media_id
                JOIN folders f ON f.id = m.folder_id
                WHERE mp.id = ? AND mp.person_id = ? AND mp.state = 1 AND m.is_video = 0
                LIMIT 1;
                """,
                (target_face_id, pid),
            ).fetchone()

        if cand_row is None:
            cand_row = conn_read.execute(
                """
                SELECT mp.id AS face_id, mp.rect_x, mp.rect_y, mp.rect_w, mp.rect_h,
                       m.filename, m.orientation, f.path AS folder_path
                FROM media_people mp
                JOIN media m ON m.id = mp.media_id
                JOIN folders f ON f.id = m.folder_id
                WHERE mp.person_id = ? AND mp.state = 1
                  AND mp.rect_w > 0.005 AND mp.rect_h > 0.005 AND m.is_video = 0
                ORDER BY mp.is_manual DESC, mp.id ASC
                LIMIT 1;
                """,
                (pid,),
            ).fetchone()

        new_blob: bytes | None = None
        new_avatar_fid: int | None = None
        if cand_row is not None:
            new_avatar_fid = int(cand_row["face_id"])
            # Prefer the pre-computed upright face thumbnail directly (matches the UI face crop)
            new_blob = self.get_face_thumbnail(new_avatar_fid)
            if not new_blob:
                full_path = os.path.join(str(cand_row["folder_path"]), str(cand_row["filename"]))
                rect = (
                    float(cand_row["rect_x"] or 0.0),
                    float(cand_row["rect_y"] or 0.0),
                    float(cand_row["rect_w"] or 0.0),
                    float(cand_row["rect_h"] or 0.0),
                )
                orient = int(cand_row["orientation"] or 1)
                new_blob = self.extract_face_crop_blob(full_path, rect, orientation=orient)

        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute(
                "UPDATE people SET thumbnail_blob = ?, avatar_media_people_id = ? WHERE id = ?;",
                (new_blob, new_avatar_fid, pid),
            )
            conn.commit()

    def upsert_person(self, name: str, thumbnail_blob: bytes | None = None) -> int:
        """Creates or retrieves a person by name."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("person name cannot be empty")
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            cur = conn.execute(
                """
                INSERT INTO people (name, thumbnail_blob)
                VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    thumbnail_blob = COALESCE(excluded.thumbnail_blob, people.thumbnail_blob)
                RETURNING id;
                """,
                (clean_name, thumbnail_blob),
            )
            person_id = int(cur.fetchone()[0])
            conn.commit()
            return person_id

    def assign_person(
        self,
        media_id: int,
        person_name: str | int,
        rect: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        thumbnail_blob: bytes | None = None,
        *,
        mark_dirty: bool = True,
        extract_crop: bool = True,
    ) -> int:
        """
        Assigns a person tag to media_id.
        Per specification, face bounding-box coordinates (MP:RegionInfo) are stored
        only for static images (is_video = 0); videos store (0, 0, 0, 0).
        Sets state = 1, is_manual = 1 on media_people and increments people.confirmed_count.
        Extracts face crop blob OUTSIDE write_lock so GUI and readers are never blocked.
        """
        # 1. Read metadata outside write lock
        conn_read = self.engine.catalog_conn()

        if isinstance(person_name, int):
            person_id_arg = person_name
            p_row = conn_read.execute(
                "SELECT id, name, thumbnail_blob FROM people WHERE id = ?;",
                (person_id_arg,),
            ).fetchone()
            if p_row is None:
                raise KeyError(f"Person id {person_id_arg} not found")
            clean_name = str(p_row["name"])
        else:
            clean_name = str(person_name).strip()
            if not clean_name:
                raise ValueError("person_name cannot be empty")

        media_row = conn_read.execute(
            """
            SELECT m.is_video, m.filename, m.orientation, f.path AS folder_path
            FROM media m
            JOIN folders f ON f.id = m.folder_id
            WHERE m.id = ?;
            """,
            (media_id,),
        ).fetchone()
        if media_row is None:
            raise KeyError(f"Media {media_id} not found")

        is_video = bool(media_row["is_video"])
        rx, ry, rw, rh = (0.0, 0.0, 0.0, 0.0) if is_video else rect
        orient = int(media_row["orientation"] or 1)

        # 2. Extract face thumbnail crop OUTSIDE write_lock if needed
        face_crop_blob: bytes | None = None
        if extract_crop and not is_video and (rw > 0.005 and rh > 0.005):
            full_img_path = os.path.join(
                str(media_row["folder_path"]), str(media_row["filename"])
            )
            face_crop_blob = self.extract_face_crop_blob(
                full_img_path, (rx, ry, rw, rh), orientation=orient
            )
        if extract_crop and thumbnail_blob is None and not is_video:
            existing_p = conn_read.execute(
                "SELECT thumbnail_blob FROM people WHERE name = ?;",
                (clean_name,),
            ).fetchone()
            if existing_p is None or existing_p["thumbnail_blob"] is None:
                if face_crop_blob is not None:
                    thumbnail_blob = face_crop_blob
                else:
                    full_img_path = os.path.join(
                        str(media_row["folder_path"]), str(media_row["filename"])
                    )
                    thumbnail_blob = self.extract_face_crop_blob(
                        full_img_path, (rx, ry, rw, rh), orientation=orient
                    )

        # 3. Perform atomic DB mutations inside write lock
        old_displaced_pid: int | None = None
        mp_id: int = 0
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            was_clean = False
            if not mark_dirty:
                d_row = conn.execute(
                    "SELECT metadata_dirty FROM media WHERE id = ?;", (media_id,)
                ).fetchone()
                was_clean = d_row is not None and int(d_row[0] or 0) == 0
            cur = conn.execute(
                """
                INSERT INTO people (name, thumbnail_blob)
                VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    thumbnail_blob = COALESCE(excluded.thumbnail_blob, people.thumbnail_blob)
                RETURNING id;
                """,
                (clean_name, thumbnail_blob),
            )
            person_id = int(cur.fetchone()[0])

            # Deduplicate against overlapping face regions on this media
            existing_mp = conn.execute(
                """
                SELECT id, person_id, state FROM media_people
                WHERE media_id = ?
                  AND (
                      (ABS(rect_x - ?) < 0.02 AND ABS(rect_y - ?) < 0.02
                       AND ABS(rect_w - ?) < 0.02 AND ABS(rect_h - ?) < 0.02)
                      OR (state = 0 AND ABS(rect_x - ?) < 0.04 AND ABS(rect_y - ?) < 0.04
                          AND ABS(rect_w - ?) < 0.04 AND ABS(rect_h - ?) < 0.04)
                      OR (person_id = ? AND ? = 0.0 AND ? = 0.0)
                  );
                """,
                (media_id, rx, ry, rw, rh, rx, ry, rw, rh, person_id, rw, rh),
            ).fetchone()

            if existing_mp is not None:
                mp_id = int(existing_mp["id"])
                old_pid = existing_mp["person_id"]
                old_state = int(existing_mp["state"])
                conn.execute(
                    """
                    UPDATE media_people
                    SET person_id = ?, state = 1, is_manual = 1, suggested_person_id = NULL,
                        rect_x = CASE WHEN ? > 0 THEN ? ELSE rect_x END,
                        rect_y = CASE WHEN ? > 0 THEN ? ELSE rect_y END,
                        rect_w = CASE WHEN ? > 0 THEN ? ELSE rect_w END,
                        rect_h = CASE WHEN ? > 0 THEN ? ELSE rect_h END
                    WHERE id = ?;
                    """,
                    (person_id, rw, rx, rh, ry, rw, rw, rh, rh, mp_id),
                )
                if old_state != 1 or old_pid != person_id:
                    conn.execute(
                        "UPDATE people SET confirmed_count = confirmed_count + 1 WHERE id = ?;",
                        (person_id,),
                    )
                    if old_pid is not None and old_state == 1 and old_pid != person_id:
                        old_displaced_pid = int(old_pid)
                        conn.execute(
                            "UPDATE people SET confirmed_count = MAX(0, confirmed_count - 1) WHERE id = ?;",
                            (old_pid,),
                        )
            else:
                cur_mp = conn.execute(
                    """
                    INSERT INTO media_people (media_id, person_id, state, is_manual, rect_x, rect_y, rect_w, rect_h)
                    VALUES (?, ?, 1, 1, ?, ?, ?, ?)
                    RETURNING id;
                    """,
                    (media_id, person_id, rx, ry, rw, rh),
                )
                mp_id = int(cur_mp.fetchone()[0])
                conn.execute(
                    "UPDATE people SET confirmed_count = confirmed_count + 1 WHERE id = ?;",
                    (person_id,),
                )
            if mp_id > 0 and thumbnail_blob is not None:
                conn.execute(
                    "UPDATE people SET avatar_media_people_id = COALESCE(avatar_media_people_id, ?) WHERE id = ?;",
                    (mp_id, person_id),
                )
            if mark_dirty:
                conn.execute(
                    "UPDATE media SET metadata_dirty = 1 WHERE id = ?;",
                    (media_id,),
                )
            elif was_clean:
                # Reset any AFTER INSERT/UPDATE trigger dirtiness atomically before commit
                conn.execute(
                    "UPDATE media SET metadata_dirty = 0 WHERE id = ?;",
                    (media_id,),
                )
            conn.commit()

        if mp_id > 0 and face_crop_blob is not None:
            self._put_face_thumbnail_direct(mp_id, face_crop_blob)
        if old_displaced_pid is not None and mp_id > 0:
            self.refresh_person_avatar(old_displaced_pid, invalidated_face_ids=[mp_id])
        return person_id

    def add_person_region(
        self,
        media_id: int,
        person_name: str,
        rx: float,
        ry: float,
        rw: float,
        rh: float,
    ) -> int:
        """Assigns a person tag with normalized MP:RegionInfo coordinates [rx, ry, rw, rh] to media_id."""
        return self.assign_person(media_id, person_name, rect=(rx, ry, rw, rh))

    def get_people_for_media(self, media_id: int) -> list[dict[str, Any]]:
        """Returns confirmed people and their normalized face rectangles for media_id."""
        conn = self.engine.catalog_conn()
        rows = conn.execute(
            """
            SELECT p.id, p.id AS person_id, p.name, p.thumbnail_blob,
                   mp.rect_x, mp.rect_y, mp.rect_w, mp.rect_h
            FROM people p
            JOIN media_people mp ON mp.person_id = p.id
            WHERE mp.media_id = ? AND mp.state = 1
            ORDER BY p.name COLLATE NOCASE;
            """,
            (media_id,),
        ).fetchall()
        return [
            {
                **dict(r),
                "person_id": int(r["id"]),
                "rect": (
                    float(r["rect_x"] or 0.0),
                    float(r["rect_y"] or 0.0),
                    float(r["rect_w"] or 0.0),
                    float(r["rect_h"] or 0.0),
                ),
            }
            for r in rows
        ]

    def get_untagged_faces_for_media(self, media_id: int) -> list[dict[str, Any]]:
        """Returns unassigned detected face regions (state = 0) on media_id, largest first."""
        conn = self.engine.catalog_conn()
        rows = conn.execute(
            """
            SELECT id, media_id, rect_x, rect_y, rect_w, rect_h, suggested_person_id, similarity_score
            FROM media_people
            WHERE media_id = ? AND state = 0 AND rect_w > 0 AND rect_h > 0
            ORDER BY (rect_w * rect_h) DESC, id ASC;
            """,
            (media_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def backfill_missing_people_thumbnails(self) -> int:
        """
        Populates or heals `people.thumbnail_blob` and `people.avatar_media_people_id`
        for any person row whose avatar blob is NULL or whose avatar face link is missing/stale.
        """
        conn = self.engine.catalog_conn()
        missing = conn.execute(
            """
            SELECT p.id, p.name
            FROM people p
            LEFT JOIN media_people amp ON amp.id = p.avatar_media_people_id
            WHERE p.thumbnail_blob IS NULL
               OR p.avatar_media_people_id IS NULL
               OR amp.id IS NULL
               OR amp.person_id != p.id
               OR amp.state != 1;
            """
        ).fetchall()
        if not missing:
            return 0

        updated = 0
        for p_row in missing:
            pid = int(p_row["id"])
            best_mp = conn.execute(
                """
                SELECT mp.id AS face_id, mp.rect_x, mp.rect_y, mp.rect_w, mp.rect_h,
                       m.filename, f.path AS folder_path
                FROM media_people mp
                JOIN media m ON m.id = mp.media_id
                JOIN folders f ON f.id = m.folder_id
                WHERE mp.person_id = ? AND mp.state = 1
                  AND m.is_video = 0 AND mp.rect_w > 0.005 AND mp.rect_h > 0.005
                ORDER BY mp.is_manual DESC, (mp.rect_w * mp.rect_h) DESC, m.date_taken DESC
                LIMIT 1;
                """,
                (pid,),
            ).fetchone()
            if best_mp is None:
                continue
            best_fid = int(best_mp["face_id"])
            full_path = os.path.join(
                str(best_mp["folder_path"]), str(best_mp["filename"])
            )
            rect = (
                float(best_mp["rect_x"] or 0.0),
                float(best_mp["rect_y"] or 0.0),
                float(best_mp["rect_w"] or 0.0),
                float(best_mp["rect_h"] or 0.0),
            )
            blob = self.extract_face_crop_blob(full_path, rect)
            if blob:
                try:
                    with self.engine.write_lock:
                        conn.execute(
                            "UPDATE people SET thumbnail_blob = ?, avatar_media_people_id = ? WHERE id = ?;",
                            (blob, best_fid, pid),
                        )
                        conn.commit()
                    updated += 1
                except Exception:
                    break
        return updated

    def remove_person_tag(self, media_id: int, person_id: int) -> None:
        removed_ids: list[int] = []
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            rows = conn.execute(
                "SELECT id FROM media_people WHERE media_id = ? AND person_id = ?;",
                (media_id, person_id),
            ).fetchall()
            removed_ids = [int(r["id"]) for r in rows]
            conn.execute(
                "DELETE FROM media_people WHERE media_id = ? AND person_id = ?;",
                (media_id, person_id),
            )
            conn.execute(
                """
                UPDATE people
                SET confirmed_count = (
                    SELECT COUNT(*) FROM media_people WHERE person_id = people.id AND state = 1
                )
                WHERE id = ?;
                """,
                (person_id,),
            )
            conn.execute(
                "UPDATE media SET metadata_dirty = 1 WHERE id = ?;",
                (media_id,),
            )
            conn.commit()
        if removed_ids:
            self._invalidate_face_thumbnails_direct(removed_ids)
            self.refresh_person_avatar(person_id, invalidated_face_ids=removed_ids)

    remove_person_from_media = remove_person_tag

    def rename_person(self, person_id: int, new_name: str) -> int:
        """
        Renames a person contact. If another contact with `new_name` already exists,
        merges `person_id` into that existing contact. Marks affected media dirty.
        """
        clean_name = new_name.strip()
        if not clean_name:
            raise ValueError("Person name cannot be empty")
        conn_read = self.engine.catalog_conn()
        existing = conn_read.execute(
            "SELECT id FROM people WHERE name = ? COLLATE NOCASE;",
            (clean_name,),
        ).fetchone()
        if existing is not None and int(existing["id"]) != int(person_id):
            target_id = int(existing["id"])
            self.merge_people(int(person_id), target_id)
            return target_id

        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute(
                "UPDATE people SET name = ? WHERE id = ?;",
                (clean_name, int(person_id)),
            )
            conn.execute(
                """
                UPDATE media SET metadata_dirty = 1
                WHERE id IN (SELECT media_id FROM media_people WHERE person_id = ? AND state = 1);
                """,
                (int(person_id),),
            )
            conn.commit()
        return int(person_id)

    def merge_people(self, source_person_id: int, target_person_id: int) -> None:
        """
        Merges `source_person_id` into `target_person_id`:
        - Reassigns all confirmed and suggested face regions to `target_person_id`
        - Reassigns rejected suggestions and face rejections without constraint collisions
        - Preserves avatar thumbnail if target has none
        - Deduplicates any double-assigned face regions on shared photos
        - Updates running vector centroid for target person
        - Marks affected media dirty and deletes `source_person_id`
        """
        src_id = int(source_person_id)
        dst_id = int(target_person_id)
        if src_id == dst_id:
            return
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            src_row = conn.execute("SELECT id FROM people WHERE id = ?;", (src_id,)).fetchone()
            dst_row = conn.execute("SELECT id FROM people WHERE id = ?;", (dst_id,)).fetchone()
            if not src_row or not dst_row:
                logger.warning("merge_people: invalid source (%s) or target (%s) person id", src_id, dst_id)
                return

            conn.execute(
                """
                UPDATE people
                SET thumbnail_blob = COALESCE(
                    thumbnail_blob,
                    (SELECT thumbnail_blob FROM people WHERE id = ?)
                ),
                avatar_media_people_id = COALESCE(
                    avatar_media_people_id,
                    (SELECT avatar_media_people_id FROM people WHERE id = ?)
                )
                WHERE id = ?;
                """,
                (src_id, src_id, dst_id),
            )
            conn.execute(
                "UPDATE media_people SET person_id = ? WHERE person_id = ?;",
                (dst_id, src_id),
            )
            conn.execute(
                "UPDATE media_people SET suggested_person_id = ? WHERE suggested_person_id = ?;",
                (dst_id, src_id),
            )
            # Reassign rejections without UNIQUE constraint collisions
            conn.execute(
                "UPDATE OR IGNORE face_rejections SET person_id = ? WHERE person_id = ?;",
                (dst_id, src_id),
            )
            conn.execute("DELETE FROM face_rejections WHERE person_id = ?;", (src_id,))
            conn.execute(
                "UPDATE OR IGNORE face_rejected_suggestions SET person_id = ? WHERE person_id = ?;",
                (dst_id, src_id),
            )
            conn.execute("DELETE FROM face_rejected_suggestions WHERE person_id = ?;", (src_id,))

            conn.execute(
                """
                UPDATE people
                SET confirmed_count = (
                    SELECT COUNT(*) FROM media_people WHERE person_id = ? AND state = 1
                )
                WHERE id = ?;
                """,
                (dst_id, dst_id),
            )
            conn.execute(
                """
                UPDATE media SET metadata_dirty = 1
                WHERE id IN (SELECT media_id FROM media_people WHERE person_id = ? AND state = 1);
                """,
                (dst_id,),
            )
            conn.execute("DELETE FROM people WHERE id = ?;", (src_id,))
            conn.commit()

        # Deduplicate any duplicate person face tags on photos where both people were tagged
        try:
            self.deduplicate_media_people(person_id=dst_id)
        except Exception:
            pass

        # Recalculate target person centroid / sub-centroids
        self.recalculate_person_centroid(dst_id)

    def set_person_display_picture(self, person_id: int, media_id: int) -> bool:
        """
        Sets `people.thumbnail_blob` for `person_id` using their face crop on `media_id`
        (or center square crop if no face rectangle is stored).
        """
        conn_read = self.engine.catalog_conn()
        row = conn_read.execute(
            """
            SELECT m.filename, m.orientation, f.path AS folder_path,
                   mp.id AS face_id, mp.rect_x, mp.rect_y, mp.rect_w, mp.rect_h
            FROM media m
            JOIN folders f ON f.id = m.folder_id
            LEFT JOIN media_people mp ON mp.media_id = m.id AND mp.person_id = ?
            WHERE m.id = ?
            ORDER BY (COALESCE(mp.rect_w, 0) * COALESCE(mp.rect_h, 0)) DESC
            LIMIT 1;
            """,
            (int(person_id), int(media_id)),
        ).fetchone()
        if row is None:
            return False
        fid = int(row["face_id"]) if row["face_id"] is not None else None
        blob: bytes | None = None
        if fid is not None:
            # Prefer the pre-computed upright face thumbnail directly
            blob = self.get_face_thumbnail(fid)

        if not blob:
            full_path = os.path.join(str(row["folder_path"]), str(row["filename"]))
            rect = (
                float(row["rect_x"] or 0.0),
                float(row["rect_y"] or 0.0),
                float(row["rect_w"] or 0.0),
                float(row["rect_h"] or 0.0),
            )
            orient = int(row["orientation"] or 1)
            blob = self.extract_face_crop_blob(full_path, rect, orientation=orient)

        if not blob:
            return False
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute(
                "UPDATE people SET thumbnail_blob = ?, avatar_media_people_id = ? WHERE id = ?;",
                (blob, fid, int(person_id)),
            )
            conn.commit()
        return True

    def _get_restricted_person_ids(self, conn: sqlite3.Connection) -> set[int]:
        """Returns person IDs appearing in private/excluded folders using indexed folder_id lookup."""
        if self.include_private:
            return set()
        restricted_fids = [
            int(r[0])
            for r in conn.execute(
                """
                SELECT id FROM folders
                WHERE COALESCE(is_private, 0) = 1
                   OR COALESCE(monitor_mode, 'always') = 'excluded';
                """
            ).fetchall()
        ]
        if not restricted_fids:
            return set()
        restricted_pids: set[int] = set()
        for chunk in _iter_id_chunks(restricted_fids, 900):
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"""
                SELECT mp.person_id, mp.suggested_person_id
                FROM media m
                JOIN media_people mp ON mp.media_id = m.id
                WHERE m.folder_id IN ({placeholders})
                  AND (mp.person_id IS NOT NULL OR mp.suggested_person_id IS NOT NULL);
                """,
                list(chunk),
            ).fetchall()
            for pid, spid in rows:
                if pid is not None:
                    restricted_pids.add(int(pid))
                if spid is not None:
                    restricted_pids.add(int(spid))
        return restricted_pids

    def get_person_names(self) -> list[str]:
        """Lightweight (<1ms) person name list for autocomplete widgets without BLOBs or face joins."""
        conn = self.engine.catalog_conn()
        restricted_pids = self._get_restricted_person_ids(conn)
        rows = conn.execute(
            "SELECT id, name FROM people ORDER BY name COLLATE NOCASE ASC;"
        ).fetchall()
        if not restricted_pids:
            return [str(r[1]) for r in rows if r[1]]
        return [str(r[1]) for r in rows if r[1] and int(r[0]) not in restricted_pids]

    def get_people_list(self) -> list[dict[str, Any]]:
        conn = self.engine.catalog_conn()
        priv_clause = self._private_folder_clause("f")
        restricted_pids = self._get_restricted_person_ids(conn)

        rows = conn.execute(
            f"""
            SELECT p.id, p.name, p.thumbnail_blob, p.confirmed_count,
                   COALESCE(c.media_count, 0) AS media_count
            FROM people p
            LEFT JOIN (
                SELECT mp.person_id, COUNT(DISTINCT mp.media_id) AS media_count
                FROM media_people mp
                JOIN media m ON m.id = mp.media_id
                JOIN folders f ON f.id = m.folder_id
                WHERE mp.state = 1 AND mp.person_id IS NOT NULL
                  AND COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
                GROUP BY mp.person_id
            ) c ON c.person_id = p.id
            ORDER BY p.name COLLATE NOCASE ASC;
            """
        ).fetchall()
        if not restricted_pids:
            return [dict(r) for r in rows]
        return [
            dict(r)
            for r in rows
            if int(r["media_count"] or 0) > 0 or int(r["id"]) not in restricted_pids
        ]

    get_all_people = get_people_list

    def get_named_people_directory_data(
        self, min_confidence: float = 0.0
    ) -> list[dict[str, Any]]:
        """Fast query (<60ms) for named people with photo and pending suggestion counts."""
        conn = self.engine.catalog_conn()
        thresh = float(min_confidence)
        priv_clause = self._private_folder_clause("f")
        restricted_pids = self._get_restricted_person_ids(conn)
        named_rows = conn.execute(
            f"""
            SELECT p.id, p.name, p.thumbnail_blob, p.confirmed_count,
                   COALESCE(c.photo_count, 0) AS photo_count,
                   COALESCE(s.suggestion_count, 0) AS suggestion_count
            FROM people p
            LEFT JOIN (
                SELECT mp.person_id, COUNT(DISTINCT mp.media_id) AS photo_count
                FROM media_people mp
                JOIN media m ON m.id = mp.media_id
                JOIN folders f ON f.id = m.folder_id
                WHERE mp.state = 1 AND mp.person_id IS NOT NULL
                  AND COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
                GROUP BY mp.person_id
            ) c ON c.person_id = p.id
            LEFT JOIN (
                SELECT mp.suggested_person_id, COUNT(*) AS suggestion_count
                FROM media_people mp
                JOIN media m ON m.id = mp.media_id
                JOIN folders f ON f.id = m.folder_id
                WHERE mp.state = 0 AND mp.suggested_person_id IS NOT NULL
                  AND COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
                  AND (? <= 0.0 OR COALESCE(mp.similarity_score, 1.0) >= ?)
                GROUP BY mp.suggested_person_id
            ) s ON s.suggested_person_id = p.id
            ORDER BY photo_count DESC, p.name COLLATE NOCASE ASC;
            """,
            (thresh, thresh),
        ).fetchall()

        named_people: list[dict[str, Any]] = []
        for r in named_rows:
            pid = int(r["id"])
            pc = int(r["photo_count"] or 0)
            sc = int(r["suggestion_count"] or 0)
            if restricted_pids and pc == 0 and sc == 0 and pid in restricted_pids:
                continue
            named_people.append({
                "id": pid,
                "name": str(r["name"]),
                "thumbnail_blob": bytes(r["thumbnail_blob"]) if r["thumbnail_blob"] else None,
                "photo_count": pc,
                "suggestion_count": sc,
            })
        return named_people

    def get_unnamed_face_clusters(
        self,
        min_confidence: float = 0.52,
        *,
        max_faces: int = 4000,
        cancel_flag: Any = None,
    ) -> list[dict[str, Any]]:
        """Clusters unassigned face embeddings without a confident suggestion into recurring unnamed groups."""
        if cancel_flag is not None and getattr(cancel_flag, "is_set", False):
            return []
        conn = self.engine.catalog_conn()
        priv_clause = self._private_folder_clause("f")
        unassigned_rows = conn.execute(
            f"""
            SELECT mp.id, mp.embedding, mp.suggested_person_id, p.name AS suggested_person_name
            FROM media_people mp
            JOIN media m ON m.id = mp.media_id
            JOIN folders f ON f.id = m.folder_id
            LEFT JOIN people p ON p.id = mp.suggested_person_id
            WHERE mp.state = 0
              AND mp.embedding IS NOT NULL
              AND COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
              AND (mp.suggested_person_id IS NULL OR COALESCE(mp.similarity_score, 0.0) < ?)
            ORDER BY mp.id DESC
            LIMIT ?;
            """,
            (float(min_confidence), max(100, int(max_faces))),
        ).fetchall()

        unnamed_clusters: list[dict[str, Any]] = []
        if unassigned_rows:
            from src.core.face_matcher import FaceVectorMatcher, deserialize_embedding
            import numpy as np

            valid_fids: list[int] = []
            emb_list: list[np.ndarray] = []
            face_suggestions: dict[int, str] = {}
            for r in unassigned_rows:
                if cancel_flag is not None and getattr(cancel_flag, "is_set", False):
                    return []
                raw_emb = bytes(r["embedding"]) if r["embedding"] else None
                if raw_emb and len(raw_emb) >= 2048:
                    fid = int(r["id"])
                    valid_fids.append(fid)
                    emb_list.append(deserialize_embedding(raw_emb))
                    if r["suggested_person_name"]:
                        face_suggestions[fid] = str(r["suggested_person_name"])

            if len(valid_fids) >= 3 and len(emb_list) >= 3:
                clusters = FaceVectorMatcher.cluster_unnamed_faces(
                    valid_fids,
                    np.stack(emb_list, axis=0),
                    sim_threshold=0.58,
                    min_cluster_size=3,
                    cancel_flag=cancel_flag,
                )
                for cluster_name, fids in clusters.items():
                    if cluster_name != "Other Unnamed Faces" and len(fids) >= 3:
                        cluster_suggs = [
                            face_suggestions[fid] for fid in fids if fid in face_suggestions
                        ]
                        best_sugg = (
                            max(set(cluster_suggs), key=cluster_suggs.count)
                            if cluster_suggs else None
                        )
                        unnamed_clusters.append({
                            "cluster_name": cluster_name,
                            "face_ids": fids,
                            "face_count": len(fids),
                            "representative_face_id": fids[0],
                            "suggested_person_name": best_sugg,
                        })
        return unnamed_clusters

    def get_people_directory_data(self, include_unnamed_clusters: bool = True) -> dict[str, Any]:
        """
        Fetches named people with confirmed photo counts and pending suggestion counts,
        plus recurring unnamed face clusters (>= 3 unassigned faces) if requested.
        """
        named_people = self.get_named_people_directory_data()
        unnamed_clusters = self.get_unnamed_face_clusters() if include_unnamed_clusters else []
        return {
            "named_people": named_people,
            "unnamed_clusters": unnamed_clusters,
        }

    def get_unconfirmed_suggestions_for_person(
        self, person_id: int, limit: int = 15
    ) -> list[dict[str, Any]]:
        """
        Returns pending suggestions (state = 0) for person_id, with face_id, confidence, media_id.
        """
        conn = self.engine.catalog_conn()
        priv_clause = self._private_folder_clause("f")
        rows = conn.execute(
            f"""
            SELECT mp.id AS face_id, mp.media_id, mp.rect_x, mp.rect_y, mp.rect_w, mp.rect_h,
                   COALESCE(mp.similarity_score, 0.0) AS confidence, m.filename, f.path AS folder_path
            FROM media_people mp
            JOIN media m ON m.id = mp.media_id
            JOIN folders f ON f.id = m.folder_id
            WHERE mp.state = 0 AND mp.suggested_person_id = ?
              AND COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
            ORDER BY COALESCE(mp.similarity_score, 0.0) DESC
            LIMIT ?;
            """,
            (int(person_id), int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]

    def set_person_thumbnail_blob(self, person_id: int, thumb_blob: bytes) -> None:
        """Sets the cover thumbnail blob for person_id."""
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute(
                "UPDATE people SET thumbnail_blob = ? WHERE id = ?;",
                (thumb_blob, int(person_id)),
            )
            conn.commit()

    def reparse_and_heal_face_regions_from_files(self) -> int:
        """
        Re-parses XMP face regions from disk using the fixed MWG/WLPG orientation-aware parser
        for all media files with media_people rows or face_scan_version > 0.
        Updates confirmed coordinates, resets embedding = NULL, removes unassigned state = 0 detections,
        and resets face_scan_version = 0 so FaceDetectionWorker verifies and snaps historical boxes.
        Returns the count of media records updated.
        """
        from src.workers.xmp_sync_worker import read_xmp_from_file
        from PIL import Image

        conn_read = self.engine.catalog_conn()
        rows = conn_read.execute(
            """
            SELECT DISTINCT m.id, m.filename, m.orientation, m.width, m.height, f.path AS folder_path
            FROM media m
            JOIN folders f ON f.id = m.folder_id
            WHERE m.is_missing = 0 AND m.is_video = 0
              AND (
                EXISTS (SELECT 1 FROM media_people mp WHERE mp.media_id = m.id)
                OR m.face_scan_version > 0
              );
            """
        ).fetchall()

        if not rows:
            return 0

        updated_count = 0
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            for r in rows:
                mid = int(r["id"])
                fpath = Path(r["folder_path"]) / str(r["filename"])
                if not fpath.exists():
                    continue

                orient = int(r["orientation"] or 1)
                disp_w = r["width"]
                disp_h = r["height"]
                raw_w = None
                raw_h = None
                try:
                    with Image.open(fpath) as img:
                        raw_w, raw_h = img.width, img.height
                        if orient in (5, 6, 7, 8):
                            disp_w, disp_h = raw_h, raw_w
                        else:
                            disp_w, disp_h = raw_w, raw_h
                except Exception:
                    pass

                xmp = read_xmp_from_file(
                    fpath,
                    image_width=disp_w,
                    image_height=disp_h,
                    raw_width=raw_w,
                    raw_height=raw_h,
                    orientation=orient,
                )

                people = xmp.get("people", [])
                if people:
                    # Remove unassigned detections
                    conn.execute("DELETE FROM media_people WHERE media_id = ? AND state != 1;", (mid,))
                    # For confirmed faces: update or insert
                    for p in people:
                        pname = p.get("name", "").strip()
                        if not pname:
                            continue
                        person_id = self.upsert_person(pname)
                        rect = p.get("rect", (0.0, 0.0, 0.0, 0.0))
                        rx, ry, rw, rh = rect

                        existing = conn.execute(
                            "SELECT id FROM media_people WHERE media_id = ? AND person_id = ? AND state = 1 LIMIT 1;",
                            (mid, person_id),
                        ).fetchone()
                        if existing:
                            conn.execute(
                                """
                                UPDATE media_people
                                SET rect_x = ?, rect_y = ?, rect_w = ?, rect_h = ?, embedding = NULL
                                WHERE id = ?;
                                """,
                                (rx, ry, rw, rh, int(existing["id"])),
                            )
                        else:
                            conn.execute(
                                """
                                INSERT INTO media_people (
                                    media_id, person_id, suggested_person_id, similarity_score,
                                    state, is_manual, rect_x, rect_y, rect_w, rect_h, embedding
                                )
                                VALUES (?, ?, NULL, NULL, 1, 1, ?, ?, ?, ?, NULL);
                                """,
                                (mid, person_id, rx, ry, rw, rh),
                            )
                else:
                    # Reset embeddings on existing confirmed faces to force re-verification
                    conn.execute(
                        "UPDATE media_people SET embedding = NULL WHERE media_id = ? AND state = 1;",
                        (mid,),
                    )
                    conn.execute("DELETE FROM media_people WHERE media_id = ? AND state != 1;", (mid,))

                # Reset face_scan_version so FaceDetectionWorker verifies and snaps
                conn.execute("UPDATE media SET face_scan_version = 0 WHERE id = ?;", (mid,))
                updated_count += 1

            conn.commit()

        return updated_count

    def deduplicate_media_people(
        self, media_id: int | None = None, person_id: int | None = None
    ) -> int:
        """
        Removes duplicate / phantom face boxes for the same person on the same media.
        Keeps the highest-quality box (having embedding, valid face dimensions)
        and deletes phantom duplicate tags, then marks metadata_dirty = 1 so the clean metadata is written back.
        """
        deleted_count = 0
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            if media_id is not None:
                media_ids = [media_id]
            elif person_id is not None:
                rows = conn.execute(
                    """
                    SELECT media_id
                    FROM media_people
                    WHERE person_id = ? AND state = 1
                    GROUP BY media_id
                    HAVING COUNT(*) > 1;
                    """,
                    (int(person_id),),
                ).fetchall()
                media_ids = [int(r["media_id"]) for r in rows]
            else:
                rows = conn.execute(
                    """
                    SELECT media_id, person_id
                    FROM media_people
                    WHERE state = 1 AND person_id IS NOT NULL
                    GROUP BY media_id, person_id
                    HAVING COUNT(*) > 1;
                    """
                ).fetchall()
                media_ids = list({int(r["media_id"]) for r in rows})

            for mid in media_ids:
                dup_people = conn.execute(
                    """
                    SELECT person_id
                    FROM media_people
                    WHERE media_id = ? AND state = 1 AND person_id IS NOT NULL
                    GROUP BY person_id
                    HAVING COUNT(*) > 1;
                    """,
                    (mid,),
                ).fetchall()

                for dp in dup_people:
                    pid = int(dp["person_id"])
                    all_rows = conn.execute(
                        """
                        SELECT id, rect_x, rect_y, rect_w, rect_h, embedding
                        FROM media_people
                        WHERE media_id = ? AND person_id = ? AND state = 1
                        ORDER BY (embedding IS NOT NULL) DESC,
                                 (CASE WHEN rect_w > 0.05 AND rect_h > 0.05 THEN 1 ELSE 0 END) DESC,
                                 id ASC;
                        """,
                        (mid, pid),
                    ).fetchall()

                    if len(all_rows) > 1:
                        # Keep the best row (index 0), remove duplicates
                        keep_id = int(all_rows[0]["id"])
                        remove_ids = [int(r["id"]) for r in all_rows[1:]]
                        placeholders = ",".join("?" for _ in remove_ids)
                        conn.execute(
                            f"DELETE FROM media_people WHERE id IN ({placeholders});",
                            remove_ids,
                        )
                        conn.execute(
                            "UPDATE people SET confirmed_count = MAX(1, confirmed_count - ?) WHERE id = ?;",
                            (len(remove_ids), pid),
                        )
                        conn.execute("UPDATE media SET metadata_dirty = 1 WHERE id = ?;", (mid,))
                        deleted_count += len(remove_ids)

            conn.commit()
        return deleted_count

    # -------------------------------------------------------------------------
    # Tier 2 Face Recognition Engine Repository Operations
    # -------------------------------------------------------------------------
    def insert_detected_faces(
        self, media_id: int, detections: Sequence[dict[str, Any]]
    ) -> list[int]:
        """
        Inserts detected face regions with ArcFace 512-D float32 embeddings into media_people.
        Deduplicates against existing face boxes on media_id within 0.02 normalized coordinate tolerance.
        Sets state = 0 (unassigned), is_manual = 0.
        Returns list of media_people.id values created or updated.
        """
        if not detections:
            return []
        result_ids: list[int] = []
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            for d in detections:
                if "rect" in d and isinstance(d["rect"], (tuple, list)) and len(d["rect"]) == 4:
                    rx, ry, rw, rh = [float(x) for x in d["rect"]]
                else:
                    rx = float(d.get("rect_x", 0.0))
                    ry = float(d.get("rect_y", 0.0))
                    rw = float(d.get("rect_w", 0.0))
                    rh = float(d.get("rect_h", 0.0))
                embedding = d.get("embedding")
                similarity_score = d.get("similarity_score")

                existing = conn.execute(
                    """
                    SELECT id, embedding FROM media_people
                    WHERE media_id = ?
                      AND ABS(rect_x - ?) < 0.02 AND ABS(rect_y - ?) < 0.02
                      AND ABS(rect_w - ?) < 0.02 AND ABS(rect_h - ?) < 0.02;
                    """,
                    (media_id, rx, ry, rw, rh),
                ).fetchone()

                if existing is not None:
                    mp_id = int(existing["id"])
                    if existing["embedding"] is None and embedding is not None:
                        conn.execute(
                            "UPDATE media_people SET embedding = ?, similarity_score = COALESCE(?, similarity_score) WHERE id = ?;",
                            (embedding, similarity_score, mp_id),
                        )
                    result_ids.append(mp_id)
                else:
                    cur = conn.execute(
                        """
                        INSERT INTO media_people (
                            media_id, person_id, suggested_person_id, similarity_score,
                            state, is_manual, rect_x, rect_y, rect_w, rect_h, embedding
                        )
                        VALUES (?, NULL, NULL, ?, 0, 0, ?, ?, ?, ?, ?)
                        RETURNING id;
                        """,
                        (media_id, similarity_score, rx, ry, rw, rh, embedding),
                    )
                    row = cur.fetchone()
                    if row:
                        result_ids.append(int(row[0]))
            conn.commit()
        return result_ids

    def get_unassigned_face_embeddings(
        self, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Fetches unassigned face detections (state = 0) that have embeddings for batch matching.
        """
        conn = self.engine.catalog_conn()
        query = """
            SELECT id, media_id, embedding, similarity_score,
                   suggested_person_id, rect_x, rect_y, rect_w, rect_h
            FROM media_people
            WHERE state = 0 AND embedding IS NOT NULL
            ORDER BY id ASC
        """
        if limit is not None and limit > 0:
            query += f" LIMIT {int(limit)}"
        rows = conn.execute(query).fetchall()
        return [dict(r) for r in rows]

    def get_people_centroids(self) -> dict[int, bytes]:
        """Returns mapping {person_id: centroid_embedding_blob} for all people with centroids."""
        conn = self.engine.catalog_conn()
        rows = conn.execute(
            "SELECT id, centroid_embedding FROM people WHERE centroid_embedding IS NOT NULL;"
        ).fetchall()
        return {int(r["id"]): bytes(r["centroid_embedding"]) for r in rows if r["centroid_embedding"]}

    def update_person_centroid(
        self, person_id: int, centroid_blob: bytes | None
    ) -> None:
        """Updates the 512-D L2-normalized ArcFace centroid embedding for a person."""
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute(
                "UPDATE people SET centroid_embedding = ? WHERE id = ?;",
                (centroid_blob, person_id),
            )
            conn.commit()

    def confirm_faces(
        self, media_people_ids: Sequence[int] | int, person_id: int
    ) -> None:
        """
        Confirms suggested or unassigned face regions as belonging to person_id (state = 1).
        Updates people.confirmed_count and marks affected media dirty for XMP write-back.
        """
        if isinstance(media_people_ids, int):
            ids: Sequence[int] = [media_people_ids]
        else:
            ids = media_people_ids
        if not ids:
            return
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            for chunk in _iter_id_chunks(ids, 900):
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"""
                    UPDATE media_people
                    SET person_id = ?, state = 1, suggested_person_id = NULL
                    WHERE id IN ({placeholders});
                    """,
                    [person_id, *chunk],
                )
            conn.execute(
                """
                UPDATE people
                SET confirmed_count = (
                    SELECT COUNT(*) FROM media_people WHERE person_id = people.id AND state = 1
                )
                WHERE id = ?;
                """,
                (person_id,),
            )
            conn.commit()

    confirm_face_suggestion = confirm_faces

    def assign_person_to_face(self, face_id: int, person_id: int) -> None:
        """Assigns person_id to a specific face_id, updating confirmed counts and marking media dirty."""
        old_pid: int | None = None
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            old_row = conn.execute("SELECT media_id, person_id FROM media_people WHERE id = ?;", (face_id,)).fetchone()
            old_pid = int(old_row["person_id"]) if old_row and old_row["person_id"] else None

            conn.execute(
                "UPDATE media_people SET person_id = ?, state = 1, suggested_person_id = NULL WHERE id = ?;",
                (person_id, face_id),
            )
            for pid in {old_pid, person_id}:
                if pid:
                    conn.execute(
                        """
                        UPDATE people
                        SET confirmed_count = (
                            SELECT COUNT(*) FROM media_people WHERE person_id = people.id AND state = 1
                        )
                        WHERE id = ?;
                        """,
                        (pid,),
                    )
            if old_row:
                conn.execute("UPDATE media SET metadata_dirty = 1 WHERE id = ?;", (old_row["media_id"],))
            conn.commit()
        if old_pid and old_pid != int(person_id):
            self.refresh_person_avatar(old_pid, invalidated_face_ids=[int(face_id)])
        self.refresh_person_avatar(int(person_id))

    def unassign_face_person(self, face_id: int, person_id: int | None = None) -> None:
        """
        Removes the person tag from a face bounding box without deleting the detected box itself,
        setting it to state 0 (unassigned) so it displays as 'Who is this?'.
        Also records the rejection so this face won't be re-suggested as this person.
        """
        deleted_row = False
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            if person_id is None:
                row = conn.execute("SELECT person_id FROM media_people WHERE id = ?;", (face_id,)).fetchone()
                if row and row["person_id"]:
                    person_id = int(row["person_id"])

            if person_id:
                conn.execute(
                    "INSERT OR IGNORE INTO face_rejections (face_id, person_id) VALUES (?, ?);",
                    (face_id, person_id),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO face_rejected_suggestions (face_id, person_id) VALUES (?, ?);",
                    (face_id, person_id),
                )

            # Check if this face has a valid bounding box (rect_w > 0 and rect_h > 0)
            row = conn.execute("SELECT media_id, rect_w, rect_h FROM media_people WHERE id = ?;", (face_id,)).fetchone()
            if row and row["rect_w"] > 0 and row["rect_h"] > 0:
                conn.execute(
                    "UPDATE media_people SET person_id = NULL, suggested_person_id = NULL, state = 0 WHERE id = ?;",
                    (face_id,),
                )
            else:
                deleted_row = True
                conn.execute("DELETE FROM media_people WHERE id = ?;", (face_id,))

            if person_id:
                conn.execute(
                    """
                    UPDATE people
                    SET confirmed_count = (
                        SELECT COUNT(*) FROM media_people WHERE person_id = people.id AND state = 1
                    )
                    WHERE id = ?;
                    """,
                    (person_id,),
                )
            if row:
                conn.execute("UPDATE media SET metadata_dirty = 1 WHERE id = ?;", (row["media_id"],))
            conn.commit()
        if deleted_row:
            self._invalidate_face_thumbnails_direct([int(face_id)])
        if person_id:
            self.refresh_person_avatar(int(person_id), invalidated_face_ids=[int(face_id)])

    def verify_face_tag(self, face_id: int) -> None:
        """Marks a confirmed face tag as verified to exclude from outlier reviews."""
        conn = self.engine.catalog_conn()
        with self.engine.write_lock:
            conn.execute(
                "UPDATE media_people SET similarity_score = 1.0 WHERE id = ?;", (int(face_id),)
            )
            conn.commit()

    def recalculate_person_centroid(self, person_id: int) -> None:
        """Recalculates ArcFace centroid/sub-centroids for person_id from confirmed faces."""
        try:
            from src.core.face_matcher import (
                FaceVectorMatcher,
                deserialize_embedding,
                normalize_vector,
                serialize_embedding,
            )
            import numpy as np

            conn = self.engine.catalog_conn()
            rows = conn.execute(
                "SELECT embedding FROM media_people WHERE person_id = ? AND state = 1 AND embedding IS NOT NULL;",
                (int(person_id),),
            ).fetchall()
            vectors = [
                deserialize_embedding(bytes(r["embedding"]))
                for r in rows
                if r["embedding"] and len(bytes(r["embedding"])) >= 2048
            ]
            if vectors:
                if len(vectors) >= 10:
                    sub_c = FaceVectorMatcher.compute_sub_centroids(np.stack(vectors, axis=0), max_k=5)
                else:
                    sub_c = normalize_vector(np.mean(np.stack(vectors, axis=0), axis=0, keepdims=True))
                self.update_person_centroid(int(person_id), serialize_embedding(sub_c))
        except Exception:
            pass

    def assign_face_to_person(self, face_id: int, person_name: str | int) -> int:
        """
        Names/assigns an unknown or suggested face region (media_people.id) to a person.
        Promotes state = 1, updates person's centroid embedding, increments confirmed count,
        and marks media dirty for metadata sync. Consolidates any existing boxless row for
        this person on the same media.
        """
        if isinstance(person_name, int):
            person_id = int(person_name)
        else:
            clean_name = str(person_name).strip()
            if not clean_name:
                raise ValueError("person_name cannot be empty")
            person_id = self.upsert_person(clean_name)
        old_row = self.engine.catalog_conn().execute(
            "SELECT person_id FROM media_people WHERE id = ?;", (int(face_id),)
        ).fetchone()
        old_pid = int(old_row["person_id"]) if old_row and old_row["person_id"] else None
        self.confirm_faces(int(face_id), person_id)
        removed_placeholder_ids: list[int] = []
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            row = conn.execute(
                "SELECT media_id, embedding FROM media_people WHERE id = ?;", (int(face_id),)
            ).fetchone()
            if row:
                mid = int(row["media_id"])
                ph_rows = conn.execute(
                    """
                    SELECT id FROM media_people
                    WHERE media_id = ? AND person_id = ? AND id != ?
                      AND (rect_w <= 0.005 OR rect_h <= 0.005);
                    """,
                    (mid, person_id, int(face_id)),
                ).fetchall()
                removed_placeholder_ids = [int(r["id"]) for r in ph_rows]
                # Consolidate duplicate boxless placeholder row for this person on this photo
                conn.execute(
                    """
                    DELETE FROM media_people
                    WHERE media_id = ? AND person_id = ? AND id != ?
                      AND (rect_w <= 0.005 OR rect_h <= 0.005);
                    """,
                    (mid, person_id, int(face_id)),
                )
                conn.execute(
                    """
                    UPDATE people
                    SET confirmed_count = (
                        SELECT COUNT(*) FROM media_people WHERE person_id = people.id AND state = 1
                    )
                    WHERE id = ?;
                    """,
                    (person_id,),
                )
                if old_pid and old_pid != person_id:
                    conn.execute(
                        """
                        UPDATE people
                        SET confirmed_count = (
                            SELECT COUNT(*) FROM media_people WHERE person_id = people.id AND state = 1
                        )
                        WHERE id = ?;
                        """,
                        (old_pid,),
                    )
                conn.execute("UPDATE media SET metadata_dirty = 1 WHERE id = ?;", (mid,))
                conn.commit()
                if row["embedding"]:
                    self.recalculate_person_centroid(person_id)
        if removed_placeholder_ids:
            self._invalidate_face_thumbnails_direct(removed_placeholder_ids)
        if old_pid and old_pid != person_id:
            self.refresh_person_avatar(old_pid, invalidated_face_ids=[int(face_id)])
        self.refresh_person_avatar(person_id)
        return person_id

    def dismiss_face_region(self, face_id: int) -> None:
        """Dismisses a false-positive detected face box (sets state = -1) and marks media dirty."""
        old_pid: int | None = None
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            row = conn.execute("SELECT media_id, person_id FROM media_people WHERE id = ?;", (int(face_id),)).fetchone()
            if row and row["person_id"]:
                old_pid = int(row["person_id"])
            conn.execute("UPDATE media_people SET state = -1, person_id = NULL, suggested_person_id = NULL WHERE id = ?;", (int(face_id),))
            if old_pid:
                conn.execute(
                    """
                    UPDATE people
                    SET confirmed_count = (
                        SELECT COUNT(*) FROM media_people WHERE person_id = people.id AND state = 1
                    )
                    WHERE id = ?;
                    """,
                    (old_pid,),
                )
            if row:
                conn.execute("UPDATE media SET metadata_dirty = 1 WHERE id = ?;", (int(row["media_id"]),))
            conn.commit()
        self._invalidate_face_thumbnails_direct([int(face_id)])
        if old_pid:
            self.refresh_person_avatar(old_pid, invalidated_face_ids=[int(face_id)])

    def remove_face_assignment(self, face_id: int) -> None:
        """Removes person assignment from a face region. If manual, deletes it; if detected, resets to state = 0."""
        pid = None
        is_manual = False
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            row = conn.execute(
                "SELECT media_id, person_id, is_manual FROM media_people WHERE id = ?;", (int(face_id),)
            ).fetchone()
            if not row:
                return
            pid = row["person_id"]
            is_manual = bool(row["is_manual"])
            mid = int(row["media_id"])
            if is_manual:
                conn.execute("DELETE FROM media_people WHERE id = ?;", (int(face_id),))
            else:
                conn.execute(
                    "UPDATE media_people SET person_id = NULL, state = 0, suggested_person_id = NULL WHERE id = ?;",
                    (int(face_id),),
                )
            if pid is not None:
                conn.execute(
                    """
                    UPDATE people
                    SET confirmed_count = (
                        SELECT COUNT(*) FROM media_people WHERE person_id = people.id AND state = 1
                    )
                    WHERE id = ?;
                    """,
                    (int(pid),),
                )
                self.recalculate_person_centroid(int(pid))
            conn.execute("UPDATE media SET metadata_dirty = 1 WHERE id = ?;", (mid,))
            conn.commit()
        if is_manual:
            self._invalidate_face_thumbnails_direct([int(face_id)])
        if pid is not None:
            self.refresh_person_avatar(int(pid), invalidated_face_ids=[int(face_id)])

    def update_face_rect(
        self,
        media_people_id: int,
        rect: tuple[float, float, float, float],
        thumb_repo: Any | None = None,
    ) -> bool:
        """
        Updates the normalized bounding box (rect_x, rect_y, rect_w, rect_h) for a face,
        regenerates the face thumbnail cache, and marks media dirty for metadata sync.
        """
        rx, ry, rw, rh = rect
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            row = conn.execute(
                """
                SELECT mp.media_id, mp.person_id, m.filename, f.path AS folder_path
                FROM media_people mp
                JOIN media m ON m.id = mp.media_id
                JOIN folders f ON f.id = m.folder_id
                WHERE mp.id = ?;
                """,
                (int(media_people_id),),
            ).fetchone()
            if not row:
                return False
            conn.execute(
                """
                UPDATE media_people
                SET rect_x = ?, rect_y = ?, rect_w = ?, rect_h = ?, is_manual = 1
                WHERE id = ?;
                """,
                (rx, ry, rw, rh, int(media_people_id)),
            )
            media_id = int(row["media_id"])
            conn.execute("UPDATE media SET metadata_dirty = 1 WHERE id = ?;", (media_id,))
            conn.commit()

        # Regenerate face crop outside write lock
        try:
            full_path = os.path.join(str(row["folder_path"]), str(row["filename"]))
            blob = self.extract_face_crop_blob(full_path, rect)
            if blob:
                if thumb_repo is not None:
                    thumb_repo.put_face_thumbnail(int(media_people_id), blob)
                else:
                    self._put_face_thumbnail_direct(int(media_people_id), blob)
            else:
                self._invalidate_face_thumbnails_direct([int(media_people_id)])
        except Exception:
            self._invalidate_face_thumbnails_direct([int(media_people_id)])
        if row["person_id"] is not None:
            self.refresh_person_avatar(int(row["person_id"]), updated_face_id=int(media_people_id))
        return True

    def set_person_display_picture_from_face(self, person_id: int, face_id: int) -> bool:
        """Sets people.thumbnail_blob for person_id from a specific media_people face record."""
        # 1. Prefer existing face crop thumbnail directly (matches the upright UI face crop)
        blob = self.get_face_thumbnail(int(face_id))

        if not blob:
            conn_read = self.engine.catalog_conn()
            row = conn_read.execute(
                """
                SELECT m.id AS media_id, m.filename, m.orientation, f.path AS folder_path,
                       mp.rect_x, mp.rect_y, mp.rect_w, mp.rect_h
                FROM media_people mp
                JOIN media m ON m.id = mp.media_id
                JOIN folders f ON f.id = m.folder_id
                WHERE mp.id = ?
                LIMIT 1;
                """,
                (int(face_id),),
            ).fetchone()
            if not row:
                return False
            full_path = os.path.join(str(row["folder_path"]), str(row["filename"]))
            rect = (
                float(row["rect_x"] or 0.0),
                float(row["rect_y"] or 0.0),
                float(row["rect_w"] or 0.0),
                float(row["rect_h"] or 0.0),
            )
            orient = int(row["orientation"] or 1)
            blob = self.extract_face_crop_blob(full_path, rect, orientation=orient)

        if not blob:
            return False
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute(
                "UPDATE people SET thumbnail_blob = ?, avatar_media_people_id = ? WHERE id = ?;",
                (blob, int(face_id), int(person_id)),
            )
            conn.commit()
        return True

    def reject_face_suggestion(
        self, media_people_id: int, rejected_person_id: int | None = None
    ) -> None:
        """
        Rejects a suggested match for a face.
        Records (media_people_id, rejected_person_id) into face_rejected_suggestions / face_rejections
        to prevent future centroid updates from re-suggesting this person, and resets suggestion.
        """
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            if rejected_person_id is None:
                row = conn.execute(
                    "SELECT suggested_person_id FROM media_people WHERE id = ?;",
                    (media_people_id,),
                ).fetchone()
                if row and row["suggested_person_id"] is not None:
                    rejected_person_id = int(row["suggested_person_id"])

            conn.execute(
                """
                UPDATE media_people
                SET suggested_person_id = NULL, similarity_score = NULL, state = 0
                WHERE id = ?;
                """,
                (media_people_id,),
            )

            if rejected_person_id is not None:
                conn.execute(
                    "INSERT OR IGNORE INTO face_rejected_suggestions (face_id, person_id) VALUES (?, ?);",
                    (media_people_id, rejected_person_id),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO face_rejections (face_id, person_id) VALUES (?, ?);",
                    (media_people_id, rejected_person_id),
                )
            conn.commit()

    def add_face_rejection(self, media_people_id: int, person_id: int) -> None:
        """
        Records that media_people_id should never be suggested as person_id.
        """
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute(
                "INSERT OR IGNORE INTO face_rejected_suggestions (face_id, person_id) VALUES (?, ?);",
                (media_people_id, person_id),
            )
            conn.execute(
                "INSERT OR IGNORE INTO face_rejections (face_id, person_id) VALUES (?, ?);",
                (media_people_id, person_id),
            )
            conn.commit()

    def get_face_rejections(self) -> dict[int, set[int]]:
        """Returns mapping {media_people_id: set_of_rejected_person_ids}."""
        conn = self.engine.catalog_conn()
        rows = conn.execute(
            "SELECT face_id, person_id FROM face_rejections;"
        ).fetchall()
        rejections: dict[int, set[int]] = {}
        for r in rows:
            fid = int(r["face_id"])
            pid = int(r["person_id"])
            rejections.setdefault(fid, set()).add(pid)
        return rejections

    def mark_not_a_face(self, media_people_ids: Sequence[int] | int) -> None:
        """
        Marks face regions as 'Not a face / Delete region' (state = -1).
        Permanently excluded from suggestions and never re-detected.
        """
        if isinstance(media_people_ids, int):
            ids: Sequence[int] = [media_people_ids]
        else:
            ids = media_people_ids
        if not ids:
            return
        affected_pids: set[int] = set()
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            for chunk in _iter_id_chunks(ids, 900):
                placeholders = ",".join("?" for _ in chunk)
                p_rows = conn.execute(
                    f"SELECT DISTINCT person_id FROM media_people WHERE id IN ({placeholders}) AND person_id IS NOT NULL;",
                    list(chunk),
                ).fetchall()
                for pr in p_rows:
                    if pr["person_id"]:
                        affected_pids.add(int(pr["person_id"]))
                conn.execute(
                    f"""
                    UPDATE media_people
                    SET state = -1, person_id = NULL, suggested_person_id = NULL, similarity_score = NULL
                    WHERE id IN ({placeholders});
                    """,
                    list(chunk),
                )
            for pid in affected_pids:
                conn.execute(
                    """
                    UPDATE people
                    SET confirmed_count = (
                        SELECT COUNT(*) FROM media_people WHERE person_id = people.id AND state = 1
                    )
                    WHERE id = ?;
                    """,
                    (pid,),
                )
            conn.commit()
        self._invalidate_face_thumbnails_direct(ids)
        for pid in affected_pids:
            self.refresh_person_avatar(pid, invalidated_face_ids=ids)

    def purge_unconfirmed_suggestions(self) -> int:
        """
        Deletes all unconfirmed face detections (state != 1) from media_people,
        resets face_scan_version = 0 for media so background detector will re-scan,
        and invalidates corresponding face thumbnail cache.
        Returns count of purged records.
        """
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            rows = conn.execute("SELECT id FROM media_people WHERE state != 1;").fetchall()
            purged_ids = [int(r["id"]) for r in rows]
            conn.execute("DELETE FROM media_people WHERE state != 1;")
            conn.execute("UPDATE media SET face_scan_version = 0 WHERE face_scan_version != 0;")
            conn.commit()

        if purged_ids:
            try:
                ThumbnailRepository(self.engine).invalidate_face_thumbnails(purged_ids)
            except Exception:
                pass

        return len(purged_ids)

    def recalculate_signatures_and_rematch(
        self, min_match_score: float = 0.52, high_match_score: float = 0.65
    ) -> int:
        """
        Fast in-memory recalculation (<1s):
        1. Loads all confirmed embeddings (state = 1) and computes (sub-)centroids for each person.
        2. Updates people.centroid_embedding in SQLite.
        3. Loads all unassigned embeddings (state = 0).
        4. Runs FaceVectorMatcher.match_unassigned_faces with negative rejection filtering.
        5. Writes back suggested matches to media_people.
        Returns count of updated suggestions.
        """
        import numpy as np
        from src.core.face_matcher import (
            FaceVectorMatcher,
            deserialize_embedding,
            normalize_vector,
            serialize_embedding,
        )

        conn = self.engine.catalog_conn()
        confirmed_rows = conn.execute(
            """
            SELECT person_id, embedding
            FROM media_people
            WHERE state = 1 AND person_id IS NOT NULL AND embedding IS NOT NULL;
            """
        ).fetchall()

        person_vectors: dict[int, list[np.ndarray]] = {}
        for r in confirmed_rows:
            pid = int(r["person_id"])
            blob = r["embedding"]
            if blob:
                vec = deserialize_embedding(bytes(blob))
                person_vectors.setdefault(pid, []).append(vec)

        centroids_dict: dict[int, np.ndarray] = {}
        with self.engine.write_lock:
            for pid, vecs in person_vectors.items():
                if len(vecs) >= 10:
                    sub_c = FaceVectorMatcher.compute_sub_centroids(np.stack(vecs, axis=0), max_k=5)
                else:
                    sub_c = normalize_vector(np.mean(np.stack(vecs, axis=0), axis=0, keepdims=True))
                centroids_dict[pid] = sub_c
                conn.execute(
                    "UPDATE people SET centroid_embedding = ? WHERE id = ?;",
                    (serialize_embedding(sub_c), pid),
                )
            conn.commit()

        if not centroids_dict:
            return 0

        unassigned_rows = conn.execute(
            "SELECT id, embedding FROM media_people WHERE state = 0 AND embedding IS NOT NULL;"
        ).fetchall()
        if not unassigned_rows:
            return 0

        face_ids = [int(r["id"]) for r in unassigned_rows]
        face_matrix = np.vstack([deserialize_embedding(bytes(r["embedding"])) for r in unassigned_rows])
        rejected_pairs = self.get_rejected_face_pairs()

        suggestions = FaceVectorMatcher.match_unassigned_faces(
            face_ids=face_ids,
            face_matrix=face_matrix,
            person_ids=list(centroids_dict.keys()),
            centroid_matrix=centroids_dict,
            rejected_pairs=rejected_pairs,
            high_thresh=high_match_score,
            low_thresh=min_match_score,
            top_k=1,
            assume_normalized=True,
        )

        match_triplets = [(s.face_id, s.person_id, s.score) for s in suggestions]
        with self.engine.write_lock:
            conn.execute(
                "UPDATE media_people SET suggested_person_id = NULL, similarity_score = NULL WHERE state = 0;"
            )
            conn.commit()

        self.update_suggested_matches(match_triplets)
        return len(match_triplets)

    def ignore_face(self, media_people_ids: Sequence[int] | int) -> None:
        """
        Marks face regions as ignored / 'Don't show again' (state = -2).
        These will never appear in suggestions or face detection workflows again.
        """
        if isinstance(media_people_ids, int):
            ids: Sequence[int] = [media_people_ids]
        else:
            ids = media_people_ids
        if not ids:
            return
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            for chunk in _iter_id_chunks(ids, 900):
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"""
                    UPDATE media_people
                    SET state = -2, suggested_person_id = NULL, similarity_score = NULL
                    WHERE id IN ({placeholders});
                    """,
                    list(chunk),
                )
            conn.commit()

    def mark_face_scan_failed(self, media_ids: Sequence[int] | int) -> None:
        """
        Marks corrupted or un-decodable files with face_scan_version = -1 so background
        face detectors will not repeatedly crash or re-attempt poison-pill files.
        """
        ids = [media_ids] if isinstance(media_ids, int) else list(media_ids)
        if not ids:
            return
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            for chunk in _iter_id_chunks(ids, 900):
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"UPDATE media SET face_scan_version = -1 WHERE id IN ({placeholders});",
                    list(chunk),
                )
            conn.commit()

    def update_face_scan_version(self, media_ids: Sequence[int] | int, version: int = 1) -> None:
        """Updates face_scan_version for media records."""
        ids = [media_ids] if isinstance(media_ids, int) else list(media_ids)
        if not ids:
            return
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            for chunk in _iter_id_chunks(ids, 900):
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"UPDATE media SET face_scan_version = ? WHERE id IN ({placeholders});",
                    [version] + list(chunk),
                )
            conn.commit()

    def get_rejected_face_pairs(self) -> set[tuple[int, int]]:
        """
        Returns all (face_id, person_id) pairs that have been explicitly rejected or
        where the person is already confirmed on the same photo (preventing duplicate person tags).
        Used by FaceVectorMatcher to mask re-suggestion scores.
        """
        conn = self.engine.catalog_conn()
        rows = conn.execute(
            "SELECT face_id, person_id FROM face_rejected_suggestions;"
        ).fetchall()
        pairs = {(int(r["face_id"]), int(r["person_id"])) for r in rows}

        # Also prevent suggesting a person who is already confirmed on the same media
        confirmed_on_media = conn.execute(
            """
            SELECT mp_unassigned.id AS face_id, mp_confirmed.person_id
            FROM media_people mp_unassigned
            JOIN media_people mp_confirmed
              ON mp_confirmed.media_id = mp_unassigned.media_id
             AND mp_confirmed.state = 1
             AND mp_confirmed.person_id IS NOT NULL
            WHERE mp_unassigned.state = 0;
            """
        ).fetchall()
        for r in confirmed_on_media:
            pairs.add((int(r["face_id"]), int(r["person_id"])))

        return pairs

    def update_suggested_matches(
        self, suggestions: Sequence[tuple[int, int, float]]
    ) -> None:
        """
        Batches proposed face match suggestions: list of (media_people_id, suggested_person_id, confidence).
        Only updates rows currently in state = 0.
        """
        if not suggestions:
            return
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.executemany(
                """
                UPDATE media_people
                SET suggested_person_id = ?,
                    similarity_score = ?
                WHERE id = ? AND state = 0 AND person_id IS NULL;
                """,
                [(pid, conf, mpid) for mpid, pid, conf in suggestions],
            )
            conn.commit()

    def update_face_suggestion(
        self, face_id: int, person_id: int | None, confidence: float | None
    ) -> None:
        """Updates a single face's suggested match and similarity score."""
        if person_id is not None and confidence is not None:
            self.update_suggested_matches([(face_id, person_id, confidence)])

    def get_pending_face_suggestions(
        self,
        limit: int = 500,
        min_confidence: float = 0.0,
        person_id: int | None = None,
        include_unnamed: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Returns unconfirmed face suggestions for the 'Batch people tag' view, sorted by confidence.
        Supports filtering by specific suggested person_id, or returning all suggestions + unnamed faces.
        """
        conn = self.engine.catalog_conn()
        priv_clause = self._private_folder_clause("f")
        if person_id is not None:
            rows = conn.execute(
                f"""
                SELECT mp.id, mp.id AS face_id, mp.media_id, mp.rect_x, mp.rect_y, mp.rect_w, mp.rect_h,
                       COALESCE(mp.similarity_score, 0.0) AS confidence,
                       mp.suggested_person_id, p.name AS suggested_name, p.thumbnail_blob AS person_thumb,
                       m.filename, f.path AS folder_path
                FROM media_people mp
                JOIN people p ON p.id = mp.suggested_person_id
                JOIN media m ON m.id = mp.media_id
                JOIN folders f ON f.id = m.folder_id
                WHERE mp.state = 0 AND mp.suggested_person_id = ?
                  AND COALESCE(mp.similarity_score, 0.0) >= ?
                  AND COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
                ORDER BY COALESCE(mp.similarity_score, 0.0) DESC
                LIMIT ?;
                """,
                (int(person_id), float(min_confidence), int(limit)),
            ).fetchall()
        elif include_unnamed:
            rows = conn.execute(
                f"""
                SELECT mp.id, mp.id AS face_id, mp.media_id, mp.rect_x, mp.rect_y, mp.rect_w, mp.rect_h,
                       COALESCE(mp.similarity_score, 0.0) AS confidence,
                       mp.suggested_person_id, p.name AS suggested_name, p.thumbnail_blob AS person_thumb,
                       m.filename, f.path AS folder_path
                FROM media_people mp
                LEFT JOIN people p ON p.id = mp.suggested_person_id
                JOIN media m ON m.id = mp.media_id
                JOIN folders f ON f.id = m.folder_id
                WHERE mp.state = 0
                  AND (mp.suggested_person_id IS NULL OR COALESCE(mp.similarity_score, 0.0) >= ?)
                  AND COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
                ORDER BY (mp.suggested_person_id IS NULL) ASC, COALESCE(mp.similarity_score, 0.0) DESC
                LIMIT ?;
                """,
                (float(min_confidence), int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT mp.id, mp.id AS face_id, mp.media_id, mp.rect_x, mp.rect_y, mp.rect_w, mp.rect_h,
                       COALESCE(mp.similarity_score, 0.0) AS confidence,
                       mp.suggested_person_id, p.name AS suggested_name, p.thumbnail_blob AS person_thumb,
                       m.filename, f.path AS folder_path
                FROM media_people mp
                JOIN people p ON p.id = mp.suggested_person_id
                JOIN media m ON m.id = mp.media_id
                JOIN folders f ON f.id = m.folder_id
                WHERE mp.state = 0 AND mp.suggested_person_id IS NOT NULL
                  AND COALESCE(mp.similarity_score, 0.0) >= ?
                  AND COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
                ORDER BY COALESCE(mp.similarity_score, 0.0) DESC
                LIMIT ?;
                """,
                (float(min_confidence), int(limit)),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_confirmed_faces(
        self, limit: int = 2000, person_id: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Returns confirmed face records (state = 1) for the 'Batch people tag' view.
        Supports filtering by specific person_id or returning all confirmed faces.
        """
        conn = self.engine.catalog_conn()
        priv_clause = self._private_folder_clause("f")
        if person_id is not None:
            rows = conn.execute(
                f"""
                SELECT mp.id, mp.id AS face_id, mp.media_id, mp.rect_x, mp.rect_y, mp.rect_w, mp.rect_h,
                       1.0 AS confidence, mp.person_id AS suggested_person_id, p.name AS suggested_name,
                       p.thumbnail_blob AS person_thumb, m.filename, f.path AS folder_path, 1 AS is_confirmed
                FROM media_people mp
                JOIN people p ON p.id = mp.person_id
                JOIN media m ON m.id = mp.media_id
                JOIN folders f ON f.id = m.folder_id
                WHERE mp.state = 1 AND mp.person_id = ?
                  AND COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
                ORDER BY m.id DESC
                LIMIT ?;
                """,
                (int(person_id), int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT mp.id, mp.id AS face_id, mp.media_id, mp.rect_x, mp.rect_y, mp.rect_w, mp.rect_h,
                       1.0 AS confidence, mp.person_id AS suggested_person_id, p.name AS suggested_name,
                       p.thumbnail_blob AS person_thumb, m.filename, f.path AS folder_path, 1 AS is_confirmed
                FROM media_people mp
                JOIN people p ON p.id = mp.person_id
                JOIN media m ON m.id = mp.media_id
                JOIN folders f ON f.id = m.folder_id
                WHERE mp.state = 1
                  AND COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
                ORDER BY p.name COLLATE NOCASE ASC, m.id DESC
                LIMIT ?;
                """,
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_face_tag_outliers(
        self,
        min_similarity_threshold: float = 0.38,
        limit: int = 500,
        include_no_face_boxes: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Scans confirmed face tags (state = 1) and detects potential errors:
        1. Confirmed face box with missing embedding (box drawn on non-face region) - optional
        2. Low cosine similarity (< threshold) to assigned person's centroid (likely wrong face)
        3. Significantly higher similarity to another known person (likely misidentified person)
        Returns structured records ready for BatchPeopleView review.
        """
        import numpy as np
        from src.core.face_matcher import deserialize_embedding

        conn = self.engine.catalog_conn()
        # 1. Fetch people centroids
        p_rows = conn.execute(
            "SELECT id, name, centroid_embedding FROM people WHERE centroid_embedding IS NOT NULL;"
        ).fetchall()
        if not p_rows:
            return []

        p_centroids: dict[int, np.ndarray] = {}
        p_names: dict[int, str] = {}
        p_ids_list: list[int] = []
        c_matrix_list: list[np.ndarray] = []

        for pr in p_rows:
            pid = int(pr["id"])
            p_names[pid] = str(pr["name"])
            try:
                vec = deserialize_embedding(pr["centroid_embedding"])
                if vec is not None and vec.size == 512:
                    norm = float(np.linalg.norm(vec))
                    if norm > 0:
                        vec = vec / norm
                    p_centroids[pid] = vec
                    p_ids_list.append(pid)
                    c_matrix_list.append(vec)
            except Exception:
                continue

        if not c_matrix_list:
            return []

        c_matrix = np.stack(c_matrix_list, axis=0)  # (N_people, 512)

        # 2. Fetch confirmed faces with boxes (excluding already human-verified outliers)
        priv_clause = self._private_folder_clause("f")
        face_rows = conn.execute(
            f"""
            SELECT mp.id AS face_id, mp.media_id, mp.person_id, p.name AS person_name,
                   mp.rect_x, mp.rect_y, mp.rect_w, mp.rect_h, mp.embedding,
                   m.filename, f.path AS folder_path
            FROM media_people mp
            JOIN people p ON p.id = mp.person_id
            JOIN media m ON m.id = mp.media_id
            JOIN folders f ON f.id = m.folder_id
            WHERE mp.state = 1 AND mp.rect_w > 0.005
              AND (mp.similarity_score IS NULL OR mp.similarity_score < 0.99)
              AND COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
            ORDER BY m.id DESC
            LIMIT 4000;
            """
        ).fetchall()

        outliers: list[dict[str, Any]] = []

        for fr in face_rows:
            pid = int(fr["person_id"])
            pname = str(fr["person_name"])
            emb_blob = fr["embedding"]

            # Case A: Missing embedding on non-zero box (optional to show/hide)
            if not emb_blob:
                if include_no_face_boxes:
                    outliers.append({
                        "id": fr["face_id"],
                        "face_id": fr["face_id"],
                        "media_id": fr["media_id"],
                        "person_id": pid,
                        "person_name": pname,
                        "suggested_person_id": pid,
                        "suggested_name": f"{pname} (No face in box)",
                        "rect_x": fr["rect_x"],
                        "rect_y": fr["rect_y"],
                        "rect_w": fr["rect_w"],
                        "rect_h": fr["rect_h"],
                        "confidence": 0.0,
                        "filename": fr["filename"],
                        "folder_path": fr["folder_path"],
                        "is_confirmed": True,
                        "outlier_reason": "No face detected in box",
                    })
                    if len(outliers) >= limit:
                        break
                continue

            try:
                f_vec = deserialize_embedding(emb_blob)
                if f_vec is None or f_vec.size != 512:
                    continue
                norm = float(np.linalg.norm(f_vec))
                if norm > 0:
                    f_vec = f_vec / norm
            except Exception:
                continue

            # Compare to assigned centroid
            own_centroid = p_centroids.get(pid)
            own_sim = float(np.dot(f_vec, own_centroid)) if own_centroid is not None else 1.0

            # Compare to all centroids
            all_sims = np.dot(c_matrix, f_vec)  # (N_people,)
            best_idx = int(np.argmax(all_sims))
            best_sim = float(all_sims[best_idx])
            best_pid = p_ids_list[best_idx]
            best_pname = p_names.get(best_pid, "Unknown")

            is_outlier = False
            reason = ""
            disp_name = pname
            disp_conf = own_sim

            if own_centroid is not None and own_sim < min_similarity_threshold:
                is_outlier = True
                disp_conf = own_sim
                if best_pid != pid and best_sim >= 0.58 and best_sim >= (own_sim + 0.15):
                    reason = f"Looks like {best_pname} ({int(best_sim*100)}%)"
                    disp_name = f"{pname} → {best_pname}?"
                    disp_conf = best_sim
                else:
                    reason = f"Low match ({int(own_sim*100)}%)"
                    disp_name = f"{pname} (Low match)"
            elif best_pid != pid and best_sim >= 0.65 and best_sim >= (own_sim + 0.20):
                is_outlier = True
                reason = f"Looks like {best_pname} ({int(best_sim*100)}%)"
                disp_name = f"{pname} → {best_pname}?"
                disp_conf = best_sim

            if is_outlier:
                outliers.append({
                    "id": fr["face_id"],
                    "face_id": fr["face_id"],
                    "media_id": fr["media_id"],
                    "person_id": pid,
                    "person_name": pname,
                    "suggested_person_id": pid,
                    "suggested_name": disp_name,
                    "swap_person_id": best_pid if "Looks like" in reason else None,
                    "swap_person_name": best_pname if "Looks like" in reason else None,
                    "rect_x": fr["rect_x"],
                    "rect_y": fr["rect_y"],
                    "rect_w": fr["rect_w"],
                    "rect_h": fr["rect_h"],
                    "confidence": disp_conf,
                    "filename": fr["filename"],
                    "folder_path": fr["folder_path"],
                    "is_confirmed": True,
                    "outlier_reason": reason,
                })
                if len(outliers) >= limit:
                    break

        return outliers

    def get_boxless_people_matches(self, limit: int = 500) -> list[dict[str, Any]]:
        """
        Finds photos where a person is tagged without a face rectangle (state = 1, rect_w <= 0.005)
        paired with unassigned detected face boxes (state = 0, rect_w > 0.005) on the same photo.
        """
        conn = self.engine.catalog_conn()
        priv_clause = self._private_folder_clause("f")
        rows = conn.execute(
            f"""
            SELECT mp.id AS boxless_id, mp.media_id, mp.person_id, p.name AS person_name,
                   mp2.id AS face_id, mp2.rect_x, mp2.rect_y, mp2.rect_w, mp2.rect_h,
                   COALESCE(mp2.similarity_score, 0.0) AS confidence,
                   m.filename, f.path AS folder_path
            FROM media_people mp
            JOIN people p ON p.id = mp.person_id
            JOIN media_people mp2 ON mp2.media_id = mp.media_id AND mp2.state = 0 AND mp2.rect_w > 0.005
            JOIN media m ON m.id = mp.media_id
            JOIN folders f ON f.id = m.folder_id
            WHERE mp.state = 1 AND (mp.rect_w IS NULL OR mp.rect_w <= 0.005)
              AND COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
            ORDER BY p.name COLLATE NOCASE ASC, m.id DESC
            LIMIT ?;
            """,
            (int(limit),),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for r in rows:
            results.append({
                "id": r["face_id"],
                "face_id": r["face_id"],
                "boxless_id": r["boxless_id"],
                "media_id": r["media_id"],
                "person_id": r["person_id"],
                "person_name": r["person_name"],
                "suggested_person_id": r["person_id"],
                "suggested_name": f"{r['person_name']} (Missing Box)",
                "rect_x": r["rect_x"],
                "rect_y": r["rect_y"],
                "rect_w": r["rect_w"],
                "rect_h": r["rect_h"],
                "confidence": r["confidence"],
                "filename": r["filename"],
                "folder_path": r["folder_path"],
                "is_confirmed": False,
                "is_boxless_match": True,
            })
        return results

    def auto_link_unambiguous_boxless_faces(self) -> int:
        """
        Auto-links photos that have EXACTLY 1 boxless confirmed person tag and EXACTLY 1 unassigned face box.
        Consolidates the boxless row and assigns the face box to that person in one transaction.
        Returns the number of faces linked.
        """
        conn = self.engine.catalog_conn()
        priv_clause = self._private_folder_clause("f")
        rows = conn.execute(
            f"""
            SELECT mp.media_id, mp.id AS boxless_id, mp.person_id, p.name AS person_name,
                   mp2.id AS face_id
            FROM media_people mp
            JOIN people p ON p.id = mp.person_id
            JOIN media_people mp2 ON mp2.media_id = mp.media_id AND mp2.state = 0 AND mp2.rect_w > 0.005
            JOIN media m ON m.id = mp.media_id
            JOIN folders f ON f.id = m.folder_id
            WHERE mp.state = 1 AND (mp.rect_w IS NULL OR mp.rect_w <= 0.005)
              AND COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
            GROUP BY mp.media_id
            HAVING COUNT(DISTINCT mp.id) = 1 AND COUNT(DISTINCT mp2.id) = 1;
            """
        ).fetchall()
        if not rows:
            return 0

        linked_count = 0
        for r in rows:
            face_id = int(r["face_id"])
            p_name = str(r["person_name"])
            self.assign_face_to_person(face_id, p_name)
            linked_count += 1
        return linked_count

    def get_places_hierarchy(self) -> list[dict[str, Any]]:
        """
        Retrieves aggregate location counts (country, state, city) for media,
        normalizing admin subdivision codes (e.g. UK ENG -> England) and stripping
        numeric administrative codes (e.g. '01', '07').
        """
        conn = self.engine.catalog_conn()
        priv_clause = self._private_folder_clause("f")
        rows = conn.execute(
            f"""
            SELECT m.location_country, m.location_state, m.location_city, COUNT(*) as cnt
            FROM media m
            JOIN folders f ON f.id = m.folder_id
            WHERE COALESCE(m.is_missing, 0) = 0 AND {priv_clause}
              AND (
                (m.location_country IS NOT NULL AND m.location_country != '')
                OR (m.location_state IS NOT NULL AND m.location_state != '')
                OR (m.location_city IS NOT NULL AND m.location_city != '')
              )
            GROUP BY m.location_country, m.location_state, m.location_city
            ORDER BY m.location_country, m.location_state, m.location_city;
            """
        ).fetchall()

        grouped: dict[tuple[str, str, str], int] = {}
        for r in rows:
            country = (r["location_country"] or "").strip()
            state = (r["location_state"] or "").strip()
            city = (r["location_city"] or "").strip()
            cnt = int(r["cnt"])

            if country in ("United Kingdom", "GB", "UK"):
                if state.upper() in ("ENG", "ENGLAND"):
                    state = "England"
                elif state.upper() in ("WLS", "WALES"):
                    state = "Wales"
                elif state.upper() in ("SCT", "SCOTLAND"):
                    state = "Scotland"
                elif state.upper() in ("NIR", "NORTHERN IRELAND"):
                    state = "Northern Ireland"

            if state.isdigit():
                state = ""

            key = (country, state, city)
            grouped[key] = grouped.get(key, 0) + cnt

        result = []
        for (country, state, city), count in sorted(grouped.items()):
            result.append({
                "country": country,
                "state": state,
                "city": city,
                "count": count,
            })
        return result

    def reset_face_scan_for_media(self, media_ids: Sequence[int]) -> int:
        """
        Resets face_scan_version = 0, deletes unassigned face suggestions (state != 1),
        and resets embeddings on confirmed faces so the media can be freshly re-scanned.
        """
        ids = list(media_ids)
        if not ids:
            return 0
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            count = 0
            for chunk in _iter_id_chunks(ids, 900):
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"UPDATE media SET face_scan_version = 0 WHERE id IN ({placeholders});",
                    chunk,
                )
                conn.execute(
                    f"DELETE FROM media_people WHERE media_id IN ({placeholders}) AND state != 1;",
                    chunk,
                )
                conn.execute(
                    f"UPDATE media_people SET embedding = NULL WHERE media_id IN ({placeholders}) AND state = 1;",
                    chunk,
                )
                count += len(chunk)
            conn.commit()
            return count

    def reset_unscanned_and_zero_face_media(self) -> int:
        """
        Resets face_scan_version = 0 for photos with 0 faces or missing embeddings,
        allowing a complete, non-destructive re-scan that preserves all confirmed tags.
        """
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE media
                SET face_scan_version = 0
                WHERE is_video = 0
                  AND (
                      face_scan_version < 2
                      OR NOT EXISTS (
                          SELECT 1 FROM media_people mp
                          WHERE mp.media_id = media.id AND mp.embedding IS NOT NULL
                      )
                  );
                """
            )
            count = cur.rowcount
            conn.commit()
            return max(0, count)

    # -------------------------------------------------------------------------
    # Non-Destructive Pixel Edit Backup & 30-Day Retention Policy
    # -------------------------------------------------------------------------
    def record_pixel_edit_backup(
        self, media_id: int, backup_path: str, edited_at: int | None = None
    ) -> None:
        """Records the stashed original file path and edit timestamp for 'Revert to original'."""
        ts = int(time.time()) if edited_at is None else int(edited_at)
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute(
                """
                UPDATE media
                SET original_backup_path = COALESCE(original_backup_path, ?),
                    edited_at = ?
                WHERE id = ?;
                """,
                (backup_path, ts, media_id),
            )
            conn.commit()

    def clear_pixel_edit_backup(self, media_id: int) -> str | None:
        """Clears and returns the original_backup_path after reverting to original."""
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            row = conn.execute(
                "SELECT original_backup_path FROM media WHERE id = ?;", (media_id,)
            ).fetchone()
            if row is None or not row["original_backup_path"]:
                return None
            backup_path = str(row["original_backup_path"])
            conn.execute(
                "UPDATE media SET original_backup_path = NULL, edited_at = NULL WHERE id = ?;",
                (media_id,),
            )
            conn.commit()
            return backup_path

    def prune_expired_backups(
        self, retention_days: int = 30, now_epoch: int | None = None
    ) -> list[str]:
        """
        Finds all original_backup_path entries older than retention_days (default 30 days),
        clears their backup columns in catalog.db, and returns the expired paths for disk removal.
        """
        now_ts = int(time.time()) if now_epoch is None else int(now_epoch)
        cutoff = now_ts - (retention_days * 86400)
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            rows = conn.execute(
                """
                SELECT id, original_backup_path
                FROM media
                WHERE edited_at IS NOT NULL
                  AND edited_at <= ?
                  AND original_backup_path IS NOT NULL;
                """,
                (cutoff,),
            ).fetchall()
            if not rows:
                return []
            expired_ids = [int(r["id"]) for r in rows]
            expired_paths = [str(r["original_backup_path"]) for r in rows]
            placeholders = ",".join("?" for _ in expired_ids)
            conn.execute(
                f"UPDATE media SET original_backup_path = NULL, edited_at = NULL WHERE id IN ({placeholders});",
                expired_ids,
            )
            conn.commit()
            return expired_paths

    # -------------------------------------------------------------------------
    # World Map R*Tree Spatial Queries & Right Dock Selection Summary
    # -------------------------------------------------------------------------
    @staticmethod
    def build_map_filter_clauses(
        year: int | None = None,
        person_id: int | None = None,
        min_rating: int | None = None,
        alias: str = "m",
    ) -> tuple[list[str], list[Any]]:
        """Builds SQL WHERE fragments and bind params for Map View filters (year, person, rating)."""
        clauses: list[str] = [f"COALESCE({alias}.is_missing, 0) = 0"]
        params: list[Any] = []
        if year is not None:
            start_ts, end_ts = _compute_date_bounds_utc(int(year))
            clauses.append(f"{alias}.date_taken BETWEEN ? AND ?")
            params.extend([start_ts, end_ts])
        if person_id is not None:
            clauses.append(
                f"EXISTS (SELECT 1 FROM media_people mp "
                f"WHERE mp.media_id = {alias}.id AND mp.person_id = ? AND mp.state = 1)"
            )
            params.append(int(person_id))
        if min_rating is not None:
            r_val = int(min_rating)
            if r_val == 0:
                clauses.append(f"COALESCE({alias}.rating, 0) = 0")
            elif r_val > 0:
                clauses.append(f"COALESCE({alias}.rating, 0) >= ?")
                params.append(r_val)
        return clauses, params

    def get_geotagged_years(self) -> list[int]:
        """Returns distinct years (descending) that have valid non-missing geotagged photos."""
        conn = self.engine.catalog_conn()
        rows = conn.execute(
            """
            SELECT DISTINCT strftime(
                '%Y',
                datetime(
                    CASE WHEN date_taken > 100000000000 THEN date_taken / 1000 ELSE date_taken END,
                    'unixepoch'
                )
            ) AS yr
            FROM media
            WHERE COALESCE(is_missing, 0) = 0
              AND lat IS NOT NULL AND lon IS NOT NULL
              AND lat BETWEEN -85.0 AND 85.0
              AND lon BETWEEN -180.0 AND 180.0
              AND NOT (ABS(lat) < 0.0001 AND ABS(lon) < 0.0001)
              AND date_taken IS NOT NULL AND date_taken != 0
            ORDER BY yr DESC;
            """
        ).fetchall()
        years: list[int] = []
        for r in rows:
            yr_str = r["yr"] if isinstance(r, dict) else r[0]
            if yr_str and yr_str.isdigit():
                yr = int(yr_str)
                if 1800 <= yr <= 2200:
                    years.append(yr)
        return years

    def query_map_clusters(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        grid_degrees: float | None = None,
        year: int | None = None,
        person_id: int | None = None,
        min_rating: int | None = None,
    ) -> list[dict[str, Any]]:
        """Queries media_rtree for geotagged items within the active map bounding box, optionally clustering in SQL."""
        conn = self.engine.catalog_conn()
        extra_clauses, extra_params = self.build_map_filter_clauses(
            year=year, person_id=person_id, min_rating=min_rating, alias="m"
        )
        extra_sql = (" AND " + " AND ".join(extra_clauses)) if extra_clauses else ""

        if grid_degrees is not None and grid_degrees > 0:
            sql_params: list[Any] = [min_lat, max_lat, min_lon, max_lon]
            sql_params.extend(extra_params)
            sql_params.extend([float(grid_degrees), float(grid_degrees)])
            rows = conn.execute(
                f"""
                SELECT
                    COUNT(*) AS count,
                    AVG(r.min_lat) AS lat,
                    AVG(r.min_lon) AS lon,
                    MIN(m.id) AS id,
                    GROUP_CONCAT(m.id) AS media_ids_str,
                    MAX(m.location_name) AS location_name
                FROM media_rtree r
                JOIN media m ON m.id = r.id
                WHERE r.max_lat >= ? AND r.min_lat <= ?
                  AND r.max_lon >= ? AND r.min_lon <= ?{extra_sql}
                GROUP BY CAST(FLOOR(r.min_lat / ?) AS INTEGER), CAST(FLOOR(r.min_lon / ?) AS INTEGER);
                """,
                sql_params,
            ).fetchall()
            results = []
            for r in rows:
                mids = [int(x) for x in str(r["media_ids_str"]).split(",") if x]
                loc = str(r["location_name"] or "").strip()
                lat = float(r["lat"])
                lon = float(r["lon"])
                label = loc if loc else f"{lat:.2f}°, {lon:.2f}°"
                results.append(
                    {
                        "id": int(r["id"]),
                        "lat": lat,
                        "lon": lon,
                        "count": int(r["count"]),
                        "media_ids": mids,
                        "location_name": loc,
                        "label": label,
                    }
                )
            results.sort(key=lambda c: c["count"], reverse=True)
            return results

        sql_params_flat: list[Any] = [min_lat, max_lat, min_lon, max_lon]
        sql_params_flat.extend(extra_params)
        rows = conn.execute(
            f"""
            SELECT m.id, m.filename, m.lat, m.lon, m.location_name, m.date_taken, m.rating
            FROM media_rtree r
            JOIN media m ON m.id = r.id
            WHERE r.max_lat >= ? AND r.min_lat <= ?
              AND r.max_lon >= ? AND r.min_lon <= ?{extra_sql};
            """,
            sql_params_flat,
        ).fetchall()
        return [dict(r) for r in rows]

    def _apply_map_scope_ids(
        self, conn: Any, scope_media_ids: Sequence[int] | None
    ) -> str:
        """Populates a connection-local temp table when scope_media_ids is active and returns an AND clause."""
        if scope_media_ids is None:
            return ""
        ids = [int(x) for x in scope_media_ids]
        if not ids:
            return " AND 1 = 0"
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS _map_scope_ids (id INTEGER PRIMARY KEY);")
        conn.execute("DELETE FROM _map_scope_ids;")
        conn.executemany(
            "INSERT OR IGNORE INTO _map_scope_ids (id) VALUES (?);",
            [(mid,) for mid in ids],
        )
        return " AND m.id IN (SELECT id FROM _map_scope_ids)"

    def query_map_overview_cells(
        self,
        year: int | None = None,
        person_id: int | None = None,
        min_rating: int | None = None,
        scope_media_ids: Sequence[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        High-speed overview query for World Map View (zero GROUP_CONCAT).
        Aggregates all geotagged items (optionally scoped to `scope_media_ids`) into ~1 km
        cells (ROUND(lat, 2), ROUND(lon, 2)) returning lightweight dicts with a single
        representative media ID and exact cell bounds.
        Member media_ids are resolved lazily via `query_map_cluster_media_ids()` only on click.
        """
        conn = self.engine.catalog_conn()
        extra_clauses, extra_params = self.build_map_filter_clauses(
            year=year, person_id=person_id, min_rating=min_rating, alias="m"
        )
        if not self.include_private:
            extra_clauses.append(
                "NOT EXISTS (SELECT 1 FROM folders f_priv WHERE f_priv.id = m.folder_id AND COALESCE(f_priv.is_private, 0) = 1)"
            )
        extra_clauses.append("COALESCE(m.is_missing, 0) = 0")
        extra_clauses.append("NOT (ABS(r.min_lat) < 0.0001 AND ABS(r.min_lon) < 0.0001)")
        scope_sql = self._apply_map_scope_ids(conn, scope_media_ids)
        extra_sql = (" AND " + " AND ".join(extra_clauses)) if extra_clauses else ""
        extra_sql += scope_sql

        rows = conn.execute(
            f"""
            SELECT
                COUNT(*) AS count,
                AVG(r.min_lat) AS lat,
                AVG(r.min_lon) AS lon,
                MIN(r.min_lat) AS min_lat,
                MAX(r.max_lat) AS max_lat,
                MIN(r.min_lon) AS min_lon,
                MAX(r.max_lon) AS max_lon,
                MIN(m.id) AS rep_media_id,
                MAX(m.mtime) AS rep_mtime,
                MAX(m.location_name) AS location_name
            FROM media_rtree r
            JOIN media m ON m.id = r.id
            WHERE r.min_lat BETWEEN -85.0 AND 85.0
              AND r.min_lon BETWEEN -180.0 AND 180.0{extra_sql}
            GROUP BY ROUND(r.min_lat, 2), ROUND(r.min_lon, 2);
            """,
            extra_params,
        ).fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            lat = float(r["lat"])
            lon = float(r["lon"])
            loc = str(r["location_name"] or "").strip()
            results.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "count": int(r["count"]),
                    "rep_media_id": int(r["rep_media_id"]),
                    "rep_mtime": int(r["rep_mtime"] or 0),
                    "min_lat": float(r["min_lat"]),
                    "max_lat": float(r["max_lat"]),
                    "min_lon": float(r["min_lon"]),
                    "max_lon": float(r["max_lon"]),
                    "label": loc if loc else f"{lat:.2f}°, {lon:.2f}°",
                }
            )
        results.sort(key=lambda c: c["count"], reverse=True)
        return results

    def query_map_cluster_media_ids(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        year: int | None = None,
        person_id: int | None = None,
        min_rating: int | None = None,
        scope_media_ids: Sequence[int] | None = None,
    ) -> list[int]:
        """Lazily resolves all media IDs inside a clicked cluster's bounding box via media_rtree."""
        conn = self.engine.catalog_conn()
        extra_clauses, extra_params = self.build_map_filter_clauses(
            year=year, person_id=person_id, min_rating=min_rating, alias="m"
        )
        if not self.include_private:
            extra_clauses.append(
                "NOT EXISTS (SELECT 1 FROM folders f_priv WHERE f_priv.id = m.folder_id AND COALESCE(f_priv.is_private, 0) = 1)"
            )
        extra_clauses.append("COALESCE(m.is_missing, 0) = 0")
        scope_sql = self._apply_map_scope_ids(conn, scope_media_ids)
        extra_sql = (" AND " + " AND ".join(extra_clauses)) if extra_clauses else ""
        extra_sql += scope_sql

        pad = 1e-4
        sql_params: list[Any] = [
            float(min_lat) - pad,
            float(max_lat) + pad,
            float(min_lon) - pad,
            float(max_lon) + pad,
        ]
        sql_params.extend(extra_params)
        rows = conn.execute(
            f"""
            SELECT m.id
            FROM media_rtree r
            JOIN media m ON m.id = r.id
            WHERE r.max_lat >= ? AND r.min_lat <= ?
              AND r.max_lon >= ? AND r.min_lon <= ?{extra_sql}
            ORDER BY COALESCE(m.date_taken, m.mtime, 0) DESC, m.id ASC;
            """,
            sql_params,
        ).fetchall()
        return [int(r[0]) for r in rows]

    def get_media_decode_info_batch(self, media_ids: Sequence[int]) -> list[dict[str, Any]]:
        """Returns minimal file path and decode metadata for the requested media IDs."""
        if not media_ids:
            return []
        conn = self.engine.catalog_conn()
        results: list[dict[str, Any]] = []
        for chunk in _iter_id_chunks(media_ids, 900):
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"""
                SELECT m.id, f.path AS folder_path, m.filename, m.mtime, m.is_video, m.orientation
                FROM media m
                JOIN folders f ON f.id = m.folder_id
                WHERE m.id IN ({placeholders});
                """,
                list(chunk),
            ).fetchall()
            results.extend(dict(r) for r in rows)
        return results

    def get_media_id_by_file_path(self, file_path: Path | str) -> int | None:
        """Finds media id by full file path."""
        p = Path(file_path).resolve()
        norm_f1 = str(p.parent).replace("\\", "/")
        norm_f2 = str(p.parent)
        filename = p.name
        conn = self.engine.catalog_conn()
        row = conn.execute(
            """
            SELECT m.id
            FROM media m
            JOIN folders f ON f.id = m.folder_id
            WHERE (f.path = ? OR f.path = ?) AND m.filename = ? COLLATE NOCASE;
            """,
            (norm_f1, norm_f2, filename),
        ).fetchone()
        return int(row[0]) if row else None

    def get_media_detail(self, media_id: int) -> dict[str, Any]:
        conn = self.engine.catalog_conn()
        row = conn.execute(
            """
            SELECT m.*, f.path AS folder_path, f.name AS folder_name, COALESCE(f.volume_offline, 0) AS volume_offline
            FROM media m
            JOIN folders f ON f.id = m.folder_id
            WHERE m.id = ?;
            """,
            (media_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Media {media_id} not found")
        detail = dict(row)
        detail["tags"] = [
            dict(r)
            for r in conn.execute(
                """
                SELECT t.id, t.name, t.full_path
                FROM tags t
                JOIN media_tags mt ON mt.tag_id = t.id
                WHERE mt.media_id = ?
                ORDER BY t.full_path COLLATE NOCASE;
                """,
                (media_id,),
            ).fetchall()
        ]
        # Query all non-rejected face regions (state >= 0: confirmed, unconfirmed suggestion, unknown)
        face_rows = conn.execute(
            """
            SELECT mp.id AS face_id,
                   mp.id AS region_id,
                   COALESCE(mp.person_id, mp.suggested_person_id) AS person_id,
                   p.name AS person_name,
                   p.thumbnail_blob AS person_thumbnail_blob,
                   mp.state,
                   mp.similarity_score AS confidence,
                   mp.rect_x, mp.rect_y, mp.rect_w, mp.rect_h,
                   mp.is_manual
            FROM media_people mp
            LEFT JOIN people p ON p.id = COALESCE(mp.person_id, mp.suggested_person_id)
            WHERE mp.media_id = ? AND mp.state >= 0
            ORDER BY mp.state DESC, mp.rect_x ASC;
            """,
            (media_id,),
        ).fetchall()

        face_ids = [int(r["face_id"]) for r in face_rows]
        thumb_repo = ThumbnailRepository(self.engine)
        face_thumbs = thumb_repo.get_face_thumbnails_batch(face_ids) if face_ids else {}

        all_regions: list[dict[str, Any]] = []
        for r in face_rows:
            fid = int(r["face_id"])
            pid = int(r["person_id"]) if r["person_id"] is not None else None
            pname = str(r["person_name"]) if r["person_name"] else None
            st = int(r["state"])
            rect = (
                float(r["rect_x"] or 0.0),
                float(r["rect_y"] or 0.0),
                float(r["rect_w"] or 0.0),
                float(r["rect_h"] or 0.0),
            )
            has_valid_rect = rect[2] > 0.005 and rect[3] > 0.005
            blob = (face_thumbs.get(fid) if has_valid_rect else None) or r["person_thumbnail_blob"]
            reg_dict = {
                "id": pid if pid is not None else fid,
                "region_id": fid,
                "face_id": fid,
                "person_id": pid,
                "person_name": pname,
                "name": pname or "",
                "state": st,
                "is_unconfirmed": bool(st == 0),
                "confidence": float(r["confidence"]) if r["confidence"] is not None else None,
                "rect": rect,
                "rect_x": rect[0],
                "rect_y": rect[1],
                "rect_w": rect[2],
                "rect_h": rect[3],
                "is_manual": bool(r["is_manual"]),
                "thumbnail_blob": blob,
            }
            all_regions.append(reg_dict)

        detail["face_regions"] = all_regions
        detail["confirmed_people"] = [reg for reg in all_regions if reg["state"] == 1]
        detail["unconfirmed_people"] = [reg for reg in all_regions if reg["state"] == 0 and reg["person_name"]]
        detail["unknown_people"] = [reg for reg in all_regions if reg["state"] == 0 and not reg["person_name"]]
        detail["people"] = all_regions

        # Video properties (Duration / Frame rate / Bitrate)
        detail["is_video"] = bool(detail.get("is_video"))
        dur_ms = detail.get("duration_ms")
        duration = float(dur_ms) / 1000.0 if dur_ms else 0.0
        fps = float(detail.get("video_fps") or 0.0)
        size_b = int(detail.get("size") or 0)
        detail["duration"] = duration
        detail["fps"] = fps
        detail["bitrate_kbps"] = (
            int((size_b * 8) / (duration * 1000.0)) if duration > 0 and size_b > 0 else 0
        )
        return detail

    def get_selection_summary(self, media_ids: Sequence[int]) -> dict[str, Any]:
        """Aggregates metadata, common tags, and people for single or multi-selection in Right Dock."""
        if not media_ids:
            return {"count": 0}
        if len(media_ids) == 1:
            detail = self.get_media_detail(media_ids[0])
            detail["count"] = 1
            return detail

        conn = self.engine.catalog_conn()
        lead_detail = self.get_media_detail(media_ids[0])
        total_cnt = 0
        total_size = 0
        min_rating: int | None = None
        max_rating: int | None = None
        min_flag: int | None = None
        max_flag: int | None = None
        min_fav: int | None = None
        max_fav: int | None = None
        min_folder_id: int | None = None
        max_folder_id: int | None = None

        for chunk in _iter_id_chunks(media_ids, 900):
            placeholders = ",".join("?" for _ in chunk)
            agg = conn.execute(
                f"""
                SELECT
                    COUNT(*) AS cnt,
                    SUM(size) AS total_size,
                    MIN(rating) AS min_rating,
                    MAX(rating) AS max_rating,
                    MIN(CASE WHEN COALESCE(is_rejected, 0) = 1 THEN -1 ELSE COALESCE(flag, 0) END) AS min_flag,
                    MAX(CASE WHEN COALESCE(is_rejected, 0) = 1 THEN -1 ELSE COALESCE(flag, 0) END) AS max_flag,
                    MIN(COALESCE(is_favourite, 0)) AS min_fav,
                    MAX(COALESCE(is_favourite, 0)) AS max_fav,
                    MIN(COALESCE(folder_id, -1)) AS min_folder_id,
                    MAX(COALESCE(folder_id, -1)) AS max_folder_id
                FROM media
                WHERE id IN ({placeholders});
                """,
                list(chunk),
            ).fetchone()
            if agg and agg["cnt"]:
                total_cnt += int(agg["cnt"])
                total_size += int(agg["total_size"] or 0)
                cur_min_r = int(agg["min_rating"]) if agg["min_rating"] is not None else None
                cur_max_r = int(agg["max_rating"]) if agg["max_rating"] is not None else None
                cur_min_f = int(agg["min_flag"]) if agg["min_flag"] is not None else None
                cur_max_f = int(agg["max_flag"]) if agg["max_flag"] is not None else None
                cur_min_fv = int(agg["min_fav"]) if agg["min_fav"] is not None else None
                cur_max_fv = int(agg["max_fav"]) if agg["max_fav"] is not None else None
                cur_min_fid = int(agg["min_folder_id"]) if agg["min_folder_id"] is not None else None
                cur_max_fid = int(agg["max_folder_id"]) if agg["max_folder_id"] is not None else None

                if cur_min_r is not None:
                    min_rating = cur_min_r if min_rating is None else min(min_rating, cur_min_r)
                if cur_max_r is not None:
                    max_rating = cur_max_r if max_rating is None else max(max_rating, cur_max_r)
                if cur_min_f is not None:
                    min_flag = cur_min_f if min_flag is None else min(min_flag, cur_min_f)
                if cur_max_f is not None:
                    max_flag = cur_max_f if max_flag is None else max(max_flag, cur_max_f)
                if cur_min_fv is not None:
                    min_fav = cur_min_fv if min_fav is None else min(min_fav, cur_min_fv)
                if cur_max_fv is not None:
                    max_fav = cur_max_fv if max_fav is None else max(max_fav, cur_max_fv)
                if cur_min_fid is not None:
                    min_folder_id = cur_min_fid if min_folder_id is None else min(min_folder_id, cur_min_fid)
                if cur_max_fid is not None:
                    max_folder_id = cur_max_fid if max_folder_id is None else max(max_folder_id, cur_max_fid)

        # Tags shared by ALL selected items
        tag_occurrence_counts: dict[int, int] = {}
        tag_details: dict[int, dict[str, Any]] = {}
        for chunk in _iter_id_chunks(media_ids, 900):
            placeholders = ",".join("?" for _ in chunk)
            chunk_tags = conn.execute(
                f"""
                SELECT t.id, t.name, t.full_path, COUNT(DISTINCT mt.media_id) AS chunk_cnt
                FROM tags t
                JOIN media_tags mt ON mt.tag_id = t.id
                WHERE mt.media_id IN ({placeholders})
                GROUP BY t.id;
                """,
                list(chunk),
            ).fetchall()
            for row in chunk_tags:
                tid = int(row["id"])
                tag_occurrence_counts[tid] = tag_occurrence_counts.get(tid, 0) + int(row["chunk_cnt"])
                if tid not in tag_details:
                    tag_details[tid] = {"id": tid, "name": str(row["name"]), "full_path": str(row["full_path"])}

        total_selected = len(media_ids)
        tags_list = [
            {
                "id": tid,
                "name": tag_details[tid]["name"],
                "full_path": tag_details[tid]["full_path"],
                "count": count,
                "is_shared_by_all": (count == total_selected),
            }
            for tid, count in tag_occurrence_counts.items()
        ]
        tags_list.sort(key=lambda t: (-t["count"], t["name"].lower()))

        # People appearing in ANY selected item
        people_seen: dict[int, dict[str, Any]] = {}
        for chunk in _iter_id_chunks(media_ids, 900):
            placeholders = ",".join("?" for _ in chunk)
            chunk_people = conn.execute(
                f"""
                SELECT DISTINCT p.id, p.name, p.thumbnail_blob
                FROM people p
                JOIN media_people mp ON mp.person_id = p.id
                WHERE mp.media_id IN ({placeholders}) AND mp.state = 1;
                """,
                list(chunk),
            ).fetchall()
            for row in chunk_people:
                pid = int(row["id"])
                if pid not in people_seen:
                    people_seen[pid] = dict(row)

        people_list = list(people_seen.values())
        people_list.sort(key=lambda p: str(p["name"]).lower())

        same_folder = (
            min_folder_id is not None
            and max_folder_id is not None
            and min_folder_id == max_folder_id
            and min_folder_id > 0
        )

        return {
            "count": total_cnt,
            "lead": lead_detail,
            "total_size": total_size,
            "common_rating": min_rating if min_rating == max_rating else None,
            "common_flag": min_flag if min_flag == max_flag else None,
            "common_favourite": min_fav if min_fav == max_fav else None,
            "folder_path": str(lead_detail.get("folder_path") or "") if same_folder else "",
            "folder_name": str(lead_detail.get("folder_name") or "") if same_folder else "",
            "multiple_folders": not same_folder,
            "tags": tags_list,
            "people": people_list,
        }

    # -------------------------------------------------------------------------
    # Background XMP Dirty Queue Synchronization
    # -------------------------------------------------------------------------
    def fetch_dirty_media_batch(self, limit: int = 50) -> list[dict[str, Any]]:
        """Fetches up to `limit` rows with metadata_dirty = 1 using idx_media_dirty_partial."""
        conn = self.engine.catalog_conn()
        dirty_ids = [
            int(r["id"])
            for r in conn.execute(
                "SELECT id FROM media WHERE metadata_dirty = 1 LIMIT ?;", (limit,)
            ).fetchall()
        ]
        return [self.get_media_detail(mid) for mid in dirty_ids]

    def count_dirty_media(self) -> int:
        """Returns the total number of media rows with metadata_dirty = 1."""
        conn = self.engine.catalog_conn()
        row = conn.execute("SELECT COUNT(*) AS cnt FROM media WHERE metadata_dirty = 1;").fetchone()
        return int(row["cnt"]) if row else 0

    def get_total_media_count(self) -> int:
        """Returns the total number of non-missing media records in the catalog."""
        conn = self.engine.catalog_conn()
        priv_clause = self._private_folder_clause("f")
        row = conn.execute(
            f"""
            SELECT COUNT(*) AS cnt
            FROM media m
            JOIN folders f ON f.id = m.folder_id
            WHERE COALESCE(m.is_missing, 0) = 0 AND {priv_clause};
            """
        ).fetchone()
        return int(row["cnt"]) if row else 0

    def mark_media_clean(
        self,
        media_ids: Sequence[int],
        updated_mtimes: dict[int, int] | None = None,
        updated_sizes: dict[int, int] | None = None,
    ) -> None:
        if not media_ids:
            return
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            for mid in media_ids:
                has_mtime = bool(updated_mtimes and mid in updated_mtimes)
                has_size = bool(updated_sizes and mid in updated_sizes)
                if has_mtime and has_size:
                    conn.execute(
                        "UPDATE media SET metadata_dirty = 0, mtime = ?, size = ? WHERE id = ?;",
                        (updated_mtimes[mid], updated_sizes[mid], mid),
                    )
                elif has_mtime:
                    conn.execute(
                        "UPDATE media SET metadata_dirty = 0, mtime = ? WHERE id = ?;",
                        (updated_mtimes[mid], mid),
                    )
                elif has_size:
                    conn.execute(
                        "UPDATE media SET metadata_dirty = 0, size = ? WHERE id = ?;",
                        (updated_sizes[mid], mid),
                    )
                else:
                    conn.execute(
                        "UPDATE media SET metadata_dirty = 0 WHERE id = ?;", (mid,)
                    )
            conn.commit()

    # -------------------------------------------------------------------------
    # Static Albums & 17-Dimension Boolean Smart Album Engine
    # -------------------------------------------------------------------------
    @staticmethod
    def compile_smart_album_where(
        rules_dict: dict[str, Any] | str, alias: str = "m"
    ) -> tuple[str, list[Any]]:
        """Compile a Smart Album rules dict/JSON into parameterized SQLite WHERE SQL."""
        return compile_smart_album_where(rules_dict, alias=alias)

    def create_album(
        self, name: str, description: str = "", cover_media_id: int | None = None
    ) -> int:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Album name cannot be empty")
        now_ts = int(time.time())
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            cur = conn.execute(
                """
                INSERT INTO albums (name, description, cover_media_id, created_at)
                VALUES (?, ?, ?, ?);
                """,
                (clean_name, description.strip(), cover_media_id, now_ts),
            )
            conn.commit()
            return int(cur.lastrowid)

    def rename_album(self, album_id: int, new_name: str) -> None:
        clean_name = new_name.strip()
        if not clean_name:
            raise ValueError("Album name cannot be empty")
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute(
                "UPDATE albums SET name = ? WHERE id = ?;",
                (clean_name, int(album_id)),
            )
            conn.commit()

    def delete_album(self, album_id: int) -> None:
        """Delete a Static Album (cascades album_items without deleting media rows)."""
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute("DELETE FROM albums WHERE id = ?;", (int(album_id),))
            conn.commit()

    def add_media_to_album(self, album_id: int, media_ids: Sequence[int]) -> int:
        if not media_ids:
            return 0
        aid = int(album_id)
        added = 0
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            row = conn.execute(
                "SELECT COALESCE(MAX(sort_order), 0) FROM album_items WHERE album_id = ?;",
                (aid,),
            ).fetchone()
            next_order = int(row[0]) + 1 if row else 1
            for mid in media_ids:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO album_items (album_id, media_id, sort_order)
                    VALUES (?, ?, ?);
                    """,
                    (aid, int(mid), next_order),
                )
                if cur.rowcount > 0:
                    added += 1
                    next_order += 1
            conn.execute(
                "UPDATE albums SET cover_media_id = COALESCE(cover_media_id, ?) WHERE id = ?;",
                (int(media_ids[0]), aid),
            )
            conn.commit()
        return added

    def remove_media_from_album(self, album_id: int, media_ids: Sequence[int]) -> int:
        """Remove media associations from album_items without deleting media rows."""
        if not media_ids:
            return 0
        aid = int(album_id)
        removed = 0
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            for chunk in _iter_id_chunks([int(m) for m in media_ids], 900):
                placeholders = ",".join("?" for _ in chunk)
                cur = conn.execute(
                    f"DELETE FROM album_items WHERE album_id = ? AND media_id IN ({placeholders});",
                    [aid, *chunk],
                )
                removed += max(0, cur.rowcount)
            cover_row = conn.execute(
                "SELECT cover_media_id FROM albums WHERE id = ?;", (aid,)
            ).fetchone()
            if cover_row and cover_row["cover_media_id"] in set(int(m) for m in media_ids):
                next_cover = conn.execute(
                    "SELECT media_id FROM album_items WHERE album_id = ? ORDER BY sort_order ASC LIMIT 1;",
                    (aid,),
                ).fetchone()
                conn.execute(
                    "UPDATE albums SET cover_media_id = ? WHERE id = ?;",
                    (int(next_cover[0]) if next_cover else None, aid),
                )
            conn.commit()
        return removed

    def get_albums_with_counts(self) -> list[dict[str, Any]]:
        conn = self.engine.catalog_conn()
        priv_clause = self._private_folder_clause("f")
        rows = conn.execute(
            f"""
            SELECT
                a.id,
                a.name,
                a.description,
                a.cover_media_id,
                a.created_at,
                COUNT(CASE WHEN m.id IS NOT NULL AND {priv_clause} THEN 1 END) AS item_count
            FROM albums a
            LEFT JOIN album_items ai ON ai.album_id = a.id
            LEFT JOIN media m ON m.id = ai.media_id AND COALESCE(m.is_missing, 0) = 0
            LEFT JOIN folders f ON f.id = m.folder_id
            GROUP BY a.id
            ORDER BY LOWER(a.name) ASC, a.id ASC;
            """
        ).fetchall()
        return [
            {
                "id": int(r["id"]),
                "name": str(r["name"]),
                "description": str(r["description"] or ""),
                "cover_media_id": r["cover_media_id"],
                "created_at": int(r["created_at"]),
                "item_count": int(r["item_count"] or 0),
            }
            for r in rows
        ]

    def set_album_cover(self, album_id: int, media_id: int) -> None:
        """Sets the cover photo (cover_media_id) for a static album."""
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute(
                "UPDATE albums SET cover_media_id = ? WHERE id = ?;",
                (int(media_id), int(album_id)),
            )
            conn.commit()

    def get_album_cover_candidates(self, album_id: int) -> list[dict[str, Any]]:
        """Returns non-missing media items in an album for cover selection, current cover first."""
        conn = self.engine.catalog_conn()
        rows = conn.execute(
            """
            SELECT m.id, m.filename, a.cover_media_id
            FROM album_items ai
            JOIN albums a ON a.id = ai.album_id
            JOIN media m ON m.id = ai.media_id
            WHERE ai.album_id = ? AND COALESCE(m.is_missing, 0) = 0
            ORDER BY CASE WHEN m.id = a.cover_media_id THEN 0 ELSE 1 END ASC,
                     ai.sort_order ASC, m.date_taken DESC, m.id DESC;
            """,
            (int(album_id),),
        ).fetchall()
        return [
            {
                "id": int(r["id"]),
                "filename": str(r["filename"]),
                "is_cover": (r["cover_media_id"] is not None and int(r["id"]) == int(r["cover_media_id"])),
            }
            for r in rows
        ]

    def get_album_preview_media_ids(self, album_id: int, limit: int = 3) -> list[int]:
        """Return up to `limit` non-missing media IDs for an album's stack preview, cover first."""
        conn = self.engine.catalog_conn()
        rows = conn.execute(
            """
            SELECT m.id
            FROM album_items ai
            JOIN albums a ON a.id = ai.album_id
            JOIN media m ON m.id = ai.media_id
            WHERE ai.album_id = ? AND COALESCE(m.is_missing, 0) = 0
            ORDER BY CASE WHEN m.id = a.cover_media_id THEN 0 ELSE 1 END ASC,
                     ai.sort_order ASC, m.date_taken DESC, m.id DESC
            LIMIT ?;
            """,
            (int(album_id), int(limit)),
        ).fetchall()
        return [int(r["id"]) for r in rows]

    def create_smart_album(
        self, name: str, rules_dict: dict[str, Any] | str
    ) -> int:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Smart album name cannot be empty")
        rules_json = (
            rules_dict
            if isinstance(rules_dict, str)
            else json.dumps(rules_dict, ensure_ascii=False)
        )
        now_ts = int(time.time())
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            cur = conn.execute(
                """
                INSERT INTO smart_albums (name, rules_json, created_at)
                VALUES (?, ?, ?);
                """,
                (clean_name, rules_json, now_ts),
            )
            conn.commit()
            return int(cur.lastrowid)

    def update_smart_album(
        self,
        smart_album_id: int,
        name: str | None = None,
        rules_dict: dict[str, Any] | str | None = None,
        rules: dict[str, Any] | str | None = None,
    ) -> None:
        if rules_dict is None and rules is not None:
            rules_dict = rules
        updates: list[str] = []
        params: list[Any] = []
        if name is not None:
            clean_name = name.strip()
            if not clean_name:
                raise ValueError("Smart album name cannot be empty")
            updates.append("name = ?")
            params.append(clean_name)
        if rules_dict is not None:
            rules_json = (
                rules_dict
                if isinstance(rules_dict, str)
                else json.dumps(rules_dict, ensure_ascii=False)
            )
            updates.append("rules_json = ?")
            params.append(rules_json)
        if not updates:
            return
        params.append(int(smart_album_id))
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute(
                f"UPDATE smart_albums SET {', '.join(updates)} WHERE id = ?;",
                params,
            )
            conn.commit()

    def delete_smart_album(self, smart_album_id: int) -> None:
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute(
                "DELETE FROM smart_albums WHERE id = ?;", (int(smart_album_id),)
            )
            conn.commit()

    def get_smart_album(self, smart_album_id: int) -> dict[str, Any] | None:
        conn = self.engine.catalog_conn()
        row = conn.execute(
            "SELECT id, name, rules_json, created_at FROM smart_albums WHERE id = ?;",
            (int(smart_album_id),),
        ).fetchone()
        if not row:
            return None
        raw_json = str(row["rules_json"] or "{}")
        try:
            rules_dict = json.loads(raw_json)
        except Exception:
            rules_dict = {"match_mode": "all", "rules": []}
        return {
            "id": int(row["id"]),
            "name": str(row["name"]),
            "rules_json": raw_json,
            "rules_dict": rules_dict,
            "rules": rules_dict,
            "created_at": int(row["created_at"]),
        }

    def count_smart_album_matches(self, rules_dict: dict[str, Any] | str) -> int:
        where_sql, params = compile_smart_album_where(rules_dict, alias="m")
        conn = self.engine.catalog_conn()
        priv_clause = self._private_folder_clause("f")
        row = conn.execute(
            f"""
            SELECT COUNT(*) FROM media m
            JOIN folders f ON f.id = m.folder_id
            WHERE COALESCE(m.is_missing, 0) = 0 AND {priv_clause} AND ({where_sql});
            """,
            params,
        ).fetchone()
        return int(row[0]) if row else 0

    def get_smart_albums_with_counts(self) -> list[dict[str, Any]]:
        conn = self.engine.catalog_conn()
        rows = conn.execute(
            "SELECT id, name, rules_json, created_at, cover_media_id FROM smart_albums ORDER BY id ASC;"
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            raw_json = str(r["rules_json"] or "{}")
            try:
                rules_dict = json.loads(raw_json)
            except Exception:
                rules_dict = {"match_mode": "all", "rules": []}
            cnt = self.count_smart_album_matches(rules_dict)
            out.append(
                {
                    "id": int(r["id"]),
                    "name": str(r["name"]),
                    "rules_json": raw_json,
                    "rules_dict": rules_dict,
                    "created_at": int(r["created_at"]),
                    "cover_media_id": r["cover_media_id"],
                    "item_count": cnt,
                }
            )
        return out

    def set_smart_album_cover(self, smart_album_id: int, media_id: int) -> None:
        """Sets the cover photo (cover_media_id) for a Smart Album."""
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute(
                "UPDATE smart_albums SET cover_media_id = ? WHERE id = ?;",
                (int(media_id), int(smart_album_id)),
            )
            conn.commit()

    def get_smart_album_cover_candidates(self, smart_album_id: int, limit: int = 120) -> list[dict[str, Any]]:
        """Returns non-missing media items matching a Smart Album for cover selection."""
        sa = self.get_smart_album(int(smart_album_id))
        if not sa:
            return []
        conn = self.engine.catalog_conn()
        cov_row = conn.execute(
            "SELECT cover_media_id FROM smart_albums WHERE id = ?;",
            (int(smart_album_id),),
        ).fetchone()
        cover_id = int(cov_row["cover_media_id"]) if (cov_row and cov_row["cover_media_id"] is not None) else None
        where_sql, params = compile_smart_album_where(sa.get("rules") or {}, alias="m")
        if cover_id is not None:
            order_sql = "ORDER BY CASE WHEN m.id = ? THEN 0 ELSE 1 END ASC, m.date_taken DESC, m.id DESC"
            q_params = (*params, cover_id, int(limit))
        else:
            order_sql = "ORDER BY m.date_taken DESC, m.id DESC"
            q_params = (*params, int(limit))
        rows = conn.execute(
            f"""
            SELECT m.id, m.filename
            FROM media m
            WHERE COALESCE(m.is_missing, 0) = 0 AND ({where_sql})
            {order_sql}
            LIMIT ?;
            """,
            q_params,
        ).fetchall()
        return [
            {
                "id": int(r["id"]),
                "filename": str(r["filename"]),
                "is_cover": (cover_id is not None and int(r["id"]) == cover_id),
            }
            for r in rows
        ]

    def get_smart_album_preview_media_ids(
        self, rules_dict: dict[str, Any] | str, limit: int = 3, cover_media_id: int | None = None
    ) -> list[int]:
        """Return up to `limit` non-missing media IDs matching a Smart Album's rules, cover first."""
        where_sql, params = compile_smart_album_where(rules_dict, alias="m")
        conn = self.engine.catalog_conn()
        if cover_media_id is not None:
            order_sql = "ORDER BY CASE WHEN m.id = ? THEN 0 ELSE 1 END ASC, m.date_taken DESC, m.id DESC"
            q_params = (*params, int(cover_media_id), int(limit))
        else:
            order_sql = "ORDER BY m.date_taken DESC, m.id DESC"
            q_params = (*params, int(limit))
        rows = conn.execute(
            f"""
            SELECT m.id
            FROM media m
            WHERE COALESCE(m.is_missing, 0) = 0 AND ({where_sql})
            {order_sql}
            LIMIT ?;
            """,
            q_params,
        ).fetchall()
        return [int(r["id"]) for r in rows]

    def get_camera_makes(self) -> list[str]:
        """Return distinct non-empty camera manufacturer strings for Smart Album autocomplete."""
        conn = self.engine.catalog_conn()
        rows = conn.execute(
            """
            SELECT DISTINCT TRIM(camera_make) AS make
            FROM media
            WHERE COALESCE(is_missing, 0) = 0
              AND COALESCE(TRIM(camera_make), '') != ''
            ORDER BY make COLLATE NOCASE ASC;
            """
        ).fetchall()
        return [str(r["make"]) for r in rows if str(r["make"] or "").strip()]

    def get_camera_models(self) -> list[str]:
        """Return distinct non-empty camera make/model strings for Smart Album autocomplete."""
        conn = self.engine.catalog_conn()
        rows = conn.execute(
            """
            SELECT DISTINCT TRIM(
                CASE
                    WHEN COALESCE(camera_make, '') != '' AND COALESCE(camera_model, '') != ''
                         AND LOWER(camera_model) NOT LIKE LOWER(camera_make) || '%'
                        THEN camera_make || ' ' || camera_model
                    WHEN COALESCE(camera_model, '') != ''
                        THEN camera_model
                    ELSE COALESCE(camera_make, '')
                END
            ) AS cam
            FROM media
            WHERE COALESCE(is_missing, 0) = 0
              AND (COALESCE(TRIM(camera_make), '') != '' OR COALESCE(TRIM(camera_model), '') != '')
            ORDER BY cam COLLATE NOCASE ASC;
            """
        ).fetchall()
        return [str(r[0]) for r in rows if r[0]]

    def query_camera_counts(self) -> list[tuple[str, int]]:
        """Return `[(normalized_camera_name, count), ...]` sorted alphabetically, plus '__NO_CAMERA__' if any."""
        conn = self.engine.catalog_conn()
        priv_clause = self._private_folder_clause("f")
        norm_cam_expr = (
            "TRIM(CASE "
            "WHEN COALESCE(TRIM(m.camera_make), '') != '' AND COALESCE(TRIM(m.camera_model), '') != '' "
            "AND LOWER(TRIM(m.camera_model)) NOT LIKE LOWER(TRIM(m.camera_make)) || '%' "
            "THEN TRIM(m.camera_make) || ' ' || TRIM(m.camera_model) "
            "WHEN COALESCE(TRIM(m.camera_model), '') != '' THEN TRIM(m.camera_model) "
            "ELSE COALESCE(TRIM(m.camera_make), '') END)"
        )
        rows = conn.execute(
            f"""
            SELECT {norm_cam_expr} AS cam, COUNT(*) AS cnt
            FROM media m
            JOIN folders f ON f.id = m.folder_id
            WHERE COALESCE(m.is_missing, 0) = 0 AND m.live_companion_for_id IS NULL AND {priv_clause}
              AND (COALESCE(TRIM(m.camera_make), '') != '' OR COALESCE(TRIM(m.camera_model), '') != '')
            GROUP BY LOWER({norm_cam_expr})
            ORDER BY cam COLLATE NOCASE ASC;
            """
        ).fetchall()
        out = [(str(r[0]), int(r[1])) for r in rows if str(r[0] or "").strip()]
        no_cam_row = conn.execute(
            f"""
            SELECT COUNT(*) FROM media m
            JOIN folders f ON f.id = m.folder_id
            WHERE COALESCE(m.is_missing, 0) = 0 AND m.live_companion_for_id IS NULL AND {priv_clause}
              AND COALESCE(TRIM(m.camera_make), '') = '' AND COALESCE(TRIM(m.camera_model), '') = '';
            """
        ).fetchone()
        no_cnt = int(no_cam_row[0]) if no_cam_row else 0
        if no_cnt > 0:
            out.append(("__NO_CAMERA__", no_cnt))
        return out

    def get_distinct_places(self, limit: int = 200) -> list[str]:
        """Return distinct non-empty place strings for Smart Album autocomplete."""
        conn = self.engine.catalog_conn()
        seen: dict[str, str] = {}
        for col in ("location_name", "location_city", "location_state", "location_country"):
            rows = conn.execute(
                f"""
                SELECT DISTINCT TRIM({col}) AS val
                FROM media
                WHERE COALESCE(is_missing, 0) = 0
                  AND {col} IS NOT NULL AND TRIM({col}) != ''
                LIMIT ?;
                """,
                (int(limit),),
            ).fetchall()
            for r in rows:
                s = str(r[0]).strip()
                if s and s.lower() not in seen:
                    seen[s.lower()] = s
        return sorted(seen.values(), key=str.lower)[:limit]

    def get_distinct_extensions(self) -> list[str]:
        """Return distinct lowercase file extensions in the catalog without leading dots."""
        conn = self.engine.catalog_conn()
        rows = conn.execute(
            """
            SELECT DISTINCT LOWER(LTRIM(COALESCE(ext, ''), '.')) AS e
            FROM media
            WHERE COALESCE(is_missing, 0) = 0 AND COALESCE(TRIM(ext), '') != ''
            ORDER BY e ASC;
            """
        ).fetchall()
        return [str(r[0]) for r in rows if r[0]]


class ThumbnailRepository:
    """High-speed BLOB repository for thumbnails.db (256x256 WebP blobs)."""

    def __init__(self, engine: DualDatabaseEngine) -> None:
        self.engine = engine
        self.include_private: bool = False
        self._has_ocr_fts: bool | None = None

    def _supports_ocr_fts(self) -> bool:
        if self._has_ocr_fts is None:
            try:
                row = self.engine.catalog_conn().execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='media_ocr_fts';"
                ).fetchone()
                self._has_ocr_fts = row is not None
            except Exception:
                self._has_ocr_fts = False
        return bool(self._has_ocr_fts)

    def set_include_private(self, include_private: bool) -> None:
        """Toggle session visibility of Private Folders across all catalog queries."""
        self.include_private = bool(include_private)

    def has_private_folders(self) -> bool:
        """Returns True if at least one folder in catalog.db is marked is_private = 1."""
        conn = self.engine.catalog_conn()
        row = conn.execute(
            "SELECT 1 FROM folders WHERE COALESCE(is_private, 0) = 1 LIMIT 1;"
        ).fetchone()
        return row is not None

    def _private_folder_clause(self, alias: str = "f") -> str:
        """SQL predicate enforcing excluded-folder and Private Folder session isolation."""
        base = f"COALESCE({alias}.monitor_mode, 'always') != 'excluded'"
        if self.include_private:
            return base
        return f"({base} AND COALESCE({alias}.is_private, 0) = 0)"

    @property
    def _write_lock(self) -> Any:
        return getattr(self.engine, "thumb_write_lock", self.engine.write_lock)

    def get_webp_blob(
        self, media_id: int, expected_mtime: int | None = None
    ) -> bytes | None:
        conn = self.engine.thumbnails_conn()
        cur = conn.cursor()
        cur.row_factory = None
        row = cur.execute(
            "SELECT webp_blob, mtime FROM thumbnails WHERE media_id = ?;", (media_id,)
        ).fetchone()
        if row is None:
            return None
        if expected_mtime is not None and int(row[1]) < int(expected_mtime):
            return None
        return bytes(row[0])

    def get_webp_blobs_batch(
        self,
        media_ids: Sequence[int],
        expected_mtimes: dict[int, int] | None = None,
    ) -> dict[int, bytes]:
        if not media_ids:
            return {}
        conn = self.engine.thumbnails_conn()
        cur = conn.cursor()
        cur.row_factory = None
        result: dict[int, bytes] = {}
        for chunk in _iter_id_chunks(media_ids, 900):
            placeholders = ",".join("?" for _ in chunk)
            rows = cur.execute(
                f"SELECT media_id, webp_blob, mtime FROM thumbnails WHERE media_id IN ({placeholders});",
                list(chunk),
            ).fetchall()
            if expected_mtimes is not None:
                for mid_val, blob_val, mtime_val in rows:
                    mid_i = int(mid_val)
                    exp = expected_mtimes.get(mid_i)
                    if exp is None or int(mtime_val) >= int(exp):
                        result[mid_i] = bytes(blob_val)
            else:
                for mid_val, blob_val, _ in rows:
                    result[int(mid_val)] = bytes(blob_val)
        return result

    def get_signatures_batch(
        self,
        media_ids: Sequence[int],
    ) -> dict[int, tuple[int, int, int, float, bytes]]:
        """
        Returns cached perceptual signatures {media_id: (mtime, phash64_u64, dhash64_u64, sharpness, lab_haar)}
        from thumbnails.db.
        """
        if not media_ids:
            return {}
        conn = self.engine.thumbnails_conn()
        cur = conn.cursor()
        cur.row_factory = None
        result: dict[int, tuple[int, int, int, float, bytes]] = {}
        try:
            for chunk in _iter_id_chunks(media_ids, 900):
                ph = ",".join("?" for _ in chunk)
                rows = cur.execute(
                    f"SELECT media_id, mtime, phash64, dhash64, sharpness, lab_haar "
                    f"FROM media_signatures WHERE media_id IN ({ph});",
                    list(chunk),
                ).fetchall()
                for mid_v, mt_v, ph_v, dh_v, sh_v, lh_v in rows:
                    result[int(mid_v)] = (
                        int(mt_v),
                        int(ph_v) & 0xFFFFFFFFFFFFFFFF,
                        int(dh_v) & 0xFFFFFFFFFFFFFFFF,
                        float(sh_v or 0.0),
                        bytes(lh_v) if lh_v is not None else b"",
                    )
        except Exception:
            pass
        return result

    def upsert_signatures_batch(
        self,
        rows: Sequence[tuple[int, int, int, int, float, bytes]],
    ) -> None:
        """
        Persists perceptual signatures [(media_id, mtime, phash64_u64, dhash64_u64, sharpness, lab_haar)]
        into thumbnails.db.
        """
        if not rows:
            return
        converted = [
            (
                int(mid),
                int(mt),
                int.from_bytes((int(ph) & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "big"), "big", signed=True),
                int.from_bytes((int(dh) & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "big"), "big", signed=True),
                float(sh),
                bytes(lh) if lh else b"",
            )
            for mid, mt, ph, dh, sh, lh in rows
        ]
        try:
            with self._write_lock:
                conn = self.engine.thumbnails_conn()
                conn.executemany(
                    """
                    INSERT INTO media_signatures(media_id, mtime, phash64, dhash64, sharpness, lab_haar)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(media_id) DO UPDATE SET
                        mtime = excluded.mtime,
                        phash64 = excluded.phash64,
                        dhash64 = excluded.dhash64,
                        sharpness = excluded.sharpness,
                        lab_haar = excluded.lab_haar;
                    """,
                    converted,
                )
                conn.commit()
        except Exception:
            pass

    def get_first_cached_media_id(self, media_ids: Sequence[int]) -> int | None:
        """Returns the first media_id from media_ids that already has a cached thumbnail."""
        if not media_ids:
            return None
        conn = self.engine.thumbnails_conn()
        cur = conn.cursor()
        cur.row_factory = None
        for chunk in _iter_id_chunks(media_ids, 900):
            placeholders = ",".join("?" for _ in chunk)
            row = cur.execute(
                f"SELECT media_id FROM thumbnails WHERE media_id IN ({placeholders}) LIMIT 1;",
                list(chunk),
            ).fetchone()
            if row is not None:
                return int(row[0])
        return None

    def put_webp_blob(
        self,
        media_id: int,
        webp_blob: bytes,
        mtime: int,
        width: int = 256,
        height: int = 256,
    ) -> None:
        with self._write_lock:
            conn = self.engine.thumbnails_conn()
            conn.execute(
                """
                INSERT INTO thumbnails (media_id, webp_blob, mtime, width, height)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(media_id) DO UPDATE SET
                    webp_blob = excluded.webp_blob,
                    mtime = excluded.mtime,
                    width = excluded.width,
                    height = excluded.height;
                """,
                (media_id, webp_blob, mtime, width, height),
            )
            conn.commit()

    def put_webp_blobs_batch(
        self,
        entries: Sequence[tuple[int, bytes, int, int, int]],
    ) -> None:
        if not entries:
            return
        with self._write_lock:
            conn = self.engine.thumbnails_conn()
            conn.executemany(
                """
                INSERT INTO thumbnails (media_id, webp_blob, mtime, width, height)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(media_id) DO UPDATE SET
                    webp_blob = excluded.webp_blob,
                    mtime = excluded.mtime,
                    width = excluded.width,
                    height = excluded.height;
                """,
                entries,
            )
            conn.commit()

    def invalidate(self, media_ids: Sequence[int]) -> None:
        if not media_ids:
            return
        with self._write_lock:
            conn = self.engine.thumbnails_conn()
            for chunk in _iter_id_chunks(media_ids, 900):
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"DELETE FROM thumbnails WHERE media_id IN ({placeholders});",
                    list(chunk),
                )
            conn.commit()

    def get_face_thumbnail(self, media_people_id: int) -> bytes | None:
        conn = self.engine.thumbnails_conn()
        row = conn.execute(
            "SELECT crop_webp FROM face_thumbnails WHERE face_id = ?;",
            (int(media_people_id),),
        ).fetchone()
        return bytes(row["crop_webp"]) if row and row["crop_webp"] else None

    def get_face_thumbnails_batch(
        self, media_people_ids: Sequence[int]
    ) -> dict[int, bytes]:
        if not media_people_ids:
            return {}
        result: dict[int, bytes] = {}
        conn = self.engine.thumbnails_conn()
        for chunk in _iter_id_chunks(media_people_ids, 900):
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT face_id, crop_webp FROM face_thumbnails WHERE face_id IN ({placeholders});",
                list(chunk),
            ).fetchall()
            for r in rows:
                if r["crop_webp"]:
                    result[int(r["face_id"])] = bytes(r["crop_webp"])
        return result

    def put_face_thumbnail(self, media_people_id: int, webp_blob: bytes) -> None:
        with self._write_lock:
            conn = self.engine.thumbnails_conn()
            conn.execute(
                """
                INSERT INTO face_thumbnails (face_id, crop_webp)
                VALUES (?, ?)
                ON CONFLICT(face_id) DO UPDATE SET crop_webp = excluded.crop_webp;
                """,
                (int(media_people_id), webp_blob),
            )
            conn.commit()

    def put_face_thumbnails_batch(
        self, entries: Sequence[tuple[int, bytes]]
    ) -> None:
        if not entries:
            return
        with self._write_lock:
            conn = self.engine.thumbnails_conn()
            conn.executemany(
                """
                INSERT INTO face_thumbnails (face_id, crop_webp)
                VALUES (?, ?)
                ON CONFLICT(face_id) DO UPDATE SET crop_webp = excluded.crop_webp;
                """,
                [(int(fid), blob) for fid, blob in entries],
            )
            conn.commit()

    def invalidate_face_thumbnails(self, media_people_ids: Sequence[int]) -> None:
        if not media_people_ids:
            return
        with self._write_lock:
            conn = self.engine.thumbnails_conn()
            for chunk in _iter_id_chunks(media_people_ids, 900):
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"DELETE FROM face_thumbnails WHERE face_id IN ({placeholders});",
                    list(chunk),
                )
            conn.commit()

    def get_face_thumbnail_for_media_person(self, media_id: int, person_id: int) -> bytes | None:
        """Returns the WebP face thumbnail bytes for person_id on media_id."""
        cat_conn = self.engine.catalog_conn()
        row = cat_conn.execute(
            """
            SELECT id, rect_w, rect_h
            FROM media_people
            WHERE media_id = ? AND person_id = ? AND state = 1
              AND rect_w > 0.005 AND rect_h > 0.005
            ORDER BY is_manual DESC, id DESC
            LIMIT 1;
            """,
            (int(media_id), int(person_id)),
        ).fetchone()
        if not row:
            return None
        return self.get_face_thumbnail(int(row["id"]))

    def get_face_thumbnails_for_media_person_batch(
        self, media_ids: Sequence[int], person_id: int
    ) -> dict[int, bytes]:
        """Batched lookup returning {media_id: webp_bytes} for person_id's confirmed face crops."""
        if not media_ids:
            return {}
        cat_conn = self.engine.catalog_conn()
        media_to_face_id: dict[int, int] = {}
        pid = int(person_id)
        for chunk in _iter_id_chunks(media_ids, 900):
            placeholders = ",".join("?" for _ in chunk)
            rows = cat_conn.execute(
                f"""
                SELECT id, media_id
                FROM media_people
                WHERE media_id IN ({placeholders})
                  AND person_id = ? AND state = 1
                  AND rect_w > 0.005 AND rect_h > 0.005
                ORDER BY is_manual DESC, id DESC;
                """,
                list(chunk) + [pid],
            ).fetchall()
            for r in rows:
                mid = int(r["media_id"])
                if mid not in media_to_face_id:
                    media_to_face_id[mid] = int(r["id"])

        if not media_to_face_id:
            return {}

        face_blobs = self.get_face_thumbnails_batch(list(media_to_face_id.values()))
        return {
            mid: face_blobs[fid]
            for mid, fid in media_to_face_id.items()
            if fid in face_blobs
        }

    def purge_mismatched_orientation_thumbnails(self) -> int:
        """Purges any cached thumbnails whose aspect ratio conflicts with catalog orientation."""
        try:
            with self._write_lock:
                tconn = self.engine.thumbnails_conn()
                cat_path = getattr(self.engine, "catalog_path", None)
                if cat_path and Path(cat_path).exists():
                    clean_cat = str(cat_path).replace("\\", "/")
                    tconn.execute(f"ATTACH DATABASE '{clean_cat}' AS cat_db;")
                    cur = tconn.cursor()
                    cur.execute(
                        """
                        DELETE FROM thumbnails
                        WHERE media_id IN (
                            SELECT t.media_id
                            FROM thumbnails t
                            JOIN cat_db.media m ON t.media_id = m.id
                            WHERE (m.orientation IN (6, 8) AND t.width > t.height)
                               OR (m.orientation IN (1, 3) AND m.width > m.height AND t.height > t.width)
                        );
                        """
                    )
                    count = cur.rowcount
                    tconn.commit()
                    tconn.execute("DETACH DATABASE cat_db;")
                    return max(0, count)
        except Exception:
            pass
        return 0


def _escape_like_term(val: str) -> str:
    """Escape SQL LIKE wildcards using backslash as escape character."""
    return val.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _parse_numeric_pair(val: Any) -> tuple[float, float] | None:
    """Parse a two-value range from list/tuple, dict, or comma/dash/range string."""
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        try:
            a, b = float(val[0]), float(val[1])
            return (min(a, b), max(a, b))
        except (TypeError, ValueError):
            return None
    if isinstance(val, dict):
        s_v = val.get("start", val.get("min"))
        e_v = val.get("end", val.get("max"))
        if s_v is not None and e_v is not None:
            try:
                a, b = float(s_v), float(e_v)
                return (min(a, b), max(a, b))
            except (TypeError, ValueError):
                return None
    txt = str(val or "").strip()
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*(?:-|to|\.\.|,)\s*(-?\d+(?:\.\d+)?)\s*$", txt, re.I)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (min(a, b), max(a, b))
    return None


def _parse_month_number(val: Any) -> int | None:
    if val is None:
        return None
    if isinstance(val, int) or str(val).strip().isdigit():
        num = int(str(val).strip())
        return num if 1 <= num <= 12 else None
    s = str(val).strip().lower()
    for m_idx in range(1, 13):
        if calendar.month_name[m_idx].lower() == s or calendar.month_abbr[m_idx].lower() == s:
            return m_idx
    return None


def _compile_numeric_expr_rule(
    expr: str, op: str, val: Any, scale: float = 1.0
) -> tuple[str, list[Any]] | None:
    """Shared compiler for numeric range dimensions (file_size_mb, megapixels)."""
    if op == "between":
        pair = _parse_numeric_pair(val)
        if not pair:
            return None
        return f"({expr} BETWEEN ? AND ?)", [pair[0] * scale, pair[1] * scale]
    try:
        num = float(val) * scale
    except (TypeError, ValueError):
        return None
    if op in ("gte", ">="):
        return f"({expr} >= ?)", [num]
    if op in ("lte", "<="):
        return f"({expr} <= ?)", [num]
    if op in ("eq", "="):
        return f"({expr} = ?)", [num]
    return None


def _compile_single_smart_rule(
    rule: dict[str, Any], alias: str = "m"
) -> tuple[str, list[Any]] | None:
    """Compile a single {field, op, value} Smart Album rule row into parameterized SQL."""
    raw_field = str(rule.get("field") or "").strip().lower()
    op = str(rule.get("op") or "").strip().lower()
    val = rule.get("value")
    val_str = "" if val is None else str(val).strip()

    # 1. People (confirmed faces state = 1)
    if raw_field in ("people", "person"):
        if op == "is_empty":
            return f"NOT EXISTS (SELECT 1 FROM media_people mp WHERE mp.media_id = {alias}.id AND mp.state = 1)", []
        if op == "is_not_empty":
            return f"EXISTS (SELECT 1 FROM media_people mp WHERE mp.media_id = {alias}.id AND mp.state = 1)", []
        if not val_str:
            return None
        if isinstance(val, int) and not isinstance(val, bool):
            pred = "mp.person_id = ?"
            param: Any = int(val)
        else:
            pred = "LOWER(p.name) = LOWER(?)"
            param = val_str
        sub = (
            f"SELECT 1 FROM media_people mp JOIN people p ON p.id = mp.person_id "
            f"WHERE mp.media_id = {alias}.id AND mp.state = 1 AND {pred}"
        )
        if op in ("excludes", "not_includes", "does_not_include"):
            return f"NOT EXISTS ({sub})", [param]
        return f"EXISTS ({sub})", [param]

    # 2. Descriptive Tags (with tag_closure hierarchy)
    if raw_field in ("tag", "tags", "descriptive_tag"):
        if op == "is_empty":
            return f"NOT EXISTS (SELECT 1 FROM media_tags mt WHERE mt.media_id = {alias}.id)", []
        if op == "is_not_empty":
            return f"EXISTS (SELECT 1 FROM media_tags mt WHERE mt.media_id = {alias}.id)", []
        if not val_str:
            return None
        sub = (
            f"SELECT 1 FROM media_tags mt "
            f"JOIN tag_closure tc ON tc.descendant_id = mt.tag_id "
            f"JOIN tags t ON t.id = tc.ancestor_id "
            f"WHERE mt.media_id = {alias}.id "
            f"AND (LOWER(t.name) = LOWER(?) OR LOWER(t.full_path) = LOWER(?))"
        )
        if op in ("excludes", "not_includes", "does_not_include"):
            return f"NOT EXISTS ({sub})", [val_str, val_str]
        return f"EXISTS ({sub})", [val_str, val_str]

    # 3. Geotag / Place
    if raw_field in ("place", "geotag", "location"):
        if op == "has_gps":
            return f"({alias}.lat IS NOT NULL AND {alias}.lon IS NOT NULL)", []
        if op == "no_gps":
            return f"({alias}.lat IS NULL OR {alias}.lon IS NULL)", []
        if not val_str:
            return None
        pat = f"%{_escape_like_term(val_str.lower())}%"
        loc_sql = (
            f"(LOWER(COALESCE({alias}.location_name, '')) LIKE ? ESCAPE '\\' "
            f"OR LOWER(COALESCE({alias}.location_country, '')) LIKE ? ESCAPE '\\' "
            f"OR LOWER(COALESCE({alias}.location_state, '')) LIKE ? ESCAPE '\\' "
            f"OR LOWER(COALESCE({alias}.location_city, '')) LIKE ? ESCAPE '\\')"
        )
        if op in ("not_contains", "excludes"):
            return f"NOT {loc_sql}", [pat, pat, pat, pat]
        return loc_sql, [pat, pat, pat, pat]

    # 4. Time of Day
    if raw_field in ("time_of_day", "time_taken"):
        has_dt = f"({alias}.date_taken IS NOT NULL AND {alias}.date_taken != 0)"
        hr_expr = f"CAST(strftime('%H', {alias}.date_taken, 'unixepoch', 'localtime') AS INTEGER)"
        if op == "between_hours":
            hours_txt = val_str
            m_hm = re.match(r"^\s*(\d{1,2})(?::\d{2})?\s*-\s*(\d{1,2})(?::\d{2})?\s*$", hours_txt)
            if m_hm:
                h1, h2 = int(m_hm.group(1)), int(m_hm.group(2))
            else:
                pair = _parse_numeric_pair(val)
                if not pair:
                    return None
                h1, h2 = int(pair[0]), int(pair[1])
            h1 = max(0, min(23, h1))
            h2 = max(0, min(23, h2))
            if h1 <= h2:
                return f"({has_dt} AND {hr_expr} BETWEEN ? AND ?)", [h1, h2]
            return f"({has_dt} AND ({hr_expr} >= ? OR {hr_expr} <= ?))", [h1, h2]
        preset = val_str.lower()
        if "morning" in preset:
            return f"({has_dt} AND {hr_expr} BETWEEN 5 AND 11)", []
        if "afternoon" in preset:
            return f"({has_dt} AND {hr_expr} BETWEEN 12 AND 16)", []
        if "evening" in preset:
            return f"({has_dt} AND {hr_expr} BETWEEN 17 AND 20)", []
        if "night" in preset:
            return f"({has_dt} AND ({hr_expr} >= 21 OR {hr_expr} <= 4))", []
        return None

    # 5. On This Day
    if raw_field == "on_this_day":
        sql = (
            f"({alias}.date_taken IS NOT NULL AND {alias}.date_taken != 0 "
            f"AND strftime('%m-%d', {alias}.date_taken, 'unixepoch', 'localtime') = strftime('%m-%d', 'now', 'localtime'))"
        )
        return ("NOT " + sql if op in ("is_not", "neq") else sql), []

    # 6. Year
    if raw_field == "year":
        if op == "between":
            pair = _parse_numeric_pair(val)
            if not pair:
                return None
            y1, y2 = int(pair[0]), int(pair[1])
            s_ts, _ = _compute_date_bounds_utc(y1)
            _, e_ts = _compute_date_bounds_utc(y2)
            return f"({alias}.date_taken BETWEEN ? AND ?)", [s_ts, e_ts]
        try:
            yr = int(val_str)
        except ValueError:
            return None
        s_ts, e_ts = _compute_date_bounds_utc(yr)
        if op in ("neq", "!=", "is_not"):
            return f"({alias}.date_taken IS NULL OR {alias}.date_taken NOT BETWEEN ? AND ?)", [s_ts, e_ts]
        if op in ("gte", ">="):
            return f"({alias}.date_taken >= ?)", [s_ts]
        if op in ("lte", "<="):
            return f"({alias}.date_taken IS NOT NULL AND {alias}.date_taken != 0 AND {alias}.date_taken <= ?)", [e_ts]
        return f"({alias}.date_taken BETWEEN ? AND ?)", [s_ts, e_ts]

    # 7. Month
    if raw_field == "month":
        m_num = _parse_month_number(val)
        if m_num is None:
            return None
        has_dt = f"({alias}.date_taken IS NOT NULL AND {alias}.date_taken != 0)"
        m_expr = f"CAST(strftime('%m', {alias}.date_taken, 'unixepoch', 'localtime') AS INTEGER)"
        if op in ("neq", "!=", "is_not"):
            return f"({has_dt} AND {m_expr} != ?)", [m_num]
        return f"({has_dt} AND {m_expr} = ?)", [m_num]

    # 8. Date Taken (YYYY-MM-DD exact/range)
    if raw_field in ("date_taken", "date"):
        has_dt = f"({alias}.date_taken IS NOT NULL AND {alias}.date_taken != 0)"
        d_expr = f"date({alias}.date_taken, 'unixepoch', 'localtime')"
        if op == "between":
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                d1, d2 = str(val[0]).strip(), str(val[1]).strip()
            else:
                parts = [p.strip() for p in re.split(r"\s*(?:to|\.\.|,)\s*", val_str) if p.strip()]
                if len(parts) < 2:
                    return None
                d1, d2 = parts[0], parts[1]
            return f"({has_dt} AND {d_expr} BETWEEN ? AND ?)", [min(d1, d2), max(d1, d2)]
        if not val_str:
            return None
        # Support passing just a 4-digit year string in unified Year / Date UI field
        if re.fullmatch(r"\d{4}", val_str):
            return _compile_single_smart_rule({"field": "year", "op": op, "value": int(val_str)}, alias=alias)
        if op in ("gte", ">=", "after"):
            return f"({has_dt} AND {d_expr} >= ?)", [val_str]
        if op in ("lte", "<=", "before"):
            return f"({has_dt} AND {d_expr} <= ?)", [val_str]
        if op in ("neq", "!=", "is_not"):
            return f"({has_dt} AND {d_expr} != ?)", [val_str]
        return f"({has_dt} AND {d_expr} = ?)", [val_str]

    # 9. Camera (Make / Model)
    if raw_field == "camera":
        if op == "is_empty":
            return f"(COALESCE(TRIM({alias}.camera_make), '') = '' AND COALESCE(TRIM({alias}.camera_model), '') = '')", []
        if op == "is_not_empty":
            return f"(COALESCE(TRIM({alias}.camera_make), '') != '' OR COALESCE(TRIM({alias}.camera_model), '') != '')", []
        if not val_str:
            return None
        cam_expr = f"LOWER(TRIM(COALESCE({alias}.camera_make, '') || ' ' || COALESCE({alias}.camera_model, '')))"
        if op in ("equals", "eq", "is"):
            return f"(LOWER(COALESCE({alias}.camera_model, '')) = LOWER(?) OR {cam_expr} = LOWER(?))", [val_str, val_str]
        pat = f"%{_escape_like_term(val_str.lower())}%"
        if op in ("not_contains", "excludes"):
            return f"({cam_expr} NOT LIKE ? ESCAPE '\\')", [pat]
        return f"({cam_expr} LIKE ? ESCAPE '\\')", [pat]

    # 10. Rating (0 = Unrated, 1..5 = Stars)
    if raw_field == "rating":
        if val_str.lower() == "unrated":
            r_num = 0
        else:
            m_r = re.search(r"-?\d+", val_str)
            if not m_r:
                return None
            r_num = int(m_r.group(0))
        r_expr = f"COALESCE({alias}.rating, 0)"
        if op in ("gte", ">="):
            return f"({r_expr} >= ?)", [r_num]
        if op in ("lte", "<="):
            return f"({r_expr} <= ?)", [r_num]
        if op in ("neq", "!=", "is_not"):
            return f"({r_expr} != ?)", [r_num]
        return f"({r_expr} = ?)", [r_num]

    # 11. Flag
    if raw_field in ("flag", "flagged"):
        if op == "is_rejected" or (op in ("eq", "is") and val_str.lower() in ("-1", "rejected")):
            return f"COALESCE({alias}.is_rejected, 0) = 1", []
        if op == "is_unflagged" or (op in ("eq", "is") and val_str.lower() in ("0", "false", "unflagged", "no")):
            return f"(COALESCE({alias}.flag, 0) = 0 AND COALESCE({alias}.is_rejected, 0) = 0)", []
        return f"COALESCE({alias}.flag, 0) = 1", []

    # 12. Media Type
    if raw_field == "media_type":
        mt = val_str.lower()
        neg = op in ("is_not", "neq", "!=")
        if mt in ("video", "videos"):
            return (f"{alias}.is_video = 0" if neg else f"{alias}.is_video = 1"), []
        if mt == "raw":
            raw_set = ",".join(f"'{e}'" for e in RAW_EXTENSIONS)
            sql = f"({alias}.is_video = 0 AND LOWER(LTRIM(COALESCE({alias}.ext, ''), '.')) IN ({raw_set}))"
            return (f"NOT {sql}" if neg else sql), []
        raw_set = ",".join(f"'{e}'" for e in RAW_EXTENSIONS)
        photo_sql = f"({alias}.is_video = 0 AND LOWER(LTRIM(COALESCE({alias}.ext, ''), '.')) NOT IN ({raw_set}))"
        return (f"NOT {photo_sql}" if neg else photo_sql), []

    # 13. File Extension
    if raw_field in ("file_ext", "ext", "file_type"):
        ext_norm = val_str.lower().lstrip(".")
        if not ext_norm:
            return None
        if ext_norm == "raw":
            raw_set = ",".join(f"'{e}'" for e in RAW_EXTENSIONS)
            sql = f"LOWER(LTRIM(COALESCE({alias}.ext, ''), '.')) IN ({raw_set})"
            return (f"NOT ({sql})" if op in ("is_not", "neq", "!=") else sql), []
        if op in ("is_not", "neq", "!="):
            return f"LOWER(LTRIM(COALESCE({alias}.ext, ''), '.')) != ?", [ext_norm]
        return f"LOWER(LTRIM(COALESCE({alias}.ext, ''), '.')) = ?", [ext_norm]

    # 14. Orientation
    if raw_field == "orientation":
        ori = val_str.lower()
        dims_ok = f"(COALESCE({alias}.width, 0) > 0 AND COALESCE({alias}.height, 0) > 0)"
        neg = op in ("is_not", "neq", "!=")
        if "portrait" in ori:
            sql = f"({dims_ok} AND {alias}.height > {alias}.width)"
        elif "square" in ori:
            sql = f"({dims_ok} AND {alias}.width = {alias}.height)"
        else:
            sql = f"({dims_ok} AND {alias}.width > {alias}.height)"
        return (f"NOT {sql}" if neg else sql), []

    # 15. File Size (MB)
    if raw_field in ("file_size_mb", "file_size", "size_mb"):
        return _compile_numeric_expr_rule(
            f"COALESCE({alias}.size, 0)", op, val, scale=1024.0 * 1024.0
        )

    # 16. Megapixels
    if raw_field == "megapixels":
        mp_expr = (
            f"((CAST(COALESCE({alias}.width, 0) AS REAL) * CAST(COALESCE({alias}.height, 0) AS REAL)) / 1000000.0)"
        )
        return _compile_numeric_expr_rule(mp_expr, op, val, scale=1.0)

    # 17. Folder (recursive subtree by ID, or folder name/path match)
    if raw_field == "folder":
        if not val_str:
            return None
        neg = op in ("not_in_folder", "excludes", "is_not")
        if isinstance(val, int) or val_str.isdigit():
            fid = int(val_str)
            sub = (
                f"{alias}.folder_id IN ("
                "WITH RECURSIVE subfolder_tree(id) AS ("
                "SELECT ? UNION ALL "
                "SELECT f.id FROM folders f JOIN subfolder_tree st ON f.parent_id = st.id"
                ") SELECT id FROM subfolder_tree)"
            )
            return (f"NOT ({sub})" if neg else sub), [fid]
        pat = f"%{_escape_like_term(val_str.lower())}%"
        sub = (
            f"{alias}.folder_id IN ("
            "WITH RECURSIVE subfolder_tree(id) AS ("
            "SELECT id FROM folders WHERE LOWER(name) LIKE ? ESCAPE '\\' OR LOWER(path) LIKE ? ESCAPE '\\' "
            "UNION ALL "
            "SELECT f.id FROM folders f JOIN subfolder_tree st ON f.parent_id = st.id"
            ") SELECT id FROM subfolder_tree)"
        )
        return (f"NOT ({sub})" if neg else sub), [pat, pat]

    # 18. Caption / Filename
    if raw_field in ("caption", "filename", "caption_filename"):
        if raw_field == "caption":
            cols = [f"COALESCE({alias}.caption, '')"]
        elif raw_field == "filename":
            cols = [f"COALESCE({alias}.filename, '')"]
        else:
            cols = [f"COALESCE({alias}.caption, '')", f"COALESCE({alias}.filename, '')"]

        if op == "is_empty":
            return "(" + " AND ".join(f"TRIM({c}) = ''" for c in cols) + ")", []
        if op == "is_not_empty":
            return "(" + " OR ".join(f"TRIM({c}) != ''" for c in cols) + ")", []
        if not val_str:
            return None
        pat = f"%{_escape_like_term(val_str.lower())}%"
        likes = [f"LOWER({c}) LIKE ? ESCAPE '\\'" for c in cols]
        joined = "(" + " OR ".join(likes) + ")"
        params = [pat] * len(cols)
        if op in ("not_contains", "excludes"):
            return f"NOT {joined}", params
        return joined, params

    # 19. Camera Manufacturer (camera_make)
    if raw_field in ("camera_make", "camera_manufacturer", "make"):
        if op == "is_empty":
            return f"COALESCE(TRIM({alias}.camera_make), '') = ''", []
        if op == "is_not_empty":
            return f"COALESCE(TRIM({alias}.camera_make), '') != ''", []
        if not val_str:
            return None
        make_expr = f"LOWER(TRIM(COALESCE({alias}.camera_make, '')))"
        if op in ("equals", "eq", "is"):
            return f"{make_expr} = LOWER(?)", [val_str]
        pat = f"%{_escape_like_term(val_str.lower())}%"
        if op in ("not_contains", "excludes"):
            return f"({make_expr} NOT LIKE ? ESCAPE '\\')", [pat]
        return f"({make_expr} LIKE ? ESCAPE '\\')", [pat]

    # 19b. OCR Text (Text in Photo)
    if raw_field in ("ocr_text", "ocr", "text_in_photo"):
        if op == "is_empty":
            return f"TRIM(COALESCE({alias}.ocr_text, '')) = ''", []
        if op == "is_not_empty":
            return f"TRIM(COALESCE({alias}.ocr_text, '')) != ''", []
        if not val_str:
            return None
        pat = f"%{_escape_like_term(val_str.lower())}%"
        sql = f"LOWER(COALESCE({alias}.ocr_text, '')) LIKE ? ESCAPE '\\'"
        if op in ("not_contains", "excludes"):
            return f"NOT ({sql})", [pat]
        return sql, [pat]

    # 20. Number of Detected Faces (confirmed state=1 or unconfirmed suggestion state=0, excluding rejected state=2)
    if raw_field in ("face_count", "detected_faces"):
        face_cnt_expr = (
            f"(SELECT COUNT(*) FROM media_people mp "
            f"WHERE mp.media_id = {alias}.id AND COALESCE(mp.state, 0) IN (0, 1))"
        )
        if op == "between":
            pair = _parse_numeric_pair(val)
            if not pair:
                return None
            lo_f, hi_f = int(min(pair[0], pair[1])), int(max(pair[0], pair[1]))
            return f"({face_cnt_expr} BETWEEN ? AND ?)", [max(0, lo_f), max(0, hi_f)]
        m_fc = re.search(r"-?\d+", val_str)
        if not m_fc:
            return None
        fc_num = max(0, int(m_fc.group(0)))
        if op in ("gte", ">=", "at_least"):
            return f"({face_cnt_expr} >= ?)", [fc_num]
        if op in ("lte", "<=", "at_most"):
            return f"({face_cnt_expr} <= ?)", [fc_num]
        if op in ("neq", "!=", "is_not"):
            return f"({face_cnt_expr} != ?)", [fc_num]
        return f"({face_cnt_expr} = ?)", [fc_num]

    # 21. Favourite (♥)
    if raw_field in ("favourite", "favorite", "is_favourite", "is_favorite"):
        if op in ("is_not_favourite", "is_not_favorite", "is_not", "neq", "!=", "false", "no"):
            want_fav = False
        elif op in ("is_favourite", "is_favorite", "is", "eq", "=", "true", "yes", ""):
            want_fav = True
        else:
            want_fav = val_str.lower() in ("1", "true", "yes", "favourite", "favorite", "")
        if val_str.lower() in ("0", "false", "no"):
            want_fav = not want_fav
        return f"COALESCE({alias}.is_favourite, 0) = ?", [1 if want_fav else 0]

    # 22. Rejected Culling Flag
    if raw_field in ("rejected", "is_rejected"):
        if op in ("is_not_rejected", "is_not", "neq", "!=", "false", "no"):
            want_rej = False
        elif op in ("is_rejected", "is", "eq", "=", "true", "yes", ""):
            want_rej = True
        else:
            want_rej = val_str.lower() in ("1", "true", "yes", "rejected", "")
        if val_str.lower() in ("0", "false", "no"):
            want_rej = not want_rej
        return f"COALESCE({alias}.is_rejected, 0) = ?", [1 if want_rej else 0]

    return None


def compile_smart_album_where(
    rules_dict: dict[str, Any] | str, alias: str = "m"
) -> tuple[str, list[Any]]:
    """
    Compile a Boolean Smart Album rule tree into parameterized SQL WHERE SQL.

    Supports ``match_mode`` ("all" | "any") over ``rules`` rows across all 17+
    grouping dimensions (people, tag, place, time_of_day, date_taken, year, month,
    on_this_day, camera, rating, flag, media_type, file_ext, orientation,
    file_size_mb, megapixels, folder, caption/filename).
    """
    if isinstance(rules_dict, str):
        try:
            parsed = json.loads(rules_dict)
        except Exception:
            return "1 = 1", []
    else:
        parsed = rules_dict or {}

    match_mode = str(parsed.get("match_mode") or "all").strip().lower()
    raw_rules = parsed.get("rules") or []

    compiled_sql: list[str] = []
    params: list[Any] = []
    for r in raw_rules:
        if not isinstance(r, dict):
            continue
        res = _compile_single_smart_rule(r, alias=alias)
        if res is not None:
            sql_part, p_part = res
            compiled_sql.append(f"({sql_part})")
            params.extend(p_part)

    if not compiled_sql:
        return ("1 = 0" if raw_rules else "1 = 1"), []

    combiner = " OR " if match_mode == "any" else " AND "
    return combiner.join(compiled_sql), params
