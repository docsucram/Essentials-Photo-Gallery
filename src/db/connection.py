"""
Dual-Database SQLite Engine (catalog.db & thumbnails.db).
Manages thread-local WAL connections, PRAGMA tuning, and schema initialization.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
from collections.abc import Generator
from pathlib import Path

from src.utils.paths import get_schema_path

logger = logging.getLogger(__name__)

THUMBNAILS_SCHEMA_SPLIT_MARKER = "-- === THUMBNAILS_DB_SCHEMA ==="


class DualDatabaseEngine:
    """Thread-safe, thread-local SQLite WAL connection manager for catalog.db and thumbnails.db."""

    def __init__(
        self,
        catalog_path: Path | str,
        thumbnails_path: Path | str,
        schema_sql_path: Path | str | None = None,
        auto_init: bool = True,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.thumbnails_path = Path(thumbnails_path)
        if schema_sql_path is None:
            self.schema_sql_path = get_schema_path()
        else:
            self.schema_sql_path = Path(schema_sql_path)

        self._local = threading.local()
        self._lock = threading.Lock()
        self._all_connections: list[sqlite3.Connection] = []
        self.write_lock = threading.RLock()
        self.thumb_write_lock = threading.RLock()

        if str(self.catalog_path) != ":memory:":
            self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        if str(self.thumbnails_path) != ":memory:":
            self.thumbnails_path.parent.mkdir(parents=True, exist_ok=True)

        if auto_init:
            self.initialize_schemas()

    def _configure_catalog_conn(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA cache_size = -64000;")
        conn.execute("PRAGMA mmap_size = 268435456;")

    def _configure_thumbnails_conn(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA cache_size = -32000;")
        conn.execute("PRAGMA mmap_size = 268435456;")

    def catalog_conn(self) -> sqlite3.Connection:
        """Returns the current thread's SQLite connection to catalog.db."""
        conn: sqlite3.Connection | None = getattr(self._local, "catalog_conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.catalog_path),
                check_same_thread=False,
                timeout=5.0,
            )
            conn.row_factory = sqlite3.Row
            self._configure_catalog_conn(conn)
            self._local.catalog_conn = conn
            with self._lock:
                self._all_connections.append(conn)
        return conn

    @contextlib.contextmanager
    def catalog_transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding the thread's catalog_conn with write_lock and atomic transaction commit/rollback."""
        with self.write_lock:
            conn = self.catalog_conn()
            with conn:
                yield conn

    def thumbnails_conn(self) -> sqlite3.Connection:
        """Returns the current thread's SQLite connection to thumbnails.db."""
        conn: sqlite3.Connection | None = getattr(self._local, "thumbnails_conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.thumbnails_path),
                check_same_thread=False,
                timeout=5.0,
            )
            conn.row_factory = sqlite3.Row
            self._configure_thumbnails_conn(conn)
            self._local.thumbnails_conn = conn
            with self._lock:
                self._all_connections.append(conn)
        return conn

    @contextlib.contextmanager
    def thumbnails_transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding the thread's thumbnails_conn with atomic transaction commit/rollback."""
        with self.thumb_write_lock:
            conn = self.thumbnails_conn()
            with conn:
                yield conn

    def initialize_schemas(self) -> None:
        """Executes db/schema.sql against catalog.db and thumbnails.db."""
        raw_sql = self.schema_sql_path.read_text(encoding="utf-8")
        if THUMBNAILS_SCHEMA_SPLIT_MARKER in raw_sql:
            catalog_ddl, thumbnails_ddl = raw_sql.split(THUMBNAILS_SCHEMA_SPLIT_MARKER, 1)
        else:
            catalog_ddl, thumbnails_ddl = raw_sql, ""

        with self.write_lock:
            cat_conn = self.catalog_conn()
            self._migrate_catalog_schema(cat_conn)
            cat_conn.executescript(catalog_ddl)
            cat_conn.commit()
            self._configure_catalog_conn(cat_conn)

            if thumbnails_ddl.strip():
                thumb_conn = self.thumbnails_conn()
                self._migrate_thumbnails_schema(thumb_conn)
                thumb_conn.executescript(thumbnails_ddl)
                thumb_conn.commit()
                self._configure_thumbnails_conn(thumb_conn)

    def _migrate_catalog_schema(self, conn: sqlite3.Connection) -> None:
        """Adds any newly defined columns to existing tables in catalog.db."""
        cur = conn.cursor()
        cur.row_factory = None

        # 1. People table columns
        try:
            people_cols = {r[1] for r in cur.execute("PRAGMA table_info(people);").fetchall()}
            if people_cols:
                if "avatar_media_people_id" not in people_cols:
                    cur.execute("ALTER TABLE people ADD COLUMN avatar_media_people_id INTEGER REFERENCES media_people(id) ON DELETE SET NULL;")
                if "centroid_embedding" not in people_cols:
                    cur.execute("ALTER TABLE people ADD COLUMN centroid_embedding BLOB;")
                if "confirmed_count" not in people_cols:
                    cur.execute("ALTER TABLE people ADD COLUMN confirmed_count INTEGER NOT NULL DEFAULT 0;")
        except Exception:
            pass

        # 2. Media_people table columns & nullable person_id migration
        try:
            mp_info = cur.execute("PRAGMA table_info(media_people);").fetchall()
            if mp_info:
                mp_cols = {r[1] for r in mp_info}
                # Ensure missing columns exist first
                if "suggested_person_id" not in mp_cols:
                    cur.execute("ALTER TABLE media_people ADD COLUMN suggested_person_id INTEGER REFERENCES people(id) ON DELETE SET NULL;")
                if "similarity_score" not in mp_cols:
                    cur.execute("ALTER TABLE media_people ADD COLUMN similarity_score REAL;")
                if "state" not in mp_cols:
                    cur.execute("ALTER TABLE media_people ADD COLUMN state INTEGER NOT NULL DEFAULT 0;")
                if "embedding" not in mp_cols:
                    cur.execute("ALTER TABLE media_people ADD COLUMN embedding BLOB;")
                if "is_manual" not in mp_cols:
                    cur.execute("ALTER TABLE media_people ADD COLUMN is_manual INTEGER NOT NULL DEFAULT 0;")

                # Check if person_id has legacy NOT NULL constraint (r[3] is notnull flag)
                person_id_info = next((r for r in mp_info if r[1] == "person_id"), None)
                if person_id_info and person_id_info[3] == 1:
                    cur.execute("PRAGMA foreign_keys = OFF;")
                    cur.execute("DROP TRIGGER IF EXISTS trg_media_people_dirty_insert;")
                    cur.execute("DROP TRIGGER IF EXISTS trg_media_people_dirty_update;")
                    cur.execute("DROP TRIGGER IF EXISTS trg_media_people_dirty_delete;")
                    cur.execute("DROP TABLE IF EXISTS media_people_rebuild;")
                    cur.execute("""
                        CREATE TABLE media_people_rebuild (
                            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                            media_id            INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
                            person_id           INTEGER REFERENCES people(id) ON DELETE SET NULL,
                            suggested_person_id INTEGER REFERENCES people(id) ON DELETE SET NULL,
                            similarity_score    REAL,
                            state               INTEGER NOT NULL DEFAULT 0,
                            is_manual           INTEGER NOT NULL DEFAULT 0,
                            rect_x              REAL NOT NULL,
                            rect_y              REAL NOT NULL,
                            rect_w              REAL NOT NULL,
                            rect_h              REAL NOT NULL,
                            embedding           BLOB
                        );
                    """)
                    cur.execute("""
                        INSERT INTO media_people_rebuild (
                            id, media_id, person_id, suggested_person_id, similarity_score,
                            state, is_manual, rect_x, rect_y, rect_w, rect_h, embedding
                        )
                        SELECT
                            id, media_id, person_id, suggested_person_id, similarity_score,
                            state, is_manual, rect_x, rect_y, rect_w, rect_h, embedding
                        FROM media_people;
                    """)
                    cur.execute("DROP TABLE media_people;")
                    cur.execute("ALTER TABLE media_people_rebuild RENAME TO media_people;")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_media_people_person_media ON media_people(person_id, media_id);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_media_people_media_id ON media_people(media_id);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_media_people_suggested ON media_people(suggested_person_id, similarity_score DESC) WHERE state = 0;")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_media_people_state ON media_people(state);")
                    cur.execute("PRAGMA foreign_keys = ON;")
                    conn.commit()

            cur.execute("DROP TRIGGER IF EXISTS trg_media_people_dirty_update;")
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_media_people_dirty_update
                AFTER UPDATE OF person_id, rect_x, rect_y, rect_w, rect_h, state ON media_people
                BEGIN
                    UPDATE media SET metadata_dirty = 1 WHERE id = NEW.media_id;
                END;
            """)
        except Exception as exc:
            logger.warning("Error migrating media_people schema: %s", exc)

        # 3. Data Migration: Mark existing manual/imported face tags as confirmed (state = 1)
        try:
            cur.execute(
                """
                UPDATE media_people
                SET state = 1, is_manual = 1
                WHERE person_id IS NOT NULL AND state = 0 AND suggested_person_id IS NULL;
                """
            )
            cur.execute(
                """
                UPDATE people
                SET confirmed_count = (
                    SELECT COUNT(*)
                    FROM media_people mp
                    WHERE mp.person_id = people.id AND mp.state = 1
                );
                """
            )
        except Exception:
            pass

        # 4. Face rejection tables
        try:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS face_rejected_suggestions (
                    face_id INTEGER NOT NULL REFERENCES media_people(id) ON DELETE CASCADE,
                    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    PRIMARY KEY (face_id, person_id)
                ) WITHOUT ROWID;
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS face_rejections (
                    face_id INTEGER NOT NULL REFERENCES media_people(id) ON DELETE CASCADE,
                    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    PRIMARY KEY (face_id, person_id)
                ) WITHOUT ROWID;
                """
            )
        except Exception:
            pass

        # 5. Scale Indexes (<10ms 400k covering index)
        try:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_effective_date ON media(COALESCE(date_taken, mtime, 0) DESC, id ASC);"
            )
        except Exception:
            pass

        # 6. Media table columns: face_scan_version, is_missing
        try:
            media_cols = {r[1] for r in cur.execute("PRAGMA table_info(media);").fetchall()}
            if media_cols:
                if "face_scan_version" not in media_cols:
                    cur.execute("ALTER TABLE media ADD COLUMN face_scan_version INTEGER NOT NULL DEFAULT 0;")
                if "is_missing" not in media_cols:
                    cur.execute("ALTER TABLE media ADD COLUMN is_missing INTEGER NOT NULL DEFAULT 0;")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_media_face_scan ON media(face_scan_version) WHERE is_video = 0;")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_media_is_missing ON media(is_missing) WHERE is_missing = 1;")
        except Exception:
            pass

        # 7. One-time video re-index: force mtime = 0 so the next sync re-extracts true
        #    date_taken (mvhd/XMP/filename), embedded XMP rating, duration, fps, and orientation.
        #    Guarded by a flag table so it does not repeat on every launch.
        try:
            cur.execute("CREATE TABLE IF NOT EXISTS schema_flags (name TEXT PRIMARY KEY);")
            done = cur.execute(
                "SELECT 1 FROM schema_flags WHERE name = 'video_metadata_reindex_v2';"
            ).fetchone()
            if not done:
                if cur.execute("PRAGMA table_info(media);").fetchall():
                    cur.execute("UPDATE media SET mtime = 0 WHERE is_video = 1;")
                cur.execute(
                    "INSERT OR IGNORE INTO schema_flags(name) VALUES ('video_metadata_reindex_v2');"
                )
        except Exception:
            pass

        # 8. One-time clean re-import of confirmed face regions:
        #    Ensures 100% deterministic coordinates from XMP on disk, clears metadata_dirty,
        #    and purges stale face thumbnails so Stage 1 backfill can generate clean crops.
        try:
            cur.execute("CREATE TABLE IF NOT EXISTS schema_flags (name TEXT PRIMARY KEY);")
            done = cur.execute(
                "SELECT 1 FROM schema_flags WHERE name = 'clean_xmp_media_people_v3';"
            ).fetchone()
            if not done:
                cur.execute("UPDATE media SET metadata_dirty = 0;")

                # Fetch all media records with confirmed media_people
                media_rows = cur.execute("""
                    SELECT DISTINCT m.id, m.filename, m.orientation, f.path AS folder_path
                    FROM media m
                    JOIN folders f ON f.id = m.folder_id
                    JOIN media_people mp ON mp.media_id = m.id
                    WHERE mp.state = 1 AND m.is_missing = 0;
                """).fetchall()

                # Invalidate stale face thumbnails
                try:
                    thumb_path = str(self.thumbnails_path.resolve())
                    cur.execute("ATTACH DATABASE ? AS thumb_db;", (thumb_path,))
                    cur.execute(
                        "DELETE FROM thumb_db.face_thumbnails WHERE face_id IN (SELECT id FROM media_people WHERE state = 1);"
                    )
                    cur.execute("DETACH DATABASE thumb_db;")
                except Exception:
                    pass

                from src.workers.scanner_thumb_worker import extract_media_metadata

                for m_row in media_rows:
                    mid = int(m_row[0])
                    fname = str(m_row[1])
                    folder_path = str(m_row[3])
                    fpath = Path(folder_path) / fname
                    if not fpath.exists():
                        continue
                    try:
                        meta = extract_media_metadata(fpath, is_video=False)
                        xmp_people = meta.get("people", [])
                        if not xmp_people:
                            continue

                        existing_mp = cur.execute("""
                            SELECT mp.id, p.name
                            FROM media_people mp
                            JOIN people p ON p.id = mp.person_id
                            WHERE mp.media_id = ? AND mp.state = 1;
                        """, (mid,)).fetchall()

                        remaining_xmp = list(xmp_people)
                        for emp in existing_mp:
                            mp_id = int(emp[0])
                            pname = (emp[1] or "").strip().lower()
                            matched_idx = next(
                                (i for i, xp in enumerate(remaining_xmp) if (xp.get("name") or "").strip().lower() == pname),
                                None,
                            )
                            if matched_idx is not None:
                                matched_xp = remaining_xmp.pop(matched_idx)
                                rect = matched_xp["rect"]
                                cur.execute("""
                                    UPDATE media_people
                                    SET rect_x = ?, rect_y = ?, rect_w = ?, rect_h = ?, embedding = NULL
                                    WHERE id = ?;
                                """, (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]), mp_id))

                        cur.execute("UPDATE media SET face_scan_version = 0 WHERE id = ?;", (mid,))
                    except Exception:
                        pass

                cur.execute("UPDATE media SET metadata_dirty = 0;")

                cur.execute(
                    "INSERT OR IGNORE INTO schema_flags(name) VALUES ('clean_xmp_media_people_v3');"
                )
        except Exception as exc:
            logger.warning("Error in clean_xmp_media_people_v3 migration: %s", exc)

        # 9. One-time deduplication of redundant flat root tags:
        try:
            cur.execute("CREATE TABLE IF NOT EXISTS schema_flags (name TEXT PRIMARY KEY);")
            done = cur.execute(
                "SELECT 1 FROM schema_flags WHERE name = 'dedup_media_tags_v2';"
            ).fetchone()
            if not done:
                if cur.execute("PRAGMA table_info(media_tags);").fetchall():
                    cur.execute(
                        """
                        DELETE FROM media_tags
                        WHERE (media_id, tag_id) IN (
                            SELECT mt_root.media_id, mt_root.tag_id
                            FROM media_tags mt_root
                            JOIN tags t_root ON mt_root.tag_id = t_root.id
                            JOIN media_tags mt_hier ON mt_root.media_id = mt_hier.media_id AND mt_root.tag_id != mt_hier.tag_id
                            JOIN tags t_hier ON mt_hier.tag_id = t_hier.id
                            WHERE t_root.parent_id IS NULL
                              AND t_hier.parent_id IS NOT NULL
                              AND LOWER(t_root.name) = LOWER(t_hier.name)
                        );
                        """
                    )
                    cur.execute(
                        """
                        DELETE FROM tags
                        WHERE parent_id IS NULL
                          AND id NOT IN (SELECT DISTINCT tag_id FROM media_tags)
                          AND id NOT IN (SELECT DISTINCT parent_id FROM tags WHERE parent_id IS NOT NULL);
                        """
                    )
                cur.execute(
                    "INSERT OR IGNORE INTO schema_flags(name) VALUES ('dedup_media_tags_v2');"
                )
        except Exception as exc:
            logger.warning("Error in dedup_media_tags_v2 migration: %s", exc)

        # 10. One-time cleanup of boxless face crops (16x16 corner crops) and unscanned face versions:
        try:
            has_people = cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='people';"
            ).fetchone()
            cur.execute("CREATE TABLE IF NOT EXISTS schema_flags (name TEXT PRIMARY KEY);")
            done = cur.execute(
                "SELECT 1 FROM schema_flags WHERE name = 'heal_boxless_and_unscanned_faces_v1';"
            ).fetchone()
            if has_people and not done:
                try:
                    thumb_path = str(self.thumbnails_path.resolve())
                    cur.execute("ATTACH DATABASE ? AS thumb_db;", (thumb_path,))
                    cur.execute(
                        """
                        DELETE FROM thumb_db.face_thumbnails
                        WHERE face_id IN (
                            SELECT id FROM media_people WHERE rect_w <= 0.005 OR rect_h <= 0.005
                        );
                        """
                    )
                    cur.execute("DETACH DATABASE thumb_db;")
                except Exception:
                    pass

                # Clear people avatars that came from boxless or unverified rows
                cur.execute(
                    """
                    UPDATE people
                    SET thumbnail_blob = NULL
                    WHERE id IN (
                        SELECT DISTINCT person_id FROM media_people
                        WHERE person_id IS NOT NULL AND (rect_w <= 0.005 OR rect_h <= 0.005)
                    )
                    AND id NOT IN (
                        SELECT DISTINCT person_id FROM media_people
                        WHERE person_id IS NOT NULL AND rect_w > 0.005 AND rect_h > 0.005 AND embedding IS NOT NULL
                    );
                    """
                )

                # If photos were marked scanned while FaceEngine was unavailable (0 embeddings), reset scan version
                emb_cnt_row = cur.execute(
                    "SELECT COUNT(*) FROM media_people WHERE embedding IS NOT NULL;"
                ).fetchone()
                emb_cnt = int(emb_cnt_row[0]) if emb_cnt_row else 0
                if emb_cnt == 0:
                    cur.execute("UPDATE media SET face_scan_version = 0 WHERE face_scan_version > 0;")
                else:
                    cur.execute(
                        """
                        UPDATE media
                        SET face_scan_version = 0
                        WHERE id IN (
                            SELECT DISTINCT media_id FROM media_people
                            WHERE state = 1 AND (rect_w <= 0.005 OR rect_h <= 0.005)
                        );
                        """
                    )

                cur.execute(
                    "INSERT OR IGNORE INTO schema_flags(name) VALUES ('heal_boxless_and_unscanned_faces_v1');"
                )
        except Exception as exc:
            logger.warning("Error in heal_boxless_and_unscanned_faces_v1 migration: %s", exc)

        # 11. One-time healing of EXIF SubIFD DateTimeOriginal (aligns with camera capture date):
        try:
            has_media = cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='media';"
            ).fetchone()
            cur.execute("CREATE TABLE IF NOT EXISTS schema_flags (name TEXT PRIMARY KEY);")
            done = cur.execute(
                "SELECT 1 FROM schema_flags WHERE name = 'heal_exif_subifd_datetime_original_v1';"
            ).fetchone()
            if has_media and not done:
                from PIL import Image, ExifTags
                from src.workers.scanner_thumb_worker import _parse_iso_or_exif_datetime

                media_rows = cur.execute("""
                    SELECT m.id, f.path, m.filename, m.date_taken
                    FROM media m
                    JOIN folders f ON f.id = m.folder_id
                    WHERE m.is_video = 0;
                """).fetchall()

                for mid, fpath, fname, db_dt in media_rows:
                    full_p = Path(fpath) / fname
                    if not full_p.exists():
                        continue
                    try:
                        with Image.open(full_p) as img:
                            ex = img.getexif()
                            if not ex:
                                continue
                            ifd = ex.get_ifd(ExifTags.IFD.Exif)
                            if not ifd:
                                continue
                            dto = ifd.get(36867) or ifd.get(36868)
                            if dto:
                                exif_ts = _parse_iso_or_exif_datetime(str(dto))
                                if exif_ts and (db_dt is None or abs(exif_ts - db_dt) > 60):
                                    cur.execute("UPDATE media SET date_taken = ? WHERE id = ?;", (exif_ts, mid))
                    except Exception:
                        pass

                cur.execute(
                    "INSERT OR IGNORE INTO schema_flags(name) VALUES ('heal_exif_subifd_datetime_original_v1');"
                )
                conn.commit()
        except Exception as exc:
            logger.warning("Error in heal_exif_subifd_datetime_original_v1 migration: %s", exc)

        # 12. One-time healing of bogus filename/QuickTime/Exiv2 dates (e.g. 1055 Facebook IDs,
        #     1904 QuickTime epoch zero, 1946-1948 Exiv2 66-year shift, 1969-12-31 Unix epoch zero):
        try:
            has_media = cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='media';"
            ).fetchone()
            cur.execute("CREATE TABLE IF NOT EXISTS schema_flags (name TEXT PRIMARY KEY);")
            done = cur.execute(
                "SELECT 1 FROM schema_flags WHERE name = 'heal_bogus_video_and_filename_dates_v1';"
            ).fetchone()
            if has_media and not done:
                from src.workers.scanner_thumb_worker import (
                    _MAC_EPOCH_OFFSET,
                    _MAX_VALID_CAMERA_EPOCH,
                    _MIN_VALID_CAMERA_EPOCH,
                    _extract_mp4_creation_time,
                    _timestamp_from_filename,
                    extract_media_metadata,
                )

                bad_rows = cur.execute(
                    """
                    SELECT m.id, f.path, m.filename, m.is_video, m.date_taken
                    FROM media m
                    JOIN folders f ON f.id = m.folder_id
                    WHERE m.date_taken IS NOT NULL
                      AND (m.date_taken < -2000000000 OR (m.is_video = 1 AND m.date_taken < 315532800));
                    """
                ).fetchall()

                for mid, fpath, fname, is_vid, old_dt in bad_rows:
                    full_p = Path(fpath) / fname
                    new_dt: int | None = None
                    if is_vid:
                        mvhd_dt = _extract_mp4_creation_time(full_p) if full_p.exists() else None
                        fn_dt = _timestamp_from_filename(Path(fname).stem)
                        recovered: int | None = None
                        if old_dt is not None and abs(int(old_dt) - (-2082844800)) > 172800 and abs(int(old_dt)) > 86400:
                            shifted = int(old_dt) + _MAC_EPOCH_OFFSET
                            if _MIN_VALID_CAMERA_EPOCH <= shifted <= _MAX_VALID_CAMERA_EPOCH:
                                recovered = shifted
                        if mvhd_dt is not None and fn_dt is not None:
                            if abs(mvhd_dt - fn_dt) <= 90000 or (mvhd_dt % 3600 == 0 and fn_dt % 60 != 0):
                                new_dt = fn_dt
                            else:
                                new_dt = mvhd_dt
                        else:
                            new_dt = mvhd_dt or fn_dt or recovered
                    else:
                        if full_p.exists():
                            try:
                                new_dt = extract_media_metadata(full_p, False).get("date_taken")
                            except Exception:
                                new_dt = None
                    cur.execute("UPDATE media SET date_taken = ? WHERE id = ?;", (new_dt, int(mid)))

                cur.execute(
                    "INSERT OR IGNORE INTO schema_flags(name) VALUES ('heal_bogus_video_and_filename_dates_v1');"
                )
                conn.commit()
        except Exception as exc:
            logger.warning("Error in heal_bogus_video_and_filename_dates_v1 migration: %s", exc)

        conn.commit()

    def _migrate_thumbnails_schema(self, conn: sqlite3.Connection) -> None:
        """Ensures face_thumbnails table exists and invalidates old video thumbnails for re-orientation."""
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS face_thumbnails (
                    face_id INTEGER PRIMARY KEY,
                    crop_webp BLOB NOT NULL
                );
                """
            )
            conn.commit()
        except Exception:
            pass

        try:
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS schema_flags (name TEXT PRIMARY KEY);")
            done = cur.execute(
                "SELECT 1 FROM schema_flags WHERE name = 'video_thumbnails_reorient_v2';"
            ).fetchone()
            if not done:
                cat_path = str(self.catalog_path.resolve())
                try:
                    conn.execute("ATTACH DATABASE ? AS cat_db;", (cat_path,))
                    conn.execute(
                        "DELETE FROM thumbnails WHERE media_id IN (SELECT id FROM cat_db.media WHERE is_video = 1);"
                    )
                    conn.execute("DETACH DATABASE cat_db;")
                except Exception:
                    pass
                cur.execute(
                    "INSERT OR IGNORE INTO schema_flags(name) VALUES ('video_thumbnails_reorient_v2');"
                )
                conn.commit()
        except Exception:
            pass

    def analyze_catalog(self) -> None:
        """Runs ANALYZE on catalog.db to update sqlite_stat1 query planner statistics."""
        with self.write_lock:
            conn = self.catalog_conn()
            conn.execute("ANALYZE;")
            conn.commit()

    def backup_catalog(
        self,
        destination_path: Path | str,
        pages: int = 250,
        progress_callback: Any = None,
    ) -> None:
        """
        Creates a consistent, online snapshot of catalog.db using SQLite's backup API.
        Safe to run concurrently with active WAL readers and writers.
        """
        dest_p = Path(destination_path)
        dest_p.parent.mkdir(parents=True, exist_ok=True)
        dest_conn = sqlite3.connect(str(dest_p), timeout=15.0)
        try:
            with self.write_lock:
                src_conn = self.catalog_conn()
                src_conn.backup(dest_conn, pages=pages, progress=progress_callback)
        finally:
            dest_conn.close()

    def close_thread_connections(self) -> None:
        """Closes SQLite connections owned by the calling thread, running PRAGMA optimize."""
        for attr in ("catalog_conn", "thumbnails_conn"):
            conn: sqlite3.Connection | None = getattr(self._local, attr, None)
            if conn is not None:
                setattr(self._local, attr, None)
                with self._lock:
                    if conn in self._all_connections:
                        self._all_connections.remove(conn)
                try:
                    conn.execute("PRAGMA optimize;")
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass

    def close_all(self) -> None:
        """Closes all tracked SQLite connections across all threads, running PRAGMA optimize."""
        with self._lock:
            conns = list(self._all_connections)
            self._all_connections.clear()
        for attr in ("catalog_conn", "thumbnails_conn"):
            if hasattr(self._local, attr):
                delattr(self._local, attr)
        for conn in conns:
            try:
                conn.execute("PRAGMA optimize;")
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
