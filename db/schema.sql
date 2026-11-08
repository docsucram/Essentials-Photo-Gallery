-- ============================================================================
-- PhotoGallery64 (WLPG Clone) — Complete SQLite Schema
-- Engine: SQLite 3.35+ (WAL Mode, R*Tree enabled)
-- Databases: catalog.db (Metadata, Hierarchy, Spatial) & thumbnails.db (WebP BLOBs)
-- ============================================================================

-- ============================================================================
-- PART 1: CATALOG DATABASE (catalog.db)
-- ============================================================================

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA temp_store = MEMORY;
PRAGMA cache_size = -64000;      -- 64 MB page cache
PRAGMA mmap_size = 268435456;    -- 256 MB memory-mapped I/O

-- ----------------------------------------------------------------------------
-- 1. Folders Hierarchy
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS folders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    path            TEXT    NOT NULL UNIQUE,
    name            TEXT    NOT NULL,
    parent_id       INTEGER REFERENCES folders(id) ON DELETE CASCADE,
    mtime           INTEGER NOT NULL DEFAULT 0,
    is_favourite    INTEGER NOT NULL DEFAULT 0 CHECK (is_favourite IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_folders_parent_id
    ON folders(parent_id);

CREATE INDEX IF NOT EXISTS idx_folders_favourite
    ON folders(is_favourite) WHERE is_favourite = 1;

-- ----------------------------------------------------------------------------
-- 2. Media Catalog (500,000+ Photos & Videos)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS media (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    folder_id       INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
    filename        TEXT    NOT NULL,
    ext             TEXT    NOT NULL,
    size            INTEGER NOT NULL DEFAULT 0,
    mtime           INTEGER NOT NULL DEFAULT 0,
    date_taken      INTEGER,                                            -- Unix epoch seconds (UTC/local normalized)
    rating          INTEGER NOT NULL DEFAULT 0 CHECK (rating BETWEEN 0 AND 5),
    flag            INTEGER NOT NULL DEFAULT 0 CHECK (flag IN (0, 1)),
    width           INTEGER,
    height          INTEGER,
    orientation     INTEGER NOT NULL DEFAULT 1 CHECK (orientation BETWEEN 1 AND 8),
    lat             REAL,
    lon             REAL,
    location_name   TEXT,
    location_country TEXT,
    location_state  TEXT,
    location_city   TEXT,
    caption         TEXT     NOT NULL DEFAULT '',
    camera_make     TEXT,
    camera_model    TEXT,
    author          TEXT,
    exposure_time   TEXT,
    f_number        REAL,
    focal_length    REAL,
    iso             INTEGER,
    published       INTEGER NOT NULL DEFAULT 0 CHECK (published IN (0, 1)),
    is_video        INTEGER NOT NULL DEFAULT 0 CHECK (is_video IN (0, 1)),
    duration_ms     INTEGER,
    video_codec     TEXT,
    video_fps       REAL,
    original_backup_path TEXT,                                          -- Stashed original file path for 'Revert to original'
    edited_at       INTEGER,                                            -- Unix epoch seconds when pixel edit was saved (for 30-day expiry)
    metadata_dirty  INTEGER NOT NULL DEFAULT 0 CHECK (metadata_dirty IN (0, 1)),
    face_scan_version INTEGER NOT NULL DEFAULT 0,
    is_missing      INTEGER NOT NULL DEFAULT 0 CHECK (is_missing IN (0, 1)),
    UNIQUE(folder_id, filename)
);

-- Core filtering & sorting indexes (<10ms query budget at 500k rows)
CREATE INDEX IF NOT EXISTS idx_media_folder_id
    ON media(folder_id);

CREATE INDEX IF NOT EXISTS idx_media_date_taken
    ON media(date_taken DESC, id ASC);

CREATE INDEX IF NOT EXISTS idx_media_effective_date
    ON media(COALESCE(date_taken, mtime, 0) DESC, id ASC);

CREATE INDEX IF NOT EXISTS idx_media_rating_date
    ON media(rating DESC, date_taken DESC, id ASC);

CREATE INDEX IF NOT EXISTS idx_media_filename_nocase
    ON media(filename COLLATE NOCASE ASC);

CREATE INDEX IF NOT EXISTS idx_media_is_video
    ON media(is_video, date_taken DESC);

CREATE INDEX IF NOT EXISTS idx_media_location
    ON media(location_name) WHERE location_name IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_media_flag_partial
    ON media(flag, date_taken DESC) WHERE flag = 1;

CREATE INDEX IF NOT EXISTS idx_media_dirty_partial
    ON media(id) WHERE metadata_dirty = 1;

CREATE INDEX IF NOT EXISTS idx_media_edited_expiry_partial
    ON media(edited_at) WHERE edited_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_media_face_scan
    ON media(face_scan_version) WHERE is_video = 0;

CREATE INDEX IF NOT EXISTS idx_media_is_missing
    ON media(is_missing) WHERE is_missing = 1;

CREATE INDEX IF NOT EXISTS idx_media_day_bucket
    ON media(CAST(FLOOR((CASE WHEN date_taken > 100000000000 THEN date_taken / 1000.0 ELSE date_taken * 1.0 END) / 86400.0) AS INTEGER))
    WHERE date_taken IS NOT NULL AND date_taken != 0;

-- ----------------------------------------------------------------------------
-- 3. Descriptive Tags & Closure Table (Boolean AND / OR / NOT Hierarchy)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tags (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id       INTEGER REFERENCES tags(id) ON DELETE CASCADE,
    name            TEXT    NOT NULL,
    full_path       TEXT    NOT NULL UNIQUE                             -- e.g. 'PeopleTags/Family/Craig'
);

CREATE INDEX IF NOT EXISTS idx_tags_parent_id
    ON tags(parent_id);

CREATE INDEX IF NOT EXISTS idx_tags_name_nocase
    ON tags(name COLLATE NOCASE);

-- Transitive closure table storing all ancestor -> descendant paths
CREATE TABLE IF NOT EXISTS tag_closure (
    ancestor_id     INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    descendant_id   INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    depth           INTEGER NOT NULL DEFAULT 0 CHECK (depth >= 0),
    PRIMARY KEY (ancestor_id, descendant_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_tag_closure_descendant
    ON tag_closure(descendant_id, ancestor_id, depth);

-- Media <-> Tag many-to-many junction table
CREATE TABLE IF NOT EXISTS media_tags (
    media_id        INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    tag_id          INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (media_id, tag_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_media_tags_tag_media
    ON media_tags(tag_id, media_id);

-- ----------------------------------------------------------------------------
-- 4. People & Face Regions (XMP MP:RegionInfo compatible & Face Recognition Engine)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS people (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    name                    TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    avatar_media_people_id  INTEGER REFERENCES media_people(id) ON DELETE SET NULL,  -- Representative face crop
    centroid_embedding      BLOB,                                                   -- Normalized 512xfloat32 vector = 2048 bytes (running mean of confirmed faces)
    confirmed_count         INTEGER NOT NULL DEFAULT 0,
    thumbnail_blob          BLOB                                                    -- Circular/square avatar WebP blob for Quick Find
);

CREATE INDEX IF NOT EXISTS idx_people_confirmed_count
    ON people(confirmed_count DESC);

CREATE TABLE IF NOT EXISTS media_people (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id            INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    person_id           INTEGER REFERENCES people(id) ON DELETE SET NULL,       -- NULL if unassigned
    suggested_person_id INTEGER REFERENCES people(id) ON DELETE SET NULL,       -- Suggested match if similarity >= threshold
    similarity_score    REAL,                                                   -- Cosine similarity against suggested_person_id centroid
    state               INTEGER NOT NULL DEFAULT 0 CHECK (state IN (-2, -1, 0, 1)), -- 0 = Unassigned/Suggested, 1 = Confirmed, -1 = Rejected suggestion, -2 = Ignored ("Don't show again")
    is_manual           INTEGER NOT NULL DEFAULT 0,
    rect_x              REAL    NOT NULL CHECK (rect_x BETWEEN 0.0 AND 1.0),    -- Normalized 0.0-1.0 coordinates matching MPReg:Rectangle
    rect_y              REAL    NOT NULL CHECK (rect_y BETWEEN 0.0 AND 1.0),
    rect_w              REAL    NOT NULL CHECK (rect_w BETWEEN 0.0 AND 1.0),
    rect_h              REAL    NOT NULL CHECK (rect_h BETWEEN 0.0 AND 1.0),
    embedding           BLOB                                                    -- 512xfloat32 L2-normalized ArcFace vector = 2048 bytes
);

CREATE INDEX IF NOT EXISTS idx_media_people_person_media
    ON media_people(person_id, media_id);

CREATE INDEX IF NOT EXISTS idx_media_people_media_id
    ON media_people(media_id);

CREATE INDEX IF NOT EXISTS idx_media_people_suggested
    ON media_people(suggested_person_id, similarity_score DESC) WHERE state = 0;

CREATE INDEX IF NOT EXISTS idx_media_people_state
    ON media_people(state);

-- Tracks rejected suggestions to prevent 'Not this person' rejections from reappearing on centroid updates
CREATE TABLE IF NOT EXISTS face_rejected_suggestions (
    face_id     INTEGER NOT NULL REFERENCES media_people(id) ON DELETE CASCADE,
    person_id   INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    PRIMARY KEY (face_id, person_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS face_rejections (
    face_id     INTEGER NOT NULL REFERENCES media_people(id) ON DELETE CASCADE,
    person_id   INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    PRIMARY KEY (face_id, person_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_face_rejected_person
    ON face_rejected_suggestions(person_id, face_id);

-- ----------------------------------------------------------------------------
-- 5. Spatial R*Tree Index for World Map Bounding-Box Queries
-- ----------------------------------------------------------------------------
CREATE VIRTUAL TABLE IF NOT EXISTS media_rtree USING rtree(
    id,             -- Matches media.id
    min_lat,
    max_lat,
    min_lon,
    max_lon
);

-- ----------------------------------------------------------------------------
-- 6. Triggers for Automatic Closure Table, R*Tree, and Dirty Flag Maintenance
-- ----------------------------------------------------------------------------

-- 6a. Tag Closure Maintenance on INSERT
CREATE TRIGGER IF NOT EXISTS trg_tags_after_insert
AFTER INSERT ON tags
BEGIN
    -- Self-reference link (depth 0)
    INSERT INTO tag_closure (ancestor_id, descendant_id, depth)
    VALUES (NEW.id, NEW.id, 0);

    -- Copy all ancestor paths from parent_id (depth + 1)
    INSERT INTO tag_closure (ancestor_id, descendant_id, depth)
    SELECT ancestor_id, NEW.id, depth + 1
    FROM tag_closure
    WHERE descendant_id = NEW.parent_id
      AND NEW.parent_id IS NOT NULL;
END;

-- 6b. Tag Closure Maintenance on Parent Move (UPDATE OF parent_id)
CREATE TRIGGER IF NOT EXISTS trg_tags_after_update_parent
AFTER UPDATE OF parent_id ON tags
WHEN IFNULL(OLD.parent_id, -1) != IFNULL(NEW.parent_id, -1)
BEGIN
    -- Remove old ancestor links for the subtree rooted at NEW.id
    DELETE FROM tag_closure
    WHERE descendant_id IN (
        SELECT descendant_id FROM tag_closure WHERE ancestor_id = NEW.id
    )
    AND ancestor_id IN (
        SELECT ancestor_id FROM tag_closure WHERE descendant_id = NEW.id AND ancestor_id != NEW.id
    );

    -- Insert new ancestor links for the subtree rooted at NEW.id
    INSERT INTO tag_closure (ancestor_id, descendant_id, depth)
    SELECT supertree.ancestor_id, subtree.descendant_id, supertree.depth + subtree.depth + 1
    FROM tag_closure AS supertree
    CROSS JOIN tag_closure AS subtree
    WHERE supertree.descendant_id = NEW.parent_id
      AND NEW.parent_id IS NOT NULL
      AND subtree.ancestor_id = NEW.id;
END;

-- 6c. R*Tree Spatial Index Maintenance on Media INSERT / UPDATE / DELETE
CREATE TRIGGER IF NOT EXISTS trg_media_rtree_insert
AFTER INSERT ON media
WHEN NEW.lat IS NOT NULL AND NEW.lon IS NOT NULL
BEGIN
    INSERT OR REPLACE INTO media_rtree (id, min_lat, max_lat, min_lon, max_lon)
    VALUES (NEW.id, NEW.lat, NEW.lat, NEW.lon, NEW.lon);
END;

CREATE TRIGGER IF NOT EXISTS trg_media_rtree_update
AFTER UPDATE OF lat, lon ON media
BEGIN
    DELETE FROM media_rtree WHERE id = OLD.id;
    INSERT INTO media_rtree (id, min_lat, max_lat, min_lon, max_lon)
    SELECT NEW.id, NEW.lat, NEW.lat, NEW.lon, NEW.lon
    WHERE NEW.lat IS NOT NULL AND NEW.lon IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_media_rtree_delete
AFTER DELETE ON media
BEGIN
    DELETE FROM media_rtree WHERE id = OLD.id;
END;

-- 6d. Mark parent media row metadata_dirty = 1 when tags or people change
CREATE TRIGGER IF NOT EXISTS trg_media_tags_dirty_insert
AFTER INSERT ON media_tags
BEGIN
    UPDATE media SET metadata_dirty = 1 WHERE id = NEW.media_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_media_tags_dirty_delete
AFTER DELETE ON media_tags
BEGIN
    UPDATE media SET metadata_dirty = 1 WHERE id = OLD.media_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_media_people_dirty_insert
AFTER INSERT ON media_people
BEGIN
    UPDATE media SET metadata_dirty = 1 WHERE id = NEW.media_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_media_people_dirty_update
AFTER UPDATE ON media_people
BEGIN
    UPDATE media SET metadata_dirty = 1 WHERE id = NEW.media_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_media_people_dirty_delete
AFTER DELETE ON media_people
BEGIN
    UPDATE media SET metadata_dirty = 1 WHERE id = OLD.media_id;
END;


-- === THUMBNAILS_DB_SCHEMA ===
-- ============================================================================
-- PART 2: THUMBNAILS DATABASE (thumbnails.db)
-- Executed against thumbnails.db connection to isolate BLOB I/O from catalog.db
-- ============================================================================

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA temp_store = MEMORY;
PRAGMA cache_size = -32000;      -- 32 MB page cache
PRAGMA mmap_size = 268435456;    -- 256 MB memory-mapped I/O

CREATE TABLE IF NOT EXISTS thumbnails (
    media_id        INTEGER PRIMARY KEY,
    webp_blob       BLOB    NOT NULL,                                   -- 256x256 WebP image (~8-12 KB)
    mtime           INTEGER NOT NULL,
    width           INTEGER NOT NULL DEFAULT 256,
    height          INTEGER NOT NULL DEFAULT 256
);

CREATE INDEX IF NOT EXISTS idx_thumbnails_mtime
    ON thumbnails(mtime);

-- Pre-cropped 96x96 WebP face thumbnails (~2-3 KB) for instant Batch People Tag rendering
CREATE TABLE IF NOT EXISTS face_thumbnails (
    face_id         INTEGER PRIMARY KEY,                                -- References media_people(id) in catalog.db
    crop_webp       BLOB    NOT NULL                                    -- Pre-cropped 96x96 WebP face region
);
