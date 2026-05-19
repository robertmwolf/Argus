-- ARGUS database schema.
-- Compatible with PostgreSQL 16 and SQLite (via aiosqlite).

CREATE TABLE observations (
    id            TEXT PRIMARY KEY,   -- UUID as string for SQLite compat
    filename      TEXT NOT NULL,
    uploaded_at   TEXT DEFAULT (datetime('now')),
    exposure_time REAL,
    obs_epoch     TEXT,               -- ISO8601
    fits_wcs_json TEXT,               -- JSON-serialized WCS params
    status        TEXT DEFAULT 'queued'
    -- status: queued / processing / complete / failed
);

CREATE TABLE detections (
    id               TEXT PRIMARY KEY,
    observation_id   TEXT REFERENCES observations(id),
    method           TEXT DEFAULT 'ml',
    confidence       REAL NOT NULL,
    bbox_x1          REAL, bbox_y1 REAL,
    bbox_x2          REAL, bbox_y2 REAL,
    obb_cx           REAL, obb_cy  REAL,
    obb_w            REAL, obb_h   REAL,
    obb_angle_deg    REAL,
    streak_length_px REAL,
    ra_tip1_deg      REAL,
    dec_tip1_deg     REAL,
    ra_tip2_deg      REAL,
    dec_tip2_deg     REAL
);

CREATE TABLE identifications (
    id             TEXT PRIMARY KEY,
    detection_id   TEXT REFERENCES detections(id),
    norad_id       INTEGER,
    satellite_name TEXT,
    confidence     REAL,
    separation_deg REAL,
    rank           INTEGER,   -- 1 = best match, up to 3
    tle_epoch      TEXT,
    tle_age_hours  REAL,
    photo_taken_at TEXT,
    tle_data_fresh_at TEXT,
    tle_source     TEXT,
    tle_search_mode TEXT,
    epoch_search_window_days INTEGER,
    epoch_drift_hours REAL,
    position_score REAL,
    epoch_penalty  REAL
);

CREATE TABLE tracklets (
    id         TEXT PRIMARY KEY,
    created_at TEXT DEFAULT (datetime('now'))
);

-- TLE catalog: local copy of TLE data from bootstrap zips (historical) and
-- CelesTrak (live edge, ≤ once per 2 hours).  Space-Track gp_history is a
-- one-time download resource and is never called at runtime.
CREATE TABLE IF NOT EXISTS tle_catalog (
    norad_id     INTEGER NOT NULL,
    epoch        TEXT NOT NULL,          -- ISO8601 UTC
    object_name  TEXT NOT NULL,
    object_type  TEXT,                   -- PAYLOAD / DEBRIS / ROCKET BODY / UNKNOWN / ANALYST
    mean_motion  REAL,                   -- rev/day
    tle_line1    TEXT NOT NULL,
    tle_line2    TEXT NOT NULL,
    source       TEXT,                   -- 'bootstrap' | 'celestrak' | 'spacetrack_gp'
    ingested_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (norad_id, epoch)
);

CREATE INDEX IF NOT EXISTS idx_tle_catalog_epoch ON tle_catalog(epoch);
CREATE INDEX IF NOT EXISTS idx_tle_catalog_norad  ON tle_catalog(norad_id);

-- Coverage log: records which data sources have been fully loaded so the
-- bootstrap script is a no-op when re-run against an already-populated DB.
CREATE TABLE IF NOT EXISTS tle_catalog_coverage (
    source_tag    TEXT PRIMARY KEY,      -- e.g. 'zip_2025', 'gp_current'
    description   TEXT,
    record_count  INTEGER,
    downloaded_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE tracklet_detections (
    tracklet_id  TEXT REFERENCES tracklets(id),
    detection_id TEXT REFERENCES detections(id),
    frame_index  INTEGER,
    PRIMARY KEY (tracklet_id, detection_id)
);
