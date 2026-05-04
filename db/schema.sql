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
    confidence       REAL NOT NULL,
    bbox_x1          REAL, bbox_y1 REAL,
    bbox_x2          REAL, bbox_y2 REAL,
    obb_cx           REAL, obb_cy  REAL,
    obb_w            REAL, obb_h   REAL,
    obb_angle_deg    REAL,
    streak_length_px REAL,
    ra_deg           REAL,
    dec_deg          REAL
);

CREATE TABLE identifications (
    id             TEXT PRIMARY KEY,
    detection_id   TEXT REFERENCES detections(id),
    norad_id       INTEGER,
    satellite_name TEXT,
    confidence     REAL,
    separation_deg REAL,
    rank           INTEGER    -- 1 = best match, up to 3
);

CREATE TABLE tracklets (
    id         TEXT PRIMARY KEY,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE tracklet_detections (
    tracklet_id  TEXT REFERENCES tracklets(id),
    detection_id TEXT REFERENCES detections(id),
    frame_index  INTEGER,
    PRIMARY KEY (tracklet_id, detection_id)
);
