# Space-Track API Guide

## Account Setup
Register at: https://www.space-track.org/auth/createAccount
Free account. Agree to terms of service (required).
Do not share credentials or redistribute raw TLE data.

## Credentials
Always use environment variables. Never hardcode.

```bash
export SPACETRACK_USER=your@email.com
export SPACETRACK_PASS=yourpassword
# Local development defaults to the Space-Track test site. Production defaults
# to the official site when ARGUS_ENV=production.
export ARGUS_ENV=development
export SPACETRACK_BASE_URL=https://for-testing-only.space-track.org/
```

ARGUS chooses the Space-Track endpoint in this order:

1. `SPACETRACK_BASE_URL` when set.
2. `https://for-testing-only.space-track.org/` for development/local runs.
3. `https://www.space-track.org/` when `ARGUS_ENV=production`.

The test server API is identical to production and uses the same credentials,
but Space-Track usage guidelines still apply.  Test-site and production
responses are cached under separate keys so data from the two hosts is never
mixed.

---

## API Policy (read carefully)

Space-Track has flagged the following as violations:
- Querying `gp_history` repeatedly for the same date ranges.
- Using `gp_history` when the GP class is appropriate.

**Rules**:
1. Once you download historical TLE data from `gp_history`, store it locally and never re-download it.  Historical TLEs are immutable — cache them permanently.
2. For current/recent TLEs (live pipeline), use the `GP` class, at most **once per hour**.
3. For large historical date ranges or full-catalog dumps, download the **annual TLE zip bundles** — do not use `gp_history` for bulk retrieval.

**GP_History policy (important):** Space-Track explicitly prohibits using `gp_history` without
both a specific NORAD_CAT_ID list AND a short epoch range.  It cannot be used to bootstrap the
full catalog across a broad date window.  Use the annual zip bundles instead.

---

## Bootstrapping Historical TLE Coverage

ARGUS cross-identification requires TLEs with epochs within ≤3 days of the observation time
(LEO staleness limit).  A fresh install has no historical coverage; images older than 72 hours
will produce unknown cross-ID results unless the catalog is bootstrapped first.

### Recommended approach — daily CREATION_DATE chunks

Space-Track admin confirmed the approved method for building rolling historical coverage:
query `gp_history` by **CREATION_DATE** one calendar day at a time.  Each day's data is
fetched once and cached permanently in `tle_catalog_coverage`.

`scripts/bootstrap_recent_tles.py` implements this pattern and serves as both the
one-time bootstrap tool and the daily keep-up cron job.

```bash
export SPACETRACK_USER=your@email.com
export SPACETRACK_PASS=yourpassword
# ARGUS_ENV defaults to development (test site), which mirrors production data.
# No need to switch to production for the bootstrap.

# One-time bootstrap — last 90 days (default), one query per day, ~4.5 min:
python scripts/bootstrap_recent_tles.py

# Explicit window:
python scripts/bootstrap_recent_tles.py --days 60
python scripts/bootstrap_recent_tles.py --start 2026-01-01 --end 2026-04-30

# Re-fetch already-cached days:
python scripts/bootstrap_recent_tles.py --force
```

Coverage tags written: `gp_history_creation_YYYY_MM_DD` (one per day).
The script is fully idempotent — re-running skips already-covered days.

**Daily keep-up cron job (required for ongoing pipeline use):**

```bash
# Run at 00:16 UTC — off-hour as required by Space-Track:
16 0 * * * cd /path/to/Argus && \
  SPACETRACK_USER=your@email.com \
  SPACETRACK_PASS=yourpassword \
  /Users/robert/miniconda3/envs/satid/bin/python scripts/bootstrap_recent_tles.py \
  >> logs/tle_keepup.log 2>&1
```

> **Scheduling rule:** Never schedule at :00 or :30 past the hour.
> Use :16 or :44 (Space-Track admin recommendation).

### Prior years — annual zip bundles

For observations older than the 90-day bootstrap window, load the relevant annual bundle:

```bash
# The 2025 bundle is already downloaded locally:
python scripts/bootstrap_tle_catalog.py \
    --zip data/tle_zips/data/exports/tle2025.txt

# For other years, use download_tle_bundle.py if fileshare access is available:
python scripts/download_tle_bundle.py --year 2024

# Manual fallback (download from sync.com, then ingest):
# https://ln5.sync.com/dl/afd354190/c5cd2q72-a5qjzp4q-nbjdiqkr-cenajuqu
python scripts/bootstrap_tle_catalog.py --zip-dir data/tle_zips/ --years 2024
```

### Coverage strategy

| Observation age | Source | Script |
|----------------|--------|--------|
| < 72 h | CelesTrak live refresh (auto) | none — handled by `TLECatalogManager` |
| 72 h – 90 days | GP_History by CREATION_DATE | `bootstrap_recent_tles.py` |
| > 90 days (current year) | GP_History by CREATION_DATE | `bootstrap_recent_tles.py --start … --end …` |
| Prior years | Annual zip bundle | `bootstrap_tle_catalog.py --zip …` |

Each annual bundle covers one full calendar year.  The 2025 bundle (`tle2025.txt`)
is already present locally and covers Apr 2024–Dec 2025.

---

## Rate Limits

| Limit | Value | Notes |
|-------|-------|-------|
| Requests per minute | 30 | Hard limit — exceed it and you get 429 |
| Requests per hour | 300 | Soft limit |
| Max rows per query | 10,000 | Use pagination for more |
| Simultaneous sessions | 1 | One login at a time |

**Always add a 3-second sleep between requests in loops:**
```python
import time
time.sleep(3)  # between Space-Track calls
```

---

## Key API Classes

### GP class — explicit current/live catalog maintenance only
ARGUS inference does **not** call this class automatically.  Use it only for a
deliberate current-catalog maintenance workflow if/when ARGUS is configured to
ingest live data.  Call at most **once per hour**.  Time calls 10–20 minutes off
the top and bottom of the hour (e.g. HH:12 or HH:48, **never** HH:00 or HH:30)
to avoid peak load periods.

Space-Track's recommended query:
```
https://www.space-track.org/basicspacedata/query/class/gp/decay_date/null-val/CREATION_DATE/%3Enow-0.042/format/tle
```

Equivalent via the spacetrack Python library (use JSON format so the pipeline
gets dict records instead of raw TLE text):
```python
import spacetrack.operators as op
from datetime import datetime, timedelta, timezone

cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=1)
results = st.gp(
    decay_date="null-val",
    creation_date=op.greater_than(cutoff.strftime("%Y-%m-%dT%H:%M:%S")),
    orderby="norad_cat_id asc",
)
```

In ARGUS code, `query_gp_current()` remains available from
`src/matching/spacetrack_query.py` for explicit maintenance scripts. It is not
used as an inference fallback.

### GP_History class — one-time ad-hoc historical queries only
For explicit historical diagnostics or approved backfills only.

**Do not poll this class.**  Do not use broad `gp_history` windows from
inference.  Fetch targeted or approved data once, store it locally, and do not
download it again.

```python
import spacetrack.operators as op
from datetime import datetime, timedelta

obs_time = datetime(2024, 4, 2, 2, 55, 24)
window = timedelta(days=3)

results = st.gp_history(
    epoch=op.inclusive_range(
        obs_time - window,
        obs_time
    ),
    mean_motion=op.greater_than(11.25),  # LEO only
    orderby='epoch desc',
)
```

In ARGUS code, `query_gp_history()` remains available from
`src/matching/spacetrack_query.py` for explicit diagnostics/backfills. It is not
used by the normal inference path.

**For large date ranges**: do not use `gp_history` — download the annual zip
bundles from:
https://ln5.sync.com/dl/afd354190/c5cd2q72-a5qjzp4q-nbjdiqkr-cenajuqu

---

## Routing in ARGUS

`inference/crossid.py → _fetch_tle_catalog()` delegates to
`src/matching/tle_manager.py → TLECatalogManager.get_tles()`.

### Live track (obs_time within last 72 hours)

| Local TLE coverage | Behavior | Network call |
|---|---|---|
| Present | Cross-identify against local `tle_catalog` | None |
| Missing + CelesTrak cooldown active | Leave detections unknown for this inference | None |
| Missing + cooldown elapsed (≥ 2 h) | Refresh from CelesTrak, re-check, then cross-identify | CelesTrak GP API |

CelesTrak is a public mirror; no auth required. ARGUS fetches two groups:
- `GROUP=active` — full active satellite catalog
- `SPECIAL=GPZ-PLUS` — analyst/unclassified 80000-series objects

The refresh is rate-gated to at most once per 2 hours via the
`tle_catalog_coverage` table (tag `'celestrak_refresh'`).

### Historical track (obs_time older than 72 hours)

| Local TLE coverage | Behavior | Network call |
|---|---|---|
| Present (bootstrapped from zip bundles) | Cross-identify | None |
| Missing | Leave detections unknown; log operator instruction | None |

Missing historical coverage is accepted as unknown. The operator should
pre-load the relevant year's zip bundle before processing historical images:

```bash
python scripts/bootstrap_tle_catalog.py --zip-dir data/tle_zips/ --years 2024
```

**Space-Track `gp_history` is never called at runtime.**  It is a one-time
download resource (policy: 1 query per lifetime per object) and may only be
used via explicit operator/admin scripts, not from inference code.

### Coverage source tags in `tle_catalog_coverage`

| Tag | Source | Written by |
|---|---|---|
| `zip_YYYY` / `txt_YYYY` | Annual Space-Track zip bundle | `bootstrap_tle_catalog.py` |
| `celestrak_refresh` | CelesTrak GP API (live edge) | `celestrak_client.py` / `TLECatalogManager` |
| `gp_current` | Space-Track GP class (explicit maintenance) | `bootstrap_tle_catalog.py --update-current` |

---

## TLE Format Reference

The JSON response from both GP and GP_History contains these fields:

```json
{
  "OBJECT_NAME": "STARLINK-2183",
  "NORAD_CAT_ID": "48274",
  "OBJECT_TYPE": "PAYLOAD",
  "EPOCH": "2024-04-02T02:30:00.123456",
  "MEAN_MOTION": "15.06389548",
  "ECCENTRICITY": ".0001423",
  "INCLINATION": "53.0538",
  "RA_OF_ASC_NODE": "142.5671",
  "ARG_OF_PERICENTER": "89.4284",
  "MEAN_ANOMALY": "270.6936",
  "BSTAR": ".35291E-3",
  "TLE_LINE1": "1 48274U ...",
  "TLE_LINE2": "2 48274  ..."
}
```

**Key fields for your pipeline:**
- `NORAD_CAT_ID`: unique identifier, use as primary key
- `OBJECT_NAME`: human readable name
- `EPOCH`: TLE epoch datetime (compute age vs obs_time here)
- `TLE_LINE1`, `TLE_LINE2`: pass directly to sgp4 Satrec.twoline2rv()
- `MEAN_MOTION`: quick orbit classification (LEO/MEO/GEO filter)
- `OBJECT_TYPE`: PAYLOAD / DEBRIS / ROCKET BODY / UNKNOWN

---

## Orbit Type Classification by Mean Motion

```python
def classify_orbit(mean_motion_rev_day: float) -> str:
    if mean_motion_rev_day > 11.25:
        return 'LEO'    # < 2000 km altitude
    elif mean_motion_rev_day > 2.0:
        return 'MEO'    # 2000–35786 km
    elif mean_motion_rev_day > 0.9:
        return 'GEO'    # ~35786 km
    else:
        return 'HEO'    # highly elliptical
```

**TLE staleness limits by orbit type:**
```python
MAX_TLE_AGE_HOURS = {
    'LEO': 72,    # 3 days — beyond this, position error > 100 km
    'MEO': 336,   # 2 weeks
    'GEO': 720,   # 1 month
    'HEO': 168,   # 1 week
}
```

---

## Error Handling

```python
from spacetrack.base import AuthenticationError
import requests

try:
    results = st.gp(...)
except AuthenticationError:
    logger.error("Space-Track authentication failed. Check SPACETRACK_USER/PASS env vars.")
    raise
except requests.exceptions.Timeout:
    logger.warning("Space-Track request timed out. Will retry with backoff.")
    time.sleep(30)
    results = st.gp(...)  # one retry
except requests.exceptions.ConnectionError:
    logger.error("Cannot reach Space-Track. Check network connection.")
    raise
```

---

## Useful Test Queries

```python
# Verify your account works — fetch ISS current TLE via GP class
iss = st.gp(norad_cat_id=25544, format='json')
print(iss[0]['OBJECT_NAME'])  # Should print: ISS (ZARYA)

# Explicit maintenance only: fetch all active TLEs (GP class, ≤ once/hour):
from src.matching.spacetrack_query import query_gp_current
tles = query_gp_current()
print(f"Active objects: {len(tles)}")

# Explicit diagnostic/backfill only: fetch historical TLEs for a specific query.
# Prefer targeted NORAD filters or Space-Track-approved bulk files.
from src.matching.spacetrack_query import query_gp_history
from datetime import datetime, timezone
obs = datetime(2024, 4, 2, 2, 55, 24, tzinfo=timezone.utc)
tles = query_gp_history(obs, epoch_window_days=1)
print(f"TLEs in 1-day window: {len(tles)}")
```
