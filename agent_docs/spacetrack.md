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
```

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

### GP (current element sets)
Current TLE/OMM for all catalogued objects.
Use this only if you need today's positions — NOT for historical matching.

```python
# Get current TLE for ISS:
st.gp(norad_cat_id=25544, format='json')

# Get all currently active satellites:
st.gp(decay_date='null-val', format='json')
```

### GP_History (historical element sets — USE THIS for your pipeline)
**This is the primary API for your system.**
Contains ALL historical TLE sets — 138+ million records.

```python
import spacetrack.operators as op
from datetime import datetime, timedelta

obs_time = datetime(2024, 4, 2, 2, 55, 24)
window = timedelta(days=3)

# Fetch TLEs within 3 days before observation
results = st.gp_history(
    epoch=op.inclusive_range(
        obs_time - window,
        obs_time
    ),
    orderby='epoch desc',    # most recent first per object
    format='json'
)
```

**Filtering by object type (reduces result size significantly):**
```python
# LEO objects only (most likely to appear in ground-based images)
results = st.gp_history(
    epoch=op.inclusive_range(obs_time - window, obs_time),
    mean_motion=op.greater_than(11.25),  # >11.25 rev/day = LEO
    format='json'
)

# Starlink constellation only (for Starlink-dense images)
results = st.gp_history(
    epoch=op.inclusive_range(obs_time - window, obs_time),
    object_name='STARLINK~',  # ~ is wildcard
    format='json'
)
```

---

## Caching Strategy

**Critical:** Never re-query Space-Track for the same time window.
Cache everything to disk.

```python
import diskcache as dc
import hashlib
import json
from datetime import datetime, timedelta

cache = dc.Cache('data/cache')

def make_cache_key(obs_time: datetime, window_days: int) -> str:
    # Round obs_time to nearest hour to maximize cache hits
    # Images taken minutes apart will share the same cache entry
    rounded = obs_time.replace(minute=0, second=0, microsecond=0)
    return f"gp_history_{rounded.strftime('%Y%m%d%H')}_{window_days}d"

def get_ttl(obs_time: datetime) -> int:
    """Historical queries never change. Recent queries do."""
    age_days = (datetime.utcnow() - obs_time).days
    if age_days > 7:
        return 30 * 24 * 3600   # 30 days for historical
    elif age_days > 1:
        return 24 * 3600         # 24 hours for recent
    else:
        return 2 * 3600          # 2 hours for very recent
```

---

## TLE Format Reference

The JSON response from GP_History contains these fields:

```json
{
  "CCSDS_OPM_VERS": "2.0",
  "COMMENT": "GENERATED VIA SPACETRACK.ORG API",
  "CREATION_DATE": "2024-04-02T04:00:00",
  "ORIGINATOR": "18 SPCS",
  "OBJECT_NAME": "STARLINK-2183",
  "OBJECT_ID": "2021-044AP",
  "NORAD_CAT_ID": "48274",
  "OBJECT_TYPE": "PAYLOAD",
  "CLASSIFICATION_TYPE": "U",
  "EPOCH": "2024-04-02T02:30:00.123456",
  "MEAN_MOTION": "15.06389548",       // rev/day — >11.25 = LEO
  "ECCENTRICITY": ".0001423",
  "INCLINATION": "53.0538",           // degrees
  "RA_OF_ASC_NODE": "142.5671",
  "ARG_OF_PERICENTER": "89.4284",
  "MEAN_ANOMALY": "270.6936",
  "EPHEMERIS_TYPE": "0",
  "ELEMENT_SET_NO": "999",
  "REV_AT_EPOCH": "16982",
  "BSTAR": ".35291E-3",
  "MEAN_MOTION_DOT": ".51230E-4",
  "MEAN_MOTION_DDOT": "0",
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
    results = st.gp_history(...)
except AuthenticationError:
    logger.error("Space-Track authentication failed. Check SPACETRACK_USER/PASS env vars.")
    raise
except requests.exceptions.Timeout:
    logger.warning("Space-Track request timed out. Will retry with backoff.")
    time.sleep(30)
    results = st.gp_history(...)  # one retry
except requests.exceptions.ConnectionError:
    logger.error("Cannot reach Space-Track. Check network connection.")
    raise
```

---

## Useful Test Queries

```python
# Verify your account works — fetch ISS current TLE
iss = st.gp(norad_cat_id=25544, format='json')
print(iss[0]['OBJECT_NAME'])  # Should print: ISS (ZARYA)

# Count LEO objects in a historical window (sanity check)
count = st.gp_history(
    epoch=op.inclusive_range('2024-04-01', '2024-04-02'),
    mean_motion=op.greater_than(11.25),
    format='count'
)
print(f"LEO objects with TLEs on 2024-04-01/02: {count}")
# Expect: 5,000–15,000

# Fetch all Starlink passes in a 3-day window (moderate size query)
starlinks = st.gp_history(
    epoch=op.inclusive_range('2024-04-01', '2024-04-04'),
    object_name='STARLINK~',
    orderby='epoch desc',
    format='json'
)
print(f"Starlink TLEs in window: {len(list(starlinks))}")
```
