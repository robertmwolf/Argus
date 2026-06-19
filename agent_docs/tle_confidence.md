# TLE Cross-Identification Confidence

ARGUS exposes the methodology behind each satellite identification confidence
score. For an unclipped streak with a valid sky direction, the score is the
product of three independent Gaussian fit factors:

```text
identification confidence = rotation fit × lateral fit × TLE-age fit
```

## Factors

| UI label | API score | API measurement | Meaning |
| --- | --- | --- | --- |
| Rotation fit | `rotation_score` | `atrk_arcsec` | Along-track difference: how far ahead or behind the observed streak is relative to the propagated TLE position. This is orbital-track position, not physical rotation of the satellite. |
| Lateral fit | `lateral_score` | `xtrk_arcsec` | Cross-track difference: the sideways displacement from the propagated orbital path. |
| TLE age fit | `epoch_penalty` | `tle_age_hours` | Confidence retained after accounting for the age of the TLE at observation time. |

Rotation and lateral fits use a Gaussian with a 900 arcsecond sigma:

```text
fit = exp(-0.5 × (offset_arcsec / 900)²)
```

Consequently, an offset of 0 arcseconds scores 100%, 900 arcseconds scores
about 61%, and larger offsets fall away rapidly. The UI rounds percentages, so
a small non-zero offset may display as 100%.

The signs on `atrk_arcsec` and `xtrk_arcsec` describe the selected streak-vector
orientation. Reversing the endpoint order reverses both signs; use the absolute
values when judging match quality. The score always uses the absolute offset.

TLE age uses a 24-hour Gaussian sigma for normal catalog matches. Broad-epoch
fallback modes use a 168-hour sigma because older elements are expected in that
search mode. `tle_age_hours` may be signed around the epoch, but the penalty
uses its absolute value.

## API payload

Each entry in `detections[].identifications[]` returned by
`GET /api/result/{job_id}` includes:

```json
{
  "confidence": 0.952,
  "confidence_method": "rotation_x_lateral_x_tle_age",
  "atrk_arcsec": -253.2,
  "rotation_score": 0.9612,
  "xtrk_arcsec": 52.5,
  "lateral_score": 0.9983,
  "tle_age_hours": 3.0,
  "epoch_penalty": 0.9922
}
```

`position_score` is the product of `rotation_score` and `lateral_score` for
this method. The displayed confidence can be reconstructed by multiplying the
three score fields (allowing for serialized rounding).

For edge-clipped streaks or segments without a trustworthy direction,
along-track and cross-track decomposition is unavailable. In that case the API
returns `confidence_method: "position_x_tle_age"`, leaves the rotation/lateral
fields null or absent, and calculates confidence from the safe visible-position
fit and TLE-age fit.

Expected-versus-observed streak length remains a diagnostic when it can be
calculated; it is not a hidden confidence multiplier.

## UI behavior

The detection table shows the best candidate's total identification confidence
followed by Rotation, Lateral, and TLE age factors. The candidate detail panel
shows the same factors for each of the top three candidates, including offsets
in arcseconds and tooltips describing the measurements. When decomposition is
unsafe, the UI shows unavailable factors and explains the positional fallback.
