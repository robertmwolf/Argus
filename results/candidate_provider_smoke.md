# Candidate Provider Evaluation

Baseline provider: `local`
Images: 1

## Provider Summary

| Provider | Empty Rate | Error Rate | Mean Candidates | Mean Fetch ms | Identified Rate |
|---|---:|---:|---:|---:|---:|
| `local` | 0.0 | 0.0 | 87.0 | 1079.0 | 1.0 |
| `satchecker` | 1.0 | 0.0 | 0.0 | 13200.5 | 0.0 |

## Baseline Comparison

| Provider | Top-1 Agreement | Top-3 Contains Baseline Top-1 | Mean Confidence Delta |
|---|---:|---:|---:|
| `satchecker` | 0.0 | 0.0 | None |

## Per Image

### `data/test/synth_streak_000.fits`

- obs_time: `2024-04-02T02:00:00Z`; detections: 1; wcs_source: `fixture`
- `local`: candidates=87, top1=40685 FENGYUN 1C DEB, conf=0.0, total_ms=2337.8, error=None
- `satchecker`: candidates=0, top1=None , conf=None, total_ms=13200.5, error=None

