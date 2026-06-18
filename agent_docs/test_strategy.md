# Test Strategy

Unit tests cover endpoint conversion, heatmap component extraction, endpoint
remapping, suppression, stitching, grouping, WCS conversion, persistence, API
serialization, and frontend-facing payloads. Tests must remain offline and avoid
loading production checkpoints.

Geometry evaluation uses segment angle, perpendicular offset, along-track overlap,
and endpoint error. Include horizontal, vertical, diagonal, reversed-endpoint,
zero-length, border-clipped, duplicate, and fragmented cases.

Run the suite with:

```bash
/Users/robert/miniconda3/envs/satid/bin/python -m pytest tests/ -q
```
