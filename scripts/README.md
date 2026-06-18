# ARGUS Script Inventory

Active scripts support endpoint dataset ingestion, heatmap caching/evaluation,
and TLE catalog maintenance.

## Detection data and models

- `convert_gtimages.py`: preserves GTImages source endpoints.
- `merge_annotations.py`: fits endpoints to source masks and creates splits.
- `generate_synthetic_streaks.py`: creates synthetic images with endpoint labels.
- `cache_dinov3_heatmap_features.py`: caches spatial features and centerline targets.
- `evaluate_dinov3_heatmap.py`: evaluates endpoint segments from heatmaps.
- `evaluate_dinov3_orientation_centerline.py`: evaluates orientation-centerline
  checkpoints.
- `propose_dinov3_centerline_segments.py`: extracts endpoint proposals.
- `sweep_dinov3_centerline_segments.py`: sweeps proposal parameters.

## Operations

- `bootstrap_tle_catalog.py`: initializes local historical TLE coverage.
- `bootstrap_recent_tles.py`: bootstraps recent coverage and repairs zero-record gaps.
- `update_tle_catalog.py`: refreshes current catalog data.
- `fetch_weights.sh`: retrieves checkpoints and training logs from a remote host.

Generated datasets, caches, results, credentials, and weights are not committed.
Archived scripts are provenance only and must not be used as active workflows.
