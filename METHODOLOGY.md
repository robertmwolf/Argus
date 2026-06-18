# ARGUS Methodology

ARGUS detects linear satellite trails in astronomical images and represents each
detection as an image-space segment with endpoints `(x1, y1)` and `(x2, y2)`.
Those endpoints are the complete geometric definition of a streak.

## Detection

The active DINOv3 models predict a single centerline heatmap. Images are processed
as native-scale overlapping tiles so short trails remain visible. Connected
heatmap components are reduced to their principal-axis endpoints and mapped back
to full-image coordinates.

## Post-processing

Duplicate segments are suppressed using endpoint geometry. Nearby collinear
fragments may be stitched when their angular difference, perpendicular offset,
gap, and growth ratio are within configured limits. Optional image-based
refinement adjusts the segment axis or extent without introducing an independent
width parameter.

## Evaluation

`eval.geometry_metrics` reports:

- detection precision and recall from segment compatibility;
- angular error;
- perpendicular centerline offset;
- along-track overlap;
- endpoint error;
- recall by short, medium, and long length bands.

Metrics are computed from endpoints for both predictions and ground truth.
Historical annotations are converted to endpoints at load time before scoring.

## Identification

When WCS is available, both endpoints are transformed to sky coordinates. The
observation time and resulting sky track are compared with locally stored TLE
propagations. Runtime inference never depends on a live catalog query.

## Reproducibility

Training and evaluation commands, checkpoints, thresholds, dataset split, native
tile size, overlap, and normalization mode must be recorded for every experiment.
Generated results belong under `results/<run>/`; model weights are not committed.
