# ML Pipeline

ARGUS trains one-channel centerline heatmaps from endpoint annotations. The target
renderer draws each annotated segment into the heatmap; there is no separate
geometry head.

Inference runs at the training normalization and native tile scale, extracts
connected centerline components, fits segment endpoints, maps them to the source
image, suppresses duplicates, and stitches compatible fragments. The resulting
endpoint segments flow unchanged through evaluation, persistence, API responses,
WCS conversion, and frontend rendering.

Checkpoint metadata must include backbone, input size, normalization, threshold,
and model channel count. Reject incompatible checkpoints with a clear error.
