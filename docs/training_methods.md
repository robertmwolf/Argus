# Training Methods

The active training family is a DINOv3 spatial backbone with a one-channel
centerline heatmap head. Endpoint annotations are rasterized into centerline
targets. Cached-feature and end-to-end trainers share the same target contract.

Record the dataset version, coordinate frame, backbone, normalization, native
tile size, overlap, loss, optimizer, seed, checkpoint, and evaluation threshold
for every run. Evaluate checkpoints with `eval.geometry_metrics` and retain the
prediction file used to produce each report.

Old experiment-specific training paths are intentionally not part of the current
workflow. New work should extend the endpoint heatmap path rather than revive a
retired detector family.
