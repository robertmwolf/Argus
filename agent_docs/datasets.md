# ARGUS Datasets

ARGUS uses FITS observations from Atwood Observatory, Frigate, and other reviewed
sources, plus selected synthetic centerline examples. A canonical annotation has
`x1`, `y1`, `x2`, and `y2` in the coordinate frame of its referenced image.

Historical source files may encode annotations differently. Dataset loaders and
builders must normalize them through `training.annotation_endpoints` before any
split, crop, augmentation, cache, target, or evaluation step.

Keep source data and large generated datasets outside git. Preserve deterministic
split seeds and provenance metadata. Never use validation or test images as
training negatives.
