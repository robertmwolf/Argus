# Dataset Naming and Coordinate Frames

Full-frame annotation files contain full-frame endpoint coordinates. Materialized
crop or tile datasets contain local endpoint coordinates and must reference the
matching cropped pixels. Never mix those frames.

Use descriptive dataset names containing source, split, native tile size, and
version where relevant. Run names belong to experiment outputs, not reusable base
datasets. Every generated annotation file should record its builder, seed, source
files, and coordinate frame in metadata.
