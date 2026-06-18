# Data Strategy

Prefer reviewed real FITS observations and preserve source-night provenance.
Balance training data across streak length, brightness, orientation, sensor, and
background conditions. Synthetic examples may supplement scarce bands but must
not enter validation or test splits.

All working annotations are endpoint segments. Convert historical source labels
at ingestion, then apply crops, flips, rotations, and scaling directly to both
endpoints. Validate that coordinates remain in the same frame as the referenced
pixels.
