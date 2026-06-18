"""Convert legacy source annotations to canonical streak endpoints.

This is the only training boundary allowed to understand the historical box
fields. New annotation files should store ``x1``, ``y1``, ``x2``, and ``y2``.
"""

from __future__ import annotations

import math
from typing import Any


def annotation_to_endpoints(annotation: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return canonical endpoints from a source annotation.

    Args:
        annotation: Annotation with native endpoints or historical geometry.

    Returns:
        Endpoint tuple ``(x1, y1, x2, y2)``.

    Raises:
        KeyError: If no supported geometry is present.
    """
    if all(key in annotation for key in ("x1", "y1", "x2", "y2")):
        return tuple(float(annotation[key]) for key in ("x1", "y1", "x2", "y2"))

    segment = annotation.get("line_segment") or annotation.get("endpoints")
    if isinstance(segment, dict):
        return tuple(float(segment[key]) for key in ("x1", "y1", "x2", "y2"))
    if isinstance(segment, (list, tuple)) and len(segment) == 4:
        return tuple(float(value) for value in segment)

    # Historical source labels encoded the same centerline as a rotated box.
    legacy = annotation.get("obb")
    if legacy is not None:
        if isinstance(legacy, dict):
            cx = float(legacy["cx"])
            cy = float(legacy["cy"])
            length = float(legacy["w"])
            angle_deg = float(legacy.get("angle_deg", 0.0))
        else:
            cx, cy, length, _, angle_deg = (float(value) for value in legacy)
        half = length / 2.0
        radians = math.radians(angle_deg)
        return (
            cx - half * math.cos(radians),
            cy - half * math.sin(radians),
            cx + half * math.cos(radians),
            cy + half * math.sin(radians),
        )

    # Last-resort conversion for the oldest axis-aligned source annotations.
    x, y, width, height = (float(value) for value in annotation["bbox"])
    if width >= height:
        return x, y + height / 2.0, x + width, y + height / 2.0
    return x + width / 2.0, y, x + width / 2.0, y + height


if __name__ == "__main__":
    sample = {"x1": 1, "y1": 2, "x2": 3, "y2": 4}
    assert annotation_to_endpoints(sample) == (1.0, 2.0, 3.0, 4.0)
