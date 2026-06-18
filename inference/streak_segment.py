"""Canonical line-segment representation for streak detections.

This is the authoritative home for :class:`StreakSegment`.  Evaluation and
runtime code import it from here so endpoint geometry has one definition.

A ``StreakSegment`` stores two pixel-space endpoints ``(x1, y1)`` and
``(x2, y2)`` plus metadata.  Derived geometry (angle, length, midpoint) is
computed lazily in ``__post_init__``.

# Source: StreakMind — segment-based detection representation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Canonical dataclass
# ---------------------------------------------------------------------------


@dataclass
class StreakSegment:
    """Line-segment representation of a streak detection or annotation.

    Attributes:
        x1: X-coordinate of endpoint 1.
        y1: Y-coordinate of endpoint 1.
        x2: X-coordinate of endpoint 2.
        y2: Y-coordinate of endpoint 2.
        confidence: Detection confidence score in [0, 1].
        image_id: Identifier of the source image.
        method: Optional detector name for provenance tracking.
        angle_deg: Streak orientation in degrees (0-180, atan2 reduced).
            Computed in __post_init__.
        length_px: Euclidean length of the segment in pixels.
            Computed in __post_init__.
        cx: X-coordinate of the segment midpoint.
            Computed in __post_init__.
        cy: Y-coordinate of the segment midpoint.
            Computed in __post_init__.
        streak_length_px: Alias for length_px (pipeline compatibility).
            Computed in __post_init__.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    image_id: str | int
    method: str = ""

    # Derived fields populated by __post_init__
    angle_deg: float = field(init=False)
    length_px: float = field(init=False)
    cx: float = field(init=False)
    cy: float = field(init=False)
    streak_length_px: float = field(init=False)

    def __post_init__(self) -> None:
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        self.length_px = math.sqrt(dx * dx + dy * dy)
        self.cx = (self.x1 + self.x2) / 2.0
        self.cy = (self.y1 + self.y2) / 2.0
        self.angle_deg = math.degrees(math.atan2(dy, dx)) % 180.0
        self.streak_length_px = self.length_px


def detection_dict_to_segment(det: dict[str, Any]) -> StreakSegment:
    """Build a StreakSegment from a pipeline detection dict.

    Args:
        det: Pipeline detection dict with endpoint fields.

    Returns:
        StreakSegment in image coordinates.

    Raises:
        KeyError: If an endpoint field is absent.
    """
    confidence = float(det.get("confidence", 0.0))
    image_id = det.get("image_id", "")
    method = str(det.get("method", ""))

    return StreakSegment(
        x1=float(det["x1"]),
        y1=float(det["y1"]),
        x2=float(det["x2"]),
        y2=float(det["y2"]),
        confidence=confidence,
        image_id=image_id,
        method=method,
    )


def apply_segment_geometry(det: dict[str, Any]) -> dict[str, Any]:
    """Populate derived geometry from a detection's endpoints.

    Args:
        det: Mutable detection dictionary with endpoint fields.

    Returns:
        The same dictionary with midpoint, angle, and length fields updated.
    """
    segment = detection_dict_to_segment(det)
    det["cx"] = segment.cx
    det["cy"] = segment.cy
    det["angle_deg"] = segment.angle_deg
    det["streak_length_px"] = segment.length_px
    return det


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== inference/streak_segment.py smoke test ===")

    det = {"x1": 0.0, "y1": 0.0, "x2": 300.0, "y2": 0.0,
           "confidence": 0.8, "image_id": "img2", "method": "ml"}
    seg2 = detection_dict_to_segment(det)
    assert abs(seg2.length_px - 300.0) < 1e-6
    assert abs(seg2.angle_deg) < 1e-9
    print("PASS: detection_dict_to_segment via x1/y1/x2/y2")

    apply_segment_geometry(det)
    assert det["cx"] == 150.0
    assert det["streak_length_px"] == 300.0

    print("\nAll smoke tests passed.")
