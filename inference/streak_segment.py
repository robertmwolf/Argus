"""Canonical line-segment representation for streak detections.

This is the authoritative home for :class:`StreakSegment` and the conversion
helpers.  ``eval/streak_metrics.py`` imports from here so the dataclass is
defined only once.

A ``StreakSegment`` stores two pixel-space endpoints ``(x1, y1)`` and
``(x2, y2)`` plus metadata.  Derived geometry (angle, length, midpoint) is
computed lazily in ``__post_init__``.

Compatibility guarantee: every detection dict in the pipeline also carries an
``obb`` key (backward-compat, derived from the segment).  Use
:func:`segment_to_obb` to reconstruct it.

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


# ---------------------------------------------------------------------------
# OBB <-> segment conversions
# ---------------------------------------------------------------------------


def obb_to_segment(
    obb: dict[str, Any],
    confidence: float,
    image_id: str | int,
    method: str = "",
) -> StreakSegment:
    """Convert an OBB dict to a StreakSegment.

    The segment runs along the major axis (width dimension) of the OBB,
    centred at (cx, cy).

    Args:
        obb: Dict with keys cx, cy, w, h, angle_deg.
        confidence: Detection confidence score.
        image_id: Image identifier.
        method: Optional detector/method name.

    Returns:
        Equivalent StreakSegment.
    """
    cx = float(obb["cx"])
    cy = float(obb["cy"])
    half_w = float(obb["w"]) / 2.0
    rad = math.radians(float(obb["angle_deg"]))
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)

    x1 = cx - half_w * cos_r
    y1 = cy - half_w * sin_r
    x2 = cx + half_w * cos_r
    y2 = cy + half_w * sin_r

    return StreakSegment(
        x1=x1, y1=y1, x2=x2, y2=y2,
        confidence=confidence,
        image_id=image_id,
        method=method,
    )


def segment_to_obb(seg: StreakSegment) -> dict[str, Any]:
    """Convert a StreakSegment to a backward-compat OBB dict.

    The OBB is centred on the segment midpoint.  w equals the segment
    length; h is the nominal PSF width (3.0 px) used only for display
    and backward-compat code that needs an OBB.

    Args:
        seg: Source streak segment.

    Returns:
        Dict with keys cx, cy, w, h, angle_deg.
    """
    return {
        "cx": seg.cx,
        "cy": seg.cy,
        "w": seg.length_px,
        "h": 3.0,
        "angle_deg": seg.angle_deg,
    }


def detection_dict_to_segment(det: dict[str, Any]) -> StreakSegment:
    """Build a StreakSegment from a pipeline detection dict.

    Prefers the native x1/y1/x2/y2 fields when present; falls back to
    deriving endpoints from the obb sub-dict.

    Args:
        det: Pipeline detection dict.  Must have either x1/y1/x2/y2 at
            top level, or an obb sub-dict with cx, cy, w, angle_deg.

    Returns:
        StreakSegment in image coordinates.

    Raises:
        KeyError: If neither x1/y1/x2/y2 nor a valid obb are present.
    """
    confidence = float(det.get("confidence", 0.0))
    image_id = det.get("image_id", "")
    method = str(det.get("method", ""))

    if all(k in det for k in ("x1", "y1", "x2", "y2")):
        return StreakSegment(
            x1=float(det["x1"]),
            y1=float(det["y1"]),
            x2=float(det["x2"]),
            y2=float(det["y2"]),
            confidence=confidence,
            image_id=image_id,
            method=method,
        )

    obb = det["obb"]
    return obb_to_segment(obb, confidence, image_id, method)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== inference/streak_segment.py smoke test ===")

    # Round-trip: obb -> segment -> obb
    obb_in = {"cx": 200.0, "cy": 100.0, "w": 400.0, "h": 5.0, "angle_deg": 30.0}
    seg = obb_to_segment(obb_in, confidence=0.9, image_id="img1", method="test")
    obb_out = segment_to_obb(seg)

    assert abs(seg.cx - 200.0) < 1e-9, f"cx wrong: {seg.cx}"
    assert abs(seg.cy - 100.0) < 1e-9, f"cy wrong: {seg.cy}"
    assert abs(seg.length_px - 400.0) < 1e-6, f"length wrong: {seg.length_px}"
    assert abs(seg.angle_deg - 30.0) < 0.01, f"angle wrong: {seg.angle_deg}"
    assert abs(obb_out["cx"] - 200.0) < 1e-9
    assert abs(obb_out["w"] - 400.0) < 1e-6
    assert obb_out["h"] == 3.0
    print("PASS: obb -> segment -> obb round-trip")

    # detection_dict_to_segment via explicit endpoints
    det = {"x1": 0.0, "y1": 0.0, "x2": 300.0, "y2": 0.0,
           "confidence": 0.8, "image_id": "img2", "method": "ml"}
    seg2 = detection_dict_to_segment(det)
    assert abs(seg2.length_px - 300.0) < 1e-6
    assert abs(seg2.angle_deg) < 1e-9
    print("PASS: detection_dict_to_segment via x1/y1/x2/y2")

    # detection_dict_to_segment via obb fallback
    det_obb = {"obb": obb_in, "confidence": 0.7, "image_id": "img3", "method": "ml"}
    seg3 = detection_dict_to_segment(det_obb)
    assert abs(seg3.length_px - 400.0) < 1e-6
    print("PASS: detection_dict_to_segment via obb fallback")

    print("\nAll smoke tests passed.")
