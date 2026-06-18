"""Endpoint-only streak data shared by astrometry and matching."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class StreakDetection:
    """A streak defined by two pixel endpoints plus optional sky coordinates."""

    x_start: float
    y_start: float
    x_end: float
    y_end: float
    ra_start: float | None = None
    dec_start: float | None = None
    ra_end: float | None = None
    dec_end: float | None = None
    ra_center: float | None = None
    dec_center: float | None = None
    angular_velocity_arcsec_s: float | None = None
    position_angle_deg: float | None = None

    @property
    def x_center(self) -> float:
        return (self.x_start + self.x_end) / 2.0

    @property
    def y_center(self) -> float:
        return (self.y_start + self.y_end) / 2.0

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.y_end - self.y_start, self.x_end - self.x_start)) % 180.0

    @property
    def length_px(self) -> float:
        return math.hypot(self.x_end - self.x_start, self.y_end - self.y_start)
