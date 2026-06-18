"""WCS plate solving for the ARGUS pipeline.

Converts pixel-space streak endpoints to sky coordinates using an
astropy WCS object derived from the FITS header, then computes
position angle and angular velocity.
"""

from __future__ import annotations

import logging
import math
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS, FITSFixedWarning
import warnings

from src.detection.streak import StreakDetection
from src.ingest.fits_parser import FITSImage

logger = logging.getLogger(__name__)


class PlateSolver:
    """Converts pixel streak endpoints to sky coordinates via WCS.

    Uses the WCS stored in the FITSImage header to convert pixel
    coordinates to RA/Dec, then derives position angle and angular
    velocity.

    Example::

        solver = PlateSolver()
        detection = solver.solve(fits_image, detection)
    """

    def solve(
        self,
        fits_image: FITSImage,
        detection: StreakDetection,
    ) -> StreakDetection:
        """Populate sky coordinate fields on detection in-place.

        Uses astropy.wcs.WCS from fits_image.header.
        Converts pixel coords via wcs.all_pix2world().
        Sets: ra_start, dec_start, ra_end, dec_end, ra_center, dec_center.
        Computes position_angle_deg (celestial, North through East).
        Computes angular_velocity_arcsec_s = streak_angular_length_arcsec / exptime_sec.
        If no valid WCS in header: logs warning, leaves sky fields as None, returns.

        Args:
            fits_image: Parsed FITS image containing the WCS header.
            detection: StreakDetection whose pixel fields are already populated.
                Sky fields are set in-place.

        Returns:
            The same StreakDetection object (modified in place).
        """
        # Suppress common WCS warnings about non-standard keywords
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            try:
                wcs = WCS(fits_image.header)
            except Exception:
                logger.warning(
                    "Failed to construct WCS from header of %s — sky fields remain None",
                    fits_image.filepath.name,
                )
                return detection

        # WCS is invalid if it has no celestial axes (naxis==0 or no CRVAL)
        if not _wcs_is_valid(wcs):
            logger.warning(
                "No valid WCS found in header of %s — sky fields remain None",
                fits_image.filepath.name,
            )
            return detection

        # Pixel coordinates: [[x, y], ...] — astropy uses (x, y) == (col, row)
        pixels = np.array([
            [detection.x_start,  detection.y_start],
            [detection.x_end,    detection.y_end],
            [detection.x_center, detection.y_center],
        ], dtype=np.float64)

        try:
            sky = wcs.all_pix2world(pixels, 0)  # 0-indexed
        except Exception:
            logger.warning(
                "WCS pixel→sky conversion failed for %s — sky fields remain None",
                fits_image.filepath.name,
            )
            return detection

        ra_start,  dec_start  = float(sky[0, 0]), float(sky[0, 1])
        ra_end,    dec_end    = float(sky[1, 0]), float(sky[1, 1])
        ra_center, dec_center = float(sky[2, 0]), float(sky[2, 1])

        detection.ra_start  = ra_start
        detection.dec_start = dec_start
        detection.ra_end    = ra_end
        detection.dec_end   = dec_end
        detection.ra_center = ra_center
        detection.dec_center = dec_center

        # --- Position angle (North through East, celestial) ---
        # Source: standard spherical trig position angle formula
        detection.position_angle_deg = _celestial_position_angle(
            ra_start, dec_start, ra_end, dec_end
        )

        # --- Angular length of the streak (arcseconds) ---
        c_start  = SkyCoord(ra=ra_start  * u.deg, dec=dec_start  * u.deg)
        c_end    = SkyCoord(ra=ra_end    * u.deg, dec=dec_end    * u.deg)
        angular_length_arcsec = float(c_start.separation(c_end).arcsec)

        # --- Angular velocity (arcsec/s) ---
        if fits_image.exptime_sec and fits_image.exptime_sec > 0:
            detection.angular_velocity_arcsec_s = (
                angular_length_arcsec / fits_image.exptime_sec
            )
        else:
            logger.warning(
                "exptime_sec unavailable for %s — angular_velocity_arcsec_s remains None",
                fits_image.filepath.name,
            )

        logger.debug(
            "Plate solved: ra_center=%.4f dec_center=%.4f pa=%.1f°  "
            "ang_vel=%.2f arcsec/s",
            ra_center,
            dec_center,
            detection.position_angle_deg or 0.0,
            detection.angular_velocity_arcsec_s or 0.0,
        )
        return detection


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wcs_is_valid(wcs: WCS) -> bool:
    """Return True only when the WCS has at least 2 celestial axes and CRVAL.

    Args:
        wcs: astropy WCS object.

    Returns:
        True if the WCS can map pixel→sky coordinates.
    """
    try:
        if wcs.naxis < 2:
            return False
        # has_celestial is available in astropy WCS
        if not wcs.has_celestial:
            return False
        # CRVAL must be set to non-zero values (unset WCS has CRVAL=[0,0])
        crval = wcs.wcs.crval
        if crval is None or len(crval) < 2:
            return False
        # A completely unset WCS has CRPIX all zeros and CDELT all zeros too
        cdelt = wcs.wcs.cdelt
        cd = getattr(wcs.wcs, "cd", None)
        pc = wcs.wcs.get_pc()
        has_scale = (
            (cdelt is not None and np.any(cdelt != 0.0)) or
            (cd is not None and np.any(cd != 0.0)) or
            (pc is not None and not np.allclose(pc, np.eye(pc.shape[0])))
        )
        return bool(has_scale)
    except Exception:
        return False


def _celestial_position_angle(
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
) -> float:
    """Compute the celestial position angle from point 1 → point 2.

    Position angle is measured North through East (standard astronomical
    convention), in degrees [0, 360).

    # Source: standard spherical-trig position angle formula
    # Ref: https://en.wikipedia.org/wiki/Position_angle

    Args:
        ra1: Right ascension of start point, degrees.
        dec1: Declination of start point, degrees.
        ra2: Right ascension of end point, degrees.
        dec2: Declination of end point, degrees.

    Returns:
        Position angle in degrees [0, 360).
    """
    ra1_r  = math.radians(ra1)
    dec1_r = math.radians(dec1)
    ra2_r  = math.radians(ra2)
    dec2_r = math.radians(dec2)

    delta_ra = ra2_r - ra1_r
    x = math.cos(dec2_r) * math.sin(delta_ra)
    y = (math.cos(dec1_r) * math.sin(dec2_r)
         - math.sin(dec1_r) * math.cos(dec2_r) * math.cos(delta_ra))

    pa_deg = math.degrees(math.atan2(x, y))
    return pa_deg % 360.0
