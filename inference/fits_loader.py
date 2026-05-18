"""FITS file loading and normalisation for ARGUS inference.

Loads a FITS image, normalises it, and packages the result for downstream
ML inference.

Two normalisation modes are supported, selected by the ARGUS_NORM env var:

  ARGUS_NORM=autostretch  (default for new training / cloud-trained Swin-L)
      PixInsight AutoSTF: robust nMAD background removal + midtone transfer
      function.  Sets the sky background median to ~0.25 regardless of
      exposure or sky brightness.  Use this once the model has been trained
      or fine-tuned with autostretch preprocessing.

  ARGUS_NORM=zscore  (required for current dino_tiny.pth trained weights)
      Z-score with 3-sigma clipping.  This is what all training runs up to
      and including the local Swin-T baseline used.  Must match training to
      avoid a domain shift that suppresses detection confidence.

  ARGUS_NORM=zscale
      IRAF/Astropy ZScale stretch, useful for StreakMind-style FITS→PNG
      detector training runs.

Set this to match whichever preprocessing was used during the training run
whose weights are loaded.  Mismatch → very low detection scores.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS

logger = logging.getLogger(__name__)

# Match this to whatever normalisation the loaded model weights were trained with.
# 'zscore'      — current dino_tiny.pth (trained on dev subset, CPU)
# 'autostretch' — future cloud-trained Swin-L (retrain with ARGUS_NORM=autostretch)
_NORM_MODE: str = os.environ.get("ARGUS_NORM", "zscore").lower()


def _valid_celestial_wcs(header: fits.Header) -> WCS | None:
    """Return a celestial WCS from *header*, or None if unavailable."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            wcs = WCS(header)
        if wcs.naxis == 0 or not wcs.has_celestial:
            return None
        return wcs
    except Exception:
        return None


def _load_sidecar_wcs(path: Path) -> WCS | None:
    """Load same-stem ASTAP/SkyTrack .wcs sidecar if the FITS lacks WCS."""
    for suffix in (".wcs", ".WCS"):
        sidecar = path.with_suffix(suffix)
        if not sidecar.exists():
            continue
        try:
            header = fits.Header.fromtextfile(sidecar)
        except Exception as exc:
            logger.warning("Could not read WCS sidecar %s: %s", sidecar.name, exc)
            continue
        wcs = _valid_celestial_wcs(header)
        if wcs is not None:
            logger.debug("Loaded WCS from sidecar %s", sidecar)
            return wcs
    return None


def _normalise_zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score normalisation with 3-sigma clipping → uint8 [0, 255].

    Steps:
      1. Compute mean and std of finite pixels.
      2. Clip to [mean − 3σ, mean + 3σ].
      3. Scale to [0, 255] uint8.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    mean = float(finite.mean())
    std  = float(finite.std())
    lo, hi = mean - 3.0 * std, mean + 3.0 * std
    clipped = np.clip(arr, lo, hi)
    rng = hi - lo
    if rng == 0.0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((clipped - lo) / rng * 255.0).astype(np.uint8)


def _normalise_autostretch(arr: np.ndarray) -> np.ndarray:
    """PixInsight AutoSTF normalisation → uint8 [0, 255].

    Delegates to inference.autostretch.autostretch() which applies nMAD
    background removal and a midtone transfer function.
    """
    from inference.autostretch import autostretch  # local import — module optional
    stretched = autostretch(arr)   # float32 in [0, 1]
    return (stretched * 255.0).astype(np.uint8)


def _normalise_zscale(arr: np.ndarray) -> np.ndarray:
    """ZScale normalisation → uint8 [0, 255].

    Uses Astropy's ZScaleInterval, matching the common astronomical display
    stretch used when converting FITS frames to detector-ready PNGs.
    """
    from astropy.visualization import ZScaleInterval

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    interval = ZScaleInterval()
    try:
        lo, hi = interval.get_limits(finite)
    except Exception:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)

    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


class FITSLoader:
    """Load and normalise FITS images for ML inference.

    Normalisation mode is controlled by the ARGUS_NORM environment variable
    (see module docstring).  The default is 'zscore' to match the current
    trained weights.
    """

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load a FITS file and return normalised data.

        Normalisation is selected by ARGUS_NORM (module-level constant):
          'zscore'      — Z-score with 3-sigma clipping → uint8
          'autostretch' — PixInsight AutoSTF → uint8

        Args:
            path: Path to the FITS file.

        Returns:
            Dictionary with keys:
              array          — np.ndarray uint8 shape (H, W, 3)
              wcs            — astropy.wcs.WCS or None
              exposure_time  — float seconds or None
              filename       — str (basename only)
              shape          — tuple (H, W)
              obs_time       — datetime (UTC) or None
              observer_lat   — float degrees or None
              observer_lon   — float degrees or None
              observer_alt_m — float metres or None
              norm_mode      — str, normalisation mode actually applied
              wcs_source     — 'fits', 'sidecar', or None

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not valid FITS or hdul[0].data is None.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FITS file not found: {path}")

        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            import cv2
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"Cannot read image file: {path.name}")
            arr_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return {
                "array": arr_u8,
                "header": {},
                "exposure_time": None,
                "filename": path.name,
                "shape": arr_u8.shape,
                "obs_time": None,
                "observer_lat": None,
                "observer_lon": None,
                "observer_alt_m": None,
                "norm_mode": "raw",
                "wcs": None,
                "wcs_source": None,
            }

        try:
            with fits.open(path) as hdul:
                header = hdul[0].header
                raw = hdul[0].data

                if raw is None:
                    raise ValueError(
                        f"Primary HDU in {path.name} contains no image data"
                    )

                arr = raw.astype(np.float32)
                if not np.isfinite(arr).any():
                    raise ValueError(
                        f"No finite pixel values found in {path.name}"
                    )

                # --- Normalisation -------------------------------------------
                norm = os.environ.get("ARGUS_NORM", _NORM_MODE).lower()
                if norm == "autostretch":
                    arr_u8 = _normalise_autostretch(arr)
                elif norm == "zscale":
                    arr_u8 = _normalise_zscale(arr)
                else:
                    if norm != "zscore":
                        logger.warning(
                            "Unknown ARGUS_NORM='%s'; falling back to zscore", norm
                        )
                        norm = "zscore"
                    arr_u8 = _normalise_zscore(arr)

                logger.debug("FITSLoader: norm=%s  shape=%s", norm, arr_u8.shape)

                # Stack to 3-channel (H, W, 3)
                array_3ch = np.stack([arr_u8, arr_u8, arr_u8], axis=-1)

                # --- WCS -----------------------------------------------------
                wcs = _valid_celestial_wcs(header)
                wcs_source = "fits" if wcs is not None else None
                if wcs is None:
                    wcs = _load_sidecar_wcs(path)
                    wcs_source = "sidecar" if wcs is not None else None

                # --- Exposure time (accept common header spellings) ----------
                exposure_time: float | None = None
                for _exp_key in ("EXPTIME", "EXPOSURE", "EXP_TIME"):
                    if _exp_key in header:
                        try:
                            exposure_time = float(header[_exp_key])
                            break
                        except (TypeError, ValueError):
                            logger.warning(
                                "Could not parse %s in %s", _exp_key, path.name
                            )

                # --- Observation time ----------------------------------------
                obs_time = None
                if "DATE-OBS" in header:
                    try:
                        from datetime import datetime, timezone
                        date_str = str(header["DATE-OBS"]).rstrip("Z")
                        obs_time = datetime.fromisoformat(date_str).replace(
                            tzinfo=timezone.utc
                        )
                    except (ValueError, TypeError):
                        logger.warning("Could not parse DATE-OBS in %s", path.name)

                # --- Observer location ----------------------------------------
                def _float_header(key: str) -> float | None:
                    val = header.get(key)
                    if val is None:
                        return None
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return None

                h, w = arr_u8.shape

                return {
                    "array":          array_3ch,
                    "wcs":            wcs,
                    "exposure_time":  exposure_time,
                    "filename":       path.name,
                    "shape":          (h, w),
                    "obs_time":       obs_time,
                    "observer_lat":   _float_header("SITELAT"),
                    "observer_lon":   _float_header("SITELONG"),
                    "observer_alt_m": _float_header("SITEELEV"),
                    "norm_mode":      norm,
                    "wcs_source":     wcs_source,
                }
        except fits.verify.VerifyError as exc:
            raise ValueError(f"Invalid FITS file {path.name}: {exc}") from exc
        except OSError as exc:
            raise ValueError(f"Cannot open {path.name} as FITS: {exc}") from exc

    def fits_to_png(self, fits_path: str | Path, output_path: str | Path) -> None:
        """Convert a FITS file to PNG using the active normalisation mode.

        Args:
            fits_path: Path to the source FITS file.
            output_path: Destination path for the PNG.
        """
        result = self.load(fits_path)
        output_path = Path(output_path)
        cv2.imwrite(str(output_path), result["array"])
        logger.info("Saved PNG to %s", output_path)

    def extract_wcs_metadata(
        self,
        wcs: WCS,
        pixel_coords: list[tuple[float, float]],
    ) -> list[dict[str, float]]:
        """Convert pixel (x, y) coordinates to RA/Dec using WCS.

        Args:
            wcs: An astropy WCS object with celestial axes.
            pixel_coords: List of (x, y) pixel coordinate tuples.

        Returns:
            List of dicts, each with keys: x_pix, y_pix, ra_deg, dec_deg.
        """
        results: list[dict[str, float]] = []
        for x, y in pixel_coords:
            sky = wcs.pixel_to_world(x, y)
            results.append({
                "x_pix":   float(x),
                "y_pix":   float(y),
                "ra_deg":  float(sky.ra.deg),
                "dec_deg": float(sky.dec.deg),
            })
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python fits_loader.py <path/to/image.fits>")
        sys.exit(1)

    fits_path = Path(sys.argv[1])
    loader = FITSLoader()
    data = loader.load(fits_path)

    print(f"\n=== {data['filename']} ===")
    print(f"  shape         : {data['shape']}")
    print(f"  array shape   : {data['array'].shape}")
    print(f"  norm_mode     : {data['norm_mode']}")
    print(f"  exposure_time : {data['exposure_time']} s")
    print(f"  wcs           : {data['wcs']}")

    png_out = fits_path.with_suffix(".png")
    loader.fits_to_png(fits_path, png_out)
    print(f"\nPNG saved to: {png_out}")
