"""FITS file loading and normalisation for StreakMind inference.

Loads a FITS image, applies Z-score normalisation, and packages the
result for downstream ML inference.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)


class FITSLoader:
    """Load and normalise FITS images for ML inference.

    Applies Z-score normalisation with 3-sigma clipping and returns a
    uint8 (H, W, 3) array suitable for model input.
    """

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load a FITS file and return normalised data.

        Normalisation: Z-score with 3-sigma clipping.
          1. Extract float32 pixel data from hdul[0].data
          2. Compute mean and std of finite pixel values
          3. Clip to [mean - 3*std, mean + 3*std]
          4. Scale clipped range to [0, 255] uint8
          5. Stack grayscale to 3 channels: np.stack([arr, arr, arr], axis=-1)
             Result shape: (H, W, 3)

        Args:
            path: Path to the FITS file.

        Returns:
            Dictionary with keys:
              array        — np.ndarray uint8 shape (H, W, 3)
              wcs          — astropy.wcs.WCS or None if header has no WCS
              exposure_time — float seconds or None (from EXPTIME header key)
              filename     — str (basename only)
              shape        — tuple (H, W)

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not valid FITS or hdul[0].data is None.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FITS file not found: {path}")

        try:
            with fits.open(path) as hdul:
                header = hdul[0].header
                raw = hdul[0].data

                if raw is None:
                    raise ValueError(
                        f"Primary HDU in {path.name} contains no image data"
                    )

                # --- Normalisation: Z-score with 3-sigma clipping ---
                arr = raw.astype(np.float32)
                finite_mask = np.isfinite(arr)
                if not finite_mask.any():
                    raise ValueError(
                        f"No finite pixel values found in {path.name}"
                    )
                mean = float(arr[finite_mask].mean())
                std = float(arr[finite_mask].std())

                lo = mean - 3.0 * std
                hi = mean + 3.0 * std
                arr = np.clip(arr, lo, hi)

                # Scale to [0, 255] uint8
                rng = hi - lo
                if rng == 0.0:
                    arr_u8 = np.zeros_like(arr, dtype=np.uint8)
                else:
                    arr_u8 = ((arr - lo) / rng * 255.0).astype(np.uint8)

                # Stack to 3-channel (H, W, 3)
                array_3ch = np.stack([arr_u8, arr_u8, arr_u8], axis=-1)

                # --- WCS ---
                try:
                    wcs = WCS(header)
                    # WCS is considered valid only if it has celestial axes
                    if wcs.naxis == 0 or not wcs.has_celestial:
                        wcs = None
                except Exception:
                    wcs = None

                # --- Exposure time ---
                exposure_time: float | None = None
                if "EXPTIME" in header:
                    try:
                        exposure_time = float(header["EXPTIME"])
                    except (TypeError, ValueError):
                        logger.warning("Could not parse EXPTIME in %s", path.name)

                h, w = arr_u8.shape

                return {
                    "array": array_3ch,
                    "wcs": wcs,
                    "exposure_time": exposure_time,
                    "filename": path.name,
                    "shape": (h, w),
                }
        except fits.verify.VerifyError as exc:
            raise ValueError(
                f"Invalid FITS file {path.name}: {exc}"
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"Cannot open {path.name} as FITS: {exc}"
            ) from exc

    def fits_to_png(self, fits_path: str | Path, output_path: str | Path) -> None:
        """Convert a FITS file to PNG.

        Loads via load() and saves the resulting uint8 array with cv2.imwrite.

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
            List of dicts, each with keys:
              x_pix   — input x pixel coordinate
              y_pix   — input y pixel coordinate
              ra_deg  — right ascension in degrees
              dec_deg — declination in degrees
        """
        results: list[dict[str, float]] = []
        for x, y in pixel_coords:
            sky = wcs.pixel_to_world(x, y)
            results.append(
                {
                    "x_pix": float(x),
                    "y_pix": float(y),
                    "ra_deg": float(sky.ra.deg),
                    "dec_deg": float(sky.dec.deg),
                }
            )
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
    print(f"  shape          : {data['shape']}")
    print(f"  array shape    : {data['array'].shape}")
    print(f"  array dtype    : {data['array'].dtype}")
    print(f"  exposure_time  : {data['exposure_time']} s")
    print(f"  wcs            : {data['wcs']}")

    # Save PNG next to input
    png_out = fits_path.with_suffix(".png")
    loader.fits_to_png(fits_path, png_out)
    print(f"\nPNG saved to: {png_out}")
