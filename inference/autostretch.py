"""PixInsight AutoSTF (Screen Transfer Function) auto-stretch for ARGUS.

Implements the AutoSTF algorithm from PixInsight's AdP AutoStretch script,
adapted for numpy arrays. Replaces Z-score normalisation in FITSLoader with
an astronomically-aware stretch that:

  - Sets the black point at (median - 2.8 * nMAD) to remove sky background
  - Applies a midtone transfer function (MTF) so the background median lands
    at target_bg (default 0.25)
  - Preserves streak signal while suppressing background noise

Algorithm source:
  # Source: Juan Conejero / Pleiades Astrophoto — AutoSTF algorithm
  # Ref: /Applications/PixInsight/src/scripts/AdP/AutoStretch.js
  # See also: PixInsight Reference Manual, Screen Transfer Function

Usage::

    from inference.autostretch import autostretch

    arr_f32 = raw_adu.astype(np.float32)          # any range
    stretched = autostretch(arr_f32)               # float32 in [0, 1]
    uint8_img = (stretched * 255).astype(np.uint8)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Default AutoSTF parameters matching PixInsight's STF dialog defaults.
_SHADOWS_CLIP: float = -2.8   # sigma units below the median
_TARGET_BG: float = 0.25      # desired output level for the background median
_MTF_EPS: float = 5e-5        # binary-search convergence tolerance


def _mtf(m: float, x: np.ndarray) -> np.ndarray:
    """Apply PixInsight's midtone transfer function element-wise.

    # Source: Juan Conejero / Pleiades Astrophoto — MTF formula
    # Ref: /Applications/PixInsight/src/scripts/AdP/AutoStretch.js

    MTF(m, x) maps:
      x = 0 → 0
      x = m → 0.5
      x = 1 → 1

    Args:
        m: Midtone balance parameter in (0, 1).
        x: Float array of values in [0, 1].

    Returns:
        Transformed array, same shape and dtype as x.
    """
    out = np.empty_like(x)
    denom = (2.0 * m - 1.0) * x - m
    safe = np.abs(denom) > 1e-15
    out[safe] = (m - 1.0) * x[safe] / denom[safe]
    out[~safe] = 0.5  # at the exact midtone singularity
    # Clamp boundary conditions
    out = np.where(x <= 0.0, 0.0, np.where(x >= 1.0, 1.0, out))
    return out


def _find_midtone(target: float, v1: float, eps: float = _MTF_EPS) -> float:
    """Binary-search for midtone m such that MTF(m, v1) == target.

    # Source: Juan Conejero / Pleiades Astrophoto — findMidtonesBalance
    # Ref: /Applications/PixInsight/src/scripts/AdP/AutoStretch.js

    Args:
        target: Desired output value (target_bg, typically 0.25).
        v1: Normalised median after shadow clipping: (median - c0).
        eps: Convergence tolerance.

    Returns:
        Midtone balance value m in [0, 1].
    """
    if v1 <= 0.0:
        return 0.0
    if v1 >= 1.0:
        return 1.0

    target = float(np.clip(target, 0.0, 1.0))
    m0, m1 = (0.0, 0.5) if v1 < target else (0.5, 1.0)

    for _ in range(64):  # converges in <20 iterations for eps=5e-5
        m = (m0 + m1) * 0.5
        v = float(_mtf(m, np.array([v1], dtype=np.float64))[0])
        if abs(v - target) < eps:
            return m
        if v < target:
            m1 = m
        else:
            m0 = m

    return (m0 + m1) * 0.5


def autostretch(
    image: np.ndarray,
    shadows_clip: float = _SHADOWS_CLIP,
    target_bg: float = _TARGET_BG,
) -> np.ndarray:
    """Apply PixInsight AutoSTF to an astronomical image.

    Works on grayscale (H, W) or multi-channel (H, W, C) images. For
    multi-channel input, linked-channel mode is used: a single c0 and
    midtone m are computed from the luminance (per-pixel mean across
    channels) so all channels stretch identically.

    Steps:
      1. Normalise raw ADU to [0, 1] via percentile clipping (0.01–99.99 %)
         to remove hot/cold pixels before statistics.
      2. Compute median and nMAD = 1.4826 * MAD on the normalised image.
      3. c0 = clamp(median + shadows_clip * nMAD, 0, 1)  (shadow black point)
         c1 = 1.0                                          (highlight white point)
      4. Binary-search midtone m such that MTF(m, median - c0) = target_bg.
      5. Clip to [c0, 1], rescale to [0, 1], apply MTF.

    # Source: Juan Conejero / Pleiades Astrophoto — AutoSTF algorithm
    # Ref: /Applications/PixInsight/src/scripts/AdP/AutoStretch.js

    Args:
        image: Float32 (or any float) array, shape (H, W) or (H, W, C).
               Input range is arbitrary (raw ADU counts are fine).
        shadows_clip: Shadow clipping in sigma units from the median.
                      PixInsight default is -2.8.
        target_bg: Target background level in [0, 1] for the stretched
                   median. PixInsight default is 0.25.

    Returns:
        float32 array in [0, 1], same shape as input.

    Raises:
        ValueError: If the image has no finite pixels or is not 2D/3D.
    """
    img = np.asarray(image, dtype=np.float32)

    if img.ndim not in (2, 3):
        raise ValueError(
            f"autostretch expects a 2D or 3D array, got shape {img.shape}"
        )

    # --- 1. Percentile normalise to [0, 1] -----------------------------------
    finite_px = img[np.isfinite(img)]
    if finite_px.size == 0:
        raise ValueError("autostretch received an image with no finite pixels")

    lo = float(np.percentile(finite_px, 0.01))
    hi = float(np.percentile(finite_px, 99.99))
    if hi <= lo:
        hi = lo + 1.0

    norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0)

    # --- 2. Compute statistics on linked luminance ---------------------------
    if norm.ndim == 3:
        lum = norm.mean(axis=2)  # (H, W)
    else:
        lum = norm

    finite_lum = lum[np.isfinite(lum)]
    median = float(np.median(finite_lum))
    mad = float(np.median(np.abs(finite_lum - median)))
    n_mad = 1.4826 * mad  # normalised MAD ≈ σ for Gaussian noise

    # --- 3. Shadow clipping point --------------------------------------------
    c0 = float(np.clip(median + shadows_clip * n_mad, 0.0, 1.0))
    c1 = 1.0

    logger.debug(
        "AutoSTF: median=%.4f  nMAD=%.4f  c0=%.4f  c1=%.1f",
        median, n_mad, c0, c1,
    )

    # --- 4. Midtone balance --------------------------------------------------
    v1 = median - c0  # normalised median after shadow removal
    m = _find_midtone(target_bg, v1)

    logger.debug("AutoSTF: v1=%.4f  midtone_m=%.4f", v1, m)

    # --- 5. Apply stretch: clip → rescale → MTF ------------------------------
    stretched = np.clip((norm - c0) / (c1 - c0), 0.0, 1.0)
    stretched = _mtf(m, stretched)

    return stretched.astype(np.float32)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        # Smoke-test on a synthetic low-contrast sky image
        rng = np.random.default_rng(42)
        sky_level = 1000.0
        sky_noise = 20.0
        synthetic = (rng.normal(sky_level, sky_noise, (256, 256))).astype(np.float32)
        # Add a faint streak
        synthetic[100:110, 50:200] += 80.0

        out = autostretch(synthetic)
        stretched_median = float(np.median(out))
        print(f"Input  range: [{synthetic.min():.1f}, {synthetic.max():.1f}]")
        print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
        print(f"Output median (should be ≈ 0.25): {stretched_median:.4f}")
        sys.exit(0)

    fits_path = Path(sys.argv[1])
    from astropy.io import fits as afits
    with afits.open(fits_path) as hdul:
        raw = hdul[0].data.astype(np.float32)

    out = autostretch(raw)
    print(f"Input  range  : [{raw.min():.1f}, {raw.max():.1f}]")
    print(f"Output range  : [{out.min():.4f}, {out.max():.4f}]")
    print(f"Output median : {float(np.median(out)):.4f}  (target ≈ 0.25)")
