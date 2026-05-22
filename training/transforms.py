"""Custom mmcv/mmdet data transforms for astronomy image loading.

Register this module before building the MMDetection runner so that
``LoadFITSFromFile`` is available in the transform registry.

Import by adding to the MMDetection config::

    custom_imports = dict(imports=['training.transforms'], allow_failed_imports=False)

Tile encoding
-------------
Frigate images are tiled at build time.  A virtual tile path encodes the crop
coordinates as a suffix on the stem::

    /path/to/Capture_00143__tx300_ty200_ts400.png

``LoadFITSFromFile`` detects this pattern, loads the *original* file (without
the suffix), and crops the requested tile before returning.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

_TILE_RE = re.compile(r"^(.+?)__tx(\d+)_ty(\d+)_ts(\d+)$")

logger = logging.getLogger(__name__)

try:
    from mmcv.transforms import BaseTransform, TRANSFORMS
    _MMCV_AVAILABLE = True
except ImportError:
    _MMCV_AVAILABLE = False
    BaseTransform = object  # type: ignore[misc,assignment]

    class _FakeRegistry:
        def register_module(self):
            def decorator(cls):
                return cls
            return decorator

    TRANSFORMS = _FakeRegistry()  # type: ignore[assignment]


@TRANSFORMS.register_module()
class LoadFITSFromFile(BaseTransform):
    """Load a FITS telescope image or regular image for mmcv.

    Uses ``inference.FITSLoader`` for Z-score normalisation and uint8
    conversion when the path is a FITS file.  Non-FITS images are loaded via
    mmcv so mixed SatStreaks + GTImages COCO splits can use one pipeline.
    The result is compatible with all downstream mmcv transforms.

    Expected keys in ``results``:
        img_path (str): Path to the FITS file.

    Added keys:
        img (np.ndarray): uint8 BGR image, shape (H, W, 3).
        img_shape (tuple[int, int]): (H, W).
        ori_shape (tuple[int, int]): (H, W).
    """

    def __init__(self, to_float32: bool = False) -> None:
        """Initialise the transform.

        Args:
            to_float32: If True, cast the uint8 image to float32 before
                returning.  Default False (mmdet normalises later).
        """
        self.to_float32 = to_float32
        # Lazy-import to avoid circular imports at module load time
        self._loader: Any = None

    def _get_loader(self):
        if self._loader is None:
            from inference.fits_loader import FITSLoader
            self._loader = FITSLoader()
        return self._loader

    def transform(self, results: dict) -> dict:
        """Load FITS image.

        Args:
            results: Dict with key ``img_path``.

        Returns:
            Updated results dict with ``img``, ``img_shape``, ``ori_shape``.
        """
        img_path = str(results.get("img_path") or results.get("filename", ""))

        # Detect tiled Frigate virtual paths: stem__tx<x0>_ty<y0>_ts<size>.ext
        p = Path(img_path)
        tile_crop: tuple[int, int, int] | None = None
        m = _TILE_RE.match(p.stem)
        if m:
            real_stem, x0, y0, ts = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            img_path = str(p.parent / (real_stem + p.suffix))
            tile_crop = (x0, y0, ts)

        loader = self._get_loader()
        try:
            suffix = Path(img_path).suffix.lower()
            if suffix in {".fits", ".fit", ".fts"}:
                loaded = loader.load(img_path)
                arr = loaded["array"]  # uint8 (H, W, 3) — already normalised
            else:
                import mmcv
                arr = mmcv.imread(img_path, channel_order="bgr")
                if arr is None:
                    raise FileNotFoundError(img_path)

            if tile_crop is not None:
                x0, y0, ts = tile_crop
                h, w = arr.shape[:2]
                # Pad if image is smaller than the tile end (matches tiling logic)
                pad_h = max(0, y0 + ts - h)
                pad_w = max(0, x0 + ts - w)
                if pad_h > 0 or pad_w > 0:
                    arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
                arr = arr[y0:y0 + ts, x0:x0 + ts].copy()

        except Exception as exc:
            logger.warning("Failed to load image %s: %s — using zeros", img_path, exc)
            # Fall back to a zero image so training doesn't crash on a bad file
            ts = tile_crop[2] if tile_crop else 256
            arr = np.zeros((ts, ts, 3), dtype=np.uint8)

        if self.to_float32:
            arr = arr.astype(np.float32)

        results["img"] = arr
        results["img_shape"] = arr.shape[:2]
        results["ori_shape"] = arr.shape[:2]
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(to_float32={self.to_float32})"


if __name__ == "__main__":
    # Smoke-test: load one dev_subset FITS and print shape
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    sample = Path("data/dev_subset/dev_blank_000.fits")
    if not sample.exists():
        print(f"Sample not found at {sample} — run training/make_dev_subset.py first")
        sys.exit(1)

    t = LoadFITSFromFile()
    result = t.transform({"img_path": str(sample)})
    arr = result["img"]
    print(f"Loaded {sample.name}: shape={arr.shape} dtype={arr.dtype} "
          f"min={arr.min()} max={arr.max()}")
