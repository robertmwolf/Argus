"""Custom mmcv/mmdet data transforms for FITS image loading.

Register this module before building the MMDetection runner so that
``LoadFITSFromFile`` is available in the transform registry.

Import by adding to the MMDetection config::

    custom_imports = dict(imports=['training.transforms'], allow_failed_imports=False)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

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
    """Load a FITS telescope image and convert it to uint8 BGR for mmcv.

    Uses ``inference.FITSLoader`` for Z-score normalisation and uint8
    conversion (identical to the inference pipeline).  The result is
    compatible with all downstream mmcv transforms (``Resize``,
    ``RandomFlip``, ``Normalize``, etc.).

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
        img_path = results.get("img_path") or results.get("filename", "")
        loader = self._get_loader()
        try:
            loaded = loader.load(img_path)
            arr = loaded["array"]  # uint8 (H, W, 3) — already Z-score → uint8
        except Exception as exc:
            logger.warning("Failed to load FITS %s: %s — using zeros", img_path, exc)
            # Fall back to a zero image so training doesn't crash on a bad file
            arr = np.zeros((256, 256, 3), dtype=np.uint8)

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
