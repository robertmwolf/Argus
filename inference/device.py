"""Hardware-aware device selection and configuration.

Provides a single source of truth for device selection (CUDA → MPS → CPU)
and device-appropriate training hyperparameters. Every module that touches
PyTorch tensors must import get_device() from here — never hardcode
torch.device('cuda') or call .cuda() directly.

Usage::

    from inference.device import get_device, get_device_config

    device = get_device()
    cfg    = get_device_config()
    model  = model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available compute device in priority order.

    Priority: CUDA (cloud GPU) → MPS (Apple Silicon) → CPU.

    The PYTORCH_ENABLE_MPS_FALLBACK=1 environment variable should be set
    before running on Apple Silicon so that ops without native MPS kernels
    silently fall back to CPU rather than raising an error.

    Returns:
        A ``torch.device`` instance for the selected backend.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
        except Exception:
            logger.info("Using CUDA")
    elif _mps_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU available — using CPU (training will be slow)")
    return device


def get_device_config() -> dict:
    """Return device-appropriate training hyperparameters.

    Values are chosen to be safe and correct for each backend:

    +-------------------------+------+-------+-----+
    | Key                     | CUDA |  MPS  | CPU |
    +=========================+======+=======+=====+
    | batch_size              |  2   |   1   |  1  |
    | num_workers             |  4   |   0   |  2  |
    | pin_memory              | True | False | False|
    | image_size              | 800  |  400  | 400 |
    | mixed_precision         | True | False | False|
    | gradient_checkpointing  | True |  True | True|
    +-------------------------+------+-------+-----+

    MPS notes:
    - ``num_workers=0``: multiprocessing DataLoader workers crash on MPS.
    - ``pin_memory=False``: pinned memory is not supported on MPS.
    - ``mixed_precision=False``: torch.autocast MPS support is incomplete.

    Returns:
        Dict with keys: ``batch_size``, ``num_workers``, ``pin_memory``,
        ``image_size``, ``mixed_precision``, ``gradient_checkpointing``.
    """
    device = get_device()

    _configs: dict[str, dict] = {
        "cuda": {
            "batch_size": 2,
            "num_workers": 4,
            "pin_memory": True,
            "image_size": 800,
            "mixed_precision": True,
            "gradient_checkpointing": True,
        },
        "mps": {
            "batch_size": 1,
            "num_workers": 0,        # MPS doesn't support multiprocessing workers
            "pin_memory": False,     # not supported on MPS
            "image_size": 400,       # halved to fit in 16 GB unified memory
            "mixed_precision": False,  # MPS autocast support is incomplete
            "gradient_checkpointing": True,
        },
        "cpu": {
            "batch_size": 1,
            "num_workers": 2,
            "pin_memory": False,
            "image_size": 400,
            "mixed_precision": False,
            "gradient_checkpointing": True,
        },
    }

    return _configs[device.type]


def safe_autocast(device: torch.device):
    """Return a torch.autocast context manager that is safe on all backends.

    On CUDA, enables AMP (fp16). On MPS and CPU, returns a no-op context
    because autocast support on those backends is incomplete or absent.

    Args:
        device: The active compute device returned by ``get_device()``.

    Returns:
        A context manager compatible with ``with safe_autocast(device):``.

    Example::

        device = get_device()
        with safe_autocast(device):
            outputs = model(inputs)
    """
    use_amp = device.type == "cuda"
    return torch.autocast(device_type=device.type, enabled=use_amp)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mps_available() -> bool:
    """Return True if MPS is available and not explicitly disabled."""
    if not hasattr(torch.backends, "mps"):
        return False
    if not torch.backends.mps.is_available():
        return False
    # Respect opt-out env var for CI environments that lack Metal
    if os.environ.get("DISABLE_MPS", "").lower() in ("1", "true", "yes"):
        return False
    return True


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    device = get_device()
    cfg = get_device_config()

    print(f"\nDevice : {device}")
    print("Config :")
    for k, v in cfg.items():
        print(f"  {k:<25} {v}")

    # Verify a simple tensor round-trip on the selected device
    t = torch.tensor([1.0, 2.0, 3.0]).to(device)
    result = (t * 2).sum().item()
    assert result == 12.0, f"Unexpected result: {result}"
    print(f"\nTensor smoke-test passed on {device}  (sum={result})")

    # Verify safe_autocast doesn't raise
    with safe_autocast(device):
        t2 = torch.tensor([1.0]).to(device) + 1.0
    print(f"safe_autocast smoke-test passed  (value={t2.item()})")
