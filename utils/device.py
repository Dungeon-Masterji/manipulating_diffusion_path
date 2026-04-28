"""
Device management utility.
Detects and configures the best available device (MPS > CPU).
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Returns the best available device for inference.
    Priority: MPS (Apple Silicon) > CPU
    CUDA is intentionally excluded for Mac-only deployment.
    """
    if torch.backends.mps.is_available():
        logger.info("✅ Using MPS (Apple Silicon) backend")
        return torch.device("mps")
    else:
        logger.warning("⚠️  MPS not available, falling back to CPU")
        return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """
    Returns the appropriate dtype for the device.
    MPS can be unstable with float16, so we use float32 everywhere.
    """
    # Always use float32 for stability on MPS and CPU
    return torch.float32


def move_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Safely move a tensor to the target device."""
    return tensor.to(device)
