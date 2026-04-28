"""
Image processing utilities.
Handles loading, resizing, tensor conversion for the diffusion pipeline.
"""

import numpy as np
import torch
from PIL import Image
from typing import Union


TARGET_SIZE = (256, 256)


def load_and_resize(image: Union[Image.Image, np.ndarray, str], size: tuple = TARGET_SIZE) -> Image.Image:
    """
    Load and resize image to target dimensions.
    Accepts PIL Image, numpy array, or file path.
    """
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Convert to RGB (handles RGBA, grayscale, etc.)
    image = image.convert("RGB")
    image = image.resize(size, Image.LANCZOS)
    return image


def pil_to_tensor(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert PIL Image to normalized tensor in range [-1, 1].
    Shape: [1, 3, H, W]
    """
    arr = np.array(image).astype(np.float32) / 255.0  # [0, 1]
    arr = (arr * 2.0) - 1.0                            # [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return tensor.to(device=device, dtype=dtype)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert decoded tensor back to PIL Image.
    Expects tensor in range [-1, 1], shape [1, 3, H, W] or [3, H, W].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dim -> [3, H, W]

    # Clamp and convert to [0, 255]
    tensor = tensor.detach().cpu().float()
    tensor = (tensor.clamp(-1, 1) + 1) / 2.0  # [0, 1]
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def latent_to_pil(latents: torch.Tensor, vae, scale_factor: float = 0.18215) -> Image.Image:
    """
    Decode latents using VAE decoder and return PIL image.
    """
    with torch.no_grad():
        # Scale latents back before decoding
        scaled = latents / scale_factor
        decoded = vae.decode(scaled).sample
    return tensor_to_pil(decoded)
