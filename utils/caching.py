"""
Caching utilities.
Caches text embeddings and VAE latents to avoid redundant computation.
"""

import torch
import hashlib
from typing import Optional, Dict, Tuple
from PIL import Image
import numpy as np


class EmbeddingCache:
    """
    Simple in-memory cache for text embeddings.
    Key: (prompt_text, guidance_scale) hash
    """

    def __init__(self, max_size: int = 50):
        self._cache: Dict[str, torch.Tensor] = {}
        self.max_size = max_size

    def _make_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()

    def get(self, prompt: str) -> Optional[torch.Tensor]:
        key = self._make_key(prompt)
        return self._cache.get(key, None)

    def set(self, prompt: str, embedding: torch.Tensor) -> None:
        if len(self._cache) >= self.max_size:
            # Evict oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        key = self._make_key(prompt)
        self._cache[key] = embedding.clone()

    def clear(self) -> None:
        self._cache.clear()


class LatentCache:
    """
    Cache for VAE-encoded image latents.
    Key: image content hash
    """

    def __init__(self):
        self._cache: Dict[str, torch.Tensor] = {}
        self._last_image_hash: Optional[str] = None
        self._last_latent: Optional[torch.Tensor] = None

    def _image_hash(self, image: Image.Image) -> str:
        arr = np.array(image)
        return hashlib.md5(arr.tobytes()).hexdigest()

    def get(self, image: Image.Image) -> Optional[torch.Tensor]:
        h = self._image_hash(image)
        if h == self._last_image_hash and self._last_latent is not None:
            return self._last_latent.clone()
        return None

    def set(self, image: Image.Image, latent: torch.Tensor) -> None:
        h = self._image_hash(image)
        self._last_image_hash = h
        self._last_latent = latent.clone()

    def clear(self) -> None:
        self._last_image_hash = None
        self._last_latent = None
