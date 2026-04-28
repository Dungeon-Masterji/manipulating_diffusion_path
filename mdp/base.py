"""
MDP Base Class.
Defines the interface for Manipulating Diffusion Path strategies.

The MDP concept: during diffusion denoising, we can intercept the
intermediate predictions and blend them between "original" and "edited"
directions to steer the output toward the edit prompt.
"""

import torch
from abc import ABC, abstractmethod
from typing import Tuple


class MDPBase(ABC):
    """
    Abstract base for MDP editing strategies.
    Subclasses implement specific manipulation of either:
      - epsilon (the predicted noise)
      - x0 (the predicted clean image)
    """

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Blending factor. 0.0 = original only, 1.0 = edited only.
        """
        self.alpha = alpha

    @abstractmethod
    def modify(
        self,
        original_pred: torch.Tensor,
        edited_pred: torch.Tensor,
        t: int,
        scheduler,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Modify the diffusion prediction.

        Args:
            original_pred: UNet output for original/null prompt
            edited_pred:   UNet output for edit prompt
            t:             Current timestep
            scheduler:     DDIM scheduler (for alpha/beta access)
            latents:       Current latent state

        Returns:
            Modified prediction tensor
        """
        pass

    def blend(self, original: torch.Tensor, edited: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between original and edited predictions."""
        return (1.0 - self.alpha) * original + self.alpha * edited
