"""
MDP Epsilon Strategy.
Directly modifies the predicted noise (epsilon) in the diffusion process.

This is the simplest form of MDP: instead of following the original
noise prediction, we blend it toward the noise predicted using the edit prompt.
The effect: gradually steers denoising toward the edit.
"""

import torch
from .base import MDPBase


class MDPEpsilon(MDPBase):
    """
    Epsilon-space MDP editing.

    At each active timestep:
        epsilon_modified = (1 - alpha) * epsilon_original + alpha * epsilon_edited

    Working in epsilon space is more direct—we're steering the noise
    residual itself, which tends to produce globally coherent edits.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha)

    def modify(
        self,
        original_pred: torch.Tensor,
        edited_pred: torch.Tensor,
        t: int,
        scheduler,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Blend original and edited epsilon predictions.

        Args:
            original_pred: Predicted noise for original prompt (or unconditional)
            edited_pred:   Predicted noise for edit prompt
            t:             Current timestep (unused here, alpha is fixed)
            scheduler:     DDIM scheduler
            latents:       Current latent (unused for epsilon blending)

        Returns:
            Blended epsilon prediction
        """
        # Simple linear interpolation in noise space
        modified_epsilon = self.blend(original_pred, edited_pred)
        return modified_epsilon
