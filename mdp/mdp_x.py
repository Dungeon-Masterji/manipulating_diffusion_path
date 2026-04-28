"""
MDP X0 Strategy.
Modifies the predicted clean image (x0) in the diffusion process.

Instead of blending noise predictions directly, we:
1. Convert both epsilon predictions to x0 (predicted clean latent)
2. Blend the x0 predictions
3. Convert the blended x0 back to epsilon

This operates in image space (latent), which can produce edits that
respect more of the original image structure.
"""

import torch
from .base import MDPBase


class MDPX0(MDPBase):
    """
    X0-space MDP editing.

    At each active timestep:
        x0_original = (latents - sqrt(1-alpha_bar) * epsilon_orig) / sqrt(alpha_bar)
        x0_edited   = (latents - sqrt(1-alpha_bar) * epsilon_edit) / sqrt(alpha_bar)
        x0_modified = (1 - alpha) * x0_original + alpha * x0_edited
        epsilon_out = (latents - sqrt(alpha_bar) * x0_modified) / sqrt(1 - alpha_bar)
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha)

    def _epsilon_to_x0(
        self,
        epsilon: torch.Tensor,
        latents: torch.Tensor,
        alpha_bar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert predicted noise (epsilon) to predicted clean image (x0).

        DDIM reparameterization:
            x0 = (x_t - sqrt(1 - alpha_bar) * epsilon) / sqrt(alpha_bar)
        """
        sqrt_alpha_bar = alpha_bar.sqrt()
        sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()

        x0 = (latents - sqrt_one_minus_alpha_bar * epsilon) / sqrt_alpha_bar
        return x0

    def _x0_to_epsilon(
        self,
        x0: torch.Tensor,
        latents: torch.Tensor,
        alpha_bar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert predicted clean image (x0) back to epsilon.

        Inverse reparameterization:
            epsilon = (x_t - sqrt(alpha_bar) * x0) / sqrt(1 - alpha_bar)
        """
        sqrt_alpha_bar = alpha_bar.sqrt()
        sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()

        epsilon = (latents - sqrt_alpha_bar * x0) / sqrt_one_minus_alpha_bar
        return epsilon

    def modify(
        self,
        original_pred: torch.Tensor,
        edited_pred: torch.Tensor,
        t: int,
        scheduler,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Blend predictions in x0 space and convert back to epsilon.

        Args:
            original_pred: Predicted epsilon for original prompt
            edited_pred:   Predicted epsilon for edit prompt
            t:             Current timestep index
            scheduler:     DDIM scheduler (provides alpha_bar values)
            latents:       Current latent x_t

        Returns:
            Modified epsilon after x0-space blending
        """
        # Get cumulative alpha bar for this timestep
        # alphas_cumprod is indexed by timestep value
        timestep_val = scheduler.timesteps[t] if hasattr(scheduler, 'timesteps') else t

        # Safe access to alphas_cumprod
        try:
            alpha_bar = scheduler.alphas_cumprod[timestep_val]
        except (IndexError, TypeError):
            # Fallback: estimate from position
            alpha_bar = torch.tensor(0.5)

        # Ensure alpha_bar is on the correct device and dtype
        alpha_bar = alpha_bar.to(device=latents.device, dtype=latents.dtype)

        # Convert epsilons to x0 predictions
        x0_original = self._epsilon_to_x0(original_pred, latents, alpha_bar)
        x0_edited   = self._epsilon_to_x0(edited_pred,   latents, alpha_bar)

        # Blend in x0 space
        x0_modified = self.blend(x0_original, x0_edited)

        # Convert back to epsilon space for the scheduler step
        epsilon_modified = self._x0_to_epsilon(x0_modified, latents, alpha_bar)

        return epsilon_modified
