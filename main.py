"""
main.py — Pipeline Orchestration
==================================
Imports the global sampler from model_loader and exposes run_edit().

WHAT CHANGED vs. the original
------------------------------
• load_pipeline() is GONE — model_loader.py owns that responsibility.
• ALL imports are at module top-level (not inside functions).
• run_edit() never touches StableDiffusionPipeline; it only calls sampler.run().
• MDP strategy objects (MDPEpsilon / MDPX0) are still instantiated per-request —
  they are tiny stateless objects, not model weights.
"""

import logging
from typing import List, Optional, Tuple

from PIL import Image

# ── Top-level imports — executed once when main.py is first imported ──────────
from mdp.mdp_epsilon import MDPEpsilon
from mdp.mdp_x import MDPX0
from utils.image import load_and_resize

# model_loader is the singleton: importing it here triggers from_pretrained()
# on first import, then Python's module cache takes over for every subsequent
# import anywhere in the process.
import model_loader   # noqa: F401  (side-effect import: loads model + sampler)
from model_loader import sampler  # the single Img2ImgMDPSampler instance

logger = logging.getLogger(__name__)


def run_edit(
    image: Image.Image,
    edit_prompt: str,
    original_prompt: str = "",
    mdp_mode: str = "epsilon",
    alpha: float = 1.0,
    mdp_start_step: int = 0,
    mdp_end_step: int = 10,
    num_steps: int = 15,
    guidance_scale: float = 7.5,
    strength: float = 0.75,
    seed: Optional[int] = None,
    progress_callback=None,
) -> Tuple[Image.Image, List[Image.Image]]:
    """
    Run MDP-guided image editing.

    The global `sampler` imported from model_loader is reused on every call —
    no weights are loaded, moved, or re-initialised here.

    Args:
        image:             Input PIL image (any size — resized internally).
        edit_prompt:       Text describing the desired edit.
        original_prompt:   Text describing the original image (may be empty).
        mdp_mode:          "epsilon" or "x".
        alpha:             MDP blend factor (0.0 = original, 1.0 = fully edited).
        mdp_start_step:    First denoising step index to apply MDP.
        mdp_end_step:      Last  denoising step index to apply MDP.
        num_steps:         Total DDIM denoising steps (keep ≤ 15 for speed).
        guidance_scale:    Classifier-free guidance scale.
        strength:          Img2img noise strength (0 = no change, 1 = full regen).
        seed:              Optional RNG seed for reproducibility.
        progress_callback: Optional callable(step: int, total: int).

    Returns:
        (final_image, list_of_intermediate_pil_images)
    """
    logger.info(
        "🚀 Starting inference | mode=%s | alpha=%.2f | steps=%d | strength=%.2f",
        mdp_mode, alpha, num_steps, strength,
    )

    # ── Input validation ──────────────────────────────────────────────────────
    if not edit_prompt or not edit_prompt.strip():
        raise ValueError("Edit prompt cannot be empty.")

    if mdp_start_step > mdp_end_step:
        raise ValueError(
            f"MDP start step ({mdp_start_step}) must be ≤ end step ({mdp_end_step})."
        )

    # Clamp end step to valid range
    mdp_end_step = min(mdp_end_step, num_steps - 1)

    # ── Pre-processing ────────────────────────────────────────────────────────
    image = load_and_resize(image)   # → 256 × 256 RGB PIL

    # ── Select MDP strategy (tiny stateless object, cheap to construct) ───────
    if mdp_mode == "epsilon":
        strategy = MDPEpsilon(alpha=alpha)
    elif mdp_mode == "x":
        strategy = MDPX0(alpha=alpha)
    else:
        raise ValueError(f"Unknown MDP mode: '{mdp_mode}'. Use 'epsilon' or 'x'.")

    # ── Build capture step list (≈6 evenly spaced intermediates) ─────────────
    effective_steps = max(1, int(num_steps * strength))
    step_interval   = max(1, effective_steps // 6)
    capture_steps   = list(range(0, effective_steps, step_interval))

    # ── Delegate to the global sampler — NO model loading here ───────────────
    final_image, intermediates = sampler.run(
        image=image,
        edit_prompt=edit_prompt.strip(),
        original_prompt=original_prompt.strip(),
        mdp_strategy=strategy,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        mdp_start_step=mdp_start_step,
        mdp_end_step=mdp_end_step,
        capture_steps=capture_steps,
        seed=seed,
        progress_callback=progress_callback,
    )

    logger.info("✅ Inference complete | intermediates=%d", len(intermediates))
    return final_image, intermediates
