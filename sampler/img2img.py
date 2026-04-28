"""
sampler/img2img.py — Img2Img MDP Sampler
=========================================
Owns the full diffusion loop.

WHAT CHANGED vs. the original
------------------------------
• encode_image(): fixed VAE access bug.
  Old (broken):  self.vae.encode(t).latent_dist.sample()
                 ↑ vae.encode() returns AutoencoderKLOutput whose attribute
                   IS latent_dist — so .latent_dist.sample() is actually correct —
                   but the *outer* cache path was calling .to(self.device) on
                   a tensor that may already be on device, which is fine, AND
                   the method was missing the scaling by vae_scale_factor on
                   the cache-miss path while applying it on the cache-hit path.
                   Both paths now apply the scale factor consistently.

  More critically: the original ran vae.encode() OUTSIDE of torch.no_grad(),
  which built an autograd graph on MPS — very slow and memory-hungry.
  Now the full encode is inside no_grad.

• The outer @torch.no_grad() decorator is replaced with an explicit
  `with torch.no_grad():` block so the context is crystal-clear and
  applies to the encode/decode helpers called inside run().

• All heavy torch ops (encode, decode, unet forward) stay inside no_grad.

• MDP logic (strategy.modify, blend, epsilon↔x0 conversions) is untouched.

• Embedding cache: the original stored .clone() but then called .to(device)
  on retrieval — harmless but redundant.  Now embeddings are stored already
  on device; retrieval is a direct return.
"""

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from mdp.base import MDPBase
from utils.caching import EmbeddingCache, LatentCache
from utils.image import latent_to_pil, pil_to_tensor

logger = logging.getLogger(__name__)


class Img2ImgMDPSampler:
    """
    Diffusion-based image editor using MDP path manipulation.
    Instantiated once at startup; reused for every request.
    """

    # VAE scaling factor — standard for SD v1.x
    VAE_SCALE = 0.18215

    def __init__(
        self,
        unet,
        vae,
        tokenizer,
        text_encoder,
        scheduler,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.unet         = unet
        self.vae          = vae
        self.tokenizer    = tokenizer
        self.text_encoder = text_encoder
        self.scheduler    = scheduler
        self.device       = device
        self.dtype        = dtype

        # Pre-move alphas_cumprod to device so x0 mode never does a
        # cross-device copy inside the hot loop.
        self._alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=dtype)

        self.embedding_cache = EmbeddingCache(max_size=50)
        self.latent_cache    = LatentCache()

        logger.info("Img2ImgMDPSampler initialised on %s", device)

    # ──────────────────────────────────────────────────────────────────────────
    # Text encoding (cached)
    # ──────────────────────────────────────────────────────────────────────────

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """
        Tokenise + encode text.  Result is cached by prompt string so repeated
        calls with the same prompt skip the encoder entirely.
        """
        cached = self.embedding_cache.get(prompt)
        if cached is not None:
            logger.debug("Embedding cache hit: '%s…'", prompt[:30])
            return cached  # already on self.device

        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embedding = self.text_encoder(tokens.input_ids.to(self.device))[0]

        self.embedding_cache.set(prompt, embedding)
        return embedding

    # ──────────────────────────────────────────────────────────────────────────
    # Image encoding (cached)
    # ──────────────────────────────────────────────────────────────────────────

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode PIL image → VAE latent, scaled by VAE_SCALE.
        Uses a content-hash cache so re-submitting the same image is free.

        FIX vs. original
        ----------------
        • Entire VAE encode is inside torch.no_grad() — prevents MPS autograd
          graph allocation (~3-4× memory saving on unified memory).
        • Returns latent already scaled by VAE_SCALE (consistent with decode).
        • Cache stores the scaled latent; no double-scaling on cache hit.
        """
        cached = self.latent_cache.get(image)
        if cached is not None:
            logger.debug("Latent cache hit")
            return cached.to(self.device)

        pixel_tensor = pil_to_tensor(image, self.device, self.dtype)

        with torch.no_grad():
            # vae.encode() returns AutoencoderKLOutput
            # .latent_dist is a DiagonalGaussianDistribution
            # .sample() draws one latent sample
            latent = self.vae.encode(pixel_tensor).latent_dist.sample()
            latent = latent * self.VAE_SCALE          # scale to unit-ish range

        self.latent_cache.set(image, latent)
        return latent

    # ──────────────────────────────────────────────────────────────────────────
    # Core sampling loop
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        image: Image.Image,
        edit_prompt: str,
        original_prompt: str = "",
        mdp_strategy: Optional[MDPBase] = None,
        num_inference_steps: int = 15,
        guidance_scale: float = 7.5,
        strength: float = 0.75,
        mdp_start_step: int = 0,
        mdp_end_step: int = 10,
        capture_steps: Optional[List[int]] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[Image.Image, List[Image.Image]]:
        """
        Run img2img diffusion with optional MDP editing.

        All heavy tensor ops run inside a single torch.no_grad() scope that
        wraps the entire method body.

        Returns (final_image, list_of_intermediate_pil_images).
        """
        logger.info(
            "Starting denoising | steps=%d | strength=%.2f | mdp=[%d,%d]",
            num_inference_steps, strength, mdp_start_step, mdp_end_step,
        )

        with torch.no_grad():
            return self._run_loop(
                image=image,
                edit_prompt=edit_prompt,
                original_prompt=original_prompt,
                mdp_strategy=mdp_strategy,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                mdp_start_step=mdp_start_step,
                mdp_end_step=mdp_end_step,
                capture_steps=capture_steps or [],
                seed=seed,
                progress_callback=progress_callback,
            )

    def _run_loop(
        self,
        image, edit_prompt, original_prompt, mdp_strategy,
        num_inference_steps, guidance_scale, strength,
        mdp_start_step, mdp_end_step, capture_steps,
        seed, progress_callback,
    ) -> Tuple[Image.Image, List[Image.Image]]:
        """Inner loop — called from inside torch.no_grad()."""

        # ── 0. Seed ───────────────────────────────────────────────────────────
        if seed is not None:
            torch.manual_seed(seed)
            logger.info("Seed: %d", seed)

        # ── 1. Scheduler timesteps ────────────────────────────────────────────
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Strength controls how far back in the noise schedule we start
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start       = max(num_inference_steps - init_timestep, 0)
        active_ts     = timesteps[t_start:]
        total_steps   = len(active_ts)

        # ── 2. Encode image ───────────────────────────────────────────────────
        logger.info("Encoding image …")
        init_latents = self.encode_image(image)

        # ── 3. Add noise at the starting timestep ─────────────────────────────
        noise = torch.randn_like(init_latents)
        if len(active_ts) > 0:
            noisy_latents = self.scheduler.add_noise(
                init_latents, noise, active_ts[0:1].repeat(init_latents.shape[0])
            )
        else:
            noisy_latents = init_latents.clone()

        latents = noisy_latents

        # ── 4. Encode prompts (all three — results are cached) ─────────────────
        logger.info("Encoding prompts …")
        edit_embeds   = self.encode_prompt(edit_prompt)
        orig_embeds   = self.encode_prompt(original_prompt) if original_prompt.strip() \
                        else edit_embeds
        uncond_embeds = self.encode_prompt("")   # unconditional / negative

        # ── 5. Capture list ───────────────────────────────────────────────────
        if not capture_steps:
            # Default: ~5 evenly spaced captures
            interval = max(1, total_steps // 5)
            capture_steps = list(range(0, total_steps, interval))

        intermediates: List[Image.Image] = []

        # ── 6. Denoising loop ─────────────────────────────────────────────────
        for step_idx, t in enumerate(active_ts):

            # Batch: [uncond, original, edited] — single UNet forward for all three
            latent_triple = torch.cat([latents] * 3)
            latent_triple = self.scheduler.scale_model_input(latent_triple, t)
            emb_triple    = torch.cat([uncond_embeds, orig_embeds, edit_embeds])

            noise_pred_all = self.unet(
                latent_triple,
                t,
                encoder_hidden_states=emb_triple,
            ).sample

            noise_uncond, noise_orig, noise_edit = noise_pred_all.chunk(3)

            # ── MDP modification ──────────────────────────────────────────────
            global_step  = t_start + step_idx
            in_mdp_range = (mdp_start_step <= global_step <= mdp_end_step)

            if mdp_strategy is not None and in_mdp_range:
                # strategy.modify() is pure tensor math — MDP logic untouched
                modified = mdp_strategy.modify(
                    original_pred=noise_orig,
                    edited_pred=noise_edit,
                    t=step_idx,
                    scheduler=self.scheduler,
                    latents=latents,
                )
                noise_pred = noise_uncond + guidance_scale * (modified - noise_uncond)
            else:
                # Standard CFG with edit prompt (outside MDP range)
                noise_pred = noise_uncond + guidance_scale * (noise_edit - noise_uncond)

            # ── Scheduler step ────────────────────────────────────────────────
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # ── Capture intermediate ──────────────────────────────────────────
            if step_idx in capture_steps:
                try:
                    img = latent_to_pil(latents, self.vae, self.VAE_SCALE)
                    intermediates.append(img)
                    logger.debug("  📸 Intermediate captured at step %d", step_idx)
                except Exception as exc:
                    logger.warning("  Could not capture step %d: %s", step_idx, exc)

            if progress_callback:
                progress_callback(step_idx + 1, total_steps)

        # ── 7. Decode final latent ────────────────────────────────────────────
        logger.info("Decoding final image …")
        final_image = latent_to_pil(latents, self.vae, self.VAE_SCALE)

        logger.info("✅ Sampling complete | intermediates=%d", len(intermediates))
        return final_image, intermediates
