"""
model_loader.py
===============
Global singleton for the Stable Diffusion pipeline and sampler.

HOW THE SINGLETON WORKS
------------------------
Python's import system executes a module's body exactly once per interpreter
process and then caches it in sys.modules.  Every subsequent
`import model_loader` or `from model_loader import sampler` returns the
cached module — no re-execution, no second from_pretrained() call.

This is intentionally stronger than a dict-guard (like the old
_pipeline_state["loaded"] pattern), which only works if the same module
object is reused.  The import cache guarantee holds even across threads.

NOTHING in this file may be called at request time to "reload" anything.
Inference code must only *read* the exported names below.
"""

import logging
import sys

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from utils.device import get_device, get_dtype

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Device — resolved once
# ──────────────────────────────────────────────────────────────────────────────

device: torch.device = get_device()
dtype: torch.dtype   = get_dtype(device)

print(f"🖥️  Using device : {device}")
print(f"    dtype        : {dtype}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Pipeline — from_pretrained() called ONCE at module level
# ──────────────────────────────────────────────────────────────────────────────

MODEL_ID = "runwayml/stable-diffusion-v1-5"

print(f"⏳ Loading '{MODEL_ID}' — this happens exactly once per process …")

_pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False,
)

# DDIM gives fast, deterministic sampling (replaces the default PNDM)
_pipe.scheduler = DDIMScheduler.from_config(_pipe.scheduler.config)

_pipe = _pipe.to(device)
_pipe.enable_attention_slicing()   # reduces peak unified-memory usage on MPS

# ──────────────────────────────────────────────────────────────────────────────
# 3. Expose individual components
#    (Img2ImgMDPSampler receives these; it never touches the full pipeline again)
# ──────────────────────────────────────────────────────────────────────────────

unet         = _pipe.unet
vae          = _pipe.vae
tokenizer    = _pipe.tokenizer
text_encoder = _pipe.text_encoder
scheduler    = _pipe.scheduler

# ──────────────────────────────────────────────────────────────────────────────
# 4. Sampler — constructed once from the already-loaded components
#    The local import is placed here (after the components are ready) to avoid
#    a circular import: sampler/img2img.py imports from utils, not model_loader.
# ──────────────────────────────────────────────────────────────────────────────

from sampler.img2img import Img2ImgMDPSampler  # noqa: E402 — intentional late import

sampler: Img2ImgMDPSampler = Img2ImgMDPSampler(
    unet=unet,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
    device=device,
    dtype=dtype,
)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Confirmation log — if you see this more than once, something is wrong
# ──────────────────────────────────────────────────────────────────────────────

print("✅ Model loaded successfully — ready to serve all requests")
print(f"   sys.modules key : 'model_loader' (id={id(sys.modules.get('model_loader'))})")
