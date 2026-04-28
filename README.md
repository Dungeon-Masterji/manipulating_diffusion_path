# MDP Interactive Playground — Mac Edition 🍎

A lightweight, Apple Silicon–optimized diffusion-based image editing tool
inspired by **MDP (Manipulating Diffusion Path)**.

Runs fully on **MPS (Metal Performance Shaders)** — no CUDA required.

---

## What is MDP?

MDP intercepts the diffusion denoising process at selected timesteps and
**blends the noise predictions** between two prompts — the original and the
edit target. This steers the generated image toward the edit without requiring
full retraining.

Two modes are supported:
- **epsilon mode**: blend predicted noise ε directly (fast, global edits)
- **x mode**: blend in predicted image space x₀ (structure-preserving edits)

---

## Features

- 🖼 Upload any image (auto-resized to 256×256)
- ✏️ Enter an edit prompt and optional original prompt
- 🎛 Choose MDP mode: `epsilon` or `x`
- 📐 Control which timestep range gets MDP treatment
- 🌊 View intermediate denoising steps as a gallery
- 🔒 Reproducible results via seed control
- ⚡️ Text embedding + latent caching for speed
- 🍎 Pure MPS / CPU — no CUDA dependency

---

## Installation

```bash
# 1. Create a virtual environment (recommended)
python3.10 -m venv venv
source venv/bin/activate

# 2. Install PyTorch with MPS support
# Visit https://pytorch.org for the latest Mac install command
# Example:
pip install torch torchvision

# 3. Install all other dependencies
pip install -r requirements.txt
```

> **Note:** First run downloads Stable Diffusion v1.5 (~4 GB) from HuggingFace.
> It is cached in `~/.cache/huggingface/` for subsequent runs.

---

## Usage

```bash
# From the project root:
python app.py
```

The app opens automatically at `http://127.0.0.1:7860`.

---

## Project Structure

```
mdp_playground/
│
├── app.py              ← Gradio UI entry point
├── main.py             ← Pipeline orchestrator
├── requirements.txt
│
├── mdp/
│   ├── __init__.py
│   ├── base.py         ← Abstract MDP strategy
│   ├── mdp_epsilon.py  ← Epsilon-space editing
│   └── mdp_x.py        ← X0-space editing
│
├── sampler/
│   ├── __init__.py
│   └── img2img.py      ← Core DDIM + MDP loop
│
└── utils/
    ├── __init__.py
    ├── device.py        ← MPS/CPU detection
    ├── image.py         ← Tensor ↔ PIL helpers
    └── caching.py       ← Embedding & latent cache
```

---

## Parameters Guide

| Parameter | Description | Range |
|-----------|-------------|-------|
| **Edit Prompt** | Describe the desired edit | text |
| **Original Prompt** | Describe the source image (optional) | text |
| **MDP Mode** | `epsilon` = noise space, `x` = image space | dropdown |
| **Alpha** | How strongly to apply the edit (0=original, 1=full edit) | 0.0–1.0 |
| **MDP Start Step** | First denoising step to apply MDP | 0–19 |
| **MDP End Step** | Last denoising step to apply MDP | 0–19 |
| **Inference Steps** | Total DDIM steps (more = slower but better) | 5–20 |
| **Guidance Scale** | CFG scale (higher = more prompt-aligned) | 1.0–15.0 |
| **Noise Strength** | How much noise to add (lower = closer to original) | 0.1–1.0 |
| **Seed** | Integer for reproducible outputs | integer |

---

## Tips

- Start with **epsilon** mode and `alpha=1.0` for straightforward edits.
- Use **x mode** to preserve more of the original structure.
- Lower `strength` (0.5–0.7) to keep more of the original image.
- Apply MDP only in the **middle timesteps** (e.g. start=3, end=12) for
  balanced results — early steps set global structure, late steps add detail.
- Increase `guidance_scale` (8–12) for stronger adherence to the edit prompt.

---

## Performance

On Apple M1/M2/M3:
- ~30–90 seconds per generation (256×256, 15 steps)
- First run is slower due to model compilation on MPS
- Subsequent edits on the same image are faster (latent cache)

---

## License

MIT License. Model weights are subject to their original licenses.
Stable Diffusion v1.5: CreativeML Open RAIL-M License.
