"""
app.py — MDP Interactive Playground, Gradio UI (Mac Edition)
=============================================================

WHAT CHANGED vs. the original
------------------------------
• `from main import run_edit` moved to MODULE TOP-LEVEL.
  In the original it sat INSIDE run_edit_gradio() (line 108), meaning Python
  re-evaluated the import statement on every single button click.
  Although sys.modules usually short-circuits this, it still re-binds the
  name and — crucially — kept model_loader's module-init side-effects in a
  code path that could be triggered more than once under Gradio's reload mode.

• `initialize_model()` function removed entirely.
  model_loader.py now IS the initialisation: importing main (which imports
  model_loader) at module top-level guarantees from_pretrained() runs before
  the Gradio server starts, in the __main__ guard, with no function call needed.

• The Gradio callback run_edit_gradio() is now a pure UI shim:
  validate → call run_edit() → return results.  Zero ML code.

• All CSS / HTML / build_ui() are preserved verbatim.
"""

import logging
import sys
import os
import time
from typing import Optional

# Ensure project root is in Python path before any local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from PIL import Image

# ── ALL local imports at module top-level ─────────────────────────────────────
# Importing `main` here triggers model_loader → from_pretrained() exactly once.
# If this import has already run (e.g. in a test harness), Python returns the
# cached module — no second load.
from main import run_edit                          # noqa: E402
import model_loader                                # noqa: F401  (confirms load)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Gradio callback — UI logic only, no ML
# ──────────────────────────────────────────────────────────────────────────────

def run_edit_gradio(
    input_image,
    edit_prompt: str,
    original_prompt: str,
    mdp_mode: str,
    alpha: float,
    mdp_start_step: int,
    mdp_end_step: int,
    num_steps: int,
    guidance_scale: float,
    strength: float,
    seed_input: str,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Gradio-facing callback.
    Validates inputs, calls the pre-loaded pipeline, returns outputs.
    The global sampler (via run_edit) is already warm — no loading here.
    """

    # ── Validate image ────────────────────────────────────────────────────────
    if input_image is None:
        raise gr.Error("Please upload an input image.")

    # ── Validate prompt ───────────────────────────────────────────────────────
    if not edit_prompt or not edit_prompt.strip():
        raise gr.Error("Please enter an edit prompt.")

    # ── Parse seed ────────────────────────────────────────────────────────────
    seed: Optional[int] = None
    seed_str = seed_input.strip() if seed_input else ""
    if seed_str and seed_str.lower() not in ("", "none", "random"):
        try:
            seed = int(seed_str)
        except ValueError:
            raise gr.Error(f"Invalid seed: '{seed_str}'. Use an integer or leave blank.")

    # ── Validate timestep range ───────────────────────────────────────────────
    if mdp_start_step > mdp_end_step:
        raise gr.Error(
            f"MDP Start Step ({mdp_start_step}) must be ≤ End Step ({mdp_end_step})."
        )

    # ── Ensure PIL image ──────────────────────────────────────────────────────
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)

    # ── Progress helpers ──────────────────────────────────────────────────────
    progress(0, desc="Starting …")
    start_time = time.time()

    def update_progress(step: int, total: int):
        progress(step / max(total, 1), desc=f"Denoising step {step}/{total}")

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        progress(0.05, desc="Encoding image & prompts …")
        final_image, intermediates = run_edit(
            image=input_image,
            edit_prompt=edit_prompt.strip(),
            original_prompt=original_prompt.strip(),
            mdp_mode=mdp_mode,
            alpha=alpha,
            mdp_start_step=int(mdp_start_step),
            mdp_end_step=int(mdp_end_step),
            num_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            strength=float(strength),
            seed=seed,
            progress_callback=update_progress,
        )
    except ValueError as exc:
        raise gr.Error(str(exc))
    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        raise gr.Error(f"Generation failed: {exc}")

    elapsed = time.time() - start_time
    progress(1.0, desc=f"Done! ({elapsed:.1f}s)")
    logger.info("✅ Edit complete in %.1fs | intermediates=%d", elapsed, len(intermediates))

    return final_image, intermediates


# ──────────────────────────────────────────────────────────────────────────────
# CSS / HTML (preserved exactly from original)
# ──────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Syne:wght@400;600;800&display=swap');

:root {
  --bg-deep:    #0d0f14;
  --bg-panel:   #13161d;
  --bg-card:    #181c24;
  --bg-input:   #1e222d;
  --border:     #2a2f3e;
  --border-glow:#3a7bd5;
  --accent-1:   #3a7bd5;
  --accent-2:   #00d2ff;
  --accent-grad: linear-gradient(135deg, #3a7bd5, #00d2ff);
  --text-primary: #e8eaf0;
  --text-secondary: #8892a4;
  --text-muted:    #4a5168;
  --success:    #00c896;
  --warning:    #f5a623;
  --font-head:  'Syne', sans-serif;
  --font-mono:  'JetBrains Mono', monospace;
  --radius:     12px;
  --radius-lg:  18px;
  --shadow:     0 4px 24px rgba(0,0,0,0.4);
}
body, .gradio-container {
  background: var(--bg-deep) !important;
  font-family: var(--font-mono) !important;
  color: var(--text-primary) !important;
}
.mdp-header {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 28px 36px;
  margin-bottom: 20px;
  position: relative;
  overflow: hidden;
}
.mdp-header::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--accent-grad);
}
.mdp-header h1 {
  font-family: var(--font-head) !important;
  font-size: 2rem !important;
  font-weight: 800 !important;
  background: var(--accent-grad);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 6px 0 !important;
}
.mdp-header p {
  color: var(--text-secondary) !important;
  font-size: 0.82rem !important;
  margin: 0 !important;
  letter-spacing: 0.05em;
}
.mdp-badge {
  display: inline-block;
  background: rgba(58,123,213,0.15);
  border: 1px solid rgba(58,123,213,0.4);
  color: var(--accent-2) !important;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 0.7rem;
  font-family: var(--font-mono);
  letter-spacing: 0.08em;
  margin-left: 10px;
  vertical-align: middle;
}
.gr-panel, .gr-group, .gr-box, .gr-block.gr-box {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}
label, .gr-block label, .gr-form label {
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  color: var(--text-secondary) !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}
input[type="text"], textarea, .gr-textbox textarea, .gr-input input {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-primary) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.85rem !important;
  padding: 10px 14px !important;
  transition: border-color 0.2s ease !important;
}
input[type="text"]:focus, textarea:focus {
  border-color: var(--border-glow) !important;
  box-shadow: 0 0 0 2px rgba(58,123,213,0.15) !important;
  outline: none !important;
}
.btn-generate {
  background: var(--accent-grad) !important;
  color: #fff !important;
  font-family: var(--font-mono) !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  letter-spacing: 0.06em !important;
  border: none !important;
  border-radius: var(--radius) !important;
  padding: 14px 24px !important;
  width: 100% !important;
  margin-top: 16px !important;
  cursor: pointer !important;
  transition: transform 0.15s, box-shadow 0.2s !important;
}
.btn-generate:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 28px rgba(58,123,213,0.45) !important;
}
.btn-generate:active {
  transform: translateY(0) !important;
}
.gr-image, .image-container {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--bg-input) !important;
  overflow: hidden !important;
}
.gr-gallery {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}
.section-label {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--accent-1);
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-bottom: 12px;
  display: block;
}
.info-box {
  background: rgba(58,123,213,0.08);
  border: 1px solid rgba(58,123,213,0.25);
  border-radius: 8px;
  padding: 12px 16px;
  font-size: 0.75rem;
  color: var(--text-secondary);
  line-height: 1.6;
  margin-top: 8px;
}
.gr-accordion {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-1); }
"""

HEADER_HTML = """
<div class="mdp-header">
  <h1>MDP Playground <span class="mdp-badge">MAC EDITION</span></h1>
  <p>Manipulating Diffusion Path · Apple Silicon Optimized · MPS Backend · SD v1.5</p>
</div>
"""

MDP_INFO_HTML = """
<div class="info-box">
  <strong style="color: #3a7bd5;">What is MDP?</strong><br>
  MDP (Manipulating Diffusion Path) steers image editing by blending denoising
  predictions inside a selected timestep range. <br><br>
  <strong>epsilon mode</strong> — blend predicted noise directly (global edits)<br>
  <strong>x mode</strong> — blend in image space via x₀ (structure-preserving edits)
</div>
"""


# ──────────────────────────────────────────────────────────────────────────────
# UI construction (layout preserved from original)
# ──────────────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(css=CUSTOM_CSS, title="MDP Playground — Mac Edition") as demo:

        gr.HTML(HEADER_HTML)

        with gr.Row(equal_height=False):

            # ── LEFT: Controls ────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=320):
                gr.HTML('<span class="section-label">Input</span>')
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=256,
                    image_mode="RGB",
                )
                edit_prompt = gr.Textbox(
                    label="Edit Prompt",
                    placeholder="e.g. a cat wearing sunglasses",
                    lines=2,
                )
                original_prompt = gr.Textbox(
                    label="Original Prompt (optional)",
                    placeholder="e.g. a cat",
                    lines=1,
                )

                gr.HTML('<span class="section-label" style="margin-top:16px;">MDP Settings</span>')

                with gr.Row():
                    mdp_mode = gr.Dropdown(
                        choices=["epsilon", "x"],
                        value="epsilon",
                        label="MDP Mode",
                    )
                    alpha = gr.Slider(
                        minimum=0.0, maximum=1.0, value=1.0, step=0.05,
                        label="Alpha (blend strength)",
                    )

                with gr.Row():
                    mdp_start = gr.Slider(
                        minimum=0, maximum=19, value=0, step=1,
                        label="MDP Start Step",
                    )
                    mdp_end = gr.Slider(
                        minimum=0, maximum=19, value=15, step=1,
                        label="MDP End Step",
                    )

                gr.HTML(MDP_INFO_HTML)

                with gr.Accordion("⚙️  Advanced Settings", open=False):
                    with gr.Row():
                        num_steps = gr.Slider(
                            minimum=5, maximum=20, value=15, step=1,
                            label="Inference Steps",
                        )
                        guidance_scale = gr.Slider(
                            minimum=1.0, maximum=15.0, value=7.5, step=0.5,
                            label="Guidance Scale",
                        )
                    with gr.Row():
                        strength = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.75, step=0.05,
                            label="Noise Strength",
                        )
                        seed_input = gr.Textbox(
                            label="Seed (integer or blank for random)",
                            placeholder="42",
                            value="42",
                        )

                generate_btn = gr.Button(
                    "✦  Generate Edit",
                    variant="primary",
                    elem_classes=["btn-generate"],
                )

            # ── RIGHT: Outputs ────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=320):
                gr.HTML('<span class="section-label">Final Output</span>')
                output_image = gr.Image(
                    label="Edited Image",
                    type="pil",
                    height=300,
                    interactive=False,
                )

                gr.HTML('<span class="section-label" style="margin-top:16px;">Diffusion Path</span>')
                intermediate_gallery = gr.Gallery(
                    label="Intermediate Steps",
                    columns=3,
                    rows=2,
                    height=280,
                    object_fit="contain",
                    preview=True,
                )

        # ── Wire up ───────────────────────────────────────────────────────────
        generate_btn.click(
            fn=run_edit_gradio,
            inputs=[
                input_image, edit_prompt, original_prompt,
                mdp_mode, alpha, mdp_start, mdp_end,
                num_steps, guidance_scale, strength, seed_input,
            ],
            outputs=[output_image, intermediate_gallery],
            api_name="edit",
        )

        gr.Examples(
            examples=[
                [None, "a cat wearing sunglasses", "a cat",     "epsilon", 1.0, 0, 12, 15, 7.5,  0.75, "42"],
                [None, "a dog in a snowy forest",  "a dog",     "x",       0.8, 2, 14, 15, 8.0,  0.70, "123"],
                [None, "a painting in Van Gogh style", "",      "epsilon", 0.9, 0, 18, 20, 9.0,  0.85, "7"],
            ],
            inputs=[
                input_image, edit_prompt, original_prompt,
                mdp_mode, alpha, mdp_start, mdp_end,
                num_steps, guidance_scale, strength, seed_input,
            ],
            label="Example Edits (add your own image)",
        )

    return demo


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MDP Interactive Playground — Mac Edition")
    print("=" * 60)
    # model_loader was already imported at the top of this file.
    # "✅ Model loaded successfully" has already been printed.
    # Nothing to do here — just launch the server.
    print("\n🚀 Starting Gradio server on http://127.0.0.1:7860 …")
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
    )
