import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import traceback
import numpy as np
import torch
import gradio as gr
from PIL import Image

from configs.hf_config import (
    HF_REPO_ID,
    CHECKPOINT_NAME,
    APP_TITLE,
    APP_DESCRIPTION,
    ARTICLE,
    LOCAL_CHECKPOINT_PATH,
)
from src.inference.predict import InferencePipeline


# ---------------------------------------------------------------------------
# Model bootstrap
# ---------------------------------------------------------------------------
print("Initializing LAPANet model...")
try:
    pipeline = InferencePipeline(
        repo_id=HF_REPO_ID,
        checkpoint_filename=CHECKPOINT_NAME,
    )
    MODEL_SOURCE = f"Hugging Face Hub · {HF_REPO_ID}"
except Exception as e:
    print(f"Failed to load from HF Hub: {e}. Attempting local load...")
    pipeline = InferencePipeline(local_path=LOCAL_CHECKPOINT_PATH)
    MODEL_SOURCE = f"Local checkpoint · {LOCAL_CHECKPOINT_PATH}"

DEVICE = str(pipeline.device).upper()
OUT_SHAPE = getattr(pipeline, "out_shape", ("—", "—"))

print(f"Model ready on {DEVICE} from {MODEL_SOURCE}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_bytes(num_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def _as_5d(kspace: np.ndarray) -> np.ndarray:
    if kspace.ndim != 5:
        raise ValueError(
            f"Expected a 5D k-space array with shape (Frames, Slices, Coils, H, W); "
            f"got ndim={kspace.ndim} with shape {kspace.shape}. "
            f"Reshape your file to 5D before uploading."
        )
    return kspace


def _inspect_kspace(path):
    info = {
        "format": Path(path).suffix.lower(),
        "shape": "—",
        "dims": "—",
        "dtype": "—",
        "size": "—",
        "layout": "—",
    }
    try:
        info["size"] = _format_bytes(Path(path).stat().st_size)
        ksp = pipeline._load_kspace_file(path)
        info["shape"] = " × ".join(str(s) for s in ksp.shape)
        info["dtype"] = str(ksp.dtype)
        try:
            ksp5 = _as_5d(ksp)
            F, S, C, H, W = ksp5.shape
            info["dims"] = f"F={F} · S={S} · C={C} · H={H} · W={W}"
            info["layout"] = "Frames × Slices × Coils × H × W"
        except Exception:
            info["dims"] = f"{ksp.ndim}D"
            info["layout"] = f"{ksp.ndim}-D array"
    except Exception as e:
        info["error"] = str(e)
    return info


def _info_markdown(info):
    if "error" in info:
        return f"**Could not inspect file**\n\n```\n{info['error']}\n```"

    return f"""
**Format** · `{info['format']}`  
**Layout** · {info['layout']}  
**Shape** · `{info['shape']}`  
**Dims** · `{info['dims']}`  
**Dtype** · `{info['dtype']}`  
**Size** · {info['size']}
""".strip()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def on_upload(file_obj):
    if file_obj is None:
        return "_Upload a k-space file to see details._"
    info = _inspect_kspace(file_obj.name)
    return _info_markdown(info)


def run_prediction(file_obj,
                   z1, z2, t1, t2,
                   quiver_grid, quiver_scale,
                   progress=gr.Progress()):
    if file_obj is None:
        raise gr.Error("Please upload a k-space file first.")

    try:
        z1, z2, t1, t2 = int(z1), int(z2), int(t1), int(t2)
        quiver_grid = int(quiver_grid)
        quiver_scale = float(quiver_scale)

        progress(0.10, desc="Loading k-space")
        ksp = pipeline._load_kspace_file(file_obj.name)

        progress(0.35, desc=f"Extracting (z1={z1}, t1={t1}) and (z2={z2}, t2={t2})")
        ksp1, ksp2 = pipeline._extract_pair(ksp, z1, z2, t1, t2)

        progress(0.50, desc="Rendering input magnitudes")
        mag1 = pipeline._kspace_to_magnitude_image(ksp1)
        mag2 = pipeline._kspace_to_magnitude_image(ksp2)
        fixed_img = pipeline._magnitude_to_pil(mag1)
        moving_img = pipeline._magnitude_to_pil(mag2)

        progress(0.70, desc="Running LAPANet")
        k_fix = pipeline._prepare_single(ksp1).to(pipeline.device)
        k_mov = pipeline._prepare_single(ksp2).to(pipeline.device)
        with torch.inference_mode():
            out = pipeline.model(k_fix, k_mov)
        flow_tensor = pipeline._unpack_output(out)
        u, v = pipeline._flow_tensor_to_uv(flow_tensor)

        progress(0.85, desc="Encoding flow (color)")
        flow_color_img = pipeline._flow_to_pil(u, v)

        progress(0.95, desc="Rendering quiver")
        flow_quiver_img = pipeline._flow_to_quiver_pil(
            u, v,
            grid_size=quiver_grid,
            scale=quiver_scale,
            title=f"z1={z1}, t1={t1}  →  z2={z2}, t2={t2}",
        )

        progress(1.0, desc="Done")
        return fixed_img, moving_img, flow_color_img, flow_quiver_img

    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Motion estimation failed: {e}")


def clear_all():
    return None, None, None, None, "_Upload a k-space file to see details._"


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CSS = """
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
}
.app-header {
    padding: 4px 0 22px 0;
    border-bottom: 1px solid var(--border-color-primary);
    margin-bottom: 26px;
}
.app-header h1 {
    font-size: 1.6rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin: 0 0 6px 0;
    color: var(--body-text-color);
}
.app-header p {
    font-size: 0.95rem;
    line-height: 1.55;
    color: var(--body-text-color-subdued);
    margin: 0;
    max-width: 780px;
}
.meta-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 26px;
    padding: 12px 0 0 0;
    font-size: 0.82rem;
    color: var(--body-text-color-subdued);
}
.meta-strip .meta-item strong {
    display: block;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--body-text-color-subdued);
    opacity: 0.75;
    margin-bottom: 3px;
}
.meta-strip .meta-item span {
    color: var(--body-text-color);
    font-weight: 500;
}
.info-panel {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 0.85rem;
    line-height: 1.7;
}
.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--body-text-color-subdued);
    margin-bottom: 8px;
}
.gradio-container .gr-form,
.gradio-container .gr-panel {
    border: none !important;
}
button.primary {
    font-weight: 500 !important;
    letter-spacing: 0.01em;
}
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.slate,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ),
    title=APP_TITLE,
    css=CSS,
) as demo:

    # ---- Header -----------------------------------------------------------
    gr.HTML(
        f"""
        <div class="app-header">
            <h1>{APP_TITLE}</h1>
            <p>{APP_DESCRIPTION}</p>
            <div class="meta-strip">
                <div class="meta-item"><strong>Model</strong><span>LAPANet2D</span></div>
                <div class="meta-item"><strong>Device</strong><span>{DEVICE}</span></div>
                <div class="meta-item"><strong>Spatial size</strong><span>{OUT_SHAPE[0]} × {OUT_SHAPE[1]}</span></div>
                <div class="meta-item"><strong>Source</strong><span>{MODEL_SOURCE}</span></div>
            </div>
        </div>
        """
    )

    # ---- Tabs -------------------------------------------------------------
    with gr.Tabs():

        # ============ Motion Estimation ==================================
        with gr.Tab("Image Registration"):
            with gr.Row(equal_height=False):

                # -------- Left: input + controls ---------------------------
                with gr.Column(scale=4, min_width=320):
                    gr.HTML('<div class="card-title">Input</div>')
                    file_input = gr.File(
                        label="Raw k-space file (F, S, C, H, W)",
                        file_types=[".npy", ".npz", ".h5", ".hdf5", ".mat"],
                        type="filepath",
                        height=140,
                    )

                    gr.HTML(
                        '<div class="card-title" style="margin-top:14px;">'
                        'Pair selection</div>'
                    )
                    gr.Markdown(
                        "Pick the fixed input `(z1, t1)` and the moving input "
                        "`(z2, t2)`. Each selection yields a `(C, H, W)` "
                        "k-space fed to LAPANet.",
                    )

                    with gr.Row():
                        z1_in = gr.Number(label="z1 — slice (fixed)",
                                          value=0, precision=0)
                        t1_in = gr.Number(label="t1 — frame (fixed)",
                                          value=0, precision=0)

                    with gr.Row():
                        z2_in = gr.Number(label="z2 — slice (moving)",
                                          value=0, precision=0)
                        t2_in = gr.Number(label="t2 — frame (moving)",
                                          value=1, precision=0)

                    with gr.Accordion("Quiver settings", open=False):
                        quiver_grid = gr.Slider(
                            label="Arrow spacing (grid size, px)",
                            minimum=8, maximum=64, step=2, value=20,
                        )
                        quiver_scale = gr.Slider(
                            label="Arrow scale (smaller = longer arrows)",
                            minimum=0.1, maximum=5.0, step=0.1, value=1.0,
                        )

                    with gr.Row():
                        submit_btn = gr.Button("Run Motion Estimation",
                                               variant="primary", scale=3)
                        clear_btn = gr.Button("Clear", scale=1)

                    gr.HTML(
                        '<div class="card-title" style="margin-top:18px;">'
                        'File details</div>'
                    )
                    info_md = gr.Markdown(
                        "_Upload a k-space file to see details._",
                        elem_classes="info-panel",
                    )

                    with gr.Accordion("Supported formats", open=False):
                        gr.Markdown(
                            "**Extensions** — `.npy`, `.npz`, `.h5` / `.hdf5`, `.mat`  \n"
                            "**Accepted shape** — 5D `(F, S, C, H, W)`  \n\n"
                            "The `(z1, t1)` and `(z2, t2)` controls pick the two "
                            "`(Coils, H, W)` inputs. Indexing is 0-based."
                        )

                # -------- Right: outputs ----------------------------------
                with gr.Column(scale=7, min_width=520):
                    gr.HTML('<div class="card-title">Output</div>')
                    with gr.Tabs():

                        with gr.Tab("Inputs"):
                            with gr.Row():
                                fixed_img = gr.Image(
                                    label="Fixed  (z1, t1)",
                                    type="pil",
                                    height=420,
                                    show_label=True,
                                    show_download_button=True,
                                )
                                moving_img = gr.Image(
                                    label="Moving  (z2, t2)",
                                    type="pil",
                                    height=420,
                                    show_label=True,
                                    show_download_button=True,
                                )

                        with gr.Tab("Flow · color"):
                            flow_color_img = gr.Image(
                                label=None,
                                type="pil",
                                height=520,
                                show_label=False,
                                show_download_button=True,
                            )

                        with gr.Tab("Flow · quiver"):
                            flow_quiver_img = gr.Image(
                                label=None,
                                type="pil",
                                height=520,
                                show_label=False,
                                show_download_button=True,
                            )

            # -------- Wiring ----------------------------------------------
            file_input.change(
                fn=on_upload,
                inputs=[file_input],
                outputs=[info_md],
                show_progress="hidden",
            )

            submit_btn.click(
                fn=run_prediction,
                inputs=[file_input, z1_in, z2_in, t1_in, t2_in,
                        quiver_grid, quiver_scale],
                outputs=[fixed_img, moving_img, flow_color_img, flow_quiver_img],
                show_progress="full",
            )

            clear_btn.click(
                fn=clear_all,
                inputs=None,
                outputs=[file_input, fixed_img, moving_img,
                         flow_color_img, flow_quiver_img, info_md],
            )

        # ============ About ===============================================
        with gr.Tab("About"):
            gr.Markdown(
                f"""
### Method

LAPANet — *Local All-Pass Attention Network* — estimates cardiac motion from sub-sampled k-space.

### Processing pipeline

1. **Load** complex k-space from `.npy`, `.npz`, `.h5` / `.hdf5`, or `.mat`
   (MATLAB v7.3 compound-complex dtypes supported).
2. **Select** a fixed `(z1, t1)` and a moving `(z2, t2)` frame; each is a
   `(Coils, H, W)` k-space.
3. **Inputs view** — `ifft2c` → sum over coils → `abs`, normalized to `[0, 1]`.
4. **Model preprocessing** — `ifft2c → pad/crop to {OUT_SHAPE[0]} × {OUT_SHAPE[1]} → magnitude normalization → fft2c`.
5. **Forward pass** — LAPANet2D `forward(k_fix, k_mov)` on **{DEVICE}**.
6. **Flow visualisation**
   - **Color** — HSV encoding via `flow_vis.flow_to_color`.
   - **Quiver** — arrow plot on a black background.

### Model configuration

| Field | Value |
|---|---|
| Architecture | LAPANet2D |
| Attention heads | `[1] × 9` |
| Encoder / decoder filters | `[16, 32, 64, 192, 384, 192, 64, 32, 16]` |
| Number of coils | 10 |
| Device | {DEVICE} |
| Weights source | {MODEL_SOURCE} |

---

{ARTICLE}
"""
            )

    # ---- Footer -----------------------------------------------------------
    gr.HTML(
        """
        <div style="
            margin-top: 32px;
            padding-top: 16px;
            border-top: 1px solid var(--border-color-primary);
            font-size: 0.78rem;
            color: var(--body-text-color-subdued);
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
        ">
            <span>Local All-Pass Attention Network · University Hospital of Tübingen</span>
            <span>Research use only — not for clinical diagnosis</span>
        </div>
        """
    )


if __name__ == "__main__":
    demo.queue(max_size=16).launch(
        server_name="0.0.0.0",
        server_port=8000,
        share=False,
        show_error=True,
    )