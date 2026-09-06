"""Gradio Web Application for Stereo Disparity and Depth Estimation.

CO543 / CO5430 Computer Vision Project
Group G03 | Project ID: P12 | University of Peradeniya
"""

import os
import sys
import time
import numpy as np
import cv2
import gradio as gr

# Ensure local src directory is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.dataset import read_pfm
from src.stereo_bm import compute_stereo_bm
from src.stereo_sgbm import compute_stereo_sgbm
from src.evaluation import calculate_metrics
from src.visualization import colorize_disparity, create_error_heatmap


# Global RAFT model cache to avoid re-loading on each inference
RAFT_PRETRAINED_INSTANCE = None
RAFT_FINETUNED_INSTANCE = None


def get_raft_model(mode: str = "pretrained"):
    """Lazy loader for RAFT-Stereo models with fallback."""
    global RAFT_PRETRAINED_INSTANCE, RAFT_FINETUNED_INSTANCE

    try:
        from src.raft_wrapper import RAFTStereoInference
    except Exception as e:
        return None, f"RAFT-Stereo dependencies not available: {e}"

    if mode == "pretrained":
        if RAFT_PRETRAINED_INSTANCE is None:
            ckpt = os.path.join(current_dir, "models", "raftstereo-sceneflow.pth")
            if not os.path.exists(ckpt):
                alt_ckpt = "/content/RAFT-Stereo/models/raftstereo-sceneflow.pth"
                if os.path.exists(alt_ckpt):
                    ckpt = alt_ckpt
            try:
                RAFT_PRETRAINED_INSTANCE = RAFTStereoInference(checkpoint_path=ckpt)
            except Exception as e:
                return None, f"Could not load Pretrained RAFT: {e}"
        return RAFT_PRETRAINED_INSTANCE, "OK"

    elif mode == "finetuned":
        if RAFT_FINETUNED_INSTANCE is None:
            ckpt = os.path.join(current_dir, "models", "my_finetuned_raft_v2.pth")
            if not os.path.exists(ckpt):
                alt_ckpt = "/content/my_finetuned_raft_v2.pth"
                if os.path.exists(alt_ckpt):
                    ckpt = alt_ckpt
            try:
                RAFT_FINETUNED_INSTANCE = RAFTStereoInference(checkpoint_path=ckpt)
            except Exception as e:
                return None, f"Could not load Fine-tuned RAFT: {e}"
        return RAFT_FINETUNED_INSTANCE, "OK"

    return None, "Invalid mode"


def process_single_stereo(
    imgL: np.ndarray,
    imgR: np.ndarray,
    model_choice: str,
    block_size: int,
    num_disparities: int,
    uniqueness_ratio: int,
    cmap_name: str,
):
    """Callback for Tab 1: Single model disparity estimation with ImageSlider output."""
    if imgL is None or imgR is None:
        raise gr.Error("Please provide both Left and Right stereo images.")

    # Convert to RGB if needed
    if len(imgL.shape) == 2:
        imgL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2RGB)
    if len(imgR.shape) == 2:
        imgR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2RGB)

    hL, wL = imgL.shape[:2]
    hR, wR = imgR.shape[:2]
    if (hL, wL) != (hR, wR):
        imgR = cv2.resize(imgR, (wL, hL), interpolation=cv2.INTER_LINEAR)

    disp = None
    elapsed_ms = 0.0

    if "SGBM" in model_choice:
        disp, elapsed_ms = compute_stereo_sgbm(
            imgL, imgR,
            num_disparities=int(num_disparities),
            block_size=int(block_size),
            uniqueness_ratio=int(uniqueness_ratio)
        )
    elif "StereoBM" in model_choice:
        bm_block = max(5, int(block_size))
        if bm_block % 2 == 0:
            bm_block += 1
        disp, elapsed_ms = compute_stereo_bm(
            imgL, imgR,
            num_disparities=int(num_disparities),
            block_size=bm_block,
            uniqueness_ratio=int(uniqueness_ratio)
        )
    elif "Pre-trained" in model_choice or "Pretrained" in model_choice:
        raft, status = get_raft_model("pretrained")
        if raft is None:
            raise gr.Error(f"Pretrained RAFT error: {status}. Please select StereoSGBM or run checkpoint download.")
        disp, elapsed_ms = raft.compute_disparity(imgL, imgR, iters=32, max_dimension=768)
    elif "Fine-tuned" in model_choice:
        raft, status = get_raft_model("finetuned")
        if raft is None:
            raise gr.Error(f"Fine-tuned RAFT error: {status}. Please select StereoSGBM or run checkpoint download.")
        disp, elapsed_ms = raft.compute_disparity(imgL, imgR, iters=32, max_dimension=768)
    else:
        disp, elapsed_ms = compute_stereo_sgbm(imgL, imgR)

    # Colorize disparity
    color_disp = colorize_disparity(disp, cmap_name=cmap_name)

    valid_mask = disp > 0
    valid_d = disp[valid_mask] if np.any(valid_mask) else np.array([0.0])
    d_min, d_max = float(np.min(valid_d)), float(np.max(valid_d))

    stats_text = f"""
    ### Inference Statistics:
    - **Selected Algorithm:** {model_choice}
    - **Inference Time:** {elapsed_ms:.2f} ms ({1000.0 / max(elapsed_ms, 0.001):.1f} FPS)
    - **Resolution:** {wL} × {hL} px
    - **Disparity Dynamic Range:** {d_min:.1f} px to {d_max:.1f} px
    - **Valid Pixel Ratio:** {(np.sum(valid_mask) / disp.size) * 100.0:.1f}%
    """

    # ImageSlider takes (Image 1, Image 2)
    return (imgL, color_disp), stats_text, disp


def compare_all_models(
    imgL: np.ndarray,
    imgR: np.ndarray,
    cmap_name: str,
):
    """Callback for Tab 2: Computes BM, SGBM, and RAFT side-by-side."""
    if imgL is None or imgR is None:
        raise gr.Error("Please upload both Left and Right images.")

    if len(imgL.shape) == 2:
        imgL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2RGB)
    if len(imgR.shape) == 2:
        imgR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2RGB)

    hL, wL = imgL.shape[:2]
    imgR = cv2.resize(imgR, (wL, hL), interpolation=cv2.INTER_LINEAR)

    # 1. StereoBM
    bm_disp, bm_ms = compute_stereo_bm(imgL, imgR, num_disparities=160, block_size=25)
    bm_vis = colorize_disparity(bm_disp, cmap_name=cmap_name)
    bm_label = f"StereoBM (BlockSize=25, Time: {bm_ms:.1f} ms)"

    # 2. StereoSGBM
    sgbm_disp, sgbm_ms = compute_stereo_sgbm(imgL, imgR, num_disparities=160, block_size=3, uniqueness_ratio=10)
    sgbm_vis = colorize_disparity(sgbm_disp, cmap_name=cmap_name)
    sgbm_label = f"StereoSGBM (BlockSize=3, Time: {sgbm_ms:.1f} ms)"

    # 3. RAFT Pretrained (if available, otherwise fallback)
    raft_pre, pre_status = get_raft_model("pretrained")
    if raft_pre:
        pre_disp, pre_ms = raft_pre.compute_disparity(imgL, imgR, iters=24, max_dimension=768)
        pre_vis = colorize_disparity(pre_disp, cmap_name=cmap_name)
        pre_label = f"RAFT-Stereo Pre-trained (Time: {pre_ms:.1f} ms)"
    else:
        pre_vis = np.zeros_like(imgL)
        pre_label = f"Pretrained RAFT not loaded ({pre_status})"

    # 4. RAFT Fine-tuned (if available)
    raft_fine, fine_status = get_raft_model("finetuned")
    if raft_fine:
        fine_disp, fine_ms = raft_fine.compute_disparity(imgL, imgR, iters=24, max_dimension=768)
        fine_vis = colorize_disparity(fine_disp, cmap_name=cmap_name)
        fine_label = f"RAFT-Stereo Fine-tuned (Time: {fine_ms:.1f} ms)"
    else:
        fine_vis = np.zeros_like(imgL)
        fine_label = f"Fine-tuned RAFT not loaded ({fine_status})"

    return (
        gr.update(value=bm_vis, label=bm_label),
        gr.update(value=sgbm_vis, label=sgbm_label),
        gr.update(value=pre_vis, label=pre_label),
        gr.update(value=fine_vis, label=fine_label),
    )


def evaluate_with_ground_truth(
    imgL: np.ndarray,
    imgR: np.ndarray,
    gt_file,
    model_choice: str,
    bad_threshold: float,
):
    """Callback for Tab 3: Quantitative Middlebury Benchmark against Ground Truth."""
    if imgL is None or imgR is None:
        raise gr.Error("Please upload Left and Right stereo images.")
    if gt_file is None:
        raise gr.Error("Please upload a ground-truth disparity file (.pfm or .png).")

    # Load Ground Truth
    gt_path = gt_file.name if hasattr(gt_file, 'name') else str(gt_file)
    if gt_path.lower().endswith('.pfm'):
        disp_gt = read_pfm(gt_path)
    else:
        gt_raw = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt_raw is None:
            raise gr.Error(f"Could not read ground truth file: {gt_path}")
        if len(gt_raw.shape) == 3:
            disp_gt = cv2.cvtColor(gt_raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            disp_gt = gt_raw.astype(np.float32)

    # Compute prediction
    if "SGBM" in model_choice:
        disp_pred, _ = compute_stereo_sgbm(imgL, imgR, num_disparities=160, block_size=3)
    elif "StereoBM" in model_choice:
        disp_pred, _ = compute_stereo_bm(imgL, imgR, num_disparities=160, block_size=25)
    elif "Pre-trained" in model_choice or "Pretrained" in model_choice:
        raft, _ = get_raft_model("pretrained")
        if raft is None:
            raise gr.Error("RAFT Pretrained model checkpoint is not loaded.")
        disp_pred, _ = raft.compute_disparity(imgL, imgR, iters=32)
    elif "Fine-tuned" in model_choice:
        raft, _ = get_raft_model("finetuned")
        if raft is None:
            raise gr.Error("RAFT Fine-tuned model checkpoint is not loaded.")
        disp_pred, _ = raft.compute_disparity(imgL, imgR, iters=32)
    else:
        disp_pred, _ = compute_stereo_sgbm(imgL, imgR)

    # Calculate metrics
    metrics = calculate_metrics(disp_gt, disp_pred, bad_threshold=float(bad_threshold))

    # Colorize ground truth, prediction, and error heatmap
    gt_vis = colorize_disparity(disp_gt, cmap_name="plasma")
    pred_vis = colorize_disparity(disp_pred, cmap_name="plasma")
    error_vis = create_error_heatmap(disp_gt, disp_pred, bad_threshold=float(bad_threshold))

    metrics_md = f"""
    ### 📊 Benchmark Quantitative Results:
    | Metric | Evaluated Value | Middlebury Benchmark Description |
    | :--- | :--- | :--- |
    | **RMSE** | **`{metrics['rmse']:.4f}` px** | Root Mean Squared Error over valid pixels |
    | **AbsRel** | **`{metrics['absrel']:.4f}`** | Absolute Relative Error $|d_{{gt}} - d_{{pred}}| / d_{{gt}}$ |
    | **Bad-{bad_threshold:.0f}px Rate (D1)** | **`{metrics['bad_pixel_rate']:.2f}%`** | Percentage of pixels with error $> {bad_threshold:.1f}$ px |
    | **EPE (End-Point Error)** | **`{metrics['epe']:.4f}` px** | Mean Absolute Error across all valid coordinates |
    | **Valid Pixels Count** | **`{metrics['valid_pixels']:,}`** | Evaluated non-occluded, non-infinite points |
    """

    return gt_vis, pred_vis, error_vis, metrics_md


# -------------------------------------------------------------
# Gradio Blocks UI Definition
# -------------------------------------------------------------
custom_css = """
.gradio-container { max-width: 1200px !important; margin: auto; }
#title-banner { text-align: center; margin-bottom: 20px; }
.metric-box { border-radius: 8px; padding: 12px; background: #f8f9fa; }
"""

with gr.Blocks(title="Stereo Disparity & Depth Estimation | CO5430", css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_id="title-banner"):
        gr.Markdown(
            """
            # 👁️ Stereo Disparity and Relative Depth Estimation
            ### **CO543 / CO5430 Computer Vision Course Project**
            **Group G03 | Project ID: P12 | University of Peradeniya**
            *Team: E/23/127 H.M.K.I. Herath &bull; E/23/188 K.M.M.Y. Kumarasinge &bull; E/23/343 S.B.N.S. Samarawickrama &bull; E/23/347 S.D.M.P. Sandanayake*
            """
        )

    with gr.Tabs():
        # TAB 1: Single Prediction & Interactive Slider
        with gr.TabItem("🔍 Interactive Disparity Estimation"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 1. Stereo Inputs")
                        imgL_in = gr.Image(label="Left Camera Image (Rectified)", type="numpy")
                        imgR_in = gr.Image(label="Right Camera Image (Rectified)", type="numpy")

                    with gr.Group():
                        gr.Markdown("### 2. Model & Hyperparameters")
                        model_dropdown = gr.Dropdown(
                            label="Algorithm Selection",
                            choices=[
                                "StereoSGBM (Classical Semi-Global Matching)",
                                "StereoBM (Classical Block Matching)",
                                "RAFT-Stereo (Pre-trained SceneFlow)",
                                "RAFT-Stereo (Fine-Tuned Middlebury)"
                            ],
                            value="StereoSGBM (Classical Semi-Global Matching)"
                        )

                        with gr.Accordion("⚙️ Classical Tuning Parameters", open=True):
                            block_slider = gr.Slider(
                                label="Block Size (SAD Window)",
                                minimum=1, maximum=25, step=2, value=3,
                                info="Smaller retains thin edges; larger reduces speckle noise in flat areas."
                            )
                            disp_slider = gr.Slider(
                                label="Number of Disparities (Range)",
                                minimum=16, maximum=256, step=16, value=160,
                                info="Maximum pixel search range along epipolar lines."
                            )
                            uniq_slider = gr.Slider(
                                label="Uniqueness Ratio (%)",
                                minimum=0, maximum=25, step=1, value=10,
                                info="Filters out ambiguous matching costs."
                            )
                            cmap_dropdown = gr.Dropdown(
                                label="Color Palette",
                                choices=["plasma", "turbo", "magma", "viridis"],
                                value="plasma"
                            )

                        compute_btn = gr.Button("🚀 Compute Disparity Map", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### 3. Estimated Output (Split Image Slider)")
                    slider_out = gr.ImageSlider(
                        label="Slide: Left View vs. Estimated Disparity Map",
                        type="numpy"
                    )
                    stats_markdown = gr.Markdown(
                        "Click **Compute Disparity Map** to run inference.",
                        elem_classes=["metric-box"]
                    )

            # Sample Examples Gallery
            example_dir = os.path.join(current_dir, "examples")
            sample_left = os.path.join(example_dir, "sample_left.png")
            sample_right = os.path.join(example_dir, "sample_right.png")
            if os.path.exists(sample_left) and os.path.exists(sample_right):
                gr.Examples(
                    examples=[
                        [sample_left, sample_right, "StereoSGBM (Classical Semi-Global Matching)", 3, 160, 10, "plasma"],
                        [sample_left, sample_right, "StereoBM (Classical Block Matching)", 25, 160, 15, "turbo"],
                    ],
                    inputs=[imgL_in, imgR_in, model_dropdown, block_slider, disp_slider, uniq_slider, cmap_dropdown],
                    label="Click to Load Sample Stereo Pairs"
                )

        # TAB 2: Side-by-Side Model Comparison
        with gr.TabItem("⚖️ 4-Way Model Comparison"):
            gr.Markdown(
                """
                ### Side-by-Side Architectural Comparison
                Runs **StereoBM**, **StereoSGBM**, **Pretrained RAFT**, and **Fine-tuned RAFT** simultaneously on the same rectified stereo pair.
                """
            )
            with gr.Row():
                comp_imgL = gr.Image(label="Input Left Image", type="numpy")
                comp_imgR = gr.Image(label="Input Right Image", type="numpy")

            with gr.Row():
                comp_cmap = gr.Dropdown(
                    label="Colormap",
                    choices=["plasma", "turbo", "magma", "viridis"],
                    value="plasma",
                    scale=1
                )
                run_comp_btn = gr.Button("⚡ Compare All Algorithms", variant="primary", scale=2)

            with gr.Row():
                out_bm = gr.Image(label="StereoBM Disparity", type="numpy")
                out_sgbm = gr.Image(label="StereoSGBM Disparity", type="numpy")

            with gr.Row():
                out_pre = gr.Image(label="Pretrained RAFT-Stereo", type="numpy")
                out_fine = gr.Image(label="Fine-tuned RAFT-Stereo", type="numpy")

        # TAB 3: Quantitative Ground Truth Benchmark
        with gr.TabItem("📊 Ground Truth Benchmark & Error Heatmap"):
            gr.Markdown(
                """
                ### Middlebury Benchmark Quantitative Validation
                Upload a ground-truth disparity map (`.pfm` or `.png`) to compute **RMSE**, **Absolute Relative Error**, and **Bad-3-Pixel Rate (D1 Metric)**.
                """
            )
            with gr.Row():
                eval_imgL = gr.Image(label="Left Image", type="numpy")
                eval_imgR = gr.Image(label="Right Image", type="numpy")
                eval_gt = gr.File(label="Ground Truth Disparity (.pfm / .png)")

            with gr.Row():
                eval_model = gr.Dropdown(
                    label="Algorithm to Benchmark",
                    choices=[
                        "StereoSGBM (Classical Semi-Global Matching)",
                        "StereoBM (Classical Block Matching)",
                        "RAFT-Stereo (Pre-trained SceneFlow)",
                        "RAFT-Stereo (Fine-Tuned Middlebury)"
                    ],
                    value="StereoSGBM (Classical Semi-Global Matching)"
                )
                thresh_slider = gr.Slider(
                    label="Bad-Pixel Threshold (pixels)",
                    minimum=0.5, maximum=5.0, step=0.5, value=3.0,
                    info="Standard threshold for Middlebury benchmark is 3.0 px."
                )
                run_eval_btn = gr.Button("📈 Run Benchmark & Generate Error Map", variant="primary")

            with gr.Row():
                gt_view = gr.Image(label="Ground Truth Disparity Map", type="numpy")
                pred_view = gr.Image(label="Model Disparity Prediction", type="numpy")
                err_view = gr.Image(label="Error Heatmap (|GT - Pred| > 3px)", type="numpy")

            eval_results_md = gr.Markdown("Upload images and Ground Truth, then click **Run Benchmark**.")

        # TAB 4: Project Overview & Academic Documentation
        with gr.TabItem("📖 Documentation & Methodology"):
            gr.Markdown(
                """
                ### 📚 Project Methodology & Milestones

                #### 1. Problem Formulation
                Given two rectified stereo cameras separated by baseline $B$ and focal length $f$, the horizontal disparity $d = x_L - x_R$ relates to scene depth $Z$ via:
                $$Z = \\frac{f \\cdot B}{d}$$

                #### 2. Classical Approaches (OpenCV)
                * **StereoBM (Block Matching):** Sliding window matching using Sum of Absolute Differences (SAD). Fast but produces noisy boundaries in textureless regions.
                * **StereoSGBM (Semi-Global Matching):** Minimizes a 2D energy functional across 1D path directions with smoothness penalties:
                  $$P_1 = 8 \\times C \\times \\text{blockSize}^2, \\quad P_2 = 32 \\times C \\times \\text{blockSize}^2$$

                #### 3. Deep Learning Approach (RAFT-Stereo)
                * **Feature Extractors:** 2D CNNs (`fnet`, `cnet`) producing multi-scale representations.
                * **3D Correlation Pyramid:** Multi-scale 3D all-pairs correlation volumes.
                * **Recurrent GRU Update:** Iterative residual updates ($12$ to $32$ steps).
                * **Milestone 3 Domain Adaptation Insights:**
                  1. **Memory:** $384 \\times 512$ random spatial crops prevent Colab GPU VRAM out-of-memory.
                  2. **Regularization:** Freezing `fnet` and `cnet` prevents catastrophic forgetting on small datasets (24 images).
                  3. **Scale Alignment:** Sequence-weighted Smooth L1 loss stabilizes predictions against Middlebury ground truth.
                """
            )

    # ---------------------------------------------------------
    # Wire Callback Events
    # ---------------------------------------------------------
    hidden_raw_disp = gr.State()

    compute_btn.click(
        fn=process_single_stereo,
        inputs=[imgL_in, imgR_in, model_dropdown, block_slider, disp_slider, uniq_slider, cmap_dropdown],
        outputs=[slider_out, stats_markdown, hidden_raw_disp]
    )

    run_comp_btn.click(
        fn=compare_all_models,
        inputs=[comp_imgL, comp_imgR, comp_cmap],
        outputs=[out_bm, out_sgbm, out_pre, out_fine]
    )

    run_eval_btn.click(
        fn=evaluate_with_ground_truth,
        inputs=[eval_imgL, eval_imgR, eval_gt, eval_model, thresh_slider],
        outputs=[gt_view, pred_view, err_view, eval_results_md]
    )


if __name__ == "__main__":
    # In Colab: launch with share=True to obtain public *.gradio.live URL
    # On Hugging Face Spaces: launches automatically on port 7860
    is_colab = "google.colab" in sys.modules or os.environ.get("COLAB_GPU") is not None
    demo.launch(share=is_colab)
