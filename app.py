"""Gradio Web App - Stereo Disparity and Depth Estimation.

CO543 / CO5430 Computer Vision Project
Group G03 | Project ID: P12 | University of Peradeniya
"""
import os
import sys

# Optional compatibility shim for environments with legacy gradio
try:
    import huggingface_hub as _hfhub
    if not hasattr(_hfhub, "HfFolder"):
        class _HfFolderStub:
            @staticmethod
            def get_token(): return os.environ.get("HF_TOKEN") or None
            @staticmethod
            def save_token(token): pass
            @staticmethod
            def delete_token(): pass
        _hfhub.HfFolder = _HfFolderStub
        sys.modules["huggingface_hub"].HfFolder = _HfFolderStub
except Exception:
    pass

import numpy as np
import cv2
import gradio as gr

# Hugging Face ZeroGPU compatibility: import spaces or fallback to no-op
try:
    import spaces
except ImportError:
    class _MockSpaces:
        @staticmethod
        def GPU(fn=None, duration=None):
            def decorator(f):
                return f
            return decorator if fn is None else fn
    spaces = _MockSpaces()

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.dataset import read_pfm
from src.stereo_bm import compute_stereo_bm
from src.stereo_sgbm import compute_stereo_sgbm
from src.evaluation import calculate_metrics
from src.visualization import colorize_disparity, create_error_heatmap

_RAFT_PRE  = None
_RAFT_FINE = None

def _get_raft(mode="pretrained"):
    global _RAFT_PRE, _RAFT_FINE
    try:
        from src.raft_wrapper import RAFTStereoInference
    except Exception as e:
        import traceback
        return None, f"ImportError:\n{traceback.format_exc()}"
    if mode == "pretrained":
        if _RAFT_PRE is None:
            ckpt = ""
            for p in [os.path.join(current_dir, "models", "raftstereo-sceneflow.pth"),
                      os.path.join(current_dir, "core", "models", "raftstereo-sceneflow.pth"),
                      "/content/RAFT-Stereo/models/raftstereo-sceneflow.pth"]:
                if os.path.exists(p):
                    ckpt = p
                    break
            try:
                _RAFT_PRE = RAFTStereoInference(checkpoint_path=ckpt, mode="pretrained")
            except Exception as e:
                import traceback
                return None, f"InitError:\n{traceback.format_exc()}"
        return _RAFT_PRE, "OK"
    elif mode == "finetuned":
        if _RAFT_FINE is None:
            ckpt = ""
            for p in [os.path.join(current_dir, "models", "my_finetuned_raft_v2.pth"),
                      os.path.join(current_dir, "models", "raftstereo-middlebury.pth"),
                      "/content/my_finetuned_raft_v2.pth"]:
                if os.path.exists(p):
                    ckpt = p
                    break
            try:
                _RAFT_FINE = RAFTStereoInference(checkpoint_path=ckpt, mode="finetuned")
            except Exception as e:
                import traceback
                return None, f"InitError:\n{traceback.format_exc()}"
        return _RAFT_FINE, "OK"
    return None, "Invalid mode"

def _rgb(img):
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else img

def _align(L, R):
    h, w = L.shape[:2]
    if R.shape[:2] != (h, w):
        R = cv2.resize(R, (w, h), interpolation=cv2.INTER_LINEAR)
    return L, R

def _ensure_samples():
    import zlib, struct, math
    ex = os.path.join(current_dir, "examples")
    os.makedirs(ex, exist_ok=True)
    lf = os.path.join(ex, "sample_left.png")
    rf = os.path.join(ex, "sample_right.png")
    if os.path.exists(lf) and os.path.exists(rf):
        return lf, rf
    W, H = 480, 360
    def png(w, h, ba):
        def chunk(t, d):
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t+d)&0xFFFFFFFF)
        raw = bytearray()
        for y in range(h):
            raw.append(0)
            raw.extend(ba[y*w*3:(y+1)*w*3])
        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw),9)) + chunk(b"IEND", b"")
    def t(x, y, s=16):
        return max(0, min(255, int(128+60*math.sin(x/s)*math.cos(y/s)+40*math.sin((x+y)/(s*.7)))))
    li, ri = bytearray(W*H*3), bytearray(W*H*3)
    for y in range(H):
        for x in range(W):
            i = (y*W+x)*3
            d = 8
            r, g, b = t(x,y,12), t(x+50,y,14), t(x,y+50,16)
            if 130 <= x <= 230 and 130 <= y <= 230:
                d, r, g, b = 32, 200+(x%20)*2, 100+(y%20)*4, 80
            elif (x-340)**2+(y-200)**2 <= 60**2:
                d, r, g, b = 56, 60, 180+(x%15)*3, 210+(y%15)*2
            li[i:i+3] = bytes([r,g,b])
            xr = x - d
            if 0 <= xr < W:
                ri[(y*W+xr)*3:(y*W+xr)*3+3] = bytes([r,g,b])
    for y in range(H):
        for x in range(W):
            i = (y*W+x)*3
            if ri[i] == ri[i+1] == ri[i+2] == 0:
                ri[i:i+3] = bytes([t(x,y,12), t(x+50,y,14), t(x,y+50,16)])
    with open(lf, "wb") as f:
        f.write(png(W, H, li))
    with open(rf, "wb") as f:
        f.write(png(W, H, ri))
    return lf, rf

METHODS = [
    "StereoSGBM (Classical Semi-Global Matching)",
    "StereoBM (Classical Block Matching)",
    "RAFT-Stereo (Pre-trained SceneFlow)",
    "RAFT-Stereo (Fine-Tuned Middlebury)",
]

@spaces.GPU
def cb_single(imgL, imgR, method, block_size, num_disp, uniq, cmap):
    if imgL is None or imgR is None:
        raise gr.Error("Please upload both Left and Right stereo images.")
    imgL, imgR = _rgb(imgL), _rgb(imgR)
    imgL, imgR = _align(imgL, imgR)
    h, w = imgL.shape[:2]
    if "SGBM" in method:
        disp, ms = compute_stereo_sgbm(imgL, imgR, num_disparities=int(num_disp),
                                        block_size=int(block_size), uniqueness_ratio=int(uniq))
    elif "StereoBM" in method:
        bs = max(5, int(block_size))
        if bs % 2 == 0:
            bs += 1
        disp, ms = compute_stereo_bm(imgL, imgR, num_disparities=int(num_disp),
                                      block_size=bs, uniqueness_ratio=int(uniq))
    elif "Pre-trained" in method:
        raft, status = _get_raft("pretrained")
        if raft is None:
            raise gr.Error("Pretrained RAFT unavailable: " + status)
        try:
            disp, ms = raft.compute_disparity(imgL, imgR, iters=32, max_dimension=768)
        except Exception as e:
            import traceback
            raise gr.Error(f"Pretrained RAFT computation error: {e}")
    else:
        raft, status = _get_raft("finetuned")
        if raft is None:
            raise gr.Error("Fine-tuned RAFT unavailable: " + status)
        try:
            disp, ms = raft.compute_disparity(imgL, imgR, iters=32, max_dimension=768)
        except Exception as e:
            import traceback
            raise gr.Error(f"Fine-tuned RAFT computation error: {e}")
    color_disp = colorize_disparity(disp, cmap_name=cmap)
    valid = disp[disp > 0]
    d_min = float(np.min(valid)) if valid.size else 0.0
    d_max = float(np.max(valid)) if valid.size else 0.0
    pct   = (valid.size / disp.size) * 100.0
    stats = ("**Algorithm:** " + method + "  \n" +
             "**Inference Time:** " + str(round(ms,1)) + " ms  \n" +
             "**Resolution:** " + str(w) + " x " + str(h) + " px  \n" +
             "**Disparity Range:** " + str(round(d_min,1)) + " - " + str(round(d_max,1)) + " px  \n" +
             "**Valid Pixel Ratio:** " + str(round(pct,1)) + "%")
    return imgL, color_disp, stats

@spaces.GPU
def cb_compare(imgL, imgR, cmap):
    if imgL is None or imgR is None:
        raise gr.Error("Please upload both Left and Right stereo images.")
    imgL, imgR = _rgb(imgL), _rgb(imgR)
    imgL, imgR = _align(imgL, imgR)
    bm_d,   bm_ms   = compute_stereo_bm(imgL, imgR, num_disparities=160, block_size=25)
    sgbm_d, sgbm_ms = compute_stereo_sgbm(imgL, imgR, num_disparities=160, block_size=3, uniqueness_ratio=10)
    raft_pre, _ = _get_raft("pretrained")
    if raft_pre:
        pre_d, pre_ms = raft_pre.compute_disparity(imgL, imgR, iters=24, max_dimension=768)
        pre_vis = colorize_disparity(pre_d, cmap_name=cmap)
        pre_lbl = "RAFT Pretrained (" + str(round(pre_ms)) + " ms)"
    else:
        pre_vis = np.zeros_like(imgL)
        pre_lbl = "Pretrained RAFT - checkpoint not found"
    raft_fine, _ = _get_raft("finetuned")
    if raft_fine:
        fine_d, fine_ms = raft_fine.compute_disparity(imgL, imgR, iters=24, max_dimension=768)
        fine_vis = colorize_disparity(fine_d, cmap_name=cmap)
        fine_lbl = "RAFT Fine-tuned (" + str(round(fine_ms)) + " ms)"
    else:
        fine_vis = np.zeros_like(imgL)
        fine_lbl = "Fine-tuned RAFT - checkpoint not found"
    return (
        gr.update(value=colorize_disparity(bm_d,   cmap_name=cmap), label="StereoBM (" + str(round(bm_ms)) + " ms)"),
        gr.update(value=colorize_disparity(sgbm_d, cmap_name=cmap), label="StereoSGBM (" + str(round(sgbm_ms)) + " ms)"),
        gr.update(value=pre_vis,  label=pre_lbl),
        gr.update(value=fine_vis, label=fine_lbl),
    )

@spaces.GPU
def cb_evaluate(imgL, imgR, gt_file, method, bad_thresh):
    if imgL is None or imgR is None:
        raise gr.Error("Please upload Left and Right stereo images.")
    if gt_file is None:
        raise gr.Error("Please upload a ground-truth disparity file (.pfm or .png).")
    imgL, imgR = _rgb(imgL), _rgb(imgR)
    imgL, imgR = _align(imgL, imgR)
    gt_path = gt_file if isinstance(gt_file, str) else gt_file.name
    if gt_path.lower().endswith(".pfm"):
        disp_gt = read_pfm(gt_path)
    else:
        raw = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise gr.Error("Cannot read GT file: " + gt_path)
        disp_gt = (cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if len(raw.shape)==3 else raw).astype(np.float32)
    if "SGBM" in method:
        disp_pred, _ = compute_stereo_sgbm(imgL, imgR, num_disparities=160, block_size=3)
    elif "StereoBM" in method:
        disp_pred, _ = compute_stereo_bm(imgL, imgR, num_disparities=160, block_size=25)
    elif "Pre-trained" in method:
        raft, _ = _get_raft("pretrained")
        if raft is None:
            raise gr.Error("Pretrained RAFT checkpoint not found.")
        disp_pred, _ = raft.compute_disparity(imgL, imgR, iters=32)
    else:
        raft, _ = _get_raft("finetuned")
        if raft is None:
            raise gr.Error("Fine-tuned RAFT checkpoint not found.")
        disp_pred, _ = raft.compute_disparity(imgL, imgR, iters=32)
    m = calculate_metrics(disp_gt, disp_pred, bad_threshold=float(bad_thresh))
    md = ("### Benchmark Results\n| Metric | Value |\n| :--- | :---: |\n" +
          "| **RMSE** | " + str(round(m["rmse"],4)) + " px |\n" +
          "| **AbsRel** | " + str(round(m["absrel"],4)) + " |\n" +
          "| **Bad-" + str(int(bad_thresh)) + "px (D1)** | " + str(round(m["bad_pixel_rate"],2)) + "% |\n" +
          "| **EPE** | " + str(round(m["epe"],4)) + " px |\n" +
          "| **Valid Pixels** | " + str(m["valid_pixels"]) + " |")
    return (colorize_disparity(disp_gt, "plasma"),
            colorize_disparity(disp_pred, "plasma"),
            create_error_heatmap(disp_gt, disp_pred, bad_threshold=float(bad_thresh)),
            md)

CSS = ".gradio-container{max-width:1200px!important;margin:auto} .metric-box{border-radius:8px;padding:10px;background:#f4f4f8}"

with gr.Blocks(title="Stereo Disparity & Depth | CO5430", css=CSS) as demo:
    gr.Markdown("""
# Stereo Disparity and Relative Depth Estimation
### CO543 / CO5430 Computer Vision - Group G03 - Project P12 - University of Peradeniya
E/23/127 H.M.K.I. Herath | E/23/188 K.M.M.Y. Kumarasinge | E/23/343 S.B.N.S. Samarawickrama | E/23/347 S.D.M.P. Sandanayake
    """)

    with gr.Tabs():
        with gr.TabItem("Disparity Estimation"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Upload Stereo Pair")
                    t1_imgL = gr.Image(label="Left Image (Rectified)",  type="numpy")
                    t1_imgR = gr.Image(label="Right Image (Rectified)", type="numpy")
                    gr.Markdown("### 2. Algorithm")
                    t1_method = gr.Dropdown(choices=METHODS, value=METHODS[0], label="Select Algorithm")
                    with gr.Accordion("Parameters", open=True):
                        t1_block = gr.Slider(1,  25,  value=3,   step=2,  label="Block Size")
                        t1_disp  = gr.Slider(16, 256, value=160, step=16, label="Num Disparities")
                        t1_uniq  = gr.Slider(0,  25,  value=10,  step=1,  label="Uniqueness Ratio")
                        t1_cmap  = gr.Dropdown(choices=["plasma","turbo","magma","viridis"], value="plasma", label="Colormap")
                    t1_btn = gr.Button("Compute Disparity Map", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown("### 3. Results")
                    with gr.Row():
                        t1_out_left = gr.Image(label="Left Camera Input",       type="numpy")
                        t1_out_disp = gr.Image(label="Colorized Disparity Map", type="numpy")
                    t1_stats = gr.Markdown("Click Compute Disparity Map to run.", elem_classes=["metric-box"])
            s_left, s_right = _ensure_samples()
            gr.Examples(
                examples=[[s_left, s_right, METHODS[0], 3, 160, 10, "plasma"],
                           [s_left, s_right, METHODS[1], 25, 160, 15, "turbo"]],
                inputs=[t1_imgL, t1_imgR, t1_method, t1_block, t1_disp, t1_uniq, t1_cmap],
                label="Sample stereo pair"
            )

        with gr.TabItem("4-Way Comparison"):
            gr.Markdown("### Compare all four methods on the same stereo pair.")
            with gr.Row():
                t2_imgL = gr.Image(label="Left Image",  type="numpy")
                t2_imgR = gr.Image(label="Right Image", type="numpy")
            with gr.Row():
                t2_cmap = gr.Dropdown(choices=["plasma","turbo","magma","viridis"], value="plasma", label="Colormap", scale=1)
                t2_btn  = gr.Button("Compare All Algorithms", variant="primary", scale=2)
            with gr.Row():
                t2_bm   = gr.Image(label="StereoBM",      type="numpy")
                t2_sgbm = gr.Image(label="StereoSGBM",    type="numpy")
            with gr.Row():
                t2_pre  = gr.Image(label="RAFT Pretrained",  type="numpy")
                t2_fine = gr.Image(label="RAFT Fine-tuned",  type="numpy")

        with gr.TabItem("Ground Truth Benchmark"):
            gr.Markdown("### Quantitative evaluation against a Middlebury ground-truth disparity map.")
            with gr.Row():
                t3_imgL = gr.Image(label="Left Image",  type="numpy")
                t3_imgR = gr.Image(label="Right Image", type="numpy")
                t3_gt   = gr.File(label="Ground Truth (.pfm / .png)")
            with gr.Row():
                t3_method = gr.Dropdown(choices=METHODS, value=METHODS[0], label="Algorithm", scale=2)
                t3_thresh = gr.Slider(0.5, 5.0, value=3.0, step=0.5, label="Bad-pixel threshold (px)", scale=1)
                t3_btn    = gr.Button("Run Benchmark", variant="primary", scale=1)
            with gr.Row():
                t3_gt_vis   = gr.Image(label="Ground Truth", type="numpy")
                t3_pred_vis = gr.Image(label="Prediction",   type="numpy")
                t3_err_vis  = gr.Image(label="Error Map",    type="numpy")
            t3_metrics = gr.Markdown("Upload all inputs and click Run Benchmark.", elem_classes=["metric-box"])

        with gr.TabItem("Methodology"):
            gr.Markdown("""
### Stereo Depth Estimation Theory

Given rectified cameras with baseline B (mm) and focal length f (px), depth Z is:

Z = f * B / d

where d is the horizontal disparity in pixels.

#### Classical Methods
- **StereoBM** - sliding-window Sum of Absolute Differences. Fast but noisy at depth edges.
- **StereoSGBM** - minimises a 2-D energy functional with smoothness penalties P1, P2.

#### RAFT-Stereo (Deep Learning)
Feature extractors + 4-level correlation pyramid + recurrent GRU update (12-32 iterations).

#### Our Results (artroom1, Middlebury 2021)
| Method | RMSE | AbsRel | Bad-3px |
| :--- | :---: | :---: | :---: |
| StereoSGBM | 38.64 px | 0.2378 | 30.23% |
| RAFT Pretrained | **3.98 px** | **0.0091** | **4.46%** |
            """)

    t1_btn.click(fn=cb_single,
                 inputs=[t1_imgL, t1_imgR, t1_method, t1_block, t1_disp, t1_uniq, t1_cmap],
                 outputs=[t1_out_left, t1_out_disp, t1_stats])
    t2_btn.click(fn=cb_compare,
                 inputs=[t2_imgL, t2_imgR, t2_cmap],
                 outputs=[t2_bm, t2_sgbm, t2_pre, t2_fine])
    t3_btn.click(fn=cb_evaluate,
                 inputs=[t3_imgL, t3_imgR, t3_gt, t3_method, t3_thresh],
                 outputs=[t3_gt_vis, t3_pred_vis, t3_err_vis, t3_metrics])

if __name__ == "__main__":
    import inspect
    is_colab = "google.colab" in sys.modules or os.environ.get("COLAB_GPU") is not None
    
    launch_opts = {
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "share": is_colab,
    }
    # Disable experimental Node.js SSR on Gradio 5 to ensure reliable container execution
    if "ssr_mode" in inspect.signature(demo.launch).parameters:
        launch_opts["ssr_mode"] = False

    demo.launch(**launch_opts)
