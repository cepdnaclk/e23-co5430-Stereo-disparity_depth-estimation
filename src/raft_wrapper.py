"""RAFT-Stereo Deep Learning Wrapper with CPU/CUDA support, bundled core, and auto-download."""

import os
import sys
import time
import subprocess
import urllib.request
import numpy as np
from typing import Tuple, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Ensure project root and core directory are discoverable in sys.path
for p in [project_root, os.path.join(project_root, 'core'), '/tmp/RAFT-Stereo', '/content/RAFT-Stereo']:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)


def _get_raft_stereo_class():
    """Import and return RAFTStereo class from bundled core or fallback clone."""
    try:
        from core.raft_stereo import RAFTStereo
        return RAFTStereo
    except ImportError:
        pass

    try:
        from raft_stereo import RAFTStereo
        return RAFTStereo
    except ImportError:
        pass

    # Clone RAFT-Stereo as emergency fallback
    clone_target = '/tmp/RAFT-Stereo'
    if not os.path.exists(os.path.join(clone_target, 'core', 'raft_stereo.py')):
        print(f"[RAFT-Stereo] Cloning official RAFT-Stereo repository into {clone_target}...")
        try:
            subprocess.run(
                ['git', 'clone', '--depth', '1', 'https://github.com/princeton-vl/RAFT-Stereo.git', clone_target],
                check=True,
                capture_output=True,
                timeout=120
            )
        except Exception as e:
            print(f"[RAFT-Stereo] Fallback clone warning: {e}")

    for p in [clone_target, os.path.join(clone_target, 'core')]:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from core.raft_stereo import RAFTStereo
        return RAFTStereo
    except ImportError:
        from raft_stereo import RAFTStereo
        return RAFTStereo


def _ensure_checkpoint(ckpt_path: Optional[str], mode: str = "pretrained") -> str:
    """Ensure model checkpoint file exists, auto-downloading from Hugging Face mirror if needed."""
    if ckpt_path and os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 1000000:
        return ckpt_path

    # Determine target download path
    models_dir = os.path.join(project_root, "models")
    try:
        os.makedirs(models_dir, exist_ok=True)
    except Exception:
        models_dir = "/tmp/models"
        os.makedirs(models_dir, exist_ok=True)

    if mode == "finetuned":
        local_ckpt = os.path.join(models_dir, "raftstereo-middlebury.pth")
        url = "https://huggingface.co/shriarul5273/RAFT-Stereo/resolve/main/raftstereo-middlebury.pth"
    else:
        local_ckpt = os.path.join(models_dir, "raftstereo-sceneflow.pth")
        url = "https://huggingface.co/shriarul5273/RAFT-Stereo/resolve/main/raftstereo-sceneflow.pth"

    if os.path.exists(local_ckpt) and os.path.getsize(local_ckpt) > 1000000:
        return local_ckpt

    print(f"[RAFT-Stereo] Auto-downloading {mode} weights from {url} to {local_ckpt}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=180) as response, open(local_ckpt, 'wb') as out_file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
        print(f"[RAFT-Stereo] Download complete: {local_ckpt} ({os.path.getsize(local_ckpt)} bytes)")
        return local_ckpt
    except Exception as e:
        print(f"[RAFT-Stereo] Download failed: {e}")
        # Try gdown fallback for sceneflow
        if mode != "finetuned":
            try:
                import gdown
                gdown.download(id="1e7z5vIo7aFxVl7R5FbflAZlnS2G9IZRj", output=local_ckpt, quiet=False)
                if os.path.exists(local_ckpt) and os.path.getsize(local_ckpt) > 1000000:
                    return local_ckpt
            except Exception as e2:
                print(f"[RAFT-Stereo] gdown fallback failed: {e2}")
        return ckpt_path or ""


def get_default_args(ckpt_path: str = 'models/raftstereo-sceneflow.pth'):
    """Simulates argparse object matching the Colab and official evaluation configuration."""
    args = type('', (), {})()
    args.restore_ckpt = ckpt_path
    args.shared_backbone = False
    args.n_downsample = 2
    args.n_gru_layers = 3
    args.n_gru_vox_iters = 0
    args.slow_fast_gru = False
    args.valid_iters = 32
    args.corr_implementation = 'reg'
    args.mixed_precision = False
    args.hidden_dims = [128, 128, 128]
    args.context_dims = [128, 128, 128]
    args.context_norm = "batch"
    args.corr_levels = 4
    args.corr_radius = 4
    return args


class RAFTStereoInference:
    """Wrapper class for loading and running RAFT-Stereo models with bundled core."""

    def __init__(self, checkpoint_path: Optional[str] = None, mode: str = "pretrained", device: Optional[str] = None):
        import torch

        self.device = torch.device(device) if device else torch.device('cpu')
        self.checkpoint_path = _ensure_checkpoint(checkpoint_path, mode=mode)
        self.model = None
        self._load_model()

    def _load_model(self):
        import torch

        RAFTStereo = _get_raft_stereo_class()
        args = get_default_args(self.checkpoint_path)
        self.model = RAFTStereo(args)

        if os.path.exists(self.checkpoint_path) and os.path.getsize(self.checkpoint_path) > 1000000:
            try:
                state_dict = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
            except Exception:
                state_dict = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict, strict=False)
            print(f"[RAFT-Stereo] Successfully loaded weights from {self.checkpoint_path}")
        else:
            print(f"[RAFT-Stereo] Warning: Checkpoint not found at {self.checkpoint_path}. Initialized default weights.")

        self.model.to(self.device)
        self.model.eval()

    def compute_disparity(
        self,
        imgL: np.ndarray,
        imgR: np.ndarray,
        iters: int = 32,
        max_dimension: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        """Runs RAFT-Stereo inference on stereo pair."""
        import torch
        try:
            from core.utils.utils import InputPadder
        except ImportError:
            from utils.utils import InputPadder
        import cv2

        orig_h, orig_w = imgL.shape[:2]

        # Dynamically use CUDA when inside an active ZeroGPU or GPU context
        exec_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(exec_device)

        # Optional downscaling for CPU/memory efficiency
        scale = 1.0
        if max_dimension and max(orig_h, orig_w) > max_dimension:
            scale = max_dimension / float(max(orig_h, orig_w))
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            imgL_in = cv2.resize(imgL, (new_w, new_h), interpolation=cv2.INTER_AREA)
            imgR_in = cv2.resize(imgR, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            imgL_in = imgL
            imgR_in = imgR

        tensorL = torch.from_numpy(imgL_in).permute(2, 0, 1).float()[None].to(exec_device)
        tensorR = torch.from_numpy(imgR_in).permute(2, 0, 1).float()[None].to(exec_device)

        padder = InputPadder(tensorL.shape, divis_by=32)
        tensorL_pad, tensorR_pad = padder.pad(tensorL, tensorR)

        start_time = time.perf_counter()
        with torch.no_grad():
            _, flow_up = self.model(tensorL_pad, tensorR_pad, iters=iters, test_mode=True)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        disp = padder.unpad(flow_up).squeeze().cpu().numpy()
        disp = np.abs(disp)

        # Suppress boundary edge extrapolation spikes from padding before resizing
        h_disp, w_disp = disp.shape[:2]
        b = max(2, int(min(h_disp, w_disp) * 0.015))
        if b > 0 and h_disp > 2 * b and w_disp > 2 * b:
            disp[:b, :] = disp[b:b+1, :]
            disp[-b:, :] = disp[-b-1:-b, :]
            disp[:, :b] = disp[:, b:b+1]
            disp[:, -b:] = disp[:, -b-1:-b]

        # Release GPU memory cleanly for ZeroGPU
        if exec_device.type == 'cuda':
            self.model.to('cpu')
            torch.cuda.empty_cache()

        if scale != 1.0:
            disp = cv2.resize(disp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            disp = disp / scale

        return disp.astype(np.float32), elapsed_ms
