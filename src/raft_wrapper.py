"""RAFT-Stereo Deep Learning Wrapper with CPU/CUDA support, auto-clone, and auto-download."""

import os
import sys
import time
import subprocess
import urllib.request
import numpy as np
from typing import Tuple, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)


def _ensure_raft_stereo_core():
    """Ensure RAFT-Stereo repository is cloned and importable."""
    try:
        from raft_stereo import RAFTStereo
        return
    except ImportError:
        pass

    search_dirs = [
        os.path.join(project_root, 'RAFT-Stereo'),
        '/tmp/RAFT-Stereo',
        '/content/RAFT-Stereo',
        os.path.join(project_root, 'core'),
        '/tmp/test_raft'
    ]
    for d in search_dirs:
        core = os.path.join(d, 'core') if os.path.exists(os.path.join(d, 'core')) else d
        if os.path.exists(os.path.join(core, 'raft_stereo.py')):
            if core not in sys.path:
                sys.path.insert(0, core)
            if d not in sys.path:
                sys.path.insert(0, d)
            try:
                from raft_stereo import RAFTStereo
                return
            except ImportError:
                pass

    # Clone RAFT-Stereo automatically into /tmp/RAFT-Stereo
    clone_target = '/tmp/RAFT-Stereo'
    print(f"[RAFT-Stereo] Cloning official RAFT-Stereo repository into {clone_target}...")
    try:
        subprocess.run(
            ['git', 'clone', '--depth', '1', 'https://github.com/princeton-vl/RAFT-Stereo.git', clone_target],
            check=True,
            capture_output=True,
            timeout=120
        )
        core_dir = os.path.join(clone_target, 'core')
        if core_dir not in sys.path:
            sys.path.insert(0, core_dir)
        if clone_target not in sys.path:
            sys.path.insert(0, clone_target)
        from raft_stereo import RAFTStereo
        print("[RAFT-Stereo] Repository cloned and imported successfully.")
    except Exception as e:
        raise ImportError(
            f"Could not import or auto-clone RAFTStereo: {e}. "
            "Please ensure internet access is available or clone https://github.com/princeton-vl/RAFT-Stereo.git"
        )


def _ensure_checkpoint(ckpt_path: Optional[str], mode: str = "pretrained") -> str:
    """Ensure model checkpoint file exists, auto-downloading from Hugging Face if needed."""
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
        # Download with stream
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=180) as response, open(local_ckpt, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"[RAFT-Stereo] Download complete: {local_ckpt} ({os.path.getsize(local_ckpt)} bytes)")
        return local_ckpt
    except Exception as e:
        print(f"[RAFT-Stereo] Download failed: {e}")
        # Try gdown fallback for sceneflow
        if mode != "finetuned":
            try:
                import gdown
                gdown.download(id="1e7z5vIo7aFxVl7R5FbflAZlnS2G9IZRj", output=local_ckpt, quiet=False)
                return local_ckpt
            except Exception:
                pass
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
    """Wrapper class for loading and running RAFT-Stereo models with auto-setup."""

    def __init__(self, checkpoint_path: Optional[str] = None, mode: str = "pretrained", device: Optional[str] = None):
        import torch

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        _ensure_raft_stereo_core()
        self.checkpoint_path = _ensure_checkpoint(checkpoint_path, mode=mode)
        self.model = None
        self._load_model()

    def _load_model(self):
        import torch
        from raft_stereo import RAFTStereo

        args = get_default_args(self.checkpoint_path)
        self.model = RAFTStereo(args)

        if os.path.exists(self.checkpoint_path) and os.path.getsize(self.checkpoint_path) > 1000000:
            state_dict = torch.load(self.checkpoint_path, map_location='cpu')
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict)
            print(f"[RAFT-Stereo] Successfully loaded weights from {self.checkpoint_path}")
        else:
            print(f"[RAFT-Stereo] Warning: Checkpoint not found at {self.checkpoint_path}. Model initialized with default weights.")

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
        from utils.utils import InputPadder
        import cv2

        orig_h, orig_w = imgL.shape[:2]

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

        tensorL = torch.from_numpy(imgL_in).permute(2, 0, 1).float()[None].to(self.device)
        tensorR = torch.from_numpy(imgR_in).permute(2, 0, 1).float()[None].to(self.device)

        padder = InputPadder(tensorL.shape, divis_by=32)
        tensorL_pad, tensorR_pad = padder.pad(tensorL, tensorR)

        start_time = time.perf_counter()
        with torch.no_grad():
            _, flow_up = self.model(tensorL_pad, tensorR_pad, iters=iters, test_mode=True)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        disp = padder.unpad(flow_up).squeeze().cpu().numpy()
        disp = np.abs(disp)

        if scale != 1.0:
            disp = cv2.resize(disp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            disp = disp / scale

        return disp.astype(np.float32), elapsed_ms
