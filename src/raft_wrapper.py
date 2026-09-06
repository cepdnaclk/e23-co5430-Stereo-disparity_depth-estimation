"""RAFT-Stereo Deep Learning Wrapper with CPU/CUDA support and automatic checkpoint management."""

import os
import sys
import time
import urllib.request
import numpy as np
from typing import Tuple, Optional

# Ensure core and RAFT-Stereo paths can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
possible_paths = [
    os.path.join(project_root, 'core'),
    os.path.join(project_root, 'RAFT-Stereo', 'core'),
    os.path.join(project_root, 'RAFT-Stereo'),
    '/content/RAFT-Stereo/core',
    '/content/RAFT-Stereo'
]
for p in possible_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)


def get_default_args(ckpt_path: str = 'models/raftstereo-sceneflow.pth'):
    """Simulates argparse object matching the Colab configuration."""
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
    """Wrapper class for loading and running RAFT-Stereo models."""
    
    def __init__(self, checkpoint_path: str = 'models/raftstereo-sceneflow.pth', device: Optional[str] = None):
        import torch
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.checkpoint_path = checkpoint_path
        self.model = None
        self._load_model()

    def _load_model(self):
        import torch
        try:
            from raft_stereo import RAFTStereo
        except ImportError:
            raise ImportError(
                "Could not import RAFTStereo. Ensure 'RAFT-Stereo' repository is present or cloned."
            )

        args = get_default_args(self.checkpoint_path)
        self.model = RAFTStereo(args)

        if os.path.exists(self.checkpoint_path):
            state_dict = torch.load(self.checkpoint_path, map_location='cpu')
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict)
        else:
            print(f"Warning: Checkpoint not found at {self.checkpoint_path}. Model initialized with random weights.")

        self.model.to(self.device)
        self.model.eval()

    def compute_disparity(
        self,
        imgL: np.ndarray,
        imgR: np.ndarray,
        iters: int = 32,
        max_dimension: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        """Runs RAFT-Stereo inference on stereo pair.
        
        Args:
            imgL: Left RGB image (uint8 numpy array).
            imgR: Right RGB image (uint8 numpy array).
            iters: Number of GRU update iterations (default 32; 12-16 is faster for CPU).
            max_dimension: Optional downscale limit (e.g. 768) to speed up CPU inference.
            
        Returns:
            tuple of (disparity_map, inference_time_ms)
        """
        import torch
        from utils.utils import InputPadder
        import cv2

        orig_h, orig_w = imgL.shape[:2]

        # Optional downscaling for CPU performance
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

        # Rescale disparity back to original resolution if scaled
        if scale != 1.0:
            disp = cv2.resize(disp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            disp = disp / scale

        return disp.astype(np.float32), elapsed_ms
