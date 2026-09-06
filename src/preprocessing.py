"""Preprocessing utilities for stereo images and disparity ground truth maps."""

import numpy as np
import cv2
from typing import Tuple, Optional


def crop_stereo_pair(
    imgL: np.ndarray,
    imgR: np.ndarray,
    disp_gt: Optional[np.ndarray] = None,
    crop_size: Tuple[int, int] = (384, 512),
    random_crop: bool = False,
) -> Tuple:
    """Crops stereo pair (and optional disparity ground truth) to specified dimensions.
    
    Used in Colab Milestone 3 to prevent GPU VRAM exhaustion during training and inference.
    
    Args:
        imgL: Left image (H, W, C) or (H, W).
        imgR: Right image (H, W, C) or (H, W).
        disp_gt: Optional disparity ground-truth map (H, W).
        crop_size: Target (crop_h, crop_w).
        random_crop: If True, selects random spatial offset; if False, crops center.
        
    Returns:
        tuple of (cropped_L, cropped_R) or (cropped_L, cropped_R, cropped_gt).
    """
    h, w = imgL.shape[:2]
    crop_h, crop_w = crop_size

    if h < crop_h or w < crop_w:
        raise ValueError(f"Image dimensions ({h}, {w}) are smaller than target crop ({crop_h}, {crop_w})")

    if random_crop:
        cy = np.random.randint(0, h - crop_h + 1)
        cx = np.random.randint(0, w - crop_w + 1)
    else:
        cy = (h - crop_h) // 2
        cx = (w - crop_w) // 2

    crop_L = imgL[cy : cy + crop_h, cx : cx + crop_w]
    crop_R = imgR[cy : cy + crop_h, cx : cx + crop_w]

    if disp_gt is not None:
        crop_gt = disp_gt[cy : cy + crop_h, cx : cx + crop_w]
        return crop_L, crop_R, crop_gt

    return crop_L, crop_R


def pad_to_multiple(
    img: np.ndarray,
    multiple: int = 32,
    mode: str = 'edge',
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Pads image so height and width are integer multiples of `multiple`.
    
    Required for multi-scale pyramid architectures (e.g. RAFT-Stereo).
    
    Returns:
        padded_img: Padded numpy array.
        padding_box: (pad_top, pad_bottom, pad_left, pad_right) to allow unpadding.
    """
    h, w = img.shape[:2]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if len(img.shape) == 3:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))

    padded = np.pad(img, padding, mode=mode)
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def unpad(img: np.ndarray, padding_box: Tuple[int, int, int, int]) -> np.ndarray:
    """Removes padding added by `pad_to_multiple`."""
    pad_top, pad_bottom, pad_left, pad_right = padding_box
    h, w = img.shape[:2]
    return img[pad_top : h - pad_bottom, pad_left : w - pad_right]
