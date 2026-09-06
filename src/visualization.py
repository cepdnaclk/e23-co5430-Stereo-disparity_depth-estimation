"""Visualization utilities for disparity maps, depth maps, and error heatmaps."""

import numpy as np
import matplotlib.cm as cm
import cv2
from typing import Optional, Tuple


def colorize_disparity(
    disp: np.ndarray,
    cmap_name: str = 'plasma',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    invalid_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Colorizes a 2D disparity map into an RGB uint8 image.
    
    Args:
        disp: 2D float array of disparities.
        cmap_name: Matplotlib colormap name ('plasma', 'turbo', 'viridis', 'magma').
        vmin: Minimum value for colormap scaling. If None, computes 1st percentile of valid pixels.
        vmax: Maximum value for colormap scaling. If None, computes 99th percentile of valid pixels.
        invalid_color: RGB tuple for pixels where disp <= 0.
        
    Returns:
        RGB image as uint8 numpy array of shape (H, W, 3).
    """
    valid_mask = (disp > 0) & np.isfinite(disp)
    
    if not np.any(valid_mask):
        return np.zeros((disp.shape[0], disp.shape[1], 3), dtype=np.uint8)

    valid_vals = disp[valid_mask]
    if vmin is None:
        vmin = float(np.percentile(valid_vals, 1))
    if vmax is None:
        vmax = float(np.percentile(valid_vals, 99))

    if vmax <= vmin:
        vmax = vmin + 1.0

    # Normalize between 0.0 and 1.0
    norm_disp = np.clip((disp - vmin) / (vmax - vmin), 0.0, 1.0)

    # Apply colormap
    try:
        colormap = cm.get_cmap(cmap_name)
    except Exception:
        colormap = cm.plasma

    rgba_img = colormap(norm_disp)
    rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)

    # Set invalid pixels to invalid_color
    rgb_img[~valid_mask] = invalid_color

    return rgb_img


def create_error_heatmap(
    gt_disp: np.ndarray,
    pred_disp: np.ndarray,
    bad_threshold: float = 3.0,
    max_error: float = 15.0,
) -> np.ndarray:
    """Generates an RGB error heatmap highlighting pixel errors.
    
    Errors below bad_threshold (accurate matches) appear dark/cool.
    Errors exceeding bad_threshold (bad pixels) transition through yellow to intense red.
    Invalid ground-truth pixels are masked in gray.
    
    Args:
        gt_disp: Ground-truth disparity.
        pred_disp: Predicted disparity.
        bad_threshold: Bad pixel error threshold (default: 3.0 px).
        max_error: Saturation point for the error scale in pixels.
        
    Returns:
        RGB uint8 image of shape (H, W, 3).
    """
    if gt_disp.shape != pred_disp.shape:
        pred_disp = cv2.resize(pred_disp, (gt_disp.shape[1], gt_disp.shape[0]), interpolation=cv2.INTER_LINEAR)

    valid_mask = (gt_disp > 0) & np.isfinite(gt_disp)
    error = np.abs(gt_disp - pred_disp)

    # Normalize error to [0, 1]
    norm_error = np.clip(error / max_error, 0.0, 1.0)
    
    # Use 'turbo' or 'inferno' for high-contrast error rendering
    colormap = cm.turbo
    rgba = colormap(norm_error)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    # Mask invalid ground truth areas with dark gray (40, 40, 40)
    rgb[~valid_mask] = (40, 40, 40)

    return rgb
