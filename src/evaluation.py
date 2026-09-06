"""Quantitative evaluation metrics for stereo disparity maps.

Implements RMSE, AbsRel, and Bad-Pixel Rate (D1 metric) conforming to Middlebury benchmark standards.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def calculate_metrics(
    gt_disp: np.ndarray,
    pred_disp: np.ndarray,
    mask: Optional[np.ndarray] = None,
    bad_threshold: float = 3.0,
) -> Dict[str, float]:
    """Calculates quantitative benchmark metrics against ground-truth disparity.
    
    Conforms to the final Colab implementation (Cell 22):
        - RMSE (Root Mean Squared Error): sqrt(mean((gt - pred)**2))
        - AbsRel (Absolute Relative Error): mean(|gt - pred| / gt)
        - Bad-pixel rate (D1 error rate): percentage of valid pixels with absolute error > bad_threshold
        
    Args:
        gt_disp: Ground-truth disparity map (2D float array).
        pred_disp: Predicted disparity map (2D float array).
        mask: Optional boolean mask of valid pixels. If None, valid is (gt_disp > 0) & (gt_disp < inf).
        bad_threshold: Pixel error threshold for bad pixels (default: 3.0 px for Middlebury).
        
    Returns:
        dict containing 'rmse', 'absrel', 'bad_pixel_rate', and 'valid_pixels'.
    """
    if mask is None:
        mask = (gt_disp > 0) & (gt_disp < np.inf)

    # Ensure prediction is resized if dimensions differ slightly (e.g. from downsampling or cropping)
    if gt_disp.shape != pred_disp.shape:
        import cv2
        pred_disp = cv2.resize(pred_disp, (gt_disp.shape[1], gt_disp.shape[0]), interpolation=cv2.INTER_LINEAR)

    gt_masked = gt_disp[mask]
    pred_masked = pred_disp[mask]

    if gt_masked.size == 0:
        return {
            'rmse': float('nan'),
            'absrel': float('nan'),
            'bad_pixel_rate': float('nan'),
            'valid_pixels': 0
        }

    # Absolute difference
    abs_diff = np.abs(gt_masked - pred_masked)

    # RMSE (Root Mean Squared Error)
    rmse = float(np.sqrt(np.mean((gt_masked - pred_masked) ** 2)))

    # Absolute Relative Error (AbsRel)
    absrel = float(np.mean(abs_diff / gt_masked))

    # Bad-pixel rate (percentage of pixels with error > bad_threshold)
    bad_pixels = np.sum(abs_diff > bad_threshold)
    bad_pixel_rate = float((bad_pixels / gt_masked.size) * 100.0)

    # Mean Absolute Error / End-Point Error (EPE)
    epe = float(np.mean(abs_diff))

    return {
        'rmse': rmse,
        'absrel': absrel,
        'bad_pixel_rate': bad_pixel_rate,
        'epe': epe,
        'valid_pixels': int(gt_masked.size)
    }
