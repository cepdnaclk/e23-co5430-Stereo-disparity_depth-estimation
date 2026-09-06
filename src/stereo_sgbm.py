"""Semi-Global Block Matching (StereoSGBM) implementation."""

import cv2
import numpy as np
import time
from typing import Tuple


def compute_stereo_sgbm(
    imgL: np.ndarray,
    imgR: np.ndarray,
    min_disparity: int = 0,
    num_disparities: int = 160,
    block_size: int = 3,
    uniqueness_ratio: int = 10,
    speckle_window_size: int = 100,
    speckle_range: int = 32,
    disp12_max_diff: int = 1,
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
) -> Tuple[np.ndarray, float]:
    """Computes disparity map using OpenCV StereoSGBM algorithm.
    
    Dynamically recalculates smoothness penalties P1 and P2 based on OpenCV recommendations:
        P1 = 8 * channels * (block_size ** 2)
        P2 = 32 * channels * (block_size ** 2)
        
    Args:
        imgL: Left image (RGB or Grayscale).
        imgR: Right image (RGB or Grayscale).
        min_disparity: Minimum possible disparity value (default 0).
        num_disparities: Maximum disparity search range (must be divisible by 16).
        block_size: Matched block size (odd number >= 1, typically 3 to 11).
        uniqueness_ratio: Margin in percentage for winning disparity (typically 5-15).
        speckle_window_size: Maximum size of smooth disparity regions to consider their noise specks.
        speckle_range: Maximum disparity variation within each connected component.
        disp12_max_diff: Maximum allowed difference in left-right disparity check.
        mode: SGBM mode (SGBM_3WAY or SGBM standard).
        
    Returns:
        tuple of (disparity_map, inference_time_ms):
            disparity_map: 2D float32 array in pixels.
            inference_time_ms: Computation time in milliseconds.
    """
    # SGBM can use 3 channels or grayscale. Grayscale is standard in the course baseline.
    if len(imgL.shape) == 3:
        grayL = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
        channels = 1
    else:
        grayL = imgL.copy()
        channels = 1

    if len(imgR.shape) == 3:
        grayR = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)
    else:
        grayR = imgR.copy()

    # Ensure num_disparities is divisible by 16 and >= 16
    num_disparities = max(16, (num_disparities // 16) * 16)
    
    # Ensure block_size is odd and >= 1
    block_size = max(1, block_size)
    if block_size % 2 == 0:
        block_size += 1

    # OpenCV recommended formulas for P1 and P2
    p1 = 8 * channels * (block_size ** 2)
    p2 = 32 * channels * (block_size ** 2)

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=disp12_max_diff,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        mode=mode
    )

    start_time = time.perf_counter()
    raw_disp = stereo.compute(grayL, grayR)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Output is 16-bit fixed point with 4 fractional bits: divide by 16.0
    disparity = raw_disp.astype(np.float32) / 16.0

    # Mark invalid disparities as 0.0
    disparity[disparity < min_disparity] = 0.0

    return disparity, elapsed_ms
