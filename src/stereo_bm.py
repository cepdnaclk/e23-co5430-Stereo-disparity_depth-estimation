"""Stereo Block Matching (StereoBM) implementation."""

import cv2
import numpy as np
import time
from typing import Tuple


def compute_stereo_bm(
    imgL: np.ndarray,
    imgR: np.ndarray,
    num_disparities: int = 160,
    block_size: int = 25,
    texture_threshold: int = 10,
    uniqueness_ratio: int = 15,
) -> Tuple[np.ndarray, float]:
    """Computes disparity map using OpenCV StereoBM algorithm.
    
    Args:
        imgL: Left image (RGB or Grayscale).
        imgR: Right image (RGB or Grayscale).
        num_disparities: Maximum disparity search range (must be divisible by 16).
        block_size: Size of sliding window (must be an odd integer >= 5).
        texture_threshold: Filter out areas without enough texture.
        uniqueness_ratio: Margin in percentage by which best cost function value must beat second best.
        
    Returns:
        tuple of (disparity_map, inference_time_ms):
            disparity_map: 2D float32 array in pixels.
            inference_time_ms: Computation time in milliseconds.
    """
    # StereoBM requires single-channel 8-bit grayscale images
    if len(imgL.shape) == 3:
        grayL = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
    else:
        grayL = imgL.copy()

    if len(imgR.shape) == 3:
        grayR = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)
    else:
        grayR = imgR.copy()

    # Ensure num_disparities is divisible by 16 and >= 16
    num_disparities = max(16, (num_disparities // 16) * 16)
    
    # Ensure block_size is odd and >= 5
    block_size = max(5, block_size)
    if block_size % 2 == 0:
        block_size += 1

    stereo = cv2.StereoBM_create(
        numDisparities=num_disparities,
        blockSize=block_size
    )
    stereo.setTextureThreshold(texture_threshold)
    stereo.setUniquenessRatio(uniqueness_ratio)

    start_time = time.perf_counter()
    # Output is 16-bit fixed-point disparity with 4 fractional bits
    raw_disp = stereo.compute(grayL, grayR)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Divide by 16.0 to obtain true floating-point disparity in pixels
    disparity = raw_disp.astype(np.float32) / 16.0

    # Mark invalid disparities (OpenCV sets invalid to negative values)
    disparity[disparity < 0] = 0.0

    return disparity, elapsed_ms
