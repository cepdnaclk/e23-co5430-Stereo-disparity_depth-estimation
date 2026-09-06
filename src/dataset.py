"""Dataset loading, PFM reader, and calibration utilities for Middlebury stereo pairs."""

import os
import re
import numpy as np
import cv2


def read_pfm(file_path: str) -> np.ndarray:
    """Reads a Middlebury .pfm (Portable Float Map) ground-truth disparity file.
    
    Handles little-endian and big-endian binary float data and flips the image
    vertically to align with standard image coordinates.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PFM file not found: {file_path}")

    with open(file_path, 'rb') as file:
        header = file.readline().rstrip().decode("ascii")
        color = (header == 'PF')
        if header not in ('PF', 'Pf'):
            raise ValueError(f"Not a valid PFM file: {file_path}")

        dimensions = file.readline().decode("ascii").split()
        width, height = map(int, dimensions)
        
        scale = float(file.readline().rstrip().decode("ascii"))
        endian = '<' if scale < 0 else '>'
        
        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        
        # Replace inf with 0 or NaN handling
        data = data.astype(np.float32)
        return data


def load_stereo_pair(left_path: str, right_path: str, to_gray: bool = False):
    """Loads a rectified stereo image pair.
    
    Args:
        left_path: Path to left camera image.
        right_path: Path to right camera image.
        to_gray: If True, returns single-channel grayscale images.
        
    Returns:
        tuple of (imgL, imgR) as numpy arrays.
    """
    flags = cv2.IMREAD_GRAYSCALE if to_gray else cv2.IMREAD_COLOR
    imgL = cv2.imread(left_path, flags)
    imgR = cv2.imread(right_path, flags)

    if imgL is None:
        raise ValueError(f"Could not load left image from: {left_path}")
    if imgR is None:
        raise ValueError(f"Could not load right image from: {right_path}")

    if not to_gray:
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

    return imgL, imgR


def load_calib(calib_path: str) -> dict:
    """Parses Middlebury calib.txt calibration file.
    
    Extracts baseline B, focal length f (from cam0), ndisp, vmin, vmax.
    """
    calib = {}
    if not os.path.exists(calib_path):
        return calib

    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()
                if key in ('cam0', 'cam1'):
                    # 3x3 matrix in format: [fx 0 cx; 0 fy cy; 0 0 1]
                    matrix_vals = re.findall(r'[-+]?\d*\.\d+|\d+', val)
                    if len(matrix_vals) == 9:
                        calib[key] = np.array(matrix_vals, dtype=float).reshape(3, 3)
                elif key in ('baseline', 'ndisp', 'vmin', 'vmax', 'width', 'height'):
                    try:
                        calib[key] = float(val)
                    except ValueError:
                        calib[key] = val

    # Extract focal length from cam0 if present
    if 'cam0' in calib:
        calib['focal_length'] = float(calib['cam0'][0, 0])
        
    return calib


def disparity_to_depth(disparity: np.ndarray, baseline_mm: float, focal_length_px: float) -> np.ndarray:
    """Converts a disparity map to a metric depth map using: Z = (f * B) / d
    
    Args:
        disparity: Disparity map in pixels (2D float array).
        baseline_mm: Distance between stereo camera centers in millimeters.
        focal_length_px: Focal length in pixels.
        
    Returns:
        2D float array of depth values in millimeters (0 where disparity <= 0).
    """
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > 0
    depth[valid] = (focal_length_px * baseline_mm) / disparity[valid]
    return depth
