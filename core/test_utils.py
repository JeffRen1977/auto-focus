"""
Test Utilities for Autofocus System
====================================

Utility functions for testing and demonstration purposes.
These are used in examples and advanced modules for creating test data.
"""

import numpy as np
import cv2
from typing import Tuple
from scipy import ndimage


def create_test_image(size: Tuple[int, int] = (480, 640), 
                     blur_radius: float = 2.0) -> np.ndarray:
    """
    Create a test image with controlled blur.
    
    This function generates a synthetic test pattern useful for testing
    autofocus algorithms. The pattern has high-frequency content that
    is ideal for contrast-based focusing.
    
    Args:
        size: Image size as (height, width) tuple
        blur_radius: Gaussian blur radius (sigma) - larger = more blur
    
    Returns:
        Test image as uint8 numpy array (grayscale)
    """
    h, w = size
    
    # Create a synthetic test pattern
    y, x = np.ogrid[:h, :w]
    
    # Create high-frequency pattern (good for contrast detection)
    pattern = np.sin(x * 0.1) * np.sin(y * 0.1) * 128 + 128
    
    # Add some structure
    center_x, center_y = w // 2, h // 2
    mask = ((x - center_x)**2 + (y - center_y)**2) < (min(w, h) // 4)**2
    pattern[mask] *= 1.5
    
    # Apply blur
    if blur_radius > 0:
        pattern = ndimage.gaussian_filter(pattern, sigma=blur_radius)
    
    # Convert to uint8
    image = np.clip(pattern, 0, 255).astype(np.uint8)
    
    return image


def create_test_pd_data(image: np.ndarray, 
                       disparity: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create test PDAF data with known disparity.
    
    This function simulates left and right phase detection images by
    applying a horizontal shift (disparity) to the right image.
    
    Args:
        image: Input image to use as base
        disparity: Horizontal disparity in pixels
                  - Positive: Right image shifted right (subject behind focus)
                  - Negative: Right image shifted left (subject in front)
                  - Zero: No shift (in focus)
    
    Returns:
        Tuple of (left_pd, right_pd) images
    """
    # Simulate left and right PD images
    left_pd = image.astype(np.float32)
    
    # Shift right image by disparity
    if disparity != 0:
        shift_matrix = np.array([[1, 0, disparity], [0, 1, 0]], dtype=np.float32)
        right_pd = cv2.warpAffine(image.astype(np.float32), shift_matrix, 
                                 (image.shape[1], image.shape[0]))
    else:
        right_pd = left_pd.copy()
    
    return left_pd, right_pd







