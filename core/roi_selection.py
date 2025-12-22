"""
Region of Interest (ROI) Selection
===================================

This module implements intelligent ROI selection for autofocus operations.
Selecting the right region of the image to focus on is crucial for autofocus
performance - focusing on the subject (not background) improves accuracy and speed.

Class:
- ROISelector: Selects optimal ROI using priority-based strategy

Learning Order:
This module should be studied after config.py to understand how the system
decides which part of the image to focus on.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List

from .config import AFConfig, ROIData


class ROISelector:
    """
    ROI (Region of Interest) Selector.
    
    Intelligently selects the best area of the image for focusing.
    ROI selection is critical for autofocus performance because:
    - Focusing on the subject (not background) improves accuracy
    - Smaller ROI reduces computation time
    - User intent (touch point) should be prioritized
    - Face detection provides automatic subject selection
    
    Selection Priority (highest to lowest):
    1. User Touch Point: User explicitly selected area (highest priority)
    2. Face Detection: Automatically focus on detected faces
    3. Saliency-Based: Find visually interesting regions (edges, texture)
    4. Center-Weighted: Default to center region (fallback)
    
    Each method has a confidence score indicating how reliable the selection is.
    
    Algorithm Flow:
    The selector uses a hierarchical strategy, checking each priority level
    in order until a valid ROI is found. This ensures user intent is always
    respected, while providing intelligent fallbacks when no explicit input
    is available.
    """
    
    def __init__(self, config: AFConfig):
        """
        Initialize ROI selector with configuration.
        
        Args:
            config: AFConfig object containing ROI parameters (especially roi_size)
        
        Example:
            config = AFConfig(roi_size=(200, 200))
            selector = ROISelector(config)
        """
        self.config = config
    
    def select_roi(self, image: np.ndarray, 
                   touch_point: Optional[Tuple[int, int]] = None,
                   faces: Optional[List[Tuple[int, int, int, int]]] = None) -> ROIData:
        """
        Select the best ROI for autofocus using priority-based selection.
        
        This method implements a hierarchical selection strategy that prioritizes
        user intent and subject detection over default behavior.
        
        Selection Strategy:
        1. Touch Point (Priority 1):
           - If user touched screen, center ROI on touch point
           - Confidence: 1.0 (highest, user explicitly selected)
           - ROI size: Uses default from config
           - Fastest selection (O(1) complexity)
        
        2. Face Detection (Priority 2):
           - If faces detected, use largest face
           - Expand face region by expand_ratio (default 2.0x)
           - Confidence: 0.9 (high, but not as high as user selection)
           - ROI size: Based on face size
           - Moderate complexity (O(N) where N = number of faces)
        
        3. Saliency-Based (Priority 3):
           - Find region with highest edge density
           - Uses Sobel operator to detect edges
           - Confidence: 0.0-0.8 (based on edge density)
           - ROI size: Uses default from config
           - Slower (O(W*H) complexity for edge detection)
        
        4. Center-Weighted (Priority 4):
           - Default fallback: center of image
           - Confidence: 0.5 (lowest, generic selection)
           - ROI size: Uses default from config
           - Fastest fallback (O(1) complexity)
        
        Args:
            image: Input image as numpy array
                   - Can be grayscale or color
                   - Shape: (H, W) or (H, W, 3)
                   - Used for saliency calculation if needed
            
            touch_point: User touch point (x, y) in pixels (optional)
                        - Highest priority if provided
                        - ROI centers on this point
                        - Format: (x, y) tuple
            
            faces: List of detected faces (optional)
                  - Format: [(x, y, width, height), ...]
                  - Second priority if provided
                  - Uses largest face if multiple detected
                  - Face coordinates are in image pixel space
        
        Returns:
            ROIData object containing:
            - x, y: Top-left corner of ROI in pixels
            - width, height: ROI dimensions in pixels
            - confidence: Selection confidence (0.0-1.0)
            - method: Selection method used ('touch', 'face', 'saliency', 'center')
        
        ROI Expansion:
        For face detection, the ROI is expanded around the face to include
        context. This ensures the focus includes the face and surrounding area,
        not just the face itself. The expansion ratio is 2.0x by default.
        
        ROI Boundary Clamping:
        All ROIs are clamped to image boundaries to ensure valid pixel coordinates.
        If the requested ROI would extend beyond image edges, it is adjusted
        to fit within the image.
        
        Example:
            # User touched screen at (300, 200)
            roi = selector.select_roi(image, touch_point=(300, 200))
            # Result: ROI centers on (300, 200), method='touch', confidence=1.0
            
            # Face detected at (100, 150, 80, 100) - (x, y, width, height)
            roi = selector.select_roi(image, faces=[(100, 150, 80, 100)])
            # Result: ROI expands around face, method='face', confidence=0.9
            
            # No user input, use saliency or center
            roi = selector.select_roi(image)
            # Result: method='saliency' or 'center' depending on edge density
        """
        h, w = image.shape[:2]
        roi_w, roi_h = self.config.roi_size
        
        # Priority 1: User touch point
        if touch_point is not None:
            tx, ty = touch_point
            # Center ROI on touch point, clamp to image boundaries
            x = max(0, min(tx - roi_w//2, w - roi_w))
            y = max(0, min(ty - roi_h//2, h - roi_h))
            return ROIData(x, y, roi_w, roi_h, 1.0, 'touch')
        
        # Priority 2: Face detection
        if faces and len(faces) > 0:
            # Use the largest face (area = width * height)
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            fx, fy, fw, fh = largest_face
            
            # Expand face area for focusing (include context around face)
            expand_ratio = 2.0
            x = max(0, int(fx - (fw * expand_ratio - fw) / 2))
            y = max(0, int(fy - (fh * expand_ratio - fh) / 2))
            width = min(w - x, int(fw * expand_ratio))
            height = min(h - y, int(fh * expand_ratio))
            
            return ROIData(x, y, width, height, 0.9, 'face')
        
        # Priority 3: Saliency-based selection
        saliency_roi = self._select_saliency_roi(image, roi_w, roi_h)
        if saliency_roi is not None:
            return saliency_roi
        
        # Priority 4: Center-weighted
        x = (w - roi_w) // 2
        y = (h - roi_h) // 2
        return ROIData(x, y, roi_w, roi_h, 0.5, 'center')
    
    def _select_saliency_roi(self, image: np.ndarray, roi_w: int, roi_h: int) -> Optional[ROIData]:
        """
        Select ROI based on saliency (edge density).
        
        This method finds the region with the highest edge density, which
        typically corresponds to areas with fine detail and texture - ideal
        for autofocus operations.
        
        Algorithm:
        1. Convert image to grayscale if needed
        2. Compute edge magnitude using Sobel operator
        3. Search image for ROI with maximum average edge density
        4. Return ROI with confidence proportional to edge density
        
        Args:
            image: Input image (grayscale or color)
            roi_w: Desired ROI width
            roi_h: Desired ROI height
        
        Returns:
            ROIData object if high edge density found, None otherwise
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate edge density using Sobel operator
        # Sobel computes gradients in X and Y directions
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Edge magnitude = sqrt(gradient_x^2 + gradient_y^2)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Find region with highest edge density
        h, w = gray.shape
        max_density = 0
        best_roi = None
        
        # Search with step size to reduce computation
        step = 20  # Search step size (pixels)
        for y in range(0, h - roi_h, step):
            for x in range(0, w - roi_w, step):
                # Extract ROI edge region
                roi_edges = edge_magnitude[y:y+roi_h, x:x+roi_w]
                # Compute average edge density
                density = np.mean(roi_edges)
                
                # Track best ROI
                if density > max_density:
                    max_density = density
                    # Confidence scales with density (capped at 0.8)
                    confidence = min(0.8, density / 100.0)
                    best_roi = ROIData(x, y, roi_w, roi_h, confidence, 'saliency')
        
        return best_roi







