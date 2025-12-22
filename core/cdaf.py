"""
Contrast Detection Auto Focus (CDAF)
=====================================

This module implements Contrast Detection Auto Focus, an accurate autofocus
method that measures image sharpness by analyzing contrast.

Class:
- CDAFProcessor: Implements Laplacian variance method for contrast measurement

Key Concepts:
- Sharp images have higher contrast (more edges)
- Laplacian variance measures edge density
- Iterative search finds maximum contrast position
- Trend analysis determines focus direction

Learning Order:
Study this after pdaf.py to understand contrast-based focusing.
"""

import numpy as np
import cv2
from typing import Dict

from .config import AFConfig, ROIData


class CDAFProcessor:
    """
    Contrast Detection Auto Focus (CDAF) Processor.
    
    CDAF is an accurate autofocus method that measures image sharpness
    by analyzing contrast. It works by:
    1. Extracting region of interest (ROI) from the image
    2. Applying Laplacian filter to detect edges (sharp transitions)
    3. Computing variance of Laplacian as contrast metric
    4. Tracking contrast changes as lens position changes
    5. Finding the lens position with maximum contrast (sharpest image)
    
    How CDAF Works:
    - Sharp images have more edges (high-frequency content)
    - Laplacian operator detects edges by finding second derivatives
    - Variance of Laplacian measures edge density and strength
    - Higher variance = more edges = sharper image = better focus
    - The algorithm searches for the peak contrast position
    
    Advantages:
    - Very accurate (can achieve precise focus)
    - Works with any image content (no special pixels needed)
    - Reliable in various lighting conditions
    - Good for fine-tuning after initial positioning
    
    Limitations:
    - Slower than PDAF (requires multiple lens movements)
    - May "hunt" if contrast is low
    - Requires iterative search (hill-climbing)
    - Can be affected by noise in low-light
    
    Algorithm:
    - Laplacian variance method for contrast measurement
    - Trend analysis to determine focus direction
    - Adaptive step sizing based on contrast trends
    - History tracking to prevent oscillations
    
    Search Strategy:
    - Initial: Large steps to find approximate focus
    - Improving: Continue in same direction
    - Getting worse: Reverse direction with smaller steps
    - Oscillating: Reduce step size for fine-tuning
    
    Comparison with PDAF:
    - PDAF: Fast, hardware-assisted, provides direction+magnitude
    - CDAF: Slower, software-based, iterative search required
    - Hybrid: Use PDAF for speed, CDAF for accuracy
    """
    
    def __init__(self, config: AFConfig):
        """
        Initialize CDAF processor with configuration.
        
        Args:
            config: AFConfig object containing CDAF parameters
                - cdaf_step_size: Step size for lens movement (default: 20)
                - cdaf_max_steps: Maximum steps for search (default: 50)
                - cdaf_contrast_threshold: Minimum contrast change (default: 0.05)
        
        History Tracking:
        The processor maintains history of contrast and position values
        to analyze trends and determine optimal focus direction. History
        is reset when starting a new focus search.
        """
        self.config = config
        
        # History tracking for trend analysis
        # These lists store contrast and position values over time
        # Used to determine if focus is improving, getting worse, or oscillating
        self.contrast_history = []  # List of contrast values (float)
        self.position_history = []  # List of corresponding lens positions (float)
    
    def process_cdaf(self, image: np.ndarray, roi: ROIData, 
                    current_position: float, step_size: int) -> Dict:
        """
        Process CDAF to find optimal focus position.
        
        This method analyzes the current image contrast and determines
        the next lens movement direction and magnitude based on trend analysis.
        
        Args:
            image: Input image as numpy array
                   - Shape: (H, W) for grayscale or (H, W, 3) for color
                   - Data type: uint8, uint16, or float
                   - Used to extract ROI and calculate contrast
            
            roi: ROIData object specifying the region to analyze
                 Only this region is used for contrast calculation
            
            current_position: Current lens position (float)
                             - Typically in range 0-1023 (10-bit control)
                             - Used for history tracking and trend analysis
            
            step_size: Step size for next movement (int)
                      - Units: Focus steps
                      - Adjusted based on trends (may be reduced if oscillating)
        
        Returns:
            Dictionary containing:
            - 'suggested_movement': Recommended lens movement in steps (int)
              - Positive: Move lens forward
              - Negative: Move lens backward
              - Magnitude indicates step size
            - 'confidence': Confidence in the suggestion (float, 0.0-1.0)
              - Higher = more reliable trend
              - Lower during initial collection phase
            - 'contrast': Current contrast value (float)
              - Higher = sharper image (within same scene)
              - Values are relative, not absolute
            - 'trend': Trend analysis result (str)
              - 'collecting': Need more data (initial phase)
              - 'improving': Contrast increasing, continue direction
              - 'reversing': Contrast decreasing, reverse direction
              - 'oscillating': Alternating, reduce step size
        
        Algorithm Flow:
        1. Extract ROI from image
        2. Calculate contrast using Laplacian variance
        3. Store contrast and position in history
        4. Analyze recent history to determine trend
        5. Suggest movement direction and size based on trend
        
        Example:
            processor = CDAFProcessor(config)
            
            # First measurement (collecting data)
            result = processor.process_cdaf(image, roi, current_pos=500, step_size=20)
            # Result: suggested_movement=20, trend='collecting', confidence=0.3
            
            # After collecting more data, trend analysis kicks in
            result = processor.process_cdaf(image, roi, current_pos=520, step_size=20)
            # Result: trend='improving' or 'reversing' or 'oscillating'
        """
        # Extract ROI from image
        roi_image = image[roi.y:roi.y+roi.height, roi.x:roi.x+roi.width]
        
        # Calculate contrast using Laplacian variance
        contrast = self._calculate_contrast(roi_image)
        
        # Store history for trend analysis
        self.contrast_history.append(contrast)
        self.position_history.append(current_position)
        
        # Analyze trend to determine next step
        # Need at least 3 measurements to determine trend
        if len(self.contrast_history) < 3:
            # Need more data, continue in same direction
            return {
                'suggested_movement': step_size,
                'confidence': 0.3,
                'contrast': contrast,
                'trend': 'collecting'
            }
        
        # Get recent contrast measurements (last 3)
        recent_contrasts = self.contrast_history[-3:]
        
        # Determine trend: improving, getting worse, or oscillating
        if recent_contrasts[-1] > recent_contrasts[-2] > recent_contrasts[-3]:
            # Contrast is increasing (improving focus)
            # Continue in same direction with full step
            suggested_movement = step_size
            confidence = min(0.9, recent_contrasts[-1] / 100.0)  # Scale confidence
            trend = 'improving'
        elif recent_contrasts[-1] < recent_contrasts[-2] < recent_contrasts[-3]:
            # Contrast is decreasing (getting worse)
            # Reverse direction with half step (backtrack)
            suggested_movement = -step_size // 2
            confidence = 0.7
            trend = 'reversing'
        else:
            # Contrast is oscillating (near peak)
            # Reduce step size for fine-tuning
            suggested_movement = step_size // 2
            confidence = 0.5
            trend = 'oscillating'
        
        return {
            'suggested_movement': suggested_movement,
            'confidence': confidence,
            'contrast': contrast,
            'trend': trend
        }
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """
        Calculate image contrast using Laplacian variance method.
        
        This is the standard metric for measuring image sharpness in CDAF.
        The method works by:
        1. Applying Laplacian operator to detect edges
        2. Computing variance of the Laplacian response
        3. Higher variance = more edges = sharper image
        
        Laplacian Operator:
        The Laplacian is the second derivative operator:
        L = ∂²I/∂x² + ∂²I/∂y²
        
        It responds strongly to edges and sharp transitions, and weakly
        to smooth regions. A sharp image has many strong edges, resulting
        in high variance of the Laplacian response.
        
        Args:
            image: Input image as numpy array
                   - Can be grayscale (2D) or color (3D)
                   - If color, converted to grayscale first
                   - Data type: uint8, uint16, or float
                   - Typical shape: (height, width) or (height, width, 3)
        
        Returns:
            Contrast value as float
            - Higher values indicate sharper images
            - Typical range: thousands to hundreds of thousands
            - Depends on image content, resolution, and lighting
            - Values are relative, not absolute (compare within same scene)
        
        Formula:
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            contrast = np.var(laplacian)
        
        Why Variance?
        - Mean of Laplacian is near zero (equal positive/negative responses)
        - Variance captures the spread of edge strengths
        - Sharp images have large variance (many strong edges)
        - Blurry images have small variance (few weak edges)
        
        Mathematical Explanation:
        - Laplacian finds locations where intensity changes rapidly
        - In sharp images: many pixels with strong Laplacian response
        - In blurry images: Laplacian responses are weak and spread out
        - Variance measures the concentration of strong responses
        - Higher variance = more concentrated strong responses = sharper image
        
        Alternative Metrics:
        - Gradient magnitude variance
        - High-frequency energy (FFT-based)
        - Tenengrad (sum of squared gradients)
        - Brenner gradient
        
        Laplacian variance is preferred because:
        - Computationally efficient (single filter + variance)
        - Good discrimination between sharp/blurry
        - Robust to illumination changes
        - Standard in many autofocus systems
        """
        # Convert to grayscale if color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Laplacian filter
        # Laplacian detects edges by computing second derivatives
        # cv2.CV_64F: 64-bit float output (allows negative values)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Calculate variance as contrast measure
        # Variance captures the spread of edge strengths
        contrast = np.var(laplacian)
        
        return float(contrast)
    
    def reset_history(self):
        """
        Reset CDAF history for new focus search.
        
        This should be called when starting a new autofocus operation
        to clear previous history and start fresh trend analysis.
        
        Example:
            processor.reset_history()
            # Now ready for new focus search
        """
        self.contrast_history = []
        self.position_history = []







