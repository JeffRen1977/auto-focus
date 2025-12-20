"""
Phase Detection Auto Focus (PDAF)
==================================

This module implements Phase Detection Auto Focus, a fast hardware-assisted
autofocus method that uses dedicated phase detection pixels on the sensor.

Class:
- PDAFProcessor: Implements block matching algorithm for disparity calculation

Key Concepts:
- PDAF uses dedicated PD pixels arranged in left/right pattern
- Block Matching Algorithm (BMA) finds horizontal disparity
- Disparity indicates direction and magnitude of defocus
- Requires calibration to convert disparity to lens movement

Learning Order:
Study this after roi_selection.py to understand how phase detection works.
"""

import numpy as np
from typing import Tuple, Dict

from .config import AFConfig, ROIData


class PDAFProcessor:
    """
    Phase Detection Auto Focus (PDAF) Processor.
    
    PDAF is a fast, hardware-assisted autofocus method that uses dedicated
    phase detection pixels on the image sensor. These pixels are arranged
    in a pattern (typically 2x1 column blocks) that captures light from
    different angles, creating left and right views.
    
    How PDAF Works:
    1. Sensor has dedicated PD pixels arranged in left/right pattern
    2. Left and right PD images are extracted from the sensor
    3. Block Matching Algorithm (BMA) finds corresponding blocks
    4. Horizontal disparity (shift) between views is calculated
    5. Disparity is converted to defocus (lens movement needed)
    6. Lens is moved to correct focus position
    
    Advantages:
    - Very fast (typically < 200ms)
    - Hardware-assisted (dedicated pixels)
    - Works well in good lighting
    - Provides direction and magnitude of defocus
    
    Limitations:
    - Requires good contrast and texture
    - Limited by PD pixel coverage
    - May fail in low-light or low-contrast scenes
    - Needs calibration for accurate defocus conversion
    
    Algorithm:
    - Block Matching Algorithm (BMA) with Sum of Absolute Differences (SAD)
    - Normalized blocks for illumination invariance
    - Robust statistics (median + weighted average) for outlier rejection
    - Sub-pixel accuracy through integer search
    
    Calibration:
    The disparity-to-defocus conversion requires calibration:
    defocus = disparity * slope + offset
    
    These parameters are typically determined through:
    - Lens characterization tests
    - Known distance measurements
    - Factory calibration procedures
    """
    
    def __init__(self, config: AFConfig):
        """
        Initialize PDAF processor with configuration.
        
        Args:
            config: AFConfig object containing PDAF parameters
                - pdaf_block_size: Block size for matching (default: 8)
                - pdaf_search_range: Horizontal search range (default: 16)
                - pdaf_min_confidence: Minimum confidence threshold (default: 0.6)
        
        Calibration Parameters:
        The disparity_to_defocus_slope and offset convert pixel disparity
        to lens movement steps. In production, these would be loaded from
        factory calibration data specific to each lens-sensor combination.
        """
        self.config = config
        
        # Calibration parameters (would be loaded from device calibration)
        # These convert pixel disparity to lens movement steps
        # In production, these would come from factory calibration data
        self.disparity_to_defocus_slope = 25.0   # Conversion factor
        self.disparity_to_defocus_offset = -2.0   # Offset adjustment
        self.pdaf_confidence_threshold = config.pdaf_min_confidence
    
    def process_pdaf(self, left_pd: np.ndarray, right_pd: np.ndarray, 
                    roi: ROIData) -> Dict:
        """
        Process PDAF data to estimate defocus.
        
        This is the main entry point for PDAF processing. It extracts the ROI
        from left and right PD images, computes disparity using block matching,
        and converts the result to a defocus estimate.
        
        Args:
            left_pd: Left phase detection pixels as uint16 numpy array
                     Shape: (height, width)
                     Values: 0-65535 (16-bit sensor data)
            
            right_pd: Right phase detection pixels as uint16 numpy array
                      Shape: (height, width), same as left_pd
                      Values: 0-65535 (16-bit sensor data)
            
            roi: ROIData object specifying the region to process
                 Only this region is analyzed for focusing
        
        Returns:
            Dictionary containing:
            - 'disparity': Horizontal disparity in pixels (float)
              - Positive: Right image shifted right (subject behind focus)
              - Negative: Right image shifted left (subject in front of focus)
              - Zero: Images aligned (in focus)
            - 'confidence': Confidence score 0.0-1.0 (float)
              - Higher = more reliable estimate
              - Below pdaf_min_confidence = unreliable
            - 'defocus': Estimated defocus in lens movement steps (float)
              - Positive: Need to move lens forward
              - Negative: Need to move lens backward
            - 'valid_blocks': Number of valid block matches (int)
              - More blocks = more reliable estimate
        
        Example:
            processor = PDAFProcessor(config)
            result = processor.process_pdaf(left_pd, right_pd, roi)
            
            if result['confidence'] > 0.6:
                move_lens(result['defocus'])
        """
        # Extract ROI from PD images
        left_roi = left_pd[roi.y:roi.y+roi.height, roi.x:roi.x+roi.width]
        right_roi = right_pd[roi.y:roi.y+roi.height, roi.x:roi.x+roi.width]
        
        # Compute disparity using block matching algorithm
        disparity_result = self._compute_disparity_bma(left_roi, right_roi)
        
        # Convert disparity to defocus (lens movement steps)
        defocus = self._disparity_to_defocus(disparity_result['disparity'])
        
        return {
            'disparity': disparity_result['disparity'],
            'confidence': disparity_result['confidence'],
            'defocus': defocus,
            'valid_blocks': disparity_result['valid_blocks']
        }
    
    def _compute_disparity_bma(self, left_image: np.ndarray, 
                              right_image: np.ndarray) -> Dict:
        """
        Compute disparity using Block Matching Algorithm (BMA).
        
        This is the core PDAF algorithm that finds the horizontal shift
        between left and right PD images. The algorithm:
        1. Divides images into blocks (typically 8x8 pixels)
        2. For each block in left image, searches for best match in right image
        3. Computes horizontal disparity (shift) for each block
        4. Uses robust statistics to combine block disparities
        5. Filters outliers using median-based approach
        
        Args:
            left_image: Left PD image as uint16 numpy array
                       Shape: (height, width)
            right_image: Right PD image as uint16 numpy array
                        Shape: (height, width), same as left_image
                        Both should have the same dimensions
        
        Returns:
            Dictionary containing:
            - 'disparity': Final disparity estimate in pixels (float)
              - Positive: Right image shifted right (subject behind focus)
              - Negative: Right image shifted left (subject in front of focus)
              - Zero: Images aligned (in focus)
            - 'confidence': Confidence score 0.0-1.0 (float)
              - Based on block match quality and consistency
            - 'valid_blocks': Number of valid block matches (int)
              - Blocks with confidence > 0.3 are considered valid
        
        Algorithm Details:
            - Block size: Configurable via config.pdaf_block_size (default 8x8 pixels)
            - Search range: ±config.pdaf_search_range pixels horizontally
            - Matching metric: Sum of Absolute Differences (SAD)
            - Normalization: Blocks are normalized for illumination invariance
            - Outlier rejection: Uses median ± 2.0 pixels threshold
            - Final estimate: Weighted average of valid blocks (weighted by confidence)
        
        Performance:
            - Time complexity: O(blocks * search_range)
            - Typical: ~200ms for 756x2016 PD images with 8x8 blocks
            - Optimized by limiting search range and block count
        """
        block_size = self.config.pdaf_block_size
        search_range = self.config.pdaf_search_range
        
        h, w = left_image.shape
        n_blocks_y = h // block_size
        n_blocks_x = w // block_size
        
        disparities = []
        confidences = []
        
        # Process each block
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                y_start = by * block_size
                x_start = bx * block_size
                
                # Extract block from left image
                left_block = left_image[y_start:y_start+block_size, 
                                      x_start:x_start+block_size]
                
                # Search for best match in right image
                best_disp, confidence = self._match_block(
                    left_block, right_image, (y_start, x_start), search_range
                )
                
                # Keep blocks with sufficient confidence
                if confidence > 0.3:  # Basic threshold
                    disparities.append(best_disp)
                    confidences.append(confidence)
        
        # Handle case where no valid blocks found
        if len(disparities) == 0:
            return {'disparity': 0.0, 'confidence': 0.0, 'valid_blocks': 0}
        
        # Robust estimation using median and weighted average
        disparities = np.array(disparities)
        confidences = np.array(confidences)
        
        # Remove outliers (blocks with disparity far from median)
        median_disp = np.median(disparities)
        valid_mask = np.abs(disparities - median_disp) < 2.0
        
        if valid_mask.sum() == 0:
            # If all blocks are outliers, use median anyway
            final_disparity = median_disp
            final_confidence = np.mean(confidences)
        else:
            # Weighted average of valid blocks (weighted by confidence)
            weights = confidences[valid_mask]
            final_disparity = np.average(disparities[valid_mask], weights=weights)
            final_confidence = np.mean(confidences[valid_mask])
        
        return {
            'disparity': float(final_disparity),
            'confidence': float(final_confidence),
            'valid_blocks': int(valid_mask.sum())
        }
    
    def _match_block(self, left_block: np.ndarray, right_image: np.ndarray,
                    position: Tuple[int, int], search_range: int) -> Tuple[float, float]:
        """
        Match a single block between left and right images using SAD.
        
        This function searches for the best matching position of a left block
        in the right image by computing Sum of Absolute Differences (SAD)
        at each candidate position. The position with minimum SAD is the best match.
        
        Args:
            left_block: Block from left image, shape (block_h, block_w)
                       Values: uint16 sensor data
            right_image: Right image to search in, shape (height, width)
                        Values: uint16 sensor data
            position: (y, x) position of the block in left image
            search_range: Maximum horizontal search range in pixels
                         Typical: 8-32 pixels
        
        Returns:
            Tuple of (best_disparity, confidence):
            - best_disparity: Horizontal shift in pixels (float)
              - Positive: Right block is to the right (subject behind focus)
              - Negative: Right block is to the left (subject in front of focus)
              - Zero: Blocks aligned (in focus)
            - confidence: Match quality 0.0-1.0 (float)
              - Higher confidence = better match quality
              - Based on SAD cost relative to maximum possible cost
              - Typical good matches: > 0.5
        
        Algorithm:
            1. Normalize left block (zero mean, unit variance) for illumination invariance
            2. For each candidate disparity d in [-search_range, search_range]:
               a. Extract corresponding block from right image at (x + d, y)
               b. Normalize right block
               c. Compute SAD = sum(|left_block - right_block|)
               d. Track minimum SAD and corresponding disparity
            3. Calculate confidence from SAD value
            4. Return best disparity and confidence
        
        Normalization:
            Blocks are normalized to (mean=0, std=1) to make matching
            illumination-invariant. This is critical for robust matching
            under varying lighting conditions. Without normalization,
            brightness differences would dominate the SAD cost.
        
        Confidence Calculation:
            confidence = 1.0 - (best_sad / (block_size * 255.0))
            - Lower SAD = higher confidence
            - Normalized by maximum possible SAD (all pixels differ by 255)
            - Clamped to [0.0, 1.0]
        """
        y_pos, x_pos = position
        block_h, block_w = left_block.shape
        
        # Normalize left block (illumination invariant)
        left_block = left_block.astype(np.float32)
        left_mean = left_block.mean()
        left_std = left_block.std()
        if left_std > 1e-6:  # Avoid division by zero
            left_block = (left_block - left_mean) / left_std
        else:
            left_block = left_block - left_mean  # Uniform block
        
        best_cost = float('inf')
        best_disparity = 0.0
        
        # Search in horizontal direction only (PDAF is horizontal disparity)
        for d in range(-search_range, search_range + 1):
            x_search = x_pos + d
            
            # Boundary check
            if x_search < 0 or x_search + block_w > right_image.shape[1]:
                continue
            
            # Extract right block at candidate position
            right_block = right_image[y_pos:y_pos+block_h, x_search:x_search+block_w]
            right_block = right_block.astype(np.float32)
            
            # Normalize right block (same normalization as left)
            right_mean = right_block.mean()
            right_std = right_block.std()
            if right_std > 1e-6:
                right_block = (right_block - right_mean) / right_std
            else:
                right_block = right_block - right_mean
            
            # Compute Sum of Absolute Differences (SAD)
            sad = np.sum(np.abs(left_block - right_block))
            
            # Track best match (minimum SAD)
            if sad < best_cost:
                best_cost = sad
                best_disparity = float(d)
        
        # Calculate confidence based on cost curve sharpness
        # Normalize by maximum possible SAD (all pixels differ by max value)
        max_possible_sad = block_h * block_w * 2.0  # Normalized difference range
        confidence = max(0.0, min(1.0, 1.0 - (best_cost / max_possible_sad)))
        
        return best_disparity, confidence
    
    def _disparity_to_defocus(self, disparity: float) -> float:
        """
        Convert disparity to defocus (lens movement steps).
        
        This conversion requires calibration parameters specific to the
        lens-sensor combination. The relationship is typically linear:
        defocus = disparity * slope + offset
        
        Args:
            disparity: Horizontal disparity in pixels (float)
                      Positive = subject behind focus
                      Negative = subject in front of focus
        
        Returns:
            Defocus in lens movement steps (float)
            Positive = move lens forward
            Negative = move lens backward
        
        Calibration:
        The slope and offset parameters are determined through:
        - Lens characterization tests at known distances
        - Factory calibration procedures
        - Per-unit calibration during manufacturing
        
        These parameters are typically stored in device-specific calibration
        data and loaded at initialization.
        """
        return disparity * self.disparity_to_defocus_slope + self.disparity_to_defocus_offset

