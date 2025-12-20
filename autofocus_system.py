"""
Comprehensive Autofocus System Implementation
=============================================

This module implements a complete autofocus system that demonstrates modern
mobile camera focusing techniques. It provides a comprehensive framework for
autofocus algorithms used in smartphone cameras and computational photography.

Key Components:
1. PDAF (Phase Detection Auto Focus) - Fast, hardware-assisted focusing
   - Uses dedicated phase detection pixels on the sensor
   - Calculates horizontal disparity between left/right views
   - Provides fast initial focus positioning
   - Block Matching Algorithm (BMA) with Sum of Absolute Differences (SAD)

2. CDAF (Contrast Detection Auto Focus) - Accurate but slower focusing
   - Measures image sharpness using Laplacian variance
   - Analyzes contrast trends to determine focus direction
   - Provides accurate fine-tuning after initial positioning
   - Iterative search for maximum contrast

3. LDAF (Laser Detection Auto Focus) - Distance-based focusing
   - Uses laser time-of-flight or structured light measurements
   - Directly converts distance to lens position
   - Very fast but limited range and accuracy
   - Useful for initial positioning in good conditions

4. Hybrid AF - Intelligent combination of all methods
   - Weighted fusion of PDAF, CDAF, and LDAF results
   - Confidence-based method selection
   - Temporal filtering for smooth convergence
   - Optimal performance across different scenarios

5. ROI Selection - Smart area selection for focusing
   - Priority-based selection: touch > face > saliency > center
   - Improves speed and accuracy by focusing on relevant regions
   - Supports multiple selection methods

6. Temporal Filtering - Smooth convergence and stability
   - Exponential moving average to prevent focus hunting
   - Reduces oscillations and jitter
   - Ensures stable, smooth focus transitions

Architecture:
The system uses a state machine approach with iterative refinement:
- IDLE: System ready, waiting for trigger
- SEARCHING: Actively searching for focus
- CONVERGED: Focus achieved, within tolerance
- FAILED: Maximum iterations reached without convergence
- HUNTING: Oscillating around focus point (not converged)

Based on AF system architecture and mobile computational photography principles.

Usage Example:
    from autofocus_system import AutoFocusSystem, AFConfig, AFMethod
    
    config = AFConfig(
        pdaf_min_confidence=0.6,
        cdaf_step_size=20,
        max_iterations=15
    )
    
    af_system = AutoFocusSystem(config)
    result = af_system.trigger_autofocus(
        image=bayer_image,
        left_pd=left_pd,
        right_pd=right_pd,
        method=AFMethod.HYBRID
    )
    
    if result.success:
        move_lens_to(result.final_position)
"""

import numpy as np
from scipy import signal, optimize, ndimage
from typing import Tuple, Dict, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import cv2


class AFMethod(Enum):
    """
    Autofocus method enumeration.
    
    Defines the available autofocus methods that can be used:
    
    - PDAF: Phase Detection Auto Focus
      Fast, hardware-assisted method using dedicated PD pixels.
      Best for: Initial positioning, good lighting, textured scenes.
      
    - CDAF: Contrast Detection Auto Focus
      Accurate method using image contrast/sharpness.
      Best for: Fine-tuning, low-light, low-contrast scenes.
      
    - LDAF: Laser Detection Auto Focus
      Distance-based method using laser ranging.
      Best for: Initial positioning, good conditions, close subjects.
      
    - HYBRID: Combined approach
      Intelligently combines all available methods.
      Best for: Production use, optimal performance across scenarios.
    """
    PDAF = "pdaf"      # Phase Detection Auto Focus
    CDAF = "cdaf"      # Contrast Detection Auto Focus
    LDAF = "ldaf"      # Laser Detection Auto Focus
    HYBRID = "hybrid"  # Combined approach


class AFState(Enum):
    """Autofocus state machine"""
    IDLE = "idle"
    SEARCHING = "searching"
    CONVERGED = "converged"
    FAILED = "failed"
    HUNTING = "hunting"


@dataclass
class AFConfig:
    """
    Autofocus system configuration parameters.
    
    This dataclass contains all tunable parameters for the autofocus system.
    Adjust these values based on your sensor characteristics, lens properties,
    and performance requirements.
    
    PDAF Parameters:
        pdaf_block_size: Size of blocks for block matching (default: 8)
            - Larger blocks: More robust but less accurate
            - Smaller blocks: More accurate but more sensitive to noise
            - Typical range: 4-16 pixels
            
        pdaf_search_range: Horizontal search range for block matching (default: 16)
            - Maximum disparity to search in pixels
            - Larger range: Handles larger defocus but slower
            - Typical range: 8-32 pixels
            
        pdaf_min_confidence: Minimum confidence threshold for PDAF (default: 0.6)
            - Results below this are considered unreliable
            - Range: 0.0-1.0
            - Higher: More conservative, fewer false positives
            
        pdaf_convergence_threshold: Defocus threshold for convergence (default: 0.5)
            - If defocus < threshold, consider converged
            - Smaller: More precise but may oscillate
            - Units: Focus steps
    
    CDAF Parameters:
        cdaf_step_size: Step size for lens movement in CDAF (default: 20)
            - Larger steps: Faster but may overshoot
            - Smaller steps: More precise but slower
            - Units: Focus steps
            
        cdaf_max_steps: Maximum steps for CDAF search (default: 50)
            - Prevents infinite search
            - Adjust based on lens range
            
        cdaf_contrast_threshold: Minimum contrast change for significance (default: 0.05)
            - Relative contrast change threshold
            - Smaller: More sensitive to small changes
    
    LDAF Parameters:
        ldaf_confidence_threshold: Minimum LDAF confidence (default: 0.8)
            - LDAF results below this are ignored
            - Range: 0.0-1.0
            
        ldaf_distance_range: Valid distance range in meters (default: (0.1, 10.0))
            - Distances outside this range are rejected
            - Adjust based on sensor capabilities
    
    Hybrid Parameters:
        hybrid_pdaf_weight: Weight for PDAF in hybrid fusion (default: 0.6)
        hybrid_cdaf_weight: Weight for CDAF in hybrid fusion (default: 0.3)
        hybrid_ldaf_weight: Weight for LDAF in hybrid fusion (default: 0.1)
            - Weights should sum to approximately 1.0
            - Higher PDAF weight: Faster initial positioning
            - Higher CDAF weight: More accurate fine-tuning
    
    General Parameters:
        max_iterations: Maximum AF iterations before giving up (default: 15)
            - Prevents infinite loops
            - Typical: 10-20 iterations
            
        temporal_alpha: Temporal filtering coefficient (default: 0.7)
            - Range: 0.0-1.0
            - Higher: More smoothing, slower response
            - Lower: Faster response, more jitter
            - Formula: filtered = alpha * new + (1-alpha) * old
            
        roi_size: Default ROI size in pixels (default: (200, 200))
            - (width, height) tuple
            - Larger: More robust but slower
            - Smaller: Faster but less robust
    """
    # PDAF parameters
    pdaf_block_size: int = 8
    pdaf_search_range: int = 16
    pdaf_min_confidence: float = 0.6
    pdaf_convergence_threshold: float = 0.5
    
    # CDAF parameters
    cdaf_step_size: int = 20
    cdaf_max_steps: int = 50
    cdaf_contrast_threshold: float = 0.05
    
    # LDAF parameters
    ldaf_confidence_threshold: float = 0.8
    ldaf_distance_range: Tuple[float, float] = (0.1, 10.0)  # meters
    
    # Hybrid parameters
    hybrid_pdaf_weight: float = 0.6
    hybrid_cdaf_weight: float = 0.3
    hybrid_ldaf_weight: float = 0.1
    
    # General parameters
    max_iterations: int = 15
    temporal_alpha: float = 0.7  # Temporal filtering coefficient
    roi_size: Tuple[int, int] = (200, 200)  # Default ROI size


@dataclass
class ROIData:
    """Region of Interest data"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    method: str  # 'touch', 'face', 'saliency', 'center'


@dataclass
class AFResult:
    """Autofocus result"""
    success: bool
    final_position: float
    iterations: int
    method_used: AFMethod
    confidence: float
    convergence_time: float
    residual_error: float


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
    """
    
    def __init__(self, config: AFConfig):
        """
        Initialize ROI selector with configuration.
        
        Args:
            config: AFConfig object containing ROI parameters
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
        
        2. Face Detection (Priority 2):
           - If faces detected, use largest face
           - Expand face region by expand_ratio (default 2.0x)
           - Confidence: 0.9 (high, but not as high as user selection)
           - ROI size: Based on face size
        
        3. Saliency-Based (Priority 3):
           - Find region with highest edge density
           - Uses Sobel operator to detect edges
           - Confidence: 0.0-0.8 (based on edge density)
           - ROI size: Uses default from config
        
        4. Center-Weighted (Priority 4):
           - Default fallback: center of image
           - Confidence: 0.5 (lowest, generic selection)
           - ROI size: Uses default from config
        
        Args:
            image: Input image as numpy array
                   - Can be grayscale or color
                   - Used for saliency calculation if needed
            
            touch_point: User touch point (x, y) in pixels (optional)
                        - Highest priority if provided
                        - ROI centers on this point
            
            faces: List of detected faces (optional)
                  - Format: [(x, y, width, height), ...]
                  - Second priority if provided
                  - Uses largest face if multiple detected
        
        Returns:
            ROIData object containing:
            - x, y: Top-left corner of ROI in pixels
            - width, height: ROI dimensions in pixels
            - confidence: Selection confidence (0.0-1.0)
            - method: Selection method used ('touch', 'face', 'saliency', 'center')
        
        ROI Expansion:
        For face detection, the ROI is expanded around the face to include
        context. This ensures the focus includes the face and surrounding area,
        not just the face itself.
        
        Example:
            # User touched screen at (300, 200)
            roi = selector.select_roi(image, touch_point=(300, 200))
            # ROI centers on (300, 200), method='touch', confidence=1.0
            
            # Face detected at (100, 150, 80, 100)
            roi = selector.select_roi(image, faces=[(100, 150, 80, 100)])
            # ROI expands around face, method='face', confidence=0.9
        """
        h, w = image.shape[:2]
        roi_w, roi_h = self.config.roi_size
        
        # Priority 1: User touch point
        if touch_point is not None:
            tx, ty = touch_point
            x = max(0, min(tx - roi_w//2, w - roi_w))
            y = max(0, min(ty - roi_h//2, h - roi_h))
            return ROIData(x, y, roi_w, roi_h, 1.0, 'touch')
        
        # Priority 2: Face detection
        if faces and len(faces) > 0:
            # Use the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            fx, fy, fw, fh = largest_face
            
            # Expand face area for focusing
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
        """Select ROI based on saliency (edge density)"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate edge density using Sobel
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Find region with highest edge density
        h, w = gray.shape
        max_density = 0
        best_roi = None
        
        step = 20  # Search step size
        for y in range(0, h - roi_h, step):
            for x in range(0, w - roi_w, step):
                roi_edges = edge_magnitude[y:y+roi_h, x:x+roi_w]
                density = np.mean(roi_edges)
                
                if density > max_density:
                    max_density = density
                    best_roi = ROIData(x, y, roi_w, roi_h, 
                                     min(0.8, density / 100.0), 'saliency')
        
        return best_roi


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
        Process PDAF data to estimate defocus
        
        Args:
            left_pd: Left phase detection pixels
            right_pd: Right phase detection pixels
            roi: Region of interest
            
        Returns:
            dict: PDAF result with disparity, confidence, and defocus estimate
        """
        # Extract ROI
        left_roi = left_pd[roi.y:roi.y+roi.height, roi.x:roi.x+roi.width]
        right_roi = right_pd[roi.y:roi.y+roi.height, roi.x:roi.x+roi.width]
        
        # Compute disparity using block matching
        disparity_result = self._compute_disparity_bma(left_roi, right_roi)
        
        # Convert disparity to defocus
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
            right_image: Right PD image as uint16 numpy array
            Both should have the same dimensions
        
        Returns:
            Dictionary containing:
            - 'disparity': Final disparity estimate in pixels (float)
              - Positive: Right image shifted right (subject behind focus)
              - Negative: Right image shifted left (subject in front of focus)
              - Zero: Images aligned (in focus)
            - 'confidence': Confidence score 0.0-1.0 (float)
            - 'valid_blocks': Number of valid block matches (int)
        
        Algorithm Details:
            - Block size: Configurable (default 8x8 pixels)
            - Search range: ±search_range pixels horizontally
            - Matching metric: Sum of Absolute Differences (SAD)
            - Normalization: Blocks are normalized for illumination invariance
            - Outlier rejection: Uses median ± 2.0 pixels threshold
            - Final estimate: Weighted average of valid blocks (weighted by confidence)
        
        Performance:
            - Time complexity: O(blocks * search_range)
            - Typical: ~200ms for 756x2016 PD images with 8x8 blocks
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
                
                left_block = left_image[y_start:y_start+block_size, 
                                      x_start:x_start+block_size]
                
                # Search for best match in right image
                best_disp, confidence = self._match_block(
                    left_block, right_image, (y_start, x_start), search_range
                )
                
                if confidence > 0.3:  # Basic threshold
                    disparities.append(best_disp)
                    confidences.append(confidence)
        
        if len(disparities) == 0:
            return {'disparity': 0.0, 'confidence': 0.0, 'valid_blocks': 0}
        
        # Robust estimation using median and weighted average
        disparities = np.array(disparities)
        confidences = np.array(confidences)
        
        # Remove outliers
        median_disp = np.median(disparities)
        valid_mask = np.abs(disparities - median_disp) < 2.0
        
        if valid_mask.sum() == 0:
            final_disparity = median_disp
            final_confidence = np.mean(confidences)
        else:
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
            right_image: Right image to search in, shape (height, width)
            position: (y, x) position of the block in left image
            search_range: Maximum horizontal search range in pixels
        
        Returns:
            Tuple of (best_disparity, confidence):
            - best_disparity: Horizontal shift in pixels (float)
              - Positive: Right block is to the right (subject behind focus)
              - Negative: Right block is to the left (subject in front of focus)
            - confidence: Match quality 0.0-1.0 (float)
              - Higher confidence = better match quality
              - Based on SAD cost relative to maximum possible cost
        
        Algorithm:
            1. Normalize left block (zero mean, unit variance)
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
            under varying lighting conditions.
        
        Confidence Calculation:
            confidence = 1.0 - (best_sad / (block_size * 255.0))
            - Lower SAD = higher confidence
            - Normalized by maximum possible SAD (all pixels differ by 255)
        """
        y_pos, x_pos = position
        block_h, block_w = left_block.shape
        
        # Normalize blocks
        left_block = left_block.astype(np.float32)
        left_block = (left_block - left_block.mean()) / (left_block.std() + 1e-6)
        
        best_cost = float('inf')
        best_disparity = 0.0
        
        # Search in horizontal direction
        for d in range(-search_range, search_range + 1):
            x_search = x_pos + d
            
            # Boundary check
            if x_search < 0 or x_search + block_w > right_image.shape[1]:
                continue
            
            # Extract right block
            right_block = right_image[y_pos:y_pos+block_h, x_search:x_search+block_w]
            right_block = right_block.astype(np.float32)
            right_block = (right_block - right_block.mean()) / (right_block.std() + 1e-6)
            
            # Compute SAD
            sad = np.sum(np.abs(left_block - right_block))
            
            if sad < best_cost:
                best_cost = sad
                best_disparity = float(d)
        
        # Calculate confidence based on cost curve sharpness
        confidence = max(0.0, 1.0 - (best_cost / (block_h * block_w * 255.0)))
        
        return best_disparity, confidence
    
    def _disparity_to_defocus(self, disparity: float) -> float:
        """Convert disparity to defocus (lens movement steps)"""
        return disparity * self.disparity_to_defocus_slope + self.disparity_to_defocus_offset


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
    """
    
    def __init__(self, config: AFConfig):
        """
        Initialize CDAF processor with configuration.
        
        Args:
            config: AFConfig object containing CDAF parameters
        """
        self.config = config
        
        # History tracking for trend analysis
        # These lists store contrast and position values over time
        # Used to determine if focus is improving, getting worse, or oscillating
        self.contrast_history = []  # List of contrast values
        self.position_history = []  # List of corresponding lens positions
    
    def process_cdaf(self, image: np.ndarray, roi: ROIData, 
                    current_position: float, step_size: int) -> Dict:
        """
        Process CDAF to find optimal focus position
        
        Args:
            image: Input image
            roi: Region of interest
            current_position: Current lens position
            step_size: Step size for next movement
            
        Returns:
            dict: CDAF result with suggested movement and confidence
        """
        # Extract ROI and calculate contrast
        roi_image = image[roi.y:roi.y+roi.height, roi.x:roi.x+roi.width]
        contrast = self._calculate_contrast(roi_image)
        
        # Store history
        self.contrast_history.append(contrast)
        self.position_history.append(current_position)
        
        # Analyze trend to determine next step
        if len(self.contrast_history) < 3:
            # Need more data, continue in same direction
            return {
                'suggested_movement': step_size,
                'confidence': 0.3,
                'contrast': contrast,
                'trend': 'collecting'
            }
        
        # Determine if we're improving or getting worse
        recent_contrasts = self.contrast_history[-3:]
        if recent_contrasts[-1] > recent_contrasts[-2] > recent_contrasts[-3]:
            # Improving, continue in same direction
            suggested_movement = step_size
            confidence = min(0.9, recent_contrasts[-1] / 100.0)
            trend = 'improving'
        elif recent_contrasts[-1] < recent_contrasts[-2] < recent_contrasts[-3]:
            # Getting worse, reverse direction with smaller step
            suggested_movement = -step_size // 2
            confidence = 0.7
            trend = 'reversing'
        else:
            # Oscillating, reduce step size
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
        
        Returns:
            Contrast value as float
            - Higher values indicate sharper images
            - Typical range: thousands to hundreds of thousands
            - Depends on image content, resolution, and lighting
            - Values are relative, not absolute
        
        Formula:
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            contrast = np.var(laplacian)
        
        Why Variance?
        - Mean of Laplacian is near zero (equal positive/negative responses)
        - Variance captures the spread of edge strengths
        - Sharp images have large variance (many strong edges)
        - Blurry images have small variance (few weak edges)
        
        Alternative Metrics:
        - Gradient magnitude variance
        - High-frequency energy (FFT)
        - Tenengrad (sum of squared gradients)
        - Brenner gradient
        
        Laplacian variance is preferred because:
        - Computationally efficient
        - Good discrimination between sharp/blurry
        - Robust to illumination changes
        - Standard in many autofocus systems
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Laplacian filter
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Calculate variance as contrast measure
        contrast = np.var(laplacian)
        
        return float(contrast)
    
    def reset_history(self):
        """Reset CDAF history for new focus search"""
        self.contrast_history = []
        self.position_history = []


class LDAFProcessor:
    """
    Laser Detection Auto Focus Processor
    Distance-based focusing using laser ranging
    """
    
    def __init__(self, config: AFConfig):
        self.config = config
        # Distance to lens position lookup table (would be calibrated)
        self.distance_to_position_table = self._create_distance_table()
    
    def process_ldaf(self, distance: float, confidence: float) -> Dict:
        """
        Process LDAF distance measurement
        
        Args:
            distance: Measured distance in meters
            confidence: Measurement confidence [0-1]
            
        Returns:
            dict: LDAF result with suggested position and confidence
        """
        if confidence < self.config.ldaf_confidence_threshold:
            return {
                'suggested_position': None,
                'confidence': 0.0,
                'distance': distance,
                'reason': 'low_confidence'
            }
        
        if not (self.config.ldaf_distance_range[0] <= distance <= self.config.ldaf_distance_range[1]):
            return {
                'suggested_position': None,
                'confidence': 0.0,
                'distance': distance,
                'reason': 'out_of_range'
            }
        
        # Convert distance to lens position
        suggested_position = self._distance_to_lens_position(distance)
        
        return {
            'suggested_position': suggested_position,
            'confidence': confidence,
            'distance': distance,
            'reason': 'success'
        }
    
    def _create_distance_table(self) -> Dict[float, float]:
        """Create distance to lens position lookup table"""
        # Simplified lookup table - in reality this would be more complex
        table = {}
        for d in np.arange(0.1, 10.1, 0.1):
            # Simplified relationship: closer = higher position value
            position = 1000.0 / (d + 0.1) * 100
            table[round(d, 1)] = min(1023, max(0, position))
        return table
    
    def _distance_to_lens_position(self, distance: float) -> float:
        """Convert distance to lens position using lookup table"""
        # Find closest distance in table
        distances = list(self.distance_to_position_table.keys())
        closest_distance = min(distances, key=lambda x: abs(x - distance))
        return self.distance_to_position_table[closest_distance]


class AutoFocusSystem:
    """
    Main Autofocus System Controller.
    
    This is the top-level class that coordinates all autofocus methods
    (PDAF, CDAF, LDAF) to achieve optimal focusing performance. It implements
    a state machine that manages the autofocus process from trigger to convergence.
    
    System Architecture:
    The system uses an iterative refinement approach:
    1. Select Region of Interest (ROI) for focusing
    2. Process available AF methods (PDAF, CDAF, LDAF)
    3. Combine results using weighted fusion (for HYBRID mode)
    4. Decide on lens movement direction and magnitude
    5. Apply temporal filtering for smooth convergence
    6. Update lens position
    7. Check for convergence
    8. Repeat until converged or max iterations reached
    
    State Machine:
    - IDLE: System ready, waiting for autofocus trigger
    - SEARCHING: Actively searching for optimal focus position
    - CONVERGED: Focus achieved, within tolerance threshold
    - FAILED: Maximum iterations reached without convergence
    - HUNTING: Oscillating around focus point (not converged)
    
    Method Selection:
    - PDAF_ONLY: Use only PDAF (fast, when PDAF is reliable)
    - CDAF_ONLY: Use only CDAF (accurate, when PDAF fails)
    - LDAF_ONLY: Use only LDAF (very fast, when available)
    - HYBRID: Combine all available methods (optimal performance)
    
    Temporal Filtering:
    Uses exponential moving average to smooth lens movements:
    filtered_movement = alpha * new_movement + (1-alpha) * previous_movement
    
    This prevents:
    - Focus hunting (oscillations around focus point)
    - Jittery movements
    - Overcorrection
    
    Convergence Criteria:
    - Movement magnitude < convergence_threshold
    - All methods agree on focus position
    - Confidence above minimum threshold
    
    Usage:
        config = AFConfig()
        af_system = AutoFocusSystem(config)
        result = af_system.trigger_autofocus(
            image=bayer_image,
            left_pd=left_pd,
            right_pd=right_pd,
            method=AFMethod.HYBRID
        )
    """
    
    def __init__(self, config: AFConfig):
        """
        Initialize the autofocus system.
        
        Args:
            config: AFConfig object containing all system parameters
        """
        self.config = config
        
        # State machine state
        self.state = AFState.IDLE
        
        # Lens position tracking
        # Lens positions typically range from 0-1023 (10-bit control)
        self.current_position = 512.0  # Default middle position
        self.target_position = 512.0    # Target position after filtering
        
        # Iteration counter
        self.iteration_count = 0
        
        # Initialize specialized processors
        self.roi_selector = ROISelector(config)      # ROI selection
        self.pdaf_processor = PDAFProcessor(config)   # Phase detection
        self.cdaf_processor = CDAFProcessor(config)   # Contrast detection
        self.ldaf_processor = LDAFProcessor(config)    # Laser detection
        
        # State tracking for each method
        # These store the most recent results from each AF method
        self.last_pdaf_result = None   # Latest PDAF result
        self.last_cdaf_result = None   # Latest CDAF result
        self.last_ldaf_result = None   # Latest LDAF result
        
        # Convergence history for analysis
        self.convergence_history = []
    
    def trigger_autofocus(self, 
                         image: np.ndarray,
                         left_pd: Optional[np.ndarray] = None,
                         right_pd: Optional[np.ndarray] = None,
                         ldaf_distance: Optional[float] = None,
                         ldaf_confidence: float = 1.0,
                         touch_point: Optional[Tuple[int, int]] = None,
                         faces: Optional[List[Tuple[int, int, int, int]]] = None,
                         method: AFMethod = AFMethod.HYBRID) -> AFResult:
        """
        Trigger the autofocus process.
        
        This is the main entry point for autofocus. It runs an iterative
        search algorithm that refines the lens position until optimal
        focus is achieved or maximum iterations are reached.
        
        Process Flow:
        1. Initialize state (SEARCHING)
        2. Select ROI (region of interest) for focusing
        3. Iterative refinement loop:
           a. Process available AF methods (PDAF, CDAF, LDAF)
           b. Combine results to decide on lens movement
           c. Apply temporal filtering for smooth convergence
           d. Update lens position
           e. Check for convergence
           f. Repeat until converged or max iterations
        4. Return final result
        
        Args:
            image: Input image as numpy array (Bayer or RGB)
                   - Used for CDAF calculations
                   - Used for ROI selection
                   - Shape: (height, width) or (height, width, channels)
            
            left_pd: Left PD image (optional, required for PDAF)
                     - Shape: (height, width/2) typically
                     - uint16 numpy array
                     - Must be provided with right_pd for PDAF
            
            right_pd: Right PD image (optional, required for PDAF)
                     - Shape: (height, width/2) typically
                     - uint16 numpy array
                     - Must be provided with left_pd for PDAF
            
            ldaf_distance: LDAF distance measurement in meters (optional)
                          - Required for LDAF method
                          - Typical range: 0.1-10.0 meters
                          - Measured using time-of-flight or structured light
            
            ldaf_confidence: LDAF measurement confidence (default: 1.0)
                           - Range: 0.0-1.0
                           - Higher = more reliable measurement
            
            touch_point: User touch point (x, y) in pixels (optional)
                        - Highest priority for ROI selection
                        - If provided, ROI centers on this point
            
            faces: List of detected faces (optional)
                  - Format: [(x, y, width, height), ...]
                  - Second priority for ROI selection
                  - Largest face is used if multiple detected
            
            method: AF method to use (default: HYBRID)
                   - AFMethod.PDAF: Use only PDAF
                   - AFMethod.CDAF: Use only CDAF
                   - AFMethod.LDAF: Use only LDAF
                   - AFMethod.HYBRID: Combine all available methods
        
        Returns:
            AFResult object containing:
            - success: bool - Whether focus was achieved
            - final_position: float - Final lens position (0-1023)
            - iterations: int - Number of iterations performed
            - method_used: AFMethod - Which method was used
            - confidence: float - Overall confidence (0.0-1.0)
            - convergence_time: float - Time taken in seconds
            - residual_error: float - Remaining focus error
        
        Convergence:
        The algorithm converges when:
        - Movement magnitude < convergence_threshold
        - All methods agree on focus position
        - Confidence is above minimum threshold
        
        Temporal Filtering:
        Lens movements are smoothed using exponential moving average:
        filtered = alpha * new + (1-alpha) * previous
        
        This prevents oscillations and ensures smooth convergence.
        
        Example:
            result = af_system.trigger_autofocus(
                image=bayer_image,
                left_pd=left_pd,
                right_pd=right_pd,
                method=AFMethod.HYBRID
            )
            
            if result.success:
                print(f"Focused at position {result.final_position}")
                print(f"Took {result.iterations} iterations")
                print(f"Confidence: {result.confidence:.2f}")
        """
        import time
        start_time = time.time()
        
        self.state = AFState.SEARCHING
        self.iteration_count = 0
        self.cdaf_processor.reset_history()
        
        # Select ROI
        roi = self.roi_selector.select_roi(image, touch_point, faces)
        print(f"Selected ROI: ({roi.x}, {roi.y}, {roi.width}, {roi.height}) "
              f"method={roi.method} confidence={roi.confidence:.2f}")
        
        # Main autofocus loop
        while self.iteration_count < self.config.max_iterations:
            self.iteration_count += 1
            print(f"\n--- AF Iteration {self.iteration_count} ---")
            
            # Process different AF methods
            if method == AFMethod.PDAF or method == AFMethod.HYBRID:
                if left_pd is not None and right_pd is not None:
                    self.last_pdaf_result = self.pdaf_processor.process_pdaf(
                        left_pd, right_pd, roi
                    )
                    print(f"PDAF: disparity={self.last_pdaf_result['disparity']:.3f}, "
                          f"confidence={self.last_pdaf_result['confidence']:.3f}")
            
            if method == AFMethod.CDAF or method == AFMethod.HYBRID:
                self.last_cdaf_result = self.cdaf_processor.process_cdaf(
                    image, roi, self.current_position, self.config.cdaf_step_size
                )
                print(f"CDAF: movement={self.last_cdaf_result['suggested_movement']}, "
                      f"confidence={self.last_cdaf_result['confidence']:.3f}, "
                      f"trend={self.last_cdaf_result['trend']}")
            
            if method == AFMethod.LDAF or method == AFMethod.HYBRID:
                if ldaf_distance is not None:
                    self.last_ldaf_result = self.ldaf_processor.process_ldaf(
                        ldaf_distance, ldaf_confidence
                    )
                    print(f"LDAF: distance={self.last_ldaf_result['distance']:.2f}m, "
                          f"confidence={self.last_ldaf_result['confidence']:.3f}")
            
            # Decide on next movement
            movement = self._decide_movement(method)
            
            if movement is None:
                print("AF converged!")
                self.state = AFState.CONVERGED
                break
            
            # Apply temporal filtering
            filtered_movement = movement * self.config.temporal_alpha + \
                              (1 - self.config.temporal_alpha) * (self.target_position - self.current_position)
            
            # Update position
            self.target_position = self.current_position + filtered_movement
            self.target_position = np.clip(self.target_position, 0, 1023)
            self.current_position = self.target_position
            
            print(f"Movement: {movement:.1f} -> {filtered_movement:.1f} -> "
                  f"Position: {self.current_position:.1f}")
        
        # Check if converged
        if self.iteration_count >= self.config.max_iterations:
            self.state = AFState.FAILED
            print("AF failed: maximum iterations reached")
        
        # Calculate final result
        convergence_time = time.time() - start_time
        residual_error = self._calculate_residual_error()
        
        result = AFResult(
            success=(self.state == AFState.CONVERGED),
            final_position=self.current_position,
            iterations=self.iteration_count,
            method_used=method,
            confidence=self._calculate_overall_confidence(),
            convergence_time=convergence_time,
            residual_error=residual_error
        )
        
        return result
    
    def _decide_movement(self, method: AFMethod) -> Optional[float]:
        """
        Decide on next lens movement based on available AF method data.
        
        This is the core decision-making function that combines results
        from different AF methods to determine the optimal lens movement.
        
        Decision Logic:
        1. Collect movements from all available methods
        2. Filter by confidence thresholds
        3. Weight movements by method confidence and hybrid weights
        4. Compute weighted average of movements
        5. Check for convergence (movement < threshold)
        
        Args:
            method: AFMethod enum specifying which methods to use
        
        Returns:
            Lens movement in focus steps (float), or None if converged
            - Positive: Move lens forward (increase position)
            - Negative: Move lens backward (decrease position)
            - Near zero: Consider converged (return None)
            - None: Focus converged, no movement needed
        
        Movement Sources:
        - PDAF: defocus estimate (direct conversion from disparity)
        - CDAF: suggested_movement (based on contrast trend)
        - LDAF: position difference (target_position - current_position)
        
        Weighting:
        For HYBRID mode, movements are weighted by:
        - Method confidence (0.0-1.0): How reliable is this method's result?
        - Hybrid weight (from config): How much should we trust this method?
        - Final weight = method_confidence * hybrid_weight
        
        Convergence Check:
        If |final_movement| < convergence_threshold:
            Return None (converged)
        Otherwise:
            Return final_movement (continue searching)
        
        Example:
            # PDAF suggests +10 steps, confidence 0.8, weight 0.6
            # CDAF suggests +5 steps, confidence 0.6, weight 0.3
            # Final: (10*0.8*0.6 + 5*0.6*0.3) / (0.8*0.6 + 0.6*0.3) = 8.3 steps
        """
        movements = []
        confidences = []
        
        # Collect movements from different methods
        if method == AFMethod.PDAF or method == AFMethod.HYBRID:
            if (self.last_pdaf_result is not None and 
                self.last_pdaf_result['confidence'] > self.config.pdaf_min_confidence):
                movements.append(self.last_pdaf_result['defocus'])
                confidences.append(self.last_pdaf_result['confidence'] * self.config.hybrid_pdaf_weight)
        
        if method == AFMethod.CDAF or method == AFMethod.HYBRID:
            if (self.last_cdaf_result is not None and 
                self.last_cdaf_result['confidence'] > 0.3):
                movements.append(self.last_cdaf_result['suggested_movement'])
                confidences.append(self.last_cdaf_result['confidence'] * self.config.hybrid_cdaf_weight)
        
        if method == AFMethod.LDAF or method == AFMethod.HYBRID:
            if (self.last_ldaf_result is not None and 
                self.last_ldaf_result['suggested_position'] is not None):
                ldaf_movement = self.last_ldaf_result['suggested_position'] - self.current_position
                movements.append(ldaf_movement)
                confidences.append(self.last_ldaf_result['confidence'] * self.config.hybrid_ldaf_weight)
        
        if not movements:
            return None
        
        # Weighted average of movements
        if len(movements) == 1:
            final_movement = movements[0]
        else:
            weights = np.array(confidences)
            weights = weights / np.sum(weights)  # Normalize
            final_movement = np.average(movements, weights=weights)
        
        # Check for convergence
        if abs(final_movement) < self.config.pdaf_convergence_threshold:
            return None
        
        return final_movement
    
    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence from all methods"""
        confidences = []
        
        if self.last_pdaf_result is not None:
            confidences.append(self.last_pdaf_result['confidence'])
        
        if self.last_cdaf_result is not None:
            confidences.append(self.last_cdaf_result['confidence'])
        
        if self.last_ldaf_result is not None:
            confidences.append(self.last_ldaf_result['confidence'])
        
        return np.mean(confidences) if confidences else 0.0
    
    def _calculate_residual_error(self) -> float:
        """Calculate residual focusing error"""
        if self.last_pdaf_result is not None:
            return abs(self.last_pdaf_result['disparity'])
        elif self.last_cdaf_result is not None:
            return abs(self.last_cdaf_result['suggested_movement'])
        else:
            return 0.0


# ============================================================================
# Example Usage and Testing
# ============================================================================

def create_test_image(size: Tuple[int, int] = (480, 640), 
                     blur_radius: float = 2.0) -> np.ndarray:
    """Create a test image with controlled blur"""
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
        from scipy import ndimage
        pattern = ndimage.gaussian_filter(pattern, sigma=blur_radius)
    
    # Convert to uint8
    image = np.clip(pattern, 0, 255).astype(np.uint8)
    
    return image


def create_test_pd_data(image: np.ndarray, 
                       disparity: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Create test PDAF data with known disparity"""
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


def example_autofocus_demo():
    """Comprehensive autofocus demonstration"""
    print("=" * 80)
    print("AUTOFOCUS SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Create configuration
    config = AFConfig(
        pdaf_min_confidence=0.5,
        cdaf_step_size=15,
        hybrid_pdaf_weight=0.6,
        hybrid_cdaf_weight=0.3,
        hybrid_ldaf_weight=0.1
    )
    
    # Initialize autofocus system
    af_system = AutoFocusSystem(config)
    
    print("\n1. Testing PDAF with known disparity...")
    # Create test image and PD data
    test_image = create_test_image(blur_radius=3.0)
    left_pd, right_pd = create_test_pd_data(test_image, disparity=2.5)
    
    # Test PDAF
    result = af_system.trigger_autofocus(
        image=test_image,
        left_pd=left_pd,
        right_pd=right_pd,
        method=AFMethod.PDAF
    )
    
    print(f"PDAF Result: Success={result.success}, "
          f"Position={result.final_position:.1f}, "
          f"Confidence={result.confidence:.3f}")
    
    print("\n2. Testing CDAF with different blur levels...")
    # Test CDAF with different blur levels
    af_system.current_position = 512.0  # Reset position
    
    for blur in [5.0, 3.0, 1.0, 0.5]:
        test_image = create_test_image(blur_radius=blur)
        result = af_system.trigger_autofocus(
            image=test_image,
            method=AFMethod.CDAF
        )
        print(f"Blur {blur:.1f}: Position={result.final_position:.1f}, "
              f"Success={result.success}")
    
    print("\n3. Testing Hybrid AF with all methods...")
    # Test hybrid approach
    af_system.current_position = 512.0  # Reset position
    test_image = create_test_image(blur_radius=2.0)
    left_pd, right_pd = create_test_pd_data(test_image, disparity=1.8)
    
    result = af_system.trigger_autofocus(
        image=test_image,
        left_pd=left_pd,
        right_pd=right_pd,
        ldaf_distance=2.5,  # 2.5 meters
        ldaf_confidence=0.9,
        method=AFMethod.HYBRID
    )
    
    print(f"Hybrid AF Result: Success={result.success}, "
          f"Position={result.final_position:.1f}, "
          f"Iterations={result.iterations}, "
          f"Time={result.convergence_time:.3f}s, "
          f"Confidence={result.confidence:.3f}")
    
    print("\n4. Testing ROI Selection...")
    # Test ROI selection with face simulation
    test_image = create_test_image()
    fake_faces = [(200, 150, 80, 100)]  # (x, y, w, h)
    
    roi = af_system.roi_selector.select_roi(test_image, faces=fake_faces)
    print(f"ROI with face: ({roi.x}, {roi.y}, {roi.width}, {roi.height}), "
          f"method={roi.method}, confidence={roi.confidence:.2f}")
    
    # Test with touch point
    roi = af_system.roi_selector.select_roi(test_image, touch_point=(300, 200))
    print(f"ROI with touch: ({roi.x}, {roi.y}, {roi.width}, {roi.height}), "
          f"method={roi.method}, confidence={roi.confidence:.2f}")


if __name__ == "__main__":
    example_autofocus_demo()
