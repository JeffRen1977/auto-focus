"""
Comprehensive Autofocus System Implementation
=============================================

This module implements a complete autofocus system that demonstrates:
1. PDAF (Phase Detection Auto Focus) - Fast, hardware-assisted focusing
2. CDAF (Contrast Detection Auto Focus) - Accurate but slower focusing
3. LDAF (Laser Detection Auto Focus) - Distance-based focusing
4. Hybrid AF - Intelligent combination of all methods
5. ROI Selection - Smart area selection for focusing
6. Temporal Filtering - Smooth convergence and stability

Based on AF system architecture and mobile computational photography principles.
"""

import numpy as np
from scipy import signal, optimize, ndimage
from typing import Tuple, Dict, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import cv2


class AFMethod(Enum):
    """Autofocus methods"""
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
    """Autofocus system configuration"""
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
    ROI (Region of Interest) Selector
    Intelligently selects the best area for focusing
    """
    
    def __init__(self, config: AFConfig):
        self.config = config
    
    def select_roi(self, image: np.ndarray, 
                   touch_point: Optional[Tuple[int, int]] = None,
                   faces: Optional[List[Tuple[int, int, int, int]]] = None) -> ROIData:
        """
        Select the best ROI for autofocus
        
        Args:
            image: Input image
            touch_point: User touch point (x, y)
            faces: Detected faces [(x, y, w, h), ...]
            
        Returns:
            ROIData: Selected ROI information
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
    Phase Detection Auto Focus Processor
    Fast focusing using phase difference information
    """
    
    def __init__(self, config: AFConfig):
        self.config = config
        # Calibration parameters (would be loaded from device calibration)
        self.disparity_to_defocus_slope = 25.0
        self.disparity_to_defocus_offset = -2.0
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
        """Block Matching Algorithm for disparity computation"""
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
        """Match a single block using SAD (Sum of Absolute Differences)"""
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
    Contrast Detection Auto Focus Processor
    Accurate but slower focusing method
    """
    
    def __init__(self, config: AFConfig):
        self.config = config
        self.contrast_history = []
        self.position_history = []
    
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
        """Calculate image contrast using Laplacian variance"""
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
    Main Autofocus System Controller
    Coordinates PDAF, CDAF, and LDAF for optimal focusing
    """
    
    def __init__(self, config: AFConfig):
        self.config = config
        self.state = AFState.IDLE
        self.current_position = 512.0  # Default middle position
        self.target_position = 512.0
        self.iteration_count = 0
        
        # Initialize processors
        self.roi_selector = ROISelector(config)
        self.pdaf_processor = PDAFProcessor(config)
        self.cdaf_processor = CDAFProcessor(config)
        self.ldaf_processor = LDAFProcessor(config)
        
        # State tracking
        self.last_pdaf_result = None
        self.last_cdaf_result = None
        self.last_ldaf_result = None
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
        Trigger autofocus process
        
        Args:
            image: Input image
            left_pd: Left PDAF pixels (optional)
            right_pd: Right PDAF pixels (optional)
            ldaf_distance: LDAF distance measurement (optional)
            ldaf_confidence: LDAF confidence
            touch_point: User touch point (optional)
            faces: Detected faces (optional)
            method: AF method to use
            
        Returns:
            AFResult: Autofocus result
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
        """Decide on next lens movement based on available data"""
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
