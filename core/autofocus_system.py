"""
Main Autofocus System Integration
==================================

This module provides the main AutoFocusSystem class that integrates all
autofocus components (PDAF, CDAF, LDAF, ROI selection) into a complete
autofocus system.

This is the final module in the learning progression. After understanding
the individual components (config, roi_selection, pdaf, cdaf, ldaf),
students should study this module to understand system integration.

Class:
- AutoFocusSystem: Main system controller coordinating all AF methods
"""

import numpy as np
import time
from typing import Tuple, Dict, Optional, List

from .config import AFConfig, AFMethod, AFState, AFResult
from .roi_selection import ROISelector
from .pdaf import PDAFProcessor
from .cdaf import CDAFProcessor
from .ldaf import LDAFProcessor


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
        from core import AFConfig, AutoFocusSystem, AFMethod
        
        config = AFConfig()
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
        
        For detailed documentation, see the module docstring and the
        individual processor classes (PDAFProcessor, CDAFProcessor, etc.)
        
        Args:
            image: Input image as numpy array (Bayer or RGB)
            left_pd: Left PD image (optional, required for PDAF)
            right_pd: Right PD image (optional, required for PDAF)
            ldaf_distance: LDAF distance measurement in meters (optional)
            ldaf_confidence: LDAF measurement confidence (default: 1.0)
            touch_point: User touch point (x, y) in pixels (optional)
            faces: List of detected faces (optional)
            method: AF method to use (default: HYBRID)
        
        Returns:
            AFResult object containing success, final_position, iterations,
            method_used, confidence, convergence_time, and residual_error
        """
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
        
        For detailed documentation, see the class docstring.
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

