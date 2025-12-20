"""
Autofocus System Configuration
==============================

This module defines all configuration classes, enumerations, and data structures
used throughout the autofocus system. Understanding these classes is fundamental
to using and customizing the autofocus system.

Classes:
- AFMethod: Enumeration of available autofocus methods
- AFState: State machine states for AF system
- AFConfig: Complete configuration parameters for the system
- ROIData: Region of Interest data structure
- AFResult: Result data structure from autofocus operations

This is the first module readers should study to understand the system architecture.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple


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
    """
    Autofocus state machine enumeration.
    
    The autofocus system uses a state machine to track its current status:
    
    - IDLE: System ready, waiting for autofocus trigger
    - SEARCHING: Actively searching for optimal focus position
    - CONVERGED: Focus achieved, within tolerance threshold
    - FAILED: Maximum iterations reached without convergence
    - HUNTING: Oscillating around focus point (not converged)
    """
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
    
    Example:
        # Default configuration
        config = AFConfig()
        
        # Custom configuration for fast PDAF
        config = AFConfig(
            pdaf_min_confidence=0.5,
            pdaf_block_size=4,
            max_iterations=10
        )
        
        # Custom configuration for accurate CDAF
        config = AFConfig(
            cdaf_step_size=10,
            cdaf_contrast_threshold=0.01,
            max_iterations=20
        )
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
    """
    Region of Interest (ROI) data structure.
    
    Represents a rectangular region in the image selected for autofocus processing.
    The ROI defines which part of the image should be analyzed for focusing.
    
    Attributes:
        x: X coordinate of top-left corner (pixels)
        y: Y coordinate of top-left corner (pixels)
        width: Width of ROI (pixels)
        height: Height of ROI (pixels)
        confidence: Confidence score for ROI selection (0.0-1.0)
            - Higher confidence indicates more reliable selection
            - Used to weight ROI importance in processing
        method: Selection method used ('touch', 'face', 'saliency', 'center')
            - 'touch': User explicitly selected region
            - 'face': Automatically detected face region
            - 'saliency': Computed from image features
            - 'center': Default center region
    
    Example:
        # ROI selected by user touch
        roi = ROIData(
            x=150, y=100,
            width=200, height=200,
            confidence=1.0,
            method='touch'
        )
        
        # ROI from face detection
        roi = ROIData(
            x=200, y=150,
            width=180, height=240,
            confidence=0.9,
            method='face'
        )
    """
    x: int
    y: int
    width: int
    height: int
    confidence: float
    method: str  # 'touch', 'face', 'saliency', 'center'


@dataclass
class AFResult:
    """
    Autofocus operation result.
    
    Contains the complete result from an autofocus operation, including
    success status, final lens position, convergence information, and
    performance metrics.
    
    Attributes:
        success: Whether autofocus converged successfully (bool)
            - True: Focus achieved within tolerance
            - False: Failed to converge or timeout
            
        final_position: Final lens position after autofocus (float)
            - Typically in range 0-1023 (10-bit control)
            - Use this to move lens to focus position
            
        iterations: Number of iterations performed (int)
            - Lower is better (faster convergence)
            - Typical: 3-10 iterations
            
        method_used: Primary method used for focusing (AFMethod)
            - Indicates which AF method dominated the result
            - Useful for debugging and optimization
            
        confidence: Overall confidence in the result (float, 0.0-1.0)
            - Higher is better
            - Combines confidence from all used methods
            
        convergence_time: Time taken to converge (seconds, float)
            - Performance metric
            - Typical: 0.1-0.5 seconds
            
        residual_error: Remaining defocus error after convergence (float)
            - Smaller is better
            - Units: Focus steps
            - Should be < convergence_threshold
    
    Example:
        result = af_system.trigger_autofocus(...)
        
        if result.success:
            move_lens_to(result.final_position)
            print(f"Converged in {result.iterations} iterations")
            print(f"Confidence: {result.confidence:.2f}")
        else:
            print("Autofocus failed")
    """
    success: bool
    final_position: float
    iterations: int
    method_used: 'AFMethod'  # Forward reference - see AFMethod enum above
    confidence: float
    convergence_time: float
    residual_error: float

