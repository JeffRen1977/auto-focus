"""
Autofocus Processor
===================

This module provides the main high-level interface for the autofocus system.
It integrates raw sensor data reading (RAW10 format) with autofocus calculations
(PDAF and CDAF) to provide a complete end-to-end autofocus processing pipeline.

Key Features:
- Reads and processes PD (Phase Detection) RAW10 sensor data
- Reads and processes Bayer RAW10 sensor data
- Calculates PDAF (Phase Detection Auto Focus) from PD images
- Calculates CDAF (Contrast Detection Auto Focus) from Bayer images
- Implements hybrid AF with intelligent method selection based on confidence
- Provides complete processing pipeline from raw files to focus results

Usage:
    processor = AutofocusProcessor()
    results = processor.process_complete_af(
        pd_file, pd_params,
        bayer_file, bayer_params,
        roi
    )
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import cv2

from autofocus_system import AFConfig, PDAFProcessor, CDAFProcessor, ROISelector, ROIData
from raw_data_reader import read_pd_raw, read_bayer_raw, demosaic_bayer


class AutofocusProcessor:
    """
    Main processor that combines raw data reading with autofocus calculations.
    
    This class serves as the primary interface for the autofocus system, providing
    methods to:
    1. Read and process raw sensor data (PD and Bayer RAW10 files)
    2. Calculate PDAF and CDAF results individually
    3. Combine methods using hybrid AF with confidence-based selection
    4. Run complete end-to-end processing pipeline
    
    The processor uses a threshold-based hybrid approach:
    - If PDAF confidence >= 0.4: Use PDAF-only (fast, hardware-assisted)
    - If PDAF confidence < 0.4: Use CDAF-only (accurate fallback)
    
    This ensures robust performance by avoiding blending unreliable PDAF data.
    """
    
    def __init__(self, calibration_data: Optional[Dict] = None):
        """
        Initialize the autofocus processor.
        
        This method sets up all the necessary components:
        - Creates default AF configuration
        - Initializes PDAF, CDAF, and ROI selector processors
        - Applies calibration data if provided (for accurate disparity-to-defocus conversion)
        
        Args:
            calibration_data: Optional dictionary containing PDAF calibration parameters:
                - 'slope': Conversion factor from disparity to defocus (default: 1.0)
                - 'offset': Offset for disparity-to-defocus conversion (default: 0.0)
                
                Calibration is typically obtained through lens characterization tests
                and ensures accurate conversion from pixel disparity to lens movement steps.
        
        Example:
            # Without calibration (uses defaults)
            processor = AutofocusProcessor()
            
            # With calibration data
            calib = {'slope': 1.2, 'offset': -0.5}
            processor = AutofocusProcessor(calibration_data=calib)
        """
        # Create default configuration with standard AF parameters
        # This includes block sizes, search ranges, confidence thresholds, etc.
        self.config = AFConfig()
        
        # Initialize specialized processors for each AF method
        # PDAFProcessor: Handles phase detection autofocus calculations
        self.pdaf_processor = PDAFProcessor(self.config)
        
        # CDAFProcessor: Handles contrast detection autofocus calculations
        self.cdaf_processor = CDAFProcessor(self.config)
        
        # ROISelector: Handles region of interest selection for focusing
        self.roi_selector = ROISelector(self.config)
        
        # Apply calibration data if provided
        # Calibration converts disparity (pixels) to defocus (lens steps)
        # Formula: defocus = disparity * slope + offset
        if calibration_data:
            self.pdaf_processor.disparity_to_defocus_slope = calibration_data.get('slope', 1.0)
            self.pdaf_processor.disparity_to_defocus_offset = calibration_data.get('offset', 0.0)
        
        print("✅ Autofocus Processor initialized")
    
    def process_pd_data(self, pd_file: str, width: int, height: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read and process PD (Phase Detection) raw sensor data.
        
        This method handles the complete PD data processing pipeline:
        1. Reads RAW10 format file from disk
        2. Unpacks 10-bit packed data to 16-bit arrays
        3. Splits PD image into left and right views using 2x1 column blocks pattern
        4. Returns separated left/right PD images for disparity calculation
        
        The PD sensor uses a specific pixel arrangement where alternating columns
        capture light from different angles, creating left and right views that
        can be used to calculate phase difference (disparity) for fast focusing.
        
        Args:
            pd_file: Path to PD RAW10 file (e.g., "data/pd_frame.raw")
            width: Image width in pixels (e.g., 4032)
            height: Image height in pixels (e.g., 756)
            stride: Row stride in bytes, including padding (e.g., 5056)
                    This accounts for memory alignment in the sensor/ISP
            
        Returns:
            Tuple of (left_pd, right_pd) as uint16 numpy arrays
            - left_pd: Left view PD image, shape (height, width/2)
            - right_pd: Right view PD image, shape (height, width/2)
            
        Raises:
            FileNotFoundError: If PD file doesn't exist
            ValueError: If file size doesn't match expected dimensions
            
        Example:
            left, right = processor.process_pd_data(
                "data/pd.raw", width=4032, height=756, stride=5056
            )
        """
        print("=== Processing PD Data ===")
        left_pd, right_pd = read_pd_raw(pd_file, width, height, stride)
        
        print(f"PD data ready:")
        print(f"  Left PD:  {left_pd.shape}, range [{left_pd.min()}, {left_pd.max()}]")
        print(f"  Right PD: {right_pd.shape}, range [{right_pd.min()}, {right_pd.max()}]")
        
        return left_pd, right_pd
    
    def process_bayer_data(self, bayer_file: str, width: int, height: int, stride: int, 
                          pattern: str = 'BGGR') -> np.ndarray:
        """
        Read and process Bayer raw sensor data.
        
        This method handles the complete Bayer data processing:
        1. Reads RAW10 format file from disk
        2. Unpacks 10-bit packed data to 16-bit arrays
        3. Returns full Bayer image for contrast-based autofocus
        
        The Bayer image contains the full color sensor data arranged in a
        color filter array pattern (e.g., RGGB, BGGR). This data is used
        for CDAF (Contrast Detection Auto Focus) calculations by measuring
        image sharpness/contrast.
        
        Args:
            bayer_file: Path to Bayer RAW10 file (e.g., "data/bayer_frame.raw")
            width: Image width in pixels (e.g., 4032)
            height: Image height in pixels (e.g., 3024)
            stride: Row stride in bytes, including padding (e.g., 5120)
            pattern: Bayer color filter array pattern
                - 'BGGR': Blue-Green-Green-Red (common in many sensors)
                - 'RGGB': Red-Green-Green-Blue
                - 'GBRG': Green-Blue-Red-Green
                - 'GRBG': Green-Red-Blue-Green
                Note: Currently used for documentation, actual demosaicing
                      would use this pattern if needed.
            
        Returns:
            Bayer image as uint16 numpy array, shape (height, width)
            Each pixel contains a single color channel value (R, G, or B)
            
        Raises:
            FileNotFoundError: If Bayer file doesn't exist
            ValueError: If file size doesn't match expected dimensions
            
        Example:
            bayer = processor.process_bayer_data(
                "data/bayer.raw", width=4032, height=3024, stride=5120
            )
        """
        print("=== Processing Bayer Data ===")
        bayer_image = read_bayer_raw(bayer_file, width, height, stride)
        
        print(f"Bayer data ready: {bayer_image.shape}, range [{bayer_image.min()}, {bayer_image.max()}]")
        
        return bayer_image
    
    def calculate_pdaf(self, left_pd: np.ndarray, right_pd: np.ndarray, 
                      roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Calculate PDAF (Phase Detection Auto Focus) from left/right PD images.
        
        PDAF works by:
        1. Matching corresponding blocks between left and right PD images
        2. Computing horizontal disparity (shift) between matched blocks
        3. Converting disparity to defocus estimate using calibration
        4. Providing confidence score based on match quality
        
        This method is fast and hardware-assisted, making it ideal for initial
        focus positioning. However, it requires good contrast and sufficient
        PD pixel coverage in the ROI.
        
        Args:
            left_pd: Left PD image as uint16 numpy array, shape (height, width/2)
            right_pd: Right PD image as uint16 numpy array, shape (height, width/2)
            roi: Optional region of interest tuple (x, y, width, height)
                 - If None: Uses entire image for analysis
                 - If provided: Only analyzes the specified region
                 - This improves speed and focuses on the subject of interest
            
        Returns:
            Dictionary containing PDAF results with keys:
            - 'disparity_map': 2D array of disparity values per block (optional)
            - 'defocus_estimate': Estimated lens movement needed (in focus steps)
              - Positive: Move lens forward (subject behind focus plane)
              - Negative: Move lens backward (subject in front of focus plane)
              - Near zero: Subject is in focus
            - 'mean_disparity': Average horizontal disparity in pixels
            - 'std_disparity': Standard deviation of disparity (indicates consistency)
            - 'confidence': Confidence score (0.0 to 1.0)
              - > 0.7: High confidence, reliable result
              - 0.4-0.7: Medium confidence, use with caution
              - < 0.4: Low confidence, consider using CDAF instead
            - 'roi': The ROI used for calculation
        
        Algorithm Details:
            Uses Block Matching Algorithm (BMA) with Sum of Absolute Differences (SAD):
            - Divides images into blocks (default 8x8 pixels)
            - Searches for best match in horizontal direction
            - Applies sub-pixel accuracy through parabolic fitting
            - Uses robust statistics (median + weighted average) to filter outliers
        
        Example:
            results = processor.calculate_pdaf(left_pd, right_pd, roi=(100, 100, 500, 500))
            if results['confidence'] > 0.6:
                move_lens(results['defocus_estimate'])
        """
        print("=== Calculating PDAF ===")
        
        # Create ROI data structure
        # ROI allows focusing on specific image regions (e.g., subject, face, center)
        if roi:
            x, y, w, h = roi
            roi_data = ROIData(x, y, w, h, confidence=1.0, method='user')
            print(f"Using ROI: ({x}, {y}, {w}, {h})")
        else:
            # Use full image as ROI if none specified
            h, w = left_pd.shape
            roi_data = ROIData(0, 0, w, h, confidence=1.0, method='full')
            print("Using full image")
        
        # Calculate PDAF using the specialized processor
        # This performs block matching, disparity calculation, and defocus estimation
        pdaf_result = self.pdaf_processor.process_pdaf(left_pd, right_pd, roi_data)
        
        # Extract and organize results into a convenient dictionary format
        results = {
            'disparity_map': pdaf_result.get('disparity_map'),  # Optional: full disparity map
            'defocus_estimate': pdaf_result.get('defocus'),     # Lens movement estimate
            'mean_disparity': pdaf_result.get('disparity'),      # Average disparity
            'std_disparity': pdaf_result.get('disparity_std', 0.0),  # Disparity consistency
            'confidence': pdaf_result.get('confidence'),         # Result reliability
            'roi': roi                                          # ROI used
        }
        
        print(f"PDAF Results:")
        print(f"  Mean disparity: {results['mean_disparity']:.3f} pixels")
        print(f"  Std disparity:  {results['std_disparity']:.3f} pixels")
        print(f"  Defocus estimate: {results['defocus_estimate']:.3f}")
        print(f"  Confidence: {results['confidence']:.3f}")
        
        return results
    
    def calculate_cdaf(self, bayer_image: np.ndarray, 
                      roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Calculate CDAF (Contrast Detection Auto Focus) from Bayer image.
        
        CDAF works by:
        1. Extracting the region of interest from the Bayer image
        2. Applying Laplacian filter to measure image sharpness
        3. Computing variance of Laplacian as contrast metric
        4. Determining focus direction based on contrast trends
        
        This method is accurate but slower than PDAF, making it ideal for:
        - Fine-tuning focus after initial PDAF positioning
        - Fallback when PDAF confidence is low
        - Scenes with low contrast or poor PD coverage
        
        Args:
            bayer_image: Bayer image as uint16 numpy array, shape (height, width)
                         Contains single-channel color values in Bayer pattern
            roi: Optional region of interest tuple (x, y, width, height)
                 - If None: Uses entire image for analysis
                 - If provided: Only analyzes the specified region
                 - Typically centered on the subject or user-selected area
            
        Returns:
            Dictionary containing CDAF results with keys:
            - 'contrast_value': Laplacian variance measure of image sharpness
              - Higher values indicate sharper images
              - Values depend on scene content and lighting
              - Typically ranges from thousands to hundreds of thousands
            - 'contrast_map': 2D array of contrast values (optional, for visualization)
            - 'focus_direction': Suggested focus direction
              - 'FORWARD': Move lens forward (subject behind focus plane)
              - 'BACKWARD': Move lens backward (subject in front of focus plane)
              - 'IN_FOCUS': Subject appears to be in focus
            - 'confidence': Confidence score (0.0 to 1.0)
              - Based on contrast magnitude and trend consistency
              - Higher contrast = higher confidence
            - 'roi': The ROI used for calculation
        
        Algorithm Details:
            Uses Laplacian variance method:
            - Applies Laplacian operator to detect edges (sharp transitions)
            - Computes variance of Laplacian response
            - Higher variance = more edges = sharper image = better focus
            - This is a standard metric for image sharpness assessment
        
        Example:
            results = processor.calculate_cdaf(bayer_image, roi=(100, 100, 500, 500))
            if results['focus_direction'] == 'FORWARD':
                move_lens_forward()
        """
        print("=== Calculating CDAF ===")
        
        # Create ROI data structure for focused analysis
        if roi:
            x, y, w, h = roi
            roi_data = ROIData(x, y, w, h, confidence=1.0, method='user')
            print(f"Using ROI: ({x}, {y}, {w}, {h})")
        else:
            # Use full image as ROI if none specified
            h, w = bayer_image.shape
            roi_data = ROIData(0, 0, w, h, confidence=1.0, method='full')
            print("Using full image")
        
        # Calculate CDAF using the specialized processor
        # Note: current_position and step_size are placeholders for iterative CDAF
        # In this implementation, we get a single contrast measurement
        cdaf_result = self.cdaf_processor.process_cdaf(bayer_image, roi_data, 
                                                      current_position=512.0, step_size=10)
        
        # Extract and organize results into a convenient dictionary format
        results = {
            'contrast_value': cdaf_result.get('contrast'),      # Sharpness metric
            'contrast_map': cdaf_result.get('contrast_map'),   # Optional: full contrast map
            'focus_direction': cdaf_result.get('direction'),   # Suggested movement direction
            'confidence': cdaf_result.get('confidence'),       # Result reliability
            'roi': roi                                         # ROI used
        }
        
        print(f"CDAF Results:")
        print(f"  Contrast value: {results['contrast_value']:.3f}")
        print(f"  Focus direction: {results['focus_direction']}")
        print(f"  Confidence: {results['confidence']:.3f}")
        
        return results
    
    def calculate_hybrid_af(self, left_pd: np.ndarray, right_pd: np.ndarray, 
                           bayer_image: np.ndarray,
                           roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Calculate hybrid autofocus combining PDAF and CDAF.
        
        Strategy:
        - If PDAF confidence >= 0.4: Use PDAF-only solution (trust PDAF)
        - If PDAF confidence < 0.4: Switch to CDAF-only solution (fallback to CDAF)
        
        Args:
            left_pd: Left PD image
            right_pd: Right PD image
            bayer_image: Bayer image
            roi: Region of interest (x, y, width, height) or None for full image
            
        Returns:
            Dictionary containing hybrid AF results
        """
        print("=== Calculating Hybrid AF ===")
        
        # Calculate PDAF
        pdaf_results = self.calculate_pdaf(left_pd, right_pd, roi)
        
        # Calculate CDAF
        cdaf_results = self.calculate_cdaf(bayer_image, roi)
        
        # Get confidence scores from both methods
        # These indicate the reliability of each method's result
        pdaf_weight = pdaf_results['confidence'] or 0.0
        cdaf_weight = cdaf_results['confidence'] or 0.0
        
        # Convert CDAF focus direction to a numerical defocus estimate
        # CDAF provides direction hints, which we convert to a simple movement estimate
        # This allows us to use CDAF results when PDAF confidence is low
        cdaf_direction = cdaf_results.get('focus_direction')
        if cdaf_direction == 'FORWARD':
            cdaf_defocus = 1.0   # Positive: move forward
        elif cdaf_direction == 'BACKWARD':
            cdaf_defocus = -1.0  # Negative: move backward
        else:
            cdaf_defocus = 0.0   # Zero: in focus or uncertain
        
        # Decision logic: Intelligent method selection based on PDAF confidence
        # This threshold-based approach avoids blending unreliable PDAF data
        PDAF_CONFIDENCE_THRESHOLD = 0.4
        
        if pdaf_weight >= PDAF_CONFIDENCE_THRESHOLD:
            # PDAF confidence is sufficient (>= 0.4), use PDAF solution only
            # PDAF is fast and accurate when reliable, so we trust it completely
            print(f"PDAF confidence ({pdaf_weight:.3f}) >= {PDAF_CONFIDENCE_THRESHOLD}, using PDAF-only solution")
            hybrid_defocus = pdaf_results['defocus_estimate']  # Use PDAF's precise defocus estimate
            hybrid_confidence = pdaf_weight                     # Use PDAF's confidence
            method_used = "PDAF_ONLY"
        else:
            # PDAF confidence too low (< 0.4), use CDAF solution only
            # When PDAF is unreliable, we fallback to CDAF for accuracy
            print(f"PDAF confidence ({pdaf_weight:.3f}) < {PDAF_CONFIDENCE_THRESHOLD}, switching to CDAF-only solution")
            hybrid_defocus = cdaf_defocus                        # Use CDAF's direction-based estimate
            hybrid_confidence = cdaf_weight                       # Use CDAF's confidence
            method_used = "CDAF_ONLY"
        
        # Determine human-readable focus direction from defocus value
        # This provides clear guidance for lens movement
        if abs(hybrid_defocus) < 0.1:
            focus_direction = "IN_FOCUS"      # Defocus is negligible, already focused
        elif hybrid_defocus > 0:
            focus_direction = "MOVE_FORWARD"  # Positive defocus: move lens forward
        else:
            focus_direction = "MOVE_BACKWARD" # Negative defocus: move lens backward
        
        results = {
            'pdaf_results': pdaf_results,
            'cdaf_results': cdaf_results,
            'hybrid_defocus': hybrid_defocus,
            'hybrid_confidence': hybrid_confidence,
            'focus_direction': focus_direction,
            'method_used': method_used,  # Indicates which method was used
            'roi': roi
        }
        
        print(f"Hybrid AF Results:")
        print(f"  Method used: {method_used}")
        print(f"  Hybrid defocus: {hybrid_defocus:.3f}")
        print(f"  Hybrid confidence: {hybrid_confidence:.3f}")
        print(f"  Focus direction: {focus_direction}")
        
        return results
    
    def process_complete_af(self, pd_file: str, pd_params: Tuple[int, int, int],
                           bayer_file: str, bayer_params: Tuple[int, int, int],
                           roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Complete end-to-end autofocus processing pipeline.
        
        This is the main convenience method that handles the entire autofocus workflow:
        1. Reads and processes PD RAW10 data → splits into left/right images
        2. Reads and processes Bayer RAW10 data → extracts Bayer image
        3. Calculates PDAF from left/right PD images
        4. Calculates CDAF from Bayer image
        5. Combines results using hybrid AF with confidence-based selection
        6. Returns comprehensive results dictionary
        
        This method is ideal for production use as it handles all steps internally
        and provides a single interface for complete autofocus processing.
        
        Args:
            pd_file: Path to PD RAW10 file (e.g., "data/pd_frame.raw")
            pd_params: Tuple of (width, height, stride) for PD data
                - width: PD image width in pixels (e.g., 4032)
                - height: PD image height in pixels (e.g., 756)
                - stride: Row stride in bytes (e.g., 5056)
            bayer_file: Path to Bayer RAW10 file (e.g., "data/bayer_frame.raw")
            bayer_params: Tuple of (width, height, stride) for Bayer data
                - width: Bayer image width in pixels (e.g., 4032)
                - height: Bayer image height in pixels (e.g., 3024)
                - stride: Row stride in bytes (e.g., 5120)
            roi: Optional region of interest tuple (x, y, width, height)
                 - If None: Uses entire image for analysis
                 - If provided: Only analyzes the specified region
                 - Recommended: Center region or subject location
            
        Returns:
            Dictionary containing all autofocus results with keys:
            - 'pdaf_results': Complete PDAF results dictionary
            - 'cdaf_results': Complete CDAF results dictionary
            - 'hybrid_defocus': Final defocus estimate (lens movement needed)
            - 'hybrid_confidence': Overall confidence score
            - 'focus_direction': Recommended action ("MOVE_FORWARD", "MOVE_BACKWARD", "IN_FOCUS")
            - 'method_used': Which method was selected ("PDAF_ONLY" or "CDAF_ONLY")
            - 'roi': The ROI used for calculation
        
        Processing Flow:
            Raw Files → Unpack → Split → Calculate PDAF/CDAF → Hybrid Selection → Results
            
        Example:
            results = processor.process_complete_af(
                "data/pd.raw", (4032, 756, 5056),
                "data/bayer.raw", (4032, 3024, 5120),
                roi=(1500, 1000, 1000, 1000)
            )
            
            # Use results to move lens
            if results['focus_direction'] == 'MOVE_FORWARD':
                move_lens_forward(abs(results['hybrid_defocus']))
        """
        print("=" * 60)
        print("COMPLETE AUTOFOCUS PROCESSING")
        print("=" * 60)
        
        # Process PD data
        left_pd, right_pd = self.process_pd_data(pd_file, *pd_params)
        
        # Process Bayer data
        bayer_image = self.process_bayer_data(bayer_file, *bayer_params)
        
        # Calculate hybrid AF
        af_results = self.calculate_hybrid_af(left_pd, right_pd, bayer_image, roi)
        
        print("\n" + "=" * 60)
        print("AUTOFOCUS PROCESSING COMPLETE")
        print("=" * 60)
        
        return af_results


def main():
    """
    Example usage of the AutofocusProcessor.
    
    This function demonstrates how to use the AutofocusProcessor to:
    1. Initialize the processor
    2. Process real sensor data files
    3. Calculate autofocus results
    4. Interpret and use the results
    
    This can be run directly as a script:
        python autofocus_processor.py
    """
    
    # Initialize processor with default configuration
    # Optionally, you can provide calibration data:
    # calibration = {'slope': 1.2, 'offset': -0.5}
    # processor = AutofocusProcessor(calibration_data=calibration)
    processor = AutofocusProcessor()
    
    # Data file paths and parameters
    # These correspond to real sensor data files in RAW10 format
    pd_file = "data/20250929_223233_713_RAW10_4032x756_RS_5056_PS_1_cam_3_sensor_raw_pd_frame__default_pd_type_622.raw"
    pd_params = (4032, 756, 5056)  # (width, height, stride) for PD data
    
    bayer_file = "data/20250929_223233_713_RAW10_4032x3024_RS_5120_PS_1_cam_3_sensor_raw_output_frame_622.raw"
    bayer_params = (4032, 3024, 5120)  # (width, height, stride) for Bayer data
    
    # Define ROI (center region for focus analysis)
    # Format: (x, y, width, height) in pixels
    # This focuses the analysis on the center region of the image
    # You can adjust this based on your subject location
    roi = (1500, 1000, 1000, 1000)  # x, y, width, height
    
    try:
        # Run complete autofocus processing pipeline
        # This handles everything: reading files, processing data, calculating AF
        results = processor.process_complete_af(
            pd_file, pd_params,
            bayer_file, bayer_params,
            roi
        )
        
        # Display comprehensive results summary
        print("\n📊 Final Results Summary:")
        print(f"  PDAF Defocus: {results['pdaf_results']['defocus_estimate']:.3f}")
        print(f"  PDAF Confidence: {results['pdaf_results']['confidence']:.3f}")
        print(f"  CDAF Contrast: {results['cdaf_results']['contrast_value']:.3f}")
        print(f"  Hybrid Defocus: {results['hybrid_defocus']:.3f}")
        print(f"  Hybrid Confidence: {results['hybrid_confidence']:.3f}")
        print(f"  Focus Direction: {results['focus_direction']}")
        print(f"  Method Used: {results['method_used']}")
        
        # In a real camera system, you would use these results to move the lens:
        # if results['focus_direction'] == 'MOVE_FORWARD':
        #     move_lens_forward(abs(results['hybrid_defocus']))
        # elif results['focus_direction'] == 'MOVE_BACKWARD':
        #     move_lens_backward(abs(results['hybrid_defocus']))
        
    except Exception as e:
        print(f"❌ Error in autofocus processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
