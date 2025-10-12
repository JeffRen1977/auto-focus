"""
Autofocus Processor

Integrates raw data readers with autofocus system to calculate PDAF and CDAF.
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
    """
    
    def __init__(self, calibration_data: Optional[Dict] = None):
        """
        Initialize the autofocus processor.
        
        Args:
            calibration_data: Optional PDAF calibration data
        """
        # Create default configuration
        self.config = AFConfig()
        
        # Initialize processors
        self.pdaf_processor = PDAFProcessor(self.config)
        self.cdaf_processor = CDAFProcessor(self.config)
        self.roi_selector = ROISelector(self.config)
        
        # Set calibration if provided
        if calibration_data:
            self.pdaf_processor.disparity_to_defocus_slope = calibration_data.get('slope', 1.0)
            self.pdaf_processor.disparity_to_defocus_offset = calibration_data.get('offset', 0.0)
        
        print("✅ Autofocus Processor initialized")
    
    def process_pd_data(self, pd_file: str, width: int, height: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read and process PD raw data.
        
        Args:
            pd_file: Path to PD RAW10 file
            width: Image width
            height: Image height
            stride: Row stride in bytes
            
        Returns:
            Tuple of (left_pd, right_pd) images
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
        Read and process Bayer raw data.
        
        Args:
            bayer_file: Path to Bayer RAW10 file
            width: Image width
            height: Image height
            stride: Row stride in bytes
            pattern: Bayer pattern ('BGGR', 'RGGB', 'GBRG', 'GRBG')
            
        Returns:
            Bayer image as uint16 array
        """
        print("=== Processing Bayer Data ===")
        bayer_image = read_bayer_raw(bayer_file, width, height, stride)
        
        print(f"Bayer data ready: {bayer_image.shape}, range [{bayer_image.min()}, {bayer_image.max()}]")
        
        return bayer_image
    
    def calculate_pdaf(self, left_pd: np.ndarray, right_pd: np.ndarray, 
                      roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Calculate PDAF (Phase Detection Auto Focus) from left/right PD images.
        
        Args:
            left_pd: Left PD image
            right_pd: Right PD image
            roi: Region of interest (x, y, width, height) or None for full image
            
        Returns:
            Dictionary containing PDAF results
        """
        print("=== Calculating PDAF ===")
        
        # Create ROI data
        if roi:
            x, y, w, h = roi
            roi_data = ROIData(x, y, w, h, confidence=1.0, method='user')
            print(f"Using ROI: ({x}, {y}, {w}, {h})")
        else:
            # Use full image as ROI
            h, w = left_pd.shape
            roi_data = ROIData(0, 0, w, h, confidence=1.0, method='full')
            print("Using full image")
        
        # Calculate PDAF using the processor
        pdaf_result = self.pdaf_processor.process_pdaf(left_pd, right_pd, roi_data)
        
        # Extract results
        results = {
            'disparity_map': pdaf_result.get('disparity_map'),
            'defocus_estimate': pdaf_result.get('defocus'),
            'mean_disparity': pdaf_result.get('disparity'),
            'std_disparity': pdaf_result.get('disparity_std', 0.0),
            'confidence': pdaf_result.get('confidence'),
            'roi': roi
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
        
        Args:
            bayer_image: Bayer image
            roi: Region of interest (x, y, width, height) or None for full image
            
        Returns:
            Dictionary containing CDAF results
        """
        print("=== Calculating CDAF ===")
        
        # Create ROI data
        if roi:
            x, y, w, h = roi
            roi_data = ROIData(x, y, w, h, confidence=1.0, method='user')
            print(f"Using ROI: ({x}, {y}, {w}, {h})")
        else:
            # Use full image as ROI
            h, w = bayer_image.shape
            roi_data = ROIData(0, 0, w, h, confidence=1.0, method='full')
            print("Using full image")
        
        # Calculate CDAF using the processor
        cdaf_result = self.cdaf_processor.process_cdaf(bayer_image, roi_data, 
                                                      current_position=512.0, step_size=10)
        
        # Extract results
        results = {
            'contrast_value': cdaf_result.get('contrast'),
            'contrast_map': cdaf_result.get('contrast_map'),
            'focus_direction': cdaf_result.get('direction'),
            'confidence': cdaf_result.get('confidence'),
            'roi': roi
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
        
        # Simple hybrid combination (weighted average)
        pdaf_weight = pdaf_results['confidence'] or 0.0
        cdaf_weight = cdaf_results['confidence'] or 0.0
        total_weight = pdaf_weight + cdaf_weight
        
        if total_weight > 0:
            # Use CDAF focus direction as a simple movement indicator
            cdaf_direction = cdaf_results.get('focus_direction')
            if cdaf_direction == 'FORWARD':
                cdaf_defocus = 1.0
            elif cdaf_direction == 'BACKWARD':
                cdaf_defocus = -1.0
            else:
                cdaf_defocus = 0.0
            
            hybrid_defocus = (pdaf_results['defocus_estimate'] * pdaf_weight + 
                            cdaf_defocus * cdaf_weight) / total_weight
            hybrid_confidence = (pdaf_weight + cdaf_weight) / 2.0
        else:
            hybrid_defocus = 0.0
            hybrid_confidence = 0.0
        
        # Determine focus direction
        if abs(hybrid_defocus) < 0.1:
            focus_direction = "IN_FOCUS"
        elif hybrid_defocus > 0:
            focus_direction = "MOVE_FORWARD"
        else:
            focus_direction = "MOVE_BACKWARD"
        
        results = {
            'pdaf_results': pdaf_results,
            'cdaf_results': cdaf_results,
            'hybrid_defocus': hybrid_defocus,
            'hybrid_confidence': hybrid_confidence,
            'focus_direction': focus_direction,
            'roi': roi
        }
        
        print(f"Hybrid AF Results:")
        print(f"  Hybrid defocus: {hybrid_defocus:.3f}")
        print(f"  Hybrid confidence: {hybrid_confidence:.3f}")
        print(f"  Focus direction: {focus_direction}")
        
        return results
    
    def process_complete_af(self, pd_file: str, pd_params: Tuple[int, int, int],
                           bayer_file: str, bayer_params: Tuple[int, int, int],
                           roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Complete autofocus processing pipeline.
        
        Args:
            pd_file: Path to PD RAW10 file
            pd_params: (width, height, stride) for PD data
            bayer_file: Path to Bayer RAW10 file  
            bayer_params: (width, height, stride) for Bayer data
            roi: Region of interest or None for full image
            
        Returns:
            Dictionary containing all AF results
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
    """Example usage of the AutofocusProcessor"""
    
    # Initialize processor
    processor = AutofocusProcessor()
    
    # Data file paths and parameters
    pd_file = "data/20250929_223233_713_RAW10_4032x756_RS_5056_PS_1_cam_3_sensor_raw_pd_frame__default_pd_type_622.raw"
    pd_params = (4032, 756, 5056)  # width, height, stride
    
    bayer_file = "data/20250929_223233_713_RAW10_4032x3024_RS_5120_PS_1_cam_3_sensor_raw_output_frame_622.raw"
    bayer_params = (4032, 3024, 5120)  # width, height, stride
    
    # Define ROI (center region for focus analysis)
    roi = (1500, 1000, 1000, 1000)  # x, y, width, height
    
    try:
        # Run complete autofocus processing
        results = processor.process_complete_af(
            pd_file, pd_params,
            bayer_file, bayer_params,
            roi
        )
        
        print("\n📊 Final Results Summary:")
        print(f"  PDAF Defocus: {results['pdaf_results']['defocus_estimate']:.3f}")
        print(f"  PDAF Confidence: {results['pdaf_results']['confidence']:.3f}")
        print(f"  CDAF Contrast: {results['cdaf_results']['contrast_value']:.3f}")
        print(f"  Hybrid Defocus: {results['hybrid_defocus']:.3f}")
        print(f"  Hybrid Confidence: {results['hybrid_confidence']:.3f}")
        print(f"  Focus Direction: {results['focus_direction']}")
        
    except Exception as e:
        print(f"❌ Error in autofocus processing: {e}")


if __name__ == "__main__":
    main()
