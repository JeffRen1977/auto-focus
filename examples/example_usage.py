"""
Simple Example Usage of Autofocus System
=========================================

This example demonstrates how to use the autofocus system with real PD (Phase Detection)
and Bayer sensor data. It shows two approaches:
1. Step-by-step processing for detailed control
2. Complete pipeline for convenience

The example processes RAW10 format sensor data and calculates autofocus results using:
- PDAF (Phase Detection Auto Focus): Fast focusing using phase difference
- CDAF (Contrast Detection Auto Focus): Accurate focusing using image contrast
- Hybrid AF: Combination of both methods for optimal performance
"""

import sys
import os

# Add parent directory to path to allow imports from the main package
# This is needed because this script is in the examples/ subdirectory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core autofocus processing classes and functions
from autofocus_processor import AutofocusProcessor
from raw_data_reader import read_pd_raw, read_bayer_raw, save_image_8bit


def main():
    """
    Main example function demonstrating autofocus system usage.
    
    This function shows how to:
    1. Initialize the autofocus processor
    2. Read and process PD (Phase Detection) raw sensor data
    3. Read and process Bayer raw sensor data
    4. Calculate PDAF and CDAF results individually
    5. Run the complete autofocus pipeline
    6. Interpret and display the results
    """
    
    print("=" * 60)
    print("AUTOFOCUS SYSTEM - EXAMPLE USAGE")
    print("=" * 60)
    
    # Initialize the autofocus processor
    # This creates instances of PDAFProcessor, CDAFProcessor, and ROISelector
    # with default configuration. You can also pass custom calibration data.
    processor = AutofocusProcessor()
    
    # ========================================================================
    # Data File Parameters
    # ========================================================================
    # PD (Phase Detection) data parameters:
    # - PD files contain dedicated phase detection pixels arranged in a pattern
    # - These pixels provide fast focus information by measuring phase difference
    pd_file = "data/20250929_223233_713_RAW10_4032x756_RS_5056_PS_1_cam_3_sensor_raw_pd_frame__default_pd_type_622.raw"
    pd_params = (4032, 756, 5056)  # (width, height, stride)
    #   width:  4032 pixels - horizontal resolution of PD image
    #   height: 756 pixels  - vertical resolution of PD image  
    #   stride: 5056 bytes  - row stride in bytes (including padding)
    
    # Bayer data parameters:
    # - Bayer files contain the full color image sensor data in RAW10 format
    # - These are used for contrast-based autofocus (CDAF) calculations
    bayer_file = "data/20250929_223233_713_RAW10_4032x3024_RS_5120_PS_1_cam_3_sensor_raw_output_frame_622.raw"
    bayer_params = (4032, 3024, 5120)  # (width, height, stride)
    #   width:  4032 pixels - horizontal resolution of Bayer image
    #   height: 3024 pixels - vertical resolution of Bayer image
    #   stride: 5120 bytes  - row stride in bytes (including padding)
    
    try:
        # ====================================================================
        # Method 1: Step-by-Step Processing
        # ====================================================================
        # This approach gives you full control over each processing step.
        # Useful for debugging, understanding intermediate results, or custom workflows.
        print("\n1. STEP-BY-STEP PROCESSING")
        print("-" * 30)
        
        # --------------------------------------------------------------------
        # Step 1.1: Read PD (Phase Detection) Data
        # --------------------------------------------------------------------
        # This function:
        # - Reads the RAW10 file and unpacks it to uint16 format
        # - Splits the PD image into left and right views using 2x1 column blocks pattern
        # - Returns two numpy arrays: left_pd and right_pd
        # The splitting is based on the sensor's PD pixel arrangement where
        # alternating columns capture light from different angles (left/right views)
        left_pd, right_pd = read_pd_raw(pd_file, *pd_params)
        # Returns: (left_pd, right_pd) as uint16 numpy arrays
        #          Shape: (height, width/2) for each, typically (756, 2016)
        
        # --------------------------------------------------------------------
        # Step 1.2: Read Bayer Data
        # --------------------------------------------------------------------
        # This function:
        # - Reads the RAW10 Bayer file and unpacks it to uint16 format
        # - Returns the full Bayer image as a numpy array
        # The Bayer pattern contains color filter information (RGGB, BGGR, etc.)
        # which is used for contrast detection autofocus
        bayer_image = read_bayer_raw(bayer_file, *bayer_params)
        # Returns: bayer_image as uint16 numpy array
        #          Shape: (height, width), typically (3024, 4032)
        
        # --------------------------------------------------------------------
        # Step 1.3: Save Images for Visual Inspection
        # --------------------------------------------------------------------
        # Convert uint16 images to 8-bit and save as JPEG for visualization
        # The save_image_8bit function automatically normalizes the image
        # using the actual data range (min to max) for proper contrast display
        save_image_8bit(left_pd, "data/example_left_pd.jpg")
        save_image_8bit(right_pd, "data/example_right_pd.jpg")
        save_image_8bit(bayer_image, "data/example_bayer.jpg")
        
        print("✅ Images saved for inspection")
        print("   Check data/example_*.jpg to verify the raw data was read correctly")
        
        # ====================================================================
        # Method 2: Individual AF Calculations
        # ====================================================================
        # Calculate PDAF and CDAF separately to understand each method's contribution
        
        # --------------------------------------------------------------------
        # Step 2.1: Calculate PDAF (Phase Detection Auto Focus)
        # --------------------------------------------------------------------
        print("\n2. PDAF CALCULATION")
        print("-" * 20)
        
        # ROI (Region of Interest) defines the area of the image to focus on
        # Format: (x, y, width, height) in pixels
        roi = (1500, 1000, 1000, 1000)
        #   x:      1500 - left edge of ROI
        #   y:      1000 - top edge of ROI
        #   width:  1000 - width of ROI
        #   height: 1000 - height of ROI
        # This ROI is centered in a 4032x3024 image, focusing on the middle region
        
        # Calculate PDAF from left and right PD images
        # This uses block matching algorithm (BMA) to find disparity between views
        pdaf_results = processor.calculate_pdaf(left_pd, right_pd, roi)
        # Returns: Dictionary with keys:
        #   - 'mean_disparity': Horizontal shift in pixels between left/right views
        #   - 'defocus_estimate': Estimated lens movement needed (in focus steps)
        #   - 'confidence': Confidence score (0.0 to 1.0, higher is better)
        #   - 'valid_blocks': Number of valid block matches found
        
        # --------------------------------------------------------------------
        # Step 2.2: Calculate CDAF (Contrast Detection Auto Focus)
        # --------------------------------------------------------------------
        print("\n3. CDAF CALCULATION")
        print("-" * 20)
        
        # Calculate CDAF from Bayer image
        # This measures image sharpness using Laplacian variance
        # Higher contrast = sharper image = better focus
        cdaf_results = processor.calculate_cdaf(bayer_image, roi)
        # Returns: Dictionary with keys:
        #   - 'contrast_value': Laplacian variance (higher = sharper)
        #   - 'confidence': Confidence score (0.0 to 1.0)
        #   - 'focus_direction': Direction hint ('forward', 'backward', or 'in_focus')
        
        # ====================================================================
        # Method 3: Complete Pipeline Processing
        # ====================================================================
        # This is the recommended approach for production use.
        # It handles all steps internally: reading, processing, and combining results.
        print("\n4. COMPLETE PIPELINE")
        print("-" * 25)
        print("   Processing complete autofocus pipeline...")
        print("   This includes: PD reading, Bayer reading, PDAF, CDAF, and Hybrid AF")
        
        # Run the complete autofocus pipeline
        # This function does everything: reads files, processes data, calculates AF results
        complete_results = processor.process_complete_af(
            pd_file,        # Path to PD RAW10 file
            pd_params,      # (width, height, stride) tuple for PD data
            bayer_file,     # Path to Bayer RAW10 file
            bayer_params,   # (width, height, stride) tuple for Bayer data
            roi             # Optional ROI tuple (x, y, width, height)
        )
        # Returns: Dictionary containing:
        #   - 'pdaf_results': PDAF calculation results
        #   - 'cdaf_results': CDAF calculation results
        #   - 'hybrid_defocus': Combined defocus estimate from hybrid AF
        #   - 'hybrid_confidence': Overall confidence score
        #   - 'focus_direction': Final focus direction recommendation
        
        # ====================================================================
        # Results Interpretation and Display
        # ====================================================================
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        # Display PDAF Results
        print(f"\nPDAF Results (Phase Detection Auto Focus):")
        print(f"  Disparity: {complete_results['pdaf_results']['mean_disparity']:.3f} pixels")
        #   Disparity: Horizontal shift between left/right PD images
        #   - Positive: Subject is behind focus plane (move lens forward)
        #   - Negative: Subject is in front of focus plane (move lens backward)
        #   - Near zero: Subject is in focus
        
        print(f"  Defocus: {complete_results['pdaf_results']['defocus_estimate']:.3f}")
        #   Defocus: Estimated lens movement needed (in focus steps)
        #   - This is calculated from disparity using calibration parameters
        #   - Sign indicates direction: positive = forward, negative = backward
        
        print(f"  Confidence: {complete_results['pdaf_results']['confidence']:.3f}")
        #   Confidence: Quality score (0.0 to 1.0)
        #   - > 0.7: High confidence, reliable result
        #   - 0.4-0.7: Medium confidence, use with caution
        #   - < 0.4: Low confidence, consider using CDAF or hybrid method
        
        # Display CDAF Results
        print(f"\nCDAF Results (Contrast Detection Auto Focus):")
        print(f"  Contrast: {complete_results['cdaf_results']['contrast_value']:.1f}")
        #   Contrast: Laplacian variance measure of image sharpness
        #   - Higher values indicate sharper images
        #   - Values depend on scene content and lighting
        #   - Typically ranges from thousands to hundreds of thousands
        
        print(f"  Confidence: {complete_results['cdaf_results']['confidence']:.3f}")
        #   Confidence: Quality score based on contrast magnitude
        
        # Display Hybrid AF Results (Combined)
        print(f"\nHybrid AF Results (Combined PDAF + CDAF):")
        print(f"  Defocus: {complete_results['hybrid_defocus']:.3f}")
        #   Hybrid Defocus: Weighted combination of PDAF and CDAF defocus estimates
        #   - PDAF typically has 60% weight (fast, good initial estimate)
        #   - CDAF typically has 30% weight (accurate, good for fine-tuning)
        #   - This provides the best overall focus estimate
        
        print(f"  Confidence: {complete_results['hybrid_confidence']:.3f}")
        #   Hybrid Confidence: Combined confidence score
        #   - Takes into account both PDAF and CDAF confidence scores
        #   - Higher is better, indicates more reliable focus decision
        
        print(f"  Direction: {complete_results['focus_direction']}")
        #   Focus Direction: Recommended action
        #   - 'FORWARD': Move lens forward (subject is behind focus plane)
        #   - 'BACKWARD': Move lens backward (subject is in front of focus plane)
        #   - 'IN_FOCUS': Subject appears to be in focus
        #   - 'UNCERTAIN': Confidence too low to make a recommendation
        
        print("\n✅ Autofocus processing completed successfully!")
        print("\nNext Steps:")
        print("  1. Use the 'defocus' value to move the lens")
        print("  2. Check 'confidence' to assess result reliability")
        print("  3. Follow 'focus_direction' for the recommended action")
        print("  4. For low confidence, consider multiple frames or different ROI")
        
    except FileNotFoundError as e:
        # Handle missing data files
        print(f"❌ File not found: {e}")
        print("\nPlease ensure:")
        print("  1. Data files exist in the 'data/' directory")
        print("  2. File paths are correct")
        print("  3. File names match the expected format")
    except ValueError as e:
        # Handle parameter errors (wrong dimensions, stride, etc.)
        print(f"❌ Parameter error: {e}")
        print("\nPlease check:")
        print("  1. Image dimensions (width, height) are correct")
        print("  2. Stride values match the actual RAW10 file format")
        print("  3. ROI coordinates are within image bounds")
    except Exception as e:
        # Handle any other unexpected errors
        print(f"❌ Unexpected error: {e}")
        print("\nFull error traceback:")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    """
    Entry point for the example script.
    Run this file directly to execute the autofocus example.
    
    Usage:
        python examples/example_usage.py
        
    Or from the project root:
        python -m examples.example_usage
    """
    main()
