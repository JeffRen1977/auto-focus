"""
Simple Example Usage of Autofocus System

This example shows how to use the cleaned-up autofocus system with real PD and Bayer data.
"""

from autofocus_processor import AutofocusProcessor
from raw_data_reader import read_pd_raw, read_bayer_raw, save_image_8bit


def main():
    """Example of using the autofocus system"""
    
    print("=" * 60)
    print("AUTOFOCUS SYSTEM - EXAMPLE USAGE")
    print("=" * 60)
    
    # Initialize the autofocus processor
    processor = AutofocusProcessor()
    
    # Data file parameters
    pd_file = "data/20250929_223233_713_RAW10_4032x756_RS_5056_PS_1_cam_3_sensor_raw_pd_frame__default_pd_type_622.raw"
    pd_params = (4032, 756, 5056)  # width, height, stride
    
    bayer_file = "data/20250929_223233_713_RAW10_4032x3024_RS_5120_PS_1_cam_3_sensor_raw_output_frame_622.raw"
    bayer_params = (4032, 3024, 5120)  # width, height, stride
    
    try:
        # Method 1: Use individual functions for step-by-step processing
        print("\n1. STEP-BY-STEP PROCESSING")
        print("-" * 30)
        
        # Read PD data
        left_pd, right_pd = read_pd_raw(pd_file, *pd_params)
        
        # Read Bayer data
        bayer_image = read_bayer_raw(bayer_file, *bayer_params)
        
        # Save images for inspection
        save_image_8bit(left_pd, "data/example_left_pd.jpg")
        save_image_8bit(right_pd, "data/example_right_pd.jpg")
        save_image_8bit(bayer_image, "data/example_bayer.jpg")
        
        print("✅ Images saved for inspection")
        
        # Calculate PDAF only
        print("\n2. PDAF CALCULATION")
        print("-" * 20)
        roi = (1500, 1000, 1000, 1000)  # Center ROI
        pdaf_results = processor.calculate_pdaf(left_pd, right_pd, roi)
        
        # Calculate CDAF only
        print("\n3. CDAF CALCULATION")
        print("-" * 20)
        cdaf_results = processor.calculate_cdaf(bayer_image, roi)
        
        # Method 2: Use complete processing pipeline
        print("\n4. COMPLETE PIPELINE")
        print("-" * 25)
        complete_results = processor.process_complete_af(
            pd_file, pd_params,
            bayer_file, bayer_params,
            roi
        )
        
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"PDAF Results:")
        print(f"  Disparity: {complete_results['pdaf_results']['mean_disparity']:.3f} pixels")
        print(f"  Defocus: {complete_results['pdaf_results']['defocus_estimate']:.3f}")
        print(f"  Confidence: {complete_results['pdaf_results']['confidence']:.3f}")
        
        print(f"\nCDAF Results:")
        print(f"  Contrast: {complete_results['cdaf_results']['contrast_value']:.1f}")
        print(f"  Confidence: {complete_results['cdaf_results']['confidence']:.3f}")
        
        print(f"\nHybrid AF Results:")
        print(f"  Defocus: {complete_results['hybrid_defocus']:.3f}")
        print(f"  Confidence: {complete_results['hybrid_confidence']:.3f}")
        print(f"  Direction: {complete_results['focus_direction']}")
        
        print("\n✅ Autofocus processing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
