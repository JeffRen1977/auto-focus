"""
Simple Autofocus Example
========================

This script provides a straightforward example of how to use the autofocus system
for educational purposes. It demonstrates the core concepts without complexity.
"""

import numpy as np
import cv2
from autofocus_system import (
    AutoFocusSystem, AFConfig, AFMethod, 
    create_test_image, create_test_pd_data
)


def simple_af_demo():
    """
    Simple demonstration of autofocus system
    Perfect for understanding the basic concepts
    """
    print("=" * 60)
    print("SIMPLE AUTOFOCUS DEMONSTRATION")
    print("=" * 60)
    
    # Step 1: Create a basic configuration
    print("\n1. Creating AF Configuration...")
    config = AFConfig(
        pdaf_min_confidence=0.5,
        cdaf_step_size=25,
        max_iterations=8
    )
    print(f"   Max iterations: {config.max_iterations}")
    print(f"   PDAF confidence threshold: {config.pdaf_min_confidence}")
    
    # Step 2: Initialize the autofocus system
    print("\n2. Initializing Autofocus System...")
    af_system = AutoFocusSystem(config)
    print(f"   Initial lens position: {af_system.current_position}")
    
    # Step 3: Create test scenarios
    print("\n3. Creating Test Scenarios...")
    
    # Scenario 1: Blurry image (needs focusing)
    print("   Creating blurry test image...")
    blurry_image = create_test_image(size=(400, 600), blur_radius=4.0)
    print(f"   Image size: {blurry_image.shape}")
    
    # Scenario 2: Sharp image (already in focus)
    print("   Creating sharp test image...")
    sharp_image = create_test_image(size=(400, 600), blur_radius=0.5)
    
    # Step 4: Test CDAF (Contrast Detection Auto Focus)
    print("\n4. Testing CDAF (Contrast Detection Auto Focus)...")
    print("   This method analyzes image contrast to find the sharpest position.")
    
    # Test with blurry image
    result_blurry = af_system.trigger_autofocus(
        image=blurry_image,
        method=AFMethod.CDAF
    )
    
    print(f"   Blurry image result:")
    print(f"     Success: {result_blurry.success}")
    print(f"     Final position: {result_blurry.final_position:.1f}")
    print(f"     Iterations: {result_blurry.iterations}")
    print(f"     Time: {result_blurry.convergence_time:.3f}s")
    
    # Reset for next test
    af_system.current_position = 512.0
    
    # Test with sharp image
    result_sharp = af_system.trigger_autofocus(
        image=sharp_image,
        method=AFMethod.CDAF
    )
    
    print(f"   Sharp image result:")
    print(f"     Success: {result_sharp.success}")
    print(f"     Final position: {result_sharp.final_position:.1f}")
    print(f"     Iterations: {result_sharp.iterations}")
    
    # Step 5: Test PDAF (Phase Detection Auto Focus)
    print("\n5. Testing PDAF (Phase Detection Auto Focus)...")
    print("   This method uses phase difference between left/right pixels for fast focusing.")
    
    # Create PDAF test data with known disparity
    left_pd, right_pd = create_test_pd_data(blurry_image, disparity=3.0)
    print(f"   Created PD data with disparity: 3.0 pixels")
    
    # Reset for PDAF test
    af_system.current_position = 512.0
    
    result_pdaf = af_system.trigger_autofocus(
        image=blurry_image,
        left_pd=left_pd,
        right_pd=right_pd,
        method=AFMethod.PDAF
    )
    
    print(f"   PDAF result:")
    print(f"     Success: {result_pdaf.success}")
    print(f"     Final position: {result_pdaf.final_position:.1f}")
    print(f"     Iterations: {result_pdaf.iterations}")
    print(f"     Time: {result_pdaf.convergence_time:.3f}s")
    print(f"     Confidence: {result_pdaf.confidence:.3f}")
    
    # Step 6: Test Hybrid AF (Combined approach)
    print("\n6. Testing Hybrid AF (Combined Approach)...")
    print("   This combines PDAF, CDAF, and LDAF for optimal performance.")
    
    # Reset for hybrid test
    af_system.current_position = 512.0
    
    result_hybrid = af_system.trigger_autofocus(
        image=blurry_image,
        left_pd=left_pd,
        right_pd=right_pd,
        ldaf_distance=2.0,  # 2 meters
        ldaf_confidence=0.8,
        method=AFMethod.HYBRID
    )
    
    print(f"   Hybrid AF result:")
    print(f"     Success: {result_hybrid.success}")
    print(f"     Final position: {result_hybrid.final_position:.1f}")
    print(f"     Iterations: {result_hybrid.iterations}")
    print(f"     Time: {result_hybrid.convergence_time:.3f}s")
    print(f"     Confidence: {result_hybrid.confidence:.3f}")
    
    # Step 7: Summary
    print("\n7. Summary and Comparison...")
    print("   Method Comparison:")
    print(f"     CDAF (blurry):  {result_blurry.iterations:2d} iterations, {result_blurry.convergence_time:.3f}s")
    print(f"     CDAF (sharp):   {result_sharp.iterations:2d} iterations, {result_sharp.convergence_time:.3f}s")
    print(f"     PDAF:           {result_pdaf.iterations:2d} iterations, {result_pdaf.convergence_time:.3f}s")
    print(f"     Hybrid:         {result_hybrid.iterations:2d} iterations, {result_hybrid.convergence_time:.3f}s")
    
    print("\n   Key Observations:")
    print("     • PDAF is fastest (uses phase difference)")
    print("     • CDAF is most accurate but slower (analyzes contrast)")
    print("     • Hybrid combines best of both methods")
    print("     • Sharp images converge quickly")
    print("     • Blurry images need more iterations")


def roi_selection_demo():
    """
    Demonstrate ROI (Region of Interest) selection
    Shows how the system chooses where to focus
    """
    print("\n" + "=" * 60)
    print("ROI SELECTION DEMONSTRATION")
    print("=" * 60)
    
    # Create test image
    test_image = create_test_image(size=(480, 640), blur_radius=2.0)
    
    # Initialize AF system
    config = AFConfig()
    af_system = AutoFocusSystem(config)
    
    print("\n1. Center-weighted ROI (default)...")
    roi_center = af_system.roi_selector.select_roi(test_image)
    print(f"   ROI: ({roi_center.x}, {roi_center.y}, {roi_center.width}, {roi_center.height})")
    print(f"   Method: {roi_center.method}")
    print(f"   Confidence: {roi_center.confidence:.2f}")
    
    print("\n2. Touch-to-focus ROI...")
    roi_touch = af_system.roi_selector.select_roi(
        test_image, 
        touch_point=(300, 200)
    )
    print(f"   ROI: ({roi_touch.x}, {roi_touch.y}, {roi_touch.width}, {roi_touch.height})")
    print(f"   Method: {roi_touch.method}")
    print(f"   Confidence: {roi_touch.confidence:.2f}")
    
    print("\n3. Face-based ROI...")
    fake_faces = [(150, 100, 80, 100)]  # (x, y, width, height)
    roi_face = af_system.roi_selector.select_roi(
        test_image,
        faces=fake_faces
    )
    print(f"   ROI: ({roi_face.x}, {roi_face.y}, {roi_face.width}, {roi_face.height})")
    print(f"   Method: {roi_face.method}")
    print(f"   Confidence: {roi_face.confidence:.2f}")
    
    print("\n   ROI Selection Priority:")
    print("     1. User touch point (highest priority)")
    print("     2. Face detection (high priority)")
    print("     3. Saliency analysis (medium priority)")
    print("     4. Center-weighted (fallback)")


def algorithm_explanation():
    """
    Explain the key algorithms used in autofocus
    """
    print("\n" + "=" * 60)
    print("AUTOFOCUS ALGORITHMS EXPLAINED")
    print("=" * 60)
    
    print("\n1. PDAF (Phase Detection Auto Focus):")
    print("   How it works:")
    print("     • Camera has special 'PD pixels' that can detect phase differences")
    print("     • Left and right PD pixels see slightly different views")
    print("     • When out of focus, left/right images are shifted")
    print("     • The shift amount tells us how much to move the lens")
    print("     • Very fast - usually 1-3 iterations")
    print("     • Works best in good light with textured subjects")
    
    print("\n2. CDAF (Contrast Detection Auto Focus):")
    print("   How it works:")
    print("     • Analyzes image contrast (sharpness)")
    print("     • Moves lens and measures contrast at each position")
    print("     • Peak contrast = best focus position")
    print("     • Slower but more accurate than PDAF")
    print("     • Works in any lighting condition")
    print("     • Can be fooled by low-contrast scenes")
    
    print("\n3. LDAF (Laser Detection Auto Focus):")
    print("   How it works:")
    print("     • Uses laser to measure distance to subject")
    print("     • Converts distance to required lens position")
    print("     • Very fast - usually 1 iteration")
    print("     • Works well in low light")
    print("     • Limited by laser range and reflective surfaces")
    
    print("\n4. Hybrid AF:")
    print("   How it works:")
    print("     • Combines all three methods")
    print("     • Uses confidence weighting")
    print("     • PDAF for speed, CDAF for accuracy, LDAF for low light")
    print("     • Adapts to different shooting conditions")
    print("     • Most robust and reliable approach")


if __name__ == "__main__":
    # Run the simple demonstration
    simple_af_demo()
    
    # Show ROI selection
    roi_selection_demo()
    
    # Explain algorithms
    algorithm_explanation()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("\nThis simple example shows the core concepts of autofocus:")
    print("• Different focusing methods (PDAF, CDAF, LDAF, Hybrid)")
    print("• ROI selection strategies")
    print("• Speed vs accuracy tradeoffs")
    print("• How real camera systems work")
    print("\nFor more advanced features, see the full autofocus_system.py")
