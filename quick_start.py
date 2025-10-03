#!/usr/bin/env python3
"""
Quick Start Script for Autofocus System
=======================================

This script provides a quick way to test the autofocus system
without running the full demos. Perfect for quick testing and validation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = ['numpy', 'scipy', 'cv2', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"   ✅ OpenCV: {cv2.__version__}")
            elif package == 'numpy':
                import numpy as np
                print(f"   ✅ NumPy: {np.__version__}")
            elif package == 'scipy':
                import scipy
                print(f"   ✅ SciPy: {scipy.__version__}")
            elif package == 'matplotlib':
                import matplotlib
                print(f"   ✅ Matplotlib: {matplotlib.__version__}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}: Not installed")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are available!")
    return True


def quick_test():
    """Run a quick test of the autofocus system"""
    print("\n🚀 Running Quick Autofocus Test...")
    
    try:
        # Import the autofocus system
        from autofocus_system import (
            AutoFocusSystem, AFConfig, AFMethod,
            create_test_image, create_test_pd_data
        )
        
        print("   ✅ Autofocus system imported successfully")
        
        # Create configuration
        config = AFConfig(
            pdaf_min_confidence=0.5,
            cdaf_step_size=20,
            max_iterations=6
        )
        print("   ✅ Configuration created")
        
        # Initialize system
        af_system = AutoFocusSystem(config)
        print("   ✅ Autofocus system initialized")
        
        # Create test data
        test_image = create_test_image(size=(300, 400), blur_radius=3.0)
        left_pd, right_pd = create_test_pd_data(test_image, disparity=2.0)
        print("   ✅ Test data created")
        
        # Test PDAF
        print("   🔍 Testing PDAF...")
        result_pdaf = af_system.trigger_autofocus(
            image=test_image,
            left_pd=left_pd,
            right_pd=right_pd,
            method=AFMethod.PDAF
        )
        
        print(f"      PDAF Result: Success={result_pdaf.success}, "
              f"Position={result_pdaf.final_position:.1f}, "
              f"Iterations={result_pdaf.iterations}")
        
        # Test CDAF
        print("   🔍 Testing CDAF...")
        af_system.current_position = 512.0  # Reset position
        
        result_cdaf = af_system.trigger_autofocus(
            image=test_image,
            method=AFMethod.CDAF
        )
        
        print(f"      CDAF Result: Success={result_cdaf.success}, "
              f"Position={result_cdaf.final_position:.1f}, "
              f"Iterations={result_cdaf.iterations}")
        
        # Test Hybrid
        print("   🔍 Testing Hybrid AF...")
        af_system.current_position = 512.0  # Reset position
        
        result_hybrid = af_system.trigger_autofocus(
            image=test_image,
            left_pd=left_pd,
            right_pd=right_pd,
            ldaf_distance=2.5,
            ldaf_confidence=0.9,
            method=AFMethod.HYBRID
        )
        
        print(f"      Hybrid Result: Success={result_hybrid.success}, "
              f"Position={result_hybrid.final_position:.1f}, "
              f"Iterations={result_hybrid.iterations}")
        
        print("\n🎉 Quick test completed successfully!")
        print("\n📊 Summary:")
        print(f"   PDAF:    {'✅' if result_pdaf.success else '❌'} "
              f"({result_pdaf.iterations} iterations)")
        print(f"   CDAF:    {'✅' if result_cdaf.success else '❌'} "
              f"({result_cdaf.iterations} iterations)")
        print(f"   Hybrid:  {'✅' if result_hybrid.success else '❌'} "
              f"({result_hybrid.iterations} iterations)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        return False


def show_usage():
    """Show usage instructions"""
    print("""
📚 Autofocus System - Quick Start Guide
=======================================

This script provides a quick test of the autofocus system.

Usage:
    python quick_start.py [options]

Options:
    --check-deps    Only check dependencies
    --test         Run quick test (default)
    --help         Show this help message

Examples:
    python quick_start.py                    # Run quick test
    python quick_start.py --check-deps       # Check dependencies only
    python quick_start.py --test            # Run quick test

Available Demo Scripts:
    python simple_example.py                 # Educational examples
    python autofocus_system.py              # Full system demo
    python af_calibration.py                # Calibration demo
    python af_visualization.py              # Visualization demo

Setup:
    1. Run setup.sh to create virtual environment
    2. Activate environment: source af_env/bin/activate
    3. Run demos with run_demo.sh or directly with python
""")


def main():
    """Main function"""
    if len(sys.argv) > 1:
        if '--help' in sys.argv:
            show_usage()
            return
        elif '--check-deps' in sys.argv:
            check_dependencies()
            return
    
    print("🚀 Autofocus System Quick Start")
    print("===============================")
    
    # Check dependencies
    if not check_dependencies():
        print("\n💡 To install dependencies:")
        print("   pip install -r requirements.txt")
        print("\n💡 Or run the setup script:")
        print("   ./setup.sh")
        return
    
    # Run quick test
    success = quick_test()
    
    if success:
        print("\n🎯 Next Steps:")
        print("   1. Run full demo: python autofocus_system.py")
        print("   2. Try educational examples: python simple_example.py")
        print("   3. Explore calibrations: python af_calibration.py")
        print("   4. View visualizations: python af_visualization.py")
        print("\n📖 Read README.md for detailed documentation")
    else:
        print("\n🔧 Troubleshooting:")
        print("   1. Check all dependencies are installed")
        print("   2. Verify Python version (3.7+)")
        print("   3. Try running setup.sh again")


if __name__ == "__main__":
    main()
