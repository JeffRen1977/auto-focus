"""
Autofocus Calibration and Testing Module
========================================

This module provides calibration utilities for the autofocus system,
including PDAF calibration, performance testing, and validation tools.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from autofocus_system import AutoFocusSystem, AFConfig, AFMethod, ROIData


@dataclass
class CalibrationResult:
    """Calibration result data"""
    slope: float
    offset: float
    rmse: float
    r_squared: float
    valid_points: int
    calibration_data: List[Tuple[float, float]]  # (disparity, defocus) pairs


@dataclass
class PerformanceMetrics:
    """AF performance metrics"""
    convergence_time: float
    accuracy: float  # RMS error from target
    repeatability: float  # Standard deviation of repeated measurements
    success_rate: float
    average_iterations: float


class PDAFCalibrator:
    """
    PDAF Calibration System
    Performs factory calibration of PDAF parameters
    """
    
    def __init__(self, config: AFConfig):
        self.config = config
        self.calibration_data = []
    
    def calibrate_with_target(self, 
                             target_distance: float,
                             lens_positions: np.ndarray,
                             pdaf_images: List[Tuple[np.ndarray, np.ndarray]]) -> CalibrationResult:
        """
        Calibrate PDAF using a target at known distance
        
        Args:
            target_distance: Known distance to target (meters)
            lens_positions: Lens positions for each image (DAC values)
            pdaf_images: List of (left_pd, right_pd) image pairs
            
        Returns:
            CalibrationResult: Calibration parameters and quality metrics
        """
        print(f"Starting PDAF calibration for target at {target_distance:.2f}m...")
        
        disparities = []
        defocus_values = []
        
        # Calculate true focus position (would use CDAF in real system)
        true_focus_position = self._find_true_focus_position(
            target_distance, lens_positions, pdaf_images
        )
        
        print(f"True focus position: {true_focus_position:.1f} DAC")
        
        # Process each image pair
        for idx, (left_pd, right_pd) in enumerate(pdaf_images):
            lens_pos = lens_positions[idx]
            
            # Calculate disparity using block matching
            disparity = self._calculate_disparity(left_pd, right_pd)
            
            # Calculate defocus from lens position
            defocus = lens_pos - true_focus_position
            
            # Store valid measurements
            if abs(disparity) < 20:  # Reasonable disparity range
                disparities.append(disparity)
                defocus_values.append(defocus)
                self.calibration_data.append((disparity, defocus))
                
                print(f"  Position {lens_pos:4.0f}: disparity={disparity:6.3f}, "
                      f"defocus={defocus:6.1f}")
        
        # Linear regression: defocus = slope * disparity + offset
        if len(disparities) < 3:
            raise ValueError("Insufficient calibration data")
        
        slope, offset, rmse, r_squared = self._linear_regression(
            np.array(disparities), np.array(defocus_values)
        )
        
        result = CalibrationResult(
            slope=slope,
            offset=offset,
            rmse=rmse,
            r_squared=r_squared,
            valid_points=len(disparities),
            calibration_data=self.calibration_data.copy()
        )
        
        print(f"\nCalibration Results:")
        print(f"  Slope: {slope:.4f}")
        print(f"  Offset: {offset:.4f}")
        print(f"  RMSE: {rmse:.3f} DAC steps")
        print(f"  R²: {r_squared:.4f}")
        print(f"  Valid points: {len(disparities)}")
        
        return result
    
    def _find_true_focus_position(self, 
                                 target_distance: float,
                                 lens_positions: np.ndarray,
                                 pdaf_images: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        Find true focus position using contrast maximization
        (In real system, this would use CDAF or sharpness analysis)
        """
        # Simplified: assume focus position is at distance-based position
        # In reality, this would analyze image sharpness
        distance_to_position = 1000.0 / (target_distance + 0.1) * 100
        return np.clip(distance_to_position, lens_positions.min(), lens_positions.max())
    
    def _calculate_disparity(self, left_pd: np.ndarray, right_pd: np.ndarray) -> float:
        """Calculate disparity using block matching"""
        # Use center region for disparity calculation
        h, w = left_pd.shape
        roi_size = min(200, min(h, w) // 2)
        
        y_start = (h - roi_size) // 2
        x_start = (w - roi_size) // 2
        
        left_roi = left_pd[y_start:y_start+roi_size, x_start:x_start+roi_size]
        right_roi = right_pd[y_start:y_start+roi_size, x_start:x_start+roi_size]
        
        # Block matching with SAD
        block_size = 8
        search_range = 16
        
        disparities = []
        for by in range(0, roi_size - block_size, block_size):
            for bx in range(0, roi_size - block_size, block_size):
                left_block = left_roi[by:by+block_size, bx:bx+block_size]
                
                best_cost = float('inf')
                best_disparity = 0.0
                
                for d in range(-search_range, search_range + 1):
                    x_search = bx + d
                    if x_search < 0 or x_search + block_size > roi_size:
                        continue
                    
                    right_block = right_roi[by:by+block_size, x_search:x_search+block_size]
                    cost = np.sum(np.abs(left_block.astype(np.float32) - right_block.astype(np.float32)))
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_disparity = float(d)
                
                if best_cost < block_size * block_size * 100:  # Threshold
                    disparities.append(best_disparity)
        
        return np.median(disparities) if disparities else 0.0
    
    def _linear_regression(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
        """Perform linear regression and calculate metrics"""
        # Add constant term for offset
        A = np.vstack([x, np.ones(len(x))]).T
        slope, offset = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Calculate predictions and metrics
        y_pred = slope * x + offset
        residuals = y - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return slope, offset, rmse, r_squared
    
    def plot_calibration(self, result: CalibrationResult, save_path: Optional[str] = None):
        """Plot calibration results"""
        disparities, defocus_values = zip(*result.calibration_data)
        
        plt.figure(figsize=(10, 6))
        
        # Scatter plot of data points
        plt.scatter(disparities, defocus_values, alpha=0.6, label='Calibration Data')
        
        # Plot fitted line
        x_line = np.linspace(min(disparities), max(disparities), 100)
        y_line = result.slope * x_line + result.offset
        plt.plot(x_line, y_line, 'r-', linewidth=2, 
                label=f'Fit: y = {result.slope:.3f}x + {result.offset:.3f}')
        
        plt.xlabel('Disparity (pixels)')
        plt.ylabel('Defocus (DAC steps)')
        plt.title(f'PDAF Calibration Results (R² = {result.r_squared:.4f}, RMSE = {result.rmse:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class AFPerformanceTester:
    """
    Autofocus Performance Testing System
    Tests AF accuracy, speed, and repeatability
    """
    
    def __init__(self, config: AFConfig):
        self.config = config
        self.test_results = []
    
    def test_accuracy(self, 
                     af_system: AutoFocusSystem,
                     test_scenarios: List[Dict]) -> Dict[str, PerformanceMetrics]:
        """
        Test AF accuracy across different scenarios
        
        Args:
            af_system: Configured AF system
            test_scenarios: List of test scenarios with known ground truth
            
        Returns:
            Dict mapping scenario names to performance metrics
        """
        print("Starting AF Accuracy Testing...")
        
        results = {}
        
        for scenario in test_scenarios:
            name = scenario['name']
            ground_truth = scenario['ground_truth']
            num_tests = scenario.get('num_tests', 10)
            
            print(f"\nTesting scenario: {name}")
            
            times = []
            accuracies = []
            iterations = []
            success_count = 0
            
            for test_idx in range(num_tests):
                # Reset AF system
                af_system.current_position = scenario.get('start_position', 512)
                af_system.iteration_count = 0
                
                # Create test data
                test_data = self._create_test_data(scenario)
                
                # Run AF
                result = af_system.trigger_autofocus(
                    image=test_data['image'],
                    left_pd=test_data.get('left_pd'),
                    right_pd=test_data.get('right_pd'),
                    ldaf_distance=test_data.get('ldaf_distance'),
                    ldaf_confidence=test_data.get('ldaf_confidence', 1.0),
                    method=scenario.get('method', AFMethod.HYBRID)
                )
                
                if result.success:
                    success_count += 1
                    times.append(result.convergence_time)
                    accuracies.append(abs(result.final_position - ground_truth))
                    iterations.append(result.iterations)
                
                print(f"  Test {test_idx+1}: {'✓' if result.success else '✗'} "
                      f"Position={result.final_position:.1f}, "
                      f"Time={result.convergence_time:.3f}s")
            
            # Calculate metrics
            if success_count > 0:
                metrics = PerformanceMetrics(
                    convergence_time=np.mean(times),
                    accuracy=np.mean(accuracies),
                    repeatability=np.std(accuracies) if len(accuracies) > 1 else 0,
                    success_rate=success_count / num_tests,
                    average_iterations=np.mean(iterations)
                )
            else:
                metrics = PerformanceMetrics(
                    convergence_time=0, accuracy=0, repeatability=0,
                    success_rate=0, average_iterations=0
                )
            
            results[name] = metrics
            
            print(f"  Results: Success={metrics.success_rate:.1%}, "
                  f"Accuracy={metrics.accuracy:.1f} DAC, "
                  f"Time={metrics.convergence_time:.3f}s")
        
        return results
    
    def _create_test_data(self, scenario: Dict) -> Dict:
        """Create test data for a scenario"""
        from autofocus_system import create_test_image, create_test_pd_data
        
        # Create test image with specified blur
        blur_radius = scenario.get('blur_radius', 2.0)
        image = create_test_image(blur_radius=blur_radius)
        
        test_data = {'image': image}
        
        # Add PDAF data if needed
        if 'disparity' in scenario:
            left_pd, right_pd = create_test_pd_data(image, scenario['disparity'])
            test_data['left_pd'] = left_pd
            test_data['right_pd'] = right_pd
        
        # Add LDAF data if needed
        if 'distance' in scenario:
            test_data['ldaf_distance'] = scenario['distance']
            test_data['ldaf_confidence'] = scenario.get('ldaf_confidence', 0.9)
        
        return test_data
    
    def test_repeatability(self, 
                          af_system: AutoFocusSystem,
                          test_conditions: Dict,
                          num_repeats: int = 20) -> PerformanceMetrics:
        """
        Test AF repeatability under identical conditions
        
        Args:
            af_system: Configured AF system
            test_conditions: Test conditions (blur, disparity, etc.)
            num_repeats: Number of repeat tests
            
        Returns:
            PerformanceMetrics: Repeatability metrics
        """
        print(f"Testing repeatability with {num_repeats} repeats...")
        
        final_positions = []
        times = []
        success_count = 0
        
        for repeat in range(num_repeats):
            # Reset system
            af_system.current_position = test_conditions.get('start_position', 512)
            af_system.iteration_count = 0
            
            # Create identical test data
            test_data = self._create_test_data(test_conditions)
            
            # Run AF
            result = af_system.trigger_autofocus(
                image=test_data['image'],
                left_pd=test_data.get('left_pd'),
                right_pd=test_data.get('right_pd'),
                method=test_conditions.get('method', AFMethod.PDAF)
            )
            
            if result.success:
                success_count += 1
                final_positions.append(result.final_position)
                times.append(result.convergence_time)
        
        if len(final_positions) == 0:
            return PerformanceMetrics(0, 0, 0, 0, 0)
        
        # Calculate repeatability metrics
        position_std = np.std(final_positions)
        mean_position = np.mean(final_positions)
        mean_time = np.mean(times)
        
        # Repeatability as standard deviation
        repeatability = position_std
        
        print(f"Repeatability Results:")
        print(f"  Success rate: {success_count/num_repeats:.1%}")
        print(f"  Mean position: {mean_position:.1f} DAC")
        print(f"  Standard deviation: {position_std:.2f} DAC")
        print(f"  Mean time: {mean_time:.3f}s")
        
        return PerformanceMetrics(
            convergence_time=mean_time,
            accuracy=0,  # Not applicable for repeatability test
            repeatability=repeatability,
            success_rate=success_count / num_repeats,
            average_iterations=0
        )


def create_calibration_scenarios() -> List[Dict]:
    """Create standard calibration test scenarios"""
    scenarios = [
        {
            'name': 'Close Range (0.3m)',
            'distance': 0.3,
            'blur_radius': 1.5,
            'disparity': 8.0,
            'ground_truth': 750,
            'method': AFMethod.HYBRID,
            'num_tests': 5
        },
        {
            'name': 'Medium Range (1.5m)',
            'distance': 1.5,
            'blur_radius': 2.5,
            'disparity': 2.0,
            'ground_truth': 600,
            'method': AFMethod.HYBRID,
            'num_tests': 5
        },
        {
            'name': 'Far Range (5.0m)',
            'distance': 5.0,
            'blur_radius': 1.0,
            'disparity': -1.5,
            'ground_truth': 400,
            'method': AFMethod.HYBRID,
            'num_tests': 5
        },
        {
            'name': 'Low Light (PDAF Only)',
            'distance': 2.0,
            'blur_radius': 4.0,
            'disparity': 1.0,
            'ground_truth': 550,
            'method': AFMethod.PDAF,
            'num_tests': 3
        },
        {
            'name': 'High Contrast (CDAF)',
            'distance': 1.0,
            'blur_radius': 3.0,
            'disparity': 0.0,  # No PDAF
            'ground_truth': 650,
            'method': AFMethod.CDAF,
            'num_tests': 3
        }
    ]
    return scenarios


def run_comprehensive_af_test():
    """Run comprehensive autofocus system test"""
    print("=" * 80)
    print("COMPREHENSIVE AUTOFOCUS SYSTEM TEST")
    print("=" * 80)
    
    # Create configuration
    config = AFConfig(
        pdaf_min_confidence=0.5,
        cdaf_step_size=15,
        max_iterations=12
    )
    
    # Initialize systems
    af_system = AutoFocusSystem(config)
    calibrator = PDAFCalibrator(config)
    tester = AFPerformanceTester(config)
    
    print("\n1. PDAF Calibration...")
    # Simulate calibration data
    lens_positions = np.linspace(400, 700, 15)
    pdaf_images = []
    
    for pos in lens_positions:
        # Create test images with known disparity
        disparity = (pos - 550) / 25.0  # Simulate relationship
        image = create_test_image(blur_radius=1.0)
        left_pd, right_pd = create_test_pd_data(image, disparity)
        pdaf_images.append((left_pd, right_pd))
    
    calib_result = calibrator.calibrate_with_target(
        target_distance=2.0,
        lens_positions=lens_positions,
        pdaf_images=pdaf_images
    )
    
    # Update AF system with calibrated parameters
    af_system.pdaf_processor.disparity_to_defocus_slope = calib_result.slope
    af_system.pdaf_processor.disparity_to_defocus_offset = calib_result.offset
    
    print("\n2. Performance Testing...")
    # Test different scenarios
    test_scenarios = create_calibration_scenarios()
    performance_results = tester.test_accuracy(af_system, test_scenarios)
    
    print("\n3. Repeatability Testing...")
    # Test repeatability
    repeatability_result = tester.test_repeatability(
        af_system,
        test_conditions={
            'blur_radius': 2.0,
            'disparity': 1.5,
            'method': AFMethod.PDAF
        },
        num_repeats=15
    )
    
    print("\n4. Summary Report...")
    print(f"Calibration Quality: R² = {calib_result.r_squared:.4f}, RMSE = {calib_result.rmse:.3f}")
    
    print("\nPerformance by Scenario:")
    for name, metrics in performance_results.items():
        print(f"  {name}: Success={metrics.success_rate:.1%}, "
              f"Accuracy={metrics.accuracy:.1f} DAC, Time={metrics.convergence_time:.3f}s")
    
    print(f"\nRepeatability: {repeatability_result.repeatability:.2f} DAC std dev, "
          f"Success={repeatability_result.success_rate:.1%}")
    
    return {
        'calibration': calib_result,
        'performance': performance_results,
        'repeatability': repeatability_result
    }


if __name__ == "__main__":
    # Import test functions
    from autofocus_system import create_test_image, create_test_pd_data
    
    # Run comprehensive test
    results = run_comprehensive_af_test()
