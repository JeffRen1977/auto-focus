# Comprehensive Autofocus System

This repository contains a complete implementation of a modern autofocus system that demonstrates the principles and algorithms used in mobile computational photography. The system implements a comprehensive autofocus architecture and includes multiple focusing methods working together.

## 🎯 Overview

The autofocus system implements three main focusing technologies:

1. **PDAF (Phase Detection Auto Focus)** - Fast, hardware-assisted focusing using phase difference
2. **CDAF (Contrast Detection Auto Focus)** - Accurate but slower focusing based on contrast maximization  
3. **LDAF (Laser Detection Auto Focus)** - Distance-based focusing using laser ranging
4. **Hybrid AF** - Intelligent combination of all methods for optimal performance

## 🏗️ System Architecture

```
Input Data Sources:
├── Image Sensor (Bayer Raw + PD pixels)
├── ISP Statistics (PDAF, CDAF, Luminance)
├── Scene Analysis (Face detection, Saliency)
├── Depth Sensors (LDAF, ToF)
├── Motion Sensors (Gyroscope)
└── User Input (Touch-to-focus)

Processing Pipeline:
├── ROI Selection (Touch/Face/Saliency/Center)
├── Defocus Estimation
│   ├── PDAF Pipeline (Block Matching, Disparity)
│   ├── CDAF Pipeline (Contrast Analysis)
│   └── LDAF Pipeline (Distance Mapping)
├── Hybrid Decision Making
├── Temporal Filtering (Kalman-like)
└── Lens Control (VCM actuation)

Output:
└── Lens Position Commands → Focused Image
```

## 📁 File Structure

```
Autofocus/
├── autofocus_system.py      # Main AF system implementation
├── af_calibration.py        # Calibration and testing utilities
├── reference.txt           # Reference documentation
└── README.md              # This file
```

## 🚀 Quick Start

### Basic Usage

```python
from autofocus_system import AutoFocusSystem, AFConfig, AFMethod

# Create configuration
config = AFConfig(
    pdaf_min_confidence=0.6,
    cdaf_step_size=20,
    hybrid_pdaf_weight=0.6,
    hybrid_cdaf_weight=0.3,
    hybrid_ldaf_weight=0.1
)

# Initialize AF system
af_system = AutoFocusSystem(config)

# Trigger autofocus
result = af_system.trigger_autofocus(
    image=camera_image,
    left_pd=left_pd_pixels,      # Optional PDAF data
    right_pd=right_pd_pixels,    # Optional PDAF data
    ldaf_distance=2.5,           # Optional LDAF distance
    ldaf_confidence=0.9,         # Optional LDAF confidence
    touch_point=(300, 200),      # Optional touch point
    faces=[(100, 50, 80, 100)],  # Optional face detections
    method=AFMethod.HYBRID
)

print(f"AF Success: {result.success}")
print(f"Final Position: {result.final_position}")
print(f"Convergence Time: {result.convergence_time:.3f}s")
```

### Running the Demo

```bash
python autofocus_system.py
```

This will run a comprehensive demonstration showing:
- PDAF with known disparity
- CDAF with different blur levels  
- Hybrid AF combining all methods
- ROI selection with faces and touch points

## 🔧 Core Components

### 1. ROI Selector (`ROISelector`)

Intelligently selects the best area for focusing using multiple strategies:

- **Touch Priority**: User-specified point gets highest priority
- **Face Detection**: Automatically focuses on detected faces
- **Saliency Analysis**: Uses edge density to find interesting regions
- **Center Fallback**: Default center-weighted region

```python
roi = roi_selector.select_roi(
    image=camera_image,
    touch_point=(x, y),           # Optional
    faces=[(x, y, w, h), ...]     # Optional
)
```

### 2. PDAF Processor (`PDAFProcessor`)

Implements phase detection autofocus using block matching:

- **Block Matching Algorithm (BMA)**: Compares left/right PD pixel blocks
- **SAD Cost Function**: Sum of Absolute Differences for matching
- **Sub-pixel Accuracy**: Parabolic fitting for precise disparity
- **Outlier Rejection**: Robust estimation using median filtering

```python
pdaf_result = pdaf_processor.process_pdaf(
    left_pd=left_pd_pixels,
    right_pd=right_pd_pixels,
    roi=selected_roi
)
```

### 3. CDAF Processor (`CDAFProcessor`)

Implements contrast detection autofocus:

- **Laplacian Variance**: Measures image sharpness
- **Trend Analysis**: Determines optimal search direction
- **Adaptive Step Size**: Reduces step size when oscillating
- **History Tracking**: Uses previous measurements for guidance

```python
cdaf_result = cdaf_processor.process_cdaf(
    image=camera_image,
    roi=selected_roi,
    current_position=lens_position,
    step_size=20
)
```

### 4. LDAF Processor (`LDAFProcessor`)

Implements laser detection autofocus:

- **Distance Mapping**: Converts laser distance to lens position
- **Confidence Filtering**: Only uses high-confidence measurements
- **Range Validation**: Ensures distance is within sensor range
- **Lookup Table**: Fast distance-to-position conversion

```python
ldaf_result = ldaf_processor.process_ldaf(
    distance=2.5,      # meters
    confidence=0.9
)
```

## 🎛️ Configuration Options

### AFConfig Parameters

```python
@dataclass
class AFConfig:
    # PDAF parameters
    pdaf_block_size: int = 8              # Block size for matching
    pdaf_search_range: int = 16           # Search range (pixels)
    pdaf_min_confidence: float = 0.6      # Minimum confidence threshold
    pdaf_convergence_threshold: float = 0.5  # Convergence threshold
    
    # CDAF parameters  
    cdaf_step_size: int = 20              # Initial step size
    cdaf_max_steps: int = 50              # Maximum search steps
    cdaf_contrast_threshold: float = 0.05 # Contrast threshold
    
    # LDAF parameters
    ldaf_confidence_threshold: float = 0.8  # LDAF confidence threshold
    ldaf_distance_range: Tuple[float, float] = (0.1, 10.0)  # Valid range (m)
    
    # Hybrid weights
    hybrid_pdaf_weight: float = 0.6       # PDAF contribution weight
    hybrid_cdaf_weight: float = 0.3       # CDAF contribution weight  
    hybrid_ldaf_weight: float = 0.1       # LDAF contribution weight
    
    # General parameters
    max_iterations: int = 15               # Maximum AF iterations
    temporal_alpha: float = 0.7           # Temporal filtering coefficient
    roi_size: Tuple[int, int] = (200, 200)  # Default ROI size
```

## 📊 Calibration and Testing

The system includes comprehensive calibration and testing utilities:

### PDAF Calibration

```python
from af_calibration import PDAFCalibrator

calibrator = PDAFCalibrator(config)
result = calibrator.calibrate_with_target(
    target_distance=2.0,              # Known distance (meters)
    lens_positions=lens_positions,    # Lens positions (DAC)
    pdaf_images=image_pairs           # (left_pd, right_pd) pairs
)

print(f"Calibration slope: {result.slope}")
print(f"Calibration offset: {result.offset}")
print(f"RMSE: {result.rmse}")
```

### Performance Testing

```python
from af_calibration import AFPerformanceTester

tester = AFPerformanceTester(config)
results = tester.test_accuracy(af_system, test_scenarios)

for scenario, metrics in results.items():
    print(f"{scenario}: Success={metrics.success_rate:.1%}, "
          f"Accuracy={metrics.accuracy:.1f} DAC")
```

### Running Comprehensive Tests

```bash
python af_calibration.py
```

## 🔬 Algorithm Details

### Block Matching Algorithm (PDAF)

1. **Extract PD Pixels**: Separate left and right phase detection pixels
2. **Block Division**: Divide ROI into 8×8 pixel blocks
3. **SAD Calculation**: Compute Sum of Absolute Differences for each disparity
4. **Sub-pixel Refinement**: Use parabolic fitting for precise disparity
5. **Outlier Rejection**: Apply median filtering to remove bad matches
6. **Weighted Average**: Combine results using confidence weights

**Mathematical Formulation:**
```
SAD(d) = Σ |L(x,y) - R(x+d,y)|
d_subpixel = d_int + (C[d-1] - C[d+1]) / (2*(C[d-1] - 2*C[d] + C[d+1]))
```

### Contrast Detection (CDAF)

1. **ROI Extraction**: Extract region of interest from image
2. **Laplacian Filter**: Apply Laplacian operator for edge detection
3. **Variance Calculation**: Compute variance as sharpness measure
4. **Trend Analysis**: Determine if contrast is improving or degrading
5. **Step Size Adaptation**: Adjust step size based on convergence behavior

**Mathematical Formulation:**
```
Contrast = Var(Laplacian(I))
Laplacian(I) = ∂²I/∂x² + ∂²I/∂y²
```

### Hybrid Decision Making

The system combines multiple AF methods using weighted fusion:

```python
def decide_movement(self, method):
    movements = []
    confidences = []
    
    # Collect from PDAF
    if pdaf_confidence > threshold:
        movements.append(pdaf_defocus)
        confidences.append(pdaf_confidence * pdaf_weight)
    
    # Collect from CDAF  
    if cdaf_confidence > threshold:
        movements.append(cdaf_movement)
        confidences.append(cdaf_confidence * cdaf_weight)
        
    # Collect from LDAF
    if ldaf_confidence > threshold:
        movements.append(ldaf_movement)
        confidences.append(ldaf_confidence * ldaf_weight)
    
    # Weighted fusion
    final_movement = np.average(movements, weights=confidences)
    return final_movement
```

## 🎯 Key Features

### ✅ Implemented Features

- **Multi-method AF**: PDAF, CDAF, LDAF, and Hybrid approaches
- **Intelligent ROI Selection**: Touch, face, saliency, and center-based
- **Robust Block Matching**: SAD with sub-pixel accuracy and outlier rejection
- **Adaptive Contrast Detection**: Trend analysis and step size adaptation
- **Temporal Filtering**: Smooth convergence and hunting prevention
- **Comprehensive Calibration**: Factory calibration with quality metrics
- **Performance Testing**: Accuracy, repeatability, and speed evaluation
- **Configurable Parameters**: Extensive tuning options for different scenarios

### 🔄 State Machine

The AF system follows a clear state machine:

```
IDLE → SEARCHING → CONVERGED
  ↓        ↓           ↑
  └──→ FAILED ←────────┘
       ↓
    HUNTING
```

- **IDLE**: System ready for AF trigger
- **SEARCHING**: Actively searching for focus
- **CONVERGED**: Successfully focused
- **FAILED**: Failed to converge within iterations
- **HUNTING**: Oscillating behavior detected

## 📈 Performance Characteristics

### Typical Performance Metrics

| Method | Speed | Accuracy | Low Light | High Contrast |
|--------|-------|----------|-----------|---------------|
| PDAF   | Fast  | Good     | Poor      | Excellent     |
| CDAF   | Slow  | Excellent| Good      | Good          |
| LDAF   | Fast  | Good     | Excellent | Good          |
| Hybrid | Medium| Excellent| Good      | Excellent     |

### Convergence Times

- **PDAF**: 50-200ms (1-3 iterations)
- **CDAF**: 200-1000ms (5-15 iterations)  
- **LDAF**: 20-50ms (1 iteration)
- **Hybrid**: 100-400ms (2-8 iterations)

## 🔧 Hardware Integration

### VCM Control Interface

```python
class VCMController:
    def __init__(self, i2c_address=0x18):
        self.i2c_address = i2c_address
    
    def move_lens(self, position: int):
        """Move lens to specified position (0-1023 DAC)"""
        # I2C communication with VCM driver
        self.i2c_write(self.i2c_address, 0x01, position)
    
    def get_position(self) -> int:
        """Read current lens position"""
        return self.i2c_read(self.i2c_address, 0x02)
```

### Sensor Interface

```python
class ImageSensor:
    def get_bayer_raw(self) -> np.ndarray:
        """Get Bayer raw image data"""
        return self.camera.capture_raw()
    
    def get_pd_pixels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get left and right PD pixel arrays"""
        return self.camera.get_pd_data()
    
    def get_isp_stats(self) -> Dict:
        """Get ISP statistics (contrast, luminance, etc.)"""
        return self.camera.get_statistics()
```

## 🎓 Educational Value

This implementation serves as an excellent educational resource for:

- **Computer Vision Students**: Understanding autofocus algorithms
- **Mobile Camera Engineers**: Learning modern AF system architecture  
- **Image Processing Researchers**: Studying phase detection and contrast analysis
- **Embedded Systems Developers**: Hardware-software integration patterns

### Key Learning Points

1. **Phase Detection Theory**: How PD pixels enable fast autofocus
2. **Block Matching Algorithms**: SAD, SSD, and correlation methods
3. **Contrast Maximization**: Laplacian variance and sharpness metrics
4. **Sensor Fusion**: Combining multiple measurement sources
5. **Temporal Filtering**: Preventing oscillation and hunting
6. **Calibration Procedures**: Factory calibration and validation
7. **Performance Optimization**: Speed vs. accuracy tradeoffs

## 🚀 Future Enhancements

### Planned Features

- **Machine Learning PDAF**: Neural network-based disparity estimation
- **Multi-depth Handling**: Scene with multiple focus distances
- **Temperature Compensation**: Adjusting for thermal effects
- **Motion Prediction**: Using IMU data for motion blur prevention
- **Full-frame PDAF**: Generating complete depth maps
- **Real-time Visualization**: Live AF performance monitoring

### Research Directions

- **Deep Learning Integration**: CNN-based focus quality assessment
- **Computational Photography**: Multi-frame AF and focus stacking
- **Edge Computing**: On-device AI for intelligent AF decisions
- **Sensor Fusion**: Advanced multi-modal data integration

## 📚 References

This implementation is based on:

1. **Modern AF System**: Mobile camera autofocus architecture
2. **Phase Detection Theory**: Optical phase difference measurement
3. **Block Matching Algorithms**: Computer vision stereo matching
4. **Contrast Detection**: Image sharpness and focus quality metrics
5. **Mobile Computational Photography**: Modern smartphone camera systems

## 📄 License

This educational implementation is provided for learning and research purposes. Please cite appropriately if used in academic work.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Algorithm optimizations
- Additional AF methods
- Performance benchmarking
- Hardware integration examples
- Documentation improvements

---

*This autofocus system demonstrates the sophisticated algorithms and engineering principles used in modern mobile cameras, providing a comprehensive foundation for understanding computational photography and computer vision applications.*
