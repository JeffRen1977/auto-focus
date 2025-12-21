# Chapter 5: Autofocus System Implementation

## Abstract

This chapter details the implementation of a modern autofocus system for mobile cameras, leveraging both Phase Detection Auto Focus (PDAF) and Contrast Detection Auto Focus (CDAF). The system processes real RAW10 sensor data, including phase detection and Bayer images, to achieve fast and accurate focusing. It integrates a hybrid approach that intelligently selects between PDAF and CDAF based on confidence thresholds, ensuring optimal performance across various lighting and scene conditions. Key components include a modular architecture with separate processors for PDAF, CDAF, ROI selection, and system integration, all working together to deliver robust and efficient autofocus solutions for mobile devices.

**Keywords:** Autofocus, PDAF, CDAF, Mobile Camera, RAW10, Sensor Data, Image Processing, Hybrid AF, Region of Interest (ROI), Camera System

**Repository:** https://github.com/JeffRen1977/auto-focus

---

## 5.1 Overview

This system implements a complete autofocus pipeline designed for educational purposes and practical application in mobile computational photography. The implementation demonstrates:

- **Real RAW10 sensor data processing** from mobile camera sensors
- **Phase detection autofocus (PDAF)** using dedicated PD pixels for rapid initial focusing
- **Contrast detection autofocus (CDAF)** for accurate fine-tuning
- **Hybrid AF strategy** that intelligently selects the best method based on confidence
- **Modular architecture** that separates concerns for clarity and learning

The system architecture is organized into logical modules that follow a natural learning progression:

```
Autofocus/
├── core/                  # Core AF algorithms and system integration
│   ├── config.py         # Configuration and data structures
│   ├── roi_selection.py  # ROI selection logic
│   ├── pdaf.py           # Phase Detection AF
│   ├── cdaf.py           # Contrast Detection AF
│   ├── ldaf.py           # Laser Detection AF (optional)
│   └── autofocus_system.py  # Main system controller
├── data_processing/      # Raw sensor data processing
│   └── raw_reader.py     # RAW10 format handling
├── autofocus_processor.py  # High-level processing interface
└── examples/             # Usage examples
```

---

## 5.2 System Architecture

### 5.2.1 Processing Pipeline

The autofocus system follows a clear data flow from raw sensor data to focus commands:

```
Raw Sensor Data (RAW10 format)
    ↓
[1] Unpack RAW10 → uint16 arrays
    ├── PD data: 4032×756 → Unpacked uint16
    └── Bayer data: 4032×3024 → Unpacked uint16
    ↓
[2] Process PD Data
    ├── Split into Left/Right views (2x1 column blocks pattern)
    ├── Left PD: 756×2016 (columns 0-1, 4-5, 8-9, ...)
    └── Right PD: 756×2016 (columns 2-3, 6-7, 10-11, ...)
    ↓
[3] Calculate Autofocus
    ├── PDAF: Block matching → Disparity → Defocus estimate
    ├── CDAF: Laplacian variance → Contrast → Focus direction
    └── Hybrid: Confidence-based method selection
    ↓
[4] Output Focus Command
    └── Lens position adjustment → Sharp image
```

### 5.2.2 Module Organization

The codebase is structured for progressive learning:

1. **Configuration** (`core/config.py`): Understand system parameters and data structures
2. **ROI Selection** (`core/roi_selection.py`): Learn how focus regions are selected
3. **PDAF** (`core/pdaf.py`): Fast hardware-assisted focusing
4. **CDAF** (`core/cdaf.py`): Accurate contrast-based focusing
5. **System Integration** (`core/autofocus_system.py`): How components work together
6. **Data Processing** (`data_processing/raw_reader.py`): Raw sensor data handling

---

## 5.3 Core Components

### 5.3.1 Raw Data Reader (`data_processing/raw_reader.py`)

The raw data reader handles the low-level processing of RAW10 sensor data, the standard format for Android camera systems.

#### RAW10 Format

RAW10 is a packed 10-bit image format that efficiently stores sensor data:
- Each pixel is 10 bits (0-1023 range)
- Data is packed: 4 pixels per 5 bytes
- Format: `[P0_high P1_high P2_high P3_high P0_low|P1_low|P2_low|P3_low]`
- The 5th byte contains the 2 LSBs of each of the 4 pixels

#### Unpacking Algorithm

```python
def unpack_raw10(data: bytes, width: int, height: int, stride: int) -> np.ndarray:
    """
    Efficiently unpacks RAW10 data to uint16 arrays.
    
    Algorithm:
    1. Reshape data accounting for row stride (memory alignment)
    2. Group into sets of 5 bytes (4 pixels per group)
    3. Extract high 8 bits from bytes 0-3
    4. Extract low 2 bits from byte 4
    5. Combine: pixel_value = (high_8_bits << 2) | low_2_bits
    """
    # Implementation details in raw_reader.py
```

**Key Features:**
- Handles stride alignment (padding for memory optimization)
- Efficient NumPy-based unpacking
- Validates data size and format
- Outputs uint16 arrays (10-bit data stored in 16-bit format)

#### PD Pixel Pattern Splitting

PD sensors use a **2x1 column blocks pattern**:
- Pattern: `[L L R R L L R R ...]` (repeating every 4 columns)
- Left PD view: columns 0-1, 4-5, 8-9, ... → captures left perspective
- Right PD view: columns 2-3, 6-7, 10-11, ... → captures right perspective

This pattern creates two views with slight angular differences, enabling phase difference calculation.

#### Usage Example

```python
from data_processing.raw_reader import read_pd_raw, read_bayer_raw

# Read and split PD data
left_pd, right_pd = read_pd_raw(
    "data/pd_file.raw",
    width=4032, height=756, stride=5056
)

# Read Bayer data
bayer_image = read_bayer_raw(
    "data/bayer_file.raw",
    width=4032, height=3024, stride=5120
)
```

---

### 5.3.2 Phase Detection Auto Focus (`core/pdaf.py`)

PDAF is a fast, hardware-assisted autofocus method that uses dedicated phase detection pixels on the sensor.

#### How PDAF Works

1. **Sensor Setup**: Dedicated PD pixels arranged in left/right pattern
2. **View Extraction**: Left and right PD images extracted from sensor
3. **Block Matching**: Corresponding blocks found between views
4. **Disparity Calculation**: Horizontal shift (disparity) measured
5. **Defocus Conversion**: Disparity converted to lens movement needed
6. **Focus Adjustment**: Lens moved to correct position

#### Block Matching Algorithm (BMA)

The PDAF processor uses Block Matching with Sum of Absolute Differences (SAD):

```python
class PDAFProcessor:
    def process_pdaf(self, left_pd, right_pd, roi_data):
        """
        Calculate disparity using Block Matching Algorithm.
        
        Algorithm:
        1. Extract ROI from left and right PD images
        2. For each block in left image:
           a. Search corresponding block in right image (horizontal search)
           b. Calculate SAD for each candidate position
           c. Find position with minimum SAD (best match)
        3. Calculate disparity = matched_position - original_position
        4. Apply robust statistics (median + weighted average) for outlier rejection
        5. Convert disparity to defocus using calibration parameters
        """
```

**Key Parameters:**
- `pdaf_block_size`: Size of matching blocks (default: 8×8 pixels)
- `pdaf_search_range`: Horizontal search range in pixels (default: ±16 pixels)
- `pdaf_min_confidence`: Minimum confidence threshold (default: 0.6)

**Robust Estimation:**
- Normalizes blocks for illumination invariance
- Uses median filtering to reject outliers
- Applies weighted average for final disparity estimate
- Provides confidence score based on match quality

**Calibration:**
Disparity (pixels) must be converted to defocus (lens movement steps):
```
defocus = disparity × slope + offset
```

These parameters are determined through lens characterization tests.

#### Advantages and Limitations

**Advantages:**
- Very fast (typically < 200ms)
- Hardware-assisted (dedicated pixels)
- Provides both direction and magnitude
- Works well in good lighting conditions

**Limitations:**
- Requires good contrast and texture
- Limited by PD pixel coverage
- May fail in low-light or low-contrast scenes
- Needs calibration for accurate defocus conversion

---

### 5.3.3 Contrast Detection Auto Focus (`core/cdaf.py`)

CDAF is an accurate autofocus method that measures image sharpness by analyzing contrast.

#### How CDAF Works

1. **ROI Extraction**: Extract region of interest from image
2. **Edge Detection**: Apply Laplacian filter to detect edges
3. **Contrast Measurement**: Compute variance of Laplacian as sharpness metric
4. **Trend Analysis**: Track contrast changes as lens position changes
5. **Peak Finding**: Locate lens position with maximum contrast

#### Laplacian Variance Method

Sharp images have more edges (high-frequency content). The Laplacian operator detects edges by finding second derivatives:

```python
class CDAFProcessor:
    def process_cdaf(self, image, roi_data, current_position, step_size):
        """
        Calculate contrast using Laplacian variance.
        
        Algorithm:
        1. Extract ROI from image
        2. Convert to grayscale if needed
        3. Apply Laplacian operator:
           Laplacian = ∂²I/∂x² + ∂²I/∂y²
        4. Compute variance of Laplacian response:
           contrast = var(Laplacian(image))
        5. Higher variance = more edges = sharper image = better focus
        """
```

**Key Insight:** Sharp images have strong, well-defined edges, resulting in high Laplacian variance. Blurry images have smooth transitions, resulting in low variance.

#### Search Strategy

CDAF requires iterative search to find the focus peak:

- **Initial Phase**: Large steps to find approximate focus region
- **Improving**: Continue in same direction with same step size
- **Getting Worse**: Reverse direction with smaller steps
- **Oscillating**: Reduce step size for fine-tuning near peak

**Parameters:**
- `cdaf_step_size`: Initial lens movement step size (default: 20 steps)
- `cdaf_max_steps`: Maximum iterations (default: 50)
- `cdaf_contrast_threshold`: Minimum contrast change (default: 0.05)

#### Advantages and Limitations

**Advantages:**
- Very accurate (can achieve precise focus)
- Works with any image content
- Reliable in various lighting conditions
- Good for fine-tuning after initial positioning

**Limitations:**
- Slower than PDAF (requires multiple lens movements)
- May "hunt" if contrast is low
- Requires iterative search (hill-climbing)
- Can be affected by noise in low-light

---

### 5.3.4 Hybrid AF Controller (`autofocus_processor.py`)

The hybrid AF system intelligently combines PDAF and CDAF by selecting the most appropriate method based on confidence.

#### Decision Logic

The system uses a **threshold-based selection** rather than weighted combination:

```python
def calculate_hybrid_af(self, left_pd, right_pd, bayer_image, roi):
    """
    Hybrid AF with confidence-based method selection.
    
    Strategy:
    - PDAF confidence >= 0.4: Use PDAF-only (fast, hardware-assisted)
    - PDAF confidence < 0.4: Switch to CDAF-only (accurate fallback)
    
    This avoids blending unreliable PDAF data with CDAF.
    """
    pdaf_results = self.calculate_pdaf(left_pd, right_pd, roi)
    cdaf_results = self.calculate_cdaf(bayer_image, roi)
    
    pdaf_confidence = pdaf_results['confidence'] or 0.0
    
    PDAF_CONFIDENCE_THRESHOLD = 0.4
    
    if pdaf_confidence >= PDAF_CONFIDENCE_THRESHOLD:
        # Trust PDAF: fast and accurate when reliable
        return pdaf_results  # PDAF-only
    else:
        # Fallback to CDAF: more reliable when PDAF fails
        return cdaf_results  # CDAF-only
```

**Rationale:**
- PDAF is preferred when reliable (confidence ≥ 0.4) because it's fast and provides precise defocus estimates
- CDAF is used as fallback when PDAF is unreliable (low confidence), ensuring robust performance
- Binary selection avoids blending potentially unreliable data, maintaining accuracy

#### Usage Example

```python
from autofocus_processor import AutofocusProcessor

processor = AutofocusProcessor()

# Individual calculations
pdaf_results = processor.calculate_pdaf(left_pd, right_pd, roi)
cdaf_results = processor.calculate_cdaf(bayer_image, roi)

# Hybrid combination (automatically selects best method)
hybrid_results = processor.calculate_hybrid_af(
    left_pd, right_pd, bayer_image, roi
)

print(f"Method used: {hybrid_results['method_used']}")  # "PDAF_ONLY" or "CDAF_ONLY"
print(f"Defocus: {hybrid_results['hybrid_defocus']:.3f}")
print(f"Direction: {hybrid_results['focus_direction']}")
```

---

### 5.3.5 Core AF System (`core/autofocus_system.py`)

The `AutoFocusSystem` class orchestrates all AF methods and implements a state machine for iterative refinement.

#### System Architecture

```python
class AutoFocusSystem:
    """
    Main Autofocus System Controller.
    
    Coordinates:
    - ROI selection (touch > face > saliency > center)
    - Multiple AF methods (PDAF, CDAF, LDAF, Hybrid)
    - Temporal filtering (exponential moving average)
    - Convergence detection
    - State machine management
    """
```

#### State Machine

The system operates through these states:

- **IDLE**: System ready, waiting for autofocus trigger
- **SEARCHING**: Actively searching for optimal focus position
- **CONVERGED**: Focus achieved, within tolerance threshold
- **FAILED**: Maximum iterations reached without convergence
- **HUNTING**: Oscillating around focus point (not converged)

#### Temporal Filtering

Uses exponential moving average to smooth lens movements:
```
filtered_movement = α × new_movement + (1-α) × previous_movement
```

This prevents:
- Focus hunting (oscillations around focus point)
- Jittery movements
- Overcorrection

#### ROI Selection

The system supports multiple ROI selection methods with priority:
1. **Touch ROI**: User-selected region (highest priority)
2. **Face Detection**: Automatic face detection
3. **Saliency**: Computational saliency detection
4. **Center**: Fallback to center region

---

## 5.4 Real Data Processing

### 5.4.1 PD Data Processing Pipeline

**Input:** PD RAW10 file (e.g., 4032×756, stride 5056, ~3.8MB)

```
Step 1: Unpack RAW10
    Input: Packed bytes (3.8MB)
    Output: uint16 array (4032×756)
    Time: ~50ms

Step 2: Split into Left/Right Views
    Pattern: 2x1 column blocks [L L R R ...]
    Left PD:  756×2016 (columns 0-1, 4-5, 8-9, ...)
    Right PD: 756×2016 (columns 2-3, 6-7, 10-11, ...)
    Time: ~10ms

Step 3: PDAF Calculation
    Block matching → Disparity map → Defocus estimate
    Time: ~200ms
```

**Output:** Focus direction and magnitude (e.g., "MOVE_FORWARD" with defocus = -2.0 steps)

### 5.4.2 Bayer Data Processing Pipeline

**Input:** Bayer RAW10 file (e.g., 4032×3024, stride 5120, ~15.5MB)

```
Step 1: Unpack RAW10
    Input: Packed bytes (15.5MB)
    Output: uint16 array (4032×3024)
    Time: ~100ms

Step 2: CDAF Calculation
    Extract ROI → Laplacian filter → Variance → Contrast value
    Time: ~150ms
```

**Output:** Contrast value and focus direction (e.g., contrast = 148303.0, direction = "IN_FOCUS")

### 5.4.3 Hybrid AF Results

Example output from hybrid AF processing:

```
PDAF Results:
  Disparity: 0.000 pixels
  Defocus: -2.000 steps
  Confidence: 0.350

CDAF Results:
  Contrast: 148303.0
  Confidence: 0.300
  Direction: IN_FOCUS

Hybrid AF Results:
  Method used: CDAF_ONLY  (PDAF confidence 0.350 < 0.4 threshold)
  Hybrid defocus: 0.000
  Hybrid confidence: 0.300
  Focus direction: IN_FOCUS
```

---

## 5.5 Configuration

### 5.5.1 Data Parameters

```python
# PD data parameters
pd_file = "data/pd_file.raw"
pd_params = (4032, 756, 5056)  # width, height, stride

# Bayer data parameters  
bayer_file = "data/bayer_file.raw"
bayer_params = (4032, 3024, 5120)  # width, height, stride

# ROI (Region of Interest)
roi = (1500, 1000, 1000, 1000)  # x, y, width, height
```

### 5.5.2 AF Configuration

```python
from core import AFConfig

config = AFConfig(
    # PDAF parameters
    pdaf_block_size=8,           # Block size for matching (8×8 pixels)
    pdaf_search_range=16,        # Horizontal search range (±16 pixels)
    pdaf_min_confidence=0.6,     # Minimum confidence threshold
    
    # CDAF parameters
    cdaf_step_size=20,           # Initial lens movement step size
    cdaf_max_steps=50,           # Maximum iterations
    cdaf_contrast_threshold=0.05,  # Minimum contrast change
    
    # Hybrid AF threshold (in autofocus_processor.py)
    # PDAF confidence >= 0.4: Use PDAF-only
    # PDAF confidence < 0.4: Use CDAF-only
)
```

### 5.5.3 Calibration Data

For accurate PDAF defocus conversion, calibration parameters are required:

```python
calibration_data = {
    'slope': 1.2,      # Disparity to defocus conversion factor
    'offset': -0.5     # Offset for defocus calculation
}

processor = AutofocusProcessor(calibration_data=calibration_data)
```

---

## 5.6 Performance

### 5.6.1 Processing Performance

| Component | Input Size | Output Size | Processing Time |
|-----------|------------|-------------|-----------------|
| PD Unpacking | 3.8MB RAW10 | 756×4032 uint16 | ~50ms |
| PD Splitting | 756×4032 | 756×2016 (L/R) | ~10ms |
| Bayer Unpacking | 15.5MB RAW10 | 3024×4032 uint16 | ~100ms |
| PDAF Calculation | 756×2016 (L/R) | Disparity map | ~200ms |
| CDAF Calculation | 3024×4032 | Contrast value | ~150ms |
| Hybrid AF | Combined | Focus direction | ~50ms |

**Total Pipeline Time:** ~560ms (excluding I/O)

### 5.6.2 Quality Metrics

- **PDAF Accuracy**: Sub-pixel disparity estimation with robust statistics
- **CDAF Sensitivity**: Laplacian variance measurement with adaptive step sizing
- **Hybrid Robustness**: Confidence-based method selection ensures reliable performance
- **ROI Efficiency**: Focused analysis on relevant regions reduces processing time

### 5.6.3 System Characteristics

- **Speed**: PDAF provides fast initial focusing (< 200ms), hybrid system adds minimal overhead
- **Accuracy**: CDAF enables precise fine-tuning, PDAF provides magnitude estimates
- **Robustness**: Hybrid approach handles various lighting and scene conditions
- **Modularity**: Clean separation allows independent testing and optimization

---

## 5.7 Summary

This chapter presented a comprehensive autofocus system implementation for mobile computational photography. The system demonstrates:

1. **Real Sensor Data Processing**: Efficient handling of RAW10 format from mobile cameras
2. **PDAF Implementation**: Fast hardware-assisted focusing using block matching algorithm
3. **CDAF Implementation**: Accurate contrast-based focusing using Laplacian variance
4. **Hybrid Strategy**: Intelligent confidence-based method selection for optimal performance
5. **Modular Architecture**: Clean code organization enabling progressive learning and maintenance

The implementation serves as both a practical solution and an educational resource, clearly demonstrating how modern mobile camera autofocus systems combine multiple techniques to achieve fast, accurate, and robust focusing performance.

---

**Next Steps:**
- Experiment with different ROI selections for various scenes
- Tune calibration parameters for specific lenses
- Integrate with real-time camera pipeline
- Explore advanced topics (calibration, visualization) in `advanced/` directory

