# Clean Autofocus System

This is a cleaned-up, focused autofocus system that integrates raw data reading with PDAF and CDAF calculations.

## Files Overview

### Core Files

1. **`raw_data_reader.py`** - Raw data reading functions
   - `unpack_raw10()` - Efficient RAW10 unpacking using your provided function
   - `read_pd_raw()` - Read and split PD RAW10 data into left/right images
   - `read_bayer_raw()` - Read Bayer RAW10 data
   - `demosaic_bayer()` - Demosaic Bayer to RGB
   - `save_image_8bit()` - Save images with proper contrast scaling

2. **`autofocus_processor.py`** - Main autofocus processor
   - `AutofocusProcessor` class - Integrates raw readers with AF calculations
   - `calculate_pdaf()` - PDAF calculation from left/right PD images
   - `calculate_cdaf()` - CDAF calculation from Bayer image
   - `calculate_hybrid_af()` - Combines PDAF and CDAF
   - `process_complete_af()` - Complete pipeline processing

3. **`autofocus_system.py`** - Core autofocus algorithms (existing)
   - PDAF, CDAF, and LDAF processors
   - ROI selection and configuration

4. **`example_usage.py`** - Simple usage examples
   - Step-by-step processing
   - Complete pipeline example

## Usage

### Quick Start

```python
from autofocus_processor import AutofocusProcessor

# Initialize processor
processor = AutofocusProcessor()

# Data parameters
pd_file = "data/your_pd_file.raw"
pd_params = (width, height, stride)

bayer_file = "data/your_bayer_file.raw" 
bayer_params = (width, height, stride)

# Run complete autofocus processing
results = processor.process_complete_af(
    pd_file, pd_params,
    bayer_file, bayer_params,
    roi=(1500, 1000, 1000, 1000)  # Optional ROI
)

print(f"Focus direction: {results['focus_direction']}")
print(f"Hybrid confidence: {results['hybrid_confidence']}")
```

### Step-by-Step Processing

```python
from raw_data_reader import read_pd_raw, read_bayer_raw
from autofocus_processor import AutofocusProcessor

# Read raw data
left_pd, right_pd = read_pd_raw(pd_file, width, height, stride)
bayer_image = read_bayer_raw(bayer_file, width, height, stride)

# Calculate autofocus
processor = AutofocusProcessor()
pdaf_results = processor.calculate_pdaf(left_pd, right_pd)
cdaf_results = processor.calculate_cdaf(bayer_image)
```

## Data Processing

### PD Data Processing
- **Input**: RAW10 PD file (4032x756, stride 5056)
- **Processing**: Unpack RAW10 → Split into left/right using 2x1 column blocks pattern
- **Output**: Left PD (756x2016), Right PD (756x2016)

### Bayer Data Processing  
- **Input**: RAW10 Bayer file (4032x3024, stride 5120)
- **Processing**: Unpack RAW10 → Bayer image
- **Output**: Bayer image (3024x4032)

### Autofocus Results
- **PDAF**: Disparity calculation, defocus estimation, confidence
- **CDAF**: Contrast measurement, focus direction, confidence  
- **Hybrid**: Combined PDAF + CDAF with weighted averaging

## Key Features

✅ **Clean Architecture**: Separated raw data reading from autofocus calculations  
✅ **Efficient RAW10 Unpacking**: Uses your provided optimized function  
✅ **Proper PD Splitting**: Correct 2x1 column blocks pattern  
✅ **Better Contrast**: Uses actual data range for image scaling  
✅ **ROI Support**: Region of interest selection for focused analysis  
✅ **Hybrid AF**: Combines PDAF and CDAF for robust focusing  
✅ **Error Handling**: Comprehensive error checking and validation  

## Test Results

The system successfully processes real data:
- PD data: 4032x756 → Left/Right (756x2016 each)
- Bayer data: 4032x3024 → Bayer image (3024x4032)
- PDAF: Disparity calculation with defocus estimation
- CDAF: Contrast measurement with focus direction
- Hybrid: Combined results with confidence weighting

## Running the Examples

```bash
# Activate virtual environment
source ../virtual_env/bin/activate

# Run complete example
python example_usage.py

# Run raw data reader only
python raw_data_reader.py

# Run autofocus processor only  
python autofocus_processor.py
```

## Output Files

The system generates:
- `data/example_left_pd.jpg` - Left PD image (proper contrast)
- `data/example_right_pd.jpg` - Right PD image (proper contrast)
- `data/example_bayer.jpg` - Bayer image (proper contrast)

All images are saved with proper 8-bit scaling using actual data range for optimal contrast.
