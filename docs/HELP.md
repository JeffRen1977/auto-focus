# Autofocus System - Quick Help Guide

## 🚀 Quick Start Commands

### Setup (First Time Only)

**Note**: This project uses a virtual environment located in the parent directory at `../virtual_env`

```bash
# Linux/Mac
chmod +x setup.sh
./setup.sh

# Windows
setup.bat
```

### Run Demos
```bash
# Linux/Mac (Interactive Menu)
./run_demo.sh

# Windows (Interactive Menu)
run_demo.bat

# Direct Python Execution
source ../virtual_env/bin/activate  # Activate environment
python simple_example.py            # Educational demo
python autofocus_system.py          # Full system demo
python af_calibration.py            # Testing demo
python af_visualization.py          # Visualization demo
```

### Quick Test
```bash
python quick_start.py
```

## 📋 Available Scripts

### Setup Scripts
- **`setup.sh`** (Linux/Mac) - Automated setup with virtual environment
- **`setup.bat`** (Windows) - Windows setup script
- **`requirements.txt`** - Python dependencies

### Demo Scripts
- **`run_demo.sh`** (Linux/Mac) - Interactive demo runner
- **`run_demo.bat`** (Windows) - Windows demo runner
- **`quick_start.py`** - Quick validation test

### Main System Files
- **`autofocus_system.py`** - Complete autofocus implementation
- **`simple_example.py`** - Educational examples
- **`af_calibration.py`** - Calibration and testing
- **`af_visualization.py`** - Visualization tools

### Documentation
- **`README.md`** - Comprehensive documentation
- **`SETUP_GUIDE.md`** - Detailed setup instructions
- **`HELP.md`** - This quick help guide

## 🎯 Demo Descriptions

### 1. Simple Example (`simple_example.py`)
- **Best for**: Beginners and learning
- **Shows**: Basic AF concepts, algorithm explanations
- **Duration**: ~2-3 minutes
- **Output**: Educational explanations and results

### 2. Full System Demo (`autofocus_system.py`)
- **Best for**: Understanding complete system
- **Shows**: PDAF, CDAF, LDAF, Hybrid AF
- **Duration**: ~5-10 minutes
- **Output**: Detailed AF performance results

### 3. Calibration Demo (`af_calibration.py`)
- **Best for**: Advanced users and testing
- **Shows**: Calibration procedures, performance testing
- **Duration**: ~10-15 minutes
- **Output**: Calibration results and metrics

### 4. Visualization Demo (`af_visualization.py`)
- **Best for**: Understanding algorithms visually
- **Shows**: Disparity maps, contrast curves, convergence
- **Duration**: ~5 minutes
- **Output**: Plots and visualizations

## 🔧 Environment Management

### Activate Virtual Environment
```bash
# Linux/Mac
source ../virtual_env/bin/activate

# Windows
..\virtual_env\Scripts\activate
```

### Deactivate Virtual Environment
```bash
deactivate
```

### Virtual Environment Location
The virtual environment is located in the parent directory: `../virtual_env/`

### Check Dependencies
```bash
python quick_start.py --check-deps
```

## 🐛 Troubleshooting

### Common Issues
1. **Python not found**: Install Python 3.7+ from python.org
2. **Permission denied**: Run `chmod +x *.sh` (Linux/Mac)
3. **Import errors**: Activate virtual environment first
4. **Matplotlib issues**: Set `MPLBACKEND=Agg` for headless systems

### Verification Commands
```bash
python --version                    # Check Python version
pip list                           # List installed packages
python -c "import numpy, cv2"      # Test imports
python quick_start.py --check-deps # Full dependency check
```

## 📚 Learning Path

### For Students
1. Read `README.md` overview
2. Run `simple_example.py`
3. Study algorithm explanations
4. Try `af_visualization.py`

### For Engineers
1. Run `autofocus_system.py`
2. Study implementation details
3. Test with `af_calibration.py`
4. Modify parameters in `AFConfig`

### For Researchers
1. Analyze algorithm performance
2. Extend with new AF methods
3. Integrate with real hardware
4. Publish improvements

## 🎓 Key Concepts Demonstrated

- **PDAF**: Phase Detection Auto Focus using block matching
- **CDAF**: Contrast Detection Auto Focus using Laplacian variance
- **LDAF**: Laser Detection Auto Focus using distance measurement
- **Hybrid AF**: Intelligent combination of multiple methods
- **ROI Selection**: Smart area selection for focusing
- **Temporal Filtering**: Smooth convergence and stability

## 📊 Expected Performance

- **PDAF**: Fast (50-200ms), Good accuracy, Poor in low light
- **CDAF**: Slow (200-1000ms), Excellent accuracy, Good in low light
- **LDAF**: Very fast (20-50ms), Good accuracy, Excellent in low light
- **Hybrid**: Medium speed (100-400ms), Excellent accuracy, Robust

## 🆘 Getting Help

1. **Check this guide** for quick solutions
2. **Read SETUP_GUIDE.md** for detailed instructions
3. **Read README.md** for comprehensive documentation
4. **Run verification commands** to diagnose issues

---

*This help guide provides quick access to all commands and demos. For detailed information, see the full documentation files.*
