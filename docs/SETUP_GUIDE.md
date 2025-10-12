# Autofocus System Setup Guide

This guide will help you set up and run the autofocus system in a virtual environment.

## 📂 Project Structure

The autofocus system uses a virtual environment located in the **parent directory**:

```
Mobile_Computational_photograph/
├── virtual_env/              # Virtual environment (parent directory)
└── Autofocus/                # Autofocus system (current directory)
    ├── autofocus_system.py
    ├── simple_example.py
    ├── setup.sh
    └── ...
```

## 🚀 Quick Setup (Recommended)

### Step 1: Verify Virtual Environment
The virtual environment should already exist at `../virtual_env`. If not, create it:
```bash
cd ..
python3 -m venv virtual_env
cd Autofocus
```

### Step 2: Make Scripts Executable
```bash
chmod +x setup.sh
chmod +x run_demo.sh
```

### Step 3: Run Setup Script
```bash
./setup.sh
```

This script will:
- ✅ Check Python 3 installation
- ✅ Verify virtual environment exists at `../virtual_env`
- 📦 Install all dependencies
- 🔍 Verify installation

### Step 4: Run Demos
```bash
./run_demo.sh
```

## 📋 Manual Setup

If you prefer manual setup or encounter issues with the automated script:

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Virtual environment at `../virtual_env`

### Step 1: Activate Virtual Environment
```bash
# Activate virtual environment (Linux/Mac)
source ../virtual_env/bin/activate

# OR on Windows
..\virtual_env\Scripts\activate
```

### Step 2: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python quick_start.py --check-deps
```

## 🎯 Running the System

### Quick Test
```bash
# Activate virtual environment first
source ../virtual_env/bin/activate

# Run quick test
python quick_start.py
```

### Individual Demos
```bash
# Make sure virtual environment is activated
source ../virtual_env/bin/activate

# Educational examples (start here!)
python simple_example.py

# Full autofocus system demo
python autofocus_system.py

# Calibration and testing
python af_calibration.py

# Visualizations
python af_visualization.py
```

### Interactive Demo Runner
```bash
./run_demo.sh
```

## 🐍 Python Script Usage

### Direct Python Execution
```bash
# Activate environment
source af_env/bin/activate

# Run any demo
python simple_example.py
python autofocus_system.py
python af_calibration.py
python af_visualization.py
```

### Quick Start Script
```bash
python quick_start.py              # Run quick test
python quick_start.py --check-deps # Check dependencies only
python quick_start.py --help       # Show help
```

## 🔧 Virtual Environment Management

### Activating Environment
```bash
# Linux/Mac
source ../virtual_env/bin/activate

# Windows
..\virtual_env\Scripts\activate
```

### Deactivating Environment
```bash
deactivate
```

### Location
The virtual environment is located in the parent directory:
```
../virtual_env/
```

## 📦 Dependencies

The system requires these Python packages:
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `opencv-python>=4.5.0` - Computer vision
- `matplotlib>=3.5.0` - Plotting and visualization

### Optional Dependencies
- `numba>=0.56.0` - JIT compilation for faster block matching
- `cupy>=10.0.0` - GPU acceleration (requires CUDA)

## 🖥️ Platform-Specific Instructions

### Linux/macOS
```bash
# Make scripts executable
chmod +x setup.sh run_demo.sh

# Run setup
./setup.sh

# Run demos
./run_demo.sh
```

### Windows (PowerShell)
```powershell
# Run setup
python setup.py

# Or manually:
python -m venv af_env
af_env\Scripts\activate
pip install -r requirements.txt
python quick_start.py
```

### Windows (Command Prompt)
```cmd
python -m venv af_env
af_env\Scripts\activate.bat
pip install -r requirements.txt
python quick_start.py
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Python Not Found
```bash
# Check Python installation
python3 --version

# If not installed, install Python 3.7+ from python.org
```

#### 2. Virtual Environment Issues
```bash
# Recreate environment in parent directory
cd ..
rm -rf virtual_env
python3 -m venv virtual_env
cd Autofocus
source ../virtual_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. OpenCV Installation Issues
```bash
# Try alternative installation
pip uninstall opencv-python
pip install opencv-python-headless
```

#### 4. Matplotlib Backend Issues
```bash
# Set backend for headless systems
export MPLBACKEND=Agg
python af_visualization.py
```

#### 5. Permission Issues (Linux/Mac)
```bash
# Fix script permissions
chmod +x setup.sh run_demo.sh

# Or run directly with bash
bash setup.sh
bash run_demo.sh
```

### Verification Commands

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Test imports
python -c "import numpy, cv2, matplotlib, scipy; print('All OK!')"

# Run dependency check
python quick_start.py --check-deps
```

## 📊 Expected Output

### Successful Setup
```
==========================================
Autofocus System Setup Script
==========================================
✅ Python 3.9.7 detected
🔧 Creating virtual environment: af_env
🚀 Activating virtual environment...
⬆️  Upgrading pip...
📦 Installing dependencies...
🔍 Verifying installation...
✅ All dependencies installed successfully!
   NumPy version: 1.21.0
   OpenCV version: 4.5.0
   Matplotlib available: 3.5.0
   SciPy available: 1.7.0

🎉 Setup completed successfully!
```

### Successful Quick Test
```
🚀 Autofocus System Quick Start
===============================
🔍 Checking dependencies...
   ✅ OpenCV: 4.5.0
   ✅ NumPy: 1.21.0
   ✅ SciPy: 1.7.0
   ✅ Matplotlib: 3.5.0
✅ All dependencies are available!

🚀 Running Quick Autofocus Test...
   ✅ Autofocus system imported successfully
   ✅ Configuration created
   ✅ Autofocus system initialized
   ✅ Test data created
   🔍 Testing PDAF...
      PDAF Result: Success=True, Position=562.1, Iterations=3
   🔍 Testing CDAF...
      CDAF Result: Success=True, Position=580.3, Iterations=5
   🔍 Testing Hybrid AF...
      Hybrid Result: Success=True, Position=575.2, Iterations=4

🎉 Quick test completed successfully!
```

## 🎓 Learning Path

### For Beginners
1. **Start with**: `python simple_example.py`
2. **Read**: `README.md` for system overview
3. **Explore**: Individual components in `autofocus_system.py`

### For Advanced Users
1. **Run**: `python af_calibration.py` for performance testing
2. **Visualize**: `python af_visualization.py` for algorithm understanding
3. **Customize**: Modify parameters in `AFConfig` class

### For Developers
1. **Study**: Algorithm implementations in detail
2. **Extend**: Add new AF methods or ROI selection strategies
3. **Integrate**: Connect with real camera hardware

## 📚 Additional Resources

- **README.md**: Comprehensive system documentation
- **simple_example.py**: Educational examples and explanations
- **autofocus_system.py**: Complete implementation with comments
- **af_calibration.py**: Testing and validation tools
- **af_visualization.py**: Visualization and analysis tools

## 🆘 Getting Help

If you encounter issues:

1. **Check this guide** for common solutions
2. **Run verification commands** to diagnose problems
3. **Check Python version** (requires 3.7+)
4. **Verify virtual environment** is activated
5. **Reinstall dependencies** if needed

---

*This setup guide ensures you can get the autofocus system running quickly and efficiently on any platform.*
