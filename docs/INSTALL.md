# Installation Instructions

## 📂 Virtual Environment Setup

This autofocus system uses a virtual environment located in the **parent directory** of the project.

### Expected Directory Structure

```
Mobile_Computational_photograph/
├── virtual_env/              # Virtual environment (shared)
└── Autofocus/                # This project
    ├── autofocus_system.py
    ├── simple_example.py
    ├── setup.sh
    ├── requirements.txt
    └── ...
```

## 🚀 Quick Installation

### Option 1: Using Setup Script (Recommended)

```bash
# Navigate to the Autofocus directory
cd Autofocus

# Make script executable
chmod +x setup.sh

# Run setup (will use existing ../virtual_env)
./setup.sh
```

### Option 2: Manual Installation

```bash
# Activate the existing virtual environment
source ../virtual_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python quick_start.py --check-deps
```

## 🔧 If Virtual Environment Doesn't Exist

If the `virtual_env` directory doesn't exist in the parent folder, create it first:

```bash
# Go to parent directory
cd ..

# Create virtual environment
python3 -m venv virtual_env

# Return to Autofocus directory
cd Autofocus

# Activate and install
source ../virtual_env/bin/activate
pip install -r requirements.txt
```

## ✅ Verify Installation

```bash
# Activate virtual environment
source ../virtual_env/bin/activate

# Run quick test
python quick_start.py
```

Expected output:
```
✅ All dependencies installed successfully!
   NumPy version: 1.21.0+
   OpenCV version: 4.5.0+
   Matplotlib available
   SciPy available
```

## 🎯 Running Demos

After installation, run demos:

```bash
# Make sure virtual environment is activated
source ../virtual_env/bin/activate

# Run interactive demo menu
./run_demo.sh

# Or run individual demos
python simple_example.py      # Educational examples
python autofocus_system.py    # Full system demo
python af_calibration.py      # Calibration testing
python af_visualization.py    # Visualization tools
```

## 📚 More Information

- **SETUP_GUIDE.md** - Detailed setup instructions
- **HELP.md** - Quick reference guide
- **README.md** - Complete system documentation

## 🐛 Troubleshooting

### Virtual environment not found
```bash
# Create it in the parent directory
cd ..
python3 -m venv virtual_env
cd Autofocus
```

### Permission issues (Linux/Mac)
```bash
chmod +x setup.sh run_demo.sh quick_start.py
```

### Import errors
```bash
# Make sure virtual environment is activated
source ../virtual_env/bin/activate
```

---

*For detailed information, see SETUP_GUIDE.md*
