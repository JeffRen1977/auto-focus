#!/bin/bash

# Autofocus System Setup Script
# =============================
# This script installs dependencies into an existing virtual environment
# located in the parent directory (../virutal_ven)

set -e  # Exit on any error

echo "=========================================="
echo "Autofocus System Setup Script"
echo "=========================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION detected"

# Virtual environment path (in parent directory)
VENV_PATH="../virtual_env"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Please create it first with:"
    echo "   cd .."
    echo "   python3 -m venv virtual_env"
    exit 1
fi

echo "✅ Found virtual environment at $VENV_PATH"

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Verify installation
echo "🔍 Verifying installation..."
python -c "
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
print('✅ All dependencies installed successfully!')
print(f'   NumPy version: {np.__version__}')
print(f'   OpenCV version: {cv2.__version__}')
print(f'   Matplotlib available: {plt.__version__}')
print(f'   SciPy available: {signal.__version__}')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To activate the virtual environment, run:"
echo "   source ../virtual_env/bin/activate"
echo ""
echo "To deactivate, run:"
echo "   deactivate"
echo ""
echo "To run the autofocus system:"
echo "   source ../virtual_env/bin/activate"
echo "   python autofocus_system.py"
echo "   python simple_example.py"
echo "   python af_calibration.py"
echo "   python af_visualization.py"
echo ""
