#!/bin/bash

# Autofocus System Setup Script
# =============================
# This script sets up a virtual environment and installs all dependencies
# for the autofocus system.

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

# Check if virtualenv is available
if ! command -v virtualenv &> /dev/null; then
    echo "📦 Installing virtualenv..."
    pip3 install virtualenv
fi

# Create virtual environment
VENV_NAME="af_env"
echo "🔧 Creating virtual environment: $VENV_NAME"

if [ -d "$VENV_NAME" ]; then
    echo "⚠️  Virtual environment already exists. Removing it..."
    rm -rf "$VENV_NAME"
fi

python3 -m venv "$VENV_NAME"

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source "$VENV_NAME/bin/activate"

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
echo "   source $VENV_NAME/bin/activate"
echo ""
echo "To deactivate, run:"
echo "   deactivate"
echo ""
echo "To run the autofocus system:"
echo "   python autofocus_system.py"
echo "   python simple_example.py"
echo "   python af_calibration.py"
echo "   python af_visualization.py"
echo ""
