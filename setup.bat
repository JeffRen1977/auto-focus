@echo off
REM Autofocus System Setup Script for Windows
REM ==========================================
REM This script sets up a virtual environment and installs all dependencies

echo ==========================================
echo Autofocus System Setup Script (Windows)
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH.
    echo Please install Python 3.7+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python detected
python --version

REM Create virtual environment
set VENV_NAME=af_env
echo 🔧 Creating virtual environment: %VENV_NAME%

if exist %VENV_NAME% (
    echo ⚠️  Virtual environment already exists. Removing it...
    rmdir /s /q %VENV_NAME%
)

python -m venv %VENV_NAME%
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🚀 Activating virtual environment...
call %VENV_NAME%\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Verify installation
echo 🔍 Verifying installation...
python -c "import numpy as np; import cv2; import matplotlib.pyplot as plt; from scipy import signal; print('✅ All dependencies installed successfully!'); print('   NumPy version:', np.__version__); print('   OpenCV version:', cv2.__version__); print('   Matplotlib available'); print('   SciPy available')"
if errorlevel 1 (
    echo ❌ Verification failed
    pause
    exit /b 1
)

echo.
echo 🎉 Setup completed successfully!
echo.
echo To activate the virtual environment, run:
echo    %VENV_NAME%\Scripts\activate.bat
echo.
echo To run the autofocus system:
echo    python autofocus_system.py
echo    python simple_example.py
echo    python af_calibration.py
echo    python af_visualization.py
echo.
pause
