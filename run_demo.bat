@echo off
REM Autofocus System Demo Runner for Windows
REM ========================================

set VENV_NAME=af_env

echo ==========================================
echo Autofocus System Demo Runner (Windows)
echo ==========================================

REM Check if virtual environment exists
if not exist %VENV_NAME% (
    echo ❌ Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🚀 Activating virtual environment...
call %VENV_NAME%\Scripts\activate.bat

echo ✅ Virtual environment activated
echo.

:menu
echo Available Demos:
echo 1. Simple Autofocus Demo (Educational)
echo 2. Full Autofocus System Demo
echo 3. Calibration and Testing Demo
echo 4. Visualization Demo
echo 5. Run All Demos
echo 6. Quick Test
echo 7. Exit
echo.
set /p choice="Select demo (1-7): "

if "%choice%"=="1" goto demo1
if "%choice%"=="2" goto demo2
if "%choice%"=="3" goto demo3
if "%choice%"=="4" goto demo4
if "%choice%"=="5" goto demo5
if "%choice%"=="6" goto demo6
if "%choice%"=="7" goto exit
echo ❌ Invalid choice. Please select 1-7.
goto menu

:demo1
echo ==========================================
echo Running: Simple Autofocus Demo
echo Script: simple_example.py
echo ==========================================
if exist simple_example.py (
    python simple_example.py
    echo.
    echo ✅ Simple Autofocus Demo completed successfully!
) else (
    echo ❌ Script simple_example.py not found!
)
echo.
pause
goto menu

:demo2
echo ==========================================
echo Running: Full Autofocus System Demo
echo Script: autofocus_system.py
echo ==========================================
if exist autofocus_system.py (
    python autofocus_system.py
    echo.
    echo ✅ Full Autofocus System Demo completed successfully!
) else (
    echo ❌ Script autofocus_system.py not found!
)
echo.
pause
goto menu

:demo3
echo ==========================================
echo Running: Calibration and Testing Demo
echo Script: af_calibration.py
echo ==========================================
if exist af_calibration.py (
    python af_calibration.py
    echo.
    echo ✅ Calibration and Testing Demo completed successfully!
) else (
    echo ❌ Script af_calibration.py not found!
)
echo.
pause
goto menu

:demo4
echo ==========================================
echo Running: Visualization Demo
echo Script: af_visualization.py
echo ==========================================
if exist af_visualization.py (
    python af_visualization.py
    echo.
    echo ✅ Visualization Demo completed successfully!
) else (
    echo ❌ Script af_visualization.py not found!
)
echo.
pause
goto menu

:demo5
echo Running all demos in sequence...
echo.
call :demo1
call :demo2
call :demo3
call :demo4
echo 🎉 All demos completed!
pause
goto menu

:demo6
echo ==========================================
echo Running: Quick Test
echo Script: quick_start.py
echo ==========================================
if exist quick_start.py (
    python quick_start.py
    echo.
    echo ✅ Quick test completed successfully!
) else (
    echo ❌ Script quick_start.py not found!
)
echo.
pause
goto menu

:exit
echo 👋 Goodbye!
echo.
echo 🔧 Deactivating virtual environment...
call deactivate
echo ✅ Demo session completed!
pause
exit /b 0
