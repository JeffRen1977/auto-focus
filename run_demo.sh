#!/bin/bash

# Autofocus System Demo Runner
# ============================
# This script activates the virtual environment and runs demos
# Uses existing virtual environment in parent directory

set -e  # Exit on any error

VENV_PATH="../virtual_env"

echo "=========================================="
echo "Autofocus System Demo Runner"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Please run setup.sh first or create the virtual environment:"
    echo "   ./setup.sh"
    exit 1
fi

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source "$VENV_PATH/bin/activate"

echo "✅ Virtual environment activated"
echo ""

# Function to run a demo with error handling
run_demo() {
    local script_name=$1
    local description=$2
    
    echo "=========================================="
    echo "Running: $description"
    echo "Script: $script_name"
    echo "=========================================="
    
    if [ -f "$script_name" ]; then
        python "$script_name"
        echo ""
        echo "✅ $description completed successfully!"
    else
        echo "❌ Script $script_name not found!"
    fi
    
    echo ""
    echo "Press Enter to continue to next demo, or Ctrl+C to exit..."
    read -r
}

# Menu for selecting demos
while true; do
    echo "Available Demos:"
    echo "1. Complete Autofocus Example (examples/example_usage.py)"
    echo "2. Quick Start Validation (examples/quick_start.py)"
    echo "3. Calibration and Testing (utils/af_calibration.py)"
    echo "4. Visualization Tools (utils/af_visualization.py)"
    echo "5. Run All Demos"
    echo "6. Exit"
    echo ""
    read -p "Select demo (1-6): " choice
    
    case $choice in
        1)
            run_demo "examples/example_usage.py" "Complete Autofocus Example"
            ;;
        2)
            run_demo "examples/quick_start.py" "Quick Start Validation"
            ;;
        3)
            run_demo "utils/af_calibration.py" "Calibration and Testing Demo"
            ;;
        4)
            run_demo "utils/af_visualization.py" "Visualization Demo"
            ;;
        5)
            echo "Running all demos in sequence..."
            echo ""
            run_demo "examples/example_usage.py" "Complete Autofocus Example"
            run_demo "examples/quick_start.py" "Quick Start Validation"
            run_demo "utils/af_calibration.py" "Calibration and Testing Demo"
            run_demo "utils/af_visualization.py" "Visualization Demo"
            echo "🎉 All demos completed!"
            break
            ;;
        6)
            echo "👋 Goodbye!"
            break
            ;;
        *)
            echo "❌ Invalid choice. Please select 1-6."
            ;;
    esac
done

echo ""
echo "🔧 Deactivating virtual environment..."
deactivate
echo "✅ Demo session completed!"
