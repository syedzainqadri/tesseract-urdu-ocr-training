#!/bin/bash

# =============================================================================
# Tesseract OCR Training GUI Launcher
# =============================================================================
# 
# This script launches the Tesseract training GUI with proper environment setup.
# 
# Usage:
#   ./run_training_gui.sh
#
# Requirements:
#   - Python 3.x with tkinter
#   - Tesseract 4.x installed
#   - tesstrain framework
# =============================================================================

echo "🚀 Starting Tesseract OCR Training GUI..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if tesseract is available
if ! command -v tesseract &> /dev/null; then
    echo "❌ Error: Tesseract is not installed or not in PATH"
    echo "💡 Install with: brew install tesseract"
    exit 1
fi

# Check if training GUI exists
if [ ! -f "tesseract_gui.py" ]; then
    echo "❌ Error: tesseract_gui.py not found"
    echo "💡 Make sure you're in the correct directory"
    exit 1
fi

# Set environment variables for training
export TESSDATA_PREFIX="/opt/homebrew/share/tessdata"
export SCROLLVIEW_PATH="/opt/homebrew/bin"

# Launch the training GUI
echo "📱 Launching Training GUI..."
python3 tesseract_gui.py

echo "✅ Training GUI closed."
