#!/bin/bash

# =============================================================================
# Enhanced OCR Application Launcher
# =============================================================================
# 
# This script launches the enhanced OCR application for testing trained models.
# 
# Usage:
#   ./run_ocr_app.sh
#
# Features:
#   - Multi-format image support (.tif, .png, .webp, .jpeg)
#   - Real-time OCR processing
#   - Model comparison
#   - Batch processing
# =============================================================================

echo "🚀 Starting Enhanced OCR Application..."

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

# Check if OCR app exists
if [ ! -f "enhanced_ocr_app.py" ]; then
    echo "❌ Error: enhanced_ocr_app.py not found"
    echo "💡 Make sure you're in the correct directory"
    exit 1
fi

# Set environment variables
export TESSDATA_PREFIX="/opt/homebrew/share/tessdata"

# Check for required Python packages
echo "🔍 Checking Python dependencies..."

python3 -c "import tkinter" 2>/dev/null || {
    echo "❌ Error: tkinter not available"
    echo "💡 Install with: brew install python-tk"
    exit 1
}

python3 -c "import PIL" 2>/dev/null || {
    echo "❌ Error: Pillow not installed"
    echo "💡 Install with: pip3 install Pillow"
    exit 1
}

python3 -c "import pytesseract" 2>/dev/null || {
    echo "❌ Error: pytesseract not installed"
    echo "💡 Install with: pip3 install pytesseract"
    exit 1
}

# Launch the OCR application
echo "📱 Launching Enhanced OCR Application..."
python3 enhanced_ocr_app.py

echo "✅ OCR Application closed."
