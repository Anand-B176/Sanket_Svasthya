#!/bin/bash
# Setup Environment Script - Sanket-Svasthya
# Run this script to set up the development environment

echo "========================================"
echo "  Sanket-Svasthya Environment Setup"
echo "========================================"

# Check Python version
python --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python not found. Please install Python 3.10+"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "  Run: streamlit run ui/app.py"
echo "========================================"
