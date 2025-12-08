#!/bin/bash

# Corn Disease Detection App - Installation & Run Script

echo "🌽 Corn Disease Detection System"
echo "================================="
echo ""

# Check if running in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
    echo "Installing packages..."
    pip install -r requirements.txt
else
    echo "⚠️  No virtual environment detected"
    echo "Choose installation method:"
    echo "1. Install with --break-system-packages (may affect system Python)"
    echo "2. Create and use virtual environment (recommended)"
    echo ""
    read -p "Enter choice (1 or 2): " choice
    
    if [ "$choice" == "2" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        echo "Activating virtual environment..."
        source venv/bin/activate
        echo "Installing packages..."
        pip install -r requirements.txt
    else
        echo "Installing packages system-wide..."
        pip install -r requirements.txt --break-system-packages
    fi
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "🚀 Starting Streamlit app..."
echo "================================="
echo ""

# Run the app
streamlit run app.py
