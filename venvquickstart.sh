#!/bin/bash

# Set up a Python virtual environment
echo "Setting up Python virtual environment..."
VENV_DIR="venv"  # Specify the virtual environment directory name
python3.8 -m venv $VENV_DIR

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip in the virtual environment
pip install --upgrade pip

# Install minerl first
echo "Installing minerl from GitHub..."
pip install git+https://github.com/minerllabs/minerl

# Check if requirements.txt exists and install dependencies
REQUIREMENTS_FILE="requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from $REQUIREMENTS_FILE..."
    pip install --index-url https://download.pytorch.org/whl/cu124 -r $REQUIREMENTS_FILE
else
    echo "No $REQUIREMENTS_FILE found. Skipping dependency installation."
fi

# Verify installations
echo "Verifying installations..."
echo "Python version:"
python --version

echo "Minerl installation check:"
python -c "import minerl; print('MineRL successfully installed!')" || echo "MineRL installation failed."

echo "Setup complete!"
