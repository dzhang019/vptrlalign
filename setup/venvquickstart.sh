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

#install torch
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

#install gym, gym3, attr, attrs
pip install gym gym3 attr attrs

#install models and weight files in current directory
wget "https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-2x.model" #2x model
#wget "https://openaipublic.blob.core.windows.net/minecraft-rl/models/rl-from-early-game-2x.weights" #early game weights
wget "https://openaipublic.blob.core.windows.net/minecraft-rl/models/rl-from-house-2x.weights" #house weights

# Check if requirements.txt exists and install dependencies
REQUIREMENTS_FILE="requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from $REQUIREMENTS_FILE..."
    pip install -r $REQUIREMENTS_FILE
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
