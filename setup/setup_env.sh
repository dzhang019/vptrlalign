#!/bin/bash

# Update the system
echo "Updating the system..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install dependencies
echo "Installing dependencies..."
sudo apt-get install -y software-properties-common wget curl tar

# Install Python 3.8.10
echo "Installing Python 3.8.10..."
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.8 python3.8-venv python3.8-dev

# Set Python 3.8.10 as the default Python version
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
sudo update-alternatives --config python3 <<< '1'

# Install OpenJDK 8
echo "Installing OpenJDK 8..."
sudo apt-get install -y openjdk-8-jdk

# Configure OpenJDK environment variables
echo "Configuring JDK environment variables..."
JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
echo "export JAVA_HOME=${JAVA_HOME}" >> ~/.bashrc
echo "export PATH=\$JAVA_HOME/bin:\$PATH" >> ~/.bashrc

# Apply changes to the current session
source ~/.bashrc

# Verify installations
echo "Verifying installations..."
echo "Python version:"
python3 --version

echo "Java version:"
java -version

echo "Setup complete!"
