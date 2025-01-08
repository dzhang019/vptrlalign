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

# Download and Install JDK 8u421
echo "Installing JDK 8u421..."
JDK_VERSION="8u421"
JDK_BUILD="b11"
JDK_ARCHIVE="jdk-${JDK_VERSION}-linux-x64.tar.gz"
DOWNLOAD_URL="https://download.oracle.com/otn/java/jdk/8u421-${JDK_BUILD}/e9e7ea248e2c4826b92b3f075a80e441/${JDK_ARCHIVE}"

# Oracle requires manual acceptance of license terms, use curl to fetch
wget --no-cookies --no-check-certificate --header "Cookie: oraclelicense=accept-securebackup-cookie" -O $JDK_ARCHIVE $DOWNLOAD_URL

# Extract JDK and move it to /usr/lib/jvm
sudo mkdir -p /usr/lib/jvm
sudo tar -zxf $JDK_ARCHIVE -C /usr/lib/jvm
JDK_FOLDER=$(tar -tzf $JDK_ARCHIVE | head -1 | cut -f1 -d"/")
sudo mv /usr/lib/jvm/$JDK_FOLDER /usr/lib/jvm/jdk-$JDK_VERSION

# Set environment variables for JDK
echo "Configuring JDK environment variables..."
JAVA_HOME="/usr/lib/jvm/jdk-${JDK_VERSION}"
echo "export JAVA_HOME=${JAVA_HOME}" >> ~/.bashrc
echo "export PATH=\$JAVA_HOME/bin:\$PATH" >> ~/.bashrc

# Apply changes to the current session
source ~/.bashrc

# Clean up
rm -f $JDK_ARCHIVE

# Verify installations
echo "Verifying installations..."
echo "Python version:"
python3 --version

echo "Java version:"
java -version

echo "Setup complete!"
