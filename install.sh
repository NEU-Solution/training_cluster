#!/bin/bash
# filepath: /home/quanghung20gg/Documents/new/neu_solution/training_cluster/install.sh

set -e  # Exit on error

# Create installation directory
APP_DIR="$PWD"

echo "=== Installing system dependencies ==="
sudo apt-get update && sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    build-essential

# Set Python aliases
sudo ln -sf /usr/bin/python3 /usr/bin/python 
sudo ln -sf /usr/bin/pip3 /usr/bin/pip

echo "=== Setting up Python virtual environment ==="
python3 -m venv venv
source venv/bin/activate

# Copy requirements.txt if it exists in the same directory as this script
if [ -f "../requirements.txt" ]; then
    cp ../requirements.txt .
    pip install --no-cache-dir --ignore-installed -r requirements.txt
else
    echo "Warning: requirements.txt not found"
fi

echo "=== Installing FastAPI and related dependencies ==="
pip install fastapi uvicorn requests pydantic python-dotenv

echo "=== Cloning LLaMA-Factory repository ==="
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git

echo "=== Installing LLaMA-Factory requirements ==="
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-cache-dir
cd ..

echo "=== Setting up environment variables ==="
echo 'export PYTHONPATH="$APP_DIR:$PYTHONPATH"' >> ~/.bashrc
echo 'export NVIDIA_VISIBLE_DEVICES=all' >> ~/.bashrc
echo 'export NVIDIA_DRIVER_CAPABILITIES=compute,utility' >> ~/.bashrc
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc

# Create a run script for convenience
cat > run.sh << 'EOF'
#!/bin/bash
# Run script for training cluster
source venv/bin/activate

# Run the training server
cd api && python train_server.py
EOF

chmod +x run.sh

# Add a note about the run script to the final message
echo "=== Installation complete ==="
echo "To run the application:"
echo "Option 1: Use the run script"
echo "   ./run.sh"
echo ""
echo "Option 2: Manual steps"
echo "1. Source your environment: source ~/.bashrc"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run: cd api && python train_server.py"
echo "The API will be available on port 23477"