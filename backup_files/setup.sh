#!/bin/bash
# Multi-GPU S2V System Setup Script
# This script sets up the environment and dependencies for the speech-to-video system

set -e  # Exit on any error

echo "=================================================="
echo "Multi-GPU Speech-to-Video System Setup"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Setup external cache system
print_status "Setting up external cache system..."
python3 -c "
from cache_manager import setup_cache_system
cache_manager = setup_cache_system()
print('Cache system configured successfully')
"

if [ $? -eq 0 ]; then
    print_success "External cache system configured"
else
    print_warning "Cache system setup had issues, continuing with local storage"
fi

# Check if running in the correct directory
if [ ! -d "Wan2.2" ]; then
    print_error "Wan2.2 directory not found. Please run this script from the project root."
    exit 1
fi

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    print_success "Python version $python_version is compatible"
else
    print_error "Python 3.8+ is required. Found: $python_version"
    exit 1
fi

# Check CUDA availability
print_status "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    print_success "Found $gpu_count GPU(s)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_warning "NVIDIA GPU not found or nvidia-smi not available"
fi

# Use existing virtual environment
if [ -d "venv" ]; then
    print_success "Using existing virtual environment at /workspace/wan22-comfy-project/venv"
else
    print_error "Virtual environment not found at /workspace/wan22-comfy-project/venv"
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Check and install missing packages only
print_status "Checking for missing packages..."

# Function to check if package is installed
check_and_install() {
    local package=$1
    local pip_name=$2
    if python -c "import $package" 2>/dev/null; then
        print_success "$package already installed"
    else
        print_status "Installing missing package: $pip_name"
        pip install "$pip_name"
    fi
}

# Check essential packages
check_and_install "torch" "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
check_and_install "ray" "ray[serve]>=2.6.0"
check_and_install "PIL" "Pillow"
check_and_install "cv2" "opencv-python"
check_and_install "librosa" "librosa"
check_and_install "soundfile" "soundfile"
check_and_install "psutil" "psutil"
check_and_install "tqdm" "tqdm"
check_and_install "numpy" "numpy"
check_and_install "requests" "requests"

# Install from requirements.txt if missing packages exist
if [ -f "requirements.txt" ]; then
    print_status "Installing any remaining requirements..."
    pip install -r requirements.txt --quiet --no-deps 2>/dev/null || true
fi

# Install Wan2.2 requirements if they exist and packages are missing
if [ -f "Wan2.2/requirements.txt" ]; then
    print_status "Checking Wan2.2 requirements..."
    pip install -r Wan2.2/requirements.txt --quiet --no-deps 2>/dev/null || true
fi

# Install additional S2V requirements if they exist
if [ -f "Wan2.2/requirements_s2v.txt" ]; then
    print_status "Checking S2V-specific requirements..."
    pip install -r Wan2.2/requirements_s2v.txt --quiet --no-deps 2>/dev/null || true
fi

# Install system dependencies for video processing
print_status "Checking system dependencies..."

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    print_warning "FFmpeg not found. Installing..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v yum &> /dev/null; then
        sudo yum install -y ffmpeg
    elif command -v brew &> /dev/null; then
        brew install ffmpeg
    else
        print_warning "Could not install FFmpeg automatically. Please install it manually."
    fi
else
    print_success "FFmpeg found"
fi

# Create necessary directories
print_status "Creating directories..."
mkdir -p outputs
mkdir -p logs
mkdir -p checkpoints

# Create cache directories in /home/caches
print_status "Creating cache directories in /home/caches..."
mkdir -p /home/caches
mkdir -p /home/caches/temp
mkdir -p /home/caches/huggingface
mkdir -p /home/caches/torch
mkdir -p /home/caches/transformers
mkdir -p /home/caches/ray_temp
print_success "Cache directories created in /home/caches"

# Create environment configuration file
print_status "Creating environment configuration..."
cat > .env << EOF
# Multi-GPU S2V System Environment Configuration
# Using external cache storage to save workspace space

# Model and Output Paths
WAN_CHECKPOINT_DIR=./checkpoints
OUTPUT_DIR=./outputs
LOG_DIR=./logs

# Cache Directories (external storage)
CACHE_DIR=/home/caches
TEMP_DIR=/home/caches/temp
HF_HOME=/home/caches/huggingface
TORCH_HOME=/home/caches/torch
TRANSFORMERS_CACHE=/home/caches/transformers

# Ray Configuration (using external temp)
RAY_DASHBOARD_HOST=0.0.0.0
RAY_DASHBOARD_PORT=8265
RAY_TMPDIR=/home/caches/ray_temp

# CUDA Settings (adjust based on your setup)
CUDA_VISIBLE_DEVICES=0,1,2,3
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Python Path
PYTHONPATH=\${PYTHONPATH}:./Wan2.2

# Logging
LOG_LEVEL=INFO
ENABLE_PROFILING=false
EOF

# Create a simple test script
print_status "Creating test script..."
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify the setup"""

import sys
import torch
import ray
import numpy as np
import cv2
import librosa
from PIL import Image

def test_pytorch():
    print("Testing PyTorch...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    return torch.cuda.is_available()

def test_ray():
    print("\nTesting Ray...")
    try:
        ray.init(num_cpus=2, num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0)
        print(f"Ray version: {ray.__version__}")
        print("Ray initialized successfully")
        ray.shutdown()
        return True
    except Exception as e:
        print(f"Ray test failed: {e}")
        return False

def test_dependencies():
    print("\nTesting dependencies...")
    deps = {
        'NumPy': np.__version__,
        'OpenCV': cv2.__version__,
        'Librosa': librosa.__version__,
        'PIL': Image.__version__
    }
    
    for name, version in deps.items():
        print(f"  {name}: {version}")
    
    return True

def test_wan_import():
    print("\nTesting Wan2.2 import...")
    try:
        sys.path.append('Wan2.2')
        from wan.speech2video import WanS2V
        print("Wan2.2 import successful")
        return True
    except Exception as e:
        print(f"Wan2.2 import failed: {e}")
        return False

if __name__ == "__main__":
    print("="*50)
    print("MULTI-GPU S2V SYSTEM - SETUP VERIFICATION")
    print("="*50)
    
    tests = [
        ("PyTorch", test_pytorch),
        ("Ray", test_ray),
        ("Dependencies", test_dependencies),
        ("Wan2.2", test_wan_import)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✓ {test_name} test passed")
                passed += 1
            else:
                print(f"✗ {test_name} test failed")
        except Exception as e:
            print(f"✗ {test_name} test failed: {e}")
    
    print("\n" + "="*50)
    print(f"SETUP VERIFICATION COMPLETE: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Setup successful! The system is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("="*50)
EOF

# Make test script executable
chmod +x test_setup.py

# Run setup verification with existing environment
print_status "Running setup verification with existing environment..."
python3 test_setup.py

# Create quick start script
print_status "Creating quick start script..."
cat > quick_start.sh << 'EOF'
#!/bin/bash
# Quick start script for Multi-GPU S2V System

# Activate existing virtual environment
source /workspace/wan22-comfy-project/venv/bin/activate

# Set environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:./Wan2.2"

# Example command (modify paths as needed)
echo "Example usage:"
echo "python3 main_s2v_system.py \\"
echo "  --image /workspace/wan22-comfy-project/iphone.jpeg \\"
echo "  --audio /workspace/wan22-comfy-project/tmp_19iifpd.mp3 \\"
echo "  --prompt 'A person speaking into a phone' \\"
echo "  --output example_video \\"
echo "  --quality medium \\"
echo "  --gpus 2"
echo
echo "To run the example:"
echo "chmod +x quick_start.sh"
echo "./quick_start.sh"
EOF

chmod +x quick_start.sh

# Final status
print_success "Setup completed successfully!"
echo
echo "=================================================="
echo "SETUP SUMMARY:"
echo "=================================================="
echo "✅ Using existing virtual environment: /workspace/wan22-comfy-project/venv"
echo "✅ Missing packages checked and installed"
echo "✅ Cache directories created in /home/caches (saves workspace storage)"
echo "✅ Environment configuration updated"
echo "✅ System verification completed"
echo
echo "NEXT STEPS:"
echo "=================================================="
echo "1. Download model checkpoints to ./checkpoints/"
echo "2. Activate the environment: source venv/bin/activate"
echo "3. Load environment variables: source .env"
echo "4. Run the example: ./quick_start.sh"
echo "5. For help: python3 main_s2v_system.py --help"
echo
echo "📁 Cache locations (to save workspace storage):"
echo "   - Models/Checkpoints cache: /home/caches/"
echo "   - Ray temporary files: /home/caches/ray_temp/"
echo "   - HuggingFace cache: /home/caches/huggingface/"
echo "   - PyTorch cache: /home/caches/torch/"
echo
echo "📁 Output locations:"
echo "   - Generated videos: ./outputs/"
echo "   - Log files: ./logs/"
echo "=================================================="