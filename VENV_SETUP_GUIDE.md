# Virtual Environment Setup Guide

This guide helps you recreate the exact working environment on another machine.

## Quick Setup on New Machine

### Method 1: Install Exact Versions (Recommended)
Use the frozen requirements file with exact versions that are currently working:

```bash
# Create new virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install exact dependencies
pip install -r requirements_frozen.txt
```

### Method 2: Install with Flash Attention Build
If you need to build flash_attn from source (recommended for GPU compatibility):

```bash
# Create new virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch first (CUDA 12.8)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# Install ninja for faster compilation
pip install ninja

# Install flash_attn (this will compile from source)
pip install flash_attn==2.8.3

# Install remaining dependencies
pip install -r requirements_frozen.txt
```

## System Requirements

- **Python Version**: Python 3.10 (recommended based on cache files)
- **CUDA**: CUDA 12.8.x (based on nvidia packages)
- **GPU**: NVIDIA GPU with compute capability for flash attention
- **OS**: Linux (Ubuntu/Debian recommended)

## Key Dependencies Overview

### Core ML Frameworks
- **torch**: 2.8.0 (PyTorch with CUDA 12.8)
- **torchvision**: 0.23.0
- **torchaudio**: 2.8.0
- **transformers**: 4.56.2
- **diffusers**: 0.35.1
- **flash_attn**: 2.8.3

### Video/Audio Processing
- **opencv-python**: 4.12.0.88
- **imageio**: 2.37.0
- **imageio-ffmpeg**: 0.6.0
- **librosa**: 0.11.0
- **soundfile**: 0.13.1
- **av**: 15.1.0
- **decord**: 0.6.0

### Acceleration & Optimization
- **accelerate**: 1.10.1
- **bitsandbytes**: 0.47.0
- **triton**: 3.4.0

### ComfyUI Dependencies
- **comfyui-embedded-docs**: 0.2.6
- **comfyui_frontend_package**: 1.26.13
- **comfyui_workflow_templates**: 0.1.86

### Model & Cloud Services
- **modelscope**: 1.30.0
- **dashscope**: 1.24.6
- **huggingface-hub**: 0.35.3

### UI & Web
- **gradio**: 5.49.1
- **fastapi**: 0.117.1
- **uvicorn**: 0.37.0

### Additional Tools
- **ray**: 2.6.0 (distributed computing)
- **pytorch-lightning**: 2.5.5
- **peft**: 0.17.1 (parameter efficient fine-tuning)
- **SAM-2**: 1.0 (Segment Anything Model 2)
- **openai-whisper**: 20250625

## Installation Notes

### 1. Flash Attention
The `flash_attn==2.8.3` package requires compilation. Ensure you have:
- NVIDIA GPU with compute capability 7.5 or higher
- CUDA toolkit installed
- C++ compiler (g++)
- Ninja build system (optional but recommended for speed)

### 2. CUDA Version
All nvidia-* packages are for CUDA 12.8. Make sure your system has compatible CUDA drivers.

### 3. System Packages
You may need these system packages (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    espeak-ng
```

## Verification

After installation, verify the setup:

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Check key packages
python -c "import diffusers, transformers, gradio, flash_attn; print('All key packages imported successfully')"

# Run the simple test
cd /workspace/wan22-comfy-project
python simple_test.py
```

## Troubleshooting

### Flash Attention Install Fails
```bash
# Install with pre-built wheel
pip install flash-attn --no-build-isolation
```

### Out of Memory During Install
```bash
# Limit parallel builds
MAX_JOBS=4 pip install flash_attn==2.8.3
```

### CUDA Version Mismatch
If you have a different CUDA version, you may need to adjust torch installation:
```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Files Included

- `requirements_frozen.txt` - Exact versions from working venv (258 packages)
- `VENV_SETUP_GUIDE.md` - This setup guide
- `Wan2.2/requirements.txt` - Original project requirements

## Notes

The working venv contains **258 packages** including all dependencies and sub-dependencies. Using `requirements_frozen.txt` ensures you get the exact same versions that are currently working.
