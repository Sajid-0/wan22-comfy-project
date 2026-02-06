# RunPod S2V Model Management Guide

This guide explains how to manage Wan2.2-S2V-14B models on RunPod on-demand GPU instances where `/home` storage is volatile and gets reset on each restart.

## Quick Start (New RunPod Instance)

Every time you start a new RunPod instance, run:

```bash
cd /workspace/wan22-comfy-project
./runpod_startup.sh
```

This will:
- ✅ Check if models exist in `/home/caches`
- ✅ Download missing models (~43GB) automatically
- ✅ Verify model integrity
- ✅ Prepare everything for S2V generation

## Manual Model Management

### Setup Script Commands

```bash
cd /workspace/wan22-comfy-project/Wan2.2

# Quick check and setup (recommended)
python setup_s2v_cache.py quick

# Full interactive setup
python setup_s2v_cache.py

# Download models only
python setup_s2v_cache.py download

# Check cache status
python setup_s2v_cache.py status

# Full setup (first time)
python setup_s2v_cache.py setup
```

### Multi-GPU Generation

The `run_s2v_multi_gpu.py` script now automatically ensures models are ready:

```bash
cd /workspace/wan22-comfy-project/Wan2.2
python run_s2v_multi_gpu.py
```

## How It Works

### Automatic Model Management
- 🔍 **Auto-detection**: Scripts automatically detect if models are missing
- 📥 **Smart downloading**: Only downloads missing or corrupted files
- ✅ **Integrity checks**: Verifies all required files are complete
- 🔐 **Token management**: Automatically handles Hugging Face authentication

### RunPod Volatile Storage Handling
- 📁 **Cache location**: `/home/caches/Wan2.2-S2V-14B/` (~43GB)
- ⚠️ **Volatile warning**: `/home` gets reset when instance stops
- 🚀 **Auto-recovery**: Automatically downloads models on fresh instances
- ⚡ **Quick setup**: Optimized for repeated setups

### Model Files
The following files are automatically downloaded and verified:
- `diffusion_pytorch_model-00001-of-00004.safetensors` (9.97GB)
- `diffusion_pytorch_model-00002-of-00004.safetensors` (9.89GB)
- `diffusion_pytorch_model-00003-of-00004.safetensors` (9.96GB)
- `diffusion_pytorch_model-00004-of-00004.safetensors` (2.77GB)
- `models_t5_umt5-xxl-enc-bf16.pth` (11.4GB)
- `Wan2.1_VAE.pth` (508MB)
- Configuration and tokenizer files

## Environment Variables

You can set these environment variables for automation:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

## Troubleshooting

### Token Issues
If you see authentication errors:
```bash
# Set token manually
python -c "from huggingface_hub import login; login(token='your_token')"
```

### Download Issues
If downloads fail:
```bash
# Force redownload
python setup_s2v_cache.py download --force-redownload
```

### Disk Space
Models require ~43GB in `/home/caches`. Ensure sufficient space.

## Status Check

To verify everything is working:
```bash
python setup_s2v_cache.py status
```

Expected output:
```
✅ Wan2.2-S2V-14B: Complete (42.60 GB)
🎉 All models are ready! You can now run:
   python run_s2v_multi_gpu.py
```

## Integration

The model management is seamlessly integrated:
- `run_s2v_multi_gpu.py` automatically calls model setup
- No manual intervention needed for normal operation
- Handles RunPod instance restarts gracefully