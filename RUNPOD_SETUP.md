# Wan2.2 S2V RunPod GPU Setup

## 🚀 One-Command Setup for RunPod GPU

This setup is designed for RunPod GPU instances where `/home` storage is volatile and needs to be re-initialized each time.

### Quick Start (Recommended)

```bash
# Run the automated setup script
bash /workspace/wan22-comfy-project/runpod_s2v_init.sh
```

This single command will:
- ✅ Check GPU environment
- ✅ Setup Python virtual environment
- ✅ Install system dependencies (FFmpeg, etc.)
- ✅ Download and cache all Wan2.2 S2V models (~44GB)
- ✅ Create convenience scripts
- ✅ Validate everything is working

**Expected time:** 10-15 minutes depending on connection speed

### Manual Setup (Alternative)

If you prefer manual control:

```bash
# 1. Navigate to Wan2.2 directory
cd /workspace/wan22-comfy-project/Wan2.2

# 2. Activate virtual environment
source /workspace/wan22-comfy-project/venv/bin/activate

# 3. Run cache setup
python setup_s2v_cache.py --quick-setup

# 4. Generate video
python run_s2v_multi_gpu.py
```

### Generated Convenience Scripts

After setup, you'll have these quick commands:

```bash
# Check cache and model status
/workspace/wan22-comfy-project/check_s2v_status.sh

# Run S2V generation directly
/workspace/wan22-comfy-project/run_s2v.sh
```

### Troubleshooting

**GPU Issues:**
```bash
# Check GPU status
nvidia-smi

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**Cache Issues:**
```bash
# Clear and rebuild cache
rm -rf /home/caches/wan_s2v_cache
python setup_s2v_cache.py --quick-setup
```

**Model Download Issues:**
```bash
# Login to HuggingFace if needed
huggingface-cli login

# Manual model check
python setup_s2v_cache.py --status
```

### System Requirements

- **GPU:** 2x NVIDIA A40 48GB or similar (Multi-GPU setup)
- **RAM:** 32GB+ recommended
- **Storage:** 50GB+ for models and cache
- **Network:** Good connection for model downloads

### File Structure After Setup

```
/workspace/wan22-comfy-project/
├── runpod_s2v_init.sh          # Main setup script
├── run_s2v.sh                  # Quick run script
├── check_s2v_status.sh         # Status checker
├── venv/                       # Python environment
└── Wan2.2/
    ├── setup_s2v_cache.py      # Cache manager
    ├── run_s2v_multi_gpu.py    # Multi-GPU generation
    └── quick_s2v_setup.sh      # Generated setup script

/home/caches/wan_s2v_cache/     # Model cache (volatile on RunPod)
├── models--Wan-AI--Wan2.2-S2V-14B/
├── audio_processor_cache/
└── cache_status.json
```

### RunPod Specific Notes

- **Volatile Storage:** `/home` is cleared on instance restart
- **Persistent Storage:** `/workspace` persists between sessions
- **Re-initialization:** Run setup script each time you start fresh instance
- **GPU Memory:** Optimized for 2x 48GB GPUs with FSDP

### Usage Examples

**Basic video generation:**
```bash
cd /workspace/wan22-comfy-project/Wan2.2
source /workspace/wan22-comfy-project/venv/bin/activate
python run_s2v_multi_gpu.py
```

**Custom input files:**
```bash
# Edit run_s2v_multi_gpu.py to change:
# - audio_file_path
# - image_file_path
# - output paths
```

### Performance Tips

- **First run:** Expect 10-15 minutes for model download
- **Subsequent runs:** ~30-60 seconds for video generation
- **Memory usage:** ~80GB GPU memory for dual-GPU setup
- **Output quality:** 512x512 resolution, ~1-2 second videos

### Support

If you encounter issues:
1. Check the logs in the terminal output
2. Verify GPU memory with `nvidia-smi`
3. Ensure all models downloaded correctly
4. Try clearing cache and re-running setup

**Happy video generation!** 🎬✨