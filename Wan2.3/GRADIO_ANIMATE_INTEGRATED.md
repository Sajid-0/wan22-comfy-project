# Gradio Animate App - Integrated Cache Manager

## ✅ Integration Complete

The cache management system from `setup_animate_cache.py` has been successfully integrated into `gradio_animate_app.py`.

## Key Features

### 🔄 Integrated Cache Manager

- **Automatic Model Downloads**: Models download automatically from HuggingFace when needed
- **Smart Caching**: Checks for existing models before downloading
- **Progress Tracking**: Visual feedback during download and loading
- **HF Token Management**: Automatically configures HuggingFace authentication

### 📦 Components Merged

```python
class WanAnimateCacheManager:
    - setup_hf_token()      # Auto-configures HF authentication
    - check_model_integrity() # Verifies model files exist
    - download_models()     # Downloads from Wan-AI/Wan2.2-Animate-14B
```

### 🎯 Key Changes

1. **Removed external dependency** on `setup_animate_cache.py`
2. **Integrated cache manager** directly into Gradio app
3. **Added download-only button** for pre-downloading models
4. **Enhanced UI** with better status messages and progress tracking
5. **Improved error handling** for gated repos and network issues

## File Structure

```
/workspace/wan22-comfy-project/Wan2.2/
├── gradio_animate_app.py       # ✅ Integrated app (self-contained)
├── setup_animate_cache.py      # Still available for CLI usage
├── launch_animate_ui.sh        # Launch script
├── test_gradio_animate.py      # Test suite
└── examples/
    ├── pose.mp4                # Example driving video
    └── pose.png                # Example reference image
```

## Usage

### Option 1: Launch Gradio UI (Recommended)

```bash
cd /workspace/wan22-comfy-project/Wan2.2
./launch_animate_ui.sh
```

Then visit: http://0.0.0.0:7862

### Option 2: Direct Python

```bash
cd /workspace/wan22-comfy-project/Wan2.2
/workspace/wan22-comfy-project/venv/bin/python gradio_animate_app.py
```

### Option 3: Pre-download Models (CLI)

```bash
cd /workspace/wan22-comfy-project/Wan2.2
/workspace/wan22-comfy-project/venv/bin/python setup_animate_cache.py quick
```

## Workflow

### 1. Setup Tab
- Click **"📥 Download Models Only"** to pre-download (~50GB)
- Or click **"🚀 Load Model"** to download AND load
- Optional: Enable "Use Relighting LoRA" for replacement mode

### 2. Preprocess Tab
- Upload driving video (motion source)
- Upload reference image (character to animate)
- Select mode: **animate** or **replace**
- Click **"Preprocess"**
- Copy the generated path

### 3. Generate Tab
- Paste preprocessed path
- Adjust parameters:
  - **Frames**: 77 (≈2.5s at 30fps)
  - **Sampling Steps**: 20 (quality vs speed)
  - **Guidance Scale**: 1.0 (recommended)
  - **Seed**: -1 (random) or specific number
- Click **"Generate Video"**

## Model Information

- **Source**: [Wan-AI/Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B)
- **Size**: ~50GB
- **Cache Location**: `/home/caches/Wan2.2-Animate-14B`
- **Parameters**: 14B (14 billion)
- **Type**: Character Animation & Replacement

### Required Files (Auto-downloaded)

```
✓ diffusion_pytorch_model-00001-of-00004.safetensors
✓ diffusion_pytorch_model-00002-of-00004.safetensors
✓ diffusion_pytorch_model-00003-of-00004.safetensors
✓ diffusion_pytorch_model-00004-of-00004.safetensors
✓ models_t5_umt5-xxl-enc-bf16.pth
✓ models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
✓ Wan2.1_VAE.pth
✓ config.json
```

## Example Files

```bash
# Driving video
/workspace/wan22-comfy-project/Wan2.2/examples/pose.mp4

# Reference image
/workspace/wan22-comfy-project/Wan2.2/examples/pose.png
```

## Testing

Run the test suite to verify everything works:

```bash
cd /workspace/wan22-comfy-project/Wan2.2
/workspace/wan22-comfy-project/venv/bin/python test_gradio_animate.py
```

Expected output:
```
✅ ALL TESTS PASSED!
```

## Technical Details

### Environment
- **Python**: `/workspace/wan22-comfy-project/venv/bin/python`
- **Working Dir**: `/workspace/wan22-comfy-project/Wan2.2`
- **Port**: 7862

### GPU Optimization
- **Single GPU mode** for Gradio (simpler, more stable)
- **T5 CPU offloading** to save VRAM
- **FP16/BF16 support** for memory efficiency
- **Compile caching** speeds up subsequent runs

### Directories
- **Models**: `/home/caches/Wan2.2-Animate-14B`
- **Outputs**: `/workspace/wan22-comfy-project/outputs`
- **Preprocessing**: `/workspace/wan22-comfy-project/Wan2.2/preprocessed`

## Advantages of Integration

1. ✅ **Self-contained**: No external script dependencies
2. ✅ **User-friendly**: Download from UI or auto-download on load
3. ✅ **Progress feedback**: Visual progress during operations
4. ✅ **Error handling**: Clear error messages and recovery
5. ✅ **Flexible**: Download-only option for pre-caching
6. ✅ **Small & functional**: ~300 lines of clean, documented code

## Troubleshooting

### Models not downloading?
```bash
# Check HF token
huggingface-cli whoami

# Manual download
cd /workspace/wan22-comfy-project/Wan2.2
/workspace/wan22-comfy-project/venv/bin/python setup_animate_cache.py setup
```

### Preprocessing fails?
- Ensure video and image paths are correct
- Check video is not corrupted
- Try reducing resolution_area

### Generation fails?
- Verify preprocessed path exists
- Check required files: src_pose.mp4, src_face.mp4, src_ref.png
- For replace mode: also need src_bg.mp4, src_mask.mp4

## Performance

- **First run**: Slower (model compilation)
- **Subsequent runs**: Faster (cached compilation)
- **VRAM usage**: ~24GB with T5 CPU offloading
- **Generation time**: ~2-5 min for 77 frames (depends on GPU)

## Next Steps

1. ✅ **Test with examples**: Use provided pose.mp4 and pose.png
2. ✅ **Try your own content**: Upload custom videos and images
3. ✅ **Experiment with parameters**: Different seeds, steps, scales
4. ✅ **Explore modes**: Both animate and replace modes

---

**Status**: ✅ Fully Integrated and Tested  
**Last Updated**: 2025-10-20  
**Environment**: `/workspace/wan22-comfy-project/venv`
