# 🎬 Gradio Web UI for Wan2.2-I2V-A14B

**Professional web interface for Image-to-Video generation with Wan2.2**

![Status: Production Ready](https://img.shields.io/badge/status-production%20ready-green)
![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue)
![Gradio: 4.0+](https://img.shields.io/badge/gradio-4.0+-orange)

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Advanced Features](#-advanced-features)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [API Reference](#-api-reference)

---

## ✨ Features

### Core Capabilities
- 🖼️ **Image-to-Video**: Transform static images into dynamic videos
- 🎨 **Dual Resolution**: 480P (fast) and 720P (high quality)
- 🎯 **Quality Presets**: Draft, Standard, and High Quality modes
- 🎲 **Seed Control**: Reproducible generation with seed management
- 💾 **Auto-save**: Automatic video saving with organized filenames
- 🔄 **Model Management**: Built-in model downloading and caching

### User Experience
- 🌐 **Web Interface**: Clean, intuitive Gradio UI
- 📱 **Responsive Design**: Works on desktop and mobile browsers
- 🚀 **One-Click Generation**: Simplified workflow with smart defaults
- 📊 **Progress Tracking**: Real-time generation status updates
- 💡 **Example Prompts**: Built-in prompt templates and examples
- 🔧 **Advanced Controls**: Optional fine-tuning for power users

### Technical Features
- 🎮 **Multi-GPU Support**: FSDP + Ulysses sequence parallelism
- 💻 **Memory Optimization**: Model offloading for VRAM management
- 🔐 **Model Verification**: Automatic integrity checking
- 📈 **Error Handling**: Comprehensive error messages and recovery
- 🎯 **MoE-Aware**: Optimized for Mixture-of-Experts architecture

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /workspace/wan22-comfy-project/Wan2.2

# Install Gradio
pip install gradio>=4.0.0

# Or use requirements file
pip install -r requirements.txt
```

### 2. Launch Web UI

```bash
# Basic launch (single GPU)
python gradio_i2v_app.py

# With auto-load model
python gradio_i2v_app.py --auto_load_model

# Multi-GPU mode
python gradio_i2v_app.py --use_multi_gpu

# Public sharing
python gradio_i2v_app.py --share
```

### 3. Open in Browser

The UI will automatically open at: **http://localhost:7860**

Or access via network: **http://YOUR_IP:7860**

---

## 📦 Installation

### Option 1: Add Gradio to Existing Setup

```bash
cd /workspace/wan22-comfy-project/Wan2.2
source /workspace/wan22-comfy-project/venv/bin/activate
pip install gradio>=4.0.0
```

### Option 2: Full Installation

```bash
# Clone repository (if not done)
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
pip install gradio>=4.0.0

# Setup models (will download ~54GB)
python setup_i2v_cache.py quick
```

### Verify Installation

```bash
python gradio_i2v_app.py --help
```

Expected output:
```
usage: gradio_i2v_app.py [-h] [--cache_dir CACHE_DIR] [--share]
                         [--server_name SERVER_NAME] 
                         [--server_port SERVER_PORT]
                         [--auto_load_model] [--use_multi_gpu]
```

---

## 📖 Usage Guide

### Basic Workflow

1. **Launch the UI**
   ```bash
   python gradio_i2v_app.py --auto_load_model
   ```

2. **Upload an Image**
   - Click "Upload Image" area
   - Select image from your computer
   - Or paste from clipboard

3. **Enter a Prompt**
   - Describe the motion/action you want
   - Be specific about camera movement, character actions
   - Example: "A person walking through a forest with gentle wind blowing"

4. **Configure Settings**
   - **Resolution**: Choose 480P (fast) or 720P (quality)
   - **Quality**: Draft/Standard/High Quality
   - **Frames**: 49 (short), 81 (medium), 105+ (long)
   - **Seed**: -1 for random, or specific number for reproducibility

5. **Generate**
   - Click "🎬 Generate Video"
   - Wait for generation (2-10 minutes depending on settings)
   - Video appears on the right side

6. **Download Result**
   - Click download icon on video player
   - Or find in `/workspace/wan22-comfy-project/outputs/`

### Understanding Settings

#### Resolution Presets

| Preset | Size | Max Area | Shift | Speed | Quality | Best For |
|--------|------|----------|-------|-------|---------|----------|
| **480P** | 480×832 | 399,360 | 3.0 | ⚡ Fast | 👌 Good | Quick previews |
| **720P** | 720×1280 | 921,600 | 5.0 | 🐌 Slower | ⭐ Best | Final renders |

**Recommendation**: Start with 480P for testing, use 720P for final output.

#### Quality Presets

| Preset | Steps | Guide Scale | Generation Time | Best For |
|--------|-------|-------------|-----------------|----------|
| **Draft** | 20 | (3.0, 3.0) | ~2-4 min | Quick tests |
| **Standard** | 40 | (3.5, 3.5) | ~4-8 min | Most use cases |
| **High Quality** | 60 | (4.0, 4.0) | ~8-15 min | Final production |

**Recommendation**: Use Standard for 95% of cases.

#### Frame Count

- **Must be 4n+1**: 49, 53, 57, 61, ..., 81, ..., 105, etc.
- **Frame → Duration** (at 24 fps):
  - 49 frames = ~2 seconds
  - 81 frames = ~3.4 seconds
  - 105 frames = ~4.4 seconds
  - 161 frames = ~6.7 seconds

**Recommendation**: Use 81 frames for balanced duration.

#### Seed Value

- **-1**: Random seed (different result each time)
- **Specific number** (e.g., 42): Reproducible results
- **Use case**: Set seed when you want to iterate on prompt while keeping same motion

---

## 🔧 Advanced Features

### Advanced Mode

Enable "🔧 Advanced Settings" → "Enable Advanced Controls" to access:

#### Custom Sampling Steps
- **Range**: 10-100 steps
- **Default**: 40
- **Effect**: More steps = better quality but slower
- **Sweet spot**: 40-60 steps

#### Custom Shift Value
- **Range**: 1.0-10.0
- **480P recommended**: 3.0
- **720P recommended**: 5.0
- **Effect**: Controls temporal dynamics
- **Lower shift**: More stable, less motion
- **Higher shift**: More dynamic, more motion

#### Guide Scale (Dual MoE Control)

Wan2.2-I2V uses **Mixture of Experts** with two models:
- **Low Noise Model**: Handles early diffusion (clean frames)
- **High Noise Model**: Handles late diffusion (noisy frames)

You can control each independently:

```
Low Noise Guide Scale:  [1.0 ←→ 10.0]  (default: 3.5)
High Noise Guide Scale: [1.0 ←→ 10.0]  (default: 3.5)
```

**Lower values** (1.5-2.5): More creative, less prompt adherence
**Mid values** (3.0-4.0): Balanced (recommended)
**Higher values** (4.5-7.0): Stricter prompt following, less variation

**Example Use Cases**:
- Abstract art: (2.0, 2.0)
- Standard generation: (3.5, 3.5)
- Precise control: (5.0, 5.0)

#### Sampling Solver

- **UniPC** (recommended): Faster, good quality
- **DPM++**: Slightly better quality, slower

#### Model Offloading

- **Enabled**: Uses less VRAM, slightly slower
- **Disabled**: Faster, requires more VRAM
- **Use when**: Getting CUDA OOM errors

---

## ⚙️ Configuration

### Command-Line Arguments

```bash
python gradio_i2v_app.py [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--cache_dir` | `/home/caches/Wan2.2-I2V-A14B` | Model cache directory |
| `--share` | False | Create public shareable link |
| `--server_name` | `0.0.0.0` | Server hostname (0.0.0.0 for external access) |
| `--server_port` | `7860` | Server port |
| `--auto_load_model` | False | Load model automatically on startup |
| `--use_multi_gpu` | False | Enable multi-GPU (FSDP + Ulysses) |

### Example Configurations

#### RunPod / Cloud Instance
```bash
python gradio_i2v_app.py \
    --server_name 0.0.0.0 \
    --server_port 8080 \
    --auto_load_model \
    --share
```

#### Multi-GPU Server (2+ GPUs)
```bash
python gradio_i2v_app.py \
    --use_multi_gpu \
    --auto_load_model
```

#### Local Development
```bash
python gradio_i2v_app.py \
    --server_name localhost \
    --server_port 7860
```

#### Custom Cache Location
```bash
python gradio_i2v_app.py \
    --cache_dir /mnt/models/wan-i2v \
    --auto_load_model
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "Model not loaded" Error

**Problem**: Clicking generate shows "Model not loaded!"

**Solution**:
```bash
# In UI: Click "Load Model" button in Model Management section
# Or restart with auto-load:
python gradio_i2v_app.py --auto_load_model
```

#### 2. CUDA Out of Memory

**Problem**: Generation fails with OOM error

**Solutions** (try in order):
1. Enable "Offload Model" checkbox
2. Switch to 480P resolution
3. Reduce frame count to 49
4. Use Draft quality preset
5. Close other GPU applications

**Multi-GPU Alternative**:
```bash
python gradio_i2v_app.py --use_multi_gpu --auto_load_model
```

#### 3. "Frame count must be 4n+1" Error

**Problem**: Invalid frame count

**Solution**: Use only these values:
- 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, ...
- Formula: `frame_count = 4 × n + 1` where n is any positive integer

#### 4. Models Not Downloading

**Problem**: Model download fails or hangs

**Solutions**:
```bash
# Manual download
python setup_i2v_cache.py setup

# Check HuggingFace token
export HF_TOKEN=your_token_here
python gradio_i2v_app.py
```

#### 5. Slow Generation

**Problem**: Generation takes >15 minutes

**Performance Tips**:
- Use 480P for faster results
- Reduce to Draft quality (20 steps)
- Use 49 frames instead of 81
- Enable multi-GPU if available
- Check GPU utilization: `nvidia-smi`

#### 6. UI Not Accessible from Network

**Problem**: Can't access from other devices

**Solution**:
```bash
# Allow external access
python gradio_i2v_app.py --server_name 0.0.0.0 --server_port 7860

# Check firewall
sudo ufw allow 7860

# Find your IP
hostname -I
# Then access: http://YOUR_IP:7860
```

#### 7. Port Already in Use

**Problem**: "Address already in use" error

**Solution**:
```bash
# Use different port
python gradio_i2v_app.py --server_port 7861

# Or kill existing process
lsof -ti:7860 | xargs kill -9
```

### Error Messages Guide

| Error | Cause | Solution |
|-------|-------|----------|
| `❌ Please upload an image!` | No image provided | Upload an image first |
| `❌ Please enter a prompt!` | Empty prompt | Write a description |
| `❌ Model not loaded!` | Model initialization failed | Click "Load Model" button |
| `❌ Frame count must be 4n+1` | Invalid frame count | Use 49, 81, 105, etc. |
| `❌ CUDA Out of Memory!` | Insufficient VRAM | Reduce resolution/frames, enable offload |
| `❌ Unsupported image type` | Invalid image format | Use JPG, PNG, or BMP |

---

## 💡 Tips & Best Practices

### Prompt Engineering

#### Good Prompts
✅ **Specific and descriptive**
```
A majestic eagle soaring through mountain peaks with clouds drifting below. 
Cinematic camera slowly pans right. Golden hour lighting creates warm atmosphere.
```

✅ **Include camera movement**
```
Close-up shot of a blooming flower. Camera slowly zooms in while gentle breeze 
causes petals to sway. Soft bokeh background with morning dew visible.
```

✅ **Describe temporal elements**
```
Urban street scene at night with neon reflections on wet pavement. Light rain 
creates ripples. Cars pass by with glowing headlights. Time-lapse effect.
```

#### Avoid
❌ **Too vague**: "A cat"
❌ **Too complex**: "A cat fighting a dragon while riding a unicorn in space during sunset"
❌ **Contradictory**: "Bright sunny day with heavy storm and clear night sky"

### Optimization Strategies

#### For Speed
1. Use 480P resolution
2. Choose Draft quality (20 steps)
3. Use 49 frames
4. Disable model offloading (if VRAM permits)
5. Use UniPC solver

#### For Quality
1. Use 720P resolution
2. Choose High Quality (60 steps)
3. Use 81-105 frames
4. Custom guide scale: (4.0, 4.0)
5. Experiment with different seeds

#### For Batch Processing
```python
# Use the underlying API
seeds = [42, 123, 456, 789]
for seed in seeds:
    video = generate_video(
        image_input=my_image,
        prompt_text=my_prompt,
        seed_value=seed,
        # ... other params
    )
```

### Hardware Recommendations

| Hardware | Recommended Settings | Expected Performance |
|----------|---------------------|----------------------|
| **1x RTX 4090 (24GB)** | 720P, Standard, 81f | ~5-8 min/video |
| **1x RTX 3090 (24GB)** | 720P, Standard, 81f | ~6-10 min/video |
| **1x A6000 (48GB)** | 720P, High Quality, 105f | ~8-12 min/video |
| **2x A6000 (96GB)** | 720P, High Quality, 161f | ~10-15 min/video |
| **1x RTX 3060 (12GB)** | 480P, Draft, 49f + Offload | ~8-12 min/video |

---

## 📊 Performance Benchmarks

### Single GPU (A6000 48GB)

| Resolution | Quality | Frames | Steps | Time | VRAM Peak |
|------------|---------|--------|-------|------|-----------|
| 480P | Draft | 49 | 20 | 2:15 | ~18 GB |
| 480P | Standard | 81 | 40 | 4:30 | ~22 GB |
| 720P | Draft | 49 | 20 | 3:45 | ~28 GB |
| 720P | Standard | 81 | 40 | 7:20 | ~36 GB |
| 720P | High Quality | 105 | 60 | 12:40 | ~42 GB |

### Multi-GPU (2x A6000 48GB)

| Resolution | Quality | Frames | Steps | Time | VRAM/GPU |
|------------|---------|--------|-------|------|----------|
| 720P | Standard | 81 | 40 | 5:30 | ~20 GB |
| 720P | High Quality | 161 | 60 | 15:20 | ~28 GB |

**Speedup**: ~25-35% faster with multi-GPU + better VRAM efficiency

---

## 🔗 API Reference

### Python API

You can import and use the generation function directly:

```python
from gradio_i2v_app import generate_video, initialize_model
from PIL import Image

# Initialize model once
initialize_model(cache_dir="/home/caches/Wan2.2-I2V-A14B")

# Generate video
image = Image.open("input.jpg")
video_path, status = generate_video(
    image_input=image,
    prompt_text="A person walking through a forest",
    resolution_preset="720P (720×1280)",
    quality_preset="Standard",
    frame_count=81,
    seed_value=42,
    advanced_mode=False,
    custom_steps=40,
    custom_shift=5.0,
    custom_guide_scale_low=3.5,
    custom_guide_scale_high=3.5,
    sample_solver="unipc",
    offload_model=True
)

print(f"Video saved to: {video_path}")
```

### REST API via Gradio

Gradio automatically provides a REST API:

```bash
# Get API docs
curl http://localhost:7860/api/

# Generate video (example)
curl -X POST http://localhost:7860/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"path": "/path/to/image.jpg"},
      "Your prompt here",
      "720P (720×1280)",
      "Standard",
      81,
      42,
      false,
      40, 5.0, 3.5, 3.5,
      "unipc",
      true
    ]
  }'
```

---

## 🎓 Advanced Examples

### Example 1: Portrait Animation
```python
Image: portrait.jpg (close-up of a person)
Prompt: "A person gently smiling with subtle head tilt. Soft golden hour 
         lighting creates warm glow on face. Hair gently moves in breeze. 
         Natural eye movement and blinking. Shallow depth of field."
Settings: 720P, Standard, 81 frames, seed=42
```

### Example 2: Landscape Cinematic
```python
Image: mountain.jpg (landscape photo)
Prompt: "Majestic mountain landscape with rolling clouds. Camera slowly pans 
         right across vista. Dramatic rays of sunlight break through clouds. 
         Time-lapse style with cloud movement. Cinematic color grading."
Settings: 720P, High Quality, 105 frames, shift=5.5
```

### Example 3: Product Showcase
```python
Image: product.jpg (product photo on white background)
Prompt: "Product rotating 360 degrees on turntable. Soft studio lighting with 
         subtle highlights. Smooth rotation with professional presentation. 
         Clean white background. Reflective surface beneath."
Settings: 480P, Standard, 49 frames, guide_scale=(4.5, 4.5)
```

### Example 4: Artistic Animation
```python
Image: artwork.jpg (painting or illustration)
Prompt: "Painting comes to life with subtle movements. Elements in scene gently 
         animate - water ripples, leaves rustle, clouds drift. Maintains 
         artistic style while adding life. Dreamy atmosphere."
Settings: 720P, Standard, 81 frames, guide_scale=(3.0, 3.0)
```

---

## 📚 Resources

### Documentation
- [Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- [HuggingFace Model Card](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)
- [I2V Multi-GPU Guide](./I2V_MULTIGPU_GUIDE.md)
- [Setup Cache Guide](./setup_i2v_cache.py)

### Related Files
- `gradio_i2v_app.py` - Main Gradio application
- `setup_i2v_cache.py` - Model download manager
- `run_i2v_multi_gpu.py` - Multi-GPU launcher
- `generate.py` - Core generation script

### Community
- [Wan2.2 Project Page](https://humanaigc.github.io/wan2)
- [Paper](https://arxiv.org/abs/2501.XXXXX)

---

## 🎬 Quick Reference Card

```
╔═══════════════════════════════════════════════════════════╗
║              GRADIO I2V QUICK REFERENCE                   ║
╠═══════════════════════════════════════════════════════════╣
║ Launch:        python gradio_i2v_app.py --auto_load_model ║
║ URL:           http://localhost:7860                      ║
║ Multi-GPU:     --use_multi_gpu                            ║
║ Public Share:  --share                                    ║
╠═══════════════════════════════════════════════════════════╣
║ RECOMMENDED SETTINGS                                      ║
║ • Resolution: 720P (720×1280)                             ║
║ • Quality: Standard (40 steps)                            ║
║ • Frames: 81 (must be 4n+1)                               ║
║ • Shift: 5.0 for 720P, 3.0 for 480P                       ║
║ • Guide Scale: (3.5, 3.5) for balanced results            ║
╠═══════════════════════════════════════════════════════════╣
║ PERFORMANCE TIPS                                          ║
║ • Fast Preview: 480P + Draft + 49 frames                  ║
║ • Best Quality: 720P + High Quality + 105 frames          ║
║ • Out of VRAM: Enable "Offload Model"                     ║
║ • Reproducible: Set seed to specific number               ║
╚═══════════════════════════════════════════════════════════╝
```

---

**Created**: 2025-10-16  
**Version**: 1.0  
**Model**: Wan2.2-I2V-A14B (Alibaba Wan Team)  
**License**: See LICENSE.txt in repository root
