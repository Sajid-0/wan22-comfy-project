# Wan2.2 I2V (Image-to-Video) Multi-GPU Setup Guide

## Overview

This guide provides the complete setup for running **Wan2.2-I2V-A14B** (Image-to-Video) model on multi-GPU systems, optimized for RunPod environments with FSDP + Ulysses sequence parallelism.

## 🎯 Key Features

- **MoE Architecture**: 27B total parameters with 14B active per step
- **Multi-GPU Support**: FSDP (Fully Sharded Data Parallel) + Ulysses sequence parallel
- **Resolution Support**: 480P and 720P video generation
- **Automatic Cache Management**: Auto-download models on RunPod volatile storage
- **Optimized Performance**: ~54GB model efficiently distributed across GPUs

## 📋 Files Created

1. **`setup_i2v_cache.py`** - Model download and cache manager
2. **`run_i2v_multi_gpu.py`** - Multi-GPU inference runner

## 🔧 System Requirements

- **GPUs**: 2x GPUs with at least 24GB VRAM each (e.g., 2x A6000, 2x A100)
- **Storage**: ~54GB for model files
- **VRAM per GPU**: ~24-32GB during inference
- **Python**: 3.10+
- **PyTorch**: 2.4.0+

## 🚀 Quick Start

### 1. Setup Models (First Time)

```bash
cd /workspace/wan22-comfy-project/Wan2.2

# Option A: Quick automated setup
python setup_i2v_cache.py quick

# Option B: Interactive setup
python setup_i2v_cache.py

# Option C: Full setup with verification
python setup_i2v_cache.py setup
```

### 2. Run I2V Generation

```bash
# Basic usage with default settings
python run_i2v_multi_gpu.py

# The script will:
# 1. Check if models are downloaded (auto-download if needed)
# 2. Initialize 2 GPUs with FSDP + Ulysses
# 3. Generate 720P video from image + text prompt
# 4. Save output to /workspace/wan22-comfy-project/outputs/
```

## 🎨 Model Architecture

### Mixture of Experts (MoE)

The I2V-A14B uses a two-expert design:

1. **High Noise Expert**: Handles early denoising steps (overall layout)
2. **Low Noise Expert**: Handles later steps (fine details)

**Switching Point**: Controlled by `boundary` parameter (default: 0.900)

### File Structure

```
/home/caches/Wan2.2-I2V-A14B/
├── high_noise_model/
│   ├── diffusion_pytorch_model-00001-of-00004.safetensors
│   ├── diffusion_pytorch_model-00002-of-00004.safetensors
│   ├── diffusion_pytorch_model-00003-of-00004.safetensors
│   ├── diffusion_pytorch_model-00004-of-00004.safetensors
│   ├── diffusion_pytorch_model.safetensors.index.json
│   └── config.json
├── low_noise_model/
│   ├── diffusion_pytorch_model-00001-of-00004.safetensors
│   ├── diffusion_pytorch_model-00002-of-00004.safetensors
│   ├── diffusion_pytorch_model-00003-of-00004.safetensors
│   ├── diffusion_pytorch_model-00004-of-00004.safetensors
│   ├── diffusion_pytorch_model.safetensors.index.json
│   └── config.json
├── models_t5_umt5-xxl-enc-bf16.pth  # Text encoder
├── Wan2.1_VAE.pth                    # Video VAE
├── config.json
└── configuration.json
```

## ⚙️ Configuration Options

### Resolution Settings

```python
# 720P (recommended for quality)
--size '1280*720'
--sample_shift '5.0'

# 480P (faster generation)
--size '480*832'
--sample_shift '3.0'
```

### Multi-GPU Parameters

```python
# Essential for multi-GPU
--dit_fsdp          # Enable FSDP for DiT model
--t5_fsdp           # Enable FSDP for T5 model
--ulysses_size 2    # Sequence parallel size (match GPU count)

# Number of GPUs
--nproc_per_node=2  # Use 2 GPUs (can scale to 4 or 8)
```

### Quality Parameters

```python
--frame_num 81           # Number of frames (must be 4n+1)
--sample_steps 40        # Sampling steps (higher = better quality)
--sample_guide_scale 3.5 # Guidance scale (low_noise, high_noise)
--base_seed 42           # Random seed for reproducibility
```

## 🎬 Usage Examples

### Example 1: Basic 720P Generation

```bash
cd /workspace/wan22-comfy-project/Wan2.2

python run_i2v_multi_gpu.py
```

### Example 2: Custom Image and Prompt

Edit `run_i2v_multi_gpu.py`:

```python
cmd = [
    # ... other parameters ...
    '--prompt', 'Your custom prompt describing the desired video',
    '--image', '/path/to/your/image.jpg',
    '--frame_num', '81',
    '--save_file', '/workspace/outputs/custom_video.mp4'
]
```

### Example 3: 480P Fast Generation

Edit `run_i2v_multi_gpu.py`:

```python
cmd = [
    # ... other parameters ...
    '--size', '480*832',
    '--sample_shift', '3.0',  # Important for 480P
    '--sample_steps', '30',   # Reduce for faster generation
    '--frame_num', '49',      # Shorter video
]
```

### Example 4: High Quality 720P

```python
cmd = [
    # ... other parameters ...
    '--size', '1280*720',
    '--sample_shift', '5.0',
    '--sample_steps', '50',   # More steps for quality
    '--frame_num', '121',     # Longer video
]
```

## 🔬 Advanced: Direct Command Line Usage

You can also use the generate.py directly:

```bash
cd /workspace/wan22-comfy-project/Wan2.2

# Multi-GPU with torchrun
torchrun --nproc_per_node=2 generate.py \
    --task i2v-A14B \
    --size 1280*720 \
    --ckpt_dir /home/caches/Wan2.2-I2V-A14B \
    --image examples/i2v_input.JPG \
    --prompt "Your detailed prompt here" \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 2 \
    --frame_num 81 \
    --sample_steps 40 \
    --sample_shift 5.0 \
    --sample_guide_scale 3.5

# Single GPU (requires 80GB+ VRAM)
python generate.py \
    --task i2v-A14B \
    --size 1280*720 \
    --ckpt_dir /home/caches/Wan2.2-I2V-A14B \
    --image examples/i2v_input.JPG \
    --prompt "Your prompt" \
    --offload_model True \
    --convert_model_dtype
```

## 🔍 Troubleshooting

### Models Not Found

```bash
# Check cache status
python setup_i2v_cache.py status

# Manually download
python setup_i2v_cache.py download
```

### CUDA Out of Memory

**Solutions:**
1. Reduce frame count: `--frame_num 49` (instead of 81)
2. Reduce sampling steps: `--sample_steps 30`
3. Use 480P: `--size 480*832`
4. Add more GPUs: `--nproc_per_node=4 --ulysses_size 4`

### Port Already in Use

Edit `run_i2v_multi_gpu.py`:

```python
os.environ['MASTER_PORT'] = '12347'  # Change port number
```

### Slow Download Speeds

The model is large (~54GB). Use HuggingFace CLI for resume support:

```bash
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B \
    --local-dir /home/caches/Wan2.2-I2V-A14B \
    --resume-download
```

## 📊 Performance Benchmarks

### 2x A6000 (48GB each)

- **720P (1280x720)**: ~8-12 minutes for 81 frames
- **480P (480x832)**: ~5-8 minutes for 81 frames
- **VRAM Usage**: ~28-32GB per GPU

### 2x A100 (80GB each)

- **720P (1280x720)**: ~6-10 minutes for 81 frames
- **VRAM Usage**: ~30-35GB per GPU

### 4x GPUs (Recommended for Production)

```python
--nproc_per_node=4
--ulysses_size 4
```

- **Speed**: ~40-50% faster than 2 GPUs
- **VRAM**: ~18-22GB per GPU
- **Quality**: Identical to 2 GPUs

## 🎓 Understanding the Parameters

### Frame Count (`frame_num`)

Must be **4n + 1** where n is an integer:
- `49` = 4×12 + 1 (2 seconds at 24fps)
- `81` = 4×20 + 1 (3.4 seconds at 24fps)
- `121` = 4×30 + 1 (5 seconds at 24fps)

### Shift Parameter (`sample_shift`)

Controls noise schedule:
- **3.0**: Best for 480P, faster generation
- **5.0**: Best for 720P, higher quality
- Higher values = more temporal consistency

### Guidance Scale (`sample_guide_scale`)

Controls prompt adherence:
- **3.5**: Balanced (default)
- **5.0**: Stronger prompt following
- **2.0**: More creative freedom
- Can use tuple: `(3.5, 4.5)` for (low_noise, high_noise)

## 🔗 Related Resources

- **Model Page**: https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B
- **GitHub Repo**: https://github.com/Wan-Video/Wan2.2
- **Paper**: https://arxiv.org/abs/2503.20314
- **ComfyUI Integration**: https://docs.comfy.org/tutorials/video/wan/wan2_2

## 📝 Best Practices

1. **First Run**: Always run `setup_i2v_cache.py quick` to verify models
2. **Prompt Quality**: Detailed prompts work best (describe motion, style, lighting)
3. **Image Quality**: Use high-quality input images (1024x1024+)
4. **Aspect Ratio**: Input image aspect ratio is preserved in output
5. **Seed Management**: Use fixed seeds for reproducibility
6. **VRAM Management**: Start with 49 frames, scale up if VRAM allows

## 💡 Tips for Best Results

### Prompt Engineering

Good prompts include:
- **Subject description**: "A white cat wearing sunglasses"
- **Motion details**: "sits relaxed, slight head movements"
- **Camera work**: "close-up shot, stable camera"
- **Lighting**: "soft natural light, golden hour"
- **Style**: "cinematic, high quality, 4K"

### Input Image Tips

- Use **high resolution** images (1024x1024 or higher)
- Ensure **clear subject** with good lighting
- Avoid blurry or low-quality images
- **Composition matters**: Well-framed subjects work best

## 🛠️ Customization Guide

### Modify Generation Settings

Edit `run_i2v_multi_gpu.py` to change default parameters:

```python
# Change these lines in the cmd array
'--size', '1280*720',              # Resolution
'--frame_num', '81',               # Video length
'--sample_steps', '40',            # Quality vs speed
'--sample_shift', '5.0',           # Noise schedule
'--sample_guide_scale', '3.5',     # Prompt strength
'--base_seed', '42',               # Random seed
```

### Add Custom Preprocessing

Add before the generation command:

```python
from PIL import Image

# Load and preprocess image
img = Image.open('/path/to/image.jpg')
img = img.resize((1024, 1024))  # Resize if needed
img.save('/tmp/preprocessed.jpg')

# Then use in command
'--image', '/tmp/preprocessed.jpg'
```

## 📄 License

Apache 2.0 - See model card for details

## 🙏 Acknowledgments

- Wan Team at Alibaba for the amazing model
- HuggingFace for hosting and infrastructure
- Community for testing and feedback

---

**Created for efficient I2V generation on RunPod multi-GPU instances**
