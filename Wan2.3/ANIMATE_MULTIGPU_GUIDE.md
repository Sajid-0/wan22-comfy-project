# Wan2.2 Animate-14B Multi-GPU Setup Guide

## Overview

Complete setup for **Wan2.2-Animate-14B** - Character animation and replacement model with motion transfer capabilities. Optimized for RunPod multi-GPU environments with FSDP + Ulysses sequence parallelism.

## 🎯 Key Features

- **Character Animation**: Make any character mimic motions from a driving video
- **Character Replacement**: Swap characters in existing videos while preserving motion
- **Motion Transfer**: Holistic movement and expression replication
- **Multi-GPU Support**: FSDP + Ulysses for efficient 14B model inference
- **Single Model**: 14B parameters (NOT MoE like I2V/T2V)
- **Unique Components**: Motion encoder + Face encoder + CLIP model + T5

## 📋 Files Created

1. **`setup_animate_cache.py`** - Model download and cache manager
2. **`run_animate_multi_gpu.py`** - Multi-GPU inference runner

## 🔧 System Requirements

- **GPUs**: 2x GPUs with 24GB+ VRAM each (e.g., 2x A6000, 2x A100)
- **Storage**: ~50GB+ for model files
- **VRAM per GPU**: ~28-32GB during inference
- **Python**: 3.10+
- **PyTorch**: 2.4.0+

## 🚀 Quick Start

### 1. Setup Models (First Time)

```bash
cd /workspace/wan22-comfy-project/Wan2.2

# Quick automated setup
python setup_animate_cache.py quick
```

### 2. Preprocess Your Video (REQUIRED!)

**This step is MANDATORY - you cannot skip it!**

```bash
# For Animation Mode (character mimics motion)
python wan/modules/animate/preprocess/preprocess_data.py \
    --ckpt_path /home/caches/Wan2.2-Animate-14B/process_checkpoint \
    --video_path /path/to/driving_video.mp4 \
    --refer_path /path/to/character_image.jpg \
    --save_path ./preprocessed_output \
    --resolution_area 1280 720 \
    --retarget_flag \
    --use_flux

# For Replacement Mode (swap character in video)
python wan/modules/animate/preprocess/preprocess_data.py \
    --ckpt_path /home/caches/Wan2.2-Animate-14B/process_checkpoint \
    --video_path /path/to/video.mp4 \
    --refer_path /path/to/new_character.jpg \
    --save_path ./preprocessed_output \
    --resolution_area 1280 720 \
    --iterations 3 \
    --k 7 \
    --replace_flag
```

**Preprocessing Output:** Creates `src_pose.mp4`, `src_face.mp4`, `src_ref.png` (and `src_bg.mp4`, `src_mask.mp4` for replacement)

### 3. Run Animation Generation

```bash
# Edit run_animate_multi_gpu.py to point to your preprocessed data
# Then run:
python run_animate_multi_gpu.py
```

## 🎨 Model Architecture

### Single 14B Model (NOT MoE)

Unlike I2V-A14B and T2V-A14B, Animate-14B is a **single 14B model**, not MoE:

```
Wan2.2-Animate-14B (14B parameters)
├── DiT Model: 14B parameters
├── T5 Encoder: Text understanding
├── CLIP Model: Visual-text alignment
├── Motion Encoder: Pose/motion encoding (512 dim)
├── Face Encoder: Facial feature extraction
├── VAE: Video encoding/decoding
└── Optional: Relighting LoRA (for replacement mode)
```

### File Structure

```
/home/caches/Wan2.2-Animate-14B/
├── diffusion_pytorch_model-00001-of-00004.safetensors  # Main DiT
├── diffusion_pytorch_model-00002-of-00004.safetensors
├── diffusion_pytorch_model-00003-of-00004.safetensors
├── diffusion_pytorch_model-00004-of-00004.safetensors
├── diffusion_pytorch_model.safetensors.index.json
├── models_t5_umt5-xxl-enc-bf16.pth                      # T5 text encoder
├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth  # CLIP
├── xlm-roberta-large/                                   # CLIP tokenizer
│   └── pytorch_model.bin
├── Wan2.1_VAE.pth                                       # Video VAE
├── relighting_lora.ckpt                                 # Optional LoRA
├── config.json
├── configuration.json
└── process_checkpoint/                                  # Preprocessing models
    ├── sam2.1_hiera_large.pt                           # SAM2 for segmentation
    ├── face_landmarker.task                            # MediaPipe face
    ├── pose_landmarker_heavy.task                      # MediaPipe pose
    └── other preprocessing checkpoints
```

## ⚙️ Configuration Options

### Two Modes

**1. Animation Mode** (Character mimics motion):
```python
'--src_root_path', './preprocessed_output'
'--refert_num', '1'       # Reference frames (1 or 5)
# No --replace_flag
```

**2. Replacement Mode** (Swap character):
```python
'--src_root_path', './preprocessed_output'
'--refert_num', '1'
'--replace_flag'           # Enable replacement
'--use_relighting_lora'   # Use lighting adaptation LoRA
```

### Multi-GPU Parameters

```python
--dit_fsdp          # Enable FSDP for DiT model
--t5_fsdp           # Enable FSDP for T5 model
--ulysses_size 2    # Sequence parallel size
--nproc_per_node=2  # Number of GPUs
```

### Generation Parameters

```python
--frame_num 77          # Frames per clip (must be 4n+1)
--refert_num 1          # Reference frames (1 or 5 recommended)
--sample_steps 20       # Sampling steps (20 is default for Animate)
--sample_shift 5.0      # Noise schedule shift
--sample_guide_scale 1.0  # Guidance (usually 1.0, can increase for expression control)
--base_seed 42          # Random seed
```

## 📊 Performance (2x A6000 48GB)

| Mode | Frames | Steps | Time | VRAM/GPU |
|------|--------|-------|------|----------|
| Animation | 77 | 20 | 6-10 min | ~30GB |
| Replacement | 77 | 20 | 7-12 min | ~32GB |
| Animation | 49 | 15 | 4-6 min | ~28GB |

## 🎬 Usage Examples

### Example 1: Basic Animation

```bash
# 1. Preprocess
python wan/modules/animate/preprocess/preprocess_data.py \
    --ckpt_path /home/caches/Wan2.2-Animate-14B/process_checkpoint \
    --video_path dance_video.mp4 \
    --refer_path anime_character.jpg \
    --save_path ./dance_preprocessed \
    --resolution_area 1280 720 \
    --retarget_flag \
    --use_flux

# 2. Generate (edit run_animate_multi_gpu.py to use ./dance_preprocessed)
python run_animate_multi_gpu.py
```

### Example 2: Character Replacement

```bash
# 1. Preprocess
python wan/modules/animate/preprocess/preprocess_data.py \
    --ckpt_path /home/caches/Wan2.2-Animate-14B/process_checkpoint \
    --video_path original_video.mp4 \
    --refer_path new_character.jpg \
    --save_path ./replace_preprocessed \
    --resolution_area 1280 720 \
    --iterations 3 \
    --k 7 \
    --replace_flag

# 2. Generate with replacement flags
# Edit run_animate_multi_gpu.py:
#   - Uncomment --replace_flag
#   - Uncomment --use_relighting_lora
#   - Point to ./replace_preprocessed
python run_animate_multi_gpu.py
```

### Example 3: Direct Command Line

```bash
cd /workspace/wan22-comfy-project/Wan2.2

# Animation mode
torchrun --nproc_per_node=2 generate.py \
    --task animate-14B \
    --ckpt_dir /home/caches/Wan2.2-Animate-14B \
    --src_root_path ./preprocessed_output \
    --refert_num 1 \
    --dit_fsdp --t5_fsdp --ulysses_size 2 \
    --frame_num 77 \
    --sample_steps 20

# Replacement mode
torchrun --nproc_per_node=2 generate.py \
    --task animate-14B \
    --ckpt_dir /home/caches/Wan2.2-Animate-14B \
    --src_root_path ./preprocessed_output \
    --refert_num 1 \
    --replace_flag \
    --use_relighting_lora \
    --dit_fsdp --t5_fsdp --ulysses_size 2
```

## 🔍 Troubleshooting

### Preprocessing Fails

**Problem:** SAM2 or MediaPipe models not found

**Solution:**
```bash
# Check if preprocessing checkpoints exist
ls -lh /home/caches/Wan2.2-Animate-14B/process_checkpoint/

# If missing, re-download
python setup_animate_cache.py download --force-redownload
```

### CUDA Out of Memory

**Solutions:**
1. Reduce frames: `--frame_num 49`
2. Reduce steps: `--sample_steps 15`
3. Use single reference: `--refert_num 1`
4. Add more GPUs: `--nproc_per_node=4 --ulysses_size 4`

### Generated Video Quality Issues

**Problem:** Poor motion transfer or artifacts

**Solutions:**
1. Improve preprocessing:
   - Use higher resolution input video
   - Ensure clean pose tracking in driving video
   - Use `--use_flux` for better quality
2. Increase sampling steps: `--sample_steps 30`
3. For replacement mode, try different iterations: `--iterations 5`

### Preprocessed Files Not Found

**Problem:** `src_pose.mp4` or other files missing

**Check:**
```bash
ls -lh ./preprocessed_output/
# Should see: src_pose.mp4, src_face.mp4, src_ref.png
# For replacement: also src_bg.mp4, src_mask.mp4
```

## 📚 Preprocessing Guide

### Understanding Preprocessing

Preprocessing extracts:
- **Pose**: Body keypoints and movement from driving video
- **Face**: Facial features and expressions
- **Reference**: Character appearance from image
- **Background** (replacement only): Scene without character
- **Mask** (replacement only): Character segmentation

### Preprocessing Parameters

```bash
--ckpt_path         # Path to preprocessing models
--video_path        # Input driving video
--refer_path        # Character reference image
--save_path         # Output directory
--resolution_area   # Target area (e.g., 1280 720)
--fps              # Output FPS (default: 30)
--retarget_flag    # For animation mode
--replace_flag     # For replacement mode
--use_flux         # Use Flux for better quality
--iterations       # Refinement iterations (replacement)
--k                # Segmentation parameter (replacement)
```

### Preprocessing Best Practices

1. **Input Video**:
   - Clear, stable footage
   - Good lighting
   - Visible full body (for animation)
   - 720P or higher resolution

2. **Reference Image**:
   - High quality (1024x1024+)
   - Clear character features
   - Frontal or 3/4 view works best
   - Similar aspect ratio to video

3. **For Animation**:
   - Use `--retarget_flag`
   - Use `--use_flux` for quality
   - Ensure driving video has clear motion

4. **For Replacement**:
   - Use `--replace_flag`
   - Set `--iterations 3-5` for refinement
   - Character in video should be clearly visible

## 🎓 Understanding Frame Parameters

### frame_num (clip_len)

Must be **4n + 1** where n is an integer:
- `49` = 4×12 + 1 (~1.6 seconds at 30fps)
- `77` = 4×19 + 1 (~2.6 seconds at 30fps) **[Default]**
- `105` = 4×26 + 1 (~3.5 seconds at 30fps)

### refert_num

Reference frames for temporal guidance:
- `1`: Single frame reference (faster, standard)
- `5`: Multi-frame reference (better consistency)

## 🔬 Advanced: Scaling to More GPUs

### 4 GPUs

Edit `run_animate_multi_gpu.py`:
```python
os.environ['WORLD_SIZE'] = '4'
'--nproc_per_node=4'
'--ulysses_size', '4'
```

**Benefits:**
- Faster: ~40-50% speed improvement
- Lower VRAM per GPU: ~20-24GB
- Can handle longer videos

### 8 GPUs

```python
os.environ['WORLD_SIZE'] = '8'
'--nproc_per_node=8'
'--ulysses_size', '8'
```

**Benefits:**
- Fastest: 2-3x speed improvement
- Lowest VRAM: ~12-16GB per GPU
- Production-ready

## 💡 Tips for Best Results

### Animation Mode Tips

1. **Driving Video**:
   - Use videos with clear, exaggerated motions
   - Ensure person is centered and fully visible
   - Avoid rapid camera movements
   - Good examples: Dance videos, exercise demos, acting performances

2. **Character Image**:
   - Use high-quality, well-lit images
   - Character should be in similar pose to video start
   - Full body images work better for full body animation
   - Portrait images work for head/upper body animation

3. **Settings**:
   - Start with default `--refert_num 1`
   - Use `--sample_guide_scale 1.0` (increasing may help with expression)
   - For subtle motions, use `--sample_steps 25-30`

### Replacement Mode Tips

1. **Source Video**:
   - Character should be clearly visible throughout
   - Consistent lighting on character
   - Avoid occlusions or hand-in-front-of-face

2. **New Character**:
   - Similar proportions to original works best
   - Clear, high-quality image
   - Frontal view preferred

3. **Settings**:
   - Always use `--use_relighting_lora`
   - Set `--iterations 3-5` in preprocessing
   - May need `--sample_guide_scale 1.5-2.0` for better face transfer

## 🆚 Comparison: Animate vs I2V vs S2V

| Feature | Animate-14B | I2V-A14B | S2V-14B |
|---------|-------------|----------|---------|
| **Task** | Motion transfer | Image animation | Speech-driven |
| **Model** | Single 14B | MoE 27B | Single 14B |
| **Input** | Video + Image | Image + Text | Image + Audio |
| **Preprocessing** | Required | None | None |
| **Use Case** | Character animation | General motion | Lip-sync videos |
| **Unique** | Motion encoder, Face encoder | MoE switching | Audio encoder |
| **VRAM** | ~30GB/GPU (2GPU) | ~32GB/GPU | ~30GB/GPU |

## 📝 Common Use Cases

### 1. Anime Character Animation
```
Input: Dance video + anime character image
Output: Anime character performing the dance
Mode: Animation
```

### 2. Virtual Avatar Creation
```
Input: Acting performance + avatar design
Output: Avatar mimicking actor's movements
Mode: Animation
```

### 3. Character Swapping in Movies
```
Input: Movie scene + new character design
Output: Scene with replaced character
Mode: Replacement
```

### 4. Style Transfer Animation
```
Input: Real person video + artistic character
Output: Stylized version with same motion
Mode: Replacement or Animation
```

## 🔗 Related Resources

- **Model Page**: https://huggingface.co/Wan-AI/Wan2.2-Animate-14B
- **Project Page**: https://humanaigc.github.io/wan-animate
- **Paper**: https://arxiv.org/abs/2503.20314
- **Preprocessing Guide**: wan/modules/animate/preprocess/UserGuider.md
- **GitHub**: https://github.com/Wan-Video/Wan2.2

## 📄 License

Apache 2.0 - See model card for details

---

**Created for efficient character animation on RunPod multi-GPU instances**
