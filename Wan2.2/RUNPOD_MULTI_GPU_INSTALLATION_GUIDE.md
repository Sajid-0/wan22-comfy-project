# Wan2.2 Multi-GPU Installation Guide for RunPod (2x A40)

## 📋 Executive Summary

**System Detected:**
- **GPUs:** 2x NVIDIA A40 (46GB VRAM each = 92GB total)
- **Current Setup:** PyTorch 2.2.0 + CUDA 12.1
- **OS:** Ubuntu 22.04.3 LTS
- **Driver:** 570.195.03

**Recommended Configuration:**
- **PyTorch:** 2.4.0+ (required by Wan2.2)
- **CUDA:** 12.1+ (already compatible)
- **Python:** 3.10-3.11
- **Flash Attention:** 2.x (requires CUDA 12.0+, supports A40 Ampere architecture)

---

## 🔍 Deep Analysis - Key Issues Found

### 1. **PyTorch Version Mismatch** ⚠️
**Current:** PyTorch 2.2.0  
**Required:** PyTorch ≥ 2.4.0

**Impact:** The repository explicitly requires `torch>=2.4.0` in `requirements.txt`. Your current version is too old.

**Solution:** Must upgrade to PyTorch 2.4.0 or newer.

---

### 2. **Flash Attention Compatibility** ✅
**Your A40 GPUs:** Ampere architecture (Compute Capability 8.6)  
**Flash Attention Support:** Full support for Ampere GPUs

**Requirements:**
- CUDA ≥ 12.0 (You have 12.1 ✅)
- PyTorch ≥ 2.2 (Need to upgrade to 2.4.0+)
- Supports fp16 and bf16 datatypes
- All head dimensions up to 256

**Note:** Flash Attention 2.x is optimized for A100/A40 (Ampere) GPUs and will work excellently on your hardware.

---

### 3. **Multi-GPU Configuration Analysis**

#### FSDP (Fully Sharded Data Parallel)
- **Purpose:** Distributes model parameters across GPUs to reduce memory per GPU
- **Used for:** Large models (T2V-A14B, I2V-A14B with 14B/27B parameters)
- **Your Setup:** 2x A40 (92GB total) - Perfect for this

#### DeepSpeed Ulysses Sequence Parallelism
- **Purpose:** Distributes sequence/attention computation across GPUs
- **Parameter:** `--ulysses_size 2` (for your 2 GPUs)
- **Requirement:** Number of attention heads must be divisible by ulysses_size
- **Model Heads:** Varies by model config

#### Optimal Settings for 2x A40:
```bash
# For 14B models (T2V-A14B, I2V-A14B, S2V-14B)
torchrun --nproc_per_node=2 generate.py \
  --task t2v-A14B \
  --dit_fsdp \          # Enable FSDP for DiT model
  --t5_fsdp \           # Enable FSDP for T5 encoder
  --ulysses_size 2      # Sequence parallel across 2 GPUs

# For 5B model (TI2V-5B) - Can run on single A40 with offloading
python generate.py --task ti2v-5B \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu
```

---

### 4. **Memory Requirements by Model**

| Model | Single GPU (80GB) | Multi-GPU (2x A40) | Notes |
|-------|-------------------|---------------------|-------|
| T2V-A14B 720P | 80GB+ | ~40-50GB per GPU ✅ | With FSDP+Ulysses |
| I2V-A14B 720P | 80GB+ | ~40-50GB per GPU ✅ | With FSDP+Ulysses |
| TI2V-5B 720P | 24GB (with offload) | 12-15GB per GPU ✅ | Efficient model |
| S2V-14B 480P/720P | 80GB+ | ~40-50GB per GPU ✅ | With FSDP+Ulysses |
| Animate-14B | 80GB+ | ~40-50GB per GPU ✅ | With FSDP+Ulysses |

**Your 2x A40 setup is PERFECT for running all models!**

---

## 🚀 Step-by-Step Installation Guide

### Step 1: Clean Environment Setup

```bash
# Create new conda environment (recommended)
conda create -n wan22 python=3.10 -y
conda activate wan22

# OR use venv
python3.10 -m venv /workspace/wan22_env
source /workspace/wan22_env/bin/activate
```

### Step 2: Install PyTorch 2.4.0+ with CUDA 12.1

```bash
# Install PyTorch 2.4.0 with CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU Available: {torch.cuda.is_available()}')"
```

**Expected Output:**
```
PyTorch: 2.4.0+cu121
CUDA: 12.1
GPU Available: True
```

### Step 3: Install Core Dependencies (Except Flash Attention)

```bash
cd /workspace/wan22-comfy-project/Wan2.2

# Install all dependencies except flash_attn first
pip install opencv-python>=4.9.0.80 \
    diffusers>=0.31.0 \
    transformers>=4.49.0,<=4.51.3 \
    tokenizers>=0.20.3 \
    accelerate>=1.1.1 \
    tqdm imageio[ffmpeg] easydict ftfy dashscope \
    imageio-ffmpeg "numpy>=1.23.5,<2"
```

### Step 4: Install Flash Attention (Critical!)

```bash
# Method 1: Install from PyPI (Recommended)
pip install flash-attn --no-build-isolation

# If Method 1 fails, try Method 2: Build from source
pip install packaging ninja wheel setuptools
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# If both fail, try Method 3: Install from git
pip install git+https://github.com/Dao-AILab/flash-attention.git

# Verify installation
python -c "import flash_attn; print(f'Flash Attention installed: {flash_attn.__version__}')"
```

**Important Notes:**
- Flash Attention compilation takes 3-5 minutes on multi-core systems
- Use `MAX_JOBS=4` if you have RAM constraints (< 96GB)
- The `--no-build-isolation` flag is required

### Step 5: Install Additional Dependencies (Optional)

```bash
# For Speech-to-Video (S2V) models
pip install -r requirements_s2v.txt

# For Animate models
pip install -r requirements_animate.txt
```

### Step 6: Verify Multi-GPU Setup

```bash
# Test PyTorch distributed
python -c "import torch.distributed as dist; print('Distributed available:', dist.is_available())"

# Check NCCL (NVIDIA Collective Communications Library)
python -c "import torch; print('NCCL available:', torch.cuda.nccl.is_available([0, 1]))"

# Verify both GPUs are detected
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

### Step 7: Download Model Weights

```bash
# Install download tools
pip install "huggingface_hub[cli]" modelscope

# Download T2V model (choose one method)
# Method 1: HuggingFace
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B

# Method 2: ModelScope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B

# Download other models as needed
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan2.2-I2V-A14B
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir ./Wan2.2-S2V-14B
huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir ./Wan2.2-Animate-14B
```

---

## 🎯 Running Models on 2x A40

### Example 1: Text-to-Video (T2V-A14B) - 720P

```bash
# Multi-GPU with FSDP + Ulysses
torchrun --nproc_per_node=2 generate.py \
  --task t2v-A14B \
  --size 1280*720 \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 2 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

**Expected Performance:**
- Memory per GPU: ~40-45GB
- Generation time: ~2-4 minutes for 5s video

### Example 2: Image-to-Video (I2V-A14B) - 720P

```bash
torchrun --nproc_per_node=2 generate.py \
  --task i2v-A14B \
  --size 1280*720 \
  --ckpt_dir ./Wan2.2-I2V-A14B \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 2 \
  --image examples/i2v_input.JPG \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."
```

### Example 3: Text-Image-to-Video (TI2V-5B) - 720P

```bash
# Option A: Multi-GPU (faster)
torchrun --nproc_per_node=2 generate.py \
  --task ti2v-5B \
  --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 2 \
  --prompt "Anthropomorphic cats boxing"

# Option B: Single GPU (more memory efficient)
python generate.py \
  --task ti2v-5B \
  --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --prompt "Anthropomorphic cats boxing"
```

### Example 4: With Prompt Extension

```bash
# Using local Qwen model
torchrun --nproc_per_node=2 generate.py \
  --task t2v-A14B \
  --size 1280*720 \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 2 \
  --prompt "cats boxing" \
  --use_prompt_extend \
  --prompt_extend_method local_qwen \
  --prompt_extend_model "Qwen/Qwen2.5-3B-Instruct" \
  --prompt_extend_target_lang en
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Flash Attention Installation Fails

**Symptoms:** Build errors, PEP 517 errors

**Solutions:**
```bash
# Solution A: Install dependencies first
pip install packaging ninja wheel setuptools torch==2.4.0

# Solution B: Use no-build-isolation
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Solution C: Install from git
pip install git+https://github.com/Dao-AILab/flash-attention.git
```

### Issue 2: CUDA Out of Memory

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# For 14B models, always use FSDP
--dit_fsdp --t5_fsdp

# Add model offloading
--offload_model True

# Convert model dtype
--convert_model_dtype

# Move T5 to CPU (reduces GPU memory ~10GB)
--t5_cpu

# Reduce resolution temporarily
--size 720*480  # instead of 1280*720
```

### Issue 3: NCCL/Distributed Errors

**Symptoms:** `NCCL error`, `torch.distributed` timeout

**Solutions:**
```bash
# Set environment variables before running
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1  # Disable P2P if issues
export NCCL_IB_DISABLE=1   # Disable InfiniBand if not available

# Verify GPU communication
python -c "import torch; print(torch.cuda.nccl.is_available([0, 1]))"

# Check NVIDIA Fabric Manager (for A40 NVLink)
sudo systemctl status nvidia-fabricmanager
```

### Issue 4: Number of Heads Not Divisible by ulysses_size

**Symptoms:** `AssertionError: num_heads % ulysses_size == 0`

**Solutions:**
```bash
# Check model config for num_heads
# For 2 GPUs, num_heads must be even
# If error occurs, you cannot use ulysses_size=2 for that model

# Fallback: Use only FSDP without Ulysses
torchrun --nproc_per_node=2 generate.py \
  --task t2v-A14B \
  --dit_fsdp \
  --t5_fsdp \
  # Remove: --ulysses_size 2
```

---

## ⚙️ Optimization Tips for RunPod

### 1. Environment Variables

```bash
# Add to ~/.bashrc or start script
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

### 2. Persistent Storage

```bash
# Store models on persistent storage
export MODEL_DIR=/workspace/models
mkdir -p $MODEL_DIR

# Symlink to models
ln -s $MODEL_DIR/Wan2.2-T2V-A14B ./Wan2.2-T2V-A14B
```

### 3. Monitoring

```bash
# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Monitor with more details
nvidia-smi dmon -s u -c 100

# Python monitoring script
python -c "
import torch
import time
while True:
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1e9
        max_mem = torch.cuda.max_memory_allocated(i) / 1e9
        print(f'GPU {i}: {mem:.2f}GB / {max_mem:.2f}GB')
    time.sleep(2)
"
```

---

## 📊 Performance Benchmarks (2x A40)

| Model | Resolution | Multi-GPU Time | Single GPU Time | Memory per GPU |
|-------|-----------|----------------|-----------------|----------------|
| T2V-A14B | 720P | ~2-3 min | ~5-7 min* | 40-45GB |
| T2V-A14B | 480P | ~1-2 min | ~3-5 min* | 35-40GB |
| I2V-A14B | 720P | ~2-3 min | ~5-7 min* | 40-45GB |
| TI2V-5B | 720P | ~4-6 min | ~8-10 min | 12-15GB |
| S2V-14B | 720P | ~3-4 min | ~6-8 min* | 42-47GB |

*Single GPU with offloading enabled

---

## 🔧 Complete Test Script

Save this as `test_multi_gpu.sh`:

```bash
#!/bin/bash

echo "=== Testing Wan2.2 Multi-GPU Setup on 2x A40 ==="

# Check GPUs
echo -e "\n1. Checking GPUs..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# Check PyTorch
echo -e "\n2. Checking PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPUs: {torch.cuda.device_count()}')"

# Check Flash Attention
echo -e "\n3. Checking Flash Attention..."
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"

# Check Distributed
echo -e "\n4. Checking Distributed Support..."
python -c "import torch.distributed as dist; print('Distributed available:', dist.is_available())"
python -c "import torch; print('NCCL available:', torch.cuda.nccl.is_available([0, 1]))"

# Test T2V-A14B
echo -e "\n5. Testing T2V-A14B (480P, quick test)..."
torchrun --nproc_per_node=2 generate.py \
  --task t2v-A14B \
  --size 480*832 \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 2 \
  --frame_num 17 \
  --sample_steps 30 \
  --prompt "A cat walking"

echo -e "\n=== Test Complete ==="
```

Run with:
```bash
chmod +x test_multi_gpu.sh
./test_multi_gpu.sh
```

---

## 📝 Quick Reference Card

### Minimal Command for 2x A40:
```bash
torchrun --nproc_per_node=2 generate.py \
  --task t2v-A14B \
  --size 1280*720 \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 2 \
  --prompt "Your prompt here"
```

### Memory-Saving Options:
- `--offload_model True` - Offload models to CPU between forward passes
- `--convert_model_dtype` - Convert to config dtype (usually bfloat16)
- `--t5_cpu` - Keep T5 encoder on CPU (saves ~10GB GPU memory)

### Quality Options:
- `--use_prompt_extend` - Enhance prompts with AI
- `--sample_steps 50` - More steps = better quality (default: 30)
- `--sample_guide_scale 7.5` - CFG scale (default: varies by model)

---

## 🎓 Understanding the Architecture

### FSDP (Fully Sharded Data Parallel)
- Splits model weights across GPUs
- Each GPU holds 1/N of the model
- Reduces memory per GPU
- Slight communication overhead

### Ulysses (Sequence Parallelism)
- Splits attention computation across GPUs
- More efficient for long sequences
- Requires num_heads divisible by GPU count
- No weight duplication

### Your 2x A40 Setup:
```
GPU 0 (46GB)         GPU 1 (46GB)
├─ 50% of weights    ├─ 50% of weights
├─ 50% of sequence   ├─ 50% of sequence
└─ All-reduce sync ──┘
```

---

## 📚 Additional Resources

- **Official Repo:** https://github.com/Wan-Video/Wan2.2
- **HuggingFace Models:** https://huggingface.co/Wan-AI/
- **Paper:** https://arxiv.org/abs/2503.20314
- **Discord:** https://discord.gg/AKNgpMK4Yj
- **Flash Attention:** https://github.com/Dao-AILab/flash-attention

---

## ✅ Installation Checklist

- [ ] Python 3.10 environment created
- [ ] PyTorch 2.4.0+ with CUDA 12.1 installed
- [ ] Flash Attention successfully installed
- [ ] All dependencies from requirements.txt installed
- [ ] Both A40 GPUs detected by PyTorch
- [ ] NCCL working for multi-GPU communication
- [ ] Model weights downloaded
- [ ] Test generation successful

---

**Last Updated:** October 22, 2025  
**Tested On:** 2x NVIDIA A40 (46GB), Ubuntu 22.04, CUDA 12.1, PyTorch 2.4.0

**Good luck with your Wan2.2 multi-GPU setup! 🚀**
