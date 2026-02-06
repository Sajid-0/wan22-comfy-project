# Wan2.2 Technical Analysis & Recommendations

## 🎯 Executive Summary for RunPod 2x A40 Setup

### Current State Analysis
```
✅ Hardware: 2x NVIDIA A40 (46GB each) - EXCELLENT
✅ CUDA Driver: 570.195.03 - Compatible
✅ CUDA Version: 12.1 - Compatible
✅ OS: Ubuntu 22.04.3 LTS - Compatible
❌ PyTorch: 2.2.0 - TOO OLD (Need 2.4.0+)
❓ Flash Attention: Not detected - MUST INSTALL
```

### Verdict: **Near-Perfect Hardware, Software Needs Upgrade**

---

## 📊 Detailed Dependency Analysis

### PyTorch Version Compatibility Matrix

| Component | Minimum Required | Your Current | Status | Action |
|-----------|-----------------|--------------|--------|--------|
| PyTorch | 2.4.0 | 2.2.0 | ❌ | **MUST UPGRADE** |
| torchvision | 0.19.0 | 0.17.0* | ❌ | **MUST UPGRADE** |
| CUDA | 12.0+ | 12.1 | ✅ | No action needed |
| Python | 3.10-3.11 | Likely OK | ✅ | Verify version |
| Flash Attention | 2.x | Not installed | ❌ | **MUST INSTALL** |

*Estimated based on PyTorch 2.2.0

### Why PyTorch 2.4.0+ is Critical

1. **FSDP Improvements:** Better memory management for 14B models
2. **torch.compile Support:** Used internally by Wan2.2
3. **CUDA 12.1 Optimizations:** Better kernel performance
4. **Bug Fixes:** Critical fixes for distributed training
5. **API Changes:** Wan2.2 uses newer APIs not in 2.2.0

---

## 🔬 Deep Dive: Flash Attention Requirements

### What is Flash Attention?
Flash Attention is a fast and memory-efficient attention algorithm that's crucial for Wan2.2:

**Benefits:**
- 2-4x faster attention computation
- 5-20x less memory usage
- Enables longer sequence lengths
- Required for optimal performance on Ampere GPUs (A40)

### Your A40 GPU Compatibility

| Feature | A40 Support | Notes |
|---------|-------------|-------|
| Architecture | Ampere (8.6) | ✅ Fully supported |
| Tensor Cores | Yes | ✅ Fast FP16/BF16 |
| Memory | 46GB | ✅ Excellent for 14B models |
| CUDA 12.1 | Compatible | ✅ Flash Attention 2.x works |
| NVLink | Yes (on some configs) | ✅ Faster GPU-to-GPU |
| FlashAttention-2 | Full support | ✅ Recommended |
| FlashAttention-3 | No (H100 only) | ⚠️ Not needed |

### Installation Methods Ranked

**Method 1: PyPI (Recommended for RunPod)**
```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```
- Pros: Pre-compiled wheels, fastest
- Cons: May not have latest version
- Build time: 3-5 minutes

**Method 2: From Source**
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
MAX_JOBS=4 python setup.py install
```
- Pros: Latest version, optimized for your GPU
- Cons: Longer build time
- Build time: 5-10 minutes

**Method 3: From Git**
```bash
pip install git+https://github.com/Dao-AILab/flash-attention.git
```
- Pros: Easy, gets latest
- Cons: Slower than pre-built
- Build time: 5-10 minutes

---

## 🎮 Multi-GPU Strategy for 2x A40

### Strategy 1: FSDP Only (More Memory, Slower)
```bash
torchrun --nproc_per_node=2 generate.py \
  --task t2v-A14B \
  --dit_fsdp \
  --t5_fsdp
```

**Memory per GPU:** ~35-40GB  
**Speed:** Baseline  
**Best for:** When ulysses causes issues  

### Strategy 2: FSDP + Ulysses (Balanced) ⭐ RECOMMENDED
```bash
torchrun --nproc_per_node=2 generate.py \
  --task t2v-A14B \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 2
```

**Memory per GPU:** ~40-45GB  
**Speed:** 1.5-2x faster  
**Best for:** Most models, optimal balance  

### Strategy 3: Maximum Memory Saving
```bash
torchrun --nproc_per_node=2 generate.py \
  --task t2v-A14B \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 2 \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu
```

**Memory per GPU:** ~25-30GB  
**Speed:** 0.5-0.7x (slower due to offloading)  
**Best for:** Running multiple models simultaneously  

---

## 📈 Performance Estimates for Your Setup

### Text-to-Video (T2V-A14B)

| Resolution | Setup | Time (est.) | Memory/GPU | Quality |
|-----------|-------|-------------|------------|---------|
| 480P | Single A40 | 5-7 min | 70GB+ (OOM risk) | Same |
| 480P | 2x A40 FSDP | 3-4 min | 35-40GB | Same |
| 480P | 2x A40 FSDP+Ulysses | 2-3 min | 38-42GB | Same |
| 720P | Single A40 | N/A | OOM | N/A |
| 720P | 2x A40 FSDP | 5-6 min | 40-45GB | Same |
| 720P | 2x A40 FSDP+Ulysses | 3-4 min | 42-46GB | Same |

### Image-to-Video (I2V-A14B)

Similar to T2V-A14B, with additional image encoder overhead (+~2GB memory).

### Text-Image-to-Video (TI2V-5B)

| Resolution | Setup | Time (est.) | Memory/GPU | Notes |
|-----------|-------|-------------|------------|-------|
| 720P | Single A40 | 8-10 min | 24GB | With offloading |
| 720P | 2x A40 | 4-6 min | 12-15GB | Overkill but fast |

**Note:** TI2V-5B is so efficient it can run on 1 GPU comfortably!

---

## 🔍 Common Issues Analysis

### Issue 1: Import Errors After Installation

**Symptoms:**
```python
ImportError: cannot import name 'flash_attn_func' from 'flash_attn'
```

**Root Causes:**
1. Flash Attention not properly installed
2. Multiple Python environments mixed
3. CUDA version mismatch

**Solutions:**
```bash
# Verify installation
python -c "from flash_attn import flash_attn_func; print('OK')"

# Reinstall if needed
pip uninstall flash-attn
MAX_JOBS=4 pip install flash-attn --no-build-isolation --force-reinstall
```

### Issue 2: CUDA OOM Despite Having 92GB Total

**Why This Happens:**
- Each GPU has 46GB, but one model needs 50GB+
- FSDP splits weights, but activations still need space
- Temporary buffers during forward/backward

**Solutions:**
1. **Always use FSDP for 14B models** - splits weights across GPUs
2. **Add Ulysses** - splits activations too
3. **Enable offloading** - moves unused parts to CPU
4. **Reduce batch size** - if using batch_size > 1

### Issue 3: NCCL Initialization Hangs

**Symptoms:**
```
Initializing process group...
[hangs indefinitely]
```

**Root Causes:**
1. Network issues between GPUs
2. NCCL version incompatible
3. Missing NVIDIA Fabric Manager (for NVLink)

**Solutions:**
```bash
# Check GPU communication
python -c "import torch; torch.cuda.init(); print([torch.cuda.get_device_properties(i) for i in range(2)])"

# Set environment variables
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL  # If you have NVLink
export NCCL_SOCKET_IFNAME=eth0  # Or your network interface

# Disable P2P if problematic
export NCCL_P2P_DISABLE=1
```

### Issue 4: "num_heads not divisible by ulysses_size"

**Why:**
- Model has 20 attention heads
- You're using `--ulysses_size 2`
- But 20 is not divisible by 2... wait, it is!

Actually, this usually means:
- Model config doesn't match expected
- Or model uses different head counts in different layers

**Solution:**
```bash
# Just remove ulysses_size, use FSDP only
torchrun --nproc_per_node=2 generate.py \
  --task t2v-A14B \
  --dit_fsdp \
  --t5_fsdp
  # No --ulysses_size
```

---

## 🎓 Understanding Distributed Strategies

### FSDP (Fully Sharded Data Parallel)

```
Before FSDP:
GPU 0: [Full Model 14B params] = 28GB
GPU 1: [Full Model 14B params] = 28GB
Total: 56GB wasted duplication!

After FSDP:
GPU 0: [First 7B params] = 14GB
GPU 1: [Last 7B params] = 14GB
Total: 28GB, perfect!

During forward:
GPU 0 needs GPU 1's params? → All-gather
GPU 1 needs GPU 0's params? → All-gather
Backward: Same, but with gradients
```

### Ulysses (Sequence Parallel)

```
Without Ulysses:
GPU 0: Process full sequence [0:100]
GPU 1: Process full sequence [0:100]
= Duplicate computation!

With Ulysses:
GPU 0: Process sequence [0:50]
GPU 1: Process sequence [51:100]
Then all-to-all to get full attention

Result: 2x faster attention!
```

### Combined FSDP + Ulysses (Your Setup)

```
┌─────────────────┬─────────────────┐
│     GPU 0       │     GPU 1       │
├─────────────────┼─────────────────┤
│ Params: 0-7B    │ Params: 7-14B   │  ← FSDP splits weights
│ Sequence: 0-50% │ Sequence: 50-100%│  ← Ulysses splits sequence
│ Memory: ~42GB   │ Memory: ~42GB   │
└─────────────────┴─────────────────┘
         ↕ All-gather/All-to-all ↕
         Communication via NCCL
```

---

## 🚀 Recommended Installation Sequence

### Phase 1: Environment Preparation (5 minutes)
```bash
# 1. Clean environment
conda create -n wan22 python=3.10 -y
conda activate wan22

# 2. Update system packages
pip install --upgrade pip setuptools wheel
```

### Phase 2: Core Dependencies (10 minutes)
```bash
# 3. Install PyTorch 2.4.0
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 4. Verify
python -c "import torch; assert torch.__version__ >= '2.4.0'; print('✓ PyTorch OK')"
```

### Phase 3: Flash Attention (5 minutes)
```bash
# 5. Install build tools
pip install packaging ninja

# 6. Install Flash Attention
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# 7. Verify
python -c "import flash_attn; print('✓ Flash Attention OK')"
```

### Phase 4: Wan2.2 Dependencies (5 minutes)
```bash
# 8. Install requirements
cd /workspace/wan22-comfy-project/Wan2.2
pip install -r requirements.txt

# 9. Verify
python -c "import diffusers, transformers, accelerate; print('✓ All deps OK')"
```

### Phase 5: Model Download (30-60 minutes)
```bash
# 10. Install download tools
pip install "huggingface_hub[cli]"

# 11. Download model (choose one)
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

### Phase 6: Testing (5 minutes)
```bash
# 12. Run quick test
torchrun --nproc_per_node=2 generate.py \
  --task t2v-A14B \
  --size 480*832 \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 2 \
  --frame_num 17 \
  --sample_steps 20 \
  --prompt "test"
```

**Total Time: ~60-90 minutes** (mostly model download)

---

## 📊 Cost-Benefit Analysis

### Option A: Upgrade Current Environment
**Pros:**
- Keep existing setup
- Faster to get started
- No data migration

**Cons:**
- Risk of conflicts
- May need to reinstall if issues
- Harder to rollback

**Recommendation:** ⚠️ Risky, not recommended

### Option B: Fresh Virtual Environment
**Pros:**
- Clean slate
- Easy to debug
- Can keep old environment
- Professional approach

**Cons:**
- Takes 15 minutes to setup
- Need to reinstall packages

**Recommendation:** ✅ **HIGHLY RECOMMENDED**

### Option C: Use Docker Container
**Pros:**
- Isolated environment
- Reproducible
- Easy to share

**Cons:**
- More complex setup
- Need Docker knowledge
- Slightly more overhead

**Recommendation:** ✅ Good for production

---

## 🎯 Final Recommendations

### For Immediate Testing (Next 10 minutes)
```bash
# Quick upgrade in current environment
pip install --upgrade torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### For Production Use (Next 1 hour)
```bash
# Run the automated script
cd /workspace/wan22-comfy-project/Wan2.2
./setup_runpod_multi_gpu.sh
```

### For Development (Next 2 hours)
```bash
# Create proper conda environment
conda create -n wan22 python=3.10 -y
conda activate wan22
# Follow manual installation in QUICK_START.md
```

---

## 📚 Additional Resources

### Official Documentation
- Wan2.2 GitHub: https://github.com/Wan-Video/Wan2.2
- Wan2.2 Paper: https://arxiv.org/abs/2503.20314
- Flash Attention: https://github.com/Dao-AILab/flash-attention
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
- DeepSpeed Ulysses: https://arxiv.org/abs/2309.14509

### Community
- Discord: https://discord.gg/AKNgpMK4Yj
- HuggingFace: https://huggingface.co/Wan-AI/
- ModelScope: https://modelscope.cn/organization/Wan-AI

### Troubleshooting
- GitHub Issues: https://github.com/Wan-Video/Wan2.2/issues
- CUDA Compatibility: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- A40 Specs: https://www.nvidia.com/en-us/data-center/a40/

---

## 🎬 Conclusion

Your **2x NVIDIA A40** setup is **EXCELLENT** for Wan2.2! You just need to:

1. ✅ **Upgrade PyTorch to 2.4.0+** (Critical)
2. ✅ **Install Flash Attention** (Critical)
3. ✅ **Use FSDP + Ulysses** (Recommended)

Once done, you'll have one of the best open-source video generation setups available! 🚀

**Estimated Setup Time:** 1-2 hours  
**Estimated First Generation:** 3-4 minutes  
**Sweet Spot:** 720P video generation in ~3 minutes

Good luck! 💪
