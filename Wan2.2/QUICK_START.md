# 🚀 Wan2.2 RunPod Multi-GPU Quick Start

## Your System
- **GPUs:** 2x NVIDIA A40 (46GB each = 92GB total VRAM) ✅
- **Current PyTorch:** 2.2.0 ❌ (Need 2.4.0+)
- **Current CUDA:** 12.1 ✅
- **Driver:** 570.195.03 ✅
- **OS:** Ubuntu 22.04 ✅

## Critical Issues Found

### 1. PyTorch Version Too Old ⚠️
**Problem:** You have PyTorch 2.2.0, but Wan2.2 requires ≥ 2.4.0  
**Solution:** Must upgrade immediately

### 2. Flash Attention Required
**Status:** Likely not installed  
**Required For:** Efficient attention computation on Ampere GPUs (A40)

---

## 🎯 Quick Installation (3 Options)

### Option 1: Automated Script (Recommended)
```bash
cd /workspace/wan22-comfy-project/Wan2.2
./setup_runpod_multi_gpu.sh
```
This handles everything automatically!

### Option 2: Manual Steps
```bash
# 1. Create environment
python3 -m venv venv_wan22
source venv_wan22/bin/activate

# 2. Install PyTorch 2.4.0 + CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install opencv-python diffusers "transformers>=4.49.0,<=4.51.3" \
    tokenizers accelerate tqdm "imageio[ffmpeg]" easydict ftfy \
    dashscope imageio-ffmpeg "numpy>=1.23.5,<2"

# 4. Install Flash Attention (takes 3-5 min)
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# 5. Download model
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

### Option 3: If You're Upgrading Existing Installation
```bash
# Upgrade PyTorch
pip install --upgrade torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention if missing
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}'); import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"
```

---

## 🎬 Run Your First Generation

### Text-to-Video (720P on 2x A40)
```bash
torchrun --nproc_per_node=2 generate.py \
  --task t2v-A14B \
  --size 1280*720 \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 2 \
  --prompt "Two anthropomorphic cats boxing on a stage"
```

**Expected:**
- Time: ~2-3 minutes
- Memory per GPU: ~40-45GB
- Output: 5-second 720P video

---

## 📊 What Works on 2x A40

| Model | Resolution | Status | Notes |
|-------|-----------|--------|-------|
| T2V-A14B | 720P | ✅ Perfect | Use --dit_fsdp --t5_fsdp --ulysses_size 2 |
| I2V-A14B | 720P | ✅ Perfect | Use --dit_fsdp --t5_fsdp --ulysses_size 2 |
| TI2V-5B | 720P | ✅ Perfect | Can even run on single A40 |
| S2V-14B | 720P | ✅ Perfect | Use --dit_fsdp --t5_fsdp --ulysses_size 2 |
| Animate-14B | 720P | ✅ Perfect | Use --dit_fsdp --t5_fsdp --ulysses_size 2 |

---

## 🔧 Essential Commands

### Activate Environment
```bash
source venv_wan22/bin/activate
```

### Check Setup
```bash
# GPUs
nvidia-smi

# PyTorch
python -c "import torch; print(f'Version: {torch.__version__}'); print(f'GPUs: {torch.cuda.device_count()}')"

# Flash Attention
python -c "import flash_attn; print('Flash Attention OK')"
```

### Monitor During Generation
```bash
# In another terminal
watch -n 1 nvidia-smi
```

---

## 🐛 Common Issues & Quick Fixes

### "CUDA out of memory"
**Fix:** Already using FSDP? Try:
```bash
--t5_cpu  # Moves T5 to CPU (saves ~10GB)
--offload_model True  # Offloads between steps
```

### "ImportError: No module named flash_attn"
**Fix:**
```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### "AssertionError: num_heads % ulysses_size"
**Fix:** Remove `--ulysses_size 2`, keep only FSDP:
```bash
--dit_fsdp --t5_fsdp  # Without ulysses_size
```

### "NCCL error"
**Fix:**
```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# Then run your command
```

---

## 💡 Performance Tips

### Faster Generation
- Use 480P instead of 720P (2x faster)
- Reduce `--sample_steps` from 50 to 30
- Use `--frame_num 49` instead of 81 (shorter video)

### Better Quality
- Use `--use_prompt_extend` (AI-enhanced prompts)
- Increase `--sample_steps` to 50
- Use `--sample_guide_scale 7.5`

### Maximum Memory Efficiency
```bash
--offload_model True \
--convert_model_dtype \
--t5_cpu
```

---

## 📦 Model Downloads

```bash
# Text-to-Video (14B, ~50GB)
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B

# Image-to-Video (14B, ~50GB)
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan2.2-I2V-A14B

# Text-Image-to-Video (5B, ~20GB)
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B

# Speech-to-Video (14B, ~50GB)
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir ./Wan2.2-S2V-14B

# Animation (14B, ~50GB)
huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir ./Wan2.2-Animate-14B
```

---

## 🔍 System Requirements Met?

- [x] **Linux:** Ubuntu 22.04 ✅
- [x] **Python:** 3.10+ ✅
- [ ] **PyTorch:** 2.4.0+ ❌ **NEED TO UPGRADE**
- [x] **CUDA:** 12.1+ ✅
- [x] **GPUs:** 2x A40 (92GB total) ✅
- [ ] **Flash Attention:** ❓ **NEED TO INSTALL**

---

## 🎓 Understanding Your Setup

### What is FSDP?
Splits the model across 2 GPUs:
- GPU 0: 50% of weights
- GPU 1: 50% of weights
- **Benefit:** Can run models too large for 1 GPU

### What is Ulysses?
Splits attention computation across 2 GPUs:
- GPU 0: Processes half the sequence
- GPU 1: Processes other half
- **Benefit:** Faster processing, no weight duplication

### Your 2x A40 with FSDP + Ulysses:
```
┌─────────────┐  ┌─────────────┐
│   GPU 0     │  │   GPU 1     │
│  (46GB)     │  │  (46GB)     │
├─────────────┤  ├─────────────┤
│ 50% weights │  │ 50% weights │
│ 50% sequence│  │ 50% sequence│
└──────┬──────┘  └──────┬──────┘
       └────────┬────────┘
           Sync via NCCL
```

---

## 📚 More Information

- **Full Guide:** `RUNPOD_MULTI_GPU_INSTALLATION_GUIDE.md`
- **Official Repo:** https://github.com/Wan-Video/Wan2.2
- **Discord:** https://discord.gg/AKNgpMK4Yj

---

## ✅ Quick Validation

After installation, run:
```bash
# Test multi-GPU setup
python -c "import torch; assert torch.cuda.device_count() == 2, 'Need 2 GPUs'; print('✓ 2 GPUs detected')"
python -c "import torch; assert torch.__version__ >= '2.4.0', 'Need PyTorch 2.4+'; print('✓ PyTorch 2.4+')"
python -c "import flash_attn; print('✓ Flash Attention OK')"
python -c "import torch; assert torch.cuda.nccl.is_available([0,1]), 'NCCL not available'; print('✓ NCCL OK')"
```

All checks passed? You're ready to generate! 🚀

---

**TL;DR:**
1. You MUST upgrade PyTorch from 2.2.0 to 2.4.0
2. You MUST install Flash Attention
3. Your 2x A40 setup is PERFECT for Wan2.2
4. Use `--dit_fsdp --t5_fsdp --ulysses_size 2` for best results

Run `./setup_runpod_multi_gpu.sh` to automate everything! ✨
