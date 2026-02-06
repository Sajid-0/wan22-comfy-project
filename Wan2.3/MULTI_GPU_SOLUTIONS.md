# Multi-GPU Solutions for Wan2.2 S2V Model

## 🎯 **Problem Summary**
The Wan2.2 S2V model was experiencing hanging issues during multi-GPU distributed training, specifically during FSDP initialization with `sync_module_states=True`.

## 🔍 **Root Cause Analysis**
1. **FSDP Hanging**: The `sync_module_states=True` parameter in FSDP causes collective communication that hangs during model initialization
2. **Memory Imbalance**: Model loading was not properly distributed, causing GPU 0 to be overloaded
3. **Flash Attention Compatibility**: Version mismatch between PyTorch 2.9.0 and flash-attn
4. **Missing Dependencies**: `peft` library was missing

## ✅ **Solutions Implemented**

### **Solution 1: Pure Ulysses Sequence Parallelism** ⭐ (RECOMMENDED)
- **File**: `run_s2v_multi_gpu.py`
- **Configuration**: `--ulysses_size 2` without FSDP
- **Advantages**: 
  - No hanging issues
  - Proper sequence distribution across GPUs
  - Native to the Wan2.2 architecture
  - More stable and predictable

### **Solution 2: FSDP with Fixed Configuration** 
- **File**: `run_s2v_multi_gpu_fsdp_fixed.py`
- **Fix**: Patches `sync_module_states=False` in FSDP configuration
- **Advantages**:
  - Enables FSDP for large model support
  - Fixes the hanging issue
  - Better memory distribution for very large models

### **Solution 3: Hybrid Approach**
- **Configuration**: `--ulysses_size 2 --dit_fsdp --t5_cpu`
- **Strategy**: Uses Ulysses + FSDP for DiT, keeps T5 on CPU
- **Use Case**: When you need both sequence parallelism and parameter sharding

## 🛠️ **Environment Fixes Applied**

### **1. PyTorch Version Downgrade**
```bash
# From PyTorch 2.9.0 → 2.8.0 for better flash-attn compatibility
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### **2. Flash Attention Rebuild**
```bash
# Rebuilt flash-attn from source for PyTorch 2.8.0
pip uninstall flash-attn -y
pip install flash-attn --no-build-isolation
```

### **3. Missing Dependencies**
```bash
pip install peft  # Required for LoRA/adapter functionality
```

### **4. Memory Optimization Environment Variables**
```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
```

## 📊 **Performance Comparison**

| Method | GPU Balance | Memory Usage | Initialization Speed | Stability |
|--------|-------------|--------------|---------------------|-----------|
| **Pure Ulysses** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **FSDP Fixed** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Hybrid** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

## 🚀 **Usage Instructions**

### **Option 1: Pure Ulysses (Recommended)**
```bash
cd /workspace/wan22-comfy-project/Wan2.2
python ./run_s2v_multi_gpu.py
```

### **Option 2: FSDP Fixed**
```bash
cd /workspace/wan22-comfy-project/Wan2.2  
python ./run_s2v_multi_gpu_fsdp_fixed.py
```

### **Option 3: Manual Configuration**
```bash
# Use torch.distributed.run directly
python -m torch.distributed.run \
  --nproc_per_node=2 \
  --nnodes=1 \
  generate.py \
  --task s2v-14B \
  --ulysses_size 2 \
  --size 480*832 \
  --ckpt_dir /home/caches/Wan2.2-S2V-14B \
  [... other parameters ...]
```

## 🔧 **Technical Details**

### **Why sync_module_states=True Causes Hanging**
From PyTorch FSDP documentation:
> "If you're using the sync_module_states=True flag, you need to ensure that the module is on a GPU or use the device_id argument to specify a CUDA device that FSDP will move the module to in the FSDP constructor. This is necessary because sync_module_states=True requires GPU communication."

The issue occurs when:
1. Multiple processes try to initialize FSDP simultaneously
2. `sync_module_states=True` triggers collective communication during init
3. This communication hangs if processes are not properly synchronized
4. The hanging manifests as processes getting stuck after NCCL initialization

### **Ulysses Sequence Parallelism Advantages**
- **Native Support**: Built into the Wan2.2 architecture specifically for video generation
- **Sequence Splitting**: Divides temporal dimensions across GPUs (perfect for video)
- **Memory Efficient**: Less overhead than FSDP parameter sharding
- **No Collective Init**: Avoids the sync_module_states hanging issue
- **Better Throughput**: Optimized for the transformer attention patterns in video models

## 🐛 **Debugging Tools**

### **Check GPU Usage**
```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

### **Monitor Process Status**
```bash
ps aux | grep "generate.py" | grep -v grep
```

### **NCCL Debugging**
```bash
export NCCL_DEBUG=INFO  # For verbose NCCL logs
export NCCL_DEBUG=WARN  # For production use
```

## 📝 **Lessons Learned**

1. **Version Compatibility**: PyTorch 2.9.0 had compatibility issues with flash-attn builds
2. **FSDP Pitfalls**: `sync_module_states=True` is dangerous in multi-process environments
3. **Model Architecture**: Ulysses sequence parallelism is better suited for video models than generic FSDP
4. **Environment Variables**: Proper NCCL configuration is critical for container environments
5. **Memory Management**: PyTorch CUDA memory allocator settings significantly impact large model loading

## 🎉 **Success Metrics**
- ✅ Both GPUs properly utilized (balanced memory usage)
- ✅ No hanging during initialization
- ✅ Successful video generation completion
- ✅ Proper multi-GPU acceleration achieved
- ✅ Flash attention working correctly
- ✅ All dependencies resolved

## 📚 **References**
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [Wan2.2 GitHub Repository](https://github.com/Wan-Video/Wan2.2)
- [Flash Attention GitHub](https://github.com/Dao-AILab/flash-attention)
- [NCCL Troubleshooting Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)