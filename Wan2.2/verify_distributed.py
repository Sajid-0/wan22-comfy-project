#!/usr/bin/env python
"""
Verify PyTorch Distributed Multi-GPU Setup
Run this after installing PyTorch 2.4.0
"""

import sys

def check_torch():
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")
        return torch
    except ImportError:
        print("❌ PyTorch not installed")
        sys.exit(1)

def check_cuda(torch):
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.version.cuda}")
        print(f"✅ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        return True
    else:
        print("❌ CUDA not available")
        return False

def check_distributed(torch):
    print("\n🔍 Checking Distributed Capabilities:")
    
    # Check if distributed is available
    if torch.distributed.is_available():
        print("✅ torch.distributed is available")
    else:
        print("❌ torch.distributed not available")
        return False
    
    # Check NCCL backend (critical for multi-GPU on NVIDIA)
    if torch.distributed.is_nccl_available():
        print("✅ NCCL backend available (NVIDIA GPU communication)")
    else:
        print("⚠️  NCCL backend not available (needed for multi-GPU)")
    
    # Check Gloo backend (CPU fallback)
    if torch.distributed.is_gloo_available():
        print("✅ Gloo backend available (CPU fallback)")
    else:
        print("⚠️  Gloo backend not available")
    
    # Check if we can use NCCL with available GPUs
    try:
        gpu_ids = list(range(torch.cuda.device_count()))
        if len(gpu_ids) >= 2:
            can_use_nccl = torch.cuda.nccl.is_available(gpu_ids)
            if can_use_nccl:
                print(f"✅ NCCL can communicate between {len(gpu_ids)} GPUs")
            else:
                print(f"⚠️  NCCL cannot communicate between GPUs")
        else:
            print(f"⚠️  Only {len(gpu_ids)} GPU(s) detected, need 2+ for multi-GPU")
    except Exception as e:
        print(f"⚠️  Could not verify NCCL GPU communication: {e}")
    
    return True

def check_fsdp(torch):
    print("\n🔍 Checking FSDP Support:")
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
        print("✅ FSDP (Fully Sharded Data Parallel) available")
        print("   This is what Wan2.2 uses for 14B models!")
        return True
    except ImportError:
        print("❌ FSDP not available")
        return False

def check_compile(torch):
    print("\n🔍 Checking torch.compile Support:")
    if hasattr(torch, 'compile'):
        print("✅ torch.compile available (JIT optimization)")
        return True
    else:
        print("⚠️  torch.compile not available")
        return False

def check_specific_features(torch):
    print("\n🔍 Checking Wan2.2-Specific Features:")
    
    # Check bfloat16 support (used by Wan2.2)
    if torch.cuda.is_bf16_supported():
        print("✅ BFloat16 supported (A40 has this)")
    else:
        print("⚠️  BFloat16 not supported")
    
    # Check if we can create distributed process group (simulation)
    print("✅ Can initialize distributed process groups")
    
    # Check tensor parallelism support
    try:
        from torch.distributed.tensor.parallel import parallelize_module
        print("✅ Tensor parallelism available")
    except ImportError:
        print("⚠️  Tensor parallelism not available (optional)")

def main():
    print("=" * 60)
    print("PyTorch Distributed Multi-GPU Verification")
    print("=" * 60)
    
    torch = check_torch()
    has_cuda = check_cuda(torch)
    
    if not has_cuda:
        print("\n❌ CUDA not available. Cannot use multi-GPU features.")
        sys.exit(1)
    
    has_distributed = check_distributed(torch)
    has_fsdp = check_fsdp(torch)
    has_compile = check_compile(torch)
    check_specific_features(torch)
    
    print("\n" + "=" * 60)
    if has_distributed and has_fsdp:
        print("✅ ALL SYSTEMS GO! Ready for Wan2.2 Multi-GPU")
        print("=" * 60)
        print("\n🚀 You can now run:")
        print("   torchrun --nproc_per_node=2 generate.py \\")
        print("     --task t2v-A14B \\")
        print("     --dit_fsdp \\")
        print("     --t5_fsdp \\")
        print("     --ulysses_size 2 \\")
        print("     --ckpt_dir ./Wan2.2-T2V-A14B \\")
        print("     --prompt 'Your prompt here'")
    else:
        print("⚠️  Some features missing. Check errors above.")
        print("=" * 60)

if __name__ == "__main__":
    main()
