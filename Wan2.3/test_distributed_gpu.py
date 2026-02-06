#!/usr/bin/env python3
"""
Test Distributed GPU Setup
Simple test to verify multi-GPU PyTorch distributed training is working correctly
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime

def setup_distributed(rank, world_size, master_port="12356"):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the GPU for this process
    torch.cuda.set_device(rank)
    
    print(f"[Rank {rank}] ✅ Distributed setup complete")
    print(f"[Rank {rank}] GPU: {torch.cuda.get_device_name(rank)}")
    print(f"[Rank {rank}] Memory: {torch.cuda.get_device_properties(rank).total_memory / 1e9:.1f} GB")

def test_gpu_communication(rank, world_size):
    """Test basic GPU tensor operations and communication"""
    
    # Create a tensor on this GPU
    device = torch.device(f"cuda:{rank}")
    tensor = torch.ones(4, 4, device=device) * rank
    
    print(f"[Rank {rank}] Created tensor on GPU {rank}: shape={tensor.shape}")
    print(f"[Rank {rank}] Tensor values: {tensor[0, :2]}")
    
    # Test all-reduce operation
    print(f"[Rank {rank}] Testing all-reduce operation...")
    dist.all_reduce(tensor)
    
    expected_sum = sum(range(world_size))
    print(f"[Rank {rank}] After all-reduce: {tensor[0, :2]} (expected: {expected_sum})")
    
    # Verify the result
    if torch.allclose(tensor, torch.ones_like(tensor) * expected_sum):
        print(f"[Rank {rank}] ✅ All-reduce test PASSED")
    else:
        print(f"[Rank {rank}] ❌ All-reduce test FAILED")
    
    # Test memory allocation
    try:
        large_tensor = torch.randn(1000, 1000, device=device)
        print(f"[Rank {rank}] ✅ Large tensor allocation test PASSED")
        del large_tensor
    except Exception as e:
        print(f"[Rank {rank}] ❌ Large tensor allocation FAILED: {e}")

def test_model_loading(rank, world_size):
    """Test loading a simple model with FSDP"""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
        
        # Simple test model
        model = torch.nn.Sequential(
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 100)
        )
        
        # Wrap with FSDP
        fsdp_model = FSDP(
            model,
            cpu_offload=CPUOffload(offload_params=False),
            device_id=rank,
        )
        
        print(f"[Rank {rank}] ✅ FSDP model creation PASSED")
        
        # Test forward pass
        input_tensor = torch.randn(32, 1000, device=f"cuda:{rank}")
        output = fsdp_model(input_tensor)
        print(f"[Rank {rank}] ✅ FSDP forward pass PASSED, output shape: {output.shape}")
        
    except Exception as e:
        print(f"[Rank {rank}] ❌ FSDP test FAILED: {e}")
        import traceback
        traceback.print_exc()

def run_test(rank, world_size):
    """Main test function for each process"""
    print(f"[Rank {rank}] Starting distributed GPU test at {datetime.now()}")
    
    try:
        # Setup distributed
        setup_distributed(rank, world_size)
        
        # Test GPU communication
        test_gpu_communication(rank, world_size)
        
        # Test model loading
        test_model_loading(rank, world_size)
        
        print(f"[Rank {rank}] 🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"[Rank {rank}] ❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if dist.is_initialized():
            dist.destroy_process_group()

def main():
    """Main function"""
    print("🚀 Starting Distributed GPU Test")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("❌ Need at least 2 GPUs for distributed test")
        return
    
    print(f"Testing with {world_size} GPUs")
    
    # Set environment variables
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    os.environ['NCCL_DEBUG'] = 'INFO'
    
    # Use spawn method for multiprocessing
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
    
    print("🎉 Distributed GPU test completed!")

if __name__ == "__main__":
    main()