#!/usr/bin/env python3
"""
Simple Distributed Test Script using torch.distributed.run
Tests basic multi-GPU functionality without complex operations
"""

import os
import torch
import torch.distributed as dist
from datetime import datetime

def main():
    """Main function for distributed test"""
    
    # Get distributed info
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"[Rank {rank}] Starting simple test at {datetime.now()}")
    print(f"[Rank {rank}] Local rank: {local_rank}, World size: {world_size}")
    
    # Initialize distributed
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=torch.device(f"cuda:{local_rank}"))
        print(f"[Rank {rank}] ✅ Process group initialized with device_id")
    except Exception as e:
        print(f"[Rank {rank}] ❌ Failed to initialize process group: {e}")
        return
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"[Rank {rank}] Using device: {device}")
    print(f"[Rank {rank}] GPU: {torch.cuda.get_device_name(local_rank)}")
    print(f"[Rank {rank}] Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
    
    # Test simple tensor creation
    try:
        tensor = torch.ones(10, device=device) * rank
        print(f"[Rank {rank}] ✅ Tensor created: {tensor[:3]}")
    except Exception as e:
        print(f"[Rank {rank}] ❌ Tensor creation failed: {e}")
        return
    
    # Test memory allocation
    try:
        large_tensor = torch.randn(1000, 1000, device=device)
        print(f"[Rank {rank}] ✅ Large tensor allocation successful")
        del large_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[Rank {rank}] ❌ Large tensor allocation failed: {e}")
    
    # Test barrier synchronization
    try:
        print(f"[Rank {rank}] Testing barrier...")
        dist.barrier()
        print(f"[Rank {rank}] ✅ Barrier passed")
    except Exception as e:
        print(f"[Rank {rank}] ❌ Barrier failed: {e}")
    
    # Test simple all-reduce (with timeout)
    try:
        print(f"[Rank {rank}] Testing all-reduce...")
        test_tensor = torch.ones(4, device=device) * rank
        print(f"[Rank {rank}] Before all-reduce: {test_tensor}")
        
        # Use a work object for timeout control
        work = dist.all_reduce(test_tensor, async_op=True)
        work.wait(timeout=10.0)  # 10 second timeout
        
        expected = sum(range(world_size))
        print(f"[Rank {rank}] After all-reduce: {test_tensor[:2]} (expected: {expected})")
        
        if torch.allclose(test_tensor, torch.ones_like(test_tensor) * expected):
            print(f"[Rank {rank}] ✅ All-reduce test PASSED")
        else:
            print(f"[Rank {rank}] ❌ All-reduce test FAILED")
            
    except Exception as e:
        print(f"[Rank {rank}] ❌ All-reduce failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"[Rank {rank}] 🎉 Test completed!")
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()