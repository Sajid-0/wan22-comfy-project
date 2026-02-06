#!/usr/bin/env python3
"""
TRULY BALANCED Multi-GPU S2V Generation Script
Forces manual device placement to achieve balanced GPU memory usage
"""

import os
import sys
import subprocess
import logging

def ensure_models_ready():
    """Ensure models are downloaded and ready for use"""
    print("🔍 Checking if S2V models are ready...")
    
    # Run the cache setup script in quick mode
    setup_script = os.path.join(os.path.dirname(__file__), 'setup_s2v_cache.py')
    python_path = '/workspace/wan22-comfy-project/Wan2.2/venv/bin/python'
    
    try:
        result = subprocess.run([python_path, setup_script, 'quick'], 
                              capture_output=True, text=True, check=True)
        print("✅ Models are ready!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Model setup failed: {e}")
        return False

def main():
    # Ensure models are ready before proceeding
    if not ensure_models_ready():
        print("❌ Cannot proceed without models. Exiting.")
        sys.exit(1)
    
    # Clear any existing CUDA cache
    print("🧹 Clearing CUDA cache...")
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"✅ Found {torch.cuda.device_count()} GPUs")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                
    except Exception as e:
        print(f"⚠️  Warning: GPU test failed: {e}")
    
    # Use the built-in generate.py script with optimized parameters
    os.chdir('/workspace/wan22-comfy-project/Wan2.2')
    
    # Environment variables for TRUE balanced GPU usage
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '2' 
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12347'  # Different port
    
    # Use newer TORCH environment variables (cleaner)
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    
    # CRITICAL: Force balanced memory allocation from the start
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512,backend:native'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # NCCL configuration optimized for balance
    os.environ['NCCL_P2P_DISABLE'] = '0' 
    os.environ['NCCL_IB_DISABLE'] = '1'  
    os.environ['NCCL_DEBUG'] = 'WARN'    
    os.environ['NCCL_TIMEOUT'] = '1800'  
    os.environ['NCCL_TREE_THRESHOLD'] = '0'  # Force ring algorithm
    
    # Container environment fixes
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0,lo'
    os.environ['NCCL_IGNORE_CPU_AFFINITY'] = '1'
    
    # Force proper GPU device ordering
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    
    # CRITICAL: Force model sharding strategy
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    os.environ['FSDP_AUTO_WRAP_POLICY'] = 'TRANSFORMER_BASED_WRAP'
    
    print("🔧 Environment configured for TRULY BALANCED multi-GPU usage")
    print(f"   MASTER_PORT: {os.environ['MASTER_PORT']}")
    print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
    print("🎯 Using TRULY BALANCED approach: FSDP + T5 CPU + Aggressive offloading")
    
    # The KEY to balance: Use BOTH FSDP AND Ulysses together
    # This forces the DiT model to be truly sharded while keeping T5 on CPU
    cmd = [
        '/workspace/wan22-comfy-project/Wan2.2/venv/bin/python',
        '-m', 'torch.distributed.run',
        '--nproc_per_node=2',
        '--nnodes=1',
        '--max_restarts=0',
        '--rdzv_backend=c10d',
        '--rdzv_endpoint=localhost:12347',
        '--start_method=spawn',
        'generate.py',
        '--task', 's2v-14B',
        '--size', '480*832',
        '--ckpt_dir', '/home/caches/Wan2.2-S2V-14B',
        '--ulysses_size', '2',  # Sequence parallelism for temporal dimension
        '--dit_fsdp',  # ENABLE FSDP for parameter sharding (this is the key!)
        '--t5_cpu',  # Keep T5 on CPU
        '--offload_model', 'True',  # Enable aggressive offloading
        '--prompt', 'A sharp, high-definition video of an excited woman speaking with a bright smile.',
        '--image', '/workspace/wan22-comfy-project/prompt.png',
        '--audio', '/workspace/wan22-comfy-project/tmpi27jbzzb.wav',
        '--infer_frames', '24',
        '--sample_steps', '8',
        '--sample_shift', '2.0',
        '--sample_guide_scale', '3.0',
        '--base_seed', '42',
        '--save_file', '/workspace/wan22-comfy-project/outputs/s2v_truly_balanced.mp4'
    ]
    
    print("🚀 Running TRULY BALANCED multi-GPU command:")
    print("   Strategy: FSDP + Ulysses + T5 CPU + Model offloading")
    print("   This will use our FIXED FSDP (sync_module_states=False) that doesn't hang!")
    print(f"   Command: {' '.join(cmd)}")
    os.execv(cmd[0], cmd)

if __name__ == "__main__":
    main()