#!/usr/bin/env python3
"""
Balanced Multi-GPU S2V Generation Script
Forces better GPU memory distribution by optimizing model placement
"""

import os
import sys
import subprocess

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
    
    # Environment variables optimized for balanced GPU usage
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '2' 
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12346'  # Different port to avoid conflicts
    
    # Use newer TORCH environment variables instead of deprecated NCCL ones
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    
    # CUDA memory optimization for balanced loading
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # NCCL configuration optimized for balanced memory usage
    os.environ['NCCL_P2P_DISABLE'] = '0'  # Keep P2P enabled
    os.environ['NCCL_IB_DISABLE'] = '1'   # Disable InfiniBand
    os.environ['NCCL_DEBUG'] = 'WARN'     # Reduce debug verbosity
    os.environ['NCCL_TIMEOUT'] = '1800'   # Longer timeout
    os.environ['NCCL_TREE_THRESHOLD'] = '0'  # Force ring algorithm
    
    # Container environment fixes
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0,lo'
    os.environ['NCCL_IGNORE_CPU_AFFINITY'] = '1'
    
    # Force better GPU distribution
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    
    # Force model sharding
    os.environ['WAN_FORCE_MODEL_BALANCE'] = '1'  # Custom env var for our use
    
    print("🔧 Environment configured for BALANCED multi-GPU usage")
    print(f"   MASTER_PORT: {os.environ['MASTER_PORT']}")
    print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
    print("🎯 Using BALANCED approach: T5 on CPU + DiT Ulysses + Model offloading")
    
    # Optimized command for balanced GPU usage
    # Key strategy: Put T5 on CPU and use model offloading to balance memory
    cmd = [
        '/workspace/wan22-comfy-project/Wan2.2/venv/bin/python',
        '-m', 'torch.distributed.run',
        '--nproc_per_node=2',
        '--nnodes=1',
        '--max_restarts=0',
        '--rdzv_backend=c10d',
        '--rdzv_endpoint=localhost:12346',
        '--start_method=spawn',
        'generate.py',
        '--task', 's2v-14B',
        '--size', '480*832',
        '--ckpt_dir', '/home/caches/Wan2.2-S2V-14B',
        '--ulysses_size', '2',  # Ulysses sequence parallelism for DiT model
        '--t5_cpu',  # Move T5 to CPU to free GPU memory
        '--offload_model', 'True',  # Enable model offloading for balance
        '--prompt', 'A sharp, high-definition (4K) video of an excited woman speaking with a bright smile and expressive facial gestures.',
        '--image', '/workspace/wan22-comfy-project/prompt.png',
        '--audio', '/workspace/wan22-comfy-project/tmpi27jbzzb.wav',
        '--infer_frames', '24',
        '--sample_steps', '8',
        '--sample_shift', '2.0',
        '--sample_guide_scale', '3.0',
        '--base_seed', '42',
        '--save_file', '/workspace/wan22-comfy-project/outputs/s2v_balanced.mp4'
    ]
    
    print("🚀 Running BALANCED multi-GPU command:")
    print("   Strategy: T5 on CPU + DiT Ulysses parallelism + Model offloading")
    print(f"   Command: {' '.join(cmd)}")
    os.execv(cmd[0], cmd)

if __name__ == "__main__":
    main()