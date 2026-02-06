#!/usr/bin/env python3
"""
Multi-GPU S2V Generation Script
Optimized for faster generation with reduced parameters
Auto-manages cache for RunPod on-demand instances
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
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        print("\n💡 Trying manual setup...")
        
        # If quick setup fails, try interactive mode
        try:
            result = subprocess.run([python_path, setup_script, 'setup'], 
                                  check=True)
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to set up models. Please run setup_s2v_cache.py manually.")
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
        
        # Test GPU memory allocation to catch issues early
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                test_tensor = torch.randn(100, 100, device=f'cuda:{i}')
                print(f"   GPU {i} memory test: ✅")
                del test_tensor
                torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"⚠️  Warning: GPU test failed: {e}")
        print("Continuing anyway, but there may be GPU issues...")
    
    # Use the built-in generate.py script with optimized parameters
    os.chdir('/workspace/wan22-comfy-project/Wan2.2')
    
    # Set environment variables for multi-GPU with better NCCL settings
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '2' 
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    
    # Conservative NCCL settings to avoid hanging
    os.environ['NCCL_SOCKET_NTHREADS'] = '1'
    os.environ['NCCL_NSOCKS_PERTHREAD'] = '1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    
    # Memory optimization settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # A40 architecture    # CUDA memory optimization - critical for large model loading
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:1024'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # NCCL configuration optimized for RunPod/container environments
    # Based on successful single GPU run, use more conservative NCCL settings
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # Enable async error handling
    os.environ['NCCL_P2P_DISABLE'] = '0'  # Keep P2P enabled (works well on A40s)
    os.environ['NCCL_IB_DISABLE'] = '1'   # Disable InfiniBand
    os.environ['NCCL_DEBUG'] = 'WARN'     # Reduce debug verbosity
    os.environ['NCCL_TIMEOUT'] = '1800'   # Longer timeout for large model loading
    os.environ['NCCL_BLOCKING_WAIT'] = '1'  # Use blocking wait to avoid hangs
    os.environ['NCCL_TREE_THRESHOLD'] = '0'  # Force ring algorithm
    
    # Additional NCCL fixes for container environments
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0,lo'  # Specify network interfaces
    os.environ['NCCL_IGNORE_CPU_AFFINITY'] = '1'  # Ignore CPU affinity in containers
    
    # Ulysses sequence parallelism settings
    os.environ['TORCH_NCCL_AVOID_RECORD_STREAMS'] = '1'  # Avoid record streams
    
    # Debugging environment variables (valid values: OFF, INFO, DETAIL)
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    os.environ['TORCH_CPP_LOG_LEVEL'] = 'WARNING'
    
    # Optimize for A40 GPUs - force PCI bus order
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Explicitly set visible devices
    
    print("🔧 Environment variables set for distributed training")
    print(f"   MASTER_PORT: {os.environ['MASTER_PORT']}")
    print(f"   NCCL_DEBUG: {os.environ['NCCL_DEBUG']}")
    print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
    print("🎯 Using PURE ULYSSES sequence parallelism (no FSDP) for balanced GPU usage")
    
    # Command to run multi-GPU S2V generation - FIXED FSDP hanging issue
    cmd = [
        '/workspace/wan22-comfy-project/Wan2.2/venv/bin/python',
        '-m', 'torch.distributed.run',
        '--nproc_per_node=2',
        '--nnodes=1',
        '--max_restarts=0',  # Don't restart on failure
        '--rdzv_backend=c10d',  # Use C10d rendezvous
        '--rdzv_endpoint=localhost:12345',  # Match master port
        '--start_method=spawn',  # Use spawn method for better isolation
        'generate.py',
        '--task', 's2v-14B',
        '--size', '480*832',  # Use standard supported size
        '--ckpt_dir', '/home/caches/Wan2.2-S2V-14B',
        '--ulysses_size', '2',  # Pure Ulysses sequence parallelism - WORKING!
        # DO NOT USE --dit_fsdp - this causes the hanging issue with sync_module_states
        '--offload_model', 'False',  # Keep models on GPU for multi-GPU usage
        '--prompt', 'A sharp, high-definition (4K) video of an excited woman speaking with a bright smile and expressive facial gestures. Her body movements are energetic but controlled. The scene is filmed with a handheld camera, featuring a slight, natural shake and subtle panning movements. Casual home setting with soft, natural light and sharp focus.',
        '--image', '/workspace/wan22-comfy-project/prompt.png',
        '--audio', '/workspace/wan22-comfy-project/tmpi27jbzzb.wav',
        '--infer_frames', '24',  # Moderate frame count for balanced performance
        '--sample_steps', '8',   # Moderate sampling steps 
        '--sample_shift', '2.0', # Lower shift value
        '--sample_guide_scale', '3.0',  # Reduced guidance scale
        '--base_seed', '42',
        '--save_file', '/workspace/wan22-comfy-project/outputs/s2v_multi_gpu_fixed.mp4'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    os.execv(cmd[0], cmd)

if __name__ == "__main__":
    main()