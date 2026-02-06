#!/usr/bin/env python3
"""
Multi-GPU S2V Generation Script - Ulysses Sequence Parallelism Only
Uses only Ulysses sequence parallelism (no FSDP) for maximum stability
Optimized for 2x A40 GPUs on RunPod
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
    
    # Set environment variables for multi-GPU (simple NCCL setup)
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '2' 
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Simple NCCL configuration (avoid complex features)
    os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable P2P
    os.environ['NCCL_SHM_DISABLE'] = '1'  # Disable shared memory
    os.environ['NCCL_NET_GDR_DISABLE'] = '1'  # Disable GPUDirect
    os.environ['NCCL_DEBUG'] = 'INFO'
    
    # Command to run multi-GPU S2V generation - ULYSSES ONLY (no FSDP)
    cmd = [
        '/workspace/wan22-comfy-project/Wan2.2/venv/bin/python',
        '-m', 'torch.distributed.run',
        '--nproc_per_node=2',
        '--nnodes=1',
        '--master_port=12355',
        'generate.py',
        '--task', 's2v-14B',
        '--size', '480*832', 
        '--ckpt_dir', '/home/caches/Wan2.2-S2V-14B',
        '--ulysses_size', '2',  # ⭐ ONLY Ulysses sequence parallelism
        # NO FSDP flags - this should avoid all the hanging issues
        '--prompt', 'A woman speaking with natural expressions and slight head movements. Filmed with a handheld camera in natural lighting with soft focus.',
        '--image', '/workspace/wan22-comfy-project/prompt.png',
        '--audio', '/workspace/wan22-comfy-project/tmpi27jbzzb.wav',
        '--infer_frames', '40',  # Moderate frame count
        '--sample_steps', '12',  # Reasonable sampling steps
        '--sample_shift', '3.0',
        '--sample_guide_scale', '4.0',
        '--base_seed', '42',
        '--save_file', '/workspace/wan22-comfy-project/outputs/s2v_ulysses_only_output.mp4'
    ]
    
    print("🚀 Starting Ulysses-only multi-GPU S2V generation...")
    print(f"Command: {' '.join(cmd)}")
    print("\n" + "="*80)
    print("📊 Expected behavior:")
    print("✅ NCCL initialization (should not hang)")
    print("✅ Model loading with sequence parallelism")
    print("✅ Generation progress with 2 GPU coordination")
    print("="*80)
    
    os.execv(cmd[0], cmd)

if __name__ == "__main__":
    main()