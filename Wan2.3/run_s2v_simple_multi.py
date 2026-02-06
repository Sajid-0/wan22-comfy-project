#!/usr/bin/env python3
"""
Simple Multi-GPU S2V Generation Script
Uses DataParallel instead of FSDP to avoid hanging issues
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
    
    # Use the built-in generate.py script with simple multi-GPU
    os.chdir('/workspace/wan22-comfy-project/Wan2.2')
    
    # Set CUDA optimization without distributed training
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("🚀 Running with simple multi-GPU setup (no FSDP)")
    
    # Command - use ulysses (sequence parallel) but no FSDP
    cmd = [
        '/workspace/wan22-comfy-project/Wan2.2/venv/bin/python',
        'generate.py',
        '--task', 's2v-14B',
        '--size', '480*832',
        '--ckpt_dir', '/home/caches/Wan2.2-S2V-14B',
        '--ulysses_size', '2',  # Use sequence parallel across 2 GPUs
        '--prompt', 'A sharp, high-definition (4K) video of an excited woman speaking with a bright smile and expressive facial gestures. Her body movements are energetic but controlled. The scene is filmed with a handheld camera, featuring a slight, natural shake and subtle panning movements. Casual home setting with soft, natural light and sharp focus.',
        '--image', '/workspace/wan22-comfy-project/prompt.png',
        '--audio', '/workspace/wan22-comfy-project/tmpi27jbzzb.wav',
        '--infer_frames', '32',
        '--sample_steps', '10',
        '--sample_shift', '2.0',
        '--sample_guide_scale', '3.0',
        '--base_seed', '42',
        '--save_file', '/workspace/wan22-comfy-project/outputs/s2v_simple_multi_gpu.mp4'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    os.execv(cmd[0], cmd)

if __name__ == "__main__":
    main()