#!/usr/bin/env python3
"""
Fallback S2V Generation Script
Single GPU with optimizations if multi-GPU fails
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
        print(f"⚠️  Warning: Could not clear CUDA cache: {e}")
    
    # Use the built-in generate.py script with single GPU
    os.chdir('/workspace/wan22-comfy-project/Wan2.2')
    
    # CUDA memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only GPU 0
    
    # Command to run single GPU S2V generation
    cmd = [
        '/workspace/wan22-comfy-project/Wan2.2/venv/bin/python',
        'generate.py',
        '--task', 's2v-14B',
        '--size', '480*832',  # Smaller size for faster generation
        '--ckpt_dir', '/home/caches/Wan2.2-S2V-14B',
        '--prompt', 'A sharp, high-definition (4K) video of an excited woman speaking with a bright smile and expressive facial gestures. Her body movements are energetic but controlled. The scene is filmed with a handheld camera, featuring a slight, natural shake and subtle panning movements. Casual home setting with soft, natural light and sharp focus.',
        '--image', '/workspace/wan22-comfy-project/prompt.png',
        '--audio', '/workspace/wan22-comfy-project/tmpi27jbzzb.wav',
        '--infer_frames', '32',  # Even smaller for testing
        '--sample_steps', '10',  # Minimal steps for fast test
        '--sample_shift', '2.0', # Lower shift value
        '--sample_guide_scale', '3.0',  # Reduced guidance scale
        '--base_seed', '42',
        '--save_file', '/workspace/wan22-comfy-project/outputs/s2v_single_gpu_test.mp4'
    ]
    
    print(f"Running single GPU command: {' '.join(cmd)}")
    os.execv(cmd[0], cmd)

if __name__ == "__main__":
    main()