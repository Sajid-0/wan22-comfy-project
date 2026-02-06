#!/usr/bin/env python3
"""
Single-GPU S2V Generation Script
For testing and debugging - simpler than multi-GPU
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
    
    # Use the built-in generate.py script with optimized parameters
    os.chdir('/workspace/wan22-comfy-project/Wan2.2')
    
    # CUDA memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only GPU 0
    
    # Command to run single-GPU S2V generation
    cmd = [
        '/workspace/wan22-comfy-project/Wan2.2/venv/bin/python',
        'generate.py',
        '--task', 's2v-14B',
        '--size', '480*832',  # Smaller size for faster generation
        '--ckpt_dir', '/home/caches/Wan2.2-S2V-14B',
        '--offload_model',  # Enable model offloading for single GPU
        '--prompt', 'A sharp, high-definition (4K) video of an excited woman speaking with a bright smile and expressive facial gestures. Her body movements are energetic but controlled. The scene is filmed with a handheld camera, featuring a slight, natural shake and subtle panning movements. Casual home setting with soft, natural light and sharp focus.',
        '--image', '/workspace/wan22-comfy-project/prompt.png',
        '--audio', '/workspace/wan22-comfy-project/tmpi27jbzzb.wav',
        '--infer_frames', '48',  # Reduced frames for faster generation
        '--sample_steps', '15',  # Reduced sampling steps
        '--sample_shift', '3.0', # Lower shift value
        '--sample_guide_scale', '4.0',  # Reduced guidance scale
        '--base_seed', '42',
        '--save_file', '/workspace/wan22-comfy-project/outputs/s2v_single_output.mp4'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    os.execv(cmd[0], cmd)

if __name__ == "__main__":
    main()
