#!/usr/bin/env python3
"""
Multi-GPU I2V Generation Script for Wan2.2-I2V-A14B
Optimized for efficient generation with FSDP + Ulysses sequence parallelism
Auto-manages cache for RunPod on-demand instances
"""

import os
import sys
import subprocess

def ensure_models_ready():
    """Ensure models are downloaded and ready for use"""
    print("🔍 Checking if I2V models are ready...")
    
    # Run the cache setup script in quick mode
    setup_script = os.path.join(os.path.dirname(__file__), 'setup_i2v_cache.py')
    python_path = '/workspace/wan22-comfy-project/venv/bin/python'
    
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
            print("❌ Failed to set up models. Please run setup_i2v_cache.py manually.")
            return False

def main():
    # Ensure models are ready before proceeding
    if not ensure_models_ready():
        print("❌ Cannot proceed without models. Exiting.")
        sys.exit(1)
    
    # Use the built-in generate.py script with optimized parameters
    os.chdir('/workspace/wan22-comfy-project/Wan2.2')
    
    # Set environment variables for multi-GPU
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '2' 
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12346'  # Different port from S2V
    
    # Command to run multi-GPU I2V generation
    cmd = [
        '/workspace/wan22-comfy-project/venv/bin/python',
        '-m', 'torch.distributed.run',
        '--nproc_per_node=2',
        '--nnodes=1',
        'generate.py',
        '--task', 'i2v-A14B',
        '--size', '1280*720',  # 720P resolution (also supports 480*832 for 480P)
        '--ckpt_dir', '/home/caches/Wan2.2-I2V-A14B',
        '--dit_fsdp',  # Enable FSDP for DiT model (required for multi-GPU)
        '--t5_fsdp',   # Enable FSDP for T5 model (required for multi-GPU)
        '--ulysses_size', '2',  # Use sequence parallel with 2 GPUs
        '--prompt', 'Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline\'s intricate details and the refreshing atmosphere of the seaside.',
        '--image', '/workspace/wan22-comfy-project/Wan2.2/examples/i2v_input.JPG',  # Input image
        '--frame_num', '81',  # Number of frames (must be 4n+1)
        '--sample_steps', '40',  # Sampling steps (default for I2V)
        '--sample_shift', '5.0',  # Shift value (use 3.0 for 480P, 5.0 for 720P)
        '--sample_guide_scale', '3.5',  # Guidance scale (can use tuple for MoE: "3.5,3.5")
        '--base_seed', '42',
        '--save_file', '/workspace/wan22-comfy-project/outputs/i2v_output.mp4'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    os.execv(cmd[0], cmd)

if __name__ == "__main__":
    main()
