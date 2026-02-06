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
    python_path = '/workspace/wan22-comfy-project/Wan2.2/venv_wan22/bin/python'
    
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
    
    # Set environment variables for multi-GPU
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '2' 
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    
    # Command to run multi-GPU S2V generation
    cmd = [
        '/workspace/wan22-comfy-project/Wan2.2/venv_wan22/bin/python',
        '-m', 'torch.distributed.run',
        '--nproc_per_node=2',
        '--nnodes=1',
        'generate.py',
        '--task', 's2v-14B',
        '--size', '480*832',  # Smaller size for faster generation , 720*1280  ,480*832
        '--ckpt_dir', '/home/caches/Wan2.2-S2V-14B',
        '--dit_fsdp',  # Enable FSDP for DiT model
        '--t5_fsdp',   # Enable FSDP for T5 model
        '--ulysses_size', '2',  # Use sequence parallel
        '--prompt', 'A sharp, high-definition (4K) video of the woman from the provided image, reciting with a gentle smile and highly expressive facial gestures. Her **elegant and rhythmic emphasis comes entirely from her head and subtle upper body movements**, aligning perfectly with her recitation. Her arms and hands remain completely still or very less movements**.  The scene is filmed with a handheld camera, featuring a slight, natural shake and subtle panning movements, focused on full body movement and face. Casual home setting with soft, natural light, matching the original image.',
        '--image', '/workspace/wan22-comfy-project/prompt_004_IR25X.png',
        '--audio', '/workspace/wan22-comfy-project/tmpi27jbzzb3.wav',
        '--num_clip', '2',  # ⚠️ CRITICAL: Generate only 1 clip (prevents multiple iterations)
        '--infer_frames', '48',  # 80 frames = ~5 seconds @ 16fps (must be multiple of 4)
        '--sample_steps', '50',  # Balanced quality/speed (min: 25, recommended: 30-40)
        '--sample_shift', '3.0', # Lower shift value for 480p
        '--sample_guide_scale', '4.5',  # Guidance scale (default: 4.5)
        '--base_seed', '42',
        '--save_file', '/workspace/wan22-comfy-project/Wan2.2/outputs/s2v_output12.mp4'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    os.execv(cmd[0], cmd)

if __name__ == "__main__":
    main()
