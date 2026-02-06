#!/usr/bin/env python3
"""
Multi-GPU Wan-Animate Generation Script
Character animation and replacement with motion transfer
Auto-manages cache for RunPod on-demand instances
"""

import os
import sys
import subprocess

def ensure_models_ready():
    """Ensure models are downloaded and ready for use"""
    print("🔍 Checking if Animate models are ready...")
    
    # Run the cache setup script in quick mode
    setup_script = os.path.join(os.path.dirname(__file__), 'setup_animate_cache.py')
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
            print("❌ Failed to set up models. Please run setup_animate_cache.py manually.")
            return False

def check_preprocessed_data(data_path):
    """Check if preprocessed data exists"""
    required_files = [
        "src_pose.mp4",
        "src_face.mp4",
        "src_ref.png"
    ]
    
    if not os.path.exists(data_path):
        print(f"❌ Preprocessed data directory not found: {data_path}")
        print("   You must preprocess your video first!")
        return False
    
    missing = []
    for f in required_files:
        if not os.path.exists(os.path.join(data_path, f)):
            missing.append(f)
    
    if missing:
        print(f"❌ Missing preprocessed files: {missing}")
        print("   Please run preprocessing first:")
        print("   python wan/modules/animate/preprocess/preprocess_data.py \\")
        print("     --ckpt_path ./Wan2.2-Animate-14B/process_checkpoint \\")
        print("     --video_path <your_video.mp4> \\")
        print("     --refer_path <character_image.jpg> \\")
        print("     --save_path <output_dir> \\")
        print("     --resolution_area 1280 720 \\")
        print("     --retarget_flag")  # For animation mode
        return False
    
    print("✅ Preprocessed data found!")
    return True

def main():
    # Ensure models are ready before proceeding
    if not ensure_models_ready():
        print("❌ Cannot proceed without models. Exiting.")
        sys.exit(1)
    
    # Check if preprocessed data exists
    preprocessed_path = "/workspace/wan22-comfy-project/Wan2.2/examples/wan_animate/animate/process_results"
    
    if not check_preprocessed_data(preprocessed_path):
        print("\n" + "="*60)
        print("PREPROCESSING REQUIRED")
        print("="*60)
        print("Wan-Animate requires preprocessing your input video.")
        print("This extracts pose, face, and reference data.")
        print("\nRun preprocessing first, then run this script again.")
        print("="*60)
        sys.exit(1)
    
    # Use the built-in generate.py script with optimized parameters
    os.chdir('/workspace/wan22-comfy-project/Wan2.2')
    
    # Set environment variables for multi-GPU
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '2' 
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12347'  # Different port from S2V and I2V
    
    # Command to run multi-GPU Animate generation
    # This is for ANIMATION mode (character mimics motion)
    cmd = [
        '/workspace/wan22-comfy-project/venv/bin/python',
        '-m', 'torch.distributed.run',
        '--nproc_per_node=2',
        '--nnodes=1',
        'generate.py',
        '--task', 'animate-14B',
        '--ckpt_dir', '/home/caches/Wan2.2-Animate-14B',
        '--dit_fsdp',  # Enable FSDP for DiT model
        '--t5_fsdp',   # Enable FSDP for T5 model
        '--ulysses_size', '2',  # Use sequence parallel
        '--src_root_path', preprocessed_path,  # Path to preprocessed data
        '--refert_num', '1',  # Number of reference frames (1 or 5)
        '--frame_num', '77',  # Number of frames per clip (4n+1)
        '--sample_steps', '20',  # Sampling steps
        '--sample_shift', '5.0',  # Noise schedule shift
        '--sample_guide_scale', '1.0',  # Guidance scale (usually 1.0 for Animate)
        '--base_seed', '42',
        '--save_file', '/workspace/wan22-comfy-project/outputs/animate_output.mp4'
    ]
    
    # Note: For REPLACEMENT mode, add these flags:
    # '--replace_flag',
    # '--use_relighting_lora',
    
    print("="*60)
    print("Running Wan-Animate in ANIMATION mode")
    print("="*60)
    print(f"Command: {' '.join(cmd)}")
    print("\n💡 To use REPLACEMENT mode instead:")
    print("   Edit this script and uncomment --replace_flag and --use_relighting_lora")
    print("   Also ensure your preprocessed data includes src_bg.mp4 and src_mask.mp4")
    print("="*60)
    
    os.execv(cmd[0], cmd)

if __name__ == "__main__":
    main()
