#!/usr/bin/env python3
"""
Debug Multi-GPU S2V Generation Script
This version adds detailed logging to understand GPU distribution
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
    if not ensure_models_ready():
        sys.exit(1)
    
    os.chdir('/workspace/wan22-comfy-project/Wan2.2')
    
    # Set environment variables for debugging
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '2'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    
    # Memory and distributed settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable for debugging
    
    print("🐛 Running DEBUG version with detailed logging")
    
    # Simple distributed command with minimal parameters
    cmd = [
        '/workspace/wan22-comfy-project/Wan2.2/venv/bin/python',
        '-m', 'torch.distributed.run',
        '--nproc_per_node=2',
        '--nnodes=1',
        'generate.py',
        '--task', 's2v-14B',
        '--size', '480*832',
        '--ckpt_dir', '/home/caches/Wan2.2-S2V-14B',
        '--ulysses_size', '2',
        '--prompt', 'A woman speaking.',  # Shorter prompt
        '--image', '/workspace/wan22-comfy-project/prompt.png',
        '--audio', '/workspace/wan22-comfy-project/tmpi27jbzzb.wav',
        '--infer_frames', '8',   # Very small for debugging
        '--sample_steps', '4',   # Very small for debugging
        '--base_seed', '42',
        '--save_file', '/workspace/wan22-comfy-project/outputs/debug_test.mp4'
    ]
    
    print(f"🐛 Debug command: {' '.join(cmd)}")
    os.execv(cmd[0], cmd)

if __name__ == "__main__":
    main()