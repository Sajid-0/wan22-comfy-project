#!/usr/bin/env python3
"""
Robust Multi-GPU S2V Generation Script
Uses minimal FSDP with conservative settings to avoid hanging
"""

import os
import sys
import subprocess
import signal
import time

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

def run_with_timeout(cmd, timeout_seconds=900):  # 15 minutes timeout
    """Run command with timeout and return success status"""
    print(f"⏱️  Starting generation with {timeout_seconds//60} minute timeout...")
    
    try:
        # Start the process
        process = subprocess.Popen(cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT,
                                 text=True,
                                 bufsize=1,
                                 universal_newlines=True)
        
        start_time = time.time()
        generation_started = False
        
        # Monitor the process
        while True:
            # Check if process is still running
            poll = process.poll()
            if poll is not None:
                # Process finished
                remaining_output = process.stdout.read()
                if remaining_output:
                    print(remaining_output, end='')
                return poll == 0
            
            # Read any available output
            try:
                output = process.stdout.readline()
                if output:
                    print(output, end='')
                    # Check if generation actually started
                    if "Generating video" in output or "100%" in output:
                        generation_started = True
                        print("🎉 Generation has started!")
            except:
                pass
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                print(f"\n⏰ Timeout after {elapsed//60:.1f} minutes")
                if not generation_started:
                    print("❌ Process was stuck during model loading")
                else:
                    print("⚠️  Process was running but took too long")
                
                # Kill the process
                try:
                    process.terminate()
                    time.sleep(5)
                    if process.poll() is None:
                        process.kill()
                except:
                    pass
                return False
            
            time.sleep(1)
            
    except Exception as e:
        print(f"❌ Error running process: {e}")
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
    except Exception as e:
        print(f"⚠️  Warning: GPU test failed: {e}")
    
    os.chdir('/workspace/wan22-comfy-project/Wan2.2')
    
    # Ultra-minimal environment variables for NCCL
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12360'  # New port
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Minimal NCCL settings - only what's absolutely necessary
    os.environ['NCCL_DEBUG'] = 'INFO'  # More verbose to see what's happening
    os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable P2P to avoid issues
    os.environ['NCCL_SHM_DISABLE'] = '1'  # Disable shared memory 
    os.environ['NCCL_IB_DISABLE'] = '1'   # Disable InfiniBand
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # Use loopback only
    
    print("🔧 Testing multi-GPU with conservative NCCL settings")
    
    # Command with minimal FSDP - only for DiT
    cmd = [
        '/workspace/wan22-comfy-project/Wan2.2/venv/bin/python',
        '-m', 'torch.distributed.run',
        '--nproc_per_node=2',
        '--nnodes=1',
        '--master_port=12360',
        'generate.py',
        '--task', 's2v-14B',
        '--size', '480*832',
        '--ckpt_dir', '/home/caches/Wan2.2-S2V-14B',
        '--dit_fsdp',  # Only DiT with FSDP
        '--t5_cpu',    # Keep T5 on CPU
        '--ulysses_size', '2',
        '--prompt', 'A sharp, high-definition (4K) video of an excited woman speaking with a bright smile and expressive facial gestures. Her body movements are energetic but controlled. The scene is filmed with a handheld camera, featuring a slight, natural shake and subtle panning movements. Casual home setting with soft, natural light and sharp focus.',
        '--image', '/workspace/wan22-comfy-project/prompt.png',
        '--audio', '/workspace/wan22-comfy-project/tmpi27jbzzb.wav',
        '--infer_frames', '32',  # Good balance
        '--sample_steps', '10',  # Good balance
        '--sample_shift', '2.0',
        '--sample_guide_scale', '3.0',
        '--base_seed', '42',
        '--save_file', '/workspace/wan22-comfy-project/outputs/s2v_robust_multi_gpu.mp4'
    ]
    
    print(f"🚀 Running: {' '.join(cmd[-10:])}")  # Show last 10 args
    
    # Try the multi-GPU version with timeout
    success = run_with_timeout(cmd, timeout_seconds=600)  # 10 minutes
    
    if success:
        print("🎉 Multi-GPU generation completed successfully!")
    else:
        print("❌ Multi-GPU failed or timed out")
        print("🔄 Falling back to single GPU...")
        
        # Fallback to single GPU
        fallback_cmd = [
            '/workspace/wan22-comfy-project/Wan2.2/venv/bin/python',
            'generate.py',
            '--task', 's2v-14B',
            '--size', '480*832',
            '--ckpt_dir', '/home/caches/Wan2.2-S2V-14B',
            '--prompt', 'A sharp, high-definition (4K) video of an excited woman speaking with a bright smile and expressive facial gestures. Her body movements are energetic but controlled. The scene is filmed with a handheld camera, featuring a slight, natural shake and subtle panning movements. Casual home setting with soft, natural light and sharp focus.',
            '--image', '/workspace/wan22-comfy-project/prompt.png',
            '--audio', '/workspace/wan22-comfy-project/tmpi27jbzzb.wav',
            '--infer_frames', '24',
            '--sample_steps', '8',
            '--sample_shift', '2.0',
            '--sample_guide_scale', '3.0',
            '--base_seed', '42',
            '--save_file', '/workspace/wan22-comfy-project/outputs/s2v_fallback_single.mp4'
        ]
        
        print("🚀 Running single GPU fallback...")
        os.execv(fallback_cmd[0], fallback_cmd)

if __name__ == "__main__":
    main()