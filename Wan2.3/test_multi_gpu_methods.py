#!/usr/bin/env python3
"""
Multi-GPU Test Script - Tests different approaches
1. Pure Ulysses (recommended)
2. DataParallel fallback
3. Single GPU fallback
"""

import os
import sys
import subprocess
import time

def test_approach(name, cmd, timeout=300):
    """Test a specific approach with timeout"""
    print(f"\n🧪 Testing {name}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Monitor for timeout
        start_time = time.time()
        while True:
            if process.poll() is not None:  # Process finished
                stdout, _ = process.communicate()
                if process.returncode == 0:
                    print(f"✅ {name} succeeded!")
                    return True
                else:
                    print(f"❌ {name} failed with code {process.returncode}")
                    return False
            
            if time.time() - start_time > timeout:
                print(f"⏰ {name} timed out after {timeout}s")
                process.terminate()
                process.wait()
                return False
                
            time.sleep(5)
            
    except Exception as e:
        print(f"❌ {name} error: {e}")
        return False

def main():
    os.chdir('/workspace/wan22-comfy-project/Wan2.2')
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12377'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_SHM_DISABLE'] = '1'
    
    python_path = '/workspace/wan22-comfy-project/Wan2.2/venv/bin/python'
    
    # Approach 1: Pure Ulysses (most likely to work)
    ulysses_cmd = [
        python_path, '-m', 'torch.distributed.run', '--nproc_per_node=2',
        'generate.py', '--task', 's2v-14B', '--size', '480*832', 
        '--ckpt_dir', '/home/caches/Wan2.2-S2V-14B',
        '--ulysses_size', '2',  # Only sequence parallel
        '--prompt', 'A woman speaking naturally with slight movements.',
        '--image', '/workspace/wan22-comfy-project/prompt.png',
        '--audio', '/workspace/wan22-comfy-project/tmpi27jbzzb.wav',
        '--infer_frames', '24', '--sample_steps', '8',
        '--save_file', '/workspace/wan22-comfy-project/outputs/test_ulysses.mp4'
    ]
    
    # Approach 2: Single GPU with offload (known working)
    single_cmd = [
        python_path, 'generate.py', '--task', 's2v-14B', '--size', '480*832',
        '--ckpt_dir', '/home/caches/Wan2.2-S2V-14B', '--offload_model', 'True',
        '--prompt', 'A woman speaking naturally with slight movements.',
        '--image', '/workspace/wan22-comfy-project/prompt.png',
        '--audio', '/workspace/wan22-comfy-project/tmpi27jbzzb.wav',
        '--infer_frames', '24', '--sample_steps', '8',
        '--save_file', '/workspace/wan22-comfy-project/outputs/test_single.mp4'
    ]
    
    print("🎯 Multi-GPU Strategy Test")
    print("=" * 50)
    
    # Test Ulysses first (most promising)
    if test_approach("Pure Ulysses Sequence Parallel", ulysses_cmd, timeout=180):
        print("✅ SUCCESS: Ulysses sequence parallel works!")
        return
    
    print("\n🔄 Ulysses failed, trying single GPU fallback...")
    if test_approach("Single GPU with CPU Offload", single_cmd, timeout=180):
        print("✅ SUCCESS: Single GPU works (fallback option)")
        return
        
    print("\n❌ All approaches failed")

if __name__ == "__main__":
    main()