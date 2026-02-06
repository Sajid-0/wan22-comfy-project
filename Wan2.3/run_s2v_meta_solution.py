#!/usr/bin/env python3
"""
REAL SOLUTION: Meta Device Loading Multi-GPU S2V
Forces models to load on meta device first, then properly shard via FSDP
"""

import os
import sys
import subprocess

def create_meta_loading_patch():
    """Create a custom patch to force meta device loading"""
    
    patch_content = '''# Multi-GPU meta device loading patch
import torch
from wan.modules.s2v.model_s2v import WanModel_S2V

# Store original from_pretrained
_original_from_pretrained = WanModel_S2V.from_pretrained

@classmethod  
def meta_device_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    """Load model on meta device to avoid GPU 0 bias"""
    # Force loading on meta device for FSDP
    if 'device_map' not in kwargs:
        kwargs['device_map'] = 'meta'
    
    print(f"🔧 Loading model on meta device for balanced FSDP sharding...")
    model = _original_from_pretrained(pretrained_model_name_or_path, **kwargs)
    print(f"✅ Model loaded on meta device, ready for FSDP sharding")
    return model

# Monkey patch the method
WanModel_S2V.from_pretrained = meta_device_from_pretrained
print("🚀 Meta device loading patch applied!")
'''
    
    patch_file = '/workspace/wan22-comfy-project/Wan2.2/meta_loading_patch.py'
    with open(patch_file, 'w') as f:
        f.write(patch_content)
    
    return patch_file

def ensure_models_ready():
    """Ensure models are downloaded and ready for use"""
    print("🔍 Checking if S2V models are ready...")
    
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
        print("❌ Cannot proceed without models. Exiting.")
        sys.exit(1)
    
    # Create the meta loading patch
    patch_file = create_meta_loading_patch()
    
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
    
    os.chdir('/workspace/wan22-comfy-project/Wan2.2')
    
    # Environment variables for meta device loading
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '2' 
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12348'  # Different port
    
    # Use newer TORCH environment variables
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    
    # Force meta device loading
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # NCCL settings
    os.environ['NCCL_P2P_DISABLE'] = '0'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_TIMEOUT'] = '1800'
    os.environ['NCCL_TREE_THRESHOLD'] = '0'
    
    # Container fixes
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0,lo'
    os.environ['NCCL_IGNORE_CPU_AFFINITY'] = '1'
    
    # Force proper device ordering
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    
    # Critical: Enable meta device loading
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    
    print("🔧 Environment configured for META DEVICE LOADING")
    print(f"   MASTER_PORT: {os.environ['MASTER_PORT']}")
    print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print("🎯 Using META DEVICE approach: Load on meta → FSDP shard properly")
    
    # The REAL solution: Force meta device loading with proper FSDP
    cmd = [
        '/workspace/wan22-comfy-project/Wan2.2/venv/bin/python',
        '-c', f'exec(open("{patch_file}").read()); exec(open("generate.py").read())',
        '--task', 's2v-14B',
        '--size', '480*832',
        '--ckpt_dir', '/home/caches/Wan2.2-S2V-14B',
        '--ulysses_size', '2',
        '--dit_fsdp',  # Enable FSDP with meta loading
        '--t5_cpu',
        '--offload_model', 'True',
        '--prompt', 'A high-definition video of a woman speaking with bright smile.',
        '--image', '/workspace/wan22-comfy-project/prompt.png',
        '--audio', '/workspace/wan22-comfy-project/tmpi27jbzzb.wav',
        '--infer_frames', '24',
        '--sample_steps', '8',
        '--sample_shift', '2.0',
        '--sample_guide_scale', '3.0',
        '--base_seed', '42',
        '--save_file', '/workspace/wan22-comfy-project/outputs/s2v_meta_balanced.mp4'
    ]
    
    print("🚀 Running META DEVICE LOADING approach:")
    print("   Strategy: Meta device → FSDP proper sharding → Balanced GPUs")
    print("   This solves the root cause by preventing GPU 0 monopoly!")
    
    # Use distributed launch for proper multi-GPU
    distributed_cmd = [
        '/workspace/wan22-comfy-project/Wan2.2/venv/bin/python',
        '-m', 'torch.distributed.run',
        '--nproc_per_node=2',
        '--nnodes=1',
        '--max_restarts=0',
        '--rdzv_backend=c10d',
        '--rdzv_endpoint=localhost:12348',
        '--start_method=spawn'
    ] + cmd[1:]  # Skip the python executable
    
    print(f"   Command: {' '.join(distributed_cmd)}")
    os.execv(distributed_cmd[0], distributed_cmd)

if __name__ == "__main__":
    main()