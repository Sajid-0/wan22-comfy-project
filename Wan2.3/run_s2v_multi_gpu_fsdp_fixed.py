#!/usr/bin/env python3
"""
Multi-GPU S2V Generation Script - FSDP Fixed Version
FIXES the FSDP hanging issue by patching sync_module_states=False
Based on research: https://pytorch.org/docs/stable/fsdp.html#sync-module-states
"""

import os
import sys
import subprocess
import shutil

def patch_fsdp_config():
    """Patch the FSDP configuration to fix the hanging issue"""
    print("🔧 Patching FSDP configuration to fix hanging issue...")
    
    fsdp_file = '/workspace/wan22-comfy-project/Wan2.2/wan/distributed/fsdp.py'
    backup_file = fsdp_file + '.backup'
    
    # Create backup
    if not os.path.exists(backup_file):
        shutil.copy2(fsdp_file, backup_file)
        print(f"✅ Created backup: {backup_file}")
    
    # Read the current file
    with open(fsdp_file, 'r') as f:
        content = f.read()
    
    # Apply the fix: change sync_module_states=True to sync_module_states=False
    fixed_content = content.replace(
        'sync_module_states=True', 
        'sync_module_states=False  # FIXED: prevents hanging in multi-GPU setup'
    )
    
    # Also ensure device_id is properly set in the function signature
    if 'sync_module_states=sync_module_states' in fixed_content:
        # The parameter is already configurable, just change the default
        fixed_content = fixed_content.replace(
            'sync_module_states=True',
            'sync_module_states=False'  # Default to False to prevent hanging
        )
    
    # Write the fixed content
    with open(fsdp_file, 'w') as f:
        f.write(fixed_content)
    
    print("✅ FSDP configuration patched successfully!")
    return True

def restore_fsdp_config():
    """Restore the original FSDP configuration"""
    fsdp_file = '/workspace/wan22-comfy-project/Wan2.2/wan/distributed/fsdp.py'
    backup_file = fsdp_file + '.backup'
    
    if os.path.exists(backup_file):
        shutil.copy2(backup_file, fsdp_file)
        print("✅ FSDP configuration restored from backup")

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
    try:
        # Patch FSDP first
        if not patch_fsdp_config():
            print("❌ Failed to patch FSDP configuration")
            sys.exit(1)
        
        # Ensure models are ready
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
        
        os.chdir('/workspace/wan22-comfy-project/Wan2.2')
        
        # Set environment variables for distributed training with FSDP fix
        env_vars = {
            'RANK': '0',
            'WORLD_SIZE': '2',
            'LOCAL_RANK': '0', 
            'MASTER_ADDR': 'localhost',
            'MASTER_PORT': '12345',
            
            # CRITICAL: FSDP-specific environment variables to prevent hanging
            'NCCL_ASYNC_ERROR_HANDLING': '1',
            'NCCL_BLOCKING_WAIT': '1',
            'NCCL_DEBUG': 'WARN',
            'NCCL_TIMEOUT': '1800',
            
            # Memory optimization 
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,max_split_size_mb:512',
            'CUDA_LAUNCH_BLOCKING': '0',
            
            # Device management
            'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'CUDA_VISIBLE_DEVICES': '0,1',
            
            # Reduce verbosity (valid values: OFF, INFO, DETAIL)
            'TORCH_DISTRIBUTED_DEBUG': 'INFO',
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        print("🎯 Using FSDP with FIXED sync_module_states=False")
        print(f"   Environment configured for 2 GPUs with proper NCCL settings")
        
        # Command with FSDP enabled but FIXED configuration
        cmd = [
            '/workspace/wan22-comfy-project/Wan2.2/venv/bin/python',
            '-m', 'torch.distributed.run',
            '--nproc_per_node=2',
            '--nnodes=1',
            '--max_restarts=0',
            '--rdzv_backend=c10d',
            '--rdzv_endpoint=localhost:12345',
            'generate.py',
            '--task', 's2v-14B',
            '--size', '480*832',
            '--ckpt_dir', '/home/caches/Wan2.2-S2V-14B',
            '--ulysses_size', '2',  # Ulysses sequence parallelism
            '--dit_fsdp',           # FSDP enabled with FIXED configuration!
            '--t5_cpu',             # Keep T5 on CPU to avoid additional complications
            '--offload_model', 'False',
            '--prompt', 'A woman speaking with natural expressions and hand gestures.',
            '--image', '/workspace/wan22-comfy-project/prompt.png', 
            '--audio', '/workspace/wan22-comfy-project/tmpi27jbzzb.wav',
            '--infer_frames', '24',
            '--sample_steps', '8',
            '--sample_shift', '2.0',
            '--sample_guide_scale', '3.5',
            '--base_seed', '42',
            '--save_file', '/workspace/wan22-comfy-project/outputs/s2v_fsdp_fixed.mp4'
        ]
        
        print(f"🚀 Running FSDP-fixed command:")
        print(f"   {' '.join(cmd)}")
        
        # Run the command
        os.execv(cmd[0], cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        restore_fsdp_config()
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error: {e}")
        restore_fsdp_config() 
        sys.exit(1)

if __name__ == "__main__":
    main()