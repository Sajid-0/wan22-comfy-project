#!/usr/bin/env python3
"""
Multi-GPU Generation Script for Wan2.2 Animate
Use this script instead of Gradio for proper multi-GPU support
"""

import os
import sys

# Set distributed environment
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '2' 
os.environ['LOCAL_RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12347'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.distributed as dist

# Initialize distributed
dist.init_process_group(backend='nccl')

sys.path.insert(0, '/workspace/wan22-comfy-project/Wan2.2')

from wan.animate import WanAnimate
from wan.configs.wan_animate_14B import animate_14B as config
from wan.utils.utils import save_video

def main():
    preprocessed_path = "/workspace/wan22-comfy-project/Wan2.2/preprocessed/output_1761002180"
    output_path = "/workspace/wan22-comfy-project/outputs/animate_multigpu.mp4"
    
    print("🚀 Loading model with multi-GPU support...")
    print(f"   GPUs: {torch.cuda.device_count()}")
    print(f"   Preprocessed data: {preprocessed_path}")
    
    # Load model with multi-GPU FSDP
    model = WanAnimate(
        config=config,
        checkpoint_dir="/home/caches/Wan2.2-Animate-14B",
        device_id=int(os.environ['LOCAL_RANK']),
        rank=int(os.environ['RANK']),
        t5_fsdp=True,  # Enable FSDP for T5
        dit_fsdp=True,  # Enable FSDP for DiT
        use_sp=False,
        t5_cpu=False,  # Keep on GPU with FSDP
        init_on_cpu=False,
        convert_model_dtype=True,
        use_relighting_lora=False
    )
    
    print("✅ Model loaded!")
    print("🎬 Generating video...")
    
    # Generate
    video_tensor = model.generate(
        src_root_path=preprocessed_path,
        replace_flag=False,
        clip_len=49,  # Reduced frames for memory
        refert_num=1,
        shift=5.0,
        sample_solver='dpm++',
        sampling_steps=20,
        guide_scale=1.0,
        input_prompt="视频中的人在做动作",
        n_prompt="",
        seed=42,
        offload_model=True
    )
    
    # Save only on rank 0
    if dist.get_rank() == 0:
        print("💾 Saving video...")
        save_video(video_tensor, output_path, fps=30)
        print(f"✅ Video saved to: {output_path}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
