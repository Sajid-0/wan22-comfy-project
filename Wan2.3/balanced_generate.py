#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, '/workspace/wan22-comfy-project/Wan2.2')

import torch
import torch.distributed as dist
import argparse
from wan.speech2video import WanS2V
from wan.configs.wan_s2v_14B import s2v_14B as config
from wan.utils.utils import save_video

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='s2v-14B')
    parser.add_argument('--size', type=str, default='480*832')
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--ulysses_size', type=int, default=2)
    parser.add_argument('--dit_fsdp', action='store_true')
    parser.add_argument('--t5_cpu', action='store_true')
    parser.add_argument('--offload_model', type=str, default='True')
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--audio', type=str, required=True)
    parser.add_argument('--infer_frames', type=int, default=24)
    parser.add_argument('--sample_steps', type=int, default=8)
    parser.add_argument('--sample_shift', type=float, default=2.0)
    parser.add_argument('--sample_guide_scale', type=float, default=3.0)
    parser.add_argument('--base_seed', type=int, default=42)
    parser.add_argument('--save_file', type=str, required=True)
    return parser.parse_args()

def patch_model_loading():
    """Patch the WanS2V class to force balanced loading"""
    from wan.modules.s2v.model_s2v import WanModel_S2V
    
    # Store original
    _original_from_pretrained = WanModel_S2V.from_pretrained
    
    @classmethod
    def balanced_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Force loading without device bias"""
        
        # CRITICAL FIX: Remove device_map to prevent GPU 0 bias
        if 'device_map' in kwargs:
            del kwargs['device_map']
            print(f"🔧 Removed device_map to prevent GPU 0 bias")
        
        # Force loading without device specification
        print(f"🚀 Loading model for balanced FSDP distribution...")
        model = _original_from_pretrained(pretrained_model_name_or_path, **kwargs)
        print(f"✅ Model loaded, ready for balanced FSDP sharding")
        return model
    
    # Apply the patch
    WanModel_S2V.from_pretrained = balanced_from_pretrained
    print("✅ Balanced loading patch applied successfully!")

def main():
    args = get_args()
    
    # Initialize distributed
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device_id = local_rank
        rank = dist.get_rank()
    else:
        device_id = 0
        rank = 0
    
    print(f"🎮 Process rank: {rank}, device: {device_id}")
    
    # Apply the balanced loading patch BEFORE creating the model
    patch_model_loading()
    
    # Create model with balanced settings
    offload_model = args.offload_model.lower() == 'true'
    
    print(f"🏗️ Creating WanS2V model with balanced loading...")
    model = WanS2V(
        config=config,
        checkpoint_dir=args.ckpt_dir,
        device_id=device_id,
        rank=rank,
        t5_fsdp=False,  # Keep T5 handling simple
        dit_fsdp=args.dit_fsdp,  # Use FSDP for DiT model
        use_sp=args.ulysses_size > 1,  # Enable sequence parallel
        t5_cpu=args.t5_cpu,  # Keep T5 on CPU
        init_on_cpu=False,  # Let FSDP handle device placement
        convert_model_dtype=False
    )
    
    print(f"✅ Model created successfully on rank {rank}")
    
    # Only generate on rank 0
    if rank == 0:
        print(f"🎬 Starting generation...")
        
        # Prepare inputs
        import PIL.Image
        image = PIL.Image.open(args.image)
        
        # Generate
        video = model.generate(
            input_prompt=args.prompt,
            img=image,
            audio=args.audio,
            frame_num=args.infer_frames,
            shift=args.sample_shift,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=offload_model
        )
        
        # Save result
        os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
        save_video(video, args.save_file, fps=16)
        print(f"✅ Video saved to: {args.save_file}")
    
    # Cleanup
    if 'RANK' in os.environ:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
