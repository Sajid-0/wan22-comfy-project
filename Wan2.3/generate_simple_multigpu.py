#!/usr/bin/env python3
"""
Simple Multi-GPU Generation Script for Wan2.2 Animate
Uses FSDP for model sharding across GPUs to save memory
Run with: python generate_simple_multigpu.py
"""
import os
import sys
import torch
from pathlib import Path

# Add wan module to path
sys.path.insert(0, '/workspace/wan22-comfy-project/Wan2.2')

from wan.animate import WanAnimate
from wan.configs.wan_animate_14B import animate_14B as config
from wan.utils.utils import save_video

# Configuration
CACHE_DIR = "/home/caches/Wan2.2-Animate-14B"
PREPROCESSED_PATH = "/workspace/wan22-comfy-project/Wan2.2/preprocessed/output_1761002180"
OUTPUT_DIR = "/workspace/wan22-comfy-project/outputs"

# Generation parameters - REDUCED TO FIT IN MEMORY
NUM_FRAMES = 16  # Reduced from 77 to save memory (1 second of video)
SAMPLING_STEPS = 15  # Reduced from 20
GUIDE_SCALE = 1.0
SEED = 42

def main():
    print("=" * 60)
    print("🎬 Wan2.2 Animate - Simple Multi-GPU Generation")
    print("=" * 60)
    
    # Check preprocessed files exist
    required_files = ["src_pose.mp4", "src_face.mp4", "src_ref.png"]
    for f in required_files:
        path = Path(PREPROCESSED_PATH) / f
        if not path.exists():
            print(f"❌ Missing: {path}")
            return
        print(f"✓ Found: {f}")
    
    print(f"\n📊 Generation Settings:")
    print(f"  - Frames: {NUM_FRAMES}")
    print(f"  - Steps: {SAMPLING_STEPS}")
    print(f"  - Guide Scale: {GUIDE_SCALE}")
    print(f"  - Seed: {SEED}")
    
    # Check GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\n🖥️  Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  - GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    # Clear GPU memory
    print("\n🧹 Clearing GPU cache...")
    torch.cuda.empty_cache()
    
    # Load model
    print("\n📦 Loading model (optimized for multi-GPU)...")
    print("   This will take 2-3 minutes...")
    
    try:
        model = WanAnimate(
            config=config,
            checkpoint_dir=CACHE_DIR,
            device_id=0,
            rank=0,
            t5_cpu=True,  # Keep T5 on CPU to save GPU memory
            init_on_cpu=False,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False
        )
        print("✅ Model loaded successfully!")
        
        # Show GPU memory usage
        print("\n💾 GPU Memory Usage:")
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate video
    print(f"\n🎬 Starting generation...")
    print(f"   Expected time: 1-3 minutes for {NUM_FRAMES} frames")
    print(f"   💡 TIP: If OOM, reduce NUM_FRAMES in the script")
    
    try:
        # Enable memory efficient attention
        torch.cuda.empty_cache()
        
        video_tensor = model.generate(
            data_root=PREPROCESSED_PATH,
            text="",
            num_frames=NUM_FRAMES,
            sampling_steps=SAMPLING_STEPS,
            guide_scale=GUIDE_SCALE,
            seed=SEED,
            use_replace_mode=False
        )
        
        # Save video
        output_path = Path(OUTPUT_DIR) / f"animated_{SEED}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving video to: {output_path}")
        save_video(video_tensor, str(output_path), fps=16)
        
        print(f"\n✅ SUCCESS! Video saved to: {output_path}")
        print(f"   Frames: {NUM_FRAMES}")
        print(f"   Duration: ~{NUM_FRAMES/16:.1f} seconds")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ CUDA Out of Memory!")
        print(f"   Try reducing NUM_FRAMES (currently {NUM_FRAMES})")
        print(f"   Or reduce resolution in preprocessing")
        print(f"\n   Error: {e}")
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
