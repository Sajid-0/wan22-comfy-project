#!/usr/bin/env python3
"""
Simple S2V Generation Script with reduced settings for faster generation
"""

import os
import sys
import torch

# Add the Wan directory to path
sys.path.insert(0, '/workspace/wan22-comfy-project/Wan2.2')

def main():
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use both GPUs
    
    # Import after setting path
    from wan.configs.wan_s2v_14B import s2v_14B
    from wan.speech2video import WanS2V
    
    print("Initializing WanS2V model...")
    
    # Create model with reduced settings
    config = s2v_14B
    checkpoint_dir = "/home/caches/Wan2.2-S2V-14B"
    
    # Initialize the model
    wan_s2v = WanS2V(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,  # Use CPU for T5 to save GPU memory
        init_on_cpu=True,
        convert_model_dtype=True,
    )
    
    print("Starting video generation...")
    
    # Generate video with reduced settings
    video = wan_s2v.generate(
        input_prompt="A person's face and expressions, natural movement, clear details",
        ref_image_path="/workspace/wan22-comfy-project/iphone.jpeg",
        audio_path="/workspace/wan22-comfy-project/tmp_19iifpd.mp3",
        enable_tts=False,
        tts_prompt_audio=None,
        tts_prompt_text=None,
        tts_text=None,
        num_repeat=1,  # Generate only 1 clip
        pose_video=None,
        max_area=480*832,  # Smaller resolution for faster generation
        infer_frames=48,   # Fewer frames per clip
        shift=3.0,         # Reduced shift for faster generation
        sample_solver='unipc',
        sampling_steps=20, # Reduced steps for faster generation
        guide_scale=4.0,   # Reduced guidance scale
        n_prompt="blurry, low quality, distorted, static",
        seed=42,
        offload_model=True,  # Offload model to save memory
        init_first_frame=False,
    )
    
    if video is not None:
        print("Saving video...")
        from wan.utils.utils import save_video
        
        # Save the generated video
        output_path = "/workspace/wan22-comfy-project/outputs/generated_s2v_simple.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        save_video(
            tensor=video[None],
            save_file=output_path,
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        
        print(f"Video saved to: {output_path}")
    else:
        print("No video generated")

if __name__ == "__main__":
    main()