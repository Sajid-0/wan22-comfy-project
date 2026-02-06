#!/usr/bin/env python3
"""
Script to generate speech-to-video using Wan2.2-S2V-14B model
with iPhone image and audio file.
"""

import os
import sys
import subprocess

def main():
    # Set environment variables for cache
    os.environ['HF_HOME'] = '/home/caches'
    os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/caches'
    
    # Input files
    image_path = "/workspace/wan22-comfy-project/iphone.jpeg"
    audio_path = "/workspace/wan22-comfy-project/tmp_19iifpd.mp3"
    checkpoint_dir = "/home/caches/Wan2.2-S2V-14B"
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return 1
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return 1
        
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return 1
    
    # Generate prompt for the iPhone image
    prompt = "A smartphone displaying content, modern technology device, sleek design, high quality image"
    
    # Build the command
    python_path = "/workspace/wan22-comfy-project/venv/bin/python"
    
    # Single GPU inference command for runpod with 2x48GB GPUs
    cmd = [
        python_path, "generate.py",
        "--task", "s2v-14B",
        "--size", "1024*704",  # Good resolution for the model
        "--ckpt_dir", checkpoint_dir,
        "--prompt", prompt,
        "--image", image_path,
        "--audio", audio_path,
        "--offload_model", "True",  # For memory efficiency
        "--convert_model_dtype",    # For memory efficiency 
        "--infer_frames", "80",     # Standard frames per clip
        "--sample_steps", "40",     # Good quality vs speed balance
        "--sample_guide_scale", "4.5",  # Default for S2V
        "--base_seed", "42"         # Fixed seed for reproducibility
    ]
    
    print("Running Wan2.2-S2V generation...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Input image: {image_path}")
    print(f"Input audio: {audio_path}")
    print(f"Model: {checkpoint_dir}")
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, cwd="/workspace/wan22-comfy-project/Wan2.2")
        print("Generation completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error during generation: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())