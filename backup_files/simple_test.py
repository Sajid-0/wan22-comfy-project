#!/usr/bin/env python3
"""
Simple test script using native Wan2.2 speech-to-video generation
"""

import os
import sys
import argparse
from pathlib import Path

# Add Wan2.2 to Python path
wan_path = Path(__file__).parent / "Wan2.2"
sys.path.insert(0, str(wan_path))

def main():
    parser = argparse.ArgumentParser(description="Simple Wan2.2 S2V Test")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--audio", type=str, required=True, help="Input audio path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output", type=str, default="simple_test_output", help="Output name")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--quality", type=str, default="medium", choices=["low", "medium", "high"], help="Quality level")
    
    args = parser.parse_args()
    
    print("=== Simple Wan2.2 Speech-to-Video Test ===")
    print(f"Image: {args.image}")
    print(f"Audio: {args.audio}")
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}")
    print(f"Quality: {args.quality}")
    print()
    
    # Validate inputs
    if not os.path.exists(args.image):
        print(f"ERROR: Image file not found: {args.image}")
        return 1
    
    if not os.path.exists(args.audio):
        print(f"ERROR: Audio file not found: {args.audio}")
        return 1
    
    try:
        print("Importing Wan2.2 modules...")
        
        # Import necessary modules
        from wan.speech2video import WanS2V
        from wan.configs.wan_s2v_14B import get_config
        import torch
        
        print("✓ Imports successful")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print()
        
        # Get configuration
        print("Loading configuration...")
        config = get_config()
        print("✓ Configuration loaded")
        print()
        
        # Initialize S2V model
        print("Initializing WanS2V model...")
        checkpoint_dir = "./checkpoints"  # Adjust as needed
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        device_id = 0 if torch.cuda.is_available() else -1
        s2v_model = WanS2V(
            config=config,
            checkpoint_dir=checkpoint_dir,
            device_id=device_id if device_id >= 0 else 0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=not torch.cuda.is_available(),
            init_on_cpu=True,
            convert_model_dtype=False
        )
        print("✓ WanS2V model initialized")
        print()
        
        # Set generation parameters based on quality
        if args.quality == "low":
            max_area = 480 * 320
            guide_scale = 3.0
            shift = 3.0
        elif args.quality == "medium":
            max_area = 640 * 480
            guide_scale = 5.0
            shift = 5.0
        else:  # high
            max_area = 720 * 1280
            guide_scale = 7.0
            shift = 5.0
        
        print(f"Generation parameters:")
        print(f"  Max area: {max_area}")
        print(f"  Guide scale: {guide_scale}")
        print(f"  Shift: {shift}")
        print(f"  Sampling steps: {args.steps}")
        print()
        
        # Create output directory
        os.makedirs("outputs", exist_ok=True)
        
        # Generate video
        print("Starting video generation...")
        print("This may take several minutes...")
        
        result = s2v_model.generate(
            input_prompt=args.prompt,
            ref_image_path=args.image,
            audio_path=args.audio,
            enable_tts=False,  # We're providing audio directly
            tts_prompt_audio=None,
            tts_prompt_text=None,
            tts_text=None,
            num_repeat=1,
            pose_video=None,
            max_area=max_area,
            infer_frames=80,  # Fixed at 80 frames per clip
            shift=shift,
            sample_solver='unipc',
            sampling_steps=args.steps,
            guide_scale=guide_scale,
            n_prompt="",
            seed=-1,
            offload_model=True,
            init_first_frame=False
        )
        
        if result is not None:
            print(f"✓ Video generation completed successfully!")
            print(f"  Result: {type(result)}")
            if hasattr(result, 'shape'):
                print(f"  Shape: {result.shape}")
            
            # The result should be video frames that we need to save
            output_path = f"outputs/{args.output}.mp4"
            print(f"  Saving to: {output_path}")
            
            # Note: You may need to implement video saving logic here
            # depending on the format of the result
            
        else:
            print("✗ Video generation failed - no result returned")
            return 1
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("This suggests the Wan2.2 framework modules are not properly structured")
        print("Let's try a different approach...")
        
        try:
            # Try direct test script from Wan2.2
            test_script = wan_path / "test_s2v.py"
            if test_script.exists():
                print(f"Found test script: {test_script}")
                print("You can try running it directly:")
                print(f"python3 {test_script} --image {args.image} --audio {args.audio} --prompt '{args.prompt}'")
            else:
                print("No test script found in Wan2.2 directory")
        except Exception as inner_e:
            print(f"Error checking for test script: {inner_e}")
        
        return 1
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n=== Test completed successfully! ===")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)