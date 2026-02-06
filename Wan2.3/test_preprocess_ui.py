#!/usr/bin/env python3
"""
Test preprocessing function to ensure it works properly
"""
import sys
import os

# Add the Wan2.2 directory to path
sys.path.insert(0, '/workspace/wan22-comfy-project/Wan2.2')

# Test the preprocess function
def test_preprocess():
    print("Testing preprocessing function...")
    
    video_path = "/workspace/wan22-comfy-project/Wan2.2/examples/pose.mp4"
    ref_image = "/workspace/wan22-comfy-project/Wan2.2/examples/pose.png"
    
    # Check files exist
    print(f"Video exists: {os.path.exists(video_path)}")
    print(f"Image exists: {os.path.exists(ref_image)}")
    
    # Import and test the function
    from gradio_animate_app import preprocess_video
    
    print("\nTesting with string paths (simulating Gradio)...")
    output_path, status = preprocess_video(video_path, ref_image, "animate")
    
    print(f"Output path: {output_path}")
    print(f"Status: {status}")
    
    if output_path:
        print(f"\n✅ Preprocessing successful!")
        print(f"Output directory: {output_path}")
        
        # List output files
        import os
        if os.path.exists(output_path):
            files = os.listdir(output_path)
            print(f"Generated files: {files}")
    else:
        print(f"\n❌ Preprocessing failed!")

if __name__ == "__main__":
    test_preprocess()
