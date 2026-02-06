#!/usr/bin/env python3
"""
Simple S2V Client - Test Speech/Text/Image to Video generation
"""

import requests
import json
import base64
from typing import Optional

def encode_file_to_base64(file_path: str) -> str:
    """Encode a file to base64"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def test_s2v_generation(
    server_url: str = "http://localhost:8000/WanS2VServer",
    prompt: str = "A cat playing piano",
    audio_file: Optional[str] = None,
    image_file: Optional[str] = None,
    height: int = 480,
    width: int = 640,
    frames: int = 25
):
    """Test the S2V server with different input modalities"""
    
    request_data = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_frames": frames,
        "guidance_scale": 3.0,  # Lower for faster generation
        "num_inference_steps": 10,  # Minimum steps for fastest generation
        "seed": 42
    }
    
    # Add audio if provided
    if audio_file:
        request_data["audio"] = encode_file_to_base64(audio_file)
        print(f"🎵 Added audio: {audio_file}")
    
    # Add image if provided
    if image_file:
        request_data["image"] = encode_file_to_base64(image_file)
        print(f"🖼️ Added image: {image_file}")
    
    print(f"🎬 Generating video with prompt: '{prompt}'")
    
    response = requests.post(server_url, json=request_data, timeout=300)
    result = response.json()
    
    print(f"✅ Result: {json.dumps(result, indent=2)}")
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--audio", help="Audio file path")
    parser.add_argument("--image", help="Image file path") 
    parser.add_argument("--server", default="http://localhost:8000/WanS2VServer")
    
    args = parser.parse_args()
    
    test_s2v_generation(
        server_url=args.server,
        prompt=args.prompt,
        audio_file=args.audio,
        image_file=args.image
    )