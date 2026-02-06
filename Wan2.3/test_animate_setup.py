#!/usr/bin/env python3
"""
Quick test script to verify Wan2.2 Animate setup
"""

import sys
sys.path.insert(0, '/workspace/wan22-comfy-project/Wan2.2')

print("🔍 Testing Wan2.2 Animate Setup...")
print()

# Test 1: Imports
print("1️⃣ Testing imports...")
try:
    from wan.animate import WanAnimate
    from wan.configs.wan_animate_14B import animate_14B
    from wan.utils.utils import save_video
    print("   ✅ All imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Check model cache
print()
print("2️⃣ Checking model cache...")
from pathlib import Path
cache_dir = Path("/home/caches/Wan2.2-Animate-14B")
required_file = cache_dir / "diffusion_pytorch_model-00001-of-00004.safetensors"

if required_file.exists():
    print(f"   ✅ Models found in {cache_dir}")
    # Calculate size
    total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
    size_gb = total_size / (1024**3)
    print(f"   📦 Cache size: {size_gb:.2f} GB")
else:
    print(f"   ⚠️  Models not found in {cache_dir}")
    print("   💡 Run: python setup_animate_cache.py quick")

# Test 3: Check example files
print()
print("3️⃣ Checking example files...")
example_video = Path("/workspace/wan22-comfy-project/Wan2.2/examples/pose.mp4")
example_image = Path("/workspace/wan22-comfy-project/Wan2.2/examples/pose.png")

if example_video.exists() and example_image.exists():
    print("   ✅ Example files found")
    print(f"   📹 Video: {example_video}")
    print(f"   🖼️  Image: {example_image}")
else:
    print("   ⚠️  Example files not found")

# Test 4: Check Gradio
print()
print("4️⃣ Checking Gradio installation...")
try:
    import gradio as gr
    print(f"   ✅ Gradio version {gr.__version__} installed")
except Exception as e:
    print(f"   ❌ Gradio not found: {e}")

# Test 5: Check CUDA
print()
print("5️⃣ Checking CUDA/GPU...")
try:
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"   ✅ CUDA available: {gpu_count} GPU(s)")
        print(f"   🎮 GPU 0: {gpu_name}")
        
        # Check VRAM
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   💾 VRAM: {total_vram:.1f} GB")
    else:
        print("   ⚠️  CUDA not available")
except Exception as e:
    print(f"   ❌ Error checking CUDA: {e}")

# Summary
print()
print("="*50)
print("📊 Setup Summary")
print("="*50)

if required_file.exists():
    print("✅ Ready to use! Launch Gradio UI:")
    print("   ./launch_animate_ui.sh")
else:
    print("⚠️  Models not downloaded. Next steps:")
    print("   1. Run: python setup_animate_cache.py quick")
    print("   2. Then: ./launch_animate_ui.sh")

print()
print("📚 For more info, see: GRADIO_ANIMATE_GUIDE.md")
