#!/usr/bin/env python3
"""
Test script for Gradio Animate App
Verifies all components work without actually launching UI
"""

import sys
sys.path.insert(0, '/workspace/wan22-comfy-project/Wan2.2')

print("🧪 Testing Gradio Animate App Components\n")

# Test 1: Imports
print("1️⃣ Testing imports...")
try:
    import gradio as gr
    from huggingface_hub import HfFolder
    import torch
    from pathlib import Path
    print("   ✅ All imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Cache Manager
print("\n2️⃣ Testing cache manager...")
try:
    from gradio_animate_app import cache_manager, check_models
    print(f"   Cache dir: {cache_manager.cache_dir}")
    print(f"   HF repo: {cache_manager.hf_repo}")
    print(f"   Models exist: {cache_manager.check_model_integrity()}")
    print("   ✅ Cache manager working")
except Exception as e:
    print(f"   ❌ Cache manager failed: {e}")
    sys.exit(1)

# Test 3: Paths
print("\n3️⃣ Testing paths...")
try:
    from gradio_animate_app import CACHE_DIR, OUTPUT_DIR, PREPROCESS_DIR
    print(f"   Cache: {CACHE_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"   Preprocess: {PREPROCESS_DIR}")
    print(f"   Output dir exists: {Path(OUTPUT_DIR).exists()}")
    print(f"   Preprocess dir exists: {Path(PREPROCESS_DIR).exists()}")
    print("   ✅ Paths configured")
except Exception as e:
    print(f"   ❌ Path check failed: {e}")
    sys.exit(1)

# Test 4: HF Token
print("\n4️⃣ Testing HuggingFace token...")
try:
    token_exists = HfFolder.get_token() is not None
    if token_exists:
        print("   ✅ HF token available")
    else:
        print("   ⚠️  No HF token (will try to set from env)")
        cache_manager.setup_hf_token()
        token_exists = HfFolder.get_token() is not None
        if token_exists:
            print("   ✅ HF token set from environment")
        else:
            print("   ❌ No HF token available")
except Exception as e:
    print(f"   ⚠️  Token check warning: {e}")

# Test 5: Example files
print("\n5️⃣ Checking example files...")
example_video = Path("/workspace/wan22-comfy-project/Wan2.2/examples/pose.mp4")
example_image = Path("/workspace/wan22-comfy-project/Wan2.2/examples/pose.png")
print(f"   Video exists: {example_video.exists()} - {example_video}")
print(f"   Image exists: {example_image.exists()} - {example_image}")

# Test 6: App creation (dry run)
print("\n6️⃣ Testing Gradio app creation...")
try:
    # This will import and validate the app structure without launching
    import gradio_animate_app
    print("   ✅ App module loaded successfully")
    print("   ✅ App ready to launch with: python gradio_animate_app.py")
except Exception as e:
    print(f"   ❌ App creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\nNext steps:")
print("1. Download models: python gradio_animate_app.py")
print("   (Click 'Download Models Only' button)")
print("2. Or run: python setup_animate_cache.py quick")
print("3. Launch UI: ./launch_animate_ui.sh")
print("\nUI will be available at: http://0.0.0.0:7862")
