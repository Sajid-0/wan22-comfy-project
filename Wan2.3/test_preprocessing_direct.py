#!/usr/bin/env python3
"""
Direct test of preprocessing to verify it works before running in Gradio
"""
import subprocess
import sys
import os
from pathlib import Path

print("🧪 Testing Wan2.2 Animate Preprocessing...")
print("")

# Paths
video_path = "/workspace/wan22-comfy-project/Wan2.2/examples/pose.mp4"
ref_image = "/workspace/wan22-comfy-project/Wan2.2/examples/pose.png"
output_path = "/workspace/wan22-comfy-project/Wan2.2/preprocessed/test_output"
ckpt_path = "/home/caches/Wan2.2-Animate-14B/process_checkpoint"

# Check inputs
print("1️⃣ Checking input files...")
if not os.path.exists(video_path):
    print(f"   ❌ Video not found: {video_path}")
    sys.exit(1)
print(f"   ✅ Video exists: {video_path}")

if not os.path.exists(ref_image):
    print(f"   ❌ Reference image not found: {ref_image}")
    sys.exit(1)
print(f"   ✅ Reference image exists: {ref_image}")

if not os.path.exists(ckpt_path):
    print(f"   ❌ Checkpoint path not found: {ckpt_path}")
    print(f"   Run setup_animate_cache.py first!")
    sys.exit(1)
print(f"   ✅ Checkpoint exists: {ckpt_path}")
print("")

# Create output directory
Path(output_path).mkdir(parents=True, exist_ok=True)
print(f"2️⃣ Output directory: {output_path}")
print("")

# Build command
python_path = "/workspace/wan22-comfy-project/venv/bin/python"
preprocess_script = "/workspace/wan22-comfy-project/Wan2.2/wan/modules/animate/preprocess/preprocess_data.py"

cmd = [
    python_path,
    preprocess_script,
    "--ckpt_path", ckpt_path,
    "--video_path", video_path,
    "--refer_path", ref_image,
    "--save_path", output_path,
    "--resolution_area", "1280", "720",
    "--retarget_flag"
]

print("3️⃣ Running preprocessing...")
print(f"   Command: {' '.join(cmd)}")
print("")
print("   This will take 1-3 minutes... Please wait...")
print("")

# Run
try:
    result = subprocess.run(
        cmd, 
        capture_output=False,  # Show output in real-time
        text=True,
        check=True,
        timeout=600
    )
    
    print("")
    print("4️⃣ Checking output files...")
    required_files = ["src_pose.mp4", "src_face.mp4", "src_ref.png"]
    
    all_good = True
    for f in required_files:
        fpath = Path(output_path) / f
        if fpath.exists():
            size = fpath.stat().st_size / (1024*1024)  # MB
            print(f"   ✅ {f} ({size:.2f} MB)")
        else:
            print(f"   ❌ {f} - MISSING!")
            all_good = False
    
    print("")
    if all_good:
        print("✅ SUCCESS! Preprocessing completed successfully!")
        print(f"📁 Output: {output_path}")
    else:
        print("❌ INCOMPLETE! Some files are missing.")
        sys.exit(1)
        
except subprocess.TimeoutExpired:
    print("❌ Preprocessing timed out after 10 minutes")
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"❌ Preprocessing failed with exit code {e.returncode}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    sys.exit(130)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
