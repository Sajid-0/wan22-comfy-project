#!/usr/bin/env python3
"""
Multi-GPU S2V System Demonstration Script
Shows how to use the system with the provided test files
"""

import os
import sys
import time
import argparse
from pathlib import Path

def print_header():
    """Print demonstration header"""
    print("=" * 70)
    print("🎬 MULTI-GPU SPEECH-TO-VIDEO SYSTEM DEMONSTRATION")
    print("=" * 70)
    print("This demo will generate a video using the test files:")
    print(f"  📸 Image: /workspace/wan22-comfy-project/iphone.jpeg")
    print(f"  🎵 Audio: /workspace/wan22-comfy-project/tmp_19iifpd.mp3")
    print("=" * 70)
    print()

def check_files():
    """Check if test files exist"""
    image_path = "/workspace/wan22-comfy-project/iphone.jpeg"
    audio_path = "/workspace/wan22-comfy-project/tmp_19iifpd.mp3"
    
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return False, None, None
    
    if not os.path.exists(audio_path):
        print(f"❌ Test audio not found: {audio_path}")
        return False, None, None
    
    print(f"✅ Test image found: {image_path}")
    print(f"✅ Test audio found: {audio_path}")
    return True, image_path, audio_path

def run_demonstration(quality="medium", gpus=None):
    """Run the demonstration"""
    print(f"\n🚀 Starting demonstration with quality: {quality}")
    
    # Check test files
    files_ok, image_path, audio_path = check_files()
    if not files_ok:
        return False
    
    # Prepare command
    cmd_parts = [
        "python3", "main_s2v_system.py",
        "--image", image_path,
        "--audio", audio_path,
        "--prompt", "A person speaking into their iPhone with enthusiasm and energy",
        "--output", "demo_generation",
        "--quality", quality,
        "--enhance-quality",
        "--log-level", "INFO"
    ]
    
    if gpus:
        cmd_parts.extend(["--gpus", str(gpus)])
    
    command = " ".join(cmd_parts)
    
    print(f"\n📋 Command to execute:")
    print(f"   {command}")
    print(f"\n⏱️  Estimated time: {get_estimated_time(quality)} minutes")
    
    # Ask for confirmation
    response = input("\n❓ Proceed with demonstration? (y/n): ").lower().strip()
    if response != 'y':
        print("❌ Demonstration cancelled.")
        return False
    
    print(f"\n🔄 Running demonstration...")
    print("   (This may take several minutes depending on your hardware)")
    
    # Execute command
    start_time = time.time()
    exit_code = os.system(command)
    end_time = time.time()
    
    if exit_code == 0:
        print(f"\n🎉 SUCCESS! Demonstration completed in {end_time - start_time:.1f} seconds")
        
        # Check for output file
        output_files = list(Path("outputs").glob("*demo_generation*"))
        if output_files:
            latest_output = max(output_files, key=os.path.getctime)
            file_size = os.path.getsize(latest_output) / (1024 * 1024)
            print(f"📹 Output video: {latest_output} ({file_size:.1f} MB)")
            print(f"   You can now play the generated video!")
        else:
            print("⚠️  Output video not found in expected location")
        
        return True
    else:
        print(f"\n❌ Demonstration failed with exit code: {exit_code}")
        print("   Check the logs for error details")
        return False

def get_estimated_time(quality):
    """Get estimated processing time based on quality"""
    times = {
        "low": 1.0,
        "medium": 3.0,
        "high": 6.0
    }
    return times.get(quality, 3.0)

def show_system_info():
    """Show system information"""
    print("\n🖥️  SYSTEM INFORMATION:")
    print("-" * 30)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        else:
            print("⚠️  No GPUs detected - performance will be very slow")
    except ImportError:
        print("PyTorch not available")
    
    try:
        import ray
        print(f"Ray version: {ray.__version__}")
    except ImportError:
        print("Ray not available")
    
    print()

def show_file_info():
    """Show information about test files"""
    print("\n📁 TEST FILE INFORMATION:")
    print("-" * 30)
    
    files_to_check = [
        "/workspace/wan22-comfy-project/iphone.jpeg",
        "/workspace/wan22-comfy-project/tmp_19iifpd.mp3"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✅ {os.path.basename(file_path)}: {size_mb:.2f} MB")
        else:
            print(f"❌ {os.path.basename(file_path)}: Not found")
    
    print()

def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(
        description="Multi-GPU S2V System Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic demonstration
  python demo.py
  
  # High quality with specific GPU count
  python demo.py --quality high --gpus 2
  
  # Show system info only
  python demo.py --info-only
        """
    )
    
    parser.add_argument("--quality", type=str, default="medium",
                       choices=["low", "medium", "high"],
                       help="Generation quality (default: medium)")
    parser.add_argument("--gpus", type=int, default=None,
                       help="Number of GPUs to use (auto-detect if not specified)")
    parser.add_argument("--info-only", action="store_true",
                       help="Show system information only, don't run demo")
    
    args = parser.parse_args()
    
    print_header()
    show_system_info()
    show_file_info()
    
    if args.info_only:
        print("ℹ️  Information display complete. Use without --info-only to run the demo.")
        return 0
    
    # Check if we're in the right directory
    if not os.path.exists("main_s2v_system.py"):
        print("❌ Error: main_s2v_system.py not found.")
        print("   Please run this demo from the project root directory.")
        return 1
    
    # Run demonstration
    success = run_demonstration(args.quality, args.gpus)
    
    if success:
        print("\n" + "=" * 70)
        print("🎊 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("Next steps:")
        print("  1. Check the generated video in the outputs/ directory")
        print("  2. Try different quality settings (low, medium, high)")
        print("  3. Test with your own images and audio files")
        print("  4. Read the documentation for advanced usage")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("❌ DEMONSTRATION FAILED")
        print("=" * 70)
        print("Troubleshooting:")
        print("  1. Check that all dependencies are installed (run ./setup.sh)")
        print("  2. Ensure model checkpoints are available")
        print("  3. Verify GPU drivers and CUDA installation")
        print("  4. Check log files for detailed error information")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)