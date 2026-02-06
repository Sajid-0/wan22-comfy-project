#!/usr/bin/env python3
"""
Quick Setup Test - Check if the system is ready with existing venv
"""

import sys
import os
from pathlib import Path

def print_status(message):
    print(f"ℹ️  {message}")

def print_success(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def test_venv():
    """Test if we're using the correct virtual environment"""
    venv_path = "/workspace/wan22-comfy-project/venv"
    current_python = sys.executable
    
    print_status(f"Current Python executable: {current_python}")
    
    if venv_path in current_python:
        print_success(f"Using correct virtual environment: {venv_path}")
        return True
    else:
        print_error(f"Not using the expected venv. Expected: {venv_path}")
        print_status("Please activate the venv: source /workspace/wan22-comfy-project/venv/bin/activate")
        return False

def test_cache_dirs():
    """Test if cache directories are accessible"""
    cache_dirs = [
        "/home/caches",
        "/home/caches/temp",
        "/home/caches/huggingface", 
        "/home/caches/torch",
        "/home/caches/transformers",
        "/home/caches/ray_temp"
    ]
    
    all_good = True
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir) and os.access(cache_dir, os.W_OK):
            print_success(f"Cache directory ready: {cache_dir}")
        else:
            print_error(f"Cache directory not accessible: {cache_dir}")
            all_good = False
    
    return all_good

def test_essential_packages():
    """Test essential packages"""
    packages = {
        'torch': 'PyTorch',
        'ray': 'Ray', 
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'librosa': 'Librosa'
    }
    
    missing = []
    for package, name in packages.items():
        try:
            __import__(package)
            print_success(f"{name} available")
        except ImportError:
            print_error(f"{name} missing - install with: pip install {package}")
            missing.append(package)
    
    return len(missing) == 0

def test_gpu():
    """Test GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print_success(f"CUDA available with {gpu_count} GPU(s)")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print_status(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
            return True
        else:
            print_error("CUDA not available")
            return False
    except Exception as e:
        print_error(f"GPU test failed: {e}")
        return False

def test_wan_framework():
    """Test if Wan2.2 framework is accessible"""
    try:
        wan_path = Path("./Wan2.2")
        if wan_path.exists():
            print_success("Wan2.2 framework found")
            
            # Test if we can import
            sys.path.append(str(wan_path))
            try:
                from wan.configs import WAN_CONFIGS
                print_success("Wan2.2 imports successful")
                return True
            except ImportError as e:
                print_error(f"Wan2.2 import failed: {e}")
                return False
        else:
            print_error("Wan2.2 directory not found")
            return False
    except Exception as e:
        print_error(f"Wan2.2 test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("🚀 MULTI-GPU S2V SYSTEM - QUICK SETUP TEST")
    print("=" * 60)
    
    tests = [
        ("Virtual Environment", test_venv),
        ("Cache Directories", test_cache_dirs), 
        ("Essential Packages", test_essential_packages),
        ("GPU Support", test_gpu),
        ("Wan2.2 Framework", test_wan_framework)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                print_status("This test failed but setup can continue")
        except Exception as e:
            print_error(f"Test error: {e}")
    
    print("\n" + "=" * 60)
    print(f"SETUP TEST COMPLETE: {passed}/{total} tests passed")
    
    if passed >= 4:  # Allow for some flexibility
        print("🎉 System is ready to use!")
        print("\nNext steps:")
        print("1. Make sure model checkpoints are in ./checkpoints/")
        print("2. Run: python demo.py")
        print("3. Or use: python main_s2v_system.py --help")
    else:
        print("⚠️  Some issues found. Please check the errors above.")
    
    print("=" * 60)
    
    return 0 if passed >= 4 else 1

if __name__ == "__main__":
    exit(main())