#!/usr/bin/env python3
"""Test script to verify the setup"""

import sys
import torch
import ray
import numpy as np
import cv2
import librosa
from PIL import Image

def test_pytorch():
    print("Testing PyTorch...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    return torch.cuda.is_available()

def test_ray():
    print("\nTesting Ray...")
    try:
        ray.init(num_cpus=2, num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0)
        print(f"Ray version: {ray.__version__}")
        print("Ray initialized successfully")
        ray.shutdown()
        return True
    except Exception as e:
        print(f"Ray test failed: {e}")
        return False

def test_dependencies():
    print("\nTesting dependencies...")
    deps = {
        'NumPy': np.__version__,
        'OpenCV': cv2.__version__,
        'Librosa': librosa.__version__,
        'PIL': Image.__version__
    }
    
    for name, version in deps.items():
        print(f"  {name}: {version}")
    
    return True

def test_wan_import():
    print("\nTesting Wan2.2 import...")
    try:
        sys.path.append('Wan2.2')
        from wan.speech2video import WanS2V
        print("Wan2.2 import successful")
        return True
    except Exception as e:
        print(f"Wan2.2 import failed: {e}")
        return False

if __name__ == "__main__":
    print("="*50)
    print("MULTI-GPU S2V SYSTEM - SETUP VERIFICATION")
    print("="*50)
    
    tests = [
        ("PyTorch", test_pytorch),
        ("Ray", test_ray),
        ("Dependencies", test_dependencies),
        ("Wan2.2", test_wan_import)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✓ {test_name} test passed")
                passed += 1
            else:
                print(f"✗ {test_name} test failed")
        except Exception as e:
            print(f"✗ {test_name} test failed: {e}")
    
    print("\n" + "="*50)
    print(f"SETUP VERIFICATION COMPLETE: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Setup successful! The system is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("="*50)
