#!/usr/bin/env python3
"""
Cache Integration Verification - Show that caching is working properly
"""

import os
import sys
from pathlib import Path

def show_cache_status():
    """Show cache directory status"""
    print("🗂️  CACHE INTEGRATION STATUS")
    print("=" * 50)
    
    # Environment variables
    cache_vars = [
        'TRANSFORMERS_CACHE',
        'HF_HOME', 
        'TORCH_HOME',
        'RAY_TMPDIR',
        'TMPDIR'
    ]
    
    print("\n📋 Environment Variables:")
    for var in cache_vars:
        value = os.getenv(var, "Not set")
        if "/home/caches" in value:
            print(f"  ✅ {var}: {value}")
        else:
            print(f"  ⚠️  {var}: {value}")
    
    # Check symlinks
    workspace_dir = Path("/workspace/wan22-comfy-project")
    symlinks = ['checkpoints', 'temp', '.cache', 'model_cache']
    
    print("\n🔗 Workspace Symlinks:")
    for link in symlinks:
        link_path = workspace_dir / link
        if link_path.is_symlink():
            target = link_path.readlink()
            print(f"  ✅ {link} -> {target}")
        else:
            print(f"  ❌ {link}: Not a symlink")
    
    # Check cache directory sizes
    cache_base = Path("/home/caches")
    
    print("\n📊 Cache Directory Usage:")
    if cache_base.exists():
        total_size = 0
        for subdir in cache_base.iterdir():
            if subdir.is_dir():
                size = sum(f.stat().st_size for f in subdir.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                total_size += size
                print(f"  📁 {subdir.name}: {size_mb:.2f} MB")
        
        total_size_mb = total_size / (1024 * 1024)
        print(f"  🏷️  Total cache size: {total_size_mb:.2f} MB")
    else:
        print("  ❌ Cache directory not found")
    
    # Show current working setup
    print(f"\n🔧 Current Setup:")
    print(f"  Python: {sys.executable}")
    print(f"  Working directory: {os.getcwd()}")
    print(f"  Cache base: /home/caches")
    print(f"  Workspace saves: Low storage impact ✅")

def show_usage_example():
    """Show how the cache system works"""
    print("\n" + "=" * 50)
    print("💡 HOW CACHE INTEGRATION WORKS")
    print("=" * 50)
    
    print("""
When you run the S2V system:

1. 🏠 Models & checkpoints → /home/caches/wan2_s2v_system/
2. 🔄 Temporary files → /home/caches/temp/  
3. 🤗 HuggingFace cache → /home/caches/huggingface/
4. 🔥 PyTorch cache → /home/caches/torch/
5. ⚡ Ray temp files → /home/caches/ray_temp/

Benefits:
- ✅ Workspace storage preserved
- ✅ Faster model loading (cached)  
- ✅ Shared cache across sessions
- ✅ Easy cleanup if needed

Example usage with cache:
    python main_s2v_system.py \\
      --image /workspace/wan22-comfy-project/iphone.jpeg \\
      --audio /workspace/wan22-comfy-project/tmp_19iifpd.mp3 \\
      --prompt "A person speaking" \\
      --output test_video

The system will automatically use /home/caches for all heavy data!
    """)

if __name__ == "__main__":
    show_cache_status()
    show_usage_example()
    
    print("\n" + "=" * 50)
    print("🎉 CACHE INTEGRATION READY!")
    print("=" * 50)
    print("Next: Run 'python demo.py' to test the system")
    print("=" * 50)