#!/usr/bin/env python3
"""
Simple Cache Setup for S2V System
Run this to set up external cache storage to save workspace space
"""

import os
import sys
from pathlib import Path

def setup_cache_simple():
    """Simple cache setup function"""
    print("🗂️  Setting up external cache system for S2V...")
    
    try:
        # Import and setup cache manager
        from cache_manager import setup_cache_system
        
        cache_manager = setup_cache_system()
        
        print(f"\n✅ Cache system ready!")
        print(f"📁 Cache location: /home/caches/wan2_s2v_system")
        print(f"💾 This will save space in your workspace")
        
        # Show cache stats
        stats = cache_manager.get_cache_stats()
        if '_total' in stats:
            total = stats['_total']
            print(f"📊 Current cache size: {total['total_size_gb']:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Cache setup failed: {e}")
        print("⚠️  Will use local storage instead")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("EXTERNAL CACHE SETUP")
    print("=" * 50)
    
    success = setup_cache_simple()
    
    if success:
        print("\n🎉 Cache system configured successfully!")
        print("\nBenefits:")
        print("  • Saves workspace storage space")
        print("  • Persistent cache across sessions") 
        print("  • Faster model loading after first use")
        print("  • Better memory management")
    else:
        print("\n⚠️  Using local storage - watch disk space!")
    
    print("=" * 50)