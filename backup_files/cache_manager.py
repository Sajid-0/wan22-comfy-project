#!/usr/bin/env python3
"""
Cache Management System for Multi-GPU S2V
Handles efficient cache storage on external directories to save workspace storage
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, List
import json
import hashlib
import time

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages external cache storage for the S2V system"""
    
    def __init__(self, cache_base_dir: str = "/home/caches"):
        """
        Initialize cache manager
        
        Args:
            cache_base_dir: Base directory for all caches
        """
        self.cache_base_dir = Path(cache_base_dir)
        self.s2v_cache_dir = self.cache_base_dir / "wan2_s2v_system"
        
        # Create cache structure
        self.setup_cache_directories()
        
        # Cache subdirectories
        self.model_cache = self.s2v_cache_dir / "models"
        self.checkpoint_cache = self.s2v_cache_dir / "checkpoints"
        self.temp_cache = self.s2v_cache_dir / "temp"
        self.huggingface_cache = self.s2v_cache_dir / "huggingface"
        self.ray_cache = self.s2v_cache_dir / "ray_tmp"
        self.torch_cache = self.s2v_cache_dir / "torch"
        
        logger.info(f"Cache manager initialized: {self.s2v_cache_dir}")
    
    def setup_cache_directories(self):
        """Create all necessary cache directories"""
        try:
            # Create main cache directory
            self.s2v_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            subdirs = [
                "models",           # Model files and weights
                "checkpoints",      # Wan2.2 checkpoints
                "temp",            # Temporary processing files
                "huggingface",     # HuggingFace model cache
                "ray_tmp",         # Ray temporary files
                "torch",           # PyTorch cache
                "audio_temp",      # Audio processing temp
                "video_temp",      # Video processing temp
                "logs"             # Cache-related logs
            ]
            
            for subdir in subdirs:
                (self.s2v_cache_dir / subdir).mkdir(exist_ok=True)
            
            # Create cache info file
            self.create_cache_info()
            
            logger.info(f"Cache directories created successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup cache directories: {e}")
            raise
    
    def create_cache_info(self):
        """Create cache information file"""
        cache_info = {
            "created": time.time(),
            "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system": "Multi-GPU S2V System",
            "version": "1.0.0",
            "cache_structure": {
                "models": "Model files and weights",
                "checkpoints": "Wan2.2 model checkpoints", 
                "temp": "Temporary processing files",
                "huggingface": "HuggingFace transformers cache",
                "ray_tmp": "Ray distributed computing temp files",
                "torch": "PyTorch cache and temp files",
                "audio_temp": "Audio processing temporary files",
                "video_temp": "Video processing temporary files"
            }
        }
        
        info_file = self.s2v_cache_dir / "cache_info.json"
        with open(info_file, 'w') as f:
            json.dump(cache_info, f, indent=2)
    
    def setup_environment_variables(self):
        """Set up environment variables to use cache directories"""
        env_vars = {
            # HuggingFace cache
            'TRANSFORMERS_CACHE': str(self.huggingface_cache),
            'HF_HOME': str(self.huggingface_cache),
            'HF_DATASETS_CACHE': str(self.huggingface_cache / "datasets"),
            
            # PyTorch cache
            'TORCH_HOME': str(self.torch_cache),
            'TORCH_CACHE_DIR': str(self.torch_cache),
            
            # Ray temporary directory
            'RAY_TMPDIR': str(self.ray_cache),
            
            # General temp directory
            'TMPDIR': str(self.temp_cache),
            'TEMP': str(self.temp_cache),
            'TMP': str(self.temp_cache),
            
            # Custom S2V cache paths
            'S2V_CACHE_DIR': str(self.s2v_cache_dir),
            'S2V_MODEL_CACHE': str(self.model_cache),
            'S2V_CHECKPOINT_CACHE': str(self.checkpoint_cache)
        }
        
        # Set environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
        
        logger.info(f"Environment variables configured for external cache")
        return env_vars
    
    def get_cache_stats(self) -> Dict:
        """Get cache directory statistics"""
        try:
            stats = {}
            
            for subdir in self.s2v_cache_dir.iterdir():
                if subdir.is_dir():
                    size = self.get_directory_size(subdir)
                    file_count = len(list(subdir.rglob("*"))) if subdir.exists() else 0
                    
                    stats[subdir.name] = {
                        'size_mb': size / (1024 * 1024),
                        'size_gb': size / (1024 * 1024 * 1024),
                        'file_count': file_count,
                        'path': str(subdir)
                    }
            
            # Total statistics
            total_size = sum(stat['size_mb'] for stat in stats.values())
            total_files = sum(stat['file_count'] for stat in stats.values())
            
            stats['_total'] = {
                'total_size_mb': total_size,
                'total_size_gb': total_size / 1024,
                'total_files': total_files,
                'cache_root': str(self.s2v_cache_dir)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'error': str(e)}
    
    def get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        try:
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Error calculating size for {path}: {e}")
        
        return total_size
    
    def clean_temp_cache(self, max_age_hours: int = 24):
        """Clean temporary cache files older than specified hours"""
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            temp_dirs = [self.temp_cache, self.s2v_cache_dir / "audio_temp", 
                        self.s2v_cache_dir / "video_temp"]
            
            cleaned_count = 0
            cleaned_size = 0
            
            for temp_dir in temp_dirs:
                if not temp_dir.exists():
                    continue
                
                for file_path in temp_dir.rglob("*"):
                    if file_path.is_file():
                        try:
                            file_age = current_time - file_path.stat().st_mtime
                            if file_age > max_age_seconds:
                                file_size = file_path.stat().st_size
                                file_path.unlink()
                                cleaned_count += 1
                                cleaned_size += file_size
                        except Exception as e:
                            logger.warning(f"Could not clean {file_path}: {e}")
            
            logger.info(f"Cleaned {cleaned_count} files, freed {cleaned_size / (1024*1024):.1f} MB")
            return {'files_cleaned': cleaned_count, 'size_freed_mb': cleaned_size / (1024*1024)}
            
        except Exception as e:
            logger.error(f"Cache cleaning failed: {e}")
            return {'error': str(e)}
    
    def create_symlinks_in_workspace(self, workspace_dir: str = "/workspace/wan22-comfy-project"):
        """Create symlinks from workspace to cache directories"""
        try:
            workspace_path = Path(workspace_dir)
            
            # Symlink mappings
            symlinks = {
                'checkpoints': self.checkpoint_cache,
                'temp': self.temp_cache,
                '.cache': self.s2v_cache_dir / 'workspace_cache',
                'model_cache': self.model_cache
            }
            
            created_links = []
            
            for link_name, target_path in symlinks.items():
                link_path = workspace_path / link_name
                
                # Remove existing file/dir if it exists
                if link_path.exists():
                    if link_path.is_symlink():
                        link_path.unlink()
                    elif link_path.is_dir():
                        if not any(link_path.iterdir()):  # Only remove if empty
                            link_path.rmdir()
                    elif link_path.is_file():
                        link_path.unlink()
                
                # Create target directory if it doesn't exist
                target_path.mkdir(parents=True, exist_ok=True)
                
                # Create symlink
                if not link_path.exists():
                    link_path.symlink_to(target_path)
                    created_links.append(f"{link_name} -> {target_path}")
            
            logger.info(f"Created symlinks: {created_links}")
            return created_links
            
        except Exception as e:
            logger.error(f"Failed to create symlinks: {e}")
            return []
    
    def migrate_existing_data(self, workspace_dir: str = "/workspace/wan22-comfy-project"):
        """Migrate existing data from workspace to cache"""
        try:
            workspace_path = Path(workspace_dir)
            migrated = []
            
            # Data to migrate
            migration_map = {
                'checkpoints': self.checkpoint_cache,
                'temp': self.temp_cache,
                'outputs': self.s2v_cache_dir / 'outputs',  # Also cache outputs
            }
            
            for source_name, target_path in migration_map.items():
                source_path = workspace_path / source_name
                
                if source_path.exists() and source_path.is_dir():
                    # Create target directory
                    target_path.mkdir(parents=True, exist_ok=True)
                    
                    # Move contents
                    for item in source_path.iterdir():
                        target_item = target_path / item.name
                        if not target_item.exists():
                            shutil.move(str(item), str(target_item))
                            migrated.append(f"{item} -> {target_item}")
                    
                    # Remove source directory if empty
                    if not any(source_path.iterdir()):
                        source_path.rmdir()
            
            logger.info(f"Migrated data: {len(migrated)} items")
            return migrated
            
        except Exception as e:
            logger.error(f"Data migration failed: {e}")
            return []


def setup_cache_system():
    """Setup the complete cache system"""
    print("🗂️  Setting up external cache system...")
    
    try:
        # Initialize cache manager
        cache_manager = CacheManager()
        
        # Setup environment variables
        env_vars = cache_manager.setup_environment_variables()
        print(f"✅ Environment variables configured:")
        for key, value in env_vars.items():
            print(f"   {key}={value}")
        
        # Migrate existing data
        print("\n📦 Migrating existing data...")
        migrated = cache_manager.migrate_existing_data()
        if migrated:
            print(f"✅ Migrated {len(migrated)} items")
        else:
            print("ℹ️  No data to migrate")
        
        # Create symlinks
        print("\n🔗 Creating workspace symlinks...")
        symlinks = cache_manager.create_symlinks_in_workspace()
        if symlinks:
            print("✅ Created symlinks:")
            for link in symlinks:
                print(f"   {link}")
        
        # Get cache stats
        print("\n📊 Cache statistics:")
        stats = cache_manager.get_cache_stats()
        if '_total' in stats:
            total = stats['_total']
            print(f"   Cache directory: {total['cache_root']}")
            print(f"   Total size: {total['total_size_gb']:.2f} GB")
            print(f"   Total files: {total['total_files']}")
        
        print(f"\n🎉 Cache system setup complete!")
        print(f"📁 Cache location: {cache_manager.s2v_cache_dir}")
        
        return cache_manager
        
    except Exception as e:
        print(f"❌ Cache setup failed: {e}")
        raise


if __name__ == "__main__":
    setup_cache_system()