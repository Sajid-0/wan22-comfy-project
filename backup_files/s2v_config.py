# Multi-GPU Speech-to-Video Generation System Configuration

import os
from pathlib import Path

class S2VConfig:
    """Configuration class for Multi-GPU S2V System"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    WAN_PATH = PROJECT_ROOT / "Wan2.2"
    
    # Cache-aware paths - will use external cache if available
    CACHE_BASE_DIR = Path(os.getenv('S2V_CACHE_DIR', '/home/caches'))
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"  # Keep checkpoints in project for now
    VENV_PATH = Path("/workspace/wan22-comfy-project/venv")  # Use existing venv
    TEMP_DIR = Path(os.getenv('S2V_TEMP_DIR', str(CACHE_BASE_DIR / 'temp')))
    
    # Ensure cache directories exist
    @classmethod
    def ensure_cache_dirs(cls):
        """Ensure cache directories exist"""
        cls.CACHE_BASE_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Model configurations
    SUPPORTED_MODELS = {
        "s2v-14B": {
            "config_name": "s2v-14B",
            "description": "Speech-to-Video 14B parameter model",
            "min_memory_gb": 16,
            "recommended_gpus": 2
        }
    }
    
    # Video generation defaults
    DEFAULT_VIDEO_PARAMS = {
        "height": 480,
        "width": 640,
        "fps": 30,
        "quality": "medium",
        "guidance_scale": 3.0,
        "num_inference_steps": 15
    }
    
    # Quality presets
    QUALITY_PRESETS = {
        "low": {
            "guidance_scale": 2.0,
            "num_inference_steps": 8,
            "description": "Fastest generation, lower quality"
        },
        "medium": {
            "guidance_scale": 3.0,
            "num_inference_steps": 15,
            "description": "Balanced speed and quality"
        },
        "high": {
            "guidance_scale": 4.0,
            "num_inference_steps": 25,
            "description": "Best quality, slower generation"
        }
    }
    
    # GPU configuration
    GPU_CONFIG = {
        "min_memory_gb": 8,
        "recommended_memory_gb": 16,
        "max_batch_size": 4,
        "memory_fraction": 0.9
    }
    
    # Ray configuration - use external cache for temp files
    RAY_CONFIG = {
        "object_store_memory": 2000000000,  # 2GB
        "num_cpus": None,  # Auto-detect
        "num_gpus": None,  # Auto-detect
        "dashboard_host": "0.0.0.0",
        "dashboard_port": 8265,
        "_temp_dir": str(CACHE_BASE_DIR / 'ray_tmp')  # Use external cache for Ray temp
    }
    
    # Audio processing
    AUDIO_CONFIG = {
        "sample_rate": 16000,
        "target_fps": 30,
        "feature_dim": 1024,
        "window_size": 0.025,  # 25ms
        "hop_length": 0.01     # 10ms
    }
    
    # Image processing
    IMAGE_CONFIG = {
        "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
        "max_size": (1920, 1080),
        "min_size": (256, 256),
        "aspect_ratio_tolerance": 0.1
    }
    
    # Logging
    LOG_CONFIG = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "multi_gpu_s2v.log",
        "max_size_mb": 100,
        "backup_count": 5
    }
    
    # Performance monitoring
    MONITORING_CONFIG = {
        "gpu_check_interval": 5,  # seconds
        "memory_warning_threshold": 0.9,  # 90% memory usage
        "enable_profiling": False,
        "stats_output_file": "generation_stats.json"
    }
    
    @classmethod
    def validate_environment(cls):
        """Validate the environment setup"""
        issues = []
        
        # Check paths
        if not cls.WAN_PATH.exists():
            issues.append(f"Wan2.2 directory not found: {cls.WAN_PATH}")
        
        if not cls.VENV_PATH.exists():
            issues.append(f"Virtual environment not found: {cls.VENV_PATH}")
        
        # Check CUDA availability
        try:
            import torch
            if not torch.cuda.is_available():
                issues.append("CUDA not available")
            elif torch.cuda.device_count() == 0:
                issues.append("No CUDA devices found")
        except ImportError:
            issues.append("PyTorch not installed")
        
        # Check Ray availability
        try:
            import ray
        except ImportError:
            issues.append("Ray not installed")
        
        return issues
    
    @classmethod
    def get_optimal_gpu_config(cls, available_gpus: int, memory_per_gpu: float):
        """Get optimal GPU configuration based on available resources"""
        config = {
            "num_workers": min(available_gpus, 4),  # Max 4 workers for efficiency
            "frames_per_worker": 25,
            "batch_size": 1 if memory_per_gpu < 12 else 2,
            "enable_memory_optimization": memory_per_gpu < 16
        }
        
        if memory_per_gpu < cls.GPU_CONFIG["min_memory_gb"]:
            config["warnings"] = [f"GPU memory ({memory_per_gpu:.1f}GB) below minimum requirement ({cls.GPU_CONFIG['min_memory_gb']}GB)"]
        
        return config