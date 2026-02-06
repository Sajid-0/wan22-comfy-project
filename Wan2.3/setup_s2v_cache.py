#!/usr/bin/env python3
"""
Wan2.2 S2V Model Download and Cache Setup Script for RunPod GPU
Handles automatic model downloading and cache management for volatile storage environments
Designed for RunPod on-demand instances where /home gets reset on each restart
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
# Import the necessary library for direct downloading
from huggingface_hub import snapshot_download, HfFolder
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class WanS2VCacheManager:
    def __init__(self, cache_dir="/home/caches", workspace_dir="/workspace/wan22-comfy-project/Wan2.2"):
        self.cache_dir = Path(cache_dir)
        self.workspace_dir = Path(workspace_dir)
        self.venv_path = self.workspace_dir / "venv" / "bin" / "python"
        
        # Model configurations
        self.models = {
            "s2v-14b": {
                "name": "Wan2.2-S2V-14B",
                "hf_repo": "Wan-AI/Wan2.2-S2V-14B",
                "cache_path": self.cache_dir / "Wan2.2-S2V-14B",
                "required_files": [
                    "diffusion_pytorch_model-00001-of-00004.safetensors",
                    "diffusion_pytorch_model-00002-of-00004.safetensors", 
                    "diffusion_pytorch_model-00003-of-00004.safetensors",
                    "diffusion_pytorch_model-00004-of-00004.safetensors",
                    "diffusion_pytorch_model.safetensors.index.json",
                    "models_t5_umt5-xxl-enc-bf16.pth",
                    "Wan2.1_VAE.pth",
                    "config.json",
                    "configuration.json"
                ]
            }
        }
    
    def check_system_requirements(self):
        """Check if all system requirements are met"""
        logger.info("🔍 Checking system requirements...")
        
        # Check if we're in RunPod environment
        if not os.path.exists("/workspace"):
            logger.warning("⚠️  Not detected as RunPod environment (/workspace not found)")
        else:
            logger.info("✅ RunPod environment detected")
        
        # Check venv
        if not self.venv_path.exists():
            logger.error(f"❌ Virtual environment not found at {self.venv_path}")
            return False

        # Check for Hugging Face token
        if not HfFolder.get_token():
            logger.warning("⚠️  Hugging Face token not found.")
            logger.info("   Trying to set token automatically...")
            # Try to set a default token if available
            try:
                from huggingface_hub import login
                # You can set your token here or pass it as environment variable
                token = os.getenv('HF_TOKEN', 'your_token_here')
                if token and token != 'your_token_here':
                    login(token=token)
                    logger.info("✅ Hugging Face token set successfully")
                else:
                    logger.error("❌ No valid Hugging Face token found")
                    logger.info("   Please set HF_TOKEN environment variable or login manually")
                    return False
            except Exception as e:
                logger.error(f"❌ Failed to set Hugging Face token: {e}")
                return False
        
        logger.info("✅ System requirements check passed")
        return True
    
    def setup_cache_directories(self):
        """Create necessary cache directories"""
        logger.info("📁 Setting up cache directories...")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for model_key, model_info in self.models.items():
            model_info["cache_path"].mkdir(parents=True, exist_ok=True)
        logger.info("✅ Cache directories are ready.")
    
    def install_requirements(self):
        """Install necessary Python packages for model downloading"""
        logger.info("📦 Installing required packages...")
        packages = ["huggingface_hub[cli]", "safetensors"] # Only essential packages for download
        for package in packages:
            try:
                subprocess.run(
                    [str(self.venv_path), "-m", "pip", "install", "-U", package], 
                    check=True, capture_output=True, text=True
                )
                logger.info(f"✅ Installed/updated {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {package}: {e.stderr}")
                return False
        return True
    
    def check_model_integrity(self, model_key):
        """Check if model files are complete and valid"""
        model_info = self.models[model_key]
        cache_path = model_info["cache_path"]
        
        if not cache_path.exists(): return False
        
        missing_files = [f for f in model_info["required_files"] if not (cache_path / f).exists()]
        if missing_files:
            logger.warning(f"🔍 Integrity check: Missing files for {model_info['name']}: {missing_files}")
            return False
            
        logger.info(f"✅ Integrity check passed for {model_info['name']}")
        return True
    
    def download_model(self, model_key, force_redownload=False):
        """Download model from HuggingFace using the huggingface_hub library."""
        model_info = self.models[model_key]
        
        if not force_redownload and self.check_model_integrity(model_key):
            logger.info(f"✅ Model {model_info['name']} already exists and is complete. Skipping download.")
            return True
        
        logger.info(f"⬇️  Downloading {model_info['name']} from {model_info['hf_repo']}...")
        
        try:
            # Use snapshot_download with proper parameters
            snapshot_download(
                repo_id=model_info["hf_repo"],
                local_dir=model_info["cache_path"],
                local_dir_use_symlinks=False,
                resume_download=True,
                allow_patterns=["*.json", "*.safetensors", "*.pth"], # Main model files
                token=HfFolder.get_token()  # Use the stored token
            )
            logger.info(f"✅ Download completed for {model_info['name']}. Verifying files...")
            
            # Verify the download
            if self.check_model_integrity(model_key):
                logger.info(f"🎉 Successfully downloaded and verified {model_info['name']}.")
                return True
            else:
                logger.error("❌ Download completed, but integrity check failed.")
                return False

        except GatedRepoError:
            logger.error(f"❌ GATED REPOSITORY: You must accept the terms of use on the model page.")
            logger.error(f"   Please visit https://huggingface.co/{model_info['hf_repo']} to accept the agreement.")
            return False
        except HfHubHTTPError as e:
            logger.error(f"❌ HTTP Error downloading model: {e}")
            logger.error("   This could be a network issue or if the repository is private/unavailable.")
            return False
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred during download: {e}", exc_info=True)
            return False

    def run_full_setup(self):
        """Run complete setup process - optimized for RunPod on-demand instances"""
        logger.info("🚀 Starting full Wan2.2 S2V setup for RunPod...")
        logger.info("📝 Note: This will download ~43GB of model files to /home/caches")
        logger.info("💡 On RunPod, /home is volatile and will be reset when instance stops")
        
        if not self.check_system_requirements(): 
            return False
            
        self.setup_cache_directories()
        
        if not self.install_requirements(): 
            return False

        # Always check and download models (since /home is volatile on RunPod)
        logger.info("🔄 Checking model availability (RunPod /home is volatile)...")
        all_models_downloaded = True
        
        for model_key in self.models.keys():
            if not self.check_model_integrity(model_key):
                logger.info(f"📥 Model {self.models[model_key]['name']} not found or incomplete, downloading...")
                if not self.download_model(model_key):
                    all_models_downloaded = False
            else:
                logger.info(f"✅ Model {self.models[model_key]['name']} already available")

        self.display_status()
        
        if all_models_downloaded:
            logger.info("🎉 All models are set up successfully!")
            logger.info("💡 Ready to run: python run_s2v_multi_gpu.py")
        else:
            logger.error("❌ Model setup failed. Please check the logs.")
            
        return all_models_downloaded

    def quick_check_and_setup(self):
        """Quick check and setup - ideal for calling before generation"""
        logger.info("⚡ Quick setup check for S2V models...")
        
        # Quick system check
        if not self.venv_path.exists():
            logger.error(f"❌ Virtual environment not found at {self.venv_path}")
            return False
            
        # Set up token if needed
        if not HfFolder.get_token():
            try:
                from huggingface_hub import login
                token = os.getenv('HF_TOKEN', 'your_token_here')
                if token and token != 'your_token_here':
                    login(token=token)
                    logger.info("✅ HF token set")
            except:
                logger.error("❌ No HF token available")
                return False
        
        # Create cache directories
        self.setup_cache_directories()
        
        # Check if models exist, download if missing
        for model_key in self.models.keys():
            if not self.check_model_integrity(model_key):
                logger.info(f"📥 Downloading missing model: {self.models[model_key]['name']}")
                if not self.download_model(model_key):
                    logger.error(f"❌ Failed to download {self.models[model_key]['name']}")
                    return False
            else:
                logger.info(f"✅ Model ready: {self.models[model_key]['name']}")
        
        logger.info("🎉 Quick setup completed successfully!")
        return True

    def display_status(self):
        """Display current cache status"""
        logger.info("\n" + "="*20 + " CACHE STATUS " + "="*20)
        logger.info(f"Cache Directory: {self.cache_dir}")
        for model_key, model_info in self.models.items():
            is_complete = self.check_model_integrity(model_key)
            status = "✅ Complete" if is_complete else "❌ Incomplete"
            
            size_gb = 0
            if model_info["cache_path"].exists():
                total_size = sum(f.stat().st_size for f in model_info["cache_path"].rglob("*") if f.is_file())
                size_gb = total_size / (1024**3)
            logger.info(f"  - {model_info['name']}: {status} ({size_gb:.2f} GB)")
            
        logger.info("="*54)
        
        if all(self.check_model_integrity(key) for key in self.models.keys()):
            logger.info("🎉 All models are ready! You can now run:")
            logger.info("   cd /workspace/wan22-comfy-project/Wan2.2")
            logger.info("   python run_s2v_multi_gpu.py")
        else:
            logger.info("⚠️  Some models are missing. Run setup or download to get them.")

def main():
    parser = argparse.ArgumentParser(description="Wan2.2 S2V Cache and Model Manager for RunPod")
    parser.add_argument("action", nargs="?", default="interactive", 
                       choices=["setup", "download", "status", "quick", "interactive"], 
                       help="Action to perform.")
    parser.add_argument("--force-redownload", action="store_true", 
                       help="Force redownload even if files exist")
    args = parser.parse_args()
    
    manager = WanS2VCacheManager()

    try:
        if args.action == "status":
            manager.display_status()
        elif args.action == "download":
            manager.download_model("s2v-14b", force_redownload=args.force_redownload)
            manager.display_status()
        elif args.action == "setup":
            success = manager.run_full_setup()
            sys.exit(0 if success else 1)
        elif args.action == "quick":
            success = manager.quick_check_and_setup()
            sys.exit(0 if success else 1)
        else: # Interactive mode
            print("\n" + "="*50)
            print("    Wan2.2 S2V Cache Manager for RunPod")
            print("="*50)
            print("\nRunPod Note: /home is volatile and resets on instance restart")
            print("This script will auto-download models when needed (~43GB)")
            print("\nChoose an option:")
            print("  1. Full Setup (Recommended for first time)")
            print("  2. Quick Check & Setup (Fast, only downloads if missing)")
            print("  3. Download Models Only")
            print("  4. Show Cache Status")
            print("  5. Exit")
            choice = input("\nEnter your choice (1-5): ").strip()

            if choice == "1": 
                success = manager.run_full_setup()
                sys.exit(0 if success else 1)
            elif choice == "2":
                success = manager.quick_check_and_setup()
                sys.exit(0 if success else 1)
            elif choice == "3":
                force = input("Force redownload even if files exist? (y/N): ").strip().lower() == 'y'
                success = manager.download_model("s2v-14b", force_redownload=force)
                manager.display_status()
                sys.exit(0 if success else 1)
            elif choice == "4": 
                manager.display_status()
            else: 
                print("Exiting.")
                sys.exit(0)

    except KeyboardInterrupt:
        logger.info("\n❌ Operation interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Add argparse import here to keep the script self-contained
    import argparse
    main()