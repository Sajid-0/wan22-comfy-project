#!/usr/bin/env python3
"""
Gradio Interface for Wan2.2 Animate-14B
Simple and functional character animation interface with integrated cache management
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging first
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

logger.info("🚀 Starting Wan2.2 Animate Gradio App...")

# Import lightweight dependencies first
import gradio as gr
from huggingface_hub import snapshot_download, HfFolder
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

# Add the wan module to path
sys.path.insert(0, '/workspace/wan22-comfy-project/Wan2.2')

# Lazy imports for heavy dependencies (torch, model classes)
# These will be imported only when needed
WanAnimate = None
config = None
save_video = None
torch = None

def lazy_import_torch():
    """Lazy import torch and model dependencies"""
    global torch, WanAnimate, config, save_video
    if torch is None:
        logger.info("Loading torch and model classes (this may take a moment)...")
        import torch as torch_module
        from wan.animate import WanAnimate as WanAnimateClass
        from wan.configs.wan_animate_14B import animate_14B as model_config
        from wan.utils.utils import save_video as save_video_func
        
        torch = torch_module
        WanAnimate = WanAnimateClass
        config = model_config
        save_video = save_video_func
        logger.info("✅ Heavy dependencies loaded")

# Global model instance
model = None
CACHE_DIR = "/home/caches/Wan2.2-Animate-14B"
OUTPUT_DIR = "/workspace/wan22-comfy-project/outputs"
PREPROCESS_DIR = "/workspace/wan22-comfy-project/Wan2.2/preprocessed"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(PREPROCESS_DIR).mkdir(parents=True, exist_ok=True)

# ============= INTEGRATED CACHE MANAGER =============

class WanAnimateCacheManager:
    """Integrated cache manager for automatic model downloading"""
    
    def __init__(self):
        self.cache_dir = Path(CACHE_DIR)
        self.hf_repo = "Wan-AI/Wan2.2-Animate-14B"
        self.required_files = [
            "diffusion_pytorch_model-00001-of-00004.safetensors",
            "diffusion_pytorch_model-00002-of-00004.safetensors",
            "diffusion_pytorch_model-00003-of-00004.safetensors",
            "diffusion_pytorch_model-00004-of-00004.safetensors",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            "Wan2.1_VAE.pth",
            "config.json",
        ]
    
    def setup_hf_token(self):
        """Setup Hugging Face token"""
        if not HfFolder.get_token():
            try:
                from huggingface_hub import login
                token = os.getenv('HF_TOKEN', 'your_token_here')
                if token and token != 'your_token_here':
                    login(token=token, add_to_git_credential=False)
                    logger.info("✅ HF token set")
                    return True
            except Exception as e:
                logger.error(f"❌ Failed to set HF token: {e}")
                return False
        return True
    
    def check_model_integrity(self):
        """Check if required model files exist"""
        if not self.cache_dir.exists():
            return False
        
        missing = [f for f in self.required_files if not (self.cache_dir / f).exists()]
        if missing:
            logger.warning(f"Missing {len(missing)} files: {missing[:3]}")
            return False
        return True
    
    def download_models(self, progress_callback=None):
        """Download models from HuggingFace"""
        if self.check_model_integrity():
            logger.info("✅ Models already present")
            return True, "✅ Models already downloaded"
        
        if not self.setup_hf_token():
            return False, "❌ HuggingFace token not available"
        
        logger.info(f"⬇️ Downloading {self.hf_repo}... (~50GB)")
        
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            snapshot_download(
                repo_id=self.hf_repo,
                local_dir=self.cache_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                allow_patterns=["*.json", "*.safetensors", "*.pth", "*.pt", "*.bin"],
                token=HfFolder.get_token()
            )
            
            if self.check_model_integrity():
                logger.info("✅ Download complete and verified")
                return True, "✅ Models downloaded successfully"
            else:
                logger.warning("⚠️ Download complete but some files may be missing")
                return True, "⚠️ Core models downloaded (some optional files may be missing)"
                
        except GatedRepoError:
            msg = f"❌ Gated repo: Accept terms at https://huggingface.co/{self.hf_repo}"
            logger.error(msg)
            return False, msg
        except Exception as e:
            msg = f"❌ Download failed: {str(e)}"
            logger.error(msg)
            return False, msg

# Global cache manager
cache_manager = WanAnimateCacheManager()

# ============= END CACHE MANAGER =============

def download_preprocessing_checkpoints():
    """Download missing preprocessing checkpoints"""
    import urllib.request
    
    checkpoint_dir = Path(CACHE_DIR) / "process_checkpoint"
    det_dir = checkpoint_dir / "det"
    pose2d_dir = checkpoint_dir / "pose2d"
    
    # Create directories
    det_dir.mkdir(parents=True, exist_ok=True)
    pose2d_dir.mkdir(parents=True, exist_ok=True)
    
    yolo_path = det_dir / "yolov10m.onnx"
    vitpose_path = pose2d_dir / "vitpose_h_wholebody.onnx"
    
    # Download YOLO
    if not yolo_path.exists():
        print("⏬ Downloading YOLOv10m model...")
        urllib.request.urlretrieve(
            "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.onnx",
            str(yolo_path)
        )
        print(f"✅ Downloaded YOLO to {yolo_path}")
    
    # Download ViTPose - try multiple sources
    if not vitpose_path.exists():
        print("⏬ Downloading ViTPose model...")
        urls = [
            "https://huggingface.co/VikramSingh178/yolov10-vitpose-onnx/resolve/main/vitpose_h_wholebody.onnx",
            "https://github.com/ViTAE-Transformer/ViTPose/releases/download/v1.0/vitpose-h-wholebody.onnx",
        ]
        
        downloaded = False
        for url in urls:
            try:
                print(f"  Trying: {url}")
                urllib.request.urlretrieve(url, str(vitpose_path))
                if vitpose_path.exists() and vitpose_path.stat().st_size > 1000000:  # At least 1MB
                    print(f"✅ Downloaded ViTPose to {vitpose_path}")
                    downloaded = True
                    break
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        if not downloaded:
            return False, "❌ Failed to download ViTPose model. Please download manually from https://github.com/ViTAE-Transformer/ViTPose/releases"
    
    return True, f"✅ Preprocessing checkpoints ready:\n  - {yolo_path}\n  - {vitpose_path}"

def check_models():
    """Check if all required models are downloaded"""
    cache_path = Path(CACHE_DIR)
    
    # Check main model files
    required_files = [
        "diffusion_pytorch_model.safetensors.index.json",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth"
    ]
    
    for f in required_files:
        if not (cache_path / f).exists():
            return False
    
    # Check preprocessing checkpoints
    det_checkpoint = cache_path / "process_checkpoint" / "det" / "yolov10m.onnx"
    pose_checkpoint = cache_path / "process_checkpoint" / "pose2d" / "vitpose_h_wholebody.onnx"
    
    if not det_checkpoint.exists() or not pose_checkpoint.exists():
        # Try to download them
        success, msg = download_preprocessing_checkpoints()
        if not success:
            return False
    
    return True

def download_models(progress=gr.Progress()):
    """Download models with progress tracking"""
    if progress:
        progress(0.1, desc="Checking models...")
    
    success, msg = cache_manager.download_models()
    
    if progress:
        progress(1.0, desc="Complete")
    
    return success, msg

def preprocess_video(video_path, reference_image, mode="animate", progress=gr.Progress()):
    """Preprocess video using the preprocess pipeline"""
    
    if video_path is None or reference_image is None:
        return None, "❌ Please provide both video and reference image"
    
    # Handle File component - can be string path or file object
    if hasattr(video_path, 'name'):
        video_path = video_path.name
    elif isinstance(video_path, dict):
        video_path = video_path.get('name') or video_path.get('path')
    
    if not video_path or not os.path.exists(video_path):
        return None, f"❌ Video file not found: {video_path}"
    
    if not os.path.exists(reference_image):
        return None, f"❌ Reference image not found: {reference_image}"
    
    print(f"✓ Video: {video_path}")
    print(f"✓ Image: {reference_image}")
    
    # Create unique output directory
    import time
    timestamp = int(time.time())
    output_path = Path(PREPROCESS_DIR) / f"output_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    if progress:
        progress(0.1, desc="Starting preprocessing...")
    
    # Preprocessing command
    python_path = "/workspace/wan22-comfy-project/venv/bin/python"
    preprocess_script = "/workspace/wan22-comfy-project/Wan2.2/wan/modules/animate/preprocess/preprocess_data.py"
    ckpt_path = f"{CACHE_DIR}/process_checkpoint"
    
    cmd = [
        python_path,
        preprocess_script,
        "--ckpt_path", ckpt_path,
        "--video_path", video_path,
        "--refer_path", reference_image,
        "--save_path", str(output_path),
        "--resolution_area", "1280", "720",
    ]
    
    # Add mode-specific flags
    if mode == "animate":
        cmd.append("--retarget_flag")
    elif mode == "replace":
        cmd.append("--replace_flag")
    
    try:
        print(f"Running preprocessing command: {' '.join(cmd)}")
        
        if progress:
            progress(0.2, desc="Extracting pose and face data...")
        
        # Run with longer timeout and stream output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Wait for completion with progress updates
        stdout_lines = []
        stderr_lines = []
        
        start_time = time.time()
        timeout = 600  # 10 minutes for preprocessing
        
        while True:
            if progress and time.time() - start_time > 10:
                elapsed = int(time.time() - start_time)
                progress(0.5, desc=f"Processing... ({elapsed}s elapsed)")
            
            return_code = process.poll()
            if return_code is not None:
                # Process finished
                stdout_lines.extend(process.stdout.readlines())
                stderr_lines.extend(process.stderr.readlines())
                break
            
            if time.time() - start_time > timeout:
                process.kill()
                return None, f"❌ Preprocessing timed out after {timeout}s"
            
            time.sleep(2)
        
        stdout_text = ''.join(stdout_lines)
        stderr_text = ''.join(stderr_lines)
        
        if return_code != 0:
            return None, f"❌ Preprocessing failed with exit code {return_code}\n\nStderr: {stderr_text}\n\nStdout: {stdout_text}"
        
        if progress:
            progress(0.9, desc="Verifying output files...")
        
        # Check if output files were created
        required_files = ["src_pose.mp4", "src_face.mp4", "src_ref.png"]
        missing = [f for f in required_files if not (output_path / f).exists()]
        
        if missing:
            return None, f"❌ Preprocessing incomplete. Missing files: {missing}\n\nStderr: {stderr_text}\n\nStdout: {stdout_text}"
        
        if progress:
            progress(1.0, desc="Complete!")
        
        return str(output_path), f"✅ Preprocessing complete!\nOutput: {output_path}\n\nGenerated files: {', '.join(required_files)}"
    except Exception as e:
        return None, f"❌ Preprocessing failed: {str(e)}"

def load_model(use_relighting=False, progress=gr.Progress()):
    """Load the Wan Animate model"""
    global model
    
    # Lazy load heavy dependencies
    lazy_import_torch()
    
    if progress:
        progress(0.1, desc="Checking models...")
    
    # Check and download models if needed
    if not check_models():
        if progress:
            progress(0.2, desc="Downloading models...")
        success, msg = cache_manager.download_models()
        if not success:
            return f"❌ {msg}"
    
    if model is not None:
        return "✅ Model already loaded"
    
    try:
        if progress:
            progress(0.5, desc="Loading model... (this may take a few minutes)")
        
        # Check available GPUs
        num_gpus = torch.cuda.device_count()
        logger.info(f"🎮 Detected {num_gpus} GPU(s)")
        
        # Set memory optimization environment variables
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        logger.info("🔧 Enabled expandable_segments for better memory management")
        
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 Cleared GPU cache")
        
        # Initialize model with single GPU but aggressive memory optimization
        # Note: FSDP requires torchrun/multiprocessing, not compatible with Gradio's single process
        # For Gradio, we use single GPU with heavy CPU offloading
        model = WanAnimate(
            config=config,
            checkpoint_dir=CACHE_DIR,
            device_id=0,  # Primary GPU
            rank=0,
            t5_fsdp=False,  # Disable FSDP (requires multi-process)
            dit_fsdp=False,  # Disable FSDP (requires multi-process)
            use_sp=False,  # Disable SP (requires multi-process)
            t5_cpu=True,  # Offload T5 encoder to CPU to save VRAM
            init_on_cpu=False,
            convert_model_dtype=True,
            use_relighting_lora=use_relighting
        )
        
        logger.info(f"💡 Model loaded on GPU 0 with CPU offloading for memory optimization")
        logger.info(f"💡 For true multi-GPU, use: torchrun --nproc_per_node=2 run_animate_multi_gpu.py")
        
        if progress:
            progress(1.0, desc="Model loaded!")
        
        gpu_info = f" across {num_gpus} GPUs" if num_gpus > 1 else ""
        return f"✅ Model loaded successfully{gpu_info}!"
    except Exception as e:
        return f"❌ Failed to load model: {str(e)}"

def generate_animation(
    preprocessed_path,
    prompt,
    seed,
    num_frames,
    sampling_steps,
    guide_scale,
    use_replace_mode,
    progress=gr.Progress()
):
    """Generate animated video"""
    global model
    
    # Ensure torch is loaded
    lazy_import_torch()
    
    # Auto-load model if not already loaded
    if model is None:
        progress(0.05, desc="Model not loaded, loading now...")
        load_status = load_model(use_relighting=use_replace_mode, progress=progress)
        if "❌" in load_status or model is None:
            return None, f"❌ Failed to load model: {load_status}"
        progress(0.1, desc="Model loaded, starting generation...")
    
    if preprocessed_path is None or not Path(preprocessed_path).exists():
        return None, "❌ Please preprocess your video first"
    
    # Check required files
    required_files = ["src_pose.mp4", "src_face.mp4", "src_ref.png"]
    for f in required_files:
        if not (Path(preprocessed_path) / f).exists():
            return None, f"❌ Missing required file: {f}"
    
    if use_replace_mode:
        for f in ["src_bg.mp4", "src_mask.mp4"]:
            if not (Path(preprocessed_path) / f).exists():
                return None, f"❌ Replace mode requires {f}"
    
    progress(0.1, desc="Starting generation...")
    
    try:
        # Clear GPU cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 Cleared GPU cache before generation")
        
        # Set random seed
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        # Generate video
        progress(0.2, desc="Generating video...")
        
        video_tensor = model.generate(
            src_root_path=preprocessed_path,
            replace_flag=use_replace_mode,
            clip_len=num_frames,
            refert_num=1,
            shift=5.0,
            sample_solver='dpm++',
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            input_prompt=prompt,
            n_prompt="",
            seed=seed,
            offload_model=True
        )
        
        progress(0.9, desc="Saving video...")
        
        # Save video
        import time
        output_path = Path(OUTPUT_DIR) / f"animate_{int(time.time())}.mp4"
        save_video(video_tensor, str(output_path), fps=30)
        
        progress(1.0, desc="Done!")
        
        return str(output_path), f"✅ Video generated!\nSeed used: {seed}"
    
    except Exception as e:
        import traceback
        return None, f"❌ Generation failed: {str(e)}\n{traceback.format_exc()}"

# Gradio Interface
with gr.Blocks(title="Wan2.2 Animate") as app:
    gr.Markdown("# 🎭 Wan2.2 Animate-14B\nCharacter Animation & Replacement")
    
    with gr.Tab("1️⃣ Setup"):
        gr.Markdown("### Model Setup")
        gr.Markdown("First time will download ~50GB from [Wan-AI/Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B)")
        
        with gr.Row():
            download_only_btn = gr.Button("📥 Download Models Only", variant="secondary")
            use_relighting = gr.Checkbox(label="Use Relighting LoRA (for replacement mode)", value=False)
            load_btn = gr.Button("🚀 Load Model", variant="primary")
        
        load_status = gr.Textbox(label="Status", interactive=False)
        
        def download_models_ui():
            success, msg = cache_manager.download_models()
            return msg
        
        download_only_btn.click(download_models_ui, outputs=[load_status])
        load_btn.click(load_model, inputs=[use_relighting], outputs=[load_status])
    
    with gr.Tab("2️⃣ Preprocess"):
        gr.Markdown("### Preprocess Video\nExtract pose, face, and reference data from your video")
        gr.Markdown("⏱️ Preprocessing takes 1-3 minutes. Watch the Status textbox for progress!")
        with gr.Row():
            with gr.Column():
                input_video = gr.File(label="Upload Video (MP4)", file_types=[".mp4", ".avi", ".mov"])
                reference_image = gr.Image(label="Reference Image (character)", type="filepath")
                mode = gr.Radio(["animate", "replace"], label="Mode", value="animate")
                
                # Quick use example button
                use_example_btn = gr.Button("📚 Use Example Files", variant="secondary")
                preprocess_btn = gr.Button("⚡ Start Preprocessing", variant="primary")
            with gr.Column():
                preprocess_output_path = gr.Textbox(
                    label="Preprocessed Output Path", 
                    interactive=False,
                    placeholder="Output path will appear here..."
                )
                preprocess_status = gr.Textbox(
                    label="Status", 
                    lines=8,
                    placeholder="Click 'Use Example Files' or upload your own, then click 'Start Preprocessing'"
                )
        
        # Function to load examples
        def load_example_files():
            return (
                "/workspace/wan22-comfy-project/Wan2.2/examples/pose.mp4",
                "/workspace/wan22-comfy-project/Wan2.2/examples/pose.png",
                "animate"
            )
        
        use_example_btn.click(
            load_example_files,
            outputs=[input_video, reference_image, mode]
        )
        
        preprocess_btn.click(
            preprocess_video,
            inputs=[input_video, reference_image, mode],
            outputs=[preprocess_output_path, preprocess_status]
        )
    
    with gr.Tab("3️⃣ Generate"):
        gr.Markdown("### Generate Animation")
        gr.Markdown("⚠️ **Memory Warning**: For 48GB VRAM, use 49 frames max. For 24GB VRAM, use 25-33 frames.")
        with gr.Row():
            with gr.Column():
                preprocessed_path_input = gr.Textbox(
                    label="Preprocessed Data Path",
                    placeholder="Paste path from preprocessing step"
                )
                prompt_input = gr.Textbox(
                    label="Prompt (optional)",
                    value="视频中的人在做动作",
                    lines=2
                )
                
                with gr.Row():
                    seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    num_frames = gr.Slider(5, 81, value=49, step=4, label="Frames (4n+1, use 49 for lower VRAM)")
                
                with gr.Row():
                    sampling_steps = gr.Slider(10, 50, value=20, step=1, label="Sampling Steps")
                    guide_scale = gr.Slider(1.0, 3.0, value=1.0, step=0.1, label="Guidance Scale")
                
                use_replace = gr.Checkbox(label="Use Replace Mode", value=False)
                generate_btn = gr.Button("Generate Video", variant="primary")
            
            with gr.Column():
                output_video = gr.Video(label="Generated Video")
                gen_status = gr.Textbox(label="Status", lines=3)
        
        generate_btn.click(
            generate_animation,
            inputs=[
                preprocessed_path_input, prompt_input, seed_input,
                num_frames, sampling_steps, guide_scale, use_replace
            ],
            outputs=[output_video, gen_status]
        )
    
    with gr.Tab("ℹ️ Info"):
        gr.Markdown("""
        ## 🎭 Wan2.2 Animate-14B Interface
        
        ### How to Use
        
        1. **Setup**: Download and load models (~50GB from HuggingFace)
           - First time downloads from [Wan-AI/Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B)
           - Models cached at `/home/caches/Wan2.2-Animate-14B`
        
        2. **Preprocess**: Upload your driving video and reference character image
           - **Animate Mode**: Character mimics the motion from video
           - **Replace Mode**: Character replaces person in video (needs background extraction)
           - Preprocessing extracts pose, face, and reference data
        
        3. **Generate**: Create your animated video!
           - Use preprocessed path from step 2
           - Adjust parameters for quality vs speed
        
        ## Parameters
        
        - **Frames**: Total frames (4n+1 format). 77 frames ≈ 2.5s at 30fps
        - **Sampling Steps**: More steps = better quality but slower (20 recommended)
        - **Guidance Scale**: 1.0 for most cases, >1.0 for expression control
        - **Seed**: -1 for random, or use specific number for reproducible results
        
        ## Example Files
        
        ```
        Video: /workspace/wan22-comfy-project/Wan2.2/examples/pose.mp4
        Image: /workspace/wan22-comfy-project/Wan2.2/examples/pose.png
        ```
        
        ## System Info
        
        - **Environment**: `/workspace/wan22-comfy-project/venv`
        - **Cache**: `/home/caches/Wan2.2-Animate-14B`
        - **Output**: `/workspace/wan22-comfy-project/outputs`
        - **Model**: Wan-AI/Wan2.2-Animate-14B (14B parameters)
        
        ## Notes
        
        - First generation will be slower as models compile
        - T5 encoder offloaded to CPU to save VRAM
        - Supports both animation and replacement modes
        - Replace mode requires relighting LoRA for better results
        """)

if __name__ == "__main__":
    logger.info("✅ Gradio interface ready!")
    logger.info("📝 Heavy model loading deferred until 'Load Model' is clicked")
    logger.info("🌐 Starting web server...")
    logger.info("")
    logger.info("=" * 60)
    logger.info("💡 QUICK START:")
    logger.info("   1. Go to 'Preprocess' tab")
    logger.info("   2. Click 'Use Example Files' button")
    logger.info("   3. Click 'Start Preprocessing' and wait 1-3 min")
    logger.info("   4. Copy output path to 'Generate' tab")
    logger.info("=" * 60)
    
    app.launch(
        server_name="0.0.0.0",
        share=True,
        show_error=True
    )
