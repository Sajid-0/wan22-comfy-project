#!/usr/bin/env python3
"""
Gradio Web UI for Wan2.2-I2V-A14B (Image-to-Video Generation)
Production-ready interface with advanced controls and multi-GPU support
"""

import os
import sys
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import random
import shutil

import gradio as gr
import torch
import torch.distributed as dist
from PIL import Image
import numpy as np

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.utils import save_video

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model management
global_model = None
global_config = None
model_loaded = False

# Default paths
DEFAULT_CACHE_DIR = "/home/caches/Wan2.2-I2V-A14B"
DEFAULT_OUTPUT_DIR = "/workspace/wan22-comfy-project/outputs"

# Resolution presets with optimal shift values
RESOLUTION_PRESETS = {
    "480P (480×832)": {
        "size": "480*832",
        "max_area": 480 * 832,
        "shift": 3.0,
        "description": "Fast generation, lower quality"
    },
    "720P (720×1280)": {
        "size": "720*1280", 
        "max_area": 720 * 1280,
        "shift": 5.0,
        "description": "Balanced quality and speed (Recommended)"
    },
}

# Quality presets
QUALITY_PRESETS = {
    "Draft (Fast)": {
        "steps": 20,
        "guide_scale": (3.0, 3.0),
        "description": "Quick preview, lower quality"
    },
    "Standard": {
        "steps": 40,
        "guide_scale": (3.5, 3.5),
        "description": "Default quality (Recommended)"
    },
    "High Quality": {
        "steps": 60,
        "guide_scale": (4.0, 4.0),
        "description": "Best quality, slower generation"
    },
}


def ensure_models_ready(cache_dir=DEFAULT_CACHE_DIR):
    """Ensure I2V models are downloaded and ready"""
    logger.info("🔍 Checking if I2V models are ready...")
    
    setup_script = Path(__file__).parent / 'setup_i2v_cache.py'
    if not setup_script.exists():
        return False, "❌ setup_i2v_cache.py not found!"
    
    try:
        # Run quick check
        result = subprocess.run(
            [sys.executable, str(setup_script), 'quick'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            logger.info("✅ Models are ready!")
            return True, "✅ Models ready"
        else:
            return False, f"❌ Model setup failed: {result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return False, "❌ Model setup timeout (>5 min)"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def initialize_model(cache_dir=DEFAULT_CACHE_DIR, device_id=0, use_multi_gpu=False):
    """Initialize the I2V model"""
    global global_model, global_config, model_loaded
    
    if model_loaded and global_model is not None:
        logger.info("✅ Model already loaded")
        return True, "✅ Model already loaded"
    
    try:
        logger.info("🚀 Initializing Wan2.2-I2V-A14B model...")
        
        # Load config
        global_config = WAN_CONFIGS["i2v-A14B"]
        
        # Initialize model with appropriate settings
        global_model = wan.WanI2V(
            config=global_config,
            checkpoint_dir=cache_dir,
            device_id=device_id,
            rank=0,
            t5_fsdp=use_multi_gpu,  # Use FSDP for multi-GPU
            dit_fsdp=use_multi_gpu,
            use_sp=use_multi_gpu,  # Use sequence parallel for multi-GPU
            t5_cpu=not use_multi_gpu,  # Only offload to CPU in single-GPU mode
            init_on_cpu=False,
            convert_model_dtype=False,
        )
        
        model_loaded = True
        logger.info("✅ Model loaded successfully!")
        return True, "✅ Model loaded successfully!"
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}", exc_info=True)
        return False, f"❌ Failed to load model: {str(e)[:200]}"


def generate_video(
    image_input,
    prompt_text,
    resolution_preset,
    quality_preset,
    frame_count,
    seed_value,
    advanced_mode,
    custom_steps,
    custom_shift,
    custom_guide_scale_low,
    custom_guide_scale_high,
    sample_solver,
    offload_model,
    progress=gr.Progress()
):
    """Main video generation function"""
    global global_model, global_config, model_loaded
    
    # Validate inputs
    if image_input is None:
        return None, "❌ Please upload an image!"
    
    if not prompt_text or len(prompt_text.strip()) == 0:
        return None, "❌ Please enter a prompt!"
    
    # Check if model is loaded
    if not model_loaded or global_model is None:
        return None, "❌ Model not loaded! Please click 'Load Model' first."
    
    try:
        progress(0.1, desc="Preparing inputs...")
        
        # Process image
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        else:
            return None, f"❌ Unsupported image type: {type(image_input)}"
        
        # Get resolution settings
        res_config = RESOLUTION_PRESETS[resolution_preset]
        max_area = res_config["max_area"]
        
        # Get quality settings
        if advanced_mode:
            steps = custom_steps
            shift = custom_shift
            guide_scale = (custom_guide_scale_low, custom_guide_scale_high)
        else:
            quality_config = QUALITY_PRESETS[quality_preset]
            steps = quality_config["steps"]
            shift = res_config["shift"]
            guide_scale = quality_config["guide_scale"]
        
        # Validate frame count (must be 4n+1)
        if (frame_count - 1) % 4 != 0:
            return None, f"❌ Frame count must be 4n+1 (e.g., 49, 81, 105). Got {frame_count}"
        
        # Set seed
        if seed_value == -1:
            seed_value = random.randint(0, 2**31 - 1)
        
        progress(0.2, desc="Generating video...")
        logger.info(f"🎬 Starting generation with:")
        logger.info(f"   Resolution: {resolution_preset} (max_area={max_area})")
        logger.info(f"   Frames: {frame_count}")
        logger.info(f"   Steps: {steps}")
        logger.info(f"   Shift: {shift}")
        logger.info(f"   Guide Scale: {guide_scale}")
        logger.info(f"   Seed: {seed_value}")
        logger.info(f"   Solver: {sample_solver}")
        
        # Generate video
        video_tensor = global_model.generate(
            input_prompt=prompt_text,
            img=img,
            max_area=max_area,
            frame_num=frame_count,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=steps,
            guide_scale=guide_scale,
            n_prompt="",
            seed=seed_value,
            offload_model=offload_model
        )
        
        progress(0.9, desc="Saving video...")
        
        # Save video
        output_dir = Path(DEFAULT_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_slug = prompt_text.replace(" ", "_").replace("/", "_")[:30]
        output_filename = f"i2v_{resolution_preset.split()[0]}_{frame_count}f_{timestamp}_{prompt_slug}.mp4"
        output_path = output_dir / output_filename
        
        save_video(
            tensor=video_tensor[None],
            save_file=str(output_path),
            fps=global_config.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        
        progress(1.0, desc="Complete!")
        logger.info(f"✅ Video saved to: {output_path}")
        
        status_msg = f"""✅ **Generation Complete!**
- Output: `{output_path.name}`
- Resolution: {resolution_preset}
- Frames: {frame_count}
- Steps: {steps}
- Seed: {seed_value}
- File size: {output_path.stat().st_size / (1024**2):.2f} MB
"""
        
        return str(output_path), status_msg
        
    except torch.cuda.OutOfMemoryError:
        return None, "❌ CUDA Out of Memory! Try: Lower resolution, fewer frames, or enable 'Offload Model'"
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}", exc_info=True)
        return None, f"❌ Generation failed: {str(e)[:300]}"


def update_preset_info(resolution_preset, quality_preset):
    """Update info text when presets change"""
    res_info = RESOLUTION_PRESETS[resolution_preset]["description"]
    quality_info = QUALITY_PRESETS[quality_preset]["description"]
    
    return f"**Resolution:** {res_info}\n**Quality:** {quality_info}"


def toggle_advanced_controls(advanced_mode):
    """Show/hide advanced controls"""
    return {
        custom_steps: gr.update(visible=advanced_mode),
        custom_shift: gr.update(visible=advanced_mode),
        custom_guide_scale_low: gr.update(visible=advanced_mode),
        custom_guide_scale_high: gr.update(visible=advanced_mode),
        quality_preset: gr.update(visible=not advanced_mode),
    }


def create_gradio_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="Wan2.2 I2V Generator",
        theme=gr.themes.Soft(),
        css="""
        .header {text-align: center; padding: 20px;}
        .footer {text-align: center; padding: 10px; font-size: 0.9em; color: #666;}
        .status-box {padding: 10px; border-radius: 5px; margin: 10px 0;}
        """
    ) as app:
        
        # Header
        gr.Markdown("""
        <div class="header">
        <h1>🎬 Wan2.2-I2V-A14B: Image-to-Video Generator</h1>
        <p><b>Transform static images into dynamic videos with AI</b></p>
        <p>Powered by Alibaba's Wan2.2 MoE (27B params, 14B active)</p>
        </div>
        """)
        
        with gr.Row():
            # Left Column - Inputs
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Input Configuration")
                
                # Image upload
                image_input = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=300,
                    sources=["upload", "clipboard"]
                )
                
                # Prompt
                prompt_text = gr.Textbox(
                    label="Prompt (describe the motion/action)",
                    placeholder="e.g., Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard...",
                    lines=4,
                    value="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression."
                )
                
                # Quick presets
                with gr.Accordion("⚙️ Generation Settings", open=True):
                    resolution_preset = gr.Radio(
                        choices=list(RESOLUTION_PRESETS.keys()),
                        value="720P (720×1280)",
                        label="Resolution Preset",
                        info="Higher resolution = better quality but slower"
                    )
                    
                    quality_preset = gr.Radio(
                        choices=list(QUALITY_PRESETS.keys()),
                        value="Standard",
                        label="Quality Preset",
                        info="Higher quality = more sampling steps"
                    )
                    
                    preset_info = gr.Markdown(
                        value=update_preset_info("720P (720×1280)", "Standard")
                    )
                    
                    frame_count = gr.Slider(
                        minimum=49,
                        maximum=161,
                        step=4,
                        value=81,
                        label="Frame Count (must be 4n+1)",
                        info="More frames = longer video"
                    )
                    
                    seed_value = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1,
                        precision=0
                    )
                
                # Advanced settings
                with gr.Accordion("🔧 Advanced Settings", open=False):
                    advanced_mode = gr.Checkbox(
                        label="Enable Advanced Controls",
                        value=False
                    )
                    
                    custom_steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        step=5,
                        value=40,
                        label="Sampling Steps",
                        visible=False
                    )
                    
                    custom_shift = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        step=0.5,
                        value=5.0,
                        label="Shift Value",
                        info="Use 3.0 for 480P, 5.0 for 720P",
                        visible=False
                    )
                    
                    custom_guide_scale_low = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        step=0.5,
                        value=3.5,
                        label="Guide Scale (Low Noise Model)",
                        visible=False
                    )
                    
                    custom_guide_scale_high = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        step=0.5,
                        value=3.5,
                        label="Guide Scale (High Noise Model)",
                        visible=False
                    )
                    
                    sample_solver = gr.Radio(
                        choices=["unipc", "dpm++"],
                        value="unipc",
                        label="Sampling Solver",
                        info="UniPC is recommended"
                    )
                    
                    offload_model = gr.Checkbox(
                        label="Offload Model (saves VRAM)",
                        value=True,
                        info="Enable if running out of memory"
                    )
                
                # Generate button
                generate_btn = gr.Button(
                    "🎬 Generate Video",
                    variant="primary",
                    size="lg"
                )
            
            # Right Column - Outputs
            with gr.Column(scale=1):
                gr.Markdown("### 🎥 Generated Video")
                
                # Video output
                video_output = gr.Video(
                    label="Generated Video",
                    height=400,
                    autoplay=True
                )
                
                # Status output
                status_output = gr.Markdown(
                    value="*Ready to generate. Upload an image and click 'Generate Video'*",
                    elem_classes="status-box"
                )
                
                # Examples
                with gr.Accordion("📚 Example Prompts", open=False):
                    gr.Markdown("""
                    **Example 1 - Beach Cat:**
                    ```
                    Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. 
                    The fluffy-furred feline gazes directly at the camera with a relaxed expression.
                    ```
                    
                    **Example 2 - Portrait Animation:**
                    ```
                    A young woman with long flowing hair smiles warmly at the camera. 
                    Gentle wind blows her hair. Soft golden hour lighting illuminates her face.
                    ```
                    
                    **Example 3 - Nature Scene:**
                    ```
                    A majestic mountain landscape with rolling clouds. The sun breaks through 
                    the mist, creating dramatic rays of light. Camera slowly pans across the vista.
                    ```
                    
                    **Example 4 - Urban Scene:**
                    ```
                    A bustling city street at night with neon lights reflecting on wet pavement. 
                    Cars pass by with glowing headlights. Gentle rain creates a cinematic atmosphere.
                    ```
                    """)
                
                # Model controls
                with gr.Accordion("🔧 Model Management", open=False):
                    model_status = gr.Textbox(
                        label="Model Status",
                        value="Not loaded",
                        interactive=False
                    )
                    
                    with gr.Row():
                        check_models_btn = gr.Button("Check Models", size="sm")
                        load_model_btn = gr.Button("Load Model", size="sm", variant="primary")
        
        # Footer
        gr.Markdown("""
        <div class="footer">
        <p>💡 <b>Tips:</b> Start with 720P + Standard quality for best results | Use 480P for faster previews | Increase steps for higher quality</p>
        <p>📝 <b>Frame Count:</b> Must be 4n+1 (49, 81, 105, etc.) | More frames = longer video but slower generation</p>
        <p>🔗 <b>Model:</b> <a href="https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B">Wan2.2-I2V-A14B</a> | 
        <a href="https://github.com/Wan-Video/Wan2.2">GitHub</a></p>
        </div>
        """)
        
        # Event handlers
        resolution_preset.change(
            fn=update_preset_info,
            inputs=[resolution_preset, quality_preset],
            outputs=preset_info
        )
        
        quality_preset.change(
            fn=update_preset_info,
            inputs=[resolution_preset, quality_preset],
            outputs=preset_info
        )
        
        advanced_mode.change(
            fn=toggle_advanced_controls,
            inputs=advanced_mode,
            outputs=[custom_steps, custom_shift, custom_guide_scale_low, 
                    custom_guide_scale_high, quality_preset]
        )
        
        check_models_btn.click(
            fn=ensure_models_ready,
            outputs=[None, model_status]
        )
        
        load_model_btn.click(
            fn=initialize_model,
            outputs=[None, model_status]
        )
        
        generate_btn.click(
            fn=generate_video,
            inputs=[
                image_input, prompt_text, resolution_preset, quality_preset,
                frame_count, seed_value, advanced_mode, custom_steps,
                custom_shift, custom_guide_scale_low, custom_guide_scale_high,
                sample_solver, offload_model
            ],
            outputs=[video_output, status_output]
        )
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Gradio Web UI for Wan2.2-I2V-A14B")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Path to model cache directory"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="Server name (use 0.0.0.0 for external access)"
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Server port"
    )
    parser.add_argument(
        "--auto_load_model",
        action="store_true",
        help="Automatically load model on startup"
    )
    parser.add_argument(
        "--use_multi_gpu",
        action="store_true",
        help="Use multi-GPU mode (FSDP + Sequence Parallel)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(DEFAULT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Starting Gradio Web UI for Wan2.2-I2V-A14B")
    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info(f"Output directory: {DEFAULT_OUTPUT_DIR}")
    
    # Auto-load model if requested
    if args.auto_load_model:
        logger.info("🔄 Auto-loading model...")
        ensure_models_ready(args.cache_dir)
        initialize_model(args.cache_dir, use_multi_gpu=args.use_multi_gpu)
    
    # Create and launch interface
    app = create_gradio_interface()
    
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
        favicon_path=None,
        inbrowser=True  # Auto-open in browser
    )


if __name__ == "__main__":
    main()
