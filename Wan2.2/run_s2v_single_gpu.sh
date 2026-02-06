#!/bin/bash
#
# Run Wan2.2 S2V-14B Single-GPU Generation (Memory Optimized)
# For testing and cost-saving on single A40
#

# Configuration
CKPT_DIR="/home/caches/Wan2.2-S2V-14B"
IMAGE="/workspace/wan22-comfy-project/prompt.png"
AUDIO="/workspace/wan22-comfy-project/tmpi27jbzzb.wav"
PROMPT="A sharp, high-definition (4K) video of an excited woman speaking with a bright smile and expressive facial gestures. Her body movements are energetic but controlled. The scene is filmed with a handheld camera, featuring a slight, natural shake and subtle panning movements. Casual home setting with soft, natural light and sharp focus."

# Resolution (for S2V, this is the area, aspect ratio follows input image)
SIZE="1024*704"  # Good balance for 720P-ish output

# Output settings
OUTPUT_DIR="./outputs"
mkdir -p "$OUTPUT_DIR"

# Set environment variables for better memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "========================================"
echo "Wan2.2 S2V Single-GPU Generation"
echo "========================================"
echo "Model: S2V-14B"
echo "GPUs: 1 x A40 (Memory Optimized)"
echo "Image: $IMAGE"
echo "Audio: $AUDIO"
echo "Resolution: $SIZE"
echo "========================================"
echo ""

# Check if checkpoint exists
if [ ! -d "$CKPT_DIR" ]; then
    echo "❌ Error: Checkpoint directory not found: $CKPT_DIR"
    echo "Please download the model first:"
    echo "  huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir /home/caches/Wan2.2-S2V-14B"
    exit 1
fi

# Check if input files exist
if [ ! -f "$IMAGE" ]; then
    echo "❌ Error: Image file not found: $IMAGE"
    exit 1
fi

if [ ! -f "$AUDIO" ]; then
    echo "❌ Error: Audio file not found: $AUDIO"
    exit 1
fi

echo "✅ All files verified"
echo ""
echo "🚀 Starting single-GPU generation (memory optimized)..."
echo ""

# Run with memory optimizations for single GPU
python generate.py \
    --task s2v-14B \
    --size "$SIZE" \
    --ckpt_dir "$CKPT_DIR" \
    --offload_model True \
    --convert_model_dtype \
    --t5_cpu \
    --prompt "$PROMPT" \
    --image "$IMAGE" \
    --audio "$AUDIO" \
    --sample_steps 50 \
    --sample_guide_scale 7.5

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ Generation completed successfully!"
    echo "========================================"
    echo "Output saved to current directory"
    echo ""
    ls -lh *.mp4 | tail -5
else
    echo ""
    echo "========================================"
    echo "❌ Generation failed"
    echo "========================================"
    echo "Check the error messages above"
fi
