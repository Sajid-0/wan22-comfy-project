#!/bin/bash
#
# Run Wan2.2 S2V-14B Multi-GPU Generation
# Optimized for 2x NVIDIA A40 GPUs
#

# Configuration
CKPT_DIR="/home/caches/Wan2.2-S2V-14B"
IMAGE="/workspace/wan22-comfy-project/prompt.png"
AUDIO="/workspace/wan22-comfy-project/tmpi27jbzzb.wav"
PROMPT="A sharp, high-definition (4K) video of an excited woman speaking with a bright smile and expressive facial gestures. Her body movements are energetic but controlled. The scene is filmed with a handheld camera, featuring a slight, natural shake and subtle panning movements. Casual home setting with soft, natural light and sharp focus."

# Resolution (for S2V, this is the area, aspect ratio follows input image)
SIZE="480*832"  # Good balance for 720P-ish output

# Multi-GPU settings
NUM_GPUS=2

# Output settings
OUTPUT_DIR="./outputs"
mkdir -p "$OUTPUT_DIR"

# Set environment variables for better multi-GPU performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8

echo "========================================"
echo "Wan2.2 S2V Multi-GPU Generation"
echo "========================================"
echo "Model: S2V-14B"
echo "GPUs: $NUM_GPUS x A40"
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
echo "🚀 Starting multi-GPU generation..."
echo ""

# Run with torchrun for multi-GPU
torchrun --nproc_per_node=$NUM_GPUS generate.py \
    --task s2v-14B \
    --size "$SIZE" \
    --ckpt_dir "$CKPT_DIR" \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size $NUM_GPUS \
    --prompt "$PROMPT" \
    --image "$IMAGE" \
    --audio "$AUDIO" \
    --num_clip 1 \
    --infer_frames 80 \
    --sample_steps 40 \
    --sample_guide_scale 7.5 \
    --sample_shift 3.0

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
