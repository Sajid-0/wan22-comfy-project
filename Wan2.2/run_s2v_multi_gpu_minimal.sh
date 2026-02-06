#!/bin/bash
#
# Wan2.2 S2V-14B Multi-GPU Generation (MINIMAL VERSION)
# Essential parameters only - uses config defaults for everything else
#

# ==================== CONFIGURATION ====================
CKPT_DIR="/home/caches/Wan2.2-S2V-14B"
IMAGE="/workspace/wan22-comfy-project/prompt.png"
AUDIO="/workspace/wan22-comfy-project/tmpi27jbzzb.wav"
PROMPT="A sharp, high-definition (4K) video of an excited woman speaking with a bright smile and expressive facial gestures. Her body movements are energetic but controlled. The scene is filmed with a handheld camera, featuring a slight, natural shake and subtle panning movements. Casual home setting with soft, natural light and sharp focus."

# ==================== VALIDATION ====================
echo "========================================"
echo "Wan2.2 S2V Multi-GPU (MINIMAL)"
echo "========================================"
echo "Model: S2V-14B"
echo "GPUs: 2 x A40"
echo "Mode: Essential parameters only"
echo "========================================"
echo ""

# Check if checkpoint exists
if [ ! -d "$CKPT_DIR" ]; then
    echo "❌ Error: Checkpoint directory not found: $CKPT_DIR"
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
echo "🚀 Starting multi-GPU generation with minimal parameters..."
echo "   Using config defaults for: steps, guide_scale, shift, frames"
echo ""

# ==================== RUN ====================
# Minimal command - only specify what's required
torchrun --nproc_per_node=2 generate.py \
    --task s2v-14B \
    --ckpt_dir "$CKPT_DIR" \
    --image "$IMAGE" \
    --audio "$AUDIO" \
    --prompt "$PROMPT" \
    --num_clip 1 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 2

# ==================== RESULT ====================
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ Generation completed successfully!"
    echo "========================================"
    echo "Output saved to current directory"
    echo ""
    ls -lh *.mp4 | tail -3
else
    echo ""
    echo "========================================"
    echo "❌ Generation failed"
    echo "========================================"
    echo "Check the error messages above"
fi
