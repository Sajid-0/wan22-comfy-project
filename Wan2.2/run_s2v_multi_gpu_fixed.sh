#!/bin/bash
#
# Wan2.2 S2V-14B Multi-GPU Generation (FIXED VERSION)
# Issue Fixed: Multiple clip generation causing repeated iterations
#

# ==================== CONFIGURATION ====================
CKPT_DIR="/home/caches/Wan2.2-S2V-14B"
IMAGE="/workspace/wan22-comfy-project/prompt.png"
AUDIO="/workspace/wan22-comfy-project/tmpi27jbzzb.wav"
PROMPT="A sharp, high-definition (4K) video of an excited woman speaking with a bright smile and expressive facial gestures. Her body movements are energetic but controlled. The scene is filmed with a handheld camera, featuring a slight, natural shake and subtle panning movements. Casual home setting with soft, natural light and sharp focus."

# Video Generation Settings
SIZE="480*832"              # Resolution area (aspect ratio from input image)
NUM_CLIP=1                  # ⚠️ IMPORTANT: Generate only 1 clip (prevents multiple iterations)
INFER_FRAMES=80             # Frames per clip (80 frames = ~5 sec @ 16fps)
SAMPLE_STEPS=40             # Diffusion steps (40 = balanced quality/speed)
GUIDE_SCALE=7.5             # CFG scale (7.5 = strong prompt adherence)
SHIFT=3.0                   # Noise schedule (3.0 for 480p, 5.0 for 720p+)
BASE_SEED=42                # Fixed seed for reproducibility

# Multi-GPU Settings
NUM_GPUS=2

# Performance Optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8

# ==================== VALIDATION ====================
echo "========================================"
echo "Wan2.2 S2V Multi-GPU (FIXED)"
echo "========================================"
echo "Model: S2V-14B"
echo "GPUs: $NUM_GPUS x A40"
echo "Clips: $NUM_CLIP (prevents multi-iteration)"
echo "Frames: $INFER_FRAMES (~$((INFER_FRAMES/16))sec)"
echo "Steps: $SAMPLE_STEPS"
echo "Resolution: $SIZE"
echo "========================================"
echo ""

if [ ! -d "$CKPT_DIR" ]; then
    echo "❌ Error: Checkpoint directory not found: $CKPT_DIR"
    exit 1
fi

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
echo "🚀 Starting single-clip generation..."
echo "   This will run through the progress bar ONCE"
echo ""

# ==================== EXECUTE ====================
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
    --num_clip $NUM_CLIP \
    --infer_frames $INFER_FRAMES \
    --sample_steps $SAMPLE_STEPS \
    --sample_guide_scale $GUIDE_SCALE \
    --sample_shift $SHIFT \
    --base_seed $BASE_SEED

# ==================== RESULT ====================
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ Generation completed successfully!"
    echo "========================================"
    echo ""
    echo "Generated video files:"
    ls -lh s2v-14B_*.mp4 2>/dev/null | tail -3
    echo ""
    echo "Total video files in directory:"
    ls -1 *.mp4 2>/dev/null | wc -l
else
    echo ""
    echo "========================================"
    echo "❌ Generation failed (exit code: $EXIT_CODE)"
    echo "========================================"
fi

exit $EXIT_CODE
