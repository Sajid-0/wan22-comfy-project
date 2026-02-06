#!/bin/bash
# Quick start script for Multi-GPU S2V System

# Activate existing virtual environment
source /workspace/wan22-comfy-project/venv/bin/activate

# Set environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:./Wan2.2"

# Example command (modify paths as needed)
echo "Example usage:"
echo "python3 main_s2v_system.py \\"
echo "  --image /workspace/wan22-comfy-project/iphone.jpeg \\"
echo "  --audio /workspace/wan22-comfy-project/tmp_19iifpd.mp3 \\"
echo "  --prompt 'A person speaking into a phone' \\"
echo "  --output example_video \\"
echo "  --quality medium \\"
echo "  --gpus 2"
echo
echo "To run the example:"
echo "chmod +x quick_start.sh"
echo "./quick_start.sh"
