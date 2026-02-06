#!/bin/bash
# Launch script for Wan2.2 Animate Gradio UI

cd /workspace/wan22-comfy-project/Wan2.2

echo "🚀 Starting Wan2.2 Animate Gradio Interface..."
echo "📍 Running from: $(pwd)"
echo "🐍 Using Python: /workspace/wan22-comfy-project/venv/bin/python"
echo ""

# Activate venv and run
/workspace/wan22-comfy-project/venv/bin/python gradio_animate_app.py
