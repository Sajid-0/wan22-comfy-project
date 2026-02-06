#!/bin/bash
# RunPod Startup Script for Wan2.2 S2V
# Run this script every time you start a new RunPod instance

echo "🚀 RunPod Wan2.2 S2V Startup Script"
echo "=================================="

# Navigate to the project directory
cd /workspace/wan22-comfy-project/Wan2.2

# Set up models (will download if missing from volatile /home)
echo "📥 Setting up models (this may take a few minutes on first run)..."
/workspace/wan22-comfy-project/venv/bin/python setup_s2v_cache.py quick

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Setup completed successfully!"
    echo ""
    echo "🎬 You can now run S2V generation with:"
    echo "   python run_s2v_multi_gpu.py"
    echo ""
    echo "💡 Tips:"
    echo "   - Models are cached in /home/caches (~43GB)"
    echo "   - /home gets reset when RunPod instance stops"
    echo "   - Run this script again after restarting RunPod"
    echo ""
else
    echo "❌ Setup failed. Please check the logs above."
    exit 1
fi