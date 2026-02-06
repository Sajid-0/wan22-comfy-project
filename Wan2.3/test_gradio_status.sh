#!/bin/bash
# Quick test script for Gradio Animate Interface

echo "🧪 Testing Gradio Animate Interface..."
echo ""

# 1. Check if Gradio is running
echo "1️⃣ Checking if Gradio is running..."
if curl -s http://localhost:7863/ > /dev/null; then
    echo "   ✅ Gradio is running at http://localhost:7863"
else
    echo "   ❌ Gradio is not running"
    echo "   Start it with: python gradio_animate_app.py"
    exit 1
fi
echo ""

# 2. Check example files exist
echo "2️⃣ Checking example files..."
if [ -f "/workspace/wan22-comfy-project/Wan2.2/examples/pose.mp4" ]; then
    echo "   ✅ Example video exists ($(du -h /workspace/wan22-comfy-project/Wan2.2/examples/pose.mp4 | cut -f1))"
else
    echo "   ❌ Example video not found"
fi

if [ -f "/workspace/wan22-comfy-project/Wan2.2/examples/pose.png" ]; then
    echo "   ✅ Example image exists ($(du -h /workspace/wan22-comfy-project/Wan2.2/examples/pose.png | cut -f1))"
else
    echo "   ❌ Example image not found"
fi
echo ""

# 3. Check if models are downloaded
echo "3️⃣ Checking if models are downloaded..."
if [ -d "/home/caches/Wan2.2-Animate-14B" ]; then
    model_size=$(du -sh /home/caches/Wan2.2-Animate-14B 2>/dev/null | cut -f1)
    echo "   ✅ Models directory exists ($model_size)"
    
    # Check for key model files
    if [ -f "/home/caches/Wan2.2-Animate-14B/diffusion_pytorch_model.safetensors.index.json" ]; then
        echo "   ✅ Main model index found"
    else
        echo "   ⚠️  Models may not be fully downloaded"
    fi
else
    echo "   ⚠️  Models not downloaded yet"
    echo "   Use the Setup tab in Gradio to download models"
fi
echo ""

# 4. Check preprocessing directory
echo "4️⃣ Checking preprocessing directory..."
if [ -d "/workspace/wan22-comfy-project/Wan2.2/preprocessed" ]; then
    preprocess_count=$(ls -1 /workspace/wan22-comfy-project/Wan2.2/preprocessed/ 2>/dev/null | wc -l)
    echo "   ✅ Preprocessed directory exists ($preprocess_count outputs)"
else
    echo "   ℹ️  No preprocessing outputs yet"
fi
echo ""

# 5. Check virtual environment
echo "5️⃣ Checking virtual environment..."
if [ -f "/workspace/wan22-comfy-project/venv/bin/python" ]; then
    python_version=$(/workspace/wan22-comfy-project/venv/bin/python --version 2>&1)
    echo "   ✅ Virtual environment active ($python_version)"
else
    echo "   ❌ Virtual environment not found"
fi
echo ""

# 6. Check Gradio logs
echo "6️⃣ Recent Gradio logs..."
if [ -f "/workspace/wan22-comfy-project/Wan2.2/gradio_animate.log" ]; then
    echo "   Last 5 lines:"
    tail -5 /workspace/wan22-comfy-project/Wan2.2/gradio_animate.log | sed 's/^/   /'
else
    echo "   ℹ️  No log file yet (not running in background)"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Fixed Issues:"
echo "   • Video upload error handling"
echo "   • Examples integrated in UI"
echo "   • Better error messages"
echo "   • Debug logging added"
echo ""
echo "🔗 Access the interface:"
echo "   http://localhost:7863"
echo ""
echo "📚 Documentation:"
echo "   • GRADIO_ANIMATE_QUICKSTART.md - How to use"
echo "   • GRADIO_FIXES_SUMMARY.md - What was fixed"
echo ""
echo "🎯 Next Steps:"
echo "   1. Open http://localhost:7863 in your browser"
echo "   2. Go to Setup tab → Download Models (if not done)"
echo "   3. Go to Preprocess tab → Click on example"
echo "   4. Click Preprocess button"
echo "   5. Copy output path to Generate tab"
echo ""
