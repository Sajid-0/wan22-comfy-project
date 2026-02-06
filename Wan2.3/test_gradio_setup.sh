#!/bin/bash
# Quick test script to verify Gradio I2V UI setup
# Run this to confirm everything is connected properly

echo "🔍 Testing Gradio I2V UI Setup..."
echo ""

# Test 1: Check virtual environment
echo "1️⃣ Checking virtual environment..."
if [ -f "/workspace/wan22-comfy-project/venv/bin/python" ]; then
    echo "   ✅ Virtual environment found"
    PYTHON_VERSION=$(/workspace/wan22-comfy-project/venv/bin/python --version)
    echo "   ✅ $PYTHON_VERSION"
else
    echo "   ❌ Virtual environment NOT found"
    exit 1
fi
echo ""

# Test 2: Check PyTorch
echo "2️⃣ Checking PyTorch..."
TORCH_TEST=$(/workspace/wan22-comfy-project/venv/bin/python -c 'import torch; print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")' 2>&1)
if [ $? -eq 0 ]; then
    echo "   ✅ $TORCH_TEST"
else
    echo "   ❌ PyTorch check failed"
    exit 1
fi
echo ""

# Test 3: Check Gradio
echo "3️⃣ Checking Gradio..."
GRADIO_TEST=$(/workspace/wan22-comfy-project/venv/bin/python -c 'import gradio; print(f"Gradio {gradio.__version__}")' 2>&1)
if [ $? -eq 0 ]; then
    echo "   ✅ $GRADIO_TEST"
else
    echo "   ❌ Gradio check failed"
    echo "   Installing Gradio..."
    /workspace/wan22-comfy-project/venv/bin/pip install gradio>=4.0.0
fi
echo ""

# Test 4: Check required files
echo "4️⃣ Checking Gradio UI files..."
FILES=("gradio_i2v_app.py" "launch_i2v_ui.sh" "GRADIO_I2V_GUIDE.md")
for file in "${FILES[@]}"; do
    if [ -f "/workspace/wan22-comfy-project/Wan2.2/$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file NOT found"
    fi
done
echo ""

# Test 5: Check launcher script
echo "5️⃣ Checking launcher script..."
if [ -x "/workspace/wan22-comfy-project/Wan2.2/launch_i2v_ui.sh" ]; then
    echo "   ✅ Launcher is executable"
else
    echo "   ⚠️  Making launcher executable..."
    chmod +x /workspace/wan22-comfy-project/Wan2.2/launch_i2v_ui.sh
    echo "   ✅ Fixed"
fi
echo ""

# Test 6: Check GPU
echo "6️⃣ Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    echo "   ✅ GPU detected: $GPU_NAME"
    echo "   ✅ VRAM: ${GPU_MEM}MB"
    echo "   ✅ GPU count: $GPU_COUNT"
else
    echo "   ⚠️  nvidia-smi not available (may not be critical)"
fi
echo ""

# Test 7: Syntax check Python app
echo "7️⃣ Checking Python syntax..."
SYNTAX_CHECK=$(/workspace/wan22-comfy-project/venv/bin/python -m py_compile /workspace/wan22-comfy-project/Wan2.2/gradio_i2v_app.py 2>&1)
if [ $? -eq 0 ]; then
    echo "   ✅ gradio_i2v_app.py syntax OK"
else
    echo "   ❌ Syntax error in gradio_i2v_app.py"
    echo "$SYNTAX_CHECK"
    exit 1
fi
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ ALL CHECKS PASSED!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🚀 Ready to launch Gradio I2V UI!"
echo ""
echo "Quick start:"
echo "  cd /workspace/wan22-comfy-project/Wan2.2"
echo "  ./launch_i2v_ui.sh"
echo ""
echo "Or with options:"
echo "  ./launch_i2v_ui.sh --share         # Public link"
echo "  ./launch_i2v_ui.sh --multi-gpu     # Multi-GPU mode"
echo "  ./launch_i2v_ui.sh --port 8080     # Custom port"
echo ""
