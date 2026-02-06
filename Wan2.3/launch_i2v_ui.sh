#!/bin/bash
# Quick launcher for Gradio I2V Web UI
# Usage: ./launch_i2v_ui.sh [OPTIONS]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
AUTO_LOAD=true
SHARE=false
MULTI_GPU=false
PORT=7860
SERVER_NAME="0.0.0.0"

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════╗"
    echo "║                                                       ║"
    echo "║     🎬 Wan2.2-I2V-A14B Gradio Web UI Launcher 🎬     ║"
    echo "║                                                       ║"
    echo "║            Image-to-Video Generation                  ║"
    echo "║                                                       ║"
    echo "╚═══════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print help
print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --no-auto-load       Don't auto-load model on startup"
    echo "  --share              Create public shareable link"
    echo "  --multi-gpu          Enable multi-GPU mode (2+ GPUs)"
    echo "  --port PORT          Set server port (default: 7860)"
    echo "  --localhost          Listen only on localhost (not 0.0.0.0)"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                              # Standard launch"
    echo "  $0 --share                      # With public link"
    echo "  $0 --multi-gpu                  # Multi-GPU mode"
    echo "  $0 --port 8080 --share          # Custom port + sharing"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-auto-load)
            AUTO_LOAD=false
            shift
            ;;
        --share)
            SHARE=true
            shift
            ;;
        --multi-gpu)
            MULTI_GPU=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --localhost)
            SERVER_NAME="localhost"
            shift
            ;;
        --help|-h)
            print_banner
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

print_banner

# Check if we're in the right directory
if [ ! -f "gradio_i2v_app.py" ]; then
    echo -e "${RED}❌ Error: gradio_i2v_app.py not found!${NC}"
    echo -e "${YELLOW}Please run this script from /workspace/wan22-comfy-project/Wan2.2${NC}"
    exit 1
fi

# Check Python - use the venv from parent directory
VENV_PYTHON="/workspace/wan22-comfy-project/venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}❌ Error: Virtual environment not found!${NC}"
    echo -e "${YELLOW}Please run: source /workspace/wan22-comfy-project/venv/bin/activate${NC}"
    exit 1
fi

PYTHON="$VENV_PYTHON"

# Check Gradio installation
echo -e "${BLUE}🔍 Checking Gradio installation...${NC}"
if ! $PYTHON -c "import gradio" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Gradio not found. Installing...${NC}"
    $PYTHON -m pip install gradio>=4.0.0 || {
        echo -e "${RED}❌ Failed to install Gradio${NC}"
        exit 1
    }
    echo -e "${GREEN}✅ Gradio installed successfully${NC}"
else
    GRADIO_VERSION=$($PYTHON -c "import gradio; print(gradio.__version__)")
    echo -e "${GREEN}✅ Gradio ${GRADIO_VERSION} found${NC}"
fi

# Build command
CMD="$PYTHON gradio_i2v_app.py --server_name $SERVER_NAME --server_port $PORT"

if [ "$AUTO_LOAD" = true ]; then
    CMD="$CMD --auto_load_model"
fi

if [ "$SHARE" = true ]; then
    CMD="$CMD --share"
fi

if [ "$MULTI_GPU" = true ]; then
    CMD="$CMD --use_multi_gpu"
fi

# Display configuration
echo -e "${BLUE}⚙️  Configuration:${NC}"
echo -e "   Server: ${GREEN}${SERVER_NAME}:${PORT}${NC}"
echo -e "   Auto-load model: ${GREEN}${AUTO_LOAD}${NC}"
echo -e "   Public sharing: ${GREEN}${SHARE}${NC}"
echo -e "   Multi-GPU: ${GREEN}${MULTI_GPU}${NC}"
echo ""

# Check GPU
echo -e "${BLUE}🎮 GPU Status:${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | while IFS=, read -r idx name total free; do
        echo -e "   GPU ${idx}: ${GREEN}${name}${NC} | Total: ${total}MB | Free: ${free}MB"
    done
else
    echo -e "${YELLOW}⚠️  nvidia-smi not available${NC}"
fi
echo ""

# Launch
echo -e "${GREEN}🚀 Launching Gradio Web UI...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Run the command
$CMD

# Cleanup (if we get here, user pressed Ctrl+C)
echo ""
echo -e "${YELLOW}👋 Shutting down...${NC}"
exit 0
