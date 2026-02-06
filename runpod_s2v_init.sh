#!/bin/bash
# One-Command Setup for Wan2.2 S2V on RunPod GPU
# Usage: curl -sSL https://raw.githubusercontent.com/your-repo/setup.sh | bash
# Or: bash runpod_s2v_init.sh

set -e  # Exit on any error

echo "🚀 Wan2.2 S2V RunPod GPU Initialization Script"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE_DIR="/workspace/wan22-comfy-project"
CACHE_DIR="/home/caches"
VENV_DIR="$WORKSPACE_DIR/venv"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if we're in RunPod environment
check_environment() {
    print_step "Checking RunPod environment..."
    
    if [[ ! -d "/workspace" ]]; then
        print_warning "Not detected as RunPod environment"
    fi
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        print_status "GPU detected: $GPU_INFO"
    else
        print_error "No GPU detected!"
        exit 1
    fi
}

# Setup workspace if needed
setup_workspace() {
    print_step "Setting up workspace..."
    
    if [[ ! -d "$WORKSPACE_DIR" ]]; then
        print_status "Creating workspace directory..."
        mkdir -p "$WORKSPACE_DIR"
        cd "$WORKSPACE_DIR"
        
        # Clone or setup your project here
        # For now, we'll assume the workspace exists
        print_warning "Workspace created. Please ensure Wan2.2 code is available."
    else
        print_status "Workspace already exists"
    fi
    
    cd "$WORKSPACE_DIR"
}

# Setup Python virtual environment
setup_venv() {
    print_step "Setting up Python virtual environment..."
    
    if [[ ! -d "$VENV_DIR" ]]; then
        print_status "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
    
    # Activate venv
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install basic requirements
    print_status "Installing basic Python packages..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers accelerate safetensors huggingface_hub[cli]
    
    print_status "Virtual environment setup complete"
}

# Install system dependencies
install_system_deps() {
    print_step "Installing system dependencies..."
    
    # Update package list
    print_status "Updating package list..."
    apt update -qq
    
    # Install FFmpeg
    if ! command -v ffmpeg &> /dev/null; then
        print_status "Installing FFmpeg..."
        apt install -y ffmpeg
    else
        print_status "FFmpeg already installed"
    fi
    
    # Install other dependencies
    print_status "Installing additional system packages..."
    apt install -y wget curl git build-essential
}

# Run the cache setup script
run_cache_setup() {
    print_step "Running Wan2.2 S2V cache setup..."
    
    cd "$WORKSPACE_DIR/Wan2.2"
    
    # Make sure the setup script is executable
    chmod +x setup_s2v_cache.py
    
    # Run the setup
    print_status "Starting model download and cache setup..."
    print_warning "This may take 10-15 minutes depending on your connection..."
    
    source "$VENV_DIR/bin/activate"
    python setup_s2v_cache.py --quick-setup
    
    if [[ $? -eq 0 ]]; then
        print_status "Cache setup completed successfully!"
    else
        print_error "Cache setup failed!"
        exit 1
    fi
}

# Create convenience scripts
create_convenience_scripts() {
    print_step "Creating convenience scripts..."
    
    # Create run script
    cat > "$WORKSPACE_DIR/run_s2v.sh" << EOF
#!/bin/bash
# Quick S2V Generation Script
cd $WORKSPACE_DIR/Wan2.2
source $VENV_DIR/bin/activate
python run_s2v_multi_gpu.py "\$@"
EOF
    chmod +x "$WORKSPACE_DIR/run_s2v.sh"
    
    # Create status check script
    cat > "$WORKSPACE_DIR/check_s2v_status.sh" << EOF
#!/bin/bash
# Check S2V Status
cd $WORKSPACE_DIR/Wan2.2
source $VENV_DIR/bin/activate
python setup_s2v_cache.py --status
EOF
    chmod +x "$WORKSPACE_DIR/check_s2v_status.sh"
    
    print_status "Convenience scripts created:"
    print_status "  - $WORKSPACE_DIR/run_s2v.sh (run S2V generation)"
    print_status "  - $WORKSPACE_DIR/check_s2v_status.sh (check cache status)"
}

# Display final instructions
show_final_instructions() {
    echo ""
    echo "🎉 Setup Complete!"
    echo "=================="
    echo ""
    echo "Quick Start Commands:"
    echo "  # Check status:"
    echo "  cd $WORKSPACE_DIR && ./check_s2v_status.sh"
    echo ""
    echo "  # Run S2V generation:"
    echo "  cd $WORKSPACE_DIR && ./run_s2v.sh"
    echo ""
    echo "  # Or manually:"
    echo "  cd $WORKSPACE_DIR/Wan2.2"
    echo "  source $VENV_DIR/bin/activate"
    echo "  python run_s2v_multi_gpu.py"
    echo ""
    echo "Cache Location: $CACHE_DIR"
    echo "Models are cached and ready for use!"
    echo ""
    echo "Note: On RunPod, /home is volatile. Run this setup script"
    echo "      each time you start a new instance."
}

# Main execution
main() {
    print_status "Starting Wan2.2 S2V RunPod initialization..."
    
    check_environment
    setup_workspace
    install_system_deps
    setup_venv
    run_cache_setup
    create_convenience_scripts
    show_final_instructions
    
    print_status "All done! 🚀"
}

# Handle script interruption
trap 'print_error "Setup interrupted by user"; exit 1' INT

# Run main function
main "$@"