#!/bin/bash
# Wan2.2 Animate Gradio - Quick Commands

echo "🎭 Wan2.2 Animate Gradio Interface - Quick Commands"
echo ""
echo "════════════════════════════════════════════════════════"
echo ""

# Function to show menu
show_menu() {
    echo "Choose an action:"
    echo ""
    echo "  1) 🚀 Start Gradio (foreground)"
    echo "  2) 🌐 Start Gradio (background)"
    echo "  3) 🧪 Test preprocessing directly"
    echo "  4) 📊 Check status"
    echo "  5) 📋 View logs (live)"
    echo "  6) 🛑 Stop Gradio"
    echo "  7) ❌ Exit"
    echo ""
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice [1-7]: " choice
    echo ""
    
    case $choice in
        1)
            echo "🚀 Starting Gradio in foreground..."
            echo "   Press Ctrl+C to stop"
            echo ""
            cd /workspace/wan22-comfy-project/Wan2.2
            /workspace/wan22-comfy-project/venv/bin/python gradio_animate_app.py
            ;;
        2)
            echo "🌐 Starting Gradio in background..."
            cd /workspace/wan22-comfy-project/Wan2.2
            nohup /workspace/wan22-comfy-project/venv/bin/python gradio_animate_app.py > gradio_animate.log 2>&1 &
            sleep 3
            echo "   ✅ Started! Check status with option 4"
            echo "   📋 View logs with option 5"
            echo ""
            ;;
        3)
            echo "🧪 Testing preprocessing..."
            echo ""
            /workspace/wan22-comfy-project/venv/bin/python /workspace/wan22-comfy-project/Wan2.2/test_preprocessing_direct.py
            echo ""
            read -p "Press Enter to continue..."
            ;;
        4)
            echo "📊 Checking status..."
            echo ""
            /workspace/wan22-comfy-project/Wan2.2/test_gradio_status.sh
            echo ""
            read -p "Press Enter to continue..."
            ;;
        5)
            echo "📋 Viewing live logs (Ctrl+C to stop)..."
            echo ""
            tail -f /workspace/wan22-comfy-project/Wan2.2/gradio_animate.log 2>/dev/null || echo "   No log file yet. Start Gradio first (option 2)"
            echo ""
            ;;
        6)
            echo "🛑 Stopping Gradio..."
            pkill -f gradio_animate_app.py
            echo "   ✅ Stopped"
            echo ""
            ;;
        7)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid choice. Please enter 1-7."
            echo ""
            ;;
    esac
done
