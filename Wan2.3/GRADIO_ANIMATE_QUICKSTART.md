# 🎭 Wan2.2 Animate Gradio Interface - Quick Start Guide

## ✅ Fixed Issues

The Gradio interface has been updated with the following fixes:

1. **Video Upload Error**: Fixed video file path handling (Gradio returns different formats)
2. **Examples Added**: Example pose video and reference image now appear in the UI
3. **Better Error Messages**: Detailed error reporting for preprocessing failures
4. **Debug Logging**: Added logging to help troubleshoot issues

## 🚀 How to Use

### Step 1: Launch the Interface

```bash
cd /workspace/wan22-comfy-project/Wan2.2
/workspace/wan22-comfy-project/venv/bin/python gradio_animate_app.py
```

The interface will be available at: `http://localhost:7863`

### Step 2: Setup (First Time Only)

1. Go to the **"Setup"** tab
2. Click **"Download Models Only"** or **"Load Model"**
   - This downloads ~50GB from HuggingFace (Wan-AI/Wan2.2-Animate-14B)
   - Models are cached at `/home/caches/Wan2.2-Animate-14B`

### Step 3: Preprocess Video

1. Go to the **"Preprocess"** tab
2. Either:
   - **Upload your own files**: Drag and drop video + reference image
   - **Use the example**: Click on the example below the inputs
3. Select mode:
   - **animate**: Character mimics the motion (default)
   - **replace**: Replace character in video (needs background segmentation)
4. Click **"Preprocess"**
5. Copy the output path from the results

### Step 4: Generate Animation

1. Go to the **"Generate"** tab
2. Paste the preprocessed path from Step 3
3. Adjust parameters (optional):
   - Frames: 77 (4n+1 format, e.g., 5, 9, 13, 17, 21, 77, 81)
   - Sampling steps: 20 (higher = better quality, slower)
   - Guidance scale: 1.0
   - Seed: -1 for random
4. Click **"Generate Video"**
5. Download the result!

## 📁 Example Files

Located in `/workspace/wan22-comfy-project/Wan2.2/examples/`:
- `pose.mp4` - Driving video (2.1MB)
- `pose.png` - Reference character image (804KB)

## 🔧 Troubleshooting

### Video Upload Shows "Error"

**Fixed!** The app now properly handles video file paths from Gradio. If you still see errors:

1. Check the terminal logs for DEBUG messages
2. Verify your video file is a valid format (MP4 recommended)
3. Try using the example files first

### Preprocessing Stuck

If preprocessing takes too long (>5 min), it will timeout. Check:

1. Models are downloaded (Setup tab)
2. Video resolution is reasonable (<1080p recommended)
3. Check logs: `tail -f /workspace/wan22-comfy-project/Wan2.2/gradio_animate.log`

### No Examples Showing

Examples should now appear at the bottom of the Preprocess tab. If not:

1. Verify example files exist:
   ```bash
   ls -lh /workspace/wan22-comfy-project/Wan2.2/examples/pose.*
   ```
2. Restart the Gradio app

## 🎯 Current Status

✅ Gradio interface created and running
✅ Video upload error fixed
✅ Examples integrated into UI
✅ Better error handling and debugging
✅ Model download integration
✅ Preprocessing pipeline connected
✅ Generation pipeline connected

## 📝 Notes

- **First run**: Downloads ~50GB of models
- **Preprocessing**: Takes 1-3 minutes depending on video length
- **Generation**: Takes 2-5 minutes for 77 frames on 2x GPU
- **Memory**: Requires ~40GB VRAM for full model
- **Replacement mode**: Requires additional preprocessing (background segmentation)

## 🚦 Running in Background

```bash
# Start in background
cd /workspace/wan22-comfy-project/Wan2.2
nohup /workspace/wan22-comfy-project/venv/bin/python gradio_animate_app.py > gradio_animate.log 2>&1 &

# Check logs
tail -f gradio_animate.log

# Stop
pkill -f gradio_animate_app.py
```

## 🔗 Resources

- Model: [Wan-AI/Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B)
- Repository: [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
- Venv: `/workspace/wan22-comfy-project/venv`

---

**Last Updated**: October 20, 2025
**Interface**: http://localhost:7863
