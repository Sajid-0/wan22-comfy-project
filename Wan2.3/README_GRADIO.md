# Wan2.2 Animate Gradio Interface

## 🚀 Quick Start

```bash
cd /workspace/wan22-comfy-project/Wan2.2
/workspace/wan22-comfy-project/venv/bin/python gradio_animate_app.py
```

## ✅ Fixed: No More "Video Not Playable" Error

**Changed Video component → File upload component**

Now you can:
1. Click "Use Example Files" button (loads pose.mp4 + pose.png automatically)
2. Or upload your own MP4 video file
3. No more confusing "Video not playable" errors!

## 📝 How to Use

### Step 1: Start Interface
Open browser → `http://localhost:7860` (or public URL shown in terminal)

### Step 2: Preprocess
1. Go to "Preprocess" tab
2. Click **"Use Example Files"** button
3. Click **"Start Preprocessing"**
4. Wait 1-3 minutes (watch Status textbox)
5. Copy output path when done

### Step 3: Generate
1. Go to "Generate" tab
2. Paste preprocessed path
3. Click "Generate Video"
4. Wait 2-5 minutes
5. Download result!

## 🔧 Main Changes

- ✅ Replaced `gr.Video()` with `gr.File()` - more reliable
- ✅ Added "Use Example Files" button - one click to load examples
- ✅ Better status messages and progress tracking
- ✅ Clearer UI with helpful placeholders

## 📁 Files

- `gradio_animate_app.py` - Main interface (working!)
- `examples/pose.mp4` - Example driving video
- `examples/pose.png` - Example reference character

That's it! Simple and functional.
