# Wan2.2 Animate Gradio Interface

## 🎭 Quick Start Guide

### Launch the UI

```bash
cd /workspace/wan22-comfy-project/Wan2.2
./launch_animate_ui.sh
```

Or manually:

```bash
/workspace/wan22-comfy-project/venv/bin/python gradio_animate_app.py
```

The interface will be available at: **http://0.0.0.0:7862**

---

## 📋 How to Use

### Step 1: Setup (First Time Only)
1. Click **"Load Model"** button
2. Wait for models to download (~50GB, one-time)
3. Model will be cached in `/home/caches/Wan2.2-Animate-14B`

### Step 2: Preprocess Your Video
1. Upload **Driving Video** (the pose/motion source)
2. Upload **Reference Image** (your character)
3. Select mode:
   - **Animate**: Character mimics the motion
   - **Replace**: Character replaces person in video
4. Click **"Preprocess"**
5. Copy the output path for next step

### Step 3: Generate Animation
1. Paste the preprocessed path from Step 2
2. Adjust parameters:
   - **Frames**: More = longer video (77 = ~2.5s @ 30fps)
   - **Steps**: Higher = better quality (slower)
   - **Seed**: -1 for random, or use specific number
3. Click **"Generate Video"**
4. Download your result!

---

## 🎨 Example Files

Test with provided examples:

- **Video**: `/workspace/wan22-comfy-project/Wan2.2/examples/pose.mp4`
- **Image**: `/workspace/wan22-comfy-project/Wan2.2/examples/pose.png`

---

## ⚙️ Parameters Explained

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| **Frames** | Number of frames (must be 4n+1) | 77 frames |
| **Sampling Steps** | Quality vs speed tradeoff | 20 steps |
| **Guidance Scale** | Expression control (>1 slower) | 1.0 |
| **Seed** | Reproducibility (-1 = random) | -1 |

---

## 🔧 Modes

### Animate Mode
- Character **mimics motion** from driving video
- Only needs: video + reference image
- Faster preprocessing

### Replace Mode  
- Character **replaces person** in video
- Needs background extraction
- Requires **"Use Relighting LoRA"** in Setup
- Slower but more realistic integration

---

## 💡 Tips

1. **First run** takes time for model download
2. **Preprocessing** is required before generation
3. **Resolution** is automatically adjusted to 1280x720 area
4. **Guide scale > 1.0** gives expression control but is slower
5. **Reuse preprocessed data** for different generations

---

## 📁 Output Locations

- **Preprocessed Data**: `/workspace/wan22-comfy-project/Wan2.2/preprocessed/`
- **Generated Videos**: `/workspace/wan22-comfy-project/outputs/`

---

## 🐛 Troubleshooting

**Models not downloading?**
- Check HuggingFace token in `setup_animate_cache.py`
- Manually run: `python setup_animate_cache.py quick`

**Preprocessing fails?**
- Ensure video has clear person/face
- Try different video or reference image
- Check CUDA/GPU availability

**Out of memory?**
- Reduce frame count
- Close other GPU processes
- Enable T5 CPU offloading (default in UI)

---

## 📖 References

- Model: [Wan-AI/Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B)
- Paper: [Wan2.2 Technical Report](https://arxiv.org/pdf/2502.06145)

---

**Happy Animating! 🎬**
