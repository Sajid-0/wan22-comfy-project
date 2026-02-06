# Wan2.2 S2V (Speech-to-Video) Parameter Reference

Complete guide to all available parameters for S2V-14B generation.

---

## 🎯 **Core Parameters**

### **Required Parameters**

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `--task` | str | Model task to run | `s2v-14B` |
| `--ckpt_dir` | str | Path to model checkpoint directory | `/home/caches/Wan2.2-S2V-14B` |
| `--prompt` | str | Text description of the video content | `"A woman speaking with expressive gestures"` |
| `--image` | str | Reference image path for face/character | `/path/to/image.jpg` |
| `--audio` | str | Audio file path (wav/mp3) | `/path/to/audio.wav` |

---

## 🎬 **Video Generation Parameters**

### **Frame Control**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--infer_frames` | int | `80` | **Frames per clip** (must be multiple of 4)<br>• 48 frames = ~3 sec @ 16fps<br>• 80 frames = ~5 sec @ 16fps<br>• 112 frames = ~7 sec @ 16fps |
| `--num_clip` | int | Auto | **Number of video clips** to generate<br>• If not set, auto-calculated based on audio length<br>• Each clip is `infer_frames` long<br>• Total video won't exceed audio duration |

### **Resolution Control**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--size` | str | `"1024*704"` | **Target area** (width × height)<br>Actual resolution may vary based on input image aspect ratio<br>**Common presets:**<br>• `"1280*720"` - 720p (HD)<br>• `"1024*704"` - Balanced quality<br>• `"960*544"` - 480p (faster) |
| `--max_area` | int | Auto | **Maximum pixel area** for resolution<br>• Not exposed in generate.py CLI<br>• Calculated from `--size` internally |

---

## 🎨 **Quality & Style Parameters**

### **Sampling Control**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sample_steps` | int | `40` | **Number of diffusion steps**<br>• More steps = higher quality, slower<br>• Recommended: 30-50<br>• Default from config: 40 |
| `--sample_guide_scale` | float | `4.5` | **Classifier-free guidance scale**<br>• Higher = follows prompt more strictly<br>• Lower = more creative/varied<br>• Range: 1.0-15.0<br>• Default from config: 4.5 |
| `--sample_shift` | float | `3.0` | **Noise schedule shift**<br>• Affects temporal dynamics<br>• For 480p: use 3.0<br>• For 720p+: use 5.0<br>• Default from config: 3.0 |
| `--sample_solver` | str | `"unipc"` | **Diffusion solver**<br>• `"unipc"` - Recommended, faster<br>• `"dpm++"` - Alternative solver |

### **Prompt Engineering**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--prompt` | str | Required | **Positive prompt** - describe desired content |
| `--sample_neg_prompt` | str | Config | **Negative prompt** - undesired content<br>Default (Chinese): 画面模糊，最差质量，情绪激动剧烈... |
| `--use_prompt_extend` | flag | `False` | Enable AI prompt expansion |
| `--prompt_extend_method` | str | `"local_qwen"` | Method: `"dashscope"` or `"local_qwen"` |
| `--prompt_extend_model` | str | None | Model path for prompt expansion |
| `--prompt_extend_target_lang` | str | `"zh"` | Target language: `"zh"` or `"en"` |

---

## 🎤 **Audio & TTS Parameters**

### **Audio Input**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--audio` | str | None | **Audio file path** (wav, mp3)<br>• Drives lip-sync and expressions<br>• Video length auto-matches audio |
| `--enable_tts` | flag | `False` | **Enable text-to-speech synthesis**<br>Uses CosyVoice to generate audio |

### **TTS (Text-to-Speech) Parameters**

Only used when `--enable_tts` is set:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--tts_text` | str | None | **Text to synthesize** into speech |
| `--tts_prompt_audio` | str | None | **Reference audio** for voice cloning<br>• Must be 16kHz+<br>• Duration: 5-15 seconds |
| `--tts_prompt_text` | str | None | **Transcript** of reference audio<br>• Must match `tts_prompt_audio` exactly |

**TTS Example:**
```bash
--enable_tts \
--tts_prompt_audio examples/zero_shot_prompt.wav \
--tts_prompt_text "希望你以后能够做的比我还好呦。" \
--tts_text "收到好友从远方寄来的生日礼物..."
```

---

## 🕺 **Motion & Pose Control**

### **Pose-Driven Generation**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--pose_video` | str | None | **DW-Pose sequence video**<br>• Provides body pose guidance<br>• Format: video file with pose data |
| `--start_from_ref` | flag | `False` | **Use reference image as first frame**<br>• `True` = standard image-to-video<br>• `False` = generate all frames |

---

## 🖥️ **Performance & Multi-GPU Parameters**

### **Distributed Training**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--ulysses_size` | int | `1` | **Sequence parallelism size**<br>• Must equal number of GPUs<br>• For 2 GPUs: `--ulysses_size 2` |
| `--dit_fsdp` | flag | `False` | **Enable FSDP for DiT model**<br>Shards DiT across GPUs |
| `--t5_fsdp` | flag | `False` | **Enable FSDP for T5 encoder**<br>Shards T5 across GPUs |

### **Memory Optimization**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--offload_model` | bool | Auto | **Offload to CPU between steps**<br>• Auto: `True` for single GPU, `False` for multi-GPU<br>• Saves VRAM but slower |
| `--convert_model_dtype` | flag | `False` | **Convert model to mixed precision**<br>Reduces memory usage |
| `--t5_cpu` | flag | `False` | **Keep T5 on CPU**<br>Saves GPU memory (slower encoding) |

---

## 🎲 **Reproducibility**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--base_seed` | int | `-1` | **Random seed** for generation<br>• `-1` = random seed<br>• Fixed value (e.g., `42`) = reproducible |

---

## 💾 **Output Control**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--save_file` | str | Auto | **Output filename**<br>• Auto-generates: `s2v-14B_1024x704_2_prompt_timestamp.mp4`<br>• Custom: `--save_file my_video.mp4` |

---

## 📋 **Example Command Variations**

### **Basic S2V (Current Setup)**
```bash
python generate.py \
  --task s2v-14B \
  --ckpt_dir /home/caches/Wan2.2-S2V-14B \
  --image prompt.png \
  --audio audio.wav \
  --prompt "A woman speaking with expressive gestures" \
  --size 1024*704 \
  --sample_steps 50 \
  --sample_guide_scale 7.5
```

### **High Quality (More Frames, Higher Resolution)**
```bash
python generate.py \
  --task s2v-14B \
  --ckpt_dir /home/caches/Wan2.2-S2V-14B \
  --image input.jpg \
  --audio speech.wav \
  --prompt "Detailed scene description" \
  --size 1280*720 \
  --infer_frames 112 \
  --sample_steps 60 \
  --sample_guide_scale 5.0 \
  --sample_shift 5.0
```

### **Fast Preview (Lower Quality)**
```bash
python generate.py \
  --task s2v-14B \
  --ckpt_dir /home/caches/Wan2.2-S2V-14B \
  --image input.jpg \
  --audio speech.wav \
  --prompt "Quick preview" \
  --size 960*544 \
  --infer_frames 48 \
  --sample_steps 25 \
  --sample_guide_scale 4.0 \
  --sample_shift 3.0
```

### **Multi-Clip Long Video**
```bash
python generate.py \
  --task s2v-14B \
  --ckpt_dir /home/caches/Wan2.2-S2V-14B \
  --image input.jpg \
  --audio long_speech.wav \
  --prompt "Person delivering a speech" \
  --size 1024*704 \
  --infer_frames 80 \
  --num_clip 5 \
  --sample_steps 40
```

### **With TTS (Synthesized Speech)**
```bash
python generate.py \
  --task s2v-14B \
  --ckpt_dir /home/caches/Wan2.2-S2V-14B \
  --image input.jpg \
  --enable_tts \
  --tts_prompt_audio examples/zero_shot_prompt.wav \
  --tts_prompt_text "Reference speech text" \
  --tts_text "Text to synthesize into speech" \
  --prompt "Character speaking synthesized text" \
  --size 1024*704
```

### **Pose-Driven Generation**
```bash
python generate.py \
  --task s2v-14B \
  --ckpt_dir /home/caches/Wan2.2-S2V-14B \
  --image input.jpg \
  --audio speech.wav \
  --pose_video pose_sequence.mp4 \
  --prompt "Person with specific body movements" \
  --size 1024*704 \
  --start_from_ref
```

### **Multi-GPU (2x A40) - RECOMMENDED**
```bash
torchrun --nproc_per_node=2 generate.py \
  --task s2v-14B \
  --ckpt_dir /home/caches/Wan2.2-S2V-14B \
  --image input.jpg \
  --audio speech.wav \
  --prompt "Description" \
  --size 1024*704 \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 2 \
  --sample_steps 50
```

---

## 🎯 **Quick Reference: Most Useful Parameters**

For daily use, these are the most important parameters to adjust:

1. **Frame Count**: `--infer_frames 80` (48, 80, 112...)
2. **Quality**: `--sample_steps 40` (25-60)
3. **Prompt Adherence**: `--sample_guide_scale 5.0` (3.0-10.0)
4. **Resolution**: `--size 1024*704` (or 1280*720)
5. **Clips**: `--num_clip 3` (for longer videos)
6. **Seed**: `--base_seed 42` (for reproducibility)

---

## 🚀 **Performance Tips**

**For Speed:**
- Lower `--sample_steps` (25-30)
- Smaller `--size` (960*544)
- Fewer `--infer_frames` (48)
- Use `--sample_shift 3.0`

**For Quality:**
- Higher `--sample_steps` (50-60)
- Larger `--size` (1280*720)
- More `--infer_frames` (112)
- Use `--sample_shift 5.0`

**For Memory Savings:**
- Add `--offload_model True`
- Add `--convert_model_dtype`
- Add `--t5_cpu`
- Lower `--infer_frames`

---

## 📊 **Default Config Values** (from wan_s2v_14B.py)

```python
sample_fps = 16              # Output video FPS
motion_frames = 73           # Internal motion guidance frames
sample_steps = 40            # Diffusion sampling steps
sample_guide_scale = 4.5     # CFG scale
sample_shift = 3             # Noise schedule shift
```

These are used when parameters are not explicitly provided.
