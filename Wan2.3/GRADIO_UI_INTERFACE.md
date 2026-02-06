# Gradio I2V UI - Interface Overview

## 🎬 Main Interface Layout

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                  Wan2.2-I2V-A14B: Image-to-Video Generator                ║
║              Transform static images into dynamic videos with AI          ║
║              Powered by Alibaba's Wan2.2 MoE (27B params, 14B active)     ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────┬─────────────────────────────────┐
│  📸 INPUT CONFIGURATION         │   🎥 GENERATED VIDEO            │
│                                 │                                 │
│  ┌───────────────────────────┐ │   ┌───────────────────────────┐ │
│  │                           │ │   │                           │ │
│  │   Upload Image            │ │   │    [Video Player]         │ │
│  │   [Drag & Drop Area]      │ │   │                           │ │
│  │   or Click to Browse      │ │   │    Auto-plays when        │ │
│  │                           │ │   │    generation complete    │ │
│  └───────────────────────────┘ │   │                           │ │
│                                 │   │    [Download Button]      │ │
│  Prompt (describe motion):      │   └───────────────────────────┘ │
│  ┌───────────────────────────┐ │                                 │
│  │ Summer beach vacation     │ │   Status:                       │
│  │ style, a white cat...     │ │   ✅ Generation Complete!       │
│  │                           │ │   • Output: i2v_720P_81f_...   │
│  │                           │ │   • Resolution: 720P            │
│  └───────────────────────────┘ │   • Frames: 81                  │
│                                 │   • Steps: 40                   │
│  ⚙️ Generation Settings         │   • Seed: 42                    │
│  ┌───────────────────────────┐ │   • File: 45.2 MB              │
│  │ Resolution Preset:        │ │                                 │
│  │ ⦿ 480P (480×832)          │ │                                 │
│  │ ○ 720P (720×1280)         │ │   📚 Example Prompts            │
│  │                           │ │   ┌───────────────────────────┐ │
│  │ Quality Preset:           │ │   │ Example 1 - Beach Cat:    │ │
│  │ ○ Draft (Fast)            │ │   │ "Summer beach vacation    │ │
│  │ ⦿ Standard                │ │   │  style, a white cat..."   │ │
│  │ ○ High Quality            │ │   │                           │ │
│  │                           │ │   │ Example 2 - Portrait:     │ │
│  │ Preset Info:              │ │   │ "A young woman with..."   │ │
│  │ Resolution: Balanced      │ │   └───────────────────────────┘ │
│  │ Quality: Default quality  │ │                                 │
│  │                           │ │   🔧 Model Management           │
│  │ Frame Count: [   81   ]  │ │   ┌───────────────────────────┐ │
│  │              ├──────────┤ │ │   │ Model Status:             │ │
│  │              49      161  │ │   │ ✅ Model loaded           │ │
│  │                           │ │   │                           │ │
│  │ Seed: [  -1  ] (random)  │ │   │ [Check Models] [Load]     │ │
│  └───────────────────────────┘ │   └───────────────────────────┘ │
│                                 │                                 │
│  🔧 Advanced Settings           │                                 │
│  ┌───────────────────────────┐ │                                 │
│  │ ☐ Enable Advanced Controls│ │                                 │
│  │                           │ │                                 │
│  │ Sampling Solver:          │ │                                 │
│  │ ⦿ unipc  ○ dpm++          │ │                                 │
│  │                           │ │                                 │
│  │ ☑ Offload Model (VRAM)    │ │                                 │
│  └───────────────────────────┘ │                                 │
│                                 │                                 │
│  ┌───────────────────────────┐ │                                 │
│  │   🎬 Generate Video       │ │                                 │
│  └───────────────────────────┘ │                                 │
│                                 │                                 │
└─────────────────────────────────┴─────────────────────────────────┘

💡 Tips: Start with 720P + Standard quality | Use 480P for faster previews
📝 Frame Count: Must be 4n+1 (49, 81, 105) | More frames = longer video
🔗 Model: Wan2.2-I2V-A14B | GitHub
```

---

## 🔧 Advanced Mode Interface

When "Enable Advanced Controls" is checked:

```
┌─────────────────────────────────────────────────────────────────┐
│  🔧 Advanced Settings (Enabled)                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ☑ Enable Advanced Controls                              │   │
│  │                                                          │   │
│  │ Sampling Steps: [    40    ]  ← 10-100                  │   │
│  │                 ├──────────┤                            │   │
│  │                 10       100                            │   │
│  │                                                          │   │
│  │ Shift Value: [   5.0   ]  ← 1.0-10.0                    │   │
│  │              ├──────────┤                               │   │
│  │              1.0      10.0                              │   │
│  │              (Use 3.0 for 480P, 5.0 for 720P)           │   │
│  │                                                          │   │
│  │ Guide Scale (Low Noise Model): [  3.5  ]                │   │
│  │                                 ├──────┤                │   │
│  │                                 1.0  10.0               │   │
│  │                                                          │   │
│  │ Guide Scale (High Noise Model): [  3.5  ]               │   │
│  │                                  ├──────┤               │   │
│  │                                  1.0  10.0              │   │
│  │                                                          │   │
│  │ Sampling Solver:                                        │   │
│  │ ⦿ unipc (Recommended)    ○ dpm++ (Better quality)       │   │
│  │                                                          │   │
│  │ ☑ Offload Model (saves VRAM)                            │   │
│  │   Enable if running out of memory                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Generation Process Flow

```
User Uploads Image
       ↓
User Enters Prompt
       ↓
User Configures Settings
(Resolution, Quality, Frames, Seed)
       ↓
User Clicks "Generate Video"
       ↓
┌──────────────────────────────┐
│ System Checks:               │
│ ✓ Model loaded?              │
│ ✓ Valid image?               │
│ ✓ Valid prompt?              │
│ ✓ Valid frame count (4n+1)?  │
└──────────────────────────────┘
       ↓
┌──────────────────────────────┐
│ Preprocessing (10%)          │
│ • Process image              │
│ • Prepare inputs             │
│ • Set random seed            │
└──────────────────────────────┘
       ↓
┌──────────────────────────────┐
│ Generation (20%-90%)         │
│ • Encode text with T5        │
│ • Encode image with VAE      │
│ • Diffusion process          │
│   - High noise model (>90%)  │
│   - Low noise model (<90%)   │
│ • Sample video frames        │
└──────────────────────────────┘
       ↓
┌──────────────────────────────┐
│ Saving (90%-100%)            │
│ • Decode latents with VAE    │
│ • Save to MP4                │
│ • Generate filename          │
└──────────────────────────────┘
       ↓
Video Displayed in UI
User Can Download
```

---

## 🎨 UI Color Coding & Icons

| Element | Icon/Color | Meaning |
|---------|------------|---------|
| ✅ | Green checkmark | Success / Complete |
| ❌ | Red X | Error / Failed |
| ⚠️ | Yellow warning | Warning / Attention needed |
| 🔍 | Magnifying glass | Checking / Searching |
| 📥 | Download arrow | Downloading |
| 🎬 | Movie camera | Generate action |
| ⚙️ | Gear | Settings |
| 💡 | Light bulb | Tips / Information |
| 🔧 | Wrench | Advanced controls |
| 📸 | Camera | Input image |
| 🎥 | Video camera | Output video |

---

## 🖱️ Interactive Elements

### Drag & Drop Image Upload
- Hover: Border highlights in blue
- Drag over: Border highlights in green
- Drop: Image immediately loads and displays

### Sliders (Frame Count, Steps, etc.)
- Click and drag handle
- Click anywhere on track to jump
- Keyboard arrows for fine control
- Displays current value dynamically

### Radio Buttons (Presets)
- Single selection
- Click to select
- Auto-updates dependent fields
- Shows description on selection

### Checkboxes (Advanced Mode, Offload)
- Click to toggle
- ☐ Unchecked (off)
- ☑ Checked (on)
- Some controls show/hide based on state

### Buttons
- **Primary (Blue)**: Main actions (Generate Video)
- **Secondary (Gray)**: Utility actions (Check Models)
- **Small**: Non-critical actions in accordions

### Video Player
- Auto-plays when generation complete
- Controls: Play/Pause, Fullscreen, Download
- Scrub timeline to any point
- Shows thumbnail when paused

---

## 📱 Mobile Responsive Design

### Desktop View (>1024px)
```
┌─────────────────────────────────────────────────────────┐
│                      Header                             │
├──────────────────────────┬──────────────────────────────┤
│                          │                              │
│    Input Controls        │    Output & Preview          │
│    (Left Column)         │    (Right Column)            │
│                          │                              │
└──────────────────────────┴──────────────────────────────┘
```

### Tablet View (768px-1024px)
```
┌──────────────────────────────────────────────────┐
│                   Header                         │
├──────────────────────────────────────────────────┤
│          Input Controls                          │
│          (Full Width)                            │
├──────────────────────────────────────────────────┤
│          Output & Preview                        │
│          (Full Width)                            │
└──────────────────────────────────────────────────┘
```

### Mobile View (<768px)
```
┌────────────────────┐
│      Header        │
├────────────────────┤
│  Input Controls    │
│  (Stacked)         │
│  • Image upload    │
│  • Prompt          │
│  • Settings        │
├────────────────────┤
│  Output            │
│  (Stacked)         │
│  • Video player    │
│  • Status          │
└────────────────────┘
```

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+V` | Paste image from clipboard |
| `Tab` | Navigate between fields |
| `Enter` | Submit/Generate (when in text field) |
| `Esc` | Close expanded accordions |
| `Space` | Play/Pause video (when focused) |
| `F` | Fullscreen video (when focused) |

---

## 🔔 Notification System

### Success Messages (Green)
```
✅ Generation Complete!
   • Output: i2v_720P_81f_20251016_143025_beach.mp4
   • Resolution: 720P
   • Frames: 81
   • Steps: 40
   • Seed: 42
   • File size: 45.2 MB
```

### Error Messages (Red)
```
❌ CUDA Out of Memory!
   Try: Lower resolution, fewer frames, or enable 'Offload Model'
```

### Warning Messages (Yellow)
```
⚠️ Frame count must be 4n+1
   Valid values: 49, 53, 57, 61, 65, 69, 73, 77, 81, ...
```

### Info Messages (Blue)
```
🔍 Checking if I2V models are ready...
```

---

## 🎯 User Flow Examples

### Beginner Flow (Minimal Interaction)
1. Launch UI → Opens automatically
2. Click "Load Model" → Wait 30 seconds
3. Upload image → Drag & drop
4. Keep default settings → Already optimal
5. Click "Generate Video" → Wait 6-8 minutes
6. Download → Click download icon

**Total steps**: 6 clicks, 7-9 minutes

### Advanced User Flow (Full Control)
1. Launch UI with auto-load
2. Upload image
3. Write custom prompt
4. Select 720P resolution
5. Enable Advanced Mode
6. Adjust sampling steps to 60
7. Adjust guide scales to (4.5, 4.5)
8. Set specific seed (42)
9. Generate
10. Download

**Total steps**: 10+ interactions, full customization

### Batch Testing Flow (Multiple Seeds)
1. Load model once
2. Upload image once
3. Write prompt once
4. Generate with seed=42
5. Download result #1
6. Change seed to 123
7. Generate again
8. Download result #2
9. Repeat for seeds 456, 789...

**Efficiency**: Model stays loaded, only change seed

---

## 📈 Progress Indicators

### Loading Model
```
🔄 Loading Wan2.2-I2V-A14B model...
[▰▰▰▰▰▰▰▰▰▱▱▱▱▱▱▱▱▱▱▱] 50%
```

### Generating Video
```
🎬 Generating video...
[▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▱▱▱▱] 80% - Diffusion sampling
```

### Saving
```
💾 Saving video...
[▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰] 100% - Complete!
```

---

## 🎨 Theme & Styling

### Colors
- **Primary**: Blue (`#3B82F6`) - Action buttons, links
- **Success**: Green (`#10B981`) - Success messages
- **Warning**: Yellow (`#F59E0B`) - Warnings
- **Error**: Red (`#EF4444`) - Error messages
- **Neutral**: Gray (`#6B7280`) - Secondary elements

### Typography
- **Headers**: Bold, larger size
- **Body**: Regular weight, readable size
- **Monospace**: Code, filenames, paths
- **Emphasis**: Italic for tips, bold for importance

### Spacing
- Generous padding for touch targets
- Clear visual hierarchy
- Grouped related controls
- Breathing room between sections

---

## 🔄 State Management

### Model States
- **Not Loaded**: Gray, "Load Model" button enabled
- **Loading**: Blue spinner, button disabled
- **Loaded**: Green checkmark, "Generate" enabled
- **Error**: Red X, error message shown

### Generation States
- **Idle**: Ready to generate
- **Validating**: Checking inputs
- **Generating**: Progress bar active
- **Saving**: Final processing
- **Complete**: Video displayed
- **Error**: Error message shown

### UI Element States
- **Disabled**: Grayed out, not clickable (when model not loaded)
- **Enabled**: Normal colors, interactive
- **Active**: Highlighted (selected radio button)
- **Hover**: Slightly lighter/darker on mouse over
- **Focus**: Blue outline when keyboard navigating

---

This interface design prioritizes:
1. **Ease of use** - Minimal clicks for common tasks
2. **Discoverability** - Clear labels and tooltips
3. **Flexibility** - Advanced mode for power users
4. **Feedback** - Clear status at every step
5. **Aesthetics** - Clean, modern design
6. **Accessibility** - Keyboard navigation, color contrast
