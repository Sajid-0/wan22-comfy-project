# Wan2.2 S2V Production System 🚀

**High-performance, scalable Speech-to-Video generation system for production environments**

## ✨ Features

### 🎯 Core Capabilities
- **Multi-modal Input**: Speech + Text + Image → Video
- **Production-Ready**: FastAPI + Ray Serve + Multi-GPU
- **High Performance**: Optimized memory usage and GPU distribution
- **Scalable**: Docker containerization and load balancing
- **Monitoring**: Real-time system and service monitoring

### 🎬 Supported Inputs/Outputs
- **Inputs**: MP3/WAV audio, JPEG/PNG images, text prompts
- **Output**: MP4 videos up to 1280x720, 81 frames
- **Quality Levels**: Fast (320p), Medium (480p), High (720p)

## 🚀 Quick Start

### Option 1: Direct Deployment
```bash
# Clone and setup
git clone https://github.com/Wan-Video/Wan2.2.git
cd wan22-comfy-project

# Deploy production system
./deploy_production.sh
```

### Option 2: Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps
```

## 📡 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Generate Video (Multi-modal)
```bash
# Using production client
python production_client.py \
  --generate \
  --prompt "A cat playing piano in a cozy room" \
  --audio speech.mp3 \
  --image reference.jpg \
  --quality medium \
  --output generated_video.mp4
```

### REST API
```bash
curl -X POST "http://localhost:8000/generate" \
  -F "prompt=A cat playing piano" \
  -F "audio_file=@speech.mp3" \
  -F "image_file=@image.jpg" \
  -F "quality=medium"
```

## 🏗️ Architecture

### System Components
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Ray Serve      │    │   GPU Cluster   │
│   (Port 8000)   │◄──►│   (Port 8001)    │◄──►│   Multi-GPU     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Model Cache    │    │   Monitoring    │
│   (Nginx)       │    │   (/home/caches) │    │   (Port 8265)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Multi-GPU Distribution
- **1 GPU**: All components on cuda:0
- **2 GPUs**: Text/Audio → GPU 0, Transformer/VAE → GPU 1
- **3+ GPUs**: Text/Audio → GPU 0, Transformer → GPU 1, VAE → GPU 2+

## 🎛️ Configuration

### Quality Settings
```python
QUALITY_PRESETS = {
    "fast": {
        "resolution": "320x512",
        "frames": 16,
        "steps": 20,
        "time": "~30s"
    },
    "medium": {
        "resolution": "480x640", 
        "frames": 25,
        "steps": 30,
        "time": "~60s"
    },
    "high": {
        "resolution": "720x1280",
        "frames": 81, 
        "steps": 50,
        "time": "~120s"
    }
}
```

### Environment Variables
```bash
# Cache Configuration
HF_HOME=/home/caches/huggingface
DIFFUSERS_CACHE=/home/caches/diffusers
TRANSFORMERS_CACHE=/home/caches/transformers

# GPU Configuration  
CUDA_VISIBLE_DEVICES=0,1,2,3

# Production Settings
WORKERS=1
LOG_LEVEL=INFO
```

## 📊 Monitoring

### Real-time Dashboard
```bash
python production_monitor.py
```

### Metrics Available
- **System**: CPU, RAM, Disk, GPU utilization
- **Service**: Health status, response times, error rates
- **Model**: Generation times, queue length, cache usage

### Logging
- **Application Logs**: `/workspace/wan22-comfy-project/logs/production.log`
- **Access Logs**: Included in FastAPI/Uvicorn output
- **Ray Logs**: Available via Ray Dashboard

## 🔧 Production Optimization

### Memory Management
- **Attention Slicing**: Reduces peak GPU memory
- **VAE Slicing**: Handles high-resolution outputs
- **Model Offloading**: CPU fallback for memory constraints
- **Batch Processing**: Efficient multi-request handling

### Performance Tuning
```python
# GPU Memory Optimization
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

# Multi-GPU Distribution
device_map = {
    "text_encoder": "cuda:0",
    "audio_encoder": "cuda:0", 
    "transformer": "cuda:1",
    "vae": "cuda:2"
}
```

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t wan2.2-s2v-production .
```

### Run Container
```bash
docker run -d \
  --name wan2.2-s2v \
  --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8265:8265 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  wan2.2-s2v-production
```

### Scale with Compose
```bash
docker-compose up -d --scale wan-s2v-production=3
```

## 🧪 Testing

### Component Test
```bash
python component_test.py
```

### Integration Test  
```bash
python real_s2v_test.py
```

### Load Testing
```bash
# Using production client
for i in {1..10}; do
  python production_client.py \
    --generate \
    --prompt "Test video $i" \
    --quality fast &
done
```

## 📈 Scaling

### Horizontal Scaling
- **Multiple Replicas**: Scale Ray Serve deployments
- **Load Balancing**: Nginx upstream configuration
- **Queue Management**: Redis-based job queuing

### Vertical Scaling  
- **GPU Memory**: Increase VRAM for higher quality
- **CPU Cores**: Parallel processing for multiple requests
- **Storage**: Fast SSD for model cache and outputs

## 🛠️ Development

### Local Development
```bash
# Install development dependencies
pip install -r requirements.txt

# Run in development mode
python production_server.py --reload
```

### API Documentation
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI**: http://localhost:8000/openapi.json

## 📋 Production Checklist

### Pre-deployment
- [ ] GPU drivers and CUDA installed
- [ ] Sufficient disk space (50GB+ for models)
- [ ] Network connectivity for model downloads
- [ ] SSL certificates configured (if HTTPS)

### Post-deployment
- [ ] Health checks passing
- [ ] Monitoring dashboard accessible
- [ ] Log rotation configured
- [ ] Backup strategy implemented
- [ ] Scaling policies defined

## 🔒 Security

### API Security
- **Rate Limiting**: Prevent abuse
- **Authentication**: API key/JWT tokens  
- **Input Validation**: File type/size limits
- **CORS**: Controlled cross-origin access

### Infrastructure Security
- **Firewall**: Restrict port access
- **SSL/TLS**: Encrypt API traffic
- **Container Security**: Non-root user, readonly filesystem
- **Secret Management**: Environment-based configuration

## 📞 Support

### Troubleshooting
1. **Check Health Endpoint**: `curl http://localhost:8000/health`
2. **View Logs**: `tail -f /workspace/wan22-comfy-project/logs/production.log`
3. **Monitor GPU**: `nvidia-smi` or production dashboard
4. **Ray Dashboard**: http://localhost:8265

### Common Issues
- **CUDA OOM**: Reduce quality settings or batch size
- **Slow Generation**: Check GPU utilization and model distribution
- **Import Errors**: Verify all dependencies installed
- **Port Conflicts**: Ensure ports 8000, 8001, 8265 available

---

## 🎉 Production Ready!

Your Wan2.2 S2V system is now ready for production workloads with:
- ✅ **Multi-modal Input Processing** (Speech + Image + Text)
- ✅ **Multi-GPU Optimization** 
- ✅ **Scalable Architecture**
- ✅ **Production Monitoring**
- ✅ **Docker Containerization**
- ✅ **REST API Interface**

**🚀 Deploy and start generating videos at scale!** 🎬
- **Optimized for S2V**: Specifically tuned for Speech-to-Video generation tasks

## 🔧 Architecture

### Model Component Distribution

For optimal GPU utilization, the system distributes model components as follows:

- **2 GPUs**: Text Encoder + Transformer on GPU 0, VAE on GPU 1
- **3 GPUs**: Text Encoder on GPU 0, Transformer on GPU 1, VAE on GPU 2
- **4+ GPUs**: Text Encoder on GPU 0, Transformer on GPU 1, Transformer_2 on GPU 2, VAE on GPU 3

### Memory Management

- **Attention Slicing**: Reduces peak memory usage during attention computation
- **VAE Slicing**: Processes VAE in chunks to handle high-resolution outputs
- **Context Managers**: Automatic GPU memory cleanup after each generation
- **Dynamic Batching**: Adjusts batch size based on available GPU memory

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone/navigate to project directory
cd /workspace/wan22-comfy-project

# Activate virtual environment (if using)
source venv/bin/activate

# Setup cache directories and pre-download model (optional)
python cache_manager.py --setup --download

# Run setup script
./setup.sh
```

### 2. Start Server

```bash
# Start optimized server (recommended)
python optimized_wan_server.py --host 0.0.0.0 --port 8000

# Or start basic server
python multi_gpu_wan_server.py --host 0.0.0.0 --port 8000
```

### 3. Test Generation

```bash
# Using the client
python wan_client.py --prompt "A cat playing piano in a cozy room" --frames 25

# Using curl
curl -X POST "http://localhost:8000/OptimizedWanS2VServer" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing piano",
    "height": 720,
    "width": 1280,
    "num_frames": 81,
    "guidance_scale": 5.0
  }'
```

## 📚 API Reference

### Generation Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text/speech description for video generation |
| `negative_prompt` | string | null | What to avoid in the generated video |
| `height` | integer | 720 | Video height in pixels (must be divisible by 16) |
| `width` | integer | 1280 | Video width in pixels (must be divisible by 16) |
| `num_frames` | integer | 81 | Number of frames to generate |
| `guidance_scale` | float | 5.0 | Classifier-free guidance scale |
| `num_inference_steps` | integer | 50 | Number of denoising steps |
| `seed` | integer | null | Random seed for reproducible generation |

### Response Format

```json
{
  "status": "success",
  "num_frames": 81,
  "height": 720,
  "width": 1280,
  "generation_params": {
    "prompt": "A cat playing piano",
    "guidance_scale": 5.0
  }
}
```

## ⚡ Performance Optimization

### Recommended Settings

- **For 720p (recommended)**: `flow_shift=5.0`
- **For 480p**: `flow_shift=3.0`
- **Memory Optimization**: Use `torch.bfloat16` for reduced memory usage
- **Batch Size**: Automatically calculated based on GPU memory

### Multi-GPU Efficiency

The system achieves optimal performance by:

1. **Component Parallelism**: Different model components run on different GPUs
2. **Memory Balancing**: Heavy components (VAE) placed on separate GPUs
3. **Pipeline Parallelism**: Overlapping computation across GPUs
4. **Asynchronous Processing**: Non-blocking request handling

### Expected Performance

| Setup | Resolution | Frames | Approx. Time |
|-------|------------|--------|--------------|
| 1x A100 | 720p | 81 | 45-60s |
| 2x A100 | 720p | 81 | 25-35s |
| 4x A100 | 720p | 81 | 15-25s |

*Times vary based on prompt complexity and model settings*

## 🔍 Monitoring

### Ray Dashboard

Access the Ray dashboard at `http://localhost:8265` to monitor:
- GPU utilization across devices
- Memory usage per component
- Request queues and processing times
- System metrics and logs

### Logging

The system provides detailed logging including:
- Model loading progress
- GPU memory allocation
- Generation timing
- Error handling and recovery

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or video resolution
   # Enable model CPU offloading
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
   ```

2. **Model Loading Failures**
   ```bash
   # Check model access and trust_remote_code
   # Verify sufficient disk space for model cache
   # Ensure stable internet connection
   ```

3. **Ray Serve Issues**
   ```bash
   # Reset Ray cluster
   ray stop
   ray start --head
   ```

### Debug Mode

Enable debug logging:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 🛠️ Configuration

### Environment Variables

```bash
# GPU allocation
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Disable tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Ray configuration
export RAY_memory_monitor_refresh_ms=0
```

### Advanced Configuration

For custom setups, modify the `OptimizedWanPipeline` class:

```python
# Custom device mapping
def _setup_multi_gpu(self):
    # Define your custom GPU assignment
    custom_mapping = {
        "text_encoder": "cuda:0",
        "transformer": "cuda:1", 
        "vae": "cuda:2"
    }
```

## 📋 Requirements

### Hardware
- NVIDIA GPUs with CUDA support
- Minimum 16GB GPU memory (recommended: 24GB+ per GPU)
- 32GB+ system RAM
- Fast storage (SSD recommended) for model caching

### Software
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- Ray 2.4+
- Diffusers 0.30+

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 💾 Cache Management

All model files are stored in `/home/caches/` to organize downloads:

```bash
# Check cache size
python cache_manager.py --size

# Pre-download model to cache
python cache_manager.py --download

# Clear all caches
python cache_manager.py --clear

# Setup cache directories
python cache_manager.py --setup
```

**Cache Structure:**
- `/home/caches/huggingface/` - HuggingFace models
- `/home/caches/transformers/` - Transformer models  
- `/home/caches/diffusers/` - Diffusion models
- `/home/caches/hub/` - HuggingFace Hub cache

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review Ray Serve documentation
- Check diffusers compatibility
- Open an issue with detailed logs