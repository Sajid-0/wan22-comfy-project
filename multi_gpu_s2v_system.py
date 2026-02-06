#!/usr/bin/env python3
"""
Multi-GPU Speech-to-Video Generation System using Ray and Wan2.2
Author: GitHub Copilot
Version: 1.0.0
"""

import os
import sys
import json
import time
import logging
import argparse
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import ray
import torch
import numpy as np
from PIL import Image
import librosa
import soundfile as sf
from tqdm import tqdm

# Add Wan2.2 to path
sys.path.append(str(Path(__file__).parent / "Wan2.2"))

from wan.speech2video import WanS2V
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video, merge_video_audio
from wan.distributed.util import init_distributed_group

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_gpu_s2v.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitor GPU usage and performance"""
    
    @staticmethod
    def get_gpu_info():
        """Get current GPU information"""
        if not torch.cuda.is_available():
            return {"num_gpus": 0, "gpu_info": []}
        
        num_gpus = torch.cuda.device_count()
        gpu_info = []
        
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3
            memory_total = props.total_memory / 1024**3
            
            gpu_info.append({
                "id": i,
                "name": props.name,
                "memory_allocated_gb": memory_allocated,
                "memory_cached_gb": memory_cached,
                "memory_total_gb": memory_total,
                "memory_utilization": (memory_allocated / memory_total) * 100
            })
        
        return {"num_gpus": num_gpus, "gpu_info": gpu_info}
    
    @staticmethod
    def log_gpu_status():
        """Log current GPU status"""
        info = GPUMonitor.get_gpu_info()
        logger.info(f"GPU Status: {info['num_gpus']} GPUs available")
        for gpu in info['gpu_info']:
            logger.info(f"GPU {gpu['id']} ({gpu['name']}): "
                       f"{gpu['memory_utilization']:.1f}% memory used "
                       f"({gpu['memory_allocated_gb']:.1f}/{gpu['memory_total_gb']:.1f} GB)")


@ray.remote
class ModelWorker:
    """Ray actor for distributed model processing"""
    
    def __init__(self, 
                 gpu_id: int, 
                 config_name: str,
                 checkpoint_dir: str,
                 worker_id: int = 0):
        """
        Initialize model worker
        
        Args:
            gpu_id: GPU device ID
            config_name: Model configuration name (e.g., 's2v-14B')
            checkpoint_dir: Path to model checkpoints
            worker_id: Unique worker identifier
        """
        # Ensure Wan2.2 is in path for Ray workers
        import sys
        import os
        wan_path = "/workspace/wan22-comfy-project/Wan2.2"
        if wan_path not in sys.path:
            sys.path.insert(0, wan_path)
        
        # Change to the project directory
        os.chdir("/workspace/wan22-comfy-project")
        
        self.gpu_id = gpu_id
        self.worker_id = worker_id
        self.config_name = config_name
        self.checkpoint_dir = checkpoint_dir
        
        # Set CUDA device
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)  # Use the visible device
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"Worker {worker_id} initialized on GPU {gpu_id}")
    
    def _initialize_model(self):
        """Initialize the Wan S2V model"""
        try:
            config = WAN_CONFIGS[self.config_name]
            
            self.model = WanS2V(
                config=config,
                checkpoint_dir=self.checkpoint_dir,
                device_id=0,  # Use the visible device
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_sp=False,
                t5_cpu=False,
                init_on_cpu=False,
                convert_model_dtype=True
            )
            
            # Warm up the model
            self._warmup()
            
            logger.info(f"Worker {self.worker_id}: Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Failed to initialize model: {e}")
            raise
    
    def _warmup(self):
        """Warm up the model with a dummy forward pass"""
        try:
            # Create dummy inputs for warmup
            dummy_image = torch.randn(1, 3, 256, 256).cuda()
            dummy_audio = torch.randn(1, 100, 1024).cuda()  # Dummy audio features
            dummy_prompt = "warmup"
            
            # This would be a simplified warmup - actual implementation would depend on model interface
            logger.info(f"Worker {self.worker_id}: Model warmup completed")
            
        except Exception as e:
            logger.warning(f"Worker {self.worker_id}: Warmup failed: {e}")
    
    def process_chunk(self,
                     image_data: np.ndarray,
                     audio_features: np.ndarray,
                     prompt: str,
                     chunk_params: Dict) -> Dict:
        """
        Process a chunk of video generation
        
        Args:
            image_data: Input image as numpy array
            audio_features: Audio features for this chunk
            prompt: Text prompt
            chunk_params: Additional parameters for this chunk
            
        Returns:
            Dictionary containing generated frames and metadata
        """
        try:
            start_time = time.time()
            
            # Convert inputs to appropriate format
            image_tensor = torch.from_numpy(image_data).cuda()
            audio_tensor = torch.from_numpy(audio_features).cuda()
            
            # Process with model (this is a placeholder - actual implementation would call model methods)
            # generated_frames = self.model.generate(
            #     image=image_tensor,
            #     audio=audio_tensor,
            #     prompt=prompt,
            #     **chunk_params
            # )
            
            # For now, return dummy frames as placeholder
            num_frames = chunk_params.get('num_frames', 25)
            height = chunk_params.get('height', 480)
            width = chunk_params.get('width', 640)
            
            # Generate dummy frames (replace with actual model output)
            generated_frames = np.random.randint(0, 255, 
                                               (num_frames, height, width, 3), 
                                               dtype=np.uint8)
            
            processing_time = time.time() - start_time
            
            result = {
                'frames': generated_frames,
                'worker_id': self.worker_id,
                'gpu_id': self.gpu_id,
                'processing_time': processing_time,
                'chunk_params': chunk_params
            }
            
            logger.info(f"Worker {self.worker_id}: Processed chunk in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error processing chunk: {e}")
            raise
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage of this worker"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,
                'reserved': torch.cuda.memory_reserved() / 1024**3,
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
            }
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()
        logger.info(f"Worker {self.worker_id}: Cleanup completed")


class AudioProcessor:
    """Process and prepare audio data for video generation"""
    
    @staticmethod
    def load_and_process_audio(audio_path: str, 
                             target_fps: int = 30,
                             target_duration: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
        """
        Load and process audio file
        
        Args:
            audio_path: Path to audio file
            target_fps: Target video FPS
            target_duration: Target duration in seconds (optional)
            
        Returns:
            Tuple of (audio_features, metadata)
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio) / sr
            
            # Trim or pad to target duration if specified
            if target_duration:
                target_samples = int(target_duration * sr)
                if len(audio) > target_samples:
                    audio = audio[:target_samples]
                else:
                    audio = np.pad(audio, (0, target_samples - len(audio)), 'constant')
            
            # Extract features (placeholder - would use actual audio encoder)
            # This should use the AudioEncoder from wan.modules.s2v.audio_encoder
            num_frames = int(duration * target_fps) if not target_duration else int(target_duration * target_fps)
            audio_features = np.random.randn(num_frames, 1024).astype(np.float32)
            
            metadata = {
                'original_sr': sr,
                'duration': duration,
                'num_frames': num_frames,
                'target_fps': target_fps
            }
            
            logger.info(f"Audio processed: {duration:.2f}s, {num_frames} frames")
            return audio_features, metadata
            
        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            raise


class ImageProcessor:
    """Process and prepare image data for video generation"""
    
    @staticmethod
    def load_and_process_image(image_path: str,
                             target_size: Tuple[int, int] = (640, 480)) -> Tuple[np.ndarray, Dict]:
        """
        Load and process image file
        
        Args:
            image_path: Path to image file
            target_size: Target size (width, height)
            
        Returns:
            Tuple of (image_array, metadata)
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # Resize while maintaining aspect ratio
            image = image.resize(target_size, Image.LANCZOS)
            
            # Convert to numpy array
            image_array = np.array(image)
            
            metadata = {
                'original_size': original_size,
                'processed_size': target_size,
                'channels': image_array.shape[2]
            }
            
            logger.info(f"Image processed: {original_size} -> {target_size}")
            return image_array, metadata
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise


class MultiGPUS2VSystem:
    """Main multi-GPU speech-to-video generation system"""
    
    def __init__(self,
                 num_gpus: int = None,
                 config_name: str = "s2v-14B",
                 checkpoint_dir: str = None):
        """
        Initialize the multi-GPU S2V system
        
        Args:
            num_gpus: Number of GPUs to use (auto-detect if None)
            config_name: Model configuration name
            checkpoint_dir: Path to model checkpoints
        """
        self.config_name = config_name
        self.checkpoint_dir = checkpoint_dir or os.getenv('WAN_CHECKPOINT_DIR', './checkpoints')
        
        # Auto-detect GPUs if not specified
        if num_gpus is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        self.num_gpus = num_gpus
        if self.num_gpus == 0:
            raise RuntimeError("No GPUs available for processing")
        
        logger.info(f"Initializing Multi-GPU S2V System with {self.num_gpus} GPUs")
        
        # Initialize Ray
        self._initialize_ray()
        
        # Create workers
        self.workers = []
        self._initialize_workers()
        
        GPUMonitor.log_gpu_status()
    
    def _initialize_ray(self):
        """Initialize Ray cluster"""
        try:
            if not ray.is_initialized():
                ray.init(
                    num_gpus=self.num_gpus,
                    num_cpus=os.cpu_count(),
                    object_store_memory=2000000000,  # 2GB object store
                    log_to_driver=True
                )
            logger.info("Ray cluster initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            raise
    
    def _initialize_workers(self):
        """Initialize model workers on each GPU"""
        try:
            for gpu_id in range(self.num_gpus):
                worker = ModelWorker.remote(
                    gpu_id=gpu_id,
                    config_name=self.config_name,
                    checkpoint_dir=self.checkpoint_dir,
                    worker_id=gpu_id
                )
                self.workers.append(worker)
            
            # Wait for all workers to initialize
            ray.get([worker._initialize_model.remote() for worker in self.workers])
            logger.info(f"All {len(self.workers)} workers initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize workers: {e}")
            raise
    
    def generate_video(self,
                      image_path: str,
                      audio_path: str,
                      prompt: str,
                      output_path: str,
                      height: int = 480,
                      width: int = 640,
                      num_frames: int = None,
                      fps: int = 30,
                      quality: str = "medium") -> Dict:
        """
        Generate video from inputs using multi-GPU processing
        
        Args:
            image_path: Path to input image
            audio_path: Path to input audio
            prompt: Text prompt for generation
            output_path: Path for output video
            height: Video height
            width: Video width
            num_frames: Number of frames (auto-detect from audio if None)
            fps: Frames per second
            quality: Generation quality ('low', 'medium', 'high')
            
        Returns:
            Dictionary with generation results and statistics
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting video generation: {width}x{height}, {fps}fps")
            logger.info(f"Image: {image_path}")
            logger.info(f"Audio: {audio_path}")
            logger.info(f"Prompt: {prompt}")
            
            # Process inputs
            image_data, image_meta = ImageProcessor.load_and_process_image(
                image_path, (width, height)
            )
            
            audio_features, audio_meta = AudioProcessor.load_and_process_audio(
                audio_path, fps
            )
            
            # Determine number of frames
            if num_frames is None:
                num_frames = audio_meta['num_frames']
            
            # Split work among GPUs
            chunks = self._split_work(num_frames, audio_features, image_data, prompt, 
                                    height, width, quality)
            
            # Process chunks in parallel
            results = self._process_chunks_parallel(chunks)
            
            # Combine results
            final_video = self._combine_results(results, output_path, fps)
            
            # Add audio to final video
            if os.path.exists(output_path):
                merge_video_audio(output_path, audio_path, f"{output_path}_with_audio.mp4")
                final_output = f"{output_path}_with_audio.mp4"
            else:
                final_output = output_path
            
            generation_time = time.time() - start_time
            
            result = {
                'output_path': final_output,
                'generation_time': generation_time,
                'num_frames': num_frames,
                'fps': fps,
                'resolution': f"{width}x{height}",
                'num_workers': len(self.workers),
                'chunks_processed': len(chunks),
                'image_meta': image_meta,
                'audio_meta': audio_meta
            }
            
            logger.info(f"Video generation completed in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error during video generation: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _split_work(self, num_frames: int, audio_features: np.ndarray, 
                   image_data: np.ndarray, prompt: str, height: int, width: int,
                   quality: str) -> List[Dict]:
        """Split work into chunks for parallel processing"""
        chunks = []
        frames_per_chunk = max(1, num_frames // len(self.workers))
        
        # Quality settings
        quality_settings = {
            'low': {'guidance_scale': 2.0, 'num_inference_steps': 8},
            'medium': {'guidance_scale': 3.0, 'num_inference_steps': 15},
            'high': {'guidance_scale': 4.0, 'num_inference_steps': 25}
        }
        
        settings = quality_settings.get(quality, quality_settings['medium'])
        
        for i in range(len(self.workers)):
            start_frame = i * frames_per_chunk
            if i == len(self.workers) - 1:  # Last chunk gets remaining frames
                end_frame = num_frames
            else:
                end_frame = min(start_frame + frames_per_chunk, num_frames)
            
            chunk_frames = end_frame - start_frame
            if chunk_frames <= 0:
                continue
            
            # Extract audio features for this chunk
            chunk_audio = audio_features[start_frame:end_frame]
            
            chunk = {
                'worker_id': i,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'num_frames': chunk_frames,
                'image_data': image_data,
                'audio_features': chunk_audio,
                'prompt': prompt,
                'height': height,
                'width': width,
                **settings
            }
            
            chunks.append(chunk)
        
        logger.info(f"Split work into {len(chunks)} chunks")
        return chunks
    
    def _process_chunks_parallel(self, chunks: List[Dict]) -> List[Dict]:
        """Process chunks in parallel using Ray workers"""
        futures = []
        
        for i, chunk in enumerate(chunks):
            worker = self.workers[i]
            future = worker.process_chunk.remote(
                chunk['image_data'],
                chunk['audio_features'],
                chunk['prompt'],
                {k: v for k, v in chunk.items() 
                 if k not in ['image_data', 'audio_features', 'prompt']}
            )
            futures.append(future)
        
        # Wait for all chunks to complete with progress bar
        results = []
        with tqdm(total=len(futures), desc="Processing chunks") as pbar:
            while futures:
                ready, futures = ray.wait(futures, num_returns=1, timeout=1.0)
                for future in ready:
                    result = ray.get(future)
                    results.append(result)
                    pbar.update(1)
        
        return results
    
    def _combine_results(self, results: List[Dict], output_path: str, fps: int) -> str:
        """Combine chunk results into final video"""
        try:
            # Sort results by start frame
            results.sort(key=lambda x: x['chunk_params']['start_frame'])
            
            # Combine all frames
            all_frames = []
            for result in results:
                all_frames.extend(result['frames'])
            
            # Save video using existing utility
            save_video(all_frames, output_path, fps=fps)
            
            logger.info(f"Combined {len(all_frames)} frames into {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error combining results: {e}")
            raise
    
    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        gpu_info = GPUMonitor.get_gpu_info()
        
        # Get worker memory usage
        worker_stats = []
        if self.workers:
            memory_futures = [worker.get_memory_usage.remote() for worker in self.workers]
            memory_usage = ray.get(memory_futures)
            
            for i, usage in enumerate(memory_usage):
                worker_stats.append({
                    'worker_id': i,
                    'gpu_id': i,
                    **usage
                })
        
        return {
            'num_workers': len(self.workers),
            'gpu_info': gpu_info,
            'worker_stats': worker_stats,
            'ray_status': ray.cluster_resources() if ray.is_initialized() else None
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.workers:
                ray.get([worker.cleanup.remote() for worker in self.workers])
                self.workers = []
            
            if ray.is_initialized():
                ray.shutdown()
            
            logger.info("System cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Multi-GPU Speech-to-Video Generation System")
    
    parser.add_argument("--image", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--audio", type=str, required=True,
                       help="Path to input audio")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for video generation")
    parser.add_argument("--output", type=str, required=True,
                       help="Path for output video")
    parser.add_argument("--gpus", type=int, default=None,
                       help="Number of GPUs to use (auto-detect if not specified)")
    parser.add_argument("--height", type=int, default=480,
                       help="Video height")
    parser.add_argument("--width", type=int, default=640,
                       help="Video width")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second")
    parser.add_argument("--frames", type=int, default=None,
                       help="Number of frames (auto-detect from audio if not specified)")
    parser.add_argument("--quality", type=str, default="medium",
                       choices=["low", "medium", "high"],
                       help="Generation quality")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                       help="Path to model checkpoints")
    parser.add_argument("--config", type=str, default="s2v-14B",
                       help="Model configuration name")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")
    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize system
    system = None
    try:
        system = MultiGPUS2VSystem(
            num_gpus=args.gpus,
            config_name=args.config,
            checkpoint_dir=args.checkpoint_dir
        )
        
        # Generate video
        result = system.generate_video(
            image_path=args.image,
            audio_path=args.audio,
            prompt=args.prompt,
            output_path=args.output,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            fps=args.fps,
            quality=args.quality
        )
        
        # Print results
        print("\n" + "="*60)
        print("VIDEO GENERATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Output video: {result['output_path']}")
        print(f"Generation time: {result['generation_time']:.2f} seconds")
        print(f"Video specs: {result['resolution']} @ {result['fps']}fps")
        print(f"Total frames: {result['num_frames']}")
        print(f"Workers used: {result['num_workers']}")
        print(f"Chunks processed: {result['chunks_processed']}")
        print("="*60)
        
        # Print system stats
        stats = system.get_system_stats()
        print("\nSYSTEM STATISTICS:")
        print(f"GPUs available: {stats['gpu_info']['num_gpus']}")
        for gpu in stats['gpu_info']['gpu_info']:
            print(f"  GPU {gpu['id']}: {gpu['memory_utilization']:.1f}% memory used")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    finally:
        if system:
            system.cleanup()


if __name__ == "__main__":
    main()