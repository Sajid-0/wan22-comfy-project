#!/usr/bin/env python3
"""
Multi-GPU Speech-to-Video Generation System - Main Entry Point
Integrated system combining all components for production-ready S2V generation
"""

import os
import sys
import json
import time
import uuid
import logging
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Core imports
import ray
import torch
import numpy as np

# Local imports
from s2v_config import S2VConfig
from cache_manager import CacheManager
from enhanced_audio_processor import EnhancedAudioProcessor
from distributed_processing_manager import (
    DistributedTaskScheduler, 
    WorkerMonitor, 
    ProcessingTask,
    LoadBalancer
)
from video_output_handler import VideoCompiler, OutputManager
from multi_gpu_s2v_system import MultiGPUS2VSystem, GPUMonitor

# Setup logging
def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup comprehensive logging"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Reduce Ray logging verbosity
    logging.getLogger("ray").setLevel(logging.WARNING)
    logging.getLogger("ray.serve").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class IntegratedS2VSystem:
    """Integrated multi-GPU speech-to-video generation system"""
    
    def __init__(self, 
                 config: Dict = None,
                 num_gpus: int = None,
                 checkpoint_dir: str = None,
                 output_dir: str = None):
        """
        Initialize the integrated S2V system
        
        Args:
            config: System configuration
            num_gpus: Number of GPUs to use
            checkpoint_dir: Path to model checkpoints
            output_dir: Output directory for results
        """
        self.config = config or {}
        
        # Initialize cache manager first
        self.cache_manager = CacheManager()
        self.cache_manager.setup_environment_variables()
        
        # Use cache-aware paths
        self.checkpoint_dir = checkpoint_dir or str(S2VConfig.CHECKPOINT_DIR)
        self.output_dir = output_dir or './outputs'
        
        # Ensure cache directories exist
        S2VConfig.ensure_cache_dirs()
        
        # Validate environment
        self._validate_environment()
        
        # Initialize components
        self.generation_id = None
        self.audio_processor = None
        self.video_compiler = None
        self.output_manager = None
        self.s2v_system = None
        self.monitor = None
        
        # Auto-detect GPUs if not specified
        if num_gpus is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        self.num_gpus = num_gpus
        if self.num_gpus == 0:
            raise RuntimeError("No GPUs available for processing")
        
        logger.info(f"Initializing Integrated S2V System with {self.num_gpus} GPUs")
        
        # Initialize system
        self._initialize_system()
    
    def _validate_environment(self):
        """Validate the system environment"""
        issues = S2VConfig.validate_environment()
        if issues:
            logger.error("Environment validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            raise RuntimeError("Environment validation failed. Please fix the issues above.")
        
        logger.info("Environment validation passed")
    
    def _initialize_system(self):
        """Initialize all system components"""
        try:
            # Initialize output manager
            self.output_manager = OutputManager(self.output_dir)
            
            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                ray_config = S2VConfig.RAY_CONFIG.copy()
                ray_config['num_gpus'] = self.num_gpus
                ray_config['num_cpus'] = os.cpu_count()
                ray.init(**ray_config)
            
            # Initialize monitor
            self.monitor = WorkerMonitor.remote()
            
            # Initialize S2V system
            self.s2v_system = MultiGPUS2VSystem(
                num_gpus=self.num_gpus,
                config_name=self.config.get('model_name', 's2v-14B'),
                checkpoint_dir=self.checkpoint_dir
            )
            
            # Initialize audio processor
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.audio_processor = EnhancedAudioProcessor(
                device=device,
                target_fps=S2VConfig.DEFAULT_VIDEO_PARAMS['fps']
            )
            
            # Initialize video compiler
            self.video_compiler = VideoCompiler()
            
            logger.info("All system components initialized successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    def generate_video(self,
                      image_path: str,
                      audio_path: str,
                      prompt: str,
                      output_name: str = None,
                      **generation_params) -> Dict:
        """
        Generate video from inputs with full pipeline
        
        Args:
            image_path: Path to input image
            audio_path: Path to input audio
            prompt: Text prompt for generation
            output_name: Output video name (optional)
            **generation_params: Additional generation parameters
            
        Returns:
            Generation results with metadata
        """
        # Generate unique ID for this generation
        self.generation_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # Default parameters
        params = {
            'height': S2VConfig.DEFAULT_VIDEO_PARAMS['height'],
            'width': S2VConfig.DEFAULT_VIDEO_PARAMS['width'],
            'fps': S2VConfig.DEFAULT_VIDEO_PARAMS['fps'],
            'quality': S2VConfig.DEFAULT_VIDEO_PARAMS['quality'],
            'num_frames': None,
            'enhance_quality': True,
            'optimize_streaming': False
        }
        params.update(generation_params)
        
        logger.info(f"Starting video generation {self.generation_id}")
        logger.info(f"Image: {image_path}")
        logger.info(f"Audio: {audio_path}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Parameters: {params}")
        
        try:
            # Phase 1: Input Validation and Preprocessing
            logger.info("Phase 1: Input validation and preprocessing")
            self._validate_inputs(image_path, audio_path)
            
            # Enhance audio quality if needed
            enhanced_audio_path = self.audio_processor.enhance_audio_quality(
                audio_path, 
                os.path.join(self.output_dir, f"enhanced_audio_{self.generation_id}.wav")
            )
            
            # Phase 2: Feature Extraction
            logger.info("Phase 2: Feature extraction")
            audio_features = self._extract_audio_features(enhanced_audio_path, params)
            image_data, image_metadata = self._process_image(image_path, params)
            
            # Determine final parameters
            if params['num_frames'] is None:
                params['num_frames'] = audio_features['num_frames']
            
            # Phase 3: Distributed Processing Setup
            logger.info("Phase 3: Distributed processing setup")
            processing_tasks = self._create_processing_tasks(
                image_data, audio_features, prompt, params
            )
            
            # Phase 4: Parallel Video Generation
            logger.info("Phase 4: Parallel video generation")
            chunk_results = self._execute_parallel_processing(processing_tasks)
            
            # Phase 5: Video Compilation
            logger.info("Phase 5: Video compilation")
            output_path = self._determine_output_path(output_name)
            compilation_results = self._compile_final_video(
                chunk_results, output_path, params
            )
            
            # Phase 6: Post-processing
            logger.info("Phase 6: Post-processing")
            final_output_path = self._finalize_video_output(
                compilation_results['output_path'],
                enhanced_audio_path,
                params
            )
            
            # Phase 7: Results and Cleanup
            generation_time = time.time() - start_time
            results = self._finalize_generation_results(
                final_output_path, generation_time, params, 
                audio_features, image_metadata, chunk_results, compilation_results
            )
            
            logger.info(f"Video generation {self.generation_id} completed successfully in {generation_time:.2f}s")
            return results
            
        except Exception as e:
            error_msg = f"Video generation {self.generation_id} failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Save error information
            error_results = {
                'success': False,
                'error': str(e),
                'generation_id': self.generation_id,
                'generation_time': time.time() - start_time,
                'traceback': traceback.format_exc()
            }
            
            self.output_manager.save_generation_metadata(
                self.generation_id, error_results
            )
            
            raise
    
    def _validate_inputs(self, image_path: str, audio_path: str):
        """Validate input files"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check file formats
        image_ext = Path(image_path).suffix.lower()
        if image_ext not in S2VConfig.IMAGE_CONFIG['supported_formats']:
            raise ValueError(f"Unsupported image format: {image_ext}")
        
        logger.info("Input validation passed")
    
    def _extract_audio_features(self, audio_path: str, params: Dict) -> Dict:
        """Extract enhanced audio features"""
        try:
            features_data = self.audio_processor.extract_features(
                audio_path,
                return_phonemes=True,
                return_timing=True
            )
            
            # Align to target parameters
            if params.get('num_frames'):
                aligned_features = self.audio_processor.align_audio_to_video(
                    features_data['features'],
                    params['fps'],
                    params['num_frames']
                )
                features_data['features'] = aligned_features
                features_data['num_frames'] = params['num_frames']
            
            logger.info(f"Audio features extracted: {features_data['num_frames']} frames")
            return features_data
            
        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
            raise
    
    def _process_image(self, image_path: str, params: Dict) -> Tuple[np.ndarray, Dict]:
        """Process input image"""
        from multi_gpu_s2v_system import ImageProcessor
        
        try:
            image_data, metadata = ImageProcessor.load_and_process_image(
                image_path, (params['width'], params['height'])
            )
            
            logger.info(f"Image processed: {metadata['processed_size']}")
            return image_data, metadata
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise
    
    def _create_processing_tasks(self, 
                               image_data: np.ndarray,
                               audio_features: Dict,
                               prompt: str,
                               params: Dict) -> List[ProcessingTask]:
        """Create processing tasks for distributed execution"""
        try:
            # Use load balancer to determine optimal chunk size
            load_balancer = LoadBalancer(self.monitor)
            optimal_chunk_size = load_balancer.suggest_optimal_chunk_size(
                params['num_frames'], self.num_gpus
            )
            
            # Chunk audio features
            audio_chunks = self.audio_processor.chunk_audio_features(
                audio_features['features'],
                chunk_size=optimal_chunk_size,
                overlap=2
            )
            
            # Create processing tasks
            tasks = []
            quality_settings = S2VConfig.QUALITY_PRESETS[params['quality']]
            
            for i, audio_chunk in enumerate(audio_chunks):
                task_data = {
                    'image_data': image_data,
                    'audio_features': audio_chunk['features'],
                    'prompt': prompt,
                    'start_frame': audio_chunk['start_frame'],
                    'end_frame': audio_chunk['end_frame'],
                    'num_frames': audio_chunk['end_frame'] - audio_chunk['start_frame'],
                    'height': params['height'],
                    'width': params['width'],
                    'chunk_id': i,
                    **quality_settings
                }
                
                if 'overlap_start' in audio_chunk:
                    task_data['overlap_start'] = audio_chunk['overlap_start']
                if 'overlap_end' in audio_chunk:
                    task_data['overlap_end'] = audio_chunk['overlap_end']
                
                task = ProcessingTask(
                    task_id=f"{self.generation_id}_chunk_{i}",
                    chunk_data=task_data,
                    priority=i,  # Process in order
                    estimated_processing_time=optimal_chunk_size * 0.5,  # Rough estimate
                    gpu_memory_required=S2VConfig.GPU_CONFIG['recommended_memory_gb'] / 2
                )
                
                tasks.append(task)
            
            logger.info(f"Created {len(tasks)} processing tasks")
            return tasks
            
        except Exception as e:
            logger.error(f"Task creation failed: {e}")
            raise
    
    def _execute_parallel_processing(self, tasks: List[ProcessingTask]) -> List[Dict]:
        """Execute tasks in parallel using distributed scheduler"""
        try:
            # Create scheduler
            scheduler = DistributedTaskScheduler(
                self.s2v_system.workers, 
                self.monitor
            )
            
            # Submit all tasks
            for task in tasks:
                scheduler.submit_task(task)
            
            # Process tasks in parallel
            results_data = scheduler.process_tasks_parallel(
                max_concurrent=self.num_gpus
            )
            
            # Check for failures
            if results_data['failed_count'] > 0:
                logger.warning(f"{results_data['failed_count']} tasks failed")
                for task_id, failure_info in results_data['failed_tasks'].items():
                    logger.error(f"Task {task_id} failed: {failure_info['error']}")
            
            # Sort results by chunk order
            chunk_results = list(results_data['results'].values())
            chunk_results.sort(key=lambda x: x['chunk_params']['start_frame'])
            
            logger.info(f"Parallel processing completed: {len(chunk_results)} chunks")
            return chunk_results
            
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            raise
    
    def _determine_output_path(self, output_name: str = None) -> str:
        """Determine the output video path"""
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"s2v_generation_{self.generation_id}_{timestamp}"
        
        output_path = self.output_manager.get_unique_output_path(output_name, '.mp4')
        return output_path
    
    def _compile_final_video(self, 
                           chunk_results: List[Dict], 
                           output_path: str, 
                           params: Dict) -> Dict:
        """Compile chunks into final video"""
        try:
            compilation_results = self.video_compiler.compile_chunks(
                chunk_results,
                output_path,
                fps=params['fps'],
                enhance_quality=params['enhance_quality'],
                codec='mp4v'
            )
            
            logger.info(f"Video compilation completed: {output_path}")
            return compilation_results
            
        except Exception as e:
            logger.error(f"Video compilation failed: {e}")
            raise
    
    def _finalize_video_output(self, 
                             video_path: str, 
                             audio_path: str, 
                             params: Dict) -> str:
        """Finalize video with audio and optimizations"""
        try:
            # Add audio to video
            video_with_audio = self.video_compiler.add_audio(
                video_path, audio_path
            )
            
            final_path = video_with_audio
            
            # Optimize for streaming if requested
            if params.get('optimize_streaming', False):
                optimized_path = self.video_compiler.optimize_for_streaming(
                    video_with_audio
                )
                final_path = optimized_path
            
            logger.info(f"Final video output: {final_path}")
            return final_path
            
        except Exception as e:
            logger.error(f"Video finalization failed: {e}")
            raise
    
    def _finalize_generation_results(self, 
                                   output_path: str,
                                   generation_time: float,
                                   params: Dict,
                                   audio_features: Dict,
                                   image_metadata: Dict,
                                   chunk_results: List[Dict],
                                   compilation_results: Dict) -> Dict:
        """Finalize and save generation results"""
        try:
            # Create comprehensive results
            results = {
                'success': True,
                'generation_id': self.generation_id,
                'output_path': output_path,
                'generation_time': generation_time,
                'parameters': params,
                'audio_metadata': {
                    'duration': audio_features['duration'],
                    'num_frames': audio_features['num_frames'],
                    'sample_rate': audio_features['sample_rate'],
                    'feature_dim': audio_features['feature_dim']
                },
                'image_metadata': image_metadata,
                'processing_stats': {
                    'num_workers': self.num_gpus,
                    'chunks_processed': len(chunk_results),
                    'compilation_time': compilation_results['compilation_time'],
                    'total_frames_generated': compilation_results['total_frames']
                },
                'system_stats': ray.get(self.monitor.get_all_stats.remote())
            }
            
            # Add file information
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                results['output_file_size'] = file_size
                results['output_file_size_mb'] = file_size / (1024 * 1024)
            
            # Save metadata and stats
            self.output_manager.save_generation_metadata(self.generation_id, results)
            self.output_manager.save_generation_stats(results['processing_stats'])
            
            # Create summary
            summary = self.output_manager.create_generation_summary(results)
            logger.info(f"Generation summary:\n{summary}")
            
            return results
            
        except Exception as e:
            logger.error(f"Results finalization failed: {e}")
            raise
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        try:
            status = {
                'timestamp': time.time(),
                'ray_initialized': ray.is_initialized(),
                'num_gpus': self.num_gpus,
                'gpu_info': GPUMonitor.get_gpu_info(),
                'system_config': {
                    'checkpoint_dir': self.checkpoint_dir,
                    'output_dir': self.output_dir,
                    'model_config': self.config
                }
            }
            
            if self.monitor:
                status['worker_stats'] = ray.get(self.monitor.get_all_stats.remote())
            
            if self.s2v_system:
                status['s2v_system_stats'] = self.s2v_system.get_system_stats()
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Clean up system resources"""
        try:
            logger.info("Starting system cleanup")
            
            if self.video_compiler:
                self.video_compiler.cleanup()
            
            if self.s2v_system:
                self.s2v_system.cleanup()
            
            if ray.is_initialized():
                ray.shutdown()
            
            logger.info("System cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


def create_argument_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Multi-GPU Speech-to-Video Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic generation
  python main_s2v_system.py --image image.jpg --audio audio.wav --prompt "A cat" --output video.mp4

  # High quality with specific parameters
  python main_s2v_system.py --image image.jpg --audio audio.wav --prompt "A cat" --quality high --fps 60

  # Multi-GPU with optimization
  python main_s2v_system.py --image image.jpg --audio audio.wav --prompt "A cat" --gpus 4 --optimize-streaming
        """
    )
    
    # Required arguments
    parser.add_argument("--image", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--audio", type=str, required=True,
                       help="Path to input audio")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for video generation")
    
    # Output arguments
    parser.add_argument("--output", type=str, default=None,
                       help="Output video name (without extension)")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Output directory")
    
    # System arguments
    parser.add_argument("--gpus", type=int, default=None,
                       help="Number of GPUs to use (auto-detect if not specified)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                       help="Path to model checkpoints")
    parser.add_argument("--model", type=str, default="s2v-14B",
                       help="Model configuration name")
    
    # Video parameters
    parser.add_argument("--height", type=int, default=480,
                       help="Video height")
    parser.add_argument("--width", type=int, default=640,
                       help="Video width")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second")
    parser.add_argument("--frames", type=int, default=None,
                       help="Number of frames (auto-detect from audio if not specified)")
    
    # Quality parameters
    parser.add_argument("--quality", type=str, default="medium",
                       choices=["low", "medium", "high"],
                       help="Generation quality")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps for generation")
    parser.add_argument("--enhance-quality", action="store_true",
                       help="Apply video quality enhancements")
    parser.add_argument("--optimize-streaming", action="store_true",
                       help="Optimize video for streaming")
    
    # Logging and debugging
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file", type=str, default=None,
                       help="Log file path")
    parser.add_argument("--save-stats", action="store_true",
                       help="Save detailed statistics")
    
    return parser


def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file or f"s2v_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(args.log_level, log_file)
    
    logger.info("="*60)
    logger.info("MULTI-GPU SPEECH-TO-VIDEO GENERATION SYSTEM")
    logger.info("="*60)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Validate inputs
    if not os.path.exists(args.image):
        logger.error(f"Image file not found: {args.image}")
        return 1
    
    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        return 1
    
    system = None
    try:
        # Initialize system
        config = {
            'model_name': args.model
        }
        
        system = IntegratedS2VSystem(
            config=config,
            num_gpus=args.gpus,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir
        )
        
        # Generation parameters
        generation_params = {
            'height': args.height,
            'width': args.width,
            'fps': args.fps,
            'num_frames': args.frames,
            'quality': args.quality,
            'num_inference_steps': args.num_inference_steps,
            'enhance_quality': args.enhance_quality,
            'optimize_streaming': args.optimize_streaming
        }
        
        # Generate video
        logger.info("Starting video generation...")
        results = system.generate_video(
            image_path=args.image,
            audio_path=args.audio,
            prompt=args.prompt,
            output_name=args.output,
            **generation_params
        )
        
        # Print results
        print("\n" + "="*60)
        print("VIDEO GENERATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Output: {results['output_path']}")
        print(f"Generation ID: {results['generation_id']}")
        print(f"Generation Time: {results['generation_time']:.2f} seconds")
        print(f"Video Duration: {results['processing_stats']['total_frames_generated'] / args.fps:.1f} seconds")
        print(f"Resolution: {args.width}x{args.height} @ {args.fps}fps")
        print(f"Workers Used: {results['processing_stats']['num_workers']}")
        print(f"Chunks Processed: {results['processing_stats']['chunks_processed']}")
        
        if 'output_file_size_mb' in results:
            print(f"File Size: {results['output_file_size_mb']:.1f} MB")
        
        print("="*60)
        
        # Save detailed stats if requested
        if args.save_stats:
            stats_file = os.path.join(args.output_dir, f"detailed_stats_{results['generation_id']}.json")
            with open(stats_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Detailed statistics saved: {stats_file}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        logger.error(traceback.format_exc())
        return 1
        
    finally:
        if system:
            system.cleanup()
        
        logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    exit(main())