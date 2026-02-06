#!/usr/bin/env python3
"""
Video Output Handler for Multi-GPU S2V System
Handles video compilation, post-processing, and quality optimization
"""

import os
import sys
import cv2
import logging
import numpy as np
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import tempfile
import shutil
from PIL import Image, ImageFilter, ImageEnhance
import json

# Add Wan2.2 to path
sys.path.append(str(Path(__file__).parent / "Wan2.2"))

from wan.utils.utils import save_video, merge_video_audio

logger = logging.getLogger(__name__)


class VideoQualityEnhancer:
    """Enhance video quality through post-processing"""
    
    @staticmethod
    def enhance_frame(frame: np.ndarray, 
                     enhance_params: Dict = None) -> np.ndarray:
        """
        Enhance a single video frame
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            enhance_params: Enhancement parameters
            
        Returns:
            Enhanced frame
        """
        if enhance_params is None:
            enhance_params = {
                'brightness': 1.1,
                'contrast': 1.2,
                'saturation': 1.1,
                'sharpness': 1.1
            }
        
        try:
            # Convert to PIL Image
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(frame)
            
            # Apply enhancements
            if 'brightness' in enhance_params:
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(enhance_params['brightness'])
            
            if 'contrast' in enhance_params:
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(enhance_params['contrast'])
            
            if 'saturation' in enhance_params:
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(enhance_params['saturation'])
            
            if 'sharpness' in enhance_params:
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(enhance_params['sharpness'])
            
            # Convert back to numpy
            enhanced_frame = np.array(pil_image)
            
            return enhanced_frame
            
        except Exception as e:
            logger.warning(f"Frame enhancement failed: {e}")
            return frame
    
    @staticmethod
    def temporal_smoothing(frames: List[np.ndarray], 
                          window_size: int = 3) -> List[np.ndarray]:
        """
        Apply temporal smoothing to reduce flicker
        
        Args:
            frames: List of video frames
            window_size: Smoothing window size (odd number)
            
        Returns:
            Smoothed frames
        """
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd window size
        
        half_window = window_size // 2
        smoothed_frames = []
        
        try:
            for i in range(len(frames)):
                # Define window bounds
                start_idx = max(0, i - half_window)
                end_idx = min(len(frames), i + half_window + 1)
                
                # Average frames in window
                window_frames = frames[start_idx:end_idx]
                weights = np.exp(-0.5 * np.linspace(-1, 1, len(window_frames))**2)
                weights /= np.sum(weights)
                
                # Weighted average
                smoothed_frame = np.zeros_like(frames[i], dtype=np.float32)
                for j, frame in enumerate(window_frames):
                    smoothed_frame += frame.astype(np.float32) * weights[j]
                
                smoothed_frames.append(smoothed_frame.astype(np.uint8))
            
            logger.info(f"Applied temporal smoothing with window size {window_size}")
            return smoothed_frames
            
        except Exception as e:
            logger.warning(f"Temporal smoothing failed: {e}")
            return frames


class VideoCompiler:
    """Compile video chunks into final output"""
    
    def __init__(self, temp_dir: str = None):
        """
        Initialize video compiler
        
        Args:
            temp_dir: Temporary directory for processing
        """
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="s2v_video_")
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info(f"Video compiler initialized with temp dir: {self.temp_dir}")
    
    def compile_chunks(self, 
                      chunk_results: List[Dict],
                      output_path: str,
                      fps: int = 30,
                      enhance_quality: bool = True,
                      codec: str = 'mp4v') -> Dict:
        """
        Compile video chunks into final video
        
        Args:
            chunk_results: List of chunk processing results
            output_path: Output video path
            fps: Frames per second
            enhance_quality: Whether to apply quality enhancements
            codec: Video codec to use
            
        Returns:
            Compilation statistics
        """
        start_time = time.time()
        
        try:
            # Sort chunks by start frame
            sorted_chunks = sorted(chunk_results, 
                                 key=lambda x: x['chunk_params']['start_frame'])
            
            # Combine all frames
            all_frames = []
            total_frames = 0
            
            for chunk in sorted_chunks:
                frames = chunk['frames']
                if isinstance(frames, np.ndarray):
                    if frames.ndim == 4:  # (N, H, W, C)
                        frame_list = [frames[i] for i in range(frames.shape[0])]
                    else:
                        frame_list = [frames]
                else:
                    frame_list = frames
                
                all_frames.extend(frame_list)
                total_frames += len(frame_list)
            
            logger.info(f"Combined {total_frames} frames from {len(sorted_chunks)} chunks")
            
            # Apply quality enhancements if requested
            if enhance_quality:
                all_frames = self._enhance_video_quality(all_frames)
            
            # Handle overlapping chunks if needed
            all_frames = self._handle_chunk_overlaps(all_frames, sorted_chunks)
            
            # Save video
            self._save_compiled_video(all_frames, output_path, fps, codec)
            
            compilation_time = time.time() - start_time
            
            # Get video statistics
            video_stats = self._get_video_statistics(output_path)
            
            result = {
                'output_path': output_path,
                'total_frames': len(all_frames),
                'fps': fps,
                'duration_seconds': len(all_frames) / fps,
                'compilation_time': compilation_time,
                'chunks_processed': len(sorted_chunks),
                'enhanced': enhance_quality,
                'codec': codec,
                **video_stats
            }
            
            logger.info(f"Video compilation completed in {compilation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Video compilation failed: {e}")
            raise
    
    def _enhance_video_quality(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply video quality enhancements"""
        try:
            enhancer = VideoQualityEnhancer()
            
            # Apply frame-wise enhancements
            enhanced_frames = []
            for frame in frames:
                enhanced_frame = enhancer.enhance_frame(frame)
                enhanced_frames.append(enhanced_frame)
            
            # Apply temporal smoothing
            enhanced_frames = enhancer.temporal_smoothing(enhanced_frames)
            
            logger.info("Applied video quality enhancements")
            return enhanced_frames
            
        except Exception as e:
            logger.warning(f"Quality enhancement failed: {e}")
            return frames
    
    def _handle_chunk_overlaps(self, 
                              frames: List[np.ndarray], 
                              chunk_results: List[Dict]) -> List[np.ndarray]:
        """Handle overlapping regions between chunks"""
        try:
            # Check if chunks have overlap information
            has_overlaps = any('overlap_start' in chunk['chunk_params'] 
                             for chunk in chunk_results)
            
            if not has_overlaps:
                return frames
            
            # Process overlaps with blending
            processed_frames = []
            frame_idx = 0
            
            for i, chunk in enumerate(chunk_results):
                chunk_params = chunk['chunk_params']
                chunk_frames = chunk['frames']
                
                if isinstance(chunk_frames, np.ndarray) and chunk_frames.ndim == 4:
                    chunk_frame_list = [chunk_frames[j] for j in range(chunk_frames.shape[0])]
                else:
                    chunk_frame_list = chunk_frames
                
                overlap_start = chunk_params.get('overlap_start', 0)
                overlap_end = chunk_params.get('overlap_end', 0)
                
                if i == 0:
                    # First chunk - add all frames
                    processed_frames.extend(chunk_frame_list)
                else:
                    # Subsequent chunks - blend overlapping regions
                    if overlap_start > 0:
                        # Blend overlapping frames
                        blend_frames = chunk_frame_list[:overlap_start]
                        for j, blend_frame in enumerate(blend_frames):
                            existing_idx = len(processed_frames) - overlap_start + j
                            if existing_idx < len(processed_frames):
                                # Blend frames
                                alpha = (j + 1) / (overlap_start + 1)
                                blended_frame = (
                                    (1 - alpha) * processed_frames[existing_idx].astype(np.float32) +
                                    alpha * blend_frame.astype(np.float32)
                                ).astype(np.uint8)
                                processed_frames[existing_idx] = blended_frame
                        
                        # Add non-overlapping frames
                        processed_frames.extend(chunk_frame_list[overlap_start:])
                    else:
                        processed_frames.extend(chunk_frame_list)
            
            logger.info(f"Processed chunk overlaps, final frame count: {len(processed_frames)}")
            return processed_frames
            
        except Exception as e:
            logger.warning(f"Overlap handling failed: {e}")
            return frames
    
    def _save_compiled_video(self, 
                           frames: List[np.ndarray], 
                           output_path: str, 
                           fps: int, 
                           codec: str):
        """Save compiled video using OpenCV"""
        try:
            if not frames:
                raise ValueError("No frames to save")
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Define codec
            fourcc = cv2.VideoWriter_fourcc(*codec)
            
            # Create video writer
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise RuntimeError(f"Could not open video writer for {output_path}")
            
            # Write frames
            for i, frame in enumerate(frames):
                # Ensure frame is in correct format
                if frame.shape[2] == 3:  # RGB
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                out.write(frame_bgr)
                
                if i % 100 == 0:
                    logger.debug(f"Written {i}/{len(frames)} frames")
            
            out.release()
            
            # Verify output file
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output video was not created: {output_path}")
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            logger.info(f"Video saved: {output_path} ({file_size:.1f} MB)")
            
        except Exception as e:
            logger.error(f"Failed to save video: {e}")
            raise
    
    def _get_video_statistics(self, video_path: str) -> Dict:
        """Get statistics about the compiled video"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {'error': 'Could not open video for statistics'}
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            file_size = os.path.getsize(video_path)
            
            return {
                'frame_count': frame_count,
                'actual_fps': fps,
                'resolution': f"{width}x{height}",
                'file_size_bytes': file_size,
                'file_size_mb': file_size / (1024 * 1024),
                'duration_seconds': frame_count / fps if fps > 0 else 0
            }
            
        except Exception as e:
            logger.warning(f"Could not get video statistics: {e}")
            return {'error': str(e)}
    
    def add_audio(self, 
                 video_path: str, 
                 audio_path: str, 
                 output_path: str = None) -> str:
        """
        Add audio to compiled video
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            output_path: Output path (defaults to video_path + '_with_audio')
            
        Returns:
            Path to video with audio
        """
        if output_path is None:
            base_path = os.path.splitext(video_path)[0]
            output_path = f"{base_path}_with_audio.mp4"
        
        try:
            # Use FFmpeg for better audio/video synchronization
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', video_path,  # Video input
                '-i', audio_path,  # Audio input
                '-c:v', 'copy',  # Copy video stream
                '-c:a', 'aac',   # Encode audio as AAC
                '-map', '0:v:0',  # Map video from first input
                '-map', '1:a:0',  # Map audio from second input
                '-shortest',      # Stop at shortest stream
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if os.path.exists(output_path):
                logger.info(f"Audio added successfully: {output_path}")
                return output_path
            else:
                raise RuntimeError("FFmpeg command succeeded but output file not found")
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"FFmpeg failed: {e.stderr}")
            
            # Fallback to existing utility
            try:
                merge_video_audio(video_path, audio_path, output_path)
                logger.info(f"Audio added using fallback method: {output_path}")
                return output_path
            except Exception as fallback_error:
                logger.error(f"Both FFmpeg and fallback methods failed: {fallback_error}")
                raise
        
        except FileNotFoundError:
            logger.warning("FFmpeg not found, using fallback method")
            
            # Use existing utility
            merge_video_audio(video_path, audio_path, output_path)
            logger.info(f"Audio added using fallback method: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to add audio: {e}")
            raise
    
    def optimize_for_streaming(self, 
                             video_path: str, 
                             output_path: str = None,
                             bitrate: str = '2M') -> str:
        """
        Optimize video for streaming
        
        Args:
            video_path: Input video path
            output_path: Output path (optional)
            bitrate: Target bitrate
            
        Returns:
            Path to optimized video
        """
        if output_path is None:
            base_path = os.path.splitext(video_path)[0]
            output_path = f"{base_path}_optimized.mp4"
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-b:v', bitrate,
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', 'faststart',  # Enable streaming
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            logger.info(f"Video optimized for streaming: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Video optimization failed: {e}")
            return video_path  # Return original if optimization fails
        
        except FileNotFoundError:
            logger.warning("FFmpeg not available for optimization")
            return video_path
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")


class OutputManager:
    """Manage video outputs and metadata"""
    
    def __init__(self, output_dir: str):
        """
        Initialize output manager
        
        Args:
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.output_dir / "generation_metadata.json"
        self.stats_file = self.output_dir / "generation_stats.json"
        
        logger.info(f"Output manager initialized: {self.output_dir}")
    
    def save_generation_metadata(self, 
                                generation_id: str,
                                metadata: Dict):
        """Save metadata for a generation"""
        try:
            # Load existing metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    all_metadata = json.load(f)
            else:
                all_metadata = {}
            
            # Add new metadata
            all_metadata[generation_id] = {
                'timestamp': time.time(),
                'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                **metadata
            }
            
            # Save updated metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(all_metadata, f, indent=2, default=str)
            
            logger.info(f"Saved metadata for generation {generation_id}")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def save_generation_stats(self, stats: Dict):
        """Save generation statistics"""
        try:
            stats_data = {
                'timestamp': time.time(),
                'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                **stats
            }
            
            # Load existing stats
            if self.stats_file.exists():
                with open(self.stats_file, 'r') as f:
                    all_stats = json.load(f)
                    if isinstance(all_stats, list):
                        all_stats.append(stats_data)
                    else:
                        all_stats = [all_stats, stats_data]
            else:
                all_stats = [stats_data]
            
            # Save updated stats
            with open(self.stats_file, 'w') as f:
                json.dump(all_stats, f, indent=2, default=str)
            
            logger.info("Saved generation statistics")
            
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")
    
    def get_unique_output_path(self, 
                              base_name: str, 
                              extension: str = '.mp4') -> str:
        """Get a unique output path"""
        counter = 0
        while True:
            if counter == 0:
                path = self.output_dir / f"{base_name}{extension}"
            else:
                path = self.output_dir / f"{base_name}_{counter}{extension}"
            
            if not path.exists():
                return str(path)
            
            counter += 1
    
    def create_generation_summary(self, generation_results: Dict) -> str:
        """Create a summary report for the generation"""
        try:
            summary = []
            summary.append("="*60)
            summary.append("VIDEO GENERATION SUMMARY")
            summary.append("="*60)
            
            # Basic info
            summary.append(f"Output: {generation_results.get('output_path', 'N/A')}")
            summary.append(f"Generation Time: {generation_results.get('generation_time', 0):.2f}s")
            summary.append(f"Resolution: {generation_results.get('resolution', 'N/A')}")
            summary.append(f"FPS: {generation_results.get('fps', 'N/A')}")
            summary.append(f"Total Frames: {generation_results.get('num_frames', 'N/A')}")
            summary.append(f"Duration: {generation_results.get('num_frames', 0) / generation_results.get('fps', 1):.1f}s")
            summary.append("")
            
            # Performance info
            summary.append("PERFORMANCE METRICS:")
            summary.append(f"Workers Used: {generation_results.get('num_workers', 'N/A')}")
            summary.append(f"Chunks Processed: {generation_results.get('chunks_processed', 'N/A')}")
            
            if 'chunks_processed' in generation_results and generation_results['chunks_processed'] > 0:
                avg_chunk_time = generation_results['generation_time'] / generation_results['chunks_processed']
                summary.append(f"Average Chunk Time: {avg_chunk_time:.2f}s")
            
            summary.append("")
            
            # File info
            output_path = generation_results.get('output_path')
            if output_path and os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                summary.append(f"Output File Size: {file_size:.1f} MB")
            
            summary.append("="*60)
            
            summary_text = "\n".join(summary)
            
            # Save summary to file
            summary_path = self.output_dir / "latest_generation_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(summary_text)
            
            return summary_text
            
        except Exception as e:
            logger.error(f"Failed to create summary: {e}")
            return "Summary generation failed"


import time  # Add missing import