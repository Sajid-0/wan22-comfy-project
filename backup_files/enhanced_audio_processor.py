#!/usr/bin/env python3
"""
Enhanced Audio Processing Module for Multi-GPU S2V System
Handles audio feature extraction, phoneme alignment, and temporal synchronization
"""

import os
import sys
import logging
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import soundfile as sf
from pathlib import Path

# Add Wan2.2 to path
sys.path.append(str(Path(__file__).parent / "Wan2.2"))

from wan.modules.s2v.audio_encoder import AudioEncoder, get_sample_indices, linear_interpolation

logger = logging.getLogger(__name__)


class EnhancedAudioProcessor:
    """Enhanced audio processor with phoneme extraction and temporal alignment"""
    
    def __init__(self, 
                 device: str = 'cuda',
                 model_path: str = None,
                 target_fps: int = 30):
        """
        Initialize the enhanced audio processor
        
        Args:
            device: Processing device ('cuda' or 'cpu')
            model_path: Path to audio model checkpoints
            target_fps: Target video frame rate
        """
        self.device = device
        self.target_fps = target_fps
        self.sample_rate = 16000
        
        # Initialize audio encoder
        model_id = model_path or "facebook/wav2vec2-large-xlsr-53-english"
        self.audio_encoder = AudioEncoder(device=device, model_id=model_id)
        self.audio_encoder.video_rate = target_fps
        
        logger.info(f"Enhanced audio processor initialized on {device}")
    
    def extract_features(self, 
                        audio_path: str,
                        return_phonemes: bool = True,
                        return_timing: bool = True) -> Dict:
        """
        Extract comprehensive audio features
        
        Args:
            audio_path: Path to audio file
            return_phonemes: Whether to extract phoneme information
            return_timing: Whether to extract timing information
            
        Returns:
            Dictionary containing audio features and metadata
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = len(audio) / sr
            
            # Extract basic features using Wan2.2 encoder
            audio_features = self.audio_encoder.extract_audio_feat(
                audio_path, 
                return_all_layers=False,
                dtype=torch.float32
            )
            
            # Convert to numpy for processing
            features_np = audio_features.squeeze(0).cpu().numpy()
            
            result = {
                'features': features_np,
                'duration': duration,
                'sample_rate': sr,
                'num_frames': features_np.shape[0],
                'feature_dim': features_np.shape[1],
                'fps': self.target_fps
            }
            
            # Extract additional features if requested
            if return_phonemes:
                result['phonemes'] = self._extract_phonemes(audio, sr)
            
            if return_timing:
                result['timing'] = self._extract_timing(audio, sr, features_np.shape[0])
            
            # Extract spectral features
            result['spectral'] = self._extract_spectral_features(audio, sr)
            
            logger.info(f"Extracted features: {result['num_frames']} frames, {result['feature_dim']} dims")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            raise
    
    def _extract_phonemes(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract phoneme information from audio"""
        try:
            # This is a simplified phoneme extraction
            # In a real implementation, you'd use a phoneme recognition model
            
            # Extract MFCC features for phoneme-like representation
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Calculate phoneme-like segments using energy and spectral centroid
            hop_length = 512
            frame_length = 2048
            
            # Energy-based segmentation
            energy = librosa.feature.rms(y=audio, hop_length=hop_length, frame_length=frame_length)[0]
            
            # Find speech segments (above threshold)
            energy_threshold = np.percentile(energy, 20)
            speech_frames = energy > energy_threshold
            
            # Group consecutive speech frames into segments
            segments = self._group_consecutive_frames(speech_frames, hop_length, sr)
            
            return {
                'mfcc': mfcc,
                'energy': energy,
                'segments': segments,
                'speech_frames': speech_frames,
                'num_segments': len(segments)
            }
            
        except Exception as e:
            logger.warning(f"Error extracting phonemes: {e}")
            return {'segments': [], 'num_segments': 0}
    
    def _extract_timing(self, audio: np.ndarray, sr: int, num_video_frames: int) -> Dict:
        """Extract timing information for synchronization"""
        try:
            # Beat tracking for rhythm
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            
            # Onset detection for speech starts
            onset_frames = librosa.onset.onset_detect(
                y=audio, sr=sr, units='time', hop_length=512
            )
            
            # Map to video frames
            video_frame_times = np.linspace(0, len(audio)/sr, num_video_frames)
            
            # Find closest onsets for each video frame
            onset_alignment = []
            for frame_time in video_frame_times:
                if len(onset_frames) > 0:
                    closest_onset_idx = np.argmin(np.abs(onset_frames - frame_time))
                    onset_alignment.append({
                        'frame_time': frame_time,
                        'closest_onset': onset_frames[closest_onset_idx],
                        'onset_distance': abs(onset_frames[closest_onset_idx] - frame_time)
                    })
                else:
                    onset_alignment.append({
                        'frame_time': frame_time,
                        'closest_onset': 0,
                        'onset_distance': frame_time
                    })
            
            return {
                'tempo': tempo,
                'beats': beats,
                'onsets': onset_frames,
                'onset_alignment': onset_alignment,
                'video_frame_times': video_frame_times
            }
            
        except Exception as e:
            logger.warning(f"Error extracting timing: {e}")
            return {'tempo': 120, 'beats': [], 'onsets': []}
    
    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract spectral features for enhanced processing"""
        try:
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            
            # Spectral rolloff (frequency below which 85% of energy is concentrated)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            
            # Zero crossing rate (speech/music discrimination)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Chroma features (harmonic content)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            
            return {
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'zero_crossing_rate': zcr,
                'chroma': chroma,
                'spectral_mean': np.mean(spectral_centroid),
                'spectral_std': np.std(spectral_centroid)
            }
            
        except Exception as e:
            logger.warning(f"Error extracting spectral features: {e}")
            return {}
    
    def _group_consecutive_frames(self, speech_frames: np.ndarray, 
                                 hop_length: int, sr: int) -> List[Dict]:
        """Group consecutive speech frames into segments"""
        segments = []
        in_segment = False
        start_frame = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_segment:
                # Start of new segment
                start_frame = i
                in_segment = True
            elif not is_speech and in_segment:
                # End of segment
                start_time = start_frame * hop_length / sr
                end_time = i * hop_length / sr
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'start_frame': start_frame,
                    'end_frame': i
                })
                in_segment = False
        
        # Handle case where audio ends during a segment
        if in_segment:
            start_time = start_frame * hop_length / sr
            end_time = len(speech_frames) * hop_length / sr
            segments.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'start_frame': start_frame,
                'end_frame': len(speech_frames)
            })
        
        return segments
    
    def align_audio_to_video(self, 
                           audio_features: np.ndarray,
                           video_fps: int,
                           target_frames: int) -> np.ndarray:
        """Align audio features to video frames"""
        try:
            # Convert to torch tensor for interpolation
            features_tensor = torch.from_numpy(audio_features).unsqueeze(0)  # Add batch dim
            features_tensor = features_tensor.transpose(1, 2)  # [B, D, T]
            
            # Interpolate to target frame count
            aligned_features = F.interpolate(
                features_tensor,
                size=target_frames,
                mode='linear',
                align_corners=True
            )
            
            # Convert back to numpy
            aligned_features = aligned_features.transpose(1, 2).squeeze(0).numpy()
            
            logger.info(f"Aligned audio: {audio_features.shape} -> {aligned_features.shape}")
            return aligned_features
            
        except Exception as e:
            logger.error(f"Error aligning audio to video: {e}")
            raise
    
    def chunk_audio_features(self, 
                           features: np.ndarray,
                           chunk_size: int,
                           overlap: int = 2) -> List[Dict]:
        """Split audio features into chunks for parallel processing"""
        try:
            chunks = []
            num_frames = features.shape[0]
            
            if num_frames <= chunk_size:
                # Single chunk
                chunks.append({
                    'features': features,
                    'start_frame': 0,
                    'end_frame': num_frames,
                    'chunk_id': 0
                })
            else:
                # Multiple chunks with overlap
                step = chunk_size - overlap
                chunk_id = 0
                
                for start in range(0, num_frames - overlap, step):
                    end = min(start + chunk_size, num_frames)
                    
                    chunks.append({
                        'features': features[start:end],
                        'start_frame': start,
                        'end_frame': end,
                        'chunk_id': chunk_id,
                        'overlap_start': max(0, overlap if start > 0 else 0),
                        'overlap_end': max(0, overlap if end < num_frames else 0)
                    })
                    
                    chunk_id += 1
                    
                    if end >= num_frames:
                        break
            
            logger.info(f"Created {len(chunks)} audio chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking audio features: {e}")
            raise
    
    def enhance_audio_quality(self, audio_path: str, output_path: str = None) -> str:
        """Enhance audio quality before processing"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Noise reduction (simple spectral gating)
            audio_denoised = self._spectral_gating(audio, sr)
            
            # Normalize audio
            audio_normalized = librosa.util.normalize(audio_denoised)
            
            # Resample to target rate if needed
            if sr != self.sample_rate:
                audio_resampled = librosa.resample(
                    audio_normalized, orig_sr=sr, target_sr=self.sample_rate
                )
            else:
                audio_resampled = audio_normalized
            
            # Save enhanced audio
            if output_path is None:
                output_path = audio_path.replace('.', '_enhanced.')
            
            sf.write(output_path, audio_resampled, self.sample_rate)
            
            logger.info(f"Enhanced audio saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error enhancing audio: {e}")
            return audio_path  # Return original if enhancement fails
    
    def _spectral_gating(self, audio: np.ndarray, sr: int, 
                        gate_threshold: float = -30) -> np.ndarray:
        """Simple spectral gating for noise reduction"""
        try:
            # Compute STFT
            stft = librosa.stft(audio, hop_length=512, n_fft=2048)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Calculate power in dB
            power_db = librosa.amplitude_to_db(magnitude, ref=np.max)
            
            # Create mask based on threshold
            mask = power_db > gate_threshold
            
            # Apply mask
            gated_magnitude = magnitude * mask
            
            # Reconstruct audio
            gated_stft = gated_magnitude * np.exp(1j * phase)
            audio_gated = librosa.istft(gated_stft, hop_length=512)
            
            return audio_gated
            
        except Exception as e:
            logger.warning(f"Spectral gating failed: {e}")
            return audio  # Return original if gating fails
    
    def get_audio_statistics(self, features: np.ndarray) -> Dict:
        """Get statistical information about audio features"""
        return {
            'shape': features.shape,
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0),
            'energy': np.sum(features ** 2, axis=1),
            'total_energy': np.sum(features ** 2)
        }