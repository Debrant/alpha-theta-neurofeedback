"""
Real-time Audio Feedback System for Alpha-Theta Neurofeedback

Implements the audio feedback component of the Peniston-Kulkosky protocol
with binaural tones, nature sounds, and adaptive volume control.
"""

import numpy as np
import pyaudio
import threading
import time
import queue
from typing import Dict, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass


class FeedbackMode(Enum):
    """Different audio feedback modes"""
    BINAURAL_BEATS = "binaural"        # Binaural beats for brainwave entrainment
    NATURE_SOUNDS = "nature"           # Ocean waves, rain, etc.
    PURE_TONES = "tones"              # Simple sine waves
    HYBRID = "hybrid"                  # Combination approach
    SILENCE = "silence"                # No audio feedback


@dataclass
class FeedbackParameters:
    """Parameters for audio feedback generation"""
    mode: FeedbackMode = FeedbackMode.BINAURAL_BEATS
    base_frequency: float = 200.0      # Base carrier frequency (Hz)
    alpha_target_freq: float = 10.0    # Target alpha frequency (Hz)
    theta_target_freq: float = 6.0     # Target theta frequency (Hz)
    volume_range: Tuple[float, float] = (0.1, 0.8)  # Min/max volume
    fade_time: float = 2.0             # Fade transition time (seconds)
    enable_reward_chime: bool = True   # Play reward sounds
    sample_rate: int = 44100           # Audio sample rate


class AudioFeedbackSystem:
    """
    Real-time audio feedback system for neurofeedback training.
    
    Provides continuous audio feedback based on EEG analysis results,
    implementing the therapeutic protocols described in van der Kolk's work.
    """
    
    def __init__(self, params: FeedbackParameters = None):
        """Initialize the audio feedback system"""
        self.params = params or FeedbackParameters()
        
        # Audio setup
        self.sample_rate = self.params.sample_rate
        self.chunk_size = 1024  # Audio buffer size
        self.channels = 2       # Stereo for binaural beats
        
        # PyAudio interface
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_playing = False
        
        # Audio generation
        self.phase_left = 0.0
        self.phase_right = 0.0
        self.current_volume = 0.0
        self.target_volume = 0.0
        
        # Feedback control
        self.feedback_queue = queue.Queue(maxsize=100)
        self.audio_thread = None
        self.stop_event = threading.Event()
        
        # Reward system
        self.last_reward_time = 0.0
        self.reward_duration = 0.5  # seconds
        
        # Nature sounds (precomputed)
        self.nature_sounds = self._generate_nature_base()
        self.nature_index = 0
        
    def start(self):
        """Start the audio feedback system"""
        if self.is_playing:
            return
            
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_playing = True
            self.stop_event.clear()
            
            # Start audio processing thread
            self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
            self.audio_thread.start()
            
            print("Audio feedback system started")
            
        except Exception as e:
            print(f"Failed to start audio system: {e}")
            self.stop()
    
    def stop(self):
        """Stop the audio feedback system"""
        if not self.is_playing:
            return
            
        self.is_playing = False
        self.stop_event.set()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
            
        print("Audio feedback system stopped")
    
    def update_feedback(self, feedback_signals: Dict[str, float]):
        """
        Update audio feedback based on EEG analysis results.
        
        Args:
            feedback_signals: Dictionary from AlphaThetaDetector.get_feedback_signals()
        """
        if not self.is_playing:
            return
            
        try:
            # Add to processing queue (non-blocking)
            self.feedback_queue.put_nowait(feedback_signals)
        except queue.Full:
            # Queue full, skip this update (maintaining real-time performance)
            pass
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for real-time audio generation"""
        try:
            # Generate audio based on current feedback state
            audio_data = self._generate_audio_chunk(frame_count)
            return (audio_data.astype(np.float32).tobytes(), pyaudio.paContinue)
        except Exception as e:
            print(f"Audio callback error: {e}")
            return (np.zeros((frame_count, self.channels), dtype=np.float32).tobytes(), pyaudio.paContinue)
    
    def _audio_worker(self):
        """Background thread for processing feedback updates"""
        while not self.stop_event.is_set():
            try:
                # Process feedback updates
                if not self.feedback_queue.empty():
                    feedback = self.feedback_queue.get_nowait()
                    self._process_feedback(feedback)
                
                time.sleep(0.01)  # 10ms processing interval
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio worker error: {e}")
                time.sleep(0.1)
    
    def _process_feedback(self, feedback: Dict[str, float]):
        """Process feedback signals and update audio parameters"""
        combined_feedback = feedback.get('combined_feedback', 0.0)
        reward_active = feedback.get('reward_active', False)
        alpha_relative = feedback.get('alpha_relative', 1.0)
        theta_relative = feedback.get('theta_relative', 1.0)
        
        # Update target volume based on feedback strength
        min_vol, max_vol = self.params.volume_range
        self.target_volume = min_vol + (max_vol - min_vol) * combined_feedback
        
        # Play reward chime if threshold reached
        if reward_active and self.params.enable_reward_chime:
            current_time = time.time()
            if current_time - self.last_reward_time > 2.0:  # Minimum 2s between rewards
                self._trigger_reward_chime()
                self.last_reward_time = current_time
    
    def _generate_audio_chunk(self, frame_count: int) -> np.ndarray:
        """Generate a chunk of audio data based on current parameters"""
        # Smooth volume transitions
        self.current_volume = self._smooth_transition(
            self.current_volume, 
            self.target_volume, 
            frame_count / self.sample_rate
        )
        
        # Generate audio based on mode
        if self.params.mode == FeedbackMode.BINAURAL_BEATS:
            audio_chunk = self._generate_binaural_beats(frame_count)
        elif self.params.mode == FeedbackMode.NATURE_SOUNDS:
            audio_chunk = self._generate_nature_sounds(frame_count)
        elif self.params.mode == FeedbackMode.PURE_TONES:
            audio_chunk = self._generate_pure_tones(frame_count)
        elif self.params.mode == FeedbackMode.HYBRID:
            audio_chunk = self._generate_hybrid_audio(frame_count)
        else:  # SILENCE
            audio_chunk = np.zeros((frame_count, self.channels))
        
        # Apply volume
        audio_chunk *= self.current_volume
        
        return audio_chunk
    
    def _generate_binaural_beats(self, frame_count: int) -> np.ndarray:
        """
        Generate binaural beats for brainwave entrainment.
        
        Left ear: base_frequency
        Right ear: base_frequency + target_frequency
        Brain perceives the difference as the target frequency
        """
        # Calculate frequencies for alpha/theta entrainment
        # Use weighted average based on current feedback state
        alpha_weight = 0.6  # Slight alpha emphasis for relaxed awareness
        theta_weight = 0.4
        
        target_beat_freq = (
            alpha_weight * self.params.alpha_target_freq +
            theta_weight * self.params.theta_target_freq
        )
        
        left_freq = self.params.base_frequency
        right_freq = self.params.base_frequency + target_beat_freq
        
        # Generate time array
        t = np.arange(frame_count) / self.sample_rate
        
        # Generate stereo audio
        audio_chunk = np.zeros((frame_count, 2))
        
        # Left channel
        self.phase_left += 2 * np.pi * left_freq * frame_count / self.sample_rate
        audio_chunk[:, 0] = np.sin(2 * np.pi * left_freq * t + self.phase_left % (2 * np.pi))
        
        # Right channel
        self.phase_right += 2 * np.pi * right_freq * frame_count / self.sample_rate
        audio_chunk[:, 1] = np.sin(2 * np.pi * right_freq * t + self.phase_right % (2 * np.pi))
        
        # Keep phases reasonable
        if self.phase_left > 10 * np.pi:
            self.phase_left -= 10 * np.pi
        if self.phase_right > 10 * np.pi:
            self.phase_right -= 10 * np.pi
        
        return audio_chunk * 0.3  # Reduce amplitude for comfort
    
    def _generate_nature_sounds(self, frame_count: int) -> np.ndarray:
        """Generate nature-based background sounds"""
        # Use precomputed nature sounds
        audio_chunk = np.zeros((frame_count, 2))
        
        for i in range(frame_count):
            audio_chunk[i, 0] = self.nature_sounds[self.nature_index % len(self.nature_sounds)]
            audio_chunk[i, 1] = self.nature_sounds[self.nature_index % len(self.nature_sounds)]
            self.nature_index += 1
            
        return audio_chunk
    
    def _generate_pure_tones(self, frame_count: int) -> np.ndarray:
        """Generate simple pure tones"""
        # Gentle sine wave at alpha frequency
        t = np.arange(frame_count) / self.sample_rate
        freq = self.params.alpha_target_freq * 20  # Audible frequency
        
        self.phase_left += 2 * np.pi * freq * frame_count / self.sample_rate
        tone = np.sin(2 * np.pi * freq * t + self.phase_left % (2 * np.pi))
        
        audio_chunk = np.column_stack((tone, tone)) * 0.4
        
        if self.phase_left > 10 * np.pi:
            self.phase_left -= 10 * np.pi
            
        return audio_chunk
    
    def _generate_hybrid_audio(self, frame_count: int) -> np.ndarray:
        """Generate combination of binaural beats and nature sounds"""
        binaural = self._generate_binaural_beats(frame_count) * 0.7
        nature = self._generate_nature_sounds(frame_count) * 0.3
        return binaural + nature
    
    def _generate_nature_base(self) -> np.ndarray:
        """Generate base nature sounds (simplified ocean waves)"""
        duration = 30.0  # 30 seconds of base audio
        samples = int(duration * self.sample_rate)
        
        # Create ocean-like sound using filtered noise
        noise = np.random.normal(0, 0.3, samples)
        
        # Apply low-pass filter for wave-like sound
        from scipy import signal
        b, a = signal.butter(4, 500, btype='low', fs=self.sample_rate)
        ocean_sound = signal.filtfilt(b, a, noise)
        
        # Add slow amplitude modulation for wave rhythm
        wave_freq = 0.3  # Wave every ~3 seconds
        t = np.arange(samples) / self.sample_rate
        wave_envelope = 0.7 + 0.3 * np.sin(2 * np.pi * wave_freq * t)
        
        return ocean_sound * wave_envelope
    
    def _smooth_transition(self, current: float, target: float, time_step: float) -> float:
        """Smooth transition between current and target values"""
        if abs(target - current) < 0.001:
            return target
            
        # Exponential smoothing
        alpha = 1.0 - np.exp(-time_step / self.params.fade_time)
        return current + alpha * (target - current)
    
    def _trigger_reward_chime(self):
        """Trigger a reward chime (simple implementation)"""
        # This is a placeholder - in practice you might queue a special audio event
        # For now, we just boost the volume briefly
        self.target_volume = min(1.0, self.target_volume * 1.2)
    
    def set_mode(self, mode: FeedbackMode):
        """Change feedback mode during operation"""
        self.params.mode = mode
        print(f"Audio feedback mode changed to: {mode.value}")
    
    def set_volume_range(self, min_vol: float, max_vol: float):
        """Update volume range during operation"""
        self.params.volume_range = (
            max(0.0, min(1.0, min_vol)),
            max(0.0, min(1.0, max_vol))
        )
    
    def get_status(self) -> Dict[str, any]:
        """Get current status of the audio system"""
        return {
            'is_playing': self.is_playing,
            'current_volume': self.current_volume,
            'target_volume': self.target_volume,
            'mode': self.params.mode.value,
            'queue_size': self.feedback_queue.qsize(),
            'sample_rate': self.sample_rate
        }
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop()
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()