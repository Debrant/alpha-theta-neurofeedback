"""
Alpha-Theta Neurofeedback Signal Processing Module

Implementation of the Peniston-Kulkosky alpha-theta protocol as described in
van der Kolk's "The Body Keeps the Score" for neurofeedback training.

Based on research by Eugene Peniston and Paul Kulkosky at the VA Medical Center,
Fort Lyon, Colorado for PTSD treatment through neurofeedback.
"""

import numpy as np
import scipy.signal as signal
import time
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class BandPowerResult:
    """Results from alpha-theta band power analysis"""
    alpha_power: float
    theta_power: float
    alpha_theta_ratio: float
    dominant_frequency: float
    signal_quality: float
    timestamp: float


class AlphaThetaDetector:
    """
    Real-time alpha-theta wave detector for neurofeedback training.
    
    Implements the Peniston-Kulkosky protocol:
    - Alpha band: 8-12 Hz (conscious awareness, relaxed focus)
    - Theta band: 4-8 Hz (deep relaxation, memory access)
    - Protocol alternately rewards both bands for therapeutic effect
    """
    
    def __init__(self, 
                 sampling_rate: int = 250,
                 window_size: float = 2.0,
                 stride: float = 1.0,
                 alpha_band: Tuple[float, float] = (8.0, 12.0),
                 theta_band: Tuple[float, float] = (4.0, 8.0),
                 notch_freq: float = 60.0):
        """
        Initialize the alpha-theta detector.
        
        Args:
            sampling_rate: Sample rate in Hz
            window_size: Analysis window size in seconds
            stride: Time between analyses in seconds
            alpha_band: Alpha frequency range (low, high) in Hz
            theta_band: Theta frequency range (low, high) in Hz
            notch_freq: Power line frequency to filter (50/60 Hz)
        """
        self.fs = sampling_rate
        self.window_size = window_size
        self.stride = stride
        self.window_samples = int(window_size * sampling_rate)
        self.stride_samples = int(stride * sampling_rate)
        self.alpha_band = alpha_band
        self.theta_band = theta_band
        
        # Design filters
        self._setup_filters(notch_freq)
        
        # Circular buffer for real-time processing
        self.buffer = np.zeros(self.window_samples)
        self.buffer_index = 0
        self.buffer_full = False
        self.samples_since_analysis = 0
        
        # Baseline calibration
        self.baseline_alpha = None
        self.baseline_theta = None
        self.calibration_samples = []
        self.calibrated = False
        
    def _setup_filters(self, notch_freq: float):
        """Setup bandpass and notch filters for EEG processing"""
        # High-pass filter to remove DC and slow drifts (0.5 Hz)
        self.hp_b, self.hp_a = signal.butter(4, 0.5, btype='high', fs=self.fs)
        
        # Low-pass filter to remove high-frequency noise (30 Hz)
        self.lp_b, self.lp_a = signal.butter(4, 30, btype='low', fs=self.fs)
        
        # Notch filter for power line interference
        if notch_freq > 0:
            self.notch_b, self.notch_a = signal.iirnotch(notch_freq, Q=30, fs=self.fs)
        else:
            self.notch_b, self.notch_a = None, None
            
    def add_sample(self, sample: float) -> Optional[BandPowerResult]:
        """
        Add a new EEG sample and return analysis if stride interval is reached.
        
        Args:
            sample: Single EEG sample value
            
        Returns:
            BandPowerResult if stride interval reached and window is full, None otherwise
        """
        # Add sample to circular buffer
        self.buffer[self.buffer_index] = sample
        self.buffer_index = (self.buffer_index + 1) % self.window_samples
        self.samples_since_analysis += 1
        
        if not self.buffer_full and self.buffer_index == 0:
            self.buffer_full = True
            
        # Only analyze when we have a full window and stride interval is reached
        if self.buffer_full and self.samples_since_analysis >= self.stride_samples:
            self.samples_since_analysis = 0
            return self._analyze_window()
        
        return None
    
    def add_samples(self, samples: np.ndarray) -> list[BandPowerResult]:
        """
        Add multiple samples and return all completed analyses.
        
        Args:
            samples: Array of EEG samples
            
        Returns:
            List of BandPowerResult objects for completed windows
        """
        results = []
        for sample in samples:
            result = self.add_sample(sample)
            if result is not None:
                results.append(result)
        return results
    
    def _analyze_window(self) -> BandPowerResult:
        """Analyze the current window buffer for alpha-theta content"""
        # Get current window (handle circular buffer)
        if self.buffer_index == 0:
            window_data = self.buffer.copy()
        else:
            window_data = np.concatenate([
                self.buffer[self.buffer_index:],
                self.buffer[:self.buffer_index]
            ])
        
        # Apply filtering
        filtered_data = self._filter_signal(window_data)
        
        # Calculate power spectral density
        freqs, psd = signal.welch(filtered_data, self.fs, nperseg=len(filtered_data)//2)
        
        # Extract band powers
        alpha_power = self._get_band_power(freqs, psd, self.alpha_band)
        theta_power = self._get_band_power(freqs, psd, self.theta_band)
        
        # Calculate alpha/theta ratio
        alpha_theta_ratio = alpha_power / (theta_power + 1e-10)  # Avoid division by zero
        
        # Find dominant frequency
        dominant_freq = freqs[np.argmax(psd)]
        
        # Assess signal quality (SNR approximation)
        signal_quality = self._assess_signal_quality(filtered_data, psd)
        
        return BandPowerResult(
            alpha_power=alpha_power,
            theta_power=theta_power,
            alpha_theta_ratio=alpha_theta_ratio,
            dominant_frequency=dominant_freq,
            signal_quality=signal_quality,
            timestamp=time.time()
        )
    
    def _filter_signal(self, data: np.ndarray) -> np.ndarray:
        """Apply EEG preprocessing filters"""
        # High-pass filter
        filtered = signal.filtfilt(self.hp_b, self.hp_a, data)
        
        # Low-pass filter
        filtered = signal.filtfilt(self.lp_b, self.lp_a, filtered)
        
        # Notch filter (if configured)
        if self.notch_b is not None:
            filtered = signal.filtfilt(self.notch_b, self.notch_a, filtered)
            
        return filtered
    
    def _get_band_power(self, freqs: np.ndarray, psd: np.ndarray, 
                       band: Tuple[float, float]) -> float:
        """Extract power in a specific frequency band"""
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        return np.trapz(psd[band_mask], freqs[band_mask])
    
    def _assess_signal_quality(self, filtered_data: np.ndarray, psd: np.ndarray) -> float:
        """
        Assess signal quality based on noise characteristics.
        Returns value between 0 (poor) and 1 (excellent).
        """
        # Check for signal dropout (very low amplitude)
        signal_amplitude = np.std(filtered_data)
        if signal_amplitude < 1.0:  # Very low signal
            return 0.1
            
        # Check for saturation (very high amplitude)
        if np.max(np.abs(filtered_data)) > 1000.0:  # Saturation
            return 0.2
            
        # Assess frequency domain characteristics
        freqs = np.linspace(0, self.fs/2, len(psd))
        
        # EEG power should be in reasonable bands
        eeg_band_power = self._get_band_power(freqs, psd, (1.0, 30.0))
        noise_power = self._get_band_power(freqs, psd, (30.0, self.fs/2))
        
        if eeg_band_power > 0 and noise_power >= 0:
            snr = eeg_band_power / (noise_power + 1e-10)
            quality = min(1.0, snr / 50.0)  # Normalize SNR to 0-1
        else:
            quality = 0.5
            
        # Penalize excessive high-frequency content (artifacts)
        high_freq_ratio = noise_power / (eeg_band_power + 1e-10)
        if high_freq_ratio > 2.0:  # Too much high-frequency noise
            quality *= 0.5
            
        return max(0.0, min(1.0, quality))
    
    def calibrate_baseline(self, calibration_data: np.ndarray, 
                          duration_seconds: float = 60.0):
        """
        Establish baseline alpha and theta levels for personalized feedback.
        
        Args:
            calibration_data: EEG data for baseline measurement
            duration_seconds: Duration of calibration period
        """
        print(f"Starting baseline calibration ({duration_seconds:.1f}s)...")
        
        results = self.add_samples(calibration_data)
        
        if len(results) > 0:
            alpha_powers = [r.alpha_power for r in results]
            theta_powers = [r.theta_power for r in results]
            
            self.baseline_alpha = np.mean(alpha_powers)
            self.baseline_theta = np.mean(theta_powers)
            self.calibrated = True
            
            print(f"Baseline established:")
            print(f"  Alpha: {self.baseline_alpha:.2f} μV²/Hz")
            print(f"  Theta: {self.baseline_theta:.2f} μV²/Hz")
        else:
            print("Warning: Insufficient data for calibration")
    
    def get_feedback_signals(self, result: BandPowerResult) -> Dict[str, float]:
        """
        Generate feedback signals based on current alpha-theta state.
        
        Implements Peniston-Kulkosky protocol feedback logic:
        - Rewards both alpha and theta activity
        - Alternating emphasis based on current state
        
        Returns:
            Dictionary with feedback values (0.0 to 1.0)
        """
        if not self.calibrated:
            # No feedback without baseline
            return {
                'alpha_feedback': 0.0,
                'theta_feedback': 0.0,
                'combined_feedback': 0.0,
                'reward_active': False
            }
        
        # Normalize relative to baseline
        alpha_relative = result.alpha_power / self.baseline_alpha
        theta_relative = result.theta_power / self.baseline_theta
        
        # Alpha feedback (8-12 Hz rewarded for relaxed awareness)
        alpha_feedback = min(1.0, max(0.0, (alpha_relative - 0.8) / 0.4))
        
        # Theta feedback (4-8 Hz rewarded for deep relaxation)  
        theta_feedback = min(1.0, max(0.0, (theta_relative - 0.8) / 0.4))
        
        # Combined feedback emphasizes the Peniston-Kulkosky approach
        # High theta with moderate alpha is most rewarded
        if theta_relative > 1.2 and alpha_relative > 0.8:
            combined_feedback = 0.8 + 0.2 * min(theta_feedback, alpha_feedback)
            reward_active = True
        elif theta_relative > 1.0 or alpha_relative > 1.1:
            combined_feedback = 0.5 + 0.3 * max(theta_feedback, alpha_feedback) 
            reward_active = True
        else:
            combined_feedback = 0.3 * max(theta_feedback, alpha_feedback)
            reward_active = False
            
        return {
            'alpha_feedback': alpha_feedback,
            'theta_feedback': theta_feedback, 
            'combined_feedback': combined_feedback,
            'reward_active': reward_active,
            'alpha_relative': alpha_relative,
            'theta_relative': theta_relative
        }
    
    def reset_buffer(self):
        """Reset the analysis buffer (useful between sessions)"""
        self.buffer = np.zeros(self.window_samples)
        self.buffer_index = 0
        self.buffer_full = False
        self.samples_since_analysis = 0