"""
EEG Signal Simulator for Alpha-Theta Neurofeedback Testing

Generates realistic EEG signals with controllable alpha and theta content
for testing and development of neurofeedback algorithms.
"""

import numpy as np
import scipy.signal as signal
from typing import Optional, Tuple
from enum import Enum


class EEGState(Enum):
    """Different EEG states for simulation"""
    AWAKE_ALERT = "awake_alert"          # High beta, moderate alpha
    RELAXED_EYES_CLOSED = "relaxed"       # High alpha, low beta  
    DROWSY = "drowsy"                     # High theta, decreasing alpha
    MEDITATION_LIGHT = "meditation_light" # Increased alpha
    MEDITATION_DEEP = "meditation_deep"   # High theta with alpha
    ARTIFACT_MOVEMENT = "artifact"        # Movement artifacts
    POOR_SIGNAL = "poor_signal"          # High noise, poor contact


class EEGSimulator:
    """
    Simulate realistic EEG signals for alpha-theta neurofeedback testing.
    
    Generates multi-band EEG signals with realistic noise characteristics,
    artifacts, and state transitions for comprehensive testing.
    """
    
    def __init__(self, sampling_rate: int = 250):
        """
        Initialize EEG simulator.
        
        Args:
            sampling_rate: Sample rate in Hz
        """
        self.fs = sampling_rate
        self.time_step = 1.0 / sampling_rate
        self.current_time = 0.0
        
        # Phase accumulators for coherent signal generation
        self.phases = {
            'delta': 0.0,    # 1-4 Hz
            'theta': 0.0,    # 4-8 Hz 
            'alpha': 0.0,    # 8-12 Hz
            'beta': 0.0,     # 12-30 Hz
            'gamma': 0.0,    # 30-100 Hz
        }
        
        # Noise generators
        self.noise_generator = np.random.RandomState(42)  # Reproducible noise
        
    def generate_sample(self, state: EEGState = EEGState.RELAXED_EYES_CLOSED,
                       alpha_amplitude: float = 1.0,
                       theta_amplitude: float = 1.0,
                       noise_level: float = 0.2) -> float:
        """
        Generate a single EEG sample based on specified state.
        
        Args:
            state: Current EEG state
            alpha_amplitude: Alpha band amplitude multiplier
            theta_amplitude: Theta band amplitude multiplier  
            noise_level: Noise level (0.0 to 1.0)
            
        Returns:
            Single EEG sample in microvolts
        """
        sample = 0.0
        
        # Get state-specific parameters
        params = self._get_state_parameters(state)
        
        # Generate each frequency band
        for band, config in params.items():
            if band in self.phases:
                frequency = config['freq'] + self.noise_generator.normal(0, config['freq_jitter'])
                amplitude = config['amplitude']
                
                # Apply user amplitude modulation
                if band == 'alpha':
                    amplitude *= alpha_amplitude
                elif band == 'theta':
                    amplitude *= theta_amplitude
                    
                # Generate sinusoid with phase continuity
                sample += amplitude * np.sin(self.phases[band])
                self.phases[band] += 2 * np.pi * frequency * self.time_step
                
                # Keep phase in reasonable range
                if self.phases[band] > 10 * np.pi:
                    self.phases[band] -= 10 * np.pi
        
        # Add realistic noise
        sample += self._generate_noise(noise_level, state)
        
        # Add artifacts based on state
        sample += self._generate_artifacts(state)
        
        # Update time
        self.current_time += self.time_step
        
        return sample
    
    def generate_signal(self, duration: float, 
                       state: EEGState = EEGState.RELAXED_EYES_CLOSED,
                       alpha_modulation: Optional[np.ndarray] = None,
                       theta_modulation: Optional[np.ndarray] = None,
                       noise_level: float = 0.2) -> np.ndarray:
        """
        Generate a continuous EEG signal.
        
        Args:
            duration: Signal duration in seconds
            state: EEG state for the signal
            alpha_modulation: Time-varying alpha amplitude (optional)
            theta_modulation: Time-varying theta amplitude (optional)
            noise_level: Background noise level
            
        Returns:
            EEG signal array in microvolts
        """
        num_samples = int(duration * self.fs)
        signal_data = np.zeros(num_samples)
        
        for i in range(num_samples):
            # Get modulation values if provided
            alpha_amp = alpha_modulation[i] if alpha_modulation is not None else 1.0
            theta_amp = theta_modulation[i] if theta_modulation is not None else 1.0
            
            signal_data[i] = self.generate_sample(
                state=state,
                alpha_amplitude=alpha_amp,
                theta_amplitude=theta_amp,
                noise_level=noise_level
            )
            
        return signal_data
    
    def generate_neurofeedback_session(self, duration: float = 300.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a realistic neurofeedback session with state transitions.
        
        Simulates a typical 5-minute session with:
        - Initial alertness
        - Gradual relaxation  
        - Periods of deep theta states
        - Occasional artifacts
        
        Args:
            duration: Session duration in seconds
            
        Returns:
            Tuple of (eeg_signal, state_labels) arrays
        """
        num_samples = int(duration * self.fs)
        signal_data = np.zeros(num_samples)
        state_labels = np.zeros(num_samples)
        
        # Define session timeline
        timeline = [
            (0.0, 30.0, EEGState.AWAKE_ALERT),        # Initial alertness
            (30.0, 60.0, EEGState.RELAXED_EYES_CLOSED), # Early relaxation
            (60.0, 180.0, EEGState.MEDITATION_LIGHT),  # Light meditation
            (180.0, 240.0, EEGState.MEDITATION_DEEP),  # Deep theta state
            (240.0, 270.0, EEGState.DROWSY),          # Getting drowsy
            (270.0, 300.0, EEGState.RELAXED_EYES_CLOSED), # Back to relaxed
        ]
        
        # Add occasional artifacts
        artifact_times = [90.0, 150.0, 210.0]  # Movement artifacts
        
        current_segment = 0
        for i in range(num_samples):
            time_sec = i / self.fs
            
            # Determine current state
            while (current_segment < len(timeline) - 1 and 
                   time_sec >= timeline[current_segment + 1][0]):
                current_segment += 1
                
            current_state = timeline[current_segment][2]
            
            # Check for artifacts
            artifact_active = any(abs(time_sec - t) < 2.0 for t in artifact_times)
            if artifact_active:
                current_state = EEGState.ARTIFACT_MOVEMENT
            
            # Generate sample with gradual state transitions
            alpha_mod = self._get_alpha_modulation(time_sec, timeline, current_segment)
            theta_mod = self._get_theta_modulation(time_sec, timeline, current_segment)
            
            signal_data[i] = self.generate_sample(
                state=current_state,
                alpha_amplitude=alpha_mod,
                theta_amplitude=theta_mod,
                noise_level=0.15
            )
            
            state_labels[i] = list(EEGState).index(current_state)
            
        return signal_data, state_labels
    
    def _get_state_parameters(self, state: EEGState) -> dict:
        """Get frequency band parameters for different EEG states"""
        base_params = {
            'delta': {'freq': 2.5, 'amplitude': 10.0, 'freq_jitter': 0.3},
            'theta': {'freq': 6.0, 'amplitude': 8.0, 'freq_jitter': 0.5}, 
            'alpha': {'freq': 10.0, 'amplitude': 15.0, 'freq_jitter': 0.8},
            'beta': {'freq': 20.0, 'amplitude': 5.0, 'freq_jitter': 2.0},
            'gamma': {'freq': 40.0, 'amplitude': 2.0, 'freq_jitter': 5.0},
        }
        
        # Modify amplitudes based on state
        if state == EEGState.AWAKE_ALERT:
            base_params['alpha']['amplitude'] *= 0.7
            base_params['beta']['amplitude'] *= 2.0
            base_params['theta']['amplitude'] *= 0.3
            
        elif state == EEGState.RELAXED_EYES_CLOSED:
            base_params['alpha']['amplitude'] *= 1.5
            base_params['beta']['amplitude'] *= 0.5
            base_params['theta']['amplitude'] *= 0.8
            
        elif state == EEGState.DROWSY:
            base_params['alpha']['amplitude'] *= 0.8
            base_params['theta']['amplitude'] *= 2.0
            base_params['delta']['amplitude'] *= 1.5
            
        elif state == EEGState.MEDITATION_LIGHT:
            base_params['alpha']['amplitude'] *= 2.0
            base_params['beta']['amplitude'] *= 0.3
            base_params['theta']['amplitude'] *= 1.2
            
        elif state == EEGState.MEDITATION_DEEP:
            base_params['alpha']['amplitude'] *= 1.5
            base_params['theta']['amplitude'] *= 2.5
            base_params['delta']['amplitude'] *= 0.8
            
        elif state == EEGState.ARTIFACT_MOVEMENT:
            # High amplitude, broadband artifact
            for band in base_params:
                base_params[band]['amplitude'] *= 3.0
                base_params[band]['freq_jitter'] *= 5.0
                
        elif state == EEGState.POOR_SIGNAL:
            # Reduced signal, increased noise  
            for band in base_params:
                base_params[band]['amplitude'] *= 0.3
                
        return base_params
    
    def _generate_noise(self, noise_level: float, state: EEGState) -> float:
        """Generate realistic EEG noise"""
        if state == EEGState.POOR_SIGNAL:
            noise_level *= 3.0
        elif state == EEGState.ARTIFACT_MOVEMENT:
            noise_level *= 2.0
            
        # Pink noise (1/f characteristic)
        white_noise = self.noise_generator.normal(0, noise_level * 5.0)
        return white_noise
    
    def _generate_artifacts(self, state: EEGState) -> float:
        """Generate typical EEG artifacts"""
        artifact = 0.0
        
        if state == EEGState.ARTIFACT_MOVEMENT:
            # Large, slow movement artifact
            artifact += self.noise_generator.normal(0, 20.0)
            # Add occasional spikes
            if self.noise_generator.random() < 0.01:  # 1% chance per sample
                artifact += self.noise_generator.normal(0, 50.0)
                
        elif state == EEGState.POOR_SIGNAL:
            # Intermittent signal dropout
            if self.noise_generator.random() < 0.05:  # 5% dropout rate
                artifact = -1000.0  # Signal loss
                
        return artifact
    
    def _get_alpha_modulation(self, time_sec: float, timeline: list, current_segment: int) -> float:
        """Get time-varying alpha modulation for realistic transitions"""
        base_modulation = 1.0
        
        # Add some natural variation
        base_modulation += 0.2 * np.sin(2 * np.pi * time_sec / 60.0)  # 1-minute cycle
        base_modulation += 0.1 * np.sin(2 * np.pi * time_sec / 15.0)  # 15-second variation
        
        return max(0.1, base_modulation)
    
    def _get_theta_modulation(self, time_sec: float, timeline: list, current_segment: int) -> float:
        """Get time-varying theta modulation for realistic transitions"""
        base_modulation = 1.0
        
        # Theta often increases gradually during meditation
        if time_sec > 120.0:  # After 2 minutes
            base_modulation += 0.5 * ((time_sec - 120.0) / 180.0)  # Gradual increase
            
        # Add natural variation
        base_modulation += 0.15 * np.sin(2 * np.pi * time_sec / 45.0)  # 45-second cycle
        
        return max(0.1, base_modulation)
    
    def reset(self):
        """Reset simulator to initial state"""
        self.current_time = 0.0
        for band in self.phases:
            self.phases[band] = 0.0