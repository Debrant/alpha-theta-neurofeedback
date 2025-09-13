"""
Alpha-Theta Neurofeedback Session Manager

Orchestrates complete neurofeedback sessions, integrating EEG signal processing,
audio feedback, and session logging based on the Peniston-Kulkosky protocol.
"""

import numpy as np
import time
import threading
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum

# Import our modules
from signal_processing.alpha_theta_detector import AlphaThetaDetector, BandPowerResult
from signal_processing.eeg_simulator import EEGSimulator, EEGState
from feedback.audio_feedback import AudioFeedbackSystem, FeedbackParameters, FeedbackMode


class SessionState(Enum):
    """Current state of a neurofeedback session"""
    IDLE = "idle"
    CALIBRATING = "calibrating" 
    TRAINING = "training"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SessionConfig:
    """Configuration for a neurofeedback session"""
    # Session parameters
    duration_minutes: float = 20.0
    calibration_minutes: float = 2.0
    
    # Signal processing
    sampling_rate: int = 250
    window_size: float = 2.0
    analysis_stride: float = 0.5
    
    # Frequency bands
    alpha_band: tuple = (8.0, 12.0)
    theta_band: tuple = (4.0, 8.0)
    
    # Audio feedback
    feedback_mode: str = "binaural"
    base_frequency: float = 200.0
    volume_range: tuple = (0.2, 0.7)
    enable_reward_chimes: bool = True
    
    # Session management
    auto_save: bool = True
    data_directory: str = "data/sessions"
    participant_id: str = "participant_001"


@dataclass 
class SessionMetrics:
    """Metrics collected during a session"""
    start_time: datetime
    end_time: Optional[datetime] = None
    state: SessionState = SessionState.IDLE
    
    # Calibration data
    baseline_alpha: float = 0.0
    baseline_theta: float = 0.0
    calibration_quality: float = 0.0
    
    # Training metrics
    total_samples: int = 0
    analysis_windows: int = 0
    average_alpha_power: float = 0.0
    average_theta_power: float = 0.0
    average_signal_quality: float = 0.0
    
    # Feedback metrics
    average_feedback_score: float = 0.0
    reward_time_percentage: float = 0.0
    feedback_updates: int = 0
    
    # Session quality
    artifact_percentage: float = 0.0
    signal_dropout_count: int = 0


class NeurofeedbackSession:
    """
    Complete neurofeedback session manager.
    
    Orchestrates EEG processing, audio feedback, and data logging
    for therapeutic alpha-theta training sessions.
    """
    
    def __init__(self, config: SessionConfig = None, 
                 eeg_data_source: Optional[Callable] = None):
        """
        Initialize session manager.
        
        Args:
            config: Session configuration
            eeg_data_source: Function that returns EEG samples (for real hardware)
                           If None, uses simulator for testing
        """
        self.config = config or SessionConfig()
        self.eeg_data_source = eeg_data_source
        
        # Initialize components
        self.detector = AlphaThetaDetector(
            sampling_rate=self.config.sampling_rate,
            window_size=self.config.window_size,
            stride=self.config.analysis_stride,
            alpha_band=self.config.alpha_band,
            theta_band=self.config.theta_band
        )
        
        # Audio feedback system
        feedback_params = FeedbackParameters(
            mode=FeedbackMode(self.config.feedback_mode),
            base_frequency=self.config.base_frequency,
            volume_range=self.config.volume_range,
            enable_reward_chime=self.config.enable_reward_chimes
        )
        self.audio_system = AudioFeedbackSystem(feedback_params)
        
        # EEG simulator (for testing without hardware)
        if self.eeg_data_source is None:
            self.simulator = EEGSimulator(self.config.sampling_rate)
        else:
            self.simulator = None
            
        # Session state
        self.metrics = SessionMetrics(start_time=datetime.now())
        self.is_running = False
        self.session_thread = None
        self.stop_event = threading.Event()
        
        # Data storage
        self.session_data = []
        self.raw_eeg_data = []
        
        # Event callbacks
        self.on_state_change: Optional[Callable] = None
        self.on_metrics_update: Optional[Callable] = None
        self.on_calibration_complete: Optional[Callable] = None
        
        # Create data directory
        Path(self.config.data_directory).mkdir(parents=True, exist_ok=True)
    
    def start_session(self) -> bool:
        """
        Start a complete neurofeedback session.
        
        Returns:
            True if session started successfully
        """
        if self.is_running:
            print("Session already running")
            return False
            
        try:
            print("Starting neurofeedback session...")
            
            # Reset session state
            self.metrics = SessionMetrics(start_time=datetime.now())
            self.session_data = []
            self.raw_eeg_data = []
            self.stop_event.clear()
            
            # Start audio system
            self.audio_system.start()
            
            # Start session processing thread
            self.is_running = True
            self.session_thread = threading.Thread(target=self._session_worker, daemon=True)
            self.session_thread.start()
            
            self._update_state(SessionState.CALIBRATING)
            print("Neurofeedback session started")
            return True
            
        except Exception as e:
            print(f"Failed to start session: {e}")
            self._update_state(SessionState.ERROR)
            return False
    
    def stop_session(self):
        """Stop the current session"""
        if not self.is_running:
            return
            
        print("Stopping neurofeedback session...")
        self.is_running = False
        self.stop_event.set()
        
        # Stop audio system
        self.audio_system.stop()
        
        # Wait for session thread
        if self.session_thread:
            self.session_thread.join(timeout=2.0)
            
        # Finalize metrics
        self.metrics.end_time = datetime.now()
        self._update_state(SessionState.COMPLETED)
        
        # Save session data
        if self.config.auto_save:
            self._save_session_data()
            
        print("Neurofeedback session stopped")
    
    def pause_session(self):
        """Pause the current session"""
        if self.metrics.state == SessionState.TRAINING:
            self._update_state(SessionState.PAUSED)
            self.audio_system.set_mode(FeedbackMode.SILENCE)
    
    def resume_session(self):
        """Resume a paused session"""
        if self.metrics.state == SessionState.PAUSED:
            self._update_state(SessionState.TRAINING)
            self.audio_system.set_mode(FeedbackMode(self.config.feedback_mode))
    
    def _session_worker(self):
        """Main session processing loop"""
        calibration_start = time.time()
        calibration_duration = self.config.calibration_minutes * 60.0
        training_duration = self.config.duration_minutes * 60.0
        
        calibration_data = []
        
        sample_interval = 1.0 / self.config.sampling_rate
        last_sample_time = time.time()
        
        while not self.stop_event.is_set() and self.is_running:
            current_time = time.time()
            
            # Maintain sampling rate timing
            if current_time - last_sample_time < sample_interval:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
                
            last_sample_time = current_time
            
            try:
                # Get EEG sample
                if self.simulator:
                    # Use simulator
                    if self.metrics.state == SessionState.CALIBRATING:
                        sample = self.simulator.generate_sample(EEGState.RELAXED_EYES_CLOSED)
                    else:
                        sample = self.simulator.generate_sample(EEGState.MEDITATION_LIGHT)
                else:
                    # Get from real EEG hardware
                    sample = self.eeg_data_source()
                
                self.raw_eeg_data.append(sample)
                self.metrics.total_samples += 1
                
                # Process sample through detector
                result = self.detector.add_sample(sample)
                
                if result is not None:
                    self.metrics.analysis_windows += 1
                    self._process_analysis_result(result)
                    
                    # Handle calibration phase
                    if self.metrics.state == SessionState.CALIBRATING:
                        calibration_data.append(result)
                        
                        if current_time - calibration_start >= calibration_duration:
                            self._complete_calibration(calibration_data)
                            self._update_state(SessionState.TRAINING)
                    
                    # Handle training phase
                    elif self.metrics.state == SessionState.TRAINING:
                        # Update audio feedback
                        feedback = self.detector.get_feedback_signals(result)
                        self.audio_system.update_feedback(feedback)
                        self._update_feedback_metrics(feedback)
                        
                        # Check if session duration reached
                        session_elapsed = current_time - calibration_start
                        if session_elapsed >= (calibration_duration + training_duration):
                            self.stop_session()
                            break
                
                # Update metrics periodically
                if self.metrics.total_samples % (self.config.sampling_rate * 5) == 0:  # Every 5 seconds
                    self._update_session_metrics()
                    
            except Exception as e:
                print(f"Session processing error: {e}")
                self.metrics.signal_dropout_count += 1
                time.sleep(0.01)
    
    def _process_analysis_result(self, result: BandPowerResult):
        """Process a single analysis result"""
        # Store data
        self.session_data.append({
            'timestamp': result.timestamp,
            'alpha_power': result.alpha_power,
            'theta_power': result.theta_power,
            'alpha_theta_ratio': result.alpha_theta_ratio,
            'dominant_frequency': result.dominant_frequency,
            'signal_quality': result.signal_quality
        })
        
        # Update running averages
        n = self.metrics.analysis_windows
        self.metrics.average_alpha_power = (
            (self.metrics.average_alpha_power * (n-1) + result.alpha_power) / n
        )
        self.metrics.average_theta_power = (
            (self.metrics.average_theta_power * (n-1) + result.theta_power) / n
        )
        self.metrics.average_signal_quality = (
            (self.metrics.average_signal_quality * (n-1) + result.signal_quality) / n
        )
        
        # Check for artifacts
        if result.signal_quality < 0.3:
            artifact_count = getattr(self.metrics, '_artifact_count', 0) + 1
            setattr(self.metrics, '_artifact_count', artifact_count)
            self.metrics.artifact_percentage = (artifact_count / n) * 100
    
    def _complete_calibration(self, calibration_data: List[BandPowerResult]):
        """Complete the calibration phase"""
        if not calibration_data:
            print("Warning: No calibration data collected")
            return
            
        # Calculate baseline values
        alpha_powers = [r.alpha_power for r in calibration_data]
        theta_powers = [r.theta_power for r in calibration_data]
        qualities = [r.signal_quality for r in calibration_data]
        
        self.metrics.baseline_alpha = np.mean(alpha_powers)
        self.metrics.baseline_theta = np.mean(theta_powers)
        self.metrics.calibration_quality = np.mean(qualities)
        
        # Set detector baseline
        calibration_array = np.array([0.0] * len(calibration_data))  # Placeholder
        self.detector.baseline_alpha = self.metrics.baseline_alpha
        self.detector.baseline_theta = self.metrics.baseline_theta
        self.detector.calibrated = True
        
        print(f"Calibration complete:")
        print(f"  Baseline Alpha: {self.metrics.baseline_alpha:.2f} μV²/Hz")
        print(f"  Baseline Theta: {self.metrics.baseline_theta:.2f} μV²/Hz") 
        print(f"  Signal Quality: {self.metrics.calibration_quality:.2f}")
        
        if self.on_calibration_complete:
            self.on_calibration_complete(self.metrics)
    
    def _update_feedback_metrics(self, feedback: Dict[str, float]):
        """Update feedback-related metrics"""
        self.metrics.feedback_updates += 1
        n = self.metrics.feedback_updates
        
        # Update average feedback score
        combined = feedback.get('combined_feedback', 0.0)
        self.metrics.average_feedback_score = (
            (self.metrics.average_feedback_score * (n-1) + combined) / n
        )
        
        # Update reward percentage
        if feedback.get('reward_active', False):
            reward_count = getattr(self.metrics, '_reward_count', 0) + 1
            setattr(self.metrics, '_reward_count', reward_count)
            self.metrics.reward_time_percentage = (reward_count / n) * 100
    
    def _update_session_metrics(self):
        """Update overall session metrics"""
        if self.on_metrics_update:
            self.on_metrics_update(self.metrics)
    
    def _update_state(self, new_state: SessionState):
        """Update session state and notify callbacks"""
        old_state = self.metrics.state
        self.metrics.state = new_state
        
        print(f"Session state: {old_state.value} → {new_state.value}")
        
        if self.on_state_change:
            self.on_state_change(old_state, new_state)
    
    def _save_session_data(self):
        """Save session data and metrics to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{self.config.participant_id}_{timestamp}"
        
        # Save detailed session data
        data_file = Path(self.config.data_directory) / f"{session_id}_data.csv"
        with open(data_file, 'w', newline='') as f:
            if self.session_data:
                writer = csv.DictWriter(f, fieldnames=self.session_data[0].keys())
                writer.writeheader()
                writer.writerows(self.session_data)
        
        # Save session metrics and config
        metrics_file = Path(self.config.data_directory) / f"{session_id}_metrics.json"
        session_summary = {
            'session_id': session_id,
            'config': asdict(self.config),
            'metrics': asdict(self.metrics),
            'duration_actual': (
                (self.metrics.end_time - self.metrics.start_time).total_seconds() / 60.0
                if self.metrics.end_time else 0.0
            )
        }
        
        # Convert datetime objects to ISO format for JSON serialization
        if self.metrics.end_time:
            session_summary['metrics']['start_time'] = self.metrics.start_time.isoformat()
            session_summary['metrics']['end_time'] = self.metrics.end_time.isoformat()
        else:
            session_summary['metrics']['start_time'] = self.metrics.start_time.isoformat()
            session_summary['metrics']['end_time'] = None
            
        session_summary['metrics']['state'] = self.metrics.state.value
        
        with open(metrics_file, 'w') as f:
            json.dump(session_summary, f, indent=2)
        
        print(f"Session data saved:")
        print(f"  Data: {data_file}")
        print(f"  Metrics: {metrics_file}")
    
    def get_current_metrics(self) -> SessionMetrics:
        """Get current session metrics"""
        return self.metrics
    
    def get_recent_data(self, seconds: int = 30) -> List[Dict]:
        """Get recent session data for display"""
        if not self.session_data:
            return []
            
        current_time = time.time()
        cutoff_time = current_time - seconds
        
        return [
            data for data in self.session_data[-100:] 
            if data['timestamp'] >= cutoff_time
        ]