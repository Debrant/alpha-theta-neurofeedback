#!/usr/bin/env python3
"""
Alpha-Theta Neurofeedback System Demo

Complete demonstration of the neurofeedback system with:
- Real-time EEG simulation
- Alpha-theta detection
- Audio feedback
- Session management
- Live metrics display
- Comprehensive logging and crash reporting
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import time
import threading
import logging
from datetime import datetime, timedelta

from session_manager import NeurofeedbackSession, SessionConfig, SessionState, SessionMetrics
from feedback.audio_feedback import FeedbackMode
from utils.logging_config import LoggingContext, get_logger, log_session_event


class NeurofeedbackDemo:
    """Interactive demonstration of the neurofeedback system"""
    
    def __init__(self):
        self.session = None
        self.running = False
        self.metrics_thread = None
        
    def run_demo(self):
        """Run the interactive demo"""
        print("=" * 60)
        print("Alpha-Theta Neurofeedback System Demo")
        print("Based on van der Kolk's 'The Body Keeps the Score'")
        print("Implementation of Peniston-Kulkosky Protocol")
        print("=" * 60)
        print()
        
        while True:
            try:
                choice = self._show_menu()
                
                if choice == '1':
                    self._run_quick_test()
                elif choice == '2':
                    self._run_full_session()
                elif choice == '3':
                    self._test_audio_modes()
                elif choice == '4':
                    self._show_system_status()
                elif choice == '5':
                    self._configure_session()
                elif choice == '6':
                    print("Thank you for using Alpha-Theta Neurofeedback!")
                    break
                else:
                    print("Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\nDemo interrupted by user")
                self._cleanup()
                break
            except Exception as e:
                print(f"Demo error: {e}")
                self._cleanup()
    
    def _show_menu(self):
        """Display the main menu and get user choice"""
        print("\nChoose a demo option:")
        print("1. Quick System Test (30 seconds)")
        print("2. Full Neurofeedback Session (5 minutes)")
        print("3. Test Audio Feedback Modes") 
        print("4. Show System Status")
        print("5. Configure Session Parameters")
        print("6. Exit")
        print()
        return input("Enter your choice (1-6): ").strip()
    
    def _run_quick_test(self):
        """Run a quick 30-second system test"""
        print("\n" + "=" * 50)
        print("QUICK SYSTEM TEST (30 seconds)")
        print("=" * 50)
        print()
        print("This will test all system components:")
        print("• EEG signal simulation")
        print("• Alpha-theta detection")
        print("• Real-time audio feedback")
        print("• Session management")
        print()
        
        input("Put on your headphones and press Enter to start...")
        
        # Configure short test session
        config = SessionConfig(
            duration_minutes=0.4,  # 24 seconds training
            calibration_minutes=0.1,  # 6 seconds calibration
            feedback_mode="binaural",
            volume_range=(0.3, 0.6)
        )
        
        self._run_session_with_display(config, "Quick Test")
    
    def _run_full_session(self):
        """Run a complete 5-minute neurofeedback session"""
        print("\n" + "=" * 50)
        print("FULL NEUROFEEDBACK SESSION (5 minutes)")
        print("=" * 50)
        print()
        print("This implements the complete Peniston-Kulkosky protocol:")
        print("• 2-minute calibration period")
        print("• 3-minute alpha-theta training")
        print("• Real-time binaural beat feedback") 
        print("• Session data logging")
        print()
        print("Find a comfortable, quiet position.")
        print("Close your eyes and focus on relaxation.")
        print()
        
        proceed = input("Ready to begin 5-minute session? (y/n): ").strip().lower()
        if proceed != 'y':
            return
            
        # Configure full session
        config = SessionConfig(
            duration_minutes=3.0,  # 3 minutes training
            calibration_minutes=2.0,  # 2 minutes calibration
            feedback_mode="binaural",
            volume_range=(0.2, 0.7),
            enable_reward_chimes=True
        )
        
        self._run_session_with_display(config, "Full Session")
    
    def _run_session_with_display(self, config: SessionConfig, session_name: str):
        """Run a session with live metrics display and comprehensive logging"""
        logger = get_logger('demo.session')
        
        try:
            print(f"\nStarting {session_name}...")
            logger.info(f"Starting {session_name} with config: {config.__dict__}")
            log_session_event('session_start', {'session_name': session_name, 'config': config.__dict__})
            
            # Create and configure session
            self.session = NeurofeedbackSession(config)
            self.session.on_state_change = self._on_state_change
            self.session.on_calibration_complete = self._on_calibration_complete
            
            # Start session
            if not self.session.start_session():
                error_msg = "Failed to start session!"
                logger.error(error_msg)
                print(error_msg)
                return
            
            # Start metrics display
            self.running = True
            self.metrics_thread = threading.Thread(target=self._display_metrics, daemon=True)
            self.metrics_thread.start()
            logger.info("Metrics display thread started")
            
            try:
                # Wait for session completion
                while self.session.is_running:
                    time.sleep(0.5)
                    
            except KeyboardInterrupt:
                logger.info("Session interrupted by user (Ctrl+C)")
                print("\nSession interrupted by user")
                log_session_event('session_interrupt', {'reason': 'user_keyboard'})
                
        except Exception as e:
            logger.error(f"Error during session execution: {e}", exc_info=True)
            log_session_event('session_error', {'error': str(e), 'type': type(e).__name__})
            print(f"\n⚠️  Session error: {e}")
            print("Check logs directory for detailed error information.")
            
        finally:
            try:
                self.running = False
                if self.session:
                    self.session.stop_session()
                    logger.info("Session stopped successfully")
                
                # Show final results
                self._show_session_results()
                log_session_event('session_end', {'session_name': session_name})
                
            except Exception as e:
                logger.error(f"Error during session cleanup: {e}", exc_info=True)
    
    def _display_metrics(self):
        """Display live session metrics"""
        last_display = 0
        
        while self.running and self.session and self.session.is_running:
            current_time = time.time()
            
            # Update display every 2 seconds
            if current_time - last_display >= 2.0:
                metrics = self.session.get_current_metrics()
                self._print_live_metrics(metrics)
                last_display = current_time
                
            time.sleep(0.5)
    
    def _print_live_metrics(self, metrics: SessionMetrics):
        """Print current session metrics"""
        # Clear screen (basic version)
        print("\n" * 3)
        
        # Session info
        elapsed = datetime.now() - metrics.start_time
        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
        
        print(f"Session State: {metrics.state.value.upper():12} | Elapsed: {elapsed_str}")
        print("-" * 60)
        
        if metrics.state == SessionState.CALIBRATING:
            print("CALIBRATION IN PROGRESS")
            print("Remain still and relaxed...")
            print(f"Samples collected: {metrics.total_samples:,}")
            
        elif metrics.state == SessionState.TRAINING:
            print("NEUROFEEDBACK TRAINING ACTIVE")
            print(f"Alpha Power:  {metrics.average_alpha_power:8.1f} μV²/Hz")
            print(f"Theta Power:  {metrics.average_theta_power:8.1f} μV²/Hz") 
            print(f"Signal Quality: {metrics.average_signal_quality:6.1%}")
            print(f"Feedback Score: {metrics.average_feedback_score:6.1%}")
            print(f"Reward Time:    {metrics.reward_time_percentage:6.1f}%")
            
            # Visual feedback bar
            feedback_bar = "█" * int(metrics.average_feedback_score * 20)
            print(f"Feedback: |{feedback_bar:<20}| {metrics.average_feedback_score:.1%}")
        
        print("-" * 60)
        print("Press Ctrl+C to stop session")
    
    def _test_audio_modes(self):
        """Test different audio feedback modes"""
        print("\n" + "=" * 50)
        print("AUDIO FEEDBACK MODE TEST")
        print("=" * 50)
        
        modes = [
            ("binaural", "Binaural Beats (recommended)"),
            ("nature", "Nature Sounds (ocean waves)"),
            ("tones", "Pure Tones (simple)"),
            ("hybrid", "Hybrid (binaural + nature)")
        ]
        
        print("\nTesting each audio mode for 10 seconds each:")
        print("Put on headphones for the best experience.")
        print()
        
        input("Press Enter to start audio tests...")
        
        from feedback.audio_feedback import AudioFeedbackSystem, FeedbackParameters, FeedbackMode
        
        for mode_key, description in modes:
            print(f"\nNow playing: {description}")
            
            # Create audio system for this mode
            params = FeedbackParameters(
                mode=FeedbackMode(mode_key),
                volume_range=(0.4, 0.4)  # Fixed volume for testing
            )
            audio_system = AudioFeedbackSystem(params)
            
            try:
                audio_system.start()
                
                # Simulate feedback updates
                for i in range(20):  # 10 seconds at 0.5s intervals
                    feedback = {
                        'combined_feedback': 0.5 + 0.3 * (i % 4) / 4.0,
                        'reward_active': (i % 6) == 0,
                        'alpha_relative': 1.2,
                        'theta_relative': 1.1
                    }
                    audio_system.update_feedback(feedback)
                    time.sleep(0.5)
                    
            finally:
                audio_system.stop()
                
        print("\nAudio mode testing complete!")
    
    def _show_system_status(self):
        """Display current system status"""
        print("\n" + "=" * 50)
        print("SYSTEM STATUS")
        print("=" * 50)
        
        # Test imports and basic functionality
        try:
            from signal_processing.alpha_theta_detector import AlphaThetaDetector
            from signal_processing.eeg_simulator import EEGSimulator
            from feedback.audio_feedback import AudioFeedbackSystem
            print("✓ All modules imported successfully")
        except Exception as e:
            print(f"✗ Module import error: {e}")
            return
        
        # Test signal processing
        try:
            detector = AlphaThetaDetector()
            simulator = EEGSimulator()
            test_signal = simulator.generate_signal(2.0)
            results = detector.add_samples(test_signal)
            print(f"✓ Signal processing: {len(results)} analysis windows from 2s signal")
        except Exception as e:
            print(f"✗ Signal processing error: {e}")
            
        # Test audio system
        try:
            import pyaudio
            audio = pyaudio.PyAudio()
            device_count = audio.get_device_count()
            audio.terminate()
            print(f"✓ Audio system: {device_count} audio devices available")
        except Exception as e:
            print(f"✗ Audio system error: {e}")
            
        # Show configuration
        print("\nCurrent Configuration:")
        config = SessionConfig()
        print(f"  Sample Rate: {config.sampling_rate} Hz")
        print(f"  Alpha Band: {config.alpha_band[0]}-{config.alpha_band[1]} Hz")
        print(f"  Theta Band: {config.theta_band[0]}-{config.theta_band[1]} Hz")
        print(f"  Window Size: {config.window_size}s")
        print(f"  Default Duration: {config.duration_minutes} minutes")
    
    def _configure_session(self):
        """Configure session parameters"""
        print("\n" + "=" * 50)
        print("SESSION CONFIGURATION")
        print("=" * 50)
        
        print("Current configuration will be used for subsequent sessions.")
        print("(This is a demo - configuration is not permanently saved)")
        print()
        
        # This would implement a configuration interface
        # For now, just show current settings
        config = SessionConfig()
        print("Current Settings:")
        print(f"  Training Duration: {config.duration_minutes} minutes")
        print(f"  Calibration Duration: {config.calibration_minutes} minutes")
        print(f"  Audio Mode: {config.feedback_mode}")
        print(f"  Volume Range: {config.volume_range[0]:.1f} - {config.volume_range[1]:.1f}")
        print(f"  Reward Chimes: {'Enabled' if config.enable_reward_chimes else 'Disabled'}")
        print()
        print("Configuration interface would be implemented here.")
    
    def _on_state_change(self, old_state, new_state):
        """Handle session state changes"""
        if new_state == SessionState.TRAINING:
            print("\n🧠 Starting alpha-theta training phase...")
        elif new_state == SessionState.COMPLETED:
            print("\n✅ Session completed successfully!")
    
    def _on_calibration_complete(self, metrics):
        """Handle calibration completion"""
        print(f"\n📊 Calibration complete!")
        print(f"Baseline established - beginning training...")
    
    def _show_session_results(self):
        """Display session results summary"""
        if not self.session:
            return
            
        metrics = self.session.get_current_metrics()
        
        print("\n" + "=" * 50)
        print("SESSION RESULTS SUMMARY")
        print("=" * 50)
        
        # Duration
        if metrics.end_time:
            duration = metrics.end_time - metrics.start_time
            duration_str = str(duration).split('.')[0]
            print(f"Session Duration: {duration_str}")
        
        # Signal quality
        print(f"Average Signal Quality: {metrics.average_signal_quality:.1%}")
        print(f"Total Samples Processed: {metrics.total_samples:,}")
        print(f"Analysis Windows: {metrics.analysis_windows:,}")
        
        if metrics.analysis_windows > 0:
            print(f"Average Alpha Power: {metrics.average_alpha_power:.2f} μV²/Hz")
            print(f"Average Theta Power: {metrics.average_theta_power:.2f} μV²/Hz")
            
        # Feedback performance
        if metrics.feedback_updates > 0:
            print(f"Average Feedback Score: {metrics.average_feedback_score:.1%}")
            print(f"Reward Time: {metrics.reward_time_percentage:.1f}%")
            
        # Issues
        if metrics.artifact_percentage > 10:
            print(f"⚠️  High artifact rate: {metrics.artifact_percentage:.1f}%")
        if metrics.signal_dropout_count > 0:
            print(f"⚠️  Signal dropouts: {metrics.signal_dropout_count}")
            
        print()
        if metrics.reward_time_percentage > 30:
            print("🎉 Excellent session! High reward achievement.")
        elif metrics.reward_time_percentage > 15:
            print("👍 Good session! You're learning to reach alpha-theta states.")
        else:
            print("📈 Keep practicing! Neurofeedback improves with regular sessions.")
            
        print()
        print("Session data has been saved to the data/sessions directory.")
    
    def _cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.session:
            self.session.stop_session()
            self.session = None


def main():
    """Main demo entry point with comprehensive logging"""
    session_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set up logging context
    with LoggingContext(session_id=session_id, log_directory="logs") as logger:
        demo_logger = get_logger('demo')
        demo_logger.info("Starting Alpha-Theta Neurofeedback Demo")
        log_session_event('demo_start', {'session_id': session_id})
        
        demo = NeurofeedbackDemo()
        
        try:
            demo.run_demo()
            demo_logger.info("Demo completed successfully")
            log_session_event('demo_complete', {'status': 'success'})
            
        except KeyboardInterrupt:
            demo_logger.info("Demo interrupted by user")
            log_session_event('demo_interrupt', {'reason': 'user_interrupt'})
            print("\nDemo interrupted by user")
            
        except Exception as e:
            demo_logger.error(f"Demo crashed with error: {e}", exc_info=True)
            log_session_event('demo_error', {'error': str(e), 'type': type(e).__name__})
            print(f"\n💥 Demo crashed! Check logs directory for full error details.")
            print(f"Session ID: {session_id}")
            raise  # Re-raise to trigger crash handler
            
        finally:
            try:
                demo._cleanup()
                demo_logger.info("Demo cleanup completed")
            except Exception as e:
                demo_logger.error(f"Error during cleanup: {e}", exc_info=True)


if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    main()