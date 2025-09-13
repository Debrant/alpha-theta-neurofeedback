"""
Logging configuration for Alpha-Theta Neurofeedback System

Provides comprehensive logging with file output, crash detection,
and session-specific log files for debugging and monitoring.
"""

import logging
import logging.handlers
import sys
import traceback
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


class NeurofeedbackLogger:
    """
    Comprehensive logging system for neurofeedback sessions.
    
    Features:
    - Session-specific log files
    - Real-time console output
    - Crash detection and reporting
    - Thread-safe logging
    - Automatic log rotation
    """
    
    def __init__(self, session_id: Optional[str] = None, log_directory: str = "logs"):
        """
        Initialize the logging system.
        
        Args:
            session_id: Unique session identifier
            log_directory: Directory for log files
        """
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Install crash handler
        self._setup_crash_handler()
        
        self.logger = logging.getLogger('neurofeedback')
        self.logger.info(f"Logging system initialized for session: {self.session_id}")
    
    def _setup_logging(self):
        """Configure logging handlers and formatters"""
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler for detailed logs
        log_file = self.log_directory / f"{self.session_id}_detailed.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB files, 5 backups
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Console handler for user feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
        
        # Error file handler
        error_file = self.log_directory / f"{self.session_id}_errors.log"
        error_handler = logging.FileHandler(error_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        
        # Session data handler (for metrics and events)
        session_file = self.log_directory / f"{self.session_id}_session.log"
        self.session_handler = logging.FileHandler(session_file)
        self.session_handler.setLevel(logging.INFO)
        session_formatter = logging.Formatter('%(asctime)s - %(message)s')
        self.session_handler.setFormatter(session_formatter)
        
        # Create session logger
        session_logger = logging.getLogger('neurofeedback.session')
        session_logger.addHandler(self.session_handler)
        session_logger.setLevel(logging.INFO)
        session_logger.propagate = False  # Don't duplicate to root logger
    
    def _setup_crash_handler(self):
        """Set up crash detection and reporting"""
        def crash_handler(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                # Handle Ctrl+C gracefully
                logging.getLogger('neurofeedback').info("Session interrupted by user (Ctrl+C)")
                return
            
            # Log the crash
            crash_logger = logging.getLogger('neurofeedback.crash')
            crash_logger.critical("SYSTEM CRASH DETECTED!", exc_info=(exc_type, exc_value, exc_traceback))
            
            # Save crash report
            self._save_crash_report(exc_type, exc_value, exc_traceback)
            
            # Call original handler
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        
        # Install crash handler
        sys.excepthook = crash_handler
    
    def _save_crash_report(self, exc_type, exc_value, exc_traceback):
        """Save detailed crash report"""
        crash_file = self.log_directory / f"{self.session_id}_CRASH_REPORT.txt"
        
        with open(crash_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("NEUROFEEDBACK SYSTEM CRASH REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Crash Time: {datetime.now().isoformat()}\n")
            f.write(f"Exception Type: {exc_type.__name__}\n")
            f.write(f"Exception Message: {exc_value}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("FULL STACK TRACE:\n")
            f.write("=" * 60 + "\n")
            
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("SYSTEM INFORMATION:\n")
            f.write("=" * 60 + "\n")
            f.write(f"Python Version: {sys.version}\n")
            f.write(f"Platform: {sys.platform}\n")
            f.write(f"Current Thread: {threading.current_thread().name}\n")
            f.write(f"Active Threads: {threading.active_count()}\n")
            
        print(f"\n💥 CRASH DETECTED! Full report saved to: {crash_file}")
        print("Please share this crash report if seeking support.")
    
    def log_session_event(self, event_type: str, data: dict):
        """Log session-specific events with structured data"""
        session_logger = logging.getLogger('neurofeedback.session')
        event_data = {
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            **data
        }
        session_logger.info(f"EVENT: {event_type} - {event_data}")
    
    def log_metrics(self, metrics_dict: dict):
        """Log session metrics"""
        self.log_session_event('metrics_update', metrics_dict)
    
    def log_eeg_data(self, sample_data: dict):
        """Log EEG data points (use sparingly to avoid huge logs)"""
        # Only log every 10th sample to keep logs manageable
        if hasattr(self, '_eeg_sample_count'):
            self._eeg_sample_count += 1
        else:
            self._eeg_sample_count = 1
            
        if self._eeg_sample_count % 10 == 0:
            self.log_session_event('eeg_sample', sample_data)
    
    def log_audio_event(self, event_data: dict):
        """Log audio system events"""
        self.log_session_event('audio_event', event_data)
    
    def log_state_change(self, old_state: str, new_state: str):
        """Log session state changes"""
        self.log_session_event('state_change', {
            'old_state': old_state,
            'new_state': new_state
        })
    
    def close(self):
        """Close logging system and flush all handlers"""
        logging.getLogger('neurofeedback').info(f"Closing logging system for session: {self.session_id}")
        
        # Close all handlers
        for handler in logging.getLogger().handlers[:]:
            handler.close()
            logging.getLogger().removeHandler(handler)
            
        if hasattr(self, 'session_handler'):
            self.session_handler.close()


# Global logger instance
_global_logger: Optional[NeurofeedbackLogger] = None


def setup_logging(session_id: Optional[str] = None, log_directory: str = "logs") -> NeurofeedbackLogger:
    """
    Set up global logging for the neurofeedback system.
    
    Args:
        session_id: Unique session identifier
        log_directory: Directory for log files
        
    Returns:
        NeurofeedbackLogger instance
    """
    global _global_logger
    
    if _global_logger:
        _global_logger.close()
    
    _global_logger = NeurofeedbackLogger(session_id, log_directory)
    return _global_logger


def get_logger(name: str = 'neurofeedback') -> logging.Logger:
    """Get a logger instance with the specified name"""
    return logging.getLogger(name)


def log_session_event(event_type: str, data: dict):
    """Log session event using global logger"""
    if _global_logger:
        _global_logger.log_session_event(event_type, data)


def log_metrics(metrics_dict: dict):
    """Log metrics using global logger"""
    if _global_logger:
        _global_logger.log_metrics(metrics_dict)


def close_logging():
    """Close global logging system"""
    global _global_logger
    if _global_logger:
        _global_logger.close()
        _global_logger = None


# Context manager for automatic logging setup/cleanup
class LoggingContext:
    """Context manager for neurofeedback logging"""
    
    def __init__(self, session_id: Optional[str] = None, log_directory: str = "logs"):
        self.session_id = session_id
        self.log_directory = log_directory
        self.logger = None
    
    def __enter__(self) -> NeurofeedbackLogger:
        self.logger = setup_logging(self.session_id, self.log_directory)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger:
            if exc_type is not None:
                # Log any exception that occurred
                logging.getLogger('neurofeedback').error(
                    f"Exception in logging context: {exc_type.__name__}: {exc_val}",
                    exc_info=(exc_type, exc_val, exc_tb)
                )
            close_logging()
        return False  # Don't suppress exceptions