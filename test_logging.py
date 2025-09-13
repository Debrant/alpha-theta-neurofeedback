#!/usr/bin/env python3
"""
Test logging system and basic functionality before running full demo
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import time
from utils.logging_config import LoggingContext, get_logger, log_session_event

def test_basic_imports():
    """Test that all modules can be imported without errors"""
    logger = get_logger('test.imports')
    logger.info("Testing module imports...")
    
    try:
        from signal_processing.alpha_theta_detector import AlphaThetaDetector
        logger.info("✓ AlphaThetaDetector imported successfully")
        
        from signal_processing.eeg_simulator import EEGSimulator, EEGState
        logger.info("✓ EEGSimulator imported successfully")
        
        from feedback.audio_feedback import AudioFeedbackSystem, FeedbackParameters, FeedbackMode
        logger.info("✓ AudioFeedbackSystem imported successfully")
        
        from session_manager import NeurofeedbackSession, SessionConfig
        logger.info("✓ SessionManager imported successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Import failed: {e}", exc_info=True)
        return False

def test_signal_processing():
    """Test basic signal processing functionality"""
    logger = get_logger('test.signal_processing')
    logger.info("Testing signal processing...")
    
    try:
        from signal_processing.alpha_theta_detector import AlphaThetaDetector
        from signal_processing.eeg_simulator import EEGSimulator, EEGState
        
        # Create components
        detector = AlphaThetaDetector(sampling_rate=250, window_size=1.0)
        simulator = EEGSimulator(sampling_rate=250)
        
        # Generate test signal
        logger.info("Generating 5-second test signal...")
        test_signal = simulator.generate_signal(duration=5.0, state=EEGState.MEDITATION_LIGHT)
        
        # Process signal
        logger.info("Processing signal through detector...")
        results = detector.add_samples(test_signal)
        
        logger.info(f"✓ Generated {len(results)} analysis windows from 5s signal")
        
        if results:
            avg_alpha = sum(r.alpha_power for r in results) / len(results)
            avg_theta = sum(r.theta_power for r in results) / len(results)
            logger.info(f"✓ Average alpha power: {avg_alpha:.2f}")
            logger.info(f"✓ Average theta power: {avg_theta:.2f}")
            
        log_session_event('signal_processing_test', {
            'windows_generated': len(results),
            'test_duration': 5.0,
            'success': True
        })
        
        return True
        
    except Exception as e:
        logger.error(f"Signal processing test failed: {e}", exc_info=True)
        log_session_event('signal_processing_test', {'success': False, 'error': str(e)})
        return False

def test_audio_system():
    """Test audio system initialization (without actually playing sound)"""
    logger = get_logger('test.audio')
    logger.info("Testing audio system...")
    
    try:
        from feedback.audio_feedback import AudioFeedbackSystem, FeedbackParameters, FeedbackMode
        
        # Create audio system
        params = FeedbackParameters(
            mode=FeedbackMode.SILENCE,  # Use silence mode for testing
            volume_range=(0.1, 0.5)
        )
        audio_system = AudioFeedbackSystem(params)
        
        # Test basic functionality without starting audio stream
        status = audio_system.get_status()
        logger.info(f"✓ Audio system created: {status}")
        
        log_session_event('audio_test', {'success': True, 'status': status})
        
        return True
        
    except Exception as e:
        logger.error(f"Audio system test failed: {e}", exc_info=True)
        log_session_event('audio_test', {'success': False, 'error': str(e)})
        return False

def test_session_manager():
    """Test session manager creation and basic setup"""
    logger = get_logger('test.session')
    logger.info("Testing session manager...")
    
    try:
        from session_manager import NeurofeedbackSession, SessionConfig
        
        # Create session config
        config = SessionConfig(
            duration_minutes=0.1,  # Very short for testing
            calibration_minutes=0.05,
            sampling_rate=250
        )
        
        # Create session (but don't start it)
        session = NeurofeedbackSession(config)
        logger.info("✓ Session manager created successfully")
        
        # Test configuration
        logger.info(f"✓ Session config: {config.duration_minutes}min duration")
        
        log_session_event('session_manager_test', {
            'success': True,
            'config': config.__dict__
        })
        
        return True
        
    except Exception as e:
        logger.error(f"Session manager test failed: {e}", exc_info=True)
        log_session_event('session_manager_test', {'success': False, 'error': str(e)})
        return False

def main():
    """Run all tests with logging"""
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    session_id = f"test_{int(time.time())}"
    
    with LoggingContext(session_id=session_id, log_directory="logs") as logger_system:
        logger = get_logger('test.main')
        logger.info("Starting neurofeedback system tests")
        log_session_event('test_start', {'session_id': session_id})
        
        tests = [
            ("Module Imports", test_basic_imports),
            ("Signal Processing", test_signal_processing),
            ("Audio System", test_audio_system),
            ("Session Manager", test_session_manager)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n{'='*50}")
            print(f"Running: {test_name}")
            print('='*50)
            
            try:
                start_time = time.time()
                result = test_func()
                duration = time.time() - start_time
                
                results[test_name] = result
                status = "PASSED" if result else "FAILED"
                
                print(f"{test_name}: {status} ({duration:.2f}s)")
                logger.info(f"{test_name}: {status} in {duration:.2f}s")
                
            except Exception as e:
                results[test_name] = False
                print(f"{test_name}: CRASHED - {e}")
                logger.error(f"{test_name}: CRASHED", exc_info=True)
        
        # Summary
        print(f"\n{'='*50}")
        print("TEST SUMMARY")
        print('='*50)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{status:8} {test_name}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        log_session_event('test_complete', {
            'total_tests': total,
            'passed_tests': passed,
            'results': results
        })
        
        if passed == total:
            print("\n🎉 All tests passed! System is ready for use.")
            logger.info("All tests passed successfully")
            return True
        else:
            print("\n⚠️  Some tests failed. Check logs for details.")
            logger.warning(f"Only {passed}/{total} tests passed")
            return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)