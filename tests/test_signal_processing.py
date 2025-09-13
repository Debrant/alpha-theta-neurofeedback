#!/usr/bin/env python3
"""
Test suite for alpha-theta neurofeedback signal processing modules.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from signal_processing.alpha_theta_detector import AlphaThetaDetector
from signal_processing.eeg_simulator import EEGSimulator, EEGState


def test_detector_basic():
    """Test basic detector functionality"""
    print("Testing AlphaThetaDetector basic functionality...")
    
    detector = AlphaThetaDetector(sampling_rate=250, window_size=2.0)
    simulator = EEGSimulator(sampling_rate=250)
    
    # Generate test signal with known alpha content
    test_signal = simulator.generate_signal(
        duration=10.0, 
        state=EEGState.MEDITATION_LIGHT,
        noise_level=0.1
    )
    
    # Process signal
    results = detector.add_samples(test_signal)
    
    assert len(results) > 0, "No analysis results generated"
    
    # Check that alpha power is detected
    alpha_powers = [r.alpha_power for r in results]
    mean_alpha = np.mean(alpha_powers)
    
    print(f"  Mean alpha power: {mean_alpha:.3f} μV²/Hz")
    print(f"  Analysis windows: {len(results)}")
    
    assert mean_alpha > 0, "Alpha power should be positive"
    print("✓ Basic detector test passed")


def test_detector_calibration():
    """Test detector calibration and feedback generation"""
    print("\nTesting detector calibration and feedback...")
    
    detector = AlphaThetaDetector(sampling_rate=250)
    simulator = EEGSimulator(sampling_rate=250)
    
    # Generate baseline calibration data (relaxed state)
    baseline_signal = simulator.generate_signal(
        duration=30.0,
        state=EEGState.RELAXED_EYES_CLOSED,
        noise_level=0.15
    )
    
    # Calibrate detector
    detector.calibrate_baseline(baseline_signal, duration_seconds=30.0)
    assert detector.calibrated, "Detector should be calibrated"
    
    # Test feedback generation with enhanced alpha/theta
    enhanced_signal = simulator.generate_signal(
        duration=5.0,
        state=EEGState.MEDITATION_DEEP,
        alpha_modulation=np.ones(int(5.0 * 250)) * 1.5,
        theta_modulation=np.ones(int(5.0 * 250)) * 2.0,
        noise_level=0.1
    )
    
    results = detector.add_samples(enhanced_signal)
    
    # Test feedback signals
    feedback_results = []
    for result in results:
        feedback = detector.get_feedback_signals(result)
        feedback_results.append(feedback)
        
    avg_combined_feedback = np.mean([f['combined_feedback'] for f in feedback_results])
    reward_active_pct = np.mean([f['reward_active'] for f in feedback_results])
    
    print(f"  Baseline alpha: {detector.baseline_alpha:.3f}")
    print(f"  Baseline theta: {detector.baseline_theta:.3f}")
    print(f"  Average combined feedback: {avg_combined_feedback:.3f}")
    print(f"  Reward active: {reward_active_pct*100:.1f}% of time")
    
    assert avg_combined_feedback > 0.3, "Should show elevated feedback for enhanced state"
    print("✓ Calibration and feedback test passed")


def test_state_discrimination():
    """Test detector's ability to discriminate between EEG states"""
    print("\nTesting state discrimination...")
    
    detector = AlphaThetaDetector(sampling_rate=250)
    simulator = EEGSimulator(sampling_rate=250)
    
    states_to_test = [
        EEGState.AWAKE_ALERT,
        EEGState.RELAXED_EYES_CLOSED,
        EEGState.MEDITATION_LIGHT,
        EEGState.MEDITATION_DEEP
    ]
    
    state_results = {}
    
    for state in states_to_test:
        # Generate signal for this state
        signal_data = simulator.generate_signal(
            duration=20.0,
            state=state,
            noise_level=0.15
        )
        
        # Reset detector for each test
        detector.reset_buffer()
        results = detector.add_samples(signal_data)
        
        if results:
            avg_alpha = np.mean([r.alpha_power for r in results])
            avg_theta = np.mean([r.theta_power for r in results]) 
            avg_ratio = np.mean([r.alpha_theta_ratio for r in results])
            
            state_results[state.value] = {
                'alpha': avg_alpha,
                'theta': avg_theta,
                'ratio': avg_ratio
            }
    
    # Display results
    print("  State discrimination results:")
    for state, metrics in state_results.items():
        print(f"    {state:20}: α={metrics['alpha']:.2f}, θ={metrics['theta']:.2f}, α/θ={metrics['ratio']:.2f}")
    
    # Verify expected patterns
    meditation_deep = state_results['meditation_deep']
    awake_alert = state_results['awake_alert']
    
    assert meditation_deep['theta'] > awake_alert['theta'], "Deep meditation should have higher theta"
    print("✓ State discrimination test passed")


def test_realtime_processing():
    """Test real-time sample-by-sample processing"""
    print("\nTesting real-time processing...")
    
    detector = AlphaThetaDetector(sampling_rate=250, window_size=1.0)  # Faster updates
    simulator = EEGSimulator(sampling_rate=250)
    
    # Simulate real-time processing
    total_samples = 250 * 5  # 5 seconds
    results_count = 0
    
    for i in range(total_samples):
        # Generate one sample at a time
        sample = simulator.generate_sample(
            state=EEGState.MEDITATION_LIGHT,
            alpha_amplitude=1.2,
            theta_amplitude=1.1
        )
        
        # Process sample
        result = detector.add_sample(sample)
        if result is not None:
            results_count += 1
            
    print(f"  Processed {total_samples} samples")
    print(f"  Generated {results_count} analysis results")
    
    # Should get approximately one result per second
    expected_results = 5  # 5 seconds with 1-second windows
    assert abs(results_count - expected_results) <= 1, f"Expected ~{expected_results} results, got {results_count}"
    
    print("✓ Real-time processing test passed")


def test_signal_quality_assessment():
    """Test signal quality assessment"""
    print("\nTesting signal quality assessment...")
    
    detector = AlphaThetaDetector(sampling_rate=250)
    simulator = EEGSimulator(sampling_rate=250)
    
    # Test good signal quality
    clean_signal = simulator.generate_signal(
        duration=10.0,
        state=EEGState.MEDITATION_LIGHT,
        noise_level=0.05  # Very low noise
    )
    
    results_clean = detector.add_samples(clean_signal)
    quality_clean = np.mean([r.signal_quality for r in results_clean])
    
    # Test poor signal quality
    detector.reset_buffer()
    noisy_signal = simulator.generate_signal(
        duration=10.0,
        state=EEGState.POOR_SIGNAL,
        noise_level=0.8  # High noise
    )
    
    results_noisy = detector.add_samples(noisy_signal)
    quality_noisy = np.mean([r.signal_quality for r in results_noisy])
    
    print(f"  Clean signal quality: {quality_clean:.3f}")
    print(f"  Noisy signal quality: {quality_noisy:.3f}")
    
    assert quality_clean > quality_noisy, "Clean signal should have higher quality score"
    print("✓ Signal quality assessment test passed")


def run_full_session_simulation():
    """Run a complete neurofeedback session simulation"""
    print("\nRunning full session simulation...")
    
    detector = AlphaThetaDetector(sampling_rate=250)
    simulator = EEGSimulator(sampling_rate=250)
    
    # Generate a complete neurofeedback session
    session_signal, state_labels = simulator.generate_neurofeedback_session(duration=60.0)  # 1 minute for testing
    
    # Calibrate with first 20 seconds
    calibration_data = session_signal[:250*20]
    detector.calibrate_baseline(calibration_data, duration_seconds=20.0)
    
    # Process remaining session
    session_data = session_signal[250*20:]
    results = detector.add_samples(session_data)
    
    # Generate feedback throughout session
    feedback_timeline = []
    for result in results:
        feedback = detector.get_feedback_signals(result)
        feedback_timeline.append({
            'time': result.timestamp,
            'alpha_power': result.alpha_power,
            'theta_power': result.theta_power,
            'combined_feedback': feedback['combined_feedback'],
            'reward_active': feedback['reward_active']
        })
    
    print(f"  Session duration: 60.0 seconds")
    print(f"  Analysis windows: {len(results)}")
    print(f"  Average feedback: {np.mean([f['combined_feedback'] for f in feedback_timeline]):.3f}")
    print(f"  Reward active: {np.mean([f['reward_active'] for f in feedback_timeline])*100:.1f}% of time")
    
    print("✓ Full session simulation completed")
    return feedback_timeline


def main():
    """Run all tests"""
    print("Alpha-Theta Neurofeedback Signal Processing Tests")
    print("=" * 50)
    
    try:
        test_detector_basic()
        test_detector_calibration()
        test_state_discrimination()
        test_realtime_processing()
        test_signal_quality_assessment()
        session_data = run_full_session_simulation()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed successfully!")
        print("\nThe alpha-theta detector is ready for neurofeedback applications.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)