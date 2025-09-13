#!/usr/bin/env python3
"""
Test script to verify neurofeedback setup
"""


import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pyaudio
import time

def test_signal_processing():
	""" Test signal processing capabilities"""
	print("Testing signal processing...")

	# Generate EEG-Like signal
	fs = 250  # sample rate
	t = np.linspace(0, 4, fs * 4)

 	# Simulate alph (10 Hz) and  theta (6 Hz) waves
	alpha_wave = np.sin(2 * np.pi * 10 * t)
	theta_wave = 0.5 * np.sin(2 * np.pi * 6 * t)
	noise = 0.1 * np.random.randn(len(t))

	signal_data = alpha_wave + theta_wave + noise

	# Perform FFT
	freqs = np.fft.fftfreq(len(signal_data), 1/fs)
	fft_data = np.fft.fft(signal_data)

	# Find peak frequencies
	positive_freqs = freqs[:len(signal_data)//2]
	magnitude = np.abs(fft_data[:len(fft_data)//2])

	# Skip DC component while finding peack frequency
	magnitude_no_dc = magnitude[1:]
	peak_idx_no_dc = np.argmax(magnitude_no_dc)
	peak_idx = peak_idx_no_dc + 1                      
                       
	peak_freq = positive_freqs[peak_idx]

	print(f"Peack Frequency detected: {peak_freq:.1f} Hz")
	print("Signal processing test: PASSED")

def test_audio():
	"""Test Audio output capabilities"""
	print("Testing audio output...")

	try:
		#generate test tone
		fs = 44100
		duration = 1.0 
		frequency = 440  # A4 note

		t = np.linspace(0, duration, int(fs * duration))
		tone = 0.3 * np.sin(2 * np.pi * frequency * t)

		# Play tone
		p = pyaudio.PyAudio()
		stream =p.open(format=pyaudio.paFloat32,
			channels=1,
			rate=fs,
			output=True)

		print("Playing test tone (440 HZ for 1 second)...")
		stream.write(tone.astype(np.float32).tobytes())
		stream.stop_stream()
		stream.close()
		p.terminate()
	
		print("Audio test: PASSED")

	except Exception as e:
		print(f"Audio test: FAILED - {e}")

def test_hardware_interface():
	"""Test hardware interface availability"""
	print("Testing hardware interfaces...")
	
	try:
		import spidev
		print("SPI interface: Available")
	except ImportError:	
		print("SPI interface: NOT AVAILABLE")


	try:
		import smbus
		print("I2C interface: Available")
	except ImportError:
		print("I2C interface: NOT AVAILABLE")

	try:	
		import RPi.GPIO as GPIO
		print("GPIO interface: Available")
	except ImportError:
		print("GPIO interface: NOT AVAILABLE")

if __name__ == "__main__":
	print("===Neurofeedback Setup Test ===")
	print()


	test_signal_processing()
	print()
	
	test_audio()
	print()

	test_hardware_interface()
	print()

	print("Setup test complete")
