# EEG Configuration
SAMPLING_RATE = 250 # Hz
BUFFER_SIZE = 1024 # samples
CHANNELS = ['Pz', 'P3', 'P4'] # Electrode locations

# Alpha-Theta Bands
ALPHA_BAND = (8, 12)  # Hz
THETA_BAND = (4, 8) # Hz
BETA_BAND = (13, 30) # Hz (for inhibition)

# Feedback Configuration
FEEDBACK_TYPE = 'audio'  # 'audio', 'visual', or 'both'
AUDIO_SAMPLE_RATE = 44100
AUDIO_BUFFER_SIZE = 512

# Hardware COnfiguration
ADC_TYPE = 'ADS1115' # or build in for Pi ADC
SPI_BUS = 0
SPI_DEVICE = 0
I2C_ADDRESS = 0x48
