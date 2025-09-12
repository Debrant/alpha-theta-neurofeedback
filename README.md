# Alpha-Theta Neurofeedback System

A Raspberry Pi-based system for alpha-theta neurofeedback training, implementing the protocols described in Bessel van der Kolk's "The Body Keeps the Score."

## Project Goals

- Build custom EEG hardware using AD620 instrumentation amplifiers
- Implement real-time alpha (8-12 Hz) and theta (4-8 Hz) detection
- Provide audio/visual feedback for neurofeedback training
- Support for various electrode types (traditional gel, dry, experimental)

## Hardware Components

- Raspberry Pi 4
- AD620 instrumentation amplifier
- ADS1115 16-bit ADC
- Custom electrode interfaces
- Audio output for feedback

## Current Status

🚧 **In Development**
- [ ] Hardware design and testing
- [ ] Signal processing algorithms
- [ ] Real-time feedback systems
- [ ] User interface development

## Usage
Coming soon…

## Contributing
This is a research/personal project. Feel free to fork and experiment!

## References
	•	van der Kolk, B. A. (2014). The Body Keeps the Score: Brain, Mind, and Body in the Healing of Trauma
	•	Peniston, E. G., & Kulkosky, P. J. (1991). Alpha-theta brainwave neurofeedback for Vietnam veterans with combat-related post-traumatic stress disorder

## Installation

```bash
# Clone repository
git clone https://github.com/Debant/alpha-theta-neurofeedback.git
cd alpha-theta-neurofeedback

# Create virtual environment
python3 -m venv neurofeedback_env
source neurofeedback_env/bin/activate

# Install dependencies
pip install -r requirements.txt

