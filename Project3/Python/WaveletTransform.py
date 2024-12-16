import pywt
import numpy as np
import matplotlib.pyplot as plt
from utils import GWSignalGenerator

# Example with Continuous Wavelet Transform
def wavelet_transform(signal, wavelet='cmor', scales=np.arange(1, 128)):
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
    return coefficients, frequencies  # Coefficients and frequencies

time_for_1_sample = 50
time_steps = 5000
num_samples = 5
SNR = 2

x = np.linspace(0, time_for_1_sample, time_steps)

# Background noise
y = [
    0.5*np.sin(90*x) - 0.5*np.cos(60*x)*np.sin(-5.*x) + 0.3*np.cos(30*x) + 0.05*np.sin(time_steps/40*x),
    0.5*np.sin(50*x) - 0.5*np.cos(80*x)*np.sin(-10*x) + 0.3*np.cos(40*x) + 0.05*np.sin(time_steps/20*x),
    0.5*np.sin(40*x) - 0.5*np.cos(25*x)*np.sin(-10*x) + 0.3*np.cos(60*x) + 0.10*np.sin(time_steps/18*x),
    0.7*np.sin(70*x) - 0.4*np.cos(10*x)*np.sin(-15*x) + 0.4*np.cos(80*x) + 0.05*np.sin(time_steps/12*x),
    0.1*np.sin(80*x) - 0.4*np.cos(50*x)*np.sin(-3.*x) + 0.3*np.cos(20*x) + 0.02*np.sin(time_steps/30*x)
]

for signal in y:
    signal += np.random.normal(0, 0.2, len(x))
    signal /= SNR  # Quick rescaling, the division factor is ~ SNR

event_lengths = [(time_steps//10, time_steps//8), (time_steps//7, time_steps//6), 
                 (time_steps//14, time_steps//12), (time_steps//5, time_steps//3),
                 (time_steps//5, time_steps//4)]

events = []
labels = []

# Add a single synthetic GW event to each sample
for i in range(num_samples):
    generator = GWSignalGenerator(signal_length=time_steps)
    events_i = generator.generate_random_events(1, event_lengths[i])
    generator.apply_events(y[i], events_i)
    events.append(events_i)
    labels.append(generator.labels)

# Simple example of a GW signal
x = np.linspace(0, 2000, 5000)  # Time axis
t_start = 1000
GW_len = 300
y = np.zeros_like(x)
y += (x - t_start) / 1000 * np.sin((t_start - x)**2/1000) * (x > t_start) * (x < t_start + GW_len)

# Wavelet transform application
scales = np.arange(1, 256)  # Define scales for wavelet transform
wavelet = 'cmor'  # Complex Morlet wavelet

# Perform the wavelet transform
padded_signal = np.pad(y, pad_width=5000, mode='reflect')  # Reflective padding
coeffs, freqs = wavelet_transform(padded_signal, wavelet=wavelet, scales=scales)
coeffs = coeffs[:, 5000:-5000]  # Remove padding after transform


# Plot the scalogram
plt.figure(figsize=(15, 10))
plt.contourf(x, scales, np.abs(coeffs), extend='both', cmap='viridis')
plt.title("Scalogram for the Signal")
plt.xlabel("Time")
plt.ylabel("Scales (Frequency)")
plt.colorbar(label="Magnitude")
plt.tight_layout()

# Plot the original signal
plt.figure(figsize=(12, 6))
plt.plot(x, y, label="Signal")
plt.title("Synthetic GW Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()