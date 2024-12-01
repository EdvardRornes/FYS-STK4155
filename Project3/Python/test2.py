import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from datetime import datetime

# Event name and data
event_name = "GW200129_065458-v1"
# event_name = "GW190814-v2"
# Load the pre-saved pickle data
data = pd.read_pickle(f"Data/PickleFiles/{event_name}.pkl")
# print(data["duration"])
# exit()
# Extract necessary data
strain = data["data"]  # Gravitational wave strain data
dt = data["dt"]  # Sampling interval (seconds)
UTC_start_time = data["UTC_event_end_time"]  # Event's UTC end time
indices = data["data_event_indices"]  # Indices for event
t = data["t"]  # Time array in UTC seconds

# Extract event-related parameters
start_index = indices[0]  # Start of event
end_index = indices[-1]  # End of event

# Crop data to focus on ±100 seconds around the event
time_window = 100  # seconds before and after the event
crop_index_before = np.argmin(np.abs(t - (t[start_index] - time_window)))
crop_index_after = np.argmin(np.abs(t - (t[start_index] + time_window)))

event_time = t[crop_index_before:crop_index_after]
event_strain = strain[crop_index_before:crop_index_after]

# Generate relative time for visualization
relative_time = event_time - t[start_index]

# Spectrogram settings
NFFT = 1024  # Length of FFT for better resolution
Fs = 1 / dt  # Sampling frequency
noverlap = int(0.9 * NFFT)  # Overlap between segments for smoother spectrogram
# Create spectrogram

def whiten(strain, psd, dt):
    """Whiten strain data using the power spectral density (PSD)."""
    hf = np.fft.rfft(strain)
    whitened_hf = hf / (np.sqrt(psd) + 1e-12)
    whitened_strain = np.fft.irfft(whitened_hf, n=len(strain))
    return whitened_strain

# Compute PSD for the strain data (e.g., using scipy.signal.welch)
psd = np.abs(np.fft.rfft(strain))**2
whitened_strain = whiten(strain, psd, dt)

plt.figure(figsize=(12, 6))
Pxx, freqs, bins, im = plt.specgram(
    whitened_strain,
    NFFT=NFFT,
    Fs=Fs,
    noverlap=noverlap,
    cmap="viridis",
    mode="psd",
)

new_start_index = np.argmin(np.abs(bins - (t[start_index]-t[0])))
new_end_index = np.argmin(np.abs(bins - (t[end_index]-t[0])))

print(t[start_index]-t[0], new_start_index)
print(t[end_index]-t[0], new_end_index)
print(bins[new_end_index])
plt.vlines(bins[new_start_index], np.min(Pxx), np.max(Pxx), colors=["red"], lw=10)
plt.vlines(bins[new_end_index], np.min(Pxx), np.max(Pxx), colors=["blue"], lw=10)
plt.xlim(t[start_index]-t[0], t[start_index]-t[0])
# plt.xlim(51.5, 53)
# Adjust frequency range for better visualization
plt.ylim(10, 2000)

# Convert bins to time relative to the event
# bins are relative to the cropped signal's start
relative_bins = bins + (t[crop_index_before] - t[start_index])

# Adjust x-axis to display time relative to the event
plt.xlabel("Time [seconds] relative to event")
plt.ylabel("Frequency [Hz]")

# Optionally zoom in to specific times around the event
# plt.xlim(-0.1, 0.7)  # Adjust to your desired zoom range

# Add colorbar and scale
plt.colorbar(im, label="Normalized energy")
plt.yscale("log")  # Logarithmic scale for frequency
plt.title(f"Spectrogram centered around event (Event Time: {datetime.utcfromtimestamp(t[start_index])} UTC)")
plt.show()


from pycbc.types import TimeSeries
from pycbc.filter.qtransform import qtransform
import matplotlib.pyplot as plt
import numpy as np

# Convert strain data into a PyCBC TimeSeries object
strain_series = TimeSeries(event_strain, delta_t=dt)

# Perform the Q-transform
hq, freqs, times = qtransform(strain_series, logfsteps=100, qrange=(8, 64), frange=(10, 2000), mismatch=0.2)

# Normalize the Q-transform output for visualization
hq_normalized = 20 * np.log10(abs(hq) / np.max(abs(hq)))

# Plot the Q-transform
plt.figure(figsize=(12, 6))
plt.pcolormesh(times - times[0], freqs, hq_normalized, shading="auto", cmap="viridis")
plt.colorbar(label="Normalized Energy (dB)")
plt.yscale("log")
plt.ylim(10, 2000)
plt.xlabel("Time [seconds] relative to event")
plt.ylabel("Frequency [Hz]")
plt.title("Q-transform of Strain Data")
plt.show()

# Fourier Transform of the strain data
frequencies = np.fft.fftfreq(len(event_strain), d=dt)  # Positive frequencies
strain_ft = np.fft.fft(event_strain)  # Fourier Transform

# Compute power spectral density
psd = np.abs(strain_ft)**2

# Plot the Fourier Transform
plt.figure(figsize=(12, 6))
plt.plot(frequencies, np.abs(strain_ft), label="Amplitude Spectrum")
# plt.xscale("log")
# plt.yscale("log")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title("Fourier Transform of the Strain Data")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.xlim(10, 2000)  # Focus on relevant gravitational wave band
plt.show()


exit()
obj = pd.read_pickle("Data/PickleFiles/GW191216_213338-v1.pkl")
print(obj["detector"])
print(obj["UTC_event_end_time"])

indices = obj["data_event_indices"]
print(obj["t"][indices[0]])
# strain = obj["data"][indices[0]-10000:indices[1]+10000]
# t = obj["t"]#[indices[0]-10000:indices[1]+10000]

fs = 4096  # Sampling frequency in Hz

# Calculate the spectrogram
frequencies, times, Sxx = spectrogram(strain, fs, nperseg=1024, noverlap=512, scaling='density')

# Convert to decibels for better energy visualization
Sxx_normalized = Sxx / np.max(Sxx)

# Create the spectrogram plot
plt.figure(figsize=(12, 6))
plt.pcolormesh(times, frequencies, Sxx_normalized, shading='gouraud', cmap='viridis')

print(times[np.argmin(np.abs(times/fs - t_end))])
plt.colorbar(label='Normalized energy [dB]')
plt.yscale('log')  # Logarithmic frequency scale

plt.ylim(10, fs / 2)  # Frequency range
plt.xlim(times[0], times[-1])  # Time range
plt.xlabel('Time [seconds]')
plt.ylabel('Frequency [Hz]')
plt.title('Spectrogram of Strain Data')
plt.show()
exit()
print(obj)
print(obj["noise"])

indices = obj["data_event_indices"]

# plt.plot(obj["t"], obj["data"], lw=0.2)
# plt.plot(obj["t"][indices[0]:indices[1]], obj["data"][indices[0]:indices[1]], color="red", lw=0.2)
# plt.show()

dt = obj["dt"]
fs = 1/dt  # Sampling frequency
t_true = obj["t"] * dt  # Scale to seconds if 't' is in samples

# Plot the time-domain signal
plt.plot(t_true[indices[0]-1000:indices[1]+1000], obj["data"][indices[0]-1000:indices[1]+1000], color="blue", lw=0.2)
plt.plot(t_true[indices[0]:indices[1]], obj["data"][indices[0]:indices[1]], color="red", lw=0.2)
plt.show()

# Perform FFT
data_mod = obj["data"]
data_mod_centered = data_mod# - np.mean(data_mod)  # Remove mean
data_mod_F = np.fft.fft(data_mod_centered) / len(data_mod_centered)
freq = np.fft.fftfreq(len(data_mod_centered), d=1/fs)
plt.plot(freq, np.abs(data_mod_F))
plt.xlim(-10, fs/2)
plt.show()


data_mod_F = np.fft.fft(data_mod) / len(data_mod)  # Normalize FFT
freq = np.fft.fftfreq(len(data_mod), d=1/fs)      # Scale frequency axis

# Plot frequency-domain signal
plt.plot(freq, np.abs(data_mod_F))
plt.xlim(-10, fs/2)  # Positive frequencies only
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()


plt.plot(np.arange(0,len(obj["noise"]))/4096, obj["noise"])
plt.show()

dt = obj["dt"]

print(len(obj["data"])/4096, len(obj["noise"]))