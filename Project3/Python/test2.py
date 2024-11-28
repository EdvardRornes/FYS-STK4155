import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
import h5py

from gwosc.locate import get_event_urls
import requests
from datetime import datetime

# event_name = "GW170817-v1"
# urls = get_event_urls(event_name)
# with h5py.File("Data/GW170817-v3.hdf5", 'r') as file:
#     print(file.keys())
#     print(file["meta"].keys())
#     print(file["quality"].keys())
#     print(file["meta"]["Detector"][()])
#     print(file["meta"]["Observatory"][()])
#     print(file["meta"]["Type"][()])

# Load the pickled data
with h5py.File("Data/GW200311_115853-v1.hdf5", 'r') as file:

    strain = np.array(file["strain"]["Strain"][()])  # Downsample by factor of 100
    UTC_start_time = file["meta"]["UTCstart"][()].decode("utf-8")

    # Metadata for time
    T = float(file["meta"]["Duration"][()])
    N = len(strain)
    dt = T / N  # Time step
    t = np.arange(0, T, dt)

    # Convert to UTC timestamps
    t_zero = datetime.strptime(UTC_start_time, "%Y-%m-%dT%H:%M:%S").timestamp()
    t += t_zero

    # Event time
    t_event = datetime.strptime("2020-03-11T11:58:53", "%Y-%m-%dT%H:%M:%S").timestamp()

    # Extract indices for 1 second before and after the event
    t_start = t_event - 1
    t_end = t_event + 1
    start_index = np.argmin(np.abs(t - t_start))
    end_index = np.argmin(np.abs(t - t_end))

    # Slice strain data for the desired range
    strain = strain[start_index:end_index]
    print(strain)
    t = t[start_index:end_index]

# Spectrogram parameters
fs = 1 / dt  # Sampling frequency

# Set nperseg and ensure noverlap < nperseg
nperseg = min(1024, len(strain))  # Use 1024 or the length of the strain, whichever is smaller
noverlap = nperseg // 2  # Set overlap to 50% of segment length

# Calculate spectrogram
frequencies, times, Sxx = spectrogram(strain, fs, nperseg=nperseg, noverlap=noverlap, scaling='density')

# Normalize energy (scaled to 0-25)
Sxx_normalized = Sxx / np.max(Sxx) * 25

# Adjust spectrogram times to align with UTC timeline
times = t_start + times  # Shift spectrogram times to start from the correct UTC time
print(times)
# Plot the spectrogram
plt.figure(figsize=(12, 6))
plt.pcolormesh(times, frequencies, Sxx_normalized, shading='gouraud', cmap='viridis')
plt.colorbar(label='Normalized energy')
plt.yscale('log')  # Logarithmic frequency scale
plt.ylim(10, fs / 2)  # Set frequency limits
plt.xlim(t_start, t_end)  # Show 1 second before and after the event
plt.xlabel(f"Time [seconds] from {UTC_start_time} UTC")
plt.ylabel("Frequency [Hz]")
plt.title(f"Spectrogram of Gravitational Wave Strain Data (±1 second from event)")
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