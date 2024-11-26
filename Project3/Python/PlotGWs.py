import numpy as np
import h5py
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# GPS Epoch starts on January 6, 1980
GPS_EPOCH = datetime(1980, 1, 6)

def gps_to_utc(gps_time):
    """
    Converts GPS time (seconds since 1980-01-06) to UTC datetime.
    
    Parameters:
        gps_time (int or float): The GPS timestamp to convert.
    
    Returns:
        datetime: Corresponding UTC datetime.
    """
    return GPS_EPOCH + timedelta(seconds=int(gps_time))  # Convert to int explicitly

def plot_strain_for_event(event_file, label="GW Event"):
    """
    Plots the strain data for a single event, normalized to 0 start time and 300 seconds duration.
    Displays the start time in UTC in the title.

    Parameters:
        event_file (str): Path to the file containing the event strain data.
        label (str): Label for the event.
    """
    with h5py.File(event_file, "r") as f:
        strain = f["strain"][:]
        sample_rate = f.attrs["sample_rate"]
        start_time = f.attrs["start_time"]
    
    # Normalize the time to start at 0
    time = np.arange(len(strain)) / sample_rate
    
    # Convert the GPS start time to UTC
    start_time_utc = gps_to_utc(start_time).strftime('%Y-%m-%d %H:%M:%S')
    
    # Plot the strain as a function of time, from 0 to 300 seconds
    plt.figure(figsize=(10, 6))
    plt.plot(time, strain, lw=0.15)
    plt.xlim(0, 300)  # Ensure the time axis is from 0 to 300 seconds
    plt.xlabel("Time (seconds)")
    plt.ylabel("Strain")
    plt.title(f"{label} - Start Time: {start_time_utc}")
    plt.grid(True)

def plot_strain_for_noise(noise_file, label="Noise Interval"):
    """
    Plots the strain data for a single noise interval, normalized to 0 start time and 300 seconds duration.
    Displays the start time in UTC in the title.

    Parameters:
        noise_file (str): Path to the file containing the noise strain data.
        label (str): Label for the noise interval.
    """
    with h5py.File(noise_file, "r") as f:
        strain = f["strain"][:]
        sample_rate = f.attrs["sample_rate"]
        start_time = f.attrs["start_time"]
    
    # Normalize the time to start at 0
    time = np.arange(len(strain)) / sample_rate
    
    # Convert the GPS start time to UTC
    start_time_utc = gps_to_utc(start_time).strftime('%Y-%m-%d %H:%M:%S')
    
    # Plot the strain as a function of time, from 0 to 300 seconds
    plt.figure(figsize=(10, 6))
    plt.plot(time, strain, lw=0.15)
    plt.xlim(0, 300)  # Ensure the time axis is from 0 to 300 seconds
    plt.xlabel("Time (seconds)")
    plt.ylabel("Strain")
    plt.title(f"{label} - Start Time: {start_time_utc}")
    plt.grid(True)

# List of event files (you need to specify the correct paths)
event_files = [
    "Data/GW150914.hdf5",
    "Data/GW151226.hdf5",
    "Data/GW170104.hdf5",
    "Data/GW170814.hdf5",
    "Data/GW170817.hdf5"
]

# List of noise files (you need to specify the correct paths)
noise_files = [
    "Data/noise_1.hdf5",
    "Data/noise_2.hdf5",
    "Data/noise_3.hdf5"
]

# Plot the strain for each event individually
for i, event_file in enumerate(event_files):
    plot_strain_for_event(event_file, label=f"GW Event {i + 1}")

# Plot the strain for each noise interval individually
for i, noise_file in enumerate(noise_files):
    plot_strain_for_noise(noise_file, label=f"Noise Interval {i + 1}")


plt.show()