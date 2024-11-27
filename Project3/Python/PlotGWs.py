import h5py
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

# Path to GWData folder
data_folder = "GWData"

# Event details
events = [
    {"name": "GW150914", "gps_start": 1126259462.4, "duration": 0.2},
    {"name": "GW170817", "gps_start": 1187008882.4, "duration": 100},
    {"name": "GW190814", "gps_start": 1249852257.0, "duration": 0.1},
]

def plot_event(event):
    file_path = f"{data_folder}/{event['name']}.hdf5"
    
    with h5py.File(file_path, "r") as f:
        # Retrieve strain data
        strain = f["strain"][:]
        
        # Retrieve metadata
        gps_start_time = f.attrs.get("start_time", None)  # This is the file start time
        sampling_rate = f.attrs.get("sample_rate", None)
        
        if gps_start_time is None or sampling_rate is None:
            raise ValueError(f"Missing necessary attributes in {file_path}.")
        
        # Calculate the time array, relative to the file's start time (start at 0)
        time = np.arange(0, len(strain)) / sampling_rate
        
        # Convert the event's start time (GPS) to UTC
        event_start_utc = event_times_utc([event])[0]
        
        # Retrieve signal labels if they exist
        signal = f["signal"][:] if "signal" in f else None
    
    # Plot strain data
    plt.figure(figsize=(10, 6))
    plt.plot(time, strain, label="Strain Data", color="blue", linewidth=0.5)
    
    # Overlay signal presence as a shaded region
    if signal is not None:
        signal_regions = np.where(signal == 1)[0]
        if len(signal_regions) > 0:
            start_idx, end_idx = signal_regions[0], signal_regions[-1]
            plt.axvspan(
                time[start_idx], time[end_idx], 
                color="red", alpha=0.3, label="Signal Present"
            )
    
    # Adjust x-axis from 0 to 300 seconds
    plt.xlim(0, 300)
    
    # Plot the event's start time in UTC at x=0
    plt.text(0, np.max(strain), f"Event Start (UTC): {event_start_utc}", 
             horizontalalignment='left', verticalalignment='top', 
             fontsize=10, color="black", backgroundcolor="white")
    
    # Plot formatting
    plt.title(f"Strain Data with Signal Highlighted - {event['name']}")
    plt.xlabel("Time (seconds since file start)")
    plt.ylabel("Strain")
    plt.legend()
    plt.grid()
    plt.tight_layout()

# Plot each event
for event in events:
    plot_event(event)

plt.show()