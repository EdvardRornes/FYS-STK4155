import h5py
import numpy as np
import shutil

# Original file path
original_file_path = "GWData/GW150914T090343_cut_3min.hdf5"  # Adjust this to your actual file path

# Path to GWData folder
data_folder = "GWData"

# Event details for GW150914
event = {"name": "GW150914", "gps_start": 1126259462.4, "duration": 0.2}

def label_signal(event):
    # Define new file path for the copy (do not overwrite the original file)
    new_file_path = f"{data_folder}/{event['name']}_labeled.hdf5"

    # Create a copy of the original file
    shutil.copy(original_file_path, new_file_path)
    
    with h5py.File(new_file_path, "r+") as f:
        # Copy attributes from the original file to the new one
        with h5py.File(original_file_path, "r") as original_f:
            for key, value in original_f.attrs.items():
                f.attrs[key] = value
        
        # Access the 'meta' group for attributes like GPSstart and Duration
        meta_group = f["meta"]
        
        gps_start_time = meta_group["GPSstart"][()]  # Start time of file in GPS seconds
        duration = meta_group["Duration"][()]  # Duration of the signal (seconds)
        
        if gps_start_time is None or duration is None:
            print(f"Missing meta information in {new_file_path}: {list(meta_group.keys())}")
            raise ValueError(f"Attributes 'GPSstart' or 'Duration' missing in {new_file_path}.")
        
        # Access strain data
        strain_group = f["strain"]
        strain = strain_group["Strain"][:]
        
        # Assuming the sampling rate is fixed or provided elsewhere, set it here (for example, 4096 Hz)
        sampling_rate = 4096  # You might want to adjust this based on actual data
        
        # Calculate indices for the signal region
        signal_start_idx = int((event["gps_start"] - gps_start_time) * sampling_rate)
        signal_end_idx = signal_start_idx + int(event["duration"] * sampling_rate)
        
        # Create the "signal" label array
        signal_label = np.zeros_like(strain, dtype=int)
        signal_label[signal_start_idx:signal_end_idx] = 1
        
        # Add or overwrite the "signal" dataset in the new file
        if "signal" in f:
            del f["signal"]
        f.create_dataset("signal", data=signal_label)
    
    print(f"Labeled signal for event: {event['name']} in {new_file_path}")

# Process the GW150914 event
label_signal(event)
