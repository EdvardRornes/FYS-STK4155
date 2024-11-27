import h5py
import numpy as np
from astropy.time import Time

# Define the new desired time window around the event
original_start_time_str = "2015-09-14 09:03:43"  # Original start time of the file
event_time_str = "2015-09-14 09:50:45"  # Event time
new_start_time_str = "2015-09-14 09:49:00"  # New start time for the 3-minute window

# Convert the time strings to GPS times (in seconds)
original_start_time = Time(original_start_time_str, format='iso').gps
event_time = Time(event_time_str, format='iso').gps
new_start_time = Time(new_start_time_str, format='iso').gps

# Calculate the time difference between the original start time and the new start time
time_diff = new_start_time - original_start_time

# File path for the original data file
original_file_path = "GWData/GW150914T090343.hdf5"  # Adjust this to your actual file path
new_file_path = "GWData/GW150914T090343_cut_3min.hdf5"  # New file path for the 3-minute dataset

# Open the original file and create a new file for the 3-minute dataset
with h5py.File(original_file_path, "r") as original_file:
    # Access the strain data
    strain_data = original_file["strain/Strain"][:]  # Correct dataset path
    
    # Sample rate should be part of the metadata or quality attributes, if available
    sample_rate = 4096  # Assuming it's 4096 Hz if not found in the file
    
    # Determine the number of samples to shift by
    shift_samples = int(time_diff * sample_rate)
    
    # Create a new strain dataset for the 3-minute window
    duration = 3 * 60  # 3 minutes in seconds
    new_end_sample = shift_samples + int(duration * sample_rate)
    
    # Select the appropriate slice of strain data
    new_strain = strain_data[shift_samples:new_end_sample]
    
    # Create a new file to store the 3-minute data
    with h5py.File(new_file_path, "w") as new_file:
        # Copy the metadata from the original file to the new file
        original_file.copy('meta', new_file)  # Copy metadata
        original_file.copy('quality', new_file)  # Copy quality information (if needed)
        
        # Create the new strain dataset in the new file
        new_file.create_dataset("strain/Strain", data=new_strain)
        
        # Optionally, copy other groups or datasets as needed (e.g., signal, quality, etc.)
        # original_file.copy('quality/simple', new_file)  # Example if you need quality data
        
        # Update the 'meta' group with the new start time
        if 'meta' in new_file:
            new_file['meta/UTCstart'] = new_start_time_str
        
        print(f"3-minute data saved to {new_file_path}")

