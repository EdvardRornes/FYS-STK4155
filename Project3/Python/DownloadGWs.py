import numpy as np
import h5py
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps, run_segment
from numpy import random
import os
from datetime import datetime, timedelta

def download_and_save_gw_data(events, duration=300, sample_rate=4096, output_dir="Data"):
    """
    Downloads gravitational wave data for specified events and saves it locally.

    Parameters:
        events (list): List of event names (e.g., ["GW150914", "GW151226"]).
        duration (int): Duration of data to download in seconds.
        sample_rate (int): Sampling rate in Hz.
        output_dir (str): Directory to save the downloaded data.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for event in events:
        try:
            gps = event_gps(event)
        except Exception as e:
            print(f"Did not find known event {event}. Skipping...")
            continue  # Skip to the next event
        
        start_time = gps - duration // 2
        end_time = gps + duration // 2
        
        # Check if data already exists
        file_path = os.path.join(output_dir, f"{event}.hdf5")
        if os.path.exists(file_path):
            print(f"Data for {event} already exists. Skipping download.")
            continue
        
        print(f"Downloading data for {event}...")
        try:
            strain = TimeSeries.fetch_open_data("H1", start_time, end_time, sample_rate=sample_rate)
        except Exception as e:
            print(f"Failed to download data for {event}: {e}")
            continue
        
        # Save the data
        with h5py.File(file_path, "w") as f:
            f.create_dataset("strain", data=strain.value)
            f.attrs["start_time"] = start_time
            f.attrs["end_time"] = end_time
            f.attrs["sample_rate"] = sample_rate
        
        print(f"Data for {event} saved to {file_path}.")

def split_into_intervals(file_path, interval=1):
    """
    Splits gravitational wave data into smaller intervals.

    Parameters:
        file_path (str): Path to the HDF5 file with gravitational wave data.
        interval (int): Interval length in seconds.
        
    Returns:
        np.ndarray: Array of split intervals.
    """
    with h5py.File(file_path, "r") as f:
        strain = f["strain"][:]
        sample_rate = f.attrs["sample_rate"]
    
    samples_per_interval = interval * int(sample_rate)
    return np.split(strain, np.arange(samples_per_interval, len(strain), samples_per_interval))

def convert_gps_to_utc(gps_time):
    """
    Converts GPS time to UTC.

    Parameters:
        gps_time (int): GPS time (seconds since 1980-01-06).

    Returns:
        datetime: Corresponding UTC datetime.
    """
    gps_epoch = datetime(1980, 1, 6)  # GPS epoch start date
    return gps_epoch + timedelta(seconds=gps_time)

def download_noise_data(runs, duration=300, sample_rate=4096, output_dir="Data", n_samples=3, retry_limit=5):
    """
    Downloads random 5-minute noise intervals without known gravitational wave signals.
    The periods are from different observation runs: O1, O2, O3, O4, and O5.

    Parameters:
        runs (list): List of run names (e.g., ["O1", "O2", "O3", "O4", "O5"]).
        duration (int): Duration of data to download in seconds.
        sample_rate (int): Sampling rate in Hz.
        output_dir (str): Directory to save the downloaded data.
        n_samples (int): Number of noise intervals to download.
        retry_limit (int): Number of retries before giving up on a specific interval.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to keep track of runs with the number of samples downloaded
    samples_downloaded = {run: 0 for run in runs}
    
    for run in runs:
        # Skip if we already have two samples for the current run
        if samples_downloaded[run] >= 2:
            print(f"Already have 2 samples for run {run}. Skipping...")
            continue
        
        # Fetch available segments for the run
        try:
            segment = run_segment(run)  # Fetch the segments for the given run
            print(f"Segment for run {run}: {segment}")  # Debugging line to inspect the segment
        except Exception as e:
            print(f"Error fetching segment for {run}: {e}")
            continue
        
        noise_intervals = []
        
        # Randomly sample intervals from the segment
        if isinstance(segment, tuple) and len(segment) == 2:
            start_segment, end_segment = segment
            while len(noise_intervals) < n_samples:
                start_time = random.randint(start_segment, end_segment - duration)
                end_time = start_time + duration
                noise_intervals.append((start_time, end_time))
        else:
            print(f"Segment data structure is invalid for {run}: {segment}")
            continue
        
        for i, (start_time, end_time) in enumerate(noise_intervals):
            # Convert start time to UTC and add to the filename
            start_utc = convert_gps_to_utc(start_time)
            date_str = start_utc.strftime('%y%m%d')  # Ensure proper UTC format for the date
            file_path = os.path.join(output_dir, f"NOISE{run}_{date_str}_{i + 1}.hdf5")
            
            # Check if noise data already exists
            if os.path.exists(file_path):
                print(f"Noise data interval {i + 1} for run {run} already exists. Skipping download.")
                continue
            
            retries = 0
            while retries < retry_limit:
                print(f"Downloading noise data interval {i + 1} for run {run} (attempt {retries + 1})...")
                try:
                    strain = TimeSeries.fetch_open_data("H1", start_time, end_time, sample_rate=sample_rate)
                    # Save the data
                    with h5py.File(file_path, "w") as f:
                        f.create_dataset("strain", data=strain.value)
                        f.attrs["start_time"] = start_time
                        f.attrs["end_time"] = end_time
                        f.attrs["sample_rate"] = sample_rate
                    print(f"Noise data interval {i + 1} for run {run} saved to {file_path}.")
                    
                    # Increment the count for this run
                    samples_downloaded[run] += 1
                    break  # Exit the retry loop if the download is successful
                except Exception as e:
                    print(f"Failed to download noise data interval {i + 1}: {e}")
                    retries += 1
                    if retries >= retry_limit:
                        print(f"Giving up after {retry_limit} attempts for interval {i + 1}.")
                        # Pick a new random time if the current interval fails
                        print("Retrying with a new random time...")
                        start_time = random.randint(start_segment, end_segment - duration)
                        end_time = start_time + duration
                        retries = 0  # Reset the retry counter after picking a new time
                        continue



# Example usage: download noise data for multiple runs
runs = ["O2", "O3", "O4", "O5"]
download_noise_data(runs, n_samples=2)

events = [
    "GW150914",  # First detected binary black hole merger
    "GW151226",  # Second detected binary black hole merger
    "GW170104",  # Binary black hole merger detected with high confidence
    "GW170814",  # Detected by all three LIGO and Virgo detectors
    "GW170817",  # First neutron star merger with electromagnetic counterpart
    "GW170823",  # Another binary black hole merger
    "GW190521",  # Large mass binary black hole merger
    "GW190412",  # Binary black hole merger with unequal masses
    "GW190425",  # First binary neutron star merger with heavy neutron stars
    "GW190814",  # Highly significant black hole merger with unusual mass
    "GW200105",  # Event detected during O4
    "GW200115",  # Notable 2020 detection
    "GW200129",  # Major detection during O4
    "GW220922",  # Virgo detection from O5
    "GW230307"   # Recent event from 2023
]

download_and_save_gw_data(events)
