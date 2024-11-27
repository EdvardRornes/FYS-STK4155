import numpy as np
import h5py
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps, run_segment
from numpy import random
import os
from datetime import datetime, timedelta
from astropy.time import Time

def is_noise_interval(start_time, end_time, known_events, buffer=10):
    """
    Checks if the given interval overlaps with any known gravitational wave events.

    Parameters:
        start_time (int): Start GPS time of the interval.
        end_time (int): End GPS time of the interval.
        known_events (list): List of known gravitational wave event times (GPS).
        buffer (int): Time buffer in seconds to account for close overlaps.

    Returns:
        bool: True if the interval does not overlap with any events, False otherwise.
    """
    for event_time in known_events:
        if start_time - buffer <= event_time <= end_time + buffer:
            return False
    return True

def event_times_utc(events):
    # Convert each event's GPS time to a Time object in UTC
    known_events_time_gps = [event_gps(event) for event in events]
    print("GPS Times:", known_events_time_gps)
    
    # Convert GPS times to Time objects and get UTC representation
    known_events_time_utc = [Time(gps_time, format='gps').utc.iso for gps_time in known_events_time_gps]
    print("UTC Times:", known_events_time_utc)
    
    return known_events_time_utc



def download_and_save_gw_data(events, duration=300, sample_rate=4096, output_dir="Data"):
    os.makedirs(output_dir, exist_ok=True)
    known_events = [event_gps(event) for event in events]

    for event in events:
        try:
            gps = event_gps(event)
        except Exception as e:
            print(f"Did not find known event {event}. Skipping...")
            continue

        start_time = gps - duration // 2
        end_time = gps + duration // 2
        file_path = os.path.join(output_dir, f"{event}.hdf5")
        if os.path.exists(file_path):
            print(f"Data for {event} already exists. Skipping download.")
            continue

        print(f"Downloading data for {event}...")
        try:
            strain = TimeSeries.fetch_open_data("H1", start_time, end_time, sample_rate=sample_rate)
            save_data_with_label(file_path, strain, start_time, end_time, sample_rate, label=1)
            print(f"Data for {event} saved to {file_path}.")
        except Exception as e:
            print(f"Failed to download data for {event}: {e}")

def download_noise_data(runs, duration=300, sample_rate=4096, output_dir="Data", n_samples=2, retry_limit=5):
    os.makedirs(output_dir, exist_ok=True)
    known_events = [event_gps(event) for event in events]
    samples_downloaded = {run: 0 for run in runs}

    for run in runs:
        if samples_downloaded[run] >= 2:
            print(f"Already have 2 samples for run {run}. Skipping...")
            continue

        try:
            segment = run_segment(run)
        except Exception as e:
            print(f"Error fetching segment for {run}: {e}")
            continue

        if isinstance(segment, tuple) and len(segment) == 2:
            start_segment, end_segment = segment
            for _ in range(n_samples):
                retries = 0
                while retries < retry_limit:
                    start_time = random.randint(start_segment, end_segment - duration)
                    end_time = start_time + duration
                    if is_noise_interval(start_time, end_time, known_events):
                        break
                    retries += 1
                else:
                    continue

                file_path = os.path.join(output_dir, f"NOISE{run}_{start_time}.hdf5")
                if os.path.exists(file_path):
                    print(f"Noise data for {file_path} already exists. Skipping...")
                    continue

                try:
                    strain = TimeSeries.fetch_open_data("H1", start_time, end_time, sample_rate=sample_rate)
                    save_data_with_label(file_path, strain, start_time, end_time, sample_rate, label=0)
                    samples_downloaded[run] += 1
                    print(f"Noise data saved to {file_path}.")
                except Exception as e:
                    print(f"Failed to download noise data: {e}")


# Example usage: download noise data for multiple runs
runs = ["O2", "O3", "O4", "O5"]
# download_noise_data(runs, n_samples=2)

# StN = Signal to Noise ratio
events = [
    "GW150914",  # First detected binary black hole merger 
                 # StN = 24.4
    "GW170817",  # First neutron star merger with electromagnetic counterpart
                 # StN = 33.0
    "GW190814"   # Highly significant black hole merger with unusual mass
                 # StN = 24.9
]

event_time = event_times_utc(events)
print(event_time)

# download_and_save_gw_data(events)
