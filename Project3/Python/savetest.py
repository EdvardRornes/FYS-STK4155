import numpy as np
import h5py
import datetime
from gwosc import datasets
from gwpy.timeseries import TimeSeries
from gwpy.timeseries import TimeSeries

# List of events (1 first GW detection, 3 high SNR, 1 low SNR)
events = [
    'GW150914',  # First detection
    'GW170104',  # High SNR event
    'GW170817',  # High SNR event
    'GW190521',  # High SNR event
    'GW170223',  # Low SNR event (SNR ~10)
]

# Fetch the event data using gwosc.datasets
for event_name in events:
    event_info = datasets.event_gps(event_name)

    # If the event exists in the catalog
    if event_info:
        gps_start = event_info
        gps_end = gps_start + 300  # Assuming the event lasts for 5 minutes

        # Randomly select a time within the 5-minute window, and adjust for 5-minute segment
        random_time = gps_start + np.random.randint(0, 300)

        # Construct the channel name for the event (L1 detector, strain data)
        channel = f"L1:GWOSC-XXXX"  # Replace XXXX with the event-specific suffix if needed

        try:
            # Fetch strain data from the detector for the 5-minute window
            strain = TimeSeries.fetch(channel, start=random_time-300, end=random_time+300)

            # Create a signal label (1 for within 1 second of the event, 0 otherwise)
            signal_label = np.zeros(strain.size)

            # Mark data as signal (1) if within 1 second of the event
            signal_label[np.abs(strain.times - gps_start) < 1] = 1
            signal_label[np.abs(strain.times - gps_end) < 1] = 1

            # Save data and labels in HDF5 format
            event_date = datetime.datetime.strptime(event_name[2:], '%y%m%d').date()
            filename = f'GWData/{event_date}.h5'

            with h5py.File(filename, 'w') as f:
                f.create_dataset('strain', data=strain.value)
                f.create_dataset('signal', data=signal_label)

            print(f'Saved {filename} with labels.')
        except Exception as e:
            print(f"Error fetching strain data for {event_name}: {e}")
    else:
        print(f'Event {event_name} not found in the event catalog.')
