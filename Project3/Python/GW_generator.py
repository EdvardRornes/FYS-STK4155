import numpy as np

class GWSignalGenerator:
    def __init__(self, signal_length: int):
        """
        Initialize the GWSignalGenerator with a signal length.
        """
        self.signal_length = signal_length
        self.labels = np.zeros(signal_length, dtype=int)  # Initialize labels to 0 (background noise)
        self.regions = []  # Store regions for visualization or further analysis

    def add_gw_event(self, y, start, end, amplitude_factor=0.2, spike_factor=0.5, spin_start=1, spin_end=20, scale=1):
        """
        Adds a simulated gravitational wave event to the signal and updates labels for its phases.
        Includes a spin factor that increases during the inspiral phase.

        Parameters:
        y:                Signal to append GW event to.
        start:            Start index for GW event.
        end:              End index for GW event.
        amplitude_factor: Peak of the oscillating signal in the insipral phase.
        spike_factor:     Peak of the signal in the merge phase.
        spin_start:       Oscillation frequency of the start of the inspiral phase.
        spin_end:         Oscillation frequency of the end of the inspiral phase.
        scale:            Scale the amplitude of the entire event.

        returns:
        Various parameters to be used by apply_events function
        """
        event_sign = np.random.choice([-1, 1])  # Random polarity for the GW event

        amplitude_factor=amplitude_factor*scale
        spike_factor=spike_factor*scale

        # Inspiral phase
        inspiral_end = int(start + 0.7 * (end - start))  # Define inspiral region as 70% of event duration
        time_inspiral = np.linspace(0, 1, inspiral_end - start)  # Normalized time for the inspiral
        amplitude_increase = np.linspace(0, amplitude_factor, inspiral_end - start)
        
        # Spin factor: linearly increasing frequency
        spin_frequency = np.linspace(spin_start, spin_end, inspiral_end - start)  # Spin frequency in Hz
        spin_factor = np.sin(2 * np.pi * spin_frequency * time_inspiral)
        
        y[start:inspiral_end] += event_sign * amplitude_increase * spin_factor
        # self.labels[start:inspiral_end] = 1  # Set label to 1 for inspiral

        # Merger phase
        merge_start = inspiral_end
        merge_end = merge_start + int(0.1 * (end - start))  # Define merger as 10% of event duration
        y[merge_start:merge_end] += event_sign * spike_factor * np.exp(-np.linspace(3, 0, merge_end - merge_start))
        # self.labels[merge_start:merge_end] = 2  # Set label to 2 for merger

        # Ringdown phase
        dropoff_start = merge_end
        dropoff_end = dropoff_start + int(0.2 * (end - start))  # Define ringdown as 20% of event duration
        dropoff_curve = spike_factor * np.exp(-np.linspace(0, 15, dropoff_end - dropoff_start))
        y[dropoff_start:dropoff_end] += event_sign * dropoff_curve
        # self.labels[dropoff_start:dropoff_end] = 3  # Set label to 3 for ringdown

        # We cut off 2/3rds of the ringdown event due to the harsh exponential supression.
        # It is not expected that the NN will detect anything past this and may cause confusion for the program.
        self.labels[start:(2*dropoff_start+dropoff_end)//3] = 1

        # Store region details for visualization or debugging
        self.regions.append((start, end, inspiral_end, merge_start, merge_end, dropoff_start, dropoff_end))

    def generate_random_events(self, num_events: int, event_length_range: tuple, scale=1, 
                               amplitude_factor_range = (0, 0.5), spike_factor_range = (0.2, 1.5),
                               spin_start_range = (1, 5), spin_end_range = (5, 20)):
        """
        Generate random gravitational wave events with no overlaps.
        """
        events = []
        used_intervals = []

        for _ in range(num_events):
            while True:
                # Randomly determine start and length of event
                event_length = np.random.randint(*event_length_range)
                event_start = np.random.randint(0, self.signal_length - event_length)
                event_end = event_start + event_length

                # Ensure no overlap
                if not any(s <= event_start <= e or s <= event_end <= e for s, e in used_intervals):
                    used_intervals.append((event_start, event_end))
                    break  # Valid event, exit loop

            # Randomize event properties
            amplitude_factor = np.random.uniform(*amplitude_factor_range)
            spike_factor = np.random.uniform(*spike_factor_range)
            
            # Randomize spin start and end frequencies
            spin_start =np.random.uniform(*spin_start_range)  # Starting spin frequency (in Hz)
            spin_end = np.random.uniform(*spin_end_range)  # Ending spin frequency (in Hz)

            events.append((event_start, event_end, amplitude_factor * scale, spike_factor * scale, spin_start, spin_end))

        return events

    def apply_events(self, y, events):
        """
        Apply generated events generated by add_gw_signal function to the input signal.
        Can be manually created using this function 
        """
        for start, end, amplitude, spike, spin_start, spin_end in events:
            self.add_gw_event(y, start, end, amplitude_factor=amplitude, spike_factor=spike, spin_start=spin_start, spin_end=spin_end)