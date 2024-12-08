from NNs import DynamicallyWeightedLoss, WeightedBinaryCrossEntropyLoss, FocalLoss
from test5 import RNN
from utils import * 
import numpy as np

class GWSignalGenerator:
    def __init__(self, signal_length, noise_level=0.1):
        """
        Initialize the GWSignalGenerator with a signal length and noise level.
        """
        self.signal_length = signal_length
        self.noise_level = noise_level
        self.labels = np.zeros(signal_length, dtype=int)  # Initialize labels to 0 (background noise)
        self.regions = []  # Store regions for visualization or further analysis

    def add_gw_event(self, y, start, end, amplitude_factor=1, spike_factor=0.8, spin_start=10, spin_end=100, scale=1):
        """
        Adds a simulated gravitational wave event to the signal and updates labels for its phases.
        Includes a spin factor that increases during the inspiral phase.
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

        self.labels[start:(2*dropoff_start+dropoff_end)//3] = 1

        # Store region details for visualization or debugging
        self.regions.append((start, end, inspiral_end, merge_start, merge_end, dropoff_start, dropoff_end))


    def generate_random_events(self, num_events, event_length_min, event_length_max, scale=1):
        """
        Generate random gravitational wave events with no overlaps.
        """
        events = []
        used_intervals = []

        for _ in range(num_events):
            while True:
                # Randomly determine start and length of event
                event_length = np.random.randint(event_length_min, event_length_max)
                event_start = np.random.randint(0, self.signal_length - event_length)
                event_end = event_start + event_length

                # Ensure no overlap
                if not any(s <= event_start <= e or s <= event_end <= e for s, e in used_intervals):
                    used_intervals.append((event_start, event_end))
                    break  # Valid event, exit loop

            # Randomize event properties
            amplitude_factor = np.random.uniform(0, 0.5)
            spike_factor = np.random.uniform(0.2, 1.5)
            
            # Randomize spin start and end frequencies
            spin_start = np.random.uniform(5, 30)  # Starting spin frequency (in Hz)
            spin_end = np.random.uniform(50, 500)  # Ending spin frequency (in Hz)

            events.append((event_start, event_end, amplitude_factor * scale, spike_factor * scale, spin_start, spin_end))

        return events

    def apply_events(self, y, events):
        """
        Apply generated events to the input signal.
        """
        for start, end, amplitude, spike, spin_start, spin_end in events:
            self.add_gw_event(y, start, end, amplitude_factor=amplitude, spike_factor=spike, spin_start=spin_start, spin_end=spin_end)




# Parameters
time_steps = 10000
t = np.linspace(0, 50, time_steps)
noise = 0.02

# Base signal: sine wave + noise
y = 2*(0.5 * np.sin(100 * t) - 0.5 * np.cos(60 * t)*np.sin(-5*t) + 0.3*np.cos(30*t) + 0.05*np.sin(10000*t)) #+ noise * np.random.randn(time_steps)
y_noGW = y.copy()

# Initialize generator and create events
generator = GWSignalGenerator(signal_length=time_steps, noise_level=noise)
# events = generator.generate_random_events(1, time_steps//3, time_steps//2, scale=1e-19)
# generator.apply_events(y, events)
print(time_steps // 2)
GWSignalGenerator.add_gw_event(generator, y, time_steps//2, 5*time_steps//6, spin_start=3, spin_end=15, spike_factor=2, scale=1)
# generator.labels = np.zeros(len(generator.labels))
# generator.labels[50] = 1
# GWSignalGenerator.add_gw_event(generator, y, np.argmin(np.abs(t-10)),  np.argmin(np.abs(t-20)), spin_start=3, spin_end=15, spike_factor=2, scale=1)

strain = y
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
np.random.seed(42)
# Assuming y is your label array
unique_classes = np.unique(generator.labels)
class_weights = compute_class_weight('balanced', classes=unique_classes, y=generator.labels)

# Extract weights
weight_0 = class_weights[0]  # Weight for class 0
weight_1 = class_weights[1]  # Weight for class 1
print(weight_0, weight_1)

testRNN = RNN(1, [5, 10, 2], 1, Adam(learning_rate=0.005, momentum=1), "tanh", "sigmoid", 
              lambda_reg=0.001, loss_function=WeightedBinaryCrossEntropyLoss(weight_0=weight_0, weight_1=weight_1),
              scaler="no scaling")

X = copy.deepcopy(strain.reshape(-1, 1))
labels = copy.deepcopy(generator.labels) 

batch_size = time_steps//50

t_max_index = np.argmin(np.abs(t-10)); window_size = np.argmin(np.abs(t-20))

window_size = 200
testRNN.train(X, labels.reshape(-1, 1), 10, batch_size=batch_size, window_size=window_size)

labels_pred = testRNN.predict(X.reshape(-1, 1, 1))

plt.figure(figsize=(15, 6))
plt.plot(t, y_noGW, label="No GW Signal", lw=0.4, color="gray", alpha=0.7)
plt.plot(t, strain, label="Signal (with GW events)", lw=0.6, color="blue")
plt.plot(t, generator.labels, label="Actual")
plt.plot(t, labels_pred[:, 0, 0], label="Predicted")
plt.legend()
plt.show()