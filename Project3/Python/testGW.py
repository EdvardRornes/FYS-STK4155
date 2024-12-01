import numpy as np
import matplotlib.pyplot as plt
from test3 import * 

# Generate synthetic data for gravitational event detection
# Generate synthetic data for gravitational event detection
def generate_gravitational_event_data(num_samples=1000, event_start=300, event_end=350):
    """
    Generates synthetic data representing a simple gravitational wave event with binary labels.
    
    Arguments:
    * num_samples: Total number of samples in the dataset
    * event_start: Index where the gravitational wave event starts
    * event_end: Index where the gravitational wave event ends
    
    Returns:
    * X: Input data (amplitude) of shape (num_samples, 1)
    * y: Labels (binary) of shape (num_samples, 1), 1 when event occurs, 0 otherwise
    """
    # Create the time series (amplitude) with noise
    X = np.zeros(num_samples)
    X[event_start:event_end] = np.exp(-0.5 * ((np.linspace(-2, 2, event_end - event_start))**2))
    noise = np.random.normal(0, 0.1, size=num_samples)
    X += noise
    X = np.clip(X, 0, 1)

    # Create binary labels, 1 during the event, 0 otherwise
    y = np.zeros(num_samples)
    y[event_start:event_end] = 1

    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return X, y

# Generate data
X, y = generate_gravitational_event_data()
print(np.shape(X), np.shape(y), "her")

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
                event_length = random.randint(event_length_min, event_length_max)
                event_start = random.randint(0, self.signal_length - event_length)
                event_end = event_start + event_length

                # Ensure no overlap
                if not any(s <= event_start <= e or s <= event_end <= e for s, e in used_intervals):
                    used_intervals.append((event_start, event_end))
                    break  # Valid event, exit loop

            # Randomize event properties
            amplitude_factor = random.uniform(0, 0.5)
            spike_factor = random.uniform(0.2, 1.5)
            
            # Randomize spin start and end frequencies
            spin_start = random.uniform(5, 30)  # Starting spin frequency (in Hz)
            spin_end = random.uniform(50, 500)  # Ending spin frequency (in Hz)

            events.append((event_start, event_end, amplitude_factor * scale, spike_factor * scale, spin_start, spin_end))

        return events

    def apply_events(self, y, events):
        """
        Apply generated events to the input signal.
        """
        for start, end, amplitude, spike, spin_start, spin_end in events:
            self.add_gw_event(y, start, end, amplitude_factor=amplitude, spike_factor=spike, spin_start=spin_start, spin_end=spin_end)


# # Plot the generated data and labels
# plt.figure(figsize=(10, 4))
# plt.plot(X, label='Amplitude')
# plt.plot(y, label='Event Label', linestyle='--')
# plt.legend()
# plt.title("Synthetic Gravitational Event Data")
# plt.xlabel("Time")
# plt.ylabel("Amplitude / Label")
# plt.show()

time_steps = 10000
x = np.linspace(0, 50, time_steps)
noise = 0.02

# Base signal: sine wave + noise
X = np.zeros_like(x)#1e-19*(0.5 * np.sin(100 * x) - 0.5 * np.cos(60 * x)*np.sin(-5*x) + 0.3*np.cos(30*x) + 0.05*np.sin(10000*x)) #+ noise * np.random.randn(time_steps)
X_noGW = X.copy()


generator = GWSignalGenerator(signal_length=time_steps, noise_level=noise)
# events = generator.generate_random_events(1, time_steps//3, time_steps//2, scale=1e-19)
# generator.apply_events(y, events)
GWSignalGenerator.add_gw_event(generator, X, time_steps//2, 5*time_steps//6, spin_start=3, spin_end=15, spike_factor=2, scale=1e-19)

y = generator.labels*1e19
X_reshaped = X.reshape(-1, 1)
y_reshaped = y.reshape(-1, 1)*1e19
print(np.shape(X))
print(np.shape(y))
# Plot the signal
plt.figure(figsize=(15, 6))
plt.plot(x, X_noGW, label="No GW Signal", lw=0.4, color="gray", alpha=0.7)
plt.plot(x, X, label="Signal (with GW events)", lw=0.6, color="blue")
plt.show()

def binary(y):

    return 1*(y>=0.5)

from sklearn.utils.class_weight import compute_class_weight

class_labels = np.unique(y)
class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y.ravel())
class_weight_dict = dict(zip(class_labels, class_weights))
print(class_weights)

rnn = RNN(input_size=1, hidden_layers=[4, 8, 16, 16, 8, 4], output_size=1, optimizer=Adam(), activation="tanh", activation_out="sigmoid",
          loss_function=WeightedBinaryCrossEntropyLoss(class_weights[0], class_weights[1]))

# Train the RNN model on the generated data
rnn.train(X_reshaped, y_reshaped, epochs=10, batch_size=100, window_size=250)

# Evaluate the trained model
# X, y = generate_gravitational_event_data()
loss, accuracy = rnn.evaluate(X_reshaped, y, window_size=25)

# X, y = generate_gravitational_event_data()
y_pred = rnn.predict(X.reshape(-1,1,1))
# y_pred = 1*(y_pred>=0.5)
# y_pred = binary(y_pred)


plt.figure(figsize=(15, 6))
plt.plot(x, X_noGW, label="No GW Signal", lw=0.4, color="gray", alpha=0.7)
plt.plot(x, X, label="Signal (with GW events)", lw=0.6, color="blue")
plt.plot(y, label='Event Label', linestyle='--')
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.legend()
plt.show()

# plt.figure(figsize=(10, 4))
# plt.plot(X, label='Amplitude')
# plt.plot(y, label='Event Label', linestyle='--')
# plt.plot(y_pred, label='Predicted', linestyle='--')
# plt.legend()
# plt.title("Synthetic Gravitational Event Data")
# plt.xlabel("Time")
# plt.ylabel("Amplitude / Label")
# plt.show()