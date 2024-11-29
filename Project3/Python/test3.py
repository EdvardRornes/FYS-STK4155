import numpy as np
import random
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt
import os
from utils import latex_fonts

import numpy as np
import tensorflow.keras as ker
from keras.callbacks import ModelCheckpoint

from keras.regularizers import l2
from keras.optimizers import Adam, SGD, RMSprop
import time

latex_fonts()

class KerasRNN:
    def __init__(self, hidden_layers: list, dim_output: int, dim_input: int,  
                 loss_function="binary_crossentropy", optimizer="adam", labels=None, 
                 gw_class_early_boost=1, learning_rate=1e-2, l2_regularization=0.0):
        """
        Initializes an RNN model for multi-class classification.
        """
        self.hidden_layers = hidden_layers
        self.dim_output = dim_output  # Number of output classes
        self.dim_input = dim_input
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.gw_class_early_boost = gw_class_early_boost
        self.labels = labels
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization

        # Initialize the RNN model
        self.model = ker.models.Sequential()

        # Add RNN layers with optional L2 regularization
        for units in hidden_layers:
            if self.model.layers:
                self.model.add(SimpleRNN(units, 
                                         return_sequences=True if units != hidden_layers[-1] else False, 
                                         kernel_regularizer=l2(l2_regularization)))
            else:
                self.model.add(SimpleRNN(units, 
                                         return_sequences=True if units != hidden_layers[-1] else False, 
                                         input_shape=(dim_input, 1),
                                         kernel_regularizer=l2(l2_regularization)))

        # Output layer
        self.model.add(ker.layers.Dense(units=self.dim_output, activation="sigmoid",
                                        kernel_regularizer=l2(l2_regularization)))

        # Compile the model with the selected optimizer
        self.compile_model()

    def compile_model(self):
        """
        Compiles the model with the selected optimizer and learning rate.
        """
        optimizers = {
            "adam": Adam(learning_rate=self.learning_rate),
            "sgd": SGD(learning_rate=self.learning_rate),
            "rmsprop": RMSprop(learning_rate=self.learning_rate)
        }

        if self.optimizer not in optimizers:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}. Choose from {list(optimizers.keys())}.")

        self.model.compile(
            loss=self.loss_function, 
            optimizer=optimizers[self.optimizer], 
            metrics=['accuracy']
        )

    def prepare_sequences_RNN(self, X: np.ndarray, y: np.ndarray, step_length: int):
        """
        Converts data into sequences for RNN training.
        
        Parameters:
        X:              1D array of scaled data.
        y:              Output data.
        step_length:    Length of each sequence.
        
        Returns:
        tuple:          Sequences (3D array) and corresponding labels (1D array).
        """
        sequences = []
        labels = []

        for i in range(len(X) - step_length + 1):
            seq = X[i:i + step_length]
            sequences.append(seq)
            label = y[i + step_length - 1]
            labels.append(label)

        X_seq, y_seq = np.array(sequences).reshape(-1, step_length, 1), np.array(labels)
        return X_seq, y_seq
    
    def compute_class_weights(self, epoch: int, total_epochs: int):
        """
        Compute class weights dynamically based on the label distribution.
        """
        if self.labels is not None:
            initial_boost = self.gw_class_early_boost
            scale = initial_boost - (initial_boost - 1) * epoch / total_epochs
            gw_class_weight = (len(self.labels) - np.sum(self.labels)) / np.sum(self.labels) * scale
            return {0: 1, 1: gw_class_weight}
        else:
            print("Labels required for training, exiting")
            exit()

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int, step_length: int, verbose=1):
        """
        Train the RNN model using the dynamically computed class weights.
        """
        # Prepare sequences for RNN input
        X_train_seq, y_train_seq = self.prepare_sequences_RNN(X_train, y_train, step_length)

        # Compute class weights dynamically based on training labels
        for epoch in range(epochs):
            class_weights = self.compute_class_weights(epoch, epochs)

            # Create a folder for saving model weights (if it doesn't exist)
            checkpoint_dir = 'model_checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)  # creates the folder if it doesn't exist

            # Create a checkpoint callback with a file path inside the specified folder
            if epoch != 0:
                checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, f"model_epoch_{epoch}.keras"), 
                                             monitor='val_loss', save_best_only=True, mode='min', verbose=1)

            # Then pass this callback to the fit function
            self.model.fit(
                X_train_seq, y_train_seq,
                epochs=1,
                batch_size=batch_size,
                verbose=verbose,
                class_weight=class_weights,
                callbacks=[checkpoint] if epoch != 0 else None  # Save model after each epoch
            )

    def predict(self, X_test, y_test, step_length, verbose=1):
        """
        Generate predictions for test data.
        """
        X_test_seq, y_test_seq = self.prepare_sequences_RNN(X_test, y_test, step_length)
        return self.model.predict(X_test_seq, verbose=verbose)


        


class GWSignalGenerator:
    def __init__(self, signal_length: int):
        """
        Initialize the GWSignalGenerator with a signal length and noise level.
        """
        self.signal_length = signal_length
        self.labels = np.zeros(signal_length, dtype=int)  # Initialize labels to 0 (background noise)
        self.regions = []  # Store regions for visualization or further analysis

    def add_gw_event(self, y, start, end, amplitude_factor=0.2, spike_factor=0.5, spin_start=1, spin_end=20, scale=1):
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
                event_length = random.randint(*event_length_range)
                event_start = random.randint(0, self.signal_length - event_length)
                event_end = event_start + event_length

                # Ensure no overlap
                if not any(s <= event_start <= e or s <= event_end <= e for s, e in used_intervals):
                    used_intervals.append((event_start, event_end))
                    break  # Valid event, exit loop

            # Randomize event properties
            amplitude_factor = random.uniform(*amplitude_factor_range)
            spike_factor = random.uniform(*spike_factor_range)
            
            # Randomize spin start and end frequencies
            spin_start = random.uniform(*spin_start_range)  # Starting spin frequency (in Hz)
            spin_end = random.uniform(*spin_end_range)  # Ending spin frequency (in Hz)

            events.append((event_start, event_end, amplitude_factor * scale, spike_factor * scale, spin_start, spin_end))

        return events

    def apply_events(self, y, events):
        """
        Apply generated events to the input signal.
        """
        for start, end, amplitude, spike, spin_start, spin_end in events:
            self.add_gw_event(y, start, end, amplitude_factor=amplitude, spike_factor=spike, spin_start=spin_start, spin_end=spin_end)


# Create the GWSignalGenerator instance
time_steps = 10000
time_for_1_sample = 50
x = np.linspace(0, time_for_1_sample, time_steps)
num_samples = 5
step_length = time_steps//1000//(num_samples-1)
batch_size = time_steps//100
epochs = 25

y = []
events = []
labels = []

y = [
    0.5 * np.sin(100 * x) - 0.5 * np.cos(60 * x) * np.sin(-5 * x) + 0.3 * np.cos(30 * x) + 0.05 * np.sin(10000 * x),  # Original
    0.5 * np.sin(50 * x) - 0.5 * np.cos(80 * x) * np.sin(-10 * x) + 0.3 * np.cos(40 * x) + 0.05 * np.sin(5000 * x),  # Lower frequency, similar amplitude
    0.5 * np.sin(200 * x) - 0.5 * np.cos(120 * x) * np.sin(-10 * x) + 0.3 * np.cos(60 * x) + 0.1 * np.sin(20000 * x),  # Higher frequency, similar amplitude
    0.7 * np.sin(300 * x) - 0.4 * np.cos(150 * x) * np.sin(-15 * x) + 0.4 * np.cos(80 * x) + 0.05 * np.sin(12000 * x),  # Higher amplitude, high frequency
    0.1 * np.sin(80 * x) - 0.4 * np.cos(50 * x) * np.sin(-3 * x) + 0.3 * np.cos(20 * x) + 0.02 * np.sin(15000 * x)  # Low amplitude, high frequency
]

for i in range(len(y)):
    y[i] /= 330

event_lengths = [(time_steps//10, time_steps//8), (time_steps//7, time_steps//6), 
                 (time_steps//14, time_steps//12), (time_steps//5, time_steps//3),
                 (time_steps//5, time_steps//4)]

for i in range(num_samples):
    generator = GWSignalGenerator(signal_length=time_steps)
    # y_i = np.zeros_like(x)
    events_i = generator.generate_random_events(1, event_lengths[i])
    generator.apply_events(y[i], events_i)

    # y.append(y_i)
    events.append(events_i)
    labels.append(generator.labels)

# Convert lists into numpy arrays
y = np.array(y)
labels = np.array(labels)

# Check the output
print("y:", y)
print("events:", events)
print("labels:", labels)

# Reshape y for RNN input: (samples, time_steps, features)
y = y.reshape((y.shape[0], y.shape[1], 1))

# Check the reshaped y
print("Reshaped y:", y.shape)

learning_rates = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3]  # Example learning rates
regularization_values = np.logspace(-7, 0, 8)
gw_earlyboosts = np.linspace(1, 1.5, 5)
epoch_list = np.logspace(1, 3, 3)

progress = 0
total_iterations = len(learning_rates)*len(regularization_values)*len(gw_earlyboosts)*len(epoch_list)*num_samples
start_time = time.time()
# Loop over learning rates and regularization values
for epochs in epoch_list:
    for boost in gw_earlyboosts:
        for lr in learning_rates:
            for reg_value in regularization_values:
                plt.figure(figsize=(20, 12))
                plt.suptitle(fr"$\eta={lr}$, $\lambda={reg_value}$")
                print(f"\nTraining with Learning Rate: {lr}, L2 Regularization: {reg_value}")

                for fold in range(num_samples):                    
                    # Split the data into train and test sets for this fold
                    X_test = x  # Use the fold as the test set
                    y_test = y[fold]  # Corresponding labels for the test set
                    test_labels = labels[fold]

                    # Create the training set using all other samples
                    X_train = np.linspace(0, (num_samples - 1) * time_for_1_sample, time_steps * (num_samples - 1))
                    y_train = np.concatenate([y[i] for i in range(num_samples) if i != fold], axis=0)
                    train_labels = np.concatenate([labels[i] for i in range(num_samples) if i != fold], axis=0)
                    
                    # Initialize the KerasRNN model with the current learning rate and regularization
                    hidden_layers = [64, 32]  # Example hidden layers
                    model = KerasRNN(
                        hidden_layers, 
                        dim_output=1, 
                        dim_input=1, 
                        labels=train_labels, 
                        gw_class_early_boost=boost, 
                        learning_rate=lr
                    )

                    # Add L2 regularization to all Dense layers in the model
                    for layer in model.model.layers:
                        if isinstance(layer, ker.layers.Dense):
                            layer.kernel_regularizer = ker.regularizers.L2(reg_value)

                    # Recompile the model with updated regularization
                    model.model.compile(
                        loss=model.loss_function, 
                        optimizer=model.optimizer, 
                        metrics=['accuracy']
                    )
                    # Train the model for this fold
                    model.train(y_train, train_labels, epochs=int(epochs), batch_size=batch_size, step_length=step_length, verbose=0)

                    # Predict with the trained model
                    predictions = model.predict(y_test, test_labels, step_length, verbose=0)
                    predictions = predictions.reshape(-1)
                    predicted_labels = (predictions > 0.5).astype(int)
                    x_pred = x[step_length - 1:]
                    
                    plt.subplot(2, 3, fold + 1)
                    plt.title(f"Round {fold+1}")
                    plt.plot(x, y[fold], label=f'Data {fold+1}', lw=0.5, color='b')
                    plt.plot(x, test_labels, label=f"Solution {fold+1}", lw=1.6, color='g')

                    # Highlight predicted events
                    predicted_gw_indices = np.where(predicted_labels == 1)[0]
                    if len(predicted_gw_indices) == 0:
                        print("No gravitational wave events predicted.")
                    else:
                        threshold = 2
                        grouped_gw_indices = []
                        current_group = [predicted_gw_indices[0]]

                        for i in range(1, len(predicted_gw_indices)):
                            if predicted_gw_indices[i] - predicted_gw_indices[i - 1] <= threshold:
                                current_group.append(predicted_gw_indices[i])
                            else:
                                grouped_gw_indices.append(current_group)
                                current_group = [predicted_gw_indices[i]]

                        grouped_gw_indices.append(current_group)
                        for i, group in zip(range(len(grouped_gw_indices)), grouped_gw_indices):
                            plt.axvspan(x[group[0]], x[group[-1]], color="red", alpha=0.3, label="Predicted event" if i == 0 else "")
                    progress += 1
                    percentage_progress = (progress / total_iterations) * 100
                    # Elapsed time
                    elapsed_time = time.time() - start_time
                    ETA = elapsed_time*(100/percentage_progress-1)

                    print(f"Progress: {progress}/{total_iterations} ({percentage_progress:.2f}%), Time elapsed = {elapsed_time:.1f}s, ETA = {ETA:1f}s")

                    plt.legend()
                plt.savefig(f"../Figures/SyntheticGWs_SNR300_lr{lr}_lambd{reg_value}_epochs{int(epochs)}_earlyboost{boost}.pdf")  # Save the figure
                plt.close()  # Close the figure




