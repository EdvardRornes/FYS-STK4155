
import os
# Removes some print out details
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import tensorflow.keras as ker # type: ignore
import pickle
import tensorflow as tf
import copy

from sklearn.model_selection import KFold, train_test_split
from keras.models import Sequential, load_model # type: ignore
from keras.optimizers import Adam, SGD, RMSprop # type: ignore
from keras.layers import SimpleRNN, Dense, Input # type: ignore
from keras.callbacks import ModelCheckpoint # type: ignore
from keras.regularizers import l2 # type: ignore
from utils import latex_fonts
from tensorflow.keras import mixed_precision # type: ignore
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# tf.config.threading.set_intra_op_parallelism_threads(12)  # Adjust number of threads
# tf.config.threading.set_inter_op_parallelism_threads(12)

latex_fonts()
savefigs = True

class KerasRNN:
    def __init__(self, hidden_layers: list, dim_output: int, dim_input: int,  
                 loss_function="binary_crossentropy", optimizer="adam", labels=None, 
                 gw_class_early_boost=1, learning_rate=1e-2, l2_regularization=0.0, activation_func='tanh'):
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
        self.activation_func = activation_func

        # Initialize the RNN model
        self.model = self.create_model()  # Call create_model during initialization to set up the model

    def create_model(self):
        """
        Creates and returns a fresh RNN model with the specified configurations.
        """
        model = ker.models.Sequential()

        # Add the input layer (input shape)
        model.add(Input(shape=(self.dim_input, 1)))  # Specify input shape here

        # Add RNN layers with optional L2 regularization
        for idx, units in enumerate(self.hidden_layers):
            model.add(SimpleRNN(units, activation=self.activation_func,
                                return_sequences=True if units != self.hidden_layers[-1] else False, 
                                kernel_regularizer=l2(self.l2_regularization)))

        # Output layer
        model.add(Dense(units=self.dim_output, activation="sigmoid",
                        kernel_regularizer=l2(self.l2_regularization)))

        # Compile the model
        self.compile_model(model)

        return model

    def compile_model(self, model):
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

        model.compile(
            loss=self.loss_function, 
            optimizer=optimizers[self.optimizer], 
            metrics=['accuracy']
        )

    def prepare_sequences_RNN(self, X: np.ndarray, y: np.ndarray, step_length: int):
        """
        Converts data into sequences for RNN training.
        """
        n_samples = len(X) - step_length + 1
        X_seq = np.array([X[i:i + step_length] for i in range(n_samples)]).reshape(-1, step_length, 1)
        y_seq = y[step_length-1:]
        print(X_seq.shape)
        print(y_seq.shape)
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

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int, step_length: int, verbose=1, verbose1=1):
        """
        Train the RNN model using dynamically computed class weights, stopping early if no improvement occurs for 30% of the epochs.
        Continues training if the loss is sufficiently low, even if no improvement is observed.
        """
        # Split training data into a validation set
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Prepare sequences for RNN input
        X_train_seq, y_train_seq = self.prepare_sequences_RNN(X_train, y_train, step_length)
        X_val_seq, y_val_seq = self.prepare_sequences_RNN(X_val, y_val, step_length)

        # Reinitialize model (this ensures no previous weights are carried over between parameter runs)
        self.model = self.create_model()

        # Initialize variables to track the best model and validation loss
        best_val_loss = float('inf')
        best_weights = None  # Keep track of the best model's weights, not the entire model

        # Define thresholds
        patience_threshold = int(np.ceil(0.15 * epochs))  # Early stopping threshold
        epochs_without_improvement = 0
        low_loss_threshold = 0.2  # Continue training even without improvement if loss is below this value

        # Compute class weights dynamically based on training labels
        for epoch in range(epochs):
            class_weights = self.compute_class_weights(epoch, epochs)

            # Fit the model for one epoch and save the history
            history = self.model.fit(
                X_train_seq, y_train_seq,
                epochs=1, # 1 epoch due to dynamic class weight, still doing "epoch" epochs due to loop
                batch_size=batch_size,
                verbose=verbose,
                class_weight=class_weights,
                validation_data=(X_val_seq, y_val_seq)
            )

            # Extract val_loss for the current epoch
            current_val_loss = history.history['val_loss'][0]

            # Check if this is the best val_loss so far
            if current_val_loss < best_val_loss:
                best_weights = self.model.get_weights()  # Save the best weights
                if verbose1 == 1:
                    print(f"Epoch {epoch + 1} - val_loss improved from {best_val_loss:.3f} to {current_val_loss:.3f}. Best model updated.")
                best_val_loss = current_val_loss
                epochs_without_improvement = 0  # Reset counter
            else:
                if verbose1 == 1:
                    print(f"Epoch {epoch + 1} - val_loss did not improve ({current_val_loss:.3f} >= {best_val_loss:.3f}).")
                epochs_without_improvement += 1

            # Check early stopping conditions
            if epochs_without_improvement >= patience_threshold:
                if best_val_loss >= low_loss_threshold:
                    if verbose1 == 1 and epoch != epochs-1:
                        print(f"No improvement for {patience_threshold} consecutive epochs and val_loss >= {low_loss_threshold}. Stopping early at epoch {epoch + 1}.")
                    break
                else:
                    if verbose1 == 1 and epoch != epochs-1:
                        print(f"No improvement for {patience_threshold} consecutive epochs, but val_loss < {low_loss_threshold}. Continuing training.")

        # After training, restore the best model weights (so model doesn't carry over worse performance)
        if best_weights is not None:
            self.model.set_weights(best_weights)  # Set the model's weights to the best found during training

    def predict(self, X_test, y_test, step_length, verbose=1):
        """
        Generate predictions for test data.
        """
        X_test_seq, y_test_seq = self.prepare_sequences_RNN(X_test, y_test, step_length)
        prediction = self.model.predict(X_test_seq, verbose=verbose)
        loss, accuracy = self.model.evaluate(X_test_seq, y_test_seq, verbose=verbose)
        return prediction, loss, accuracy


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
        Apply generated events generated by add_gw_signal function to the input signal.
        Can be manually created using this function 
        """
        for start, end, amplitude, spike, spin_start, spin_end in events:
            self.add_gw_event(y, start, end, amplitude_factor=amplitude, spike_factor=spike, spin_start=spin_start, spin_end=spin_end)


# Create the GWSignalGenerator instance
time_steps = 5000
time_for_1_sample = 50
x = np.linspace(0, time_for_1_sample, time_steps)
num_samples = 5
step_length = time_steps//100
batch_size = time_steps//50*(num_samples-1)
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
regularization_values = np.logspace(-10, 0, 11)
gw_earlyboosts = np.linspace(1, 1.5, 6)
epoch_list = [10, 25, 50, 100]
SNR = 100

# epoch_list=[50]
# learning_rates=[5e-3, 1e-2]
# gw_earlyboosts=np.linspace(1.5,1,6)
# regularization_values=np.logspace(-9,-6,4)

events = []
labels = []

# Background noise
y = [
    0.5*np.sin(90*x) - 0.5*np.cos(60*x)*np.sin(-5.*x) + 0.3*np.cos(30*x) + 0.05*np.sin(time_steps/40*x),
    0.5*np.sin(50*x) - 0.5*np.cos(80*x)*np.sin(-10*x) + 0.3*np.cos(40*x) + 0.05*np.sin(time_steps/20*x),
    0.5*np.sin(40*x) - 0.5*np.cos(25*x)*np.sin(-10*x) + 0.3*np.cos(60*x) + 0.10*np.sin(time_steps/18*x),
    0.7*np.sin(70*x) - 0.4*np.cos(10*x)*np.sin(-15*x) + 0.4*np.cos(80*x) + 0.05*np.sin(time_steps/12*x),
    0.1*np.sin(80*x) - 0.4*np.cos(50*x)*np.sin(-3.*x) + 0.3*np.cos(20*x) + 0.02*np.sin(time_steps/30*x)
]


for i in range(len(y)):
    y[i] /= SNR # Quick rescaling, the division factor is ~ SNR

# y = []

event_lengths = [(time_steps//10, time_steps//8), (time_steps//7, time_steps//6), 
                 (time_steps//14, time_steps//12), (time_steps//5, time_steps//3),
                 (time_steps//5, time_steps//4)]

# Add a single synthetic GW event to each sample
for i in range(num_samples):
    generator = GWSignalGenerator(signal_length=time_steps)
    # y_i = np.zeros_like(x) # For no background signal tests
    events_i = generator.generate_random_events(1, event_lengths[i])
    generator.apply_events(y[i], events_i)

    # y.append(y_i)
    events.append(events_i)
    labels.append(generator.labels)

# Convert lists into numpy arrays
y = np.array(y)
labels = np.array(labels)

# Reshape y for RNN input: (samples, time_steps, features)
y = y.reshape((y.shape[0], y.shape[1], 1))

# Prepare to save data
save_path = "GW_Parameter_Tuning_Results"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Function to save results incrementally
def save_results_incrementally(results, base_filename):
    filename = f"{base_filename}.pkl"
    with open(os.path.join(save_path, filename), "wb") as f:
        pickle.dump(results, f)
    print(f'File {filename} saved to {save_path}.')

progress = 0
total_iterations = len(learning_rates)*len(regularization_values)*len(gw_earlyboosts)*len(epoch_list)*num_samples
start_time = time.time()

# Loop over learning rates and regularization values
for epochs in epoch_list:
    for boost in gw_earlyboosts:
        for lr in learning_rates:
            for reg_value in regularization_values:
                results = []

                # Create a unique filename for the current parameter combination
                base_filename = f"Synthetic_GW_Parameter_Tuning_Results_timesteps{time_steps}_SNR{SNR}_epoch{int(epochs)}_lamd{reg_value}_eta{lr}_boost{boost:.1f}"

                # Check if results already exist
                if os.path.exists(os.path.join(save_path, f"{base_filename}.pkl")):
                    print(f"Skipping calculation for {base_filename} as the results already exist.")
                    total_iterations -= num_samples
                    continue  # Skip the calculation and move to the next combination

                print(f"\nTraining with eta = {lr}, lambda = {reg_value}, epochs = {epochs}, early boost = {boost:.1f}")

                for fold in range(num_samples):                    
                    # Split the data into train and test sets for this fold
                    x_test = x  
                    y_test = y[fold]  # Use the fold as the test set
                    test_labels = labels[fold] # Corresponding labels for the test set

                    # Create the training set using all other samples
                    x_train = np.linspace(0, (num_samples - 1) * time_for_1_sample, time_steps * (num_samples - 1))  # Just for plotting
                    y_train = np.concatenate([y[i] for i in range(num_samples) if i != fold], axis=0)
                    train_labels = np.concatenate([labels[i] for i in range(num_samples) if i != fold], axis=0)
                    
                    # Initialize the KerasRNN model with the current learning rate and regularization
                    hidden_layers = [5, 10, 2]  # Example hidden layers
                    model = KerasRNN(
                        hidden_layers, 
                        dim_output=1, 
                        dim_input=1, 
                        labels=train_labels, 
                        gw_class_early_boost=boost, 
                        learning_rate=lr,
                        l2_regularization=reg_value
                    )

                    # Recompile the model with updated regularization
                    model.model.compile(
                        loss=model.loss_function, 
                        optimizer=model.optimizer, 
                        metrics=['accuracy']
                    )
                    # Train the model for this fold
                    model.train(y_train, train_labels, epochs=int(epochs), batch_size=batch_size, step_length=step_length, verbose=0)

                    # Predict with the trained model
                    predictions, loss, accuracy = model.predict(y_test, test_labels, step_length, verbose=0)
                    predictions = predictions.reshape(-1)
                    predicted_labels = (predictions > 0.5).astype(int)
                    x_pred = x[step_length - 1:]

                    results.append({
                        "epochs": epochs,
                        "boost": boost,
                        "learning_rate": lr,
                        "regularization": reg_value,
                        "fold": fold,
                        "train_labels": train_labels.tolist(),
                        "y_test": y_test.tolist(),
                        "test_labels": test_labels.tolist(),
                        "predictions": predictions.tolist(),
                        "loss": loss,
                        "accuracy": accuracy
                    })

                    progress += 1
                    percentage_progress = (progress / total_iterations) * 100
                    # Elapsed time
                    elapsed_time = time.time() - start_time
                    ETA = elapsed_time*(100/percentage_progress-1)

                    print(f"Progress: {progress}/{total_iterations} ({percentage_progress:.2f}%), Time elapsed = {elapsed_time:.1f}s, ETA = {ETA:.1f}s, Test loss = {loss:.3f}, Test accuracy = {100*accuracy:.1f}%\n")

                save_results_incrementally(results, base_filename)