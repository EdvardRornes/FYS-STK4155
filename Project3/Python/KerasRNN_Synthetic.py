import numpy as np
import time
import pickle
import os

from utils import latex_fonts, GWSignalGenerator
from NNs import KerasRNN
from utils import * 
np.random.seed(0)

# latex_fonts()
savefigs = True

# Parameters
time_steps = 5000
time_for_1_sample = 50
num_samples = 5
window_size = time_steps//100
batch_size = time_steps//50*(num_samples-1)
batch_size = 128
learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
regularization_values = np.logspace(-12, -6, 4)
gw_earlyboosts = np.linspace(1, 1.5, 4)
# gw_earlyboosts = [1.4]
epoch_list = [10, 25, 50, 100]
SNR = 30

x = np.linspace(0, time_for_1_sample, time_steps)

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

events = []
labels = []

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
save_path = "RNN_Data/Custom_RNN/GW_Parameter_Search_V1"
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
                    y_test = y[fold]  # Use the fold as the test set
                    test_labels = labels[fold] # Corresponding labels for the test set

                    # Initialize the KerasRNN model with the current learning rate and regularization
                    hidden_layers = [5, 10, 2]
                    model = KerasRNN(1, hidden_layers, 1, Adam(), "tanh",
                        activation_out="sigmoid",
                        lambda_reg=reg_value,
                    )

                    # Recompile the model with updated regularization
                    # model.create_model()
                    model.model.compile(
                        loss=model._loss_function.type, 
                        optimizer=model.optimizer.name, 
                        metrics=['accuracy']
                    )
                    
                    for i in range(num_samples):
                        if i != fold:
                            model.train(y[i], labels[i].reshape(-1,1), int(epochs), batch_size, window_size)
                    # Train the model for this fold
                    

                    # Predict with the trained model
                    predictions = model.predict(y_test, test_labels, window_size, verbose=0)
                    predictions = predictions.reshape(-1)
                    loss = model._loss_function(y_test, predictions, epochs)
                    loss, accuracy = model.evaluate(y_test, predictions)
                    x_pred = x[window_size - 1:]

                    results.append({
                        "epochs": epochs,
                        "boost": boost,
                        "learning_rate": lr,
                        "regularization": reg_value,
                        "fold": fold,
                        # "train_labels": train_labels.tolist(),
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