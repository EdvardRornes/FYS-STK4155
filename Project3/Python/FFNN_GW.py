import h5py
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils import *

# Function to load strain data and labels from HDF5 file
def load_hdf5_data(file_path):
    with h5py.File(file_path, 'r') as f:
        # Loading the strain data (signal data) and labels (0 or 1)
        X = np.array(f['strain'])  # Input data (gravitational wave data)
        y = np.array(f['signal'])  # Labels indicating signal (1) or noise (0)
    return X, y

# Function to train FFNN with provided training and testing data
def train_fnn(X_train, y_train, X_test, y_test, input_size, hidden_layers, output_size, optimizer, activation, lambda_reg=0.0, alpha=0.1, loss_function='mse'):
    # Initialize the neural network
    ffnn = FFNN(input_size=input_size, hidden_layers=hidden_layers, output_size=output_size, optimizer=optimizer, activation=activation, 
                lambda_reg=lambda_reg, alpha=alpha, loss_function=loss_function)
    
    # Train the network
    loss_history = ffnn.train(X_train, y_train, epochs=100, batch_size=32, shuffle=True)

    # Get the predictions and compute confusion matrix
    y_pred = ffnn.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)  # For binary classification (0 or 1)

    cm = confusion_matrix(y_test.flatten(), y_pred_classes.flatten())
    
    return loss_history, cm

# Function to plot and save confusion matrix
def plot_confusion_matrix(cm, class_names, filename):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.show()

# Main execution to process data, train the network, and save results
def main(train_file, test_file, output_file, input_size, hidden_layers, output_size, optimizer, activation, lambda_reg=0.0, alpha=0.1, loss_function='mse'):
    # Load training data
    X_train, y_train = load_hdf5_data(train_file)
    
    # Load testing data
    X_test, y_test = load_hdf5_data(test_file)
    
    # Train the FFNN and get loss history and confusion matrix
    loss_history, cm = train_fnn(X_train, y_train, X_test, y_test, input_size, hidden_layers, output_size, optimizer, activation, lambda_reg, alpha, loss_function)
    
    # Save results using the Data class (assuming Data class and store method are in utils)
    data_to_save = Data('ffnn')  # Assuming 'ffnn' is the trained model
    data_to_save.data.update({
        "loss_history": loss_history,
        "confusion_matrix": cm.tolist(),  # Converting to list to make it serializable
    })
    data_to_save.store(output_file)

    # Plot and save confusion matrix plot
    plot_confusion_matrix(cm, class_names=["Noise", "Signal"], filename=f"{output_file}_confusion_matrix.png")

# Example usage:
if __name__ == "__main__":
    train_file = 'Data/G-G1_GWOSC_O3GK_4KHZ_R1-1270702080-4096.hdf5'  # Path to training data file
    test_file = 'Data/H-H1_GWOSC_O3b_4KHZ_R1-1264312320-4096.hdf5'    # Path to testing data file
    output_file = 'Data/output_results'        # File to store results

    # Set parameters for the FFNN
    input_size = 2  # Example input size, adjust based on your data
    hidden_layers = [64, 32]  # Example hidden layers
    output_size = 1  # Binary classification (Signal or Noise)
    optimizer = 'adam'  # Optimizer (Adam)
    activation = 'relu'  # Activation function (ReLU)

    # Call the main function to run the entire process
    main(train_file, test_file, output_file, input_size, hidden_layers, output_size, optimizer, activation)
