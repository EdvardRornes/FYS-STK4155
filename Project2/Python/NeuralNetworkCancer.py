import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from utils import *
import time
import warnings

# Suppress specific warnings (they stem from poor parameter combinations, specifically high learning rates)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in matmul")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in matmul")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in multiply")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in subtract")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in reduce")

latex_fonts()
save = False

# Load the dataset
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target.reshape(-1, 1), random_state=1)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter ranges
hidden_layers = [2, 4, 8, 16, 32, 16, 8, 4, 2]
epochs = [100, 250, 1000]
epochs = [10]
# Nicer logspace for plotting of learning rates
eta_lower = -4
eta_upper = -1
etas = []
for m in range(eta_lower, eta_upper):
    etas.append(float(10**m))
    etas.append(float(2.5*10**m))
    etas.append(float(5*10**m))
    etas.append(float(7.5*10**m))
etas.append(float(10**eta_upper))

etas = [0.01]
lambdas = np.logspace(-15, -6, 10)
activations = ['relu', 'lrelu', 'sigmoid']

# Prepare to store results
results = {}
MSE_matrix = np.zeros((len(etas), len(lambdas)))
accuracy_matrix = np.zeros((len(etas), len(lambdas)))

# Prepare to store best parameters for each activation and epoch count
best_params = {epoch: {} for epoch in epochs}

# Timer
start_time = time.time()
total_iterations = len(epochs)*len(activations)*len(lambdas)*len(etas)
completed_iterations = 0

for epoch in epochs:
    for activation in activations:
        # Initialize variables to track the best parameters
        best_mse = float('inf')
        best_accuracy = 0
        best_lambda = None
        best_eta = None

        for j, lambd in enumerate(lambdas):
            for i, eta in enumerate(etas):
                # Suppress print from class to avoid clutter
                blockPrint()
                # Initialize the model
                model = FFNN(input_size=X_train.shape[1], hidden_layers=hidden_layers, output_size=1, activation=activation, lambda_reg=lambd)
                # Train the model
                print(f"Training with activation: {activation}, lambda: {lambd}, eta: {eta}")
                model.train(X_train, y_train, learning_rate=eta, epochs=epoch, batch_size=epoch//10, lambda_reg=lambd)
                # Predictions
                y_pred = model.predict(X_test)
                # Ensure y_pred is a 1D array for binary classification
                y_pred_binary = (y_pred > 0.5).astype(int).flatten()
                # Calculate MSE and accuracy
                mse = mean_squared_error(y_test, y_pred_binary)
                accuracy = accuracy_score(y_test, y_pred_binary)
                # Store the MSE in the matrix
                MSE_matrix[i, j] = mse
                accuracy_matrix[i, j] = accuracy
                # Store the results
                results[(activation, lambd, eta)] = {
                    'mse': mse,
                    'accuracy': accuracy
                }
                enablePrint()

                # Calculate progress percentage
                completed_iterations += 1
                percentage_progress = (completed_iterations / total_iterations) * 100
                # Elapsed time
                elapsed_time = time.time() - start_time
                # Output results
                print(f"Activation: {activation}, Epochs: {epoch}, Lambda: {lambd:.2e}, Learning Rate: {eta:.2e}, MSE: {mse:.4f}, "
                      f"Accuracy: {100*accuracy:.1f}% | Progress: {percentage_progress:.1f}%, Time Elapsed: {elapsed_time:.2f}s")

                # Update the best parameters if the current MSE is lower
                if mse < best_mse:
                    best_mse = mse
                    best_accuracy = accuracy
                    best_lambda = lambd
                    best_eta = eta

        # Store the best parameters for the current activation function and epoch count
        best_params[epoch][activation] = (best_lambda, best_eta, best_mse, best_accuracy)

        # 2D map of MSE
        plot_2D_parameter_lambda_eta(
            lambdas=lambdas,
            etas=etas,
            value=MSE_matrix,
            title=fr'MSE for different combinations of $\lambda$ and $\eta$ with activation {activation} and {epoch} epochs',
            x_log=True,
            y_log=True,
            savefig=save,
            filename=fr'mse_heatmap_{activation}_epochs{epoch}',
            exclude_less_than=0.8
        )

# Best parameters for each activation function for each epoch count
for epoch, activations_dict in best_params.items():
    for activation, (lambd, eta, mse, accuracy) in activations_dict.items():
        print(f"Best combination for activation '{activation}' at {epoch} epochs: "
              f"Lambda = {lambd:.2e}, Learning Rate = {eta:.2e}, MSE = {mse:.4f}, Accuracy = {100*accuracy:.1f}%")
plt.show()