import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils import *
import time
import warnings

# Due to using autograd instead of numpy
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")


np.random.seed(1)

latex_fonts()
save = True

# Load the dataset
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target.reshape(-1, 1), random_state=42, test_size=0.25)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter ranges
hidden_layers = [15, 30, 15, 8, 4, 2]
epochs = [100]
batchsize = 50
# Tighter logspace needed for etas, nicer values for plotting.
eta_lower = -4
eta_upper = -1
etas = []
for m in range(eta_lower, eta_upper):
    etas.append(float(10**m))
    etas.append(float(2.5*10**m))
    etas.append(float(5*10**m))
    etas.append(float(7.5*10**m))
etas.append(float(10**eta_upper))

lambdas = np.logspace(-10, 0, 11)

activations = ['relu', 'lrelu', 'sigmoid']

# Prepare to store results
results = {}
accuracy_matrix = np.zeros((len(etas), len(lambdas)))
best_params = {epoch: {} for epoch in epochs}

# Timer
start_time = time.time()
total_iterations = len(epochs)*len(activations)*len(lambdas)*len(etas)
completed_iterations = 0

best_predictions = {}

for epoch in epochs:
    for activation in activations:
        # Initialize variables to track the best parameters and predictions
        best_mse = float('inf')
        best_accuracy = 0
        best_lambda = None
        best_eta = None
        best_y_pred_binary = None  # Variable to save best prediction

        for j, lambd in enumerate(lambdas):
            for i, eta in enumerate(etas):
                # Suppress print from class to avoid clutter
                blockPrint()
                # Initialize the model using cross entropy as loss function
                model = FFNN(input_size=X_train.shape[1], hidden_layers=hidden_layers, output_size=1, activation=activation, lambda_reg=lambd, loss_function='bce')
                # Train the model
                print(f"Training with activation: {activation}, lambda: {lambd}, eta: {eta}")
                model.train(X_train, y_train, learning_rate=eta, epochs=epoch, batch_size=batchsize, lambda_reg=lambd)
                # Predictions
                y_pred = model.predict(X_test)
                # Ensure y_pred is a 1D array for binary classification
                y_pred_binary = (y_pred > 0.5).astype(int).flatten()
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred_binary)
                # Store the MSE in the matrix
                accuracy_matrix[i, j] = accuracy
                # Store the results
                results[(activation, lambd, eta)] = {'accuracy': accuracy}
                enablePrint()

                # Calculate progress percentage
                completed_iterations += 1
                percentage_progress = (completed_iterations / total_iterations) * 100
                # Elapsed time
                elapsed_time = time.time() - start_time
                ETA = elapsed_time*(100/percentage_progress-1)
                print(f"Activation: {activation}, Epochs: {epoch}, Lambda: {lambd:.1e}, Learning Rate: {eta:.1e}, "
                      f"Accuracy: {100*accuracy:.1f}% | Progress: {completed_iterations}/{total_iterations} ({percentage_progress:.1f}%), Time Elapsed: {elapsed_time:.1f}s, ETA: {ETA:.1f}s")

                # Check if current model has the best accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_lambda = lambd
                    best_eta = eta
                    best_y_pred_binary = y_pred_binary

        # Save the best parameters and predictions for each activation function
        best_params[epoch][activation] = (best_lambda, best_eta, best_mse, best_accuracy)
        best_predictions[activation] = (best_y_pred_binary, y_test)

        # 2D map of MSE
        plot_2D_parameter_lambda_eta(
            lambdas=lambdas,
            etas=etas,
            value=accuracy_matrix,
            title=fr'Accuracy for FFNN with activation {activation}, {epoch} epochs and {batchsize} batch size',
            x_log=True,
            y_log=True,
            savefig=save,
            filename=fr'Cancer_Accuracy_Heatmap_{activation}_Epochs{epoch}_randomstate42',
            Reverse_cmap=True
        )

# Best parameters for each activation function for each epoch count
# for epoch, activations_dict in best_params.items():
#     for activation, (lambd, eta, accuracy) in activations_dict.items():
#         print(f"Best combination for activation '{activation}' at {epoch} epochs: "
#               f"Lambda = {lambd:.2e}, Learning Rate = {eta:.2e}, Accuracy = {100*accuracy:.1f}%")


# Plotting the confusion matrix for the best predictions
for activation in activations:
    y_pred_binary, y_test = best_predictions[activation]
    cm = confusion_matrix(y_test, y_pred_binary)
    
    # Extract values for true negatives, false positives, false negatives, and true positives
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate percentages
    total = tn + fp + fn + tp
    tn_percent = (tn / total) * 100
    fp_percent = (fp / total) * 100
    fn_percent = (fn / total) * 100
    tp_percent = (tp / total) * 100
    
    # Create an annotated matrix with both counts and percentages
    annotated_cm = np.array([
        [fr"{tn_percent:.2f}\%", fr"{fp_percent:.2f}\%"],
        [fr"{fn_percent:.2f}\%", fr"{tp_percent:.2f}\%"]
    ])
    
    # Plot the confusion matrix with annotated cells
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=annotated_cm, fmt='', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"FFNN Confusion Matrix: {activation} Activation\n"+fr"$\lambda={best_params[epoch][activation][0]:.1e}, \eta={best_params[epoch][activation][1]:.1e}$")
    if save:
        plt.savefig(f'../Figures/ConfusionMatrixFFNN_{activation}_Epochs{epoch}_randomstate42.pdf')
plt.show()

