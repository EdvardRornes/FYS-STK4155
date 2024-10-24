import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from utils import FFNN, plot_2D_parameter_lambda_eta, latex_fonts  # Import the FFNN and your plotting function

latex_fonts()

# Load the dataset
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target.reshape(-1, 1), random_state=1)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter ranges
num_Minibatch = np.array([2, 4, 8, 16, 32, 64])
num_Epochs = np.logspace(1, 4, 4).astype(int)
num_Epochs = [1000]
etas = np.logspace(-5, -1, 5)
lambdas = np.logspace(-10, 0, 11)
activations = ['relu', 'lrelu', 'sigmoid']

# Prepare to store results
results = {}
MSE_matrix = np.zeros((len(etas), len(lambdas)))
accuracy_matrix = np.zeros((len(etas), len(lambdas)))

for activation in activations:
    for j, lambd in enumerate(lambdas):
        for i, eta in enumerate(etas):
            # Initialize the model
            model = FFNN(input_size=X_train.shape[1], hidden_layers=[64, 64], output_size=1, activation=activation)
            # Train the model
            print(f"Training with activation: {activation}, lambda: {lambd}, eta: {eta}")
            model.train(X_train, y_train, learning_rate=eta, epochs=num_Epochs[-1], batch_size=32, lambda_reg=lambd)
            # Predictions
            y_pred = model.predict(X_test)
            # Ensure y_pred is a 1D array for binary classification
            y_pred_binary = (y_pred > 0.5).astype(int).flatten()
            # Calculate MSE
            mse = mean_squared_error(y_test, y_pred_binary)
            accuracy = accuracy_score(y_test, y_pred_binary)
            # Store the MSE in the matrix
            MSE_matrix[i, j] = mse
            accuracy_matrix[i, j] = accuracy
            # Store the results
            results[(activation, lambd, eta)] = {
                'mse': mse,
                'accuracy': accuracy_score(y_test, y_pred_binary)
            }
            # Output results
            print(f"Activation: {activation}, Lambda: {lambd:.2e}, Learning Rate: {eta:.2e}, MSE: {mse:.4f}, Accuracy: {accuracy:.3f}")

    # Visualization of results
    plot_2D_parameter_lambda_eta(
        lambdas=lambdas,
        etas=etas,
        MSE=MSE_matrix,
        title=fr'MSE Heatmap for Different $\lambda$ and $\eta$ with activation {activation}',
        x_log=True,
        y_log=True,
        savefig=True,
        filename=fr'mse_heatmap_{activation}'
    )
    # Visualization of results
    plot_2D_parameter_lambda_eta(
        lambdas=lambdas,
        etas=etas,
        MSE=accuracy_matrix,
        title=fr'Accuracy for Different $\lambda$ and $\eta$ with activation {activation}',
        x_log=True,
        y_log=True,
        savefig=True,
        filename=fr'accuracy_heatmap_{activation}',
        Reverse_cmap=True
    )

plt.show()
