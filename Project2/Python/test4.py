import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from mpl_toolkits.mplot3d import Axes3D

# Assuming FFNN, Franke, and necessary Keras modules are already imported
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class Franke:

    def __init__(self, N:int, eps:float):
        """
        Parameters
            * N:    number of data points 
            * eps:  noise-coefficient         
        """
        self.N = N; self.eps = eps
        self.x = np.random.rand(N)
        self.y = np.random.rand(N)

        self.z_without_noise = self.franke(self.x, self.y)
        self.z = self.z_without_noise + self.eps * np.random.normal(0, 1, self.z_without_noise.shape)

    def franke(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        """
        Parameters
            * x:    x-values
            * y:    y-values

        Returns
            - franke function evaluated at (x,y) 
        """
    
        term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2))
        term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y + 1))
        term3 = 0.5*np.exp(-(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2))
        term4 = -0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
        return term1 + term2 + term3 + term4
    
class FFNN:
    def __init__(self, input_size, hidden_layers, output_size, activation):
        pass
    def train(self, X, y, epochs, learning_rate):
        return np.random.rand(epochs)  # Dummy MSE history
    def predict(self, X):
        return np.random.rand(X.shape[0], 1)  # Dummy predictions

import numpy as np
import os
import pickle

class FFNNAnalyzer:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', alpha=0.01):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        self.activation_func = activation
        self.alpha = alpha  # For LeakyReLU

        for i in range(len(self.layers) - 1):
            weight_matrix = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2. / self.layers[i])  # He initialization
            self.weights.append(weight_matrix)
            bias_vector = np.zeros((1, self.layers[i + 1]))
            self.biases.append(bias_vector)

    def relu(self, z):
        return np.where(z > 0, z, 0)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)
    
    def Lrelu(self, z):
        return np.where(z > 0, z, self.alpha * z)

    def Lrelu_derivative(self, z):
        return np.where(z > 0, 1, self.alpha)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        A = X
        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i] + self.biases[i]
            self.z_values.append(Z)
            if self.activation_func.lower() == 'relu':
                A = self.relu(Z)
            elif self.activation_func.lower() == 'sigmoid':
                A = self.sigmoid(Z)
            elif self.activation_func.lower() == 'lrelu':
                A = self.Lrelu(Z)
            self.activations.append(A)

        Z = A @ self.weights[-1] + self.biases[-1]
        self.z_values.append(Z)
        self.activations.append(Z)
        return Z

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        y = y.reshape(-1, 1)

        delta = self.activations[-1] - y
        for i in reversed(range(len(self.weights))):
            dw = (self.activations[i].T @ delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
            if i > 0:
                if self.activation_func.lower() == 'relu':
                    delta = (delta @ self.weights[i].T) * self.relu_derivative(self.z_values[i - 1])
                elif self.activation_func.lower() == 'sigmoid':
                    delta = (delta @ self.weights[i].T) * self.sigmoid_derivative(self.z_values[i - 1])
                elif self.activation_func.lower() == 'lrelu':
                    delta = (delta @ self.weights[i].T) * self.Lrelu_derivative(self.z_values[i - 1])

    def train(self, X, y, learning_rate=0.01, epochs=1000, batch_size=None, shuffle=True):
        mse_history = []
        m = X.shape[0]
        
        if batch_size is None:  # Default to full-batch (i.e., all data at once)
            batch_size = m

        for epoch in range(epochs):
            if shuffle:
                indices = np.random.permutation(m)
                X, y = X[indices], y[indices]

            for i in range(0, m, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)

            mse = np.mean((self.activations[-1] - y_batch) ** 2)
            
            if np.isnan(mse) or mse > 100:
                mse = 1e10
                print(f'Epoch {epoch}, MSE: {mse} (Issue encountered, breaking)')
                break
            
            mse_history.append(mse)
            if epoch % max(1, (epochs // 5)) == 0:
                print(f'Epoch {epoch}, MSE: {mse}')
        
        return mse_history

    def predict(self, X):
        return self.forward(X)

    def save_model(self, filename, overwrite=False):
        """Saves the model parameters to a file.

        Args:
            filename (str): The name of the file to save the model to.
            overwrite (bool): Whether to overwrite the file if it exists.
        """
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f"The file '{filename}' already exists. Use overwrite=True to overwrite it.")

        data = {
            'layers': self.layers,
            'weights': self.weights,
            'biases': self.biases,
            'activation_func': self.activation_func,
            'alpha': self.alpha
        }

        # Convert numpy arrays to lists for pickling
        data['weights'] = [w.tolist() for w in self.weights]
        data['biases'] = [b.tolist() for b in self.biases]

        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to '{filename}'.")

    def load_model(self, filename):
        """Loads model parameters from a file.

        Args:
            filename (str): The name of the file to load the model from.
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        self.layers = data['layers']
        self.activation_func = data['activation_func']
        self.alpha = data['alpha']

        # Convert lists back to numpy arrays
        self.weights = [np.array(w) for w in data['weights']]
        self.biases = [np.array(b) for b in data['biases']]

        print(f"Model loaded from '{filename}'.")



# Initialize the analyzer
analyzer = FFNNAnalyzer()

# Run the analysis
analyzer.run_analysis()

# Save the data to a file, with overwrite option
analyzer.save_data('analysis_results.pkl', overwrite=True)

# Later on, or in a different script, load the data
analyzer.load_data('analysis_results.pkl')

# Plot MSE vs Learning Rate
analyzer.plot_mse_vs_learning_rate()

# Plot 3D outputs of the trained networks
analyzer.plot_3d_outputs()
