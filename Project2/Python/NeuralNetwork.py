from utils import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class FFNN:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', alpha=0.01):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        self.activation_func = activation
        self.alpha = alpha

        for i in range(len(self.layers) - 1):
            weight_matrix = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2. / self.layers[i])
            self.weights.append(weight_matrix)
            bias_vector = np.zeros((1, self.layers[i + 1]))
            self.biases.append(bias_vector)

    def relu(self, z):
        return np.where(z > 0, z, 0)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)
    
    def Lrelu(self, z):
        return np.where(z > 0, z, self.alpha*z)

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
            A = self.relu(Z) if self.activation_func.lower() == 'relu' else self.sigmoid(Z) if self.activation_func.lower() == 'sigmoid' else self.Lrelu(Z)
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
                delta = (delta @ self.weights[i].T) * (self.relu_derivative(self.z_values[i - 1]) if self.activation_func.lower() == 'relu' else self.sigmoid_derivative(self.z_values[i - 1]) if self.activation_func.lower() == 'sigmoid' else self.Lrelu_derivative(self.z_values[i - 1]))

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
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)

            mse = np.mean((self.activations[-1] - y_batch) ** 2)

            if np.isnan(mse) or mse > 100:
                mse = 1e10
                print(f'Epoch {epoch}, MSE ({self.activation_func}): {mse} (Issue encountered, breaking)')
                break
            
            mse_history.append(mse)

            # Print MSE every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, MSE ({self.activation_func}): {mse}')
        
        return mse_history

    def predict(self, X):
        return self.forward(X)

# Sample data
N = 250; eps = 0.0
franke = Franke(N, eps)
x = franke.x; y = franke.y; z = franke.z
epochs = 1000
hidden_layers = [10, 20]

# Create feature matrix X
X_train = np.c_[x, y]
z_train = z.reshape(-1, 1)

# Learning rates
learning_rates = []
for m in range(-3, 0):
    learning_rates.append(float(10**m))
    learning_rates.append(float(5*10**m))
learning_rates = sorted(learning_rates)

# Store MSE for different learning rates
mse_relu = []
mse_sigmoid = []
mse_lrelu = []
mse_keras = []

# Testing learning rates for FFNN ReLU
for lr in learning_rates:
    print('ReLU learning rate:', lr)
    nn_relu = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='relu')
    mse_history = nn_relu.train(X_train, z_train, epochs=epochs, learning_rate=lr)
    mse_relu.append(mse_history[-1])  # Store final MSE

# Testing learning rates for FFNN Sigmoid
for lr in learning_rates:
    print('Sigmoid learning rate:', lr)
    nn_sigmoid = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='sigmoid')
    mse_history = nn_sigmoid.train(X_train, z_train, epochs=epochs, learning_rate=lr)
    mse_sigmoid.append(mse_history[-1])  # Store final MSE

# Testing learning rates for FFNN Leaky ReLU
for lr in learning_rates:
    print('Leaky ReLU learning rate:', lr)
    nn_lrelu = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='lrelu')
    mse_history = nn_lrelu.train(X_train, z_train, epochs=epochs, learning_rate=lr)
    mse_lrelu.append(mse_history[-1])  # Store final MSE

# Keras implementation with Sigmoid for different learning rates
for lr, i in zip(learning_rates, range(len(learning_rates))):
    print(f'{i + 1}/{len(learning_rates)} Keras learning rate:', lr)
    model = Sequential()
    model.add(Dense(64, activation='sigmoid', input_shape=(2,)))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(1))  # Output layer with linear activation

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    # Train the model and store the history
    history = model.fit(X_train, z_train, epochs=epochs, verbose=0)
    mse_keras.append(history.history['loss'][-1])  # Store final MSE

# Plot MSE as a function of learning rate
plt.figure(figsize=(10, 5))
plt.plot(learning_rates, mse_relu, label='Custom FFNN MSE (ReLU)', marker='o', color='r')
plt.plot(learning_rates, mse_sigmoid, label='Custom FFNN MSE (Sigmoid)', marker='o', color='g')
plt.plot(learning_rates, mse_lrelu, label='Custom FFNN MSE (Leaky ReLU)', marker='o', color='y')
plt.plot(learning_rates, mse_keras, label='Keras FFNN MSE (Sigmoid)', marker='o', color='purple')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Final Mean Squared Error')
plt.title('MSE vs Learning Rate')
plt.legend()
plt.grid()

# Find the best learning rates
best_lr_relu = learning_rates[np.argmin(mse_relu)]
best_lr_sigmoid = learning_rates[np.argmin(mse_sigmoid)]
best_lr_lrelu = learning_rates[np.argmin(mse_lrelu)]
best_lr_keras = learning_rates[np.argmin(mse_keras)]

# Train the networks with the best learning rates
nn_relu_best = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='relu')
mse_history_relu_best = nn_relu_best.train(X_train, z_train, epochs=epochs, learning_rate=best_lr_relu)

nn_sigmoid_best = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='sigmoid')
mse_history_sigmoid_best = nn_sigmoid_best.train(X_train, z_train, epochs=epochs, learning_rate=best_lr_sigmoid)

nn_lrelu_best = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='lrelu')
mse_history_lrelu_best = nn_lrelu_best.train(X_train, z_train, epochs=epochs, learning_rate=best_lr_lrelu)

model_keras_best = Sequential()
model_keras_best.add(Dense(64, activation='sigmoid', input_shape=(2,)))
model_keras_best.add(Dense(64, activation='sigmoid'))
model_keras_best.add(Dense(1))
model_keras_best.compile(optimizer=Adam(learning_rate=best_lr_keras), loss='mean_squared_error')
history_keras_best = model_keras_best.fit(X_train, z_train, epochs=epochs, verbose=0)

# Plot final training losses
plt.figure(figsize=(10, 5))
plt.plot(mse_history_relu_best, label=f'FFNN ReLU $\eta={best_lr_relu}$')
plt.plot(mse_history_sigmoid_best, label=f'FFNN Sigmoid $\eta={best_lr_sigmoid}$')
plt.plot(mse_history_lrelu_best, label=f'FFNN Leaky ReLU $\eta={best_lr_lrelu}$')
plt.plot(history_keras_best.history['loss'], label=f'Keras FFNN $\eta={best_lr_keras}$')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid()

# 3D plot of final outputs
fig = plt.figure(figsize=(10, 10))
xx, yy = np.meshgrid(np.linspace(np.min(x), np.max(x), 50), np.linspace(np.min(y), np.max(y), 50))

# ReLU
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(x, y, z, color='blue', label='Original Data', s=10)
zz_relu = nn_relu_best.predict(np.c_[xx.ravel(), yy.ravel()])
zz_relu = zz_relu.reshape(xx.shape)
ax1.plot_surface(xx, yy, zz_relu, color='red', alpha=0.5)
ax1.set_title(fr'ReLU FFNN Output, $\eta={best_lr_relu}$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# Sigmoid
ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(x, y, z, color='blue', label='Original Data', s=10)
zz_sigmoid = nn_sigmoid_best.predict(np.c_[xx.ravel(), yy.ravel()])
zz_sigmoid = zz_sigmoid.reshape(xx.shape)
ax2.plot_surface(xx, yy, zz_sigmoid, color='green', alpha=0.5)
ax2.set_title(fr'Sigmoid FFNN Output, $\eta={best_lr_sigmoid}$')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

# Leaky ReLU
ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(x, y, z, color='blue', label='Original Data', s=10)
zz_lrelu = nn_lrelu_best.predict(np.c_[xx.ravel(), yy.ravel()])
zz_lrelu = zz_lrelu.reshape(xx.shape)
ax3.plot_surface(xx, yy, zz_lrelu, color='yellow', alpha=0.5)
ax3.set_title(fr'Leaky ReLU FFNN Output, $\eta={best_lr_lrelu}$')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')

# Keras
ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(x, y, z, color='blue', label='Original Data', s=10)
zz_keras = model_keras_best.predict(np.c_[xx.ravel(), yy.ravel()])
zz_keras = zz_keras.reshape(xx.shape)
ax4.plot_surface(xx, yy, zz_keras, color='purple', alpha=0.5)
ax4.set_title(fr'Keras FFNN Output, $\eta={best_lr_keras}$')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_zlabel('z')

plt.tight_layout()

plt.show()
