from new_utils import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

np.random.seed(0)

# Sample data
N = 250; eps = 0.0
franke = Franke(N, eps)
x = franke.x; y = franke.y; z = franke.z
epochs = 100
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

# Initialize dictionary to store MSE for each activation type
mse_results = {"ReLU": [], "Sigmoid": [], "Leaky ReLU": [], "Keras Sigmoid": []}

# Loop over each learning rate and activation type
for lr in learning_rates:
    print(f"Testing learning rate: {lr}")

    # FFNN with ReLU
    nn_relu = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='relu')
    mse_history = nn_relu.train(X_train, z_train, epochs=epochs, learning_rate=lr)
    mse_results["ReLU"].append(mse_history[-1])

    # FFNN with Sigmoid
    nn_sigmoid = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='sigmoid')
    mse_history = nn_sigmoid.train(X_train, z_train, epochs=epochs, learning_rate=lr)
    mse_results["Sigmoid"].append(mse_history[-1])

    # FFNN with Leaky ReLU
    nn_lrelu = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='lrelu')
    mse_history = nn_lrelu.train(X_train, z_train, epochs=epochs, learning_rate=lr)
    mse_results["Leaky ReLU"].append(mse_history[-1])

    # Keras implementation with Sigmoid
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation='sigmoid', input_shape=(2,)))
    model.add(Dense(hidden_layers[1], activation='sigmoid'))
    model.add(Dense(1))  # Output layer with linear activation

    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    history = model.fit(X_train, z_train, epochs=epochs, verbose=0)
    mse_results["Keras Sigmoid"].append(history.history['loss'][-1])

# Plot MSE as a function of learning rate
plt.figure(figsize=(10, 5))
plt.plot(learning_rates, mse_results["ReLU"], label='Custom FFNN MSE (ReLU)', marker='o', color='r')
plt.plot(learning_rates, mse_results["Sigmoid"], label='Custom FFNN MSE (Sigmoid)', marker='o', color='g')
plt.plot(learning_rates, mse_results["Leaky ReLU"], label='Custom FFNN MSE (Leaky ReLU)', marker='o', color='y')
plt.plot(learning_rates, mse_results["Keras Sigmoid"], label='Keras FFNN MSE (Sigmoid)', marker='o', color='purple')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Final Mean Squared Error')
plt.title('MSE vs Learning Rate')
plt.legend()
plt.grid()

# Find the best learning rates
best_lr_relu = learning_rates[np.argmin(mse_results["ReLU"])]
best_lr_sigmoid = learning_rates[np.argmin(mse_results["Sigmoid"])]
best_lr_lrelu = learning_rates[np.argmin(mse_results["Leaky ReLU"])]
best_lr_keras = learning_rates[np.argmin(mse_results["Keras Sigmoid"])]

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
plt.plot(mse_history_relu_best, label=fr'FFNN ReLU $\eta={best_lr_relu}$')
plt.plot(mse_history_sigmoid_best, label=fr'FFNN Sigmoid $\eta={best_lr_sigmoid}$')
plt.plot(mse_history_lrelu_best, label=fr'FFNN Leaky ReLU $\eta={best_lr_lrelu}$')
plt.plot(history_keras_best.history['loss'], label=fr'Keras FFNN $\eta={best_lr_keras}$')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.xscale('log')
plt.yscale('log')
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