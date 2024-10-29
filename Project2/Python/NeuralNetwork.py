from utils import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score
import tensorflow as tf

np.random.seed(0)
latex_fonts()
save = True

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
log_learning_rate_min = -4
log_learning_rate_max = 0
for m in range(log_learning_rate_min, -log_learning_rate_max):
    learning_rates.append(float(10**m))
    learning_rates.append(float(5*10**m))
# learning_rates.append(float(10**log_learning_rate_max))
learning_rates = sorted(learning_rates)

# Initialize dictionary to store MSE and R² for each activation type
mse_results = {"ReLU": [], "Sigmoid": [], "Leaky ReLU": [], "Keras Sigmoid": []}
r2_results = {"ReLU": [], "Sigmoid": [], "Leaky ReLU": [], "Keras Sigmoid": []}

# Loop over each learning rate and activation type
for lr in learning_rates:
    print(f"Testing learning rate: {lr}")
    # FFNN with ReLU
    nn_relu = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='relu')
    mse_history = nn_relu.train(X_train, z_train, epochs=epochs, learning_rate=lr)
    y_pred_relu = nn_relu.predict(X_train)
    mse_results["ReLU"].append(mse_history[-1])
    r2_results["ReLU"].append(r2_score(z_train, y_pred_relu))

    # FFNN with Sigmoid
    nn_sigmoid = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='sigmoid')
    mse_history = nn_sigmoid.train(X_train, z_train, epochs=epochs, learning_rate=lr)
    y_pred_sigmoid = nn_sigmoid.predict(X_train)
    mse_results["Sigmoid"].append(mse_history[-1])
    r2_results["Sigmoid"].append(r2_score(z_train, y_pred_sigmoid))

    # FFNN with Leaky ReLU
    nn_lrelu = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='lrelu')
    mse_history = nn_lrelu.train(X_train, z_train, epochs=epochs, learning_rate=lr)
    y_pred_lrelu = nn_lrelu.predict(X_train)
    mse_results["Leaky ReLU"].append(mse_history[-1])
    r2_results["Leaky ReLU"].append(r2_score(z_train, y_pred_lrelu))

    # Keras implementation with Sigmoid
    model = Sequential()
    model.add(Input(shape=(2,)))  # Specify the input shape here
    model.add(Dense(hidden_layers[0], activation='sigmoid'))
    model.add(Dense(hidden_layers[1], activation='sigmoid'))
    model.add(Dense(1))  # Output layer with linear activation

    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    history = model.fit(X_train, z_train, epochs=epochs, verbose=0)
    y_pred_keras = model.predict(X_train)
    mse_results["Keras Sigmoid"].append(history.history['loss'][-1])
    r2_results["Keras Sigmoid"].append(r2_score(z_train, y_pred_keras))

# Plot MSE and R² as a function of learning rate
fig, axs = plt.subplots(2, 1, figsize=(8, 12))  # Changed to 2 rows, 1 column

# Plot MSE
axs[0].plot(learning_rates, mse_results["ReLU"], label='Custom FFNN MSE (ReLU)', marker='o', color='r')
axs[0].plot(learning_rates, mse_results["Sigmoid"], label='Custom FFNN MSE (Sigmoid)', marker='o', color='g')
axs[0].plot(learning_rates, mse_results["Leaky ReLU"], label='Custom FFNN MSE (Leaky ReLU)', marker='o', color='y')
axs[0].plot(learning_rates, mse_results["Keras Sigmoid"], label='Keras FFNN MSE (Sigmoid)', marker='o', color='purple')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel(r'$\eta$')
axs[0].set_ylabel('MSE')
axs[0].set_title(r'MSE vs $\eta$')
axs[0].legend()
axs[0].grid()

# Plot R²
axs[1].plot(learning_rates, r2_results["ReLU"], label=r'Custom FFNN $R^2$ (ReLU)', marker='o', color='r')
axs[1].plot(learning_rates, r2_results["Sigmoid"], label=r'Custom FFNN $R^2$ (Sigmoid)', marker='o', color='g')
axs[1].plot(learning_rates, r2_results["Leaky ReLU"], label=r'Custom FFNN $R^2$ (Leaky ReLU)', marker='o', color='y')
axs[1].plot(learning_rates, r2_results["Keras Sigmoid"], label=r'Keras FFNN $R^2$ (Sigmoid)', marker='o', color='purple')
axs[1].set_ylim(0, 1)
axs[1].set_xscale('log')
axs[1].set_xlabel(r'$\eta$')
axs[1].set_ylabel(r'$R^2$')
axs[1].set_title(rf'$R^2$ vs $\eta$ with {epochs} epochs')
axs[1].legend()
axs[1].grid()
if save:
    plt.savefig(f'Figures/NN_MSE_R2_Franke_LearningRate_Epochs{epochs}.pdf')

plt.tight_layout()


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
model_keras_best.add(Input(shape=(2,)))  # Specify the input shape here
model_keras_best.add(Dense(hidden_layers[0], activation='sigmoid'))
model_keras_best.add(Dense(hidden_layers[1], activation='sigmoid'))
model_keras_best.add(Dense(1))  # Output layer with linear activation
model_keras_best.compile(optimizer=Adam(learning_rate=best_lr_keras), loss='mean_squared_error')
history_keras_best = model_keras_best.fit(X_train, z_train, epochs=epochs, verbose=0)

# Plot final training losses
plt.figure(figsize=(10, 5))
plt.plot(mse_history_relu_best, label=fr'FFNN ReLU $\eta={best_lr_relu}$')
plt.plot(mse_history_sigmoid_best, label=fr'FFNN Sigmoid $\eta={best_lr_sigmoid}$')
plt.plot(mse_history_lrelu_best, label=fr'FFNN Leaky ReLU $\eta={best_lr_lrelu}$')
plt.plot(history_keras_best.history['loss'], label=fr'Keras FFNN $\eta={best_lr_keras}$')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.xscale('log')
plt.yscale('log')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid()
if save:
    plt.savefig('Figures/NN_MSE_Franke_Epoch.pdf')

# 3D plot of final outputs
# Used ChatGPT to remove redundant copy paste code
fig = plt.figure(figsize=(10, 10))
xx, yy = np.meshgrid(np.linspace(np.min(x), np.max(x), 50), np.linspace(np.min(y), np.max(y), 50))

# Define the activation functions and their corresponding parameters
activation_functions = {
    'ReLU': (nn_relu_best, best_lr_relu, 'red'),
    'Sigmoid': (nn_sigmoid_best, best_lr_sigmoid, 'green'),
    'Leaky ReLU': (nn_lrelu_best, best_lr_lrelu, 'yellow'),
    'Keras': (model_keras_best, best_lr_keras, 'purple')
}

# Iterate over activation functions to create subplots
for i, (name, (model, learning_rate, color)) in enumerate(activation_functions.items()):
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    ax.scatter(x, y, z, color='blue', label='Original Data', s=10)
    
    # Predict the output
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = zz.reshape(xx.shape)
    
    # Plot the surface
    ax.plot_surface(xx, yy, zz, color=color, alpha=0.5)
    
    # Set titles and labels
    ax.set_title(fr'{name} FFNN Output, $\eta={learning_rate}$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'Franke$(x,y)$')

plt.tight_layout()
if save:
    plt.savefig(f'Figures/NN_3D_Predict_Franke_Epochs{epochs}.pdf')
plt.show()