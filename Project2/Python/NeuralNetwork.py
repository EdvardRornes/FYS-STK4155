from utils import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score

np.random.seed(0)
latex_fonts()
save = True

# Sample data
N = 100; eps = 0.0
franke = Franke(N, eps)
x = franke.x; y = franke.y; z = franke.z
epochs = 1000
hidden_layers = [10, 20]

X_train = np.c_[x, y]
z_train = z.reshape(-1, 1)

# Learning rates
learning_rates = []
log_learning_rate_min = -4
log_learning_rate_max = 0
for m in range(log_learning_rate_min, log_learning_rate_max):
    learning_rates.append(float(10**m))
    learning_rates.append(float(3*10**m))
learning_rates = sorted(learning_rates)

# Prepare to store MSE and R2 for each activation type
mse_results = {"ReLU": [], "Sigmoid": [], "Leaky ReLU": [], "Keras Sigmoid": []}
r2_results = {"ReLU": [], "Sigmoid": [], "Leaky ReLU": [], "Keras Sigmoid": []}

# Initialize FFNN
ffnn_relu = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='relu')
ffnn_sigmoid = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='sigmoid')
ffnn_lrelu = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='lrelu')

# Keras model initialization
model_keras = Sequential()
model_keras.add(Input(shape=(2,)))  # Specify the input shape here
model_keras.add(Dense(hidden_layers[0], activation='sigmoid'))
model_keras.add(Dense(hidden_layers[1], activation='sigmoid'))
model_keras.add(Dense(1))  # Output layer with linear activation

# Loop over each learning rate and activation type
for lr in learning_rates:
    print(f"Testing learning rate: {lr:.1e}")

    # FFNN with ReLU
    mse_history_relu = ffnn_relu.train(X_train, z_train, epochs=epochs, learning_rate=lr)
    y_pred_relu = ffnn_relu.predict(X_train)
    mse_results["ReLU"].append(mse_history_relu[-1])
    r2_results["ReLU"].append(r2_score(z_train, y_pred_relu))

    # FFNN with Sigmoid
    mse_history_sigmoid = ffnn_sigmoid.train(X_train, z_train, epochs=epochs, learning_rate=lr)
    y_pred_sigmoid = ffnn_sigmoid.predict(X_train)
    mse_results["Sigmoid"].append(mse_history_sigmoid[-1])
    r2_results["Sigmoid"].append(r2_score(z_train, y_pred_sigmoid))

    # FFNN with Leaky ReLU
    mse_history_lrelu = ffnn_lrelu.train(X_train, z_train, epochs=epochs, learning_rate=lr)
    y_pred_lrelu = ffnn_lrelu.predict(X_train)
    mse_results["Leaky ReLU"].append(mse_history_lrelu[-1])
    r2_results["Leaky ReLU"].append(r2_score(z_train, y_pred_lrelu))

    # Keras implementation with Sigmoid
    print('Keras is slow :(')
    model_keras.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    history = model_keras.fit(X_train, z_train, epochs=epochs, verbose=0)
    y_pred_keras = model_keras.predict(X_train)
    mse_results["Keras Sigmoid"].append(history.history['loss'][-1])
    r2_results["Keras Sigmoid"].append(r2_score(z_train, y_pred_keras))

# Plot MSE and R2 as a function of learning rate
fig, axs = plt.subplots(2, 1, figsize=(8, 12))  # Changed to 2 rows, 1 column
fig.subplots_adjust(hspace=0.4)

# Plot MSE
axs[0].plot(learning_rates, mse_results["ReLU"], label='Own FFNN (ReLU)', marker='o', color='r')
axs[0].plot(learning_rates, mse_results["Sigmoid"], label='Own FFNN (Sigmoid)', marker='o', color='g')
axs[0].plot(learning_rates, mse_results["Leaky ReLU"], label='Own FFNN (Leaky ReLU)', marker='o', color='y')
axs[0].plot(learning_rates, mse_results["Keras Sigmoid"], label='Keras FFNN (Sigmoid)', marker='o', color='purple')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel(r'$\eta$')
axs[0].set_ylabel('MSE')
axs[0].set_title(fr'MSE vs $\eta$ with {epochs} epochs')
axs[0].legend()
axs[0].grid()

# Plot R²
axs[1].plot(learning_rates, r2_results["ReLU"], label=r'Own FFNN (ReLU)', marker='o', color='r')
axs[1].plot(learning_rates, r2_results["Sigmoid"], label=r'Own FFNN (Sigmoid)', marker='o', color='g')
axs[1].plot(learning_rates, r2_results["Leaky ReLU"], label=r'Own FFNN (Leaky ReLU)', marker='o', color='y')
axs[1].plot(learning_rates, r2_results["Keras Sigmoid"], label=r'Keras FFNN (Sigmoid)', marker='o', color='purple')
axs[1].set_ylim(-1, 1)
axs[1].set_xscale('log')
axs[1].set_xlabel(r'$\eta$')
axs[1].set_ylabel(r'$R^2$')
axs[1].set_title(rf'$R^2$ vs $\eta$ with {epochs} epochs')
axs[1].legend()
axs[1].grid()

if save:
    plt.savefig(f'Figures/NN_MSE_R2_Franke_LearningRate_Epochs{epochs}.pdf')

# Find the best learning rates
best_lr_relu = learning_rates[np.argmin(mse_results["ReLU"])]
best_lr_sigmoid = learning_rates[np.argmin(mse_results["Sigmoid"])]
best_lr_lrelu = learning_rates[np.argmin(mse_results["Leaky ReLU"])]
best_lr_keras = learning_rates[np.argmin(mse_results["Keras Sigmoid"])]

# Train the networks with the best learning rates
# Reset FFNN instances for best learning rate training
ffnn_relu_best = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='relu')
mse_history_relu_best = ffnn_relu_best.train(X_train, z_train, epochs=epochs, learning_rate=best_lr_relu)

ffnn_sigmoid_best = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='sigmoid')
mse_history_sigmoid_best = ffnn_sigmoid_best.train(X_train, z_train, epochs=epochs, learning_rate=best_lr_sigmoid)

ffnn_lrelu_best = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='lrelu')
mse_history_lrelu_best = ffnn_lrelu_best.train(X_train, z_train, epochs=epochs, learning_rate=best_lr_lrelu)

# Compile Keras model with the best learning rate
model_keras_best = Sequential()
model_keras_best.add(Input(shape=(2,)))  # Specify the input shape here
model_keras_best.add(Dense(hidden_layers[0], activation='sigmoid'))
model_keras_best.add(Dense(hidden_layers[1], activation='sigmoid'))
model_keras_best.add(Dense(1))  # Output layer with linear activation
model_keras_best.compile(optimizer=Adam(learning_rate=best_lr_keras), loss='mean_squared_error')
history_keras_best = model_keras_best.fit(X_train, z_train, epochs=epochs, verbose=0)

# Plot final training losses
plt.figure(figsize=(10, 8))
plt.plot(mse_history_relu_best, label=fr'FFNN ReLU' + '\n' + fr'$\eta={best_lr_relu:.1e}$', color='r')
plt.plot(mse_history_sigmoid_best, label=fr'FFNN Sigmoid' + '\n' + fr'$\eta={best_lr_sigmoid:.1e}$', color='g')
plt.plot(mse_history_lrelu_best, label=fr'FFNN Leaky ReLU' + '\n' + fr'$\eta={best_lr_lrelu:.1e}$', color='y')
plt.plot(history_keras_best.history['loss'], label=fr'Keras FFNN' + '\n' + fr'$\eta={best_lr_keras:.1e}$', color='purple')
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
    'ReLU': (ffnn_relu_best, best_lr_relu, 'red'),
    'Sigmoid': (ffnn_sigmoid_best, best_lr_sigmoid, 'green'),
    'Leaky ReLU': (ffnn_lrelu_best, best_lr_lrelu, 'yellow'),
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
    ax.set_title(fr'{name} FFNN Output' + '\n' fr'$\eta={learning_rate:.1e}$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'Franke$(x,y)$')

plt.tight_layout()
if save:
    plt.savefig(f'Figures/NN_3D_Predict_Franke_Epochs{epochs}.pdf')
plt.show()