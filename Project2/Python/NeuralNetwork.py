from utils import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata


np.random.seed(0)
latex_fonts()
save = True

# Sample data
N = 250; eps = 0.01
franke = Franke(N, eps)
x = franke.x; y = franke.y; z = franke.z
batch_size = 50
epochs = 1000
hidden_layers = [4, 8, 16, 32, 16, 8, 4, 2]

X = np.c_[x, y]
z = minmax_scale(z)
z_reshaped = z.reshape(-1, 1)
X_train, X_test, z_train, z_test = train_test_split(X, z_reshaped, test_size=0.25)

# Learning rates
learning_rates = []
log_learning_rate_min = -5
log_learning_rate_max = -2
for m in range(log_learning_rate_min, log_learning_rate_max):
    learning_rates.append(float(10**m))
    learning_rates.append(float(2*10**m))
    learning_rates.append(float(5*10**m))

# Prepare to store MSE and R2 for each activation type
mse_results = {"ReLU": [], "Sigmoid": [], "Leaky ReLU": [], "Keras Leaky ReLU": []}
r2_results = {"ReLU": [], "Sigmoid": [], "Leaky ReLU": [], "Keras Leaky ReLU": []}

# Initialize FFNN
ffnn_relu = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='relu')
ffnn_sigmoid = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='sigmoid')
ffnn_lrelu = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='lrelu')

# Keras model initialization
model_keras = Sequential()
model_keras.add(Input(shape=(2,)))
for layers in hidden_layers:
    model_keras.add(Dense(layers))
    model_keras.add(LeakyReLU(alpha=0.01))
model_keras.add(Dense(1))


test_MSE_relu_best = np.inf
test_MSE_sigmoid_best = np.inf
test_MSE_lrelu_best = np.inf
test_MSE_keras_best = np.inf

mse_history_relu_best = []
mse_history_sigmoid_best = []
mse_history_lrelu_best = []
mse_hisory_keras_best = []

# Loop over each learning rate and activation type
for lr in learning_rates:
    print(f"Testing learning rate: {lr:.1e}")

    # FFNN with ReLU
    mse_history_relu = ffnn_relu.train(X_train, z_train, epochs=epochs, batch_size=batch_size, learning_rate=lr)
    z_pred_relu = ffnn_relu.predict(X_test)
    test_MSE_relu = mean_squared_error(z_pred_relu, z_test)
    mse_results["ReLU"].append(test_MSE_relu)
    r2_results["ReLU"].append(r2_score(z_test, z_pred_relu))
    if test_MSE_relu < test_MSE_relu_best:
        test_MSE_relu_best = test_MSE_relu
        mse_history_relu_best = mse_history_relu
        z_pred_relu_best = z_pred_relu
        best_lr_relu = lr

    # FFNN with Sigmoid
    mse_history_sigmoid = ffnn_sigmoid.train(X_train, z_train, epochs=epochs, batch_size=batch_size, learning_rate=lr)
    z_pred_sigmoid = ffnn_sigmoid.predict(X_test)
    test_MSE_sigmoid = mean_squared_error(z_pred_sigmoid, z_test)
    mse_results["Sigmoid"].append(test_MSE_sigmoid)
    r2_results["Sigmoid"].append(r2_score(z_test, z_pred_sigmoid))
    if test_MSE_sigmoid < test_MSE_sigmoid_best:
        test_MSE_sigmoid_best = test_MSE_sigmoid
        mse_history_sigmoid_best = mse_history_sigmoid
        z_pred_sigmoid_best = z_pred_sigmoid
        best_lr_sigmoid = lr

    # FFNN with Leaky ReLU
    mse_history_lrelu = ffnn_lrelu.train(X_train, z_train, epochs=epochs, batch_size=batch_size, learning_rate=lr)
    z_pred_lrelu = ffnn_lrelu.predict(X_test)
    test_MSE_lrelu = mean_squared_error(z_pred_lrelu, z_test)
    mse_results["Leaky ReLU"].append(test_MSE_lrelu)
    r2_results["Leaky ReLU"].append(r2_score(z_test, z_pred_lrelu))
    if test_MSE_lrelu < test_MSE_lrelu_best:
        test_MSE_relu_lbest = test_MSE_lrelu
        mse_history_lrelu_best = mse_history_lrelu
        z_pred_lrelu_best = z_pred_lrelu
        best_lr_lrelu = lr

    # Keras implementation with LReLU
    print('Keras is slow :(')
    model_keras.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    mse_hisory_keras = model_keras.fit(X_train, z_train, epochs=epochs, batch_size=batch_size, verbose=0)
    z_pred_keras = model_keras.predict(X_test)
    test_MSE_keras = mean_squared_error(z_pred_keras, z_test)
    mse_results["Keras Leaky ReLU"].append(test_MSE_keras)
    r2_results["Keras Leaky ReLU"].append(r2_score(z_test, z_pred_keras))
    if test_MSE_keras < test_MSE_keras_best:
        test_MSE_keras_best = test_MSE_keras
        mse_hisory_keras_best = mse_hisory_keras
        z_pred_keras_best = z_pred_keras
        best_lr_keras = lr
    print(f'Keras mse: {test_MSE_keras:.3f}')

# Plot MSE and R2 as a function of learning rate
fig, axs = plt.subplots(2, 1, figsize=(8, 12))  # Changed to 2 rows, 1 column
fig.subplots_adjust(hspace=0.4)

# Plot MSE
axs[0].plot(learning_rates, mse_results["ReLU"], label='Own FFNN (ReLU)', marker='o', color='r')
axs[0].plot(learning_rates, mse_results["Sigmoid"], label='Own FFNN (Sigmoid)', marker='o', color='g')
axs[0].plot(learning_rates, mse_results["Leaky ReLU"], label='Own FFNN (LReLU)', marker='o', color='y')
axs[0].plot(learning_rates, mse_results["Keras Leaky ReLU"], label='Keras FFNN (LReLU)', marker='o', color='purple')
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
axs[1].plot(learning_rates, r2_results["Leaky ReLU"], label=r'Own FFNN (LReLU)', marker='o', color='y')
axs[1].plot(learning_rates, r2_results["Keras Leaky ReLU"], label=r'Keras FFNN (LReLU)', marker='o', color='purple')
axs[1].set_ylim(0, 1)
axs[1].set_xscale('log')
axs[1].set_xlabel(r'$\eta$')
axs[1].set_ylabel(r'$R^2$')
axs[1].set_title(rf'$R^2$ vs $\eta$ with {epochs} epochs')
axs[1].legend()
axs[1].grid()

if save:
    plt.savefig(f'../Figures/NN_MSE_R2_Franke_LearningRate_Epochs{epochs}.pdf')

# Plot final training losses
plt.figure(figsize=(10, 8))
plt.plot(mse_history_relu_best, label=fr'FFNN ReLU' + '\n' + fr'$\eta={best_lr_relu:.1e}$', color='r')
plt.plot(mse_history_sigmoid_best, label=fr'FFNN Sigmoid' + '\n' + fr'$\eta={best_lr_sigmoid:.1e}$', color='g')
plt.plot(mse_history_lrelu_best, label=fr'FFNN Leaky ReLU' + '\n' + fr'$\eta={best_lr_lrelu:.1e}$', color='y')
plt.plot(mse_hisory_keras_best.history['loss'], label=fr'Keras FFNN' + '\n' + fr'$\eta={best_lr_keras:.1e}$', color='purple')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.xscale('log')
plt.yscale('log')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid()
if save:
    plt.savefig('../Figures/NN_MSE_Franke_Epoch.pdf')

# 3D plot of final outputs
# Used ChatGPT to remove redundant copy paste code
fig = plt.figure(figsize=(10, 10))


# Define the activation functions and their corresponding parameters
activation_functions = {
    'ReLU': (z_pred_relu_best, best_lr_relu, 'red'),
    'Sigmoid': (z_pred_sigmoid_best, best_lr_sigmoid, 'green'),
    'Leaky ReLU': (z_pred_lrelu_best, best_lr_lrelu, 'yellow'),
    'Keras': (z_pred_keras_best, best_lr_keras, 'purple')
}

# Iterate over activation functions to create subplots
for i, (name, (z_pred_best, learning_rate, color)) in enumerate(activation_functions.items()):
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    ax.scatter(x, y, z, color='blue', label='Original Data', s=10)
    
    zz = z_pred_best
    x_grid, y_grid = np.meshgrid(np.unique(x), np.unique(y))
    # Interpolate for plotting
    z_grid_pred = griddata((X_test[:,0], X_test[:,1]), z_pred_best.flatten(), (x_grid, y_grid), method='cubic')
    
    # Plot the surface
    ax.plot_surface(x_grid, y_grid, z_grid_pred, alpha=0.5, color=color)
    
    # Set titles and labels
    ax.set_title(fr'{name} FFNN Output' + '\n' fr'$\eta={learning_rate:.1e}$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'Franke$(x,y)$')

plt.tight_layout()
if save:
    plt.savefig(f'../Figures/NN_3D_Predict_Franke_Epochs{epochs}.pdf')
plt.show()