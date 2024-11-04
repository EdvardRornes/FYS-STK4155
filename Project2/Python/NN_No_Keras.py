import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
from utils import *
import time

np.random.seed(0)

latex_fonts()
save = True

# Sample data
N = 250; eps = 0.01
franke = Franke(N, eps)
x = franke.x; y = franke.y; z = franke.z
epochs = 1000
batch_size = 50
hidden_layers = [4, 8, 16, 32, 16, 8, 4, 2]

X = np.c_[x, y]
z = minmax_scale(z)
z_reshaped = z.reshape(-1, 1)
X_train, X_test, z_train, z_test = train_test_split(X, z_reshaped, test_size=0.25, random_state=42)

# Learning rates
learning_rates = []
log_learning_rate_min = -5
log_learning_rate_max = -2
for m in range(log_learning_rate_min, log_learning_rate_max):
    learning_rates.append(float(10**m))
    learning_rates.append(float(2*10**m))
    learning_rates.append(float(5*10**m))
learning_rates.append(float(10**log_learning_rate_max))

lambdas = np.logspace(-10, 0, 11)

# Prepare to store MSE and R2 for each activation type
mse_results = {"ReLU": [], "Sigmoid": [], "Leaky ReLU": []}
r2_results = {"ReLU": [], "Sigmoid": [], "Leaky ReLU": []}

# Initialize FFNNs w/ Adam optimizer
ffnn_relu = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='relu')
ffnn_sigmoid = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='sigmoid')
ffnn_lrelu = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='lrelu')

# Loop over each learning rate, regularization lambda, and activation type
heatmap_data = {"ReLU": [], "Sigmoid": [], "Leaky ReLU": []}

start_time = time.time()
total_iterations = 3 * len(learning_rates) * len(lambdas)
iteration = 0

# Store best results for each activation type
best_results = {"ReLU": {"mse": np.inf, "params": (None, None)}, 
                "Sigmoid": {"mse": np.inf, "params": (None, None)},
                "Leaky ReLU": {"mse": np.inf, "params": (None, None)}}

# Store best MSE history for each activation function (for epochs plot)
best_mse_history = {"ReLU": [], "Sigmoid": [], "Leaky ReLU": []}
# Store best predictions for each activation type
best_predictions = {"ReLU": None, "Sigmoid": None, "Leaky ReLU": None}

for activation_type, ffnn in zip(["ReLU", "Sigmoid", "Leaky ReLU"], [ffnn_relu, ffnn_sigmoid, ffnn_lrelu]):
    mse_for_lambdas = []
    for lambda_reg in lambdas:
        mse_for_etas = []
        for lr in learning_rates:
            iteration += 1
            percentage = 100 * iteration / total_iterations
            print(f"\ractivation = {activation_type:<7} | lambda = {lambda_reg:<6.1g} | learning rate = {lr:<7.1e} | iteration {iteration}/{total_iterations} ({percentage:<4.1f}%) | time elapsed = {time.time() - start_time:.1f} | ", end='', flush=True)
            blockPrint()

            # Train FFNN and get MSE history
            mse_history = ffnn.train(X_train, z_train, epochs=epochs, batch_size=batch_size, learning_rate=lr, lambda_reg=lambda_reg)

            z_pred = ffnn.predict(X_test)
            mse = np.mean((z_pred - z_test) ** 2)
            r2 = r2_score(z_test, z_pred)

            mse_for_etas.append(mse)
            r2_results[activation_type].append(r2)
            enablePrint()

            # Check if this MSE is the best we've seen so far for this activation type
            if mse < best_results[activation_type]["mse"]:
                best_results[activation_type]["mse"] = mse
                best_results[activation_type]["params"] = (lambda_reg, lr)  # Store best lambda and learning rate
                best_mse_history[activation_type] = mse_history  # Store MSE history for the best parameters
                
                # Save the best predictions for this activation type
                best_predictions[activation_type] = z_pred  # Renamed to z_pred
                
                print(f"\nNew best parameter combination found for {activation_type}: lambda={lambda_reg:.2e} | eta={lr:.2e}")
        
        mse_for_lambdas.append(mse_for_etas)
    
    heatmap_data[activation_type] = mse_for_lambdas

# Plot heatmaps for each activation type
for activation_type, data in heatmap_data.items():
    data_array = np.array(data).T
    plot_2D_parameter_lambda_eta(
        lambdas=lambdas,
        etas=learning_rates,
        value=data_array,
        title=f"Heatmap for MSE with {activation_type} activation",
        x_log=True,
        y_log=True,
        savefig=save,
        filename=f"Heatmap_MSE_{activation_type}_Franke_Epochs{epochs}")

# Report best parameters for each activation
for activation_type, result in best_results.items():
    best_lambda, best_eta = result["params"]
    print(f"Best parameters for {activation_type} activation: lambda={best_lambda:.2e}, eta={best_eta:.2e}")

# Plot MSE as a function of epochs for best parameters
plt.figure(figsize=(12, 6))

for activation_type in best_mse_history:
    best_lambda, best_eta = best_results[activation_type]["params"]
    label = fr"{activation_type} ($\lambda={best_lambda:.2e}, \eta={best_eta:.2e}$)"
    plt.plot(range(1, epochs + 1), best_mse_history[activation_type], label=label)

plt.title('MSE as a function of epochs for best parameter combinations')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.yscale('log')
plt.legend()
plt.grid()
if save:
    plt.savefig(f'../Figures/Best_MSE_vs_Epochs{epochs}.pdf')

# Prepare for 3D plot
fig = plt.figure(figsize=(10, 10))
x_grid, y_grid = np.meshgrid(np.unique(x), np.unique(y))

# Define activation functions and their corresponding FFNN models
activation_functions = {
    'ReLU': (best_results['ReLU']['params']),
    'Sigmoid': (best_results['Sigmoid']['params']),
    'Leaky ReLU': (best_results['Leaky ReLU']['params']),
}

# Plot each FFNN model's prediction in 3D with optimal parameters
for idx, (activation_type, ((best_lambda, best_eta))) in enumerate(activation_functions.items(), start=1):
    ax = fig.add_subplot(2, 2, idx, projection='3d')
    print(f'Activation {activation_type} using lambda={best_lambda} and eta={best_eta}')
    
    # Scatter plot of the original data
    ax.scatter(x, y, z, color='blue', label='Original Data', s=10, alpha=0.6)

    # Use the best predictions
    z_pred_best = best_predictions[activation_type]  # Using the best predictions saved

    # Grid for the surface
    x_grid, y_grid = np.meshgrid(np.unique(x), np.unique(y))

    # Interpolate to get smooth surface values
    z_grid_pred = griddata((X_test[:,0], X_test[:,1]), z_pred_best.flatten(), (x_grid, y_grid), method='cubic')

    # Plot the surface
    ax.plot_surface(x_grid, y_grid, z_grid_pred, alpha=0.5, color='orange' if activation_type == 'ReLU' else 'green' if activation_type == 'Sigmoid' else 'yellow')
    
    # Set titles and labels
    ax.set_title(fr'{activation_type} Activation' + '\n' + fr'Best $\lambda={best_lambda:.1g}$, $\eta={best_eta:.1g}$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    ax.legend(loc='upper left')

# Adjust subplot layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.suptitle("3D Plots of FFNN Predictions with Optimal Parameters")
if save:
    plt.savefig(f'../Figures/NN_noKeras_3D_Franke_Epochs{epochs}.pdf')
plt.show()

