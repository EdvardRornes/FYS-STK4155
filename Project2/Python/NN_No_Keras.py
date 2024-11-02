import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import ticker
from scipy.interpolate import griddata
from utils import *
import warnings
import time

np.random.seed(1)

# Suppress specific warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in matmul")
# warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")
# warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in matmul")
# warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in multiply")
# warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in subtract")
# warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in reduce")

latex_fonts()
save = True

# Sample data
N = 100; eps = 0.0
franke = Franke(N, eps)
x = franke.x; y = franke.y; z = franke.z
epochs = 1000
hidden_layers = [2, 4, 8, 16, 32, 16, 8, 4, 2]

X_train = np.c_[x, y]
z_train = z.reshape(-1, 1)

# Learning rates
learning_rates = []
log_learning_rate_min = -5
log_learning_rate_max = -1
for m in range(log_learning_rate_min, log_learning_rate_max):
    learning_rates.append(float(10**m))
    learning_rates.append(float(3*10**m))
learning_rates.append(float(10**log_learning_rate_max))

# Prepare to store MSE and R2 for each activation type
mse_results = {"ReLU": [], "Sigmoid": [], "Leaky ReLU": []}
r2_results = {"ReLU": [], "Sigmoid": [], "Leaky ReLU": []}

# Initialize FFNNs
ffnn_relu = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='relu')
ffnn_sigmoid = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='sigmoid')
ffnn_lrelu = FFNN(input_size=2, hidden_layers=hidden_layers, output_size=1, activation='lrelu')

# Loop over each learning rate, regularization lambda, and activation type
lambdas = np.logspace(-11, 1, 13)
heatmap_data = {"ReLU": [], "Sigmoid": [], "Leaky ReLU": []}

start_time = time.time()
iterations = 3 * len(learning_rates) * len(lambdas)
iterations_so_far = 0

# Store best results for each activation type
best_results = {"ReLU": {"mse": np.inf, "params": (None, None)}, 
                "Sigmoid": {"mse": np.inf, "params": (None, None)},
                "Leaky ReLU": {"mse": np.inf, "params": (None, None)}}

# Store best predictions for each activation type
best_predictions = {"ReLU": None, "Sigmoid": None, "Leaky ReLU": None}

# Modified loop to create an N x M matrix in `heatmap_data`
for activation_type, ffnn in zip(["ReLU", "Sigmoid", "Leaky ReLU"], [ffnn_relu, ffnn_sigmoid, ffnn_lrelu]):
    mse_for_lambdas = []
    
    for lambda_reg in lambdas:
        mse_for_etas = []
        
        for lr in learning_rates:
            iterations_so_far += 1
            percentage = 100 * iterations_so_far / iterations
            print(f"activation = {activation_type:<7} | lambda = {lambda_reg:<6.1g} | learning rate = {lr:<7.1e} | percentage = {percentage:<4.1f}% | time elapsed = {time.time() - start_time:.1f}")
            blockPrint()
            # Train FFNN
            ffnn.train(X_train, z_train, epochs=epochs, learning_rate=lr, lambda_reg=lambda_reg)
            z_pred = ffnn.predict(X_train)  # Renamed from y_pred to z_pred
            
            # Compute metrics
            mse = np.mean((z_pred - z_train) ** 2)
            r2 = r2_score(z_train, z_pred)
            
            mse_for_etas.append(mse)
            r2_results[activation_type].append(r2)
            enablePrint()

            # Check if this MSE is the best we've seen so far for this activation type
            if mse < best_results[activation_type]["mse"]:
                best_results[activation_type]["mse"] = mse
                best_results[activation_type]["params"] = (lambda_reg, lr)  # Store best lambda and learning rate
                
                # Save the best predictions for this activation type
                best_predictions[activation_type] = z_pred  # Renamed to z_pred
                
                # Print the new best found message
                print(f"New best parameter combination found for {activation_type}: lambda={lambda_reg:.2e} | eta={lr:.2e}")
        
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
        filename=f"Heatmap_MSE_{activation_type}_Franke"
    )

# Report best parameters for each activation
for activation_type, result in best_results.items():
    best_lambda, best_eta = result["params"]
    print(f"Best parameters for {activation_type} activation: lambda={best_lambda:.2e}, eta={best_eta:.2e}")

# Prepare for 3D plot
fig = plt.figure(figsize=(10, 10))
x_grid, y_grid = np.meshgrid(np.unique(x), np.unique(y))

# Define activation functions and their corresponding FFNN models
activation_functions = {
    'ReLU': (ffnn_relu, best_results['ReLU']['params']),
    'Sigmoid': (ffnn_sigmoid, best_results['Sigmoid']['params']),
    'Leaky ReLU': (ffnn_lrelu, best_results['Leaky ReLU']['params']),
}


# Plot each FFNN model's prediction in 3D with optimal parameters
for idx, (activation_type, (model, (best_lambda, best_eta))) in enumerate(activation_functions.items(), start=1):
    ax = fig.add_subplot(2, 2, idx, projection='3d')
    print(f'Activation {activation_type} using lambda={best_lambda} and eta={best_eta}')
    
    # Scatter plot of the original data
    ax.scatter(x, y, z, color='blue', label='Original Data', s=10, alpha=0.6)

    # Use the best predictions saved earlier
    z_pred_best = best_predictions[activation_type]  # Using the best predictions saved

    # Generate a grid for the surface
    x_grid, y_grid = np.meshgrid(np.unique(x), np.unique(y))

    # Interpolate to get smooth surface values
    z_grid_pred = griddata((x, y), z_pred_best.flatten(), (x_grid, y_grid), method='cubic')

    # Plot the surface with transparency
    ax.plot_surface(x_grid, y_grid, z_grid_pred, alpha=0.5, color='orange' if activation_type == 'ReLU' else 'green' if activation_type == 'Sigmoid' else 'yellow')
    
    # Set titles and labels
    ax.set_title(fr'{activation_type} Activation' + '\n' + fr'Best $\lambda={best_lambda:.1g}$, $\eta={best_eta:.1g}$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    ax.legend(loc='upper left')

# Adjust subplot layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add space for suptitle
fig.suptitle("3D Plots of FFNN Predictions with Optimal Parameters")
if save:
    plt.savefig(f'Figures/NN_noKeras_3D_Franke_Epochs{epochs}.pdf')
plt.show()

