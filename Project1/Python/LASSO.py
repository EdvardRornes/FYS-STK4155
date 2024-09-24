import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
from utils import *

# Latex fonts
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlepad'] = 25 

plt.rcParams.update({
    'font.family' : 'euclid',
    'font.weight' : 'bold',
    'font.size': 17,       # General font size
    'axes.labelsize': 17,  # Axis label font size
    'axes.titlesize': 22,  # Title font size
    'xtick.labelsize': 22, # X-axis tick label font size
    'ytick.labelsize': 22, # Y-axis tick label font size
    'legend.fontsize': 17, # Legend font size
    'figure.titlesize': 25 # Figure title font size
})

# Generate data
# np.random.seed(2024)
N = 100  # Number of data points
x = np.sort(np.random.rand(N))
y = np.sort(np.random.rand(N))
z = Franke(x, y)
z = z + 0.1 * np.random.normal(N, 1, z.shape)  # Add some noise to the data

deg_max = 8
degrees = np.arange(1, deg_max+1)
lambdas = [1e-10, 1e-5, 1e-2, 1e-1]

MSE_train = np.zeros(len(degrees))
MSE_test = np.zeros(len(degrees))
R2_train = np.zeros(len(degrees))
R2_test = np.zeros(len(degrees))
beta_coefficients = [0] * (deg_max+1)

cmap = cm.get_cmap('tab10', len(lambdas))

plt.figure(figsize=(10, 6))

MSE_train_array, MSE_test_array = np.zeros((len(lambdas), deg_max)), np.zeros((len(lambdas), deg_max))
R2_train_array, R2_test_array = np.zeros((len(lambdas), deg_max)), np.zeros((len(lambdas), deg_max))

for l, i in zip(lambdas, range(len(lambdas))):
    for deg in range(deg_max):
        # Create polynomial features
        X = Design_Matrix(x, y, degrees[deg])
        # Split into training and testing and scale
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25, random_state=42)
        X_train, X_test, z_train, z_test = scale_data(X_train, X_test, z_train, z_test)

        beta_coefficients[deg], MSE_train[deg], MSE_test[deg], R2_train[deg], R2_test[deg] = LASSO_fit(X_train, X_test, z_train, z_test, l)

    # Plot MSE
    MSE_train_array[i, :], MSE_test_array[i, :] = MSE_train, MSE_test
    R2_train_array[i, :], R2_test_array[i, :] = R2_train, R2_test
    plt.plot(degrees, MSE_train_array[i, :], lw=2.5, color=cmap(i))
    plt.plot(degrees, MSE_test_array[i, :], lw=2.5, linestyle='--', color=cmap(i))

# Overly complicated way to make the legends look nice
train_handle = Line2D([0], [0], color='black', lw=2.5, label='Train')
test_handle = Line2D([0], [0], color='black', lw=2.5, linestyle='--', label='Test')

lambda_handles = []
for i, l in enumerate(lambdas):
    if l < 1:  # If lambda is a negative power of 10
        power = int(np.log10(l))
        label = rf'$\lambda=10^{{{power}}}$'
    else:  # For positive powers
        label = rf'$\lambda={l}$'
    lambda_handles.append(Line2D([0], [0], color=cmap(i), lw=2.5, label=label))

plt.legend(handles=[train_handle, test_handle] + lambda_handles)
plt.xlabel(r'Degree')
plt.ylabel(r'MSE')
plt.xlim(1, deg_max)
plt.title(rf"LASSO MSE")
plt.grid(True)
plt.savefig(f'Figures/LASSO-MSE-degree.pdf')

# # Plot R²
plt.figure(figsize=(10, 6))
for l, i in zip(lambdas, range(len(lambdas))):
    plt.plot(degrees, R2_train_array[i, :], lw=2.5, color=cmap(i))
    plt.plot(degrees, R2_test_array[i, :], lw=2.5, linestyle='--', color=cmap(i))

plt.legend(handles=[train_handle, test_handle] + lambda_handles)
plt.xlabel(r'Degree')
plt.ylabel(r'$R^2$')
plt.xlim(1, deg_max)
plt.title(rf"LASSO $R^2$")
plt.grid(True)
plt.savefig(f'Figures/LASSO-R2-degree.pdf')

lambda_exp_start = -10
lambda_exp_stop = -1
lambda_num = 100

lambdas = np.logspace(lambda_exp_start, lambda_exp_stop, num=lambda_num)
MSE_train_array = np.zeros(lambda_num)
MSE_test_array = np.zeros(lambda_num)
R2_train_array = np.zeros(lambda_num)
R2_test_array = np.zeros(lambda_num)
beta_list = [0]*lambda_num

for i in range(lambda_num):
    beta_list[i], MSE_train_array[i], MSE_test_array[i], R2_train_array[i], R2_test_array[i] = LASSO_fit(X_train, X_test, z_train, z_test, lambdas[i])

plt.figure(figsize=(10, 6))
plt.title(rf"MSE deg {deg_max}.")
plt.plot(np.log10(lambdas), MSE_train_array, label="MSE train", lw=2.5)
plt.plot(np.log10(lambdas), MSE_test_array, label="MSE test", lw=2.5)
plt.xlabel(r"$\log_{10}(\lambda)$")
plt.ylabel("MSE")
plt.xlim(lambda_exp_start, lambda_exp_stop)
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.title(rf"$R^2$ with deg {deg_max}.")
plt.plot(np.log10(lambdas), R2_train_array,label=r"$R^2$ train", lw=2.5)
plt.plot(np.log10(lambdas), R2_test_array,label=r"$R^2$ test", lw=2.5)
plt.xlabel(r"$\log_{10}(\lambda)$")
plt.ylabel(r"$R^2$")
plt.xlim(lambda_exp_start, lambda_exp_stop)
plt.legend()
plt.grid(True)

plt.show()