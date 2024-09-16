import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
np.random.seed(2024)
N = 100  # Number of data points
x = np.sort(np.random.rand(N))
y = np.sort(np.random.rand(N))
z = Franke(x, y)
z = z + 0.1 * np.random.normal(N, 1, z.shape)  # Add some noise to the data

deg_max = 8
degrees = np.arange(1, deg_max+1)
MSE_train = np.zeros(len(degrees))
MSE_test = np.zeros(len(degrees))
R2_train = np.zeros(len(degrees))
R2_test = np.zeros(len(degrees))
beta_coefficients = [0]*(deg_max+1)
# beta_coefficients = [] # See below

for deg in range(deg_max):
    # Create polynomial features
    X = Design_Matrix(x, y, degrees[deg])
    # Split into training and testing and scale
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25, random_state=42)
    X_train, X_test, z_train, z_test = scale_data(X_train, X_test, z_train, z_test)

    beta_coefficients[deg], MSE_train[deg], MSE_test[deg], R2_train[deg], R2_test[deg] = OLS_fit(X_train, X_test, z_train, z_test)
    
    '''
    # I have troubles when trying to plot the different β's. What GPT proposed is that instead of using the above we instead do this

    beta[deg], MSE_train[deg], MSE_test[deg], R2_train[deg], R2_test[deg] = OLS_fit(X_train, X_test, z_train, z_test)

    # Flatten beta if necessary
    if beta.ndim == 2:
        beta = beta.flatten()
    
    beta_coefficients.append(beta)
    '''


# Plot MSE
plt.figure(figsize=(10, 6))
plt.plot(degrees, MSE_train, label=r"MSE train", lw=2.5)
plt.plot(degrees, MSE_test, label=r"MSE test", lw=2.5)
plt.xlabel(r'Degree')
plt.ylabel(r'MSE')
plt.xlim(1, deg_max)
plt.title(r"OLS MSE")
plt.legend()
plt.grid(True)
plt.savefig(f'Figures/OLS-MSE-degree.pdf')

# Plot R²
plt.figure(figsize=(10, 6))
plt.plot(degrees, R2_train, label=r"$R^2$ train", lw=2.5)
plt.plot(degrees, R2_test, label=r"$R^2$ test", lw=2.5)
plt.xlabel(r'Degree')
plt.ylabel(r'$R^2$')
plt.xlim(1, deg_max)
plt.title(r"OLS $R^2$")
plt.legend()
plt.grid(True)
plt.savefig(f'Figures/OLS-R2-degree.pdf')
plt.show()

'''
# With the commented code previously we can then get an omega scuffed plot using the below. Isak you know how to do this?

# Process beta coefficients for plotting
max_len = max([len(beta) for beta in beta_coefficients])
beta_array = np.full((deg_max, max_len), np.nan)
for i, beta in enumerate(beta_coefficients):
    beta_array[i, :len(beta)] = beta

# Plot the beta coefficients
plt.figure(figsize=(10, 6))
for i in range(max_len):
    plt.plot(degrees, beta_array[:, i], label=rf'$\beta_{i}$', lw=2.5)

plt.xlabel(r'Degree')
plt.ylabel(r'$\beta$ values')
plt.xlim(1, deg_max)
plt.title(r'OLS $\beta$ coefficients')
plt.legend()
plt.grid(True)
plt.savefig(f'Figures/OLS-beta-degree.pdf')
plt.show()
'''