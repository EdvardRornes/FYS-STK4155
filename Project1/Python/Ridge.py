import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import *

# Latex fonts
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlepad'] = 25 

font = {'family' : 'euclid',
        'weight' : 'bold',
        'size'   : 25}

# Generate data
np.random.seed(2023)
N = 100  # Number of data points
x = np.sort(np.random.rand(N))
y = np.sort(np.random.rand(N))
z = Franke(x, y)
z = z + 0.1 * np.random.normal(N, 1, z.shape)  # Add some noise to the data

deg_max = 5
degrees = np.arange(1, deg_max+1)
# Random lambda values
lambdas = [1e-20, 1e-10, 1e-5, 1e-3, 1e-2, 1e-1, 1, 10, 100]
MSE_train = np.zeros(len(degrees))
MSE_test = np.zeros(len(degrees))
R2_train = np.zeros(len(degrees))
R2_test = np.zeros(len(degrees))
beta_coefficients = [0]*(deg_max+1)

plt.figure(figsize=(10, 6))

MSE_train_array, MSE_test_array = np.zeros((len(lambdas), deg_max)), np.zeros((len(lambdas), deg_max))
R2_train_array, R2_test_array = np.zeros((len(lambdas), deg_max)), np.zeros((len(lambdas), deg_max))

for l,i in zip(lambdas, range(len(lambdas))):
    for deg in range(deg_max):
        # Create polynomial features
        X = Design_Matrix(x, y, degrees[deg])
        # Split into training and testing and scale
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25, random_state=42)
        X_train, X_test, z_train, z_test = scale_data(X_train, X_test, z_train, z_test)

        beta_coefficients[deg], MSE_train[deg], MSE_test[deg], R2_train[deg], R2_test[deg] = Ridge_fit(X_train, X_test, z_train, z_test, l)
        
        '''
        # I have troubles when trying to plot the different beta's. What is proposed is that instead of using the above we must instead have

        beta[deg], MSE_train[deg], MSE_test[deg], R2_train[deg], R2_test[deg] = Ridge_fit(X_train, X_test, z_train, z_test)

        # Flatten beta if necessary
        if beta.ndim == 2:
            beta = beta.flatten()
        
        beta_coefficients.append(beta)
        '''

    # Plot MSE
    MSE_train_array[i,:], MSE_test_array[i,:] = MSE_train, MSE_test
    R2_train_array[i,:], R2_test_array[i,:] = R2_train, R2_test
    plt.plot(degrees, MSE_train_array[i,:], label=rf"MSE train $\lambda={l}$", lw=2.5)
    plt.plot(degrees, MSE_test_array[i,:], label=rf"MSE test $\lambda={l}$", lw=2.5)


plt.xlabel(r'Degree')
plt.ylabel(r'MSE')
plt.xlim(1, deg_max)
plt.title(rf"Ridge MSE")
plt.legend()
plt.grid(True)
plt.savefig(f'Figures/Ridge-MSE-degree.pdf')

plt.figure(figsize=(10, 6))

for l,i in zip(lambdas, range(len(lambdas))):
    plt.plot(degrees, R2_train_array[i,:], label=rf"$R^2$ train $\lambda={l}$", lw=2.5)
    plt.plot(degrees, R2_test_array[i,:], label=rf"$R^2$ test $\lambda={l}$", lw=2.5)

# # Plot R²
plt.xlabel(r'Degree')
plt.ylabel(r'$R^2$')
plt.xlim(1, deg_max)
plt.title(rf"Ridge $R^2$")
plt.legend()
plt.grid(True)
plt.savefig(f'Figures/Ridge-R2-degree.pdf')
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
plt.title(r'Ridge $\beta$ coefficients')
plt.legend()
plt.grid(True)
plt.savefig(f'Figures/Ridge-beta-degree.pdf')
plt.show()
'''