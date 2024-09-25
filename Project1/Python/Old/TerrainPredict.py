import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from utils import *

# Configure Matplotlib for LaTeX and font settings
plt.rcParams.update({
    'text.usetex': True,
    'axes.titlepad': 25,
    'font.family': 'euclid',
    'font.weight': 'bold',
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 25,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 20,
    'figure.titlesize': 30
})

# Define terrain files
files = {
    'Grand Canyon': 'Data/GrandCanyon.tif',
    'Mount Everest': 'Data/Everest.tif'
}

# Initialize storage for results
deg_max = 15
degrees = np.arange(1, deg_max + 1)
lambdas = [1e-10, 1e-5, 1e-2, 1e-1]
k = 5

# Iterate through terrain files
for name, file in files.items():
    terrain = imageio.imread(file)

    # Use a subset of the terrain data for analysis
    N = 100
    terrain_subset = terrain[:N, :N]
    z_shape = np.shape(terrain_subset)
    x = np.linspace(0, 1, z_shape[0])
    y = np.linspace(0, 1, z_shape[1])
    x, y = np.meshgrid(x, y)

    z = terrain_subset.flatten()
    x = x.flatten()
    y = y.flatten()

    # OLS storage
    MSE_OLS = np.zeros(deg_max)
    MSE_OLS_CV = np.zeros(deg_max)
    MSE_OLS_CV_STD = np.zeros(deg_max)
    
    # Calculate OLS
    for i in range(deg_max):
        X = Design_Matrix2D(x, y, degrees[i])

        # OLS with CV
        MSE_train_mean, MSE_train_STD, MSE_test_mean, MSE_test_STD = Cross_Validation(X, z, k, 0, 0)
        MSE_OLS_CV[i] = MSE_test_mean
        MSE_OLS_CV_STD[i] = MSE_test_STD

        # OLS without CV
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)
        X_train, X_test, z_train, z_test = scale_data(X_train, X_test, z_train, z_test)

        beta, MSE_train, MSE_test, r2_train, r2_test = OLS_fit(X_train, X_test, z_train, z_test)
        MSE_OLS[i] = MSE_test

    
    # Ridge regression
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, MSE_OLS, label=f"{name} - OLS", lw=2.5)
    plt.errorbar(degrees, MSE_OLS_CV, MSE_OLS_CV_STD, label=f"{name} - OLS CV", capsize=5, lw=2.5)
    for lambd in lambdas:
        MSE_Ridge_CV = np.zeros(deg_max)
        MSE_Ridge_CV_STD = np.zeros(deg_max)
        MSE_Ridge = np.zeros(deg_max)

        for i in range(deg_max):
            X = Design_Matrix2D(x, y, degrees[i])
            MSE_train_mean, MSE_train_STD, MSE_test_mean, MSE_test_STD = Cross_Validation(X, z, k, 1, lambd)
            MSE_Ridge_CV[i] = MSE_test_mean
            MSE_Ridge_CV_STD[i] = MSE_test_STD
            
            # Ridge without CV
            X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)
            X_train, X_test, z_train, z_test = scale_data(X_train, X_test, z_train, z_test)

            beta, MSE_train, MSE_test, r2_train, r2_test = Ridge_fit(X_train, X_test, z_train, z_test, lambd)
            MSE_Ridge[i] = MSE_test

        # Plot Ridge results
        plt.plot(degrees, MSE_Ridge, label=f"{name} - Ridge (lambda={lambd})", lw=2.5)
        plt.errorbar(degrees, MSE_Ridge_CV, MSE_Ridge_CV_STD, label=f"{name} - Ridge CV (lambda={lambd})", capsize=5, lw=2.5)
    plt.title("MSE of OLS and Ridge regression with and without CV")
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"Figures/Terrain_{name}_Ridge_CV_Regression.pdf")

    # LASSO regression
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, MSE_OLS, label=f"{name} - OLS", lw=2.5)
    plt.errorbar(degrees, MSE_OLS_CV, MSE_OLS_CV_STD, label=f"{name} - OLS CV", capsize=5, lw=2.5)
    for lambd in lambdas:
        MSE_LASSO_CV = np.zeros(deg_max)
        MSE_LASSO_CV_STD = np.zeros(deg_max)
        MSE_LASSO = np.zeros(deg_max)

        for i in range(deg_max):
            X = Design_Matrix2D(x, y, degrees[i])
            MSE_train_mean, MSE_train_STD, MSE_test_mean, MSE_test_STD = Cross_Validation(X, z, k, 2, lambd)
            MSE_LASSO_CV[i] = MSE_test_mean
            MSE_LASSO_CV_STD[i] = MSE_test_STD
            
            # LASSO without CV
            X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)
            X_train, X_test, z_train, z_test = scale_data(X_train, X_test, z_train, z_test)

            beta, MSE_train, MSE_test, r2_train, r2_test = LASSO_fit(X_train, X_test, z_train, z_test, lambd)
            MSE_LASSO[i] = MSE_test

        # Plot LASSO results
        plt.plot(degrees, MSE_LASSO, label=f"{name} - LASSO (lambda={lambd})", lw=2.5)
        plt.errorbar(degrees, MSE_LASSO_CV, MSE_LASSO_CV_STD, label=f"{name} - LASSO CV (lambda={lambd})", capsize=5, lw=2.5)
    
    plt.title("MSE of OLS and LASSO regression with and without CV")
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"Figures/Terrain_{name}_LASSO_CV_Regression.pdf")

plt.show()
