import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from utils import *
import matplotlib.lines as mlines


# Define terrain files
files = {
    'Grand Canyon': 'Data/GrandCanyon.tif',
    'Mount Everest': 'Data/Everest.tif'
}
# Setup:
save = False
# latex_fonts()
# Initialize storage for results
deg_max = 25
degrees = np.arange(1, deg_max + 1)
lmbdas = [1e-10, 1e-5, 1e-2, 1e-1]
k = 5

N = 25


# Iterate through terrain files
for name, file in files.items():
    terrain = imageio.imread(file)
    # Use a subset of the terrain data for analysis
    terrain_subset = terrain[:N, :N]
    z_shape = np.shape(terrain_subset)
    x = np.linspace(0, 1, z_shape[0])
    y = np.linspace(0, 1, z_shape[1])
    x, y = np.meshgrid(x, y)

    z = terrain_subset.flatten()
    x = x.flatten()
    y = y.flatten()

    ########## Reg-models ##########
    OLS = PolynomialRegression("OLS", deg_max, [x,y,z])
    RIDGE = PolynomialRegression("Ridge", deg_max, [x,y,z], lmbdas=lmbdas)
    LASSO = PolynomialRegression("LASSO", deg_max, [x,y,z], lmbdas=lmbdas)

    MSE_OLS = OLS.MSE_test
    MSE_RIDGE = RIDGE.MSE_test
    MSE_LASSO = LASSO.MSE_test
    
    ########## CV-storage ##########
    MSE_OLS_CV = np.zeros(deg_max); MSE_OLS_CV_STD = np.zeros(deg_max)
    MSE_RIDGE_CV = np.zeros((deg_max, len(lmbdas))); MSE_RIDGE_CV_STD = np.zeros((deg_max, len(lmbdas)))
    MSE_LASSO_CV = np.zeros((deg_max, len(lmbdas))); MSE_LASSO_CV_STD = np.zeros((deg_max, len(lmbdas)))
    
    plt.figure(1, figsize=(10, 6))
    line = plt.plot(degrees, MSE_OLS, "--", label=f"{name} - OLS", lw=2.5)
    color = line[0].get_color()
    plt.errorbar(degrees, MSE_OLS_CV, MSE_OLS_CV_STD, fmt="none", color=color, capsize=5, lw=2.5)
    
    # Calculate OLS
    for i in range(deg_max):
        X = OLS.Design_Matrix(x, y, degrees[i])

        # CV:
        _, _, MSE_test_mean, MSE_test_STD = OLS.Cross_Validation(X, z, k)
        MSE_OLS_CV[i] = MSE_test_mean; MSE_OLS_CV_STD[i] = MSE_test_STD

        # lambdas:
        for j in range(len(lmbdas)):
            _ , _, MSE_test_mean, MSE_test_STD = RIDGE.Cross_Validation(X, z, k, lmbda=lmbdas[j])
            MSE_RIDGE_CV[i,j] = MSE_test_mean; MSE_RIDGE_CV_STD[i,j] = MSE_test_STD

            _ , _, MSE_test_mean, MSE_test_STD = LASSO.Cross_Validation(X, z, k, lmbda=lmbdas[j])
            MSE_LASSO_CV[i,j] = MSE_test_mean; MSE_LASSO_CV_STD[i,j] = MSE_test_STD

    """
    - style: LASSO, -. style: RIDGE, errorbars means CV.
    """
    for j in range(len(lmbdas)):
        # LASSO:
        line = plt.plot(degrees, MSE_LASSO[:,j], label=rf"$\lambda$={lmbdas[j]}", lw=2.5)
        color = line[0].get_color()
        plt.errorbar(degrees, MSE_LASSO_CV[:,j], MSE_LASSO_CV_STD[:,j], fmt="none", color=color, capsize=5, lw=2.5)

        # RIDGE: 
        plt.plot(degrees, MSE_RIDGE[:,j], "-.", color=color, lw=2.5, zorder=25)
        plt.errorbar(degrees, MSE_RIDGE_CV[:,j], MSE_RIDGE_CV_STD[:,j], fmt="none", color=color, capsize=5, lw=2.5)
    
    lasso_line = mlines.Line2D([], [], color='black', label='LASSO')
    ridge_line = mlines.Line2D([], [], linestyle='-.', color='black', label='Ridge')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(lasso_line)
    handles.append(ridge_line)
    labels.append("LASSO")
    labels.append("Ridge")

    plt.title(f"MSE of OLS, Ridge, and LASSO with and without CV ({name})")
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend(handles=handles, labels=labels)
    if save:
        plt.savefig(f"Figures/Terrain_CV_Regression_{name}.pdf")
    plt.show()
