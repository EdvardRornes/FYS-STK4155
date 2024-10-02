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
save = True
latex_fonts()

deg_max = 30
degrees = np.arange(1, deg_max + 1)
lmbdas = [1e-8]
k = 10

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
    OLS = PolynomialRegression("OLS", deg_max, [x,y,z], start_training=False)
    RIDGE = PolynomialRegression("Ridge", deg_max, [x,y,z], lmbdas=lmbdas, start_training=False)
    LASSO = PolynomialRegression("LASSO", deg_max, [x,y,z], lmbdas=lmbdas, start_training=False)
    
    ########## CV-storage ##########
    MSE_OLS_CV = np.zeros(deg_max); MSE_OLS_CV_STD = np.zeros(deg_max)
    MSE_RIDGE_CV = np.zeros((deg_max, len(lmbdas))); MSE_RIDGE_CV_STD = np.zeros((deg_max, len(lmbdas)))
    MSE_LASSO_CV = np.zeros((deg_max, len(lmbdas))); MSE_LASSO_CV_STD = np.zeros((deg_max, len(lmbdas)))
    
    plt.figure(1, figsize=(10, 6))
    
    start_time = time.time()
    for i in range(deg_max):
        X = OLS.Design_Matrix(x, y, degrees[i])

        # OLS:
        _, _, MSE_test_mean, MSE_test_STD = OLS.Cross_Validation(X, z, k)
        MSE_OLS_CV[i] = MSE_test_mean; MSE_OLS_CV_STD[i] = MSE_test_STD

        # LASSO/RIDGE:
        for j in range(len(lmbdas)):
            _ , _, MSE_test_mean, MSE_test_STD = RIDGE.Cross_Validation(X, z, k, lmbda=lmbdas[j])
            MSE_RIDGE_CV[i,j] = MSE_test_mean; MSE_RIDGE_CV_STD[i,j] = MSE_test_STD

            _ , _, MSE_test_mean, MSE_test_STD = LASSO.Cross_Validation(X, z, k, lmbda=lmbdas[j])
            MSE_LASSO_CV[i,j] = MSE_test_mean; MSE_LASSO_CV_STD[i,j] = MSE_test_STD

        print(f"{i/deg_max*100:.1f}%, duration: {time.time()-start_time:.2f}s", end="\r")

    print(f"100.0, duration: {time.time()-start_time:.2f}s")
    """
    - style: LASSO, -. style: RIDGE, errorbars means CV.
    """
    plt.errorbar(degrees, MSE_OLS_CV, MSE_OLS_CV_STD, label="OLS", capsize=5, lw=2.5)
    for j in range(len(lmbdas)):
        # LASSO:
        plt.errorbar(degrees, MSE_LASSO_CV[:,j], MSE_LASSO_CV_STD[:,j], label=rf"LASSO, $\lambda = {lmbdas[j]}$", capsize=5, lw=2.5)

        # RIDGE: 
        plt.errorbar(degrees, MSE_RIDGE_CV[:,j], MSE_RIDGE_CV_STD[:,j], label=rf"Ridge, $\lambda = {lmbdas[j]}$", capsize=5, lw=2.5)

    plt.title(f"MSE of OLS, Ridge, and LASSO with CV ({name})")
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    # plt.legend(handles=handles, labels=labels)
    if save:
        plt.savefig(f"Figures/Terrain_CV_Regression_{name}.pdf")
    plt.show()
