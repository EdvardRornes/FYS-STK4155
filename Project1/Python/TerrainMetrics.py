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
# Plot
np.random.seed(42)
latex_fonts()
save = False; overwrite = False
run_CV = True; run_log = True 

folder = "Figures/Terrain"

################ Scaling options ################
additional_description = "Unscaled"
# additional_description = "MINMAX"
additional_description = "StandardScaling"

# Setup
deg_max = 17; deg_max_cv = 17
degrees = np.arange(1, deg_max + 1); degrees_cv = np.arange(1, deg_max_cv + 1)
lmbdas = [1e-2]
k = 10

N = 10

# For lmbda-log plot:
log_lambda_start = -10
log_lambda_stop = -1
lambda_num = 100
deg_analysis = 4

lmbdadas = np.logspace(log_lambda_start, log_lambda_stop, lambda_num)

MSE_train_array_Ridge = np.zeros(lambda_num); MSE_train_array_LASSO = np.zeros(lambda_num)
MSE_test_array_Ridge = np.zeros(lambda_num); MSE_test_array_LASSO = np.zeros(lambda_num)
R2_train_array_Ridge = np.zeros(lambda_num); R2_train_array_LASSO = np.zeros(lambda_num)
R2_test_array_Ridge = np.zeros(lambda_num); R2_test_array_LASSO = np.zeros(lambda_num)

# Iterate through terrain files
for name, file in files.items():
    terrain = imageio.imread(file)
    # Use a subset of the terrain data for analysis
    terrain_subset = terrain[:N, :N]
    terrain_subset = terrain[::int(terrain.shape[0]/N), ::int(terrain.shape[1]/N)]

    z_shape = np.shape(terrain_subset)
    x = np.linspace(0, 1, z_shape[0])
    y = np.linspace(0, 1, z_shape[1])
    x, y = np.meshgrid(x, y)

    z = terrain_subset.flatten()
    x = x.flatten()
    y = y.flatten()

    if run_CV:
        ########## Reg-models ##########
        OLS = PolynomialRegression("OLS", deg_max_cv, [x,y,z], start_training=True, scaling=additional_description)
        RIDGE = PolynomialRegression("RIDGE", deg_max_cv, [x,y,z], lmbdas=lmbdas, start_training=True, scaling=additional_description)
        LASSO = PolynomialRegression("LASSO", deg_max_cv, [x,y,z], lmbdas=lmbdas, start_training=True, scaling=additional_description, 
                                     tol=0.5)
        
        ########## CV-storage ##########
        MSE_OLS_CV   = np.zeros(deg_max_cv);                MSE_OLS_CV_STD   = np.zeros(deg_max_cv)
        MSE_RIDGE_CV = np.zeros((deg_max_cv, len(lmbdas))); MSE_RIDGE_CV_STD = np.zeros((deg_max_cv, len(lmbdas)))
        MSE_LASSO_CV = np.zeros((deg_max_cv, len(lmbdas))); MSE_LASSO_CV_STD = np.zeros((deg_max_cv, len(lmbdas)))
        
        plt.figure(1, figsize=(10, 6))

        start_time = time.time()
        for i in range(deg_max_cv):
            X = OLS.Design_Matrix(x, y, degrees_cv[i])

            # OLS:
            _, _, MSE_test_mean, MSE_test_STD, _, _, _, _ = OLS.Cross_Validation(X, z, k)
            MSE_OLS_CV[i] = MSE_test_mean; MSE_OLS_CV_STD[i] = MSE_test_STD

            # LASSO/RIDGE:
            for j in range(len(lmbdas)):
                _ , _, MSE_test_mean, MSE_test_STD, _, _, _, _ = RIDGE.Cross_Validation(X, z, k, lmbda=lmbdas[j])
                MSE_RIDGE_CV[i,j] = MSE_test_mean; MSE_RIDGE_CV_STD[i,j] = MSE_test_STD

                _ , _, MSE_test_mean, MSE_test_STD, _, _, _, _ = LASSO.Cross_Validation(X, z, k, lmbda=lmbdas[j])
                MSE_LASSO_CV[i,j] = MSE_test_mean; MSE_LASSO_CV_STD[i,j] = MSE_test_STD

            print(f"CV: {i/deg_max_cv*100:.1f}%, duration: {time.time()-start_time:.2f}s", end="\r")

        print(f"CV: 100.0%, duration: {time.time()-start_time:.2f}s")

        """
        - style: LASSO, -. style: RIDGE, errorbars means CV.
        """
        plt.errorbar(degrees_cv, MSE_OLS_CV, MSE_OLS_CV_STD, label="OLS", capsize=5, lw=2.5)
        for j in range(len(lmbdas)):
            # LASSO:
            plt.errorbar(degrees_cv, MSE_LASSO_CV[:,j], MSE_LASSO_CV_STD[:,j], label=rf"LASSO, $\lambda = {lmbdas[j]}$", capsize=5, lw=2.5)

            # RIDGE: 
            plt.errorbar(degrees_cv, MSE_RIDGE_CV[:,j], MSE_RIDGE_CV_STD[:,j], label=rf"Ridge, $\lambda = {lmbdas[j]}$", capsize=5, lw=2.5)

        plt.title(fr"MSE of OLS, Ridge, and LASSO with CV({name}) {additional_description}")
        plt.xlabel("Degree")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.legend()
        # plt.legend(handles=handles, labels=labels)
        if save:
            save_plt(f"{folder}/Terrain_CV_Regression_{name}_{additional_description}", overwrite=overwrite)

    if run_log:
        RIDGE = PolynomialRegression("RIDGE", deg_max, [x,y,z], lmbdas=lmbdadas, start_training=False)
        X_Ridge = PolynomialRegression.Design_Matrix(x, y, deg_analysis)

        # Split into training and testing and scale
        X_train_Ridge, X_test_Ridge, z_train_Ridge, z_test_Ridge = train_test_split(X_Ridge, z, test_size=0.25, random_state=4)
        X_train_Ridge, X_test_Ridge, z_train_Ridge, z_test_Ridge = PolynomialRegression.scale_data(X_train_Ridge, X_test_Ridge, z_train_Ridge, z_test_Ridge)   
        
        LASSO = PolynomialRegression("LASSO", deg_max, [x,y,z], lmbdas=lmbdadas, start_training=False, tol=0.5)
        X_LASSO = PolynomialRegression.Design_Matrix(x, y, deg_analysis)

        # Split into training and testing and scale
        X_train_LASSO, X_test_LASSO, z_train_LASSO, z_test_LASSO = train_test_split(X_LASSO, z, test_size=0.25, random_state=4)
        X_train_LASSO, X_test_LASSO, z_train_LASSO, z_test_LASSO = PolynomialRegression.scale_data(X_train_LASSO, X_test_LASSO, z_train_LASSO, z_test_LASSO)
        
        start_time = time.time()
        for i in range(lambda_num):
            _, MSE_train_array_Ridge[i], MSE_test_array_Ridge[i], R2_train_array_Ridge[i], R2_test_array_Ridge[i] = RIDGE.Ridge_fit(X_train_Ridge, X_test_Ridge, z_train_Ridge, z_test_Ridge, lmbdadas[i])

            _, MSE_train_array_LASSO[i], MSE_test_array_LASSO[i], R2_train_array_LASSO[i], R2_test_array_LASSO[i] = LASSO_default(X_train_LASSO, X_test_LASSO, z_train_LASSO, z_test_LASSO, lmbdadas[i])

            print(f"log10 lambda analysis: {i/lambda_num*100:.1f}%, duration: {time.time()-start_time:.2f}s", end="\r")

        print(f"log10 lambda analysis: 100.0%, duration: {time.time()-start_time:.2f}s", end="\r")
        
        plt.figure(figsize=(10, 6))
        plt.title(rf"MSE with polynomial degree $p={deg_analysis}$ ({name}) {additional_description}")
        line = plt.plot(np.log10(lmbdadas), MSE_train_array_Ridge, label="Ridge train", lw=2.5)
        color = line[0].get_color()
        plt.plot(np.log10(lmbdadas), MSE_test_array_Ridge, "--", label="Ridge test", color=color, lw=2.5)

        line = plt.plot(np.log10(lmbdadas), MSE_train_array_LASSO, label="LASSO train", lw=2.5)
        color = line[0].get_color()
        plt.plot(np.log10(lmbdadas), MSE_test_array_LASSO, "--", label="LASSO test", color=color, lw=2.5)

        plt.xlabel(r"$\log_{10}\lambda$")
        plt.ylabel("MSE")
        plt.xlim(log_lambda_start, log_lambda_stop)
        plt.legend()
        plt.grid(True)
        if save:
            save_plt(f"{folder}/logMSE_{name}_{additional_description}_{deg_analysis}", overwrite=overwrite)

        plt.figure(figsize=(10, 6))
        plt.title(rf"$R^2$ with polynomial degree $p={deg_analysis}$ ({name}) {additional_description}")
        line = plt.plot(np.log10(lmbdadas), R2_train_array_Ridge, label=r"Ridge", lw=2.5)
        color = line[0].get_color()
        plt.plot(np.log10(lmbdadas), R2_test_array_Ridge, "--", color=color, lw=2.5)

        line = plt.plot(np.log10(lmbdadas), R2_train_array_LASSO, label=r"LASSO", lw=2.5)
        color = line[0].get_color()
        plt.plot(np.log10(lmbdadas), R2_test_array_LASSO, "--", color=color, lw=2.5)

        plt.xlabel(r"$\log_{10}\lambda$")
        plt.ylabel(r"$R^2$")
        plt.xlim(log_lambda_start, log_lambda_stop)
        plt.legend()
        plt.grid(True)
        if save:
            save_plt(f"{folder}/logR2_{name}_{additional_description}", overwrite=overwrite)



        plt.show()