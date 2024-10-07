import numpy as np
import matplotlib.pyplot as plt

from utils import *

# Plot
save = False; overwrite = False
latex_fonts()

################ Scaling options ################
additional_description = "Unscaled"
# additional_description = "MINMAX"
# additional_description = "StandardScaling"

# Setup
N = 100; eps = 0.1
franke = Franke(N, eps)
x,y,z = franke.x, franke.y, franke.z
data = [x,y,z]
lmbdas = [1e-6] # Add more lambdas to the list to get more plots. Only 1e-6 considered since LASSO does not care and this is optimal for Ridge

# Number of folds
k = 10

deg_max = 17
degrees = np.arange(1, deg_max+1)
MSE_OLS_CV       = np.zeros(len(degrees))
MSE_OLS_CV_STD   = np.zeros(len(degrees))
MSE_Ridge_CV     = np.zeros((len(degrees), len(lmbdas)))
MSE_Ridge_CV_STD = np.zeros((len(degrees), len(lmbdas)))
MSE_LASSO_CV     = np.zeros((len(degrees), len(lmbdas)))
MSE_LASSO_CV_STD = np.zeros((len(degrees), len(lmbdas)))



OLS = PolynomialRegression("OLS", deg_max, data, start_training=False, scaling=additional_description)
RIDGE = PolynomialRegression("RIDGE", deg_max, data, start_training=False, scaling=additional_description)
LASSO = PolynomialRegression("LASSO", deg_max, data, start_training=False, scaling=additional_description)
start_time = time.time()
for i in range(deg_max):
    X = PolynomialRegression.Design_Matrix(x, y, degrees[i])

    MSE_train_mean, MSE_train_STD, MSE_test_mean, MSE_test_STD = OLS.Cross_Validation(X, z, k)
    MSE_OLS_CV[i] = MSE_test_mean
    MSE_OLS_CV_STD[i] = MSE_test_STD
    for lmbda, j in zip(lmbdas, range(len(lmbdas))):

        # RIDGE
        MSE_train_mean, MSE_train_STD, MSE_test_mean, MSE_test_STD = RIDGE.Cross_Validation(X, z, k, lmbda=lmbda)  
        MSE_Ridge_CV[i,j] = MSE_test_mean
        MSE_Ridge_CV_STD[i,j] = MSE_test_STD

        MSE_train_mean, MSE_train_STD, MSE_test_mean, MSE_test_STD = LASSO.Cross_Validation(X, z, k, lmbda=lmbda)  
        MSE_LASSO_CV[i,j] = MSE_test_mean
        MSE_LASSO_CV_STD[i,j] = MSE_test_STD  

    print(f"CV: {i/deg_max*100:.1f}%, duration: {(time.time()-start_time):.2f}s", end="\r")

print(f"CV: 100.0%, duration: {(time.time()-start_time):.2f}s")

method_names = ["OLS", "Ridge", "LASSO"]
for lmbda, j in zip(lmbdas, range(len(lmbdas))):
    plt.figure(figsize=(10, 6))
    plt.title("MSE of CV")
    plt.errorbar(degrees, MSE_OLS_CV, MSE_OLS_CV_STD, label="OLS", lw=2.5, linestyle="--", capsize=5, elinewidth=1.5, capthick=1.5)

    plt.errorbar(degrees, MSE_Ridge_CV[:, j], MSE_Ridge_CV_STD[:, j], lw=2.5, linestyle="--", label=rf"Ridge with $\lambda={lmbda:.1e}$", capsize=5, elinewidth=1.5, capthick=1.5)

    plt.errorbar(degrees, MSE_LASSO_CV[:, j], MSE_LASSO_CV_STD[:, j], lw=2.5, linestyle="--", label=rf"LASSO with $\lambda={lmbda:.1e}$", capsize=5, elinewidth=1.5, capthick=1.5)


    # Find best
    print(f"lmbda = {lmbda}:")
    print(f"    OLS: min: {np.min(MSE_OLS_CV):.2e}, mean: {np.mean(MSE_OLS_CV):.2e}, best poly deg: {np.argmin(MSE_OLS_CV)+1}")
    print(f"    Ridge: min: {np.min(MSE_Ridge_CV[:, j]):.2e}, mean: {np.mean(MSE_Ridge_CV[:, j]):.2e}, best poly deg: {np.argmin(MSE_Ridge_CV[:, j]) + 1}")
    print(f"    LASSO: min: {np.min(MSE_LASSO_CV[:, j]):.2e}, mean: {np.mean(MSE_LASSO_CV[:, j]):.2e}, best poly deg: {np.argmin(MSE_LASSO_CV[:, j]) + 1}")
    minmize_me = [np.min(x) for x in [MSE_OLS_CV, MSE_Ridge_CV[:, j], MSE_LASSO_CV[:,j]]]
    print(f"    Best: min: {np.min(minmize_me):.2e}, mean: {np.min([np.mean(x) for x in [MSE_OLS_CV, MSE_Ridge_CV[:,j], MSE_LASSO_CV[:,j]]]):.2e}, w/ best poly deg: {method_names[np.argmin(minmize_me)]}")
    print()

    plt.xlim(1, deg_max)
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.yscale("log"); plt.ylabel(r"$MSE$")

    plt.ylim(6e-3, 2e-1)
    plt.grid(True)
    plt.legend(loc="lower left")
    if save:
        save_plt(f"Figures/CV/CV_{additional_description}_{j}", overwrite=overwrite)

plt.show()