import numpy as np
import matplotlib.pyplot as plt

from utils import *

save = True
# latex_fonts()

N = 75; eps = 0.1
franke = Franke(N, eps)
x,y,z = franke.x, franke.y, franke.z
data = [x,y,z]
lmbdas = [1e-10, 1e-7, 1e-4, 1e-1]

k = 10

deg_max = 6
degrees = np.arange(1, deg_max+1)
MSE_OLS_CV       = np.zeros(len(degrees))
MSE_OLS_CV_STD   = np.zeros(len(degrees))
MSE_Ridge_CV     = np.zeros((len(degrees), len(lmbdas)))
MSE_Ridge_CV_STD = np.zeros((len(degrees), len(lmbdas)))
MSE_LASSO_CV     = np.zeros((len(degrees), len(lmbdas)))
MSE_LASSO_CV_STD = np.zeros((len(degrees), len(lmbdas)))



OLS = PolynomialRegression("OLS", deg_max, data, start_training=False)
Ridge = PolynomialRegression("OLS", deg_max, data, start_training=False)
LASSO = PolynomialRegression("OLS", deg_max, data, start_training=False)
start_time = time.time()
for i in range(deg_max):
    X = PolynomialRegression.Design_Matrix(x, y, degrees[i])

    MSE_train_mean, MSE_train_STD, MSE_test_mean, MSE_test_STD = OLS.Cross_Validation(X, z, k)
    MSE_OLS_CV[i] = MSE_test_mean
    MSE_OLS_CV_STD[i] = MSE_test_STD
    for lmbda, j in zip(lmbdas, range(len(lmbdas))):

        # Ridge
        MSE_train_mean, MSE_train_STD, MSE_test_mean, MSE_test_STD = Ridge.Cross_Validation(X, z, k, lmbda=lmbda)  
        MSE_Ridge_CV[i,j] = MSE_test_mean
        MSE_Ridge_CV_STD[i,j] = MSE_test_STD

        MSE_train_mean, MSE_train_STD, MSE_test_mean, MSE_test_STD = Ridge.Cross_Validation(X, z, k, lmbda=lmbda)  
        MSE_LASSO_CV[i,j] = MSE_test_mean
        MSE_LASSO_CV_STD[i,j] = MSE_test_STD  

    print(f"CV: {i/deg_max*100:.1f}%, duration: {(time.time()-start_time):.2f}s", end="\r")

print(f"CV: 100.0%, duration: {(time.time()-start_time):.2f}s")

for lmbda, j in zip(lmbdas, range(len(lmbdas))):
    plt.figure(figsize=(10, 6))
    plt.title("MSE of CV")
    plt.errorbar(degrees, MSE_OLS_CV, MSE_OLS_CV_STD, label="OLS", capsize=5)
    plt.errorbar(degrees, MSE_Ridge_CV[:,j], MSE_Ridge_CV_STD[:,j], label=rf"Ridgewith $\lambda=${lmbda}", capsize=5)
    plt.errorbar(degrees, MSE_LASSO_CV[:,j], MSE_LASSO_CV_STD[:,j], label=rf"LASSOwith $\lambda=${lmbda}", capsize=5)
    print(f"lmbda = {lmbda}:")
    print(f"    OLS: min: {np.min(MSE_OLS_CV):.2e}, mean: {np.mean(MSE_OLS_CV):.2e}, best poly deg: {np.argmin(MSE_OLS_CV)+1}")
    print(f"    Ridge: min: {np.min(MSE_Ridge_CV[:,j]):.2e}, mean: {np.mean(MSE_Ridge_CV[:,j]):.2e}, best poly deg: {np.argmin(MSE_Ridge_CV[:,j])+1}")
    print(f"    LASSO: min: {np.min(MSE_LASSO_CV[:,j]):.2e}, mean: {np.mean(MSE_LASSO_CV[:,j]):.2e}, best poly deg: {np.argmin(MSE_LASSO_CV[:,j])+1}")
    print()
    plt.xlabel("Degree")
    plt.ylabel("MSE")

    ymax = np.max([np.max(q) for q in [MSE_OLS_CV[5], MSE_Ridge_CV[5,j], MSE_LASSO_CV[5,j]]]) # the 6-th poly
    plt.ylim(0, ymax)
    plt.grid(True)
    plt.legend(loc="upper left")
    if save:
        plt.savefig(f"Figures/CV/CV_{j}.pdf")

plt.show()