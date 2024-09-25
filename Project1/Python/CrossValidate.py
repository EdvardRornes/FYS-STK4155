import numpy as np
import matplotlib.pyplot as plt

from utils import *

save = False 
latex_fonts()

N = 100; eps = 0.1
franke = Franke(N, eps)
x,y,z = franke.x, franke.y, franke.z
data = [x,y,z]


k = 5

deg_max = 10
degrees = np.arange(1, deg_max+1)
MSE_OLS_CV       = np.zeros(len(degrees))
MSE_OLS_CV_STD   = np.zeros(len(degrees))
MSE_OLS          = np.zeros(len(degrees))
MSE_Ridge_CV     = np.zeros(len(degrees))
MSE_Ridge_CV_STD = np.zeros(len(degrees))
MSE_Ridge        = np.zeros(len(degrees))
MSE_LASSO_CV     = np.zeros(len(degrees))
MSE_LASSO_CV_STD = np.zeros(len(degrees))
MSE_LASSO        = np.zeros(len(degrees))

lmbdadas = [1e-10, 1e-5, 1e-2, 1e-1, 1, 10]


for lmbda in lmbdadas:
    for i in range(deg_max):
        X = Design_Matrix(x, y, degrees[i])

        MSE_train_mean, MSE_train_STD, MSE_test_mean, MSE_test_STD = Cross_Validation(X, z, k)
        MSE_OLS_CV[i] = MSE_test_mean
        MSE_OLS_CV_STD[i] = MSE_test_STD

        # OLS without CV
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)
        X_train, X_test, z_train, z_test = scale_data(X_train, X_test, z_train, z_test)
        beta, MSE_train, MSE_test, R2_train, R2_test = OLS_fit(X_train, X_test, z_train, z_test)
        MSE_OLS[i] = MSE_test

        # Ridge
        MSE_train_mean, MSE_train_STD, MSE_test_mean, MSE_test_STD = Cross_Validation(X, z, k, reg_method="RIDGE", lmbda=lmbda)  
        MSE_Ridge_CV[i] = MSE_test_mean
        MSE_Ridge_CV_STD[i] = MSE_test_STD

        # Ridge without CV
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)
        X_train, X_test, z_train, z_test = scale_data(X_train, X_test, z_train, z_test)
        beta, MSE_train, MSE_test, R2_train, R2_test = Ridge_fit(X_train, X_test, z_train, z_test, lmbda=lmbda)
        MSE_Ridge[i] = MSE_test

        
        MSE_train_mean, MSE_train_STD, MSE_test_mean, MSE_test_STD = Cross_Validation(X, z, k, reg_method="RIDGE", lmbda=lmbda)  
        MSE_LASSO_CV[i] = MSE_test_mean
        MSE_LASSO_CV_STD[i] = MSE_test_STD

        # LASSO without CV
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)
        X_train, X_test, z_train, z_test = scale_data(X_train, X_test, z_train, z_test)
        beta, MSE_train, MSE_test, R2_train, R2_test = LASSO_default(X_train, X_test, z_train, z_test, lmbda=lmbda)
        MSE_LASSO[i] = MSE_test   

    # Again this is kind of scuffed, I just felt like having everything in 1 loop but its prob best to separate OLS and the others.

    plt.figure(figsize=(10, 6))
    plt.title("MSE of OLS regression with and without CV")
    plt.plot(degrees, MSE_OLS, label="Without CV")
    plt.errorbar(degrees, MSE_OLS_CV, MSE_OLS_CV_STD, label="With CV", capsize=5)
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    if save:
        plt.savefig(f"Figures/CV_OLS.pdf")

    plt.figure(figsize=(10, 6))
    plt.title(fr"MSE of Ridge regression with and without CV and $\lmbdada=${lmbda}")
    plt.plot(degrees, MSE_Ridge, label="Without CV")
    plt.errorbar(degrees, MSE_Ridge_CV, MSE_Ridge_CV_STD, label="With CV", capsize=5)
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    if save:
        plt.savefig(f"Figures/CV_Ridge_lmbdada={lmbda}.pdf")

    plt.figure(figsize=(10, 6))
    plt.title(fr"MSE of LASSO regression with and without CV and $\lmbdada=${lmbda}")
    plt.plot(degrees, MSE_LASSO, label="Without CV")
    plt.errorbar(degrees, MSE_LASSO_CV, MSE_LASSO_CV_STD, label="With CV", capsize=5)
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    if save:
        plt.savefig(f"Figures/CV_LASSO_lmbdada={lmbda}.pdf")

plt.show()