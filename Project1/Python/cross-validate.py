import numpy as np
import matplotlib.pyplot as plt

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

N = 100
x = np.sort(np.random.rand(N))
y = np.sort(np.random.rand(N))
z = Franke(x, y)
z = z + 0.1*np.random.normal(0, 1, z.shape)
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

lambdas = [1e-10, 1e-5, 1e-2, 1e-1, 1, 10]


for lamb in lambdas:
    for i in range(deg_max):
        X = Design_Matrix(x, y, degrees[i])

        MSE_train_mean, MSE_train_STD, MSE_test_mean, MSE_test_STD = Cross_Validation(X, z, k, 0, lambd=0)
        MSE_OLS_CV[i] = MSE_test_mean
        MSE_OLS_CV_STD[i] = MSE_test_STD

        # OLS without CV
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)
        X_train, X_test, z_train, z_test = scale_data(X_train, X_test, z_train, z_test)
        beta, MSE_train, MSE_test, R2_train, R2_test = OLS_fit(X_train, X_test, z_train, z_test)
        MSE_OLS[i] = MSE_test

        # Ridge
        MSE_train_mean, MSE_train_STD, MSE_test_mean, MSE_test_STD = Cross_Validation(X, z, k, 1, lamb)  
        MSE_Ridge_CV[i] = MSE_test_mean
        MSE_Ridge_CV_STD[i] = MSE_test_STD

        # Ridge without CV
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)
        X_train, X_test, z_train, z_test = scale_data(X_train, X_test, z_train, z_test)
        beta, MSE_train, MSE_test, R2_train, R2_test = Ridge_fit(X_train, X_test, z_train, z_test, lamb)
        MSE_Ridge[i] = MSE_test

        
        MSE_train_mean, MSE_train_STD, MSE_test_mean, MSE_test_STD = Cross_Validation(X, z, k, 2, lamb)  
        MSE_LASSO_CV[i] = MSE_test_mean
        MSE_LASSO_CV_STD[i] = MSE_test_STD

        # LASSO without CV
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)
        X_train, X_test, z_train, z_test = scale_data(X_train, X_test, z_train, z_test)
        beta, MSE_train, MSE_test, R2_train, R2_test = LASSO_fit(X_train, X_test, z_train, z_test, lamb)
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
    plt.savefig(f"Figures/CV_OLS.pdf")

    plt.figure(figsize=(10, 6))
    plt.title(fr"MSE of Ridge regression with and without CV and $\lambda=${lamb}")
    plt.plot(degrees, MSE_Ridge, label="Without CV")
    plt.errorbar(degrees, MSE_Ridge_CV, MSE_Ridge_CV_STD, label="With CV", capsize=5)
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"Figures/CV_Ridge_lambda={lamb}.pdf")

    plt.figure(figsize=(10, 6))
    plt.title(fr"MSE of LASSO regression with and without CV and $\lambda=${lamb}")
    plt.plot(degrees, MSE_LASSO, label="Without CV")
    plt.errorbar(degrees, MSE_LASSO_CV, MSE_LASSO_CV_STD, label="With CV", capsize=5)
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"Figures/CV_LASSO_lambda={lamb}.pdf")