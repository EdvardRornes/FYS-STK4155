import numpy as np
import matplotlib.pyplot as plt

from utils import *
from sklearn.utils import resample

np.random.seed(7)

# latex_fonts()
save = False; overwrite = False
folder = "Figures/OLS"
additional_description = "no_scaling"
# additional_description = "MINMAX"
# additional_description = "StandardScaling"

N = 50; eps = 0.1
franke = Franke(N, eps)
x,y,z = franke.x, franke.y, franke.z
data = [x,y,z]

samples = 100
deg_max = 10
degrees = np.arange(1, deg_max+1)
BOOTSTRAP = PolynomialRegression("OLS", deg_max, data, scaling=additional_description, start_training=False)
MSE_train_mean = np.zeros(len(degrees))
MSE_test_mean  = np.zeros(len(degrees))
MSE_train_std  = np.zeros(len(degrees))
MSE_test_std   = np.zeros(len(degrees))
bias = []; variance = []

start_time = time.time()
# for i in range(deg_max):
#     X = PolynomialRegression.Design_Matrix(x, y, degrees[i])
#     MSE_train_mean[i], MSE_train_std[i], MSE_test_mean[i], MSE_test_std[i], y_pred = BOOTSTRAP.Bootstrap(X, franke.z_without_noise, samples, "OLS")

    # print(f"{i/deg_max*100:.1f}%, duration: {(time.time()-start_time):.2f}s", end="\r")

# X = PolynomialRegression.Design_Matrix(x, y, degrees[i])
error, bias, variance = BOOTSTRAP.Bootstrap(x, y, z, deg_max, samples)

plt.figure(figsize=(10, 6))
plt.title("Bias and Variance of OLS reg")
plt.plot(degrees, bias, label="Bias")
plt.plot(degrees, variance, label="Var")
plt.plot(degrees, error, label="error")
# plt.plot(degrees, OLS.MSE_test, label="MSE_OLS")
plt.yscale('log')
# plt.ylim(1e-5, 1e2)
plt.xlabel("Degree")
plt.grid(True)
plt.legend()
if save:
    save_plt(f"{folder}/OLS_Bootstrap_BiasVar_{additional_description}", overwrite=overwrite)

plt.show()

# print(f"100.0%, duration: {(time.time()-start_time):.2f}s")
# # print(np.size(BOOTSTRAP.y_tilde))
# plt.figure(figsize=(10, 6))
# plt.title("OLS reg MSE with bootstrap and std errorbars")
# plt.errorbar(degrees, MSE_train_mean, MSE_train_std, label="MSE for training data", capsize=5)
# plt.errorbar(degrees, MSE_test_mean, MSE_test_std, label="MSE for test data", capsize=5)
# plt.yscale('log')
# plt.ylim(1e-5, 1e2)
# plt.xlabel("Degree")
# plt.ylabel("MSE")
# plt.grid(True)
# plt.legend()
# if save:
#     save_plt(f"{folder}/OLS_Bootstrap_MSE_{additional_description}", overwrite=overwrite)

# plt.show()

# plt.figure(figsize=(10, 6))
# OLS = PolynomialRegression("OLS", deg_max, data, scaling=additional_description)
# MSE_train, MSE_test = OLS.MSE_train, OLS.MSE_test
# R2_train, R2_test = OLS.R2_train, OLS.R2_test
# plt.title("Bias and Variance of OLS reg")
# plt.plot(degrees, bias, label="Bias")
# plt.plot(degrees, variance, label="Var")
# plt.plot(degrees, MSE_test_mean, label="MSE")
# # plt.plot(degrees, OLS.MSE_test, label="MSE_OLS")
# # plt.yscale('log')
# # plt.ylim(1e-5, 1e2)
# plt.xlabel("Degree")
# plt.grid(True)
# plt.legend()
# if save:
#     save_plt(f"{folder}/OLS_Bootstrap_BiasVar_{additional_description}", overwrite=overwrite)

# plt.show()