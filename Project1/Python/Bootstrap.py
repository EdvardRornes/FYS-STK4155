import numpy as np
import matplotlib.pyplot as plt

from utils import *

save = True 
latex_fonts()

N = 100; eps = 0.1
franke = Franke(N, eps)
x,y,z = franke.x, franke.y, franke.z
data = [x,y,z]

samples = 15
deg_max = 10
degrees = np.arange(1, deg_max+1)
MSE_train_mean = np.zeros(len(degrees))
MSE_test_mean  = np.zeros(len(degrees))
MSE_train_std  = np.zeros(len(degrees))
MSE_test_std   = np.zeros(len(degrees))

for i in range(deg_max):
    X = Design_Matrix(x, y, degrees[i])
    MSE_train_mean[i], MSE_train_std[i], MSE_test_mean[i], MSE_test_std[i] = Bootstrap(X, z, samples)

plt.figure(figsize=(10, 6))
plt.title("OLS reg MSE with bootstrap and std errorbars")
plt.errorbar(degrees, MSE_train_mean, MSE_train_std, label="MSE for training data", capsize=5)
plt.errorbar(degrees, MSE_test_mean, MSE_test_std, label="MSE for test data", capsize=5)
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.grid(True)
plt.legend()
if save:
    plt.savefig("Figures/Bootstrap_OLS.pdf")

plt.show()
