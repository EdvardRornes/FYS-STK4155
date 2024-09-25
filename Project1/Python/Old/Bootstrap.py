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

samples = 15
deg_max = 10
degrees = np.arange(1, deg_max+1)
MSE_train_mean = np.zeros(len(degrees))
MSE_test_mean  = np.zeros(len(degrees))
MSE_train_std  = np.zeros(len(degrees))
MSE_test_std   = np.zeros(len(degrees))

for i in range(deg_max):
    X = Design_Matrix(x, y, degrees[i])
    MSE_train_mean[i], MSE_train_std[i], MSE_test_mean[i], MSE_test_std[i] = Bootstrap_OLS(X, z, samples, 0.25)

plt.figure(figsize=(10, 6))
plt.title("OLS reg MSE with bootstrap and std errorbars")
plt.errorbar(degrees, MSE_train_mean, MSE_train_std, label="MSE for training data", capsize=5)
plt.errorbar(degrees, MSE_test_mean, MSE_test_std, label="MSE for test data", capsize=5)
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.grid(True)
plt.legend()
plt.savefig("Figures/Bootstrap_OLS.pdf")