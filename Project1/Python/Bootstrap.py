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

samples = 250
deg_max = 10

BOOTSTRAP = PolynomialRegression("OLS", deg_max, data, scaling=additional_description, start_training=False)

error, bias, variance = BOOTSTRAP.Bootstrap(x, y, z, deg_max, samples)
degrees = np.arange(1, deg_max+1)

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