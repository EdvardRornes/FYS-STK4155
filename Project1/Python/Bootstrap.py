import numpy as np
import matplotlib.pyplot as plt
from utils import *
from sklearn.utils import resample

latex_fonts()
save = True; overwrite = True
folder = "Figures/OLS"
additional_description = "no_scaling"
# additional_description = "MINMAX"
# additional_description = "StandardScaling"

np.random.seed(42)
N = 200; eps = 0.1
franke = Franke(N, eps)
x,y,z = franke.x, franke.y, franke.z
data = [x,y,z]

samples = 100
deg_max = 20

BOOTSTRAP = PolynomialRegression("OLS", deg_max, data, scaling=additional_description, start_training=False)

error, bias, variance = BOOTSTRAP.Bootstrap(x, y, z, deg_max, samples)
degrees = np.arange(1, deg_max+1)

plt.figure(figsize=(10, 6))
plt.title("Bias and Variance of OLS reg")
plt.plot(degrees, bias, label="Bias", lw=2.5)
plt.plot(degrees, variance, label="Var", lw=2.5)
plt.plot(degrees, error, label="error", lw=2.5)
plt.xlabel("Degree")
plt.grid(True)
plt.legend()
if save:
    save_plt(f"{folder}/OLS_Bootstrap_BiasVar_{additional_description}_GOOD", overwrite=overwrite)

plt.show()