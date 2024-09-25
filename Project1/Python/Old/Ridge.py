import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import *


# Setup
latex_fonts()
save = False 
deg_max = 8
lmbdas = [1e-10, 1e-5, 1e-2, 1e-1, 1, 10]
franke = Franke(100, 0.1)
data = [franke.x, franke.y, franke.z]

# Training
RIDGE = PolynomialRegression(Ridge_fit, deg_max, data, lmbdas=lmbdas)
MSE_train, MSE_test = RIDGE.MSE()
R2_train, R2_test = RIDGE.R2()
beta = RIDGE.beta
degrees = RIDGE.degrees

################ PLOT ################
# cmap = cm.get_cmap('tab10', len(lmbdas))
cmap = plt.colormaps["tab10"]


################ MSE-plot ################
plt.figure(figsize=(10, 6))

for l, i in zip(lmbdas, range(len(lmbdas))):
    plt.plot(degrees, MSE_train[:,i], lw=2.5, label=rf"$\lambda = {l:.2e}$", color=cmap(i))
    plt.plot(degrees, MSE_test[:,i], color=cmap(i), lw=2.5, linestyle='--')

plt.legend()
plt.xlabel(r'Degree')
plt.ylabel(r'MSE')
plt.xlim(1, deg_max)
plt.title(rf"RIDGE MSE")
plt.grid(True)
if save:
    plt.savefig(f'Figures/RIDGE_MSE.pdf')

################ R²-plot ################
plt.figure(figsize=(10, 6))
for l, i in zip(lmbdas, range(len(lmbdas))):
    plt.plot(degrees, R2_train[:,i], label=rf"$\lambda = {l:.2e}$", lw=2.5, color=cmap(i))
    plt.plot(degrees, R2_test[:,i], lw=2.5, linestyle='--', color=cmap(i))

plt.legend()
plt.xlabel(r'Degree')
plt.ylabel(r'$R^2$')
plt.xlim(1, deg_max)
plt.title(rf"RIDGE $R^2$")
plt.grid(True)
if save:
    plt.savefig(f'Figures/RIDGE_R2.pdf')

lambda_exp_start = -10
lambda_exp_stop = -1
lambda_num = 100

lmbdas = np.logspace(lambda_exp_start, lambda_exp_stop, num=lambda_num)

MSE_train_array = np.zeros(lambda_num)
MSE_test_array = np.zeros(lambda_num)
R2_train_array = np.zeros(lambda_num)
R2_test_array = np.zeros(lambda_num)
beta_list = [0]*lambda_num

X = Design_Matrix(franke.x, franke.y, deg_max)
# Split into training and testing and scale
X_train, X_test, z_train, z_test = train_test_split(X, franke.z, test_size=0.25, random_state=42)
X_train, X_test, z_train, z_test = scale_data(X_train, X_test, z_train, z_test)

for i in range(lambda_num):
    beta_list[i], MSE_train_array[i], MSE_test_array[i], R2_train_array[i], R2_test_array[i] = Ridge_fit(X_train, X_test, z_train, z_test, lmbdas[i])

plt.figure(figsize=(10, 6))
plt.title(rf"MSE deg {deg_max}.")
plt.plot(np.log10(lmbdas), MSE_train_array, label="MSE train", lw=2.5)
plt.plot(np.log10(lmbdas), MSE_test_array, label="MSE test", lw=2.5)
plt.xlabel(r"$\log_{10}(\lambda)$")
plt.ylabel("MSE")
plt.xlim(lambda_exp_start, lambda_exp_stop)
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.title(rf"$R^2$ with deg {deg_max}.")
plt.plot(np.log10(lmbdas), R2_train_array,label=r"$R^2$ train", lw=2.5)
plt.plot(np.log10(lmbdas), R2_test_array,label=r"$R^2$ test", lw=2.5)
plt.xlabel(r"$\log_{10}(\lambda)$")
plt.ylabel(r"$R^2$")
plt.xlim(lambda_exp_start, lambda_exp_stop)
plt.legend()
plt.grid(True)

plt.show()