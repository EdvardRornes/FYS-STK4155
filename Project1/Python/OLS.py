import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import *

# Plot
latex_fonts()
save = True; overwrite = True
folder = "Figures/OLS"

################ Scaling options ################
additional_description = "no_scaling"
# additional_description = "MINMAX"
# additional_description = "StandardScaling"

# Setup
deg_max = 15
N = 100; eps = 0.1
franke = Franke(N, eps)
data = [franke.x, franke.y, franke.z]

# Regression
OLS = PolynomialRegression("OLS", deg_max, data, scaling=additional_description)
MSE_train, MSE_test = OLS.MSE_train, OLS.MSE_test
R2_train, R2_test = OLS.R2_train, OLS.R2_test
beta = OLS.beta

degrees = OLS.degrees

################ PLOT ################
minor_ticks = np.arange(0, len(beta[-1]), 1)
major_ticks = [n*(n+1)/2 for n in range(deg_max)]
yticks = np.arange(-1, 6, 1)
_, ax = plt.subplots(figsize=(12, 6)) # Wider so x-axis is less clustered, makes the font a little bit small in latex though
colors = plt.colormaps["RdYlGn_r"]

################ Beta-plot ################
for i in range(len(beta)):
    N = len(beta[i])
    num = range(N)
    deg = i + 1
    tmp = beta[i]

    color = colors(i / (len(beta) - 1))
    ax.plot(num, tmp, color=color, lw=2.5, alpha=0.7, marker='o')

# Colorbar:
norm = mcolors.Normalize(vmin=0, vmax=len(beta)-1) 
sm = cm.ScalarMappable(cmap=colors, norm=norm)
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Poly degree')

plt.title(r'$\beta$ coefficient dependence for various polynomial degrees $p$')
y0 = ax.get_ylim()
plt.vlines(major_ticks, y0[0], y0[1], colors="black", alpha=0.3)
plt.ylim(y0[0], y0[1])
plt.xlim(0, deg_max*2)
ax.set_xlabel(r"$\beta_n$")
ax.set_ylabel(r"$\beta$")
if save:
    save_plt(f"{folder}/OLS_beta_{additional_description}", overwrite=overwrite)

############### MSE-plot ###############
plt.figure(figsize=(10, 6))
line = plt.plot(degrees, MSE_train, label=r"MSE train", lw=2.5)
color = line[0].get_color()
plt.plot(degrees, MSE_test, "--", color=color, label=r"MSE test", lw=2.5)
plt.xlabel(r'Degree')
plt.ylabel(r'MSE')
plt.xlim(1, deg_max)
plt.title(r"OLS MSE")
plt.legend()
plt.grid(True)
if save:
    save_plt(f"{folder}/OLS_MSE_{additional_description}", overwrite=overwrite)

############### R²-plot ###############
plt.figure(figsize=(10, 6))
line = plt.plot(degrees, R2_train, label=r"$R^2$ train", lw=2.5)
color = line[0].get_color()
plt.plot(degrees, R2_test, "--", color=color, label=r"$R^2$ test", lw=2.5)
plt.xlabel(r'Degree')
plt.ylabel(r'$R^2$')
plt.xlim(1, deg_max)
plt.title(r"OLS $R^2$")
plt.legend()
plt.grid(True)
if save:
    save_plt(f"{folder}/OLS_R2_{additional_description}", overwrite=overwrite)
plt.show()