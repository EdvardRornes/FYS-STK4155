import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter

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
deg_max = 17
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


################ Beta-plot ################
p_stop = 8 # Where to stop the left plot
minor_ticks = np.arange(0, len(beta[-1]), 1)
major_ticks = [np.math.factorial(n + 2)/(np.math.factorial(2)* np.math.factorial(n)) for n in range(deg_max)] # Poly-degree
yticks = np.arange(-1, 6, 1)
fig, axs = plt.subplots(1,2, figsize=(12,6))
fig.suptitle(r'$\beta$-coefficient for various polynomial degrees $p$ (Franke)')

colors = plt.colormaps["RdYlGn_r"]


for i in range(len(beta)):
    N_tmp = len(beta[i])
    num = range(N_tmp)
    deg = i + 1

    if i <= 7:
        axs[0].plot(num, beta[i], color=colors(i / (len(beta) - 1)), lw=2.5, alpha=0.7, marker='o')
    axs[1].plot(num, beta[i], color=colors(i / (len(beta) - 1)), lw=2.5, alpha=0.7, marker='o')

# Colorbar:
norm = mcolors.Normalize(vmin=0, vmax=deg_max) 
sm = cm.ScalarMappable(cmap=colors, norm=norm)
cbar = plt.colorbar(sm, ax=axs[1])
cbar.set_ticks(np.arange(0, deg_max + 1, 2)) #Ensures integer steps
cbar.set_label('Poly degree')

for ax in axs:
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))


# axs[0].set_ylim(y0[0], y0[1])
axs[0].set_xlim(0, np.math.factorial(p_stop + 2)/(np.math.factorial(2)* np.math.factorial(p_stop)))
axs[0].set_xlabel(r"$\beta_n$")
axs[0].set_ylabel(r"$\beta$")

axs[1].yaxis.set_visible(False)
y0 = axs[1].get_ylim()
axs[1].vlines(major_ticks, y0[0], y0[1], colors="black", alpha=0.3)
# axs[1].set_ylim(y0[0], y0[1])
axs[1].set_xlabel(r"$\beta_n$")
axs[1].set_ylabel(r"$\beta$")

axs[0].set_ylim(y0[0], y0[1])
axs[0].vlines(major_ticks, y0[0], y0[1], colors="black", alpha=0.3)
plt.tight_layout()



if save:
    save_plt(f"{folder}/OLS_beta_{additional_description}", overwrite=overwrite)

############### MSE-plot ###############
plt.figure(figsize=(10, 6))
line = plt.plot(degrees, MSE_train, label=r"MSE train", lw=2.5)
color = line[0].get_color()
plt.plot(degrees, MSE_test, "--", color=color, label=r"MSE test", lw=2.5)
plt.xlabel(r'Degree')
plt.xlim(1, deg_max)
# plt.ylim(0, np.max([np.max(x) for x in [MSE_test[0], MSE_test[1], MSE_test[2], MSE_test[3], MSE_test[4]]]))
plt.yscale("log")
plt.ylabel(r'$\log_{10}(MSE)$')
# plt.ylabel(r'MSE')
plt.title(r"OLS MSE (Franke)")
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
plt.xlim(1, deg_max)
# plt.ylim(9/10*np.min([np.min(x) for x in [R2_test[0], R2_test[1], R2_test[2], R2_test[3], R2_test[4]]]), 
#          11/10*np.max([np.max(x) for x in [R2_train[0], R2_train[1], R2_train[2], R2_train[3], R2_train[4]]]))

# plt.yscale("log")
plt.ylabel(r'$\log_{10}(R^2)$')
plt.ylabel(r'$R^2$')

plt.title(r"OLS MSE (Franke)")
plt.title(r"OLS $R^2 (Franke)$")
plt.legend()
plt.grid(True)
if save:
    save_plt(f"{folder}/OLS_R2_{additional_description}", overwrite=overwrite)
plt.show()