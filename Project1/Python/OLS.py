import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math

from matplotlib.ticker import ScalarFormatter
from utils import *

# Plot
latex_fonts()
save = True; overwrite = True
folder = "Figures/OLS"

################ Scaling options ################
additional_description = "Unscaled"
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
major_ticks = [math.factorial(n + 2)/(math.factorial(2)* math.factorial(n)) for n in range(deg_max)] # Poly-degree
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

axs[0].set_xlim(0, math.factorial(p_stop + 2)/(math.factorial(2)* math.factorial(p_stop)))
axs[0].set_xlabel(r"$\beta_n$")
axs[0].set_ylabel(r"$\beta$")

axs[1].yaxis.set_visible(False)
y0 = axs[1].get_ylim()
axs[1].vlines(major_ticks, y0[0], y0[1], colors="black", alpha=0.3)
axs[1].set_xlabel(r"$\beta_n$")
axs[1].set_ylabel(r"$\beta$")

axs[0].set_ylim(y0[0], y0[1])
axs[0].vlines(major_ticks, y0[0], y0[1], colors="black", alpha=0.3)
plt.tight_layout()

if save:
    save_plt(f"{folder}/OLS_beta_{additional_description}", overwrite=overwrite)

############### MSE & R2 plot ###############
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

line = axs[0].plot(degrees, MSE_train, label=r"MSE train", lw=2.5)
color = line[0].get_color()
axs[0].plot(degrees, MSE_test, "--", color=color, label=r"MSE test", lw=2.5)
axs[0].set_xlabel(r'Degree')
axs[0].set_xlim(1, deg_max)
axs[0].set_yscale("log")
axs[0].set_ylabel(r'MSE')
axs[0].set_title(rf"OLS MSE as a function of polynomial degree (Franke)")
axs[0].legend()
axs[0].grid(True)

line = axs[1].plot(degrees, R2_train, label=r"$R^2$ train", lw=2.5)
color = line[0].get_color()
axs[1].plot(degrees, R2_test, "--", color=color, label=r"$R^2$ test", lw=2.5)
axs[1].set_xlabel(r'Degree')
axs[1].set_xlim(1, deg_max)
axs[1].set_ylabel(r'$R^2$')
axs[1].set_title(rf"OLS $R^2$ as a function of polynomial degree (Franke)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)
if save:
    save_plt(f"{folder}/OLS_MSE_R2_{additional_description}", overwrite=overwrite)

# Show the plot
plt.show()
