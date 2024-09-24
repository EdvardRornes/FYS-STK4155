import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from sklearn.model_selection import train_test_split
from utils import *
import time

# Latex fonts
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlepad'] = 25 

plt.rcParams.update({
    'font.family' : 'euclid',
    'font.weight' : 'bold',
    'font.size': 20,       # General font size
    'axes.labelsize': 20,  # Axis label font size
    'axes.titlesize': 25,  # Title font size
    'xtick.labelsize': 25, # X-axis tick label font size
    'ytick.labelsize': 25, # Y-axis tick label font size
    'legend.fontsize': 20, # Legend font size
    'figure.titlesize': 30 # Figure title font size
})

# Generate data
# np.random.seed(2024)
N = 50  # Number of data points
x = np.sort(np.random.rand(N))
y = np.sort(np.random.rand(N))
z = Franke(x, y)
z = z + 0.1 * np.random.normal(N, 1, z.shape)  # Add some noise to the data

deg_max = 8
degrees = np.arange(1, deg_max+1)
MSE_train = np.zeros(len(degrees))
MSE_test = np.zeros(len(degrees))
R2_train = np.zeros(len(degrees))
R2_test = np.zeros(len(degrees))
beta_coefficients = [0]*deg_max

start_time = time.time()
for deg in range(deg_max):
    # Create polynomial features
    X = Design_matrix2D(x, y, degrees[deg])
    
    # Split into training and testing and scale
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25, random_state=42)
    X_train, X_test, z_train, z_test = scale_data(X_train, X_test, z_train, z_test)# , scaler_type="MINMAX")

    beta_coefficients[deg], MSE_train[deg], MSE_test[deg], R2_train[deg], R2_test[deg] = OLS_fit(X_train, X_test, z_train, z_test)

    print(f"p={deg}/{deg_max}, duration: {(time.time()-start_time):.2f}s", end="\r")

print(f"p={deg_max}/{deg_max}, duration: {(time.time()-start_time):.2f}s")

############### Plotting the beta's ############### 

# x-axis ticks:
minor_ticks = np.arange(0, len(beta_coefficients[-1]), 1)
major_ticks = [n*(n+1)/2 for n in range(deg_max)]
yticks = np.arange(-1, 6, 1)
_, ax = plt.subplots(figsize=(12, 6)) # Wider so x-axis is less clustered, makes the font a little bit small in latex though

colors = cm.get_cmap('RdYlGn_r', len(beta_coefficients))  # color grading for poly degree from green to red

for i in range(len(beta_coefficients)):
    N = len(beta_coefficients[i])
    num = range(N)
    deg = i + 1
    # tmp = np.log10(np.abs(beta_coefficients[i]) + 1e-1)
    tmp = beta_coefficients[i]

    ax.plot(num, tmp, color=colors(i), lw=2.5, alpha=0.7)#label=rf"$p={deg}$", lw=2.5)

# ax.set_xticks(minor_ticks)
# ax.set_yticks(yticks)
# ax.grid(which="both", axis="x")

# Colorbar:
norm = mcolors.Normalize(vmin=0, vmax=len(beta_coefficients)-1) 
sm = cm.ScalarMappable(cmap=colors, norm=norm)
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Poly degree')

plt.title(r'$\beta$ coefficient dependence for various polynomial degrees $p$')
y0 = ax.get_ylim()
plt.vlines(major_ticks, y0[0], y0[1], colors="black", alpha=0.3)
plt.ylim(y0[0], y0[1])
plt.xlim(0, N-1)
ax.set_xlabel(r"$\beta_n$")
ax.set_ylabel(r"$\beta$")
# ax.set_ylabel(r"$\log_{10}|\beta + 0.1|$")
# plt.legend(loc="lower right")
plt.savefig(f'Figures/OLS-beta-degree.pdf')

############### Plot MSE ###############
plt.figure(figsize=(10, 6))
plt.plot(degrees, MSE_train, label=r"MSE train", lw=2.5)
plt.plot(degrees, MSE_test, label=r"MSE test", lw=2.5)
plt.xlabel(r'Degree')
plt.ylabel(r'MSE')
plt.xlim(1, deg_max)
plt.title(r"OLS MSE")
plt.legend()
plt.grid(True)
plt.savefig(f'Figures/OLS-MSE-degree.pdf')

############### Plot R² ###############
plt.figure(figsize=(10, 6))
plt.plot(degrees, R2_train, label=r"$R^2$ train", lw=2.5)
plt.plot(degrees, R2_test, label=r"$R^2$ test", lw=2.5)
plt.xlabel(r'Degree')
plt.ylabel(r'$R^2$')
plt.xlim(1, deg_max)
plt.title(r"OLS $R^2$")
plt.legend()
plt.grid(True)
plt.savefig(f'Figures/OLS-R2-degree.pdf')
plt.show()