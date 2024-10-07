import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.model_selection import train_test_split
from utils import *

# Plot
np.random.seed(0)
latex_fonts()
save = True; overwrite = True
folder = "Figures/Ridge"

################ Scaling options ################
additional_description = "Unscaled"
# additional_description = "MINMAX"
# additional_description = "StandardScaling"

# Setup
deg_max = 17
lmbdas = [1e-10, 1e-7, 1e-4, 1e-1]
N = 50; eps = 0.1
franke = Franke(N, eps)
data = [franke.x, franke.y, franke.z]

# Training
RIDGE = PolynomialRegression("Ridge", deg_max, data, lmbdas=lmbdas, scaling=additional_description)
MSE_train, MSE_test = RIDGE.MSE_train, RIDGE.MSE_test
R2_train, R2_test = RIDGE.R2_train, RIDGE.R2_test
beta = RIDGE.beta
degrees = RIDGE.degrees

################ PLOT ################
# cmap = cm.get_cmap('tab10', len(lmbdas))
cmap = plt.colormaps["tab10"]


fig, axs = plt.subplots(2, 1, figsize=(10, 12))

################ MSE-plot ################
for l, i in zip(lmbdas, range(len(lmbdas))):
    axs[0].plot(degrees, MSE_train[:, i], lw=2.5, label=rf"$\lambda = {l:.2e}$", color=cmap(i))
    axs[0].plot(degrees, MSE_test[:, i], color=cmap(i), lw=2.5, linestyle='--')

train_line = mlines.Line2D([], [], color='black', lw=2.5, label='Train')
test_line = mlines.Line2D([], [], color='black', lw=2.5, linestyle='--', label='Test')
handles, labels = axs[0].get_legend_handles_labels()
handles.append(train_line)
handles.append(test_line)
labels.append("Train")
labels.append("Test")
axs[0].legend(handles=handles, labels=labels)

axs[0].set_xlabel(r'Degree')
axs[0].set_ylabel(r'MSE')
axs[0].set_xlim(1, deg_max)
axs[0].set_ylim(1e-3, 1e-1)
axs[0].set_yscale('log')
axs[0].set_title(rf"RIDGE MSE as a function of polynomial degree (Franke)")
axs[0].grid(True)

################ R²-plot ################
for l, i in zip(lmbdas, range(len(lmbdas))):
    axs[1].plot(degrees, R2_train[:, i], label=rf"$\lambda = {l:.2e}$", lw=2.5, color=cmap(i))
    axs[1].plot(degrees, R2_test[:, i], lw=2.5, linestyle='--', color=cmap(i))

# Use the same legend as before
axs[1].legend(handles=handles, labels=labels)

axs[1].set_xlabel(r'Degree')
axs[1].set_ylabel(r'$R^2$')
axs[1].set_xlim(1, deg_max)
axs[1].set_ylim(0.7, 1)
axs[1].set_title(rf"RIDGE $R^2$ as a function of polynomial degree (Franke)")
axs[1].grid(True)

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)

if save:
    save_plt(f"{folder}/RIDGE_MSE_R2_{additional_description}", overwrite=overwrite)

# Setup
log_lambda_start = -10
log_lambda_stop = -1
lambda_num = 100
deg_analysis = 6

lmbdas = np.logspace(log_lambda_start, log_lambda_stop, lambda_num)

MSE_train_array = np.zeros(lambda_num)
MSE_test_array = np.zeros(lambda_num)
R2_train_array = np.zeros(lambda_num)
R2_test_array = np.zeros(lambda_num)
beta_list = [0]*lambda_num

RIDGE = PolynomialRegression("Ridge", deg_analysis, data, start_training=False)
X = PolynomialRegression.Design_Matrix(franke.x, franke.y, deg_analysis)
# Split into training and testing and scale
X_train, X_test, z_train, z_test = train_test_split(X, franke.z, test_size=0.25, random_state=4)
X_train, X_test, z_train, z_test = PolynomialRegression.scale_data(X_train, X_test, z_train, z_test)

for i in range(lambda_num):
    beta_list[i], MSE_train_array[i], MSE_test_array[i], R2_train_array[i], R2_test_array[i] = RIDGE.Ridge_fit(X_train, X_test, z_train, z_test, lmbdas[i])


fig, axs = plt.subplots(2, 1, figsize=(10, 12))

################ MSE-plot ################
axs[0].set_title(rf"Ridge MSE on Franke function with polynomial degree $p={deg_analysis}$")
axs[0].plot(np.log10(lmbdas), MSE_train_array, label="MSE train", lw=2.5)
axs[0].plot(np.log10(lmbdas), MSE_test_array, label="MSE test", lw=2.5)
axs[0].set_xlabel(r"$\log_{10}\lambda$")
axs[0].set_ylabel("MSE")
axs[0].set_xlim(log_lambda_start, log_lambda_stop)
axs[0].legend()
axs[0].grid(True)

################ R²-plot ################
axs[1].set_title(rf"Ridge $R^2$ on Franke function with polynomial degree $p={deg_analysis}$")
axs[1].plot(np.log10(lmbdas), R2_train_array, label=r"$R^2$ train", lw=2.5)
axs[1].plot(np.log10(lmbdas), R2_test_array, label=r"$R^2$ test", lw=2.5)
axs[1].set_xlabel(r"$\log_{10}\lambda$")
axs[1].set_ylabel(r"$R^2$")
axs[1].set_xlim(log_lambda_start, log_lambda_stop)
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)

if save:
    save_plt(f"{folder}/RIDGE_logMSE_R2_{additional_description}", overwrite=overwrite)

plt.show()
