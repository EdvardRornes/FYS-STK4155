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


################ MSE-plot ################
plt.figure(figsize=(10, 6))

for l, i in zip(lmbdas, range(len(lmbdas))):
    plt.plot(degrees, MSE_train[:,i], lw=2.5, label=rf"$\lambda = {l:.2e}$", color=cmap(i))
    plt.plot(degrees, MSE_test[:,i], color=cmap(i), lw=2.5, linestyle='--')

# Make more efficient legend
train_line = mlines.Line2D([], [], color='black', lw=2.5, label='Train')
test_line = mlines.Line2D([], [], color='black', lw=2.5, linestyle='--', label='Test')
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(train_line)
handles.append(test_line)
labels.append("Train")
labels.append("Test")
plt.legend(handles=handles,labels=labels)

plt.xlabel(r'Degree')
plt.ylabel(r'MSE')
plt.xlim(1, deg_max)
plt.ylim(1e-3, 1e-1)
plt.yscale('log')
plt.title(rf"RIDGE MSE (Franke)")
plt.grid(True)
if save:
    save_plt(f"{folder}/RIDGE_MSE_{additional_description}", overwrite=overwrite)

################ R²-plot ################
plt.figure(figsize=(10, 6))
for l, i in zip(lmbdas, range(len(lmbdas))):
    plt.plot(degrees, R2_train[:,i], label=rf"$\lambda = {l:.2e}$", lw=2.5, color=cmap(i))
    plt.plot(degrees, R2_test[:,i], lw=2.5, linestyle='--', color=cmap(i))

# Use same legend as before
plt.legend(handles=handles,labels=labels)
plt.xlabel(r'Degree')
plt.ylabel(r'$R^2$')
plt.xlim(1, deg_max)
plt.ylim(0.7, 1)
plt.title(rf"RIDGE $R^2$ (Franke)")
plt.grid(True)
if save:
    save_plt(f"{folder}/RIDGE_R2_{additional_description}", overwrite=overwrite)

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

plt.figure(figsize=(10, 6))
plt.title(rf"Ridge MSE on Franke function with polynomial degree $p={deg_analysis}$")
plt.plot(np.log10(lmbdas), MSE_train_array, label="MSE train", lw=2.5)
plt.plot(np.log10(lmbdas), MSE_test_array, label="MSE test", lw=2.5)
plt.xlabel(r"$\log_{10}(\lambda)$")
plt.ylabel("MSE")
plt.xlim(log_lambda_start, log_lambda_stop)
plt.legend()
plt.grid(True)
if save:
    save_plt(f"{folder}/RIDGE_logMSE_{additional_description}", overwrite=overwrite)

plt.figure(figsize=(10, 6))
plt.title(rf"Ridge $R^2$ on Franke function with polynomial degree $p={deg_analysis}$")
plt.plot(np.log10(lmbdas), R2_train_array,label=r"$R^2$ train", lw=2.5)
plt.plot(np.log10(lmbdas), R2_test_array,label=r"$R^2$ test", lw=2.5)
plt.xlabel(r"$\log_{10}(\lambda)$")
plt.ylabel(r"$R^2$")
plt.xlim(log_lambda_start, log_lambda_stop)
plt.legend()
plt.grid(True)
if save:
    save_plt(f"{folder}/RIDGE_logR2_{additional_description}", overwrite=overwrite)

plt.show()