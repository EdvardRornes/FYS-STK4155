import numpy as np
import matplotlib.pyplot as plt

from utils import *

np.random.seed(1) # 9

# latex_fonts()
save = True; overwrite = True
folder = "Figures/LASSO"
additional_description = "no_scaling"
# additional_description = "MINMAX"
# additional_description = "StandardScaling"

N = 50; eps = 0.1
franke = Franke(N, eps)
x,y,z = franke.x, franke.y, franke.z
data = [x,y,z]

samples = 10
deg_max = 10
degrees = np.arange(1, deg_max+1)
MSE_train_mean = np.zeros(len(degrees))
MSE_test_mean  = np.zeros(len(degrees))
MSE_train_std  = np.zeros(len(degrees))
MSE_test_std   = np.zeros(len(degrees))

plt.figure(figsize=(10, 6))
ymin, ymax = 1e-5, 1e2
plt.title("OLS Regression MSE with Bootstrap and Standard Error Bars")

def errorbar_with_caution(degrees: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, ymin: float, ymax: float, label="") -> None:
    # Calculate lower and upper errors
    lower_err = y_std.copy()
    upper_err = y_std.copy()

    # Identify where the lower error bar would go below ymin
    uplims = (y_mean - lower_err) < ymin
    # Identify where the upper error bar would go above ymax
    lolims = (y_mean + upper_err) > ymax

    # Adjust lower errors where they would go below ymin
    lower_err[uplims] = y_mean[uplims] - ymin  # Error bar ends at ymin

    # Adjust upper errors where they would go above ymax
    upper_err[lolims] = ymax - y_mean[lolims]  # Error bar ends at ymax

    print("Lower errors:", lower_err)
    print("Upper errors:", upper_err)
    
    lower_err = np.maximum(lower_err, 0)
    upper_err = np.maximum(upper_err, 0)


    print("Lower errors:", lower_err)
    print("Upper errors:", upper_err)

    # Plot error bars with limits
    plt.errorbar(
        degrees,
        y_mean,
        yerr=[lower_err, upper_err],
        fmt='o',
        capsize=5,
        label=label,
        uplims=uplims,
        lolims=lolims
    )

# Call the function for test data
errorbar_with_caution(degrees, MSE_test_mean, MSE_test_std, ymin, ymax, label="MSE for test data")

# Uncomment the following line to include training data as well
# errorbar_with_caution(degrees, MSE_train_mean, MSE_train_std, ymin, ymax, label="MSE for training data")

plt.yscale('log')
plt.ylim(ymin, ymax)
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.grid(True)
plt.legend()

# Save the figure if needed
# if save:
#     plt.savefig("Figures/Bootstrap_OLS.pdf")

plt.show()