import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample


def poly_reg(x, p):
    poly = PolynomialFeatures(p)
    X_poly = poly.fit_transform(x[:, np.newaxis])
    
    return X_poly

def Ridge_Reg(X, y, lambd):
    XTX = X.T @ X
    y = y.reshape(-1, 1)
    return np.linalg.pinv(XTX + lambd * np.identity(XTX.shape[0])) @ X.T @ y


# Setup:
np.random.seed(42)
n_samples = 100
x = np.sort(np.random.rand(n_samples))
y = np.sin(2 * np.pi * x) + np.random.random(len(x))
x_test = np.linspace(0, 1, 100)
y_test_true = np.sin(2 * np.pi * x)

p = 15; N = 50
mse_train = np.zeros(p); mse_test = np.zeros(p)
bias2 = np.zeros(p); variance = np.zeros(p)

for i in range(1, p+1):
    y_pred_bootstrap = np.zeros((N, len(x_test)))
    mse_train_bootstrap = []
    
    for j in range(N):
        # Resample
        x_bootstrap, y_bootstrap = resample(x, y)
        
        # Training
        X_poly_bootstrap = poly_reg(x_bootstrap, i)
        beta = Ridge_Reg(X_poly_bootstrap, y_bootstrap, 0)
        
        # Predicting from training and test data

        y_train_pred = X_poly_bootstrap @ beta 
        X_poly_test = PolynomialFeatures(i).fit_transform(x_test[:, np.newaxis])
        X_poly_test = poly_reg(x_test, i)
        y_test_pred = X_poly_test @ beta

        y_pred_bootstrap[j, :] = y_test_pred[0]
        mse_train_bootstrap.append(mean_squared_error(y_bootstrap, y_train_pred))

    # Saving data
    mse_train[i-1] = np.mean(mse_train_bootstrap)
    bias2[i-1] = np.mean((np.mean(y_pred_bootstrap, axis=0) - y_test_true) ** 2)
    variance[i-1] = np.mean(np.var(y_pred_bootstrap, axis=0))


deg = np.arange(1, p+1)

plt.title("Bias-Variance Tradeoff")
plt.plot(deg, bias2, label=r"Bias$^2$")
plt.plot(deg, variance, label="var")
plt.plot(deg, np.array(bias2) + np.array(variance), label=r"Bias$^2$ + var")

plt.xlabel("Model Complexity/poly")
plt.ylabel("Error")
plt.legend()
plt.show()

plt.title("MSE vs Model Complexity")
plt.plot(deg, mse_train, label="Training MSE")

plt.xlabel("Model Complexity/poly")
plt.ylabel("MSE")
plt.legend()
plt.show()