
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# a)
np.random.seed()
n = 100
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5*np.exp(-(x - 2)**2) + np.random.normal(0, 0.1, x.shape)

poly_5 = PolynomialFeatures(degree = 5)
X = poly_5.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

# b)
# model = LinearRegression()
# model.fit(X_train, y_train)

# y_tilde_train = model.predict(X_train)
# y_tilde_test  = model.predict(X_test)

# MSE_train = mean_squared_error(y_train, y_tilde_train)
# MSE_test  = mean_squared_error(y_test, y_tilde_test)

# c)
poly_deg = 50
MSE_train_arr = np.zeros(poly_deg)
MSE_test_arr = np.zeros(poly_deg)
degrees = np.arange(1, poly_deg+1)

for deg in degrees:
    poly15 = PolynomialFeatures(degree=deg)
    X = poly15.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_tilde_train = model.predict(X_train)
    y_tilde_test = model.predict(X_test)

    MSE_train_arr[deg-1] = mean_squared_error(y_train, y_tilde_train)
    MSE_test_arr[deg-1] = mean_squared_error(y_test, y_tilde_test)

plt.plot(degrees, MSE_train_arr, label="MSE_train")
plt.plot(degrees, MSE_test_arr, label="MSE_test")
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.legend()
plt.savefig('mse.pdf')
plt.show()
