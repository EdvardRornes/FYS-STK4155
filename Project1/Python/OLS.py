
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import *

np.random.seed()
n = 100
x = np.sort(np.random.rand(n))
y = np.sort(np.random.rand(n))
z = Franke(x, y)
# Noise
noise = 0.1
z = z + noise*np.random.normal(0, 1, z.shape)

degrees = 5
for i in range(degrees):
    poly = PolynomialFeatures(degree = i)
    X = Design_Matrix(x, y, i)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.3)
    X_train, X_test, z_train, z_test = scale_data(X_train, X_test, z_train, z_test)
    MSE_train = MSE(z, z_train)
    MSE_test = MSE(z, z_test)
    R2_train = R2(z, z_train)
    R2_test = R2(z, z_test)


model = LinearRegression()
model.fit(X_train, y_train)

y_tilde_train = model.predict(X_train)
y_tilde_test  = model.predict(X_test)

MSE_train = mean_squared_error(y_train, y_tilde_train)
MSE_test  = mean_squared_error(y_test, y_tilde_test)

# c)
poly_deg      = 15
MSE_train_arr = np.zeros(poly_deg-1)
MSE_test_arr  = np.zeros(poly_deg-1)
degrees       = np.linspace(2, poly_deg, poly_deg-1, dtype=int)

for deg in degrees:
    poly15 = PolynomialFeatures(degree = deg)
    X = poly15.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_tilde_train = model.predict(X_train)
    y_tilde_test  = model.predict(X_test)

    MSE_train_arr[deg-2] = mean_squared_error(y_train, y_tilde_train)
    MSE_test_arr[deg-2]  = mean_squared_error(y_test, y_tilde_test)

plt.plot(degrees, MSE_train_arr, label="MSE_train")
plt.plot(degrees, MSE_test_arr, label="MSE_test")
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.legend()
plt.savefig('mse.pdf')
plt.show()