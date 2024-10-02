import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from utils import * 
np.random.seed(2018)

n = 40
n_boostraps = 100
maxdegree = 14


N = 50; eps = 0.1
franke = Franke(N, eps)
x,y,z = franke.x, franke.y, franke.z
data = [x,y,z]


# Make data set.
# x = np.linspace(-3, 3, n).reshape(-1, 1)
# y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)
error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

for degree in range(maxdegree):
    TEST = PolynomialRegression("OLS", degree, data, start_training=False)
    X = PolynomialRegression.Design_Matrix(x, y, degree)
    X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.2)
    # beta, _, _, _, _ = TEST.regr_model(X_train, X_test, y_train, y_train, None)
    # model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
    y_test = y_test.reshape(-1,1)
    y_pred = np.empty((y_test.shape[0], n_boostraps))
    error_i = []
    for i in range(n_boostraps):
        x_, y_ = resample(X_train, y_train)
        beta, _, _, _, _ = TEST.regr_model(x_, X_test, y_, y_test, None)
        # y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()
        tmp = X_test @ beta#; print(np.shape(tmp))
        y_pred[:, i] = tmp[:,0]
        error_i.append(np.mean((y_test - y_pred[:,i])**2))

    polydegree[degree] = degree
    print(np.shape(y_test), np.shape(y_pred))
    error[degree] = np.mean( error_i)
    bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
    variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )
    # print('Polynomial degree:', degree)
    # print('Error:', error[degree])
    # print('Bias^2:', bias[degree])
    # print('Var:', variance[degree])
    # print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.yscale("log")
plt.legend()
plt.show()