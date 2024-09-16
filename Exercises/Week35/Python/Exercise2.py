import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Own code

# Coefficient
c = 0.1
x = np.random.rand(100, 1)
y = 2.0 + 5.0*x**2 + c*np.random.randn(100, 1)

n = x.size
p = 3

X = np.zeros((n,p))
for i in range(p):
    X[:,i] = x[:,0]**i
    
XT      = np.transpose(X)
XTX     = np.matmul(XT, X)
inv_XTX = np.linalg.inv(XTX)

b = np.matmul(np.matmul(inv_XTX, XT), y)
b0, b1, b2 = b[:,0]

print(f"Own model:   beta = ({b0}, {b1}, {b2})")

## Scikitlearn

poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(x)
model = LinearRegression()
model.fit(X,y)

y_predicted = model.predict(X)
print(f"Scikitlearn: beta = ({model.intercept_}, {model.coef_[0,1]}, {model.coef_[0,2]})")

## MSE & R2
print(f"MSE: {mean_squared_error(y, y_predicted)}")
print(f"R2:  {r2_score(y, y_predicted)}")


