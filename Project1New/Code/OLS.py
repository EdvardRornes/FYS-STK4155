import numpy as np
import matplotlib.pyplot as plt
from utils import *

N=100
x = np.sort(np.random.rand(N))
y = np.sort(np.random.rand(N))
z = franke(x,y)
z = z + 0.1*np.random.normal(0, 1, z.shape)

deg_max = 50
degs = np.linspace(1, deg_max, deg_max, dtype=int)

MSE_train_array = np.zeros(deg_max)
MSE_test_array = np.zeros(deg_max)
R2_train_array = np.zeros(deg_max)
R2_test_array = np.zeros(deg_max)
betas = [0]*deg_max

for i in range(deg_max):
    X = Design_Matrix(x, y, degs[i])
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)
    
    betas[i], MSE_train_array[i], MSE_test_array[i], R2_train_array[i], R2_test_array[i] = OLS_fit(X_train, X_test, z_train, z_test)

plt.figure()
plt.title(f"NEW OLS MSE")
plt.plot(degs,MSE_train_array,label="MSE-train")
plt.plot(degs,MSE_test_array,label="MSE-test")
plt.xlabel("degree")
plt.ylabel("MSE")
plt.yticks()
plt.xticks()
plt.legend()

plt.figure()
plt.title(f"NEW OLS R2")
plt.plot(degs,R2_train_array,label="R2-train")
plt.plot(degs,R2_test_array,label="R2-test")
plt.xlabel("degree")
plt.ylabel("R2-score")
plt.yticks()
plt.xticks()
plt.legend()
plt.show()