import numpy as np
import matplotlib.pyplot as plt
from utils import *

N=100
x = np.sort(np.random.rand(N))
y = np.sort(np.random.rand(N))
z = franke(x,y)
z = z + 0.1*np.random.normal(0, 1, z.shape)


deg = 30 # Polynomial degree
lambda_exp_start = -10
lambda_exp_stop = 0
lambda_num = 1000

lambdas = np.logspace(lambda_exp_start, lambda_exp_stop, num=lambda_num)
MSE_train_array = np.zeros(lambda_num)
MSE_test_array = np.zeros(lambda_num)
R2_train_array = np.zeros(lambda_num)
R2_test_array = np.zeros(lambda_num)
beta_list = [0]*lambda_num
X = Design_Matrix(x, y, deg) # Compute feature matrix
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2) # Split into training and test data

for i in range(lambda_num):
    beta_list[i], MSE_train_array[i], MSE_test_array[i], R2_train_array[i], R2_test_array[i] = Ridge_fit(X_train, X_test, z_train, z_test, lambdas[i])

plt.figure()
plt.title(f"MSE Ridge deg={deg}.", fontsize=20)
plt.plot(np.log10(lambdas),MSE_train_array,label="MSE-train")
plt.plot(np.log10(lambdas),MSE_test_array,label="MSE-test")
plt.xlabel("log10lambda", fontsize=20)
plt.ylabel("MSE", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.legend(fontsize=20)

plt.figure()
plt.title(f"R2 Ridge deg={deg}.", fontsize=20)
plt.plot(np.log10(lambdas),R2_train_array,label="R2-train")
plt.plot(np.log10(lambdas),R2_test_array,label="R2-test")
plt.xlabel("log10lambda", fontsize=20)
plt.ylabel("R2", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.legend(fontsize=20)



deg_max = 10
degs = np.linspace(1, deg_max, deg_max, dtype=int)

MSE_train_array = np.zeros(deg_max)
MSE_test_array = np.zeros(deg_max)
R2_train_array = np.zeros(deg_max)
R2_test_array = np.zeros(deg_max)
betas = [0]*deg_max

for i in range(deg_max):
    X = Design_Matrix(x, y, degs[i])
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)
    
    betas[i], MSE_train_array[i], MSE_test_array[i], R2_train_array[i], R2_test_array[i] = Ridge_fit(X_train, X_test, z_train, z_test, lmbda=1e-2)

plt.figure()
plt.title(f"NEW Ridge MSE", fontsize=20)
plt.plot(degs,MSE_train_array,label="MSE-train")
plt.plot(degs,MSE_test_array,label="MSE-test")
plt.xlabel("degree", fontsize=20)
plt.ylabel("MSE", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.legend(fontsize=20)

plt.figure()
plt.title(f"NEW Ridge R2", fontsize=20)
plt.plot(degs,R2_train_array,label="R2-train")
plt.plot(degs,R2_test_array,label="R2-test")
plt.xlabel("degree", fontsize=20)
plt.ylabel("R2-score", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.legend(fontsize=20)


plt.show()