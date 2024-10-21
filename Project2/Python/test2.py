import numpy as np
import matplotlib.pyplot as plt 
from utils import * 



test = f(1, 1/2, 13, -16)
test_derivative = f(1/2, 13*2, -16*3)

x = np.linspace(-10,10,10_000)

plt.plot(x, test(x))
plt.plot(x, np.ones(len(x)) + 1/2 * x + 13*x**2 - 16*x**3)
test_derivative = test.derivative(); print(type(test_derivative))
plt.plot(x, test_derivative(x))
plt.plot(x, 1/2 * np.ones(len(x)) + 13*2*x -16*3*x**2)
plt.show()

latex_fonts()

lambdas = np.linspace(1,30,5)
etas = np.logspace(-8,1,10)
H = np.random.rand(len(etas), len(lambdas))

plot_2D_parameter_lambda_eta(lambdas, etas, H, title='This is a title', filename='test', savefig=False, y_log=True)
plt.show()