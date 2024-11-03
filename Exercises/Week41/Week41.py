import numpy as np
from autograd import grad
import autograd.numpy as anp


class f:
    def __init__(self, a0, a1, a2, a3=0):
        self.a0 = a0; self.a1 = a1; self.a2 = a2; self.a3 = a3

    def __call__(self, x):
        return self.a0 + self.a1 * x + self.a2 * x**2 + self.a3 * x**3

    def derivative(self):
        return f(self.a1, 2*self.a2, 3*self.a3)


# Thought to handle all methods (GD, Momentum, AdaGrad, RMSprop, Adam)
class New_theta:
    def __init__(self, use_momentum=False, use_AdaGrad=False, use_RMSprop=False, use_Adam=False, beta1=0.9, beta2=0.999):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8  # Avoid division by zero

        if use_momentum and not (use_AdaGrad or use_RMSprop or use_Adam):
            # Momentum Gradient Descent
            def call_me(gradients, theta, eta, change, Giter, v, t):
                new_change = eta * gradients + self.beta1 * change
                theta -= new_change
                return theta, new_change, Giter, v, t

        elif use_AdaGrad and not (use_momentum or use_RMSprop or use_Adam):
            # AdaGrad
            def call_me(gradients, theta, eta, change, Giter, v, t):
                Giter += gradients * gradients
                update = gradients * eta / (self.epsilon + np.sqrt(Giter))
                theta -= update
                return theta, change, Giter, v, t

        elif use_RMSprop and not (use_momentum or use_AdaGrad or use_Adam):
            # RMSprop
            def call_me(gradients, theta, eta, change, Giter, v, t):
                Giter = self.beta1 * Giter + (1 - self.beta1) * (gradients ** 2)
                update = gradients * eta / (self.epsilon + np.sqrt(Giter))
                theta -= update
                return theta, change, Giter, v, t

        elif use_Adam:
            # Adam
            def call_me(gradients, theta, eta, change, Giter, v, t):
                v = self.beta1 * v + (1 - self.beta1) * gradients  # Momentum term
                Giter = self.beta2 * Giter + (1 - self.beta2) * (gradients ** 2)  # RMSprop term

                # Bias correction for momentum and RMSprop terms
                v_corrected = v / (1 - self.beta1 ** (t + 1))
                Giter_corrected = Giter / (1 - self.beta2 ** (t + 1))

                # Update theta
                update = eta * v_corrected / (self.epsilon + np.sqrt(Giter_corrected))
                theta -= update
                return theta, change, Giter, v, t + 1

        else:
            # Standard Gradient Descent
            def call_me(gradients, theta, eta, change, Giter, v, t):
                theta -= eta * gradients
                return theta, change, Giter, v, t

        self.call_me = call_me

    def __call__(self, gradients, theta, eta, change, Giter, v, t):
        return self.call_me(gradients, theta, eta, change, Giter, v, t)


# Stochastic Gradient Descent with different methods (plain, momentum, AdaGrad, RMSprop, Adam)
def stochastic_gradient_descent(derivative_func, X, y, M, n_epochs, learning_schedule, new_theta, momentum=0.9, delta=1e-8):
    n = len(y)
    theta = np.random.randn(X.shape[1], 1)  # Initialize theta
    m = int(n / M)  # Number of minibatches

    change = np.zeros_like(theta)  # Initialize change for momentum
    Giter = np.zeros_like(theta)  # Initialize Giter for AdaGrad/RMSprop/Adam
    v = np.zeros_like(theta)  # Initialize v for Adam
    t = 0  # Time step for Adam
    
    for epoch in range(n_epochs):
        for i in range(m):
            # Minibatch generation
            random_index = M * np.random.randint(m)
            xi = X[random_index:random_index + M]
            yi = y[random_index:random_index + M]

            # Calculate gradients
            gradients = (1.0 / M) * derivative_func(yi, xi, theta)

            # Learning rate adjustment
            eta = learning_schedule(epoch * m + i)

            # Update theta using chosen method (momentum, AdaGrad, RMSprop, Adam)
            theta, change, Giter, v, t = new_theta(gradients, theta, eta, change, Giter, v, t)

    return theta


# Inspired by code given in excercise

# Cost function for OLS
def CostOLS(y, X, theta):
    return anp.sum((y - X @ theta) ** 2)

# Generate random data
n = 100
x = 2 * np.random.rand(n, 1)
theta = [1, 1/2, -3/4, 5/18]
y = f(theta[0], theta[1], theta[2], theta[3])
y = y(x)

# Design matrix
X = np.c_[np.ones((n, 1)), x, x**2, x**3]  # This will give theta 4 components

# Gradient of the cost function
training_gradient = grad(CostOLS, 2)

# Learning schedule
def learning_schedule(t, t0=5, t1=50):
    return t0 / (t + t1)

averageN_times = 10
theta_rmsprop_array, theta_AdaGrad_array, theta_momentum_array, theta_adam_array = (
    np.zeros((4, averageN_times)),
    np.zeros((4, averageN_times)),
    np.zeros((4, averageN_times)),
    np.zeros((4, averageN_times)),
)

momentum = 1e-1

# Initializing the different optimizers
optimizer_RMS = New_theta(use_RMSprop=True)
optimizer_AdaGrad = New_theta(use_AdaGrad=True)
optimizer_momentum = New_theta(use_momentum=True)
optimizer_Adam = New_theta(use_Adam=True)
for n in range(averageN_times):

    # Running stochastic gradient descent for each optimizer
    theta_rmsprop_array[:,n] = stochastic_gradient_descent(training_gradient, X, y, M=5, n_epochs=50, 
                                                learning_schedule=learning_schedule, new_theta=optimizer_RMS)[0]
    theta_AdaGrad_array[:,n] = stochastic_gradient_descent(training_gradient, X, y, M=5, n_epochs=50, 
                                                learning_schedule=learning_schedule, new_theta=optimizer_AdaGrad)[0]
    theta_momentum_array[:,n] = stochastic_gradient_descent(training_gradient, X, y, M=5, n_epochs=50, 
                                                learning_schedule=learning_schedule, new_theta=optimizer_momentum)[0]
    theta_adam_array[:,n] = stochastic_gradient_descent(training_gradient, X, y, M=5, n_epochs=50, 
                                                learning_schedule=learning_schedule, new_theta=optimizer_Adam)[0]

# Printing mean of coefficients
print("Theta: Original:", theta)
print("Theta from RMSprop:", [np.mean(theta_rmsprop_array[i,:]) for i in range(4)])
print("Theta from AdaGrad:", [np.mean(theta_AdaGrad_array[i,:]) for i in range(4)])
print("Theta w/ momentum:", [np.mean(theta_momentum_array[i,:]) for i in range(4)])
print("Theta from Adam:", [np.mean(theta_adam_array[i,:]) for i in range(4)])
