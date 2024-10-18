import numpy as np
import jax.numpy as jnp
from jax import grad, jit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

class f:
    def __init__(self, a0, a1, a2, a3=0):
        self.a0 = a0; self.a1 = a1; self.a2 = a2; self.a3 = a3

    def __call__(self, x):
        return self.a0 + self.a1 * x + self.a2 * x**2 + self.a3 * x**3

    def derivative(self):
        return f(self.a1, 2*self.a2, 3*self.a3)

class Optimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9, epsilon=1e-8, beta1=0.9, beta2=0.999, decay_rate=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay_rate = decay_rate
        self.velocity = None  # For momentum term

    def initialize_velocity(self, theta):
        if self.velocity is None:
            self.velocity = np.zeros_like(theta)

    def update(self, theta, gradient):
        raise NotImplementedError("This method should be implemented by subclasses")


class GradientDescent(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate, momentum)
    
    def update(self, theta, gradient):
        self.initialize_velocity(theta)
        # Apply momentum
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradient
        theta -= self.velocity
        return theta


class Adagrad(Optimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        super().__init__(learning_rate, epsilon=epsilon)
        self.G = None  # Accumulated squared gradients
    
    def initialize_accumulation(self, theta):
        if self.G is None:
            self.G = np.zeros_like(theta)

    def update(self, theta, gradient):
        self.initialize_accumulation(theta)
        self.G += gradient**2
        theta -= (self.learning_rate / (np.sqrt(self.G + self.epsilon))) * gradient
        return theta


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate, epsilon=epsilon, decay_rate=decay_rate)
        self.G = None

    def initialize_accumulation(self, theta):
        if self.G is None:
            self.G = np.zeros_like(theta)

    def update(self, theta, gradient):
        self.initialize_accumulation(theta)
        self.G = self.decay_rate * self.G + (1 - self.decay_rate) * gradient**2
        theta -= (self.learning_rate / (np.sqrt(self.G + self.epsilon))) * gradient
        return theta


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate, epsilon=epsilon, beta1=beta1, beta2=beta2)
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0  # Time step

    def initialize_moments(self, theta):
        if self.m is None:
            self.m = np.zeros_like(theta)
        if self.v is None:
            self.v = np.zeros_like(theta)

    def update(self, theta, gradient):
        self.t += 1
        self.initialize_moments(theta)

        # Update biased first and second moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2

        # Compute bias-corrected moments
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Update parameters
        theta -= (self.learning_rate / (np.sqrt(v_hat) + self.epsilon)) * m_hat
        return theta


def optimize(X, y, theta, epochs, optimizer, batch_size=None, mode='GD', use_jax=False):
    """
    Optimizer function with momentum, Adagrad, RMSprop, and Adam.
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - theta: Initial parameters
    - epochs: Number of epochs
    - optimizer: Instance of an optimizer class
    - batch_size: Size of mini-batches for SGD. None for full-batch GD.
    - mode: 'GD' for gradient descent, 'SGD' for stochastic gradient descent
    - use_jax: Boolean indicating whether to use JAX for computations
    
    Returns:
    - theta: Optimized parameters
    """
    m = len(y)

    # Use JAX if requested
    if use_jax:
        X = jnp.array(X)
        y = jnp.array(y)
        theta = jnp.array(theta)

    # Define the cost function and its gradient
    def cost_function(theta):
        predictions = jnp.dot(X, theta)
        return jnp.mean((predictions - y) ** 2)

    # Use JAX's automatic differentiation for gradient computation
    gradient_func = jit(grad(cost_function))

    for epoch in range(epochs):
        if mode == 'GD':  # Full-batch gradient descent
            gradient = gradient_func(theta)

            # Update theta using the chosen optimizer
            theta = optimizer.update(theta, gradient)

        elif mode == 'SGD':  # Stochastic gradient descent
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, m, batch_size):
                X_i = X_shuffled[i:i + batch_size]
                y_i = y_shuffled[i:i + batch_size]

                predictions = jnp.dot(X_i, theta)
                gradient = (2 / batch_size) * jnp.dot(X_i.T, predictions - y_i)

                # Update theta using the chosen optimizer
                theta = optimizer.update(theta, gradient)

        else:
            raise ValueError(f"Invalid mode '{mode}'. Use 'GD' for gradient descent or 'SGD' for stochastic gradient descent.")

    return theta

def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivative of the Sigmoid activation function."""
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    """ReLU activation function."""
    return np.where(z > 0, z, 0)

def relu_derivative(z):
    """Derivative of ReLU activation function."""
    return np.where(z > 0, 1, 0)

def feed_forward_batch(inputs, layers, activations):
    a = inputs
    for (W, b), activations in zip(layers, activations):
        z = a @ W.T + b
        a = activations(z)
    return a

def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(layer_output_size, i_size)
        b = np.random.rand(layer_output_size) # Use uniform dist on bias
        layers.append((W,b))

        i_size = layer_output_size
    return layers

def plot_2D_parameter(lambdas, etas, MSE, savefig = False, filename=''):
    fig, ax = plt.subplots(figsize = (10, 12))
    tick = ticker.ScalarFormatter(useOffset=False, useMathText=True)
    tick.set_powerlimits((0, 0))
    t_x = [u'${}$'.format(tick.format_data(lambd)) for lambd in lambdas]
    t_y = [u'${}$'.format(tick.format_data(eta)) for eta in etas]
    sns.heatmap(data = MSE, ax = ax, cmap = 'plasma')
    plt.xlim(0, len(lambdas))
    plt.ylim(0, len(etas))
    plt.tight_layout()
    if savefig:
        plt.savefig('Figures/{filename}.pdf')