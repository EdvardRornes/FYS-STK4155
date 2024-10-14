import numpy as np
import jax.numpy as jnp
from jax import grad, jit

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
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU activation function."""
    return np.where(z > 0, 1, 0)

def ffnn_forward(X, weights, activation='sigmoid'):
    """Forward pass for a FFNN with multiple hidden layers."""
    layer_inputs = {}
    layer_outputs = {}
    
    # Input layer
    layer_outputs[0] = X
    for i in range(len(weights) // 2):  # Since weights contain both W and b
        layer_inputs[i + 1] = layer_outputs[i] @ weights[f'W{i + 1}'] + weights[f'b{i + 1}']
        if i < (len(weights) // 2 - 1):
            # Use activation function for hidden layers
            layer_outputs[i + 1] = sigmoid(layer_inputs[i + 1]) if activation == 'sigmoid' else relu(layer_inputs[i + 1])
        else:
            # Linear output for the final layer
            layer_outputs[i + 1] = layer_inputs[i + 1]
    
    return layer_outputs[len(weights) // 2], layer_outputs  # Return output and all layer outputs for backward pass

def ffnn_backward(X, y, layer_outputs, weights, learning_rate, activation='sigmoid'):
    """Backward pass for updating weights using gradient descent."""
    m = len(y)
    gradients = {}

    # Output layer error (linear output for regression)
    dz = layer_outputs[len(weights) // 2] - y
    gradients[f'dW{len(weights) // 2}'] = (1 / m) * layer_outputs[len(weights) // 2 - 1].T @ dz
    gradients[f'db{len(weights) // 2}'] = (1 / m) * np.sum(dz, axis=0)

    # Backpropagation through hidden layers
    for i in range(len(weights) // 2 - 1, 0, -1):
        if activation == 'sigmoid':
            dz = dz @ weights[f'W{i + 1}'].T * sigmoid_derivative(layer_outputs[i])
        else:
            dz = dz @ weights[f'W{i + 1}'].T * relu_derivative(layer_outputs[i])
        
        gradients[f'dW{i}'] = (1 / m) * layer_outputs[i - 1].T @ dz
        gradients[f'db{i}'] = (1 / m) * np.sum(dz, axis=0)

    # Update weights
    for i in range(len(weights) // 2):
        weights[f'W{i + 1}'] -= learning_rate * gradients[f'dW{i + 1}']
        weights[f'b{i + 1}'] -= learning_rate * gradients[f'db{i + 1}']

    return weights

# Initialize weights for a FFNN with a flexible number of hidden layers
def initialize_weights(layer_sizes):
    weights = {}
    for i in range(len(layer_sizes) - 1):
        weights[f'W{i + 1}'] = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2. / layer_sizes[i])  # He initialization
        weights[f'b{i + 1}'] = np.zeros(layer_sizes[i + 1])
    return weights