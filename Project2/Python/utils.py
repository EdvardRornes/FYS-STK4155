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

# Taken from week42 lecture slides
class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=50,
            n_categories=10,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self, activation):
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = activation(self.z_h)

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X, activation):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = activation(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self):
        error_output = self.probabilities - self.Y_data
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()

def plot_2D_parameter_lambda_eta(lambdas, etas, MSE, title=None, x_log=False, y_log=False, savefig=False, filename=''):
    fig, ax = plt.subplots(figsize = (12, 10))
    tick = ticker.ScalarFormatter(useOffset=False, useMathText=True)
    tick.set_powerlimits((0, 0))
    if x_log:
        t_x = [u'${}$'.format(tick.format_data(lambd)) for lambd in lambdas]
    else:
        t_x = [fr'${lambd}$' for lambd in lambdas]
    if y_log:
        t_y = [u'${}$'.format(tick.format_data(eta)) for eta in etas]
    else:
        t_y = [fr'${eta}$' for eta in etas]
    sns.heatmap(data = MSE, ax = ax, cmap = 'plasma', annot=True,  xticklabels=t_x, yticklabels=t_y)
    if title is not None:
        plt.title(title)
    plt.xlim(0, len(lambdas))
    plt.ylim(0, len(etas))
    plt.tight_layout()
    if savefig:
        plt.savefig(f'Figures/{filename}.pdf')


# Latex fonts
def latex_fonts():
    plt.rcParams['text.usetex'] = True
    plt.rcParams['axes.titlepad'] = 25 

    plt.rcParams.update({
        'font.family': 'euclid',
        'font.weight': 'bold',
        'font.size': 17,       # General font size
        'axes.labelsize': 17,  # Axis label font size
        'axes.titlesize': 22,  # Title font size
        'xtick.labelsize': 22, # X-axis tick label font size
        'ytick.labelsize': 22, # Y-axis tick label font size
        'legend.fontsize': 17, # Legend font size
        'figure.titlesize': 25 # Figure title font size
    })
