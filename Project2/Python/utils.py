import numpy as np
import jax.numpy as jnp
from jax import grad, jit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import time 

class f:

    def __init__(self, *coeffs):    
        self.coeffs = np.array(coeffs)
    
    def __call__(self, x):
        return np.polyval(np.flip(self.coeffs), x)

    def derivative(self):
        new_coeffs = np.zeros(len(self.coeffs))
        for i in range(len(new_coeffs)-1):
            new_coeffs[i] = self.coeffs[i+1] * (i+1)
        return f(*new_coeffs)

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

    def __call__(self, args):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def __str__(self):
        return "Not defined"


class GradientDescent(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate, momentum)
    
    def __call__(self, *args):

        theta, gradient = args[0], args[1]
        self.initialize_velocity(theta)
        # Apply momentum
        self.velocity = self.momentum + self.learning_rate * gradient
        theta -= self.velocity
        return theta

    def __str__(self):
        return f"Plane Gradient descent: eta: {self.learning_rate}, momentum: {self.momentum}"
    
class Adagrad(Optimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        if isinstance(learning_rate, int) or isinstance(learning_rate, float):
            tmp = learning_rate
            def learning_rate(epochs, i):
                return tmp 
            
            self._str_learning_rate = str(tmp)
        else:
            self._str_learning_rate = "callable"

        self.learning_rate = learning_rate

        super().__init__(learning_rate, epsilon=epsilon)
        self.G = None  # Accumulated squared gradients
    
    def initialize_accumulation(self, theta):
        if self.G is None:
            self.G = np.zeros_like(theta)

    def __call__(self, *args):
        theta, gradient, epochs, i = args[0], args[1], args[2], args[3]

        self.initialize_accumulation(theta)
        self.G += gradient**2
        theta -= (self.learning_rate(epochs, i) / (np.sqrt(self.G + self.epsilon))) * gradient
        return theta
    
    def __str__(self):
        return f"Adagrad: eta: {self._str_learning_rate}, eps: {self.momentum}"


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        if isinstance(learning_rate, int) or isinstance(learning_rate, float):
            tmp = learning_rate
            def learning_rate(epochs, i):
                return tmp 

            self._str_learning_rate = str(tmp)
        else:
            self._str_learning_rate = "callable"

        self.learning_rate = learning_rate

        super().__init__(learning_rate, epsilon=epsilon, decay_rate=decay_rate)
        self.G = None

    def initialize_accumulation(self, theta):
        if self.G is None:
            self.G = np.zeros_like(theta)

    def __call__(self, *args):
        theta, gradient, epochs, i = args[0], args[1], args[2], args[3]

        self.initialize_accumulation(theta)
        self.G = self.decay_rate * self.G + (1 - self.decay_rate) * gradient**2
        theta -= (self.learning_rate(epochs, i) / (np.sqrt(self.G + self.epsilon))) * gradient
        return theta

    def __str__(self):
        return f"RMSprop: eta: {self._str_learning_rate}, eps: {self.momentum}, decay_rate = {self.decay_rate}"

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if isinstance(learning_rate, int) or isinstance(learning_rate, float):
            tmp = learning_rate
            def learning_rate(epochs, i):
                return tmp 
            
            self._str_learning_rate = str(tmp)
        else:
            self._str_learning_rate = "callable"

        self.learning_rate = learning_rate

        self.learning_rate = learning_rate

        super().__init__(learning_rate, epsilon=epsilon, beta1=beta1, beta2=beta2)
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0  # Time step

    def initialize_moments(self, theta):
        if self.m is None:
            self.m = np.zeros_like(theta)
        if self.v is None:
            self.v = np.zeros_like(theta)

    def __call__(self, *args):
        theta, gradient, epochs, i = args[0], args[1], args[2], args[3]

        self.t += 1
        self.initialize_moments(theta)

        # Update biased first and second moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2

        # Compute bias-corrected moments
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Update parameters
        theta -= (self.learning_rate(epochs, i) / (np.sqrt(v_hat) + self.epsilon)) * m_hat
        return theta
    
    def __str__(self):
        return f"Adam: eta: {self._str_learning_rate}, eps: {self.momentum}, beta1 = {self.beta1}, beta2 = {self.beta2}"

class beta_giver:

    def __init__(self, optimization_methods:list|Optimizer, mode="GD"):
        
        if isinstance(optimization_methods, Optimizer):
            optimization_methods = [optimization_methods]
        
        self._optimization_methods = optimization_methods; self._N = len(self._optimization_methods)

        self._thetas_method = [0]*self._N

        def update(gradient, epochs, i):
            for i in range(self._N):
                self._thetas_method[i] = self._optimization_methods[i](self._thetas_method[i], gradient[i], epochs, i)
        
        self.update = update 
        self.gradients = [None]*self._N 

        self._message = [str(i) for i in self._optimization_methods]
        
        if mode.upper() == "GD":
            self._descent_method_call = self._GD  
        elif mode.upper() == "SGD":
            self._descent_method_call = self._SGD  
    
    @property
    def gradient(self):
        return self.gradient
    
    @gradient.setter 
    def gradient(self, gradient_methods, index=None):
        if index is None:
            if not isinstance(gradient_methods, list):
                self.gradients = [gradient_methods]*self._N
            else:
                for i in range(len(gradient_methods)):
                    self.gradients[i] = gradient_methods[i]
        else:
            self.gradients[index] = gradient_methods

    def _gradients_call(self, X, y):
        return [self.gradients[i](X, y, self._thetas_method[i]) for i in range(self._N)]
            
    def _GD(self, *args, theta="RANDOM"):
        start_time = time.time()

        if len(args) != 3:
            raise ValueError(f"Gradient descent needs three arguments: X, y, N")
        
        X, y, N = args[0], args[1], args[2]

        if isinstance(theta, str):
            if theta.upper() == "RANDOM":
                self._thetas_method = [np.random.randn(X.shape[1], 1)]*self._N
        else:
            self._thetas_method = [theta] * self._N
        for iter in range(N):
            gradients = self._gradients_call(X, y)
            self.update(gradients, None, None)

            current_time = time.time()
            print(f"{iter/N*100:.1f}%, time elapsed: {current_time-start_time:.1f}s", end="\r")

        print(f"100.0%, time elapsed: {current_time-start_time:.1f}s         ")
    
    def _SGD(self, *args, theta="RANDOM"):
        start_time = time.time()
        if len(args) != 5:
            raise ValueError(f"Gradient descent needs three arguments: X, y, epochs, m, M")
        
        X, y, epochs, m, M = args[0], args[1], args[2], args[3], args[4]

        if isinstance(theta, str):
            if theta.upper() == "RANDOM":
                self._thetas_method = [np.random.randn(X.shape[1], 1)]*self._N

        N = epochs * m*M
        for epoch in range(epochs):
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, m, M):
                X_i = X_shuffled[i: i + m]
                y_i = y_shuffled[i: i + m]

                gradients = self._gradients_call(X_i, y_i)
                self.update(gradients, epoch, i)

                current_time = time.time()
                print(f"{epoch * i /N:.1f}%, time elapsed: {current_time-start_time:.1f}s", end="\r")

        print(f"100.0%, time elapsed: {current_time-start_time:.1f}s         ")

    def __call__(self, *args, theta="RANDOM") -> list:
        for i in range(self._N):
            if self.gradients[i] is None:
                raise ValueError(f"No gradient is given for {i}-method: {str(self._optimization_methods[i])}")
        
        print("Starting training w/")
        for i in range(self._N):
            print(self._optimization_methods[i])
        
        self._descent_method_call(*args, theta=theta)
        return self._thetas_method 




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

class Franke:

    def __init__(self, N:int, eps:float):
        """
        Parameters
            * N:    number of data points 
            * eps:  noise-coefficient         
        """
        self.N = N; self.eps = eps
        self.x = np.random.rand(N)
        self.y = np.random.rand(N)

        self.z_without_noise = self.franke(self.x, self.y)
        self.z = self.z_without_noise + self.eps * np.random.normal(0, 1, self.z_without_noise.shape)

    def franke(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        """
        Parameters
            * x:    x-values
            * y:    y-values

        Returns
            - franke function evaluated at (x,y) 
        """
    
        term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2))
        term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y + 1))
        term3 = 0.5*np.exp(-(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2))
        term4 = -0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
        return term1 + term2 + term3 + term4

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
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\eta$')
    plt.tight_layout()
    if savefig is not None:
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
