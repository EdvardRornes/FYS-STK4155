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

class FFNN:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', alpha=0.01, lambda_reg=0.0):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        self.activation_func = activation
        self.alpha = alpha
        self.lambda_reg = lambda_reg

        for i in range(len(self.layers) - 1):
            weight_matrix = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2. / self.layers[i])
            self.weights.append(weight_matrix)
            bias_vector = np.zeros((1, self.layers[i + 1]))
            self.biases.append(bias_vector)

    def relu(self, z):
        return np.where(z > 0, z, 0)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def Lrelu(self, z):
        return np.where(z > 0, z, self.alpha * z)

    def Lrelu_derivative(self, z):
        return np.where(z > 0, 1, self.alpha)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        A = X
        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i] + self.biases[i]
            self.z_values.append(Z)
            A = self.relu(Z) if self.activation_func.lower() == 'relu' else self.sigmoid(Z) if self.activation_func.lower() == 'sigmoid' else self.Lrelu(Z)
            self.activations.append(A)

        Z = A @ self.weights[-1] + self.biases[-1]
        self.z_values.append(Z)
        self.activations.append(Z)
        return Z

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        y = y.reshape(-1, 1)

        delta = self.activations[-1] - y
        for i in reversed(range(len(self.weights))):
            # Calculate gradients
            dw = (self.activations[i].T @ delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            # Add L2 regularization term
            dw += (self.lambda_reg / m) * self.weights[i]  # Regularization term

            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
            
            if i > 0:
                delta = (delta @ self.weights[i].T) * (self.relu_derivative(self.z_values[i - 1]) if self.activation_func.lower() == 'relu' else self.sigmoid_derivative(self.z_values[i - 1]) if self.activation_func.lower() == 'sigmoid' else self.Lrelu_derivative(self.z_values[i - 1]))

    def train(self, X, y, learning_rate=0.01, epochs=1000, batch_size=None, shuffle=True, lambda_reg=0.0):
        self.lambda_reg = lambda_reg  # Update regularization parameter
        mse_history = []
        m = X.shape[0]

        if batch_size is None:  # Default to full-batch (i.e., all data at once)
            batch_size = m

        for epoch in range(epochs):
            if shuffle:
                indices = np.random.permutation(m)
                X, y = X[indices], y[indices]

            for i in range(0, m, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)

            mse = np.mean((self.activations[-1] - y_batch) ** 2)

            if np.isnan(mse) or mse > 100:
                mse = 1e10
                print(f'Epoch {epoch}, MSE ({self.activation_func}): {mse} (Issue encountered, breaking)')
                break
            
            mse_history.append(mse)

            # Print MSE every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, MSE ({self.activation_func}): {mse}')
        
        return mse_history

    def predict(self, X):
        return self.forward(X)


def plot_2D_parameter_lambda_eta(lambdas, etas, MSE, title=None, x_log=False, y_log=False, savefig=False, filename='', Reverse_cmap=False):
    cmap = 'plasma'
    if Reverse_cmap == True:
        cmap = 'plasma_r'
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
    sns.heatmap(data = MSE, ax = ax, cmap = cmap, annot=True, fmt=".3f",  xticklabels=t_x, yticklabels=t_y)
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
