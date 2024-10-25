import numpy as np
import autograd.numpy as anp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import jax.numpy as jnp
from jax import grad, jit
from autograd import grad
import seaborn as sns
from sklearn.metrics import mean_squared_error

import sys, os
import time 
import pickle

# Disable print
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore print
def enablePrint():
    sys.stdout = sys.__stdout__

def gradientOLS(X, y, beta):
    n=len(y)

    return 2.0/n*X.T @ (X @ (beta)-y)

def CostOLS(X, y, theta):
    n=len(y)
    return 1/n * anp.sum((y-X @ theta)**2)

class CostRidge:

    def __init__(self, lmbda:float):
        self.lmbda = lmbda 

    def __call__(self, X, y, theta):
        n = len(y)
        return (1.0 / n) * anp.sum((y-X @ theta) ** 2) + self.lmbda / n * anp.sum(theta**2)

class AutoGradCostFunction:

    def __init__(self, cost_function:callable, argument_index=2):
        """
        Creates callable gradient of given cost function. The cost function is a property of the class, and so changing it will change the gradient.
        Assumes that the cost_function has a function call on the form cost_function(X, y, theta).

        Arguments 
            * cost_function:        callable cost function 
            * argument_index:       index of argument to take gradient over
        """
        self._gradient = grad(cost_function, argument_index)
        self._cost_function = cost_function
        self._argument_index = argument_index

    @property
    def cost_function(self):
        return self._cost_function
    
    @cost_function.setter 
    def cost_function(self, new_cost_function):
        self._cost_function = new_cost_function 
        self._gradient = grad(new_cost_function, self._argument_index)

    def __call__(self, X, y, theta):
        """
        Returns gradient of current cost function.
        """
        return self._gradient(X, y, theta)

def create_Design_Matrix(*args):
    """
    Creates design matrix. Can handle 1D and 2D.

    Arguments
        * x:        x-values (1D and 2D case)
        * y:        y-values (2D case only)
        * degree:   polynomial degree
    """

    x, degree = args

    if len(x) == 2: # 2D case 
        x, y, degree = args
        degree -= 1 # Human to computer language 

        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int((degree+1)*(degree+2)/2) # Number of elements in beta
        X = np.ones((N,l))

        for i in range(1,degree+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:,q+k] = (x**(i-k))*(y**k)

        return X
    
    else: # 1D case 
        x, degree = args

        degree -= 1 # Human to computer language 

        x = np.asarray(x)
        x = x.flatten()
        
        # Create the design matrix with shape (len(x), degree + 1)
        X = np.vander(x, N=degree + 1, increasing=True)
        return X
    
class Polynomial:

    def __init__(self, *coeffs):    
        """
        Class representing polynomial function, taking in coefficients a_0 + a_1x + a_2x^2 + ... a_Nx^N.
        """
        self.coeffs = np.array(coeffs) 
    
    def __call__(self, x):
        return np.polyval(np.flip(self.coeffs), x) 

    def derivative(self):
        """
        Returns new instance of 'Polynomial' corresponding to the derivative.
        """
        new_coeffs = np.zeros(len(self.coeffs))
        for i in range(len(new_coeffs)-1):
            new_coeffs[i] = self.coeffs[i+1] * (i+1)
        return Polynomial(*new_coeffs)

class Optimizer:
    """
    Arguments
        * learning_rate:        number or callable(epochs, i), essentially the coefficient before the gradient
        * momentum:             number added in descent
        * epsilon:              small number used to not divide by zero in Adagrad, RMSprop and Adam
        * beta1:                used in Adam, in bias handling
        * beta2:                used in Adam, in bias handling
        * decay_rate:           used in ... 
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9, epsilon=1e-8, beta1=0.9, beta2=0.999, decay_rate=0.9):
        """
        Class to be inherited by PlaneGD, Adagrad, RMSprop or Adam, representing the method of choice for Stochastic Gradient Descent (SGD).
        """
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

class PlaneGD(Optimizer):

    __doc__ = Optimizer.__doc__
    def __init__(self, learning_rate=0.01, momentum=0.9, epsilon=None, beta1=None, beta2=None, decay_rate=None):
        """
        Class implementing basic Gradient Descent with optional momentum.
        """
        super().__init__(learning_rate, momentum)
    
    def __call__(self, *args):

        theta, gradient = args[0], args[1]
        self.initialize_velocity(theta)

        # Apply momentum
        theta -= self.momentum + self.learning_rate * gradient
        return theta

    def __str__(self):
        return f"Plane Gradient descent: eta: {self.learning_rate}, momentum: {self.momentum}"

class Adagrad(Optimizer):

    __doc__ = Optimizer.__doc__
    def __init__(self, learning_rate=0.01, epsilon=1e-8, momentum=None, beta1=None, beta2=None, decay_rate=None):
        """
        Class implementing the Adagrad optimization algorithm. 
        Adagrad adapts the learning rate for each parameter based on the accumulation of past squared gradients.
        """

        # Learing_rate is either callable or number
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
        # Initializes accumulation matrix G if not already initialized
        if self.G is None:
            self.G = np.zeros_like(theta)

    def __call__(self, *args):
        theta, gradient, epochs, i = args[0], args[1], args[2], args[3]

        self.initialize_accumulation(theta)

        # Accumulating squared gradients
        self.G += gradient**2

        #Updating theta
        theta -= (self.learning_rate(epochs, i) / (np.sqrt(self.G + self.epsilon))) * gradient
        return theta
    
    def __str__(self):
        return f"Adagrad: eta: {self._str_learning_rate}, eps: {self.momentum}"

class RMSprop(Optimizer):
    __doc__ = Optimizer.__doc__
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8, momentum=None, beta1=None, beta2=None):
        """
        Class implementing the RMSprop optimization algorithm.
        RMSprop maintains a moving average of the squared gradients to normalize the gradient.
        """

        # Learing_rate is either callable or number
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
        # Initializes accumulation matrix G if not already initialized
        if self.G is None:
            self.G = np.zeros_like(theta)

    def __call__(self, *args):
        theta, gradient, epochs, i = args[0], args[1], args[2], args[3]

        self.initialize_accumulation(theta)

        # Updating moving average of the squared gradients
        self.G = self.decay_rate * self.G + (1 - self.decay_rate) * gradient**2

        # Update theta
        theta -= (self.learning_rate(epochs, i) / (np.sqrt(self.G + self.epsilon))) * gradient
        return theta

    def __str__(self):
        return f"RMSprop: eta: {self._str_learning_rate}, eps: {self.momentum}, decay_rate = {self.decay_rate}"

class Adam(Optimizer):
    __doc__ = Optimizer.__doc__
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=None, momentum=None):
        """
        Class implementing the Adam optimization algorithm.
        Adam combines the advantages of both RMSprop and momentum by maintaining both first and second moment estimates.
        """

        # Learing_rate is either callable or number
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
        # Initializes first and second moment vectors if not already initialized
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

        # Update theta
        theta -= (self.learning_rate(epochs, i) / (np.sqrt(v_hat) + self.epsilon)) * m_hat
        return theta
    
    def __str__(self):
        return f"Adam: eta: {self._str_learning_rate}, eps: {self.momentum}, beta1 = {self.beta1}, beta2 = {self.beta2}"

class DescentSolver:

    def __init__(self, optimization_methods:list|Optimizer, degree:int, mode="GD"):
        
        if isinstance(optimization_methods, Optimizer):
            optimization_methods = [optimization_methods]
        
        self._optimization_methods = optimization_methods; self._N = len(self._optimization_methods)

        self._thetas_method = [0]*self._N

        self.degree = degree

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
            raise ValueError(f"Gradient descent needs three arguments: X, y, N, not {len(args)}")
        
        x, y, N = args[0], args[1], args[2]
        
        X = create_Design_Matrix(x, self.degree)

        if isinstance(theta, str):
            if theta.upper() == "RANDOM":
                self._thetas_method = [np.random.randn(X.shape[1], 1)]*self._N
        else:
            self._thetas_method = [np.copy(theta)] * self._N
        for iter in range(N):
            gradients = self._gradients_call(X, y)
            self.update(gradients, None, None)

            current_time = time.time()
            print(f"GD: {iter/N*100:.1f}%, time elapsed: {current_time-start_time:.1f}s", end="\r")

        print(f"GD: 100.0%, time elapsed: {current_time-start_time:.1f}s         ")
    
    def _SGD(self, *args, theta="RANDOM"):
        start_time = time.time()
        if len(args) != 4:
            raise ValueError(f"Gradient descent needs 4 arguments: X, y, epochs, m, M, not {len(args)}")
        
        x, y, epochs, batch_size = args[0], args[1], args[2], args[3]

        X = create_Design_Matrix(x, self.degree)

        m = int(X.shape[0]/batch_size)
        if isinstance(theta, str):
            if theta.upper() == "RANDOM":
                self._thetas_method = [np.random.randn(X.shape[1], 1)]*self._N
        else:
            self._thetas_method = [np.copy(theta)] * self._N 
        N = epochs * m
        for epoch in range(epochs):
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, m, batch_size):
                X_i = X_shuffled[i: i + m]
                y_i = y_shuffled[i: i + m]

                # print(np.shape(X_i))
                # print(np.shape(y_i))
                gradients = self._gradients_call(X_i, y_i)
                self.update(gradients, epoch, i)

                current_time = time.time()

            tmp = epoch * m /N * 100
            print(f"SGD: {tmp:.1f}%, time elapsed: {current_time-start_time:.1f}s", end="\r")

        print(f"SGD: 100.0%, time elapsed: {current_time-start_time:.1f}s         ")

    def __call__(self, *args, theta="RANDOM") -> list:
        for i in range(self._N):
            if self.gradients[i] is None:
                raise ValueError(f"No gradient is given for {i}-method: {str(self._optimization_methods[i])}")
        
        print("Starting training w/")
        for i in range(self._N):
            print(self._optimization_methods[i])
        
        self._descent_method_call(*args, theta=theta)
        return self._thetas_method

def create_list_of_it(item):
    """Utility function to ensure the item is a list."""
    if not isinstance(item, list):
        return [item]
    return item

class DescentAnalyzer:

    def __init__(self, x: np.ndarray, y: np.ndarray, optimizer: str,
                 degree: int, epochs:int, batch_size=None, GD_SGD="GD",
                 momentum=0, epsilon=1e-8, beta1=0.9, beta2=0.999, decay_rate=0.9, princt_percentage=False):
        self.x = x
        self.y = y
        self.optimizer_name = optimizer
        self.degree = degree
        self.epochs = epochs
        self.batch_size = batch_size
        self.GD_SGD = GD_SGD
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay_rate = decay_rate

        self._print_percentage = princt_percentage

        # Placeholder for results
        self.thetas = []
        self.learning_rates = []

        self.data = {
            "thetas": self.thetas,
            "learning_rates": self.learning_rates,
            "optimizer_name": self.optimizer_name,
            "degree": self.degree,
            "epochs": self.epochs,
            "batch_size": self.batch_size ,
            "GD_SGD": self.GD_SGD,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "decay_rate": self.decay_rate
        }

    def run_analysis(self, gradient:callable, learning_rate:float, save_learning_rate=True) -> np.ndarray:
        """Runs the descent analysis and stores the results."""

        if save_learning_rate:
            self.learning_rates.append(learning_rate)

        if not self._print_percentage:
            blockPrint()

        if self.optimizer_name.upper() == "PLANEGD":  # Can use GD or SGD
            if self.GD_SGD == "GD":
                planeGD = PlaneGD(learning_rate=learning_rate, momentum=self.momentum)
                descentSolver = DescentSolver(planeGD, self.degree, mode=self.GD_SGD)
                descentSolver.gradient = gradient
                
                theta = descentSolver(self.x, self.y, self.epochs)
                self.thetas.append(theta)

            elif self.GD_SGD == "SGD":
                planeGD = PlaneGD(learning_rate=learning_rate, momentum=self.momentum)
                descentSolver = DescentSolver(planeGD, self.degree, mode=self.GD_SGD)
                descentSolver.gradient = gradient

                theta = descentSolver(self.x, self.y, self.epochs, self.batch_size)
                self.thetas.append(theta)

        elif self.optimizer_name.upper() in ["ADAGRAD", "RMSPROP", "ADAM"]:
            optimizers = [Adagrad, RMSprop, Adam]
            optimizers_name = ["ADAGRAD", "RMSPROP", "ADAM"]
            optimizer_index = optimizers_name.index(self.optimizer_name.upper())

            
            optimizer_class = optimizers[optimizer_index]
            optimizer = optimizer_class(learning_rate=learning_rate, momentum=self.momentum,
                                        epsilon=self.epsilon, beta1=self.beta1, beta2=self.beta2, decay_rate=self.decay_rate)

            descentSolver = DescentSolver(optimizer, self.degree, mode=self.GD_SGD)
            descentSolver.gradient = gradient

            theta = descentSolver(self.x, self.y, self.epochs, self.batch_size)
            self.thetas.append(theta)
        else:
            raise ValueError(f"Unknown optimizer '{self.optimizer_name}'.")
        
        self._update_data()

        enablePrint()

        return self.thetas

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def save_data(self, filename, overwrite=False, stop=50):
        """Saves the analysis data to a file.

        Args:
            filename (str): The name of the file to save the data to.
            overwrite (bool): Whether to overwrite the file if it exists.
        """
        filename_full = f"{filename}_{0}.pkl"

        if not overwrite:
            if os.path.exists(filename_full):
                i = 1
                while i <= stop: # May already be additional files created
                    filename_full = f"{filename}_{i}.pkl"
                    if not os.path.exists(filename_full):
                        with open(filename_full, 'wb') as f:
                            pickle.dump(self.data, f)
                        
                        return 
                    i += 1
            
                raise ValueError(f"You have {stop} additional files of this sort?")
                
        with open(filename_full, 'wb') as f:
            pickle.dump(self.data, f)


    def load_data(self, filename):
        """Loads analysis data from a file.

        Args:
            filename (str): The name of the file to load the data from.
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        self.thetas = data['thetas']
        self.optimizer_name = data['optimizer_name']
        self.degree = data['degree']
        self.epochs = data['epochs']
        self.learning_rates = data['learning_rates']
        self.batch_size  = data['batch_size']
        self.GD_SGD = data['GD_SGD']
        self.momentum = data['momentum']
        self.epsilon = data['epsilon']
        self.beta1 = data['beta1']
        self.beta2 = data['beta2']
        self.decay_rate = data['decay_rate']

        self._update_data()

    def get_data(self):
        return self.data

    def _update_data(self):
        self.data['thetas'] = self.thetas; self.data["optimizer_name"] =  self.optimizer_name
        self.data["degree"] =  self.degree; self.data["epochs"] =  self.epochs
        self.data["learning_rates"] =  self.learning_rates; self.data["batch_size"] =  self.batch_size
        self.data["GD_SGD"] =  self.GD_SGD; self.data["momentum"] =  self.momentum
        self.data["epsilon"] =  self.epsilon; self.data["beta1"] =  self.beta1
        self.data["beta2"] =  self.beta2; self.data["decay_rate"] =  self.decay_rate

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
