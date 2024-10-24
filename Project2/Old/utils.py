import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import jax.numpy as jnp
from jax import grad, jit
from autograd import grad
import seaborn as sns
from sklearn.metrics import mean_squared_error

import time 

def gradientOLS(X, y, beta):
    n=len(y)

    return 2.0/n*X.T @ (X @ (beta)-y)

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

    N = len(args) 
    if N == 2: # 1D case 
        x, degree = args
        degree -= 1 # Human to computer language 

        x = np.asarray(x)

        # Create the design matrix with shape (len(x), degree + 1)
        X = np.vander(x, N=degree + 1, increasing=True)
        return X

    if N == 3: # 2D case 
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
        
        X = create_Design_Matrix(x, y, self.degree)

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
        
        x, y, epochs, M = args[0], args[1], args[2], args[3]

        X = create_Design_Matrix(x, y, self.degree)

        m = int(X.shape[0]/M)
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

            for i in range(0, m, M):
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

# def create_list_of_it(arg):
#     if isinstance(arg, list):
#         return arg 
    
#     return [arg]


def analyze_descent(filename:str, x:np.ndarray, y:np.ndarray, optimizer:str, gradient:callable, 
                    degree:int, epochs:list, learing_rates:list, M:list, 
                    GD_SGD="GD", momentum=0.9, epsilon=1e-8, beta1=0.9, beta2=0.999, decay_rate=0.9, overwrite=False) -> list:
    
    epochs = create_list_of_it(epochs); learing_rates = create_list_of_it(learing_rates)
    Ms = create_list_of_it(M)

    # Testing that len(epochs) == len(learning_rates) == len(Ms), the last one only for SGD
    args_length = [len(epochs), len(learing_rates), len(Ms)]; arg_names = ["epochs", "learning_rates", "M"]
    if GD_SGD == "GD":
        args_length = args_length[:-1]; arg_names = arg_names[:-1]

    if len(set(args_length)) != 1:
        for i in range(len(args_length)):
            for j in range(len(args_length)):
                if args_length[i] != args_length[j]:
                    raise ValueError(f"Size of {arg_names[i]} is not the same as {arg_names[j]}; {args_length[i]} != {args_length[j]}")
    
    thetas = []
    if optimizer.upper() == "PLANEGD": # Can use GD or SGD
        if GD_SGD == "GD":
            for i in range(len(learing_rates)):
                planeGD = PlaneGD(learning_rate=learing_rates[i], momentum=momentum)
                descentSolver = DescentSolver(planeGD, degree, mode=GD_SGD)
                descentSolver.gradient = gradient

                thetas.append(descentSolver(x, y, epochs[i]))

        elif GD_SGD == "SGD":
            for i in range(len(learing_rates)):
                planeGD = PlaneGD(learning_rate=learing_rates[i], momentum=momentum)
                descentSolver = DescentSolver(planeGD, degree, mode=GD_SGD)
                descentSolver.gradient = gradient

                thetas.append(descentSolver(x, y, epochs[i], Ms[i]))

    elif optimizer.upper() in ["ADAGRAD", "RMSPROP", "ADAM"]:
        optimizers = [Adagrad, RMSprop, Adam]; optimizers_name = ["ADAGRAD", "RMSPROP", "ADAM"]
        optimizer_index = optimizers_name.index(optimizer.upper())

        for i in range(len(learing_rates)):
            optimizer = optimizers[optimizer_index]
            optimizer = optimizer(learning_rate=learing_rates[i], momentum=momentum, 
                                epsilon=epsilon, beta1=beta1, beta2=beta2, decay_rate=decay_rate)
            
            descentSolver = DescentSolver(optimizer, degree, mode=GD_SGD)
            descentSolver.gradient = gradient

            thetas.append(descentSolver(x, y, epochs[i], Ms[i]))

    else:
        raise ValueError(f"What is '{optimizer}'?")

    with open(filename, "w") as file:
        if overwrite:
            for i in range(len(thetas)):
                pass 

    return thetas 


import autograd.numpy as np
x = np.random.rand(1000,1)
beta_true = np.array([[1, -8, 16],]).T
func = Polynomial(*beta_true)
y = func(x)

def CostOLS(X,y,theta):
        return np.sum((y-X @ theta)**2)

    
gradient = grad(CostOLS, 2)