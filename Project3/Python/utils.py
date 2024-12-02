import numpy as np
import autograd.numpy as anp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import jax.numpy as jnp
from jax import grad, jit
from autograd import grad, elementwise_grad
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import minmax_scale

import sys, os
import time 
import pickle
import copy
from typing import Tuple, List

############ Utility functions ############
# Disable print
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore print
def enablePrint():
    sys.stdout = sys.__stdout__

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
        # self.x = np.sort(np.random.uniform(0, 1, N)) 
        # self.y = np.sort(np.random.uniform(0, 1, N)) 

        self.z_without_noise = self.franke(self.x, self.y)
        self.z = self.z_without_noise + self.eps * np.random.normal(0, 1, self.z_without_noise.shape)
    @staticmethod
    def franke(x:np.ndarray, y:np.ndarray) -> np.ndarray:
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
    
############ Metric functions ############
def MSE(y,ytilde):
    n = len(y)
    return 1/n * np.sum(np.abs(y-ytilde)**2)

############ Activation functions ############
class Activation:
    @staticmethod
    def sigmoid(z):
        """Sigmoid activation function."""
        return 1 / (1 + anp.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        """Derivative of the Sigmoid activation function."""
        sigmoid_z = Activation.sigmoid(z)
        return sigmoid_z * (1 - sigmoid_z)

    @staticmethod
    def relu(z):
        """ReLU activation function."""
        return np.where(z > 0, z, 0)

    @staticmethod
    def relu_derivative(z):
        """Derivative of ReLU activation function."""
        return np.where(z > 0, 1, 0)

    @staticmethod
    def Lrelu(z, alpha=0.01):
        """Leaky ReLU activation function."""
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def Lrelu_derivative(z, alpha=0.01):
        """Derivative of Leaky ReLU activation function."""
        return np.where(z > 0, 1, alpha)
    
    @staticmethod
    def tanh(z):
        """tanh activation function."""
        return np.tanh(z)
    
    @staticmethod
    def tanh_derivative(z):
        """Derivative of tanh activation function."""
        return 1/np.cosh(z)**2


############ Cost/gradient functions ############
def gradientOLS(X, y, beta):
    n=len(y)

    return 2.0/n*X.T @ (X @ (beta)-y)

def CostOLS(X, y, theta):
    n=len(y)
    return 1/n * anp.sum((y-X @ theta)**2)

def CostRidge(X, y, theta, lmbda):
    n = len(y)
    return (1.0 / n) * anp.sum((y-X @ theta) ** 2) + lmbda / n * anp.sum(theta**2)

class LogisticCost:

    def __init__(self, exp_clip=1e3, log_clip=1e-13, hypothesis=Activation.sigmoid):
        """
        Logistic cost function which removes too high values for exp/too low for log.
        """
        self.exp_clip = exp_clip; self.log_clip = log_clip
        self.hypothesis_func = hypothesis

    def __call__(self, x, y, w, lmbda):

        # computing hypothesis
        z = anp.dot(x,w)
        z = anp.clip(z, -self.exp_clip, self.exp_clip)
        h = self.hypothesis_func(z)

        cost = (-1 / len(y)) * anp.sum(y * anp.log(h + self.log_clip) + (1 - y) * anp.log(1 - h + self.log_clip))
        reg_term = lmbda * anp.sum(w[1:] ** 2)

        # Compute total cost
        return cost + reg_term
    
class AutoGradCostFunction:

    def __init__(self, cost_function:callable, argument_index=2, elementwise=False):
        """
        Creates callable gradient of given cost function. The cost function is a property of the class, and so changing it will change the gradient.
        Assumes that the cost_function has a function call on the form cost_function(X, y, theta).

        Arguments 
            * cost_function:        callable cost function 
            * argument_index:       index of argument to take gradient over
        """
        self._gradient = grad(cost_function, argument_index)
        if elementwise:
            self._gradient = elementwise_grad(cost_function, argument_index)
        self._cost_function = cost_function
        self._argument_index = argument_index
        self.elementwise = elementwise

    @property
    def cost_function(self):
        return self._cost_function
    
    @cost_function.setter 
    def cost_function(self, new_cost_function):
        self._cost_function = new_cost_function 
        self._gradient = grad(new_cost_function, self._argument_index)
        if self.elementwise:
            self._gradient = elementwise_grad(new_cost_function, self._argument_index)

    def __call__(self, X, y, theta, lmbda):
        """
        Returns gradient of current cost function.
        """
        return self._gradient(X, y, theta, lmbda)

############ Optimization methods (and related) ############
class LearningRate:

    def __init__(self, t0:float, t1:float, N:int, batch_size:int, name=None, const=None):
        
        self.name = name 
        if name is None:
            if const is None:
                self.name = f"callable({t0}, {t1})"
            else:
                print("hei")
                self.name = str(const)

        self.t0 = t0; self.t1 = t1
        self.N = N; self.batch_size = batch_size
        self.const = const 

        self._call = self._varying_learning_rate
        if not (const is None):
            self._call = self._const_learning_rate

    def _varying_learning_rate(self, epoch, i):
        t = epoch * int(self.N / self.batch_size) + i 
        return self.t0 / (t + self.t1)
    
    def _const_learning_rate(self, epoch, i):
        return self.const
    
    def __call__(self, epoch, i):
        return self._call(epoch, i)
    
    def __str__(self):
        return self.name 
 
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
        Class to be inherited by PlaneGradient, Adagrad, RMSprop or Adam, representing the method of choice for Stochastic Gradient Descent (SGD).
        """
        # self._learning_rate = learning_rate
        self.learning_rate = learning_rate # calls property method

        self.momentum = momentum
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay_rate = decay_rate
        self.velocity = None  # For momentum term

    @property
    def learning_rate(self):
        return self._learning_rate 
    
    @learning_rate.setter
    def learning_rate(self, new_learning_rate):
        # Makes sure learning_rate is a callable
        if not isinstance(new_learning_rate, LearningRate):
            tmp = new_learning_rate
            
            self._str_learning_rate = str(tmp)
            new_learning_rate = LearningRate(0, 0, 0, 0, self._str_learning_rate, const=new_learning_rate)
            self._learning_rate = new_learning_rate
        else:
            self._learning_rate = new_learning_rate
            self._str_learning_rate = "callable"

    def initialize_velocity(self, theta):
        if self.velocity is None:
            self.velocity = np.zeros_like(theta)

    def __call__(self, theta:np.ndarray, gradient:np.ndarray, epoch_index:int, batch_index:int):
        """
        Arguments
        * theta:            variable to be updated
        * gradient:         gradient
        * epoch_index:      current epoch
        * batch_index:      current batch

        Returns
        updated theta
        """
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def copy(self):
        """
        Creates and returns a copy of itself
        """

        raise Warning("Creating copy of Optimizer class, you probably want to create a copy of a subclass of Optimizer?")
        optimizer = Optimizer(learning_rate=self.learning_rate, momentum=self.momentum, 
                              epsilon=self.epsilon, beta1=self.beta1, beta2=self.beta2, decay_rate=self.decay_rate)
        
        return optimizer

    def __str__(self):
        return "Not defined"

class PlaneGradient(Optimizer):

    __doc__ = Optimizer.__doc__
    def __init__(self, learning_rate=0.01, momentum=0.9, epsilon=None, beta1=None, beta2=None, decay_rate=None):
        """
        Class implementing basic Gradient Descent with optional momentum. Does support stochastic GD as well.
        """
        super().__init__(learning_rate, momentum)

    def __call__(self, *args):

        theta, gradient = args[0], args[1]; epoch_index = None; batch_index = None 

        if len(args) == 4:
            epoch_index, batch_index = args[2], args[3]
        self.initialize_velocity(theta)

        self.velocity = self.momentum * self.velocity - self._learning_rate(epoch_index, batch_index) * gradient

        # Apply momentum
        theta += self.velocity
        return theta

    def __str__(self):
        return f"Plane Gradient descent: eta: {self.learning_rate}, momentum: {self.momentum}"
    
    def copy(self):
        """
        Creates and returns a copy of itself
        """

        optimizer = PlaneGradient(learning_rate=self.learning_rate, momentum=self.momentum, 
                              epsilon=self.epsilon, beta1=self.beta1, beta2=self.beta2, decay_rate=self.decay_rate)
        
        return optimizer

class Adagrad(Optimizer):

    __doc__ = Optimizer.__doc__
    def __init__(self, learning_rate=0.01, epsilon=1e-8, momentum=None, beta1=None, beta2=None, decay_rate=None):
        """
        Class implementing the Adagrad optimization algorithm. 
        Adagrad adapts the learning rate for each parameter based on the accumulation of past squared gradients.
        """

        super().__init__(learning_rate, epsilon=epsilon)
        self.G = None  # Accumulated squared gradients
    
    def initialize_accumulation(self, theta):
        # Initializes accumulation matrix G if not already initialized
        if self.G is None:
            self.G = 0

    def __call__(self, *args):
        theta, gradient, epoch_index, batch_index = args[0], args[1], args[2], args[3]

        self.initialize_accumulation(theta)

        # Accumulating squared gradients
        self.G += gradient*gradient

        #Updating theta
        theta -= (self._learning_rate(epoch_index, batch_index) / (np.sqrt(self.G) + self.epsilon)) * gradient
        
        return theta
    
    def __str__(self):
        return f"Adagrad: eta: {self._str_learning_rate}, eps: {self.momentum}"
    
    def copy(self):
        """
        Creates and returns a copy of itself
        """

        optimizer = Adagrad(learning_rate=self.learning_rate, momentum=self.momentum, 
                              epsilon=self.epsilon, beta1=self.beta1, beta2=self.beta2, decay_rate=self.decay_rate)
        
        return optimizer

class RMSprop(Optimizer):
    __doc__ = Optimizer.__doc__
    def __init__(self, learning_rate=0.01, decay_rate=0.99, epsilon=1e-8, momentum=None, beta1=None, beta2=None):
        """
        Class implementing the RMSprop optimization algorithm.
        RMSprop maintains a moving average of the squared gradients to normalize the gradient.
        """

        super().__init__(learning_rate, epsilon=epsilon, decay_rate=decay_rate)
        self.G = None

    def initialize_accumulation(self, theta):
        # Initializes accumulation matrix G if not already initialized
        if self.G is None:
            self.G = np.zeros_like(theta)

    def __call__(self, *args):
        theta, gradient, epoch_index, batch_index = args[0], args[1], args[2], args[3]

        self.initialize_accumulation(theta)

        # Updating moving average of the squared gradients
        self.G = self.decay_rate * self.G + (1 - self.decay_rate) * gradient*gradient

        # Update theta
        theta -= (self._learning_rate(epoch_index, batch_index) / (np.sqrt(self.G) + self.epsilon)) * gradient
        return theta

    def __str__(self):
        return f"RMSprop: eta: {self._str_learning_rate}, eps: {self.momentum}, decay_rate = {self.decay_rate}"
    
    def copy(self):
        """
        Creates and returns a copy of itself
        """

        optimizer = RMSprop(learning_rate=self.learning_rate, momentum=self.momentum, 
                              epsilon=self.epsilon, beta1=self.beta1, beta2=self.beta2, decay_rate=self.decay_rate)
        
        return optimizer

class Adam(Optimizer):
    __doc__ = Optimizer.__doc__
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=None, momentum=None):
        """
        Class implementing the Adam optimization algorithm.
        Adam combines the advantages of both RMSprop and momentum by maintaining both first and second moment estimates.
        """

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
        theta, gradient, epoch_index, batch_index = args[0], args[1], args[2], args[3]

        self.t += 1
        self.initialize_moments(theta)

        # Update biased first and second moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2

        # Compute bias-corrected moments
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Update theta
        theta -= (self._learning_rate(epoch_index, batch_index) / (np.sqrt(v_hat) + self.epsilon)) * m_hat
        return theta
    
    def __str__(self):
        return f"Adam: eta: {self._str_learning_rate}, eps: {self.momentum}, beta1 = {self.beta1}, beta2 = {self.beta2}"
    
    def copy(self):
        """
        Creates and returns a copy of itself
        """

        optimizer = Adam(learning_rate=self.learning_rate, momentum=self.momentum, 
                              epsilon=self.epsilon, beta1=self.beta1, beta2=self.beta2, decay_rate=self.decay_rate)
        
        return optimizer


class Scalers:

    def __init__(self, scaler_name:str):
        scaler_names = ["STANDARD", "MINMAX"]
        scalers = [Scalers.standard_scaler, Scalers.minmax_scaler]

        try:
            index = scaler_names.index(scaler_name.upper())
            self._call = scalers[index]
        except:
            raise TypeError(f"Did not recognize '{scaler_name}' as a scaler type, available: ['standard', 'minmax']")
        
        self.scaler_name = scaler_name

    def __call__(self, X_train:np.ndarray, X_test:np.ndarray) -> list[np.ndarray, np.ndarray]:
        return self._call(X_train, X_test)
    
    def __str__(self):
        return self.scaler_name
    
    @staticmethod
    def standard_scaler(X_train, X_test):
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test
    
    @staticmethod
    def minmax_scaler(X_train, X_test):
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test
