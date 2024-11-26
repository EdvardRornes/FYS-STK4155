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
        
class NeuralNetwork:

    def __init__(self) -> None:
        self.input_size =           None
        self.hidden_layers =        None
        self.output_size =          None
        self.layers =               None
        self.weights =              None
        self.biases =               None
        self.activation_func_name = None
        self.loss_function =        None
        self.optimizer =            None

        self.lambda_reg =           None
        self.alpha =                None

        # Initialize activation functions mapping
        self.activation_map =       None 
        
        self.activation, self.activation_derivative = None, None

############ Feed-forward neural network ############
class FFNN(NeuralNetwork):
    def __init__(self, input_size:int, hidden_layers:list, output_size:int, optimizer:Optimizer, activation, 
                 lambda_reg=0.0, alpha=0.1, loss_function='mse'):
        """
        Implements the Feedforward Neural Network (FFNN).
        
        Positional Arguments
        * input_size:           Number of input features.
        * hidden_layers:        List of integers representing the size of each hidden layer.
        * output_size:          Number of output neurons.
        * optimizer:            Type of optimizer used for weights and biases (PlaneGradient, AdaGrad, RMSprop or Adam)
        * activation:           Activation function to use ('relu', 'sigmoid', 'lrelu').

        Keyword Arguments
        * loss_function (str):  Loss function to use ('mse' or 'bce').
        * alpha (float):        Leaky ReLU parameter (only for 'lrelu').
        """
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        self.activation_func_name = activation
        self.loss_function = loss_function  # Added loss function parameter
        self.optimizer = optimizer

        self.lambda_reg = lambda_reg
        self.alpha = alpha

        # Initialize activation functions mapping
        self.activation_map = {
            'relu': (Activation.relu, Activation.relu_derivative),
            'sigmoid': (Activation.sigmoid, Activation.sigmoid_derivative),
            'lrelu': (lambda z: Activation.Lrelu(z, self.alpha), lambda z: Activation.Lrelu_derivative(z, self.alpha))
        }
        
        self.activation, self.activation_derivative = self.activation_map.get(self.activation_func_name.lower(), (Activation.relu, Activation.relu_derivative))

        # Initialize weights, biases, and Adam parameters
        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        """
        Initializes weights and biases. 
        """
        self.weights = []; self.weight_optimizers = []
        self.biases = []; self.bias_optimizers = []
        for i in range(len(self.layers) - 1):
            weight_matrix = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2. / self.layers[i])
            self.weights.append(weight_matrix)
            bias_vector = np.zeros((1, self.layers[i + 1]))
            self.biases.append(bias_vector)
            
            self.weight_optimizers.append(self.optimizer.copy())
            self.bias_optimizers.append(self.optimizer.copy())
    
    def forward(self, X:np.ndarray):
        """
        Forward propagating through the network, computing the output from the input (X).
        """

        self.activations = [X]
        self.z_values = []
        A = X

        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i] + self.biases[i]
            self.z_values.append(Z)
            A = self.activation(Z)
            A = np.clip(A, -1e10, 1e10)
            self.activations.append(A)

        Z = A @ self.weights[-1] + self.biases[-1]
        self.z_values.append(Z)
        if self.loss_function.lower() == 'mse':
            # Linear activation of regression
            A_output = self.activation(Z)
        else:
            # Sigmoid activation for classification
            A_output = Activation.sigmoid(Z)
        self.activations.append(A_output)
        return A_output

    def compute_loss(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        Computes the loss using the true (y_true) output data and the predicted (y_pred) output data.
        """
        if self.loss_function == 'mse':
            return np.mean((y_pred - y_true) ** 2)
        elif self.loss_function == 'bce':
            # Adding a small epsilon to avoid log(0)
            return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        else:
            raise ValueError("Invalid loss function specified. Use 'mse' or 'bce'.")

    def backward(self, X:np.ndarray, y:np.ndarray, epoch_index:int, batch_index:int):
        """
        Propagating backwards through the network, updating the weights and biases.
        """
        m = X.shape[0]
        y = y.reshape(-1, 1)

        output = self.activations[-1]

        delta = output - y

        for i in reversed(range(len(self.weights))):
            dw = (self.activations[i].T @ delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            dw += (self.lambda_reg / m) * self.weights[i]

            self.weights[i] = self.weight_optimizers[i](self.weights[i], dw, epoch_index, batch_index)
            self.biases[i] = self.bias_optimizers[i](self.biases[i], db, epoch_index, batch_index)

            if i > 0:
                delta = (delta @ self.weights[i].T) * self.activation_derivative(self.z_values[i - 1])

    def train(self, X:np.ndarray, y:np.ndarray, epochs=1000, batch_size=None, shuffle=True) -> list:
        """
        Trains the neural network. 

        Positional Arguments
        * X:                input data
        * y:                output data

        Keyword Arguments
        * epochs (int):     `number of iterations'
        * batch_size (int): size of data partition
        * shuffle (bool):   if true: shuffles the data for each epoch

        Returns
        loss history
        """
        self.initialize_weights_and_biases()
        loss_history = []
        m = X.shape[0]
        if batch_size is None:
            batch_size = m

        start_time = time.time()
        N = epochs * (m // batch_size + (1 if m % batch_size != 0 else 0)); counter = 0
        for epoch_index in range(epochs):
            if shuffle:
                indices = np.random.permutation(m)
                X, y = X[indices], y[indices]

            for batch_index in range(0, m, batch_size):
                X_batch = X[batch_index:batch_index + batch_size]
                y_batch = y[batch_index:batch_index + batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch, epoch_index, batch_index)

                counter += 1
                tmp = counter/N*100
                print(f"Training FNN, {tmp:.1f}% complete, time taken: {time.time() - start_time:.1f}s", end="\r")

            # Calculate and store the loss
            loss = self.compute_loss(y_batch, self.activations[-1])
            loss_history.append(loss)
        
        print(f"Training FFNN, 100.0% complete, time taken: {time.time() - start_time:.1f}s         ")
        return loss_history

    def predict(self, X:np.ndarray):
        """
        Predicts on the data X.
        """
        return self.forward(X)

    def accuracy(self, X:np.ndarray, y:np.ndarray):
        """
        Computes and returns the accuracy score.
        """
        predictions = self.predict(X)
        if self.loss_function == 'bce':
            predicted_classes = (predictions > 0.5).astype(int)
            return np.mean(predicted_classes.flatten() == y.flatten())
        else:
            return np.mean(np.round(predictions) == y.flatten())
        
    @property
    def learning_rate(self):
        return self.weight_optimizer.learning_rate
    
    @learning_rate.setter
    def learning_rate(self, new_learning_rate):
        self.optimizer.learning_rate = new_learning_rate
        for i in range(len(self.weight_optimizers)):
            self.weight_optimizers[i].learning_rate = new_learning_rate
            self.bias_optimizers[i].learning_rate = new_learning_rate

class RNN(NeuralNetwork):
    def __init__(self, input_size:int, hidden_layers:list, output_size:int, optimizer:Optimizer, activation, 
                 lambda_reg=0.0, alpha=0.1, loss_function='mse'):
        """
        Implements the Recurrent Neural Network (RNN).
        
        Positional Arguments
        * input_size:           Number of input features.
        * hidden_layers:        List of integers representing the size of each hidden layer.
        * output_size:          Number of output neurons.
        * optimizer:            Type of optimizer used for weights and biases (PlaneGradient, AdaGrad, RMSprop or Adam)
        * activation:           Activation function to use ('relu', 'sigmoid', 'lrelu').

        Keyword Arguments
        * loss_function (str):  Loss function to use ('mse' or 'bce').
        * alpha (float):        Leaky ReLU parameter (only for 'lrelu').
        """
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        self.activation_func_name = activation
        self.loss_function = loss_function  # Added loss function parameter
        self.optimizer = optimizer

        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.hidden_size = hidden_layers[-1]  # Define the hidden layer size

        # Initialize activation functions mapping
        self.activation_map = {
            'relu': (Activation.relu, Activation.relu_derivative),
            'sigmoid': (Activation.sigmoid, Activation.sigmoid_derivative),
            'lrelu': (lambda z: Activation.Lrelu(z, self.alpha), lambda z: Activation.Lrelu_derivative(z, self.alpha))
        }
        
        self.activation, self.activation_derivative = self.activation_map.get(self.activation_func_name.lower(), (Activation.relu, Activation.relu_derivative))
        self.output_activation = self.activation

        # Initialize weights, biases, and optimizers
        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        """
        Initializes weights and biases. 
        """
        self.weights = []; self.weight_optimizers = []
        self.recurrent_weights = []; self.recurrent_weight_optimizers = []
        self.biases = []; self.bias_optimizers = []

        # Initialize input to hidden layer weights and biases
        weight_matrix = np.random.randn(self.layers[0], self.layers[1]) * np.sqrt(2. / self.layers[0])
        self.weights.append(weight_matrix)
        bias_vector = np.zeros((1, self.layers[1]))
        self.biases.append(bias_vector)
        self.weight_optimizers.append(self.optimizer.copy())
        self.bias_optimizers.append(self.optimizer.copy())

        # Initialize recurrent weights and hidden-to-hidden layer weights
        for i in range(1, len(self.layers) - 1):
            recurrent_weight_matrix = np.random.randn(self.layers[i], self.layers[i]) * np.sqrt(2. / self.layers[i])
            self.recurrent_weights.append(recurrent_weight_matrix)

            weight_matrix = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2. / self.layers[i])
            self.weights.append(weight_matrix)
            bias_vector = np.zeros((1, self.layers[i + 1]))
            self.biases.append(bias_vector)
            self.weight_optimizers.append(self.optimizer.copy())
            self.bias_optimizers.append(self.optimizer.copy())

            self.recurrent_weight_optimizers.append(self.optimizer.copy())

        # Output layer weights and biases
        self.output_weights = np.random.randn(self.hidden_size, self.layers[-1]) * np.sqrt(2. / self.hidden_size)
        self.output_biases = np.zeros((1, self.layers[-1]))
        self.output_optimizer = self.optimizer.copy()
        self.output_bias_optimizer = self.optimizer.copy()
    
    def forward(self, X: np.ndarray):
        """
        Forward propagates through the RNN for sequential 1D time-series input X.
        
        Arguments:
        * X:                Input data of shape (batch_size, sequence_length, input_size).
        
        Returns:
        * Outputs:          Array of shape (batch_size, sequence_length, output_size).
        * Hidden_states:    List of hidden states for each layer and time step.
        """
        batch_size, sequence_length, input_size = X.shape

        # Initialize the hidden states for all layers (batch_size, hidden_size)
        h_t = [np.zeros((batch_size, size)) for size in self.layers[1:-1]]  # List for each hidden layer
        self.hidden_states = [[] for _ in range(len(self.layers) - 2)]  # List of lists for each layer
        self.z_s = [[] for _ in range(len(self.layers) - 2)]            # Pre-activation values
        self.outputs = []  # Store outputs across time steps

        # Forward pass through each time step
        for t in range(sequence_length):
            x_t = X[:, t, :]  # Input at time step t (batch_size, input_size)

            # Iterate through all hidden layers
            for layer_idx in range(len(self.layers) - 2):
                # Compute z for current layer
                if layer_idx == 0:
                    # First hidden layer: input comes from x_t and its own previous hidden state
                    z = x_t @ self.weights[layer_idx] + h_t[layer_idx] @ self.recurrent_weights[layer_idx] + self.biases[layer_idx]
                else:
                    # Subsequent hidden layers: input comes from previous hidden layer's current h and their own previous hidden state
                    z = h_prev @ self.weights[layer_idx] + h_t[layer_idx] @ self.recurrent_weights[layer_idx] + self.biases[layer_idx]
                
                # Apply activation function
                h = self.activation(z)
                
                # Store pre-activation and activation
                self.z_s[layer_idx].append(z)
                self.hidden_states[layer_idx].append(h)
                
                # Update h_prev for next layer
                h_prev = h

                # Update the hidden state
                h_t[layer_idx] = h

            # Compute output from the last hidden layer
            o_t = h_prev @ self.output_weights + self.output_biases
            o_t = self.output_activation(o_t)  # Apply output activation function
            self.outputs.append(o_t)

        # Convert lists to arrays for easier handling
        # hidden_states[layer][time] -> (batch_size, hidden_size)
        self.hidden_states = [np.stack(layer, axis=1) for layer in self.hidden_states]  # List of arrays
        # Outputs: list of (batch_size, output_size) -> (batch_size, sequence_length, output_size)
        self.outputs = np.stack(self.outputs, axis=1)  # Shape: (batch_size, sequence_length, output_size)

        return self.outputs, self.hidden_states


    def backward(self, X: np.ndarray, y: np.ndarray, epoch_index: int, batch_index: int):
        """
        Performs backpropagation through time and updates the weights and biases.
        
        Arguments:
        * X: Input data of shape (batch_size, sequence_length, input_size).
        * y: Labels of shape (batch_size,).
        """
        batch_size, sequence_length, input_size = X.shape
        hidden_size = self.hidden_size

        # Initialize gradients
        dW_xh = np.zeros_like(self.weights[0])           # Shape: (input_size, hidden_size)
        dW_hh = np.zeros_like(self.recurrent_weights[0]) # Shape: (hidden_size, hidden_size)
        db_h = np.zeros_like(self.biases[0])             # Shape: (1, hidden_size)
        dW_hy = np.zeros_like(self.output_weights)       # Shape: (hidden_size, output_size)
        db_y = np.zeros_like(self.output_biases)         # Shape: (1, output_size)

        dh_next = np.zeros((batch_size, hidden_size))

        for t in reversed(range(sequence_length)):
            h_t = self.hidden_states[:, t, :]  # Shape: (batch_size, hidden_size)
            z_t = self.z_s[t]                  # Shape: (batch_size, hidden_size)
            x_t = X[:, t, :]                   # Shape: (batch_size, input_size)

            if t == sequence_length - 1:
                o_t = self.outputs[:, t, :]    # Shape: (batch_size, output_size)
                # Compute gradient of loss w.r.t o_t
                # Ensure y has shape (batch_size, output_size)
                y_true = y[:, None]            # Shape: (batch_size, 1)
                # Compute derivative of output activation
                dL_do_t = (o_t - y_true) * self.output_activation_derivative(o_t)  # Shape: (batch_size, output_size)

                # Gradients for output layer
                dW_hy += h_t.T @ dL_do_t       # Shape: (hidden_size, output_size)
                db_y += np.sum(dL_do_t, axis=0, keepdims=True)  # Shape: (1, output_size)

                dh = dL_do_t @ self.output_weights.T  # Shape: (batch_size, hidden_size)
            else:
                dh = dh_next

            dh_total = dh

            # Backprop through activation function
            dh_raw = dh_total * self.activation_derivative(z_t)  # Shape: (batch_size, hidden_size)

            # Gradients for weights and biases
            dW_xh += x_t.T @ dh_raw  # Shape: (input_size, hidden_size)

            if t > 0:
                h_prev = self.hidden_states[:, t -1, :]  # Shape: (batch_size, hidden_size)
            else:
                h_prev = np.zeros_like(h_t)  # Shape: (batch_size, hidden_size)

            dW_hh += h_prev.T @ dh_raw      # Shape: (hidden_size, hidden_size)
            db_h += np.sum(dh_raw, axis=0, keepdims=True)  # Shape: (1, hidden_size)

            # Compute gradient for next time step
            dh_next = dh_raw @ self.recurrent_weights[0].T  # Shape: (batch_size, hidden_size)

        # Update weights and biases using optimizers
        self.weights[0] = self.weight_optimizers[0].update(self.weights[0], dW_xh)
        self.recurrent_weights[0] = self.recurrent_weight_optimizers[0].update(self.recurrent_weights[0], dW_hh)
        self.biases[0] = self.bias_optimizers[0].update(self.biases[0], db_h)

        self.output_weights = self.output_optimizer.update(self.output_weights, dW_hy)
        self.output_biases = self.output_bias_optimizer.update(self.output_biases, db_y)

    def compute_loss(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        Computes the loss using the true (y_true) output data and the predicted (y_pred) output data.
        """
        if self.loss_function == 'mse':
            return np.mean((y_pred - y_true) ** 2)
        elif self.loss_function == 'bce':
            return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        else:
            raise ValueError("Invalid loss function specified. Use 'mse' or 'bce'.")

    def _create_sequences(self, t:np.ndarray, y:np.ndarray, window_size:int, step_size:int) -> list:
        """
        Segments the time-series data into sequences for RNN training.

        Arguments:
        * t:                Input data (1D 'time')
        * y:                Binary labels for presence of a gravitational wave (1D array).
        * window_size:      Number of time steps in each window (sequence length).
        * step_size:        Number of time steps to shift the window (overlapping control).

        Returns:
        * X:                Segmented time-series data (shape: num_windows, window_size, 1).
        * y_seq:            Labels for each window (shape: num_windows or num_windows, window_size).
        """
        X = []
        y_seq = []

        for i in range(0, len(t) - window_size + 1, step_size):
            # Create a window of input data
            window = t[i:i + window_size].reshape(-1, 1)  # Shape: (window_size, 1)
            X.append(window)

            # Create the corresponding label
            window_label = y[i:i + window_size]  # Shape: (window_size,)
            # Example: Label the whole window based on the presence of any gravitational wave
            y_seq.append(int(np.any(window_label == 1)))

        X = np.array(X)  # Shape: (num_windows, window_size, 1)
        y_seq = np.array(y_seq)  # Shape: (num_windows,)
        return X, y_seq

    def train(self, t:np.ndarray, y:np.ndarray, 
              epochs=1000, batch_size=None, window_size=None, step_size=10, shuffle=True) -> list:
        """
        Trains the neural network. 

        Positional Arguments
        * t:                input data (1D 'time')
        * y:                output data 

        Keyword Arguments
        * epochs (int):     `number of iterations'
        * batch_size (int): size of data partition
        * shuffle (bool):   if true: shuffles the data for each epoch

        Returns
        loss history
        """

        self.initialize_weights_and_biases()

        N = len(t)
        if window_size is None:
            window_size = int(N / 10)

        X, y_seq = self._create_sequences(t, y, window_size, step_size)

        loss_history = []
        m = X.shape[0]  # Total number of windows
        if batch_size is None:
            batch_size = m

        # For printing percentage/duration
        start_time = time.time()
        N = epochs * (m // batch_size + (1 if m % batch_size != 0 else 0)); counter = 0

        for epoch_index in range(epochs):
            if shuffle:
                indices = np.random.permutation(m)
                X, y_seq = X[indices], y_seq[indices]

            for batch_index in range(0, m, batch_size):
                X_batch = X[batch_index:batch_index + batch_size]
                y_batch = y_seq[batch_index:batch_index + batch_size]

                outputs, _ = self.forward(X_batch)
                self.backward(X_batch, y_batch, epoch_index, batch_index)

                counter += 1
                tmp = counter / N * 100
                print(f"Training RNN, {tmp:.1f}% complete, time taken: {time.time() - start_time:.1f}s", end="\r")

            # Calculate and store the loss
            loss = self.compute_loss(y_batch, outputs)
            loss_history.append(loss)
        
        print(f"Training RNN, 100.0% complete, time taken: {time.time() - start_time:.1f}s         ")
        return loss_history

    def predict(self, X:np.ndarray):
        """
        Predicts on the data X.
        """
        return self.forward(X)

    def accuracy(self, X:np.ndarray, y:np.ndarray):
        """
        Computes and returns the accuracy score.
        """
        predictions = self.predict(X)
        if self.loss_function == 'bce':
            predicted_classes = (predictions > 0.5).astype(int)
            return np.mean(predicted_classes.flatten() == y.flatten())
        else:
            return np.mean(np.round(predictions) == y.flatten())
        
    @property
    def learning_rate(self):
        return self.weight_optimizer.learning_rate
    
    @learning_rate.setter
    def learning_rate(self, new_learning_rate):
        self.optimizer.learning_rate = new_learning_rate
        for i in range(len(self.weight_optimizers)):
            self.weight_optimizers[i].learning_rate = new_learning_rate
            self.bias_optimizers[i].learning_rate = new_learning_rate
    
class Data:

    def __init__(self, neural_network:NeuralNetwork, additional_description=""):
        self.data = {
            "input_size":               str(neural_network.input_size),
            "hidden_layers":            str(neural_network.hidden_layers),
            "output_size":              str(neural_network.output_size),
            "activation":               str(neural_network.activation_func_name),
            "loss_function":            str(neural_network.loss_function),
            "optimizer":                str(neural_network.optimizer),
            "optimizer_parameters":     [neural_network.optimizer._str_learning_rate, 
                                         neural_network.optimizer.momentum,
                                         neural_network.optimizer.epsilon,
                                         neural_network.optimizer.beta1,
                                         neural_network.optimizer.beta2,
                                         neural_network.optimizer.decay_rate],
            "additional_description":   additional_description                                     
        }
    
    def store(self, filename_and_path:str, overwrite=False, stop=50):
        """Saves the analysis data to a file.

        Args:
            * filename:         name of the file to save the data to
            * overwrite         whether to overwrite the file if it exists
            * stop:             if 50 files for this 'type' of data exists, stop
        """
        filename_full = f"{filename_and_path}_{0}.pkl"

        if not overwrite:
            if os.path.exists(filename_full):
                i = 1
                while i <= stop: # May already be additional files created
                    filename_full = f"{filename_and_path}_{i}.pkl"
                    if not os.path.exists(filename_full):
                        with open(filename_full, 'wb') as f:
                            pickle.dump(self.data, f)
                        
                        return 
                    i += 1
            
                raise ValueError(f"You have {stop} additional files of this sort?")
                
        with open(filename_full, 'wb') as f:
            pickle.dump(self.data, f)
    
    def load_data(self, filename_and_path:str) -> dict:
        """Loads analysis data from a file.

        Args:
            * filename:         The name of the file to load the data from.
        """
        with open(filename_and_path, 'rb') as f:
            self.data = pickle.load(f)

        return self.data
