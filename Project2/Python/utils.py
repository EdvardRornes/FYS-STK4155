import numpy as np
import autograd.numpy as anp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import jax.numpy as jnp
from jax import grad, jit
from autograd import grad
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import sys, os
import time 
import pickle
import copy

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

def MSE(y,ytilde):
    n = len(y)
    return 1/n * np.sum(np.abs(y-ytilde)**2)

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

def sigmoid(x):
    return 1/(1+np.exp(-x))

def logistic_reg(x,w):
    return sigmoid(np.dot(x,w))

def logistic_cost(x,w,t,_lambda = 0):
    pred = logistic_reg(x,w)
    cost_inner = np.log(pred) * t + np.log(1 - pred) * (1 - t)
    
    #return -(np.sum(np.log(cost_inner)) + _lambda*np.sum(w**2))
    return -np.sum((cost_inner)) + _lambda*np.sum(w**2)
        
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

    def __call__(self, X, y, theta, lmbda):
        """
        Returns gradient of current cost function.
        """
        return self._gradient(X, y, theta, lmbda)

############ Design Matrix ############
def create_Design_Matrix(*args):
    """
    Creates design matrix. Can handle 1D and 2D.

    Arguments
        * x:        x-values (1D and 2D case)
        * y:        y-values (2D case only)
        * degree:   polynomial degree
    """

    if len(args[0]) == 2: # 2D case 
        x, y = args[0]
        degree = args[-1]
        
        degree -= 1 # Human to computer language 

        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int((degree+1)*(degree+2)/2) # Number of elements in beta
        X = np.ones((N,l))
        # X = np.vander(x, N=degree + 1, increasing=True)
        # return X
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
        
        # Create the design matrix with shape (len(x), degree + 1
        l = int((degree+1)*(degree+2)/2)
        X = np.vander(x, N=l, increasing=True)
        return X
    
############ Data generators ############
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

############ Optimization methods (and related) ############
class LearningRate:

    def __init__(self, t0:float, t1:float, N:int, batch_size:int, name=None, const=None):
        
        self.name = name 
        if name is None:
            if const is None:
                self.name = f"callable({t0}, {t1})"
            else:
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

    def __call__(self, args):
        raise NotImplementedError("This method should be implemented by subclasses")
    
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

        theta, gradient = args[0], args[1]; epochs = None; i = None 

        if len(args) == 4:
            epochs, i = args[2], args[3]
        self.initialize_velocity(theta)

        self.velocity = self.momentum * self.velocity - self._learning_rate(epochs, i) * gradient

        # Apply momentum
        theta += self.velocity
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

        super().__init__(learning_rate, epsilon=epsilon)
        self.G = None  # Accumulated squared gradients
    
    def initialize_accumulation(self, theta):
        # Initializes accumulation matrix G if not already initialized
        if self.G is None:
            self.G = 0

    def __call__(self, *args):
        theta, gradient, epochs, i = args[0], args[1], args[2], args[3]

        self.initialize_accumulation(theta)

        # Accumulating squared gradients
        self.G += gradient*gradient

        #Updating theta
        theta -= (self._learning_rate(epochs, i) / (np.sqrt(self.G) + self.epsilon)) * gradient
        
        return theta
    
    def __str__(self):
        return f"Adagrad: eta: {self._str_learning_rate}, eps: {self.momentum}"

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
        theta, gradient, epochs, i = args[0], args[1], args[2], args[3]

        self.initialize_accumulation(theta)

        # Updating moving average of the squared gradients
        self.G = self.decay_rate * self.G + (1 - self.decay_rate) * gradient*gradient

        # Update theta
        theta -= (self._learning_rate(epochs, i) / (np.sqrt(self.G) + self.epsilon)) * gradient
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
        theta -= (self._learning_rate(epochs, i) / (np.sqrt(v_hat) + self.epsilon)) * m_hat
        return theta
    
    def __str__(self):
        return f"Adam: eta: {self._str_learning_rate}, eps: {self.momentum}, beta1 = {self.beta1}, beta2 = {self.beta2}"

############ Descent ############
class DescentSolver:

    def __init__(self, optimization_method:Optimizer, degree:int, mode="GD", cost_function=None):
        """
        This class uses an 'optimization method' (PlaneGD/SGD, AdaGrad, RMSprop, Adam) to create a callable on the form: 
                descentSovler(X, y, epochs, lmbda, batch_size=None).

        Arguments 
            * optimization_method:          represents algorithm for computing next theta
            * degree:                       polynomial or 'feature' degree
            * mode (GD):                    either gradient descent (GD) or stochastic GD (SGD)
            * cost_function (None):         cost function of choice, on the form (X, y, theta, lmbda)
        """

        # Storing callables
        self._cost_function = cost_function
        self._gradient = AutoGradCostFunction(cost_function) 
        self._optimization_method = optimization_method

        # Storing parameters
        self._theta = None; self.degree = degree
        self.lmbda = 0 # (corresponding to OLS) still needs to be given when calling

        self._message = str(optimization_method)
        
        if mode.upper() == "GD":
            self._descent_method_call = self._GD  
        elif mode.upper() == "SGD":
            self._descent_method_call = self._SGD  
    
    def __call__(self, *args, theta="RANDOM") -> list:
        """
        Arguments
            * X:                        design matrix 
            * y:                        (given) output data
            * epochs:                   number of itterations
            * lmbda:                    for gradient, set to zero for OLS gradient
            * batch_size (optional):    batch size
        
        Returns
            * self._theta:              polynomial coefficient
        """

        if self._gradient is None:
            raise ValueError(f"No gradient is given for {str(self._optimization_method)}-method.")
        
        print(f"Starting training w/ {self._optimization_method}")
        
        self._descent_method_call(*args, theta=theta)
        return self._theta
    
    @property
    def cost_function(self):
        return self._cost_function
    
    @cost_function.setter 
    def cost_function(self, new_cost_function):
        self._cost_function = new_cost_function
        self._gradient = AutoGradCostFunction(new_cost_function)

    ######## Private methods ########
    def _GD(self, *args, theta="RANDOM"):
        """
        Gradient Descent(X, y, epochs, lmbda)
        """

        start_time = time.time()

        if len(args) != 4:
            if not (len(args) == 5 and args[-1] is None):
                raise ValueError(f"Gradient descent needs 4 arguments: X, y, N, lmbda, not {len(args)}")
        
        X, y, epochs, lmbda = args[0], args[1], args[2], args[3]

        self.lmbda = lmbda # passed into self._gradients_call method 

        if isinstance(theta, str):
            if theta.upper() == "RANDOM":
                self._theta = np.random.randn(X.shape[1], 1)
        else:
            self._theta = copy.deepcopy(theta)
        for iter in range(epochs):
            gradient = self._gradients_call(X, y)
            self._update(gradient, None, None)

            current_time = time.time()
            print(f"GD: {iter/epochs*100:.1f}%, time elapsed: {current_time-start_time:.1f}s", end="\r")

        print(f"GD: 100.0%, time elapsed: {current_time-start_time:.1f}s         ")
    
    def _SGD(self, *args, theta="RANDOM"):
        """
        Stochastic Gradient Descent(X, y, epochs, lmbda, batch_size)
        """

        start_time = time.time()
        if len(args) != 5:
            raise ValueError(f"Gradient descent needs 5 arguments: X, y, epochs, lmbda, batch_size, not {len(args)}")
        
        X, y, epochs, lmbda, batch_size = args[0], args[1], args[2], args[3], args[4]
        
        self.lmbda = lmbda # passed into self._gradients_call method 

        m = int(X.shape[0]/batch_size) # number of minibatches
        if isinstance(theta, str):
            if theta.upper() == "RANDOM":
                self._theta = np.random.randn(X.shape[1], 1)
        else:
            self._theta = copy.deepcopy(theta)

        N = epochs * m

        for epoch in range(epochs):
            for i in range(m):
                miniBach = np.random.randint(m)

                miniBachMin, miniBachMax = batch_size * miniBach,(batch_size) * (miniBach+1)
                X_i = X[miniBachMin: miniBachMax]; y_i = y[miniBachMin:miniBachMax]

                gradient = self._gradients_call(X_i, y_i)
                self._update(gradient, epoch, i)

                current_time = time.time()

            tmp = epoch * m /N * 100
            print(f"SGD: {tmp:.1f}%, time elapsed: {current_time-start_time:.1f}s", end="\r")

        print(f"SGD: 100.0%, time elapsed: {current_time-start_time:.1f}s         ")

    def _update(self, gradient:np.ndarray, epochs:int, i:int):
        """
        Updates the previous theta-value
        Arguments
            * gradient:         
        """
        self._theta = self._optimization_method(self._theta, gradient, epochs, i) 

    def _gradients_call(self, X, y):
        return self._gradient(X, y, self._theta, self.lmbda)

class DescentAnalyzer:

    def __init__(self, x: np.ndarray, y: np.ndarray,
                 degree: int, epochs:int, batch_size=None, GD_SGD="GD",
                 momentum=0, epsilon=1e-8, beta1=0.9, beta2=0.999, decay_rate=0.9, print_percentage=True,
                 test_size=0.2):
        
        self.x = x
        self.y = y
        self.optimizer_name = None
        self.degree = degree
        self.epochs = epochs
        self.batch_size = batch_size
        self.GD_SGD = GD_SGD
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay_rate = decay_rate
        
        self.test_size = 0.2

        self._print_percentage = print_percentage

        self.lambdas = None
        # Placeholder for results
        self.thetas = []
        self.learning_rates = []

        # Metrics:
        self.MSE_test = None; self.MSE_train = None
        self.R2_test = None; self.R2_train = None

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
            "decay_rate": self.decay_rate,
            "MSE_test": self.MSE_test,
            "MSE_train": self.MSE_train,
            "x":    self.x,
            "y":    self.y,
            "R2_test": self.R2_test,
            "R2_train": self.R2_train,
            "lambdas": self.lambdas
        }

    def _analyze(self, optimization_method:Optimizer, cost_function:callable, learning_rate:list, lmbdas:list, N_bootstraps:int):
        """
        ...
        """
        self.data["optimizer_name"] = str(optimization_method)

        descentSolver = DescentSolver(optimization_method, self.degree, mode=self.GD_SGD)
        descentSolver.cost_function = cost_function

        X = create_Design_Matrix(self.x, self.degree)
        X_train, X_test, z_train, z_test = train_test_split(X, self.y.reshape(-1,1), test_size=0.2)
    
        self.MSE_test = np.zeros((len(learning_rate), len(lmbdas))); self.MSE_train = np.zeros((len(learning_rate), len(lmbdas)))
        self.R2_test = np.zeros((len(learning_rate), len(lmbdas))); self.R2_train = np.zeros((len(learning_rate), len(lmbdas)))

        start_time = time.time()

        for i in range(len(learning_rate)):

            descentSolver._optimization_method.learning_rate = learning_rate[i]
            self.learning_rates.append(descentSolver._optimization_method.learning_rate)
            
            for j in range(len(lmbdas)):
                MSE_test_tmp = np.zeros(N_bootstraps); R2_test_tmp = np.zeros(N_bootstraps)
                MSE_train_tmp = np.zeros(N_bootstraps); R2_train_tmp = np.zeros(N_bootstraps)
                
                for k in range(N_bootstraps):
                    blockPrint()

                    theta = np.random.randn(X.shape[1], 1)
                    
                    theta = descentSolver(X_train, z_train, self.epochs, lmbdas[j], self.batch_size, theta=theta)
                    enablePrint()

                    z_predict_test = X_test @ theta 
                    z_predict_train = X_train @ theta 

                    z_predict_test = np.nan_to_num(z_predict_test)
                    z_predict_train = np.nan_to_num(z_predict_train)

                    MSE_test_tmp[k] = MSE(z_test, z_predict_test)
                    MSE_train_tmp[k] = MSE(z_train, z_predict_train)

                    R2_test_tmp[k] = r2_score(z_test, z_predict_test)
                    R2_train_tmp[k] = r2_score(z_train, z_predict_train)

                self.MSE_test[i,j] = np.mean(MSE_test_tmp); self.MSE_train[i,j] = np.mean(MSE_train_tmp)
                self.R2_test[i,j] = np.mean(R2_test_tmp); self.R2_train[i,j] = np.mean(R2_train_tmp)

            if self._print_percentage:
                print(f"Analyzing, {(i+1)/(len(learning_rate))*100:.1f}%, duration: {time.time() - start_time:.1f}s", end="\r")

            self.thetas.append(theta)
        self.lambdas = lmbdas

        if self._print_percentage:
            print(f"Analyzing, 100%, duration: {time.time() - start_time:.1f}s            ")
    
    def run_analysis(self, optimization_method:Optimizer, cost_function:callable, learning_rate:float|list, lmbdas=0, N_bootstraps=1) -> np.ndarray:
        """
        Runs the descent analysis and stores the results.
        """

        if not issubclass(type(optimization_method), Optimizer):
            raise ValueError(f"Unknown optimizer '{self.optimizer_name}'.")
        
        if (not isinstance(learning_rate, list)) and (not isinstance(learning_rate, np.ndarray)):
            learning_rate = [learning_rate]

        if (not isinstance(lmbdas, list)) and (not isinstance(lmbdas, np.ndarray)):
            lmbdas = [lmbdas]

        if not (callable(cost_function)):
            raise TypeError(f"Argument 'cost_function' {(type(cost_function))} is not callable")
        
        self._analyze(optimization_method, cost_function, learning_rate, lmbdas, N_bootstraps)
        self._update_data()

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
        
        learning_rates_values = []
        for i in range(len(self.learning_rates)):
            learning_rates_values.append(str(self.learning_rates[i])) # each element should be instance of LearningRate

        self.data["learning_rates"] =  learning_rates_values
        self.data["lambdas"] =  self.lambdas

        self.data["batch_size"] =  self.batch_size
        self.data["GD_SGD"] =  self.GD_SGD; self.data["momentum"] =  self.momentum
        self.data["epsilon"] =  self.epsilon; self.data["beta1"] =  self.beta1
        self.data["beta2"] =  self.beta2; self.data["decay_rate"] =  self.decay_rate
        self.data["MSE_test"] = self.MSE_test; self.data["MSE_train"] = self.MSE_train
        self.data["R2_test"] = self.R2_test; self.data["R2_train"] = self.R2_train

############ Activation functions ############
class Activation:
    @staticmethod
    def sigmoid(z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

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

############ Analyzing/creating data ############
def create_data(x:np.ndarray, y:np.ndarray, method:str, epochs:int, learning_rates:list, lmbdas:list, batch_size=None, N_bootstraps=30, overwrite=False) -> None:

    # Choose method:
    methods = [PlaneGradient, Adagrad, RMSprop, Adam]
    methods_name = ["PlaneGradient", "Adagrad", "RMSprop", "Adam"]
    methods_name_upper = ["PLANEGRADIENT", "ADAGRAD", "RMSPROP", "ADAM"]
    if method.upper() in methods_name_upper:
        method_index = methods_name_upper.index(method.upper())
    else:
        raise TypeError(f"What is '{method}'?")
    
    method = methods[method_index]
    GD_SGD = "SGD"
    if batch_size is None:
        GD_SGD = "GD"

    # Saving parameters:
    file_path = f"../Data/Regression/{methods_name[method_index]}"
    os.makedirs(file_path, exist_ok=True)
    size = len(lmbdas)
    filename_OLS = file_path + f"/OLS{size}x{size}"
    filename_Ridge = file_path + f"/Ridge{size}x{size}"

    # Making sure there are as many lmdas as learning_rates
    assert len(lmbdas) == len(learning_rates), f"The length og 'lmbdas' need to be the same as 'learning_rates'."

    # Setting up optimizer:
    method = method()

    # Analyzer setup
    analyzer_OLS = DescentAnalyzer(x, y, 5, epochs,
        batch_size=batch_size,
        GD_SGD=GD_SGD)
    
    analyzer_Ridge = DescentAnalyzer(x, y, 5, epochs,
        batch_size=batch_size,
        GD_SGD=GD_SGD)

    ############## OLS ##############
    analyzer_OLS.run_analysis(method, CostRidge, learning_rates, 0, N_bootstraps)

    ############## Ridge ##############
    analyzer_Ridge.run_analysis(method, CostRidge, learning_rates, lmbdas, N_bootstraps)

    analyzer_OLS.save_data(filename_OLS, overwrite=overwrite)
    analyzer_Ridge.save_data(filename_Ridge, overwrite=overwrite)

def analyze_save_data(method:str, size:int, index:int, key="MSE_train"):
    
    
    methods = ["PlaneGradient", "Adagrad", "RMSprop", "Adam"]
    methods_name_upper = ["PLANEGRADIENT", "ADAGRAD", "RMSPROP", "ADAM"]
    if method.upper() in methods_name_upper:
        method_index = methods_name_upper.index(method.upper())
    else:
        raise TypeError(f"What is '{method}'?")
    
    method = methods[method_index]

    file_path = f"../Data/Regression/{method}"

    with open(f"{file_path}/OLS{size}x{size}_{index}.pkl", 'rb') as f:
        data_OLS = pickle.load(f)

    with open(f"{file_path}/Ridge{size}x{size}_{index}.pkl", 'rb') as f:
        data_Ridge = pickle.load(f)

    lmbdas = data_Ridge["lambdas"]
    learning_rates = data_Ridge["learning_rates"]

    learning_rates = [float(x) for x in learning_rates]

    OLS_MSE = data_OLS[key]
    Ridge_MSE = data_Ridge[key]

    ############# Plotting #############
    fig, ax = plt.subplots(1, figsize=(12,7))
    ax.plot(learning_rates, OLS_MSE)
    ax.set_xlabel(r"$\eta$")
    ax.set_yscale("log")
    ax.set_ylabel(r"MSE")
    ax.set_title(f"{method} using OLS cost function")


    tick = ticker.ScalarFormatter(useOffset=False, useMathText=True)
    tick.set_powerlimits((0,0))

    xtick_labels = [f"{l:.1e}" for l in lmbdas]
    ytick_labels = [f"{l:.1e}" for l in learning_rates]

    fig, ax = plt.subplots(figsize = (12, 7))
    sns.heatmap(Ridge_MSE, ax=ax, cmap="viridis", annot=True, xticklabels=xtick_labels, yticklabels=ytick_labels)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\eta$')
    ax.set_title(f"{method} using Ridge cost function")
    plt.tight_layout()
    plt.show()

    # Saving
    save = input("Save (y/n)? ")
    if save.upper() in ["Y", "YES", "YE"]:
        while True:
            only_less_than = input("Exclude values less than: ")
            plot_2D_parameter_lambda_eta(lmbdas, learning_rates, Ridge_MSE, only_less_than=float(only_less_than))
            plt.show()

            happy = input("Happy (y/n)? ")
            if happy.upper() in ["Y", "YES", "YE"]:
                title = input("Title: ") 
                filename = input("Filename: ")
                latex_fonts()
                plot_2D_parameter_lambda_eta(lmbdas, learning_rates, Ridge_MSE, only_less_than=float(only_less_than), title=title, savefig=True, filename=filename)
                plt.show()
                exit()
            
            elif happy.upper() in ["Q", "QUIT", "X"]:
                exit()

class FFNN:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', alpha=0.01, lambda_reg=0.0):
        """
        Initialize the Feedforward Neural Network (FFNN).

        Parameters:
        - input_size (int): Number of input features.
        - hidden_layers (list): List of integers representing the size of each hidden layer.
        - output_size (int): Number of output neurons.
        - activation (str): Activation function to use ('relu', 'sigmoid', 'lrelu').
        - alpha (float): Leaky ReLU parameter (only for 'lrelu').
        - lambda_reg (float): L2 regularization parameter.
        """
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        self.activation_func = activation
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        
        # Initialize activation functions mapping
        self.activation_map = {
            'relu': (Activation.relu, Activation.relu_derivative),
            'sigmoid': (Activation.sigmoid, Activation.sigmoid_derivative),
            'lrelu': (lambda z: Activation.Lrelu(z, self.alpha), lambda z: Activation.Lrelu_derivative(z, self.alpha))
        }
        
        # Initialize the activation function
        self.activation, self.activation_derivative = self.activation_map.get(self.activation_func.lower(), (Activation.relu, Activation.relu_derivative))

        # Initialize weights and biases
        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            weight_matrix = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2. / self.layers[i])
            self.weights.append(weight_matrix)
            bias_vector = np.zeros((1, self.layers[i + 1]))
            self.biases.append(bias_vector)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        A = X

        # Hidden layers
        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i] + self.biases[i]
            self.z_values.append(Z)
            A = self.activation(Z)

            # Clip activations to prevent overflow
            A = np.clip(A, -1e10, 1e10)
            self.activations.append(A)

        # Output layer
        Z = A @ self.weights[-1] + self.biases[-1]
        self.z_values.append(Z)
        A_output = self.activation(Z)

        # Clip output activations
        A_output = np.clip(A_output, -1e10, 1e10)
        self.activations.append(A_output)
        return A_output
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        y = y.reshape(-1, 1)

        output = self.activations[-1]
        delta = output - y  # Gradient of the loss

        for i in reversed(range(len(self.weights))):
            dw = (self.activations[i].T @ delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            dw += (self.lambda_reg / m) * self.weights[i]  # Regularization term

            # Update weights and biases
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

            # Update delta for next layer
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.activation_derivative(self.z_values[i - 1])
    
    def train(self, X, y, learning_rate=0.01, epochs=1000, batch_size=None, shuffle=True, lambda_reg=0.0):
        self.lambda_reg = lambda_reg  # Update regularization parameter
        self.initialize_weights_and_biases()  # Reset weights and biases for new training
        mse_history = []
        m = X.shape[0]
        if batch_size is None:
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
            mse_history.append(mse)

            if epoch % (epochs // 5) == 0:
                print(f'Epoch {epoch}, MSE ({self.activation_func}): {mse}')
        
        return mse_history

    def predict(self, X):
        return self.forward(X)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        predicted_classes = (predictions > 0.5).astype(int)
        return np.mean(predicted_classes.flatten() == y.flatten())


def plot_2D_parameter_lambda_eta(lambdas, etas, value, title=None, x_log=False, y_log=False, savefig=False, filename='', Reverse_cmap=False, annot=True, only_less_than=None):
    """
    Plots a 2D heatmap with lambda and eta as inputs.

    Arguments:
        lambdas: array-like
            Values for the regularization parameter (lambda) on the x-axis.
        etas: array-like
            Values for the learning rate (eta) on the y-axis.
        value: 2D array-like
            Values for each combination of lambda and eta.
        title: str
            Title of the plot.
        x_log: bool
            If True, x-axis is logarithmic.
        y_log: bool
            If True, y-axis is logarithmic.
        savefig: bool
            If True, saves the plot as a PDF. Don't include file extension
        filename: str
            Name for the saved file if savefig is True.
        Reverse_cmap: bool
            If True, reverses the color map. Useful for comparison between MSE (low = good) and accuracy (high = good).
    """
    cmap = 'plasma'
    if Reverse_cmap == True:
        cmap = 'plasma_r'
    fig, ax = plt.subplots(figsize = (12, 7))
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

    
    if only_less_than is not None:
        annot_data = np.where(value < only_less_than, np.round(value, 3).astype(str), "")
        sns.heatmap(
            data=value,
            ax=ax,
            cmap=cmap,
            annot=annot_data,  # Use the formatted conditional annotations
            fmt="",  # Set fmt to an empty string as annotations are pre-formatted
            annot_kws={"size": 6.5},
        )
        x_labels = [f"{float(label.get_text()):.1e}" for label in ax.get_xticklabels()]
        y_labels = [f"{float(label.get_text()):.1e}" for label in ax.get_yticklabels()]

        # Apply the formatted labels with rotation and font adjustments
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(y_labels, rotation=0, fontsize=8)
    else:
        sns.heatmap(data = value, ax = ax, cmap = cmap, annot=annot, fmt=".3f",  xticklabels=t_x, yticklabels=t_y)
    if title is not None:
        plt.title(title)
    plt.xlim(0, len(lambdas))
    plt.ylim(0, len(etas))
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\eta$')
    plt.tight_layout()
    if savefig:
        plt.savefig(f'Figures/{filename}.pdf')