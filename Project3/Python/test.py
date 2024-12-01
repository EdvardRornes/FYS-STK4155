from utils import * 
import copy
import numpy as np
import keras as ker
import random
# from keras.models import Sequential 
# from keras.layers import Dense, SimpleRNN # type: ignore

class Activation:

    def __init__(self, acitvation_name:str, is_derivative=False):
        """
        Creates a callable activation function corresponding to the string 'acitvation_name' given.
        """
        self.activation_functions =            [Activation.Lrelu, Activation.relu, 
                                                Activation.sigmoid, Activation.tanh]
        self.activation_functions_derivative = [Activation.Lrelu_derivative, Activation.relu_derivative, 
                                                Activation.sigmoid_derivative, Activation.tanh_derivative]
        self.activation_functions_name = ["LRELU", "RELU", "SIGMOID", "TANH"]

        self.acitvation_name = acitvation_name
        try:
            index = self.activation_functions_name.index(acitvation_name.upper())
            self.activation_func, self.activation_func_derivative  = self.activation_functions[index], self.activation_functions_derivative[index]
        except:
            raise TypeError(f"Did not recognize '{acitvation_name}' as an activation function.")

        self._call = self.activation_func
        if is_derivative:   # Then call-method will return derivative instead
            self._call = self.activation_func_derivative

    def __call__(self, z):
        return self._call(z)

    def __str__(self):
        return self.acitvation_name
    
    def derivative(self) -> Activation:
        return Activation(self.acitvation_name, True)
    
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
        return np.tanh(z)
    
    @staticmethod
    def tanh_derivative(z):
        return 1 / np.cosh(z)**2
    
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
          
class NeuralNetwork:
    
    data = {}

    def __init__(self, activation_func:str, activation_func_output:str, scaler:str, 
                 test_percentage:float, random_state:int):
        self.activation_func = Activation(activation_func)
        self.activation_func_derivative = self.activation_func.derivative()
        self.test_percentage = test_percentage
        self.random_state = random_state

        if activation_func_output is None:
            self.activation_func_out = Activation(activation_func)
            self.activation_func__out_derivative = self.activation_func_out.derivative()
        
        else:
            self.activation_func_out = Activation(activation_func_output)
            self.activation_func__out_derivative = self.activation_func_out.derivative()
        
        self._scaler = Scalers(scaler)

    @staticmethod
    def get_var_name(var):
        for name, value in globals().items():
            if value is var:
                return name
            
    def create_data(self):
        self.data = copy.deepcopy(vars(self))
        for key in self.data.keys():
            self.data[key] = str(self.data[key])

        # Property variables need to be handles seperatly:
        del self.data["_scaler"]
        self.data["scaler"] = str(self._scaler)

    def store_train_test_from_data(self, x:np.ndarray, y:np.ndarray) -> None:
        """
        Converts data into a training and a testing set. Applies the scaling assicated with the instance of the class.
        """

        if self.random_state is None:
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.test_percentage)
        else: 
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.test_percentage, random_state=self.random_state)
        
        X_train = np.reshape(X_train, (len(X_train), 1))
        X_test = np.reshape(X_test, (len(X_test), 1))
        self.y_train = y_train; self.y_test = y_test
        self.X_train_scaled, self.X_test_scaled = self._scaler(X_train, X_test)

    def prepare_sequences_RNN(self, X:np.ndarray, y:np.ndarray, step_length:int):
        """
        Converts data into sequences for RNN training.
        
        Parameters:
        X:              1D array of scaled data.
        y:              Output data.
        step_length:    Length of each sequence.
        
        Returns:
        tuple:          Sequences (3D array) and corresponding labels (1D array).
        """
        sequences = []
        labels = []

        for i in range(len(X) - step_length + 1):
            seq = X[i:i + step_length]
            sequences.append(seq)
            label = y[i + step_length - 1]
            labels.append(label)

        X_seq, y_seq = np.array(sequences).reshape(-1, step_length, 1), np.array(labels)
        return X_seq, y_seq

    @property
    def scaler(self):
        return self._scaler
    
    @scaler.setter
    def scaler(self, new_scaler:str):
        self._scaler = Scalers(new_scaler)
        self.data["scaler"] = str(self._scaler)



class FFNN(NeuralNetwork):
    def __init__(self, input_size:int, hidden_layers:list, output_size:int, optimizer:Optimizer, activation:str, 
                 activation_out=None, lambda_reg=0.0, alpha=0.1, loss_function='mse', scaler="standard"):
        """
        Implements the Feedforward Neural Network (FFNN).
        
        Positional Arguments
        * input_size:           Number of input features.
        * hidden_layers:        List of integers representing the size of each hidden layer.
        * output_size:          Number of output neurons.
        * optimizer:            Type of optimizer used for weights and biases (PlaneGradient, AdaGrad, RMSprop or Adam)
        * activation:           Activation function to use ('relu', 'sigmoid', 'lrelu').

        Keyword Arguments
        * activation_out (str): Activation function for the output layer
        * lambda_reg (float):   L2 regularization parameter
        * loss_function (str):  Loss function to use ('mse' or 'bce').
        * alpha (float):        Leaky ReLU parameter (only for 'lrelu').
        * scaler (str):         Type of scaler.
        """

        # Initializes acitvation functions
        super().__init__(activation, activation_out, scaler)
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        self.activation_func_name = activation
        self.loss_function = loss_function  # Added loss function parameter
        self.optimizer = optimizer

        self.lambda_reg = lambda_reg
        self.alpha = alpha

        self.create_data()

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
    def __init__(self, input_size: int, hidden_layers: list, output_size: int, optimizer: Optimizer, activation:str,
                 activation_out=None, lambda_reg=0.0, alpha=0.1, loss_function='mse', scaler="standard"):
        """
        Implements the Recurrent Neural Network (RNN).
        
        Positional Arguments
        * input_size:           Number of input features.
        * hidden_layers:        List of integers representing the size of each hidden layer.
        * output_size:          Number of output neurons.
        * optimizer:            Type of optimizer used for weights and biases (PlaneGradient, AdaGrad, RMSprop or Adam)
        * activation:           Activation function to use ('relu', 'sigmoid', 'lrelu').

        Keyword Arguments
        * activation_out (str): Activation function for the output layer
        * lambda_reg (float):   L2 regularization parameter
        * loss_function (str):  Loss function to use ('mse' or 'bce').
        * alpha (float):        Leaky ReLU parameter (only for 'lrelu').
        * scaler (str):         Type of scaler
        """

        # Initializes acitvation functions
        super().__init__(activation, activation_out, scaler)
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        self.activation_func_name = activation
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.hidden_size = hidden_layers[-1]

        self.create_data()

        # Initialize weights, biases, and optimizers
        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        """
        Initializes weights and biases. 
        """
        self.weights = []
        self.recurrent_weights = []
        self.biases = []
        self.weight_optimizers = []
        self.bias_optimizers = []
        self.recurrent_weight_optimizers = []

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
                    z = h_t[layer_idx - 1] @ self.weights[layer_idx] + h_t[layer_idx] @ self.recurrent_weights[layer_idx] + self.biases[layer_idx]
                
                # Apply activation function
                h = self.activation(z)
                
                # Store pre-activation and activation
                self.z_s[layer_idx].append(z)
                self.hidden_states[layer_idx].append(h)
                
                # Update the hidden state
                h_t[layer_idx] = h

            # Compute output from the last hidden layer
            o_t = h_t[-1] @ self.output_weights + self.output_biases
            o_t = self.output_activation(o_t)  # Apply output activation function
            self.outputs.append(o_t)

        # Convert lists to arrays for easier handling
        # hidden_states[layer][time] -> (batch_size, hidden_size)
        self.hidden_states = [np.stack(layer, axis=1) for layer in self.hidden_states]  # List of arrays
        # Outputs: list of (batch_size, output_size) -> (batch_size, sequence_length, output_size)
        self.outputs = np.stack(self.outputs, axis=1)  # Shape: (batch_size, sequence_length, output_size)

        return self.outputs, self.hidden_states

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Computes the loss using the true (y_true) output data and the predicted (y_pred) output data.
        """
        if self.loss_function == 'mse':
            return np.mean((y_pred - y_true) ** 2)
        elif self.loss_function == 'bce':
            return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        else:
            raise ValueError("Invalid loss function specified. Use 'mse' or 'bce'.")

    def predict(self, X: np.ndarray):
        """
        Predicts on the data X.
        """
        outputs, _ = self.forward(X)
        return outputs

    def accuracy(self, X: np.ndarray, y: np.ndarray):
        """
        Computes and returns the accuracy score.
        """
        predictions = self.predict(X)
        if self.loss_function == 'bce':
            predicted_classes = (predictions > 0.5).astype(int)
            return np.mean(predicted_classes.flatten() == y.flatten())
        else:
            return np.mean(np.round(predictions).flatten() == y.flatten())

    def backward(self, X: np.ndarray, y: np.ndarray, epoch_index: int, batch_index: int):
        """
        Performs backpropagation through time and updates the weights and biases.
        
        Arguments:
        * X: Input data of shape (batch_size, sequence_length, input_size).
        * y: Labels of shape (batch_size, sequence_length, output_size).
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

        counter = 0
        for t in reversed(range(sequence_length)):
            h_t = self.hidden_states[-1][:, t, :]  # Shape: (batch_size, hidden_size)
            z_t = self.z_s[-1][t]                  # Shape: (batch_size, hidden_size)
            x_t = X[:, t, :]                       # Shape: (batch_size, input_size)

            o_t = self.outputs[:, t, :]            # Shape: (batch_size, output_size)
            y_true = y[counter]                    # Shape: (batch_size, output_size)
            counter += 1

            # Compute gradient of loss w.r.t o_t
            dL_do_t = (o_t - y_true) * self.output_activation(o_t)

            # Gradients for output layer
            dW_hy += h_t.T @ dL_do_t
            db_y += np.sum(dL_do_t, axis=0, keepdims=True)
            dh = dL_do_t @ self.output_weights.T + dh_next

            # Backprop through activation function
            dh_raw = dh * self.activation_derivative(z_t)

            # Gradients for weights and biases
            dW_xh += x_t.T @ dh_raw if x_t.shape[0] == dh_raw.shape[0] else np.dot(x_t.T, dh_raw)
            if t > 0:
                h_prev = self.hidden_states[-1][:, t - 1, :]
            else:
                h_prev = np.zeros_like(h_t)
            dW_hh += h_prev.T @ dh_raw if h_prev.shape[0] == dh_raw.shape[0] else np.dot(h_prev.T, dh_raw)
            db_h += np.sum(dh_raw, axis=0, keepdims=True)

            # Compute gradient for next time step
            dh_next = dh_raw @ self.recurrent_weights[0].T

        # Update weights and biases using optimizers
        self.weights[0] = self.weight_optimizers[0](self.weights[0], dW_xh, epoch_index, batch_index)
        self.recurrent_weights[0] = self.recurrent_weight_optimizers[0](self.recurrent_weights[0], dW_hh, epoch_index, batch_index)
        self.biases[0] = self.bias_optimizers[0](self.biases[0], db_h, epoch_index, batch_index)

        self.output_weights = self.output_optimizer(self.output_weights, dW_hy, epoch_index, batch_index)
        self.output_biases = self.output_bias_optimizer(self.output_biases, db_y, epoch_index, batch_index)

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
        
    @property
    def learning_rate(self):
        return self.weight_optimizer.learning_rate
    
    @learning_rate.setter
    def learning_rate(self, new_learning_rate):
        self.optimizer.learning_rate = new_learning_rate
        for i in range(len(self.weight_optimizers)):
            self.weight_optimizers[i].learning_rate = new_learning_rate
            self.bias_optimizers[i].learning_rate = new_learning_rate

import numpy as np
import tensorflow.keras as ker
from keras.callbacks import ModelCheckpoint

class KerasRNN(NeuralNetwork):
    def __init__(self, hidden_layers:int, dim_output:int, dim_input:int, activation_func:str, 
                 activation_func_out=None, loss_function="binary_crossentropy", optimizer="adam", scaler="standard", 
                 test_percentage=0.25, random_state=None, labels=None, gw_class_early_boost=1):
        """
        Initializes an RNN model for multi-class classification.
        """
        super().__init__(activation_func, activation_func_out, scaler, test_percentage, random_state)

        self.hidden_layers        = hidden_layers
        self.dim_output           = dim_output  # Number of output classes
        self.dim_input            = dim_input
        self.loss_function        = loss_function
        self.optimizer            = optimizer
        self.gw_class_early_boost = gw_class_early_boost

        # Store labels to compute class weights
        self.labels = labels

        self.create_data()
        self.model = ker.models.Sequential()
        self.model.add(ker.layers.SimpleRNN(self.hidden_layers, input_shape=self.dim_input, 
                        activation=activation_func))
        # Output layer with softmax activation for multi-class classification
        self.model.add(ker.layers.Dense(units=self.dim_output, activation=activation_func))
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])
    
    def compute_class_weights(self, epoch, total_epochs):
        """
        Compute class weights dynamically based on the label distribution.
        """
        initial_boost = self.gw_class_early_boost
        scale = initial_boost - (initial_boost-1)*epoch/total_epochs
        gw_class_weight = (len(self.labels)-np.sum(self.labels))/np.sum(self.labels)*scale
        print(gw_class_weight, epoch)
        return {0:1,1:gw_class_weight}
    

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, step_length: int, verbose=1):
        """
        Train the RNN model using the dynamically computed class weights.
        """
        # Store train/test splits
        self.store_train_test_from_data(x, y)

        # Prepare sequences for RNN input
        X_train_seq, y_train_seq = self.prepare_sequences_RNN(self.X_train_scaled, self.y_train, step_length)
        X_test_seq, y_test_seq = self.prepare_sequences_RNN(self.X_test_scaled, self.y_test, step_length)

        # Compute class weights dynamically based on training labels
        for epoch in range(epochs):
            class_weights = self.compute_class_weights(epoch, epochs)

            # Create a folder for saving model weights (if it doesn't exist)
            checkpoint_dir = 'model_checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)  # creates the folder if it doesn't exist

            # Create a checkpoint callback with a file path inside the specified folder
            if epoch != 0:
                checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, f"model_epoch_{epoch}.keras"), monitor='val_loss', save_best_only=True, mode='min', verbose=1)
                # checkpoint = ModelCheckpoint(
                #     os.path.join(checkpoint_dir, f"model_epoch_{epoch}.weights.h5"), 
                #     save_weights_only=True, 
                #     save_freq='epoch'
                # )

            # Then pass this callback to the fit function
            self.model.fit(
                X_train_seq, y_train_seq,
                epochs=1,
                batch_size=batch_size,
                verbose=verbose,
                class_weight=class_weights,
                callbacks=[checkpoint] if epoch!=0 else None  # Save model after each epoch
            )


        # # Predictions and evaluation
        # train_predict = self.model.predict(X_train_seq)
        # test_predict = self.model.predict(X_test_seq)

        # print(f"Training Accuracy: {self.model.evaluate(X_train_seq, y_train_seq)}")
        # print(f"Test Accuracy: {self.model.evaluate(X_test_seq, y_test_seq)}")



class GWSignalGenerator:
    def __init__(self, signal_length, noise_level=0.1):
        """
        Initialize the GWSignalGenerator with a signal length and noise level.
        """
        self.signal_length = signal_length
        self.noise_level = noise_level
        self.labels = np.zeros(signal_length, dtype=int)  # Initialize labels to 0 (background noise)
        self.regions = []  # Store regions for visualization or further analysis

    def add_gw_event(self, y, start, end, amplitude_factor=1, spike_factor=0.8, spin_start=10, spin_end=100, scale=1):
        """
        Adds a simulated gravitational wave event to the signal and updates labels for its phases.
        Includes a spin factor that increases during the inspiral phase.
        """
        event_sign = np.random.choice([-1, 1])  # Random polarity for the GW event

        amplitude_factor=amplitude_factor*scale
        spike_factor=spike_factor*scale

        # Inspiral phase
        inspiral_end = int(start + 0.7 * (end - start))  # Define inspiral region as 70% of event duration
        time_inspiral = np.linspace(0, 1, inspiral_end - start)  # Normalized time for the inspiral
        amplitude_increase = np.linspace(0, amplitude_factor, inspiral_end - start)
        
        # Spin factor: linearly increasing frequency
        spin_frequency = np.linspace(spin_start, spin_end, inspiral_end - start)  # Spin frequency in Hz
        spin_factor = np.sin(2 * np.pi * spin_frequency * time_inspiral)
        
        y[start:inspiral_end] += event_sign * amplitude_increase * spin_factor
        # self.labels[start:inspiral_end] = 1  # Set label to 1 for inspiral

        # Merger phase
        merge_start = inspiral_end
        merge_end = merge_start + int(0.1 * (end - start))  # Define merger as 10% of event duration
        y[merge_start:merge_end] += event_sign * spike_factor * np.exp(-np.linspace(3, 0, merge_end - merge_start))
        # self.labels[merge_start:merge_end] = 2  # Set label to 2 for merger

        # Ringdown phase
        dropoff_start = merge_end
        dropoff_end = dropoff_start + int(0.2 * (end - start))  # Define ringdown as 20% of event duration
        dropoff_curve = spike_factor * np.exp(-np.linspace(0, 15, dropoff_end - dropoff_start))
        y[dropoff_start:dropoff_end] += event_sign * dropoff_curve
        # self.labels[dropoff_start:dropoff_end] = 3  # Set label to 3 for ringdown

        self.labels[start:(2*dropoff_start+dropoff_end)//3] = 1

        # Store region details for visualization or debugging
        self.regions.append((start, end, inspiral_end, merge_start, merge_end, dropoff_start, dropoff_end))


    def generate_random_events(self, num_events, event_length_min, event_length_max, scale=1):
        """
        Generate random gravitational wave events with no overlaps.
        """
        events = []
        used_intervals = []

        for _ in range(num_events):
            while True:
                # Randomly determine start and length of event
                event_length = random.randint(event_length_min, event_length_max)
                event_start = random.randint(0, self.signal_length - event_length)
                event_end = event_start + event_length

                # Ensure no overlap
                if not any(s <= event_start <= e or s <= event_end <= e for s, e in used_intervals):
                    used_intervals.append((event_start, event_end))
                    break  # Valid event, exit loop

            # Randomize event properties
            amplitude_factor = random.uniform(0, 0.5)
            spike_factor = random.uniform(0.2, 1.5)
            
            # Randomize spin start and end frequencies
            spin_start = random.uniform(5, 30)  # Starting spin frequency (in Hz)
            spin_end = random.uniform(50, 500)  # Ending spin frequency (in Hz)

            events.append((event_start, event_end, amplitude_factor * scale, spike_factor * scale, spin_start, spin_end))

        return events

    def apply_events(self, y, events):
        """
        Apply generated events to the input signal.
        """
        for start, end, amplitude, spike, spin_start, spin_end in events:
            self.add_gw_event(y, start, end, amplitude_factor=amplitude, spike_factor=spike, spin_start=spin_start, spin_end=spin_end)




# Parameters
time_steps = 10000
x = np.linspace(0, 50, time_steps)
noise = 0.02

# Base signal: sine wave + noise
y = np.zeros_like(x)#1e-19*(0.5 * np.sin(100 * x) - 0.5 * np.cos(60 * x)*np.sin(-5*x) + 0.3*np.cos(30*x) + 0.05*np.sin(10000*x)) #+ noise * np.random.randn(time_steps)
y_noGW = y.copy()

# Initialize generator and create events
generator = GWSignalGenerator(signal_length=time_steps, noise_level=noise)
# events = generator.generate_random_events(1, time_steps//3, time_steps//2, scale=1e-19)
# generator.apply_events(y, events)
GWSignalGenerator.add_gw_event(generator, y, time_steps//2, 5*time_steps//6, spin_start=3, spin_end=15, spike_factor=2, scale=1)

# Plot the signal
plt.figure(figsize=(15, 6))
plt.plot(x, y_noGW, label="No GW Signal", lw=0.4, color="gray", alpha=0.7)
plt.plot(x, y, label="Signal (with GW events)", lw=0.6, color="blue")

# Highlight regions
for i, (start, end, inspiral_end, merge_start, merge_end, dropoff_start, dropoff_end) in enumerate(generator.regions):
    plt.axvspan(x[start], x[merge_start], color="lightblue", alpha=0.3, label="Inspiral" if i == 0 else "")
    plt.axvspan(x[merge_start], x[merge_end], color="orange", alpha=0.3, label="Merger" if i == 0 else "")
    plt.axvspan(x[dropoff_start], x[(2*dropoff_start+dropoff_end)//3], color="lightgreen", alpha=0.3, label="Ringdown" if i == 0 else "")


# Add labels and legend
plt.xlim(0, np.max(x))
plt.title("Simulated Gravitational Wave Signal with Highlighted Epochs")
plt.xlabel("Time (ms)")
plt.ylabel("Strain")
plt.legend()
# plt.show()

# Perform FFT on the signal with GW events
y_fft = np.fft.fft(y)
frequencies = np.fft.fftfreq(time_steps, d=(x[1] - x[0]))  # d is the sampling interval

# Compute the amplitude spectrum (magnitude of FFT) and focus on positive frequencies
amplitude_spectrum = np.abs(y_fft)[:time_steps // 2]
positive_frequencies = frequencies[:time_steps // 2]

# Plot the FFT
plt.figure(figsize=(15, 6))
plt.plot(positive_frequencies, amplitude_spectrum, color="blue", lw=0.7)
plt.xlim(1e-2, 5e2)
plt.ylim(1e-19, 1e-13)
plt.xscale('log')
plt.yscale('log')
plt.title("FFT of Signal with Gravitational Wave Events")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
# plt.show()


from sklearn.metrics import confusion_matrix


########## I am not doing what I think I am doing, fix later! ##########

y_test = np.zeros_like(x)#1e-19*(0.3 * np.sin(40 * x) - 0.8 * np.cos(70 * x)*np.sin(-12*x) + 0.2*np.cos(70*x) + 0.05*np.sin(10000*x))
y_test_noGW = y_test.copy()

test_generator = GWSignalGenerator(signal_length=time_steps, noise_level=noise)
# test_events = test_generator.generate_random_events(1, time_steps//3, time_steps//2, scale=1e-19)
# test_generator.apply_events(y_test, test_events)
GWSignalGenerator.add_gw_event(test_generator, y_test, time_steps//3, 2*time_steps//3, spin_start=4, spin_end=7, spike_factor=1.5, scale=1)


y_test_reshaped = y_test.reshape((time_steps, 1, 1))  # RNN expects 3D input (samples, time steps, features)
labels = generator.labels
labels_reshaped = labels.reshape((time_steps, 1))  # Labels are 2D (samples, output)

# Create an instance of the KerasRNN model, passing the labels for class weight computation
test = KerasRNN(hidden_layers=5, dim_output=1, dim_input=(1, 1), activation_func="sigmoid", labels=labels_reshaped, gw_class_early_boost=1.3)

# Train the model with the reshaped data and class weights calculated inside KerasRNN
test.train(x=y_test_reshaped, y=labels, epochs=25, batch_size=250, step_length=2000)

# Get predictions
predictions = test.model.predict(y_test_reshaped)

# Convert softmax probabilities to class labels (argmax for multi-class)
predicted_labels = (predictions > 0.5).astype(int)  # 0.5 threshold for GW detection

# Plot the signal with highlighted predicted GW events
plt.figure(figsize=(15, 6))
plt.plot(x, y_test_noGW, label="No GW test Signal", lw=0.4, color="gray", alpha=0.7)
plt.plot(x, y_test, label="Test Signal (with GW events)", lw=0.6, color="blue")

# Highlight predicted GW regions with shaded areas
predicted_gw_indices = np.where(predicted_labels == 1)[0]  # Find indices where prediction is GW (1)

# for i, prediction in zip(range(len(predicted_labels)), predicted_labels):
#     print(f"prediction {i} = {prediction}")

# If there are no predicted GW events, print a message and skip the shading
if len(predicted_gw_indices) == 0:
    print("No gravitational wave events predicted.")
else:
    # Define a threshold for grouping consecutive predictions into regions
    threshold = 2  # Number of consecutive time steps to group as a single GW event

    # Loop through the predicted GW indices and group consecutive ones
    grouped_gw_indices = []
    current_group = [predicted_gw_indices[0]]

    for i in range(1, len(predicted_gw_indices)):
        if predicted_gw_indices[i] - predicted_gw_indices[i - 1] <= threshold:
            current_group.append(predicted_gw_indices[i])
        else:
            grouped_gw_indices.append(current_group)
            current_group = [predicted_gw_indices[i]]

    grouped_gw_indices.append(current_group)  # Append last group

    # Shade the regions
    for i, group in zip(range(len(grouped_gw_indices)), grouped_gw_indices):
        plt.axvspan(x[group[0]], x[group[-1]], color="red", alpha=0.3, label="Predicted event" if i==0 else "")
# Highlight regions
for i, (start, end, inspiral_end, merge_start, merge_end, dropoff_start, dropoff_end) in enumerate(test_generator.regions):
    plt.axvspan(x[start], x[merge_start], color="lightblue", alpha=0.3, label="Inspiral" if i == 0 else "")
    plt.axvspan(x[merge_start], x[merge_end], color="orange", alpha=0.3, label="Merger" if i == 0 else "")
    plt.axvspan(x[dropoff_start], x[(2*dropoff_start+dropoff_end)//3], color="lightgreen", alpha=0.3, label="Ringdown" if i == 0 else "")

# Add labels and legend
plt.xlim(0, np.max(x))
plt.title("Predicted Gravitational Wave Signal with Shaded Epochs")
plt.xlabel("Time (ms)")
plt.ylabel("Strain")
plt.legend()


# Convert softmax probabilities to class labels (argmax for multi-class)
predicted_labels = (predictions > 0.5).astype(int)
cm = confusion_matrix(labels_reshaped, predicted_labels)
print("confusion matrix =", cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=["Noise (0)", "GW (1)"], 
            yticklabels=["Noise (0)", "GW (1)"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

exit()


class Data:

    def __init__(self, neural_network:NeuralNetwork, additional_description=""):
        self.data = neural_network.data
        if not (additional_description == ""):
            self.data["additional_description"] = additional_description
    
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


# Generate synthetic time-series data
time_steps = 1000
t = np.linspace(0, 100, time_steps)  # Simulated time

# Create a sinusoidal wave with Gaussian noise
wave_signal = np.sin(0.2 * t) + np.random.normal(0, 0.1, time_steps)

# Add synthetic gravitational events as spikes
events_indices = np.random.choice(time_steps, size=10, replace=False)
wave_signal[events_indices] += np.random.normal(5, 1, len(events_indices))

# Create labels: 1 if a gravitational event (spike) occurs, otherwise 0
y = np.zeros(time_steps)
y[events_indices] = 1

adam = Adam()
rnn = RNN(1, [4, 8, 16], 1, adam, "relu")
rnn.train(t, y)
y_predict = rnn.predict(t)

# Plot synthetic signal
plt.plot(t, wave_signal)
plt.scatter(t[events_indices], wave_signal[events_indices], color='red', label='Gravitational Events')
plt.scatter(t, y_predict, color='green', label='Predicted Gravitational Events')
plt.title('Synthetic Gravitational Wave Signal with Events')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
