<<<<<<< HEAD
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
        """
        Parent class for FFNN, RNN and KeirasRNN. Contains methods shared for all child-classes. 

        Arguments:
        * activation_func:  activation function for hidden layers
        * activation_func_output:   activation function for output layer
        * scaler:                   type of scaler ('standard' or 'minmax')
        * test_percentage:          percentage of data converted to test-data when training
        * random_state:             argument in sklearn.train_test_split allowing for consistency
        """
        self.activation_func = Activation(activation_func)
        self.activation_func_derivative = self.activation_func.derivative()
        self.test_percentage = test_percentage
        self.random_state = random_state

        if activation_func_output is None:
            self.activation_func_out = Activation(activation_func)
            self.activation_func_out_derivative = self.activation_func_out.derivative()
        
        else:
            self.activation_func_out = Activation(activation_func_output)
            self.activation_func_out_derivative = self.activation_func_out.derivative()
        
        self._scaler = Scalers(scaler)

    @staticmethod
    def get_var_name(var):
        for name, value in globals().items():
            if value is var:
                return name
            
    def store_variables(self):
        self.data = copy.deepcopy(vars(self))
        for key in self.data.keys():
            self.data[key] = str(self.data[key])

        # Property variables need to be handles seperatly:
        del self.data["_scaler"]
        self.data["scaler"] = str(self._scaler)

    def store_train_test_from_data(self, X:np.ndarray, y:np.ndarray) -> None:
        """
        Converts data into a training and a testing set. Applies the scaling assicated with the instance of the class.

        Arguments:
        * X:                input time series data of shape (num_samples, input_size)
        * y:                output labels of shape (num_samples, output_size)        
        """

        if self.random_state is None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_percentage)
        else: 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_percentage, random_state=self.random_state)
        
        self.y_train = y_train; self.y_test = y_test
        
        self.X_train = X_train; self.X_test = X_test
        self.X_train_scaled, self.X_test_scaled = self._scaler(X_train, X_test)

    def prepare_sequences_RNN(self, X:np.ndarray, y:np.ndarray, step_length:int):
        """
        Converts data into sequences for RNN training.
        
        Parameters:
        * X:                scaled data 
        * y:                output data.
        * step_length:      length of each sequence.
        
        Returns:
        * X_seq, y_seq:     sequences (3D array) and corresponding labels (1D array).
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

    @staticmethod
    def _xavier_init(layer1:int, layer2:int):
        return np.random.randn(layer1, layer2) * np.sqrt(2 / layer1)

class RNN(NeuralNetwork):
    def __init__(self, input_size:int, hidden_layers:list, output_size:int, optimizer: Optimizer, activation:str,
                 activation_out=None, lambda_reg=0.0, alpha=0.1, loss_function=None, scaler="standard",
                 test_percentage=0.2, random_state=None):
        """
        Implements the Recurrent Neural Network (RNN).
        
        Positional Arguments
        * input_size:           number of input features.
        * hidden_layers:        list of integers representing the size of each hidden layer.
        * output_size:          number of output neurons.
        * optimizer:            type of optimizer used for weights and biases (PlaneGradient, AdaGrad, RMSprop or Adam)
        * activation:           activation function to use ('relu', 'sigmoid', 'lrelu').

        Keyword Arguments
        * activation_out (str): activation function for the output layer
        * lambda_reg (float):   L2 regularization parameter
        * loss_function (str):  loss function to use.
        * alpha (float):        leaky ReLU parameter (only for 'lrelu').
        * scaler (str):         type of scaler
        """

        # Initializes acitvation functions
        super().__init__(activation, activation_out, scaler, test_percentage, random_state) # See parent netwrok NeuralNetwork

        self.input_size = input_size; self.output_size = output_size
        self.weights = []
        self.biases = []
        self.activation_func_name = activation
        self.optimizer = optimizer

        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.hidden_size = hidden_layers[-1]

        # Storing variables:
        self.num_hidden_layers = len(hidden_layers)
        self.store_variables()
        
        if loss_function is None:   # Defaults to Focal loss
            self._loss_function = FocalLoss() 
            self.data["loss_function"] = str(self._loss_function)
        
        elif isinstance(loss_function, Loss):
            self._loss_function = loss_function
            self.data["loss_function"] = str(self._loss_function)
        
        else:
            raise TypeError(f"Need an instance of the Loss class as loss function.")
    


        self.layers = [input_size] + hidden_layers + [output_size]

        ### Private variables:
        self._trained = True

        # Initialize weights, biases, and optimizers
        self._initialize_weights_and_biases()

    #################### Public Methods ####################
    def train(self, X:np.ndarray, y:np.ndarray, epochs=100, batch_size=32, window_size=10) -> None:
        """
        Trains the RNN on the given dataset.
        
        Arguments:
        * X:                input time series data of shape (num_samples, input_size)
        * y:                output labels of shape (num_samples, output_size)
        
        Keyword Arguments:
        * epochs:           number of training epochs
        * batch_size:       size of each mini-batch for training
        * window_size:      size of the window for creating sequences (step length)
        """
        self._trained = True

        # Prepare sequences for training
        self.store_train_test_from_data(X, y) # applies scaling
        X_seq, y_seq = self.prepare_sequences_RNN(self.X_train_scaled, self.y_train, window_size)

        start_time = time.time()
        for epoch in range(epochs):
            for i in range(0, X_seq.shape[0], batch_size):
                X_batch = X_seq[i:i + batch_size]
                y_batch = y_seq[i:i + batch_size]

                # Forward pass
                y_pred = self._forward(X_batch)

                # Backward pass
                self._backward(X_batch, y_batch, y_pred, epoch, i)

            # Optional: add code for logging loss or accuracy per epoch
            print(f"Epoch {epoch + 1}/{epochs} completed, time elapsed: {time.time()-start_time:.1f}s", end="\r")

        print(f"Epoch {epochs}/{epochs} completed, time elapsed: {time.time()-start_time:.1f}s            ")

        X_test_seq, y_test_seq = self.prepare_sequences_RNN(self.X_test_scaled, self.y_test, window_size)

        y_test_pred = self._forward(X_test_seq)
        test_loss = self.calculate_loss(y_test_seq, y_test_pred)
        print(f"Test Loss: {test_loss}")
        print("Training completed")

    def calculate_loss(self, y_true, y_pred):
        """
        Calculate the loss value using the provided loss function.
        """
        return self._loss_function.forward(y_true, y_pred)


    def evaluate(self, X:np.ndarray, y:np.ndarray, window_size=10) -> tuple[float, float]:
        """
        Evaluate the model on a given dataset.
        
        Arguments:
        * X:        Input time series data of shape (num_samples, input_size)
        * y:        True labels of shape (num_samples, output_size)
        
        Keyword Arguments:
        * window_size: Size of the window for creating sequences (step length)
        
        Returns:
        * Loss value and accuracy score
        """
        self.store_train_test_from_data(X, y)

        # Prepare sequences for evaluation
        X_test_seq, y_test_seq = self.prepare_sequences_RNN(self.X_test_scaled, self.y_test, window_size)

        # Forward pass to get predictions
        y_pred = self._forward(X_test_seq)

        # Calculate loss
        loss = self.calculate_loss(y_test_seq, y_pred)

        # Calculate accuracy
        y_pred_classes = (y_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_test_seq, y_pred_classes)

        print(f"Evaluation - Loss: {loss}, Accuracy: {accuracy}")
        return loss, accuracy
    
    def predict(self, X:np.ndarray):
        return self._forward(X)
    
    #################### Private Methods ####################
    def _forward(self, X):
        batch_size, window_size, _ = X.shape
        num_hidden_layers = len(self.layers) - 1  # Exclude the output layer

        # Initialize hidden states for each time step
        self.hidden_states = [[np.zeros((batch_size, self.layers[i + 1])) for i in range(num_hidden_layers)] for _ in range(window_size)]

        for t in range(window_size):
            input_t = X[:, t, :]

            for l in range(num_hidden_layers):
                if l == 0:
                    prev_activation = input_t
                else:
                    prev_activation = self.hidden_states[t][l - 1]

                h_prev = np.zeros_like(self.hidden_states[t - 1][l]) if t == 0 else self.hidden_states[t - 1][l]

                h = self.activation_func(
                    prev_activation @ self.W_input[l] + h_prev @ self.W_recurrent[l] + self.biases[l]
                )
                self.hidden_states[t][l] = h

        # Output layer uses the hidden state from the last time step
        output = self.hidden_states[-1][-1] @ self.W_output + self.b_output
        output = self.activation_func_out(output)
        output = 1*(output >= 0.5)
        return output

    def _backward(self, X, y, y_pred, epoch_index, batch_index):
        batch_size, window_size, _ = X.shape
        num_hidden_layers = len(self.layers) - 2  # Exclude input and output layers

        # Initialize gradients
        total_dW_input = [np.zeros_like(w) for w in self.W_input]
        total_dW_recurrent = [np.zeros_like(w) for w in self.W_recurrent]
        total_db_hidden = [np.zeros_like(b) for b in self.biases]

        # Initialize delta_h for each layer
        delta_h = [np.zeros((batch_size, self.layers[i + 1])) for i in range(num_hidden_layers)]

        # Initialize gradients for output layer
        dW_output = np.zeros_like(self.W_output)
        db_output = np.zeros_like(self.b_output)

        # Backpropagation through time
        for t in reversed(range(window_size)):
            # Compute error at the output layer for the last time step
            if t == window_size - 1:
                error = self._loss_function.backward(y, y_pred) * self.activation_func_out_derivative(y_pred)
                # error = y_pred - y  # Shape: (batch_size, output_size)
                dW_output += self.hidden_states[t][-1].T @ error / batch_size
                db_output += np.sum(error, axis=0, keepdims=True) / batch_size

                delta_output = error @ self.W_output.T  # Shape: (batch_size, hidden_size_last_layer)
            else:
                delta_output = np.zeros_like(delta_h[-1])

            # Backpropagate through hidden layers
            for l in reversed(range(num_hidden_layers)):
                h = self.hidden_states[t][l]
                h_prev = self.hidden_states[t - 1][l] if t > 0 else np.zeros_like(h)

                # Delta from next time step
                delta_t_next = delta_h[l] @ self.W_recurrent[l].T if t < window_size - 1 else np.zeros_like(h)

                # Delta from output layer (only for last hidden layer)
                delta_from_output = delta_output if l == num_hidden_layers - 1 else 0

                # Delta from next layer at the same time step
                if l < num_hidden_layers - 1:
                    delta_from_next_layer = delta_h[l + 1] @ self.W_input[l + 1].T
                else:
                    delta_from_next_layer = np.zeros_like(h)

                # Total delta for current layer
                delta = (delta_t_next + delta_from_output + delta_from_next_layer) * self.activation_func_derivative(h)

                # Compute gradients
                if l == 0:
                    prev_activation = X[:, t, :]
                else:
                    prev_activation = self.hidden_states[t][l - 1]

                total_dW_input[l] += prev_activation.T @ delta / batch_size
                total_dW_recurrent[l] += h_prev.T @ delta / batch_size
                total_db_hidden[l] += np.sum(delta, axis=0, keepdims=True) / batch_size

                # Update delta_h for the next time step
                delta_h[l] = delta

        # Update weights and biases
        for l in range(num_hidden_layers):
            self.W_input[l] = self.optimizer_W_input[l](self.W_input[l], total_dW_input[l], epoch_index, batch_index)
            self.W_recurrent[l] = self.optimizer_W_recurrent[l](self.W_recurrent[l], total_dW_recurrent[l], epoch_index, batch_index)
            self.biases[l] = self.optimizer_biases[l](self.biases[l], total_db_hidden[l], epoch_index, batch_index)

        self.W_output = self.optimizer_W_output(self.W_output, dW_output, epoch_index, batch_index)
        self.b_output = self.optimizer_b_output(self.b_output, db_output, epoch_index, batch_index)




    def _initialize_weights_and_biases(self):
        """
        Initializes weights and biases for each layer of the RNN.
        """
        num_hidden_layers = len(self.layers) - 1  # Exclude the output layer
        self.W_input = []
        self.W_recurrent = []
        self.biases = []
        self.optimizer_W_input = []
        self.optimizer_W_recurrent = []
        self.optimizer_biases = []

        for i in range(num_hidden_layers):
            # Input weights: from input or previous hidden layer to current hidden layer
            input_dim = self.layers[i] #if i == 0 else self.layers[i + 1]
            output_dim = self.layers[i + 1] 
            self.W_input.append(NeuralNetwork._xavier_init(input_dim, output_dim))
            self.optimizer_W_input.append(self.optimizer.copy())

            # Recurrent weights: from current hidden state to next hidden state
            self.W_recurrent.append(NeuralNetwork._xavier_init(output_dim, output_dim))
            self.optimizer_W_recurrent.append(self.optimizer.copy())

            # Biases for hidden layers
            self.biases.append(np.zeros((1, output_dim)))
            self.optimizer_biases.append(self.optimizer.copy())

        # Output layer weights and biases
        self.W_output = NeuralNetwork._xavier_init(self.layers[-1], self.output_size)

        self.b_output = np.zeros((1, self.output_size))
        self.optimizer_W_output = self.optimizer.copy()
        self.optimizer_b_output = self.optimizer.copy()


class Loss:
    def forward(self, y_true, y_pred):
        """
        Computes the loss value.

        Arguments:
        - y_true: True labels.
        - y_pred: Predicted outputs.

        Returns:
        - Loss value.
        """
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, y_true, y_pred):
        """
        Computes the gradient of the loss with respect to y_pred.

        Arguments:
        - y_true: True labels.
        - y_pred: Predicted outputs.

        Returns:
        - Gradient of the loss with respect to y_pred.
        """
        raise NotImplementedError("Backward method not implemented.")


class WeightedBinaryCrossEntropyLoss(Loss):
    def __init__(self, weight_0, weight_1):
        """
        Initializes the loss function with class weights.

        Arguments:
        - class_weight: Dictionary with weights for each class {0: weight_0, 1: weight_1}.
        """
        self.weight_1 = weight_1
        self.weight_0 = weight_0

    def forward(self, y_true, y_pred):
        """
        Computes the weighted binary cross-entropy loss.
        """
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # To prevent log(0)
        loss = -np.mean(
            self.weight_1 * y_true * np.log(y_pred) +
            self.weight_0 * (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    def backward(self, y_true, y_pred):
        """
        Computes the gradient of the loss with respect to y_pred.
        """
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # To prevent division by zero
        grad = - (self.weight_1 * y_true / y_pred) + (self.weight_0 * (1 - y_true) / (1 - y_pred))
        return grad

    def __str__(self):
        return "I am WeightedBinaryCrossEntropyLoss"

class FocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Initializes the focal loss function.

        Arguments:
        - alpha: Weighting factor for the positive class.
        - gamma: Focusing parameter.
        """
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_true, y_pred):
        """
        Computes the focal loss.
        """
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        loss = -np.mean(alpha_t * (1 - pt) ** self.gamma * np.log(pt))
        return loss

    def backward(self, y_true, y_pred):
        """
        Computes the gradient of the loss with respect to y_pred.
        """
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        grad = -alpha_t * (1 - pt) ** (self.gamma - 1) * (
            self.gamma * pt * np.log(pt) + (1 - pt)
        ) / pt
        grad *= y_pred - y_true
        return grad
    
    def __str__(self):

        return "I am Focal."
=======

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import tensorflow.keras as ker # type: ignore
import pickle
import tensorflow as tf
import copy

from sklearn.model_selection import KFold, train_test_split
from keras.models import Sequential, load_model # type: ignore
from keras.optimizers import Adam, SGD, RMSprop # type: ignore
from keras.layers import SimpleRNN, Dense, Input # type: ignore
from keras.callbacks import ModelCheckpoint # type: ignore
from keras.regularizers import l2 # type: ignore
from utils import latex_fonts
from tensorflow.keras import mixed_precision # type: ignore
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

tf.config.threading.set_intra_op_parallelism_threads(12)  # Adjust number of threads
tf.config.threading.set_inter_op_parallelism_threads(12)

latex_fonts()
savefigs = True

class KerasRNN:
    def __init__(self, hidden_layers: list, dim_output: int, dim_input: int,  
                 loss_function="binary_crossentropy", optimizer="adam", labels=None, 
                 gw_class_early_boost=1, learning_rate=1e-2, l2_regularization=0.0, activation_func='tanh'):
        """
        Initializes an RNN model for multi-class classification.
        """
        self.hidden_layers = hidden_layers
        self.dim_output = dim_output  # Number of output classes
        self.dim_input = dim_input
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.gw_class_early_boost = gw_class_early_boost
        self.labels = labels
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.activation_func = activation_func

        # Initialize the RNN model
        self.model = self.create_model()  # Call create_model during initialization to set up the model

    def create_model(self):
        """
        Creates and returns a fresh RNN model with the specified configurations.
        """
        model = ker.models.Sequential()

        # Add the input layer (input shape)
        model.add(Input(shape=(self.dim_input, 1)))  # Specify input shape here

        # Add RNN layers with optional L2 regularization
        for idx, units in enumerate(self.hidden_layers):
            model.add(SimpleRNN(units, activation=self.activation_func,
                                return_sequences=True if units != self.hidden_layers[-1] else False, 
                                kernel_regularizer=l2(self.l2_regularization)))

        # Output layer
        model.add(Dense(units=self.dim_output, activation="sigmoid",
                        kernel_regularizer=l2(self.l2_regularization)))

        # Compile the model
        self.compile_model(model)

        return model

    def compile_model(self, model):
        """
        Compiles the model with the selected optimizer and learning rate.
        """
        optimizers = {
            "adam": Adam(learning_rate=self.learning_rate),
            "sgd": SGD(learning_rate=self.learning_rate),
            "rmsprop": RMSprop(learning_rate=self.learning_rate)
        }

        if self.optimizer not in optimizers:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}. Choose from {list(optimizers.keys())}.")

        model.compile(
            loss=self.loss_function, 
            optimizer=optimizers[self.optimizer], 
            metrics=['accuracy']
        )

    def prepare_sequences_RNN(self, X: np.ndarray, y: np.ndarray, step_length: int):
        """
        Converts data into sequences for RNN training.
        """
        n_samples = len(X) - step_length + 1
        X_seq = np.array([X[i:i + step_length] for i in range(n_samples)]).reshape(-1, step_length, 1)
        y_seq = y[step_length-1:]
        return X_seq, y_seq

    
    def compute_class_weights(self, epoch: int, total_epochs: int):
        """
        Compute class weights dynamically based on the label distribution.
        """
        if self.labels is not None:
            initial_boost = self.gw_class_early_boost
            scale = initial_boost - (initial_boost - 1) * epoch / total_epochs
            gw_class_weight = (len(self.labels) - np.sum(self.labels)) / np.sum(self.labels) * scale
            return {0: 1, 1: gw_class_weight}
        else:
            print("Labels required for training, exiting")
            exit()


    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int, step_length: int, verbose=1, verbose1=1):
        """
        Train the RNN model using dynamically computed class weights, keeping the best model in memory.
        The model is carried over through epochs, but is reinitialized between parameter runs.
        """
        # Split training data into a validation set
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Prepare sequences for RNN input
        X_train_seq, y_train_seq = self.prepare_sequences_RNN(X_train, y_train, step_length)
        X_val_seq, y_val_seq = self.prepare_sequences_RNN(X_val, y_val, step_length)

        # Reinitialize model (this ensures no previous weights are carried over between parameter runs)
        self.model = self.create_model()  # Recreate a fresh model

        # Initialize variables to track the best model and validation loss
        best_val_loss = float('inf')
        best_weights = None  # Keep track of the best model's weights, not the entire model

        # Compute class weights dynamically based on training labels
        for epoch in range(epochs):
            class_weights = self.compute_class_weights(epoch, epochs)

            # Fit the model for one epoch and save the history
            history = self.model.fit(
                X_train_seq, y_train_seq,
                epochs=1,
                batch_size=batch_size,
                verbose=verbose,
                class_weight=class_weights,
                validation_data=(X_val_seq, y_val_seq)
            )

            # Extract val_loss for the current epoch
            current_val_loss = history.history['val_loss'][0]

            # Check if this is the best val_loss so far
            if current_val_loss < best_val_loss:
                best_weights = self.model.get_weights()  # Save the best weights
                if verbose1==1:
                    print(f"Epoch {epoch + 1} - val_loss improved from {best_val_loss:.3f} to {current_val_loss:.3f}. Best model updated.")
                best_val_loss = current_val_loss
            else:
                if verbose1==1:
                    print(f"Epoch {epoch + 1} - val_loss did not improve ({current_val_loss:.3f} >= {best_val_loss:.3f}).")

        # After all epochs, restore the best model weights (so model doesn't carry over worse performance)
        if best_weights is not None:
            self.model.set_weights(best_weights)  # Set the model's weights to the best found during training


    def predict(self, X_test, y_test, step_length, verbose=1):
        """
        Generate predictions for test data.
        """
        X_test_seq, y_test_seq = self.prepare_sequences_RNN(X_test, y_test, step_length)
        prediction = self.model.predict(X_test_seq, verbose=verbose)
        loss, accuracy = self.model.evaluate(X_test_seq, y_test_seq, verbose=verbose)
        return prediction, loss, accuracy




class GWSignalGenerator:
    def __init__(self, signal_length: int):
        """
        Initialize the GWSignalGenerator with a signal length.
        """
        self.signal_length = signal_length
        self.labels = np.zeros(signal_length, dtype=int)  # Initialize labels to 0 (background noise)
        self.regions = []  # Store regions for visualization or further analysis

    def add_gw_event(self, y, start, end, amplitude_factor=0.2, spike_factor=0.5, spin_start=1, spin_end=20, scale=1):
        """
        Adds a simulated gravitational wave event to the signal and updates labels for its phases.
        Includes a spin factor that increases during the inspiral phase.

        Parameters:
        y:                Signal to append GW event to.
        start:            Start index for GW event.
        end:              End index for GW event.
        amplitude_factor: Peak of the oscillating signal in the insipral phase.
        spike_factor:     Peak of the signal in the merge phase.
        spin_start:       Oscillation frequency of the start of the inspiral phase.
        spin_end:         Oscillation frequency of the end of the inspiral phase.
        scale:            Scale the amplitude of the entire event.

        returns:
        Various parameters to be used by apply_events function
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

        # We cut off 2/3rds of the ringdown event due to the harsh exponential supression.
        # It is not expected that the NN will detect anything past this and may cause confusion for the program.
        self.labels[start:(2*dropoff_start+dropoff_end)//3] = 1

        # Store region details for visualization or debugging
        self.regions.append((start, end, inspiral_end, merge_start, merge_end, dropoff_start, dropoff_end))


    def generate_random_events(self, num_events: int, event_length_range: tuple, scale=1, 
                               amplitude_factor_range = (0, 0.5), spike_factor_range = (0.2, 1.5),
                               spin_start_range = (1, 5), spin_end_range = (5, 20)):
        """
        Generate random gravitational wave events with no overlaps.
        """
        events = []
        used_intervals = []

        for _ in range(num_events):
            while True:
                # Randomly determine start and length of event
                event_length = random.randint(*event_length_range)
                event_start = random.randint(0, self.signal_length - event_length)
                event_end = event_start + event_length

                # Ensure no overlap
                if not any(s <= event_start <= e or s <= event_end <= e for s, e in used_intervals):
                    used_intervals.append((event_start, event_end))
                    break  # Valid event, exit loop

            # Randomize event properties
            amplitude_factor = random.uniform(*amplitude_factor_range)
            spike_factor = random.uniform(*spike_factor_range)
            
            # Randomize spin start and end frequencies
            spin_start = random.uniform(*spin_start_range)  # Starting spin frequency (in Hz)
            spin_end = random.uniform(*spin_end_range)  # Ending spin frequency (in Hz)

            events.append((event_start, event_end, amplitude_factor * scale, spike_factor * scale, spin_start, spin_end))

        return events

    def apply_events(self, y, events):
        """
        Apply generated events generated by add_gw_signal function to the input signal.
        Can be manually created using this function 
        """
        for start, end, amplitude, spike, spin_start, spin_end in events:
            self.add_gw_event(y, start, end, amplitude_factor=amplitude, spike_factor=spike, spin_start=spin_start, spin_end=spin_end)


# Create the GWSignalGenerator instance
time_steps = 5000
time_for_1_sample = 50
x = np.linspace(0, time_for_1_sample, time_steps)
num_samples = 5
step_length = time_steps//100//(num_samples-1)
batch_size = time_steps//50*(num_samples-1)
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
regularization_values = np.logspace(-10, 0, 11)
gw_earlyboosts = np.linspace(1, 1.5, 6)
epoch_list = [10, 25, 50, 100]
SNR = 100

y = []
events = []
labels = []

y = [
    0.5*np.sin(90*x) - 0.5*np.cos(60*x)*np.sin(-5.*x) + 0.3*np.cos(30*x) + 0.05*np.sin(time_steps/40*x),
    0.5*np.sin(50*x) - 0.5*np.cos(80*x)*np.sin(-10*x) + 0.3*np.cos(40*x) + 0.05*np.sin(time_steps/20*x),
    0.5*np.sin(40*x) - 0.5*np.cos(25*x)*np.sin(-10*x) + 0.3*np.cos(60*x) + 0.10*np.sin(time_steps/18*x),
    0.7*np.sin(70*x) - 0.4*np.cos(10*x)*np.sin(-15*x) + 0.4*np.cos(80*x) + 0.05*np.sin(time_steps/12*x),
    0.1*np.sin(80*x) - 0.4*np.cos(50*x)*np.sin(-3.*x) + 0.3*np.cos(20*x) + 0.02*np.sin(time_steps/30*x)
]


for i in range(len(y)):
    y[i] /= SNR # Quick rescaling, the division factor is ~ SNR

event_lengths = [(time_steps//10, time_steps//8), (time_steps//7, time_steps//6), 
                 (time_steps//14, time_steps//12), (time_steps//5, time_steps//3),
                 (time_steps//5, time_steps//4)]

for i in range(num_samples):
    generator = GWSignalGenerator(signal_length=time_steps)
    # y_i = np.zeros_like(x) # For no background signal tests
    events_i = generator.generate_random_events(1, event_lengths[i])
    generator.apply_events(y[i], events_i)

    # y.append(y_i)
    events.append(events_i)
    labels.append(generator.labels)

# Convert lists into numpy arrays
y = np.array(y)
labels = np.array(labels)

# Reshape y for RNN input: (samples, time_steps, features)
y = y.reshape((y.shape[0], y.shape[1], 1))

# Prepare to save data
save_path = "GW_Parameter_Tuning_Results"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Function to save results incrementally
def save_results_incrementally(results, base_filename):
    filename = f"{base_filename}.pkl"
    with open(os.path.join(save_path, filename), "wb") as f:
        pickle.dump(results, f)

progress = 0
total_iterations = len(learning_rates)*len(regularization_values)*len(gw_earlyboosts)*len(epoch_list)*num_samples
start_time = time.time()

# Loop over learning rates and regularization values
for epochs in epoch_list:
    for boost in gw_earlyboosts:
        for lr in learning_rates:
            for reg_value in regularization_values:
                results = []

                # Create a unique filename for the current parameter combination
                base_filename = f"Synthetic_GW_Parameter_Tuning_Results_timesteps{time_steps}_SNR{SNR}_epoch{int(epochs)}_lamd{reg_value}_eta{lr}_boost{boost:.1f}"

                # Check if results already exist
                if os.path.exists(os.path.join(save_path, f"{base_filename}.pkl")):
                    print(f"Skipping calculation for {base_filename} as the results already exist.")
                    total_iterations -= num_samples
                    continue  # Skip the calculation and move to the next combination

                # plt.figure(figsize=(20, 12))
                # plt.suptitle(fr"$\eta={lr}$, $\lambda={reg_value}$")
                print(f"\nTraining with eta = {lr}, lambda = {reg_value}, epochs = {epochs}, early boost = {boost:.1f}")

                for fold in range(num_samples):                    
                    # Split the data into train and test sets for this fold
                    x_test = x  # Use the fold as the test set
                    y_test = y[fold]  # Corresponding labels for the test set
                    test_labels = labels[fold]

                    # Create the training set using all other samples
                    x_train = np.linspace(0, (num_samples - 1) * time_for_1_sample, time_steps * (num_samples - 1))  # Just for plotting
                    y_train = np.concatenate([y[i] for i in range(num_samples) if i != fold], axis=0)
                    train_labels = np.concatenate([labels[i] for i in range(num_samples) if i != fold], axis=0)
                    
                    # Initialize the KerasRNN model with the current learning rate and regularization
                    hidden_layers = [5, 10, 2]  # Example hidden layers
                    model = KerasRNN(
                        hidden_layers, 
                        dim_output=1, 
                        dim_input=1, 
                        labels=train_labels, 
                        gw_class_early_boost=boost, 
                        learning_rate=lr,
                        l2_regularization=reg_value
                    )

                    # Recompile the model with updated regularization
                    model.model.compile(
                        loss=model.loss_function, 
                        optimizer=model.optimizer, 
                        metrics=['accuracy']
                    )
                    # Train the model for this fold
                    model.train(y_train, train_labels, epochs=int(epochs), batch_size=batch_size, step_length=step_length, verbose=0)

                    # Predict with the trained model
                    predictions, loss, accuracy = model.predict(y_test, test_labels, step_length*(num_samples-1), verbose=0)
                    predictions = predictions.reshape(-1)
                    predicted_labels = (predictions > 0.5).astype(int)
                    x_pred = x[step_length - 1:]

                    results.append({
                        "epochs": epochs,
                        "boost": boost,
                        "learning_rate": lr,
                        "regularization": reg_value,
                        "fold": fold,
                        "x_train": x_train.tolist(),
                        "y_train": y_train.tolist(),
                        "train_labels": train_labels.tolist(),
                        "x_test": x_test.tolist(),
                        "y_test": y_test.tolist(),
                        "test_labels": test_labels.tolist(),
                        "predictions": predictions.tolist(),
                        "loss": loss,
                        "accuracy": accuracy
                    })

                    # plt.subplot(2, 3, fold + 1)
                    # plt.title(f"Round {fold+1}")
                    # plt.plot(x, y[fold], label=f'Data {fold+1}', lw=0.5, color='b')
                    # plt.plot(x, test_labels, label=f"Solution {fold+1}", lw=1.6, color='g')

                    # Highlight predicted events
                    # predicted_gw_indices = np.where(predicted_labels == 1)[0]
                    # if len(predicted_gw_indices) == 0:
                    #     print("No gravitational wave events predicted.")
                    # else:
                    #     threshold = 2
                    #     grouped_gw_indices = []
                    #     current_group = [predicted_gw_indices[0]]

                    #     for i in range(1, len(predicted_gw_indices)):
                    #         if predicted_gw_indices[i] - predicted_gw_indices[i - 1] <= threshold:
                    #             current_group.append(predicted_gw_indices[i])
                    #         else:
                    #             grouped_gw_indices.append(current_group)
                    #             current_group = [predicted_gw_indices[i]]

                    #     grouped_gw_indices.append(current_group)
                    #     for i, group in zip(range(len(grouped_gw_indices)), grouped_gw_indices):
                    #         plt.axvspan(x[group[0]], x[group[-1]], color="red", alpha=0.3, label="Predicted event" if i == 0 else "")
                    progress += 1
                    percentage_progress = (progress / total_iterations) * 100
                    # Elapsed time
                    elapsed_time = time.time() - start_time
                    ETA = elapsed_time*(100/percentage_progress-1)

                    print(f"Progress: {progress}/{total_iterations} ({percentage_progress:.2f}%), Time elapsed = {elapsed_time:.1f}s, ETA = {ETA:.1f}s, Test loss = {loss:.3f}, Test accuracy = {100*accuracy:.1f}%\n")

                    # plt.legend()
                # if savefigs:
                    # plt.savefig(f"../Figures/SyntheticGWs_timesteps{time_steps}_SNR{SNR}_lr{lr}_lambd{reg_value}_epochs{int(epochs)}_earlyboost{boost:.1f}.pdf")  # Save the figure
                # Save results incrementally
                save_results_incrementally(results, base_filename)

>>>>>>> 9b22f8cc0b32e471421531d9ff6410a612ece053
