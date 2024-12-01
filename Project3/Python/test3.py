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
