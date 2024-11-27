from utils import * 
import numpy as np

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

import numpy as np
import time

class RNN(NeuralNetwork):
    def __init__(self, input_size: int, hidden_layers: list, output_size: int, optimizer: Optimizer, activation,
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
        super().__init__()
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        self.activation_func_name = activation
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.hidden_size = hidden_layers[-1]

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
