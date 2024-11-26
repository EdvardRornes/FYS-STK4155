import numpy as np
import autograd.numpy as anp


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
    
class FFNNOLD:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', alpha=0.01, lambda_reg=0.0, 
                 beta1=0.9, beta2=0.999, epsilon=1e-8, loss_function='mse'):
        """
        Initialize the Feedforward Neural Network (FFNN) with Adam optimizer.
        
        Parameters:
        - input_size (int): Number of input features.
        - hidden_layers (list): List of integers representing the size of each hidden layer.
        - output_size (int): Number of output neurons.
        - activation (str): Activation function to use ('relu', 'sigmoid', 'lrelu').
        - alpha (float): Leaky ReLU parameter (only for 'lrelu').
        - lambda_reg (float): L2 regularization parameter.
        - beta1 (float): Exponential decay rate for the first moment estimate in Adam.
        - beta2 (float): Exponential decay rate for the second moment estimate in Adam.
        - epsilon (float): Small constant to prevent division by zero in Adam.
        - loss_function (str): Loss function to use ('mse' or 'bce').
        """
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        self.activation_func = activation
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Initialize time step for Adam
        self.loss_function = loss_function  # Added loss function parameter

        # Initialize activation functions mapping
        self.activation_map = {
            'relu': (Activation.relu, Activation.relu_derivative),
            'sigmoid': (Activation.sigmoid, Activation.sigmoid_derivative),
            'lrelu': (lambda z: Activation.Lrelu(z, self.alpha), lambda z: Activation.Lrelu_derivative(z, self.alpha))
        }
        
        self.activation, self.activation_derivative = self.activation_map.get(self.activation_func.lower(), (Activation.relu, Activation.relu_derivative))

        # Initialize weights, biases, and Adam parameters
        self.initialize_weights_and_biases()
        self.initialize_adam_parameters()

    def initialize_weights_and_biases(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            weight_matrix = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2. / self.layers[i])
            self.weights.append(weight_matrix)
            bias_vector = np.zeros((1, self.layers[i + 1]))
            self.biases.append(bias_vector)
    
    def initialize_adam_parameters(self):
        # Initialize moment estimates for Adam optimizer
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
    
    def forward(self, X):
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

    def compute_loss(self, y_true, y_pred):
        if self.loss_function == 'mse':
            return np.mean((y_pred - y_true) ** 2)
        elif self.loss_function == 'bce':
            # Adding a small epsilon to avoid log(0)
            return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        else:
            raise ValueError("Invalid loss function specified. Use 'mse' or 'bce'.")

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        y = y.reshape(-1, 1)

        output = self.activations[-1]

        delta = output - y
        self.t += 1  # Increment time step

        for i in reversed(range(len(self.weights))):
            dw = (self.activations[i].T @ delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            dw += (self.lambda_reg / m) * self.weights[i]

            # Update first moment estimate
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * dw
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * db
            # Update second moment estimate
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (dw ** 2)
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (db ** 2)
            # Correct bias in moment estimates
            m_hat_w = self.m_weights[i] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m_biases[i] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v_weights[i] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_biases[i] / (1 - self.beta2 ** self.t)
            # Update weights and biases
            self.weights[i] -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            self.biases[i] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

            if i > 0:
                delta = (delta @ self.weights[i].T) * self.activation_derivative(self.z_values[i - 1])

    def train(self, X, y, learning_rate=0.001, epochs=1000, batch_size=None, shuffle=True, lambda_reg=0.0):
        self.lambda_reg = lambda_reg
        self.initialize_weights_and_biases()
        self.initialize_adam_parameters()
        loss_history = []
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

            # Calculate and store the loss
            loss = self.compute_loss(y_batch, self.activations[-1])
            loss_history.append(loss)

            # if epoch % (epochs // 5) == 0:
            #     print(f'Epoch {epoch}, Loss ({self.loss_function}): {loss:.3f}')
        
        return loss_history

    def predict(self, X):
        return self.forward(X)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        if self.loss_function == 'bce':
            predicted_classes = (predictions > 0.5).astype(int)
            return np.mean(predicted_classes.flatten() == y.flatten())
        else:
            return np.mean(np.round(predictions) == y.flatten())