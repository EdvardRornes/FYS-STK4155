
import numpy as np
from NNs import NeuralNetwork, Optimizer, Activation, Scalers, FocalLoss, DynamicallyWeightedLoss, WeightedBinaryCrossEntropyLoss, Loss
import copy 
import time 

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
            self.data["loss_function"] = {str(type(self._loss_function)): self._loss_function.data}
        
        elif isinstance(loss_function, Loss):
            self._loss_function = loss_function
            self.data["loss_function"] = {str(type(self._loss_function)): self._loss_function.data}
        
        else:
            raise TypeError(f"Need an instance of the Loss class as loss function.")
    


        self.layers = [input_size] + hidden_layers + [output_size]

        ### Private variables:
        self._trained = True

        # Initialize weights, biases, and optimizers
        self._initialize_weights_and_b_hh()

    #################### Public Methods ####################
    def train(self, X:np.ndarray, y:np.ndarray, epochs=100, batch_size=32, window_size=10, truncation_steps=None, split_data=False) -> None:
        """
        Trains the RNN on the given dataset.
        
        Arguments:
        * X:                input time series data of shape (num_samples, input_size)
        * y:                output labels of shape (num_samples, output_size)
        
        Keyword Arguments:
        * epochs:           number of training epochs
        * batch_size:       size of each mini-batch for training
        * window_size:      size of the window for creating sequences (step length)
        * truncation_steps: index, corresponding to the maximum amount of time points to propagate backward when computing gradient
        """
        self._trained = True

        # Prepare sequences for training
        self.store_train_test_from_data(X, y, split_data=True) # applies scaling
        self.X_seq, self.y_seq = self.prepare_sequences_RNN(self.X_train_scaled, self.y_train, window_size, self.input_size)
        self.X_seq_test, self.y_seq_test = self.prepare_sequences_RNN(self.X_test_scaled, self.y_test, window_size, self.input_size)

        self.truncation_steps = truncation_steps # Sent to backward propagation method 

        if isinstance(self._loss_function, DynamicallyWeightedLoss):
            self._loss_function.epochs = epochs
            self._loss_function.labels = y

        start_time = time.time(); best_loss = 1e4

        prev_weights = copy.deepcopy(self.weights)
        prev_weights_recurrent = copy.deepcopy(self.W_hh)
        
        test_loss = 1
        for epoch in range(epochs):
            
            self.weights = prev_weights
            self.W_hh = prev_weights_recurrent
            for i in range(0, self.X_seq.shape[0], batch_size):
                X_batch = self.X_seq[i:i + batch_size]
                y_batch = self.y_seq[i:i + batch_size]

                # Forward pass
                y_pred = self._forward(X_batch)

                # Backward pass
                self._backward(X_batch, y, y_pred, epoch, i)

                print(f"Epoch {epoch + 1}/{epochs} completed, loss: {test_loss:.3f}, time elapsed: {time.time()-start_time:.1f}s", end="\r")
            
            y_pred = self._forward(self.X_seq_test)
    
            test_loss = self.calculate_loss(self.y_seq_test, y_pred, epoch)

            if test_loss < best_loss:
                best_loss = test_loss

                prev_weights = copy.deepcopy(self.weights)
                prev_weights_recurrent = copy.deepcopy(self.W_hh)

        print("                                                                                           ", end="\r")
        print(f"Epoch {epochs}/{epochs} completed, time elapsed: {time.time()-start_time:.1f}s")

        X_test_seq, y_test_seq = self.prepare_sequences_RNN(self.X_test_scaled, self.y_test, window_size, self.input_size)

        y_test_pred = self._forward(X_test_seq)
        test_loss = self.calculate_loss(y_test_seq, y_test_pred, 1)
        print(f"Test Loss: {test_loss}")
        print("Training completed")

    def calculate_loss(self, y_true, y_pred, epoch_index):
        """
        Calculate the loss value using the provided loss function.
        """
        return self._loss_function(y_true, y_pred, epoch_index)
    
    def predict(self, X:np.ndarray):
        return self._forward(X)
    
    #################### Private Methods ####################
    def _forward(self, X):
        batch_size, window_size, _ = X.shape
        num_hidden_layers = len(self.layers) - 1  # Exclude the output layer

        # Initialize hidden states for each time step
        self.hidden_states = [[np.zeros((batch_size, self.layers[i + 1])) for i in range(num_hidden_layers)] for _ in range(window_size)]

        for i in range(window_size):
            input_t = X[:, i, :]

            for j in range(num_hidden_layers):
                if j == 0:
                    prev_activation = input_t
                else:
                    prev_activation = self.hidden_states[i][j - 1]

                h_prev = np.zeros_like(self.hidden_states[i - 1][j]) if i == 0 else self.hidden_states[i - 1][j]

                h = self.activation_func(
                    prev_activation @ self.W_xh[j] + h_prev @ self.W_hh[j] + self.biases[j]
                )
                self.hidden_states[i][j] = h

        # Output layer uses the hidden state from the last time step
        output = self.hidden_states[-1][-1] @ self.W_hh[-1] + self.b_hh[-1]
        output = self.activation_func_out(output)
        return output

    def _backward(self, X, y, y_pred, epoch_index, batch_index):

        batch_size, window_size, _ = X.shape

        for t in reversed(range(window_size)):
            for j in range(len(self.W_hh) - 1):
                dL_dht = self._dL_dht(X, j, t, window_size, y, epoch_index)
                
                print(np.shape(dL_dht), np.shape(self.hidden_states[t-1][j].T))
                dL_dW_hh = self.hidden_states[t-1][j].T @ dL_dht
                self.W_xh[j] = self.optimizer_W_xh[j](self.W_xh[j], dL_dW_hh, epoch_index, batch_index)
                self.W_hh[j] = self.optimizer_W_hh[j](self.W_hh[j], dL_dW_hh, epoch_index, batch_index)

            dL_dhT = self.W_hh[-1].T @ self._loss_function.gradient(y[:, window_size-1, :], y_pred)
            self.W_hh[-1] = self.optimizer_W_hh[-1](self.W_hh[-1], dL_dhT, epoch_index, batch_index)

    def _dL_dht(self, X: np.ndarray, j: int, t: int, window_size: int, y: np.ndarray, epoch_index: int):
        tmp = 0
        batch_size, _, _ = X.shape

        # Iterate over future time steps
        for i in range(t, window_size):
            index = window_size - 1 + t - i

            # Compute predicted output
            y_pred = self.hidden_states[index][-1] @ self.W_hh[-1] + self.b_hh[-1]
            print(np.shape(y_pred), "hsdf")
            # Compute gradient with respect to the loss
            print(np.shape(self.y_seq[index]), np.shape(y_pred))
            loss_grad = self._loss_function.gradient(self.y_seq[index], y_pred, epoch_index)
            # Compute matrix powers for W_hh
            print(j, self.W_hh[j].T.shape, "der")
            power_matrix = np.eye(self.W_hh[j].T.shape)
            for _ in range(window_size - i):
                power_matrix = power_matrix @ self.W_hh[j].T

            # Accumulate gradient contributions
            print(np.shape(power_matrix))
            print(np.shape(self.hidden_states[i-1][j].T))
            print(np.shape(loss_grad), np.shape((self.hidden_states[i-1][j].T @ loss_grad)), "her")

            tmp += power_matrix @ (self.hidden_states[i-1][j].T @ loss_grad)

        return tmp


    def _initialize_weights_and_b_hh(self):
        """
        Initializes weights and biases for each layer of the RNN.
        """
        num_hidden_layers = len(self.layers) - 1  # Exclude the output layer
        self.W_xh = []
        self.W_hh = []
        self.b_hh = []
        self.optimizer_W_xh = []
        self.optimizer_W_hh = []
        self.optimizer_b_hh = []

        for i in range(num_hidden_layers):
            # Input weights: from input or previous hidden layer to current hidden layer
            input_dim = self.layers[i] #if i == 0 else self.layers[i + 1]
            output_dim = self.layers[i + 1] 
            self.W_xh.append(NeuralNetwork._xavier_init(input_dim, output_dim))
            self.optimizer_W_xh.append(self.optimizer.copy())

            # Recurrent weights: from current hidden state to next hidden state
            self.W_hh.append(NeuralNetwork._xavier_init(output_dim, output_dim))
            self.optimizer_W_hh.append(self.optimizer.copy())

            # Biases for hidden layers
            self.biases.append(np.zeros((1, output_dim)))
            self.optimizer_b_hh.append(self.optimizer.copy())

        # Output layer weights and biases
        self.W_hh.append(NeuralNetwork._xavier_init(self.layers[-1], self.output_size))

        self.b_hh.append(np.zeros((1, self.output_size)))
        self.optimizer_W_hh = self.optimizer.copy()
        self.optimizer_b_hh = self.optimizer.copy()