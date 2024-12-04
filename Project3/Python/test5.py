
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
        """
        self._trained = True

        # Prepare sequences for training
        self.store_train_test_from_data(X, y, split_data=False) # applies scaling
        self.X_seq, self.y_seq = self.prepare_sequences_RNN(self.X_train_scaled, self.y_train, window_size, self.input_size)
        self.X_seq_test, self.y_seq_test = self.prepare_sequences_RNN(self.X_test_scaled, self.y_test, window_size, self.input_size)

        print(np.shape(self.y_seq))
        self.truncation_steps = truncation_steps # Sent to backward propagation method 

        if isinstance(self._loss_function, DynamicallyWeightedLoss):
            self._loss_function.epochs = epochs
            self._loss_function.labels = y

        start_time = time.time()
        best_loss = 1e4

        # Initialize best weights
        best_weights = copy.deepcopy(self.W_hx)
        best_weights_recurrent = copy.deepcopy(self.W_hh)
        best_W_yh = copy.deepcopy(self.W_yh)
        best_b_hh = copy.deepcopy(self.b_hh)
        best_b_y = copy.deepcopy(self.b_y)

        best_loss = float('inf')
        best_weights = None
        for epoch in range(epochs):
            for i in range(0, self.X_seq.shape[0], batch_size):
                X_batch = self.X_seq[i:i + batch_size]
                y_batch = self.y_seq[i:i + batch_size]

                # Forward pass
                y_pred = self._forward(X_batch)

                # Backward pass
                self._backward(X_batch, y_batch, y_pred, epoch, i)

            # Validation after each epoch
            y_pred = self._forward(self.X_seq_test)
            test_loss = self.calculate_loss(self.y_seq_test, y_pred, epoch)

            if test_loss < best_loss:
                best_loss = test_loss
                # Save best weights
                best_weights = {
                    'W_hx': copy.deepcopy(self.W_hx),
                    'W_hh': copy.deepcopy(self.W_hh),
                    'W_yh': copy.deepcopy(self.W_yh),
                    'b_hh': copy.deepcopy(self.b_hh),
                    'b_y': copy.deepcopy(self.b_y),
                }

            print(f"Epoch {epoch + 1}/{epochs} completed, loss: {test_loss:.3f}, time elapsed: {time.time()-start_time:.1f}s", end="\r")

        # Restore best weights after training
        if best_weights is not None:
            self.W_hx = best_weights['W_hx']
            self.W_hh = best_weights['W_hh']
            self.W_yh = best_weights['W_yh']
            self.b_hh = best_weights['b_hh']
            self.b_y = best_weights['b_y']

        print("\nTraining completed.")


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

        # Initialize hidden states and z values for each time step
        self.hidden_states = [[np.zeros((batch_size, self.layers[i + 1])) for i in range(num_hidden_layers)] for _ in range(window_size)]
        self.z_values = [[np.zeros((batch_size, self.layers[i + 1])) for i in range(num_hidden_layers)] for _ in range(window_size)]
        self.outputs = []  # To store outputs at each time step
        self.z_out = []    # To store z_out at each time step

        for i in range(window_size):  # For each time step
            for j in range(num_hidden_layers):
                # Previous hidden state for the current layer
                if i == 0:
                    h_prev = np.zeros((batch_size, self.layers[j + 1]))
                else:
                    h_prev = self.hidden_states[i - 1][j]

                # Input to the current layer
                if j == 0:
                    x_t = X[:, i, :]
                else:
                    x_t = self.hidden_states[i][j - 1]

                # Compute the pre-activation and activation
                z = x_t @ self.W_hx[j] + h_prev @ self.W_hh[j] + self.b_hh[j]
                self.z_values[i][j] = z
                self.hidden_states[i][j] = self.activation_func(z)

            # Compute output at each time step
            z_out = self.hidden_states[i][-1] @ self.W_yh + self.b_y
            self.z_out.append(z_out)
            output = self.activation_func_out(z_out)
            self.outputs.append(output)

        # Convert outputs and z_out to arrays
        self.z_out = np.array(self.z_out)  # Shape: (window_size, batch_size, output_size)
        self.outputs = np.array(self.outputs)  # Shape: (window_size, batch_size, output_size)

        # Shape: (window_size, batch_size, output_size) -> (batch_size, window_size, output_size)
        self.z_out = self.z_out.transpose(1, 0, 2)
        self.outputs = self.outputs.transpose(1, 0, 2)

        return self.outputs  # Shape: (batch_size, window_size, output_size)




    def _backward(self, X_batch, y_batch, y_pred, epoch, batch_index):
        batch_size, window_size, _ = X_batch.shape
        num_hidden_layers = len(self.layers) - 1  # Exclude the output layer

        # Initialize gradients
        dW_hx = [np.zeros_like(w) for w in self.W_hx]
        dW_hh = [np.zeros_like(w) for w in self.W_hh]
        db_hh = [np.zeros_like(b) for b in self.b_hh]
        dW_yh = np.zeros_like(self.W_yh)
        db_y = np.zeros_like(self.b_y)

        # Initialize delta arrays
        delta = [[np.zeros((batch_size, self.layers[i + 1])) for i in range(num_hidden_layers)] for _ in range(window_size)]
        delta_out = np.zeros((batch_size, window_size, self.output_size))

        # Compute delta_out for each time step
        loss_derivative = self._loss_function.gradient(y_pred, y_batch, epoch)  # Shape: (batch_size, window_size, output_size)
        for t in reversed(range(window_size)):
            # Shape: (batch_size, output_size)
            delta_out_t = loss_derivative[:, t, :] * self.activation_func_out_derivative(self.z_out[:, t, :])  
            delta_out[:, t, :] = delta_out_t

            # Gradients for output weights and biases
            h_t = self.hidden_states[t][-1]  # Shape: (batch_size, hidden_size)
            dW_yh += h_t.T @ delta_out_t   # Shape: (hidden_size, output_size)
        
            db_y += np.sum(delta_out_t, axis=0) 

        # Backpropagate through time
        truncation_steps = self.truncation_steps if self.truncation_steps else window_size
        truncation_steps = window_size - 100
        for t in reversed(range(window_size)):
            if window_size - t > truncation_steps:
                break  # Stop backpropagation after truncation_steps

            for l in reversed(range(num_hidden_layers)):
                # Sum contributions
                if l == num_hidden_layers - 1:
                    delta_t = delta_out[:, t, :] @ self.W_yh.T
                else:
                    delta_t = delta[t][l + 1] @ self.W_hx[l + 1].T

                if t < window_size - 1:
                    delta_t += delta[t + 1][l] @ self.W_hh[l].T

                # Apply activation derivative
                delta_t *= self.activation_func_derivative(self.z_values[t][l])

                delta[t][l] = delta_t

                # Compute gradients
                if l == 0:
                    x_input = X_batch[:, t, :]
                else:
                    x_input = self.hidden_states[t][l - 1]

                # h_prev = self.hidden_states[t - 1][l] if t > 0 else np.zeros_like(self.hidden_states[0][l])
                h_prev = self.hidden_states[t - 1][l] if t > 0 else self.hidden_states[0][l]

                dW_hx[l] += x_input.T @ delta_t 
                dW_hh[l] += h_prev.T @ delta_t 
                db_hh[l] += np.sum(delta_t, axis=0) 

        clip_value = 1
        if clip_value is not None:
            dW_hx = [self._clip_gradient(w, clip_value) for w in dW_hx]
            dW_hh = [self._clip_gradient(w, clip_value) for w in dW_hh]
            db_hh = [self._clip_gradient(b, clip_value) for b in db_hh]
            dW_yh = self._clip_gradient(dW_yh, clip_value)
            db_y = self._clip_gradient(db_y, clip_value)

        # Update weights with gradients
        for l in range(num_hidden_layers):
            self.W_hx[l] = self.optimizer_W_hx[l](self.W_hx[l], dW_hx[l], epoch, batch_index)
            self.W_hh[l] = self.optimizer_W_hh[l](self.W_hh[l], dW_hh[l], epoch, batch_index)
            self.b_hh[l] = self.optimizer_b_hh[l](self.b_hh[l], db_hh[l], epoch, batch_index)

        self.W_yh = self.optimizer_W_yh(self.W_yh, dW_yh, epoch, batch_index)
        self.b_y = self.optimizer_b_y(self.b_y, db_y, epoch, batch_index)

    def _clip_gradient(self, grad, clip_value):
        """
        Clips the gradients to prevent exploding gradients.
        Arguments:
            grad: The gradient array to clip.
            clip_value: The maximum allowable value for the L2 norm of the gradient.
        Returns:
            The clipped gradient.
        """
        grad_norm = np.linalg.norm(grad)
        if grad_norm > clip_value:
            grad = grad * (clip_value / grad_norm)  # Scale down gradient to the threshold
        return grad

    def _initialize_weights_and_b_hh(self):
        """
        Initializes weights and biases for each layer of the RNN.
        """
        num_hidden_layers = len(self.layers) - 1  # Exclude the output layer
        self.W_hx = []
        self.W_hh = []
        self.b_hh = []
        self.optimizer_W_hx = []
        self.optimizer_W_hh = []
        self.optimizer_b_hh = []

        for i in range(num_hidden_layers):
            # Input weights: from input or previous hidden layer to current hidden layer
            input_dim = self.layers[i] #if i == 0 else self.layers[i + 1]
            output_dim = self.layers[i + 1] 
            self.W_hx.append(NeuralNetwork._xavier_init(input_dim, output_dim))
            self.optimizer_W_hx.append(self.optimizer.copy())

            # Recurrent weights: from current hidden state to next hidden state
            self.W_hh.append(NeuralNetwork._xavier_init(output_dim, output_dim))
            self.optimizer_W_hh.append(self.optimizer.copy())

            # Biases for hidden layers
            self.b_hh.append(np.zeros((1, output_dim)))
            self.optimizer_b_hh.append(self.optimizer.copy())

        # Output layer weights and biases
        self.W_yh = NeuralNetwork._xavier_init(self.layers[-1], self.output_size)

        self.b_y = np.zeros((1, self.output_size))
        self.optimizer_W_yh = self.optimizer.copy()
        self.optimizer_b_y = self.optimizer.copy()