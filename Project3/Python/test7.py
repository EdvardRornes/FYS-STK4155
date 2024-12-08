
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

        self.truncation_steps = truncation_steps # Sent to backward propagation method 

        if isinstance(self._loss_function, DynamicallyWeightedLoss):
            self._loss_function.epochs = epochs
            self._loss_function.labels = y

        start_time = time.time()
        best_loss = 1e4

        # Initialize best weights
        best_weights = copy.deepcopy(self.W_hx)

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
            y_pred = self.predict(self.X_seq_test)
            # print(np.shape(y_pred.T), np.shape(self.y_seq_test))
            test_loss = self.calculate_loss(self.y_seq_test, y_pred, epoch)

            if test_loss < best_loss:
                best_loss = test_loss
                # Save best weights
                best_weights = {
                    'W_hx': copy.deepcopy(self.W_hx),
                    'W_hh': copy.deepcopy(self.W_hh),
                    'W_yh': copy.deepcopy(self.W_yh),
                    'b_hh': copy.deepcopy(self.b_h),
                    'b_y': copy.deepcopy(self.b_y),
                }

            print(f"Epoch {epoch + 1}/{epochs} completed, loss: {test_loss:.3f}, time elapsed: {time.time()-start_time:.1f}s", end="\r")

        # Restore best weights after training
        if best_weights is not None:
            self.W_hx = best_weights['W_hx']
            self.W_hh = best_weights['W_hh']
            self.W_yh = best_weights['W_yh']
            self.b_h = best_weights['b_hh']
            self.b_y = best_weights['b_y']

        print("\nTraining completed.")


    def calculate_loss(self, y_true, y_pred, epoch_index):
        """
        Calculate the loss value using the provided loss function.
        """
        return self._loss_function(y_true, y_pred, epoch_index)
    
    def predict(self, X:np.ndarray):
        return np.array(self._forward(X)).transpose(1, 0, 2)
    
    #################### Private Methods ####################
    def _forward(self, X_batch: np.ndarray):
        """
        Forward pass for the RNN.
        Parameters:
            X_batch: The input data for a batch (batch_size, window_size, input_size).
        Returns:
            y_pred: The predicted output for the batch.
            z: Pre-activation values for hidden layers.
            hidden_states: The hidden states of each layer for each timestep.
        """
        batch_size, window_size, _ = X_batch.shape

        # Initialize hidden states and z (pre-activation values)
        self.hidden_states = [[np.zeros((self.layers[l + 1], batch_size)) for _ in range(window_size)] for l in range(self.L)]
        self.z = [[np.zeros((self.layers[l + 1], batch_size)) for _ in range(window_size)] for l in range(self.L)]

        # Output for each timestep
        outputs = []
        
        for t in range(window_size):
            x_t = X_batch[:, t, :]
            prev_state = np.zeros_like(self.hidden_states[0][0])
            
            for l in range(self.L - 1):
                self.z[l][t] = self.W_hx[l] @ x_t.T + self.W_hh[l] @ prev_state + self.b_h[l]
                # if l == 0:
                #     print(np.shape(self.z[l][t]), np.shape(self.W_hx[l]), np.shape(x_t), "der")
                self.hidden_states[l][t] = self.activation_func(self.z[l][t])
                
                # Next iteration:
                prev_state = self.hidden_states[l+1][t]
                x_t = self.hidden_states[l][t].T
            
            self.z[-1][t] = self.W_yh @ self.hidden_states[-2][t] + self.b_y 
            
            self.hidden_states[-1][t] = self.activation_func_out(self.z[-1][t]).T

        return self.hidden_states[-1]


    def _backward(self, X_batch: np.ndarray, y_batch: np.ndarray, y_pred: np.ndarray, epoch: int, batch_index: int):
        """
        Backward pass for the RNN.

        Parameters:
            X_batch: Input data for the batch, shape (batch_size, window_size, input_size)
            y_batch: True output data for the batch, shape (batch_size, window_size, output_size)
            y_pred: Predicted output data, shape (batch_size, window_size, output_size)
            epoch: Current epoch number
            batch_index: Index of the batch in the current epoch
        """
        batch_size, window_size, _ = X_batch.shape

        # Initialize gradients for output layer
        dL_dW_yh = np.zeros_like(self.W_yh)    # Shape: (output_size, hidden_size)
        dL_db_y = np.zeros_like(self.b_y)      # Shape: (output_size, 1)

        # Initialize gradients for hidden layers
        dL_dW_hx = [np.zeros_like(w) for w in self.W_hx]  # List of arrays, one per hidden layer
        dL_dW_hh = [np.zeros_like(w) for w in self.W_hh]
        dL_db_h = [np.zeros_like(b) for b in self.b_h]

        # Compute the derivative of the loss with respect to the output (dL/dy_pred)
        # Assuming the loss function has a method `derivative` that returns dL/dy_pred
        y_pred = np.array(y_pred).transpose(1, 0, 2)
        dL_dy_pred = self._loss_function.gradient(y_batch, y_pred)  # Shape: (batch_size, window_size, output_size)

        # Iterate over each time step to compute gradients
        for t in range(window_size):
            # Compute gradients for the output layer at time step t
            # Shape considerations:
            # y_pred[:, t, :] -> (batch_size, output_size)
            # self.hidden_states[-1][t] -> (output_size, batch_size)
            # Transpose hidden_states to match dimensions
            h_prev = self.hidden_states[-2][t].T  # Shape: (batch_size, hidden_size)

            # Compute dL/dz_y at time step t
            # dL_dy_pred[:, t, :] has shape (batch_size, output_size)
            # d_activation_out has shape (batch_size, output_size)
            d_activation_out = self.activation_func_out_derivative(self.z[-1][t].T)  # Shape: (batch_size, output_size)
            dL_dz_y = dL_dy_pred[:, t, :] * d_activation_out  # Shape: (batch_size, output_size)

            # Accumulate gradients for W_yh and b_y
            dL_dW_yh += dL_dz_y.T @ h_prev  # (output_size, batch_size) @ (batch_size, hidden_size) -> (output_size, hidden_size)
            dL_db_y += np.sum(dL_dz_y, axis=0, keepdims=True).T  # (output_size, 1)

        # Compute gradients for hidden layers using BPTT
        # Initialize delta terms for each hidden layer
        delta_hh = [np.zeros((self.layers[l+1], self.layers[l+1])) for l in range(self.L)]  # List per layer
        delta_hx = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(self.L)]

        delta_hh = []
        delta_hx = []

        # Compute delta terms recursively for each hidden layer
        for l in range(self.L-1):
            delta_hh.append([])
            delta_hx.append([])

            # Recursive computation
            for k in range(window_size):
                if k == 0:
                    delta_hh[-1].append(np.zeros((self.layers[l+1], batch_size)))
                else:
                    if l == 0:
                        delta_hh[-1].append(self.activation_func_derivative(self.z[l][k]) * (self.W_hh[0] @ delta_hh[0][-1]))
                    else:
                        delta_hh[-1].append(self.activation_func_derivative(self.z[l][k]) * (self.hidden_states[l][k-1] + self.W_hh[l] @ delta_hh[l][-1]))
                
                if l == 0:
                    if k == 0:
                        delta_hx[-1].append(self.activation_func_derivative(self.z[l][k]) * (np.ones_like(self.W_hx[l]) @ X_batch[:, k, :].T))
                    else:
                        delta_hx[-1].append(self.activation_func_derivative(self.z[l][k]) * (np.ones_like(self.W_hx[l]) @ X_batch[:, k, :].T + self.W_hh[l] @ delta_hx[0][-1]))
                else:
                    if k == 0:
                        delta_hx[-1].append(self.activation_func_derivative(self.z[l][k]) * self.hidden_states[l][k])
                    else:
                        delta_hx[-1].append(self.activation_func_derivative(self.z[l][k]) * (self.hidden_states[l][k] + self.W_hh[l] @ delta_hx[l][-1]))
                # print(np.shape(delta_hx[-1][k]))
            print(np.shape(delta_hh[-1][-1]), np.shape(delta_hx[-1][-1]))


        # Accumulate gradients for hidden weights and biases
        for l in range(self.L-1):
            for n in range(window_size):
                for k in range(n + 1):
                    # Compute the product term: product from j=k+1 to n of sigma'_h(z_j)*W_hh
                    if k == n:
                        prod_term = 1  # Empty product
                    else:
                        prod_term = 1
                        for j in range(k + 1, n + 1):
                            sigma_prime = self.activation_func_derivative(self.z[l][j]).T  # Shape: (batch_size, hidden_size)
                            prod_term *= (sigma_prime @ self.W_hh[l])

                    # Compute dL/dh_n
                    # dL/dh_n is accumulated from the output layer gradients
                    # For simplicity, assume dL/dh_n is from the output layer if last layer
                    if l == self.L - 1:
                        dL_dh_n = self.W_yh.T @ (dL_dy_pred[:, n, :] * self.activation_func_out_derivative(self.z[-1][n]).T)  # Shape: (hidden_size, batch_size)
                    else:
                        # If not the last layer, propagate gradients from the upper layer
                        # This requires storing additional information during the forward pass
                        # For simplicity, we'll assume single hidden layer
                        dL_dh_n = np.zeros((self.layers[l + 1], batch_size))  # Placeholder

                    # Update gradients
                    print(np.shape(dL_dh_n), np.shape(delta_hx[l][k]), np.shape(prod_term))
                    print(np.shape(dL_dW_hx[l] ), np.shape(delta_hx[l][k] * prod_term))
                    dL_dW_hh[l] += dL_dh_n @ (delta_hh[l][k] * prod_term).T 
                    dL_dW_hx[l] += dL_dh_n @ (delta_hx[l][k] * prod_term).T 
                    dL_db_h[l] += np.sum(delta_hh[l][k] * prod_term, axis=1, keepdims=True)


        # Optionally, apply gradient clipping to prevent exploding gradients
        clip_value = 5.0  # You can adjust this value or make it a class parameter
        dL_dW_yh = self._clip_gradient(dL_dW_yh, clip_value)
        dL_db_y = self._clip_gradient(dL_db_y, clip_value)
        for l in range(self.L):
            dL_dW_hx[l] = self._clip_gradient(dL_dW_hx[l], clip_value)
            dL_dW_hh[l] = self._clip_gradient(dL_dW_hh[l], clip_value)
            dL_db_h[l] = self._clip_gradient(dL_db_h[l], clip_value)

        # Update weights and biases using the optimizer
        for l in range(self.L):
            self.W_hx[l] = self.optimizer_W_hx[l].update(self.W_hx[l], dL_dW_hx[l], epoch, batch_index)
            self.W_hh[l] = self.optimizer_W_hh[l].update(self.W_hh[l], dL_dW_hh[l], epoch, batch_index)
            self.b_h[l] = self.optimizer_b_hh[l].update(self.b_h[l], dL_db_h[l], epoch, batch_index)

        self.W_yh = self.optimizer_W_yh.update(self.W_yh, dL_dW_yh, epoch, batch_index)
        self.b_y = self.optimizer_b_y.update(self.b_y, dL_db_y, epoch, batch_index)


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
        self.L = len(self.layers) - 1
        self.W_hx = []
        self.W_hh = []
        self.b_h = []
        self.optimizer_W_hx = []
        self.optimizer_W_hh = []
        self.optimizer_b_hh = []

        for i in range(self.L-1):
            # Input weights: from input or previous hidden layer to current hidden layer
            input_dim = self.layers[i] #if i == 0 else self.layers[i + 1]
            output_dim = self.layers[i + 1] 
            self.W_hx.append(NeuralNetwork._xavier_init(output_dim, input_dim))
            self.optimizer_W_hx.append(self.optimizer.copy())

            # Recurrent weights: from current hidden state to next hidden state
            self.W_hh.append(NeuralNetwork._xavier_init(output_dim, output_dim))
            self.optimizer_W_hh.append(self.optimizer.copy())

            # Biases for hidden layers
            self.b_h.append(np.zeros((output_dim, 1)))
            self.optimizer_b_hh.append(self.optimizer.copy())

        # Output layer weights and biases
        self.W_yh = NeuralNetwork._xavier_init(self.output_size, self.layers[-2])

        self.b_y = np.zeros((self.output_size, 1))
        self.optimizer_W_yh = self.optimizer.copy()
        self.optimizer_b_y = self.optimizer.copy()