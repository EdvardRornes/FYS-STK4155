
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
    def train(self, X:np.ndarray, y:np.ndarray, epochs=100, batch_size=32, window_size=10, truncation_steps=None, split_data=False, clip_value=1e12) -> None:
        """
        Trains the RNN on the given dataset.
        """
        self._trained = True
        self.clip_value = clip_value

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
                y_pred[i] = 1 * (y_pred[i] <= 0.5)

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
        y_pred = np.array(y_pred).transpose(1, 0, 2)
        dL_dy_pred = self._loss_function.gradient(y_batch, y_pred, epoch)  # Shape: (batch_size, window_size, output_size)

        # Compute gradients for hidden layers using BPTT
        # Initialize delta terms for each hidden layer
        delta_hh = []
        delta_hx = []
        dL_dh_n = []

        ### Looping thorugh all layers
        for l in range(self.L - 1):
            delta_hh.append([])
            delta_hx.append([])
            dL_dh_n.append([])

            ### dL_dh_n:
            for k in range(window_size):
                # Gradient from output layer
                dL_dy = self._loss_function.gradient(y_batch[:, k, :], y_pred[:, k, :], epoch)  # (B, d_out)
                dL_dhL = dL_dy * self.activation_func_out_derivative(self.z[-1][k]).T  # (d_out, B) * (d_out, B)
                
                # Starting from layer l+1:
                W_hx_j = self.W_hx[l+1]  if l + 1 < len(self.W_hx) else np.eye(self.output_size, self.layers[-2]) # (d_{h_j}, d_{h_{j-1}}):

                sigma_prime = self.activation_func_derivative(self.z[l+1][k]).T  # (B, d_{h_j})
                product_term = sigma_prime[..., None] * W_hx_j[None, :, :]  # (B, d_{h_j}, d_{h_{j-1}})

                # Chain upward through layers above j
                for j in range(l+1 + 1, self.L - 1):
                    sigma_prime_next = self.activation_func_derivative(self.z[j][k]).T  # (B, d_{h_{jj}})
                    W_hx_jj = self.W_hx[j]  # (d_{h_{jj}}, d_{h_{jj-1}})
                    next_product = sigma_prime_next[..., None] * W_hx_jj[None, :, :]  # (B, d_{h_{jj}}, d_{h_{jj-1}})
                    product_term = np.einsum('bij,bjk->bik', next_product, product_term)

                dL_dh_n[-1].append(np.einsum('bik,bi->bk', product_term, dL_dhL))  # (B, d_{h_l})


            ### \delta's:
                if k == 0:  # Initial condition
                    # Base case: No temporal dependency for k=0
                    delta = np.zeros((self.hidden_states[l][k].shape[0], self.W_hh[l].shape[0], X_batch.shape[0]))  # Shape: (d_{h_l}, d_{h_l}, B)
                    delta_hh[-1].append(delta)
                else:  # Recursive case
                    # Activation derivative: Shape (d_{h_l}, B)
                    activation_derivative = self.activation_func_derivative(self.z[l][k])

                    # Previous hidden state: Shape (d_{h_l}, B)
                    h_km1_l = self.hidden_states[l][k-1]  # Shape: (d_{h_l}, B)

                    # Contribution from previous delta term: Shape (d_{h_l}, d_{h_l}, B)
                    last_term = np.einsum('ij,jkb->ikb', self.W_hh[l], delta_hh[-1][-1])

                    # Combine contributions
                    combined_term = h_km1_l[:, np.newaxis, :] + last_term  # Shape: (d_{h_l}, d_{h_l}, B)

                    # Apply activation derivative
                    delta = activation_derivative[:, np.newaxis, :] * combined_term  # Shape: (d_{h_l}, d_{h_l}, B)
                    delta_hh[-1].append(delta)

                
                if l == 0:  # Input layer case
                    if k == 0:  # First time step
                        # Compute base case for delta: Shape (d_{h_l}, d_{h_{l-1}}, B)
                        activation_derivative = self.activation_func_derivative(self.z[l][k])  # Shape: (d_{h_l}, B)
                        h_k_lm1 = X_batch[:, k, :].T  # Shape: (d_{h_{l-1}}, B)

                        # Compute outer product for base case
                        delta = activation_derivative[:, np.newaxis, :] * h_k_lm1[np.newaxis, :, :]  # Shape: (d_{h_l}, d_{h_{l-1}}, B)
                        delta_hx[-1].append(delta)
                    else:  # Subsequent time steps
                        # Activation derivative: Shape (d_{h_l}, B)
                        activation_derivative = self.activation_func_derivative(self.z[l][k])

                        # Compute contribution from previous delta term
                        last_term = np.einsum('ij,jkb->ikb', self.W_hh[l], delta_hx[-1][-1])  # Shape: (d_{h_l}, d_{h_{l-1}}, B)

                        # Add new contribution from X_batch
                        h_k_lm1 = X_batch[:, k, :].T  # Shape: (d_{h_{l-1}}, B)
                        combined_term = h_k_lm1[np.newaxis, :, :] + last_term  # Shape: (d_{h_l}, d_{h_{l-1}}, B)

                        # Apply activation derivative
                        delta = activation_derivative[:, np.newaxis, :] * combined_term  # Shape: (d_{h_l}, d_{h_{l-1}}, B)
                        delta_hx[-1].append(delta)
                else:  # Hidden layers
                    if k == 0:  # First time step
                        # Compute base case for delta: Shape (d_{h_l}, d_{h_{l-1}}, B)
                        activation_derivative = self.activation_func_derivative(self.z[l][k])  # Shape: (d_{h_l}, B)
                        h_k_lm1 = self.hidden_states[l-1][k]  # Shape: (d_{h_{l-1}}, B)

                        # Compute outer product for base case
                        delta = activation_derivative[:, np.newaxis, :] * h_k_lm1[np.newaxis, :, :]  # Shape: (d_{h_l}, d_{h_{l-1}}, B)
                        delta_hx[-1].append(delta)
                    else:  # Subsequent time steps
                        # Activation derivative: Shape (d_{h_l}, B)
                        activation_derivative = self.activation_func_derivative(self.z[l][k])

                        # Compute contribution from previous delta term
                        last_term = np.einsum('ij,jkb->ikb', self.W_hh[l], delta_hx[-1][-1])  # Shape: (d_{h_l}, d_{h_{l-1}}, B)

                        # Add new contribution from hidden states
                        h_k_lm1 = self.hidden_states[l-1][k] # Shape: (d_{h_{l-1}}, B)
                        combined_term = h_k_lm1[np.newaxis, :, :] + last_term  # Shape: (d_{h_l}, d_{h_{l-1}}, B)

                        # Apply activation derivative
                        delta = activation_derivative[:, np.newaxis, :] * combined_term  # Shape: (d_{h_l}, d_{h_{l-1}}, B)
                        delta_hx[-1].append(delta)

            ### Output layer gradient:
            h_prev = self.hidden_states[-2][k].T  # Shape: (batch_size, hidden_size)

            d_activation_out = self.activation_func_out_derivative(self.z[-1][k].T)  # Shape: (batch_size, output_size)
            dL_dz_y = dL_dy_pred[:, k, :] * d_activation_out  # Shape: (batch_size, output_size)

            # Accumulate gradients for W_yh and b_y
            dL_dW_yh += dL_dz_y.T @ h_prev  # (output_size, batch_size) @ (batch_size, hidden_size) -> (output_size, hidden_size)
            dL_db_y += np.sum(dL_dz_y, axis=0, keepdims=True).T  # (output_size, 1)


        # Accumulate gradients for hidden weights and biases
        for l in range(self.L-1):
            for n in range(window_size):
                for k in range(n + 1):
                    # Compute the product term: product from j=k+1 to n of sigma'_h(z_j)*W_hh
                    if k == n:
                        prod_term = np.eye(self.hidden_states[l][k].shape[0], batch_size)
                    else:
                        prod_term = np.eye(self.hidden_states[l][k].shape[0], batch_size)
                        for j in range(k + 1, n + 1):
                            sigma_prime = self.activation_func_derivative(self.z[l][j]).T  # Shape: (batch_size, hidden_size)
                            prod_term *= (sigma_prime @ self.W_hh[l]).T
                    
                    dL_dW_hh[l] += np.einsum("ik,ijk->ij", dL_dh_n[l][k].T * prod_term, delta_hh[l][k])
                    dL_dW_hx[l] += np.einsum("ik,ijk->ij", dL_dh_n[l][k].T * prod_term, delta_hx[l][k])

                    tmp = dL_dh_n[l][k] @ (prod_term @ self.activation_func_derivative(self.z[l][k]).T)  
                    # Sum over the batch dimension (axis=0)
                    tmp = np.sum(tmp, axis=0, keepdims=True).T  # This turns (256,7) -> (1,7) -> (7,1) after transpose

                    dL_db_h[l] += tmp

        ### Clipping:
        dL_dW_yh = self._clip_gradient(dL_dW_yh, self.clip_value)
        dL_db_y = self._clip_gradient(dL_db_y, self.clip_value)

        ### Updating weights and biases 
        for l in range(self.L-1):
            dL_dW_hx[l] = self._clip_gradient(dL_dW_hx[l], self.clip_value)
            dL_dW_hh[l] = self._clip_gradient(dL_dW_hh[l], self.clip_value)
            dL_db_h[l] = self._clip_gradient(dL_db_h[l], self.clip_value)

            self.W_hx[l] = self.optimizer_W_hx[l](self.W_hx[l], dL_dW_hx[l], epoch, batch_index)
            self.W_hh[l] = self.optimizer_W_hh[l](self.W_hh[l], dL_dW_hh[l], epoch, batch_index)
            self.b_h[l] = self.optimizer_b_hh[l](self.b_h[l], dL_db_h[l], epoch, batch_index)

        self.W_yh = self.optimizer_W_yh(self.W_yh, dL_dW_yh, epoch, batch_index)
        self.b_y = self.optimizer_b_y(self.b_y, dL_db_y, epoch, batch_index)


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