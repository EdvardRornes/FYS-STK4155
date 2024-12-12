
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
        L = self.L  # number of layers including output
        # L-1 = number of hidden layers, the L-th "layer" is the output layer.

        # Extract top layer sizes
        d_y = self.output_size
        # Layers: input -> h1 -> h2 -> ... -> h_{L-1} -> output
        # self.layers = [input_size, hidden_1, hidden_2, ..., output_size]

        # Compute dL/dy_pred
        # shape: (batch, time, d_y)
        # According to the theory, let's assume we only consider the final time step N=window_size for the output gradients
        dL_dy_pred = self._loss_function.gradient(y_batch, y_pred)  # (B, T, d_y)

        # Focus on final time step for output gradients (theory scenario)
        N = window_size

        # Compute dL/dz_out_N = dL/dy_N * sigma'_out(z_N)
        # z_N for output layer is self.z[-1][N-1] shape: (d_y, B)
        sigma_prime_out = self.activation_func_out_derivative(self.z[-1][N-1])  # (d_y, B)
        dL_dz_out_N = (dL_dy_pred[:, N-1, :].T * sigma_prime_out)  # (d_y, B)

        # Grad for output layer weights and biases (W_yh, b_y)
        # According to theory:
        # dL/dW_yh = dL/dz_out_N * h_{N}^{(L-1)}^T
        # dL/db_y = sum over batch of dL/dz_out_N
        h_N_top = self.hidden_states[-2][N-1]  # (d_{h_{L-1}}, B)
        dL_dW_yh = dL_dz_out_N @ h_N_top.T
        dL_db_y = np.sum(dL_dz_out_N, axis=1, keepdims=True)

        # Backprop through output layer to get dL/dh_N^{(L-1)}
        dL_dh_N = self.W_yh.T @ dL_dz_out_N  # (d_{h_{L-1}}, B)

        # Now, we handle the hidden layers using the theory:
        # For each hidden layer l (indexed from 0 to L-2), we have:
        # dL/dW_{hh}^{(l)}, dL/dW_{hx}^{(l)}, dL/db_h^{(l)}
        #
        # According to your theory:
        # δ_{hh}^1 = 0
        # δ_{hh}^k = diag(sigma'_h(z_k^{(l)})) [h_{k-1}^{(l)} + W_{hh}^{(l)} δ_{hh}^{k-1}]
        # δ_{hx}^k = diag(sigma'_h(z_k^{(l)})) [x_k^{(l)} + W_{hh}^{(l)} δ_{hx}^{k-1}]
        #
        # Similarly for gradients, we use the double summation and product terms.

        # Initialize gradients for hidden layers
        dL_dW_hx = [np.zeros_like(w) for w in self.W_hx]
        dL_dW_hh = [np.zeros_like(w) for w in self.W_hh]
        dL_db_h = [np.zeros_like(b) for b in self.b_h]

        # We must first compute dL/dh_n^{(l)} for all time steps n and for all layers l.
        # Start from top hidden layer: we have dL/dh_N^{(L-1)} already from output layer backprop.
        # Propagate backwards in time:
        # dL/dh_n^{(l)} = W_{hh}^{(l)T} [ (dL/dh_{n+1}^{(l)}) * sigma'_h(z_{n+1}^{(l)}) ] + (from upper layers if stacked)
        #
        # If stacked: we also need to add contributions from layer above via W_hx of that layer.
        # According to the original theory snippet, it focuses on a single-layer scenario. For multiple layers,
        # you must incorporate top-down errors. Let's follow a layer-by-layer approach:
        #
        # Step 1: Assume we handle the topmost hidden layer first (l = L-2), then proceed downward.

        # Let's store dL/dh_t^{(l)} for each layer l and time t
        # Initialize all zeros
        dL_dh_all = [ [np.zeros((self.layers[l+1], batch_size)) for _ in range(N)] for l in range(L-1) ]
        # The topmost hidden layer gets dL/dh_N^{(L-1)} from output:
        dL_dh_all[L-2][N-1] = dL_dh_N

        # Backprop through time for the top layer:
        for t in reversed(range(N-1)):
            l = L-2
            sigma_prime_h = self.activation_func_derivative(self.z[l][t])  # (d_h_l, B)
            dL_dh_next = dL_dh_all[l][t+1]
            # dL/dh_t^{(l)} = W_hh[l]^T (dL/dh_{t+1}^{(l)} * sigma'_h(z_{t+1}^{(l)}))
            # Wait, we need sigma'_h(z_{t+1}), but we have z[l][t], careful:
            sigma_prime_h_next = self.activation_func_derivative(self.z[l][t+1])
            dL_dh_all[l][t] = self.W_hh[l].T @ (dL_dh_next * sigma_prime_h_next)

        # If there are multiple hidden layers below, we must now propagate the errors downward in the network.
        # For a fully correct multi-layer scenario, you'd propagate dL/dh_t^{(l)} to layer (l-1) using W_hx[l].
        # But your theory did not explicitly cover stacked RNN. Let's assume a single hidden layer for now, 
        # as the theory primarily addresses that scenario. If you must handle multiple layers, you'd need to:
        # dL/dh_t^{(l-1)} += W_hx[l]^T (dL/dh_t^{(l)}) * sigma'_h(z_{t}^{(l-1)})
        # and then repeat the backward pass in time for each lower layer.
        #
        # For brevity, here we proceed as if there's only one hidden layer. If multiple layers exist, 
        # you would repeat the process above for each layer in descending order.

        # Now that we have dL/dh_n for all n, we can compute δ_{hh}^k and δ_{hx}^k and the gradients.
        # Let's assume a single hidden layer l=0 for demonstration. If multiple layers, wrap in a loop over layers.
        if L > 1:
            l = 0  # the first (and possibly only) hidden layer
            d_h_l = self.layers[l+1]
            d_x = self.layers[l]
            W_hh_l = self.W_hh[l]
            W_hx_l = self.W_hx[l]

            # Compute δ_{hh}^k and δ_{hx}^k
            delta_hh = [np.zeros((d_h_l, d_h_l)) for _ in range(N+1)]
            delta_hx = [np.zeros((d_h_l, d_x)) for _ in range(N+1)]

            # Base cases
            # δ_{hh}^1 = 0 by definition
            delta_hh[1] = np.zeros((d_h_l, d_h_l))

            # δ_{hx}^1 = σ'_h(z_1)(x_1)
            # Here we must treat σ'_h(z_1) as a diagonal matrix and x_1 as a (d_x,)
            # We'll handle batches by summation. The theory given is more aligned with a single sample.
            # For simplicity, let's just consider the average over the batch (or handle one sample at a time).
            # Ideally, you'd loop over batch or average. We'll do an average over batch for demonstration.
            x_1 = np.mean(X_batch[:,0,:], axis=0)  # average over batch: shape (d_x,)
            sigma_prime_1 = np.diag(np.mean(self.activation_func_derivative(self.z[l][0]), axis=1))
            delta_hx[1] = sigma_prime_1 @ x_1.reshape(d_x,1)  # (d_h_l, d_x)
            delta_hx[1] = delta_hx[1].reshape(d_h_l, d_x)

            # For k=2,...,N
            for k in range(2, N+1):
                # sigma'_h(z_k)
                sigma_prime_k = np.diag(np.mean(self.activation_func_derivative(self.z[l][k-1]), axis=1))
                h_km1 = np.mean(self.hidden_states[l][k-2], axis=1) if k > 1 else np.zeros(d_h_l)
                x_k = np.mean(X_batch[:, k-1, :], axis=0)

                delta_hh[k] = sigma_prime_k @ (np.outer(h_km1, np.ones(d_h_l)).T + W_hh_l @ delta_hh[k-1])
                delta_hx[k] = sigma_prime_k @ (np.outer(x_k, np.ones(d_x)).T + W_hh_l @ delta_hx[k-1])

            # Now compute gradients using the big summation:
            # dL/dW_hh = ∑_{n=1}^N ∑_{k=1}^n (dL/dh_n) [∏_{j=k+1}^n σ'_h(z_j)W_hh] δ_{hh}^k
            # Similarly for dL/dW_hx and dL/db_h
            #
            # We'll average over the batch dimension as well. dL/dh_n is (d_h_l, B), take mean over batch:
            dL_dh_mean = [np.mean(dL_dh_all[l][n], axis=1) for n in range(N)]  # each in R^{d_h_l}

            for n_idx in range(1, N+1):
                for k_idx in range(1, n_idx+1):
                    # product_term = ∏_{j=k+1}^{n} σ'_h(z_j)W_hh
                    product_term = np.eye(d_h_l)
                    for j_idx in range(k_idx+1, n_idx+1):
                        sigma_j = np.diag(np.mean(self.activation_func_derivative(self.z[l][j_idx-1]), axis=1))
                        product_term = product_term @ (sigma_j @ W_hh_l)

                    dL_dh_n = dL_dh_mean[n_idx-1].reshape(1, d_h_l)  # (1, d_h_l)
                    # δ_{hh}^k and δ_{hx}^k are (d_h_l, d_h_l) and (d_h_l, d_x) respectively
                    # We'll assume δ_{hh}^k, δ_{hx}^k computed from averaged states.

                    # dL/dW_hh += dL/dh_n * product_term * δ_{hh}^k
                    # Align dimensions: (1,d_h_l) * (d_h_l,d_h_l) * (d_h_l,d_h_l) = (d_h_l, d_h_l)
                    dL_dW_hh[l] += dL_dh_n @ (product_term @ delta_hh[k_idx])

                    # dL/dW_hx += dL/dh_n * product_term * δ_{hx}^k
                    # (1,d_h_l)*(d_h_l,d_h_l)*(d_h_l,d_x) = (d_h_l,d_x)
                    dL_dW_hx[l] += dL_dh_n @ (product_term @ delta_hx[k_idx])

                    # dL/db_h += dL/dh_n * product_term * σ'_h(z_k)
                    sigma_k = np.diag(np.mean(self.activation_func_derivative(self.z[l][k_idx-1]), axis=1))
                    # (1,d_h_l)*(d_h_l,d_h_l)*(d_h_l,d_h_l (from sigma_k? Actually sigma_k is (d_h_l,d_h_l))
                    # Wait the formula states: ... * σ'_h(z_k)
                    # σ'_h(z_k) is (d_h_l,) so use it as diag or vector:
                    sigma_k_vec = np.mean(self.activation_func_derivative(self.z[l][k_idx-1]), axis=1)
                    # product_term*(σ'_h(z_k)) -> (d_h_l,d_h_l)*(d_h_l,d_h_l)? Actually σ'_h(z_k) is a diagonalizable vector.
                    # According to formula: just multiply by σ'_h(z_k). It's elementwise:
                    # Let's apply elementwise multiplication on the last step:
                    temp_vec = (product_term @ sigma_k)  # (d_h_l, d_h_l)
                    # Summation of that doesn't quite match theory dimensionally. The theory might assume vectorization differently.
                    # We'll interpret final step as a vector multiplication:
                    # Actually, the formula has σ'_h(z_k) alone (no matrix form), so we should multiply (dL/dh_n * product_term) by σ'_h(z_k) elementwise:
                    # (dL/dh_n * product_term) gives (1,d_h_l), apply σ'_h(z_k) elementwise:
                    dL_db_h[l] += ((dL_dh_n @ product_term) * sigma_k_vec).T

        # Gradient clipping and parameter updates:
        clip_value = 5.0
        dL_dW_yh = self._clip_gradient(dL_dW_yh, clip_value)
        dL_db_y = self._clip_gradient(dL_db_y, clip_value)

        # Update hidden layer parameters (if any)
        for l in range(self.L-1):
            dL_dW_hx[l] = self._clip_gradient(dL_dW_hx[l], clip_value)
            dL_dW_hh[l] = self._clip_gradient(dL_dW_hh[l], clip_value)
            dL_db_h[l] = self._clip_gradient(dL_db_h[l], clip_value)

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