
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
        best_weights_recurrent = copy.deepcopy(self.W_hh)
        best_W_yh = copy.deepcopy(self.W_yh)
        best_b_hh = copy.deepcopy(self.b_h)
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
        """
        batch_size, window_size, _ = X_batch.shape

        # Initialize hidden states and pre-activations
        self.hidden_states = [[np.zeros((self.layers[l + 1], batch_size)) for _ in range(window_size)] for l in range(self.L)]
        self.z = [[np.zeros((self.layers[l + 1], batch_size)) for _ in range(window_size)] for l in range(self.L)]
        y_preds = []

        for t in range(window_size):
            x_t = X_batch[:, t, :]  # Shape: (batch_size, input_size)
            for l in range(self.L):
                if t == 0:
                    h_prev = np.zeros((self.layers[l + 1], batch_size))
                else:
                    h_prev = self.hidden_states[l][t - 1]
                z_l_t = self.W_hx[l] @ x_t.T + self.W_hh[l] @ h_prev + self.b_h[l]
                self.z[l][t] = z_l_t
                self.hidden_states[l][t] = self.activation_func(z_l_t)
                x_t = self.hidden_states[l][t].T  # Output to next layer

            # Output layer
            y_pred_t = self.W_yh @ self.hidden_states[-1][t] + self.b_y  # Shape: (output_size, batch_size)
            y_pred_t = self.activation_func_out(y_pred_t)  # Apply output activation
            y_preds.append(y_pred_t.T)  # Shape: (batch_size, output_size)

        y_pred = np.stack(y_preds, axis=1)  # Shape: (batch_size, window_size, output_size)
        return y_pred



    def _backward(self, X_batch: np.ndarray, y_batch: np.ndarray, y_pred: np.ndarray, epoch: int, batch_index: int):
        """
        Backward pass for the RNN using Backpropagation Through Time (BPTT).
        """
        batch_size, window_size, _ = X_batch.shape

        # Initialize gradients
        dL_dW_yh = np.zeros_like(self.W_yh)    # Shape: (output_size, hidden_size)
        dL_db_y = np.zeros_like(self.b_y)      # Shape: (output_size, 1)

        dL_dW_hx = [np.zeros_like(w) for w in self.W_hx]  # List of arrays, one per hidden layer
        dL_dW_hh = [np.zeros_like(w) for w in self.W_hh]
        dL_db_h = [np.zeros_like(b) for b in self.b_h]

        # Compute the derivative of the loss with respect to the output (dL/dy_pred)
        # Assuming the loss function's gradient returns shape (batch_size, window_size, output_size)
        dL_dy_pred = self._loss_function.gradient(y_batch, y_pred)  # Shape: (batch_size, window_size, output_size)

        # Iterate over each time step to compute gradients for the output layer
        for t in range(window_size):
            # Compute dL/dz_y at time step t
            # dL_dy_pred[:, t, :] has shape (batch_size, output_size)
            # self.z[-1][t] has shape (output_size, batch_size)
            # Compute derivative of activation function at output layer
            d_activation_out = self.activation_func_out_derivative(self.z[-1][t])  # Shape: (output_size, batch_size)
            d_activation_out = d_activation_out.T  # Shape: (batch_size, output_size)

            # Element-wise multiplication
            dL_dz_y = dL_dy_pred[:, t, :] * d_activation_out  # Shape: (batch_size, output_size)

            # Accumulate gradients for W_yh and b_y
            h_prev = self.hidden_states[-1][t]  # Shape: (hidden_size, batch_size)
            # Compute dL_dW_yh: (output_size, hidden_size)
            dL_dW_yh += dL_dz_y.T @ h_prev.T  # (output_size, batch_size) @ (batch_size, hidden_size) -> (output_size, hidden_size)
            # Compute dL_db_y: (output_size, 1)
            dL_db_y += np.sum(dL_dz_y, axis=0, keepdims=True).T  # (output_size,1)

        # Initialize delta for hidden layers
        delta_h = [np.zeros((self.layers[l + 1], batch_size)) for l in range(self.L)]

        # Backpropagate through time
        for t in reversed(range(window_size)):
            # Compute delta for the output layer
            dL_dz_y_t = dL_dy_pred[:, t, :] * self.activation_func_out_derivative(self.z[-1][t]).T  # Shape: (batch_size, output_size)
            # delta_output = W_yh.T @ dL_dz_y_t.T  # (hidden_size, output_size) @ (output_size, batch_size) = (hidden_size, batch_size)
            delta_output = self.W_yh.T @ dL_dz_y_t.T  # Shape: (hidden_size, batch_size)

            # Accumulate delta for the last hidden layer
            delta_h[-1] += delta_output  # Shape: (hidden_size, batch_size)

            # Iterate through each hidden layer
            for l in reversed(range(self.L)):
                # Compute delta for current layer
                z_l_t = self.z[l][t]  # Shape: (hidden_size, batch_size)
                sigma_prime = self.activation_func_derivative(z_l_t)  # Shape: (hidden_size, batch_size)
                # Element-wise multiplication
                delta_h[l] *= sigma_prime  # Shape: (hidden_size, batch_size)

                # Compute gradients for W_hh, W_hx, and b_h
                if t > 0:
                    h_prev = self.hidden_states[l][t - 1]  # Shape: (hidden_size, batch_size)
                else:
                    h_prev = np.zeros((self.layers[l + 1], batch_size))  # Initial hidden state

                # Gradient w.r.t W_hh: (hidden_size, batch_size) @ (hidden_size, batch_size).T = (hidden_size, hidden_size)
                dL_dW_hh[l] += delta_h[l] @ h_prev.T  # (hidden_size, batch_size) @ (batch_size, hidden_size)

                # Gradient w.r.t W_hx: (hidden_size, batch_size) @ (input_size, batch_size).T = (hidden_size, input_size)
                x_t = X_batch[:, t, :].T  # Shape: (input_size, batch_size)
                dL_dW_hx[l] += delta_h[l] @ x_t.T  # (hidden_size, batch_size) @ (batch_size, input_size)

                # Gradient w.r.t b_h: sum over batch
                dL_db_h[l] += np.sum(delta_h[l], axis=1, keepdims=True)  # Shape: (hidden_size,1)

                # Propagate delta to the previous hidden layer
                delta_h[l] = self.W_hh[l].T @ delta_h[l]  # Shape: (hidden_size, hidden_size) @ (hidden_size, batch_size) = (hidden_size, batch_size)

        # Optional: Gradient Clipping to prevent exploding gradients
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

        for i in range(self.L):
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

        # for l in range(len(self.W_hh)):
        #     print(f"W_hx: {np.shape(self.W_hx[l])}, W_hh: {np.shape(self.W_hh[l])}, b_h: {np.shape(self.b_h[l])}")
        
        # print(f"W_yh: {np.shape(self.W_yh)}, b_h: {np.shape(self.b_y)}")
        # exit()