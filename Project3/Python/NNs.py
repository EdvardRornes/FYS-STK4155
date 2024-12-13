import os
# Removes some print out details from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

# I get warnings which dont do anything, thus type: ignore on these
import tensorflow.keras as ker # type: ignore
from keras.models import Sequential, load_model # type: ignore
from keras.optimizers import Adam, SGD, RMSprop # type: ignore
from keras.layers import SimpleRNN, Dense, Input # type: ignore
from keras.callbacks import ModelCheckpoint # type: ignore
from keras.regularizers import l2 # type: ignore

import time 
import pickle
import copy
from utils import Activation, Scalers, Optimizer


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
        
        if scaler.upper() in ["NO SCALING", "NONE", "NO_SCALING"]:
            
            def tmp(X_train, X_test):
                return X_train, X_test 
            self._scaler = tmp 
        else: 
            self._scaler = Scalers(scaler)
    
    def save_data(self, filename:str, overwrite=False, stop=50) -> None:
        """Saves the analysis data to a file.

        Args:
            * filename:         name of the file to save the data to
            * overwrite         whether to overwrite the file if it exists
            * stop:             if 50 files for this 'type' of data exists, stop
        """
        filename_full = f"{filename}_{0}.pkl"

        if not overwrite:
            if os.path.exists(filename_full):
                i = 1
                while i <= stop: # May already be additional files created
                    filename_full = f"{filename}_{i}.pkl"
                    if not os.path.exists(filename_full):
                        with open(filename_full, 'wb') as f:
                            pickle.dump(self.data, f)
                        
                        return 
                    i += 1
            
                raise ValueError(f"You have {stop} additional files of this sort?")
                
        with open(filename_full, 'wb') as f:
            pickle.dump(self.data, f)

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

    def store_train_test_from_data(self, X:np.ndarray, y:np.ndarray, split_data=True) -> None:
        """
        Converts data into a training and a testing set. Applies the scaling assicated with the instance of the class.

        Arguments:
        * X:                input time series data of shape (num_samples, input_size)
        * y:                output labels of shape (num_samples, output_size)        
        """

        if split_data:
            if self.random_state is None:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_percentage)
            else: 
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_percentage, random_state=self.random_state)
        
        else:
            X_train = X; X_test = X
            y_train = y; y_test = y 

        self.y_train = y_train; self.y_test = y_test
        
        self.X_train = X_train; self.X_test = X_test
        self.X_train_scaled, self.X_test_scaled = self._scaler(X_train, X_test)

    def prepare_sequences_RNN(self, X:np.ndarray, y:np.ndarray, step_length:int, input_size:int, overlap:float=50):
        """
        Converts data into sequences for RNN training.
        
        Parameters:
        * X:                scaled data 
        * y:                output data.
        * step_length:      length of each sequence.
        * input_size:       number of features in input data.
        * overlap:          overlap percentage between sequences (0 to 100).
        
        Returns:
        * X_seq, y_seq:     sequences (3D array) and corresponding labels (1D array).
        """
        if not (0 <= overlap < 100):
            raise ValueError("Overlap percentage must be between 0 and 100 (exclusive).")

        step_size = max(1, int(step_length * (1 - overlap / 100)))  # Calculate step size based on overlap percentage
        sequences = []
        labels = []

        for i in range(0, len(X) - step_length + 1, step_size):
            seq = X[i:i + step_length]
            label_seq = y[i:i + step_length]  # Create a sequence of labels

            sequences.append(seq)
            labels.append(label_seq)

        X_seq, y_seq = np.array(sequences).reshape(-1, step_length, input_size), np.array(labels)

        return X_seq, y_seq


    # def prepare_sequences_RNN(self, X:np.ndarray, y:np.ndarray, step_length:int, input_size:int, overlap_percentage:float=0.9):
    #     """
    #     Converts data into sequences for RNN training with configurable overlap.
        
    #     Parameters:
    #     * X: np.ndarray
    #         Scaled input data (2D array, shape: [time_steps, features]).
    #     * y: np.ndarray
    #         Target data (1D array, shape: [time_steps]).
    #     * step_length: int
    #         Length of each sequence.
    #     * input_size: int
    #         Number of features in the input data.
    #     * overlap_percentage: float
    #         Percentage of overlap between consecutive sequences (0.0 to 1.0).
        
    #     Returns:
    #     * X_seq: np.ndarray
    #         3D array of sequences, shape: (num_sequences, step_length, input_size).
    #     * y_seq: np.ndarray
    #         1D array of corresponding labels, shape: (num_sequences,).
    #     """
    #     if not (0 <= overlap_percentage <= 1):
    #         raise ValueError("overlap_percentage must be between 0.0 and 1.0")
        
    #     # Calculate step size based on the overlap percentage
    #     step_size = int(step_length * (1 - overlap_percentage))
    #     if step_size <= 0:
    #         raise ValueError("The overlap percentage is too high, resulting in a non-positive step size.")
        
    #     sequences = []
    #     labels = []

    #     # Generate sequences with the specified overlap
    #     for i in range(0, len(X) - step_length + 1, step_size):
    #         seq = X[i:i + step_length]
    #         sequences.append(seq)
    #         label = y[i + step_length - 1]
    #         labels.append(label)

    #     X_seq = np.array(sequences).reshape(-1, step_length, input_size)
    #     y_seq = np.array(labels)

    #     return X_seq, y_seq

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
                y_pred[:, :] = 1 * (y_pred[:, :] <= 0.5)

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

        return np.array(self.hidden_states[-1])


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



class KerasRNN:
    def __init__(self, hidden_layers: list, dim_output: int, dim_input: int,  
                 loss_function="binary_crossentropy", optimizer="adam", labels=None, 
                 gw_class_early_boost=1, learning_rate=1e-2, l2_regularization=0.0, activation_func='tanh', grad_clip=2):
        """
        Initializes an RNN model for multi-class classification.

        Parameters:
        - hidden_layers (list): List of integers specifying the number of units in each hidden layer.
        - dim_output (int): Number of output classes.
        - dim_input (int): Dimension of the input data.
        - loss_function (str): Loss function for training the model (default is 'binary_crossentropy').
        - optimizer (str): Optimizer for training the model (default is 'adam').
        - labels (list or None): Labels used for dynamic class weight computation.
        - gw_class_early_boost (float): Boost factor for the gravitational wave class during early epochs (default is 1).
        - learning_rate (float): Learning rate for the optimizer (default is 1e-2).
        - l2_regularization (float): L2 regularization parameter for the layers (default is 0.0).
        - activation_func (str): Activation function used in RNN layers (default is 'tanh').
        - grad_clip (float): Gradient clipping norm value to prevent exploding gradients (default is 2).
        """
        self.hidden_layers = hidden_layers
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.gw_class_early_boost = gw_class_early_boost
        self.labels = labels
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.activation_func = activation_func
        self.grad_clip = grad_clip

        # Initialize the RNN model
        self.model = self.create_model()  # Call create_model during initialization to set up the model

    def create_model(self):
        """
        Creates and returns a fresh RNN model with the specified configurations.

        Returns:
        - model: A compiled Keras RNN model with the defined layers, loss function, and optimizer.
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

        Parameters:
        - model: The Keras model to compile.
        """
        optimizers = {
            "adam": Adam(learning_rate=self.learning_rate, clipnorm=self.grad_clip),
            "sgd": SGD(learning_rate=self.learning_rate, clipnorm=self.grad_clip),
            "rmsprop": RMSprop(learning_rate=self.learning_rate, clipnorm=self.grad_clip)
        }

        if self.optimizer not in optimizers:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}. Choose from {list(optimizers.keys())}.")

        model.compile(
            loss=self.loss_function, 
            optimizer=optimizers[self.optimizer], 
            metrics=['accuracy']
        )

    def prepare_sequences_RNN(self, X: np.ndarray, y: np.ndarray, step_length: int, overlap=0.9):
        """
        Converts data into sequences for RNN training with control over the non-overlapping part of sequences.
        
        Parameters:
        - X: Input data as a numpy array.
        - y: Target data as a numpy array.
        - step_length: Length of each sequence.
        - overlap: Fractional overlap between consecutive sequences (default is 0.9, meaning 90% overlap).
        
        Returns:
        - X_seq: Sequences of input data.
        - y_seq: Corresponding target data.
        """
        # Calculate the step size based on overlap
        step_size = int(step_length * overlap)
        # Calculate the total number of sequences we can generate
        n_samples = (len(X) - step_length) // step_size + 1

        # Create sequences with the specified overlap
        X_seq = np.array([X[i:i + step_length] for i in range(0, n_samples * step_size, step_size)]).reshape(-1, step_length, 1)
        y_seq = y[step_length - 1:step_length - 1 + len(X_seq)]

        return X_seq, y_seq

    def compute_class_weights(self, epoch: int, total_epochs: int):
        """
        Compute class weights dynamically based on the label distribution.

        Parameters:
        - epoch (int): The current epoch number.
        - total_epochs (int): The total number of training epochs.
        
        Returns:
        - dict: A dictionary with the computed class weights for each class.
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
        Train the RNN model using dynamically computed class weights, stopping early if no improvement occurs for 30% of the epochs.
        Continues training if the loss is sufficiently low, even if no improvement is observed.
        
        Parameters:
        - X_train (np.ndarray): Training input data.
        - y_train (np.ndarray): Training target data.
        - epochs (int): Number of training epochs.
        - batch_size (int): Size of each training batch.
        - step_length (int): Length of the input sequences.
        - verbose (int): Verbosity level (0 = silent, 1 = progress bar, 2 = one line per epoch).
        - verbose1 (int): Additional verbosity control for printing epoch updates (default is 1).
        """
        # Split training data into a validation set
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Prepare sequences for RNN input
        X_train_seq, y_train_seq = self.prepare_sequences_RNN(X_train, y_train, step_length)
        X_val_seq, y_val_seq = self.prepare_sequences_RNN(X_val, y_val, step_length)

        # Reinitialize model (this ensures no previous weights are carried over between parameter runs)
        self.model = self.create_model()

        # Initialize variables to track the best model and validation loss
        best_val_loss = float('inf')
        best_weights = None  # Keep track of the best model's weights, not the entire model

        # Define thresholds
        patience_threshold = int(np.ceil(0.5 * epochs))  # Early stopping threshold
        epochs_without_improvement = 0
        low_loss_threshold = 0.2  # Continue training even without improvement if loss is below this value

        # Compute class weights dynamically based on training labels
        for epoch in range(epochs):
            class_weights = self.compute_class_weights(epoch, epochs)

            # Fit the model for one epoch and save the history
            history = self.model.fit(
                X_train_seq, y_train_seq,
                epochs=1,  # 1 epoch due to dynamic class weight, still doing "epoch" epochs due to loop
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
                if verbose1 == 1:
                    print(f"Epoch {epoch + 1} - val_loss improved from {best_val_loss:.3f} to {current_val_loss:.3f}. Best model updated.")
                best_val_loss = current_val_loss
                epochs_without_improvement = 0  # Reset counter
            else:
                if verbose1 == 1:
                    print(f"Epoch {epoch + 1} - val_loss did not improve ({current_val_loss:.3f} >= {best_val_loss:.3f}).")
                epochs_without_improvement += 1

            # Check early stopping conditions
            if epochs_without_improvement >= patience_threshold:
                if best_val_loss >= low_loss_threshold:
                    if verbose1 == 1 and epoch != epochs-1:
                        print(f"No improvement for {patience_threshold} consecutive epochs and val_loss >= {low_loss_threshold}. Stopping early at epoch {epoch + 1}.")
                    break
                else:
                    if verbose1 == 1 and epoch != epochs-1:
                        print(f"No improvement for {patience_threshold} consecutive epochs, but val_loss < {low_loss_threshold}. Continuing training.")

        # After training, restore the best model weights (so model doesn't carry over worse performance)
        if best_weights is not None:
            self.model.set_weights(best_weights)  # Set the model's weights to the best found during training

    def predict(self, X_test, y_test, step_length, verbose=1):
        """
        Predicts labels for the test set using the trained RNN model.
        
        Parameters:
        - X_test: Input data for prediction.
        - y_test: True labels for the test data.
        - step_length: Length of the input sequences.
        - verbose: Verbosity level (default is 1, progress bar style).
        
        Returns:
        - y_pred: Predicted labels for the test data.
        """
        X_test_seq, _ = self.prepare_sequences_RNN(X_test, y_test, step_length)
        y_pred = self.model.predict(X_test_seq, verbose=verbose)
        return y_pred


class Loss:
    data = {}

    def __call__(self, y_true, y_pred, epoch=None):
        """
        Computes the loss value.

        Arguments:
        - y_true: True labels.
        - y_pred: Predicted outputs.

        Returns:
        - Loss value.
        """
        raise NotImplementedError("Forward method not implemented.")

    def gradient(self, y_true, y_pred, epoch=None):
        """
        Computes the gradient of the loss with respect to y_pred.

        Arguments:
        - y_true: True labels.
        - y_pred: Predicted outputs.

        Returns:
        - Gradient of the loss with respect to y_pred.
        """
        raise NotImplementedError("Backward method not implemented.")


class DynamicallyWeightedLoss(Loss):

    def __init__(self, initial_boost=1, epochs=None, labels=None, weight_0=1, epsilon=1e-8):
        self.initial_boost = 1; self.epochs = epochs
        self.labels = labels; self.weight_0 = weight_0
        self._epsilon = epsilon 

        self.data = {"initial_boost": self.initial_boost,
                  "epochs": epochs,
                  "weight_0": weight_0}
        
    def __call__(self, y_true, y_pred, epoch):
        scale = self.initial_boost - (self.initial_boost-1)*epoch/self.epochs
        weight_1 = (len(self.labels)-np.sum(self.labels))/np.sum(self.labels)*scale
        
        # y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # To prevent log(0)
        loss = -np.mean(weight_1 * y_true * np.log(y_pred) + self.weight_0 * (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def gradient(self, y_true, y_pred, epoch):
        scale = self.initial_boost - (self.initial_boost-1)*epoch/self.epochs
        weight_1 = (len(self.labels)-np.sum(self.labels))/np.sum(self.labels)*scale

        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # To prevent division by zero
        grad = - (weight_1 * y_true / (y_pred + self._epsilon)) + (self.weight_0 * (1 - y_true) / (1 - y_pred + self._epsilon))
        return grad

class WeightedBinaryCrossEntropyLoss(Loss):
    def __init__(self, weight_0, weight_1):
        """
        Initializes the loss function with class weights.

        Arguments:
        - class_weight: Dictionary with weights for each class {0: weight_0, 1: weight_1}.
        """
        self.weight_1 = weight_1
        self.weight_0 = weight_0

        self.data = {"weight_0": weight_0,
                     "weight_1": weight_1}

    def __call__(self, y_true, y_pred, epoch=None):
        """
        Computes the weighted binary cross-entropy loss.
        """
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # To prevent log(0)
        loss = -np.mean(
            self.weight_1 * y_true * np.log(y_pred) +
            self.weight_0 * (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    def gradient(self, y_true, y_pred, epoch=None):
        """
        Computes the gradient of the loss with respect to y_pred.
        """
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # To prevent division by zero
        grad = - (self.weight_1 * y_true / y_pred) + (self.weight_0 * (1 - y_true) / (1 - y_pred))
        return grad

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

        self.data = {"alpha": alpha,
                     "gamma": gamma}


    def __call__(self, y_true, y_pred, epoch=None):
        """
        Computes the focal loss.
        """
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        loss = -np.mean(alpha_t * (1 - pt) ** self.gamma * np.log(pt))
        return loss

    def gradient(self, y_true, y_pred, epoch=None):
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