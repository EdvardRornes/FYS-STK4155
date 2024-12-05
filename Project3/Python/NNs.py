import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

import os
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

    # def prepare_sequences_RNN(self, X:np.ndarray, y:np.ndarray, step_length:int, input_size:int):
    #     """
    #     Converts data into sequences for RNN training.
        
    #     Parameters:
    #     * X:                scaled data 
    #     * y:                output data.
    #     * step_length:      length of each sequence.
        
    #     Returns:
    #     * X_seq, y_seq:     sequences (3D array) and corresponding labels (1D array).
    #     """
    #     sequences = []
    #     labels = []
    #     # exit()
    #     for i in range(len(X) - step_length + 1):
    #         seq = X[i:i + step_length]
    #         label_seq = y[i:i + step_length]  # Create a sequence of labels

    #         sequences.append(seq)
    #         labels.append(label_seq)

    #     X_seq, y_seq = np.array(sequences).reshape(-1, step_length, input_size), np.array(labels)

    #     return X_seq, y_seq  

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
        self._initialize_weights_and_biases()

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
        X_seq, y_seq = self.prepare_sequences_RNN(self.X_train_scaled, self.y_train, window_size, self.input_size)
        X_seq_test, y_seq_test = self.prepare_sequences_RNN(self.X_test_scaled, self.y_test, window_size, self.input_size)

        self.truncation_steps = truncation_steps # Sent to backward propagation method 

        if isinstance(self._loss_function, DynamicallyWeightedLoss):
            self._loss_function.epochs = epochs
            self._loss_function.labels = y

        start_time = time.time(); best_loss = 1e4

        prev_weights = copy.deepcopy(self.weights)
        prev_weights_recurrent = copy.deepcopy(self.W_recurrent)
        
        test_loss = 1
        for epoch in range(epochs):
            
            self.weights = prev_weights
            self.W_recurrent = prev_weights_recurrent
            for i in range(0, X_seq.shape[0], batch_size):
                X_batch = X_seq[i:i + batch_size]
                y_batch = y_seq[i:i + batch_size]

                # Forward pass
                y_pred = self._forward(X_batch)

                # Backward pass
                self._backward(X_batch, y_batch.reshape(-1,1), y_pred, epoch, i)

                print(f"Epoch {epoch + 1}/{epochs} completed, loss: {test_loss:.3f}, time elapsed: {time.time()-start_time:.1f}s", end="\r")
            
            y_pred = self._forward(X_seq_test)
    
            test_loss = self.calculate_loss(y_seq_test, y_pred, epoch)

            if test_loss < best_loss:
                best_loss = test_loss

                prev_weights = copy.deepcopy(self.weights)
                prev_weights_recurrent = copy.deepcopy(self.W_recurrent)

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

    # def evaluate(self, X:np.ndarray, y:np.ndarray, epoch_index=1, window_size=10) -> tuple[float, float]:
    #     """
    #     Evaluate the model on a given dataset.
        
    #     Arguments:
    #     * X:        Input time series data of shape (num_samples, input_size)
    #     * y:        True labels of shape (num_samples, output_size)
        
    #     Keyword Arguments:
    #     * window_size: Size of the window for creating sequences (step length)
        
    #     Returns:
    #     * Loss value and accuracy score
    #     """
    #     self.store_train_test_from_data(X, y)

    #     # Prepare sequences for evaluation
    #     X_test_seq, y_test_seq = self.prepare_sequences_RNN(self.X_test_scaled, self.y_test, window_size)

    #     # Forward pass to get predictions
    #     y_pred = self._forward(X_test_seq)

    #     # Calculate loss
    #     loss = self.calculate_loss(y_test_seq, y_pred, epoch_index)

    #     # Calculate accuracy
    #     y_pred_classes = (y_pred > 0.5).astype(int)
    #     accuracy = accuracy_score(y_test_seq, y_pred_classes)

    #     print(f"Evaluation - Loss: {loss}, Accuracy: {accuracy}")
    #     return loss, accuracy
    
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
                    prev_activation @ self.W_input[j] + h_prev @ self.W_recurrent[j] + self.biases[j]
                )
                self.hidden_states[i][j] = h

        # Output layer uses the hidden state from the last time step
        output = self.hidden_states[-1][-1] @ self.W_output + self.b_output
        output = self.activation_func_out(output)
        return output

    def _backward(self, X, y, y_pred, epoch_index, batch_index):
        batch_size, window_size, _ = X.shape
        num_hidden_layers = len(self.layers) - 2  # Excludings

        # initializing total gradients
        total_dW_input = [np.zeros_like(w) for w in self.W_input]
        total_dW_recurrent = [np.zeros_like(w) for w in self.W_recurrent]
        total_db_hidden = [np.zeros_like(b) for b in self.biases]

        # used in computing error from previous time step
        delta_h_next = [np.zeros((batch_size, self.layers[i + 1])) for i in range(num_hidden_layers)]

        # gradients for output layer
        dW_output = np.zeros_like(self.W_output)
        db_output = np.zeros_like(self.b_output)

        # determine the number of truncation steps
        # truncation_steps = min(self.truncation_steps, window_size)
        # t_start = max(0, window_size - truncation_steps)

        # propagating backwards through time, restricted to a maximum amount dictated by t_start
        for t in reversed(range(window_size)):

            
            if t == window_size - 1:    # output lauer
                error = self._loss_function.gradient(y, y_pred, epoch_index) * self.activation_func_out_derivative(y_pred)
                dW_output += self.hidden_states[t][-1].T @ error / batch_size
                db_output += np.sum(error, axis=0, keepdims=True) / batch_size

                delta_output = error @ self.W_output.T  # Shape: (batch_size, hidden_size_last_layer)
            else:
                delta_output = np.zeros_like(self.hidden_states[t][-1])

            # seting up "error" list for each hidden layer
            delta_h_current = [np.zeros((batch_size, self.layers[i + 1])) for i in range(num_hidden_layers)]

            # backpropagating through hidden layers
            for l in reversed(range(num_hidden_layers)):
                h = self.hidden_states[t][l]
                h_prev = self.hidden_states[t - 1][l] if t > 0 else np.zeros_like(h)

                # delta from the next time step
                delta_t_next = delta_h_next[l] @ self.W_recurrent[l].T

                # only for last layer
                delta_from_output = delta_output if l == num_hidden_layers - 1 else np.zeros_like(h)

                # delta from the 'next' (previous in time) time step 
                if l < num_hidden_layers - 1:
                    delta_from_next_layer = delta_h_current[l + 1] @ self.W_input[l + 1].T
                else:
                    delta_from_next_layer = np.zeros_like(h)

                # total "error":
                delta = (delta_t_next + delta_from_output + delta_from_next_layer) * self.activation_func_derivative(h)

                ## Gradients:
                if l == 0:
                    prev_activation = X[:, t, :]
                else:
                    prev_activation = self.hidden_states[t][l - 1]

                total_dW_input[l] += prev_activation.T @ delta / batch_size
                total_dW_recurrent[l] += h_prev.T @ delta / batch_size
                total_db_hidden[l] += np.sum(delta, axis=0, keepdims=True) / batch_size

                # storing for next time step:
                delta_h_current[l] = delta

            # update delta_h_next for the next time step
            delta_h_next = delta_h_current

        # Update weights and biases using the optimizer functions
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