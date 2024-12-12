
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
                print(f"X_batch.shape = {X_batch.shape}")
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
        return self._forward(X)
    
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
        
        # print(np.shape(X_batch))
        # for i in range(len(self.hidden_states)):
        #     print(np.shape(self.hidden_states[i]), np.shape(self.z[i]))
        # exit()
        # Output for each timestep
        outputs = []
        
        for t in range(window_size):
            x_t = X_batch[:, t, :]
            prev_state = np.zeros_like(self.hidden_states[0][0])
            
            for l in range(self.L - 1):
                self.z[l][t] = self.W_hx[l] @ x_t.T + self.W_hh[l] @ prev_state + self.b_h[l]
                self.hidden_states[l][t] = self.activation_func(self.z[l][t])
                
                # Next iteration:
                prev_state = self.hidden_states[l+1][t]
                x_t = self.hidden_states[l][t].T
            
            self.z[-1][t] = self.W_yh @ self.hidden_states[-2][t] + self.b_y 
            
            self.hidden_states[-1][t] = self.activation_func_out(self.z[-1][t])

        return self.hidden_states[-1]


    def _backward(self, X_batch: np.ndarray, y_batch: np.ndarray, y_pred: np.ndarray, epoch: int, batch_index: int):
        """
        Backward pass for the RNN.
        Parameters:
            X_batch: Input data for the batch (batch_size, window_size, input_size).
            y_batch: True output labels for the batch.
            y_pred: Predicted outputs from the forward pass.
        """
        batch_size, window_size, _ = X_batch.shape

        # Gradients initialization
        dL_dW_hx = [np.zeros_like(self.W_hx[l]) for l in range(self.L-1)]
        dL_dW_hh = [np.zeros_like(self.W_hh[l]) for l in range(self.L-1)]
        dL_db_h = [np.zeros_like(self.b_h[l]) for l in range(self.L-1)]
        dL_dW_yh = np.zeros_like(self.W_yh)
        dL_db_y = np.zeros_like(self.b_y)

        delta_hx = [np.zeros_like(self.W_hx[l]) for l in range(self.L-1)]
        delta_hh = [np.zeros_like(self.W_hh[l]) for l in range(self.L-1)]

        
        for l in range(self.L-1):
            Delta = [[1 for k in range(window_size)] for n in range(window_size)]
            Delta = [[np.eye(self.W_hh[l].shape[0]) for k in range(window_size)] for j in range(window_size)]

            for k in range(window_size): # n= window_size - 1
                n = window_size - 1

                dL_dy_n = self._loss_function.gradient(y_batch[:, n, :], y_pred[n].T, epoch)
                print(np.shape(dL_dy_n), np.shape(self.W_yh), np.shape(self.activation_func_out_derivative(y_pred[n])))
                dy_n_dh_n = self.W_yh @ self.activation_func_out_derivative(y_pred[n])
                print(np.shape(dy_n_dh_n), np.shape(self.activation_func_out_derivative(y_pred[n])), np.shape(self.W_yh))
                dL_dh_n = dL_dy_n.T @ dy_n_dh_n
                print(np.shape(dL_dh_n), "her")
                # exit()
                prod_ = np.eye(self.W_hh[l].shape[0]) 
                for j in range(k+1, window_size):
                    

                    activation_derivative = self.activation_func_derivative(self.z[l][j])
                    prod_ = prod_ @ (self.W_hh[l] @ activation_derivative)
                    Delta[j][k] = prod_
            

                print(np.shape(dL_dh_n), np.shape(Delta[-1][k]), np.shape(delta_hh[l][k]))
                dL_dW_hh[l] += dL_dh_n * Delta[-1][k] * delta_hh[l][k]
                dL_dW_hx[l] += dL_dh_n * Delta[-1][k] * delta_hx[l][k]

                dL_db_h[l] += dL_dh_n * self.activation_func_derivative(self.z[l][k]) 
                
            for n in reversed(range(window_size-1)):
                for k in range(n):
                    dL_dy_n = self._loss_function.gradient(y_batch[:, n, :], y_pred[n], epoch)
                    dy_n_dh_n = self.activation_func_out_derivative(y_pred[n]) * self.W_yh
                    dL_dh_n = dL_dy_n * dy_n_dh_n

                    for j in range(k+1, n):
                        prod = self.activation_func_derivative(self.z[l][j]) * self.W_hh[l][j] * prod
                    
                    dL_dW_hh[l] += dL_dh_n * delta_hh[l][k]
                    dL_dW_hx[l] += dL_dh_n * delta_hx[l][k]

                    dL_db_h[l] += dL_dh_n * self.activation_func_derivative(self.z[l][k]) 



        # Update weights and biases
        for l in range(self.L):
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

        # for l in range(len(self.W_hh)):
        #     print(f"W_hx: {np.shape(self.W_hx[l])}, W_hh: {np.shape(self.W_hh[l])}, b_h: {np.shape(self.b_h[l])}")
        
        # print(f"W_yh: {np.shape(self.W_yh)}, b_h: {np.shape(self.b_y)}")
        # exit()