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