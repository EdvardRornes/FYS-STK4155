from utils import GWSignalGenerator
from NNs import NeuralNetwork, RNN, DynamicallyWeightedLoss, WeightedBinaryCrossEntropyLoss
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from utils import * 


def parameter_scan(X:np.ndarray, y:np.ndarray, t:np.ndarray, model:NeuralNetwork, optimizer:Optimizer,
                   epoch_list:list, gw_earlyboosts:list, learning_rates:list, regularization_values:list, 
                   batch_size:int, window_size:int, 
                   clip_value=1e12,
                   save_path="GW_Parameter_Tuning_Results",
                   activation_str="tanh", activation_str_out="sigmoid"):
    

    # Conflicting definition of variables:
    y = X; x = t; labels = y 

    time_steps = len(t)

    progress = 0
    total_iterations = len(learning_rates)*len(regularization_values)*len(gw_earlyboosts)*len(epoch_list)*num_samples
    start_time = time.time()

    print(f"Parameters: time_steps: {time_steps}, batch_size: {batch_size}, window_size: {window_size},")
    print(f"            activation_func: {activation_str}, activation_func_out: {activation_str_out}")
    # Loop over learning rates and regularization values
    for epochs in epoch_list:
        for boost in gw_earlyboosts:
            for lr in learning_rates:
                for reg_value in regularization_values:
                    results = []

                    # Create a unique filename for the current parameter combination
                    base_filename = f"Synthetic_GW_Parameter_Tuning_Results_timesteps{time_steps}_SNR{SNR}_epoch{int(epochs)}_lamd{reg_value}_eta{lr}_boost{boost:.1f}"

                    # Check if results already exist
                    if os.path.exists(os.path.join(save_path, f"{base_filename}.pkl")):
                        print(f"Skipping calculation for {base_filename} as the results already exist.")
                        total_iterations -= num_samples
                        continue  # Skip the calculation and move to the next combination

                    print(f"\nTraining with eta = {lr}, lambda = {reg_value}, epochs = {epochs}, early boost = {boost:.1f}")

                    for fold in range(num_samples):                    
                        # Split the data into train and test sets for this fold
                        x_test = x  
                        y_test = y[fold]  # Use the fold as the test set

                        test_labels = labels[fold] # Corresponding labels for the test set

                        # Create the training set using all other samples
                        x_train = np.linspace(0, (num_samples - 1) * time_for_1_sample, time_steps * (num_samples - 1))  # Just for plotting
                        y_train = np.concatenate([y[i] for i in range(num_samples) if i != fold], axis=0)
                        train_labels = np.concatenate([labels[i] for i in range(num_samples) if i != fold], axis=0)
                            
                        # Initialize the KerasRNN model with the current learning rate and regularization
                        hidden_layers = [5, 10, 2]  # Example hidden layers
                        optimizer.learning_rate = lr

                        unique_classes = np.unique(y[:,0])
                        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y[:,0])

                        # Extract weights
                        weight_0 = class_weights[0]  # Weight for class 0
                        weight_1 = class_weights[1]  # Weight for class 1

                        loss_func = WeightedBinaryCrossEntropyLoss(weight_0=weight_0, weight_1=weight_1)
                        # loss_func = DynamicallyWeightedLoss(initial_boost=1.4)

                        I_am_the_trainer = model(
                            1, hidden_layers, 1,
                            optimizer, activation=activation_str, activation_out=activation_str_out,
                            lambda_reg=reg_value,
                            loss_function=loss_func
                        )

                        y_train = y_train.reshape(-1, 1); train_labels = train_labels.reshape(-1, 1)
                        I_am_the_trainer.train(y_train, train_labels, epochs, batch_size=batch_size, window_size=window_size, clip_value=clip_value)

                        # Predict with the trained model
                        predictions = I_am_the_trainer.predict(y_test.reshape(-1, 1, 1))

                        predictions = predictions[:, 0, 0]
                        predicted_labels = 1 * (predictions >= 0.5)

                        loss = I_am_the_trainer.calculate_loss(test_labels, predicted_labels, epochs)


                        predicted_labels = np.array(predicted_labels, dtype=int); test_labels = np.array(test_labels, dtype=int)
                        accuracy = accuracy_score(test_labels, predicted_labels)
                        x_pred = x[window_size - 1:]

                        results.append({
                            "epochs": epochs,
                            "boost": boost,
                            "learning_rate": lr,
                            "regularization": reg_value,
                            "fold": fold,
                            "train_labels": train_labels.tolist(),
                            "y_test": y_test.tolist(),
                            "test_labels": test_labels.tolist(),
                            "predictions": predictions.tolist(),
                            "loss": loss,
                            "accuracy": accuracy
                        })

                        progress += 1
                        percentage_progress = (progress / total_iterations) * 100
                        # Elapsed time
                        elapsed_time = time.time() - start_time
                        ETA = elapsed_time*(100/percentage_progress-1)

                        print(f"Progress: {progress}/{total_iterations} ({percentage_progress:.2f}%), Time elapsed = {elapsed_time:.1f}s, ETA = {ETA:.1f}s, Test loss = {loss:.3f}, Test accuracy = {100*accuracy:.1f}%\n")

                    save_results_incrementally(results, base_filename, save_path=save_path)

if __name__ == "__main__":
    # Create the GWSignalGenerator instance
    time_steps = 5000
    time_for_1_sample = 50
    t = np.linspace(0, time_for_1_sample, time_steps)
    num_samples = 5

    window_size = time_steps//100
    batch_size = time_steps//50*(num_samples-1)

    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    regularization_values = np.logspace(-10, 0, 11)

    gw_earlyboosts = np.linspace(1, 1.5, 6)
    epoch_list = [10, 25, 50, 100]
    SNR = 100

    events = []
    y = []

    # Background noise
    X = [
        (0.5*np.sin(90*t) - 0.5*np.cos(60*t)*np.sin(-5.*t) + 0.3*np.cos(30*t) + 0.05*np.sin(time_steps/40*t))/SNR,
        (0.5*np.sin(50*t) - 0.5*np.cos(80*t)*np.sin(-10*t) + 0.3*np.cos(40*t) + 0.05*np.sin(time_steps/20*t))/SNR,
        (0.5*np.sin(40*t) - 0.5*np.cos(25*t)*np.sin(-10*t) + 0.3*np.cos(60*t) + 0.10*np.sin(time_steps/18*t))/SNR,
        (0.7*np.sin(70*t) - 0.4*np.cos(10*t)*np.sin(-15*t) + 0.4*np.cos(80*t) + 0.05*np.sin(time_steps/12*t))/SNR,
        (0.1*np.sin(80*t) - 0.4*np.cos(50*t)*np.sin(-3.*t) + 0.3*np.cos(20*t) + 0.02*np.sin(time_steps/30*t))/SNR]

    event_lengths = [(time_steps//10, time_steps//8), (time_steps//7, time_steps//6), 
                    (time_steps//14, time_steps//12), (time_steps//5, time_steps//3),
                    (time_steps//5, time_steps//4)]

    # Add a single synthetic GW event to each sample
    for i in range(num_samples):
        generator = GWSignalGenerator(signal_length=time_steps)
        # y_i = np.zeros_like(x) # For no background signal tests
        events_i = generator.generate_random_events(1, event_lengths[i])
        generator.apply_events(X[i], events_i)

        # y.append(y_i)
        events.append(events_i)
        y.append(generator.labels)

    # Convert lists into numpy arrays
    X = np.array(X)
    y = np.array(y)

    parameter_scan(X, y, t, RNN, Adam(), epoch_list, gw_earlyboosts, learning_rates, regularization_values,
                   batch_size, window_size, 
                   save_path="GW_Merged_Results/RNN", clip_value=2)