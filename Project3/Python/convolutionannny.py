from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pywt
from NNs import *
from utils import GWSignalGenerator


def preprocess_wavelet_data(X, fs, scales=np.arange(1, 256)):
    all_data = []
    for x in X:
        coefficients, _ = pywt.cwt(x, scales, 'morl', sampling_period=1/fs)
        real_coefficients = np.real(coefficients)
        imag_coefficients = np.imag(coefficients)
        data = np.stack([real_coefficients, imag_coefficients], axis=-1)
        data = np.moveaxis(data, 1, 0)
        data = data[..., np.newaxis]
        all_data.append(data)
    return all_data


if __name__ == "__main__":
     #### Creating example of synthethic GW data:

    # Parameters
    N = 10_000
    T = 50; event_length = (N//10, N//8)
    event_length_test = (N//10, N//8)

    batch_size = N

    # Parameters
    N = 5_000
    T = 50
    num_samples = 5
    batch_size = N
    etas = [5e-3, 1e-3, 5e-2, 1e-2, 5e-2]
    regularization_values = np.logspace(-12, -6, 7)
    gw_earlyboosts = np.linspace(1, 1.5, 6)
    epoch_list = [1, 25, 50, 100]
    clip_value = 5
    n_filters = 16
    SNR = 5

    t = np.linspace(0, T, N)
    fs = 1/T

    # Background noise
    X = [
        0.5*np.sin(90*t) - 0.5*np.cos(60*t)*np.sin(-5.*t) + 0.3*np.cos(30*t) + 0.05*np.sin(N/40*t),
        0.5*np.sin(50*t) - 0.5*np.cos(80*t)*np.sin(-10*t) + 0.3*np.cos(40*t) + 0.05*np.sin(N/20*t),
        0.5*np.sin(40*t) - 0.5*np.cos(25*t)*np.sin(-10*t) + 0.3*np.cos(60*t) + 0.10*np.sin(N/18*t),
        0.7*np.sin(70*t) - 0.4*np.cos(10*t)*np.sin(-15*t) + 0.4*np.cos(80*t) + 0.05*np.sin(N/12*t),
        0.1*np.sin(80*t) - 0.4*np.cos(50*t)*np.sin(-3.*t) + 0.3*np.cos(20*t) + 0.02*np.sin(N/30*t)
    ]


    for i in range(len(X)):
        X[i] /= SNR # Quick rescaling, the division factor is ~ SNR

    event_lengths = [(N//10, N//8), (N//7, N//6), (N//14, N//12), (N//5, N//3), (N//5, N//4)]

    events = []
    y = []

    # Add a single synthetic GW event to each sample
    for i in range(num_samples):
        generator = GWSignalGenerator(signal_length=N)
        # y_i = np.zeros_like(x) # For no background signal tests
        events_i = generator.generate_random_events(1, event_lengths[i])
        generator.apply_events(X[i], events_i)

        # y.append(y_i)
        events.append(events_i)
        y.append(generator.labels)

    coefficients = []

    scales = np.arange(1, 256)
    _coefficients, _frequencies = pywt.cwt(X[0], scales, 'morl', sampling_period=1/fs)

    # Prepare data: split real and imaginary parts of coefficients
    real_coefficients = np.real(_coefficients)
    imag_coefficients = np.imag(_coefficients)

    # Stack along the channel dimension
    data = np.stack([real_coefficients, imag_coefficients], axis=-1)  # Shape: (scales, time, 2)
    
    # Reshape data for input into CNN
    data = np.moveaxis(data, 1, 0)  # Shape: (time, scales, 2)
    data = data[..., np.newaxis]  # Add channel dimension, final shape: (time, scales, 2, 1)

    input_shape = (data.shape[1], data.shape[2], data.shape[3]) 

    datas = preprocess_wavelet_data(X, fs, scales=np.arange(1, 256))

    y = [q.reshape(-1,1) for q in y]

    progress = 0
    total_iterations = len(etas)*len(regularization_values)*len(gw_earlyboosts)*len(epoch_list)*num_samples
    start_time = time.time()

    # Prepare to save data
    save_path = "CNN_Parameter_Search"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epochs in epoch_list:
        for boost in gw_earlyboosts:
            for eta in etas:
                optimizer = Adam(learning_rate=eta)
                for lamb in regularization_values:
                    base_filename = f"CNN_Synthetic_GW_Parameter_Tuning_Results_timesteps{N}_SNR{SNR}_epoch{int(epochs)}_lamd{lamb}_eta{eta}_boost{boost:.1f}"

                    # Check if results already exist
                    if os.path.exists(os.path.join(save_path, f"{base_filename}.pkl")):
                        print(f"Skipping calculation for {base_filename} as the results already exist.")
                        total_iterations -= num_samples
                        continue  # Skip the calculation and move to the next combination
                    
                    print(f"\nTraining with eta = {eta}, lambda = {lamb}, epochs = {epochs}, early boost = {boost:.1f}")

                    results = []
                    model = KerasCNN(input_shape, n_filters, optimizer, initial_boost=boost, lambda_reg=lamb)

                    for fold in range(num_samples):
                        X_test = X[fold]
                        y_test = y[fold]
                        X_train = [X[i] for i in range(num_samples) if i != fold]
                        y_train = [y[i] for i in range(num_samples) if i != fold]
                        model.train_multiple_datas(datas, y, epochs, batch_size, verbose=1)

                        # Exclude the current test set to create the training set
                        scales = np.arange(1, 256)
                        _coefficients, _frequencies = pywt.cwt(X_test, scales, 'morl', sampling_period=1/fs)

                        # Prepare data: split real and imaginary parts of coefficients
                        real_coefficients = np.real(_coefficients)
                        imag_coefficients = np.imag(_coefficients)

                        # Stack along the channel dimension
                        data = np.stack([real_coefficients, imag_coefficients], axis=-1)  # Shape: (scales, time, 2)

                        # Move axes to (time_steps, scales, 2)
                        data = np.moveaxis(data, 1, 0)

                        # Add a new channel dimension: (time_steps, scales, 2) -> (time_steps, scales, 2, 1)
                        data = data[..., np.newaxis]

                        # Predict with the correctly shaped data
                        predictions = model.predict(data)


                        predictions = predictions.reshape(-1)

                        
                        # loss, weighted_Acc = model.evaluate(y_test, predictions)
                        test_loss, test_accuracy = model.model.evaluate(data, y_test, verbose=1)
                        print(test_loss, test_accuracy, "her")

                        # Plot the results
                        plt.figure(figsize=(10,4))
                        plt.plot(t, y_test.flatten(), label='True Labels', color='green', lw=2, alpha=0.7)
                        plt.plot(t, X_test, label='Signal', color='purple', lw=1)

                        # Predictions are probabilities, plot them for the test portion
                        plt.plot(t, predictions, label='Predicted Probability', color='red', lw=2, alpha=0.7)

                        plt.xlabel('Time [s]')
                        plt.ylabel('Amplitude / Probability')
                        plt.title('True Labels vs Predicted vs Original Signal')
                        plt.grid(True)
                        plt.legend()

                        results.append({
                            "epochs": epochs,
                            "boost": boost,
                            "learning_rate": eta,
                            "regularization": lamb,
                            "fold": fold,
                            "y_test": X_test.tolist(),
                            "test_labels": y_test.tolist(),
                            "predictions": predictions.tolist(),
                            "loss": loss,
                            "accuracy": weighted_Acc
                        })

                        progress += 1
                        percentage_progress = (progress / total_iterations) * 100
                        # Elapsed time
                        elapsed_time = time.time() - start_time
                        ETA = elapsed_time*(100/percentage_progress-1)

                        print(f"Progress: {progress}/{total_iterations} ({percentage_progress:.2f}%), Time elapsed = {elapsed_time:.1f}s, ETA = {ETA:.1f}s, Test loss = {loss:.3f}, Test accuracy = {100*weighted_Acc:.1f}%\n")
                        
                    save_results_incrementally(results, base_filename, save_path)