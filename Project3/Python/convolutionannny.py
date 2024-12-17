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
    N = 5_000
    T = 50
    num_samples = 5
    batch_size = N//50
    etas = [5e-3, 1e-3, 5e-2, 1e-2, 5e-2]
    regularization_values = np.logspace(-12, -6, 7)
    gw_earlyboosts = np.linspace(1, 1.5, 6)
    epoch_list = [50, 25, 10]
    clip_value = 5
    n_filters = 16
    SNR = 5

    t = np.linspace(0, T, N)
    fs = 1/T

    # Background noise
    X = [
        0.5*np.sin(90*t) - 0.5*np.cos(60*t)*np.sin(-5.*t) + 0.3*np.cos(30*t) + 0.05*np.sin(N/40*t),
        0.5*np.sin(50*t) - 0.5*np.cos(80*t)*np.sin(-10*t) + 0.3*np.cos(40*t) + 0.05*np.sin(N/20*t),
        0.5*np.sin(40*t) - 0.5*np.cos(25*t)*np.sin(-10*t) + 0.3*np.cos(60*t) + 0.03*np.sin(N/18*t),
        0.3*np.sin(70*t) - 0.4*np.cos(10*t)*np.sin(-15*t) + 0.4*np.cos(80*t) + 0.05*np.sin(N/12*t),
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

    datas = preprocess_wavelet_data(X, fs, scales=np.arange(1, 256))

    input_shape = (datas[0].shape[1], datas[0].shape[2], datas[0].shape[3]) 

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
                        data_test = datas[fold]
                        X_train = [X[i] for i in range(num_samples) if i != fold]
                        y_train = [y[i] for i in range(num_samples) if i != fold]
                        data_train = [datas[i] for i in range(num_samples) if i != fold]
                        model.train_multiple_datas(data_train, y_train, epochs, batch_size, verbose=1)

                        # Predict with the correctly shaped data
                        predictions = model.predict(data_test)
                        predictions = predictions.reshape(-1)
                        
                        loss, weighted_Acc = model.evaluate(y_test, predictions)

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

                        plt.figure(figsize=(10,6))
                        plt.plot(t, X_test)
                        plt.plot(t, predictions)
                        plt.plot(t, y_test)

                        progress += 1
                        percentage_progress = (progress / total_iterations) * 100
                        # Elapsed time
                        elapsed_time = time.time() - start_time
                        ETA = elapsed_time*(100/percentage_progress-1)

                        print(f"Progress: {progress}/{total_iterations} ({percentage_progress:.2f}%), Time elapsed = {elapsed_time:.1f}s, ETA = {ETA:.1f}s, Test loss = {loss:.3f}, Test accuracy = {100*weighted_Acc:.1f}%\n")
                        plt.show()
                    save_results_incrementally(results, base_filename, save_path)