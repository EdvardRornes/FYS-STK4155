from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pywt
from NNs import *
from utils import GWSignalGenerator

if __name__ == "__main__":
     #### Creating example of synthethic GW data:

    # Parameters
    N = 1_000
    T = 50; event_length = (N//10, N//8)
    event_length_test = (N//10, N//8)

    window_size = 10
    batch_size = N
    learning_rate = 1e-2
    regularization_value = 1e-7
    gw_earlyboost = 1
    epochs = 100
    clip_value = 5

    SNR = 1.2

    t = np.linspace(0, T, N)
    fs = 1/T

    # Background noise
    X1 = (0.5*np.sin(90*t) - 0.5*np.cos(60*t)*np.sin(-5.*t) + 0.3*np.cos(30*t) + 0.05*np.sin(N/40*t))/SNR
    X2 = (0.5*np.sin(40*t) - 0.5*np.cos(25*t)*np.sin(-10*t) + 0.3*np.cos(60*t) + 0.10*np.sin(N/18*t))/SNR
    X_test = (0.5*np.sin(50*t) - 0.5*np.cos(80*t)*np.sin(-10*t) + 0.3*np.cos(40*t) + 0.05*np.sin(N/20*t))/SNR

    # Add a single synthetic GW event to each sample
    generator = GWSignalGenerator(signal_length=N)
    # y_i = np.zeros_like(x) # For no background signal tests
    event = generator.generate_random_events(1, event_length)
    generator.apply_events(X1, event)
    y1 = np.array(generator.labels)

    generator = GWSignalGenerator(signal_length=N)
    # y_i = np.zeros_like(x) # For no background signal tests
    event = generator.generate_random_events(1, event_length)
    generator.apply_events(X2, event)
    y2 = np.array(generator.labels)

    generator = GWSignalGenerator(signal_length=N)
    event = generator.generate_random_events(1, event_length)
    generator.apply_events(X_test, event)
    y_test = np.array(generator.labels)

    X = [X1, X2]
    y = [y1, y2]

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
    n_filters = 16
    epochs = 10
    batch_size = 32
    eta = 1e-2
    optimizer = Adam(learning_rate=eta)

    datas = []
    for i in range(len(X)):
        scales = np.arange(1, 256)
        _coefficients, _frequencies = pywt.cwt(X[i], scales, 'morl', sampling_period=1/fs)

        # Prepare data: split real and imaginary parts of coefficients
        real_coefficients = np.real(_coefficients)
        imag_coefficients = np.imag(_coefficients)

        # Stack along the channel dimension
        data = np.stack([real_coefficients, imag_coefficients], axis=-1)  # Shape: (scales, time, 2)

        # Reshape data for input into CNN
        data = np.moveaxis(data, 1, 0)  # Shape: (time, scales, 2)
        data = data[..., np.newaxis]  # Add channel dimension, final shape: (time, scales, 2, 1)

        datas.append(data)

    y = [q.reshape(-1,1) for q in y]

    test = KerasCNN(input_shape, n_filters, optimizer)
    test.train_multiple_datas(datas, y, epochs, batch_size)

    scales = np.arange(1, 256)
    _coefficients, _frequencies = pywt.cwt(X_test, scales, 'morl', sampling_period=1/fs)

    # Prepare data: split real and imaginary parts of coefficients
    real_coefficients = np.real(_coefficients)
    imag_coefficients = np.imag(_coefficients)

    # Stack along the channel dimension
    data = np.stack([real_coefficients, imag_coefficients], axis=-1)  # Shape: (scales, time, 2)
    
    # Reshape data for input into CNN
    data = np.moveaxis(data, 1, 0)  # Shape: (time, scales, 2)
    data = data[..., np.newaxis]  # Add channel dimension, final shape: (time, scales, 2, 1)

    print("hei")
    predictions = test.predict(data)
    print("hei")

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
    plt.show()