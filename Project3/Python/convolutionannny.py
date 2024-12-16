from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pywt

def create_convolutional_network(input_shape, n_filters, lmbd, loss_function, optimizer):
    """
    Creates and compiles a convolutional neural network.

    Parameters:
    - input_shape: tuple, shape of the input data (e.g., (height, width, channels)).
    - n_filters: int, number of filters in the convolutional layers.
    - lmbd: float, L2 regularization parameter.
    - loss_function: str, loss function for the model (e.g., 'binary_crossentropy').
    - optimizer: str, optimizer for the model (e.g., 'adam').

    Returns:
    - model: compiled Keras model.
    """
    model = Sequential()
    model.add(layers.Conv2D(n_filters, (3, 3), activation='relu',
                            kernel_regularizer=regularizers.l2(lmbd),
                            input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(n_filters * 2, (3, 3), activation='relu',
                            kernel_regularizer=regularizers.l2(lmbd), padding='same'))
    model.add(layers.UpSampling2D(size=(2, 2)))
    model.add(layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same'))

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    return model

def generate_wavelet_data(n_samples, signal_length, wavelet, scales):
    """
    Generate synthetic wavelet-transformed data with binary labels.

    Parameters:
    - n_samples: int, number of samples to generate.
    - signal_length: int, length of the time-domain signals.
    - wavelet: str, wavelet type for the transform.
    - scales: array-like, scales for the wavelet transform.

    Returns:
    - data: numpy array, wavelet-transformed data of shape (n_samples, len(scales), signal_length, 1).
    - labels: numpy array, binary labels of shape (n_samples, len(scales), signal_length, 1).
    """
    data, labels = [], []
    for _ in range(n_samples):
        spike_position = np.random.randint(signal_length // 4, 3 * signal_length // 4)
        spike_height = np.random.uniform(1, 10)
        signal = np.zeros(signal_length)
        signal[spike_position] = spike_height
        coeffs, _ = pywt.cwt(signal, scales, wavelet)
        coeffs = np.abs(coeffs)  # Take magnitude of coefficients

        label = np.zeros((len(scales), signal_length))
        label[:, spike_position] = 1

        data.append(coeffs)
        labels.append(label)

    data = np.array(data)[..., np.newaxis]  # Add channel dimension
    labels = np.array(labels)[..., np.newaxis]  # Add channel dimension

    return data, labels

if __name__ == "__main__":
    # Time vector
    fs = 5012  # Sampling frequency (high for resolution)
    duration = 10 # Total duration in seconds
    time = np.linspace(0, duration, int(fs * duration))

    # Gravitational wave-like signal
    mid_point = 0.5  # Mid-point where the frequency starts increasing
    sigma = 0.1  # Controls the width of the Gaussian envelope
    amplitude = np.exp(-((time - mid_point) ** 2) / (2 * sigma ** 2))  # Gaussian envelope

    # Frequency increases quadratically over time
    frequency = 10 + 100 * (time - mid_point)**2  # Quadratic frequency increase
    phase = 2 * np.pi * np.cumsum(frequency) / fs  # Integrate frequency to get phase

    # The oscillating signal
    signal = amplitude * np.sin(phase)

    # Add small noise for realism
    noise = np.random.normal(0, 0.7, size=len(time))
    signal += noise

    # Plot the signal
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal, color='purple', lw=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Synthetic Gravitational Wave Signal")
    plt.grid()
    plt.show()

    # Perform Continuous Wavelet Transform
    scales = np.arange(1, 256)
    coefficients, frequencies = pywt.cwt(signal, scales, 'morl', sampling_period=1/fs)

    # Plot the Wavelet Transform
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(coefficients), extent=[0, duration, frequencies[-1], frequencies[0]],
            cmap='viridis', aspect='auto', vmax=np.abs(coefficients).max())
    plt.colorbar(label="Wavelet Coefficient Magnitude")
    plt.title("Wavelet Transform of Gravitational Wave-like Signal")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.yscale('log')  # Logarithmic frequency axis
    plt.show()


        # Reshape the input data to match (batch_size, height, width, channels)
    data = np.abs(coefficients)[..., np.newaxis]  # Shape: (scales, time, 1)
    data = np.expand_dims(data, axis=2)  # Add width dimension, shape: (scales, 1, time, 1)

    # Labels: match the reshaped input data
    labels = labels[..., np.newaxis]  # Expand channel dimension if necessary
    labels = np.expand_dims(labels, axis=2)  # Shape: (scales, 1, time, 1)

    # Verify shapes
    print("Input data shape:", data.shape)
    print("Labels shape:", labels.shape)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, train_size=0.8, random_state=42)


    # X_train, X_test, Y_train, Y_test = train_test_split(data, labels, train_size=0.8, random_state=42)

    # Model parameters
    input_shape = X_train.shape[1:]  # (len(scales), signal_length, 1)
    n_filters = 16
    lmbd = 0.01
    loss_function = 'binary_crossentropy'
    optimizer = 'adam'
    epochs = 10
    batch_size = 32

    # Create and train the model
    model = create_convolutional_network(input_shape, n_filters, lmbd, loss_function, optimizer)
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.3f}")