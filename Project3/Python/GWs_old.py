import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import butter, filtfilt
from gwpy.timeseries import TimeSeries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_gw_data(noise_only=False):
    """
    Load either a gravitational wave event data or noise-only data.
    If noise_only=True, the function will fetch noise-only data.
    If noise_only=False, the function will fetch gravitational wave data (e.g., GW150914).
    
    The function also saves the data to a CSV file in the 'Data' folder.
    """
    # Check if the 'Data' directory exists, and create it if not
    if not os.path.exists("Data"):
        os.makedirs("Data")
    
    # Define the file path for GW data or noise data based on the flag
    if noise_only:
        file_path = os.path.join("Data", "noise_data.csv")
        # For noise-only data, use a segment without GW signal (adjust timestamps)
        start_time, end_time = "1126259400", "1126259446"  # Example of noise segment
        data_type = "Noise"
    else:
        file_path = os.path.join("Data", "GW150914_4_NR_waveform.csv")
        # For GW data, use a known GW event (GW150914)
        start_time, end_time = "1126259446", "1126259478"  # Example of GW event
        data_type = "GW Signal"
    
    # Check if the file already exists
    if not os.path.exists(file_path):
        print(f"{data_type} data file not found, downloading data...")
        
        # Use LIGO's public data for the selected type (GW or Noise)
        data = TimeSeries.fetch_open_data("L1", start_time, end_time, verbose=True)
        
        # Save the data to the 'Data' folder in CSV format
        time = data.times.value  # Time values
        strain = data.value  # Strain values
        
        # Save as a CSV file using pandas
        df = pd.DataFrame({'Time': time, 'Strain': strain})
        df.to_csv(file_path, index=False)
        print(f"{data_type} data saved to {file_path}")
    else:
        print(f"{data_type} data already exists: {file_path}")
    
    # Load the existing data from CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Convert the DataFrame back to a TimeSeries object
    data = TimeSeries(df['Strain'].values, times=df['Time'].values)
    
    return data


# Define a high-pass filter function using scipy
def highpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def preprocess_data(data, signal_duration=1.0, sample_rate=4096, overlap=0.5):
    num_samples = int(signal_duration * sample_rate)  # 4096 for 1 second or 2048 for 0.5 second
    step_size = int(overlap * sample_rate)  # 0.5 overlap means step size of 2048
    segments = []
    labels = []

    data_values = data.value  # Full data values
    total_samples = len(data_values)

    # Create noise-only segments (label = 0)
    for i in range(0, total_samples - num_samples, step_size):
        segments.append(data_values[i:i+num_samples])
        labels.append(0)  # Noise

    # Add signal-containing segments (label = 1)
    for i in range(0, total_samples - num_samples, step_size):
        segments.append(data_values[i:i+num_samples])
        labels.append(1)  # Signal

    segments = np.array(segments)
    labels = np.array(labels)

    # Normalize data
    segments = (segments - np.mean(segments)) / np.std(segments)

    return segments, labels


from tensorflow.keras import regularizers

def create_model(input_shape, learning_rate=0.001):
    # Adam optimizer with a specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 5, activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 5, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification (signal or noise)
    ])
    
    # Compile the model with the Adam optimizer and binary cross-entropy loss
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model



# Load and preprocess the GW data
data = load_gw_data()  # Load the data using the updated load function

# Load the noise data
noise_data = load_gw_data(noise_only=True)

# Extract time and strain values
time_noise = noise_data.times.value  # Time values
strain_noise = noise_data.value  # Strain values

# Plot the noise data
plt.figure(figsize=(12, 6))
plt.plot(time_noise, strain_noise, label='Noise Data', lw=0.4)
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.title('LIGO Noise Data')
plt.legend()
plt.grid(True)


# Extract time and strain values
time = data.times.value  # Time values
strain = data.value  # Strain values

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(time, strain, label='Gravitational Wave Signal', lw=0.4)
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.title('Gravitational Wave Signal Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Preprocess the GW data for training (label = 1)
segments_gw, labels_gw = preprocess_data(data)  # Process GW signal data for training

# Preprocess the noise data for testing (label = 0)
segments_noise, labels_noise = preprocess_data(noise_data)  # Process Noise data for testing

# Ensure GW data is used for training, noise data is used for testing
X_train = segments_gw
y_train = labels_gw
X_test = segments_noise
y_test = labels_noise

# Reshape data to fit the model's input shape (samples, time steps, channels)
X_train = X_train[..., np.newaxis]  # Add channel dimension
X_test = X_test[..., np.newaxis]    # Add channel dimension

# Set learning rate
learning_rate = 0.001

# Create and train the model
model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]), learning_rate=learning_rate)
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set (which is noise data)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy on Noise Data: {test_acc:.4f}')

# Predict on the test set (Noise data)
y_pred = model.predict(X_test)

# Convert probabilities to binary class labels (0 or 1)
y_pred = (y_pred < 0.5).astype(int)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using Seaborn heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Noise', 'Signal'], yticklabels=['Noise', 'Signal'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix on Noise Data')

# Plot training & validation accuracy
plt.figure(figsize=(10,8))
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Training & Validation Accuracy')

# Plot training & validation loss
plt.figure(figsize=(10,8))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Training & Validation Loss')
plt.show()
