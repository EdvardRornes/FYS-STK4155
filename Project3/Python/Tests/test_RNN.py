import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils import *


from test import *
import numpy as np

# Generate synthetic time-series data
time_steps = 1000
t = np.linspace(0, 100, time_steps)  # Simulated time

# Create a sinusoidal wave with Gaussian noise
wave_signal = np.sin(0.2 * t) + np.random.normal(0, 0.1, time_steps)

# Add synthetic gravitational events as spikes
events_indices = np.random.choice(time_steps, size=10, replace=False)
wave_signal[events_indices] += np.random.normal(5, 1, len(events_indices))

# Create labels: 1 if a gravitational event (spike) occurs, otherwise 0
y = np.zeros(time_steps)
y[events_indices] = 1

adam = Adam()
rnn = RNN(1, [4, 8, 16], 1, adam, "relu")
rnn.train(t, y)
y_predict = rnn.predict(t)

# Plot synthetic signal
plt.plot(t, wave_signal)
plt.scatter(t[events_indices], wave_signal[events_indices], color='red', label='Gravitational Events')
plt.scatter(t, y_predict, color='green', label='Predicted Gravitational Events')
plt.title('Synthetic Gravitational Wave Signal with Events')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
