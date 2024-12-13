
import matplotlib.pyplot as plt 
import numpy as np
from utils import Activation, Adam
from test2 import RNN, WeightedBinaryCrossEntropyLoss, DynamicallyWeightedLoss
from sklearn.utils.class_weight import compute_class_weight
import time 
from GW_generator import GWSignalGenerator

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
    X = np.array(X[0]); X = X.reshape(-1,1)
    y = np.array(y[0]); y = y.reshape(-1,1)

    unique_classes = np.unique(y[:,0])
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y[:,0])

    # Extract weights
    weight_0 = class_weights[0]  # Weight for class 0
    weight_1 = class_weights[1]  # Weight for class 1

    loss_func = WeightedBinaryCrossEntropyLoss(weight_0=weight_0, weight_1=weight_1)
    # loss_func = DynamicallyWeightedLoss(initial_boost=1.4, labels=y[:,0])
    eta = 0.5
    test = RNN(1, [5, 10, 2], 1, Adam(learning_rate=eta), activation="tanh", activation_out="sigmoid", lambda_reg=1e-7,
               loss_function=loss_func)
    
    test.train(X, y, epochs=100, batch_size=256, window_size=100, clip_value=2)

    y_pred = test.predict(X.reshape(-1, 1, 1))
    y_pred[:,0,0] = 1 * (y_pred[:,0,0])
    print(np.shape(y_pred))
    plt.plot(t, y_pred[:,0,0], label="predict")
    plt.plot(t, X[:,0])
    plt.legend()
    plt.show()

exit()

T = 10; N = 1000
t = np.linspace(0, T, N)

X = np.sin(t*20) + np.sin(t*2) + np.random.random(N) + np.sin(t)* np.random.random(N)
X = X/2

event_start = 250; event_end = 350
tmp = 4*np.exp(-(t-t[event_start] - abs(t-t[event_start])/2)**2/(0.5)**2)

X += tmp

y = np.zeros(N); y[event_start:event_end] = 1


def _xavier_init(layer1:int, layer2:int):
    return np.random.randn(layer1, layer2) * np.sqrt(2 / layer1)

input_dim = 1; output_dim = 1
hidden_dim = 10

W_hx = _xavier_init(hidden_dim, input_dim)
W_hh = _xavier_init(hidden_dim, hidden_dim)
W_yh = _xavier_init(output_dim, hidden_dim)

activation = Activation("tanh")
activation_derivative = activation.derivative()
activation_out = Activation("sigmoid")
activation_out_derivative = activation_out.derivative()


import numpy as np
np.random.seed(42)
unique_classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)

# Extract weights
weight_0 = class_weights[0]  # Weight for class 0
weight_1 = class_weights[1]  # Weight for class 1

loss_func = WeightedBinaryCrossEntropyLoss(weight_0=weight_0, weight_1=weight_1)


y = y.reshape(-1, 1)
X = X.reshape(-1, 1)

eta = 0.005
loss_function = DynamicallyWeightedLoss(initial_boost=1.3)
loss_function = WeightedBinaryCrossEntropyLoss(weight_0=weight_0, weight_1=weight_1)
testRNN = RNN(1, [5, 10, 2], 1, Adam(learning_rate=eta, momentum=0), scaler="standard",
              activation="tanh", activation_out="sigmoid", loss_function=loss_function, lambda_reg=1e-7)

print(np.shape(y), np.shape(X))
testRNN.train(X, y, epochs=50, batch_size=64, window_size=50)

y_pred = testRNN.predict(X.reshape(-1, 1, 1))

plt.plot(t, X[:,0])
plt.plot(t, y[:,0], label="true")
plt.plot(t, y_pred[:,0], label="my")
plt.legend()
plt.show()