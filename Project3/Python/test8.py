
import matplotlib.pyplot as plt 
import numpy as np
from utils import Activation, Adam
from test5 import WeightedBinaryCrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight
import time 

T = 10; N = 10000
t = np.linspace(0, T, N)

X = np.sin(t*20) + np.sin(t*2) + np.random.random(N) + np.sin(t)* np.random.random(N)

event_start = 250; event_end = 350
tmp = 2*np.exp(-(t-t[event_start])**2)

X[event_start:event_end] += tmp[event_start: event_end]

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
eta = 0.001



from test7 import RNN

eta = 0.005
testRNN = RNN(1, [7, 9, 11], 1, Adam(learning_rate=eta, momentum=0), scaler="standard",
              activation="tanh", activation_out="sigmoid", loss_function=WeightedBinaryCrossEntropyLoss(weight_0=weight_0, weight_1=weight_1))
testRNN.train(X, y, epochs=100, batch_size=256, window_size=10)

y_pred = testRNN.predict(X.reshape(-1, 1, 1))
# train(X, y, W_h, b_h, W_yh, b_yh, epochs=100, batch_size=16)

# y_pred, z, hidden_states = forward(X, W_h, b_h, W_yh, b_yh)

# print(np.array(y_pred))
plt.plot(t, X[:,0])
plt.plot(t, y[:,0], label="true")
plt.plot(t, y_pred[:,0], label="my shit")
plt.legend()
plt.show()