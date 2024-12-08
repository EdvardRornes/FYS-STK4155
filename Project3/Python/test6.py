import numpy as np
from utils import Activation
from test5 import WeightedBinaryCrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight
import time 

T = 10; N = 1000
t = np.linspace(0, T, N)

X = np.sin(t/5) + np.sin(t/8)

event_start = 250; event_end = 350
tmp = 10*np.exp(-(t-t[event_start])**2)

X[event_start: event_end] += tmp[event_start: event_end]

y = np.zeros(N); y[event_start:event_end] = 1


def _xavier_init(layer1:int, layer2:int):
    return np.random.randn(layer1, layer2) * np.sqrt(2 / layer1)

input_dim = 1; output_dim = 1
hidden_dim = 4

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

def forward(X, W_hh, W_hx, W_yh):

    N, _ = np.shape(X)
    hidden_states = [np.zeros((hidden_dim)) for i in range(N)]
    z = [np.zeros((hidden_dim)) for i in range(N)]

    output = []

    h_prev = np.zeros(hidden_dim)
    for n in range(N):
        
        z_n = W_hx @ X[n,:] + W_hh @ h_prev
        hidden_states[n] = activation(z_n)
        z[n] = z_n 

        h_prev = hidden_states[n]

        output.append(activation_out(W_yh @ h_prev))

    return output, z, hidden_states

def backward(X, y_true, output, z, hidden_states, N, epoch, W_hh, W_hx, W_yh):
    
    dL_dW_hh = np.zeros_like(W_hh)
    dL_dW_hx = np.zeros_like(W_hx)
    dL_dW_yh = np.zeros_like(W_yh)

    delta_hh = [np.zeros((hidden_dim, hidden_dim)) for i in range(N)]
    delta_hx = [np.zeros((hidden_dim, hidden_dim)) for i in range(N)]

    for n in range(1, N):
        delta_hh[n] = activation_derivative(z[n]) * (hidden_states[n-1] + W_hh * delta_hh[n-1])
        delta_hx[n] = activation_derivative(z[n]) * (X[n] + W_hh * delta_hh[n-1])

    for n in range(N):
        for k in range(1, n):
            dL_dy_n = loss_func.gradient(y_true[n], output[n], epoch)
            dyn_dh_n = activation_out_derivative(z[n]) * W_yh 
            dL_dhn = dL_dy_n * dyn_dh_n 

            prod = np.eye(hidden_dim, hidden_dim)
            for j in reversed(range(n-k-1)):
                prod = prod * activation_derivative(z[n-j]*W_hh)
            

            # print(np.shape(dL_dhn), np.shape(prod), np.shape(delta_hx[k]))
            # print(np.shape(prod @ delta_hx[k]))
            # print(np.shape(dL_dW_hx))
            # print(np.shape(dL_dhn @ (prod * delta_hx[k])))
            dL_dW_yh += dL_dhn * (prod @ activation_out_derivative(z[k]))
            dL_dW_hh += dL_dhn * (prod @ delta_hh[k])
            dL_dW_hx += (dL_dhn @ (prod * delta_hx[k])).T

    
    W_hh -= eta * dL_dW_hh 
    W_hx -= eta * dL_dW_hx
    W_yh -= eta * dL_dW_yh


def train(X:np.ndarray, y:np.ndarray, W_hh, W_hx, W_yh, epochs=10, batch_size=128) -> None:
    """
    Trains the RNN on the given dataset.
    """

    start_time = time.time()
    best_loss = 1e4

    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            # Forward pass
            y_pred, z, hidden_states = forward(X_batch, W_hh, W_hx, W_yh)

            # Backward pass
            backward(X_batch, y_batch, y_pred, z, hidden_states, batch_size, epoch, W_hh, W_hx, W_yh)

        # Validation after each epoch
        y_pred = forward(X)
        loss = loss_func(y, y_pred)
        print(loss)


train(X, y, W_hh, W_hx, W_yh)