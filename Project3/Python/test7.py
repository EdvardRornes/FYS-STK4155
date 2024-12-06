import matplotlib.pyplot as plt 
import numpy as np
from utils import Activation
from test5 import WeightedBinaryCrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight
import time 

T = 10; N = 1000
t = np.linspace(0, T, N)

X = np.sin(t/2) + np.sin(t/2) + np.random.random(N)

event_start = 250; event_end = 350
tmp = np.exp(-(t-t[event_start])**2)

X[event_start:event_end] = tmp[event_start: event_end]

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
eta = 0.00001

def forward(X, W_hh, W_hx, W_yh):
    N, _ = np.shape(X)
    hidden_states = [np.zeros((hidden_dim)) for _ in range(N)]
    z = [np.zeros((hidden_dim)) for _ in range(N)]

    output = []
    h_prev = np.zeros(hidden_dim)

    for n in range(N):
        z_n = W_hx @ X[n, :] + W_hh @ h_prev
        hidden_states[n] = activation(z_n)
        z[n] = z_n
        h_prev = hidden_states[n]
        output.append(activation_out(W_yh @ h_prev))

    output = np.array(output).reshape(-1, 1)
    return output, z, hidden_states


def backward(X, y_true, output, z, hidden_states, N, epoch, W_hh, W_hx, W_yh):
    dL_dW_hh = np.zeros_like(W_hh)
    dL_dW_hx = np.zeros_like(W_hx)
    dL_dW_yh = np.zeros_like(W_yh)

    dh_next = np.zeros((hidden_dim,))
    dz_next = np.zeros((hidden_dim,))

    for n in reversed(range(N)):
        dL_dy = loss_func.gradient(y_true[n], output[n], epoch)
        dy_dz = activation_out_derivative(output[n])
        dz = dL_dy * dy_dz

        dL_dW_yh += dz * hidden_states[n]

        dh = W_yh.T @ dz + dh_next
        dz = dh * activation_derivative(z[n])
        dL_dW_hh += np.outer(dz, hidden_states[n - 1] if n > 0 else 0)
        dL_dW_hx += np.outer(dz, X[n])

        dh_next = W_hh.T @ dz

    # Update weights
    W_hh -= eta * dL_dW_hh
    W_hx -= eta * dL_dW_hx
    W_yh -= eta * dL_dW_yh


def train(X, y, W_hh, W_hx, W_yh, epochs=100, batch_size=128):
    start_time = time.time()

    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            # Forward pass
            y_pred, z, hidden_states = forward(X_batch, W_hh, W_hx, W_yh)

            # Backward pass
            backward(X_batch, y_batch, y_pred, z, hidden_states, len(X_batch), epoch, W_hh, W_hx, W_yh)

        # Validation after each epoch
        y_pred, _, _ = forward(X, W_hh, W_hx, W_yh)
        loss = loss_func(y, y_pred)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    print("Training complete!")


train(X, y, W_hh, W_hx, W_yh, epochs=100)

y_pred, z, hidden_states = forward(X, W_hh, W_hx, W_yh)

print(np.array(y_pred))
plt.plot(t, X[:,0])
plt.plot(t, y[:,0], label="true")
plt.plot(t, y_pred[:,0], label="my shit")
plt.legend()
plt.show()