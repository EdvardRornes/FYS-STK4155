
import matplotlib.pyplot as plt 
import numpy as np
from utils import Activation
from test5 import WeightedBinaryCrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight
import time 

T = 10; N = 1000
t = np.linspace(0, T, N)

X = np.sin(t*20) + np.sin(t*2) + np.random.random(N) + np.sin(t)* np.random.random(N)

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
eta = 0.001
from utils import Adam
optimizer = Adam(learning_rate=eta)

def _xavier_init(layer1: int, layer2: int):
    return np.random.randn(layer1, layer2) * np.sqrt(2 / (layer1 + layer2))


# Initialize weights for L hidden layers
L = 3 # Number of hidden layers
hidden_dim = 32
layers = [32, 32, 32]
hidden_dims = layers

W_h = [None] * L#; W_h = [1000, 1600, 1000]
b_h = [None] * L#; b_h = [1000, 1600, 1000]

optimizer_W_h = []; optimizer_b_h = []
for l in range(L):
    W_h[l] = _xavier_init(hidden_dims[l], input_dim if l == 0 else hidden_dims[l - 1])
    optimizer_W_h.append(optimizer.copy())
    b_h[l] = np.zeros((hidden_dims[l],))
    optimizer_b_h.append(optimizer.copy())

W_yh = _xavier_init(output_dim, hidden_dims[-1])  # Output layer weights
b_yh = np.zeros((output_dim,))  # Output layer bias

optimizer_W_yh = optimizer.copy()
optimizer_b_yh = optimizer.copy()

def forward(X, W_h, b_h, W_yh, b_yh):
    N, _ = np.shape(X)
    hidden_states = [np.zeros((N, h_dim)) for h_dim in hidden_dims]
    z = [np.zeros((N, h_dim)) for h_dim in hidden_dims]

    outputs = []

    for n in range(N):
        x_n = X[n, :]
        for l in range(L):
            z[l][n] = (W_h[l] @ x_n if l == 0 else W_h[l] @ hidden_states[l - 1][n]) + b_h[l]
            hidden_states[l][n] = activation(z[l][n])

        # Output layer
        y_n = W_yh @ hidden_states[-1][n] + b_yh
        outputs.append(activation_out(y_n))

    outputs = np.array(outputs).reshape(-1, 1)
    return outputs, z, hidden_states


def backward(X, y_true, output, z, hidden_states, N, epoch, W_h, b_h, W_yh, b_yh):
    dL_dW_h = [np.zeros_like(W_h[l]) for l in range(L)]
    dL_db_h = [np.zeros_like(b_h[l]) for l in range(L)]

    dL_dW_yh = np.zeros_like(W_yh)
    dL_db_yh = np.zeros_like(b_yh)

    dh_next = [np.zeros((hidden_dims[l],)) for l in range(L)]

    for n in reversed(range(N)):
        dL_dy = loss_func.gradient(y_true[n], output[n], epoch)
        dy_dz = activation_out_derivative(output[n])
        dz = dL_dy * dy_dz

        dL_dW_yh += dz * hidden_states[-1][n]
        dL_db_yh += dz

        dh = W_yh.T @ dz

        for l in reversed(range(L)):
            dz = dh * activation_derivative(z[l][n])
            dL_dW_h[l] += np.outer(dz, X[n] if l == 0 else hidden_states[l - 1][n])
            dL_db_h[l] += dz

            if l > 0:
                dh = W_h[l].T @ dz

    # Update weights and biases
    eps = 5
    for l in range(L):
        
        norm = np.linalg.norm(W_h[l])
        W_h[l] = W_h[l] if norm < eps else W_h[l] / norm * eps
        W_h[l] = optimizer_W_h[l](W_h[l], dL_dW_h[l], epoch, 0)
        
        b_h[l] = optimizer_b_h[l](b_h[l], dL_db_h[l], epoch, 0)
        # W_h[l] -= eta * dL_dW_h[l]
        # b_h[l] -= eta * dL_db_h[l]
    
    W_yh = optimizer_W_yh(W_yh, dL_dW_yh, epoch, 0)
    
    b_yh = optimizer_b_yh(b_yh, dL_db_yh, epoch, 0)

    # W_yh -= eta * dL_dW_yh
    # b_yh -= eta * dL_db_yh


def train(X, y, W_h, b_h, W_yh, b_yh, epochs=10, batch_size=128):
    start_time = time.time()

    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            # Forward pass
            y_pred, z, hidden_states = forward(X_batch, W_h, b_h, W_yh, b_yh)

            # Backward pass
            backward(X_batch, y_batch, y_pred, z, hidden_states, len(X_batch), epoch, W_h, b_h, W_yh, b_yh)

        # Validation after each epoch
        y_pred, _, _ = forward(X, W_h, b_h, W_yh, b_yh)
        loss = loss_func(y, y_pred)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    print("Training complete!")


from test5 import RNN

# eta = 0.1
testRNN = RNN(1, [2, 64, 2], 1, Adam(learning_rate=eta, momentum=0), activation="tanh", activation_out="sigmoid")
testRNN.train(X, y, epochs=100, batch_size=4, window_size=200)

y_pred = testRNN.predict(X.reshape(-1, 1, 1))
# train(X, y, W_h, b_h, W_yh, b_yh, epochs=100, batch_size=16)

# y_pred, z, hidden_states = forward(X, W_h, b_h, W_yh, b_yh)

# print(np.array(y_pred))
plt.plot(t, X[:,0])
plt.plot(t, y[:,0], label="true")
plt.plot(t, y_pred[:,0], label="my shit")
plt.legend()
plt.show()