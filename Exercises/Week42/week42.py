import autograd.numpy as np
from scipy.special import softmax
from sklearn import datasets
import matplotlib.pyplot as plt


np.random.seed(2024)

def ReLU(z):
    return np.where(z > 0, z, 0)

x = np.random.randn(2)
W1 = np.random.randn(4, 2)

# Exercise 1
# a) Define the bias of the first layer
b1 = np.random.randn(4)

# b) Compute intermediary z1
z1 = W1 @ x + b1

# c) Compute the activation a1
a1 = ReLU(z1)

# Confirm the activation is correct
sol1 = np.array([0.60610368, 4.0076268, 0.0, 0.56469864])
print("Activation a1 matches expected solution:", np.allclose(a1, sol1))  # Should print: True

# Exercise 2
# a) Define the weight and bias of the second layer
W2 = np.random.randn(8, 4)
b2 = np.random.randn(8)

# b) Compute intermediary z2 and activation a2
z2 = W2 @ a1 + b2
a2 = ReLU(z2)

# Confirm the activation shape
print("a2 shape matches the expected shape:", a2.shape == (8,))  # Should print: True

# Exercise 3
# a) Create layers function
def create_layers(network_input_size, output_sizes):
    layers = []
    i_size = network_input_size
    for output_size in output_sizes:
        W = np.random.rand(output_size, i_size)
        b = np.random.rand(output_size)
        layers.append((W, b))
        i_size = output_size
    return layers

# b) Complete feed_forward function
def feed_forward(layers, input):
    a = input
    for W, b in layers:
        z = W @ a + b
        a = ReLU(z)
    return a

# c) Create a network and evaluate it
network_input_size = 8
output_sizes = [10, 16, 6, 2]
layers = create_layers(network_input_size, output_sizes)

# Test the network with a random input
x_test = np.random.rand(network_input_size)
output = feed_forward(layers, x_test)
print("Output shape:", output.shape)  # Should print: Output shape: (2,)

# Exercise 4
# a) Update create_layers function to accept activation functions
def create_layers_4(network_input_size, output_sizes, activation_funcs):
    layers = []
    i_size = network_input_size
    for output_size, activation in zip(output_sizes, activation_funcs):
        W = np.random.rand(output_size, i_size)
        b = np.random.rand(output_size)
        layers.append((W, b, activation))
        i_size = output_size
    return layers

# b) Update feed_forward function
def feed_forward_4(layers, input):
    a = input
    for W, b, activation in layers:
        z = W @ a + b
        a = activation(z)
    return a

# c) Create and evaluate the new neural network
network_input_size = 4
output_sizes = [12, 10, 3]
activation_funcs = [ReLU, ReLU, softmax]
layers = create_layers_4(network_input_size, output_sizes, activation_funcs)

# Test with random input
x = np.random.randn(network_input_size)
predict = feed_forward_4(layers, x)
print("Prediction shape:", predict.shape)  # Should print: Prediction shape: (3,)

# Load and plot the iris dataset
iris = datasets.load_iris()

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)

# c) Loop over the iris dataset and evaluate the network for each data point
for x in iris.data:
    prediction = feed_forward_4(layers, x)
    print(prediction)

plt.savefig("iris_plot.pdf")
plt.show()
