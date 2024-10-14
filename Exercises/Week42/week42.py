import autograd.numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

def ReLU(z):
    return np.where(z > 0, z, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z."""
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Change axis to 1 for batch processing
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z."""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

# Exercise 1

np.random.seed(2024)

x = np.random.randn(2)  # network input. This is a single input with two features
W1 = np.random.randn(4, 2)  # first layer weights


# a) Given the shape (n, m) for the first layer weight matrix the input shape of the neural network is (m,). The output shape of the first layer is (n,).


# b) Define the bias of the first layer
b1 = np.random.randn(4)


# c) Compute intermediary z1
z1 = W1 @ x + b1


# d) Compute the activation a1
a1 = ReLU(z1)

# Confirm the activation is correct
sol1 = np.array([0.60610368, 4.0076268, 0.0, 0.56469864])
print("Activation a1 matches expected solution:", np.allclose(a1, sol1))  # Should print: True



# Exercise 2


# a) The input of the 2nd layer is the output of the first layer which has shape (4,).


# b) Define the weight and bias of the second layer
W2 = np.random.randn(8, 4)
b2 = np.random.randn(8)


# c) Compute intermediary z2 and activation a2
z2 = W2 @ a1 + b2
a2 = ReLU(z2)

# Confirm the activation shape
print("a2 shape matches the expected shape:", a2.shape == (8,))  # Should print: True



# Exercise 3


# a) Complete create_layers function
def create_layers(network_input_size, output_sizes):
    layers = []
    i_size = network_input_size
    for output_size in output_sizes:
        W = np.random.randn(output_size, i_size)
        b = np.random.randn(output_size)
        layers.append((W, b))
        i_size = output_size
    return layers


# b) Complete feed_forward function
def feed_forward_all_relu(layers, input):
    a = input
    for W, b in layers:
        z = W @ a + b
        a = ReLU(z)
    return a


# c) Create a network and evaluate it
input_size = 8
layer_output_sizes = [10, 16, 6, 2]
x = np.random.rand(input_size)
layers = create_layers(input_size, layer_output_sizes)
predict = feed_forward_all_relu(layers, x)
print(predict)  # Should print: Output shape: (2,)


# d) Any composition of linear function is just another linear function. 
# Thus with no activation functions we could just make one linear transformation which does not change the output.



# Exercise 4


# a) Complete feed_forward function which accepts activation arguments
def feed_forward(input, layers, activations):
    a = input
    for (W, b), activation in zip(layers, activations):
        z = W @ a + b
        a = activation(z)
    return a


# b) List of activation functions w/ two ReLU and one sigmoid
activations = [ReLU, sigmoid, ReLU]

# Evaluate the new neural network
network_input_size = 4
layer_output_sizes = [4, 6, 8]
layers = create_layers(network_input_size, layer_output_sizes)

x = np.random.randn(network_input_size)
output = feed_forward(x, layers, activations)
print(output)



# Exercise 5


# a) Complete create_layers_batch function
def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []
    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(layer_output_size, i_size)
        b = np.random.randn(layer_output_size)
        layers.append((W, b))
        i_size = layer_output_size
    return layers


# b) Make input matrix and complete feed_forward_batch
inputs = np.random.rand(1000, 4)  # 1000 samples, 4 features

def feed_forward_batch(inputs, layers, activations):
    a = inputs
    for (W, b), activation in zip(layers, activations):
        z = a @ W.T + b  # Adjusted matrix multiplication
        a = activation(z)
    return a


# c) Create and evaluate a neural network w/ 4 inputs and layers with output sizes 12, 10, 3 and activations ReLU, ReLU, softmax
network_input_size = 4
layer_output_sizes = [12, 10, 3]
activations = [ReLU, ReLU, softmax]

# Create layers
layers = create_layers_batch(network_input_size, layer_output_sizes)

# Perform feed forward on the batch of inputs
predictions = feed_forward_batch(inputs, layers, activations)
print("Predictions:", predictions)



# Exercise 6

iris = datasets.load_iris()

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
plt.savefig('iris_plot.pdf')
plt.show()
inputs = iris.data

# Since each prediction is a vector with a score for each of the three types of flowers,
# we need to make each target a vector with a 1 for the correct flower and a 0 for the others.
targets = np.zeros((len(iris.data), 3))
for i, t in enumerate(iris.target):
    targets[i, t] = 1

def accuracy(predictions, targets):
    one_hot_predictions = np.zeros(predictions.shape)

    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return np.mean(np.all(one_hot_predictions == targets, axis=1))  # accuracy score was not defined?


# a) Input size is the number of features in the dataset: 4. The output shape is the number of flowers: 3


# b) Create a network w/ 2 hidden layers w/ activation sigmoid and softmax for the first and 2nd.
activations = [sigmoid, softmax]
layers = create_layers_batch(4, [8, 3])  # 8 nodes in the hidden layer and 3 output nodes


# c) Evaluate the model on entire dataset
predictions = feed_forward_batch(inputs, layers, activations)


# d) Compute the accuracy of the model
print("Accuracy:", accuracy(predictions, targets))



# Exercise 7

from autograd import grad

def cost(input, layers, activations, target):
    predict = feed_forward_batch(input, layers, activations)
    return cross_entropy(predict, target)

def cross_entropy(predict, target):
    return np.sum(-target * np.log(predict))

# Mistake was made in the below on the website.
gradient_func = grad(
    cost, 1
)  # Taking the gradient wrt. the second input to the cost function


# a) The gradient of the cost function w.r.t. weights and biases should have shape (output_size, input_size) and (output_size,) respectively.


# b) Use gradient_func to take gradient of cross entropy w.r.t. weights and biases of the network
layers_grad = gradient_func(inputs, layers, activations, targets)  # Don't change this

# Check shapes of gradients
for (W, b), (W_g, b_g) in zip(layers, layers_grad):
    print(f"W shape: {W_g.shape}, b shape: {b_g.shape}")

# The grad function from autograd computes the derivative of the function w.r.t. its inputs, in this case the grad of the cross-entropy loss coming from the inputs.



# c) Implementing a simple gradient descent optimizer (Updated to evaluate the accuracy)
def train_network(
    inputs, layers, activations, targets, learning_rate=0.001, epochs=101
):
    accuracies = []  # List to store accuracies over epochs
    for i in range(epochs):
        layers_grad = gradient_func(inputs, layers, activations, targets)
        for (W, b), (W_g, b_g) in zip(layers, layers_grad):
            W -= learning_rate * W_g
            b -= learning_rate * b_g

        predictions = feed_forward_batch(inputs, layers, activations)
        accuracy_value = accuracy(predictions, targets)
        accuracies.append(accuracy_value)

    return layers, accuracies


# d) Batch gradient decent


# e) Train the network using gradient descent
epochs = 101
layers, training_accuracies = train_network(inputs, layers, activations, targets, epochs=epochs)
plt.figure(figsize=(10, 8))
plt.plot(range(0, epochs), training_accuracies, marker='o')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid()
plt.savefig('accuracy.pdf')
plt.show()
