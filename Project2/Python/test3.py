import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers           # Use any regularizer (l1, l2, l1_l2)
from tensorflow.keras.models import Sequential      # Append layers to models
from tensorflow.keras.layers import Dense           # Define characteristics of a layer
from tensorflow.keras import optimizers             # Use any optimizer (SGD, Adam, RMSprop)
from utils import *

# Load breast cancer dataset
cancer = load_breast_cancer()

inputs = cancer.data              # Feature matrix (569 samples, 30 parameters)
outputs = cancer.target           # Label array (0 for benign, 1 for malignant)
labels = cancer.feature_names[:30]

print('The content of the breast cancer dataset is:')
print(labels)
print('-------------------------')
print("inputs  =  " + str(inputs.shape))
print("outputs =  " + str(outputs.shape))
print("labels  =  " + str(labels.shape))

# Reassign to shorter variable names
x = inputs
y = outputs

# Visualization of the dataset (correlation analysis)
plt.figure()
plt.scatter(x[:, 0], x[:, 2], s=40, c=y, cmap=plt.cm.Spectral)
plt.xlabel('Mean radius', fontweight='bold')
plt.ylabel('Mean perimeter', fontweight='bold')

plt.figure()
plt.scatter(x[:, 5], x[:, 6], s=40, c=y, cmap=plt.cm.Spectral)
plt.xlabel('Mean compactness', fontweight='bold')
plt.ylabel('Mean concavity', fontweight='bold')

plt.figure()
plt.scatter(x[:, 0], x[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
plt.xlabel('Mean radius', fontweight='bold')
plt.ylabel('Mean texture', fontweight='bold')

plt.figure()
plt.scatter(x[:, 2], x[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
plt.xlabel('Mean perimeter', fontweight='bold')
plt.ylabel('Mean compactness', fontweight='bold')
plt.show()

# Select features relevant to classification (texture, perimeter, compactness, symmetry)
temp1 = np.reshape(x[:, 1], (len(x[:, 1]), 1))
temp2 = np.reshape(x[:, 2], (len(x[:, 2]), 1))
X = np.hstack((temp1, temp2))
temp = np.reshape(x[:, 5], (len(x[:, 5]), 1))
X = np.hstack((X, temp))
temp = np.reshape(x[:, 8], (len(x[:, 8]), 1))
X = np.hstack((X, temp))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Convert labels to categorical for cross entropy
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

del temp1, temp2, temp

# Define tunable parameters
eta = np.logspace(-7, 1, 9)                   # Learning rates
lamda = 0.01                                   # Regularization parameter
n_layers = 4                                   # Number of hidden layers
n_neuron = np.logspace(0, 3, 4, dtype=int)     # Neurons per layer
epochs = 100                                   # Epochs
batch_size = 100                               # Samples per gradient update

# Define function to return Deep Neural Network model
def NN_model(inputsize, n_layers, n_neuron, eta, lamda):
    model = Sequential()
    
    for i in range(n_layers):
        if i == 0:
            model.add(Dense(n_neuron, activation='relu', kernel_regularizer=regularizers.l2(lamda), input_dim=inputsize))
        else:
            model.add(Dense(n_neuron, activation='relu', kernel_regularizer=regularizers.l2(lamda)))
    
    model.add(Dense(2, activation='softmax'))  # Output layer for binary classification
    sgd = optimizers.SGD(learning_rate=eta)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

# Initialize accuracy matrices
Train_accuracy = np.zeros((len(n_neuron), len(eta)))
Test_accuracy = np.zeros((len(n_neuron), len(eta)))

# Train model with different configurations
for i in range(len(n_neuron)):
    for j in range(len(eta)):
        DNN_model = NN_model(X_train.shape[1], n_layers, n_neuron[i], eta[j], lamda)
        DNN_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        Train_accuracy[i, j] = DNN_model.evaluate(X_train, y_train)[1]
        Test_accuracy[i, j] = DNN_model.evaluate(X_test, y_test)[1]

# Plot results
def plot_data(x, y, data, title=None):
    fontsize = 16
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest', vmin=0, vmax=1)
    
    cbar = fig.colorbar(cax)
    cbar.ax.set_ylabel('accuracy (%)', rotation=90, fontsize=fontsize)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            c = "${0:.1f}\\%$".format(100 * data[j, i])
            ax.text(x_val, y_val, c, va='center', ha='center')

    x = [str(i) for i in x]
    y = [str(i) for i in y]
    
    ax.set_xticklabels([''] + x)
    ax.set_yticklabels([''] + y)
    ax.set_xlabel('$\\mathrm{learning\\ rate}$', fontsize=fontsize)
    ax.set_ylabel('$\\mathrm{hidden\\ neurons}$', fontsize=fontsize)
    
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

plot_data(eta, n_neuron, Train_accuracy, 'training')
plot_data(eta, n_neuron, Test_accuracy, 'testing')


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Taken from week42 lecture notes and edited to support different activation functions
class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=50,
            n_categories=10,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0,
            activations=None):  # Add activations argument

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        # Default to ReLU for hidden and softmax for output if not specified
        self.activations = activations or ['relu', 'softmax']

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def activation_function(self, z, func_name):
        func_name = func_name.lower()
        if func_name == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif func_name == 'relu':
            return np.maximum(0, z)
        elif func_name == 'tanh':
            return np.tanh(z)
        elif func_name == 'softmax':
            exp_term = np.exp(z)
            return exp_term / np.sum(exp_term, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown activation function: {func_name}")

    def feed_forward(self):
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = self.activation_function(self.z_h, self.activations[0])

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
        self.probabilities = self.activation_function(self.z_o, self.activations[1])

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = self.activation_function(z_h, self.activations[0])

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        probabilities = self.activation_function(z_o, self.activations[1])
        return probabilities

    def backpropagation(self):
        error_output = self.probabilities - self.Y_data
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()




import autograd.numpy as np

class Scheduler:
    """
    Abstract class for Schedulers
    """

    def __init__(self, eta):
        self.eta = eta

    # should be overwritten
    def update_change(self, gradient):
        raise NotImplementedError

    # overwritten if needed
    def reset(self):
        pass


class Constant(Scheduler):
    def __init__(self, eta):
        super().__init__(eta)

    def update_change(self, gradient):
        return self.eta * gradient
    
    def reset(self):
        pass


class Momentum(Scheduler):
    def __init__(self, eta: float, momentum: float):
        super().__init__(eta)
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient):
        self.change = self.momentum * self.change + self.eta * gradient
        return self.change

    def reset(self):
        pass


class Adagrad(Scheduler):
    def __init__(self, eta):
        super().__init__(eta)
        self.G_t = None

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t += gradient @ gradient.T

        G_t_inverse = 1 / (
            delta + np.sqrt(np.reshape(np.diagonal(self.G_t), (self.G_t.shape[0], 1)))
        )
        return self.eta * gradient * G_t_inverse

    def reset(self):
        self.G_t = None


class AdagradMomentum(Scheduler):
    def __init__(self, eta, momentum):
        super().__init__(eta)
        self.G_t = None
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t += gradient @ gradient.T

        G_t_inverse = 1 / (
            delta + np.sqrt(np.reshape(np.diagonal(self.G_t), (self.G_t.shape[0], 1)))
        )
        self.change = self.change * self.momentum + self.eta * gradient * G_t_inverse
        return self.change

    def reset(self):
        self.G_t = None


class RMS_prop(Scheduler):
    def __init__(self, eta, rho):
        super().__init__(eta)
        self.rho = rho
        self.second = 0.0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero
        self.second = self.rho * self.second + (1 - self.rho) * gradient * gradient
        return self.eta * gradient / (np.sqrt(self.second + delta))

    def reset(self):
        self.second = 0.0


class Adam(Scheduler):
    def __init__(self, eta, rho, rho2):
        super().__init__(eta)
        self.rho = rho
        self.rho2 = rho2
        self.moment = 0
        self.second = 0
        self.n_epochs = 1

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        self.moment = self.rho * self.moment + (1 - self.rho) * gradient
        self.second = self.rho2 * self.second + (1 - self.rho2) * gradient * gradient

        moment_corrected = self.moment / (1 - self.rho**self.n_epochs)
        second_corrected = self.second / (1 - self.rho2**self.n_epochs)

        return self.eta * moment_corrected / (np.sqrt(second_corrected + delta))

    def reset(self):
        self.n_epochs += 1
        self.moment = 0
        self.second = 0
from utils import *
# from autograd import grad

def gradientOLS(X, y, beta):
    n=len(y)
    return 2.0/n*X.T @ (X @ (beta)-y)

def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2) # Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X

planeGD = GradientDescent(momentum=0)
test = beta_giver(planeGD)


N = 1000
x = 10*np.random.rand(N,1)
# x = np.linspace(-10,10,N)
beta_true = np.array([[1, 1/2, -1/18, 3, 1, 1],]).T
func = f(*beta_true)
y = func(x)
X = np.c_[np.ones((N,len(beta_true)-1)), x]
beta = np.random.randn(np.shape(X)[1],1)

test.gradient = gradientOLS
beta = test(X, y, 100, theta=beta)

print(beta)

f_train = f(*beta[0])
x_test = np.linspace(0,10, 10_000)

plt.plot(x_test, func(x_test), label="analytic")
plt.plot(x_test, f_train(x_test), label="train")
plt.legend()
plt.show()


n = 1000
X = np.c_[np.ones((n,4)), x]
# Hessian matrix
H = (2.0/n)* X.T @ X
# Get the eigenvalues
EigValues, EigVectors = np.linalg.eig(H)
print(f"Eigenvalues of Hessian Matrix:{EigValues}")

# beta_linreg = np.linalg.inv(X.T @ X + 1e-3*np.identity(np.shape(X))) @ X.T @ y
# print(beta_linreg)
beta = np.random.randn(5,1)

eta = 1.0/np.max(EigValues)
Niterations = 1000

for iter in range(Niterations):
    gradient = (2.0/n)*X.T @ (X @ beta-y)
    beta -= eta*gradient

print(beta)
f_train = f(*beta)
print(f_train.coeffs)
x_test = np.linspace(0,10, 10_000)

plt.plot(x_test, func(x_test), label="analytic")
plt.plot(x_test, f_train(x_test), label="train")
plt.legend()
plt.show()
