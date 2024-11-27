import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from Old.oldFFNN import FFNNOLD

def test_FFNN():
    input_size = 2; hidden_layers = [4, 8, 16, 32, 16, 8, 4, 2]
    output_size = 1

    # Sample data
    N = 250; eps = 0.01
    franke = Franke(N, eps)
    x = franke.x; y = franke.y; z = franke.z

    epochs = 100; batch_size = 50
    adam = Adam()
    ffnn_relu_new = FFNN(input_size, hidden_layers, output_size, adam, "relu")
    ffnn_lrelu_new = FFNN(input_size, hidden_layers, output_size, adam, "lrelu")
    ffnn_sigmoid_new = FFNN(input_size, hidden_layers, output_size, adam, "sigmoid")
    
    ffnns_new = [ffnn_relu_new, ffnn_lrelu_new, ffnn_sigmoid_new]
    mse_results_new = {"ReLU": [], "Sigmoid": [], "Leaky ReLU": []}

    # Initialize FFNN
    ffnn_relu_old = FFNNOLD(input_size=input_size, hidden_layers=hidden_layers, output_size=output_size, activation='relu')
    ffnn_sigmoid_old = FFNNOLD(input_size=input_size, hidden_layers=hidden_layers, output_size=output_size, activation='sigmoid')
    ffnn_lrelu_old = FFNNOLD(input_size=input_size, hidden_layers=hidden_layers, output_size=output_size, activation='lrelu')

    ffnns_old = [ffnn_relu_old, ffnn_lrelu_old, ffnn_sigmoid_old]
    mse_results_old = {"ReLU": [], "Sigmoid": [], "Leaky ReLU": []}


    X = np.c_[x, y]
    z = minmax_scale(z)
    z_reshaped = z.reshape(-1, 1)
    X_train, X_test, z_train, z_test = train_test_split(X, z_reshaped, test_size=0.25)

    learning_rates = np.logspace(-5, -2, 7)

    for i in range(len(learning_rates)):
        for ffnn_new, ffnn_old, key in zip(ffnns_new, ffnns_old, mse_results_new.keys()):
            ffnn_new.learning_rate = learning_rates[i]
            ffnn_new.train(X_train, z_train, epochs=epochs, batch_size=batch_size)

            z_predict = ffnn_new.predict(X_test)
            mse_results_new[key].append(MSE(z_predict, z_test))

            mse_history_relu = ffnn_old.train(X_train, z_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rates[i])
            z_pred_relu = ffnn_old.predict(X_test)
            test_MSE_relu = MSE(z_pred_relu, z_test)
            mse_results_old[key].append(test_MSE_relu)

    
    fig, axs = plt.subplots(2, 2, figsize=(12,7))

    for ax, key in zip(axs.ravel(), mse_results_new.keys()):
        ax.plot(learning_rates, mse_results_old[key], label=f"Old FFNN ({key})", marker="o", color="r")
        ax.plot(learning_rates, mse_results_new[key], label=f"New FFNN ({key})", marker="o", color="g")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\eta$')
        ax.set_ylabel('MSE')
        ax.set_title(fr'MSE vs $\eta$ with {epochs} epochs')
        ax.legend()
        ax.grid()
    
    plt.show()

if __name__ == "__main__":
    test_FFNN()
