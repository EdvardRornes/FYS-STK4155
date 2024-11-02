from utils import *


if __name__ == "__main__":

    # Data setup
    N = 20
    x = anp.random.rand(N,1)
    y = Franke(N, 0.1)
    
    x = [y.x, y.y]
    y = y.z

    # Choose method:
    methods_name = ["PlaneGradient", "Adagrad", "RMSprop", "Adam"]
    method_index = 3
    method = methods_name[method_index]; GD_SGD = "SGD"

    # Parameters:
    epochs = 10; batch_size = 4
    lmbdas = np.logspace(-10, 1, 25)
    learning_rates = np.logspace(np.log10(6.2e-3), np.log10(3.5e0), 25)
    
    # Varying learning rate
    learning_rates = [LearningRate(2, 2/learning_rates[i], N, batch_size, str(learning_rates[i])) for i in range(len(learning_rates))]

    # create_data(x, y, method, epochs, learning_rates, lmbdas, batch_size=batch_size)
    analyze_save_data(method, 25, 0)