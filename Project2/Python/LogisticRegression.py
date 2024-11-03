from utils import *
from sklearn.datasets import load_breast_cancer

if __name__ == "__main__":

    # Data setup
    cancer = load_breast_cancer()
    x = cancer.data; y = cancer.target#.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target.reshape(-1, 1), random_state=1)
    N = len(x)
    
    # Choose method:
    methods_name = ["PlaneGradient", "Adagrad", "RMSprop", "Adam"]
    method_index = 0
    method = methods_name[method_index]; GD_SGD = "SGD"

    # Parameters:
    epochs = 10; batch_size = 50; size = 5
    lmbdas = np.logspace(-6, -6, size)
    learning_rates = np.logspace(np.log10(1e-4), np.log10(1e-1), size)
    
    # Varying learning rate
    learning_rates = [LearningRate(2, 2/learning_rates[i], N, batch_size, str(learning_rates[i])) for i in range(len(learning_rates))]

    costfunction = logistic_cost

    create_data(None, y, method, epochs, learning_rates, lmbdas, batch_size=batch_size, X=x, type_regression="Logistic", cost_function=costfunction, overwrite=True, N_bootstraps=1)
    data_OLS, data_Ridge = analyze_save_data(method, size, 0, type_regression="Logistic", ask_me_werd_stuff_in_the_terminal=True, plot=True, key="accuracy_train")
    print(data_Ridge["learning_rates"])