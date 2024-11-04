from utils import *
from sklearn.datasets import load_breast_cancer

"""
This codes stores and saves data calling it data_OLS and data_Ridge. This has nothing to do with OLS or Ridge, just a refering to the 
lambda=0, or lambda \neq 0 case.

The codes generates a large amount of data, but it can also be used to generate a 'single batch' of data, just uncomment/comment what is needed.
"""
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
    epochs = [10, 100]; batch_size = [50, 150]; size = 25
    N_batches = int(N / batch_size[0])

    lmbdas = np.logspace(-10, np.log10(1e0), size)
    learning_rates = np.logspace(np.log10(1e-10), np.log10(1e0), size)
    varying_learing_rates = [True, False]
    costfunction = LogisticCost() # Defaults to clipping for large/small values, uses sigmoid.
    
    ######## Single run ########
    # Varying learning rate
    # learning_rate = [LearningRate(2, 2/learning_rates[i], N, batch_size, str(learning_rates[i])) for i in range(len(learning_rates))]

    # print(f"Running {epochs} epochs and {N_batches} batches")

    # # create_data(None, y, method, epochs[0], learning_rate, lmbdas, batch_size=batch_size[0], X=x, type_regression="Logistic", cost_function=costfunction, overwrite=False, N_bootstraps=4)

    # data_OLS, data_Ridge = analyze_save_data(method, size, 4, type_regression="Logistic", ask_me_werd_stuff_in_the_terminal=True, plot=True, key="accuracy_test", xaxis_fontsize=14, yaxis_fontsize=14)


    ######## Large data ########
    start_time = time.time()
    for method_index in range(len(methods_name)):
        for varying_learing_rate in varying_learing_rates:
            for e in epochs:
                for b in batch_size:
                    method = methods_name[method_index]; GD_SGD = "SGD"
                    
                    if varying_learing_rate:                    
                        learning_rate = [LearningRate(2, 2/learning_rates[i], N, b, str(learning_rates[i])) for i in range(len(learning_rates))]
                    else:
                        learning_rate = learning_rates

                    create_data(None, y, method, e, learning_rates, lmbdas, batch_size=b, X=x, type_regression="Logistic", cost_function=costfunction, overwrite=False, N_bootstraps=4)
        
        print(f"{method} completed, time taken: {(time.time() - start_time)/60:1f} min.")

    print(f"Total duration: {(time.time() - start_time)/60:1f} min.") # 229.338515 min.

    ######## Reads data ########
    for method_index in range(len(methods_name)):
        method = methods_name[method_index]
        index = 0
        while True:
            try:
                data_OLS, data_Ridge = analyze_save_data(method, size, index, type_regression="Logistic", ask_me_werd_stuff_in_the_terminal=True, plot=True, key="accuracy_test", xaxis_fontsize=14, yaxis_fontsize=14)

                index += 1
            except:
                break