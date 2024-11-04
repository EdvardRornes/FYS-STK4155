from utils import *

"""
The codes generates a large amount of data, but it can also be used to generate a 'single batch' of data, just uncomment/comment what is needed.
"""

if __name__ == "__main__":

    # Data setup
    N = 20
    x = anp.random.rand(N,1)
    y = Franke(N, 0.1)
    
    x = [y.x, y.y]
    y = y.z
    y = minmax_scale(y) # Scaling data

    # Choose method:
    methods_name = ["PlaneGradient", "Adagrad", "RMSprop", "Adam"]
    
    # Parameters:
    epochs = [10, 100]; batch_size = [5, 10]
    size = 25
    lmbdas = np.logspace(-10, 1, size)
    learning_rates = np.logspace(np.log10(1e-10), np.log10(1e0), size)

    varying_learing_rates = [True, False]

    ######## Single run ########
    learning_rate = [LearningRate(2, 2/learning_rates[i], N, b, str(learning_rates[i])) for i in range(len(learning_rates))]
    create_data(x, y, methods_name[0], epochs[0], learning_rate, lmbdas, batch_size=batch_size[0], overwrite=False, scaling="no scaling")
    
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

                    create_data(x, y, method, e, learning_rate, lmbdas, batch_size=b, overwrite=False, scaling="no scaling")
        
        print(f"{method} completed, time taken: {(time.time() - start_time)/60:1f} min.")

    print(f"Total duration: {(time.time() - start_time)/60:1f} min.") # 386.653960 min.

    ######## Reads data ########
    for method_index in range(len(methods_name)):
        method = methods_name[method_index]
        index = 0
        while True:
            try:
                data_OLS, data_Ridge = analyze_save_data(method, size, index, ask_me_werd_stuff_in_the_terminal=True, plot=True)
                index += 1
            except:
                break