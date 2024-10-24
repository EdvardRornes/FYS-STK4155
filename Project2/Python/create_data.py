
from utils import *
import autograd.numpy as anp 


if __name__ == "__main__":
    x = anp.random.rand(1000,1)
    beta_true = np.array([[1, -8, 16],]).T
    func = Polynomial(*beta_true)
    y = func(x)

    def CostOLS(X,y,theta):
            return anp.sum((y-X @ theta)**2)

        
    gradient1 = AutoGradCostFunction(CostOLS)
    lmbdas = [1e-1, 1e-5, 1e-10]
    lmbdas = np.linspace(1e-10, 1, 25)

    methods = ["PLANEGD", "ADAGRAD", "RMSPROP", "ADAM"]
    # Choose method:
    method = methods[0]
    epochs = [1000]*len(lmbdas); batch_size = [50] * len(lmbdas)
    learning_rates = [1, 0.5, 1e-1]
    learning_rates = np.linspace(1e-10, 1, 25)

    counter = 0

    start_time = time.time()

    batch_number = 1
    os.makedirs(f"../Data/Batch{batch_number}_{method}", exist_ok=True)
    for i in range(len(lmbdas)):
        for j in range(len(learning_rates)):
            cost_function_ridge = CostRidge(lmbdas[i])
            gradient2 = AutoGradCostFunction(cost_function_ridge)

            analyzer1 = DescentAnalyzer(x=x, y=y,
                optimizer=method,
                gradient=gradient1,
                degree=2,
                epochs=epochs,
                learning_rates=[learning_rates[j]]*len(lmbdas),
                batch_size=batch_size,
                GD_SGD='SGD', princt_percentage=False)
            

            analyzer2 = DescentAnalyzer(x=x, y=y,
                optimizer=method,
                gradient=gradient2,
                degree=2,
                epochs=epochs,
                learning_rates=[learning_rates[j]]*len(lmbdas),
                batch_size=batch_size,
                GD_SGD='SGD', princt_percentage=False)
            


            # Run the analysis
            analyzer1.run_analysis(); analyzer2.run_analysis

            # Save the data to a file, with overwrite option
            analyzer1.data["lambda"] = lmbdas[i]
            analyzer2.data["lambda"] = lmbdas[i]

            analyzer1.save_data(f"../Data/Batch{batch_number}_{method}/analysis_results{int(counter)}OLS.pkl", overwrite=True)
            analyzer2.save_data(f"../Data/Batch{batch_number}_{method}/analysis_results{int(counter)}Ridge.pkl", overwrite=True)
            counter += 1

            print(f"Analyzing, {counter/(len(lmbdas) * len(learning_rates))*100:.1f}%, duration: {time.time() - start_time:.1f}s", end="\r")

    print(f"Analyzing, 100%, duration: {time.time() - start_time:.1f}s            ")

    data_OLS = {}; data_Ridge= {}

    for i in range(counter):

        with open(f"../Data/Batch{batch_number}_{method}/analysis_results{i}Ridge.pkl", 'rb') as f:
            data_Ridge_tmp = pickle.load(f)
            data_Ridge_tmp = [data_Ridge_tmp["thetas"], data_Ridge_tmp["learning_rates"], data_Ridge_tmp["lambda"]]
            data_Ridge["str{i}"] = data_Ridge_tmp

        with open(f"../Data/Batch{batch_number}_{method}/analysis_results{i}OLS.pkl", 'rb') as f:
            data_OLS_tmp = pickle.load(f)
            data_OLS_tmp = [data_OLS_tmp["thetas"], data_OLS_tmp["learning_rates"], data_OLS_tmp["lambda"]]
            data_OLS["str{i}"] = data_OLS_tmp
    
    data = {"OLS": data_OLS,
            "Ridge": data_Ridge}
    
    with open(f"../Data/{method}_Batch{batch_number}.pkl", 'wb') as f:
        pickle.dump(data, f)
    analyzer = DescentAnalyzer(x=x, y=y,
                optimizer=method,
                gradient=gradient1,
                degree=2,
                epochs=epochs,
                learning_rates=[learning_rates[j]]*len(lmbdas),
                batch_size=batch_size,
                GD_SGD='SGD')
