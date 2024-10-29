
from utils import *
import autograd.numpy as anp 


if __name__ == "__main__":

    # Data setup
    x = anp.random.rand(1000,1)
    beta_true = np.array([[1, -8, 16],]).T
    func = Polynomial(*beta_true)
    y = func(x)

    # Choose method:
    methods = ["PlaneGD", "AdaGrad", "RMSProp", "Adam"]
    method = methods[0]; GD_SGD = "GD"

    # Parameters:
    epochs = 1000; batch_size = 50
    lmbdas = np.logspace(-10, 1, 10)
    learning_rates = np.logspace(np.log10(7.5e-2), np.log10(0.5), 10)

    learning_rates = np.logspace(-10, 1, 10)
    # Saving parameters:
    file_path = f"../Data/Regression/{method}"
    os.makedirs(file_path, exist_ok=True)
    size = len(lmbdas)
    filename_OLS = file_path + f"/OLS{size}x{size}"
    filename_Ridge = file_path + f"/Ridge{size}x{size}"
    overwrite = False

    # Just making sure 
    assert len(lmbdas) == len(learning_rates), f"The length og 'lmbdas' need to be the same as 'learning_rates'."

    # Analyzer setup
    analyzer_OLS = DescentAnalyzer(x, y, method, 3, epochs,
        batch_size=batch_size,
        GD_SGD=GD_SGD, princt_percentage=False)
    
    analyzer_Ridge = DescentAnalyzer(x, y, method, 3, epochs,
        batch_size=batch_size,
        GD_SGD=GD_SGD, princt_percentage=False)
    
    # Ols gradient:
    gradient_OLS = AutoGradCostFunction(CostOLS)

    # To store lambda values for saving
    analyzer_Ridge["lambda"] = []
    start_time = time.time()

    counter = 0
    for i in range(len(learning_rates)):
        analyzer_OLS.run_analysis(gradient_OLS, learning_rates[i])
        analyzer_Ridge["lambda"].append(lmbdas[i])
        analyzer_Ridge["learning_rates"].append(learning_rates[i])

        for j in range(len(lmbdas)):
            cost_function_ridge = CostRidge(lmbdas[j])
            gradient_Ridge = AutoGradCostFunction(cost_function_ridge)

            analyzer_Ridge.run_analysis(gradient_Ridge, learning_rates[i], save_learning_rate=False)

            counter += 1

            print(f"Analyzing, {counter/(len(lmbdas) * len(learning_rates))*100:.1f}%, duration: {time.time() - start_time:.1f}s", end="\r")

    
    analyzer_OLS.save_data(filename_OLS, overwrite=overwrite)
    analyzer_Ridge.save_data(filename_Ridge, overwrite=overwrite)
    print(f"Analyzing, 100%, duration: {time.time() - start_time:.1f}s            ")
