
from utils import *
import autograd.numpy as anp 
    
if __name__ == "__main__":

    # Data setup
    N = 20
    x = anp.random.rand(N,1)
    y = Franke(N, 0.1)
    
    x = [y.x, y.y]
    y = y.z

    # Choose method:
    methods = [PlaneGradient, Adagrad, RMSprop, Adam]
    methods_name = ["PlaneGradient", "Adagrad", "RMSprop", "Adam"]
    method_index = 3
    method = methods[method_index]; GD_SGD = "SGD"

    # Parameters:
    epochs = 10; batch_size = 4
    lmbdas = np.logspace(-10, 1, 25)
    learning_rates = np.logspace(np.log10(6.2e-3), np.log10(3.5e0), 25)
    
    # Varying learning rate
    learning_rates = [LearningRate(2, 2/learning_rates[i], N, batch_size, str(learning_rates[i])) for i in range(len(learning_rates))]

    N_bootstraps = 30

    # Saving parameters:
    file_path = f"../Data/Regression/{methods_name[method_index]}"
    os.makedirs(file_path, exist_ok=True)
    size = len(lmbdas)
    filename_OLS = file_path + f"/OLS{size}x{size}"
    filename_Ridge = file_path + f"/Ridge{size}x{size}"
    overwrite = False

    # Just making sure 
    assert len(lmbdas) == len(learning_rates), f"The length og 'lmbdas' need to be the same as 'learning_rates'."

    # Setting up optimizer:
    method = method()

    # Analyzer setup
    analyzer_OLS = DescentAnalyzer(x, y, 5, epochs,
        batch_size=batch_size,
        GD_SGD=GD_SGD)
    
    analyzer_Ridge = DescentAnalyzer(x, y, 5, epochs,
        batch_size=batch_size,
        GD_SGD=GD_SGD)

    ############## OLS ##############
    analyzer_OLS.run_analysis(method, CostRidge, learning_rates, 0, N_bootstraps)

    ############## Ridge ##############
    analyzer_Ridge.run_analysis(method, CostRidge, learning_rates, lmbdas, N_bootstraps)

    analyzer_OLS.save_data(filename_OLS, overwrite=overwrite)
    analyzer_Ridge.save_data(filename_Ridge, overwrite=overwrite)
    
