
from utils import *
import autograd.numpy as anp 


if __name__ == "__main__":

    # Data setup
    N = 20
    x = anp.random.rand(N,1)
    y = Franke(N, 0.1)
    
    x = [y.x, y.y]
    y = y.z_without_noise
    # print(len(y))
    # exit()
    # beta_true = np.array([[1, -3.5, -5, 0.5, 1.2],]).T
    
    # # beta_true = np.random.randint(-1,1, size=(4,1))
    
    # x = anp.random.rand(N,1)
    # x = np.linspace(-2,2,N)
    # func = Polynomial(*beta_true)
    # y = func(x)

    # Choose method:
    methods = [PlaneGradient, Adagrad, RMSprop, Adam]
    methods_name = ["PlaneGradient", "Adagrad", "RMSprop", "Adam"]
    method_index = 0
    method = methods[method_index]; GD_SGD = "SGD"

    # Parameters:
    epochs = 100; batch_size = 10
    lmbdas = np.logspace(-4, 1, 10)
    learning_rates = np.logspace(-4, -1, 10)
    N_bootstraps = 2

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
    analyzer_OLS = DescentAnalyzer(x, y, methods_name[method_index], 5, epochs,
        batch_size=batch_size,
        GD_SGD=GD_SGD, princt_percentage=True)
    
    analyzer_Ridge = DescentAnalyzer(x, y, methods_name[method_index], 5, epochs,
        batch_size=batch_size,
        GD_SGD=GD_SGD, princt_percentage=True)
    
    # OLS gradient:
    gradient_OLS = AutoGradCostFunction(CostRidge, 2)
    # X = create_Design_Matrix(x, y, 4)
    # print(gradient_OLS(X, y, np.linspace(0,1,10), 0), "her") 
    analyzer_OLS.run_analysis(method, gradient_OLS, learning_rates, 0, N_bootstraps)

    # Ridge gradient:
    gradient_Rdige = AutoGradCostFunction(CostRidge, 2)
    # X_test = create_Design_Matrix(x, y, 5)
    # gradient_Rdige(X_test, y, np.linspace(0,1,5), )
    analyzer_Ridge.run_analysis(method, gradient_Rdige, learning_rates, lmbdas, N_bootstraps)

    analyzer_OLS.save_data(filename_OLS, overwrite=overwrite)
    analyzer_Ridge.save_data(filename_Ridge, overwrite=overwrite)
    
