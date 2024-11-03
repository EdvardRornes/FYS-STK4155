
from utils import *
import autograd.numpy as anp 



def create_data(x:np.ndarray, method:str, epochs:int, learning_rates:list, lmbda:list, batch_size=None, N_bootstraps=30, overwrite=False) -> None:

    # Choose method:
    methods = [PlaneGradient, Adagrad, RMSprop, Adam]
    methods_name = ["PlaneGradient", "Adagrad", "RMSprop", "Adam"]
    methods_name_upper = ["PLANEGRADIENT", "ADAGRAD", "RMSPROP", "ADAM"]
    if method.upper() in methods_name_upper:
        method_index = methods_name_upper.index(method.upper())
    else:
        raise TypeError(f"What is '{method}'?")
    
    method = methods[method_index]
    GD_SGD = "SGD"
    if batch_size is None:
        GD_SGD = "GD"

    # Saving parameters:
    file_path = f"../Data/Regression/{methods_name[method_index]}"
    os.makedirs(file_path, exist_ok=True)
    size = len(lmbdas)
    filename_OLS = file_path + f"/OLS{size}x{size}"
    filename_Ridge = file_path + f"/Ridge{size}x{size}"

    # Making sure there are as many lmdas as learning_rates
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

def analyze_save_data(method:str, size:int, index:int, key="MSE_train"):
    
    
    methods = ["PlaneGradient", "Adagrad", "RMSprop", "Adam"]
    methods_name_upper = ["PLANEGRADIENT", "ADAGRAD", "RMSPROP", "ADAM"]
    if method.upper() in methods_name_upper:
        method_index = methods_name_upper.index(method.upper())
    else:
        raise TypeError(f"What is '{method}'?")
    
    method = methods[method_index]

    file_path = f"../Data/Regression/{method}"

    with open(f"{file_path}/OLS{size}x{size}_{index}.pkl", 'rb') as f:
        data_OLS = pickle.load(f)

    with open(f"{file_path}/Ridge{size}x{size}_{index}.pkl", 'rb') as f:
        data_Ridge = pickle.load(f)

    lmbdas = data_Ridge["lambdas"]
    learning_rates = data_Ridge["learning_rates"]

    learning_rates = [float(x) for x in learning_rates]

    OLS_MSE = data_OLS[key]
    Ridge_MSE = data_Ridge[key]

    ############# Plotting #############
    fig, ax = plt.subplots(1, figsize=(12,7))
    ax.plot(learning_rates, OLS_MSE)
    ax.set_xlabel(r"$\eta$")
    ax.set_yscale("log")
    ax.set_ylabel(r"MSE")
    ax.set_title(f"{method} using OLS cost function")


    tick = ticker.ScalarFormatter(useOffset=False, useMathText=True)
    tick.set_powerlimits((0,0))

    xtick_labels = [f"{l:.1e}" for l in lmbdas]
    ytick_labels = [f"{l:.1e}" for l in learning_rates]

    fig, ax = plt.subplots(figsize = (12, 7))
    sns.heatmap(Ridge_MSE, ax=ax, cmap="viridis", annot=True, xticklabels=xtick_labels, yticklabels=ytick_labels)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\eta$')
    ax.set_title(f"{method} using Ridge cost function")
    plt.tight_layout()
    plt.show()

    # Saving
    save = input("Save (y/n)? ")
    if save.upper() in ["Y", "YES", "YE"]:
        while True:
            only_less_than = input("Exclude values less than: ")
            plot_2D_parameter_lambda_eta(lmbdas, learning_rates, Ridge_MSE, only_less_than=float(only_less_than))
            plt.show()

            happy = input("Happy (y/n)? ")
            if happy.upper() in ["Y", "YES", "YE"]:
                title = input("Title: ") 
                filename = input("Filename: ")
                latex_fonts()
                plot_2D_parameter_lambda_eta(lmbdas, learning_rates, Ridge_MSE, only_less_than=float(only_less_than), title=title, savefig=True, filename=filename)
                plt.show()
                exit()
            
            elif happy.upper() in ["Q", "QUIT", "X"]:
                exit()
    
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
    
