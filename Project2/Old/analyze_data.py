from utils import * 
import seaborn as sns


N = 25
methods_name = ["PlaneGradient", "Adagrad", "RMSprop", "Adam"]
method = methods_name[3]
file_path = f"../Data/Regression/{method}"; index = 1

with open(f"{file_path}/OLS{N}x{N}_{index}.pkl", 'rb') as f:
    data_OLS = pickle.load(f)

with open(f"{file_path}/Ridge{N}x{N}_{index}.pkl", 'rb') as f:
    data_Ridge = pickle.load(f)

lmbdas = data_Ridge["lambdas"]
learning_rates = data_Ridge["learning_rates"]

learning_rates = [float(x) for x in learning_rates]

OLS_MSE = data_OLS["MSE_train"]
Ridge_MSE = data_Ridge["MSE_train"]

############# Plotting #############
fig, ax = plt.subplots(1, figsize=(12,7))
ax.plot(learning_rates, OLS_MSE)
ax.set_xlabel(r"$\eta$")
# ax.set_xlim(0, 0.6)
ax.set_yscale("log")
# ax.set_ylim(0, 1)
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
            plot_2D_parameter_lambda_eta(lmbdas, learning_rates, Ridge_MSE, only_less_than=float(only_less_than), title=title, savefig=True, filename=filename)
            plt.show()
            exit()
        
        elif happy.upper() in ["Q", "QUIT", "X"]:
            exit()
    
