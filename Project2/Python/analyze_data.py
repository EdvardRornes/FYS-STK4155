from utils import * 
import seaborn as sns


N = 10
method = "PlaneGD"
file_path = f"../Data/Regression/{method}"; index = 2

with open(f"{file_path}/OLS{N}x{N}_{index}.pkl", 'rb') as f:
    data_OLS = pickle.load(f)

with open(f"{file_path}/Ridge{N}x{N}_{index}.pkl", 'rb') as f:
    data_Ridge = pickle.load(f)

lmbdas = data_Ridge["lambda"]
learning_rates = data_Ridge["learning_rates"]

# True data:

x = np.random.rand(1000,1)
beta_true = np.array([[1, -8, 16],]).T
func = Polynomial(*beta_true)
y = func(x)

############## OLS MSE ##############
OLS_MSE = []

for i in range(len(data_OLS["thetas"])):

    theta = data_OLS["thetas"][i][0]

    if np.isnan(np.sum(theta)):
        OLS_MSE.append(1)
    else:
        y_OLS = Polynomial(*theta)
        y_OLS = y_OLS(x)
        
        OLS_MSE.append(mean_squared_error(y, y_OLS))

############## Ridge MSE ##############
Ridge_MSE = np.zeros((len(learning_rates), len(lmbdas)))
counter = 0
N = len(data_Ridge["thetas"])
for i in range(len(learning_rates)):
    for j in range(len(lmbdas)):
        
        
        theta = data_Ridge["thetas"][counter][0]
        counter += 1
        if np.isnan(np.sum(theta)):
            Ridge_MSE[i,j] = 1

        else:
            y_Ridge = Polynomial(*theta)
            y_Ridge = y_Ridge(x)

            Ridge_MSE[i,j] = mean_squared_error(y, y_Ridge)


############# Plotting #############
fig, ax = plt.subplots(1, figsize=(12,7))
ax.plot(learning_rates, OLS_MSE)
ax.set_xlabel(r"$\eta$")
# ax.set_xlim(0, 0.6)
ax.set_yscale("log")
# ax.set_ylim(0, 1)
ax.set_ylabel(r"MSE")
ax.set_title("Plane GD using OLS cost function")


tick = ticker.ScalarFormatter(useOffset=False, useMathText=True)
tick.set_powerlimits((0,0))

xtick_labels = [f"{l:.1e}" for l in lmbdas]
ytick_labels = [f"{l:.1e}" for l in learning_rates]

fig, ax = plt.subplots(figsize = (12, 7))
sns.heatmap(Ridge_MSE, ax=ax, cmap="viridis", annot=True, xticklabels=xtick_labels, yticklabels=ytick_labels)
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'$\eta$')
ax.set_title("Plane GD using Ridge cost function")
plt.tight_layout()
plt.show()