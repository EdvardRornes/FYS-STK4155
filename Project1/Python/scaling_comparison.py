from utils import * 

################ Plot ################
# latex_fonts()
save = False; overwrite = False
folder = "Figures/Comparison"
scaling_type1 = "no_scaling"
scaling_type2 = "MINMAX"
scaling_type3 = "StandardScaling"
scaling_types = [scaling_type1, scaling_type2, scaling_type3]
cmap = plt.colormaps["tab10"]

################ Setup ################
deg_max = 25
lmbdas = [0.01]
N = 200; eps = 0.1
franke = Franke(100, 0.1)
data = [franke.x, franke.y, franke.z]
for scaling_type in scaling_types:

    ################ Training ################
    LASSO = PolynomialRegression(LASSO_default, deg_max, data, lmbdas=lmbdas, scaling=scaling_type)
    LASSO_MSE_train, LASSO_MSE_test = LASSO.MSE()
    LASSO_R2_train, LASSO_R2_test = LASSO.R2()
    LASSO_beta = LASSO.beta
    LASSO_degrees = LASSO.degrees

    Ridge = PolynomialRegression(Ridge_fit, deg_max, data, lmbdas=lmbdas, scaling=scaling_type)
    Ridge_MSE_train, Ridge_MSE_test = Ridge.MSE()
    Ridge_R2_train, Ridge_R2_test = Ridge.R2()
    Ridge_beta = Ridge.beta
    Ridge_degrees = Ridge.degrees

    OLS = PolynomialRegression(OLS_fit, deg_max, data, lmbdas=lmbdas, scaling=scaling_type)
    OLS_MSE_train, OLS_MSE_test = OLS.MSE()
    OLS_R2_train, OLS_R2_test = OLS.R2()
    OLS_beta = OLS.beta
    OLS_degrees = OLS.degrees

    for k in range(len(lmbdas)):
        ################ MSE-plot ################
        fig, axs = plt.subplots(2,2, figsize=(10,6))
        if scaling_type == "no_scaling":
            fig.suptitle(rf"No scaling, N={N}, $\epsilon = {eps}$")
        elif scaling_type == "MINMAX":
            fig.suptitle(rf"MIN-MAX scaling, N={N}, $\epsilon = {eps}$")
        elif scaling_type == "StandardScaling":
            fig.suptitle(rf"StandardScaling, N={N}, $\epsilon = {eps}$")
        else:
            raise ValueError(f"Huh?")

        axs[0,0].set_title("OLS")
        axs[0,0].plot(OLS_degrees, OLS_MSE_train[:,k], lw=2.5, color=cmap(k))
        axs[0,0].plot(OLS_degrees, OLS_MSE_test[:,k], color=cmap(k), lw=2.5, linestyle='--')
        axs[0,0].grid(); axs[1,1].set_xlim(1, deg_max)
        axs[0,0].set_ylabel(r'MSE')

        axs[0,1].set_title(rf"Ridge ($\lambda={lmbdas[k]:.1e}$)")
        axs[0,1].plot(Ridge_degrees, Ridge_MSE_train[:,k], lw=2.5, color=cmap(k))
        axs[0,1].plot(Ridge_degrees, Ridge_MSE_test[:,k], color=cmap(k), lw=2.5, linestyle='--')
        axs[0,1].grid(); axs[1,1].set_xlim(1, deg_max)

        axs[1,0].set_title(rf"LASSO ($\lambda={lmbdas[k]:.1e}$)")
        axs[1,0].plot(LASSO_degrees, LASSO_MSE_train[:,k], lw=2.5, color=cmap(k))
        axs[1,0].plot(LASSO_degrees, LASSO_MSE_test[:,k], color=cmap(k), lw=2.5, linestyle='--')
        axs[1,0].grid(); axs[1,1].set_xlim(1, deg_max)
        axs[1,0].set_ylabel(r'MSE')
        axs[1,0].set_xlabel(r'Degree')

        axs[1,1].set_title(rf"Comparison ($\lambda={lmbdas[k]:.1e}$)")
        line = axs[1,1].plot(OLS_degrees, OLS_MSE_train[:,k], label="OLS", lw=2.5)
        color = plt.gca().lines[-1].get_color()
        axs[1,1].plot(OLS_degrees, OLS_MSE_test[:,k], color=color, lw=2.5, linestyle='--')

        line = axs[1,1].plot(Ridge_degrees, Ridge_MSE_train[:,k], label="Ridge", lw=2.5)
        color = plt.gca().lines[-1].get_color()
        axs[1,1].plot(Ridge_degrees, Ridge_MSE_test[:,k], color=color, lw=2.5, linestyle='--')

        line = axs[1,1].plot(LASSO_degrees, LASSO_MSE_train[:,k], label="LASSO", lw=2.5)
        color = plt.gca().lines[-1].get_color()
        axs[1,1].plot(LASSO_degrees, LASSO_MSE_test[:,k], color=color, lw=2.5, linestyle='--')
        axs[1,1].grid(); axs[1,1].set_xlim(1, deg_max)
        axs[1,1].set_xlabel(r'Degree')
        

        plt.legend()
        plt.tight_layout()
        # plt.title(rf"RIDGE MSE")
        # plt.grid(True)

        if save:
            save_plt(f"{folder}/comparison_MSE_{scaling_type}", overwrite=overwrite, fig=fig)

plt.show()