import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

def plot_2D_parameter_lambda_eta(
        lambdas,
        etas,
        value,
        title=None,
        x_log=False,
        y_log=False,
        savefig=False,
        filename='',
        Reverse_cmap=False,
        annot=True,
        only_less_than=None,
        only_greater_than=None,
        xaxis_fontsize=None,
        yaxis_fontsize=None,
        xlim=None,
        ylim=None,
        ylabel=r"$\eta$"
        ):
    """
    Plots a 2D heatmap with lambda and eta as inputs.
    """
    cmap = 'plasma'
    if Reverse_cmap == True:
        cmap = 'plasma_r'
    fig, ax = plt.subplots(figsize=(12, 7))
    tick = ticker.ScalarFormatter(useOffset=False, useMathText=True)
    tick.set_powerlimits((0, 0))

    lambda_indices = np.array([True] * len(lambdas))
    eta_indices = np.array([True] * len(etas))

    if not (xlim is None):
        xmin = xlim[0]
        xmax = xlim[1]
        lambda_indices = [i for i, l in enumerate(lambdas) if xmin <= l <= xmax]

    if not (ylim is None):
        ymin = ylim[0]
        ymax = ylim[1]
        eta_indices = [i for i, e in enumerate(etas) if ymin <= e <= ymax]

    lambdas = np.array(lambdas)[lambda_indices]
    etas = np.array(etas)[eta_indices]
    value = value[np.ix_(eta_indices, lambda_indices)]

    if x_log:
        t_x = [u'${}$'.format(tick.format_data(lambd)) for lambd in lambdas]
    else:
        t_x = [fr'${lambd}$' for lambd in lambdas]

    if y_log:
        t_y = [u'${}$'.format(tick.format_data(eta)) for eta in etas]
    else:
        t_y = [fr'${eta}$' for eta in etas]

    if only_less_than is not None and only_greater_than is None:
        annot_data = np.where(value < only_less_than, np.round(value, 3).astype(str), "")
    elif only_greater_than is not None and only_less_than is None:
        annot_data = np.where(value > only_greater_than, np.round(value, 3).astype(str), "")
    else:
        annot_data = np.round(value, 3).astype(str) if annot else None

    sns.heatmap(
        data=value,
        ax=ax,
        cmap=cmap,
        annot=annot_data,
        fmt="",
        xticklabels=t_x,
        yticklabels=t_y,
    )

    # Adjust x and y tick labels
    ax.set_xticks(np.arange(len(lambdas)) + 0.5)
    ax.set_xticklabels([f"{float(label):.1e}" for label in lambdas], rotation=45, ha='right', fontsize=14)

    ax.set_yticks(np.arange(len(etas)) + 0.5)
    ax.set_yticklabels([f"{float(label):.1e}" for label in etas], rotation=0, fontsize=14)

    # Add title and labels
    if title:
        plt.title(title)

    plt.xlabel(r'$\lambda$', fontsize=xaxis_fontsize or 12)
    plt.ylabel(ylabel, fontsize=yaxis_fontsize or 12)

    plt.tight_layout()

    if savefig:
        plt.savefig(f'../Figures/{filename}.pdf')


# Directory containing the .pkl files
pkl_dir = "GW_Parameter_Tuning_Results"

# Containers for the parameters and results
lambdas = []
etas = []
losses = []
epochs = []
boosts = []

# Iterate through all .pkl files in the directory
for filename in os.listdir(pkl_dir):
    if filename.endswith(".pkl"):
        with open(os.path.join(pkl_dir, filename), "rb") as f:
            results = pickle.load(f)
            for result in results:
                # Skip lambdas == 1.0
                if result["regularization"] == 1.0:
                    continue
                lambdas.append(result["regularization"])
                etas.append(result["learning_rate"])
                # Average the 5 losses for the current result
                mean_loss = np.mean(result["loss"])  # Averaging the 5 losses
                losses.append(mean_loss)
                epochs.append(result["epochs"])
                boosts.append(result["boost"])

# Convert lists to numpy arrays
lambdas = np.array(lambdas)
etas = np.array(etas)
losses = np.array(losses)
epochs = np.array(epochs)
boosts = np.array(boosts)

# Get unique values for lambdas, etas, epochs, and boosts
unique_lambdas = np.unique(lambdas)
unique_etas = np.unique(etas)
unique_epochs = np.unique(epochs)
unique_boosts = np.unique(boosts)

# Loop over epochs and boosts to create plots
for epoch in unique_epochs:
    for boost in unique_boosts:
        # Filter data for the current epoch and boost
        mask = (epochs == epoch) & (boosts == boost)
        filtered_lambdas = lambdas[mask]
        filtered_etas = etas[mask]
        filtered_losses = losses[mask]

        # Skip if no data for this combination
        if len(filtered_losses) == 0:
            continue

        # Create a 2D grid for loss values
        loss_grid = np.full((len(unique_etas), len(unique_lambdas)), np.nan)

        # Populate the grid with averaged losses
        for lam, eta, loss in zip(filtered_lambdas, filtered_etas, filtered_losses):
            lambda_idx = np.where(unique_lambdas == lam)[0][0]
            eta_idx = np.where(unique_etas == eta)[0][0]
            loss_grid[eta_idx, lambda_idx] = loss

        # Generate the plot with the current epoch and boost in the title
        title = f"Loss for Epochs={epoch}, Boost={boost}"
        plot_2D_parameter_lambda_eta(
            lambdas=unique_lambdas,
            etas=unique_etas,
            value=loss_grid,
            title=title,
            Reverse_cmap=False,
            annot=True,
            savefig=False,
            filename=f"Loss_Epoch{epoch}_Boost{boost}"  # Save plots with unique filenames
        )

plt.show()
