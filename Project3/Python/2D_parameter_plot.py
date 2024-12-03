import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from tqdm import tqdm
from utils import latex_fonts
import threading

latex_fonts()

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
        ylabel=r"$\eta$",
        on_click=None,
        log_cbar=False
        ):
    """
    Plots a 2D heatmap with lambda and eta as inputs, and adds interactivity for clicks.
    """
    cmap = 'plasma'
    if Reverse_cmap == True:
        cmap = 'plasma_r'
    fig, ax = plt.subplots(figsize = (12, 7))
    tick = ticker.ScalarFormatter(useOffset=False, useMathText=True)
    tick.set_powerlimits((0, 0))

    lambda_indices = np.array([True]*len(lambdas))
    eta_indices = np.array([True]*len(etas))

    if not (xlim is None):
        xmin = xlim[0]; xmax = xlim[1]
        lambda_indices = [i for i, l in enumerate(lambdas) if xmin <= l <= xmax]

    if not (ylim is None):
        ymin = ylim[0]; ymax = ylim[1]
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

    if log_cbar:
        value = np.log(value)
        # value = np.sqrt(value)

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
        if log_cbar:
            title += rf'. Color bar is logged.'
        plt.title(title)

    plt.xlabel(r'$\lambda$', fontsize=xaxis_fontsize or 12)
    plt.ylabel(ylabel, fontsize=yaxis_fontsize or 12)

    # Register the click event to trigger a new plot
    if on_click:
        fig.canvas.mpl_connect('button_press_event', on_click)

    plt.tight_layout()

    if savefig:
        plt.savefig(f'../Figures/{filename}.pdf')

    return fig, ax

def on_click(event, lambdas, etas, losses, epochs, boosts, unique_lambdas, unique_etas, plot_info):
    """
    Handle the click event on the plot.
    """
    # Check if the click is within the axes
    if event.inaxes:
        # Get the clicked coordinates (in axis space)
        x, y = event.xdata, event.ydata

        x = int(np.floor(x))
        y = int(np.floor(y))

        print(f"Clicked coordinates: x={x}, y={y}, Plot Info (Epoch, Boost): {plot_info}")

        # Find the closest parameter combination based on the click position
        lambda_idx = x
        eta_idx = y

        # Get the corresponding parameter values
        clicked_lambda = unique_lambdas[lambda_idx]
        clicked_eta = unique_etas[eta_idx]

        # Extract the epoch and boost directly from the passed tuple
        epoch, boost = plot_info

        # Filter data for the clicked combination of lambda, eta, epoch, and boost
        mask = (lambdas == clicked_lambda) & (etas == clicked_eta) & (epochs == epoch) & (boosts == boost)

        # Check if there is data for this specific combination
        if np.any(mask):
            print(f"Clicked Lambda: {clicked_lambda}, Eta: {clicked_eta}, Fig with Epoch: {epoch} & Boost: {boost}")

            filepath = f'{pkl_dir}/Synthetic_GW_Parameter_Tuning_Results_timesteps{time_steps}_SNR{SNR}_epoch{epoch}_lamd{clicked_lambda}_eta{clicked_eta}_boost{boost:.1f}.pkl'
                
            results = load_results(filepath)

            # Example: Plot predictions for the first set of results
            plt.figure(figsize=(20, 12))

            plt.suptitle(fr"$\eta={clicked_eta}$, $\lambda={clicked_lambda}$, Epochs$\,={epoch}$, $\phi={boost:.1f}$")

            # Loop through the folds and plot the results
            for fold, result in enumerate(results):
                plt.subplot(2, 3, fold + 1)
                plt.title(f"Round {fold + 1}")

                # Actual data
                x_test = np.linspace(0, 50, time_steps)
                y_test = np.array(result['y_test'])
                test_labels = np.array(result['test_labels'])

                # Predictions
                predictions = np.array(result['predictions'])
                predicted_labels = (predictions > 0.5).astype(int)

                # Plot the data and predicted events
                plt.plot(x_test, y_test, label=f'Data {fold+1}', lw=0.5, color='b')
                plt.plot(x_test, test_labels, label=f"Solution {fold+1}", lw=1.6, color='g')

                # Highlight predicted events
                predicted_gw_indices = np.where(predicted_labels == 1)[0]
                if len(predicted_gw_indices) != 0:
                    threshold = 2
                    grouped_gw_indices = []
                    current_group = [predicted_gw_indices[0]]

                    for i in range(1, len(predicted_gw_indices)):
                        if predicted_gw_indices[i] - predicted_gw_indices[i - 1] <= threshold:
                            current_group.append(predicted_gw_indices[i])
                        else:
                            grouped_gw_indices.append(current_group)
                            current_group = [predicted_gw_indices[i]]

                    grouped_gw_indices.append(current_group)
                    for i, group in zip(range(len(grouped_gw_indices)), grouped_gw_indices):
                        plt.axvspan(x_test[group[0]], x_test[group[-1]], color="red", alpha=0.3, label="Predicted event" if i == 0 else "")

                plt.legend()

            # Display the plot
            plt.tight_layout()
            plt.show()
            if save_option.lower() == 'y':
                save_fig = input("Would you like to save the previously generated figure? y/n\n")
                if save_fig.lower() == 'y' and save_fig is not None:
                    plt.savefig(f'../Figures/Synthetic_GW_Results_timesteps{time_steps}_SNR{SNR}_epoch{epoch}_lamd{clicked_lambda}_eta{clicked_eta}_boost{boost:.1f}.pdf')


# Function to load the results
def load_results(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

save_option = input("Would you like to be prompted to save files? y/n\nNB!: If you choose yes, THE TERMINAL WILL CRASH if you do not give the later prompts an answer!! \n")

# Directory containing the .pkl files
pkl_dir = "GW_Parameter_Tuning_Results"
time_steps = 5000
SNR = 100

# Containers for the parameters and results
lambdas = []
etas = []
losses = []
epochs = []
boosts = []

# Get all .pkl files in the directory
pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith(".pkl")]

# for filename in pkl_files:
#     file_path = os.path.join(pkl_dir, filename)
#     if os.path.getsize(file_path) == 0:
#         print(f"File {filename} is empty.")
#         continue
#     # Proceed with loading
#     with open(file_path, "rb") as f:
#         results = pickle.load(f)

# Initialize the progress bar with the total number of files
with tqdm(total=len(pkl_files), desc="Loading .pkl files", ncols=100) as pbar:
    for filename in pkl_files:
        with open(os.path.join(pkl_dir, filename), "rb") as f:
            results = pickle.load(f)
            # Group results by parameter combinations
            grouped_results = {}
            for result in results:
                key = (result["epochs"], result["boost"], result["regularization"], result["learning_rate"])
                if key not in grouped_results:
                    grouped_results[key] = []
                grouped_results[key].append(result["loss"])  # Append fold loss for this parameter combination
            
            # Average losses across folds
            for (epoch, boost, reg, lr), fold_losses in grouped_results.items():
                if reg == 1.0:  # Skip bad lambdas for better cmap
                    continue
                lambdas.append(reg)
                etas.append(lr)
                epochs.append(epoch)
                boosts.append(boost)
                losses.append(np.mean(fold_losses))  # Average the losses for all folds

        # Update the progress bar after processing each file
        pbar.update(1)

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

        # Compute mean loss across folds for each (lambda, eta)
        unique_combinations = set(zip(filtered_lambdas, filtered_etas))
        for lam, eta in unique_combinations:
            # Get losses corresponding to this (lambda, eta) pair
            combination_mask = (filtered_lambdas == lam) & (filtered_etas == eta)
            mean_loss = np.mean(filtered_losses[combination_mask])  # Average across folds

            # Populate the grid
            lambda_idx = np.where(unique_lambdas == lam)[0][0]
            eta_idx = np.where(unique_etas == eta)[0][0]
            loss_grid[eta_idx, lambda_idx] = mean_loss

        # Generate the plot with the current epoch and boost in the title
        title = fr"Mean loss for Epoch$\,={epoch}$, $\phi={boost}$"
        fig, ax = plot_2D_parameter_lambda_eta(
            lambdas=unique_lambdas,
            etas=unique_etas,
            value=loss_grid,
            title=title,
            x_log=True,
            y_log=True,
            Reverse_cmap=False,
            annot=True,
            savefig=False,
            filename=f"Loss_Epoch{epoch}_Boost{boost}",  # Save plots with unique filenames
            on_click=lambda event, plot_info=(epoch, boost): on_click(event, lambdas, etas, losses, epochs, boosts, unique_lambdas, unique_etas, plot_info),  # Pass epoch and boost as tuple
            log_cbar=True
        )
print("Click on one of the grids to plot the results for the given parameter combination :)")
plt.show()


