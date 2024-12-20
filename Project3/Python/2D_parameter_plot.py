import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import latex_fonts, plot_2D_parameter_lambda_eta, on_click

latex_fonts()

save_option = input("Would you like to be prompted to save files? y/n\nNB! If you choose yes, THE TERMINAL WILL CRASH if you do not give the later prompts an answer!! \n")

# Define the input and output paths
### Remove RNN comment (along with the RNN SNR) to see the CNN results ###
pkl_dir = "CNN_Data/Merged_Results"  # Merged results path
SNR = 5
pkl_dir = "RNN_Data/Merged_Results"  # Merged results path
SNR = 100
time_steps = 5000

# Containers for the parameters and results
lambdas = []
etas = []
losses = []
epochs = []
boosts = []

# Get all .pkl files in the directory
pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith(".pkl")]

# Initialize the progress bar with the total number of files
with tqdm(total=len(pkl_files), desc="Loading .pkl files", ncols=100) as pbar:
    for filename in pkl_files:

        base_name = filename[:filename.index("timesteps")-1]
        file_path = os.path.join(pkl_dir, filename)
        
        # Load the merged results file
        with open(file_path, "rb") as f:
            merged_data = pickle.load(f)
        
        # Extract parameters from the filename (Epoch and Boost)
        parts = filename.split('_')

        epoch_index = None; boost_index = None
        for i in range(len(parts)):
            if "epoch" in parts[i]:
                epoch_index = i
            if "boost" in parts[i]:
                boost_index = i
    

        epochs_value = int(parts[epoch_index].replace('epoch', ''))
        boost_value = float(parts[boost_index].replace('boost', '').replace('.pkl', ''))

        # Iterate over the lambda-eta combinations in the merged data
        for lambda_eta_key, fold_data in merged_data["data"].items():
            # Split the lambda_eta_key to extract lambda and eta values
            lambda_value = float(lambda_eta_key.split('_')[1])
            eta_value = float(lambda_eta_key.split('_')[3])
            if lambda_value == 1: # These make the colorbars not very instructive due to terrible results so we skip them
                continue
            
            # Collect the loss values
            fold_losses = [item["loss"] for item in fold_data]  # Extract losses for this lambda-eta combination
            if len(fold_losses) > 0:
                lambdas.append(lambda_value)
                etas.append(eta_value)
                epochs.append(epochs_value)
                boosts.append(boost_value)
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
            savefig=True,
            filename=f"CNN_2D_Plot_Loss_Epoch{epoch}_Boost{boost}",
            on_click=lambda event, plot_info=(epoch, boost): on_click(event, lambdas, etas, epochs, boosts, unique_lambdas, 
                                                                      unique_etas, plot_info, time_steps, SNR, pkl_dir, save_option, filename_start=base_name),
            log_cbar=True
        )
print("Click on one of the grids to plot the results for the given parameter combination :)")
plt.show()