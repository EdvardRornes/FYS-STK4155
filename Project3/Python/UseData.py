import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to load the results
def load_results(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

# Path to the saved results
save_path = "GW_Parameter_Tuning_Results"
filename = "Synthetic_GW_Parameter_Tuning_Results_epoch10_lamd0.0001_eta0.01_boost0.5.pkl"  # Example filename, change as necessary
filepath = "C:/Users/edvar/Github/FYS-STK4155/Project3/Python/GW_Parameter_Tuning_Results/Synthetic_GW_Parameter_Tuning_Results_epoch10_lamd0.001_eta0.1_boost1.1.pkl"


# Load the results
results = load_results(filepath)

# Example: Plot predictions for the first set of results
plt.figure(figsize=(20, 12))

# Extract parameters for the first result
first_result = results[0]
epochs = first_result['epochs']
boost = first_result['boost']
learning_rate = first_result['learning_rate']
regularization = first_result['regularization']

plt.suptitle(fr"$\eta={learning_rate}$, $\lambda={regularization}$, Epochs={epochs}")

# Loop through the folds and plot the results
for fold, result in enumerate(results):
    plt.subplot(2, 3, fold + 1)
    plt.title(f"Round {fold + 1}")

    # Actual data
    x_train = np.array(result['x_train'])
    y_train = np.array(result['y_train'])
    test_labels = np.array(result['test_labels'])

    # Predictions
    predictions = np.array(result['predictions'])
    predicted_labels = (predictions > 0.5).astype(int)
    x_test = np.array(result['x_test'])  # Assuming same x_train for prediction

    print(x_test.shape, test_labels.shape)

    # Plot the data and predicted events
    plt.plot(x_train, y_train, label=f'Data {fold+1}', lw=0.5, color='b')
    plt.plot(x_test, test_labels, label=f"Solution {fold+1}", lw=1.6, color='g')

    # Highlight predicted events
    predicted_gw_indices = np.where(predicted_labels == 1)[0]
    if len(predicted_gw_indices) == 0:
        print("No gravitational wave events predicted.")
    else:
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
            plt.axvspan(x_train[group[0]], x_train[group[-1]], color="red", alpha=0.3, label="Predicted event" if i == 0 else "")

    plt.legend()

# Display the plot
plt.tight_layout()
plt.show()
