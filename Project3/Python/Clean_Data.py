import os
import pickle

#### This program simply removes some part of the unmerged results so that they take up less space

# Define the path where your data files are stored
data_path = "GW_Parameter_Search"

# # Loop through all files in the directory
for filename in os.listdir(data_path):
    # Check if the file matches the expected naming pattern and is a .pkl file
    if filename.startswith("Synthetic_GW_Parameter_Tuning_Results_") and filename.endswith(".pkl"):
        file_path = os.path.join(data_path, filename)
        
        # Load the .pkl file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Remove 'x_train' and 'x_test' from each result in the file
        for result in data:
            result.pop("x_train", None)  # Remove 'x_train' if it exists
            result.pop("x_test", None)   # Remove 'x_test' if it exists
        
        # Save the updated data back to the same file
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Processed and updated: {filename}")

print("All matching files have been processed.")

# Check each file in the directory
for filename in os.listdir(data_path):
    if filename.startswith("Synthetic_GW_Parameter_Tuning_Results_") and filename.endswith(".pkl"):
        file_path = os.path.join(data_path, filename)
        
        # Load the .pkl file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Verify each result
        for i, result in enumerate(data):
            # Check if 'x_train' and 'x_test' are gone
            if "x_train" in result or "x_test" in result:
                print(f"Error in {filename}, result {i}: 'x_train' or 'x_test' still exists.")
            else:
                print(f"{filename}, result {i}: Cleaned successfully.")

            # Optional: Inspect the rest of the keys in the result
            remaining_keys = list(result.keys())
            print(f"Remaining keys in result {i}: {remaining_keys}")
        
        print(f"Finished checking {filename}.\n")

print("Verification completed.")