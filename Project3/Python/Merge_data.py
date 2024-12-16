import os
import pickle
from collections import defaultdict

##### This program merges the files created by the NN's #####

# Define the input and output paths
input_path = "GW_Parameter_Search_V2"
output_path = "GW_Parameter_Search_V2"#"GW_Merged_Results"
time_steps = 5000
SNR = 30

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Collect all files
files = [f for f in os.listdir(input_path) if f.endswith('.pkl')]

# Organize files by epochs and boost values
organized_data = defaultdict(list)
for file in files:
    parts = file.split('_')
    try:
        # Extract epoch and boost
        epochs = int(parts[7].replace('epoch', ''))
        boost = float(parts[10].replace('boost', '').replace('.pkl', ''))
        
        # Add the file to organized_data
        organized_data[(epochs, boost)].append(file)
    except (IndexError, ValueError) as e:
        print(f"Skipping file {file} due to error: {e}")
        continue

# Function to merge data files
def merge_files(file_list, epochs, boost):
    super_dict = {}

    for file in file_list:
        with open(os.path.join(input_path, file), "rb") as f:
            data = pickle.load(f)
        
        # Extract lambda and eta values from the filename
        parts = file.split('_')
        try:
            reg_value = float(parts[8].replace('lamd', ''))
            lr = float(parts[9].replace('eta', ''))
        except (IndexError, ValueError) as e:
            print(f"Skipping file {file} due to error: {e}")
            continue
        
        # Add to the super dictionary
        if (reg_value, lr) not in super_dict:
            super_dict[(reg_value, lr)] = []

        super_dict[(reg_value, lr)].extend(data)

    # Convert tuples to a more descriptive dictionary format
    formatted_super_dict = {
        "epochs": epochs,
        "boost": boost,
        "data": {f"lambda_{reg_value}_eta_{lr}": sub_data for (reg_value, lr), sub_data in super_dict.items()}
    }
    
    # Save the merged file
    base_filename = f"Synthetic_GW_Merged_Results_timesteps{time_steps}_SNR{SNR}_epoch{epochs}_boost{boost:.1f}"
    output_file = os.path.join(output_path, f"{base_filename}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(formatted_super_dict, f)
    print(f"File {output_file} saved with {len(super_dict)} combinations merged.")

# Merge files by epochs and boost values
for (epochs, boost), file_list in organized_data.items():
    merge_files(file_list, epochs, boost)
