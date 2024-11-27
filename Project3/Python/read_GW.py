import h5py 
import pandas as pd 

# file_path = 'Data/test.hdf5'
# df = pd.read_hdf(file_path)  # Replace 'your_dataset_name' with the appropriate key
# print(df)

file_path = 'Data/test.hdf5'
# with h5py.File(file_path, 'r') as hdf:
#     data = {}
#     for key in hdf.keys():
#         data[key] = hdf
#     print("Keys: ", list(hdf.keys()))

data = h5py.File(file_path, "r")
print(data.keys())
print(data["quality"])

strain = data["strain"]


import h5py

# Open the HDF5 file
# file_path = 'path_to_your_file.hdf5'
with h5py.File(file_path, 'r') as data:
    # List all groups
    print("her")
    print("Keys: ", list(data.keys()))

    # Access 'meta' group
    meta_group = data['meta']
    print("Meta Keys: ", list(meta_group.keys()))
    for key in meta_group.keys():
        print(f"{key}: {meta_group[key][()]}")

    # Access 'quality' group
    quality_group = data['quality']
    print("Quality Keys: ", list(quality_group.keys()))
    for key in quality_group.keys():
        print(f"{key}: {quality_group[key]}")

    # Access 'strain' group
    strain_group = data['strain']
    print("Strain Keys: ", list(strain_group.keys()))
    strain_data = strain_group['Strain'][()]
    print("Strain Data: ", len(strain_data))
