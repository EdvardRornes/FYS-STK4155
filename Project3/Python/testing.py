import h5py

# Open the file to inspect its structure
with h5py.File("GWData/GW150914T090343_cut_3min.hdf5", "r") as f:
    # Print the structure of the file
    def print_structure(name, obj):
        print(f"{name}: {obj}")
    
    f.visititems(print_structure)  # This will list all datasets and groups
