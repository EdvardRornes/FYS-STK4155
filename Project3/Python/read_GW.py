import h5py 
import pandas as pd 
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt 
import h5py
import pandas as pd

# Open the HDF5 file
filename_n_path = "Data/test.hdf5"
# with h5py.File(filename_n_path, 'r') as data:
#     # List all groups
#     print("her")
#     print("Keys: ", list(data.keys()))

#     # Access 'meta' group
#     meta_group = data['meta']
#     print("Meta Keys: ", list(meta_group.keys()))
#     for key in meta_group.keys():
#         print(f"{key}: {meta_group[key][()]}")

#     # Access 'quality' group
#     quality_group = data['quality']
#     print("Quality Keys: ", list(quality_group.keys()))
#     for key in quality_group.keys():
#         print(f"{key}: {quality_group[key]}")

#     # Access 'strain' group
#     strain_group = data['strain']
#     print("Strain Keys: ", list(strain_group.keys()))
#     strain_data = strain_group['Strain'][()]
#     print("Strain Data: ", len(strain_data))

def read_GW(filename_n_path:str, gravitational_wave_start_UTC:str, duration:float, max_time_around_event:float, store_as=None):
    """
    Reads gravitational wave data from hdf5 files. Assumes data is stored on the form:

    data.keys() = ["meta", "quality", "strain"]
    
    data["meta"] = {"UTCstart": x,
                    "Duration", x, ...}
    
    data["strain"]["Strain"] = actual data

    
    """
    with h5py.File(filename_n_path, 'r') as file:
        data = np.array(file["strain"]["Strain"][()])
        UTC_start_time = file["meta"]["UTCstart"][()].decode("utf-8")

        # Creating time-array
        T = float(file["meta"]["Duration"][()]); N = len(data)
        dt = 1/T
        dt = T / N 
        t = np.arange(0, T, dt)
        
        # Finding gravitational signal 
        t_zero = datetime.strptime(UTC_start_time, "%Y-%m-%dT%H:%M:%S").timestamp() #defining zero-point

        # Finding indices of start and stop of gravitational wave
        t = t + float(t_zero)
        t_event_start = datetime.strptime(gravitational_wave_start_UTC, "%Y-%m-%dT%H:%M:%S").timestamp()
        if float(t_event_start) > t[-1]:
            raise TypeError(f"Event start time, {gravitational_wave_start_UTC} is outside data.")
        
        t_event_start_index = np.argmin(np.abs(t - t_event_start))
        t_event_stop = t_event_start + duration
        t_event_stop_index = np.argmin(np.abs(t - t_event_stop))

        # Slicing data randomly
        max_time_around_event_one_side = max_time_around_event / 2
        
        # Random cut of data:
        max_time_around_event_one_left = np.min([abs(t[t_event_start_index] - t[0]), 
                                                 max_time_around_event_one_side])
        max_time_around_event_one_right = np.min([max_time_around_event_one_side, 
                                                  abs(t[-1] - t[t_event_stop_index])])
        
        random_time_left = t[t_event_start_index] - np.random.random(1)[0] * max_time_around_event_one_left
        random_time_right = t[t_event_stop_index] + np.random.random(1)[0] * max_time_around_event_one_right

        random_time_left_index = np.argmin(np.abs(t - random_time_left))
        random_time_right_index = np.argmin(np.abs(t - random_time_right))
        
        data = data[random_time_left_index:random_time_right_index]
        t = t[random_time_left_index:random_time_right_index]

        t_event_start_index = np.argmin(np.abs(t - t_event_start))
        t_event_stop_index = np.argmin(np.abs(t - t_event_stop))
        new_data = {"data" :                data,
                    "t":                    t - t_zero, 
                    "data_event_indices" :  [t_event_start_index, t_event_stop_index],
                    "UTC_start_time" :      t_zero,
                    "dt" :                  dt}
    
    if not(store_as is None):
        if "." in store_as:
            print("I choose my own file endings, and I choose pickle (Rick)!")
            store_as = store_as.split(".")[0]
        pd.to_pickle(new_data, f"{store_as}.pkl")
    return new_data


        


    
    # print(data)
    
filename_n_path = "Data/test.hdf5"
data = read_GW(filename_n_path, "2015-09-14T09:30:43", 1, 60, store_as="test.pkl")


data_event_indices = data["data_event_indices"]


plt.plot(data["t"], data["data"])
plt.plot(data["t"][data_event_indices[0]:data_event_indices[1]], data["data"][data_event_indices[0]:data_event_indices[1]], color="red")
plt.show()

