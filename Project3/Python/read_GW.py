import h5py 
import pandas as pd 
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt 
import h5py
import pandas as pd

from gwosc.datasets import find_datasets
from gwosc.datasets import event_gps
from gwosc.locate import get_event_urls
import requests
from gwpy.time import from_gps
import time 
import sys 


def read_save_GW(filename_n_path:str, gravitational_wave_end_UTC:str, duration:float, max_time_around_event:float, store_as=None):
    """
    Reads gravitational wave data from hdf5 files. 
    
    Assumptions:
    * Assumes data is stored on the form:

            data.keys() = ["meta", "quality", "strain"]
    
            data["meta"] = {"UTCstart": x,
                            "Duration", x, ...}
    
            data["strain"]["Strain"] = actual data
    
    * Assumes that the duration starts before the event time.

    Positional Arguments:
    * filename_n_path:                  filname and path
    * gravitational_wave_end_UTC:       when the gravitational event ended 
    * duration:                         duration of event
    * max_time_around_event:            max amount of time to crop the data around the event

    Keyword Arguments:
    * store_as (str):                   filename for where to save  

    
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
        if isinstance(gravitational_wave_end_UTC, str):
            t_event_end = datetime.strptime(gravitational_wave_end_UTC, "%Y-%m-%dT%H:%M:%S").timestamp()
        else: #assumes gravitational_wave_end_UTC is datetime.datetime
            t_event_end = gravitational_wave_end_UTC.timestamp()

        if float(t_event_end) > t[-1]:
            raise TypeError(f"Event start time, {gravitational_wave_end_UTC} is outside data.")
        
        t_event_end_index = np.argmin(np.abs(t - t_event_end))
        t_event_start = np.max([t_event_end - duration, t[0]])
        t_event_start_index = np.argmin(np.abs(t - t_event_start))

        # Slicing data randomly
        max_time_around_event_one_side = max_time_around_event / 2
        
        # Random cut of data:
        max_time_around_event_one_left = np.min([abs(t[t_event_start_index] - t[0]), 
                                                 max_time_around_event_one_side])
        max_time_around_event_one_right = np.min([max_time_around_event_one_side, 
                                                  abs(t[-1] - t[t_event_end_index])])
        
        random_crop_left = t[t_event_start_index] - np.random.random(1)[0] * max_time_around_event_one_left
        random_crop_right = t[t_event_end_index] + np.random.random(1)[0] * max_time_around_event_one_right

        random_crop_left_index = np.argmin(np.abs(t - random_crop_left))
        random_crop_right_index = np.argmin(np.abs(t - random_crop_right))
        
        y = np.random.randint(0,2)
        if y== 0:
            noise = data[random_crop_right_index:]
        else:
            noise = data[0:random_crop_left_index]


        data = data[random_crop_left_index:random_crop_right_index]
        N = len(data); N = np.min([len(noise), N])
        noise = noise[0:N]

        t = t[random_crop_left_index:random_crop_right_index]

        t_event_start_index = np.argmin(np.abs(t - t_event_start))
        t_event_end_index = np.argmin(np.abs(t - t_event_end))
        new_data = {"data" :                data,
                    "t":                    t - t_zero, 
                    "data_event_indices" :  [t_event_start_index, t_event_end_index],
                    "UTC_event_end_time":   t_event_end,
                    "dt" :                  dt,
                    "duration" :            duration,
                    "noise":                noise,
                    "detector":             file["meta"]["Detector"][()].decode("utf-8")}
    
    if not(store_as is None):
        if "." in store_as:
            print("I choose my own file endings, and I choose pickle (Rick)!")
            store_as = store_as.split(".")[0]

        filename = store_as.split("/")[-1]
        filepath = store_as.split("/")[:-1]; filepath = "".join(filepath) + "/"
        
        if not "PickleFiles" in filepath:
            filepath += "PickleFiles/"
        
        store_as = filepath + filename
            
        pd.to_pickle(new_data, f"{store_as}.pkl")
        print(f"Stored croped data from {filename_n_path}.hdf5 as {store_as}.pkl.")

    return new_data

def ask_for_event():
    all_events = find_datasets(type="event"); N = len(all_events)

    print("Type event on the form 'GW150914' to read specifically.")
    
    x = input("Random event and detector (y/n)? ")
    if x.upper() in ["Y", "YES"]:
        
        index = np.random.randint(0, N)
        event_name = all_events[index] 
        return event_name, None
    
    elif x.upper() in ["", "NO"]:

        x = input("Choose detector (y/n)? ")
        if x.upper() in ["Y", "YES"]:
            while True:
                x = input("Detector ('H1', 'L1' or 'V1'): ")
                if x.upper() in ["H1", "L1", "V1"]:
                    detector_type = x.upper()
                    all_events = find_datasets(detector=detector_type, type="event"); N = len(all_events)

                    x = input("Random event? ")
                    if x.upper() in ["Y", "YES"]:
        
                        index = np.random.randint(0, N)
                        event_name = all_events[index]

                    else:
                        while True:
                            x = input("Event name: ")
                            y = input("Version (v1, v2 , v3 or v4): ")

                            x = x + f"-{y}"
                            if x in all_events:
                                return x, detector_type
                            
                            else:
                                print(f"Did not find {x}, try again.")

                    return event_name, detector_type
                else:
                    print(f"Did not recognize {x}, try again.")
        
        else:
            print(f"Okay so you DID want a random event?")
            time.sleep(5)

            text = "I just waited 5 seconds for nothing."
            for char in text:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.1)

            index = np.random.randint(0, N)
            event_name = all_events[index] 
            return event_name, None
    
    else:
        while True:
            event_name = x 
            detector_type = input("Detector ('H1', 'L1' or 'V1'): ")
            if detector_type.upper() in ["H1", "L1", "V1"]:
                all_events = find_datasets(detector=detector_type, type="event"); N = len(all_events)

                while True:
                    y = input("Version (v1, v2 , v3 or v4): ")
                    event_name = event_name + f"-{y}"
                    if event_name in all_events:
                        return event_name, detector_type
                    
                    else:
                        print(f"Did not find {event_name}, try again.")
                        event_name = input("Event name: ")

            else:
                print(f"Did not recognize {x}, try again.")
                x = input("Event name: ")


def ask_for_GW(folder_path=""):
    event_name, detector_type = ask_for_event()

    x = input("Download (y/n)? ")
    if x.upper() in ["Y", "YES"]:
        urls = get_event_urls(event_name)
        
        if not (detector_type is None): # Detector type is given, now limiting urls:
            urls_new = []
            for i in range(len(urls)):
                if detector_type in urls[i]:
                    urls_new.append(urls[i])
            
            if len(urls_new) == 0:  # No detector of this sort or for this event
                raise TypeError(f"Did not find the event '{event_name}' measured by detector {detector_type}.")
            
            if len(urls_new) != 1: # Multiple possible urls
                print("Found multiple possible links. Choose one of these (by python-index): ")
                for i in range(len(urls_new)-1):
                    print(urls_new[i])
                
                url = input(urls_new[-1] + " ")
                while True:
                    try:
                        url = int(url)
                        if url in range(0, len(urls_new)):
                            url = urls_new[url]
                            break
                        else:
                            url = input(f"Out of range(0, {len(urls_new)}), try again: ")
                        
                    except:
                        url = input("That was not a integer, try again: ")
                
                
            else:      # Only one url (great!)
                url = urls_new[0]
            
        else:   # Apparently doesn't matter what detector you use?
            url = urls[np.random.randint(0,len(urls))]

        response = requests.get(url)
        with open(f"{folder_path}{event_name}.hdf5", 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {folder_path}{event_name}.hdf5")
        
        return event_name
    
    else:
        if not ("." in event_name):
            event_name = event_name + ".hdf5"
        return event_name
        x = input("Filename: ")
        if not ("." in x):
            x = x + ".hdf5"
        
        return x

def analyze_and_store_GW(filenamePath_and_event_name:str, max_duration_crop_outside_event=60*2):
    """
    Filename and event name must be the same!
    """

    if "." in filenamePath_and_event_name:
        filenamePath_and_event_name = filenamePath_and_event_name[0:filenamePath_and_event_name.index(".")]
    print(f"Analyzing {filenamePath_and_event_name}.")
    print()

    while True:
        duration = input("Event duration (seconds): ")

        try:
            duration = float(duration)
            event_name = filenamePath_and_event_name.split("/")[-1]
            event_time = event_gps(event_name)
            event_time = from_gps(event_time)
            print(f"Found that the event happend at {event_time}.")
            try:
                read_save_GW(f"{filenamePath_and_event_name}.hdf5", event_time, duration, max_duration_crop_outside_event, store_as=filenamePath_and_event_name)
            except TypeError as e:
                error_message = str(e)
                print(f"A TypeError occurred: {error_message}") 

            return 
        except:
            print("Could not make a float of this, try again.")


def analyze_GW(folder_path="Data/"):

    if folder_path is None:
        while True:
            folder_path = input("Folder path? ")
            if not (folder_path[-1] == "/"):
                folder_path = folder_path + "/"
    
    event_name = ask_for_GW(folder_path=folder_path)
    analyze_and_store_GW(folder_path + event_name)

if __name__ == "__main__":
    # "GW150914"
    analyze_GW()

