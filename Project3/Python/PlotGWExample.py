import pandas as pd
import matplotlib.pyplot as plt
from utils import latex_fonts
from datetime import datetime

# This example program illustrates how difficult it is to detect GWs without performing a proper treatment of the data first.

latex_fonts()

obj = pd.read_pickle("Data/PickleFiles/GW170104-v2.pkl")

indices = obj["data_event_indices"]
strain = obj["data"]
t = obj["t"]
event_time = obj["UTC_event_end_time"]  # Event's UTC end time

start_index = indices[0]  # Start of event
end_index = indices[-1]  # End of event

date_seconds = datetime.utcfromtimestamp(int(event_time+3600)) # +3600 for timezone change
date_fraction = event_time - int(event_time)
date = date_seconds.strftime("%Y-%m-%d %H:%M:%S") + f'{date_fraction:.1f}'[1:]

plt.figure(figsize=(14,6))
plt.title(f"GW170104 Event at time {date} UTC")
plt.plot(t-t[start_index], strain*1e18, color='b', label="Noise", lw=0.5)
plt.plot(t[start_index:end_index]-t[start_index], strain[start_index:end_index]*1e18, color='r', label="Event")
plt.xlim(t[0]-t[start_index], t[-1]-t[start_index])
plt.ylabel(r"Strain $[10^{-18}]$")
plt.xlabel("Time relative to event end [s]")
plt.legend(loc="upper right", framealpha=1)
plt.savefig('../Figures/GW170104GWEvent.pdf')
plt.show()
