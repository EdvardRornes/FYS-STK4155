from gwosc.datasets import find_datasets
from gwosc.datasets import event_gps, event_at_gps
from gwpy.time import from_gps

# Retrieve a list of datasets
datasets = find_datasets(type="event", detector="L1" )
print(datasets)

# Fetch metadata for a specific event
event_name = 'GW150914'  # Replace with your event of interest
event_info = event_gps(event_name)
print(event_info)
event_info = from_gps(event_info)
print(event_info)



from gwosc.locate import get_event_urls

# Get URLs for data files associated with the event
urls = get_event_urls(event_name)
print(urls)

# Download the data files
import requests

# for url in urls:
#     response = requests.get(url)
#     filename = url.split('/')[-1]
#     with open(filename, 'wb') as f:
#         f.write(response.content)
#     print(f"Downloaded {filename}")


from gwpy.timeseries import TimeSeries

# Define the GPS time and duration around the event
gps_time = event_gps(event_name)
gps_start = gps_time - 16  # 16 seconds before the event
gps_end = gps_time + 16    # 16 seconds after the event

# Fetch strain data from the GWOSC
strain = TimeSeries.fetch_open_data('H1', gps_start, gps_end, sample_rate=4096)

# Plot the strain data
plot = strain.plot()
plot.show()


['151008-v1', '151012.2-v1', '151116-v1', '161202-v1', '161217-v1', '170208-v1', '170219-v1', '170405-v1', '170412-v1', 
 '170423-v1', '170616-v1', '170630-v1', '170705-v1', '170720-v1', '190924_232654-v1', '191118_212859-v1', '191223_014159-v1', 
 '191225_215715-v1', '200114_020818-v1', '200121_031748-v1', '200201_203549-v1', '200214_224526-v1', '200214_224526-v2', 
 '200219_201407-v1', '200311_103121-v1', 'GRB051103-v1', 'GW150914-v1', 'GW150914-v2', 'GW150914-v3', 'GW151012-v1', 
 'GW151012-v2', 'GW151012-v3', 'GW151226-v1', 'GW151226-v2', 'GW170104-v1', 'GW170104-v2', 'GW170608-v1', 'GW170608-v2', 
 'GW170608-v3', 'GW170729-v1', 'GW170809-v1', 'GW170814-v1', 'GW170814-v2', 'GW170814-v3', 'GW170817-v1', 'GW170817-v2', 
 'GW170817-v3', 'GW170818-v1', 'GW170823-v1', 'GW190403_051519-v1', 'GW190408_181802-v1', 'GW190408_181802-v2', 'GW190412-v1', 
 'GW190412-v2', 'GW190412-v3', 'GW190412_053044-v4', 'GW190413_052954-v1', 'GW190413_052954-v2', 'GW190413_134308-v1', 
 'GW190413_134308-v2', 'GW190421_213856-v1', 'GW190421_213856-v2', 'GW190424_180648-v1', 'GW190424_180648-v2', 'GW190425-v1', 
 'GW190425-v2', 'GW190425_081805-v3', 'GW190426_152155-v1', 'GW190426_152155-v2', 'GW190426_190642-v1', 'GW190503_185404-v1', 
 'GW190503_185404-v2', 'GW190512_180714-v1', 'GW190512_180714-v2', 'GW190513_205428-v1', 'GW190513_205428-v2', 'GW190514_065416-v1', 
 'GW190514_065416-v2', 'GW190517_055101-v1', 'GW190517_055101-v2', 'GW190519_153544-v1', 'GW190519_153544-v2', 'GW190521-v1', 
 'GW190521-v2', 'GW190521-v3', 'GW190521_030229-v4', 'GW190521_074359-v1', 'GW190521_074359-v2', 'GW190527_092055-v1', 
 'GW190527_092055-v2', 'GW190531_023648-v1', 'GW190602_175927-v1', 'GW190602_175927-v2', 'GW190620_030421-v1', 'GW190620_030421-v2', 
 'GW190630_185205-v1', 'GW190630_185205-v2', 'GW190701_203306-v1', 'GW190701_203306-v2', 'GW190706_222641-v1', 'GW190706_222641-v2', 
 'GW190707_093326-v1', 'GW190707_093326-v2', 'GW190708_232457-v1', 'GW190708_232457-v2', 'GW190719_215514-v1', 'GW190719_215514-v2', 
 'GW190720_000836-v1', 'GW190720_000836-v2', 'GW190725_174728-v1', 'GW190727_060333-v1', 'GW190727_060333-v2', 'GW190728_064510-v1', 
 'GW190728_064510-v2', 'GW190731_140936-v1', 'GW190731_140936-v2', 'GW190803_022701-v1', 'GW190803_022701-v2', 'GW190805_211137-v1', 
 'GW190814-v1', 'GW190814-v2', 'GW190814_211039-v3', 'GW190828_063405-v1', 'GW190828_063405-v2', 'GW190828_065509-v1', 'GW190828_065509-v2', 
 'GW190909_114149-v1', 'GW190909_114149-v2', 'GW190910_112807-v1', 'GW190910_112807-v2', 'GW190915_235702-v1', 'GW190915_235702-v2', 
 'GW190916_200658-v1', 'GW190917_114630-v1', 'GW190924_021846-v1', 'GW190924_021846-v2', 'GW190925_232845-v1', 'GW190926_050336-v1', 
 'GW190929_012149-v1', 'GW190929_012149-v2', 'GW190930_133541-v1', 'GW190930_133541-v2', 'GW191103_012549-v1', 'GW191105_143521-v1', 
 'GW191109_010717-v1', 'GW191113_071753-v1', 'GW191126_115259-v1', 'GW191127_050227-v1', 'GW191129_134029-v1', 'GW191204_110529-v1', 
 'GW191204_171526-v1', 'GW191215_223052-v1', 'GW191216_213338-v1', 'GW191219_163120-v1', 'GW191222_033537-v1', 'GW191230_180458-v1', 
 'GW200105-v1', 'GW200105_162426-v2', 'GW200112_155838-v1', 'GW200115-v1', 'GW200115_042309-v2', 'GW200128_022011-v1', 'GW200129_065458-v1', 
 'GW200202_154313-v1', 'GW200208_130117-v1', 'GW200208_222617-v1', 'GW200209_085452-v1', 'GW200210_092254-v1', 'GW200216_220804-v1', 
 'GW200219_094415-v1', 'GW200220_061928-v1', 'GW200220_124850-v1', 'GW200224_222234-v1', 'GW200225_060421-v1', 'GW200302_015811-v1', 
 'GW200306_093714-v1', 'GW200308_173609-v1', 'GW200311_115853-v1', 'GW200316_215756-v1', 'GW200322_091133-v1', 'GW230529_181500-v1', 
 'blind_injection-v1']