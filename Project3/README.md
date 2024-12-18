# Project 3: Recurrent Neural Networks and Synthetic Gravitational Waves

## Abstract
Insert Abstract

## Structure

### Project
The project is `project3.pdf`.

### Python
This folder contains all the Python scripts used in the project.

#### Utility 
- `NNs.py`: Contains the Neural Network classes; RNN, KerasRNN and KerasCNN.
- `utils.py`: Contains utility functions used across all the scripts.

#### Data Generation and Merging
- `read_GW.py`: Downloads data surrounding a GW event from gwosc. Given the duration of the signal, this program automatically labels the data, saving it to the `Data` folder.
- `Clean_Data.py`: Utility file used to remove certain parts from the data which did not need to be saved.
- `Merge_data.py`: Merges parameter combinations of the learning rate and l2 regularization into larger files.
- `KerasRNN_Synthetic.py`: Performs a parameter scan with using Keras' RNN which is initialized in `NNs.py`. Saves the result to a chosen filepath, which is ignored by .gitignore, however thought to be merged using `Merge_data.py`.
- `RNN_Synthetic.py`: Same function as `KerasRNN_Synthetic.py`, only instead using the class `RNN`.
- `KerasRNN_Synthetic.py`: Same function as `KerasRNN_Synthetic.py`, only instead using the class `KerasRNN`.

#### Data Visualization 
- `2D_parameter_plot.py`: Uses data from `GW_Merged_Results/` to create 2D parameter plots. Each of these plots are interactive, where you can click on the grid and the results from the corresponding parameter combination test results will be shown. The input prompt gives you the option to save these generated figures, which will be automatically saved with the given parameter combination to `Figures`.
- `PlotGWExample.py`: A simple plot of the GW data downloaded from gwosc with `read_GW.py`.
- `WaveletTransform.py`: Shows a simple example of a wavelet transform of a synthetic GW event.

#### Data (Folder)
This folder contains the data saved from `read_GW.py`. Filenames are made suggestive.

#### CNN_data (Folder)
Contains data from parameter search using CNN. The folders here starting with `Merged_Results` are the merged results from running `Merge_data.py`after having created parameter-search data.

#### RNN_data (Folder)
Contains data from parameter search using `KerasRNN`and `RNN`. The folders here starting with `Merged_Results` are the merged results from running `Merge_data.py`after having created parameter-search data.


### Figures
Contains subfolders for figures generated during the project.

### LaTeX Files
- **project3.tex**: The LaTeX file for the report documenting the project, including regression methods and results.
- **appendix.tex**: The LaTeX file for the appendix of the report.
- **JHEP.bst**: The bibliography style file used for formatting references in the LaTeX document.
- **project3.bib**: The BibTeX file containing references for the report.
- **revtex4-2.cls.txt**: A copy of the revtex 4.2 class file used for formatting the report.

## Running the Code
1. Navigate to the `Python/` folder.
2. Execute the relevant scripts based on the desired regression model or task. For example, run `KerasCNN_Synthetic.py` to start creating data files to the parameter `save_path` inside the file. Or, check the current directory (`pkl_dir`) in `2D_parameter_plot.py`to see plots of the data there. 

