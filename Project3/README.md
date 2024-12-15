# Project2: Recurrent Neural Networks and Synthetic Gravitational Waves

## Abstract
Insert Abstract

## Structure

### Project
The project is `project3.pdf`.

### Python
This folder contains all the Python scripts used in the project.

- `2D_parameter_plot.py`: Uses data from `GW_Merged_Results/` to create 2D parameter plots. Each of these plots are interactive, where you can click on the grid and the results from the corresponding parameter combination test results will be shown. The input prompt gives you the option to save these generated figures, which will be automatically saved with the given parameter combination to `Figures`.
- `Clean_Data.py`: Utility file used to remove certain parts from the data which did not need to be saved.
- `KerasRNN_Synthetic.py`: Performs a parameter scan with using Keras' RNN which is initialized in `NNs.py`.
- `Merge_data.py`: Merges parameter combinations of the learning rate and l2 regularization into larger files.
- `parameters.py`: Runs a parameter scan, saving the data to a chosen folder (`GW_Merged_Resulst` by default).
- `read_GW.py`: Downloads data surrounding a GW event from gwosc. Given the duration of the signal, this program automatically labels the data, saving it to the `Data` folder.
- `PlotGWExample.py`: A simple plot of the GW data downloaded from gwosc with `read_GW.py`.
- `NNs.py`: Contains the Neural Network classes, RNN and KerasRNN.
- `utils.py`: Contains utility functions used across all the scripts.

#### Data
This folder contains the data saved from `read_GW.py`.

#### GW_Merged_Results
Merged results from `GW_Parameter_Search`.

#### GW_Parameter_Search_SNR{x}
Stores results generated from the NN's to be later merged using `Merge_data.py`.


### Figures
Contains subfolders for figures generated during the project.

### LaTeX Files
- **project3.tex**: The LaTeX file for the report documenting the project, including regression methods and results.
- **JHEP.bst**: The bibliography style file used for formatting references in the LaTeX document.
- **project3.bib**: The BibTeX file containing references for the report.
- **revtex4-2.cls.txt**: A copy of the revtex 4.2 class file used for formatting the report.

## Running the Code
1. Navigate to the `Python/` folder.
2. Execute the relevant scripts based on the desired regression model or task. For example:
   - To run KerasRNN: `KerasRNN_Synthetic.py`

Running **2d_parameter_plot.py** will read existing data and display multiple interactive 2D parameter plots. To simply test either the RNN class or the KerasRNN class on synthetic GW data, modify and run **Train_Synthetic.py** with chosen variables.