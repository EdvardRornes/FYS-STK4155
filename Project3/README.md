# Project3: Sequential data stuff

## Abstract
Insert Abstract

## Structure

### Python
This folder contains all the Python scripts used in the project.

- `2D_parameter_plot.py`: Uses data from `GW_Merged_Results/` to create 2D parameter plots. Each of these plots are interactive, where you can click on the grid and the results from the corresponding parameter combination test results will be shown. The input prompt gives you the option to save these generated figures, which will be automatically saved with the given parameter combination to `Figures`.
- `Clean_Data.py`: Utility file used to remove certain parts from the data which did not need to be saved.
- `KerasRNN_Synthetic.py`: Performs a parameter scan with using Keras' RNN which is initialized in `NNs.py`.
- `Merge_data.py`: Merges parameter combinations of the learning rate and l2 regularization into larger files.
- `parameters.py`: 
- `PlotGWExample.py`: A simple plot of the GW data downloaded from gwosc with `read_GW.py`.
- `read_GW.py`: Downloads data surrounding a GW event from gwosc. Given the duration of the signal, this program automatically chooses a range around the GW event, labels the the data accordingly and saves this data.
- `.py`: 
- `utils.py`: Contains utility functions used across all the scripts.

#### Data
This folder contains the data saved from `read_GW.py`.

#### GW_Merged_Results
Merged results from `GW_Parameter_Search`.

#### GW_Parameter_Search_SNR{x}
Stores results generated from the NN's to be later merged using `Merge_data.py`.


### Figures
Contains subfolders for figures generated during the project. These figures are categorized by the type of analysis or model:

- **Insert**

### LaTeX Files
- **project3.tex**: The LaTeX file for the report documenting the project, including regression methods and results.
- **JHEP.bst**: The bibliography style file used for formatting references in the LaTeX document.
- **project3.bib**: The BibTeX file containing references for the report.
- **revtex4-2.cls.txt**: A copy of the revtex 4.2 class file used for formatting the report.

## Running the Code

1. Navigate to the `Python/` folder.
2. Execute the relevant scripts based on the desired regression model or task. For example:
   - To run KerasRNN: `KerasRNN_Synthetic OLS.py`
3. Figures will be saved in the corresponding subfolders under `Figures/` if `save` is set to `True`.