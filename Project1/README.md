# Project1: Application of Regression and Resampling on USGS Terrain Data

## Abstract
Regression models are widely used for analyzing and predicting data in many fields. We investigate and compare three regression models: Ordinary Least Squares, Ridge, and Least Absolute Shrinkage and Selection Operator. These were applied and analyzed on a two-dimensional Franke function and later tested on real terrain data from the US Geological Survey Earth Explorer. The models' performances are evaluated using metrics such as the Mean Squared Error and the coefficient of determination $R^2$. The bias-variance trade-off for OLS is studied where it was found that the optimal polynomial degree for the Franke function was $p\sim4$. We used resampling techniques such as bootstrapping and cross-validation to probe the quality of the evaluations and determine the predictiveness and generalizability of the models. The terrain data has a noticeably larger complexity as one needed to go $p\gtrsim 20$ to find reasonable fits. In our analysis of both the Franke function and the terrain data the best performing method was Ridge regression with OLS following closely after. LASSO performed appreciably worse than the other regression methods, likely stemming from the polynomial coefficients not following a Laplace distribution.

## Structure

### Python
This folder contains all the Python scripts used in the project.

- **Bootstrap.py**: Implements the bootstrapping technique to compute the bias-variance trade-off for the regression models.
- **CrossValidate.py**: Performs k-fold cross-validation on the regression models (OLS, Ridge, and LASSO) to evaluate performance.
- **Franke.py**: Contains the implementation of the Franke function, used as a synthetic dataset for regression analysis.
- **LASSO.py**: Script for running LASSO regression on the datasets.
- **OLS.py**: Script for running Ordinary Least Squares regression on the datasets.
- **Ridge.py**: Script for running Ridge regression on the datasets.
- **TerrainPredict.py**: Script for applying the trained models to real terrain data and plots them.
- **TerrainVisualize.py**: Visualizes terrain data and regression results as contour plots.
- **utils.py**: Contains utility functions used across all the scripts.

### Data
This folder holds the datasets used for analysis. These include both synthetic data generated from the Franke function and real-world terrain data.

### Figures
Contains subfolders for figures generated during the project. These figures are categorized by the type of analysis or model:

- **CV**: Plots generated during cross-validation.
- **LASSO**: Figures related to LASSO regression.
- **OLS**: Figures related to OLS regression.
- **Ridge**: Figures related to Ridge regression.
- **Terrain**: Figures related to terrain data regression results.

### LaTeX Files
- **project1.tex**: The LaTeX file for the report documenting the project, including regression methods and results.
- **JHEP.bst**: The bibliography style file used for formatting references in the LaTeX document.
- **project1.bib**: The BibTeX file containing references for the report.
- **revtex4-2.cls.txt**: A copy of the revtex 4.2 class file used for formatting the report.

## Running the Code

1. Navigate to the `Python/` folder.
2. Execute the relevant scripts based on the desired regression model or task. For example:
   - To run OLS regression: `python OLS.py`
3. Figures will be saved in the corresponding subfolders under `Figures/` if `save` is set to `True`.