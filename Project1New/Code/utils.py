import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rasterio

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from inspect import signature
import time

from pathlib import Path

# Latex fonts
def latex_fonts():
    plt.rcParams['text.usetex'] = True
    plt.rcParams['axes.titlepad'] = 25 

    plt.rcParams.update({
        'font.family' : 'euclid',
        'font.weight' : 'bold',
        'font.size': 17,       # General font size
        'axes.labelsize': 17,  # Axis label font size
        'axes.titlesize': 22,  # Title font size
        'xtick.labelsize': 22, # X-axis tick label font size
        'ytick.labelsize': 22, # Y-axis tick label font size
        'legend.fontsize': 17, # Legend font size
        'figure.titlesize': 25 # Figure title font size
    })

def franke(x:np.ndarray,y:np.ndarray) -> np.ndarray:
        """
        Prameters
            * x:    x-values
            * y:    y-values

        Returns
            - franke function evaluated at (x,y) 
        """
    
        term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2))
        term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y + 1))
        term3 = 0.5*np.exp(-(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2))
        term4 = -0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
        return term1 + term2 + term3 + term4

def Design_Matrix(x:np.ndarray, y:np.ndarray, p:int) -> np.ndarray:
    """
    Creates a polynomial design matrix up to degree 'deg' for 2D-data
    """
    N = len(x)                          # Number of data points
    num_cols = (p + 1) * (p + 2) // 2   # Number of columns
    
    X = np.zeros((N, num_cols))         # Initialize the design matrix
    
    col_idx = 0
    for degree in range(p + 1):
        for k in range(degree + 1):
            X[:, col_idx] = (x ** (degree - k)) * (y ** k)
            col_idx += 1
    
    return X


############# Error-measure functions #############
def MSE(x, y):
    """
    Returns the mean squared error using sklearn.metrics.mean_squared_error
    """
    return mean_squared_error(x, y)

def R2(x, y):
    """
    Returns the R2 score using sklearn.metrics.mean_squared_error.r2_score
    """
    return r2_score(x, y)

############# Fitting #############

def Ridge_Reg(X, y, lambd):
    """
    Calculates and returns (X^TX + lambda I )^{-1}X^ty
    """
    XTX = X.T @ X
    y = y.reshape(-1, 1)
    return np.linalg.pinv(XTX + lambd * np.identity(XTX.shape[0])) @ X.T @ y

def Ridge_fit(X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, lmbda:float) -> list:
    """
    Parameters
        * X_train:          array of size (n1 x p) 
        * X_test:           array of size (n2 x p) 
        * y_train:          array of size (n1) 
        * y_test:           array of size (n2)
        * lmbda:            float sent to Ridge_Reg 

    Uses Ridge_Reg on the train-data to calculate the the beta-coefficient in y = X@beta + epsilon. Using the obtained beta, predicts y on test and train data, and then their respective MSE/R2. 

    Returns 
        beta, MSE_train, MSE_test, R2_train, R2_test 
    """
    beta = Ridge_Reg(X_train, y_train, lmbda)

    y_tilde = X_train @ beta
    y_pred = X_test @ beta  

    MSE_train = MSE(y_train, y_tilde)
    MSE_test = MSE(y_test, y_pred)
    R2_train = R2(y_train, y_tilde)
    R2_test = R2(y_test, y_pred)

    return beta, MSE_train, MSE_test, R2_train, R2_test

def OLS_fit(X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray) -> list:
    """
    Parameters
        * X_train:          array of size (n1 x p) 
        * X_test:           array of size (n2 x p) 
        * y_train:          array of size (n1) 
        * y_test:           array of size (n2)

    Calculates beta-coefficient, and the MSE and R2 score for train and test-data. Uses the fact that beta_OLS = beta_Ridge(lambda = 0).

    Returns 
        beta, MSE_train, MSE_test, R2_train, R2_test 
    """
    return Ridge_fit(X_train, X_test, y_train, y_test, 0)

# Introduced as a class in the case of experimenting with maximum number of iterations and fit-tolerance
def LASSO_fit(X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, lmbda:float) -> list:
    clf = linear_model.Lasso(lmbda, fit_intercept=False, max_iter=int(1e5), tol=1e-1)
    clf.fit(X_train,y_train)

    y_tilde = clf.predict(X_train)
    y_pred = clf.predict(X_test)
    beta = clf.coef_

    MSE_train = MSE(y_train, y_tilde)
    MSE_test = MSE(y_test, y_pred)
    R2_train = R2(y_train, y_tilde)
    R2_test = R2(y_test, y_pred)

    return beta, MSE_train, MSE_test, R2_train, R2_test

############# Scaling #############
def scale_data(X_train, X_test, y_train, y_test, scaler_type="StandardScaler", b=1, a=0):
    """
    Scales data using sklearn.preprocessing. Currently only supports standard scaling and min-max scaling.
    """
    if scaler_type.upper() == "STANDARDSCALER" or scaler_type.upper() == "STANDARDSCALING":
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
    
    elif scaler_type.upper() in ["MINMAX", "MIN_MAX"]:
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
    
    else:
        raise ValueError(f"Did not recognize: {scaler_type}")
    # Fit on training data and transform both training and testing sets
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale target values
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

def Bootstrap(X:np.ndarray, y:np.ndarray, samples:int, reg_method="OLS", scaling_type="StandardScalar", 
              lmbda=None, test_percentage=0.25,  max_iter=int(1e5), tol=1e-1):
    if reg_method.upper() == "OLS":
        def fit(X_train, X_test, y_train, y_test, lmbda):
            return OLS_fit(X_train, X_test, y_train, y_test)
    elif reg_method.upper() == "LASSO":
        def fit(X_train, X_test, y_train, y_test, lmbda):
            LASS = LASSO_fit(max_iter=max_iter, tol=tol)
            return LASS(X_train, X_test, y_train, y_test, lmbda)
        if lmbda is None:
            raise ValueError(f"Argument 'lmbda' needs to be a float for LASSO-method")
    elif reg_method.upper() == "RIDGE":
        def fit(X_train, X_test, y_train, y_test, lmbda):
            return Ridge_fit(X_train, X_test, y_train, y_test, lmbda)
        if lmbda is None:
            raise ValueError(f"Argument 'lmbda' needs to be a float for RIDGE-method")
    else:
        raise ValueError(f"Does not recognize reg-method: {reg_method}")
        
    MSE_train = np.zeros(samples)
    MSE_test = np.zeros(samples)

    N = np.shape(X)[0]

    # Perform bootstrapping for 'samples' number of resamples
    for i in range(samples):
        idx = np.random.randint(0, N, N)  # Random resampling with replacement
        X_i = X[idx, :]
        y_i = y[idx]
        X_train, X_test, y_train, y_test = train_test_split(X_i, y_i, test_size=test_percentage)
        X_train, X_test, y_train, y_test = scale_data(X_train, X_test, y_train, y_test, scaler_type=scaling_type)

        _, MSE_train[i], MSE_test[i], _, _ = fit(X_train, X_test, y_train, y_test, lmbda)

    # Calculate mean and std of MSE for train and test sets
    MSE_train_mean = np.mean(MSE_train)
    MSE_train_std = np.std(MSE_train)
    MSE_test_mean = np.mean(MSE_test)
    MSE_test_std = np.std(MSE_test)

    return MSE_train_mean, MSE_train_std, MSE_test_mean, MSE_test_std

def Cross_Validation(X:np.ndarray, y:np.ndarray, k:int, reg_method="OLS", lmbda=None, max_iter=int(1e5), tol=1e-1):
    """
    k-fold cross-validation for OLS, Ridge, or LASSO models
    """
    if reg_method.upper() == "OLS":
        def fit(X_train, X_test, y_train, y_test, lmbda):
            return OLS_fit(X_train, X_test, y_train, y_test)
    elif reg_method.upper() == "LASSO":
        def fit(X_train, X_test, y_train, y_test, lmbda):
            LASS = LASSO_fit(max_iter=max_iter, tol=tol)
            return LASS(X_train, X_test, y_train, y_test, lmbda)
        if lmbda is None:
            raise ValueError(f"Argument 'lmbda' needs to be a float for LASSO-method")
    elif reg_method.upper() == "RIDGE":
        def fit(X_train, X_test, y_train, y_test, lmbda):
            return Ridge_fit(X_train, X_test, y_train, y_test, lmbda)
        if lmbda is None:
            raise ValueError(f"Argument 'lmbda' needs to be a float for RIDGE-method")
    else:
        raise ValueError(f"Does not recognize reg-method: {reg_method}")
    
    N = np.shape(X)[0]
    shuffle_idx = np.random.permutation(N)  # Shuffle the dataset
    X = X[shuffle_idx, :]
    y = y[shuffle_idx]

    kfold_idx = np.linspace(0, N, k + 1)  # Indices for k folds

    MSE_train_array = np.zeros(k)
    MSE_test_array = np.zeros(k)

    for i in range(k):
        # Define training and testing sets based on k-folds
        i_0 = int(kfold_idx[i])
        i_1 = int(kfold_idx[i + 1])
        i_test = np.arange(i_0, i_1)
        X_test = X[i_test, :]
        X_train = np.delete(X, i_test, 0)

        y_test = y[i_test]
        y_train = np.delete(y, i_test)

        X_train, X_test, y_train, y_test = scale_data(X_train, X_test, y_train, y_test)

        _, MSE_train_array[i], MSE_test_array[i], _, _ = fit(X_train, X_test, y_train, y_test, lmbda)
        
    # Calculate mean and std for MSE for train and test sets across all folds
    MSE_train_mean = np.mean(MSE_train_array)
    MSE_train_std = np.std(MSE_train_array)
    MSE_test_mean = np.mean(MSE_test_array)
    MSE_test_std = np.std(MSE_test_array)

    return MSE_train_mean, MSE_train_std, MSE_test_mean, MSE_test_std