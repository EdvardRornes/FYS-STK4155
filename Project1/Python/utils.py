import numpy as np
import sklearn
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Franke function
def Franke(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2))
    term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y + 1))
    term3 = 0.5*np.exp(-(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2))
    term4 = -0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4

# Creates a polynomial design matrix up to degree 'deg'
def Design_Matrix(x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
    ## Create feature matrix
    xy_dic = {
        "x": x,
        "y": y
    }
    xy = pd.DataFrame(xy_dic)

    poly = PolynomialFeatures(degree=deg)
    X = poly.fit_transform(xy)

    return X

# Ridge regression, lambd = 0 for OLS
def Ridge_Reg(X, y, lambd):
    XTX = X.T @ X
    y = y.reshape(-1, 1)
    return np.linalg.pinv(XTX + lambd * np.identity(XTX.shape[0])) @ X.T @ y

# LASSO regression using sklearn's Lasso class
def LASSO_Reg(X, y, lambd):
    return Lasso(alpha=lambd).fit(X, y)

def MSE(x, y):
    return mean_squared_error(x, y)

def R2(x, y):
    return r2_score(x, y)

# Scales the input data using StandardScaler
def scale_data(X_train, X_test, y_train, y_test):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit on training data and transform both training and testing sets
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale target values
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

# OLS fitting
def OLS_fit(X_train, X_test, y_train, y_test):
    # Ridge reg with lambda = 0
    beta = Ridge_Reg(X_train, y_train, 0)

    # Predictions on training and testing data
    y_tilde = X_train @ beta
    y_pred = X_test @ beta

    # Calculate MSE and R2
    MSE_train = MSE(y_train, y_tilde)
    MSE_test = MSE(y_test, y_pred)
    R2_train = R2(y_train, y_tilde)
    R2_test = R2(y_test, y_pred)

    return beta, MSE_train, MSE_test, R2_train, R2_test

# Ridge fitting
def Ridge_fit(X_train, X_test, y_train, y_test, lambd):
    beta = Ridge_Reg(X_train, y_train, lambd)

    y_tilde = X_train @ beta
    y_pred = X_test @ beta

    MSE_train = MSE(y_train, y_tilde)
    MSE_test = MSE(y_test, y_pred)
    R2_train = R2(y_train, y_tilde)
    R2_test = R2(y_test, y_pred)

    return beta, MSE_train, MSE_test, R2_train, R2_test

# LASSO fitting
def LASSO_fit(X_train, X_test, y_train, y_test, lambd):
    # Configure and fit LASSO model
    model = linear_model.Lasso(lambd, fit_intercept=False, max_iter=int(1e6), tol=1e-2)
    model.fit(X_train, y_train)

    # Predictions on training and testing data
    y_tilde = model.predict(X_train)
    y_pred = model.predict(X_test)
    beta = model.coef_

    # Calculate MSE and R2
    MSE_train = MSE(y_train, y_tilde)
    MSE_test = MSE(y_test, y_pred)
    R2_train = R2(y_train, y_tilde)
    R2_test = R2(y_test, y_pred)

    return beta, MSE_train, MSE_test, R2_train, R2_test

# Bootstrapping for OLS
def Bootstrap_OLS(X, y, samples, test_percentage):
    MSE_train = np.zeros(samples)
    MSE_test = np.zeros(samples)

    N = np.shape(X)[0]

    # Perform bootstrapping for 'samples' number of resamples
    for i in range(samples):
        idx = np.random.randint(0, N, N)  # Random resampling with replacement
        X_i = X[idx, :]
        y_i = y[idx]
        X_train, X_test, y_train, y_test = train_test_split(X_i, y_i, test_size=test_percentage)
        X_train, X_test, y_train, y_test = scale_data(X_train, X_test, y_train, y_test)

        MSE_train[i], MSE_test[i], R2_train, R2_test, beta = OLS_fit(X_train, X_test, y_train, y_test)

    # Calculate mean and std of MSE for train and test sets
    MSE_train_mean = np.mean(MSE_train)
    MSE_train_std = np.std(MSE_train)
    MSE_test_mean = np.mean(MSE_test)
    MSE_test_std = np.std(MSE_test)

    return MSE_train_mean, MSE_train_std, MSE_test_mean, MSE_test_std

# k-fold cross-validation for OLS, Ridge, or LASSO models
def Cross_Validation(X, y, k, model, lambd=0):
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

        # Model selection: 0 for OLS, 1 for Ridge, 2 for LASSO
        if model == 0:
            beta, MSE_train_array[i], MSE_test_array[i], R2_train, R2_test = OLS_fit(X_train, X_test, y_train, y_test)
        elif model == 1:
            beta, MSE_train_array[i], MSE_test_array[i], R2_train, R2_test = Ridge_fit(X_train, X_test, y_train, y_test, lambd)
        elif model == 2:
            beta, MSE_train_array[i], MSE_test_array[i], R2_train, R2_test = LASSO_fit(X_train, X_test, y_train, y_test, lambd)
        else:
            print("Model argument must be 0 for OLS, 1 for Ridge, or 2 for LASSO")
        
    # Calculate mean and std for MSE for train and test sets across all folds
    MSE_train_mean = np.mean(MSE_train_array)
    MSE_train_std = np.std(MSE_train_array)
    MSE_test_mean = np.mean(MSE_test_array)
    MSE_test_std = np.std(MSE_test_array)

    return MSE_train_mean, MSE_train_std, MSE_test_mean, MSE_test_std
