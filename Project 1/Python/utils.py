import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def Franke(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def Design_Matrix(x, y, deg):
    # Creates a polynomial design matrix given x and y up to degree deg
    terms = [(x**i * y**j) for i in range(deg + 1) for j in range(deg + 1 - i)]
    return np.vstack(terms).T

# Default to OLS if no input lambd
def Ridge_Reg(X, y, lambd):
    XTX = X.T @ X
    return np.linalg.pinv(X.T @ X + lambd*np.identity(XTX.shape[0])) @ X.T @ y

def LASSO_Reg(X, y, lambd):
    return Lasso(alpha=lambd).fit(X, y)

def MSE(x, y):
    return mean_squared_error(x, y)

def R2(x, y):
    return r2_score(x, y)

def scale_data(X_train, X_test, y_train, y_test):
    """
    Scales the input data using StandardScaler.
    
    Parameters:
    - X_train: Training features.
    - X_test: Testing features.
    - y_train: Training target values.
    - y_test: Testing target values.

    Returns:
    - Scaled versions of X_train, X_test, y_train, and y_test.
    """
    # Initialize the scaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit the scaler on the training data
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled  = scaler_X.transform(X_test)
    
    # Fit the scaler on the training target values
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

def OLS_fit(X_train, X_test, y_train, y_test):
    beta = Ridge_Reg(X_train, y_train, 0)

    y_tilde = X_train @ beta
    y_pred  = X_test @ beta

    MSE_train = MSE(y_train, y_tilde)
    MSE_test  = MSE(y_test, y_pred)
    R2_train  = R2(y_train, y_tilde)
    R2_test   = MSE(y_test, y_pred)

    return beta, MSE_train, MSE_test, R2_train, R2_test

def Ridge_fit(X_train, X_test, y_train, y_test, lambd):
    beta = Ridge_Reg(X_train, y_train, lambd)

    y_tilde = X_train @ beta
    y_pred  = X_test @ beta
    
    MSE_train = MSE(y_train, y_tilde)
    MSE_test  = MSE(y_test, y_pred)
    R2_train  = R2(y_train, y_tilde)
    R2_test   = R2(y_test, y_pred)

    return beta, MSE_train, MSE_test, R2_train, R2_test

# Probably should just use LASSO_reg
def LASSO_fit(X_train, X_test, y_train, y_test, lambd):
    model = linear_model.Lasso(lambd, fit_intercept=False, max_iter=int(1e6), tol=1e-2)
    model.fit(X_train, y_train)

    y_tilde = model.predict(X_train)
    y_pred  = model.predict(X_test)
    beta    = model.coef_
    
    MSE_train = MSE(y_train, y_tilde)
    MSE_test  = MSE(y_test, y_pred)
    R2_train  = R2(y_train, y_tilde)
    R2_test   = R2(y_test, y_pred)
    
    return beta, MSE_train, MSE_test, R2_train, R2_test

def Bootstrap_OLS(X, y, samples, test_percentage):
    MSE_train = np.zeros(samples)
    MSE_test  = np.zeros(samples)

    N = np.shape(X)[0]

    for i in range(samples):
        idx = np.random.randint(0, N, N)
        X_i = X[idx, :]
        y_i = y[idx]
        X_train, X_test, y_train, y_test = train_test_split(X_i, y_i, test_size=test_percentage)
        X_train, X_test, y_train, y_test = scale_data(X_train, X_test, y_train, y_test)

        MSE_train[i], MSE_test[i], R2_train, R2_test, beta = OLS_fit(X_train, X_test, y_train, y_test)

    MSE_train_mean = np.mean(MSE_train)
    MSE_train_std = np.std(MSE_train)
    MSE_test_mean = np.mean(MSE_test)
    MSE_test_std = np.std(MSE_test)

    return MSE_train_mean, MSE_train_std, MSE_test_mean, MSE_test_std

def Cross_Validation(X, y, k, model, lambd = 0):
    # model = 0 for OLS, 1 for Ridge and 2 for LASSO
    N = np.shape(X)[0]
    shuffle_idx = np.random.permutation(N)
    X = X[shuffle_idx, :]
    y = y[shuffle_idx]

    kfold_idx = np.linspace(0, N, k+1)

    MSE_train_array = np.zeros(k)
    MSE_test_array = np.zeros(k)

    for i in range(k):
        i_0 = kfold_idx[i]
        i_1 = kfold_idx[i+1]
        i_test = np.array(range(i_0, i_1))
        X_test = X[i_test, :]

        X_copy = X.copy()
        X_train = np.delete(X_copy, i_test, 0)

        y_test = y[i_test]
        y_copy = y.copy()
        y_train = np.delete(y_copy, i_test)

        X_train, X_test, y_train, y_test = scale_data(X_train, X_test, y_train, y_test)

        if model == 0:
            MSE_train_array[i], MSE_test_array[i], R2_train, R2_test, beta = OLS_fit(X_train, X_test, y_train, y_test)
        if model == 0:
            MSE_train_array[i], MSE_test_array[i], R2_train, R2_test, beta = Ridge_fit(X_train, X_test, y_train, y_test, lambd)
        if model == 0:
            MSE_train_array[i], MSE_test_array[i], R2_train, R2_test, beta = LASSO_fit(X_train, X_test, y_train, y_test, lambd)
        else:
            print("Model argument must be 0 for OLS, 1 for Ridge or 2 for LASSO")
        
    MSE_train_mean = np.mean(MSE_train_array)
    MSE_train_std = np.std(MSE_train_array)
    MSE_test_mean = np.mean(MSE_test_array)
    MSE_test_std = np.std(MSE_test_array)

    return MSE_train_mean, MSE_train_std, MSE_test_mean, MSE_test_std