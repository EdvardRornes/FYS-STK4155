import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def Franke(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def Design_Matrix(x, y, deg):
    terms = [(x**i * y**j) for i in range(deg + 1) for j in range(deg + 1 - i)]
    return np.vstack(terms).T

# Default to OLS if no input lambd
def Ridge_Reg(X, y, lambd):
    return np.linalg.pinv(X.T @ X + lambd*np.eye(X.shape[1])) @ X.T @ y

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
    X_test_scaled = scaler_X.transform(X_test)
    
    # Fit the scaler on the training target values
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled