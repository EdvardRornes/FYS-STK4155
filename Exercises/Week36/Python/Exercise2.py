import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Calculate Ridge coefficients
def ridge_regression(X, y, lambd):
    n_features = X.shape[1]
    I = np.eye(n_features)
    beta_ridge = np.linalg.inv(X.T@X + lambd*I)@X.T@y
    return beta_ridge

# Predict Ridge
def predict_ridge(X, beta):
    return X@beta

# Generate data
np.random.seed(777)
n = 100
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5*np.exp(-(x - 2)**2) + np.random.normal(0, 0.1, x.shape)

# Lambda values for Ridge
lambdas = [0.0001, 0.001, 0.01, 0.1, 1.0]

# Degrees for polynomials
degrees = [5, 10, 15]

for deg in degrees:
    # Create polynomial features
    poly = PolynomialFeatures(degree=deg)
    X = poly.fit_transform(x)

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # OLS, i.e. lambda=0
    beta_ols = ridge_regression(X_train, y_train, lambd=0)
    y_pred_train_ols = predict_ridge(X_train, beta_ols)
    y_pred_test_ols = predict_ridge(X_test, beta_ols)
    
    print(np.size(X_train))
    print(np.size(y_pred_train_ols))
    plt.plot(X_train, y_pred_train_ols)
    plt.show()
    mse_train_ols = mean_squared_error(y_train, y_pred_train_ols)
    mse_test_ols = mean_squared_error(y_test, y_pred_test_ols)
    
    print(f"\nDegree: {deg}")
    print(f"OLS - MSE Train: {mse_train_ols:.5f}, MSE Test: {mse_test_ols:.5f}")
    
    # Ridge
    mse_train_ridge = []
    mse_test_ridge = []
    
    for lambd in lambdas:
        beta_ridge = ridge_regression(X_train, y_train, lambd)
        y_pred_train_ridge = predict_ridge(X_train, beta_ridge)
        y_pred_test_ridge = predict_ridge(X_test, beta_ridge)
        
        mse_train_ridge.append(mean_squared_error(y_train, y_pred_train_ridge))
        mse_test_ridge.append(mean_squared_error(y_test, y_pred_test_ridge))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, mse_train_ridge, label="Ridge train", marker='o', lw=2.5)
    plt.plot(lambdas, mse_test_ridge, label="Ridge test", marker='o', lw=2.5)
    plt.axhline(y=mse_train_ols, color='r', linestyle='--', label="OLS train", lw=2.5)
    plt.axhline(y=mse_test_ols, color='b', linestyle='--', label="OLS test", lw=2.5)
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('MSE')
    plt.title(f"Degree {deg} - OLS vs. Ridge")
    plt.legend()
    plt.savefig(f'OLS-Ridge-degree-{deg}.pdf')
    plt.show()
