import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import utils as ut
 
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
 

################ Brute force ################
N = 100
x = np.sort(np.random.rand(N))
y = np.sort(np.random.rand(N))
z = franke(x,y)
z = z + 0.1*np.random.normal(0, 1, z.shape)
 
deg_max = 25
degs = np.linspace(1, deg_max, deg_max, dtype=int)
 
MSE_train_brute_force = np.zeros(deg_max)
MSE_test_brute_force = np.zeros(deg_max)
R2_train_brute_force = np.zeros(deg_max)
R2_test_brute_force = np.zeros(deg_max)
betas = [0]*deg_max
 
for i in range(deg_max):
    X = Design_Matrix(x, y, degs[i])
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25, random_state=42)
 
    betas[i], MSE_train_brute_force[i], MSE_test_brute_force[i], R2_train_brute_force[i], R2_test_brute_force[i] = OLS_fit(X_train, X_test, z_train, z_test)

################ utils.PolynomialRegression ################
# Setup
data = [x,y,z] # Test on same data

# Regression
OLS = ut.PolynomialRegression(ut.OLS_fit, deg_max, data, scaling="no_scaling")
MSE_train, MSE_test = OLS.MSE()
R2_train, R2_test = OLS.R2()
beta = OLS.beta

degrees = OLS.degrees

################ PLOT ################
minor_ticks = np.arange(0, len(beta[-1]), 1)
major_ticks = [n*(n+1)/2 for n in range(deg_max)]
yticks = np.arange(-1, 6, 1)

############### Norm #################
print(np.linalg.norm(MSE_train - MSE_train_brute_force))
print(np.linalg.norm(MSE_test - MSE_test_brute_force), np.linalg.norm(MSE_test), np.linalg.norm(MSE_test_brute_force))
with open("tmp.text", "r") as file:
    lines = file.readlines()
lines.append(f"{np.linalg.norm(MSE_test):.15f}, {np.linalg.norm(MSE_test_brute_force):.15f}")

with open("tmp.text", "w") as file:
    for i in range(len(lines)-1):
        file.write(lines[i].strip() + "\n")
    if len(lines) != 0:
        file.write(lines[-1].strip())

############### MSE-plot ###############
plt.figure()
plt.title(f"Comparison MSE")
line = plt.plot(degs,MSE_train_brute_force,label="MSE-train, brute force")
color = line[0].get_color()
plt.plot(degs,MSE_test_brute_force, "--", color=color, label="MSE-test, brute force")

line = plt.plot(degrees, MSE_train, label=r"MSE train", lw=2.5)
color = line[0].get_color()
plt.plot(degrees, MSE_test, "--", color=color, label=r"MSE test", lw=2.5)

plt.xlabel("degree")
plt.ylabel("MSE")
plt.yticks()
plt.xticks()
plt.xlim(1, deg_max)
plt.legend()
plt.grid(True)
 

if len(lines) != 0:
    brute_force_norm = []; utils_nomr = []
    for i in range(len(lines)):
        brute_force_norm.append(float(lines[i].split(",")[1]))
        utils_nomr.append(float(lines[i].split(",")[0]))
    
    brute_force_norm = np.array(brute_force_norm); utils_nomr = np.array(utils_nomr)
    plt.figure()
    plt.title("Norm of MSE_test")
    plt.plot(np.linspace(0,len(lines)-1, len(lines)), np.log10(utils_nomr), label="utils.PolynomialRegression")
    plt.plot(np.linspace(0,len(lines)-1, len(lines)), np.log10(brute_force_norm), label="brute force")
    plt.ylabel(r"$\log||MSE||^2$")
    plt.legend()
    plt.show()