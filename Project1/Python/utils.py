import numpy as np
import matplotlib.pyplot as plt
import rasterio
import time

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from inspect import signature
from pathlib import Path
from sklearn.utils import resample

# Saving files
def save_plt(filename_n_path, overwrite=False, type="pdf", stop=50, fig=None) -> None:
    filename = f"{filename_n_path}.{type}"
    my_file = Path(filename)
    if overwrite:
        if fig is None:
            plt.savefig(filename)
        else:
            fig.savefig(filename)
        return
    
    if my_file.is_file():
        i = 1
        while i <= stop: # May already be additional files created
            filename = f"{filename_n_path}{i}.{type}"
            my_file = Path(filename)
            if not my_file.is_file():
                if fig is None:
                    plt.savefig(f"{filename_n_path}{i}.{type}")
                else:
                    fig.savefig(f"{filename_n_path}{i}.{type}")
                return  
            i += 1
    
        print(f"You have {stop} additional files of this sort?")
    else:
        
        if fig is None:
            plt.savefig(filename)
        else:
            fig.savefig(filename)
            

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

class Franke:

    def __init__(self, N:int, eps:float):
        """
        Parameters
            * N:    number of data points 
            * eps:  noise-coefficient         
        """
        self.N = N; self.eps = eps
        self.x = np.sort(np.random.rand(N))
        self.y = np.sort(np.random.rand(N))
        self.z_without_noise = self.franke(self.x, self.y)
        self.z = self.z_without_noise + self.eps * np.random.normal(0, 1, self.z_without_noise.shape)

    def franke(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        """
        Parameters
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
        
class PolynomialRegression:

    def __init__(self, regr_model:str, deg_max:int, data:list, lmbdas=None, scaling="no_scaling", max_iter=int(1e5), tol=1e-1, 
                 test_size_percentage=0.25, random_state=4, start_training=True):
        """
        Parameters
            * regr_model:                   regression model of choice (OLS, LASSO or RIDGE)
            * deg_max:                      maximum polynomial degree
            * data:                         list on the form [x,y,z] 
            * lmbdas (None):                lambda values, in case of LASS or RIDGE   
            * scaling (no_scaling):         scaling type, by default no scaling, possible values: ['no_scaling', 'MINMAX', 'StandardScaling']     
            * max_iter (1e5):               for LASSO only: max number of iterations
            * tol (1e-1):                   for LASSO only: tolerance
            * test_size_percentage (0.25):  percentage of data beeing used for testing
            * random_state (4):             keyword argument for sklearn.model_selection.train_test_split, useful to keep fixed to analyze data over multiple runs
            * start_training (True):        if True; start training on data
        """
        self.test_size_percentage = test_size_percentage
        self.random_state = random_state

        self.deg_max = deg_max; self.lmbdas = lmbdas

        self.x, self.y, self.z = data

        self.degrees = np.arange(1, deg_max+1)

        #### Finding type of regresion model:
        if regr_model.upper() == "OLS":
            def correct_regr_model(X_train, X_test, z_train, z_test, lmbda):
                return self.OLS_fit(X_train, X_test, z_train, z_test)
            self.regr_model = correct_regr_model
            self._regr_model_name = "OLS"
            
        elif regr_model.upper() == "LASSO":
            LASSO = LASSO_fit(max_iter, tol)
            self.regr_model = LASSO
            self._regr_model_name = "LASSO"

            
        elif regr_model.upper() == "RIDGE":
            self.regr_model = self.Ridge_fit
            self._regr_model_name = "Ridge"
            
        else:
            raise ValueError(f"Does not recognize reg-method: {regr_model}")

        #### Scaling type:
        self.scale_data(np.zeros((1,1)), np.zeros((1,1)), np.zeros(1), np.zeros(1), scaling) # Testing function call 
        
        self.scaling = scaling 
        self.y_tilde = []; self.y_pred = []
        if start_training:
            self.train()
    
    
    def MSE(self) -> list:
        """
        Returns mean squared error of train and test-data
        """
        return self.MSE_train, self.MSE_test 
    
    def R2(self) -> list:
        """
        Returns the R2 score of train and test-data
        """
        return self.R2_train, self.R2_test
    
    def train(self) -> None:
        """
        Computes and stores:
            MSE_train, MSE_test, R2_train, R2_test (arrays):    (deg_max, size(lambdas))
            beta (list):                                        (deg_max, size(lambdas)) w/ beta[i,j] being of size i(i+1)/2
        
        """
        if self.lmbdas is None:
            start_time = time.time()

            self.MSE_train = np.zeros(len(self.degrees))
            self.MSE_test = np.zeros(len(self.degrees))
            self.R2_train = np.zeros(len(self.degrees))
            self.R2_test = np.zeros(len(self.degrees))
            self.beta = [0]*self.deg_max
            self.X = []
            self.X_train = []

            for deg in range(self.deg_max):
                # Create polynomial features
                X = self.Design_Matrix(self.x, self.y, self.degrees[deg])
                self.X.append(X)
                
                # Split into training and testing and scale
                X_train, X_test, z_train, z_test = train_test_split(X, self.z, test_size=self.test_size_percentage, random_state=self.random_state)
                X_train, X_test, z_train, z_test = self.scale_data(X_train, X_test, z_train, z_test, self.scaling)

                self.beta[deg], self.MSE_train[deg], self.MSE_test[deg], self.R2_train[deg], self.R2_test[deg] = self.regr_model(X_train, X_test, z_train, z_test, None)
                self.X_train.append(X_train)

                print(f"{self._regr_model_name}: p={deg}/{self.deg_max}, duration: {(time.time()-start_time):.2f}s", end="\r")

            print(f"{self._regr_model_name}: p={self.deg_max}/{self.deg_max}, duration: {(time.time()-start_time):.2f}s")

        else: 
            start_time = time.time()

            self.MSE_train = np.zeros((len(self.degrees), len(self.lmbdas)))
            self.MSE_test = np.zeros((len(self.degrees), len(self.lmbdas)))
            self.R2_train = np.zeros((len(self.degrees), len(self.lmbdas)))
            self.R2_test = np.zeros((len(self.degrees), len(self.lmbdas)))
            self.beta = []
            self.X = []
            self.X_train = []

            for l, i in zip(self.lmbdas, range(len(self.lmbdas))):

                self.beta.append([])
                for deg in range(self.deg_max):
                    # Create polynomial features
                    X = self.Design_Matrix(self.x, self.y, self.degrees[deg])
                    self.X.append(X)
                    # Split into training and testing and scale
                    X_train, X_test, z_train, z_test = train_test_split(X, self.z, test_size=self.test_size_percentage, random_state=self.random_state)
                    X_train, X_test, z_train, z_test = self.scale_data(X_train, X_test, z_train, z_test, self.scaling)

                    beta, self.MSE_train[deg, i], self.MSE_test[deg, i], self.R2_train[deg, i], self.R2_test[deg, i] = self.regr_model(X_train, X_test, z_train, z_test, l)
                    self.X_train.append(X_train)
                    self.beta[-1].append(beta)

                    print(f"{self._regr_model_name}: p={deg}/{self.deg_max}, duration: {(time.time()-start_time):.2f}s", end="\r")

            print(f"{self._regr_model_name}: p={self.deg_max}/{self.deg_max}, duration: {(time.time()-start_time):.2f}s")

    def Ridge_fit(self, X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, lmbda:float) -> list:
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
        beta = self.Ridge_Reg(X_train, y_train, lmbda)

        y_tilde = X_train @ beta
        y_pred = X_test @ beta 
        self.y_tilde.append(y_tilde); self.y_pred.append(y_pred)
        
        MSE_train = self.MSE(y_train, y_tilde)
        MSE_test = self.MSE(y_test, y_pred)
        R2_train = self.R2(y_train, y_tilde)
        R2_test = self.R2(y_test, y_pred)

        return beta, MSE_train, MSE_test, R2_train, R2_test
    
    def OLS_fit(self, X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray) -> list:
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
        return self.Ridge_fit(X_train, X_test, y_train, y_test, 0)

    @staticmethod
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

    @staticmethod
    def scale_data(X_train, X_test, y_train, y_test, scaler_type="StandardScaler", b=1, a=0):
        """
        Scales data using sklearn.preprocessing. Currently only supports standard scaling and min-max scaling.
        """
        if scaler_type.upper() == "STANDARDSCALER" or scaler_type.upper() == "STANDARDSCALING":
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
        elif scaler_type.upper() == "NO_SCALING":
            return X_train, X_test, y_train, y_test
        elif scaler_type.upper() in ["MINMAX", "MIN_MAX"]:
            scaler_X = MinMaxScaler(feature_range=(a, b))
            scaler_y = MinMaxScaler(feature_range=(a, b))
            
        elif scaler_type.upper() == "NO_SCALING":
            return X_train, X_test, y_train, y_test
        
        else:
            raise ValueError(f"Did not recognize: {scaler_type}")
        # Fit on training data and transform both training and testing sets
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        # Scale target values
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    ############# Error-measure functions #############
    @staticmethod
    def MSE(x, y):
        """
        Returns the mean squared error using sklearn.metrics.mean_squared_error
        """
        return mean_squared_error(x, y)

    @staticmethod
    def R2(x, y):
        """
        Returns the R2 score using sklearn.metrics.mean_squared_error.r2_score
        """
        return r2_score(x, y)

    ############# Fitting #############
    @staticmethod
    def Ridge_Reg(X, y, lambd):
        """
        Calculates and returns (X^TX + lambda I)^{-1}X^T y
        """
        XTX = X.T @ X
        y = y.reshape(-1, 1)
        return np.linalg.pinv(XTX + lambd * np.identity(XTX.shape[0])) @ X.T @ y

    def Bootstrap(self, x:np.ndarray, y:np.ndarray, z:np.ndarray, max_deg:int, samples:int, lmbda=None):
        
        error = np.zeros(max_deg)
        bias = np.zeros(max_deg)
        variance = np.zeros(max_deg)

        self.z_pred_bootstrap = []
        start_time = time.time()
        for degree in range(max_deg):
            X = self.Design_Matrix(x, y, degree)
            X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=self.test_size_percentage)
            z_test = z_test.reshape(-1,1)
            self.z_pred_bootstrap.append(np.empty((z_test.shape[0], samples)))
            error_i = []
            bias_i = []

            for i in range(samples):
                X_, z_ = resample(X_train, z_train)
                # X_, z_ = X_train, z_train
                self.regr_model(X_, X_test, z_, z_test, lmbda)
                self.z_pred_bootstrap[-1][:, i] = self.y_pred[-1][:,0] # self.regr_model creates next self.z_pred

                error_i.append(np.mean((z_test- self.z_pred_bootstrap[-1][:,i])**2))
                bias_i.append(np.mean(self.z_pred_bootstrap[-1][:,i]))

            print(f"{degree/max_deg*100:.1f}%, duration: {(time.time()-start_time):.2f}s", end="\r")

            error[degree] = np.mean(np.mean((z_test - self.z_pred_bootstrap[-1]) ** 2, axis=1, keepdims=True))
            bias[degree] = np.mean((z_test - np.mean(self.z_pred_bootstrap[-1], axis=1, keepdims=True)) ** 2)
            variance[degree] = np.mean(np.var(self.z_pred_bootstrap[-1], axis=1, keepdims=True))
            
        print(f"Bootstrap: 100.0%, duration: {(time.time()-start_time):.2f}s")
        return error, bias, variance
    
    def Cross_Validation(self, X:np.ndarray, y:np.ndarray, k:int, lmbda=None):
        """
        k-fold cross-validation for OLS, Ridge, or LASSO models
        """
        
        if lmbda is None:
            if not self._regr_model_name == "OLS":
                raise Warning(f"Should probably give a lmbda value since you are using {self._regr_model_name}.")
        
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

            X_train, X_test, y_train, y_test = self.scale_data(X_train, X_test, y_train, y_test, self.scaling)

            _, MSE_train_array[i], MSE_test_array[i], _, _ = self.regr_model(X_train, X_test, y_train, y_test, lmbda)
            
        # Calculate mean and std for MSE for train and test sets across all folds
        MSE_train_mean = np.mean(MSE_train_array)
        MSE_train_std = np.std(MSE_train_array)
        MSE_test_mean = np.mean(MSE_test_array)
        MSE_test_std = np.std(MSE_test_array)

        return MSE_train_mean, MSE_train_std, MSE_test_mean, MSE_test_std
    
# Introduced as a class in the case of experimenting with maximum number of iterations and fit-tolerance
class LASSO_fit:

    def __init__(self, max_iter=int(1e5), tol=1e-1):
        """
        Parameters
            * max_itr:      max iterations
            * tol:          fit-tolerance 
        """
        self.max_iter = max_iter; self.tol = tol
    
    def __call__(self, X_train, X_test, y_train, y_test, lmbda):
        # Configure and fit LASSO model
        # Lowering the tolerance causes it to not converge and increasing max iterations causes it to be very slow
        model = linear_model.Lasso(lmbda, fit_intercept=False, max_iter=self.max_iter, tol=self.tol) 
        model.fit(X_train, y_train)

        # Predictions on training and testing data
        y_tilde = model.predict(X_train)
        y_pred = model.predict(X_test)
        beta = model.coef_

        # Calculate MSE and R2
        MSE_train = PolynomialRegression.MSE(y_train, y_tilde)
        MSE_test = PolynomialRegression.MSE(y_test, y_pred)
        R2_train = PolynomialRegression.R2(y_train, y_tilde)
        R2_test = PolynomialRegression.R2(y_test, y_pred)
        
        return beta, MSE_train, MSE_test, R2_train, R2_test
    
LASSO_default = LASSO_fit() # Usually used


def get_latitude_and_conversion(filename):
    with rasterio.open(filename) as dataset:
        meta = dataset.meta
        transform = meta['transform']
        # The latitude is the y-coordinate of the top left corner
        latitude = transform[5]
        # Calculate the latitude conversion factor in meters per degree
        lat_conversion = 111412.84 * np.cos(np.radians(latitude))

        return latitude, lat_conversion