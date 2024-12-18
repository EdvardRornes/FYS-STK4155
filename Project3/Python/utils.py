import numpy as np
import autograd.numpy as anp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import jax.numpy as jnp
from jax import grad, jit
from autograd import grad, elementwise_grad
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import minmax_scale

import sys, os
import time 
import pickle
import copy
import random
from typing import Tuple, List

                        
############ Utility functions ############
# Function to save results incrementally
def save_results_incrementally(results, base_filename, save_path="GW_Parameter_Search"):
    filename = f"{base_filename}.pkl"
    with open(os.path.join(save_path, filename), "wb") as f:
        pickle.dump(results, f)
    print(f'File {filename} saved to {save_path}.')

# Function to load the results
def load_results(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

# Disable print
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore print
def enablePrint():
    sys.stdout = sys.__stdout__

# Latex fonts for figures
def latex_fonts():
    plt.rcParams['text.usetex'] = True
    plt.rcParams['axes.titlepad'] = 25 

    plt.rcParams.update({
        'font.family': 'euclid',
        'font.weight': 'bold',
        'font.size': 17,       # General font size
        'axes.labelsize': 17,  # Axis label font size
        'axes.titlesize': 22,  # Title font size
        'xtick.labelsize': 22, # X-axis tick label font size
        'ytick.labelsize': 22, # Y-axis tick label font size
        'legend.fontsize': 17, # Legend font size
        'figure.titlesize': 25 # Figure title font size
    })


# Can remove?
class Franke:

    def __init__(self, N:int, eps:float):
        """
        Parameters
            * N:    number of data points 
            * eps:  noise-coefficient         
        """
        self.N = N; self.eps = eps
        self.x = np.random.rand(N)
        self.y = np.random.rand(N)
        # self.x = np.sort(np.random.uniform(0, 1, N)) 
        # self.y = np.sort(np.random.uniform(0, 1, N)) 

        self.z_without_noise = self.franke(self.x, self.y)
        self.z = self.z_without_noise + self.eps * np.random.normal(0, 1, self.z_without_noise.shape)
    @staticmethod
    def franke(x:np.ndarray, y:np.ndarray) -> np.ndarray:
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
    
def plot_2D_parameter_lambda_eta(
        lambdas,
        etas,
        value,
        title=None,
        x_log=False,
        y_log=False,
        savefig=False,
        filename='',
        Reverse_cmap=False,
        annot=True,
        only_less_than=None,
        only_greater_than=None,
        xaxis_fontsize=None,
        yaxis_fontsize=None,
        xlim=None,
        ylim=None,
        ylabel=r"$\eta$",
        on_click=None,
        log_cbar=False
        ):
    """
    Plots a 2D heatmap of parameter values with lambda (x-axis) and eta (y-axis) as inputs.

    This function creates a heatmap to visualize a 2D grid of values based on given lambda and eta 
    arrays.

    Parameters:
    ----------
    lambdas: list or array-like.        Array of lambda values for the x-axis.
    etas: list or array-like.           Array of eta values for the y-axis.
    value: 2D array-like.               Matrix of values corresponding to the combinations of lambda and eta.
    title: str, optional.               Title of the plot. Default is None.
    x_log: bool, optional.              If True, formats the x-axis (lambda) in log scale. Default is False.
    y_log: bool, optional.              If True, formats the y-axis (eta) in log scale. Default is False.
    savefig: bool, optional.            If True, saves the plot to a PDF file. Default is False.
    filename: str, optional.            Name of the file to save the plot if `savefig` is True. Default is an empty string.
    Reverse_cmap: bool, optional.       If True, reverses the color map used for the heatmap. Default is False.
    annot: bool, optional.              If True, adds annotations (values) to each cell in the heatmap. Default is True.
    only_less_than: float, optional.    If provided, annotates only the cells with values less than this threshold. Default is None.
    only_greater_than: float, optional. If provided, annotates only the cells with values greater than this threshold. Default is None.
    xaxis_fontsize: int, optional.      Font size for the x-axis labels. Default is None (uses 12).
    yaxis_fontsize: int, optional.      Font size for the y-axis labels. Default is None (uses 12).
    xlim: tuple, optional.              Tuple specifying the limits for the x-axis (lambda). Format: (xmin, xmax). Default is None.
    ylim: tuple, optional.              Tuple specifying the limits for the y-axis (eta). Format: (ymin, ymax). Default is None.
    ylabel: str, optional.              Label for the y-axis.
    on_click: callable, optional.       Function to call when a point on the plot is clicked. The function should take a single 
                                        `matplotlib.backend_bases.MouseEvent` object as an argument. Default is None.
    log_cbar: bool, optional.           If True, applies a logarithmic transformation to the color bar values. Default is False.

    Returns:
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object for the plot.

    Notes:
    -----
    - The `value` matrix should have dimensions consistent with the lengths of `lambdas` and `etas`.
    - When using the `on_click` parameter, ensure the callable is designed to handle `matplotlib` click events.
    """
    cmap = 'plasma'
    if Reverse_cmap == True:
        cmap = 'plasma_r'
    fig, ax = plt.subplots(figsize = (12, 7))
    tick = ticker.ScalarFormatter(useOffset=False, useMathText=True)
    tick.set_powerlimits((0, 0))

    lambda_indices = np.array([True]*len(lambdas))
    eta_indices = np.array([True]*len(etas))

    if not (xlim is None):
        xmin = xlim[0]; xmax = xlim[1]
        lambda_indices = [i for i, l in enumerate(lambdas) if xmin <= l <= xmax]

    if not (ylim is None):
        ymin = ylim[0]; ymax = ylim[1]
        eta_indices = [i for i, e in enumerate(etas) if ymin <= e <= ymax]

    lambdas = np.array(lambdas)[lambda_indices]
    etas = np.array(etas)[eta_indices]
    value = value[np.ix_(eta_indices, lambda_indices)]

    if x_log:
        t_x = [u'${}$'.format(tick.format_data(lambd)) for lambd in lambdas]
    else:
        t_x = [fr'${lambd}$' for lambd in lambdas]

    if y_log:
        t_y = [u'${}$'.format(tick.format_data(eta)) for eta in etas]
    else:
        t_y = [fr'${eta}$' for eta in etas]

    if only_less_than is not None and only_greater_than is None:
        annot_data = np.where(value < only_less_than, np.round(value, 3).astype(str), "")
    elif only_greater_than is not None and only_less_than is None:
        annot_data = np.where(value > only_greater_than, np.round(value, 3).astype(str), "")
    else:
        annot_data = np.round(value, 3).astype(str) if annot else None

    if log_cbar:
        value = np.log(value)
        # value = np.sqrt(value)

    sns.heatmap(
        data=value,
        ax=ax,
        cmap=cmap,
        annot=annot_data,
        fmt="",
        xticklabels=t_x,
        yticklabels=t_y,
    )

    # Adjust x and y tick labels
    ax.set_xticks(np.arange(len(lambdas)) + 0.5)
    ax.set_xticklabels([f"{float(label):.1e}" for label in lambdas], rotation=45, ha='right', fontsize=14)

    ax.set_yticks(np.arange(len(etas)) + 0.5)
    ax.set_yticklabels([f"{float(label):.1e}" for label in etas], rotation=0, fontsize=14)

    # Add title and labels
    if title:
        if log_cbar:
            title += rf'. Color bar is logged.'
        plt.title(title)

    plt.xlabel(r'$\lambda$', fontsize=xaxis_fontsize or 12)
    plt.ylabel(ylabel, fontsize=yaxis_fontsize or 12)

    # Register the click event to trigger a new plot
    if on_click:
        fig.canvas.mpl_connect('button_press_event', on_click)

    plt.tight_layout()

    if savefig:
        plt.savefig(f'../Figures/{filename}.pdf')

    return fig, ax

def on_click(event, lambdas, etas, epochs, boosts, unique_lambdas, unique_etas, plot_info, time_steps, SNR, pkl_dir, save_option, filename_start="Synthetic_GW_Merged_Results_timesteps"):
    """
    Handles click events on a heatmap plot, allowing interactive exploration of the corresponding data.

    This function identifies the grid cell clicked by the user, maps it to the corresponding lambda 
    and eta values, and loads associated data for visualization.

    This function should not be called on its own, but in relation to a clicked event.
    """
    # Check if the click is within the axes
    if event.inaxes:
        # Get the clicked coordinates (in axis space)
        x, y = event.xdata, event.ydata

        # Floor the coordinates to align with displayed plot indices
        x = int(np.floor(x))
        y = int(np.floor(y))

        print(f"Clicked coordinates: x={x}, y={y}, Plot Info (Epoch, Boost): {plot_info}")

        # Map the clicked coordinates to the closest lambda and eta indices
        lambda_idx = x
        eta_idx = y

        # Retrieve the parameter values for the clicked indices
        clicked_lambda = unique_lambdas[lambda_idx]
        clicked_eta = unique_etas[eta_idx]

        # Extract epoch and boost from the plot_info tuple
        epoch, boost = plot_info

        # Filter data based on the clicked parameters
        mask = (lambdas == clicked_lambda) & (etas == clicked_eta) & (epochs == epoch) & (boosts == boost)
        
        x_test = np.linspace(0, 50, time_steps)

        if np.any(mask):
            print(f"Clicked Lambda: {clicked_lambda}, Eta: {clicked_eta}, Epochs: {epoch}, Boost: {boost:.1f}")

            # Load results from the corresponding file
            filepath = f'{pkl_dir}/{filename_start}_timesteps{time_steps}_SNR{SNR}_epoch{epoch}_boost{boost:.1f}.pkl'
            results = load_results(filepath)
            key = f"lambda_{clicked_lambda}_eta_{clicked_eta}"

            folds_data = results["data"][key]

            # Create a new figure for displaying results
            plt.figure(figsize=(20, 12))
            plt.suptitle(fr"$\eta={clicked_eta}$, $\lambda={clicked_lambda}$, Epochs$\,={epoch}$, $\phi={boost:.1f}$")

            # Loop through results and plot each fold
            threshold = int(input('Threshhold: '))
            for fold_idx, fold in enumerate(folds_data):
                plt.subplot(2, 3, fold_idx + 1)
                plt.title(f"Fold {fold_idx + 1}")

                # Data preparation
                predictions = np.array(fold['predictions'])
                y_test = np.array(fold['y_test'])
                test_labels = np.array(fold['test_labels'])
                predicted_labels = (predictions > 0.5).astype(int)

                # Plot original and solution data
                plt.plot(x_test, y_test, label=f'Data {fold_idx+1}', lw=0.5, color='b')
                plt.plot(x_test, test_labels, label=f"Solution {fold_idx+1}", lw=1.6, color='g')

                # Highlight predicted events
                predicted_gw_indices = np.where(predicted_labels == 1)[0]
                if len(predicted_gw_indices) != 0:
                    grouped_gw_indices = []
                    current_group = [predicted_gw_indices[0]]

                    for i in range(1, len(predicted_gw_indices)):
                        if predicted_gw_indices[i] - predicted_gw_indices[i - 1] <= 1:  # Points are consecutive
                            current_group.append(predicted_gw_indices[i])
                        else:
                            if len(current_group) >= threshold:  # Only keep groups meeting the threshold
                                grouped_gw_indices.append(current_group)
                            current_group = [predicted_gw_indices[i]]

                    # Check the last group
                    if len(current_group) >= threshold:
                        grouped_gw_indices.append(current_group)

                    # Highlight the regions for valid groups
                    for i, group in zip(range(len(grouped_gw_indices)), grouped_gw_indices):
                        plt.axvspan(x_test[group[0]], x_test[group[-1]], color="red", alpha=0.3, label="Predicted event" if i == 0 else "")


                plt.legend()

            # Adjust layout and show the plot
            plt.tight_layout()
            plt.show()

            # Optionally save the plot
            if save_option.lower() == 'y':
                save_fig = input("Would you like to save the previously generated figure? y/n\n")
                if save_fig.lower() == 'y':
                    save_path = f'../Figures/{filename_start}_timesteps{time_steps}_SNR{SNR}_epoch{epoch}_lamd{clicked_lambda}_eta{clicked_eta}_boost{boost:.1f}.pdf'
                    plt.savefig(save_path)
                    print(f"Figure saved to {save_path}")

class GWSignalGenerator:
    def __init__(self, signal_length: int):
        """
        Initialize the GWSignalGenerator with a signal length.
        """
        self.signal_length = signal_length
        self.labels = np.zeros(signal_length, dtype=int)  # Initialize labels to 0 (background noise)
        self.regions = []  # Store regions for visualization or further analysis

    def add_gw_event(self, y, start, end, amplitude_factor=0.2, spike_factor=0.5, spin_start=1, spin_end=20, scale=1):
        """
        Adds a simulated gravitational wave event to the signal and updates labels for its phases.
        Includes a spin factor that increases during the inspiral phase.

        Parameters:
        y:                Signal to append GW event to.
        start:            Start index for GW event.
        end:              End index for GW event.
        amplitude_factor: Peak of the oscillating signal in the insipral phase.
        spike_factor:     Peak of the signal in the merge phase.
        spin_start:       Oscillation frequency of the start of the inspiral phase.
        spin_end:         Oscillation frequency of the end of the inspiral phase.
        scale:            Scale the amplitude of the entire event.

        returns:
        Various parameters to be used by apply_events function
        """
        event_sign = np.random.choice([-1, 1])  # Random polarity for the GW event

        amplitude_factor=amplitude_factor*scale
        spike_factor=spike_factor*scale

        # Inspiral phase
        inspiral_end = int(start + 0.7 * (end - start))  # Define inspiral region as 70% of event duration
        time_inspiral = np.linspace(0, 1, inspiral_end - start)  # Normalized time for the inspiral
        amplitude_increase = np.linspace(0, amplitude_factor, inspiral_end - start)
        
        # Spin factor: linearly increasing frequency
        spin_frequency = np.linspace(spin_start, spin_end, inspiral_end - start)  # Spin frequency in Hz
        spin_factor = np.sin(2 * np.pi * spin_frequency * time_inspiral)
        
        y[start:inspiral_end] += event_sign * amplitude_increase * spin_factor
        # self.labels[start:inspiral_end] = 1  # Set label to 1 for inspiral

        # Merger phase
        merge_start = inspiral_end
        merge_end = merge_start + int(0.1 * (end - start))  # Define merger as 10% of event duration
        y[merge_start:merge_end] += event_sign * spike_factor * np.exp(-np.linspace(3, 0, merge_end - merge_start))
        # self.labels[merge_start:merge_end] = 2  # Set label to 2 for merger

        # Ringdown phase
        dropoff_start = merge_end
        dropoff_end = dropoff_start + int(0.2 * (end - start))  # Define ringdown as 20% of event duration
        dropoff_curve = spike_factor * np.exp(-np.linspace(0, 15, dropoff_end - dropoff_start))
        y[dropoff_start:dropoff_end] += event_sign * dropoff_curve
        # self.labels[dropoff_start:dropoff_end] = 3  # Set label to 3 for ringdown

        # We cut off 2/3rds of the ringdown event due to the harsh exponential supression.
        # It is not expected that the NN will detect anything past this and may cause confusion for the program.
        self.labels[start:(2*dropoff_start+dropoff_end)//3] = 1

        # Store region details for visualization or debugging
        self.regions.append((start, end, inspiral_end, merge_start, merge_end, dropoff_start, dropoff_end))

    def generate_random_events(self, num_events: int, event_length_range: tuple, scale=1, 
                               amplitude_factor_range = (0, 0.5), spike_factor_range = (0.2, 0.8),
                               spin_start_range = (1, 5), spin_end_range = (5, 20)):
        """
        Generate random gravitational wave events with no overlaps.
        """
        events = []
        used_intervals = []

        for _ in range(num_events):
            while True:
                # Randomly determine start and length of event
                event_length = random.randint(*event_length_range)
                event_start = random.randint(0, self.signal_length - event_length)
                event_end = event_start + event_length

                # Ensure no overlap
                if not any(s <= event_start <= e or s <= event_end <= e for s, e in used_intervals):
                    used_intervals.append((event_start, event_end))
                    break  # Valid event, exit loop

            # Randomize event properties
            amplitude_factor = random.uniform(*amplitude_factor_range)
            spike_factor = random.uniform(*spike_factor_range)
            
            # Randomize spin start and end frequencies
            spin_start = random.uniform(*spin_start_range)  # Starting spin frequency (in Hz)
            spin_end = random.uniform(*spin_end_range)  # Ending spin frequency (in Hz)

            events.append((event_start, event_end, amplitude_factor * scale, spike_factor * scale, spin_start, spin_end))

        return events

    def apply_events(self, y, events):
        """
        Apply generated events generated by add_gw_signal function to the input signal.
        Can be manually created using this function 
        """
        for start, end, amplitude, spike, spin_start, spin_end in events:
            self.add_gw_event(y, start, end, amplitude_factor=amplitude, spike_factor=spike, spin_start=spin_start, spin_end=spin_end)
    
############ Metric functions ############
def MSE(y,ytilde):
    n = len(y)
    return 1/n * np.sum(np.abs(y-ytilde)**2)

############ Activation functions ############

class Activation:
    def __init__(self):
        pass 

class Activation:

    def __init__(self, acitvation_name:str, is_derivative=False):
        """
        Creates a callable activation function corresponding to the string 'acitvation_name' given.
        """
        self.activation_functions =            [Activation.Lrelu, Activation.relu, 
                                                Activation.sigmoid, Activation.tanh]
        self.activation_functions_derivative = [Activation.Lrelu_derivative, Activation.relu_derivative, 
                                                Activation.sigmoid_derivative, Activation.tanh_derivative]
        self.activation_functions_name = ["LRELU", "RELU", "SIGMOID", "TANH"]

        self.acitvation_name = acitvation_name
        try:
            index = self.activation_functions_name.index(acitvation_name.upper())
            self.activation_func, self.activation_func_derivative  = self.activation_functions[index], self.activation_functions_derivative[index]
        except:
            raise TypeError(f"Did not recognize '{acitvation_name}' as an activation function.")

        self._call = self.activation_func
        if is_derivative:   # Then call-method will return derivative instead
            self._call = self.activation_func_derivative

    def __call__(self, z):
        return self._call(z)

    def __str__(self):
        return self.acitvation_name
    
    def derivative(self) -> Activation:
        return Activation(self.acitvation_name, True)
    
    @staticmethod
    def sigmoid(z):
        """Sigmoid activation function."""
        return 1 / (1 + anp.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        """Derivative of the Sigmoid activation function."""
        sigmoid_z = Activation.sigmoid(z)
        return sigmoid_z * (1 - sigmoid_z)

    @staticmethod
    def relu(z):
        """ReLU activation function."""
        return np.where(z > 0, z, 0)

    @staticmethod
    def relu_derivative(z):
        """Derivative of ReLU activation function."""
        return np.where(z > 0, 1, 0)

    @staticmethod
    def Lrelu(z, alpha=0.01):
        """Leaky ReLU activation function."""
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def Lrelu_derivative(z, alpha=0.01):
        """Derivative of Leaky ReLU activation function."""
        return np.where(z > 0, 1, alpha)
    
    @staticmethod
    def tanh(z):
        return np.tanh(z)
    
    @staticmethod
    def tanh_derivative(z):
        return 1 / np.cosh(z)**2


############ Cost/gradient functions ############
def gradientOLS(X, y, beta):
    n=len(y)

    return 2.0/n*X.T @ (X @ (beta)-y)

def CostOLS(X, y, theta):
    n=len(y)
    return 1/n * anp.sum((y-X @ theta)**2)

def CostRidge(X, y, theta, lmbda):
    n = len(y)
    return (1.0 / n) * anp.sum((y-X @ theta) ** 2) + lmbda / n * anp.sum(theta**2)

class LogisticCost:

    def __init__(self, exp_clip=1e3, log_clip=1e-13, hypothesis=Activation.sigmoid):
        """
        Logistic cost function which removes too high values for exp/too low for log.
        """
        self.exp_clip = exp_clip; self.log_clip = log_clip
        self.hypothesis_func = hypothesis

    def __call__(self, x, y, w, lmbda):

        # computing hypothesis
        z = anp.dot(x,w)
        z = anp.clip(z, -self.exp_clip, self.exp_clip)
        h = self.hypothesis_func(z)

        cost = (-1 / len(y)) * anp.sum(y * anp.log(h + self.log_clip) + (1 - y) * anp.log(1 - h + self.log_clip))
        reg_term = lmbda * anp.sum(w[1:] ** 2)

        # Compute total cost
        return cost + reg_term
    
class AutoGradCostFunction:

    def __init__(self, cost_function:callable, argument_index=2, elementwise=False):
        """
        Creates callable gradient of given cost function. The cost function is a property of the class, and so changing it will change the gradient.
        Assumes that the cost_function has a function call on the form cost_function(X, y, theta).

        Arguments 
            * cost_function:        callable cost function 
            * argument_index:       index of argument to take gradient over
        """
        self._gradient = grad(cost_function, argument_index)
        if elementwise:
            self._gradient = elementwise_grad(cost_function, argument_index)
        self._cost_function = cost_function
        self._argument_index = argument_index
        self.elementwise = elementwise

    @property
    def cost_function(self):
        return self._cost_function
    
    @cost_function.setter 
    def cost_function(self, new_cost_function):
        self._cost_function = new_cost_function 
        self._gradient = grad(new_cost_function, self._argument_index)
        if self.elementwise:
            self._gradient = elementwise_grad(new_cost_function, self._argument_index)

    def __call__(self, X, y, theta, lmbda):
        """
        Returns gradient of current cost function.
        """
        return self._gradient(X, y, theta, lmbda)

############ Optimization methods (and related) ############
class LearningRate:

    def __init__(self, t0:float, t1:float, N:int, batch_size:int, name=None, const=None):
        
        self.name = name 
        if name is None:
            if const is None:
                self.name = f"callable({t0}, {t1})"
            else:
                self.name = str(const)

        self.t0 = t0; self.t1 = t1
        self.N = N; self.batch_size = batch_size
        self.const = const 

        self._call = self._varying_learning_rate
        if not (const is None):
            self._call = self._const_learning_rate

    def _varying_learning_rate(self, epoch, i):
        t = epoch * int(self.N / self.batch_size) + i 
        return self.t0 / (t + self.t1)
    
    def _const_learning_rate(self, epoch, i):
        return self.const
    
    def __call__(self, epoch, i):
        return self._call(epoch, i)
    
    def __str__(self):
        return self.name 
 
class Optimizer:
    """
    Arguments
        * learning_rate:        number or callable(epochs, i), essentially the coefficient before the gradient
        * momentum:             number added in descent
        * epsilon:              small number used to not divide by zero in Adagrad, RMSprop and Adam
        * beta1:                used in Adam, in bias handling
        * beta2:                used in Adam, in bias handling
        * decay_rate:           used in ... 
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9, epsilon=1e-8, beta1=0.9, beta2=0.999, decay_rate=0.9):
        """
        Class to be inherited by PlaneGradient, Adagrad, RMSprop or Adam, representing the method of choice for Stochastic Gradient Descent (SGD).
        """
        # self._learning_rate = learning_rate
        self.learning_rate = learning_rate # calls property method

        self.momentum = momentum
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay_rate = decay_rate
        self.velocity = None  # For momentum term

    @property
    def learning_rate(self):
        return self._learning_rate 
    
    @learning_rate.setter
    def learning_rate(self, new_learning_rate):
        # Makes sure learning_rate is a callable
        if not isinstance(new_learning_rate, LearningRate):
            tmp = new_learning_rate
            
            self._str_learning_rate = str(tmp)
            new_learning_rate = LearningRate(0, 0, 0, 0, self._str_learning_rate, const=new_learning_rate)
            self._learning_rate = new_learning_rate
        else:
            self._learning_rate = new_learning_rate
            self._str_learning_rate = "callable"

    def initialize_velocity(self, theta):
        if self.velocity is None:
            self.velocity = np.zeros_like(theta)

    def __call__(self, theta:np.ndarray, gradient:np.ndarray, epoch_index:int, batch_index:int):
        """
        Arguments
        * theta:            variable to be updated
        * gradient:         gradient
        * epoch_index:      current epoch
        * batch_index:      current batch

        Returns
        updated theta
        """
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def copy(self):
        """
        Creates and returns a copy of itself
        """

        raise Warning("Creating copy of Optimizer class, you probably want to create a copy of a subclass of Optimizer?")
        optimizer = Optimizer(learning_rate=self.learning_rate, momentum=self.momentum, 
                              epsilon=self.epsilon, beta1=self.beta1, beta2=self.beta2, decay_rate=self.decay_rate)
        
        return optimizer

    def __str__(self):
        return "Not defined"
    
    def __float__(self):
        return self._learning_rate.const

class PlaneGradient(Optimizer):

    __doc__ = Optimizer.__doc__
    def __init__(self, learning_rate=0.01, momentum=0.9, epsilon=None, beta1=None, beta2=None, decay_rate=None):
        """
        Class implementing basic Gradient Descent with optional momentum. Does support stochastic GD as well.
        """
        super().__init__(learning_rate, momentum)
        self.name = "PlaneGradient"

    def __call__(self, *args):

        theta, gradient = args[0], args[1]; epoch_index = None; batch_index = None 

        if len(args) == 4:
            epoch_index, batch_index = args[2], args[3]
        self.initialize_velocity(theta)

        self.velocity = self.momentum * self.velocity - self._learning_rate(epoch_index, batch_index) * gradient

        # Apply momentum
        theta += self.velocity
        return theta

    def __str__(self):
        return f"Plane Gradient descent: eta: {self.learning_rate}, momentum: {self.momentum}"
    
    def copy(self):
        """
        Creates and returns a copy of itself
        """

        optimizer = PlaneGradient(learning_rate=self.learning_rate, momentum=self.momentum, 
                              epsilon=self.epsilon, beta1=self.beta1, beta2=self.beta2, decay_rate=self.decay_rate)
        
        return optimizer

class Adagrad(Optimizer):

    __doc__ = Optimizer.__doc__
    def __init__(self, learning_rate=0.01, epsilon=1e-8, momentum=None, beta1=None, beta2=None, decay_rate=None):
        """
        Class implementing the Adagrad optimization algorithm. 
        Adagrad adapts the learning rate for each parameter based on the accumulation of past squared gradients.
        """

        super().__init__(learning_rate, epsilon=epsilon)
        self.G = None  # Accumulated squared gradients
        self.name = "Adagrad"
    
    def initialize_accumulation(self, theta):
        # Initializes accumulation matrix G if not already initialized
        if self.G is None:
            self.G = 0

    def __call__(self, *args):
        theta, gradient, epoch_index, batch_index = args[0], args[1], args[2], args[3]

        self.initialize_accumulation(theta)

        # Accumulating squared gradients
        self.G += gradient*gradient

        #Updating theta
        theta -= (self._learning_rate(epoch_index, batch_index) / (np.sqrt(self.G) + self.epsilon)) * gradient
        
        return theta
    
    def __str__(self):
        return f"Adagrad: eta: {self._str_learning_rate}, eps: {self.momentum}"
    
    def copy(self):
        """
        Creates and returns a copy of itself
        """

        optimizer = Adagrad(learning_rate=self.learning_rate, momentum=self.momentum, 
                              epsilon=self.epsilon, beta1=self.beta1, beta2=self.beta2, decay_rate=self.decay_rate)
        
        return optimizer

class RMSprop(Optimizer):
    __doc__ = Optimizer.__doc__
    def __init__(self, learning_rate=0.01, decay_rate=0.99, epsilon=1e-8, momentum=None, beta1=None, beta2=None):
        """
        Class implementing the RMSprop optimization algorithm.
        RMSprop maintains a moving average of the squared gradients to normalize the gradient.
        """

        super().__init__(learning_rate, epsilon=epsilon, decay_rate=decay_rate)
        self.G = None
        self.name = "RMSprop"

    def initialize_accumulation(self, theta):
        # Initializes accumulation matrix G if not already initialized
        if self.G is None:
            self.G = np.zeros_like(theta)

    def __call__(self, *args):
        theta, gradient, epoch_index, batch_index = args[0], args[1], args[2], args[3]

        self.initialize_accumulation(theta)

        # Updating moving average of the squared gradients
        self.G = self.decay_rate * self.G + (1 - self.decay_rate) * gradient*gradient

        # Update theta
        theta -= (self._learning_rate(epoch_index, batch_index) / (np.sqrt(self.G) + self.epsilon)) * gradient
        return theta

    def __str__(self):
        return f"RMSprop: eta: {self._str_learning_rate}, eps: {self.momentum}, decay_rate = {self.decay_rate}"
    
    def copy(self):
        """
        Creates and returns a copy of itself
        """

        optimizer = RMSprop(learning_rate=self.learning_rate, momentum=self.momentum, 
                              epsilon=self.epsilon, beta1=self.beta1, beta2=self.beta2, decay_rate=self.decay_rate)
        
        return optimizer

class Adam(Optimizer):
    __doc__ = Optimizer.__doc__
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=None, momentum=None):
        """
        Class implementing the Adam optimization algorithm.
        Adam combines the advantages of both RMSprop and momentum by maintaining both first and second moment estimates.
        """

        super().__init__(learning_rate, epsilon=epsilon, beta1=beta1, beta2=beta2)
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0  # Time step

        self.name = "Adam"

    def initialize_moments(self, theta):
        # Initializes first and second moment vectors if not already initialized
        if self.m is None:
            self.m = np.zeros_like(theta)
        if self.v is None:
            self.v = np.zeros_like(theta)

    def __call__(self, *args):
        theta, gradient, epoch_index, batch_index = args[0], args[1], args[2], args[3]

        self.t += 1
        self.initialize_moments(theta)

        # Update biased first and second moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2

        # Compute bias-corrected moments
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Update theta
        theta -= (self._learning_rate(epoch_index, batch_index) / (np.sqrt(v_hat) + self.epsilon)) * m_hat
        return theta
    
    def __str__(self):
        return f"Adam: eta: {self._str_learning_rate}, eps: {self.momentum}, beta1 = {self.beta1}, beta2 = {self.beta2}"
    
    def copy(self):
        """
        Creates and returns a copy of itself
        """

        optimizer = Adam(learning_rate=self.learning_rate, momentum=self.momentum, 
                              epsilon=self.epsilon, beta1=self.beta1, beta2=self.beta2, decay_rate=self.decay_rate)
        
        return optimizer


class Scalers:

    def __init__(self, scaler_name:str):
        scaler_names = ["STANDARD", "MINMAX"]
        scalers = [Scalers.standard_scaler, Scalers.minmax_scaler]

        try:
            index = scaler_names.index(scaler_name.upper())
            self._call = scalers[index]
        except:
            raise TypeError(f"Did not recognize '{scaler_name}' as a scaler type, available: ['standard', 'minmax']")
        
        self.scaler_name = scaler_name

    def __call__(self, X_train:np.ndarray, X_test:np.ndarray) -> list[np.ndarray, np.ndarray]:
        return self._call(X_train, X_test)
    
    def __str__(self):
        return self.scaler_name
    
    @staticmethod
    def standard_scaler(X_train, X_test):
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test
    
    @staticmethod
    def minmax_scaler(X_train, X_test):
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test

class Loss:
    data = {}

    def __call__(self, y_true, y_pred, epoch=None):
        """
        Computes the loss value.

        Arguments:
        - y_true: True labels.
        - y_pred: Predicted outputs.

        Returns:
        - Loss value.
        """
        raise NotImplementedError("Forward method not implemented.")

    def gradient(self, y_true, y_pred, epoch=None):
        """
        Computes the gradient of the loss with respect to y_pred.

        Arguments:
        - y_true: True labels.
        - y_pred: Predicted outputs.

        Returns:
        - Gradient of the loss with respect to y_pred.
        """
        raise NotImplementedError("Backward method not implemented.")


class DynamicallyWeightedLoss(Loss):

    def __init__(self, initial_boost=1.0, epochs=None, labels=None, weight_0=1, epsilon=1e-8):
        self.initial_boost = initial_boost; self.epochs = epochs
        self.labels = labels; self.weight_0 = weight_0
        self._epsilon = epsilon 

        self.data = {"initial_boost": self.initial_boost,
                  "epochs": epochs,
                  "weight_0": weight_0}
        self.type = "binary_crossentropy"         
        
        self._current_epoch = None
        self.weight_1 = None
        
    def __call__(self, y_true:np.ndarray, y_pred:np.ndarray, epoch=None):
        """
        Computes the weighted binary cross-entropy loss.
        """
        if epoch == 0: # Then WeightedBinaryCrossEntropyLoss
            weight_1 = (len(self.labels) - np.sum(self.labels)) / np.sum(self.labels)
            weight_0 = 1
            y_pred = np.clip(y_pred, self._epsilon, 1 - self._epsilon)  # To prevent log(0)
            loss = -np.mean(
                weight_1 * y_true * np.log(y_pred) +
                weight_0 * (1 - y_true) * np.log(1 - y_pred)
            )
            return loss
    
        if epoch != self._current_epoch:
            self.calculate_weights(epoch)

        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # To prevent log(0)
        loss = -np.mean(
            self.weight_1 * y_true * np.log(y_pred) +
            self.weight_0 * (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    def gradient(self, y_true:np.ndarray, y_pred:np.ndarray, epoch=None):
        """
        Computes the gradient of the loss with respect to y_pred.
        """
        if epoch != self._current_epoch:
            self.calculate_weights(epoch)
        
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # To prevent division by zero
        grad = - (self.weight_1 * y_true / y_pred) + (self.weight_0 * (1 - y_true) / (1 - y_pred))
        return grad
    
    def calculate_weights(self, epoch:int) -> None:
        self._current_epoch = epoch 
        initial_boost = self.initial_boost
        scale = initial_boost - (initial_boost - 1) * epoch / self.epochs
        self.weight_1 = (len(self.labels) - np.sum(self.labels)) / np.sum(self.labels) * scale
        
        return {0: self.weight_0, 1: self.weight_1}

class WeightedBinaryCrossEntropyLoss(Loss):
    def __init__(self, weight_0, weight_1):
        """
        Initializes the loss function with class weights.

        Arguments:
        - class_weight: Dictionary with weights for each class {0: weight_0, 1: weight_1}.
        """
        self.weight_1 = weight_1
        self.weight_0 = weight_0

        self.data = {"weight_0": weight_0,
                     "weight_1": weight_1}
        self.type = "binary_crossentropy"


    def __call__(self, y_true, y_pred):
        """
        Computes the weighted binary cross-entropy loss.
        """
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # To prevent log(0)
        loss = -np.mean(
            self.weight_1 * y_true * np.log(y_pred) +
            self.weight_0 * (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    def gradient(self, y_true, y_pred):
        """
        Computes the gradient of the loss with respect to y_pred.
        """
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # To prevent division by zero
        grad = - (self.weight_1 * y_true / y_pred) + (self.weight_0 * (1 - y_true) / (1 - y_pred))
        return grad
    
    def calculate_weights(self, epoch:int) -> None:
        return {0: self.weight_0, 1: self.weight_1}

    
class FocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Initializes the focal loss function.

        Arguments:
        - alpha: Weighting factor for the positive class.
        - gamma: Focusing parameter.
        """
        self.alpha = alpha
        self.gamma = gamma

        self.data = {"alpha": alpha,
                     "gamma": gamma}


    def __call__(self, y_true, y_pred, epoch=None):
        """
        Computes the focal loss.
        """
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        loss = -np.mean(alpha_t * (1 - pt) ** self.gamma * np.log(pt))
        return loss

    def gradient(self, y_true, y_pred, epoch=None):
        """
        Computes the gradient of the loss with respect to y_pred.
        """
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        grad = -alpha_t * (1 - pt) ** (self.gamma - 1) * (
            self.gamma * pt * np.log(pt) + (1 - pt)
        ) / pt
        grad *= y_pred - y_true
        return grad

def weighted_Accuracy(y_true:np.ndarray, y_pred:np.ndarray, eps:float=1e-13) -> float:
    """
    Calculates a weighted accuracy, usage is in particular for binary unbiased features.  
    """
    # Making sure the correct dim
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    N = len(y_true)
    TP = np.sum((np.abs(y_true-1)< eps) & (np.abs(y_pred-1)< eps))
    TN = np.sum((np.abs(y_true)< eps) & (np.abs(y_pred)< eps))

    tmp = np.sum(y_true)
    if tmp == 0:
        return TN / N 
    
    W_1 = N / tmp - 1

    A_w = 1 / (2*N * (1 - np.sum(y_true)/N)) * (TN + W_1 * TP)

    B = 1 / (2*(N - np.sum(y_true)))
    A_w = B * (TN + W_1*TP)
    return A_w 