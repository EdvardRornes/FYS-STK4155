import numpy as np
import matplotlib.pyplot as plt
from utils import *

def bootstrap(data, n_bootstraps=10, plot_first=False):
    """Performs a bootstraping using the OLS method to train the model.

    Args:
        data (dict): Data dictionairy containing training and test set.
        n_bootstraps (int, optional): Number of bootstraps to perform. Defaults to 10.
        plot_first (bool, optional): Plots predicted surface of first model training in bootstrap when true. Defaults to False.

    Returns:
        (float, float, float, float, float): Returns the error, bias and variance of bootstrap as well as average train loss and average test loss.
    """
    # Get data
    train_X, train_Z, test_X, test_Z = (
        data["train_X"],
        data["train_Z"],
        data["test_X"],
        data["test_Z"],
    )

    # Array for storing finds
    Z_pred = np.empty((test_Z.shape[0], n_bootstraps))

    bootstraps_train_loss = []
    bootstraps_test_loss = []

    # Run the bootstrap
    for i in range(n_bootstraps):
        # Resample
        data["train_X"], data["train_Z"] = resample(train_X, train_Z)

        # Run model
        if i > 0:
            plot_first = False
        run_data = fit_OLS(data, plot_or_not=plot_first)

        bootstraps_train_loss.append(run_data["train_loss"])
        bootstraps_test_loss.append(run_data["test_loss"])

        # Make prediction
        Z_pred[:, i] = run_data["predictor"](test_X).ravel()

    # Compute error, bias and variance estimators
    error = np.mean(np.mean((test_Z - Z_pred) ** 2, axis=1, keepdims=True))
    bias = np.mean((test_Z - np.mean(Z_pred, axis=1, keepdims=True)) ** 2)
    variance = np.mean(np.var(Z_pred, axis=1, keepdims=True))

    return (
        error,
        bias,
        variance,
        sum(bootstraps_train_loss) / len(bootstraps_train_loss),
        sum(bootstraps_test_loss) / len(bootstraps_test_loss),
    )

def tradeoff_experiment(
    num_points=100,
    num_features=100,
    n_bootstraps=50,
    nth=10,
    data=None,
    seed=42,
    mark_deg_nn=True,
    filename1=False,
    filename2=False,
    sigma2=0.0,
):
    """Performs a bootstrapping experiment to showcase the bias-variance tradeoff. Performs a bootstrap at different number of features up until num_features.

    Args:
        num_features (int): Maximum number of features to use for the models.
        num_points (int, optional): Number of points to generate for the Franke function dataset.
        n_bootstraps (int, optional): Number of bootstraps to perform per feature number. Defaults to 50.
        nth (int, optional): Plots nth model prediction surface. Defaults to 3.
        data (dict, optional): Optional data dictionary containing custom data to use instead of Franke function. Needs to contain x, y and z arrays for datapoints. Defaults to None.
        seed (int, optional): Seed to use for experiment. Defaults to 42.
        mark_deg_nn (boolean, optional): Marks the points where the number of features containing x and number of features containing y are the same.
        filename1/2 (str, optional): Optional filename for where to save figure 1 and 2. Defaults to None.
        sigma2 (float, optional): Variance of Gaussian noise to be added to generated Franke function data.

    Returns:
        (ndarray, ndarray, ndarray, ndarray): Feature number array with the corresponding error, bias and variance arrays.
    """

    # Set seed
    np.random.seed(seed)

    # Initialise data storage
    feature_numbers = np.arange(1, num_features + 1, 1)
    errors = np.zeros(num_features)
    biases = np.zeros(num_features)
    variances = np.zeros(num_features)

    train_losses = np.zeros(num_features)
    test_losses = np.zeros(num_features)

    # If no data is given, sample from Franke.
    if data is None:
        data = r2_sampling(num_points, sigma2=sigma2)

    # Start with test_index = None to make random_dat... generate test_indeces.
    # These same indeces is used throughout
    test_index = None

    # Run experiment for each feature number
    for num in feature_numbers:
        # Create feature matrix
        new_data = random_dataset_split_preparation(
            data, num, scale=True, test_index=test_index
        )
        test_index = new_data["test_index"]  # Ensure same test split

        # Compute error, bias and variance
        plot = False
        if num % nth == 0:
            plot = True
        error, bias, variance, train_mean, test_mean = bootstrap(
            new_data, n_bootstraps=n_bootstraps, plot_first=plot
        )

        # We need to scale bias and variance to get a meaningfull result
        bias, variance = bias, variance

        # Add the result
        errors[num - 1] = error
        biases[num - 1] = bias
        variances[num - 1] = variance

        train_losses[num - 1] = train_mean
        test_losses[num - 1] = test_mean

    ## Make plot of bias and variance

    # Highlight p(x)=a0+...+an0 x^n +...+a0n y^n (whole number final), plot biases and variances
    if mark_deg_nn:
        max_deg = maximal_degree(num_features)
        xy_deg_indeces = nn_deg_indeces(max_deg)
        plt.plot(feature_numbers[xy_deg_indeces], biases[xy_deg_indeces], "o", c="b")

    plt.plot(feature_numbers, errors, label="error", c="r")
    plt.plot(feature_numbers, biases, label="bias", c="b")
    plt.plot(feature_numbers, variances, label="variance", c="g")

    # Add legend and labels
    plt.legend()
    plt.xlabel("Number of features")
    plt.title("The Bias Variance Tradeoff")

    # Save figure if filename given
    if filename1:
        plt.savefig(filename1, bbox_inches="tight")
    plt.show()

    plt.plot(feature_numbers, train_losses, label="Bootstrap Train Loss", c="b")
    plt.plot(feature_numbers, test_losses, label="Bootstrap Test Loss", c="r")

    # Add legend and labels
    plt.legend()
    plt.xlabel("Number of features")
    plt.ylabel("MSE")
    plt.title("Bootstrap Train/Test loss")

    # Save figure if filename given
    if filename2:
        plt.savefig(filename2, bbox_inches="tight")
    plt.show()

    return feature_numbers, errors, biases, variances