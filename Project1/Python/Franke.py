import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from utils import *

def FrankeFunction(x, y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

if __name__ == "__main__":
    latex_fonts()
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')  # Use add_subplot instead of gca

    # Make data.
    x = np.arange(0, 1, 0.01)
    y = np.arange(0, 1, 0.01)
    x, y = np.meshgrid(x, y)

    z = FrankeFunction(x, y)
    surface_3D(x, y, z, ax, fig)
    plt.savefig('Figures/FrankeFunction.pdf', transparent=True, bbox_inches='tight')

    plt.show()
