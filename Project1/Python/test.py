from utils import * 
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def test_Design_matrix2D(design_matrix=Design_matrix2D):
    eps = 1e-14

    N = 4
    x = 2*np.ones(N)
    y = 3*np.ones(N)
    
    p = 4; n = len(x)
    deg = (p + 1) * (p + 2) // 2
    Y = np.zeros((n, deg))
    for i in range(n):
        Y[i, 0] = 1
        Y[i, 1] = x[i]
        Y[i, 2] = y[i]
        Y[i, 3] = x[i]**2
        Y[i, 4] = x[i]*y[i]
        Y[i, 5] = y[i]**2
        Y[i, 6] = x[i]**3
        Y[i, 7] = x[i]**2 * y[i]
        Y[i, 8] = x[i] * y[i]**2
        Y[i, 9] = y[i]**3
        Y[i,10] = x[i]**4
        Y[i,11] = x[i]**3*y[i]
        Y[i,12] = x[i]**2*y[i]**2
        Y[i,13] = x[i]*y[i]**3
        Y[i,14] = y[i]**4

    assert np.linalg.norm(Y-design_matrix(x,y,p)) < eps, f"test_Design_matrix2D failed w/ |X-Y|={np.linalg.norm(Y-design_matrix(x,y,p))}"

if __name__ == "__main__":
    test_Design_matrix2D()


    
    