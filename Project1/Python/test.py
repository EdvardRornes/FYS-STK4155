import numpy as np
from utils import *

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

# Can delete?

# # Sample data
# x = np.random.rand(10)  # 5 random x values
# y = np.random.rand(10)  # 5 random y values
# degree = 7  # Set the degree of the polynomial

# # Generate design matrices
# X_manual = Design_matrix2D(x, y, degree)
# X_poly = Design_Matrix(x, y, degree)

# # Compare the two design matrices
# print("Design Matrix (Manual):")
# print(X_manual)

# print("\nDesign Matrix (PolynomialFeatures):")
# print(X_poly)

# # Check if they are approximately equal
# if np.allclose(X_manual, X_poly, rtol=1e-5, atol=1e-8):
#     print("\nThe matrices are equivalent!")
# else:
#     print("\nThe matrices are NOT equivalent.")

    
    
# import itertools
# import matplotlib.pyplot as plt
# m = 5
# n = 5
# x = np.zeros(shape=(m, n))
# plt.figure(figsize=(5.15, 5.15))
# plt.clf()
# plt.subplot(111)
# marker = itertools.cycle(('o', 'v', '^', '<', '>', 's', '8', 'p'))

# ax = plt.gca().set_prop_cycle(None) 
# for i in range(1, n):
#     x = np.dot(i, [1, 1.1, 1.2, 1.3])
#     y = x ** 2
#     #
#     #for matplotlib before 1.5, use
#     #color = next(ax._get_lines.color_cycle)
#     #instead of next line (thanks to Jon Loveday for the update)
#     #
#     color = next(ax._get_lines.prop_cycler)['color']
#     plt.plot(x, y, linestyle='', markeredgecolor='none', marker=marker.next(), color=color)
#     plt.plot(x, y, linestyle='-', color = color)
# plt.ylabel(r'$y$', labelpad=6)
# plt.xlabel(r'$x$', labelpad=6)
# plt.show()
