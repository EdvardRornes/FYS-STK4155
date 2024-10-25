def f(*args):

    if len(args) == 1:
        x = args[0]
        print(x)
    
    else:
        print(args)
        x, y = args
        print(x)
        print(y)

import numpy as np
x = np.array([1,2,3])
f(*x)
xy = [[1,2,3], [4,5,6]]
f(*xy)

