import numpy as np
from numpy.random import rand
from numpy.random import seed
from matplotlib import pyplot
import matplotlib.pyplot as plt 


class f:

    def __init__(self, a0, a1, a2, a3=None):
        self.a0 = a0; self.a1 = a1; self.a2 = a2; self.a3 = a3

        if a3 is None:
            self.a3 = 0

    def __call__(self, x):
        return self.a0 + self.a1 * x + self.a2 * x**2 + self.a3 * x**3

    def derivative(self):
        return f(self.a1, 2*self.a2, 3*self.a3)
    


# gradient descent algorithm (Code from lecture notes)
def gradient_descent(objective, derivative, bounds, n_iter, learning_rate, momentum):
    # track all solutions
    solutions, scores = list(), list()
    # generate an initial point
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    # keep track of the change
    change = 0.0
    # run the gradient descent
    for i in range(n_iter):
        # calculate gradient
        gradient = derivative(solution)
        # calculate update
        new_change = learning_rate * gradient + momentum * change
        # take a step
        solution = solution - new_change
        # save the change
        change = new_change
        # evaluate candidate point
        solution_eval = objective(solution)
        # store solution
        solutions.append(solution)
        scores.append(solution_eval)
        # report progress
        print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
    return [solutions, scores]

def stochastic_gradient_descent(x:np.ndarray, y:np.ndarray, m:int, M:int, n_epochs:int, learning_schedule:callable):
    n = len(x)

    if isinstance(learning_schedule, int) or isinstance(learning_schedule, float):
        tmp = learning_schedule
        def learning_schedule(t):
            return tmp
        
    X = np.c_[np.ones((n,1)), x]

    theta = np.random.randn(2,1)


    for epoch in range(n_epochs):
        for i in range(m):
            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]
            gradients = (2.0/M)* xi.T @ ((xi @ theta)-yi)
            eta = learning_schedule(epoch*m+i)
            theta = theta - eta*gradients

    return theta

# seed the pseudo random number generator
seed(4)
# define range for input
bounds = np.asarray([[-1.0, 1.0]])
# define the total iterations
n_iter = 30
# define the step size
learning_rate = 0.1
# define momentum
momentum = 0.3

a0 = 1; a1 = -1/2; a2 = 1/18; a3 = 1/2
objective = f(a0, a1, a2, a3); derivative = objective.derivative()
x = np.linspace(-10, 10, 100_000)
plt.plot(x, objective(x), label="f(x)")
plt.plot(x, derivative(x), label="f'(x)")
plt.legend()
plt.show()

# perform the gradient descent search with momentum
solutions, scores = gradient_descent(objective, derivative, bounds, n_iter, learning_rate, momentum)
# sample input range uniformly at 0.1 increments
inputs = np.arange(bounds[0,0], bounds[0,1]+0.1, 0.1)
# compute targets
results = objective(inputs)
# create a line plot of input vs result
plt.plot(inputs, results)
# plot the solutions found
plt.plot(solutions, scores, '.-', color='red')
# show the plot
plt.show()



########## Stochastic gradient descent ###########
n = 100
n_epochs = 50
M = 5   #size of each minibatch
m = int(n/M) #number of minibatches
t0, t1 = 5, 50

def learning_schedule(t):
    return t0/(t+t1)

n = 100
x = 2*np.random.rand(n,1)
x = np.linspace(-10,10, 100)
y = objective(x)

theta = stochastic_gradient_descent(x, y, m, M, n_epochs, learning_schedule)

xnew = np.array([[x[0]],[x[-1]]])
Xnew = np.c_[np.ones((2,1)), xnew]
ypredict = Xnew.dot(theta)

plt.plot(xnew, ypredict, "r-")
plt.plot(x, y ,'ro')
# plt.axis([0,2.0,0, 15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
# plt.title(r'Random numbers ')
plt.show()