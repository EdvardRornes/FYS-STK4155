from utils import *
# from autograd import grad

def gradientOLS(X, y, beta):
    n=len(y)
    return 2.0/n*X.T @ (X @ (beta)-y)

def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2) # Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X

planeGD = GradientDescent(momentum=0)
test = beta_giver(planeGD)


N = 1000
x = 10*np.random.rand(N,1)
# x = np.linspace(-10,10,N)
beta_true = np.array([[1, 1/2, -1/18, 3, 1, 1],]).T
func = f(*beta_true)
y = func(x)
X = np.c_[np.ones((N,len(beta_true)-1)), x]
beta = np.random.randn(np.shape(X)[1],1)

test.gradient = gradientOLS
beta = test(X, y, 100, theta=beta)

print(beta)

f_train = f(*beta[0])
x_test = np.linspace(0,10, 10_000)

plt.plot(x_test, func(x_test), label="analytic")
plt.plot(x_test, f_train(x_test), label="train")
plt.legend()
plt.show()


n = 1000
X = np.c_[np.ones((n,4)), x]
# Hessian matrix
H = (2.0/n)* X.T @ X
# Get the eigenvalues
EigValues, EigVectors = np.linalg.eig(H)
print(f"Eigenvalues of Hessian Matrix:{EigValues}")

# beta_linreg = np.linalg.inv(X.T @ X + 1e-3*np.identity(np.shape(X))) @ X.T @ y
# print(beta_linreg)
beta = np.random.randn(5,1)

eta = 1.0/np.max(EigValues)
Niterations = 1000

for iter in range(Niterations):
    gradient = (2.0/n)*X.T @ (X @ beta-y)
    beta -= eta*gradient

print(beta)
f_train = f(*beta)
print(f_train.coeffs)
x_test = np.linspace(0,10, 10_000)

plt.plot(x_test, func(x_test), label="analytic")
plt.plot(x_test, f_train(x_test), label="train")
plt.legend()
plt.show()