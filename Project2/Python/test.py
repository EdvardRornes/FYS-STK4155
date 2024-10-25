from new_utils import *
import sys, os

# Disable print
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore print
def enablePrint():
    sys.stdout = sys.__stdout__

def test_GD(eps=3):
    # blockPrint()

    N = 1000
    x = np.random.rand(N,1)
    beta_true = np.array([[1, -8, 16],]).T
    func = Polynomial(*beta_true)
    y = func(x)

    planeGD = PlaneGD(momentum=0, learning_rate=0.3590565341426001)
    test = DescentSolver(planeGD, 3)

    beta_test = np.random.randn(3, 1)

    test.gradient = gradientOLS
    beta_test = test(x, y, N)

    test_analyzer = DescentAnalyzer(x, y, "planeGD", 3, N)

    test_analyzer.run_analysis(gradientOLS, 0.3590565341426001)
    beta_test_analyzer = test_analyzer["thetas"]

    print(beta_test)
    print(beta_test_analyzer)

    enablePrint()
    assert np.linalg.norm(beta_true - beta_test) < eps, f"Test failed with error {np.linalg.norm(beta_true - beta_test)}"

def test_SGD(eps=3):
    blockPrint()

    N = 1000; batch_size = 5
    x = np.random.rand(N,1)
    beta_true = np.array([[1, -8, 16],]).T
    func = Polynomial(*beta_true)
    y = func(x)

    planeSGD = PlaneGD(momentum=0, learning_rate=0.3590565341426001)
    test = DescentSolver(planeSGD, 3, mode="SGD")

    beta_test = np.random.randn(N,1)

    test.gradient = gradientOLS
    beta_test = test(x, y, N, batch_size, theta=beta_test)

    enablePrint()
    assert np.linalg.norm(beta_true - beta_test) < eps, f"Test failed with error {np.linalg.norm(beta_true - beta_test)}"

def test_AdaGrad(eps=5):
    blockPrint()

    N = 10_000; batch_size = 5
    x = np.random.rand(N,1)
    beta_true = np.array([[1, -8, 16],]).T
    func = Polynomial(*beta_true)
    y = func(x)

    adagraddiradd = Adagrad(learning_rate=0.3590565341426001)
    test1 = DescentSolver(adagraddiradd, 3)
    test2 = DescentSolver(adagraddiradd, 3, mode="SGD")

    beta_test = np.random.randn(N,1)

    test1.gradient = gradientOLS; test2.gradient = gradientOLS
    beta_test1 = test1(x, y, N, theta=beta_test); beta_test2 = test2(x, y, N, batch_size, theta=beta_test)

    enablePrint()

    assert np.linalg.norm(beta_true - beta_test1) < eps, f"Test failed with error {np.linalg.norm(beta_true - beta_test1)}"
    assert np.linalg.norm(beta_true - beta_test2) < eps, f"Test failed with error {np.linalg.norm(beta_true - beta_test2)}"

def test_RMSprop(eps=3):
    blockPrint()
    

    N = 10_000; batch_size = 5
    x = np.random.rand(N,1)
    beta_true = np.array([[1, -8, 16],]).T
    func = Polynomial(*beta_true)
    y = func(x)

    # t0, t1 = 5, 50
    # def learning_schedule(epoch, i):
    #     t = epoch*m + i 
    #     return t0/(t+t1)

    RMS_propproppapp = RMSprop(learning_rate=0.3590565341426001)
    test1 = DescentSolver(RMS_propproppapp, 3)
    test2 = DescentSolver(RMS_propproppapp, 3, mode="SGD")

    beta_test = np.random.randn(N,1)

    test1.gradient = gradientOLS; test2.gradient = gradientOLS
    beta_test1 = test1(x, y, N, theta=beta_test)
    beta_test2 = test2(x, y, N, batch_size, theta=beta_test)

    enablePrint()
    print(beta_test1)
    print(beta_test2)
    assert np.linalg.norm(beta_true - beta_test1) < eps, f"Test failed with error {np.linalg.norm(beta_true - beta_test1)}"
    assert np.linalg.norm(beta_true - beta_test2) < eps, f"Test failed with error {np.linalg.norm(beta_true - beta_test2)}"

def test_Adam(eps=3):
    # blockPrint()

    N = 10_0; batch_size = 5
    x = np.random.rand(N,1)
    beta_true = np.array([[1, -8, 16],]).T
    func = Polynomial(*beta_true)
    y = func(x)

    adamramdampadamkada = Adam(learning_rate=0.3590565341426001)
    test1 = DescentSolver(adamramdampadamkada, 3)
    test2 = DescentSolver(adamramdampadamkada, 3, mode="SGD")

    # X = create_Design_Matrix(x.flatten(), 3)
    beta_test = np.random.randn(N,1)

    test1.gradient = gradientOLS; test2.gradient = gradientOLS
    beta_test1 = test1(x, y, N, theta=beta_test); beta_test2 = test2(x, y, N, batch_size, theta=beta_test)

    enablePrint()

    print(beta_test1)
    print(beta_test2)
    assert np.linalg.norm(beta_true - beta_test1) < eps, f"Test failed with error {np.linalg.norm(beta_true - beta_test1)}"
    assert np.linalg.norm(beta_true - beta_test2) < eps, f"Test failed with error {np.linalg.norm(beta_true - beta_test2)}"

if __name__ == "__main__":
    test_GD()
    # test_SGD()
    # test_AdaGrad()
    # test_RMSprop()
    # test_Adam()

    exit()
    import autograd.numpy as np
    N = 10_000
    x = np.random.rand(N,1)
    beta_true = np.array([[1, -8, 16],]).T
    func = Polynomial(*beta_true)
    y = func(x)

    def gradient_me(X, y, beta):
        return y - X @ beta 
    
    def CostOLS(X,y,theta):
        return np.sum((y-X @ theta)**2)

    
    g_gradient = grad(CostOLS, 2)

    adagraddiradd = Adagrad(learning_rate=0.3590565341426001)
    test = DescentSolver(adagraddiradd, 3, mode="SGD")
    beta_test = np.random.randn(np.shape(X)[1],1)

    test.gradient = g_gradient
    # X = np.c_[np.ones((N,2)), x]
    # y = func(x)
    # print(CostOLS(X,y, np.random.randn(3,1)))
    # print(g_gradient(X, y, np.random.randn(3,1)))
    # exit()
    beta_test = test(X, y, N, 20, 5, theta=beta_test)

    enablePrint()

    assert np.linalg.norm(beta_true - beta_test) < 1, f"Test failed with error {np.linalg.norm(beta_true - beta_test)}"