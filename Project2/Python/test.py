from utils import *
import sys, os

# Disable print
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore print
def enablePrint():
    sys.stdout = sys.__stdout__

def test_GD(eps=3, eps_MSE=0.1):
    blockPrint()

    N = 1000
    x = np.random.rand(N,1)
    beta_true = np.random.rand(3)
    func = Polynomial(*beta_true)
    y = func(x)

    # Testing DescentSolver
    planeGD = PlaneGradient(momentum=0, learning_rate=0.3590565341426001)
    test = DescentSolver(planeGD, 3)

    beta_test = np.random.randn(3, 1)

    gradientOLS = AutoGradCostFunction(CostRidge, 2)
    test.gradient = gradientOLS
    
    X = create_Design_Matrix(x, 3)
    beta_test = test(X, y, N, 0)

    # Testing DescentAnalyzer
    test_analyzer = DescentAnalyzer(x, y, 3, N, print_percentage=False)

    planeGD = PlaneGradient(momentum=0, learning_rate=0.3590565341426001)
    gradientOLS = AutoGradCostFunction(CostRidge, 2)

    test.gradient = gradientOLS
    test_analyzer.run_analysis(planeGD, gradientOLS, 0.3590565341426001, 0)
    
    mse_train = test_analyzer["MSE_train"]
    mse_test = test_analyzer["MSE_test"]

    enablePrint()
    assert np.linalg.norm(beta_true - beta_test) < eps, f"Test failed with error {np.linalg.norm(beta_true - beta_test)}"
    assert np.linalg.norm(mse_train) < eps_MSE, f"Test GD for MSE-train failed with error {np.linalg.norm(mse_train)}"
    assert np.linalg.norm(mse_test) < eps_MSE, f"Test GD for MSE-test failed with error {np.linalg.norm(mse_test)}"

def test_SGD(eps=3, eps_MSE=0.1):
    blockPrint()

    # Testing DescentSolver
    N = 1000; N_epochs = 10; batch_size = 4
    x = np.random.rand(N,1)
    beta_true = np.random.rand(3)
    func = Polynomial(*beta_true)
    y = func(x)
    
    planeGD = PlaneGradient(momentum=0, learning_rate=0.3590565341426001)
    test = DescentSolver(planeGD, 3, mode="SGD")

    gradientOLS = AutoGradCostFunction(CostRidge, 2)
    
    test.gradient = gradientOLS
    
    X = create_Design_Matrix(x, 2)
    
    beta_test = test(X, y, N_epochs, 0, batch_size)

    # Testing DescentAnalyzer
    N = 20; N_epochs = 10; batch_size = 4

    x = np.sort(np.random.uniform(0, 1, N)) 
    y_ = np.sort(np.random.uniform(0, 1, N))
    y = Franke.franke(x,y_); y = y.reshape(-1, 1)
    x = [x,y_]
    
    t0 = 2; t1 = 2
    learningrate = LearningRate(t0, t1, N, batch_size, "test")
    
    
    test_analyzer = DescentAnalyzer(x, y, 5, N_epochs, batch_size=batch_size, GD_SGD="SGD", print_percentage=False)

    planeGD = PlaneGradient(momentum=0, learning_rate=learningrate)
    gradientOLS = AutoGradCostFunction(CostRidge, 2)

    test.gradient = gradientOLS
    test_analyzer.run_analysis(planeGD, gradientOLS, learningrate, 0)
    
    mse_train = test_analyzer["MSE_train"]
    mse_test = test_analyzer["MSE_test"]

    enablePrint()
    assert np.linalg.norm(beta_true - beta_test) < eps, f"Test failed with error {np.linalg.norm(beta_true - beta_test)}"
    assert np.linalg.norm(mse_train) < eps_MSE, f"Test SGD for MSE-train failed with error {np.linalg.norm(mse_train)}"
    assert np.linalg.norm(mse_test) < eps_MSE, f"Test SGD for MSE-test failed with error {np.linalg.norm(mse_test)}"

def test_AdaGrad(eps=5, eps_MSE=0.1):
    blockPrint()

    N = 1000; batch_size = 100
    x = np.random.rand(N,1)
    beta_true = np.random.rand(3)
    func = Polynomial(*beta_true)
    y = func(x)

    # Testing DescentSolver
    adagraddiradd = Adagrad(learning_rate=0.3590565341426001)
    test = DescentSolver(adagraddiradd, 3, mode="SGD")

    beta_test = np.random.randn(3, 1)

    gradientOLS = AutoGradCostFunction(CostRidge, 2)
    test.gradient = gradientOLS
    
    X = create_Design_Matrix(x, 2)
    beta_test = test(X, y, N, 0, batch_size)

    # Testing DescentAnalyzer
    test_analyzer = DescentAnalyzer(x, y, 3, N, batch_size=batch_size, GD_SGD="SGD", print_percentage=False)

    adagraddiradd = Adagrad(momentum=0, learning_rate=0.3590565341426001)
    gradientOLS = AutoGradCostFunction(CostRidge, 2)

    test.gradient = gradientOLS
    test_analyzer.run_analysis(adagraddiradd, gradientOLS, 0.3590565341426001, 0)
    
    mse_train = test_analyzer["MSE_train"]
    mse_test = test_analyzer["MSE_test"]

    enablePrint()
    assert np.linalg.norm(beta_true - beta_test) < eps, f"Test failed with error {np.linalg.norm(beta_true - beta_test)}"
    assert np.linalg.norm(mse_train) < eps_MSE, f"Test AdaGrad for MSE-train failed with error {np.linalg.norm(mse_train)}"
    assert np.linalg.norm(mse_test) < eps_MSE, f"Test AdaGrad for MSE-test failed with error {np.linalg.norm(mse_test)}"
    
def test_RMSprop(eps=3, eps_MSE=0.1):
    blockPrint()

    N = 1000; batch_size = 100
    x = np.random.rand(N,1)
    beta_true = np.random.rand(3)
    func = Polynomial(*beta_true)
    y = func(x)

    # Testing DescentSolver
    RMS_propproppapp = RMSprop(learning_rate=0.01)
    test = DescentSolver(RMS_propproppapp, 3, mode="SGD")

    beta_test = np.random.randn(3, 1)

    gradientOLS = AutoGradCostFunction(CostRidge, 2)
    test.gradient = gradientOLS
    
    X = create_Design_Matrix(x, 2)
    beta_test = test(X, y, N, 0, batch_size)

    # Testing DescentAnalyzer
    test_analyzer = DescentAnalyzer(x, y, 3, N, batch_size=batch_size, GD_SGD="SGD", print_percentage=False)

    RMS_propproppapp = RMSprop(momentum=0, learning_rate=0.01)
    gradientOLS = AutoGradCostFunction(CostRidge, 2)

    test.gradient = gradientOLS
    test_analyzer.run_analysis(RMS_propproppapp, gradientOLS, 0.01, 0)
    
    mse_train = test_analyzer["MSE_train"]
    mse_test = test_analyzer["MSE_test"]

    enablePrint()
    assert np.linalg.norm(beta_true - beta_test) < eps, f"Test failed with error {np.linalg.norm(beta_true - beta_test)}"
    assert np.linalg.norm(mse_train) < eps_MSE, f"Test RMSprop for MSE-train failed with error {np.linalg.norm(mse_train)}"
    assert np.linalg.norm(mse_test) < eps_MSE, f"Test RMSprop for MSE-test failed with error {np.linalg.norm(mse_test)}"

def test_Adam(eps=3, eps_MSE=0.1):
    blockPrint()

    N = 1000; batch_size = 100
    x = np.random.rand(N,1)
    beta_true = np.random.rand(3)
    func = Polynomial(*beta_true)
    y = func(x)

    # Testing DescentSolver
    adamramdampadamkada = Adam(learning_rate=0.01)
    test = DescentSolver(adamramdampadamkada, 3, mode="SGD")

    beta_test = np.random.randn(3, 1)

    gradientOLS = AutoGradCostFunction(CostRidge, 2)
    test.gradient = gradientOLS
    
    X = create_Design_Matrix(x, 2)
    beta_test = test(X, y, N, 0, batch_size)

    # Testing DescentAnalyzer
    test_analyzer = DescentAnalyzer(x, y, 3, N, batch_size=batch_size, GD_SGD="SGD", print_percentage=False)

    adamramdampadamkada = Adam(momentum=0, learning_rate=0.01)
    gradientOLS = AutoGradCostFunction(CostRidge, 2)

    test.gradient = gradientOLS
    test_analyzer.run_analysis(adamramdampadamkada, gradientOLS, 0.01, 0)
    
    mse_train = test_analyzer["MSE_train"]
    mse_test = test_analyzer["MSE_test"]

    enablePrint()
    assert np.linalg.norm(beta_true - beta_test) < eps, f"Test failed with error {np.linalg.norm(beta_true - beta_test)}"
    assert np.linalg.norm(mse_train) < eps_MSE, f"Test Adam for MSE-train failed with error {np.linalg.norm(mse_train)}"
    assert np.linalg.norm(mse_test) < eps_MSE, f"Test Adam for MSE-test failed with error {np.linalg.norm(mse_test)}"

if __name__ == "__main__":
    # test_GD()
    test_SGD()
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