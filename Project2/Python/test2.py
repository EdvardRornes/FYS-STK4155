from utils import *

latex_fonts()

lambdas = np.linspace(1,30,5)
etas = np.logspace(-8,1,10)
H = np.random.rand(len(etas), len(lambdas))

plot_2D_parameter_lambda_eta(lambdas, etas, H, title='This is a title', filename='test', savefig=False, y_log=True)
plt.show()