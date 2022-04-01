import numpy as np
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import matplotlib


# Latex font rendering
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
matplotlib.rc('text', usetex = True)

# Unused (Prior samples)
# ----------------------
def function_1(x):
    return 0 * x

train_x = torch.linspace(0, 5, 35)
# Function1 with Gaussian noise
train_y = function_1(train_x) + torch.load('noise_0.20std.pt')

# ----------------------

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel( 1, offset_constraint= gpytorch.constraints.GreaterThan(-1e-1) ) )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-16))
model = ExactGPModel(train_x, train_y, likelihood)


""" training_iter = 120
# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f  sigma: %.3f  noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.covar_module.outputscale,
        model.likelihood.noise.item()
    ))
    optimizer.step() """

# Setting hyperparameters
model.covar_module.base_kernel.offset = 1.0   # obs: If (x.x + c) < 0, error (not Positive definite)
model.covar_module.outputscale = 1.0
model.likelihood.noise = 1e-16
offset, sigma = model.covar_module.base_kernel.offset.item(), model.covar_module.outputscale
sigma_like = model.likelihood.noise.item()

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.prior_mode(True):
    endpt = (-5, 5)
    test_x = torch.linspace(endpt[0], endpt[1] , 40)
    test_y = function_1(test_x) 
    observed_pred = likelihood(model(test_x))
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot True function
    ax.plot(test_x.detach().numpy(), test_y.numpy(), 'b')
    # Plot predictive means as blue line
    ax.plot(test_x.detach().numpy(), observed_pred.mean.detach().numpy(), 'r')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5, color = 'pink')
    ax.legend(['True function', 'Mean', 'Confidence'])
    # Plot function samples 
    nsamples = 5
    for i in range(0, nsamples):
        ax.plot(test_x.detach().numpy(), observed_pred.sample().detach().numpy(), linestyle = 'dashed')

    # Title containing hyperparameters
    ax.set_title('$\sigma_0 = {0:.3f}$, $\sigma_f = {1:.3f}$, $\sigma_l = {2:.3f}$'.format(offset, sigma, sigma_like) )
    # Set xlim
    ax.set_xlim([endpt[0], endpt[1]])
    # Set ylim
    ax.set_ylim([-4, 4])
    plt.show()