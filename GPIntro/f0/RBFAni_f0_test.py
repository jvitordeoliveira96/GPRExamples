import numpy as np
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import matplotlib


# Latex font rendering
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
matplotlib.rc('text', usetex = True)


def function_1(x):
    return 0 * x[:, 0] + 0 * x[:, 1]

torch.set_default_tensor_type(torch.DoubleTensor)


train_x = torch.stack( (torch.linspace(-5, 5, 35) , torch.linspace(-5, 5, 35) ), -1 )
# Function1 with Gaussian noise
train_y = function_1(train_x)  + torch.load('noise_0.20std.pt')

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel( gpytorch.kernels.RBFKernel( ard_num_dims = train_x.shape[1] ) )

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
model.covar_module.base_kernel.lengthscale = torch.tensor([2.25, 0.25])
model.covar_module.outputscale = 1
model.likelihood.noise = 1e-8
#length_scl, sigma = model.covar_module.base_kernel.lengthscale.item(), model.covar_module.outputscale
sigma_like = model.likelihood.noise.item()

# Get into evaluation (predictive) mode
model.eval()
likelihood.eval()


# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.prior_mode(True):
    endpt = (-5, 5)
    x1, x2 = torch.linspace(endpt[0], endpt[1], 90), torch.linspace(endpt[0], endpt[1], 90)
    x1, x2 = torch.meshgrid(x1, x2)
    test_x = torch.stack( (x1.flatten(), x2.flatten() ), -1 ) 
    test_y = function_1(test_x) 
    observed_pred = likelihood(model(test_x))
    # Initialize plot
    f = plt.figure(figsize=(6, 4))
    ax = f.add_subplot(111, projection="3d")

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    
    # Plot function samples 
    nsamples = 1
    for i in range(0, nsamples):
        x, y, z = test_x[:, 0].detach().numpy(), test_x[:, 1].detach().numpy(), observed_pred.sample().detach().numpy()
        ax.plot_trisurf(x, y, z, cmap=matplotlib.cm.jet)
        
    ax.set_xlabel('$x_1^*$')
    ax.set_ylabel('$x_2^*$')
    ax.set_zlabel('$h(x_1^*, x_2^*)$')
    plt.show()
