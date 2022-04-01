import sys
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
    return torch.sin(4 * x)


torch.set_default_tensor_type(torch.DoubleTensor)


train_x = torch.linspace(-2, 2, 35)
# Function1 with Gaussian noise
train_y = function_1(train_x) + torch.load('noise_0.20std.pt')
# Number of gaussians
n_gaussians_Q = int(sys.argv[1])

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures = n_gaussians_Q)
        #self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def reset(self):
        self.covar_module.initialize_from_data(train_x, train_y)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-16))
model = ExactGPModel(train_x, train_y, likelihood)


"""
# saving hyperparameters:
output_f = open('SM_sin4x_'+str(n_gaussians_Q)+'gauss_.txt', 'w')

output_f.write('hyperparameters (initial guess)\n')
output_f.write('Mean: \n')
output_f.write( str(model.covar_module.mixture_means) + '\n')
output_f.write('Variances (lengthscales): \n')
output_f.write( str(model.covar_module.mixture_scales) + '\n')
output_f.write('Weights: \n')
output_f.write( str(model.covar_module.mixture_weights) )


training_iter = 450
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
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

output_f.write('\n\n------------------------------------------------------\n')
output_f.write('hyperparameters (after tuning)\n')
output_f.write('Mean: \n')
output_f.write( str(model.covar_module.mixture_means) + '\n')
output_f.write('Variances (lengthscales): \n')
output_f.write( str(model.covar_module.mixture_scales) + '\n')
output_f.write('Weights: \n')
output_f.write( str(model.covar_module.mixture_weights) )
output_f.close()
"""

# Setting hyperparameters
#print(model.covar_module.mixture_means)
#print(model.covar_module.mixture_scales)
#print(model.covar_module.mixture_weights)
model.likelihood.noise = 1e-5
sigma_like = model.likelihood.noise.item()

# Get into evaluation (predictive) mode
model.eval()
likelihood.eval()


# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.prior_mode(True):
    endpt = (-6, 6)
    test_x = torch.linspace(endpt[0], endpt[1] , 200)
    test_y = function_1(test_x) 
    observed_pred = likelihood(model(test_x))
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot True function
    #ax.plot(test_x.numpy(), test_y.numpy(), 'b')
    # Plot predictive means as blue line
    ax.plot(test_x.detach().numpy(), observed_pred.mean.detach().numpy(), 'r')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.detach().numpy(), lower.numpy(), upper.numpy(), alpha=0.5, color = 'pink')
    ax.legend(['Mean', 'Confidence'])
    # Plot function samples 
    nsamples = 3
    for i in range(0, nsamples):
        ax.plot(test_x.detach().numpy(), observed_pred.sample().detach().numpy(), linestyle = 'dashed')

    # Title containing number of Gaussians
    ax.set_title('$Q = {0}$'.format(n_gaussians_Q) )
    # Set xlim
    ax.set_xlim([endpt[0], endpt[1]])
    # Set ylim
    ax.set_ylim([-4, 4])
    plt.show()