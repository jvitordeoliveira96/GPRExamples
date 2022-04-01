import numpy as np
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import matplotlib
import copy
import os
import sys

# Latex font rendering
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
matplotlib.rc('text', usetex = True)


def function_6(x):
    return torch.stack( (3 * torch.cos(x), 2 * torch.cos(x+0.3) ), -1 )

torch.set_default_tensor_type(torch.DoubleTensor)


train_x = torch.linspace(-3 * np.pi,  3 * np.pi, 35)  
# Function6 with Gaussian noise
train_y = function_6(train_x) 
train_y[:, 0] += torch.load('noise_35_0.10std.pt')  
train_y[:, 1] += torch.load('noise_35_0.10std_2.pt') 


# Number of gaussians
n_gaussians_Q = int(sys.argv[1])
# Number of restarts
restarts = int(sys.argv[2])

# We will use the simplest form of GP model, exact inference
class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([2]))
        self.covar_module =  gpytorch.kernels.ScaleKernel( 
            gpytorch.kernels.SpectralMixtureKernel(num_mixtures = n_gaussians_Q), batch_shape=torch.Size([2])
        )
        self.covar_module.base_kernel.initialize_from_data(train_x, train_y)
        self.covar_module.outputscale = torch.ones(  self.covar_module.outputscale.size()  )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )
    
    def reset(self):
        self.covar_module.base_kernel.initialize_from_data(train_x, train_y)
        self.covar_module.outputscale = torch.ones(  self.covar_module.outputscale.size()  )


# initialize likelihood and model
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2, noise_constraint=gpytorch.constraints.Interval(1e-2, 0.8))
model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood)


# saving hyperparameters:
path = '../logs/f6/SM_{0}gauss_IndOutputs.txt'.format(n_gaussians_Q)
if(not os.path.exists(path)):
    os.makedirs(path)
output_f = open(path, 'w')

output_f.write('Mean: \n')
output_f.write( str(model.covar_module.base_kernel.mixture_means) + '\n')
output_f.write('Variances (lengthscales): \n')
output_f.write( str(model.covar_module.base_kernel.mixture_scales) + '\n')
output_f.write('Weights: \n')
output_f.write( str(model.covar_module.base_kernel.mixture_weights) + '\n')
output_f.write('Scale (outputs): \n')
output_f.write( str(model.covar_module.outputscale) )
output_f.write('Noise (likelihood): \n')
output_f.write( str(model.likelihood.noise) )


training_iter = 1200
# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.covar_module.base_kernel.parameters()},
    {'params': model.likelihood.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.06)

best_loss = float('inf')

for r in range(0, restarts):
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

    if(float(loss.item()) < best_loss):
        # Gp hypers
        best_model = copy.deepcopy(model)
        # Likelihood (data noise)
        best_likelihood = copy.deepcopy(likelihood)
        # Negative MLL value
        best_loss = float(loss.item())    
    
        
    model.reset()




# Get best model/likelihood
model, likelihood = best_model, best_likelihood

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
output = model(train_x)
loss = -mll(output, train_y)
print('\nBest Loss: {0:.3f} x Best Loss: {1:.3f}'.format( loss.item(), best_loss ) )


# saving hyperparameters
output_f.write('\n\n------------------------------------------------------\n')
output_f.write('hyperparameters (after tuning)\n')
output_f.write('Mean: \n')
output_f.write( str(model.covar_module.base_kernel.mixture_means) + '\n')
output_f.write('Variances (lengthscales): \n')
output_f.write( str(model.covar_module.base_kernel.mixture_scales) + '\n')
output_f.write('Weights: \n')
output_f.write( str(model.covar_module.base_kernel.mixture_weights) + '\n')
output_f.write('Scale (outputs): \n')
output_f.write( str(model.covar_module.outputscale) )
output_f.write('Noise (likelihood): \n')
output_f.write( str(model.likelihood.noise) )
output_f.close()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad():
    endpt = (-4 * np.pi, 4 * np.pi)
    test_x = torch.linspace(endpt[0], endpt[1] , 400)
    test_y = function_6(test_x) 
    observed_pred = likelihood(model(test_x))

# Initialize plot
plt.figure(figsize = (4, 3))

# Get upper and lower confidence bounds
lower, upper = observed_pred.confidence_region()
# Plot training data as black stars
plt.plot(train_y[:,0].detach().numpy(), train_y[:,1].detach().numpy(), 'k*')
# Plot True function
plt.plot(test_y[:,0].detach().numpy(), test_y[:,1].detach().numpy(), 'b')
# Plot predictive means as blue line
plt.plot(observed_pred.mean[:,0].detach().numpy(), observed_pred.mean[:,1].detach().numpy(), 'r')

# Adding legend
plt.legend(['Observed Data', 'True function: $C(\\theta) = (\cos(\\theta), \sin(\\theta))$', 'Mean' ])

# Title containing hyperparameters
plt.title('SM_{0}gauss_IndOutputs'.format(n_gaussians_Q))
# Set xlim
plt.xlim([-3.5, 3.5])
# Set ylim
plt.ylim([-3, 3])
plt.show(block = False)

#-----------------#
# Initialize plot
plt.figure(figsize = (4,3))

# Get upper and lower confidence bounds
lower, upper = observed_pred.confidence_region()
# Plot training data as black stars
plt.plot(train_x.detach().numpy(), train_y[:,0].detach().numpy(), 'k*')
# Plot True function
plt.plot(test_x.detach().numpy(), test_y[:,0].detach().numpy(), 'b')
# Plot predictive means as blue line
plt.plot(test_x.detach().numpy(), observed_pred.mean[:,0].detach().numpy(), 'r')
# Shade between the lower and upper confidence bounds
plt.fill_between(test_x.detach().numpy(), lower[:,0].detach().numpy(), upper[:,0].detach().numpy(), alpha=0.5, color = 'pink')
plt.legend(['Observed Data', 'True function: $f_1(\\theta) = \cos(\\theta)$', 'Mean', 'Confidence'])
# Title containing number of Gaussians
plt.title( 'SM_{0}gauss_IndOutputs'.format(n_gaussians_Q) )
# Set xlim
plt.xlim([endpt[0], endpt[1]])
# Set ylim
plt.ylim([-4, 4])
plt.show(block = False)

#-------------#
# Initialize plot
plt.figure(figsize = (4,3))

# Get upper and lower confidence bounds
lower, upper = observed_pred.confidence_region()
# Plot training data as black stars
plt.plot(train_x.detach().numpy(), train_y[:,1].detach().numpy(), 'k*')
# Plot True function
plt.plot(test_x.detach().numpy(), test_y[:,1].detach().numpy(), 'b')
# Plot predictive means as blue line
plt.plot(test_x.detach().numpy(), observed_pred.mean[:,1].detach().numpy(), 'r')
# Shade between the lower and upper confidence bounds
plt.fill_between(test_x.detach().numpy(), lower[:,1].detach().numpy(), upper[:,1].detach().numpy(), alpha=0.5, color = 'pink')
plt.legend(['Observed Data', 'True function: $f_2(\\theta) = \sin(\\theta)$', 'Mean', 'Confidence'])
# Title containing number of Gaussians
plt.title( 'SM_{0}gauss_IndOutputs'.format(n_gaussians_Q) )
# Set xlim
plt.xlim([endpt[0], endpt[1]])
# Set ylim
plt.ylim([-3, 3])
plt.show()