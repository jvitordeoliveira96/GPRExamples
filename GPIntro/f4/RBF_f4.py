#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import tikzplotlib as tikzplt
import copy
import sys


torch.set_default_tensor_type(torch.DoubleTensor)

filename = 'CO2_train.dat'
train_x, train_y = np.loadtxt(filename, delimiter='      ',unpack=True)

# Convert Numpy arrays to Torch tensors
train_x, train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)

# Number of restarts
restarts = int(sys.argv[1])

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def reset(self):
        self.covar_module.base_kernel.lengthscale = (
            torch.rand(self.covar_module.base_kernel.lengthscale.size()
            )
        )
        self.covar_module.outputscale = (
            torch.rand(self.covar_module.outputscale.size()   
            )
        )
        
class ExactLikelihood(gpytorch.likelihoods.GaussianLikelihood):
    def reset(self):
        self.noise = (
            torch.rand(self.noise.size()   
            )
        )

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.LessThan(1.25))
model = ExactGPModel(train_x, train_y, likelihood)

# saving hyperparameters:
output_f = open('../logs/f4/RBF.txt', 'w')

output_f.write('Scale (signal variance): \n')
output_f.write( str(model.covar_module.outputscale) + '\n')
output_f.write('Lengthscale: \n')
output_f.write( str(model.covar_module.base_kernel.lengthscale.item()) + '\n')
output_f.write('Noise (likelihood): \n')
output_f.write( str(model.likelihood.noise) )


training_iter = 450
# Find optimal model hyperparameters
model.train()
likelihood.train()



# Use ADAM optimizer 
optimizer = torch.optim.Adam([
    #---------------------!!!!!!!!!!--------------- #
    # Comment / Uncomment     
    #{'params': model.covar_module.parameters()},   # Does not include GaussianLikelihood parameter (noise)
    {'params': model.parameters()},                 # Includes GaussianLikelihood parameter (noise)
], lr= 0.1)



# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

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


# Saving hyperparameters
length_scl, sigma = model.covar_module.base_kernel.lengthscale.item(), model.covar_module.outputscale

output_f.write('\n\n------------------------------------------------------\n')
output_f.write('Scale (signal variance): \n')
output_f.write( str(model.covar_module.outputscale) + '\n')
output_f.write('Lengthscale: \n')
output_f.write( str(model.covar_module.base_kernel.lengthscale.item()) + '\n')
output_f.write('Noise (likelihood): \n')
output_f.write( str(model.likelihood.noise) )
output_f.close()


# Make predictions by feeding model through likelihood
with torch.no_grad():
    endpt = (1957.58, 1987.16)
    test_x = torch.linspace(endpt[0], endpt[1], 420)
    observed_pred = likelihood(model(test_x))

    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'r')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, color = 'pink')
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    # Title containing hyperparameters
    ax.set_title('RBF: $\sigma = {0:.3f}$, $\lambda = {1:.3f}$'.format(sigma, length_scl) )
    # Set xlim
    ax.set_xlim([endpt[0], endpt[1]])
    tikzplt.save("../results/tikz/f4/RBF.tex")
    plt.show()
