import numpy as np
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import matplotlib
import os
import copy
import sys

# Latex font rendering
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
matplotlib.rc('text', usetex = True)

torch.set_default_tensor_type(torch.DoubleTensor)

filename = 'CO2_train.dat'
train_x, train_y = np.loadtxt(filename, delimiter='      ',unpack=True)

# Convert Numpy arrays to Torch tensors
train_x, train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)

# Number of restarts
restarts = int(sys.argv[1])

# The Feature extractor definition
data_dim = train_x.size(-1)

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(1, 100))
        self.add_module('relu1', torch.nn.LeakyReLU())
        self.add_module('linear2', torch.nn.Linear(100, 50))
        self.add_module('relu2', torch.nn.LeakyReLU())
        self.add_module('linear3', torch.nn.Linear(50, 5))
        self.add_module('relu3', torch.nn.LeakyReLU())
        self.add_module('linear4', torch.nn.Linear(5, 1))

feature_extractor = LargeFeatureExtractor()


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ( gpytorch.kernels.SpectralMixtureKernel( num_mixtures = 4 ) )  
                            
        self.feature_extractor = feature_extractor

    def forward(self, x):
        # First pass the input x on the NN
        projected_x =  self.feature_extractor(x)
        
        # Now, the GPR uses the extracted features as input
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def reset(self):
        # Kernel
        self.covar_module.initialize_from_data(train_x, train_y)
        # NN
        for layer in self.feature_extractor.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.LessThan(1.25))
model = ExactGPModel(train_x, train_y, likelihood)

# saving hyperparameters:
path = '../logs/f4/DKL-SM.txt'
if(not os.path.exists(path)):
    os.makedirs(path)
output_f = open(path, 'w')

output_f.write('Mean: \n')
output_f.write( str(model.covar_module.mixture_means) + '\n')
output_f.write('Variances (lengthscales): \n')
output_f.write( str(model.covar_module.mixture_scales) + '\n')
output_f.write('Weights: \n')
output_f.write( str(model.covar_module.mixture_weights) + '\n')
output_f.write('Noise (likelihood): \n')
output_f.write( str(model.likelihood.noise) )


training_iter = 8000
# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.00066)

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

output_f.write('\n\n------------------------------------------------------\n')
output_f.write('Mean: \n')
output_f.write( str(model.covar_module.mixture_means) + '\n')
output_f.write('Variances (lengthscales): \n')
output_f.write( str(model.covar_module.mixture_scales) + '\n')
output_f.write('Weights: \n')
output_f.write( str(model.covar_module.mixture_weights) + '\n')
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
    ax.set_title('DKL SM 4' )
    # Set xlim
    ax.set_xlim([endpt[0], endpt[1]])
    plt.show()