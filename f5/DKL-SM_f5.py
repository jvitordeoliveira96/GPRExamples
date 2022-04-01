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


def function_5(x):
    return -torch.sqrt( x[:, 0] ) * torch.sqrt( x[:, 1] ) * torch.sin( x[:, 0] ) * torch.sin( x[:, 1] )

torch.set_default_tensor_type(torch.DoubleTensor)


train_x = torch.load('train_85.pt')

# Function5 with Gaussian noise
train_y = function_5(train_x)  + torch.load('noise_85_0.20std.pt')

# Number of restarts
restarts = int(sys.argv[1])

# The Feature extractor definition
data_dim = train_x.size(-1)

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(2, 100))
        self.add_module('relu1', torch.nn.LeakyReLU())
        self.add_module('linear2', torch.nn.Linear(100, 50))
        self.add_module('relu2', torch.nn.LeakyReLU())
        self.add_module('linear3', torch.nn.Linear(50, 5))
        self.add_module('relu3', torch.nn.LeakyReLU())
        self.add_module('linear4', torch.nn.Linear(5, 2))

feature_extractor = LargeFeatureExtractor()


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ( gpytorch.kernels.SpectralMixtureKernel( num_mixtures = 4, ard_num_dims=2 ) )  
                            
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
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-1, 0.8))
model = ExactGPModel(train_x, train_y, likelihood)

# saving hyperparameters:
path = '../logs/f5/DKL-SM.txt'
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
    endpt = (0, 10)
    x1, x2 = torch.linspace(endpt[0], endpt[1], 90), torch.linspace(endpt[0], endpt[1], 90)
    x1, x2 = torch.meshgrid(x1, x2)
    test_x = torch.stack( (x1.flatten(), x2.flatten() ), -1 ) 
    test_y = function_5(test_x) 
    observed_pred = likelihood(model(test_x))
    # Initialize plot
    f = plt.figure(figsize=(9, 4))
    ax = f.add_subplot(111, projection="3d")

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    
    # Plot predictive mean
    x, y, z = test_x[:, 0].detach().numpy(), test_x[:, 1].detach().numpy(), observed_pred.mean.detach().numpy()
    ax.plot_trisurf(x, y, z, cmap=matplotlib.cm.jet, alpha = 0.8)

    # Plotting training points
    x, y, z = train_x[:, 0].detach().numpy(), train_x[:, 1].detach().numpy(), train_y.detach().numpy()
    ax.scatter3D(x, y, z, color = "black", s = 15);
        
    
    #ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.view_init(18.89946, -63.4406451612931)
    ax.set_xlim(0,10)
    ax.set_ylim(0,10)
    ax.set_zlim(-9,8)
    ax.set_xlabel('$x_1^*$')
    ax.set_ylabel('$x_2^*$')
    ax.set_zlabel('$h(x_1^*, x_2^*)$')
    plt.show()
    print('ax.azim {}'.format(ax.azim))
    print('ax.elev {}'.format(ax.elev))