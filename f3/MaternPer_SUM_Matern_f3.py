import numpy as np
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import tikzplotlib as tikzplt
import copy
import sys
sys.path.append("..")  # Import Dot Product Kernel
from UserKernels.dot_product_kernel import DotProductKernel

def function_3(x):
    if(x < -2):
        return -2
    elif(x < -1):
        return 0 
    elif(x < 1):
        return 2
    elif(x < 2):
        return 1
    else:
        return -2 


torch.set_default_tensor_type(torch.DoubleTensor)

train_x = torch.linspace(-4.5, 4.5, 35)
# Function3 (no noise)
train_y = train_x.apply_(function_3) 
train_x = torch.linspace(-4.5, 4.5, 35)

# Number of restarts
restarts = int(sys.argv[1])



# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = (   
                            (   gpytorch.kernels.ScaleKernel( gpytorch.kernels.MaternKernel(nu = 2.5)  )  
                                *
                                ( gpytorch.kernels.MaternKernel(nu = 2.5)  +  
                                  gpytorch.kernels.PeriodicKernel()   
                                )
                            ) 
                            + 
                            (
                                gpytorch.kernels.ScaleKernel( gpytorch.kernels.RBFKernel() * DotProductKernel() )
                            )
                        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def reset(self):
        # Matern (0,0)
        self.covar_module.kernels[0].kernels[0].outputscale = torch.rand(1) * torch.max(train_y)
        self.covar_module.kernels[0].kernels[0].base_kernel.lengthscale = torch.rand(1)
        # Periodic (0,1)
        self.covar_module.kernels[0].kernels[1].kernels[1].lengthscale = torch.rand(1) 
        self.covar_module.kernels[0].kernels[1].kernels[1].period_length = torch.rand(1)
        # Matern (0,1)
        self.covar_module.kernels[0].kernels[1].kernels[0].lengthscale = torch.rand(1) 
        # RBF (0,0)
        self.covar_module.kernels[1].outputscale = torch.rand(1)
        self.covar_module.kernels[1].base_kernel.kernels[0].lengthscale = torch.rand(1) 
        # DotProduct
        self.covar_module.kernels[1].base_kernel.kernels[1].offset = torch.rand(1) 
        
        
        

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-5, 0.8))
model = ExactGPModel(train_x, train_y, likelihood)

# saving hyperparameters:
output_f = open('../logs/f3/MaternPer_SUM_Matern_f3.txt', 'w')

output_f.write('Kernel (0): \n')
output_f.write('\tScale (signal variance): \n')
output_f.write( '\t' + str(model.covar_module.kernels[0].kernels[0].outputscale) + '\n')
output_f.write('\tMatern (0,0) Lengthscale: \n')
output_f.write( '\t' + str(model.covar_module.kernels[0].kernels[0].base_kernel.lengthscale) + '\n')
output_f.write('\tPeriodic (0,1) Lengthscale: \n')
output_f.write( '\t' +str(model.covar_module.kernels[0].kernels[1].kernels[1].lengthscale) + '\n')
output_f.write('\tPeriodic (0,1) Period: \n')
output_f.write( '\t' + str(model.covar_module.kernels[0].kernels[1].kernels[1].period_length) + '\n')
output_f.write('\tMatern (0,1) Lengthscale: \n')
output_f.write( '\t' +str(model.covar_module.kernels[0].kernels[1].kernels[0].lengthscale) + '\n')
output_f.write('Kernel (1): \n')
output_f.write('\tScale (signal variance): \n')
output_f.write( '\t' + str(model.covar_module.kernels[1].outputscale) + '\n')
output_f.write('\tRBF (1) Lengthscale: \n')
output_f.write( '\t' + str(model.covar_module.kernels[1].base_kernel.kernels[0].lengthscale) + '\n')
output_f.write('\tDotProduct (1) Lengthscale: \n')
output_f.write( '\t' + str(model.covar_module.kernels[1].base_kernel.kernels[1].offset) + '\n')

output_f.write('Noise (likelihood): \n')
output_f.write( str(model.likelihood.noise) )



training_iter = 1000
# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.08)

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
output_f.write('Kernel (0): \n')
output_f.write('\tScale (signal variance): \n')
output_f.write( '\t' + str(model.covar_module.kernels[0].kernels[0].outputscale) + '\n')
output_f.write('\tMatern (0,0) Lengthscale: \n')
output_f.write( '\t' + str(model.covar_module.kernels[0].kernels[0].base_kernel.lengthscale) + '\n')
output_f.write('\tPeriodic (0,1) Lengthscale: \n')
output_f.write( '\t' +str(model.covar_module.kernels[0].kernels[1].kernels[1].lengthscale) + '\n')
output_f.write('\tPeriodic (0,1) Period: \n')
output_f.write( '\t' + str(model.covar_module.kernels[0].kernels[1].kernels[1].period_length) + '\n')
output_f.write('\tMatern (0,1) Lengthscale: \n')
output_f.write( '\t' +str(model.covar_module.kernels[0].kernels[1].kernels[0].lengthscale) + '\n')
output_f.write('Kernel (1): \n')
output_f.write('\tScale (signal variance): \n')
output_f.write( '\t' + str(model.covar_module.kernels[1].outputscale) + '\n')
output_f.write('\tRBF (1) Lengthscale: \n')
output_f.write( '\t' + str(model.covar_module.kernels[1].base_kernel.kernels[0].lengthscale) + '\n')
output_f.write('\tDotProduct (1) Lengthscale: \n')
output_f.write( '\t' + str(model.covar_module.kernels[1].base_kernel.kernels[1].offset) + '\n')

output_f.write('Noise (likelihood): \n')
output_f.write( str(model.likelihood.noise) )
output_f.close()



# Make predictions by feeding model through likelihood
with torch.no_grad():
    endpt = (-7, 7)
    test_x = torch.linspace(endpt[0], endpt[1] , 450)
    test_y = test_x.apply_(function_3)
    test_x = torch.linspace(endpt[0], endpt[1] , 450)
    observed_pred = likelihood(model(test_x))
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # Plot True function
    ax.plot(test_x.numpy(), test_y.numpy(), 'b')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'r')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, color = 'pink')
    ax.legend(['Observed Data', 'True function', 'Mean', 'Confidence'])
    # Title containing hyperparameters
    ax.set_title('Matern(Periodic + Matern) + RBF * DOT' )
    # Set xlim
    ax.set_xlim([endpt[0], endpt[1]])
    tikzplt.save("../results/tikz/f3/MaternPer_SUM_Matern_f3.tex")
    plt.show()