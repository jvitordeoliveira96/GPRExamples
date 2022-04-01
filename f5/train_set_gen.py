import numpy as np
import scipy as sp
import math
import torch
import gpytorch
from matplotlib import pyplot as plt

# The training set will be composed by x1, x2 from a uniform distribution in [0, 8]
torch.set_default_tensor_type(torch.DoubleTensor)

npoints = 85
x1 = torch.rand(npoints) * (8+1e-2)
x2 = torch.rand(npoints) * (8+1e-2)
train_x = torch.stack( (x1, x2), -1 )

torch.save(train_x, 'train_{0}.pt'.format( npoints ))