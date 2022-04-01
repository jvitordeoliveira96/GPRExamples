import numpy as np
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import matplotlib


# Latex font rendering
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
matplotlib.rc('text', usetex = True)


def function_5(x):
    return -torch.sqrt( x[:, 0] ) * torch.sqrt( x[:, 1] ) * torch.sin( x[:, 0] ) * torch.sin( x[:, 1] )

torch.set_default_tensor_type(torch.DoubleTensor)


train_x = torch.load('train_85.pt')

# Function5 with Gaussian noise
train_y = function_5(train_x) + torch.load('noise_85_0.20std.pt')

# Initialize plot
endpt = (0, 10)
x1, x2 = torch.linspace(endpt[0], endpt[1], 90), torch.linspace(endpt[0], endpt[1], 90)
x1, x2 = torch.meshgrid(x1, x2)
test_x = torch.stack( (x1.flatten(), x2.flatten() ), -1 ) 
test_y = function_5(test_x) 
f = plt.figure(figsize=(9, 4))
ax = f.add_subplot(111, projection="3d")

# Plot z = 0
ax.plot_surface(x1.detach().numpy(), x2.detach().numpy(), x2.detach().numpy() * 0 - 9.5 , cmap=matplotlib.cm.Wistia, alpha = 0.36)

# Plotting training points
x, y, z = train_x[:, 0].detach().numpy(), train_x[:, 1].detach().numpy(), train_y.detach().numpy() * 0. - 9.5
ax.scatter3D(x, y, z, color = "blue", s = 15);

# Plot true function
x, y, z = test_x[:, 0].detach().numpy(), test_x[:, 1].detach().numpy(), test_y.detach().numpy()
ax.plot_trisurf(x, y, z, cmap=matplotlib.cm.jet, alpha = 0.75)



ax.view_init(18.89946, -63.4406451612931)
ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.set_zlim(-9,8)
ax.set_xlabel('$x_1^*$')
ax.set_ylabel('$x_2^*$')
ax.set_zlabel('$f(x_1^*, x_2^*)$')
plt.show()
print('ax.azim {}'.format(ax.azim))
print('ax.elev {}'.format(ax.elev))