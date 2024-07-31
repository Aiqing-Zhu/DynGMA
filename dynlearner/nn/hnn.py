import torch
 
from .fnn import FNN
from .odenet import ODENet
from ..utils import lazy_property, grad

class HNN(ODENet):
    '''Hamiltonian neural networks.
    '''
    def __init__(self, dim=2, layers=3, width=30, activation='tanh', initializer='orthogonal', integrator='midpoint'):
        super(HNN, self).__init__()
        self.dim = dim
        self.layers = layers
        self.width = width
        self.activation = activation
        self.initializer = initializer
        self.integrator = integrator
        
        self.modus = self.__init_modules()
     
    @lazy_property
    def J(self):
        d = int(self.dim / 2)
        res = np.eye(self.dim, k=d) - np.eye(self.dim, k=-d)
        return torch.tensor(res, dtype=self.Dtype, device=self.Device)

    def vf(self, x):
        with torch.enable_grad():
            x_0 = x.requires_grad_(True)
            gradH = grad(self.modus['H'](x_0), x_0)
            vf = gradH @ self.J
        return vf
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['H'] = FNN(self.dim, 1, self.layers, self.width, self.activation, self.initializer)
        return modules 
 
