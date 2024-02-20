import torch

from .module import Module, DynNN
from .fnn import FNN
from ..utils import grad

class QuadraticModule(Module):
    '''quadratic module.
    '''
    def __init__(self, dim):
        super(QuadraticModule, self).__init__()
        self.dim = dim
        self.ps = self.__init_params()
        
    def forward(self, x):  
        return ( (x-self.ps['c'])**2 * torch.log(1+ torch.exp(self.ps['p'])) ).sum(dim=-1, keepdim=True)

    def __init_params(self):
        params = torch.nn.ParameterDict()
        params['c'] = torch.nn.Parameter((torch.randn([self.dim]) * 0.01).requires_grad_(True))
        params['p'] = torch.nn.Parameter((torch.randn([self.dim]) * 0.01).requires_grad_(True))
        return params
        
class ID_NN(DynNN):

    def __init__(self, f=None, D_bar=None, eps=0.1, dim=2,
                layers=2, width=128, activation='tanh'):
        super(ID_NN, self).__init__()
        self.f=f
        self.D_bar= D_bar
        self.eps = eps
        self.dim = dim
        
        self.layers=layers
        self.width=width
        self.activation=activation
        
        self.modus = self.__init_modules()
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['net'] = FNN(self.dim, 1, layers=self.layers, width=self.width, activation=self.activation)
        modules['qua'] = QuadraticModule(self.dim)
        modules['q'] = FNN(self.dim, self.dim, layers=self.layers, width=self.width, activation=self.activation)
        
        return modules 
        
    def criterion(self, X, y):
        '''
        solve PDE: \nabla V \cdot f + \nabla V \cdot D_bar \nabla V 
                  = \eps \nabla \cdot f + eps \nabla \cdot D_bar\nabla V
        '''
        z = X.requires_grad_(True)
        V = self.modus['net'](z) + self.modus['qua'](z)
        q = self.modus['q'](z) 
        F = self.f(z)
        
        V_g = grad(V, z)
        q_g = grad(q, z) 
        
        Dq = torch.einsum('ijj->ij', q_g).sum(dim=-1)
        return torch.nn.MSELoss()((q*V_g).sum(-1), self.eps * Dq) + torch.nn.MSELoss()(V_g@self.D_bar + F, q)

    
    def predict(self, z):
        return self.modus['net'](z) + self.modus['qua'](z)
    