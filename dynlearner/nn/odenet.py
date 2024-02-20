import torch

from .module import DynNN
from .fnn import FNN
from ..integrator.rungekutta import RK4, Integrator_list 
    
class ODENet(DynNN):
    '''Neural ODEs.
    '''
    def __init__(self, dim=4, layers=2, width=128, activation='tanh', initializer='orthogonal', 
                 integrator='euler', steps=1, iterations =1):
        super(ODENet, self).__init__()
        self.dim = dim
        self.layers = layers
        self.width = width       
        self.activation = activation
        self.initializer = initializer
        self.integrator = integrator
        self.steps = steps
        self.iterations = iterations
        
        self.modus = self.__init_modules()
    
    def criterion(self, x0h, x1):
        x0, h = (x0h[..., :-1], x0h[..., -1:])
        return self.integrator_loss(x0, x1, h)
    
    def predict(self, x0, h=0.1, steps=1, keepinitx=False, returnnp=False):
        solver = RK4(self.vf, N= int(h/0.001)) 
        res = solver.flow(x0, h, steps) if keepinitx else solver.flow(x0, h, steps)[..., 1:, :].squeeze()
        return res.cpu().detach().numpy() if returnnp else res
        

    def vf(self, x):
        return self.modus['f'](x)
        
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['f'] = FNN(self.dim, self.dim, self.layers, self.width, self.activation, self.initializer)
        return modules 
    
    def integrator_loss(self, x0, x1, h):
        n=int(self.steps)
        solver = Integrator_list[self.integrator](self.vf, n)
        x=solver.solve(x0, h)
        return torch.nn.MSELoss()(x1, x)
        