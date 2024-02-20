# -*- coding: utf-8 -*-
import abc
import torch
import torch.nn as nn

from .module import Module, DynNN
from .fnn import FNN
from ..integrator.sto_int import EM
from ..integrator.sto_int import NMGaussAppro_V, NMGaussDensity, GaussDensity
from ..utils import grad


class SDENet_V(DynNN):
    '''GaussSDENnet
    '''
    def __init__(self):
        super(SDENet_V, self).__init__()
        self.dim = None
        self.modus = None
    
    
    @abc.abstractmethod
    def criterion(self, x0, time, x1):
        pass
 

    def predict(self, x0, h=0.1, steps=1, returnnp=False):
        solver = EM(self.vf, self.sigma, N=10)
        X = [x0]
        for i in range(steps):
            X.append(solver.solve(X[-1], torch.tensor(h)))
        Y = torch.cat(X,dim=0).view([steps+1, -1, self.dim])
        return Y.cpu().detach().numpy() if returnnp else Y

    def predict0(self, x0, h=0.1, steps=1, returnnp=False):
        def s0(x):
            return self.sigma(x)*0
        solver = EM(self.vf, s0, N=10)
        X = [x0]
        for i in range(steps):
            X.append(solver.solve(X[-1], torch.tensor(h)))
        Y = torch.cat(X,dim=0).view([steps+1, -1, self.dim])
        return Y.cpu().detach().numpy() if returnnp else Y
    

    def vf(self, x):
        return self.modus['f'](x)
    
    def dvf(self, x):
        x_a = x.requires_grad_(True) 
        return grad(self.modus['f'](x_a), x_a)        
        
    def sigma(self, x):
        return self.modus['sigma'](x)
    
    def dsigma(self, x):
        x_a = x.requires_grad_(True)
        return grad((self.modus['sigma'](x_a).reshape(-1, self.dim*self.dim)), x_a)          
    
    def Sigma(self, x):
        return self.modus['sigma'](x)@self.modus['sigma'](x).transpose(dim0=-2, dim1=-1)

    def Sigma_cpu(self, x):
        x = torch.tensor(x, device=self.Device)
        return (self.modus['sigma'](x)@self.modus['sigma'](x).transpose(dim0=-2, dim1=-1)).cpu().detach()

    
class LinearSigma(Module):
    def __init__(self, dim):
        super(LinearSigma, self).__init__()
        self.dim = dim
        self.sigma = nn.Parameter((torch.randn([dim, dim]) * 0.5+0.5).requires_grad_(True))
        
    def forward(self, x):
        return self.sigma+0

    
class TrilSigma(Module):
    def __init__(self, dim, layers=2, width=128, activation='tanh'):
        super(TrilSigma, self).__init__()
        self.dim = dim
        self.layers = layers
        self.width = width       
        self.activation = activation
        
        self.modus = self.__init_modules()
        # self.__initialize()
        
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['diagnet'] = FNN(self.dim, self.dim, layers=self.layers, width=self.width, activation=self.activation)
        modules['trilnet']  = FNN(self.dim, self.dim*self.dim, layers=self.layers, 
                                 width=self.width, activation=self.activation)    
        return modules         

        
    def forward(self, x):
        shape=list(x.shape)
        shape.append(self.dim)

        diag = self.modus['diagnet'](x)
        posi_diag = (torch.sqrt(diag**2+1) + diag)/2 
        
        tril = self.modus['trilnet'](x).reshape(shape)
        return torch.diag_embed(posi_diag) + torch.tril(tril, diagonal=-1)

    def __initialize(self):
        nn.init.orthogonal_(self.modus['diagnet'].modus['LinMout'].weight, gain=0.01)
        nn.init.orthogonal_(self.modus['trilnet'].modus['LinMout'].weight, gain=0.01)

        nn.init.constant_(self.modus['diagnet'].modus['LinMout'].bias, 0)
        nn.init.constant_(self.modus['trilnet'].modus['LinMout'].bias, 0)
    
    
    
class NMGaussSDENet_V(SDENet_V):
    '''GaussSDENnet
    '''
    def __init__(self, dim=2, 
                 layers=2, width=128, activation='tanh', initializer='orthogonal', 
                 sigmalayers=2, sigmawidth=50, 
                 steps=1, regu_weight=0.1, Maxgrad=20):
        super(NMGaussSDENet_V, self).__init__()
        self.dim = dim
        
        
        self.layers = layers
        self.width = width       
        self.activation = activation
        self.initializer = initializer
        
        self.sigmalayers = sigmalayers
        self.sigmawidth = sigmawidth  
        
        self.steps = steps
        self.regu_weight = regu_weight
        self.Maxgrad = Maxgrad
        self.modus = self.__init_modules()
        

    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['f'] = FNN(self.dim, self.dim, self.layers, self.width, self.activation, self.initializer)
        modules['sigma'] = TrilSigma(self.dim, self.sigmalayers, self.sigmawidth, self.activation)
        return modules 
    
    def criterion(self, x0, x1):
        NMga = NMGaussAppro_V(self.vf, self.dvf, self.sigma, steps=self.steps)

        m, sqrtP, W= NMga.solverTL(x0, x1[:, :,:1])
        logdensity = 0
        for i in range(x1.shape[0]):
            Den = NMGaussDensity(m[i], sqrtP[i], W[i])
            logdensity = logdensity + Den.SafeLogDensity(x1[i, :, 1:]).mean()
        return -logdensity/x1.shape[0]
    
    
class EMSDENet_V(SDENet_V):
    '''GaussSDENnet
    '''
    def __init__(self, dim=2,
                 layers=2, width=128, activation='tanh', initializer='orthogonal', 
                 linear_sigma=True, sigmalayers=2, sigmawidth=50):
        super(EMSDENet_V, self).__init__()
        self.dim = dim
        
        
        self.layers = layers
        self.width = width       
        self.activation = activation
        self.initializer = initializer
        
        self.linear_sigma = linear_sigma
        self.sigmalayers = sigmalayers
        self.sigmawidth = sigmawidth

        self.modus = self.__init_modules()
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['f'] = FNN(self.dim, self.dim, self.layers, self.width, self.activation, self.initializer)
        modules['sigma'] = LinearSigma(self.dim) if self.linear_sigma else TrilSigma(self.dim, self.sigmalayers, self.sigmawidth, self.activation)
        return modules 

    def criterion(self, x0, x1):
        m = x0 + x1[0, :,:1] * self.vf(x0)
        P =  x1[0, :,:1].unsqueeze(-1) * self.sigma(x0)@self.sigma(x0).transpose(dim0=-2, dim1=-1)
        Den = GaussDensity(m, P)
        return -Den.LogDensity(x1[0, :, 1:]).mean()

 