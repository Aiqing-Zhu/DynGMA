# -*- coding: utf-8 -*-
import abc
import torch
import torch.nn as nn

from .module import Module, DynNN
from .fnn import FNN
from ..integrator.sto_int import EM
from ..integrator.sto_int import MGaussAppro, MGaussDensity, GaussDensity
from ..integrator.sto_int import NMGaussAppro, NMGaussDensity, GaussCubAppro
from ..utils import grad


class SDENet(DynNN):
    '''SDENnet
    '''
    def __init__(self):
        super(SDENet, self).__init__()
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
        

    def vf(self, x):
        return self.modus['f'](x)
    
    def dvf(self, x):
        x_a = x.requires_grad_(True)
        return grad(self.modus['f'](x_a), x_a)        
        
    def sigma(self, x):
        return self.modus['sigma'](x)
    

    
class LinearSigma(Module):
    def __init__(self, dim):
        super(LinearSigma, self).__init__()
        self.dim = dim
        self.sigma = nn.Parameter((torch.randn([dim, dim]) * 0.5+0.5).requires_grad_(True))
        
        
    def forward(self, x):
        return self.sigma.expand([x.shape[0], self.dim, self.dim])


class NLinearSigma(Module):
    def __init__(self, dim):
        super(NLinearSigma, self).__init__()
        self.dim = dim
        self.tril = nn.Parameter((torch.randn([dim, dim]) * 0.5+0.5).requires_grad_(True))
        self.diag = nn.Parameter((torch.randn([dim]) * 0.5+0.5).requires_grad_(True))
        
        
    def forward(self, x):
        posi_diag = (torch.sqrt(self.diag**2+1) + self.diag)/2
        sigma = torch.diag_embed(posi_diag)+ torch.tril(self.tril, diagonal=-1)
        return sigma.expand([x.shape[0], self.dim, self.dim])    
    
class TrilSigma(Module):
    def __init__(self, dim, layers=2, width=128, activation='tanh'):
        super(TrilSigma, self).__init__()
        self.dim = dim
        self.layers = layers
        self.width = width       
        self.activation = activation
        
        self.modus = self.__init_modules()
    
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
        return torch.diag_embed(posi_diag)+ torch.tril(tril, diagonal=-1)



class MGaussSDENet(SDENet):
    '''GaussSDENnet using algorithm 1
    '''
    def __init__(self, dim=2, timestep=0.1,
                 layers=2, width=128, activation='tanh', initializer='orthogonal', linear_sigma=True,
                 sigmalayers=2, sigmawidth=50, 
                 st=1, sN=1):
        super(MGaussSDENet, self).__init__()
        self.dim = dim
        
        self.timestep=timestep
        
        self.layers = layers
        self.width = width       
        self.activation = activation
        self.initializer = initializer
        self.linear_sigma = linear_sigma

        self.sigmalayers = sigmalayers
        self.sigmawidth = sigmawidth          

        self.st = st
        self.sN = sN 
        
        
        self.regu_weight = 0.1
        self.Maxgrad = 20
        self.modus = self.__init_modules()
        

    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['f'] = FNN(self.dim, self.dim, self.layers, self.width, self.activation, self.initializer)
        modules['sigma'] = LinearSigma(self.dim) if self.linear_sigma else TrilSigma(self.dim, self.sigmalayers, self.sigmawidth, self.activation)
        return modules 
    
    def criterion(self, x0, x1):
        Mga = MGaussAppro(self.vf, self.dvf, self.sigma, st=self.st, sN = self.sN)
        m, P, W= Mga.solverTL(x0, [self.timestep]*x1.shape[0])
 
        logdensity = 0
        for i in range(x1.shape[0]):
            Den = MGaussDensity(m[i], P[i], W[i])
            logdensity = logdensity + Den.SafeLogDensity(x1[i]).mean()
        return -logdensity/x1.shape[0]

    def regularization(self, x0, x1):
        gradf = (self.dvf(x0)** 2).mean()        
        regu = gradf * self.regu_weight/x1.shape[0] - self.Maxgrad
        return (torch.sqrt(regu**2+1.523) + regu)/2
    
    
class NMGaussSDENet(SDENet):
    '''GaussSDENnet using algorithm 2
    '''
    def __init__(self, dim=2, timestep=0.1,
                 layers=2, width=128, activation='tanh', initializer='orthogonal', linear_sigma=True,
                 sigmalayers=2, sigmawidth=50, 
                 st=1):
        super(NMGaussSDENet, self).__init__()
        self.dim = dim
        
        self.timestep=timestep
        
        self.layers = layers
        self.width = width       
        self.activation = activation
        self.initializer = initializer
        self.linear_sigma = linear_sigma
        
        self.sigmalayers = sigmalayers
        self.sigmawidth = sigmawidth  
        
        self.st = st 
        self.modus = self.__init_modules()
        

    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['f'] = FNN(self.dim, self.dim, self.layers, self.width, self.activation, self.initializer)
        modules['sigma'] = NLinearSigma(self.dim) if self.linear_sigma else TrilSigma(self.dim, self.sigmalayers, self.sigmawidth, self.activation)
        

        return modules 
    
    def criterion(self, x0, x1):
        NMga = NMGaussAppro(self.vf, self.dvf, self.sigma, st=self.st)
        m, sqrtP, W= NMga.solverTL(x0, [self.timestep]*x1.shape[0])
 
        logdensity = 0
        for i in range(x1.shape[0]):
            Den = NMGaussDensity(m[i], sqrtP[i], W[i])
            logdensity = logdensity + Den.SafeLogDensity(x1[i]).mean()
        return -logdensity/x1.shape[0]
     
       
        

class EMSDENet(SDENet):
    '''GaussSDENnet
    '''
    def __init__(self, dim=2, timestep=0.1,
                 layers=2, width=128, activation='tanh', initializer='orthogonal', 
                 linear_sigma=True, sigmalayers=2, sigmawidth=50):
        super(EMSDENet, self).__init__()
        self.dim = dim
        
        self.timestep=timestep
        
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
        m = x0 + self.timestep * self.vf(x0)
        P =  self.timestep * self.sigma(x0)@self.sigma(x0).transpose(dim0=-2, dim1=-1)
        Den = GaussDensity(m, P)
        return -Den.LogDensity(x1[0]).mean()
    
    
    
    
class GaussCubSDENet(SDENet):
    '''SDENnet - Gauss Cubture
    '''
    def __init__(self, dim=2, timestep=0.1, N=4,
                 layers=2, width=128, activation='tanh', initializer='orthogonal', 
                 linear_sigma=True, sigmalayers=2, sigmawidth=50):
        super(GaussCubSDENet, self).__init__()
        self.dim = dim
        
        self.timestep=timestep
        self.N=N
        
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
        GCA = GaussCubAppro(self.vf, self.dvf, self.sigma, N=self.N, D=self.dim)
        m, P= GCA.solver(x0, self.timestep)
        Den = GaussDensity(m, P)
        return -Den.LogDensity(x1[0]).mean()   

    
    
    
    
    
    
    
    
    
    
    
    