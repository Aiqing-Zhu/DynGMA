# -*- coding: utf-8 -*-
import torch

from .fnn import FNN
from .odenet import ODENet
from ..utils import lazy_property, grad

    
class DFNN(ODENet):
    '''Neural divergence free ODEs.
    '''
    def __init__(self, dim=4, layers=2, width=128, activation='tanh', initializer='orthogonal', integrator='euler', iteration=1):
        super(DFNN, self).__init__()
        self.dim = dim
        self.layers = layers
        self.width = width       
        self.activation = activation
        self.initializer = initializer
        self.integrator = integrator
        self.iteration = iteration
        
        self.modus = self.__init_modules()
      
    @lazy_property 
    def Jg(self):
        d= int(self.dim)
        Jg=torch.zeros([d-1,d,d],dtype=self.Dtype, device=self.Device)
        for i in range(d-1):
            Jg[i, 0+i, 1+i] = 1
            Jg[i, 1+i, 0+i] = -1
        return Jg
    
    def vf(self, x):
        d=int(self.dim)
        x_0 = x.requires_grad_(True)
        gradH = grad(self.modus['H'](x_0), x_0)
        vf = gradH[...,0,:]@self.Jg[0]
        for i in range(1,d-1):
            vf = vf + gradH[...,i,:]@self.Jg[i]
        return vf
        
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['H'] = FNN(self.dim, self.dim-1, self.layers, self.width, self.activation, self.initializer)
        return modules 
    

