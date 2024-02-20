# -*- coding: utf-8 -*-
import abc
import numpy as np
import torch
from ...utils import grad

class RK(abc.ABC):
    '''Runge-Kutta method.
    '''
    def __init__(self):
        self.f = None
        self.N = None
    
    @abc.abstractmethod
    def solver(self, x, h):
        pass
        
    def solve(self, x, h):
        for _ in range(self.N):
            x = self.solver(x, h / self.N)
        return x
    
    def flow(self, x, h, steps):
        dim = x.shape[-1] if isinstance(x, np.ndarray) else x.size(-1)
        size = len(x.shape) if isinstance(x, np.ndarray) else len(x.size())
        X = [x]
        for i in range(steps):
            X.append(self.solve(X[-1], h))
        shape = [steps + 1, dim] if size == 1 else [-1, steps + 1, dim]
        return np.hstack(X).reshape(shape) if isinstance(x, np.ndarray) else torch.cat(X, dim=-1).view(shape)

class Euler(RK):
    '''Explicit Euler method.
    '''
    def __init__(self, f, N=1):

        self.f = f
        self.N = N
        
    def solver(self, x, h):
        '''Order 1.
        x: np.ndarray or torch.Tensor of shape [dim] or [num, dim].
        h: float
        '''
        return x + h * self.f(x)
    
class ImEuler(RK):
    '''Implicit Euler method using Fix point iteration
    '''
    def __init__(self, f, N=1, iteration=2):
        self.f=f
        self.N=N
        self.iteration=iteration
        
    def solver(self, x, h):
        '''Order 1.
        x: np.ndarray or torch.Tensor of shape [dim] or [num, dim].
        h: float
        '''
        y=x
        for _ in range(self.iteration):
            y = x + h * self.f(y)
        return y    

class ImEuler_NR(RK):
    '''Implicit Euler method using Newton Raphson 
    '''
    def __init__(self, f, N=1, iteration=2):
        self.f=f
        self.N=N
        self.iteration=iteration
        
    def solver(self, x, h):
        '''Order 1.
        x: torch.Tensor of shape [num, dim].
        h: torch.Tensor of shape [num, 1].
        '''
        if isinstance(x, torch.Tensor):
            y=x.requires_grad_(True)
            I = torch.eye(x.shape[-1])
            for _ in range(self.iteration):
                df = grad(self.f(y), y).transpose(-1,-2)
                inverse = torch.inverse(I - h.unsqueeze(-2)* df)
                y = y+ ((x + h * self.f(y) - y).unsqueeze(-2) @ inverse).squeeze(-2)
                
        else: raise ValueError
        return y  

class ImEuler_SNR(RK):
    '''Implicit Euler method using Simplified Newton Raphson 
    '''
    def __init__(self, f, N=1, iteration=2):
        self.f=f
        self.N=N
        self.iteration=iteration
        
    def solver(self, x, h):
        '''Order 1.
        x: torch.Tensor of shape [dim] or [num, dim].
        h: float
        '''
        if isinstance(x, torch.Tensor):
            y=x.requires_grad_(True)
            I = torch.eye(x.shape[-1])                
            df = grad(self.f(y), y).transpose(-1,-2)
            inverse = torch.inverse(I - h* df)
            for _ in range(self.iteration):
                y = y+ (x + h * self.f(y) - y) @ inverse
        else: raise ValueError
        return y      

class Midpoint(RK):
    '''Explicit midpoint method
    '''
    def __init__(self, f, N=1):
        self.f = f
    
        self.N = N
        
    def solver(self, x, h):
        '''Order 2
            x: np.ndarray or torch.Tensor of shape [dim] or [num, dim].
            h: float
        ''' 

        return x + h * self.f(x + h/2*self.f(x))      

class ImMidpoint(RK):
    '''Implicit midpoint method
    '''
    def __init__(self, f, N=1, iteration=10):
        self.f = f
        self.N = N
        self.iteration = iteration
        
    def solver(self, x, h):
        '''Order 2
            x: np.ndarray or torch.Tensor of shape [dim] or [num, dim].
            h: float
        ''' 
        y=x
        for _ in range(self.iteration):
            y = x + h * self.f((x+y)/2)
        return y

class RK4(RK):
    '''Runge-Kutta method of order 4.
    '''
    def __init__(self, f, N=1):
        self.f = f
        self.N = N
        
    def solver(self, x, h):
        '''Order 4.
        x: np.ndarray or torch.Tensor of shape [dim] or [num, dim].
        h: float
        '''
        k1 = self.f(x)
        k2 = self.f(x + h * k1 / 2)
        k3 = self.f(x + h * k2 / 2)
        k4 = self.f(x + h * k3)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) * (h / 6) 


Integrator_list = {'explicit midpoint': Midpoint,
                   'explicit euler': Euler,
                   'rk4': RK4,
                   'implicit euler':ImEuler,
                   'implicit euler NR':ImEuler_NR,
                   'implicit midpoint':ImMidpoint, 
                  } 
        