# -*- coding: utf-8 -*-
import abc
import numpy as np
import torch

class sto_inte(abc.ABC):
    def __init__(self):
        self.f = None
        self.sig = None
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

class EM(sto_inte):
    '''Explicit Euler-Maruyama method.
    '''
    def __init__(self, f, sigma, N=1):

        self.f = f
        self.sigma = sigma
        self.N=N
        
    def solver(self, x, h):
        '''Order 1.
        x: np.ndarray or torch.Tensor of shape [dim] or [num, 1, dim].
        h: float
        '''
        return (x + h * self.f(x) 
                   + torch.sqrt(h) * ( torch.randn(x.size(), dtype=x.dtype, device=x.device).unsqueeze(dim=-2) @ self.sigma(x).transpose(dim0=-2, dim1=-1)).squeeze(dim=-2)
                  )


        