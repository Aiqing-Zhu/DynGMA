#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# The following code segment is obtained by modified the code in https://github.com/yuhj1998/OnsagerNet/blob/main/Lorenz/test_ode_Lorenz.py
# Original author: Yu Haijun
# Original project: OnsagerNet

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from .fnn import FNN
from .odenet import ODENet

def makePDM(matA):
    """ Make Positive Definite Matrix from a given matrix
    matA has a size (batch_size x N x N) """
    AL = torch.tril(matA, 0)
    AU = torch.triu(matA, 1)
    Aant = AU - torch.transpose(AU, 1, 2)
    Asym = torch.bmm(AL, torch.transpose(AL, 1, 2))
    return Asym,  Aant


def makeSPD(A, n):
    """ Make Symmetric Positive Definite matrix from a given matrix
    A has a size (batch_size x N), where N = n*(n+1)/2 """
    A = A.view(-1, (n*(n+1))//2)
    bs = A.shape[0]
    matA = torch.zeros(bs, n*n)
    tril_ind = torch.tril_indices(n, n)
    matA[:, tril_ind[0, :]*n+tril_ind[1, :]] = A
    matA = matA.view(-1, n, n)
    AL = torch.tril(matA, 0)
    Asym = torch.bmm(AL, torch.transpose(AL, 1, 2))
    return Asym


class OnsagerNet(ODENet):
    """ A neural network to for the rhs function of an ODE,
    used to fitting data """

    def __init__(self, n_nodes=None, forcing=True, ResNet=True,
                 pot_beta=0.0,
                 ons_min_d=0.0,
                 init_gain=0.1,
                 activation='tanh',
                 f_linear=True,
                 ):
        super().__init__()
        if n_nodes is None:   # used for subclasses
            return
        self.nL = n_nodes.size
        self.nVar = n_nodes[0]
        self.nNodes = np.zeros(self.nL+1, dtype=np.int32)
        self.nNodes[:self.nL] = n_nodes
        self.nNodes[self.nL] = self.nVar**2
        self.nPot = self.nVar
        self.forcing = forcing
        
        self.pot_beta = pot_beta
        self.ons_min_d = ons_min_d
        self.init_gain = init_gain
        
        self.activation = activation 
        
        self.f_linear = f_linear
        if ResNet:
            self.ResNet = 1.0
            assert np.sum(n_nodes[1:]-n_nodes[1]) == 0, \
                f'ResNet structure is not implemented for {n_nodes}'
        else:
            self.ResNet = 0.0
        
        self.modus = self.__init_modules()
            
    def __init_modules(self): 
        self.baselayer = nn.ModuleList([nn.Linear(self.nNodes[i],
                                                  self.nNodes[i+1])
                                        for i in range(self.nL-1)])
        self.MatLayer = nn.Linear(self.nNodes[self.nL-1], self.nVar**2)
        self.PotLayer = nn.Linear(self.nNodes[self.nL-1], self.nPot)
        self.PotLinear = nn.Linear(self.nVar, self.nPot)

        bias_eps = 0.5
        for i in range(self.nL-1):
            init.xavier_uniform_(self.baselayer[i].weight, gain=self.init_gain)
            init.uniform_(self.baselayer[i].bias, 0, bias_eps*self.init_gain)

        init.xavier_uniform_(self.MatLayer.weight, gain=self.init_gain)
        w = torch.empty(self.nVar, self.nVar, requires_grad=True)
        nn.init.orthogonal_(w, gain=1.0)
        self.MatLayer.bias.data = w.view(-1, self.nVar**2)

        init.orthogonal_(self.PotLayer.weight, gain=self.init_gain)
        init.uniform_(self.PotLayer.bias, 0, self.init_gain)
        init.orthogonal_(self.PotLinear.weight, gain=self.init_gain)
        init.uniform_(self.PotLinear.bias, 0, self.init_gain)

        if self.forcing:
            if self.f_linear:
                self.lforce = nn.Linear(self.nVar, self.nVar)
            else:
                self.lforce = nn.Linear(self.nNodes[self.nL-1], self.nVar)
            init.orthogonal_(self.lforce.weight, self.init_gain)
            init.uniform_(self.lforce.bias, 0.0, bias_eps*self.init_gain)

        self.F_act = self.Act
            
    def vf(self, inputs, test=False):
        shape = inputs.shape
        inputs = inputs.view(-1, self.nVar)
        with torch.enable_grad():
            inputs.requires_grad_(True)
            if not test:
                inputs.retain_grad()
            output = self.F_act(self.baselayer[0](inputs))
            for i in range(1, self.nL-1):
                output = (self.F_act(self.baselayer[i](output))
                          + self.ResNet*output)
            PotLinear = self.PotLinear(inputs)
            Pot = self.PotLayer(output) + PotLinear
            V = torch.sum(Pot**2) + self.pot_beta * torch.sum(inputs**2)
            if test:
                g, = torch.autograd.grad(V, inputs)
            else:
                g, = torch.autograd.grad(V, inputs, create_graph=True)
            g = - g.view(-1, self.nVar, 1)

        matA = self.MatLayer(output)
        matA = matA.view(-1, self.nVar, self.nVar)
        AM, AW = makePDM(matA)
        MW = AW+AM

        if self.forcing:
            if self.f_linear:
                lforce = self.lforce(inputs)
            else:
                lforce = self.lforce(output)

        output = torch.matmul(MW, g) + self.ons_min_d * g
        if self.forcing:
            output = output + lforce.view(-1, self.nVar, 1)

        output = output.view(*shape)
        return output

    def calc_potential(self, inputs):
        ''' Calculate the potential for post-analysis '''
        output = inputs.view(-1, self.nVar)
        PotLinear = self.PotLinear(output)
        output = self.F_act(self.baselayer[0](output))
        for i in range(1, self.nL-1):
            output = (self.F_act(self.baselayer[i](output))
                      + self.ResNet*output)
        Pot = self.PotLayer(output) + PotLinear
        V = torch.sum(Pot**2, dim=1)
        V += self.pot_beta*torch.sum(inputs**2, dim=1)
        return V

    def save_potential(self, hrange, n=30,
                       savefile='results/OnsagerNet_test_pot'):
        d = min(len(hrange[0]), self.nVar)
        np.savetxt(savefile+'_meta.txt', hrange,
                   delimiter=', ', fmt='%.3e')
        nx = ny = nz = 100
        x = np.linspace(hrange[0][0], hrange[1][0], nx)
        y = np.linspace(hrange[0][1], hrange[1][1], ny)
        device = torch.device('cpu')
        if d == 2:
            xx, yy = np.meshgrid(x, y)
            inputs = np.stack([xx, yy], axis=-1).reshape([-1, 2])
            inputs = torch.FloatTensor(inputs)
        else:
            z = np.linspace(hrange[0][2], hrange[1][2], nz)
            xx, yy, zz = np.meshgrid(x, y, z)
            input3 = np.stack([xx, yy, zz], axis=-1).reshape([-1, 3])
            inputs = np.zeros([input3.shape[0], self.nVar])
            inputs[:, :3] = input3
            inputs = torch.FloatTensor(inputs)
        inputs.to(device)
        self.to(device)
        with torch.no_grad():
            Pot = self.calc_potential(inputs)
        Pot = Pot.detach().numpy().reshape([nx, ny, nz])
        np.savetxt(savefile+'.txt.gz', Pot.reshape([-1, nz]),
                   delimiter=', ', fmt='%.3e')

    def calc_potHessian(self, inputs):
        inputs = inputs.view(-1, self.nVar)
        bs = inputs.shape[0]
        with torch.enable_grad():
            inputs.requires_grad_(True)
            inputs.retain_grad()
            output = inputs.view(-1, self.nVar)
            PotLinear = self.PotLinear(output)
            output = self.F_act(self.baselayer[0](output))
            for i in range(1, self.nL-1):
                output = (self.F_act(self.baselayer[i](output))
                          + self.ResNet*output)
            Pot = self.PotLayer(output) + PotLinear
            V0 = torch.sum(Pot**2, dim=1)
            V0 += self.pot_beta * torch.sum(inputs**2, dim=1)
            V = torch.sum(V0)
            G, = torch.autograd.grad(V, inputs, create_graph=True)

            nVar = self.nVar
            H = torch.zeros(bs, nVar, nVar, requires_grad=False)
            for i in range(nVar):
                y = torch.zeros(nVar, requires_grad=False)
                y[i] = 1
                Gy = torch.sum(G @ y)
                Gy.backward(retain_graph=True)
                H[:, i, :] = inputs.grad
                inputs.grad.data.zero_()
        G = G.detach().view(-1, self.nVar)
        H = H.detach().view(-1, self.nVar, self.nVar)
        V0 = V0.detach()
        return G, H, V0  # Gradient, Hessian, and Potential

    def calc_matA(self, inputs):
        inputs = torch.tensor(inputs).float().view(-1, self.nVar)
        output = self.F_act(self.baselayer[0](inputs))
        for i in range(1, self.nL-1):
            output = (self.F_act(self.baselayer[i](output))
                      + self.ResNet*output)
        matA = self.MatLayer(output)
        matA = matA.view(-1, self.nVar, self.nVar)
        AM, AW = makePDM(matA)
        return AM
