import torch
import numpy as np



import dynlearner as ln
from dynlearner.integrator.sto_int import EM

sig, r, b = 10,28, 8/3
print('r=',r)
class LSData(ln.Data):
    def __init__(self, h=0.01, steps = 500, length=10, num_train_traj=20, num_test_traj=5, add_h=False, add_noise=False, delta=0.1, r=28):
        super(LSData, self).__init__()     
        self.h=h
        self.steps = steps
        self.length=length
        self.r=r

        self.solver = EM(self.f, self.g, N=20)
        self.add_h=add_h
        
        self.train_traj = None
        self.test_traj = None
        self.__init_traj(num_train_traj, num_test_traj) 
        if add_noise:
            self.__AddNoise(delta)
        
        self.__init_data()
    
    @property
    def dim(self):
        return 3
    
    @staticmethod 
    def f(y):
        y=10*y
        f=torch.zeros(y.shape,dtype = y.dtype, device=y.device)
        f[...,0] = sig * (y[...,1]-y[...,0])
        f[...,1]= -y[...,0]*y[...,2]+ r *y[...,0] - y[...,1]
        f[...,2] = y[...,0]*y[...,1] - b*y[...,2]
        return f/10
    
    @staticmethod
    def df(y):
        df1 = torch.empty(y.shape,dtype = y.dtype, device=y.device)
        df1[...,0] = -sig
        df1[...,1] =  sig
        df1[...,2] = 0
        
        df2 = torch.empty(y.shape)
        df2[...,0] = - y[...,2] + r
        df2[...,1] = -1
        df2[...,2] = -y[...,0]

        df3 = torch.empty(y.shape)
        df3[...,0] = y[...,1]
        df3[...,1] = y[...,0]
        df3[...,2] = -b
        

        return torch.cat([df1.unsqueeze(dim=-2), df2.unsqueeze(dim=-2), df3.unsqueeze(dim=-2)], dim=-2)
    
    @staticmethod
    def g(y): 
        return torch.diag_embed(0.3*y)
        # return torch.diag_embed(
        #     0.1*torch.sin(y*torch.tensor([1.2,0.8,1.5],dtype = y.dtype, device=y.device))
        #     +torch.tensor([0.2,0.3,0.15],dtype = y.dtype, device=y.device))
        # return torch.diag_embed(0.1*torch.sin(y*torch.tensor([0.2,0.8,0.5]))+torch.tensor([0.22,0.3,0.15]))

    def Sigma(self, x):
        return self.g(x)**2
    
    def __init_traj(self, N_train, N_test, region=[-25., 25, -30, 30, -10, 60]):
        x01 = np.random.uniform(region[0],region[1],(N_train, 1))
        x02 = np.random.uniform(region[2],region[3],(N_train, 1))
        x03 = np.random.uniform(region[4],region[5],(N_train, 1))
        x0=np.hstack([x01, x02, x03])/10
        print(x0.shape)
        x0 = torch.tensor(x0, dtype = torch.float)
        self.train_traj = self.solver.flow(x0, torch.tensor(self.h), self.steps)
        '''Generate N trajectories
            train_traj.shape = [N_train, self.step+1, 3] N_train: number of initial points'''
            
        x01 = np.random.uniform(region[0],region[1],(N_test, 1))
        x02 = np.random.uniform(region[2],region[3],(N_test, 1))
        x03 = np.random.uniform(region[4],region[5],(N_test, 1))
        x0=np.hstack([x01, x02, x03])/10
        print(x0.shape)
        x0 = torch.tensor(x0, dtype = torch.float)
        self.test_traj = self.solver.flow(x0, torch.tensor(self.h), self.steps)
        '''Generate N trajectories
            test_traj.shape = [N_test, self.step+1, 3] N_test: number of initial points'''        

    def __AddNoise(self, delta):
        noise = torch.rand(self.train_traj.shape)*2*delta - delta
        self.train_traj = self.train_traj*(1+noise)
        
        noise = torch.rand(self.test_traj.shape)*2*delta - delta
        self.test_traj = self.test_traj*(1+noise)
   
        
    def __init_data(self):
        self.X_train, self.y_train = self.reshape_traj(self.train_traj, self.length)
        self.X_test,  self.y_test  = self.reshape_traj(self.test_traj,  self.length)  
    
    def reshape_traj(self, X_traj, length):
        end = int(self.steps/self.length)*self.length
        
        X_temp = X_traj[:, 0:end:self.length, :]
        '''X_temp = X_traj[:, 0, lenght, 2length, ..., :]'''
        X = X_temp.reshape([-1, self.dim])

        Y_temp=[]
        for i in range(1, self.length+1):
            Y_temp.append(X_traj[:, i:end+i:self.length, :].reshape([-1, self.dim]))
        Y = torch.cat(Y_temp, dim=0).view([self.length, -1, self.dim])
        
        return X, Y