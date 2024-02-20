import torch
import numpy as np
import os


import dynlearner as ln
from dynlearner.integrator.sto_int import EM


class EMTData(ln.Data):
    '''Training data for EMT
    '''
    def __init__(self, h=0.01, steps = 3900, length=1, num_train_traj=20, num_test_traj=5, add_h=False, add_noise=False, delta=0.1):
        super(EMTData, self).__init__()     
        self.h=h
        self.steps = steps
        self.length=length
        

        self.solver = EM(self.f, self.g, N=10)
        self.add_h=add_h
        
        self.train_traj = None
        self.test_traj = None
        self.__init_traj(num_train_traj, num_test_traj) 
        if add_noise:
            self.__AddNoise(delta)
        
        self.__init_data()
    
    @property
    def dim(self):
        return 10
    
    @staticmethod 
    def f(y):
        f=torch.zeros(y.shape,dtype = y.dtype, device=y.device)
        f[...,0] = 0.8* (0.5/(0.5+y[...,0]) + 0.5**2/(0.5**2+y[...,4]**2) + 0.5**4/(0.5**4+y[...,6]**4) ) - y[...,0]
        
        f[...,1] = (  0.2* (y[...,0]/(0.5+y[...,0]) + y[...,1]**2/(0.5**2+y[...,1]**2)) 
                    + 0.8* (0.5**6/(0.5**6+y[...,3]**6) + 0.5**4/(0.5**4+y[...,5]**4)) - y[...,1]  )
        
        f[...,2] = (  0.2* (y[...,2]**2/(0.5**2+y[...,2]**2) + y[...,8]**4/(0.5**4+y[...,8]**4))
                    + 0.8* 0.5**4/(0.5**4+y[...,5]**4) - y[...,2]   )
        
        f[...,3] = (  0.2* y[...,2]**4/(0.5**4+y[...,2]**4)
                    + 0.8* ( 0.5 /(0.5 +y[...,0]) + 0.5**3/(0.5**3 +y[...,1]**3 ))- y[...,3]   )
        
        
        f[...,4] = (0.8* ( 0.5 /(0.5 +y[...,0]) + 0.5**2/(0.5**2 +y[...,1]**2 ))- y[...,4]   )        

        
        f[...,5] = (0.8* ( 0.5**4 /(0.5**4 +y[...,1]**4) + 0.5**4/(0.5**4 +y[...,2]**4 ))- y[...,5]   )     
        
        f[...,6] = (0.8* y[...,6]**2 /(0.5**2 +y[...,6]**2) + 7*y[...,7]**5/(0.5**5 +y[...,7]**5)
                    +0.8* 0.5**4/(0.5**4 +y[...,8]**4)- 4*y[...,6]*y[...,9] - y[...,6] ) 
        
        f[...,7] = (0.8* ( 0.5**4 /(0.5**4 +y[...,0]**4) + 0.5/(0.5 +y[...,9]))- y[...,7]   ) 
        
        f[...,8] = ( 0.2* y[...,8]**2/(0.5**2 +y[...,8]**2) + 0.8* 0.5**4/(0.5**4 +y[...,6]**4 ) - y[...,8]   ) 
        
        f[...,9] = ( 0.1 + 4/(1 +y[...,9]**3) - 4*y[...,6]*y[...,9] - y[...,9] ) 
        
        return f 

    
    @staticmethod
    def g(y):
        sigma =torch.eye(10, dtype = y.dtype, device=y.device)*0.2
        return sigma.repeat(y.shape[0],1,1)
    
    def Sigma(self, x):
        return torch.eye(10, dtype = y.dtype, device=y.device)*0.04
    
    def __init_traj(self, N_train, N_test):
        region= np.array([[0,2]]*10)
        region[6] = [0,6]
        x0 = []
        for i in range(10):
            x0.append(np.random.uniform(region[i, 0],region[i, 1],(N_train, 1)))
        x0=np.hstack(x0)
        print(x0.shape)
        x0 = torch.tensor(x0, dtype = torch.float)
        
        if os.path.exists('EMTdata/train_traj_steps{}_num{}.npy'.format(self.steps, N_train)):
                print('Train data has been generated')
        else:
            if not os.path.isdir('./EMTdata'): os.makedirs('./EMTdata')
            train_traj = self.solver.flow(x0, torch.tensor(0.005), self.steps)[:, 400:self.steps:100, :]
            np.save('EMTdata/train_traj_steps{}_num{}.npy'.format(self.steps, N_train), train_traj)
                
        train_traj = np.load('EMTdata/train_traj_steps{}_num{}.npy'.format(self.steps, N_train))
        self.train_traj = torch.tensor(train_traj, dtype=torch.float32)
        
                          
        x0 = []
        for i in range(10):
            x0.append(np.random.uniform(region[i, 0],region[i, 1],(N_test, 1)))
        x0=np.hstack(x0)
        print(x0.shape)
        x0 = torch.tensor(x0, dtype = torch.float)
        
        if os.path.exists('EMTdata/test_traj_steps{}_num{}.npy'.format(self.steps, N_test)):
                print('Test data has been generated')
        else:
            if not os.path.isdir('./EMTdata'): os.makedirs('./EMTdata')
            test_traj = self.solver.flow(x0, torch.tensor(0.005), self.steps)[:, 400:self.steps:100, :]
            np.save('EMTdata/test_traj_steps{}_num{}.npy'.format(self.steps, N_test), test_traj)
                
        test_traj = np.load('EMTdata/test_traj_steps{}_num{}.npy'.format(self.steps, N_test))
        self.test_traj = torch.tensor(test_traj, dtype=torch.float32)  
 
    def __AddNoise(self, delta):
        noise = torch.rand(self.train_traj.shape)*2*delta - delta
        self.train_traj = self.train_traj*(1+noise)
        
        noise = torch.rand(self.test_traj.shape)*2*delta - delta
        self.test_traj = self.test_traj*(1+noise)
   
        
    def __init_data(self):
        print(self.train_traj.shape)
        self.X_train = self.train_traj.reshape([-1, self.dim])
        print(self.X_train.shape)
        
        Y_temp=[]
        Y=self.X_train
        solver = EM(self.f, self.g, N=10*int(self.h/0.005))
        for i in range(self.length):
            Y = solver.solve(Y, torch.tensor(self.h))
            Y_temp.append(Y)
        self.y_train = torch.cat(Y_temp, dim=0).view([self.length, -1, self.dim])

        self.X_test = self.test_traj.reshape([-1, self.dim])
        Y_temp=[]
        Y=self.X_test 
        for i in range(self.length):
            Y = solver.solve(Y, torch.tensor(self.h))
            Y_temp.append(Y)
        self.y_test = torch.cat(Y_temp, dim=0).view([self.length, -1, self.dim])
        
    
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