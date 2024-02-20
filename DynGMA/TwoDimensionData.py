import torch
import numpy as np



import dynlearner as ln
from dynlearner.integrator.sto_int import EM

a1 = np.sqrt(1/50).astype(np.float32)
a2 = np.sqrt(1/5).astype(np.float32)
class TwoDimensionData(ln.Data):
    def __init__(self, h=0.01, steps = 500, length=10, num_train_traj=20, num_test_traj=5, add_h=False, add_noise=False, delta=0.1):
        super(TwoDimensionData, self).__init__()     
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

            
        # self.__add_h()
        # # self.__init_testdata()
        # self.X_train_batch, self.y_train_batch = None, None
        # self.X_test_batch, self.y_test_batch = None, None
    
    @property
    def dim(self):
        return 2
    
    @staticmethod 
    def f(y):
        f=torch.empty(y.shape)
        f[...,0]=1/5*y[...,0]*(1 - y[...,0]**2) + y[...,1]*(1 + torch.sin(y[...,0]))
        f[...,1]=-y[...,1] + 2*y[...,0]*(1 - y[...,0]**2)*(1 + torch.sin(y[...,0]))
        return f
    
    @staticmethod
    def df(y):
        df1 = torch.empty(y.shape)
        df1[...,0]= 1/5* (1 - 3*y[...,0]**2) + y[...,1]*torch.cos(y[...,0])
        df1[...,1]=  1 + torch.sin(y[...,0])
        
        df2 = torch.empty(y.shape)
        df2[...,0]=  (2 - 6*y[...,0]**2)*(1 + torch.sin(y[...,0])) +  2*y[...,0]*(1 - y[...,0]**2)*(torch.cos(y[...,0]))
        df2[...,1]= -1

        return torch.cat([df1.unsqueeze(dim=-2), df2.unsqueeze(dim=-2)], dim=-2)
    
    @staticmethod
    def g(u):
        sigma = torch.tensor([[a1,0],[0,a2]])
        return sigma.repeat(u.shape[0],1,1)
    
    def __init_traj(self, N_train, N_test):
        x0 = np.random.uniform(-1,1,(N_train, self.dim)) * np.array([2,3])
        x0 = torch.tensor(x0, dtype = torch.float)
        self.train_traj = self.solver.flow(x0, torch.tensor(self.h), self.steps)
        '''Generate N trajectories
            train_traj.shape = [N_train, self.step+1, 3] N_train: number of initial points'''
            
        x0 = np.random.uniform(-1,1,(N_test, self.dim)) * np.array([2,3])
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
        
        # if self.add_h:
        #     X = torch.cat([X, self.h*torch.ones([X.shape[0], self.length])], dim=-1)
           
        return X, Y, 
        ''' Operation of reshape_traj
          X_traj = [[[00],        
                      [01],
                      [02],
                      ...
                      [0self.steps]],
                    
                    [[10],          
                      [11],
                      [12],
                      ...
                      [1self.steps]]
                    
                    ....
                    
                    [[N0],          
                      [N1],
                      [N2],
                      ...
                      [Nself.steps]]] 
          
        To make the length of trajectory is divisible by sub length
        Define end = int(self.steps/self.length)*self.length
        
        Obtain initial state 
          X_temp = [[[00],        
                      [0self.length],
                      [0self.length*2],
                      ...
                      [0self.length*{end-1}]],
                    
                    [[10],        
                      [1self.length],
                      [1self.length*2],
                      ...
                      [1self.length*{end-1}]],
                    
                    ....
                    
                    [[N0],        
                      [Nself.length],
                      [Nself.length*2],
                      ...
                      [Nself.length*{end-1}]]] 
        Stack X_temp 
          X =   [[00],        
                  [0self.length],
                  [0self.length*2],
                  ...
                  [0self.length*{end-1}],
                  [10],        
                  [1self.length],
                  [1self.length*2],
                  ...
                  [1self.length*{end-1}],
                
                ......
                
                  [N0],        
                  [Nself.length],
                  [Nself.length*2],
                  ...
                  [Nself.length*{end-1}]]
          
        Obtain the states after the initial state 
          Y_temp = [ [[01],        
                      [0self.length+1],
                      [0self.length*2+1],
                      ...
                      [0self.length*{end-1}+1],
                      [11],        
                      [1self.length+1],
                      [1self.length*2+1],
                      ...
                      [1self.length*{end-1}+1],
                    
                      ....
                    
                      [N1],        
                      [Nself.length+1],
                      [Nself.length*2+1],
                      ...
                      [Nself.length*{end-1}+1]], 
                    
                    [[02],        
                      [0self.length+2],
                      [0self.length*2+2],
                      ...
                      [0self.length*{end-1}+2],
                      [12],        
                      [1self.length+2],
                      [1self.length*2+2],
                      ...
                      [1self.length*{end-1}+2],
                    
                      ....
                    
                      [N2],        
                      [Nself.length+2],
                      [Nself.length*2+2],
                      ...
                      [Nself.length*{end-1}+2]],
                     
                    ......
                    
                    [[0self.length],        
                      [0self.length+self.length],
                      [0self.length*2+self.length],
                      ...
                      [0self.length*{end-1}+self.length],
                      [1self.length],        
                      [1self.length+self.length],
                      [1self.length*2+self.length],
                      ...
                      [1self.length*{end-1}+self.length],
                    
                      ....
                    
                      [Nself.length],        
                      [Nself.length+self.length],
                      [Nself.length*2+self.length],
                      ...
                      [Nself.length*{end-1}+self.length]]
                    ]
         
        Finally, Y is the tensor verson of Y_temp
        
        Returns
        -------
        X : data input.
        Y : data output.

        '''   

        
        

    
def main():
    import matplotlib.pyplot as plt
    Data=TwoDimensionData(h=0.1, steps = 40, length=40, num_train_traj=5, num_test_traj=5)
    x= Data.X_train
    y= Data.y_train
    

    for i in range(1):
        plt.scatter(x[i, 0], x[i, 1])
        plt.scatter(y[:, i, 0], y[:, i, 1])

    
    return 0  


if __name__=='__main__':
    main()    
