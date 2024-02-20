import torch
import numpy as np 

eps_safe = 1e-45
def safe_log(x):
    return torch.log(torch.clamp(x, min=eps_safe)) 

######################################################################### 

#           Algorithm 2 for variable step sizes

######################################################################### 
class NGaussAppro_V():
    '''Gauss Assumed Density Approximation
    '''
    def __init__(self, f, df, sigma):

        self.f = f
        self.df = df
        self.sigma = sigma
        
    def solver(self, x, H):
        '''
        x: np.ndarray or torch.Tensor of shape [dim] or [num, 1, dim].
        h: float
        ''' 

        ma= x + H /2 * self.f(x)
        m = x + H * self.f(ma)
        
        sP = torch.sqrt(H).unsqueeze(-1) * self.sigma(ma)
        return m, sP
 
        
class NMGaussAppro_V():
    '''Multiple Gauss Assumed Density Approximation
    '''
    def __init__(self, f, df, sigma, steps):

        self.f = f
        self.df = df
        self.sigma = sigma
        
        self.subsolve = NGaussAppro_V(f, df, sigma)
        
        self.steps=steps
        
 
        self.lambd = 1 
        ''' Control the distance between new and old mean points:
            new mean = old mean + sqrt(self.lambd + Dimension) * sqrt(covariance)
        '''
        self.epsilon = 1
        ''' Contron the size of the last step:
            self.epsilon * self.subtime \leq the size of the last step < (1 + self.epsilon) * self.subtime
        '''
    
    
    def __GenerateNewPoint(self, m, sqrtP, D, Weight, time):
        '''
        Using the mean and covarance gennetates new points

        '''
        X_out = m

        xi_i = np.sqrt(self.lambd + D)

        '''Define coefficients'''
        Time=time
        for d in range(D):
            xminus = m - xi_i  * sqrtP[...,d, :]               
            xplus  = m + xi_i * sqrtP[...,d, :]           
            X_out = torch.cat([X_out, xminus, xplus], dim=-2)
            
            Time = torch.cat([time, Time, time], dim=-2)
        '''return new vectors using the mean and covariance with given coefficients'''

        W = torch.ones([2*D + 1, 1], dtype=m.dtype, device=m.device) * 1/2/(self.lambd + D)
        W[0] = self.lambd/(self.lambd + D)
        '''Define coefficients'''
        Weight = (W@Weight).reshape([1, -1])
        '''[1, number of Gauss]'''

        return X_out, Weight, Time
    

    def solver(self, x, Time):
        '''
        input initial point x, final time Time
        output the mean and covarance of several Gaussion, as well as the corresponding weights

        '''
        x_a=x
        D = x.shape[-1]
        x_size = x.shape[0]
        
        Weight=torch.ones([1, 1], dtype=x.dtype, device=x.device)
 
        for i in range(self.steps-1):
            m_out, sqrtP_out = self.subsolve.solver(x_a, Time/self.steps)
            x_a, Weight, Time = self.__GenerateNewPoint(m_out, sqrtP_out, D, Weight, Time)

        m_out, sqrtP_out = self.subsolve.solver(x_a, Time/self.steps)
        '''final step'''
        
        m_out = m_out.reshape(-1, x_size, D)
        '''reshape mean as [number of Gaussian, number of x, dimension]'''
        sqrtP_out = sqrtP_out.reshape(-1, x_size, D, D)
        '''reshape covariance as [number of Gaussian, number of x, dimension, dimension]'''
    
        return m_out, sqrtP_out, Weight 

        
  
    def solverTL(self, x, Time):
        '''similar to solver(), Here Time is a list [time1, time2, ... ]
        For nonlinear case, it is required that timei = ki * subtime 
        '''
        D = x.shape[-1]
        x_size = x.shape[0]
        
        Weight=torch.ones([1, 1], dtype=x.dtype, device=x.device)
        Mean=[]
        sqrtCov=[]
        WeightList=[]
        
        
        m_a=x
         
        '''Initial value of iteration'''

        for index in range(Time.shape[-1]):
            time = Time[index] 
            if index == 0:
                m_a, sqrtP_a = self.subsolve.solver(m_a, time/self.steps)
                
            else:
                X_a, Weight, time = self.__GenerateNewPoint(m_a, sqrtP_a, D, Weight, time)
                m_a, sqrtP_a = self.subsolve.solver(X_a, time/self.steps)

            for i in range(self.steps-1):
                X_a, Weight, time = self.__GenerateNewPoint(m_a, sqrtP_a, D, Weight, time)
                m_a, sqrtP_a = self.subsolve.solver(X_a, time/self.steps)
            
            m_out, sqrtP_out, Weight_out = m_a, sqrtP_a, Weight
            WeightList.append(Weight_out)
            Mean.append(m_out.reshape(-1, x_size, D))
            sqrtCov.append(sqrtP_out.reshape(-1, x_size, D, D))
            
        return Mean, sqrtCov, WeightList    