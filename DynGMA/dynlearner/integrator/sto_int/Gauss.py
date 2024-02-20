import torch
import numpy as np 
import math

eps_safe = 1e-45
def safe_log(x):
    return torch.log(torch.clamp(x, min=eps_safe)) 

######################################################################### 

#                     Algorithm 1

######################################################################### 
class GaussAppro():
    '''Gauss Assumed Density Approximation
    '''
    def __init__(self, f, df, sigma, N=1):

        self.f = f
        self.df = df
        self.sigma = sigma
        self.N=N
        
    
    def Cov_ODE(self, m, P):
        return (
                P @ self.df(m).transpose(dim0=-2, dim1=-1)
                + self.df(m) @ P
                + self.sigma(m)@self.sigma(m).transpose(dim0=-2, dim1=-1)
                )

    def ODEsolver(self, m, P, h):
        
        ma = m + h/2 * self.f(m)
        m1 = m + h * self.f(ma)
        
        Grad_f =  self.df(ma)
        Iden = torch.eye(m.shape[-1], dtype=m.dtype, device=m.device)
        
        P_sig_a1 = (Iden + h/2 * Grad_f) @ self.sigma(ma)
        P_sig_a2 = (Iden + h * Grad_f)
        P1 = (
            h*P_sig_a1 @ P_sig_a1.transpose(dim0=-2, dim1=-1)
            + 
            P_sig_a2 @ P @ P_sig_a2.transpose(dim0=-2, dim1=-1)
            )
        '''
         (I + h/2 df(ma) ) @ sigma(ma)@ sigma(ma)^T @ (I + h/2 df(ma) )^T 
        +(I + h df(ma) + h^2/2 df(ma)^2) @ P @ (I + h df(ma) + h^2/2 df(ma)^2)^T 
        '''
                        
        return m1, P1
        
    def solver_0P(self, x, H):
        '''
        x: np.ndarray or torch.Tensor of shape [dim] or [num, 1, dim].
        h: float
        '''
        ma = x + H / self.N /2 * self.f(x)
        m = x + H / self.N * self.f(ma)
        
        Grad_f =  self.df(ma)
        Iden = torch.eye(m.shape[-1], dtype=m.dtype, device=m.device)
        
        P_sig_a1 = (Iden + H / self.N /2 * Grad_f) @ self.sigma(ma)
        P = (
            H / self.N*P_sig_a1 @ P_sig_a1.transpose(dim0=-2, dim1=-1)
            ) 
        for _ in range(self.N-1):
            m, P = self.ODEsolver(m, P, H / self.N) 
        return m, P

    def solver(self, m, P, H):
        '''
        m: np.ndarray or torch.Tensor of shape [dim] or [num, 1, dim].
        P: float
        '''
        for _ in range(self.N):
            m, P = self.ODEsolver(m, P, H / self.N)
        return m, P
    
    def Solver(self, m, P, H):
        if H==0:
            return m, P
        else:
            return self.solver(m, P, H)
    
    def Solver_0P(self, m, H):
        if H==0:
            return m
        else:
            return self.solver_0P(m, H)      
        
class MGaussAppro():
    '''Multiple Gauss Assumed Density Approximation
    '''
    def __init__(self, f, df, sigma, st = 1, sN=10):

        self.f = f
        self.df = df
        self.sigma = sigma
        self.sN=sN
        self.subtime = st
        self.subsolve = GaussAppro(f, df, sigma, sN)
        
 
        self.lambd = 1 
        ''' Control the distance between new and old mean points:
            new mean = old mean + sqrt(self.lambd + Dimension) * sqrt(covariance)
        '''
        self.epsilon = 1
        ''' Contron the size of the last step:
            self.epsilon * self.subtime \leq the size of the last step < (1 + self.epsilon) * self.subtime
        '''
        

    def solver(self, x, Time):
        '''
        input initial point x, final time Time
        output the mean and covarance of several Gaussion, as well as the corresponding weights

        '''
        x_a=x
        D = x.shape[-1]
        x_size = x.shape[0]
        
        Weight=torch.ones([1, 1], dtype=x.dtype, device=x.device)
        steps = int(Time/self.subtime-self.epsilon)
        '''self.epsilon * self.subtime \leq the size of the last step < (1 + self.epsilon) * self.subtime
        '''
        
        for i in range(steps):
            m_out, P_out = self.subsolve.solver_0P(x_a, self.subtime)
            x_a, Weight = self.__GenerateNewPoint(m_out, P_out, D, Weight)

        m_out, P_out = self.subsolve.solver_0P(x_a, Time - steps * self.subtime)
        '''final step'''
        
        m_out = m_out.reshape(-1, x_size, D)
        '''reshape mean as [number of Gaussian, number of x, dimension]'''
        P_out = P_out.reshape(-1, x_size, D, D)
        '''reshape covariance as [number of Gaussian, number of x, dimension, dimension]'''
    
        return m_out, P_out, Weight 
    
    
    def __GenerateNewPoint(self, m, P, D, Weight):
        '''
        Using the mean and covarance gennetates new points

        '''
        X_out = m
        try:
            sqrtP = torch.linalg.cholesky(P + eps_safe*torch.eye(m.shape[-1], dtype=m.dtype, device=m.device))
            xi_i = np.sqrt(self.lambd + D)

            '''Define coefficients'''

            for d in range(D):
                xminus = m - xi_i  * sqrtP[...,d, :]               
                xplus  = m + xi_i * sqrtP[...,d, :]           
                X_out = torch.cat([X_out, xminus, xplus], dim=-2)
            '''return new vectors using the mean and covariance with given coefficients'''

            W = torch.ones([2*D + 1, 1], dtype=m.dtype, device=m.device) * 1/2/(self.lambd + D)
            W[0] = self.lambd/(self.lambd + D)
            '''Define coefficients'''
            Weight = (W@Weight).reshape([1, -1])
            '''[1, number of Gauss]'''
            
        except RuntimeError as e:
            print(e)    
            
        return X_out, Weight
        
  
    def solverTL(self, x, Time):
        '''similar to solver(), Here Time is a list [time1, time2, ... ]
        '''
        D = x.shape[-1]
        x_size = x.shape[0]
        
        Weight=torch.ones([1, 1], dtype=x.dtype, device=x.device)
        Mean=[]
        Cov=[]
        WeightList=[]
        
        
        m_a=x
        PassedTime = 0
        '''Initial value of iteration'''

        for index, timestep in enumerate(Time):
            steps = int((timestep + PassedTime)/self.subtime-self.epsilon)
            '''similar to solver, self.epsilon here is to control the step size of the final step'''

            if steps ==0:
                if index == 0:
                    m_a, P_a = self.subsolve.Solver_0P(m_a, timestep)
                    '''Generate the covarance P for the first iteration'''
                else:
                    m_a, P_a = self.subsolve.Solver(m_a, P_a, timestep)
                PassedTime = timestep + PassedTime
                m_out, P_out, Weight_out = m_a, P_a, Weight
                '''first and final step'''
                
            else:
                if index == 0:
                    m_a, P_a = self.subsolve.Solver_0P(m_a, self.subtime - PassedTime)
                    '''Generate the covarance P for the first iteration'''
                else:
                    m_a, P_a = self.subsolve.Solver(m_a, P_a, self.subtime - PassedTime)
                '''first step'''    
                
                for i in range(steps-1):
                    X_a, Weight = self.__GenerateNewPoint(m_a, P_a, D, Weight)
                    m_a, P_a = self.subsolve.Solver_0P(X_a, self.subtime)

                    
                PassedTime = (timestep + PassedTime) - steps * self.subtime
                
                '''final step'''
                if PassedTime > self.subtime:
                    X_a, Weight = self.__GenerateNewPoint(m_a, P_a, D, Weight)
                    m_a, P_a = self.subsolve.Solver_0P(X_a, self.subtime)
                    
                    m_out, P_out = self.subsolve.Solver(m_a, P_a, PassedTime - self.subtime)
                    Weight_out = Weight
                    PassedTime = PassedTime - self.subtime
                    
                    if index != len(Time) - 1:
                        X_a, Weight = self.__GenerateNewPoint(m_a, P_a, D, Weight)
                        m_a, P_a = self.subsolve.Solver_0P(X_a, PassedTime)
                        
                    
                else:
                    X_a, Weight = self.__GenerateNewPoint(m_a, P_a, D, Weight)
                    m_a, P_a = self.subsolve.Solver_0P(X_a, PassedTime)
                    
                    m_out, P_out, Weight_out = m_a, P_a, Weight
                '''final step'''
                
            WeightList.append(Weight_out)
            Mean.append(m_out.reshape(-1, x_size, D))
            Cov.append(P_out.reshape(-1, x_size, D, D))

        return Mean, Cov, WeightList


    
class GaussDensity():
    def __init__(self, m, P):
        self.m = m
        self.P = P
        ''' m: torch.tensor of shape [number of data, dimension]
            P: torch.tensor of shape [number of data, dimension, dimension]

        '''
    
    def density(self, x):
        '''x torch.tensor of shape [number of data, dimension]
            
        Limitation: Unable to calculate the value of a density function at multiple points
        '''
        
        a1 = (
            -(x-self.m).unsqueeze(-2)@(torch.inverse(self.P)/2 @ (x-self.m).unsqueeze(-1))
            )    
        '''(x-m)^T P^{-1}/2 (x-m)'''
        

        a2 = -torch.log(torch.det(self.P))/2
 
        density = torch.exp(a1.squeeze(dim=-1).squeeze(dim=-1) + a2)# / (2*3.1415926)**(x.shape[-1]/2) 
        return density  
    
    def LogDensity(self, x):         
        a1 = (
            -(x-self.m).unsqueeze(-2)@(torch.inverse(self.P)/2 @ (x-self.m).unsqueeze(-1))
            ).squeeze(dim=-1).squeeze(dim=-1)
        '''(x-m)^T P^{-1}/2 (x-m)'''
         
        a2 = -torch.log(torch.det(self.P))/2

        logdensity = a1 + a2 - np.log(2*3.1415926)*(x.shape[-1]/2)
        return logdensity     
    
  
class MGaussDensity():
    def __init__(self, m, P, Weight):
        self.m = m
        self.P = P
        self.Weight = Weight
        ''' m: torch.tensor of shape [number of Gauss, number of data, dimension]
            P: torch.tensor of shape [number of Gauss, number of data, dimension, dimension]
            Weight: torch.tensor of shape [1, number of Gauss]
        '''
    
    def density(self, x):
        '''x torch.tensor of shape [number of data, dimension]
            
        Limitation: Unable to calculate the value of a density function at multiple points
        '''
        
        a1 = (
            -(x-self.m).unsqueeze(-2)@(torch.inverse(self.P)/2 @ (x-self.m).unsqueeze(-1))
            )    
        '''(x-m)^T P^{-1}/2 (x-m)'''
        
        a2 = -torch.log(torch.det(self.P))/2
        # 
        SingleDensity = torch.exp(a1.squeeze(dim=-1).squeeze(dim=-1) + a2) / (2*3.1415926)**(x.shape[-1]/2) 
        density  = self.Weight@SingleDensity

        return density

    def LogDensity(self, x):
        a1 = (
            -(x-self.m).unsqueeze(-2)@(torch.inverse(self.P)/2 @ (x-self.m).unsqueeze(-1))
            ).squeeze(dim=-1).squeeze(dim=-1)
        '''(x-m)^T P^{-1}/2 (x-m)'''
        
        a2 = 1/torch.sqrt(torch.det(self.P)) *  (self.Weight.transpose(dim0=-1, dim1=-2))
        logdensity1 = a1[0] + torch.log(a2[0]) - np.log(2*3.1415926)*(x.shape[-1]/2)
        logdensity2 = torch.log(1 + (a2[1:]/a2[0] * torch.exp(a1[1:]-a1[0])).sum(dim=-2))
        return logdensity1 + logdensity2 

    

    def SafeLogDensity(self, x):
        a1 = (
            -(x-self.m).unsqueeze(-2)@(torch.inverse(self.P)/2 @ (x-self.m).unsqueeze(-1))
            ).squeeze(dim=-1).squeeze(dim=-1)
        '''(x-m)^T P^{-1}/2 (x-m)'''
        
        a2 = -safe_log(torch.det(self.P))/2        
        a3 = torch.log(self.Weight).transpose(dim0=-1, dim1=-2)

        power = a1 + a2 + a3
        Max = power.max(dim=-2)[0]
        return Max - np.log(2*3.1415926)*(x.shape[-1]/2) + torch.log(torch.exp(power-Max).sum(dim=-2))
    
    

######################################################################### 

#                     Algorithm 2

######################################################################### 
class NGaussAppro():
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
        ma = x + H /2 * self.f(x)
        m = x + H * self.f(ma)
        return m, math.sqrt(H) *self.sigma(ma)
 
        
class NMGaussAppro():
    '''Multiple Gauss Assumed Density Approximation
    '''
    def __init__(self, f, df, sigma, st = 1):

        self.f = f
        self.df = df
        self.sigma = sigma
        
        self.subtime = st
        self.subsolve = NGaussAppro(f, df, sigma)
        
        
        # self.h=0.01
        self.lambd = 1 
        ''' Control the distance between new and old mean points:
            new mean = old mean + sqrt(self.lambd + Dimension) * sqrt(covariance)
        '''
        self.epsilon = 1
        ''' Contron the size of the last step:
            self.epsilon * self.subtime \leq the size of the last step < (1 + self.epsilon) * self.subtime
        '''
    
    
    def __GenerateNewPoint(self, m, sqrtP, D, Weight):
        '''
        Using the mean and covarance gennetates new points

        '''
        X_out = m

        xi_i = np.sqrt(self.lambd + D)

        '''Define coefficients'''

        for d in range(D):
            xminus = m - xi_i  * sqrtP[...,d, :]               
            xplus  = m + xi_i * sqrtP[...,d, :]           
            X_out = torch.cat([X_out, xminus, xplus], dim=-2)
        '''return new vectors using the mean and covariance with given coefficients'''

        W = torch.ones([2*D + 1, 1], dtype=m.dtype, device=m.device) * 1/2/(self.lambd + D)
        W[0] = self.lambd/(self.lambd + D)
        '''Define coefficients'''
        Weight = (W@Weight).reshape([1, -1])
        '''[1, number of Gauss]'''

        return X_out, Weight
    

    def solver(self, x, Time):
        '''
        input initial point x, final time Time
        output the mean and covarance of several Gaussion, as well as the corresponding weights

        '''
        x_a=x
        D = x.shape[-1]
        x_size = x.shape[0]
        
        Weight=torch.ones([1, 1], dtype=x.dtype, device=x.device)
        steps = int(Time/self.subtime-self.epsilon)
        '''self.epsilon * self.subtime \leq the size of the last step < (1 + self.epsilon) * self.subtime
        '''
        
        for i in range(steps):
            m_out, sqrtP_out = self.subsolve.solver(x_a, self.subtime)
            x_a, Weight = self.__GenerateNewPoint(m_out, sqrtP_out, D, Weight)

        m_out, sqrtP_out = self.subsolve.solver(x_a, Time - steps * self.subtime)
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

        for index, timestep in enumerate(Time):
            if index == 0:
                m_a, sqrtP_a = self.subsolve.solver(m_a, self.subtime)
            else:
                X_a, Weight = self.__GenerateNewPoint(m_a, sqrtP_a, D, Weight)
                m_a, sqrtP_a = self.subsolve.solver(X_a, self.subtime)
            steps = int(timestep/self.subtime) 
            for i in range(steps-1):
                X_a, Weight = self.__GenerateNewPoint(m_a, sqrtP_a, D, Weight)
                m_a, sqrtP_a = self.subsolve.solver(X_a, self.subtime)
            
            m_out, sqrtP_out, Weight_out = m_a, sqrtP_a, Weight
            WeightList.append(Weight_out)
            Mean.append(m_out.reshape(-1, x_size, D))
            sqrtCov.append(sqrtP_out.reshape(-1, x_size, D, D))
            
        return Mean, sqrtCov, WeightList

class NMGaussDensity():
    def __init__(self, m, sqrtP, Weight):
        self.m = m
        self.sqrtP = sqrtP
        self.Weight = Weight
        ''' m: torch.tensor of shape [number of Gauss, number of data, dimension]
            P: torch.tensor of shape [number of Gauss, number of data, dimension, dimension]
            Weight: torch.tensor of shape [1, number of Gauss]
        '''
    
    def density(self, x):
        '''x torch.tensor of shape [number of data, dimension]
            
        Limitation: Unable to calculate the value of a density function at multiple points
        '''  
        a1 = (
            -(x-self.m).unsqueeze(-2)@(torch.inverse(self.sqrtP)/2 @ (x-self.m).unsqueeze(-1))
            ).squeeze(dim=-1).squeeze(dim=-1)        
        '''(x-m)^T P^{-1}/2 (x-m)'''
        
        a2 = -torch.log(torch.det(self.sqrtP))
        # 
        SingleDensity = torch.exp(a1.squeeze(dim=-1).squeeze(dim=-1) + a2) / (2*3.1415926)**(x.shape[-1]/2) 
        density  = (self.Weight*SingleDensity).sum()

        return density

    def SafeLogDensity(self, x):
        a1 = (
            -(x-self.m).unsqueeze(-2)@(torch.cholesky_inverse(self.sqrtP)/2 @ (x-self.m).unsqueeze(-1))
            ).squeeze(dim=-1).squeeze(dim=-1)
        '''(x-m)^T P^{-1}/2 (x-m)'''
        
        a2 = -safe_log(torch.einsum('ijkk->ijk', self.sqrtP)).sum(dim=-1)
 
        a3 = torch.log(self.Weight).transpose(dim0=-1, dim1=-2)

        power = a1 + a2 + a3
        
        Max = power.max(dim=-2)[0]
        return Max - np.log(2*3.1415926)*(x.shape[-1]/2) + torch.log(torch.exp(power-Max).sum(dim=-2))
    
        # return - np.log(2*3.1415926)*(x.shape[-1]/2) + torch.log(torch.exp(power).sum(dim=-2))

#########################################################################   
#              Gaussian Cubature method
#########################################################################  
class GaussCubAppro():
    '''Gauss Assumed Density Approximation
    '''
    def __init__(self, f, df, sigma, N=1, D=3):

        self.f = f
        self.df = df
        self.sigma = sigma
        
        self.N=N
        self.D=D
        
        self.lambd = 1
        self.xi_i = np.sqrt(self.lambd + D)
        
        W = 1/2/(self.lambd + D)
        W0 = self.lambd/(self.lambd + D)
        self.W=W
        self.W0=W0
        
    def solver(self, x, H):
        '''
        x: np.ndarray or torch.Tensor of shape [dim] or [num, 1, dim].
        h: float
        '''
        D=self.D
        xi_i= np.sqrt(self.lambd + D)
        
        m = x + H / self.N * self.f(x)
        P = H / self.N * self.sigma(m)@self.sigma(m).transpose(dim0=-2, dim1=-1)
        
        for _ in range(self.N-1):
            sqrtP = torch.linalg.cholesky(P + eps_safe*torch.eye(m.shape[-1], dtype=m.dtype, device=m.device))

            f_a = self.f(m)*self.W0
            for d in range(D):
                f_a=f_a + self.W * self.f(m - xi_i * sqrtP[...,d, :]) + self.W * self.f(m + xi_i * sqrtP[...,d, :])

            L_a = self.sigma(m)@self.sigma(m).transpose(dim0=-2, dim1=-1)*self.W0
            for d in range(D):
                L_a=L_a + (
                  self.W * self.sigma(m - xi_i * sqrtP[...,d, :])@self.sigma(m - xi_i * sqrtP[...,d, :]).transpose(dim0=-2, dim1=-1)
                + self.W * self.sigma(m + xi_i * sqrtP[...,d, :])@self.sigma(m + xi_i * sqrtP[...,d, :]).transpose(dim0=-2, dim1=-1)
                )

            for d in range(D): 
                L_af=(
                 self.W * (-xi_i * sqrtP[...,d:d+1, :]).transpose(dim0=-2, dim1=-1) @ self.f(m - xi_i * sqrtP[...,d, :]).unsqueeze(dim=-2)
                +self.W * (xi_i * sqrtP[...,d:d+1, :]).transpose(dim0=-2, dim1=-1) @ self.f(m + xi_i * sqrtP[...,d, :]).unsqueeze(dim=-2)
                ) 
                L_a = L_a + L_af + L_af.transpose(dim0=-2, dim1=-1)
            m = m+ H / self.N * f_a
            P = P+ H / self.N * L_a
        return m, P
######################################################################       
    
    
    
    
    
    
    