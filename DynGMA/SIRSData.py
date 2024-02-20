# The following code segment is obtained by modified the code in 
# https://gitlab.com/felix.dietrich/sde-identification/-/tree/master/gillespie?ref_type=heads
# Original author: Felix Dietrich
# Original project: sde-identification

import torch
import numpy as np

import dynlearner as ln


class kmcData(ln.Data):
    def __init__(self, time_max=4, time_step=0.05, num_traj=2500, init_traj=False):
        super(kmcData, self).__init__()     
        self.time_max= time_max
        self.time_step = time_step
        self.n_skip_steps = int(time_step/0.001)
        if init_traj:
            self.__init_traj(num_traj)
        self.__init_data()
    
    @property
    def dim(self):
        return 2
    

    def __init_traj(self, N_traj):
        sig = SIRG(N = 1024)
        rng = np.random.default_rng(1)
        y0_all = []
        for k in range(N_traj):
            # randomly sample the unit cube
            y0 = rng.uniform(low=0.0, high=1.0, size=(3,))

            # # transform to sample more points on the boundary
            # y0 = (np.tanh((y0-.5)*5)+1)/2

            # make sure we only sample admissible initial conditions
            y0 = np.clip(y0, 0, 1)
            y0 = y0/np.sum(y0)

            # only take the first two
            y0 = y0[:2] 
            y0_all.append(y0)
        traj=[]
        length=[]
        X=[]
        Y=[]
        for i in range(N_traj):
            a = sig.simulate_single(y0=y0_all[i], time_max=self.time_max, time_step=self.time_step/self.n_skip_steps, rng=None)
            # a[:, 1] = 1-a[:, 1]-a[:, 2]
            # print(a.shape)
            
            a_s = a[::self.n_skip_steps]
            if len(a_s)>2:
                traj.append(a_s)
                length.append(len(a_s))
                X.append(a_s[:-1])
                Y.append(a_s[1:])
            # else:
            #     print(i, 'length<=2', len(a), a.shape)

        traj = np.vstack(traj)
        length=np.array(length)
        X = np.vstack(X)
        Y = np.vstack(Y)
    
        np.save('X0{}'.format(self.time_step), np.hstack(y0_all))
        np.save('X{}'.format(self.time_step), X)
        np.save('Y{}'.format(self.time_step), Y)
        np.save('traj{}'.format(self.time_step), traj)
        np.save('length{}'.format(self.time_step), length)
   
        
    def __init_data(self):
        X = np.load('X{}.npy'.format(self.time_step))
        Y = np.load('Y{}.npy'.format(self.time_step))
        Y[:,0]=Y[:,0] - X[:,0]
        # print(X[Y[:,0]>0.005].shape,X.shape)
        # X = X[Y[:,0]>0.005]
        # Y = Y[Y[:,0]>0.005]
        
         
        np.random.seed(42)
        random_indices = np.random.permutation(len(X))
        split_point = int(0.8 * len(X))
        
        self.X_train = X[random_indices[:split_point], 1:]
        self.X_test = X[random_indices[split_point:], 1:]

        self.y_train = (Y[random_indices[:split_point], :])[np.newaxis, :]
        self.y_test = (Y[random_indices[split_point:], :])[np.newaxis, :]

class SIRG:
    """
    Python implementation of the SIR Gillespie simulation.
    """
    
    def __init__(self, N = 1024, k1 = 1.0, k2 = 1.0, k3 = 0.5, random_state = 1):
        self.N = N       # system size; number of sites
        self.k1 = k1       # rate constant of event 1
        self.k2 = k2       # rate constant of event 2
        self.k3 = k3       # rate constant of event 3
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        
    def simulate_single(self, y0, time_max=4, time_step=0.01, rng=None):
        """
        y0[0]     initial condition for y1 (I, infected species)
        y0[1]     initial condition for y2 (R, recovered species)
        time_max  max time
        tstep     output dt
        """
        
        if rng is None:
            rng = self.rng
        
        # initialize internal parameters
        k1 = 4.0*self.k1
        k2 = self.k2
        k3 = self.k3
        
        CN = 1.0 / self.N
        curtime = 0.0
        NP = int(np.ceil(time_max/time_step))
        time = np.zeros((NP,))
        y = np.zeros((NP,2))
        R = np.zeros((3,))
        N1 = int(y0[0]*self.N)
        N2 = int(y0[1]*self.N)
        i = 0
        
        while (curtime <= time_max):
    
            # calculate all rates and their sum
            y1 = np.clip(N1 * CN, 0, 1);    # I concentration
            y2 = np.clip(N2 * CN, 0, 1);    # R concentration
            R[0] = k1*y1*(1-y1-y2);  # I + S --> I + I 
            R[1] = k2*y1;            # I --> R
            R[2] = k3*y2;            # R --> S
            RSum = np.sum(R) #R[0]+R[1]+R[2]
            
            if RSum == 0: # happens if y1 is zero
                break
            if i >= NP:
                break

            # call RNG (0,1)
            x = rng.uniform(0,1)*RSum;

            # select one elementary event
            RA = R[0]
            Act = 0 # python is zero based...
            while (RA < x and Act < len(R)-1):
                Act = Act + 1
                RA = RA + R[Act]

            # update N's according to the selected event 
            # Python does not have a switch/case keyword structure
            if Act==0:
                    N1 = N1 + 1
            if Act==1:
                    N1 = N1 - 1
                    N2 = N2 + 1
            if Act==2:
                    N2 = N2 - 1

            # update time (clip the argument to be on the safe side)
            if RSum == 0:
                print("RSum error: is zero...")
                RSum = 1
            dt = -np.log(rng.uniform(1e-10,1))/(RSum*self.N)
            # dt = 1.0/(RSum*self.N);
            curtime = curtime + dt

            # save solution
            if (curtime >= time_step*i):
                time[i] = curtime
                y[i, 0] = y1
                y[i, 1] = y2
                i = i + 1
                
        time = time[:i]
        y = y[:i,:]
        return np.column_stack([time, y])