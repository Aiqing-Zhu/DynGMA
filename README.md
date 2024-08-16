# DynGMA
DynGMA: a robust approach for learning stochastic differential equations from data

---

We demonstrate how to use this code by explaining the `LS.py` file. The `LS.py` file is designed for fixed data steps and state-dependent diffusions. For variable data steps, you can refer to the code in `SIR.py` and `SIRS.py`. For constant diffusions, the relevant code can be found in `TwoDimension.py` and `EMT.py`.

---
```
import torch

import dynlearner as ln
from LSData import LSData # to genetate data 

def main(seed=0):
    if torch.cuda.is_available():
        device = 'gpu'
        torch.cuda.set_device(2)
    else: 
        device ='cpu'
    # set device

    num_train_traj = 10000
    num_test_traj=1
    h=0.01
    #set parameter of data

    lr=0.001
    #set parameter of optimizer

    data = LSData(h=h, steps = 200, length=1, num_train_traj=num_train_traj, num_test_traj=num_test_traj)
    #generate data

    net = ln.nn.NMGaussSDENet(dim=3, timestep=h, layers=3, width=128, activation='tanh', st=h/2,
                            sigmalayers=2, sigmawidth=50)
    # Initialize NN
    # There are also some other NN. For example,
    # net = ln.nn.EMSDENet(dim=3,  timestep=h, layers=3, width=128, linear_sigma=False, sigmalayers=2, sigmawidth=50)
    # net = ln.nn.GaussCubSDENet(dim=3,  N=2, timestep=h, layers=3, width=128, linear_sigma=False, sigmalayers=2, sigmawidth=50)
                          
                            
 
    arguments = {
        'filename': 'LS_28_seed{}'.format(seed),
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'lr_decay': 1,
        'iterations': 100000,
        'batch_size': 100000,
        'print_every': 1000,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }


    ln.Brain.Init(**arguments)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output() 
    #train and restore NN, the path of the trained NN is './outputs/filename/model_best.pkl'.
```
    
