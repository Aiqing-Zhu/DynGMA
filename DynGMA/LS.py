import torch
import argparse
import warnings
warnings.filterwarnings("ignore")

import dynlearner as ln
from LSData import LSData

def main(seed=0):
    if torch.cuda.is_available():
        device = 'gpu'
        torch.cuda.set_device(2)
    else: 
        device ='cpu'
    print(device)
    Nlayers =3
    Nwidth =128
    Nactivation = 'tanh'

    num_train_traj = 10000
    num_test_traj=1

    h=0.01

    lr=0.001

    data = LSData(h=h, steps = 200, length=1, num_train_traj=num_train_traj, num_test_traj=num_test_traj) 

    print(data.X_train.shape, data.y_train.shape)
    net = ln.nn.NMGaussSDENet(dim=3, timestep=h, layers=Nlayers, width=128, activation='tanh', st=h/2,
                            sigmalayers=2, sigmawidth=50)
 
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

    net = ln.nn.EMSDENet(dim=3,  timestep=h, layers=Nlayers, width=128, 
                 linear_sigma=False, sigmalayers=2, sigmawidth=50)
    arguments = {
        'filename': 'LSem_28_seed{}'.format(seed),
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
    
    lr=0.0001
    net = ln.nn.GaussCubSDENet(dim=3,  N=2, timestep=h, layers=Nlayers, width=128, 
                 linear_sigma=False, sigmalayers=2, sigmawidth=50)
    arguments = {
        'filename': 'LScub_28_seed{}'.format(seed),
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
    
    return 0

if __name__ == '__main__':
    for i in range(5):
        main(i) 
