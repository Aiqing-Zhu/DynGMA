import torch
import argparse
import warnings
warnings.filterwarnings("ignore")

import dynlearner as ln
from EMTData import EMTData


def main(h=0.005, seed=0):
    if torch.cuda.is_available():
        device = 'gpu'
        torch.cuda.set_device(1)
    else: 
        device ='cpu'
    print(device)
    Nlayers =3
    Nwidth =128
    Nactivation = 'tanh'

    num_train_traj = 100000
    num_test_traj=50

    lr=0.01
    lr_decay = 100
    filename = 'testEMT{}_seed{}'.format(h, seed) 

    data = EMTData(h=h, steps = 3901, length=1, num_train_traj=num_train_traj, num_test_traj=num_test_traj) 

    print(data.X_train.shape, data.y_train.shape) 
    net = ln.nn.MGaussSDENet(dim=10, timestep=h, layers=Nlayers, width=Nwidth, activation='tanh', st=h, sN =2)

    
    arguments = {
        'filename': filename,
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'lr_decay': lr_decay,
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


    net = ln.nn.EMSDENet(dim=10, timestep=h, layers=Nlayers, width=Nwidth, activation=Nactivation)
    arguments = {
        'filename': filename+'EM',
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'lr_decay': lr_decay,
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
    
main(h=0.08, seed=0)

###
# 0.02 sn=1, 0.04, sn =2 