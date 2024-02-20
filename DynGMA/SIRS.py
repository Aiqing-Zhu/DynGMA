import torch
import argparse
import warnings
warnings.filterwarnings("ignore")
import argparse


import dynlearner as ln
from SIRSData import kmcData

def main(seed=0, time_step=0.1, deviceindex = 0, init_traj=False):
    if torch.cuda.is_available():
        device = 'gpu'
        torch.cuda.set_device(deviceindex)
    else: 
        device ='cpu'
    print(device)
    Nlayers =2
    Nwidth =64
    Nactivation = 'tanh' 
    lr=0.001 
    data = kmcData(num_traj=12500, time_max=1, time_step=time_step, init_traj=init_traj)
    
    print(data.X_train.shape, data.y_train.shape)
    print(data.y_train[0,:,0].max())
    
    steps=2 if time_step==0.2 else 1
    print(steps)
    net = ln.nn.NMGaussSDENet_V(dim=2,layers=Nlayers, width=Nwidth, activation='tanh', steps=steps,
                            sigmalayers=2, sigmawidth=64)
    arguments = {
        'filename': 'kmc{}_seed{}'.format(time_step, seed),
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'lr_decay': 1,
        'iterations': 20000,
        'batch_size': None,
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
    
    
    net = ln.nn.EMSDENet(dim=2,
                 layers=2, width=64, activation='tanh', initializer='orthogonal', 
                 linear_sigma=False, sigmalayers=2, sigmawidth=64)
    arguments = {
        'filename': 'kmc{}em_seed{}'.format(time_step, seed),
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'lr_decay': 1,
        'iterations': 20000,
        'batch_size': None,
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
    parser = argparse.ArgumentParser() 
    parser.add_argument('--s',type=int, default=0)
    parser.add_argument('--device',type=int, default=2) 

    args = parser.parse_args()
    
    main(seed=args.s, time_step=0.1, deviceindex = args.device, init_traj=True)
    main(seed=args.s, time_step=0.05, deviceindex = args.device, init_traj=True)
    main(seed=args.s, time_step=0.2, deviceindex = args.device, init_traj=True)
