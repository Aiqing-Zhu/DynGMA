import torch  

import dynlearner as ln
from TwoDimensionData import TwoDimensionData

 
def main_ID():  
    j=0
    for i in range(5):
        for h in [0.05, 0.1,0.2]:
            local = 'h={}seed={}MG'.format(h, i)
            net = torch.load('outputs/' + local + '/model_best.pkl', map_location=torch.device('cuda:{}'.format(j)))  
            run_ID(local+'_PINN', net, h, deviceindex=j)

            local = 'h={}seed={}EM'.format(h, i)
            net = torch.load('outputs/' + local + '/model_best.pkl', map_location=torch.device('cuda:{}'.format(j)))  
            run_ID(local+'_PINN', net, h, deviceindex=j)

def main(): 
    j=3
    for i in range(5):
        run(h=0.05, length=1, st=0.1, sN=1, deviceindex=j, seed=i)
        run(h=0.1, length=1, st=0.1, sN=2, deviceindex=j, seed=i)
        run(h=0.2, length=1, st=0.1, sN=2, deviceindex=j, seed=i)

        

    
def run(h=0.05, length=1,
       st=0.1, sN=1,
       deviceindex=0, seed=0):
    filename = 'h={}'.format(h) + 'seed={}'.format(seed)
    
    if torch.cuda.is_available():
        device = 'gpu'
        torch.cuda.set_device(deviceindex)
    else: 
        device ='cpu'
    print(device)
    Nlayers =3
    Nwidth =128
    Nactivation = 'tanh'


    num_train_traj=2000*int(h/0.05)
    num_test_traj=200*int(h/0.05)

    lr=0.01
    lr_decay = 100
    iterations = 5000
    batch_size = None
    print_every = 500

    data = TwoDimensionData(h=h, steps = int(1/h), length=length, 
                            num_train_traj=num_train_traj, num_test_traj=num_test_traj) 

    print(data.X_train.shape, data.y_train.shape, filename+'MG')
    net = ln.nn.MGaussSDENet(dim=2, timestep=h, layers=Nlayers, width=Nwidth, activation=Nactivation, st=st, sN=sN)
    
    arguments = {
        'filename': filename+'MG',
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'lr_decay': lr_decay,
        'iterations': iterations,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }


    ln.Brain.Init(**arguments)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    
    print(filename+'EM')
    net = ln.nn.EMSDENet(dim=2, timestep=h, layers=Nlayers, width=Nwidth, activation=Nactivation)
    arguments = {
        'filename': filename+'EM',
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'lr_decay': lr_decay,
        'iterations': iterations,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }


    ln.Brain.Init(**arguments)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output() 

    
def run_ID(filename, Net, h, deviceindex=0):
    
    if torch.cuda.is_available():
        device = 'gpu'
        torch.cuda.set_device(deviceindex)
    else: 
        device ='cpu'
    print(device)
    Nlayers =3
    Nwidth =128
    Nactivation = 'tanh'

    num_train_traj=2000*int(h/0.05)
    num_test_traj=200*int(h/0.05)

    lr=0.01
 
    data = TwoDimensionData(h=h, steps = int(1/h)+1, length=1, num_train_traj=num_train_traj, num_test_traj=num_test_traj) 
    
    for param in Net.parameters():
        param.requires_grad = False
    f = Net.vf
    A = Net.modus['sigma'].sigma
    D = A@ A.transpose(dim0=-2, dim1=-1)
    eps    = torch.norm(D,2)/2
    D_bar = (D/eps/2)
    print(eps, D_bar)

    
    print(data.X_train.shape)
    net = ln.nn.ID_NN(f, D_bar, eps, dim=2, layers=3, width=128, activation='tanh')
    print(net)
    arguments = {
        'filename': filename,
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'lr_decay': 100,
        'iterations': 50000,
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
     

if __name__ == '__main__':
    
    main()
    main_ID()