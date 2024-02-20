import torch
import numpy as np


from EMTData import EMTData

import time

h=0.05
num_train_traj = 100000
num_test_traj=2
data = EMTData(h=h, steps = 3901, length=1, num_train_traj=num_train_traj, num_test_traj=num_test_traj) 


t=time.time()


N_traj=1000
steps=200000
ite=2
region= np.array([[0,2]]*10)
region[6] = [0,6]
x0 = []
for i in range(10):
    x0.append(np.random.uniform(region[i, 0],region[i, 1],(N_traj, 1)))
x0=np.hstack(x0)
print(x0.shape)
x0 = torch.tensor(x0, dtype = torch.float32)


local = 'outputs/EMT0.04_seed0/model_best.pkl'
net = torch.load(local, map_location=torch.device('cpu'))
with torch.no_grad():
    traj = net.predict(x0, h=0.05, steps=int(steps/10), returnnp=True)
print(traj.shape)        
np.save('EMTdata/ID_pre0.04traj_steps{}_num{}.npy'.format(steps, N_traj), traj)


local = 'outputs/EMT0.08_seed0/model_best.pkl'
net = torch.load(local, map_location=torch.device('cpu'))
with torch.no_grad():
    traj = net.predict(x0, h=0.05, steps=int(steps/10), returnnp=True)
print(traj.shape)        
np.save('EMTdata/ID_pre0.08traj_steps{}_num{}.npy'.format(steps, N_traj), traj)


local = 'outputs/EMT0.04_seed0EM/model_best.pkl'
net = torch.load(local, map_location=torch.device('cpu'))
with torch.no_grad():
    traj = net.predict(x0, h=0.05, steps=int(steps/10), returnnp=True)
print(traj.shape)        
np.save('EMTdata/ID_em0.04traj_steps{}_num{}.npy'.format(steps, N_traj), traj)

local = 'outputs/EMT0.08_seed0EM/model_best.pkl'
net = torch.load(local, map_location=torch.device('cpu'))
with torch.no_grad():
    traj = net.predict(x0, h=0.05, steps=int(steps/10), returnnp=True)
print(traj.shape)        
np.save('EMTdata/ID_em0.08traj_steps{}_num{}.npy'.format(steps, N_traj), traj)



print(time.time()-t)
with torch.no_grad():
    for i in range(ite):
        traj = data.solver.flow(x0, torch.tensor(0.005), int(steps/ite))[:, ::10, :]
        x0 = traj[:,-1,:]
        np.save('EMTdata/ID_truetraj_steps{}_num{}_time{}.npy'.format(steps, N_traj, i), traj)
 