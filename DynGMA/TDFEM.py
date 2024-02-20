import numpy as np
import matplotlib.pyplot as plt
import time

import math
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

def get_prob_flux_sparse(f,eps,D,xrange,yrange,Nx,Ny,px=[0,0]):
    Lx = xrange[1]-xrange[0]
    Ly = yrange[1]-yrange[0]
    N  = Nx*Ny
    hx = Lx/Nx
    hy = Ly/Ny
    D11,D22 = D[0][0],D[1][1]
    def idx(i,j,Nx=Nx,Ny=Ny): return (i-1)*Ny+j-1
    def pos(i,j,hx=hx,hy=hy): return np.array([xrange[0]+i*hx,yrange[0]+j*hy])
    row = []
    col = []
    data = []
    
    # construct the system
    for i in range(1,Nx+1):
        for j in range(1,Ny+1):
            # J1(i,j-1/2)/hx
            ff = f(np.vstack([pos(i,j-1/2),pos(i-1,j-1/2),pos(i-1/2,j),pos(i-1/2,j-1)]))
            if i<Nx:
                row += [idx(i,j),idx(i,j)]
                col += [idx(i,  j),idx(i+1,j)]
                data += [(-.5*ff[0][0]-eps*D11/hx)/hx,
                         (-.5*ff[0][0]+eps*D11/hx)/hx]

            # -J1(i-1,j-1/2)/hx
            if i>1:
                row += [idx(i,j),idx(i,j)]
                col += [idx(i-1,j),idx(i  ,j)]
                data += [-(-.5*ff[1][0]-eps*D11/hx)/hx,
                         -(-.5*ff[1][0]+eps*D11/hx)/hx]

            # J2(i-1/2,j)/hy
            if j<Ny:
                row += [idx(i,j),idx(i,j)]
                col += [idx(i,j),idx(i,j+1)]
                data += [(-.5*ff[2][1]-eps*D22/hy)/hy,
                         (-.5*ff[2][1]+eps*D22/hy)/hy]

            # -J2(i-1/2,j-1)/hy
            if j>1:
                row += [idx(i,j),idx(i,j)]
                col += [idx(i,j-1),idx(i,j)]
                data += [-(-.5*ff[3][1]-eps*D22/hy)/hy,
                         -(-.5*ff[3][1]+eps*D22/hy)/hy]
    A          = csc_matrix( (data,(row,col)) )
    print(A.astype)

    # solve the system
    idx_       = idx(np.int_((px[0]-xrange[0])/hx+.5),np.int_((px[1]-yrange[0])/hy+.5))
    ei         = np.zeros(dtype=np.float64,shape=(N,1))
    ei[idx_,0] = 1
    mask       = np.reshape(ei==1,-1)
    ei         = csc_matrix(ei, dtype=np.float64)
    b          = -A@ei
    mask       = ~mask
    A_         = A[:,mask]
    prob_      = spsolve(A_[:-1],b[:-1])
    prob       = np.insert(prob_,idx_,np.array([1]),0)

    # normalization
    prob       = np.maximum(prob,0)
    Z          = prob.mean()*Lx*Ly
    prob       = prob/Z
    return (np.linspace(xrange[0],xrange[1],Nx+1)[:-1]+np.linspace(xrange[0],xrange[1],Nx+1)[1:])/2,\
           (np.linspace(yrange[0],yrange[1],Ny+1)[:-1]+np.linspace(yrange[0],yrange[1],Ny+1)[1:])/2,\
           np.transpose(prob.reshape(Nx,Ny))
def plot_epslog_prob(prob,eps,xx,yy,xv,yv,overeps=0,Vmax=2):
    XX,YY  = np.meshgrid(xx,yy)
    x_list = np.concatenate([XX[:,:,None],YY[:,:,None]],axis=-1).reshape(-1,2)
    fig,ax = plt.subplots(1,3,figsize=(20,5),constrained_layout=True)
    if overeps==0: V = -np.log(prob+1e-15)*eps
    else: V = -np.log(prob+1e-15)
    V      = np.minimum(V-V.min(),Vmax)
    c      = ax[0].contourf(XX,YY,V,20,cmap='terrain')
    cbar   = fig.colorbar(c,ax=ax[0],format='%.2f',aspect=50)
    cbar.ax.tick_params(labelsize=18)
    if overeps==0: title = '$-\epsilon log(P)$'
    else: title = '$-log(P)$'
    ax[0].set_title(title +' with $\epsilon$='+str(eps),fontsize=20)
    ax[0].set_xlabel('x',fontsize=20)
    ax[0].set_ylabel('y',fontsize=20)
    
    idx = np.argmin(np.abs(xx-xv))
    ax[1].plot(yy,V[:,idx],'r-')
    points = np.zeros(dtype=np.float64,shape=(yy.shape[0],2))
    points[:,0] = xv
    points[:,1] = yy
    ax[1].set_title(title +' with x='+str(xv),fontsize=20)
    ax[1].set_xlabel('y',fontsize=20)
    
    idx = np.argmin(np.abs(yy-yv))
    ax[2].plot(xx,V[idx,:],'r-')
    points = np.zeros(dtype=np.float64,shape=(xx.shape[0],2))
    points[:,0] = xx
    points[:,1] = yv
    ax[2].set_title(title +' with y='+str(yv),fontsize=20)
    ax[2].set_xlabel('x',fontsize=20)
    for ax_ in ax: ax_.tick_params(axis="both", labelsize=20)
    plt.show()

# System
def f_ref(X): 
    if np.size(X.shape)==2: x,y = X[:,0][:,None],X[:,1][:,None]
    else: x,y = X[0],X[1]
    return np.hstack([.1*2*x*(1-x**2) + (1+np.sin(x))*y,
                      -y + 2*x*(1+np.sin(x))*(1-x**2)])


if __name__=='__main__':
    dim    = 2
    sigma  = np.diag([np.sqrt(1./50),np.sqrt(1./5)])
    D      = sigma@np.transpose(sigma)
    eps    = np.linalg.norm(D,2)/2
    Dbar   = D/eps/2

    Xrange = np.array([[-2,2],[-3,3]])


    #FDM
    Nx,Ny          = 2000,2000
    xrange, yrange = Xrange[0],Xrange[1]
    eps1,eps2,eps3 = eps/2,eps,eps*2

    xx,yy,prob1    = get_prob_flux_sparse(f_ref,eps1,Dbar,xrange,yrange,Nx,Ny,px=[-1,0])
    xx,yy,prob2    = get_prob_flux_sparse(f_ref,eps2,Dbar,xrange,yrange,Nx,Ny,px=[-1,0])
    xx,yy,prob3    = get_prob_flux_sparse(f_ref,eps3,Dbar,xrange,yrange,Nx,Ny,px=[-1,0]) 
    XX,YY          = np.meshgrid(xx,yy)
    FD_X           = np.concatenate([XX[:,:,None],YY[:,:,None]],axis=-1).reshape(-1,2)
    FD_Y           = [prob1.reshape(-1),prob2.reshape(-1),prob3.reshape(-1)]

    def get_P_FD(X,D,FD_X=FD_X,FD_Y=FD_Y): 
        # X shape: Any*dim
        if np.abs(D-2*eps1*Dbar).max()<1e-5: return np.maximum(griddata(FD_X,FD_Y[0],X,method='cubic',fill_value=0),0)
        if np.abs(D-2*eps2*Dbar).max()<1e-5: return np.maximum(griddata(FD_X,FD_Y[1],X,method='cubic',fill_value=0),0)
        if np.abs(D-2*eps3*Dbar).max()<1e-5: return np.maximum(griddata(FD_X,FD_Y[2],X,method='cubic',fill_value=0),0)
    def get_V_FD(X,D): return -np.linalg.norm(D,2)/2*np.log(get_P_FD(X,D)+1e-40)

    # Show FD results
    plot_epslog_prob(prob1,eps1,xx,yy,0,0,Vmax=eps1*15)
    plot_epslog_prob(prob2,eps2,xx,yy,0,0,Vmax=eps2*15)
    plot_epslog_prob(prob3,eps3,xx,yy,0,0,Vmax=eps3*15)
    print('sss')