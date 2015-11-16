import numpy as np
import fem.asm
import scipy.sparse.linalg
import scipy.sparse as spsp
import matplotlib.pyplot as plt
import fem.geometry as fegeom

geomlist=[
        ('+','box',0,0,0.2,2.0)
        ]
        
g=fegeom.GeometryShapelyTriangle2D(geomlist)
mesh=g.mesh(0.05)

mesh.plot()

plt.axis('equal')
plt.show()

N=mesh.p.shape[1]
DUx=np.nonzero(mesh.p[1,:]==2)[0]
DBx=np.nonzero(mesh.p[1,:]==0)[0]
DUy=DUx+N
DBy=DBx+N
Dx=np.union1d(DUx,DBx)
Dy=Dx+N
D=np.union1d(Dx,Dy)
I=np.setdiff1d(np.arange(0,2*N),D)

I1=np.arange(0,N)
I2=I1+N

a=fem.asm.AssemblerTriP1(mesh)

lam=1.
mu=1.

def dv2(dv,dw1,dw2):
    return lam*dv[1]*dw1[0] + (lam*dv[1]*dw1[0]**2)/2. + (mu*dv[0]*dw1[1])/2. + (mu*dv[0]*dw1[0]*dw1[1])/2. + (lam*dv[1]*dw1[1]**2)/2. + (mu*dv[1]*dw1[1]**2)/2. + (mu*dv[0]*dw2[0])/2. + lam*dv[0]*dw1[0]*dw2[0] + mu*dv[0]*dw1[0]*dw2[0] + (lam*dv[0]*dw1[0]**2*dw2[0])/2. + (mu*dv[0]*dw1[0]**2*dw2[0])/2. + (mu*dv[1]*dw1[1]*dw2[0])/2. + (mu*dv[1]*dw1[0]*dw1[1]*dw2[0])/2. + (lam*dv[0]*dw1[1]**2*dw2[0])/2. + (lam*dv[1]*dw2[0]**2)/2. + (mu*dv[1]*dw2[0]**2)/2. + (lam*dv[0]*dw2[0]**3)/2. + (mu*dv[0]*dw2[0]**3)/2. + lam*dv[1]*dw2[1] + mu*dv[1]*dw2[1] + lam*dv[1]*dw1[0]*dw2[1] + (lam*dv[1]*dw1[0]**2*dw2[1])/2. + (mu*dv[0]*dw1[1]*dw2[1])/2. + (mu*dv[0]*dw1[0]*dw1[1]*dw2[1])/2. + (lam*dv[1]*dw1[1]**2*dw2[1])/2. + (mu*dv[1]*dw1[1]**2*dw2[1])/2. + lam*dv[0]*dw2[0]*dw2[1] + mu*dv[0]*dw2[0]*dw2[1] + (lam*dv[1]*dw2[0]**2*dw2[1])/2. + (mu*dv[1]*dw2[0]**2*dw2[1])/2. + (3*lam*dv[1]*dw2[1]**2)/2. + (3*mu*dv[1]*dw2[1]**2)/2. + (lam*dv[0]*dw2[0]*dw2[1]**2)/2. + (mu*dv[0]*dw2[0]*dw2[1]**2)/2. + (lam*dv[1]*dw2[1]**3)/2. + (mu*dv[1]*dw2[1]**3)/2.

def dv1(dv,dw1,dw2):
    return lam*dv[0]*dw1[0] + mu*dv[0]*dw1[0] + (3*lam*dv[0]*dw1[0]**2)/2. + (3*mu*dv[0]*dw1[0]**2)/2. + (lam*dv[0]*dw1[0]**3)/2. + (mu*dv[0]*dw1[0]**3)/2. + (mu*dv[1]*dw1[1])/2. + lam*dv[1]*dw1[0]*dw1[1] + mu*dv[1]*dw1[0]*dw1[1] + (lam*dv[1]*dw1[0]**2*dw1[1])/2. + (mu*dv[1]*dw1[0]**2*dw1[1])/2. + (lam*dv[0]*dw1[1]**2)/2. + (mu*dv[0]*dw1[1]**2)/2. + (lam*dv[0]*dw1[0]*dw1[1]**2)/2. + (mu*dv[0]*dw1[0]*dw1[1]**2)/2. + (lam*dv[1]*dw1[1]**3)/2. + (mu*dv[1]*dw1[1]**3)/2. + (mu*dv[1]*dw2[0])/2. + (mu*dv[1]*dw1[0]*dw2[0])/2. + (mu*dv[0]*dw1[1]*dw2[0])/2. + (lam*dv[0]*dw2[0]**2)/2. + (mu*dv[0]*dw2[0]**2)/2. + (lam*dv[0]*dw1[0]*dw2[0]**2)/2. + (mu*dv[0]*dw1[0]*dw2[0]**2)/2. + (lam*dv[1]*dw1[1]*dw2[0]**2)/2. + lam*dv[0]*dw2[1] + lam*dv[0]*dw1[0]*dw2[1] + lam*dv[1]*dw1[1]*dw2[1] + mu*dv[1]*dw1[1]*dw2[1] + (mu*dv[1]*dw2[0]*dw2[1])/2. + (mu*dv[1]*dw1[0]*dw2[0]*dw2[1])/2. + (mu*dv[0]*dw1[1]*dw2[0]*dw2[1])/2. + (lam*dv[0]*dw2[1]**2)/2. + (lam*dv[0]*dw1[0]*dw2[1]**2)/2. + (lam*dv[1]*dw1[1]*dw2[1]**2)/2. + (mu*dv[1]*dw1[1]*dw2[1]**2)/2.
    

def dudv11(du,dv,dw1,dw2):
    return lam*du[0]*dv[0] + mu*du[0]*dv[0] + (mu*du[1]*dv[1])/2. + 3*lam*du[0]*dv[0]*dw1[0] + 3*mu*du[0]*dv[0]*dw1[0] + lam*du[1]*dv[1]*dw1[0] + mu*du[1]*dv[1]*dw1[0] + (3*lam*du[0]*dv[0]*dw1[0]**2)/2. + (3*mu*du[0]*dv[0]*dw1[0]**2)/2. + (lam*du[1]*dv[1]*dw1[0]**2)/2. + (mu*du[1]*dv[1]*dw1[0]**2)/2. + lam*du[1]*dv[0]*dw1[1] + mu*du[1]*dv[0]*dw1[1] + lam*du[0]*dv[1]*dw1[1] + mu*du[0]*dv[1]*dw1[1] + lam*du[1]*dv[0]*dw1[0]*dw1[1] + mu*du[1]*dv[0]*dw1[0]*dw1[1] + lam*du[0]*dv[1]*dw1[0]*dw1[1] + mu*du[0]*dv[1]*dw1[0]*dw1[1] + (lam*du[0]*dv[0]*dw1[1]**2)/2. + (mu*du[0]*dv[0]*dw1[1]**2)/2. + (3*lam*du[1]*dv[1]*dw1[1]**2)/2. + (3*mu*du[1]*dv[1]*dw1[1]**2)/2. + (mu*du[1]*dv[0]*dw2[0])/2. + (mu*du[0]*dv[1]*dw2[0])/2. + (lam*du[0]*dv[0]*dw2[0]**2)/2. + (mu*du[0]*dv[0]*dw2[0]**2)/2. + (lam*du[1]*dv[1]*dw2[0]**2)/2. + lam*du[0]*dv[0]*dw2[1] + lam*du[1]*dv[1]*dw2[1] + mu*du[1]*dv[1]*dw2[1] + (mu*du[1]*dv[0]*dw2[0]*dw2[1])/2. + (mu*du[0]*dv[1]*dw2[0]*dw2[1])/2. + (lam*du[0]*dv[0]*dw2[1]**2)/2. + (lam*du[1]*dv[1]*dw2[1]**2)/2. + (mu*du[1]*dv[1]*dw2[1]**2)/2.

def dudv12(du,dv,dw1,dw2):
    return lam*du[1]*dv[0] + (mu*du[0]*dv[1])/2. + lam*du[1]*dv[0]*dw1[0] + (mu*du[0]*dv[1]*dw1[0])/2. + (mu*du[0]*dv[0]*dw1[1])/2. + lam*du[1]*dv[1]*dw1[1] + mu*du[1]*dv[1]*dw1[1] + lam*du[0]*dv[0]*dw2[0] + mu*du[0]*dv[0]*dw2[0] + (mu*du[1]*dv[1]*dw2[0])/2. + lam*du[0]*dv[0]*dw1[0]*dw2[0] + mu*du[0]*dv[0]*dw1[0]*dw2[0] + (mu*du[1]*dv[1]*dw1[0]*dw2[0])/2. + (mu*du[1]*dv[0]*dw1[1]*dw2[0])/2. + lam*du[0]*dv[1]*dw1[1]*dw2[0] + lam*du[1]*dv[0]*dw2[1] + (mu*du[0]*dv[1]*dw2[1])/2. + lam*du[1]*dv[0]*dw1[0]*dw2[1] + (mu*du[0]*dv[1]*dw1[0]*dw2[1])/2. + (mu*du[0]*dv[0]*dw1[1]*dw2[1])/2. + lam*du[1]*dv[1]*dw1[1]*dw2[1] + mu*du[1]*dv[1]*dw1[1]*dw2[1]

def dudv21(du,dv,dw1,dw2):
    return (mu*du[1]*dv[0])/2. + lam*du[0]*dv[1] + (mu*du[1]*dv[0]*dw1[0])/2. + lam*du[0]*dv[1]*dw1[0] + (mu*du[0]*dv[0]*dw1[1])/2. + lam*du[1]*dv[1]*dw1[1] + mu*du[1]*dv[1]*dw1[1] + lam*du[0]*dv[0]*dw2[0] + mu*du[0]*dv[0]*dw2[0] + (mu*du[1]*dv[1]*dw2[0])/2. + lam*du[0]*dv[0]*dw1[0]*dw2[0] + mu*du[0]*dv[0]*dw1[0]*dw2[0] + (mu*du[1]*dv[1]*dw1[0]*dw2[0])/2. + lam*du[1]*dv[0]*dw1[1]*dw2[0] + (mu*du[0]*dv[1]*dw1[1]*dw2[0])/2. + (mu*du[1]*dv[0]*dw2[1])/2. + lam*du[0]*dv[1]*dw2[1] + (mu*du[1]*dv[0]*dw1[0]*dw2[1])/2. + lam*du[0]*dv[1]*dw1[0]*dw2[1] + (mu*du[0]*dv[0]*dw1[1]*dw2[1])/2. + lam*du[1]*dv[1]*dw1[1]*dw2[1] + mu*du[1]*dv[1]*dw1[1]*dw2[1]

def dudv22(du,dv,dw1,dw2):
    return (mu*du[0]*dv[0])/2. + lam*du[1]*dv[1] + mu*du[1]*dv[1] + lam*du[0]*dv[0]*dw1[0] + mu*du[0]*dv[0]*dw1[0] + lam*du[1]*dv[1]*dw1[0] + (lam*du[0]*dv[0]*dw1[0]**2)/2. + (mu*du[0]*dv[0]*dw1[0]**2)/2. + (lam*du[1]*dv[1]*dw1[0]**2)/2. + (mu*du[1]*dv[0]*dw1[1])/2. + (mu*du[0]*dv[1]*dw1[1])/2. + (mu*du[1]*dv[0]*dw1[0]*dw1[1])/2. + (mu*du[0]*dv[1]*dw1[0]*dw1[1])/2. + (lam*du[0]*dv[0]*dw1[1]**2)/2. + (lam*du[1]*dv[1]*dw1[1]**2)/2. + (mu*du[1]*dv[1]*dw1[1]**2)/2. + lam*du[1]*dv[0]*dw2[0] + mu*du[1]*dv[0]*dw2[0] + lam*du[0]*dv[1]*dw2[0] + mu*du[0]*dv[1]*dw2[0] + (3*lam*du[0]*dv[0]*dw2[0]**2)/2. + (3*mu*du[0]*dv[0]*dw2[0]**2)/2. + (lam*du[1]*dv[1]*dw2[0]**2)/2. + (mu*du[1]*dv[1]*dw2[0]**2)/2. + lam*du[0]*dv[0]*dw2[1] + mu*du[0]*dv[0]*dw2[1] + 3*lam*du[1]*dv[1]*dw2[1] + 3*mu*du[1]*dv[1]*dw2[1] + lam*du[1]*dv[0]*dw2[0]*dw2[1] + mu*du[1]*dv[0]*dw2[0]*dw2[1] + lam*du[0]*dv[1]*dw2[0]*dw2[1] + mu*du[0]*dv[1]*dw2[0]*dw2[1] + (lam*du[0]*dv[0]*dw2[1]**2)/2. + (mu*du[0]*dv[0]*dw2[1]**2)/2. + (3*lam*du[1]*dv[1]*dw2[1]**2)/2. + (3*mu*du[1]*dv[1]*dw2[1]**2)/2. 

u=np.zeros(2*N)
#u[DUy]=-0.1
#u[DUx]=1.0

alpha=0.3

# continuation
for ctr in np.arange(0,1.1,0.1):
    bcdispy=-ctr*0.075
    print "upper face displacement: "+str(bcdispy)
    # newton
    for itr in range(100):
        u[DUy]=bcdispy
        
        K11=a.iasm(dudv11,w1=u[I1],w2=u[I2],intorder=4)
        K12=a.iasm(dudv12,w1=u[I1],w2=u[I2],intorder=4)
        K21=a.iasm(dudv21,w1=u[I1],w2=u[I2],intorder=4)
        K22=a.iasm(dudv22,w1=u[I1],w2=u[I2],intorder=4)
        K=spsp.vstack((spsp.hstack((K11,K12)),spsp.hstack((K21,K22)))).tocsr()
        
        f1=a.iasm(dv1,w1=u[I1],w2=u[I2],intorder=4)
        f2=a.iasm(dv2,w1=u[I1],w2=u[I2],intorder=4)
        f=np.hstack((f1,f2))
        
        U=np.copy(u)
        u[I]=u[I]+scipy.sparse.linalg.spsolve(K[np.ix_(I,I)],-f[I])
        u[I]=alpha*u[I]+(1-alpha)*U[I]
        
        if np.linalg.norm(u-U)<=1e-5:
            break
        print np.linalg.norm(u-U)

sf=2.
mesh.p[0,:]=mesh.p[0,:]+sf*u[I1]
mesh.p[1,:]=mesh.p[1,:]+sf*u[I2]

mesh.plot()
plt.axis('equal')
mesh.show()
