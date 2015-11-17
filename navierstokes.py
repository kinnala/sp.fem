import numpy as np
import fem.asm
import scipy.sparse.linalg
import scipy.sparse as spsp
import matplotlib.pyplot as plt
import fem.geometry as fegeom
import copy

geomlist=[
        ('+','box',0,0,5,1),
        ('-','circle',1,0.5,0.2,20)
        ]
holes=[(1,0.5)]        
        
g=fegeom.GeometryShapelyTriangle2D(geomlist)
mesh=g.mesh(0.05,holes=holes)

mesh.plot()

plt.axis('equal')
plt.show()

N=mesh.p.shape[1]
# left side wall (minus upmost and lowermost nodes)
Dleftx=np.nonzero(mesh.p[0,:]==0)[0]
Dleftx=np.setdiff1d(Dleftx,np.nonzero(mesh.p[1,:]==0)[0])
Dleftx=np.setdiff1d(Dleftx,np.nonzero(mesh.p[1,:]==1)[0])
# right side wall (minus upmost and lowermost nodes)
Drightx=np.nonzero(mesh.p[0,:]==5)[0]
Drightx=np.setdiff1d(Drightx,np.nonzero(mesh.p[1,:]==0)[0])
Drightx=np.setdiff1d(Drightx,np.nonzero(mesh.p[1,:]==1)[0])
# all dirichlet nodes (all boundary nodes minus right wall nodes)
Dallx=mesh.boundary_nodes()
#Dallx=np.setdiff1d(Dallx,np.nonzero(mesh.p[0,:]==5)[0])
Dally=Dallx+N

D=np.union1d(Dallx,Dally)
I=np.setdiff1d(np.arange(0,3*N),D)

# index sets for accessing different components
I1=np.arange(0,N)
I2=I1+N
Ip=I2+N

a=fem.asm.AssemblerTriP1(mesh)

mu=1.
delta=0.1

def dudv11(du,dv):
    return mu*du[1]*dv[1]+2*mu*du[0]*dv[0]

def dudv12(du,dv):
    return 0*mu*du[0]*dv[1]

def dudv21(du,dv):
    return 0*mu*du[1]*dv[0]
    
def dudv22(du,dv):
    return 2*mu*du[1]*dv[1]+mu*du[0]*dv[0]
    
def duv1(u,dv):
    return u*dv[0]

def duv2(u,dv):
    return u*dv[1]
    
def uv(u,v):
    return u*v
    
def dpdqstab(du,dv,h):
    return h**2*(du[0]*dv[0]+du[1]*dv[1])   

# assemble static matrices
A11=a.iasm(dudv11)
A12=a.iasm(dudv12)
A21=a.iasm(dudv21)
A22=a.iasm(dudv22)
B1=a.iasm(duv1)
B2=a.iasm(duv2)
M=a.iasm(uv)

# brezzi-pitkäranta stabilization matrix
E=a.iasm(dpdqstab)

# initialize solution
u=np.zeros(3*N)

u[Dleftx]=-4*mesh.p[1,Dleftx]**2+4*mesh.p[1,Dleftx]
u[Drightx]=-4*mesh.p[1,Drightx]**2+4*mesh.p[1,Drightx]

gamma=10
alpha=0.3
eps=1.0e-3

# time stepping
for itr in range(1):
    def fv(w1,v):
        return w1*v/delta
        
    f1=a.iasm(fv,w1=u[I1])
    f2=a.iasm(fv,w1=u[I2])
    f=0*np.hstack((f1,f2,0*f1))
    
    # picard iteration
    for jtr in range(1):
        def duuv1(dw1,u,v):
            return dw1[0]*u*v
        def duuv2(dw1,u,v):
            return dw1[1]*u*v
            
        C11=0*a.iasm(duuv1,w1=u[I1],intorder=4)
        C12=0*a.iasm(duuv2,w1=u[I1],intorder=4)
        C21=0*a.iasm(duuv1,w1=u[I2],intorder=4)
        C22=0*a.iasm(duuv2,w1=u[I2],intorder=4)
        
        A=spsp.vstack((spsp.hstack((A11,A12)),spsp.hstack((A21,A22))))
        B=spsp.vstack((B1,B2))
        
        K=spsp.vstack((spsp.hstack((A,-B)),spsp.hstack((-B.T,eps*M-gamma*E)))).tocsr()
        
        #K=spsp.vstack((spsp.hstack((A11+0*M/delta+C11,A12+C12,-B1)),spsp.hstack((A21+C21,A22+0*M/delta+C22,-B2)),spsp.hstack((-B1.T,-B2.T,eps*M-gamma*E)))).tocsr()

        U=np.copy(u)
        u[I]=scipy.sparse.linalg.spsolve(K[np.ix_(I,I)],f[I]-K[np.ix_(I,D)].dot(u[D]))
        #u[I]=alpha*u[I]+(1-alpha)*U[I]
        
        mesh.plot(u[I1])
        plt.axis('equal')
        mesh.plot(u[I2])
        plt.axis('equal')
        mesh.plot(u[Ip])
        plt.axis('equal')
if 0:
    # continuation
    for ctr in np.arange(0,2.1,0.1):
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
                sf=1.
                dmesh=copy.deepcopy(mesh)
                dmesh.p[0,:]=dmesh.p[0,:]+sf*u[I1]
                dmesh.p[1,:]=dmesh.p[1,:]+sf*u[I2]
                dmesh.plot()
                plt.xlim(-1,1)
                plt.ylim(0,2)
                plt.savefig('stvenant_step'+str(step).zfill(2)+'.png')
                step=step+1
                break
            print np.linalg.norm(u-U)
    
    sf=2.
    mesh.p[0,:]=mesh.p[0,:]+sf*u[I1]
    mesh.p[1,:]=mesh.p[1,:]+sf*u[I2]
    
    mesh.plot()
    plt.axis('equal')
    mesh.show()
