import numpy as np
import fem.asm
import scipy.sparse.linalg
import scipy.sparse as spsp
import matplotlib.pyplot as plt
import fem.geometry as fegeom
import copy

#   geomlist=[
#           ('+','box',0,0,3,1),
#           ('-','circle',0.5,0.5,0.2,100),
#           ('+','circle',0.5,0.5,0.2,100),
#           ('-','circle',0.5,0.5,0.15,100),
#           ]
#   holes=[(0.5,0.5)]        
#           
#   g=fegeom.GeometryShapelyTriangle2D(geomlist)
#   mesh=g.mesh(0.03,holes=holes)
g=fegeom.GeometryMeshTriComsol("nsmesh.mphtxt")
mesh=g.mesh()

mesh.draw()

plt.axis('equal')
plt.show()

N=mesh.p.shape[1]
# left side wall (minus upmost and lowermost nodes)
Dleftx=np.nonzero(mesh.p[0,:]==0)[0]
Dleftx=np.setdiff1d(Dleftx,np.nonzero(mesh.p[1,:]==0)[0])
Dleftx=np.setdiff1d(Dleftx,np.nonzero(mesh.p[1,:]==1)[0])
# right side wall (minus upmost and lowermost nodes)
Drightx=np.nonzero(mesh.p[0,:]==3)[0]
Drightx=np.setdiff1d(Drightx,np.nonzero(mesh.p[1,:]==0)[0])
Drightx=np.setdiff1d(Drightx,np.nonzero(mesh.p[1,:]==1)[0])
# all dirichlet nodes (all boundary nodes minus right wall nodes)
Dallx=mesh.boundary_nodes()
Dallx=np.setdiff1d(Dallx,Drightx)
Dally=Dallx+N

D=np.union1d(Dallx,Dally)
I=np.setdiff1d(np.arange(0,3*N),D)

# index sets for accessing different components
I1=np.arange(0,N)
I2=I1+N
Ip=I2+N

print "NDOF: "+str(N)
print "NELS: "+str(mesh.t.shape[1])

a=fem.asm.AssemblerTriP1(mesh)

mu=0.25e-1

def dudv11(du,dv):
    return mu*du[1]*dv[1]+2*mu*du[0]*dv[0]

def dudv12(du,dv):
    return mu*du[0]*dv[1]

def dudv21(du,dv):
    return mu*du[1]*dv[0]
    
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

# brezzi-pitkaranta stabilization matrix
E=a.iasm(dpdqstab)

# initialize solution
u=np.zeros(3*N)

Vel=37.5
u[Dleftx]=Vel*(-4*mesh.p[1,Dleftx]**2+4*mesh.p[1,Dleftx])
#u[Drightx]=Vel*(-4*mesh.p[1,Drightx]**2+4*mesh.p[1,Drightx])

gamma=1e-1
eps=0 # works with zero when no-stress BC?

A=spsp.vstack((spsp.hstack((A11,A12)),spsp.hstack((A21,A22))))
B=spsp.vstack((B1,B2))
K=spsp.vstack((spsp.hstack((A,-B)),spsp.hstack((-B.T,eps*M-gamma*E)))).tocsr()

U=np.copy(u)
# initial condition from Stokes
u[I]=scipy.sparse.linalg.spsolve(K[np.ix_(I,I)],-K[np.ix_(I,D)].dot(u[D]),use_umfpack=True)

# time stepping
for itr in range(400):
    delta=0.003
    def fv(w1,v):
        return w1*v/delta
        
    f1=a.iasm(fv,w1=u[I1])
    f2=a.iasm(fv,w1=u[I2])
    f=np.hstack((f1,f2,0*f1))
    
    print "time step: "+str(itr)
    # picard iteration
    for jtr in range(50):
        alpha=np.min((0.7,float(jtr)/10.+0.1))
        def duuv(w1,w2,du,v):
            return du[0]*w1*v+du[1]*w2*v
            
        C=a.iasm(duuv,w1=u[I1],w2=u[I2],intorder=3)
        #C11=a.iasm(duuv,w1=u[I1],w2=u[I2],intorder=4)
        #C22=a.iasm(duuv,w1=u[I1],w2=u[I2],intorder=4)
        
        #A=spsp.vstack((spsp.hstack((A11+C11+M/delta,A12)),spsp.hstack((A21,A22+C22+M/delta))))
        A=spsp.vstack((spsp.hstack((A11+C+M/delta,A12)),spsp.hstack((A21,A22+C+M/delta))))
        B=spsp.vstack((B1,B2))
        
        K=spsp.vstack((spsp.hstack((A,-B)),spsp.hstack((-B.T,eps*M-gamma*E)))).tocsr()

        U=np.copy(u)
        # direct
        u[I]=scipy.sparse.linalg.spsolve(K[np.ix_(I,I)],f[I]-K[np.ix_(I,D)].dot(u[D]),use_umfpack=True)
        u[I]=alpha*u[I]+(1-alpha)*U[I]

        residual=np.linalg.norm(u-U)
        print residual
        if residual<=1e-3:
            # continue to newton
            break
         
    h0=mesh.plot(np.sqrt(u[I1]**2+u[I2]**2),smooth=True)
    plt.axis('equal')
    plt.colorbar(h0)
    plt.savefig('karman_vmag_step'+str(itr).zfill(4)+'.png')
    #   h1=mesh.plot(u[I1],smooth=True)
    #   plt.axis('equal')
    #   plt.colorbar(h1)
    #   h2=mesh.plot(u[I2],smooth=True)
    #   plt.axis('equal')
    #   plt.colorbar(h2)
    h3=mesh.plot(u[Ip],smooth=True)
    plt.axis('equal')
    plt.colorbar(h3)
    plt.savefig('karman_pres_step'+str(itr).zfill(4)+'.png')


#### NOT USEFUL STUFF AFTER THIS ####
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
