import numpy as np
import fem.asm
import scipy.sparse.linalg
import scipy.sparse as spsp
import matplotlib.pyplot as plt
import fem.geometry as fegeom
import copy
import time

geomlist=[
        ('+','box',0,0,1,1),
        ]
        
g=fegeom.GeometryShapely2D(geomlist)
mesh=g.mesh(0.025)

mesh.draw()

plt.axis('equal')
plt.show()

N=mesh.p.shape[1]
# left side wall (minus upmost and lowermost nodes)
Dleftx=np.nonzero(mesh.p[0,:]==0)[0]
Dleftx=np.setdiff1d(Dleftx,np.nonzero(mesh.p[1,:]==0)[0])
Dleftx=np.setdiff1d(Dleftx,np.nonzero(mesh.p[1,:]==1)[0])
# right side wall (minus upmost and lowermost nodes)
Drightx=np.nonzero(mesh.p[0,:]==1)[0]
Drightx=np.setdiff1d(Drightx,np.nonzero(mesh.p[1,:]==0)[0])
Drightx=np.setdiff1d(Drightx,np.nonzero(mesh.p[1,:]==1)[0])
# lower and upper side wall 
Dlowerx=np.nonzero(mesh.p[1,:]==0)[0]
Dupperx=np.nonzero(mesh.p[1,:]==1)[0]
# all dirichlet nodes 
Dallx=mesh.boundary_nodes()
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

mu=1

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

Vel=100
u[Dupperx]=Vel

gamma=1e-1
eps=1e-2 # works with zero when no-stress BC?

A=spsp.vstack((spsp.hstack((A11,A12)),spsp.hstack((A21,A22))))
B=spsp.vstack((B1,B2))
K=spsp.vstack((spsp.hstack((A,-B)),spsp.hstack((-B.T,eps*M-gamma*E)))).tocsr()

U=np.copy(u)
# initial condition from Stokes
u[I]=scipy.sparse.linalg.spsolve(K[np.ix_(I,I)],-K[np.ix_(I,D)].dot(u[D]),use_umfpack=True)

# time stepping
for itr in [3]:
    delta=0.003
    def fv(w1,v):
        return w1*v/delta
        
    f1=a.iasm(fv,w1=u[I1])
    f2=a.iasm(fv,w1=u[I2])
    f=0*np.hstack((f1,f2,0*f1))

    print "time step: "+str(itr)
    # picard iteration
    for jtr in range(50):
        alpha=np.min((0.9,float(jtr)/10.+0.1))
        def duuv(w1,w2,du,v):
            return du[0]*w1*v+du[1]*w2*v
            
        #start=time.clock()
        C=a.iasm(duuv,w1=u[I1],w2=u[I2],intorder=3)

        A=spsp.vstack((spsp.hstack((A11+C+0*M/delta,A12)),spsp.hstack((A21,A22+C+0*M/delta))))
        B=spsp.vstack((B1,B2))
        
        K=spsp.vstack((spsp.hstack((A,-B)),spsp.hstack((-B.T,eps*M-gamma*E)))).tocsr()
        #end=time.clock()
        #print "assembly time: "+str(end-start)+" sec"

        U=np.copy(u)
        # direct
        #start=time.clock()
        u[I]=scipy.sparse.linalg.spsolve(K[np.ix_(I,I)],f[I]-K[np.ix_(I,D)].dot(u[D]),use_umfpack=True)
        #end=time.clock()
        #print "solve time: "+str(end-start)+" sec"
        u[I]=alpha*u[I]+(1-alpha)*U[I]

        residual=np.linalg.norm(u-U)
        print residual
        if residual<=1e-5:
            # continue to newton
            break
         
    h0=mesh.plot(np.sqrt(u[I1]**2+u[I2]**2),smooth=True)
    plt.axis('equal')
    plt.colorbar(h0)
    plt.savefig('lid_vmag_step'+str(itr).zfill(4)+'.png')
    #   h1=mesh.plot(u[I1],smooth=True)
    #   plt.axis('equal')
    #   plt.colorbar(h1)
    #   h2=mesh.plot(u[I2],smooth=True)
    #   plt.axis('equal')
    #   plt.colorbar(h2)
    h3=mesh.plot(u[Ip],smooth=True)
    plt.axis('equal')
    plt.colorbar(h3)
    plt.savefig('lid_pres_step'+str(itr).zfill(4)+'.png')

    u1fun=mesh.interpolator(u[I1])
    u2fun=mesh.interpolator(u[I2])

    plt.figure()
    ys=np.array([0,0.0547,0.0625,0.0703,0.1016,0.1719,0.2812,0.4531,0.5000,0.6172,0.7344,0.8516,0.9531,0.9609,0.9688,0.9766])
    soltruey=np.array([0,-0.0372,-0.0419,-0.0477,-0.0643,-0.1015,-0.1566,-0.2109,-0.2058,-0.1364,0.0033,0.2315,0.6872,0.7372,0.7887,0.8412])
    plt.plot(ys,u1fun(0*ys+0.5,ys)/Vel,'bo-',hold='on')
    plt.plot(ys,soltruey,'ro-')
    plt.savefig('lid_compy_step'+str(itr).zfill(4)+'.png')

    plt.figure()
    xs=np.array([0,0.0625,0.0703,0.0781,0.0938,0.1563,0.2266,0.2344,0.5,0.8047,0.8594,0.9063,0.9453,0.9531,0.9609,0.9688,1.0000])
    soltruex=np.array([0,0.0923,0.1009,0.1089,0.1232,0.1608,0.1751,0.1753,0.0545,-0.2453,-0.2245,-0.1691,-0.1031,-0.0886,-0.0739,-0.0591,0.0000])
    plt.plot(xs,u2fun(xs,0*xs+0.5)/Vel,'bo-',hold='on')
    plt.plot(xs,soltruex,'ro-')
    plt.savefig('lid_compx_step'+str(itr).zfill(4)+'.png')

    print np.max(np.abs(soltruex-u2fun(xs,0*xs+0.5)/Vel))
    print np.max(np.abs(soltruey-u1fun(0*ys+0.5,ys)/Vel))
