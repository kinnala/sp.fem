import numpy as np
import fem.geometry
import fem.asm
import scipy.sparse.linalg

geom=fem.geometry.GeometryMeshTri()
geom.refine(3)
mesh=geom.mesh()

D1=np.nonzero(mesh.p[0,:]==0)[0]
I=np.setdiff1d(np.arange(0,mesh.p.shape[1]),D1)

a=fem.asm.AssemblerTriP1(mesh)

def dudv(u,v,du,dv,x,h,w,dw):
    return du[0]*dv[0]+du[1]*dv[1]

def uv(u,v,du,dv,x,h,n,w):
    return w*u*v

def F(x,y):
    return 1+0*x

def fv(v,dv,x,h,w,dw):
    return F(x[0],x[1])*v

u=np.zeros(mesh.p.shape[1])
K=a.iasm(dudv)
f=a.iasm(fv)

mesh.plot(u)

alpha=0.3

for itr in range(30):
    B=a.fasm(uv,w=u)
    U=np.copy(u)
    u[I]=scipy.sparse.linalg.spsolve(K[np.ix_(I,I)]+B[np.ix_(I,I)],f[I])
    u[I]=alpha*u[I]+(1-alpha)*U[I]
    print np.linalg.norm(u-U)

mesh.plot(u)
mesh.show()
