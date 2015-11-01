import numpy as np
import fem.geometry
import fem.asm
import scipy.sparse.linalg

geom=fem.geometry.GeometryMeshTri()
geom.refine(5)
mesh=geom.mesh()

D=mesh.boundary_nodes()
I=np.setdiff1d(np.arange(0,mesh.p.shape[1]),D)

a=fem.asm.AssemblerTriP1(mesh)

def dudv(u,v,du,dv,x,h,w,dw):
    return (du[0]*dv[0]+du[1]*dv[1])/np.sqrt(1+dw[0]**2+dw[1]**2)

def G(x,y):
    return np.cos(2*np.pi*np.abs(x+y))

u=np.zeros(mesh.p.shape[1])
u[D]=G(mesh.p[0,D],mesh.p[1,D])

for itr in range(5):
    K=a.iasm(dudv,w=u)
    U=np.copy(u)
    u[I]=scipy.sparse.linalg.spsolve(K[np.ix_(I,I)],-K[np.ix_(I,D)].dot(u[D]))
    mesh.plot(u-U)

mesh.plot(u)
mesh.show()
