import numpy as np
import fem.geometry
import fem.asm
import scipy.sparse.linalg

geom=fem.geometry.GeometryMeshTri()
geom.refine(3)
mesh=geom.mesh()

D=mesh.boundary_nodes()
I=np.setdiff1d(np.arange(0,mesh.p.shape[1]),D)

a=fem.asm.AssemblerTriP1(mesh)

def dudv(u,v,du,dv,x):
    return du[0]*dv[0]+du[1]*dv[1]
    
def fv(v,dv,x):
    return 2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])*v

K=a.iasm(dudv)
f=a.iasm(fv)

x=np.zeros(K.shape[0])
x[I]=scipy.sparse.linalg.spsolve(K[np.ix_(I,I)],f[I])

mesh.plot(x)
