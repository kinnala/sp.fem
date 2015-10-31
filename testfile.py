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
    return np.sin(np.pi*x)*np.cos(np.pi*y)

x=np.zeros(mesh.p.shape[1])
x[D]=G(mesh.p[0,D],mesh.p[1,D])
    
for itr in range(3):
    K=a.iasm(dudv,w=x)
    y=x
    x[I]=scipy.sparse.linalg.spsolve(K[np.ix_(I,I)],-K[np.ix_(I,D)].dot(x[D]))
    print np.linalg.norm(y-x)
    mesh.plot(x)

mesh.show()
