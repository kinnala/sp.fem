import numpy as np
import fem.geometry
import fem.asm
import scipy.sparse.linalg

geom=fem.geometry.GeometryMeshTri()
geom.refine(3)
mesh=geom.mesh()

#D=mesh.boundary_nodes()
#I=np.setdiff1d(np.arange(0,mesh.p.shape[1]),D)

a=fem.asm.AssemblerTriP1(mesh)

def dudv(u,v,du,dv,x,h):
    return du[0]*dv[0]+du[1]*dv[1]
    
gamma=0.1
def uv(u,v,du,dv,x,h):
    return gamma*1/h*u*v
    
def fv(v,dv,x,h):
    return 2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])*v
    
def G(x,y):
    return np.sin(np.pi*x)
    
def gv(v,dv,x,h):
    return G(x[0],x[1])*v+gamma*1/h*G(x[0],x[1])*v

K=a.iasm(dudv)
B=a.fasm(uv)
f=a.iasm(fv)
g=a.fasm(gv)

x=np.zeros(K.shape[0])
x=scipy.sparse.linalg.spsolve(K+B,f+g)

mesh.plot(x)
