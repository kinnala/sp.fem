import numpy as np
import fem.mesh as fm
import fem.geometry as fg
import fem.asm as fa
import oldassembly as oasm

import scipy.sparse.linalg

geom=fg.GeometryMeshTri(np.array([[0,1,0,1],[0,0,1,1]],dtype=np.float_),np.array([[0,1,2],[1,2,3]],dtype=np.intp).T)

geom.refine(5)

mesh=geom.mesh()
#mesh.plot()

# nmesh=mesh
# nmesh.t=nmesh.t.T
# nmesh.p=nmesh.p.T

bilin=lambda u,v,du,dv,x: du[0]*dv[0]+du[1]*dv[1]
mass=lambda u,v,du,dv,x: u*v

lin=lambda v,dv,x: 1*v;

b=fa.AssemblerTriP1(mesh)

A=b.iasm(bilin)
M=b.iasm(mass)

f=M.dot(np.ones(A.shape[0]))
g=b.iasm(lin)

D1=np.nonzero(mesh.p[0,:]==0)[0]
D2=np.nonzero(mesh.p[1,:]==0)[0]
D3=np.nonzero(mesh.p[0,:]==1)[0]
D4=np.nonzero(mesh.p[1,:]==1)[0]

D=np.union1d(D1,D2);
D=np.union1d(D,D3);
D=np.union1d(D,D4);

I=np.setdiff1d(np.arange(0,A.shape[0]),D)

X=np.zeros(A.shape[0])
X[I]=scipy.sparse.linalg.spsolve(A[np.ix_(I,I)],f[I])
#X[I]=x

mesh.plot(X)
