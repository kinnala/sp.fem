# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:44:11 2015

@author: knl
"""

import numpy as np
import fem.mesh as fmsh
import fem.asm as fasm
import fem.geometry as fgeo
import scipy.sparse.linalg
import scipy.sparse as spsp
import matplotlib.pyplot as plt
import fem.geometry as fegeom
import copy       

g=fgeo.GeometryMeshTri()
g.refine(5)

mesh=g.mesh()
            
a=fasm.AssemblerTriPp(mesh,2)
b=fasm.AssemblerTriPp(mesh,2,1)
c=fasm.AssemblerTriPp(mesh,1)

mu=1.

def dudv11(du,dv):
    return mu*du[1]*dv[1]+2*mu*du[0]*dv[0]

def dudv12(du,dv):
    return mu*du[0]*dv[1]

def dudv21(du,dv):
    return mu*du[1]*dv[0]
    
def dudv22(du,dv):
    return 2*mu*du[1]*dv[1]+mu*du[0]*dv[0]
    
def duv1(du,v):
    return du[0]*v

def duv2(du,v):
    return du[1]*v
    
def uv(u,v):
    return u*v


# assemble static matrices
A11=a.iasm(dudv11)
A12=a.iasm(dudv12)
A21=a.iasm(dudv21)
A22=a.iasm(dudv22)
B1=b.iasm(duv1)
B2=b.iasm(duv2)
C=c.iasm(uv)

eps=1e-2

A=spsp.vstack((spsp.hstack((A11,A12)),spsp.hstack((A21,A22))))
B=spsp.hstack((B1,B2))
K=spsp.vstack((spsp.hstack((A,-B.T)),spsp.hstack((-B,eps*C)))).tocsr()

u=np.zeros(2*a.dofnum1.N+b.dofnum1.N)

Dvupper=mesh.nodes_satisfying(lambda x,y:y==1)
Deupper=mesh.facets_satisfying(lambda x,y:y==1)
Dupper=a.dofnum1.getdofs(N=Dvupper,F=Deupper)
u[Dupper]=10

Dv=mesh.boundary_nodes()
De=mesh.boundary_facets()
D1=a.dofnum1.getdofs(N=Dv,F=De)
D2=a.dofnum1.getdofs(N=Dv,F=De)+a.dofnum1.N
D3=b.dofnum2.getdofs(N=Dv,F=De)+2*a.dofnum1.N
D=np.union1d(np.union1d(D1,D2),D3)
I=np.setdiff1d(np.arange(0,2*a.dofnum1.N+b.dofnum2.N,dtype=np.int64),D)

u[I]=scipy.sparse.linalg.spsolve(K[np.ix_(I,I)],-K[np.ix_(I,D)].dot(u[D]),use_umfpack=True)

I1=a.dofnum1.n_dof
I2=a.dofnum1.n_dof+a.dofnum1.N

mesh.plot(np.sqrt(u[I1].flatten()**2+u[I2].flatten()**2),smooth=True)