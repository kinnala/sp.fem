# -*- coding: utf-8 -*-
"""
@author: knl
"""

import numpy as np
import fem.mesh as fmsh
import fem.asm as fasm
import fem.geometry as fgeo
import scipy.sparse.linalg
import scipy.sparse as sp
import matplotlib.pyplot as plt
import copy       
import fem.element as felem
import fem.mapping as fmap
from fem.quadrature import *

plt.close('all')

m=fmsh.MeshQuad()
m.refine(5)
m.jiggle()
m.draw()

qmap=fmap.MappingQ1
e1=felem.ElementQ2()
e2=felem.ElementTriDG(felem.ElementTriP1())

a=fasm.AssemblerElement(m,qmap,e1)
b=fasm.AssemblerElement(m,qmap,e1,e2)
c=fasm.AssemblerElement(m,qmap,e2)

def dudv(du,dv):
    return du[0]*dv[0]+du[1]*dv[1]
    
def duxv(du,v):
    return du[0]*v

def duyv(du,v):
    return du[1]*v    

Iv1=a.dofnum_u.n_dof.flatten()
Iv2=a.dofnum_u.n_dof.flatten()+a.dofnum_u.N  
Ip=c.dofnum_u.t_dof.flatten()+2*a.dofnum_u.N
Ip1=c.dofnum_u.i_dof[0,:]+2*a.dofnum_u.N
Ip2=c.dofnum_u.i_dof[1,:]+2*a.dofnum_u.N
Ip3=c.dofnum_u.i_dof[2,:]+2*a.dofnum_u.N

# assemble matrices
A=a.iasm(dudv,intorder=4)
B1=b.iasm(duxv,intorder=4)
B2=b.iasm(duyv,intorder=4)
Z1=sp.csr_matrix(A.shape)
#C=spsp.csr_matrix((len(Ip),len(Ip)))
C=c.iasm(lambda u,v: 1e-5*u*v)

# system matrix
K=sp.vstack((
    sp.hstack((  A,   Z1,  -B1.T)),
    sp.hstack(( Z1,    A,  -B2.T)),
    sp.hstack((-B1,  -B2,  -C)))).tocsr()

# initialize solution vector 
u=np.zeros(K.shape[0])

# find node sets
ix_vel_nodes_upper=m.nodes_satisfying(lambda x,y:y==1)
ix_vel_facets_upper=m.facets_satisfying(lambda x,y:y==1)
ix_vel_nodes_all=m.boundary_nodes()
ix_vel_facets_all=m.boundary_facets()
Dupper=a.dofnum_u.getdofs(N=ix_vel_nodes_upper,F=ix_vel_facets_upper)
Dall=a.dofnum_u.getdofs(N=ix_vel_nodes_all,F=ix_vel_facets_all)

u[Dupper]=1.0

I=np.setdiff1d(np.arange(0,A.shape[0],dtype=np.int64),Dall)
I=np.union1d(I,I+a.dofnum_u.N) # vel y components
I=np.union1d(I,Ip) # pressure components

f=-(K.T[Dupper].T.dot(u[Dupper]))

u[I]=scipy.sparse.linalg.spsolve(K[I].T[I].T,f[I])

m.plot(u[Iv1],smooth=True)
m.plot(u[Iv2],smooth=True)
m.plot((u[Ip1]+u[Ip2]+u[Ip3])/3.)
