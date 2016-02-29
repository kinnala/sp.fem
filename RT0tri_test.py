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
import scipy.sparse as sp
import matplotlib.pyplot as plt
import copy
import fem.element as felem
import fem.mapping as fmap

mesh=fmsh.MeshTri()
mesh.refine(4)

a=fasm.AssemblerElement(mesh,fmap.MappingAffine,felem.ElementTriRT0())
b=fasm.AssemblerElement(mesh,fmap.MappingAffine,felem.ElementTriRT0(),felem.ElementP0())
c=fasm.AssemblerElement(mesh,fmap.MappingAffine,felem.ElementP0())

def sigtau(u,v):
    sig=u
    tau=v
    return sig[0]*tau[0]+sig[1]*tau[1]

def divsigv(du,v):
    divsig=du
    return divsig*v

def fv(v,x):
    return 2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])*v

def uv(u,v):
    return u*v
    
A=a.iasm(sigtau)
B=b.iasm(divsigv)
C=c.iasm(uv)
f=c.iasm(fv)

K1=sp.hstack((-A,-B.T))
K2=sp.hstack((-B,0*C))
K=sp.vstack((K1,K2)).tocsr()

F=np.hstack((np.zeros(A.shape[0]),f))

u=np.zeros(a.dofnum_u.N+c.dofnum_u.N)

u=scipy.sparse.linalg.spsolve(K,F)

Iu=np.arange(C.shape[0],dtype=np.int64)+A.shape[0]

mesh.plot(u[Iu])
mesh.show()


def exact(x):
    return np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
def exactdx(x):
    return np.pi*np.cos(np.pi*x[0])*np.sin(np.pi*x[1])
def exactdy(x):
    return np.pi*np.sin(np.pi*x[0])*np.cos(np.pi*x[1])

dexact={}
dexact[0]=exactdx
dexact[1]=exactdy

print mesh.param()
print c.L2error(u[Iu],exact)
print c.H1error(u[Iu],dexact) 
