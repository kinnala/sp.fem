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
import copy       
import fem.element as felem
import fem.mapping as fmap

plt.close('all')

def U(x):
    return 1+x[0]-x[0]**2*x[1]**2+x[0]*x[1]*x[2]**3
    
def dUdx(x):
    return 1-2*x[0]*x[1]**2+x[1]*x[2]**3
    
def dUdy(x):
    return -2*x[0]**2*x[1]+x[0]*x[2]**3
    
def dUdz(x):
    return 3*x[0]*x[1]*x[2]**2

def dudv(du,dv):
    return du[0]*dv[0]+du[1]*dv[1]+du[2]*dv[2]

def uv(u,v):
    return u*v

def F(x,y,z):
    return 2*x**2+2*y**2-6*x*y*z

def fv(v,x):
    return F(x[0],x[1],x[2])*v

def G(x,y,z):
    return (x==1)*(3-3*y**2+2*y*z**3)+\
           (x==0)*(-y*z**3)+\
           (y==1)*(1+x-3*x**2+2*x*z**3)+\
           (y==0)*(1+x-x*z**3)+\
           (z==1)*(1+x+4*x*y-x**2*y**2)+\
           (z==0)*(1+x-x**2*y**2)

def gv(v,x):
    return G(x[0],x[1],x[2])*v

dexact={}
dexact[0]=dUdx
dexact[1]=dUdy
dexact[2]=dUdz

hs=np.array([])
H1err=np.array([])
L2err=np.array([])

for itr in range(1):
    mesh=fmsh.MeshTet()
    mesh.refine(2)

    a=fasm.AssemblerElement(mesh,fmap.MappingAffine,felem.ElementTetP2())

    A=a.iasm(dudv)
    f=a.iasm(fv)

    B=a.fasm(uv)
    g=a.fasm(gv)

    u=np.zeros(a.dofnum_u.N)

    u=scipy.sparse.linalg.spsolve(A+B,f+g)

    p={}
    p[0]=mesh.p[0,:]
    p[1]=mesh.p[1,:]
    p[2]=mesh.p[2,:]

    hs=np.append(hs,mesh.param())
    L2err=np.append(L2err,a.L2error(u,U))
    H1err=np.append(H1err,a.H1error(u,dexact))

mesh.draw(lambda x,y,z:x<=0.5,u[range(mesh.p.shape[1])])

#pfit=np.polyfit(np.log10(hs),np.log10(np.sqrt(L2err**2+H1err**2)),1)
