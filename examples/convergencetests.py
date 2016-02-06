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

L2errs={}
H1errs={}
hs={}

for jtr in [1,2,3,4]:
    L2errs[jtr-1]=np.zeros(3)
    H1errs[jtr-1]=np.zeros(3)
    hs[jtr-1]=np.zeros(3)
    for itr in [1,2,3]:
        tmsh=fmsh.MeshTri(np.array([[0,0],[0,1],[1,0],[1,1]]).T,np.array([[0,1,2],[1,2,3]]).T)
        
        g=fgeo.GeometryMeshTri()
        g.refine(itr+1)
        mesh=g.mesh()
        
        a=fasm.AssemblerTriPp(mesh,jtr)
        
        def dudv(du,dv):
            return du[0]*dv[0]+du[1]*dv[1]
            
        def fv(v,x):
            return 2.*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])*v
            
        A=a.iasm(dudv)
        f=a.iasm(fv)
           
        u=np.zeros(a.dofnum.N)
        Dv=mesh.boundary_nodes()
        De=mesh.boundary_facets()
        D=a.dofnum.getdofs(N=Dv,F=De)
        I=np.setdiff1d(np.arange(0,a.dofnum.N,dtype=np.int64),D)
        
        
        u[I]=scipy.sparse.linalg.spsolve(A[np.ix_(I,I)],f[I])
        
        def exact(x,y):
            return np.sin(np.pi*x)*np.sin(np.pi*y)
            
        def exactdx(x,y):
            return np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
            
        def exactdy(x,y):
            return np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
        
        L2errs[jtr-1][itr-1]=a.L2error(u,exact)
        H1errs[jtr-1][itr-1]=a.H1error(u,exactdx,exactdy)
        hs[jtr-1][itr-1]=mesh.param()

plt.figure()
for jtr in [1,2,3,4]:
    plt.loglog(hs[jtr-1],L2errs[jtr-1],'o-')
    plt.hold('on')
    print (np.log10(L2errs[jtr-1][0])-np.log10(L2errs[jtr-1][2]))/(np.log10(hs[jtr-1][2])-np.log10(hs[jtr-1][0]))
plt.grid()

plt.figure()
for jtr in [1,2,3,4]:
    plt.loglog(hs[jtr-1],H1errs[jtr-1],'o-')
    plt.hold('on')
    print (np.log10(H1errs[jtr-1][0])-np.log10(H1errs[jtr-1][2]))/(np.log10(hs[jtr-1][2])-np.log10(hs[jtr-1][0]))
plt.grid()