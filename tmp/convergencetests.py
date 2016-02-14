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

#g=fgeo.GeometryMeshTri()
#g.refine(3)
#mesh=g.mesh()
mesh=fmsh.MeshTet()
mesh.refine(5)

a=fasm.AssemblerElement(mesh,fmap.MappingAffine,felem.ElementP1())

def dudv(du,dv):
    if len(du)==2:
        return du[0]*dv[0]+du[1]*dv[1]
    if len(du)==3:
        return du[0]*dv[0]+du[1]*dv[1]+du[2]*dv[2]

def fv(v,x):
    if len(x)==2:
        return 2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])*v
    if len(x)==3:
        return 3*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])*np.sin(np.pi*x[2])*v
    
A=a.iasm(dudv)
f=a.iasm(fv)

u=np.zeros(a.dofnum_u.N)
Dv=mesh.boundary_nodes()
De=mesh.boundary_facets()
D=a.dofnum_u.getdofs(N=Dv,F=De)
I=np.setdiff1d(np.arange(0,a.dofnum_u.N,dtype=np.int64),D)

u[I]=scipy.sparse.linalg.spsolve(A[I].T[I].T,f[I])

#def exact(x,y):
#    return np.sin(np.pi*x)*np.sin(np.pi*y)
def exact(x):
    return np.sin(np.pi*x[0])*np.sin(np.pi*x[1])*np.sin(np.pi*x[2])
def exactdx(x):
    return np.pi*np.cos(np.pi*x[0])*np.sin(np.pi*x[1])*np.sin(np.pi*x[2])
def exactdy(x):
    return np.pi*np.sin(np.pi*x[0])*np.cos(np.pi*x[1])*np.sin(np.pi*x[2])
def exactdz(x):
    return np.pi*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])*np.cos(np.pi*x[2])

dexact={}
dexact[0]=exactdx
dexact[1]=exactdy
dexact[2]=exactdz

#mesh.draw(test=lambda x,y,z: x+y>=1,u=u-exact(mesh.p[0,:],mesh.p[1,:],mesh.p[2,:]))

print mesh.param()
print a.L2error(u,exact)
print a.H1error(u,dexact)
#mesh.plot(u)

if 0:
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
            
            a=fasm.AssemblerElement(mesh,fmap.MappingAffineTri,felem.ElementP1)
            
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
