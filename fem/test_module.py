# -*- coding: utf-8 -*-
"""
Module level tests; tests that require multiple working classes.

@author: Tom Gustafsson
"""
import unittest
import fem.mesh as fmsh
import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse as spsp
import fem.asm as fasm
import fem.mapping as fmap
import fem.element as felem
import matplotlib.pyplot as plt

class RT0Test(unittest.TestCase):
    """Assemble and solve mixed Poisson equation
    using the first-order Raviart-Thomas element."""
    def runTest(self):
        mesh=fmsh.MeshTri()
        mesh.refine(2)
        
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
            
        def exact(x):
            return np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
        
        hs=np.array([])
        L2err=np.array([])
            
        for itr in range(1,4):
            mesh.refine()
            a=fasm.AssemblerElement(mesh,fmap.MappingAffine,felem.ElementTriRT0())
            b=fasm.AssemblerElement(mesh,fmap.MappingAffine,felem.ElementTriRT0(),felem.ElementP0())
            c=fasm.AssemblerElement(mesh,fmap.MappingAffine,felem.ElementP0())
            
            A=a.iasm(sigtau)
            B=b.iasm(divsigv)
            C=c.iasm(uv)
            f=c.iasm(fv)
        
            K1=spsp.hstack((-A,-B.T))
            K2=spsp.hstack((-B,0*C))
            K=spsp.vstack((K1,K2)).tocsr()
            
            F=np.hstack((np.zeros(A.shape[0]),f))
            
            u=np.zeros(a.dofnum_u.N+c.dofnum_u.N)
            
            u=spsolve(K,F)
            
            Iu=np.arange(C.shape[0],dtype=np.int64)+A.shape[0]
            
            hs=np.append(hs,mesh.param())
            L2err=np.append(L2err,c.L2error(u[Iu],exact))
    
    
        pfit=np.polyfit(np.log10(hs),np.log10(L2err),1)
        self.assertTrue(pfit[0]>=0.95)
        self.assertTrue(pfit[0]<=1.15)
        
        
class TetP2Test(unittest.TestCase):
    """Test second order tetrahedral element and facet
    assembly."""
    def runTest(self):
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
        
        for itr in range(1,4):
            mesh=fmsh.MeshTet()
            mesh.refine(itr)
        
            a=fasm.AssemblerElement(mesh,fmap.MappingAffine,felem.ElementTetP2())
        
            A=a.iasm(dudv)
            f=a.iasm(fv)
        
            B=a.fasm(uv)
            g=a.fasm(gv)
        
            u=np.zeros(a.dofnum_u.N)
        
            u=spsolve(A+B,f+g)
        
            p={}
            p[0]=mesh.p[0,:]
            p[1]=mesh.p[1,:]
            p[2]=mesh.p[2,:]
        
            hs=np.append(hs,mesh.param())
            L2err=np.append(L2err,a.L2error(u,U))
            H1err=np.append(H1err,a.H1error(u,dexact))
            
        pfit=np.polyfit(np.log10(hs),np.log10(np.sqrt(L2err**2+H1err**2)),1)
        self.assertTrue(pfit[0]>=1.95)
        self.assertTrue(pfit[0]<=2.2)

class Q1Q2Test(unittest.TestCase):
    """Test first and second-order quadrilateral elements
    and their facet assembly."""
    def runTest(self):

        def U(x):
            return 1+x[0]-x[0]**2*x[1]**2+np.exp(x[0])
        
        def dUdx(x):
            return 1-2*x[0]*x[1]**2+np.exp(x[0])
        
        def dUdy(x):
            return -2*x[0]**2*x[1]
        
        def dudv(du,dv):
            return du[0]*dv[0]+du[1]*dv[1]
        
        def uv(u,v):
            return u*v
        
        def F(x,y):
            return 2*x**2+2*y**2-np.exp(x)
        
        def fv(v,x):
            return F(x[0],x[1])*v
        
        def G(x,y):
            return (x==1)*(3-3*y**2+2*np.exp(1))+\
                    (x==0)*(0)+\
                    (y==1)*(1+x-3*x**2+np.exp(x))+\
                    (y==0)*(1+x+np.exp(x))
        
        def gv(v,x):
            return G(x[0],x[1])*v
        
        dexact={}
        dexact[0]=dUdx
        dexact[1]=dUdy
        
        # Q1
        hs=np.array([])
        H1errs=np.array([])
        L2errs=np.array([])
        
        mesh=fmsh.MeshQuad()
        
        for itr in range(3):
            mesh.refine()
        
            a=fasm.AssemblerElement(mesh,fmap.MappingQ1,felem.ElementQ1())
        
            A=a.iasm(dudv)
            f=a.iasm(fv)
        
            B=a.fasm(uv)
            g=a.fasm(gv)
        
            u=np.zeros(a.dofnum_u.N)
            u=spsolve(A+B,f+g)
        
        
            hs=np.append(hs,mesh.param())
            L2errs=np.append(L2errs,a.L2error(u,U))
            H1errs=np.append(H1errs,a.H1error(u,dexact))
        
        pfit=np.polyfit(np.log10(hs),np.log10(np.sqrt(L2errs**2+H1errs**2)),1)
        self.assertTrue(pfit[0]>=0.95)
        self.assertTrue(pfit[0]<=1.05)
        
        # Q2
        hs=np.array([])
        H1errs=np.array([])
        L2errs=np.array([])        
        
        mesh=fmsh.MeshQuad()
        
        for itr in range(3):
            mesh.refine()
        
            a=fasm.AssemblerElement(mesh,fmap.MappingQ1,felem.ElementQ2())
        
            A=a.iasm(dudv)
            f=a.iasm(fv)
        
            B=a.fasm(uv)
            g=a.fasm(gv)
        
            u=np.zeros(a.dofnum_u.N)
            u=spsolve(A+B,f+g)
        
        
            hs=np.append(hs,mesh.param())
            L2errs=np.append(L2errs,a.L2error(u,U))
            H1errs=np.append(H1errs,a.H1error(u,dexact))
        
        pfit=np.polyfit(np.log10(hs),np.log10(np.sqrt(L2errs**2+H1errs**2)),1)
        self.assertTrue(pfit[0]>=1.95)
        self.assertTrue(pfit[0]<=2.05)

class TriPpTest(unittest.TestCase):
    """Test triangular h-refinement with various p.
    Also test facet assembly."""
    def runTest(self):

        def U(x):
            return 1+x[0]-x[0]**2*x[1]**2

        def dUdx(x):
            return 1-2*x[0]*x[1]**2

        def dUdy(x):
            return -2*x[0]**2*x[1]

        def dudv(du,dv):
            return du[0]*dv[0]+du[1]*dv[1]

        def uv(u,v):
            return u*v

        def F(x,y):
            return 2*x**2+2*y**2

        def fv(v,x):
            return F(x[0],x[1])*v

        def G(x,y):
            return (x==1)*(3-3*y**2)+\
                    (x==0)*(0)+\
                    (y==1)*(1+x-3*x**2)+\
                    (y==0)*(1+x)

        def gv(v,x):
            return G(x[0],x[1])*v

        dexact={}
        dexact[0]=dUdx
        dexact[1]=dUdy

        hs={}
        H1errs={}
        L2errs={}

        for p in range(1,4):
            mesh=fmsh.MeshTri()
            mesh.refine(1)
            hs[p-1]=np.array([])
            H1errs[p-1]=np.array([])
            L2errs[p-1]=np.array([])

            for itr in range(4):
                mesh.refine()

                a=fasm.AssemblerElement(mesh,fmap.MappingAffine,felem.ElementTriPp(p))

                A=a.iasm(dudv)
                f=a.iasm(fv)

                B=a.fasm(uv)
                g=a.fasm(gv)

                u=np.zeros(a.dofnum_u.N)
                u=spsolve(A+B,f+g)

                hs[p-1]=np.append(hs[p-1],mesh.param())
                L2errs[p-1]=np.append(L2errs[p-1],a.L2error(u,U))
                H1errs[p-1]=np.append(H1errs[p-1],a.H1error(u,dexact))

            pfit=np.polyfit(np.log10(hs[p-1]),np.log10(H1errs[p-1]),1)

            self.assertTrue(pfit[0]>=0.95*p)

class TetP1Test(unittest.TestCase):
    """Test tetrahedral refinements with P1 elements.
    Also tests assembly on tetrahedral facets."""
    def runTest(self):
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

        for itr in range(2,5):
            mesh=fmsh.MeshTet()
            mesh.refine(itr)

            a=fasm.AssemblerElement(mesh,fmap.MappingAffine,felem.ElementP1(3))

            A=a.iasm(dudv)
            f=a.iasm(fv)

            B=a.fasm(uv)
            g=a.fasm(gv)

            u=np.zeros(a.dofnum_u.N)

            u=spsolve(A+B,f+g)

            p={}
            p[0]=mesh.p[0,:]
            p[1]=mesh.p[1,:]
            p[2]=mesh.p[2,:]

            hs=np.append(hs,mesh.param())
            L2err=np.append(L2err,a.L2error(u,U))
            H1err=np.append(H1err,a.H1error(u,dexact))

        pfit=np.polyfit(np.log10(hs),np.log10(np.sqrt(L2err**2+H1err**2)),1)
        
        # check that the convergence rate matches theory
        self.assertTrue(pfit[0]>=1)