# -*- coding: utf-8 -*-
"""
Module level tests; tests that require multiple working classes.
"""
import unittest
import spfem.mesh as fmsh
import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse as spsp
import spfem.asm as fasm
import spfem.mapping as fmap
import spfem.element as felem
import spfem.utils as futil
import matplotlib.pyplot as plt
import copy
from spfem.weakform import *

class RT0Test(unittest.TestCase):
    """Assemble and solve mixed Poisson equation
    using the first-order Raviart-Thomas element."""
    def runTest(self):
        mesh=fmsh.MeshTri()
        mesh.refine(2)
        
        a=fasm.AssemblerElement(mesh,felem.ElementTriRT0())
        b=fasm.AssemblerElement(mesh,felem.ElementTriRT0(),felem.ElementP0())
        c=fasm.AssemblerElement(mesh,felem.ElementP0())
        
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
            a=fasm.AssemblerElement(mesh,felem.ElementTriRT0())
            b=fasm.AssemblerElement(mesh,felem.ElementTriRT0(),felem.ElementP0())
            c=fasm.AssemblerElement(mesh,felem.ElementP0())
            
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
        
            a=fasm.AssemblerElement(mesh,felem.ElementTetP2())
        
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
        
            a=fasm.AssemblerElement(mesh,felem.ElementQ1())
        
            A=a.iasm(dudv)
            f=a.iasm(fv)
        
            B=a.fasm(uv,normals=False)
            g=a.fasm(gv,normals=False)
        
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
        
            a=fasm.AssemblerElement(mesh,felem.ElementQ2())
        
            A=a.iasm(dudv)
            f=a.iasm(fv)
        
            B=a.fasm(uv,normals=False)
            g=a.fasm(gv,normals=False)
        
            u=np.zeros(a.dofnum_u.N)
            u=spsolve(A+B,f+g)
        
        
            hs=np.append(hs,mesh.param())
            L2errs=np.append(L2errs,a.L2error(u,U))
            H1errs=np.append(H1errs,a.H1error(u,dexact))
        
        pfit=np.polyfit(np.log10(hs),np.log10(np.sqrt(L2errs**2+H1errs**2)),1)
        self.assertTrue(pfit[0]>=1.95)
        self.assertTrue(pfit[0]<=2.05)

class TriP2Test(unittest.TestCase):
    """Test triangular h-refinement.
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

        mesh=fmsh.MeshTri()
        mesh.refine(1)
        hs=np.array([])
        H1errs=np.array([])
        L2errs=np.array([])

        for itr in range(4):
            mesh.refine()

            a=fasm.AssemblerElement(mesh,felem.ElementTriP2())

            A=a.iasm(dudv)
            f=a.iasm(fv)

            B=a.fasm(uv)
            g=a.fasm(gv)

            u=np.zeros(a.dofnum_u.N)
            u=spsolve(A+B,f+g)

            hs=np.append(hs,mesh.param())
            L2errs=np.append(L2errs,a.L2error(u,U))
            H1errs=np.append(H1errs,a.H1error(u,dexact))

        pfit=np.polyfit(np.log10(hs),np.log10(H1errs),1)

        self.assertGreater(pfit[0],0.95*2)

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

        for p in range(1,3):
            mesh=fmsh.MeshTri()
            mesh.refine(1)
            hs[p-1]=np.array([])
            H1errs[p-1]=np.array([])
            L2errs[p-1]=np.array([])

            for itr in range(4):
                mesh.refine()

                a=fasm.AssemblerElement(mesh,felem.ElementTriPp(p))

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

            a=fasm.AssemblerElement(mesh,felem.ElementTetP1())

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

class AssemblerTriP1Nitsche(unittest.TestCase):
    """Solve Poisson with Nitsche approximating Dirichlet BC."""
    def runTest(self):
        mesh=fmsh.MeshTri()
        mesh.refine(2)

        def G(x,y):
            return x**3*np.sin(20.0*y)

        def U(x):
            return G(x[0],x[1])

        def dUdx(x):
            return 3.0*x[0]**2*np.sin(20.0*x[1])

        def dUdy(x):
            return 20.0*x[0]**3*np.cos(20.0*x[1])

        dU={0:dUdx,1:dUdy}

        def dudv(du,dv):
            return du[0]*dv[0]+du[1]*dv[1]
            
        gamma=100

        def uv(u,v,du,dv,x,h,n):
            return gamma*1/h*u*v-du[0]*n[0]*v-du[1]*n[1]*v-u*dv[0]*n[0]-u*dv[1]*n[1]
            
        def fv(v,dv,x):
            return (-6.0*x[0]*np.sin(20.0*x[1])+400.0*x[0]**3*np.sin(20.0*x[1]))*v
            
        def gv(v,dv,x,h,n):
            return G(x[0],x[1])*v+gamma*1/h*G(x[0],x[1])*v-dv[0]*n[0]*G(x[0],x[1])-dv[1]*n[1]*G(x[0],x[1])

        hs=np.array([])
        errs=np.array([])

        for itr in range(4):
            mesh.refine()
            a=fasm.AssemblerElement(mesh,felem.ElementTriP1())
            D=mesh.boundary_nodes()
            I=mesh.interior_nodes()

            K=a.iasm(dudv)
            B=a.fasm(uv,normals=True)
            f=a.iasm(fv)
            g=a.fasm(gv,normals=True)

            x=np.zeros(K.shape[0])
            x=spsolve(K+B,f+g)

            hs=np.append(hs,mesh.param())
            errs=np.append(errs,np.sqrt(a.L2error(x,U)**2+a.H1error(x,dU)**2))

        pfit=np.polyfit(np.log10(hs),np.log10(errs),1)
        
        # check that the convergence rate matches theory
        self.assertTrue(pfit[0]>=0.99)

class ExamplePoisson(unittest.TestCase):
    """Tetrahedral refinements with P1 and P2 elements."""
    def runTest(self,verbose=False):
        # define data
        def F(x,y,z):
            return 2*x**2+2*y**2-6*x*y*z
            
        def G(x,y,z):
            return (x==1)*(3-3*y**2+2*y*z**3)+\
                   (x==0)*(-y*z**3)+\
                   (y==1)*(1+x-3*x**2+2*x*z**3)+\
                   (y==0)*(1+x-x*z**3)+\
                   (z==1)*(1+x+4*x*y-x**2*y**2)+\
                   (z==0)*(1+x-x**2*y**2)

        # bilinear and linear forms of the problem
        def dudv(du,dv):
            return du[0]*dv[0]+du[1]*dv[1]+du[2]*dv[2]

        def uv(u,v):
            return u*v

        def fv(v,x):
            return F(x[0],x[1],x[2])*v

        def gv(v,x):
            return G(x[0],x[1],x[2])*v

        # analytical solution and its derivatives            
        def exact(x):
            return 1+x[0]-x[0]**2*x[1]**2+x[0]*x[1]*x[2]**3

        dexact={}
        dexact[0]=lambda x:1-2*x[0]*x[1]**2+x[1]*x[2]**3
        dexact[1]=lambda x:-2*x[0]**2*x[1]+x[0]*x[2]**3
        dexact[2]=lambda x:3*x[0]*x[1]*x[2]**2

        # initialize arrays for saving errors
        hs1=np.array([])
        hs2=np.array([])
        
        # P1 element
        H1err1=np.array([])
        L2err1=np.array([])
        
        # P2 element
        H1err2=np.array([])
        L2err2=np.array([])

        # create the mesh; by default a box [0,1]^3 is meshed
        mesh=fmsh.MeshTet()
        mesh.refine()

        # loop over mesh refinement levels
        for itr in range(3):
            # compute with P2 element
            b=fasm.AssemblerElement(mesh,felem.ElementTetP2())
            
            # assemble the matrices and vectors related to P2
            A2=b.iasm(dudv)
            f2=b.iasm(fv)

            B2=b.fasm(uv)
            g2=b.fasm(gv)

            # initialize the solution vector and solve            
            u2=np.zeros(b.dofnum_u.N)
            u2=spsolve(A2+B2,f2+g2)
            
            # compute error of the P2 element
            hs2=np.append(hs2,mesh.param())
            L2err2=np.append(L2err2,b.L2error(u2,exact))
            H1err2=np.append(H1err2,b.H1error(u2,dexact))

            # refine mesh once
            mesh.refine()

            # create a finite element assembler = mesh + mapping + element
            a=fasm.AssemblerElement(mesh,felem.ElementTetP1())

            # assemble the matrices and vectors related to P1
            A1=a.iasm(dudv)
            f1=a.iasm(fv)

            B1=a.fasm(uv)
            g1=a.fasm(gv)

            # initialize the solution vector and solve
            u1=np.zeros(a.dofnum_u.N)
            u1=spsolve(A1+B1,f1+g1)

            # compute errors and save them
            hs1=np.append(hs1,mesh.param())     
            L2err1=np.append(L2err1,a.L2error(u1,exact))
            H1err1=np.append(H1err1,a.H1error(u1,dexact))

        # create a linear fit on logarithmic scale
        pfit1=np.polyfit(np.log10(hs1),np.log10(np.sqrt(L2err1**2+H1err1**2)),1)
        pfit2=np.polyfit(np.log10(hs2),np.log10(np.sqrt(L2err2**2+H1err2**2)),1)
        
        if verbose:
            print "Convergence rate with P1 element: "+str(pfit1[0])
            plt.loglog(hs1,np.sqrt(L2err1**2+H1err1**2),'bo-')
            print "Convergence rate with P2 element: "+str(pfit2[0])
            plt.loglog(hs2,np.sqrt(L2err2**2+H1err2**2),'ro-')
        
        # check that convergence rates match theory
        self.assertTrue(pfit1[0]>=1)
        self.assertTrue(pfit2[0]>=2)

class ExampleElasticity(unittest.TestCase):
    """Solving the linear elasticity equations (stress) in 3D box
    and comparing to a manufactured analytical solution."""
    def runTest(self,verbose=False):
        U=TensorFunction(dim=3,torder=1)
        V=TensorFunction(dim=3,torder=1,sym='v')

        # infinitesimal strain tensor
        def Eps(W):
            return 0.5*(grad(W)+grad(W).T())

        # material parameters: Young's modulus and Poisson ratio
        E=20
        Nu=0.3

        # Lame parameters
        Lambda=E*Nu/((1+Nu)*(1-2*Nu))
        Mu=E/(2*(1+Nu))

        # definition of the stress tensor
        def Sigma(W):
            return 2*Mu*Eps(W)+Lambda*div(W)*IdentityMatrix(3)

        # define the weak formulation
        dudv=dotp(Sigma(U),Eps(V))
        dudv=dudv.handlify(verbose=verbose)

        # generate a mesh in the box
        m=fmsh.MeshTet()
        m.refine(4)

        # define the vectorial element
        e=felem.ElementH1Vec(felem.ElementTetP1())
        # create a FE-assembler object
        a=fasm.AssemblerElement(m,e)
        # assemble the stiffness matrix
        A=a.iasm(dudv)

        # define analytical solution and compute loading corresponding to it
        def exact(x):
            return -0.1*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])*np.sin(np.pi*x[2])

        def loading():
            global exactvonmises
            import sympy as sp
            x,y,z=sp.symbols('x y z')
            Ut=ConstantTensor(0.0,dim=3,torder=1)
            Ut.expr[0]=0*x
            Ut.expr[1]=0*x
            Ut.expr[2]=0.1*sp.sin(sp.pi*x)*sp.sin(sp.pi*y)*sp.sin(sp.pi*z)

            return dotp(div(Sigma(Ut)),V).handlify(verbose=verbose)

        # assemble the load vector
        f=a.iasm(loading())

        i1=m.interior_nodes()
        I=a.dofnum_u.getdofs(N=i1)

        # solve the system
        u=futil.direct(A,f,I=I,use_umfpack=True)

        e1=felem.ElementTetP1()
        c=fasm.AssemblerElement(m,e1)

        L2err=c.L2error(u[a.dofnum_u.n_dof[2,:]],exact)

        self.assertTrue(c.L2error(u[a.dofnum_u.n_dof[2,:]],exact)<=1e-3)

        if verbose:
            # displaced mesh for drawing
            mdefo=copy.deepcopy(m)
            mdefo.p[0,:]+=u[a.dofnum_u.n_dof[0,:]]
            mdefo.p[1,:]+=u[a.dofnum_u.n_dof[1,:]]
            mdefo.p[2,:]+=u[a.dofnum_u.n_dof[2,:]]

            # project von mises stress to scalar P1 element
            V=TensorFunction(dim=3,torder=0,sym='v')
            b=fasm.AssemblerElement(m,e,e1)

            S=Sigma(U)
            def vonmises(s):
                return np.sqrt(0.5*((s[0,0]-s[1,1])**2+(s[1,1]-s[2,2])**2+(s[2,2]-s[0,0])**2+\
                       6.0*(s[1,2]**2+s[2,0]**2+s[0,1]**2)))

            M=c.iasm(lambda u,v:u*v)

            # compute each component of stress tensor
            StressTensor={}
            for itr in range(3):
                for jtr in range(3):
                    duv=(S[itr,jtr]*V).handlify(verbose=True)
                    P=b.iasm(duv)
                    StressTensor[(itr,jtr)]=fsol.direct(M,P*u,use_umfpack=True)

            # draw the von Mises stress
            mdefo.draw(u=vonmises(StressTensor),test=lambda x,y,z: x>=0.5)


class TestAbstractMorley(unittest.TestCase):
    """Solve biharmonic problem with Morley elements."""
    def runTest(self,verbose=False):
        m=fmsh.MeshTri()
        m.refine(7)

        e=felem.AbstractElementMorley()
        a=fasm.AssemblerAbstract(m,e)

        A=a.iasm(lambda ddu,ddv: ddu[0][0]*ddv[0][0]+ddu[1][0]*ddv[1][0]+
                                 ddu[0][1]*ddv[0][1]+ddu[1][1]*ddv[1][1])

        # construct a loading that corresponds to the analytical solution uex
        import sympy as sp
        from sympy.abc import x,y,z
        uex=(sp.sin(sp.pi*x)*sp.sin(sp.pi*y))**2
        uexfun=sp.lambdify((x,y),uex,"numpy")
        load=sp.diff(uex,x,4)+sp.diff(uex,y,4)+2*sp.diff(sp.diff(uex,x,2),y,2)
        loadfun=sp.lambdify((x,y),load,"numpy")
        def F(x):
            return loadfun(x[0],x[1])

        f=a.iasm(lambda v,x: F(x)*v)

        D=a.dofnum_u.getdofs(N=m.boundary_nodes(),F=m.boundary_facets())
        I=np.setdiff1d(np.arange(a.dofnum_u.N),D)

        x=futil.direct(A,f,I=I)

        Linferror=np.max(np.abs(x[a.dofnum_u.n_dof[0,:]]-uexfun(m.p[0,:],m.p[1,:])))

        self.assertTrue(Linferror<=1e-3)

        # TODO convergence rates not checked

        if verbose:
            print np.max(np.abs(x[a.dofnum_u.n_dof[0,:]]-uexfun(m.p[0,:],m.p[1,:])))
            m.plot3(x[a.dofnum_u.n_dof[0,:]])
            m.plot3(uexfun(m.p[0,:],m.p[1,:]))
            m.show()


class TestAbstractArgyris(unittest.TestCase):
    """Solve biharmonic problem with Argyris elements."""
    def runTest(self,verbose=False):
        m=fmsh.MeshTri()
        m.refine(6)

        e=felem.AbstractElementArgyris()
        a=fasm.AssemblerAbstract(m,e)

        A=a.iasm(lambda ddu,ddv: ddu[0][0]*ddv[0][0]+ddu[1][0]*ddv[1][0]+
                                 ddu[0][1]*ddv[0][1]+ddu[1][1]*ddv[1][1])

        # construct a loading that corresponds to the analytical solution uex
        import sympy as sp
        from sympy.abc import x,y,z
        uex=(sp.sin(sp.pi*x)*sp.sin(sp.pi*y))**2
        uexfun=sp.lambdify((x,y),uex,"numpy")
        load=sp.diff(uex,x,4)+sp.diff(uex,y,4)+2*sp.diff(sp.diff(uex,x,2),y,2)
        loadfun=sp.lambdify((x,y),load,"numpy")
        def F(x):
            return loadfun(x[0],x[1])

        f=a.iasm(lambda v,x: F(x)*v)

        D=a.dofnum_u.getdofs(N=m.boundary_nodes(),F=m.boundary_facets())
        I=np.setdiff1d(np.arange(a.dofnum_u.N),D)

        x=futil.direct(A,f,I=I)

        Linferror=np.max(np.abs(x[a.dofnum_u.n_dof[0,:]]-uexfun(m.p[0,:],m.p[1,:])))

        self.assertTrue(Linferror<=5e-3)

