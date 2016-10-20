import unittest
import spfem.asm
import spfem.mesh as fmsh
import numpy as np
import scipy.sparse.linalg
import scipy.sparse as spsp
import spfem.asm as fasm
import spfem.mapping as fmap
import spfem.element as felem
import matplotlib.pyplot as plt

# TODO AssemblerGlobal: compute derivative of P1
# through projection and compare to utils.gradient.

class AssemblerGlobalP2Comparison(unittest.TestCase):
    """Build some matrices with AssemblerGlobal
    and AssemblerElement. Compare the results."""
    def runTest(self):
        m=fmsh.MeshTri()
        m.refine(5)

        a=fasm.AssemblerGlobal(m,felem.ElementGlobalTriP2())
        b=fasm.AssemblerElement(m,felem.ElementTriP2())

        A=a.iasm(lambda u,v: u*v)
        B=b.iasm(lambda u,v: u*v)

        self.assertAlmostEqual(np.sum(A.data),np.sum(B.data),places=10)

        C=a.iasm(lambda du,v: du[0]*v)
        D=b.iasm(lambda du,v: du[0]*v)

        self.assertAlmostEqual(C.data[0],D.data[0],places=10)

class AssemblerGlobalP1Comparison(unittest.TestCase):
    """Build some matrices with AssemblerGlobal
    and AssemblerElement. Compare the results."""
    def runTest(self):
        m=fmsh.MeshTri()
        m.refine(5)

        a=fasm.AssemblerGlobal(m,felem.ElementGlobalTriP1())
        b=fasm.AssemblerElement(m,felem.ElementTriP1())

        A=a.iasm(lambda u,v: u*v)
        B=b.iasm(lambda u,v: u*v)

        self.assertAlmostEqual(np.sum(A.data),np.sum(B.data),places=10)

        C=a.iasm(lambda du,v: du[0]*v)
        D=b.iasm(lambda du,v: du[0]*v)

        self.assertAlmostEqual(C.data[0],D.data[0],places=10)

class AssemblerTriP1BasicTest(unittest.TestCase):
    def setUp(self):
        self.mesh=fmsh.MeshTri()
        self.mesh.refine(5)

        # boundary and interior node sets
        D1=np.nonzero(self.mesh.p[0,:]==0)[0]
        D2=np.nonzero(self.mesh.p[1,:]==0)[0]
        D3=np.nonzero(self.mesh.p[0,:]==1)[0]
        D4=np.nonzero(self.mesh.p[1,:]==1)[0]

        D=np.union1d(D1,D2);
        D=np.union1d(D,D3);
        self.D=np.union1d(D,D4);

        self.I=np.setdiff1d(np.arange(0,self.mesh.p.shape[1]),self.D)

class AssemblerTriP1Poisson(AssemblerTriP1BasicTest):
    """Simple Poisson test.
    
    Solving $-\Delta u = 1$ in an unit square with $u=0$ on the boundary.
    """
    def runTest(self):
        bilin=lambda u,v,du,dv,x,h: du[0]*dv[0]+du[1]*dv[1]
        lin=lambda v,dv,x,h: 1*v

        a=fasm.AssemblerElement(self.mesh,felem.ElementTriP1())

        A=a.iasm(bilin)
        f=a.iasm(lin)

        x=np.zeros(A.shape[0])
        I=self.I
        x[I]=scipy.sparse.linalg.spsolve(A[np.ix_(I,I)],f[I])

        self.assertAlmostEqual(np.max(x),0.073614737354524146)

class AssemblerTriP1AnalyticWithXY(AssemblerTriP1BasicTest):
    """Poisson test case with analytic solution.

    Loading f=sin(pi*x)*sin(pi*y) and u=0 on the boundary.
    """
    def runTest(self):
        I=self.I
        D=self.D

        a=fasm.AssemblerElement(self.mesh,felem.ElementTriP1())

        def dudv(du,dv):
            return du[0]*dv[0]+du[1]*dv[1]
        K=a.iasm(dudv)

        def fv(v,x):
                return 2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])*v
        f=a.iasm(fv)


        x=np.zeros(K.shape[0])
        x[I]=scipy.sparse.linalg.spsolve(K[np.ix_(I,I)],f[I])

        def truex():
            X=self.mesh.p[0,:]
            Y=self.mesh.p[1,:]
            return np.sin(np.pi*X)*np.sin(np.pi*Y)

        self.assertAlmostEqual(np.max(x-truex()),0.0,places=3)



class AssemblerTriP1FullPoisson(AssemblerTriP1BasicTest):
    """Poisson test from Huhtala's MATLAB package."""
    def runTest(self):
        F=lambda x,y: 100.0*((x>=0.4)&(x<=0.6)&(y>=0.4)&(y<=0.6))
        G=lambda x,y: (y==0)*1.0+(y==1)*(-1.0)

        a=fasm.AssemblerElement(self.mesh,felem.ElementTriP1())

        dudv=lambda du,dv: du[0]*dv[0]+du[1]*dv[1]
        K=a.iasm(dudv)

        uv=lambda u,v: u*v
        B=a.fasm(uv)
        
        fv=lambda v,x: F(x[0],x[1])*v
        f=a.iasm(fv)

        gv=lambda v,x: G(x[0],x[1])*v
        g=a.fasm(gv)

        D=np.nonzero(self.mesh.p[0,:]==0)[0]
        I=np.setdiff1d(np.arange(0,self.mesh.p.shape[1]),D)

        x=np.zeros(K.shape[0])
        x[I]=scipy.sparse.linalg.spsolve(K[np.ix_(I,I)]+B[np.ix_(I,I)],
                                         f[I]+g[I])

        self.assertAlmostEqual(np.max(x),1.89635971369,places=2)

class AssemblerTriP1Interp(AssemblerTriP1BasicTest):
    """Compare A*u with f(u) for some u,
    where A is the mass matrix and f(u)=(u,v)."""
    def runTest(self):
        mesh=self.mesh
        a=fasm.AssemblerElement(self.mesh,felem.ElementTriP1())
        D=self.D
        I=self.I

        def G(x,y):
            return np.sin(np.pi*x)
        u=G(mesh.p[0,:],mesh.p[1,:])

        # interpolate just the values and compare
        def v1(v,w):
            return w[0]*v
        def v2(u,v):
            return u*v
        f=a.iasm(v1,interp={0:u})
        A=a.iasm(v2)

        self.assertAlmostEqual(np.linalg.norm(f-A*u),0.0,places=10)

class AssemblerTriP1FacetInterp(unittest.TestCase):
    """Compare M*u with f(u)."""

    def runTest(self):
        m=fmsh.MeshTri()
        m.refine(3)
        a=fasm.AssemblerElement(m,felem.ElementTriP1())

        def G(x,y):
            return np.sin(np.pi*x)
        u=G(m.p[0,:],m.p[1,:])

        # interpolate just the values and compare
        def v1(v,w):
            return w[0]*v
        def v2(u,v):
            return u*v
        f=a.fasm(v1,interp={0:u})
        A=a.fasm(v2)

        self.assertAlmostEqual(np.linalg.norm(f-A*u),0.0,places=10)




class AssemblerTriSubset(unittest.TestCase):
    """Test the subset assembly."""
    def runTest(self):
        m=fmsh.MeshTri()
        m.refine(4)
        # split mesh into two sets of triangles
        I1=np.arange(m.t.shape[1]/2)
        I2=np.setdiff1d(np.arange(m.t.shape[1]),I1)

        bix=m.boundary_facets()
        bix1=bix[0:len(bix)/2]
        bix2=np.setdiff1d(bix,bix1)

        a=fasm.AssemblerElement(m,felem.ElementTriP1())

        def dudv(du,dv):
            return du[0]*dv[0]+du[1]*dv[1]

        A=a.iasm(dudv)
        A1=a.iasm(dudv,tind=I1)
        A2=a.iasm(dudv,tind=I2)

        B=a.fasm(dudv)
        B1=a.fasm(dudv,find=bix1)
        B2=a.fasm(dudv,find=bix2)

        f=a.iasm(lambda v: 1*v)

        I=m.interior_nodes()

        x=np.zeros(A.shape[0])
        x[I]=scipy.sparse.linalg.spsolve((A+B)[I].T[I].T,f[I])

        X=np.zeros(A.shape[0])
        X[I]=scipy.sparse.linalg.spsolve((A1+B1)[I].T[I].T+(A2+B2)[I].T[I].T,f[I])

        self.assertAlmostEqual(np.linalg.norm(x-X),0.0,places=10)

