import unittest
import fem.asm
import fem.geometry
import numpy as np
import scipy.sparse.linalg

class AssemblerTriP1BasicTest(unittest.TestCase):
    def setUp(self):
        geom=fem.geometry.GeometryMeshTri()
        geom.refine(5)
        self.mesh=geom.mesh()

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
    """
    Simple Poisson test. Solving $-\Delta u = 1$ in an unit square
    with $u=0$ on the boundary.
    """
    def runTest(self):
        bilin=lambda u,v,du,dv,x: du[0]*dv[0]+du[1]*dv[1]
        lin=lambda v,dv,x: 1*v

        a=fem.asm.AssemblerTriP1(self.mesh)

        A=a.iasm(bilin)
        f=a.iasm(lin)

        x=np.zeros(A.shape[0])
        I=self.I
        x[I]=scipy.sparse.linalg.spsolve(A[np.ix_(I,I)],f[I])

        self.assertAlmostEqual(np.max(x),0.073614737354524146)

class AssemblerTriP1FullPoisson(AssemblerTriP1BasicTest):
    """
    Poisson test from Huhtala's MATLAB package.
    TODO add equation and bc's.
    """
    def runTest(self):
        f=lambda x,y: 100.0*((x>=0.4)&(x<=0.6)&(y>=0.4)&(y<=0.6))
        g=lambda x,y: (y==0)*1.0+(y==1)*(-1.0)

        a=fem.asm.AssemblerTriP1(self.mesh)

        dudv=lambda u,v,du,dv,x: du[0]*dv[0]+du[1]*dv[1]
        K=a.iasm(dudv)

        uv=lambda u,v,du,dv,x: u*v
        B=a.fasm(uv)
        
        fv=lambda v,dv,x: f(x[0],x[1])*v
        f=a.iasm(fv)

        gv=lambda v,dv,x: g(x[0],x[1])*v
        g=a.fasm(gv)

        D=np.nonzero(self.mesh.p[0,:]==0)[0]
        I=np.setdiff1d(np.arange(0,self.mesh.p.shape[1]),D)

        x=np.zeros(K.shape[0])
        x[I]=scipy.sparse.linalg.spsolve(K[np.ix_(I,I)]+B[np.ix_(I,I)],f[I]+g[I])

        self.assertAlmostEqual(np.max(x),2.064287505122583)
