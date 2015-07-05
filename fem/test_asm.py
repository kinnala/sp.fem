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
