import unittest
import fem.mesh
import fem.geometry
import fem.mapping
import numpy as np

class MappingAffineBasicTest(unittest.TestCase):
    def setUp(self):
        self.mesh=fem.mesh.MeshTri()
        self.mesh.refine()

class MappingAffineFinvF(MappingAffineBasicTest):
    """Check that F(invF(x))===x"""
    def runTest(self):
        mapping=fem.mapping.MappingAffine(self.mesh)
        y=mapping.F(np.array([[1,2,3],[1,2,3]]))
        X=mapping.invF(y)

        self.assertAlmostEqual(X[0][:,0].all(),1.0)

