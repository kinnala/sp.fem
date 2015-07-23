import unittest
import fem.mesh
import fem.geometry
import fem.mapping
import numpy as np

class MappingAffineTriBasicTest(unittest.TestCase):
    def setUp(self):
        self.geom=fem.geometry.GeometryMeshTri()
        self.geom.refine()
        self.mesh=self.geom.mesh()

class MappingAffineTriFinvF(MappingAffineTriBasicTest):
    """
    Check that F(invF(x))===x
    """
    def runTest(self):
        mapping=fem.mapping.MappingAffineTri(self.mesh)
        y=mapping.F(np.array([[1,2,3],[1,2,3]]))
        X=mapping.invF(y)

        self.assertAlmostEqual(X[0][:,0].all(),1.0)

