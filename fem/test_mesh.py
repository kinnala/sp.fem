import unittest
import fem.mesh
import fem.geometry
import numpy as np

class MeshBasicTest(unittest.TestCase):
    def setUp(self):
        self.geom=fem.geometry.GeometryMeshTri()

class MeshFacetIndexing(MeshBasicTest):
    """
    Compare the facet indexing to a known test case.
    """
    def runTest(self):
        geom=self.geom
        geom.refine(1)
        mesh=geom.mesh()

        self.assertAlmostEqual(mesh.p[1,-2],0.5)

        self.assertAlmostEqual(mesh.f2t[1,-1],7)
        self.assertAlmostEqual(mesh.f2t[1,-2],7)
        self.assertAlmostEqual(mesh.f2t[1,-3],1)
        self.assertAlmostEqual(mesh.f2t[1,-7],-1)

        self.assertAlmostEqual(mesh.t2f[2,-1],14)
        self.assertAlmostEqual(mesh.t2f[2,-2],11)
        self.assertAlmostEqual(mesh.t2f[2,-3],9)
