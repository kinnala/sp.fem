import unittest
import fem.mesh
import fem.geometry
import numpy as np
import copy

# TODO add tests for MeshTet

class MeshTriBasicTest(unittest.TestCase):
    def setUp(self):
        self.mesh=fem.mesh.MeshTri()
        self.mesh.refine()

class MeshTriSanityCheck(MeshTriBasicTest):
    """Perform a sanity check on the indexing, etc."""
    def runTest(self):
        mesh=copy.deepcopy(self.mesh)
        mesh.refine(4)
        
        # check that maximum vertex index in mesh.t exists in mesh.p
        self.assertEqual(np.max(mesh.t),mesh.p.shape[1]-1)

        # TODO add uniqueness test

class MeshTriFacetIndexing(MeshTriBasicTest):
    """Test facet indexing"""
    def runTest(self):
        mesh=copy.deepcopy(self.mesh)
        mesh.refine(2)

        # test that f2t is the inverse of t2f at least in some sense
        t2f2t=np.equal(mesh.t2f[:,mesh.f2t[0,:]],np.tile(np.arange(mesh.facets.shape[1]),(3,1))).astype(np.intp)
        self.assertEqual(np.sum(t2f2t),mesh.facets.shape[1])

        # test that repeatedly doing t2f and f2t starting from some
        # triangle reaches the whole mesh eventually
        curts=np.array([4])
        for itr in range(15):
            toaddts=np.array([])
            for jtr in curts:
                if jtr!=-1:
                    fs=mesh.t2f[:,jtr]
                    newts=np.unique(mesh.f2t[:,fs].flatten())
                    toaddts=np.append(toaddts,newts)
            curts=np.append(curts,toaddts)
            curts=np.unique(curts)
        self.assertEqual(curts.shape[0]-1,mesh.t.shape[1])
                 



