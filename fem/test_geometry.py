import unittest
import fem.geometry as fegeom
import numpy as np

class GeometryShapely2DMeshABoxTest(unittest.TestCase):
    """Check that a mesh with correct mesh parameter is given by
    the GeometryShapely2D.mesh() method."""
    def runTest(self):
        geomlist=[('+','box',0,0,1.0,1.0)]
        g=fegeom.GeometryShapely2D(geomlist)

        for itr in [0.1,0.05,0.01]:
            mesh=g.mesh(itr)

            dx=mesh.p[0,mesh.facets[0,:]]-mesh.p[0,mesh.facets[1,:]]
            dy=mesh.p[1,mesh.facets[0,:]]-mesh.p[1,mesh.facets[1,:]]
            dl=np.sqrt(dx**2+dy**2)
            
            # the number here is tand(20)
            # because default maximum angle is 20
            self.assertTrue(np.max(dl)/2.*np.sqrt(0.36397023426)<=itr)

class GeometryShapely2DMeshWithHole(unittest.TestCase):
    """Mesh a box with a circular hole."""
    def runTest(self):
        geomlist=[
                ('+','box',-1,-1,1,1),
                ('-','circle',0,0,0.5,20)
                ]
        holes=[(0,0)]

        g=fegeom.GeometryShapely2D(geomlist,holes=holes)

        mesh=g.mesh(0.05)

        #mesh.draw()
        #mesh.show()
        self.assertTrue(np.all(mesh.p[0,:]**2+mesh.p[1,:]**2>=0.49**2))


        
