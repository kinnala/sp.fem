import unittest
import spfem.geometry as fegeom
import numpy as np

class GeometryTriangle2DMeshABoxTest(unittest.TestCase):
    """Check that a mesh with correct mesh parameter is given by
    the GeometryTriangle2D.mesh() method."""
    def runTest(self):
        g=fegeom.GeometryTriangle2D()
        g.add_rectangle()

        for itr in [0.1,0.05,0.01]:
            mesh=g.mesh(itr)

            dx=mesh.p[0,mesh.facets[0,:]]-mesh.p[0,mesh.facets[1,:]]
            dy=mesh.p[1,mesh.facets[0,:]]-mesh.p[1,mesh.facets[1,:]]
            dl=np.sqrt(dx**2+dy**2)
            
            # the number here is tand(20)
            # because default maximum angle is 20
            self.assertTrue(np.max(dl)/2.*np.sqrt(0.36397023426)<=itr)

class GeometryTriangle2DMeshWithHole(unittest.TestCase):
    """Mesh a box with a circular hole."""
    def runTest(self):
        g=fegeom.GeometryTriangle2D()
        g.add_rectangle(x=-1,y=-1,width=2,height=2)
        g.add_circle(r=0.5,nodes=np.linspace(0,2*np.pi,20))
        g.add_hole((0.0,0.0))

        mesh=g.mesh(0.05)

        #mesh.draw()
        #mesh.show()
        self.assertTrue(np.all(mesh.p[0,:]**2+mesh.p[1,:]**2>=0.49**2))


class GeometryTriangle2DTriangleMarkerTest(unittest.TestCase):
    """Create multiple regions and holes and check that all
    triangles are covered by the markers."""
    def runTest(self):
        g=fegeom.GeometryTriangle2D()
        g.add_rectangle()
        g.add_circle((0.25,0.25),0.1,marker='bl')
        g.add_circle((0.75,0.75),0.1,marker='ur')
        g.add_circle((0.25,0.75),0.1,marker='ul')
        g.add_circle((0.75,0.25),0.1,marker='br')

        g.add_hole((0.75,0.25))
        g.add_region((0.25,0.25),h=0.01,marker='circ1')
        g.add_region((0.25,0.75),h=0.01,marker='circ2')
        g.add_region((0.75,0.75),h=0.01,marker='circ3')
        g.add_region((0.5,0.5),h=0.01,marker='rest')

        mesh=g.mesh(0.02)

        #mesh.draw()
        #mesh.draw_nodes('bl',mark='bo')
        #mesh.draw_nodes('ur',mark='bx')
        #mesh.draw_nodes('ul',mark='rx')
        #mesh.draw_nodes('br',mark='ro')
        #mesh.show() 

        nt=np.union1d(np.union1d(np.union1d(g.tmarkers['circ1'],\
                                            g.tmarkers['circ2']),\
                                            g.tmarkers['circ3']),\
                                            g.tmarkers['rest'])


        self.assertTrue(nt.shape[0]==mesh.t.shape[1])
