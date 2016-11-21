"""
Import and generation of meshes.
"""
import numpy as np
import spfem.mesh
import platform
import meshpy.triangle
import meshpy.tet
from meshpy.geometry import GeometryBuilder

import os
import matplotlib.pyplot as plt

class Geometry(object):
    """Geometry contains metadata and outputs meshes."""

    def __init__(self):
        raise NotImplementedError("Geometry constructor not implemented!")

    def mesh(self):
        raise NotImplementedError("Geometry mesher not implemented!")

class GeometryTetGmshFile(Geometry):
    """A faster *.msh loader for tetrahedral meshes."""

    def __init__(self,filename):
        import mmap
        import contextlib
        import sys

        node_start=-1
        node_end=-1
        no_nodes=0
        elem_start=-1
        elem_end=-1
        no_elems=0
        ix=0

        # read the locations from the file
        with open(filename,'rb') as f:
            m=mmap.mmap(f.fileno(),0,prot=mmap.PROT_READ)
            line=m.readline()
            while line:
                ix=ix+1

                # find number of nodes/elems
                if ix==node_start+1:
                    no_nodes=int(line)
                elif ix==elem_start+1:
                    no_elems=int(line)

                # find starting/ending of node/elem list
                if line[0]=='$':
                    if line=='$Nodes\n':
                        node_start=ix
                    elif line=='$EndNodes\n':
                        node_end=ix
                    elif line=='$Elements\n':
                        elem_start=ix
                    elif line=='$EndElements\n':
                        elem_end=ix

                line=m.readline()

        self.points=np.genfromtxt(filename,usecols=(1,2,3),skip_header=node_start+1,skip_footer=ix-node_end+1)

        # silence stdout
        class Dummy(object):
            def write(self,x):
                pass

        @contextlib.contextmanager
        def nostderr():
            tmp=sys.stderr
            sys.stderr=Dummy()
            yield
            sys.stderr=tmp

        with nostderr():
            self.elems=np.genfromtxt(filename,usecols=(5,6,7,8),skip_header=elem_start+1,skip_footer=ix-elem_end+1,invalid_raise=False)

        self.elems-=1

    def mesh(self):
        return spfem.mesh.MeshTet(self.points.T,self.elems.T)

# The following code depends on MeshPy

class GeometryMeshPy(Geometry):
    """A geometry defined by MeshPy constructs."""

    def _mesh_output(self):
        if len(self.m.elements)==0:
            raise Exception("Empty mesh")
        p=np.array(self.m.points).T
        t=np.array(self.m.elements).T
        if p.shape[0]==3:
            return spfem.mesh.MeshTet(p,t)
        elif p.shape[0]==2:
            return spfem.mesh.MeshTri(p,t)
        else:
            raise NotImplementedError("The type of mesh not supported")

class GeometryMeshPyTetgen(GeometryMeshPy):
    """Define and mesh 3-dimensional domains with MeshPy/Tetgen."""

    def __init__(self):
        self.geob=GeometryBuilder()

    def mesh(self,h,holes=None):
        info=meshpy.tet.MeshInfo()
        self.geob.set(info)
        if holes is not None:
            info.set_holes(holes)
        self.m=meshpy.tet.build(info,max_volume=h**3)
        return self._mesh_output()

    def extrude(self,points,z):
        """A wrapper to self.advanced_extrude to create
        a simple extrusion of a cross section.
        
        Parameters
        ==========
        points : array of tuples
            An array of tuples with (x,y)-coordinates of the boundary
            points. The boundary points are connected by straight
            line segments. The last point is connected to the first
            point.
        z : float
            The extrusion length.
        """
        self.advanced_extrude(points,[(0.,0.),(1.,0.),(1.,z),(0.,z)])

    def advanced_extrude(self,points,rz):
        """Add a geometry defined by an extrusion.
        
        Parameters
        ==========
        points : array of tuples
            An array of tuples with (x,y)-coordinates of the boundary
            points. The boundary points are connected by straight
            line segments. The last point is connected to the first
            point.
        rz : array of tuples
            An array of tuples with (r,z)-coordinates. The first
            number of each tuple acts as a multiplier to increase
            or decrease the size of the cross section. The second
            number defines the z-location of the cross section.
        """
        self.geob.add_geometry(*meshpy.geometry.generate_extrusion(rz_points=rz,
            base_shape=points))

    def revolution(self,points,N,transform=None):
        """Generate a surface of revolution.

        Parameters
        ==========
        points : array of tuples
            An array of tuples with (x,y)-coordinates of the boundary
            points of the to-be-revolved surface. The boundary points
            are connected by straight line segments. The revolution is
            performed around y-axis. The first and the last points should
            be located at x=0.
        N : integer
            The number of subdivisions in the revolution.
        transform : (OPTIONAL) A function that takes list of 3-tuples
            and modifies the list somehow. Can be used to e.g. translate
            the points after revolving.
        """
        if transform is None:
            self.geob.add_geometry(*meshpy.geometry.generate_surface_of_revolution(points,
                closure=meshpy.geometry.EXT_OPEN,radial_subdiv=N))
        else:
            a,b,c,d=meshpy.geometry.generate_surface_of_revolution(points,
                    closure=meshpy.geometry.EXT_OPEN,radial_subdiv=N)
            A=[transform(*x) for x in a]
            self.geob.add_geometry(A,b,c,d)


class GeometryMeshPyTriangle(GeometryMeshPy):
    """Define and mesh 2-dimensional domains with MeshPy/Triangle."""

    def __init__(self, points, facets=None, holes=None):
        """Define a domain using boundary segments (PLSG).
        
        Default behavior is to connect all points.
        
        Parameters
        ==========
        points : array of tuples
            The list of points that form the boundarys.
        facets : (OPTIONAL) array of tuples
            The list of indices to points that define the boundary segments.
        holes : (OPTIONAL) array of tuples
            The list of points that define the holes.
        """
        self.info = meshpy.triangle.MeshInfo()
        self.info.set_points(points)
        if holes is not None:
            self.info.set_holes(holes)
        if facets is None:
            self.info.set_facets([(i, i+1) for i in range(0, len(points) - 1)]
                                 + [(len(points) - 1, 0)])
        else:
            self.info.set_facets(facets)

    def mesh(self,h):
        def ref_func(tri_points, area):
            return bool(area > h*h)
        self.m = meshpy.triangle.build(self.info, refinement_func=ref_func)
        return self._mesh_output()

    def refine(self, ref_ts):
        p = np.array(self.m.points).T
        t = np.array(self.m.elements).T

        # define new points and stack
        points = np.hstack((p, .5*(p[:, t[0, ref_ts]] + p[:, t[1, ref_ts]]),
                               .5*(p[:, t[1, ref_ts]] + p[:, t[2, ref_ts]]),
                               .5*(p[:, t[0, ref_ts]] + p[:, t[2, ref_ts]]))).T

        # build new mesh
        self.info.set_points(np.array(points))
        self.info.set_facets(np.array(self.m.facets))
        self.m = meshpy.triangle.build(self.info, allow_volume_steiner=False,
                                       allow_boundary_steiner=False)
        return self._mesh_output()


