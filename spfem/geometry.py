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

# The following code depends on MeshPy

class GeometryMeshPy(Geometry):
    """A geometry defined by MeshPy constructs."""
    pass

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

    def _mesh_output(self):
        p=np.array(self.m.points).T
        t=np.array(self.m.elements).T
        return spfem.mesh.MeshTet(p,t)

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

    def revolution(self,points,N):
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
        """
        self.geob.add_geometry(*meshpy.geometry.generate_surface_of_revolution(points,
            closure=meshpy.geometry.EXT_OPEN,radial_subdiv=N))


class GeometryMeshPyTriangle(GeometryMeshPy):
    """Define and mesh 2-dimensional domains with MeshPy/Triangle."""

    def __init__(self,points,facets=None,holes=None):
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
        self.info=meshpy.triangle.MeshInfo()
        self.info.set_points(points)
        if holes is not None:
            self.info.set_holes(holes)
        if facets is None:
            self.info.set_facets([(i,i+1) for i in range(0,len(points)-1)])
        else:
            self.info.set_facets(facets)

    def mesh(self,h):
        def ref_func(tri_points,area):
            return bool(area>h*h)
        self.m=meshpy.triangle.build(self.info,refinement_func=ref_func)
        return self._mesh_output()

    def _mesh_output(self):
        p=np.array(self.m.points).T
        t=np.array(self.m.elements).T
        return spfem.mesh.MeshTri(p,t)

    def refine(self,ref_elems):
        p=np.array(self.m.points).T
        t=np.array(self.m.elements).T

        # define new points and stack
        points=np.hstack((p,.5*(p[:,t[0,ref_elems]]+p[:,t[1,ref_elems]]),
                            .5*(p[:,t[1,ref_elems]]+p[:,t[2,ref_elems]]),
                            .5*(p[:,t[0,ref_elems]]+p[:,t[2,ref_elems]]))).T

        # build new mesh
        self.info.set_points(np.array(points))
        self.info.set_facets(np.array(self.m.facets))
        self.m=meshpy.triangle.build(self.info,allow_volume_steiner=False,
                allow_boundary_steiner=False)
        return self._mesh_output()


