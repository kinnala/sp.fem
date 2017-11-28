# -*- coding: utf-8 -*-
"""
Classes that represent different types of meshes.

Currently implemented mesh types are

    * :class:`spfem.mesh.MeshTri`, a triangular mesh
    * :class:`spfem.mesh.MeshTet`, a tetrahedral mesh
    * :class:`spfem.mesh.MeshQuad`, a mesh consisting of quadrilaterals
    * :class:`spfem.mesh.MeshLine`, one-dimensional mesh

Examples
--------

Obtain a three times refined mesh of the unit square and draw it.

.. code-block:: python

    from spfem.mesh import MeshTri
    m = MeshTri()
    m.refine(3)
    m.draw()
    m.show()

"""
try:
    from mayavi import mlab
    OPT_MAYAVI = True
except:
    OPT_MAYAVI = False
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import scipy.interpolate as spi
import spfem.mapping
import copy
import abc
from mpl_toolkits.mplot3d import Axes3D


class Mesh(object):
    """Finite element mesh."""
    __metaclass__ = abc.ABCMeta

    refdom = "none" #: A string defining type of the mesh
    brefdom = "none" #: A string defining type of the boundary mesh

    p = np.array([]) #: The vertices of the mesh, size: dim x Npoints
    t = np.array([]) #: The element connectivity, size: verts/elem x Nelems

    @abc.abstractmethod
    def __init__(self, p, t):
        pass

    def show(self):
        """Call the correct pyplot/mayavi show commands after plotting."""
        if self.dim() <= 2:
            plt.show()
        else:
            mlab.show()

    def dim(self):
        """Return the spatial dimension of the mesh."""
        return float(self.p.shape[0])

    @abc.abstractmethod
    def mapping(self):
        """Default local-to-global mapping for the mesh."""
        raise NotImplementedError("Default mapping not implemented!")

    def remove_elements(self, element_indices):
        """Construct new mesh with elements removed
        based on their indices.

        Parameters
        ----------
        element_indices : numpy array
            List of element indices to remove.

        Returns
        -------
        spfem.Mesh
            A new mesh object with elements removed as per requested.
        """
        keep = np.setdiff1d(np.arange(self.t.shape[1]), element_indices)
        newt = self.t[:, keep]
        ptix = np.unique(newt)
        reverse = np.zeros(self.p.shape[1])
        reverse[ptix] = np.arange(len(ptix))
        newt = reverse[newt]
        newp = self.p[:, ptix]
        return newp, newt.astype(np.intp)

    def scale(self, scale):
        """Scale the mesh.

        Parameters
        ----------
        scale : float OR tuple of size dim
            Scale each dimension by a factor. If a floating
            point number is provided, same scale is used
            for each dimension.
        """
        for itr in range(int(self.dim())):
            if isinstance(scale, tuple):
                self.p[itr, :] *= scale[itr]
            else:
                self.p[itr, :] *= scale

    def translate(self, vec):
        """Translate the mesh.

        Parameters
        ----------
        vec : tuple of size dim
            Translate the mesh by a vector.
        """
        for itr in range(int(self.dim())):
            self.p[itr, :] += vec[itr]

    def _validate(self):
        """Perform mesh validity checks."""
        # check that element connectivity contains integers
        # NOTE: this is neccessary for some plotting functionality
        if not np.issubdtype(self.t[0, 0], int):
            msg = ("Mesh._validate(): Element connectivity "
                   "must consist of integers.")
            raise Exception(msg)
        # check that vertex matrix has "correct" size
        if self.p.shape[0] > 3:
            msg = ("Mesh._validate(): We do not allow meshes "
                   "embedded into larger than 3-dimensional "
                   "Euclidean space! Please check that "
                   "the given vertex matrix is of size Ndim x Nvertices.")
            raise Exception(msg)
        # check that element connectivity matrix has correct size
        nvertices = {'line': 2, 'tri': 3, 'quad': 4, 'tet': 4}
        if self.t.shape[0] != nvertices[self.refdom]:
            msg = ("Mesh._validate(): The given connectivity "
                   "matrix has wrong shape!")
            raise Exception(msg)
        # check that there are no duplicate points
        tmp = np.ascontiguousarray(self.p.T)
        if self.p.shape[1] != np.unique(tmp.view([('', tmp.dtype)]
                                                 * tmp.shape[1])).shape[0]:
            msg = "Mesh._validate(): Mesh contains duplicate vertices."
            raise Exception(msg)
        # check that all points are at least in some element
        if len(np.setdiff1d(np.arange(self.p.shape[1]), np.unique(self.t))):
            msg = ("Mesh._validate(): Mesh contains a vertex "
                   "not belonging to any element.")
            raise Exception(msg)

class MeshLine(Mesh):
    """One-dimensional mesh."""

    refdom = "line"
    brefdom = "point"

    def __init__(self, p=None, t=None, validate=True):
        super(MeshLine, self).__init__(p, t)
        if p is None and t is None:
            p = np.array([[0, 1]])
            t = np.array([[0], [1]])
        elif p is None or t is None:
            raise Exception("Must provide p AND t or neither")
        self.p = p
        self.t = t
        if validate:
            self._validate()

    def refine(self, N=1):
        """Perform one or more uniform refines on the mesh."""
        for _ in range(N):
            self._single_refine()

    def _single_refine(self):
        """Perform a single mesh refine that halves 'h'."""
        # rename variables
        t = self.t
        p = self.p

        mid = range(self.t.shape[1]) + np.max(t) + 1
        # new vertices and elements
        newp = np.hstack((p, 0.5*(p[:, self.t[0, :]] + p[:, self.t[1, :]])))
        newt = np.vstack((t[0, :], mid))
        newt = np.hstack((newt, np.vstack((mid, t[1, :]))))
        # update fields
        self.p = newp
        self.t = newt

    def boundary_nodes(self):
        """Find the boundary nodes of the mesh."""
        _, counts = np.unique(self.t.flatten(), return_counts=True)
        return np.nonzero(counts == 1)[0]

    def interior_nodes(self):
        """Find the interior nodes of the mesh."""
        _, counts = np.unique(self.t.flatten(), return_counts=True)
        return np.nonzero(counts == 2)[0]

    def plot(self, u, color='ko-'):
        """Plot a function defined on the nodes of the mesh."""
        xs = []
        ys = []
        for y1, y2, s, t in zip(u[self.t[0, :]],
                                u[self.t[1, :]],
                                self.p[0, self.t[0, :]],
                                self.p[0, self.t[1, :]]):
            xs.append(s)
            xs.append(t)
            xs.append(None)
            ys.append(y1)
            ys.append(y2)
            ys.append(None)
        plt.plot(xs, ys, color)

    def mapping(self):
        return spfem.mapping.MappingAffine(self)


class MeshQuad(Mesh):
    """A mesh consisting of quadrilateral elements."""

    refdom = "quad"
    brefdom = "line"

    def __init__(self, p=None, t=None, validate=True):
        super(MeshQuad, self).__init__(p, t)
        if p is None and t is None:
            p = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
            t = np.array([[0, 1, 2, 3]]).T
        elif p is None or t is None:
            raise Exception("Must provide p AND t or neither")
        self.p = p
        self.t = t
        if validate:
            self._validate()
        self._build_mappings()

    def _build_mappings(self):
        # do not sort since order defines counterclockwise order
        # self.t=np.sort(self.t,axis=0)

        # define facets: in the order (0,1) (1,2) (2,3) (0,3)
        self.facets = np.sort(np.vstack((self.t[0, :], self.t[1, :])), axis=0)
        self.facets = np.hstack((self.facets,
                                 np.sort(np.vstack((self.t[1, :],
                                                    self.t[2, :])), axis=0)))
        self.facets = np.hstack((self.facets,
                                 np.sort(np.vstack((self.t[2, :],
                                                    self.t[3, :])), axis=0)))
        self.facets = np.hstack((self.facets,
                                 np.sort(np.vstack((self.t[0, :],
                                                    self.t[3, :])), axis=0)))

        # get unique facets and build quad-to-facet mapping: 4 (edges) x Nquads
        tmp = np.ascontiguousarray(self.facets.T)
        tmp, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)]*tmp.shape[1]),
                                  return_index=True, return_inverse=True)
        self.facets = self.facets[:, ixa]
        self.t2f = ixb.reshape((4, self.t.shape[1]))

        # build facet-to-quadrilateral mapping: 2 (quads) x Nedges
        e_tmp = np.hstack((self.t2f[0, :],
                           self.t2f[1, :],
                           self.t2f[2, :],
                           self.t2f[3, :]))
        t_tmp = np.tile(np.arange(self.t.shape[1]), (1, 4))[0]

        e_first, ix_first = np.unique(e_tmp, return_index=True)
        # this emulates matlab unique(e_tmp,'last')
        e_last, ix_last = np.unique(e_tmp[::-1], return_index=True)
        ix_last = e_tmp.shape[0] - ix_last - 1

        self.f2t = np.zeros((2, self.facets.shape[1]), dtype=np.int64)
        self.f2t[0, e_first] = t_tmp[ix_first]
        self.f2t[1, e_last] = t_tmp[ix_last]

        # second row to -1 if repeated (i.e., on boundary)
        self.f2t[1, np.nonzero(self.f2t[0, :] == self.f2t[1, :])[0]] = -1

    def boundary_nodes(self):
        """Return an array of boundary node indices."""
        return np.unique(self.facets[:, self.boundary_facets()])

    def boundary_facets(self):
        """Return an array of boundary facet indices."""
        return np.nonzero(self.f2t[1, :] == -1)[0]

    def interior_nodes(self):
        """Return an array of interior node indices."""
        return np.setdiff1d(np.arange(0, self.p.shape[1]),
                            self.boundary_nodes())

    def nodes_satisfying(self, test):
        """Return nodes that satisfy some condition."""
        return np.nonzero(test(self.p[0, :], self.p[1, :]))[0]

    def facets_satisfying(self, test):
        """Return facets whose midpoints satisfy some condition."""
        mx = 0.5*(self.p[0, self.facets[0, :]] + self.p[0, self.facets[1, :]])
        my = 0.5*(self.p[1, self.facets[0, :]] + self.p[1, self.facets[1, :]])
        return np.nonzero(test(mx, my))[0]

    def refine(self, N=1):
        """Perform one or more refines on the mesh."""
        for _ in range(N):
            self._single_refine()

    def _single_refine(self):
        """Perform a single mesh refine that halves 'h'.

        Each quadrilateral is split into four subquads."""
        # rename variables
        t = self.t
        p = self.p
        e = self.facets
        sz = p.shape[1]
        t2f = self.t2f + sz
        # quadrilateral middle point
        mid = range(self.t.shape[1]) + np.max(t2f) + 1
        # new vertices are the midpoints of edges ...
        newp1 = 0.5*np.vstack((p[0, e[0, :]] + p[0, e[1, :]],
                               p[1, e[0, :]] + p[1, e[1, :]]))
        # ... and element middle points
        newp2 = 0.25*np.vstack((p[0, t[0, :]] + p[0, t[1, :]] +
                                p[0, t[2, :]] + p[0, t[3, :]],
                                p[1, t[0, :]] + p[1, t[1, :]] +
                                p[1, t[2, :]] + p[1, t[3, :]]))
        newp = np.hstack((p, newp1, newp2))
        # build new quadrilateral definitions
        newt = np.vstack((t[0, :],
                          t2f[0, :],
                          mid,
                          t2f[3, :]))
        newt = np.hstack((newt, np.vstack((t2f[0, :],
                                           t[1, :],
                                           t2f[1, :],
                                           mid))))
        newt = np.hstack((newt, np.vstack((mid,
                                           t2f[1, :],
                                           t[2, :],
                                           t2f[2, :]))))
        newt = np.hstack((newt, np.vstack((t2f[3, :],
                                           mid,
                                           t2f[2, :],
                                           t[3, :]))))
        # update fields
        self.p = newp
        self.t = newt

        self._build_mappings()

    def _splitquads(self, x):
        """Split each quad into a triangle and return MeshTri."""
        if len(x) == self.t.shape[1]:
            # preserve elemental constant functions
            X = np.concatenate((x, x))
        else:
            X = x
        t = self.t[[0, 1, 3], :]
        t = np.hstack((t, self.t[[1, 2, 3]]))
        return MeshTri(self.p, t), X

    def plot(self, z, smooth=False):
        """Visualize nodal or elemental function (2d).

        The quadrilateral mesh is split into triangular mesh (MeshTri) and
        the respective plotting function for the triangular mesh is used.
        """
        m, z = self._splitquads(z)
        return m.plot(z, smooth)

    def plot3(self, z, smooth=False):
        """Visualize nodal function (3d i.e. three axes).

        The quadrilateral mesh is split into triangular mesh (MeshTri) and
        the respective plotting function for the triangular mesh is used.
        """
        m, z = self._splitquads(z)
        return m.plot3(z, smooth)

    def jiggle(self, z=0.2):
        """Jiggle the interior nodes of the mesh.

        Parameters
        ----------
        z : (OPTIONAL, default=0.2) float
            Mesh parameter is multiplied by this number. The resulting number
            corresponds to the standard deviation of the jiggle.
        """
        y = z*self.param()
        I = self.interior_nodes()
        self.p[0, I] = self.p[0, I] + y*np.random.rand(len(I))
        self.p[1, I] = self.p[1, I] + y*np.random.rand(len(I))

    def param(self):
        """Return mesh parameter."""
        return np.max(np.sqrt(np.sum((self.p[:, self.facets[0, :]] -
                                      self.p[:, self.facets[1, :]])**2,
                                     axis=0)))

    def draw(self):
        """Draw the mesh."""
        fig = plt.figure()
        # visualize the mesh
        # faster plotting is achieved through
        # None insertion trick.
        xs = []
        ys = []
        for s, t, u, v in zip(self.p[0, self.facets[0, :]],
                              self.p[1, self.facets[0, :]],
                              self.p[0, self.facets[1, :]],
                              self.p[1, self.facets[1, :]]):
            xs.append(s)
            xs.append(u)
            xs.append(None)
            ys.append(t)
            ys.append(v)
            ys.append(None)
        plt.plot(xs, ys, 'k')
        return fig

    def mapping(self):
        return spfem.mapping.MappingQ1(self)


class MeshTet(Mesh):
    """Tetrahedral mesh."""

    refdom = "tet"
    brefdom = "tri"

    def __init__(self, p=None, t=None, validate=True):
        super(MeshTet, self).__init__(p, t)
        if p is None and t is None:
            p = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0],
                          [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]).T
            t = np.array([[0, 1, 2, 3], [3, 5, 1, 7], [2, 3, 6, 7],
                          [2, 3, 1, 7], [1, 2, 4, 7]]).T
        elif p is None or t is None:
            raise Exception("Must provide p AND t or neither")
        self.p = p
        self.t = t
        if validate:
            self._validate()
        self._build_mappings()

    def _build_mappings(self):
        """Build element-to-facet, element-to-edges, etc. mappings."""
        # define edges: in the order (0,1) (1,2) (0,2) (0,3) (1,3) (2,3)
        self.edges = np.sort(np.vstack((self.t[0, :], self.t[1, :])), axis=0)
        e = np.array([1, 2,
                      0, 2,
                      0, 3,
                      1, 3,
                      2, 3])
        for i in range(5):
            self.edges = np.hstack((self.edges,
                                    np.sort(np.vstack((self.t[e[2*i], :],
                                                       self.t[e[2*i+1], :])),
                                            axis=0)))

        # unique edges
        tmp = np.ascontiguousarray(self.edges.T)
        tmp, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)] * tmp.shape[1]),
                                  return_index=True, return_inverse=True)
        self.edges = self.edges[:, ixa]
        self.t2e = ixb.reshape((6, self.t.shape[1]))

        # define facets
        self.facets = np.sort(np.vstack((self.t[0, :],
                                         self.t[1, :],
                                         self.t[2, :])), axis=0)
        f = np.array([0, 1, 3,
                      0, 2, 3,
                      1, 2, 3])
        for i in range(3):
            self.facets = np.hstack((self.facets,
                                     np.sort(np.vstack((self.t[f[2*i], :],
                                                        self.t[f[2*i+1], :],
                                                        self.t[f[2*i+2]])),
                                             axis=0)))

        # unique facets
        tmp = np.ascontiguousarray(self.facets.T)
        tmp, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)] * tmp.shape[1]),
                                  return_index=True, return_inverse=True)
        self.facets = self.facets[:, ixa]
        self.t2f = ixb.reshape((4, self.t.shape[1]))

        # build facet-to-tetra mapping: 2 (tets) x Nfacets
        e_tmp = np.hstack((self.t2f[0, :], self.t2f[1, :],
                           self.t2f[2, :], self.t2f[3, :]))
        t_tmp = np.tile(np.arange(self.t.shape[1]), (1, 4))[0]

        e_first, ix_first = np.unique(e_tmp, return_index=True)
        # this emulates matlab unique(e_tmp,'last')
        e_last, ix_last = np.unique(e_tmp[::-1], return_index=True)
        ix_last = e_tmp.shape[0] - ix_last-1

        self.f2t = np.zeros((2, self.facets.shape[1]), dtype=np.int64)
        self.f2t[0, e_first] = t_tmp[ix_first]
        self.f2t[1, e_last] = t_tmp[ix_last]

        # second row to zero if repeated (i.e., on boundary)
        self.f2t[1, np.nonzero(self.f2t[0, :] == self.f2t[1, :])[0]] = -1

    def refine(self, N=1):
        """Perform one or more refines on the mesh."""
        for itr in range(N):
            self._single_refine()

    def nodes_satisfying(self, test):
        """Return nodes that satisfy some condition."""
        return np.nonzero(test(self.p[0, :], self.p[1, :], self.p[2, :]))[0]

    def facets_satisfying(self, test):
        """Return facets whose midpoints satisfy some condition."""
        mx = 0.3333333*(self.p[0, self.facets[0, :]] +
                        self.p[0, self.facets[1, :]] +
                        self.p[0, self.facets[2, :]])
        my = 0.3333333*(self.p[1, self.facets[0, :]] +
                        self.p[1, self.facets[1, :]] +
                        self.p[1, self.facets[2, :]])
        mz = 0.3333333*(self.p[2, self.facets[0, :]] +
                        self.p[2, self.facets[1, :]] +
                        self.p[2, self.facets[2, :]])
        return np.nonzero(test(mx, my, mz))[0]

    def edges_satisfying(self, test):
        """Return edges whose midpoints satisfy some condition."""
        mx = 0.5*(self.p[0, self.edges[0, :]] + self.p[0, self.edges[1, :]])
        my = 0.5*(self.p[1, self.edges[0, :]] + self.p[1, self.edges[1, :]])
        mz = 0.5*(self.p[2, self.edges[0, :]] + self.p[2, self.edges[1, :]])
        return np.nonzero(test(mx, my, mz))[0]

    def _single_refine(self):
        """Perform a single mesh refine.

        Let the nodes of a tetrahedron be numbered as 0, 1, 2 and 3.
        It is assumed that the edges in self.t2e are given in the order

          I=(0,1), II=(1,2), III=(0,2), IV=(0,3), V=(1,3), VI=(2,3)

        by self._build_mappings(). Let I denote the midpoint of the edge
        (0,1), II denote the midpoint of the edge (1,2), etc. Then each
        tetrahedron is split into eight smaller subtetrahedra as follows.

        The first four subtetrahedra have the following nodes:

          1. (0,I,III,IV)
          2. (1,I,II,V)
          3. (2,II,III,VI)
          4. (3,IV,V,VI)

        The remaining middle-portion of the original tetrahedron consists
        of a union of two mirrored pyramids. This bi-pyramid can be splitted
        into four tetrahedra in a three different ways by connecting the
        midpoints of two opposing edges (there are three different pairs
        of opposite edges).

        For each tetrahedra in the original mesh, we split the bi-pyramid
        in such a way that the connection between the opposite edges
        is shortest. This minimizes the shape-regularity constant of
        the resulting mesh family.
        """
        # rename variables
        t = self.t
        p = self.p
        e = self.edges
        sz = p.shape[1]
        t2e = self.t2e + sz
        # new vertices are the midpoints of edges
        newp = 0.5*np.vstack((p[0, e[0, :]] + p[0, e[1, :]],
                              p[1, e[0, :]] + p[1, e[1, :]],
                              p[2, e[0, :]] + p[2, e[1, :]]))
        newp = np.hstack((p, newp))
        # new tets
        newt = np.vstack((t[0, :], t2e[0, :], t2e[2, :], t2e[3, :]))
        newt = np.hstack((newt, np.vstack((t[1, :], t2e[0, :], t2e[1, :], t2e[4, :]))))
        newt = np.hstack((newt, np.vstack((t[2, :], t2e[1, :], t2e[2, :], t2e[5, :]))))
        newt = np.hstack((newt, np.vstack((t[3, :], t2e[3, :], t2e[4, :], t2e[5, :]))))
        # compute middle pyramid diagonal lengths and choose shortest
        d1 = ((newp[0, t2e[2, :]] - newp[0, t2e[4, :]])**2 +
              (newp[1, t2e[2, :]] - newp[1, t2e[4, :]])**2)
        d2 = ((newp[0, t2e[1, :]] - newp[0, t2e[3, :]])**2 +
              (newp[1, t2e[1, :]] - newp[1, t2e[3, :]])**2)
        d3 = ((newp[0, t2e[0, :]] - newp[0, t2e[5, :]])**2 +
              (newp[1, t2e[0, :]] - newp[1, t2e[5, :]])**2)
        I1 = d1 < d2
        I2 = d1 < d3
        I3 = d2 < d3
        c1 = I1*I2
        c2 = (-I1)*I3
        c3 = (-I2)*(-I3)
        # splitting the pyramid in the middle.
        # diagonals are [2,4], [1,3] and [0,5]
        # CASE 1: diagonal [2,4]
        newt = np.hstack((newt, np.vstack((t2e[2, c1], t2e[4, c1],
                                           t2e[0, c1], t2e[1, c1]))))
        newt = np.hstack((newt, np.vstack((t2e[2, c1], t2e[4, c1],
                                           t2e[0, c1], t2e[3, c1]))))
        newt = np.hstack((newt, np.vstack((t2e[2, c1], t2e[4, c1],
                                           t2e[1, c1], t2e[5, c1]))))
        newt = np.hstack((newt, np.vstack((t2e[2, c1], t2e[4, c1],
                                           t2e[3, c1], t2e[5, c1]))))
        # CASE 2: diagonal [1,3]
        newt = np.hstack((newt, np.vstack((t2e[1, c2], t2e[3, c2],
                                           t2e[0, c2], t2e[4, c2]))))
        newt = np.hstack((newt, np.vstack((t2e[1, c2], t2e[3, c2],
                                           t2e[4, c2], t2e[5, c2]))))
        newt = np.hstack((newt, np.vstack((t2e[1, c2], t2e[3, c2],
                                           t2e[5, c2], t2e[2, c2]))))
        newt = np.hstack((newt, np.vstack((t2e[1, c2], t2e[3, c2],
                                           t2e[2, c2], t2e[0, c2]))))
        # CASE 3: diagonal [0,5]
        newt = np.hstack((newt, np.vstack((t2e[0, c3], t2e[5, c3],
                                           t2e[1, c3], t2e[4, c3]))))
        newt = np.hstack((newt, np.vstack((t2e[0, c3], t2e[5, c3],
                                           t2e[4, c3], t2e[3, c3]))))
        newt = np.hstack((newt, np.vstack((t2e[0, c3], t2e[5, c3],
                                           t2e[3, c3], t2e[2, c3]))))
        newt = np.hstack((newt, np.vstack((t2e[0, c3], t2e[5, c3],
                                           t2e[2, c3], t2e[1, c3]))))
        # update fields
        self.p = newp
        self.t = newt

        self._build_mappings()

    def draw_vertices(self):
        """Draw all vertices using mplot3d."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.p[0, :], self.p[1, :], self.p[2, :])
        return fig

    def draw_edges(self):
        """Draw all edges in a wireframe representation."""
        # use mayavi
        if OPT_MAYAVI:
            mlab.triangular_mesh(self.p[0, :], self.p[1, :], self.p[2, :],
                                 self.facets.T, representation='wireframe',
                                 color=(0, 0, 0))
        else:
            raise ImportError("Mayavi not supported "
                              "by the host system!")

    def draw_facets(self, test=None, u=None):
        """Draw all facets."""
        if test is not None:
            xs = 1./3.*(self.p[0, self.facets[0, :]] +
                        self.p[0, self.facets[1, :]] +
                        self.p[0, self.facets[2, :]])
            ys = 1./3.*(self.p[1, self.facets[0, :]] +
                        self.p[1, self.facets[1, :]] +
                        self.p[1, self.facets[2, :]])
            zs = 1./3.*(self.p[2, self.facets[0, :]] +
                        self.p[2, self.facets[1, :]] +
                        self.p[2, self.facets[2, :]])
            fset = np.nonzero(test(xs, ys, zs))[0]
        else:
            fset = range(self.facets.shape[1])

        # use mayavi
        if OPT_MAYAVI:
            if u is None:
                mlab.triangular_mesh(self.p[0, :], self.p[1, :], self.p[2, :],
                                     self.facets[:, fset].T)
                mlab.triangular_mesh(self.p[0, :], self.p[1, :], self.p[2, :],
                                     self.facets[:, fset].T,
                                     representation='wireframe',
                                     color=(0, 0, 0))
            else:
                if u.shape[0] == self.facets.shape[1]:
                    newp = np.vstack((self.p[0, self.facets].flatten(order='F'),
                                      self.p[1, self.facets].flatten(order='F')))
                    newp = np.vstack((newp,
                                      self.p[2, self.facets].flatten(order='F')))
                    newt = np.arange(newp.shape[1]).reshape((3, newp.shape[1]/3),
                                                            order='F')
                    newu = np.tile(u, (3, 1)).flatten(order='F')
                    mlab.triangular_mesh(newp[0, :], newp[1, :], newp[2, :],
                                         newt.T, scalars=newu)
                    mlab.triangular_mesh(newp[0, :], newp[1, :], newp[2, :],
                                         newt.T, representation='wireframe',
                                         color=(0, 0, 0))
                    mlab.axes()
                    mlab.colorbar()
                else:
                    raise Exception("Given data vector "
                                    "shape not supported")
        else:
            raise ImportError("Mayavi not supported "
                              "by the host system!")

    def draw(self, test=None, u=None):
        """Draw all tetrahedra."""
        if test is not None:
            xs = 1./4.*(self.p[0, self.t[0, :]] +
                        self.p[0, self.t[1, :]] +
                        self.p[0, self.t[2, :]] +
                        self.p[0, self.t[3, :]])
            ys = 1./4.*(self.p[1, self.t[0, :]] +
                        self.p[1, self.t[1, :]] +
                        self.p[1, self.t[2, :]] +
                        self.p[1, self.t[3, :]])
            zs = 1./4.*(self.p[2, self.t[0, :]] +
                        self.p[2, self.t[1, :]] +
                        self.p[2, self.t[2, :]] +
                        self.p[2, self.t[3, :]])
            tset = np.nonzero(test(xs, ys, zs))[0]
        else:
            tset = range(self.t.shape[1])

        fset = np.unique(self.t2f[:, tset].flatten())

        if u is None:
            u = self.p[2, :]

        if OPT_MAYAVI:
            mlab.triangular_mesh(self.p[0, :], self.p[1, :], self.p[2, :],
                                 self.facets[:, fset].T, scalars=u)
            mlab.triangular_mesh(self.p[0, :], self.p[1, :], self.p[2, :],
                                 self.facets[:, fset].T,
                                 representation='wireframe', color=(0, 0, 0))
        else:
            raise ImportError("Mayavi not supported "
                              "by the host system!")

    def boundary_nodes(self):
        """Return an array of boundary node indices."""
        return np.unique(self.facets[:, self.boundary_facets()])

    def boundary_facets(self):
        """Return an array of boundary facet indices."""
        return np.nonzero(self.f2t[1, :] == -1)[0]

    def interior_facets(self):
        """Return an array of interior facet indices."""
        return np.nonzero(self.f2t[1, :] >= 0)[0]

    def boundary_edges(self):
        """Return an array of boundary edge indices."""
        bnodes = self.boundary_nodes()[:, None]
        return np.nonzero(np.sum(self.edges[0, :] == bnodes, axis=0) *
                          np.sum(self.edges[1, :] == bnodes, axis=0))[0]

    def interior_nodes(self):
        """Return an array of interior node indices."""
        return np.setdiff1d(np.arange(0, self.p.shape[1]), self.boundary_nodes())

    def param(self):
        """Return (maximum) mesh parameter."""
        return np.max(np.sqrt(np.sum((self.p[:, self.edges[0, :]] -
                                      self.p[:, self.edges[1, :]])**2, axis=0)))

    def shapereg(self):
        """Return the largest shape-regularity constant."""
        def edgelen(n):
            return np.sqrt(np.sum((self.p[:, self.edges[0, self.t2e[n, :]]] -
                                   self.p[:, self.edges[1, self.t2e[n, :]]])**2,
                                  axis=0))
        edgelenmat = np.vstack(tuple(edgelen(i) for i in range(6)))
        return np.max(np.max(edgelenmat, axis=0)/np.min(edgelenmat, axis=0))

    def mapping(self):
        return spfem.mapping.MappingAffine(self)


class MeshTri(Mesh):
    """Triangular mesh."""

    refdom = "tri"
    brefdom = "line"

    def __init__(self, p=None, t=None, validate=True, initmesh=None, sort_t=True):
        super(MeshTri, self).__init__(p, t)
        if p is None and t is None:
            if initmesh is 'symmetric':
                p = np.array([[0, 1, 1, 0, 0.5],
                              [0, 0, 1, 1, 0.5]], dtype=np.float_)
                t = np.array([[0, 1, 4],
                              [1, 2, 4],
                              [2, 3, 4],
                              [0, 3, 4]], dtype=np.intp).T
            elif initmesh is 'sqsymmetric':
                p = np.array([[0, 0.5, 1,   0, 0.5,   1, 0, 0.5, 1],
                              [0, 0,   0, 0.5, 0.5, 0.5, 1,   1, 1]], dtype=np.float_)
                t = np.array([[0, 1, 4],
                              [1, 2, 4],
                              [2, 4, 5],
                              [0, 3, 4],
                              [3, 4, 6],
                              [4, 6, 7],
                              [4, 7, 8],
                              [4, 5, 8]], dtype=np.intp).T
            elif initmesh is 'reftri':
                p = np.array([[0, 1, 0],
                              [0, 0, 1]], dtype=np.float_)
                t = np.array([[0, 1, 2]], dtype=np.intp).T
            else:
                p = np.array([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float_)
                t = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp).T
        elif p is None or t is None:
            raise Exception("Must provide p AND t or neither")
        self.p = p
        self.t = t
        if validate:
            self._validate()
        self._build_mappings(sort_t=sort_t)

    def _build_mappings(self, sort_t=True):
        # sort to preserve orientations etc.
        if sort_t:
            self.t = np.sort(self.t, axis=0)

        # define facets: in the order (0,1) (1,2) (0,2)
        self.facets = np.sort(np.vstack((self.t[0, :], self.t[1, :])), axis=0)
        self.facets = np.hstack((self.facets,
                                 np.sort(np.vstack((self.t[1, :], self.t[2, :])),
                                         axis=0)))
        self.facets = np.hstack((self.facets,
                                 np.sort(np.vstack((self.t[0, :], self.t[2, :])),
                                         axis=0)))

        # get unique facets and build triangle-to-facet
        # mapping: 3 (edges) x Ntris
        tmp = np.ascontiguousarray(self.facets.T)
        tmp, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)] * tmp.shape[1]),
                                  return_index=True, return_inverse=True)
        self.facets = self.facets[:, ixa]
        self.t2f = ixb.reshape((3, self.t.shape[1]))

        # build facet-to-triangle mapping: 2 (triangles) x Nedges
        e_tmp = np.hstack((self.t2f[0, :], self.t2f[1, :], self.t2f[2, :]))
        t_tmp = np.tile(np.arange(self.t.shape[1]), (1, 3))[0]

        e_first, ix_first = np.unique(e_tmp, return_index=True)
        # this emulates matlab unique(e_tmp,'last')
        e_last, ix_last = np.unique(e_tmp[::-1], return_index=True)
        ix_last = e_tmp.shape[0] - ix_last - 1

        self.f2t = np.zeros((2, self.facets.shape[1]), dtype=np.int64)
        self.f2t[0, e_first] = t_tmp[ix_first]
        self.f2t[1, e_last] = t_tmp[ix_last]

        # second row to zero if repeated (i.e., on boundary)
        self.f2t[1, np.nonzero(self.f2t[0, :] == self.f2t[1, :])[0]] = -1

    def boundary_nodes(self):
        """Return an array of boundary node indices."""
        return np.unique(self.facets[:, self.boundary_facets()])

    def boundary_facets(self):
        """Return an array of boundary facet indices."""
        return np.nonzero(self.f2t[1, :] == -1)[0]

    def interior_facets(self):
        """Return an array of interior facet indices."""
        return np.nonzero(self.f2t[1, :] >= 0)[0]

    def nodes_satisfying(self, test):
        """Return nodes that satisfy some condition."""
        return np.nonzero(test(self.p[0, :], self.p[1, :]))[0]

    def elements_satisfying(self, test):
        """Return elements whose midpoints satisfy some condition."""
        mx = .33333*np.sum(self.p[0, self.t], axis=0)
        my = .33333*np.sum(self.p[1, self.t], axis=0)
        return np.nonzero(test(mx, my))[0]

    def facets_satisfying(self, test):
        """Return facets whose midpoints satisfy some condition."""
        mx = 0.5*(self.p[0, self.facets[0, :]] + self.p[0, self.facets[1, :]])
        my = 0.5*(self.p[1, self.facets[0, :]] + self.p[1, self.facets[1, :]])
        return np.nonzero(test(mx, my))[0]

    def interior_nodes(self):
        """Return an array of interior node indices."""
        return np.setdiff1d(np.arange(0, self.p.shape[1]), self.boundary_nodes())

    def interpolator(self, x):
        """Return a function which interpolates values with P1 basis."""
        triang = mtri.Triangulation(self.p[0, :], self.p[1, :], self.t.T)
        interpf = mtri.LinearTriInterpolator(triang, x)
        # contruct an interpolator handle
        def handle(X, Y):
            return interpf(X, Y).data
        return handle

    def const_interpolator(self, x):
        """Return a function which interpolates values with P0 basis."""
        triang = mtri.Triangulation(self.p[0, :], self.p[1, :], self.t.T)
        finder = triang.get_trifinder()
        # construct an interpolator handle
        def handle(X, Y):
            return x[finder(X, Y)]
        return handle

    def param(self):
        """Return mesh parameter."""
        return np.max(np.sqrt(np.sum((self.p[:, self.facets[0, :]] -
                                      self.p[:, self.facets[1, :]])**2, axis=0)))

    def smooth(self, c=1.0):
        """Apply smoothing to interior nodes."""
        from spfem.asm import AssemblerElement
        from spfem.element import ElementTriP1
        from spfem.utils import direct

        e = ElementTriP1()
        a = AssemblerElement(self, e)

        K = a.iasm(lambda du,dv: du[0]*dv[0] + du[1]*dv[1])
        M = a.iasm(lambda u,v: u*v)

        I = self.interior_nodes()
        dx = - k*direct(M, K.dot(self.p[0, :]))
        dy = - k*direct(M, K.dot(self.p[1, :]))

        self.p[0, I] += dx[I]
        self.p[1, I] += dy[I]

    def draw_debug(self):
        fig = plt.figure()
        plt.hold('on')
        for itr in range(self.t.shape[1]):
            plt.plot(self.p[0,self.t[[0,1],itr]], self.p[1,self.t[[0,1],itr]], 'k-')
            plt.plot(self.p[0,self.t[[1,2],itr]], self.p[1,self.t[[1,2],itr]], 'k-')
            plt.plot(self.p[0,self.t[[0,2],itr]], self.p[1,self.t[[0,2],itr]], 'k-')
        return fig


    def draw(self, nofig=False, trinum=False):
        """Draw the mesh."""
        if nofig:
            fig = 0
        else:
            # create new figure
            fig = plt.figure()
            ax = fig.add_subplot(111)
        # visualize the mesh faster plotting is achieved through
        # None insertion trick.
        xs = []
        ys = []
        for s, t, u, v in zip(self.p[0, self.facets[0, :]],
                              self.p[1, self.facets[0, :]],
                              self.p[0, self.facets[1, :]],
                              self.p[1, self.facets[1, :]]):
            xs.append(s)
            xs.append(u)
            xs.append(None)
            ys.append(t)
            ys.append(v)
            ys.append(None)
        ax.plot(xs, ys, 'k')
        if trinum:
            if fig==0:
                raise Exception("Must create figure to plot elem numbers")
            for itr in range(self.t.shape[1]):
                mx = .3333*(self.p[0, self.t[0, itr]] + self.p[0, self.t[1, itr]] + self.p[0, self.t[2, itr]])
                my = .3333*(self.p[1, self.t[0, itr]] + self.p[1, self.t[1, itr]] + self.p[1, self.t[2, itr]])
                ax.text(mx, my, str(itr))
        return fig

    def draw_nodes(self, nodes, mark='bo'):
        """Highlight some nodes."""
        plt.plot(self.p[0, nodes], self.p[1, nodes], mark)

    def plot(self, z, smooth=False, nofig=False, zlim=None):
        """Visualize nodal or elemental function (2d)."""
        if nofig:
            fig = 0
        else:
            fig = plt.figure()
        if zlim == None:
            if smooth:
                plt.tripcolor(self.p[0, :], self.p[1, :], self.t.T, z,
                              shading='gouraud')
            else:
                plt.tripcolor(self.p[0, :], self.p[1, :], self.t.T, z)
        else:
            if smooth:
                plt.tripcolor(self.p[0, :], self.p[1, :], self.t.T, z,
                              shading='gouraud', vmin=zlim[0], vmax=zlim[1])
            else:
                plt.tripcolor(self.p[0, :], self.p[1, :], self.t.T, z,
                              vmin=zlim[0], vmax=zlim[1])
        return fig

    def plot3(self, z, smooth=False):
        """Visualize nodal function (3d i.e. three axes)."""
        fig = plt.figure()
        if len(z) == self.p.shape[1]:
            # one value per node (piecewise linear, globally cont)
            if smooth:
                # use mayavi
                if OPT_MAYAVI:
                    mlab.triangular_mesh(self.p[0, :], self.p[1, :], z, self.t.T)
                else:
                    raise ImportError("Mayavi not supported "
                                      "by the host system!")
            else:
                # use matplotlib
                ax = fig.gca(projection='3d')
                ts = mtri.Triangulation(self.p[0, :], self.p[1, :], self.t.T)
                ax.plot_trisurf(self.p[0, :], self.p[1, :], z,
                                triangles=ts.triangles,
                                cmap=plt.cm.Spectral)
        elif len(z) == self.t.shape[1]:
            # one value per element (piecewise const)
            nt = self.t.shape[1]
            newt = np.arange(3*nt, dtype=np.int64).reshape((nt, 3))
            newpx = self.p[0, self.t].flatten(order='F')
            newpy = self.p[1, self.t].flatten(order='F')
            newz = np.vstack((z, z, z)).flatten(order='F')
            ax = fig.gca(projection='3d')
            ts = mtri.Triangulation(newpx, newpx, newt)
            ax.plot_trisurf(newpx, newpy, newz,
                            triangles=ts.triangles,
                            cmap=plt.cm.Spectral)
        elif len(z) == 3*self.t.shape[1]:
            # three values per element (piecewise linear)
            nt = self.t.shape[1]
            newt = np.arange(3*nt, dtype=np.int64).reshape((nt, 3))
            newpx = self.p[0, self.t].flatten(order='F')
            newpy = self.p[1, self.t].flatten(order='F')
            ax = fig.gca(projection='3d')
            ts = mtri.Triangulation(newpx, newpx, newt)
            ax.plot_trisurf(newpx, newpy, z,
                            triangles=ts.triangles,
                            cmap=plt.cm.Spectral)
        else:
            raise NotImplementedError("MeshTri.plot3: not implemented for "
                                      "the given shape of input vector!")

    def refine(self, N=1):
        """Perform one or more refines on the mesh."""
        for itr in range(N):
            self._single_refine()

    def _single_refine(self):
        """Perform a single mesh refine."""
        # rename variables
        t = self.t
        p = self.p
        e = self.facets
        sz = p.shape[1]
        t2f = self.t2f + sz
        # new vertices are the midpoints of edges
        newp = 0.5*np.vstack((p[0, e[0, :]] + p[0, e[1, :]],
                              p[1, e[0, :]] + p[1, e[1, :]]))
        newp = np.hstack((p, newp))
        # build new triangle definitions
        newt = np.vstack((t[0, :], t2f[0, :], t2f[2, :]))
        newt = np.hstack((newt, np.vstack((t[1, :], t2f[0, :], t2f[1, :]))))
        newt = np.hstack((newt, np.vstack((t[2, :], t2f[2, :], t2f[1, :]))))
        newt = np.hstack((newt, np.vstack((t2f[0, :], t2f[1, :], t2f[2, :]))))
        # update fields
        self.p = newp
        self.t = newt

        self._build_mappings()

    def _neighbors(self, tris=None):
        """Return the matrix 3 x len(tris) containing the indices
        of the neighboring elements. Neighboring element over boundary
        is defined as the element itself."""
        if tris is None:
            # by default, all triangles
            tris = np.arange(self.t.shape[1])
        facet_tris_0 = self.f2t[:, self.t2f[0, tris]]
        facet_tris_1 = self.f2t[:, self.t2f[1, tris]]
        facet_tris_2 = self.f2t[:, self.t2f[2, tris]]
        ntris_0 = (facet_tris_0 != tris)
        ntris_1 = (facet_tris_1 != tris)
        ntris_2 = (facet_tris_2 != tris)
        neighbors = np.vstack((facet_tris_0[ntris_0].flatten('F'),
                               facet_tris_1[ntris_1].flatten('F'),
                               facet_tris_2[ntris_2].flatten('F')))
        ix = (neighbors == -1)
        neighbors[ix] = np.vstack((tris, tris, tris))[ix]
        return neighbors

    def mirror_mesh(self, a, b, c):
        tmp = -2.0*(a*self.p[0, :] + b*self.p[1, :] + c)/(a**2 + b**2)
        newx = a*tmp + self.p[0, :]
        newy = b*tmp + self.p[1, :]
        newpoints = np.vstack((newx, newy))
        points = np.hstack((self.p, newpoints))
        tris = np.hstack((self.t, self.t + self.p.shape[1]))



        # remove duplicates
        tmp = np.ascontiguousarray(points.T)
        tmp, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)]*tmp.shape[1]), return_index=True, return_inverse=True)
        points = points[:, ixa]
        tris = ixb[tris]

        return MeshTri(points, marked)

    def local_rgb(self, marked):
        """Perform adaptive RGB refinement.
        This is more or less directly ported from the book of Sören Bartels."""
        t = self.t
        p = self.p

        #tmp = np.zeros(self.t.shape[1])
        #tmp[marked] = 1
        #marked = tmp
        #print marked

        # change (0,2) to longest edge
        # TODO vectorize this
        for itr in range(t.shape[1]):
            l01 = np.sqrt(np.sum((p[:, t[0, itr]] - p[:, t[1, itr]])**2))
            l12 = np.sqrt(np.sum((p[:, t[1, itr]] - p[:, t[2, itr]])**2))
            l02 = np.sqrt(np.sum((p[:, t[0, itr]] - p[:, t[2, itr]])**2))
            if l02 > l01 and l02 > l12:
                # OK. (0 2) is longest
                continue
            elif l01 > l02 and l01 > l12:
                # (0 1) is longest
                tmp = t[2, itr]
                t[2, itr] = t[1, itr]
                t[1, itr] = tmp
            elif l12 > l01 and l12 > l02:
                # (1 2) is longest
                tmp = t[0, itr]
                t[0, itr] = t[1, itr]
                t[1, itr] = tmp


        detA = (p[0,t[1,:]]-p[0,t[0,:]])*(p[1,t[2,:]]-p[1,t[0,:]])-\
               (p[0,t[2,:]]-p[0,t[0,:]])*(p[1,t[1,:]]-p[1,t[0,:]])


        swap = detA<0

        temppi = t[0,swap] 
        t[0,swap] = t[2,swap] 
        t[2,swap] = temppi

        detA = (p[0,t[1,:]]-p[0,t[0,:]])*(p[1,t[2,:]]-p[1,t[0,:]])-\
               (p[0,t[2,:]]-p[0,t[0,:]])*(p[1,t[1,:]]-p[1,t[0,:]])
        #print detA

        n4e = t.T
        c4n = p.T

        # edges and edge mappings
        edges = np.reshape(n4e[:,[1,2,0,2,0,1]].T, (2,-1)).T

        edges = np.sort(edges,axis=1)

        tmp = np.ascontiguousarray(edges)
        xoxo, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)] * tmp.shape[1]),
                                  return_index=True, return_inverse=True)
        edges = edges[ixa]
        el2edges = ixb.reshape((3, -1)).T

        nEdges = edges.shape[0]
        nC = c4n.shape[0]
        tmp = 1

        markedEdges = np.zeros(nEdges)
        #print el2edges.shape
        markedEdges[np.reshape(el2edges[marked], (-1, 1))] = 1

        while tmp > 0:
            tmp = np.count_nonzero(markedEdges)
            el2markedEdges = markedEdges[el2edges]
            el2markedEdges[el2markedEdges[:,0] + el2markedEdges[:,2]>0,1] = 1.0
            markedEdges[el2edges[el2markedEdges==1.0]] = 1
            tmp = np.count_nonzero(markedEdges)-tmp

        newNodes = np.zeros(nEdges)-1
        newNodes[markedEdges==1.0] = np.arange(np.count_nonzero(markedEdges)) + nC
        newInd = newNodes[el2edges]

        red = (newInd[:,0]>=0) & (newInd[:,1] >= 0) & (newInd[:,2] >= 0)
        blue1 = (newInd[:,0]>=0) & (newInd[:,1] >= 0) & (newInd[:,2] == -1)
        blue3 = (newInd[:,0]==-1) & (newInd[:,1] >= 0) & (newInd[:,2] >= 0)
        green = (newInd[:,0]==-1) & (newInd[:,1] >= 0) & (newInd[:,2] == -1)
        remain = (newInd[:,0]==-1) & (newInd[:,1] == -1) & (newInd[:,2] == -1)

        n4e_red = np.vstack((n4e[red,0],newInd[red,2],newInd[red,1]))
        n4e_red_1 = np.reshape(n4e_red,(3,-1)).T
        n4e_red = np.vstack((newInd[red,1],newInd[red,0],n4e[red,2]))
        n4e_red_2 = np.reshape(n4e_red,(3,-1)).T
        n4e_red = np.vstack((newInd[red,2],n4e[red,1],newInd[red,0]))
        n4e_red_3 = np.reshape(n4e_red,(3,-1)).T
        n4e_red = np.vstack((newInd[red,0],newInd[red,1],newInd[red,2]))
        n4e_red_4 = np.reshape(n4e_red,(3,-1)).T
        n4e_red = np.vstack((n4e_red_1, n4e_red_2, n4e_red_3, n4e_red_4))
        #n4e_red = np.vstack((n4e[red,2],newInd[red,2],newInd[red,1],
        #    newInd[red,1],newInd[red,0],n4e[red,1],
        #    newInd[red,2],n4e[red,0],newInd[red,0],
        #    newInd[red,0],newInd[red,1],newInd[red,2]))
        #n4e_red = np.reshape(n4e_red,(3,-1)).T

        n4e_blue1 = np.vstack((n4e[blue1,1],newInd[blue1,1],n4e[blue1,0]))
        n4e_blue1_1 = np.reshape(n4e_blue1,(3,-1)).T
        n4e_blue1 = np.vstack((n4e[blue1,1],newInd[blue1,0],newInd[blue1,1]))
        n4e_blue1_2 = np.reshape(n4e_blue1,(3,-1)).T
        n4e_blue1 = np.vstack((newInd[blue1,1],newInd[blue1,0],n4e[blue1,2]))
        n4e_blue1_3 = np.reshape(n4e_blue1,(3,-1)).T
        n4e_blue1 = np.vstack((n4e_blue1_1, n4e_blue1_2, n4e_blue1_3))
        #n4e_blue1 = np.vstack((n4e[blue1,1],newInd[blue1,1],n4e[blue1,0],
        #    n4e[blue1,1],newInd[blue1,0],newInd[blue1,1],
        #    newInd[blue1,1],newInd[blue1,0],n4e[blue1,2]))
        #n4e_blue1 = np.reshape(n4e_blue1,(3,-1)).T

        n4e_blue3 = np.vstack((n4e[blue3,0],newInd[blue3,2],newInd[blue3,1]))
        n4e_blue3_1 = np.reshape(n4e_blue3,(3,-1)).T
        n4e_blue3 = np.vstack((newInd[blue3,1],newInd[blue3,2],n4e[blue3,1]))
        n4e_blue3_2 = np.reshape(n4e_blue3,(3,-1)).T
        n4e_blue3 = np.vstack((n4e[blue3,2],newInd[blue3,1],n4e[blue3,1]))
        n4e_blue3_3 = np.reshape(n4e_blue3,(3,-1)).T
        n4e_blue3 = np.vstack((n4e_blue3_1, n4e_blue3_2, n4e_blue3_3))

        # TODO fix green?
        n4e_green = np.vstack((n4e[green,1],newInd[green,1],n4e[green,0],
            n4e[green,2],newInd[green,1],n4e[green,1]))
        n4e_green = np.reshape(n4e_green,(3,-1)).T

        n4e = np.vstack((n4e[remain], n4e_red, n4e_blue1, n4e_blue3, n4e_green))
        #n4e = np.vstack((n4e_red))

        newCoord = .5*(c4n[edges[markedEdges==1.0,0]]
                + c4n[edges[markedEdges==1.0,1]])

        c4n = np.vstack((c4n, newCoord))

        return MeshTri(c4n.T, n4e.T.astype(np.int64), validate=True)


    def local_refine(self, ref_elems, ref_data=None):
        """Perform adaptive refinement of the given triangles
        and return a new mesh (red green only)."""
        
        if ref_data is None:
            ref_data = np.zeros(self.t.shape[1])

        fsplit = self.t2f[:, ref_elems].flatten('F')

        ref_facets = np.zeros(self.facets.shape[1])
        ref_facets[fsplit] = 1

        for itr in range(1000):
            ix1 = np.nonzero((np.sum(ref_facets[self.t2f], axis=0) == 1)*(ref_data == 1))[0]
            ix2 = np.nonzero(np.sum(ref_facets[self.t2f], axis=0) == 2)[0]

            if len(ix1) + len(ix2) == 0:
                break

            ref_elems = np.concatenate((ref_elems, ix2, ix1))

            fsplit_n = np.hstack((self.t2f[:, ix2], self.t2f[:, ix1]))
            ref_facets[fsplit_n] = 1

            fsplit = np.concatenate((fsplit, fsplit_n.flatten('F')))

            #print ref_elems

        fsplit = np.unique(fsplit)

        f1 = self.facets[0, fsplit]
        f2 = self.facets[1, fsplit]
        p = self.p
        
        pf = .5*(p[:, f1] + p[:, f2])

        fnotsplit = np.setdiff1d(np.arange(self.facets.shape[1]), fsplit)

        fenum = np.zeros(self.facets.shape[1])
        fenum[fsplit] = np.arange(len(fsplit), dtype=np.int64)

        new_facets = np.hstack((self.facets[:, fnotsplit],
                                np.vstack((f1, fenum[fsplit] + p.shape[1])),
                                np.vstack((f2, fenum[fsplit] + p.shape[1]))))

        ix0 = np.nonzero(np.sum(ref_facets[self.t2f], axis=0) == 0)[0]
        ix1 = np.nonzero(np.sum(ref_facets[self.t2f], axis=0) == 1)[0]

        tbi = self.t[:, ix0]
        ref_data_n = ref_data[ix0]

        # split to two subtriangles
        for n in range(3):
            # splitted edge
            ind = ix1[np.nonzero(ref_facets[self.t2f[n, ix1]] == 1)[0]]

            f = self.t2f[n, ind]
            f1 = self.facets[0, f]
            f2 = self.facets[1, f]

            # free vertex
            vf = self.t[(n+2)%3, ind]

            tmp = fenum[f] + p.shape[1]
            tbi = np.hstack((tbi, np.vstack((vf, tmp, f1)), np.vstack((vf, tmp, f2))))

            ref_data_n = np.concatenate((ref_data_n, ref_data[ind]+1, ref_data[ind]+1))
        
        # split to four subtriangles
        ix3 = np.nonzero(np.sum(ref_facets[self.t2f], axis=0) == 3)[0]

        v1 = self.t[0, ix3]
        v2 = self.t[1, ix3]
        v3 = self.t[2, ix3]

        f1 = fenum[self.t2f[0, ix3]] + p.shape[1]
        f2 = fenum[self.t2f[1, ix3]] + p.shape[1]
        f3 = fenum[self.t2f[2, ix3]] + p.shape[1]

        tbi = np.hstack((tbi,
                         np.vstack((f3, v1, f1)),
                         np.vstack((f1, v2, f2)),
                         np.vstack((f2, v3, f3)),
                         np.vstack((f1, f2, f3))))

        ref_data_n = np.concatenate((ref_data_n,
                                     ref_data[ix3],
                                     ref_data[ix3],
                                     ref_data[ix3],
                                     ref_data[ix3]))

        p = np.hstack((p, pf))

        return MeshTri(p, tbi.astype(np.int64)), ref_data_n

    def mapping(self):
        return spfem.mapping.MappingAffine(self)
