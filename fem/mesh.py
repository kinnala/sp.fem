# -*- coding: utf-8 -*-
"""
Classes that represent different types of meshes.

Currently implemented mesh types are

    * MeshTri, a triangular mesh
    * MeshTet, a tetrahedral mesh
    * MeshQuad, a mesh consisting of quadrilaterals
    * MeshLine, one-dimensional mesh

Example 1. Obtain a three times refined mesh of the unit square:

.. code-block:: none

    import fem.mesh as fmsh
    m=fmsh.MeshTri()
    m.refine(3)
"""
try:
    from mayavi import mlab
    opt_mayavi=True
except:
    opt_mayavi=False
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import scipy.interpolate as spi
import fem.mapping as fmap
from mpl_toolkits.mplot3d import Axes3D


class Mesh(object):
    """Abstract finite element mesh class."""

    def __init__(self,p,t):
        raise NotImplementedError("Mesh constructor not implemented!")

    def plot(self):
        raise NotImplementedError("Mesh.plot() not implemented!")

    def dim(self):
        return float(self.p.shape[0])

    def mapping(self):
        raise NotImplementedError("Mesh.mapping() not implemented!")

    def _validate(self):
        """Perform mesh validity checks."""
        # check that vertex matrix has "correct" size
        if(self.p.shape[0]>3):
            msg=("Mesh._validate(): We do not allow meshes "
                 "embedded into larger than 3-dimensional "
                 "Euclidean space! Please check that "
                 "the given vertex matrix is of size Ndim x Nvertices.")
            raise Exception(msg)
        # check that element connectivity matrix has correct size
        nvertices={'line':2,'tri':3,'quad':4,'tet':4}
        if(self.t.shape[0]!=nvertices[self.refdom]):
            msg=("Mesh._validate(): The given connectivity "
                 "matrix has wrong shape!")
            raise Exception(msg)
        # check that all points are at least in some element
        if(len(np.setdiff1d(np.arange(self.p.shape[1]),np.unique(self.t)))!=0):
            msg=("Mesh._validate(): Mesh contains a vertex "
                 "not belonging to any element.")
            raise Exception(msg)
        # check that there are no duplicate points
        tmp=np.ascontiguousarray(self.p.T)
        if(self.p.shape[1]!=np.unique(tmp.view([('',tmp.dtype)]*tmp.shape[1])).shape[0]):
            msg=("Mesh._validate(): Mesh contains duplicate "
                 "vertices.")
            raise Exception(msg)


class MeshLine(Mesh):
    """One-dimensional mesh."""
    
    refdom="line"
    brefdom="point"

    def __init__(self,p=np.array([[0,1]]),
                 t=np.array([[0],[1]]),
                 validate=True):
        self.p=p
        self.t=t
        if validate:
            self._validate()

    def refine(self,N=1):
        """Perform one or more refines on the mesh."""
        for itr in range(N):
            self.single_refine()

    def single_refine(self):
        """Perform a single mesh refine that halves 'h'."""
        # rename variables
        t=self.t
        p=self.p

        mid=range(self.t.shape[1])+np.max(t)+1
        # new vertices and elements
        newp=np.hstack((p,0.5*(p[:,self.t[0,:]]+p[:,self.t[1,:]])))
        newt=np.vstack((t[0,:],mid))
        newt=np.hstack((newt,np.vstack((mid,t[1,:]))))
        # update fields
        self.p=newp
        self.t=newt

    def plot(self,u,color='ko-'):
        """Plot a function defined on the nodes of the mesh."""
        plt.figure()

        xs=[]
        ys=[]
        for y1,y2,s,t in zip(u[self.t[0,:]],
                             u[self.t[1,:]],
                             self.p[0,self.t[0,:]],
                             self.p[0,self.t[1,:]]):
            xs.append(s)
            xs.append(t)
            xs.append(None)
            ys.append(y1)
            ys.append(y2)
            ys.append(None)
        plt.plot(xs,ys,color)

    def mapping(self):
        return fmap.MappingAffine(self)


class MeshQuad(Mesh):
    """Quadrilateral mesh."""
    
    refdom="quad"
    brefdom="line"

    def __init__(self,p=np.array([[0,0],[1,0],[1,1],[0,1]]).T,
                 t=np.array([[0,1,2,3]]).T,
                 validate=True):
        self.p=p
        self.t=t
        if validate:
            self._validate()
        self.build_mappings()

    def build_mappings(self):
        # do not sort since order defines counterclockwise order
        # self.t=np.sort(self.t,axis=0)

        # define facets: in the order (0,1) (1,2) (2,3) (0,3)
        self.facets=np.sort(np.vstack((self.t[0,:],self.t[1,:])),axis=0)
        self.facets=np.hstack((self.facets,
                               np.sort(np.vstack((self.t[1,:],self.t[2,:])),
                                       axis=0)))
        self.facets=np.hstack((self.facets,
                               np.sort(np.vstack((self.t[2,:],self.t[3,:])),
                                       axis=0)))
        self.facets=np.hstack((self.facets,
                               np.sort(np.vstack((self.t[0,:],self.t[3,:])),
                                       axis=0)))

        # get unique facets and build quad-to-facet mapping: 4 (edges) x Nquads
        tmp=np.ascontiguousarray(self.facets.T)
        tmp,ixa,ixb=np.unique(tmp.view([('',tmp.dtype)]*tmp.shape[1]),
                              return_index=True,return_inverse=True)
        self.facets=self.facets[:,ixa]
        self.t2f=ixb.reshape((4,self.t.shape[1]))

        # build facet-to-quadrilateral mapping: 2 (quads) x Nedges
        e_tmp=np.hstack((self.t2f[0,:],
                         self.t2f[1,:],
                         self.t2f[2,:],
                         self.t2f[3,:]))
        t_tmp=np.tile(np.arange(self.t.shape[1]),(1,4))[0]

        e_first,ix_first=np.unique(e_tmp,return_index=True)
        # this emulates matlab unique(e_tmp,'last')
        e_last,ix_last=np.unique(e_tmp[::-1],return_index=True)
        ix_last=e_tmp.shape[0]-ix_last-1

        self.f2t=np.zeros((2,self.facets.shape[1]),dtype=np.int64)
        self.f2t[0,e_first]=t_tmp[ix_first]
        self.f2t[1,e_last]=t_tmp[ix_last]

        # second row to -1 if repeated (i.e., on boundary)
        self.f2t[1,np.nonzero(self.f2t[0,:]==self.f2t[1,:])[0]]=-1

    def boundary_nodes(self):
        """Return an array of boundary node indices."""
        return np.unique(self.facets[:,self.boundary_facets()])

    def boundary_facets(self):
        """Return an array of boundary facet indices."""
        return np.nonzero(self.f2t[1,:]==-1)[0]

    def interior_nodes(self):
        """Return an array of interior node indices."""
        return np.setdiff1d(np.arange(0,self.p.shape[1]),self.boundary_nodes())

    def nodes_satisfying(self,test):
        """Return nodes that satisfy some condition."""
        return np.nonzero(test(self.p[0,:],self.p[1,:]))[0]

    def facets_satisfying(self,test):
        """Return facets whose midpoints satisfy some condition."""
        mx=0.5*(self.p[0,self.facets[0,:]]+self.p[0,self.facets[1,:]])
        my=0.5*(self.p[1,self.facets[0,:]]+self.p[1,self.facets[1,:]])
        return np.nonzero(test(mx,my))[0]

    def refine(self,N=1):
        """Perform one or more refines on the mesh."""
        for itr in range(N):
            self.single_refine()

    def single_refine(self):
        """Perform a single mesh refine that halves 'h'."""
        # rename variables
        t=self.t
        p=self.p
        e=self.facets
        sz=p.shape[1]
        t2f=self.t2f+sz
        mid=range(self.t.shape[1])+np.max(t2f)+1
        # new vertices are the midpoints of edges ...
        newp1=0.5*np.vstack((p[0,e[0,:]]+p[0,e[1,:]],p[1,e[0,:]]+p[1,e[1,:]]))
        # ... and element middle points
        newp2=0.25*np.vstack((p[0,t[0,:]]+p[0,t[1,:]]+p[0,t[2,:]]+p[0,t[3,:]],
                              p[1,t[0,:]]+p[1,t[1,:]]+p[1,t[2,:]]+p[1,t[3,:]]))
        newp=np.hstack((p,newp1,newp2))
        # build new quadrilateral definitions
        newt=np.vstack((t[0,:],t2f[0,:],mid,t2f[3,:]))
        newt=np.hstack((newt,np.vstack((t2f[0,:],t[1,:],t2f[1,:],mid))))
        newt=np.hstack((newt,np.vstack((mid,t2f[1,:],t[2,:],t2f[2,:]))))
        newt=np.hstack((newt,np.vstack((t2f[3,:],mid,t2f[2,:],t[3,:]))))
        # update fields
        self.p=newp
        self.t=newt

        self.build_mappings()

    def splitquads(self,z):
        """Split each quad to triangle."""
        if len(z)==self.t.shape[1]:
            # preserve elemental constant functions
            Z=np.concatenate((z,z))
        else:
            Z=z
        t=self.t[[0,1,3],:]
        t=np.hstack((t,self.t[[1,2,3]]))
        return MeshTri(self.p,t),Z

    def plot(self,z,smooth=False):
        """Visualize nodal or elemental function (2d)."""
        m,z=self.splitquads(z)
        return m.plot(z,smooth)

    def plot3(self,z,smooth=False):
        """Visualize nodal function (3d i.e. three axes)."""
        m,z=self.splitquads(z)
        return m.plot3(z,smooth)

    def jiggle(self,z=None):
        """Jiggle the interior nodes of the mesh."""
        if z is None:
            y=0.2*self.param()
        else:
            y=z*self.param()
        I=self.interior_nodes()
        self.p[0,I]=self.p[0,I]+y*np.random.rand(len(I))
        self.p[1,I]=self.p[1,I]+y*np.random.rand(len(I))

    def param(self):
        """Return mesh parameter."""
        return np.max(np.sqrt(np.sum((self.p[:,self.facets[0,:]]-
                                      self.p[:,self.facets[1,:]])**2,
                                     axis=0)))

    def draw(self):
        """Draw the mesh."""
        fig=plt.figure()
        # visualize the mesh
        # faster plotting is achieved through
        # None insertion trick.
        xs=[]
        ys=[]
        for s,t,u,v in zip(self.p[0,self.facets[0,:]],
                           self.p[1,self.facets[0,:]],
                           self.p[0,self.facets[1,:]],
                           self.p[1,self.facets[1,:]]):
            xs.append(s)
            xs.append(u)
            xs.append(None)
            ys.append(t)
            ys.append(v)
            ys.append(None)
        plt.plot(xs,ys,'k')
        return fig

    def mapping(self):
        return fmap.MappingQ1(self)


class MeshTet(Mesh):
    """Tetrahedral mesh."""
    
    refdom="tet"
    brefdom="tri"

    def __init__(self,p=np.array([[0,0,0],[0,0,1],[0,1,0],[1,0,0],
                                  [0,1,1],[1,0,1],[1,1,0],[1,1,1]]).T,
                 t=np.array([[0,1,2,3],[3,5,1,7],[2,3,6,7],
                             [2,3,1,7],[1,2,4,7]]).T,
                 validate=True):
        self.p=p
        self.t=t
        if validate:
            self._validate()
        self.build_mappings()

    def build_mappings(self):
        """Build element-to-facet, element-to-edges, etc. mappings."""
        # define edges: in the order (0,1) (1,2) (0,2) (0,3) (1,3) (2,3)
        self.edges=np.sort(np.vstack((self.t[0,:],self.t[1,:])),axis=0)
        e=np.array([1,2, 0,2, 0,3, 1,3, 2,3])
        for i in range(5):
            self.edges=np.hstack((self.edges,
                                  np.sort(np.vstack((self.t[e[2*i],:],
                                                     self.t[e[2*i+1],:])),
                                          axis=0)))

        # unique edges
        tmp=np.ascontiguousarray(self.edges.T)
        tmp,ixa,ixb=np.unique(tmp.view([('',tmp.dtype)]*tmp.shape[1]),
                              return_index=True,return_inverse=True)
        self.edges=self.edges[:,ixa]
        self.t2e=ixb.reshape((6,self.t.shape[1]))

        # define facets
        self.facets=np.sort(np.vstack((self.t[0,:],self.t[1,:],self.t[2,:])),
                            axis=0)
        f=np.array([0,1,3, 0,2,3, 1,2,3])
        for i in range(3):
            self.facets=np.hstack((self.facets,
                                   np.sort(np.vstack((self.t[f[2*i],:],
                                                      self.t[f[2*i+1],:],
                                                      self.t[f[2*i+2]])),
                                           axis=0)))

        # unique facets
        tmp=np.ascontiguousarray(self.facets.T)
        tmp,ixa,ixb=np.unique(tmp.view([('',tmp.dtype)]*tmp.shape[1]),
                              return_index=True,return_inverse=True)
        self.facets=self.facets[:,ixa]
        self.t2f=ixb.reshape((4,self.t.shape[1]))

        # build facet-to-tetra mapping: 2 (tets) x Nfacets
        e_tmp=np.hstack((self.t2f[0,:],self.t2f[1,:],
                         self.t2f[2,:],self.t2f[3,:]))
        t_tmp=np.tile(np.arange(self.t.shape[1]),(1,4))[0]

        e_first,ix_first=np.unique(e_tmp,return_index=True)
        # this emulates matlab unique(e_tmp,'last')
        e_last,ix_last=np.unique(e_tmp[::-1],return_index=True)
        ix_last=e_tmp.shape[0]-ix_last-1

        self.f2t=np.zeros((2,self.facets.shape[1]),dtype=np.int64)
        self.f2t[0,e_first]=t_tmp[ix_first]
        self.f2t[1,e_last]=t_tmp[ix_last]

        # second row to zero if repeated (i.e., on boundary)
        self.f2t[1,np.nonzero(self.f2t[0,:]==self.f2t[1,:])[0]]=-1

    def refine(self,N=1):
        """Perform one or more refines on the mesh."""
        for itr in range(N):
            self.single_refine()

    def nodes_satisfying(self,test):
        """Return nodes that satisfy some condition."""
        return np.nonzero(test(self.p[0,:],self.p[1,:],self.p[2,:]))[0]

    def facets_satisfying(self,test):
        """Return facets whose midpoints satisfy some condition."""
        mx=0.3333333*(self.p[0,self.facets[0,:]]+
                      self.p[0,self.facets[1,:]]+
                      self.p[0,self.facets[2,:]])
        my=0.3333333*(self.p[1,self.facets[0,:]]+
                      self.p[1,self.facets[1,:]]+
                      self.p[1,self.facets[2,:]])
        mz=0.3333333*(self.p[2,self.facets[0,:]]+
                      self.p[2,self.facets[1,:]]+
                      self.p[2,self.facets[2,:]])
        return np.nonzero(test(mx,my,mz))[0]

    def edges_satisfying(self,test):
        """Return edges whose midpoints satisfy some condition."""
        mx=0.5*(self.p[0,self.edges[0,:]]+self.p[0,self.edges[1,:]])
        my=0.5*(self.p[1,self.edges[0,:]]+self.p[1,self.edges[1,:]])
        mz=0.5*(self.p[2,self.edges[0,:]]+self.p[2,self.edges[1,:]])
        return np.nonzero(test(mx,my,mz))[0]

    def single_refine(self):
        """Perform a single mesh refine.
        
        Let the nodes of a tetrahedron be numbered as 0, 1, 2 and 3.
        It is assumed that the edges in self.t2e are given in the order

          I=(0,1), II=(1,2), III=(0,2), IV=(0,3), V=(1,3), VI=(2,3)

        by self.build_mappings(). Let I denote the midpoint of the edge
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
        t=self.t
        p=self.p
        e=self.edges
        sz=p.shape[1]
        t2e=self.t2e+sz
        # new vertices are the midpoints of edges
        newp=0.5*np.vstack((p[0,e[0,:]]+p[0,e[1,:]],
                            p[1,e[0,:]]+p[1,e[1,:]],
                            p[2,e[0,:]]+p[2,e[1,:]]))
        newp=np.hstack((p,newp))
        # new tets
        newt=np.vstack((t[0,:],t2e[0,:],t2e[2,:],t2e[3,:]))
        newt=np.hstack((newt,np.vstack((t[1,:],t2e[0,:],t2e[1,:],t2e[4,:]))))
        newt=np.hstack((newt,np.vstack((t[2,:],t2e[1,:],t2e[2,:],t2e[5,:]))))
        newt=np.hstack((newt,np.vstack((t[3,:],t2e[3,:],t2e[4,:],t2e[5,:]))))
        # compute middle pyramid diagonal lengths and choose shortest
        d1=((newp[0,t2e[2,:]]-newp[0,t2e[4,:]])**2+
            (newp[1,t2e[2,:]]-newp[1,t2e[4,:]])**2)
        d2=((newp[0,t2e[1,:]]-newp[0,t2e[3,:]])**2+
            (newp[1,t2e[1,:]]-newp[1,t2e[3,:]])**2)
        d3=((newp[0,t2e[0,:]]-newp[0,t2e[5,:]])**2+
            (newp[1,t2e[0,:]]-newp[1,t2e[5,:]])**2)
        I1=d1<d2
        I2=d1<d3
        I3=d2<d3
        c1=I1*I2
        c2=(-I1)*I3
        c3=(-I2)*(-I3)
        # splitting the pyramid in the middle.
        # diagonals are [2,4], [1,3] and [0,5]
        # CASE 1: diagonal [2,4]
        newt=np.hstack((newt,np.vstack((t2e[2,c1],t2e[4,c1],
                                        t2e[0,c1],t2e[1,c1]))))
        newt=np.hstack((newt,np.vstack((t2e[2,c1],t2e[4,c1],
                                        t2e[0,c1],t2e[3,c1]))))
        newt=np.hstack((newt,np.vstack((t2e[2,c1],t2e[4,c1],
                                        t2e[1,c1],t2e[5,c1]))))
        newt=np.hstack((newt,np.vstack((t2e[2,c1],t2e[4,c1],
                                        t2e[3,c1],t2e[5,c1]))))
        # CASE 2: diagonal [1,3]
        newt=np.hstack((newt,np.vstack((t2e[1,c2],t2e[3,c2],
                                        t2e[0,c2],t2e[4,c2]))))
        newt=np.hstack((newt,np.vstack((t2e[1,c2],t2e[3,c2],
                                        t2e[4,c2],t2e[5,c2]))))
        newt=np.hstack((newt,np.vstack((t2e[1,c2],t2e[3,c2],
                                        t2e[5,c2],t2e[2,c2]))))
        newt=np.hstack((newt,np.vstack((t2e[1,c2],t2e[3,c2],
                                        t2e[2,c2],t2e[0,c2]))))
        # CASE 3: diagonal [0,5]
        newt=np.hstack((newt,np.vstack((t2e[0,c3],t2e[5,c3],
                                        t2e[1,c3],t2e[4,c3]))))
        newt=np.hstack((newt,np.vstack((t2e[0,c3],t2e[5,c3],
                                        t2e[4,c3],t2e[3,c3]))))
        newt=np.hstack((newt,np.vstack((t2e[0,c3],t2e[5,c3],
                                        t2e[3,c3],t2e[2,c3]))))
        newt=np.hstack((newt,np.vstack((t2e[0,c3],t2e[5,c3],
                                        t2e[2,c3],t2e[1,c3]))))
        # update fields
        self.p=newp
        self.t=newt

        self.build_mappings()

    def draw_vertices(self):
        """Draw all vertices using mplot3d."""
        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        ax.scatter(self.p[0,:],self.p[1,:],self.p[2,:])
        return fig

    def draw_edges(self):
        """Draw all edges in a wireframe representation."""
        # use mayavi
        if opt_mayavi:
            mlab.triangular_mesh(self.p[0,:],self.p[1,:],self.p[2,:],
                                 self.facets.T,representation='wireframe',
                                 color=(0,0,0))
        else:
            raise ImportError("MeshTet: Mayavi not supported "
                              "by the host system!")

    def draw_facets(self,test=None,u=None):
        """Draw all facets."""
        if test is not None:
            xs=1./3.*(self.p[0,self.facets[0,:]]+
                      self.p[0,self.facets[1,:]]+
                      self.p[0,self.facets[2,:]])
            ys=1./3.*(self.p[1,self.facets[0,:]]+
                      self.p[1,self.facets[1,:]]+
                      self.p[1,self.facets[2,:]])
            zs=1./3.*(self.p[2,self.facets[0,:]]+
                      self.p[2,self.facets[1,:]]+
                      self.p[2,self.facets[2,:]])
            fset=np.nonzero(test(xs,ys,zs))[0]
        else:
            fset=range(self.facets.shape[1])

        # use mayavi
        if opt_mayavi:
            if u is None:
                mlab.triangular_mesh(self.p[0,:],self.p[1,:],self.p[2,:],
                                     self.facets[:,fset].T)
                mlab.triangular_mesh(self.p[0,:],self.p[1,:],self.p[2,:],
                                     self.facets[:,fset].T,
                                     representation='wireframe',color=(0,0,0))
            else:
                if u.shape[0]==self.facets.shape[1]:
                    newp=np.vstack((self.p[0,self.facets].flatten(order='F'),
                                    self.p[1,self.facets].flatten(order='F')))
                    newp=np.vstack((newp,self.p[2,self.facets].flatten(order='F')))
                    newt=np.arange(newp.shape[1]).reshape((3,newp.shape[1]/3),
                                                          order='F')
                    newu=np.tile(u,(3,1)).flatten(order='F')
                    mlab.triangular_mesh(newp[0,:],newp[1,:],newp[2,:],newt.T,
                                         scalars=newu)
                    mlab.triangular_mesh(newp[0,:],newp[1,:],newp[2,:],newt.T,
                                         representation='wireframe',color=(0,0,0))
                    mlab.axes()
                    mlab.colorbar()
                else:
                    raise Exception("MeshTet.draw_facets: given scalar data "+
                                    "shape not supported")
        else:
            raise ImportError("MeshTet: Mayavi not supported "
                              "by the host system!")

    def draw(self,test=None,u=None):
        """Draw all tetrahedra."""
        if test is not None:
            xs=1./4.*(self.p[0,self.t[0,:]]+
                      self.p[0,self.t[1,:]]+
                      self.p[0,self.t[2,:]]+
                      self.p[0,self.t[3,:]])
            ys=1./4.*(self.p[1,self.t[0,:]]+
                      self.p[1,self.t[1,:]]+
                      self.p[1,self.t[2,:]]+
                      self.p[1,self.t[3,:]])
            zs=1./4.*(self.p[2,self.t[0,:]]+
                      self.p[2,self.t[1,:]]+
                      self.p[2,self.t[2,:]]+
                      self.p[2,self.t[3,:]])
            tset=np.nonzero(test(xs,ys,zs))[0]
        else:
            tset=range(self.t.shape[1])

        fset=np.unique(self.t2f[:,tset].flatten())

        if u is None:
            u=self.p[2,:]

        if opt_mayavi:
            mlab.triangular_mesh(self.p[0,:],self.p[1,:],self.p[2,:],
                                 self.facets[:,fset].T,scalars=u)
            mlab.triangular_mesh(self.p[0,:],self.p[1,:],self.p[2,:],
                                 self.facets[:,fset].T,
                                 representation='wireframe',color=(0,0,0))
        else:
            raise ImportError("MeshTet: Mayavi not supported "
                              "by the host system!")

    def show(self):
        """Run mayavi's mlab.show() function to draw."""
        mlab.show()

    def boundary_nodes(self):
        """Return an array of boundary node indices."""
        return np.unique(self.facets[:,self.boundary_facets()])

    def boundary_facets(self):
        """Return an array of boundary facet indices."""
        return np.nonzero(self.f2t[1,:]==-1)[0]

    def boundary_edges(self):
        """Return an array of boundary edge indices."""
        bnodes=self.boundary_nodes()[:,None]
        return np.nonzero(np.sum(self.edges[0,:]==bnodes,axis=0)*
                          np.sum(self.edges[1,:]==bnodes,axis=0))[0]

    def interior_nodes(self):
        """Return an array of interior node indices."""
        return np.setdiff1d(np.arange(0,self.p.shape[1]),self.boundary_nodes())

    def param(self):
        """Return (maximum) mesh parameter."""
        return np.max(np.sqrt(np.sum((self.p[:,self.edges[0,:]]-
                                      self.p[:,self.edges[1,:]])**2,axis=0)))

    def shapereg(self):
        """Return the largest shape-regularity constant."""
        def edgelen(n):
            return np.sqrt(np.sum((self.p[:,self.edges[0,self.t2e[n,:]]]-
                                   self.p[:,self.edges[1,self.t2e[n,:]]])**2,
                                  axis=0))
        edgelenmat=np.vstack(tuple(edgelen(i) for i in range(6)))
        return np.max(np.max(edgelenmat,axis=0)/np.min(edgelenmat,axis=0))

    def mapping(self):
        return fmap.MappingAffine(self)


class MeshTri(Mesh):
    """Triangular mesh."""

    refdom="tri"
    brefdom="line"

    def __init__(self,p=np.array([[0,1,0,1],[0,0,1,1]],dtype=np.float_),
                 t=np.array([[0,1,2],[1,3,2]],dtype=np.intp).T,
                 validate=True):
        self.p=p
        self.t=t
        if validate:
            self._validate()
        self.build_mappings()

    def build_mappings(self):
        # sort to preserve orientations etc.
        self.t=np.sort(self.t,axis=0)

        # define facets: in the order (0,1) (1,2) (0,2)
        self.facets=np.sort(np.vstack((self.t[0,:],self.t[1,:])),axis=0)
        self.facets=np.hstack((self.facets,
                               np.sort(np.vstack((self.t[1,:],self.t[2,:])),
                                       axis=0)))
        self.facets=np.hstack((self.facets,
                               np.sort(np.vstack((self.t[0,:],self.t[2,:])),
                                       axis=0)))

        # get unique facets and build triangle-to-facet
        # mapping: 3 (edges) x Ntris
        tmp=np.ascontiguousarray(self.facets.T)
        tmp,ixa,ixb=np.unique(tmp.view([('',tmp.dtype)]*tmp.shape[1]),
                              return_index=True,return_inverse=True)
        self.facets=self.facets[:,ixa]
        self.t2f=ixb.reshape((3,self.t.shape[1]))

        # build facet-to-triangle mapping: 2 (triangles) x Nedges
        e_tmp=np.hstack((self.t2f[0,:],self.t2f[1,:],self.t2f[2,:]))
        t_tmp=np.tile(np.arange(self.t.shape[1]),(1,3))[0]

        e_first,ix_first=np.unique(e_tmp,return_index=True)
        # this emulates matlab unique(e_tmp,'last')
        e_last,ix_last=np.unique(e_tmp[::-1],return_index=True)
        ix_last=e_tmp.shape[0]-ix_last-1

        self.f2t=np.zeros((2,self.facets.shape[1]),dtype=np.int64)
        self.f2t[0,e_first]=t_tmp[ix_first]
        self.f2t[1,e_last]=t_tmp[ix_last]

        # second row to zero if repeated (i.e., on boundary)
        self.f2t[1,np.nonzero(self.f2t[0,:]==self.f2t[1,:])[0]]=-1

    def boundary_nodes(self):
        """Return an array of boundary node indices."""
        return np.unique(self.facets[:,self.boundary_facets()])

    def boundary_facets(self):
        """Return an array of boundary facet indices."""
        return np.nonzero(self.f2t[1,:]==-1)[0]

    def nodes_satisfying(self,test):
        """Return nodes that satisfy some condition."""
        return np.nonzero(test(self.p[0,:],self.p[1,:]))[0]

    def facets_satisfying(self,test):
        """Return facets whose midpoints satisfy some condition."""
        mx=0.5*(self.p[0,self.facets[0,:]]+self.p[0,self.facets[1,:]])
        my=0.5*(self.p[1,self.facets[0,:]]+self.p[1,self.facets[1,:]])
        return np.nonzero(test(mx,my))[0]

    def interior_nodes(self):
        """Return an array of interior node indices."""
        return np.setdiff1d(np.arange(0,self.p.shape[1]),self.boundary_nodes())

    def interpolator(self,x):
        """Return a function which interpolates values with P1 basis."""
        triang=mtri.Triangulation(self.p[0,:],self.p[1,:],self.t.T)
        interpf=mtri.LinearTriInterpolator(triang,x)
        # contruct an interpolator handle
        def handle(X,Y):
            return interpf(X,Y).data
        return handle

    def param(self):
        """Return mesh parameter."""
        return np.max(np.sqrt(np.sum((self.p[:,self.facets[0,:]]-
                                      self.p[:,self.facets[1,:]])**2,axis=0)))

    def draw(self):
        """Draw the mesh."""
        fig=plt.figure()
        # visualize the mesh faster plotting is achieved through
        # None insertion trick.
        xs=[]
        ys=[]
        for s,t,u,v in zip(self.p[0,self.facets[0,:]],
                           self.p[1,self.facets[0,:]],
                           self.p[0,self.facets[1,:]],
                           self.p[1,self.facets[1,:]]):
            xs.append(s)
            xs.append(u)
            xs.append(None)
            ys.append(t)
            ys.append(v)
            ys.append(None)
        plt.plot(xs,ys,'k')
        return fig

    def draw_nodes(self,nodes,mark='bo'):
        """Highlight some nodes."""
        if isinstance(nodes,str):
            try:
                plt.plot(self.p[0,self.markers[nodes]],
                         self.p[1,self.markers[nodes]],mark)
            except:
                raise Exception("MeshTri: Given node set name not found!")
        else:
            plt.plot(self.p[0,nodes],self.p[1,nodes],mark)

    def plot(self,z,smooth=False):
        """Visualize nodal or elemental function (2d)."""
        fig=plt.figure()
        if smooth:
            plt.tripcolor(self.p[0,:],self.p[1,:],self.t.T,z,
                                 shading='gouraud')
        else:
            plt.tripcolor(self.p[0,:],self.p[1,:],self.t.T,z)
        return fig

    def plot3(self,z,smooth=False):
        """Visualize nodal function (3d i.e. three axes)."""
        fig=plt.figure()
        if len(z)==self.p.shape[1]:
            # one value per node (piecewise linear, globally cont)
            if smooth:
                # use mayavi
                if opt_mayavi:
                    mlab.triangular_mesh(self.p[0,:],self.p[1,:],z,self.t.T)
                else:
                    raise ImportError("MeshTri: Mayavi not supported "
                                      "by the host system!")
            else:
                # use matplotlib
                ax=fig.gca(projection='3d')
                ts=mtri.Triangulation(self.p[0,:],self.p[1,:],self.t.T)
                ax.plot_trisurf(self.p[0,:],self.p[1,:],z,
                                triangles=ts.triangles,
                                cmap=plt.cm.Spectral)
        elif len(z)==self.t.shape[1]:
            # one value per element (piecewise const)
            nt=self.t.shape[1]
            newt=np.arange(3*nt,dtype=np.int64).reshape((nt,3))
            newpx=self.p[0,self.t].flatten(order='F')
            newpy=self.p[1,self.t].flatten(order='F')
            newz=np.vstack((z,z,z)).flatten(order='F')
            ax=fig.gca(projection='3d')
            ts=mtri.Triangulation(newpx,newpx,newt)
            ax.plot_trisurf(newpx,newpy,newz,
                            triangles=ts.triangles,
                            cmap=plt.cm.Spectral)
        else:
            raise NotImplementedError("MeshTri.plot3: not implemented for "
                                      "the given shape of input vector!")

    def show(self):
        """Call after plot functions."""
        plt.show()

    def refine(self,N=1):
        """Perform one or more refines on the mesh."""
        for itr in range(N):
            self.single_refine()

    def single_refine(self):
        """Perform a single mesh refine."""
        # rename variables
        t=self.t
        p=self.p
        e=self.facets
        sz=p.shape[1]
        t2f=self.t2f+sz
        # new vertices are the midpoints of edges
        newp=0.5*np.vstack((p[0,e[0,:]]+p[0,e[1,:]],p[1,e[0,:]]+p[1,e[1,:]]))
        newp=np.hstack((p,newp))
        # build new triangle definitions
        newt=np.vstack((t[0,:],t2f[0,:],t2f[2,:]))
        newt=np.hstack((newt,np.vstack((t[1,:],t2f[0,:],t2f[1,:]))))
        newt=np.hstack((newt,np.vstack((t[2,:],t2f[2,:],t2f[1,:]))))
        newt=np.hstack((newt,np.vstack((t2f[0,:],t2f[1,:],t2f[2,:]))))
        # update fields
        self.p=newp
        self.t=newt

        self.build_mappings()

    def mapping(self):
        return fmap.MappingAffine(self)


class MeshPyTri(MeshTri):
    """Simple wrapper for reading MeshPy triangular mesh."""
    def __init__(self,meshpy):
        p=np.array(meshpy.points).T
        t=np.array(meshpy.elements).T
        MeshTri.__init__(self,p,t)


class MeshPyTet(MeshTet):
    """Simple wrapper for reading MeshPy tetrahedral mesh."""
    def __init__(self,meshpy):
        p=np.array(meshpy.points).T
        t=np.array(meshpy.elements).T
        MeshTet.__init__(self,p,t)
