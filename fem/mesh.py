# -*- coding: utf-8 -*-
"""
Tools for various finite element meshes.

Try the following subclasses of Mesh:
    * MeshTri
    * MeshTet

@author: Tom Gustafsson
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import scipy.interpolate as spi
try:
    from mayavi import mlab
    opt_mayavi=True
except:
    opt_mayavi=False
from mpl_toolkits.mplot3d import Axes3D

class Mesh:
    """Superclass for all meshes."""

    p=np.empty([0,0],dtype=np.float_)
    t=np.empty([0,0],dtype=np.intp)

    def __init__(self,p,t):
        raise NotImplementedError("Mesh constructor not implemented!")

    def plot(self):
        raise NotImplementedError("Mesh.plot() not implemented!")

class MeshTet(Mesh):
    """Tetrahedral mesh."""
    p=np.empty([3,0],dtype=np.float_)
    t=np.empty([4,0],dtype=np.intp)
    facets=np.empty([3,0],dtype=np.intp)
    edges=np.empty([2,0],dtype=np.intp)
    t2f=np.empty([4,0],dtype=np.intp)
    f2t=np.empty([2,0],dtype=np.intp)
    t2e=np.empty([6,0],dtype=np.intp)
    f2e=np.empty([3,0],dtype=np.intp)
    refdom="tet"

    def __init__(self,p=np.array([[0,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[1,1,1]]).T,\
                      t=np.array([[0,1,2,3],[3,5,1,7],[2,3,6,7],[2,3,1,7],[1,2,4,7]]).T):
        self.p=p
        self.t=np.sort(t,axis=0)

        self.build_mappings()

    def build_mappings(self):
        """Build element-to-facet, element-to-edges, etc. mappings."""

        # define edges: in the order (0,1) (1,2) (0,2) (0,3) (1,3) (2,3)
        self.edges=np.sort(np.vstack((self.t[0,:],self.t[1,:])),axis=0)
        e=np.array([1,2, 0,2, 0,3, 1,3, 2,3])
        for i in range(5):
            self.edges=np.hstack((self.edges,np.sort(np.vstack((self.t[e[2*i],:],self.t[e[2*i+1],:])),axis=0)))

        # unique edges
        tmp=np.ascontiguousarray(self.edges.T)
        tmp,ixa,ixb=np.unique(tmp.view([('',tmp.dtype)]*tmp.shape[1]),return_index=True,return_inverse=True)
        self.edges=self.edges[:,ixa]
        self.t2e=ixb.reshape((6,self.t.shape[1]))

        # define facets
        self.facets=np.sort(np.vstack((self.t[0,:],self.t[1,:],self.t[2,:])),axis=0)
        f=np.array([0,1,3, 0,2,3, 1,2,3])
        for i in range(3):
            self.facets=np.hstack((self.facets,np.sort(np.vstack((self.t[f[2*i],:],self.t[f[2*i+1],:],self.t[f[2*i+2]])),axis=0)))

        # unique facets
        tmp=np.ascontiguousarray(self.facets.T)
        tmp,ixa,ixb=np.unique(tmp.view([('',tmp.dtype)]*tmp.shape[1]),return_index=True,return_inverse=True)
        self.facets=self.facets[:,ixa]
        self.t2f=ixb.reshape((4,self.t.shape[1]))
        
        # build facet-to-tetra mapping: 2 (tets) x Nfacets
        e_tmp=np.hstack((self.t2f[0,:],self.t2f[1,:],self.t2f[2,:],self.t2f[3,:]))
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

    def single_refine(self):
        """Perform a single mesh refine."""
        # rename variables
        t=self.t
        p=self.p
        e=self.edges
        sz=p.shape[1]
        t2e=self.t2e+sz
        # new vertices are the midpoints of edges
        newp=0.5*np.vstack((p[0,e[0,:]]+p[0,e[1,:]],\
                            p[1,e[0,:]]+p[1,e[1,:]],\
                            p[2,e[0,:]]+p[2,e[1,:]]))
        newp=np.hstack((p,newp))
        # new tets
        newt=np.vstack((t[0,:],t2e[0,:],t2e[2,:],t2e[3,:]))
        newt=np.hstack((newt,np.vstack((t[1,:],t2e[0,:],t2e[1,:],t2e[4,:]))))
        newt=np.hstack((newt,np.vstack((t[2,:],t2e[1,:],t2e[2,:],t2e[5,:]))))
        newt=np.hstack((newt,np.vstack((t[3,:],t2e[3,:],t2e[4,:],t2e[5,:]))))
        # 5 = (0,1) = 0
        # 6 = (0,2) = 2
        # 7 = (0,3) = 3
        # 8 = (1,2) = 1
        # 9 = (1,3) = 4
        #10 = (2,3) = 5
        #
        # 5 6 8 9
        newt=np.hstack((newt,np.vstack((t2e[0,:],t2e[2,:],t2e[1,:],t2e[4,:]))))
        # 5 6 7 9
        newt=np.hstack((newt,np.vstack((t2e[0,:],t2e[2,:],t2e[3,:],t2e[4,:]))))
        # 6 7 9 10
        newt=np.hstack((newt,np.vstack((t2e[2,:],t2e[3,:],t2e[4,:],t2e[5,:]))))
        # 6 8 9 10
        newt=np.hstack((newt,np.vstack((t2e[2,:],t2e[1,:],t2e[4,:],t2e[5,:]))))
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
        # use mayavi
        if opt_mayavi:
            #mlab.triangular_mesh(self.p[0,:],self.p[1,:],self.p[2,:],self.facets.T)
            mlab.triangular_mesh(self.p[0,:],self.p[1,:],self.p[2,:],self.facets.T,representation='wireframe',color=(0,0,0))
        else:
            raise ImportError("MeshTet: Mayavi not supported by the host system!")

    def draw_facets(self,test=None):
        """Draw _all_ facets."""
        if test is not None:
            xs=1./3.*(self.p[0,self.facets[0,:]]+self.p[0,self.facets[1,:]]+self.p[0,self.facets[2,:]])
            ys=1./3.*(self.p[1,self.facets[0,:]]+self.p[1,self.facets[1,:]]+self.p[1,self.facets[2,:]])
            zs=1./3.*(self.p[2,self.facets[0,:]]+self.p[2,self.facets[1,:]]+self.p[2,self.facets[2,:]])
            fset=np.nonzero(test(xs,ys,zs))[0]
        else:
            fset=range(self.facets.shape[1])

        # use mayavi
        if opt_mayavi:
            mlab.triangular_mesh(self.p[0,:],self.p[1,:],self.p[2,:],self.facets[:,fset].T)
            mlab.triangular_mesh(self.p[0,:],self.p[1,:],self.p[2,:],self.facets[:,fset].T,representation='wireframe',color=(0,0,0))
        else:
            raise ImportError("MeshTet: Mayavi not supported by the host system!")

    def draw(self,test=None,u=None):
        """Draw tetrahedra."""
        if test is not None:
            xs=1./4.*(self.p[0,self.t[0,:]]+\
                      self.p[0,self.t[1,:]]+\
                      self.p[0,self.t[2,:]]+\
                      self.p[0,self.t[3,:]])
            ys=1./4.*(self.p[1,self.t[0,:]]+\
                      self.p[1,self.t[1,:]]+\
                      self.p[1,self.t[2,:]]+\
                      self.p[1,self.t[3,:]])
            zs=1./4.*(self.p[2,self.t[0,:]]+\
                      self.p[2,self.t[1,:]]+\
                      self.p[2,self.t[2,:]]+\
                      self.p[2,self.t[3,:]])
            tset=np.nonzero(test(xs,ys,zs))[0]
        else:
            tset=range(self.t.shape[1])

        fset=np.unique(self.t2f[:,tset].flatten())

        if u is None:
            u=self.p[2,:]

        if opt_mayavi:
            mlab.triangular_mesh(self.p[0,:],self.p[1,:],self.p[2,:],self.facets[:,fset].T,scalars=u)
            mlab.triangular_mesh(self.p[0,:],self.p[1,:],self.p[2,:],self.facets[:,fset].T,representation='wireframe',color=(0,0,0))
        else:
            raise ImportError("MeshTet: Mayavi not supported by the host system!")
            
    def boundary_nodes(self):
        """Return an array of boundary node indices."""
        return np.unique(self.facets[:,self.boundary_facets()])
        
    def boundary_facets(self):
        """Return an array of boundary facet indices."""
        return np.nonzero(self.f2t[1,:]==-1)[0]

class MeshTri(Mesh):
    """Triangular mesh."""
    p=np.empty([2,0],dtype=np.float_)
    t=np.empty([3,0],dtype=np.intp)
    facets=np.empty([2,0],dtype=np.intp)
    t2f=np.empty([3,0],dtype=np.intp)
    f2t=np.empty([2,0],dtype=np.intp)
    refdom="tri"

    def __init__(self,p,t,fixmesh=False,markers=None,tmarkers=None):
        self.p=p
        self.t=np.sort(t,axis=0)
        
        # if the mesh is not proper (duplicate points) then fix it
        if fixmesh:
            tmp=np.ascontiguousarray(self.p.T)
            tmp,ixa,ixb=np.unique(tmp.view([('',tmp.dtype)]*tmp.shape[1]),return_index=True,return_inverse=True)
            self.p=self.p[:,ixa]
            self.t=np.sort(ixb[self.t],axis=0)
            if markers is not None:
                # fix markers
                fixedmarkers={}
                for key,value in markers.iteritems():
                    fixedmarkers[key]=ixb[value]
                markers=fixedmarkers
  
        # define facets: in the order (0,1) (1,2) (0,2)
        self.facets=np.sort(np.vstack((self.t[0,:],self.t[1,:])),axis=0)
        self.facets=np.hstack((self.facets,np.sort(np.vstack((self.t[1,:],self.t[2,:])),axis=0)))
        self.facets=np.hstack((self.facets,np.sort(np.vstack((self.t[0,:],self.t[2,:])),axis=0)))
  
        # get unique facets and build triangle-to-facet mapping: 3 (edges) x Ntris
        tmp=np.ascontiguousarray(self.facets.T)
        tmp,ixa,ixb=np.unique(tmp.view([('',tmp.dtype)]*tmp.shape[1]),return_index=True,return_inverse=True)
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
        
        if markers is None:
            self.markers={}
            self.markers['boundary']=self.boundary_nodes()
            self.markers['interior']=self.interior_nodes()
        else:
            self.markers=markers

        self.tmarkers=tmarkers

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
        # TODO make this faster (i.e. use the mesh in self)
        return spi.LinearNDInterpolator(self.p.T,x)
        
    def param(self):
        """Return mesh parameter."""
        return np.max(np.sqrt(np.sum((self.p[:,self.facets[0,:]]-self.p[:,self.facets[1,:]])**2,axis=0)))

    def draw(self):
        """Draw the mesh."""
        fig=plt.figure()
        # visualize the mesh
        # faster plotting is achieved through
        # None insertion trick.
        xs=[]
        ys=[]
        for s,t,u,v in zip(self.p[0,self.facets[0,:]],self.p[1,self.facets[0,:]],self.p[0,self.facets[1,:]],self.p[1,self.facets[1,:]]):
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
                plt.plot(self.p[0,self.markers[nodes]],self.p[1,self.markers[nodes]],mark)
            except:
                raise Exception(self.__class__.__name__+": Given node set name not found!")
        else:
            plt.plot(self.p[0,nodes],self.p[1,nodes],mark)


    def plot(self,z,smooth=False):
        """Visualize nodal or elemental function (2d)."""
        fig=plt.figure()
        if smooth:
            return plt.tripcolor(self.p[0,:],self.p[1,:],self.t.T,z,shading='gouraud')
        else:
            return plt.tripcolor(self.p[0,:],self.p[1,:],self.t.T,z)

    def plot3(self,z,smooth=False):
        """Visualize nodal function (3d i.e. three axes)."""
        fig=plt.figure()
        # visualize a solution vector
        if smooth:
            # use mayavi
            if opt_mayavi:
                mlab.triangular_mesh(self.p[0,:],self.p[1,:],z,self.t.T)
            else:
                raise ImportError("Mayavi not supported by the host system!")
        else:
            # use matplotlib
            ax=fig.gca(projection='3d')
            ts=mtri.Triangulation(self.p[0,:],self.p[1,:],self.t.T)
            ax.plot_trisurf(self.p[0,:],self.p[1,:],z,triangles=ts.triangles,cmap=plt.cm.Spectral)

    def show(self):
        """Call after plot functions."""
        if opt_mayavi:
            mlab.show()
        else:
            plt.show()



