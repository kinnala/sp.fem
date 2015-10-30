import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
try:
    from mayavi import mlab
    opt_mayavi=True
except:
    opt_mayavi=False
from mpl_toolkits.mplot3d import Axes3D

class Mesh:
    """
    Superclass for all meshes.
    """

    p=np.empty([0,0],dtype=np.float_)
    t=np.empty([0,0],dtype=np.intp)

    def __init__(self,p,t):
        raise NotImplementedError("Mesh constructor not implemented!")

    def plot(self):
        raise NotImplementedError("Mesh.plot() not implemented!")

class MeshTri(Mesh):
    """
    Triangular mesh.
    """

    p=np.empty([2,0],dtype=np.float_)
    t=np.empty([3,0],dtype=np.intp)
    facets=np.empty([2,0],dtype=np.intp)
    t2f=np.empty([3,0],dtype=np.intp)
    f2t=np.empty([2,0],dtype=np.intp)

    def __init__(self,p,t):
        self.p=p
        self.t=t
        self.t.sort(axis=0)
  
        # define facets
        self.facets=np.vstack((self.t[0,:],self.t[1,:]))
        self.facets=np.hstack((self.facets,np.vstack((self.t[1,:],self.t[2,:]))))
        self.facets=np.hstack((self.facets,np.vstack((self.t[0,:],self.t[2,:]))))
  
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

    def boundary_nodes(self):
        """
        Return an array of boundary node indices.
        """
        return np.unique(self.facets[:,np.nonzero(self.f2t[1,:]==-1)[0]])

    def plot(self,z=None,smooth=False):
        """
        Draw the mesh or visualize nodal function.
        """
        fig=plt.figure()
        if z is None:
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

        else:
            # visualize a solution vector
            if smooth:
                # use mayavi
                if opt_mayavi:
                    mlab.triangular_mesh(self.p[0,:],self.p[1,:],z,self.t.T)
                else:
                    raise ImportError("Mayavi not imported because it is not installed!")
            else:
                # use matplotlib
                ax=fig.gca(projection='3d')
                ts=mtri.Triangulation(self.p[0,:],self.p[1,:],self.t.T)
                ax.plot_trisurf(self.p[0,:],self.p[1,:],z,triangles=ts.triangles,cmap=plt.cm.Spectral)

    def show(self):
        """
        Call after plot functions.
        """
        if opt_mayavi:
            mlab.show()
        else:
            plt.show()



