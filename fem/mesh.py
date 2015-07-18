import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mayavi import mlab
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
  t2f=np.empty([2,0],dtype=np.intp)

  def __init__(self,p,t):
    self.p=p
    self.t=t
    self.t.sort(axis=0)

    # define facets
    self.facets=np.vstack((self.t[0,:],self.t[1,:]))
    self.facets=np.hstack((self.facets,np.vstack((self.t[1,:],self.t[2,:]))))
    self.facets=np.hstack((self.facets,np.vstack((self.t[0,:],self.t[2,:]))))

    # get unique facets and build triangle-to-facet mapping
    tmp=np.ascontiguousarray(self.facets.T)
    tmp,ixa,ixb=np.unique(tmp.view([('',tmp.dtype)]*tmp.shape[1]),return_index=True,return_inverse=True)
    self.facets=self.facets[:,ixa]
    self.t2f=ixb.reshape((3,self.t.shape[1]))

    # build facet-to-triangle mapping
    e_tmp=np.hstack((self.t2f[0,:],self.t2f[1,:],self.t2f[2,:]))
    t_tmp=np.tile(np.arange(self.t.shape[1]),(1,3))
  
  def boundary_nodes(self):
    """
    Return an array of boundary node indices.
    """
    return np.nonzero(self.f2t[1,:]==0)[0] # TODO this seems to be broken

  def plot(self,z=None,smooth=False):
    """
    Draw the mesh or visualize nodal function.
    """
    fig=plt.figure()
    if z is None:
      # visualize the mesh
      xs=np.vstack((self.p[0,self.facets[0,:]],self.p[0,self.facets[1,:]]))
      ys=np.vstack((self.p[1,self.facets[0,:]],self.p[1,self.facets[1,:]]))
      plt.plot(xs,ys,'k')
      plt.show()
    else:
      # visualize a solution vector
      if smooth:
        # use mayavi
        mlab.triangular_mesh(self.p[0,:],self.p[1,:],z,self.t.T)
        mlab.show()
      else:
        # use matplotlib
        ax=fig.gca(projection='3d')
        ts=mtri.Triangulation(self.p[0,:],self.p[1,:],self.t.T)
        ax.plot_trisurf(self.p[0,:],self.p[1,:],z,triangles=ts.triangles,cmap=plt.cm.Spectral)
        plt.show()


