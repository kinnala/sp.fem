import numpy as np
import fem.mesh

class Geometry:
  """
  Superclass for all geometries.
  """

  def __init__(self):
    raise NotImplementedError("Geometry constructor not implemented!")

  def mesh(self):
    raise NotImplementedError("Geometry mesher not implemented!")

class GeometryMeshTri(Geometry):
  """
  A geometry defined by a triangular mesh.
  """

  p=np.empty([2,0],dtype=np.float_)
  t=np.empty([3,0],dtype=np.intp)

  def __init__(self,p=np.array([[0,1,0,1],[0,0,1,1]],dtype=np.float_),t=np.array([[0,1,2],[1,2,3]],dtype=np.intp).T):
    self.p=p
    self.t=t

  def mesh(self):
    return fem.mesh.MeshTri(self.p,self.t)

  def refine(self,N=1):
    """
    Perform one or more refines on the mesh.
    """
    for itr in range(N):
      self.single_refine()

  def single_refine(self):
    """
    Perform a single mesh refine.
    """
    mesh=fem.mesh.MeshTri(self.p,self.t)
    # rename variables
    t=mesh.t
    p=mesh.p
    e=mesh.facets
    t2f=mesh.t2f
    # new vertices are the midpoints of edges
    newp=0.5*np.vstack((p[0,e[0,:]]+p[0,e[1,:]],p[1,e[0,:]]+p[1,e[1,:]]))
    newp=np.hstack((p,newp))
    # build new triangle definitions
    sz=p.shape[1]
    newt=np.vstack((t[0,:],t2f[0,:]+sz,t2f[2,:]+sz))
    newt=np.hstack((newt,np.vstack((t[1,:],t2f[0,:]+sz,t2f[1,:]+sz))))
    newt=np.hstack((newt,np.vstack((t[2,:],t2f[2,:]+sz,t2f[1,:]+sz))))
    newt=np.hstack((newt,np.vstack((t2f[0,:]+sz,t2f[1,:]+sz,t2f[2,:]+sz))))
    # update fields
    self.p=newp
    self.t=newt
  
