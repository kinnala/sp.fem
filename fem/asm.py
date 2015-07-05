import numpy as np
import fem.mesh
from scipy.sparse import coo_matrix

class Assembler:
  """
  Superclass for assemblers.
  """

  def __init__(self):
    raise NotImplementedError("Assembler constructor not implemented!")

  def affine_map_tri(self,mesh):
    """
    Build affine mappings F(x)=Ax+b for all triangles of a mesh object. In addition, computes determinants and inverse of A.
    """
    if mesh.p.shape[0]==2:
      A={0:{},1:{}}

      A[0][0]=mesh.p[0,mesh.t[1,:]]-mesh.p[0,mesh.t[0,:]]
      A[0][1]=mesh.p[0,mesh.t[2,:]]-mesh.p[0,mesh.t[0,:]]
      A[1][0]=mesh.p[1,mesh.t[1,:]]-mesh.p[1,mesh.t[0,:]]
      A[1][1]=mesh.p[1,mesh.t[2,:]]-mesh.p[1,mesh.t[0,:]]

      b={}

      b[0]=mesh.p[0,mesh.t[:,0]]
      b[1]=mesh.p[0,mesh.t[:,0]]

      detA=A[0][0]*A[1][1]-A[0][1]*A[1][0]

      invA={0:{},1:{}}

      invA[0][0]=A[1][1]/detA
      invA[0][1]=-A[0][1]/detA
      invA[1][0]=-A[1][0]/detA
      invA[1][1]=A[0][0]/detA

    else:
      raise NotImplementedError("Assembler affine_map not implemented for specified dimension!")

    return A,b,detA,invA

class AssemblerTriP1(Assembler):
  """
  A fast (bi)linear form assembler with triangular P1 Lagrange elements.
  """
  def __init__(self,mesh):
    self.A,self.b,self.detA,self.invA=self.affine_map_tri(mesh)
    self.mesh=mesh

  def iasm(self,form):
    """
    Interior assembly.
    """
    nv=self.mesh.p.shape[1]
    nt=self.mesh.t.shape[1]
    
    # quadrature points and weights (2nd order accurate)
    qp=np.array([[1.666666666666666666666e-01,6.666666666666666666666e-01,1.666666666666666666666e-01],[1.666666666666666666666e-01,1.666666666666666666666e-01,6.666666666666666666666e-01]])
    qw=np.array([1.666666666666666666666e-01,1.666666666666666666666e-01,1.666666666666666666666e-01])

    # local basis functions
    phi={}
    phi[0]=1.-qp[0,:]-qp[1,:]
    phi[1]=qp[0,:]
    phi[2]=qp[1,:]

    # local basis function gradients
    gradphi={}
    gradphi[0]=np.tile(np.array([-1.,-1.]),(qp.shape[1],1)).T
    gradphi[1]=np.tile(np.array([1.,0.]),(qp.shape[1],1)).T
    gradphi[2]=np.tile(np.array([0.,1.]),(qp.shape[1],1)).T    
    
    # bilinear form
    if form.__code__.co_argcount==5:
        # initialize sparse matrix structures
        data=np.zeros(9*nt)
        rows=np.zeros(9*nt)
        cols=np.zeros(9*nt)
    
        # TODO global coords and interpolation
        x={}
    
        for j in [0,1,2]:
          u=np.tile(phi[j],(nt,1))
          du={}
          du[0]=np.outer(self.invA[0][0],gradphi[j][0,:])+\
                np.outer(self.invA[1][0],gradphi[j][1,:])
          du[1]=np.outer(self.invA[0][1],gradphi[j][0,:])+\
                np.outer(self.invA[1][1],gradphi[j][1,:])
          for i in [0,1,2]:
            v=np.tile(phi[i],(nt,1))
            dv={}
            dv[0]=np.outer(self.invA[0][0],gradphi[i][0,:])+\
                  np.outer(self.invA[1][0],gradphi[i][1,:])
            dv[1]=np.outer(self.invA[0][1],gradphi[i][0,:])+\
                  np.outer(self.invA[1][1],gradphi[i][1,:])
    
            # find correct location in data,rows,cols
            ixs=slice(nt*(3*j+i),nt*(3*j+i+1))
            
            # compute entries of local stiffness matrices
            data[ixs]=np.dot(form(u,v,du,dv,x),qw)*np.abs(self.detA)
            rows[ixs]=self.mesh.t[i,:]
            cols[ixs]=self.mesh.t[j,:]
    
        return coo_matrix((data,(rows,cols)),shape=(nv,nv)).tocsr()

    # linear form
    elif form.__code__.co_argcount==3:
        # initialize sparse matrix structures
        data=np.zeros(3*nt)
        rows=np.zeros(3*nt)
        cols=np.zeros(3*nt)
        
        # TODO global coords and interpolation
        x={}
    
        for i in [0,1,2]:
            v=np.tile(phi[i],(nt,1))
            dv={}
            dv[0]=np.outer(self.invA[0][0],gradphi[i][0,:])+\
                  np.outer(self.invA[1][0],gradphi[i][1,:])
            dv[1]=np.outer(self.invA[0][1],gradphi[i][0,:])+\
                  np.outer(self.invA[1][1],gradphi[i][1,:])
            
            # find correct location in data,rows,cols
            ixs=slice(nt*i,nt*(i+1))
            
            # compute entries of local stiffness matrices
            data[ixs]=np.dot(form(v,dv,x),qw)*np.abs(self.detA)
            rows[ixs]=self.mesh.t[i,:]
            cols[ixs]=np.zeros(nt)
    
        return coo_matrix((data,(rows,cols)),shape=(nv,1)).toarray().T[0]
    else:
        raise NotImplementedError("AssemblerTriP1 iasm not implemented for the given number of form arguments!")

