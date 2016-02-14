import numpy as np
import fem.mesh
import copy

class Mapping:
    """Abstract superclass for mappings.

    Mappings eat Meshes (possibly Geometries in isoparametric case?)
    and allow local-to-global and global-to-local mappings.
    """
    dim=0

    def __init__(self,mesh):
        raise NotImplementedError("Mapping constructor not implemented!")

    def F(self,X,tind):
        """Element local to global."""
        raise NotImplementedError("Mapping.F() not implemented!")

    def invF(self,x,tind):
        raise NotImplementedError("Mapping.invF() not implemented!")

    def DF(self,X,tind):
        raise NotImplementedError("Mapping.DF() not implemented!")

    def invDF(self,X,tind):
        raise NotImplementedError("Mapping.invDF() not implemented!")

    def detDF(self,X,tind):
        raise NotImplementedError("Mapping.detDF() not implemented!")

    def G(self,X,find):
        """Boundary local to global."""
        raise NotImplementedError("Mapping.G() not implemented!")

class MappingAffine(Mapping):
    """Affine mappings for simplex (=line,tri,tet) mesh."""
    def __init__(self,mesh):
        if isinstance(mesh,fem.mesh.MeshTri):
            self.dim=2            
            
            self.A={0:{},1:{}}
    
            self.A[0][0]=mesh.p[0,mesh.t[1,:]]-mesh.p[0,mesh.t[0,:]]
            self.A[0][1]=mesh.p[0,mesh.t[2,:]]-mesh.p[0,mesh.t[0,:]]
            self.A[1][0]=mesh.p[1,mesh.t[1,:]]-mesh.p[1,mesh.t[0,:]]
            self.A[1][1]=mesh.p[1,mesh.t[2,:]]-mesh.p[1,mesh.t[0,:]]
    
            self.b={}
    
            self.b[0]=mesh.p[0,mesh.t[0,:]]
            self.b[1]=mesh.p[1,mesh.t[0,:]]
    
            self.detA=self.A[0][0]*self.A[1][1]-self.A[0][1]*self.A[1][0]
    
            self.invA={0:{},1:{}}
    
            self.invA[0][0]=self.A[1][1]/self.detA
            self.invA[0][1]=-self.A[0][1]/self.detA
            self.invA[1][0]=-self.A[1][0]/self.detA
            self.invA[1][1]=self.A[0][0]/self.detA 
    
            # Matrices and vectors for boundary mappings: G(X)=BX+c
            self.B={}
    
            self.B[0]=mesh.p[0,mesh.facets[1,:]]-mesh.p[0,mesh.facets[0,:]]
            self.B[1]=mesh.p[1,mesh.facets[1,:]]-mesh.p[1,mesh.facets[0,:]]
    
            self.c={}
    
            self.c[0]=mesh.p[0,mesh.facets[0,:]]
            self.c[1]=mesh.p[1,mesh.facets[0,:]]
    
            self.detB=np.sqrt(self.B[0]**2+self.B[1]**2)
            
        elif isinstance(mesh,fem.mesh.MeshTet):
            self.dim=3            
            
            self.A={0:{},1:{},2:{}}
    
            self.A[0][0]=mesh.p[0,mesh.t[1,:]]-mesh.p[0,mesh.t[0,:]]
            self.A[0][1]=mesh.p[0,mesh.t[2,:]]-mesh.p[0,mesh.t[0,:]]
            self.A[0][2]=mesh.p[0,mesh.t[3,:]]-mesh.p[0,mesh.t[0,:]]
            self.A[1][0]=mesh.p[1,mesh.t[1,:]]-mesh.p[1,mesh.t[0,:]]
            self.A[1][1]=mesh.p[1,mesh.t[2,:]]-mesh.p[1,mesh.t[0,:]]
            self.A[1][2]=mesh.p[1,mesh.t[3,:]]-mesh.p[1,mesh.t[0,:]]
            self.A[2][0]=mesh.p[2,mesh.t[1,:]]-mesh.p[2,mesh.t[0,:]]
            self.A[2][1]=mesh.p[2,mesh.t[2,:]]-mesh.p[2,mesh.t[0,:]]
            self.A[2][2]=mesh.p[2,mesh.t[3,:]]-mesh.p[2,mesh.t[0,:]]
    
            self.b={}
    
            self.b[0]=mesh.p[0,mesh.t[0,:]]
            self.b[1]=mesh.p[1,mesh.t[0,:]]
            self.b[2]=mesh.p[2,mesh.t[0,:]]
    
            self.detA=self.A[0][0]*(self.A[1][1]*self.A[2][2]-self.A[1][2]*self.A[2][1])\
                      -self.A[0][1]*(self.A[1][0]*self.A[2][2]-self.A[1][2]*self.A[2][0])\
                      +self.A[0][2]*(self.A[1][0]*self.A[2][1]-self.A[1][1]*self.A[2][0])
    
            self.invA={0:{},1:{},2:{}}
    
            self.invA[0][0]=(-self.A[1][2]*self.A[2][1]+self.A[1][1]*self.A[2][2])/self.detA
            self.invA[1][0]=( self.A[1][2]*self.A[2][0]-self.A[1][0]*self.A[2][2])/self.detA
            self.invA[2][0]=(-self.A[1][1]*self.A[2][0]+self.A[1][0]*self.A[2][1])/self.detA
            self.invA[0][1]=( self.A[0][2]*self.A[2][1]-self.A[0][1]*self.A[2][2])/self.detA
            self.invA[1][1]=(-self.A[0][2]*self.A[2][0]+self.A[0][0]*self.A[2][2])/self.detA
            self.invA[2][1]=( self.A[0][1]*self.A[2][0]-self.A[0][0]*self.A[2][1])/self.detA
            self.invA[0][2]=(-self.A[0][2]*self.A[1][1]+self.A[0][1]*self.A[1][2])/self.detA
            self.invA[1][2]=( self.A[0][2]*self.A[1][0]-self.A[0][0]*self.A[1][2])/self.detA
            self.invA[2][2]=(-self.A[0][1]*self.A[1][0]+self.A[0][0]*self.A[1][1])/self.detA
    
            # TODO Matrices and vectors for boundary mappings: G(X)=BX+c
            self.B={}
    
            self.B[0]=mesh.p[0,mesh.facets[1,:]]-mesh.p[0,mesh.facets[0,:]]
            self.B[1]=mesh.p[1,mesh.facets[1,:]]-mesh.p[1,mesh.facets[0,:]]
    
            self.c={}
    
            self.c[0]=mesh.p[0,mesh.facets[0,:]]
            self.c[1]=mesh.p[1,mesh.facets[0,:]]
    
            self.detB=np.sqrt(self.B[0]**2+self.B[1]**2)
            
        else:
            raise TypeError("MappingAffine initialized with an incompatible mesh type!")

    def F(self,X,tind=None):
        """Affine map F(X)=AX+b.

        Acts on Ndim x Npoints matrices
        and returns Dict D with D[0] corresponding to
        x-coordinates and D[1] corresponding to y-coordinates.
        Both D[0] and D[1] are Nelems x Npoints.
        """
        y={}
        if self.dim==2:
            if tind is None:
                y[0]=np.outer(self.A[0][0],X[0,:]).T+np.outer(self.A[0][1],X[1,:]).T+self.b[0]
                y[1]=np.outer(self.A[1][0],X[0,:]).T+np.outer(self.A[1][1],X[1,:]).T+self.b[1]
            else: # TODO check this could have error
                y[0]=np.outer(self.A[0][0][tind],X[0,:]).T+np.outer(self.A[0][1][tind],X[1,:]).T+self.b[0][tind]
                y[1]=np.outer(self.A[1][0][tind],X[0,:]).T+np.outer(self.A[1][1][tind],X[1,:]).T+self.b[1][tind]
            y[0]=y[0].T
            y[1]=y[1].T
        elif self.dim==3:
            if tind is None:
                y[0]=np.outer(self.A[0][0],X[0,:]).T+\
                     np.outer(self.A[0][1],X[1,:]).T+\
                     np.outer(self.A[0][2],X[2,:]).T+self.b[0]
                y[1]=np.outer(self.A[1][0],X[0,:]).T+\
                     np.outer(self.A[1][1],X[1,:]).T+\
                     np.outer(self.A[1][2],X[2,:]).T+self.b[1]
                y[2]=np.outer(self.A[2][0],X[0,:]).T+\
                     np.outer(self.A[2][1],X[1,:]).T+\
                     np.outer(self.A[2][2],X[2,:]).T+self.b[2]
            else: # TODO check this could have error
                y[0]=np.outer(self.A[0][0][tind],X[0,:]).T+\
                     np.outer(self.A[0][1][tind],X[1,:]).T+\
                     np.outer(self.A[0][2][tind],X[2,:]).T+self.b[0][tind]
                y[1]=np.outer(self.A[1][0][tind],X[0,:]).T+\
                     np.outer(self.A[1][1][tind],X[1,:]).T+\
                     np.outer(self.A[1][2][tind],X[2,:]).T+self.b[1][tind]
                y[2]=np.outer(self.A[2][0][tind],X[0,:]).T+\
                     np.outer(self.A[2][1][tind],X[1,:]).T+\
                     np.outer(self.A[2][2][tind],X[2,:]).T+self.b[2][tind]
            y[0]=y[0].T
            y[1]=y[1].T
            y[2]=y[2].T
        return y

    def invF(self,x,tind=None):
        """Inverse map F^{-1}(x)=A^{-1}(x-b)."""
        Y={}
        y={}
        if tind is None:
            Y[0]=x[0].T-self.b[0]
            Y[1]=x[1].T-self.b[1]
            y[0]=self.invA[0][0]*Y[0]+self.invA[0][1]*Y[1]
            y[1]=self.invA[1][0]*Y[0]+self.invA[1][1]*Y[1]
        else:
            Y[0]=x[0].T-self.b[0][tind]
            Y[1]=x[1].T-self.b[1][tind]
            y[0]=self.invA[0][0][tind]*Y[0]+self.invA[0][1][tind]*Y[1]
            y[1]=self.invA[1][0][tind]*Y[0]+self.invA[1][1][tind]*Y[1]
        y[0]=y[0].T
        y[1]=y[1].T
        return y

    def G(self,X,find=None):
        """Boundary mapping G(X)=Bx+c."""
        y={}
        if find is None:
            y[0]=np.outer(self.B[0],X).T+self.c[0]
            y[1]=np.outer(self.B[1],X).T+self.c[1]
        else:
            y[0]=np.outer(self.B[0][find],X).T+self.c[0][find]
            y[1]=np.outer(self.B[1][find],X).T+self.c[1][find]
        y[0]=y[0].T
        y[1]=y[1].T
        return y
        
    def detDF(self,X,tind=None):
        if tind is None:
            detDF=self.detA
        else:
            detDF=self.detA[tind]
        return np.tile(detDF,(X.shape[1],1)).T  
        
    def invDF(self,X,tind=None):
        invA=copy.deepcopy(self.invA)
        
        if self.dim==2:
            if tind is None: # TODO did not test
                invA[0][0]=np.tile(invA[0][0],(X.shape[1],1)).T
                invA[0][1]=np.tile(invA[0][1],(X.shape[1],1)).T
                invA[1][0]=np.tile(invA[1][0],(X.shape[1],1)).T
                invA[1][1]=np.tile(invA[1][1],(X.shape[1],1)).T
            else:
                invA[0][0]=np.tile(invA[0][0][tind],(X.shape[1],1)).T
                invA[0][1]=np.tile(invA[0][1][tind],(X.shape[1],1)).T
                invA[1][0]=np.tile(invA[1][0][tind],(X.shape[1],1)).T
                invA[1][1]=np.tile(invA[1][1][tind],(X.shape[1],1)).T
        if self.dim==3:
            if tind is None: # TODO did not test
                invA[0][0]=np.tile(invA[0][0],(X.shape[1],1)).T
                invA[0][1]=np.tile(invA[0][1],(X.shape[1],1)).T
                invA[0][2]=np.tile(invA[0][2],(X.shape[1],1)).T
                invA[1][0]=np.tile(invA[1][0],(X.shape[1],1)).T
                invA[1][1]=np.tile(invA[1][1],(X.shape[1],1)).T
                invA[1][2]=np.tile(invA[1][2],(X.shape[1],1)).T
                invA[2][0]=np.tile(invA[2][0],(X.shape[1],1)).T
                invA[2][1]=np.tile(invA[2][1],(X.shape[1],1)).T
                invA[2][2]=np.tile(invA[2][2],(X.shape[1],1)).T
            else:
                invA[0][0]=np.tile(invA[0][0][tind],(X.shape[1],1)).T
                invA[0][1]=np.tile(invA[0][1][tind],(X.shape[1],1)).T
                invA[0][2]=np.tile(invA[0][2][tind],(X.shape[1],1)).T
                invA[1][0]=np.tile(invA[1][0][tind],(X.shape[1],1)).T
                invA[1][1]=np.tile(invA[1][1][tind],(X.shape[1],1)).T
                invA[1][2]=np.tile(invA[1][2][tind],(X.shape[1],1)).T
                invA[2][0]=np.tile(invA[2][0][tind],(X.shape[1],1)).T
                invA[2][1]=np.tile(invA[2][1][tind],(X.shape[1],1)).T
                invA[2][2]=np.tile(invA[2][2][tind],(X.shape[1],1)).T           
                
        return invA


        

