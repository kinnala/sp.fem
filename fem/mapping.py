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
        
    def detDG(self,X,find):
        raise NotImplementedError("Mapping.detDG() not implemented!")

class MappingQ1(Mapping):
    """Mapping for quadrilaterals."""
    
    def __init__(self,mesh):
        if isinstance(mesh,fem.mesh.MeshQuad):
            self.dim=2
            
            self.t=mesh.t
            self.p=mesh.p
            
            self.J={0:{},1:{}}
            
            self.J[0][0]=lambda x,y: 0.25*(-mesh.p[0,mesh.t[0,:]][:,None]*(1-y)+\
                                            mesh.p[0,mesh.t[1,:]][:,None]*(1-y)+\
                                            mesh.p[0,mesh.t[2,:]][:,None]*(1+y)-\
                                            mesh.p[0,mesh.t[3,:]][:,None]*(1+y))
            self.J[0][1]=lambda x,y: 0.25*(-mesh.p[0,mesh.t[0,:]][:,None]*(1-x)-\
                                            mesh.p[0,mesh.t[1,:]][:,None]*(1+x)+\
                                            mesh.p[0,mesh.t[2,:]][:,None]*(1+x)+\
                                            mesh.p[0,mesh.t[3,:]][:,None]*(1-x))
            self.J[1][0]=lambda x,y: 0.25*(-mesh.p[1,mesh.t[0,:]][:,None]*(1-y)+\
                                            mesh.p[1,mesh.t[1,:]][:,None]*(1-y)+\
                                            mesh.p[1,mesh.t[2,:]][:,None]*(1+y)-\
                                            mesh.p[1,mesh.t[3,:]][:,None]*(1+y))
            self.J[1][1]=lambda x,y: 0.25*(-mesh.p[1,mesh.t[0,:]][:,None]*(1-x)-\
                                            mesh.p[1,mesh.t[1,:]][:,None]*(1+x)+\
                                            mesh.p[1,mesh.t[2,:]][:,None]*(1+x)+\
                                            mesh.p[1,mesh.t[3,:]][:,None]*(1-x))
        else:
            raise NotImplementedError("MappingQ1: wrong type of mesh was given to constructor!")

    def quadbasis(self,x,y,i):
        return {
            0:lambda x,y: 0.25*(1-x)*(1-y),
            1:lambda x,y: 0.25*(1+x)*(1-y),
            2:lambda x,y: 0.25*(1+x)*(1+y),
            3:lambda x,y: 0.25*(1-x)*(1+y)
            }[i](x,y)
   
    def F(self,X,tind=None):
        """Mapping defined by Q1 basis."""
        out={}
        out[0]=np.outer(self.p[0,self.t[0,:]],self.quadbasis(X[0,:],X[1,:],0))+\
               np.outer(self.p[0,self.t[1,:]],self.quadbasis(X[0,:],X[1,:],1))+\
               np.outer(self.p[0,self.t[2,:]],self.quadbasis(X[0,:],X[1,:],2))+\
               np.outer(self.p[0,self.t[3,:]],self.quadbasis(X[0,:],X[1,:],3))
        out[1]=np.outer(self.p[1,self.t[0,:]],self.quadbasis(X[0,:],X[1,:],0))+\
               np.outer(self.p[1,self.t[1,:]],self.quadbasis(X[0,:],X[1,:],1))+\
               np.outer(self.p[1,self.t[2,:]],self.quadbasis(X[0,:],X[1,:],2))+\
               np.outer(self.p[1,self.t[3,:]],self.quadbasis(X[0,:],X[1,:],3))

        return out
        
    def detDF(self,X,tind=None):
        if isinstance(X,dict):
            detDF=self.J[0][0](X[0],X[1])*self.J[1][1](X[0],X[1])-\
                  self.J[0][1](X[0],X[1])*self.J[1][0](X[0],X[1]) 
        else:
            detDF=self.J[0][0](X[0,:],X[1,:])*self.J[1][1](X[0,:],X[1,:])-\
                  self.J[0][1](X[0,:],X[1,:])*self.J[1][0](X[0,:],X[1,:])          
        if tind is not None:
            return detDF[tind,:]
        else:
            return detDF
            
    def invDF(self,X,tind=None):
        invJ={0:{},1:{}}
        if isinstance(X,dict):
            x=X[0]
            y=X[1]
        else:
            x=X[0,:]
            y=X[1,:]
        
        if tind is None:        
            detDF=self.detDF(X)
            invJ[0][0]=(self.J[1][1](x,y)/detDF)
            invJ[0][1]=(-self.J[0][1](x,y)/detDF)
            invJ[1][0]=(-self.J[1][0](x,y)/detDF)
            invJ[1][1]=(self.J[0][0](x,y)/detDF)
        else:
            detDF=self.detDF(X)[tind,:]
            invJ[0][0]=(self.J[1][1](x,y)[tind,:]/detDF)
            invJ[0][1]=(-self.J[0][1](x,y)[tind,:]/detDF)
            invJ[1][0]=(-self.J[1][0](x,y)[tind,:]/detDF)
            invJ[1][1]=(self.J[0][0](x,y)[tind,:]/detDF)
        
        return invJ

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
    
            # Matrices and vectors for boundary mappings: G(X)=BX+c
            self.B={0:{},1:{},2:{}}
    
            self.B[0][0]=mesh.p[0,mesh.facets[1,:]]-mesh.p[0,mesh.facets[0,:]]
            self.B[0][1]=mesh.p[0,mesh.facets[2,:]]-mesh.p[0,mesh.facets[0,:]]
            self.B[1][0]=mesh.p[1,mesh.facets[1,:]]-mesh.p[1,mesh.facets[0,:]]
            self.B[1][1]=mesh.p[1,mesh.facets[2,:]]-mesh.p[1,mesh.facets[0,:]]
            self.B[2][0]=mesh.p[2,mesh.facets[1,:]]-mesh.p[2,mesh.facets[0,:]]
            self.B[2][1]=mesh.p[2,mesh.facets[2,:]]-mesh.p[2,mesh.facets[0,:]]
    
            self.c={}
    
            self.c[0]=mesh.p[0,mesh.facets[0,:]]
            self.c[1]=mesh.p[1,mesh.facets[0,:]]
            self.c[2]=mesh.p[2,mesh.facets[0,:]]
    
            crossp={}
            crossp[0]= self.B[1][0]*self.B[2][1]-self.B[2][0]*self.B[1][1]
            crossp[1]=-self.B[0][0]*self.B[2][1]+self.B[2][0]*self.B[0][1]
            crossp[2]= self.B[0][0]*self.B[1][1]-self.B[1][0]*self.B[0][1]
    
            self.detB=np.sqrt(crossp[0]**2+crossp[1]**2+crossp[2]**2)
            
        else:
            raise TypeError("MappingAffine initialized with an incompatible mesh type!")

    def F(self,X,tind=None):
        """Affine map F(X)=AX+b."""
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
        else:
             raise NotImplementedError("MappingAffine.F: given dimension not implemented yet!")
        return y

    def invF(self,x,tind=None):
        """Inverse map F^{-1}(x)=A^{-1}(x-b)."""
        Y={}
        y={}
        if self.dim==2:
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
        elif self.dim==3:
            if tind is None:
                Y[0]=x[0].T-self.b[0]
                Y[1]=x[1].T-self.b[1]
                Y[2]=x[2].T-self.b[2]
                y[0]=self.invA[0][0]*Y[0]+self.invA[0][1]*Y[1]+self.invA[0][2]*Y[2]
                y[1]=self.invA[1][0]*Y[0]+self.invA[1][1]*Y[1]+self.invA[1][2]*Y[2]
                y[2]=self.invA[2][0]*Y[0]+self.invA[2][1]*Y[1]+self.invA[2][2]*Y[2]
            else:
                Y[0]=x[0].T-self.b[0][tind]
                Y[1]=x[1].T-self.b[1][tind]
                Y[2]=x[2].T-self.b[2][tind]
                y[0]=self.invA[0][0][tind]*Y[0]+self.invA[0][1][tind]*Y[1]+self.invA[0][2][tind]*Y[2]
                y[1]=self.invA[1][0][tind]*Y[0]+self.invA[1][1][tind]*Y[1]+self.invA[1][2][tind]*Y[2]
                y[2]=self.invA[2][0][tind]*Y[0]+self.invA[2][1][tind]*Y[1]+self.invA[2][2][tind]*Y[2]
            y[0]=y[0].T
            y[1]=y[1].T
            y[2]=y[2].T
        else:
             raise NotImplementedError("MappingAffine.F: given dimension not implemented yet!")
        return y

    def G(self,X,find=None):
        """Boundary mapping G(X)=Bx+c."""
        y={}
        if self.dim==2:
            if find is None:
                y[0]=np.outer(self.B[0],X).T+self.c[0]
                y[1]=np.outer(self.B[1],X).T+self.c[1]
            else:
                y[0]=np.outer(self.B[0][find],X).T+self.c[0][find]
                y[1]=np.outer(self.B[1][find],X).T+self.c[1][find]
            y[0]=y[0].T
            y[1]=y[1].T
        elif self.dim==3:
            if find is None:
                y[0]=np.outer(self.B[0][0],X[0,:]).T+np.outer(self.B[0][1],X[1,:]).T+self.c[0]
                y[1]=np.outer(self.B[1][0],X[0,:]).T+np.outer(self.B[1][1],X[1,:]).T+self.c[1]
                y[2]=np.outer(self.B[2][0],X[0,:]).T+np.outer(self.B[2][1],X[1,:]).T+self.c[2]
            else:
                y[0]=np.outer(self.B[0][0][find],X[0,:]).T+np.outer(self.B[0][1][find],X[1,:]).T+self.c[0][find]
                y[1]=np.outer(self.B[1][0][find],X[0,:]).T+np.outer(self.B[1][1][find],X[1,:]).T+self.c[1][find]
                y[2]=np.outer(self.B[2][0][find],X[0,:]).T+np.outer(self.B[2][1][find],X[1,:]).T+self.c[2][find]
            y[0]=y[0].T
            y[1]=y[1].T
            y[2]=y[2].T
        else:
            raise NotImplementedError("MappingAffine.G: given dimension not implemented yet!")
        return y
        
    def detDF(self,X,tind=None):
        if tind is None:
            detDF=self.detA
        else:
            detDF=self.detA[tind]
        return np.tile(detDF,(X.shape[1],1)).T  
        
    def detDG(self,X,find=None):
        if find is None:
            detDG=self.detB
        else:
            detDG=self.detB[find]
        return np.tile(detDG,(X.shape[1],1)).T
        
    def invDF(self,X,tind=None):
        invA=copy.deepcopy(self.invA)
        
        if isinstance(X,dict):
            N=X[0].shape[1]
        else:
            N=X.shape[1]
        
        if self.dim==2:
            if tind is None: # TODO did not test
                invA[0][0]=np.tile(invA[0][0],(N,1)).T
                invA[0][1]=np.tile(invA[0][1],(N,1)).T
                invA[1][0]=np.tile(invA[1][0],(N,1)).T
                invA[1][1]=np.tile(invA[1][1],(N,1)).T
            else:
                invA[0][0]=np.tile(invA[0][0][tind],(N,1)).T
                invA[0][1]=np.tile(invA[0][1][tind],(N,1)).T
                invA[1][0]=np.tile(invA[1][0][tind],(N,1)).T
                invA[1][1]=np.tile(invA[1][1][tind],(N,1)).T
        if self.dim==3:
            if tind is None: # TODO did not test
                invA[0][0]=np.tile(invA[0][0],(N,1)).T
                invA[0][1]=np.tile(invA[0][1],(N,1)).T
                invA[0][2]=np.tile(invA[0][2],(N,1)).T
                invA[1][0]=np.tile(invA[1][0],(N,1)).T
                invA[1][1]=np.tile(invA[1][1],(N,1)).T
                invA[1][2]=np.tile(invA[1][2],(N,1)).T
                invA[2][0]=np.tile(invA[2][0],(N,1)).T
                invA[2][1]=np.tile(invA[2][1],(N,1)).T
                invA[2][2]=np.tile(invA[2][2],(N,1)).T
            else:
                invA[0][0]=np.tile(invA[0][0][tind],(N,1)).T
                invA[0][1]=np.tile(invA[0][1][tind],(N,1)).T
                invA[0][2]=np.tile(invA[0][2][tind],(N,1)).T
                invA[1][0]=np.tile(invA[1][0][tind],(N,1)).T
                invA[1][1]=np.tile(invA[1][1][tind],(N,1)).T
                invA[1][2]=np.tile(invA[1][2][tind],(N,1)).T
                invA[2][0]=np.tile(invA[2][0][tind],(N,1)).T
                invA[2][1]=np.tile(invA[2][1][tind],(N,1)).T
                invA[2][2]=np.tile(invA[2][2][tind],(N,1)).T           
                
        return invA


        

