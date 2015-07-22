import numpy as np
import fem.mesh

class Mapping:
    """
    Abstract superclass for mappings.

    Mappings eat Meshes (possibly Geometries in isoparametric case?)
    and allow local-to-global and global-to-local mappings.
    """

    def __init__(self,mesh):
        raise NotImplementedError("Mapping constructor not implemented!")

    def F(self,X):
        """
        Element local to global.
        """
        raise NotImplementedError("Mapping.F() not implemented!")

    def invF(self,x):
        raise NotImplementedError("Mapping.invF() not implemented!")

    def DF(self,X):
        raise NotImplementedError("Mapping.DF() not implemented!")

    def detDF(self,X):
        raise NotImplementedError("Mapping.detDF() not implemented!")

    def G(self,X):
        """
        Boundary local to global.
        """
        raise NotImplementedError("Mapping.G() not implemented!")

class MappingAffineTri(Mapping):
    """
    Affine mappings for triangular mesh.
    """
    def __init__(self,mesh):
        if not isinstance(mesh,fem.mesh.MeshTri):
            raise TypeError("MappingAffineTri initialized with an incompatible mesh type!")

        # Matrices and vectors for triangle mappings: F(X)=AX+b
        self.A={0:{},1:{}}

        self.A[0][0]=mesh.p[0,mesh.t[1,:]]-mesh.p[0,mesh.t[0,:]]
        self.A[0][1]=mesh.p[0,mesh.t[2,:]]-mesh.p[0,mesh.t[0,:]]
        self.A[1][0]=mesh.p[1,mesh.t[1,:]]-mesh.p[1,mesh.t[0,:]]
        self.A[1][1]=mesh.p[1,mesh.t[2,:]]-mesh.p[1,mesh.t[0,:]]

        self.b={}

        self.b[0]=mesh.p[0,mesh.t[:,0]]
        self.b[1]=mesh.p[0,mesh.t[:,0]]

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

    def F(self,X):
        """
        Affine map F(X)=AX+b.

        Currently acts on Ndim x Npoints matrices
        and returns Dict D with D[0] corresponding to
        x-coordinates and D[1] corresponding to y-coordinates.
        Both D[0] and D[1] are Nelems x Npoints.
        """
        y={}
        y[0]=np.outer(self.A[0][0],X[0,:])+np.outer(self.A[0][1],X[1,:])+self.b[0]
        y[1]=np.outer(self.A[1][0],X[0,:])+np.outer(self.A[1][1],X[1,:])+self.b[1]
        return y

    def invF(self,x):
        """
        Inverse map F^{-1}(x)=A^{-1}(x-b).
        """
        Y={}
        Y[0]=x[0]-self.b[0]
        Y[1]=x[1]-self.b[1]
        y={}
        y[0]=self.invA[0][0]*Y[0].T+self.invA[0][1]*Y[1].T
        y[1]=self.invA[1][0]*Y[0].T+self.invA[1][1]*Y[1].T
        y[0]=y[0].T
        y[1]=y[1].T
        return y

    def G(self,X):
        """
        Boundary mapping G(X)=Bx+c.
        """
        y={}
        y[0]=np.outer(self.B[0],X)+self.c[0]
        y[1]=np.outer(self.B[1],X)+self.c[1]
        return y


        

