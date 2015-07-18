import numpy as np
import fem.mesh
import fem.mapping
from scipy.sparse import coo_matrix

class Assembler:
    """
    Superclass for assemblers.
    """
    def __init__(self):
        raise NotImplementedError("Assembler constructor not implemented!")

class AssemblerTriP1(Assembler):
    """
    A fast (bi)linear form assembler with triangular P1 Lagrange elements.
    """
    def __init__(self,mesh):
        self.mapping=fem.mapping.MappingAffineTri(mesh)
        self.A=self.mapping.A
        self.b=self.mapping.b
        self.detA=self.mapping.detA
        self.invA=self.mapping.invA
        self.mesh=mesh

    def iasm(self,form):
        """
        Interior assembly.
        """
        nv=self.mesh.p.shape[1]
        nt=self.mesh.t.shape[1]
        
        # quadrature points and weights (2nd order accurate)
        # TODO use quadrature interface
        X=np.array([[1.666666666666666666666e-01,6.666666666666666666666e-01,1.666666666666666666666e-01],[1.666666666666666666666e-01,1.666666666666666666666e-01,6.666666666666666666666e-01]])
        W=np.array([1.666666666666666666666e-01,1.666666666666666666666e-01,1.666666666666666666666e-01])

        # local basis functions
        phi={}
        phi[0]=1.-X[0,:]-X[1,:]
        phi[1]=X[0,:]
        phi[2]=X[1,:]

        # local basis function gradients
        gradphi={}
        gradphi[0]=np.tile(np.array([-1.,-1.]),(X.shape[1],1)).T
        gradphi[1]=np.tile(np.array([1.,0.]),(X.shape[1],1)).T
        gradphi[2]=np.tile(np.array([0.,1.]),(X.shape[1],1)).T    
        # TODO investigate turning these into two 1d arrays; could be faster?
        
        # bilinear form
        if form.__code__.co_argcount==5:
            # initialize sparse matrix structures
            data=np.zeros(9*nt)
            rows=np.zeros(9*nt)
            cols=np.zeros(9*nt)
        
            x=self.mapping.F(X)
            # TODO interpolation
        
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
                    data[ixs]=np.dot(form(u,v,du,dv,x),W)*np.abs(self.detA)
                    rows[ixs]=self.mesh.t[i,:]
                    cols[ixs]=self.mesh.t[j,:]
        
            return coo_matrix((data,(rows,cols)),shape=(nv,nv)).tocsr()

        # linear form
        elif form.__code__.co_argcount==3:
            # initialize sparse matrix structures
            data=np.zeros(3*nt)
            rows=np.zeros(3*nt)
            cols=np.zeros(3*nt)
            
            # TODO interpolation
            x=self.mapping.F(X)
        
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
                data[ixs]=np.dot(form(v,dv,x),W)*np.abs(self.detA)
                rows[ixs]=self.mesh.t[i,:]
                cols[ixs]=np.zeros(nt)
        
            return coo_matrix((data,(rows,cols)),shape=(nv,1)).toarray().T[0]
        else:
            raise NotImplementedError("AssemblerTriP1 iasm not implemented for the given number of form arguments!")

    def fasm(self,form):
        """
        Facet assembly.
        """
        nv=self.mesh.p.shape[1]
        nt=self.mesh.t.shape[1]
        ne=self.mesh.facets.shape[1]

        X=np.array([1.127016653792584e-1,5.0000000000000000e-1,8.872983346207417e-1])
        W=np.array([2.777777777777779e-1,4.4444444444444444e-1,2.777777777777778e-1])
        
        #   # local basis functions
        #   phi={}
        #   phi[0]=1.-X
        #   phi[1]=X

        #   # local basis function gradients
        #   gradphi={}
        #   gradphi[0]=np.ones(X.shape[0])*(-1.)
        #   gradphi[1]=np.ones(X.shape[0])
        # local basis
        phi={}
        phi[0]=lambda x,y: 1-x-y
        phi[1]=lambda x,y: x
        phi[2]=lambda x,y: y

        #gradphi={}
        #TODO gradient support


        # bilinear form
        if form.__code__.co_argcount==5:
            # initialize sparse matrix structures
            data=np.zeros(9*ne)
            rows=np.zeros(9*ne)
            cols=np.zeros(9*ne)

            # mappings
            x=self.mapping.G(X) # reference face to global face
            Y=self.mapping.invF(x) # global triangle to reference triangle

            # TODO interpolation

            for j in [0,1,2]:
                u=phi[j](Y[0],Y[1])
                #du={}
                #du[0]=np.outer(self.invA[0][0],gradphi[j][0,:])+\
                #      np.outer(self.invA[1][0],gradphi[j][1,:])
                #du[1]=np.outer(self.invA[0][1],gradphi[j][0,:])+\
                #      np.outer(self.invA[1][1],gradphi[j][1,:])
                for i in [0,1,2]:
                    v=phi[i](Y[0],Y[1])
                    #   dv={}
                    #   dv[0]=np.outer(self.invA[0][0],gradphi[i][0,:])+\
                    #         np.outer(self.invA[1][0],gradphi[i][1,:])
                    #   dv[1]=np.outer(self.invA[0][1],gradphi[i][0,:])+\
                    #         np.outer(self.invA[1][1],gradphi[i][1,:])
            
                    # find correct location in data,rows,cols
                    ixs=slice(nt*(3*j+i),nt*(3*j+i+1))
                    
                    # compute entries of local stiffness matrices
                    data[ixs]=np.dot(form(u,v,du,dv,x),W)*np.abs(self.detA)
                    rows[ixs]=self.mesh.t[i,:]
                    cols[ixs]=self.mesh.t[j,:]
        
            return coo_matrix((data,(rows,cols)),shape=(nv,nv)).tocsr()


