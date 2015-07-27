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
        self.detB=self.mapping.detB
        self.mesh=mesh

    def iasm(self,form):
        """
        Interior assembly.
        """
        nv=self.mesh.p.shape[1]
        nt=self.mesh.t.shape[1]
        # TODO add support for assembling on a subset
        
        # quadrature points and weights (2nd order accurate)
        # TODO use quadrature interface
        X=np.array([[1.666666666666666666666e-01,6.666666666666666666666e-01,1.666666666666666666666e-01],
                    [1.666666666666666666666e-01,1.666666666666666666666e-01,6.666666666666666666666e-01]])
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
        x=self.mapping.F(X)
        
        # bilinear form
        if form.__code__.co_argcount==5:
            # initialize sparse matrix structures
            data=np.zeros(9*nt)
            rows=np.zeros(9*nt)
            cols=np.zeros(9*nt)
        
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
            raise NotImplementedError("AssemblerTriP1.iasm not implemented for the given number of form arguments!")

    def fasm(self,form,find=None):
        """
        Facet assembly on all exterior facets.
        
        Include 'find' (facet indices) parameter
        to assemble on some other set of facets.
        """
        if find is None:
            find=np.nonzero(self.mesh.f2t[1,:]==-1)[0]
        nv=self.mesh.p.shape[1]
        nt=self.mesh.t.shape[1]
        ne=find.shape[0]

        X=np.array([1.127016653792584e-1,5.0000000000000000e-1,8.872983346207417e-1])
        W=np.array([2.777777777777779e-1,4.4444444444444444e-1,2.777777777777778e-1])
        
        # local basis
        phi={}
        phi[0]=lambda x,y: 1-x-y
        phi[1]=lambda x,y: x
        phi[2]=lambda x,y: y

        gradphi_x={}
        gradphi_x[0]=lambda x,y: -1.+0*x
        gradphi_x[1]=lambda x,y: 1.+0*x
        gradphi_x[2]=lambda x,y: 0*x

        gradphi_y={}
        gradphi_y[0]=lambda x,y: -1.+0*x
        gradphi_y[1]=lambda x,y: 0+0*x
        gradphi_y[2]=lambda x,y: 1.+0*x

        # bilinear form
        if form.__code__.co_argcount==5:
            # initialize sparse matrix structures
            data=np.zeros(9*ne)
            rows=np.zeros(9*ne)
            cols=np.zeros(9*ne)

            # mappings
            tind=self.mesh.f2t[0,find]
            x=self.mapping.G(X,find=find) # reference face to global face
            Y=self.mapping.invF(x,tind=tind) # global triangle to reference triangle

            # TODO interpolation

            for j in [0,1,2]:
                u=phi[j](Y[0],Y[1])
                du={}
                du[0]=self.invA[0][0][tind,None]*gradphi_x[j](Y[0],Y[1])+\
                      self.invA[1][0][tind,None]*gradphi_y[j](Y[0],Y[1])
                du[1]=self.invA[1][0][tind,None]*gradphi_x[j](Y[0],Y[1])+\
                      self.invA[1][1][tind,None]*gradphi_y[j](Y[0],Y[1])
                for i in [0,1,2]:
                    v=phi[i](Y[0],Y[1])
                    dv={}
                    dv[0]=self.invA[0][0][tind,None]*gradphi_x[i](Y[0],Y[1])+\
                          self.invA[1][0][tind,None]*gradphi_y[i](Y[0],Y[1])
                    dv[1]=self.invA[1][0][tind,None]*gradphi_x[i](Y[0],Y[1])+\
                          self.invA[1][1][tind,None]*gradphi_y[i](Y[0],Y[1])
           
                    # find correct location in data,rows,cols
                    ixs=slice(ne*(3*j+i),ne*(3*j+i+1))
                    
                    # compute entries of local stiffness matrices
                    data[ixs]=np.dot(form(u,v,du,dv,x),W)*np.abs(self.detB[find])
                    rows[ixs]=self.mesh.t[i,tind]
                    cols[ixs]=self.mesh.t[j,tind]
        
            return coo_matrix((data,(rows,cols)),shape=(nv,nv)).tocsr()
        # linear form
        elif form.__code__.co_argcount==3:
            # initialize sparse matrix structures
            data=np.zeros(3*ne)
            rows=np.zeros(3*ne)
            cols=np.zeros(3*ne)

            # mappings
            tind=self.mesh.f2t[0,find]
            x=self.mapping.G(X,find=find) # reference face to global face
            Y=self.mapping.invF(x,tind=tind) # global triangle to reference triangle

            # TODO interpolation

            for i in [0,1,2]:
                v=phi[i](Y[0],Y[1])
                dv={}
                dv[0]=self.invA[0][0][tind,None]*gradphi_x[i](Y[0],Y[1])+\
                      self.invA[1][0][tind,None]*gradphi_y[i](Y[0],Y[1])
                dv[1]=self.invA[1][0][tind,None]*gradphi_x[i](Y[0],Y[1])+\
                      self.invA[1][1][tind,None]*gradphi_y[i](Y[0],Y[1])
        
                # find correct location in data,rows,cols
                ixs=slice(ne*i,ne*(i+1))
                
                # compute entries of local stiffness matrices
                data[ixs]=np.dot(form(v,dv,x),W)*np.abs(self.detB[find])
                rows[ixs]=self.mesh.t[i,tind]
                cols[ixs]=np.zeros(ne)
        
            return coo_matrix((data,(rows,cols)),shape=(nv,1)).toarray().T[0]
        else:
            raise NotImplementedError("AssemblerTriP1.fasm not implemented for the given number of form arguments!")


