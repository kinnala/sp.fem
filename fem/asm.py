import numpy as np
import fem.mesh
import fem.mapping
from fem.quadrature import get_quadrature
from scipy.sparse import coo_matrix

import matplotlib.pyplot as plt
import time

class Assembler:
    """Superclass for assemblers."""
    def __init__(self):
        raise NotImplementedError("Assembler: constructor not implemented!")

class AssemblerTriP1(Assembler):
    """A fast (bi)linear form assembler with triangular P1 Lagrange elements."""
    def __init__(self,mesh):
        self.mapping=fem.mapping.MappingAffineTri(mesh)
        self.A=self.mapping.A
        self.b=self.mapping.b
        self.detA=self.mapping.detA
        self.invA=self.mapping.invA
        self.detB=self.mapping.detB
        self.mesh=mesh

    def iasm(self,form,intorder=2,w=None):
        """Interior assembly."""
        nv=self.mesh.p.shape[1]
        nt=self.mesh.t.shape[1]
        # TODO add support for assembling on a subset
        
        X,W=get_quadrature("tri",intorder)

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

        # global quadrature points
        x=self.mapping.F(X)

        # mesh parameters
        h=np.tile(np.array([np.sqrt(np.abs(self.detA))]).T,(1,W.shape[0]))

        # interpolation of a previous solution vector
        if w is not None:
            w1=np.outer(w[self.mesh.t[0,:]],phi[0])+\
               np.outer(w[self.mesh.t[1,:]],phi[1])+\
               np.outer(w[self.mesh.t[2,:]],phi[2])
            dw1={}
            dw1[0]=np.outer(w[self.mesh.t[0,:]],gradphi[0][0,:])+\
                   np.outer(w[self.mesh.t[1,:]],gradphi[1][0,:])+\
                   np.outer(w[self.mesh.t[2,:]],gradphi[2][0,:])
            dw1[1]=np.outer(w[self.mesh.t[0,:]],gradphi[0][1,:])+\
                   np.outer(w[self.mesh.t[1,:]],gradphi[1][1,:])+\
                   np.outer(w[self.mesh.t[2,:]],gradphi[2][1,:])
        
        # bilinear form
        if form.__code__.co_argcount==8:
            # initialize sparse matrix structures
            data=np.zeros(9*nt)
            rows=np.zeros(9*nt)
            cols=np.zeros(9*nt)
        
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
                    data[ixs]=np.dot(form(u,v,du,dv,x,h,w1,dw1),W)*np.abs(self.detA)
                    rows[ixs]=self.mesh.t[i,:]
                    cols[ixs]=self.mesh.t[j,:]
        
            return coo_matrix((data,(rows,cols)),shape=(nv,nv)).tocsr()

        # linear form
        elif form.__code__.co_argcount==6:
            # initialize sparse matrix structures
            data=np.zeros(3*nt)
            rows=np.zeros(3*nt)
            cols=np.zeros(3*nt)
            
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
                data[ixs]=np.dot(form(v,dv,x,h,w1,dw1),W)*np.abs(self.detA)
                rows[ixs]=self.mesh.t[i,:]
                cols[ixs]=np.zeros(nt)
        
            return coo_matrix((data,(rows,cols)),shape=(nv,1)).toarray().T[0]
        else:
            raise NotImplementedError("AssemblerTriP1.iasm not implemented for the given number of form arguments!")

    def fasm(self,form,find=None,intorder=2):
        """Facet assembly on all exterior facets.
        
        Keyword arguments:
        find - include to assemble on some other set of facets
        intorder - change integration order (default 2)
        """
        if find is None:
            find=np.nonzero(self.mesh.f2t[1,:]==-1)[0]
        nv=self.mesh.p.shape[1]
        nt=self.mesh.t.shape[1]
        ne=find.shape[0]

        X,W=get_quadrature("line",intorder)
        
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
        
        # boundary triangle indices
        tind=self.mesh.f2t[0,find]
        h=np.tile(np.sqrt(np.abs(self.detB[tind,None])),(1,W.shape[0]))

        # mappings
        x=self.mapping.G(X,find=find) # reference face to global face
        Y=self.mapping.invF(x,tind=tind) # global triangle to reference triangle
        
        # tangent vectors
        t={}
        t[0]=self.mesh.p[0,self.mesh.facets[0,find]]-self.mesh.p[0,self.mesh.facets[1,find]]
        t[1]=self.mesh.p[1,self.mesh.facets[0,find]]-self.mesh.p[1,self.mesh.facets[1,find]]
        
        # normalize tangent vectors
        t[2]=np.sqrt(t[0]**2+t[1]**2)
        t[0]/=t[2]
        t[1]/=t[2]
        
        # normal vectors
        n={}
        n[0]=-t[1]
        n[1]=t[0]

        # map normal vectors to reference coords to correct sign (outward normals wanted)
        n_ref={}
        n_ref[0]=self.invA[0][0][tind]*n[0]+self.invA[1][0][tind]*n[1]
        n_ref[1]=self.invA[0][1][tind]*n[0]+self.invA[1][1][tind]*n[1]
        
        # change the sign of the following normal vectors
        meps=1e-14
        csgn=np.nonzero((n_ref[0]<0)*(n_ref[1]<0)+\
                        (n_ref[0]>0)*(n_ref[1]<meps)*(n_ref[1]>-meps)+\
                        (n_ref[0]<meps)*(n_ref[0]>-meps)*(n_ref[1]>0))[0]
        n[0][csgn]=(-1.)*(n[0][csgn])
        n[1][csgn]=(-1.)*(n[1][csgn])
        
        n[0]=np.tile(n[0][:,None],(1,W.shape[0]))
        n[1]=np.tile(n[1][:,None],(1,W.shape[0]))

        # bilinear form
        if form.__code__.co_argcount==7:
            # initialize sparse matrix structures
            data=np.zeros(9*ne)
            rows=np.zeros(9*ne)
            cols=np.zeros(9*ne)

            # TODO interpolation

            for j in [0,1,2]:
                u=phi[j](Y[0],Y[1])
                du={}
                du[0]=self.invA[0][0][tind,None]*gradphi_x[j](Y[0],Y[1])+\
                      self.invA[1][0][tind,None]*gradphi_y[j](Y[0],Y[1])
                du[1]=self.invA[0][1][tind,None]*gradphi_x[j](Y[0],Y[1])+\
                      self.invA[1][1][tind,None]*gradphi_y[j](Y[0],Y[1])
                for i in [0,1,2]:
                    v=phi[i](Y[0],Y[1])
                    dv={}
                    dv[0]=self.invA[0][0][tind,None]*gradphi_x[i](Y[0],Y[1])+\
                          self.invA[1][0][tind,None]*gradphi_y[i](Y[0],Y[1])
                    dv[1]=self.invA[0][1][tind,None]*gradphi_x[i](Y[0],Y[1])+\
                          self.invA[1][1][tind,None]*gradphi_y[i](Y[0],Y[1])
           
                    # find correct location in data,rows,cols
                    ixs=slice(ne*(3*j+i),ne*(3*j+i+1))
                    
                    # compute entries of local stiffness matrices
                    data[ixs]=np.dot(form(u,v,du,dv,x,h,n),W)*np.abs(self.detB[find])
                    rows[ixs]=self.mesh.t[i,tind]
                    cols[ixs]=self.mesh.t[j,tind]
        
            return coo_matrix((data,(rows,cols)),shape=(nv,nv)).tocsr()
        # linear form
        elif form.__code__.co_argcount==5:
            # initialize sparse matrix structures
            data=np.zeros(3*ne)
            rows=np.zeros(3*ne)
            cols=np.zeros(3*ne)

            # TODO interpolation

            for i in [0,1,2]:
                v=phi[i](Y[0],Y[1])
                dv={}
                dv[0]=self.invA[0][0][tind,None]*gradphi_x[i](Y[0],Y[1])+\
                      self.invA[1][0][tind,None]*gradphi_y[i](Y[0],Y[1])
                dv[1]=self.invA[0][1][tind,None]*gradphi_x[i](Y[0],Y[1])+\
                      self.invA[1][1][tind,None]*gradphi_y[i](Y[0],Y[1])
        
                # find correct location in data,rows,cols
                ixs=slice(ne*i,ne*(i+1))
                
                # compute entries of local stiffness matrices
                data[ixs]=np.dot(form(v,dv,x,h,n),W)*np.abs(self.detB[find])
                rows[ixs]=self.mesh.t[i,tind]
                cols[ixs]=np.zeros(ne)
        
            return coo_matrix((data,(rows,cols)),shape=(nv,1)).toarray().T[0]
        else:
            raise NotImplementedError("AssemblerTriP1.fasm not implemented for the given number of form arguments!")


