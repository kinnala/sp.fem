# -*- coding: utf-8 -*-
"""
Assembly of matrices related to linear and bilinear forms.

@author: Tom Gustafsson
"""
import numpy as np
import fem.mesh
import fem.mapping
import inspect
from fem.quadrature import get_quadrature
from scipy.sparse import coo_matrix

import matplotlib.pyplot as plt
import time

class Assembler:
    """Finite element assembler."""
    def __init__(self):
        raise NotImplementedError("Assembler: constructor not implemented!")

    def fillargs(self,oldform,newargs):
        """Used for filling functions with required set of arguments."""
        oldargs=inspect.getargspec(oldform).args
        if oldargs==newargs:
            # the given form already has correct arguments
            return oldform

        y=[]
        for oarg in oldargs:
            # add corresponding new argument index to y for
            # each old argument
            for ix,narg in enumerate(newargs):
                if oarg==narg:
                    y.append(ix)
                    break

        if len(oldargs)==1:
            def newform(*x):
                return oldform(x[y[0]])
        elif len(oldargs)==2:
            def newform(*x):
                return oldform(x[y[0]],x[y[1]])
        elif len(oldargs)==3:
            def newform(*x):
                return oldform(x[y[0]],x[y[1]],x[y[2]])
        elif len(oldargs)==4:
            def newform(*x):
                return oldform(x[y[0]],x[y[1]],x[y[2]],x[y[3]])
        elif len(oldargs)==5:
            def newform(*x):
                return oldform(x[y[0]],x[y[1]],x[y[2]],x[y[3]],x[y[4]])
        elif len(oldargs)==6:
            def newform(*x):
                return oldform(x[y[0]],x[y[1]],x[y[2]],x[y[3]],x[y[4]],x[y[5]])
        elif len(oldargs)==7:
            def newform(*x):
                return oldform(x[y[0]],x[y[1]],x[y[2]],x[y[3]],x[y[4]],x[y[5]],x[y[6]])
        elif len(oldargs)==8:
            def newform(*x):
                return oldform(x[y[0]],x[y[1]],x[y[2]],x[y[3]],x[y[4]],x[y[5]],x[y[6]],x[y[7]])
        else:
            raise NotImplementedError("Assembler.fillargs: the maximum number of arguments reached")

        return newform

class AssemblerElement(Assembler):
    """A quasi-fast assembler for arbitrary element/mesh/mapping."""
    def __init__(self,mesh,mapping,elem_u,elem_v=None):
        # TODO check consistency between (mesh,mapping,elem)
        if not isinstance(elem_u,fem.element.Element):
            raise Exception("AssemblerElement: elem_u must be an instace of Element!")

        self.mapping=mapping(mesh)
        self.mesh=mesh

        self.elem_u=elem_u
        self.dofnum_u=Dofnum(mesh,elem_u)

        if elem_v is None:
            self.elem_v=elem_u
            self.dofnum_v=self.dofnum_u
        else:
            self.elem_v=elem_v
            self.dofnum_v=Dofnum(mesh,elem_v)
     
    def iasm(self,form,intorder=None,tind=None):
        """Interior assembly."""
        nt=self.mesh.t.shape[1]
        if tind is None:
            # Assemble on all elements by default
            tind=range(nt)
        if intorder is None:
            intorder=self.elem_u.maxdeg+self.elem_v.maxdeg
        
        # check and fix parameters of form
        oldparams=inspect.getargspec(form).args
        if 'u' in oldparams or 'du' in oldparams:
            # TODO get these from element (maybe): L2 elems have only u,v etc.
            paramlist=['u','v','du','dv','ddu','ddv','x']
            #paramlist=['u','v','du','dv','x','h']
            bilinear=True
        else:
            paramlist=['v','dv','ddv','x']
            #paramlist=['v','dv','x','h']
            bilinear=False
        fform=self.fillargs(form,paramlist)
        
        # TODO add support for assembling on a subset
        
        X,W=get_quadrature(self.mesh.refdom,intorder)

        # global quadrature points
        x=self.mapping.F(X,tind)

        # jacobian at quadrature points
        detDF=self.mapping.detDF(X,tind)
        
        Nbfun_u=self.dofnum_u.t_dof.shape[0]
        Nbfun_v=self.dofnum_v.t_dof.shape[0]  

        #TODO think about precomputing [u,du,ddu] in inner loop MAKE OPTIONAL FLAG FOR OPTIMIZATION
        
        # bilinear form
        if bilinear:
            # initialize sparse matrix structures
            data=np.zeros(Nbfun_u*Nbfun_v*nt)
            rows=np.zeros(Nbfun_u*Nbfun_v*nt)
            cols=np.zeros(Nbfun_u*Nbfun_v*nt)
        
            for j in range(Nbfun_u):
                u,du,ddu=self.elem_u.gbasis(self.mapping,X,j,tind)
                for i in range(Nbfun_v):
                    v,dv,ddv=self.elem_v.gbasis(self.mapping,X,i,tind)
            
                    # find correct location in data,rows,cols
                    ixs=slice(nt*(Nbfun_v*j+i),nt*(Nbfun_v*j+i+1))
                    
                    # compute entries of local stiffness matrices
                    data[ixs]=np.dot(fform(u,v,du,dv,ddu,ddv,x)*np.abs(detDF),W)
                    rows[ixs]=self.dofnum_v.t_dof[i,:]
                    cols[ixs]=self.dofnum_u.t_dof[j,:]
        
            return coo_matrix((data,(rows,cols)),shape=(self.dofnum_v.N,self.dofnum_u.N)).tocsr()
            
        else:
            # initialize sparse matrix structures
            data=np.zeros(Nbfun_v*nt)
            rows=np.zeros(Nbfun_v*nt)
            cols=np.zeros(Nbfun_v*nt)
            
            for i in range(Nbfun_v):
                v,dv,ddv=self.elem_v.gbasis(self.mapping,X,i,tind)

                # find correct location in data,rows,cols
                ixs=slice(nt*i,nt*(i+1))
                
                # compute entries of local stiffness matrices
                data[ixs]=np.dot(fform(v,dv,ddv,x)*np.abs(detDF),W)
                rows[ixs]=self.dofnum_v.t_dof[i,:]
                cols[ixs]=np.zeros(nt)
        
            return coo_matrix((data,(rows,cols)),shape=(self.dofnum_v.N,1)).toarray().T[0]
            
    def fasm(self,form,find=None,intorder=None): # TODO fix and test
        """Facet assembly on all exterior facets."""
        if find is None:
            find=self.mesh.boundary_facets()
        if intorder is None:
            intorder=self.elem_u.maxdeg+self.elem_v.maxdeg            
            
        nv=self.mesh.p.shape[1]
        nt=self.mesh.t.shape[1]
        ne=find.shape[0]

        # check and fix parameters of form
        oldparams=inspect.getargspec(form).args
        if 'u' in oldparams or 'du' in oldparams:
            paramlist=['u','v','du','dv','x']
            #paramlist=['u','v','du','dv','x','h','n','w']
            bilinear=True
        else:
            paramlist=['v','dv','x']
            #paramlist=['v','dv','x','h','n','w']
            bilinear=False
        fform=self.fillargs(form,paramlist)

        X,W=get_quadrature(self.mesh.brefdom,intorder)
        
        # boundary element indices
        tind=self.mesh.f2t[0,find]
        #h=np.tile(np.sqrt(np.abs(self.detB[tind,None])),(1,W.shape[0]))

        # mappings
        x=self.mapping.G(X,find=find) # reference facet to global facet
        Y=self.mapping.invF(x,tind=tind) # global facet to reference element
        
        Nbfun_u=self.dofnum_u.t_dof.shape[0]
        Nbfun_v=self.dofnum_v.t_dof.shape[0] 

        detDG=self.mapping.detDG(X,find)        
        
#        # tangent vectors
#        t={}
#        t[0]=self.mesh.p[0,self.mesh.facets[0,find]]-self.mesh.p[0,self.mesh.facets[1,find]]
#        t[1]=self.mesh.p[1,self.mesh.facets[0,find]]-self.mesh.p[1,self.mesh.facets[1,find]]
#        
#        # normalize tangent vectors
#        tlen=np.sqrt(t[0]**2+t[1]**2)
#        t[0]/=tlen
#        t[1]/=tlen
#        
#        # normal vectors
#        n={}
#        n[0]=-t[1]
#        n[1]=t[0]
#
#        # map normal vectors to reference coords to correct sign (outward normals wanted)
#        n_ref={}
#        n_ref[0]=self.invA[0][0][tind]*n[0]+self.invA[1][0][tind]*n[1]
#        n_ref[1]=self.invA[0][1][tind]*n[0]+self.invA[1][1][tind]*n[1]
#        
#        # change the sign of the following normal vectors
#        meps=np.finfo(float).eps
#        csgn=np.nonzero((n_ref[0]<0)*(n_ref[1]<0)+\
#                        (n_ref[0]>0)*(n_ref[1]<meps)*(n_ref[1]>-meps)+\
#                        (n_ref[0]<meps)*(n_ref[0]>-meps)*(n_ref[1]>0))[0]
#        n[0][csgn]=(-1.)*(n[0][csgn])
#        n[1][csgn]=(-1.)*(n[1][csgn])
#        
#        n[0]=np.tile(n[0][:,None],(1,W.shape[0]))
#        n[1]=np.tile(n[1][:,None],(1,W.shape[0]))
#
#        if w is not None:
#            w1=phi[0](Y[0],Y[1])*w[self.mesh.t[0,tind][:,None]]+\
#               phi[1](Y[0],Y[1])*w[self.mesh.t[1,tind][:,None]]+\
#               phi[2](Y[0],Y[1])*w[self.mesh.t[2,tind][:,None]]
#        else:
#            w1=None

        # bilinear form
        if bilinear:
            # initialize sparse matrix structures
            data=np.zeros(Nbfun_u*Nbfun_v*ne)
            rows=np.zeros(Nbfun_u*Nbfun_v*ne)
            cols=np.zeros(Nbfun_u*Nbfun_v*ne)

            for j in range(Nbfun_u):
                u,du,_=self.elem_u.gbasis(self.mapping,Y,j,tind)
                for i in range(Nbfun_v):
                    v,dv,_=self.elem_v.gbasis(self.mapping,Y,i,tind)
           
                    # find correct location in data,rows,cols
                    ixs=slice(ne*(Nbfun_v*j+i),ne*(Nbfun_v*j+i+1))
                    
                    # compute entries of local stiffness matrices
                    data[ixs]=np.dot(fform(u,v,du,dv,x)*np.abs(detDG),W)
                    rows[ixs]=self.dofnum_v.t_dof[i,tind]
                    cols[ixs]=self.dofnum_u.t_dof[j,tind]
                    #rows[ixs]=self.mesh.t[i,tind]
                    #cols[ixs]=self.mesh.t[j,tind]
        
            return coo_matrix((data,(rows,cols)),shape=(self.dofnum_v.N,self.dofnum_u.N)).tocsr()
        # linear form
        else:
            # initialize sparse matrix structures
            data=np.zeros(Nbfun_v*ne)
            rows=np.zeros(Nbfun_v*ne)
            cols=np.zeros(Nbfun_v*ne)

            for i in range(Nbfun_v):
                v,dv,_=self.elem_v.gbasis(self.mapping,Y,i,tind)
        
                # find correct location in data,rows,cols
                ixs=slice(ne*i,ne*(i+1))
                
                # compute entries of local stiffness matrices
                data[ixs]=np.dot(fform(v,dv,x)*np.abs(detDG),W)
                rows[ixs]=self.dofnum_v.t_dof[i,tind]
                #rows[ixs]=self.mesh.t[i,tind]
                cols[ixs]=np.zeros(ne)
        
            return coo_matrix((data,(rows,cols)),shape=(self.dofnum_v.N,1)).toarray().T[0]
            
    def L2error(self,uh,exact,intorder=None):
        """Compute L2 error against exact solution."""
        if self.elem_u.maxdeg!=self.elem_v.maxdeg:
            raise NotImplementedError("AssemblyElement.L2error: elem_u.maxdeg must be elem_v.maxdeg when computing errors!")
        if intorder is None:
            intorder=2*self.elem_u.maxdeg
            
        X,W=get_quadrature(self.mesh.refdom,intorder)
            
        # assemble some helper matrices
        # the idea is to use the identity: (u-uh,u-uh)=(u,u)+(uh,uh)-2(u,uh)
        def uv(u,v):
            return u*v
    
        def fv(v,x):
            return exact(x)*v
            
        M=self.iasm(uv)
        f=self.iasm(fv)
        
        detDF=self.mapping.detDF(X)
        x=self.mapping.F(X)
        
        uu=np.sum(np.dot(exact(x)**2*np.abs(detDF),W))
        
        return np.sqrt(uu+np.dot(uh,M.dot(uh))-2.*np.dot(uh,f))
        
    def H1error(self,uh,dexact,intorder=None):
        """Compute H1 seminorm error against exact solution."""
        if self.elem_u.maxdeg!=self.elem_v.maxdeg:
            raise NotImplementedError("AssemblyElement.H1error: elem_u.maxdeg must be elem_v.maxdeg when computing errors!")
        if intorder is None:
            intorder=2*self.elem_u.maxdeg
            
        X,W=get_quadrature(self.mesh.refdom,intorder)
            
        # assemble some helper matrices
        # the idea is to use the identity: (u-uh,u-uh)=(u,u)+(uh,uh)-2(u,uh)
        def uv(du,dv):
            if len(du)==2:
                return du[0]*dv[0]+du[1]*dv[1]
            elif len(du)==3:
                return du[0]*dv[0]+du[1]*dv[1]+du[2]*dv[2]
            else:
                raise NotImplementedError("AssemblerElement.H1error: not implemented for current domain dimension!")
    
        def fv(dv,x):
            if len(x)==2:
                return dexact[0](x)*dv[0]+dexact[1](x)*dv[1]
            elif len(x)==3:
                return dexact[0](x)*dv[0]+dexact[1](x)*dv[1]+dexact[2](x)*dv[2]
            else:
                raise NotImplementedError("AssemblerElement.H1error: not implemented for current domain dimension!")
            
        M=self.iasm(uv)
        f=self.iasm(fv)
        
        detDF=self.mapping.detDF(X)
        x=self.mapping.F(X)
        
        if len(x)==2:
            uu=np.sum(np.dot((dexact[0](x)**2+dexact[1](x)**2)*np.abs(detDF),W))
        elif len(x)==3:
            uu=np.sum(np.dot((dexact[0](x)**2+dexact[1](x)**2+dexact[2](x)**2)*np.abs(detDF),W))
        else:
            raise NotImplementedError("AssemblerElement.H1error: not implemented for current domain dimension!")
        
        return np.sqrt(uu+np.dot(uh,M.dot(uh))-2.*np.dot(uh,f))

class Dofnum():
    """Generate a global degree-of-freedom numbering for arbitrary mesh."""
    def __init__(self,mesh,element):
        self.n_dof=np.reshape(np.arange(element.n_dofs*mesh.p.shape[1],dtype=np.int64),(element.n_dofs,mesh.p.shape[1]),order='F')
        offset=element.n_dofs*mesh.p.shape[1]
        if hasattr(mesh,'edges'): # 3d mesh
            self.e_dof=np.reshape(np.arange(element.e_dofs*mesh.edges.shape[1],dtype=np.int64),(element.e_dofs,mesh.edges.shape[1]),order='F')+offset
            offset=offset+element.e_dofs*mesh.edges.shape[1]
        self.f_dof=np.reshape(np.arange(element.f_dofs*mesh.facets.shape[1],dtype=np.int64),(element.f_dofs,mesh.facets.shape[1]),order='F')+offset
        offset=offset+element.f_dofs*mesh.facets.shape[1]
        self.i_dof=np.reshape(np.arange(element.i_dofs*mesh.t.shape[1],dtype=np.int64),(element.i_dofs,mesh.t.shape[1]),order='F')+offset
        
        # global numbering
        self.t_dof=np.zeros((0,mesh.t.shape[1]),dtype=np.int64)
        
        # nodal dofs
        for itr in range(mesh.t.shape[0]):
            self.t_dof=np.vstack((self.t_dof,self.n_dof[:,mesh.t[itr,:]]))
        
        # edge dofs (if 3D)
        if hasattr(mesh,'edges'):
            for itr in range(mesh.t2e.shape[0]):
                self.t_dof=np.vstack((self.t_dof,self.e_dof[:,mesh.t2e[itr,:]]))

        # facet dofs (TODO if 2D or 3D)        
        for itr in range(mesh.t2f.shape[0]):
            self.t_dof=np.vstack((self.t_dof,self.f_dof[:,mesh.t2f[itr,:]]))
        
        self.t_dof=np.vstack((self.t_dof,self.i_dof))
        
        self.N=np.max(self.t_dof)+1
        
    def getdofs(self,N=None,F=None,T=None):
        """Return global DOF numbers corresponding to each node(N), facet(F) and triangle(T)"""
        dofs=np.zeros(0,dtype=np.int64)        
        if N is not None:
            dofs=np.hstack((dofs,self.n_dof[:,N].flatten()))
        if F is not None:
            dofs=np.hstack((dofs,self.f_dof[:,F].flatten()))
        if T is not None:
            dofs=np.hstack((dofs,self.i_dof[:,T].flatten()))
        return dofs.flatten()

class AssemblerTriP1(Assembler):
    """A fast (bi)linear form assembler with triangular P1 Lagrange elements."""
    def __init__(self,mesh):
        self.mapping=fem.mapping.MappingAffine(mesh)
        self.A=self.mapping.A
        self.b=self.mapping.b
        self.detA=self.mapping.detA
        self.invA=self.mapping.invA
        self.detB=self.mapping.detB
        self.mesh=mesh

    def iasm(self,form,intorder=2,w1=None,w2=None):
        """Interior assembly."""
        nv=self.mesh.p.shape[1]
        nt=self.mesh.t.shape[1]

        # check and fix parameters of form
        oldparams=inspect.getargspec(form).args
        if 'u' in oldparams or 'du' in oldparams:
            paramlist=['u','v','du','dv','x','h','w1','dw1','w2','dw2']
            bilinear=True
        else:
            paramlist=['v','dv','x','h','w1','dw1','w2','dw2']
            bilinear=False
        fform=self.fillargs(form,paramlist)

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
        if w1 is not None:
            W1=np.outer(w1[self.mesh.t[0,:]],phi[0])+\
               np.outer(w1[self.mesh.t[1,:]],phi[1])+\
               np.outer(w1[self.mesh.t[2,:]],phi[2])
            dW1={}
            dW1[0]=(np.outer(self.invA[0][0],gradphi[0][0,:])+\
                    np.outer(self.invA[1][0],gradphi[0][1,:]))*\
                    w1[self.mesh.t[0,:]][:,None]+\
                   (np.outer(self.invA[0][0],gradphi[1][0,:])+\
                    np.outer(self.invA[1][0],gradphi[1][1,:]))*\
                    w1[self.mesh.t[1,:]][:,None]+\
                   (np.outer(self.invA[0][0],gradphi[2][0,:])+\
                    np.outer(self.invA[1][0],gradphi[2][1,:]))*\
                    w1[self.mesh.t[2,:]][:,None]
            dW1[1]=(np.outer(self.invA[0][1],gradphi[0][0,:])+\
                    np.outer(self.invA[1][1],gradphi[0][1,:]))*\
                    w1[self.mesh.t[0,:]][:,None]+\
                   (np.outer(self.invA[0][1],gradphi[1][0,:])+\
                    np.outer(self.invA[1][1],gradphi[1][1,:]))*\
                    w1[self.mesh.t[1,:]][:,None]+\
                   (np.outer(self.invA[0][1],gradphi[2][0,:])+\
                    np.outer(self.invA[1][1],gradphi[2][1,:]))*\
                    w1[self.mesh.t[2,:]][:,None]
        else:
            W1=None
            dW1=None

        if w2 is not None:
            W2=np.outer(w2[self.mesh.t[0,:]],phi[0])+\
               np.outer(w2[self.mesh.t[1,:]],phi[1])+\
               np.outer(w2[self.mesh.t[2,:]],phi[2])
            dW2={}
            dW2[0]=(np.outer(self.invA[0][0],gradphi[0][0,:])+\
                    np.outer(self.invA[1][0],gradphi[0][1,:]))*\
                    w2[self.mesh.t[0,:]][:,None]+\
                   (np.outer(self.invA[0][0],gradphi[1][0,:])+\
                    np.outer(self.invA[1][0],gradphi[1][1,:]))*\
                    w2[self.mesh.t[1,:]][:,None]+\
                   (np.outer(self.invA[0][0],gradphi[2][0,:])+\
                    np.outer(self.invA[1][0],gradphi[2][1,:]))*\
                    w2[self.mesh.t[2,:]][:,None]
            dW2[1]=(np.outer(self.invA[0][1],gradphi[0][0,:])+\
                    np.outer(self.invA[1][1],gradphi[0][1,:]))*\
                    w2[self.mesh.t[0,:]][:,None]+\
                   (np.outer(self.invA[0][1],gradphi[1][0,:])+\
                    np.outer(self.invA[1][1],gradphi[1][1,:]))*\
                    w2[self.mesh.t[1,:]][:,None]+\
                   (np.outer(self.invA[0][1],gradphi[2][0,:])+\
                    np.outer(self.invA[1][1],gradphi[2][1,:]))*\
                    w2[self.mesh.t[2,:]][:,None]
        else:
            W2=None
            dW2=None

        # bilinear form
        if bilinear:
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
                    data[ixs]=np.dot(fform(u,v,du,dv,x,h,W1,dW1,W2,dW2),W)*np.abs(self.detA)
                    rows[ixs]=self.mesh.t[i,:]
                    cols[ixs]=self.mesh.t[j,:]
        
            return coo_matrix((data,(rows,cols)),shape=(nv,nv)).tocsr()

        # linear form
        else:
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
                data[ixs]=np.dot(fform(v,dv,x,h,W1,dW1,W2,dW2),W)*np.abs(self.detA)
                rows[ixs]=self.mesh.t[i,:]
                cols[ixs]=np.zeros(nt)
        
            return coo_matrix((data,(rows,cols)),shape=(nv,1)).toarray().T[0]

    def fasm(self,form,find=None,intorder=2,w=None):
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

        # check and fix parameters of form
        oldparams=inspect.getargspec(form).args
        if 'u' in oldparams or 'du' in oldparams:
            paramlist=['u','v','du','dv','x','h','n','w']
            bilinear=True
        else:
            paramlist=['v','dv','x','h','n','w']
            bilinear=False
        fform=self.fillargs(form,paramlist)

        X,W=get_quadrature("line",intorder)
        
        # local basis
        phi={}
        phi[0]=lambda x,y: 1.-x-y
        phi[1]=lambda x,y: x
        phi[2]=lambda x,y: y

        gradphi_x={}
        gradphi_x[0]=lambda x,y: -1.+0*x
        gradphi_x[1]=lambda x,y: 1.+0*x
        gradphi_x[2]=lambda x,y: 0*x

        gradphi_y={}
        gradphi_y[0]=lambda x,y: -1.+0*x
        gradphi_y[1]=lambda x,y: 0*x
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
        tlen=np.sqrt(t[0]**2+t[1]**2)
        t[0]/=tlen
        t[1]/=tlen
        
        # normal vectors
        n={}
        n[0]=-t[1]
        n[1]=t[0]

        # map normal vectors to reference coords to correct sign (outward normals wanted)
        n_ref={}
        n_ref[0]=self.invA[0][0][tind]*n[0]+self.invA[1][0][tind]*n[1]
        n_ref[1]=self.invA[0][1][tind]*n[0]+self.invA[1][1][tind]*n[1]
        
        # change the sign of the following normal vectors
        meps=np.finfo(float).eps
        csgn=np.nonzero((n_ref[0]<0)*(n_ref[1]<0)+\
                        (n_ref[0]>0)*(n_ref[1]<meps)*(n_ref[1]>-meps)+\
                        (n_ref[0]<meps)*(n_ref[0]>-meps)*(n_ref[1]>0))[0]
        n[0][csgn]=(-1.)*(n[0][csgn])
        n[1][csgn]=(-1.)*(n[1][csgn])
        
        n[0]=np.tile(n[0][:,None],(1,W.shape[0]))
        n[1]=np.tile(n[1][:,None],(1,W.shape[0]))

        if w is not None:
            w1=phi[0](Y[0],Y[1])*w[self.mesh.t[0,tind][:,None]]+\
               phi[1](Y[0],Y[1])*w[self.mesh.t[1,tind][:,None]]+\
               phi[2](Y[0],Y[1])*w[self.mesh.t[2,tind][:,None]]
        else:
            w1=None

        # bilinear form
        if bilinear:
            # initialize sparse matrix structures
            data=np.zeros(9*ne)
            rows=np.zeros(9*ne)
            cols=np.zeros(9*ne)

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
                    data[ixs]=np.dot(fform(u,v,du,dv,x,h,n,w1),W)*np.abs(self.detB[find])
                    rows[ixs]=self.mesh.t[i,tind]
                    cols[ixs]=self.mesh.t[j,tind]
        
            return coo_matrix((data,(rows,cols)),shape=(nv,nv)).tocsr()
        # linear form
        else:
            # initialize sparse matrix structures
            data=np.zeros(3*ne)
            rows=np.zeros(3*ne)
            cols=np.zeros(3*ne)

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
                data[ixs]=np.dot(fform(v,dv,x,h,n,w1),W)*np.abs(self.detB[find])
                rows[ixs]=self.mesh.t[i,tind]
                cols[ixs]=np.zeros(ne)
        
            return coo_matrix((data,(rows,cols)),shape=(nv,1)).toarray().T[0]

