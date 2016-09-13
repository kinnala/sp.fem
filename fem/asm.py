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
                return oldform(x[y[0]],x[y[1]],x[y[2]],x[y[3]],x[y[4]],
                               x[y[5]])
        elif len(oldargs)==7:
            def newform(*x):
                return oldform(x[y[0]],x[y[1]],x[y[2]],x[y[3]],x[y[4]],
                               x[y[5]],x[y[6]])
        elif len(oldargs)==8:
            def newform(*x):
                return oldform(x[y[0]],x[y[1]],x[y[2]],x[y[3]],x[y[4]],
                               x[y[5]],x[y[6]],x[y[7]])
        elif len(oldargs)==9:
            def newform(*x):
                return oldform(x[y[0]],x[y[1]],x[y[2]],x[y[3]],x[y[4]],
                               x[y[5]],x[y[6]],x[y[7]],x[y[8]])
        elif len(oldargs)==10:
            def newform(*x):
                return oldform(x[y[0]],x[y[1]],x[y[2]],x[y[3]],x[y[4]],
                               x[y[5]],x[y[6]],x[y[7]],x[y[8]],x[y[9]])
        else:
            raise NotImplementedError("Assembler.fillargs(): the maximum "
                                      "number of arguments reached!")

        return newform

class AssemblerGlobal(Assembler):
    """A slowish assembler for globally defined elements.
    
    The finite element given to this assembler is defined through
    degrees-of-freedom. The actual basis functions are solved
    in the constructor and the assembly is performed by looping
    over elements and computing local stiffness matrices.
    """
    def __init__(self,mesh,elem_u,elem_v=None):
        if not isinstance(mesh,fem.mesh.Mesh):
            raise Exception("AssemblerGlobal.__init__(): first parameter "
                            "must be an instance of fem.mesh.Mesh!")
        if not isinstance(elem_u,fem.element.ElementGlobal):
            raise Exception("AssemblerGlobal.__init__(): second parameter "
                            "must be an instance of fem.element.ElementGlobal!")

        self.mesh=mesh
        self.elem_u=elem_u
        self.dofnum_u=Dofnum(mesh,elem_u)
        self.mapping=mesh.mapping()

        if elem_v is None:
            self.elem_v=elem_u
            self.dofnum_v=self.dofnum_u
        else:
            self.elem_v=elem_v
            self.dofnum_v=Dofnum(mesh,elem_v)

        def quick_inverse(A):
            # quick inversion of an array of matrices
            identity=np.identity(A.shape[2],dtype=A.dtype)
            return np.array([np.linalg.solve(x,identity) for x in A])

        # solve basis functions
        self.Nbfun_u=len(self.elem_u.C)
        V=np.zeros((self.Nbfun_u,self.Nbfun_u,mesh.t.shape[1]))
        for itr in range(self.Nbfun_u):
            for jtr in range(self.Nbfun_u):
                V[itr,jtr]=self.elem_u.gdofs(self.mapping,itr,jtr)
        Vinv=quick_inverse(np.transpose(V,(2,1,0))) 

        # construct basis function matrices
        tind=np.arange(self.mesh.t.shape[1])
        self.bfuns_u={}
        for k in tind:
            self.bfuns_u[k]={}
            for jtr in range(self.Nbfun_u):
                self.bfuns_u[k][jtr]=self.elem_u.C[0]*0.0
                for itr in range(self.Nbfun_u):
                    self.bfuns_u[k][jtr]+=Vinv[k,jtr,itr]*self.elem_u.C[itr]

        if elem_v is None:
            self.bfuns_v=self.bfuns_u
            self.Nbfun_v=self.Nbfun_u
        else:
            raise NotImplementedError("Two separate elements not yet supported!")

    def iasm(self,form,intorder=2):
        tind=np.arange(self.mesh.t.shape[1])

        # check and fix parameters of form
        oldparams=inspect.getargspec(form).args
        if 'u' in oldparams or 'du' in oldparams:
            paramlist=['u','v','du','dv','ddu','ddv','x','w','h']
            bilinear=True
        else:
            paramlist=['v','dv','ddv','x','w','h']
            bilinear=False
        fform=self.fillargs(form,paramlist)

        # quadrature points and weights
        X,W=get_quadrature(self.mesh.refdom,intorder)

        x=self.mapping.F(X,tind)

        # jacobian at quadrature points
        detDF=self.mapping.detDF(X,tind)

        data=np.zeros(tind.shape[0]*self.Nbfun_u*self.Nbfun_v)
        rows=np.zeros(tind.shape[0]*self.Nbfun_u*self.Nbfun_v)
        cols=np.zeros(tind.shape[0]*self.Nbfun_u*self.Nbfun_v)
        ktr=0
        # loop over elements and do assembly
        dim=self.mesh.p.shape[0]
        import numpy.polynomial.polynomial as npp
        for k in tind:
            for jtr in range(self.Nbfun_u):
                if dim==2:
                    u=npp.polyval2d(x[0][k,:],x[1][k,:],self.bfuns_u[k][jtr])
                    du={}
                    du[0]=npp.polyval2d(x[0][k,:],x[1][k,:],npp.polyder(self.bfuns_u[k][jtr],axis=0))
                    du[1]=npp.polyval2d(x[0][k,:],x[1][k,:],npp.polyder(self.bfuns_u[k][jtr],axis=1))
                else:
                    raise NotImplementedError("Used dimension not supported!")
                for itr in range(self.Nbfun_v):
                    if dim==2:
                        v=npp.polyval2d(x[0][k,:],x[1][k,:],self.bfuns_v[k][itr])
                        dv={}
                        dv[0]=npp.polyval2d(x[0][k,:],x[1][k,:],npp.polyder(self.bfuns_v[k][itr],axis=0))
                        dv[1]=npp.polyval2d(x[0][k,:],x[1][k,:],npp.polyder(self.bfuns_v[k][itr],axis=1))
                    else:
                        raise NotImplementedError("Used dimension not supported!")

                    data[ktr]=np.dot(fform(u,v,du,dv,0*u,0*u,0*u,0*u,0*u),W*np.abs(detDF[k]))
                    rows[ktr]=self.dofnum_v.t_dof[itr,k]
                    cols[ktr]=self.dofnum_u.t_dof[jtr,k]
                    ktr+=1

        return coo_matrix((data,(rows,cols)),shape=(self.dofnum_v.N,self.dofnum_u.N)).tocsr()

class AssemblerElement(Assembler):
    """A quasi-fast assembler for arbitrary element/mesh/mapping."""
    def __init__(self,mesh,elem_u,elem_v=None,mapping=None):
        if not isinstance(mesh,fem.mesh.Mesh):
            raise Exception("AssemblerElement.__init__(): first parameter "
                            "must be an instance of fem.mesh.Mesh!")
        if not isinstance(elem_u,fem.element.Element):
            raise Exception("AssemblerElement.__init__(): second parameter "
                            "must be an instance of fem.element.Element!")

        # get default mapping from the mesh
        if mapping is None:
            self.mapping=mesh.mapping()
        else:
            self.mapping=mapping # assumes an already initialized mapping

        self.mesh=mesh
        self.elem_u=elem_u
        self.dofnum_u=Dofnum(mesh,elem_u)

        # duplicate test function element type if None is given
        if elem_v is None:
            self.elem_v=elem_u
            self.dofnum_v=self.dofnum_u
        else:
            self.elem_v=elem_v
            self.dofnum_v=Dofnum(mesh,elem_v)
     
    def iasm(self,form,intorder=None,tind=None,interp=None):
        """Interior assembly."""
        if tind is None:
            # assemble on all elements by default
            tind=range(self.mesh.t.shape[1])
        nt=len(tind)
        if intorder is None:
            # compute the maximum polynomial degree from elements
            intorder=self.elem_u.maxdeg+self.elem_v.maxdeg
        
        # check and fix parameters of form
        oldparams=inspect.getargspec(form).args
        if 'u' in oldparams or 'du' in oldparams:
            paramlist=['u','v','du','dv','ddu','ddv','x','w','h']
            bilinear=True
        else:
            paramlist=['v','dv','ddv','x','w','h']
            bilinear=False
        fform=self.fillargs(form,paramlist)
        
        # quadrature points and weights
        X,W=get_quadrature(self.mesh.refdom,intorder)

        # global quadrature points
        x=self.mapping.F(X,tind)

        # jacobian at quadrature points
        detDF=self.mapping.detDF(X,tind)
        
        Nbfun_u=self.dofnum_u.t_dof.shape[0]
        Nbfun_v=self.dofnum_v.t_dof.shape[0]  

        # interpolate some previous discrete function at quadrature points
        w={}
        if interp is not None:
            for k in interp:
                w[k]=0.0*x[0]
                for j in range(Nbfun_u):
                    phi,_=self.elem_u.lbasis(X,j)
                    w[k]=w[k]+np.outer(interp[k][self.dofnum_u.t_dof[j,:]],phi)

        # compute the mesh parameter from jacobian determinant
        h=np.abs(detDF)**(1.0/self.mesh.dim())

        # TODO think about precomputing [u,du,ddu] in inner loop MAKE OPTIONAL FLAG FOR OPTIMIZATION
        
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
                    data[ixs]=np.dot(fform(u,v,du,dv,ddu,ddv,x,w,h)*np.abs(detDF),W)
                    rows[ixs]=self.dofnum_v.t_dof[i,tind]
                    cols[ixs]=self.dofnum_u.t_dof[j,tind]
        
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
                data[ixs]=np.dot(fform(v,dv,ddv,x,w,h)*np.abs(detDF),W)
                rows[ixs]=self.dofnum_v.t_dof[i,:]
                cols[ixs]=np.zeros(nt)
        
            return coo_matrix((data,(rows,cols)),shape=(self.dofnum_v.N,1)).toarray().T[0]

    # TODO add ifasm (interior facet assembly) for DG methods etc.
            
    def fasm(self,form,find=None,intorder=None,normals=True): # TODO fix and test
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
            paramlist=['u','v','du','dv','x','h','n']
            #paramlist=['u','v','du','dv','x','h','n','w']
            bilinear=True
        else:
            paramlist=['v','dv','x','h','n']
            #paramlist=['v','dv','x','h','n','w']
            bilinear=False
        fform=self.fillargs(form,paramlist)

        X,W=get_quadrature(self.mesh.brefdom,intorder)
        
        # boundary element indices
        tind=self.mesh.f2t[0,find]

        # mappings
        x=self.mapping.G(X,find=find) # reference facet to global facet
        Y=self.mapping.invF(x,tind=tind) # global facet to reference element
        
        Nbfun_u=self.dofnum_u.t_dof.shape[0]
        Nbfun_v=self.dofnum_v.t_dof.shape[0] 

        detDG=self.mapping.detDG(X,find)        

        # compute normal vectors
        n={}
        if normals:
            n=self.mapping.normals(Y,tind,find,self.mesh.t2f)

        # compute the mesh parameter from jacobian determinant
        if self.mesh.dim()>=1.0:
            h=np.abs(detDG)**(1.0/(self.mesh.dim()-1.0))
        else: # exception for 1D mesh
            h=None
        
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
                    data[ixs]=np.dot(fform(u,v,du,dv,x,h,n)*np.abs(detDG),W)
                    rows[ixs]=self.dofnum_v.t_dof[i,tind]
                    cols[ixs]=self.dofnum_u.t_dof[j,tind]
        
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
                data[ixs]=np.dot(fform(v,dv,x,h,n)*np.abs(detDG),W)
                rows[ixs]=self.dofnum_v.t_dof[i,tind]
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
            if not isinstance(du,dict):
                return du*dv
            elif len(du)==2:
                return du[0]*dv[0]+du[1]*dv[1]
            elif len(du)==3:
                return du[0]*dv[0]+du[1]*dv[1]+du[2]*dv[2]
            else:
                raise NotImplementedError("AssemblerElement.H1error: not implemented for current domain dimension!")
    
        def fv(dv,x):
            if not isinstance(x,dict):
                return dexact(x)*dv
            elif len(x)==2:
                return dexact[0](x)*dv[0]+dexact[1](x)*dv[1]
            elif len(x)==3:
                return dexact[0](x)*dv[0]+dexact[1](x)*dv[1]+dexact[2](x)*dv[2]
            else:
                raise NotImplementedError("AssemblerElement.H1error: not implemented for current domain dimension!")
            
        M=self.iasm(uv)
        f=self.iasm(fv)
        
        detDF=self.mapping.detDF(X)
        x=self.mapping.F(X)
        
        if not isinstance(x,dict):
            uu=np.sum(np.dot((dexact(x)**2)*np.abs(detDF),W))
        elif len(x)==2:
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
        if hasattr(mesh,'facets'): # 2d or 3d mesh
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

        # facet dofs (if 2D or 3D)
        if hasattr(mesh,'facets'):      
            for itr in range(mesh.t2f.shape[0]):
                self.t_dof=np.vstack((self.t_dof,self.f_dof[:,mesh.t2f[itr,:]]))
        
        self.t_dof=np.vstack((self.t_dof,self.i_dof))
        
        self.N=np.max(self.t_dof)+1
        
    def getdofs(self,N=None,F=None,E=None,T=None):
        """Return global DOF numbers corresponding to each node(N), facet(F) and triangle(T)"""
        dofs=np.zeros(0,dtype=np.int64)        
        if N is not None:
            dofs=np.hstack((dofs,self.n_dof[:,N].flatten()))
        if F is not None:
            dofs=np.hstack((dofs,self.f_dof[:,F].flatten()))
        if E is not None:
            dofs=np.hstack((dofs,self.e_dof[:,E].flatten()))
        if T is not None:
            dofs=np.hstack((dofs,self.i_dof[:,T].flatten()))
        return dofs.flatten()
