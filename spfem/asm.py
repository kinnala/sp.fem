# -*- coding: utf-8 -*-
"""
Assembly of matrices related to linear and bilinear forms.

*Example 1*. Assemble the stiffness matrix related to
the Poisson problem using the piecewise linear elements.

.. code-block:: python

    import spfem.mesh as fmsh
    import spfem.asm as fasm
    import spfem.element as felem

    m=fmsh.MeshTri()
    m.refine(3)
    e=felem.ElementTriP1()
    a=fasm.AssemblerElement(m,e)

    def bilinear_form(du,dv):
        return du[0]*dv[0]+du[1]*dv[1]

    K=a.iasm(bilinear_form)
"""
import numpy as np
import spfem.mesh
import spfem.mapping
import inspect
from spfem.quadrature import get_quadrature
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
    """An assembler for globally defined elements,
    cf. :class:`spfem.element.ElementGlobal`.
    
    The finite element given to this assembler is defined
    in such a way that given a (global) triangle and (global)
    quadrature points, the element must be able to evaluate
    the global basis functions at the given quadrature points. 

    The assembler is useful for elements where the reference
    triangle approach is not simple. Such elements include
    for example plate elements (i.e. Argyris element).

    Note: This assembler is slow since it explicitly loops
    over the elements using Python for-loop.
    """
    def __init__(self,mesh,elem):
        if not isinstance(mesh,spfem.mesh.Mesh):
            raise Exception("AssemblerGlobal.__init__(): first parameter "
                            "must be an instance of spfem.mesh.Mesh!")
        if not isinstance(elem,spfem.element.ElementGlobal):
            raise Exception("AssemblerGlobal.__init__(): second parameter "
                            "must be an instance of spfem.element.ElementGlobal!")

        self.mesh=mesh
        self.elem=elem
        self.dofnum=Dofnum(mesh,elem)
        self.mapping=mesh.mapping()
        self.Nbfun=self.dofnum.t_dof.shape[0]

    def ifposteriori(self,form,w,intorder=None):
        # evaluate norm on all interior facets
        find=np.nonzero(self.mesh.f2t[1,:]>0)[0]

        if intorder is None:
            intorder=2*self.elem.maxdeg            

        # check and fix parameters of form
        oldparams=inspect.getargspec(form).args
        #paramlist=['u','v','du','dv','ddu','ddv','x']
        paramlist=['u1','u2','x']
        fform=self.fillargs(form,paramlist)

        X,W=get_quadrature(self.mesh.brefdom,intorder)

        # boundary element indices
        tind1=self.mesh.f2t[0,find]
        tind2=self.mesh.f2t[1,find]

        # mappings
        x=self.mapping.G(X,find=find) # reference facet to global facet
        detDG=self.mapping.detDG(X,find)        

        # initialize sparse matrix
        data=np.zeros(self.mesh.facets.shape[1])

        # loop over elements and do assembly
        dim=self.mesh.p.shape[0]
        ktr=0

        #import pdb; pdb.set_trace()

        for k in find:
            # quadrature points in current facet
            xk={}
            for itr in range(dim):
                xk[itr]=x[itr][ktr,:]

            # evaluate global bases of both elements
            t1=self.mesh.f2t[0,k]
            t2=self.mesh.f2t[1,k]
            u1,du1,ddu1=self.elem.gbasis(self.mesh,xk,t1)
            u2,du2,ddu2=self.elem.gbasis(self.mesh,xk,t2)
            
            U1=0*u1[0]
            U2=0*u1[0]
            # interpolate basis functions and solution vector
            # at quadrature points
            for jtr in range(self.Nbfun):
                ix1=self.dofnum.t_dof[jtr,t1]
                ix2=self.dofnum.t_dof[jtr,t2]
                U1+=w[ix1]*u1[jtr]
                U2+=w[ix2]*u2[jtr]

            # loop over basis functions
            for jtr in range(self.Nbfun):
                ix1=self.dofnum.t_dof[jtr,t1]
                ix2=self.dofnum.t_dof[jtr,t2]
                data[k]+=np.dot(fform(U1,U2,
                                      xk)**2,W*np.abs(detDG[ktr]))
            ktr+=1

        return data


    def iasm(self,form,intorder=None,tind=None):
        if tind is None:
            # by default, all elements
            tind=np.arange(self.mesh.t.shape[1])

        # check and fix parameters of form
        oldparams=inspect.getargspec(form).args
        if 'u' in oldparams or 'du' in oldparams or 'ddu' in oldparams:
            paramlist=['u','v','du','dv','ddu','ddv','x']
            bilinear=True
        else:
            paramlist=['v','dv','ddv','x']
            bilinear=False
        fform=self.fillargs(form,paramlist)

        if intorder is None:
            # compute the maximum polynomial degree from elements
            intorder=2*self.elem.maxdeg

        # quadrature points and weights
        X,W=get_quadrature(self.mesh.refdom,intorder)

        x=self.mapping.F(X,tind)

        # jacobian at quadrature points
        detDF=self.mapping.detDF(X,tind)

        # loop over elements and do assembly
        dim=self.mesh.p.shape[0]
        ktr=0

        if bilinear:
            # initialize sparse matrix
            data=np.zeros(tind.shape[0]*self.Nbfun**2)
            rows=np.zeros(tind.shape[0]*self.Nbfun**2)
            cols=np.zeros(tind.shape[0]*self.Nbfun**2)

            for k in tind:
                # quadrature points in current element
                xk={}
                for itr in range(dim):
                    xk[itr]=x[itr][k,:]

                # basis function and derivatives in quadrature points
                u,du,ddu=self.elem.gbasis(self.mesh,xk,k)

                # assemble local stiffness matrix
                for jtr in range(self.Nbfun):
                    for itr in range(self.Nbfun):
                        data[ktr]=np.dot(fform(u[jtr],u[itr],
                                               du[jtr],du[itr],
                                               ddu[jtr],ddu[itr],xk),W*np.abs(detDF[k]))
                        rows[ktr]=self.dofnum.t_dof[itr,k]
                        cols[ktr]=self.dofnum.t_dof[jtr,k]
                        ktr+=1

            return coo_matrix((data,(rows,cols)),shape=(self.dofnum.N,self.dofnum.N)).tocsr()

        else:
            # initialize sparse matrix structures
            data=np.zeros(tind.shape[0]*self.Nbfun)
            rows=np.zeros(tind.shape[0]*self.Nbfun)
            cols=np.zeros(tind.shape[0]*self.Nbfun)

            for k in tind:
                # quadrature points in current element
                xk={}
                for itr in range(dim):
                    xk[itr]=x[itr][k,:]

                u,du,ddu=self.elem.gbasis(self.mesh,xk,k)

                # assemble local laod vector
                for itr in range(self.Nbfun):
                    data[ktr]=np.dot(fform(u[itr],du[itr],ddu[itr],xk),W*np.abs(detDF[k]))
                    rows[ktr]=self.dofnum.t_dof[itr,k]
                    ktr+=1
            
            return coo_matrix((data,(rows,cols)),shape=(self.dofnum.N,1)).toarray().T[0]

class AssemblerElement(Assembler):
    """A quasi-fast assembler for arbitrary element/mesh/mapping.
    
    Parameters
    ----------
    mesh : :class:`spfem.mesh.Mesh`
        The finite element mesh.

    elem_u : :class:`spfem.element.Element`
        The element for the solution function.
        
    elem_v : (OPTIONAL) :class:`spfem.element.Element`
        The element for the test function. By default, same element is used for both.
        
    mapping : (OPTIONAL) :class:`spfem.mapping.Mapping`
        The mesh will give some sort of default mapping but sometimes, e.g.
        when using isoparametric elements, the user might have to support
        a different mapping.
    """
    def __init__(self,mesh,elem_u,elem_v=None,mapping=None):
        if not isinstance(mesh,spfem.mesh.Mesh):
            raise Exception("AssemblerElement.__init__(): first parameter "
                            "must be an instance of spfem.mesh.Mesh!")
        if not isinstance(elem_u,spfem.element.Element):
            raise Exception("AssemblerElement.__init__(): second parameter "
                            "must be an instance of spfem.element.Element!")

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
        """Return a matrix related to a bilinear or linear form
        where the integral is over the interior of the domain.
        
        Parameters
        ----------
        form : function handle
            The bilinear or linear form function handle.
            The supported parameters can be found in the
            following table.

            +-----------+----------------------+--------------+
            | Parameter | Explanation          | Supported in |
            +-----------+----------------------+--------------+
            | u         | solution             | bilinear     |
            +-----------+----------------------+--------------+
            | v         | test fun             | both         |
            +-----------+----------------------+--------------+
            | du        | solution derivatives | bilinear     |
            +-----------+----------------------+--------------+
            | dv        | test fun derivatives | both         |
            +-----------+----------------------+--------------+
            | x         | spatial location     | both         |
            +-----------+----------------------+--------------+
            | w         | cf. interp           | both         |
            +-----------+----------------------+--------------+
            | h         | the mesh parameter   | both         |
            +-----------+----------------------+--------------+

            The function handle must use these exact names for
            the variables. Unused variable names can be omitted.
            
            Examples of valid bilinear forms:
            ::
                
                def bilin_form1(du,dv):
                    # Note that the element must be
                    # defined for two-dimensional
                    # meshes for this to make sense!
                    return du[0]*dv[0]+du[1]*dv[1]

                def bilin_form2(du,v):
                    return du[0]*v

                bilin_form3 = lambda u,v,x: x[0]**2*u*v

            Examples of valid linear forms:
            ::
                
                def lin_form1(v):
                    return v

                def lin_form2(h,x,v):
                    import numpy as np
                    mesh_parameter=h
                    X=x[0]
                    Y=x[1]
                    return mesh_parameter*np.sin(np.pi*X)*np.sin(np.pi*Y)*v

            The linear forms are automatically detected to be
            non-bilinear through the omission of u or du.

        intorder : int
            The order of polynomials for which the applied
            quadrature rule is exact. By default,
            2*Element.maxdeg is used. Reducing this number
            can sometimes reduce the computation time.

        interp : numpy array
            Using this flag, the user can provide
            a solution vector that is interpolated
            to the quadrature points and included in
            the computation of the bilinear form
            (the variable w). Useful e.g. when solving
            nonlinear problems.
        """
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
            paramlist=['u','v','du','dv','x','w','h']
            bilinear=True
        else:
            paramlist=['v','dv','x','w','h']
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
                u,du=self.elem_u.gbasis(self.mapping,X,j,tind)
                for i in range(Nbfun_v):
                    v,dv=self.elem_v.gbasis(self.mapping,X,i,tind)
            
                    # find correct location in data,rows,cols
                    ixs=slice(nt*(Nbfun_v*j+i),nt*(Nbfun_v*j+i+1))
                    
                    # compute entries of local stiffness matrices
                    data[ixs]=np.dot(fform(u,v,du,dv,x,w,h)*np.abs(detDF),W)
                    rows[ixs]=self.dofnum_v.t_dof[i,tind]
                    cols[ixs]=self.dofnum_u.t_dof[j,tind]
        
            return coo_matrix((data,(rows,cols)),shape=(self.dofnum_v.N,self.dofnum_u.N)).tocsr()
            
        else:
            # initialize sparse matrix structures
            data=np.zeros(Nbfun_v*nt)
            rows=np.zeros(Nbfun_v*nt)
            cols=np.zeros(Nbfun_v*nt)
            
            for i in range(Nbfun_v):
                v,dv=self.elem_v.gbasis(self.mapping,X,i,tind)

                # find correct location in data,rows,cols
                ixs=slice(nt*i,nt*(i+1))
                
                # compute entries of local stiffness matrices
                data[ixs]=np.dot(fform(v,dv,x,w,h)*np.abs(detDF),W)
                rows[ixs]=self.dofnum_v.t_dof[i,:]
                cols[ixs]=np.zeros(nt)
        
            return coo_matrix((data,(rows,cols)),shape=(self.dofnum_v.N,1)).toarray().T[0]

    # TODO add ifasm (interior facet assembly) for DG methods etc.
            
    def fasm(self,form,find=None,intorder=None,normals=True,interp=None):
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
            paramlist=['u','v','du','dv','x','h','n','w']
            bilinear=True
        else:
            paramlist=['v','dv','x','h','n','w']
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
        if self.mesh.dim()>1.0:
            h=np.abs(detDG)**(1.0/(self.mesh.dim()-1.0))
        else: # exception for 1D mesh (no boundary h defined)
            h=None

        # interpolate some previous discrete function at quadrature points
        w={}
        if interp is not None:
            for k in interp:
                w[k]=0.0*x[0]
                for j in range(Nbfun_u):
                    phi,_=self.elem_u.gbasis(self.mapping,Y,j,tind)
                    w[k]=w[k]+interp[k][self.dofnum_u.t_dof[j,tind],None]*phi
        
        # bilinear form
        if bilinear:
            # initialize sparse matrix structures
            data=np.zeros(Nbfun_u*Nbfun_v*ne)
            rows=np.zeros(Nbfun_u*Nbfun_v*ne)
            cols=np.zeros(Nbfun_u*Nbfun_v*ne)

            for j in range(Nbfun_u):
                u,du=self.elem_u.gbasis(self.mapping,Y,j,tind)
                for i in range(Nbfun_v):
                    v,dv=self.elem_v.gbasis(self.mapping,Y,i,tind)
           
                    # find correct location in data,rows,cols
                    ixs=slice(ne*(Nbfun_v*j+i),ne*(Nbfun_v*j+i+1))
                    
                    # compute entries of local stiffness matrices
                    data[ixs]=np.dot(fform(u,v,du,dv,x,h,n,w)*np.abs(detDG),W)
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
                v,dv=self.elem_v.gbasis(self.mapping,Y,i,tind)
        
                # find correct location in data,rows,cols
                ixs=slice(ne*i,ne*(i+1))
                
                # compute entries of local stiffness matrices
                data[ixs]=np.dot(fform(v,dv,x,h,n,w)*np.abs(detDG),W)
                rows[ixs]=self.dofnum_v.t_dof[i,tind]
                cols[ixs]=np.zeros(ne)
        
            return coo_matrix((data,(rows,cols)),shape=(self.dofnum_v.N,1)).toarray().T[0]
            
    def L2error(self,uh,exact,intorder=None):
        """
        Compute :math:`L^2` error against exact solution.
        
        The computation is based on the following identity:
            
        .. math::
            
            \|u-u_h\|_0^2 = (u,u)+(u_h,u_h)-2(u,u_h).
        """
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
        """
        Compute :math:`H^1` seminorm error against exact solution.

        The computation is based on the following identity:
            
        .. math::
            
            \|\\nabla(u-u_h)\|_0^2 = (\\nabla u,\\nabla u)+(\\nabla u_h, \\nabla u_h)-2(\\nabla u, \\nabla u_h).
        """
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
        self.n_dof=np.reshape(np.arange(element.n_dofs*mesh.p.shape[1],dtype=np.int64),
                (element.n_dofs,mesh.p.shape[1]),order='F')
        offset=element.n_dofs*mesh.p.shape[1]
        if hasattr(mesh,'edges'): # 3d mesh
            self.e_dof=np.reshape(np.arange(element.e_dofs*mesh.edges.shape[1],dtype=np.int64),
                    (element.e_dofs,mesh.edges.shape[1]),order='F')+offset
            offset=offset+element.e_dofs*mesh.edges.shape[1]
        if hasattr(mesh,'facets'): # 2d or 3d mesh
            self.f_dof=np.reshape(np.arange(element.f_dofs*mesh.facets.shape[1],dtype=np.int64),
                    (element.f_dofs,mesh.facets.shape[1]),order='F')+offset
            offset=offset+element.f_dofs*mesh.facets.shape[1]
        self.i_dof=np.reshape(np.arange(element.i_dofs*mesh.t.shape[1],dtype=np.int64),
                (element.i_dofs,mesh.t.shape[1]),order='F')+offset
        
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
