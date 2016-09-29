"""
The finite element definitions.
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyder,polyval2d
from fem.utils import const_cell

class Element(object):
    """A finite element."""

    maxdeg=0 #: Maximum polynomial degree
    dim=0 #: Spatial dimension

    n_dofs=0 #: Number of nodal dofs
    i_dofs=0 #: Number of interior dofs
    f_dofs=0 #: Number of facet dofs (2d and 3d only)
    e_dofs=0 #: Number of edge dofs (3d only)

    def lbasis(self,X,i):
        """Returns local basis functions evaluated at some local points."""
        raise NotImplementedError("Element.lbasis: local basis (lbasis) not implemented!")

    def gbasis(self,X,i,tind):
        """Returns global basis functions evaluated at some local points."""
        raise NotImplementedError("Element.gbasis: local basis (lbasis) not implemented!")

class ElementGlobal(Element):
    """An element defined globally. These elements are used by :class:`fem.asm.AssemblerGlobal`."""

    def gbasis(self,mesh,qps,k):
        """Return the global basis functions of an element evaluated at
        the given quadrature points.

        Parameters
        ----------
        mesh
            The :class:`fem.mesh.Mesh` object.
        qps : dict of global quadrature points
            The global quadrature points as given by :func:`fem.mapping.Mapping.F`.
        k : int
            The index of the element in mesh structure.

        Returns
        -------
        u : dict
            A dictionary with integer keys from 0 to Nbfun.
            Here i'th dictionary entry contains the values
            of the i'th basis function evaluated at the
            quadrature points (np.array).
        du : dict
            The first derivatives. The actual contents
            are fully defined by the element behavior
            although du[i] should correspond to the
            i'th basis function.
        ddu : dict
            The second derivatives. The actual contents
            are fully defined by the element behavior
            although ddu[i] should correspond to the i'th
            basis function.
        """
        raise NotImplementedError("ElementGlobal.gbasis not implemented!")

    def _pbasis1(self,x):
        """This function defines the initial power basis
        up to p=1 for 2d elements."""
        return np.array([1.0,x[0],x[1]])

    def _pbasis1dx(self):
        return np.array([0.0,1.0,0.0])

    def _pbasis1dy(self):
        return np.array([0.0,0.0,1.0])

    def _pbasis2(self,x):
        """This function defines the initial power basis
        up to p=2 for 2d elements."""
        return np.array([1.0,x[0],x[1],x[0]**2,x[0]*x[1],x[1]**2])

    def _pbasis2dx(self,x):
        return np.array([0.0,1.0,0.0,2.0*x[0],x[1],0.0])

    def _pbasis2dy(self,x):
        return np.array([0.0,0.0,1.0,0.0,x[0],2.0*x[1]])

    def _pbasis2dxx(self):
        return np.array([0.0,0.0,0.0,2.0,0.0,0.0])

    def _pbasis2dxy(self):
        return np.array([0.0,0.0,0.0,0.0,1.0,0.0])

    def _pbasis2dyy(self):
        return np.array([0.0,0.0,0.0,0.0,0.0,2.0])

    def _pbasisNinit(self,dim,N,debug=False):
        if not hasattr(self,'_pbasis'+str(N)):
            import sympy as sp
            from sympy.abc import x,y,z
            R=range(N+1)
            if dim==1:
                pbasis=sp.Matrix([x**i for i in R if i<=N]) # TODO fix this
                setattr(self,'_pbasis'+str(N),lambda X:sp.lambdify(x,pbasis)(X).flatten().astype(np.float64))
                setattr(self,'_pbasis'+str(N)+'dx',lambda X:sp.lambdify(x,sp.diff(pbasis,x))(X).flatten().astype(np.float64))
                setattr(self,'_pbasis'+str(N)+'dxx',lambda X:sp.lambdify(x,sp.diff(pbasis,x,2))(X).flatten().astype(np.float64))
            elif dim==2:
                pbasis=sp.Matrix([x**i*y**j for i in R for j in R if i+j<=N])
                tmp1=sp.lambdify((x,y),pbasis)
                setattr(self,'_pbasis'+str(N),lambda X:tmp1(X[0],X[1]).flatten().astype(np.float64))
                tmp2=sp.lambdify((x,y),sp.diff(pbasis,x))
                setattr(self,'_pbasis'+str(N)+'dx',lambda X:tmp2(X[0],X[1]).flatten().astype(np.float64))
                tmp3=sp.lambdify((x,y),sp.diff(pbasis,y))
                setattr(self,'_pbasis'+str(N)+'dy',lambda X:tmp3(X[0],X[1]).flatten().astype(np.float64))
                tmp4=sp.lambdify((x,y),sp.diff(pbasis,x,2))
                setattr(self,'_pbasis'+str(N)+'dxx',lambda X:tmp4(X[0],X[1]).flatten().astype(np.float64))
                tmp5=sp.lambdify((x,y),sp.diff(pbasis,x,y))
                setattr(self,'_pbasis'+str(N)+'dxy',lambda X:tmp5(X[0],X[1]).flatten().astype(np.float64))
                tmp6=sp.lambdify((x,y),sp.diff(pbasis,y,2))
                setattr(self,'_pbasis'+str(N)+'dyy',lambda X:tmp6(X[0],X[1]).flatten().astype(np.float64))
            else:
                raise NotImplementedError("ElementGlobal._pbasisNinit: the given "+\
                                          "dimension not implemented!")
            if debug:
                print pbasis

    def visualize_basis_2d(self,show_du=False,show_ddu=False):
        """Draw the basis functions given by self.gbasis.
        Only for 2D triangular elements. For debugging purposes."""
        if self.dim!=2:
            raise NotImplementedError("ElementGlobal.visualize_basis_2d supports "+
                                      "only two dimensional triangular elements.")
        import copy
        import fem.mesh as fmsh
        m=fmsh.MeshTri(np.array([[0.5,0.0,1.0],[0.0,1.0,1.0]]),
                       np.array([[0],[1],[2]]))
        M=copy.deepcopy(m)
        m.refine(4)
        qps={}
        qps[0]=np.array([m.p[0,:]])
        qps[1]=np.array([m.p[1,:]])
        u,du,ddu=self.gbasis(M,qps,0)

        for itr in range(len(u)):
            m.plot3(u[itr])

        if show_du:
            for itr in range(len(u)):
                m.plot3(du[itr][0])
            for itr in range(len(u)):
                m.plot3(du[itr][1])

        if show_ddu:
            for itr in range(len(u)):
                m.plot3(ddu[itr][0][0])
            for itr in range(len(u)):
                m.plot3(ddu[itr][0][1])
            for itr in range(len(u)):
                m.plot3(ddu[itr][1][0])
            for itr in range(len(u)):
                m.plot3(ddu[itr][1][1])

        m.show()
        
class ElementGlobalArgyris(ElementGlobal):
    """Argyris element for fourth-order problems."""

    n_dofs=6
    f_dofs=1
    dim=2
    maxdeg=5

    def __init__(self,optimize_u=False,optimize_du=False,optimize_ddu=False):
        self.optimize_u=optimize_u
        self.optimize_du=optimize_du
        self.optimize_ddu=optimize_ddu

    def gbasis(self,mesh,qps,k):
        X=qps[0][k,:]
        Y=qps[1][k,:]
        # solve local basis functions
        N=21
        V=np.zeros((N,N))

        v1=mesh.p[:,mesh.t[0,k]]
        v2=mesh.p[:,mesh.t[1,k]]
        v3=mesh.p[:,mesh.t[2,k]]

        e1=0.5*(v1+v2)
        e2=0.5*(v2+v3)
        e3=0.5*(v1+v3)

        t1=v1-v2
        t2=v2-v3
        t3=v1-v3

        n1=np.array([t1[1],-t1[0]])
        n2=np.array([t2[1],-t2[0]])
        n3=np.array([t3[1],-t3[0]])

        n1/=np.linalg.norm(n1)
        n2/=np.linalg.norm(n2)
        n3/=np.linalg.norm(n3)

        # initialize fifth order power basis
        self._pbasisNinit(2,5)

        # evaluate dofs
        V[0,:]=self._pbasis5(v1)
        V[1,:]=self._pbasis5dx(v1)
        V[2,:]=self._pbasis5dy(v1)
        V[3,:]=self._pbasis5dxx(v1)
        V[4,:]=self._pbasis5dxy(v1)
        V[5,:]=self._pbasis5dyy(v1)

        V[6,:]=self._pbasis5(v2)
        V[7,:]=self._pbasis5dx(v2)
        V[8,:]=self._pbasis5dy(v2)
        V[9,:]=self._pbasis5dxx(v2)
        V[10,:]=self._pbasis5dxy(v2)
        V[11,:]=self._pbasis5dyy(v2)

        V[12,:]=self._pbasis5(v3)
        V[13,:]=self._pbasis5dx(v3)
        V[14,:]=self._pbasis5dy(v3)
        V[15,:]=self._pbasis5dxx(v3)
        V[16,:]=self._pbasis5dxy(v3)
        V[17,:]=self._pbasis5dyy(v3)

        V[18,:]=self._pbasis5dx(e1)*n1[0]+\
                self._pbasis5dy(e1)*n1[1]
        V[19,:]=self._pbasis5dx(e2)*n2[0]+\
                self._pbasis5dy(e2)*n2[1]
        V[20,:]=self._pbasis5dx(e3)*n3[0]+\
                self._pbasis5dy(e3)*n3[1]

        Vinv=np.linalg.inv(V).T

        u=const_cell(np.zeros(len(X)),N)
        du=const_cell(np.zeros(len(X)),N,2)
        ddu=const_cell(np.zeros(len(X)),N,2,2)
        for itr in range(len(X)):
            for jtr in range(N):
                if not self.optimize_u:
                    u[jtr][itr]=np.sum(Vinv[jtr,:]*self._pbasis5([X[itr],Y[itr]]))
                if not self.optimize_du:
                    du[jtr][0][itr]=np.sum(Vinv[jtr,:]*self._pbasis5dx([X[itr],Y[itr]]))
                    du[jtr][1][itr]=np.sum(Vinv[jtr,:]*self._pbasis5dy([X[itr],Y[itr]]))
                if not self.optimize_ddu:
                    ddu[jtr][0][0][itr]=np.sum(Vinv[jtr,:]*self._pbasis5dxx([X[itr],Y[itr]]))
                    ddu[jtr][0][1][itr]=np.sum(Vinv[jtr,:]*self._pbasis5dxy([X[itr],Y[itr]]))
                    ddu[jtr][1][0][itr]=ddu[jtr][0][1][itr]
                    ddu[jtr][1][1][itr]=np.sum(Vinv[jtr,:]*self._pbasis5dyy([X[itr],Y[itr]]))

        return u,du,ddu

class ElementGlobalMorley(ElementGlobal):
    """Morley element for fourth-order problems."""

    n_dofs=1
    f_dofs=1
    dim=2
    maxdeg=2

    def gbasis(self,mesh,qps,k):
        X=qps[0][k,:]
        Y=qps[1][k,:]
        # solve local basis functions
        V=np.zeros((6,6))

        v1=mesh.p[:,mesh.t[0,k]]
        v2=mesh.p[:,mesh.t[1,k]]
        v3=mesh.p[:,mesh.t[2,k]]

        e1=0.5*(v1+v2)
        e2=0.5*(v2+v3)
        e3=0.5*(v1+v3)

        t1=v1-v2
        t2=v2-v3
        t3=v1-v3

        n1=np.array([t1[1],-t1[0]])
        n2=np.array([t2[1],-t2[0]])
        n3=np.array([t3[1],-t3[0]])

        n1/=np.linalg.norm(n1)
        n2/=np.linalg.norm(n2)
        n3/=np.linalg.norm(n3)

        # evaluate dofs
        V[0,:]=self._pbasis2(v1)
        V[1,:]=self._pbasis2(v2)
        V[2,:]=self._pbasis2(v3)
        V[3,:]=self._pbasis2dx(e1)*n1[0]+\
               self._pbasis2dy(e1)*n1[1]
        V[4,:]=self._pbasis2dx(e2)*n2[0]+\
               self._pbasis2dy(e2)*n2[1]
        V[5,:]=self._pbasis2dx(e3)*n3[0]+\
               self._pbasis2dy(e3)*n3[1]

        Vinv=np.linalg.inv(V).T

        dxx=self._pbasis2dxx()
        dxy=self._pbasis2dxy()
        dyy=self._pbasis2dyy()

        u=const_cell(np.zeros(len(X)),6)
        du=const_cell(np.zeros(len(X)),6,2)
        ddu=const_cell(np.zeros(len(X)),6,2,2)
        for itr in range(len(X)):
            for jtr in range(6):
                u[jtr][itr]=np.sum(Vinv[jtr,:]*self._pbasis2([X[itr],Y[itr]]))
                du[jtr][0][itr]=np.sum(Vinv[jtr,:]*self._pbasis2dx([X[itr],Y[itr]]))
                du[jtr][1][itr]=np.sum(Vinv[jtr,:]*self._pbasis2dy([X[itr],Y[itr]]))
                ddu[jtr][0][0][itr]=2.0*Vinv[jtr,3]
                ddu[jtr][0][1][itr]=Vinv[jtr,4]
                ddu[jtr][1][0][itr]=Vinv[jtr,4]
                ddu[jtr][1][1][itr]=2.0*Vinv[jtr,5]

        return u,du,ddu

class ElementGlobalTriP2(ElementGlobal):
    """Second-order triangular elements, globally
    defined version. This should only be used for debugging purposes.
    Use :class:`ElementTriP2` instead."""

    n_dofs=1
    f_dofs=1
    dim=2
    maxdeg=2

    def gbasis(self,mesh,qps,k):
        X=qps[0][k,:]
        Y=qps[1][k,:]
        # solve local basis functions
        V=np.zeros((6,6))

        v1=mesh.p[:,mesh.t[0,k]]
        v2=mesh.p[:,mesh.t[1,k]]
        v3=mesh.p[:,mesh.t[2,k]]

        e1=0.5*(v1+v2)
        e2=0.5*(v2+v3)
        e3=0.5*(v1+v3)

        # evaluate dofs
        V[0,:]=self._pbasis2(v1)
        V[1,:]=self._pbasis2(v2)
        V[2,:]=self._pbasis2(v3)
        V[3,:]=self._pbasis2(e1)
        V[4,:]=self._pbasis2(e2)
        V[5,:]=self._pbasis2(e3)

        Vinv=np.linalg.inv(V).T

        u=const_cell(np.zeros(len(X)),6)
        du=const_cell(np.zeros(len(X)),6,2)
        ddu=u # TODO
        for itr in range(len(X)):
            for jtr in range(6):
                u[jtr][itr]=np.sum(Vinv[jtr,:]*self._pbasis2([X[itr],Y[itr]]))
                du[jtr][0][itr]=np.sum(Vinv[jtr,:]*self._pbasis2dx([X[itr],Y[itr]]))
                du[jtr][1][itr]=np.sum(Vinv[jtr,:]*self._pbasis2dy([X[itr],Y[itr]]))

        return u,du,ddu

class ElementGlobalTriP1(ElementGlobal):
    """The simplest possible globally defined elements.
    This should only be used for debugging purposes.
    Use :class:`ElementTriP1` instead."""

    n_dofs=1
    dim=2
    maxdeg=1

    def gbasis(self,mesh,qps,k):
        X=qps[0][k,:]
        Y=qps[1][k,:]
        # solve local basis functions
        V=np.zeros((3,3))

        v1=mesh.p[:,mesh.t[0,k]]
        v2=mesh.p[:,mesh.t[1,k]]
        v3=mesh.p[:,mesh.t[2,k]]

        # evaluate dofs
        V[0,:]=self._pbasis1(v1)
        V[1,:]=self._pbasis1(v2)
        V[2,:]=self._pbasis1(v3)

        Vinv=np.linalg.inv(V).T

        u=const_cell(np.zeros(len(X)),3)
        du=const_cell(np.zeros(len(X)),3,2)
        ddu=const_cell(np.zeros(len(X)),3,2,2)
        for itr in range(len(X)):
            u[0][itr]=np.sum(Vinv[0,:]*self._pbasis1([X[itr],Y[itr]]))
            u[1][itr]=np.sum(Vinv[1,:]*self._pbasis1([X[itr],Y[itr]]))
            u[2][itr]=np.sum(Vinv[2,:]*self._pbasis1([X[itr],Y[itr]]))

            du[0][0][itr]=Vinv[0,1]
            du[1][0][itr]=Vinv[1,1]
            du[2][0][itr]=Vinv[2,1]

            du[0][1][itr]=Vinv[0,2]
            du[1][1][itr]=Vinv[1,2]
            du[2][1][itr]=Vinv[2,2]

        # output u[i] for each i must be array of size Nqp
        return u,du,ddu


class ElementHdiv(Element):
    """Abstract :math:`H_{div}` conforming finite element."""

    def gbasis(self,mapping,X,i,tind):
        if isinstance(X,dict):
            raise NotImplementedError("Calling ElementHdiv gbasis with dict not implemented!")
        else:
            x={}
            x[0]=X[0,:]
            x[1]=X[1,:]
            [phi,dphi]=self.lbasis(x,i)

        DF=mapping.DF(X,tind)
        detDF=mapping.detDF(X,tind)

        u={}
        u[0]=(DF[0][0]*phi[0]+DF[0][1]*phi[1])/detDF
        u[1]=(DF[1][0]*phi[0]+DF[1][1]*phi[1])/detDF

        du=dphi/detDF

        ddu=None

        return u,du,ddu

class ElementTriRT0(ElementHdiv):
    """Lowest order Raviart-Thomas element for triangle."""

    maxdeg=1
    f_dofs=1
    dim=2

    def lbasis(self,X,i):
        phi={}
        phi[0]={
                0:lambda x,y: x,
                1:lambda x,y: x,
                2:lambda x,y: -x+1.,
                }[i](X[0],X[1])
        phi[1]={
                0:lambda x,y: y-1.,
                1:lambda x,y: y,
                2:lambda x,y: -y,
                }[i](X[0],X[1])
        dphi={
            0:lambda x,y: 2+0.*x,
            1:lambda x,y: 2+0.*x,
            2:lambda x,y: -2+0.*x,
            }[i](X[0],X[1])

        return phi,dphi

class ElementH1(Element):
    """Abstract :math:`H^1` conforming finite element."""

    def gbasis(self,mapping,X,i,tind):
        if isinstance(X,dict):
            [phi,dphi]=self.lbasis(X,i)
            u=phi
            du={}
        else:
            x={}
            x[0]=X[0,:]
            if mapping.dim>=2:
                x[1]=X[1,:]
            if mapping.dim>=3:
                x[2]=X[2,:]
            [phi,dphi]=self.lbasis(x,i)
            u=np.tile(phi,(len(tind),1))
            du={}
            
        invDF=mapping.invDF(X,tind) # investigate if 'x' should used after else

        self.dim=mapping.dim

        if mapping.dim==1:
            du=np.outer(invDF,dphi)
        elif mapping.dim==2:
            du[0]=invDF[0][0]*dphi[0]+invDF[1][0]*dphi[1]
            du[1]=invDF[0][1]*dphi[0]+invDF[1][1]*dphi[1]
        elif mapping.dim==3:
            du[0]=invDF[0][0]*dphi[0]+invDF[1][0]*dphi[1]+invDF[2][0]*dphi[2]
            du[1]=invDF[0][1]*dphi[0]+invDF[1][1]*dphi[1]+invDF[2][1]*dphi[2]
            du[2]=invDF[0][2]*dphi[0]+invDF[1][2]*dphi[1]+invDF[2][2]*dphi[2]
        else:
            raise NotImplementedError("ElementH1.gbasis: not implemented for the given dim.")
        ddu=None # TODO fix ddu (for H1 element, Laplacian?)

        return u,du,ddu

class ElementH1Vec(ElementH1):
    """Convert :math:`H^1` element to vectorial :math:`H^1` element."""
    def __init__(self,elem):
        if elem.dim==0:
            print "ElementH1Vec.__init__(): Warning! Parent element has no dim-variable!"
        self.dim=elem.dim
        self.elem=elem
        # multiplicate the amount of DOF's with dim
        self.n_dofs=self.elem.n_dofs*self.dim
        self.f_dofs=self.elem.f_dofs*self.dim
        self.i_dofs=self.elem.i_dofs*self.dim
        self.e_dofs=self.elem.e_dofs*self.dim
        self.maxdeg=elem.maxdeg

    def gbasis(self,mapping,X,i,tind):
        ind=np.floor(float(i)/float(self.dim))
        n=i-self.dim*ind

        u={}
        du={}

        if isinstance(X,dict):
            [phi,dphi]=self.elem.lbasis(X,ind)
        else:
            x={}
            x[0]=X[0,:]
            x[1]=X[1,:]
            if mapping.dim>=3:
                x[2]=X[2,:]
            [phi,dphi]=self.elem.lbasis(x,ind)
            phi=np.tile(phi,(len(tind),1))
            for itr in range(self.dim):
                dphi[itr]=np.tile(dphi[itr],(len(tind),1))

        # fill appropriate slots of u and du (u[0] -> x-component of u etc.)
        for itr in range(self.dim):
            if itr==n:
                u[itr]=phi
                invDF=mapping.invDF(X,tind) # investigate if 'x' should used after else
                du[itr]={}
                if mapping.dim==2:
                    du[itr][0]=invDF[0][0]*dphi[0]+invDF[1][0]*dphi[1]
                    du[itr][1]=invDF[0][1]*dphi[0]+invDF[1][1]*dphi[1]
                elif mapping.dim==3:
                    du[itr][0]=invDF[0][0]*dphi[0]+invDF[1][0]*dphi[1]+invDF[2][0]*dphi[2]
                    du[itr][1]=invDF[0][1]*dphi[0]+invDF[1][1]*dphi[1]+invDF[2][1]*dphi[2]
                    du[itr][2]=invDF[0][2]*dphi[0]+invDF[1][2]*dphi[1]+invDF[2][2]*dphi[2]
                else:
                    raise NotImplementedError("ElementH1Vec.gbasis: not implemented for the given dim.")
            else:
                u[itr]=0*phi
                du[itr]={}
                for jtr in range(self.dim):
                    du[itr][jtr]=0*phi
            
        ddu=None
        return u,du,ddu
        
class ElementQ1(ElementH1):
    """Simplest quadrilateral element."""
    
    maxdeg=2
    n_dofs=1
    dim=2
        
    def lbasis(self,X,i):
        phi={
            0:lambda x,y: 0.25*(1-x)*(1-y),
            1:lambda x,y: 0.25*(1+x)*(1-y),
            2:lambda x,y: 0.25*(1+x)*(1+y),
            3:lambda x,y: 0.25*(1-x)*(1+y)
            }[i](X[0],X[1])
        dphi={}
        dphi[0]={
            0:lambda x,y: 0.25*(-1+y),
            1:lambda x,y: 0.25*(1-y),
            2:lambda x,y: 0.25*(1+y),
            3:lambda x,y: 0.25*(-1-y)
            }[i](X[0],X[1])
        dphi[1]={
            0:lambda x,y: 0.25*(-1+x),
            1:lambda x,y: 0.25*(-1-x),
            2:lambda x,y: 0.25*(1+x),
            3:lambda x,y: 0.25*(1-x)
            }[i](X[0],X[1])
        return phi,dphi
        
class ElementQ2(ElementH1):
    """Second order quadrilateral element."""

    maxdeg=3
    n_dofs=1
    f_dofs=1
    i_dofs=1
    dim=2
        
    def lbasis(self,X,i):
        phi={
            0:lambda x,y: 0.25*(x**2-x)*(y**2-y),
            1:lambda x,y: 0.25*(x**2+x)*(y**2-y),
            2:lambda x,y: 0.25*(x**2+x)*(y**2+y),
            3:lambda x,y: 0.25*(x**2-x)*(y**2+y),
            4:lambda x,y: 0.5*(y**2-y)*(1-x**2),
            5:lambda x,y: 0.5*(x**2+x)*(1-y**2),
            6:lambda x,y: 0.5*(y**2+y)*(1-x**2),
            7:lambda x,y: 0.5*(x**2-x)*(1-y**2),
            8:lambda x,y: (1-x**2)*(1-y**2)
            }[i](X[0],X[1])
        dphi={}
        dphi[0]={
            0:lambda x,y:((-1 + 2*x)*(-1 + y)*y)/4.,
            1:lambda x,y:((1 + 2*x)*(-1 + y)*y)/4.,
            2:lambda x,y:((1 + 2*x)*y*(1 + y))/4.,
            3:lambda x,y:((-1 + 2*x)*y*(1 + y))/4.,
            4:lambda x,y:-(x*(-1 + y)*y),
            5:lambda x,y:-((1 + 2*x)*(-1 + y**2))/2.,
            6:lambda x,y:-(x*y*(1 + y)),
            7:lambda x,y:-((-1 + 2*x)*(-1 + y**2))/2.,
            8:lambda x,y:2*x*(-1 + y**2)
            }[i](X[0],X[1])
        dphi[1]={
            0:lambda x,y:((-1 + x)*x*(-1 + 2*y))/4.,
            1:lambda x,y:(x*(1 + x)*(-1 + 2*y))/4.,
            2:lambda x,y:(x*(1 + x)*(1 + 2*y))/4.,
            3:lambda x,y:((-1 + x)*x*(1 + 2*y))/4.,
            4:lambda x,y:-((-1 + x**2)*(-1 + 2*y))/2.,
            5:lambda x,y:-(x*(1 + x)*y),
            6:lambda x,y:-((-1 + x**2)*(1 + 2*y))/2.,
            7:lambda x,y:-((-1 + x)*x*y),
            8:lambda x,y:2*(-1 + x**2)*y
            }[i](X[0],X[1])
        return phi,dphi

class ElementTriPp(ElementH1):
    """A somewhat slow implementation of hierarchical
    p-basis for triangular mesh."""

    dim=2

    def __init__(self,p):
        self.p=p
        self.maxdeg=p
        self.n_dofs=1
        self.f_dofs=np.max([p-1,0])
        self.i_dofs=np.max([(p-1)*(p-2)/2,0])

        self.nbdofs=3*self.n_dofs+3*self.f_dofs+self.i_dofs

    def intlegpoly(self,x,n):
        # Generate integrated Legendre polynomials.
        n=n+1
        
        P={}
        P[0]=np.ones(x.shape)
        if n>1:
            P[1]=x
        
        for i in np.arange(1,n):
            P[i+1]=((2.*i+1.)/(i+1.))*x*P[i]-(i/(i+1.))*P[i-1]
            
        iP={}
        iP[0]=np.ones(x.shape)
        if n>1:
            iP[1]=x
            
        for i in np.arange(1,n-1):
            iP[i+1]=(P[i+1]-P[i-1])/(2.*i+1.)
            
        dP={}
        dP[0]=np.zeros(x.shape)
        for i in np.arange(1,n):
            dP[i]=P[i-1]
        
        return iP,dP
        
    def lbasis(self,X,n):
        # Evaluate n'th Lagrange basis of order self.p.
        p=self.p

        if len(X)!=2:
            raise NotImplementedError("ElementTriPp: not implemented for the given dimension of X.")

        phi={}
        phi[0]=1.-X[0]-X[1]
        phi[1]=X[0]
        phi[2]=X[1]
        
        # local basis function gradients TODO fix these somehow
        gradphi_x={}
        gradphi_x[0]=-1.*np.ones(X[0].shape)
        gradphi_x[1]=1.*np.ones(X[0].shape)
        gradphi_x[2]=np.zeros(X[0].shape)
        
        gradphi_y={}
        gradphi_y[0]=-1.*np.ones(X[0].shape)
        gradphi_y[1]=np.zeros(X[0].shape)
        gradphi_y[2]=1.*np.ones(X[0].shape)

        if n<=2:
            # return first three
            dphi={}
            dphi[0]=gradphi_x[n]
            dphi[1]=gradphi_y[n]
            return phi[n],dphi

        # use same ordering as in mesh
        e=np.array([[0,1],[1,2],[0,2]]).T
        offset=3
        
        # define edge basis functions
        if(p>1):
            for i in range(3):
                eta=phi[e[1,i]]-phi[e[0,i]]
                deta_x=gradphi_x[e[1,i]]-gradphi_x[e[0,i]]
                deta_y=gradphi_y[e[1,i]]-gradphi_y[e[0,i]]
                
                # generate integrated Legendre polynomials
                [P,dP]=self.intlegpoly(eta,p-2)
                
                for j in range(len(P)):
                    phi[offset]=phi[e[0,i]]*phi[e[1,i]]*P[j]
                    gradphi_x[offset]=gradphi_x[e[0,i]]*phi[e[1,i]]*P[j]+\
                                      gradphi_x[e[1,i]]*phi[e[0,i]]*P[j]+\
                                      deta_x*phi[e[0,i]]*phi[e[1,i]]*dP[j]
                    gradphi_y[offset]=gradphi_y[e[0,i]]*phi[e[1,i]]*P[j]+\
                                      gradphi_y[e[1,i]]*phi[e[0,i]]*P[j]+\
                                      deta_y*phi[e[0,i]]*phi[e[1,i]]*dP[j]
                    if offset==n:
                        # return if computed
                        dphi={}
                        dphi[0]=gradphi_x[n]
                        dphi[1]=gradphi_y[n]
                        return phi[n],dphi
                    offset=offset+1  
        
        # define interior basis functions
        if(p>2):
            B={}
            dB_x={}
            dB_y={}
            if(p>3):
                pm3=ElementTriPp(p-3)
                for itr in range(pm3.nbdofs):
                    pphi,pdphi=self.lbasis(X,itr)
                    B[itr]=pphi
                    dB_x[itr]=pdphi[0]
                    dB_y[itr]=pdphi[1]
                N=pm3.nbdofs
            else:
                B[0]=np.ones(X[0].shape)
                dB_x[0]=np.zeros(X[0].shape)
                dB_y[0]=np.zeros(X[0].shape)
                N=1
                
            bubble=phi[0]*phi[1]*phi[2]
            dbubble_x=gradphi_x[0]*phi[1]*phi[2]+\
                      gradphi_x[1]*phi[2]*phi[0]+\
                      gradphi_x[2]*phi[0]*phi[1]
            dbubble_y=gradphi_y[0]*phi[1]*phi[2]+\
                      gradphi_y[1]*phi[2]*phi[0]+\
                      gradphi_y[2]*phi[0]*phi[1]
            
            for i in range(N):
                phi[offset]=bubble*B[i]
                gradphi_x[offset]=dbubble_x*B[i]+dB_x[i]*bubble
                gradphi_y[offset]=dbubble_y*B[i]+dB_y[i]*bubble
                if offset==n:
                    # return if computed
                    dphi={}
                    dphi[0]=gradphi_x[n]
                    dphi[1]=gradphi_y[n]
                    return phi[n],dphi
                offset=offset+1

        raise IndexError("ElementTriPp.lbasis: reached end of lbasis without returning anything.")

class ElementTriDG(ElementH1):
    """Transform a H1 conforming triangular element
    into a discontinuous one by turning all DOFs into
    interior DOFs."""
    def __init__(self,elem):
        # change all dofs to interior dofs
        self.elem=elem
        self.maxdeg=elem.maxdeg
        self.i_dofs=3*elem.n_dofs+3*elem.f_dofs+elem.i_dofs
        self.dim=2
    def lbasis(self,X,i):
        return self.elem.lbasis(X,i)

class ElementTetDG(ElementH1):
    """Convert a H1 tetrahedral element into a DG element.
    All DOFs are converted to interior DOFs."""
    def __init__(self,elem):
        # change all dofs to interior dofs
        self.elem=elem
        self.maxdeg=elem.maxdeg
        self.i_dofs=4*elem.n_dofs+4*elem.f_dofs+6*elem.e_dofs+elem.i_dofs
        self.dim=3
    def lbasis(self,X,i):
        return self.elem.lbasis(X,i)
    
class ElementTetP0(ElementH1):
    """Piecewise constant element for tetrahedral mesh."""

    i_dofs=1
    maxdeg=1
    dim=3

    def lbasis(self,X,i):
        phi={
            0:lambda x,y,z: 1+0*x
            }[i](X[0],X[1],X[2])
        dphi={}
        dphi[0]={
                0:lambda x,y,z: 0*x
                }[i](X[0],X[1],X[2])
        dphi[1]={
                0:lambda x,y,z: 0*x
                }[i](X[0],X[1],X[2])
        dphi[2]={
                0:lambda x,y,z: 0*x
                }[i](X[0],X[1],X[2])
        return phi,dphi

class ElementTriP0(ElementH1):
    """Piecewise constant element for triangular mesh."""

    i_dofs=1
    maxdeg=1
    dim=2

    def lbasis(self,X,i):
        phi={
            0:lambda x,y: 1+0*x
            }[i](X[0],X[1])
        dphi={}
        dphi[0]={
                0:lambda x,y: 0*x
                }[i](X[0],X[1])
        dphi[1]={
                0:lambda x,y: 0*x
                }[i](X[0],X[1])
        return phi,dphi

# this is for legacy
class ElementP0(ElementTriP0):
    pass

class ElementTriMini(ElementH1):
    """The MINI-element for triangular mesh."""

    dim=2
    n_dofs=1
    i_dofs=1
    maxdeg=3

    def lbasis(self,X,i):
        phi={
            0:lambda x,y: 1-x-y,
            1:lambda x,y: x,
            2:lambda x,y: y,
            3:lambda x,y: (1-x-y)*x*y
            }[i](X[0],X[1])

        dphi={}
        dphi[0]={
                0:lambda x,y: -1+0*x,
                1:lambda x,y: 1+0*x,
                2:lambda x,y: 0*x,
                3:lambda x,y: (1-x-y)*y-x*y
                }[i](X[0],X[1])
        dphi[1]={
                0:lambda x,y: -1+0*x,
                1:lambda x,y: 0*x,
                2:lambda x,y: 1+0*x,
                3:lambda x,y: (1-x-y)*x-x*y
                }[i](X[0],X[1])
        return phi,dphi
        
class ElementTetP2(ElementH1):
    """The quadratic tetrahedral element."""
    
    dim=3
    n_dofs=1
    e_dofs=1
    maxdeg=2
    
    def lbasis(self,X,i):
        
        phi={ # order (0,0,0) (1,0,0) (0,1,0) (0,0,1) and then according to mesh local t2e
            0:lambda x,y,z: 1. - 3.*x + 2.*x**2 - 3.*y + 4.*x*y + 2.*y**2 - 3.*z + 4.*x*z + 4.*y*z + 2.*z**2,
            1:lambda x,y,z: 0. - 1.*x + 2.*x**2,
            2:lambda x,y,z: 0. - 1.*y + 2.*y**2,
            3:lambda x,y,z: 0. - 1.*z + 2.*z**2,
            4:lambda x,y,z: 0. + 4.*x - 4.*x**2 - 4.*x*y - 4.*x*z,
            5:lambda x,y,z: 0. + 4.*x*y,
            6:lambda x,y,z: 0. + 4.*y - 4.*x*y - 4.*y**2 - 4.*y*z,
            7:lambda x,y,z: 0. + 4.*z - 4.*x*z - 4.*y*z - 4.*z**2,
            8:lambda x,y,z: 0. + 4.*x*z,
            9:lambda x,y,z: 0. + 4.*y*z,
            }[i](X[0],X[1],X[2])
    
        dphi={}
        dphi[0]={
                0:lambda x,y,z: -3. + 4.*x + 4.*y + 4.*z,
                1:lambda x,y,z: -1. + 4.*x,
                2:lambda x,y,z: 0.*x,
                3:lambda x,y,z: 0.*x,
                4:lambda x,y,z: 4. - 8.*x - 4.*y - 4.*z,
                5:lambda x,y,z: 4.*y,
                6:lambda x,y,z: -4.*y,
                7:lambda x,y,z: -4.*z,
                8:lambda x,y,z: 4.*z,
                9:lambda x,y,z: 0.*x,
                }[i](X[0],X[1],X[2])
        dphi[1]={
                0:lambda x,y,z: -3. + 4.*x + 4.*y + 4.*z,
                1:lambda x,y,z: 0.*x,
                2:lambda x,y,z: -1. + 4.*y,
                3:lambda x,y,z: 0.*x,
                4:lambda x,y,z: -4.*x,
                5:lambda x,y,z: 4.*x,
                6:lambda x,y,z: 4. - 4.*x - 8.*y - 4.*z,
                7:lambda x,y,z: -4.*z,
                8:lambda x,y,z: 0.*x,
                9:lambda x,y,z: 4.*z,
                }[i](X[0],X[1],X[2])
        dphi[2]={
                0:lambda x,y,z: -3. + 4.*x + 4.*y + 4.*z,
                1:lambda x,y,z: 0.*x,
                2:lambda x,y,z: 0.*x,
                3:lambda x,y,z: -1. + 4.*z,
                4:lambda x,y,z: -4.*x,
                5:lambda x,y,z: 0.*x,
                6:lambda x,y,z: -4.*y,
                7:lambda x,y,z: 4. - 4.*x - 4.*y - 8.*z,
                8:lambda x,y,z: 4.*x,
                9:lambda x,y,z: 4.*y,       
                }[i](X[0],X[1],X[2])
                
        return phi,dphi
        
class ElementLineP1(ElementH1):
    """Linear element for one dimension."""

    n_dofs=1
    dim=1
    maxdeg=1
    
    def lbasis(self,X,i):
        phi={
            0:lambda x: 1-x,
            1:lambda x: x,
            }[i](X[0])

        dphi={}
        dphi={
                0:lambda x: -1+0*x,
                1:lambda x: 1+0*x,
                }[i](X[0])
                
        return phi,dphi        
        
class ElementTriP1(ElementH1):
    """The simplest triangular element."""

    n_dofs=1
    dim=2
    maxdeg=1
    
    def lbasis(self,X,i):
        phi={
            0:lambda x,y: 1-x-y,
            1:lambda x,y: x,
            2:lambda x,y: y
            }[i](X[0],X[1])

        dphi={}
        dphi[0]={
                0:lambda x,y: -1+0*x,
                1:lambda x,y: 1+0*x,
                2:lambda x,y: 0*x
                }[i](X[0],X[1])
        dphi[1]={
                0:lambda x,y: -1+0*x,
                1:lambda x,y: 0*x,
                2:lambda x,y: 1+0*x
                }[i](X[0],X[1])
                
        return phi,dphi

class ElementTriP2(ElementH1):
    """The quadratic triangular element."""

    n_dofs=1
    f_dofs=1
    dim=2
    maxdeg=2
    
    def lbasis(self,X,i):
        phi={
            0:lambda x,y: 1-3*x-3*y+2*x**2+4*x*y+2*y**2,
            1:lambda x,y: 2*x**2-x,
            2:lambda x,y: 2*y**2-y,
            3:lambda x,y: 4*x-4*x**2-4*x*y,
            4:lambda x,y: 4*x*y,
            5:lambda x,y: 4*y-4*x*y-4*y**2,
            }[i](X[0],X[1])

        dphi={}
        dphi[0]={
                0:lambda x,y: -3+4*x+4*y,
                1:lambda x,y: 4*x-1,
                2:lambda x,y: 0*x,
                3:lambda x,y: 4-8*x-4*y,
                4:lambda x,y: 4*y,
                5:lambda x,y: -4*y,
                }[i](X[0],X[1])
        dphi[1]={
                0:lambda x,y: -3+4*x+4*y,
                1:lambda x,y: 0*x,
                2:lambda x,y: 4*y-1,
                3:lambda x,y: -4*x,
                4:lambda x,y: 4*x,
                5:lambda x,y: 4-4*x-8*y,
                }[i](X[0],X[1])
                
        return phi,dphi
        
class ElementTetP1(ElementH1):
    """The simplest tetrahedral element."""
    
    n_dofs=1
    maxdeg=1
    dim=3

    def lbasis(self,X,i):

        phi={
            0:lambda x,y,z: 1-x-y-z,
            1:lambda x,y,z: x,
            2:lambda x,y,z: y,
            3:lambda x,y,z: z,
            }[i](X[0],X[1],X[2])

        dphi={}
        dphi[0]={
                0:lambda x,y,z: -1+0*x,
                1:lambda x,y,z: 1+0*x,
                2:lambda x,y,z: 0*x,
                3:lambda x,y,z: 0*x
                }[i](X[0],X[1],X[2])
        dphi[1]={
                0:lambda x,y,z: -1+0*x,
                1:lambda x,y,z: 0*x,
                2:lambda x,y,z: 1+0*x,
                3:lambda x,y,z: 0*x
                }[i](X[0],X[1],X[2])
        dphi[2]={
                0:lambda x,y,z: -1+0*x,
                1:lambda x,y,z: 0*x,
                2:lambda x,y,z: 0*x,
                3:lambda x,y,z: 1+0*x
                }[i](X[0],X[1],X[2])

        return phi,dphi

