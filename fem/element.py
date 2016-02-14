import numpy as np

class Element:
    """Finite element."""

    maxdeg=0 # maximum polynomial degree; for determining quadrature
    dim=0 # spatial dimension
    tdim=0 # target dimension
    torder=0 # target tensorial order

    """
    The discretized field is a mapping
       U : C^dim -> C^(tdim x ... x tdim)
    where the product is taken 'torder' times.
    """

    # number of ...
    n_dofs=0 # nodal dofs
    i_dofs=0 # interior dofs
    f_dofs=0 # facet dofs (2d and 3d only)
    e_dofs=0 # edge dofs (3d only)

    def lbasis(self,X,i):
        """Returns local basis functions evaluated at some local points."""
        raise NotImplementedError("Element.lbasis: local basis (lbasis) not implemented!")

    def gbasis(self,X,i,tind):
        """Returns global basis functions evaluated at some local points."""
        raise NotImplementedError("Element.gbasis: local basis (lbasis) not implemented!")

class ElementH1(Element):
    """H1 conforming finite element."""

    def gbasis(self,mapping,X,i,tind):
        [phi,dphi]=self.lbasis(X,i)
        u=np.tile(phi,(len(tind),1))
        du={}
        invDF=mapping.invDF(X,tind)
        
        if X.shape[0]==2:
            du[0]=invDF[0][0]*dphi[0]+invDF[1][0]*dphi[1]
            du[1]=invDF[0][1]*dphi[0]+invDF[1][1]*dphi[1]
        elif X.shape[0]==3:
            du[0]=invDF[0][0]*dphi[0]+invDF[1][0]*dphi[1]+invDF[2][0]*dphi[2]
            du[1]=invDF[0][1]*dphi[0]+invDF[1][1]*dphi[1]+invDF[2][1]*dphi[2]
            du[2]=invDF[0][2]*dphi[0]+invDF[1][2]*dphi[1]+invDF[2][2]*dphi[2]
        else:
            raise NotImplementedError("ElementH1.gbasis: not implemented for the given X.shape[0].")
        ddu=None

        return u,du,ddu

    def gbasis_facet(self,mapping,X,i,tind):
        [phi,dphi]=

class ElementP1(ElementH1):
    
    n_dofs=1
    maxdeg=1

    def lbasis(self,X,i):
        # TODO implement for higher dimension

        if X.shape[0]==2:
            phi={
                0:lambda x,y: 1-x-y,
                1:lambda x,y: x,
                2:lambda x,y: y
                }[i](X[0,:],X[1,:])
    
            dphi={}
            dphi[0]={
                    0:lambda x,y: -1+0*x,
                    1:lambda x,y: 1+0*x,
                    2:lambda x,y: 0*x
                    }[i](X[0,:],X[1,:])
            dphi[1]={
                    0:lambda x,y: -1+0*x,
                    1:lambda x,y: 0*x,
                    2:lambda x,y: 1+0*x
                    }[i](X[0,:],X[1,:])
        elif X.shape[0]==3:
            phi={
                0:lambda x,y,z: 1-x-y-z,
                1:lambda x,y,z: x,
                2:lambda x,y,z: y,
                3:lambda x,y,z: z,
                }[i](X[0,:],X[1,:],X[2,:])
    
            dphi={}
            dphi[0]={
                    0:lambda x,y,z: -1+0*x,
                    1:lambda x,y,z: 1+0*x,
                    2:lambda x,y,z: 0*x,
                    3:lambda x,y,z: 0*x
                    }[i](X[0,:],X[1,:],X[2,:])
            dphi[1]={
                    0:lambda x,y,z: -1+0*x,
                    1:lambda x,y,z: 0*x,
                    2:lambda x,y,z: 1+0*x,
                    3:lambda x,y,z: 0*x
                    }[i](X[0,:],X[1,:],X[2,:])
            dphi[2]={
                    0:lambda x,y,z: -1+0*x,
                    1:lambda x,y,z: 0*x,
                    2:lambda x,y,z: 0*x,
                    3:lambda x,y,z: 1+0*x
                    }[i](X[0,:],X[1,:],X[2,:])
        else:
            raise NotImplementedError("ElementP1.lbasis: not implemented for the given X.shape[0].")

        return phi,dphi


