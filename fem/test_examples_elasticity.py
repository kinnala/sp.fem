import unittest
import numpy as np
import fem.mesh as fmsh
import fem.asm as fasm
import fem.element as felem
import fem.solvers as fsol
import scipy.sparse as spsp
import copy
from fem.weakform import *

class ExampleElasticity(unittest.TestCase):
    """Solving the linear elasticity equations (stress) in 3D box
    and comparing to a manufactured analytical solution."""
    def runTest(self,verbose=False):
        U=TensorFunction(dim=3,torder=1)
        V=TensorFunction(dim=3,torder=1,sym='v')

        # infinitesimal strain tensor
        def Eps(W):
            return 0.5*(grad(W)+grad(W).T())

        # material parameters: Young's modulus and Poisson ratio
        E=20
        Nu=0.3

        # Lame parameters
        Lambda=E*Nu/((1+Nu)*(1-2*Nu))
        Mu=E/(2*(1+Nu))

        # definition of the stress tensor
        def Sigma(W):
            return 2*Mu*Eps(W)+Lambda*div(W)*IdentityMatrix(3)

        # define the weak formulation
        dudv=dotp(Sigma(U),Eps(V))
        dudv=dudv.handlify(verbose=verbose)

        # generate a mesh in the box
        m=fmsh.MeshTet()
        m.refine(4)

        # define the vectorial element
        e=felem.ElementH1Vec(felem.ElementTetP1())
        # create a FE-assembler object
        a=fasm.AssemblerElement(m,e)
        # assemble the stiffness matrix
        A=a.iasm(dudv)

        # define analytical solution and compute loading corresponding to it
        def exact(x):
            return -0.1*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])*np.sin(np.pi*x[2])

        def loading():
            global exactvonmises
            import sympy as sp
            x,y,z=sp.symbols('x y z')
            Ut=ConstantTensor(0.0,dim=3,torder=1)
            Ut.expr[0]=0*x
            Ut.expr[1]=0*x
            Ut.expr[2]=0.1*sp.sin(sp.pi*x)*sp.sin(sp.pi*y)*sp.sin(sp.pi*z)

            return dotp(div(Sigma(Ut)),V).handlify(verbose=verbose)

        # assemble the load vector
        f=a.iasm(loading())

        i1=m.interior_nodes()
        I=a.dofnum_u.getdofs(N=i1)

        # solve the system
        u=fsol.direct(A,f,I=I,use_umfpack=True)

        e1=felem.ElementTetP1()
        c=fasm.AssemblerElement(m,e1)

        L2err=c.L2error(u[a.dofnum_u.n_dof[2,:]],exact)

        self.assertTrue(c.L2error(u[a.dofnum_u.n_dof[2,:]],exact)<=1e-3)

        if verbose:
            # displaced mesh for drawing
            mdefo=copy.deepcopy(m)
            mdefo.p[0,:]+=u[a.dofnum_u.n_dof[0,:]]
            mdefo.p[1,:]+=u[a.dofnum_u.n_dof[1,:]]
            mdefo.p[2,:]+=u[a.dofnum_u.n_dof[2,:]]

            # project von mises stress to scalar P1 element
            V=TensorFunction(dim=3,torder=0,sym='v')
            b=fasm.AssemblerElement(m,e,e1)

            S=Sigma(U)
            def vonmises(s):
                return np.sqrt(0.5*((s[0,0]-s[1,1])**2+(s[1,1]-s[2,2])**2+(s[2,2]-s[0,0])**2+\
                       6.0*(s[1,2]**2+s[2,0]**2+s[0,1]**2)))

            M=c.iasm(lambda u,v:u*v)

            # compute each component of stress tensor
            StressTensor={}
            for itr in range(3):
                for jtr in range(3):
                    duv=(S[itr,jtr]*V).handlify(verbose=True)
                    P=b.iasm(duv)
                    StressTensor[(itr,jtr)]=fsol.direct(M,P*u,use_umfpack=True)

            # draw the von Mises stress
            mdefo.draw(u=vonmises(StressTensor),test=lambda x,y,z: x>=0.5)
