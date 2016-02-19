import unittest
import fem.asm
import fem.geometry
import fem.mesh as fmsh
import numpy as np
import scipy.sparse.linalg
import scipy.sparse as spsp
import fem.asm as fasm
import fem.mapping as fmap
import fem.element as felem
import matplotlib.pyplot as plt

class AssemblerElementTriPpTest(unittest.TestCase):
    """Test triangular h-refinement with various p.
    Also test facet assembly."""
    def runTest(self):

        def U(x):
            return 1+x[0]-x[0]**2*x[1]**2

        def dUdx(x):
            return 1-2*x[0]*x[1]**2

        def dUdy(x):
            return -2*x[0]**2*x[1]

        def dudv(du,dv):
            return du[0]*dv[0]+du[1]*dv[1]

        def uv(u,v):
            return u*v

        def F(x,y):
            return 2*x**2+2*y**2

        def fv(v,x):
            return F(x[0],x[1])*v

        def G(x,y):
            return (x==1)*(3-3*y**2)+\
                    (x==0)*(0)+\
                    (y==1)*(1+x-3*x**2)+\
                    (y==0)*(1+x)

        def gv(v,x):
            return G(x[0],x[1])*v

        dexact={}
        dexact[0]=dUdx
        dexact[1]=dUdy

        hs={}
        H1errs={}
        L2errs={}

        for p in range(1,4):
            mesh=fmsh.MeshTri()
            mesh.refine(2)
            hs[p-1]=np.array([])
            H1errs[p-1]=np.array([])
            L2errs[p-1]=np.array([])

            for itr in range(4):
                mesh.refine()

                a=fasm.AssemblerElement(mesh,fmap.MappingAffine,felem.ElementTriPp(p))

                A=a.iasm(dudv)
                f=a.iasm(fv)

                B=a.fasm(uv)
                g=a.fasm(gv)

                u=np.zeros(a.dofnum_u.N)
                u=scipy.sparse.linalg.spsolve(A+B,f+g)

                hs[p-1]=np.append(hs[p-1],mesh.param())
                L2errs[p-1]=np.append(L2errs[p-1],a.L2error(u,U))
                H1errs[p-1]=np.append(H1errs[p-1],a.H1error(u,dexact))

            pfit=np.polyfit(np.log10(hs[p-1]),np.log10(H1errs[p-1]),1)

            self.assertTrue(pfit[0]>=0.95*p)

class AssemblerElementTetP1Test(unittest.TestCase):
    """Test down tetrahedral refinements with P1 elements.
    Also tests assembly on tetrahedral facets."""
    # TODO make tutorial out of this
    def runTest(self):
        def U(x):
            return 1+x[0]-x[0]**2*x[1]**2+x[0]*x[1]*x[2]**3
            
        def dUdx(x):
            return 1-2*x[0]*x[1]**2+x[1]*x[2]**3
            
        def dUdy(x):
            return -2*x[0]**2*x[1]+x[0]*x[2]**3
            
        def dUdz(x):
            return 3*x[0]*x[1]*x[2]**2

        def dudv(du,dv):
            return du[0]*dv[0]+du[1]*dv[1]+du[2]*dv[2]

        def uv(u,v):
            return u*v

        def F(x,y,z):
            return 2*x**2+2*y**2-6*x*y*z

        def fv(v,x):
            return F(x[0],x[1],x[2])*v

        def G(x,y,z):
            return (x==1)*(3-3*y**2+2*y*z**3)+\
                   (x==0)*(-y*z**3)+\
                   (y==1)*(1+x-3*x**2+2*x*z**3)+\
                   (y==0)*(1+x-x*z**3)+\
                   (z==1)*(1+x+4*x*y-x**2*y**2)+\
                   (z==0)*(1+x-x**2*y**2)

        def gv(v,x):
            return G(x[0],x[1],x[2])*v

        dexact={}
        dexact[0]=dUdx
        dexact[1]=dUdy
        dexact[2]=dUdz

        hs=np.array([])
        H1err=np.array([])
        L2err=np.array([])

        for itr in range(2,5):
            mesh=fmsh.MeshTet()
            mesh.refine(itr)

            a=fasm.AssemblerElement(mesh,fmap.MappingAffine,felem.ElementP1(3))

            A=a.iasm(dudv)
            f=a.iasm(fv)

            B=a.fasm(uv)
            g=a.fasm(gv)

            u=np.zeros(a.dofnum_u.N)

            u=scipy.sparse.linalg.spsolve(A+B,f+g)

            p={}
            p[0]=mesh.p[0,:]
            p[1]=mesh.p[1,:]
            p[2]=mesh.p[2,:]

            hs=np.append(hs,mesh.param())
            L2err=np.append(L2err,a.L2error(u,U))
            H1err=np.append(H1err,a.H1error(u,dexact))

        pfit=np.polyfit(np.log10(hs),np.log10(np.sqrt(L2err**2+H1err**2)),1)

        # check that the convergence rate matches theory
        self.assertTrue(pfit[0]>=1)

class AssemblerTriP1BasicTest(unittest.TestCase):
    def setUp(self):
        self.mesh=fmsh.MeshTri()
        self.mesh.refine(5)

        # boundary and interior node sets
        D1=np.nonzero(self.mesh.p[0,:]==0)[0]
        D2=np.nonzero(self.mesh.p[1,:]==0)[0]
        D3=np.nonzero(self.mesh.p[0,:]==1)[0]
        D4=np.nonzero(self.mesh.p[1,:]==1)[0]

        D=np.union1d(D1,D2);
        D=np.union1d(D,D3);
        self.D=np.union1d(D,D4);

        self.I=np.setdiff1d(np.arange(0,self.mesh.p.shape[1]),self.D)

class AssemblerTriP1Poisson(AssemblerTriP1BasicTest):
    """Simple Poisson test.
    
    Solving $-\Delta u = 1$ in an unit square with $u=0$ on the boundary.
    """
    def runTest(self):
        bilin=lambda u,v,du,dv,x,h: du[0]*dv[0]+du[1]*dv[1]
        lin=lambda v,dv,x,h: 1*v

        a=fem.asm.AssemblerTriP1(self.mesh)

        A=a.iasm(bilin)
        f=a.iasm(lin)

        x=np.zeros(A.shape[0])
        I=self.I
        x[I]=scipy.sparse.linalg.spsolve(A[np.ix_(I,I)],f[I])

        self.assertAlmostEqual(np.max(x),0.073614737354524146)

class AssemblerTriP1AnalyticWithXY(AssemblerTriP1BasicTest):
    """Poisson test case with analytic solution.

    Loading f=sin(pi*x)*sin(pi*y) and u=0 on the boundary.
    """
    def runTest(self):
        I=self.I
        D=self.D

        a=fem.asm.AssemblerTriP1(self.mesh)

        def dudv(u,v,du,dv,x,h):
            return du[0]*dv[0]+du[1]*dv[1]
        K=a.iasm(dudv)

        def fv(v,dv,x,h):
                return 2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])*v
        f=a.iasm(fv)


        x=np.zeros(K.shape[0])
        x[I]=scipy.sparse.linalg.spsolve(K[np.ix_(I,I)],f[I])

        def truex():
            X=self.mesh.p[0,:]
            Y=self.mesh.p[1,:]
            return np.sin(np.pi*X)*np.sin(np.pi*Y)

        self.assertAlmostEqual(np.max(x-truex()),0.0,places=3)



class AssemblerTriP1FullPoisson(AssemblerTriP1BasicTest):
    """Poisson test from Huhtala's MATLAB package."""
    def runTest(self):
        F=lambda x,y: 100.0*((x>=0.4)&(x<=0.6)&(y>=0.4)&(y<=0.6))
        G=lambda x,y: (y==0)*1.0+(y==1)*(-1.0)

        a=fem.asm.AssemblerTriP1(self.mesh)

        dudv=lambda du,dv: du[0]*dv[0]+du[1]*dv[1]
        K=a.iasm(dudv)

        uv=lambda u,v: u*v
        B=a.fasm(uv)
        
        fv=lambda v,x: F(x[0],x[1])*v
        f=a.iasm(fv)

        gv=lambda v,x: G(x[0],x[1])*v
        g=a.fasm(gv)

        D=np.nonzero(self.mesh.p[0,:]==0)[0]
        I=np.setdiff1d(np.arange(0,self.mesh.p.shape[1]),D)

        x=np.zeros(K.shape[0])
        x[I]=scipy.sparse.linalg.spsolve(K[np.ix_(I,I)]+B[np.ix_(I,I)],f[I]+g[I])

        self.assertAlmostEqual(np.max(x),1.89635971369,places=2)

class AssemblerTriP1Nitsche(AssemblerTriP1BasicTest):
    """Solve Poisson with and without Nitsche approximating Dirichlet BC."""
    def runTest(self):
        mesh=self.mesh
        a=fem.asm.AssemblerTriP1(mesh)
        D=self.D
        I=self.I

        def dudv(u,v,du,dv,x,h):
            return du[0]*dv[0]+du[1]*dv[1]
            
        gamma=200
        def uv(u,v,du,dv,x,h,n):
            return gamma*1/h*u*v-du[0]*n[0]*v-du[1]*n[1]*v-u*dv[0]*n[0]-u*dv[1]*n[1]
            
        def fv(v,dv,x,h):
            return 2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])*v
            
        def G(x,y):
            return np.sin(np.pi*x)
            
        def gv(v,dv,x,h,n):
            return G(x[0],x[1])*v+gamma*1/h*G(x[0],x[1])*v-dv[0]*n[0]*G(x[0],x[1])-dv[1]*n[1]*G(x[0],x[1])

        K=a.iasm(dudv)
        B=a.fasm(uv)
        f=a.iasm(fv)
        g=a.fasm(gv)

        x=np.zeros(K.shape[0])
        x=scipy.sparse.linalg.spsolve(K+B,f+g)

        y=np.zeros(K.shape[0])
        y[D]=G(mesh.p[0,D],mesh.p[1,D])
        y[I]=scipy.sparse.linalg.spsolve(K[np.ix_(I,I)],f[I]-K[np.ix_(I,D)].dot(y[D]))

        self.assertAlmostEqual(np.linalg.norm(x-y),0.0,places=1)

class AssemblerTriP1Interp(AssemblerTriP1BasicTest):
    """Compare K*u with g(u) for some u, where K is the stiffness matrix and g(u)=<grad u,grad v>."""
    def runTest(self):
        mesh=self.mesh
        a=fem.asm.AssemblerTriP1(mesh)
        D=self.D
        I=self.I

        def G(x,y):
            return np.sin(np.pi*x)

        def test(v,dv,x,h,w1,dw1):
            return dw1[0]*dv[0]+dw1[1]*dv[1]
        def test2(u,v,du,dv,x,h,w1,dw1):
            return du[0]*dv[0]+du[1]*dv[1]

        u=G(mesh.p[0,:],mesh.p[1,:])
        g=a.iasm(test,w1=u)
        K=a.iasm(test2)
        
        self.assertAlmostEqual(np.linalg.norm(g-K*u),0.0,places=10)

class AssemblerTriP1NavierStokes(unittest.TestCase):
    """Solve Navier-Stokes equations (lid-driven cavity flow)
    with Brezzi-Pitkaranta elements and and compare the results to Ghia et al."""

    def runTest(self):
        g=fem.geometry.GeometryPSLG2D()
        g.add_rectangle()
        mesh=g.mesh(0.025)
        #mesh=fmsh.MeshTri()
        #mesh.refine()

        N=mesh.p.shape[1]
        # left side wall (minus upmost and lowermost nodes)
        Dleftx=np.nonzero(mesh.p[0,:]==0)[0]
        Dleftx=np.setdiff1d(Dleftx,np.nonzero(mesh.p[1,:]==0)[0])
        Dleftx=np.setdiff1d(Dleftx,np.nonzero(mesh.p[1,:]==1)[0])
        # right side wall (minus upmost and lowermost nodes)
        Drightx=np.nonzero(mesh.p[0,:]==1)[0]
        Drightx=np.setdiff1d(Drightx,np.nonzero(mesh.p[1,:]==0)[0])
        Drightx=np.setdiff1d(Drightx,np.nonzero(mesh.p[1,:]==1)[0])
        # lower and upper side wall 
        Dlowerx=np.nonzero(mesh.p[1,:]==0)[0]
        Dupperx=np.nonzero(mesh.p[1,:]==1)[0]
        # all dirichlet nodes 
        Dallx=mesh.boundary_nodes()
        Dally=Dallx+N

        D=np.union1d(Dallx,Dally)
        I=np.setdiff1d(np.arange(0,3*N),D)

        # index sets for accessing different components
        I1=np.arange(0,N)
        I2=I1+N
        Ip=I2+N

        a=fem.asm.AssemblerTriP1(mesh)
        mu=1

        def dudv11(du,dv):
            return mu*du[1]*dv[1]+2*mu*du[0]*dv[0]

        def dudv12(du,dv):
            return mu*du[0]*dv[1]

        def dudv21(du,dv):
            return mu*du[1]*dv[0]
            
        def dudv22(du,dv):
            return 2*mu*du[1]*dv[1]+mu*du[0]*dv[0]
            
        def duv1(u,dv):
            return u*dv[0]

        def duv2(u,dv):
            return u*dv[1]
            
        def uv(u,v):
            return u*v
            
        def dpdqstab(du,dv,h):
            return h**2*(du[0]*dv[0]+du[1]*dv[1])

        # assemble static matrices
        A11=a.iasm(dudv11)
        A12=a.iasm(dudv12)
        A21=a.iasm(dudv21)
        A22=a.iasm(dudv22)
        B1=a.iasm(duv1)
        B2=a.iasm(duv2)
        M=a.iasm(uv)

        # brezzi-pitkaranta stabilization matrix
        E=a.iasm(dpdqstab)

        # initialize solution
        u=np.zeros(3*N)

        Vel=100
        u[Dupperx]=Vel

        gamma=1e-1
        eps=1e-2 # works with zero when no-stress BC?

        A=spsp.vstack((spsp.hstack((A11,A12)),spsp.hstack((A21,A22))))
        B=spsp.vstack((B1,B2))
        K=spsp.vstack((spsp.hstack((A,-B)),spsp.hstack((-B.T,eps*M-gamma*E)))).tocsr()

        U=np.copy(u)
        # initial condition from Stokes
        u[I]=scipy.sparse.linalg.spsolve(K[np.ix_(I,I)],-K[np.ix_(I,D)].dot(u[D]),use_umfpack=True)

        # picard iteration
        for jtr in range(20):
            alpha=np.min((0.9,float(jtr)/10.+0.1))
            def duuv(w1,w2,du,v):
                return du[0]*w1*v+du[1]*w2*v
                
            C=a.iasm(duuv,w1=u[I1],w2=u[I2],intorder=3)
            A=spsp.vstack((spsp.hstack((A11+C,A12)),spsp.hstack((A21,A22+C))))
            B=spsp.vstack((B1,B2))
            K=spsp.vstack((spsp.hstack((A,-B)),spsp.hstack((-B.T,eps*M-gamma*E)))).tocsr()

            U=np.copy(u)
            # direct
            u[I]=scipy.sparse.linalg.spsolve(K[np.ix_(I,I)],-K[np.ix_(I,D)].dot(u[D]),use_umfpack=True)
            u[I]=alpha*u[I]+(1-alpha)*U[I]

            residual=np.linalg.norm(u-U)
            if residual<=100:
                break
                 
        u1fun=mesh.interpolator(u[I1])
        u2fun=mesh.interpolator(u[I2])

        # simulated reference solutions from Ghia et al.
        ys=np.array([0,0.0547,0.0625,0.0703,0.1016,0.1719,0.2812,0.4531,0.5000,0.6172,0.7344,0.8516,0.9531,0.9609,0.9688,0.9766])
        soltruey=np.array([0,-0.0372,-0.0419,-0.0477,-0.0643,-0.1015,-0.1566,-0.2109,-0.2058,-0.1364,0.0033,0.2315,0.6872,0.7372,0.7887,0.8412])

        xs=np.array([0,0.0625,0.0703,0.0781,0.0938,0.1563,0.2266,0.2344,0.5,0.8047,0.8594,0.9063,0.9453,0.9531,0.9609,0.9688,1.0000])
        soltruex=np.array([0,0.0923,0.1009,0.1089,0.1232,0.1608,0.1751,0.1753,0.0545,-0.2453,-0.2245,-0.1691,-0.1031,-0.0886,-0.0739,-0.0591,0.0000])
        
        self.assertTrue(np.max(np.abs(soltruex-u2fun(xs,0*xs+0.5)/Vel))<=0.025)
        self.assertTrue(np.max(np.abs(soltruey-u1fun(0*ys+0.5,ys)/Vel))<=0.025)
