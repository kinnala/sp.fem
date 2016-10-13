import numpy as np
import matplotlib.pyplot as plt
from spfem.mesh import MeshTri
from spfem.geometry import GeometryMeshPyTriangle
from spfem.element import ElementH1Vec,ElementTriP1
from spfem.asm import AssemblerElement
from spfem.utils import direct

scale=0.4

def contact_mesh(h,nref=0):
    points=[(-1.0,0.0),(-1.0,-1.0),(1.0,-1.0),(1.0,0.0), # lower block
            (-scale,0.0),(-scale,1.0),(scale,1.0),(scale,0.0)] # upper block
    facets=[(0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4)]
    g=GeometryMeshPyTriangle(points,facets)
    m=g.mesh(h)
    if nref>0:
        m.refine(nref)
    my=.3333*np.sum(m.p[1,m.t],axis=0)
    ixlower=np.nonzero(my<0.0)[0]
    ixupper=np.nonzero(my>0.0)[0]
    mupper=MeshTri(*m.remove_elements(ixlower))
    mlower=MeshTri(*m.remove_elements(ixupper))
    return mlower,mupper

m1,m2=contact_mesh(0.01)
m1.draw()
plt.hold('on')
m2.draw(nofig=True)

e1=ElementTriP1()
e=ElementH1Vec(e1)
a1=AssemblerElement(m1,e)
a2=AssemblerElement(m2,e)

E1=1.0
nu1=0.3

E2=0.5
nu2=0.3

mu1=E1/(2.0*(1.0+nu1))
lam1=nu1*E1/((1.0+nu1)*(1.0-2.0*nu1))

mu2=E2/(2.0*(1.0+nu2))
lam2=nu2*E2/((1.0+nu2)*(1.0-2.0*nu2))

def Eps(dU):
    return np.array([[dU[0][0],.5*(dU[0][1]+dU[1][0])],
                     [.5*(dU[1][0]+dU[0][1]),dU[1][1]]])
def ddot(T1,T2):
    return T1[0,0]*T2[0,0]+\
           T1[0,1]*T2[0,1]+\
           T1[1,0]*T2[1,0]+\
           T1[1,1]*T2[1,1]

def C1(T):
    trT=T[0,0]+T[1,1]
    return np.array([[2.*mu1*T[0,0]+lam1*trT,2.*mu1*T[0,1]],
                     [2.*mu1*T[1,0],2.*mu1*T[1,1]+lam1*trT]])

def C2(T):
    trT=T[0,0]+T[1,1]
    return np.array([[2.*mu2*T[0,0]+lam2*trT,2.*mu2*T[0,1]],
                     [2.*mu2*T[1,0],2.*mu2*T[1,1]+lam2*trT]])

def dudv1(du,dv):
    return ddot(C1(Eps(du)),Eps(dv))

def dudv2(du,dv):
    return ddot(C2(Eps(du)),Eps(dv))

K1=a1.iasm(dudv1)
K2=a2.iasm(dudv2)

B1low=m1.nodes_satisfying(lambda x,y:y==-1.0)
B1up=m1.nodes_satisfying(lambda x,y:(y==0.0)*(x<=scale)*(x>=-scale))
B2up=m2.nodes_satisfying(lambda x,y:y==1.0)
B2low=m2.nodes_satisfying(lambda x,y:y==0.0)

def glue_dofs(Ax,Ay,bx,by,ix,iy,Ix,Iy):
    import scipy.sparse as spsp

    indices1=np.arange(Ax.shape[0])
    indices2=np.arange(Ay.shape[0])

    ix1=np.setdiff1d(indices1,ix)
    ix2=ix

    iy1=np.setdiff1d(indices2,iy)
    iy2=iy

    # find indices of Ix and Iy in the new matrix
    zeros1=np.zeros(Ax.shape[0])
    zeros2=np.zeros(Ay.shape[0])
    zeros1[Ix]=1
    zeros2[Iy]=1
    zeros1=zeros1[ix1]
    zeros2=zeros2[iy1]
    new_Ix=np.nonzero(zeros1)[0]
    new_Iy=np.nonzero(zeros2)[0]

    # glue matrices
    Ax11=Ax[ix1].T[ix1].T
    Ax12=Ax[ix1].T[ix2].T
    Ax21=Ax[ix2].T[ix1].T
    Ax22=Ax[ix2].T[ix2].T

    Ay11=Ay[iy1].T[iy1].T
    Ay12=Ay[iy1].T[iy2].T
    Ay21=Ay[iy2].T[iy1].T
    Ay22=Ay[iy2].T[iy2].T

    Z1=spsp.csr_matrix((Ax11.shape[0],Ay11.shape[1]))
    Z2=spsp.csr_matrix((Ay11.shape[0],Ax11.shape[1]))

    Atmp1=spsp.hstack((Ax11,Z1,Ax12))
    Atmp2=spsp.hstack((Z2,Ay11,Ay12))
    Atmp3=spsp.hstack((Ax21,Ay21,Ax22+Ay22))

    def glue_unpack(x):
        X=np.zeros(Ax.shape[0])
        Y=np.zeros(Ay.shape[0])
        X[ix1]=x[np.arange(len(ix1))]
        Y[iy1]=x[len(ix1)+np.arange(len(iy1))]
        X[ix2]=x[len(ix1)+len(iy1)+np.arange(len(ix2))]
        Y[iy2]=x[len(ix1)+len(iy1)+np.arange(len(ix2))]
        return X,Y

    return spsp.vstack((Atmp1,Atmp2,Atmp3)).tocsr(),\
        np.concatenate((bx[ix1],by[iy1],bx[ix2]+by[iy2])),\
        np.concatenate((new_Ix,new_Iy+len(ix1))),\
        glue_unpack


displ=-0.2
x2=np.zeros(K2.shape[0])
x2[2*B2up+1]=displ

K,b,D,unpack=glue_dofs(K1,K2,
        np.zeros(K1.shape[0]),-K2.dot(x2),
        2*B1up+1,2*B2low+1,
        np.concatenate((2*B1low+1,2*B1low)),np.concatenate((2*B2up+1,2*B2up)))

x=np.zeros(K.shape[0])

I=np.setdiff1d(np.arange(K.shape[0]),D)
x=direct(K,b,I=I)

X,Y=unpack(x)
Y[2*B2up+1]=displ

import copy
m1defo=copy.deepcopy(m1)
m2defo=copy.deepcopy(m2)

m1defo.p[0,:]+=X[a1.dofnum_u.n_dof[0,:]]
m1defo.p[1,:]+=X[a1.dofnum_u.n_dof[1,:]]
m2defo.p[0,:]+=Y[a2.dofnum_u.n_dof[0,:]]
m2defo.p[1,:]+=Y[a2.dofnum_u.n_dof[1,:]]

m1defo.draw()
plt.hold('on')
m2defo.draw(nofig=True)

import spfem.utils as spu
Du1=spu.const_cell(0.0,2,2)
Du2=spu.const_cell(0.0,2,2)
Du1[0][0],Du1[0][1]=spu.gradient(X[a1.dofnum_u.n_dof[0,:]],m1)
Du1[1][0],Du1[1][1]=spu.gradient(X[a1.dofnum_u.n_dof[1,:]],m1)
Du2[0][0],Du2[0][1]=spu.gradient(Y[a2.dofnum_u.n_dof[0,:]],m2)
Du2[1][0],Du2[1][1]=spu.gradient(Y[a2.dofnum_u.n_dof[1,:]],m2)

S1=C1(Eps(Du1))
S2=C2(Eps(Du2))

smax=np.max((np.max(S1[1][1]),np.max(S1[1][1])))
smin=np.min((np.min(S2[1][1]),np.min(S2[1][1])))

m1.plot(C1(Eps(Du1))[1,1],zlim=(smin,smax))
plt.hold('on')
m2.plot(C2(Eps(Du2))[1,1],nofig=True,zlim=(smin,smax))

m1defo.plot(C1(Eps(Du1))[1,1],zlim=(smin,smax))
plt.hold('on')
m2defo.plot(C2(Eps(Du2))[1,1],nofig=True,zlim=(smin,smax))

m1.show()



