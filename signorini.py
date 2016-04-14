import numpy as np
import fem.mesh as fmsh
import fem.asm as fasm
import fem.element as felem
import fem.solvers as fsol
import scipy.sparse as spsp
import copy
from fem.weakform import *
from scipy.sparse import diags

def create_beam(vol):
    # build using MeshPy
    from meshpy.tet import MeshInfo,build

    mesh_info = MeshInfo()

    mesh_info.set_points([
        (0,0,0),
        (0,1,0),
        (0,1,1),
        (0,0,1),
        (5,0,0),
        (5,1,0),
        (5,1,1),
        (5,0,1),
        ])

    mesh_info.set_facets([
        [0,1,2,3],
        [4,5,6,7],
        [0,1,5,4],
        [1,2,6,5],
        [0,3,7,4],
        [3,2,6,7],
        ])


    mesh = build(mesh_info,max_volume=vol)

    return fmsh.MeshPyTet(mesh)


U=TensorFunction(dim=3,torder=1)
V=TensorFunction(dim=3,torder=1,sym='v')

F=ConstantTensor(-1e-1,dim=3,torder=1)
F.expr[0]=0.0
F.expr[1]=0.0

def Eps(W):
    return 0.5*(grad(W)+grad(W).T())

E=20
Nu=0.3

Lambda=E*Nu/((1+Nu)*(1-2*Nu))
Mu=E/(2*(1+Nu))

def Sigma(W):
    return 2*Mu*Eps(W)+Lambda*div(W)*IdentityMatrix(3)

dudv=dotp(Sigma(U),Eps(V))
dudv=dudv.handlify(verbose=True)

fv=dotp(F,V)
fv=fv.handlify(verbose=True)

m=create_beam(0.005)

# interior element and assembler
e=felem.ElementH1Vec(felem.ElementTetP2())
a=fasm.AssemblerElement(m,e)

# assemble stiffness matrix and load vector
A=a.iasm(dudv)
f=a.iasm(fv)

# boundary element
eb=felem.ElementTetP0()
b=fasm.AssemblerElement(m,e,eb)
c=fasm.AssemblerElement(m,eb)

# define contact facets
contactfacets=m.facets_satisfying(lambda x,y,z:z==0.0)
contactelems=m.f2t[0,contactfacets]
noncontactelems=np.setdiff1d(np.arange(m.t.shape[1]),contactelems)

# assemble obstacle
def g(x):
    return 0.1*(4.9-x[0])
g=c.fasm(lambda x,v: g(x)*v,find=contactfacets)

# assemble saddle point matrices
def Buv(u,v,h,n):
    return v*(u[0]*n[0]+u[1]*n[1]+u[2]*n[2])
B=b.fasm(Buv,find=contactfacets)

# assemble stabilization matrices
N=TensorFunction(dim=3,torder=1,sym='n')
normaltraction=dotp(Sigma(U)*N,N).handlify(verbose=True,boundary=True)

def Buvstab(u,du,v,h,n):
    return h*normaltraction(u,None,du,None,None,n)*v
Bstab=b.fasm(Buvstab,find=contactfacets)

def Astab(u,du,v,dv,h,n):
    return h*normaltraction(u,None,du,None,None,n)*normaltraction(v,None,dv,None,None,n)
Astab=a.fasm(Astab,find=contactfacets)

def Cuv(u,v,h):
    return h*u*v
C=c.fasm(Cuv,find=contactfacets)

# compute the inverse of C
invC=diags(np.array([1.0/(C.todia().data[0])]),[0])
invC.data[invC.data==np.inf]=0
invC=invC.tocsr()

# find index sets of Dirichlet boundary and iterate
i1=m.nodes_satisfying(lambda x,y,z: x!=0.0)
i2=m.edges_satisfying(lambda x,y,z: x!=0.0)
I=a.dofnum_u.getdofs(N=i1,E=i2)

# initialize iteration
maxiters=10
u={}
L={}

if 0:
    # draw contact facets to verify
    indifun=np.zeros(m.facets.shape[1])
    indifun[contactfacets]=1.0
    m.draw_facets(u=indifun)
    raise Exception("!")

# stabilization parameter and resulting matrices
alpha=1e-1
A1=A+alpha*Astab
B1=B+alpha*Bstab
invC1=invC/alpha

u[0]=fsol.direct(A1,f,I=I,use_umfpack=True)
L[0]=invC1.dot(-g+B1.dot(u[0]))
L[0][noncontactelems]=0.0

prevact=[]
# primal-dual active set
for k in range(maxiters):
    # active set
    act=np.nonzero(L[k]>0)[0]
    if np.array_equal(np.sort(act),np.sort(prevact)):
        break
    # only bottom face
    act=np.intersect1d(contactelems,act)
    # inactive set
    nact=np.setdiff1d(np.arange(m.t.shape[1]),act)

    if 0:
        # draw active set for verification
        indifun=np.zeros(m.facets.shape[1])
        indifun[m.t2f[0,act]]=1.0
        indifun[m.t2f[1,act]]=1.0
        indifun[m.t2f[2,act]]=1.0
        indifun[m.t2f[3,act]]=1.0
        m.draw_facets(u=indifun)
        raise Exception("!")

    Bact=B1[act]
    invCact=invC1[act].T[act].T

    K=A1+Bact.T.dot(invCact.dot(Bact))
    F=f+Bact.T.dot(invCact.dot(g[act]))

    u[k+1]=fsol.direct(K,F,I=I,use_umfpack=True)
    L[k+1]=invC1.dot(-g+B1.dot(u[k+1]))
    L[k+1][nact]=0.0

    prevact=act

    print act

# displaced mesh for drawing
mdefo=copy.deepcopy(m)
sf=1.0

mdefo.p[0,:]+=sf*u[k][a.dofnum_u.n_dof[0,:]]
mdefo.p[1,:]+=sf*u[k][a.dofnum_u.n_dof[1,:]]
mdefo.p[2,:]+=sf*u[k][a.dofnum_u.n_dof[2,:]]

facetu=np.zeros(m.facets.shape[1])
facetu[contactfacets]=L[k][m.f2t[0,contactfacets]]
mdefo.draw_facets(u=facetu)

raise Exception("!")
# compute von mises stress
V=TensorFunction(dim=3,torder=0,sym='v')

# project von mises stress to scalar P1 element
e1=felem.ElementTetP1()
b=fasm.AssemblerElement(m,e,e1)
S=Sigma(U)
def vonmises(s):
    return np.sqrt(0.5*((s[0,0]-s[1,1])**2+(s[1,1]-s[2,2])**2+(s[2,2]-s[0,0])**2+6.0*(s[1,2]**2+s[2,0]**2+s[0,1]**2)))

c=fasm.AssemblerElement(m,e1)
M=c.iasm(lambda u,v:u*v)

StressTensor={}
for itr in range(3):
    for jtr in range(3):
        duv=(S[itr,jtr]*V).handlify(verbose=True)
        P=b.iasm(duv)
        StressTensor[(itr,jtr)]=fsol.direct(M,P*u[maxiters],use_umfpack=True)

mdefo.draw(u=vonmises(StressTensor))

