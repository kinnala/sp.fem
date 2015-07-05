import numpy as np
from scipy.sparse import coo_matrix


dunavant1 = np.array([
    3.333333333333333333333e-01,
    3.333333333333333333333e-01,
    5.000000000000000000000e-01])

dunavant2 = np.array([
    1.666666666666666666666e-01,
    6.666666666666666666666e-01,
    1.666666666666666666666e-01,
    6.666666666666666666666e-01,
    1.666666666666666666666e-01,
    1.666666666666666666666e-01,
    1.666666666666666666666e-01,
    1.666666666666666666666e-01,
    1.666666666666666666666e-01])

def triquad(d):
    """Return Dunavant quadrature rules."""
    n = d.size/3
    return np.vstack((d[0:n],d[n:2*n])).T, d[2*n:3*n]

def affine_map(mesh):
    """
    Build affine mappings F(x)=Ax+b for all triangles.
    In addition, calculates the determinants and
    the inverses of A's.
    """

    A = {0:{},1:{}}

    A[0][0] = mesh.p[mesh.t[:,1],0]-mesh.p[mesh.t[:,0],0]
    A[0][1] = mesh.p[mesh.t[:,2],0]-mesh.p[mesh.t[:,0],0]
    A[1][0] = mesh.p[mesh.t[:,1],1]-mesh.p[mesh.t[:,0],1]
    A[1][1] = mesh.p[mesh.t[:,2],1]-mesh.p[mesh.t[:,0],1]

    b = {}

    b[0] = mesh.p[mesh.t[:,0],0]
    b[1] = mesh.p[mesh.t[:,0],1]

    detA = A[0][0]*A[1][1]-A[0][1]*A[1][0]

    invA = {0:{},1:{}}

    invA[0][0] = A[1][1]/detA
    invA[0][1] = -A[0][1]/detA
    invA[1][0] = -A[1][0]/detA
    invA[1][1] = A[0][0]/detA

    return A,b,detA,invA

def affine_map3(mesh):
    """
    Build affine mappings F(x)=Ax+b for all triangles.
    In addition, calculates the determinants and
    the inverses of A's.
    """

    A = {0:{},1:{}}

    A[0][0] = mesh.p[0,mesh.t[1,:]]-mesh.p[0,mesh.t[0,:]]
    A[0][1] = mesh.p[0,mesh.t[2,:]]-mesh.p[0,mesh.t[0,:]]
    A[1][0] = mesh.p[1,mesh.t[1,:]]-mesh.p[1,mesh.t[0,:]]
    A[1][1] = mesh.p[1,mesh.t[2,:]]-mesh.p[1,mesh.t[0,:]]

    b = {}

    b[0] = mesh.p[0,mesh.t[:,0]]
    b[1] = mesh.p[0,mesh.t[:,0]]

    detA = A[0][0]*A[1][1]-A[0][1]*A[1][0]

    invA = {0:{},1:{}}

    invA[0][0] = A[1][1]/detA
    invA[0][1] = -A[0][1]/detA
    invA[1][0] = -A[1][0]/detA
    invA[1][1] = A[0][0]/detA

    return A,b,detA,invA

def lin_basis(qp):
    """
    Return the values and gradients of linear
    reference triangle basis functions at local
    quadrature points (qp).
    """
    phi = {}

    phi[0] = 1.-qp[:,0]-qp[:,1]
    phi[1] = qp[:,0]
    phi[2] = qp[:,1]

    gradphi = {}

    gradphi[0] = np.tile(np.array([-1.,-1.]),(qp.shape[0],1))
    gradphi[1] = np.tile(np.array([1.,0.]),(qp.shape[0],1))
    gradphi[2] = np.tile(np.array([0.,1.]),(qp.shape[0],1))

    return phi, gradphi

def lin_basis3(qp):
    """
    Return the values and gradients of linear
    reference triangle basis functions at local
    quadrature points (qp).
    """
    phi = {}

    phi[0] = 1.-qp[:,0]-qp[:,1]
    phi[1] = qp[:,0]
    phi[2] = qp[:,1]

    gradphi = {}

    gradphi[0] = np.tile(np.array([-1.,-1.]),(qp.shape[0],1))
    gradphi[1] = np.tile(np.array([1.,0.]),(qp.shape[0],1))
    gradphi[2] = np.tile(np.array([0.,1.]),(qp.shape[0],1))

    return phi, gradphi

def bilin_asm(bilin, mesh):
    """
    Assembly the bilinear form.
    """
    N = mesh.p.shape[0]
    T = mesh.t.shape[0]

    qp,qw = triquad(dunavant2)
    phi,gradphi = lin_basis(qp)
    A,b,detA,invA = affine_map(mesh)

    # Initialize sparse matrix structures for collecting K values
    data = np.array([])
    rows = np.array([])
    cols = np.array([])

    for i in [0,1,2]:
        u = np.tile(phi[i],(T,1))
        ux = np.outer(invA[0][0],gradphi[i][:,0])+np.outer(invA[1][0],gradphi[i][:,1])
        uy = np.outer(invA[0][1],gradphi[i][:,0])+np.outer(invA[1][1],gradphi[i][:,1])
        for j in [0,1,2]:
            v = np.tile(phi[j],(T,1))
            vx = np.outer(invA[0][0],gradphi[j][:,0])+np.outer(invA[1][0],gradphi[j][:,1])
            vy = np.outer(invA[0][1],gradphi[j][:,0])+np.outer(invA[1][1],gradphi[j][:,1])
            
            refKij = np.dot(bilin(u,v,ux,uy,vx,vy,0,0),qw)*np.abs(detA)
            # Save the values
            data = np.concatenate((data,refKij))
            rows = np.concatenate((rows,mesh.t[:,i]))
            cols = np.concatenate((cols,mesh.t[:,j]))

    return coo_matrix((data,(rows,cols)), shape=(N,N)).tocsr()

def bilin_asm2(bilin, mesh):
    """
    Assembly the bilinear form.
    """
    N = mesh.p.shape[0]
    T = mesh.t.shape[0]

    qp,qw = triquad(dunavant2)
    phi,gradphi = lin_basis(qp)
    A,b,detA,invA = affine_map(mesh)

    # Initialize sparse matrix structures for collecting K values
    data = np.zeros(9*T)
    rows = np.zeros(9*T)
    cols = np.zeros(9*T)

    for i in [0,1,2]:
        u = np.tile(phi[i],(T,1))
        ux = np.outer(invA[0][0],gradphi[i][:,0])+np.outer(invA[1][0],gradphi[i][:,1])
        uy = np.outer(invA[0][1],gradphi[i][:,0])+np.outer(invA[1][1],gradphi[i][:,1])
        for j in [0,1,2]:
            v = np.tile(phi[j],(T,1))
            vx = np.outer(invA[0][0],gradphi[j][:,0])+np.outer(invA[1][0],gradphi[j][:,1])
            vy = np.outer(invA[0][1],gradphi[j][:,0])+np.outer(invA[1][1],gradphi[j][:,1])
            
            ixs=slice(0+T*(3*i+j),T+T*(3*i+j))
            data[ixs]=np.dot(bilin(u,v,ux,uy,vx,vy,0,0),qw)*np.abs(detA)
            rows[ixs]=mesh.t[:,i]
            cols[ixs]=mesh.t[:,j]

    return coo_matrix((data,(rows,cols)), shape=(N,N)).tocsr()

def bilin_asm3(bilin, mesh):
    """
    Assembly the bilinear form.
    """
    N = mesh.p.shape[1]
    T = mesh.t.shape[1]

    qp,qw = triquad(dunavant2)
    phi,gradphi = lin_basis(qp)
    A,b,detA,invA = affine_map3(mesh)

    # Initialize sparse matrix structures for collecting K values
    data = np.zeros(9*T)
    rows = np.zeros(9*T)
    cols = np.zeros(9*T)

    for i in [0,1,2]:
        u = np.tile(phi[i],(T,1))
        ux = np.outer(invA[0][0],gradphi[i][:,0])+np.outer(invA[1][0],gradphi[i][:,1])
        uy = np.outer(invA[0][1],gradphi[i][:,0])+np.outer(invA[1][1],gradphi[i][:,1])
        for j in [0,1,2]:
            v = np.tile(phi[j],(T,1))
            vx = np.outer(invA[0][0],gradphi[j][:,0])+np.outer(invA[1][0],gradphi[j][:,1])
            vy = np.outer(invA[0][1],gradphi[j][:,0])+np.outer(invA[1][1],gradphi[j][:,1])
            
            ixs=slice(0+T*(3*i+j),T+T*(3*i+j))
            #print qp.shape
            #print invA[0][0].shape
            #print gradphi[i][0,:].shape
            #print ux.shape
            data[ixs]=np.dot(bilin(u,v,ux,uy,vx,vy,0,0),qw)*np.abs(detA)
            rows[ixs]=mesh.t[i,:]
            cols[ixs]=mesh.t[j,:]

    return coo_matrix((data,(rows,cols)), shape=(N,N)).tocsr()

def lin_asm(lin, mesh):
    """
    Assembly the linear form.
    """
    N = mesh.p.shape[0]
    T = mesh.t.shape[0]

    qp,qw = triquad(dunavant2)
    phi,gradphi = lin_basis(qp)
    A,b,detA,invA = affine_map(mesh)

    # Initialize sparse matrix structures
    # NOTE: This is constructed as matrix
    #       because substitution with indexing
    #       doesn't support duplicate indices.
    #       A better way to do this would be welcome.
    data = np.array([])
    rows = np.array([])
    cols = np.array([])

    for i in [0,1,2]:
        v = np.tile(phi[i],(T,1))
        vx = np.outer(invA[0][0],gradphi[i][:,0])+np.outer(invA[1][0],gradphi[i][:,1])
        vy = np.outer(invA[0][1],gradphi[i][:,0])+np.outer(invA[1][1],gradphi[i][:,1])

        refFij = np.dot(lin(v,vx,vy,0,0),qw)*np.abs(detA)
        data = np.concatenate((data,refFij))
        rows = np.concatenate((rows,np.zeros(T)))
        cols = np.concatenate((cols,mesh.t[:,i]))

    return coo_matrix((data,(rows,cols)), shape=(1,N)).toarray()[0]
