import numpy as np

def get_quadrature(refdom,norder):
    """Return a nth order accurate quadrature rule for any (supported) reference domain."""
    if refdom is "tri":
        return get_quadrature_tri(norder) 
    elif refdom is "line":
        return get_quadrature_line(norder)
    else:
        raise NotImplementedError("get_quadrature: the given mesh type is not supported!")

def get_quadrature_tri(norder):
    """Return a nth order accurate quadrature rule for triangle (0,0) (0,1) (1,0)."""
    return {
            2:(np.array([[1.666666666666666666666e-01,6.666666666666666666666e-01,1.666666666666666666666e-01],[1.666666666666666666666e-01,1.666666666666666666666e-01,6.666666666666666666666e-01]]),np.array([1.666666666666666666666e-01,1.666666666666666666666e-01,1.666666666666666666666e-01]))
    }[norder]

def get_quadrature_line(norder):
    """Return a nth order accurate quadrature rule for line [0,1]."""
    return {
            2:(np.array([1.127016653792584e-1,5.0000000000000000e-1,8.872983346207417e-1]),np.array([2.777777777777779e-1,4.4444444444444444e-1,2.777777777777778e-1]))
    }[norder]
