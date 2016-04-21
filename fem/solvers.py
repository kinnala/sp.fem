# -*- coding: utf-8 -*-
"""
Some linear solvers for the system Ax=b.

@author: Tom Gustafsson


This file is part of sp.fem.

sp.fem is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

sp.fem is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with sp.fem.  If not, see <http://www.gnu.org/licenses/>. 
"""

import scipy.sparse as sp
import scipy.sparse.linalg as spl
import numpy as np

def direct(A,b,x=None,I=None,use_umfpack=True):
    """Solve system Ax=b."""

    if I is None:
        x=spl.spsolve(A,b,use_umfpack=use_umfpack)
    else:
        if x is None:
            x=np.zeros(A.shape[0])
            x[I]=spl.spsolve(A[I].T[I].T,b[I],use_umfpack=use_umfpack)
        else:
            D=np.setdiff1d(np.arange(A.shape[0]),I)
            x[I]=spl.spsolve(A[I].T[I].T,b[I]-A[I].T[D].T.dot(x[D]),use_umfpack=use_umfpack)

    return x

def cg(A,b,tol,maxiter,pc="diag",verbose=True):
    print "Starting conjugate gradient with preconditioner \""+pc+"\"..."
    
    def callback(x):
        print "- Vector-2 norm: "+str(np.linalg.norm(x))

    if pc=="diag":
        # diagonal preconditioner
        M=sp.spdiags(1/(A.diagonal()),0,A.shape[0],A.shape[1])
    
    if verbose:
        u=spl.cg(A,b,maxiter=maxiter,M=M,tol=tol,callback=callback)
    else:
        u=spl.cg(A,b,maxiter=maxiter,M=M,tol=tol)

    if verbose:
        if u[1]==0:
            print "Achieved tolerance "+str(tol)+"."
        elif u[1]>0:
            print "Maximum number of iterations "+str(maxiter)+" reached."
        
    return u[0]
