# -*- coding: utf-8 -*-
"""
Utility functions.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy

def const_cell(nparr,*arg):
    """
    Initialize a cell array (i.e. python dictionary)
    with the given parameter array/float by performing
    a deep copy.

    *Example*. Initializing a cell array with zeroes.

    .. code-block:: python

        >>> from fem.utils import const_cell
        >>> const_cell(0.0,3,2)
        {0: {0: 0.0, 1: 0.0}, 1: {0: 0.0, 1: 0.0}, 2: {0: 0.0, 1: 0.0}}
    """
    if len(arg)==1:
        u={i: deepcopy(nparr) for (i,_) in enumerate(range(arg[0]))}
    else:
        u={i: const_cell(nparr,*arg[1:]) for (i,_) in enumerate(range(arg[0]))}
    return u

def direct(A,b,x=None,I=None,use_umfpack=True):
    """Solve system Ax=b with Dirichlet boundary conditions."""

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

def cg(A,b,tol,maxiter,x0=None,I=None,pc="diag",verbose=True,viewiters=False):
    """Conjugate gradient solver wrapped for FEM purposes."""
    print "Starting conjugate gradient with preconditioner \""+pc+"\"..."
    
    def callback(x):
        if viewiters:
            print "- Vector-2 norm: "+str(np.linalg.norm(x))

    if pc=="diag":
        # diagonal preconditioner
        M=sp.spdiags(1/(A[I].T[I].diagonal()),0,I.shape[0],I.shape[0])
    
    if I is None:
        u=spl.cg(A,b,x0=x0,maxiter=maxiter,M=M,tol=tol,callback=callback)
    else:
        if x0 is None:
            u=spl.cg(A[I].T[I].T,b[I],maxiter=maxiter,M=M,tol=tol,callback=callback)
        else:
            u=spl.cg(A[I].T[I].T,b[I],x0=x0[I],maxiter=maxiter,M=M,tol=tol,callback=callback)

    if verbose:
        if u[1]==0:
            print "* Achieved tolerance "+str(tol)+"."
        elif u[1]>0:
            print "* WARNING! Maximum number of iterations "+str(maxiter)+" reached."

    if I is None: 
        return u[0]
    else:
        U=np.zeros(A.shape[0])
        U[I]=u[0]
        return U

class ConvergencePoint(object):
    pass    

class ConvergenceStudy(object):
    """
    A module to simplify creating convergence studies.
    Uses *.pkl (pickle) files as key-value-type storage
    and enables simple plotting and fitting of linear
    functions on logarithmic scale.
    """
    def __init__(self,fname):
        self.fname=fname+".plk"

    def add_point(self,x,y,tag='default'):
        # open datastore if exists
        try:
            with open(self.fname,'rb') as fh:
                datastore=pickle.load(fh)
        except IOError:
            datastore={}

        # save point to datastore
        datastore[(tag,x)]=y

        # save datastore to file
        with open(self.fname,'wb') as fh:
            pickle.dump(datastore,fh)

    def plot(self,xlabel='Mesh parameter',ylabel='Error',
             show_labels=False,loc='upper right',exclude_tags=None,draw_fit=True):
        try:
            with open(self.fname,'rb') as fh:
                datastore=pickle.load(fh)
        except IOError:
            raise Exception("ConvergenceStudy.plot(): File "+self.fname+" not found!")

        graphs_x={}
        graphs_y={}
        for key in datastore:
            pt=datastore[key]
            tag=key[0]
            if exclude_tags is not None and tag in exclude_tags:
                pass
            else:
                if tag in graphs_x:
                    graphs_x[tag]=np.append(graphs_x[tag],key[1])
                    graphs_y[tag]=np.append(graphs_y[tag],pt)
                else:
                    graphs_x[tag]=np.array([key[1]])
                    graphs_y[tag]=np.array([pt])

        fig,ax=plt.subplots()
        for tag in graphs_x:
            I=np.argsort(graphs_x[tag])
            ax.loglog(graphs_x[tag][I],graphs_y[tag][I],'o',
                      label=tag)
            if draw_fit:
                fitcoeffs=np.polyfit(np.log10(graphs_x[tag]),np.log10(graphs_y[tag]),1)
                def fitmap(x):
                    return 10.0**(fitcoeffs[0]*np.log10(x)+fitcoeffs[1])
                def default_fit_label(tag,rate):
                    ratestr='%.2f'%round(rate,2)
                    return "polynomial fit ("+tag+"), slope: "+ratestr
                pts=np.array([graphs_x[tag][I[0]],graphs_x[tag][I[-1]]])
                ax.loglog(pts,fitmap(pts),'-',label=default_fit_label(tag,fitcoeffs[0]))

        if show_labels:
            ax.legend(loc=loc)

        ax.grid(b=True,which='major',color='k',linestyle='-')
        ax.grid(b=True,which='minor',color='0.5',linestyle='--')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig,ax

    def show(self):
        plt.show()

def gradient(u,mesh):
    """Compute the gradient of a piecewise linear function."""

    x1=mesh.p[0,mesh.t[0,:]]
    x2=mesh.p[0,mesh.t[1,:]]
    x3=mesh.p[0,mesh.t[2,:]]

    y1=mesh.p[1,mesh.t[0,:]]
    y2=mesh.p[1,mesh.t[1,:]]
    y3=mesh.p[1,mesh.t[2,:]]

    z1=u[mesh.t[0,:]]
    z2=u[mesh.t[1,:]]
    z3=u[mesh.t[2,:]]

    dx=(-y2*z1+y3*z1+y1*z2-y3*z2-y1*z3+y2*z3)/(x2*y1-x3*y1-x1*y2+x3*y2+x1*y3-x2*y3)
    dy=(x2*z1-x3*z1-x1*z2+x3*z2+x1*z3-x2*z3)/(x2*y1-x3*y1-x1*y2+x3*y2+x1*y3-x2*y3)

    return dx,dy

