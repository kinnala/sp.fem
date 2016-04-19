# -*- coding: utf-8 -*-
"""
Performance/memory tests. 

@author: Tom Gustafsson
"""

import timeit
import time
import platform

# patch timeit to give a return value
# (c) unutbu (from http://stackoverflow.com/questions/24812253/how-can-i-capture-return-value-with-python-timeit-module)
def _template_func(setup,func):
    """Create a timer function. Used if the "statement" is a callable."""
    def inner(_it,_timer,_func=func):
        setup()
        _t0=_timer()
        for _i in _it:
            retval=_func()
        _t1=_timer()
        return _t1-_t0,retval
    return inner

timeit._template_func=_template_func

import fem.mesh as fmsh
import fem.asm as fasm
import fem.element as felem

import numpy as np
import matplotlib.pyplot as plt

class PerformanceTest(object):
    """Class from which all other PerformanceTests are inherited."""
    def init(self,N):
        """Initialize the test and return a function handle to run the test."""
        raise NotImplementedError("PerformanceTest.init() not implemented!")
    def values(self):
        """Return a list of values that correspond to 'feasible' test cases."""
        raise NotImplementedError("PerformanceTest.values() not implemented!")

# ***************************  
# Write tests after this line
# ***************************

class PoissonTriP1InteriorAssemble(PerformanceTest):
    """Assemble standard Poisson stiffness matrix with P1 elements in 2D triangular mesh."""
    def init(self,N):
        m=fmsh.MeshTri()
        m.refine(N)
        a=fasm.AssemblerElement(m,felem.ElementTriP1())
        def _run():
            a.iasm(lambda du,dv: du[0]*dv[0]+du[1]*dv[1])
            return a.dofnum_u.N
        return _run
    def values(self):
        return [3,4,5,6,7,8,9]
        
class PoissonTriP1FacetAssemble(PerformanceTest):
    """Assemble Poisson facet mass matrix with P1 elements in 2D triangular mesh."""
    def init(self,N):
        m=fmsh.MeshTri()
        m.refine(N)
        a=fasm.AssemblerElement(m,felem.ElementTriP1())
        def _run():
            a.fasm(lambda u,v: u*v)
            return a.dofnum_u.N
        return _run
    def values(self):
        return [3,4,5,6,7,8,9]
        
# ****************************
# Write tests before this line
# ****************************

if __name__ == '__main__':
    verbose=False
    for t in vars()['PerformanceTest'].__subclasses__():
        tname=t.__name__
        test=t()
        Ns=np.array([])
        Times=np.array([])
        for N in test.values():
            timer=timeit.Timer(test.init(N))
            result=timer.timeit(3)
            if verbose:
                print str(result[1])+","+str(result[0]/3.0)
            Ns=np.append(Ns,result[1])
            Times=np.append(Times,result[0]/3.0)
        pfit=np.polyfit(np.log10(Ns),np.log10(Times),1)
        print tname+","+str(pfit[0])+","+str(pfit[1])
