# -*- coding: utf-8 -*-
"""
Performance tests. 

@author: Tom Gustafsson
"""

import timeit
import time

# patch timeit to give a return value
def _template_func(setup, func):
    """Create a timer function. Used if the "statement" is a callable."""
    def inner(_it, _timer, _func=func):
        setup()
        _t0 = _timer()
        for _i in _it:
            retval = _func()
        _t1 = _timer()
        return _t1 - _t0, retval
    return inner

timeit._template_func = _template_func

import fem.mesh as fmsh
import fem.asm as fasm
import fem.element as felem

import numpy as np
import matplotlib.pyplot as plt

def init(N):
    m=fmsh.MeshTri()
    m.refine(N)
    a=fasm.AssemblerElement(m,felem.ElementTriP1())
    def _run():
        a.iasm(lambda du,dv: du[0]*dv[0]+du[1]*dv[1])
        return a.dofnum_u.N
    return _run

def test():
    results=np.array([])
    for N in range(10):
        t=timeit.Timer(init(N))
        print t.timeit(3)

if __name__ == '__main__':
    test()

