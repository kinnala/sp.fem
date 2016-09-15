# -*- coding: utf-8 -*-
"""
Utility functions.
"""
import numpy as np
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
